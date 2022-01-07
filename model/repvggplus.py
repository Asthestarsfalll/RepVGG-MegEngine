import copy
import os
from typing import Sequence, Union

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np

from .se_block import SEBlock

"""
References:
    https://github.com/DingXiaoH/RepVGG/blob/main/repvggplus.py
"""


class RepVGGplusBlock(M.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
        use_post_se: bool = False
    ):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy  # 是否是推理, inference or not
        self.groups = groups
        self.in_channels = in_ch

        padding_11 = 0  # padding11用于1乘1卷积, used for pointwise convolution

        self.nonlinearity = M.ReLU()

        if use_post_se:
            self.post_se = SEBlock(out_ch, ratio=4)
        else:
            self.post_se = M.Identity()

        if deploy:
            self.reparam = M.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            self.identity = M.BatchNorm2d(
                num_features=in_ch) if out_ch == in_ch and stride == 1 else None
            # 3x3
            self.dense = M.ConvBn2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            )
            # 1x1
            self.pointwise = M.ConvBn2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
            )

    def forward(self, inputs, L2):
        if self.deploy:  # 判断是不是推理, inference or not
            return self.post_se(self.nonlinearity(self.reparam(inputs))), None

        if self.identity is None:
            identity = 0
        else:
            identity = self.identity(inputs)
        out = self.post_se(self.nonlinearity(self.dense(inputs) + self.pointwise(inputs) + identity))
        l2_loss = self.get_custom_L2()

        return out, L2 + l2_loss

    def get_custom_L2(self):
        t3 = (self.dense.bn.weight / (F.sqrt(self.dense.bn.running_var +
                                             self.dense.bn.eps))).reshape(-1, 1, 1, 1).detach()  # bn
        t1 = (self.pointwise.bn.weight / (F.sqrt(self.pointwise.bn.running_var +
                                                 self.pointwise.bn.eps))).reshape(-1, 1, 1, 1).detach()
        K3 = self.dense.conv.weight
        K1 = self.pointwise.conv.weight

        # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        # The equivalent resultant central point of 3x3 kernel.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        # Normalize for an L2 coefficient comparable to regular L2.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def _zero_padding(self, weight):
        if weight is None:
            return 0
        else:
            # windows 1.6版本会报错，可使用以下代码
            # kernel = F.zeros((*weight.shape[:-2], 3, 3), device=weight.device)
            # kernel[..., 1:2, 1:2] = weight
            kernel = F.nn.pad(
                weight, [*[(0, 0) for i in range(weight.ndim - 2)], (1, 1), (1, 1)])
            return kernel

    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, M.ConvBn2d):
            kernel = branch.conv.weight
            branch = branch.bn
        else:
            assert isinstance(branch, M.BatchNorm2d)  # 只有BN层 ,"self.identity"
            if not hasattr(self, 'bn_identity'):  # 对于BN层，初始化时创建一个bn_identity
                # group convlution kernel shape:
                # [groups, out_channels // groups, in_channels // groups, kernel_size, kernel_size]
                kernel_value = np.zeros(
                    (self.groups_channel * self.groups, self.groups_channel, 3, 3), dtype=np.float32)
                for i in range(self.groups_channel * self.groups):
                    kernel_value[i, i % self.groups_channel, 1, 1] = 1
                if self.groups > 1:
                    kernel_value = kernel_value.reshape(
                        self.groups, self.groups_channel, self.groups_channel, 3, 3)
                self.bn_identity = mge.Parameter(kernel_value)
            kernel = self.bn_identity
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = F.sqrt(running_var + eps)
        t = (gamma / std).reshape(*kernel.shape[:-3], 1, 1, 1)  # 广播, broadcast
        return kernel * t, beta - running_mean * gamma / std

    def _convert_equivalent_kernel_bias(self):  # 等价转换
        kernel3x3, bias3x3 = self._fuse_bn(self.dense)
        kernel1x1, bias1x1 = self._fuse_bn(self.pointwise)
        kernelid, biasid = self._fuse_bn(self.identity)
        return kernel3x3 + self._zero_padding(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self._convert_equivalent_kernel_bias()
        self.reparam = M.Conv2d(
            in_channels=self.dense.conv.in_channels,
            out_channels=self.dense.conv.out_channels,
            kernel_size=3,
            stride=self.dense.conv.stride,
            padding=1,
            groups=self.dense.conv.groups,
            bias=True
        )
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        # 删除, delete
        for para in self.parameters():
            para.detach()
        self.__delattr__('dense')
        self.__delattr__('pointwise')
        if hasattr(self, 'identity'):
            self.__delattr__('identity')
        if hasattr(self, 'bn_identity'):
            self.__delattr__('bn_identity')
        self.deploy = True


class RepVGGplusStage(M.Module):

    def __init__(
        self,
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        use_post_se: bool = False,
        deploy: bool = False
    ):
        super(RepVGGplusStage, self).__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for stride in strides:
            cur_groups = 1
            block = RepVGGplusBlock(
                in_ch=self.in_planes,
                out_ch=planes,
                stride=stride,
                groups=cur_groups,
                deploy=deploy,
                use_post_se=use_post_se,
            )
            blocks.append(block)
            self.in_planes = planes
        self.blocks = M.Sequential(*blocks)

    def forward(self, inputs, L2):
        for block in self.blocks:
            inputs, L2 = block(inputs, L2)
        return inputs, L2


class AuxSideOutput(M.Module):
    def __init__(
            self,
            planes: int,
            nums_classes: int = 1000,
    ):
        super(AuxSideOutput, self).__init__()
        self.downsample = M.ConvBnRelu2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.gap = M.AdaptiveAvgPool2d((1, 1))
        self.fc = M.Linear(planes, nums_classes)

    def forward(self, inputs):
        out = self.downsample(inputs)
        out = self.gap(out)
        out = F.flatten(out, 1)
        out = self.fc(out)
        return out


class RepVGGplus(M.Module):
    base_channels = [64, 128, 256, 512]

    def __init__(
        self,
        num_blocks: Sequence[int],
        width_multiplier=Sequence[Union[int, float]],
        num_classes: int = 1000,
        override_groups_map=None,
        deploy: bool = False,
        use_post_se: bool = False
    ):
        super(RepVGGplus, self).__init__()

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_post_se = use_post_se

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGplusBlock(
            in_ch=3,
            out_ch=self.in_planes,
            stride=2,
            deploy=self.deploy,
            use_post_se=self.use_post_se
        )
        self.cur_layer_idx = 1  # current layer index
        channels = [int(bc * w) for bc, w in zip(self.base_channels, width_multiplier)]
        self.stage1 = RepVGGplusStage(
            self.in_planes,
            channels[0],
            num_blocks[0],
            stride=2,
            use_post_se=use_post_se,
            deploy=deploy
        )
        self.stage2 = RepVGGplusStage(
            channels[0],
            channels[1],
            num_blocks[1],
            stride=2,
            use_post_se=use_post_se,
            deploy=deploy
        )
        #   split stage3 so that we can insert an auxiliary classifier
        self.stage3_first = RepVGGplusStage(
            channels[1],
            channels[2],
            num_blocks[2] // 2,
            stride=2,
            use_post_se=use_post_se,
            deploy=deploy
        )
        self.stage3_second = RepVGGplusStage(
            channels[2],
            channels[2],
            num_blocks[2] // 2,
            stride=1,
            use_post_se=use_post_se,
            deploy=deploy
        )
        self.stage4 = RepVGGplusStage(
            channels[2],
            channels[3],
            num_blocks[3] // 2,
            stride=2,
            use_post_se=use_post_se,
            deploy=deploy
        )
        #   aux classifiers
        if not self.deploy:
            self.stage1_aux = AuxSideOutput(channels[0], num_classes)
            self.stage2_aux = AuxSideOutput(channels[1], num_classes)
            self.stage3_first_aux = AuxSideOutput(channels[2], num_classes)

        self.gap = M.AdaptiveAvgPool2d((1, 1))

        self.linear = M.Linear(int(512 * width_multiplier[3]), num_classes)

    def _switch_to_deploy_and_save(self, save_path=None, save_name='RepVGG_deploy'):
        for name, module in self.named_modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
            if hasattr(module, 'deploy'):
                module.deploy = False
        print(self)
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, save_name + '.pkl')
            mge.save(self.state_dict, save_path)
            print(f'save state_dict to {save_path}')
        self.deploy = True

    def forward(self, inputs):
        if self.deploy:
            out, _ = self.stage0(inputs, L2=None)
            out, _ = self.stage1(out, L2=None)
            out, _ = self.stage2(out, L2=None)
            out, _ = self.stage3_first(out, L2=None)
            out, _ = self.stage3_second(out, L2=None)
            out, _ = self.stage4(out, L2=None)
            out = self.gap(out)
            out = F.flatten(out, 1)
            out = self.linear(out)
            return out
        else:
            out, L2 = self.stage0(inputs, L2=0.0)
            out, L2 = self.stage1(out, L2=L2)
            stage1_aux = self.stage1_aux(out)
            out, L2 = self.stage2(out, L2=L2)
            stage2_aux = self.stage2_aux(out)
            out, L2 = self.stage3_first(out, L2=L2)
            stage3_first_aux = self.stage3_first_aux(out)
            out, L2 = self.stage3_second(out, L2=L2)
            out, L2 = self.stage4(out, L2=L2)
            out = self.gap(out)
            out = F.flatten(out, 1)
            out = self.linear(out)
            return {
                'main': out,
                'stage1_aux': stage1_aux,
                'stage2_aux': stage2_aux,
                'stage3_first_aux': stage3_first_aux,
                'L2': L2
            }


def RepVGGplus_L2pse(deploy=False):
    return RepVGGplus(num_blocks=[8, 14, 24, 1], num_classes=1000,
                      width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_post_se=True)


if __name__ == '__main__':
    model = RepVGGplus_L2pse(False)
    inputs = mge.tensor(np.random.random((2, 3, 224, 224)))
    out = model(inputs)
    # print(out['main'].shape)
    # print(out['stage1_aux'].shape)
    # print(out['stage2_aux'].shape)
    # print(out['stage3_first_aux'].shape)
    # print(out['L2'])
    model._switch_to_deploy_and_save()
    out = model(inputs)
    print(out.shape)
