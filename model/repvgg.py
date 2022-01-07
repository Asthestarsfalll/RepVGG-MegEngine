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
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
"""


class RepVGGBlock(M.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
        use_se: bool = False
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy  # 是否是推理, inference or not
        self.groups = groups
        self.groups_channel = in_ch // groups  # in_ch = out_ch

        padding_11 = 0  # padding11用于1乘1卷积, used for pointwise convolution

        self.nonlinearity = M.ReLU()

        if use_se:
            self.se = SEBlock(out_ch, ratio=16)
        else:
            self.se = M.Identity()

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
            # self.identity = None
            # 3x3
            self.dense = M.ConvBn2d(in_ch, out_ch,
                                    kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
            # 1x1
            self.pointwise = M.ConvBn2d(in_ch, out_ch,
                                        kernel_size=1, stride=stride, padding=padding_11, groups=groups, bias=False)

    def forward(self, inputs):
        if self.deploy:  # 判断是不是推理, inference or not
            return self.nonlinearity(self.se(self.reparam(inputs)))

        if self.identity is None:
            identity = 0
        else:
            identity = self.identity(inputs)

        return self.nonlinearity(self.se(self.dense(inputs) + self.pointwise(inputs) + identity))

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
        self.reparam.weight[:] = kernel
        self.reparam.bias[:] = bias
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


class RepVGG(M.Module):

    def __init__(
        self,
        num_blocks: Sequence[int],
        width_multiplier=Sequence[Union[int, float]],
        num_classes: int = 1000,
        override_groups_map=None,
        deploy: bool = False,
        use_se: bool = False
    ):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(
            in_ch=3,
            out_ch=self.in_planes,
            stride=2,
            deploy=self.deploy,
            use_se=self.use_se
        )
        self.index = 1  # current layer index
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = M.AdaptiveAvgPool2d((1, 1))

        self.linear = M.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(
                self.index, 1)
            blocks.append(RepVGGBlock(in_ch=self.in_planes, out_ch=planes, stride=stride,
                                      groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes  # 更新
            self.index += 1
        return M.Sequential(*blocks)

    def _switch_to_deploy_and_save(self, save_path=None, save_name='RepVGG_deploy'):
        for name, module in self.named_modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        print(self)
        if save_path is not None:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, save_name + '.pkl')
            mge.save(self.state_dict, save_path)
            print(f'save state_dict to {save_path}')
        self.deploy = True

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = F.flatten(out, 1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def RepVGGA0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def RepVGGA1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def RepVGGA2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def RepVGGB0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def RepVGGB1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def RepVGGB1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def RepVGGB1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def RepVGGB2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def RepVGGB2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def RepVGGB2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def RepVGGB3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)


def RepVGGB3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def RepVGGB3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


def RepVGGD2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)


if __name__ == '__main__':
    pass
