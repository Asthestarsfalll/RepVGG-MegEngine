import megengine as mge
import megengine.module as M
import megengine.functional as F
from .se_block import SEBlock
import numpy as np


def ConvBnRelu(in_ch, out_ch, kernel_size, stride, padding, groups=1):  # conv bn relu
    result = M.Sequential(
        M.Conv2d(in_channels=in_ch, out_channels=out_ch,
                 kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        M.BatchNorm2d(num_features=out_ch),
        M.ReLU()
    )
    return result


def ConvBn(in_ch, out_ch, kernel_size, stride, padding, groups=1):  # conv bn relu
    result = M.Sequential(
        M.Conv2d(in_channels=in_ch, out_channels=out_ch,
                 kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        M.BatchNorm2d(num_features=out_ch)
    )
    return result


class RepVGGplusBlock(M.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, deploy=False, use_post_se=False):
        super(RepVGGplusBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_ch = in_ch

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = M.ReLU()

        if use_post_se:  # ?
            self.post_se = SEBlock(
                out_ch, internal_neurons=out_ch // 4)
        else:
            self.post_se = M.Identity()

        if deploy:
            self.reparam = M.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            if out_ch == in_ch and stride == 1:
                self.identity = M.BatchNorm2d(
                    num_features=out_ch)  # identity
            else:
                self.identity = None
            self.dense = ConvBn(in_ch=in_ch, out_ch=out_ch,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            padding_11 = padding - kernel_size // 2
            self.pointwise = ConvBn(in_ch=in_ch, out_ch=out_ch,
                                    kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x, L2):

        if self.deploy:
            return self.post_se(self.nonlinearity(self.reparam(x))), None

        if self.identity is None:
            id_out = 0
        else:
            id_out = self.identity(x)

        out = self.dense(x) + self.pointwise(x) + id_out
        out = self.post_se(self.nonlinearity(out))
        t3 = (self.dense[1].weight / (F.sqrt(self.dense[1].running_var +
                                             self.dense[1].eps))).reshape(-1, 1, 1, 1).detach()  # bn
        t1 = (self.pointwise[1].weight / (F.sqrt(self.pointwise[1].running_var +
                                                 self.pointwise[1].eps))).reshape(-1, 1, 1, 1).detach()
        K3 = self.dense[0].weight  # conv
        K1 = self.pointwise[0].weight

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()

        return out, L2 + l2_loss_circle + l2_loss_eq_kernel

    def _zero_padding(self, weight):
        if weight is None:
            return 0
        else:
            kernel = F.nn.pad(weight, [(0,0),(0,0),(1,1),(1,1)])
            return kernel

    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, M.Sequential):
            kernel, running_mean, running_var, gamma, beta, eps = branch[0].weight, branch[
                1].running_mean, branch[1].running_var, branch[1].weight, branch[1].bias, branch[1].eps
        else:
            assert isinstance(branch, M.BatchNorm2d)  # 只有BN层
            if not hasattr(self, 'bn_identity'):  # 对于BN层，初始化时创建一个identity
                input_dim = self.in_ch // self.groups
                kernel_value = np.zeros(
                    (self.in_ch, input_dim, 3, 3), dtype=np.float32)  # 填0
                for i in range(self.in_ch):
                    kernel_value[i, i % input_dim, 1, 1] = 1  # identity
                self.bn_identity = mge.tensor(
                    kernel_value).to(branch.weight.device)
            kernel, running_mean, running_var, gamma, beta, eps = self.bn_identity, branch.running_mean, branch.running_var, branch.weight, branch.bias, branch.eps
        std = F.sqrt(running_var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)  # 广播
        return kernel * t, beta - running_mean * gamma / std

    def convert_kernel_bias(self):  # 等价转换
        kernel3x3, bias3x3 = self._fuse_bn(self.dense)
        kernel1x1, bias1x1 = self._fuse_bn(self.pointwise)
        kernelid, biasid = self._fuse_bn(self.identity)
        return kernel3x3 + self._zero_padding(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.convert_kernel_bias()
        self.reparam = M.Conv2d(in_channels=self.dense[0].in_channels, out_channels=self.dense[0].out_channels,
                                kernel_size=self.dense[0].kernel_size, stride=self.dense[0].stride,
                                padding=self.dense[0].padding, dilation=self.dense[0].dilation, groups=self.dense[0].groups, bias=True)
        self.reparam.weight.data = kernel
        self.reparam.bias.data = bias
        # 删除
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
    def __init__(self, in_planes, planes, num_blocks, stride, use_checkpoint=False, use_post_se=False, deploy=False):
        super().__init__()
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        self.in_planes = in_planes
        for stride in strides:
            cur_groups = 1
            blocks.append(RepVGGplusBlock(in_ch=self.in_planes, out_ch=planes, kernel_size=3,
                                          stride=stride, padding=1, groups=cur_groups, deploy=deploy, use_post_se=use_post_se))
            self.in_planes = planes
        self.blocks = blocks
        self.use_checkpoint = use_checkpoint

    def forward(self, x, L2):
        for block in self.blocks:
            x, L2 = block(x, L2)
        return x, L2


class RepVGGplus(M.Module):

    def __init__(self, num_blocks, num_classes,
                 width_multiplier, override_groups_map=None,
                 deploy=False,
                 use_post_se=False):
        super().__init__()

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_post_se = use_post_se
        self.num_classes = num_classes

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGplusBlock(in_ch=3, out_ch=self.in_planes,
                                      kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_post_se=use_post_se)
        self.cur_layer_idx = 1
        self.stage1 = RepVGGplusStage(self.in_planes, int(
            64 * width_multiplier[0]), num_blocks[0], stride=2, use_post_se=use_post_se, deploy=deploy)
        self.stage2 = RepVGGplusStage(int(64 * width_multiplier[0]), int(
            128 * width_multiplier[1]), num_blocks[1], stride=2, use_post_se=use_post_se, deploy=deploy)
        #   split stage3 so that we can insert an auxiliary classifier
        self.stage3_first = RepVGGplusStage(int(128 * width_multiplier[1]), int(
            256 * width_multiplier[2]), num_blocks[2] // 2, stride=2, use_post_se=use_post_se, deploy=deploy)
        self.stage3_second = RepVGGplusStage(int(256 * width_multiplier[2]), int(
            256 * width_multiplier[2]), num_blocks[2] // 2, stride=1, use_post_se=use_post_se, deploy=deploy)
        self.stage4 = RepVGGplusStage(int(256 * width_multiplier[2]), int(
            512 * width_multiplier[3]), num_blocks[3], stride=2, use_post_se=use_post_se, deploy=deploy)
        self.gap = M.AdaptiveAvgPool2d((1, 1))
        self.linear = M.Linear(int(512 * width_multiplier[3]), num_classes)
        #   aux classifiers
        if not self.deploy:  # 貌似是实现在任意一层输出
            self.stage1_aux = self._build_aux_for_stage(self.stage1)
            self.stage2_aux = self._build_aux_for_stage(self.stage2)
            self.stage3_first_aux = self._build_aux_for_stage(
                self.stage3_first)

    def _build_aux_for_stage(self, stage):
        stage_out_channels = stage.blocks[-1].dense[0].out_channels
        downsample = ConvBnRelu(in_ch=stage_out_channels,
                                out_ch=stage_out_channels, kernel_size=3, stride=2, padding=1)
        # fc = M.Linear(stage_out_channels, self.num_classes, bias=True)
        fc = M.Conv2d(stage_out_channels, self.num_classes,
                      kernel_size=1, bias=True)
        return M.Sequential(downsample, M.AdaptiveAvgPool2d((1, 1)), fc)

    def forward(self, x):
        if self.deploy:
            out, _ = self.stage0(x, L2=None)
            out, _ = self.stage1(out, L2=None)
            out, _ = self.stage2(out, L2=None)
            out, _ = self.stage3_first(out, L2=None)
            out, _ = self.stage3_second(out, L2=None)
            out, _ = self.stage4(out, L2=None)
            y = self.gap(out)
            y = y.reshape(y.shape[0], -1)
            y = self.linear(y)
            return y

        else:
            out, L2 = self.stage0(x, L2=0.0)
            out, L2 = self.stage1(out, L2=L2)
            stage1_aux = self.stage1_aux(out)
            out, L2 = self.stage2(out, L2=L2)
            stage2_aux = self.stage2_aux(out)
            out, L2 = self.stage3_first(out, L2=L2)
            stage3_first_aux = self.stage3_first_aux(out)
            out, L2 = self.stage3_second(out, L2=L2)
            out, L2 = self.stage4(out, L2=L2)
            y = self.gap(out)
            y = y.reshape(y.shape[0], -1)
            y = self.linear(y)
            return {
                'main': y,
                'stage1_aux': stage1_aux,
                'stage2_aux': stage2_aux,
                'stage3_first_aux': stage3_first_aux,
                'L2': L2
            }

    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                M.switch_to_deploy()
            # if hasattr(m, 'use_checkpoint'):
            #     # Disable checkpoint. I am not sure whether using checkpoint slows down inference.
            #    M.use_checkpoint = False
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True


def create_RepVGGplus_L2pse(deploy=False, use_checkpoint=False):
    return RepVGGplus(num_blocks=[8, 14, 24, 1], num_classes=1000,
                      width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_post_se=True,
                      use_checkpoint=use_checkpoint)


repvggplus_dict = {
    'RepVGGplus-L2pse': create_RepVGGplus_L2pse,
}


def get_RepVGGplus_by_name(name):
    return repvggplus_dict[name]


if __name__ == '__main__':
    inputs = mge.tensor(np.random.random((5, 3, 224, 224)))
    block = RepVGGplusBlock(in_ch=3, out_ch=3, kernel_size=3, stride=1,
                            padding=1, dilation=1, groups=1, deploy=False)
    out = block(inputs, 0.0)
    print(out[0].shape)
    print(out[1])
    block.eval()
    print(block)
    block.switch_to_deploy()
    print(block)
    vgg = RepVGGplus(num_blocks=[2, 4, 14, 1], num_classes=19, width_multiplier=[
        0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
    print(vgg)
    vgg.switch_repvggplus_to_deploy()
    out = vgg(inputs)
    print(out.shape)
