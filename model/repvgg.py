import megengine.module as m
import megengine.functional as F
import numpy as np
import megengine
import copy
from se_block import SEBlock


def ConvBn(in_ch, out_ch, stride, padding, kernel_size=3, groups=1):
    result = m.Sequential(
        m.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups),
        m.BatchNorm2d(num_features=out_ch)
    )
    return result


class RepVGGBlock(m.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """
            删去padding_mode，megengine仅支持填充0值
        """
        self.deploy = deploy  # 是否是推理
        self.groups = groups
        self.in_channels = in_ch

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2  # padding11用于1乘1卷积

        self.nonlinearity = m.ReLU()

        if use_se:
            self.se = SEBlock(
                out_ch, internal_neurons=out_ch // 16)
        else:
            self.se = m.Identity()

        if deploy:
            self.reparam = m.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            self.identity = m.BatchNorm2d(
                num_features=in_ch) if out_ch == in_ch and stride == 1 else None
            # 3x3
            self.dense = ConvBn(in_ch, out_ch,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            # 1x1
            self.pointwise = ConvBn(in_ch, out_ch,
                                    kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.identity)

    def forward(self, inputs):
        if self.deploy:  # 判断是不是推理
            return self.nonlinearity(self.se(self.reparam(inputs)))

        if self.identity is None:
            identity = 0
        else:
            identity = self.identity(inputs)

        return self.nonlinearity(self.se(self.dense(inputs)+self.pointwise(inputs)+identity))

    def get_custom_L2(self):
        K3 = self.dense[0].weight  # kernel 3x3
        K1 = self.pointwise[0].weight  # kernel 1x1
        t3 = (self.dense[1].weight / ((self.rbr_dense[1].running_var +
                                       self.rbr_dense[1].eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.pointwise[1].weight / ((self.pointwise[1].running_var +
                                           self.pointwise[1].eps).sqrt())).reshape(-1, 1, 1, 1).detach()

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
            out_ch, in_ch = weight.shape[0:2]
            kernel = megengine.tensor(np.zeros((out_ch, in_ch, 3, 3)))
            # print(kernel.device)
            # print(kernel.dtype)
            kernel[:, :, 1, 1] = weight[:, :, 0, 0]
            return kernel

    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, m.Sequential):
            kernel, running_mean, running_var, gamma, beta, eps = branch[0].weight, branch[
                1].running_mean, branch[1].running_var, branch[1].weight, branch[1].bias, branch[1].eps
        else:
            assert isinstance(branch, m.BatchNorm2d)  # 只有BN层
            if not hasattr(self, 'bn_identity'):  # 对于BN层，初始化时创建一个bn_identity
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32)  # 填0
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1  # identity
                self.bn_identity = megengine.tensor(
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
        self.reparam = m.Conv2d(in_channels=self.dense[0].in_channels, out_channels=self.dense[0].out_channels,
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


class RepVGG(m.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(RepVGG, self).__init__()
        """
        override_groups_map: 
        width_multiplier: 乘法器控制各层宽度
        """
        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_ch=3, out_ch=self.in_planes,
                                  kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.index = 1  # current layer index
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = m.AdaptiveAvgPool2d((1, 1))  # ?

        self.linear = m.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(
                self.index, 1)  # 随机选取分组数？
            blocks.append(RepVGGBlock(in_ch=self.in_planes, out_ch=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes  # 更新
            self.index += 1
        return m.Sequential(*blocks)

    def _switch_to_deploy_and_save(self, save_path=None):
        for module in self.modules():
            print(module)
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        print(self)
        if save_path is not None:
            megengine.save(self.state_dict, save_path)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.reshape(out.shape[0], -1)
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


if __name__ == "__main__":
    # inputs = megengine.tensor(np.random.random((2, 3, 256, 256)))
    # block = RepVGGBlock(in_ch=3, out_ch=3, kernel_size=3, stride=1,
    # padding=1, dilation=1, groups=1, deploy=False, use_se=False)
    # out = block(inputs)
    # print(out.shape)
    # block.eval()
    # print(block)
    # block.switch_to_deploy()
    # print(block)
    vgg = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=19, width_multiplier=[
                 0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
    print(vgg)
    vgg._switch_to_deploy_and_save(None)
    # out = vgg(inputs)
    # print(out.shape)
