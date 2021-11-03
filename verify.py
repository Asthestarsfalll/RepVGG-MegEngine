import megengine as mge
import megengine.module as m
import numpy as np
from model import RepVGGB0,se_block,RepVGGBlock



if __name__ == '__main__':
    inputs = mge.tensor(np.random.random((2, 3, 224, 224)))
    print('___________RepVGGBlock____________')
    block = RepVGGBlock(in_ch=3, out_ch=3, kernel_size=3, stride=1,
    padding=1, dilation=1, groups=1, deploy=False, use_se=False)
    print(block(inputs).shape)
    block.eval()
    print(block)
    print('___________RepVGGBlock switch to deploy____________')
    block.switch_to_deploy()
    print(block)
    print('___________RepVGGB0____________')
    vgg = RepVGGB0(False)
    print(vgg)
    print('___________RepVGGB0 switch to deploy____________')
    vgg._switch_to_deploy_and_save('./ckpt','test')
    out = vgg(inputs)
    print(out.shape)
