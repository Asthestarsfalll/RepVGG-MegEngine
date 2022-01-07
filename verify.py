import megengine as mge
import megengine.functional as F
import numpy as np

import model as repvgg


class Classifier(mge.module.Module):
    def __init__(self, planes):
        super(Classifier, self).__init__()
        self.downsample = mge.module.Conv2d(
            in_channels=planes, 
            out_channels=planes, 
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.gap = mge.module.AdaptiveAvgPool2d((1, 1))
        self.fc = mge.module.Linear(planes, 1000)
    
    def forward(self, inputs):
        out = self.downsample(inputs)
        out = self.gap(out)
        out = F.flatten(out, 1)
        out = self.fc(out)
        return out


def calDiff(out1, out2):
    print('___________test diff____________')
    print(out1.shape)
    print(out2.shape)
    print(F.argmax(out1, axis=1))
    print(F.argmax(out2, axis=1))
    print(((out1 - out2)**2).sum())


def verifyBlock():
    print('___________RepVGGBlock____________')
    inputs = mge.tensor(np.random.random((8, 16, 224, 224)))
    block = repvgg.RepVGGBlock(in_ch=16, out_ch=16, stride=1,
            groups=2, deploy=False, use_se=True)

    downsampe = Classifier(16)
    downsampe.eval()
    block.eval()
    out1 = downsampe(block(inputs))
    print(block)

    print('___________RepVGGBlock switch to deploy____________')
    block.switch_to_deploy()
    block.eval()
    out2 = downsampe(block(inputs))
    print(block)
    calDiff(out1, out2)


def verifyRepVGG(model_name, state_dict=None):
    print(f'___________{model_name}____________')
    inputs = mge.tensor(np.random.random((2, 3, 224, 224)))

    model = repvgg.__dict__[model_name](False)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    out1 = model(inputs)

    print(f'___________{model_name} switch to deploy____________')
    model._switch_to_deploy_and_save('./ckpt', 'test')
    model.eval()
    out2 = model(inputs)

    calDiff(out1, out2)



if __name__ == '__main__':
    verifyBlock()
    verifyRepVGG('RepVGGA0')


