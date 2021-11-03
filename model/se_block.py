import megengine as meg
import megengine.module as M
import megengine.functional as F
import numpy as np


class SEBlock(M.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = M.Conv2d(in_channels=input_channels,
                             out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = M.Conv2d(in_channels=internal_neurons,
                           out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.nn.adaptive_avg_pool2d(inputs, 1)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = F.sigmoid(x)
        x = x.reshape(-1, self.input_channels, 1, 1)
        return inputs * x


if __name__ == "__main__":
    se = SEBlock(64, 64//16)
    a = mge.tensor(np.random.random((2, 64, 9, 9)))
    a = se(a)
    print(a.shape)
