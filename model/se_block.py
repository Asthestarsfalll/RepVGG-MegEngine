import megengine as mge
import megengine.module as M
import megengine.functional as F
import numpy as np


class SEBlock(M.Module):

    def __init__(self, input_channels, ratio: int = 16):
        super(SEBlock, self).__init__()
        internal_neurons = input_channels // ratio
        assert internal_neurons > 0
        self.gap = M.AdaptiveAvgPool2d((1, 1))
        self.down = M.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            bias=True
        )
        self.relu = M.ReLU()
        self.up = M.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            bias=True
        )
        self.sigmoid = M.Sigmoid()

    def forward(self, inputs):
        x = self.sigmoid(self.up(self.relu(self.down(self.gap(inputs)))))
        return inputs * x


if __name__ == "__main__":
    se = SEBlock(64, 16)
    a = mge.tensor(np.random.random((2, 64, 9, 9)))
    a = se(a)
    print(a.shape)
