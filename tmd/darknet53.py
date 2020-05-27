import torch.nn as nn
import torch
import numpy as np


class Conv2d(nn.Module):
    def __init__(self, inc, outc, k, s, p):
        super(Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, k, s, p,bias=False),
            nn.BatchNorm2d(outc, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class ConvSet(nn.Module):
    def __init__(self, inc, outc):
        super(ConvSet, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(inc, outc, 1, 1, 0),
            Conv2d(outc, outc, 3, 1, 1),
            Conv2d(outc, outc * 2, 1, 1, 0),
            Conv2d(outc * 2, outc * 2, 3, 1, 1),
            Conv2d(outc * 2, outc, 1, 1, 0)

        )

    def forward(self, x):
        return self.forward(x)


class Residual(nn.Module):
    def __init__(self, inc):
        super(Residual, self).__init__()
        self.res = nn.Sequential(
            Conv2d(inc, inc // 2, 1, 1, 0),
            Conv2d(inc // 2, inc, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.res(x)


class DownSampling(nn.Module):
    def __init__(self, inc, outc):
        super(DownSampling, self).__init__()
        self.down = nn.Sequential(
            Conv2d(inc, outc, 3, 2, 1)
        )

    def forward(self, x):
        return self.down(x)


class DarkNet(nn.Module):

    def __init__(self, num_classes=100):
        super(DarkNet, self).__init__()

        self.d32 = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1),  # 256
            Conv2d(32, 64, 3, 2, 1),  # 128

            # 1x
            Conv2d(64, 32, 1, 1, 0),
            Conv2d(32, 64, 3, 1, 1),
            Residual(64),

            DownSampling(64, 128),  # 64

            # 2x
            Conv2d(128, 64, 1, 1, 0),
            Conv2d(64, 128, 3, 1, 1),
            Residual(128),

            Conv2d(128, 64, 1, 1, 0),
            Conv2d(64, 128, 3, 1, 1),
            Residual(128),

            DownSampling(128, 256),  # 32

            # 8x
            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

            Conv2d(256, 128, 1, 1, 0),
            Conv2d(128, 256, 3, 1, 1),
            Residual(256),

        )
        self.d16 = nn.Sequential(
            DownSampling(256, 512),  # 16

            # 8x
            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

            Conv2d(512, 256, 1, 1, 0),
            Conv2d(256, 512, 3, 1, 1),
            Residual(512),

        )

        self.d8 = nn.Sequential(
            DownSampling(512, 1024),  # 8

            # 4x
            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),

            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),

            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),

            Conv2d(1024, 512, 1, 1, 0),
            Conv2d(512, 1024, 3, 1, 1),
            Residual(1024),
        )
        self.avgpool = nn.AvgPool2d(8, 1, 0)
        # self.logits = nn.Conv2d(1024, 100, 1, 1, 0)
        self.logits = nn.Linear(1024, num_classes)

    def forward(self, x):
        x_32 = self.d32(x)
        x_16 = self.d16(x_32)
        x_8 = self.d8(x_16)
        x_8 = self.avgpool(x_8)
        x_8 = x_8.view(x_8.size(0), -1)
        x_8 = self.logits(x_8)
        return x_8


if __name__ == "__main__":
    darknet = DarkNet(num_classes=100)
    chkpt = torch.load("./weights/yolov3.pt")
    print(chkpt)
    # darknet.load_state_dict('../weights/yolov3.pt')
    darknet.eval()
    x = torch.Tensor(1, 1, 256, 256)
    y = darknet(x)
    print(y.shape)
