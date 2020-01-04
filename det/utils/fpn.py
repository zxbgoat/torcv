import torch.nn as nn


class FeaturePyramid(nn.Module):

    def __init__(self, fpnszs, ftrsz=256):
        super(Pyrm, self).__init__()
        C3sz, C4sz, C5sz = fpnszs
        self.P5_1 = nn.Conv2d(C5sz, ftrsz, kernel_size=1, stride=1, padding=0)
        self.P5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(ftrsz, ftrsz, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(C4sz, ftrsz, kernel_size=1, stride=1, padding=0)
        self.P4_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(ftrsz, ftrsz, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(C3sz, ftrsz, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(ftrsz, ftrsz, kernel_size=3, stride=1, padding=1)
        self.P6 = nn.Conv2d(C5sz, ftrsz, kernel_size=3, stride=2, padding=1)
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(ftrsz, ftrsz, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        # upsample C5 to get P5
        p5x = self.P5_1(C5)
        p5xup = self.P5_up(p5x)
        p5x = self.P5_2(p5x)
        # add P5 elementwise to C4
        p4x = self.P4_1(C4)
        p4x = p5xup + p4x
        p4xup = self.P4_up(p4x)
        p4x = self.P4_2(p4x)
        # add P4 elementwise to C3
        p3x = self.P3_1(C3)
        p3x = p3x + p4xup
        p3x = self.P3_2(p3x)
        # P6 is obtained via a 3x3 stride-2 conv on C5
        p6x = self.P6(C5)
        # P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6
        p7x = self.P7_1(p6x)
        p7x = self.P7_2(p7x)
        return [p3x, p4x, p5x, p6x, p7x]
