import torch
import torch.nn as nn


class SSD(nn.Module):

    def __init__(self, ctgnum):
        super(SSD, self).__init__()
        self.ctgnum = self.ctgnum
        self.ancs = Anchors()
        self.feat = VGG()
        self.head = Head()

    def forward(self, img):
        srcs, locs, cfds = [], [], []
