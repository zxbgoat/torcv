import torch
from .head import DetHead
from .upresnet import UpResNet


class CenterNet(nn.Module):

    def __init__(self, cfg):
        super(CenterNet, self).__init__()
        self.bkbn = UpResNet(cfg)
        self.head = DetHead(cfg)

    def forward(self, img):
        ftr = self.bkbn(img)
        rst = self.head(ftr)
        return rst
