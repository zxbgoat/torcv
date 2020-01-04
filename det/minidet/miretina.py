import torch
import torch.nn as nn
from ..utils import Anchors, FeaturePyramid
from ..utils import BBoxTransform, ClipBoxes
from ..losses import FocalLoss


def initwt(m):
    if isinstance(m, torch.nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Feat(nn.Module):

    def __init__(self):
        super(Feat, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.LeakyReLU(0.1))
        self.conv1 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return [x3, x4, x5]


class Clsf(nn.Module):
    """ classification subnet """

    def __init__(self, ncls, nanc):
        super(Clsf, self).__init__()
        self.ncls = ncls
        self.nanc = nanc
        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1))
        self.convc = nn.Sequential(nn.Conv2d(512, nanc*ncls, 3, 1, 1),
                                   nn.Sigmoid())

    def forward(self, ftr):
        out = self.conv1(ftr)
        out = self.conv2(out)
        out = self.convc(out)
        out = out.permute(0, 2, 3, 1)
        batchsz, width, height, channels = out.shape
        out = out.view(batchsz, width, height, self.nanc, self.ncls)
        return out.contiguous().view(ftr.shape[0], -1, self.ncls)


class Rgrs(nn.Module):
    """ regression subnet """

    def __init__(self, nanc):
        super(Rgrs, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU(0.1))
        self.convr = nn.Conv2d(512, nanc*4, 3, 1, 1)

    def forward(self, ftr):
        out = self.conv1(ftr)
        out = self.conv2(out)
        out = self.convr(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)


class MiniRetina(nn.Module):

    def __init__(self, cfg):
        super(MiniRetina, self).__init__()
        self.feat = Feat()
        self.fpns = FeaturePyramid()
        self.clsf = Clsf(cfg.ncls, len(cfg.ancs))
        self.rgrs = Rgrs(len(cfg.ancs))
        self.ancs = Anchors()
        self.regrBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.loss = FocalLoss()
        self.training = False

    def forward(self, imgs, anns):
        ftr = self.feat(imgs)
        ftrs = self.fpns(ftr)
        regs = torch.cat([self.rgrs(ftr) for ftr in ftrs], dim=1)
        ctgs = torch.cat([self.clsf(ftr) for ftr in ftrs], dim=1)
        ancs = self.ancs(x)
        if self.training:
            return self.loss(ctgs, regs, ancs, anns)
        transformed_anchors = self.regrBoxes(anchors, regress)
        transformed_anchors = self.clipBoxes(transformed_anchors, imgbatch)
        scores = torch.max(classif, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores>0.05)[0, :, 0]
        if scores_over_thresh.sum() == 0:
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
        classif= classif[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)
        nms_scores, nms_class = classif[0, anchors_nms_idx, :].max(dim=1)


if __name__ == '__main__':
    from cfg import cfg

    miretina = MiniRetina(cfg)
    inp = torch.rand(4, 3, 288, 288)
    ctg, reg = miretina(inp)
    print(ctg.shape, reg.shape)
