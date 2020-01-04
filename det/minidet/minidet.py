import torch
import torch.nn as nn


def initwt(m):
    if isinstance(m, torch.nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Feat(nn.Module):

    def __init__(self):
        super(Feat, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(32),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.MaxPool2d(2, 2),
                                   nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Head(nn.Module):
    """ Detection layer """

    def __init__(self, ncls, nanc):
        super(Head, self).__init__()
        self.nanc = nanc
        self.natt = 5 + ncls
        self.conv = nn.Conv2d(256, nanc*self.natt, 1)

    def forward(self, ftr):
        ftr = self.conv(ftr)
        nA, nT = self.nanc, self.natt
        nB, nG = ftr.size(0), ftr.size(2)  # batch number and feat map size
        # Reshape the pred as (batch, anc, featH, featW, att)
        pred = ftr.view(nB, nA, nT, nG, nG).permute(0, 1, 3, 4, 2).contiguous()
        pred[..., :2] = torch.sigmoid(pred[..., :2])
        pred[..., 4] = torch.sigmoid(pred[..., 4])  # Conf
        pred[..., 5:] = torch.sigmoid(pred[..., 5:])  # Cls
        return pred


class MiniYolo(nn.Module):

    def __init__(self, cfg):
        super(MiniYolo, self).__init__()
        self.feat = Feat()
        self.head = Head(cfg.ncls, len(cfg.ancs))

    def forward(self, x):
        x = self.feat(x)
        x = self.head(x)
        return x


if __name__ == '__main__':
    from cfg import cfg

    miniyolo = MiniYolo(cfg)
    inp = torch.rand(4, 3, 160, 160)
    pred = miniyolo(inp)
    print(pred.shape)
