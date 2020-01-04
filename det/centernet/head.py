import torch.nn as nn


class DetHead(nn.Module):

    def __init__(self, cfg):
        self.ctgnum = cfg.ctgnum
        self.ftrnum = cfg.ftrnum
        self.conv1 = nn.Conv2d(self.ftrnum, self.ftrnum, kernel_size=3,
                               padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.ftrnum, self.ctgnum+4, kernel_size=1)

    def forward(self, ftr):
        rst = self.conv1(ftr)
        rst = self.relu(ftr)
        rst = self.conv2(ftr)
        return rst


    if __name__ == '__main__':
        import torch
        from easydict import EasyDict as edict

        cfg = edict()
        cfg.ctgnum = 80
        cfg.ftrnum = 256
        ftr = torch.rand(4, 256, 64, 64)
        head = DetHead(cfg)
        rst = head(ftr)
        print(rst.shape)
