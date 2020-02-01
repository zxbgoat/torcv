import torch.nn as nn


class KpLoss(nn.Module):

    def __init__(self, cfg):
        super(KPLoss, self).__init__()
        self.alpha = cfg.alpha
        self.beta = cfg.beta

    def forward(self, pd, gt, mask):
        width, height, ctgnum = pd.shape
        pos = (1-pd[gt==1.0])**self.alpha * torch.log(pd[gt==1.0])
        neg = (1-gt[gt!=1.0])**self.beta * pd[gt!=1.0]**self.alpha *\
            torch.log(1-pd[gt!=1.0]) 
        return (pos + neg) / sum(mask)


class OffLoss(nn.Module):

    def __init__(self, cfg):
        super(OffLoss, self).__init__()

    def forward(self, pd, gt, mask):
        diffs = torch.abs(pd[mask] - gt[mask])
        return torch.sum(diffs) / torch.sum(mask)


class SzLoss(nn.Module):

    def __init__(self, cfg):
       super(SzLoss, self).__init__()

    def forward(self, pd, gt, mask):
        diffs = torch.abs(pd[mask] - gt[mask])
        return torch.sum(diffs) / torch.sum(mask)


class DetLoss(nn.Module):
    
    def __init__(self, cfg):
        super(DetLoss, self).__init__()
        self.lmdsz = cfg.lmdsz
        self.lmdoff = cfg.lmdoff
        self.kploss = KpLoss(cfg)
        self.ofloss = OffLoss(cfg)
        self.szloss = SzLoss(cfg)

    def forward(self, rsts, anns):
        mask = anns['mask']
        kpl = self.kploss(rsts['heatmap'], anns['heatmap'], mask)
        ofl = self.ofloss(rsts['offset'], anns['offset'], mask)
        szl = self.szloss(rsts['size'], anns['size'], mask)
        return kpl + szl*self.lmdsz + ofl*self.lmdof
