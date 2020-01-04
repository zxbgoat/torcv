import torch
import itertools as it
import torch.nn as nn
from .utils import iou
from .utils import cvt


class Targets:
    """ choose positive and negative samples for training """

    def __init__(self, cfg):
        self.thres = cfg.cfdthres
        self.imgsz = cfg.imgsz
        self.ancs = torch.tensor(cfg.ancs, dtype=torch.float32)

    def __call__(self, pred, lbl):
        nB, nA, nG, _, nT = pred.shape
        nL, stride = lbl.shape[1], self.imgsz / nG
        ancs = self.ancs / stride
        pmask = torch.zeros(nB, nA, nG, nG)
        cmask = torch.ones(nB, nA, nG, nG)
        targ = torch.zeros_like(pred)
        for b, l in it.product(range(nB), range(nL)):
            gtcls = int(lbl[b, l, 0])
            gtbox = lbl[b, l, 1:] * nG
            gtshp = torch.Tensor([0, 0, gtbox[2], gtbox[3]]).unsqueeze(0)
            ancshps = torch.cat((torch.zeros(nA, 2), ancs), 1)
            ancious = iou(gtshp, ancshps)
            gi, gj = list(map(int, gtbox[:2]))  # Get grid box indices
            cmask[b, ancious > self.thres, gj, gi] = 0
            bestn = torch.argmax(ancious)
            pmask[b, bestn, gj, gi] = 1  # mark the predbox
            cmask[b, bestn, gj, gi] = 1
            targ[b, bestn, gj, gi, :4] = cvt(gtbox, (gi, gj), ancs[bestn],
                                             mode='target')
            targ[b, bestn, gj, gi, 4] = 1
            targ[b, bestn, gj, gi, gtcls+5] = 1
        return pmask, cmask, targ


class L1Smooth:

    def __init__(self, cfg):
        self.mse = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce = nn.BCELoss(size_average=True)  # Confidence loss
        self.cre = nn.CrossEntropyLoss()  # Class loss
        self.tgts = Targets(cfg)

    def __call__(self, pred, lbl):
        """
        Arguments:
            lbl: labels of the images, its shape is (batchsz, annotnum, 5)
        """
        pmsk, cmsk, targ = self.tgts(pred.data, lbl)
        pmsk = pmsk.type(torch.ByteTensor).to(pred.device)
        cmsk = cmsk.type(torch.ByteTensor).to(pred.device)
        targ = targ.to(device=pred.device)
        # Get conf mask where gt and where there is no gt
        ctru, cfal = pmsk, cmsk - pmsk
        nps, nng = torch.sum(ctru).item(), torch.sum(cfal).item()
        lx = self.mse(pred[..., 0][pmsk], targ[..., 0][pmsk])
        ly = self.mse(pred[..., 1][pmsk], targ[..., 1][pmsk])
        lw = self.mse(pred[..., 2][pmsk], targ[..., 2][pmsk])
        lh = self.mse(pred[..., 3][pmsk], targ[..., 3][pmsk])
        lreg = lx + ly + lw + lh  # regression loss
        lps = self.bce(pred[..., 4][ctru], targ[..., 4][ctru])  # pstv samples
        lng = self.bce(pred[..., 4][cfal], targ[..., 4][cfal])  # ngtv samples
        lcfd = lps + lng  # confidence loss
        lcls = self.cre(pred[..., 5:][pmsk],
                        torch.argmax(targ[..., 5:][pmsk], 1)) / lbl.shape[0]
        loss = lreg + lcfd + lcls
        rslt = {'los': loss.item(), 'reg': lreg.item(), 'cfd': lcfd.item(),
                'cls': lcls.item(), 'ps': lps.item(), 'ng': lng.item(),
                'x': lx.item(), 'y': ly.item(), 'w': lw.item(), 'h': lh.item(),
                'nps': nps, 'nng': nng}
        return loss, rslt
