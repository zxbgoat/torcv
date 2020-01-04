import numpy as np
import torch
import torch.nn as nn
from .utils import iou


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, ctgs, regs, ancs, anns):
        batchsz = ctgs.shape[0]
        ctglosses, reglosses = [], []
        anc = ancs[0, :, :]
        ancw = anc[:, 2] - anc[:, 0]
        anch = anc[:, 3] - anc[:, 1]
        ancx = anc[:, 0] + 0.5 * ancw
        ancy = anc[:, 1] + 0.5 * anch
        for j in range(batchsz):
            ctg = ctgs[j, :, :]
            reg = regs[j, :, :]
            ann = anns[j, :, :]
            ann = ann[ann[:, 4] != -1]
            if ann.shape[0] == 0:
                reglosses.append(torch.tensor(0).float().cuda())
                ctglosses.append(torch.tensor(0).float().cuda())
                continue
            ctg = torch.clamp(ctg, 1e-4, 1.0 - 1e-4)
            ious = iou(ancs[0, :, :], ann[:, :4]) # ancnum x annnum
            maxiou, argmaxiou = torch.max(ious, dim=1) # ancnum x 1
            # compute the loss for classification
            tgts = torch.ones(ctg.shape) * -1
            tgts = tgts.cuda()
            tgts[torch.lt(maxiou, 0.4), :] = 0
            posinds = torch.ge(maxiou, 0.5)
            posancnum = posinds.sum()
            asianns = ann[argmaxiou, :]
            tgts[posinds, :] = 0
            tgts[posinds, asianns[posinds, 4].long()] = 1
            alpha = torch.ones(tgts.shape).cuda() * self.alpha
            alpha = torch.where(torch.eq(tgts, 1.), alpha, 1. - alpha)
            focal = torch.where(torch.eq(tgts, 1.), 1. - ctg, ctg)
            focal = alpha * torch.pow(focal, self.gamma)
            bce = -(tgts * torch.log(ctg) + (1.0 - tgts) * torch.log(1.0 - ctg))
            # ctgloss = focal_weight * torch.pow(bce, gamma)
            ctgloss = focal * bce
            ctgloss = torch.where(torch.ne(tgts, -1.0), ctgloss, torch.zeros(ctgloss.shape).cuda())
            ctglosses.append(ctgloss.sum()/torch.clamp(posancnum.float(), min=1.0))
            # compute the loss for regression
            if posinds.sum() > 0:
                asianns = asianns[posinds, :]
                posancw = ancw[posinds]
                posanch = anch[posinds]
                posancx = ancx[posinds]
                posancy = ancy[posinds]
                gtw  = asianns[:, 2] - asianns[:, 0]
                gth = asianns[:, 3] - asianns[:, 1]
                gtx   = asianns[:, 0] + 0.5 * gtw
                gty   = asianns[:, 1] + 0.5 * gth
                # clip widths to 1
                gtw  = torch.clamp(gtw, min=1)
                gth = torch.clamp(gth, min=1)
                tgtdx = (gtx - posancx) / posancw
                tgtdy = (gty - posancy) / posanch
                tgtdw = torch.log(gtw / posancw)
                tgtdh = torch.log(gth / posanch)
                tgts = torch.stack((tgtdx, tgtdy, tgtdw, tgtdh))
                tgts = tgts.t()
                tgts = tgts/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                neginds = 1 - posinds
                regdiff = torch.abs(tgts - reg[posinds, :])
                regloss = torch.where(torch.le(regdiff, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regdiff, 2), regdiff - 0.5 / 9.0)
                reglosses.append(regloss.mean())
            else:
                reglosses.append(torch.tensor(0).float().cuda())
        return torch.stack(ctglosses).mean(dim=0, keepdim=True), torch.stack(reglosses).mean(dim=0, keepdim=True)
