import torch
import torch.nn as nn
from .iou import caliou


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2.0

    def forward(self, clsfs, rgrss, ancs, annos):
        batchsz = clsfs.shape[0]
        clslosses = []
        reglosses = []
        anc = ancs[0, :, :]
        ancw = anc[:, 2] - anc[:, 0]
        anch = anc[:, 3] - anc[:, 1]
        ancx = anc[:, 0] + 0.5 * ancw
        ancy = anc[:, 1] + 0.5 * anch
        for j in range(batchsz):
            clsf = clsfs[j, :, :]
            rgrs = rgrss[j, :, :]
            anns = annos[j, :, :]
            anns = anns[anns[:, 4] != -1]
            if anns.shape[0] == 0:
                reglosses.append(torch.tensor(0).float().cuda())
                clslosses.append(torch.tensor(0).float().cuda())
                continue
            clsf = torch.clamp(clsf, 1e-4, 1.0 - 1e-4)
            ious = caliou(ancs[0, :, :], anns[:, :4]) # ancnum x annnum
            maxiou, maxidx = torch.max(ious, dim=1) # num_anchors x 1
            # compute the loss for classification
            targets = torch.ones(clsf.shape) * -1
            targets = targets.cuda()
            targets[torch.lt(maxiou, 0.4), :] = 0
            posids = torch.ge(maxiou, 0.5)
            posnum = posids.sum()
            assignedanns = anns[maxidx, :]
            targets[posids, :] = 0
            targets[posids, assignedanns[posids, 4].long()] = 1
            alpha = torch.ones(targets.shape).cuda() * self.alpha
            alpha = torch.where(torch.eq(targets, 1.), alpha, 1. - alpha)
            focal = torch.where(torch.eq(targets, 1.), 1. - clsf, clsf)
            focal = alpha * torch.pow(focal, self.gamma)
            bce = -(targets * torch.log(clsf) + (1.0 - targets) * torch.log(1.0 - clsf))
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            clsloss = focal * bce
            clsloss = torch.where(torch.ne(targets, -1.0), clsloss, torch.zeros(clsloss.shape).cuda())
            clslosses.append(clsloss.sum()/torch.clamp(posnum.float(), min=1.0))
            # compute the loss for regression
            if posids.sum() > 0:
                assignedanns = assignedanns[posids, :]
                posancw = ancw[posids]
                posanch = anch[posids]
                posancx = ancx[posids]
                posancy = ancy[posids]
                gtw = assignedanns[:, 2] - assignedanns[:, 0]
                gth = assignedanns[:, 3] - assignedanns[:, 1]
                gtx = assignedanns[:, 0] + 0.5 * gtw
                gty = assignedanns[:, 1] + 0.5 * gth
                # clip widths to 1
                gtw = torch.clamp(gtw, min=1)
                gth = torch.clamp(gth, min=1)
                tdx = (gtx - posancx) / posancw
                tdy = (gty - posancy) / posanch
                tdw = torch.log(gtw / posancw)
                tdh = torch.log(gth / posanch)
                targets = torch.stack((tdx, tdy, tdw, tdh))
                targets = targets.t()
                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                neginds = 1 - posids
                regdif = torch.abs(targets - rgrs[posids, :])
                reglos = torch.where(torch.le(regdif, 1.0 / 9.0), 0.5 * 9.0 * torch.pow(regdif, 2), regdif - 0.5 / 9.0)
                reglosses.append(reglos.mean())
            else:
                reglosses.append(torch.tensor(0).float().cuda())
        return torch.stack(clslosses).mean(dim=0, keepdim=True), torch.stack(reglosses).mean(dim=0, keepdim=True)
