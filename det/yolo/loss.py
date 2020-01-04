import torch.nn as nn


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    batchsz = pred_boxes.size(0)
    ancnum = pred_boxes.size(1)
    ctgnum = pred_cls.size(-1)
    gridsz = pred_boxes.size(2)
    # Output tensors
    obj_mask = ByteTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    noobj_mask = ByteTensor(batchsz, ancnum, gridsz, gridsz).fill_(1)
    class_mask = FloatTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    iou_scores = FloatTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    tx = FloatTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    ty = FloatTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    tw = FloatTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    th = FloatTensor(batchsz, ancnum, gridsz, gridsz).fill_(0)
    tcls = FloatTensor(batchsz, ancnum, gridsz, gridsz, ctgnum).fill_(0)
    # Convert to position relative to box
    target_boxes = target[:, 2:6] * gridsz
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0
    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0
    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


class L1Smooth(nn.Module):

    def __init__(self, objscl=1, nobscl=100):
        self.objscl = objscl
        self.nobscl = nobscl

    def forward(self, res, lbls):
        iouscrs, clsmask, objmask, nobmask, tx, ty, tw, th, tcls, tcfd = bldtargets(predbxs, ctg, lbls, scldancs, thres)
        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        lossx = self.mse_loss(x[objmask], tx[objmask])
        lossy = self.mse_loss(y[objmask], ty[objmask])
        lossw = self.mse_loss(w[objmask], tw[objmask])
        lossh = self.mse_loss(h[objmask], th[objmask])
        losscfdobj = self.bce_loss(predcfd[objmask], tcfd[objmask])
        losscfdnob = self.bce_loss(predcfd[nobmask], tcfd[nobmask])
        losscfd = self.objscl * losscfdobj + self.nobscl * losscfdnob
        losscls = self.bce_loss(predcls[objmask], tcls[objmask])
        loss = lossx + lossy + lossw + lossh + losscfd + losscls
        return loss
