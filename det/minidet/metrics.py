import torch
import numpy as np
from .utils import iou


def mean(lst, keys):
    avg = dict()
    for key in keys:
        avg[key] = sum([item[key] for item in lst]) / len(lst)
    return avg


def calap(recall, precision):
    # correct AP calculation, first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calind(annots, detects, iouthres, ncls):
    avgprcs = {}
    for lbl in range(ncls):
        trupos, scrs, nann = [], [], 0
        for i in range(len(detects)):
            dets, anns = detects[i][lbl], annots[i][lbl]
            nann += anns.shape[0]
            detanns = []
            for det in dets:
                bbox, scr = det[:4], det[-1]
                scrs.append(scr)
                if anns.shape[0] == 0:
                    trupos.append(0)
                    continue
                overlaps = iou(bbox.unsqueeze(0), anns.type(torch.FloatTensor),
                               mode='m2n')
                assignann = torch.argmax(overlaps, axis=1)
                maxoverlap = overlaps[0, assignann]
                if maxoverlap >= iouthres and assignann not in detanns:
                    trupos.append(1)
                    detanns.append(assignann)
                else:
                    trupos.append(0)
        if nann == 0:
            avgprcs[lbl] = 0
            continue
        trupos, falpos = np.array(trupos), np.ones_like(trupos) - trupos
        indices = np.argsort(-np.array(scrs))
        trupos = np.cumsum(trupos[indices])
        falpos = np.cumsum(falpos[indices])
        recall = trupos / nann
        prc = trupos / np.maximum(trupos + falpos, np.finfo(np.float64).eps)
        # compute average precision
        avgprcs[lbl] = calap(recall, prc)
    mAP = np.mean(list(avgprcs.values()))
    return avgprcs, mAP
