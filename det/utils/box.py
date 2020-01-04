import torch.nn as nn


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        self.mean = mean if mean else torch.tensor([0., 0., 0., 0.]).cuda()
        self.std = std if std else torch.tensor([.1, .1, .2, .2]).cuda()

    def forward(self, bxs, deltas):
        gtw = bxs[:, :, 2] - bxs[:, :, 0]
        gth = bxs[:, :, 3] - bxs[:, :, 1]
        gtx = bxs[:, :, 0] + 0.5 * gtw
        gty = bxs[:, :, 1] + 0.5 * gth
        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]
        predx = gtx + dx * gtw
        predy = gty + dy * gth
        predw = torch.exp(dw) * gtw
        predh = torch.exp(dh) * gth
        predx1 = predx - 0.5 * predw
        predy1 = predy - 0.5 * predh
        predx2 = predx + 0.5 * predw
        predy2 = predy + 0.5 * predh
        predbxs = torch.stack([predx1, predy1, predx2, predy2], dim=2)
        return predbxs


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, bxs, img):
        batchsz, channum, height, width = img.shape
        bxs[:, :, 0] = torch.clamp(bxs[:, :, 0], min=0)
        bxs[:, :, 1] = torch.clamp(bxs[:, :, 1], min=0)
        bxs[:, :, 2] = torch.clamp(bxs[:, :, 2], max=width)
        bxs[:, :, 3] = torch.clamp(bxs[:, :, 3], max=height)
        return bxs
