import torch
import numpy as np
import torch.nn as nn


class Anchors(nn.Module):

    def __init__(self, levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        self.levels = levels if levels else [3, 4, 5, 6, 7]
        self.strides = strides if strides else [2 ** x for x in self.levels]
        self.sizes = sizes if sizes else [2 ** (x + 2) for x in self.levels]
        self.ratios = ratios if ratios else np.array([0.5, 1, 2])
        self.scales = scales if scales else np.array([1, 2**(1/3), 2**(2/3)])

    def forward(self, img):
        imgshape = np.array(img.shape[2:])
        imgshapes = [(imgshape + 2**x - 1) // (2**x) for x in self.levels]
        # compute anchors over all pyramid levels
        allancs = np.zeros((0, 4)).astype(np.float32)
        for idx, l in enumerate(self.levels):
            ancs = self.genancs(self.sizes[idx], self.ratios, self.scales)
            shiftedancs = self.shift(imgshapes[idx], self.strides[idx], ancs)
            allancs = np.append(allancs, shiftedancs, axis=0)
        allancs = np.expand_dims(allancs, axis=0)
        return torch.from_numpy(allancs.astype(np.float32)).cuda()

    def genancs(self, basesz, ratios, scales):
        ancnum = len(ratios) * len(scales)
        # initialize output anchors
        ancs = np.zeros((ancnum, 4))
        # scale basesize
        ancs[:, 2:] = basesz * np.tile(scales, (2, len(ratios))).T
        # compute areas of ancs
        areas = ancs[:, 2] * ancs[:, 3]
        # correct for ratios
        ancs[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        ancs[:, 3] = ancs[:, 2] * np.repeat(ratios, len(scales))
        # transform from (cx, cy, w, h) -> (x1, y1, x2, y2)
        ancs[:, 0::2] -= np.tile(ancs[:, 2] * 0.5, (2, 1)).T
        ancs[:, 1::2] -= np.tile(ancs[:, 3] * 0.5, (2, 1)).T
        return ancs

    def shift(self, shape, stride, ancs):
        shiftx = (np.arange(0, shape[1]) + 0.5) * stride
        shifty = (np.arange(0, shape[0]) + 0.5) * stride
        shiftx, shifty = np.meshgrid(shiftx, shifty)
        shifts = np.vstack((shiftx.ravel(), shifty.ravel(), shiftx.ravel(),
                            shifty.ravel())).transpose()
        # add A anchors (1, A, 4) to  cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)  reshape to (K*A, 4) shifted anchors
        A = ancs.shape[0]
        K = shifts.shape[0]
        allancs = (ancs.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        allancs = allancs.reshape((K * A, 4))
        return allancs


if __name__ == '__main__':
    ancs = Anchors()
    img = torch.ones(64, 3, 520, 520)
    ancs = ancs(img)
    print(ancs.shape)
