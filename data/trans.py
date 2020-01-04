import torch
import numpy as np
import skimage as skim


class Resizer:

    def __init__(self, minside=608, maxside=1024):
        self.minside = minside
        self.maxside = maxside

    def __call__(self, sample):
        img, anns = sample['img'], sample['ann']
        rows, cols, cns = img.shape
        small, large = min(rows, cols), max(rows, cols)
        scl = min(self.minside/small, self.maxside/large)
        img = skim.transform.resize(img, (round(rows*scl), round(cols*scl)))
        rows, cols, cns = img.shape
        padw, padh = 32 - rows % 32, 32 - cols % 32
        padl, padt = padw // 2, padh // 2
        newimg = np.zeros((rows + padw, cols + padh, cns)).astype(np.float32)
        newimg[padl:rows+padl, padt:cols+padt, :] = img.astype(np.float32)
        anns[:, :4] *= scl
        anns[0, :] += padl
        anns[2, :] += padt
        return {'img': torch.from_numpy(newimg), 'ann': torch.from_numpy(anns),
                'scl': scl}


class Augmenter:

    def __init__(self, flip=0.5):
        self.flip = flip

    def __call__(self, sample):
        if np.random.rand() < self.flip:
            img, anns = sample['img'], sample['ann']
            img = img[:, ::-1, :]
            rows, cols, cns = img.shape
            x1 = anns[:, 0].copy()
            x2 = anns[:, 2].copy()
            xtmp = x1.copy()
            anns[:, 0] = cols - x2
            anns[:, 2] = cols - xtmp
            sample = {'img': img, 'ann': anns}
        return sample


class Normalizer:

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        img, anns = sample['img'], sample['ann']
        return {'img': ((img.astype(np.float32)-self.mean)/self.std),
                'ann': anns}
