import torch
import numpy as np
import random as rd
import os.path as osp
from PIL import Image
from utils import cvt
from torch.utils.data import Dataset
import torchvision.transforms.functional as func


def readimg(imgpath, imgsz):
    """ Load an image as a paddedly squared, resized and normalized tensor """
    img = Image.open(imgpath)
    w, h = img.size
    diff = abs(h - w)
    pad0, pad1 = diff // 2, diff - diff // 2
    pad = (0, pad0, 0, pad1) if h <= w else (pad0, 0, pad1, 0)
    img = func.pad(img, pad, 128)
    img = func.resize(img, (imgsz, imgsz))
    img = func.to_tensor(img)
    return img, pad, (h, w, max(h, w))


def readlbl(lblpath, pad, shp, ctgnames):
    h, w, sz = shp
    lbl = torch.tensor(np.loadtxt(lblpath).reshape(-1, 5), dtype=torch.float32)
    lbl[:, 0] = torch.tensor(list(map(ctgnames.index, lbl[:, 0])))
    cord = cvt(lbl[:, 1:], facts=(w, h), mode='center')
    cord += torch.tensor(pad, dtype=torch.float32).reshape(-1, 4)
    lbl[:, 1:] = cvt(cord, facts=(1/sz, 1/sz), mode='corner')
    return lbl


class MiniCoco(Dataset):
    """
    Data class of the MiniCoco dataset.
    The format of the label is (label, ctx/imw, cty/imh, w/imw, h/imh).
    When called by the index, return the resized img and the lbl data.
    """

    def __init__(self, cfg, mode='train'):
        self.datadir = cfg.datadir
        self.imgsz = cfg.imgsz
        self.ctgnames = cfg.ctgnames
        self.names = cfg.tranames if mode == 'train' else cfg.valnames
        rd.shuffle(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        imgpath = osp.join(self.datadir, 'images', name+'.jpg')
        img, pad, shp = readimg(imgpath, self.imgsz)
        lblpath = osp.join(self.datadir, 'labels', name+'.txt')
        lbl = readlbl(lblpath, pad, shp, self.ctgnames)
        return name, img, lbl

    def __len__(self):
        return len(self.names)


if __name__ == '__main__':
    from cfg import cfg

    tracoco = MiniCoco(cfg)
    name, img, lbl = tracoco[555]
    print(name)
    print(img.shape)
    print(lbl)
    valcoco = MiniCoco(cfg, 'val')
    name, img, lbl = valcoco[666]
    print(name)
    print(img.shape)
    print(lbl)
