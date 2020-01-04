import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler
from .coco import Coco
from .trans import Resizer, Augmenter, Normalizer


def collater(data):
    imgs = [s['img'] for s in data]
    anns = [s['ann'] for s in data]
    scls = [s['scl'] for s in data]
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batchsz = len(imgs)
    maxwidth = np.array(widths).max()
    maxheight = np.array(heights).max()
    padimgs = torch.zeros(batchsz, maxwidth, maxheight, 3)
    for i in range(batchsz):
        img = imgs[i]
        padimgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
    maxnum = max(ann.shape[0] for ann in anns)
    if maxnum > 0:
        padanns = torch.ones((len(anns), maxnum, 5)) * -1
        for idx, ann in enumerate(anns):
            if ann.shape[0] > 0:
                padanns[idx, :ann.shape[0], :] = ann
    else:
        padanns = torch.ones((len(anns), 1, 5)) * -1
    padimgs = padimgs.permute(0, 3, 1, 2)
    return {'img': padimgs, 'ann': padanns, 'scl': scls}


class AspectRatioBasedSampler(Sampler):

    def __init__(self, datasrc, batchsz, droplast=False):
        self.datasrc = datasrc
        self.batchsz = batchsz
        self.droplast = droplast
        self.groups = self.groupimgs()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.droplast:
            return len(self.datasrc) // self.batchsz
        else:
            return (len(self.datasrc) + self.batchsz - 1) // self.batchsz

    def groupimgs(self):
        # determine the order of the images
        order = list(range(len(self.datasrc)))
        order.sort(key=lambda x: self.datasrc.aspratio(x))
        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batchsz)]
                for i in range(0, len(order), self.batchsz)]


def loaddata(cfg):
    tratrans = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
    tradata = Coco(cfg.datadir, setname='train2017', transform=tratrans)
    valtrans = transforms.Compose([Normalizer(), Resizer()])
    valdata = Coco(cfg.datadir, setname='val2017', transform=valtrans)
    trasampler = AspectRatioBasedSampler(tradata, batchsz=cfg.batchsz,
                                         droplast=False)
    traloader = DataLoader(tradata, num_workers=3, collate_fn=collater,
                           batch_sampler=trasampler)
    return traloader, valdata


if __name__ == '__main__':
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.dataset = 'coco'
    cfg.datadir = '/home/tesla/Workspace/dataset/coco'
    traloader, valloader = loaddata(cfg)
    d = next(iter(traloader))
    print(d)
