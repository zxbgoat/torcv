import torch
import numpy as np
import skimage as skim
import os.path as osp
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class Coco(Dataset):
    """ Coco dataset """

    def __init__(self, datadir, setname='train2017', transform=None):
        self.setname = setname
        self.datadir = datadir
        self.transform = transform
        self.annpath = osp.join(datadir, 'annotations',
                                'instances_'+setname+'.json')
        self.coco = COCO(self.annpath)
        self.imgids = self.coco.getImgIds()
        self.loadctg()

    def loadctg(self):
        ctgids = self.coco.getCatIds()
        ctginfos = self.coco.loadCats(ctgids)
        ctginfos.sort(key=lambda x: x['id'])
        self.ctgs = {}      # {ctgname: trainlabel}
        self.lbls = {}      # {trainlabel: ctgname}
        self.cocolbls = {}  # {trainlabel: cocolabel}
        self.invlbls = {}   # {cocolabel: trainlabel}
        for c in ctginfos:
            self.cocolbls[len(self.ctgs)] = c['id']
            self.invlbls[c['id']] = len(self.ctgs)
            self.lbls[len(self.ctgs)] = c['name']
            self.ctgs[c['name']] = len(self.ctgs)

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        img = self.loadimg(idx)
        ann = self.loadann(idx)
        sample = {'img': img, 'ann': ann}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def loadimg(self, idx):
        imginfo = self.coco.loadImgs(self.imgids[idx])[0]
        imgpath = osp.join(self.datadir, 'images', self.setname,
                           imginfo['file_name'])
        img = skim.io.imread(imgpath)
        if len(img.shape) == 2:
            img = skim.color.gray2rgb(img)
        return img.astype(np.float32) / 255.0

    def loadann(self, idx):
        annids = self.coco.getAnnIds(imgIds=self.imgids[idx], iscrowd=False)
        anns = np.zeros((0, 5))
        if len(annids) == 0:
            return anns
        cocoanns = self.coco.loadAnns(annids)
        for i, ca in enumerate(cocoanns):
            if ca['bbox'][2] < 1 or ca['bbox'][3] < 1:
                continue
            ann = np.zeros((1, 5))
            ann[0, :4] = ca['bbox']
            ann[0, 4] = self.invlbls[ca['category_id']]
            anns = np.append(anns, ann, axis=0)
        anns[:, 2] = anns[:, 0] + anns[:, 2]
        anns[:, 3] = anns[:, 1] + anns[:, 3]
        return anns

    def getlbl(self, cocolbl):
        return self.invlbls[cocolbl]

    def getcocolbl(self, label):
        return self.cocolbls[label]

    def aspratio(self, idx):
        imginfo = self.coco.loadImgs(self.imgids[idx])[0]
        return float(imginfo['width']) / float(imginfo['height'])
