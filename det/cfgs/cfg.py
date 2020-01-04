import torch
import os.path as osp
from easydict import EasyDict as edict

cfg = edict()
cfg.datadir = osp.join(osp.expanduser('~'), 'Workspace/dataset/coco')
cfg.savedir = 'weights'
cfg.ctgnum = 80
cfg.detname = 'retinanet'
cfg.backbone = 'resnet50'
cfg.epochs = 100
cfg.schdl = 'ReduceLROnPlateau'
cfg.optim = 'Adam'
cfg.dataset = 'coco'
cfg.savedir = 'weights'
cfg.pretrain = True
cfg.batchsz = 64
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def updcfg(args, cfg):
    for att in dir(args):
        if not att.startswith('_'):
            if eval('args.'+att):
                cfg[att] = eval('args.' + att)
    return cfg
