import torch
import argparse as ap
import os.path as osp
import pickle as pkl
from easydict import EasyDict as edict

datadir = osp.join(osp.expanduser('~'), 'Workspace/dataset/minicoco')
nmpath = osp.join(datadir, 'ctgnames')
nmdict = pkl.load(open(nmpath, 'rb'))
tranames, valnames = list(), list()
for value in nmdict.values():
    ttlnum = len(value)
    tranum = int(ttlnum / 10 * 9)
    tranames += value[:tranum]
    valnames += value[tranum:]
ctgnames = sorted(list(nmdict.keys()))

cfg = edict()
# data
cfg.imgsz = 160
cfg.datadir = datadir
cfg.ncls = 60
cfg.nanc = 9
cfg.ctgnames = ctgnames
cfg.tranames = tranames
cfg.valnames = valnames
# model
cfg.ancs = [(65.28928295, 63.22346352),
            (93.43047914, 119.67837258),
            (136.68185298, 73.0129707),
            (28.31546183, 27.77671066)]
cfg.stride = 32
cfg.cfdthres = 0.5
cfg.nmsthres = 0.4
cfg.savedir = './checkpoints'
cfg.stride = 32
cfg.wtpath = None
# outer
cfg.epochs = 1000
cfg.stepsz = 0.001
cfg.batchsz = 16
# test
cfg.iouthres = 0.45
cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pararg():
    parser = ap.ArgumentParser(description='Arguments for minidet')
    parser.add_argument('--dev', type=str, default=None,
                        help='device which the model runs on')
    parser.add_argument('--wtp', type=str, default=None,
                        help='path of the weights is it exists')
    parser.add_argument('--isz', type=int, default=None,
                        help='the input image size')
    return parser.parse_args


def updcfg(cfg, arg):
    if arg.dev is not None:
        cfg.device = torch.device(arg.dev)
    if arg.wtp is not None:
        cfg.wtpath = arg.wtp
    if arg.isz is not None:
        cfg.imgsz = arg.isz
    return cfg
