from utils import parsargs
from cfgs import cfg, updcfg
from utils import Train
from models import RetinaNet, MiniRetina
from data import loaddata
from pprint import pprint

args = parsargs()
cfg = updcfg(args, cfg)
print('The configs of the model:')
pprint(cfg)
print(cfg.detname)
if cfg.detname == 'retinanet':
    print('Loading the retinanet model ...')
    model = RetinaNet(cfg)
elif cfg.detname == 'miretina':
    print('Loading the miniretina model ...')
    model = MiniRetina(cfg)
data = loaddata(cfg)
if args.train:
    train = Train(cfg)
    train(model, data)
