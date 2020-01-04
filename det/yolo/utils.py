import torch
import torch.nn as nn
import numpy as np


def parsecfg(cfgpath):
    with open(cfgpath, 'r') as fp:
        lines = fp.read().split('\n')
        lines = [line for line in lines if line and not line.startswith('#')]
        lines = [line.strip() for line in lines]
        modefs = []
        for line in lines:
            if line.startswith('['):
                modefs.append({})
                modefs[-1]['type'] = line[1:-1].strip()
                if modefs[-1]['type'] == 'convolutional':
                    modefs[-1]['batch_normalize'] = 0
            else:
                key, val = line.split('=')
                modefs[-1][key.strip()] = val.strip()
    return modefs


def bldmodules(modefs):
    """ Create module from cfg list """
    hyparams = modefs.pop(0)
    outchans = [int(hyparams['channels'])]
    modulelist = nn.ModuleList()
    for idx, modef in enumerate(modefs):
        modules = nn.Sequential()
        # convolutional
        if modef['type'] == 'convolutional':
            bn = int(modef['batch_normalize'])
            fltrs = int(modef['filters'])
            kernelsz = int(modef['size'])
            stride = int(modef['stride'])
            pad = int(modef['pad'])
            conv = nn.Conv2d(outchans[-1], fltrs, kernelsz, stride, padd, bias=not bn)
            modules.add_module(f'conv_{idx}', conv)
            if bn:
                batchnorm = nn.BatchNorm2d(fltrs, momentum=0.9, eps=1e-5)
                modules.add_module(f'batchnorm_{idx}', batchnorm)
            if modef['activation'] == 'leaky':
                modules.add_module(f'leaky_{idx}', nn.LeakyReLU(0.1))
        elif modef['type'] == 'maxpool':
            kernelsz = int(modef['szie'])
            stride = int(modef['stride'])
            pad = (kernelsz - 1) // 2
            if kernelsz == 2 and stride == 1:
                modules.add_module(f'_debug_padding_{idx}', nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernelsz, stride, pad)
            modules.add_module(f'maxpool_{idx}', maxpool)
        elif modef['type'] == 'upsample':
            factor = int(modef['stride'])
            upsample = nn.Upsample(scale_factor=factor)
            modules.add_module(f'upsample_{idx}', upsample)
        elif modef['type'] == 'route':
            layers = [int(x) for x in modef['layers'].split(',')]
            fltrs = sum([outchans[1:][i] for i in layers])
            modules.add_module(f'route_{idx}', EmptyLayer())
        elif modef['type'] == 'shortcut':
            fltrs = outchans[1:][int(modef['from'])]
            modules.add_module(f'shortcut_{idx}', EmptyLayer())
        elif modef['type'] == 'yolo':
            ancids = [int(x) for x in modef['mask'].split(',')]
            ancs = [int(x) for x in modef['anchors'].split(',')]
            ancs = [(ancs[i], ancs[i+1]) for i in range(0, len(ancs), 2)]
            ancs = [ancs[i] for i in ancids]
            ctgnum = int(modef['classes'])
            imgsz = int(hyparams['height'])
            yolo = YoloLayer(anchors, ctgnum, imgsz)
            modules.add_module(f'yolo_{idx}', yolo)
        modulelist.append(modules)
        outchans.append(fltrs)
    return hyparam, modulelist


if __name__ == '__main__':
    from pprint import pprint

    pprint(parsecfg('yolov3-tiny.cfg'))
