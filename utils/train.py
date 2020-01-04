import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from torch.optim import lr_scheduler as lrs
from .test import test


class Train():

    def __init__(self, cfg):
        self.device = cfg.device
        self.epochs = cfg.epochs
        self.dataset = cfg.dataset
        self.detname = cfg.detname

    def __call__(self, model, data):
        tradata, valdata = data
        model = model.to(self.device)
        model = torch.nn.DataParallel(model).to(self.device)
        # Prepare for training
        model.training = True
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = lrs.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        losshist = collections.deque(maxlen=500)
        model.train()
        model.module.freezebn()
        print('Num training images: {}'.format(len(tradata)))
        for epoch in range(self.epochs)[:1]:
            model.train()
            model.module.freezebn()
            epochloss = []
            for idx, data in enumerate(tradata):
                try:
                    optimizer.zero_grad()
                    clsfloss, rgrsloss = model([data['img'].to(self.device).float(), data['ann']])
                    clsfloss = clsfloss.mean()
                    rgrsloss = rgrsloss.mean()
                    loss = clsfloss + rgrsloss
                    if bool(loss == 0):
                            continue
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    losshist.append(float(loss))
                    epochloss.append(float(loss))
                    print('Epoch: {} | Iteration: {} | \
                           Classification loss: {:1.5f} | \
                           Regression loss: {:1.5f} | \
                           Running loss: {:1.5f}'.format(
                               epoch, idx, float(clsfloss),
                               float(rgrsloss), np.mean(losshist)))
                    del clsfloss
                    del rgrsloss
                except Exception as e:
                    print(e)
                    continue
            if self.dataset == 'coco':
                print('Evaluating dataset')
                test(model, valdata)
            scheduler.step(np.mean(epochloss))	
            torch.save(model.module, '{}_{}_{}.pt'.format(
                self.dataset, self.detname, epoch))
        model.eval()
        torch.save(model, 'model_final.pt'.format(epoch))
