import torch
import torch.nn as nn
import troch.nn.functional as F


class DenseLayer(nn.Module):

    def __init__(self, num_inftrs, growrt, bnsz, droprt, mem_eff=False):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_inftrs))
        self.add_module('relu1', nn.ReLU(inplace=True))
