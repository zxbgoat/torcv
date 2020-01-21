import torch
import torch.nn as nn


class SingleShotActionDetection(nn.Module):

    def __init__(self):
        super(SingleShotActionDetection, self).__init__()
