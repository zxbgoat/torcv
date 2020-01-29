import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, num_layer, batch_norm):
        super(VGG, self).__init__()
        self.features = self.mkfeats()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self.mkclsf()

        def mkftrs():


        def mkclsf():
        clsfr = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                              nn.ReLU(True),
                              nn.Dropout(),
                              nn.Linear(4096, 4096),
                              nn.ReLU(True),
                              nn.Dropout(),
                              nn.Linear(4096, num_classes))
        return clsfr
