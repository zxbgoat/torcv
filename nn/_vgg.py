import torch
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, num_ctgs=1000):
        super(VGG, self).__init__()
        feats = []
        dim_in = 3
        dim_out = 64
        for i in range(13):
            feats += [nn.Conv2d(dim_in, dim_out, 3, 1, 1),
                      nn.ReLU(inplace=True)]
            dim_in = dim_out
            if i in [1, 3, 6, 9, 12]:
                feats += [nn.MaxPool2d(2, 2)]
                if i != 9:
                    dim_out *= 2
        self.features = nn.Sequential(*feats)
        self.classifier = nn.Sequential(
            nn.Linear(),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_ctgs))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
