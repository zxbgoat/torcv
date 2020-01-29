import torch
import torch.nn as nn
# from .blocks import Basic, Bottleneck
from blocks import Basic, Bottleneck


urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}

blocks = {'resnet18': Basic,
          'resnet34': Basic,
          'resnet50': Bottleneck,
          'resnet101': Bottleneck,
          'resnet152': Bottleneck}

layers = {'resnet18': [2, 2, 2, 2],
          'resnet34': [3, 4, 6, 3],
          'resnet50': [3, 4, 6, 3],
          'resnet101': [3, 4, 23, 3],
          'resnet152': [3, 4, 36, 3]}


class ResNet(nn.Module):

    def __init__(self, name):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[name], 64, layers[name][0])
        self.layer2 = self._make_layer(blocks[name], 128, layers[name][1], stride=2)
        self.layer3 = self._make_layer(blocks[name], 256, layers[name][2], stride=2)
        self.layer4 = self._make_layer(blocks[name], 512, layers[name][3], stride=2)


    def forward(self, inputs):
        x = self.conv1(inputs)
        print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)
        x1 = self.layer1(x)
        print(x1.shape)
        x2 = self.layer2(x1)
        print(x2.shape)
        x3 = self.layer3(x2)
        print(x3.shape)
        x4 = self.layer4(x3)
        print(x4.shape)
        return [x2, x3, x4]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


if __name__ == '__main__':
    resnet = ResNet('resnet50')
    print(resnet)
    x = torch.rand(1, 3, 256, 256)
    x2, x3, x4 = resnet(x)
