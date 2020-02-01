import torch.nn as nn
from .blocks import Basic, Bottleneck
# from blocks import Basic, Bottleneck


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

class UpResNet(nn.Module):

    def __init__(self, name):
        self.inplanes = 64
        super(UpResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._mklayer(blocks[name], 64, layers[name][0])
        self.layer2 = self._mklayer(blocks[name], 128, layers[name][1], stride=2)
        self.layer3 = self._mklayer(blocks[name], 256, layers[name][2], stride=2)
        self.layer4 = self._mklayer(blocks[name], 512, layers[name][3], stride=2)
        self.trpconv1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.trpconv2 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.trpconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.trpconv1(x)
        x = self.trpconv2(x)
        x = self.trpconv3(x)
        return x

    def _mklayer(self, block, planes, blocks, stride=1):
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
    import torch

    upresnet = UpResNet('resnet152')
    print(upresnet)
    inp = torch.rand(4, 3, 256, 256)
    ftr = upresnet(inp)
    print(ftr.shape)
