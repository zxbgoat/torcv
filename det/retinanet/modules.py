import torch.nn as nn


class Regressor(nn.Module):
    """ Regression subnet of the RetinaNet """

    def __init__(self, infeatnum, ancnum=9, featsz=256):
        super(Regressor, self).__init__()
        self.conv1 = nn.Conv2d(infeatnum, featsz, kernel_size=3, padding=1)
        self.acti1 = nn.ReLU()
        self.conv2 = nn.Conv2d(featsz, featsz, kernel_size=3, padding=1)
        self.acti2 = nn.ReLU()
        self.conv3 = nn.Conv2d(featsz, featsz, kernel_size=3, padding=1)
        self.acti3 = nn.ReLU()
        self.conv4 = nn.Conv2d(featsz, featsz, kernel_size=3, padding=1)
        self.acti4 = nn.ReLU()
        self.output = nn.Conv2d(featsz, ancnum*4, kernel_size=3, padding=1)

    def forward(self, x):
        """ The shape of x is [batchsz, featsz, width, height] """
        out = self.conv1(x)
        out = self.acti1(out)
        out = self.conv2(out)
        out = self.acti2(out)
        out = self.conv3(out)
        out = self.acti3(out)
        out = self.conv4(out)
        out = self.acti4(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        # out is [B, W x H x A, 4]
        return out.contiguous().view(out.shape[0], -1, 4)


class Classifier(nn.Module):
    """ Classification subnet of the RetinaNet """

    def __init__(self, infeatnum, ancnum=9, ctgnum=80,  featsz=256):
        super(Classifier, self).__init__()
        self.ctgnum = ctgnum
        self.ancnum = ancnum
        self.conv1 = nn.Conv2d(infeatnum, featsz, kernel_size=3, padding=1)
        self.acti1 = nn.ReLU()
        self.conv2 = nn.Conv2d(featsz, featsz, kernel_size=3, padding=1)
        self.acti2 = nn.ReLU()
        self.conv3 = nn.Conv2d(featsz, featsz, kernel_size=3, padding=1)
        self.acti3 = nn.ReLU()
        self.conv4 = nn.Conv2d(featsz, featsz, kernel_size=3, padding=1)
        self.acti4 = nn.ReLU()
        self.output = nn.Conv2d(featsz, ancnum*ctgnum, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.acti1(out)
        out = self.conv2(out)
        out = self.acti2(out)
        out = self.conv3(out)
        out = self.acti3(out)
        out = self.conv4(out)
        out = self.acti4(out)
        out = self.output(out)
        out = self.output_act(out)
        out1 = out.permute(0, 2, 3, 1)
        batchsz, width, height, channels = out1.shape
        out2 = out1.view(batchsz, width, height, self.ancnum, self.ctgnum)
        # out is [B, W x H x A, C]
        return out2.contiguous().view(batchsz, -1, self.ctgnum)
