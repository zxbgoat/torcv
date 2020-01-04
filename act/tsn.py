import torch
import torch.nn as nn


class MaxPooling(nn.Module):

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(x):
        return torch.max(x)


class AvgPooling(nn.Module):
    
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(x):
        return torch.mean(x)


class TopKPooling(nn.Module):

    def __init__(self):
        super(TopKPooling, self).__init__()

    def forward(x):
        return torch.max(x, k)


class LinearWeighting(nn.Module):

    def __init__(self):
        super(LinearWeighting, self).__init__()
        self.linear = nn.Linear()

    def forward(x):
        pass


class AttentionWeighting(nn.Module):

    def __init__(self):
        super(AttentionWeighting, self).__init__()

    def forward(x):
        pass


class TemporalSegmentNetWork(nn.Module):

    def __init__(self):
        super(TemporalSegmentNetwork, self).__init__()
        self.convnet = ConvNet()
        self.concensus = Concensus()
        self.predict = nn.Softmax()

    def forward(video):
        pass
