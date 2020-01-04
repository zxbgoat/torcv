import torch
import torch.nn as nn


class StructuredTemporalPyramidPooling(nn.Module):

    def __init__(self, L, Bs):
        super(StructuredTemporalPyramidPooling, self).__init__()
        self.L = L
        self.Bs = Bs
        self.idxs = []
        for l in range(self.L):
            B = self.Bs[l]
            seq = [0]

    def forward(stage):
        """ features of the stage [fs,...,fe] """
        pool = []
        for l in range(self.L):
            B = self.Bs[l]


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

    def forward():
        pass


class StructuredSegmentNetwork(nn.Module):

    def __init__(self):
        super(StructuredSegmentNetwork, self).__init__()
        self.sttstpp = StructuredTemporalPyramidalPooling()
        self.crsstpp = StructuredTemporalPyramidalPooling()
        self.endstpp = StructuredTemporalPyramidalPooling()
        self.actclsr = Classifier()
        self.cmpclsr = Classifier()

    def forward(video, proposals):
        # splitting the proposal into three stages
        stttm, endtm = proposal
        duration = (endtm - stttm) // 2
        start = (max(0, stttm-duration), stttm)
        course = (stttm, endtm)
        end = (endtm, min(endtm+duration, len(video)))
        # building temporal pyramidal representation for each stage
        sttrepr = self.sttstpp()
        crsrepr = self.crsstpp()
        endrepr = self.endstpp()
        # building global representation for the whole proposal
        glbrepr = torch.cat(sttrepr, crsrepr, endrepr)
        actconf = self.actclsr(crsrepr)
        cmpconf = self.cmpclsr(glbrepr)
        return actconf, cmpconf
