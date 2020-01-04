import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as modelzoo
from .modules import Regressor, Classifier
from .anc import Anchors
from .fpn import FeaturePyramid
from .loss import FocalLoss
from .utils import nms
from .resnet import ResNet, fpnsizes, urls
from .blocks import BBoxTransform, ClipBoxes


class RetinaNet(nn.Module):

    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.training = False
        self.backbone = ResNet(cfg.backbone)
        self.fpn = FeaturePyramid(fpnsizes[cfg.backbone])
        self.regressor = Regressor(256)
        self.classifier = Classifier(256, ctgnum=cfg.ctgnum)
        self.anchors = Anchors()
        self.regrBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focaloss = FocalLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.classifier.output.weight.data.fill_(0)
        self.classifier.output.bias.data.fill_(-math.log((1.0-prior)/prior))
        self.regressor.output.weight.data.fill_(0)
        self.regressor.output.bias.data.fill_(0)
        self.freezebn()
        if cfg.pretrain == True:
            "Loading the pretrained parameters of the backbone..."
            self.backbone.load_state_dict(modelzoo.load_url(
                urls[cfg.backbone], model_dir=cfg.savedir), strict=False)

    def forward(self, inputs):
        if self.training:
            imgbatch, annots = inputs
            print(f'The shape of the training data is: imgbatch-{imgbatch.shape}, annots-{annots.shape}')
        else:
            imgbatch = inputs
        results = self.backbone(imgbatch)
        features = self.fpn(results)
        regress = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classif = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(imgbatch)
        if self.training:
            return self.focaloss(classif, regress, anchors, annots)
        transformed_anchors = self.regrBoxes(anchors, regress)
        transformed_anchors = self.clipBoxes(transformed_anchors, imgbatch)
        scores = torch.max(classif, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores>0.05)[0, :, 0]
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
        classif= classif[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)
        nms_scores, nms_class = classif[0, anchors_nms_idx, :].max(dim=1)
        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def freezebn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == '__main__':
    pass
