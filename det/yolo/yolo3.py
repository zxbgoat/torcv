import torch.nn as nn


class EmptyLayer(nn.Module):

    def __init__(self):
        super(EmptyLayer, self).__init__


class Yolo(nn.Module):

    def __init__(self, ancs, ctgnum, imgsz):
        super(YoloLayer, self).__init__()
        self.ancs = ancs
        self.ancnum = len(ancs)
        self.threshold = 0.5
        self.imgsz = imgsz
        self.training = False

    def forward(self, ftr, tgts):
        sampnum = ftr.size(0)
        gridsz = ftr.size(2)
        pred = ftr.view(sampnum, self.ancnum, self.ctgnum+5, gridsz, gridsz).permute(0, 1, 3, 4, 2).contiguous()
        x = torch.sigmoid(pred[..., 0])
        y = torch.sigmoid(pred[..., 1])
        w = pred[..., 2]
        h = pred[..., 3]
        cfd = torch.sigmoid(pred[..., 4])
        ctg = torch.sigmoid(pred[..., 5:])
        stride = self.imgsz / gridsz
        grid = torch.arange(gridsz).repeat(gridsz, 1)
        gridx = grid.view([1, 1, gridsz, gridsz])
        gridy = grid.t().view([1, 1, gridsz, gridsz])
        scldancs = [(aw/stride, ah/stride) for aw, ah in self.ancs]
        ancw = scldancs[:, 0:1].view((1, self.ancnum, 1, 1))
        anch = scldancs[:, 1:2].view((1, self.ancnum, 1, 1))
        predbxs = pred[..., :4]
        predbxs[..., 0] = x.data + gridx
        predbxs[..., 1] = y.data + gridy
        predbxs[..., 2] = torch.exp(w.data) * ancw
        predbxs[..., 3] = torch.exp(h.data) * anch
        res = torch.cat(predbxs.view(sampnum, -1, 4), cfd.view(sampnum, -1, 1), ctg.view(sampnum, -1, self.ctgnum))
        if not self.training:
            return res


class Yolo3(nn.Module):

    def __init__(self, cfgpath, imgsz=416):
        super(Yolo3, self).__init__()
        self.modefs = parsecfg(cfgpath)
        self.hyparams, self.modlist = bldmodules(self.modefs)
        self.yololayers = [layer[0] for layer in self.modlist if hasattr(layer[0], 'metrics')]
        self.imgsz = imgsz

    def forward(self, imgs):
        loss = 0
        layerouts, outputs = [], []
        for i, (modef, module) in enumerate(zip(self.modefs, self.modlist)):
            if modef["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif modef["type"] == "route":
                x = torch.cat([layerouts[int(layer_i)] for layer_i in modef["layers"].split(",")], 1)
            elif modef["type"] == "shortcut":
                layer_i = int(modef["from"])
                x = layerouts[-1] + layerouts[layer_i]
            elif modef["type"] == "yolo":
                x, layerloss = module[0](x, targets, self.imgsz)
                loss += layerloss
                outputs.append(x)
            layerouts.append(x)
        outputs = to_cpu(torch.cat(outputs, 1))
        if self.training:
            return loss
        return outputs
