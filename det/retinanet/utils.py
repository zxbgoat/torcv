import torch


def cvt(cords, base=(0, 0), facts=(1, 1), mode='center'):
    """ convert coords between center and corner, yet predict and target """
    _cords = torch.empty_like(cords)
    if mode == 'center':  # From (x, y, w, h) to (x1, y1, x2, y2)
        _cords[:, 0] = (cords[:, 0] - cords[:, 2] / 2) * facts[0]
        _cords[:, 1] = (cords[:, 1] - cords[:, 3] / 2) * facts[1]
        _cords[:, 2] = (cords[:, 0] + cords[:, 2] / 2) * facts[0]
        _cords[:, 3] = (cords[:, 1] + cords[:, 3] / 2) * facts[1]
    elif mode == 'corner':  # From (x1, y1, x2, y2) to (x, y, w, h)
        _cords[:, 0] = (cords[:, 0] + cords[:, 2]) / 2 * facts[0]
        _cords[:, 1] = (cords[:, 1] + cords[:, 3]) / 2 * facts[1]
        _cords[:, 2] = (cords[:, 2] - cords[:, 0]) * facts[0]
        _cords[:, 3] = (cords[:, 3] - cords[:, 1]) * facts[1]
    elif mode == 'predict':  # From (px, py, pw, ph) to (tx, ty, tw, th)
        _cords[..., 0] = cords[..., 0].data + base[0]
        _cords[..., 1] = cords[..., 1].data + base[1]
        _cords[..., 2] = torch.exp(cords[..., 2]) * facts[0]
        _cords[..., 3] = torch.exp(cords[..., 3]) * facts[1]
    elif mode == 'target':  # From (tx, ty, tw, th) to (px, py, pw, ph)
        _cords[..., 0] = cords[0] - base[0]
        _cords[..., 1] = cords[1] - base[1]
        _cords[..., 2] = torch.log(cords[..., 2] / facts[0] + 1e-16)
        _cords[..., 3] = torch.log(cords[..., 3] / facts[1] + 1e-16)
    return _cords


def nms(pred, ncls, cfdthres=0.5, nmsthres=0.4):
    """ Performs Non-Maximum Suppression to filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, cfd, scr, ctg)
    """
    pred[:, :, :4] = cvt(pred[:, :, :4])
    output = [None for _ in range(len(pred))]
    for image_i, image_pred in enumerate(pred):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= cfdthres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5+ncls], 1,
                                           keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, objconf, classconf, classpred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(),
                                class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if pred.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4],
                                            descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detec with highest confidence and save as max detect
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nmsthres]
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (max_detections if output[image_i] is None else
                               torch.cat((output[image_i], max_detections)))
    return output


def iou(a, b):
    """ Calculate iou value for two sets of boxes """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU
