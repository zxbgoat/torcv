import torch
import torch.nn as nn


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


def iou(box1, box2, mode='1to1'):
    """ Returns the IoU of two bounding boxes.
    Parameters
        box1: tensor of (M, 4), each row is (x1, y1, x2, y2)
        box1: tensor of (N, 4), each row is (x1, y1, x2, y2)
    Returns
        IoUs tensor of (M, 1) or (M, N)
    """
    if mode == 'm2n':
        M, N = box1.shape[0], box2.shape[0]
        box1 = box1.repeat(1, N).view(M*N, -1)
        box2 = box2.repeat(M, 1)
    # Get the coordinates of bounding boxes
    b1x1, b1y1, b1x2, b1y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2x1, b2y1, b2x2, b2y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # get the corrdinates of the intersection rectangle
    ix1, iy1 = torch.max(b1x1, b2x1), torch.max(b1y1, b2y1)
    ix2, iy2 = torch.min(b1x2, b2x2), torch.min(b1y2, b2y2)
    # Intersection area
    inarea = torch.clamp(ix2-ix1, min=0) * torch.clamp(iy2-iy1, min=0)
    # Union Area
    b1area = (b1x2 - b1x1) * (b1y2 - b1y1)
    b2area = (b2x2 - b2x1) * (b2y2 - b2y1)
    unarea = torch.clamp(b1area + b2area - inarea, min=1e-16)
    iou = inarea / unarea
    if mode == 'm2n':
        iou = iou.view(-1, N)
    return iou


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


if __name__ == '__main__':
    box1 = torch.tensor([[20., 30., 80., 120.],
                         [30., 40., 90., 160.]])
    box2 = torch.tensor([[10., 10., 90., 130.],
                         [50., 60., 150., 160.],
                         [120., 140., 150., 160.]])
    print(iou(box1, box2, mode='m2n'))
