import torch


def nms(pred, ncls, cfdthres=0.5, nmsthres=0.4):
    """
    Performs Non-Maximum Suppression to filter detections.
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
