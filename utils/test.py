import json
import torch
from pycocotools.cocoeval import COCOeval


def test(model, valdata, threshold=0.05):
    model.eval()
    with torch.no_grad():
        # start collecting results
        results, imgids = [], []
        for idx, data in enumerate(valdata):
            img = data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            scrs, lbls, bbxs = model(img)
            scrs, lbls, bbxs = scrs.cpu(), lbls.cpu(), bbxs.cpu()
            # correct bboxs for image scale
            bbxs /= data['scl']
            if bbxs.shape[0] > 0:
                # change to (x, y, w, h)
                bbxs[:, 2] -= bbxs[:, 0]
                bbxs[:, 3] -= bbxs[:, 1]
                for bbxid in range(bbxs.shape[0]):
                    scr = float(scrs[bbxid])
                    lbl = int(lbls[bbxid])
                    bbx = bbxs[bbxid, :]
                    # scores are sorted, so we can break
                    if scr < threshold:
                        break
                    # append detection for each positively labeled class
                    imgres = {'image_id': data.imgids[idx],
                              'category_id': data.getcocolbl(lbl),
                              'score': float(scr),
                              'bbox': bbx.tolist()}
                    # append detection to results
                    results.append(imgres)
            # append image to list of processed images
            imgids.append(valdata.imgids[idx])
            # print progress
            print('{}/{}'.format(idx, len(data)), end='\r')
        if not len(results):
            return
        # write output
        json.dump(results, open(f'{data.setname}_bbox_results.json', 'w'), indent=4)
        # load results in COCO evaluation tool
        cocotrue = data.coco
        cocopred = cocotrue.loadRes(f'{data.setname}_bbox_results.json')
        # run COCO evaluation
        cocoeval = COCOeval(coco_true, coco_pred, 'bbox')
        cocoeval.params.imgIds = imgids
        cocoeval.evaluate()
        cocoeval.accumulate()
        cocoeval.summarize()
        model.train()
