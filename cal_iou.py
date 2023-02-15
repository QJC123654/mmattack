import torch
from mmdet.core.bbox.iou_calculators import bbox_overlaps

bboxes1 = torch.FloatTensor([
        [164, 425, 710, 794.5],
    ])
bboxes2 = torch.FloatTensor([
    [218, 234, 777, 821],
    ])


overlaps = bbox_overlaps(bboxes1, bboxes2)

print(overlaps.max(dim = -1)[0].item())



    