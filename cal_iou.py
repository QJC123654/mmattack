import torch
from mmdet.core.bbox.iou_calculators import bbox_overlaps

bboxes1 = torch.FloatTensor([
        [0, 0, 10, 10],
    ])
bboxes2 = torch.FloatTensor([
    [0, 0, 10, 20],
    [0, 10, 10, 19],
    [10, 10, 20, 20],
    ])


overlaps = bbox_overlaps(bboxes1, bboxes2)

print(overlaps.max(dim = -1)[0].item())



    