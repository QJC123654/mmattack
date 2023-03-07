# from load_model import load_model
# from init_inference import init_inference, inference_detector2, data_process
# from normalize import normalize, denormalize
# from loss import total_loss
# import torch
# import cv2
# from mmdet.core.bbox.iou_calculators import bbox_overlaps

# g_min_iou = 1
# keep_x = 0
# # init model
# model = load_model()
# # img path
# img = './dog_cat.jpg'
# results, gboxes, data = init_inference(model, img)
# data['img_metas'][0][0]['ori_shape'] = data['img_metas'][0][0]['img_shape']
# #get origin xyxy
# l, r, u, d = 164, 710.5, 425, 794.5
# bboxes1 = torch.FloatTensor([[l, u, r, d]])

# #test black model
# config = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
# checkpoint = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
# model2 = load_model(config, checkpoint)

# #add attack
# eps = 8
# alpha = 2
# mean = torch.tensor(data['img_metas'][0][0]['img_norm_cfg']['mean']).to(data['img'][0].device)
# std = torch.tensor(data['img_metas'][0][0]['img_norm_cfg']['std']).to(data['img'][0].device)

# mean_test = torch.tensor(model2.cfg.img_norm_cfg.mean, device=mean.device)
# std_test = torch.tensor(model2.cfg.img_norm_cfg.std, device=mean.device)

# att_data = data.copy()
# images = att_data['img'][0].clone().detach()
# images = denormalize(images, mean=mean, std=std)
# init_adv_images = images.clone().detach()
# init_adv_images = init_adv_images + torch.empty_like(init_adv_images).uniform_(-eps, eps)
# init_adv_images = torch.clamp(init_adv_images, min=0, max=255).detach()

# for x in range(1000000, -1, -10):
#     print(x)
#     x = x / 1e3
#     adv_images = init_adv_images.clone().detach()
#     for _ in range(20):
#         print('epoch:  ', _)
#         adv_images.requires_grad = True
#         att_data['img'][0] = normalize(adv_images, mean, std)
#         cost = total_loss(model, att_data, gboxes, x)
#         if not cost:
#             break
#         # Update adversarial images
#         grad = torch.autograd.grad(cost, adv_images,
#                                     retain_graph=False, create_graph=False)[0]
#         # print(grad)
#         adv_images = adv_images.detach() + alpha*grad.sign()
#         delta = torch.clamp(adv_images - images, min=-eps, max=eps)
#         adv_images = torch.clamp(images + delta, min=0, max=255).detach()
#     # cv2.imwrite('/content/adv_image.jpg', adv_images.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()[...,::-1])
#     # is_batch_, data_ = data_process(model2, '/content/adv_image.jpg')
#     # test
#     test_img = adv_images.clone().detach()
#     test_img = normalize(test_img, mean_test, std_test)  # normalize不一样
#     test_data = data.copy()
#     test_data['img'][0] = test_img
#     result = inference_detector2(model2, test_data)
#     # result = inference_detector2(model2, data_)
#     result = result[0]
    
#     idx = result[0][:, 4] > 0.3
#     det_labels = result[1][idx]
#     if not (det_labels == 15).any():
#         with open('./log.txt', 'a+') as f:
#             f.write(str(x) + '没检测到\n')
#         f.close()
#         continue
#     else:
#         # cal iou
#         det_boxes = result[0][idx]
#         att_boxes = det_boxes[det_labels == 15]
#         max_iou = 0
#         bboxes2 = att_boxes[..., 0:4]
#         print(bboxes2)
#         overlaps = bbox_overlaps(bboxes1.to(bboxes2.device), bboxes2)
#         max_iou = overlaps.max(dim = -1)[0].item()
#         print('max_iou:', max_iou)
#         if max_iou < 0.5:
#             with open('./log.txt', 'a+') as f:
#                 f.write(str(x) + ' 检测不准的iou为 ' + str(max_iou) + '\n')
#             f.close()
#         if max_iou < g_min_iou:
#             g_min_iou = max_iou
#             keep_x = x

# with open('./log.txt', 'a+') as f:
#     f.write('最小iou为:'+ str(g_min_iou) + 'x为' + str(keep_x) + '\n')
# f.close()

                
from statistics import mode
from load_model import load_model
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.apis import inference_detector
import numpy as np
import json
import os 
os.sys.path.insert(0, './mmeval/')
from mmeval import COCODetection
from pycocotools.coco import COCO

coco = COCO('../annotations/instances_val2017.json')
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


cat_ids = coco.getCatIds(catNms=CLASSES)
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

# load model
model = load_model()

def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = np.split(bbox, 4, axis=-1)
    bbox_new = [(cx), (cy), (cx + w), (cy + h)]
    return np.concatenate(bbox_new, axis=-1)


data = json.load(open('../annotations/instances_val2017.json'))

# key=image_id val=all gt bboxes and labels
gt_box_label = dict()
for annotation in data['annotations']:
  key = annotation['image_id']
  annotation['bbox'] = bbox_cxcywh_to_xyxy(np.array(annotation['bbox']).reshape(-1, 4))
  annotation['category_id'] = cat2label[annotation['category_id']]
  val = gt_box_label.get(key)
  if not val:
    gt_box_label[key] = {'labels':np.array([annotation['category_id']], dtype=np.int64)}
    gt_box_label[key].update({'bboxes':annotation['bbox']})
  else:
    val['bboxes'] = np.concatenate([val['bboxes'].reshape(-1, 4), annotation['bbox']], axis=0)
    val['labels'] = np.concatenate([val['labels'].reshape(-1), np.array([annotation['category_id']], dtype=np.int64).reshape(-1)], axis=0)

# get groundtruths and predictions
groundtruths = []
predictions = []
root_path = '../val2017/'
for img_info in data['images']:
  groundtruth = dict()
  prediction = dict()
  img_path = root_path + img_info['file_name']
  img_id = img_info['id']
  # get gt box and label
  if gt_box_label.get(img_id):
    bboxes = gt_box_label[img_id]['bboxes']
    labels = gt_box_label[img_id]['labels']
    width = img_info['width']
    height = img_info['height']
    groundtruth.update({'img_id':img_id})
    groundtruth.update({'width':width})
    groundtruth.update({'height':height})
    groundtruth.update({'bboxes':bboxes})
    groundtruth.update({'labels':labels})
    groundtruths.append(groundtruth)
    # get pred box and label
    prediction.update({'img_id':img_id})
    # Use the detector to do inference
    result = inference_detector(model, img_path)
    labels = np.empty((0), dtype=np.int64)
    bboxes = np.empty((0, 4))
    scores = np.empty((0))
    for i, res in enumerate(result):
      if res.shape[0] != 0:
        idx = res[..., 4] > 0.3
        if idx.any():
          valid_res = res[idx]
          bboxes = np.concatenate([bboxes, valid_res[..., :-1]], axis=0)
          scores = np.concatenate([scores, valid_res[..., -1]], axis=0)
          labels = np.concatenate([labels, np.array([i] * valid_res.shape[0], dtype=np.int64)], axis=0)
    prediction.update({'bboxes':bboxes})
    prediction.update({'scores':scores})
    prediction.update({'labels':labels})
    predictions.append(prediction)
  
# judge base on iou and labels
for gt, pred in zip(groundtruths, predictions):
  overlaps = bbox_overlaps(gt['bboxes'], pred['bboxes'])
  