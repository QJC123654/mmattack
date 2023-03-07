import numpy as np
import json
import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
import os 
os.sys.path.insert(0, './mmeval/')
from mmeval import COCODetection
from pycocotools.coco import COCO
import argparse

def apEval(attack_mode:str, *args, **kwargs):
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

  # Choose to use a config and initialize the detector
  # config = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
  config = 'configs/gfl/gfl_r50_fpn_1x_coco.py'
  # Setup a checkpoint file to load
  # checkpoint = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
  checkpoint = 'checkpoints/gfl_r50_fpn_1x_coco_20200629_121244-25944287.pth'

  # Set the device to be used for evaluation
  device='cuda:0'

  # Load the config
  config = mmcv.Config.fromfile(config)
  # Set pretrained to be None since we do not need pretrained model here
  config.model.pretrained = None

  # Initialize the detector
  model = build_detector(config.model)

  # Load checkpoint
  checkpoint = load_checkpoint(model, checkpoint, map_location=device)

  # Set the classes of models for inference
  model.CLASSES = checkpoint['meta']['CLASSES']

  # We need to set the model's cfg for inference
  model.cfg = config

  # Convert the model to GPU
  model.to(device)
  # Convert the model into evaluation mode
  model.eval()

  # do eval
  ann_path = '../annotations/' + attack_mode + '_val2017.json'
  data = json.load(open(ann_path))


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
  name = kwargs['mode'] if attack_mode == 'TargetedAttack' else ''
  root_path = '../' + attack_mode + name + '/'
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

  fake_dataset_metas = {
      'CLASSES': tuple([str(i) for i in range(80)])
  }
  coco_det_metric = COCODetection(
      #ann_file = '/content/annotations/instances_val2017.json',
      dataset_meta=fake_dataset_metas,
      metric=['bbox'],
      classwise=True
  )

  coco_det_metric(predictions=predictions, groundtruths=groundtruths)

def parse_args():
    parser = argparse.ArgumentParser(description='generate attack on images')
    parser.add_argument('attack_mode', help='which attack to choose')
    parser.add_argument('--mode', type=str, default='ll', help='which targeted attack to choose')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    apEval(args.attack_mode, mode = args.mode)
    
main()
