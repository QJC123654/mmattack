import json
import cv2
from attack import MyAttack_Untargeted, UntargetedAttack
from attack_test import MyAttack_Vanishing, VanishingAttack
import argparse
import shutil
from functools import partial
from mmcv import Config
import numpy as np
# os.sys.path.append('torch-cam/')
# from torchcam.methods import GradCAM
import os
os.sys.path.append('pytorch-grad-cam-master/')
try:
    from pytorch_grad_cam import GradCAM

except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.') 
from det_cam_visualizer import DetCAMModel, DetCAMVisualizer, reshape_transform, DetBoxScoreTarget

def attack(args):
    attack_mode = args.attack_mode
    assert attack_mode in ('UntargetedAttack', 'TargetedAttack', 'MyAttack_Targeted', 'VanishingAttack', 'MyAttack_Vanishing', 'MyAttack', 'MyAttack_Untargeted'), print('attack do not support!!')
    ann_file = '../annotations/instances_val2017.json'
    img_dir = '../val2017/'
    ann_data = json.load(open(ann_file))
    name = args.mode if attack_mode in ['TargetedAttack', 'MyAttack_Targeted'] else ''
    keep_img_dir_path = '../' + attack_mode + name + '/'
    keep_ann_path = '../annotations/' + attack_mode + name + 'val2017.json'
    # key = image_id val = annotations_index (list)
    gt_box_idx = dict()
    for i, annotation in enumerate(ann_data['annotations']):
        key = annotation['image_id']
        val = gt_box_idx.get(key)
        if not val:
            gt_box_idx[key] = [i]
        else:
            gt_box_idx[key].append(i)
    # 
    # load model
    cfg = Config.fromfile(args.config)
    model = DetCAMModel(
    cfg, args.checkpoint, args.score_thr, device=args.device)
    if attack_mode.startswith('My'):
        # get target_layer
        target_layers = []
        for target_layer in args.target_layers:
            try:
                target_layers.append(eval(f'model.detector.{target_layer}'))
            except Exception as e:
                print(model.detector)
                raise RuntimeError('layer does not exist', e)
        extra_params = {}
        # 暂时只加入了gradcam
        is_need_grad = True
        max_shape = args.max_shape
        if not isinstance(max_shape, list):
            max_shape = [args.max_shape]
        assert len(max_shape) == 1 or len(max_shape) == 2

        det_cam_visualizer = DetCAMVisualizer(
            GradCAM,
            model,
            target_layers,
            reshape_transform=partial(
                reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
            is_need_grad=is_need_grad,
            extra_params=extra_params)

    for i, img_info in enumerate(ann_data['images']):
        if i % 10 == 0:
            print('picture----------------', i + 1)
        img_id = img_info['id']
        img_file_name = img_info['file_name']
        img_path = img_dir + img_file_name
        img_keep_path = keep_img_dir_path + img_file_name
        # get gt info
        image = cv2.imread(img_path)
        model.set_input_data(image)
        result = model()[0]
        # results, data = inference_detector(model, img_path)
        numElement = result['bboxes'].numel()
        # inference result in clean samples as gt
        if numElement > 0:
            gt_bboxes = result['bboxes'].clone().detach()
            gt_bboxes_list = [gt_bboxes[..., :-1]]
            gt_labels = result['labels'].clone().detach()
            gt_labels_list = [gt_labels]
            # gt_logits = results[2][idx].clone().detach()
            bboxes = result['bboxes'][..., :4].detach().cpu().numpy()
            scores = result['bboxes'][..., 4].detach().cpu().numpy()
            labels = result['labels'].detach().cpu().numpy()
            segms = result['segms']
            assert bboxes is not None and len(bboxes) > 0
            if args.topk > 0:
                idxs = np.argsort(-scores)
                bboxes = bboxes[idxs[:args.topk]]
                labels = labels[idxs[:args.topk]]
                if segms is not None:
                    segms = segms[idxs[:args.topk]]
            if attack_mode.startswith('My'):
                targets = [
                    DetBoxScoreTarget(bboxes=bboxes, labels=labels, segms=segms)
                ]
            if attack_mode == 'MyAttack_Vanishing':
                adv_images = MyAttack_Vanishing(model, det_cam_visualizer, targets, gt_bboxes_list, gt_labels_list, bboxes, labels, image)
            elif attack_mode == "VanishingAttack":
                adv_images = VanishingAttack(model, gt_bboxes_list, gt_labels_list, bboxes, labels, image)
            elif attack_mode == "MyAttack_Untargeted":
                adv_images = MyAttack_Untargeted(model, det_cam_visualizer, targets, gt_bboxes_list, gt_labels_list, bboxes, labels, image)
            elif attack_mode == "UntargetedAttack":
                adv_images = UntargetedAttack(model, gt_bboxes_list, gt_labels_list, bboxes, labels, image)
            cv2.imwrite(img_keep_path, adv_images.detach().squeeze().permute(1, 2, 0).cpu().numpy()[...,::-1])
            height = model.input_data['img_metas'][0]['img_shape'][0]
            width = model.input_data['img_metas'][0]['img_shape'][1]
            # change images size and also to change bboxes size
            scale_h = height / img_info['height']
            scale_w = width / img_info['width']
            if gt_box_idx.get(img_id):
                for i in gt_box_idx[img_id]:
                    ann_data['annotations'][i]['bbox'][0] = ann_data['annotations'][i]['bbox'][0] * scale_w
                    ann_data['annotations'][i]['bbox'][2] = ann_data['annotations'][i]['bbox'][2] * scale_w
                    ann_data['annotations'][i]['bbox'][1] = ann_data['annotations'][i]['bbox'][1] * scale_h
                    ann_data['annotations'][i]['bbox'][3] = ann_data['annotations'][i]['bbox'][3] * scale_h
                img_info['height'] = height
                img_info['width'] = width
        else:
            shutil.copy(img_path, img_keep_path)

    f = open(keep_ann_path, 'w')
    json.dump(ann_data, f)
    f.close()



def parse_args():
    parser = argparse.ArgumentParser(description='generate attack on images')
    parser.add_argument('attack_mode', help='which attack to choose')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--mode', type=str, default='ll', help='which targeted attack to choose')
    parser.add_argument('--device', type=str, default='cuda:0', help='which cuda to choose')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument(
        '--target-layers',
        default=['backbone.layer3'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the backbone.layer3')
    parser.add_argument(
        '--max-shape',
        nargs='+',
        type=int,
        default=25,
        help='max shapes. Its purpose is to save GPU memory. '
        'The activation map is scaled and then evaluated. '
        'If set to -1, it means no scaling.')
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='Topk of the predicted result to visualizer')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    attack(args)

