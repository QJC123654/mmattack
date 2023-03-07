from email import parser
import json
from operator import gt
import cv2
from init_inference import inference_detector
from load_model import load_model
from normalize import normalize, denormalize
from attack import UntargetedAttack, TargetedAttack, VanishingAttack, MyAttack
import torch
import argparse
import shutil
def attack(attack_mode:str, *args, **kwargs):
    assert attack_mode in ('UntargetedAttack', 'TargetedAttack', 'VanishingAttack', 'MyAttack'), print('attack do not support!!')
    # load model
    model = load_model()
    # attack coco2017val 
    ann_file = '../annotations/instances_val2017.json'
    img_dir = '../val2017/'
    ann_data = json.load(open(ann_file))
    name = kwargs['mode'] if attack_mode == 'TargetedAttack' else ''
    keep_img_dir_path = '../' + attack_mode + name + '/'
    keep_ann_path = '../annotations/' + attack_mode + '_val2017.json'
    # key = image_id val = annotations_index (list)
    gt_box_idx = dict()
    for i, annotation in enumerate(ann_data['annotations']):
        key = annotation['image_id']
        val = gt_box_idx.get(key)
        if not val:
            gt_box_idx[key] = [i]
        else:
            gt_box_idx[key].append(i)

    for i, img_info in enumerate(ann_data['images']):
        print('picture', i)
        img_id = img_info['id']
        img_file_name = img_info['file_name']
        img_path = img_dir + img_file_name
        img_keep_path = keep_img_dir_path + img_file_name
        results, data = inference_detector(model, img_path)
        idx = results[0][:, 4] > 0.3
        # inference result in clean samples as gt
        if idx.any():
            gt_bboxes = results[0][idx].clone().detach()
            gt_bboxes_list = [gt_bboxes[..., :-1]]
            gt_labels = results[1][idx].clone().detach()
            gt_labels_list = [gt_labels]
            gt_logits = results[2][idx].clone().detach()
            #add attack
            eps = 8
            alpha = 2
            mean = torch.tensor(data['img_metas'][0][0]['img_norm_cfg']['mean']).to(data['img'][0].device)
            std = torch.tensor(data['img_metas'][0][0]['img_norm_cfg']['std']).to(data['img'][0].device)
            images = data['img'][0]
            images = denormalize(images, mean=mean, std=std)
            adv_images = images.clone().detach()
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, min=0, max=255).detach()
            # choose attack type
            if attack_mode == 'UntargetedAttack':
                adv_images = UntargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps, alpha)
            if attack_mode == 'TargetedAttack':
                adv_images = TargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_logits, eps, alpha, mode=kwargs['mode'])
            if attack_mode == 'VanishingAttack':
                adv_images = VanishingAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps, alpha)
            if attack_mode == 'MyAttack':
                adv_images = MyAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps, alpha)
            cv2.imwrite(img_keep_path, adv_images.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()[...,::-1])
            height = data['img_metas'][0][0]['img_shape'][0]
            width = data['img_metas'][0][0]['img_shape'][1]
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



def parse_args():
    parser = argparse.ArgumentParser(description='generate attack on images')
    parser.add_argument('attack_mode', help='which attack to choose')
    parser.add_argument('--mode', type=str, default='ll', help='which targeted attack to choose')
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    attack(args.attack_mode, mode = args.mode)
main()
