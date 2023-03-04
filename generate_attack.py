import json
import cv2
from init_inference import inference_detector
from load_model import load_model
from normalize import normalize, denormalize
from attack import UntargetedAttack, TargetedAttack, VanishingAttack
import torch

# load model
model = load_model()
# attack coco2017val 
ann_file = '../annotations/instances_val2017.json'
img_root_dir = '../val2017/'
f = open(ann_file)
ann_data = json.load(f)
for img_info in ann_data['images']:
    img_file_name = img_info['file_name']
    img_path = img_root_dir + img_file_name
    results, data = inference_detector(model, img_path)
    idx = results[0][:, 4] > 0.3
    # inference result in clean samples as gt
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
    adv_images = UntargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps, alpha)
    cv2.imwrite(img_path, adv_images.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()[...,::-1])
    img_info['height'] = data['img_metas'][0][0]['img_shape'][0]
    img_info['width'] = data['img_metas'][0][0]['img_shape'][1]
json.dump(ann_data, f)
