import torch
from normalize import normalize

def generate_attack_targets(detections, mode, confidence_threshold):
    assert mode.lower() in ['ml', 'll'], '`mode` should be one of `ML` or `LL`.'
    pred_logits = detections.clone().detach()
    if mode.lower() == 'll':
        if pred_logits.shape[1] % 10 == 1:  # ignore index 1 if it is referring to background class (SSD and FRCNN)
            pred_logits[:, 0] = float('inf')
        target_class_id = torch.argmin(pred_logits, dim=-1)
    else:
        pred_logits[torch.softmax(pred_logits, dim=-1) > confidence_threshold] = float('-inf')
        if pred_logits.shape[1] % 10 == 1:  # ignore index 1 if it is referring to background class (SSD and FRCNN)
            pred_logits[:, 0] = float('-inf')
        target_class_id = torch.argmax(pred_logits, dim=-1)
    return target_class_id

def TargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_logits, eps=8, alpha=2, mode='ll'):
    target_attack_labels = [generate_attack_targets(gt_logits, mode, 0.3)]
    for _ in range(10):
        print('epoch : ', _)
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]
        # Targeted Attacks
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=target_attack_labels)
        loss, log_vars = model._parse_losses(losses)
        for key, val in log_vars.items():
            print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()   
    return adv_images

def UntargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps=8, alpha=2):
    for _ in range(10):
        print('epoch : ', _)
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]
        # Untargeted Attacks
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list)
        loss, log_vars = model._parse_losses(losses)
        for key, val in log_vars.items():
            print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                    retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images 

def VanishingAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps=8, alpha=2):
    for _ in range(10):
        print('epoch : ', _)
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]
        # generate_bg_labels
        bg_labels_list = []
        for labels in gt_labels_list:
            bg_labels_list.append(torch.full_like(labels, 80))
        # Untargeted Attacks
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=bg_labels_list)
        loss, log_vars = model._parse_losses(losses)
        for key, val in log_vars.items():
            print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images


import os
os.sys.path.append('torch-cam/')
import torch.nn as nn
from torchcam.methods import GradCAM

def L1_norm_loss(amap):
  L1_loss = nn.L1Loss(reduction="mean")
  zero_map = torch.zeros_like(amap)
  return L1_loss(amap, zero_map) 

def MyAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps, alpha):

    cam_extractor = GradCAM(model, ['neck.fpn_convs.0.conv', 'neck.fpn_convs.1.conv', 'neck.fpn_convs.2.conv',
        'neck.fpn_convs.3.conv', 'neck.fpn_convs.4.conv'])

    for _ in range(10):
        print('epoch : ', _)
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]

        # attention loss
        att_results = model(return_loss=False, rescale=True, **data)[0]
        att_idx = att_results[0][:, 4] > 0.3
        if att_idx.any():
            activation_map = []
            for cls, logit in zip(att_results[1][att_idx], att_results[2][att_idx]):
                activation_map.append(cam_extractor(cls.item(), logit.unsqueeze(0)))
            
            att_loss = sum(L1_norm_loss(single_level_map) for single_level_map in activation_map[0])
        else:
            att_loss = torch.zeros(1)
        # inference and get loss
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list)
        loss_cls = losses.pop('loss_cls', None)
        loss_cls = dict(loss_cls = loss_cls)
        #losses.pop('loss_dfl', None)
        loss, log_vars = model._parse_losses(losses)
        loss_cls, log_cls = model._parse_losses(loss_cls)
        loss = loss + loss_cls - att_loss 
        log_vars.update(log_cls)
        log_vars.update({'att_loss':att_loss})
        log_vars.update({'loss':loss})
        for key, val in log_vars.items():
            print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                    retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images