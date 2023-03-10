from re import L
from this import d
import torch
from normalize import normalize
import copy


import torch.nn as nn

def L1_norm_loss(amap):
  L1_loss = nn.L1Loss(reduction="mean")
  zero_map = torch.zeros_like(amap)
  return L1_loss(amap, zero_map) 


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

def MyAttack_Targeted(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_logits, gt_labels, cam_extractor, eps=8, alpha=2, mode='ll'):
    target_attack_labels = [generate_attack_targets(gt_logits, mode, 0.3)]
    for _ in range(10):
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]
        # attention loss
        att_results = model(return_loss=False, rescale=True, **data)[0]
        att_idx = att_results[0][:, 4] > 0.3
        # choose idx
        target_labels = []
        victim_labels = []
        if att_idx.any():
            for i, label in enumerate(att_results[1][att_idx]):
                if (label == target_attack_labels[0]).any():
                    target_labels.append(i)
                if (label == gt_labels).any():
                    victim_labels.append(i)
            target_labels = torch.tensor(target_labels)
            victim_labels = torch.tensor(victim_labels)
            activation_map_target = []  
            for cls, logit in zip(att_results[1][target_labels], att_results[2][target_labels]):
                activation_map_target.append(L1_norm_loss(cam_extractor(cls.item(), logit.unsqueeze(0))[0])) 
            att_loss_target = sum(activation_map_target)
            activation_map_victim = []
            for cls, logit in zip(att_results[1][victim_labels], att_results[2][victim_labels]):
                activation_map_victim.append(L1_norm_loss(cam_extractor(cls.item(), logit.unsqueeze(0))[0])) 
            att_loss_victim = sum(activation_map_victim)
            att_loss = att_loss_victim - att_loss_target
        else:
            att_loss = adv_images.new_zeros(1)
        # Targeted Attacks
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=target_attack_labels)
        loss_gfl = losses.pop('loss_dfl', None)
        loss_bbox = losses.pop('loss_bbox', None)
        loss, log_vars = model._parse_losses(losses)
        loss = loss + att_loss
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()   
    return adv_images    

def UntargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps=8, alpha=2):
    for _ in range(10):
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
    # generate_bg_labels
    bg_labels_list = []
    for labels in gt_labels_list:
        bg_labels_list.append(torch.full_like(labels, 80))
    for _ in range(10):
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]
        # Untargeted Attacks
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=bg_labels_list)
        loss, log_vars = model._parse_losses(losses)
        # for key, val in log_vars.items():
        #     print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images

def MyAttack_Vanishing(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, cam_extractor, eps=8, alpha=2):
    # generate_bg_labels        
    bg_labels_list = []
    for labels in gt_labels_list:
        bg_labels_list.append(torch.full_like(labels, 80))
    for _ in range(10):
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
                activation_map.append(L1_norm_loss(cam_extractor(cls.item(), logit.unsqueeze(0))[0])) 
            att_loss = sum(activation_map)
        else:
            att_loss = adv_images.new_zeros(1)
        # inference and get loss
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=bg_labels_list)
        loss_gfl = losses.pop('loss_dfl', None)
        loss_bbox = losses.pop('loss_bbox', None)
        loss, log_vars = model._parse_losses(losses)
        # loss_att = {'loss_att':att_loss}
        # log_vars.update(loss_att)
        loss = loss + att_loss
        # log_vars.update({'loss':loss})
        # for key, val in log_vars.items():
        #     print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images


def MyAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, cam_extractor, eps, alpha):
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
            att_loss = sum(L1_norm_loss(single_cls_map[0]) for single_cls_map in activation_map)
        else:
            att_loss = adv_images.new_zeros(1)
        # inference and get loss
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list)
        loss_cls = losses.pop('loss_cls', None)
        loss_dfl = losses.pop('loss_dfl', None)
        loss_cls = dict(loss_cls = loss_cls)
        #losses.pop('loss_dfl', None)
        loss, log_vars = model._parse_losses(losses)
        loss_cls, log_cls = model._parse_losses(loss_cls)
        loss = loss + loss_cls - att_loss.to(loss.device) 
        # log_vars.update(log_cls)
        # log_vars.update({'att_loss':att_loss})
        # log_vars.update({'loss':loss})
        # for key, val in log_vars.items():
        #     print('key = ', key, ' val = ', val)
        grad = torch.autograd.grad(loss, adv_images,
                                    retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images
