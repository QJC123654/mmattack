import torch
from normalize import normalize
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os

def L1_norm_loss(amap):
  L1_loss = nn.L1Loss(reduction="sum")
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
    # target_attack_labels = [generate_attack_targets(gt_logits, mode, 0.3)]
    adv_attack_labels = [generate_attack_targets(gt_logits, mode, 0.3)]
    target_attack_labels = []
    new_gt_bboxes_list = []
    for gt_bboxes, labels in zip(gt_bboxes_list, adv_attack_labels):
        full_labels = torch.argmax(gt_logits, dim=-1)
        adv_cls_labels = torch.full_like(full_labels, 16)
        is_adv_labels = (adv_cls_labels == full_labels)
        adv_labels = labels[is_adv_labels]
        gt_bboxes = gt_bboxes[is_adv_labels]
        target_attack_labels.append(adv_labels)
        new_gt_bboxes_list.append(gt_bboxes)
        print('target num ----- ', adv_labels.shape)
    # way2
        # full_labels = torch.argmax(gt_logits, dim=-1)
        # adv_cls_labels = torch.full_like(full_labels, 16)
        # is_adv_labels = (adv_cls_labels == full_labels)
        # adv_labels = torch.where(is_adv_labels, labels, full_labels)
        # target_attack_labels.append(adv_labels)
        # new_gt_bboxes_list.append(gt_bboxes)
        
    for _ in range(10):
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        # img_t = data['img'][0]
        # img_metas_t = data['img_metas'][0]
        # Targeted Attacks
        results = model(return_loss=False, resale=False, **data)
        
        loss = 1
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
            target_labels = torch.tensor(target_labels, dtype=torch.long)
            victim_labels = torch.tensor(victim_labels, dtype=torch.long)
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
    for i in range(10):
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        # get loss
        losses = model.forward_train(data['img'][0], data['img_metas'][0], adv_gt_bboxes_list, adv_gt_labels_list)
        losses = dict(loss_cls = losses.get('loss_cls', None))
        assert losses.get('loss_cls', None) is not None, 'can not get loss_cls'
        loss, loss_vars = model._parse_losses(losses)
        print(i + 1, "-------------", loss_vars)
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images 

def VanishingAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, eps=8, alpha=2):
    adv_gt_bboxes_list = []
    adv_gt_labels_list = []
    # solver1
    for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
        victim_labels = torch.full_like(gt_labels, 16)
        is_victim_idx = (gt_labels != victim_labels)
        adv_gt_lables = gt_labels[is_victim_idx]
        adv_gt_bboxes = gt_bboxes[is_victim_idx]
        adv_gt_labels_list.append(adv_gt_lables)
        adv_gt_bboxes_list.append(adv_gt_bboxes)
    # solver2
    # adv_gt_bboxes_list.append(torch.empty(0, 4))
    # adv_gt_labels_list.append(torch.empty(0))
    # solver3
    # for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
    #     victim_labels = torch.full_like(gt_labels, 16)
    #     bg_labels = torch.full_like(gt_labels, 80)
    #     is_victim_idx = (gt_labels != victim_labels)
    #     adv_gt_lables = torch.where(is_victim_idx, gt_labels, bg_labels)
    #     adv_gt_labels_list.append(adv_gt_lables)
    #     adv_gt_bboxes_list.append(gt_bboxes)

    for i in range(10):
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        # get loss
        losses = model.forward_train(data['img'][0], data['img_metas'][0], adv_gt_bboxes_list, adv_gt_labels_list)
        losses = dict(loss_cls = losses.get('loss_cls', None))
        assert losses.get('loss_cls', None) is not None, 'can not get loss_cls'
        loss, loss_vars = model._parse_losses(losses)
        print(i + 1, "-------------", loss_vars)
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images

def MyAttack_Vanishing(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, cam_extractor, eps=8, alpha=2):
    adv_gt_bboxes_list = []
    adv_gt_labels_list = [] 
    # get adv_gt_bboxes_list adv_gt_labels_list
    for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
        victim_labels = torch.full_like(gt_labels, 16)
        bg_labels = torch.full_like(gt_labels, 81)
        is_victim_idx = (gt_labels != victim_labels)
        adv_gt_lables = torch.where(is_victim_idx, gt_labels, bg_labels)
        adv_gt_labels_list.append(adv_gt_lables)
        adv_gt_bboxes_list.append(gt_bboxes)

    folder = '../hot_img/'
    file_name = os.path.basename(data['img_metas'][0][0]['filename'])
    for i in range(10):
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        # get loss
        losses = model.forward_train(data['img'][0], data['img_metas'][0], adv_gt_bboxes_list, adv_gt_labels_list)
        victim_logits = losses.get('victim_logits', None)
        victim_logits = torch.cat(victim_logits)
        victim_score = losses.get('victim_score', None)
        victim_score = torch.cat(victim_score)
        victim_logits = victim_logits * victim_score.view(-1, 1)
        victim_labels = torch.full_like(torch.Tensor(victim_logits.shape[0]), 16).long()
        # num_pos_anchors = victim_logits.shape[0]
        # 可能需要收到socre的监督,暂时先取avg
        
        avg_logits = torch.mean(victim_logits, dim = 0)
        
        avg_labels = avg_logits.new_tensor(16).long()
        activation_map = cam_extractor(avg_labels.item(), avg_logits.unsqueeze(0))
        if i == 0:
            for idx, _ in enumerate(activation_map):
                hot_img_path = os.path.join(folder, str(idx) + "_" + file_name)
                tensor_img = (activation_map[idx].clone().detach().squeeze() * 255).cpu()
                numpy_img = tensor_img.to(torch.uint8).numpy()
                cv_image = cv2.cvtColor(numpy_img, cv2.COLOR_GRAY2BGR)
                color_heatmap = cv2.applyColorMap(cv_image, cv2.COLORMAP_JET)
                cv2.imwrite(hot_img_path, color_heatmap)

        att_loss = sum([L1_norm_loss(single_level_map) for single_level_map in activation_map])
        
        losses = dict(loss_cls = losses.get('loss_cls', None))
        assert losses.get('loss_cls', None) is not None, 'can not get loss_cls'
        loss, loss_vars = model._parse_losses(losses)
        loss += att_loss
        print(i + 1, "-------------", loss_vars,"--------", att_loss.item())
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images


def MyAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_labels_list, cam_extractor, eps, alpha):
    # generate_bg_labels        
    bg_labels_list = []
    for labels in gt_labels_list:
        bg_labels_list.append(torch.full_like(labels, 80))
    for _ in range(10):
        print('epoch------- : ', _)
        adv_images.requires_grad = True
        data['img'][0] = normalize(adv_images, mean, std)
        img_t = data['img'][0]
        img_metas_t = data['img_metas'][0]
        # test
        # att_results = model(return_loss=False, rescale=False, **data)[0]
        # att_idx = att_results[0][:, 4] > 0.3
        # att_loss = adv_images.new_zeros(1, requires_grad=True)
        # if att_idx.any():
        #     activation_map = []
        #     for cls, logit in zip(att_results[1][att_idx], att_results[2][att_idx]):
        #         activation_map.append(cam_extractor(cls.item(), logit.unsqueeze(0)))
        # print('number of activation_map', len(activation_map))
        # inference and get loss
        losses = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=gt_labels_list)
        pos_scores = losses.pop('pos_scores', None)
        pos_labels = losses.pop('pos_labels', None)
        pos_scores = torch.cat(pos_scores)
        pos_labels = torch.cat(pos_labels)
        print('number of pos_samplers-------------',len(pos_labels))
        activation_map = []
        for cls, logit in zip(pos_labels, pos_scores):
            activation_map = cam_extractor(cls.item(), logit.unsqueeze(0))
        
        att_loss = adv_images.new_zeros(1, dtype=torch.float)
        for single_cls_map in activation_map:
            for single_level_map in single_cls_map:
                att_loss = att_loss + L1_norm_loss(single_level_map)
        
        losses2 = model(img=img_t, img_metas=img_metas_t, gt_bboxes=gt_bboxes_list, gt_labels=bg_labels_list)
        loss_bbox = losses2.pop('loss_bbox', None)
        loss_gfl = losses2.pop('loss_dfl', None)
        losses2.pop('pos_scores', None)
        losses2.pop('pos_labels', None)
        losses2.update(dict(att_loss = att_loss))
        #losses.pop('loss_dfl', None)
        loss, log_vars = model._parse_losses(losses2)
        for key, val in log_vars.items():
            print(key, '---------------', val)
        grad = torch.autograd.grad(loss, adv_images,
                                    retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() - alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    return adv_images
    
