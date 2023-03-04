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

def TargetedAttack(model, images, adv_images, data, mean, std, gt_bboxes_list, gt_logits, eps=8, alpha=2):
    target_attack_labels = [generate_attack_targets(gt_logits, 'll', 0.3)]
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
    