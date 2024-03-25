import torch
from normalize import normalize, denormalize
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import build_loss

def GenerateMask(target_size, boxes):
    boxes = boxes.to(torch.int32)
    Mask = torch.ones(target_size)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = Mask * 0
        img[y1:y2,
            x1:x2] = Mask[y1:y2,x1:x2].clone()
        images.append(img)
    images = torch.stack(images, dim=0)
    adv_mask, _ = torch.max(images, dim=0)
    adv_mask = adv_mask.to('cuda')
    return adv_mask
   
def MyAttack_Vanishing(model, det_cam_visualizer, targets, gt_bboxes_list, gt_labels_list, bboxes, labels, image_numpy):
    # clean cam
    MSELoss = nn.MSELoss(reduction='mean')
    adv_gt_bboxes_list = []
    adv_gt_labels_list = []
    # solver
    adv_gt_bboxes_list.append(torch.empty(0, 4))
    adv_gt_labels_list.append(torch.empty(0))
    # init adv_image
    eps = 8
    alpha = 2
    mean = torch.tensor(model.input_data['img_metas'][0][0]['img_norm_cfg']['mean']).to(model.device)
    std = torch.tensor(model.input_data['img_metas'][0][0]['img_norm_cfg']['std']).to(model.device)
    images = model.input_data['img'][0].clone().detach()
    images = denormalize(images, mean=mean, std=std).detach()
    adv_patch = torch.empty_like(images).uniform_(0, 255)
    adv_mask = GenerateMask(adv_patch.shape[-2:], torch.from_numpy(bboxes))

    model.set_return_loss(True)
    model.set_input_data(image_numpy, bboxes=bboxes, labels=labels)
    
    clean_cam = det_cam_visualizer(
        images.clone(),
        targets=targets,
        aug_smooth=None,
        eigen_smooth=None)
    for i in range(10):
        print("iter----", i)
        adv_patch.requires_grad = True
        adv_images = images * (1 - adv_mask) + adv_mask * adv_patch
        model.input_data['img'] = normalize(adv_images, mean, std)
        # get loss
        losses = model.detector.forward_train(model.input_data['img'], model.input_data['img_metas'], adv_gt_bboxes_list, adv_gt_labels_list)
        losses = dict(loss_cls = losses.get('loss_cls', None))
        assert losses.get('loss_cls', None) is not None, 'can not get loss_cls'
        loss_pred, loss_vars = model.detector._parse_losses(losses)

        grayscale_cam = det_cam_visualizer(
            adv_images,
            targets=targets,
            aug_smooth=None,
            eigen_smooth=None)

        loss_cam = MSELoss(clean_cam, grayscale_cam)
        loss = -loss_cam + loss_pred
        print('loss_cam  ', loss_cam.item(), "--loss_pred  ", loss_pred.item())
        grad = torch.autograd.grad(loss, adv_patch,
                                retain_graph=False, create_graph=False)[0]
        adv_patch = adv_patch.detach() - alpha*grad.sign()
        adv_patch = torch.clamp(adv_patch, min=0, max=255).detach() 
    model.set_return_loss(False)
    adv_images = images * (1 - adv_mask) + adv_mask * adv_patch
    return adv_images

def VanishingAttack(model, gt_bboxes_list, gt_labels_list, bboxes, labels, image_numpy):
    adv_gt_bboxes_list = []
    adv_gt_labels_list = []
    # solver
    adv_gt_bboxes_list.append(torch.empty(0, 4))
    adv_gt_labels_list.append(torch.empty(0))
    # init adv_image
    eps = 8
    alpha = 20
    mean = torch.tensor(model.input_data['img_metas'][0][0]['img_norm_cfg']['mean']).to(model.device)
    std = torch.tensor(model.input_data['img_metas'][0][0]['img_norm_cfg']['std']).to(model.device)
    images = model.input_data['img'][0].clone().detach()
    images = denormalize(images, mean=mean, std=std).detach()
    adv_patch = torch.empty_like(images).uniform_(0, 255)
    adv_mask = GenerateMask(adv_patch.shape[-2:], torch.from_numpy(bboxes))

    model.set_return_loss(True)
    model.set_input_data(image_numpy, bboxes=bboxes, labels=labels)
    
    for i in range(10):
        print("iter----", i)
        adv_patch.requires_grad = True
        adv_images = images * (1 - adv_mask) + adv_mask * adv_patch
        model.input_data['img'] = normalize(adv_images, mean, std)
        # get loss
        losses = model.detector.forward_train(model.input_data['img'], model.input_data['img_metas'], adv_gt_bboxes_list, adv_gt_labels_list)
        losses = dict(loss_cls = losses.get('loss_cls', None))
        assert losses.get('loss_cls', None) is not None, 'can not get loss_cls'
        loss_pred, loss_vars = model.detector._parse_losses(losses)

        print("--loss_pred  ", loss_pred.item())
        grad = torch.autograd.grad(loss_pred, adv_patch,
                                retain_graph=False, create_graph=False)[0]
        adv_patch = adv_patch.detach() - alpha*grad.sign()
        adv_patch = torch.clamp(adv_patch, min=0, max=255).detach() 
    model.set_return_loss(False)
    adv_images = images * (1 - adv_mask) + adv_mask * adv_patch
    return adv_images

def UntargetedAttack(model, gt_bboxes_list, gt_labels_list, bboxes, labels, image_numpy):
    # init adv_image
    eps = 8
    alpha = 2
    mean = torch.tensor(model.input_data['img_metas'][0][0]['img_norm_cfg']['mean']).to(model.device)
    std = torch.tensor(model.input_data['img_metas'][0][0]['img_norm_cfg']['std']).to(model.device)

    clean_pred_corners = None
    images = model.input_data['img'][0].clone().detach()
    images = denormalize(images, mean=mean, std=std)
    adv_imagas = images.clone().detach()
    adv_images = adv_imagas + torch.empty_like(adv_imagas).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=255).detach()

    model.set_return_loss(True)
    model.set_input_data(image_numpy, bboxes=bboxes, labels=labels)

    # inference for clean_pred_corners
    losses = model.detector.forward_train(model.input_data['img'], model.input_data['img_metas'], gt_bboxes_list, gt_labels_list)
    clean_pred_corners = losses.pop('pred_corners')
    weights = losses.pop('weights')

    for i in range(10):
        # print("iter----", i)
        adv_images.requires_grad = True
        model.input_data['img'] = normalize(adv_images, mean, std)
        # get loss
        losses = model.detector.forward_train(model.input_data['img'], model.input_data['img_metas'], gt_bboxes_list, gt_labels_list)
        losses.pop('loss_dfl', None)
        adv_pred_corners = losses.pop('pred_corners')
        # adv_ld_losses = adv_ld_loss(adv_pred_corners, clean_pred_corners)
        adv_ld_losses2 = MSELoss(adv_pred_corners, clean_pred_corners, weights)
        # print(losses.keys())
        loss_pred, loss_vars = model.detector._parse_losses(losses)
        loss = loss_pred + 0.15 * adv_ld_losses2 + 0.1 * loss_ld(adv_corners=adv_pred_corners, clean_corners=clean_pred_corners, weight=weights, avg_factor=4.0)
        print('loss1 --- ', loss_pred.item(), '     loss2 -- ', adv_ld_losses2.item())
        # loss = loss_pred + adv_ld_weight_loss(adv_pred_corners, clean_pred_corners, weights, loss_weight=0.1, avg_factor=1.0, T=3)
        # print('loss_pred  ', loss_pred.item(), 'adv_ld_losses  ', adv_ld_losses.item())
        grad = torch.autograd.grad(loss, adv_images,
                                retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach() 
    model.set_return_loss(False)
    return adv_images


def adv_ld_loss(adv_pred_corners, clean_pred_corners):
    assert len(adv_pred_corners) == len(clean_pred_corners), print('kl loss input error')
    n, c = len(adv_pred_corners), adv_pred_corners[0].shape[-1]
    for i in range(n):
        adv_pred_corners[i].view(-1, c)
        clean_pred_corners[i].view(-1, c)
    adv_pred_corners = torch.cat(adv_pred_corners, dim=0)
    clean_pred_corners = torch.cat(clean_pred_corners, dim=0)
    T = 14
    clean_pred_corners = F.softmax(clean_pred_corners / T, dim=-1)
    return F.kl_div(F.log_softmax(adv_pred_corners / T, dim=-1), clean_pred_corners, reduction='mean') * (T * T)

def MSELoss(adv_pred_corners, clean_pred_corners, weights):
    assert len(adv_pred_corners) == len(clean_pred_corners), print('mse loss input error')
    n, c = len(adv_pred_corners), adv_pred_corners[0].shape[-1]
    for i in range(n):
        adv_pred_corners[i].view(-1, c)
        clean_pred_corners[i].view(-1, c)
    adv_pred_corners = torch.cat(adv_pred_corners, dim=0)
    clean_pred_corners = torch.cat(clean_pred_corners, dim=0)
    weights = torch.cat(weights)
    loss = nn.MSELoss(reduction='none')
    losses = loss(adv_pred_corners, clean_pred_corners).mean(-1) * weights
    return losses.mean()

def adv_ld_weight_loss(adv_pred_corners, clean_pred_corners, weights, loss_weight, avg_factor, T):
    assert len(adv_pred_corners) == len(clean_pred_corners), print('kl loss input error')
    n, c = len(adv_pred_corners), adv_pred_corners[0].shape[-1]
    for i in range(n):
        adv_pred_corners[i].view(-1, c)
        clean_pred_corners[i].view(-1, c)
    adv_pred_corners = torch.cat(adv_pred_corners, dim=0)
    clean_pred_corners = torch.cat(clean_pred_corners, dim=0)
    weights = torch.cat(weights)
    clean_pred_corners = F.softmax(clean_pred_corners / T, dim=-1)
    loss = F.kl_div(F.log_softmax(adv_pred_corners / T, dim=-1), clean_pred_corners, reduction='none').mean(-1) * (T * T)
    if weights is not None:
        loss = loss * weights
    eps = torch.finfo(torch.float32).eps
    loss = loss.sum() / (avg_factor + eps)
    return loss * loss_weight
    
def loss_ld(adv_corners, clean_corners, weight, avg_factor):
    loss_cfg = dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.1, T=13)
    loss = build_loss(loss_cfg)
    assert len(adv_corners) == len(clean_corners), print('kl loss input error')
    adv_corners = torch.cat(adv_corners, dim=0)
    clean_corners = torch.cat(clean_corners, dim=0)
    weight = torch.cat(weight)
    return loss(adv_corners, clean_corners, weight, avg_factor)