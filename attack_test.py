import torch
from normalize import normalize, denormalize
import torch.nn as nn

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