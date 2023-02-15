from load_model import load_model
from init_inference import init_inference, inference_detector2
from normalize import normalize, denormalize
from loss import total_loss

import torch

from mmdet.core.bbox.iou_calculators import bbox_overlaps

g_min_iou = 1
keep_x = 0
# init model
model = load_model()
# img path
img = './dog_cat.jpg'
results, gboxes, data = init_inference(model, img)
#get origin xyxy
l, r, u, d = 150.5, 710.5, 418.6, 794.5
bboxes1 = torch.FloatTensor([[l, u, r, d]])

#test black model
config = 'configs/yolo/yolov3_d53_mstrain-608_273e_coco.py'
checkpoint = 'checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth'
model2 = load_model(config, checkpoint)

#add attack
eps = 8
alpha = 2
mean = torch.tensor(data['img_metas'][0][0]['img_norm_cfg']['mean']).to(data['img'][0].device)
std = torch.tensor(data['img_metas'][0][0]['img_norm_cfg']['std']).to(data['img'][0].device)


for x in range(1000000, 0, -10):
    att_data = data.copy()
    images = att_data['img'][0]
    images = denormalize(images, mean=mean, std=std)
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=255).detach()

    x = x / 1e6
    for _ in range(20):
        print('epoch:  ', _)
        adv_images.requires_grad = True
        att_data['img'][0] = normalize(adv_images, mean, std)
        cost = total_loss(model, att_data, gboxes, x)
        if not cost:
            break
        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                    retain_graph=False, create_graph=False)[0]
        # print(grad)
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=255).detach()
    # test
    test_img = adv_images.clone().detach()
    test_img = normalize(test_img, mean, std)
    test_data = data.copy()
    test_data['img'][0] = test_img
    result = inference_detector2(model2, test_data)
    result = result[0]
    idx = result[0][:, 4] > 0.3
    det_labels = result[1][idx]
    if not (det_labels == 15).any():
        with open('./log.txt', 'a+') as f:
            f.write(str(x) + '没检测到\n')
        f.close()
        continue
    else:
        # cal iou
        det_boxes = result[0][idx]
        att_boxes = det_boxes[det_labels == 15]
        max_iou = 0
        bboxes2 = att_boxes[..., 0:4]
        overlaps = bbox_overlaps(bboxes1.to(bboxes2.device), bboxes2)
        max_iou = overlaps.max(dim = -1)[0].item()
        print('max_iou:', max_iou)
        if max_iou < 0.5:
            with open('./log.txt', 'a+') as f:
                f.write(str(x) + ' 检测不准的iou为 ' + str(max_iou) + '\n')
            f.close()
        if max_iou < g_min_iou:
            g_min_iou = max_iou
            keep_x = x

with open('./log.txt', 'a+') as f:
    f.write('最小iou为:'+ str(g_min_iou) + 'x为' + str(x) + '\n')
f.close()

                
                            