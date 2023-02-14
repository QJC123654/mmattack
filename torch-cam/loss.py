# loss
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
os.sys.path.append('/content/mmattack/torch-cam')
from torch.functional import Tensor
# Define your model
# Set your CAM extractor
from torchcam.methods import GradCAM
import torch
import torch.nn.functional as F
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
import mmcv

def klloss(map1, map2):
  # y_pre = F.log_softmax(map1, dim=-1)
  y_pre = F.softmax(map1, dim=-1)
  y_tru = F.softmax(map2, dim=-1)
  criterion = nn.KLDivLoss(reduction="sum")
  klloss = criterion(y_pre.log(), y_tru)
  return klloss

def L1_norm_loss(amap):
  L1_loss = nn.L1Loss(reduction="sum")
  zero_map = torch.zeros_like(amap)
  return L1_loss(amap, zero_map) 

def cross_entropy_loss(logits, labels):
  criterion = nn.CrossEntropyLoss()
  print(logits.shape)
  print(labels.shape)
  if logits.ndim < 2:
    n = 1
  else:
    n = logits.shape[0]
  det_labels = torch.zeros((n, logits.shape[-1]), device=logits.device)
  for i in range(n):
    det_labels[i][labels[i]] = 1 
  logits = F.softmax(logits, dim=-1)
  return criterion(logits, det_labels)
  
def loss(amap, gboxes, gboxes_adv, pred, label, x):
  l1 = L1_norm_loss(amap)
  l2 = klloss(gboxes, gboxes_adv)
  l3 = cross_entropy_loss(pred, label)
  # total = l2 + 1e-2 * l3
  total = l2 + x * l3
  print('l1 = ', l1)
  print('l2 = ', l2)
  print('l3 = ', l3)
  print('totoal = ', total)
  return total



# cam_extractor = GradCAM(model, ['neck.fpn_convs.3.conv'])
# cam_extractor = GradCAM(model, ['neck.fpn_convs.2.conv'])


def total_loss(model, data, gboxes_old, x):
  cam_extractor = GradCAM(model, ['neck.fpn_convs.3.conv'])

  results = model(return_loss=False, rescale=True, **data)[0]
  idx = results[0][:, 4] > 0.3
  # #dets and labels
  # print(results[0][idx])
  # print(results[1][idx])
  # get amap and gboxess
  activation_map = []
  for cls, logit in zip(results[1][idx], results[2][idx]):
    activation_map.append(cam_extractor(cls.item(), logit.unsqueeze(0)))
  
  gboxes = F.softmax(results[3][idx].reshape(-1, 17), dim=1).reshape(-1, 17*4)
  #gboxes = F.linear(gboxes, torch.linspace(0, 16, 17).type_as(gboxes)).reshape(-1, 4)

  #logits, labels
  det_logits = results[2][idx]
  det_labels = results[1][idx]
  print(det_labels)
  att_label = det_labels[det_labels == 15] 
  att_logits = det_logits[det_labels == 15]
  print(att_label.shape)
  print(att_logits.shape)
  if not (det_labels == 15).any():
    print('attack success')
    print(att_label)
    print(det_labels)
    return
  # cal loss
  loss_total = loss(activation_map[0][0], gboxes_old[0], gboxes[0], att_logits, att_label, x)
  return loss_total