import torch
from torch.utils import data
from torchvision import transforms

import cv2
import os
import pandas as pd
import numpy as np

CLASSES = [
 'skin',
 'hair',
 'neck',
 'neck_l',
 'r_eye',
 'r_brow',
 'hat',
 'ear_r',
 'l_lip',
 'u_lip',
 'r_ear',
 'mouth',
 'l_ear',
 'cloth',
 'eye_g',
 'l_brow',
 'l_eye',
 'nose'
 ]

NUM_CLASSES = 18


def dice_score(input, target, smooth=1):
  input = torch.sigmoid(input)
  temp = torch.where(input>0.7, 1, 0)
  intersect = torch.sum(temp*target, dim=(2,3))
  union = torch.sum(temp, dim=(2,3)) + torch.sum(target, dim=(2,3))
  score = 2 * (intersect+smooth)/(union+smooth)
  return score

class CelebADataSet(data.Dataset):
  def __init__(self,selected, transforms, image_path, mask_path, split='train'):
    self.image_path = image_path
    self.mask_path = mask_path
    self.selected = selected
    self.masks = os.listdir(mask_path)
    self.split = split
    self.transforms = transforms
    if split == 'train':
      self.images = selected[selected['is_train']==True]['name'].tolist()
    elif split == 'val':
      self.images = selected[selected['is_train']==False]['name'].tolist()

  
  def decode(self, name):
    new_name = name.split('.')[0]
    while len(new_name)<5:
      new_name = '0' + new_name
    return new_name

  def __getitem__(self, idx):
    img_name = self.images[idx]
    mask_pivot = self.decode(img_name)
    mask_names = []
    target = dict()
    for CLASS in CLASSES:
      mask_names.append(mask_pivot + '_' + CLASS + '.png')
    
    #print(mask_names)
    image = cv2.resize(cv2.imread(self.image_path+img_name), (512,512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transforms is not None:
            image = self.transforms(image)
    
    temp_trans = transforms.ToTensor()
    target = None
    for i in range(len(CLASSES)):
      mask = cv2.imread(self.mask_path+mask_names[i],0)
      if not mask is None:
        mask = temp_trans(mask)
      else:
        mask = torch.zeros(1,512,512)
      if target is None:
        target = mask
      else:
        target = torch.cat([target, mask])
        

    return image, target

  def __len__(self):
    return len(self.images)