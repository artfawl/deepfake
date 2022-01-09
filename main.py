import pandas as pd
import numpy as np
import torch
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models

import os
import argparse

from utils import dice_score, CelebADataSet
from train import train_one_epoch, evaluate

def get_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--root', type=str, help='path to root folder')
  parser.add_argument('-im', '--images', type=str, help='path to folder with images (without root part)')
  parser.add_argument('-ma', '--masks', type=str, help='path to folder with masks (without root part)')
  parser.add_argument('-a', '--action', type=str, default='train', help='train or test')
  parser.add_argument('-d', '--device', type=str, default='cpu', help='device: cuda/cpu') 
  parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs')
  parser.add_argument('-bs', '--batch', type=int, default=4, help='batch size')
  return parser

def main(args):
  ROOT = args.root
  IMAGE_PATH = ROOT + args.images + '/'
  MASK_PATH = ROOT + args.masks + '/'
  split = pd.read_csv(ROOT+'/'+'train_test_split.csv')
  
  model = models.segmentation.deeplabv3_resnet50(num_classes=18, pretrained_backbone=True)

  device = torch.device(args.device)
  
  transformations= transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                    ])
  print('creating loaders...')
  train_set = CelebADataSet(split, transformations, IMAGE_PATH, MASK_PATH, 'train')
  val_set = CelebADataSet(split, transformations, IMAGE_PATH, MASK_PATH, 'val')
  
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, num_workers=4)
  
  flag = 'n'
  
  if args.action == 'train':
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)

    num_epochs = args.epochs
    print('start training...')
    for epoch in range(num_epochs):
      train_one_epoch(model, optimizer, train_loader, device)
      torch.save(model.state_dict(), ROOT+'/'+'model.pth')
    print('training is over, wanna test? [y/n]')
    flag = input()
  
  if args.action == 'test' or flag == 'y':
    print('start testing')
    model.load_state_dict(torch.load(ROOT+'/'+'model.pth', map_location=torch.device('cpu')))
    model.to(device)
    evaluate(model, val_loader, device)

if __name__ == '__main__':
  parser = get_argparser()
  args = parser.parse_args()
  main(args)