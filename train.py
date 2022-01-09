import torch
import torchvision.ops as ops
import tqdm

from collections import deque

from utils import CLASSES, NUM_CLASSES, dice_score

def train_one_epoch(model, optimizer, train_loader, device):
  model.train()
  dice_losses = deque(maxlen=50)
  for images, targets in tqdm.tqdm(train_loader):
    images = images.to(device)
    targets = targets.to(device)
    res = model(images)['out']
    loss = []
    for i in range(NUM_CLASSES):
      alp = 1 - targets[::,i,::,::].mean()
      loss.append(ops.sigmoid_focal_loss(res[::,i,::,::],targets[::,i,::,::],alpha=alp,reduction='mean'))

    loss = torch.stack(loss).mean()
                  
                  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    dice_losses.append(dice_score(res, targets).mean(dim=0).to('cpu'))
  final = torch.stack(list(dice_losses)).mean(dim=0)
  for i in range(NUM_CLASSES):
    print(f'dice score of class {CLASSES[i]} on this epoch is {final[i].item()}')
      
@torch.no_grad()
def evaluate(model, val_loader, device):
  model.eval()
  batch_losses = []
  for images, targets in tqdm.tqdm(val_loader):
    images = images.to(device)
    #targets = targets.to(device)
    res = model(images)['out'].to('cpu')
    batch_losses.append(dice_score(res, targets).mean(dim=0))
  final = torch.stack(batch_losses).mean(dim=0)
  for i in range(NUM_CLASSES):
    print(f'dice score of class {CLASSES[i]} is {final[i].item()}')
  
      