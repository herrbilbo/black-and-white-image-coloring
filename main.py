import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import time
from torch.utils.tensorboard import SummaryWriter
from model import ColorizationNet
from utils import GrayscaleImageFolder, AverageMeter, to_rgb


def validate(val_loader, model, criterion, epoch):
    model.eval()
    
    losses = AverageMeter()
    
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        if use_gpu: 
            input_gray = input_gray.cuda()
            input_ab = input_ab.cuda()
            target = target.cuda()

        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        
        losses.update(loss.item(), input_gray.size(0))

        if i % 25 == 0:
            print(f'batch: {i}/{len(train_loader)}, loss value: {losses.value}, loss average: {losses.average}') 

    print(f'Finished validation after epoch: {epoch}.')
    return losses.average


def train(train_loader, model, criterion, optimizer, epoch):
    print(f'Starting training epoch: {epoch}')
    model.train()
  
    losses = AverageMeter()
    
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        if use_gpu:
            input_gray = input_gray.cuda()
            input_ab = input_ab.cuda()
            target = target.cuda()

        output_ab = model(input_gray) 
        loss = criterion(output_ab, input_ab) 
        losses.update(loss.item(), input_gray.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            print(f'batch: {i}/{len(train_loader)}, loss value: {losses.value}, loss average: {losses.average}') 

    print(f'Finished training epoch: {epoch}')
    
    return losses.average


if __name__ == "__main__":
    writer = SummaryWriter("output")
    use_gpu = True
    print(use_gpu)
    
    print('Start!')
    
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder('images/train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=1, shuffle=True)

    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    val_imagefolder = GrayscaleImageFolder('images/val' , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=1, shuffle=False)
    
    print('Dataloader --- done!')
    
    adam_param = [(1e-3, 0.0), (1e-4, 0.0), (1e-2, 0.01), (1e-3, 0.01), (1e-4, 0.01)]
    
    for _id, (_lr, _weight_decay) in enumerate(adam_param):
    
        model = ColorizationNet()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=_lr, weight_decay=_weight_decay)
        
        if use_gpu:
            criterion = criterion.cuda()
            model = model.cuda()

        epochs = 35
        
        print("Let's train!")

        for epoch in range(epochs):
            trains_loss = train(train_loader, model, criterion, optimizer, epoch)
            
            with torch.no_grad():
                validate_loss = validate(val_loader, model, criterion, epoch)
     
            writer.add_scalar(f'{_id}_train_loss', trains_loss, epoch)
            writer.add_scalar(f'{_id}_validation_loss', validate_loss, epoch)
            writer.flush()
            
            torch.save(model.state_dict(), f'checkpoints/{_id}_model-epoch-{epoch}-losses-{validate_loss}.pth')
            
    writer.close()
