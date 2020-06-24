import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
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
        
        for j in range(len(output_ab)):
            save_path = {'grayscale': 'test_imgs/gray/', 'colorized': 'test_imgs/color/'}
            save_name = f'img-{i * val_loader.batch_size + j}.jpg'
            to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

        if i % 25 == 0:
            print(f'batch: {i}/{len(val_loader)}, loss value: {losses.value}, loss average: {losses.average}') 

    print(f'Finished validation after epoch: {epoch}.')
    return losses.average


if __name__ == "__main__":
    use_gpu = True
    print(use_gpu)
    
    print('Start!')

    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    test_imagefolder = GrayscaleImageFolder('images/test' , test_transforms)
    test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=False)
    
    print('Dataloader --- done!')
    
    model = ColorizationNet()
    pretrained = torch.load('worst.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained)
    criterion = nn.MSELoss()
    
    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    print("Let's test!")

    with torch.no_grad():
        validate_loss = validate(test_loader, model, criterion, 0)
        
    print(f'average loss: {validate_loss}')
