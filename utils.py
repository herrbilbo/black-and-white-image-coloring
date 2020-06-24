import torch
import numpy as np
from torchvision import datasets
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import matplotlib.pyplot as plt

class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        
        if self.transform is not None:
            original_img = self.transform(img)
            original_img = np.asarray(original_img)
            lab_img = rgb2lab(original_img)
            lab_img = (lab_img + 128) / 255
            ab_img = lab_img[:, :, 1:3]
            ab_img = torch.from_numpy(ab_img.transpose((2, 0, 1))).float()
            original_img = rgb2gray(original_img)
            original_img = torch.from_numpy(original_img).unsqueeze(0).float()
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return original_img, ab_img, target


class AverageMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.value = 0 
        self.average = 0
        self.sum = 0
        self.count = 0
    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
        

def to_rgb(grayscale_input, ab_input, save_path, save_name):
    plt.clf()
    
    color_img = torch.cat((grayscale_input, ab_input), 0).numpy()
    color_img = color_img.transpose((1, 2, 0))
    color_img[:, :, 0:1] = color_img[:, :, 0:1] * 100
    color_img[:, :, 1:3] = color_img[:, :, 1:3] * 255 - 128  
    color_img = lab2rgb(color_img.astype(np.float64))
    
    grayscale_img = grayscale_input.squeeze().numpy()
    
    plt.imsave(arr=grayscale_img, fname=f"{save_path['grayscale']}{save_name}", cmap='gray')
    plt.imsave(arr=color_img, fname=f"{save_path['colorized']}{save_name}")