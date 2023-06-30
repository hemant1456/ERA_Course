import matplotlib.pyplot as plt
from torchvision import datasets
import torch

from albumentations import PadIfNeeded
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations import CenterCrop,HorizontalFlip, VerticalFlip, Rotate, Normalize
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.transforms import Affine

import numpy as np
from albumentations import Compose


class Albumentations:
  def __init__(self, transforms):
    self.transforms= Compose(transforms)
  
  def __call__(self,image):
    img= np.array(image)
    return self.transforms(image=img)['image']


train_transforms= Albumentations([
    PadIfNeeded(min_height=64,min_width=64),
    CoarseDropout(min_height=16,min_width=16, max_height=16, max_width=16, max_holes=1,p=1,fill_value=(0.49139968*255,0.48215841*255,0.44653091*255)),
    CenterCrop(32,32,p=1),
    Affine(scale=(0.5, 2),translate_percent=(0.2, 0.2),rotate=(-45, 45),shear=(-10, 10),interpolation=1,p=0.3,cval=(0.49139968*255,0.48215841*255,0.44653091*255)),
    Normalize((0.49139968,0.48215841,0.44653091),(0.24703223,0.24348513,0.26158784)),
    ToTensorV2()
])

test_transforms= Albumentations([
    Normalize((0.49139968,0.48215841,0.44653091),(0.24703223,0.24348513,0.26158784)),
    ToTensorV2()
])

cifar10_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}


def visualise_transformation():

    train_transforms1= Albumentations([
        PadIfNeeded(min_height=64,min_width=64),
        CoarseDropout(min_height=16,min_width=16, max_height=16, max_width=16, max_holes=1,p=1,fill_value=(0.49139968*255,0.48215841*255,0.44653091*255)),
        CenterCrop(32,32,p=1),
        Affine(scale=(0.5, 2),translate_percent=(0.1, 0.1),rotate=(-20, 20),shear=(-10, 10),interpolation=1,p=0.3,cval=(0.49139968*255,0.48215841*255,0.44653091*255)),
        #Normalize((0.49139968,0.48215841,0.44653091),(0.24703223,0.24348513,0.26158784)),
        ToTensorV2()
    ])

    train_transforms2= Albumentations([
        ToTensorV2()
    ])

    train_data1= datasets.CIFAR10(root= '../data', train= True, download= True, transform= train_transforms1)
    train_data2= datasets.CIFAR10(root= '../data', train= True, download= True, transform= train_transforms2)

    train_loader1= torch.utils.data.DataLoader(train_data1, batch_size=64,shuffle=False)
    train_loader2= torch.utils.data.DataLoader(train_data2, batch_size=64,shuffle=False)

    # get a single batch
    dataiter1 = iter(train_loader1)
    transformed_images, labels1 = next(dataiter1)

    dataiter2 = iter(train_loader2)
    original_images, labels2 = next(dataiter2)

    # plot the images
    plot_images(original_images, transformed_images, labels1)




def plot_images(original_images, transformed_images,labels):
    temp=0
    pointer=0
    plt.figure(figsize=(8,8))
    for i in range(1,17):
        
        plt.subplot(4,4,i)
        plt.axis('off')
        plt.tight_layout()
        if temp==0:
          
          plt.imshow(original_images[pointer].permute(1,2,0))
          plt.title(cifar10_classes[labels[pointer].item()])
          temp=1
        else:
          plt.imshow(transformed_images[pointer].permute(1,2,0))
          temp=0
          pointer+=1

    
    plt.show()


