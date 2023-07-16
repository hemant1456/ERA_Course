import matplotlib.pyplot as plt
from torchvision import datasets
import torch

from albumentations import PadIfNeeded
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations import Compose, Normalize, HorizontalFlip, PadIfNeeded, RandomCrop, CoarseDropout
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

from albumentations import Lambda

train_transforms = Albumentations([
    PadIfNeeded(min_height=40, min_width=40),  # Padding
    RandomCrop(32, 32),  # RandomCrop after padding
    HorizontalFlip(p=0.5),  # FlipLR
    CoarseDropout(max_holes=1, max_height=8, max_width=8, min_height=8, min_width=8, fill_value=(0.4914*255, 0.4822*255, 0.4465*255), p=0.5),  # Cutout
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalizing
    ToTensorV2()  # Convert to tensor
])

test_transforms= Albumentations([
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalizing
    ToTensorV2()  # Convert to tensor
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

    train_transforms1 = Albumentations([
    PadIfNeeded(min_height=40, min_width=40),  # Padding
    RandomCrop(32, 32),  # RandomCrop after padding
    HorizontalFlip(p=0.5),  # FlipLR
    CoarseDropout(max_holes=1, max_height=8, max_width=8, min_height=8, min_width=8, fill_value=(0.4914*255, 0.4822*255, 0.4465*255), p=0.5),  # Cutout
    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalizing
    ToTensorV2()  # Convert to tensor
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


def unnormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):  # for each channel
        t.mul_(s).add_(m)  # unnormalize
    return img

def plot_images(original_images, transformed_images,labels):
    temp=0
    pointer=0
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
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
          img = unnormalize(transformed_images[pointer], mean, std)
          plt.imshow(img.permute(1,2,0))
          temp=0
          pointer+=1

    
    plt.show()


def test_and_find_misclassified(model, dataloader,device):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            # find the indices of the misclassified images
            misclassified_indices = torch.where(predictions != labels)[0]
            
            # store the misclassified images, true labels and predictions
            for index in misclassified_indices:
               misclassified_images.append(inputs[index].cpu())
               misclassified_labels.append(labels[index].cpu())
               misclassified_preds.append(predictions[index].cpu())
            
    return misclassified_images, misclassified_labels, misclassified_preds

def display_misclassified_images(images, labels, preds, title):
    plt.figure(figsize=(10, 10))
    for i in range(12):
        plt.subplot(4, 3, i+1)
        image = images[i].numpy().transpose((1, 2, 0))  # adjust this if your image is not 3-channel RGB
        image = (image - image.min()) / (image.max() - image.min())  # normalise to [0,1]
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Actual: {cifar10_classes[labels[i].item()]}, Predicted: {cifar10_classes[preds[i].item()]}')
    plt.suptitle(title)
    plt.show()

