# importing important libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


train_transforms = transforms.Compose([
    # Apply center crop and resize with a probability of 10%
    transforms.RandomApply([transforms.CenterCrop(25), transforms.Resize(28)], p=0.1),
    # Apply random translation, scaling, rotation with a probability of 10%
    transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.1),
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

# downloading the mnist dataset

train = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)


# creating the dataloader
def get_data_loaders():
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=128, shuffle=True, num_workers=2)
    return train_loader, test_loader
