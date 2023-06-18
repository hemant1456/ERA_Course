# importing important libraries
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# defining the transformation to be applied on the images
train_transforms = transforms.Compose([
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
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test, batch_size=64, shuffle=True, num_workers=2)
    return train_loader, test_loader
