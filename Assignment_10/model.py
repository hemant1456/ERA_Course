import torch.nn as nn
import torch.nn.functional as F

def convblock(in_channels,out_channels,padding=0,**kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,padding=padding,**kwargs),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
        )

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.PrepLayer = convblock(in_channels=3,out_channels=64,kernel_size=3,padding=1,bias=False) #32
    self.convblock1 = convblock(in_channels=64,out_channels=128,kernel_size=3,padding=1,bias=False) #32

    self.pool1= nn.MaxPool2d(2,2) #16

    self.R1_c1= convblock(in_channels=128,out_channels=128,kernel_size=3,bias=False,padding=1) #16
    self.R1_c2= convblock(in_channels=128,out_channels=128,kernel_size=3,padding=1,bias=False) #16

    self.convblock2 = convblock(in_channels=128,out_channels=256,kernel_size=3,padding=1,bias=False) #16

    self.pool2 = nn.MaxPool2d(2,2) #8

    self.convblock3 = convblock(in_channels=256,out_channels=512,kernel_size=3,padding=1,bias=False) #8
    self.pool3 = nn.MaxPool2d(2,2) #4

    self.R2_c1= convblock(in_channels=512,out_channels=512,kernel_size=3,padding=1,bias=False) #4
    self.R2_c2= convblock(in_channels=512,out_channels=512,kernel_size=3,padding=1,bias=False) #4

    self.pool4 = nn.MaxPool2d(4,4) #1

    self.fc1 = nn.Linear(512,10)

  def forward(self,x):
    x=self.PrepLayer(x)
    #layer1
    x=self.pool1(self.convblock1(x))
    R1 = self.R1_c2(self.R1_c1(x))
    x = x + R1
    #layer2
    x=self.pool2(self.convblock2(x))
    #layer3
    x=self.pool3(self.convblock3(x))
    R2 = self.R2_c2(self.R2_c1(x))
    x = x + R2
    #layer4
    x=self.pool4(x)

    x=x.view(-1,512)
    x= self.fc1(x)

    x=x.view(-1,10)
    return x
