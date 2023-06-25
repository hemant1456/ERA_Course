from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn as nn
import torch.nn.functional as F

class Model_batch_norm(nn.Module):
  def __init__(self):
    super(Model_batch_norm,self).__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(3,16,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16)
        ) #32
    self.convblock2 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #32
    self.convblock3 = nn.Sequential(
        nn.Conv2d(32,16,1,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16)
        ) #32
    self.pool1 = nn.MaxPool2d(2,2) #16

    self.convblock4 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #16
    self.convblock5 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #16
    self.convblock6 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #16
    self.convblock7 = nn.Sequential(
        nn.Conv2d(32,16,1,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16)
        ) #16

    self.pool2= nn.MaxPool2d(2,2) #8

    self.convblock8 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #6
    self.convblock9 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #4

    self.convblock10 = nn.Sequential(
        nn.Conv2d(32,64,3,bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64)
        ) #2

    self.gap = nn.Sequential(
        nn.AdaptiveAvgPool2d(1)
        ) #1

    self.convblock11 = nn.Sequential(
        nn.Conv2d(64,10,1,bias=False),
        ) #1
  def forward(self,x):
    x= self.pool1(self.convblock3(self.convblock2(self.convblock1(x))))
    x= self.pool2(self.convblock7(self.convblock6(self.convblock5(self.convblock4(x)))))
    x= self.convblock10(self.convblock9(self.convblock8(x)))
    x= self.gap(x)
    x=self.convblock11(x)
    x=x.view(-1,10)
    return F.log_softmax(x)
  

from torch.nn.modules.normalization import GroupNorm
class Model_group_norm(nn.Module):
  def __init__(self):
    super(Model_group_norm,self).__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(3,16,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,16)
        ) #32
    self.convblock2 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,32)
        ) #32
    self.convblock3 = nn.Sequential(
        nn.Conv2d(32,16,1,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,16)
        ) #32
    self.pool1 = nn.MaxPool2d(2,2) #16

    self.convblock4 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,32)
        ) #16
    self.convblock5 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,32)
        ) #16
    self.convblock6 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,32)
        ) #16
    self.convblock7 = nn.Sequential(
        nn.Conv2d(32,16,1,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(4,16)
        ) #16

    self.pool2= nn.MaxPool2d(2,2) #8

    self.convblock8 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False),
        nn.ReLU(),
        GroupNorm(4,32)
        ) #6
    self.convblock9 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False),
        nn.ReLU(),
        GroupNorm(4,32)
        ) #4

    self.convblock10 = nn.Sequential(
        nn.Conv2d(32,64,3,bias=False),
        nn.ReLU(),
        GroupNorm(4,64)
        ) #2

    self.gap = nn.Sequential(
        nn.AdaptiveAvgPool2d(1)
        ) #1

    self.convblock11 = nn.Sequential(
        nn.Conv2d(64,10,1,bias=False),
        ) #1
  def forward(self,x):
    x= self.pool1(self.convblock3(self.convblock2(self.convblock1(x))))
    x= self.pool2(self.convblock7(self.convblock6(self.convblock5(self.convblock4(x)))))
    x= self.convblock10(self.convblock9(self.convblock8(x)))
    x= self.gap(x)
    x=self.convblock11(x)
    x=x.view(-1,10)
    return F.log_softmax(x)
  


from torch.nn.modules.normalization import GroupNorm
class Model_layer_norm(nn.Module):
  def __init__(self):
    super(Model_layer_norm,self).__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(3,16,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,16)
        ) #32
    self.convblock2 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,32)
        ) #32
    self.convblock3 = nn.Sequential(
        nn.Conv2d(32,16,1,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,16)
        ) #32
    self.pool1 = nn.MaxPool2d(2,2) #16

    self.convblock4 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,32)
        ) #16
    self.convblock5 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,32)
        ) #16
    self.convblock6 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,32)
        ) #16
    self.convblock7 = nn.Sequential(
        nn.Conv2d(32,16,1,bias=False,padding=1),
        nn.ReLU(),
        GroupNorm(1,16)
        ) #16

    self.pool2= nn.MaxPool2d(2,2) #8

    self.convblock8 = nn.Sequential(
        nn.Conv2d(16,32,3,bias=False),
        nn.ReLU(),
        GroupNorm(1,32)
        ) #6
    self.convblock9 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False),
        nn.ReLU(),
        GroupNorm(1,32)
        ) #4

    self.convblock10 = nn.Sequential(
        nn.Conv2d(32,64,3,bias=False),
        nn.ReLU(),
        GroupNorm(1,64)
        ) #2

    self.gap = nn.Sequential(
        nn.AdaptiveAvgPool2d(1)
        ) #1

    self.convblock11 = nn.Sequential(
        nn.Conv2d(64,10,1,bias=False),
        ) #1
  def forward(self,x):
    x= self.pool1(self.convblock3(self.convblock2(self.convblock1(x))))
    x= self.pool2(self.convblock7(self.convblock6(self.convblock5(self.convblock4(x)))))
    x= self.convblock10(self.convblock9(self.convblock8(x)))
    x= self.gap(x)
    x=self.convblock11(x)
    x=x.view(-1,10)
    return F.log_softmax(x)

