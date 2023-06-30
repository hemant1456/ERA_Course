import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(3,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #32

    self.convblock2 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #32

    self.convblock3 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=2,dilation=2),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #32

    self.pool1 = nn.Conv2d(32,32,3,stride=2,padding=1) #16

    self.convblock4 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #16

    self.convblock5 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=2,dilation=2),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #16
    
    self.convblock6 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #16

    self.pool2 = nn.Conv2d(32,32,3,stride=2,padding=1) #8

    self.convblock7 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #8

    self.convblock8 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,dilation=2,padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #8
    self.convblock9 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ,nn.Dropout(0.05)
        ) #8

    self.pool3 = nn.Conv2d(32,32,3,stride=2,padding=1) #4

    self.convblock10 = nn.Sequential(
        nn.Conv2d(32,32,3,bias=False,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32)
        ) #4
    self.convblock11 = nn.Sequential(
        nn.Conv2d(32,64,3,bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(64)
        ) #4

    self.gap = nn.Sequential(
        nn.AdaptiveAvgPool2d(1)
        ) #1

    self.convblock12 = nn.Sequential(
        nn.Conv2d(64,10,1,bias=False),
        ) #1
  def forward(self,x):
    x= self.convblock1(x)
    x= x+ self.convblock3(x) + self.convblock2(x)
    x= self.pool1(x)

    x= self.convblock4(x)
    x= x+ self.convblock6(x) + self.convblock5(x)
    x= self.pool2(x)

    x= self.convblock7(x)
    x= x+ self.convblock9(x) + self.convblock8(x)
    x= self.pool3(x)

    x= self.convblock10(x)
    x= self.convblock11(x)

    x= self.gap(x)
    x=self.convblock12(x)
    x=x.view(-1,10)
    return F.log_softmax(x)
