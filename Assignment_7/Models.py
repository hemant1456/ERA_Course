import torch.nn as nn
import torch.nn.functional as F


class Model1(nn.Module):
    def __init__(self):
        super(Model1,self).__init__()
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=4,kernel_size=(3,3),padding=1,bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(4),
                                        nn.Dropout(0.1)) #28 rf=3
        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=4,out_channels=8,kernel_size=(3,3),bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(8),
                                        nn.Dropout(0.1)) #26 rf=5
        self.pool1= nn.MaxPool2d(2,2) #13
        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout(0.1)) #11 rf=6
        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(16),
                                        nn.Dropout(0.1)) #11 rf=10
        self.pool2= nn.MaxPool2d(2,2) #5 , rf=12
        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),bias=False),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(32),
                                        nn.Dropout(0.1)) #3 rf=20
        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=10,kernel_size=(3,3),bias=False)) #1 rf=28

    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = x.view(-1,10)
        return F.log_softmax(x,dim=-1)


