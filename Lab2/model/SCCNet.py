import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import sys, os
# sys.path.append(os.pardir) 
from Dataloader import *

# reference paper: https://ieeexplore.ieee.org/document/8716937

class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x) #torch.pow(x, 2) 

class SCCNet(nn.Module):
    def __init__( self, numClasses=4, timeSample=438, Nu=6, C=22, Nt=1, dropoutRate=0.5 ):
        super(SCCNet, self).__init__()
        self.timeSample = timeSample
        self.C = C
        
        # CNN#1
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels=Nu, kernel_size=(C, Nt), padding=(0, Nt-1))
        self.bn1 = nn.BatchNorm2d(Nu)
        #C * TimeSample
        
        # CNN#2
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels=20, kernel_size=(Nu, 11), padding=(0, 11//2))
        self.bn2 = nn.BatchNorm2d(20)
        self.square = SquareLayer()
        self.dropout = nn.Dropout(dropoutRate)
        #20 * TimeSample
        
        # Pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
        #20 * T/12 

        # fully connected（softmax分類）
        self.fc = nn.Linear((math.floor((438 - 62)/12) + 1 ) * 20, numClasses)
        #self.fc = nn.Linear(int(20*timeSample/12), numClasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = F.relu(x)
        x = x.permute(0, 2, 1, 3)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.square.forward(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3)

        x = self.pool(x)
        # 展平
        x = x.view(-1 , (math.floor((438 - 62)/12) + 1 ) * 20)
        #x = x.view(int(20*timeSample/12) , -1)
        
        # Fully connected & softmax
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        
        return x


    # 計算池化層之後的特徵尺寸
    def get_size(self, C, N):
        with torch.no_grad():
            x = torch.zeros(1, 1, C, N)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.square(x)
            x = self.dropout(x)
            x = self.pool(x)
            size = x.view(1, -1).size(1)
        return size




