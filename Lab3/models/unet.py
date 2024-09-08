# Implement your UNet model here

#assert False, "Not implemented yet!"

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self) :
        super(UNet,self).__init__()

        self.encoder_conv1 = self.two_conv(3,64)
        self.encoder_conv2 = self.two_conv(64,128)
        self.encoder_conv3 = self.two_conv(128,256)
        self.encoder_conv4 = self.two_conv(256,512)
        self.encoder_pool = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        self.center_conv = self.two_conv(512,1024)

        self.up_conv1 = nn.ConvTranspose2d(1024,512, kernel_size=(2,2), stride = 2)
        self.decoder_conv1 = self.two_conv(1024,512)
        self.up_conv2 = nn.ConvTranspose2d(512,256, kernel_size=(2,2), stride = 2)
        self.decoder_conv2 = self.two_conv(512,256)
        self.up_conv3 = nn.ConvTranspose2d(256,128, kernel_size=(2,2), stride = 2)
        self.decoder_conv3 = self.two_conv(256,128)
        self.up_conv4 = nn.ConvTranspose2d(128,64, kernel_size=(2,2), stride = 2)
        self.decoder_conv4 = self.two_conv(128,64)

        self.output = nn.Conv2d(64, 1, kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()

    def two_conv(self, in_channel , out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel , out_channel ,kernel_size=(3,3) , padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel , out_channel ,kernel_size=(3,3) , padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, x):
        x_e1 = self.encoder_conv1(x)
        x = self.encoder_pool(x_e1)
        x_e2 = self.encoder_conv2(x)
        x = self.encoder_pool(x_e2)
        x_e3 = self.encoder_conv3(x)
        x = self.encoder_pool(x_e3)
        x_e4 = self.encoder_conv4(x)
        x = self.encoder_pool(x_e4)
        
        x = self.center_conv(x)
        
        x = self.up_conv1(x)
        x = torch.cat([x_e4, x], dim=1) #串接 因為up convolution讓channel變一半
        x = self.decoder_conv1(x)
        x = self.up_conv2(x)
        x = torch.cat([x_e3,x],dim=1)
        x = self.decoder_conv2(x)
        x = self.up_conv3(x)
        x = torch.cat([x_e2,x],dim=1)
        x = self.decoder_conv3(x)
        x = self.up_conv4(x)
        x = torch.cat([x_e1,x],dim=1)
        x = self.decoder_conv4(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x
    
if __name__ == '__main__':
    model = UNet()
    print(model)
    
    # pseudo input
    x = torch.randn(1, 3, 256, 256)
    y_pred = model(x)
    
    print(x.shape, y_pred.shape)