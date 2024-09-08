# Implement your ResNet34_UNet model here

#assert False, "Not implemented yet!"
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet34_UNet(nn.Module):
    def __init__(self) :
        super(ResNet34_UNet,self).__init__()
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder_conv2 = self.residual_conv(64,64,3)
        self.encoder_conv3 = self.residual_conv(64,128,4)
        self.encoder_conv4 = self.residual_conv(128,256,6)
        self.encoder_conv5 = self.residual_conv(256,512,3)

        self.center_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up_conv1 = nn.ConvTranspose2d(256+512,32, kernel_size=(2,2), stride = 2)
        self.decoder_conv1 = self.two_conv(32,32)
        self.up_conv2 = nn.ConvTranspose2d(32+256,32, kernel_size=(2,2), stride = 2)
        self.decoder_conv2 = self.two_conv(32,32)
        self.up_conv3 = nn.ConvTranspose2d(32+128,32, kernel_size=(2,2), stride = 2)
        self.decoder_conv3 = self.two_conv(32,32)
        self.up_conv4 = nn.ConvTranspose2d(32+64,32, kernel_size=(2,2), stride = 2)
        self.decoder_conv4 = self.two_conv(32,32)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(2,2), stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=(2,2), stride=2),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def two_conv(self, in_channel , out_channel):
        return nn.Sequential(
            nn.Conv2d(out_channel , out_channel ,kernel_size=(3,3) , padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel , out_channel ,kernel_size=(3,3) , padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def residual_conv(self , in_channel , out_channel ,conv_layers):
        conv = []
        conv.append(nn.Conv2d(in_channel , out_channel , kernel_size=(3,3) , stride= 2, padding=1))
        conv.append(nn.BatchNorm2d(out_channel))
        conv.append(nn.ReLU())

        for i in range(2 * conv_layers - 1):
            conv.append(nn.Conv2d(out_channel , out_channel , kernel_size=(3,3), padding=1))
            conv.append(nn.BatchNorm2d(out_channel))
            conv.append(nn.ReLU())
        
        return nn.Sequential(*conv)

    def forward(self , x):
        x = self.encoder_conv1(x)
        x_e1 = self.encoder_conv2(x)
        x_e2 = self.encoder_conv3(x_e1)
        x_e3 = self.encoder_conv4(x_e2)
        x_e4 = self.encoder_conv5(x_e3)
        
        x_center = self.center_conv(x_e4)

        x = torch.cat([x_center, x_e4], dim=1)#串接 因為up convolution讓channel變一半
        x = self.up_conv1(x)
        x = self.decoder_conv1(x)
        x = torch.cat([x, x_e3], dim=1)
        x = self.up_conv2(x)
        x = self.decoder_conv2(x)
        x = torch.cat([x, x_e2], dim=1)
        x = self.up_conv3(x)
        x = self.decoder_conv3(x)
        x = torch.cat([x,x_e1],dim=1)
        x = self.up_conv4(x)
        x = self.decoder_conv4(x)

        return self.output(x)

if __name__ == '__main__':
    model = ResNet34_UNet()
    print(model)
    
    # pseudo input
    x = torch.randn(1, 3, 256, 256)
    y_pred = model(x)
    
    print(x.shape, y_pred.shape)