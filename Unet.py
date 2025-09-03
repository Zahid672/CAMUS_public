import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    




class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        ### 1st part encoder
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = DoubleConv(1, 64)
        self.down_conv_2 = DoubleConv(64, 128)
        self.down_conv_3 = DoubleConv(128, 256)
        self.down_conv_4 = DoubleConv(256, 512)
        self.down_conv_5 = DoubleConv(512, 1024)
        
    ### 2nd part decoder
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024, 
            out_channels=512, 
            kernel_size=2, 
            stride=2)
        
        self.up_conv_1 = DoubleConv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512, 
            out_channels=256, 
            kernel_size=2, 
            stride=2)

        self.up_conv_2 = DoubleConv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256, 
            out_channels=128, 
            kernel_size=2, 
            stride=2)

        self.up_conv_3 = DoubleConv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64, 
            kernel_size=2, 
            stride=2)

        self.up_conv_4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=4, # number of classes
            kernel_size=1
        )
        
    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image) # we will need x1, x3, x5, x7 in the decoder part. We don't need x9 in the decoder part because there is no maxpooling.
        print("Output after 1st convolution:", x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print("Output after 5th convolution:", x9.size())
        
        #This is decoder part
        x = self.up_trans_1(x9)
        print("Output of 1st up-convolution:", x.size())
        print("Output after 4th convolution:", x7.size())
        x = self.up_conv_1(torch.cat([x, x7], 1))
        print(x.size())
        x = self.up_trans_2(x)
        print("Output of 2nd up-convolution:", x.size())
        x = self.up_conv_2(torch.cat([x, x5], 1))
        print(x.size())
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x3], 1))
        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x, x1], 1))
        print(x.size())
        x = self.out(x)
        return x
    



if __name__ == '__main__':
        image = torch.rand((1, 3, 512, 512))
        model = UNet()
        # print(model(image))
        output = model(image)
        print("Final shape", output.shape)