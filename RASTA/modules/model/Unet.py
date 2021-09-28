import torch
import torch.nn as nn
def double_conv5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 5, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        
        nn.Conv2d(out_channels, out_channels, 5, padding=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
    )   
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
    )   
def init_weights(m):
    if type(m)== nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)
        
def up_sampling(input_tensor):
    return nn.functional.interpolate(input_tensor,scale_factor=2, mode='bilinear', align_corners=True)   

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        #updated       
        self.dconv_down0 = double_conv(1, 32)
        self.dconv_down0.apply(init_weights)
        
        self.dconv_down1 = double_conv(32, 64)
        self.dconv_down1.apply(init_weights)
        
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down2.apply(init_weights)
        
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down2.apply(init_weights)
        
        self.dconv_down4 = double_conv(256, 512) 
        self.dconv_down4.apply(init_weights)


        #3
        self.conv_g3 = double_conv(512+256,256)
        self.conv_g3.apply(init_weights)
        
        #2
        self.conv_g2 = double_conv(256+128,128)
        self.conv_g2.apply(init_weights)
        
        #1
        self.conv_g1 = double_conv(128+64,64)
        self.conv_g1.apply(init_weights)
         
        #0
        self.conv_g0 = double_conv(64+32,32)
        self.conv_g0.apply(init_weights)
        
        self.soft2d = nn.Softmax2d()
        self.sig = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv_last = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last.weight)
        
    def forward(self, x):
        tensor_ax = 1
        #encoder std
        conv01 = self.dconv_down0(x)
        #print(conv01.shape)
        pool0 = self.maxpool(conv01)
        
        conv11 = self.dconv_down1(pool0)
        #print(conv11.shape)
        pool1 = self.maxpool(conv11)

        conv21 = self.dconv_down2(pool1)
        #print(conv21.shape)
        pool2 = self.maxpool(conv21)
        
        conv31 = self.dconv_down3(pool2)
        #print(conv31.shape)
        pool3 = self.maxpool(conv31)   
        
        conv41 = self.dconv_down4(pool3)
        #print(conv41.shape)

        #UPSAMPLING PART 
        up3 = up_sampling(conv41)
        conv3 = self.conv_g3(torch.cat((up3,conv31),dim=tensor_ax))

        up2 =up_sampling(conv3)
        conv2 = self.conv_g2(torch.cat((up2,conv21),dim=tensor_ax))


        up1 = up_sampling(conv2)
        conv1 = self.conv_g1(torch.cat((up1,conv11),dim=tensor_ax))

        
        up0 = up_sampling(conv1)
        conv0 = self.conv_g0(torch.cat((up0,conv01),dim=tensor_ax))
        
        
        out = self.soft2d(self.conv_last(conv0))
        
        return out

