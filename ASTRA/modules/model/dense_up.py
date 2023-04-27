
# ver due con upsample module instead of conv2d

import torch.nn.functional as F
import pretrainedmodels
import torch
import torch.nn as nn


def double_conv3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        
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


def upsample_dense(in_ch,d):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch*(d**2), 3, padding=1),
        nn.BatchNorm2d(in_ch*(d**2)),
        nn.ReLU(inplace=True),
        nn.PixelShuffle(d)           
            )

class dense_up(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        
        self.pad_up=(1,1,1,1)
        
        
        #updated    
        IncResV2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained=None)#'imagenet')
        
#        self.dconv_down0 = double_conv(1, 48)
 #       self.dconv_down0.apply(init_weights)
        
        self.dconv_down1 = double_conv(1, 96)
        self.dconv_down1.apply(init_weights)
        
        
        self.dconv_down2 = double_conv(96,192)
        self.dconv_down2.apply(init_weights)
        
        
        #IncResPretrained
        
        self.dconv_down3 = nn.Sequential(*list(IncResV2.children())[7:9]) 
      
        self.dconv_down4 = nn.Sequential(*list(IncResV2.children())[9:11]) 
        
        self.dconv_down5 =nn.Sequential(*list(IncResV2.children())[11:15])

        #4
#        self.upsample4 = upsample_dense(1536,2)
#        self.upsample4.apply(init_weights)
        self.conv_g4 = double_conv(1536+1088,1088)
        self.conv_g4.apply(init_weights)
        #3
#        self.upsample3 = upsample_dense(1088,2)
#        self.upsample3.apply(init_weights)
        self.conv_g3 = double_conv(1088+320,320)
        self.conv_g3.apply(init_weights)
        
        #2
#        self.upsample2 = upsample_dense(320,2)
#        self.upsample2.apply(init_weights)
        self.conv_g2 = double_conv(320+192,192)
        self.conv_g2.apply(init_weights)
        

        #1
#        self.upsample1 = upsample_dense(192,2)
#        self.upsample1.apply(init_weights)
        self.conv_g1 = double_conv(192+96,96)
        self.conv_g1.apply(init_weights)
        

        
        #0
#        self.upsample0 = upsample_dense(96,2)
#        self.upsample0.apply(init_weights)
        #self.conv_g0 = double_conv(96+48,48)
        #self.conv_g0.apply(init_weights)
        
        self.maxpool = nn.MaxPool2d(2)
      
        
        self.conv_last = nn.Conv2d(96, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last.weight)
        
        self.soft2d = nn.Softmax2d()
        
    def forward(self, x):
        tensor_ax = 1
        
         
        
        #encoder std
        
        #conv01 = self.dconv_down0(x) 
        #pool0 = self.maxpool(conv01)

        conv11 = self.dconv_down1(x)
        pool1 = self.maxpool(conv11)
        #print(pool1.size())
        
        conv21 = self.dconv_down2(pool1)
        pool2 = self.maxpool(conv21)
        #print(pool2.size())
        
        #IncResv2
        conv31 = self.dconv_down3(pool2)  
        #print(conv31.size())
        conv41 = self.dconv_down4(F.pad(conv31,self.pad_up, "constant", 0))
        #print(conv41.size())
        conv51 = self.dconv_down5(F.pad(conv41,self.pad_up, "constant", 0))
        
        up4 = up_sampling(conv51) 
        conv4 = self.conv_g4(torch.cat((up4,conv41),dim=tensor_ax))
        
        #3
        up3 = up_sampling(conv4) 
        conv3 = self.conv_g3(torch.cat((up3,conv31),dim=tensor_ax))

        #2
        up2 =up_sampling(conv3)
        conv2 = self.conv_g2(torch.cat((up2,conv21),dim=tensor_ax))
        
               
        #1
        up1 = up_sampling(conv2)
        conv1 = self.conv_g1(torch.cat((up1,conv11),dim=tensor_ax))
        
        #0
        #up0 = up_sampling(conv1)
        #conv0 = self.conv_g0(torch.cat((up0,conv01),dim=tensor_ax))
        
        
        
        out = self.soft2d(self.conv_last(conv1))
        
        
        return out


class dense_up_1(nn.Module):

    def __init__(self, n_class,flag_1cl=False):
        super().__init__()
        self.flag_1cl = flag_1cl
        #updated       
        self.dconv_down0 = double_conv(1, 32)
        self.dconv_down0.apply(init_weights)
        
        self.dconv_down1 = double_conv(32, 64)
        self.dconv_down1.apply(init_weights)
        
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down2.apply(init_weights)
        
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down2.apply(init_weights)
        
        self.dconv_down4 = double_conv(256, 256*n_class) 
        self.dconv_down4.apply(init_weights)

       
        self.soft2d = nn.Softmax2d()
        self.maxpool = nn.MaxPool2d(2)
        self.pixel_reorder = nn.PixelShuffle(16)        
       
    def forward(self, x):
        tensor_ax = 1
        b,c,h,w = x.size()
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

        out = self.soft2d(self.pixel_reorder(conv41))        
        
        
       #out = [self.conv_last1(conv02),self.conv_last2(conv03),self.conv_last3(conv04),self.conv_last4(conv05)]
       #for i in range(len(out)):
       #    if self.flag_1cl:
       #        pass
       #        #out[i] = self.sig(out[i])
       #    else:
       #        out[i] =  self.soft2d(out[i])
       #
        return out


class dense_up_ver2(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        
        self.pad_up=(1,1,1,1)
        
        
        #updated    
        IncResV2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained=None)#'imagenet')
        
        self.dconv_down0 = double_conv(1, 48)
        self.dconv_down0.apply(init_weights)
        
        self.dconv_down1 = double_conv(48, 96)
        self.dconv_down1.apply(init_weights)
        
        
        self.dconv_down2 = double_conv(96,192)
        self.dconv_down2.apply(init_weights)
        
        
        #IncResPretrained
        
        self.dconv_down3 = nn.Sequential(*list(IncResV2.children())[7:9]) 
      
        self.dconv_down4 = nn.Sequential(*list(IncResV2.children())[9:11]) 
        
        self.dconv_down5 =nn.Sequential(*list(IncResV2.children())[11:15])

        #4
       # self.upsample4 = upsample_dense(1536,2)
       # self.upsample4.apply(init_weights)
        self.conv_g4 = double_conv(384+1088,1088)
        self.conv_g4.apply(init_weights)
       # #3
       # self.upsample3 = upsample_dense(1088,2)
       # self.upsample3.apply(init_weights)
        self.conv_g3 = double_conv(272+320,320)
        self.conv_g3.apply(init_weights)
       # 
       # #2
       # self.upsample2 = upsample_dense(320,2)
       # self.upsample2.apply(init_weights)
        self.conv_g2 = double_conv(80+192,192)
        self.conv_g2.apply(init_weights)
       # 

       # #1
       # self.upsample1 = upsample_dense(192,2)
       # self.upsample1.apply(init_weights)
        self.conv_g1 = double_conv(48+96,96)
        self.conv_g1.apply(init_weights)
       # 

       # 
       # #0
       # self.upsample0 = upsample_dense(96,2)
       # self.upsample0.apply(init_weights)
        self.conv_g0 = double_conv(24+48,48)
        self.conv_g0.apply(init_weights)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample =  nn.PixelShuffle(2)
        
        self.conv_last = nn.Conv2d(48, n_class,1,padding=0)
        self.conv_last.apply(init_weights) 
        self.soft2d = nn.Softmax2d()
        
    def forward(self, x):
        tensor_ax = 1
        
         
        
        #encoder std
        
        conv01 = self.dconv_down0(x) 
        pool0 = self.maxpool(conv01)

        conv11 = self.dconv_down1(pool0)
        pool1 = self.maxpool(conv11)
        #print(pool1.size())
        
        conv21 = self.dconv_down2(pool1)
        pool2 = self.maxpool(conv21)
        #print(pool2.size())
        
        #IncResv2
        conv31 = self.dconv_down3(pool2)  
        #print(conv31.size())
        conv41 = self.dconv_down4(F.pad(conv31,self.pad_up, "constant", 0))
        #print(conv41.size())
        conv51 = self.dconv_down5(F.pad(conv41,self.pad_up, "constant", 0))
        #print(conv51.size())
        
        #up4 = self.upsample4(conv51) 
        conv4 = torch.cat((self.upsample(conv51),conv41),dim=tensor_ax)
        conv4 = self.conv_g4(conv4)  
        #3
        #up3 = up_sampling(conv4) 
        conv3 = torch.cat((self.upsample(conv4),conv31),dim=tensor_ax)
        conv3 = self.conv_g3(conv3)
        #2
        #up2 =up_sampling(conv3)
        conv2 = torch.cat((self.upsample(conv3),conv21),dim=tensor_ax)
        conv2 =self.conv_g2(conv2) 
               
        #1
        #up1 = up_sampling(conv2)
        conv1 = torch.cat((self.upsample(conv2),conv11),dim=tensor_ax)
        conv1 = self.conv_g1(conv1)
        #0
        #up0 = up_sampling(conv1)
        conv0 = torch.cat((self.upsample(conv1),conv01),dim=tensor_ax)
        conv0 = self.conv_g0(conv0)
        
        
        out = self.soft2d(self.conv_last(conv0))
        
        
        
        return out

