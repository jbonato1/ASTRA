import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding, 
                              bias=False) 
        nn.init.kaiming_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, 
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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

class IncResV2(nn.Module):

    def __init__(self,n_class):
        super().__init__()
        
        self.pad_up=(1,1,1,1)
        
        
        #updated    
        IncResV2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        
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
        self.conv_g42 = double_conv(1536+1088,1088)
        self.conv_g42.apply(init_weights)
        #3
        self.conv_g32 = double_conv(1088+320,320)
        self.conv_g32.apply(init_weights)
        
        self.conv_g33 = double_conv(1088+320+320,320)
        self.conv_g33.apply(init_weights)
        #2
        self.conv_g22 = double_conv(320+192,192)
        self.conv_g22.apply(init_weights)
        

        self.conv_g23 = double_conv(320+192+192,192)
        self.conv_g23.apply(init_weights)
        
        self.conv_g24 = double_conv(320+192+192+192,192)
        self.conv_g24.apply(init_weights)
        #1
        self.conv_g12 = double_conv(192+96,96)
        self.conv_g12.apply(init_weights)
        

        self.conv_g13 = double_conv(192+96+96,96)
        self.conv_g13.apply(init_weights)
        

        self.conv_g14 = double_conv(192+96+96+96,96)
        self.conv_g14.apply(init_weights)
        
        self.conv_g15 = double_conv(192+96+96+96+96,96)
        self.conv_g15.apply(init_weights)
        #0
        self.conv_g02 = double_conv(96+48,48)
        self.conv_g02.apply(init_weights)
        
        self.conv_g03 = double_conv(96+48+48,48)
        self.conv_g03.apply(init_weights)
        
        self.conv_g04 = double_conv(96+48+48+48,48)
        self.conv_g04.apply(init_weights)
        
        self.conv_g05 = double_conv(96+48+48+48+48,48)
        self.conv_g05.apply(init_weights)
        
        self.conv_g06 = double_conv(96+48+48+48+48+48,48)
        self.conv_g06.apply(init_weights)
        
        self.maxpool = nn.MaxPool2d(2)
        
      
        
        self.conv_last1 = nn.Conv2d(48, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last1.weight)
        self.conv_last2 =nn.Conv2d(48, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last2.weight)
        self.conv_last3 = nn.Conv2d(48, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        self.conv_last4 =nn.Conv2d(48, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last4.weight)
        self.conv_last5 = nn.Conv2d(48, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last5.weight)
        
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
        
        up42 = up_sampling(conv51) 
        conv42 = self.conv_g42(torch.cat((up42,conv41),dim=tensor_ax))
        
        #3
        up32 = up_sampling(conv41) 
        conv32 = self.conv_g32(torch.cat((up32,conv31),dim=tensor_ax))

        up33 = up_sampling(conv42) 
        conv33 = self.conv_g33(torch.cat((up33,conv31,conv32),dim=tensor_ax))
        
        #2
        up22 =up_sampling(conv31)
        conv22 = self.conv_g22(torch.cat((up22,conv21),dim=tensor_ax))
        
        up23 = up_sampling(conv32)
        conv23 = self.conv_g23(torch.cat((up23,conv21,conv22),dim=tensor_ax))
        
        up24 = up_sampling(conv33)
        conv24 = self.conv_g24(torch.cat((up24,conv21,conv22,conv23),dim=tensor_ax))
       
        #1
        up12 = up_sampling(conv21)
        conv12 = self.conv_g12(torch.cat((up12,conv11),dim=tensor_ax))
        
        
        up13 = up_sampling(conv22)
        conv13 = self.conv_g13(torch.cat((up13,conv11,conv12),dim=tensor_ax))

        up14 =up_sampling(conv23)
        conv14 = self.conv_g14(torch.cat((up14,conv11,conv12,conv13),dim=tensor_ax))
        
        up15 = up_sampling(conv24)
        conv15 = self.conv_g15(torch.cat((up15,conv11,conv12,conv13,conv14),dim=tensor_ax))
        
        #0
        up02 = up_sampling(conv11)
        conv02 = self.conv_g02(torch.cat((up02,conv01),dim=tensor_ax))
        
        up03 = up_sampling(conv12)
        conv03 = self.conv_g03(torch.cat((up03,conv01,conv02),dim=tensor_ax))
        
        up04 = up_sampling(conv13)
        conv04 = self.conv_g04(torch.cat((up04,conv01,conv02,conv03),dim=tensor_ax))
        
        up05 = up_sampling(conv14)
        conv05 = self.conv_g05(torch.cat((up05,conv01,conv02,conv03,conv04),dim=tensor_ax))
        
        up06 = up_sampling(conv15)
        conv06 = self.conv_g06(torch.cat((up05,conv01,conv02,conv03,conv04,conv05),dim=tensor_ax))

        out = [self.conv_last1(conv02), 
               self.conv_last2(conv03),
               self.conv_last3(conv04),
               self.conv_last4(conv05),
               self.conv_last5(conv06),
              ]
        
        for i in range(len(out)):
            out[i] = self.soft2d(out[i])
        
        
        return out

