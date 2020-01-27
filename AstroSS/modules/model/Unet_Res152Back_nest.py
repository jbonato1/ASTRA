import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.checkpoint as checkpoint

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

class nestedUNetUp152(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        #updated       
        ResNet152 = models.resnet152(pretrained=True)
        
        self.dconv_down00 = double_conv(1, 32)
        self.dconv_down00.apply(init_weights)
        
        self.dconv_down0 = double_conv(32, 64)
        self.dconv_down0.apply(init_weights)
        
        self.dconv_down1 = nn.Sequential(*list(ResNet152.children())[4])
        
        self.dconv_down2 = nn.Sequential(*list(ResNet152.children())[5]) #double_conv(64, 128)
        #self.dconv_down2.apply(init_weights)
        
        self.dconv_down3 = nn.Sequential(*list(ResNet152.children())[6]) #double_conv(128, 256)
        #self.dconv_down2.apply(init_weights)
        
        self.dconv_down4 = nn.Sequential(*list(ResNet152.children())[7]) #double_conv(256, 512) 
        #self.dconv_down4.apply(init_weights)


        
        self.conv_g32 = double_conv(2048+1024,1024)
        self.conv_g32.apply(init_weights)
        

        self.conv_g22 = double_conv(1024+512,512)
        self.conv_g22.apply(init_weights)
        

        self.conv_g23 = double_conv(1024+512+512,512)
        self.conv_g23.apply(init_weights)
        

        self.conv_g12 = double_conv(512+256,256)
        self.conv_g12.apply(init_weights)
        

        self.conv_g13 = double_conv(512+256+256,256)
        self.conv_g13.apply(init_weights)
        

        self.conv_g14 = double_conv(512+256+256+256,256)
        self.conv_g14.apply(init_weights)
        
        #0
        self.conv_g02 = double_conv(256+64,64)
        self.conv_g02.apply(init_weights)
        
        self.conv_g03 = double_conv(256+64+64,64)
        self.conv_g03.apply(init_weights)
        
        self.conv_g04 = double_conv(256+64+64+64,64)
        self.conv_g04.apply(init_weights)
        
        self.conv_g05 = double_conv(256+64+64+64+64,64)
        self.conv_g05.apply(init_weights)
        
        #-1
        self.conv_g_12 = double_conv(32+64,32)
        self.conv_g_12.apply(init_weights)
        
        self.conv_g_13 = double_conv(64+32+32,32)
        self.conv_g_13.apply(init_weights)
        
        self.conv_g_14 = double_conv(64+32+32+32,32)
        self.conv_g_14.apply(init_weights)
        
        self.conv_g_15 = double_conv(64+32+32+32+32,32)
        self.conv_g_15.apply(init_weights)
        
        self.conv_g_16 = double_conv(64+32+32+32+32+32,32)
        self.conv_g_16.apply(init_weights)
        
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv_last1 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last1.weight)
        self.conv_last2 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last2.weight)
        self.conv_last3 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        self.conv_last4 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last4.weight)
        self.conv_last5 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last5.weight)
        
        self.soft2d = nn.Softmax2d()
        
    def forward(self, x):
        tensor_ax = 1
        #encoder std
        conv00 = self.dconv_down00(x)
        #print(conv01.size())
        pool00 = self.maxpool(conv00) 
        
        conv01 = self.dconv_down0(pool00)
        #print(conv01.size())
        pool01 = self.maxpool(conv01)  
        
        #RESNET152
        conv11 = self.dconv_down1(pool01)
        #print(conv11.size())
        conv21 = self.dconv_down2(conv11)
        #print(conv21.size())
        conv31 = self.dconv_down3(conv21)  
        #print(conv31.size())
        conv41 = self.dconv_down4(conv31)
        #print(conv41.size())
#         conv11 = checkpoint(self.dconv_down1, pool01)
#         #print(conv11.size())
#         conv21 = checkpoint(self.dconv_down2, conv11)
#         #print(conv21.size())
#         conv31 = checkpoint(self.dconv_down3,conv21)  
#         #print(conv31.size())
#         conv41 = checkpoint(self.dconv_down4,conv31)
#         #print(conv41.size())
        
        #3
        up32 = up_sampling(conv41) 
        conv32 = self.conv_g32(torch.cat((up32,conv31),dim=tensor_ax))

        #2
        up22 =up_sampling(conv31)
        conv22 = self.conv_g22(torch.cat((up22,conv21),dim=tensor_ax))

        up23 = up_sampling(conv32)
        conv23 = self.conv_g23(torch.cat((up23,conv21,conv22),dim=tensor_ax))

        #1
        up12 = up_sampling(conv21)
        conv12 = self.conv_g12(torch.cat((up12,conv11),dim=tensor_ax))

        up13 = up_sampling(conv22)
        conv13 = self.conv_g13(torch.cat((up13,conv11,conv12),dim=tensor_ax))

        up14 =up_sampling(conv23)
        conv14 = self.conv_g14(torch.cat((up14,conv11,conv12,conv13),dim=tensor_ax))
        
        #0
        up02 = up_sampling(conv11)
        conv02 = self.conv_g02(torch.cat((up02,conv01),dim=tensor_ax))
        
        up03 = up_sampling(conv12)
        conv03 = self.conv_g03(torch.cat((up03,conv01,conv02),dim=tensor_ax))
        
        up04 = up_sampling(conv13)
        conv04 = self.conv_g04(torch.cat((up04,conv01,conv02,conv03),dim=tensor_ax))
        
        up05 = up_sampling(conv14)
        conv05 = self.conv_g05(torch.cat((up05,conv01,conv02,conv03,conv04),dim=tensor_ax))
        
        #print(conv05.size())
        #-1
        up_12 = up_sampling(conv01)
        conv_12 = self.conv_g_12(torch.cat((up_12,conv00),dim=tensor_ax))
        
        up_13 = up_sampling(conv02)
        conv_13 = self.conv_g_13(torch.cat((up_13,conv00,conv_12),dim=tensor_ax))
        
        up_14 = up_sampling(conv03)
        conv_14 = self.conv_g_14(torch.cat((up_14,conv00,conv_12,conv_13),dim=tensor_ax))
        
        up_15 = up_sampling(conv04)
        conv_15 = self.conv_g_15(torch.cat((up_15,conv00,conv_12,conv_13,conv_14),dim=tensor_ax))
        
        up_16 = up_sampling(conv05)
        conv_16 = self.conv_g_16(torch.cat((up_16,conv00,conv_12,conv_13,conv_14,conv_15),dim=tensor_ax))
        #print(conv05.size())
        out = [self.conv_last1(conv_12),self.conv_last2(conv_13),self.conv_last3(conv_14),self.conv_last4(conv_15),self.conv_last5(conv_16)]
        
        for i in range(len(out)):
            out[i] = self.soft2d(out[i])
        
        
        
        return out

