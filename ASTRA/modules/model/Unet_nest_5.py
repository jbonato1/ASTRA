# ver due con upsample module instead of conv2d
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

class nestedUNetUp5(nn.Module):

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
        
        self.dconv_down4 = double_conv(256, 512) 
        self.dconv_down4.apply(init_weights)


        #3
        self.conv_g32 = double_conv(512+256,256)
        self.conv_g32.apply(init_weights)
        
        #2
        self.conv_g22 = double_conv(256+128,128)
        self.conv_g22.apply(init_weights)
        

        self.conv_g23 = double_conv(256+128+128,128)
        self.conv_g23.apply(init_weights)
        
        #1
        self.conv_g12 = double_conv(128+64,64)
        self.conv_g12.apply(init_weights)
        

        self.conv_g13 = double_conv(128+64+64,64)
        self.conv_g13.apply(init_weights)
        

        self.conv_g14 = double_conv(128+64+64+64,64)
        self.conv_g14.apply(init_weights)
        
        #0
        self.conv_g02 = double_conv(64+32,32)
        self.conv_g02.apply(init_weights)
        
        self.conv_g03 = double_conv(64+32+32,32)
        self.conv_g03.apply(init_weights)
        
        self.conv_g04 = double_conv(64+32+32+32,32)
        self.conv_g04.apply(init_weights)
        
        self.conv_g05 = double_conv(64+32+32+32+32,32)
        self.conv_g05.apply(init_weights)
        
        self.soft2d = nn.Softmax2d()
        self.sig = nn.Sigmoid()
        self.maxpool = nn.MaxPool2d(2)
        
        self.conv_last1 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last1.weight)
        self.conv_last2 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last2.weight)
        self.conv_last3 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        self.conv_last4 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        
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
        
        up32 = up_sampling(conv41)
        #print(up32.shape)
        conv32 = self.conv_g32(torch.cat((up32,conv31),dim=tensor_ax))

        up22 =up_sampling(conv31)
        #print(up22.shape)
        conv22 = self.conv_g22(torch.cat((up22,conv21),dim=tensor_ax))

        up23 = up_sampling(conv32)
        #print(up23.shape)
        conv23 = self.conv_g23(torch.cat((up23,conv21,conv22),dim=tensor_ax))

        up12 = up_sampling(conv21)
        conv12 = self.conv_g12(torch.cat((up12,conv11),dim=tensor_ax))

        up13 = up_sampling(conv22)
        conv13 = self.conv_g13(torch.cat((up13,conv11,conv12),dim=tensor_ax))

        up14 =up_sampling(conv23)
        conv14 = self.conv_g14(torch.cat((up14,conv11,conv12,conv13),dim=tensor_ax))
        
        up02 = up_sampling(conv11)
        conv02 = self.conv_g02(torch.cat((up02,conv01),dim=tensor_ax))
        
        up03 = up_sampling(conv12)
        conv03 = self.conv_g03(torch.cat((up03,conv01,conv02),dim=tensor_ax))
        
        up04 = up_sampling(conv13)
        conv04 = self.conv_g04(torch.cat((up04,conv01,conv02,conv03),dim=tensor_ax))
        
        up05 = up_sampling(conv14)
        conv05 = self.conv_g05(torch.cat((up05,conv01,conv02,conv03,conv04),dim=tensor_ax))
        
        
        
        out = [self.conv_last1(conv02),self.conv_last2(conv03),self.conv_last3(conv04),self.conv_last4(conv05)]
        for i in range(len(out)):
            if self.flag_1cl:
                pass
                #out[i] = self.sig(out[i])
            else:
                out[i] =  self.soft2d(out[i])
        
        return out



class nestedUNetUp5_pretr(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        pretr = nestedUNetUp5(2)
        pretr.load_state_dict(torch.load('/home/jbonato/Documents/U-Net/weights/pretr.pt'))

        self.model = nn.Sequential(*list(pretr.children())[:])
        
        self.soft2d = nn.Softmax2d()
        
        self.conv_last1 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last1.weight)
        self.conv_last2 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last2.weight)
        self.conv_last3 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        self.conv_last4 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)

    def forward(self, x):

        x = self.model(x)

        # out = []
        # cnt=0
        # for child in x:
        #     if cnt in {11,12,13,14}:
        #         out.append(child)
        #     cnt+=1
        # out = [self.conv_last1(conv02),self.conv_last2(conv03),self.conv_last3(conv04),self.conv_last4(conv05)]
        # for i in range(len(out)):
        #     out[i] =  self.soft2d(out[i])
        return x
def upsample_dense(in_ch,d):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch*(d**2), 3, padding=1),
        nn.BatchNorm2d(in_ch*(d**2)),
        nn.ReLU(inplace=True),
        nn.PixelShuffle(d)           
            )
    

class nestedUNetUp5_dense(nn.Module):

    def __init__(self, n_class,flag_1cl=False):
        super().__init__()
        self.flag_1cl = flag_1cl

        self.maxpool = nn.MaxPool2d(2)
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
        #4
        #self.up4 = upsample_dense(512,2)
        #self.up4.apply(init_weights)
        #3
        #self.up31 = upsample_dense(256,2)
        #self.up31.apply(init_weights)

        #self.up32 = upsample_dense(256,2)
        #self.up31.apply(init_weights)

        self.conv_g32 = double_conv(128+256,256)#
        self.conv_g32.apply(init_weights)
        
        #2
        #self.up21 = upsample_dense(128,2)
        #self.up21.apply(init_weights)


        #self.up22 = upsample_dense(128,2)
        #self.up22.apply(init_weights)
        self.conv_g22 = double_conv(64+128,128)
        self.conv_g22.apply(init_weights)
        
        #self.up23 = upsample_dense(128,2)
        #self.up23.apply(init_weights)
        self.conv_g23 = double_conv(64+128+128,128)
        self.conv_g23.apply(init_weights)
        
        #1
        #self.up11 = upsample_dense(64,2)
        #self.up11.apply(init_weights)
        #self.up12 = upsample_dense(64,2)
        #self.up12.apply(init_weights)
        self.conv_g12 = double_conv(32+64,64)
        self.conv_g12.apply(init_weights)
        
        #self.up13 = upsample_dense(64,2)
        #self.up13.apply(init_weights)
        self.conv_g13 = double_conv(32+64+64,64)
        self.conv_g13.apply(init_weights)
        
        #self.up14 = upsample_dense(64,2)
        #self.up14.apply(init_weights)
        self.conv_g14 = double_conv(32+64+64+64,64)
        self.conv_g14.apply(init_weights)
        
        #0
        self.conv_g02 = double_conv(16+32,32)
        self.conv_g02.apply(init_weights)
        
        self.conv_g03 = double_conv(16+32+32,32)
        self.conv_g03.apply(init_weights)
        
        self.conv_g04 = double_conv(16+32+32+32,32)
        self.conv_g04.apply(init_weights)
        
        self.conv_g05 = double_conv(16+32+32+32+32,32)
        self.conv_g05.apply(init_weights)
        
        self.soft2d = nn.Softmax2d()
        self.sig = nn.Sigmoid()
        
        self.conv_last1 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last1.weight)
        self.conv_last2 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last2.weight)
        self.conv_last3 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        self.conv_last4 = nn.Conv2d(32, n_class, 1)
        nn.init.kaiming_uniform_(self.conv_last3.weight)
        
        self.up = nn.PixelShuffle(2)
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
        
        up32 = self.up(conv41)
        #print(up32.shape)
        conv32 = self.conv_g32(torch.cat((up32,conv31),dim=tensor_ax))

        up22 =self.up(conv31)
        #print(up22.shape)
        conv22 = self.conv_g22(torch.cat((up22,conv21),dim=tensor_ax))

        up23 = self.up(conv32)
        #print(up23.shape)
        conv23 = self.conv_g23(torch.cat((up23,conv21,conv22),dim=tensor_ax))

        up12 = self.up(conv21)
        conv12 = self.conv_g12(torch.cat((up12,conv11),dim=tensor_ax))

        up13 = self.up(conv22)
        conv13 = self.conv_g13(torch.cat((up13,conv11,conv12),dim=tensor_ax))

        up14 = self.up(conv23)
        conv14 = self.conv_g14(torch.cat((up14,conv11,conv12,conv13),dim=tensor_ax))
        
        up02 = self.up(conv11)
        conv02 = self.conv_g02(torch.cat((up02,conv01),dim=tensor_ax))
        
        up03 = self.up(conv12)
        conv03 = self.conv_g03(torch.cat((up03,conv01,conv02),dim=tensor_ax))
        
        up04 = self.up(conv13)
        conv04 = self.conv_g04(torch.cat((up04,conv01,conv02,conv03),dim=tensor_ax))
        
        up05 = self.up(conv14)
        conv05 = self.conv_g05(torch.cat((up05,conv01,conv02,conv03,conv04),dim=tensor_ax))
        
        
        out = [self.conv_last1(conv02),self.conv_last2(conv03),self.conv_last3(conv04),self.conv_last4(conv05)]
        for i in range(len(out)):
            out[i] =  self.soft2d(out[i])
        
        return out
       

               


