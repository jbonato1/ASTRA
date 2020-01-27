import numpy as np
import numpy.ma as ma
import os
import h5py
from skimage import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
import cv2
import pandas as pd
import h5py


import argparse

from modules.test_fun import *
from modules.sel_active_reg_gen import *
from modules.gen_single_astro  import *

from modules.model.Unet_nest_5 import nestedUNetUp5 as Unet_nest_5
from modules.model.Unet_nest_5 import nestedUNetUp5_dense as Unet_nest_5_dense
from modules.model.dense_up import dense_up


import math


model_dict={'Unet_nest_5': Unet_nest_5(3),
             'nestedUNetUp5_dense': Unet_nest_5_dense(3),
           'dense_up': dense_up(3)
        }


parser = argparse.ArgumentParser(description ='Testing set up' )

parser.add_argument('-m','--model',default ='Unet_nest_5',choices=['Unet_nest_5','nestedUNetUp5_dense','dense_up'], help='model name (default:Unet_nest_5)')

parser.add_argument('-d','--data', metavar='DIR', default='/home/jbonato/Documents/U-Net/set/train_single',help='path to dataset')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 1)')


parser.add_argument('-fov','--test_folder_str',help='test folder which has to be excluded from training')

args = parser.parse_args()

# model to import
model = model_dict[args.model]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')#('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


val_a = np.loadtxt('/media/DATA/jbonato/segm_project/U-net/data1.csv',delimiter=',')

dict_area = dict()

list_el = [i for i in range(1,26) if i !=20]
cnt_di=0
for el in list_el:
    dict_area[str(el)] = val_a[cnt_di,:]
    cnt_di+=1

N=256
for jj in range(1,26):
    t1 = time.time()
    Res_1 = np.zeros((256,256,3))
    Res_2 = np.zeros((256,256,3))
    if jj !=20:
        test_folder_str =str(jj)
        if len(test_folder_str)==1:
            test_folder_str1='00'+test_folder_str
        else:
            test_folder_str1='0'+test_folder_str
        max_min = dict_area[str(jj)]
      
        model.load_state_dict(torch.load('/media/DATA/jbonato/segm_project/weights/no_pproc/'+args.model+test_folder_str1+'nf_enh.pt')) 
       
        print('FOV',test_folder_str,args.model+test_folder_str1)
        #collect stack to analyze
        stack_dir = '/media/DATA/jbonato/segm_project/set/'+test_folder_str+'/'
        items_stack = os.listdir(stack_dir)

        print(stack_dir + items_stack[0])
        stack = io.imread(stack_dir + items_stack[0]).astype(np.uint16)
        frames,_,_ = stack.shape
         
        
        if len(test_folder_str)==1:
            test_folder_str='00'+test_folder_str
        else:
            test_folder_str='0'+test_folder_str
        
        
        set_dir = '/media/DATA/jbonato/segm_project/set/dataset/'
        items = os.listdir(set_dir)
        for i in items:
            if i=='SMALL_'+test_folder_str+'_nf.hdf5':
                mask_test = np.empty((N,N,2))
        
                dset= h5py.File(set_dir+i,'r') 
                
                mask_test[:,:,0] =  np.asarray(dset['Values'])
                mask_test[:,:,1] =  np.asarray(dset['Values_soma'])
        
        
        mask = np.zeros((256,256),dtype=np.uint16)
        filter_ = filt_im(stack_dir + items_stack[0],mask,86)
        _,image_to_plot = filter_.create_im()

        
        imageL_set = image_to_plot.copy().astype(np.float32)
        imageL_set-=np.mean(imageL_set)
        imageL_set= imageL_set[np.newaxis,np.newaxis,:,:]

        class SimDataset_test(Dataset):
            def __init__(self):
                self.input_images = imageL_set[:,:,:,:]    


            def __len__(self):
                return len(self.input_images)

            def __getitem__(self, idx):        
                image = self.input_images[idx]
                image = torch.from_numpy(image).float()

                return image

        model.eval()   # Set model to evaluate mode

        test_dataset = SimDataset_test()
        test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False, num_workers=0)

        inputs = next(iter(test_loader))
        inputs = inputs.to(device)

        pred = model(inputs)
        if 'dense_up'=='dense_up':
            qq = pred
        else:
            qq = pred[3]

        pred_mean = qq.data.cpu().numpy()
        del test_dataset,test_loader, inputs,pred
        mean = pred_mean[0]
        maxim = np.amax(mean,axis=0)
        mean[mean<maxim]=0
        mean[mean>0]=1
        Res_1[:,:,0] = mean[0,:,:]
        Res_1[:,:,1] = mean[1,:,:] 

        Res_1[:,:,0] -= Res_1[:,:,1]
        Res_1[Res_1<1]=0
        Res_1[Res_1>0]=1
        
        #remove possible artifacts
        small_soma = small_soma_to_proc(Res_1[:,:,1],int(0.9*max_min[1]),dilation=False)
        Res_1[:,:,1]-=small_soma

        Res_1[:,:,0] = Res_1[:,:,0]-Res_1[:,:,1]
        Res_1[Res_1<1]=0
        Res_1[Res_1>0]=1
        
        #remove large region classified as soma Area>500
        Res_1_filt,removal = art_rem_large(Res_1[:,:,1],Res_1[:,:,0],N=int(1.15*max_min[0]))
        if removal<2:
            Res_1-=Res_1_filt[:,:,np.newaxis]
            
        Res_1_filt,removal = art_rem_large(Res_1[:,:,1],Res_1[:,:,0],N=int(max_min[0]))
        if removal<2:
            Res_1-=Res_1_filt[:,:,np.newaxis]
        

        #remove processes without soma
        Res_1_filt = art_rem(Res_1[:,:,1],Res_1[:,:,0])
        Res_1*=Res_1_filt[:,:,np.newaxis]
        
        
        
        
        #save DNN output
        with h5py.File('/media/DATA/jbonato/segm_project/U-net/Results/no_pproc/SMALL_'+test_folder_str+'_no_swin.hdf5','w') as f:
            dset = f.create_dataset('Values',data=Res_1[:,:,0])
            dset2 = f.create_dataset('Values_soma',data=Res_1[:,:,1])


        
        #plot the develop of segmentation and correlation
        fig, ax = plt.subplots(figsize=(20, 20), nrows=1, ncols=2)
        ax[0].imshow(image_to_plot)
        ax[1].imshow(Res_1)
        plt.savefig('/media/DATA/jbonato/segm_project/U-net/test/'+args.model+'_no_swin'+str(jj)+'.png')
        plt.close()










