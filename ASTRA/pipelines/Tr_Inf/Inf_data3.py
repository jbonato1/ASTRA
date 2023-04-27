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
import math
import sys

import argparse
sys.path.insert(0,'/media/DATA/jbonato/astro_segm/Astro3S/modules/')

from test_fun import *
from sel_active_reg_gen import *
from gen_single_astro  import *


from model.dense_up import dense_up





model_dict={
           'dense_up': dense_up(2)
        }


dict_param = {
    'list':[i for i in range(0,400,30)],
    'blocks':14*2,
    'threads':20,
    'BPM_ratio':2,
    'bb':40,
    'N_pix_st':25, #starting minimum area
    'astr_min':22, 
    'percentile': 90,
    'pad':5,
    'astro_num':95, # number of astro min in FOV
    'init_th_':0.5, # threshold initialization approx. 125
    'decr_dim':3, # astro area decrease
    'decr_th':12, # temporal threshold decrease
    'corr_int':True, # intensity correction flag
    'gpu_flag':True
}

def prob(mask):
    buff = mask.copy()
    buff[buff>0]=1
    out = np.zeros_like(mask)
    ret,label = cv2.connectedComponents(np.uint8(buff))
    for i in range(1,ret):
        pts = np.where(label==i)
        if len(pts[0])>20:
            prob = np.sum(mask[pts[0],pts[1]])/len(pts[0])
            if prob>=0.75:
                out[pts[0],pts[1]]=1
    return out

def small_roi(mask,min_a,max_a):
    mask_tot = mask.copy()
    mask_tot[mask_tot>0.5]=255
    mask_tot[mask_tot<=0.5]=0
    mask_tot= np.uint8(mask_tot)
    ret, labels = cv2.connectedComponents(mask_tot)
    
    for i in range(1, ret+1):
        pts =  np.where(labels == i)

        if len(pts[0]) >= int(0.9*min_a):
   #         print(len(pts[0]))
            labels[pts] = 1
        else:
            #print(len(pts[0]))
            labels[pts] = 0
    mask_tot=labels
    return mask_tot


parser = argparse.ArgumentParser(description ='Testing set up' )

parser.add_argument('-m','--model',default ='dense_up',choices=['dense_up'], help='model name (default:dense_up)')

# parser.add_argument('-d','--data', metavar='DIR', default='/home/jbonato/Documents/U-Net/set/train_single',help='path to dataset')

# parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 1)')


# parser.add_argument('-fov','--test_folder_str',help='test folder which has to be excluded from training')

args = parser.parse_args()

# model to import
model = model_dict[args.model]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')#('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)



N=430
max_min = np.loadtxt('/media/DATA/jbonato/astro_segm/Astro3S/pipelines/data3.csv',delimiter=',')

for jj in range(1,7):

    Res_1 = np.zeros((430,430,2))
    
    if jj !=20 and jj!=23:
        test_folder_str =str(jj)
        if len(test_folder_str)==1:
            test_folder_str1='00'+test_folder_str
        else:
            test_folder_str1='0'+test_folder_str
        
    
        model.load_state_dict(torch.load('/media/DATA/jbonato/astro_segm/weights/'+args.model+test_folder_str1+'D3.pt'))
        
        
        
        #collect stack to analyze
        stack_dir = '/media/DATA/jbonato/astro_segm/set3/'+test_folder_str+'/'
        items_stack = os.listdir(stack_dir)
        items_stack = [i for i in items_stack if not('im_enh' in i)]
        print(stack_dir + items_stack[0])
        
        ##############STACK
        stack = io.imread(stack_dir + items_stack[0])
        stack = stack[:,2:-3,2:-3]
        frames,_,_ = stack.shape
         
        
        if len(test_folder_str)==1:
            test_folder_str='00'+test_folder_str
        else:
            test_folder_str='0'+test_folder_str
        
        ### find zone
        sp_pp = spatial_pp(stack_dir + items_stack[0])
        stack_new,image_to_plot =sp_pp.create_img_large()
        
        a_reg = sel_active_reg(stack_new.astype(np.float32),dict_param)
        mask = a_reg.get_mask()
        
        ### generate images
        
        filter_ = filt_im(stack_dir + items_stack[0], mask,40,filt_meth='ad_hoc')
        
        
        coord_l = filter_.get_instances()
        image_set,filt_image_L = filter_.save_im(pad=4,case=3)
        print('IMAGE',image_set.shape)
        if len(coord_l)!=0:
            
            image_stack = np.empty((len(coord_l),48,48))
           
            #print('MEM',torch.cuda.memory_allocated(),torch.cuda.memory_cached())
            class SimDataset_test(Dataset):
                def __init__(self):
                    a,b,c,d = image_set.shape
                    self.input_images = image_set[:,0,:,:].reshape(a,1,c,d)    
                    
                
                def __len__(self):
                    return len(self.input_images)
                
                def __getitem__(self, idx):        
                    image = self.input_images[idx]
                    image = torch.from_numpy(image).float()
                        
                    return image
            
            import math
            
            model.eval()   # Set model to evaluate mode
            
            test_dataset = SimDataset_test()
            test_loader = DataLoader(test_dataset, batch_size=70, shuffle=False, num_workers=0)         
            
            pred_mean=[]
            for inputs in test_loader:
                inputs = inputs.to(device)

                pred = model(inputs)
                if args.model=='dense_up':
                    qq = pred
                else:
                    qq = pred[3]

                pred_mean.append(qq.data.cpu().numpy())

                del inputs,qq
            #print('MEM',torch.cuda.memory_allocated(),torch.cuda.memory_cached())
            torch.cuda.empty_cache()
            #print('MEM',torch.cuda.memory_allocated(),torch.cuda.memory_cached())

            for j in range(1,len(pred_mean)):
                
                pred_mean[0]=np.vstack((pred_mean[0],pred_mean[j]))
            
            print(pred_mean[0].shape,len(coord_l))
           
            for i in range(len(coord_l)):
                mean= np.zeros((2,48,48))
                mean = pred_mean[0][i,:,:,:].copy()
                maxim = np.amax(mean,axis=0)
                mean[mean<maxim]=0
                
                out = prob(mean[0,4:-4,4:-4])
                mean[mean>=maxim]=1
                coord = coord_l[i]
                Res_1[coord[1]:coord[3],coord[0]:coord[2],0] += (mean[0,4:-4,4:-4]*out)
                
                
                

            
            Res_1[Res_1<1]=0
            Res_1[Res_1>0]=1
            
            filt = small_roi(Res_1[:,:,0],max_min[jj-1,1],max_min[jj-1,0])
            Res_1[:,:,0]*=filt
            ### set to 20 the limit of number of removal
            Res_1_filt,removal = art_rem_large(Res_1[:,:,0],N=int(1.15*max_min[jj-1,0]))
            if removal<20:
                Res_1[:,:,0]-=Res_1_filt
                
            Res_1_filt,removal = art_rem_large(Res_1[:,:,0],N=int(max_min[jj-1,0]))
            if removal<20:
                Res_1[:,:,0]-=Res_1_filt
            
            ###
            
            with h5py.File('/media/DATA/jbonato/astro_segm/Results/D3/LARGE_'+test_folder_str1+'D3.hdf5','w') as f:
                        dset2 = f.create_dataset('Values_soma',data=Res_1[:,:,0])
            


        else:
            Res_1[:,:,:]=0






