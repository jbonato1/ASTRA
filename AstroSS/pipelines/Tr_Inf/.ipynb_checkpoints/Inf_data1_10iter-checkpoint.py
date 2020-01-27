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




args = parser.parse_args()

# model to import
model = model_dict[args.model]
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')#('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

dict_param = {
    'list':[0,40,80,120,160],
    'blocks':15,
    'threads':32,
    'BPM_ratio':3,
    'bb':96,
    'N_pix_st':100, #starting minimum area
    'astr_min':80, # approx. 0.9 min in dataset
    'percentile': 80,
    'pad':5,
    'astro_num':4, # number of astro min in FOV
    'init_th_':0.6, # threshold initialization
    'decr_dim':10, # astro area decrease
    'decr_th':25, # temporal threshold decrease
    'corr_int':False, # intensity correction flag
    'gpu_flag':True
}




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
    if jj !=20:
        test_folder_str =str(jj)
        if len(test_folder_str)==1:
            test_folder_str1='00'+test_folder_str
        else:
            test_folder_str1='0'+test_folder_str
        
#         model.load_state_dict(torch.load('/media/DATA/jbonato/segm_project/weights/data_iter/'+args.model+'_iter_'+args.iter+'_'+test_folder_str1+'.pt'))
        
        print('FOV',test_folder_str,args.model+test_folder_str1)
        #collect stack to analyze
        stack_dir = '/media/DATA/jbonato/segm_project/set/'+test_folder_str+'/'
        items_stack = os.listdir(stack_dir)

        print(stack_dir + items_stack[0])
        stack = io.imread(stack_dir + items_stack[0])
        stack = stack[:,:,:]
        frames,_,_ = stack.shape
         
        
        a_reg = sel_active_reg(stack.astype(np.float32),dict_param)
        mask = a_reg.get_mask()
        #mask = sel_active_reg_gpu(stack.astype(np.float32))
        mask = fix_mask(mask)
        
        filter_ = filt_im(stack_dir + items_stack[0],mask,86)
        #filter_ = filt_im(stack,mask,jj)
        _,image_to_plot = filter_.create_im()
        coord_l = filter_.get_instances()
        
        if len(coord_l)!=0:
            image_stack = np.empty((len(coord_l),96,96))
            #single cell image
            image_stack, filt_imageL = filter_.save_im()
            image_set = image_stack[:,0,:,:]
            image_set = image_set[:,np.newaxis,:,:]
            #large fov image
            imageL_set = image_to_plot*filt_imageL
            imageL_set-=np.mean(imageL_set)
            imageL_set= imageL_set[np.newaxis,np.newaxis,:,:]
            #large im
            class SimDataset_test_single(Dataset):
                def __init__(self):
                    self.input_images = imageL_set[:,:,:,:]    


                def __len__(self):
                    return len(self.input_images)

                def __getitem__(self, idx):        
                    image = self.input_images[idx]
                    image = torch.from_numpy(image).float()

                    return image
            #single cell im
            class SimDataset_test(Dataset):
                def __init__(self):
                    self.input_images = image_set[:,:,:,:]    
                    
                
                def __len__(self):
                    return len(self.input_images)
                
                def __getitem__(self, idx):        
                    image = self.input_images[idx]
                    image = torch.from_numpy(image).float()
                        
                    return image
            
            
            for iteration in range(1,11):
                print("ITERATION: " , iteration)
                Res_1 = np.zeros((256,256,3))
                model.load_state_dict(torch.load('/media/DATA/jbonato/segm_project/weights/data_iter/'+args.model+'_iter_'+str(iteration)+'_'+test_folder_str1+'.pt'))
                model.eval()   # Set model to evaluate mode
                ######################################### IMAGE
                test_dataset = SimDataset_test_single()
                test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False, num_workers=0)

                inputs = next(iter(test_loader))
                inputs = inputs.to(device)

                pred = model(inputs)
                if args.model == 'dense_up':
                    qq = pred
                else:
                    qq = pred[3]

                pred_mean = qq.data.cpu().numpy()
                del test_dataset,test_loader, inputs,pred
                mean = pred_mean[0]
                maxim = np.amax(mean,axis=0)
                mean[mean<maxim]=0

                max_min = dict_area[str(jj)]
                prob_mapPL,sm_ent = prob_calc(mean[1,:,:],max_min[0],max_min[1])

                ########################################### SINGLE CELL
                test_dataset = SimDataset_test()
                test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False, num_workers=0)

                inputs = next(iter(test_loader))
                inputs = inputs.to(device)

                pred = model(inputs)
                if args.model=='dense_up':
                    qq = pred
                else:
                    qq = pred[3]

                pred_mean = qq.data.cpu().numpy()


                input_images = [x for x in inputs.cpu()]
                del inputs,qq

                #inserire un while cambiando la soglia 
                cnt_soma=0
                th_ = int((2/3)*max_min[1])
    #             while(cnt_soma<3 and th_>=30):
    #                 print('Analysis with soma th{:d} pix*2, revealed soma are {:d}'.format(th_,cnt_soma))
                prob_map = np.zeros((256,256,2))

                for i in range(len(input_images)):
                    mean= np.zeros((3,96,96))
                    mean = pred_mean[i].copy()
                    maxim = np.amax(mean,axis=0)
                    mean[mean<maxim]=0
                    coord = coord_l[i]
                    prob_map[coord[1]:coord[3],coord[0]:coord[2],0] +=mean[1,5:-5,5:-5]
                    prob_map[coord[1]:coord[3],coord[0]:coord[2],1] +=1
                    mean[mean>=maxim]=1

                    small_soma = small_soma_to_proc(mean[1,:,:],N = th_)
                    mean[0,:,:]+=small_soma
                    mean[1,:,:]-=small_soma

    #                 fig,ax = plt.subplots(figsize=(10,10),ncols=3,nrows=1)
    #                 ax[0].imshow(small_soma)
    #                 ax[1].imshow(input_images[i][0,:,:])
    #                 ax[2].imshow(mean[1,:,:])
    #                 plt.savefig('/media/DATA/jbonato/segm_project/U-net/Results/im'+str(i)+'.png')
    #                 plt.close()
                    coord = coord_l[i]
                    Res_1[coord[1]:coord[3],coord[0]:coord[2],0] += mean[0,5:-5,5:-5]
                    Res_1[coord[1]:coord[3],coord[0]:coord[2],1] += mean[1,5:-5,5:-5]


    #             prob_map[:,:,1][prob_map[:,:,1]<1]=1
    #             PM = prob_map[:,:,0]/prob_map[:,:,1]

                Res_1[:,:,0] -= Res_1[:,:,1]
                Res_1[Res_1<1]=0
                Res_1[Res_1>0]=1

                soma_f = common_merge(Res_1[:,:,1],sm_ent)
                Res_1[:,:,1]=soma_f

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
                with h5py.File('/media/DATA/jbonato/segm_project/U-net/Results/iter_10/SMALL_'+test_folder_str+'_'+str(iteration)+'.hdf5','w') as f:
                    dset = f.create_dataset('Values',data=Res_1[:,:,0])
                    dset2 = f.create_dataset('Values_soma',data=Res_1[:,:,1])

        else:
            Res_1[:,:,:]=0






