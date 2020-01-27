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

dict_param = {
    'list':[0],
    'blocks':8,
    'threads':32,
    'BPM_ratio':8,
    'bb':256,
    'N_pix_st':250, #starting minimum area
    'astr_min':225, # approx. 0.9 min in dataset
    'percentile': 80,
    'pad':5,
    'astro_num':1, # number of astro min in FOV

    'init_th_':0.65, # threshold initialization approx. 120 frame
    'decr_dim':25, # astro area decrease
    'decr_th':25, # temporal threshold decrease
    'corr_int':False, # intensity correction flag
    'gpu_flag':True
}

parser = argparse.ArgumentParser(description ='Testing set up' )

parser.add_argument('-m','--model',default ='Unet_nest_5',choices=['Unet_nest_5','nestedUNetUp5_dense','dense_up'], help='model name (default:Unet_nest_5)')


args = parser.parse_args()

# model to import
model = model_dict[args.model]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

def count_soma(soma):
    ret_m, labels_m = cv2.connectedComponents(np.uint8(soma))
    for i in range(1,ret_m):
        pts = np.where(labels_m==i)
        print(len(pts[0]))
    return ret_m-1

def prob_calc(prob_map,max_a,min_a,conv):
    font= cv2.FONT_HERSHEY_DUPLEX
    map_ = np.zeros_like(prob_map)
    map_[prob_map>0]=1
    plt.imshow(map_)
    
    ret, labels = cv2.connectedComponents(np.uint8(map_))
    print('ssss',ret-1)
    for i in range(1,ret):
        pts = np.where(labels==i)
        if len(pts[0])*conv>(0.9*min_a*(0.144*0.144)) and len(pts[0])*conv<(1.15*max_a*(0.144*0.144)):
            
            q = np.around(np.sum(prob_map[pts])/len(pts[0]),decimals=2)
            cv2.putText(prob_map,str(q), (int(np.mean(pts[1])),int(np.mean(pts[0]))), font, 0.4, (2,0,0), 1, cv2.LINE_AA)
            print(len(pts[0]),q)
            if q<90 and len(pts[0])*conv<min_a*(0.144*0.144):
                map_[pts]=0
        else:
            print('qw',len(pts[0]))
            map_[pts]=0
            if len(pts[0])>400:
                map_[pts]=0
                prob_map[pts]=0
            
    return prob_map,map_



val_a = np.loadtxt('/media/DATA/jbonato/segm_project/U-net/data2.csv',delimiter=',')
conv_l = [0.146484,0.146484,0.130208,0.146484,0.195312,0.146484,0.146484,0.130208,0.130208,0.130208,0.146484,0.130208,0.146484,0.167411,0.146484,0.130208,0.167411,0.146484,0.130208,0.130208,0.146484,0.130208,0.146484,0.146484,0.146484]

dict_area = dict()

list_el = [i for i in range(26,51) ]
cnt_di=0
for el in list_el:
    dict_area[str(el)] = (val_a[cnt_di,:]//4)
    dict_area[str(el)+'conv'] = (conv_l[cnt_di]**2)
    cnt_di+=1


N=512

for jj in range(26,51):

    Res_1 = np.zeros((256,256,3))
    Res_1_512 = np.zeros((512,512,3))
    
    if jj !=20:
        test_folder_str =str(jj)
        if len(test_folder_str)==1:
            test_folder_str1='00'+test_folder_str
        else:
            test_folder_str1='0'+test_folder_str
       
        #model.load_state_dict(torch.load('/media/DATA/jbonato/segm_project/weights/'+args.model+test_folder_str1+'_256_newf.pt'))
        
        model.load_state_dict(torch.load('/media/DATA/jbonato/segm_project/weights/'+args.model+test_folder_str1+'_256_nopp.pt'))
        print('FOV',test_folder_str,args.model+test_folder_str1)
        #collect stack to analyze
        stack_dir = '/media/DATA/jbonato/segm_project/set/'+test_folder_str+'/'
        
        path_stack = glob.glob(stack_dir+'*')
        path_stack = [i for i in path_stack if 'ch_1' in i]
        
         
        
        if len(test_folder_str)==1:
            test_folder_str='00'+test_folder_str
        else:
            test_folder_str='0'+test_folder_str
        
        
        
        set_dir = '/media/DATA/jbonato/segm_project/set/dataset_car/'
        items = os.listdir(set_dir)
        for i in items:
            
            if i=='SMALL_'+test_folder_str+'_nf.hdf5':
                mask_test = np.empty((N,N,2))
        
                dset= h5py.File(set_dir+i,'r') 
                
                mask_test[:,:,0] =  np.asarray(dset['Values'])
                mask_test[:,:,1] =  np.asarray(dset['Values_soma'])
        
        #scrivere il selezionatore
        
        ################################
        sp_pp = spatial_pp(path_stack[0])
        stack,image_to_plot = sp_pp.create_img_d2()        
        a_reg = sel_active_reg(stack.astype(np.float32),dict_param)
        mask = a_reg.get_mask()
        
        
        filt = filt_im(path_stack[0],mask,192)
        coord_l = filt.get_instances()
        image_stack, mask_filtering = filt.save_im()
        ###################################
        
        if len(coord_l)!=0:
            
            ################################## prob Map
            image_set = image_to_plot*mask_filtering
        
            image_set = image_set[np.newaxis,np.newaxis,:,:]

            class SimDataset_test(Dataset):
                def __init__(self):
                    self.input_images = image_set[:,:,:,:]    


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
           
            qq = pred

            pred_mean = qq.data.cpu().numpy()
            mean = pred_mean[0]
            maxim = np.amax(mean,axis=0)
            mean[mean<maxim]=0

            _,sm_filt = prob_calc(mean[1,:,:],\
                                  dict_area[str(jj)][0],\
                                  dict_area[str(jj)][1],\
                                  dict_area[str(jj)+'conv'])

            ##################################
            image_set = image_stack[:,0,:,:]
            image_set = image_set[:,np.newaxis,:,:]

            class SimDataset_test(Dataset):
                def __init__(self):
                    self.input_images = image_set[:,:,:,:]    
                    
                
                def __len__(self):
                    return len(self.input_images)
                
                def __getitem__(self, idx):        
                    image = self.input_images[idx]
                    image = torch.from_numpy(image).float()
                        
                    return image
            
            import math
            
            model.eval()   # Set model to evaluate mode
            
            test_dataset = SimDataset_test()
            test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False, num_workers=0)
                    
            inputs = next(iter(test_loader))
            inputs = inputs.to(device)

            pred = model(inputs)
            qq = pred
            pred_mean = qq.data.cpu().numpy()


            input_images = [x for x in inputs.cpu()]
            del inputs,qq
            print('immgine di dim:',input_images[0].shape)
            #label_im = [x for x in labels.cpu().numpy()]

            score_arr=np.empty((len(input_images),2))
            index_list=[]
            onlysoma = np.zeros_like(mask_test[:,:,1])  
            fin_mask = np.zeros_like(mask_test[:,:,1])
            fin_mask_2 = np.zeros_like(mask_test[:,:,1])
            #inserire un while cambiando la soglia 
            cnt_soma=0
            th_ = 300
            
            for i in range(len(input_images)):
                mean= np.zeros((3,96,96))
                mean = pred_mean[i].copy()
                maxim = np.amax(mean,axis=0)
                mean[mean<maxim]=0
                mean[mean>=maxim]=1

                small_soma = small_soma_to_proc(mean[1,:,:],N = th_)
                mean[0,:,:]+=small_soma
                mean[1,:,:]-=small_soma
               # print('qqqqq',i,input_images[i].shape)
               #fig,ax = plt.subplots(figsize=(10,10),ncols=3,nrows=1)
               #ax[0].imshow(small_soma)
               #ax[1].imshow(input_images[i][0,:,:])
               #ax[2].imshow(mean[1,:,:])
               #plt.savefig('/media/DATA/jbonato/segm_project/U-net/Results/im'+str(i)+'.png')
               #plt.close()
                coord = coord_l[i]
                onlysoma[coord[1]:coord[3],coord[0]:coord[2]]+= mean[1,:,:] 

                Res_1[coord[1]:coord[3],coord[0]:coord[2],0] += mean[0,:,:]
                Res_1[coord[1]:coord[3],coord[0]:coord[2],1] += mean[1,:,:]
                     
                Res_1[:,:,0] -= Res_1[:,:,1]
                Res_1[Res_1<1]=0
                Res_1[Res_1>0]=1
                
                
                soma_f = common_merge(Res_1[:,:,1],sm_filt)
                Res_1[:,:,1]=soma_f
                
                Res_1_filt = art_rem(Res_1[:,:,1],Res_1[:,:,0])
                Res_1_bis = Res_1*Res_1_filt[:,:,np.newaxis]
                
                filt = dilation_fun(Res_1_bis[:,:,0]+Res_1_bis[:,:,1],Res_1[:,:,0])
                filt = art_rem(Res_1[:,:,1],filt)
                Res_1*=filt[:,:,np.newaxis]
                
                cnt_soma = count_soma(Res_1[:,:,1])
                print('Analysis with soma th{:d} pix*2, revealed soma are {:d}'.format(th_,cnt_soma))
                th_-=10

            Res_1[:,:,0] -= Res_1[:,:,1]
            Res_1[Res_1<1]=0
            Res_1[Res_1>0]=1

            Res_1_512[:,:,0] = cv2.resize(np.uint8(Res_1[:,:,0]),(512,512),interpolation=cv2.INTER_AREA)
            Res_1_512[:,:,1] = cv2.resize(np.uint8(Res_1[:,:,1]),(512,512),interpolation=cv2.INTER_AREA)
            #save DNN output
            with h5py.File('/media/DATA/jbonato/segm_project/U-net/Results/'+args.model+'/SMALL_'+test_folder_str+'.hdf5','w') as f:
                dset = f.create_dataset('Values',data=Res_1_512[:,:,0])
                dset2 = f.create_dataset('Values_soma',data=Res_1_512[:,:,1])
                    ##########################         PLOT               ########
#             im512 = gen_Image512(path_stack[0])
#             GT_mask = np.zeros((512,512,3))
#             GT_small = '/media/DATA/jbonato/segm_project/set/dataset_car/SMALL_'+test_folder_str+'_nf.hdf5'
#             dset= h5py.File(GT_small,'r') 
#             GT_mask[:,:,0] =  np.asarray(dset['Values'])
#             GT_mask[:,:,1] =  np.asarray(dset['Values_soma'])
            
#             Res_mask = np.zeros((512,512,3))
#             Res_mask[:,:,0] = Res_1_512[:,:,0]
#             Res_mask[:,:,1] = Res_1_512[:,:,1]
            
#             fig,ax = plt.subplots(figsize=(20,20),nrows=1,ncols=3)
#             ax[0].imshow(im512)
#             ax[1].imshow(GT_mask)
#             ax[2].imshow(Res_mask)
#             plt.savefig('/media/DATA/jbonato/segm_project/U-net/test/SMALL_'+test_folder_str+'.png')
#             plt.close()
            
            
            
            
            
            
            
