root_folder='/media/DATA/jbonato/astro_segm'

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io
import sys
import h5py
import glob
import shutil
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from collections import defaultdict

import time
import copy
 

# ##################import modules
sys.path.insert(0,root_folder+'/RASTA/modules/')
from aug_images import compose_tr

from model.dense_up import dense_up
from train_mod import train_model


# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True


model_dict={'dense_up': dense_up(3)
            }
folder_w = root_folder+'/weights/'
if not(os.path.exists(folder_w)):
    os.mkdir(folder_w)
    print('Created',folder_w)
    
#########################
model_n = 'dense_up'
parallel_n =True
data_dir = root_folder+'/set4/train_single'
workers = 1
epochs = 12
batch_size = 35
lr = 1e-4
parallel = True
test_folders = ['7']
for test_folders in ['8']:#,'5','6','7','8'
#image size: this value must be divisible for 2^4 i.e. 98, 128,256,512
    N=48
    M=48
    ###########################
    items = os.listdir(data_dir)
    for test_f in test_folders:

        test_folder_str = test_f
        if len(test_folder_str)==1:
            test_folder_str='00'+test_folder_str
        else:
            test_folder_str='0'+test_folder_str

        print('Removing: FOV:',test_folder_str)
        test=[]
        query_num = len(items)
        for i in items:
            if 'SMALL_'+test_folder_str in i:
                items.remove(i)
                test.append(i)

        assert query_num-len(test)==len(items),'Error in removing test folders'



    counter = 0
    counter2 = 0
    query_pref = 'nh'# the files in train_single folder with this suffix will not be loaded

    im = np.empty((N,M),dtype = np.float32)

    flag=True
    for item in items:
        path_to_item = os.path.join(data_dir,item)
        filename, file_extension = os.path.splitext(path_to_item)
        if os.path.isfile(path_to_item) and file_extension == '.tif' and filename[-2:]!='nh':        
            im = io.imread(path_to_item)
            #if counter == 0: 
            if flag:
                out_im = im
                out_im = out_im[np.newaxis,:,:]
            else:
                out_im = np.concatenate((out_im,im[np.newaxis,:,:]),axis=0)

            dset= h5py.File(filename+'.hdf5','r') 
            proc_mask =  np.asarray(dset['Values'])
            soma_mask =  np.asarray(dset['Values_soma'])
            proc_mask[np.where(proc_mask==soma_mask)]=0
            back = np.ones((N,M),dtype=np.int64)-proc_mask-soma_mask
            back[back<0]=0
            mask = np.concatenate((proc_mask[:,:,np.newaxis],soma_mask[:,:,np.newaxis],back[:,:,np.newaxis]),axis=2).astype(np.float32)

            if flag:
                label = mask
                label = label[np.newaxis,:,:,:]
                flag=False
            else:
                label = np.concatenate((label,mask[np.newaxis,:,:,:]),axis=0)


    label = np.swapaxes(label,1,3)
    label = np.swapaxes(label,2,3)

    out_im = out_im[:,np.newaxis,:,:].astype(np.float32)

    print('im shape:',out_im.shape)
    print('label shape', label.shape)

    NN = out_im.shape[0]-(out_im.shape[0]*3)//10
    #val im
    out_im_val = out_im[NN:,:,:,:]
    #tr im
    out_im =out_im[:NN,:,:,:]
    #val labels
    label_val =label[NN:,:,:,:]
    #tr labels
    label =label[:NN,:,:,:]

    print('Training Images:', out_im.shape[0])
    print('Cross Validation Images:', out_im_val.shape[0])


    ##blur
    param_blur = {
        'sigma':6
    }
    N=48
    M=48
    ##perspective
    Nper20 = int(N*0.2)
    Mper20 = int(M*0.2)
    pts2m = [np.float32([[0,0],[0,N],[M-Mper20,Nper20],[M-Mper20,N-Nper20]]),
            np.float32([[Mper20,Nper20],[Mper20,N-Nper20],[M,0],[M,N]]),
            np.float32([[0,0],[Mper20,N-Nper20],[M,0],[M-Mper20,N-Nper20]]),
            np.float32([[Mper20,Nper20],[0,N],[M-Mper20,Nper20],[M,N]])]
    param_persp ={
        'len':len(pts2m),
        'pts2m': pts2m
        }
    #optic
    param_pin = {
         'pin_fact': -0.5 
    }
    param_bar = {
        'bar_fact': 0.8
    }
    ##elastic spec:
    param_el = {
        'alpha': N*0.3,
        'sigma': N*0.08,
        'alpha_affine': N*0.08,
        'iteration':2
    }
    ########################################################################
    #Dict for augmentation
    ########################################################################
    augmenters_dict = {
        'rot':[3],
        'blur':[1,param_blur],
        'noise_gauss':[1],
        'noise_sp':[1],
        'scal1':[1],
        'scal2':[1],
        'persp':[param_persp['len'],param_persp],
        'flip_ver':[1],
        'flip_or':[1],
        'scal_int1':[1],
        'scal_int2':[1],
        #'optic_pin':[1,param_pin],
        #'optic_bar':[1,param_bar],
        #'elastic':[param_el['iteration'],param_el]
        }

    foo_list = compose_tr(augmenters_dict)



    n_transf=0
    for key in augmenters_dict.keys():
        n_transf += augmenters_dict[key][0]
    print(n_transf)
    def fun (i,N,M,n_tr,foolist):

        foolambda = lambda a,b,foolist : [x(a,b) for x in foolist]
        sample = np.empty((n_tr,4,N,M))

        k = foolambda(out_im[i,0,:,:],np.dstack((label[i,0,:,:],label[i,1,:,:])),foolist)
        c_ind = 0
        for j in range(len(k)):
            disc = k[j][0].shape
            if len(disc)==3:
                ind = disc[0]
            else:
                ind=1
            sample[c_ind:c_ind+ind,0,:,:]=k[j][0]
            sample[c_ind:c_ind+ind,1:,:,:]=k[j][1]
            c_ind+=ind

        del k
        return sample




    list_samples = Parallel(n_jobs=12,verbose=1,require='sharedmem')(delayed(fun) (i,N,M,n_transf,foo_list) for i in range(out_im.shape[0]))
    list_samples = np.asarray(list_samples)
    rank,batch,ch,N,M = list_samples.shape
    list_samples = list_samples.reshape(rank*batch,ch,N,M)

    out_im = np.vstack((out_im,list_samples[:,0,:,:][:,np.newaxis,:,:]))
    label = np.vstack((label,list_samples[:,1:,:,:]))
    label[label<0.2]=0.0
    label[label>=0.2]=1.0

    print('Training set loaded\nTraining Images:', out_im.shape[0])
    print('Cross Validation Images:', out_im_val.shape[0])

    mean_out_im =  np.mean(np.mean(out_im,axis=3),axis=2)
    mean_out_im_val = np.mean(np.mean(out_im_val,axis=3),axis=2)

    out_im = (out_im-mean_out_im[:,:,np.newaxis,np.newaxis])
    out_im_val = (out_im_val-mean_out_im_val[:,:,np.newaxis,np.newaxis])

    class SimDataset(Dataset):
        def __init__(self,flag=True):
            if flag:
                self.input_images, self.target_masks = out_im[:,:,:,:],label[:,:,:,:]     
            else:
                self.input_images, self.target_masks = out_im_val[:,:,:,:],label_val[:,:,:,:]

        def __len__(self):
            return len(self.input_images)

        def __getitem__(self, idx):        
            image = self.input_images[idx]
            mask = self.target_masks[idx]
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()
            return [image, mask]


    train_set = SimDataset()
    val_set = SimDataset(flag = False)

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    if parallel:
        batch_size =3*batch_size
    else:
        batch_size =batch_size

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    }


    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }

    print(dataset_sizes)

    # model to import
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model_dict[model_n]
    print(10*'-','\n','MODEL',str(model_n))
    ct=0
    if model == 'dense_up':
        #for child in model.children():
        for child in model.children():
            if ct>1 and ct<5:
                print('freezing child', ct)
                for params in child.parameters():
                    params.requires_grad=False
            ct += 1

    if parallel_n:
        model = nn.DataParallel(model,device_ids=[0,1,2])

    model = model.to(device)

    if model_n == 'dense_up' or model_n == 'UNet' :
        single_loss =True
    else:
        single_loss = False

    weights_str=folder_w+model_n+test_folder_str+'_set4_v2.pt'
    print('SAVING in ',weights_str)

    optimizer_ft = optim.Adam(model.parameters(), lr=lr)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6,30], gamma=0.1)

    use_visdom = False


    model,loss_val,_,_ = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=epochs-3,\
                                     dataloaders=dataloaders,device=device,single_loss=single_loss,\
                                     use_visdom=use_visdom)



    if model_n == 'dense_up':
        ct=0
        #for child in model.children():
        for child in model.module.children():
            if ct>1 and ct<5:
                print('freezing child', ct)
                for params in child.parameters():
                    params.requires_grad=True
            ct += 1

        optimizer_ft = optim.Adam(model.parameters(), lr=(0.05*lr))
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)

        model,loss_val,_,_ = train_model(model, optimizer_ft, exp_lr_scheduler,num_epochs=3,\
                                         best_loss=loss_val,dataloaders=dataloaders,device=device,\
                                         single_loss=single_loss,use_visdom=use_visdom)


    qq =  model.module.state_dict()
    #qq = model.state_dict()
    for k, v in qq.items():
        qq[k] = v.cpu()
    torch.save(qq,weights_str)

    ###free mem
    del model,out_im,out_im_val,label,label_val,train_set,val_set