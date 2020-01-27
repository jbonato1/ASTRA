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
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from aug_samples_single96 import * 
from sklearn.metrics import f1_score
import argparse
from model.Unet_nest_5 import nestedUNetUp5 as Unet_nest_5
from model.Unet_nest_5 import nestedUNetUp5_dense as Unet_nest_5_dense
from model.dense_up import dense_up
from model.Unet import UNet
import torch
from torchsummary import summary
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

model_dict={'Unet_nest_5': Unet_nest_5(3),
            'nestedUNetUp5_dense': Unet_nest_5_dense(3),
            'UNet':UNet(3),
            'dense_up': dense_up(3)
        }


parser = argparse.ArgumentParser(description ='Training set up' )

parser.add_argument('-m','--model',default ='Unet_nest_5',choices=['UNet','Unet_nest_5','nestedUNetUp5_dense','dense_up'], help='model name (default:Unet_nest_5)')

parser.add_argument('-d','--data', metavar='DIR', default='/media/DATA/jbonato/segm_project/set/train_single',help='path to dataset')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',help='number of data loading workers (default: 1)')

parser.add_argument('-ep','--epochs', default=150, type=int, metavar='N',help='number of total epochs to run (default: 150)')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch_size', default=110, type=int,metavar='N', help='mini-batch size (default: 110)')

parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,metavar='LR', help='initial learning rate (default: 1e-4)')

parser.add_argument('-parallel',action='store_true',default=False,dest ='parallel',help='Multi gpu devices setting, default 1 gpu use this switch to use multigpu')


parser.add_argument('-fov','--test_folder_str',help='test folder which has to be excluded from training')

args = parser.parse_args()




test_folder_str = args.test_folder_str
if len(test_folder_str)==1:
    test_folder_str='00'+test_folder_str
else:
    test_folder_str='0'+test_folder_str
set_dir= args.data
items = os.listdir(set_dir)
print(len(items))
cnt=0
test=[]
for i in items:
    if 'SMALL_'+test_folder_str in i:
        print(i)
        test.append(i)
for i in test:
    items.remove(i)

print(len(items))

counter = 0
counter2 = 0
im = np.empty((96,96),dtype = np.float32 )
for item in items:
    path_to_item = os.path.join(set_dir,item)
    filename, file_extension = os.path.splitext(path_to_item)
    if os.path.isfile(path_to_item) and file_extension == '.tif' and filename[-6:]=='nf_enh' and int(item[6:9])<26 :
        im = io.imread(path_to_item)
        if counter == 0: 
            out_im = im
            out_im = out_im[np.newaxis,:,:]
        else:
            out_im = np.concatenate((out_im,im[np.newaxis,:,:]),axis=0)
        dset= h5py.File(filename+'.hdf5','r') 
        proc_mask =  np.asarray(dset['Values'])
        soma_mask =  np.asarray(dset['Values_soma'])
        proc_mask[np.where(proc_mask==soma_mask)]=0
        back = np.ones((96,96),dtype=np.int64)-proc_mask-soma_mask
        back[back<0]=0
        mask = np.concatenate((proc_mask[:,:,np.newaxis],soma_mask[:,:,np.newaxis],back[:,:,np.newaxis]),axis=2).astype(np.float32)
        
        if counter ==0:
            label = mask
            label = label[np.newaxis,:,:,:]
        else:
            label = np.concatenate((label,mask[np.newaxis,:,:,:]),axis=0)
        counter+=1

label = np.swapaxes(label,1,3)
label = np.swapaxes(label,2,3)

out_im = out_im[:,np.newaxis,:,:].astype(np.float32)

print('im shape:',out_im.shape)
print('label shape', label.shape)

NN = out_im.shape[0]-(out_im.shape[0]*3)//10
print('NN',NN)
#val im
out_im_val = out_im[NN:,:,:,:] #out_im2[N2:,:,:,:]#

#tr im
out_im =out_im[:NN,:,:,:]#out_im2[:N2,:,:,:]# 

#val labels
label_val =label[NN:,:,:,:]# label2[N2:,:,:,:]#

#tr labels
label =label[:NN,:,:,:]# label2[:N2,:,:,:]#


def fun (i):
    N=96
    M=96
    
    im_cont = np.empty((20,1,N,M))
    mask_cont = np.empty((20,3,N,M))

    sample = np.empty((20,4,N,M))

    im_cont, mask_cont =  aug(im = out_im[i,0,:,:], mask =np.dstack((label[i,0,:,:],label[i,1,:,:])))
    
    sample[:,0,:,:]=im_cont[:,0,:,:]
    sample[:,1:,:,:]=mask_cont
    del im_cont,mask_cont
    return sample

#q =fun(0)


list_samples = Parallel(n_jobs=12,verbose=1,require='sharedmem')(delayed(fun) (i) for i in range(out_im.shape[0]))
list_samples = np.asarray(list_samples)
print(list_samples.shape)
rank,batch,ch,N,M = list_samples.shape
list_samples = list_samples.reshape(rank*batch,ch,N,M)

out_im = np.vstack((out_im,list_samples[:,0,:,:][:,np.newaxis,:,:]))
label = np.vstack((label,list_samples[:,1:,:,:]))
label[label<0.2]=0.0
label[label>=0.2]=1.0
print('training set loaded')

#///////////////////////////////////////// Here start the pytorch code
mean_out_im =  np.mean(np.mean(out_im,axis=3),axis=2)
#print(mean_out_im.shape)
mean_out_im_val = np.mean(np.mean(out_im_val,axis=3),axis=2)
std_out_im = np.std(np.std(out_im,axis=3),axis=2)
std_out_im_val = np.std(np.std(out_im_val,axis=3),axis=2)

out_im = (out_im-mean_out_im[:,:,np.newaxis,np.newaxis])#/std_out_im[:,:,np.newaxis,np.newaxis]
out_im_val = (out_im_val-mean_out_im_val[:,:,np.newaxis,np.newaxis])#/std_out_im_val[:,:,np.newaxis,np.newaxis]
#print(np.mean(out_im))
#input('')

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

trans = True

train_set = SimDataset()
val_set = SimDataset(flag = False)

image_datasets = {
    'train': train_set, 'val': val_set
}
print(args.parallel)
if args.parallel:
    batch_size =3*args.batch_size
else:
    batch_size =args.batch_size

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1),#args.workers),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)#args.workers)
}


dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)

# model to import
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model_dict[args.model]
print(10*'-','\n','MODEL',str(args.model))
ct=0
if args.model == 'dense_up':
    for child in model.children():
        if ct>1 and ct<5:
            print('freezing child', ct)
            for params in child.parameters():
                params.requires_grad=False
        ct += 1

if args.parallel:
    model = nn.DataParallel(model,device_ids=[0,1,2])

model = model.to(device)

if args.model == 'dense_up' or args.model == 'UNet' :
    single_loss =True
else:
    single_loss = False

#////////////////////////////Training
from train_mod import train_model
weights_str='/media/DATA/jbonato/segm_project/weights/no_pproc/'+args.model+test_folder_str+'nf_enh.pt'
optimizer_ft = optim.Adam(model.parameters(), lr=args.learning_rate)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6,30], gamma=0.1)

model,loss_val = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs-3,dataloaders=dataloaders,device=device,single_loss=single_loss,use_visdom=False)#
pret=True

if args.model == 'dense_up':
    ct=0
    for child in model.module.children():
        if ct>1 and ct<5:
            print('freezing child', ct)
            for params in child.parameters():
                params.requires_grad=True
        ct += 1

    optimizer_ft = optim.Adam(model.parameters(), lr=(0.05*args.learning_rate))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=200, gamma=0.1)
    model,loss_val = train_model(model, optimizer_ft, exp_lr_scheduler,num_epochs=3,best_loss=loss_val,dataloaders=dataloaders,device=device,single_loss=single_loss,use_visdom=False)
    qq =  model.module.state_dict()
    for k, v in qq.items():
        qq[k] = v.cpu()


    torch.save(qq,weights_str)

del qq,model,out_im,out_im_val,label,label_val,train_set,val_set
