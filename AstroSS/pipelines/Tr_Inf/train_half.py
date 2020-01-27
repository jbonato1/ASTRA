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
import augm_samples as a_s
from apex import amp
from apex.fp16_utils import *

#amp_handle = amp.init(enabled=True)

assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
set_dir='/home/jbonato/Documents/U-Net/set/train'

items = os.listdir(set_dir)
print(len(items))
counter = 0
im = np.empty((256,256),dtype = np.float32 )
for item in items:
    path_to_item = os.path.join(set_dir,item)
    filename, file_extension = os.path.splitext(path_to_item)
    if os.path.isfile(path_to_item) and file_extension == '.tif' :
        #print(filename)
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
        back = np.ones((256,256),dtype=np.int64)-proc_mask-soma_mask
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

def fun (i):
    N=256
    M=256
    
    im_cont = np.empty((16,1,N,M))
    mask_cont = np.empty((16,3,N,M))
    sample = np.empty((16,4,N,M))

    im_cont, mask_cont =  a_s.aug(im = out_im[i,0,:,:], mask =np.dstack((label[i,0,:,:],label[i,1,:,:])))
    
    sample[:,0,:,:]=im_cont[:,0,:,:]
    sample[:,1:,:,:]=mask_cont
    del im_cont,mask_cont
    return sample

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

set_dir='/home/jbonato/Documents/U-Net/set/val'

items = os.listdir(set_dir)
print(len(items))
counter = 0
im = np.empty((256,256),dtype = np.float32 )
for item in items:
    path_to_item = os.path.join(set_dir,item)
    filename, file_extension = os.path.splitext(path_to_item)
    if os.path.isfile(path_to_item) and file_extension == '.tif' :
        #print(filename)
        im = io.imread(path_to_item)
        if counter == 0: 
            out_im_val = im
            out_im_val = out_im_val[np.newaxis,:,:]
        else:
            out_im_val = np.concatenate((out_im_val,im[np.newaxis,:,:]),axis=0)
        dset= h5py.File(filename+'.hdf5','r') 
        proc_mask =  np.asarray(dset['Values'])
        soma_mask =  np.asarray(dset['Values_soma'])
        proc_mask[np.where(proc_mask==soma_mask)]=0
        back = np.ones((256,256),dtype=np.int64)-proc_mask-soma_mask
        back[back<0]=0
        mask = np.concatenate((proc_mask[:,:,np.newaxis],soma_mask[:,:,np.newaxis],back[:,:,np.newaxis]),axis=2).astype(np.float32)
        
        if counter ==0:
            label_val = mask
            label_val = label_val[np.newaxis,:,:,:]
        else:
            label_val = np.concatenate((label_val,mask[np.newaxis,:,:,:]),axis=0)
        counter+=1

label_val = np.swapaxes(label_val,1,3)
label_val = np.swapaxes(label_val,2,3)
out_im_val = out_im_val[:,np.newaxis,:,:].astype(np.float32)
print('im shape:',out_im_val.shape)
print('label_val shape', label_val.shape)

def fun_val (i):
    N=256
    M=256
    
    im_cont = np.empty((16,1,N,M))
    mask_cont = np.empty((16,3,N,M))
    sample = np.empty((16,4,N,M))

    im_cont, mask_cont =  a_s.aug(im = out_im[i,0,:,:], mask =np.dstack((label_val[i,0,:,:],label_val[i,1,:,:])))
    
    sample[:,0,:,:]=im_cont[:,0,:,:]
    sample[:,1:,:,:]=mask_cont
    del im_cont,mask_cont
    return sample

list_samples = Parallel(n_jobs=12,verbose=1,require='sharedmem')(delayed(fun_val) (i) for i in range(out_im_val.shape[0]))
list_samples = np.asarray(list_samples)
print(list_samples.shape)
rank,batch,ch,N,M = list_samples.shape
list_samples = list_samples.reshape(rank*batch,ch,N,M)

out_im_val = np.vstack((out_im_val,list_samples[:,0,:,:][:,np.newaxis,:,:]))
label_val = np.vstack((label_val,list_samples[:,1:,:,:]))
label_val[label_val<0.2]=0.0
label_val[label_val>=0.2]=1.0



# n = input('Would you like to see a sample and its mask? type a number between 0 and '+str(out_im.shape[0])+', N otherwise\n')
# if n !='N':
#     n=int(n)
# print(np.mean(out_im[n,0,:,:]),np.std(out_im[n,0,:,:]))

# fig, ax = plt.subplots(figsize=(20, 20), ncols=4)
# ax[0].imshow(out_im[n,0,:,:])
# ax[1].imshow(label[n,0,:,:])
# ax[2].imshow(label[n,1,:,:])
# b=out_im[n,0,:,:]
# A = np.zeros((256,256,3),dtype=np.float32)
# maximum = 255/np.amax(b) #np.amax(conv_im)//250
# c = b*maximum

# A[:,:,0] = c
# A[:,:,2] = 255*label[n,1,:,:]+255*label[n,0,:,:]
# ax[3].imshow(A)
# plt.show()
#///////////////////////////////////////// Visdom connection
from visdom import Visdom
import numpy as np
from six.moves import urllib
import time
vis = Visdom(port=8097, server="http://localhost")
Y = np.arange(1)
X = np.arange(1)
loss_vis = vis.line(
    X=np.column_stack((X,X)),
    Y=np.column_stack((Y,Y)),
    opts=dict(
        legend=['train','val'],
        xtickmin=0,
        xtickmax=50,
        xtickstep=10,
        ytickmin=0,
        ytickmax=10,
        ytickstep=1,
        title = 'Loss'
       
    ),
)

#///////////////////////////////////////// Here start the pytorch code

# N_train=out_im.shape[0]-(out_im.shape[0]*2)//10
# print('training: ',out_im.shape[0]-N_train,' test: ',N_train)


class SimDataset(Dataset):
    def __init__(self, transform=None,flag=True):
        if flag:
            self.input_images, self.target_masks = out_im[:,:,:,:],label[:,:,:,:]     
        else:
            self.input_images, self.target_masks = out_im_val[:,:,:,:],label_val[:,:,:,:]
        self.transform = transform
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):        
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()
        return [image, mask]

trans = True

train_set = SimDataset(transform = trans)
val_set = SimDataset(transform = trans, flag = False)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 12

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
    'val': DataLoader(val_set, batch_size=12, shuffle=True, num_workers=0)
}


dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

print(dataset_sizes)
# model to import
#from model.Unet_Res152Back_nest import nestedUNetUp152
#from model.Unet_IncResV2_nest import IncResV2
from model.Unet_nest_5 import nestedUNetUp5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = IncResV2(3)
model = nestedUNetUp5(3)

# ct=0
# for child in model.children():
#     if ct ==5 or ct==3 or ct==4 :
#         print('freezing child ',ct)
#         for param in child.parameters():
#             param.requires_grad = False
#     ct +=1
model = model.to(device)
model = model.half()

#summary(model, input_size=(1, 256, 256),batch_size=1)

#Loss
def dice_loss(prediction, tar, smooth = 1.):
    
    pred = prediction.contiguous()
    target = tar.contiguous()    

    intersection = (pred * target).sum(dim=[1,2])
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=[1,2]) + target.sum(dim=[1,2])+ smooth)))


    return loss.mean()

crit = nn.BCEWithLogitsLoss()

loss_dict=[]
loss_dict_val=[]

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = crit(pred.float(), target.float())
    dice1 = dice_loss(pred[:,0,:,:].float(), target[:,0,:,:].float())
    dice2 = dice_loss(pred[:,1,:,:].float(), target[:,1,:,:].float())
    
    loss = (bce * bce_weight + dice1+dice2)#*0.01
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += (dice1+dice2).data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    loss = loss*128

    return loss.half()

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

#train function
def train_model(model, optimizer, scheduler, num_epochs=25,best_loss=1e10):
    best_model_wts = copy.deepcopy(model.state_dict())
    
    
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
           
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.half()
                labels = labels.half()

                inputs = inputs.to(device)
                labels = labels.to(device)             
                #print('cached',torch.cuda.memory_cached(),'alloc',torch.cuda.memory_allocated())
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    
                    loss = 0
                    for output in outputs:
                        loss += (calc_loss(output, labels, metrics))
                        print(loss)
                    loss /= len(outputs)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                        loss.backward()
                        optimizer.step()
            
                # statistics
                epoch_samples += inputs.size(0)
            
            if phase == 'train':
                loss_dict.append(metrics['loss'] / epoch_samples)
                
            elif phase == 'val':
                loss_dict_val.append(metrics['loss'] / epoch_samples)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        vis.line(
                X=np.column_stack((np.arange(len(loss_dict)),np.arange(len(loss_dict)))),
                Y=np.column_stack((np.asarray(loss_dict),np.asarray(loss_dict_val))),
                win=loss_vis,
                opts=dict(
                    xtickmax=len(loss_dict),
                    legend=['train','val'],
                    title = 'Loss',
                    ytickmin=0,
                    xtickmin=0,
                    ytickmax=10,
                    ytickstep=1,
                    ),
                update='insert')
        torch.cuda.empty_cache()
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss

#////////////////////////////Training
weights_str='/home/jbonato/Documents/U-Net/weights/IncResV2q_h.pt'
weights_str_f='/home/jbonato/Documents/U-Net/weights/IncResV2_fine.pt'
optimizer_ft = optim.Adam(model.parameters(), lr=1e-6)


exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

model,loss_val = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)
torch.save(model.state_dict(),weights_str)
# del model,dataloaders

# batch_size = 6

# dataloaders = {
#     'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
#     'val': DataLoader(val_set, batch_size=2, shuffle=True, num_workers=0)
# }


# model_fine = IncResV2(3)
# model_fine = model_fine.to(device)
# summary(model_fine, input_size=(1, 256, 256),batch_size=1)
# model_fine.load_state_dict(torch.load(weights_str))

# optimizer_ft = optim.Adam(model_fine.parameters(), lr=1e-5)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
# model_fine,_ = train_model(model_fine, optimizer_ft, exp_lr_scheduler, num_epochs=10,best_loss=loss_val)
# torch.save(model_fine.state_dict(),weights_str_f)
