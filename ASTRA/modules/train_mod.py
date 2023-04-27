from visdom import Visdom
#from six.moves import urllib
import time
import copy
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np


#Loss
def dice_loss(prediction, tar, smooth = .5):
    
    pred = prediction.contiguous()
    target = tar.contiguous()    

    intersection = (pred * target).sum(dim=[1,2])
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=[1,2]) + target.sum(dim=[1,2])+ smooth)))


    return loss.mean()


def calc_loss(pred, target, metrics,crit,flag_dice=True, bce_weight=0.5):
   # maxim,_ = torch.max(pred,dim=1)
   # maxim = maxim[:,None,:,:]
   # buff= torch.tensor(pred.data,requires_grad=False,device=device)
   # buff[buff<maxim]=0
   # buff[buff>=maxim]=1
   # 
   #     
   # f1_s = f1_score(torch.flatten(target[:,1,:,:]) , torch.flatten(buff[:,1,:,:]),average='binary')
   # f1_p = f1_score(torch.flatten(target[:,0,:,:]) , torch.flatten(buff[:,0,:,:]),average='binary')
   # 
   # #print(f1_p)
   # if (f1_p>=8):
   #     fig , ax = plt.subplots(1,2)
   #     ax[0].imshow(target[6,0,:,:])  
   #     ax[1].imshow(buff[6,0,:,:])
   #     plt.show()
    bce = crit(pred, target[:,:,:,:])
    
    dice1 = dice_loss(pred[:,0,:,:], target[:,0,:,:])
    dice2 = dice_loss(pred[:,1,:,:], target[:,1,:,:])
    if flag_dice: 
        loss = bce* bce_weight + (dice2+dice1)
        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += (dice2+dice1).data.cpu().numpy() * target.size(0) 
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
        #metrics['f1'] += f1_s* target.size(0)
        #metrics['f1_p'] += f1_p* target.size(0)
        del  bce, dice1, dice2#,maxim,buff, f1_s
    else:
        loss = dice2+dice1
        metrics['bce'] += 0#bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] +=(dice2+dice1).data.cpu().numpy() * target.size(0) 
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    



#train function
def train_model(model, optimizer, scheduler,dataloaders,device,flag_dice=True, num_epochs=25,best_loss=1e10,single_loss=False,use_visdom=True):
   
    if use_visdom:
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
               ),)
    
    crit = nn.BCELoss()
    
    loss_dict=[]
    loss_dict_val=[]

    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                #for param_group in optimizer.param_groups:
                #    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            #print('first', torch.cuda.max_memory_allocated(),torch.cuda.max_memory_cached())
            for inputs, labels in dataloaders[phase]:

                torch.cuda.empty_cache()
                inputs = inputs.to(device)
                labels = labels.to(device)             
                #print('LOADED')   
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    #print('first', torch.cuda.max_memory_allocated(),torch.cuda.max_memory_cached())
                    loss = 0
                    if single_loss:
                        loss =calc_loss(pred = outputs,target=labels,metrics = metrics,crit=crit,flag_dice=flag_dice)
                    else:
                        for output in outputs:
                            loss += calc_loss(pred = output,target=labels,metrics = metrics,crit=crit)                 
                        loss /= len(outputs)
#
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            #print('learning_step')
            ###ch
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
            ###ch
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
                best_dice = metrics['dice'] / epoch_samples

                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since

        if use_visdom:
            vis.line(
                X=np.column_stack((np.arange(len(loss_dict)),np.arange(len(loss_dict)))),
                Y=np.column_stack((np.asarray(loss_dict),np.asarray(loss_dict_val))),
                win=loss_vis,
                update='insert')

            vis.update_window_opts(
                     win=loss_vis,
                     opts=dict(
                         xtickmax=len(loss_dict),        
                         legend=['train','val'],         
                         title = 'Loss',                 
                         ytickmin=0,
                         xtickmin=0,
                         ytickmax=10,                    
                         ytickstep=1,                    
                       ),)
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    del best_model_wts
    ##mod
    loss_dict = np.asarray(loss_dict)
    loss_dict_val = np.asarray(loss_dict_val)
    return model, best_loss,loss_dict,loss_dict_val
