import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import h5py

def create_bb_coord(soma_mask,BB_dim):
    #use even BB_dim
    ################################################
    #    0------------------------>x
    #    |
    #    |
    #    |     MATRICES
    #    |
    #    y
    ################################################
    N,M= soma_mask.shape
    soma = np.empty_like(soma_mask)
    soma = soma_mask.copy()
    soma[soma>0.1]=255
    BBh = BB_dim//2
   
    # convert the grayscale image to binary image
    _,thresh = cv2.threshold(np.uint8(soma),127,255,0)
    
    # find contours in the binary image
    contours, _= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    #list of array with the coordinate
    coord_list=[]
    filt_im_zone = np.zeros((N,M,len(contours)))
    cnt=0    
    for c in contours:
        # compute the center of the contour
        filt_= np.zeros((N,M))
        Mom = cv2.moments(c)
        cX = int(Mom["m10"] / Mom["m00"])
        cY = int(Mom["m01"] / Mom["m00"])
        #print(cX,cY)
        cv2.circle(filt_,(cX,cY),43,255,thickness = -1,lineType=8)
        filt_im_zone[:,:,cnt]=filt_
        cnt+=1

        casex=2
        casey=2
        if cX-BBh<0:
            casex=0
        elif cX+BBh>M:
            casex=1
        if cY-BBh<0:
            casey=0
        elif cY+BBh>N:
            casey=1
        #x
        if casex==2:
            c1x=cX-BBh
            c2x=cX+BBh
        elif casex==0:
            c1x=0
            c2x=BB_dim
        else:
            c1x=M-BB_dim
            c2x=M
        #y
        if casey==2:
            c1y=cY-BBh
            c2y=cY+BBh
        elif casey==0:
            c1y=0
            c2y=BB_dim
        else:
            c1y=N-BB_dim
            c2y=N
        
        coord = np.array([c1x,c1y,c2x,c2y])
        #print(coord)
        coord_list.append(coord)
    filt_im_zone[filt_im_zone>0]=1
    #print(len(coord_list))
    return coord_list,filt_im_zone

############################################### TRAINING PP

def filtering(stack,th1_p,th2_p):
    T,N,M = stack.shape
    percent = np.percentile(stack.reshape(T*N*M),90)
    stack_cp = stack.copy()
    stack_cp[stack<percent]=0
    stack_cp[stack>=percent]=1

    sum_= np.sum(stack_cp,axis=0)
    th1 = round(T*th1_p)
    sum_[sum_<th1]=0
    sum_[sum_>=th1]=1
    if th2_p ==None:
        return sum_
    else:
        fill_s= np.zeros_like(sum_)
        fill_s[sum_==0]=1
        stack_f=stack.copy()*fill_s
        percent = np.percentile(stack_f.reshape(T*N*M),90)
        stack_f[stack_f<percent]=0
        stack_f[stack_f>=percent]=1

        sum_p =np.sum(stack_f,axis=0)
        th2 = round(T*th2_p) 
        sum_p[sum_p<th2]=0
        sum_p[sum_p>=th2]=1
        sum_p+=sum_
        return sum_p

def save_im(im,stack,mask_soma,mask_proc,save_folder,item,BB_dim,th1_p,th2_p,pad=5):
    T,N,M = stack.shape
    new_mask = np.zeros((2,N,M))
    
    coord_list,filt_im_zone = create_bb_coord(mask_soma,BB_dim)
    
    
    im_to_crop = np.empty_like(im)
    stack_to_crop = np.empty_like(stack)
    crop_stack = np.empty((T,BB_dim,BB_dim))
    crop_im = np.empty((BB_dim,BB_dim))
    crop_mask = np.empty((2,BB_dim,BB_dim))
    counter=0
    cnt=0
    for coord in coord_list:
        ################## modified for test on preprocessing
        im_to_crop = im.copy()
        stack_to_crop = stack.copy()
        mask_soma_c = mask_soma.copy()
        mask_proc_c = mask_proc.copy()

        cnt+=1

        crop_stack = stack_to_crop[:,coord[1]:coord[3],coord[0]:coord[2]]
        crop_im =im_to_crop[coord[1]:coord[3],coord[0]:coord[2]]
        crop_mask[0,:,:] = mask_soma_c[coord[1]:coord[3],coord[0]:coord[2]]
        crop_mask[1,:,:] = mask_proc_c[coord[1]:coord[3],coord[0]:coord[2]]

        ########################NOT FILTERED enhanced
        io.imsave(save_folder+'_single_'+ str(counter) +'nf_enh.tif',(np.pad(crop_im,pad,'constant')).astype(np.float32))
        crop_mask_ = np.pad(crop_mask,((0,0),(pad,pad),(pad,pad)),'constant')
        
        with h5py.File(save_folder+'_single_'+ str(counter) +'nf_enh.hdf5','w') as f:
            dset = f.create_dataset('Values',data=crop_mask_[1,:,:])
            dset2 = f.create_dataset('Values_soma',data=crop_mask_[0,:,:])
        
        #########################FILTERED 
        #filt stack
        crop_mask_filt = filtering(crop_stack,th1_p,th2_p)

        #filt im and mask
        crop_im = crop_im*crop_mask_filt
        crop_mask = crop_mask*crop_mask_filt
        new_mask[:,coord[1]:coord[3],coord[0]:coord[2]] += crop_mask
        
        io.imsave(save_folder+'_single_'+ str(counter) +'.tif',(np.pad(crop_im,pad,'constant')).astype(np.float32))

        crop_mask_p = np.pad(crop_mask,((0,0),(pad,pad),(pad,pad)),'constant')
        with h5py.File(save_folder+'_single_'+ str(counter) +'.hdf5','w') as f:
            dset = f.create_dataset('Values',data=crop_mask_p[1,:,:])
            dset2 = f.create_dataset('Values_soma',data=crop_mask_p[0,:,:])
        counter+=1
    new_mask[new_mask>1]=1
    return new_mask[1,:,:], new_mask[0,:,:]

def save_im_l(im,stack,mask_soma,save_folder,item,BB_dim,th1_p,th2_p,pad=5):
    T,N,M = stack.shape
    new_mask = np.zeros((1,N,M))
    
    coord_list,filt_im_zone = create_bb_coord(mask_soma,BB_dim)
    
    
    im_to_crop = np.empty_like(im)
    stack_to_crop = np.empty_like(stack)
    crop_stack = np.empty((T,BB_dim,BB_dim))
    crop_im = np.empty((BB_dim,BB_dim))
    crop_mask = np.empty((1,BB_dim,BB_dim))
    counter=0
    cnt=0
    for coord in coord_list:
        ################## modified for test on preprocessing
        im_to_crop = im.copy()
        stack_to_crop = stack.copy()
        mask_soma_c = mask_soma.copy()
        

        cnt+=1

        crop_stack = stack_to_crop[:,coord[1]:coord[3],coord[0]:coord[2]]
        crop_im =im_to_crop[coord[1]:coord[3],coord[0]:coord[2]]
        crop_mask[0,:,:] = mask_soma_c[coord[1]:coord[3],coord[0]:coord[2]]
        

        ########################NOT FILTERED enhanced
        io.imsave(save_folder+'_single_'+ str(counter) +'nf_enh.tif',(np.pad(crop_im,pad,'constant')).astype(np.float32))
        crop_mask_ = np.pad(crop_mask,((0,0),(pad,pad),(pad,pad)),'constant')
        
        with h5py.File(save_folder+'_single_'+ str(counter) +'nf_enh.hdf5','w') as f:
            
            dset2 = f.create_dataset('Values_soma',data=crop_mask_[0,:,:])
        
        #########################FILTERED 
        #filt stack
        crop_mask_filt = filtering(crop_stack,th1_p,th2_p)

        #filt im and mask
        crop_im = crop_im*crop_mask_filt
        crop_mask = crop_mask*crop_mask_filt
        new_mask[:,coord[1]:coord[3],coord[0]:coord[2]] += crop_mask
        
        io.imsave(save_folder+'_single_'+ str(counter) +'.tif',(np.pad(crop_im,pad,'constant')).astype(np.float32))

        crop_mask_p = np.pad(crop_mask,((0,0),(pad,pad),(pad,pad)),'constant')
        with h5py.File(save_folder+'_single_'+ str(counter) +'.hdf5','w') as f:
            
            dset2 = f.create_dataset('Values_soma',data=crop_mask_p[0,:,:])
        counter+=1
    new_mask[new_mask>1]=1
    return  new_mask[0,:,:]

################################################# SPATIAL PP
class spatial_pp():
    def __init__(self,file_path):
        self.stack = io.imread(file_path)
        
    def create_img(self,T_st=0):

        _,N,M = self.stack.shape
        if T_st==0:
            self.stack = self.stack.astype(np.float32)
        else:
            self.stack = self.stack[:T_st,:,:].astype(np.float32)
            
        stack_new = self.stack.copy()

        T,N,M = stack_new.shape
        for t in range(T):
            stack_new[t,:,:] = stack_new[t,:,:] - np.percentile(stack_new[t,:,:],10)
            stack_new[t,:,:][stack_new[t,:,:]<0]=0 

        conv_im = np.median(stack_new,axis=0)       
        maximum = 65535/np.amax(conv_im) 
        conv_im =conv_im.astype(np.float32)*maximum   


        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        cl1 = clahe.apply(np.uint16(conv_im))       


        kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])

        denoise = np.uint16(cl1)
        image = cv2.filter2D(np.uint16(cl1), -1, kernel_sharpening)
        
        return stack_new,image

    def create_img_d2(self):
        
        T,_,_ = self.stack.shape
        stack_new = np.empty((T,256,256))
        a = 0
        for t in range(T):
            stack_new[t,:,:] = cv2.resize(self.stack[t,:,:],(256,256),interpolation=cv2.INTER_AREA)
            
        median = np.median(stack_new,axis=0)
        median = cv2.GaussianBlur(np.uint16(median),(3,3),0)
        maximum = 65535/np.amax(median)
        median = median.astype(np.float32)*maximum
        
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16,16))
        cl1 = clahe.apply(np.uint16(median))

        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                      [-1,-1,-1]])
        image = cv2.filter2D(np.uint16(cl1), -1, kernel_sharpening)
        
        return stack_new,image
    
    def create_img_large(self):

        self.stack = self.stack[:,2:-3,2:-3]

        T,N,M = self.stack.shape
        self.stack = self.stack.astype(np.float32)
        

        stack_new = np.empty_like(self.stack)
        for i in range(N):
            for j in range(M):
                
                stack_new[:,i,j]= np.convolve(self.stack[:,i,j],1/5*np.ones((5,)),'same')

        conv_im = np.mean(stack_new,axis=0)   
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(5,5))
        cl1 = clahe.apply(np.uint16(conv_im))       

        return stack_new,cl1
    
    def create_img_d4(self,T_st=0):

        _,N,M = self.stack.shape
        if T_st==0:
            self.stack = self.stack.astype(np.float32)
        else:
            self.stack = self.stack[:T_st,:,:].astype(np.float32)
            
        stack_new = self.stack.copy()

        T,N,M = stack_new.shape
        for t in range(T):
            stack_new[t,:,:] = stack_new[t,:,:] - np.percentile(stack_new[t,:,:],10)
            stack_new[t,:,:][stack_new[t,:,:]<0]=0 

        conv_im = np.median(stack_new,axis=0)       
        maximum = 65535/np.amax(conv_im) 
        conv_im =conv_im.astype(np.float32)*maximum   


        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
        cl1 = clahe.apply(np.uint16(conv_im))       


        kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])

        denoise = np.uint16(cl1)
        
        return stack_new,denoise


###################################################### Inference PP

class filt_im(spatial_pp):

    def __init__(self, file_path, mask_soma,BB_dim,filt_meth='std'):
        super().__init__(file_path)
        self.BB_dim = BB_dim   
        #self.stack = stack
        self.mask_soma = mask_soma
        self.coord_list,self.filt_im_zone = create_bb_coord(mask_soma,BB_dim)
        self.filt_meth = filt_meth
        assert self.filt_meth =='std' or self.filt_meth=='ad_hoc','Undefined local activity filter'
        
    def get_instances(self):
        return self.coord_list

    
    def filtering(self,stack,th1_p=0.25,th2_p=0.1):
        
        T,N,M = stack.shape
        
        if T>1000:
            
            T_pr = 700
        else:
            
            T_pr = T
            
        percent = np.percentile(stack[:T,:,:].reshape(T*N*M),90)
        stack_cp = stack.copy()
        stack_cp[stack<percent]=0
        stack_cp[stack>=percent]=1

        sum_= np.sum(stack_cp,axis=0)
        th1 = round(T*th1_p)
        sum_[sum_<th1]=0
        sum_[sum_>=th1]=1


        fill_s= np.zeros_like(sum_)
        fill_s[sum_==0]=1
        stack_f=stack.copy()*fill_s
        percent = np.percentile(stack_f[:T,:,:].reshape(T*N*M),90)
        stack_f[stack_f<percent]=0
        stack_f[stack_f>=percent]=1

        sum_p =np.sum(stack_f,axis=0)
        th2 = round(T*th2_p) 
        sum_p[sum_p<th2]=0
        sum_p[sum_p>=th2]=1    
        sum_p+=sum_

        return sum_p
    
    def filter_hoc(self,stack,perc=90,th=0.25):

        T,N,M = stack.shape
        perc = np.percentile(stack.reshape(T*N*M),perc)
        mask=stack.copy()
        mask[mask<perc]=0
        mask[mask>=perc]=1
        mask = np.sum(mask,axis=0)
        th2 = round(T*th) 
        mask[mask<th2]=0
        mask[mask>=th2]=1

        return mask
    
    def save_im(self,pad=5,stack=None,case=1):
        if not(stack is None):
            self.stack = stack
        
        dim = self.BB_dim
        out_dim = self.BB_dim + 2*pad
        #print(out_dim)
        T,N,M = self.stack.shape
        if case==1:
            _,im = self.create_img()
        elif case==2:
            _,im = self.create_img_d2()
        elif case==3:
            _,im = self.create_img_large()
            
        im_to_crop = np.empty_like(im)
        stack_to_crop = np.empty_like(self.stack)
        crop_stack = np.empty((T,dim,dim))
        crop_im = np.empty((dim,dim))
        crop_im_pad = np.empty((out_dim,out_dim))
        crop_mask = np.empty((2,dim,dim))
        out_stack = np.empty((len(self.coord_list),2,out_dim,out_dim))
        act_filt =np.zeros((N,M))
        
        counter=0
        for coord in self.coord_list:
            im_to_crop = im.copy()
            
          
            stack_to_crop = self.stack.copy()
            
            mask_soma_c = self.mask_soma.copy()
    
            crop_im = im_to_crop[coord[1]:coord[3],coord[0]:coord[2]]
            crop_stack = stack_to_crop[:,coord[1]:coord[3],coord[0]:coord[2]]
           
            #filt stack
            if self.filt_meth == 'std':
                crop_mask_filt = self.filtering(crop_stack)
            elif self.filt_meth == 'ad_hoc':
                crop_mask_filt = self.filter_hoc(crop_stack)
            
                             
            act_filt[coord[1]:coord[3],coord[0]:coord[2]] += crop_mask_filt
            #filt im and mask
            ## for non filtered version this line must be commented
            crop_im = crop_im*crop_mask_filt
            
            crop_im_pad = np.pad(crop_im,pad,'constant').astype(np.float32)
        
            crop_im_pad -= np.mean(crop_im_pad)
            
            out_stack[counter,0,:,:] = crop_im_pad
            out_stack[counter,1,:,:] = np.pad(crop_mask_filt,pad,'constant').astype(np.float32) 
            counter+=1
            
        act_filt[act_filt>1]=1
        #print('AA',np.sum(act_filt))
        return out_stack,act_filt
    

class tune_th(filt_im):
    
    def __init__(self, stack,mask,BB_dim,filt_meth='std'):
        self.BB_dim = BB_dim   
        self.stack = stack
        self.mask = mask
        self.coord_list,self.filt_im_zone = create_bb_coord(mask[:,:,1],BB_dim)
        self.filt_meth = filt_meth
        assert self.filt_meth =='std' or self.filt_meth=='ad_hoc','Undefined local activity filter'
    
    def save_im(self):
        dim = self.BB_dim
        T,N,M = self.stack.shape
        
        
        stack_to_crop = np.empty_like(self.stack)
        crop_stack = np.empty((T,dim,dim))
        crop_mask = np.empty((2,dim,dim))
        act_filt =np.zeros((N,M))
        
        TP_err=[]
        if self.filt_meth == 'std':
                for th1_p,th2_p in zip([0.3,0.25,0.20,0.15],[0.15,0.1,0.07,0.05]):
                    print('THRESH',th1_p,th2_p)
                    for coord in self.coord_list[:20]:
          
                        stack_to_crop = self.stack.copy()
                        crop_stack = stack_to_crop[:,coord[1]:coord[3],coord[0]:coord[2]]        
                        crop_mask_filt = self.filtering(crop_stack,th1_p,th2_p)
                        act_filt[coord[1]:coord[3],coord[0]:coord[2]] += crop_mask_filt
                        
                    act_filt[act_filt>1]=1
                    soma_err = 100*np.sum(self.mask[:,:,1]-self.mask[:,:,1]*act_filt)/np.sum(self.mask[:,:,1])
                    proc_err = 100*np.sum(self.mask[:,:,0]-self.mask[:,:,0]*act_filt)/np.sum(self.mask[:,:,0])
                    TP_err.append(np.asarray([soma_err,proc_err]))
                                  
        elif self.filt_meth == 'ad_hoc':
                for th1_p in [0.3,0.25,0.20,0.15]:
                    
                    for coord in self.coord_list:
                        stack_to_crop = self.stack.copy()
                        crop_stack = stack_to_crop[:,coord[1]:coord[3],coord[0]:coord[2]]        
                        crop_mask_filt = self.filtering_hoc(crop_stack,th1_p)
                        act_filt[coord[1]:coord[3],coord[0]:coord[2]] += crop_mask_filt
                        
                    act_filt[act_filt>1]=1
                    TP_err.append(100*np.sum(self.mask-self.mask*act_filt)/np.sum(self.mask))
       
        TP_err = np.asarray(TP_err)
        return TP_err

