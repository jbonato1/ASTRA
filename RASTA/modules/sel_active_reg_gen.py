import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from sklearn.externals.joblib import Parallel, delayed
from skimage.restoration import denoise_nl_means,estimate_sigma
from scipy import signal
from joblib import Parallel, delayed
import os
import h5py
from numba import cuda,float32,uint16,float64  

class ThScal():
    def __init__(self,stack):
        kernel = np.ones((50,50),np.float32)/(50*50)
        density = cv2.filter2D(np.median(stack,axis=0),-1,kernel)
        self.density = density/np.amax(density)
        
    def ThMat(self,ff,th_):
        N,M = ff.shape
        mask_th = th_*np.ones((N,M))
        
        cnt=1
        for i in [0.6,0.4,0.2,0]:
            buff = self.density.copy()
            buff[buff<i]=0
            buff[buff>=i+0.2]=0
            buff[buff>0]=1
            mask_th-=buff*(th_*0.05*cnt)
            cnt+=1

        ff[ff<mask_th]=0
        ff[ff>0]=1
        return ff

@cuda.jit
def sel_active_gpu(T,per_mat,stack,im_out,cover,BPM_ratio,stp,iter_block):
    b_dimx = cuda.blockDim.x
    b_dimy = cuda.blockDim.y
    
    bx = cuda.blockIdx.x  
    by = cuda.blockIdx.y
            
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    for bz in range(T):
        if stack[bz,(bx//BPM_ratio)*stp+(bx%BPM_ratio)*b_dimx+tx,(by//BPM_ratio)*stp+(by%BPM_ratio)*b_dimy+ty] >= per_mat[bz,bx//BPM_ratio,by//BPM_ratio]:
            cuda.atomic.add(im_out,(bz,(bx//BPM_ratio)*stp+(bx%BPM_ratio)*b_dimx+tx,(by//BPM_ratio)*stp+(by%BPM_ratio)*b_dimy+ty),1)
   
        cuda.atomic.add(cover,(bz,(bx//BPM_ratio)*stp+(bx%BPM_ratio)*b_dimx+tx,(by//BPM_ratio)*stp+(by%BPM_ratio)*b_dimy+ty),1)


@cuda.jit
def sel_active_gpu_gen(bz,time_ref,per_mat,stack,im_out,cover,BPM_ratio,stp,iter_block,last_stp):

    size = cuda.gridDim.x
    iterat = int(iter_block//(size//BPM_ratio))
    if iter_block%(size//BPM_ratio)>0:
        iterat+=1
        
    b_dimx = cuda.blockDim.x
    b_dimy = cuda.blockDim.y
    stp_iter = stp*(size//BPM_ratio)
    
    bx = cuda.blockIdx.x  
    by = cuda.blockIdx.y
            
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    for it in range(iterat*iterat):
        it_bk =it//iterat
        it_bk_y = it%iterat
        if it_bk*stp_iter+((bx//BPM_ratio)*stp)<=last_stp and it_bk_y*stp_iter+((by//BPM_ratio)*stp)<=last_stp:

            if stack[bz,it_bk*stp_iter+(bx//BPM_ratio)*stp+(bx%BPM_ratio)*b_dimx+tx,it_bk_y*stp_iter+(by//BPM_ratio)*stp+(by%BPM_ratio)*b_dimy+ty] >= per_mat[bz+time_ref,it_bk*5+bx//BPM_ratio,it_bk_y*5+by//BPM_ratio]:
                cuda.atomic.add(im_out,(it_bk*stp_iter+(bx//BPM_ratio)*stp+(bx%BPM_ratio)*b_dimx+tx,it_bk_y*stp_iter+(by//BPM_ratio)*stp+(by%BPM_ratio)*b_dimy+ty),1)

            if bz ==0 and time_ref==0:
                cuda.atomic.add(cover,(it_bk*stp_iter+(bx//BPM_ratio)*stp+(bx%BPM_ratio)*b_dimx+tx,it_bk_y*stp_iter+(by//BPM_ratio)*stp+(by%BPM_ratio)*b_dimy+ty),1)
    
    
    
class sel_active_reg():
    
    def __init__(self,stack,dict_params,verbose=True,static=False):
        self.stack = stack
        self.step_list = dict_params['list']
        if len(self.step_list)==1:
            self.stp=1
        else:
            self.stp = self.step_list[1]-self.step_list[0]
        self.blocks = dict_params['blocks']
        self.threads = dict_params['threads']
        self.BPM_ratio = dict_params['BPM_ratio'] # # of block inside a patch
        self.bb = dict_params['bb']

        self.N_pix_st = dict_params['N_pix_st']
        self.astr_min = dict_params['astr_min']
        self.per_tile = dict_params['percentile']
        self.astro_num = dict_params['astro_num']
        self.init_th_ = dict_params['init_th_']
        self.decr_dim = dict_params['decr_dim']
        self.decr_th = dict_params['decr_th']
        self.corr_int = dict_params['corr_int']
        self.gpu_flag = dict_params['gpu_flag']
        self.jobs = -1
        self.static = static
        self.verbose = verbose
        self.iter_block = len(dict_params['list'])
        self.gpu_num = 0
    
    def check_sel_active_reg_gpu_gen(self):

        _,N,M = self.stack.shape
        T=1
        cuda.select_device(self.gpu_num)    

        threadsperblock = (self.threads,self.threads)
        blockspergrid = (self.blocks,self.blocks)
             
        ### allocate percentile matrix
        
        if self.verbose: print('Iteration per block: ',self.iter_block/(self.blocks//self.BPM_ratio))
        
        if self.verbose: print('GPU started with ',blockspergrid,' blocks and ', threadsperblock,' threads per block')
        mat_per = np.zeros((1,len(self.step_list),len(self.step_list)),dtype=np.int32)
        mat_per_g = cuda.to_device(mat_per) 
        ### allocate in ram
        im_out = np.zeros((N,M),dtype=np.int32)
        cover = np.zeros((N,M),dtype=np.int32)
        ### allocate and load in DRAM
        im_out_g = cuda.to_device(im_out)
        cover_g = cuda.to_device(cover)

        blocks_to_load =[i*1000 for i in range((T//1000)+1)]
        blocks_to_load.append(T)
       
        for stps in range(len(blocks_to_load)-1):
            stack_gpu = cuda.to_device(self.stack[blocks_to_load[stps]:blocks_to_load[stps+1],:,:])
            
            for bz in range(blocks_to_load[stps+1]-blocks_to_load[stps]):
                sel_active_gpu_gen[blockspergrid, threadsperblock](bz,blocks_to_load[stps],mat_per_g,stack_gpu,im_out_g,cover_g,self.BPM_ratio,self.stp,self.iter_block,self.step_list[-1])#
                
            ### free from old stack
            del stack_gpu

        im_out = im_out_g.copy_to_host()
        cover = cover_g.copy_to_host()
        assert cover.min()!=0, 'Check steps positions and BB'
        assert (im_out/cover).max()==1,'Check steps positions,BB and input stack, MAX val too much high'
        return im_out,cover
    
    @staticmethod
    def percent_matrix_par(stack,t,listx,bb,per_tile):
        listy = listx
        dim = len(listx)
        matrix= t*np.ones((dim+1,dim),dtype=np.float32)

        for i in range(dim):
            for j in range(dim):

                x = listx[i] 
                y = listy[j]
                matrix[i,j] = np.percentile(stack[t,x:x+bb,y:y+bb],per_tile)  

        return matrix.astype(np.float32) 

    def sel_active_reg_cpu(self):

        T,N,M = self.stack.shape
        

        percent_list = Parallel(n_jobs=self.jobs,verbose=0)(delayed(self.percent_matrix_par) (self.stack,i,self.step_list,self.bb,self.per_tile) for i in range(T))
        percentiles = np.asarray(percent_list)
        mat_per = percentiles[:,:-1,:]
        mat_per = mat_per[percentiles[:,-1,0].astype(np.int32),:,:]

        im_out = np.empty((T,N,M)) 
        cover = np.zeros((T,N,M)) 
        for i in range(T):
            for x in self.step_list:
                for y in self.step_list:

                    buffer_im = self.stack[i,x:x+self.bb,y:y+self.bb]-mat_per[i,x//self.stp,y//self.stp]
                    buffer_im[buffer_im<0]=0.
                    buffer_im[buffer_im>0]=1.

                    im_out[i,x:x+self.bb,y:y+self.bb]+=buffer_im
                    cover[i,x:x+self.bb,y:y+self.bb]+=1


        
        self.mask_tot = np.empty_like(im_out)
        self.mask_tot  = np.sum(im_out/cover,axis=0)
    
    def sel_active_reg_gpu(self):

        T,N,M = self.stack.shape
        cuda.select_device(2)    
        ### allocate in ram
        im_out = np.zeros((T,N,M),dtype=np.float32)
        cover = np.zeros((T,N,M),dtype=np.intc)
        ### allocate and load in DRAM
        stack_gpu = cuda.to_device(self.stack.astype(np.float64))
        im_out_g = cuda.to_device(im_out)
        cover_g = cuda.to_device(cover)

        threadsperblock = (self.threads,self.threads)
        blockspergrid = (self.blocks,self.blocks)
        
        # compute percentile in patches
        if not(self.static):
            percent_list = Parallel(n_jobs=self.jobs)(delayed(self.percent_matrix_par) (self.stack,i,self.step_list,self.bb,self.per_tile) for i in range(T))
            percentiles = np.asarray(percent_list)
            mat_per = percentiles[:,:-1,:]

            mat_per = mat_per[percentiles[:,-1,0].astype(np.int32),:,:]# reorder the embarasing parallel collection of mat
            
        #### mod for static fluorophore
        # compute a single percentile for all the stack, and than generate a T x num_patch x num_patch 
        elif self.static:
            mat_per = np.percentile(self.stack.flatten(),self.per_tile).reshape(1,1)
            mat_per = np.tile(mat_per,(T,1,1))
        ### allocate percentile matrix
        mat_per_g = cuda.to_device(mat_per)    
        sel_active_gpu[blockspergrid, threadsperblock](T,mat_per_g,stack_gpu,im_out_g,cover_g,self.BPM_ratio,self.stp,self.iter_block)
        im_out = im_out_g.copy_to_host()
        cover = cover_g.copy_to_host()
        
        self.mask_tot = np.empty_like(im_out)
        self.mask_tot  = np.sum(im_out/cover,axis=0)
        
    
    def sel_active_reg_gpu_gen(self):

        T,N,M = self.stack.shape
        cuda.select_device(self.gpu_num)    

        threadsperblock = (self.threads,self.threads)
        blockspergrid = (self.blocks,self.blocks)
            
        if self.verbose: print('Computing local thresholds')
        # compute percentile in patches
        if not(self.static):
            percent_list = Parallel(n_jobs=self.jobs,verbose=1)(delayed(self.percent_matrix_par) (self.stack,i,self.step_list,self.bb,self.per_tile) for i in range(T))
            percentiles = np.asarray(percent_list).astype(np.float32)
            mat_per = percentiles[:,:-1,:]

            mat_per = mat_per[percentiles[:,-1,0].astype(np.int32),:,:]# reorder the embarasing parallel collection of mat
            
        #### mod for static fluorophore
        # compute a single percentile for all the stack, and than generate a T x num_patch x num_patch 
        elif self.static:
            mat_per = np.percentile(self.stack.flatten(),self.per_tile).reshape(1,1)
            mat_per = np.tile(mat_per,(T,1,1))
            
            
        #mat_per = np.zeros((T,len(self.step_list),len(self.step_list)))#,dtype=np.int32   
        ### allocate percentile matrix
        
        if self.verbose: print('Iteration per block: ',self.iter_block/(self.blocks//self.BPM_ratio))
        
        if self.verbose: print('GPU started with ',blockspergrid,' blocks and ', threadsperblock,' threads per block')
        
        mat_per_g = cuda.to_device(mat_per) 
        ### allocate in ram
        im_out = np.zeros((N,M),dtype=np.int32)
        cover = np.zeros((N,M),dtype=np.int32)
        ### allocate and load in DRAM
        im_out_g = cuda.to_device(im_out)
        cover_g = cuda.to_device(cover)

        blocks_to_load =[i*1000 for i in range((T//1000)+1)]
        blocks_to_load.append(T)
       
        for stps in range(len(blocks_to_load)-1):
            stack_gpu = cuda.to_device(self.stack[blocks_to_load[stps]:blocks_to_load[stps+1],:,:])
            print(stps)
            
            for bz in range(blocks_to_load[stps+1]-blocks_to_load[stps]):
                sel_active_gpu_gen[blockspergrid, threadsperblock](bz,blocks_to_load[stps],mat_per_g,stack_gpu,im_out_g,cover_g,self.BPM_ratio,self.stp,self.iter_block,self.step_list[-1])#
                
            ### free from old stack
            del stack_gpu

        im_out = im_out_g.copy_to_host()
        cover = cover_g.copy_to_host()
        if self.verbose: print('GPU done')
        del im_out_g, cover_g, mat_per_g
        
        self.mask_tot = np.empty_like(im_out).astype(np.float64)
        self.mask_tot  = im_out.astype(np.float64)/cover.astype(np.float64) 
        return im_out,cover
    
    
    def get_mask(self,find_round=True,long_rec=False):
        T,_,_ = self.stack.shape
        
        if self.gpu_flag and not(long_rec):
            self.sel_active_reg_gpu()
        elif self.gpu_flag and long_rec:
            self.sel_active_reg_gpu_gen()
        else:
            self.sel_active_reg_cpu()
    
        if self.corr_int:
            scaling = ThScal(self.stack)

       
        th_ =round(T*self.init_th_)
        
        if find_round:
            #this is an alternative strategy to select the strating point threshold the nearest to th_, it is a seed for the while below 
            #this strategy can be removed and th_ will be the T*self.init_th_ and not one of the seed points below
            # we used this strategy for dataset-1, this approach reduces large variation in th_ due to small variation in self.init_th_
            ratio=1
            if T>500: 
                ratio = (T/750)
                th_list = [200,250,300,350,400,450,500,550,600,650,700]
                th_list = (T/750)*np.asarray(th_list)
                th_ref =th_list-th_
                th_ = th_list[np.argmin(np.abs(th_ref))]
                
        cnt=0
        if self.verbose: print('Init threshold',th_)
        starting_th = th_
        flag_th=True
        N_pix = self.N_pix_st
        
        while(cnt<self.astro_num and N_pix>=self.N_pix_st*0.3 and th_>round(T*0.3)):
            if flag_th:
                mask_tot_s = self.mask_tot.copy()#np.sum(self.mask_tot,axis=0)

                if self.corr_int:
                    mask_tot_s = scaling.ThMat(mask_tot_s,th_)
                    mask_tot_s= np.uint8(mask_tot_s)
                else:
                    mask_tot_s[mask_tot_s<=th_]=0
                    mask_tot_s[mask_tot_s>0.5]=255
                    mask_tot_s= np.uint8(mask_tot_s)  


                ret, labels_r = cv2.connectedComponents(mask_tot_s)
                #print('ZONES',ret)
                flag_th = False

            labels = labels_r.copy()
            cnt=0
            for i in range(1, ret):
                pts =  np.where(labels == i)    
                #print(len(pts[0]))
                if len(pts[0]) < N_pix:
                    
                    labels[pts] = 0
                else:
                    cnt+=1

                    labels[pts] = 255         
            #print('Found',cnt)
            N_pix-=self.decr_dim
            if N_pix<=self.astr_min and (starting_th-th_)<(ratio*105):

                th_-=self.decr_th
                flag_th = True  
                N_pix=self.N_pix_st


        if self.verbose: print('Zones',cnt)
        # clean eventual artifacts
        ret, labels = cv2.connectedComponents(np.uint8(labels))
        for i in range(1, ret):
            pts =  np.where(labels == i)    
            if len(pts[0])<self.astr_min//4:
                labels[pts]=0
        labels[labels>0]=1
        return labels
    