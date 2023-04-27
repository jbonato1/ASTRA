import torch
import torch.nn.functional as F
import numpy as np
import cv2
from get_traces import allineate_stack
import time

@torch.no_grad()
def corr_mask(roi_proc,crop_stack,device,th_corr,nf_mFilter=5):#5
    ##tensor definition
    roi_num,N,M = roi_proc.shape
    map_ = torch.from_numpy(np.uint8(roi_proc.reshape(roi_num,N*M)))
#    print(np.sum(roi_proc[0,:,:]),np.sum(roi_proc[1,:,:]),torch.sum(map_[0,:]),)
    crop_stack = crop_stack.reshape(crop_stack.shape[0],N*M)
    crop_ten = torch.from_numpy(crop_stack.T).float()
    pix_crop,T = crop_ten.size()


    #load to device
    map_ = map_.reshape(roi_num,pix_crop,1,1).to(device)
    crop_ten = crop_ten.reshape(pix_crop,1,T).to(device)
    #convolution

    filter_ = (1/nf_mFilter)*torch.ones((1,1,nf_mFilter),dtype=torch.float).to(device)
    crop_ten = F.conv1d(crop_ten,filter_,padding=(nf_mFilter-1)//2)
    #initialize tensor
    map_to_cover = torch.empty((pix_crop,1,1),dtype = torch.uint8).to(device)

    coor_pix_ten = torch.zeros((roi_num,N,M),dtype=torch.float).to(device)
    #loop over rois
    for j in range(roi_num):

        map_to_cover = map_[j,:,:,:]
        el_test = torch.sum(map_to_cover)
        if el_test!=0:
            test_ten = torch.empty((int(el_test),1,T),dtype=torch.float).to(device)
            #get test pixels and normalize it for correlation
            test_ten = torch.masked_select(crop_ten, map_to_cover.bool()).reshape(el_test,1,T)

            test_ten = (test_ten-torch.mean(test_ten,dim=2).reshape(el_test,1,1))/(0.0001+torch.std(test_ten,dim=2).reshape(el_test,1,1)*T)

            #normalize pixel for correlation
            crop_ten = (crop_ten - torch.mean(crop_ten,dim=2).reshape(pix_crop,1,1))/(0.0001+torch.std(crop_ten,dim=2).reshape(pix_crop,1,1))

            #compute correlation 
      #      print('SIZE crop_ten:',crop_ten.size(),'test_ten',test_ten.size())
            cycles = (el_test.float()/500).floor()
            cyc=0
            starting = 500
            #print('CYCLES',cycles)
            while(cyc<=cycles):
                #print('safty check',cyc,cycles)
                if cyc<cycles :
                    buff2 = F.conv1d(crop_ten,test_ten[starting*cyc:starting*cyc+starting,:,:],padding=10)
                    cyc+=1
                else:
                    buff2 = F.conv1d(crop_ten,test_ten[starting*cyc:,:,:],padding=10)
                    cyc+=1

                buff2[buff2<th_corr]=0
                buff2 =torch.sum(buff2.float(),dim=2)
                buff2 =torch.sum(buff2,dim=1)
                buff2 = buff2.reshape(N,M)
                buff2[buff2>0]=1
                coor_pix_ten[j,:,:] =coor_pix_ten[j,:,:]+ buff2.float()
            
    del crop_ten, map_,map_to_cover, test_ten,buff2

    return coor_pix_ten



class comp_err_correlation():
    
    def __init__(self,coord_st_l,coord_cir_l,stack,mask_test,radius=45,nf_mFilter=5):
        self.coord_st_l = coord_st_l
        self.coord_cir_l = coord_cir_l
        self.stack = stack
        self.mask_test = mask_test
        self.T,_,_ = stack.shape
        self.radius = radius
        self.nf_mFilter = nf_mFilter
        
    def gen_false(self,stack,coord_cir_l,rm_px = 6):
        _,N,M = stack.shape
        rnd_mask = np.zeros((N,M))
        ref = np.ones((N,M))
        for j in range(len(coord_cir_l)):
            coord_c = coord_cir_l[j]
            ref[coord_c[0],coord_c[1]]=0
        ref[0:rm_px,:]=0
        ref[-rm_px:,:]=0
        ref[:,0:rm_px]=0
        ref[:,-rm_px:]=0
        coord_false = np.where(ref==1)
        #print("qwqwq",len(coord_false[0]))
        coord_rnd = np.random.choice(len(coord_false[0]), 250, replace=False)

        rnd_mask[coord_false[0][coord_rnd],coord_false[1][coord_rnd]]=1
        #plt.imshow(rnd_mask)
        coord_out = np.where(rnd_mask==1)
        return stack[:,coord_false[0][coord_rnd],coord_false[1][coord_rnd]]

    def coor_false_discovery(self,roi,crop_stack,false_stack,device):
            T,N,M = crop_stack.shape
            _,N_f = false_stack.shape
            roi = roi.reshape(-1)
            coord_roi = np.where(roi==1)
            crop_stack = crop_stack.reshape(T,N*M)

            roi_stack = np.empty((T,len(coord_roi[0])))
            roi_stack = crop_stack[:,coord_roi[0]].astype(np.float32)

            map_ = torch.from_numpy(roi_stack.T).float()
            false_ten = torch.from_numpy(false_stack.T.astype(np.float32)).float()


            #load to device
            map_ = map_.reshape(len(coord_roi[0]),1,T).to(device)

            false_ten = false_ten.reshape(N_f,1,T).to(device)
           #convolution

            filter_ = (1/self.nf_mFilter)*torch.ones((1,1,self.nf_mFilter),dtype=torch.float).to(device)
            map_ = F.conv1d(map_,filter_,padding=(self.nf_mFilter-1)//2)
            false_ten = F.conv1d(false_ten,filter_,padding=(self.nf_mFilter-1)//2)

           #initialize tensor

            #normalize pixel for correlation
            map_ = (map_-torch.mean(map_,dim=2).reshape(len(coord_roi[0]),1,1))/((torch.std(map_,dim=2).reshape(len(coord_roi[0]),1,1)*T)+0.001)
            false_ten = (false_ten - torch.mean(false_ten,dim=2).reshape(N_f,1,1))/((torch.std(false_ten,dim=2).reshape(N_f,1,1))+0.001)


            buff = F.conv1d(map_,false_ten,padding=5)

            return buff.data.cpu().numpy()
    
    def find_threshold(self):
           
        th_corr=0.85
       
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ind_list=0
        ind_list_red=0
        
        th_list=[0.85,0.88,0.9,0.91,0.92,0.93,0.94,0.95]
        th_corr=th_list[0]
        th_list_red=[0.85,0.80,0.75,0.7,0.65,0.60]#,0.55,0.45,0.40]
      
        flag_red = True
        flag=True
        
        list_corr = []
        for iter_ in range(5):
            print('ITER: ',iter_)
            list_cell = []
            for j in range(len( self.coord_st_l)):

                coord_bb = self.coord_st_l[j]
                coord_circonf = self.coord_cir_l[j]

                stack_crop = np.empty((self.T,self.radius*2,self.radius*2))
                stack_crop = self.stack[:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]].copy()



                mask_crop = np.sum(self.mask_test[j,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2],:],axis=2)
                mask_crop[mask_crop>1]=1

                f_stack = self.gen_false(self.stack,self.coord_cir_l)
                corr_vect = self.coor_false_discovery(mask_crop,stack_crop,f_stack,device)
                max_ = np.amax(corr_vect,axis=2)
                max_ = np.amax(max_,axis=0)
                list_cell.append(max_.copy())
                
            list_corr.append(list_cell)
        print('Ref Corr matrix computed')    
        while( flag and ind_list_red<(len(th_list_red)-1) and ind_list<(len(th_list)-1) ):
            for iter_ in range(5):
                if iter_==0:
                    res_final=0
                res=0
                for j in range(len( self.coord_st_l)):
                    
                    th = list_corr[iter_][j].copy()
                    th[th<th_corr]=0
                    th[th>0]=1
                    #print('tmp',np.sum(th)/1000)
                    res+=(np.sum(th)/250)

                res/=len(self.coord_st_l) 
                
                res_final+=res
                #print('res_final',res_final)
            res_final/=5

            print('correlation error = {:f} with threshold = {:f}'.format(res_final,th_corr))#res_final,th_corr
            
            if res_final<0.02 and flag_red:
                ind_list_red+=1
                th_corr = th_list_red[ind_list_red]
                if ind_list_red==(len(th_list_red)-1):
                    th_out = th_corr
                    print('Min correlation threshold: ',th_out)
                print('New th',th_corr)
            
            elif res_final<0.06 and res_final>=0.02 and flag_red:
                th_out = th_corr
                flag=False
                print('Reduction finish')
                
            
            if res_final<0.06 and not(flag_red):
                th_out = th_corr
                flag=False
                print('Incr finish')
                
            if res_final>=0.06:
                if flag_red and th_corr!=0.85:
                    flag=False
                    th_out = th_list_red[ind_list_red-1]
                else:
                    ind_list+=1
                    th_corr = th_list[ind_list]
                    if ind_list == len(th_list)-1:
                        th_out = th_corr
                        print('Max correlation threshold: ',th_out)
                    flag_red=False

            
        return th_out
    
    
    
def clean_outer_pixel(astro_roi,num_pix):
    #rem H
    astro_roi[:,:num_pix,:,:]=0
    astro_roi[:,-num_pix:,:,:]=0
    #rem W
    astro_roi[:,:,:num_pix,:]=0
    astro_roi[:,:,-num_pix:,:]=0
    
    return astro_roi


def cleanCC(mask_ROI_or,MAX_ROI_AREA_PROC):
    mask_ROI = mask_ROI_or.copy()
    _,N,M,_ = mask_ROI.shape
    
    for i in range(mask_ROI.shape[0]):#mask_ROI.shape[0]
        num,comp = cv2.connectedComponents(mask_ROI_or[i,:,:,2].astype(np.uint8))
        for k in range(1,num):
            pt = np.where(comp==k)
            if pt[0].shape[0]>=(MAX_ROI_AREA_PROC)//2:
                mask_ROI[i,pt[0],pt[1],2]=1
            else:
                mask_ROI[i,pt[0],pt[1],2]=0
                
    return mask_ROI

def main_CC(stack_o,mask_sp,device,r,coord_dict,shift):
    print('radius',r)
    
    print(10*'/','CORR ANALysis',10*'/')
    stack = stack_o.copy()
    T,N,M = stack.shape
    print(stack.flags['C_CONTIGUOUS'])
    
    coord_st_l = []
    coord_cir_l = []
    for j in range(mask_sp.shape[0]):
        name = str(j)
        coord_st_l.append(coord_dict['ST_'+f'{name:0>3}'] )
        coord_cir_l.append(coord_dict['CIRCLE_'+f'{name:0>3}'] )


    err_corr = comp_err_correlation(coord_st_l,coord_cir_l,stack,mask_sp)
    th_corr =err_corr.find_threshold()
    #######################

    mask_out  = np.zeros((mask_sp.shape[0],N,M,3))
    mask_out[:,:,:,:2] = mask_sp
    
    mask_single_corr = np.zeros((len(coord_st_l),2*r,2*r))
    mask_single_corr2 = np.zeros((len(coord_st_l),2*r,2*r))
    t1 = time.time()
    
    for j in range(mask_sp.shape[0]):

        #########################################pick coord
        coord_bb = coord_st_l[j]

        ########################################define buffer
        mask_crop = np.zeros((2,r*2,r*2))
        mask_crop[0,:,:] = mask_sp[j,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2],0]
        mask_crop[1,:,:] = mask_sp[j,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2],1]

        ########################################define stack for cross corr
        stack_crop = np.empty((T,2*r,2*r))
        
        ###to do insert shift corr
        if not(shift is None):
            stack_buffer = allineate_stack(stack,shift['Shift_'+f'{str(j):0>3}'],r_domain=r)
            stack_crop = stack_buffer[:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]].copy()
        else:
            stack_crop = stack[:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]].copy()
        ###
        
        

        #########################################definr map for selected pixels
        map_ = np.zeros((N,M))
        map_[coord_cir_l[j][0],coord_cir_l[j][1]]=1

        ###### filter the corners from the stack and from buff mask other cell segmentation 
        stack_crop = stack_crop*map_[coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]]


        if np.sum(mask_sp[j,:,:,0])!=0 and np.sum(mask_sp[j,:,:,1])!=0: 

            out_tensor = corr_mask(mask_crop,stack_crop,device,th_corr)
            out_tensor_np = out_tensor[0,:,:].data.cpu().numpy()
            out__ = out_tensor[0,:,:].data.cpu().numpy() + out_tensor[1,:,:].data.cpu().numpy()
            del out_tensor
            torch.cuda.empty_cache()
            out_tensor_np = np.nan_to_num(out_tensor_np)
            out_tensor_np[out_tensor_np>1]=1 
            
            out__ = np.nan_to_num(out__)
            out__[out__>1]=1 
            out__-=np.sum(mask_crop,axis=0)
            mask_out[j,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2],2] += out__#out_tensor_np


            mask_single_corr[j,:,:]=out_tensor_np
            mask_single_corr2[j,:,:]=out__
    print(time.time()-t1)
    mask_out[mask_out>1]=1
    return mask_single_corr2,mask_out
