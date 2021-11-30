from sklearn.cluster import KMeans
import numpy as np
import cv2
from motion_correction import Motion_Correction
from skimage import io
import pandas as pd

class Extr_miniROI():
    
    def __init__(self,Area_mu,mu_px,proc_to_split,soma,split_proc=False,dilate_ROI=1):
        
        
        self.Area_mu = Area_mu
        self.mu_px = mu_px
        
        self.Area_px = Area_mu/mu_px**2
        self.proc_to_split = proc_to_split
        self.soma = soma
        self.split_proc = split_proc
        self.dilate_ROI=dilate_ROI
    
    @staticmethod
    def det_conn_comp(processes,soma,dilate_ROI,split_proc):
        #print(processes.shape)
        N,M = processes.shape
        num,comp = cv2.connectedComponents(processes.astype(np.uint8))
        print('ROIS',num-1)
        processes_out = np.zeros((num-1,N,M))
        for k in range(1,num):
            pt = np.where(comp==k)
            buff = np.zeros((N,M))
            buff[pt[0],pt[1]]=1
            ###dilation
            for _ in range(dilate_ROI):
                element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
                buff = cv2.dilate(buff, element)
            if not(split_proc):
                ### this is done only for ImageJ rendering 
                filtered = buff
            else:
                filtered = buff-soma-np.sum(processes_out,axis=0)
            filtered[filtered<1]=0
            processes_out[k-1,:,:]=filtered
            
        return processes_out

    def get_k(self,area):
        return area/self.Area_px
    
    def get_miniROI(self):
        
        self.proc_to_split = self.det_conn_comp(self.proc_to_split,self.soma,self.dilate_ROI,self.split_proc)
        #print('check size',self.proc_to_split.shape)
        if self.split_proc :
            Nroi,H,W = self.proc_to_split.shape
            collROI = []
            for i in range(Nroi):
                N = np.sum(self.proc_to_split[i,:,:])
                k = self.get_k(N)
                #print('ratio',k,int(k))
                if int(k)<=1:
                    collROI.append(self.proc_to_split[i,:,:][:,:,np.newaxis])
                else:
                    rr = k.astype(np.int64)
                    buff = np.zeros((H,W,rr))
                    pt = np.where(self.proc_to_split[i,:,:]==1)
                    X = np.asarray([pt[0],pt[1]]).T

                    kmeans = KMeans(n_clusters=int(k), random_state=0).fit(X)
                    labels = kmeans.labels_

                    for j in range(int(k)):
                        pt_lb = np.where(labels==j)
                        buff[X[pt_lb[0],0],X[pt_lb[0],1],j]=1

                    collROI.append(buff)
            out = collROI[0]       

            if len(collROI)>1:
                for i in range(1,len(collROI)):  
                    #print(collROI[i].shape)
                    out = np.dstack((out,collROI[i]))
            print('SPLIT DONE',out.shape)
            return out
        else:
            print('SPLIT DONE',self.proc_to_split.shape)
            out = np.moveaxis(self.proc_to_split,[0,1,2],[2,0,1])
            return out
    
    
    
def get_signals(roi,stack):
    T,_,_ = stack.shape
    _,_,N = roi.shape
    
    signals = []
    for i in range(N-1):
        pt = np.where(roi[:,:,i]==1)
        #ev add single pixels rem
        signals.append(np.mean(stack[:,pt[0],pt[1]],axis=1))
    return signals

def get_signal(roi,stack):
    pt = np.where(roi==1)
    return np.mean(stack[:,pt[0],pt[1]],axis=1)

def allineate_stack(stack,shift,r_domain):
    T,N,M = stack.shape
    stack_out = np.zeros_like(stack)
    stack_pad = np.pad(stack,((0,0),(r_domain,r_domain),(r_domain,r_domain)), 'constant', constant_values=0)
    
    for t in range(T):
        if np.abs(shift[t,0])>(r_domain/2) or np.abs(shift[t,1])>(r_domain/2):
            
            if t<T-1 and t!=0:
                shift[t,0] = (shift[t-1,0]+shift[t+1,0])/2
                shift[t,1] = (shift[t-1,1]+shift[t+1,1])/2
            elif t==T-1:
                shift[t,0] = shift[t-1,0]
                shift[t,1] = shift[t-1,1]
            elif t==0:
                shift[t,0] = shift[t+1,0]
                shift[t,1] = shift[t+1,1]
                
        
        stack_out[t,:,:]= stack_pad[t,r_domain-shift[t,0]:r_domain+N-shift[t,0],r_domain-shift[t,1]:r_domain+M-shift[t,1]]
    return stack_out

def box(cX,cY,radius,N):
    casex=2
    casey=2
    if cX-radius<0:
        casex=0
    elif cX+radius>N:
        casex=1
    if cY-radius<0:
        casey=0
    elif cY+radius>N:
        casey=1
    #x
    if casex==2:
        c1x=cX-radius
        c2x=cX+radius
    elif casex==0:
        c1x=0
        c2x=2*radius
    else:
        c1x=N-2*radius
        c2x=N
    #y
    if casey==2:
        c1y=cY-radius
        c2y=cY+radius
    elif casey==0:
        c1y=0
        c2y=2*radius
    else:
        c1y=N-2*radius
        c2y=N
    return c1x,c1y,c2x,c2y

def create_bb_coord_domain(soma_mask,radius = 60):
   
    N,M = soma_mask.shape
    soma = np.empty_like(soma_mask)
    soma = soma_mask.copy()
    soma[soma>0.1]=255

    _,thresh = cv2.threshold(np.uint8(soma),127,255,0)

    # find contours in the binary image
    contours, _= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    # loop over the contours
    #list of array with the coordinate
    coord_list_st = []
    coord_list_cell = []
    coord_list_circle = []
    for c in contours:
        if c.shape[0]!=1:
            im_buff = np.zeros((N,M))
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"]+1e-5))
            cY = int(M["m01"] / (M["m00"]+1e-5))
            #compute the region of astrocyte radius 60
            c1x,c1y,c2x,c2y = box(cX,cY,48,N)
            coord_list_cell.append([c1x,c1y,c2x,c2y])
            c1x,c1y,c2x,c2y = box(cX,cY,radius,N)
            cv2.circle(im_buff,(cX,cY),radius,(255,0,0),thickness =-1,lineType=8)
            coord_circle = np.where(im_buff==255)
            coord_list_circle.append(coord_circle)

            coord = np.array([c1x,c1y,c2x,c2y])
            coord_list_st.append(coord)
        return coord_list_st,coord_list_circle, coord_list_cell

        
def update_dict_DNN(dict_im,single_astro_roi,fov_num,motion_corr,MAX_ROI_AREA_PROC,MU_PX,MiniROI=False):
    dict_im['Single_cell_mask_'+fov_num] = single_astro_roi
    dict_roi={}
    dict_traces={}
    dict_cell_coord={}
    dict_cell_shift={}
    dict_im['Cell_num_'+fov_num] = single_astro_roi.shape[0]
    
    print("ROI NUM",single_astro_roi.shape[0])
   
    if motion_corr : mc  = Motion_Correction(pix_precision=1)
    
    for s_roi_num in range(single_astro_roi.shape[0]):
        name = str(s_roi_num)
        
        coord_list_st,coord_list_circle, coord_list_cell = create_bb_coord_domain(single_astro_roi[s_roi_num,:,:,1])
        dict_cell_coord['ST_'+f'{name:0>3}'] = coord_list_st[0]
        dict_cell_coord['CIRCLE_'+f'{name:0>3}'] = coord_list_circle[0]
        dict_cell_coord['BB_cell_'+f'{name:0>3}'] = coord_list_cell[0]
        if motion_corr:
            coord_bb = coord_list_cell[0]
#             print('Shift_'+f'{name:0>3}')
            dict_cell_shift['Shift_'+f'{name:0>3}'] = mc.motion_corr(dict_im['t-series_'+fov_num][:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]],\
                           ref_image=np.mean(dict_im['t-series_'+fov_num][:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]],axis=0))
        
            stack_buffer = allineate_stack(dict_im['t-series_'+fov_num],dict_cell_shift['Shift_'+f'{name:0>3}'],r_domain=dict_im['Astro_domain_radius'])
        else:
            stack_buffer = dict_im['t-series_'+fov_num]
            
        print(50*'%','Extracting cell:',s_roi_num)
        constr_split_roi = Extr_miniROI(MAX_ROI_AREA_PROC,MU_PX,single_astro_roi[s_roi_num,:,:,0],single_astro_roi[s_roi_num,:,:,1],MiniROI)
        arr_out_proc = constr_split_roi.get_miniROI()
        if  s_roi_num==0:
            list_out=arr_out_proc
        else:
            list_out = np.dstack((list_out,arr_out_proc))
       
        dict_roi['Soma_'+f'{name:0>3}'] = np.where(single_astro_roi[s_roi_num,:,:,1]==1)
        dict_traces['Soma_'+f'{name:0>3}'] = get_signal(single_astro_roi[s_roi_num,:,:,1],stack_buffer)
        for proc_num in range(arr_out_proc.shape[2]):
            name_proc = str(proc_num)
            dict_roi['Proc_'+f'{name:0>3}'+'_'+f'{name_proc:0>3}'] = np.where(arr_out_proc[:,:,proc_num]==1)
            dict_traces['Proc_'+f'{name:0>3}'+'_'+f'{name_proc:0>3}']  = get_signal(arr_out_proc[:,:,proc_num],stack_buffer)
        print('Extraction: done')
    
    dict_im['Signals_extr_'+fov_num] = dict_traces
    dict_im['ROI_'+fov_num] = dict_roi
    dict_im['crop_coord_ROI_'+fov_num] = dict_cell_coord
    
    if motion_corr:
        dict_im['shift_ROI_'+fov_num] = dict_cell_shift
    else:
        dict_im['shift_ROI_'+fov_num] = None
        
    #### for Visualization purposes
    list_out = np.dstack((list_out,dict_im['Final_Mask_'+fov_num][:,:,1:]))
    dict_im['Final_Mask_fraction_'+fov_num] = list_out
    
    return dict_im


def gen_dataframe(dict_traces):
    col_name = list(dict_traces.keys())
    N = dict_traces[col_name[0]].shape[0]
    vect = np.empty((N,len(col_name)))
    
    i=0
    for key in dict_traces.keys():
        vect[:,i] = dict_traces[key]
        i+=1
        
    return pd.DataFrame(vect,columns = col_name)

def gen_raw_dataframe(single_astro_roi,stack_list_path,motion_corr,MAX_ROI_AREA_PROC,MU_PX,Astro_radius,MiniROI=False,gpu=True):
    
    print("ROI NUM",single_astro_roi.shape[0])    
    
    dict_im = {}
    dataframe_list = []
    
    if motion_corr : mc  = Motion_Correction(pix_precision=1,gpu=gpu)
    fov_num = 0 
    for stack_path in stack_list_path:
        
        print('Analyzing: ',stack_path)
        stack_current = io.imread(stack_path)
        
        dict_roi={}
        dict_traces={}
        dict_cell_coord={}
        dict_cell_shift={}
        
        for s_roi_num in range(single_astro_roi.shape[0]):
            name = str(s_roi_num)

            coord_list_st,coord_list_circle, coord_list_cell = create_bb_coord_domain(single_astro_roi[s_roi_num,:,:,1])
            dict_cell_coord['ST_'+f'{name:0>3}'] = coord_list_st[0]
            dict_cell_coord['CIRCLE_'+f'{name:0>3}'] = coord_list_circle[0]
            dict_cell_coord['BB_cell_'+f'{name:0>3}'] = coord_list_cell[0]
            if motion_corr:
                coord_bb = coord_list_cell[0]
    #             print('Shift_'+f'{name:0>3}')
                dict_cell_shift['Shift_'+f'{name:0>3}'] = mc.motion_corr(stack_current[:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]],\
                               ref_image=np.mean(stack_current[:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]],axis=0))

                stack_buffer = allineate_stack(stack_current,dict_cell_shift['Shift_'+f'{name:0>3}'],r_domain=Astro_radius)
            else:
                stack_buffer = stack_current

            print(50*'%','Extracting cell:',s_roi_num)
            constr_split_roi = Extr_miniROI(MAX_ROI_AREA_PROC,MU_PX,single_astro_roi[s_roi_num,:,:,0],single_astro_roi[s_roi_num,:,:,1],MiniROI,2)
            arr_out_proc = constr_split_roi.get_miniROI()
            if  s_roi_num==0:
                list_out=arr_out_proc
            else:
                list_out = np.dstack((list_out,arr_out_proc))

            dict_roi['Soma_'+f'{name:0>3}'] = np.where(single_astro_roi[s_roi_num,:,:,1]==1)
            dict_traces['Soma_'+f'{name:0>3}'] = get_signal(single_astro_roi[s_roi_num,:,:,1],stack_buffer)
            for proc_num in range(arr_out_proc.shape[2]):
                name_proc = str(proc_num)
                dict_roi['Proc_'+f'{name:0>3}'+'_'+f'{name_proc:0>3}'] = np.where(arr_out_proc[:,:,proc_num]==1)
                dict_traces['Proc_'+f'{name:0>3}'+'_'+f'{name_proc:0>3}']  = get_signal(arr_out_proc[:,:,proc_num],stack_buffer)
            print('Extraction: done')
        
        
        dataframe_list.append(gen_dataframe(dict_traces))
        
        dict_im['ROI_'+str(fov_num)] = dict_roi
        dict_im['crop_coord_ROI_'+str(fov_num)] = dict_cell_coord

        if motion_corr:
            dict_im['shift_ROI_'+str(fov_num)] = None
        else:
            dict_im['shift_ROI_'+str(fov_num)] = dict_cell_shift

        fov_num+=1
        
    
    return dict_im,dataframe_list