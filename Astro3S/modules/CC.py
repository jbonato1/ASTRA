import torch
import torch.nn.functional as F
import numpy as np
import cv2



@torch.no_grad()
def corr_mask(roi_proc,crop_stack,device,th_corr,nf_mFilter=5):
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





def create_bb_coord_correlation(soma_mask,radius = 60):
   
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
            # coordinate cell
            contours_poly = cv2.approxPolyDP(c, 3, True)
            boundRect = cv2.boundingRect(contours_poly)

            coord = np.array([int(boundRect[0])-2,int(boundRect[1])-2,int(boundRect[0])+int(boundRect[2])+2,int(boundRect[1])+int(boundRect[3])+2])
            for el in range(4):
                if coord[el]>N:
                    coord[el]=N
                elif coord[el]<0:
                    coord[el]=0
            coord_list_cell.append(coord)

            # coordinate bb 100
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"]+1e-5))
            cY = int(M["m01"] / (M["m00"]+1e-5))

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
            #compute the region of astrocyte radius 60
            cv2.circle(im_buff,(cX,cY),radius,(255,0,0),thickness =-1,lineType=8)
            coord_circle = np.where(im_buff==255)
            coord_list_circle.append(coord_circle)

            coord = np.array([c1x,c1y,c2x,c2y])
            coord_list_st.append(coord)
        return coord_list_st,coord_list_circle, coord_list_cell

class comp_err_correlation():
    
    def __init__(self,coord_st_l,coord_cir_l,stack,mask_test,radius=60,nf_mFilter=5):
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
        coord_rnd = np.random.choice(len(coord_false[0]), 1000, replace=False)

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
        th_list_red=[0.85,0.80,0.75,0.65,0.60]
      
        flag_red = True
        flag=True
        while( flag and ind_list_red<4 and ind_list<7 ):
            for iter_ in range(20):
                if iter_==0:
                    res_final=0
                res=0
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
                    th =max_.copy()
                    th[th<th_corr]=0
                    th[th>0]=1
                    #print('tmp',np.sum(th)/1000)
                    
                    res+=(np.sum(th)/1000)

                res/=len(self.coord_st_l) 
                
                res_final+=res
                #print('res_final',res_final)
            res_final/=20

            print('correlation error = {:f} with threshold = {:f}'.format(res_final,th_corr))#res_final,th_corr
            
            if res_final<0.01 and flag_red:
                ind_list_red+=1
                th_corr = th_list_red[ind_list_red]
                if ind_list_red==4:
                    th_out = th_corr
                    print('Min correlation threshold 0.6')
                print('New th',th_corr)
            
            elif res_final<0.06 and res_final>=0.01 and flag_red:
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
                    if ind_list == 7:
                        th_out = th_corr
                        print('Max correlation threshold 0.95')
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


def main_CC(stack_o,mask_sp,r,device):
    
    print(10*'/','CORR ANALysis',10*'/')
    stack = stack_o.copy()
    T,N,M = stack.shape
    print(stack.flags['C_CONTIGUOUS'])
    
    coord_st_l = []
    coord_cir_l = []
    for j in range(mask_sp.shape[0]):
        a,b,_ = create_bb_coord_correlation(np.sum(mask_sp[j,:,:,:],axis=2)) 
        coord_st_l.append(a[0])
        coord_cir_l.append(b[0])


    err_corr = comp_err_correlation(coord_st_l,coord_cir_l,stack,mask_sp)
    th_corr =err_corr.find_threshold()

    #######################

    mask_out  = np.zeros((mask_sp.shape[0],N,M,3))
    mask_out[:,:,:,:2] = mask_sp
    
    mask_single_corr = np.zeros((len(coord_st_l),2*r,2*r))
    mask_single_corr2 = np.zeros((len(coord_st_l),2*r,2*r))

    for j in range(mask_sp.shape[0]):

        #########################################pick coord
        coord_bb = coord_st_l[j]

        ########################################define buffer
        mask_crop = np.zeros((2,r*2,r*2))
        mask_crop[0,:,:] = mask_sp[j,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2],0]
        mask_crop[1,:,:] = mask_sp[j,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2],1]

        ########################################define stack for cross corr
        stack_crop = np.empty((T,2*r,2*r))
        stack_crop = stack[:,coord_bb[1]:coord_bb[3],coord_bb[0]:coord_bb[2]].copy()

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
    mask_out[mask_out>1]=1
    return mask_single_corr2,mask_out