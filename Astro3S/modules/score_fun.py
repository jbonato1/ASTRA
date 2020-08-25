import numpy as np
import h5py 
from sklearn.metrics import f1_score,jaccard_similarity_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as pRf1
import glob
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import linear_sum_assignment

def create_bb(soma_mask):
    N,_ = soma_mask.shape
    soma = np.empty((N,N))
    soma = soma_mask.copy()
    soma[soma>0.1]=255

    _,thresh = cv2.threshold(np.uint8(soma),127,255,0)

    # find contours in the binary image
    contours, _= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    #list of array with the coordinate
    coord_list_st = []
    coord_list_cell = []
    coord_list_circle = []
    for c in contours:
        im_buff = np.zeros((N,N))

        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / (M["m00"]+1e-5))
        cY = int(M["m01"] / (M["m00"]+1e-5))


        cv2.circle(im_buff,(cX,cY),40,(255,0,0),thickness =-1,lineType=8)
        coord_circle = np.where(im_buff==255)
        coord_list_circle.append(coord_circle)
       
    return coord_list_circle

def IoU(a,b):
    ###
    I = a*b
    U = a+b
    U[U>1]=1
    return np.sum(I)/np.sum(U)

# def gen_sc_mask(mask):
#     N,M = mask.shape
#     ret, labels = cv2.connectedComponents(np.uint8(mask.copy()))
#     sc_mask = np.zeros((ret-1,N,M))
#     for i in range(1, ret):        
#         pts_s = np.where(labels==i)
#         sc_mask[i-1,pts_s[0],pts_s[1]]=1
#     return sc_mask
    
def true_pos_l(soma_GT,soma,threshold=0.5,N=256):
    ###
    ret_m, labels_m = cv2.connectedComponents(np.uint8(soma.copy()))
    ret_s, labels_s = cv2.connectedComponents(np.uint8(soma_GT.copy()))
    
    cnt=0
    for i in range(1, ret_s):
        soma_GT_buff = np.zeros((N,N))
        pts_s = np.where(labels_s==i)
        soma_GT_buff[pts_s]=1
        
        for j in range(1,ret_m):
            soma_buff = np.zeros((N,N))
            pts_m = np.where(labels_m==j)
            soma_buff[pts_m]=1
            inter = soma_buff*soma_GT_buff
            if np.sum(inter)==np.sum(soma_buff) or np.sum(inter)==np.sum(soma_buff):
                cnt+=1
            else:
                score = IoU(soma_GT_buff.flatten(),soma_buff.flatten())
                if score >= threshold:
                    cnt+=1
            
    return cnt,(ret_m-1),(ret_s-1)


def gen_sc_mask(mask):
    ###
    N,M,cl = mask.shape
    
    ret_s, labels_s = cv2.connectedComponents(np.uint8(np.sum(mask,axis=2)))
    inst_list = []
    qq= np.zeros((N,M,3))
    
    for j in range(1,ret_s):
        
        CC = np.zeros((N,N))
        pts_s = np.where(labels_s==j)
        CC[pts_s]=1            
        cnt_m = CC*mask[:,:,1]
        proc_single_m = CC*mask[:,:,0]
        soma_num,mark = cv2.connectedComponents(np.uint8(cnt_m.copy()))
        mark+=1
        
        if soma_num>2:
            circ_coord = create_bb(cnt_m)

            for circ in circ_coord:
                circ_f =np.zeros_like(cnt_m)
                circ_f[circ[0],circ[1]]=1
                instance = np.zeros((N,M,2))
                instance[:,:,0]=proc_single_m*circ_f
                instance[:,:,1]=cnt_m*circ_f
                inst_list.append(instance)
        else:  
            instance = np.zeros((N,M,2))
            instance[:,:,0]=proc_single_m
            instance[:,:,1]=cnt_m
            inst_list.append(instance)
            
    inst_list = np.asarray(inst_list)
    return inst_list    
    
def score_cw(GT,mask,):
    ###
    N,M,cl = GT.shape
    GTs = gen_sc_mask(GT)
    masks = gen_sc_mask(mask)    
    proc_l = []
    soma_l = []
    for i in range(GTs.shape[0]):
        for j in range(masks.shape[0]):
            GT_sample = GTs[i,:,:,:]
            mask_sample = masks[j,:,:,:]
            GT_sample[GT_sample>1]=1
            mask_sample[mask_sample>1]=1
            
            if IoU(GT_sample[:,:,1],mask_sample[:,:,1])>=0.5:
                res = pRf1(GT_sample[:,:,1].flatten(),mask_sample[:,:,1].flatten(),average='binary')
                soma_l.append(res)
                res = pRf1(GT_sample[:,:,0].flatten(),mask_sample[:,:,0].flatten(),average='binary')
                proc_l.append(res)
                
                
    final_res = np.dstack((np.asarray(soma_l),np.asarray(proc_l))) 
    return final_res            
#####################################################################################################                
def filter_GT1(GT,mask):
    N,M,cl = GT.shape
    ret_m, labels_m = cv2.connectedComponents(np.uint8(mask))
    
    ret_s, labels_s = cv2.connectedComponents(np.uint8(np.sum(GT,axis=2)))
    cnt=0
    final = np.zeros((256,256))
    for i in range(1, ret_s):
        soma_GT_buff = np.zeros((N,N))
        pts_s = np.where(labels_s==i)
        soma_GT_buff[pts_s]=1
        rem = soma_GT_buff.copy()
        soma_GT_buff*=GT[:,:,1]
        
        for j in range(1,ret_m):
            soma_buff = np.zeros((N,N))
            pts_m = np.where(labels_m==j)
            soma_buff[pts_m]=1
            inter = soma_buff*soma_GT_buff
            if np.sum(inter)==np.sum(soma_buff) or np.sum(inter)==np.sum(soma_buff):
                final+=rem
            else:
                score = IoU(soma_GT_buff.flatten(),soma_buff.flatten())
                if score >= 0.5:
                    final+=rem
            
    return final  
    
           
def filt_art(soma):
    N,M = soma.shape
    mask = np.zeros_like(soma)
    ret_m, labels_m = cv2.connectedComponents(np.uint8(soma.copy()))
    #plt.imshow(labels_m)
    for j in range(1,ret_m):
        soma_buff = np.zeros((N,N))
        pts_m = np.where(labels_m==j)
        soma_buff[pts_m]=1 
        if np.sum(soma_buff)>500:
            mask+=soma_buff
    return mask
    
def true_pos_l_c(soma_GT,soma_t,threshold=0.5,N=256):
    soma = filt_art(soma_t)
    ret_m, labels_m = cv2.connectedComponents(np.uint8(soma.copy()))
    ret_s, labels_s = cv2.connectedComponents(np.uint8(soma_GT.copy()))
    cnt=0
    for i in range(1, ret_s):
        soma_GT_buff = np.zeros((N,N))
        pts_s = np.where(labels_s==i)
        soma_GT_buff[pts_s]=1
        
        for j in range(1,ret_m):
            soma_buff = np.zeros((N,N))
            pts_m = np.where(labels_m==j)
            soma_buff[pts_m]=1
            inter = soma_buff*soma_GT_buff

            if np.sum(inter)==np.sum(soma_buff) and np.sum(inter)!=0:
                cnt+=1
            else:
                score = IoU(soma_GT_buff.flatten(),soma_buff.flatten())
                if score >= threshold:
                    cnt+=1
               

    return cnt,(ret_m-1),(ret_s-1)


def filter_GT(GT,mask):
    N=512
    ret_m, labels_m = cv2.connectedComponents(np.uint8(mask))
    maskGT_T = np.sum(GT,axis=2)
    maskGT_T[maskGT_T>1]=1
    qq = dilation_fun(maskGT_T)
    ret_s, labels_s = cv2.connectedComponents(np.uint8(qq))
    cnt=0
    final = np.zeros((512,512))
    for i in range(1, ret_s):
        soma_GT_buff = np.zeros((N,N))
        pts_s = np.where(labels_s==i)
        soma_GT_buff[pts_s]=1
        rem = soma_GT_buff.copy()
        soma_GT_buff*=GT[:,:,1]
        
        for j in range(1,ret_m):
            soma_buff = np.zeros((N,N))
            pts_m = np.where(labels_m==j)
            soma_buff[pts_m]=1
            inter = soma_buff*soma_GT_buff
            #print(IoU(soma_GT_buff.flatten(),soma_buff.flatten()))
            if np.sum(inter)==np.sum(soma_buff) or np.sum(inter)==np.sum(soma_buff):
                final+=rem*maskGT_T
            else:
                score = IoU(soma_GT_buff.flatten(),soma_buff.flatten())
                if score >= 0.2:
                    final+=rem*maskGT_T
            
    return final

def dilation_fun(sp_filt,dilatation_size=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dil_sp_filt = cv2.dilate(sp_filt, element)
    q =dil_sp_filt
    q[q>1]=1
    return q

def F1_cw_nd_car(GT_o,mask,flag_t=True):
    GT = GT_o.copy()
    proc_l = []
    soma_l = []
    
    N,M,cl = GT.shape
    
    #plt.imshow(mask[:,:,0])
    if flag_t:
        mask_GT  = filter_GT(GT,mask[:,:,1])
    else:
        mask_GT  = filter_GT(GT,mask[:,:,0])
        
        
    GT*=mask_GT[:,:,np.newaxis]
    
    #plt.imshow(GT[:,:,1])
    gg=np.sum(GT,axis=2)
    gg[gg>1]=1
    gg = dilation_fun(gg)
    ret_m, labels_m = cv2.connectedComponents(np.uint8(gg))
    #print('qwqwq',ret_m-1)
    inc_mask = np.sum(mask,axis=2)
    q = dilation_fun(inc_mask)
    
    ret_s, labels_s = cv2.connectedComponents(np.uint8(q))
    #plt.imshow(labels_m)
    res = np.empty((4,))
    flag = True
    

    for j in range(1,ret_s):
        soma_buff = np.zeros((N,N))
        pts_s = np.where(labels_s==j)
        soma_buff[pts_s]=1            
        flag = False
        if flag_t:
            cnt_m = soma_buff*mask[:,:,1]
            proc_single_m = soma_buff*mask[:,:,0]
        else:
            cnt_m = soma_buff*mask[:,:,0]
            proc_single_m = soma_buff*mask[:,:,1]
        soma_num,_ = cv2.connectedComponents(np.uint8(cnt_m.copy()))
        
 
        for i in range(1,ret_m):
            GT_buff = np.zeros((N,N))
            pts_m = np.where(labels_m==i)
            GT_buff[pts_m]=1
            soma_single = GT[:,:,1]*GT_buff
            proc_single = GT[:,:,0]*GT_buff
            soma_single[soma_single>1]=1
            proc_single[proc_single>1]=1
            #print('SOMMA',np.sum(soma_single*cnt_m))
            
            if IoU(soma_single,cnt_m)>=0.5:                   
                res = pRf1(soma_single.flatten(),cnt_m.flatten(),average='binary')
                soma_l.append(res)
                res = pRf1(proc_single.flatten(),proc_single_m.flatten(),average='binary')
                proc_l.append(res)

    final_res = np.dstack((np.asarray(soma_l),np.asarray(proc_l))) 
    return final_res
 