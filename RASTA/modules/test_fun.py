import numpy as np
import cv2 
from sklearn.metrics import f1_score 
from joblib import Parallel, delayed

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

def score(y_pred,y_true):
    return f1_score(y_true,y_pred,average='binary')

def clean_art(soma,proc,N,bound=0):
    
    mask_tot = soma+proc
    mask_tot[mask_tot>0.5]=255
    mask_tot[mask_tot<=0.5]=0
    mask_tot= np.uint8(mask_tot)
    ret, labels = cv2.connectedComponents(mask_tot)
    # N=200 std N = 150 for segm
    for i in range(1, ret+1):
        pts =  np.where(labels == i)

        if len(pts[0]) < N:
    #         print(len(pts[0]))
            labels[pts] = 0
        else:
            #print(len(pts[0]))
            labels[pts] = 255
    mask_tot=labels
    labels_out= labels.copy()
    labels_out[labels_out!=255]=0

    labels_out[labels_out==255]=1
    #draw bounding box
    
    ret, thresh = cv2.threshold(np.uint8(mask_tot), 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        
        contours_poly[i] = cv2.approxPolyDP(c, 1, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    

    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)    

    for i in range(len(contours)):
        color = (255,0,0)
        #drawing =  cv2.drawContours(drawing, contours_poly, i, (255,0,0))
        cv2.rectangle(drawing, (int(boundRect[i][0])-bound, int(boundRect[i][1])-bound),(int(boundRect[i][0]+boundRect[i][2])+bound, int(boundRect[i][1]+boundRect[i][3])+bound), color, cv2.FILLED)
     
    #plt.imshow(drawing)
    #plt.show()
    
    drawing[drawing>0]=1
    if bound==0:
        return labels_out
    else:
        return drawing[:,:,0]

def clean_soma_art(soma_mask,soma_target):
    

    soma_mask[soma_mask>0.5]=255
    soma_mask[soma_mask<=0.5]=0
    soma_mask= np.uint8(soma_mask)
    soma_target[soma_target>0.5]=255
    soma_target[soma_target<=0.5]=0
    soma_target= np.uint8(soma_target)
   
    ret, labels = cv2.connectedComponents(soma_mask)
    ret1, labels1 = cv2.connectedComponents(soma_target)
  
    for i in range(1, ret+1):
        pts =  np.where(labels == i)
        mask_tmp = np.zeros((96,96))
        mask_tmp[pts]=1
        for j in range(1, ret1+1):
            pts1 = np.where(labels1 == j)
            mask_tmp1 = np.zeros((96,96))
            mask_tmp1[pts1]=1
            mask_tmp+=mask_tmp1
        if np.amax(mask_tmp)<2:
            soma_mask[pts]=0
    soma_mask[soma_mask==255]=1

    return soma_mask

def small_soma_to_proc(soma_mask,N,dilation=True,watershed=False):
    mask_tot = soma_mask.copy()
    if dilation:
        kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

        mask_tot = cv2.dilate(np.uint8(mask_tot),kernel,iterations = 1)
        kernel2=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        mask_tot = cv2.morphologyEx(np.uint8(mask_tot), cv2.MORPH_CLOSE, kernel)    
    
    
    mask_tot[mask_tot>0.5]=255
    mask_tot[mask_tot<=0.5]=0
    mask_tot= np.uint8(mask_tot)
    ret, labels = cv2.connectedComponents(mask_tot)
    labels = labels*soma_mask
    for i in range(1, ret+1):
        pts =  np.where(labels == i)

        if len(pts[0]) <= N:
            labels[pts] = 1
        else:
            labels[pts] = 0
            #if watershed:
                
                
    mask_tot=labels
    return mask_tot

def small_roi(mask,N=150):
    mask_tot = mask.copy()
    mask_tot[mask_tot>0.5]=255
    mask_tot[mask_tot<=0.5]=0
    mask_tot= np.uint8(mask_tot)
    ret, labels = cv2.connectedComponents(mask_tot)
    
    for i in range(1, ret+1):
        pts =  np.where(labels == i)

        if len(pts[0]) >= N:
   #         print(len(pts[0]))
            labels[pts] = 1
        else:
            #print(len(pts[0]))
            labels[pts] = 0
    mask_tot=labels
    return mask_tot

def large_soma(mask,N):
    mask_tot = mask.copy()
    mask_tot[mask_tot>0.5]=255
    mask_tot[mask_tot<=0.5]=0
    mask_tot= np.uint8(mask_tot)
    ret, labels = cv2.connectedComponents(mask_tot)
    
    for i in range(1, ret):
        pts =  np.where(labels == i)

        if len(pts[0]) >= N:
            print("Too large zone",len(pts[0]))
            labels[pts] = 1
        else:
            #print(len(pts[0]))
            labels[pts] = 0
    mask_tot=labels
    return mask_tot

def art_rem_large(soma,proc=None,N=300):
    cnt=0
    soma_l = large_soma(soma,N)
    N,M = soma.shape
    if proc is None:
        ret_m, labels_m = cv2.connectedComponents(np.uint8(soma))
    else:
        ret_m, labels_m = cv2.connectedComponents(np.uint8(soma+proc))
    mask_fin = np.zeros((N,M))
    for i in range(1, ret_m):
        mask_buff = np.zeros((N,M))
        
        pts_m = np.where(labels_m==i)
        mask_buff[pts_m]=1
        
        if(np.sum(mask_buff*soma_l)>0):
            mask_fin[pts_m]=1
            cnt+=1
    return mask_fin,cnt


def art_rem(soma,proc):
    N,M = soma.shape
    ret_m, labels_m = cv2.connectedComponents(np.uint8(soma+proc))
    cells= soma+proc
    mask_fin = np.zeros((N,M))
    for i in range(1, ret_m):
        mask_buff = np.zeros((N,M))
        
        pts_m = np.where(labels_m==i)
        mask_buff[pts_m]=1
        
        if(np.sum(mask_buff*soma)>0):
            mask_fin[pts_m]=1
    return mask_fin

def dilation_fun(sp_filt,proc,dilatation_size=3):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    dil_sp_filt = cv2.dilate(sp_filt, element)
    dil_proc = cv2.dilate(proc, element)
    q =dil_proc+dil_sp_filt
    q[q>1]=1
    return q

def filter_sm_activity(stack):
    T,N,M=stack.shape
    if T>550:
        flag_fil=True
    else:
        flag_fil=False
    percent = np.percentile(stack.reshape(T*N*M),90)
    stack_cp = stack.copy()
    stack_cp[stack<percent]=0
    stack_cp[stack>=percent]=1

    sum_= np.sum(stack_cp,axis=0)
    if flag_fil:
        sum_[sum_<200]=0
        sum_[sum_>=200]=1
    else:
        sum_[sum_<150]=0
        sum_[sum_>=150]=1
    return sum_

def count_soma(soma):
    ret_m, labels_m = cv2.connectedComponents(np.uint8(soma))
    return ret_m-1

def no_proc_soma(mask_proc, mask_soma):
    soma_to_check = np.zeros_like(mask_proc)
    mask_cp = mask_proc + mask_soma
    ret, labels = cv2.connectedComponents(np.uint8(mask_cp))
    ret_s, labels_s = cv2.connectedComponents(np.uint8(mask_soma))
    
    for i in range(1,ret):
        buff_cell = np.zeros_like(mask_proc)
        pts = np.where(labels==i)
        buff_cell[pts]=1
        for j in range(1,ret_s):
            buff_soma = np.zeros_like(mask_proc)
            pts_s = np.where(labels_s==j)
            buff_soma[pts_s]=1
            if np.sum(buff_soma*buff_cell)>0:
                buff_cell-=buff_soma
                if np.sum(buff_cell)<np.sum(buff_soma)*0.1:
                    soma_to_check+=buff_soma
            
    return soma_to_check
    
    return soma_tocheck


def fix_mask(mask):
    N,M = mask.shape
    mask_fixed = mask.copy()
    #dilation
    dilatation_size = 2
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2*dilatation_size + 1, 2*dilatation_size+1), (dilatation_size, dilatation_size))
    mask_dilated =  cv2.dilate(np.uint8(mask), element)
    ret_m, labels_m = cv2.connectedComponents(np.uint8(mask_dilated))

    mask_fin = np.zeros((N,M))
    for i in range(1, ret_m):
        mask_cp = mask.copy()
        mask_buff = np.zeros((N,M),dtype=np.uint8)
        pts_m = np.where(labels_m==i)
        mask_buff[pts_m]=1
        mask_cp*=mask_buff
        ret, labels = cv2.connectedComponents(np.uint8(mask_cp))

        if ret-1>1:
            min_cell=np.inf
            for j in range(1, ret):
                pts_cell = np.where(labels==j)
               
                if len(pts_cell[0])<min_cell:
                    out_index = j
                    min_cell=len(pts_cell[0])
                    
            pts_rem = np.where(labels==out_index)
            mask_fixed[pts_rem]=0
    return mask_fixed



def prob_calc(prob_map,max_a,min_a,verbose=False):
    font= cv2.FONT_HERSHEY_DUPLEX
    map_ = np.zeros_like(prob_map)
    map_[prob_map>0]=1
    ret, labels = cv2.connectedComponents(np.uint8(map_))
    for i in range(1,ret):
        pts = np.where(labels==i)
        if len(pts[0])>int(0.9*min_a) and len(pts[0])<int(1.15*max_a):
            
            q = np.around(np.sum(prob_map[pts])/len(pts[0]),decimals=2)
            cv2.putText(prob_map,str(q), (int(np.mean(pts[1])),int(np.mean(pts[0]))), font, 0.4, (2,0,0), 1, cv2.LINE_AA)
            if verbose: print('Area in px: ',len(pts[0]),'Prob:', q)
            if q<90 and len(pts[0])<min_a:
                map_[pts]=0
        else:
            if verbose: print('Area: ',len(pts[0]), 'removed')
            map_[pts]=0
            if len(pts[0])>int(max_a*1.15):                
                prob_map[pts]=0
            
    return prob_map,map_

def common_merge(sm_fr,sm_ent):
    N,M = sm_fr.shape
    ret, labels = cv2.connectedComponents(np.uint8(sm_fr))
    ret1, labels1 = cv2.connectedComponents(np.uint8(sm_ent))
    
    merge = np.zeros((N,M))
    
    for i in range(1, ret):
        pts =  np.where(labels == i)
        mask_tmp = np.zeros((N,M))
        mask_tmp[pts]=1
        for j in range(1, ret1):
            pts1 = np.where(labels1 == j)
            mask_tmp1 = np.zeros((N,M))
            mask_tmp1[pts1]=1
            if len(pts1[0])>len(pts[0]):
                ref = len(pts[0])
            else:
                ref = len(pts1[0])
            if np.sum(mask_tmp*mask_tmp1)>0.1*ref:
                merge+=mask_tmp
                merge+=mask_tmp1
    
    merge[merge>1]=1
    return merge


def common_merge_par(sm_fr,sm_ent):
    merge = np.zeros_like(sm_fr)
    
    def merge_masks(i,labels_fr,labels_ent,ret_ent):
        
        N,M = labels_fr.shape
        merged = np.zeros((N,M))
        
        pts =  np.where(labels_fr == i)
        mask_tmp = np.zeros((N,M))
        mask_tmp[pts]=1
        for j in range(1, ret_ent):
            pts1 = np.where(labels_ent == j)
            mask_tmp1 = np.zeros((N,M))
            mask_tmp1[pts1]=1
            if len(pts1[0])>len(pts[0]):
                ref = len(pts[0])
            else:
                ref = len(pts1[0])
            if np.sum(mask_tmp*mask_tmp1)>0.1*ref:
                merged+=mask_tmp
                merged+=mask_tmp1
        return merged
    
    N,M = sm_fr.shape
    ret, labels = cv2.connectedComponents(np.uint8(sm_fr))
    ret1, labels1 = cv2.connectedComponents(np.uint8(sm_ent))
    
    
    
    out_el = Parallel(n_jobs=-1,verbose=0,require='sharedmem')(delayed(merge_masks)(i,labels,labels1,ret1)for i in range(1, ret))
    for out in out_el:
        merge+=out
    
    merge[merge>1]=1
    return merge




def create_bb_split(soma_mask):
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
    centre_coord = []
    for c in contours:
        im_buff = np.zeros((N,N))

        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / (M["m00"]+1e-5))
        cY = int(M["m01"] / (M["m00"]+1e-5))


        cv2.circle(im_buff,(cX,cY),40,(255,0,0),thickness =-1,lineType=8)
        coord_circle = np.where(im_buff==255)
        coord_list_circle.append(coord_circle)
        centre_coord.append([cX,cY])
        
    return coord_list_circle,centre_coord


def gen_sc_mask(mask):
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
        print(soma_num)
        if soma_num>2:

            print('Split CComp')
            circ_coord,centre_coord = create_bb_split(cnt_m)
            
            
            cnt_soma=0    
            for circ in circ_coord:
                circ_f =np.zeros_like(cnt_m)
                circ_f[circ[0],circ[1]]=1
                instance = np.zeros((N,M,2))
                
                for sm in range(1,soma_num):
                    pt_soma = np.where(mark==sm)
                    if (centre_coord[cnt_soma][1] in pt_soma[0]) and (centre_coord[cnt_soma][0] in pt_soma[1]):
                        instance[pt_soma[0],pt_soma[1],1]=1
                
                instance[:,:,0]=proc_single_m*circ_f
                
                ret_instance, labels_instance = cv2.connectedComponents(np.uint8(np.sum(instance,axis=2)))
                for inst in range(1,ret_instance):
                    pt_dealloc = np.where(labels_instance==inst)
                    buff_inst = np.zeros((N,M))
                    buff_inst[pt_dealloc[0],pt_dealloc[1]]=1
                    
                    if np.sum(buff_inst*instance[:,:,1])==0:
                        instance[pt_dealloc[0],pt_dealloc[1],0]=0
                        
                if np.sum(instance[:,:,1])>0:
                    if cnt_soma>0:
                        instance[:,:,0]-=inst_list[-1][:,:,0]
                        instance[instance<0]=0
                    inst_list.append(instance)
                cnt_soma+=1
                
  

        elif soma_num==2:  
        
            instance = np.zeros((N,M,2))
            instance[:,:,0]=proc_single_m
            instance[:,:,1]=cnt_m
            inst_list.append(instance)
            
    inst_list = np.asarray(inst_list)
    return inst_list 
