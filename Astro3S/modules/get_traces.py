from sklearn.cluster import KMeans
import numpy as np
import cv2
from skimage.feature import register_translation


class Extr_miniROI():
    
    def __init__(self,Area_mu,mu_px,proc_to_split,soma,split_proc=False,dilate_ROI=2):
        
        
        self.Area_mu = Area_mu
        self.mu_px = mu_px
        
        self.Area_px = Area_mu/mu_px**2
        self.proc_to_split = proc_to_split
        self.soma = soma
        self.split_proc = split_proc
        self.dilate_ROI=dilate_ROI
    
    @staticmethod
    def det_conn_comp(processes,soma,dilate_ROI):
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
            buff = buff-soma-np.sum(processes_out,axis=0)
            processes_out[k-1,:,:]=buff
        processes_out[processes_out>0]=1
        processes_out[processes_out!=1]=0
        print('mat_out',processes_out.shape)
        return processes_out

    def get_k(self,area):
        return area/self.Area_px
    
    def get_miniROI(self):
        if self.split_proc :
            self.proc_to_split = self.det_conn_comp(self.proc_to_split,self.soma,self.dilate_ROI)
        print('check size',self.proc_to_split.shape)
        Nroi,H,W = self.proc_to_split.shape
        collROI = []
        for i in range(Nroi):
            N = np.sum(self.proc_to_split[i,:,:])
            k = self.get_k(N)
            print('ratio',k,int(k))
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
        for i in range(len(collROI)):  
            #print(collROI[i].shape)
            out = np.dstack((out,collROI[i]))
        print('SPLIT DONE',out.shape)
        return out
    
    
    
def get_signals(roi,stack):
    T,_,_ = stack.shape
    _,_,N = roi.shape
    print(roi.shape)
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
            # coordinate cell
#             contours_poly = cv2.approxPolyDP(c, 3, True)
#             boundRect = cv2.boundingRect(contours_poly)

#             coord = np.array([int(boundRect[0])-2,int(boundRect[1])-2,int(boundRect[0])+int(boundRect[2])+2,int(boundRect[1])+int(boundRect[3])+2])
#             for el in range(4):
#                 if coord[el]>N:
#                     coord[el]=N
#                 elif coord[el]<0:
#                     coord[el]=0
#             coord_list_cell.append(coord)

            # coordinate bb 100
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"]+1e-5))
            cY = int(M["m01"] / (M["m00"]+1e-5))
            
#             casex=2
#             casey=2
#             if cX-radius<0:
#                 casex=0
#             elif cX+radius>N:
#                 casex=1
#             if cY-radius<0:
#                 casey=0
#             elif cY+radius>N:
#                 casey=1
#             #x
#             if casex==2:
#                 c1x=cX-radius
#                 c2x=cX+radius
#             elif casex==0:
#                 c1x=0
#                 c2x=2*radius
#             else:
#                 c1x=N-2*radius
#                 c2x=N
#             #y
#             if casey==2:
#                 c1y=cY-radius
#                 c2y=cY+radius
#             elif casey==0:
#                 c1y=0
#                 c2y=2*radius
#             else:
#                 c1y=N-2*radius
#                 c2y=N
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

    
class Motion_Correction():
    """ inputs: 
        - im_stream numpy array of shape TxNxM (N and M can be equal) 
        - ref_image an image numpy array of shape NxM(same dim of im_stream)
        - eventually ref_frame a scalar
        outputs:
        - shift 
        This class return a stream of image of shape TxNxN (numpy array), all the images are corrected with the algorithm 
        Guizar-Sicairos et al., “Efficient subpixel image registration algorithms,”Optics Letters 33, 156-158 (2008).
        
        Obs.
        -This class must be initialized with the pixel precision that are required
        -If ref_image and ref_frame are not given the reference is the first frame
    """
    def __init__(self,pix_precision):
        self.pix_precision = pix_precision
        

    def motion_corr(self,im_stream,ref_image=None,ref_frame=None):
        if (ref_frame != None and ref_image.all() != None):
            raise Exception('ref_frame or ref_image must be None')
                    
        T,cols,rows = im_stream.shape

        if(ref_image.all()!=None):
            pass
        elif(ref_frame!=None):
            ref_image = im_stream[ref_frame,:,:]
        else:
            ref_image = im_stream[0,:,:]

        shift_vec = np.empty((T,2),dtype=np.float64)
        for i in range(T):
            shift_vec[i,:],_,_ = register_translation(ref_image, im_stream[i,:,:],upsample_factor=self.pix_precision)

        X_shift = np.array([np.arange(T),shift_vec[:,1]])
        Y_shift = np.array([np.arange(T),shift_vec[:,0]])
        
        self.X_shift = X_shift
        self.Y_shift = Y_shift
        return shift_vec.astype(np.int64)
    
    def apply_corr(self,im_stream2):
        
        T,cols,rows = im_stream2.shape
        for i in range(T):
            
            M = np.float32([[1,0,self.X_shift[i,1]],[0,1,self.Y_shift[i,0]]])
            im_stream2[i,:,:] = cv2.warpAffine(im_stream2[i,:,:],M,(rows,cols))
        return im_stream2
        