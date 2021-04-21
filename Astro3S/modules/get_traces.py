from sklearn.cluster import KMeans
import numpy as np
import cv2

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

