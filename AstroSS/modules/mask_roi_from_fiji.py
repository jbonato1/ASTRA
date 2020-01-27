import imageManip_library as im_man
import numpy as np
import os
import sys
#import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import grid_points_in_poly as isInROI

class create_mask():
    """Create NxNxT tensor where T is the number of roi where in each slice there is a mask with all the roi 
       Input: path of the file and eventually the dim of the image
       Output: Roi_mask NxNxT, roi_label T 
    """
    def __init__(self,path):
        self.path = path
        self.counter,self.roi_xy = im_man.read_roi_zip2(path)
        self.roi_xy = np.array(self.roi_xy)
        #print('shape',self.roi_xy.shape[0])
        pass

    def get_dim(self):
        return self.roi_xy.shape[0]

    def create_multiple_mask_im(self,im_dim=None):
        if im_dim == None:
            im_dim = 256 #512
        
        
        num_roi = self.roi_xy.shape[0]

        #da controllare che tipo
        mask = np.zeros(shape=(im_dim,im_dim,num_roi), dtype = np.int64)
        w = np.chararray(shape=(num_roi,))
        for i in range(num_roi):
            mask[:,:,i] = isInROI((im_dim,im_dim),self.roi_xy[i])
            w[i] = 'roi' 
        return self.counter,mask

if __name__ == '__main__':
    path = '/Users/jbonato/Desktop/cartella_prova/RoiSetSMALL9.zip'
    create_mask_cl = create_mask(path)

    W,A = create_mask_cl.create_multiple_mask_im() 
    print('Soma',W)
    plt.imshow(np.sum(A[:,:,:W],axis=2))
    plt.show()
    
