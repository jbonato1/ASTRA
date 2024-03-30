import numpy as np
import cv2

from skimage.registration import phase_cross_correlation

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
    def __init__(self,pix_precision,gpu=False):
        self.pix_precision = pix_precision
        self.gpu = gpu
        if self.gpu:
            print('loading gpu_mod')
            from compute_shift_gpu import register_translation_gpu
            self.register_translation_gpu = register_translation_gpu
    
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
        if self.gpu:
            shift_vec = self.register_translation_gpu(ref_image, im_stream,upsample_factor=self.pix_precision)
        else:
            for i in range(T):
                shift_vec[i,:],_,_ = phase_cross_correlation(ref_image, im_stream[i,:,:],upsample_factor=self.pix_precision)

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
    
    
