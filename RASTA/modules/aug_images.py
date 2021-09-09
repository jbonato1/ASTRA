import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage import io
import random

class elastic_deform():
    def __init__(self,alpha, sigma, alpha_affine,iteration,random_state=None):
        self.random_state = random_state
        if self.random_state is None:
            self.random_state = np.random.RandomState(None)
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.it = iteration
       
    def elastic_transf(self,im,mask):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
                Convolutional Neural Networks applied to Visual Document Analysis", in
                Proc. of the International Conference on Document Analysis and
                Recognition, 2003.

            Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        
        alpha = float(self.alpha)
        sigma = float(self.sigma)
        alpha_affine = float(self.alpha_affine)
        shape = im.shape
        img_out = np.empty((self.it,shape[0],shape[1]),dtype=im.dtype)
        mask_cont = np.empty((self.it,3,shape[0],shape[1]),dtype=mask.dtype)
        shape_size = shape[:2]
        image = np.empty((shape[0],shape[1],3))
        im_buff = np.empty_like(image)

        image = np.dstack((im[:,:,np.newaxis],mask)) 
        # Random affine pts1 punti di rif in im originale pts2 nuovi punti spostati a random con distrib 
        #uniforma nell'intervallo +- alpha affine 
        
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3

        pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
       
        for ii in range(self.it):
            pts2 = pts1 + self.random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)

            M = cv2.getAffineTransform(pts1, pts2)#crea la mappa affine
            im_buff = cv2.warpAffine(image, M, shape_size, borderMode=cv2.BORDER_CONSTANT,borderValue=0)



            dx = gaussian_filter((self.random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((self.random_state.rand(*shape_size) * 2 - 1), sigma) * alpha
            dz = np.zeros_like(dx)

            x, y= np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
            mapx = np.float32(x+dx)
            mapy = np.float32(y+dy)

            im_buff = cv2.remap(im_buff,mapx,mapy,interpolation= cv2.INTER_LINEAR, borderMode= cv2.BORDER_CONSTANT,borderValue=0)
            
            img_out[ii,:,:]=im_buff[:,:,0]
            mask_cont[ii,0,:,:] = im_buff[:,:,1]
            mask_cont[ii,1,:,:] = im_buff[:,:,2]
            mask_cont[ii,2,:,:] = np.ones((shape[0],shape[1]))-im_buff[:,:,1]-im_buff[:,:,2]

        return img_out, mask_cont
class optic_bar():
    def __init__(self,bar_fact):
        self.bar_fact = bar_fact
        
    def barrel_transf(self, im, mask):
        
        N,M = im.shape
        mask_cont = np.empty((3,N,M))
        mask_out = np.empty_like(mask)
        
        # Define camera matrix K
        K = np.array([[100, 0., N/2],
                    [0.,100, M/2],
                    [0., 0., 1.]])
        
        # Define distortion coefficients d
        d = np.array([self.bar_fact, 0, 0, 0, 0])
       
        # Generate new camera matrix from parameters
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (N,M), 0)

        # Generate look-up tables for remapping the camera image
        mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (N, M), 5)

        mask_out = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)
        
        mask_cont[0,:,:]=mask_out[:,:,0]
        mask_cont[1,:,:]=mask_out[:,:,1]
        mask_cont[2,:,:]=np.ones((N,M))-mask_out[:,:,0]-mask_out[:,:,1]
        # Remap the original image to a new image
        return cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR),mask_cont


class optic_pin():
    def __init__(self,pin_fact):
        self.pin_fact = pin_fact
    def pincushion_transf(self, im, mask):
        N,M = im.shape
        mask_cont = np.empty((3,N,M))
        mask_out = np.empty_like(mask)
        
        #800->-0.004
        #400->-0.02
        #200->-0.1
        #100->-0.5
        #50->-2.5
        # Define camera matrix K
        K = np.array([[100, 0., N/2],
                    [0.,100, M/2],
                    [0., 0., 1.]])

        # Define distortion coefficients d
        d = np.array([self.pin_fact, 0, 0, 0, 0])
       
        # Generate new camera matrix from parameters
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (N,M), 0)

        # Generate look-up tables for remapping the camera image
        mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (N, M), 5)


        # Remap the original image to a new image
        Matrix = cv2.getRotationMatrix2D((M/2,N/2),180,1)
       
        mask_out = cv2.warpAffine(cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR),Matrix,(M,N))
        
        mask_cont[0,:,:]=mask_out[:,:,0]
        mask_cont[1,:,:]=mask_out[:,:,1]
        mask_cont[2,:,:]=np.ones((N,M))-mask_out[:,:,0]-mask_out[:,:,1]
        return cv2.warpAffine(cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR),Matrix,(M,N)), mask_cont

class rotation():
    """execute 3 rotation on the input image and mask. Angle 90, 180, 270
    Return the three transformation in stacked numpy one for image e one for mask
    NxMx3
    """
    def __init__(self):
        pass
    def rotation_transf(self,im,mask):

        N,M,inst = mask.shape
        angles = [90,180,270]
        img_out = np.empty((3,N,M),dtype=im.dtype)
        mask_out = np.empty((3,N,M,inst+1),dtype=mask.dtype)
        counter = 0 
        for i in angles:

            Matrix = cv2.getRotationMatrix2D((M/2,N/2),i,1)
            
            img_out[counter,:,:] = cv2.warpAffine(im,Matrix,(M,N))
            mask_out[counter,:,:,:2] = cv2.warpAffine(mask,Matrix,(M,N))

            mask_out[counter,:,:,2]=np.ones((N,M))-mask_out[counter,:,:,0]-mask_out[counter,:,:,1]
            counter+=1
        mask_out = np.swapaxes(mask_out,1,3)
        mask_out = np.swapaxes(mask_out,2,3)
        
        return img_out,mask_out

class blurring():
    
    def __init__(self,sigma):
        self.sigma = sigma
        
    def blurring_transf(self,im,mask):
        N,M = im.shape
        img_out = np.empty_like(im)
        mask_cont = np.empty((3,N,M))
        img_out = cv2.GaussianBlur(im,(5,5),self.sigma)
        
        mask_cont[0,:,:]=mask[:,:,0]
        mask_cont[1,:,:]=mask[:,:,1]
        mask_cont[2,:,:]=np.ones((N,M))-mask[:,:,0]-mask[:,:,1]
        return img_out,mask_cont 

class scaling():
    def __init__(self):
        pass
    def scaling_transf(self, im, mask):
        N,M = im.shape
        img_out = np.empty_like(im)
        mask_cont = np.empty((3,N,M))
        
        res = cv2.resize(im ,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_AREA)
        mask_out = cv2.resize(mask ,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_AREA)
        H_center0 = res.shape[0]//2
        rem0 = N%2
        
        H_center1 = res.shape[1]//2
        rem1 = M%2
        
        
        img_out= res[H_center0-(N//2):H_center0+N//2+rem0,H_center1-(M//2):H_center1+M//2+rem1]
        mask_out= mask_out[H_center0-(N//2):H_center0+N//2+rem0,H_center1-(M//2):H_center1+M//2+rem1,:]
        
        mask_cont[0,:,:]=mask_out[:,:,0]
        mask_cont[1,:,:]=mask_out[:,:,1]
        mask_cont[2,:,:]=np.ones((N,M))-mask_out[:,:,0]-mask_out[:,:,1]
        return img_out,mask_cont
        
        
        
class scaling2():
    def __init__(self):
        pass
    def scaling_transf(self, im, mask):
        N,M = im.shape
        img_out = np.empty_like(im)
        mask_cont = np.empty((3,N,M))
        
        res = cv2.resize(im ,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)
        mask_out = cv2.resize(mask ,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)
        #padding
        a=(N-res.shape[0])//2
        b=(M-res.shape[1])//2
        arem=(N-res.shape[0])%2
        brem=(M-res.shape[1])%2
        
        img_out = cv2.copyMakeBorder(res,a,a+arem,b,b+brem,cv2.BORDER_CONSTANT,value=0)
        
        mask_out = cv2.copyMakeBorder(mask_out,a,a+arem,b,b+brem,cv2.BORDER_CONSTANT,value=0)
        

        mask_cont[0,:,:]=mask_out[:,:,0]
        mask_cont[1,:,:]=mask_out[:,:,1]
        mask_cont[2,:,:]=np.ones((N,M))-mask_out[:,:,0]-mask_out[:,:,1]
        return img_out,mask_cont
        

class perspective():
    def __init__(self,pts2):
        self.pts2 = pts2
        
    def perspective_transf(self,im,mask):
        
        N,M = im.shape  
        pts1 = np.float32([[0,0],[0,N],[M,0],[M,N]])
        mask_buff = np.empty_like(mask)
        img_out = np.empty((len(self.pts2),N,M),dtype=im.dtype)
        mask_out = np.empty((len(self.pts2),N,M,3),dtype=mask.dtype)
        
        for i in range(len(self.pts2)):
            Mat = cv2.getPerspectiveTransform(pts1,self.pts2[i])
            img_out[i,:,:] = cv2.warpPerspective(im,Mat,(N,M))
            mask_buff = cv2.warpPerspective(mask,Mat,(N,M))
            mask_out[i,:,:,0] = mask_buff[:,:,0]
            mask_out[i,:,:,1] = mask_buff[:,:,0]
            mask_out[i,:,:,2] = np.ones((N,M))-mask_buff[:,:,0]-mask_buff[:,:,1]   
        mask_out = np.swapaxes(mask_out,1,3)
        mask_out = np.swapaxes(mask_out,2,3)
        return img_out,mask_out
        
        
def GaussN(im,mask):
    N,M = im.shape
    img_out = np.empty_like(im)
    mask_cont = np.empty((3,N,M))
    
    noise = np.random.randn(N*M) 
    img_out = im+ 0.4*noise.reshape(N,M) 

    mask_cont[0,:,:]=mask[:,:,0]
    mask_cont[1,:,:]=mask[:,:,1]
    mask_cont[2,:,:]=np.ones((N,M))-mask[:,:,0]-mask[:,:,1]
    return img_out,mask_cont

def SPN(im,mask):
    
    N,M = im.shape
    img_out = np.empty_like(im)
    mask_cont = np.empty((3,N,M))
    row,col = im.shape
    s_vs_p = 0.5
    amount = 0.04
    img_out = im.copy()
    max_val=np.percentile(im,96)
    min_val=np.amin(im)
    # Salt mode
    num_salt = np.ceil(amount * im.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in im.shape]
    img_out[tuple(coords)] = max_val

    # Pepper mode
    num_pepper = np.ceil(amount* im.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in im.shape]
    img_out[tuple(coords)] = min_val
    img_out-=0#np.mean(img_out)
    #save noise
    mask_cont[0,:,:]=mask[:,:,0]
    mask_cont[1,:,:]=mask[:,:,1]
    mask_cont[2,:,:]=np.ones((N,M))-mask[:,:,0]-mask[:,:,1]
    return img_out,mask_cont
    
def flip_ver(im,mask):    
    N,M = im.shape
    img_out = np.empty_like(im)
    mask_cont = np.empty((3,N,M))
    
    img_out = np.flip(im.copy(),axis=1)#vert
    mask_out = np.flip(mask.copy(),axis=1)
    #save persp
    
    mask_cont[0,:,:]=mask_out[:,:,0]
    mask_cont[1,:,:]=mask_out[:,:,1]
    mask_cont[2,:,:]=np.ones((N,M))-mask_out[:,:,0]-mask_out[:,:,1]
    return img_out, mask_cont

def flip_or(im,mask):
    N,M = im.shape
    img_out = np.empty_like(im)
    mask_cont = np.empty((3,N,M))
    img_out = np.flip(im.copy(),axis=0)#oriz
    mask_out = np.flip(mask.copy(),axis=0)
    #save persp

    mask_cont[0,:,:]=mask_out[:,:,0]
    mask_cont[1,:,:]=mask_out[:,:,1]
    mask_cont[2,:,:]=np.ones((N,M))-mask_out[:,:,0]-mask_out[:,:,1]
    return img_out, mask_cont

def scal_int1(im,mask):
    N,M = im.shape
    img_out = np.empty_like(im)
    mask_cont = np.empty((3,N,M))
    img_out = im.copy()/0.5
    #save
  
    mask_cont[0,:,:]=mask[:,:,0]
    mask_cont[1,:,:]=mask[:,:,1]
    mask_cont[2,:,:]=np.ones((N,M))-mask[:,:,0]-mask[:,:,1]
    return img_out, mask_cont

def scal_int2(im,mask):##ev check max val
    N,M = im.shape
    img_out = np.empty_like(im)
    mask_cont = np.empty((3,N,M))
    img_out = im.copy()/3

    mask_cont[0,:,:]=mask[:,:,0]
    mask_cont[1,:,:]=mask[:,:,1]
    mask_cont[2,:,:]=np.ones((N,M))-mask[:,:,0]-mask[:,:,1]
    return img_out, mask_cont


        
def compose_tr(dict_tr):
    foo_list = []
                       
    if 'rot' in dict_tr.keys():
        rot = rotation()
        foo_list.append(rot.rotation_transf)

    #blurring
    if 'blur' in dict_tr.keys():
        param = dict_tr['blur'][1]
        blr = blurring(param['sigma'])
        foo_list.append(blr.blurring_transf)     

    #noise guassiano
    if 'noise_gauss' in dict_tr.keys():
        foo_list.append(GaussN)

    #salt&pepper noise
    if 'noise_sp' in dict_tr.keys():
        foo_list.append(SPN)

    #scaling
    if 'scal1' in dict_tr.keys():
        scal = scaling()
        foo_list.append(scal.scaling_transf)               

    if 'scal2' in dict_tr.keys():
        scal2 = scaling2()
        foo_list.append(scal2.scaling_transf)

    #perspective
    if 'persp' in dict_tr.keys():
        param = dict_tr['persp'][1]        
        persp = perspective(param['pts2m'])
        foo_list.append(persp.perspective_transf)

    #flipp
    if 'flip_ver' in dict_tr.keys():
        foo_list.append(flip_ver)
                       
    if 'flip_or' in dict_tr.keys():
        foo_list.append(flip_or)

    #scaling intensity
    if 'scal_int1' in dict_tr.keys():
        foo_list.append(scal_int1)

    if 'scal_int2' in dict_tr.keys():
        foo_list.append(scal_int2)

    #pincushin barrel
    if 'optic_pin' in dict_tr.keys():
        param = dict_tr['optic_pin'][1]
        opt_pin = optic_pin(param['pin_fact'])
        foo_list.append(opt_pin.pincushion_transf)
    
    if 'optic_bar' in dict_tr.keys():
        param = dict_tr['optic_bar'][1]
        opt_bar = optic_bar(param['bar_fact'])    
        foo_list.append(opt_bar.barrel_transf)

    #elastic transf
    if 'elastic' in dict_tr.keys():
        param = dict_tr['elastic'][1]
        el = elastic_deform(param['alpha'],param['sigma'],param['alpha_affine'],param['iteration'])
        foo_list.append(el.elastic_transf)

    return foo_list
    
