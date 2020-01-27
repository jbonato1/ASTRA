import numpy as np
import os
from skimage import io
import sys
import h5py
import glob
from skimage import io
import matplotlib.pyplot as plt
import pickle
import cv2

#### MODULES used in the code
sys.path.insert(0,'/media/DATA/jbonato/astro_segm/AstroSS/modules/')
#from img_Sp_Sharp import create_img
from mask_roi_from_fiji import create_mask
from gen_single_astro import save_im, spatial_pp


N=512
M=512

#set_dir='/media/DATA/jbonato/astro_segm/set/'
set_dir='/media/DATA/jbonato/segm_project/set/'
folder_to_save = '/media/DATA/jbonato/astro_segm/set2/dataset/'
folder_single = '/media/DATA/jbonato/astro_segm/set2/train_single/'
mask_path_root = '/media/DATA/jbonato/astro_segm/zip_mask/consensus/D2/'

if not(os.path.exists(folder_to_save)):
    os.mkdir(folder_to_save)
    print('Created',folder_to_save)

if not(os.path.exists(folder_single)):
    os.mkdir(folder_single)
    print('Created',folder_single)


prefix='SMALL_'
folder_to_save+= prefix
folder_single+=prefix


dict_param = {
    'list':[0],
    'blocks':8,
    'threads':32,
    'BPM_ratio':8,
    'bb':256,
    'th1_p':0.25,
    'th2_p':0.10,
    'N_pix_st':250, #starting minimum area
    'astr_min':225, # approx. 0.9 min in dataset
    'percentile': 80,
    'pad':5,
    'astro_num':1, # number of astro min in FOV

    'init_th_':0.65, # threshold initialization approx. 120 frame
    'decr_dim':25, # astro area decrease
    'decr_th':25, # temporal threshold decrease
    'corr_int':False, # intensity correction flag
    'gpu_flag':True
}

remove_flag = True  # set true if you want to remove all
if remove_flag:
    for fl in glob.glob(folder_to_save+'*.hdf5'):
        os.remove(fl)
    for fl in glob.glob(folder_to_save+'*.tif'):
        os.remove(fl)
    for fl in glob.glob(folder_single+'*'):
        os.remove(fl)

items = os.listdir(set_dir)#collect folder

list_remotion = ['20'] #insert the folder name to remove them
for rem in list_remotion:
    items.remove(rem)
    
im_enh_l=[]
th1_p = dict_param['th1_p']
th2_p = dict_param['th2_p']
for item in items:
    dataset_dir = os.path.join(set_dir,item)
    try:
        int(item)
        flag_go=True
    except Exception as ex:
        flag_go=False
        pass   
    if os.path.isdir(dataset_dir) and flag_go and int(item)>25:
        
        
        if len(item)==1:
            zer='00'
        else:
            zer='0'

       
        os.chdir(dataset_dir)
        image_ids = glob.glob('*')
        
        ######################################Mask#################################################
        mask_path = mask_path_root+'RoiSet_SMALL_'+zer+item +'.zip'
        print('mask num: ',zer+item,' in ',mask_path)

        c_mask = create_mask(mask_path)###mask generation from .zip file generated using fiji
        instances_num = c_mask.get_dim()

        mask = np.empty((N, M, instances_num))
        soma_num, mask = c_mask.create_multiple_mask_im(im_dim = 512)#im_dim=None add this attribute 
        #if N,M are different from 256

        values_soma = np.zeros((N,M),dtype = np.float32 )

        values_soma = np.sum(mask[:,:,:soma_num],axis=2)
        values = np.sum(mask[:,:,soma_num:],axis=2)
        values[values>1]=1
        values = values.astype(np.float32) 
        values_soma = values_soma.astype(np.float32)
        
                                            #resize
        values = cv2.resize(np.uint8(values),(256,256),interpolation=cv2.INTER_AREA) 
        values_soma = cv2.resize(np.uint8(values_soma),(256,256),interpolation=cv2.INTER_AREA)
        ###########################################################################################
        
        for image_id in image_ids:
            path_to_file = os.path.join(dataset_dir,image_id)
            if not(os.path.isdir(path_to_file)):

                print('Processing: ','SMALL',item,' ',image_id)
                print(10*'*')

                file_path = os.path.join(dataset_dir, image_id)

                # Generate spatial sharpened map
                sp_pp = spatial_pp(file_path)
                stack,im_enh = sp_pp.create_img_d2() #see gen_single_astro.py for different spatial pp
                
                #appen to a list for visualization purposes
                im_enh_l.append(im_enh)
                
                #save unfiltered mask
                with h5py.File(folder_to_save +zer+item+'_nf.hdf5','w') as f:
                    dset = f.create_dataset('Values',data=values)
                    dset2 = f.create_dataset('Values_soma',data=values_soma)
                #SINGLE CELL CREATION   
                values,values_soma =  save_im(im=im_enh,stack=stack,mask_soma=values_soma,mask_proc=values,\
                                              save_folder = folder_single+zer+item,item=int(item),\
                                              BB_dim=192,th1_p=th1_p,th2_p=th2_p,pad=0)