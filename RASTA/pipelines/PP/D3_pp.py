import os
from skimage import io
import sys
import h5py
import glob
from skimage import io
import matplotlib.pyplot as plt
import pickle
import numpy as np

#### MODULES used in the code
sys.path.insert(0,'/media/DATA/jbonato/astro_segm/AstroSS/modules/')
#from img_Sp_Sharp import create_img
from mask_roi_from_fiji import create_mask
from gen_single_astro import save_im_l, spatial_pp

N = 435
M = 435
#set_dir='/media/DATA/jbonato/astro_segm/set/'
set_dir='/media/DATA/jbonato/segm_project/set_large/'
folder_to_save = '/media/DATA/jbonato/astro_segm/set3/dataset/'
folder_single = '/media/DATA/jbonato/astro_segm/set3/train_single/'
mask_path_root = '/media/DATA/jbonato/astro_segm/zip_mask/consensus/D3/'

if not(os.path.exists(folder_to_save)):
    os.mkdir(folder_to_save)
    print('Created',folder_to_save)

if not(os.path.exists(folder_single)):
    os.mkdir(folder_single)
    print('Created',folder_single)


prefix='LARGE_'
folder_to_save+= prefix
folder_single+=prefix


dict_param = {
    'list':[i for i in range(0,390,30)],
    'blocks':15,
    'threads':20,
    'BPM_ratio':2,
    'bb':40,
    'th1_p':0.25,
    'th2_p':None,
    'N_pix_st':25, #starting minimum area
    'astr_min':22, 
    'percentile': 90,
    'pad':5,
    'astro_num':95, # number of astro min in FOV
    'init_th':0.5, # threshold initialization approx. 125
    'decr_dim':3, # astro area decrease
    'decr_th':12, # temporal threshold decrease
    'corr_int':True, # intensity correction flag
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
    if os.path.isdir(dataset_dir) and flag_go :
        
        
        if len(item)==1:
            zer='00'
        else:
            zer='0'

       
        os.chdir(dataset_dir)
        image_ids = glob.glob('*')
        
        image_ids = [kk for kk in image_ids if not ('enh' in kk) ]
        ######################################Mask#################################################
        mask_path = mask_path_root+'RoiSet_LARGE_'+zer+item +'.zip'
        print('mask num: ',zer+item,' in ',mask_path)

        c_mask = create_mask(mask_path)###mask generation from .zip file generated using fiji
        instances_num = c_mask.get_dim()

        mask = np.empty((N, M, instances_num))
        soma_num, mask = c_mask.create_multiple_mask_im(im_dim=435)#im_dim=None add this attribute 
        #if N,M are different from 256
        
        values_soma = np.zeros((435,435),dtype = np.float32 )
        values_soma = np.sum(mask,axis=2)
        values_soma = values_soma.astype(np.float32)
        values_soma = values_soma[2:-3,2:-3]

        ###########################################################################################
        
        for image_id in image_ids:
            path_to_file = os.path.join(dataset_dir,image_id)
            if not(os.path.isdir(path_to_file)):

                print('Processing: ','SMALL',item,' ',image_id)
                print(10*'*','cane')

                file_path = os.path.join(dataset_dir, image_id)

                # Generate spatial sharpened map
                print(file_path)
                sp_pp = spatial_pp(file_path)
                stack,im_enh = sp_pp.create_img_large() #see gen_single_astro.py for different spatial pp
                
                #appen to a list for visualization purposes
                im_enh_l.append(im_enh)
                
                #save unfiltered mask
                with h5py.File(folder_to_save +zer+item+'_nf.hdf5','w') as f:
                    dset2 = f.create_dataset('Values_soma',data=values_soma)
                #SINGLE CELL CREATION   
                values_soma =  save_im_l(im=im_enh,stack=stack,mask_soma=values_soma,\
                                              save_folder = folder_single+zer+item,item=int(item),\
                                              BB_dim=40,th1_p=th1_p,th2_p=th2_p,pad=4)