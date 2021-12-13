import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from sklearn.externals.joblib import Parallel, delayed
from skimage.restoration import denoise_nl_means,estimate_sigma
from scipy import signal
from joblib import Parallel, delayed
import os
import h5py
from numba import cuda,float32,uint16,float64  
import time
import pickle
from sel_active_reg_gen import sel_active_reg




with open('/media/DATA/jbonato/astro_segm/set1/.tmp/dict_dataset1.txt', "rb") as fp:   
    dict_param = pickle.load(fp)
    

# dict_param['blocks']=8
# dict_param['threads']=20
# dict_param['BPM_ratio']=4
# dict_param['N_pix_st']=50
# dict_param['astr_min']=50
# dict_param['th1_p']=.25
# dict_param['th2_p']=.1
# dict_param['max_min']=np.asarray([345,60])
# dict_param['astro_num']=40
# dict_param['bb']=80
# dict_param['pad']=8
# dict_param['list']=[i*54 for i in range(9)]
# dict_param['list'][-1]=432
# dict_param['decr_dim'] = 5
# dict_param['init_th'] = 0.5
# dict_param['decr_th'] = 7.333333333333333*25

# with open('/media/DATA/jbonato/astro_segm/set4/.tmp/dict_dataset.txt', "rb") as fp:   #Pickling
#     dict_param = pickle.load(fp)
# dict_param['init_th_'] = 0.5

# dict_param['BPM_ratio']=3
# dict_param['blocks']= 15
# dict_param['threads']=32
dict_param

print(dict_param)

stack = np.zeros((4000,256,256),dtype=np.float32)

a_reg = sel_active_reg(stack.astype(np.float32),dict_param)

a_reg.check_sel_active_reg_gpu_gen(void_out=True,debug=True)