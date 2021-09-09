import numpy as np
import cv2
from readroi_pkg import *

import os


def export_roi(dict_im,fov_list,N=256,M=256,folder_save = '/media/DATA/jbonato/astro_segm/notebook/'):
    

    for fov in fov_list:
        folder = f'{str(fov):0>3}'
        if os.path.isfile(folder_save+'ROI_'+folder+'_MC.zip'):
            print('Attention: File ',folder_save+'ROI_'+folder+'_MC.zip',' already present')
        else:
            print('FOV',folder)
            list_roi = []

            for key in dict_im['ROI_'+folder]:
                buff = np.zeros((N,M))
                a,b = dict_im['ROI_'+folder][key]
                buff[a,b]=255

                _,thresh = cv2.threshold(np.uint8(buff),127,255,0)
                # find contours in the binary image
                contours, _= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                list_roi.append(ImagejRoi.frompoints(contours[0][:,0,:],name=key))

            roiwrite(folder_save+'ROI_'+folder+'_MC.zip', list_roi)

        

def read_roi_curated(fov,N=256,M=256,folder_read='/media/DATA/jbonato/astro_segm/notebook/'):
    
    rois_instances = roiread(folder_read+'ROI_'+fov+'_MC.zip')
    cnt_soma = 0
    for roi in rois_instances:
        if 'Soma' in roi.name:
            cnt_soma+=1

    mask_out = np.zeros((cnt_soma,N,M,3))
    for roi in rois_instances:
        if 'Soma' in roi.name:
            num = int(roi.name[5:8])
            c = 1
            #print(roi.name,num)
        else:
            num = int(roi.name[5:8])
            c = 0 
            #print(roi.name,num)
        coord = roi.coordinates()
        coord= coord[:,[1,0]]
        buff = np.zeros((N,M))

        for x in range(N):
            for y in range(M):
                res = cv2.pointPolygonTest(coord, (x,y), False)
                if res>=0:
                    mask_out[num,x,y,c]=1
            
    mask_out[mask_out>1]=1
    return mask_out
        

def clean_dict(dict_im,fov_list):
    for fov in fov_list:
        fov_num = f'{str(fov):0>3}'
        del dict_im['Single_cell_mask_'+fov_num] 
        del dict_im['Cell_num_'+fov_num] 
        del dict_im['Signals_extr_'+fov_num] 
        del dict_im['ROI_'+fov_num] 
        del dict_im['crop_coord_ROI_'+fov_num] 
        del dict_im['shift_ROI_'+fov_num] 
        del dict_im['Final_Mask_fraction_'+fov_num] 
    return dict_im