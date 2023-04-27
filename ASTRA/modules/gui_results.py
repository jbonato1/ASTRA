import matplotlib.pyplot as plt 
from ipywidgets import Button, Layout
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display,clear_output,Video
from copy import copy
import numpy.ma as ma
import matplotlib.colors as colors
import imageio
import numpy as np
import cv2

def draw_cont(mask,numCC):
    N,M,c = mask.shape
    
    sm_c = np.zeros((N,M))
    pr_c = np.zeros((N,M))
    CC_c = np.zeros((N,M))
    
    if numCC is None:
        numCC = 0
        
    print('check',numCC,c-1)
    for cnt in range(c-1):
        
        if cnt==c-2:
            
            contours, hierarchy = cv2.findContours(np.uint8(mask[:,:,cnt]), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for j in contours:
                
                sm_c[j[:,0,1],j[:,0,0]]=1
        elif cnt>=numCC:
            print('proc',cnt)
            contours, hierarchy = cv2.findContours(np.uint8(mask[:,:,cnt]), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for j in contours:
                pr_c[j[:,0,1],j[:,0,0]]=1
        elif cnt<numCC:
            print(cnt)
            contours, hierarchy = cv2.findContours(np.uint8(mask[:,:,cnt]), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for j in contours:
                CC_c[j[:,0,1],j[:,0,0]]=1
                
    pr_c[pr_c>1]=1
    CC_c[CC_c>1]=1
    if numCC==0:
        return np.where(sm_c==1),np.where(pr_c==1),None
    else:
         return np.where(sm_c==1),np.where(pr_c==1),np.where(CC_c==1)

def draw_on_Im(image,c_sm,c_pr,c_CC):
    image_pl = image.copy()
    image_pl = image_pl.astype(np.float64)
    maxim = image_pl.max()
    image_pl/=maxim
    image_pl[c_sm]=10
    image_pl[c_pr]=4
    if not(c_CC is None):
        print('tests')
        image_pl[c_CC]=-1
    return image_pl


def get_im(mask,image,numCC=None):
    sm_net,pr_net,CC_net = draw_cont(mask,numCC=numCC)
    image_net = draw_on_Im(image,sm_net,pr_net,CC_net)
    return image_net



def gen_mask_sc(dict_,cell_num,im_shape):
    roi_list = []
    
    for key in dict_.keys():
        if 'CC_'+f'{str(cell_num):0>3}'+'_' in key:
            pt = dict_[key]
            buff = np.zeros(im_shape)
            buff[pt[0],pt[1]]=1
            roi_list.append(buff)
    print('CC num',len(roi_list))        
    for key in dict_.keys():
        if 'Proc_'+f'{str(cell_num):0>3}'+'_' in key:
            pt = dict_[key]
            buff = np.zeros(im_shape)
            buff[pt[0],pt[1]]=1
            roi_list.append(buff)
    print('ALL num',len(roi_list))
    pt = dict_['Soma_'+f'{str(cell_num):0>3}']
    buff = np.zeros(im_shape)
    buff[pt[0],pt[1]]=1
    roi_list.append(buff)
    
    buff = np.zeros(im_shape)
    roi_list.append(buff)
    
    out = roi_list[0]
    for i in range(1,len(roi_list)):  
        
        out = np.dstack((out,roi_list[i]))
    return out

def get_f0(traces,window):
    baseline = np.zeros_like(traces).astype(np.float32)
    for i in range(traces.shape[1]):
        if i<window:
            st=0
        else:
            st=i-window
        if i>traces.shape[1]-window:
            end = -1
        else:
            end=i+window

        baseline[:,i] = np.percentile(traces[:,st:end],20,axis=1)

    return baseline
    
def plot_traces(dict_,cell_num,window,ax):
    MAX_traces=7
    
    traces = []
    name=[]
    traces.append(dict_['Soma_'+f'{str(cell_num):0>3}'])
    name.append('Soma')
    cnt=1
    for key in dict_.keys():
        if 'Proc_'+f'{str(cell_num):0>3}'+'_' in key and cnt<= MAX_traces:
            traces.append(dict_[key])
            name.append('Proc.'+str(key[-3:]))
            cnt+=1
    traces = np.asarray(traces)
    #print(traces.shape)
    f0 = get_f0(traces,window)
    traces = (traces-f0)/(f0+0.0001)
    delta = 0
    for j in range(np.minimum(traces.shape[0],MAX_traces)):
        if np.max(traces[j,:])>100 or np.max(traces[j,:])==np.nan:
            traces[j,:]=0
            ax.plot(np.arange(traces.shape[1]),traces[j,:]+delta)
            ax.text(x = -traces.shape[1]/5, y= 0+delta, s =name[j] , rotation = 0,fontsize= 10)
            delta+= traces[j,:].max()+1
        else:
            ax.plot(np.arange(traces.shape[1]),traces[j,:]+delta)
            ax.text(x = -traces.shape[1]/5, y= 0+delta, s =name[j] , rotation = 0,fontsize= 10)
            delta+= traces[j,:].max()+0.4
        
def draw_roi_traces(im,Res_mat,dict_im,mapped,id_,cell_num,window):
    imC = get_im(Res_mat,im)
    f, [ax,ax1,ax2] = plt.subplots(1, 3, figsize=(24, 8))
    c = ['snow','lime','gainsboro']
    palette = copy(mapped.value)
    palette.set_over(c[0],1.0)
    palette.set_under(c[2],1.0)
    palette.set_bad(c[1],1.0)
    zm = ma.masked_where(imC==4,imC)
    ax.imshow(zm,cmap=palette,norm=colors.Normalize(vmin=0,vmax=1.0))
    ax.axis('off')

    rois = gen_mask_sc(dict_im['ROI_'+id_],cell_num.value,im.shape)
    if 'Num_CC_'+id_ in dict_im.keys():
        
        im_rois = get_im(rois,im,dict_im['Num_CC_'+id_]['Num_CC_'+f'{str(cell_num.value):0>3}'])
    else:
        im_rois = get_im(rois,im)
    c = ['snow','lime','red']
    
    palette = copy(mapped.value)
    palette.set_bad(c[1],1.0)
    palette.set_over(c[0],1.0)
    palette.set_under(c[2],1.0)
    

   
    zm2 = ma.masked_where(im_rois==4,im_rois)
    

    ax1.imshow(zm2,cmap=palette,norm=colors.Normalize(vmin=0,vmax=1.0))
    ax1.axis('off')

    plot_traces(dict_im['Signals_extr_'+id_],cell_num.value,window.value,ax2)
    ax2.yaxis.set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    ax2.set_xlabel('frames',fontsize=12)
    
    f.set_facecolor('white')
    return f

def gen_video(stack):
    video = np.empty((stack.shape[0],256,256),dtype=np.uint8)
    for j in range(stack.shape[0]):
        img = stack[j,:,:]
        bl = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        video[j,:,:]=bl

    imageio.mimwrite('test2.mp4',video, fps=30);
    
def layout(fov_name,dict_im):
    
    window = widgets.IntSlider(
    value=20,
    min=0,
    max=80,
    step=1,
    description='f0 window',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
    
    num= widgets.Dropdown(
        options=fov_name,
        value=fov_name[0],
        description='FOV',
        disabled=False,
    )

    cell_num= widgets.Dropdown(
        options=[i for i in range(dict_im['Cell_num_'+num.value])],
        value=0,
        description='cell',
        disabled=False,
    )

    mapped = widgets.Dropdown(
        options=[('inferno',plt.cm.inferno), ('viridis',plt.cm.viridis), ('Greys',plt.cm.gray)],
        value=plt.cm.gray,
        description='Cmap:',
        disabled=False,
    )
    projection =  widgets.Dropdown(
        options=['median', 'max', 'mean'],
        value='median',
        description='Projection:',
        disabled=False,
    )

    ROI_type =  widgets.Dropdown(
        options=[('All','_'), ('Fraction','_fraction_')],
        value='_',
        description='Roi Type',
        disabled=False,
    )


    button = widgets.Button(
        description='Plot',
    )
    out=widgets.Output(layout=Layout(width='1200px', height='400px'))
    out2=widgets.Output(layout=Layout(width='400px', height='400px',margin='0 0 0 100px'))
    #button=widgets.Button(description='Refresh')
    vbox1=widgets.VBox(children=(num,cell_num,mapped,projection,ROI_type,window,button),layout=Layout(margin='50px 0 0 0'))
    hbox=widgets.HBox(children=(vbox1,out2),layout=Layout(margin='50px 0 0 0'))
    vbox = widgets.VBox(children=(out,hbox))
    def display_plot(b=None):
        id_ = num.value    
        stack = dict_im['t-series_'+id_]
        T,N,M = stack.shape

        Res_mat = dict_im['Final_Mask'+ROI_type.value+id_]
        gen_video(stack)
        with out2:
            clear_output(wait=True)
            display(Video('test2.mp4',width=400,height=400))

        if projection.value=='median':
            im = np.median(stack,axis=0)
        elif projection.value == 'mean':
            im = np.mean(stack,axis=0)
        else:
            im = np.amax(stack,axis=0)
        print(cell_num.value)
        f = draw_roi_traces(im,Res_mat,dict_im,mapped,id_,cell_num,window)



        with out:
            clear_output(wait=True)
            display(f)
        
    return vbox,button,display_plot