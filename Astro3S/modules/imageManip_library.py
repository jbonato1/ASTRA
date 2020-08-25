
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import skimage.io as io
from skimage.measure import points_in_poly as isInROI
import math
#import itertools
#import pandas as pd
#from tifffile import imsave
# In[2]:


def imageReader (readFiles):
    '''
    input list of file addresses
    output list of lists, each object of the list contains an image read as np.array by CV2.imread as unsigned 8bit grayscale 
    function reads a list of files
    Created by Zebastiano
    '''
    img = []
    for i in range(0,len(readFiles)):
        buffer = cv2.imread((myPath+readFiles[i]), cv2.IMREAD_UNCHANGED)
        img.append(buffer) 
    return img


# In[3]:


#basic stack projection functions, as in ImageJ
# they all are designed to work after loading the stack with imageReader in the format t(z),x,y Numpy Array

def stackAverage(stack):
    ''' 
    Generates the average projection of an image stack of arbitrary frame size. It uses np.arrays
    
    Input: 
    stack: a np.array with dimesions t(or z), x, y
    Output:
    avgProj: a np.array with the average projection dimensions(x,y)
    Created by Zebastiano
    '''
    t, x , y = np.shape(stack)
    avgProj = np.zeros_like(stack[0])
    for i in range(0,x):
        for j in range(0,y):
            avgProj[i,j] = np.mean(stack[:,i,j])
    return avgProj

def stackMax(stack):
    ''' 
    Generates the max projection of an image stack of arbitrary frame size. It uses np.arrays
    
    Input: 
    stack: a np.array with dimesions t(or z), x, y
    Output:
    maxProj: a np.array with the max projection dimensions(x,y)
    Created by Zebastiano
    '''
    t, x , y = np.shape(stack)
    maxProj = np.zeros_like(stack[0])
    for i in range(0,x):
        for j in range(0,y):
            maxProj[i,j] = np.amax(stack[:,i,j])
    return maxProj

def stackMin(stack):
    ''' 
    Generates the min projection of an image stack of arbitrary frame size. It uses np.arrays
    
    Input: 
    stack: a np.array with dimesions t(or z), x, y
    Output:
    minProj: a np.array with the min projection dimensions(x,y)
    Created by Zebastiano
    '''
    t, x , y = np.shape(stack)
    minProj = np.zeros_like(stack[0])
    for i in range(0,x):
        for j in range(0,y):
            minProj[i,j] = np.amin(stack[:,i,j])
    return minProj

def stackSD(stack):
    ''' 
    Generates the standard deviation projection of an image stack of arbitrary frame size. It uses np.arrays
    
    Input: 
    stack: a np.array with dimesions t(or z), x, y
    Output:
    stdProj: a np.array with the standard deviation projection dimensions(x,y)
    Created by Zebastiano
    '''
    t, x , y = np.shape(stack)
    stdProj = np.zeros_like(stack[0])
    for i in range(0,x):
        for j in range(0,y):
            stdProj[i,j] = np.std(stack[:,i,j])
    return stdProj

def stackMedian(stack):
    ''' 
    Generates the median projection of an image stack of arbitrary frame size. It uses np.arrays
    
    Input: 
    stack: a np.array with dimesions t(or z), x, y
    Output:
    medProj: a np.array with the median projection dimensions(x,y)
    Created by Zebastiano
    '''
    t, x , y = np.shape(stack)
    medProj = np.zeros_like(stack[0])
    for i in range(0,x):
        for j in range(0,y):
            medProj[i,j] = np.median(stack[:,i,j])
    return medProj


# In[4]:


def map_uint12_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image containing 12 bit dynamic range trough a lookup table to convert it to 8-bit.
    To preserve the full dynamic range it's necessary to provide lower_bound=0, upper_bound=4095
    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 4095]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 4095]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    Created by Zebastiano
    '''
    if not(0 <= lower_bound < 2**12) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**12) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = 0
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**12 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


# Let's generate an example image (normally you would load the 16-bit image: cv2.imread(filename, cv2.IMREAD_UNCHANGED))
#img = (np.random.random((100, 100)) * 2**16).astype(np.uint16)

# Convert it to 8-bit
#map_uint16_to_uint8(img)


# In[5]:


def readAndOutpaths2 (myPath):
    '''
    this function prepares the input parameters for roiContourDetector
    
    input [string]: the path to the t-series folder
    output [list]: is composed of 5 lists 
    First element is a list of tiff files composing the t-series
    Second element is the list of file addresses to save dynamic ROIs t-series to be fed to cv2.imwrite
    Third element is the list of file addresses to save original images and identified ROIs overlap t-series 
            to be fed to cv2.imwrite
    Fourth element is the name of a file address to save identified ROIs properties
    Fifth element is a list containing the addresses of the two video outputs
    Created by Zebastiano
    '''
    outBinaryROIs = []
    outTracked = []
    outCoordsFiles = []
    outVideoFiles = []
        
    binaryTagString = "binaryROIs/BinaryROI_"
    trackTagString = "track/trackedROI_"
    outCoordsDirectoryString = "coord_and_max/tracked_coords_"
    outVideoDirectoryString = "videos/"
    
    
    readFiles = [f for f in os.listdir(myPath) if os.path.isfile(os.path.join(myPath, f))]
    
    buffer, ext = os.path.splitext(os.path.abspath(myPath+outCoordsDirectoryString+readFiles[0]))
    outCoordsFiles.append(buffer+".txt")
    buffer, ext = os.path.splitext(os.path.abspath(myPath+outVideoDirectoryString+readFiles[0]))
    outVideoFiles.append(buffer+"_ROIs.avi")
    outVideoFiles.append(buffer+"_Tracked.avi")
        
    for index in range(0, len(readFiles)):
        outBinaryROIs.append((os.path.abspath(myPath+binaryTagString+readFiles[index])))
        outTracked.append((os.path.abspath(myPath+trackTagString+readFiles[index])))
        readFiles[index] = os.path.abspath(myPath+readFiles[index])
        
    return (readFiles, outBinaryROIs, outTracked, outCoordsFiles, outVideoFiles)


# In[6]:


def roiContourDetector (tSeriesFiles, binaryROIsFile, trackedFile, coordsMotionTXT, videoOut,
                        referenceFrame, referenceFramesNumber, decisionThreshold):
    
    '''
    Performs automatic detection of dynamic pixels in a t series.
    It's based on the GMG algorithm implemented in OpenCV 3.3.0
    (see https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html)
    
    input -->
    
    '''
    #initialization of a foreground background extraction mask using GMG algorythm from OpenCV background segmentation
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames= referenceFramesNumber, decisionThreshold = decisionThreshold)
    #declaration of the spatial kernel used by cv2.morphologyEx to refine the dyamic pixels  
    #kernel = np.ones((3,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    #check the total number of frames to be analyzed
    totalFrames = len(tSeriesFiles)
    #extraction of tseries sizing prameters, obtained parsing a sample image
    buffered = cv2.imread((tSeriesFiles[0]), cv2.IMREAD_UNCHANGED)
    outTrackHeight, outTrackWidth = np.shape(buffered)
        
    # Create VideoWriter objects, and its codec, to save analysis results
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    roisBinary = cv2.VideoWriter(videoOut[0], fourcc, 6.0, (outTrackWidth,outTrackHeight))
    roisOverlay = cv2.VideoWriter(videoOut[1], fourcc, 6.0, (outTrackWidth,outTrackHeight))
    
    #initialization of the buffering numpy structures
    fgmaskImage = np.zeros((outTrackHeight, outTrackWidth,3),dtype = np.uint8)
    frame = np.zeros((outTrackHeight, outTrackWidth,3),dtype = np.uint8)
    zeros = np.zeros((outTrackHeight, outTrackWidth),dtype = np.uint8)
    dynPix = np.zeros((totalFrames, outTrackHeight, outTrackWidth),dtype = np.uint8)
    analyzedFrames = 1
    
    # reference image generation
    #bufferAdd, ext = os.path.splitext(os.path.abspath(coordsMotionTXT[0]))
    #bufferAdd = (bufferAdd+".tiff")
    
    #cv2.imwrite(bufferAdd, referenceFrame)
        
    # enter main analysis loop
    while (analyzedFrames<(totalFrames+referenceFramesNumber)):
        analyzedFrames = analyzedFrames
    
        if analyzedFrames <= referenceFramesNumber:
            buffered = cv2.imread(os.path.abspath(referenceFrame), 
                                  cv2.IMREAD_UNCHANGED)
    
        elif analyzedFrames > referenceFramesNumber:
            buffered = cv2.imread((tSeriesFiles[analyzedFrames-referenceFramesNumber]), cv2.IMREAD_UNCHANGED)
    
    
        buffered = map_uint12_to_uint8(buffered, lower_bound=0, upper_bound=4095) #convert buffered image to 8 bit
        
        # build a three channels image 
        frame[:,:,0] = zeros
        frame[:,:,2] = zeros
        frame[:,:,1] = buffered
        
        rawFrame = frame
    
        fgmask = fgbg.apply(frame) # do the background estimation trick...
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) # refine using the spatial kernel defined before       
        image, cnts, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            contourId = 0
        # retrieve size filtered contours and draw it on the frame,
            if cv2.contourArea(c) > 5 and cv2.contourArea(c) < 200:
                cv2.drawContours(frame, cnts, contourId, (0,255,0), -1)
                area = float(cv2.contourArea(c))
                    #trackingCoords.append([x, y, w, h, analyzedFrames, area]) 
                    
            #if the contour is too small, ignore it
            else: 
                #cv2.drawContours(frame, cnts, -1, (0,255,0), 1)
                continue
            contourId = contourId+1

    # populate a three channel image with the binaryzed data coming from the foreground mask
    #fgmaskImage = np.empty_like(frame)
        fgmaskImage[:,:,0] = zeros
        fgmaskImage[:,:,1] = zeros
        fgmaskImage[:,:,2] = fgmask
        
        
    # binarized pixels stack
        if analyzedFrames > referenceFramesNumber:
            dynPix[analyzedFrames-referenceFramesNumber,:,:] = fgmask
            cv2.imwrite(binaryROIsFile[analyzedFrames-referenceFramesNumber], fgmaskImage)
            cv2.imwrite(trackedFile[analyzedFrames-referenceFramesNumber], frame)
    
    #openCV live display of the analysis
        cv2.imshow('raw t-series', rawFrame)
        cv2.imshow('dynamic signal highlight', frame)
        cv2.imshow('foreground-background mask', fgmaskImage)
        
        
        
        roisBinary.write(fgmaskImage)
        roisOverlay.write(frame)
            
        analyzedFrames = analyzedFrames + 1
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
            
    #release all CV2 video instances and kill windows
    roisBinary.release()
    roisOverlay.release()
    #video.release()
    cv2.destroyAllWindows()
    print(cnts)
    return (dynPix)

	
def stackReader2 (readFiles):
    ''' Reads image sequence and returns np 3D array (t(or z),x,y)
        
        Input = list of full paths to files
        
        Output = np array (t(or z),x,y)
        
        Note: Relies on cv2.imread() method to import single images and then appends it to a list
        which then is converted to np.array
        
        Created by Zebastiano
    '''
    img = []
    for i in range(0,len(readFiles)):
        buffer = cv2.imread((readFiles[i]), cv2.IMREAD_UNCHANGED)
        img.append(buffer)
    img = np.array(img)
    return img

def multiLayerStackReader(fName):
    ''' Reads multilayer tiff files and returns a np 3D array (t(or z),x,y)
        Input = full path to the multilayer tiff file
        Output = np array (t(or z),x,y)
        
        Note: Relies on skimage.io.imread() method to import the multilayer tiff. 
        --> skimage.io.MultiImage method returns an object with 2 fields (alpha in the code) 
        - field alpha[0] contains the images, which are then exported as np.array (stack in code)
        - field aplha[1] contains the full path to the file (indicated as "name")
        Created by: Zebastiano
    '''
    alpha = io.MultiImage(fName)
    stack = np.array(alpha[0])
    return(stack)

def multiLayerStackWriter(stack, fName):
    ''' Writes a multilayer tiff file from returns a np 3D array (t(or z),x,y)
        Input: 
        stack = np 3D array (t(or z), x, y)
        fName = full path to the multilayer tiff file
        
        Output
        True if saved correctly 
        
        Note: Relies on tifffile package. 
        --> skimage.io.MultiImage method returns an object with 2 fields (alpha in the code) 
        
        Created by: Zebastiano
    '''
    imsave(fName, stack, shape = stack.shape, dtype = stack.dtype)
    return(True)


# Copyright: Luis Pedro Coelho <luis@luispedro.org>, 2012
# License: MIT
# Updated and adapted for Python 3.x
# by Zebastiano

def read_roi(fileobj):
    '''
    points = read_roi(fileobj)
    Read ImageJ's ROI format
    
    Remember the format of ROI import is a np.ndArray with:
    2 columns, the first one are Y the second one are X
    
    # This is based on:
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
    # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html
    '''



    SPLINE_FIT = 1
    DOUBLE_HEADED = 2
    OUTLINE = 4
    OVERLAY_LABELS = 8
    OVERLAY_NAMES = 16
    OVERLAY_BACKGROUNDS = 32
    OVERLAY_BOLD = 64
    SUB_PIXEL_RESOLUTION = 128
    DRAW_OFFSET = 256


    pos = [4]
    def get8():
        pos[0] += 1
        s = fileobj.read(1)
        if not s:
            raise IOError('readroi: Unexpected EOF')
        return ord(s)

    def get16():
        b0 = get8()
        b1 = get8()
        return (b0 << 8) | b1

    def get32():
        s0 = get16()
        s1 = get16()
        return (s0 << 16) | s1

    def getfloat():
        v = np.int32(get32())
        return v.view(np.float32)

    magic = fileobj.read(4)
    magic = magic.decode("utf-8")
    #print (magic)
    if magic != 'Iout':
        raise IOError('Magic number not found')
    version = get16()

    # It seems that the roi type field occupies 2 Bytes, but only one is used
    roi_type = get8()
    # Discard second Byte:
    get8()

    if not (0 <= roi_type < 11):
        raise ValueError('roireader: ROI type %s not supported' % roi_type)

    if roi_type != 7:
        raise ValueError('roireader: ROI type %s not supported (!= 7)' % roi_type)

    top = get16()
    left = get16()
    bottom = get16()
    right = get16()
    n_coordinates = get16()

    x1 = getfloat() 
    y1 = getfloat() 
    x2 = getfloat() 
    y2 = getfloat()
    stroke_width = get16()
    shape_roi_size = get32()
    stroke_color = get32()
    fill_color = get32()
    subtype = get16()
    if subtype != 0:
        raise ValueError('roireader: ROI subtype %s not supported (!= 0)' % subtype)
    options = get16()
    arrow_style = get8()
    arrow_head_size = get8()
    rect_arc_size = get16()
    position = get32()
    header2offset = get32()

    if options & SUB_PIXEL_RESOLUTION:
        getc = getfloat
        points = np.empty((n_coordinates, 2), dtype=np.float32)
    else:
        getc = get16
        points = np.empty((n_coordinates, 2), dtype=np.int16)
    points[:,1] = [getc() for i in range(n_coordinates)]#[getc() for i in xrange(n_coordinates)]
    points[:,0] = [getc() for i in range(n_coordinates)]#[getc() for i in xrange(n_coordinates)]
    points[:,1] += left
    points[:,0] += top
    
    return points

def read_roi_zip(fname):
    '''
    This sub-routine uses zipfile to run multiple iterations of  read_roi() 
    Remember the format of ROI import is a np.ndArray with:
    2 columns, the first one are Y the second one are X
    
    NOTE: for most use this function will require to convert the output list to np.array:
    i.e.
    x = np.array(read_roi_zip(fname))
    '''
    
    import zipfile
    with zipfile.ZipFile(fname) as zf:
        return [read_roi(zf.open(n))
                    for n in zf.namelist()]

def read_roi_zip2(fname):
    '''
    This sub-routine uses zipfile to run multiple iterations of  read_roi() 
    Remember the format of ROI import is a np.ndArray with:
    2 columns, the first one are Y the second one are X
    
    NOTE: for most use this function will require to convert the output list to np.array:
    i.e.
    x = np.array(read_roi_zip(fname))
    '''
    
    import zipfile
    soma_list=[]
    proc_list=[]
    counter=0
    with zipfile.ZipFile(fname) as zf:
        for n in zf.namelist():
            if n[:5]=='Soma_' or n[:5]=='SOMA_'or n[:4]=='SOMA':
                soma_list.append(read_roi(zf.open(n)))
                counter+=1
            else:

                proc_list.append(read_roi(zf.open(n)))
    return counter,soma_list+proc_list

def ROITrace(roi, tSeries, fps, verbose = 0):
    ''' Computes the frame-wise average intensity of a ROI
        Inputs: 
        roi is a contour coming from imagej routine read using read_roi function
        tSeries is a stack (t,x,y)
        fps is the acquisition frame rate
        Verbose (default=0) whether to show the image of the Average projection side by side with the ROI intensity profile
        
        Outputs:
        trace --> a np.array containing the framewise average intensity profile.
        # This is based on:
        # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
        # http://rsbweb.nih.gov/ij/developer/source/ij/io/RoiEncoder.java.html
    '''
    
    frames, xDim, yDim = tSeries.shape
    trace = np.zeros(frames)
    # isInROI(points, verts)
    # points is the grid of confrontation 
    # verts are the vertices of the ROI
    # output is a mask
    points = np.zeros((xDim*yDim,2))
    count = 0
    for i in np.arange(0,xDim):
        for k in np.arange(0,yDim):
            points[count] = (i,k)
            count = count+1
    mask = isInROI(points, roi)
    
    
    for i in np.arange(0,frames):    
        #tempROI = tSeries[i][roi[:,1] , roi[:,0]]
        tempROI = tSeries[i][mask.reshape((xDim,yDim))]
        trace[i] = tempROI.mean()

    if verbose == 1:
        time = (np.arange(0, frames))/fps
        plt.figure(figsize=(10,10))
        maskForROI = np.zeros((xDim, yDim))
        maskForROI[roi[:,0] , roi[:,1]] = 1
        masked = np.ma.masked_where(maskForROI == 0, maskForROI)
        avg = zImage.stackAverage(tSeries)
        
        fig = plt.figure(figsize=(10,10))
        plt.rc('font', size = 8, family = 'Arial')
        gridShape = (3,6) 
        ax1 = plt.subplot2grid(shape = gridShape, loc = (0, 0), rowspan = 3, colspan = 3, )
        ax2 = plt.subplot2grid(shape = gridShape, loc = (1, 3), rowspan = 1, colspan = 3)
        
        ax1.imshow(avg, cmap = plt.cm.inferno, interpolation='none')
        ax1.imshow(masked, cmap = plt.cm.Oranges, interpolation='none', alpha=1)
        ax2.plot(time, trace, 'k', lw = .5,)
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("intensity [16bit]")
        plt.tight_layout(w_pad=0.5)
        plt.show()
        
    return(trace)


def pixwiseRunningBSL(activityTrace, FPS = 3, smoothingWindow = 30, percentile = 20):
    '''
       Input: pixel-wise fluorescence intensity trace as a np.array (or slice of it) 
       Output: t-series of (F-f0)/fo and f0
       As default f0 is computed as the n-th percentile (default 20) of fluorescence intensity along a user defined time window (in
       seconds;default is 30). 
       
       This function computes the n-th percentile using the NEXT *smoothingWindow* seconds.
       
       To allow for f0 normalization if f0 goes to 0 it's substituted by its actual value in the input trace.
       Outputs:
       smoothedTrace and bslTrace which are both np.arrays
       Created by Zebastiano
       '''
    frameNumber = activityTrace.shape[0]
    
    bslTrace = np.zeros(frameNumber)
    smoothedTrace = np.zeros(frameNumber)
    
    window = round(smoothingWindow*FPS)
    
    for i in np.arange(0,frameNumber-window):
        bslTrace[i]=(np.percentile(a = activityTrace[i:(i+window)], q = percentile, interpolation = 'linear'))
    for i in np.arange(frameNumber-window,frameNumber):
        bslTrace[i]=(np.percentile(a = activityTrace[i:frameNumber], q = percentile, interpolation = 'linear'))
    
    zeros = np.where(bslTrace == 0)
    bslTrace[zeros] = 1
    smoothedTrace = (activityTrace - bslTrace)/bslTrace
        
    return(smoothedTrace, bslTrace)


def centered_pixwiseRunningBSL(activityTrace, FPS = 3, smoothingWindow = 30, percentile = 20):
    '''
       Input: pixel-wise fluorescence intensity trace as a np.array (or slice of it) 
       Output: t-series of (F-f0)/fo and f0
       As default f0 is computed as the n-th percentile (default 20) of fluorescence intensity along a user defined time window (in
       seconds;default is 30). 
       
       This function computes the n-th percentile CENTERED on a *smoothingWindow* [s] wide window.
       
       To allow for f0 normalization if f0 goes to 0 it's substituted by its actual value in the input trace.
       Outputs:
       smoothedTrace and bslTrace which are both np.arrays
       Created by Zebastiano
       '''
    frameNumber = activityTrace.shape[0]
    
    bslTrace = np.zeros(frameNumber)
    smoothedTrace = np.zeros(frameNumber)
    
    window = round(smoothingWindow*FPS)
    window_half1 = int(math.trunc(window/2))
    window_half2 = int(math.ceil(window/2))
    
    if(window_half1+window_half2)==window:
        for i in np.arange(0 , window_half1):
            bslTrace[i]=(np.percentile(a = activityTrace[window_half1-(window_half1-i):(i+window_half1)], q = percentile, interpolation = 'linear'))
        for i in np.arange(window_half1,frameNumber-window_half2):
            bslTrace[i]=(np.percentile(a = activityTrace[(i-window_half1):(i+window_half2)], q = percentile, interpolation = 'linear'))
        for i in np.arange(frameNumber-window_half2,frameNumber):
            bslTrace[i]=(np.percentile(a = activityTrace[(i-window_half1):(frameNumber-(window_half2-i))], q = percentile, interpolation = 'linear'))
    
        zeros = np.where(bslTrace == 0)
        bslTrace[zeros] = 1
        smoothedTrace = (activityTrace - bslTrace)/bslTrace
    else:
        print('error')
    return(smoothedTrace, bslTrace)

def evs(trx, evsLimits):
    '''
    Input: 
    *trx* as np.array 
    event limits (*evsLimits*) as df.
    Returns: 
    *events_trace* a filtered trace containg just the significative events
    *events_list* a df containing all the events in the columns, indexes are frames

    Note:
    This function is cool because it uses itertools to append a list of lists of different lenght to df columns
    ''' 
    events_trace = np.zeros(trx.shape[0])

    events_columns = ['Event_'+str(i) for i in np.arange(0,evsLimits.shape[0])]
    evsDF = pd.DataFrame(columns = events_columns)

    events_list = []

    for i in evsLimits.index:
        start = evsLimits.at[i,'event_start']
        stop = evsLimits.at[i,'event_stop']
        tempTrace = trx[start:stop]
        events_trace[start:stop] = tempTrace
        tempTrace = tempTrace.tolist()
        events_list.append(tempTrace)

    eventDF = pd.DataFrame((_ for _ in itertools.zip_longest(*events_list)), columns=events_columns)
    return(events_trace, eventDF)

def eventFinder(trace = None, start_nSigma_bsl = 2, stop_nSigma_bsl = .5, FPS = 3, minimumDuration = .5, debugPlot = False):
    ''' This function gets a *trace* as input, computes the whole trace SD and filter the trace. 
    The filtered trace will be depleted of the biggest events, and considered like the baseline. A new SD (*bslSD*) is computed
    and start and stop thresholds will be computed as n times *bslSD* according to *start_nSigma_bsl* (default = 2), *stop_nSigma_bsl* (default = .5).
    
    Events (pos or neg) are defined as *trace* segments above startTreshold and stopTreshold and duration bigger than *minimumDuration* in [s].
    FPS has to be passed in order to compute the duration. In this section events are identified as transition points through the tresholds (limits).
    
    Then calling the ad hoc fucntion *evs* computes the *event_trace* and the *event_df* for both positive and negative events.
    
    Returns:
    
    pos_events_trace
    neg_events_trace
    positiveEventsDF_limits --> [in frames]
    negativeEventsDF_limits --> [in frames]
    pos_eventDF --> all positve events ordered according appearence
    neg_eventDF --> all negative events ordered according appearence
    
    Note:
    The ratio of (neg_eventDF/pos_eventDF) represent a measure of false discovery rate see Dombeck DA 2007.
    '''
    SD = trace.std()
    belowSD = np.where(trace<trace.std(), trace, np.nan)
    
    belowSD_nanRemoved = belowSD[~np.isnan(belowSD)]
    bslSD = belowSD_nanRemoved.std()
    
    startTreshold = start_nSigma_bsl*bslSD
    stopTreshold = stop_nSigma_bsl*bslSD
    
        
    pos_eventStart = []
    pos_eventStop = []
    neg_eventStart = []
    neg_eventStop = []
    #for i in np.arange(0,centered_smoothedTrace.shape[0]):
    i=0
    while i<trace.shape[0]-1:
            
        while trace[i]>=startTreshold and i<(trace.shape[0]-2):
            pos_eventStart.append(i)
            while trace[i]>=stopTreshold and i<(trace.shape[0]-2):
                i = i+1
            pos_eventStop.append(i)
            i = i+1
            
        while trace[i]<=-startTreshold and (i<trace.shape[0]-2):
            neg_eventStart.append(i)
            while trace[i]<=-stopTreshold and (i<trace.shape[0]-2):
                i = i+1
            neg_eventStop.append(i)
            i = i+1
        #if i<(trace.shape[0]-2):
        i = i+1
    
    if len(pos_eventStart) != 0:
        if len(pos_eventStart)/len(pos_eventStart) != 1:
            if len(pos_eventStart) > len(pos_eventStop):
                posToRemove = len(pos_eventStart) - len(pos_eventStop)
                pos_eventStart = pos_eventStart[:len(pos_eventStart)-posToRemove]
            if len(pos_eventStop) > len(pos_eventStart):
                posToRemove = len(pos_eventStop) - len(pos_eventStart)
                pos_eventStop = pos_eventStop[:len(pos_eventStop)-posToRemove]

    if len(neg_eventStop) != 0:
        if len(neg_eventStart)/len(neg_eventStop) != 1:
            if len(neg_eventStart) > len(neg_eventStop):
                posToRemove = len(neg_eventStart) - len(neg_eventStop)
                neg_eventStart = neg_eventStart[:len(neg_eventStart)-posToRemove]
            if len(neg_eventStop) > len(neg_eventStart):
                posToRemove = len(neg_eventStop) - len(neg_eventStart)
                neg_eventStop = neg_eventStop[:len(neg_eventStop)-posToRemove]

    data_pos = [pos_eventStart,pos_eventStop]
    positiveEventsDF = pd.DataFrame(data = data_pos, index=['event_start','event_stop'])
    positiveEventsDF = positiveEventsDF.transpose()
    positiveEventsDF['event_duration'] = (positiveEventsDF.event_stop - positiveEventsDF.event_start)/FPS 
    positiveEventsDF_limits = positiveEventsDF[positiveEventsDF.event_duration >= minimumDuration]
    positiveEventsDF_limits.reset_index(drop=True, inplace = True)
    
    data_neg = [neg_eventStart,neg_eventStop]
    negativeEventsDF = pd.DataFrame(data = data_neg, index=['event_start','event_stop'])
    negativeEventsDF = negativeEventsDF.transpose()
    negativeEventsDF['event_duration'] = (negativeEventsDF.event_stop - negativeEventsDF.event_start)/FPS 
    negativeEventsDF_limits = negativeEventsDF[negativeEventsDF.event_duration >= minimumDuration]
    negativeEventsDF_limits.reset_index(drop=True, inplace = True)
    
    
    
    
    pos_events_trace, pos_eventDF = evs(trx = trace, evsLimits = positiveEventsDF_limits)
    neg_events_trace, neg_eventDF = evs(trx = trace, evsLimits = negativeEventsDF_limits)
        
    if debugPlot == True:
        ax = plt.subplot(311)
        plt.plot(np.arange(trace.shape[0]), trace, 'gray')
        ax.axhline(y = SD, c = 'b')
        plt.plot(np.arange(belowSD.shape[0]), belowSD, 'b')
        ax.axhline(y = start_nSigma_bsl*bslSD, c = 'r', ls = ':', lw = .3)
        ax.axhline(y = -start_nSigma_bsl*bslSD, c = 'k', ls = ':', lw = .3)
        ax.axhline(y = stop_nSigma_bsl*bslSD, c = 'r',  ls = ':', lw = .3)
        ax.axhline(y = -stop_nSigma_bsl*bslSD, c = 'k',  ls = ':', lw = .3)

        ax2 = plt.subplot(312)
        plt.plot(np.arange(trace.shape[0]), trace, 'gray')
        plt.plot(pos_eventStart, trace[pos_eventStart], '.r')
        plt.plot(pos_eventStop, trace[pos_eventStop], '.k')

        plt.plot(neg_eventStart, trace[neg_eventStart], '.g')
        plt.plot(neg_eventStop, trace[neg_eventStop], '.y')
        
        ax3 = plt.subplot(313)
        plt.plot(np.arange(trace.shape[0]),trace, 'gray')
        plt.plot(np.arange(trace.shape[0]),pos_events_trace, c = 'r', alpha = .7)
        plt.plot(np.arange(trace.shape[0]),neg_events_trace, c = 'b', alpha = .7)
        plt.show()
    
    return(pos_events_trace, neg_events_trace, positiveEventsDF_limits, negativeEventsDF_limits, pos_eventDF, neg_eventDF)
