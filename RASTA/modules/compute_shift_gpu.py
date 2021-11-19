import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
    
def register_translation_gpu(ref_image,stack,upsample_factor=1):
    
    if upsample_factor>1:
        raise ValueError("Upsampling not implemented yet")
    
    shape = ref_image.shape
    ref_image_gpu = cp.asarray(ref_image) 
    stack_gpu = cp.asarray(stack) 
    shifts_out = np.empty((stack.shape[0],2))
    for i in range(stack.shape[0]):

        image_freq = cufft.fft2(ref_image_gpu)
        Nimage_freq = cufft.fft2(stack_gpu[i,:,:])
        
        image_product = image_freq * Nimage_freq.conj()
        image_product /= cp.maximum(cp.absolute(image_product), 100 * 2.220446049250313e-16)

        cross_correlation = cufft.ifft2(image_product)
        cross_correlation = cp.asnumpy(cross_correlation)

        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),cross_correlation.shape)
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
        float_dtype = image_product.real.dtype
        shifts = np.stack(maxima).astype(float_dtype, copy=False)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]
        shifts_out[i,:] = shifts
    
    return shifts_out 