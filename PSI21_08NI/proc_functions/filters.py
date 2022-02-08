# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:40:28 2020

@author: boillat
"""


# ******************************************************************

#                           WHITE SPOT FILTER

# ******************************************************************

# ------------------------------------------------------------------
#                           ro_filter_2D
# ------------------------------------------------------------------
# This function removes white spots in the input image. 
# In src_img, the median is computed for each pixel accounting for
# neighbouring N pixels (ws_filter_size). The difference between 
# the pixel value and median is compared to the threshold (scaler or median proportion) 
# to identify the white spots (pos_ws). Then, the white spots are replaced with NaN or median.  

# input parameters: src_img = the input image to be processed
#                   ws_fiter_size = the size of neighbour for computing median (ODD NUMBER ONLY)
#                   threshold = user defined threshold value
#                   type_replace = 0 for a number threshold, 1 for proportion median
#                   type_replace = 0 for NaN replacemnt, 1 for median replacement
#                   **kwargs = collection of named parameters

# return value: the filtered image
# ------------------------------------------------------------------


def ro_filter_2D(src_img, filter_size, threshold, type_threshold='abs', 
                 type_replace='med', **kwargs):

    import numpy as np
    from scipy import signal
    
    src_img_float = np.asarray(src_img, dtype = np.float32)
    median_img = signal.medfilt2d(src_img_float, filter_size)
    difference_img = src_img_float - median_img
    
    filtered_img = np.copy(src_img_float)
    
    if type_threshold == 'abs': #number threshold
        pos_ws = np.where(abs(difference_img) > threshold)
    elif type_threshold == 'prop': #median proportion threshold
        pos_ws = np.where(abs(difference_img) > threshold*np.sqrt(median_img))
    else:
        raise ValueError('"type_threshold" can only be "abs" (absolute) or "prop" (proportional)')
        
    if type_replace == 'med':
        nan_replace = False
    elif type_replace == 'nan':
        nan_replace = True
    else:
        raise ValueError('"type_replace" can only be "med" (median) or "nan" (not a number)')

    for i in range(len(pos_ws[0])):
        if nan_replace:
            filtered_img[pos_ws[0][i],pos_ws[1][i]] = np.nan
        else:
            filtered_img[pos_ws[0][i],pos_ws[1][i]] = median_img[pos_ws[0][i],pos_ws[1][i]]
    
    filtered_img_origtype = np.asarray(filtered_img, dtype=src_img.dtype)
        
    return filtered_img_origtype


# ------------------------------------------------------------------
#                           ro_filter_3D
# ------------------------------------------------------------------


# ------------------------------------------------------------------


def ro_filter_3D(src_img, filter_size, threshold, type_threshold='abs',
                 type_replace='med', **kwargs):

    import numpy as np
    from processors import get_neighbors_stack
    
    from scipy.ndimage.filters import median_filter
    
    if type(filter_size) == list:
        szz = filter_size[2]
    else:
        szz = filter_size
    
    stk = get_neighbors_stack(szz, **kwargs)
    if stk.size == 0:
        return np.zeros([0,0])
    
    stk_med = median_filter(stk.astype(float), filter_size, mode='constant')
    
    src_img_float = np.asarray(src_img, dtype = np.float32)
    median_img = stk_med[:,:,szz//2]
    difference_img = src_img_float - median_img
    
    filtered_img = np.copy(src_img_float)
    
    if type_threshold == 'abs': #number threshold
        pos_ws = np.where(abs(difference_img) > threshold)
    elif type_threshold == 'prop': #median proportion threshold
        pos_ws = np.where(abs(difference_img) > threshold*np.sqrt(median_img))
    else:
        raise ValueError('"type_threshold" can only be "abs" (absolute) or "prop" (proportional)')
        
    if type_replace == 'med':
        nan_replace = False
    elif type_replace == 'nan':
        nan_replace = True
    else:
        raise ValueError('"type_replace" can only be "med" (median) or "nan" (not a number)')

    for i in range(len(pos_ws[0])):
        if nan_replace:
            filtered_img[pos_ws[0][i],pos_ws[1][i]] = np.nan
        else:
            filtered_img[pos_ws[0][i],pos_ws[1][i]] = median_img[pos_ws[0][i],pos_ws[1][i]]
    
    filtered_img_origtype = np.asarray(filtered_img, dtype=src_img.dtype)
        
    return filtered_img_origtype


# ------------------------------------------------------------------
#                           ro_auto_threshold
# ------------------------------------------------------------------


# ------------------------------------------------------------------


def ro_auto_threshold(test_imgf, test_roi=[0,0,0,0], test_filter_size=25, base_dir = ''):
    
    from img_utils import get_img, crop_img
    from parameters import param
    from scipy.ndimage.filters import median_filter
    from scipy.stats import median_absolute_deviation
    import scipy
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy import exp
    
    def gaussian_func(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))
    
        # if no base directory is specified, get it from the general processing parameters
    if base_dir == '' and param('base_dir') != None:
        base_dir = param('base_dir')
    
        # load the test image
    test_img = get_img(test_imgf, base_dir)
    
        # get an image of the average difference with the neigborhood (defined by 'test_filter_size')
    test_img_diff = crop_img(test_img - median_filter(test_img.astype(float), test_filter_size, mode='constant'), test_roi)
    
        # get the estimated standard deviation. The MAD (median absolute deviation)
        # is used as a robust estimator for the standard deviation (note that the
        # used "median_absolute_deviation" function from scipy already applies the
        # scaling factor of 1.4826 to match the standard deviation)
    eval_std = median_absolute_deviation(test_img_diff, axis=None)
    
        # the threshold is set as the value corresponding to a 99% confidence interval
    threshold = scipy.stats.norm.interval(0.99)[1]*eval_std
    
        # get the histogram
    h = np.histogram(test_img_diff, bins=50, range=[-2*threshold, 5*threshold])
    
        # fit the histogram with a gaussian function
    popt,_ = curve_fit(gaussian_func,h[1][:-1],h[0],p0=[max(h[0]),0,eval_std])
    x0 = popt[1]
    
        # display a histogram plot with the illustration of the selected threshold
    ymin = max(h[0])*1e-4
    ymax = max(h[0])*2
    plt.semilogy(h[1][:-1], h[0], 'b+', label='histogram')
    plt.semilogy(h[1][:-1], gaussian_func(h[1][:-1], *popt), 'r-', label='gaussian fit')
    plt.semilogy([x0, x0], [ymin, ymax], 'k-')
    plt.semilogy([x0+threshold, x0+threshold], [ymin, ymax], 'k--')
    plt.semilogy([x0-threshold, x0-threshold], [ymin, ymax], 'k--')
    plt.legend()
    plt.ylim(bottom=ymin, top=ymax)
    plt.xlabel('Pixel intensity [-]')
    plt.ylabel('Frequency [-]')
    plt.show()
    
        # report the estimated std and selected threshold
    print('Estimated noise standard deviation: ', eval_std)
    print('Selected threshold: ', threshold)
    
        # return the selected threshold
    return threshold
    
    
    
    