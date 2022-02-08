# ******************************************************************

#                           IMG_UTILS

#           Collection of help functions to deal with images

# get_img - loads an image (use astropy for FITS images and matplotlib 
#       for other images)

# prep_img_for_display - scales the intensity and dimensions of an
#       image to prepare it for being displayed

# show_img - display an image using matplotlib

# crop_img - return a subregion from an image

# get_img - loads a stack of images (use astropy for FITS images and matplotlib 
#       for other images)

# ******************************************************************

# from file_utils import file_list

# ------------------------------------------------------------------
#                            get_img
# ------------------------------------------------------------------
                            
# loads an image and returns it as an numpy array     
# In case of a FITS image, the astropy package is used for reading
# (future implementation will return the FITS header as well)
# In case of a TIFF or other image, the general "imread" function
# from the 'matplotlib.pyplot' package is used                       

# Parameters:
#   file_name = file name including the path relative to the base directory
#   base_dir = path of the base directory (default is '', in which case
#       'file_name' should contain an absolute path)      
#
# Return value:
#   img = array containing the image data                      
                            
# ------------------------------------------------------------------

def get_img(file_name, base_dir='', squeeze=True):
    
    from astropy.io import fits
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    
        # get the file extension to identify the type
    _ , ext = os.path.splitext(file_name)
    
        # case of FITS files
    if ext == '.fits':
        
            # disable warnings because of possible bad characters in the header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
                # open the file and get the image data
            with fits.open(os.path.join(base_dir, file_name)) as hdul:
                img = hdul[0].data
        
                # reverse the y coordinates (for consistency with ImageJ)
            img = np.flip(img, 0)
    
        # case of TIFF files
    elif ext == '.tif':
        
            # get the image
        img = plt.imread(os.path.join(base_dir, file_name))
        
        # general case
    else:
        
            # get the image
        img = plt.imread(os.path.join(base_dir, file_name))
    
        # if the 'squeeze' option is on, remove dimensions with size 1
    if squeeze:
        img = np.squeeze(img)
    
        # return the data
    return(img)

# ------------------------------------------------------------------
#                            write_img
# ------------------------------------------------------------------
                            
# Writes a FITS image. If the destination path does not exist, create it                 

# Parameters:
#   img = image to write (2D numpy array)
#   file_name = file name including the path relative to the base directory
#   base_dir = path of the base directory (default is '', in which case
#       'file_name' should contain an absolute path)      
#
# Return value:
#   img = array containing the image data                      
                            
# ------------------------------------------------------------------

def write_img(img, file_name, base_dir='', overwrite=False):
    
    import os
    from astropy.io import fits
    
        # compute the destination file name
    dst_name = os.path.join(base_dir, file_name)
    
        # get the destination directory
    dst_name_dir, _ = os.path.split(dst_name)
    
        # and create this destination directory if it does not exist
    if not os.path.exists(dst_name_dir):
        os.makedirs(os.path.join(dst_name_dir,''))
    
            # save the processed image
    try:                        
        hdu = fits.PrimaryHDU(img)
        hdu.writeto(dst_name, overwrite=overwrite)
    
        # output the error message if necessary
    except Exception as detail:
        print("Could not write the destination image:", detail) 
    
# ------------------------------------------------------------------
#                            prep_img_for_display
# ------------------------------------------------------------------
                            
# Scale and image intensity to display it based on percentile values.
# The data in the range between the lower and higher percentiles is mapped
# to an output data between 0.0 and 1.0. Additionally, the size of the
# image can be channged by applying a zoom factor.                    

# Parameters:
#   img = array containing the image data
#   zf = zoom factor (default 1.0)                        
#   lp = lower percentile (default 5%)                      
#   lp = higher percentile (default 95%)  
#
# Return value:
#   disp_img = array containing the rescaled data                          
# ------------------------------------------------------------------
    
def prep_img_for_display(img, zf=1, lp=5, hp=95, stretch=[1.0,1.0]):
    
    import numpy as np
    from scipy.ndimage.interpolation import zoom
    import warnings
    
        # get the min and max values for the defined percentiles
    scale_min = np.percentile(np.nan_to_num(img), lp)
    scale_max = np.percentile(np.nan_to_num(img), hp)
    
        # define the offset and scale based on the min and max values
    offset = scale_min
    if scale_max != scale_min:
        scale = 1.0/float(scale_max-scale_min)
    else:
        scale = 1.0
    
        # disable warnings because of potential disturbing warnings in the zoom function
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
            # rescale the intensities and resize the image
        disp_img = zoom(scale*(img-offset), np.flip(zf*np.array(stretch),0))
    
        #return the data (replacing the NaN values)
    return np.nan_to_num(disp_img)
    
# ------------------------------------------------------------------
#                            show_img
# ------------------------------------------------------------------
                            
# Display an image using matplotlib                  

# Parameters:
#   img = array containing the image data 
#   title = title to display above the image
#   wmax = width (in pixels) up to which the image is maginfied
#   dr = array of rectangles to display on top of the image. Each
#       element is a tuple defined as (rectangle, color)
#
# Return value:
#   None                       
# ------------------------------------------------------------------
    
def show_img(img, title='', wmax=500, dr=[], stretch=[1.0,1.0]):
    
    import matplotlib
    import matplotlib.pyplot as plt 
    
        # rescale the intensities for display
    disp_img = prep_img_for_display(img, stretch=stretch)
    
        # get the image size and compute the display resolution
    szx = disp_img.shape[1]
    szy = disp_img.shape[0]
    dpi_img = 50.0*szx/min(szx,wmax)
    
        # create the plot with the desired size
    f = plt.figure(figsize = (szx/dpi_img, szy/dpi_img), dpi=100)
    sp = f.add_subplot(111)
    
        # include the image
    sp.imshow(disp_img, cmap='gray', vmin=0.0, vmax=1.0, interpolation='none')
    
        # if necessary, write the title
    if title != '':
        sp.set_title(title)
        
        # hide the axes
    sp.axis('off')

        # color palette. this means that a max of 10 ROIs can be selected
    c = ['orange', 'y','c', 'deeppink', 'm', 'mediumpurple', 'r', 'b', 'g','lightcoral']
    
        # draw all rectangles    
    
    for label, rect in enumerate (dr):
        rx = rect[0]*stretch[0]
        ry = rect[1]*stretch[1]
        rw = rect[2]*stretch[0]
        rh = rect[3]*stretch[1]
        rect_p = matplotlib.patches.Rectangle((rx,ry), rw, rh, color = c[label], label = str(label+1), fill=False, linewidth = 2)
        sp.add_patch(rect_p)

        # display the plot
    plt.show()
    # plt.legend()
    
# ------------------------------------------------------------------
#                            crop_img
# ------------------------------------------------------------------
                            
# Return a sub region from an image                  

# Parameters:
#   img = array containing the image data
#   roi = rectangle (x1, y1, x2, y2) defining the region of interest      
#
# Return value:
#   sub image for the defined region of interest                     
# ------------------------------------------------------------------
    
def crop_img(img, roi_crop, **kwargs):
    
        # if no ROI is defined, select the full image
    if roi_crop == 0:
        retval = img
        
        # otherwise, select the corresponding data
    else:
        retval = img[roi_crop[1]:roi_crop[1]+roi_crop[3],roi_crop[0]:roi_crop[0]+roi_crop[2]]
        
        # return the data
    return retval
    

# =============================================================================
#                              select_rois
# =============================================================================
def select_rois(img, list_rois = [], title = 'Select ROI' , flip_img = False, **kwargs):
    '''
    This function provides an interactive interface for selecting a rectangular region of interest in an image.
    This function is meant to be used with the `%load`  magic in jupyter.
    This function requires to have 'img_utils' functions loaded (specifically the function 'prep_img_for_display')
    
    'select_roi' function loads and displays the first image in this folder, as a guide for the ROI selection.

    Parameters
    ----------
    img : 2D numpy array
        Source image
    list_rois : list of strs
        list of strs with the names assigned that want to be assigned to the ROIs
    title : str, optional
        Assign a name to the ROI selection window. The default is 'Select ROI'.
    flip_img : bool, optional
        Vertical flip to match imageJ formet, if required. The default is False.

    Returns
    -------
    variable
        Variable name = values (x,y,w,h)

    '''
    
        # import necessary packages
    import cv2
    import pandas as pd
    import numpy as np
    from img_utils_mod import prep_img_for_display
    
        # if the case requires it, it flips the image to match it with ImageJ
    if flip_img:
        img = np.flip(img)
        
        # select the ROIs (after selecting each ROI press ENTER, when finish, press ESC)
    rois = cv2.selectROIs(title ,prep_img_for_display(img), showCrosshair = False)
    cv2.destroyAllWindows()

        # 
    if len(list_rois) < len(rois):
        
            # if more ROIs than names requested  were selected, add 'extra_roi_#' to the final result
        for idx in range (abs(len(list_rois) -len(rois))):
            list_rois.append('extra_roi_'+str(idx + 1))
        
        # if the ROIs selected do not match with the number of names requested, display an error message
    elif len(list_rois) > len(rois):
        return ('Error: number of ROIS selected does not agree with the requested names extension. Missing one or more ROIs.' '\n Try again.')
    
        # Create a DataFrame with the list of names, ROIs selected and column names
    ROI = pd.DataFrame(rois,index = list_rois,columns = ['X','Y','Width','Height'])
    
        # initialize and create a list with name = values for each region
    rois = []
    for i in range(len(ROI)):
        rois.append( ROI.index[i] + ' = ' + str(ROI.values.tolist()[i]))
        
        # return the list of values in a legible format for %load magic
    return ('\n'.join(rois))




