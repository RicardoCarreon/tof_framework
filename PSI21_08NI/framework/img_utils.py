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

from file_utils import file_list
import scipy.interpolate
import numpy as np
from warnings import warn

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

def get_img(file_name, base_dir='', squeeze=True, get_header=False):
    
    from astropy.io import fits
    import os
    import warnings
    import matplotlib.pyplot as plt
    import numpy as np
    import h5py
    
        # get the file extension to identify the type
    _ , ext = os.path.splitext(file_name)
    
        # if the header was requested but the format is not FITS, issue a warning
    if get_header and ext != '.fits':
        warn('Header requested for a non-FITS file')
    
        # case of FITS files
    if ext == '.fits':
        
            # disable warnings because of possible bad characters in the header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
                # open the file and get the image data and header
            with fits.open(os.path.join(base_dir, file_name), ignore_missing_end=True) as hdul:
                hdul.verify('fix')
                img = hdul[0].data
                header = hdul[0].header
        
                # reverse the y coordinates (for consistency with ImageJ)
            img = np.flip(img, 0)
    
        # case of TIFF files
    elif ext == '.tif':
        
            # get the image
        img = plt.imread(os.path.join(base_dir, file_name))
        
        # case of HDF files
    elif ext == '.hdf':
        
        hdf = h5py.File(os.path.join(base_dir, file_name),'r')['entry']['data']['data']
        img = np.zeros(hdf.shape)
        for i, frame in enumerate(hdf):
            img[i,:,:] = frame
        
        # general case
    else:
        
            # get the image
        img = plt.imread(os.path.join(base_dir, file_name))
    
        # if the 'squeeze' option is on, remove dimensions with size 1
    if squeeze:
        img = np.squeeze(img)
    
        # if the format is FITS and the header was requested
    if ext == '.fits' and get_header:
            # return a tuple with the data and header
        return (img, header)
    
    else:
            # otherwise return only the data
        return img

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

def write_img(img, file_name, base_dir='', overwrite=False, header=None):
    
    import os
    from astropy.io import fits
    import numpy as np
    
        # compute the destination file name
    dst_name = os.path.join(base_dir, file_name)
    
        # get the destination directory
    dst_name_dir, _ = os.path.split(dst_name)
    
        # and create this destination directory if it does not exist
    if not os.path.exists(dst_name_dir):
        os.makedirs(os.path.join(dst_name_dir,''))
        
        # reverse the y coordinates (for consistency with ImageJ)
    img = np.flip(img, 0)
    
        # try to save the processed image
    try:                        
            # create the HDU based on the image
        hdu = fits.PrimaryHDU(img)
        
            # if a header was specified
        if header != None:
                # add the header to the HDU
            hdu.header = header
                # reset the scale parameters
            hdu.header['BZERO'] = 0
            hdu.header['BSCALE'] = 1
            
            # write the file
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
    
def show_img(img, title='', wmax=750, dr=[], stretch=[1.0,1.0], keep_fig=False,
             nrows=1, ncols=1, index=1, do_show=True, font_size=9):
    
    import matplotlib
    import matplotlib.pyplot as plt  
    
    global current_fig
    
        # update the font size for the title
    matplotlib.rcParams.update({'font.size': font_size})
    
        # rescale the intensities for display
    disp_img = prep_img_for_display(img, stretch=stretch)
    
        # get the image size and compute the display resolution
    szx = disp_img.shape[1]
    szy = disp_img.shape[0]
    dpi_img = 50.0*(szx*ncols)/min((szx*ncols),wmax)
    
        # create the plot with the desired size
    if not keep_fig:
        f = plt.figure(figsize = (szx*ncols/dpi_img, szy*nrows/dpi_img), dpi=100)
        current_fig = f
    else:
        f = current_fig
    sp = f.add_subplot(nrows, ncols, index)
    
        # include the image
    sp.imshow(disp_img, cmap='gray', vmin=0.0, vmax=1.0, interpolation='none')
    
        # if necessary, write the title
    if title != '':
        sp.set_title(title)
        
        # hide the axes
    sp.axis('off')

        # draw all rectangles    
    for rects, col in dr:
        if type(rects[0]) == list or (isinstance(rects[0], tuple)):
            rlist = rects
            show_num = True
        else:
            rlist = [rects]
            show_num = False
        
        for i, rect_def in enumerate(rlist):
            if isinstance(rect_def, tuple):
                rect = rect_def[0]
                name = str(i+1) + ' (' + rect_def[1] + ')'
                pass
            else:
                rect = rect_def
                name = str(i+1)
            rx = rect[0]*stretch[0]
            ry = rect[1]*stretch[1]
            rw = rect[2]*stretch[0]
            rh = rect[3]*stretch[1]
            rect_fill = matplotlib.patches.Rectangle((rx,ry), rw, rh, color=col, 
                                                  fill=True, alpha=0.25)
            rect_border = matplotlib.patches.Rectangle((rx,ry), rw, rh, color=col, fill=False)
            sp.add_patch(rect_fill)
            sp.add_patch(rect_border)
            if show_num:
                sp.text(rx+rw/2, ry+rh/2, name, color='black', fontsize='medium',
                        fontweight='bold', ha='center', va='center', backgroundcolor=col)
        
        # display the plot
    if do_show:
        plt.show()
    
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
    
def crop_img(img, roi):
    
        # if no ROI is defined, select the full image
    if roi == 0:
        retval = img
        
        # otherwise, select the corresponding data
    else:
        retval = img[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        
        # return the data
    return retval
    
# ------------------------------------------------------------------
#                            get_3d_img
# ------------------------------------------------------------------
                            
# loads a stack of images and returns it as a 3D array       
                     
# note: the order of indices in the 3D array is (z, y, x) to be consistent
#   with the matrix indices order used by numpy. z represents the 'depth'
#   direction (image number) and x and y represents the horizontal (x) and 
#   vertical (y) image coordinates     
    
# Parameters:
#   src_dir = directory containing the image stack relative to the base directory
#   base_dir = path of the base directory (default is '', in which case
#       'dir_name' should contain an absolute path)      
#   ftype = file type (default is 'fits')
#
# Return value:
#   img = 3D array containing the image data   
    
# ------------------------------------------------------------------
         
      
def get_3d_img(src_dir, base_dir='', ftype='fits', silent=False):
    
    from processors import exec_start, exec_end
    import numpy as np      
    import os          
    
        # processing start indicator
    if not silent:
        start_time = exec_start()
    
        # get the list of all FITS files in the specified source directory
        # (including in its subdirectories)
    flist = file_list(src_dir, base_dir, ftype)
        
        # get the number of files in the list
    nfiles = len(flist)
    
        # get the first image
    first_img = get_img(os.path.join(src_dir, flist[0]), base_dir)
    
        # get the x and y size
    szx = first_img.shape[1]
    szy = first_img.shape[0]
    
        # create the destination buffer
    img_3d = np.zeros([nfiles, szy, szx], dtype=first_img.dtype)
    
        # put the first image into the buffer
    img_3d[0,:,:] = first_img
    
        # loop for all remaining files
    for z in np.arange(1, nfiles):      
        
            # load the image and copy into the buffer
        img_3d[z,:,:] = get_img(os.path.join(src_dir, flist[z]), base_dir)
        
            # if not in silent mode, output the progress indication
        if not silent:
            print('Reading (' + str(round(1000*(z+1)/nfiles)/10.0) + '%)', end='\r')
    
        # processing end indicator
    if not silent:
        print('')
        exec_end(start_time)
            
    return img_3d

# ------------------------------------------------------------------
#                            write_3d_img
# ------------------------------------------------------------------
                            
# writes a stack of FITS images from a 3D numpy array    

# note: the order of indices in the 3D array is (z, y, x) to be consistent
#   with the matrix indices order used by numpy. z represents the 'depth'
#   direction (image number) and x and y represents the horizontal (x) and 
#   vertical (y) image coordinates                      

# Parameters:
#   img = 3D numpy array containing the images
#   dst_fname = template for the output file names (an index is added at the end of the name)
#   base_dir = path of the base directory (default is '', in which case
#       'dir_name' should contain an absolute path)   
#   silent = if set to True, the progress indication is not output
#
# Return value:
#   none
    
# ------------------------------------------------------------------      
    
def write_3d_img(img, dst_fname, base_dir='', silent=False, overwrite=False):

    from processors import exec_start, exec_end
    import numpy as np      
    import os       
    
        # processing start indicator
    if not silent:
        exec_start()
    
        # get the number of files to be written
    nfiles = img.shape[0]
    
        # extract the base file name and the extension
    file_base, ext = os.path.splitext(dst_fname)
    
        # loop for all slices
    for z in np.arange(0, nfiles):
        
            # compute the destination file name by adding an index
        current_dst = file_base + '_' + format(z, '05d') + ext
        
            # and write the corresponding image
        write_img(np.flip(img[z,:,:], 0), current_dst, base_dir, overwrite=overwrite)
        
            # if not in silent mode, output the progress indication
        if not silent:
            print('Writing (' + str(round(1000*(z+1)/nfiles)/10.0) + '%)', end='\r')
    
        # processing end indicator
    if not silent:
        print('')
        exec_end()
        
# ------------------------------------------------------------------

def oversample_img(img, factor=10, order=1):
    """
    Resamples an image with sub-pixel resolution

    Parameters
    ----------
    img : 2D numpy array
        Source image.
    factor : integer, optional
        Resampling factor. The default is 10.
    order : integer, optional
        order of interpolation. The default is 1, corresponding to
        bilinear interpolation.

    Returns
    -------
    img_int : 2D numpy array
        Resampled image.

    """
        # get the size of the original image
    szx = img.shape[0]
    szy = img.shape[1]
    
        # create an interpolation function of the desired order
    intf = scipy.interpolate.RectBivariateSpline(range(szx), range(szy), img, \
                                              kx=order, ky=order)
    
        # apply the interpolation function to a grid with sub-pixel resolution
    img_int = intf(np.arange(szx*factor)/factor, np.arange(szy*factor)/factor)
    
        # return the interpolated image
    return img_int
# ------------------------------------------------------------------

app = 0

def img_roi_selector(img, default=None, title='ROI selector',
                     multiple=False, roi_names=[], options=[]):
    
    import pyqtgraph as pg
    from PyQt5.QtCore import QRectF
    import PyQt5.QtCore
    from PyQt5.QtWidgets import QPushButton, QWidget, QHBoxLayout, QVBoxLayout, QApplication
    
    global app
    
    roi_list = []
    roi_label_list = []
    roi_opt_list = []
    
    flex_list = multiple and roi_names == []
    
    def updateLabels():
        
        for i, label in enumerate(roi_label_list):
            if roi_names != []:
                txt = roi_names[i]
            else:
                txt = str(i+1)
            if options != []:
                txt = txt + ' (' + options[roi_opt_list[i]] + ')'
            label.setText(txt)
    
    def addRoi(x0, y0, w, h, option=''):
    
            # create the ROI and add it to the view
        roi = pg.ROI([x0,y0], [w,h], maxBounds=QRectF(0,0,szx,szy), scaleSnap=True, 
                     translateSnap=True, pen=pg.mkPen('r', width=2), rotatable=False,
                     removable=flex_list)
        roi.handlePen = pg.mkPen('r', width=2)
        view.addItem(roi)
        
            # add all 8 resizing handles to the ROI
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        roi.addScaleHandle([0, 0.5], [1, 0.5])
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        roi.addScaleHandle([0.5, 1], [0.5, 0])
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
            
        if flex_list:
            roi.sigRemoveRequested.connect(roiRemoveRequested)
            
        if options != []:
            roi.setAcceptedMouseButtons(PyQt5.QtCore.Qt.LeftButton)
            roi.sigClicked.connect(roiClicked)
        
        if multiple:
            roi_label = pg.TextItem('', color='r', border=pg.mkPen('r'),
                                    fill=pg.mkBrush(255,255,255,192))
            view.addItem(roi_label)
            roi_label.setParentItem(roi)
            roi_label_list.append(roi_label)
            
        roi_list.append(roi)
        if options != []:
            if option == '':
                opt_index = 0
            else:
                opt_index = options.index(option)
            roi_opt_list.append(opt_index)
        
        updateLabels()
        
        # function responding to the OK button
    def addRoiButtonPressed():
        if options == []:
            addRoi(*default_roi)
        else:
            addRoi(*(default_roi[0]), default_roi[1])

        # function responding to the OK button
    def okButtonPressed():
        app.quit()
    
        # function responding to the Cancel button
    def cancelButtonPressed():
            # quit with an empty ROI value
        roi_list = []
        app.quit()
    
        # function responding to the Cancel button
    def roiRemoveRequested(roi):
        
        i = roi_list.index(roi)
        view.removeItem(roi_list[i])
        view.removeItem(roi_label_list[i])
        roi_list.pop(i)
        roi_label_list.pop(i)
        
        updateLabels()
        
    def roiClicked(roi):
        
        i = roi_list.index(roi)
        roi_opt_list[i] = (roi_opt_list[i] + 1) % len(options)
        
        updateLabels()
        
    
        # get the image size
    szy, szx = img.shape
    
    if options == []:
        default_roi = [int(szx/2-szx/20), int(szy/2-szy/20), int(szx/10), int(szy/10)]
    else:
        default_roi = ([int(szx/2-szx/20), int(szy/2-szy/20), int(szx/10), int(szy/10)], '')
        

        # create the applications
    if app == 0:
        app = QApplication([])
    
        # create the main window
    main = QWidget()
    main.setWindowTitle(title)
    
        # create the graphics layout (to display the image)
    pg.setConfigOptions(imageAxisOrder='row-major')
    glw = pg.GraphicsLayoutWidget(size=(1000,1000*szy/szx), border=True)
    w1 = glw.addLayout(row=0, col=0)
    view = w1.addViewBox(row=0, col=0, lockAspect=True)
    
        # add the image to the layout
    img_item = pg.ImageItem(img, levels=[0,1])
    view.addItem(img_item)
    
        # case of multiple ROI selector
    if multiple:
            # case of open list (no names specified)
        if roi_names == []:
                # if a defualt ROI list is given, place them
                # (otherwise we start with zero ROIs)
            if default != None:
                for roi in default:
                    if options == []:
                        addRoi(*roi)
                    else:
                        addRoi(*(roi[0]), roi[1])
                    
            # case of named list (fixed length)
        else:
            for i, roi in enumerate(roi_names):
                if default != None and i < len(default):
                    init_roi = default[i]
                else:
                    init_roi = default_roi
                if options == []:
                    addRoi(*init_roi)
                else:
                    addRoi(*(init_roi[0]), init_roi[1])
    
        # case of single ROI selector
    else:
            # if a default position/size is given, use it
        if default != None:
            init_roi = default
            # otherwise place the ROi in the middle, with 10% of the image size
        else:
            init_roi = default_roi
            
            # create the ROI
        addRoi(*init_roi)
    
        # disable auto range in x and y directions
    view.disableAutoRange('xy')
    view.autoRange()

        # create the OK and Cancel buttons
    okButton = QPushButton("OK")
    cancelButton = QPushButton("Cancel")
    if flex_list:
        addRoiButton = QPushButton("Add ROI")
    
        # connect the buttons to their handling functions
    okButton.clicked.connect(okButtonPressed)
    cancelButton.clicked.connect(cancelButtonPressed)
    if flex_list:
        addRoiButton.clicked.connect(addRoiButtonPressed)

        # create a horizontal layout with the buttons
    hbox = QHBoxLayout()
    if flex_list:
        hbox.addWidget(addRoiButton)
    hbox.addStretch(1)
    hbox.addWidget(okButton)
    hbox.addWidget(cancelButton)

        # create a vertical layout with the image view and the
        # buttons layout below
    vbox = QVBoxLayout()
    vbox.addWidget(glw)
    vbox.addLayout(hbox)

        # set this vertical layout as main widget
    main.setLayout(vbox)
    
        # show the main window
    main.show()

        # start the vent loop
    app.exec_()
    
        # set the return value to the selected ROI(s) and quit
    if multiple:
        if options != []:
            retval = []
            for i, roi in enumerate(roi_list):
                retval.append(([int(roi.pos()[0]), int(roi.pos()[1]),
                          int(roi.size()[0]), int(roi.size()[1])], options[roi_opt_list[i]]))
                
        else:
            retval = [[int(roi.pos()[0]), int(roi.pos()[1]),
                      int(roi.size()[0]), int(roi.size()[1])] for roi in roi_list]
    else:
        retval = [int(roi_list[0].pos()[0]), int(roi_list[0].pos()[1]),
                  int(roi_list[0].size()[0]), int(roi_list[0].size()[1])]
        
        # return the ROI
    return retval