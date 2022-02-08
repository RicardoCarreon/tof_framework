#                       -------- SBKG.py --------

# This file contains a list of functions usefule for measuring and correcting
# the scattered background

# =============================================================================
#                          create_sbkg
# =============================================================================
def create_sbkg(src_img, BB_mask_img, **kwargs):
    """
    Creates a SBKG image from a base image and its respective BB mask

    Parameters
    ----------
    base_img : 2D array
        source image, base to create the sbkg 
    BB_mask_img : 2D BB mask image. array
        BB mask corresponding to the source image
        
    Returns
    -------
    2D array corresponding to the sbkg image for correction
    
    """
        # initialize libraries
    import numpy as np
    from skimage.measure import label
    from scipy.interpolate import Rbf

        # extract the image shape
    s_row = src_img.shape[0]
    s_col = BB_mask_img.shape[1]
    
        # identify and enumerate each BB in the mask
    msk_reg = label(BB_mask_img)
    
        # take the max number of BBs in the image
    bb_count = np.max(msk_reg)
    
        # initialize the region value variables
    xvals = np.zeros(bb_count)
    yvals = np.zeros(bb_count)
    ivals = np.zeros(bb_count)
    
        # create a matrix of values corresponding to the images shape
    xb = np.matmul(np.ones(s_row).reshape(-1,1),np.arange(s_col,dtype='float').reshape(1,s_col))
    yb = np.matmul(np.arange(s_row).reshape(-1,1),np.ones(s_col,dtype='float').reshape(1,s_col))
    
    for i in range (bb_count):
            # for each BB take all the values that satisfy the requirement in the loop 
        reg = np.where(msk_reg == i+1)
        
            # vector construction for x/y and the values per region in the base image
        xvals [i] = round(np.nanmean(xb[reg]))
        yvals [i] = round(np.nanmean(yb[reg]))
        ivals [i] = np.nanmean(src_img[reg])

        # prepare vectors for linear least squared regression
    A = np.array([xvals*0+1, xvals, yvals]).T
    B = ivals.flatten()
    
        # linear least square approximation for linear regression 
    coeff, _, _, _ = np.linalg.lstsq(A, B,rcond=None)
    
        # extract the coefficient
    vl = coeff[0] + coeff[1]*xvals + coeff[2]*yvals
    vli = coeff[0] + coeff[1]*xb + coeff[2]*yb
    
        # interpolate values for the SBKG image constrcution 
    rbfi2 = Rbf(xvals, yvals, ivals-vl, function='thin_plate')

    SBKG_image = rbfi2(xb, yb) + vli

    return SBKG_image