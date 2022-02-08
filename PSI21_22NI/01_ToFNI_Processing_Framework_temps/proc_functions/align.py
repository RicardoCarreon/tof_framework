# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:45:03 2020

@author: boillat
"""
import numpy as np
import scipy as sp
import cv2 
from img_utils import oversample_img

def img_align(src_img, ref_img, rois_list, max_shift=20, subpix=10,
              dof=['tx','ty','sx','sy'], debug_data={}):
    
    meas_shifts = []
    
    debug_data['corr_maps'] = []
    debug_data['meas_shifts'] = []
    
    A = np.empty([0,6])
    B = np.empty([0,1])
    
    ########################## cropping ROIs in ref,src images with extra boundary K
    for i, roi_def in enumerate(rois_list):
        
            # check if the roi definition is a tuple. In this case,
            # the first element contains the ROi coordinates, and the
            # second a string defining the type (horizontal, vertical or both)
        if isinstance(roi_def, tuple):
            roi = roi_def[0]
            roi_type = roi_def[1]
            # if not a tuple, the element contains only the coordinates anf the
            # type is implicitely 'both'
        else:
            roi = roi_def
            roi_type = 'both'
        
            # compute the horizontal and vertical margins as a function of the type
        if roi_type in ['v', 'vert', 'vertical']:
            kx = 0
            ky = max_shift
        elif roi_type in ['h', 'horiz', 'horizontal']:
            kx = max_shift
            ky = 0
        else: 
            kx = max_shift
            ky = max_shift
            
            # get the sub-region in the reference image
        crop_ref_int16 = ref_img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        
            # get the sub-region in the image to be aligned, including the margins
            # TODO: avoid erros if these margins get over the border
        crop_shift_int16 = src_img[roi[1]-ky:roi[1]+roi[3]+ky, roi[0]-kx:roi[0]+roi[2]+kx]

    ######################### resample the images with sub-pixel resolution and template matching
    
            # resample the images
        crop_shift_r = oversample_img(crop_shift_int16, factor=subpix)
        crop_ref_r = oversample_img(crop_ref_int16, factor=subpix)
            
            # convert to floating point
        crop_ref = np.asarray(crop_ref_r, dtype = np.float32)
        crop_shift = np.asarray(crop_shift_r, dtype = np.float32)
        
            # perform the template matching
        res = cv2.matchTemplate(crop_shift, crop_ref, cv2.TM_CCORR_NORMED)
        
            # find the position with aximum correlation
        shifts = np.array(cv2.minMaxLoc(res)[3])/subpix - np.array([kx,ky])
        
            # add to the results
        meas_shifts.append(shifts)
        
            # store for debugging purpose
        debug_data['corr_maps'].append(res)
        debug_data['meas_shifts'].append(shifts)
        
    ######################## add the corresponding equations
    
        # (order of variables in B: r11 r12 r21 r22 tx ty)
        
            # if the type is both or horizontal, add the equation from the horizontal shift
        if not roi_type in ['v', 'vert', 'vertical']:
            line_A = np.array([roi[0]+roi[2]/2, roi[1]+roi[3]/2, 0, 0, 1, 0]) #x parts for A
            line_B = shifts[0] + roi[0]+roi[2]/2 #x parts for B
            A = np.vstack((A, line_A))
            B = np.vstack((B, line_B))
        
            # if the type is both or vertical, add the equation from the vertical shift
        if not roi_type in ['h', 'horiz', 'horizontal']:
            line_A = np.array([0, 0, roi[0]+roi[2]/2, roi[1]+roi[3]/2, 0, 1]) #y parts for A
            line_B = shifts[1] + roi[1]+roi[3]/2 #y parts for B
            A = np.vstack((A, line_A))
            B = np.vstack((B, line_B))
            
    ########################## constrains for displacement only 
        
    dof_list = [x in dof for x in ['zx','sx','sy','zy','tx','ty']]
    def_vals = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        
        # remove the columns in A corresponding to fixed variables
    for i in reversed(range(6)):
        if not dof_list[i]:
            col_val = A[:,i]
            B = B - def_vals[i]*np.reshape(col_val, B.shape)
            A = np.delete(A, i, 1)
        
        
    debug_data['A'] = A
    debug_data['B'] = B
    
    ########################## solve for M matrix 
    if A.shape[0] > A.shape[1]:
        AT = A.transpose()
        C1 = np.linalg.inv(np.dot(AT,A))
        C2 = np.dot(C1, AT)
        C3 = np.dot(C2, B)
        
    elif A.shape[0] == A.shape[1]:
        C3 = np.linalg.solve(A, B)
        
    else:
        raise ValueError("Needs more ROIs")
        
    ic = 0
    C = np.zeros(6)
    
    for i in range(6):
        if dof_list[i]:
            C[i] = C3[ic]
            ic += 1
        else:
            C[i] = def_vals[i]
        
    
    debug_data['C'] = C
    
    ########################## finding M according to constrains
    """if 'v' in const and 'h' in const:
        M = np.array([[1,C[0,0],C[2,0]], [C[1,0],1,C[3,0]], [0,0,1]]) #not sure about the orders here
        
    elif 'v' in const:
        M = np.array([[C[0,0],C[2,0],C[3,0]], [C[1,0],1,C[4,0]], [0,0,1]])
      
    elif 'h' in const:
        M = np.array([[1,C[1,0],C[3,0]], [C[0,0],C[2,0],C[4,0]], [0,0,1]])
        
    else:"""
    M = np.array([[C[3],C[2],C[5]], [C[1],C[0],C[4]], [0,0,1]])    
    
    debug_data['M'] = M 
    
    #rows, cols = src_img.shape[0], src_img.shape[1] 
    
    corrected_img = sp.ndimage.affine_transform(src_img, M, order=1)

    return corrected_img