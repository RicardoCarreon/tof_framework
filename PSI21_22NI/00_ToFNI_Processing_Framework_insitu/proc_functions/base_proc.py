# ******************************************************************

#                           BASE_PROC

# Collection of base processing functions for the Pyerre framework

# ******************************************************************

import numpy as np
from img_utils import show_img

"""
------------------------------------------------------------------
                           int_corr
------------------------------------------------------------------

description:
    to be completed ...
    
input parameters: 
    src_img = source image
    ref_img = reference image
    nca = non changing area (x1, y1, width, height)
    dnca = dark non changing areas(x1, y1, width, height)

return value: 
    processed image
"""

def int_corr(src_img, ref_img, nca=None, dnca=None, **kwargs):
    
    def avg_val(img, nca_def):
        
        nca_msk = np.zeros(img.shape)
        
        if not isinstance(nca_def[0], list):
            nca_list = [nca_def]
        else:
            nca_list = nca_def
            
        for nca in nca_list:
            x0 = nca[0]
            y0 = nca[1]
            w = nca[2]
            h = nca[3]
            nca_msk[y0:y0+h, x0:x0+w] = 1
            
        nca_avg = np.sum(img*nca_msk) / np.sum(nca_msk)
        
        return nca_avg
    
    # -----
    
        # if no NCA is defined, just return the source image
    if nca is None:
        dst_img = src_img
        
        # if NCA is defined but not DNCA, just multiply the source image by a factor to match the ref image intensity
    elif dnca is None:
            nca_src_val = avg_val(src_img, nca)
            nca_ref_val = avg_val(ref_img, nca)
            int_ratio = nca_src_val / nca_ref_val
            dst_img = src_img / int_ratio
            
        # if both NCA and DNCA are defined, compute a gain and offset so that the image matches the reference
        # in both regions
    else:
            # get the values in NCA and DNCA regions for the source and ref images
        nca_src_val = avg_val(src_img, nca)
        nca_ref_val = avg_val(ref_img, nca)
        dnca_src_val = avg_val(src_img, dnca)
        dnca_ref_val = avg_val(ref_img, dnca)
            # compute gain and offset
        gain = (nca_ref_val - dnca_ref_val)/(nca_src_val - dnca_src_val)
        offset = dnca_ref_val - dnca_src_val*gain
            # apply the intensity correction
        dst_img = gain*src_img + offset
        
    return dst_img