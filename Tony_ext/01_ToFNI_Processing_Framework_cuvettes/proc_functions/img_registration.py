# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:42:08 2022

@author: carreon_r
"""

from img_utils_mod import *
import matplotlib.pyplot as plt

# =============================================================================
#                       img_registration
# =============================================================================

def img_registration (ref_img, src_img, num_iter = 10, show_corr = False):
    '''
    This function takes an image that requires registration (rotation, translation) and compares it with a reference image to make the 
    necessary operations and correct any displacements that ocurred during the experiments.

    Returns
    -------
    full parameters and a 2D array
        DESCRIPTION.

    '''
    import os
    import scipy as sp
    import scipy.misc
    import imreg_dft as ird
    import numpy as np
    
        # ignores warning messages
    np.seterr(all='ignore')
    result = ird.similarity(ref_img, src_img, numiter=num_iter)
    
    assert "timg" in result
    
    if show_corr:
        ird.imshow(ref_img, src_img, result['timg'])
    
    return result, result['timg']