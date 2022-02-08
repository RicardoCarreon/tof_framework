# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:42:08 2022

@author: carreon_r
"""

from img_utils_mod import *
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
#                       img_registration
# =============================================================================

def img_registration (img, ref= np.zeros([512,512]), num_iter = 10, show_corr = False, **kwargs):
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
    result = ird.similarity(ref, src, numiter=num_iter)
    
    assert "timg" in result
    
    if show_corr:
        ird.imshow(ref, src, result['timg'])
    
    return result['timg']




# =============================================================================
#                       img_registration_dict
# =============================================================================

def img_registration_dict (stack_dict, proc_folder, ref= np.zeros([512,512]), num_iter = 3, show_corr = False, **kwargs):
    
    for key, values in stack_dict.items():

        if key in proc_folder:
            new_imgs = []
            print('Performing Image Registration')
            for img in values:
                new_img = img_registration (img[0], ref[0], num_iter = num_iter, show_corr = False)
                new_list.append((new_img, img[1]))
            
            stack_dict[key] = new_list



# =============================================================================
#                         select_file
# =============================================================================
def select_file(var_name, base_dir=''):
    '''
    This function is meant to be used together with the '%load' magic in jupyter
    notebook to provide an interactive interface for directory selection.

    Parameters
    ----------
    var_name : str
        String with the name of the variable which will contain the resulting path in the notebook
    base_dir : str, optional
        directory relative to which the directory path is specified (absolute path is used if omitted)

    Returns
    -------
    str
        statement to be inserted in the jupyter code cell to assign the selected path to this variable

    '''
        # import necessary packages
    import os
    import tkinter
    from tkinter import filedialog
    
        # Create Tk root
    root = tkinter.Tk()
    
        # Hide the main window
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    
        # If a base path is not specified, select and return the absolute path
    if base_dir == '':
        dir = filedialog.askopenfilename()
        
        # If a base path is specified, use it as starting directory and return
        # the selected directory relative to this base path
    else: 
        dir = os.path.relpath(filedialog.askopenfilename(initialdir=base_dir), start=base_dir)
    
        # generate a statement to be inserted in the jupyter notebook
    return var_name + ' = r\"' + os.path.normpath(dir) + '\"'