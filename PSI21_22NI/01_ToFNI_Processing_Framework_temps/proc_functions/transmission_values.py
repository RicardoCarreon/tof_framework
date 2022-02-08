# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:47:00 2020

@author: carreon_r
"""

import sys, os
sys.path.append('framework')
from magic_selectors import * # import image utilities
from img_utils import * # import image utilities



src_dir = r"H:\700 Campaigns - internal\780 2020\PSI20_01NI\temp\IP_Yvette_01\Transmission_results"


# this function will create an excel sheet by selecting different regions of interest in the routine.
# if used with %load magic, it will give you all the ROIs that you selected

def get_transmission (src_dir, proc_folder= [], name_xlsx = 'Transmission_values.xlsx', **kwargs):
    
    import cv2, os, sys
    import numpy as np
    import pandas as pd
    
    Values = pd.DataFrame()
    
    list_rois = []
    
    if proc_folder != []:
        folder_list = proc_folder
    
    else:
        folder_list = os.listdir(src_dir)
    
    for folder in folder_list:
        folder_dir = src_dir + '/' + folder
            # reads the whole folder
        flist = file_list(src_dir + '/' + folder, base_dir = '')
            # get the middle image in that folder
        img_ref = get_img(flist[np.int(len(flist)/2)], os.path.join(folder_dir))
        
        zf = min(1000.0/img_ref.shape[1], 500.0/ img_ref.shape[0])
        
            # Select the ROI
        rois = cv2.selectROIs('Folder ' + folder , prep_img_for_display(img_ref, zf), showCrosshair = False)
        cv2. destroyAllWindows()
        
        list_rois.append('roi_' + folder + ' = ' +str([list(np.array(roi))for roi in rois]))
        
        results_vec = np.zeros([len(flist), len(rois)])
        
        for i, name in enumerate(flist):
            img = get_img(name, os.path.join(folder_dir))
            
            for j, roi in enumerate(rois.tolist()):
                results_vec[i,j] = np.average(crop_img(img, roi))
            
        for k in range(results_vec.shape[1]):
            Values[folder + '_roi_' + str(k+1)] = results_vec[:,k]
    
    Values.to_excel(src_dir + '/' + name_xlsx)
    
    return('\n'.join(list_rois))
        

# this function creates an excel sheet of transmission values by taking ROI values from ImageJ (manual input)
# foledrs and rois must be in order 1:1 to be able to extract the desire ROI

def get_transmission_rois (src_dir, proc_folder= [], list_rois = [], name_xlsx = 'Transmission_values.xlsx', **kwargs):

    import os, sys
    import numpy as np
    import pandas as pd
    
    Values = pd.DataFrame()

    if proc_folder == []:
        return print( 'Please select specific folders to process. Try again')
    
    for folder, roi in zip(proc_folder, list_rois):

        folder_dir = os.path.join(src_dir, folder)
            # reads the whole folder
        flist = file_list(src_dir + '/' + folder, base_dir = '')
        
        results_vec = np.zeros([len(flist), len(roi)])
        
        for i, name in enumerate(flist):
            img = get_img(name, os.path.join(folder_dir))
            
            for j, roi_val in enumerate(roi):
                results_vec[i,j] = np.average(crop_img(img, roi_val))
            
        for k in range(results_vec.shape[1]):
            Values[folder + '_roi_' + str(k+1)] = results_vec[:,k]
    
    Values.to_excel(src_dir + '/' + name_xlsx)
    
    return


















