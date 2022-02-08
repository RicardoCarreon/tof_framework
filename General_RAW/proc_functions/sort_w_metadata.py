# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 10:39:03 2021

@author: carreon_r
"""

import os, glob, time
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt


from img_utils_mod import *

# =============================================================================
#                         select_directory
# =============================================================================
def select_directory(var_name, base_dir=''):
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
        dir = filedialog.askdirectory()
        
        # If a base path is specified, use it as starting directory and return
        # the selected directory relative to this base path
    else: 
        dir = os.path.relpath(filedialog.askdirectory(initialdir=base_dir), start=base_dir)
    
        # generate a statement to be inserted in the jupyter notebook
    return var_name + ' = r\"' + os.path.normpath(dir) + '\"'



# =============================================================================
#                            ifnot_create_folder
# =============================================================================   
def ifnot_create_folder(save_path):
    """
    Simple instruction to create a folder if it does not exist already

    Parameters
    ----------
    save_path : String
        Source directory, where your .TXT files for the given experiment are saved.

    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return 



# =============================================================================
#                         filter_files_dir
# =============================================================================

# This function takes a directory and puts in a list all the files containing a specific name or extension

def filter_files_dir (src_dir, file_name= 'Metadata.txt'):
    
    #from posixpath import join
    files_list=[]
    
    for root,dirs,files in os.walk(src_dir):
            
        files_list = [os.path.join(root, names) for names in files if names.endswith(file_name)]

    return sorted(files_list)

# =============================================================================
#                         read_n_sort_metadata
# =============================================================================
    
    # this funtion reads the sample name from the MCP detector  and gives the numbers until experiment change
def read_n_sort_metadata (metadata_list, max_number_exps = 100):
    
    import re
    import pandas as pd
    
        # create two pandas dataframes, one to save the image directories and the other to print a documentation
    Sorted_dataframe = pd.DataFrame({'Empty': np.arange(0,max_number_exps,1)})
    
        # initialize the lists to save the names and values
    list_names_exp =[]
    
    for idx, item in enumerate(metadata_list):
        
            # read the txt file and put it into a dataframe
        data = pd.read_csv(item, sep='\t', names=['metadata'], header = None)
        
            # generate the name of the general experiment
        global_exp_new = re.search("'(.*)'", data['metadata'][42]).group(1)
        
            # if it is the first experiment, take it as part of the first set
        if idx == 0:
            global_exp_old = global_exp_new
            list_names_exp.append( re.search('(.*)_Metadata', item.split('\\',-1)[-1]).group(1))

        # create a new column everytime the global experiment name changes
        if global_exp_old != global_exp_new:

                # Save the file names in a new column in the general dataframe
            Sorted_dataframe[global_exp_old] = pd.Series(list_names_exp)
            list_names_exp =[]       
        
            # add the new values corresponding to the next item in the metadata_list
        list_names_exp.append(re.search('(.*)_Metadata', item.split('\\',-1)[-1]).group(1))
        
            # save the old global name in a variable to compare later again
        global_exp_old = re.search("'(.*)'", data['metadata'][42]).group(1)
        
            # instruction to add the last set of saved values to the datafra,me
        if idx == len(metadata_list)-1:
            Sorted_dataframe[global_exp_old] = pd.Series(list_names_exp)
            
    del Sorted_dataframe['Empty']
    
    return Sorted_dataframe


# =============================================================================
#                       find_matching_files
# =============================================================================
    
# return a list with all the items matching a pattern
    
# Pattern SHOULD be a list 

def find_matching_files(patterns, src_dir):
    
    import glob
    import os
    
    matches = []
    for pattern in patterns:
       search_path = os.path.join(src_dir, '*{}*'.format(pattern))
       for match in glob.iglob(search_path):
          matches.append(match)
    return sorted(matches)





# =============================================================================
#                       copy_sorted_files
# =============================================================================

def copy_sorted_files(src_dir, dst_dir, Sorted_dataframe, action = 'copy'):
    
    import shutil
    import pandas as pd

        # read the dataframe header 
    column_names = list(Sorted_dataframe)
    for j,glob_exp in zip (tqdm(range(len(column_names)), desc = 'Copying...'),column_names):
    #for glob_exp in column_names:
        
            # create the dst folder and add the path
        dst_folder = dst_dir + '/' + glob_exp
        ifnot_create_folder(dst_folder)
        
            # for each header (full exp) create a list with the names containing the wanted pattern
        exp_names = [item for item in Sorted_dataframe[glob_exp].to_list() if not(pd.isnull(item)) == True]

            # extract the files containing the required pattern 
        exp_files = find_matching_files(exp_names, src_dir)
        
        for file in exp_files:
                # if the action is set to move, move the files to the dst folder, else copy the data
            if action == 'move':
                shutil.move(file, dst_folder)
            else:
                shutil.copy2(file, dst_folder)
    return


