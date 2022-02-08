# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 20:36:15 2020

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
#                         prep_stack_dict
# =============================================================================
def prep_stack_dict (base_dir):
    '''
    Creates a dictionary with the folder names as key argument and subfolders as values.
    The values are image addresses that serve as path to read them.

    Parameters
    ----------
    base_dir : str. 
        base directory for retrieving the paths.

    Returns
    -------
    stack_dict : dict
        Dictionary with first folders as key arguments and list of lists as values.

    '''
        # Import libraries
    import os,glob
    from astropy.io import fits
    
        # function to know how deep to search in the folders and subfolders
    depth, str_key = get_depth_path(base_dir)
    
        # take the folder names and sort them 
    folder_names = [item for item in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, item))]
    folder_names.sort()
    
        # initialize the dictionary
    stack_dict={}
    
        # str_key automatically gives you the argument to search the within the folders
    for name in folder_names:
        stack_dict['{}'.format(name)] = [glob.glob(d + '/*.fits') for d in glob.glob(os.path.join(base_dir,name) + str_key)]
        
        # return a dictionary of image addresses
    return stack_dict


# =============================================================================
#                            exec_proc
# =============================================================================
def exec_proc(src_img, seq, start_before=0, start_after=0, stop_before=0, stop_after=0, **kwargs):
    '''
    This function applies a defined sequence of image processing steps to an image.
    Each processing step should be a function accepting an image (2D array) as
        the first parameter, plus an undefined count of named parameters, and returning
        the processed image.
    The processing steps are applied in the order defined in the 'seq' array. It is
        possible to apply only a subset of the sequence by using the parameters 'start_before',
        'start_after', 'stop_before' and 'stop_after'.
        
    This function is normally not called directly from the notebook, but is used
        as a sub-function of other processing steps


    Parameters
    ----------
    src_img : 2D array
        2D array containing the source image
    seq : list
        array with a list of processing functions to be applied
    start_before : int
        indicates that the starting point of the processing sequence is just before this function (this parameter
        has priority over 'start_after')
    start_after : int
        indicates that the starting point of the processing sequence is just after this function
    stop_before :int
        indicates that the ending point of the processing sequence is just before this function (this parameter
        has priority over 'stop_after')
    stop_after : int
        indicates that the ending point of the processing sequence is just after this function
    **kwargs : dict
        collection of additional named parameters

    Returns
    -------
    img : 2D array
        returns the modified image according to the arrange of the given sequence

    '''
        # if 'start_before' or 'start_after' is defined, identify at which step to start,
        # otherwise start at the first step of the sequence
    if start_before != 0:
        first_step = seq.index(start_before)
    elif start_after != 0:
        first_step = seq.index(start_after) + 1
    else:
        first_step = 0
        
        # if 'stop_before' or 'stop_after' is defined, identify at which step to stop,
        # otherwise stop at the last step of the sequence
    if stop_before != 0:
        last_step = seq.index(stop_before)
    elif stop_after != 0:
        last_step = seq.index(stop_after) + 1
    else:
        last_step = len(seq)
        
        # extract the selected subset of the processing sequence
    proc_seq = seq[first_step:last_step]
    
        # create an copy of the source image
    img = src_img.copy()
    
        # apply succesively all processing steps to the image
    for step in proc_seq:
        img = step(img, **kwargs)
        # return the processed image
    return img


# =============================================================================
#                         stack_avg
# =============================================================================
def stack_avg(base_folder):
    '''
    Base function to average image stacks after their overlap correction.
    This function is meant to be used with 'exec_averaging' as one of the sequence functions

    Parameters
    ----------
    base_folder : list of arrays
        can be a list of images or a list of directories with their individual address.

    Returns
    -------
    base folder with all the images averaged

    '''
        # import libraries
    import numpy as np
    from astropy.io import fits
    
        # takes the amount of images in the last folder
    nimg = len(base_folder[0])
    
        # Takes the amount of folders to be avergaed
    zimg = len(base_folder)
        # Counts the total number of images in a folder
    for i in range(nimg):
        
            # Counts the tolat number of folders to find the same x-position image
        for j in range (zimg):
            
                # read each image if the folder given was a list of images instead of folders.
            if type(base_folder[j][i]) == np.ndarray:
                img = base_folder[j][i]
                
            else:
                    # read the file
                # with fits.open(base_folder[j][i]) as f:
                #     img = (f[0].data)
                img = get_img (base_folder[j][i])
                # if it is the first image
            if j == 0:
                img_tot = img.astype(float)
            else:
                img_tot = img_tot + img
                
            # Save the image in the same variable, locally.
        base_folder.append( img_tot / zimg)
        
        # Eliminates the folders with the image directories to keep just the images
    del (base_folder[:zimg])
    return;


# =============================================================================
#                             keep_key
# =============================================================================
def keep_key (dictionary, list_keep = '' ):
    '''
    Given a list of strings (key names), keeps the key and its values in a dictionary

    Parameters
    ----------
    dictionary : TYPE
         target dictionary to keep the parameters
    list_keep : list of str
        list of keys strings that will remain in the dictionary

    Returns
    -------
    None.

    '''
        # search the names of a given list of keys, then it keep them and remove the others
    for key in dictionary.copy():
        
            # keep the key
        if key not in list_keep:
            dictionary.pop(key, None)
    return 


# =============================================================================
#                            remove_key
# =============================================================================
def remove_key (dictionary, list_remove = '' ):
    '''
    Given a list of strings (key names), removes the key and its values in a dictionary

    Parameters
    ----------
    dictionary : dict
        target dictionary to remove the parameters
    list_remove : list of str
        list of keys strings that will be removed from the dictionary

    Returns
    -------
    None.

    '''
        # search the names of a given list of keys, then it removes them and keep the others
    for key in dictionary.copy():
        
            # remove the key
        if key in list_remove:
            dictionary.pop(key, None)
    return


# =============================================================================
#                         get_depth_path
# =============================================================================
def get_depth_path(path, depth=0):
    '''
    Gives the depth of the given folder (number of subdirectories).
    This function is used in 'prep_stack_dict' to construct automatically the dictionary with the given directory

    Parameters
    ----------
    path : str
        base directory
    depth : int
        counter for the directory depth.

    Returns
    -------
    depth: int
        integer depth value for the base folder 
    sub_dirs: str
        key for searching in all the folders and subfolders used in 'prep_stack_dict'

    '''
        #import libraries
    import os
    
        # Search in the directory 
    for root, dirs, files in os.walk(path):
        
            # if the next folder is a subfolder,add to the sum
        if dirs:
            depth+=1
            
                # check for next subfolders until there are no more
            return max(get_depth_path(os.path.join(root, d), depth) for d in dirs)
        # initialize variable
    sub_dirs = []
    
        # depending on the depth calculated before, construct the automatic searching parameter
    for i in range(depth):
        
            # if there is no more subdirectories, append the final contruct
        if i == 0 :
            sub_dirs.append('/')
            
            # for each level or subdirectory, append the search key for all of them
        else:
            sub_dirs.append('*/')
    
        # returns an integer value for the depth and the a string key to be inserted in prep_stack_dict
    return  depth, "".join(sub_dirs)


# =============================================================================
#                         exec_averaging
# =============================================================================
def exec_averaging (stack_dict, seq = [stack_avg], count_time = False, **kwargs):
    '''
    Averages image stacks after the overlap correction
    A sequence is required, this sequence can contain any function for image enhancement (filtering, outlier removal, etc.).
    The function 'stack_avg' does the averaging, thus it is required to be in any position inide the sequence

    Parameters
    ----------
    stack_dict : dict
        dictionary with all the image addresses to average. This can be obtained from 'prep_stack_dict'
    seq : list of functions
        list of image enhancement functions for their processing in order.
    **kwargs : extra arguments. dict
        extra arguments carried on for future processings

    Returns
    -------
    None. Keeps the list with the averaged images to continue the processing steps

    '''
        # import libraries
    import time
    from tqdm import tqdm
        
        # starts a timer in acse you want to know the process total time
    start = time.time()
    
        #read the dictionary's keys and values
    for key,values in stack_dict.items():

        new_values = []
        for i, val in enumerate (values):
            new_values.append(slice_list(val,**kwargs))
        
        stack_dict[key] = new_values
            
            # read the first subfolder ans start a visual processing time
        for j,folder in zip (tqdm(range(len(new_values)), desc = 'Averaging'),new_values):
            
            if kwargs.get ('binning') == True:
                row_values = []
                folder = binning_proc(folder, **kwargs)
                for i,img in enumerate(folder):
                    row_values.append( exec_proc(img,seq,stop_before = stack_avg,**kwargs))
                new_values [j] = row_values
            else:
                    # read each subfolder in the subfolder
                for i,img_dir in enumerate(folder):
                        # read the image just if it was not read before by binning_proc function
                    img = get_img(img_dir)
                        # replace the image path with the image data 
                    new_values[j][i] = exec_proc(img,seq,stop_before = stack_avg,**kwargs)
                    
            # performs the averaging
        stack_avg(new_values)
        
            # If there is any function performed after the averaging , this part does it
        for n,img in enumerate(new_values):
            new_values[n] = exec_proc(img,seq,start_after = stack_avg,**kwargs)
    
            #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/after_averaging/' +key+ '/'+key+'_averaged_' + str(n).zfill(5) +  '.fits', new_values[n])
        # finish the timer
    end = time.time()
    
        # in case that precise information about the time is required, an option to print the total required time is available
    if count_time:
        print('Total time: %ds' %(end - start))
        




# =============================================================================
#                          create_SBKG_Wmask
# =============================================================================
def SBKG_Wmask (base_img, BB_mask_img, **kwargs):
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
    from skimage.measure import label
    from scipy.interpolate import Rbf

        # extract the image shape
    s_row = base_img.shape[0]
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
        ivals [i] = np.nanmean(base_img[reg])

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




# =============================================================================
#                         weighting_func
# =============================================================================


def find_nearest_lower_value(key, sorted_li):
    if key <= sorted(sorted_li)[0]:
        value = sorted(sorted_li)[0]
    else:
        value = max(i for i in sorted_li if i <= key)
    return value

def find_nearest_upper_value(key, sorted_li):
    if key >= sorted(sorted_li)[-1]:
        value = sorted(sorted_li)[-1]
    else:
        value = min(i for i in sorted_li if i >= key)
    return value



def weighting_func (src_dir, **kwargs):
    '''
    This function manages the timestamps.txt file created in the overlap correction step.

    Parameters
    ----------
    scr_dir : str
        Takes the same source directory from the process. it is where the timestamps.txt is

    Returns
    -------
    Weights : DataFrame
        DataFrame with the weightings for each folder

    '''
        # import libraries
    import pandas as pd
    import warnings
    
        # disable warnings because of possible bad characters in the header
    warnings.filterwarnings("ignore")
        
        # Reads the timestamps.txt file created in the overlap correction step and order the indexes according to the timestamps
    Timestamps = pd.read_csv(src_dir + '/timestamps.txt')
    Timestamps = Timestamps.sort_values(by=['Modification (s)'], ascending=True)
    Timestamps = Timestamps.reset_index(drop=True)
    
        # copies the DataFrame and mask it to contain just the OBs
    OB_df = Timestamps[Timestamps['Folder'].str.contains('ob',case=False)]
    
        # copies the DataFrame and mask it to contain everything BUT the OBs
    Weights = Timestamps[~Timestamps['Folder'].str.contains('ob', case=False)]
    
        # takes the TRUE indices for both OBs and no OBs
    idx_ob = OB_df['Folder'].index
    idx_w = Weights['Folder'].index
    
        #convert the OB data frame into a list for easier search
    OB_list = OB_df['Modification (s)'].to_list()
    
    for i in range (len(idx_w)):
        
            # find the lower and upper OBs from an experiment
        lower_OB = find_nearest_lower_value(Weights['Modification (s)'][idx_w[i]], OB_list)
        upper_OB = find_nearest_upper_value(Weights['Modification (s)'][idx_w[i]], OB_list)
        
            # if the value for OB found is the same as the one in the experiment, then there is no OB and sets the OB registration to 1
        if lower_OB == upper_OB:
            
            Weights.loc[idx_w[i],'w1'] = 1
            Weights.loc[idx_w[i],'w2'] = 0

            Weights.loc[idx_w[i],'OB1'] = OB_df['Folder'][idx_ob[-1]]
            Weights.loc[idx_w[i],'OB2'] = OB_df['Folder'][idx_ob[-1]]
            
                # if there is OBs before and after, then takes the timestamps to create the anti-scrubbing weights
        else:
            
            Weights.loc[idx_w[i],'w1'] = (upper_OB - Weights['Modification (s)'][idx_w[i]]) / (upper_OB - lower_OB)
            Weights.loc[idx_w[i],'w2'] = (Weights['Modification (s)'][idx_w[i]] - lower_OB) / (upper_OB - lower_OB)
            
            Weights.loc[idx_w[i],'OB1'] = Timestamps['Folder'][OB_df.index[OB_df['Modification (s)'] == lower_OB].values[0]]
            Weights.loc[idx_w[i],'OB2'] = Timestamps['Folder'][OB_df.index[OB_df['Modification (s)'] == upper_OB].values[0]]
        
        # remove unnecessary columns in the target DataFrame
    Weights.drop(Weights.iloc[:, 1:5], inplace=True, axis=1)
    
    return Weights



# =============================================================================
#                         conv_seconds
# =============================================================================
def conv_seconds (time_in_seconds):
    '''
    The function returns an understandable enlapsed time between 2 events.
    

    Parameters
    ----------
    time : float/int
        time IN SECONDS to convert in hours, minutes, seconds

    Returns
    -------
    result : str
        enlapsed time in hours, minutes and secods.

    '''
        # separate hours and minutes 
    hour, minute = divmod(time_in_seconds/60/60, 1)
    
        # calculate the minutes with the remaining decimals
    minute *= 60
    
        # separate the minutes and the seconds
    _, second = divmod(minute, 1)
    
        # calculate the seconds with the remaining decimals
    second *= 60
    
        # print the result
    result = '{} hours {} minutes {} seconds'.format(int(hour), int(minute), round(second))
    return result


# =============================================================================
#                         slice_list
# =============================================================================
def slice_list (list, start_slice = 0, end_slice = '', **kwargs):
    '''
    Slices a list of values to given parameters

    Parameters
    ----------
    list : list
        list of values
    start_slice : int
        starting slice parameter. The default is 0.
    end_slice : int
        ending slice parameter. The default is ''.

    Returns
    -------
    list
        sliced list.

    '''
        # if there is no specification for end_slice, it takes the whole list of values
    if end_slice  == '':
        end_slice = len(list)
        
    return list[start_slice:end_slice]






# =============================================================================
#                         get_idx_DF
# =============================================================================
def get_idx_DF (DataFrame, value):
    '''
    Search a specific value in a DataFrame and returns the true indices where the values is found

    Parameters
    ----------
    DataFrame : pandas DataFrame
        base DataFrame 
    value : str, int, float
        target value

    Returns
    -------
    list_pos : TYPE
        DESCRIPTION.

    '''
    list_pos = list()
    
        # Get bool dataframe with True at positions where the given value exists
    result = DataFrame.isin([value])
    
        # Get list of columns that contains the value
    series_obj = result.any()
    col_names = list(series_obj[series_obj == True].index)
    
        # Iterate over list of columns and fetch the rows indexes where value exists
    for col in col_names:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            list_pos.append((row, col))
            
        # Return a list of tuples indicating the positions of value in the DataFrame
    return list_pos




# =============================================================================
#                        slice_stack_dict
# =============================================================================
def slice_stack_dict (dictionary, dict_key = [], start_slice = 0, end_slice = '', **kwargs):
    '''
    slices specific dictionary values (lists) for given keys

    Parameters
    ----------
    dictionary : dictionary 
        {key:value}
    dict_key : str, list of strs
        all keys that require to be sliced. if left empty, it will loop all the keys in the dictionay  The default is [].
    start_slice : int
        slice starting boundary. The default is 0.
    end_slice : int
        slice end boundary. The default is ''.

    Returns
    -------
    sliced dictionary values

    '''
        # if there is no specification for end_slice, it takes the whole list of values
    if end_slice  == '':
            
                # len of a list in the list (dictionary) takes the total lenght of the first key
            end_slice = len(list(list(stack_dir.items())[0])[1])
            
        #check whether specific keys were given, if not it performs the operation to all keys in the dictionary
    if dict_key == []:
        for key in dictionary.keys():
            
                # given a dictionary and a key, the slicing goes the same as in a list
            dictionary[key] = dictionary[key][start_slice : end_slice]
            
        # If specific keys were give, it applies the operation just to those keys
    else:
            
        for key in dict_key:

                # given a dictionary and a key, the slicing goes the same as in a list
            dictionary[key] = dictionary[key][start_slice : end_slice]
    return




# =============================================================================
#                        keep_key_weights
# =============================================================================
def keep_key_weights (dictionary, weights_DataFrame, keep_folder = []):
    '''
    keep_folder is a list of string values that correspond to the folders where the samples are.
    given the keep_folder list, there are specific OBs needed for the scrubbing correction of each experiment.
    This function mantains the target experiment and keep at the same time the OBs required for future processes

    Parameters
    ----------
    dictionary : dictionary
        {key:value}
    weights_DataFrame : pandas DataFrame 
        DataFrame containing experiment's OB weights 
    keep_folder : list of strings
        target keys/folders to keep in the process. The default is [].

    Returns
    -------
    the dictionary with just the target (wanted) folders/ keys

    '''
        # to add non-existent value in a list is easier to do it with a set
    ob_list = set([])
    
        # if specific folders were given, work with those
    if keep_folder!= []:
        
            # for each folder in the list
        for folder in keep_folder:
            
                # get the index of the folder name
            index = get_idx_DF (weights_DataFrame, folder)
            
                # with the index, extract the OB's names and add them into the set
            ob_list.add(weights_DataFrame.loc[index[0][0]][3])
            ob_list.add(weights_DataFrame.loc[index[0][0]][4])
    
            # transform the set into a list
        ob_list = list(ob_list)
    
            # keeps the values 
        keep_key (dictionary, keep_folder+ob_list)
    return



# =============================================================================
#                        scrubbing_corr
# =============================================================================
def scrubbing_corr (src_img, ob_01_img, ob_02_img, weight_01 = 0.5, weight_02 = 0.5, **kwargs):
    '''
    Does the scrubbing correction with the calculated OBs weighting 

    Parameters
    ----------
    src_img : 2D numpy array
        target image
    ob_01_img : 2D numpy array
        fisrt OB to be taken into account
    ob_02_img : 2D numpy array
        second OB to be taken into account
    weight_01 : int
        weight for first OB. The default is 0.5.
    weight_02 : int
        weight for second OB. The default is 0.5.

    Returns
    -------
    2D numpy array
        image corrected

    '''
        # divide the image by its weightened OBs
    return src_img/(ob_01_img * weight_01 + ob_02_img * weight_02)




# =============================================================================
#                        stack_scrubbing_corr
# =============================================================================
def stack_scrubbing_corr (dictionary, weights_DataFrame, **kwargs):
    '''
    Does the scrubbing correction with the calculated OBs weighting for a given dictionary of target folders

    Parameters
    ----------
    dictionary : dictionary
        {key:value}
    weights_DataFrame : pandas DataFrame 
        DataFrame containing experiment's OB weights 

    Returns
    -------
    new_dict : TYPE
        DESCRIPTION.

    '''
        # initialize the new dictionary
    new_dict = {}
    for key, value in dictionary.items():
            # search for the key folder within the DataFrame values
        if key in weights_DataFrame.Folder.values:
            
                # search and return the row that contain the key
            row_data = weights_DataFrame[weights_DataFrame['Folder'].str.contains(key)]
            
                # extract the values for each column, they contain the information required
            weight_01 = row_data['w1'].values[0]
            weight_02 = row_data['w2'].values[0]
            ob_01_img = np.nanmean(dictionary[row_data['OB1'].values[0]], axis = 0)
            ob_02_img = np.nanmean(dictionary[row_data['OB2'].values[0]], axis = 0)
            
                # with the extracted values, operate to obrain the scrubbing correction
            for i, src_img in enumerate(value):
                value[i] = scrubbing_corr (src_img, ob_01_img, ob_02_img, weight_01,weight_02, **kwargs)
                #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/img_after_scrubbing/' +key+ '/'+key+'_scrubbed_' + str(i).zfill(5) +  '.fits', value[i])
                # update the values and append them to the new dictionaries for the current key
            new_dict[key] = value
            
        # gives the new (corrected) dictionary
    return new_dict



# =============================================================================
#                        TFC_corr
# =============================================================================
def TFC_corr(src_img, ref_img, nca=[0,0,0,0], **kwargs):

    if sum(nca) == 0:
        TFC = 1.0
    else:
        TFC = np.average(crop_img(ref_img, nca)) / np.average(crop_img(src_img, nca))
    return TFC



# =============================================================================
#                    stack_transmission_img
# =============================================================================
def stack_transmission_img (dictionary, ref_mask_img, img_mask_img, ref_key = '', print_TFC = False, **kwargs):
    '''
    Computes the transmission images for the keys contained in a dictionary, with respect to a reference  (dry image)
    functional kwargs:
        nca = [x,y,w,h]. a list of int values to extrat the non-changing area, the default is [0,0,0,0]
    
    Parameters
    ----------
    dictionary : dictionary
        {key:value}
    ref_mask_img : numpy 2D array
        mask to construct the sbkg image 
    ref_key : str
        string of the key/folder name containing the reference images (ref_bb, dry, dry_bb, etc)
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
        # import libraries
    from tqdm import tqdm
    
        # loops through the dictionary keys and values
    for key,value in dictionary.items():
            # this takes care if the BB mask for the ref image is different as for the stacks
        if key == ref_key:
            for i, ref in zip (tqdm(range(len(value)), desc = 'Processing'),value):
                sbkg_ref = SBKG_Wmask (ref, ref_mask_img, **kwargs)
                value[i] = ref - sbkg_ref
                test = ref - sbkg_ref
                #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/sbkg_ref/' + 'SBKG_' + str(i).zfill(5) +  '.fits', sbkg_ref)
                #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/ref_min_sbkg/' + 'ref_min_sbkg_' + str(i).zfill(5) +  '.fits', test)
        else:
                # creates the sbkg image and substract it for 
            for i, img in zip (tqdm(range(len(value)), desc = 'Processing'),value):
                sbkg_img = SBKG_Wmask (img, img_mask_img, **kwargs)
                value[i] = img - sbkg_img
                test = img - sbkg_img
                #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/sbkg_img/' + 'SBKG_' + str(i).zfill(5) +  '.fits', sbkg_img)
                #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/img_min_sbkg/' + 'img_min_sbkg_' + str(i).zfill(5) +  '.fits', test)

        # identify the reference values in the dictionary
    ref_dict = dictionary[''.join(ref_key)]
    
        # removes the reference values
    remove_key(dictionary, ref_key)
    
        #the remaining folders are the target folders
    for key,value in dictionary.items():
        
        for i, src_img, ref_img in zip (np.arange(0, len(value)),value, ref_dict):
            
            #sbkg_ref = SBKG_Wmask (ref_img, ref_mask_img, **kwargs)
            #ref_corr_img = ref_img - sbkg_ref
                # intensity correction 
            TFC = TFC_corr(src_img, ref_img, **kwargs)
            if  print_TFC:
                print(TFC)
            value [i] = (src_img/ref_img)*TFC
            #fits.writeto('H:/700 Campaigns - internal/780 2020/PSI20_01NI/temp/IP_Yvette_02/Full_Transmission_results_4/img_before_TFC/' + 'image_before_TFC_' + str(i).zfill(5) +  '.fits', value[i])
    return 



# =============================================================================
#                        save_dict
# =============================================================================
def save_dict (dictionary, dst_dir, start_slice = '', testing_mode = False, **kwargs):
    '''
    takes a dictionay and saves each key and its values in a target directory
    possible kwargs:
        start_slice : int
            if the dictionary was sliced, starts the numbe index with that reference
    Parameters
    ----------
    dictionary : dictionary
        {key:value}
    dst_dir : str
        target directory to save the images
    testing_mode : boolean
        instead of saving the image, it shows is

    Returns
    -------
    None.

    '''

        # takes every key and its values in the dictionary
    for key, values in dictionary.items():
        
            # if a sliceing was registered, gives the index to that value
        if start_slice != '':
            index = start_slice
        else:
                index = 0
                
            # takes each value in the subfolder (values)
        for img in values:
            
            if testing_mode:
                
                plt.figure(figsize=[15,10])
                show_img(prep_img_for_display(img), title = 'Transmission image')
                plt.hist(img.ravel(), bins=256,range=(0, 1));

            else:
                    # creates the saving path 
                file_name = os.path.join(dst_dir, key, key + '_' + format(index, '05d') + '.fits')
            
                    # utilize pyerre's function to save each image in the folder
                write_img(np.flip(img, 0), file_name, base_dir='', overwrite=False)
            
            index+=1
        
    return




# =============================================================================
#                        extract_kwargs
# =============================================================================
def extract_kwargs(**kwargs):
    '''
    Receive and start the **kwargs entered in a fucntion so the nested functions can read them

    Parameters
    ----------
    **kwargs : TYPE
        kwargs and initilaized varaibles that the functions do not require

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    new_kwargs = kwargs.copy()
        # create a dictionary out of the kwargs
    kwargs_dict = dict(**kwargs)
    
        # make 2 tuple with the key and values (position dependent)
    keys,values = zip (*kwargs_dict.items())
    
        # initialize the list
    list = []
    
        # retrive the data in a format 'x=1'
    for i in range (len(keys)):
        list.append( keys[i] + ' = ' + str(values[i]))
        
    #new_kwargs = (','.join(list))
        # print the results
    return print('\n'.join(list))
    #return new_kwargs
    # return new_kwargs





# =============================================================================
#                        stack_batch_proc
# =============================================================================
def stack_batch_proc (src_dir, dst_dir, ref_mask_img, img_mask_img, proc_folder = [], ref_folder = '15_ref_bb', avg_seq = [stack_avg], save_results = True, **kwargs):
    '''
    This function applies a normal image processing for a ToF stack images folder and subfolders

    possible kwargs: 
        nca = [x,y,w,h]. list of integers
            defines the non-changing area in the image to perform the intensity correction
        roi_x = [x,y,w,h]. list of integers
            where x is a number starting from 1. extra regions of interest selected to extract the tranmission values. 
        start_slice = int
            first value for the slicing
        end_slice = int
            last value for the slicing
        
    Parameters
    ----------
    src_dir : str
        source directory
    dst_dir : str
        destination directory
    proc_folder : list of strings
        folders/ keys taht are to be processed. If the list is empty, it processes all the key/folders in the dictionary. The default is [].
    ref_folder : TYPE, optional
        DESCRIPTION. The default is '15_ref_bb'.
    avg_seq : TYPE, optional
        DESCRIPTION. The default is [stack_avg].
    save_results : boolean
        True if the results are meant to be saved in the destination folder. The default is True.

    Returns
    -------
    None.

    '''
        # import libraries
    import time
        #start the counting of real exp time
    start = time.time()
    
    extract_kwargs(**kwargs)
    
        # constructs the diactionary
    stack_dict = prep_stack_dict(src_dir)
    weights = weighting_func (src_dir)
    
        # for the case where no specific folders are given, works with all folders
    if proc_folder == []:
        
            # unpacksthe key arguments and transform them into a list
        proc_folder = [*stack_dict]
        
        # constructs the list of targeted folders and reference
    keep_folder = proc_folder + list([ref_folder])
    
        # keeps the target folders and reference in the dictionary, it removes other keys and values
    keep_key_weights (stack_dict, weights, keep_folder)
    #keep_key(stack_dict, keep_folder)
        # the main point of the averaging sequence is to average the overlap corrected images, if it was forgotten, then adds it to the list
    if stack_avg not in avg_seq:
        avg_seq.append(stack_avg)
    
        # executes the averaging process
    exec_averaging(stack_dict, avg_seq, **kwargs )

        # corrects for MCP's scrubbing effect
    stack_dict = stack_scrubbing_corr (stack_dict, weights, **kwargs)

        # computes the transmission images
    stack_transmission_img (stack_dict, ref_mask_img, img_mask_img, ref_key = ref_folder, **kwargs)
    
        # saves the results 
    if save_results:
        save_dict(stack_dict, dst_dir, **kwargs)
        #save_stack_energies (stack_dict, dst_dir, HE_slice = [5, 19], LE_slice = [36,50], **kwargs)
    
    else:
        end = time.time()
        print('Total execution time: %ds' % (end-start))
        return stack_dict
    
        # finish the timer
    end = time.time()
    print('Total execution time: %ds' % (end-start))

    return 




# =============================================================================
#                        test_filtering_stack
# =============================================================================
def test_filtering_stack (src_dir,dst_dir, test_folder = '', test_seq = [stack_avg], img_number = '', give_data = False, **kwargs):
    '''
    This function test the filtering and averaging sequences.
    The test_seq list contains all the filters that want to be tested.
    if a filter function requires a parameter, it can be given in the kwargs.
    The result is the plot og the image with the filters

    Parameters
    ----------
    src_dir : str
        source directory
    dst_dir : str
        destination directory
    test_folder : str
        name of the folder/key to be tested
    test_seq : list of functions
        list contains all the filters that want to be tested. The default is [stack_avg].
    img_number : int
        target image number , if left empty, then it takes the medain item in the folder
    dive_data : boolean
        retrieves the image data (2D numpy array) and saves it into a variable. This is useful when you want to operate with the function's  result

    Returns
    -------
    None.

    '''
        # import lbraries
    from img_utils_mod import prep_img_for_display
    from img_utils_mod import show_img
    
    extract_kwargs(**kwargs)
        # constructs the diactionary
    stack_dict = prep_stack_dict(src_dir)
        
        # constructs the list of targeted folders and reference
    keep_folder = list([test_folder])

        # keeps the target folder in the dictionary, it removes other keys and values
    keep_key (stack_dict, keep_folder)
    
        # to take the image target, check if a value for the image number was given
    if img_number != '' or img_number in locals():
        start_slice = img_number
        
        #if no specific image number was given, takes the median number
    else:
            #if the name given is a string, proceed directly to calculate the median value
        if type(test_folder) == str:
            start_slice = int(len(stack_dict[test_folder][0])/2)
            
            # if it was a list, calculate the median value of the first item in the list
        else:
            start_slice = int(len(stack_dict[test_folder[0]][0])/2)
    
        # to take just one image, the ending slice requires to be one more unit
    end_slice = start_slice+1
    
        # the main point of the averaging sequence is to average the overlap corrected images, if it was forgotten, then adds it to the list
    if stack_avg not in test_seq:
        test_seq.append(stack_avg)
    
        # executes the averaging process
    exec_averaging(stack_dict, test_seq, start_slice = start_slice, end_slice = end_slice, **kwargs)
    
    for key,values in stack_dict.items():
        img = values[0]
    
    show_img(prep_img_for_display(img), title = 'Filtered and Averaged Image #'+str(start_slice))
    
    if give_data:
        return img



# =============================================================================
#                        save_stack_energies
# =============================================================================
def save_stack_energies (stack_dict, dst_dir, HE_slice = [5, 19], LE_slice = [36,50],start_slice = '', testing_mode = False, **kwargs):
    
    
        # if a sliceing was registered, gives the index to that value
    if start_slice != '':
        index = start_slice
    else:
        index = 0
    
        # takes every key and its values in the dictionary
    for key, values in stack_dict.items():
        
        HE_avg = np.nanmean(values[HE_slice[0]:HE_slice[1]], axis = 0)
        LE_avg = np.nanmean(values[LE_slice[0]:LE_slice[1]], axis = 0)
            # takes each value in the subfolder (values)
        
        if testing_mode:
            show_img(prep_img_for_display(HE_avg), title = 'Transmission image HE')
            show_img(prep_img_for_display(LE_avg), title = 'Transmission image LE')
            
        else:
                # creates the saving path 
            file_name_HE = os.path.join(dst_dir, 'HE_stack', key + '_HE.fits')#'_HE_' + format(index, '05d') + '.fits')
            file_name_LE = os.path.join(dst_dir, 'LE_stack', key + '_LE.fits')#'_LE_' + format(index, '05d') + '.fits')
        
                # utilize pyerre's function to save each image in the folder
            write_img(np.flip(HE_avg, 0), file_name_HE, base_dir='', overwrite=False)
            write_img(np.flip(LE_avg, 0), file_name_LE, base_dir='', overwrite=False)
        
        index+=1
    
    return



# =============================================================================
#                           roi_in_image
# =============================================================================
def roi_in_image(im,sel_roi,show=False,titleOne='Original image with selected ROI',titleTwo='ROI',shape=False):
    """
    Takes the x, y, height and width parameters of an image (taken from extract_roi) and crops it (from NIAGs).

    Parameters
    ----------
    im : 2D array
        image to crop.
    sel_roi[0] : Integer
        value in the x axis.
    sel_roi[1] : Integer
        value in the y axis.
    sel_roi[2] : Integer
        width of the ROI.
    sel_roi[3] : Integer
        width of the ROI.
    show : Boolean, optional
        True if you would like to plot the cropped area. The default is False.
    titleOne : String, optional
        Title of the window with the selected ROI. The default is 'Original image with selected ROI'.
    titleTwo : String, optional
        Title of the window with the cropped ROI. The default is 'ROI'.
    shape : Boolean, optional
        If you want to get the image shape. The default is False.

    Returns
    -------
    None.

    """

    import matplotlib.patches as patches
    from matplotlib import gridspec
    import numpy as np
    
    im[np.isnan(im)] = 0
    upper_thres = 5
    lower_thres = -5
    im[im>upper_thres] = 0
    im[im<lower_thres] = 0
    
    if shape:
        print(im.shape)    
        
#    if np.size(sel_roi[0]) > 1 :
#        print ('\nThe option to show more than 1 ROI is not available for the moment, please give max. 1 ROI \n')   
   
    else:       
        
        if (0<=sel_roi[0]<=im.shape[1] and 0<=sel_roi[0]+sel_roi[2]<=im.shape[1] and 0<=sel_roi[1]<=im.shape[0] and 0<=sel_roi[1]+sel_roi[3]<=im.shape[0]):
            imROI = im[sel_roi[1]:sel_roi[1]+sel_roi[3],sel_roi[0]:sel_roi[0]+sel_roi[2]]
            if show:
                #vmin,vmax=np.nanmin(im),np.nanmax(im)
                vmin,vmax=np.nanmean(im)-2*np.nanstd(im),np.nanmean(im)+2*np.nanstd(im)
                fig = plt.figure(figsize=(15,10))
                gs = gridspec.GridSpec(1, 2,width_ratios=[1,1],height_ratios=[1])
                gs = gridspec.GridSpec(1,2, width_ratios = [1,1], height_ratios = [1])
                ax = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                ax.imshow(im,vmin=vmin, vmax=vmax,interpolation='nearest',cmap='gray')
                rectNorm = patches.Rectangle((sel_roi[0],sel_roi[1]),sel_roi[2],sel_roi[3],linewidth=1,edgecolor='b',facecolor='none')
                ax.add_patch(rectNorm)
                ax.set_title(titleOne)
                ax2.imshow(im,vmin=vmin, vmax=vmax,interpolation='nearest',cmap='gray')
                ax2.set_title(titleTwo)
                ax2.set_xlim([sel_roi[0],sel_roi[0]+sel_roi[2]])
                ax2.set_ylim([sel_roi[1],sel_roi[1]+sel_roi[3]])
                plt.tight_layout()
                plt.show()    

            return imROI;
        else:
            print('!!!WARNING!!! \nROI out of range')
            
            


# =============================================================================
#                           binning_proc
# =============================================================================

def binning_proc (folder, binning_factor, **kwargs):
        
        #create a new folder for image results
    new_folder = []
    
        # create the array to match with the desired indices
    list_idx = np.arange(0, len(folder), binning_factor)
    
        # if the binning number desired does not match with the total of images required. it will not process the first images (leave them out)
        # this step is prioritizing the 'good' images with enough neutrons (not usual for first images in NI-ToF)
    if len(folder)%binning_factor != 0:
        
            #move the indices a "binning factor amount" and delete the last item, this will agree with the general indices by leaving out the first arrays/images/values
        list_idx = list_idx + (len(folder)%binning_factor)
        list_idx = np.delete(list_idx,-1)
        
        # start the binning process
    for value in list_idx:
        arr = 0
        for idx in range (value, value + binning_factor):
                #read the image directory and get the array
            img  = get_img(folder [idx])
            
                #sum arrays
            arr = arr + img
            
            # after summing the arrays, divide by the binning fatcor to get the average
        avg_img = arr/binning_factor
        
            # write the results in the new folder
        new_folder.append(avg_img)
    return new_folder

