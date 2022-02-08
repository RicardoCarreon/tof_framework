# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:03:49 2021

@author: carreon_r
"""

import os, glob, time
import numpy as np
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt

from img_utils_4_transmission import *



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
#                         avg_transmission_img
# =============================================================================
def avg_transmission_img(dictionary, proc_folder, **kwargs):
    
    
    extract_kwargs(**kwargs)
        #keep the desired folders in the process
    keep_key (dictionary, proc_folder)
    
        #sum up the images to 
    exec_averaging(dictionary, seq = [stack_avg], **kwargs)
    
        #remove undesire values from the images in the dictionary
    new_imgs = []
    for key,values in dictionary.items():
        
        for i, img in enumerate (values):
                #filter transmission values (between 0 and 1). Outliers (usually from BBs) are set to 1 or 0
            img[np.isnan(img)] = 0
            img[img > 1.05] = 1
            img[img < 0] = 0
        new_imgs.append(np.mean(values, axis = 0))
    return new_imgs



# =============================================================================
#                         select_multiple_rois
# =============================================================================
def select_multiple_rois(var_name, img, just_rois=False, cmap = 'gray', **kwargs):
    '''
    This function provides an interactive interface for selecting a rectangular region of interest in an image.
    This function is meant to be used with the `%load`  magic in jupyter.
    
    The difference with 'select_rois' is that this function assign all the ROIs selected to only one variable name

    Parameters
    ----------
    var_name : str
        strs or list with the names assigned that want to be assigned to the ROIs
    src_img : str
        image directory and name
    img : 2D numpy array
        source image
    base_dir : str, optional
        directory relative to which the 'src_dir' variable is specified (absolute path is used if omitted) 

    Returns
    -------
    variables
        Variable name = [[x,y,w,h], [x2,y2,w2,h2],...]

    '''
            # import necessary packages
    from img_utils_4_transmission import prep_img_for_display, img_roi_selector
    
        # if one variable is given ie. string, it convert it into a list to enter the names and ROIs loop
    if type(var_name) == list:
        
            # generate the window title
        title = 'Please select following ROIs: ' + ','.join(var_name)
        
            # perform the interactive rectangular ROI selection
        roi_def = img_roi_selector(img, title=title, 
                multiple=True, roi_names=var_name, cmap=cmap)
        
            # if an empty ROI is returned, this means the input was canceled
        if roi_def == None:
            raise ValueError('Interactive input canceled by the user')
        # when you want to extract just the rois info
        if just_rois:
            return roi_def
            # generate a statement to be inserted in the jupyter notebook
        return ( 'ROIs_'+ var_name + ' = ' + str(roi_def))
    
    else:
        
            # generate the window title
        title = 'Please define one or more ROIs for ' + var_name
        
            # perform the interactive rectangular ROI selection
        roi_def = img_roi_selector(img, title=title, 
                                   multiple=True, cmap=cmap)
        
            # if an empty ROI is returned, this means the input was canceled
        if roi_def == None:
            raise ValueError('Interactive input canceled by the user')
        
            # when you want to extract just the rois info
        if just_rois:
            return roi_def
        
            # return the list of values in a legible format for %load magic
        return ( 'ROIs_'+ var_name + ' = ' + str(roi_def))
    


# =============================================================================
#                              get_transmission_values
# =============================================================================
# this function creates an excel sheet of transmission values by taking ROI values

def get_transmission_values (dictionary, rois_dictionary, spectra_file, src_dir='', binning = 1, flight_path = 1, start_slice = '', end_slice='', name_xlsx = 'Transmission_values.xlsx', save_results = False, **kwargs):
    import numpy as np
    import pandas as pd
    
    Values = pd.DataFrame()
    table_wvl = pd.DataFrame()
    for roi_name,rois in rois_dictionary.items():
            # get the real name of the folder connected with the regions of interest
        roi_name  = roi_name.split('ROIs_')[1]
        
            # prepare the dictionary and the table to include the results
        folder_images = dictionary.get(roi_name)
        results_vec = np.zeros([len(folder_images), len(rois)])
        
        for ndx, img in enumerate (folder_images):
            
            for idx, roi_val in enumerate(rois):
                
                results_vec[ndx,idx] = np.average(crop_img(img, roi_val))
        
        for k in range(results_vec.shape[1]):
            Values['raw_trans_' + roi_name + '_roi_' + str(k+1)] = results_vec[:,k]
            
        # Reads the spectra, creates the table and calculate the equivalent wavelengths in Angstroms
    table_spectra = pd.read_csv(spectra_file,header=None, usecols=[0], sep='\t', names=['Arrival Time'])
    table_wvl["Wavelength [Å]"] = (3956.034/(flight_path/table_spectra["Arrival Time"]))
    
        #if the transmission results were sliced and/or binned, this will take it into account and calculate the new table by the binning weight (1 is no binning)
    if end_slice == '':
        end_slice= len(table_wvl)
    table_wvl = table_wvl [start_slice:end_slice]
    table_wvl = table_wvl.groupby(np.arange(len(table_wvl))//binning).mean()
    
        # add the results obtained from the transmission in the ROIs selcted
    table_raw_transmission = pd.concat([table_wvl, Values], axis =1)
        
        # Send to the results to the directory from which you extracted the transmission images
    if save_results:
        table_raw_transmission.to_excel(src_dir + '/' + name_xlsx)
         
    return table_raw_transmission
        

# =============================================================================
#                              keep_roi_values_dict
# =============================================================================
def keep_roi_values_dict(dictionary, **kwargs):
    
        # initialize and create a list with name = values for each region
    rois = []
    for key,val in dictionary.items():
        rois.append( 'ROIs_'+ key + ' = ' + str(val))
        
        # return the list of values in a legible format for %load magic
    return ('\n'.join(rois))

    
# =============================================================================
#                       get_cross_sections
# =============================================================================

def get_cross_sections (transmission_values_tab, casing_data, compounds_dict, requested_cs , dst_dir = '' , save_table = True, name_xlsx = 'CS_results.xlsx', **kwargs):
    
        # import libraries
    import numpy as np
    import pandas as pd
    import collections, functools, operator
    
    barns = 1e24
    avogadro = 6.02214076e23
        # final results table
    cs_result_tab = pd.DataFrame()
    cs_result_tab["Wavelength [Å]"] = transmission_values_tab["Wavelength [Å]"]
    
        # get the current working directory
    current_dir = os.path.dirname(os.path.realpath(__file__))
        # get the reference data for wavelength and cross sections
    H_CS = pd.read_csv(current_dir+ r'\Ref_cs\H.txt', usecols=[1,2], sep='\t', names=['Wavelength [Å]', 'CS [barns]'], header=None, skiprows=1)
    H_ref_cs = H_CS.sort_values('Wavelength [Å]')
    O_CS = pd.read_csv(current_dir+ r'\Ref_cs\O.txt', usecols=[1,2], sep='\t', names=['Wavelength [Å]', 'CS [barns]'], header=None, skiprows=1)
    O_ref_cs = O_CS.sort_values('Wavelength [Å]')
    C_CS = pd.read_csv(current_dir+ r'\Ref_cs\C.txt', usecols=[1,2], sep='\t', names=['Wavelength [Å]', 'CS [barns]'], header=None, skiprows=1)
    C_ref_cs = C_CS.sort_values('Wavelength [Å]')
    LI_CS = pd.read_csv(current_dir+ r'\Ref_cs\Li_nat.txt', usecols=[1,2], sep='\t', names=['Wavelength [Å]', 'CS [barns]'], header=None, skiprows=1)
    LI_ref_cs = LI_CS.sort_values('Wavelength [Å]')
    P_CS = pd.read_csv(current_dir+ r'\Ref_cs\P.txt', usecols=[1,2], sep='\t', names=['Wavelength [Å]', 'CS [barns]'], header=None, skiprows=1)
    P_ref_cs = P_CS.sort_values('Wavelength [Å]')
    F_CS = pd.read_csv(current_dir+ r'\Ref_cs\F.txt', usecols=[1,2], sep='\t', names=['Wavelength [Å]', 'CS [barns]'], header=None, skiprows=1)
    F_ref_cs = F_CS.sort_values('Wavelength [Å]')
    molecular_weights_data = pd.read_csv(current_dir+'\Ref_cs\mol_weights.txt', sep='\t')

    
        # firts position is first ROI... Read the compound vector 
    for column, compound in zip(transmission_values_tab.loc[:, transmission_values_tab.columns != 'Wavelength [Å]'], compounds_dict):
            # out of the compounds, read the composition
        
        if compound.get('molecules'):
            partial_concentration = []
            total_atoms = []
            solvent_dict={}
            for indx, molecule in enumerate(compound.get('molecules')):

                if type(molecule) == dict:
                    total_atoms.append([key for key in molecule.get('composition').keys()])
                    solvent_dict[molecule.get('abbv')] = solvent_dict.get(molecule.get('abbv'), calc_tot_conc_compound (molecule, molecular_weights_data))
                    
                    atoms = [key for key in molecule.get('composition').keys()]
                    number_atoms = [key for key in molecule.get('composition').values()]
                    # calculate the molecular weight with the number of atoms and their respective atomic weight
                    tot_weight = 0
                    for atom, number in zip (atoms, number_atoms):
                        tot_weight += molecular_weights_data.iloc[0][atom]*number
                
                    # calculate the total concentration 
                    partial_concentration.append(((molecule.get('density')/tot_weight)*avogadro)*compound.get('molecules')[indx+1])

            tot_concentration = sum(partial_concentration)
            
            atoms = [key for key in dict(functools.reduce(operator.add, map(collections.Counter, total_atoms)))]
            
                # cross section references table for the selected compound
            cs_references_tab = pd.DataFrame()
            cs_references_tab["Wavelength [Å]"] = transmission_values_tab["Wavelength [Å]"]
            
                # generate the table of reference cross section with the required atoms
            for  atom in atoms:
                name_library = atom + '_ref_cs'
                cs_atom_lib = eval(name_library)
                cs_references_tab[atom] = [np.interp (wvl, list(cs_atom_lib["Wavelength [Å]"]), list(cs_atom_lib["CS [barns]"]))/barns for wvl in list(cs_references_tab["Wavelength [Å]"])]
        
            for cs_req in requested_cs:
        #THIS PART IS FOR CALCULATIONG THE TOTAL CROSS_SECTIONS
                cs_req = cs_req.split("_", 1)[0].upper()
                if cs_req == 'TOTAL' :
                    # calculate the total cross sections
                    cs_result_tab['CS_tot_'+compound.get('abbv')] = (-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1])/(compound.get('thickness')*tot_concentration))*barns

        # THIS PART IS FOR CALCULATIONG THE H CROSS_SECTIONS
                else:
                    
                    target_atom = cs_req
                    conc_cs_ref_tab = pd.DataFrame()
                    conc_target_atom = []
                    for i, pure_solvent in enumerate(compound.get('molecules')):
                        
                        if type(pure_solvent) == dict:
                            conc_cs_ref_tab [pure_solvent['abbv']], conc_target_atom_compound  = calc_atom_conc_4_cs (pure_solvent.get('composition'), cs_references_tab, solvent_dict[pure_solvent['abbv']], compound.get('molecules')[i+1], target_atom = target_atom)
                            conc_target_atom.extend(conc_target_atom_compound)
                            
                    conc_cs_ref_tab = conc_cs_ref_tab.sum(axis=1)
                    conc_target_atom = sum(conc_target_atom)
                    
                        # calculate the other atom cross sections according to their wavelength 
                    cs_result_tab['CS_'+target_atom+'_in_'+compound.get('abbv')] = ((((-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1]))/compound.get('thickness'))-(conc_cs_ref_tab))/(conc_target_atom))*barns

        else:
            tot_number_atoms = compound.get('composition')
            atoms = [key for key in tot_number_atoms.keys()]
            number_atoms = [key for key in tot_number_atoms.values()]
                # calculate the molecular weight with the number of atoms and their respective atomic weight
            tot_weight = 0
            for atom, number in zip (atoms, number_atoms):
                tot_weight += molecular_weights_data.iloc[0][atom]*number
                
                # calculate the total concentration 
            tot_concentration = (compound.get('density')/tot_weight)*avogadro
            
            
                # cross section references table for the selected compound
            cs_references_tab = pd.DataFrame()
            cs_references_tab["Wavelength [Å]"] = transmission_values_tab["Wavelength [Å]"]
            
                # generate the table of reference cross section with the required atoms
            for  atom in atoms:
                name_library = atom + '_ref_cs'
                cs_atom_lib = eval(name_library)
                cs_references_tab[atom] = [np.interp (wvl, list(cs_atom_lib["Wavelength [Å]"]), list(cs_atom_lib["CS [barns]"]))/barns for wvl in list(cs_references_tab["Wavelength [Å]"])]
            
    
            
        #THIS PART IS FOR CALCULATIONG THE TOTAL CROSS_SECTIONS
            if 'total_cs'  in requested_cs:
                # calculate the total cross sections
                cs_result_tab['CS_tot_'+compound.get('abbv')] = (-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1])/(compound.get('thickness')*tot_concentration))*barns
    
        # THIS PART IS FOR CALCULATIONG THE H CROSS_SECTIONS
            if 'h_cs'  in requested_cs:
                
                if 'H' in atoms:
                        # remove the target atom and leave the remaining for calculation
                    new_references = cs_references_tab.copy()
                    del new_references["H"]
                    del new_references["Wavelength [Å]"]
                    
                        # calculate the cross sections and the concentration per atom in the compound and sum the values. This is done per atom in the compound
                    atom_conc_cs = []
                    for atom_cs in list(new_references):
                        atom_conc_cs.append(list(new_references[atom_cs]*compound.get ('composition')[atom_cs]*tot_concentration))
                    new_references['CS_n_conc']=sum(map(np.array, atom_conc_cs))
                    
                    # calculate the other atom cross sections according to their wavelength 
                    cs_result_tab['CS_H_in_'+compound.get('abbv')] = ((((-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1]))/compound.get('thickness'))-(new_references['CS_n_conc']))/(compound.get ('composition')['H']*tot_concentration))*barns
                else:
                    print('No H found in this molecule. Please revise your composition tree')
                    
                    
                    
        # THIS PART IS FOR CALCULATIONG THE O CROSS_SECTIONS
            if 'o_cs'  in requested_cs:
                
                if 'O' in atoms:
                        # remove the target atom and leave the remaining for calculation
                    new_references = cs_references_tab.copy()
                    del new_references["O"]
                    del new_references["Wavelength [Å]"]
                    
                        # calculate the cross sections and the concentration per atom in the compound and sum the values. This is done per atom in the compound
                    atom_conc_cs = []
                    for atom_cs in list(new_references):
                        atom_conc_cs.append(list(new_references[atom_cs]*compound.get ('composition')[atom_cs]*tot_concentration))
                    new_references['CS_n_conc']=sum(map(np.array, atom_conc_cs))
                    
                    # calculate the other atom cross sections according to their wavelength 
                    cs_result_tab['CS_O_in_'+compound.get('abbv')] = ((((-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1]))/compound.get('thickness'))-(new_references['CS_n_conc']))/(compound.get ('composition')['O']*tot_concentration))*barns
                else:
                    print('No O found in this molecule. Please revise your composition tree')
                    
        # THIS PART IS FOR CALCULATIONG THE C CROSS_SECTIONS
            if 'c_cs'  in requested_cs:
                
                if 'C' in atoms:
                        # remove the target atom and leave the remaining for calculation
                    new_references = cs_references_tab.copy()
                    del new_references["C"]
                    del new_references["Wavelength [Å]"]
                    
                        # calculate the cross sections and the concentration per atom in the compound and sum the values. This is done per atom in the compound
                    atom_conc_cs = []
                    for atom_cs in list(new_references):
                        atom_conc_cs.append(list(new_references[atom_cs]*compound.get ('composition')[atom_cs]*tot_concentration))
                    new_references['CS_n_conc']=sum(map(np.array, atom_conc_cs))
                    
                    # calculate the other atom cross sections according to their wavelength 
                    cs_result_tab['CS_C_in_'+compound.get('abbv')] = ((((-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1]))/compound.get('thickness'))-(new_references['CS_n_conc']))/(compound.get ('composition')['C']*tot_concentration))*barns
                else:
                    print('No C found in this molecule. Please revise your composition tree')
                    
        # THIS PART IS FOR CALCULATIONG THE Li CROSS_SECTIONS
            if 'li_cs'  in requested_cs:
                
                if 'li' in atoms:
                        # remove the target atom and leave the remaining for calculation
                    new_references = cs_references_tab.copy()
                    del new_references["li"]
                    del new_references["Wavelength [Å]"]
                    
                        # calculate the cross sections and the concentration per atom in the compound and sum the values. This is done per atom in the compound
                    atom_conc_cs = []
                    for atom_cs in list(new_references):
                        atom_conc_cs.append(list(new_references[atom_cs]*compound.get ('composition')[atom_cs]*tot_concentration))
                    new_references['CS_n_conc']=sum(map(np.array, atom_conc_cs))
                    
                    # calculate the other atom cross sections according to their wavelength 
                    cs_result_tab['CS_Li_in_'+compound.get('abbv')] = ((((-np.log(transmission_values_tab[column]/casing_data.iloc[:, 1]))/compound.get('thickness'))-(new_references['CS_n_conc']))/(compound.get ('composition')['Li']*tot_concentration))*barns
                else:
                    print('No Li found in this molecule. Please revise your composition tree')
            
        if save_table:
            cs_result_tab.to_excel(dst_dir + '/' + name_xlsx)
        # generate a statement to be inserted in the jupyter notebook
    return cs_result_tab



def calc_tot_conc_compound (compound_composition_dict, molecular_weights_data,**kwargs):
    avogadro = 6.02214076e23
    atoms = [key for key in compound_composition_dict.get('composition').keys()]
    number_atoms = [key for key in compound_composition_dict.get('composition').values()]
        # calculate the molecular weight with the number of atoms and their respective atomic weight
    tot_weight = 0
    for atom, number in zip (atoms, number_atoms):
        tot_weight += molecular_weights_data.iloc[0][atom]*number
        
        # calculate the total concentration 
    return (compound_composition_dict.get('density')/tot_weight)*avogadro



def calc_atom_conc_4_cs (compound_composition_dict, list_atom_references, compound_conc, vol_fraction, target_atom, **kwargs):

    new_references = list_atom_references.copy()
    
    for atom in list(new_references):

        if atom not in compound_composition_dict.keys():
            del new_references[atom] 

        # calculate the cross sections and the concentration per atom in the compound and sum the values. This is done per atom in the compound
    atom_conc_cs_table = []
    conc_target_atom = []
    
    for atom_cs in list(new_references):
        
        if atom_cs == target_atom:
            conc_target_atom.append(compound_composition_dict[atom_cs]*compound_conc*vol_fraction)
        else:
            atom_conc_cs_table.append(list(new_references[atom_cs]*compound_composition_dict[atom_cs]*compound_conc*vol_fraction))
        
        #give the dataframe with the calculated concentration and cross section per atom in a compound
    atom_conc_cs_table=list(sum(map(np.array, atom_conc_cs_table)))
    return atom_conc_cs_table, conc_target_atom
