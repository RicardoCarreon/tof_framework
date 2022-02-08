# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------

                                overlap_corr_v2

------------------------------------------------------------------------------

New MCP overlap correction functions. These functions can be used to perform
the overlap correction on one specific acquition on applied as a batch on
all acquisitions contained in a single folder. The processing is nearly the same
as described in the following paper:

A.S. Tremsin, J.V. Vallerga, J.B. McPhate, and O.H.W. Siegmund, 
Optimization of Timepix count rate capabilities for the applications with a 
periodic input signal, J. Instrum., vol. 9, no. 5, 2014. 
https://doi.org/10.1088/1748-0221/9/05/C05026

A short summary of this correction algorithm is given here: For each image (time
bin), the probability "p" of overlap is computed as the pixel-wise sum of all
previous images in the same shutter windows, divided by the number of pulses
for which data was effectively recieved (obtained from the "ShutterCount.txt"
file). The intensity of the image is then corrected by dividing it by (1-p).
The intensity is further normalized base on an "ideal" number of received pulses.

The present algorithm slightly differs from the one described by Tremsin et al.:
Here, the overlap probabilty for one image is calculated based on the sum of
all previous images in the shutter windows, plus the half of the current image
itself. This is to take into account the probabilty of losing an event due to
the presence of two events within the same time bin. It is only relevant if
the time bins are relatively long.

------------------------------------------------------------------------------

The functions to be used for the correction are the following:
    - mcpoc_get_exp_names: used to extract all the experiment names in a folder
    - mcpoc_correct_exp: perform the correction for one single experiment
    - mcpoc_correct_batch: perform the correction for all experiments in a folder
    
The following subfunctions are also available for debugging and analysis purposes:
    - mcpoc_get_shutter_counts: get the list of recorded pulses for each shutter
    - mcpoc_get_shutter_times: get the start of end time of each shutter window
    - mcpoc_get_spectrum: get the spectrum
    - mcpoc_get_shutter_indexes: get the indexes of the first and last images for
            each shutter window

------------------------------------------------------------------------------

P. Boillat, Paul Scherrer Institut (PSI), 3.1.2020

------------------------------------------------------------------------------
"""

#    MODIFICATION

'''
Modifications to the function :
        mcpoc_correct_batch
        
    * Global folder selection added. This allows to perform the overlpa correction 
        to different folders at the same time
    * Creation of timestamps.txt file added to the function. This allows to know 
        when the experiments where modified (experiment median timestamp)
------------------------------------------------------------------------------

R. Carreon, Paul Scherrer Institut (PSI), 14.04.2020

------------------------------------------------------------------------------
'''

    # necessary imports
import glob, os
import numpy as np
from astropy.io import fits
from tqdm import tqdm

    # constants for defining the different text files
SCOUNT_NAME = 'ShutterCount'
STIMES_NAME = 'ShutterTimes'
SPEC_NAME = 'Spectra'

"""
------------------------------------------------------------------------------

                                mcpoc_get_exp_names

------------------------------------------------------------------------------

Gets the list of experiments names for a given folder

Parameters:
    src_dir = source directory containing the uncorrected images
    
Return value:
    a list containing the experiment names
    
------------------------------------------------------------------------------
"""

def mcpoc_get_exp_names(src_dir):
    
        # get the list of all *.txt files
    sorted_TXT= sorted(glob.glob(src_dir+'/*.txt'))
    
        # keep only the 'ShutterCount' files
    sorted_TXT = [os.path.basename(x) for x in sorted_TXT if SCOUNT_NAME in x ]
    
        # get the base names
    names = [x[0:x.rfind('_')] for x in sorted_TXT]
    
        # return the result
    return names

"""
------------------------------------------------------------------------------

                                mcpoc_read_txt

------------------------------------------------------------------------------

Base function for reading from the text files and discarding the extra lines
only containing zeros. Used by specific reading functions below.

Parameters:
    src_dir = source directory containing the uncorrected images
    exp_name = experiment name (the list of valid experiment names can be
              obtained using the mcpoc_get_exp_names() function)
    ftype = type of file (one of the constants SCOUNT_NAME, STIMES_NAME or
            SPEC_NAME)
    usecols = columns of the text file to be read
    
Return value:
    a numpy 1D or 2D array (depending on the value of usecols) with the data
    
------------------------------------------------------------------------------
"""

def mcpoc_read_txt(src_dir, exp_name, ftype, usecols):
    
        # compute the file name
    fname = os.path.join(src_dir, exp_name + '_' + ftype + '.txt')
    
        # read the file
    vals = np.genfromtxt(fname, usecols=usecols)
    
        # only keep lines until the last non-zero value
        # (checks if the array is 1D or 2D)
    if len(vals.shape) == 1:
        vals = vals[0:max(np.nonzero(vals)[0])+1]
    else:
        vals = vals[0:max(np.nonzero(vals[:,0])[0])+1,:]
    
        # return the values
    return vals

"""
------------------------------------------------------------------------------

                            mcpoc_get_shutter_counts

------------------------------------------------------------------------------

Get the number of effectively recorded pulses for each shutter window
(from the "<exp name>_ShutterCount.txt" file).

Parameters:
    src_dir = source directory containing the uncorrected images
    exp_name = experiment name (the list of valid experiment names can be
              obtained using the mcpoc_get_exp_names() function)
    
Return value:
    a numpy 1D array containing the shutter times
    
------------------------------------------------------------------------------
"""

def mcpoc_get_shutter_counts(src_dir, exp_name):
    
    return mcpoc_read_txt(src_dir, exp_name, SCOUNT_NAME, 1)

"""
------------------------------------------------------------------------------

                            mcpoc_get_shutter_times

------------------------------------------------------------------------------

Get the start and end times of each shutter window. The shutter times are 
computed by successively adding the gap and windows widths from all shutters

Parameters:
    src_dir = source directory containing the uncorrected images
    exp_name = experiment name (the list of valid experiment names can be
              obtained using the mcpoc_get_exp_names() function)
    
Return value:
    a numpy array with two columns:
        - The first column contains the window start time
        - The second colum contains the window end time
    
------------------------------------------------------------------------------
"""

def mcpoc_get_shutter_times(src_dir, exp_name):
    
        # get the 'raw' shutter times (gap and window width)
    times_raw = mcpoc_read_txt(src_dir, exp_name, STIMES_NAME, (1,2))
    
        # prepare a buffer for the results
    times = np.zeros([times_raw.shape[0], 2])
    
        # compute the shutter times based on the initial gap and window width
    times[0,0] = times_raw[0,0]
    times[0,1] = times[0,0] + times_raw[0,1]
    
        # compute all other shutter times based on the gap and window widths
    for i in range(1, times_raw.shape[0]):
        times[i, 0] = times[i-1, 1] + times_raw[i, 0]
        times[i, 1] = times[i, 0] + times_raw[i, 1]
        
        # return the results
    return times

"""
------------------------------------------------------------------------------

                                mcpoc_get_spectrum

------------------------------------------------------------------------------

Get the measured intensity spectrum

Parameters:
    src_dir = source directory containing the uncorrected images
    exp_name = experiment name (the list of valid experiment names can be
              obtained using the mcpoc_get_exp_names() function)
    
Return value:
    a numpy array with two columns:
        - The first column contains the time of flight
        - The second colum contains the intensity
    
------------------------------------------------------------------------------
"""

def mcpoc_get_spectrum(src_dir, exp_name):
    
    return mcpoc_read_txt(src_dir, exp_name, SPEC_NAME, (0,1))

"""
------------------------------------------------------------------------------

                            mcpoc_get_shutter_indexes

------------------------------------------------------------------------------

For each shutter, get the indexes of the first and last bin contained in its
time window. A tolerance value of 1us is used so that bins that are "just
outside" of a window due to ronuding errors are not discarded. 

Parameters:
    src_dir = source directory containing the uncorrected images
    exp_name = experiment name (the list of valid experiment names can be
              obtained using the mcpoc_get_exp_names() function)
    
Return value:
    a numpy array with two columns containing the first and last index values
    
------------------------------------------------------------------------------
"""

def mcpoc_get_shutter_indexes(src_dir, exp_name):
    
        # set a tolerance so that if a bin is "just outside" of a window
        # due to rounding errors it is still considered
    tolerance = 1e-6
    
        # get the shutter times and the spectrum
    sh_times = mcpoc_get_shutter_times(src_dir, exp_name)
    spec_times = mcpoc_get_spectrum(src_dir, exp_name)[:,0]
    
        # get the maximum value of the spectrum
    spec_max = np.max(spec_times)
    
        # get the number of shutters
    nshut = sh_times.shape[0]
    
        # buffer for the results
    ivals = np.zeros([nshut, 2])
    
        # loop for each shutter
    for i in range(nshut):
        
            # get the corresponding time window
        tmin = sh_times[i,0]
        tmax = sh_times[i,1]
        
            # if the spectrum is at least partially covering this time window
        if tmin < spec_max:
            
                # get the min and max indices of the bins in the window
            imin = np.min(np.where(spec_times >= (tmin - tolerance))[0])
            imax = np.max(np.where(spec_times <= (tmax + tolerance))[0])
            ivals[i,:] = [imin, imax]
    
        # remove the lines where no bins where found
    ivals = ivals[0:max(np.nonzero(ivals[:,1])[0])+1,:]
    
        # return the results        
    return ivals

def mcpoc_read_img(src_dir, exp_name, index):
    import time
        # compute the file name
    fname = os.path.join(src_dir, exp_name + '_' + format(index, '05d') + '.fits')
    
        # read the file
    with fits.open(fname) as f:
        time_data = os.path.getmtime(fname)
        time_mod = time.ctime(os.path.getmtime(fname))
        img = (f[0].data)
        timestamp = [time_data, time_mod]
        # return the image
    return img, timestamp

def mcpoc_write_img(dst_img, dst_dir, dst_name, index, timestamp):

        # compute the destination name
    fname = os.path.join(dst_dir, dst_name + '_' + format(index, '05d') + '.fits')
    
    from astropy.io import fits
    hdu = fits.PrimaryHDU(dst_img)
    hdu.header ['TIMESTMP'] = timestamp[1]
    hdu.header ['TIME_DAT'] = timestamp[0]
    hdu.header['COMMENT'] = 'User: Ricardo Carreon'
    hdu.writeto(fname)
        
        # write the image to the destination file
    #fits.writeto(fname, dst_img)
    
"""
------------------------------------------------------------------------------

                            mcpoc_correct_exp

------------------------------------------------------------------------------

Perform the MCP overlap correction on one experiment

Parameters:
    src_dir = source directory containing the uncorrected images
    dst_dir = destination directory
    exp_name = experiment name (the list of valid experiment names can be
              obtained using the mcpoc_get_exp_names() function)
    ref_sh_count = reference number of pulses measured per experiment (the
                  number we would have in the 'ShutterCount.txt' file if
                  no packets were lost)
------------------------------------------------------------------------------
"""

def mcpoc_correct_exp(src_dir, dst_dir, exp_name, ref_sh_count, scale=1.0,
                      savetype=float):
    
        # get the indices of the bins covered by each shutter
    sh_indices = mcpoc_get_shutter_indexes(src_dir, exp_name)
    
        # get the measured shutters counts
    sh_counts = mcpoc_get_shutter_counts(src_dir, exp_name)
    
        # get the number of shutters
    n_shut = sh_indices.shape[0]
    
        # loop for all shutters
    for i_sh in range(n_shut):
        
            # get the minimum and maximum nidices for this shutter
        imin = int(sh_indices[i_sh, 0])
        imax = int(sh_indices[i_sh, 1])
        
            # for all bins in the current shutter window
        for i_bin in range(imin, imax+1):
            
                # get the source image
            src_img, timestamp = mcpoc_read_img(src_dir, exp_name, i_bin)
            
                # get the image dimensions
            szx = src_img.shape[0]
            szy = src_img.shape[1]
            
                # if this is the first image of the shutter window, reset the
                # correction buffer
            if i_bin == imin:
                oc_buffer = np.zeros([szx, szy])
                
                # compute the overlap probability (pixel-wise)
            p_img = (oc_buffer + src_img/2).astype(float)/sh_counts[i_sh]
            
                # compute the corrected destination image
            dst_img = src_img/(1-p_img)*ref_sh_count/sh_counts[i_sh]
            
                # applied the scale and convert to the saving format
            dst_img = (dst_img*scale).astype(savetype)
            
                # get the source image
            mcpoc_write_img(dst_img, dst_dir, exp_name, i_bin, timestamp)
                
                # add the normalized image to the correction buffer
            oc_buffer = oc_buffer + src_img

"""
------------------------------------------------------------------------------

                            mcpoc_correct_batch

------------------------------------------------------------------------------

Perform the MCP overlap correction on one batch of experiments (all in the
same folder).

Parameters:
    src_dir = source directory containing the uncorrected images
    dst_basedir = base destination directory (the corrected images will be put
                  in subdirectories, one per experiment)
    ref_sh_count = reference number of pulses measured per experiment (the
                  number we would have in the 'ShutterCount.txt' file if
                  no packets were lost)
    desc = description of the processing step (for display purposes)
------------------------------------------------------------------------------

"""

def mcpoc_correct_batch(src_dir, dst_basedir, ref_sh_count, scale=1.0,
                        savetype=float, desc=''):
    
        # import libraries
    from os.path import normpath, basename
    
        # construct the timestamps.txt file
    txt_timestamps (src_dir, dst_basedir)
        
            # this allow to select the global folder 
    for exp_dir in glob.glob(src_dir + '/*/'):
        
            # get the list of experiment names
        exp_names = mcpoc_get_exp_names(exp_dir)
            
            # loop for all experiments
        for i in tqdm(range(len(exp_names)), desc=desc):
            
                # get the current experiment name
            exp_name = exp_names[i]
            
                # compute the folder name of the destination stack
            dst_dir = os.path.join(dst_basedir,os.path.basename(normpath(exp_dir)), 'Corrected_' + exp_name +'_set_' +
                                   format(i, '03d'))
            
                # if not existing, creates the destination directory
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir) 
            
                # perform the overlap correction on this experiment
            mcpoc_correct_exp(exp_dir, dst_dir, exp_name, ref_sh_count,
                              scale=scale, savetype=savetype)

#_________________________________________________________________________________________________________________
        
# =============================================================================
#                         txt_timestamps
# =============================================================================
def txt_timestamps (src_dir, dst_dir):
    '''
    Creates a txt file that contain the timestamp with the median file in each subfolder.
    This function is meant to be used with 'the overlap_corr_v2' and will serve as an evaluation of the 
        OB weightening.
    The function detects if there is already a 'timestamps.txt' file in the destination folder, writing the 
        obtained timestamps, if not, it creates it and write the timestamps.

    Parameters
    ----------
    src_dir : str
        source directory, same path as in the overlap correction
    dst_dir : str
         destination directory, same path as in the overlap correction

    Returns
    -------
    Creates a timestamps.txt file in the destination directory, if it exists already, it modifies it.

    '''
        # import libraries
    import pandas as pd
    from pathlib import Path
    import time
    
        # create the txt file path
    txt_file = dst_dir + '/timestamps.txt'
    
        # check if there is already a file in the destination folder
    if not os.path.exists(txt_file):
        Path(txt_file).touch()
        
            # writes a header with the most important information
        with open(txt_file, 'a') as f:
            f.write('Folder,Modification (s),Creation (s),Formatted modification,Formatted creation' + "\n")
            f.close
        
        # read the subfolders included in the source folder
    subfolders = os.listdir(src_dir)
    
        # creates a pandas dataframe for handling the information
    Timestamps = pd.DataFrame(columns = ['Folder','Modification (s)','Creation (s)','Formatted modification','Formatted creation'], index = np.arange(len(subfolders)))
    
    for idx,name in enumerate (subfolders):
        
            # creates a list with all the files contained in the sibfolder
        files = os.listdir(os.path.join(src_dir, name))
        
            # search for the median value and takes its path
        med_file_path = os.path.join (os.path.join(src_dir, name),files[round(len(files)/2)])
        
            # writes the folder name in the dataframe
        Timestamps ['Folder'][idx] = name 
        
            # writes the modification time (in seconds) 
        Timestamps ['Modification (s)'][idx] = os.path.getmtime(med_file_path)
        
            # writes the creation time (in seconds)
        Timestamps ['Creation (s)'][idx] = os.path.getctime(med_file_path)
        
            # writes the formatted modification time
        Timestamps ['Formatted modification'][idx] = time.ctime(os.path.getmtime(med_file_path))
        
            # writes the formatted creation time
        Timestamps ['Formatted creation'][idx] = time.ctime(os.path.getctime(med_file_path))
    
        # converts the dataframe into a string and writes it into the timestamps.txt file
    with open(txt_file, 'a') as f:
        f.write(Timestamps.to_csv(header = False, index = False))
        f.close


# ------------------------------------------------------------------
#                           select_directory
# ------------------------------------------------------------------
# this function is meant to be used together with the '%load' magic in jupyter
# notebook to provide an interactive interface for directory selection.

# input parameters: var_name = string with the name of the variable which will 
#                               contain the resulting path in the notebook
#                   base_dir = directory relative to which the directory path
#                               is specified (absolute path is used if omitted)

# return value: statement to be inserted in the jupyter code cell to 
#               assign the selected path to this variable

# ----------------------------------------------------------------------------------------------------------

def select_directory(var_name, base_dir=''):
    
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