# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:37:06 2020

@author: boillat
"""

def dynpar_lut(src_fname, base_dir, lut_fname):
    """
    Function used to read the value of a dynamic parameter from a lookup table.
    The lookup table is a CSV file (delimited with ';') with two columns named
    'src' and 'val'. The'src' column contains file patterns using wildcards
    (e.g. '*' or '?') and the 'val' column contains the corresponding parameter
    values.
    
    This function is means to be used as a dynamic parameter definition,
    for example for dynamic image parameters ending with '_imgf'

    Parameters
    ----------
    src_fname : string
        Current source file name.
    base_dir : string
        Base directory.
    lut_fname : string
        File name of the lookup table.

    Raises
    ------
    ValueError
        This error is raised if the current source file name does not match
        any line in the lookup table.

    Returns
    -------
    string
        Value of the dynamic parameter corresponding to the current source file 
        name.

    """
    
    import numpy as np
    import pandas
    import os
    from fnmatch import fnmatch
    
        # load the look up table
    df = pandas.read_csv(os.path.join(base_dir, lut_fname), delimiter=';')
   
        # check which lines match with the source file name
    matches = [fnmatch(src_fname, x) for x in df['src']]
        
        # if no match is found, return an error
    if not any(matches):
        raise ValueError('Source file "' + src_fname + '" does not match any line of the lookup table')
        
        # get the index of the first match
    imatch = np.where(matches)[0][0]
    
        # return the corresponding value
    return df['val'][imatch]
        
        
        
        
    
    