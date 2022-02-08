# ******************************************************************

#                           MERGERS

# Collection of functions to perform merge operation between images

# ******************************************************************

    # import packages
from img_utils import get_img, write_img
from processors import exec_proc, load_img_params, load_imgdir_params, \
    load_dyn_params, exec_start, exec_end
from file_utils import file_index
from tqdm.auto import tqdm
from astropy.time import Time
import numpy as np
from parameters import param, get_all_params

# ------------------------------------------------------------------
#                           mrg_medproj
# ------------------------------------------------------------------
# Performs a pixel-wise median projection of a stack of images

# input parameters: src_stack = stack of images as a 3D array
#                   **kwargs = collection of named parameters (unused here)

# return value: single image (2D array) containing the projection
# ------------------------------------------------------------------

def mrg_medproj(src_stack, **kwargs):
        
        # import package
    import numpy as np

        # perform pixel-wise median projection in the stack depth direction
        # and recast the image type to the original one
    return np.median(src_stack, axis=2).astype(src_stack.dtype)

# ------------------------------------------------------------------
#                           mrg_avgproj
# ------------------------------------------------------------------
# Performs a pixel-wise average projection of a stack of images

# input parameters: src_stack = stack of images as a 3D array
#                   **kwargs = collection of named parameters (unused here)

# return value: single image (2D array) containing the projection
# ------------------------------------------------------------------

def mrg_avgproj(src_stack, **kwargs):
    
        # import package
    import numpy as np

        # perform pixel-wise median projection in the stack depth direction
        # and recast the image type to the original one
    return np.average(src_stack, axis=2).astype(src_stack.dtype)


def mrg_avgproj_nan(src_stack, **kwargs):
    
        # import package
    import numpy as np

        # perform pixel-wise median projection in the stack depth direction
        # and recast the image type to the original one
    return np.nanmean(src_stack, axis=2).astype(src_stack.dtype)

# ------------------------------------------------------------------
#                           simple_merge
# ------------------------------------------------------------------
# Merges the images contained in a single directory and saves the resulting
# image. The 'merger' is defined as a function which casts a 3D stack of
# images of size (x,y,n) into a single image of size (x,y). Example of
# mergers are the pixel-wise median and average projections.
    
# Optionally, a sequence of pre-processing steps can be defined.

# input parameters: src_dir = src_dir = source directory (relative to 'base_dir')
#                   dst_file = name of the destination file
#                   merger = name of the merger function
#                   base_dir = base directory to define the relative paths
#                   pre_proc = optional sequence of processing steps to be
#                       applied to each image of the stack before merging.
#                   index_range = [min, max] array used to select only a given
#                       range of files to merge (base on their index)
#                       Default: empty array meaning all files are used
#                   tag_pos = position of the tag representing the index in the
#                       file name (tags are separated by '_'). Default: last one
#                   overwrite = if true, destination is overwritten if exists
#                       Default: False
#                   **kwargs = additional list of named parameters. **kwargs will
#                           be passed to different sub functions, but in particular:
#                   - All values defined in **kwargs are made available to
#                     the processing steps (if a pre-processing sequence is defined)
#                   - All parameters in **kwargs whose name ends with '_imgf'
#                     are assumed to be image file names. The corresponding images
#                     are loaded and made available to the processing steps
#                     as named parameters with the '_imgf' ending changed to 'img'
#                   - The 'start_before', 'start_after', 'stop_before' and
#                     'stop_after' parameters are passed to the 'exec_proc'
#                     function and can be used to define a subset of the
#                     pre-processing sequence to be applied.

# return value: none
# ------------------------------------------------------------------
    
def simple_merge(src, dst, merger, seq=None, merge_at=None, proc_from=None, proc_to=None,
                 index_range=[], tag_pos=-1, overwrite=False, silent=False, **kwargs):
    
        # import packages
    import os
    import numpy as np
    import warnings
    from img_utils import crop_img
    
        # output start timing indication
    if not silent:
        start_time = exec_start()
        
        # get all the general processing parameters and add them to kwargs
        # if the same name is present in both, kwargs has the priority
    params = get_all_params()
    kwargs_p = {**params, **kwargs}
    
        # if no base directory is specified, get it from the general processing parameters
    base_dir = param('base_dir')
    
        # get the 'roi' parameter (if defined)
    if 'roi' in kwargs_p.keys():
        roi_val = kwargs_p['roi']
    else:
        roi_val = None
    
        # compute the absolute path of the source directory
    src_dir_abs = os.path.join(base_dir, src)
    
        # compute the absolute path of the destination file
    dst_name = os.path.join(base_dir, dst)
    dst_name_dir, _ = os.path.split(dst_name)
    
        # if the destination already exists, skip the computation of the
        # merged image
    if os.path.exists(dst_name) and not overwrite:
        print('Destination file "' + dst + '" already exists, skipping"')
        nfiles = 0
    
        # otherwise proceed to the calculation
    else:
        
            # get the list of files in the source directory and the number of files
        file_list = os.listdir(src_dir_abs)
        nfiles = len(file_list)
        
            # first image indicator
        first_img = True
        
            # create the image buffer (for 3D processing)
        kwargs_p['img_buffer'] = {}
        
            # initialize empty list of headers
        headers = []
        
            # initialize file index and image index
        i = 0
        i_img = 0
           
        if seq is None:
            pre_proc = None
            post_proc = None
        else:
            
                          # find the index of the start milestone, if any
            if proc_from is not None:
               try:
                   first_step = seq.index(proc_from)
               except ValueError:
                   raise ValueError('Start milestone "'+ proc_from +'" not found in processing sequence')
            else:
                first_step = 0
                
                # find the index of the end milestone, if any
            if proc_to is not None:
               try:
                   last_step = seq.index(proc_to)
               except ValueError:
                   raise ValueError('End milestone "'+ proc_to +'" not found in processing sequence')
            else:
                last_step = len(seq)
                   
            sub_seq = seq[first_step:last_step]
            
            if merge_at is None:
                pre_proc = sub_seq
                post_proc = None
            else:
                i_mrg = sub_seq.index(merge_at) + 1
                pre_proc = sub_seq[:i_mrg]
                post_proc = sub_seq[i_mrg:]
        
            # loop for each source file in the list
        if silent:
            loop_list = file_list
        else:
            loop_list = tqdm(file_list, desc='Processing')
        for name in loop_list:
            
                # if an index range is defined, test whether the current file is in range
            do_proc = 1
            if index_range != []:
                fi = file_index(name, tag_pos)
                if fi < index_range[0] or fi > index_range[1]:
                    do_proc = 0
            
                # if the image is selected for processing
            if do_proc:
                
                    # load the image
                file_name = os.path.join(src, name)
                        
                if pre_proc is not None or post_proc is not None:
                    
                        # load the images which are used as parameters
                        # (see detailed description of the 'load_img_params' function)
                    kwargs_img = load_img_params(**kwargs_p)
                    
                        # load the images which are used as dynamic parameters
                        # (see detailed description of the 'load_img_params' function)
                    kwargs_img = load_dyn_params(name, **kwargs_img)
                
                    # if a pre-processing sequence is defined
                if pre_proc is not None:
                
                        # disable the warnings occuring during the processing
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    
                            # execute the pre-processing sequence on the image
                        img, header = exec_proc(0, seq=pre_proc, allow_3D=True,
                                        src_fname = os.path.join(base_dir, file_name),
                                        get_header=True, **kwargs_img)
                    
                    # if there is no pre-processing sequence, just load the image
                else:
                    img, header = get_img(file_name, base_dir, get_header=True)
                    
                    # only include the image if it is valid
                if img.size > 0:
                
                        # if this is the first image, create the 3D image stack
                    if first_img:
                        stack = np.zeros([img.shape[0], img.shape[1], len(file_list)], img.dtype)
                        stack[:,:,0] = img
                        first_img = False
                        # otherwise, add the image to the stack
                    else:
                        stack[:,:,i_img] = img
                        
                        # increase image index
                    i_img = i_img + 1
                    
                        # add the header to the list
                    headers.append(header)
                
                    # increase file index
                i = i+1
                
            # remove the empty images end the end of the stack
        nimg = i_img
        stack = stack[:,:,0:nimg]
            
            # apply the selected merge operation on the loaded stack
        mrg_img = merger(stack, **kwargs_p)
    
            # merge the headers
        mrg_header = merge_headers(headers)
        
        mrg_header['HISTORY'] = 'Processing: Image merged from ' + str(nimg) + ' images'
        
        if post_proc is not None:
            dst_img, dst_header = exec_proc(mrg_img, post_proc, get_header=True,
                                            header=mrg_header, **kwargs_img)
        else:
            dst_img = mrg_img
            dst_header = mrg_header
        
            # if the destination directory does not exist, create it
        if not os.path.exists(dst_name_dir):
            os.makedirs(os.path.join(dst_name_dir,''))
                    
            # if a ROI is defined, crop the processed image
        if roi_val is not None:
            dst_img = crop_img(dst_img, roi_val)
        
            # write the merged image
        write_img(dst_img, dst_name, header=dst_header, overwrite=overwrite)  
        
        # output end timing indication
    if not silent:
        exec_end(start_time, nfiles)

    
# ------------------------------------------------------------------

def merge_headers(head_list):
    
    comment_fields = ['COMMENT', 'HISTORY']
    
    fields_list = []
    
        # get the list of all fields present in the headers, except comments
    for header in head_list:
        fields_list.extend([x for x in header.keys() if x not in fields_list \
                            and x not in comment_fields])
    
        # use the first of the headers to create the merged one
    head_mrg = head_list[0].copy()
    
        # loop for all fields
    for field in fields_list:
        
            # if all headers have the curent field:
        if all([field in header for header in head_list]):
        
                # get all values for this field
            val_list = [header[field] for header in head_list]
            
                # if all values are the same, keep the value from first header
            if all([x == val_list[0] for x in val_list]):            
                pass
            
                # if we have different values
            else:
            
                    # try to compute the average, assuming numerical values
                try:
                    head_mrg[field] = sum(val_list)/len(val_list)
                
                    # if not a numerical value
                except (ValueError, TypeError):
                    
                        # try to average the value as a time stamp
                    try:
                        ts_list = Time(val_list)
                        ts_mean = ts_list.min() + np.mean(ts_list - ts_list.min())
                        head_mrg[field] = str(ts_mean)
                    
                        # if not a time stamp
                    except (ValueError, TypeError):
                        
                            # write an empty value
                        head_mrg[field] = (None, 'Cannot merge values')
        
            # if any of the headers does not have the current field
        else:
                # write an empty value
            head_mrg[field] = (None, 'Field value missing')
    
    return head_mrg

    
# ------------------------------------------------------------------

def multi_merge(src_dir, dst_dir, merger, dst_name_rule=None, **kwargs):
    
    from file_utils import file_list
    import os
    
        # output start timing indication
    start_time = exec_start()
    
        # if no base directory is specified, get it from the general processing parameters
    base_dir = param('base_dir')
    
        # get the list of FITS files
    flist = file_list(src_dir, base_dir, 'fits')
    dirlist = []
    
        # get the number of files
    nfiles = len(flist)
    
        # get the list of directories containing FITS files
    for fname in flist:
        dirname = os.path.dirname(fname)
        if dirname not in dirlist:
            dirlist.append(dirname)
            
        # if no naming rule is defined, use the default one:
    if dst_name_rule is None:
        dst_names = [os.path.join(x, x.split('\\')[-1]+'.fits') for x in dirlist] 
        
        # else use the specified naming rule
    else:
        dst_names = [dst_name_rule(x) for x in dirlist]
        
        # loop for all directories to merge
    for i in tqdm(range(len(dirlist))):
        simple_merge(os.path.join(src_dir, dirlist[i]), os.path.join(dst_dir, dst_names[i]), 
                     merger=merger, silent=True, **kwargs)
        
        # output end timing indication
    exec_end(start_time, nfiles)
    
    
    
    
    