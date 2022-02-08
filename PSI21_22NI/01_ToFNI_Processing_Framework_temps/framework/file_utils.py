""" 
******************************************************************

                           FILE_UTILS

             Collection of functions to manage files

******************************************************************
"""
# ------------------------------------------------------------------
#                           file_list
# ------------------------------------------------------------------
# This function searches for all files with a given extension in a directory
# and its sub-directories
        
# input parameters: src_dir = source directory (relative to 'base_dir')
#                   base_dir = base directory to define the relative paths
#                   ftype = file type (default is 'fits')
#
#
# return value: file list
# ------------------------------------------------------------------

def file_list(src_dir, base_dir, ftype='fits'):
       
        # import packages
    import os
    
        # compute the desitred extension string
    sel_ext = ('.' + ftype)
    
        # get the list of all  files in the specified source directory matching
        # the desired extension (including in its subdirectories)
    flist = []
    for root, dirs, files in os.walk(os.path.join(base_dir, src_dir)):
        for name in files:
            _ , ext = os.path.splitext(name)
            
            if ext == sel_ext:
                flist.append(os.path.relpath(os.path.join(root, name), os.path.join(base_dir, src_dir)))
        
        # return the file list
    return flist

# ------------------------------------------------------------------
    
def change_ext(file_name, new_ext):
       
        # import packages
    import os
    
    
    fname, _ = os.path.splitext(file_name)
    new_name = fname + '.' + new_ext
       
    return new_name

# ------------------------------------------------------------------
    
def add_tag(file_name, new_tag):
       
        # import packages
    import os
    
    
    fname, ext = os.path.splitext(file_name)
    new_name = fname + '_' + new_tag + ext
       
    return new_name

# ------------------------------------------------------------------
    
def get_tag(file_name, tag_pos=-1):
       
        # import packages
    import os
        
        # extract the file name without directory and extension
    f_noext, _ = os.path.splitext(file_name)
    _ , name = os.path.split(f_noext)
    
        # extract the tag list
    tags = name.split('_')    
    
        # get the number of tage in the file name
    ntags = len(tags)
    
        # if tag_pos is positive, position is defined from the first tag
    if tag_pos >= 0 and tag_pos < ntags:
        ret_val = tags[tag_pos]
        
        # if tag_pos is positive, position is defined from the last tag
    elif tag_pos >= (-ntags):
        ret_val = tags[ntags+tag_pos]
        
        # if tag_pos is out of bounds, return no result
    else:
        ret_val = ''
        
        # return the extracted tag value
    return ret_val


# ------------------------------------------------------------------
    
def change_tag(file_name, new_tag, tag_pos=-1):
       
        # import packages
    import os
        
        # extract the file name without directory and extension
    f_noext, ext = os.path.splitext(file_name)
    fdir , name = os.path.split(f_noext)
    
        # extract the tag list
    tags = name.split('_')  
    
        # get the number of tage in the file name
    ntags = len(tags)
    
        # if tag_pos is positive, position is defined from the first tag
    if tag_pos >= 0 and tag_pos < ntags:
            # update the tag at the defined value
        tags[tag_pos] = new_tag
        
        # if tag_pos is positive, position is defined from the last tag
    elif tag_pos >= (-ntags):
            # update the tag at the defined value
        tags[ntags+tag_pos] = new_tag
          
        # reform the name from all tags
    new_name = '_'.join(tags)
    
        # join with the full path and extension
    new_fname = os.path.join(fdir, new_name) + ext
    
        # return the new value
    return new_fname
    
# ------------------------------------------------------------------
    
def file_index(file_name, tag_pos=-1):
    
        # get the index tag
    tag = get_tag(file_name, tag_pos)
    
        # try to convert to int
    try:
        f_index = int(tag)
        # if the tag is not a number, return -1
    except ValueError:
        f_index = -1

        # return the file index value
    return f_index
# ------------------------------------------------------------------

    
def write_results(dst_file, base_dir, res, keys, create_dir=False):
    
    import os

        # compute the absolute path of the destination file
    dst_name = os.path.join(base_dir, dst_file)
    
    if os.path.isfile(dst_name):
        exist = 1
    else:
        exist = 0

        # compute the absolute path of the destination file
    dst_name = os.path.join(base_dir, dst_file)
    
        #if the 'create_dir' option is set
    if create_dir:
        
            # get the destination directory
        dst_name_dir, _ = os.path.split(dst_name)
        
            # and create this destination directory if it does not exist
        if not os.path.exists(dst_name_dir):
            os.makedirs(os.path.join(dst_name_dir,''))
    
    file = open(dst_name, 'a')
    
    if exist == 0:
        
        for i, k in enumerate(keys):
            
            if i != 0:
                file.write(';')
            file.write(k)
            
        file.write('\r')
        
    for r in res:
        
        for i, k in enumerate(keys):
            
            if i != 0:
                file.write(';')
            file.write(str(r[k]))
            
        file.write('\r')
        
    
    file.close()

# -----------
    
def read_file(src_file, base_dir, max_lines=1e20, silent=True):
    
    import os
    
        # compute the absolute path of the destination file
    src_name = os.path.join(base_dir, src_file)
    
    file = open(src_name)
    
    stop = False
    i = 0
    nbytes = 0
    
    retval = []
    
    while not stop > 0:
        line = file.readline().rstrip()
        l = len(line)
        if l > 0:
            retval.append(line)
            nbytes = nbytes + l
        else:
            stop = True
        i = i + 1
        if i >= max_lines:
            stop = True
        if not silent and (i % 100000) == 0:
            print('Reading (' + str(round(float(nbytes)/(1024.0**2)*10)/10) + ' MB)', end='\r')
    
    file.close()
    
    return retval
    
    
        
        
        
    
    

# ------------------------------------------------------------------