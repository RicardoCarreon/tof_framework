# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 17:11:31 2020

@author: boillat
"""

_proc_params = {'scope':'general', 'general':{}}

"""-------------------------------------------------------------------------"""

def set_group_scope(group):
    """
    Selects a parameter group. If the name of the group is not existing yet,
    the function creates an empty dictionnary for this group. Once a group is
    selected, all subsequent parameter definitions will be associated with
    this group

    Parameters
    ----------
    group : str
        Name of the group to select.

    Returns
    -------
    None.

    """
        # link to the parameters dictionnary
    global _proc_params
    
        # list of names which cannot be used for naming groups
    reserved_names = ['scope', 'general']
    
        # if the selected group is in the resevred list, throw an error
    if group in reserved_names:
        raise ValueError('"'+group+'" is reserved and cannot be used as a group name')
    
        # if the scope is new, create an empty dictionnary
    if group not in _proc_params.keys():
        _proc_params[group] = {}
        
        # select the new scope
    _proc_params['scope'] = group
    

"""-------------------------------------------------------------------------"""

def set_general_scope():
    """
    Set the parameter scope to the general parameters.

    Returns
    -------
    None.

    """
    
        # link to the parameters dictionnary
    global _proc_params
        
        # select the new scope
    _proc_params['scope'] = 'general'
    

"""-------------------------------------------------------------------------"""

def set_params(general=False, **kwargs):
    """
    Store one or several values as general processing parameters. For example,
    "set_params(filter_size=5)" will retain the value of 5 for the
    parameter 'filter_size'.
    If the scope has been set to a processing group, the parameter will be
    associated to this processing group. Otherwise, it will be placed in the
    general parameters.
    Some parameters, such as 'base_dir', have a predefined meaning.

    Parameters
    ----------
    Open list of parameters to store
    general: if True, for the parameter(s) to be stored in the general scope
        even if the current scope is a group.

    Returns
    -------
    None.

    """
    
        # get the processing parameters global variable
    global _proc_params
    
        # if the general flag is set, use the general scope
    if general:
        scope = 'general'
        # otherwise use the selected scope
    else:
        scope = _proc_params['scope']
    
        # set all defined parameter values
    for key, value in kwargs.items():
        _proc_params[scope][key] = value

"""-------------------------------------------------------------------------"""

def param(name):
    """
    Get the value of a previously defined processing parameter. If the scope
    is currently set to a group, the name will be searched in the group parameters
    first, and if not found there, in the general parameters. If the scope is
    not set to a group, the parameter will only be searched in the general
    parameters.

    Parameters
    ----------
    name : string
        Name of the parameter to get.

    Returns
    -------
    The value of the chosen parameter.

    """
    
        # get the processing parameters global variable
    global _proc_params
    
        # get the currently set scope
    scope = _proc_params['scope']
    
        # if the scope is set to a group, first look in the group parameters
    if scope != 'general':
        if name in _proc_params[scope].keys():
            return _proc_params[scope][name]
    
        # if not found, of if the scope is not set to a group, look in the
        # general parameters
    if name in _proc_params['general'].keys():
        return _proc_params['general'][name]
    
        # otherwise return an empty value
    else:
        return None
        
"""-------------------------------------------------------------------------"""

def get_all_params():
    
        # get the processing parameters global variable
    global _proc_params
    
        # get the currently set scope
    scope = _proc_params['scope']
    
        # if the scope is general, just return the corresonding dictionnary
    if scope == 'general':
        return _proc_params['general']
    
        # if a group scope is set, merge the group and general dictionnaries
        # group parameters have priority
    else:
        group_params = _proc_params[scope]
        gen_params = _proc_params['general']
        return {**gen_params, **group_params}
    