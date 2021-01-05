#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:41:49 2020

@author: Nastasha


pipeline for creating CDDFs without intermediate column density maps
useful for creating large sets of CDDFs with minimal disk storage
note that map files are created as an intermediate, but they are deleted
afterwards 
"""

import os
import sys
import numpy as np
import h5py

import make_maps_opts_locs as ol
import make_maps_v3_master as m3
import makehistograms_basic as mh


def create_cddf_singleslice(bins, *args, **kwargs):
    '''
    UNTESTED
    creates a file named cddf_<autonamed column density map file> in 
    make_maps_opts_locs.pdir with the histogram of the map

    Parameters
    ----------
    *args : 
        arguments for make_maps_v3_master.make_map
    **kwargs : 
        keyword arguments for make_maps_v3_master.make_map
    bins : array-like of floats or tuple of two such objects
        bin edges for the histogram; if the kwargs produce a weighted map,
        a second list is used for its histogram

    Returns
    -------
    None.

    '''    
    
    if 'nameonly' in kwargs:
        del kwargs['nameonly']
    if 'saveres' in kwargs:
        del kwargs['saveres']
    kwargs['hdf5'] = True
    
    filens = m3.make_map(*args, saveres=False, nameonly=True, **kwargs)
    
    run = True
    if filens[1] is None:
        if os.path.isfile(filens[0]):
            run = False
    else:
        if os.path.isfile(filens[0]) and os.path.isfile(filens[1]):
            run = False
    
    if run:
        m3.make_map(*args, savres=True, nameonly=False, **kwargs)
    
    
    with h5py.File(filens[0], 'r') as f0:
        if hasattr(bins[0], '__len__'):
            _bins = bins[0]
        else:
            _bins = bins
        _bins = list(bins)
        if _bins[0] != -np.inf:
            _bins = [-np.inf] + _bins
        if _bins[-1] != np.inf:
            _bins = _bins + [np.inf]
        _bins = np.array(_bins)
        
        vals = f0['map'][:].flatten()
        # max size ~32000**2 * 4 is good enough; for size maps, should be fine
        if len(vals) > 32000**2 * 4:
            print('Attempting a histogram with a {}-element array'.format(\
                  len(vals)))
        hist, edges = np.histogram(vals, bins=_bins)
 
        fparts = filens[0].split('/')
        fon0 = ol.pdir + '/cddf_' + fparts[-1] 
        with h5py.File(fon0, 'a') as fo0:
            f0.copy('Header', fo0, name='Header')
            fo0.create_dataset('bin_edges', _bins)
            fo0.create_dataset('histogram', hist)
    
    if filens[1] is not None:
        with h5py.File(filens[1], 'r') as f1:
            if hasattr(bins[0], '__len__'):
                _bins = bins[1]
            else:
                _bins = bins
            _bins = list(bins)
            if _bins[0] != -np.inf:
                _bins = [-np.inf] + _bins
            if _bins[-1] != np.inf:
                _bins = _bins + [np.inf]
            _bins = np.array(_bins)
            
            vals = f1['map'][:].flatten()
            # max size ~32000**2 * 4 is good enough; for size maps, should be fine
            if len(vals) > 32000**2 * 4:
                print('Attempting a histogram with a {}-element array'.format(\
                      len(vals)))
            hist, edges = np.histogram(vals, bins=_bins)
     
            fparts = filens[1].split('/')
            fon1 = ol.pdir + '/cddf_' + fparts[-1] 
            with h5py.File(fon1, 'a') as fo1:
                f1.copy('Header', fo1, name='Header')
                fo1.create_dataset('bin_edges', _bins)
                fo1.create_dataset('histogram', hist)
    
    if run:
        os.remove(filens[0])
        if filens[1] is not None:
            os.remove(filens[1])
    

def get_names_and_pars(mapslices, *args, **kwargs):
    '''
    get the files names and arguments for each slice in make_maps_v3_master

    Parameters
    ----------
    mapslices : int
        number of slices to divide the region in *args into along the line of 
        sight.
    *args : 
        arguments for make_maps_v3_master.make_map (region before division
        into mapslices).
    **kwargs :
        key word arguments for make_maps_v3_master.make_maps. saveres and hdf5
        are always set to True, nameonly is ignored

    Returns
    -------
    names: list of tuples of strings
        names of the hdf5 files created (in order of slices)
    args: list of lists
        arguments to input to make_map (in order of slices)
    kwargs: list of dicts
        key-word arguments for make_map (in order of slices)
    '''
    
    _kwargs = kwargs.copy()
    _kwargs['saveres'] = True
    _kwargs['hdf5'] = True
    if 'nameonly' in _kwargs:
        del _kwargs['nameonly']
    if 'axis' not in _kwargs:
        _kwargs['axis'] = 'z'
    
    if not isinstance(mapslices, int) and mapslices > 0:
        raise ValueError('mapslices should be a positive integer; was' +\
                         ' {}'.format(mapslices))
    
    kwargslist = [_kwargs] * mapslices
    
    argslist = []
    nameslist = []
    
    if mapslices == 1:
        argslist.append(args)
        names = m3.make_map(*args, nameonly=True, **kwargs)
        nameslist.append(names)
    else:
        simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, ptypeW = args
        # generate per-slice args and names
        if 'axis' in _kwargs:
            axis = _kwargs['axis']
        else: # use make_map default 
            axis = 'z'
    
        if axis == 'z':
            slind = 2
            slext = L_z
            L_z /= float(mapslices)
        elif axis == 'y':
            slind = 1
            slext = L_y
            L_y /= float(mapslices)  
        elif axis == 'x':
            slind = 0
            slext = L_x
            L_x /= float(mapslices)
            
        for i in range(mapslices):
            _centre = np.copy(centre)
            _centre[slind] = centre[slind] + \
                             (0.5 / float(mapslices) - 0.5) * slext + \
                             float(i) / float(mapslices) * slext
    
            _args = (simnum, snapnum, _centre, L_x, L_y, L_z,
                     npix_x, npix_y, ptypeW)
            argslist.append(_args)
            
            names = m3.make_map(*_args, nameonly=True, **kwargs)
            nameslist.append(names)
            
    return nameslist, argslist, kwargslist
    

def create_histset(bins, args, kwargs, mapslices=1,
                   deletemaps=False, kwargs_hist=[{}]):
    '''
    create a histogram from a set of maps (slices); does not use weighted maps
    

    Parameters
    ----------
    bins : array-like of floats (1D)
        bin edges for the histogram
    args : tuple 
        arguments for make_maps_v3_master.make_map
    kwargs : dict
        key word arguments for make_maps_v3_master.make_map. saveres and hdf5
        are always set to True, nameonly is ignored
    mapslices : int, optional
        number of slices to divide the volume in for projections; works like 
        make_map numslices argument, but only uses less memory if read_region
        can be used in read_eagle. The default is 1.
    kwargs_hist : list of dictionaries 
        list of arguments for makehist_cddf_sliceadd
        add : int, optional
            how many slices to add together before histogramming. 
            The default is 1.
        addoffset : int, optional
            for values add =/= 1, the first slice index in the first added set. 
            The default is 0.
        resreduce : int, optional
            Factor by which to reduce the resolution of the map before 
            histogramming. Must divide the number of pixels in each dimension. The
            default is 1.
        includeinf : bool, optional
            Check if the left- and rightmost bin edges are -/+ infinity, and add 
            those values if not. The default is True.
    deletemaps : bool, optional
        delete the created maps after making the histograms. Pre-existing maps
        are left alone. The default is False.

    Returns
    -------
    None.

    Note
    ----
    even in deletemaps is set, only the 'W' maps are deleted, since maps of 
    weighted quantities are ignored. 
    '''
    
    
    nameslist, argslist, kwargslist = get_names_and_pars(mapslices,\
                                                         *args, **kwargs)
    already_exists = []
    _nameslist = []    
    for name, _args, _kwargs in zip(nameslist, argslist, kwargslist):
        name = name[0]
        _nameslist.append(name)
        if os.path.isfile(name):
            already_exists.append(True)
        else:
            already_exists.append(False)
            print('Creating file: {}'.format(name))
            print('-'*80)
            m3.make_map(*_args, nameonly=False, **kwargs)
            print('-'*80)
            print('\n'*3)
    
    if len(_nameslist) == 1:
        fills = None
        filebase = _nameslist[0]
    else:
        _base = _nameslist[0]
        keyword = '{}cen'.format(kwargslist[0]['axis'])
        pathparts = _base.split('/')
        nameparts = pathparts[-1].split('_')
        index = np.where([keyword in part for part in nameparts])[0][0]
        filebase = '_'.join(nameparts[:index] +\
                            [keyword + '{}'] +\
                            nameparts[index + 1:])
        filebase = '/'.join(pathparts[:-1] + [filebase])
        
        fills = []
        for name in _nameslist:
            pathparts = name.split('/')
            nameparts = pathparts[-1].split('_')
            index = np.where([keyword in part for part in nameparts])[0][0]
            part = nameparts[index]
            fill = part[len(keyword):]            
            fills.append(fill)
    
    if isinstance(kwargs_hist, dict):
        kwargs_hist = [kwargs_hist]
    for hkwargs in kwargs_hist:
        mh.makehist_cddf_sliceadd(filebase, fills=fills, bins=bins, 
                                  outname=None, **hkwargs)
    
    if deletemaps:
        for preexisting, name in zip(already_exists, _nameslist):
            if not preexisting:
                os.remove(name)

def rungrids_emlines(index):
    '''
    generate the histograms for line emission line convergence tests

    Parameters
    ----------
    index : int
        which set of histograms to generate, in the range [0, 35]. The fast 
        index sets the box to run (100 cMpc or the others), the slow index 
        sets which line is run (the paper 3 set + N VI (f)).

    Returns
    -------
    None.

    '''
    
    lines = ['c5r', 'n6-actualr', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12',
             'si13r', 'fe18', 'fe17-other1', 'fe19', 'o7r', 'o7iy', 'o7f',
             'o8', 'fe17', 'c6', 'n7']
    
    lineind = index // 2
    line = lines[lineind]
    
    simset = index % 2
    if simset == 0:
        simnums = ['L0100N1504']
        varlist = ['REFERENCE']
        npix = [32000]
        centres = [[50.] * 3]
        mapslices = [16]
        kwargs_hist = [{'add': 1, 'addoffset':0, 'resreduce':1},
                       {'add': 2, 'addoffset':0, 'resreduce':1},
                       {'add': 4, 'addoffset':0, 'resreduce':1},
                       {'add': 8, 'addoffset':0, 'resreduce':1},
                       {'add': 1, 'addoffset':0, 'resreduce':2},
                       {'add': 1, 'addoffset':0, 'resreduce':4},
                       {'add': 1, 'addoffset':0, 'resreduce':8},
                       ]
    elif simset == 1:
        simnums = ['L0050N0752', 'L0025N0376',
                   'L0025N0752', 'L0025N0752']
        varlist = ['REFERENCE', 'REFERENCE',\
                   'REFERENCE', 'RECALIBRATED']
        npix = [16000, 8000, 8000, 8000]
        centres = [[25.] * 3] + [[12.5] * 3] * 3
        mapslices = [8, 4, 4, 4]
        kwargs_hist = [{'add': 1, 'addoffset':0, 'resreduce':1}]
        
    
    snapnum = 27
    centre = [50., 50., 50.]
    ptypeW = 'emission'
    
    kwargs = {'abundsW': 'Sm', 'excludeSFRW': True, 'ptypeQ': None,
              'axis': 'z', 'periodic': True, 'kernel': 'C2',
              'log': True, 'saveres': True, 'hdf5': True,
              'simulation': 'eagle', 'ompproj': True,
              }
    bins = np.array([-np.inf] + list(np.arange(-50., 10.1, 0.1)) + [np.inf])
    
    for simnum, var, _npix, centre, _mapslices in\
        zip(simnums, varlist, npix, centres, mapslices):
        
        L_x, L_y, L_z = (centre[0],) * 3
        npix_x, npix_y = (_npix,) * 2
        args = (simnum, snapnum, centre, L_x, L_y, L_z,
                npix_x, npix_y, ptypeW)
        kwargs['var'] = var
        kwargs['ionW'] = line
        
        print('\n')
        print(args)
        print(kwargs)
        #print(bins)
        print('mapslices: {}'.format(_mapslices))
        print(kwargs_hist)
        
        #create_histset(bins, args, kwargs, mapslices=_mapslices,
        #               deletemaps=True, kwargs_hist=kwargs_hist)
        
if __name__ == '__main__':
    index = int(sys.argv[1])
    if not 'OMP_NUM_THREADS' in os.environ:
        raise RuntimeError('OMP_NUM_THREADS environment variable needs to be set')
    
    if index >=0 and index < 36:
        rungrids_emlines(index)
    