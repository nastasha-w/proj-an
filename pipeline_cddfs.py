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
import numpy as np
import h5py

import make_maps_opts_locs as ol
import make_maps_v3_master as m3


def create_cddf_singleslice(bins, *args, **kwargs):
    '''
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
    
def getargs(arglist):
    '''
    

    Parameters
    ----------
    arglist : list
        argument list from the command line ()

    Returns
    -------
    None.

    '''
    
    
    make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW=None, abundsW='auto', quantityW=None,\
         ionQ=None, abundsQ='auto', quantityQ=None, ptypeQ=None,\
         excludeSFRW=False, excludeSFRQ=False, parttype='0',\
         theta=0.0, phi=0.0, psi=0.0, \
         sylviasshtables=False, bensgadget2tables=False,\
         var='auto', axis='z',log=True, velcut=False,\
         periodic=True, kernel='C2', saveres=False,\
         simulation='eagle', LsinMpc=None,\
         select=None, misc=None, halosel=None, kwargs_halosel=None,\
         ompproj=False, nameonly=False, numslices=None, hdf5=False,\
         override_simdatapath=None)

