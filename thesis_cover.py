#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:24:52 2021

@author: Nastasha


create and plot the maps for my thesis cover
"""

import os
import numpy as np
import h5py

import matplotlib.pyplot as plt


import make_maps_v3_master as m3

mdir = '/path/to/save/imgs/'
m3.ol.ndir = '/path/to/save/maps/' # save map files

# check!
minvals_em = {'o7r': -1.8,
              'o8':  -1.8}
minvals_abs = {'o7': 15.5,
               'o8': 15.6}


def getmaps(ion, line, region_cMpc, axis, pixsize_regionunits, nameonly=False):
    
    simnum = 'L0100N1504'
    snapnum = 27
    centre = [0.5 * (region_cMpc[0] + region_cMpc[1]), 
              0.5 * (region_cMpc[2] + region_cMpc[3]),
              0.5 * (region_cMpc[4] + region_cMpc[5])]
    L_x = region_cMpc[1] - region_cMpc[0]
    L_y = region_cMpc[3] - region_cMpc[2]
    L_z = region_cMpc[5] - region_cMpc[4]
    
    if axis == 'z':
        _npix_x = L_x / pixsize_regionunits
        _npix_y = L_y / pixsize_regionunits
    if axis == 'x':
        _npix_x = L_y / pixsize_regionunits
        _npix_y = L_z / pixsize_regionunits
    if axis == 'y':
        _npix_x = L_z / pixsize_regionunits
        _npix_y = L_x / pixsize_regionunits
    npix_x = int(_npix_x + 0.5)
    npix_y = int(_npix_y + 0.5)
    if not (np.isclose(npix_x, _npix_x) and np.isclose(npix_y, _npix_y)):
        msg = 'The region size should be an integer multiple of the'+\
              ' pixel size.'
        raise ValueError(msg)
    
    args_all = (simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y)
    args_var = [('coldens',), ('emission',), ('basic',)]
    
    kwargs_all = {'excludeSFRW': 'T4', 'excludeSFRQ': 'T4', 'parttype': 0,
              'axis': axis, 'var': 'REFERENCE', 'periodic': False,
              'saveres': True, 'hdf5': True, 'simulation': 'EAGLE',
              'ompproj': True}
    
    kwargs_var = [{'ionW': ion, 'abundsW': 'Pt'},
                  {'ionW': line, 'abundsW': 'Sm'},
                  {'ionW': None, 'quantityW': 'Mass', 'ptypeQ': 'basic',
                   'quantityQ': 'Temperature'},
                  ]
    
    outnames = []
    for _args, _kwargs in zip(args_var, kwargs_var):
        args = args_all + _args
        kwargs = kwargs_all.copy()
        kwargs.update(_kwargs)
        
        outname = m3.make_map(*args, nameonly=True, **kwargs)   
        outnames.append(outname)             
        if os.path.isfile(outname):
            continue # already have it -> done
        if not nameonly:
            m3.make_map(*args, nameonly=False, **kwargs)
    return outnames
            
# make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y,
#          ptypeW,
#          ionW=None, abundsW='auto', quantityW=None,
#          ionQ=None, abundsQ='auto', quantityQ=None, ptypeQ=None,
#          excludeSFRW=False, excludeSFRQ=False, parttype='0',
#          theta=0.0, phi=0.0, psi=0.0,
#          sylviasshtables=False, bensgadget2tables=False,
#          ps20tables=False, ps20depletion=True,
#          var='auto', axis='z',log=True, velcut=False,
#          periodic=True, kernel='C2', saveres=False,
#          simulation='eagle', LsinMpc=None,
#          select=None, misc=None, halosel=None, kwargs_halosel=None,
#          excludedirectfb=False, deltalogT_directfb=0.2, 
#          deltatMyr_directfb=10., inclhotgas_maxlognH_snfb=-2.,
#          logTK_agnfb=8.499, logTK_snfb=7.499,
#          ompproj=False, nameonly=False, numslices=None, hdf5=False,
#          override_simdatapath=None)

def plotmaps(ion, line, region_cMpc, axis, pixsize_regionunits):
    files = getmaps(ion, line, region_cMpc, axis, pixsize_regionunits)
    
    cdfile = files[0][0]
    emfile = files[1][0]
    mtfiles = files[2]
    
    dynrange = 7.
    
    with h5py.File(cdfile, 'r') as f:
        cdmap = f['map']
        
    