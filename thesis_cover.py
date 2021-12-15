#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 17:24:52 2021

@author: Nastasha


create and plot the maps for my thesis cover
"""

import os
import sys
import numpy as np
import h5py

import matplotlib.pyplot as plt

import plot_utils as pu
import make_maps_v3_master as m3

mdir = '/net/luttero/data2/imgs/pretty/thesis_cover/imgs/'
m3.ol.ndir = '/net/luttero/data2/imgs/pretty/thesis_cover/maps/' # save map files

# abs: Athena X_IFU EW_min = 0.18 eV
# em: Athena X-IFU 5 sigma for Delta Omega * Delta t = 1e7 arcmin**2 * s
#     (extreme exposure/binning)
# using 1e6 arcmin**2 * s: 
minvals_em = {'o7r': -0.9, 'o8': -0.9}
# using 1e7 arcmin**2 * s:
#minvals_em = {'o7r': -1.6,
#              'o8':  -1.5}
minvals_abs = {'o7': 15.4,
               'o8': 15.6}

region1 = [62.5, 72., 73., 90., 87.5, 93.75]

region_default = region1
ion_default = 'o8'
line_default = 'o8'
axis = 'z'
pixsize_regionunits = 0.0125 # 800 pixels for a 10 cMpc 

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
    args_var = [('coldens',), ('emission',), ('basic',), ('basic'), ('basic')]
    
    kwargs_all = {'excludeSFRW': 'T4', 'excludeSFRQ': 'T4', 'parttype': 0,
                  'axis': axis, 'var': 'REFERENCE', 'periodic': False,
                  'saveres': True, 'hdf5': True, 'simulation': 'EAGLE',
                  'ompproj': True}
    
    kwargs_var = [{'ionW': ion, 'abundsW': 'Pt'},
                  {'ionW': line, 'abundsW': 'Sm'},
                  {'ionW': None, 'quantityW': 'Mass', 'ptypeQ': 'basic',
                   'quantityQ': 'Temperature'},
                  {'ionW': None, 'quantityW': 'Mass', 'ptypeQ': 'basic',
                   'quantityQ': 'Temperature', 
                   'select': [({'ptype': 'basic', 'quantity': 'Temperature'}, 
                               None, 10**5.5)]},
                  {'ionW': None, 'quantityW': 'Mass', 'ptypeQ': 'basic',
                   'quantityQ': 'Temperature', 
                   'select': [({'ptype': 'basic', 'quantity': 'Temperature'}, 
                               10**5.5, None)]},
                  {'quantityW': 'Mass', 'parttype': '4'}
                  ]
    
    outnames = []
    for _args, _kwargs in zip(args_var, kwargs_var):
        args = args_all + _args
        kwargs = kwargs_all.copy()
        kwargs.update(_kwargs)
        
        outname = m3.make_map(*args, nameonly=True, **kwargs)   
        outnames.append(outname)             
        if os.path.isfile(outname[0]):
            if outname[1] is None:
                continue # already have it -> done
            elif os.path.isfile(outname[1]):
                continue
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

def readmap(filen, dynrange=7.):
    _map = f['map'][:]
    map_max = f['map'].attrs['max']
    map_min = max(f['map'].attrs['minfinite'], cdmap_max - dynrange)
    axis = f['Header/inputpars'].attrs['axis'].decode()
    if axis == 'x':
        axis0 = 1
        axis1 = 2
    elif axis == 'y':
        axis0 = 2
        axis1 = 0
    elif axis == 'z':
        axis0 = 0
        axis1 = 1
    axn0 = ['x', 'y', 'z'][axis0]
    axn1 = ['x', 'y', 'z'][axis0]
    center = ['Header/inputpars'].attrs['centre']
    cen0 = center[axis0]
    cen1 = center[axis1]
    L0 = ['Header/inputpars'].attrs['L_{}'.format(axn0)]
    L1 = ['Header/inputpars'].attrs['L_{}'.format(axn1)]
    extent = (cen0 - 0.5 * L0, cen0 + 0.5 * L0,
              cen1 - 0.5 * L1, cen1 + 0.5 * L1)
    return _map, map_min, map_max, extent
    
def plotstrips(ax, map, extent, locations, axis='y',
               pixwidth=3, **kwargs_imshow):
    xpix, ypix = map.shape 
    for _loc in locations:
        if axis == 'y':
            pixcen = (_loc - extent[0]) / (extent[1] - extent[0]) * xpix
            selax = 0
        elif axis == 'x':
            pixcen = (_loc - extent[2]) / (extent[3] - extent[2]) * ypix
            selax = 1
        pixmin = int(pixcen - 0.5 * pixwidth + 0.5)
        pixmax = int(pixcen + 0.5 * pixwidth + 0.5)
        sel = [slice(None, None, None)] * 2
        sel[selax] = slice(pixmin, pixmax, None)
        sel = tuple(sel)
        if axis == 'y':
            subext = extent
            subext[0] = extent[0] + pixmin * (extent[1] - extent[0]) / float(xpix)
            subext[1] = extent[0] + pixmax * (extent[1] - extent[0]) / float(xpix)
        elif axis == 'x':
            subext = extent
            subext[2] = extent[2] + pixmin * (extent[3] - extent[2]) / float(ypix)
            subext[3] = extent[2] + pixmax * (extent[3] - extent[2]) / float(ypix)
        ax.imshow(map[sel].T, extent=subext, **kwargs_imshow)

def plotmaps(ion, line, region_cMpc, axis, pixsize_regionunits,
             subregion=None):
    '''
    ion:  str
        ion for the column density map
    line: str
        line for the emission map
    region_cMpc: list of 6 floats 
        Minimum and maximum along each simulation coordinate axis
        [xmin, xmax, ymin, ymax, zmin, zmax]
    axis: str
        projection axis; 'x', 'y', or 'z'
    pixsize_regionunits: float
        size of each pixel in the same units as the region.
    subregion: list of 4 floats
        plot only a subregion along the *map* x and y axes
        [xmin, xmax, ymin, ymax] 
             
    '''
    xsize = 5.5   
    pixwidth_strips = 3    
    striplocs = [71., 65.]  
             
    files = getmaps(ion, line, region_cMpc, axis, pixsize_regionunits)
    
    cdfile = files[0][0]
    emfile = files[1][0]
    mtfiles = files[2]
    mtfiles_cool = files[3]
    mtfiles_hot  = files[4]
    stfile = files[5][0]
    dynrange = 7.
    
    cdmap, cd_min, cd_max, cdext = readmap(cdfile, dynrange=dyrange)
    emmap, em_min, em_max, emext = readmap(emfile, dynrange=dyrange)
    mcmap, mc_min, mc_max, mcext = readmap(mtfiles_cool[0], dynrange=dyrange)
    tcmap, tc_min, tc_max, tcext = readmap(mtfiles_cool[1], dynrange=dyrange)
    mhmap, mh_min, mh_max, mhext = readmap(mtfiles_hot[0], dynrange=dyrange)
    thmap, th_min, th_max, thext = readmap(mtfiles_hot[1], dynrange=dyrange)
    stmap, st_min, st_max, stext = readmap(mtfiles_hot[1], dynrange=dyrange)
    
    xovery = (cdext[1] - cdext[0]) / (cdext[3] - cdext[2])
    obsfig, obsax = plt.subplots(1, 1, 1, figsize=(5.5, 5.5 / xovery))
    gasfig, gasax = plt.subplots(1, 1, 1, figsize=(5.5, 5.5 / xovery))
    
    cd_cmap = pu.paste_cmaps(['gist_yarg', 'inferno'], 
                             [cd_min, minvals_abs[ion], cd_max],
                             trunclist=[[0., 0.7], [0., 0.95]])
    em_cmap = pu.paste_cmaps(['bone_r', 'plasma'], 
                             [cd_min, minvals_abs[ion], cd_max],
                             trunclist=[[0., 0.7], [0., 1.]])
    
    coolvals = np.zeros(mcmap.shape + (4,), dtype=np.float32)
    coolvals[:, :, 2] = np.max(mcmap, mc_min) / (mc_max - mc_min)
    coolvals[:, :, 3] = 0.7 * np.max(mcmap, mc_min) / (mc_max - mc_min)
    
    hotvals = np.zeros(mcmap.shape + (4,), dtype=np.float32)
    hotvals[:, :, 0] = np.max(mhmap, mh_min) / (mh_max - mh_min)
    hotvals[:, :, 3] = 0.7 * np.max(mhmap, mh_min) / (mh_max - mh_min)
    
    gasax.set_facecolor('black')
    gasax.imshow(stmap.transpose(1, 0, 2), interpolation='nearest', 
                 origin='lower', extent=stext, cmap='gray')
    gasax.imshow(coolvals.transpose(1, 0, 2), interpolation='nearest', 
                 origin='lower', extent=mcext)
    gasax.imshow(hotvals.transpose(1, 0, 2), interpolation='nearest', 
                 origin='lower', extent=mhext)
    gasax.axis('off')
    
    obsax.set_facecolor('black')
    #obsax.imshow(cdmap.T, interpolation='nearest', origin='lower',
    #             cmap=cd_cmap, extent=cdext)
    obsax.imshow(emmap.T, interpolation='nearest', origin='lower', 
                 cmap=em_map, extent=emext)                 
    plotstrips(obsax, cdmap, cdext, striplocs, axis='y',
               pixwidth=3, interpolation='nearest', origin='lower',
                cmap=cd_cmap)
    obsax.axis('off')
    
    if subregion is not None:
        obsax.set_xlim((subregion[0], subregion[1]))
        gasax.set_xlim((subregion[0], subregion[1]))
        obsax.set_xlim((subregion[2], subregion[3]))
        gasax.set_xlim((subregion[2], subregion[3]))
        
    obsfig.savefig(mdir + 'emission_absorption_map.pdf')
    gasfig.savefig(mdir + 'gas_phase_map.pdf')
    
def plotdefaults(settings=1):
    region_default = None 
    _ion = ion_default
    _line = line_default
    _axis = axis 
    _pixsize = pixsize_regionunits
    if settings == 1:
        _region = region1
        _subregion = None
    plotmaps(_ion, _line, _region, _axis, _pixsize, 
             subregion=_subregion)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1:
        if args[0].startswith('--settings'):
            settings = int(args[0].split['='][-1])
        else:
            settings = int(args[0])
    else:
        settings=1
    plotdefaults(settings=settings)