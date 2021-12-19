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
import matplotlib as mpl

import plot_utils as pu
import make_maps_v3_master as m3

# luttero:
mdir = '/net/luttero/data2/imgs/pretty/thesis_cover/imgs/'
m3.ol.ndir = '/net/luttero/data2/imgs/pretty/thesis_cover/maps/' # save map files
# laptop
#mdir = '/Users/Nastasha/phd/imgs/thesis_cover/'
#m3.ol.ndir = '/Users/Nastasha//phd/sim_maps/thesis_cover/'


# abs: Athena X_IFU EW_min = 0.18 eV
# em: Athena X-IFU 5 sigma for Delta Omega * Delta t = 1e7 arcmin**2 * s
#     (extreme exposure/binning)
# using 1e6 arcmin**2 * s: 
#minvals_em = {'o7r': -0.9, 'o8': -0.9}
# using 1e7 arcmin**2 * s:
minvals_em = {'o7r': -1.6,
              'o8':  -1.5}
minvals_abs = {'o7': 15.4,
               'o8': 15.6}

region1 = [ 62.5, 72.0, 73.0,  90.0, 87.5,  93.75]
region2 = [ 52.5, 72.0, 73.0,  90.0, 87.5,  93.75]
region3 = [ 86.0, 96.0, 48.5,  59.0,  0.0,  6.25 ]
region4 = [ 22.0, 40.0, 60.0,  72.0,  6.25, 12.5 ]
region5 = [ 40.0, 73.0, -5.0,  35.0, 12.5,  18.75]
region6 = [ 70.0, 86.0, 86.0, 106.0, 37.5,  43.75]
region7 = [-18.0, 39.0, 84.0,  92.0, 43.75, 50.0 ]
region8 = [ -7.0, 15.0, 50.0,  67.0, 50.0,  56.25]


region_default = region1
ion_default = 'o8'
line_default = 'o8'
axis_default = 'z'
pixsize_regionunits_default = 0.0125 # 800 pixels for a 10 cMpc 

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
    args_var = [('coldens',), ('emission',), ('basic',), ('basic',), ('basic',),
                ('basic',)]
    
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

def readmap(filen):
    with h5py.File(filen, 'r') as f:
        _map = f['map'][:]
        map_max = f['map'].attrs['max']
        map_min = f['map'].attrs['minfinite']
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
        axn1 = ['x', 'y', 'z'][axis1]
        center = f['Header/inputpars'].attrs['centre']
        cen0 = center[axis0]
        cen1 = center[axis1]
        L0 = f['Header/inputpars'].attrs['L_{}'.format(axn0)]
        L1 = f['Header/inputpars'].attrs['L_{}'.format(axn1)]
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
        # NaN values outside selected region
        basemap = np.zeros(map.shape, dtype=map.dtype) / 0.
        basemap[sel] = map[sel]
        basemap[sel][np.isnan(basemap[sel])] = -np.inf
        #if axis == 'y':
        #    subext = list(extent)
        #    subext[0] = extent[0] + pixmin * (extent[1] - extent[0]) / float(xpix)
        #    subext[1] = extent[0] + pixmax * (extent[1] - extent[0]) / float(xpix)
        #elif axis == 'x':
        #    subext = list(extent)
        #    subext[2] = extent[2] + pixmin * (extent[3] - extent[2]) / float(ypix)
        #    subext[3] = extent[2] + pixmax * (extent[3] - extent[2]) / float(ypix)
        #subext = tuple(subext)
        ax.imshow(basemap.T, extent=extent, **kwargs_imshow)

## from https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
rgb_brightness_weights = np.array([0.299, 0.587, 0.144])
def brightness_score(RGB):
    sw = (np.newaxis,) * (len(RGB.shape) - 1) + (slice(None, None, None),)
    return np.sqrt(np.sum(rgb_brightness_weights[sw] \
                          * RGB**2, axis=len(RGB.shape) - 1))
    
def rescale_RGB_tobrightness(rgb, score):
    _rgb = np.zeros(rgb.shape, dtype=rgb.dtype)
    wR = rgb_brightness_weights[0]
    wG = rgb_brightness_weights[1]
    wB = rgb_brightness_weights[2]
    
    _s = (slice(None, None, None),) * (len(rgb.shape) - 1)
    s0 = _s + (0,) 
    s1 = _s + (1,) 
    s2 = _s + (2,)
     
    GoverR = rgb[s1] / rgb[s0]
    BoverR = rgb[s2] / rgb[s0] 
    _rgb[s0] = score / np.sqrt(wR + BoverR**2 * wB + GoverR**2 * wG)
    _rgb[s1] = _rgb[s0] * GoverR
    _rgb[s2] = _rgb[s0] * BoverR 
    
    R0 = np.where(rgb[s0] == 0.)
    if len(R0[0]) > 0: 
        BoverG = rgb[s2] / rgb[s1]
        subsel = hasattr(_rgb[s0], 'shape')
        if subsel:
            subsel = len(_rgb[s0].shape) > 0
             
        if subsel:  # single rgb value -> any R0 == all R0
            _rgb[s0][R0] = 0.
            _rgb[s1][R0] = score[R0] / np.sqrt(wB + BoverG[R0]**2 * wG)
            _rgb[s2][R0] = _rgb[s1][R0] * BoverG[R0]       
        else:
            _rgb[s0] = 0.
            _rgb[s1] = score / np.sqrt(wB + BoverG**2 * wG)
            _rgb[s2] = _rgb[s1] * BoverG     
         
    RG0 = np.where(np.logical_and(rgb[s0] == 0.,
                                  rgb[s1] == 0.))
    if len(RG0[0]) > 0: 
        subsel = hasattr(_rgb[s0], 'shape')
        if subsel:
            subsel = len(_rgb[s0].shape) > 0
            
        if subsel:
            _rgb[s1][RG0] = 0. 
            _rgb[s2][RG0] = score[RG0] / np.sqrt(wB) 
        else:
            _rgb[s1] = 0. 
            _rgb[s2] = score / np.sqrt(wB) 
    return _rgb
    
def equalize_brightness(rgb1, rgb2, step=0.95):
    #print(rgb1.shape, rgb2.shape)
    bs1 = brightness_score(rgb1)
    bs2 = brightness_score(rgb2)
    #print(bs1.shape, bs2.shape)
    bstarget = bs2
    loopcount = 0
    while (not np.allclose(bs1, bs2, rtol=1e-2, atol=1e-3)) \
           or np.max(rgb1) > 1. or np.max(rgb2) > 1.:
        rgb1 = rescale_RGB_tobrightness(rgb1, bstarget)
        rgb2 = rescale_RGB_tobrightness(rgb2, bstarget)   
        bs1 = brightness_score(rgb1)
        bs2 = brightness_score(rgb2)
        #print(bs1.shape, bs2.shape)
        bstarget *= step  
        #print('equalize_brightness loop {}; target {}'.format(loopcount, 
        #                                                      bstarget))
        #print(rgb1, rgb2)
        if loopcount > 100:
            raise RuntimeError('equalize_brightness not converging')
        loopcount += 1
    return rgb1, rgb2

def brightness_rescale(cmap, cmap_brightness, n=-1):
    if n == -1:
        n = cmap.N
    out_colorlist = cmap(np.linspace(0., 1., n))[..., :3] 
    target_colorlist = cmap_brightness(np.linspace(0., 1., n))[..., :3] 
    out_colorlist, _ = equalize_brightness(out_colorlist, target_colorlist, 
                                           step=0.95)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'rescale({name},{target})'.format(name=cmap.name, 
                                           target=cmap_brightness.name),
         out_colorlist)
    return new_cmap    

def plotmaps(ion, line, region_cMpc, axis, pixsize_regionunits,
             subregion=None, outnames=None):
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
    
    # some issues with system- or python-version-dependent precision of select cut-off values
    striplocs = None
    if np.all(region_cMpc == region1) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_T4EOS.hdf5',
                   m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
                        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
                       m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_760pix_6.25slice_zcen90.625_x67.25-pm9.5_y81.5-pm17.0_z-projection_wiEOS.hdf5'
        striplocs = [75., 82.3, 83.] 
    
    elif np.all(region_cMpc == region2) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_1560pix_6.25slice_zcen90.625_x62.25-pm19.5_y81.5-pm17.0_z-projection_wiEOS.hdf5'

    elif np.all(region_cMpc == region3) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_800pix_6.25slice_zcen3.125_x91.0-pm10.0_y53.75-pm10.5_z-projection_wiEOS.hdf5'
        
    elif np.all(region_cMpc == region4) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_1440pix_6.25slice_zcen9.375_x31.0-pm18.0_y66.0-pm12.0_z-projection_wiEOS.hdf5'
    
    elif np.all(region_cMpc == region5) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_2640pix_6.25slice_zcen15.625_x56.5-pm33.0_y15.0-pm40.0_z-projection_wiEOS.hdf5'        

    elif np.all(region_cMpc == region6) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_1280pix_6.25slice_zcen40.625_x78.0-pm16.0_y96.0-pm20.0_z-projection_wiEOS.hdf5'

    elif np.all(region_cMpc == region7) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:        
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_4560pix_6.25slice_zcen46.875_x10.5-pm57.0_y88.0-pm8.0_z-projection_wiEOS.hdf5'

    elif np.all(region_cMpc == region8) and ion == ion_default and \
        line == line_default and axis == axis_default and \
        pixsize_regionunits == pixsize_regionunits_default:   
        cdfile = m3.ol.ndir + 'coldens_o8_L0100N1504_27_test3.7_PtAb_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_T4EOS.hdf5'
        emfile = m3.ol.ndir + 'emission_o8_L0100N1504_27_test3.7_SmAb_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_T4EOS.hdf5'
        mtfiles = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_T4EOS.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection.hdf5')
        mtfiles_cool = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_partsel_Temperature_T4EOS_min-None_max-316227.766017_endpartsel.hdf5')
        mtfiles_hot = (m3.ol.ndir + 'Mass_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_T4EOS_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5',
        m3.ol.ndir + 'Temperature_T4EOS_Mass_T4EOS_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_partsel_Temperature_T4EOS_min-316227.766017_max-None_endpartsel.hdf5')
        stfile = m3.ol.ndir + 'Mass_PartType4_L0100N1504_27_test3.7_C2Sm_1760pix_6.25slice_zcen53.125_x4.0-pm22.0_y58.5-pm17.0_z-projection_wiEOS.hdf5'

    else:
        files = getmaps(ion, line, region_cMpc, axis, pixsize_regionunits)
        cdfile = files[0][0]
        emfile = files[1][0]
        mtfiles = files[2]
        mtfiles_cool = files[3]
        mtfiles_hot  = files[4]
        stfile = files[5][0]
        striplocs = None
        
        ## copy for future use:
        sp = ' ' * 8
        print(sp + "cdfile = m3.ol.ndir + '{}'".format(cdfile.split('/')[-1]))
        print(sp + "emfile = m3.ol.ndir + '{}'".format(emfile.split('/')[-1]))
        _st = "mtfiles = (m3.ol.ndir + '{}',\n{sp}m3.ol.ndir + '{}')"
        print(sp + _st.format(mtfiles[0].split('/')[-1], 
                              mtfiles[1].split('/')[-1], sp=sp))
        _st = "mtfiles_cool = (m3.ol.ndir + '{}',\n{sp}m3.ol.ndir + '{}')"
        print(sp + _st.format(mtfiles_cool[0].split('/')[-1], 
                              mtfiles_cool[1].split('/')[-1], sp=sp))
        _st = "mtfiles_hot = (m3.ol.ndir + '{}',\n{sp}m3.ol.ndir + '{}')"
        print(sp + _st.format(mtfiles_hot[0].split('/')[-1], 
                              mtfiles_hot[1].split('/')[-1], sp=sp))
        print(sp + "stfile = m3.ol.ndir + '{}'".format(stfile.split('/')[-1]))
        
    dynrange_gas = 4.
    nonobsrange = 6.
    
    cdmap, cd_min, cd_max, cdext = readmap(cdfile)
    emmap, em_min, em_max, emext = readmap(emfile)
    mcmap, mc_min, mc_max, mcext = readmap(mtfiles_cool[0])
    tcmap, tc_min, tc_max, tcext = readmap(mtfiles_cool[1])
    mhmap, mh_min, mh_max, mhext = readmap(mtfiles_hot[0])
    thmap, th_min, th_max, thext = readmap(mtfiles_hot[1])
    stmap, st_min, st_max, stext = readmap(stfile)
    if striplocs is None:
        yrange = stext[3] - stext[2]
        striplocs = list(np.linspace(stext[2] + 0.1 * yrange, 
                                     stext[3] - 0.1 * yrange,
                                     4))
    
    xovery = (cdext[1] - cdext[0]) / (cdext[3] - cdext[2])
    gridspec_kw = {'top': 1., 'bottom': 0., 'left': 0., 'right': 1.}
    obsfig, obsax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 5.5 / xovery),
                                 gridspec_kw=gridspec_kw)
    gasfig, gasax = plt.subplots(nrows=1, ncols=1, figsize=(5.5, 5.5 / xovery),
                                 gridspec_kw=gridspec_kw)
    
    cd_cmap = pu.paste_cmaps(['gist_gray', 'inferno'], 
                             [max(cd_min, minvals_abs[ion] - nonobsrange), 
                             minvals_abs[ion], cd_max],
                             trunclist=[[0., 0.5], [0.2, 0.85]],
                             transwidths=[0.15])
    cd_cmap.set_bad((0., 0., 0., 0.)) # transparent outside plotted strips
    #cd_cmap.set_under(cd_cmap(0.))
    name_lo = 'bone'
    name_hi = 'plasma' #'plasma'
    sub_lo = [0., 0.5]
    sub_hi = [0.1, 1.]
    #map_lo = mpl.cm.get_cmap(name_lo)
    #map_hi = mpl.cm.get_cmap(name_hi)
    #em_cmap = pu.paste_cmaps([name_lo, name_hi], 
    #                         [max(em_min, minvals_em[line] - nonobsrange), 
    #                          minvals_em[line], em_max],
    #                         trunclist=[sub_lo, sub_hi],
    #                         transwidths=[0.3])
    #em_cmap = brightness_rescale(em_cmap, map_lo, n=-1)
    brightness_map = mpl.cm.get_cmap('bone')
    map_lo = mpl.cm.get_cmap(name_lo)
    map_hi = mpl.cm.get_cmap(name_hi)
    minshow = max(em_min, minvals_em[line] - nonobsrange)
    midpoint = (em_max - minvals_em[line]) / (em_max - minshow)
    brightness_hi = pu.truncate_colormap(brightness_map, minval=midpoint,
                                         maxval=1.)
    brightness_rescale(map_hi, brightness_map)
    print([minshow, minvals_em[line], em_max])
    em_cmap = pu.paste_cmaps([map_lo, map_hi], 
                             [minshow, minvals_em[line], em_max],
                             trunclist=[sub_lo, sub_hi],
                             transwidths=[0.5])
    #em_cmap = brightness_rescale(em_cmap, map_lo, n=-1)
    em_cmap.set_bad((0., 0., 0., 1.))
    em_cmap.set_under(em_cmap(0.))
    
    ## alpha layer mixing
    #coolvals = np.zeros(mcmap.shape + (4,), dtype=np.float32)
    #m_max = max(mc_max, mh_max)
    #m_min = m_max - dynrange
    #coolvals[:, :, 2] = (np.maximum(mcmap, m_min) - m_min) / (m_max - m_min)
    #coolvals[:, :, 3] = 0.7 * (np.maximum(mcmap, m_min) - m_min) / (m_max - m_min)
    #
    #hotvals = np.zeros(mcmap.shape + (4,), dtype=np.float32)
    #hotvals[:, :, 0] = (np.maximum(mhmap, m_min) - m_min) / (m_max - m_min)
    #hotvals[:, :, 3] = 0.7 * (np.maximum(mhmap, m_min) - m_min) / (m_max - m_min)
    
    ## equal footing hot/cool mixing
    color_h = np.array([1., 0., 0.])
    color_c = np.array([0., 0., 1.])
    color_h1, color_c1 = equalize_brightness(color_h, color_c, step=0.95)
    color_c2, color_h2 = equalize_brightness(color_c, color_h, step=0.95)
    if brightness_score(color_h1) > brightness_score(color_h2):
        color_h = color_h1
        color_c = color_c1
    else:
        color_h = color_h2
        color_c = color_c2
    
    gas_map = np.zeros(mcmap.shape + (4,), dtype=np.float32)
    totvals = np.log10(10**mcmap + 10**mhmap)
    m_max = np.max(totvals)
    m_min = np.maximum(np.min(totvals[np.isfinite(totvals)]), 
                       m_max - dynrange_gas)
    msub_max = max(mc_max, mh_max)
    msub_min = msub_max - dynrange_gas
    mcw = (mcmap - msub_min) / (msub_max - msub_min)
    mhw = (mhmap - msub_min) / (msub_max - msub_min)
    #print(mcw)
    #print(mhw)
    stretch = 1.
    wc = 0.5 * (np.tanh(stretch * (mcw - mhw)) + 1.)
    #print(wc)
    _gas_map = wc[:, :, np.newaxis] * \
                 color_c[np.newaxis, np.newaxis, :] \
               + (1. - wc[:, :, np.newaxis]) *\
                 color_h[np.newaxis, np.newaxis, :]
    totw = (np.maximum(totvals, m_min) - msub_min) / (m_max - msub_min)
    #tonorm = np.sum(gas_map[:, :, :3], axis=-1)
    #gas_map[:, :, :3] *= totw[:, :, np.newaxis] / tonorm[:, :, np.newaxis]
    
    ### formula off stackexchange: 
    ## brightness = sqrt(0.299 * R^2 + 0.587 * G^2 + 0.114 * B^2)
    ## likely to 'scramble' input colors other than R, G, and B
    ## creates very binary color look; might be implementation error
    #BoverR = _gas_map[:, :, 1] / _gas_map[:, :, 0]
    #GoverR = _gas_map[:, :, 2] / _gas_map[:, :, 0]
    #wR = 0.299
    #wG = 0.587
    #wB = 0.114
    #gas_map[:, :, 0] = totw / np.sqrt(wR + BoverR * wB + GoverR * wG)
    #gas_map[:, :, 1] = gas_map[:, :, 0] * GoverR
    #gas_map[:, :, 2] = gas_map[:, :, 0] * BoverR 
    #
    #R0 = np.where(_gas_map[:, :, 0] == 0.)
    #BoverG = _gas_map[:, :, 2] / _gas_map[:, :, 1]
    #gas_map[:, :, 0][R0] = 0.
    #gas_map[:, :, 1][R0] = totw[R0] / np.sqrt(wB + BoverG[R0] * wG)
    #gas_map[:, :, 2][R0] = gas_map[:, :, 1][R0] * BoverG[R0]
    #
    #RG0 = np.where(np.logical_and(_gas_map[:, :, 0] == 0.,
    #                              _gas_map[:, :, 1] == 0.))
    #gas_map[:, :, 1] = 0. 
    #gas_map[:, :, 2][RG0] = totw[RG0] / np.sqrt(wB) 
    #
    #tonorm = np.sqrt(0.299 * gas_map[:, :, 0]**2 +\
    #                 0.587 * gas_map[:, :, 1]**2 +\
    #                 0.114 * gas_map[:, :, 2]**2)
    
    ## try to 'equalize' colors:
    # gas_map sets channel ratios
    totv = totw[:, :, np.newaxis] * np.array([1., 1., 1.])
    gas_map[:, :, :3], _ = equalize_brightness(_gas_map, totv, step=0.95)
    gas_map[:, :, 3] = 1. #0.7 * totw 
    #print(gas_map)
    
    #st_cmap = pu.paste_cmaps(['gray'], 
    #                         [max(st_min, st_max - dynrange), st_max],
    #                         trunclist=[[0.5, 1.]])
    #st_cmap.set_under(st_cmap(0.))
    #st_cmap.set_bad(st_cmap(0.))
    #gasax.set_facecolor(st_cmap(0.))
    
    gasax.set_facecolor('black')
    _st_min = max(st_min, st_max - dynrange_gas)
    stv = (np.maximum(stmap, _st_min) - _st_min) / (st_max - _st_min)
    st_color = np.array([1., 1., 1.])
    star_map = np.zeros(stmap.shape + (4,), dtype=np.float32)
    star_map[:, :, :3] = st_color[np.newaxis, np.newaxis, :] # * stv[:, :, np.newaxis]
    star_map[:, :, 3] = 0.4 * stv
    
    gasax.imshow(gas_map.transpose(1, 0, 2), interpolation='nearest', 
                 origin='lower', extent=mhext)
    gasax.imshow(star_map.transpose(1, 0, 2), interpolation='nearest', 
                 origin='lower', extent=stext)
    #gasax.imshow(coolvals.transpose(1, 0, 2), interpolation='nearest', 
    #             origin='lower', extent=mcext)
    #gasax.imshow(hotvals.transpose(1, 0, 2), interpolation='nearest', 
    #             origin='lower', extent=mhext)
    gasax.axis('off')
    
    obsax.set_facecolor('black')
    #obsax.imshow(cdmap.T, interpolation='nearest', origin='lower',
    #             cmap=cd_cmap, extent=cdext)
    obsax.imshow(emmap.T, interpolation='nearest', origin='lower', 
                 cmap=em_cmap, extent=emext, 
                 vmin=minvals_em[line] - nonobsrange, vmax=em_max)                 
    plotstrips(obsax, cdmap, cdext, striplocs, axis='x',
               pixwidth=7, interpolation='nearest', origin='lower',
               cmap=cd_cmap)
    obsax.axis('off')
    
    if subregion is not None:
        obsax.set_xlim((subregion[0], subregion[1]))
        gasax.set_xlim((subregion[0], subregion[1]))
        obsax.set_xlim((subregion[2], subregion[3]))
        gasax.set_xlim((subregion[2], subregion[3]))
    
    if outnames is None:
        obsname = 'emission_absorption_map.pdf'
        gasname = 'gas_phase_map.pdf'
    else:
        obsname = outnames[0]
        gasname = outnames[1]
    obsfig.savefig(mdir + obsname)
    gasfig.savefig(mdir + gasname)
    
def plotdefaults(settings=1):
    region_default = None 
    _ion = ion_default
    _line = line_default
    _axis = axis_default 
    _pixsize = pixsize_regionunits_default
    if settings == 1:
        _region = region1
        _subregion = None
    elif settings == 2:
        _region = region2
        _subregion = None
    elif settings == 3:
        _region = region3
        _subregion = None
    elif settings == 4:
        _region = region4
        _subregion = None
    elif settings == 5:
        _region = region5
        _subregion = None
    elif settings == 6:
        _region = region6
        _subregion = None
    elif settings == 7:
        _region = region7
        _subregion = None
    elif settings == 8:
        _region = region8
        _subregion = None
    obsname = 'emission_absorption_map_region{}.pdf'.format(settings)
    gasname = 'gas_phase_map_region{}.pdf'.format(settings)
    plotmaps(_ion, _line, _region, _axis, _pixsize, 
             subregion=_subregion, outnames=(obsname, gasname))

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