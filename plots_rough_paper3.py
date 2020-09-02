#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:53:54 2020

@author: Nastasha
"""

import numpy as np
import pandas as pd
import h5py
import string

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.patches as mpatch
import matplotlib.ticker as ticker

import tol_colors as tc

import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu
import make_maps_opts_locs as ol

rho_to_nh = 0.752 / (c.atomw_H * c.u)
cosmopars_eagle = {'omegab': c.omegabaryon,\
                   'omegam': c.omega0,\
                   'omegalambda': c.omegalambda,\
                   'h': c.hubbleparam,\
                  }

res_arcsec = {'Athena X-IFU': 5.,\
              'Athena WFI':  3.,\
              'Lynx PSF':    1.,\
              'Lynx HDXI pixel':  0.3,\
              }
fov_arcmin = {'Athena X-IFU': 5.,\
              'Athena WFI':  40.,\
              'Lynx HDXI':  22.,\
              'Lynx X-ray microcalorimeter':  5.,\
              }

lines = ['c5r', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r', 'fe18',\
         'fe17-other1', 'fe19', 'o7r', 'o7ix', 'o7iy', 'o7f', 'o8', 'fe17',\
         'c6', 'n7']

lines = sorted(lines, key=ol.line_eng_ion.get)
nicenames_lines =  {'c5r': 'C V',\
                    'n6r': 'N VI',\
                    'ne9r': 'Ne IX',\
                    'ne10': 'Ne X',\
                    'mg11r': 'Mg XI',\
                    'mg12': 'Mg XII',\
                    'si13r': 'Si XIII',\
                    'fe18': 'Fe XVIII',\
                    'fe17-other1': 'Fe XVII (15.10 A)',\
                    'fe19': 'Fe XIX',\
                    'o7r': 'O VII (r)',\
                    'o7ix': 'O VII (x)',\
                    'o7iy': 'O VII (y)',\
                    'o7f': 'O VII (f)',\
                    'o8': 'O VIII',\
                    'fe17': 'Fe XVII (17.05 A)',\
                    'c6': 'C VI',\
                    'n7': 'N VII',\
                    }
linecolors = {'c5r':   'darkgoldenrod',\
              'c6':    'tan',\
              'n6r':   'mediumvioletred',\
              'n7':    'fuchsia',\
              'o7r':   'maroon',\
              'o7ix':  'firebrick',\
              'o7iy':  'red',\
              'o7f':   'tomato',\
              'o8':    'orange',\
              'ne9r':  'gold',\
              'ne10':  'yellow',\
              'mg11r': 'dodgerblue',\
              'mg12':  'aqua',\
              'si13r': 'navy',\
              'fe17':  'green',\
              'fe17-other1': 'olive',\
              'fe18':  'chartreuse',\
              'fe19': 'mediumspringgreen',\
              }
# if I want color + linestyle combinations; based on a color-blind friendly scheme
_c1 = tc.tol_cset('bright') # tip: don't use 'wine' and 'green' in the same plot (muted scheme)
lineargs =  {'c5r':  {'linestyle': 'solid',   'color': _c1.blue},\
             'c6':   {'linestyle': 'dashed',  'color': _c1.blue},\
             'n6r':  {'linestyle': 'solid',   'color': _c1.cyan},\
             'n7':   {'linestyle': 'dashed',  'color': _c1.cyan},\
             'o7r':  {'linestyle': 'solid',   'color': _c1.green},\
             'o7ix': {'linestyle': 'dashdot', 'color': _c1.green},\
             'o7iy': {'dashes': [6, 2],       'color': _c1.green},\
             'o7f':  {'linestyle': 'dotted',  'color': _c1.green},\
             'o8':   {'linestyle': 'dashed',  'color': _c1.green},\
             'ne9r':  {'linestyle': 'solid',  'color': _c1.yellow},\
             'ne10':  {'linestyle': 'dashed', 'color': _c1.yellow},\
             'mg11r': {'linestyle': 'solid',  'color': _c1.red},\
             'mg12':  {'linestyle': 'dashed', 'color': _c1.red},\
             'si13r': {'linestyle': 'solid',  'color': _c1.purple},\
             'fe17-other1':  {'dashes': [6, 2], 'color': _c1.grey},\
             'fe19':  {'dashes': [2, 2, 2, 2], 'color': _c1.grey},\
             'fe17':  {'linestyle': 'dotted',  'color': _c1.grey},\
             'fe18':  {'linestyle': 'dashdot', 'color': _c1.grey},\
              }
linesets = [['c5r', 'n6r', 'o7r', 'ne9r', 'mg11r', 'si13r'],\
            ['c6', 'n7', 'o8', 'ne10', 'mg12'],\
            ['o7r', 'o7ix', 'o7iy', 'o7f'],\
            ['fe17', 'fe17-other1', 'fe18', 'fe19'],\
            ]
lineargs_sets =\
            {'c5r':  {'linestyle': 'solid',   'color': _c1.blue},\
             'c6':   {'linestyle': 'dashed',  'color': _c1.blue},\
             'n6r':  {'linestyle': 'solid',   'color': _c1.cyan},\
             'n7':   {'linestyle': 'dashed',  'color': _c1.cyan},\
             'o7r':  {'linestyle': 'solid',   'color': _c1.green},\
             'o7ix': {'linestyle': 'dashdot', 'color': _c1.blue},\
             'o7iy': {'dashes': [6, 2],       'color': _c1.yellow},\
             'o7f':  {'linestyle': 'dotted',  'color': _c1.red},\
             'o8':   {'linestyle': 'dashed',  'color': _c1.green},\
             'ne9r':  {'linestyle': 'solid',  'color': _c1.yellow},\
             'ne10':  {'linestyle': 'dashed', 'color': _c1.yellow},\
             'mg11r': {'linestyle': 'solid',  'color': _c1.red},\
             'mg12':  {'linestyle': 'dashed', 'color': _c1.red},\
             'si13r': {'linestyle': 'solid',  'color': _c1.purple},\
             'fe17-other1':  {'dashes': [6, 2], 'color': _c1.green},\
             'fe19':  {'dashes': [2, 2, 2, 2], 'color': _c1.blue},\
             'fe17':  {'linestyle': 'dotted',  'color': _c1.yellow},\
             'fe18':  {'linestyle': 'dashdot', 'color': _c1.red},\
              }

for key in lineargs_sets:
    lineargs_sets[key].update({'dashes': None, 'linestyle': 'solid'})
    del lineargs_sets[key]['dashes']
    
# Tmax: direct values (0.05 dex spacing), copied by hand from table
line_Tmax = {'c5r':    5.95,\
              'c6':    6.15,\
              'n6r':   6.15,\
              'n7':    6.3,\
              'o7r':   6.3,\
              'o7ix':  6.35,\
              'o7iy':  6.35,\
              'o7f':   6.35,\
              'o8':    6.5,\
              'ne9r':  6.6,\
              'ne10':  6.8,\
              'mg11r': 6.8,\
              'mg12':  7.0,\
              'si13r': 7.0,\
              'fe17':  6.75,\
              'fe17-other1': 6.8,\
              'fe18':  6.9,\
              'fe19':  7.0,\
              }
# Trange: rounded to 0.1 dex, copied by hand from table
line_Trange = {'c5r':   (5.7, 6.3),\
               'c6':    (5.9, 6.8),\
               'n6r':   (5.9, 6.5),\
               'n7':    (6.1, 7.0),\
               'o7r':   (6.0, 6.7),\
               'o7ix':  (6.0, 6.7),\
               'o7iy':  (6.0, 6.7),\
               'o7f':   (6.0, 6.7),\
               'o8':    (6.2, 7.2),\
               'ne9r':  (6.3, 7.0),\
               'ne10':  (6.5, 7.5),\
               'mg11r': (6.4, 7.2),\
               'mg12':  (6.7, 7.8),\
               'si13r': (6.6, 7.4),\
               'fe17':  (6.4, 7.1),\
               'fe17-other1': (6.4, 7.1),\
               'fe18':  (6.6, 7.1),\
               'fe19':  (6.7, 7.2),\
               }

def getoutline(linewidth):
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"),\
               mppe.Stroke(linewidth=linewidth + 0.5, foreground="white"),\
               mppe.Normal()]
    return patheff

mass_edges_standard = (11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0)
fontsize = 12
mdir = '/net/luttero/data2/imgs/paper3/'

def getsamplemedianmass():
    mass_edges = mass_edges_standard
    galdataf = mdir + '3dprof/' + 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    galdata_all = pd.read_csv(galdataf, header=2, sep='\t', index_col='galaxyid')
    
    masses = np.log10(np.array(galdata_all['M200c_Msun']))
    minds = np.digitize(masses, mass_edges)
    meddct = {mass_edges[i]: np.median(masses[minds == i + 1])\
              for i in range(len((mass_edges)))}
    return meddct
# from getsamplemedianmass
medianmasses = {11.0: 11.197684653627299,\
                11.5: 11.693016261428347,\
                12.0: 12.203018243950218,\
                12.5: 12.69846038894407,\
                13.0: 13.176895227999415,\
                13.5: 13.666537415167888,\
                14.0: 14.235991474257528,\
                }

    
def add_cbar_mass(cax, massedges=mass_edges_standard,\
             orientation='vertical', clabel=None, fontsize=fontsize, aspect=10.):
    '''
    returns color bar object, color dictionary (keys: lower mass edges)
    '''
    massedges = np.array(massedges)
    clist = tc.tol_cmap('rainbow_discrete', lut=len(massedges))(np.linspace(0.,  1., len(massedges)))
    keys = sorted(massedges)
    colors = {keys[i]: clist[i] for i in range(len(keys))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation=orientation)
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(aspect)
    
    return cbar, colors

def get_resolution_tables(zvals=[0.01, 0.05, 0.1, 0.2]):
    ins = sorted(list(res_arcsec.keys()))
    nz = len(zvals)
    zvals = sorted(zvals)

    colhead_dec = 'r@{.}l'    
    tabstart = '\\begin{{tabular}}{{l {decl}}}'.format(\
                      decl=' '.join([colhead_dec] * (nz + 1)))
    tabend = '\\end{tabular}'
    hline = '\\hline'
    tabhead1 = '\\multicolumn{1}{c}{instrument} & ' + \
               '\\multicolumn{2}{c}{FWHM} & ' + \
               '\\multicolumn{{{nc}}}{{c}}{{$z$}} \\\\'.format(nc=2*nz)
    tabhead2 = ' & \\multicolumn{2}{c}{arcsec} &' +\
               ' & '.join(['\\multicolumn{{2}}{{c}}{{{z}}}'.format(\
                           z=z) for z in zvals]) +\
               '\\\\'
    tabline = '{inst} & {res} & ' + ' & '.join(['{}' for zval in zvals]) + \
              '\\\\'
    print(tabline)
    caption = '\\caption{Resolution of various instruments in pkpc at ' +\
         'different redshifts $z$. For the Athena instruments, ' + \
         'this is the half-energy width, for Lynx it is the half-power ' + \
         'diameter (PSF) or simply the pixel size.}'
    arcsec_to_rad = np.pi / (180. * 60. * 60.)
    sizes = {inst: {str(zval): cu.ang_diam_distance_cm(zval) \
                          / c.cm_per_mpc * 1e3 * \
                          res_arcsec[inst] * arcsec_to_rad \
                    for zval in zvals} \
             for inst in ins}
    print(sizes['Athena WFI'])
    sizetab = {inst: [('{:.1f}'.format(sizes[inst][zst])).replace('.', '&')
                      for zst in sizes[inst]] \
               for inst in ins}
    fmtres_arcsec = {inst: ('{:.1f}'.format(res_arcsec[inst])).replace('.', '&')\
                     for inst in ins}
    table = '\n'.join([tabline.format(*tuple(sizetab[inst]),\
                                      inst=inst, res=fmtres_arcsec[inst])\
                       for inst in ins])
    print(caption)
    print(tabstart)
    print(tabhead1)
    print(tabhead2)
    print(hline)
    print(table)
    print(tabend)

def get_fov_tables(zvals=[0.01, 0.05, 0.1, 0.2]):
    ins = sorted(list(fov_arcmin.keys()))
    nz = len(zvals)
    zvals = sorted(zvals)

    colhead_dec = 'r@{.}l'    
    tabstart = '\\begin{{tabular}}{{l r {decl}}}'.format(\
                      decl=' '.join([colhead_dec] * (nz )))
    tabend = '\\end{tabular}'
    hline = '\\hline'
    tabhead1 = '\\multicolumn{1}{c}{instrument} & ' + \
               '\\multicolumn{1}{c}{FOV} & ' + \
               '\\multicolumn{{{nc}}}{{c}}{{$z$}} \\\\'.format(nc=2*nz)
    tabhead2 = ' & \\multicolumn{1}{c}{arcmin} &' +\
               ' & '.join(['\\multicolumn{{2}}{{c}}{{{z}}}'.format(\
                           z=z) for z in zvals]) +\
               '\\\\'
    tabline = '{inst} & {fov} & ' + ' & '.join(['{}' for zval in zvals]) + \
              '\\\\'
    print(tabline)
    caption = '\\caption{Field of view (FOV) of various instruments in pkpc at ' +\
         'different redshifts $z$. ' +\
         'Calculations assume the same cosmological parameters as used in {\eagle}.}'
    arcmin_to_rad = np.pi / (180. * 60.)
    sizes = {inst: {str(zval): cu.ang_diam_distance_cm(zval) \
                          / c.cm_per_mpc * 1e3 * \
                          fov_arcmin[inst] * arcmin_to_rad \
                    for zval in zvals} \
             for inst in ins}
    print(sizes['Athena WFI'])
    sizetab = {inst: [('{:.1f}'.format(sizes[inst][zst])).replace('.', '&')
                      for zst in sizes[inst]] \
               for inst in ins}
    fmtfov_arcmin = {inst: ('{:.0f}'.format(fov_arcmin[inst])).replace('.', '&')\
                     for inst in ins}
    table = '\n'.join([tabline.format(*tuple(sizetab[inst]),\
                                      inst=inst, fov=fmtfov_arcmin[inst])\
                       for inst in ins])
    print(caption)
    print(tabstart)
    print(tabhead1)
    print(tabhead2)
    print(hline)
    print(table)
    print(tabend)
    
### stamp images from total maps
    
def reducedims(map_in, dims_out, weights_in=None):
    '''
    reduce the size of an image to shape dims_out, by taking average values in 
    the new pixels (weighted by weights_in, if given)
    '''
    
    shape_in = map_in.shape
    ndims = len(shape_in)
    interm_shape = [[dims_out[i], shape_in[i] // dims_out[i]] for i in range(ndims)]
    interm_shape = tuple([val for dub in interm_shape for val in dub])  
    axis = tuple([2 * i + 1 for i in range(ndims)])
      
    if weights_in is None:
        map_interm = map_in.reshape(interm_shape)
        norm = 1. / np.prod([shape_in[i] // dims_out[i] for i in range(ndims)])
        map_out = np.sum(map_interm, axis=axis) * norm
    else:
        map_interm = (map_in * weights_in).reshape(interm_shape)
        weights_interm = (weights_in).reshape(interm_shape)
        norm = 1. / np.sum(weights_interm, axis=axis)
        map_out = np.sum(map_interm, axis=axis) * norm
    return map_out

def make_and_save_stamps(filen_in, filen_weight=None,\
                         filen_out=None, group_out=None,\
                         resolution_out=None, center_out=None,\
                         diameter_out=None):
    '''
    from an input hdf5 file (simulation slice), save the 'map' array at a 
    lower resolution and/or save a subset of the pixels for images. The average
    values from filen_in are used
    
    Note that for large sets of stamps, a different function that only loads 
    the original maps once is recommended.
    
    input:
    ------
    filen_in:      (str) hdf5 file containing a 'map' dataset (2d array). 
                   Assumed to be in the format of make_maps outputs (metadata)
                   If the filename does not contain a path, it is assumed to 
                   be in make_maps_opts_locs.ndir
    filen_weight:  (str) if the data in filen_in is a map of weighted averages,
                   this file can be used to specifiy the weights, so that the 
                   stamp map contains consistently defined averages. This is 
                   only needed if the map resolution is changed. 
    filen_out:     (str) name of the hdf5 file to store the output in. Default 
                   is  filen_in with '_stamps' appended. If a path is not part 
                   of this name, it is stored in 
                   make_maps_opts_locs.pdir + 'stamps/'
    group_out:     (str or None) name of the hdf5 group to store the stamp in. 
                   Default (None) is 'stamp#', where '#' is a number 
                   (starting at 0)
    resolution_out: (float, int, or None) resolution of the final map.
                   None -> keep old resolution
                   int  -> use that many pixels (must divide the number of 
                   pixels in the selected region)
                   float -> use that many cpkc per pixel (integer multiple of 
                   original resolution; diameter_out is extended if necessary)
    center_out:    (tuple of 2 floats) center of the region to cut out of the 
                   image (units: cMpc). If None, the whole image is used
    diameter_out:  (float or tuple of 2 floats, or None) extent of the region 
                   to cut out of the image. May overlap periodic box edges. 
                   If one float, the region is square, if two, matches the axes
                   of the image. May not be None if center_out isn't.
    '''
    
    # file names: paths and defaults
    if '/' not in filen_in:
        filen_in = ol.ndir + filen_in
    if filen_out is None:
        filen_out = ol.pdir + 'stamps/' + filen_in.split('/')[-1]
        filen_out = '.'.join(filen_out.split('.')[:-1]) + '_stamps.hdf5'
    elif '/' not in filen_out:
        filen_out = ol.pdir + 'stamps/' + filen_out
    if filen_weight is not None:
        if '/' not in filen_weight:
            filen_weight = ol.ndir + filen_weight

    if center_out is None and resolution_out is None:
        print('If no center or resolution is specified, a copy of the image'+\
              ' would just be stored.')
        raise ValueError('center_out = None and resolution_out = None')
    
    with h5py.File(filen_in, 'r') as fw,\
         h5py.File(filen_out, 'a') as fo:
        if filen_weight is not None:
            fq = h5py.File(filen_weight, 'r')
            
        if group_out is None:
            si = -1
            base = 'stamp{}'
            while True:
                si += 1
                group_out = base.format(si)
                if group_out not in fo:
                    break
        else:
            if group_out in fo:
                raise RuntimeError('The group {gn} already exists in {fn}'.format(\
                                   gn=group_out, fn=filen_out))

        L_x = fw['Header/inputpars'].attrs['L_x'] 
        L_y = fw['Header/inputpars'].attrs['L_y']
        L_z = fw['Header/inputpars'].attrs['L_z']
        LsinMpc = bool(fw['Header/inputpars'].attrs['LsinMpc'])
        cosmopars = {key: val for key, val in fw['Header/inputpars/cosmopars'].attrs.items()}
        Ls = np.array([L_x, L_y, L_z])
        if not LsinMpc:
            Ls /= cosmopars['h']
        axis = fw['Header/inputpars'].attrs['axis'].decode()
        if axis == 'z':
            axis1 = 0
            axis2 = 1
            #axis3 = 2
        elif axis == 'y':
            axis1 = 2
            axis2 = 0
            #axis3 = 1
        elif axis == 'x':
            axis1 = 1
            axis2 = 2
            #axis3 = 0
        extent_x = Ls[axis1]
        extent_y = Ls[axis2]
        npix_x = fw['Header/inputpars'].attrs['npix_x']
        npix_y = fw['Header/inputpars'].attrs['npix_y']
        res_in_x = extent_x * 1e3 / npix_x
        res_in_y = extent_y * 1e3 / npix_y
        center = np.array(fw['Header/inputpars'].attrs['centre'])
        if not LsinMpc:
            center /= cosmopars['h']
        cen_x = center[axis1]
        cen_y = center[axis2]
        
        logw = bool(fw['Header/inputpars'].attrs['log'])
        logq = False
        if filen_weight is not None:
            logq = bool(fq['Header/inputpars'].attrs['log'])
            
        if not hasattr(diameter_out, '__len__'):
            diameter_out = (diameter_out,) * 2
        if center_out is None:
            sel_in = [(slice(0, npix_x, None), slice(0, npix_y, None))] + \
                     [None] * 3
            if isinstance(resolution_out, int):
                if not (npix_x % resolution_out == 0 and \
                        npix_y % resolution_out == 0 ):
                    errst = 'Target number of pixels {res_out} should be an' +\
                            ' integer factor of the input dimensions {inx}, {iny}'
                    raise ValueError(errst.format(res_out=resolution_out,\
                                                  inx=npix_x, iny=npix_y))
                outdims = (resolution_out, resolution_out)
            else:
                if not (np.isclose(resolution_out % res_in_x, 0.) and
                        np.isclose(resolution_out % res_in_y, 0.)):
                    errst = 'Target resolution {res_out} should be an' +\
                            ' integer mutliple of the input resolution {inx}, {iny}'
                    raise ValueError(errst.format(res_out=resolution_out,\
                                              inx=res_in_x, iny=res_in_y))
                outdims = (int(res_in_x / resolution_out * npix_x + 0.5),\
                           int(res_in_y / resolution_out * npix_y + 0.5))
        else:
            lowerleftcorner = (cen_x - 0.5 * extent_x, cen_y - 0.5 * extent_y)
            # extent in units of pixel size 
            extent_target = ((center_out[0] - 0.5 * diameter_out[0] \
                              - lowerleftcorner[0]) * 1e3 / res_in_x,\
                             (center_out[0] + 0.5 * diameter_out[0] \
                              - lowerleftcorner[0]) * 1e3 / res_in_x,\
                             (center_out[1] - 0.5 * diameter_out[1] \
                              - lowerleftcorner[1]) * 1e3 / res_in_y,\
                             (center_out[1] + 0.5 * diameter_out[1] \
                              - lowerleftcorner[1]) * 1e3 / res_in_y\
                             )
            npix_sel_x_target = int(extent_target[1] - extent_target[0] + 0.5)
            npix_sel_y_target = int(extent_target[3] - extent_target[2] + 0.5)
            if isinstance(resolution_out, int):
                if not (npix_x % npix_sel_x_target == 0 and \
                        npix_y % npix_sel_y_target == 0 ):
                    errst = 'Target number of pixels {res_out_x}, {res_out_y} should be' +\
                            ' integer factors of the input dimensions {inx}, {iny}'
                    raise ValueError(errst.format(res_out_x=npix_sel_x_target,\
                                                  res_out_y=npix_sel_y_target,\
                                                  inx=npix_x, iny=npix_y))
                outdims = (resolution_out, resolution_out)
                seldims = (npix_sel_x_target, npix_sel_y_target)
            else:
                if not (np.isclose(resolution_out % res_in_x, 0.) and
                        np.isclose(resolution_out % res_in_y, 0.)):
                    errst = 'Target resolution {res_out} should be an' +\
                            ' integer mutliple of the input resolution {inx}, {iny}'
                    raise ValueError(errst.format(res_out=resolution_out,\
                                              inx=res_in_x, iny=res_in_y))
                pixrat = (int(resolution_out / res_in_x + 0.5),\
                          int(resolution_out / res_in_y + 0.5))
                outdims = ((npix_sel_x_target - 1) // pixrat[0] + 1,\
                           (npix_sel_y_target - 1) // pixrat[1] + 1)
                seldims = (outdims[0] * pixrat[0], outdims[1] * pixrat[1])
            box = cosmopars['boxsize'] / cosmopars['h']
            # pix zero has its center at 0.5 in pixel units -> floored to zero
            cenpix = ((center_out[0] - lowerleftcorner[0]) * 1000 / res_in_x,\
                      (center_out[1] - lowerleftcorner[1]) * 1000 / res_in_y)   
            selregion = (cenpix[0] - 0.5 * seldims[0], cenpix[0] + 0.5 * seldims[0],\
                         cenpix[1] - 0.5 * seldims[1], cenpix[1] + 0.5 * seldims[1])
            selinds = [int(reg) for reg in selregion]
            if selinds[1] - selinds[0] != seldims[0]: # might happen (rounding errors; which is modified should then be arbitrary):
                selinds[1] = selinds[0]  + seldims[0]
            if selinds[3] - selinds[2] != seldims[1]: # might happen (rounding errors; which is modified should then be arbitrary):
                selinds[3] = selinds[2] + seldims[1]
            edges_sel = [lowerleftcorner[0] + res_in_x * 1e-3 * selinds[0],\
                         lowerleftcorner[0] + res_in_x * 1e-3 * selinds[1],\
                         lowerleftcorner[1] + res_in_x * 1e-3 * selinds[2],\
                         lowerleftcorner[1] + res_in_x * 1e-3 * selinds[3],\
                         ]
            if edges_sel[1] - edges_sel[0] > extent_x or \
               edges_sel[3] - edges_sel[2] > extent_y:
                raise RuntimeError('Selected region is larger than the region in the map. '+\
                                   'This might be the result of rounding to match the chosen resolution.')
            inds_wholebox_x = int(box * 1e3 / res_in_x + 0.5)
            inds_wholebox_y = int(box * 1e3 / res_in_y + 0.5)
            selinds = [selinds[0] % inds_wholebox_x,\
                       selinds[1] % inds_wholebox_x,\
                       selinds[2] % inds_wholebox_y,\
                       selinds[3] % inds_wholebox_y,\
                       ]
            sel_in = [None] * 4
            if selinds[0] < selinds[1]:
                if selinds[2] < selinds[3]:
                    sel_in[0] = (slice(selinds[0], selinds[1], None),\
                                 slice(selinds[2], selinds[3], None))
                else:
                    sel_in[0] = (slice(selinds[0], selinds[1], None),\
                                 slice(selinds[2], npix_y, None))
                    sel_in[2] = (slice(selinds[0], selinds[1], None),\
                                 slice(0, selinds[3], None))
            else:
                if selinds[2] < selinds[3]:
                    sel_in[0] = (slice(selinds[0], npix_x, None),\
                                 slice(selinds[2], selinds[3], None))
                    sel_in[1] = (slice(0, selinds[1], None),\
                                 slice(selinds[2], selinds[3], None))
                else:
                    sel_in[0] = (slice(selinds[0], npix_x, None),\
                                 slice(selinds[2], npix_y, None))
                    sel_in[1] = (slice(0, selinds[1], None),\
                                 slice(selinds[2], npix_y, None))
                    sel_in[2] = (slice(selinds[0], npix_x, None),\
                                 slice(0, selinds[3], None))
                    sel_in[3] = (slice(0, selinds[1], None),\
                                 slice(0, selinds[3], None))
        
        map_in_parts = []
        map_w_parts = []
        for sel in sel_in:
            if sel is None:
                map_in_parts.append(None)
                map_w_parts.append(None)
            else:
                map_in_parts.append(fw['map'][sel])
                if filen_weight is not None:
                    map_w_parts.append(fq['map'][sel])
        if logw:
            map_in_parts = [10**part if part is not None else part for part in map_in_parts]
        if logq:
            map_w_parts = [10**part if part is not None else part for part in map_w_parts]
            
        if map_in_parts[1] is None and map_in_parts[2] is None:
            map_in = map_in_parts[0]
            map_w = map_w_parts[0]
        elif map_in_parts[1] is None and map_in_parts[2] is not None:
            map_in = np.append(map_in_parts[0], map_in_parts[2], axis=1)
            if filen_weight is not None:
                map_w = np.append(map_w_parts[0], map_w_parts[2], axis=1)
            else:
                map_w = None
        elif map_in_parts[1] is not None and map_in_parts[2] is None:
            map_in = np.append(map_in_parts[0], map_in_parts[1], axis=0)
            if filen_weight is not None:
                map_w = np.append(map_w_parts[0], map_w_parts[1], axis=0)
            else:
                map_w = None
        else:
            map_in0 = np.append(map_in_parts[0], map_in_parts[2], axis=1)
            map_in1 = np.append(map_in_parts[1], map_in_parts[3], axis=1)
            map_in = np.append(map_in0, map_in1, axis=0)
            if filen_weight is not None:
                map_w0 = np.append(map_w_parts[0], map_w_parts[2], axis=1)
                map_w1 = np.append(map_w_parts[1], map_w_parts[3], axis=1)
                map_w = np.append(map_w0, map_w1, axis=0)
            else:
                map_w = None
        
        res = reducedims(map_in, outdims, weights_in=map_w)
        if logw:
            res = np.log10(res)
            
        # store info           
        grp = fo.create_group(group_out)       
        fw.copy('Header', grp, 'Header_in')
        if filen_weight is not None:
            fq.copy('Header', grp, 'Header_weight')
        ds = grp.create_dataset('map', data=res)
        ds.attrs.create('subregion', center_out is not None)
        if center_out is not None:
            ds.attrs.create('edges_axis0_cMpc', np.array([edges_sel[0], edges_sel[1]]))
            ds.attrs.create('edges_axis1_cMpc', np.array([edges_sel[2], edges_sel[3]]))
            
        if filen_weight is not None:
            fq.close()
            
def savestamps(center=(25., 25.), diameter=50., resolution=125.):
    '''
    gets stamps for an example region in the box for the different lines 
    '''
    # first slice
    basename = 'emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5'
    for line in lines:
        filen_in = basename.format(line=line)
        make_and_save_stamps(filen_in, filen_weight=None,\
                         filen_out=None, group_out=None,\
                         resolution_out=resolution, center_out=center,\
                         diameter_out=diameter)


def plotstamps(filebase=None, halocat=None,\
               outname='xraylineem_stamps_boxcorner_L0100N1504_27_test3p5_SmAb_C2Sm_32000pix_6p25slice_zcen3p125_z-projection_noEOS.pdf', \
               groups='stamp0', minhalomass=11.):
    '''
    plot the stamps stored in files filebase, overplotting the halos from 
        
    input:
    ------
    filebase:  (str) filename, with a format specifier {line} for the different
               lines. Lines with absent files are skipped with a warning.
               Assumed to be in ol.pdir + 'stamps' if no path is given
    halocat:   (str) halo catalogue for overplotting halos. Assumed to be in 
               ol.pdir if no path is given.
    groups:    (str or dict) group in each file to use. string or 
               {line: string} dictionary
    minhalomass: (float) minimum halo mass to plot (log10 solar masses)\
    outname:   (str) name of the file to save the plot to
    '''        
    if halocat is None:
        halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    if filebase is None:
        filebase = 'emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5'
    
    marklength = 10. #cMpc
    vmin = -12. # log10 photons / cm2 / s / sr 
    vtrans = -2.5
    vmax = 1.0
    scaler200 = 2. # show radii at this times R200c
    cmap = pu.paste_cmaps(['gist_yarg', 'plasma'], [vmin, vtrans, vmax],\
                          trunclist=[[0.0, 0.5], [0.2, 1.0]])
    
    # retrieve the stamps, and necessary metadata
    if '/' not in filebase:
        filebase = ol.pdir + 'stamps/' + filebase
    if not isinstance(groups, dict):
        groups = {line: groups for line in lines}
    maps = {}
    extents = {}
    depths = {}
    paxes = {}
    resolutions = {}
    cosmoparss = {}
    snapshots = {}
    for line in lines:
        filen = filebase.format(line=line)
        try:
            with h5py.File(filen, 'r') as ft:
                if groups[line] not in ft:
                    print('Could not find the group {grp} for {line}: {filen}.'.format(\
                          line=line, filen=filen, grp=groups[line]))
                    continue
                grp = ft[groups[line]] 
                maps[line] = grp['map'][:]
                cosmopars = {key: val for key, val in \
                    grp['Header_in/inputpars/cosmopars'].attrs.items()}
                cosmoparss[line] = cosmopars
                L_x = grp['Header_in/inputpars'].attrs['L_x'] 
                L_y = grp['Header_in/inputpars'].attrs['L_y']
                L_z = grp['Header_in/inputpars'].attrs['L_z']
                centre = np.array(grp['Header_in/inputpars'].attrs['centre'])
                LsinMpc = bool(grp['Header_in/inputpars'].attrs['LsinMpc'])
                Ls = np.array([L_x, L_y, L_z])
                if not LsinMpc:
                    Ls /= cosmopars['h']
                    centre /= cosmopars['h']
                axis = grp['Header_in/inputpars'].attrs['axis'].decode()
                if axis == 'z':
                    axis1 = 0
                    axis2 = 1
                    axis3 = 2
                elif axis == 'y':
                    axis1 = 2
                    axis2 = 0
                    axis3 = 1
                elif axis == 'x':
                    axis1 = 1
                    axis2 = 2
                    axis3 = 0
                paxes[line] = (axis1, axis2, axis3)
                extent_x = Ls[axis1]
                extent_y = Ls[axis2]
                snapshots[line] = grp['Header_in/inputpars'].attrs['snapnum']
                
                if bool(grp['map'].attrs['subregion']):
                    extents[line] = np.array([np.array(grp['map'].attrs['edges_axis0_cMpc']),\
                                              np.array(grp['map'].attrs['edges_axis0_cMpc'])])
                else:
                    extents[line] = np.array([[0., extent_x],\
                                              [0., extent_y]])
                depths[line] = (centre[axis3] - 0.5 * Ls[axis3], centre[axis3] + 0.5 * Ls[axis3]) 
                resolutions[line] = ((extents[line][0][1] - extents[line][0][0])\
                                      * 1e3 * cosmopars['a'] / maps[line].shape[0],\
                                     (extents[line][1][1] - extents[line][1][0])\
                                      * 1e3 * cosmopars['a'] / maps[line].shape[1])
                
        except IOError:
            print('Could not find the file for {line}: {filen}.'.format(\
                  line=line, filen=filen))
    
    print('Using map resolutions:\n{res}'.format(res=resolutions))
    _lines = sorted(maps.keys(), key=ol.line_eng_ion.get)
    
    # get halo catalogue data for overplotting
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    with h5py.File(halocat, 'r') as hc:
        snapnum = hc['Header'].attrs['snapnum']
        cosmopars = {key: val for key, val in hc['Header/cosmopars'].attrs.items()}
        if not np.all(snapnum == np.array([snapshots[line] for line in _lines])):
            raise RuntimeError('Stamp snapshots do not match halo catalogue snapshot')
        masses = np.log10(hc['M200c_Msun'][:])
        radii = hc['R200c_pkpc'] / cosmopars['a'] * 1e-3
        pos = np.array([hc['Xcom_cMpc'][:],\
                        hc['Ycom_cMpc'][:],\
                        hc['Zcom_cMpc'][:]])
        msel = masses >= minhalomass
        masses = masses[msel]
        radii = radii[msel]
        pos = pos[:, msel]
    
    ncols = 4
    nrows = (len(_lines) - 1) // ncols + 1
    figwidth = 11. 
    lrspace = len(_lines) < nrows * ncols
    
    panelwidth = figwidth / ncols
    panelheight = panelwidth
    if lrspace:
        addheight = 0.
        height_ratios = [panelheight] * nrows
        nrows_use = nrows
    else:
        addheight = 1.5
        height_ratios = [panelheight] * nrows + [addheight]
        nrows_use = nrows + 1
    
    figheight = sum(height_ratios)
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows_use, hspace=0.0, wspace=0.0,\
                        width_ratios=[panelwidth] * ncols,\
                        height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(len(_lines))]
    if lrspace:
        colstart = ncols - (nrows * ncols - len(_lines)) 
        gridreg = grid[nrows - 1, colstart:]
        cgrid = gsp.GridSpecFromSubplotSpec(nrows=3, ncols=2,\
                       subplot_spec=gridreg, wspace=0.0, hspace=0.35,\
                       height_ratios = [0.001, 0.5, 0.5],\
                       width_ratios=[0.1, 0.9])
        
        cax1  = fig.add_subplot(cgrid[1, 1:])
        cax2  = fig.add_subplot(cgrid[2, 1:])
    else:
        gridreg = grid[nrows, :]
        cgrid = gsp.GridSpecFromSubplotSpec(nrows=2, ncols=2,\
                       subplot_spec=gridreg, wspace=0.2, hspace=0.0,\
                       height_ratios=[0.5, 1.])
        cax1  = fig.add_subplot(cgrid[1, 0])
        cax2  = fig.add_subplot(cgrid[1, 1])
    
    clabel_img = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{ph.} \\, \\mathrm{cm}^{-2} \\mathrm{s}^{-1} \\mathrm{sr}^{-1}]$'
    clabel_hmass = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    cbar, colordct = add_cbar_mass(cax2, massedges=mass_edges_standard,\
             orientation='horizontal', clabel=clabel_hmass, fontsize=fontsize, aspect=0.1)
    print('Max value in maps: {}'.format(max([np.max(maps[line]) for line in _lines])))
    
    cmap_img = cmap
    cmap_img.set_under(cmap_img(0.))
    
    for li in range(len(_lines)):
        ax = axes[li]
        line = _lines[li]
        
        #labelbottom = li > len(_lines) - ncols - 1
        #labeltop = li < ncols 
        #labelleft = li % ncols == 0
        #labelright = li % ncols == ncols - 1
        #ax.tick_params(labelsize=fontsize - 1,  direction='in',\
        #               labelbottom=labelbottom, labeltop=labeltop,\
        #               labelleft=labelleft, labelright=labelright,\
        #               top=labeltop, left=labelleft, bottom=labelbottom,\
        #               right=labelright)
        #lbase = '{ax} [cMpc]'
        #axis1 = paxes[line][0]
        #axis2 = paxes[line][1]
        #axis3 = paxes[line][2]
        #if labelbottom:
        #    xl = lbase.format(ax=['X', 'Y', 'Z'][axis1])
        #    ax.set_xlabel(xl, fontsize=fontsize)
        #if labelleft:
        #    yl = lbase.format(ax=['X', 'Y', 'Z'][axis2])
        #    ax.set_ylabel(yl, fontsize=fontsize)
        ax.tick_params(top=False, bottom=False, left=False, right=False,\
                      labeltop=False, labelbottom=False, labelleft=False,\
                      labelright=False)
        
        ax.set_facecolor(cmap_img(0.))    
        img = ax.imshow(maps[line].T, origin='lower', interpolation='nearest',\
                  extent=(extents[line][0][0], extents[line][0][1],\
                          extents[line][1][0], extents[line][1][1]),\
                  cmap=cmap_img, vmin=vmin, vmax=vmax) 
        
        posx = pos[axis1]
        posy = pos[axis2]
        posz = pos[axis3]
        margin = np.max(radii)
        zrange = depths[line]
        xrange = [extents[line][0][0] - margin,\
                  extents[line][0][1] + margin]
        yrange = [extents[line][1][0] - margin,\
                  extents[line][1][1] + margin]
        hsel = np.ones(len(posx), dtype=bool)
        cosmopars = cosmoparss[line]
        boxsize = cosmopars['boxsize'] / cosmopars['h'] 
        hsel &= cu.periodic_sel(posz, zrange, boxsize)
        hsel &= cu.periodic_sel(posx, xrange, boxsize)
        hsel &= cu.periodic_sel(posy, yrange, boxsize)
        
        posx = posx[hsel]
        posy = posy[hsel]
        ms = masses[hsel]
        rd = radii[hsel]
        
        me = np.array(sorted(list(colordct.keys())) + [17.])
        mi = np.max(np.array([np.searchsorted(me, ms) - 1,\
                              np.array([0] * len(ms))]),\
                    axis=0)
        colors = np.array([colordct[me[i]] for i in mi])
        
        patches = [mpatch.Circle((posx[ind], posy[ind]), scaler200 * rd[ind]) \
                   for ind in range(len(posx))] # x, y axes only
    
        patheff = [mppe.Stroke(linewidth=1.2, foreground="black"),\
                       mppe.Stroke(linewidth=0.7, foreground="white"),\
                       mppe.Normal()] 
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=colors, facecolor='none', linewidth=0.7,\
                       path_effects=patheff)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.add_collection(collection)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        
        patheff_text = [mppe.Stroke(linewidth=2.0, foreground="white"),\
                        mppe.Stroke(linewidth=0.4, foreground="black"),\
                        mppe.Normal()]        
        ltext = nicenames_lines[line]
        ax.text(0.95, 0.95, ltext, fontsize=fontsize, path_effects=patheff_text,\
                horizontalalignment='right', verticalalignment='top',\
                transform=ax.transAxes, color='black')
        if li == 0:
            mtext = str(marklength)
            if mtext[-2:] == '.0':
                mtext = mtext[:-2]
            mtext = mtext + ' cMpc'
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            if marklength > 2.5 * xr:
                print('Marklength {} is too large for the plotted range'.format(marklength))
                continue
            xs = xlim[0] + 0.1 * xr
            ypos = ylim[0] + 0.07 * yr
            xcen = xs + 0.5 * marklength
            
            patheff = [mppe.Stroke(linewidth=3.0, foreground="white"),\
                       mppe.Stroke(linewidth=2.0, foreground="black"),\
                       mppe.Normal()] 
            ax.plot([xs, xs + marklength], [ypos, ypos], color='black',\
                    path_effects=patheff, linewidth=2)
            ax.text(xcen, ypos + 0.01 * yr, mtext, fontsize=fontsize,\
                    path_effects=patheff_text, horizontalalignment='center',\
                    verticalalignment='bottom', color='black')
            
    plt.colorbar(img, cax=cax1, orientation='horizontal', extend='both')
    cax1.set_xlabel(clabel_img, fontsize=fontsize)
    cax1.tick_params(labelsize=fontsize - 1, which='both')
    cax1.set_aspect(0.1)   
    
    print('Halos indicated at {rs} x R200c'.format(rs=scaler200))

    if outname is not None:
        if '/' not in outname:
            outname = mdir + outname
        if outname[-4:] != '.pdf':
            outname = outname + '.pdf'
        plt.savefig(outname, format='pdf', bbox_inches='tight')
        


### radial profiles
def readin_radprof(filename, seltags, ys, runit='pkpc', separate=False,\
                   binset='binset_0', retlog=True):
    '''
    from the coldens_rdist radial profile hdf5 file format: 
    extract the profiles matching the input criteria
    
    input:
    ------
    filename: name of the hdf5 file containing the profiles
    seltag:   list (iterable) of seltag attributes for the galaxy/halo set used
    ys:       list (iterable) of tuples (iterable) containing ytype 
              ('mean', 'mean_log', 'perc', or 'fcov') and 
              yval (threashold for 'fcov', percentile (0-100) for 'perc',\
              ignored for 'mean' and 'mean_log')
    runits:   units for the bins: 'pkpc' or 'R200c'
    separate: get individual profiles for the galaxies in this group (subset
              contained in this file) if True, otherwise, use the combined 
              profile 
    binset:   the name of the binset group ('binset_<number>')
    retlog:   return log unit values (ignored if log/non-log isn't recorded)
    
    output:
    -------
    two dictionaries:
    y dictionary, bin edges dictionary
    
    nested dictionaries containing the retrieved values
    keys are nested, using the input values as keys:
    seltag: ys: [galaxyid: (if separate)] array of y values or bin edges
    
    '''
    
    if '/' not in filename:
        filename = ol.pdir + 'radprof/' + filename

    spath = '{{gal}}/{runit}_bins/{bin}/{{{{ds}}}}'.format(runit=runit, bin=binset)
        
    with h5py.File(filename, 'r') as fi:
        # match groups to seltags
        gkeys = list(fi.keys())
        setkeys = [key if 'galset' in key else None for key in gkeys]
        setkeys = list(set(setkeys) - {None})
        
        seltags_f = {key: fi[key].attrs['seltag'].decode()\
                     if 'seltag' in fi[key].attrs.keys() else None \
                     for key in setkeys}
        galsets_seltag = {tag: list(set([key if seltags_f[key] == tag else\
                                         None for key in seltags_f])\
                                    - {None})\
                          for tag in seltags}
        
        indkeys = list(set(gkeys) - set(setkeys))
        indgals = [int(key.split('_')[-1]) for key in indkeys]
        
        spaths = {}
        galid_smatch = {}
        ys_out = {}
        bins_out = {}
        for seltag in seltags:
            keys_tocheck = galsets_seltag[seltag]
            if len(keys_tocheck) == 0:
                raise RuntimeError('No matches found for seltag {}'.format(seltag))
            
            if separate: # just pick one; should have the same galaxyids
                extag = keys_tocheck[0]
                galids = fi['{gs}/galaxyid'.format(gs=extag)][:]
                # cross-reference stored galids (should be a smallish number)
                # and listed set galids
                galids_use = [galid if galid in indgals else None for galid in galids]
                galids_use = list(set(galids_use) - {None})
                spaths.update({galid: spath.format(gal='galaxy_{}'.format(galid))\
                               for galid in galids_use})
                galid_smatch.update({galid: seltag for galid in galids_use})
            else:
                spaths.update({key: spath.format(gal=key) for key in keys_tocheck})
                galid_smatch.update({key: seltag for key in keys_tocheck})
        for ytv in ys:
            ykey = ytv
            temppaths = spaths.copy()
            if ytv[0] in ['fcov', 'perc']:
                ypart = '{}_{:.1f}'.format(*tuple(ytv))
            else:
                if ytv[0] == 'm':
                    ypart = ytv
                else:
                    ypart = ytv[0]
            ypaths = {key: temppaths[key].format(ds=ypart) for key in temppaths}
            bpaths = {key: temppaths[key].format(ds='bin_edges') for key in temppaths}

            for key in temppaths:
                ypath = ypaths[key]
                bpath = bpaths[key]
                if ypath in fi:
                    vals = fi[ypath][:]
                    if 'logvalues' in fi[ypath].attrs.keys():
                        logv_s = bool(fi[ypath].attrs['logvalues'])
                        if 'fcov' in ypath: # logv_s is for 
                            if retlog:
                                vals = np.log10(vals)
                        else:
                            if retlog and (not logv_s):
                                vals = np.log10(vals)
                            elif (not retlog) and logv_s:
                                vals = 10**vals
                    bins = fi[bpath][:]
                
                    seltag = galid_smatch[key]
                    if seltag not in ys_out:
                        ys_out[seltag] = {}
                        bins_out[seltag] = {}
                    if str(key) != key: # string instance? (hard to test between pytohn 2 and 3 with isinstance)
                        if ykey not in ys_out[seltag]:
                            ys_out[seltag][ykey] = {}
                            bins_out[seltag][ykey] = {}
                        ys_out[seltag][ykey][key] = vals
                        bins_out[seltag][ykey][key] = bins
                    else:
                        ys_out[seltag][ykey] = vals
                        bins_out[seltag][ykey] = bins
    return ys_out, bins_out                    
                    
def readin_3dprof_stacked(filename, Zelt='oxygen', weight='Mass',\
                          combmethod='addnormed-R200c', rbinu='R200c',\
                          ):
    '''
    read in the 3d profiles for the selected halo masses. Only works if the 
    file actually contains the requested data, and assumes group names and 
    stack axes from the paper 3 stacks (tgrpns) 
    
    input:
    ------
    filename:      (string) hdf5 file with the stacked data
    Zelt:          (string) which element to use for metallicities
    weight:        the histogram weight
    combmethod:    one of the histogram combination methods from 
                   prof3d_galsets, e.g. 'add' or 'addnormed-R200c'
    rbinu:         radial bin units: 'R200c' or 'pkpc'
    
    returns: (hists, edges, galaxyids)
    --------
    hists:   nested dictionary: the first set of keys is log10(M200c / Msun)
             for the lower edge of each halo mass bin for which data is stored
             the second level of keys is the radial profile type:
               'T' for temperature, 'n' for hydrogen number density, 
               'Z' for metallcity, and 'weight' for the cumulative profiles
             each histogram has dimensions [radius, profile type], except for
             'weight', which just contains the cumulative values
    edges:   dictionary with the same structure as hists, but contains a list 
             of the histogram bin edges [radial edges, profile type edges], 
             excpet for 'weight', which only contains [radial edges] 
             (an array in a list) 
    galaxyids: the galaxyids that went into each stack (dictionary with the 
              same halo mass keys)    
    
    even with the 'add' combination method, cumulative profiles are returned 
    normalized to the quantity within R200c in the final stack. 
    Bin edges +/- infinity are reset based on the nearest two bins, using the
    same spacing, for easier plotting
    
    '''

    elt = string.capwords(Zelt)
    tgrpns = {'T': '3Dradius_Temperature_T4EOS_StarFormationRate_T4EOS',\
              'n': '3Dradius_Niondens_hydrogen_SmAb_T4EOS_StarFormationRate_T4EOS',\
              'Z': '3Dradius_SmoothedElementAbundance-{elt}_T4EOS_StarFormationRate_T4EOS'.format(\
                                                      elt=elt),\
              }
    axns  = {'r3d':  '3Dradius',\
             'T':    'Temperature_T4EOS',\
             'n':    'Niondens_hydrogen_SmAb_T4EOS',\
             'Z':    'SmoothedElementAbundance-{elt}_T4EOS'.format(elt=elt),\
            }
    axnl = ['n', 'T', 'Z']
    
    mgrpn = 'L0100N1504_27_Mh0p5dex_1000/%s-%s'%(combmethod, rbinu)
    
    # read in data: stacked histograms -> process to plottables
    hists_main = {}
    edges_main = {}
    galids_main = {}
    
    with h5py.File(filename, 'r') as fi:
        for profq in tgrpns:
            tgrpn = tgrpns[profq]
            grp = fi[tgrpn + '/' + mgrpn]
            sgrpns = list(grp.keys())
            massbins = [grpn.split('_')[-1] for grpn in sgrpns]    
            massbins = [[np.log10(float(val)) for val in binn.split('-')] for binn in massbins]
            
            for mi in range(len(sgrpns)):
                mkey = massbins[mi][0]
                
                grp_t = grp[sgrpns[mi]]
                hist = np.array(grp_t['histogram'])
                if bool(grp_t['histogram'].attrs['log']):
                    hist = 10**hist
                
                edges = {}
                axes = {}
                
                for axn in [profq, 'r3d']:
                    edges[axn] = np.array(grp_t[axns[axn] + '/bins'])
                    if not bool(grp_t[axns[axn]].attrs['log']):
                        edges[axn] = np.log10(edges[axn])
                    axes[axn] = grp_t[axns[axn]].attrs['histogram axis']  
                
                if mkey not in edges_main:
                    edges_main[mkey] = {}
                if mkey not in hists_main:
                    hists_main[mkey] = {}
                
                # apply normalization consisent with stacking method
                if rbinu == 'pkpc':
                    edges['r3d'] += np.log10(c.cm_per_mpc * 1e-3)
                
                if combmethod == 'addnormed-R200c':
                    if rbinu != 'R200c':
                        raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                    _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                    if len(_i) != 1:
                        raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                    _i = _i[0]
                    _a = list(range(len(hist.shape)))
                    _s = [slice(None, None, None) for dummy in _a]
                    _s[axes['r3d']] = slice(None, _i, None)
                    norm_t = np.sum(hist[tuple(_s)])
                elif combmethod == 'add':
                    if rbinu != 'R200c':
                        raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                    _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                    if len(_i) != 1:
                        raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                    _i = _i[0]
                    _a = list(range(len(hist.shape)))
                    _s = [slice(None, None, None) for dummy in _a]
                    _s[axes['r3d']] = slice(None, _i, None)
                    norm_t = np.sum(hist[tuple(_s)])
                hist *= (1. / norm_t)
                
                rax = axes['r3d']
                yax = axes[profq]
                
                edges_r = np.copy(edges['r3d'])
                edges_y = np.copy(edges[profq])
                
                hist_t = np.copy(hist)
                
                # deal with edge units (r3d is already in R200c units if R200c-stacked)
                if edges_r[0] == -np.inf: # reset centre bin position
                    edges_r[0] = 2. * edges_r[1] - edges_r[2] 
                if edges_y[0] == -np.inf: # reset centre bin position
                    edges_y[0] = 2. * edges_y[1] - edges_y[2]
                if edges_y[-1] == np.inf: # reset centre bin position
                    edges_y[-1] = 2. * edges_y[-2] - edges_y[-3]
                    
                sax = list(range(len(hist_t.shape)))
                sax.remove(rax)
                sax.remove(yax)
                hist_t = np.sum(hist_t, axis=tuple(sax))
                if yax < rax:
                    hist_t = hist_t.T
                #hist_t /= (np.diff(edges_r)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
                
                hists_main[mkey][profq] = hist_t
                edges_main[mkey][profq] = [edges_r, edges_y]
                #print(hist_t.shape)
                
                # add in cumulative plot for the weight (from one of the profiles)
                if profq == 'n':
                    hist_t = np.copy(hist)
                    sax = list(range(len(hist_t.shape)))
                    sax.remove(rax)
                    hist_t = np.sum(hist_t, axis=tuple(sax))
                    hist_t = np.cumsum(hist_t)
                    hists_main[mkey]['weight'] = hist_t
                    edges_main[mkey]['weight'] = [edges_r[1:]]
                                  
                galids_main[mkey] = np.array(grp_t['galaxyids'])
    return hists_main, edges_main, galids_main

def plot_radprof1(measure='mean', mmin=10.5, rbinning=0):
    '''
    plot mean or median radial profiles for each line and halo mass bin
    panels for different lines
    
    input:
    ------
    measure:  'mean' or 'median'
    mmin:     minimum halo mass to show (log10 Msun, 
              value options: 9.0, 9.5, 10., ... 14.)
    rbinning: 0 -> 10 pkpc bins
              1 -> 10 pkpkc bins to 100 pkpc, then 0.1 dex ins
    '''
    print('Values are calculated from 3.125^2 ckpc^2 pixels in 10 pkpc annuli')
    print('z=0.1, Ref-L100N1504, 6.25 cMpc slice Z-projection, SmSb, C2 kernel')
    print('Using max. 1000 (random) galaxies in each mass bin, centrals only')
    
    fontsize = 12
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    
    if rbinning == 0:
        rfilebase = ol.pdir + 'radprof/' + 'radprof_stamps_emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5'
    elif rbinning == 1:
        rfilebase = ol.pdir + 'radprof/' + 'radprof_stamps_emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-4R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5'
    xlabel = '$\\mathrm{r}_{\perp} \\; [\\mathrm{pkpc}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{photons}\\,\\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
    
    if measure == 'mean':
        ys = [('mean',)]
    else:
        ys = [('perc', 50.)]
    
    if rbinning == 0:
        outname = mdir + 'radprof2d_10pkpc-annuli_L0100N1504_27_test3.5_SmAb_C2Sm_6.25slice_noEOS_to-2R200c_1000_centrals_' +\
                      'halomasscomp_{}'.format(measure)
    elif rbinning == 1:
        outname = mdir + 'radprof2d_10pkpc-0.1dex-annuli_L0100N1504_27_test3.5_SmAb_C2Sm_6.25slice_noEOS_to-2R200c_1000_centrals_' +\
                      'halomasscomp_{}'.format(measure)
    outname = outname.replace('.', 'p')
    outname = outname + '.pdf'
    
    medges = np.arange(mmin, 14.1, 0.5)
    seltag_keys = {medges[i]: 'geq{:.1f}_le{:.1f}'.format(medges[i], medges[i + 1])\
                               if i < len(medges) - 1 else\
                               'geq{:.1f}'.format(medges[i])\
                    for i in range(len(medges))}
    seltags = [seltag_keys[key] for key in seltag_keys]
    
    numlines = len(lines)
    ncols = 4
    nrows = (numlines - 1) // ncols + 1
    figwidth = 11. 
    caxwidth = 1.

    if ncols * nrows - numlines >= 2:
        cax_right = False
        _ncols = ncols
        panelwidth = figwidth / ncols
        width_ratios = [panelwidth] * ncols
        c_orientation = 'horizontal'
        c_aspect = 0.08
    else:
        cax_right = True
        _ncols = ncols
        panelwidth = (figwidth - caxwidth) / ncols
        width_ratios = [panelwidth] * ncols + [caxwidth]
        c_orientation = 'vertical'
        c_aspect = 10.
        
    panelheight = panelwidth    
    figheight = panelheight * nrows
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(ncols=_ncols, nrows=nrows, hspace=0.0, wspace=0.0,\
                        width_ratios=width_ratios)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(numlines)]
    if cax_right:
        if nrows > 3: 
            csl = slice(nrows // 2 - 1, nrows // 2 + 2, None)
        else:
            csl = slice(None, None, None)
        cax = fig.add_subplot(grid[csl, ncols])
    else:
        ind_min = ncols - (nrows * ncols - numlines)
        _cax = fig.add_subplot(grid[nrows - 1, ind_min:])
        _cax.axis('off')
        _l, _b, _w, _h = (_cax.get_position()).bounds
        margin = panelwidth * 0.1 / figwidth
        cax = fig.add_axes([_l + margin, _b + margin,\
                            _w - 2.* margin, _h - 2. * margin])
        
    labelax = fig.add_subplot(grid[:nrows, :ncols], frameon=False)
    labelax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    
    clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=medges,\
             orientation=c_orientation, clabel=clabel, fontsize=fontsize,\
             aspect=c_aspect)

    ykey = ys[0]
    for li, line in enumerate(lines):
        ax = axes[li]
        labely = li % ncols == 0
        labelx = numlines -1 - li < ncols
        pu.setticks(ax, fontsize=fontsize, labelleft=labely, labelbottom=labelx)
        ax.set_xscale('log')

        filename = rfilebase.format(line=line)
        yvals, bins = readin_radprof(filename, seltags, ys, runit='pkpc', separate=False,\
                                     binset='binset_0', retlog=True)

        for me in medges:
            tag = seltag_keys[me]
            
            ed = bins[tag][ykey]
            vals = yvals[tag][ykey]
            cens = ed[:-1] + 0.5 * np.diff(ed)
            ax.plot(cens, vals, color=colordct[me], linewidth=2.,\
                    path_effects=patheff)
        
        ax.text(0.98, 0.98, nicenames_lines[line], fontsize=fontsize,\
                transform=ax.transAxes, horizontalalignment='right',\
                verticalalignment='top')
    # sync plot ranges
    xlims = [ax.get_xlim() for ax in axes]
    xmin = min([xlim[0] for xlim in xlims])
    xmax = max([xlim[1] for xlim in xlims])
    [ax.set_xlim(xmin, xmax) for ax in axes]

    # three most energetic ions have very low mean SB -> impose limits
    ylims = [ax.get_ylim() for ax in axes]
    ymin = -9. #min([ylim[0] for ylim in ylims])
    ymax = 2. #max([ylim[1] for ylim in ylims])
    [ax.set_ylim(ymin, ymax) for ax in axes]
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    

def plot_radprof2(measure='mean', mmin=10.0):
    '''
    plot mean or median radial profiles for each line and halo mass bin
    panels for different halo masses 
    
    input:
    ------
    measure:  'mean' or 'median'
    mmin:     minimum halo mass to show (log10 Msun, 
              value options: 9.0, 9.5, 10., ... 14.)
    '''
    print('Values are calculated from 3.125^2 ckpc^2 pixels in 10 pkpc annuli')
    print('z=0.1, Ref-L100N1504, 6.25 cMpc slice Z-projection, SmSb, C2 kernel')
    print('Using max. 1000 (random) galaxies in each mass bin, centrals only')
    
    fontsize = 12
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    lw2 = 2.3
    pe2 = [mppe.Stroke(linewidth=lw2 + 0.7, foreground="b"),\
           mppe.Stroke(linewidth=lw2, foreground="w"),\
           mppe.Normal()]
    
    rfilebase = ol.pdir + 'radprof/' + 'radprof_stamps_emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5'
    xlabel = '$\\mathrm{r}_{\perp} \\; [\\mathrm{pkpc}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{photons}\\,\\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
    
    if measure == 'mean':
        ys = [('mean',)]
    else:
        ys = [('perc', 50.)]
    outname = mdir + 'radprof2d_10pkpc-annuli_L0100N1504_27_test3.5_SmAb_C2Sm_6.25slice_noEOS_to-2R200c_1000_centrals_' +\
                  'linecomp_{}.pdf'.format(measure)
    medges = np.arange(mmin, 14.1, 0.5)
    seltag_keys = {medges[i]: 'geq{:.1f}_le{:.1f}'.format(medges[i], medges[i + 1])\
                               if i < len(medges) - 1 else\
                               'geq{:.1f}'.format(medges[i])\
                    for i in range(len(medges))}
    seltags = [seltag_keys[key] for key in seltag_keys]
    
    #numlines = len(lines)
    nummasses = len(medges)
    ncols = 4
    nrows = (nummasses - 1) // ncols + 1
    figwidth = 11. 
    laxheight = 1.
    hspace = 0.15
    wspace = 0.28
    
    if ncols * nrows - nummasses >= 2:
        lax_below = False
        _nrows = nrows
        panelwidth = (figwidth - (ncols * wspace))/ ncols
        width_ratios = [panelwidth] * ncols
        panelheight = panelwidth    
        figheight = panelheight * nrows + hspace * (nrows - 1)
        height_ratios = [panelheight] * nrows 
        l_bbox_to_anchor = (0.5, 0.0)
        l_loc = 'lower center'
        l_ncols = ncols * nrows - nummasses
    else:
        lax_below = True
        _nrows = nrows + 1
        panelwidth = (figwidth - (ncols * wspace))/ ncols
        width_ratios = [panelwidth] * ncols 
        panelheight = panelwidth    
        figheight = panelheight * ncols + laxheight + hspace * nrows
        height_ratios = [panelheight] * nrows + [laxheight]
        l_bbox_to_anchor = (0.5, 0.80)
        l_loc = 'upper center'
        l_ncols = ncols 
    
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=_nrows, hspace=hspace, wspace=wspace,\
                        width_ratios=width_ratios, height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(nummasses)]
    if lax_below:
        lax = fig.add_subplot(grid[nrows, :])
    else:
        ind_min = ncols - (nrows * ncols - nummasses)
        lax = fig.add_subplot(grid[nrows - 1, ind_min:])
    lax.axis('off')
        #_l, _b, _w, _h = (_lax.get_position()).bounds
        #margin = panelwidth * 0.1 / figwidth
        #lax = fig.add_axes([_l + margin, _b + margin,\
        #                    _w - 2.* margin, _h - 2. * margin])
        
    labelax = fig.add_subplot(grid[:nrows, :ncols], frameon=False)
    labelax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    
    #clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    #cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=medges,\
    #         orientation=c_orientation, clabel=clabel, fontsize=fontsize,\
    #         aspect=c_aspect)
    
    ## get lines
    yvals = {}
    bins = {}
    for line in lines:
        filename = rfilebase.format(line=line)
        _yvals, _bins = readin_radprof(filename, seltags, ys, runit='pkpc', separate=False,\
                                     binset='binset_0', retlog=True)
        yvals[line] = _yvals
        bins[line] = _bins
        
    ykey = ys[0]
    for hi, hkey in enumerate(medges):
        ax = axes[hi]
        labely = True # hi % ncols == 0
        labelx = True # nummasses -1 - hi < ncols
        pu.setticks(ax, fontsize=fontsize, labelleft=labely, labelbottom=labelx)
        ax.set_xscale('log')
        
        mtag = seltag_keys[hkey]
        
        _max = -np.inf
        for line in lines:
            ed = bins[line][mtag][ykey]
            vals = yvals[line][mtag][ykey]
            cens = ed[:-1] + 0.5 * np.diff(ed)
            ax.plot(cens, vals, color=linecolors[line], linewidth=2.,\
                    path_effects=patheff, linestyle='dashed')
            _max = max(_max, np.max(vals))
        if hi == len(medges) - 1:
            text = '$\\geq {:.1f}$'.format(hkey)
        else:
            text = '${:.1f} \\emdash {:.1f}$'.format(hkey, medges[hi + 1])
        
        ax.text(0.98, 0.98, text, fontsize=fontsize,\
                transform=ax.transAxes, horizontalalignment='right',\
                verticalalignment='top')
        # set limits for panels
        ylim = ax.get_ylim()
        ymax = _max
        ymin = max(ylim[0], ymax - 6.)
        ymax = ymax + 0.05 * (ymax - ymin)
        ax.set_ylim(ymin, ymax)
        
    # sync plot ranges
    #xlims = [ax.get_xlim() for ax in axes]
    #xmin = min([xlim[0] for xlim in xlims])
    #xmax = max([xlim[1] for xlim in xlims])
    #[ax.set_xlim(xmin, xmax) for ax in axes]

    # three most energetic ions have very low mean SB -> impose limits
    #ylims = [ax.get_ylim() for ax in axes]
    #ymin = -9. #min([ylim[0] for ylim in ylims])
    #ymax = 2. #max([ylim[1] for ylim in ylims])
    #[ax.set_ylim(ymin, ymax) for ax in axes]
    
    ## legend
    handles = [mlines.Line2D([], [], color=linecolors[line],\
                             label=nicenames_lines[line],\
                             path_effects=pe2, linewidth=lw2)\
               for line in lines]
    lax.legend(handles=handles, fontsize=fontsize, loc=l_loc,\
               bbox_to_anchor=l_bbox_to_anchor, ncol=l_ncols)
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
def plot_radprof3(mmin=10.0, numex=4):
    '''
    plot SB mean, median and scatter for lines and halo mass bins
    panels for differnt bins, different plots for different lines 
    
    input:
    ------
    numex:    number of individual examples (max 10, but 10 gives very crowded
              plots)
    mmin:     minimum halo mass to show (log10 Msun, 
              value options: 9.0, 9.5, 10., ... 14.)
    '''
    
    print('Values are calculated from 3.125^2 ckpc^2 pixels in 10 pkpc annuli')
    print('z=0.1, Ref-L100N1504, 6.25 cMpc slice Z-projection, SmAb, C2 kernel')
    print('Using max. 1000 (random) galaxies in each mass bin, centrals only')
    print('Black is for the stacked samples, colors are random individual galaxies')
    
    fontsize = 12
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    
    rfilebase = ol.pdir + 'radprof/' + 'radprof_stamps_emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5'
    xlabel = '$\\mathrm{r}_{\perp} \\; [\\mathrm{pkpc}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{photons}\\,\\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
    
    ys = [('mean',), ('perc', 50.), ('perc', 10.), ('perc', 90.)]
    kwargs_y_stack = {('mean',): {'linestyle': 'dashed', 'linewidth': 2.5,\
                                  'color': 'black'},\
                      ('perc', 10.): {'linestyle': 'dotted', 'linewidth': 2.5,\
                                      'color': 'black'},\
                      ('perc', 50.): {'linestyle': 'solid', 'linewidth': 2.5,\
                                      'color': 'black'},\
                      ('perc', 90.): {'linestyle': 'dotted', 'linewidth': 2.5,\
                                      'color': 'black'},\
                     }
    kwargs_y_indiv = {('mean',): {'linestyle': 'dashed', 'linewidth': linewidth,\
                                  'path_effects': patheff},\
                      ('perc', 10.): {'linestyle': 'dotted', 'linewidth': linewidth,\
                                  'path_effects': patheff},\
                      ('perc', 50.): {'linestyle': 'solid', 'linewidth': linewidth,\
                                  'path_effects': patheff},\
                      ('perc', 90.): {'linestyle': 'dotted', 'linewidth': linewidth,\
                                  'path_effects': patheff},\
                     }
    legtags = {('mean',): 'mean',\
               ('perc', 50.): 'median',\
               ('perc', 10.): '$10^{\\mathrm{th}}$ perc.',\
               ('perc', 90.): '$90^{\\mathrm{th}}$ perc.',\
               }
    
    medges = np.arange(mmin, 14.1, 0.5)
    seltag_keys = {medges[i]: 'geq{:.1f}_le{:.1f}'.format(medges[i], medges[i + 1])\
                               if i < len(medges) - 1 else\
                               'geq{:.1f}'.format(medges[i])\
                    for i in range(len(medges))}
    seltags = [seltag_keys[key] for key in seltag_keys]
    
    #numlines = len(lines)
    nummasses = len(medges)
    ncols = 4
    nrows = (nummasses - 1) // ncols + 1
    figwidth = 11. 
    laxheight = 1.
    hspace = 0.15
    wspace = 0.28
    
    if ncols * nrows - nummasses >= 2:
        lax_below = False
        _nrows = nrows
        panelwidth = (figwidth - (ncols * wspace)) / ncols
        width_ratios = [panelwidth] * ncols
        panelheight = panelwidth    
        figheight = panelheight * nrows + hspace * (nrows - 1)
        height_ratios = [panelheight] * nrows 
        l_bbox_to_anchor = (0.5, 0.0)
        l_loc = 'lower center'
        l_ncols = ncols * nrows - nummasses
    else:
        lax_below = True
        _nrows = nrows + 1
        panelwidth = (figwidth - (ncols * wspace)) / ncols
        width_ratios = [panelwidth] * ncols 
        panelheight = panelwidth    
        figheight = panelheight * ncols + laxheight + hspace * nrows
        height_ratios = [panelheight] * nrows + [laxheight]
        l_bbox_to_anchor = (0.5, 0.80)
        l_loc = 'upper center'
        l_ncols = ncols 
    
    for line in lines:
        outname = mdir + 'radprof2d_10pkpc-annuli_L0100N1504_27_test3p5_SmAb_C2Sm_6p25slice_noEOS_to-2R200c_1000_centrals_' +\
                  'measurecomp_{}.pdf'.format(line)
                  
        fig = plt.figure(figsize=(figwidth, figheight))
        fig.suptitle(nicenames_lines[line], fontsize=fontsize)
        grid = gsp.GridSpec(ncols=ncols, nrows=_nrows, hspace=hspace, wspace=wspace,\
                            width_ratios=width_ratios, height_ratios=height_ratios)
        axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(nummasses)]
        if lax_below:
            lax = fig.add_subplot(grid[nrows, :])
        else:
            ind_min = ncols - (nrows * ncols - nummasses)
            lax = fig.add_subplot(grid[nrows - 1, ind_min:])
        lax.axis('off')
            #_l, _b, _w, _h = (_lax.get_position()).bounds
            #margin = panelwidth * 0.1 / figwidth
            #lax = fig.add_axes([_l + margin, _b + margin,\
            #                    _w - 2.* margin, _h - 2. * margin])
            
        labelax = fig.add_subplot(grid[:nrows, :ncols], frameon=False)
        labelax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        labelax.set_xlabel(xlabel, fontsize=fontsize)
        labelax.set_ylabel(ylabel, fontsize=fontsize)
        
        #clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
        #cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=medges,\
        #         orientation=c_orientation, clabel=clabel, fontsize=fontsize,\
        #         aspect=c_aspect)
        
        ## get lines
        filename = rfilebase.format(line=line)
        yvals, bins = readin_radprof(filename, seltags, ys, runit='pkpc', separate=False,\
                                     binset='binset_0', retlog=True)
        yvals_ind, bins_ind = readin_radprof(filename, seltags, ys, runit='pkpc', separate=True,\
                                     binset='binset_0', retlog=True)
        
        for hi, hkey in enumerate(medges):
            ax = axes[hi]
            labely = True # hi % ncols == 0
            labelx = True # nummasses -1 - hi < ncols
            pu.setticks(ax, fontsize=fontsize, labelleft=labely, labelbottom=labelx)
            ax.set_xscale('log')
            
            mtag = seltag_keys[hkey]
            
            _max = -100.
            
            # plot the stacked data
            for ytag in ys:
                ed = bins[mtag][ytag]
                vals = yvals[mtag][ytag]
                cens = ed[:-1] + 0.5 * np.diff(ed)
                ax.plot(cens, vals, **kwargs_y_stack[ytag])
                try:
                    tmax = np.max(vals[np.isfinite(vals)])
                    _max = max(_max, tmax)
                except ValueError: # no finite values
                    pass
            # scatter range stack
            ed = bins[mtag][('perc', 10.)]
            v1 = yvals[mtag][('perc', 10.)]
            v2 = yvals[mtag][('perc', 90.)]
            cens = ed[:-1] + 0.5 * np.diff(ed)
            ax.fill_between(cens, v1, v2, facecolor='black', alpha=0.1,\
                            edgecolor='none')
            
            # plot individual galaxy data
            gids = list(yvals_ind[mtag][ys[0]].keys())
            for gi, gid in enumerate(gids[:numex]):
                for ytag in ys:
                    ed = bins_ind[mtag][ytag][gid]
                    vals = yvals_ind[mtag][ytag][gid]
                    cens = ed[:-1] + 0.5 * np.diff(ed)
                    color = 'C{}'.format(gi % 10)
                    ax.plot(cens, vals, color=color, **kwargs_y_indiv[ytag])
                    try:
                        tmax = np.max(vals[np.isfinite(vals)])
                        _max = max(_max, tmax)
                    except ValueError: # no finite values
                        pass
                        
            if hi == len(medges) - 1:
                text = '$\\geq {:.1f}$'.format(hkey)
            else:
                text = '${:.1f} \\emdash {:.1f}$'.format(hkey, medges[hi + 1])
            
            ax.text(0.98, 0.98, text, fontsize=fontsize,\
                    transform=ax.transAxes, horizontalalignment='right',\
                    verticalalignment='top')
            # set limits for panels
            ylim = ax.get_ylim()
            ymax = _max
            ymin = max(ylim[0], ymax - 6.)
            ymax = ymax + 0.05 * (ymax - ymin)
            ax.set_ylim(ymin, ymax)
            
        # sync plot ranges
        #xlims = [ax.get_xlim() for ax in axes]
        #xmin = min([xlim[0] for xlim in xlims])
        #xmax = max([xlim[1] for xlim in xlims])
        #[ax.set_xlim(xmin, xmax) for ax in axes]
    
        # three most energetic ions have very low mean SB -> impose limits
        #ylims = [ax.get_ylim() for ax in axes]
        #ymin = -9. #min([ylim[0] for ylim in ylims])
        #ymax = 2. #max([ylim[1] for ylim in ylims])
        #[ax.set_ylim(ymin, ymax) for ax in axes]
        
        ## legend
        handles1 = [mlines.Line2D([], [],\
                                  label=legtags[ytag] + ' (stack)',\
                                  **kwargs_y_stack[ytag])\
                   for ytag in ys]
        lcs = []
        line = [[(0, 0)]]
        for ytag in ys:
            kwargs = kwargs_y_indiv[ytag].copy()
            subcols = [mpl.colors.to_rgba('C{}'.format(i)) for i in range(numex)]
            lc = mcol.LineCollection(line * len(subcols), colors=subcols,\
                                     label=legtags[ytag] + ' (indiv.)',\
                                     **kwargs)
            lcs.append(lc)
        
        lax.legend(handles=handles1 + lcs, fontsize=fontsize, loc=l_loc,\
                   bbox_to_anchor=l_bbox_to_anchor, ncol=l_ncols,\
                   handler_map={type(lc): pu.HandlerDashedLines()})
        
        plt.savefig(outname, format='pdf', bbox_inches='tight')
        

def plot_emtables(z=0.1):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
      
    outname = mdir + 'emtables_z{}_HM01_ionizedmu.pdf'.format(str(z).replace('.', 'p'))
    
    #ioncolors.update({'he2': 'darkgoldenrod'})
    Ts = {}
    Tmaxs = {}
    nHs = {}
    vals = {}
    maxvals = {}
    
    # eagle cosmology
    cosmopars = cosmopars_eagle.copy()
    cosmopars['z'] = z
    cosmopars['a'] = 1. / (1. + z)
    logrhob = np.log10(cu.rhocrit(0.) * c.omegabaryon / (1. + z)**3)
        
    fracv = 0.1
    
    for line in lines:   
        em, logTK, lognHcm3 = cu.findemtables(ol.elements_ion[line],z) 
        em = em[:,:,ol.line_nos_ion[line]]
        vals[line] = em
        nHs[line] = lognHcm3
        Ts[line] = logTK
        indmaxfrac = np.argmax(em[:, -1])
        maxem = em[indmaxfrac, -1]
        Tmax = logTK[indmaxfrac]
        Tmaxs[line] = Tmax
    
        xs = pu.find_intercepts(em[:, -1], logTK, np.log10(fracv) + maxem)
        msg = 'Line {line} has maximum emissivity (solar abunds) {maxv:3f}, at log T[K] = {T:.1f}, max range is {rng}'
        print(msg.format(line=line, maxv=maxem, T=Tmax, rng=str(xs)))
        maxvals[line] = maxem
    
    numpanels = 4
    fig, axes = plt.subplots(ncols=1, nrows=numpanels, figsize=(5.5, 12.),\
                             gridspec_kw={'hspace': 0.})
    xlim = (-8., -1.5)
    ylim = (3.5, 8.)
    [ax.set_xlim(*xlim) for ax in axes]
    [ax.set_ylim(*ylim) for ax in axes]

    axions = {0: ['c5r', 'c6', 'n6r', 'n7'],\
              1: ['o7r', 'o7ix', 'o7iy', 'o7f', 'o8'],\
              2: ['ne9r', 'ne10', 'mg11r', 'mg12', 'si13r'],\
              3: ['fe17-other1', 'fe19', 'fe17', 'fe18']}
    lsargs = {'c5r':  {'linestyle': 'solid'},\
              'c6':   {'linestyle': 'dashed'},\
              'n6r':  {'linestyle': 'dotted'},\
              'n7':   {'linestyle': 'dashdot'},\
              'o7r':  {'linestyle': 'solid'},\
              'o7ix': {'dashes': [2, 2, 2, 2]},\
              'o7iy': {'dashes': [6, 2]},\
              'o7f':  {'linestyle': 'dashdot'},\
              'o8':   {'linestyle': 'dotted'},\
              'fe17-other1':  {'linestyle': 'solid'},\
              'fe19':  {'linestyle': 'dashed'},\
              'fe17':  {'linestyle': 'dotted'},\
              'fe18':  {'linestyle': 'dashdot'},\
              'si13r': {'linestyle': 'solid'},\
              'ne9r':  {'dashes': [2, 2, 2, 2]},\
              'ne10':  {'dashes': [6, 2]},\
              'mg11r': {'linestyle': 'dashdot'},\
              'mg12':  {'linestyle': 'dotted'},\
                  }
    axions = {key: sorted(axions[key], key=ol.line_eng_ion.get) for key in axions}
    
    xlabel = r'$\log_{10} \, \mathrm{n}_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$'
    ylabel = r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$'
    [ax.set_ylabel(ylabel, fontsize=fontsize) for ax in axes]
    axes[numpanels - 1].set_xlabel(xlabel, fontsize=fontsize)
    
    _lsargs = lsargs.copy()
    _lsargs = {key: _lsargs[key] if 'linestyle' not in _lsargs[key] else\
                    {'linestyles': _lsargs[key]['linestyle']}\
               for key in _lsargs}
    for axi, ax in enumerate(axes):
        pu.setticks(ax, fontsize=fontsize, right=False,\
                    labelbottom=(axi == numpanels - 1))
        
        ax.axvline(logrhob + np.log10(rho_to_nh), 0., 0.85, color='gray',\
                   linestyle='dashed', linewidth=1.5)
        for line in axions[axi]:
            ax.contourf(nHs[line], Ts[line], vals[line], colors=linecolors[line],\
                        alpha=0.1, linewidths=[3.],\
                        levels=[np.log10(fracv) + maxvals[line], maxvals[line] + 100.],\
                        **lsargs[line])
            
            ax.contour(nHs[line], Ts[line], vals[line], colors=linecolors[line],\
                       linewidths=[2.], levels=[np.log10(fracv) + maxvals[line]],\
                       **_lsargs[line])

            ax.axhline(Tmaxs[line], 0.92, 1., color=linecolors[line],\
                       linewidth=3., **lsargs[line])
            
        #bal = bals[ion]
        #maxcol = bal[-1, :]
        #diffs = bal / maxcol[np.newaxis, :]
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal == 0)] = 0.
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] = bal[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] / 1e-18
        #diffs = np.abs(np.log10(diffs))
            
        #mask = bal < 0.6 * fracv * maxfracs[ion] # 0.6 gets the contours to ~ the edges of the ion regions
        #diffs[mask] = np.NaN

        #ax.contour(nHs[ion], Ts[ion][np.isfinite(maxcol)], (diffs[:, np.isfinite(maxcol)]).T, levels=[np.log10(ciemargin)], linestyles=['solid'], linewidths=[1.], alphas=0.5, colors=ioncolors[ion])

        axy2 = ax.twinx()
        axy2.set_ylim(*ylim)
        mhalos = np.arange(9.0, 15.1, 0.5)
        Tvals = np.log10(cu.Tvir_hot(10**mhalos * c.solar_mass,\
                                     cosmopars=cosmopars))
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_yticks(Tvals)
        axy2.set_yticklabels(Tlabels)
        pu.setticks(axy2, fontsize=fontsize, left=False, right=True,\
                    labelleft=False, labelright=True)
        axy2.minorticks_off()
        axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$',\
                        fontsize=fontsize)
    
        handles = [mlines.Line2D([], [], label=nicenames_lines[line],\
                                 color=linecolors[line],\
                                 linewidth=2., **lsargs[line])\
                   for line in axions[axi]]
        ax.legend(handles=handles, fontsize=fontsize, ncol=2,\
                  bbox_to_anchor=(1.0, 0.0), loc='lower right', frameon=False)

    plt.savefig(outname, format='pdf', bbox_inches='tight')
    

def save_emcurves(lineset=None, z=0.1, nH='CIE'):
    '''
    get a table of line emissivity as a function of temperature 
    
    input:
    ------
    lineset: list of line names (make_maps_opts_locs.py), or None for global
             lines
    z:       redshift of the tables to use
    nH:      hydrogen number density (log10 cm**-3) to use (nearest value)
             'CIE' -> highest nH tabulated
    '''
    if lineset is None:
        lineset = lines
        
    elts = set([ol.elements_ion[line] for line in lineset])
    lines_elt = {elt: set([line if ol.elements_ion[line] == elt else None \
                           for line in lineset])\
                 for elt in elts}
    lines_elt = {elt: lines_elt[elt] - {None} for elt in lines_elt}
    
    curves = {}
    Ts = {}
    nHus = {}
    for elt in elts:
        _em, logTK, lognHcm3 = cu.findemtables(elt, z)
        for line in lines_elt[elt]:
            em = _em[:, :, ol.line_nos_ion[line]]
            nHs  = np.copy(lognHcm3)
            T    = np.copy(logTK)
            if nH == 'CIE':
                nHi = -1
            else:
                nHi = np.argmin(np.abs(nHs - nH))
            curve = em[:, nHi]
            nHu = nHs[nHi]
            
            Ts[line] = T
            curves[line] = curve
            nHus[line] = nHu
    
    T = Ts[lineset[0]]
    if not np.all([np.all(T == Ts[line]) for line in lineset]):
        raise RuntimeError('T values did not match between line tables')
    nHu = nHus[lineset[0]]
    if not np.all([nHu == nHus[line] for line in lineset]):
        raise RuntimeError('used nH values did not match between line tables')
    
    outdir = mdir + 'datasets/'
    fname = 'emissivitycurves_z-{z}_nH-{nH}.txt'.format(z=z, nH=nH)
    pre = '#table: line emissvity log10 Lambda * nH**-2 * V**-1 [erg * cm**3 * s**-1]'
    pre = pre + '\n#tabulated as a function of log10 T [K] (column T)'
    pre = pre + '\n#nH [log10 cm**-3]: {nHu}'.format(nHu=nHu)
    pre = pre + '\n#z: {z}'.format(z=z)
    head = '\nT\t' + '\t'.join(lineset)
    fill = '\n{T}\t ' + '\t'.join(['{{{line}}}'.format(line=line)\
                                      for line in lineset])
    with open(outdir + fname, 'w') as fo:     
        fo.write(pre)
        fo.write(head)
        for i in range(len(T)):
            fo.write(fill.format(T=T[i], **{line: curves[line][i] for line in lineset}))
    
        
def plot_emcurves(z=0.1):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
      
    outname = mdir + 'emcurves_z{}_HM01_ionizedmu.pdf'.format(str(z).replace('.', 'p'))
    
    #ioncolors.update({'he2': 'darkgoldenrod'})
    Ts = {}
    Tmaxs = {}
    #nHs = {}
    vals = {}
    maxvals = {}
    
    # eagle cosmology
    cosmopars = cosmopars_eagle.copy()
    cosmopars['z'] = z
    cosmopars['a'] = 1. / (1. + z)
        
    fracv = 0.1
    
    ddir = mdir + 'datasets/'
    fname = 'emissivitycurves_z-{z}_nH-{nH}.txt'.format(z=z, nH='CIE')
    cdata = pd.read_csv(ddir + fname, sep='\t', index_col='T', header=4)
    logTK = cdata.index
    for line in lines:   
        em = np.array(cdata[line])
        #em = em[:,:,ol.line_nos_ion[line]]
        vals[line] = em
        #nHs[line] = lognHcm3
        Ts[line] = logTK
        indmaxfrac = np.argmax(em)
        maxem = em[indmaxfrac]
        Tmax = logTK[indmaxfrac]
        Tmaxs[line] = Tmax
    
        xs = pu.find_intercepts(em[:], logTK, np.log10(fracv) + maxem)
        msg = 'Line {line} has maximum emissivity (solar abunds) {maxv:3f}, at log T[K] = {T:.1f}, max range is {rng}'
        print(msg.format(line=line, maxv=maxem, T=Tmax, rng=str(xs)))
        maxvals[line] = maxem
    
    ncols = 2
    nrows = (len(linesets) - 1) // ncols + 1
    fig = plt.figure(figsize=(11., 7.))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols,  hspace=0.45, wspace=0.0)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(len(linesets))]
    
    lsargs = lineargs_sets.copy()
    linelabels = nicenames_lines.copy()
    linelabels['fe17-other1'] = 'Fe XVII\n(15.10 A)'
    linelabels['fe17'] = 'Fe XVII\n(17.05 A)'
    
    xlim = (5.3, 8.5)
    ylim = (-28., -23.)
    
    ylabel = '$\log_{10} \\, \\Lambda \,/\, \\mathrm{n}_{\\mathrm{H}}^{2} \\,/\\, \\mathrm{V} \\; [\\mathrm{erg} \\, \\mathrm{cm}^{3} \\mathrm{s}^{-1}]$'
    xlabel = r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$'
    xlabel2 = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}}(\\mathrm{T}_{\\mathrm{200c}}) \\; [\\mathrm{M}_{\\odot}]$'
    
    indvals = [] #[-5.]
    lsargs2 = [\
               #{'linewidth': 1.5, 'alpha': 1.},\
               #{'linewidth': 1.0, 'alpha': 0.5},\
               {'linewidth': 2.5,  'alpha': 1.0},\
               #{'linewidth': 3.,  'alpha': 0.5},\
               ]
    #offsets = {line: 0.05 *  (li - 0.5 *len(lines)) / len(lines) \
    #           for li, line in enumerate(lines)}
    
    for axi, ax in enumerate(axes):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        
        labely = axi % ncols == 0
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        pu.setticks(ax, fontsize=fontsize, top=False, labeltop=False, labelleft=labely)
        _lines = linesets[axi]
        ax.grid(b=True)
        
        for line in _lines:
            # nH values
            #for iv, nH in enumerate(indvals):
            #    nhi = np.where(np.isclose(nHs[line], nH))[0][0]
            #    emvals = vals[line][:, nhi]
            #    kwargs = lsargs[line].copy()
            #    kwargs.update(lsargs2[iv])
            #    pe = getoutline(kwargs['linewidth'])
            #    ax.plot(Ts[line], emvals, color=linecolors[line], path_effects=pe,\
            #            **kwargs)
            # CIE
            emvals = vals[line][:]
            kwargs = lsargs[line].copy()
            kwargs.update(lsargs2[-1])
            pe = getoutline(kwargs['linewidth'])
            
            ax.plot(Ts[line], emvals,\
                    path_effects=pe, **kwargs)
            
            ax.axvline(Tmaxs[line], 0.92, 1.,\
                       linewidth=3., **lsargs[line])
    
        axy2 = ax.twiny()
        axy2.set_xlim(*xlim)
        mhalos = np.arange(11.5, 15.1, 0.5)
        Tvals = np.log10(cu.Tvir_hot(10**mhalos * c.solar_mass,\
                                     cosmopars=cosmopars))
        limsel = Tvals >= xlim[0]
        Tvals = Tvals[limsel]
        mhalos = mhalos[limsel]
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_xticks(Tvals)
        axy2.set_xticklabels(Tlabels)
        pu.setticks(axy2, fontsize=fontsize, left=False, right=False,\
                    top=True, bottom=False,\
                    labelleft=False, labelright=False,\
                    labeltop=True, labelbottom=False)
        axy2.minorticks_off()
        axy2.set_xlabel(xlabel2,\
                        fontsize=fontsize)
        pe = getoutline(2.)
        handles = [mlines.Line2D([], [], label=linelabels[line],\
                                 linewidth=2., path_effects=pe, **lsargs[line])\
                   for line in _lines]
        #handles2 = [mlines.Line2D([], [], label='$\\mathrm{{n}}_{{\\mathrm{{H}}}} = {:.0f}$'.format(nH),\
        #                         color='black', **lsargs2[iv])\
        #           for iv, nH in enumerate(indvals)]
        #handles3 = [mlines.Line2D([], [], label='CIE',\
        #                         color='black', **lsargs2[-1])]
        ax.legend(handles=handles, fontsize=fontsize, ncol=1,\
                  bbox_to_anchor=(1.0, 1.0), loc='upper right', frameon=True)

    plt.savefig(outname, format='pdf', bbox_inches='tight')

###############################################################################
##################### stuff coming from 3D profiles ###########################
###############################################################################
    
def plot_luminosities(addedges=(0., 1.), toSB=False, plottype='all'):
    '''
    plottype: 'all':    L medians and scatter for the different ions (1 panel)
              'lines':  L distriutions (1 line per panel)
              'SFfrac': L from SF gas / L total (distriutions, 1 line per panel) 
    toSB:     convert Luminosities to SB, assuming uniform emission within 
              the edges (in projection) at z = 0.1
    '''
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = 'luminosities_{tp}_{mi}-{ma}-R200c'.format(tp=plottype,\
                            mi=addedges[0], ma=addedges[1])
    if toSB:
        outname = outname + '_as-SB'
    outname = outname.replace('.', 'p')
    outname = outdir + outname + '.pdf'
    
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 }
    lsargs = {'c5r':  {'linestyle': 'solid'},\
              'c6':   {'linestyle': 'dashed'},\
              'n6r':  {'linestyle': 'dotted'},\
              'n7':   {'linestyle': 'dashdot'},\
              'o7r':  {'linestyle': 'solid'},\
              'o7ix': {'dashes': [2, 2, 2, 2]},\
              'o7iy': {'dashes': [6, 2]},\
              'o7f':  {'linestyle': 'dashdot'},\
              'o8':   {'linestyle': 'dotted'},\
              'fe17-other1':  {'linestyle': 'solid'},\
              'fe19':  {'linestyle': 'dashed'},\
              'fe17':  {'linestyle': 'dotted'},\
              'fe18':  {'linestyle': 'dashdot'},\
              'si13r': {'linestyle': 'solid'},\
              'ne9r':  {'dashes': [2, 2, 2, 2]},\
              'ne10':  {'dashes': [6, 2]},\
              'mg11r': {'linestyle': 'dashdot'},\
              'mg12':  {'linestyle': 'dotted'},\
                  }
    
    filename = ol.pdir + 'luminosities_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_SmAb.hdf5'%(str(addedges[0]), str(addedges[1]))
    with h5py.File(filename, 'r') as fi:
        galids_l = fi['galaxyids'][:]
        lines = [line.decode() for line in fi.attrs['lines']]
        lums = fi['luminosities'][:]
            
    file_galdata = '/net/luttero/data2/imgs/CGM/3dprof/halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    masses = np.array(galdata_all['M200c_Msun'][galids_l])
    
    if plottype == 'all':
        mbins = np.array(list(np.arange(10., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])
    else:
        mbins = np.arange(10., 14.65, 0.1)
    
    if toSB:

        rs_in_pkpc = addedges[0] * cu.R200c_pkpc(masses, cosmopars)
        rs_out_pkpc = addedges[1] * cu.R200c_pkpc(masses, cosmopars)
        
        zcalc = cosmopars['z']
        comdist = cu.comoving_distance_cm(zcalc, cosmopars=cosmopars)
        longlen = 50. * c.cm_per_mpc
        if comdist > longlen: # even at larger values, the projection along z-axis = projection along sightline approximation will break down
            ldist = comdist * (1. + zcalc) # luminosity distance
            adist = comdist / (1. + zcalc) # angular size distance
        else:
            ldist = longlen * (1. + zcalc)
            adist = longlen / (1. + zcalc)
        # conversion (x, y are axis placeholders and may actually represent different axes in the simulation, as with numpix_x, and numpix_y)
        angle_in  = rs_in_pkpc * 1e-3 * c.cm_per_mpc / adist
        angle_out = rs_out_pkpc * 1e-3 * c.cm_per_mpc / adist
        # solid angle (alpha = radius on circle in rad) = 2 pi (1 - cos(alpha))
        if cosmopars['z'] >= 0.08: # taylor
            Omega = np.pi * (angle_out**2 - angle_in**2)
        else:
            Omega = 2 * np.pi * (np.cos(angle_in) - np.cos(angle_out))
        lums = np.sum(lums, axis=2)
        pnorms = (1. + cosmopars['z']) / np.array([ol.line_eng_ion[line] for line in lines])
        lums *= pnorms[np.newaxis, :] / Omega[:, np.newaxis] \
                / ( 4. * np.pi * ldist**2)
        lums = np.log10(lums)
        ylabel = '$ \\langle\\mathrm{SB}\\rangle \\;[\\mathrm{photons} \\,\\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
        
        minymin = -11.
    else:
        if plottype in ['lines', 'all']:
            lums = np.sum(lums, axis=2)
            lums = np.log10(lums) - np.log10(1. + cosmopars['z'])
            ylabel = '$\\mathrm{L}_{\\mathrm{obs}} \\; [\\mathrm{erg} \\,\\mathrm{s}^{-1}]$'
            minymin = 28.
            
    if plottype == 'SFfrac':
        lums = lums[:, :, 1] / np.sum(lums, axis=2)
        ylabel = '$\\mathrm{L}_{\\mathrm{SF}} \\, / \\, \\mathrm{L}_{\\mathrm{tot}}$'
        minymin = 0.
        
    xlabel = '$\\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\odot}]$' 
    
    bininds = np.digitize(np.log10(masses), mbins)
    bincen = mbins[:-1] + 0.5 * np.diff(mbins) 
        
    if plottype == 'all':
        fig = plt.figure(figsize=(5.5, 9.))
        grid = gsp.GridSpec(nrows=2, ncols=1,  hspace=0.25, wspace=0.0, \
                            height_ratios=[5., 4.])
        ax = fig.add_subplot(grid[0, 0])
        lax = fig.add_subplot(grid[1, 0]) 
        
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        pu.setticks(ax, fontsize)
        for li, line in enumerate(lines):
            med = [np.median(lums[bininds == i, li]) for i in range(1, len(mbins))]
            ax.plot(bincen, med, color=linecolors[line], label=nicenames_lines[line],\
                    **lsargs[line])
        
        handles, labels = ax.get_legend_handles_labels()
        lax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(0.5, 0.),\
                   loc='lower center', ncol=2)
        lax.axis('off')
        
        if minymin is not None:
            y0, y1 = ax.get_ylim()
            y0 = max(y0, minymin)
            ax.set_ylim(y0, y1)
            
    elif plottype in ['lines', 'SFfrac']:
        panelheight = 3.
        panelwidth = 3.
        hspace = 0.
        wspace = 0.
        legheight = 2.
        nlines = len(lines)
        
        ncols = 4
        nrows = (nlines - 1) // ncols + 1
        figheight = panelheight * nrows + hspace * (nrows - 1) + legheight
        figwidth = panelwidth * ncols + wspace * (ncols - 1) 
        height_ratios = [panelheight] * nrows + [legheight]
        width_ratios = [panelwidth] * ncols
        
        fig = plt.figure(figsize=(figwidth, figheight))
        grid = gsp.GridSpec(nrows=nrows + 1, ncols=ncols, hspace=hspace,\
                            wspace=wspace, height_ratios=height_ratios,\
                            width_ratios=width_ratios)
        axes = [fig.add_subplot(grid[li // ncols, li % ncols])\
                for li in range(nlines)]
        lax = fig.add_subplot(grid[nrows, :])
        
        alpha = 0.5
        color = 'C0'
        percv = [2., 10., 50., 90., 98.]
        
        bincen = np.append(0.5 * mbins[0] - 0.5 * mbins[1], bincen)
        bincen = np.append(bincen, 0.5 * mbins[-1] - 0.5 * mbins[-2])
        
        for li, line in enumerate(lines):
            ax = axes[li]
            labelleft = False
            labelbottom = False
            if li % ncols == 0:
                labelleft = True
                ax.set_ylabel(ylabel, fontsize=fontsize)
            if li >= nlines - ncols:
                labelbottom = True
                ax.set_xlabel(xlabel, fontsize=fontsize)
            pu.setticks(ax, fontsize, labelbottom=labelbottom, labelleft=labelleft)
            ax.text(0.98, 0.98, nicenames_lines[line], fontsize=fontsize,\
                    horizontalalignment='right', verticalalignment='top',\
                    transform=ax.transAxes)
            
            percs, outliers, xmininds = \
                pu.get_perc_and_points(np.log10(masses), lums[:, li], mbins,\
                        percentiles=percv,\
                        mincount_x=50,\
                        getoutliers_y=True, getmincounts_x=True,\
                        x_extremes_only=True)
            #percs = percs[1:-1]
            for si in range(len(percv) // 2):
                ax.fill_between(bincen[xmininds - 1], percs[xmininds - 1, si],\
                                percs[xmininds - 1, len(percv) - 1 - si],\
                                color=color, alpha=alpha)
            if len(percv) % 2 == 1:
                ax.plot(bincen, percs[:, len(percv) // 2], color=color)
            alpha_ol = alpha**((len(percv) - 1) // 2)
            ax.scatter(outliers[0], outliers[1], color=color, alpha=alpha_ol,\
                       s=10.)
        # legend
        handles = [mlines.Line2D([], [], color=color, label='{:.0f}%%'.format(percv[len(percv) // 2]))]
        handles += [mpatch.Patch(facecolor=color, alpha=alpha**i,\
                                 label='{:.0f}%%'.format(percv[len(percv) - 1 - i] - percv[i]))\
                    for i in range((len(percv) - 1) // 2)]
        lax.axis('off')
        lax.legend(handles, fontsize=fontsize, ncol=4)
        
        # sync lims
        xlims = [ax.get_xlim() for ax in axes]
        x0 = np.min([xl[0] for xl in xlims])
        x1 = np.max([xl[1] for xl in xlims])
        [ax.set_xlim(x0, x1) for ax in axes]
        
        ylims = [ax.get_ylim() for ax in axes]
        y0 = np.min([yl[0] for yl in ylims])
        y1 = np.max([yl[1] for yl in ylims])
        if minymin is not None:
            y0 = max(minymin, y0)
        [ax.set_ylim(y0, y1) for ax in axes]
        
    plt.savefig(outname, format='pdf', bbox_inches='tight')

def plot_luminosities_nice(addedges=(0., 1.)):
    '''
    '''
    
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = 'luminosities_nice_{mi}-{ma}-R200c'.format(\
                            mi=addedges[0], ma=addedges[1])
    outname = outname.replace('.', 'p')
    outname = outdir + outname + '.pdf'
    
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 }
    lsargs = lineargs_sets.copy()
    linewidth = 2.
    patheff = getoutline(linewidth)

    linelabels = nicenames_lines.copy()
    linelabels['fe17-other1'] = 'Fe XVII\n(15.10 A)'
    linelabels['fe17'] = 'Fe XVII\n(17.05 A)'
    
    filename = ol.pdir + 'luminosities_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_SmAb.hdf5'%(str(addedges[0]), str(addedges[1]))
    with h5py.File(filename, 'r') as fi:
        galids_l = fi['galaxyids'][:]
        lines = [line.decode() for line in fi.attrs['lines']]
        lums = fi['luminosities'][:]
            
    file_galdata = '/net/luttero/data2/imgs/CGM/3dprof/halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    masses = np.array(galdata_all['M200c_Msun'][galids_l])
    
    mbins = np.array(list(np.arange(11., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])

    lums = np.sum(lums, axis=2)
    lums = np.log10(lums) - np.log10(1. + cosmopars['z'])
    ylabel = '$\\mathrm{L}_{\\mathrm{obs}} \\; [\\mathrm{erg} \\,\\mathrm{s}^{-1}]$'
    ylim = (32.1, 42)
             
    xlabel = '$\\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\odot}]$' 
    
    bininds = np.digitize(np.log10(masses), mbins)
    bincen = mbins[:-1] + 0.5 * np.diff(mbins) 
    
    ncols = 1
    nrows = len(linesets)
    fig = plt.figure(figsize=(5.5, 12.))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0.0)
    axes = [fig.add_subplot(grid[i, 0]) for i in range(nrows)]
    
    for axi, ax in enumerate(axes):
        lineset = linesets[axi]
        labelx = axi >= len(linesets) - ncols
        labely = axi % ncols == 0
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        pu.setticks(ax, fontsize, labelleft=labely, labelbottom=labelx)
        ax.grid(b=True
                )
        for li, line in enumerate(lines):
            if line not in lineset:
                continue
            med = [np.median(lums[bininds == i, li]) for i in range(1, len(mbins))]
            ax.plot(bincen, med, label=linelabels[line], linewidth=linewidth,\
                    path_effects=patheff, **lsargs[line])
            
            ud = [np.percentile(lums[bininds == i, li], [10., 90.]) for i in range(1, len(mbins))]
            ud = np.array(ud).T
            ud[0, :] = med - ud[0, :]
            ud[1, :] = ud[1, :] - med
            lsi = np.where([l == line for l in lineset])[0][0]
            cycle = len(lineset)
            #print(lsi)
            #print(cycle)
            sl = slice(lsi, None, cycle) # avoid overlapping error bars
            ax.errorbar(bincen[sl], med[sl], yerr=ud[:, sl],\
                        linewidth=linewidth, linestyle=None,\
                        path_effects=patheff,\
                        **lsargs[line])
            
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(1.0, 0.),\
                  loc='lower right', ncol=2)     
    # sync lims
    xlims = [ax.get_xlim() for ax in axes]
    x0 = np.min([xl[0] for xl in xlims])
    x1 = np.max([xl[1] for xl in xlims])
    [ax.set_xlim(x0, x1) for ax in axes]
    
    #ylims = [ax.get_ylim() for ax in axes]
    #y0 = np.min([yl[0] for yl in ylims])
    #y1 = np.max([yl[1] for yl in ylims])
    [ax.set_ylim(*ylim) for ax in axes]
        
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    

def plot_sffrac_corr(addedges=(0., 1.)):
    '''
    fraction of L from SF gas as a function of total lumninosity
    '''
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = 'SFfrac_lum_{mi}-{ma}-R200c'.format(\
                            mi=addedges[0], ma=addedges[1])
    
    set_minfrac = 1e-15
    set_minL = 1e-50
    
    outname = outname.replace('.', 'p')
    outname = outdir + outname + '.pdf'
    
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 }
    lsargs = {'c5r':  {'linestyle': 'solid'},\
              'c6':   {'linestyle': 'dashed'},\
              'n6r':  {'linestyle': 'dotted'},\
              'n7':   {'linestyle': 'dashdot'},\
              'o7r':  {'linestyle': 'solid'},\
              'o7ix': {'dashes': [2, 2, 2, 2]},\
              'o7iy': {'dashes': [6, 2]},\
              'o7f':  {'linestyle': 'dashdot'},\
              'o8':   {'linestyle': 'dotted'},\
              'fe17-other1':  {'linestyle': 'solid'},\
              'fe19':  {'linestyle': 'dashed'},\
              'fe17':  {'linestyle': 'dotted'},\
              'fe18':  {'linestyle': 'dashdot'},\
              'si13r': {'linestyle': 'solid'},\
              'ne9r':  {'dashes': [2, 2, 2, 2]},\
              'ne10':  {'dashes': [6, 2]},\
              'mg11r': {'linestyle': 'dashdot'},\
              'mg12':  {'linestyle': 'dotted'},\
                  }
    
    filename = ol.pdir + 'luminosities_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_SmAb.hdf5'%(str(addedges[0]), str(addedges[1]))
    with h5py.File(filename, 'r') as fi:
        galids_l = fi['galaxyids'][:]
        lines = [line.decode() for line in fi.attrs['lines']]
        lums = fi['luminosities'][:]
    # these are galaxies with M200c < 10^10.25 Msun, stellar mass zero
    weirdones = np.any(np.isnan(lums), axis=(1, 2))       
    #file_galdata = '/net/luttero/data2/imgs/CGM/3dprof/halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    #galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    #masses = np.array(galdata_all['M200c_Msun'][galids_l])
    #print(np.log10(masses[weirdones]))
    lums = lums[np.logical_not(weirdones), :, :]
    
    # mbins = np.array(list(np.arange(10., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])   
    Ltot = np.sum(lums, axis=2)
    Ltot[Ltot == 0.] = set_minL
    Ltot = np.log10(Ltot)
    print(np.any(np.isnan(Ltot)))

    sffrac = lums[:, :, 1] / np.sum(lums, axis=2)
    sffrac[np.logical_and(sffrac == 0., np.sum(lums, axis=2) == 0.)] = set_minfrac
    sffrac[sffrac == 0.] = set_minfrac
    sffrac[sffrac < set_minfrac] = set_minfrac # the difference between 1e-15 and 1e-65 really doesn't matter here
    sffrac = np.log10(sffrac)
    print(np.any(np.isnan(sffrac)))
    
    
    
    
    
    xlabel = '$\\log_{10} \\, \\mathrm{L}_{\\mathrm{rest}} \\; [\\mathrm{erg} \\,\\mathrm{s}^{-1}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{L}_{\\mathrm{SF}} \\, / \\, \\mathrm{L}_{\\mathrm{tot}}$'
        
    panelheight = 3.
    panelwidth = 3.
    hspace = 0.
    wspace = 0.
    legheight = 2.
    nlines = len(lines)
    
    ncols = 4
    nrows = (nlines - 1) // ncols + 1
    figheight = panelheight * nrows + hspace * (nrows - 1) + legheight
    figwidth = panelwidth * ncols + wspace * (ncols - 1) 
    height_ratios = [panelheight] * nrows + [legheight]
    width_ratios = [panelwidth] * ncols
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(nrows=nrows + 1, ncols=ncols, hspace=hspace,\
                        wspace=wspace, height_ratios=height_ratios,\
                        width_ratios=width_ratios)
    axes = [fig.add_subplot(grid[li // ncols, li % ncols])\
            for li in range(nlines)]
    lax = fig.add_subplot(grid[nrows, :])
        
    alpha = 0.5
    color = 'C0'
    percv = [2., 10., 50., 90., 98.]
    deltaL = 0.2
        
    for li, line in enumerate(lines):
        ax = axes[li]
        labelleft = False
        labelbottom = False
        if li % ncols == 0:
            labelleft = True
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if li >= nlines - ncols:
            labelbottom = True
            ax.set_xlabel(xlabel, fontsize=fontsize)
        pu.setticks(ax, fontsize, labelbottom=labelbottom, labelleft=labelleft)
        ax.text(0.98, 0.98, nicenames_lines[line], fontsize=fontsize,\
                horizontalalignment='right', verticalalignment='top',\
                transform=ax.transAxes)
        
        minL = np.min(Ltot)
        maxL = np.max(Ltot)
        
        imin = np.floor(minL / deltaL)
        imax = np.ceil(maxL / deltaL)
        Lbins = np.arange(deltaL * (imin - 1), deltaL * (imax + 1.5), deltaL)
        bincen = np.append(0.5 * Lbins[0] - 0.5 * Lbins[1], Lbins[:-1] + 0.5 * np.diff(Lbins))
        bincen = np.append(bincen, 0.5 * Lbins[-1] - 0.5 * Lbins[-2]) 
        percs, outliers, xmininds = \
            pu.get_perc_and_points(Ltot[:, li], sffrac[:, li],\
                    Lbins,\
                    percentiles=percv,\
                    mincount_x=50,\
                    getoutliers_y=True, getmincounts_x=True,\
                    x_extremes_only=True)
        for si in range(len(percv) // 2):
            ax.fill_between(bincen[xmininds], percs[xmininds, si],\
                            percs[xmininds, len(percv) - 1 - si],\
                            color=color, alpha=alpha)
        if len(percv) % 2 == 1:
            ax.plot(bincen, percs[:, len(percv) // 2], color=color)
        alpha_ol = alpha**((len(percv) - 1) // 2)
        ax.scatter(outliers[0], outliers[1], color=color, alpha=alpha_ol,\
                   s=10.)
        ax.axhline(np.log10(set_minfrac), color='gray', linestyle='dotted')
        ax.axvline(np.log10(set_minL), color='gray', linestyle='dotted')
    
        # legend
        handles = [mlines.Line2D([], [], color=color, label='{:.0f}%%'.format(percv[len(percv) // 2]))]
        handles += [mpatch.Patch(facecolor=color, alpha=alpha**i,\
                                 label='{:.0f}%%'.format(percv[len(percv) - 1 - i] - percv[i]))\
                    for i in range((len(percv) - 1) // 2)]
        lax.axis('off')
        #lax.legend(handles, fontsize=fontsize, ncol=4)
        
        # sync lims
        xlims = [ax.get_xlim() for ax in axes]
        x0 = np.min([xl[0] for xl in xlims])
        x1 = np.max([xl[1] for xl in xlims])
        [ax.set_xlim(x0, x1) for ax in axes]
        
        ylims = [ax.get_ylim() for ax in axes]
        y0 = np.min([yl[0] for yl in ylims])
        y1 = np.max([yl[1] for yl in ylims])
        #if minymin is not None:
        #    y0 = max(minymin, y0)
        [ax.set_ylim(y0, y1) for ax in axes]
        
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
    
def plot3Dprof_overview(weighttype='Mass', stack='addnormed-R200c'):
    '''
    plot: cumulative profile of weight, [ion number density profile], 
          rho profile, T profile
    rows show different halo mass ranges
    
    input:
    ------
    stack:      'addnormed-R200c' or 'add'
                'add': shading/cumulative is in cgs units / proper sizes, 
                       L in erg/s (rest-frame)
    weighttype: 'Mass' or an ion name
    '''
    inclSF = True #False is not implemented in the histogram extraction
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = outdir + 'overview_radprof_L0100N1504_27_Mh0p5dex_1000_{}_{}_{stack}.pdf'.format(\
                                                                     weighttype, 'wSF' if inclSF else 'nSF',\
                                                                     stack=stack)
    defaultelt = 'Oxygen' #for M, V weighting
    
    print('Using parent element metallicity, otherwise, {}'.format(defaultelt))
    
    fontsize = 12
    cmap = pu.truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    cmap.set_under(cmap(0.))
    percentiles = [0.1, 0.50, 0.9]
    print('Showing percentiles ' + str(percentiles))
    linestyles = ['dashed', 'solid', 'dashed']
    
    rbinu = 'R200c'
    combmethod = stack

    # snapshot 27
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
    if weighttype not in ['Mass', 'Volume']:
        line = weighttype
    else:
        line = weighttype
    wname = nicenames_lines[line]  if line in nicenames_lines else \
            r'\mathrm{Mass}' if weighttype == 'Mass' else \
            r'\mathrm{Volume}' if weighttype == 'Volume' else \
            None
    wnshort = 'M' if weighttype == 'Mass' else\
              'V' if weighttype == 'Volume' else\
              'L'
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'n': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z}$',\
                'weight': r'$\log_{10} \, \mathrm{%s}(< r) \,/\, \mathrm{%s}(< \mathrm{R}_{\mathrm{200c}})$'%(wnshort, wnshort) \
                          if combmethod == 'addnormed-R200c' else \
                          r'$\log_{10} \, \mathrm{%s}_{\mathrm{tot}}$'%wnshort
                }
    if combmethod == 'addnormed-R200c': 
        clabel = r'$\log_{10} \, \left\langle %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}}) \right\rangle \, / \,$'%(wnshort, wnshort) + 'bin size'
    else:
        clabel = r'$\log_{10} \, \mathrm{%s} \, / \,$'%(wnshort) + 'bin size'
        
    if weighttype in ol.elements_ion.keys():
        filename = ol.ndir + 'particlehist_Luminosity_{line}_L0100N1504_27_test3.6_SmAb_T4EOS_galcomb.hdf5'.format(line=line)
        nprof = 4
        elt = string.capwords(ol.elements_ion[line])
        title = r'$\mathrm{L}(\mathrm{%s})$ and $\mathrm{L}(\mathrm{%s})$-weighted profiles'%(wname, wname)
        tgrpns = {'T': '3Dradius_Temperature_T4EOS_StarFormationRate_T4EOS',\
                  'n': '3Dradius_Niondens_hydrogen_SmAb_T4EOS_StarFormationRate_T4EOS',\
                  'Z': '3Dradius_SmoothedElementAbundance-{elt}_T4EOS_StarFormationRate_T4EOS'.format(\
                                                          elt=elt),\
                  }
        axns  = {'r3d':  '3Dradius',\
                 'T':    'Temperature_T4EOS',\
                 'n':    'Niondens_hydrogen_SmAb_T4EOS',\
                 'Z':    'SmoothedElementAbundance-{elt}_T4EOS'.format(elt=elt),\
                }
        axnl = ['n', 'T', 'Z']
    else:
        elt = defaultelt
        if weighttype == 'Volume':
            filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.6_T4EOS_galcomb.hdf5'%('propvol')
        else:
            filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.6_T4EOS_galcomb.hdf5'%(weighttype)
        nprof = 4
        title = r'%s and %s-weighted profiles'%(weighttype, weighttype)
        tgrpns = {'T': '3Dradius_Temperature_T4EOS_StarFormationRate_T4EOS',\
                  'n': '3Dradius_Niondens_hydrogen_SmAb_T4EOS_StarFormationRate_T4EOS',\
                  'Z': '3Dradius_SmoothedElementAbundance-{elt}_T4EOS_StarFormationRate_T4EOS'.format(elt=elt),\
                  }
        axns = {'r3d':  '3Dradius',\
                'T':    'Temperature_T4EOS',\
                'n':    'Niondens_hydrogen_SmAb_T4EOS',\
                'Z':    'SmoothedElementAbundance-{elt}_T4EOS'.format(elt=elt)}
        axnl = ['T', 'n', 'Z']
    
    tdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    fbase = 'filenames_L0100N1504_27_Mh0p5dex_1000_%s_%s.txt'
    if weighttype in ol.elements_ion:
        wt = 'em-' + weighttype
    else:
        wt = weighttype
    file_galsin = {'n': tdir + fbase%(wt, 'nrprof'),\
                   'T': tdir + fbase%(wt, 'Trprof'),\
                   'Z': tdir + fbase%(wt, 'Zrprof') if weighttype in ol.elements_ion.keys() else\
                        tdir + fbase%(wt, '{elt}-rprof'.format(elt=elt)),\
                   }
    file_galdata = '/net/luttero/data2/imgs/CGM/3dprof/halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    
    # generated randomly once
    #galids_per_bin = {11.0: [13074219,  3802158,  3978003,  3801075, 13588395, 11396298, 8769997,  12024375, 12044831, 12027193],\
    #                  11.5: [11599169, 11148475, 10435177,  9938601, 10198004,  9626515, 10925515, 10472334, 13823711, 11382071],\
    #                  12.0: [17795988,  8880354, 18016380,  8824646,  8976542,  8948515, 8593530,   9225418, 18167602,  8991644],\
    #                  12.5: [16907882, 16565965, 15934507, 15890726, 16643442, 16530723, 8364907,  14042157, 14837489, 14195766],\
    #                  13.0: [20009129, 20309020, 19987958, 19462909, 20474648, 19615775, 19488333, 19975482, 20519792, 19784480],\
    #                  13.5: [18816265, 18781590, 19634930, 18961507, 18927203, 19299051, 6004915,  20943533, 18849993, 21059563],\
    #                  14.0: [19701410, 10705995, 14978116, 21986362, 21109761, 21242351, 21573587, 21730536, 21379522],\
    #                 }
    galids_per_bin = {11.0: [13074219,  3802158,  3978003],\
                      11.5: [11599169, 11148475, 10435177],\
                      12.0: [17795988,  8880354, 18016380],\
                      12.5: [16907882, 16565965, 15934507],\
                      13.0: [20009129, 20309020, 19987958],\
                      13.5: [18816265, 18781590, 19634930],\
                      14.0: [21242351, 10705995, 21242351],\
                     }
    
    mgrpn = 'L0100N1504_27_Mh0p5dex_1000/%s-%s'%(combmethod, rbinu)
    
    # read in data: stacked histograms -> process to plottables
    hists_main = {}
    edges_main = {}
    galids_main = {}
    
    with h5py.File(filename, 'r') as fi:
        for profq in tgrpns:
            tgrpn = tgrpns[profq]
            grp = fi[tgrpn + '/' + mgrpn]
            sgrpns = list(grp.keys())
            massbins = [grpn.split('_')[-1] for grpn in sgrpns]    
            massbins = [[np.log10(float(val)) for val in binn.split('-')] for binn in massbins]
            
            for mi in range(len(sgrpns)):
                mkey = massbins[mi][0]
                
                grp_t = grp[sgrpns[mi]]
                hist = np.array(grp_t['histogram'])
                if bool(grp_t['histogram'].attrs['log']):
                    hist = 10**hist
                
                edges = {}
                axes = {}
                
                for axn in [profq, 'r3d']:
                    edges[axn] = np.array(grp_t[axns[axn] + '/bins'])
                    if not bool(grp_t[axns[axn]].attrs['log']):
                        edges[axn] = np.log10(edges[axn])
                    axes[axn] = grp_t[axns[axn]].attrs['histogram axis']  
                
                if mkey not in edges_main:
                    edges_main[mkey] = {}
                if mkey not in hists_main:
                    hists_main[mkey] = {}
                
                # apply normalization consisent with stacking method
                if rbinu == 'pkpc':
                    edges['r3d'] += np.log10(c.cm_per_mpc * 1e-3)
                
                if combmethod == 'addnormed-R200c':
                    if rbinu != 'R200c':
                        raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                    _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                    if len(_i) != 1:
                        raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                    _i = _i[0]
                    _a = list(range(len(hist.shape)))
                    _s = [slice(None, None, None) for dummy in _a]
                    _s[axes['r3d']] = slice(None, _i, None)
                    norm_t = np.sum(hist[tuple(_s)])
                elif combmethod == 'add':
                    norm_t = 1.
                hist *= (1. / norm_t)
                
                rax = axes['r3d']
                yax = axes[profq]
                
                edges_r = np.copy(edges['r3d'])
                edges_y = np.copy(edges[profq])
                
                hist_t = np.copy(hist)
                
                # deal with edge units (r3d is already in R200c units if R200c-stacked)
                if edges_r[0] == -np.inf: # reset centre bin position
                    edges_r[0] = 2. * edges_r[1] - edges_r[2] 
                if edges_y[0] == -np.inf: # reset centre bin position
                    edges_y[0] = 2. * edges_y[1] - edges_y[2]
                if edges_y[-1] == np.inf: # reset centre bin position
                    edges_y[-1] = 2. * edges_y[-2] - edges_y[-3]
                    
                sax = list(range(len(hist_t.shape)))
                sax.remove(rax)
                sax.remove(yax)
                hist_t = np.sum(hist_t, axis=tuple(sax))
                if yax < rax:
                    hist_t = hist_t.T
                #hist_t /= (np.diff(edges_r)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
                
                hists_main[mkey][profq] = hist_t
                edges_main[mkey][profq] = [edges_r, edges_y]
                #print(hist_t.shape)
                
                # add in cumulative plot for the weight (from one of the profiles)
                if profq == 'n':
                    hist_t = np.copy(hist)
                    sax = list(range(len(hist_t.shape)))
                    sax.remove(rax)
                    hist_t = np.sum(hist_t, axis=tuple(sax))
                    hist_t = np.cumsum(hist_t)
                    hists_main[mkey]['weight'] = hist_t
                    edges_main[mkey]['weight'] = [edges_r[1:]]
                    
                
                galids_main[mkey] = np.array(grp_t['galaxyids'])
    
    # read in data: individual galaxies
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    galnames_all = {key: pd.read_csv(file_galsin[key], header=0,\
                                     sep='\t', index_col='galaxyid')\
                    for key in file_galsin}
    
    hists_single = {}
    edges_single = {}
    
    for mbin in galids_per_bin:
        galids = galids_per_bin[mbin]
        for galid in galids:
            if rbinu == 'R200c':
                Runit = galdata_all.at[galid, 'R200c_cMpc'] * c.cm_per_mpc * cosmopars['a']
            else:
                Runit = c.cm_per_mpc * 1e-3 #pkpc
            
            for profq in tgrpns:
                filen = (galnames_all[profq].at[galid, 'filename'])
                with h5py.File(filen, 'r') as fi:
                    grpn = tgrpns[profq]
                    grp_t = fi[grpn]
                        
                    hist = np.array(grp_t['histogram'])
                    if bool(grp_t['histogram'].attrs['log']):
                        hist = 10**hist
                        
                    edges = {}
                    axes = {}
                    
                    for axn in [profq, 'r3d']:
                       edges[axn] = np.array(grp_t[axns[axn] + '/bins'])
                       if axn == 'r3d':
                           edges[axn] *= (1./ Runit)
                       if not bool(grp_t[axns[axn]].attrs['log']):
                           edges[axn] = np.log10(edges[axn])
                       axes[axn] = grp_t[axns[axn]].attrs['histogram axis']          
            
                    if combmethod == 'addnormed-R200c':
                        if rbinu != 'R200c':
                            raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                        _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                        if len(_i) != 1:
                            raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                        _i = _i[0]
                        _a = list(range(len(hist.shape)))
                        _s = [slice(None, None, None) for dummy in _a]
                        _s[axes['r3d']] = slice(None, _i, None)
                        norm_t = np.sum(hist[tuple(_s)])
                    elif combmethod == 'add':
                        norm_t = 1.
                    hist *= (1. / norm_t)
                    
                    if galid not in hists_single:
                        hists_single[galid] = {}
                    if galid not in edges_single:
                        edges_single[galid] = {}
                    
                    rax = axes['r3d']
                    yax = axes[profq]
                    
                    edges_r = np.copy(edges['r3d'])
                    edges_y = np.copy(edges[profq])
                    
                    hist_t = np.copy(hist)
                    
                    # deal with edge units (r3d is already in R200c units if R200c-stacked)
                    if edges_r[0] == -np.inf: # reset centre bin position
                        edges_r[0] = 2. * edges_r[1] - edges_r[2] 
                    if edges_y[0] == -np.inf: # reset centre bin position
                        edges_y[0] = 2. * edges_y[1] - edges_y[2]
                    if edges_y[-1] == np.inf: # reset centre bin position
                        edges_y[-1] = 2. * edges_y[-2] - edges_y[-3]
                        
                    sax = list(range(len(hist_t.shape)))
                    sax.remove(rax)
                    sax.remove(yax)
                    hist_t = np.sum(hist_t, axis=tuple(sax))
                    if yax < rax:
                        hist_t = hist_t.T
                    #hist_t /= (np.diff(edges_r)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
                    
                    hists_single[galid][profq] = hist_t
                    edges_single[galid][profq] = [edges_r, edges_y]
                    
                    # add in cumulative plot for the weight
                    if profq == 'n':
                        hist_t = np.copy(hist)
                        sax = list(range(len(hist_t.shape)))
                        sax.remove(rax)
                        hist_t = np.sum(hist_t, axis=tuple(sax))
                        hist_t = np.cumsum(hist_t)
                        hists_single[galid]['weight'] = hist_t
                        edges_single[galid]['weight'] = [edges_r[1:]]
            
    # set up plot grid
    panelwidth = 3.
    panelheight = 3.
    toplabelheight = 0.5
    caxwidth = 0.5
    nmassbins = len(hists_main)
    
    fig = plt.figure(figsize=(nmassbins * panelwidth + caxwidth, nprof * panelheight + toplabelheight))
    grid = gsp.GridSpec(nrows=nprof + 1, ncols=nmassbins + 1, hspace=0.0, wspace=0.0, width_ratios=[panelwidth] * nmassbins + [caxwidth], height_ratios=[toplabelheight] + [panelheight] * nprof )
    axes = np.array([[fig.add_subplot(grid[yi + 1, xi]) for xi in range(nmassbins)] for yi in range(nprof)])
    cax  = fig.add_subplot(grid[1:, nmassbins])
    laxes = [fig.add_subplot(grid[0, xi]) for xi in range(nmassbins)]
    
    vmax = np.log10(np.max([np.max([np.max(hists_main[mkey][axn]) for axn in axnl]) for mkey in hists_main]))
    vmin = np.log10(np.min([np.min([np.min(hists_main[mkey][axn]) for axn in axnl]) for mkey in hists_main]))
    vmin = max(vmin, vmax - 7.)
    
    massmins = sorted(list(hists_main.keys()))

    linewidth = 1.
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.5, foreground="black"),\
                     mppe.Stroke(linewidth=linewidth + 1., foreground="w"),\
                     mppe.Normal()]
     
    fig.suptitle(title, fontsize=fontsize + 2)
    
    
    for mi in range(nmassbins):
        ind = np.where(np.array(massbins)[:, 0] == massmins[mi])[0][0]
        text = r'$%.1f \, \endash \, %.1f$'%(massbins[ind][0], massbins[ind][1]) #r'$\log_{10} \,$' + binqn + r': 
        
        ax = laxes[mi]
        ax.text(0.5, 0.1, text, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='center', verticalalignment='bottom')
        ax.axis('off') 
        
    for mi in range(nmassbins):
        for ti in range(nprof):
            # where are we
            ax = axes[ti, mi]
            labelx = ti == nprof - 1
            labely = mi == 0
            
            if ti == 0:
                yq = 'weight'
            else:
                yq = axnl[ti - 1]
            mkey = massmins[mi]
            
            # set up axis
            pu.setticks(ax, top=True, left=True, labelleft=labely,\
                        labelbottom=labelx, fontsize=fontsize)
            if labelx:
                ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            if labely:
                ax.set_ylabel(axlabels[yq], fontsize=fontsize)
            
            # plot stacked histogram
            edges_r = edges_main[mkey][yq][0] 
            if yq != 'weight':
                edges_y = edges_main[mkey][yq][1]
                hist = hists_main[mkey][yq]
                
                img, _1, _2 = pu.add_2dplot(ax, hist, [edges_r, edges_y], toplotaxes=(0, 1),\
                              log=True, usepcolor=True, pixdens=True,\
                              cmap=cmap, vmin=vmin, vmax=vmax, zorder=-2)
                perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles))
                mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                
                for pi in range(len(percentiles)):
                    ax.plot(mid_r, perclines[pi], color='white',\
                            linestyle=linestyles[pi], alpha=1.,\
                            path_effects=patheff_thick, linewidth=linewidth + 1)
                
                mmatch = np.array(list(galids_per_bin.keys()))
                ind = np.where(np.isclose(mkey, mmatch))[0][0]
                galids = galids_per_bin[mmatch[ind]]
                
                for galidi in range(len(galids)):
                    galid = galids[galidi]
                    color = 'C%i'%(galidi % 10)
                    
                    hist = hists_single[galid][yq]
                    edges_r = edges_single[galid][yq][0]
                    edges_y = edges_single[galid][yq][1]
                    
                    perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles))
                    mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                    
                    for pi in range(len(percentiles)):
                        ax.plot(mid_r, perclines[pi], color=color,\
                                linestyle=linestyles[pi], alpha=1.,\
                                path_effects=patheff, linewidth=linewidth,\
                                zorder = -1)
            else:
                hist = hists_main[mkey][yq]
                #if combmethod == 'add':
                #    numgal = len(galids_main[mkey])
                #    hist /= float(numgal)
                    
                ax.plot(edges_r, np.log10(hist), color='black',\
                            linestyle='solid', alpha=1.,\
                            path_effects=None, linewidth=linewidth + 1.5)
                
                mmatch = np.array(list(galids_per_bin.keys()))
                ind = np.where(np.isclose(mkey, mmatch))[0][0]
                galids = galids_per_bin[mmatch[ind]]
                
                for galidi in range(len(galids)):
                    galid = galids[galidi]
                    color = 'C%i'%(galidi % 10)
                    
                    hist = hists_single[galid][yq]
                    edges_r = edges_single[galid][yq][0]
                    
                    ax.plot(edges_r, np.log10(hist), color=color,\
                            linestyle='solid', alpha=1.,\
                            path_effects=None, linewidth=linewidth + 0.5,\
                            zorder=-1)
    # color bar 
    pu.add_colorbar(cax, img=img, vmin=vmin, vmax=vmax, cmap=cmap,\
                    clabel=clabel, fontsize=fontsize, orientation='vertical',\
                    extend='min')
    cax.set_aspect(10.)
    
    # sync y limits on plots
    for yi in range(nprof):
        ylims = np.array([axes[yi, mi].get_ylim() for mi in range(nmassbins)])
        miny = np.min(ylims[:, 0])
        maxy = np.max(ylims[:, 1])
        # for Z and cumulative
        miny = max(miny, maxy - 10.)
        [[axes[yi, mi].set_ylim(miny, maxy) for mi in range(nmassbins)]]
    for xi in range(nmassbins):
        xlims = np.array([axes[i, xi].get_xlim() for i in range(nprof)])
        minx = np.min(xlims[:, 0])
        maxx = np.max(xlims[:, 1])
        [axes[i, xi].set_xlim(minx, maxx) for i in range(nprof)]
    
    plt.savefig(outname, format='pdf', box_inches='tight')
    

def plot_barchart_Ls(simple=False):
    '''
    simple: total luminosity fractions for different halo masses (True) or 
            fractions broken down by SF/nSF and subhalo membership category
    '''
    outname = mdir + 'luminosity_total_fractions_z0p1{}.pdf'.format('_simple' if simple else '')
    ddir = '/net/luttero/data2/imgs/paper3/lumfracs/'
    print('Numbers in annotations: log10 L density [erg/s/cMpc**3] rest-frame')
    
     # change order because the two two-line names overlap
    lines = ['c5r', 'c6', 'n6r', 'n7', 'o7f', 'o7iy', 'o7ix', 'o7r', 'o8',\
             'fe17', 'fe18', 'fe19', 'fe17-other1', 'ne9r', 'ne10',\
             'mg11r', 'mg12', 'si13r']
    
    # hdf5 group and histogram axis names
    grn_tot = 'StarFormationRate_T4EOS'
    grn_halo = 'M200c_halo_allinR200c_subhalo_category_StarFormationRate_T4EOS'
    axn_sf = 'StarFormationRate_T4EOS'
    axn_hm = 'M200c_halo_allinR200c'
    axn_sh = 'subhalo_category'
    
    sflabels = {0: 'nSF', 1: 'SF'}
    shlabels = {0: 'central', 1: 'subhalo', 2: 'unbound'}
    
    edges_target = np.arange(11., 15.1, 0.5)
    mmax_igm = c.solar_mass
    
    filebase_L = 'particlehist_Luminosity_{line}_L0100N1504_27_test3.6_SmAb_T4EOS.hdf5'
    filename_M = 'particlehist_Mass_L0100N1504_27_test3.6_T4EOS.hdf5'
    
    filenames = {line: ddir + filebase_L.format(line=line) for line in lines}
    filenames.update({'Mass': ddir + filename_M})
    
    labels = nicenames_lines.copy()
    labels.update({'Mass': 'Mass'})
    
    # avoid large whitespace just to fit lines names
    labels['fe17'] = 'Fe XVII\n(17.05 A)'
    labels['fe17-other1'] = 'Fe XVII\n(15.10 A)'

    keys = ['Mass'] + lines 
    
    
    data = {}
    haxes = {}
    hmedges = {}
    for key in keys:
        filen = filenames[key]
        data[key] = {}
        haxes[key] = {}
        with h5py.File(filen, 'r') as fi:
            data[key]['tot'] = fi[grn_tot]['histogram'][:]
            data[key]['halo'] = fi[grn_halo]['histogram'][:]
            haxes[key]['sf'] = fi[grn_halo][axn_sf].attrs['histogram axis']
            haxes[key]['hm'] = fi[grn_halo][axn_hm].attrs['histogram axis']
            haxes[key]['sh'] = fi[grn_halo][axn_sh].attrs['histogram axis']
            hmedges[key] = fi[grn_halo][axn_hm]['bins'][:]
            
            if key == 'Mass':
                cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
            
    if simple:
        figsize = (5.5, 10.)
        height_ratios = [8., 1.]
        ncols = 1
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=2, ncols=1, hspace=0.2, wspace=0.0,\
                            height_ratios=height_ratios)
        ax =  fig.add_subplot(grid[0, 0]) 
        cax = fig.add_subplot(grid[1, 0])
    else:
        figsize = (11., 16.)
        height_ratios = [15., 0.5]
        ncols = 3
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=2, ncols=ncols, hspace=0.2, wspace=0.4,\
                            height_ratios=height_ratios)
        axes =  [fig.add_subplot(grid[0, i]) for i in range(ncols)] 
        cax = fig.add_subplot(grid[1, :])
        
    clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    massedges = np.array(edges_target)
    
    clist = tc.tol_cmap('rainbow_discrete', lut=len(massedges))(np.linspace(0.,  1., len(massedges)))
    c_under = clist[-1]
    clist = clist[:-1]
    c_igm = 'gray'
    _keys = sorted(massedges)
    colors = {_keys[i]: clist[i] for i in range(len(_keys) - 1)}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist)
    #cmap.set_over(clist[-1])
    cmap.set_under(c_under)
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(np.array(massedges[0] - 1.), massedges),\
                                ticks=massedges,\
                                spacing='proportional', extend='min',\
                                orientation='horizontal')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(0.1)
    
    colors.update({'lom': c_under, 'igm': c_igm})
    #print(hmedges.keys())
    
    yc = np.arange((len(keys)  - 1) // ncols + 1)  # the label locations
    width = 0.9  # the width of the bars
    morder = ['igm', 'lom'] + list(edges_target[:-1]) + ['over']
    for ki, key in enumerate(keys):
        if not simple:
            axi = ki % ncols
            ax = axes[axi]
         
        # match halo mass edges:
        edges_in = hmedges[key]
        edges_t = 10**edges_target * c.solar_mass
        
        e_igm = np.where(np.isclose(edges_in, mmax_igm))[0][0]
        s_igm = slice(0, e_igm, None)
        
        e_dct = {edges_target[i]: np.where(np.isclose(edges_in, edges_t[i]))[0][0]\
                 for i in range(len(edges_target))}
        s_dct = {edges_target[i]: slice(e_dct[edges_target[i]],\
                                        e_dct[edges_target[i + 1]], None) \
                 for i in range(len(edges_target) - 1)}
        s_dct['igm'] = s_igm
        s_dct['lom'] = slice(e_igm, e_dct[edges_target[0]], None)
        s_dct['over'] = slice(e_dct[edges_target[-1]], None, None)
        
        # total mass fractions
        if simple: 
            _width = width
            zeropt = ki
        else:
            _width = width / 7.
            zeropt = yc[ki // ncols] - 3.5 * _width
        
        baseslice = [slice(None, None, None)] * len(data[key]['halo'].shape)
        
        total = np.sum(data[key]['tot'])
        
        # total IGM/ halo mass split
        cumul = 0.
        for mk in morder:
            if mk == 'igm':
                slices = list(np.copy(baseslice))
                slices[haxes[key]['hm']] = slice(e_igm, None, None)
                current = total - np.sum(data[key]['halo'][tuple(slices)])
                current /= total
            else:
                slices = list(np.copy(baseslice))
                slices[haxes[key]['hm']] = s_dct[mk]
                current = np.sum(data[key]['halo'][tuple(slices)])
                current /= total
            if mk == 'over':
                if current == 0.:
                    continue
                else:
                    print('Warning: for {line}, a fraction {} is in masses above max'.format(current, line=key))
            ax.barh(zeropt, current, _width, color=colors[mk], left=cumul)
            cumul += current
        # annotate
        if key != 'Mass':
            dens = total / (cosmopars['boxsize'] / cosmopars['h'])**3 
            text = '{:.1f}'.format(np.log10(dens))
            ax.text(0.99, zeropt, text, fontsize=fontsize,\
                    horizontalalignment='right',\
                    verticalalignment='center')
        if not simple:
            for sfi in range(2):
                for shi in range(3):
                    zeropt += _width
                    cumul = 0.
                    for mk in morder:
                        if mk in ['over', 'igm']:
                            continue
                        else:
                            slices = list(np.copy(baseslice))
                            slices[haxes[key]['hm']] = s_dct[mk]
                            slices[haxes[key]['sh']] = shi
                            slices[haxes[key]['sf']] = sfi
                            current = np.sum(data[key]['halo'][tuple(slices)])
                            current /= total
                        
                        ax.barh(zeropt, current, _width, color=colors[mk],\
                                left=cumul)
                        cumul += current
                        
                    slices = list(np.copy(baseslice))
                    slices[haxes[key]['hm']] = slice(e_igm, None, None)
                    slices[haxes[key]['sh']] = shi
                    slices[haxes[key]['sf']] = sfi
                    subtot = np.sum(data[key]['halo'][tuple(slices)])
                    text = '{:.1f}'.format(np.log10(subtot / (cosmopars['boxsize'] / cosmopars['h'])**3 ))
                    text = ', '.join([shlabels[shi], sflabels[sfi]]) + ': ' + text
                    ax.text(0.99, zeropt, text, fontsize=fontsize,\
                            horizontalalignment='right',\
                            verticalalignment='center')
    
    if simple:                
        pu.setticks(ax, fontsize + 1)
        ax.set_xlim(0., 1.)
        ax.minorticks_off()
        ax.set_yticks(yc)
        ax.set_yticklabels([labels[key] for key in keys])
        ax.set_xlabel('fraction of total', fontsize=fontsize)
    else:
        for axi, ax in enumerate(axes):
            pu.setticks(ax, fontsize + 1)
            ax.set_xlim(0., 1.)
            ax.minorticks_off()
            nkeys = len(keys) // ncols
            if nkeys * ncols + axi < len(keys):
                nkeys += 1
            ax.set_yticks(np.arange(nkeys))
            ax.set_yticklabels([labels[keys[ncols * i + axi]] for i in range(nkeys)])
            
            ax.set_xlabel('fraction of total', fontsize=fontsize)
        
        ylims = [ax.get_ylim() for ax in axes]
        y0 = np.min([ylim[0] for ylim in ylims])
        y1 = np.max([ylim[1] for ylim in ylims])
        [ax.set_ylim(y0, y1) for ax in axes]
        
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    

def crosscheck_Luminosities():
    '''
    compare average L200c from L-weighted profiles to Ltot in halo mass bins
    from halo mass histograms
    '''
    
    lines = ['c5r', 'c6', 'n6r', 'n7', 'o7f', 'o7iy', 'o7ix', 'o7r', 'o8',\
             'fe17', 'fe18', 'fe19', 'fe17-other1', 'ne9r', 'ne10',\
             'mg11r', 'mg12', 'si13r']

    lsargs = {'c5r':  {'linestyle': 'solid'},\
              'c6':   {'linestyle': 'dashed'},\
              'n6r':  {'linestyle': 'dotted'},\
              'n7':   {'linestyle': 'dashdot'},\
              'o7r':  {'linestyle': 'solid'},\
              'o7ix': {'dashes': [2, 2, 2, 2]},\
              'o7iy': {'dashes': [6, 2]},\
              'o7f':  {'linestyle': 'dashdot'},\
              'o8':   {'linestyle': 'dotted'},\
              'fe17-other1':  {'linestyle': 'solid'},\
              'fe19':  {'linestyle': 'dashed'},\
              'fe17':  {'linestyle': 'dotted'},\
              'fe18':  {'linestyle': 'dashdot'},\
              'si13r': {'linestyle': 'solid'},\
              'ne9r':  {'dashes': [2, 2, 2, 2]},\
              'ne10':  {'dashes': [6, 2]},\
              'mg11r': {'linestyle': 'dashdot'},\
              'mg12':  {'linestyle': 'dotted'},\
                  }
    
    nhalos = {11.0: 6295,\
              11.5: 2287,\
              12.0: 870,\
              12.5: 323,\
              13.0: 119,\
              13.5: 26,\
              14.0: 9,\
              }
    ## halo histogram
    # hdf5 group and histogram axis names
    grn_halo = 'M200c_halo_allinR200c_subhalo_category_StarFormationRate_T4EOS'
    axn_sf = 'StarFormationRate_T4EOS'
    axn_hm = 'M200c_halo_allinR200c'
    axn_sh = 'subhalo_category'
        
    edges_target = np.arange(11., 15.1, 0.5)
    
    ddir = '/net/luttero/data2/imgs/paper3/lumfracs/'
    filebase_L = 'particlehist_Luminosity_{line}_L0100N1504_27_test3.6_SmAb_T4EOS.hdf5'   
    filenames = {line: ddir + filebase_L.format(line=line) for line in lines}
    
    data = {}
    haxes = {}
    hmedges = {}
    for key in lines:
        filen = filenames[key]
        data[key] = {}
        haxes[key] = {}
        with h5py.File(filen, 'r') as fi:
            data[key]['halo'] = fi[grn_halo]['histogram'][:]
            haxes[key]['sf'] = fi[grn_halo][axn_sf].attrs['histogram axis']
            haxes[key]['hm'] = fi[grn_halo][axn_hm].attrs['histogram axis']
            haxes[key]['sh'] = fi[grn_halo][axn_sh].attrs['histogram axis']
            hmedges[key] = fi[grn_halo][axn_hm]['bins'][:]
            
    morder = list(edges_target[:-1]) 
    Ltot_hist = {}
    for line in lines:
        Ltot_hist[line] = {}
        # match halo mass edges:
        edges_in = hmedges[key]
        edges_t = 10**edges_target * c.solar_mass
        
        e_dct = {edges_target[i]: np.where(np.isclose(edges_in, edges_t[i]))[0][0]\
                 for i in range(len(edges_target))}
        s_dct = {edges_target[i]: slice(e_dct[edges_target[i]],\
                                        e_dct[edges_target[i + 1]], None) \
                 for i in range(len(edges_target) - 1)}
        
        # total masses fractions
        baseslice = [slice(None, None, None)] * len(data[key]['halo'].shape)
        # total IGM/ halo mass split
        for mk in morder:
            slices = list(np.copy(baseslice))
            slices[haxes[line]['hm']] = s_dct[mk]
            current = np.sum(data[line]['halo'][tuple(slices)])
            Ltot_hist[line][mk] = current
    
    ## L200c list
    filename = ol.pdir + 'luminosities_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_SmAb.hdf5'%(str(0.), str(1.))
    with h5py.File(filename, 'r') as fi:
        galids_l = fi['galaxyids'][:]
        lines_l = [line.decode() for line in fi.attrs['lines']]
        lums = fi['luminosities'][:]
            
    file_galdata = '/net/luttero/data2/imgs/CGM/3dprof/halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    masses = np.array(galdata_all['M200c_Msun'][galids_l])
    
    mbins = np.array(list(np.arange(11., 14.1, 0.5)) + [15.])
    lums = np.sum(lums, axis=2)
    bininds = np.digitize(np.log10(masses), mbins)
    #for i, m in enumerate(mbins[:-1]):
    #    print('m {}: {num}'.format(m, num=np.sum(bininds == i + 1)))
    bincen = mbins[:-1] + 0.5 * np.diff(mbins) 
    bincen[-1] = 14.25
    
    L200c_av = {}
    for li, line in enumerate(lines_l):
        avvals = {mbins[i]: np.average(lums[bininds == i + 1, li])\
                  for i in range(len(mbins) - 1)}
        L200c_av[line] = avvals
    
    #return Ltot_hist, L200c_av, nhalos
    ncols = 2
    fig = plt.figure(figsize=(5.5, 8.))
    grid = gsp.GridSpec(nrows=2, ncols=ncols,  hspace=0.25, wspace=0.0, \
                        height_ratios=[5., 3.], left=0.17)
    axes = [fig.add_subplot(grid[0, i]) for i in range(ncols)]
    lax = fig.add_subplot(grid[1, :]) 
    
    xlabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    ylabel = '$\\log_{10} \\langle \\mathrm{L} \\rangle_{\\mathrm{FoF}} \\,/\\, \\langle \\mathrm{L}_{\\mathrm{200c}} \\rangle_{\\mathrm{sample}}$'
    axes[0].set_xlabel(xlabel, fontsize=fontsize)
    axes[1].set_xlabel(xlabel, fontsize=fontsize)
    axes[0].set_ylabel(ylabel, fontsize=fontsize)
    
    nlperpn = (len(lines) - 1) // ncols + 1
    for axi, ax in enumerate(axes):
        imin = axi * nlperpn
        imax = imin + nlperpn
        _lines = lines[imin:imax]
        
        labelleft = axi % ncols == 0
        pu.setticks(ax, fontsize, labelleft=labelleft)
        
        for line in _lines:
            xv = bincen
            yv = [np.log10(Ltot_hist[line][m] / L200c_av[line][m]\
                          / float(nhalos[m])) for m in mbins[:-1]]
            #xv = [np.log10(Ltot_hist[line][m]) for m in mbins[:-1]]
            #yv = [np.log10( L200c_av[line][m] * float(nhalos[m])) for m in mbins[:-1]]
            ax.plot(xv, yv, label=nicenames_lines[line], color=linecolors[line],\
                    **lsargs[line])
        #ax.plot([xv[0], xv[1]],[xv[0], xv[1]], color='black')
        
    handles = []
    labels = []
    for ax in axes:
        _handles, _labels =  ax.get_legend_handles_labels()
        handles += _handles
        labels += _labels
    lax.axis('off')
    lax.legend(handles, labels, fontsize=fontsize, ncol=2)
        
    xlims = [ax.get_xlim() for ax in axes]
    x0 = np.min([xl[0] for xl in xlims])
    x1 = np.max([xl[1] for xl in xlims])
    [ax.set_xlim(x0, x1) for ax in axes]
    
    ylims = [ax.get_ylim() for ax in axes]
    y0 = np.min([yl[0] for yl in ylims])
    y1 = np.max([yl[1] for yl in ylims])
    [ax.set_ylim(y0, y1) for ax in axes]
    
    plt.savefig(mdir + 'linem_halomass_consistency_check.pdf', format='pdf', box_inches='tight')
    

def plot3Dprof_v1(weightset=1, M200cslice=(None, None, None)):
    '''
    plot: cumulative profile of weight, rho profile, T profile, Z profile
    rows show different weights
    
    input:
    ------
    weightset: int, which set of weight to plot. Always: M/V weighted and 
               some ions from the same element
    '''
    
    inclSF = True #False is not implemented in the histogram extraction
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = outdir + 'radprof1_L0100N1504_27_Mh0p5dex_1000_{}_set{ws}.pdf'.format('wSF' if inclSF else 'nSF',\
                                                                     ws=weightset)
    # for halo mass selections
    massslice = slice(None, None, 2)
    minrshow = np.log10(0.1) # log10 R200c
    
    weightsets = {1: ['c5r', 'c6'],\
                  2: ['n6r', 'n7'],\
                  3: ['ne9r', 'ne10'],\
                  4: ['mg11r', 'mg12'],\
                  5: ['si13r'],\
                  6: ['o7r', 'o8'],\
                  7: ['o7ix', 'o7iy', 'o7f'],\
                  8: ['fe17', 'fe17-other1', 'fe18', 'fe19'],\
                  }
    
    ws = weightsets[weightset]
    weights = ['Mass', 'Volume'] + ws
    axweights = {0: ['Mass', 'Volume']}
    axweights.update({i + 1: [ws[i]] for i in range(len(ws))})
    elt = string.capwords(ol.elements_ion[ws[0]])
    Zsol = ol.solar_abunds_ea[ol.elements_ion[ws[0]]]
    print('Using {elt} metallicity, solar value {Zsol}'.format(elt=elt,\
          Zsol=Zsol))
    
    fontsize = 12
    percentile = 0.50
    rbinu = 'R200c'
    combmethods = ['add', 'addnormed-R200c']
    print('Showing percentile ' + str(percentile))
    alphas = {'add': 0.4,\
              'addnormed-R200c': 1.,\
              }
    linestyles = {weight: 'solid' for weight in weights}
    linestyles.update({'Volume': 'dashed'})
    title = 'medians from stacked histograms'

    # snapshot 27
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'n': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \; [\mathrm{Z}_{\odot}]$',\
                'weight': r'$\log_{10} \, \mathrm{\Sigma}(< r) \,/\, \mathrm{\Sigma}(< \mathrm{R}_{\mathrm{200c}})$',\
                }
    axnl = {0: 'weight', 1: 'n', 2: 'T', 3: 'Z'}
    
    filebase_line = 'particlehist_Luminosity_{line}_L0100N1504_27_test3.6_SmAb_T4EOS_galcomb.hdf5'
    filebase_basic = 'particlehist_{qt}_L0100N1504_27_test3.6_T4EOS_galcomb.hdf5'
    filenames = {weight: ol.ndir + filebase_line.format(line=weight)\
                 if weight in ol.elements_ion.keys() else\
                 ol.ndir + filebase_basic.format(\
                   qt='propvol' if weight == 'Volume' else weight)\
                 for weight in weights}
    # read in data: stacked histograms -> process to plottables
    hists = {}
    edges = {}
    for cbm in combmethods:
        hists[cbm] = {}
        edges[cbm] = {}
        for weight in weights:
            hists[cbm][weight], edges[cbm][weight], _ =\
            readin_3dprof_stacked(filenames[weight], Zelt=elt, weight=weight,\
                          combmethod=cbm, rbinu=rbinu,\
                          )
        
    # set up plot grid
    panelwidth = 3.
    panelheight = 3.
    toplabelheight = 0.0
    caxwidth = 0.5
    #nmassbins = len(hists[combmethods[0]][weights[0]])
    nprof = 4 # cumulative, n, T, Z
    
    fig = plt.figure(figsize=(len(axweights) * panelwidth + caxwidth,\
                              nprof * panelheight + toplabelheight))
    grid = gsp.GridSpec(nrows=nprof, ncols=len(axweights) + 1,\
                        hspace=0.0, wspace=0.0,\
                        width_ratios=[panelwidth] * len(axweights) + [caxwidth],\
                        height_ratios=[panelheight] * nprof )
    axes = np.array([[fig.add_subplot(grid[yi, xi])\
                      for xi in range(len(axweights))]\
                      for yi in range(nprof)])
    cax  = fig.add_subplot(grid[:, len(axweights)])
    

    massedges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.])
    massedges.sort()
    cmapname = 'rainbow'    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
    massincl = massedges[massslice]
    massexcl = np.array([ed not in massincl for ed in massedges])
    clist[massexcl] = np.array([1., 1., 1., 1.])
    keys = massedges
    colordct = {keys[i]: clist[i] for i in range(len(keys))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)
    

    linewidth = 1.
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    linewidth_thick = 2.
    patheff_thick = [mppe.Stroke(linewidth=linewidth_thick + 0.5, foreground="black"),\
                     mppe.Stroke(linewidth=linewidth_thick, foreground="w"),\
                     mppe.Normal()]
     
    fig.suptitle(title, fontsize=fontsize + 2)
    
   
    for mi in axweights:
        for ti in range(nprof):
            # where are we
            ax = axes[ti, mi]
            labelx = ti == nprof - 1
            labely = mi == 0
            yq = axnl[ti]
            _weights = axweights[mi]
            
            # set up axis
            pu.setticks(ax, top=True, left=True, labelleft=labely,\
                        labelbottom=labelx, fontsize=fontsize)
            ax.grid(b=True)
            
            if labelx:
                ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            if labely:
                ax.set_ylabel(axlabels[yq], fontsize=fontsize)
            
            # plot stacked histogram
            for weight in _weights:
                for cmb in combmethods:

                    _hists = hists[cmb][weight]
                    _edges = edges[cmb][weight]
                    mkeys = _hists.keys()
                    for mkey in mkeys:
                        keymatch = np.isclose(massincl, mkey)
                        if not np.any(keymatch):
                            continue
                        cmkey = massincl[np.where(keymatch)[0][0]]
                        color = colordct[cmkey]
                        
                        edges_r = _edges[mkey][yq][0] 
                        si = np.where(np.isclose(edges_r, minrshow))[0][0]
                        
                        if yq != 'weight':
                            edges_y = _edges[mkey][yq][1]
                            hist = _hists[mkey][yq]
                            hist = np.append(np.sum(hist[:si, :], axis=0)[np.newaxis, :],\
                                             hist[si:, :], axis=0)
                            if yq == 'Z':
                                edges_y -= np.log10(Zsol)
                            perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1,\
                                                                      percentiles=np.array([percentile]))
                            mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                            mid_r = mid_r[si - 1:]
                            
                            
                            pi = 0
                            ax.plot(mid_r, perclines[pi], color=color,\
                                    linestyle=linestyles[weight], alpha=alphas[cmb],\
                                    path_effects=patheff_thick, linewidth=linewidth_thick)
                            
                        else:
                            if weight == 'Volume': # just takes up space in a cumulative profile
                                continue
                            hist = _hists[mkey][yq][si:]
                            edges_r = edges_r[si:]
                            #if combmethod == 'add':
                            #    numgal = len(galids_main[mkey])
                            #    hist /= float(numgal)
                                
                            ax.plot(edges_r, np.log10(hist), color=color,\
                                        linestyle=linestyles[weight], alpha=alphas[cmb],\
                                        path_effects=patheff_thick, linewidth=linewidth_thick)
                        
                        # add CIE T indicators
                        if weight in line_Tmax and yq == 'T':
                            Tcen = line_Tmax[weight]
                            Tran = line_Trange[weight]
                            ax.axhline(Tcen, color='black', linestyle='solid',\
                                       linewidth=linewidth)
                            ax.axhline(Tran[0], color='black', linestyle='dotted',\
                                       linewidth=linewidth)
                            ax.axhline(Tran[1], color='black', linestyle='dotted',\
                                       linewidth=linewidth)
                        # add Tvir indicator
                        elif weight == 'Mass' and yq == 'T':
                            medm = 10**medianmasses[cmkey] # M200c [Msun]
                            Tv = cu.Tvir(medm, cosmopars=cosmopars, mu=0.59)
                            ax.axhline(np.log10(Tv), color=color,\
                                    linestyle='dotted', linewidth=linewidth,\
                                    path_effects=patheff)
                            
            if ti == 0 and len(_weights) > 1:
                handles = [mlines.Line2D([], [], linestyle=linestyles[weight],\
                                         color='black', alpha=1., linewidth=linewidth_thick,\
                                         label=weight) for weight in _weights]
                labels = [weight[0] for weight in _weights]
                ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(1., 0.),\
                          loc='lower right')
            elif ti == 0:
                ax.text(0.05, 0.95, _weights[0], fontsize=fontsize,\
                        horizontalalignment='left', verticalalignment='top',\
                        transform=ax.transAxes)
            if ti == 0 and mi == 1:
                handles = [mlines.Line2D([], [], linestyle='solid', color='black',\
                                         alpha=alphas[cmb], linewidth=linewidth_thick,\
                                         label=cmb) for cmb in combmethods]
                labels = ['add' if cmb == 'add' else\
                          'norm.' if cmb == 'addnormed-R200c' else\
                          cmb for cmb in combmethods]
                ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(1., 0.),\
                          loc='lower right')
                
    # sync y limits on plots
    for yi in range(nprof):
        if axnl[yi] == 'T':
            y0min = 3.5
            y1max = 8.
        elif axnl[yi] == 'n':
            y0min = -6.5
            y1max = 0.
        elif axnl[yi] == 'Z':
            y0min = -2.5
            y1max = 0.8
        elif axnl[yi] == 'weight':
            y0min = -2.5
            y1max = 1.
        ylims = np.array([axes[yi, mi].get_ylim() for mi in range(len(axweights))])
        miny = max(np.min(ylims[:, 0]), y0min)
        maxy = min(np.max(ylims[:, 1]), y1max)
        # for Z and cumulative
        miny = max(miny, maxy - 10.)
        [[axes[yi, mi].set_ylim(miny, maxy) for mi in range(len(axweights))]]
    for xi in range(len(axweights)):
        xlims = np.array([axes[i, xi].get_xlim() for i in range(nprof)])
        minx = np.min(xlims[:, 0])
        maxx = np.max(xlims[:, 1])
        [axes[i, xi].set_xlim(minx, maxx) for i in range(nprof)]
    
    plt.savefig(outname, format='pdf', box_inches='tight')

def plot_r200Lweighted(weightset=1, M200cslice=slice(None, None, None)):
    '''
    plot: 
    
    input:
    ------
    weightset: int, which set of weight to plot. Always: M/V weighted and 
               some ions from the same element
    '''
    
    inclSF = True #False is not implemented in the histogram extraction
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = outdir + 'totLw_L0100N1504_27_Mh0p5dex_1000_0-1-R200c_{}_set{ws}.pdf'.format('wSF' if inclSF else 'nSF',\
                                                          ws=weightset)
    addedges = (0., 1.)
    # for halo mass selections
    massslice = M200cslice
    #minrshow = np.log10(0.1) # log10 R200c
    
    weightsets = {1: ['c5r', 'c6'],\
                  2: ['n6r', 'n7'],\
                  3: ['ne9r', 'ne10'],\
                  4: ['mg11r', 'mg12'],\
                  5: ['si13r'],\
                  6: ['o7r', 'o8'],\
                  7: ['o7ix', 'o7iy', 'o7f'],\
                  8: ['fe17', 'fe17-other1', 'fe18', 'fe19'],\
                  }
    
    ws = weightsets[weightset]
    weights = ['Mass', 'Volume'] + ws
    axweights = {0: ['Mass', 'Volume']}
    axweights.update({i + 1: [ws[i]] for i in range(len(ws))})
    elt = string.capwords(ol.elements_ion[ws[0]])
    Zsol = ol.solar_abunds_ea[ol.elements_ion[ws[0]]]
    print('Using {elt} metallicity, solar value {Zsol}'.format(elt=elt,\
          Zsol=Zsol))
    
    fontsize = 12
    percentile = 0.50
    rbinu = 'R200c'
    combmethods = ['add', 'addnormed-R200c']
    print('Showing percentile ' + str(percentile))
    alphas = {'add': 0.4,\
              'addnormed-R200c': 1.,\
              }
    linestyles = {weight: 'solid' for weight in weights}
    linestyles.update({'Volume': 'dashed'})
    title = 'weighted medians from stacked histograms'

    # snapshot 27
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'n': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \; [\mathrm{Z}_{\odot}]$',\
                'weight': r'$\log_{10} \, \mathrm{\Sigma}(< r) \,/\, \mathrm{\Sigma}(< \mathrm{R}_{\mathrm{200c}})$',\
                }
    axnl = {0: 'n', 1: 'T', 2: 'Z'}
    
    filebase_line = 'particlehist_Luminosity_{line}_L0100N1504_27_test3.6_SmAb_T4EOS_galcomb.hdf5'
    filebase_basic = 'particlehist_{qt}_L0100N1504_27_test3.6_T4EOS_galcomb.hdf5'
    filenames = {weight: ol.ndir + filebase_line.format(line=weight)\
                 if weight in ol.elements_ion.keys() else\
                 ol.ndir + filebase_basic.format(\
                   qt='propvol' if weight == 'Volume' else weight)\
                 for weight in weights}
    # read in data: stacked histograms -> process to plottables
    hists = {}
    edges = {}
    for cbm in combmethods:
        hists[cbm] = {}
        edges[cbm] = {}
        for weight in weights:
            hists[cbm][weight], edges[cbm][weight], _ =\
            readin_3dprof_stacked(filenames[weight], Zelt=elt, weight=weight,\
                          combmethod=cbm, rbinu=rbinu,\
                          )
        
    # set up plot grid
    panelwidth = 3.
    panelheight = 3.
    toplabelheight = 0.0
    #nmassbins = len(hists[combmethods[0]][weights[0]])
    nprof = 3
    
    fig = plt.figure(figsize=(len(axweights) * panelwidth,\
                              nprof * panelheight + toplabelheight))
    grid = gsp.GridSpec(nrows=nprof, ncols=len(axweights),\
                        hspace=0.0, wspace=0.0,\
                        width_ratios=[panelwidth] * len(axweights),\
                        height_ratios=[panelheight] * nprof )
    axes = np.array([[fig.add_subplot(grid[yi, xi])\
                      for xi in range(len(axweights))]\
                      for yi in range(nprof)])
    

    massedges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.])
    massedges.sort()
    massincl = massedges[massslice]
    #massexcl = np.array([ed not in massincl for ed in massedges])

    linewidth = 1.
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    linewidth_thick = 2.
    patheff_thick = [mppe.Stroke(linewidth=linewidth_thick + 0.5, foreground="black"),\
                     mppe.Stroke(linewidth=linewidth_thick, foreground="w"),\
                     mppe.Normal()]
     
    fig.suptitle(title, fontsize=fontsize + 2)
    
   
    for mi in axweights:
        for ti in range(nprof):
            # where are we
            ax = axes[ti, mi]
            labelx = ti == nprof - 1
            labely = mi == 0
            yq = axnl[ti]
            _weights = axweights[mi]
            
            # set up axis
            pu.setticks(ax, top=True, left=True, labelleft=labely,\
                        labelbottom=labelx, fontsize=fontsize)
            ax.grid(b=True)            
            if labelx:
                ax.set_xlabel(r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} [\mathrm{M}_{\odot}$', fontsize=fontsize)
            if labely:
                ax.set_ylabel(axlabels[yq], fontsize=fontsize)
            
            # plot stacked histogram
            for weight in _weights:
                for cmb in combmethods:

                    _hists = hists[cmb][weight]
                    _edges = edges[cmb][weight]
                    mkeys = _hists.keys()
                    
                    xpoints = []
                    ypoints = []
                    for mkey in mkeys:
                        keymatch = np.isclose(massincl, mkey)
                        if not np.any(keymatch):
                            continue
                        cmkey = massincl[np.where(keymatch)[0][0]]
                        xpoints.append(cmkey + 0.25)
                        
                        edges_r = _edges[mkey][yq][0] 
                        remin = np.where(np.isclose(edges_r, np.log10(addedges[0])))[0][0]\
                                if addedges[0] > 0. else 0
                        remax = np.where(np.isclose(edges_r, np.log10(addedges[1])))[0][0]
                        

                        edges_y = _edges[mkey][yq][1]
                        hist = _hists[mkey][yq]
                        hist = np.sum(hist[remin:remax, :], axis=0)
                        if yq == 'Z':
                            edges_y -= np.log10(Zsol)
                        
                        percv = pu.percentiles_from_histogram(hist[np.newaxis, :], edges_y, axis=1,\
                                                              percentiles=np.array([percentile]))
                        ypoints.append(percv[0][0])
                    
                    xpoints = np.array(xpoints)
                    ypoints = np.array(ypoints)
                    xs = np.argsort(xpoints)
                    ypoints = ypoints[xs]
                    xpoints = xpoints[xs]
                        
                    ax.plot(xpoints, ypoints, color='black',\
                            linestyle=linestyles[weight], alpha=alphas[cmb],\
                            path_effects=patheff_thick, linewidth=linewidth_thick)
                    
                    # add CIE T indicators
                    if weight in line_Tmax and yq == 'T':
                        Tcen = line_Tmax[weight]
                        Tran = line_Trange[weight]
                        ax.axhline(Tcen, color='red', linestyle='solid',\
                                   linewidth=linewidth)
                        ax.axhline(Tran[0], color='red', linestyle='dotted',\
                                   linewidth=linewidth)
                        ax.axhline(Tran[1], color='red', linestyle='dotted',\
                                   linewidth=linewidth)
                    # add Tvir indicator
                    elif weight == 'Mass' and yq == 'T':
                        xvals = 10**xpoints # M200c [Msun]
                        Tv = cu.Tvir(xvals, cosmopars=cosmopars, mu=0.59)
                        ax.plot(xpoints, np.log10(Tv), color='blue',\
                                linestyle='solid', linewidth=linewidth)
                        
            if ti == 0 and len(_weights) > 1:
                handles = [mlines.Line2D([], [], linestyle=linestyles[weight],\
                                         color='black', alpha=1., linewidth=linewidth_thick,\
                                         label=weight) for weight in _weights]
                labels = [weight[0] for weight in _weights]
                ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(0., 1.),\
                          loc='upper left')
            elif ti == 0:
                ax.text(0.05, 0.95, _weights[0], fontsize=fontsize,\
                        horizontalalignment='left', verticalalignment='top',\
                        transform=ax.transAxes)
            if ti == 0 and mi == 1:
                handles = [mlines.Line2D([], [], linestyle='solid', color='black',\
                                         alpha=alphas[cmb], linewidth=linewidth_thick,\
                                         label=cmb) for cmb in combmethods]
                labels = ['add' if cmb == 'add' else\
                          'norm.' if cmb == 'addnormed-R200c' else\
                          cmb for cmb in combmethods]
                ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(1., 0.),\
                          loc='lower right')
                
    # sync y limits on plots
    for yi in range(nprof):
        if axnl[yi] == 'T':
            y0min = 3.5
            y1max = 8.
        elif axnl[yi] == 'n':
            y0min = -6.5
            y1max = 0.
        elif axnl[yi] == 'Z':
            y0min = -2.5
            y1max = 0.8
        elif axnl[yi] == 'weight':
            y0min = -2.5
            y1max = 1.
        ylims = np.array([axes[yi, mi].get_ylim() for mi in range(len(axweights))])
        miny = max(np.min(ylims[:, 0]), y0min)
        maxy = min(np.max(ylims[:, 1]), y1max)
        # for Z and cumulative
        miny = max(miny, maxy - 10.)
        [[axes[yi, mi].set_ylim(miny, maxy) for mi in range(len(axweights))]]
    for xi in range(len(axweights)):
        xlims = np.array([axes[i, xi].get_xlim() for i in range(nprof)])
        minx = np.min(xlims[:, 0])
        maxx = np.max(xlims[:, 1])
        [axes[i, xi].set_xlim(minx, maxx) for i in range(nprof)]
    
    plt.savefig(outname, format='pdf', box_inches='tight')