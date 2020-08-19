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

def getoutline(linewidth):
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"),\
               mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"),\
               mppe.Normal()]
    return patheff

mass_edges_standard = (11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0)
fontsize = 12
mdir = '/data2/imgs/paper3/'

def add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
             orientation='vertical', clabel=None, fontsize=fontsize, aspect=10.):
    '''
    returns color bar object, color dictionary (keys: lower mass edges)
    '''
    massedges = np.array(massedges)
    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
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


def plotstamps(filebase, halocat,\
               outname='xraylineem_stamps_boxcorner_L0100N1504_27_test3p5_SmAb_C2Sm_32000pix_6p25slice_zcen3p125_z-projection_noEOS.pdf', \
               groups='stamp0', minhalomass=11.):
    '''
    plot the stamps stored in files filebase, overplotting the halos from 
    halocat
    'emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5', 
    'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    
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
    
    marklength = 10. #cMpc
    vmin = -12. # log10 photons / cm2 / s / sr 
    vmax = 1.0
    scaler200 = 2. # show radii at this times R200c
    
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
    cbar, colordct = add_cbar_mass(cax2, cmapname='rainbow', massedges=mass_edges_standard,\
             orientation='horizontal', clabel=clabel_hmass, fontsize=fontsize, aspect=0.1)
    print('Max value in maps: {}'.format(max([np.max(maps[line]) for line in _lines])))
    
    cmap_img = cm.get_cmap('viridis')
    cmap_img.set_under(cmap_img(0.))
    
    for li in range(len(_lines)):
        ax = axes[li]
        line = _lines[li]
        
        labelbottom = li > len(_lines) - ncols - 1
        labeltop = li < ncols 
        labelleft = li % ncols == 0
        labelright = li % ncols == ncols - 1
        ax.tick_params(labelsize=fontsize - 1,  direction='in',\
                       labelbottom=labelbottom, labeltop=labeltop,\
                       labelleft=labelleft, labelright=labelright,\
                       top=labeltop, left=labelleft, bottom=labelbottom,\
                       right=labelright)
        lbase = '{ax} [cMpc]'
        axis1 = paxes[line][0]
        axis2 = paxes[line][1]
        axis3 = paxes[line][2]
        if labelbottom:
            xl = lbase.format(ax=['X', 'Y', 'Z'][axis1])
            ax.set_xlabel(xl, fontsize=fontsize)
        if labelleft:
            yl = lbase.format(ax=['X', 'Y', 'Z'][axis2])
            ax.set_ylabel(yl, fontsize=fontsize)
            
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
    
        patheff = [mppe.Stroke(linewidth=1.2, foreground="b"),\
                       mppe.Stroke(linewidth=0.7, foreground="w"),\
                       mppe.Normal()] 
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=colors, facecolor='none', linewidth=0.7,\
                       path_effects=patheff)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.add_collection(collection)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        
        patheff_text = [mppe.Stroke(linewidth=0.6, foreground="b"),\
                        mppe.Stroke(linewidth=0.4, foreground="w"),\
                        mppe.Normal()]        
        ltext = nicenames_lines[line]
        ax.text(0.95, 0.95, ltext, fontsize=fontsize, path_effects=patheff_text,\
                horizontalalignment='right', verticalalignment='top',\
                transform=ax.transAxes, color='white')
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
            
            patheff = [mppe.Stroke(linewidth=2.5, foreground="b"),\
                       mppe.Stroke(linewidth=2.5, foreground="w"),\
                       mppe.Normal()] 
            ax.plot([xs, xs + marklength], [ypos, ypos], color='white',\
                    path_effects=patheff, linewidth=2)
            ax.text(xcen, ypos + 0.01 * yr, mtext, fontsize=fontsize,\
                    path_effects=patheff_text, horizontalalignment='center',\
                    verticalalignment='bottom', color='white')
            
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
                    
                
def plot_radprof1(measure='mean', mmin=10.5):
    '''
    plot mean or median radial profiles for each line and halo mass bin
    panels for different lines
    
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
    
    rfilebase = ol.pdir + 'radprof/' + 'radprof_stamps_emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5'
    xlabel = '$\\mathrm{r}_{\perp} \\; [\\mathrm{pkpc}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{photons}\\,\\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
    
    if measure == 'mean':
        ys = [('mean',)]
    else:
        ys = [('perc', 50.)]
    outname = mdir + 'radprof2d_10pkpc-annuli_L0100N1504_27_test3.5_SmAb_C2Sm_6.25slice_noEOS_to-2R200c_1000_centrals_' +\
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
    
    
def plot_emcurves(z=0.1):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
      
    outname = mdir + 'emcurves_z{}_HM01_ionizedmu.pdf'.format(str(z).replace('.', 'p'))
    
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
    
    fig, (ax, lax) = plt.subplots(ncols=1, nrows=2, figsize=(11., 5.),\
                             gridspec_kw={'hspace': 0.25,\
                                          'height_ratios': [4, 1]})
    xlim = (5.5, 7.8)
    ylim = (-29., -23.)
    ax.set_xlim(*xlim) 
    ax.set_ylim(*ylim)
    lax.axis('off')
    pu.setticks(ax, fontsize=fontsize, top=False, labeltop=False)

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
    
    ylabel = '$\log_{10} \\, \\Lambda \,/\, \\mathrm{n}_{\\mathrm{H}}^{2} \\; [\\mathrm{erg} \\, \\mathrm{cm}^{3}]$'
    xlabel = r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$'
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    
    indvals = [] #[-5.]
    lsargs2 = [\
               #{'linewidth': 1.5, 'alpha': 1.},\
               #{'linewidth': 1.0, 'alpha': 0.5},\
               {'linewidth': 2.5,  'alpha': 1.0},\
               #{'linewidth': 3.,  'alpha': 0.5},\
               ]
    offsets = {line: 0.05 *  (li - 0.5 *len(lines)) / len(lines) \
               for li, line in enumerate(lines)}
    for line in lines:
        # nH values
        for iv, nH in enumerate(indvals):
            nhi = np.where(np.isclose(nHs[line], nH))[0][0]
            emvals = vals[line][:, nhi]
            kwargs = lsargs[line].copy()
            kwargs.update(lsargs2[iv])
            pe = getoutline(kwargs['linewidth'])
            ax.plot(Ts[line], emvals, color=linecolors[line], path_effects=pe,\
                    **kwargs)
        # CIE
        emvals = vals[line][:, -1]
        kwargs = lsargs[line].copy()
        kwargs.update(lsargs2[-1])
        pe = getoutline(kwargs['linewidth'])
        
        ax.plot(Ts[line], emvals, color=linecolors[line],\
                path_effects=pe, **kwargs)
        
        ax.axvline(Tmaxs[line] + offsets[line], 0.92, 1., color=linecolors[line],\
                   linewidth=3., **lsargs[line])

    axy2 = ax.twiny()
    axy2.set_xlim(*xlim)
    mhalos = np.arange(11.5, 15.1, 0.5)
    Tvals = np.log10(cu.Tvir_hot(10**mhalos * c.solar_mass,\
                                 cosmopars=cosmopars))
    Tlabels = ['%.1f'%mh for mh in mhalos]
    axy2.set_xticks(Tvals)
    axy2.set_xticklabels(Tlabels)
    pu.setticks(axy2, fontsize=fontsize, left=False, right=False,\
                top=True, bottom=False,\
                labelleft=False, labelright=False,\
                labeltop=True, labelbottom=False)
    axy2.minorticks_off()
    axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$',\
                    fontsize=fontsize)
    pe = getoutline(2.)
    handles = [mlines.Line2D([], [], label=nicenames_lines[line],\
                             color=linecolors[line],\
                             linewidth=2., path_effects=pe, **lsargs[line])\
               for line in lines]
    handles2 = [mlines.Line2D([], [], label='$\\mathrm{{n}}_{{\\mathrm{{H}}}} = {:.0f}$'.format(nH),\
                             color='black', **lsargs2[iv])\
               for iv, nH in enumerate(indvals)]
    handles3 = [mlines.Line2D([], [], label='CIE',\
                             color='black', **lsargs2[-1])]
    lax.legend(handles=handles + handles2 + handles3, fontsize=fontsize, ncol=5,\
              bbox_to_anchor=(0.5, 1.0), loc='upper center', frameon=True)

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
        mbins = np.arange(10., 14.6, 0.1)
    
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
        
        minymin = None
    else:
        lums = np.sum(lums, axis=2)
        lums = np.log10(lums) - np.log10(1. + cosmopars['z'])
        ylabel = '$\\mathrm{L}_{\\mathrm{obs}} \\; [\\mathrm{erg} \\,\\mathrm{cm}^{-2}\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
        if plottype in ['all', 'lines']:
            minymin = 28.
        else:
            minymin = 0.
    if plottype == 'SFfrac':
        lums = lums[:, :, 1] / np.sum(lums, axis=2)
        ylabel = '$\\mathrm{L}_{\\mathrm{SF}} \\, / \\, \\mathrm{L}_{\\mathrm{tot}}$'
        
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
        color = 'black'
        percv = [2., 10., 50., 90., 98.]
        
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
            
            for si in range(len(percv) // 2):
                print(percs.shape)
                print(len(percv))
                ax.fill_between(bincen[xmininds], percs[si][xmininds],\
                                percs[len(percv) - 1 - si][xmininds],\
                                color=color, alpha=alpha)
            if len(percv) % 2 == 1:
                ax.plot(bincen, percs[len(percv) // 2], color=color)
            alpha_ol = alpha**((len(percv) - 1) // 2)
            ax.scatter(outliers[0], outliers[1], color=color, alpha=alpha_ol)
        # legend
        handles = [mlines.Line2D([], [], color=color, label='{:.0f}%%'.format(percv[len(percv) // 2]))]
        handles += [mpatch.Patch(facecolor=color, alpha=alpha**i,\
                                 label='{:.0f}%%'.format(percv[len(percv) - 1 - i] - percv[i]))\
                    for i in range((len(percs) - 1) // 2)]
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
    
    
def plot3Dprof_overview(weighttype='Mass'):
    '''
    plot: cumulative profile of weight, [ion number density profile], 
          rho profile, T profile
    rows show different halo mass ranges
    '''
    inclSF = True #False is not implemented in the histogram extraction
    outdir = '/net/luttero/data2/imgs/paper3/3dprof/'
    outname = outdir + 'overview_radprof_L0100N1504_27_Mh0p5dex_1000_%s_%s.pdf'%(weighttype, 'wSF' if inclSF else 'nSF')
    
    print('Using parent element metallicity, otherwise, oxygen')
    
    fontsize = 12
    cmap = pu.truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    cmap.set_under(cmap(0.))
    percentiles = [0.1, 0.50, 0.9]
    print('Showing percentiles ' + str(percentiles))
    linestyles = ['dashed', 'solid', 'dashed']
    
    rbinu = 'R200c'
    combmethod = 'addnormed-R200c'

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
        line = '-'.join(weighttype.split('-')[1:])
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
                'Z': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$',\
                'weight': r'$\log_{10} \, \mathrm{%s}(< r) \,/\, \mathrm{%s}(< \mathrm{R}_{\mathrm{200c}})$'%(wnshort, wnshort)
                }
    clabel = r'$\log_{10} \, \left\langle %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}}) \right\rangle \, / \,$'%(wnshort, wnshort) + 'bin size'
    
    if weighttype in ol.elements_ion.keys():
        filename = ol.ndir + 'particlehist_Nion_%s_L0100N1504_27_test3.4_PtAb_T4EOS_galcomb.hdf5'%(line)
        nprof = 4
        title = r'$\mathrm{L}(\mathrm{%s})$ and $\mathrm{L}(\mathrm{%s})$-weighted profiles'%(wname, wname)
        tgrpns = {'T': '3Dradius_Temperature_T4EOS_StarFormationRate_T4EOS',\
                  'n': '3Dradius_Niondens_hydrogen_SmAb_T4EOS_StarFormationRate_T4EOS',\
                  'Z': '',\
                  }
        axns  = {'r3d':  '3Dradius',\
                 'T':    'Temperature_T4EOS',\
                 'n':    'Niondens_hydrogen_SmAb_T4EOS',\
                 'Z':    'SmoothedElementAbundance-{elt}_T4EOS'.format(string.capwords(ol.elements_ion[line])),\
                }
        axnl = ['n', 'T', 'Z']
    else:
        if weighttype == 'Volume':
            filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.4_T4EOS_galcomb.hdf5'%('propvol')
        else:
            filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.4_T4EOS_galcomb.hdf5'%(weighttype)
        nprof = 4
        title = r'%s and %s-weighted profiles'%(weighttype, weighttype)
        tgrpns = {'T': '3Dradius_Temperature_T4EOS_StarFormationRate_T4EOS',\
                  'n': '3Dradius_Niondens_hydrogen_SmAb_T4EOS_StarFormationRate_T4EOS',\
                  'Z': '',\
                  }
        axns = {'r3d':  '3Dradius',\
                'T':    'Temperature_T4EOS',\
                'n':    'Niondens_hydrogen_SmAb_T4EOS',\
                'Z':    'SmoothedElementAbundance-Oxygen_T4EOS'}
        axnl = ['T', 'n', 'Z']
        
    file_galsin = '/net/luttero/data2/imgs/CGM/3dprof/filenames_L0100N1504_27_Mh0p5dex_1000_%s_%s.txt'%(weighttype, 'nrprof')
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
                for axn in axns:
                   edges[axn] = np.array(grp_t[axns[axn] + '/bins'])
                   if not bool(grp_t[axns[axn]].attrs['log']):
                       edges[axn] = np.log10(edges[axn])
                   axes[axn] = grp_t[axns[axn]].attrs['histogram axis']  
                
                edges_main[mkey] = {}
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
                    _a = range(len(hist.shape))
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
                    
                sax = range(len(hist_t.shape))
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
                    sax = range(len(hist_t.shape))
                    sax.remove(rax)
                    hist_t = np.sum(hist_t, axis=tuple(sax))
                    hist_t = np.cumsum(hist_t)
                    hists_main[mkey]['weight'] = hist_t
                    edges_main[mkey]['weight'] = [edges_r[1:]]
                    
                
                galids_main[mkey] = np.array(grp_t['galaxyids'])
    
    # read in data: individual galaxies
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    galname_all = pd.read_csv(file_galsin, header=0, sep='\t', index_col='galaxyid')
    
    hists_single = {}
    edges_single = {}
    
    for mbin in galids_per_bin:
        galids = galids_per_bin[mbin]
        for galid in galids:
            filen = galname_all.at[galid, 'filename']
            if rbinu == 'R200c':
                Runit = galdata_all.at[galid, 'R200c_cMpc'] * c.cm_per_mpc * cosmopars['a']
            else:
                Runit = c.cm_per_mpc * 1e-3 #pkpc
            
            with h5py.File(filen, 'r') as fi:
                for profq in tgrpns:
                    grpn = tgrpns[profq]
                    grp_t = fi[grpn]
                        
                    hist = np.array(grp_t['histogram'])
                    if bool(grp_t['histogram'].attrs['log']):
                        hist = 10**hist
                        
                    edges = {}
                    axes = {}
                    
                    for axn in axns:
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
                        _a = range(len(hist.shape))
                        _s = [slice(None, None, None) for dummy in _a]
                        _s[axes['r3d']] = slice(None, _i, None)
                        norm_t = np.sum(hist[tuple(_s)])
                    
                    hist *= (1. / norm_t)
                    
                    hists_single[galid] = {}
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
                        
                    sax = range(len(hist_t.shape))
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
                        sax = range(len(hist_t.shape))
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
        # intended for ion number densities
        miny = max(miny, maxy - 17.)
        [[axes[yi, mi].set_ylim(miny, maxy) for mi in range(nmassbins)]]
    for xi in range(nmassbins):
        xlims = np.array([axes[i, xi].get_xlim() for i in range(nprof)])
        minx = np.min(xlims[:, 0])
        maxx = np.max(xlims[:, 1])
        [axes[i, xi].set_xlim(minx, maxx) for i in range(nprof)]
    
    plt.savefig(outname, format='pdf', box_inches='tight')