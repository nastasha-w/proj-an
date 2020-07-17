#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:53:54 2020

@author: Nastasha
"""

import numpy as np
import h5py

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
              'o8':    'lightsalmon',\
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



mass_edges_standard = (11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0)
fontsize = 12
mdir = ol.mdir + '/data2/imgs/paper3/'

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


def plotstamps(filebase, halocat, outname=None, \
               groups='stamp0', minhalomass=11.):
    '''
    plot the stamps stored in files filebase, overplotting the halos from 
    halocat
    
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
                  'halomasscomp_{}.pdf'.format(measure)
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
    

def plot_radprof2(measure='mean', mmin=10.5):
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
        outname = mdir + 'radprof2d_10pkpc-annuli_L0100N1504_27_test3.5_SmAb_C2Sm_6.25slice_noEOS_to-2R200c_1000_centrals_' +\
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
        handles1 = [mlines.Line2D([], [], **kwargs_y_stack[ytag],\
                                  label=legtags[ytag] + ' (stack)')\
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
        em, logTK, lognHcm3 = cu.findiontables(line, z)
        vals[line] = em
        nHs[line] = lognHcm3
        Ts[line] = logTK
        indmaxfrac = np.argmax(em[-1, :])
        maxem = em[-1, indmaxfrac]
        Tmax = logTK[indmaxfrac]
        Tmaxs[line] = Tmax
    
        xs = pu.find_intercepts(em[-1, :], logTK, fracv * maxem)
        msg = 'Line {line} has maximum emissivity (solar abunds) {maxv:3f}, at log T[K] = {T:.1f}, max range is {rng}'
        print(msg.format(line=line, maxv=maxem, T=Tmax, rng=str(xs)))
        maxvals[line] = maxem
    
    numpanels = 4
    fig, axes = plt.subplots(ncols=1, nrows=numpanels, figsize=(5.5, 10.),\
                             gridspec_kw={'hspace': 0.})
    xlim = (-8., -1.5)
    ylim = (3.4, 7.65)
    [ax.set_xlim(*xlim) for ax in axes]
    [ax.set_ylim(*ylim) for ax in axes]

    axions = {0: ['c5r', 'c6', 'n6r', 'n7'],\
              1: ['o7r', 'o7ix', 'o7iy', 'o7f', 'o8'],\
              2: ['ne9r', 'ne10', 'mg11r', 'mg12', 'si13r'],\
              3: ['fe17-other1', 'fe19', 'fe17', 'fe18']}
    axions = {key: sorted(axions[key], key=ol.line_eng_ion.get) for key in axions}
    
    xlabel = r'$\log_{10} \, \mathrm{n}_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$'
    ylabel = r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$'
    [ax.set_ylabel(ylabel, fontsize=fontsize) for ax in axes]
    axes[numpanels - 1].set_xlabel(xlabel, fontsize=fontsize)
    for axi, ax in enumerate(axes):
        pu.setticks(ax, fontsize=fontsize, right=False,\
                    labelbottom=(axi == numpanels - 1))
        
        ax.axvline(logrhob + np.log10(rho_to_nh), 0., 0.85, color='gray',\
                   linestyle='dashed', linewidth=1.5)
     
        for line in axions[axi]:
            ax.contourf(nHs[line], Ts[line], vals[line].T, colors=linecolors[line],\
                        alpha=0.1, linewidths=[3.],\
                        levels=[0.1 * maxvals[line], maxvals[line]])
            ax.contour(nHs[line], Ts[line], vals[line].T, colors=linecolors[line],\
                       linewidths=[2.], levels=[0.1 * maxvals[line]],\
                       linestyles=['solid'])
        for line in lines:
            ax.axhline(Tmaxs[line], 0.95, 1., color=linecolors[line], linewidth=3.)
            
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
        ylim = ax.get_ylim()
        axy2.set_ylim(*ylim)
        mhalos = np.arange(9.0, 15.1, 0.5)
        Tvals = np.log10(cu.T200c_hot(10**mhalos, cosmopars))
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_yticks(Tvals)
        axy2.set_yticklabels(Tlabels)
        pu.setticks(axy2, fontsize=fontsize, left=False, right=True, labelleft=False, labelright=True)
        axy2.minorticks_off()
        axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$', fontsize=fontsize)
    
        handles = [mlines.Line2D([], [], label=nicenames_lines[line],\
                                 color=linecolors[line]) for line in axions[axi]]
        ax.legend(handles=handles, fontsize=fontsize, ncol=3,\
                  bbox_to_anchor=(0.0, 1.0), loc='upper left', frameon=False)

    plt.savefig(outname, format='pdf', bbox_inches='tight')
    