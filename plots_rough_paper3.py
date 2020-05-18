#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:53:54 2020

@author: Nastasha
"""

import numpy as np
import h5py

import eagle_constants_and_units as c
import cosmo_utils as cu
import make_maps_opts_locs as ol

res_arcsec = {'Athena X-IFU': 5.,\
              'Athena WFI':  3.,\
              'Lynx PSF':    1.,\
              'Lynx HDXI pixel':  0.3,\
              }

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
                              - lowerleftcorner[0]) / (res_in_x * 1000),\
                             (center_out[0] + 0.5 * diameter_out[0] \
                              - lowerleftcorner[0]) / (res_in_x * 1000),\
                             (center_out[1] - 0.5 * diameter_out[1] \
                              - lowerleftcorner[1]) / (res_in_y * 1000),\
                             (center_out[1] + 0.5 * diameter_out[1] \
                              - lowerleftcorner[1]) / (res_in_y * 1000)\
                             )
            print(extent_target)
            npix_sel_x_target = int(extent_target[1] - extent_target[0] + 0.5)
            npix_sel_y_target = int(extent_target[3] - extent_target[2] + 0.5)
            print(npix_sel_x_target)
            print(npix_sel_y_target)
            if isinstance(resolution_out, int):
                if not (npix_x % npix_sel_x_target == 0 and \
                        npix_y % npix_sel_y_target == 0 ):
                    errst = 'Target number of pixels {res_out_x}, {res_out_y} should be' +\
                            ' integer factors of the input dimensions {inx}, {iny}'
                    raise ValueError(errst.format(res_out_x=npix_sel_x_target,\
                                                  res_out_y=npix_sel_y_target,\
                                                  inx=npix_x, iny=npix_y))
                outdims = (npix_sel_x_target, npix_sel_y_target)
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
                print(pixrat)
                outdims = ((npix_sel_x_target - 1) // pixrat[0] + 1,\
                           (npix_sel_y_target - 1) // pixrat[1] + 1)
                seldims = (outdims[0] * pixrat[0], outdims[1] * pixrat[1])
                print(seldims)
            box = cosmopars['boxsize'] / cosmopars['h']
            # pix zero has its center at 0.5 in pixel units -> floored to zero
            cenpix = ((center_out[0] - lowerleftcorner[0]) / res_in_x,\
                      (center_out[1] - lowerleftcorner[1]) / res_in_y)   
            selregion = (cenpix[0] - 0.5 * seldims[0], cenpix[0] + 0.5 * seldims[0],\
                         cenpix[1] - 0.5 * seldims[1], cenpix[1] + 0.5 * seldims[1])
            print(selregion)
            selinds = [int(reg) for reg in selregion]
            if selinds[1] - selinds[0] != seldims[0]: # might happen (rounding errors; which is modified should then be arbitrary):
                selinds[1] = selinds[0]  + seldims[0]
            if selinds[3] - selinds[2] != seldims[1]: # might happen (rounding errors; which is modified should then be arbitrary):
                selinds[3] = selinds[2] + seldims[1]
            edges_sel = [lowerleftcorner[0] + res_in_x * selinds[0],\
                         lowerleftcorner[0] + res_in_x * selinds[1],\
                         lowerleftcorner[1] + res_in_x * selinds[2],\
                         lowerleftcorner[1] + res_in_x * selinds[3],\
                         ]
            print(edges_sel)
            print(extent_x)
            print(extent_y)
            if edges_sel[1] - edges_sel[0] > extent_x or \
               edges_sel[3] - edges_sel[2] > extent_y:
                raise RuntimeError('Selected region is larger than the region in the map. '+\
                                   'This might be the result of rounding to match the chosen resolution.')
            inds_wholebox_x = int(box / res_in_x + 0.5)
            inds_wholebox_y = int(box / res_in_y + 0.5)
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
        