#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:53:54 2020

@author: Nastasha
"""

import numpy as np

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
    group_out:     (str) name of the hdf5 group to store the stamp in. Default
                   is 'stamp#', where '#' is a number (starting at 0)
    resolution out: (float, int, or None) resolution of the final map.
                   None -> keep old resolution
                   int  -> use that many pixels (must divide the number of 
                   pixels in the selected region)
                   float -> use that many cpkc per pixel (integer multiple of 
                   original resolution)
    center_out:    (tuple of 2 floats) center of the region to cut out of the 
                   image (units: cMpc). If None, the whole image is used
    diameter out:  (float or tuple of 2 floats, or None) extent of the region 
                   to cut out of the image. May overlap periodic box edges. 
                   If one float, the region is square, if two, matches the axes
                   of the image. May not be None if center_out isn't.
    '''
    
