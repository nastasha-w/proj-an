#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:53:54 2020

@author: Nastasha
"""

import numpy as np

import eagle_constants_and_units as c
import cosmo_utils as cu

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
    
