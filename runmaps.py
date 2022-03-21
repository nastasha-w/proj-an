#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os

import make_maps_v3_master as m3

jobind = None

if __name__ == '__main__':
    args = sys.argv[1:]
    jobind = int(args[0])
    
# maps for C I, Fe II, Si IV, or C IV with O VII and O VIII
# Smita Mathur proposal: UV/X-ray ion correlations
if jobind in range(1, 97):
    ions = ['c1', 'fe2', 'si4', 'c4', 'o7', 'o8']
    totind = jobind - 1
    zind = totind % 16
    ionind = totind // 16

    cenz = 3.125 + 6.25 * zind
    ion = ions[ionind]

    simnum = 'L0100N1504'
    snapnum = 28
    L_x = 100.
    L_y = 100.
    L_z = 6.25
    centre = [50., 50., cenz]
    npix_x = 32000
    npix_y = 32000
    ptypeW = 'coldens'
    args = (simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, ptypeW)

    kwargs = {'ionW': ion, 'abundsW':'Pt', 'quantityW': None,
              'ionQ': None, 'abundsQ': 'auto', 'quantityQ': None, 
              'ptypeQ': None, 'excludeSFRW': 'T4', 'excludeSFRQ': False, 
              'parttype': '0',
              'sylviasshtables': False, 'bensgadget2tables': False,
              'ps20tables': True, 'ps20depletion': True,
              'var': 'REFERENCE', 'axis': 'z','log': True, 'velcut': False,
              'periodic': True, 'kernel': 'C2', 'saveres': True,
              'simulation': 'eagle', 'LsinMpc': True,
              'select': None, 'filelabel_select': None,
              'misc': None, 'halosel': None, 'kwargs_halosel': None,
              'excludedirectfb': False, 'deltalogT_directfb': 0.2, 
              'deltatMyr_directfb': 10., 'inclhotgas_maxlognH_snfb': 2.,
              'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
              'ompproj': True, 'numslices': None, 
              'hdf5': True, 'override_simdatapath': None,
              }
    outname = m3.make_map(*args, nameonly=True, **kwargs)
    run = True
    if os.path.isfile(outname[0]):
        if outname[1] is None:
            run = False
        elif os.path.isfile(outname[1]):
            run = False
    if run:
        m3.make_map(*args, nameonly=False, **kwargs)











