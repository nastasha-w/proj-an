#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:39:30 2020

@author: wijers

Use with caution! This just changes maps by a constant factor, and doesn't
check if the correction has already been applied

overall maps: should be ok
IGM image stamps: don't touch, ok now!
"""

#stamps overcorrected: 
# fe19 for profiles
# fe19, fe17-other1 for maps
import numpy as np
import os
import h5py
import fnmatch

import eagle_constants_and_units as c

donestamps = ['stamps_emission_fe18_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5',\
              'stamps_emission_fe17-other1_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5',\
              'stamps_emission_fe19_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5',\
              'emission_fe18_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5',\
              'emission_fe17-other1_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5',\
              'emission_fe19_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5',\
              ]

def correctmap(filename):
    print('correcting file {}'.format(filename))
    with h5py.File(filename, 'a') as fi:
        print(fi.keys())
        fi['map'][:] -= np.log10(c.planck)
        fi['map'].attrs['max'] -= np.log10(c.planck)
        fi['map'].attrs['minfinite'] -= np.log10(c.planck)
    print('... done')

def uncorrectmap(filename):
    print('uncorrecting file {}'.format(filename))
    with h5py.File(filename, 'a') as fi:
        print(fi.keys())
        fi['map'][:] += np.log10(c.planck)
        fi['map'].attrs['max'] += np.log10(c.planck)
        fi['map'].attrs['minfinite'] += np.log10(c.planck)
    print('... done')
    
def correctfiles_maps(directory):
    lines = ['fe18', 'fe17-other1', 'fe19']
    base = 'emission_{line}_*_test3.5_*.hdf5'
    for line in lines:
        files = fnmatch.filter(next(os.walk(directory))[2],\
                               base.format(line=line))
        for file in files:
            correctmap(directory + file)

def correctstamps(filename):
    print('correcting file {}'.format(filename))
    with h5py.File(filename, 'a') as fi:
        sgrps = list(fi.keys())
        # for galaxy stamps
        if 'Header' in sgrps:
            sgrps.remove('Header')
        if 'selection' in sgrps:
            sgrps.remove('selection')
        for sgrp in sgrps:
            print('correcting {}'.format(sgrp))
            # 'IGM' stamps:
            if 'stamp' in sgrp:
                fi['{g}/map'.format(g=sgrp)][:] -= np.log10(c.planck)
            else: # 'CGM stamps' 
                fi[sgrp][:] -= np.log10(c.planck)
    print('... done')
    
def uncorrectstamps(filename):
    print('uncorrecting file {}'.format(filename))
    with h5py.File(filename, 'a') as fi:
        sgrps = list(fi.keys())
        # for galaxy stamps
        if 'Header' in sgrps:
            sgrps.remove('Header')
        if 'selection' in sgrps:
            sgrps.remove('selection')
        for sgrp in sgrps:
            print('correcting {}'.format(sgrp))
            # 'IGM' stamps:
            #if 'stamp' in sgrp:
            #    fi['{g}/map'.format(g=sgrp)][:] += 2 *np.log10(c.planck)
            #else: # 'CGM stamps' 
            #    fi[sgrp][:] += np.log10(c.planck)
    print('... done')

def correctfiles_stamps(directory):
    lines = ['fe18', 'fe17-other1', 'fe19']
    base = '*emission_{line}_*_test3.5_*.hdf5'
    for line in lines:
        files = fnmatch.filter(next(os.walk(directory))[2],\
                               base.format(line=line))
        print(files)
        for file in files:
            if 'stamp' in file and file not in donestamps: # not in the pattern match because it can e before of after 'emission' in the names
                correctstamps(directory + file)

def correctall():
    #mdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/'
    #correctfiles_maps(mdir) #done
    
    sdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/proc/stamps/'
    correctfiles_stamps(sdir)

def undocorr(): # double-corrected some of these maps
    fnsstamp = ['stamps_emission_fe19_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS_1slice_to-3R200c_L0100N1504_27_Mh0p5dex_1000_centrals.hdf5',\
                'emission_fe17-other1_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5',\
                'emission_fe19_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_noEOS_stamps.hdf5',\
                ]
    #mdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/'
    sdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/proc/stamps/'
    
    for filen in fnsstamp:
        uncorrectstamps(sdir + filen)
    
    
if __name__ == '__main__':
    #correctall()
    undocorr()