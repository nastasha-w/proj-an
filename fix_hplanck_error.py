#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:39:30 2020

@author: wijers

Use with caution! This just changes maps by a constant factor, and doesn't
check if the correction has already been applied
"""

import numpy as np
import os
import h5py
import fnmatch

import eagle_constants_and_units as c

def correctmap(filename):
    print('correcting file {}'.format(filename))
    with h5py.File(filename, 'a') as fi:
        print(fi.keys())
        fi['map'][:] -= np.log10(c.planck)
        fi['map'].attrs['max'] -= np.log10(c.planck)
        fi['map'].attrs['minfinite'] -= np.log10(c.planck)
    print('... done')
    
def correctfiles_maps(directory):
    lines = ['fe18', 'fe17-other1', 'fe19']
    base = 'emission_{line}_*_test3.5_*.hdf5'
    for line in lines:
        files = fnmatch.filter(next(os.walk(directory))[2],\
                               base.format(line=line))
        for file in files:
            correctmap(file)

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
            fi['{g}/map'.format(g=sgrp)][:] -= np.log10(c.planck)
    print('... done')

def correctfiles_stamps(directory):
    lines = ['fe18', 'fe17-other1', 'fe19']
    base = '*emission_{line}_*_test3.5_*.hdf5'
    for line in lines:
        files = fnmatch.filter(next(os.walk(directory))[2],\
                               base.format(line=line))
        for file in files:
            if 'stamp' in file: # not in the pattern match because it can e before of after 'emission' in the names
                correctstamps(file)

def correctall():
    sdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/proc/stamps/'
    mdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/temp/'
    
    correctfiles_maps(mdir)
    correctfiles_stamps(sdir)

if __name__ == '__main__':
    correctall()