#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt

import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu

def quicklook_massmap(filen, savename=None, mincol=None):
    '''
    quick plot of the mass map in the file
    '''

    with h5py.File(filen, 'r') as f:
        map = f['map']
        vmin = f['map'].attrs['minfinite']
        vmax = f['map'].attrs['max']
