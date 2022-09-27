#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def quicklook_massmap(filen, savename=None, minval=None):
    '''
    quick plot of the mass map in the file
    '''

    with h5py.File(filen, 'r') as f: