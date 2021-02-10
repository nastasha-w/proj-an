#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:50:25 2021

@author: Nastasha
"""

import os
import sys
import numpy as np
import h5py

import make_maps_v3_master as m3
import prof3d_galsets as p3g 
import selecthalos as sh


# set output paths
wdir = '/cosma5/data/dp004/dc-wije1/smallprojects/absorber_nHTZ_sample/'
p3g.tdir = wdir
m3.ndir = wdir + 'maps/'

### sample definition
simnum = 'L0100N1504'
snap = 25
var = 'REFERENCE'

halocat = ol.pdir + 'catalogue_RefL0100N1504_snap25_aperture30.hdf5'


def getsample(size, logM200_Msun_min=12.0, logM200_Msun_max=12.5, seed=0):
    
    Mh_sels = [[('M200c_Msun', 10**logM200_Msun_min, 10**logM200_Msun_max)]]
    Mh_names =['geq{}_le{}'.format(logM200_Msun_min,logM200_Msun_max)]

    sel = sh.Galaxyselector(halocat, selections=Mh_sels, names=Mh_names,\
                         number=size, seed=seed)
    
    samplename = 'galaxies_logM200c-Msun-{mmin}-{mmax}_{num}_seed{seed}.txt'
    samplename = samplename.format(mmin=logM200_Msun_min,
                                   mmax=logM200_Msun_min,
                                   num=size, seed=seed)    
    
    p3g.gensample(samplename=None, galaxyselector=sel)
    return samplename

def create_maps(samplename, los_R200c=4., diameter_R200c=2.1, 
                pixelsize_ckpc=3.125):
    galfile = p3g.dataname(samplename)
    


def create_histogram():
    
    hist, edges = p3g.combine_hists(hist, temp, edges, edges_temp,
                                    rtol=1e-5, atol=1e-8, add=True)
    
    
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        size = 100
        seed = 0
    elif len(sys.argv) == 2:
        size = int(sys.argv[1])
        seed = 0
    elif len(sys.argv) == 3:
        size = int(sys.argv[1])
        seed = int(sys.argv[2])
        
    samplename = getsample(size, logM200_Msun_min=12.0, logM200_Msun_max=12.5,\
                           seed=seed)
    
    create_maps(samplename)
    create_histogram(samplename)

    

