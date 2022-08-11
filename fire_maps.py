#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd

import readin_fire_data as rf
import units_fire as uf
import cosmo_utils as cu
from make_maps_v3_master import linetable_PS20, project


def mainhalodata(path, snapnum):
    '''
    get properties of the main halo in the snapshot from halo_00000_smooth.dat
    assume units are intrinsic simulation units
    '''
    fn = path + '/halo/ahf/halo_00000_smooth.dat'
    df = pd.read_csv(fn, sep='\t')
    i = np.where(df['snum'] == snapnum)[0][0]
    out = {}
    props = ['Mvir', 'Rvir', 'Xc', 'Yc', 'Zc']
    for prop in props:
        out[prop] = df[prop][i]
    return out

def test_mainhalodata():

    dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
    snapfile = dirpath + 'output/snapdir_600/snapshot_600.0.hdf5'
    snapnum = 600

    halodat = mainhalodata(dirpath, snapnum)

    




