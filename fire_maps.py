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

def test_mainhalodata_units():

    dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
    snapfile = dirpath + 'output/snapdir_600/snapshot_600.0.hdf5'
    snapnum = 600

    halodat = mainhalodata(dirpath, snapnum)
    snap = rf.Firesnap(snapfile) 
    cen = np.array([[halodat['Xc'], halodat['Yc'], halodat['Zc']]])
    
    # gas
    coords_pt0 = snap.readarray_emulateEAGLE('PartType0/Coordinates')
    coords_pt0_toCGS = snap.toCGS
    masses_pt0 = snap.readarray_emulateEAGLE('PartType0/Masses')
    masses_pt0_toCGS = snap.toCGS
    # dm (high-res)
    coords_pt1 = snap.readarray_emulateEAGLE('PartType1/Coordinates')
    coords_pt1_toCGS = snap.toCGS
    masses_pt1 = snap.readarray_emulateEAGLE('PartType1/Masses')
    masses_pt1_toCGS = snap.toCGS
    # stars
    coords_pt4 = snap.readarray_emulateEAGLE('PartType4/Coordinates')
    coords_pt4_toCGS = snap.toCGS
    masses_pt4 = snap.readarray_emulateEAGLE('PartType4/Masses')
    masses_pt4_toCGS = snap.toCGS
    
    d2 = (coords_pt0 - cen)**2
    sel = d2 <= halodat['Rvir']**2
    hm_pt0 = np.sum(masses_pt0[sel])
    d2 = (coords_pt1 - cen)**2
    sel = d2 <= halodat['Rvir']**2
    hm_pt1 = np.sum(masses_pt1[sel])
    d2 = (coords_pt4 - cen)**2
    sel = d2 <= halodat['Rvir']**2
    hm_pt4 = np.sum(masses_pt4[sel])
    hm = hm_pt0 + hm_pt1 + hm_pt4

    msg = 'Got halo mass {hm}, listed Mvir is {Mvir}'
    print(msg.format(hm=hm, Mvir=halodat['Mvir']))
    hm_logmsun = np.log10(hm) + np.log10(masses_pt0_toCGS / cu.c.solar_mass)
    print('sum total is 10^{logm} Msun'.format(logm=hm_logmsun))









