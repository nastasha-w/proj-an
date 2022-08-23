#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd
import sys

import readin_fire_data as rf
import units_fire as uf
import cosmo_utils as cu
import eagle_constants_and_units as c
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
    # units from AHF docs: http://popia.ft.uam.es/AHF/files/AHF.pdf
    props = ['Mvir', 'Rvir', 'Xc', 'Yc', 'Zc']
    outprops = {'Mvir': 'Mvir_Msunoverh',
                'Rvir': 'Rvir_ckpcoverh',
                'Xc':   'Xc_ckpcoverh',
                'Yc':   'Yc_ckpcoverh',
                'Zc':   'Zc_ckpcoverh'}
    for prop in props:
        out[outprops[prop]] = df[prop][i]
    return out

def test_mainhalodata_units(opt=1):
    
    if opt == 1: # redshift 0 test
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_600/snapshot_600.0.hdf5'
        snapnum = 600
    elif opt == 2: # higher z test 
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_399/snapshot_399.0.hdf5'
        snapnum = 399
    else:
        msg = 'test_mainhalodata_units parameter opt = {} is invalid'
        raise ValueError(msg.format(opt))

    halodat = mainhalodata(dirpath, snapnum)
    snap = rf.Firesnap(snapfile) 
    cen = np.array([halodat['Xc_ckpcoverh'], 
                    halodat['Yc_ckpcoverh'], 
                    halodat['Zc_ckpcoverh']])
    cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    rvir_cm = halodat['Rvir_ckpcoverh'] * snap.cosmopars.a \
              * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    print('Cosmology:')
    print(snap.cosmopars.getdct())
    print('Center [AHF units]: {}'.format(cen))
    print('Rvir [AHF units]: {}'.format(halodat['Rvir_ckpcoverh']))
    print('Center [attempted cm]: {}'.format(cen_cm))
    print('Rvir [attempted cm]: {}'.format(rvir_cm))
    
    # gas
    coords_pt0 = snap.readarray_emulateEAGLE('PartType0/Coordinates')
    coords_pt0_toCGS = snap.toCGS
    masses_pt0 = snap.readarray_emulateEAGLE('PartType0/Masses')
    masses_pt0_toCGS = snap.toCGS
    # sanity check
    med_c = np.median(coords_pt0, axis=0)
    print('Median gas coords [sim units]: {}'.format(med_c))
    print('Median gas coordinates [cm]: {}'.format(med_c * coords_pt0_toCGS))

    d2 = np.sum((coords_pt0 - cen_cm / coords_pt0_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt0_toCGS) **2
    hm_pt0 = np.sum(masses_pt0[sel])
    print('Halo gas mass (sim units): ', hm_pt0)
    print('Selected {}/{} particles'.format(np.sum(sel), len(sel)))
    del coords_pt0
    del masses_pt0
    del d2
    del sel
    # dm (high-res)
    coords_pt1 = snap.readarray_emulateEAGLE('PartType1/Coordinates')
    coords_pt1_toCGS = snap.toCGS
    masses_pt1 = snap.readarray_emulateEAGLE('PartType1/Masses')
    masses_pt1_toCGS = snap.toCGS
    med_c = np.median(coords_pt1, axis=0)
    print('Median DM coords [sim units]: {}'.format(med_c))
    print('Median DM coordinates [cm]: {}'.format(med_c * coords_pt1_toCGS))
    d2 = np.sum((coords_pt1 - cen_cm / coords_pt1_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt1_toCGS) **2
    hm_pt1 = np.sum(masses_pt1[sel])
    print('Halo dm mass (sim units): ', hm_pt1)
    print('Selected {}/{} particles'.format(np.sum(sel), len(sel)))
    del coords_pt1
    del masses_pt1
    del d2
    del sel
    # stars
    coords_pt4 = snap.readarray_emulateEAGLE('PartType4/Coordinates')
    coords_pt4_toCGS = snap.toCGS
    masses_pt4 = snap.readarray_emulateEAGLE('PartType4/Masses')
    masses_pt4_toCGS = snap.toCGS
    med_c = np.median(coords_pt4, axis=0)
    print('Median star coords [sim units]: {}'.format(med_c))
    print('Median star coordinates [cm]: {}'.format(med_c * coords_pt4_toCGS))

    d2 = np.sum((coords_pt4 - cen_cm / coords_pt4_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt4_toCGS) **2
    hm_pt4 = np.sum(masses_pt4[sel])
    print('Halo stellar mass (sim units): ', hm_pt4)
    del coords_pt4
    del masses_pt4
    del d2
    del sel
    hm = hm_pt0 + hm_pt1 + hm_pt4

    msg = 'Got halo mass {hm}, listed Mvir is {Mvir}'
    hm_list_msun = halodat['Mvir_Msunoverh'] / snap.cosmopars.h
    hm_sum_msun = hm * (masses_pt0_toCGS / cu.c.solar_mass)
    print(msg.format(hm=hm_sum_msun, Mvir=hm_list_msun))
    hm_logmsun = np.log10(hm) + np.log10(masses_pt0_toCGS / cu.c.solar_mass)
    print('sum total is 10^{logm} Msun'.format(logm=hm_logmsun))

def massmap():
    pass

def fromcommandline(index):
    if index > 0 and index < 3:
        test_mainhalodata_units(opt=index)
    else:
        raise ValueError('Nothing specified for index {}'.format(index))

if __name__ == '__main__':
    print('fire_maps.py script started')
    if len(sys.argv) > 1:
        ind = int(sys.argv[1])
    else:
        raise ValueError('Please specify an integer index > 1')
    fromcommandline(ind)
    
    








