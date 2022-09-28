#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd
import sys
import os

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

def test_mainhalodata_units(opt=1, dirpath=None, snapnum=None,
                            printfile=None):
    
    if opt == 1: # redshift 0 test
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_600/snapshot_600.0.hdf5'
        snapnum = 600
    elif opt == 2: # higher z test 
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_399/snapshot_399.0.hdf5'
        snapnum = 399
    elif opt == 3: # try other z
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_492/snapshot_492.0.hdf5'
        snapnum = 492
    elif opt is None:
        pathopts = ['output/snapdir_{sn:03d}/snapshot_{sn:03d}.0.hdf5',
                    'output/snapshot_{sn:03d}.hdf5']
        goodpath = False
        for pathopt in pathopts:
            snapfile = dirpath + pathopt.format(sn=snapnum)
            if os.path.isfile(snapfile):
                goodpath = True
                break
        if not goodpath:
            tried = [dirpath + pathopts.format()]
            msg = 'Could not find snapshot {} in {}. Tried:'.format(snapnum, dirpath)
            msg = msg + '\n' + '\n'.join(tried)
            raise RuntimeError(msg)
    else:
        msg = 'test_mainhalodata_units parameter opt = {} is invalid'
        raise ValueError(msg.format(opt))

    halodat = mainhalodata(dirpath, snapnum)
    snap = rf.Firesnap(snapfile) 
    cen = np.array([halodat['Xc_ckpcoverh'], 
                    halodat['Yc_ckpcoverh'], 
                    halodat['Zc_ckpcoverh']])
    cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    rvir_cm = halodat['Rvir_ckpcoverh'] * snap.cosmopars.a\
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

    if printfile is not None:
        new = not os.path.isfile(printfile)
        with open(printfile, 'a') as f:
            if new:
                columns = ['snapnum', 'redshift', 'Mvir_sum_Msun', 'Mvir_AHF_Msun']
                f.write('\t'.join(columns) + '\n')
            vals = [snapnum, snap.cosmopars.z, hm_sum_msun, hm_list_msun]
            f.write('\t'.join([str(val) for val in vals]) + '\n')

# checkinh halo_0000_smooth.dat:
# Mvir is exactly flat over a large range of redshift values in that file
# might be an AHF issue?
def test_mainhalodata_units_multi(dirpath, printfile):
    print('running test_mainhalodata_units_multi')
    _snapdirs = os.listdir(dirpath + 'output/')
    snaps = []
    for _sd in _snapdirs:
        # looking for something like snapdir_196, extract 196
        if _sd.startswith('snapdir'):
            _snap = int(_sd.split('_')[-1])
            # special case, permissions error
            try: 
                os.listdir(dirpath + 'output/' + _sd)
                snaps.append(_snap)
            except PermissionError:
                # shows up seemingly randomly
                print('\nskipping snapshot {} due to permissions issues\n'.format(_snap))
                continue
        elif _sd.startswith('snapshot') and _sd.endswith('.hdf5'):
            # something like snapshot_164.hdf5
            _snap = int((_sd.split('_')[-1]).split('.')[0])
            try:
                f = h5py.File(dirpath + 'output/' + _sd, 'r')
                f.close()
            except Exception as err:
                print('\nSkipping snapshot {} due to h5py read issues:')
                print(err)
                print('\n')
                
    for snap in snaps:
        print('Snapshot ', snap)
        test_mainhalodata_units(opt=None, dirpath=dirpath, snapnum=snap,
                                 printfile=printfile)
        print('\n')

def test_mainhalodata_units_multi_handler(opt=1):
    if opt == 1:
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        printfile = '/projects/b1026/nastasha/tests/start_fire/AHF_unit_tests/'
        printfile += 'metal_diffusion__m12i_res7100.txt'
    elif opt == 2:
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m11i_res7100/'
        printfile = '/projects/b1026/nastasha/tests/start_fire/AHF_unit_tests/'
        printfile += 'metal_diffusion__m11i_res7100.txt'
    else:
        raise ValueError('opt {} is not allowed'.format(opt))
    print('Running test_mainhalodata_units_multi(dirpath, printfile)')
    print('dirpath: ', dirpath)
    print('printfile: ', printfile)
    test_mainhalodata_units_multi(dirpath, printfile)


def massmap(snapfile, dirpath, snapnum, radius_rvir=2., particle_type=0,
            pixsize_pkpc=3., axis='z', outfilen=None):
    '''
    Creates a mass map projected perpendicular to a line of sight axis
    by assuming the simulation resolution elements divide their mass 
    following a C2 SPH kernel.

    Parameters:
    -----------
    snapfile: str
        file (or example file, if split) containing the snapshot data
    dirpath: str
        path to the directory containing the 'output' directory with the
        snapshots
    snapnum: int
        snapshot number
    radius_rvir: float 
        radius of the cube to project in units of Rvir. Note that in the sky 
        plane, this will be (slightly) extended to get the exact pixel size.
    particle_type: int
        particle type to project (follows FIRE format)
    pixsize_pkpc: float
        size of the map pixels in proper kpc
    axis: str, 'x', 'y', or 'z'
        axis corresponding to the line of sight 
    outfilen: str or None. 
        if a string, the name of the file to save the output data to. The
        default is None, meaning the maps are returned as output
    
    Output:
    -------
    massW: 2D array of floats
        projected mass image [log g/cm^-2]
    massQ: NaN array, for future work


    '''
    if axis == 'z':
        Axis1 = 0
        Axis2 = 1
        Axis3 = 2
    elif axis == 'x':
        Axis1 = 2
        Axis2 = 0
        Axis3 = 1
    elif axis == 'y':
        Axis1 = 1
        Axis2 = 2
        Axis3 = 0
    else:
        msg = 'axis should be "x", "y", or "z", not {}'
        raise ValueError(msg.format(axis))

    halodat = mainhalodata(dirpath, snapnum)
    snap = rf.Firesnap(snapfile) 
    cen = np.array([halodat['Xc_ckpcoverh'], 
                    halodat['Yc_ckpcoverh'], 
                    halodat['Zc_ckpcoverh']])
    cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    rvir_cm = halodat['Rvir_ckpcoverh'] * snap.cosmopars.a \
              * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    
    # calculate pixel numbers and projection region based
    # on target size and extended for integer pixel number
    target_size_cm = np.array([2. * radius_rvir * rvir_cm] * 3)
    pixel_cm = pixsize_pkpc * c.cm_per_mpc * 1e-3
    npix3 = np.ceil(target_size_cm / pixel_cm) 
    npix_x = npix3[Axis1]
    npix_y = npix3[Axis2]
    size_touse_cm = target_size_cm
    size_touse_cm[Axis1] = npix_x * pixel_cm
    size_touse_cm[Axis2] = npix_y * pixel_cm

    basepath = 'PartType{}/'.format(particle_type)
    haslsmooth = particle_type == 0
    if haslsmooth: # gas
        lsmooth = snap.readarray_emulateEAGLE(basepath + 'SmoothingLength')
        lsmooth_toCGS = snap.toCGS

    coords = snap.readarray_emulateEAGLE(basepath + 'Coordinates')
    coords_toCGS = snap.toCGS
    # needed for projection step anyway
    coords -= cen_cm / coords_toCGS
    # select box region
    # zoom regions are generally centered -> don't worry
    # about edge overlap
    box_dims_coordunit = size_touse_cm / coords_toCGS

    if haslsmooth:
        # extreme values will occur at zoom region edges -> restrict
        lmax = np.max(lsmooth[filter]) 
        conv = lsmooth_toCGS / coords_toCGS
        # might be lower-density stuff outside the region, but overlapping it
        lmargin = 2. * lmax * conv
        filter = np.all(np.abs((coords)) <= 0.5 * box_dims_coordunit \
                        + lmargin, axis=1)
        lsmooth = lsmooth[filter]
        if not np.isclose(conv, 1.):
            lsmooth *= conv
    
    else:
        filter = np.all(np.abs((coords)) <= 0.5 * box_dims_coordunit, axis=1)   
    
    coords = coords[filter]
    masses = snap.readarray_emulateEAGLE(basepath + 'Masses')[filter]
    masses_toCGS = snap.toCGS
    
    # stars, black holes. DM: should do neighbour finding. Won't though.
    if not haslsmooth:
        # minimum smoothing length is set in the projection
        lsmooth = np.zeros(shape=(len(masses),), dtype=coords.dtype)
        lsmooth_toCGS = 1.
    
    tree = False
    periodic = False # zoom region
    NumPart = len(masses)
    dct = {'coords': coords, 'lsmooth': lsmooth, 
           'qW': masses, 
           'qQ': np.zeros(len(masses), dtype=np.float32)}
    Ls = box_dims_coordunit
    # cosmopars uses EAGLE-style cMpc/h units for the box
    box3 = [snap.cosmopars.boxsize * c.cm_per_mpc / snap.cosmopars.h \
            / coords_toCGS] * 3
    mapW, mapQ = project(NumPart, Ls, Axis1, Axis2, Axis3, box3,
                         periodic, npix_x, npix_y,
                         'C2', dct, tree, ompproj=True, 
                         projmin=None, projmax=None)
    lmapW = np.log10(mapW)
    lmapW += np.log10(masses_toCGS)
    if outfilen is None:
        return lmapW, mapQ
    
    with h5py.File(outfilen, 'w') as f:
        # map (emulate make_maps format)
        f.create_dataset('map', lmapW)
        f['map'].attrs.create('log', True)
        minfinite = np.min(lmapW[np.isfinite(lmapW)])
        f['map'].attrs.create('minfinite', minfinite)
        f['map'].attrs.create('max', np.max(lmapW))
        
        # cosmopars (emulate make_maps format)
        hed = f.create_group('Header')
        cgrp = hed.create_group('inputpars/cosmopars')
        csm = snap.cosmopars.getdct()
        for key in csm:
            cgrp.create_attribute(key, csm[key])
        
        # direct input parameters
        igrp = hed['inputpars']
        igrp.attrs.create('snapfile', np.string_(snapfile))
        igrp.attrs.create('dirpath', np.string_(dirpath))
        igrp.attrs.create('radius_rvir', radius_rvir)
        igrp.attrs.create('particle_type', particle_type)
        igrp.attrs.create('pixsize_pkpc', pixsize_pkpc)
        igrp.attrs.create('axis', np.string_(axis))
        igrp.attrs.create('outfilen', np.string_(outfilen))
        # useful derived/used stuff
        igrp.attrs.create('Axis1', Axis1)
        igrp.attrs.create('Axis2', Axis2)
        igrp.attrs.create('Axis3', Axis3)
        igrp.attrs.create('diameter_used_cm', np.array(size_touse_cm))
        if haslsmooth:
            igrp.attrs.create('margin_lsmooth_cm', lmargin * coords_toCGS)
        _grp = igrp.create_group('halodata')
        for key in halodat:
            _grp.attrs.create(key, halodat[key])
        
        




def fromcommandline(index):
    '''
    This mapping is just based on the order in which I (first) ran things,
    and will not generally follow any kind of logic
    '''
    print('Running fire_maps.py process {}'.format(index))
    if index > 0 and index < 4:
        test_mainhalodata_units(opt=index)
    elif index == 4:
        # test a whole lot of snapshots in one go
        test_mainhalodata_units_multi_handler(opt=1)
    elif index == 5:
        test_mainhalodata_units_multi_handler(opt=2)
    else:
        raise ValueError('Nothing specified for index {}'.format(index))

if __name__ == '__main__':
    print('fire_maps.py script started')
    if len(sys.argv) > 1:
        try:
            ind = int(sys.argv[1])
        except ValueError as msg1:
            msg2 = 'Could not interpret first command-line argument {} as int'
            msg2.format(sys.argv[1])
            raise ValueError('/n'.join(msg1, msg2))
    else:
        raise ValueError('Please specify an integer index > 1')
    fromcommandline(ind)
    
    








