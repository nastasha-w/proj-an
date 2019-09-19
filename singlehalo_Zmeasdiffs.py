#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:38:11 2019

@author: wijers
"""

import numpy as np
import h5py
import pandas as pd

import matplotlib.pyplot as plt

import make_maps_opts_locs as ol
import make_maps_v3_master as m3


# where to put text files and plots (projections go in ndir)
mdir = '/net/luttero/data2/imgs/Zmeascomps/singlegals/'


# generate a galaxy sample (centrals only)
def gensample(setname, halocat, targets_Mstar=(9.0, 9.5, 10.0, 10.5), tol_Mstar=0.05, numeach=4, onlycentrals=True):
    '''
    generate a galaxy sample, save the resulting data in mdir/setname.txt
    select by stellar mass ()
    '''
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
        
    with h5py.File(halocat, 'r') as hc:
        mstar = np.log10(hc['Mstar_Msun'])
        subn  = np.array(hc['SubGroupNumber']) 
        if onlycentrals:
            mstar[subn > 0] = np.NaN # exlcude from mass selection
        # select the halos
        subsets = [np.where(np.abs(tarm - mstar) <= tol_Mstar)[0] for tarm in targets_Mstar]
        selinds = [np.random.choice(sub, size=numeach, replace=False) if len(sub) > numeach else sub for sub in subsets]
        selinds = np.array([ind for sel in selinds for ind in sel])
        
        # get other useful data on them
        galids = np.array(hc['galaxyid'])[selinds]
        groupids = np.array(hc['groupid'])[selinds]
        mstar = mstar[selinds]
        subn = subn[selinds]
        xpos = np.array(hc['Xcom_cMpc'])[selinds]
        ypos = np.array(hc['Ycom_cMpc'])[selinds]
        zpos = np.array(hc['Zcom_cMpc'])[selinds]
        R200c_group = np.array(hc['R200c_pkpc'])[selinds] * 1e-3 / hc['Header/cosmopars'].attrs['a']
    
    with open(mdir + setname + '.txt', 'w') as fo:
        # metadata
        fo.write('halocat:\t%s\n'%halocat)
        fo.write('targets_Mstar:\t%s\n'%(str(targets_Mstar)))
        fo.write('tol_Mstar:\t%s\n'%(tol_Mstar))
        fo.write('numeach:\t%i\n'%(numeach))
        fo.write('onlycentrals:\t%s\n'%(onlycentrals))
        # column heads
        fo.write('galaxyid\tgroupid\tMstar_logMsun\tSubGroupNumber\tXcom_cMpc\tYcom_cMpc\tZcom_cMpc\tR200c_cMpc\n')
        for i in range(len(galids)):
            fo.write('%i\t%i\t%f\t%i\t%f\t%f\t%f\t%f\n'%(galids[i], groupids[i], mstar[i], subn[i], xpos[i], ypos[i], zpos[i], R200c_group[i]))

                
# run all the projections for one galaxy
def run_projections(centre, projbox,\
                    pixres=3.125, simnum='L0025N0752', snapnum=19,\
                    var='RECALIBRATED', excludeSFR='T4', axis='z',\
                    halosel=None, kwargs_halosel=None):
    '''
    In the directions _|_ to the axis, the projection region is slightly 
    extended to get the target resolution
    '''
    
    # set number of pixels, adjust box size
    pixres *= 1e-3
    npix = int(np.ceil(projbox / pixres))
    L_p = npix * pixres
    if axis == 'z':
        L_x, L_y = (L_p,) * 2
        L_z = projbox
    elif axis == 'y':
        L_x, L_z = (L_p,) * 2
        L_y = projbox
    elif axis == 'x':
        L_y, L_z = (L_p,) * 2
        L_x = projbox
    
    kwargs_base = {'var': var, 'excludeSFRW': excludeSFR, 'excludeSFRQ': excludeSFR,\
              'axis': axis, 'periodic': False, 'saveres': True, 
              'halosel': halosel, 'kwargs_halosel': kwargs_halosel,\
              'simulation': 'eagle', 'LsinMpc': True, 'ompproj': True,\
              'hdf5': True,\
              'ptypeQ': 'basic', 'quantityQ': 'SmoothedMetallicity'}
    
    args_base = [simnum, snapnum, centre, L_x, L_y, L_z, npix, npix]
    projdct = {'h1ssh': ('coldens', 'h1ssh'),\
               'hneutralssh': ('coldens', 'hneutralssh'),\
               'Gasmass': ('basic', 'Mass'),\
               'SFR': ('basic', 'StarFormationRate')}

    outdct = {}
    for tupk in projdct.keys():
        tup = projdct[tupk] 
        args = tuple(args_base + [tup[0]])
        kwargs = kwargs_base.copy()
        # which is used depends on ptype, but the other one is ignored anyway
        kwargs['ionW'] = tup[1]
        kwargs['quantityW'] = tup[1]
        
        outnames = m3.make_map(*args, nameonly=True, **kwargs)
        outdct[tupk] = outnames
        
        m3.make_map(*args, nameonly=False, **kwargs)
        
    return outdct


def project_sample(setname, pixres=3.125, excludeSFR='T4', axis='z', projrad=(50., 'pkpc')):
    '''
    projrad: (size, units). units are one of pkpc, ckpc, pMpc, cMpc, or R200c
             note that R200c applies to the current parent halo, even if the 
             galaxy is a satellite
    pixres:  size of a pixel in ckpc
    setname: name of the text file containing the info for this set of galaxies
    '''
    textname = mdir + setname + '.txt'
    mdname = mdir + setname + '_projfiles.txt'
    
    with open(textname, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%setname)
                else:
                    break
            elif line.startswith('halocat'):
                halocat = line.split(':')[1]
                halocat = halocat.strip()
                headlen += 1
            elif ':' in line or line == '\n':
                headlen += 1
                
    # infer box, snapnum, var from used catalogue 
    with h5py.File(halocat, 'r') as hc:
        hed = hc['Header']
        cosmopars = {key: item for key, item in hed['cosmopars'].attrs.items()}
        simnum = hed.attrs['simnum']
        snapnum = hed.attrs['snapnum']
        var = hed.attrs['var']
        ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
        
    galdata_all = pd.read_csv(textname, header=headlen, sep='\t')   
    ginds = np.array(galdata_all.index)
    
    with open(mdname, 'w') as to:
        to.write('galaxyid\tweight\tpfile\tqfile\n')
        
    for gind in ginds:
        gdata = galdata_all.loc[gind]
        galid = gdata['galaxyid']
        groupid = gdata['groupid']
        mstar = gdata['Mstar_logMsun']
        centre = [gdata['Xcom_cMpc'], gdata['Ycom_cMpc'], gdata['Zcom_cMpc']] 
        R200c = gdata['R200c_cMpc']
        
        projbox = projrad[0]
        if 'kpc' in projrad[1]:
            projbox *= 1e-3
        if projrad[1][0] == 'p':
            projbox /= cosmopars['a']
        if projrad[1] == 'R200c':
            projbox *= R200c
        
        output_names = run_projections(centre, projbox,\
                        pixres=pixres, simnum=simnum, snapnum=snapnum,\
                        var=var, excludeSFR=excludeSFR, axis=axis,\
                        halosel=None, kwargs_halosel=None)
        
        for namek in output_names:
            names = output_names[namek]
            for name in names:
                with h5py.File(name, 'a') as fo:
                    mgrp = fo.create_group('Header/target_galaxy')
                    mgrp.attrs.create('galaxyid', galid)
                    mgrp.attrs.create('Mstar_%ipkpc_logMsun'%(ap), mstar)
                    mgrp.attrs.create('groupid', groupid)
        
        with open(mdname, 'a') as to:
            for namek in output_names:
                to.write('%i\t%s\t%s\t%s\n'%(galid, namek, output_names[namek][0], output_names[namek][1]))
                
                