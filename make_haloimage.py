#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:25:12 2020

@author: Nastasha
"""

import numpy as np
import h5py

import matplotlib.pyplot as plt

import make_maps_v3_master as m3
import eagle_constants_and_units as c
import make_maps_opts_locs as ol


halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'

def selecthalo(logM200c, _halocat=halocat, margin=0.05):
    '''
    selects a random halo in the target mass range and prints basic data about
    it
    
    Parameters
    ----------
    logM200c : float [log10 Msun]
        target halo mass M200c.
    _halocat : string, optional
        The name of the file containing the halo data. If no path is included,
        the file is assumed to be in make_maps_opts_locs.pdir
        The default is halocat.
    margin : float, optional
        the maximum difference with the target halo mass allowed (in log10 
        Msun). A rondom halo is selected within this range.
        The default is 0.05.

    Raises
    ------
    RuntimeError
        If there is no halo in the selected mass range.

    Returns
    -------
    None.

    '''
    
    if '/' not in _halocat:
        _halocat = ol.pdir + _halocat
    with h5py.File(_halocat, 'r') as hc:
        boxdata = {key: val for key, val in hc['Header'].attrs.items()}
        cosmopars = {key: val for key, val in\
                     hc['Header/cosmopars'].attrs.items()}
        m200c = np.log10(hc['M200c_Msun'][:])
        galid = hc['galaxyid'][:]
        close = np.where(np.abs(m200c - logM200c) < margin)[0]
        if len(close) < 1:
            raise RuntimeError('No haloes in the selected mass range')
        ind = np.random.choice(close)
        m200c = m200c[ind]
        galid = galid[ind]
        cenx = hc['Xcom_cMpc'][ind]
        ceny = hc['Ycom_cMpc'][ind]
        cenz = hc['Zcom_cMpc'][ind]
        R200 = hc['R200c_pkpc'][ind] * 1e-3 / cosmopars['a'] # to cMpc
        
    print('Selected galaxy {galid} with log10 M200c / Msun {m200c}'.format(\
          galid=galid, m200c=m200c))
    print('Center [{}, {}, {}] cMpc, R200c {R200c} cMpc'.format(\
          cenx, ceny, cenz, R200c=R200))
    print(boxdata)
    return None

def getimgs(cen, size, sizemargin=2.):
    
    simnum = 'L0100N1504'
    snapnum = 27
    centre = cen
    L_z = size * sizemargin
    L_x = L_y = L_z
    npix_x = 400
    npix_y = npix_x
    firstargs = (simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y)
    
    kwargs_all = {'exlcudeSFRW': 'T4',\
                  'var': 'REFERENCE',\
                  'axis': 'z',\
                  'periodic': False,\
                  'saveres': True,\
                  'hdf5': True,\
                  'saveres': True,\
                  'ompproj': True}
    
    argsets = [('basic',), {'quantityW': 'Mass', 'ptypeQ': 'basic',\
                            'quantityQ': 'Temperature'},\
               ('basic',), {'quantityW': 'Mass', 'ptypeQ': 'basic',\
                            'quantityQ': 'Density'},\
               ('coldens',), {'ionW': 'o7', 'abunds': 'Pt'},\
               ('emission',), {'ionW': 'o7r', 'abunds': 'Sm'},\
               ]
    
    names = []
    for argset in argsets:
        args = firstargs + argsets[0]
        kwargs = kwargs_all.copy()
        kwargs.update(argset[1])
        
        name = m3.make_map(*args, nameonly=True, **kwargs)
        names.append(name)
        m3.make_map(*args, nameonly=False **kwargs)
    
    print('Done creating images:')
    print(names)
        
    