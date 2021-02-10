#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:50:25 2021

@author: Nastasha
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py


import make_maps_v3_master as m3
import prof3d_galsets as p3g 
import selecthalos as sh


# set output paths
wdir = '/cosma5/data/dp004/dc-wije1/smallprojects/absorber_nHTZ_sample/'
p3g.tdir = wdir
m3.ndir = wdir + 'maps/'

### sample
#simnum = 'L0100N1504'
#snap = 25
#var = 'REFERENCE'

halocat = ol.pdir + 'catalogue_RefL0100N1504_snap25_aperture30.hdf5'

kwargs_gen = {'excludeSFRW': 'T4', 'excludeSFRQ': False, 
              'abundsW': 'Pt', 'abundsQ': 'Pt',
              'parttype': '0', 'simulation': 'eagle', 'periodic': False,
              'kernel': 'C2', 'LsinMpc': True,
              'axis': 'z', 'log': True, 'saveres': True, 'hdf5': True,
              'velcut': False, 'ompproj': True}

kwargs_l1 = [{'ionW': 'hydrogen'},
             {'ionW': 'hneutralssh'}]
kwargs_l2 = [{'ptypeQ': 'basic', 'quantityQ': 'Density'},
             {'ptypeQ': 'basic', 'quantityQ': 'Temperature'},
             {'ptypeQ': 'basic', 'quantityQ': 'Metallicity'},
             ]

def mapname_file(samplename, kwargs):
    base = 'maps_{sample}_{ionW}_{quantityQ}.txt'
    return base.format(sample=samplename, **kwargs)


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
    
    with open(galfile, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    msg = 'Reached the end of {} without finding the halo catalogue name'
                    raise RuntimeError(msg.format(galfile))
                else:
                    break
            elif line.startswith('halocat'):
                halocat = line.split(':')[1]
                halocat = halocat.strip()
                headlen += 1
            elif ':' in line or line == '\n':
                headlen += 1
                
    with h5py.File(halocat, 'r') as hc:
        hed = hc['Header']
        cosmopars = {key: item for key, item in hed['cosmopars'].attrs.items()}
        simnum = hed.attrs['simnum']
        snapnum = hed.attrs['snapnum']
        var = hed.attrs['var']
    
    galdata_all = pd.read_csv(galfile, header=headlen,
                              sep='\t', index_col='galaxyid')
    galaxyids = np.array(galdata_all.index)
    
    for kw1 in kwargs_l1:
        for kw2 in kwargs_l2:
            _kwargs = kwargs_gen.copy()
            _kwargs.update(kw1)
            _kwargs.update(kw2)
            _kwargs['var'] = var
            
            mapname_filen = mapname_file(samplename, _kwargs)
    
            with open(mapname_filen, 'w') as fdoc:
                fdoc.write('galaxyid\tmapW\tmapQ\n')
                
                for gid in galaxyids:
                    R200c = galdata_all.at[gid, 'R200c_cMpc']
                    Xcom = galdata_all.at[gid, 'Xcom_cMpc']
                    Ycom = galdata_all.at[gid, 'Ycom_cMpc']
                    Zcom = galdata_all.at[gid, 'Zcom_cMpc']
                    M200 = galdata_all.at[gid, 'M200c_Msun']
                    
                    cen = [Xcom, Ycom, Zcom]
                    L_x = diameter_R200c * R200c
                    L_x = (np.ceil(L_x / (pixelsize_ckpc * 1e-3)) + 2.)\
                          * pixelsize_ckpc * 1e-3
                    L_y = L_x
                    L_z = L_x
                    npix = int(np.round(L_x /  (pixelsize_ckpc * 1e-3), 0))
                    
                    if _kwargs['axis'] == 'z':
                        L_z = los_R200c * R200c
                    elif _kwargs['axis'] == 'y':
                        L_y = los_R200c * R200c
                    elif _kwargs['axis'] == 'x':
                        L_x = los_R200c * R200c
                    
                    args = (simnum, snapnum, cen, L_x, L_y, L_z, npix, npix,
                            'coldens')
                
                    # ion, quantity, nameonly,
                    outname = m3.make_mape(*args, nameonly=True, **_kwargs)
                    
                    alreadyexists = False
                    if os.path.isfile(outname[0]) and os.path.isfile(outname[1]):
                        alreadyexists = True
                    if alreadyexists:
                        print('For galaxy {}, a ap already exists; skipping'.format(gid))
                    else:
                        m3.make_map(*args, nameonly=False, **_kwargs)
                    
                    fdoc.write('{}\t{}\t{}\n'.format(gid, outname[0],
                                                     outname[1]))


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

    

