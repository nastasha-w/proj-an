#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:50:25 2021

@author: Nastasha

Be careful if you change some of the pipeline elements: parts of the pipieline 
assume only e.g. ionW and quantityQ vary betwen maps.
"""

import os
import sys
import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt

import make_maps_v3_master as m3
import make_maps_opts_locs as ol
import prof3d_galsets as p3g 
import selecthalos as sh


# set output paths
wdir = '/cosma5/data/dp004/dc-wije1/smallprojects/absorber_nHTZ_sample/'
p3g.tdir = wdir
m3.ol.ndir = wdir + 'maps/'

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
units = {'Density': 'log10 g / cm**3',
         'Temperature': 'log10 K',
         'Metallicity': 'log10 mass fraction (not normalized to solar])',
         'hydrogen': 'log10 (H atoms) / cm**2',
         'hneutralssh': 'log10 (H in H1 and H_2) / cm**2',
         }
info = {'Temperature': 'ion balance for SF gas is calculated assuming'+\
                       ' a temperature of 10**4 K, but weighted' +\
                       ' temperatures do not include this adjustment',
        'hneutralssh': 'calculated using the Rahmati et al. (2013a) model' +\
                       ' assuming a Haardt & Madau (2001) UV/X-ray background',
        }

def mapname_file(samplename, kwargs):
    base = wdir + 'maps_{sample}_{ionW}_{quantityQ}.txt'
    return base.format(sample=samplename, **kwargs)

def histogram_file(samplename, ionW):
    base = wdir + 'histogram_{sample}_{ionW}.hdf5'
    return base

def getsample(size, logM200_Msun_min=12.0, logM200_Msun_max=12.5, seed=0):
    
    Mh_sels = [[('M200c_Msun', 10**logM200_Msun_min, 10**logM200_Msun_max)]]
    Mh_names =['geq{}_le{}'.format(logM200_Msun_min,logM200_Msun_max)]

    sel = sh.Galaxyselector(halocat, selections=Mh_sels, names=Mh_names,\
                         number=size, seed=seed)
    
    samplename = 'galaxies_logM200c-Msun-{mmin}-{mmax}_{num}_seed{seed}'
    samplename = samplename.format(mmin=logM200_Msun_min,
                                   mmax=logM200_Msun_max,
                                   num=size, seed=seed)    
    
    p3g.gensample(samplename=samplename, galaxyselector=sel)
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
                    outname = m3.make_map(*args, nameonly=True, **_kwargs)
                    
                    alreadyexists = False
                    if os.path.isfile(outname[0]) and os.path.isfile(outname[1]):
                        alreadyexists = True
                    if alreadyexists:
                        print('For galaxy {}, a map already exists; skipping'.format(gid))
                    else:
                        m3.make_map(*args, nameonly=False, **_kwargs)
                    
                    fdoc.write('{}\t{}\t{}\n'.format(gid, outname[0],
                                                     outname[1]))


def checksame(dct1, dct2, ignorekeys=None):
    if ignorekeys is None:
        ignorekeys = set()
    keys1 = set(dct1.keys()) - ignorekeys
    keys2 = set(dct2.keys()) - ignorekeys
    
    if keys1 != keys2:
        return False
    check = np.all([np.all(dct1[key] == dct2[key]) for key in keys1])
    return check

def read_map_and_attrs(mapfile, cosmopars_check=None, inputpars_check=None,
                       inputpars_ignore=None):
    with h5py.File(mapfile, 'r') as f:
        _map = f['map'][:]
        cosmopars = {key: val for key, val in \
                     f['Header/inputpars/cosmopars'].attrs.items()}
        inputpars = {key: val for key, val in \
                     f['Header/inputpars'].attrs.items()}
    
    if cosmopars_check is not None:
        if not checksame(cosmopars, cosmopars_check):
            msg = 'Cosmopars mismatch for file:\n\t{}'.format(mapfile)
            raise RuntimeError(msg)
    if inputpars_check is not None:
        if not checksame(inputpars, inputpars_check,
                         ignorekeys=inputpars_ignore):
            msg = 'Inputpars mismatch for file:\n\t{}'.format(mapfile)
            raise RuntimeError(msg)
            
    return _map, cosmopars, inputpars

def nicebins(values, delta):
    min_t = np.min(values[np.isfinite(values)])
    max_t = np.max(values[np.isfinite(values)])
    
    minbin = np.floor(min_t / delta) * delta
    maxbin = np.ceil(max_t / delta) * delta
    bins = np.arange(minbin, maxbin + 0.5 * delta, delta)
    return bins

def create_histogram(samplename, ionW, radius_R200c=1., binsize=0.2):
    dimensions = ['Temperature', 'Density', 'Metallicity', ionW]
    
    filesets = {}
    for kw in kwargs_l2:
        kw_map = {'ionW': ionW}
        kw_map.update(kw)
        filesets[kw['quantityQ']] = mapname_file(samplename, kw_map)
    
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
    
    
    galname_all = {key: pd.read_csv(filesets[key], header=0, sep='\t',
                                    index_col='galaxyid')\
                   for key in filesets}
    galaxyids = np.array(galdata_all.index)
    
    ignorematch = {'quantityQ'}
    first = True
    inputpars_ref = None
    ignorematch_global = {'L_x', 'L_y', 'L_z', 'centre', 'npix_x', 'npix_y',
                          'quantityQ'}

    Zbins = [-np.inf, -10., -8., -6., -5.] + \
            list(np.arange(-4., 1., binsize)) + [np.inf]
    Zbins = np.array(Zbins)
    bindct_fixed = {'Metallicity': Zbins}    
    
    print('starting loop over {} galaxies'.format(len(galaxyids)))
    for galid in galaxyids:
        print('Starting galaxy {}'.format(galid))
        
        Tvals, _cosmopars, inputpars = read_map_and_attrs(\
                                galname_all['Temperature'].at[galid, 'mapQ'],
                                cosmopars_check=cosmopars, 
                                inputpars_check=inputpars_ref,
                                inputpars_ignore=ignorematch_global)
            
        nvals, _cosmopars, inputpars = read_map_and_attrs(\
                                galname_all['Density'].at[galid, 'mapQ'],
                                cosmopars_check=cosmopars, 
                                inputpars_check=inputpars,
                                inputpars_ignore=ignorematch)
        
        Zvals, _cosmopars, inputpars = read_map_and_attrs(\
                                galname_all['Metallicity'].at[galid, 'mapQ'],
                                cosmopars_check=cosmopars, 
                                inputpars_check=inputpars,
                                inputpars_ignore=ignorematch)
        
        columns_T, _cosmopars, inputpars = read_map_and_attrs(\
                                galname_all['Temperature'].at[galid, 'mapW'],
                                cosmopars_check=cosmopars, 
                                inputpars_check=None,
                                inputpars_ignore=ignorematch)
        
        columns_n, _cosmopars, inputpars = read_map_and_attrs(\
                                galname_all['Density'].at[galid, 'mapW'],
                                cosmopars_check=cosmopars, 
                                inputpars_check=None,
                                inputpars_ignore=ignorematch)
        
        columns_Z, _cosmopars, inputpars = read_map_and_attrs(\
                                galname_all['Metallicity'].at[galid, 'mapW'],
                                cosmopars_check=cosmopars, 
                                inputpars_check=None,
                                inputpars_ignore=ignorematch)
        
        if not np.allclose(10**columns_T, 10**columns_Z) and \
            np.allclose(10**columns_T, 10**columns_n):
            msg = 'Mismatch in column maps for galaxy {}'.format(galid)
            raise RuntimeError(msg)
        
        imgcen = inputpars['centre']
        axis = inputpars['axis'].decode()
        if axis == 'z':
            imgcx = imgcen[0]
            imgcy = imgcen[1]
            imglx = inputpars['L_x']
            imgly = inputpars['L_y']
            
            galcx = galdata_all.at[galid, 'Xcom_cMpc']
            galcy = galdata_all.at[galid, 'Ycom_cMpc']
        elif axis == 'x':
            imgcx = imgcen[1]
            imgcy = imgcen[2]
            imglx = inputpars['L_y']
            imgly = inputpars['L_z']
            
            galcx = galdata_all.at[galid, 'Ycom_cMpc']
            galcy = galdata_all.at[galid, 'Zcom_cMpc']
        elif axis == 'y':
            imgcx = imgcen[2]
            imgcy = imgcen[0]
            imglx = inputpars['L_z']
            imgly = inputpars['L_x']
            
            galcx = galdata_all.at[galid, 'Zcom_cMpc']
            galcy = galdata_all.at[galid, 'Xcom_cMpc']
        
        radius_cMpc = radius_R200c * galdata_all.at[galid, 'R200c_cMpc']
        imgdx = imglx / float(inputpars['npix_x'])
        imgdy = imgly / float(inputpars['npix_y'])
        
        grid = np.indices(Tvals.shape) + 0.5
        rsq = (grid[0] * imgdx - 0.5 * imglx + imgcx - galcx)**2 + \
              (grid[1] * imgdy - 0.5 * imgly + imgcy - galcy)**2
        mask = rsq <= radius_cMpc**2
        
        print('Mask covering fraction: {}'.format(float(np.sum(mask)) \
                                                / float(np.prod(mask.shape))))
        #plt.figure()        
        #plt.imshow(mask.T, origin='lower', interpolation='nearest')
        #plt.colorbar()
        
        _Tvals = Tvals[mask]
        _nvals = nvals[mask]
        _Zvals = Zvals[mask]
        _columns = columns_T[mask]
        
        valdct = {'Temperature': _Tvals,
                  'Density': _nvals,
                  'Metallicity': _Zvals,
                  ionW: _columns}
        
        edges_temp = {key: nicebins(valdct[key], binsize) \
                      for key in dimensions}
        print(edges_temp)
        edges_temp.update(bindct_fixed)
        edges_temp = [edges_temp[key] for key in dimensions]
        values = [valdct[key] for key in dimensions]
        print(values)
        temp, _edges = np.histogramdd(values, edges_temp)
        
        if first:
            hist = temp
            edges = edges_temp
            inputpars_ref = inputpars
        else:
            hist, edges = p3g.combine_hists(hist, temp, edges, edges_temp,
                                            rtol=1e-5, atol=1e-8, add=True)
        first = False
    
    histfile = histogram_file(samplename, ionW)
    
    print('Starting save')
    with h5py.File(histfile, 'a') as fo:
        ds = fo.create_dataset('histogram', data=hist)
        _info = 'histogram of {ionW} column densities and their weighted ' +\
             'temperature, density, and metallicity at impact parameters ' +\
             ' <= {radius} times R200c from central galaxies. Zero column ' +\
             'densities are not counted.'
        _info = _info.format(ionW=ionW, radius=radius_R200c)
        ds.attrs.create('info', np.string_(_info))
        
        hed = fo.create_group('Header')
        csm = hed.create_group('cosmopars')
        for key in cosmopars:
            csm.attrs.create(key, cosmopars[key])
        inp = hed.create_group('inputpars')
        inkeys = set(inputpars.keys()) - ignorematch_global
        for key in inkeys:
            inp.attrs.create(key, inp[key])
        hed.create_dataset('galaxyids', data=galaxyids)
        
        axg = fo.create_group('histogram_axes')
        for axis, dim, ed in enumerate(zip(dimensions, edges)):
            ds = axg.create_dataset(dim, data=ed)
            ds.attrs.create('units', np.string_(units[dim]))
            if dim in info:
                ds.attrs.create('info', np.string_(info[dim]))
            ds.attrs.create('histogram_axis', axis)
            
            
    
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
        
    samplename = getsample(size, logM200_Msun_min=12.0, logM200_Msun_max=12.5,
                           seed=seed)
    
    create_maps(samplename)
    create_histogram(samplename, 'hydrogen')
    create_histogram(samplename, 'hneutralssh')

    

