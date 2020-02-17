#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:01:14 2020

@author: wijers

look for (a)typical environments of absorbers like Ton180 O VII/VIII
"""

import os
import numpy as np
import h5py

import make_maps_opts_locs as ol
import eagle_constants_and_units as cu
import cosmo_utils as c

# put stored files here
ddir = '/net/quasar/data2/wijers/slcat/'
mdir = '/net/luttero/data2/jussi_ton180_data/'

## measured column densities: logN / cm^-2 
## sigma: single +- value or (-, +) tuple
## note: O VII in the slab model is a low-significane detection
# UV priors
o6_uv = 13.76
sigma_o6_uv = 0.08

# CIE model (X-ray)
o6_cie = 13.9
sigma_o6_cie = 0.2
o7_cie = 16.4
sigma_o7_cie = 0.2
o8_cie = 16.0
sigma_o8_cie = 0.2

# slab model (O VI prior z)
o7_slab_zuv = 16.59
sigma_o7_slab_zuv = (0.28, 0.24)
o8_slab_zuv = 15.85
sigma_o8_slab_zuv = (0.65, 0.32)

# slab model (O VII z)
o7_slab_zx = 16.62
sigma_o7_slab_zx = (0.28, 0.25)
o8_slab_zx = 15.82
sigma_o8_slab_zx = (0.78, 0.49)

# redshifts: comparison to z=0.0 or z=0.1 snapshots are both fine
zuv = 0.04582
zx  = 0.0459

## data dictionary: 
# UV-priors: 'uvp', 
# CIE model (same at both z): 'cie'
# slab model (UV z): 'so6'
# slab model (X-ray z): 'so7'
# then ion: (logN, sigma+, sigma-) tuples
meas = {'uvp': {'o6': (o6_uv,) + (sigma_o6_uv,) * 2,\
                },\
        'cie': {'o6': (o6_cie,) + (sigma_o6_cie,) * 2,\
                'o7': (o7_cie,) + (sigma_o7_cie,) * 2,\
                'o8': (o8_cie,) + (sigma_o8_cie,) * 2,\
                },\
        'so6': {'o7': (o7_slab_zuv,) + sigma_o7_slab_zuv,\
                'o8': (o8_slab_zuv,) + sigma_o8_slab_zuv,\
                },\
        'so7': {'o7': (o7_slab_zx,) + sigma_o7_slab_zx,\
                'o8': (o8_slab_zx,) + sigma_o8_slab_zx,\
                } ,\
        }

detlim = {'o7': 15.5}

cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}

def integratehist(counts, edges, vmin, vmax):
    '''
    integrate a histogram between vmin and vmax, assuming a uniform 
    distribution of counts within each bin in whatever units edges, vmin, and 
    vmax are in
    '''
    ## side = 'left': a[i-1] < v <= a[i]; v is seached values, a input array, i index returned
    if vmin > vmax:
        raise ValueError('function only works for vmin <= vmax')
    if not np.all(np.diff(edges) > 0.):
        raise ValueError('function only works for edges in increasing order')
    imin, imax = np.searchsorted(edges, [vmin, vmax], side='left')
    if imax == imin:
        maxadd = counts[imin - 1] * (vmax - vmin) \
                                  / (edges[imin] - edges[imin - 1])
        return maxadd
    if vmin in edges or vmin < edges[0]: #imin = index where they are equal
        minadd = 0
    else:
        minadd = counts[imin - 1] * (edges[imin] - vmin) \
                                  / (edges[imin] - edges[imin - 1])
    if vmax in edges or vmax > edges[-1]: #imin = index where they are equal
        maxadd = 0
    else:
        maxadd = counts[imax - 1] * (vmax - edges[imax - 1]) \
                                  / (edges[imax] - edges[imax - 1])
        imax -= 1 # go up to 1 bin before the one just interpolated
    return np.sum(counts[imin : imax]) + minadd + maxadd
    

# 
def countsl(modelname, sigma_min=1, sigma_max=np.inf):
    '''
    from the z=0.1 CDDFs, count the number of sightlines between sigma_min and
    sigma_max for each ion (linear interpolation within bin sizes)
    '''
    
    
    if modelname == 'detlim':
        ions = list(detlim.keys())
        ionranges = {ion: (detlim[ion], np.inf) for ion in ions}
    else:
        ions = list(meas[modelname].keys())
        ionranges = {ion: (meas[modelname][ion][0] - meas[modelname][ion][1] * sigma_min,\
                           meas[modelname][ion][0] + meas[modelname][ion][2] * sigma_max)\
                 for ion in ions}
    
    ion_filedct_excl_1R200c_cenpos = {#'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    masknames1 = ['nomask']
    masknames = masknames1 #{0: {ion: masknames1 for ion in ions}}
        
    hists = {}
    cosmopars = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                #if _bins[0] == -np.inf:
                #    _bins[0] = -100.
                #if _bins[-1] == np.inf:
                #    _bins[-1] = 100.
                bins[var][ion] = _bins
                
                # extract number of pixels from the input filename, using naming system of make_maps
                inname = np.array(fi['input_filenames'])[0].decode()
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                #numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                #numpix_1sl.remove(None)
                #numpix_1sl = int(list(numpix_1sl)[0][:-3])
                #print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                ionind = 1 + np.where(np.array([part == 'coldens' for part in parts]))[0][0]
                ion = parts[ionind]
                
                masks = masknames
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                
                examplemaskdir = list(fi['masks'].keys())[0]
                examplemask = list(fi['masks/%s'%(examplemaskdir)].keys())[0]
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                #dXtot[var][ion] = mc.getdX(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                #dztot[var][ion] = mc.getdz(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                #dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])

        #assert checksubdct_equal(cosmopars[var])
                     
    ions = sorted(ions)
    var = 0
    out = {}
    for ion in ions:
        counts = hists[var][ion]['nomask']
        edges = bins[var][ion]
        nmin = ionranges[ion][0]
        nmax = ionranges[ion][1]
        tot = integratehist(counts, edges, nmin, nmax)
        out[ion] = tot
    return out        
        

def create_sl_cat():
    '''
    switch from using maps of all sightlines to location, value catalogues
    even the least limitng selections (UV O VI minus 1 sigma) use less than
    1 % of all sightlines, so this should reduce processing time
    
    catalogue includes anything that meets at least one ion's 
    measured minus 1 sigma column density threshold 
    '''
    
    # old npz format... oh well
    mapkey = 'arr_0'
    zcens = (np.arange(16) + 0.5) / 16. * 100.
    npix  = 32000
    snapshot = 27
    simnum = 'L0100N1504'
    var    = 'REFERENCE'
    cosmopars = cosmopars_ea_27
    mapfiles = {'o6': ol.ndir + 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen{zcen}_z-projection_T4EOS.npz',\
                'o7': ol.ndir + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen{zcen}_z-projection_T4EOS.npz',\
                'o8': ol.ndir + 'coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen{zcen}_z-projection_T4EOS.npz',\
                }
    
    # select on minimum column density:
    ions = ['o6', 'o7', 'o8']
    minvals = {ion: np.inf for ion in ions}
    for model in meas.keys():
        for ion in meas[model].keys():
            minval = meas[model][ion][0] - meas[model][ion][1]
            minvals[ion] = min(minval, minvals[ion])
    minvals['o7'] = min(minvals['o7'], detlim['o7'])
    print('selecting: {}'.format(minvals))    
    
    outname = ddir + 'sightlinecat_z-{z:.1f}_selection1.hdf5'.format(**cosmopars)
    xpos_pmpc = np.array([])
    ypos_pmpc = np.array([])
    zpos_pmpc = np.array([])
    ionvals = {ion: np.array([]) for ion in ions}
    
    if os.path.isfile(outname):
        raise RuntimeError('Output file {} already exists'.format(outname))
    with h5py.File(outname, 'w') as fo:
        # record Header data
        hed = fo.create_group('Header')
        csm = hed.create_group('cosmopars')
        for key in cosmopars:
            csm.attrs.create(key, cosmopars[key])
        hed.attrs.create('Axis3', 2)
        hed.attrs.create('simnum', np.string_(simnum))
        hed.attrs.create('snapshot', snapshot)
        hed.attrs.create('var', np.string_(var))
        hed.create_dataset('zvals_cMpc', data=zcens)
        
        fgp = hed.create_group('mapfiles')
        for ion in ions:
            fgp.create_group(ion)
        
        sgp = hed.create_group('selection')
        sgp.attrs.create('info', np.string_('Minimum column density [log10 cm^-2] for each ion. Selection is based on meeting any of these minima.'))
        for ion in ions:
            sgp.attrs.create(ion, minvals[ion])
        
        # loop over files to get the sightline selections, 
        # append N, pos to arrays
        for zcen in zcens:
            _files = {ion: mapfiles[ion].format(zcen=zcen) for ion in ions}
            for ion in ions:
                fgp[ion].attrs.create(str(zcen), _files[ion])
            _maps = {ion: np.load(_files[ion])[mapkey] for ion in ions}
            # create boolean mask 
            select = np.zeros((npix,) * 2, dtype=bool)
            for ion in ions:
                select |= _maps[ion] >= minvals[ion]
            # also needed for easy x/y position calc
            select = np.where(select)
            _ionvals = {ion: _maps[ion][select] for ion in ions}
            
            # positions (pixel and slice centres)
            indtopos = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] / npix
            _xpos_pmpc = select[0] + 0.5 
            _xpos_pmpc *= indtopos
            _ypos_pmpc = select[1] + 0.5 
            _ypos_pmpc *= indtopos
            _zpos_pmpc = np.ones(len(xpos_pmpc), dtype=np.float) * zcen * cosmopars['a']
            
            xpos_pmpc = np.append(xpos_pmpc, _xpos_pmpc)
            ypos_pmpc = np.append(ypos_pmpc, _ypos_pmpc)
            zpos_pmpc = np.append(zpos_pmpc, _zpos_pmpc)
            ionvals = {ion: np.append(ionvals[ion], _ionvals[ion]) for ion in ions}
            
        # store sightline arrays      
        fo.create_dataset('xpos_pmpc', data=xpos_pmpc)
        fo.create_dataset('ypos_pmpc', data=ypos_pmpc)
        fo.create_dataset('zpos_pmpc', data=zpos_pmpc)
        for ion in ions:
            dname = 'coldens_{}'.format(ion)
            fo.create_dataset(dname, data=zpos_pmpc)
            fo[dname].attrs.create('units', np.string_('log10 cm^-2'))