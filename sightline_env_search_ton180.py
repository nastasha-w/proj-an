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
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import make_maps_opts_locs as ol
import eagle_constants_and_units as c
import cosmo_utils as cu
import ion_line_data as ild

# put stored files here
ddir = '/net/quasar/data2/wijers/slcat/'
mdir = '/net/luttero/data2/jussi_ton180_data/'

n_jobs = int(os.environ['OMP_NUM_THREADS']) # not shared-memory, I think, but a good indicator

## measured column densities: logN / cm^-2 
## sigma: single +- value or (-, +) tuple
## note: O VII in the slab model is a low-significane detection
### old model measurements
## UV priors
#o6_uv = 13.76
#sigma_o6_uv = 0.08
#
## CIE model (X-ray)
#o6_cie = 13.9
#sigma_o6_cie = 0.2
#o7_cie = 16.4
#sigma_o7_cie = 0.2
#o8_cie = 16.0
#sigma_o8_cie = 0.2
#
## slab model (O VI prior z)
#o7_slab_zuv = 16.59
#sigma_o7_slab_zuv = (0.28, 0.24)
#o8_slab_zuv = 15.85
#sigma_o8_slab_zuv = (0.65, 0.32)
#
## slab model (O VII z)
#o7_slab_zx = 16.62
#sigma_o7_slab_zx = (0.28, 0.25)
#o8_slab_zx = 15.82
#sigma_o8_slab_zx = (0.78, 0.49)
#
## redshifts: comparison to z=0.0 or z=0.1 snapshots are both fine
#zuv = 0.04582
#zx  = 0.0459

### updated model 2020-02-25 (received)
# UV priors
o6_uv = 13.76
sigma_o6_uv = 0.08

# CIE model (X-ray, X-ray redshift)
o6_cie_zx = 13.9
sigma_o6_cie_zx = 0.2
o7_cie_zx = 16.4
sigma_o7_cie_zx = 0.2
o8_cie_zx = 16.0
sigma_o8_cie_zx = 0.2

# CIE model (X-ray, UV redshift)
o6_cie_zuv = 13.8
sigma_o6_cie_zuv = 0.2
o7_cie_zuv = 16.4
sigma_o7_cie_zuv = 0.2
o8_cie_zuv = 15.9
sigma_o8_cie_zuv = 0.2

# slab model (O VI prior z)
o7_slab_zuv = 16.52
sigma_o7_slab_zuv = (0.28, 0.25)
o8_slab_zuv = 15.7
sigma_o8_slab_zuv = (1.1, 0.4)

# slab model (O VII z)
o7_slab_zx = 16.69
sigma_o7_slab_zx = (0.39, 0.37)
o8_slab_zx = 15.7
sigma_o8_slab_zx = (1.5, 0.4)
#
## redshifts: comparison to z=0.0 or z=0.1 snapshots are both fine
zuv = 0.04582
zx  = 0.0455

## data dictionary: 
# UV-priors: 'uvp', 
# CIE model (same at both z): 'cie'
# slab model (UV z): 'so6'
# slab model (X-ray z): 'so7'
# then ion: (logN, sigma+, sigma-) tuples
meas = {'uvp': {'o6': (o6_uv,) + (sigma_o6_uv,) * 2,\
                },\
        'co7': {'o6': (o6_cie_zx,) + (sigma_o6_cie_zx,) * 2,\
                'o7': (o7_cie_zx,) + (sigma_o7_cie_zx,) * 2,\
                'o8': (o8_cie_zx,) + (sigma_o8_cie_zx,) * 2,\
                },\
        'co6': {'o6': (o6_cie_zuv,) + (sigma_o6_cie_zuv,) * 2,\
                'o7': (o7_cie_zuv,) + (sigma_o7_cie_zuv,) * 2,\
                'o8': (o8_cie_zuv,) + (sigma_o8_cie_zuv,) * 2,\
                },\
        'so6': {'o7': (o7_slab_zuv,) + sigma_o7_slab_zuv,\
                'o8': (o8_slab_zuv,) + sigma_o8_slab_zuv,\
                },\
        'so7': {'o7': (o7_slab_zx,) + sigma_o7_slab_zx,\
                'o8': (o8_slab_zx,) + sigma_o8_slab_zx,\
                } ,\
        }

detlim = {'o7': 15.5}

# z: redshift, r: impact parameter (pMpc), mstar: log10 stellar mass [Msun]
galdata_meas = [{'z': 0.04562, 'r': 0.295, 'mstar': ( 9.5,  0.5, 0.5)},\
                {'z': 0.04596, 'r': 0.620, 'mstar': (10.1,  0.5, 0.5)},\
                {'z': 0.04608, 'r': 1.500, 'mstar': (10.95, 0.2, 0.2)},\
                ]

cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}

# galaxies seem to be sort of ok down to this mass in Schaye et al. (2015)
galsel_default = [('Mstar_Msun', 10**9, None)]
halocat_default = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30_inclsatellites.hdf5'

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
    

def periodic_duplicate(propdct, targetname, period, margin):
    '''
    for a dictionary containing matching arrays, append copies with shifted 
    array targetname by period for all targetname values within margin of 
    0 and period. Values are assumed to be between 0 and period 
    '''
    dupmin = propdct[targetname] < margin
    dupmax = propdct[targetname] > period - margin
    origlen = len(propdct[targetname])
    
    for key in propdct:
        if key == targetname:
            propdct[key] = np.append(propdct[key], propdct[key][dupmin] + period)
            propdct[key] = np.append(propdct[key], propdct[key][:origlen][dupmax] - period)
        else:
            propdct[key] = np.append(propdct[key], propdct[key][dupmin])
            propdct[key] = np.append(propdct[key], propdct[key][:origlen][dupmax])
    
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
        
def create_test_sets(testnum=1):
    outname_slc = mdir + 'slcat_test_{num}.hdf5'.format(num=testnum)
    outname_hlc = mdir + 'halocat_test_{num}.hdf5'.format(num=testnum)
    
    if testnum == 1:
        simnum = 'L0100N1504'
        axis = 2
        snapshot = 27
        apsize = 30
        
        cosmopars = cosmopars_ea_27
        boxsize_cmpc = cosmopars['boxsize'] / cosmopars['h']
        boxsize_pmpc = boxsize_cmpc * cosmopars['a']
        
        ## sightline catalogue
        numabs = 20
        numsl  = 4
        ions = ['o6', 'o7', 'o8']
        mincol = 13.
        maxcol = 16.
        
        # imitate slice iteration: z inidices are low to high
        zi = np.random.randint(numsl, size=numabs)
        zi = np.sort(zi)
        zvals_cmpc = (np.arange(numsl) + 0.5) * boxsize_cmpc / float(numsl)
        coldens = {ion: np.random.uniform(mincol, maxcol, numabs) for ion in ions}
        pos     = {'{}pos_pmpc'.format(axis): np.random.uniform(0., boxsize_pmpc, numabs) for axis in ['x', 'y']}
        pos.update({'zpos_pmpc': (zvals_cmpc * cosmopars['a'])[zi]})
        
        selection = {ion: mincol for ion in ions}
        
        ## halo catalogue (larger set)
        # halo, stellar masses and R200c should be at least sort of consistent,
        # for central galaxies
        # subgroupnumbers are not, really
        numgal = 1000
        Mhmin_Msun = 10**10
        M200c_Msun = 10**np.random.uniform(np.log10(Mhmin_Msun), 15., numgal)
        R200c_pkpc = cu.R200c_pkpc(M200c_Msun, cosmopars)
        Mstar_Msun = 0.01 * M200c_Msun * 10**np.random.uniform(-0.2, 0.2)
        Xcom_cMpc  = np.random.uniform(0., boxsize_cmpc, numgal)
        Ycom_cMpc  = np.random.uniform(0., boxsize_cmpc, numgal)
        Zcom_cMpc  = np.random.uniform(0., boxsize_cmpc, numgal)
        galaxyid   = np.arange(numgal)
        # generate subgroup numbers
        satinds = np.unique(np.random.randint(numgal, size=numgal // 2))
        SubGroupNumber = np.zeros(numgal, dtype=np.int)
        SubGroupNumber[satinds] = 1
        SubGroupNumber[M200c_Msun > 10**13] = 0
        
        halodct = {'M200c_Msun': M200c_Msun,\
                   'R200c_pkpc': R200c_pkpc,\
                   'Mstar_Msun': Mstar_Msun,\
                   'Xcom_cMpc':  Xcom_cMpc,\
                   'Ycom_cMpc':  Ycom_cMpc,\
                   'Zcom_cMpc':  Zcom_cMpc,\
                   'SubGroupNumber': SubGroupNumber,\
                   'galaxyid': galaxyid,\
                   }
    else:
        raise ValueError('No options are set up for test {num}'.format(num=testnum))
    
    # store mock sightlines in a format like the real ones
    with h5py.File(outname_slc, 'w') as fs:
        hed = fs.create_group('Header')
        # record the simulation we're imitating, but using a dummy var
        hed.attrs.create('Axis3', axis)
        hed.attrs.create('simnum', np.string_(simnum))
        hed.attrs.create('snapshot', snapshot)
        hed.attrs.create('var', np.string_('dummy'))
        csm = hed.create_group('cosmopars')
        for key in cosmopars:
            csm.attrs.create(key, cosmopars[key])
        sgp = hed.create_group('selection')
        fgp = hed.create_group('mapfiles')
        for ion in ions:
            sgp.attrs.create(ion, selection[ion])
            sfgp = fgp.create_group(ion)
            for zval in zvals_cmpc:
                sfgp.attrs.create(str(zval), np.string_('None'))
        hed.create_dataset('zvals_cMpc', data=zvals_cmpc)
        
        for ion in coldens:
            fs.create_dataset('coldens_{ion}'.format(ion=ion), data=coldens[ion])
        for key in pos:
            fs.create_dataset(key, data=pos[key])
    
    with h5py.File(outname_hlc, 'w') as fh:
        hed = fh.create_group('Header')
        # record the simulation we're imitating, but using a dummy var
        hed.attrs.create('simnum', np.string_(simnum))
        hed.attrs.create('snapnum', snapshot)
        hed.attrs.create('var', np.string_('dummy'))
        hed.attrs.create('Mhalo_min_Msun', Mhmin_Msun)
        hed.attrs.create('subhalo_aperture_size_Mstar_Mbh_SFR_pkpc', apsize)
        csm = hed.create_group('cosmopars')
        for key in cosmopars:
            csm.attrs.create(key, cosmopars[key])
        for key in halodct.keys():
            fh.create_dataset(key, data=halodct[key])
        
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
    
    # selection1: used older measured values (less sightlines, too)
    outname = ddir + 'sightlinecat_z-{z:.1f}_selection2.hdf5'.format(**cosmopars)
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
        for zcen in np.sort(zcens):
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
            _zpos_pmpc = np.ones(len(_xpos_pmpc), dtype=np.float) * zcen * cosmopars['a']
            
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
            fo.create_dataset(dname, data=ionvals[ion])
            fo[dname].attrs.create('units', np.string_('log10 cm^-2'))
            

def find_galenv(slcat, halocat=halocat_default,\
                galsel='def', nngb=3, nsl=0.5, dist3d=False, catsel=None):
    '''
    given absorbers in slcat, halos/galaxies in halocat, and the halocat 
    selection galsel, find nearest neighbors for the absorbers
    
    galsel: list of array, min/None, max/None tuples
            array should be the name in the halo catalogue halocat, min/max
            are the allowed values min <= selected < max
    nngb:   nearest neighbor number to look for
    nsl:    margin (units: slice thickness) in which to 
            consider galaxies for a match. 
            0.5 -> only in the same slice
    dist3d: use the 3D distance (assuming zero peculiar velocities and 
            absorbers at slice centers) to find nearest neighbors
    catsel: part of the slcat to use (concatentate resulting files afterwards)
            (index, total) tuple of integers: 
            total is the number of parts to split into, index is which one 
            this is
            index runs from 0 to total - 1
    '''
    
    # load sightline data
    if catsel is None:
        catsel = (0, 1)
    with h5py.File(slcat, 'r') as fs:
        cosmopars = {key: val for key, val in fs['Header/cosmopars'].attrs.items()}
        ions = list(fs['Header/mapfiles'].keys())
        zcens = np.array(fs['Header/zvals_cMpc'])
        zcens = np.sort(zcens)
        zcens *= cosmopars['a']
        slice_pMpc = np.average(np.diff(zcens))
        zedges = zcens + 0.5 * slice_pMpc
        zedges = np.append([0.], zedges)
        boxperiod = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a']
        
        numabs = fs['xpos_pmpc'].shape[0]
        numperproc = (numabs - 1) // catsel[1] + 1
        procnum = catsel[0]
        _sel = slice(procnum * numperproc, (procnum + 1) * numperproc)
        
        xpos = np.array(fs['xpos_pmpc'])[_sel]
        ypos = np.array(fs['ypos_pmpc'])[_sel]
        zpos = np.array(fs['zpos_pmpc'])[_sel]
        coldens = {ion: np.array(fs['coldens_{ion}'.format(ion=ion)])[_sel] for ion in ions}
        
        absdct = {'xpos_pmpc': xpos,\
                  'ypos_pmpc': ypos,\
                  'zpos_pmpc': zpos,\
                  }
        absdct.update(coldens) 
        
        
    ## use the fact that the values are sorted by z -> get blocks of the arrays
    ## to loop over for galxy cross-matching
    # side = 'left': a[i-1] < v <= a[i]; v is seached values, a input array, i index returned
    zilims = np.searchsorted(zpos, zedges, side='right')
    zsels = [slice(zilims[i], zilims[i + 1]) for i in range(len(zcens))]
    zblocks = [(zedges[ind], zedges[ind + 1], zsels[ind]) for ind in range(len(zcens))]
    
    with h5py.File(halocat, 'r') as fh:
        # get the properties desired for the nearest neighbors, and the 
        # positions for distances
        M200c_Msun = np.array(fh['M200c_Msun'])
        R200c_pkpc = np.array(fh['R200c_pkpc'])
        Mstar_Msun = np.array(fh['Mstar_Msun'])
        Xcom_pMpc  = np.array(fh['Xcom_cMpc']) * cosmopars['a']
        Ycom_pMpc  = np.array(fh['Ycom_cMpc']) * cosmopars['a']
        Zcom_pMpc  = np.array(fh['Zcom_cMpc']) * cosmopars['a']
        SubGroupNumber = np.array(fh['SubGroupNumber'])
        galaxyid   = np.array(fh['galaxyid'])
        
        halodct = {'M200c_Msun': M200c_Msun,\
                   'R200c_pkpc': R200c_pkpc,\
                   'Mstar_Msun': Mstar_Msun,\
                   'Xcom_pMpc':  Xcom_pMpc,\
                   'Ycom_pMpc':  Ycom_pMpc,\
                   'Zcom_pMpc':  Zcom_pMpc,\
                   'SubGroupNumber': SubGroupNumber,\
                   'galaxyid': galaxyid,\
                   }
        
        # make and apply halo/galaxy property selection
        if galsel is None:
            gsel = slice(None, None, None)
        else:
            gsel = np.ones(len(halodct['galaxyid']), dtype=bool)
            if galsel == 'def':
                galsel = galsel_default
            for tup in galsel:
                if tup[0] in halodct:
                    selarr = halodct[tup[0]]
                else:
                    selarr = np.array(fh[tup[0]])
                if tup[1] is not None:
                    gsel &= selarr >= tup[1]
                if tup[2] is not None:
                    gsel &= selarr < tup[2]
                
        halodct = {key: halodct[key][gsel] for key in halodct}
    # metric can be a function that computs the distance between two input
    # vectors. I'd imagine this might be pretty slow, though, if it's 
    # looping...
    # alt: just duplicate galaxies and check that distances < duplication 
    # range? probably better for a fast solution.    
    duplication_range_pmpc = 15
    periodic_duplicate(halodct, 'Xcom_pMpc', boxperiod, duplication_range_pmpc)
    periodic_duplicate(halodct, 'Ycom_pMpc', boxperiod, duplication_range_pmpc)
    if dist3d: # if only matching slices, deal with periodicity explicitly there
        periodic_duplicate(halodct, 'Zcom_pMpc', boxperiod, duplication_range_pmpc)
    
    neighpropdct = {key: np.empty(shape=(0, nngb)) for key in halodct}
    neighpropdct.update({'neighbor_dist_pmpc': np.empty(shape=(0, nngb))})
    neighpropdct.update({'galaxyid': np.empty(shape=(0, nngb), dtype=np.int),\
                         'SubGroupNumber': np.empty(shape=(0, nngb), dtype=np.int),\
                         })
    zselmax_last = 0
    for bi in range(len(zblocks)): # blocks divide up the absorbers into slice subgroups
        zmin_abs, zmax_abs, zsel_abs = zblocks[bi] #  zmin, zmax in pMpc
        zcen = 0.5 * (zmin_abs + zmax_abs)
        zmin_gal = zcen - nsl * slice_pMpc
        zmax_gal = zcen + nsl * slice_pMpc
        if not zsel_abs.start == zselmax_last:
            print('current slice: ({start}, {stop}, {step}); last stop: {last}'.format(start=zsel_abs.start, stop=zsel_abs.stop, step=zsel_abs.step, last=zselmax_last))
            raise RuntimeError('The order of the absorber slices is mixed up; neighbor arrays will not match absorber arrays')
        
        zgals = halodct['Zcom_pMpc']
        if zmin_gal >= 0 and zmax_gal <= boxperiod:
            galsel_b = np.logical_and(zmin_gal <= zgals, zmax_gal > zgals)
        elif zmin_gal < 0 and zmax_gal <= boxperiod:
            galsel_b = np.logical_or(zmax_gal > zgals, zmin_gal + boxperiod <= zgals)
        elif zmin_gal >= 0 and zmax_gal > boxperiod:
            galsel_b = np.logical_or(zmax_gal - boxperiod > zgals, zmin_gal <= zgals)
        else: # max and min both wrapped around
            galsel_b = slice(None, None, None)
        
        halodct_b = {key: halodct[key][galsel_b] for key in halodct}
        absdct_b = {key: absdct[key][zsel_abs] for key in absdct} 
        
        if dist3d:
            halopos = np.array([halodct_b['Xcom_pMpc'],\
                                halodct_b['Ycom_pMpc'],\
                                halodct_b['Zcom_pMpc']]).T
            abspos = np.array([absdct_b['xpos_pmpc'],\
                               absdct_b['ypos_pmpc'],\
                               absdct_b['zpos_pmpc']]).T
        else:
            halopos = np.array([halodct_b['Xcom_pMpc'],\
                                halodct_b['Ycom_pMpc']]).T
            abspos = np.array([absdct_b['xpos_pmpc'],\
                               absdct_b['ypos_pmpc']]).T
    
        nnfinder = NearestNeighbors(n_neighbors=nngb, metric='euclidean',\
                                    n_jobs=n_jobs)    
        nnfinder.fit(halopos)
        ## second loop for very large datasets:
        maxlen = int(1e7)
        if len(abspos) == 0: # only arises with the test set + subset runs
            continue
        elif len(abspos) <= maxlen:   
            neigh_dist, neigh_ind = nnfinder.kneighbors(X=abspos,\
                                                        n_neighbors=nngb,\
                                                        return_distance=True)
            #print(neigh_dist)
            #print(neigh_ind)
            neighprops_this = {key: halodct_b[key][neigh_ind] for key in halodct_b}
            neighprops_this.update({'neighbor_dist_pmpc': neigh_dist})
        else:
            numiter = (len(abspos) - 1) // maxlen + 1
            #print(numiter)
            neighprops_this = {key: np.empty(shape=(0, nngb)) for key in halodct}
            neighprops_this.update({'neighbor_dist_pmpc': np.empty(shape=(0, nngb))})
            neighprops_this.update({'galaxyid': np.empty(shape=(0, nngb), dtype=np.int),\
                                    'SubGroupNumber': np.empty(shape=(0, nngb), dtype=np.int),\
                                   })
            for ind in range(numiter):
                subsel = slice(ind * maxlen, (ind + 1) * maxlen)
                neigh_dist, neigh_ind = nnfinder.kneighbors(X=abspos[subsel],\
                                                        n_neighbors=nngb,\
                                                        return_distance=True)
                neighprops_thissub = {key: halodct_b[key][neigh_ind] for key in halodct_b}
                neighprops_thissub.update({'neighbor_dist_pmpc': neigh_dist})
                
                for key in neighprops_this:
                    neighprops_this[key] = np.append(neighprops_this[key], neighprops_thissub[key], axis=0)
            
        del nnfinder
        
        for key in neighpropdct:
            neighpropdct[key] = np.append(neighpropdct[key], neighprops_this[key], axis=0)
        # to make sure the order of the absorber arrays is maintained
        zselmax_last = zsel_abs.stop
        
    ## save the data
    outfile = slcat.split('/')[-1]
    outfile = '.'.join(outfile.split('.')[:-1])
    if catsel == (0, 1):
        partstr = ''
    else:
        partstr = '_{ind}-of-{tot}'.format(ind=catsel[0], tot=catsel[1])
    outfile = ddir + outfile + '_nearest-neighbor-match_nngb-{nngb}_nsl-{nsl}{pst}.hdf5'.format(nngb=nngb, nsl=nsl, pst=partstr)
    
    with h5py.File(outfile, 'a') as fo:
        prev = list(fo.keys())
        if 'Header' not in prev:
            hed = fo.create_group('Header')
            csm = hed.create_group('cosmopars')
            for key in cosmopars:
                csm.attrs.create(key, cosmopars[key])
            hed.attrs.create('sightline_catalogue', np.string_(slcat))
            
            agp = fo.create_group('absorbers')
            for key in absdct:
                agp.create_dataset(key, data=absdct[key])
            agp.attrs.create('info', np.string_('column density [log10 cm^-2]'))
        else:
            prev.remove('Header')
            prev.remove('absorbers')
            
        # store input parameters 
        numthis = len(prev)
        grp = fo.create_group('match_{num}'.format(num=numthis))
        grp.attrs.create('halo_catalogue', np.string_(halocat))
        grp.attrs.create('number_of_neighbours', nngb)
        grp.attrs.create('number_of_slices_for_match', nsl)
        grp.attrs.create('use_3D_distance', dist3d)
        ggrp = grp.create_group('galaxy_selection')
        for ti in range(len(galsel)):
            tup = galsel[ti]
            sggrp = ggrp.create_group('tuple_{}'.format(ti))
            sggrp.attrs.create('array', np.string_(tup[0]))
            _minv = tup[1] if tup[1] is not None else np.string_('None')
            _maxv = tup[2] if tup[2] is not None else np.string_('None')
            sggrp.attrs.create('min', _minv)
            sggrp.attrs.create('max', _maxv)
        
        for key in neighpropdct:
            grp.create_dataset(key, data=neighpropdct[key])
            
def combine_galenv_subsets(examplename, matchnum=0):
    '''
    parses the given example name and derives the full set of names
    then combines those into a galenv file like would be obtained without using
    subsets
    
    matchnum: which to combine (int -> all the same, iterable of ints -> for 
              each file in order)
    
    this might still give memory errors -> make histograms and combine those
    '''    
    odir = examplename.split('/')[:-1]
    if len(odir) == 0:
        odir = ddir
    else:
        odir = '/'.join(odir) + '/'
    onm = examplename.split('/')[-1]
    subpart = onm.split('_')[-1][:-5]
    subparts = subpart.split('-')
    total = int(subparts[2])
    
    filenames = [odir + '_'.join(onm.split('_')[:-1]) + \
                 '_{ind}-of-{tot}.hdf5'.format(ind=ind, tot=total) \
                 for ind in range(total)]
    outfile = odir + '_'.join(onm.split('_')[:-1]) + '.hdf5'
    
    if isinstance(matchnum, int):
        matchnum = [matchnum] * total
        newmatchnum = False
    with h5py.File(outfile, 'a') as fo:
        absdct = {}
        nndct = {}
        first = True
        for num, filen in zip(matchnum, filenames):
            with h5py.File(filen, 'r') as fi:
                cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
                slcat = fi['Header'].attrs['sightline_catalogue'].decode()
                
                agp = fi['absorbers']
                grp = fi['match_{num}'.format(num=num)]
                ggrp = grp['galaxy_selection']
                
                prev = list(fo.keys())
                if 'Header' not in prev: # first file -> do setup
                    hed = fo.create_group('Header')
                    csm = hed.create_group('cosmopars')
                    for key in cosmopars:
                        csm.attrs.create(key, cosmopars[key])
                    hed.attrs.create('sightline_catalogue', np.string_(slcat))
                
                if first:
                    halocat = grp.attrs['halo_catalogue'].decode()
                    nngb = grp.attrs['number_of_neighbours']
                    nsl = grp.attrs['number_of_slices_for_match']
                    dist3d = bool(grp.attrs['use_3D_distance'])
                    galsel = []
                    for key in ggrp:
                        arn = ggrp[key].attrs['array'].decode()
                        minv = ggrp[key].attrs['min']
                        maxv = ggrp[key].attrs['max']
                        galsel.append((arn, minv, maxv))
                        
                    for key in agp:
                        absdct[key] = np.array(agp[key])
                    
                    for key in grp:
                        if key == 'galaxy_selection':
                            continue
                        nndct[key] = np.array(grp[key])
                    first = False
                    
                else: # not the first file: do checks
                    _cosmopars = {key: val for key, val in fo['Header/cosmopars'].attrs.items()}
                    if not set(_cosmopars.keys()) == set(cosmopars.keys()):
                        raise RuntimeError('Cosmopars in the different files do not match (keys)')
                    if not np.all([_cosmopars[key] == cosmopars[key] for key in cosmopars]):
                        raise RuntimeError('Cosmopars in the different files do not match (values)')
                    
                    _slcat = fo['Header'].attrs['sightline_catalogue'].decode()
                    if not _slcat == slcat:
                        raise RuntimeError('Sightlines in the different files come from different catalogues')
                    
                    if not set(absdct.keys()) == set(agp.keys()):
                        print(absdct)
                        print(agp.keys())
                        raise RuntimeError('The different files have different absorber properties stored')
                    if not set(nndct.keys()) == set(grp.keys()) - {'galaxy_selection'}:
                        print(grp.keys())
                        raise RuntimeError('The different files have different neighbor properties stored')
                        
                    _halocat = grp.attrs['halo_catalogue'].decode()
                    _nngb = grp.attrs['number_of_neighbours']
                    _nsl = grp.attrs['number_of_slices_for_match']
                    _dist3d = bool(grp.attrs['use_3D_distance'])
                    _galsel = []
                    for key in ggrp:
                        arn = ggrp[key].attrs['array'].decode()
                        minv = ggrp[key].attrs['min']
                        maxv = ggrp[key].attrs['max']
                        _galsel.append((arn, minv, maxv))
                    if _halocat != halocat:
                        raise RuntimeError('The different files have different halo catalogues')
                    if _nngb != nngb:
                        raise RuntimeError('The different files have different numbers of neighbours')
                    if _nsl != nsl:
                        raise RuntimeError('The different files have different slice search radii')
                    if _dist3d != dist3d:
                        raise RuntimeError('The different files have different matching dimensions')
                    if set(galsel) != set(_galsel):
                        raise RuntimeError('The different files have different galaxy selections')
                    
                    if 'Header' in prev:
                        prev.remove('Header')
                    if 'absorbers' in prev:
                        prev.remove('absorbers')
                    
                    for key in absdct:
                        absdct[key] = np.append(absdct[key], np.array(agp[key]), axis=0)
                    for key in nndct:
                        nndct[key] = np.append(nndct[key], np.array(grp[key]), axis=0)
        # store input parameters 
        numthis = len(prev)
        if newmatchnum:
            grp = fo.create_group('match_{num}'.format(num=numthis))
        else:
            grp = fo.create_group('match_{num}'.format(num=matchnum[0]))
        grp.attrs.create('halo_catalogue', np.string_(halocat))
        grp.attrs.create('number_of_neighbours', nngb)
        grp.attrs.create('number_of_slices_for_match', nsl)
        grp.attrs.create('use_3D_distance', dist3d)
        ggrp = grp.create_group('galaxy_selection')
        for ti in range(len(galsel)):
            tup = galsel[ti]
            sggrp = ggrp.create_group('tuple_{}'.format(ti))
            sggrp.attrs.create('array', np.string_(tup[0]))
            #_minv = tup[1] if tup[1] is not None else np.string_('None')
            #_maxv = tup[2] if tup[2] is not None else np.string_('None')
            # no None checking: never decoded
            sggrp.attrs.create('min', minv)
            sggrp.attrs.create('max', maxv)      
        for key in nndct:
            grp.create_dataset(key, data=nndct[key])
        
        if 'absorbers' in fo:
            pass # no checks for now...
        else:
            agp = fo.create_group('absorbers')
            for key in absdct:
                agp.create_dataset(key, data=absdct[key])
            agp.attrs.create('info', np.string_('column density [log10 cm^-2]'))
                 
def absenv_test(matchnum=0,\
                nncat=mdir + 'slcat_test_1_nearest-neighbor-match_nngb-3_nsl-0.5.hdf5'):
    '''
    check the match in match_<matchnum> in the neighbor match catalogue nncat 
    '''
    
    ## read in the required input and output data
    with h5py.File(nncat, 'r') as fn:
        slcat = fn['Header'].attrs['sightline_catalogue'].decode()
        cosmopars = {key: val for key, val in fn['Header/cosmopars'].attrs.items()}
        boxperiod = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a']
        
        abspos_pmpc = np.array([fn['absorbers/xpos_pmpc'],\
                                fn['absorbers/ypos_pmpc'],\
                                fn['absorbers/zpos_pmpc'],\
                                ]).T
        mgrpn = 'match_{}'.format(matchnum)
        mgrp = fn[mgrpn]
        halocat = mgrp.attrs['halo_catalogue'].decode()
        nngb = mgrp.attrs['number_of_neighbours']
        matchrad = mgrp.attrs['number_of_slices_for_match']
        dist3d = bool(mgrp.attrs['use_3D_distance'])
        
        galsel = []
        for tupn in mgrp['galaxy_selection'].keys():
            arn =  mgrp['galaxy_selection/{}'.format(tupn)].attrs['array'].decode()
            vmin = mgrp['galaxy_selection/{}'.format(tupn)].attrs['min']
            vmax = mgrp['galaxy_selection/{}'.format(tupn)].attrs['max']
            try:
                vmin = float(vmin)
            except ValueError:
                vmin = None
            try:
                vmax = float(vmax)
            except ValueError:
                vmax = None
            galsel.append((arn, vmin, vmax))
        
        # for the match: check distnaces, and one of the match/listed properties
        galaxyid_nn = np.array(mgrp['galaxyid'])
        dists_nn    = np.array(mgrp['neighbor_dist_pmpc'])
        
        # check selection
        selarrs = {_sel[0]: np.array(mgrp[_sel[0]])  for _sel in galsel}    
        for tup in galsel:
            arr = selarrs[tup[0]]
            if tup[1] is not None:
                if np.any(arr < tup[1]):
                    print('Selection error: some neighbours had {arn} < {vmin}'.format(arn=tup[0], vmin=tup[1]))
            if tup[2] is not None:
                if np.any(arr >= tup[2]):
                    print('Selection error: some neighbours had {arn} >= {vmax}'.format(arn=tup[0], vmax=tup[2]))
        del selarrs
        
    with h5py.File(halocat, 'r') as hc:
        galpos_pmpc = np.array([np.array(hc['Xcom_cMpc']),\
                                np.array(hc['Ycom_cMpc']),\
                                np.array(hc['Zcom_cMpc']),\
                                ]).T
        galpos_pmpc *= cosmopars['a']
        galaxyid = np.array(hc['galaxyid'])
        
        # apply galaxy selection as recorded
        gsel = np.ones(len(galaxyid), dtype=bool)
        for tup in galsel:
            arr = np.array(hc[tup[0]])
            if tup[1] is not None:
                gsel &= arr >= tup[1]
            if tup[2] is not None:
                gsel &= arr < tup[2]
        galaxyid = galaxyid[gsel]
        galpos_pmpc = galpos_pmpc[gsel]

    with h5py.File(slcat, 'r') as fs:
        zopts_abs = np.array(fs['Header/zvals_cMpc']) * cosmopars['a']
    
    ## redo the match; less efficently, more transparently
    zslice = np.average(np.diff(zopts_abs))
    
    relpos = abspos_pmpc[:, np.newaxis, :] - galpos_pmpc[np.newaxis, :, :]
    # get the right distances for the periodic setup
    relpos += 0.5 * boxperiod
    relpos %= boxperiod
    relpos -= 0.5 * boxperiod
    if dist3d:
        dists_check = np.sqrt(np.sum(relpos**2, axis=2))
    else:
        dists_check = np.sqrt(np.sum(relpos[:, :, :2]**2, axis=2))
    # exclude slices out of z range: distances -> infinity
    znsel = np.abs(relpos[:, :, 2]) > matchrad * zslice
    dists_check[znsel] = np.inf

    inds_nn_check = np.argsort(dists_check, axis=1)
    galaxyid_nn_check = galaxyid[inds_nn_check][:, :nngb]
    
    ## check the outcomes
    if np.all(galaxyid_nn_check == galaxyid_nn):
        print('Galaxy ids matched as expected!')
    else:
        print('Some galaxy ids did not match expectations.')
        print('Main calculation:')
        print(galaxyid_nn)
        print('Check:')
        print(galaxyid_nn_check)
    
    allmatch = True
    for abi in range(dists_check.shape[0]):
        _dc = dists_check[abi]
        _as = inds_nn_check[abi]
        _dc_nn = _dc[_as][:nngb]
        
        if np.allclose(dists_nn[abi], _dc_nn):
            pass
        else:
            print('nearest distance mismatch for absorber {}'.format(abi))
            print('expected from check: {}'.format(_dc_nn))
            print('got from main calc.: {}'.format(dists_nn[abi]))
            allmatch = False
    if allmatch:
        print('All nearest neighbor distances matched as expected!')
    else:
        print('Some nearest neighbor distances did not match as expected.')        
    

def plot_absenv_hist_v1(nncat, outname,\
                        matchnum=0, prop='neighbor_dist_pmpc', ionsel=None):
    '''
    ionsel: ion column density selection -- {ion: (min, max)} dict
            column densities in log10 cm^-2
    prop:   name of the galaxy property to histogram (dataset name in nncat)
    nncat:  file name for the nearest-neighbor match catalogue
    matchnum: group in the nncat file to use (match_{matchnum})
    '''
    
    if '/' not in nncat:
        nncat = '/net/quasar/data2/wijers/slcat/' + nncat
    
    with h5py.File(nncat, 'r') as fn:
        # apply ion selection
        if ionsel is None:
            ionselstr = 'N selection: abs. cat.'
            abssel = slice(None, None, None)
        else:
            ionselstr = 'N selection:\n'
            ions_tosel = sorted(list(ionsel.keys()))
            ionarr = {ion: np.array(fn['absorbers/{ion}'.format(ion=ion)])\
                      for ion in ions_tosel}
            abssel = np.ones(len(ionarr[ions_tosel[0]]), dtype=bool)
            for ion in ions_tosel:
                vmin = ionsel[ion][0]
                vmax = ionsel[ion][1]
                ionselstr = ionselstr + '{ion}: ${_min:.2f} \\endash {_max:.2f}$\n'.format(ion=ild.getnicename(ion), _min=vmin, _max=vmax)
                abssel &= vmin <= ionarr[ion]
                abssel &= vmax >  ionarr[ion]
        
        # get nn, galaxy selection data
        grp = fn['match_{num}/galaxy_selection'.format(num=matchnum)]
        tups = list(grp.keys())
        galsels = []
        for tup in tups:
            galsels.append((grp[tup].attrs['array'].decode(),\
                            grp[tup].attrs['min'],\
                            grp[tup].attrs['max'],\
                            ))
        grp = fn['match_{num}'.format(num=matchnum)]
        dist3d = bool(grp.attrs['use_3D_distance'])
        zselrad = grp.attrs['number_of_slices_for_match']
        slcat = fn['Header'].attrs['sightline_catalogue'].decode()
        
        proparr = np.array(fn['match_{num}/{prop}'.format(num=matchnum, prop=prop)])[abssel]
    
    with h5py.File(slcat, 'r') as fs:
        zvals_abs = np.array(fs['Header/zvals_cMpc'])
        zslice = np.average(np.diff(zvals_abs))
        zstr = '$\pm \Delta$ Z searched: {zrad} cMpc\n'.format(zrad=zselrad * zslice)
        
        if ionsel is None:
            ionselstr = 'N selection:\n'
            grp = fs['Header/selection']
            ions  = sorted(list(fs['Header/mapfiles'].keys()))
            istrs = []
            for ion in ions:
                istrs.append('{ion} $ > {_min:.2f}$'.format(ion=ild.getnicename(ion), _min=grp.attrs[ion]))
            ionselstr = ionselstr + ' or\n'.join(istrs) + '\n'
    
                
    labels = {'neighbor_dist_pmpc': '$\\mathrm{{r}}_{{\\mathrm{{3D}}}} \\; [\\mathrm{pMpc}]$' if dist3d else\
                                    '$\\mathrm{{r}}_{{\\perp}} \\; [\\mathrm{pMpc}]$',\
              'M200c_Msun': '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\mathrm{{200c}}}} \\; [\\mathrm{{M}}_{{\\odot}}]$',\
              'Mstar_Msun': '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\star}} \\; [\\mathrm{{M}}_{{\\odot}}]$',\
              }
    
    loglist = ['M200c_Msun', 'Mstar_Msun']
    if prop in loglist:
        proparr = np.log10(proparr)
        
    galselstrs = []
    for gsel in galsels:
        _str = labels[gsel[0]]
        logv = gsel[0] in loglist
        _min = gsel[1]
        try:
            _min = float(_min)
            if logv:
                _min = np.log10(_min)
        except ValueError:
            _min = None
        _max = gsel[2]
        try:
            _max = float(_max)
            if logv:
                _max = np.log10(_max)
        except ValueError:
            _max = None
            
        if _min is None and _max is None:
            continue
        elif _min is None and _max is not None:
            _str = _str + ' $< {_max:.1f}$'.format(_max=_max)
        elif _min is not None and _max is None:
            _str = _str + ' $\geq {_min:.1f}$'.format(_min=_min)
        else:
            _str = _str + ' ${_min:.1f}\endash{_max:.1f}$'.format(_min=_min, _max=_max)
        galselstrs.append(_str)
    galselstr = 'galaxy selection:\n' + '\n'.join(galselstrs) + '\n'
    
    diststr = 'matching on $\\mathrm{{r}}_{{\\mathrm{{3D}}}}$\n' if dist3d else\
              'matching on $\\mathrm{{r}}_{{\\perp}}$\n'
    info = galselstr + ionselstr + zstr + diststr
    
    pmin = np.min(proparr)
    pmax = np.max(proparr)
    histstep = 0.1
    
    bmin = np.floor(pmin / histstep) * histstep
    bmax = np.ceil(pmax / histstep) * histstep
    edges = np.arange(bmin, bmax + 1.5 * histstep, histstep)
    hists = []
    for nn in range(proparr.shape[1]):
        _hist, _ed = np.histogram(proparr[:, nn], bins=edges)
        hists.append(_hist)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fontsize = 12
    
    for nn in range(len(hists)):
        ax.step(edges[:-1], hists[nn], where='post', label='nb. {nn}'.format(nn=nn + 1))
    
    ax.set_yscale('log')
    ax.set_xlabel(labels[prop], fontsize=fontsize)
    ax.set_ylabel('counts', fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='upper right')
    ax.text(0.02, 0.02, info, fontsize=fontsize - 1,\
            horizontalalignment='left', verticalalignment='bottom',\
            transform=ax.transAxes)
        
    plt.savefig(outname, box_inches='tight')
    
def save_absenv_hists(nncat, outname=None,\
                      matchnum=0, props=['neighbor_dist_pmpc', 'Mstar_Msun']):
    '''
    ionsel: ion column density selection -- {ion: (min, max)} dict
            column densities in log10 cm^-2
    prop:   name of the galaxy property to histogram (dataset name in nncat)
    nncat:  file name for the nearest-neighbor match catalogue
    matchnum: group in the nncat file to use (match_{matchnum})
    '''
    
    if '/' not in nncat:
        nncat = '/net/quasar/data2/wijers/slcat/' + nncat
    
    if outname is None:
        odir = '/'.join(nncat.split('/')[:-1]) + '/'
        oname = 'savedhists_' + nncat.split('/')[-1]
        outname = odir + oname
        
    # on/off for each ion subset
    ionsel_opts = {'o6': {'uvp': (meas['uvp']['o6'][0] - meas['uvp']['o6'][1],\
                                    np.inf),\
                          },\
                   'o7': {'cuv': (meas['co6']['o7'][0] - meas['co6']['o7'][1],\
                                  meas['co6']['o7'][0] + meas['co6']['o7'][2]),\
                          'cxr': (meas['co7']['o7'][0] - meas['co7']['o7'][1],\
                                  meas['co7']['o7'][0] + meas['co7']['o7'][2]),\
                          'suv': (meas['so6']['o7'][0] - meas['so6']['o7'][1],\
                                  meas['so6']['o7'][0] + meas['so6']['o7'][2]),\
                          'sxr': (meas['so7']['o7'][0] - meas['so7']['o7'][1],\
                                  meas['so7']['o7'][0] + meas['so7']['o7'][2]),\
                          'dlm': (detlim['o7'], np.inf),\
                          },\
                    'o8': {'cuv': (meas['co6']['o8'][0] - meas['co6']['o8'][1],\
                                   meas['co6']['o8'][0] + meas['co6']['o8'][2]),\
                           'cxr': (meas['co7']['o8'][0] - meas['co7']['o8'][1],\
                                   meas['co7']['o8'][0] + meas['co7']['o8'][2]),\
                           'suv': (meas['so6']['o8'][0] - meas['so6']['o8'][1],\
                                   meas['so6']['o8'][0] + meas['so6']['o8'][2]),\
                           'sxr': (meas['so7']['o8'][0] - meas['so7']['o8'][1],\
                                   meas['so7']['o8'][0] + meas['so7']['o8'][2]),\
                          },\
                  }
    # senisible combinations to actually use
    ionsels = {'catsel': None,\
               'o6uvp': {'o6': ionsel_opts['o6']['uvp']},\
               'o6uvp-o7dlm': {'o6': ionsel_opts['o6']['uvp'],\
                               'o7': ionsel_opts['o7']['dlm'],\
                               },\
              }
    ionsels.update({'o6uvp-{ion}{meas}'.format(ion=ion, meas=meas): \
                    {'o6': ionsel_opts['o6']['uvp'],\
                     ion: ionsel_opts[ion][meas]} \
                    for ion in ['o7', 'o8'] for meas in ['cuv', 'cxr', 'suv', 'sxr']})
    ionsels.update({'{ion}{meas}'.format(ion=ion, meas=meas): \
                    {ion: ionsel_opts[ion][meas]} \
                    for ion in ['o7', 'o8'] for meas in ['cuv', 'cxr', 'suv', 'sxr']})
    ionsels.update({'o7{meas}-o8{meas}'.format(meas=meas): \
                    {ion: ionsel_opts[ion][meas] for ion in ['o7', 'o8']} \
                    for meas in ['cuv', 'cxr', 'suv', 'sxr']})
    ionsels.update({'o6uvp-o7{meas}-o8{meas}'.format(meas=meas): \
                    {'o6': ionsel_opts['o6']['uvp'],\
                     'o7': ionsel_opts['o7'][meas],\
                     'o8': ionsel_opts['o8'][meas],\
                     } \
                    for meas in ['cuv', 'cxr', 'suv', 'sxr']})
    ions = list(ionsel_opts.keys())
    
    with h5py.File(nncat, 'r') as fn:
        cosmopars = {key: val for key, val in\
                     fn['Header/cosmopars'].attrs.items()}
        # for absorber selection
        ionarr = {ion: np.array(fn['absorbers/{ion}'.format(ion=ion)])\
                  for ion in ions}
        
        # get nn, galaxy selection data
        grp = fn['match_{num}/galaxy_selection'.format(num=matchnum)]
        tups = list(grp.keys())
        galsels = []
        for tup in tups:
            galsels.append((grp[tup].attrs['array'].decode(),\
                            grp[tup].attrs['min'],\
                            grp[tup].attrs['max'],\
                            ))
        grp = fn['match_{num}'.format(num=matchnum)]
        dist3d = bool(grp.attrs['use_3D_distance'])
        zselrad = grp.attrs['number_of_slices_for_match']
        slcat = fn['Header'].attrs['sightline_catalogue'].decode()
        
        proparrs = {prop: np.array(fn['match_{num}/{prop}'.format(num=matchnum, prop=prop)]) \
                    for prop in props}
    
    with h5py.File(slcat, 'r') as fs:
        zvals_abs = np.array(fs['Header/zvals_cMpc'])
        zslice = np.average(np.diff(zvals_abs))
        
        grp = fs['Header/selection']
        _ions  = sorted(list(fs['Header/mapfiles'].keys()))
        slcatsel = {ion: (grp.attrs[ion], np.inf) for ion in _ions}

    loglist = ['M200c_Msun', 'Mstar_Msun']
    binspace = {'M200c_Msun': 0.1,\
                'Mstar_Msun': 0.1,\
                'neighbor_dist_pmpc': 0.01,\
                }
    
    proparrs = {prop: np.log10(proparrs[prop]) if prop in loglist else\
                      proparrs[prop] \
                for prop in proparrs}
    
    with h5py.File(outname, 'a') as fo:
        # slcat info
        if 'Header' not in fo:
            hed = fo.create_group('Header')
            hed.attrs.create('nncat', np.string_(nncat))
            #hed.attrs.create('dist3d', dist3d)
            hed.attrs.create('halfzrad_search_cMpc', zslice * zselrad)
            hed.attrs.create('slicewidth_cMpc', zslice)
            csm = hed.create_group('cosmopars')
            for key in cosmopars.keys():
                csm.attrs.create(key, cosmopars[key])
            ggp = hed.create_group('galsel')
            for tupn in range(len(galsels)):
                tup = galsels[tupn]
                sggp = ggp.create_group('tuple_{num}'.format(num=tupn))
                sggp.attrs.create('array', np.string_(tup[0]))
                sggp.attrs.create('min', tup[1])
                sggp.attrs.create('max', tup[2])
            igp = hed.create_group('ionsel_slcat')
            for ion in slcatsel:
                igp.create_dataset(ion, data=np.array(slcatsel[ion]))
            grp = fo.create_group('match_{num}'.format(num=matchnum))
            grp.attrs.create('use_3D_distance', dist3d)
            
        # iterate over and store selections        
        for selkey in ionsels:
            ionsel = ionsels[selkey]
            
            if ionsel is None:
                abssel = slice(None, None, None)
            else:
                ions_tosel = sorted(list(ionsel.keys()))
                abssel = np.ones(len(ionarr[ions_tosel[0]]), dtype=bool)
                for ion in ions_tosel:
                    vmin = ionsel[ion][0]
                    vmax = ionsel[ion][1]
                    abssel &= vmin <= ionarr[ion]
                    abssel &= vmax >  ionarr[ion]
                    
            for prop in props:
                mgrn = 'match_{num}/{ionsel}/{prop}'.format(num=matchnum, ionsel=selkey, prop=prop)
                if mgrn in fo: # already done and stored
                    continue
                proparr = proparrs[prop][abssel]
                if len(proparr) == 0:
                    edges = np.array([np.NaN, np.NaN])
                    hists  = np.zeros((proparr.shape[1], 1), dtype=int)                
                else:
                    pmin = np.min(proparr)
                    pmax = np.max(proparr)   
                    histstep = binspace[prop]
            
                    bmin = np.floor(pmin / histstep) * histstep
                    bmax = np.ceil(pmax / histstep) * histstep
                    edges = np.arange(bmin - histstep, bmax + 1.5 * histstep, histstep)
                    hists = []
                    for nn in range(proparr.shape[1]):
                        _hist, _ed = np.histogram(proparr[:, nn], bins=edges)
                        hists.append(_hist)
                    hists = np.array(hists)
                
                grp = fo.create_group(mgrn)
                sgp = grp.create_group('ionsel')
                if ionsel is not None:
                    print(ionsel)
                    for ion in ionsel:
                        print(ion)
                        print(np.array(ionsel[ion]))
                        sgp.create_dataset(ion, data=np.array(ionsel[ion]))
                dse = grp.create_dataset('edges', data=edges)
                dse.attrs.create('log', prop in loglist)
                dsh = grp.create_dataset('hist', data=hists)
                dsh.attrs.create('info', np.string_('axis 0: neighbor number, axis 1: histogram'))

def combine_absenv_hists(examplename, matchnum=0):
    '''
    combine the 'x-of-y' absorer subset histograms into one file
    matchnum is assumed to be the same in all files, unless a list is provided
    '''
    
    odir = examplename.split('/')[:-1]
    if len(odir) == 0:
        odir = ddir
    else:
        odir = '/'.join(odir) + '/'
    onm = examplename.split('/')[-1]
    subpart = onm.split('_')[-1][:-5]
    subparts = subpart.split('-')
    total = int(subparts[2])
    
    filenames = [odir + '_'.join(onm.split('_')[:-1]) + \
                 '_{ind}-of-{tot}.hdf5'.format(ind=ind, tot=total) \
                 for ind in range(total)]
    outfile = odir + '_'.join(onm.split('_')[:-1]) + '.hdf5'
    
    if isinstance(matchnum, int):
        matchnum = [matchnum] * total
        newmatchnum = False
    
    
    with h5py.File(outfile, 'a') as fo:
        
        first = True
        hists = {}
        edges = {}
        ionsels = {}
        elogs = {}
        for matchn, filen in zip(matchnum, filenames):
            with h5py.File(filen, 'r') as fi:
                
                mgrp = fi['match_{num}'.format(num=matchn)]
                hed = fi['Header']
                ggp = hed['galsel']
                igp = hed['ionsel_slcat']
                        
                if first:
                    cosmopars = {key: val for key, val in\
                                 fi['Header/cosmopars'].attrs.items()}
                    nncat = hed.attrs['nncat'].decode()
                    halfdz = hed.attrs['halfzrad_search_cMpc']
                    zslice = hed.attrs['slicewidth_cMpc']
                    dist3d = bool(mgrp.attrs['use_3D_distance'])
                    
                    galsel = []
                    for key in ggp:
                        arn = ggp[key].attrs['array'].decode()
                        minv = ggp[key].attrs['min']
                        maxv = ggp[key].attrs['max']
                        galsel.append((arn, minv, maxv))
                        
                    slcatsel = {ion: np.array(igp[ion]) for ion in igp}
                
                    
                    
                else: # not the first file: do checks
                    _cosmopars = {key: val for key, val in\
                                 fi['Header/cosmopars'].attrs.items()}
                    if not set(_cosmopars.keys()) == set(cosmopars.keys()):
                        raise RuntimeError('Cosmopars in the different files do not match (keys)')
                    if not np.all([_cosmopars[key] == cosmopars[key] for key in cosmopars]):
                        raise RuntimeError('Cosmopars in the different files do not match (values)')
                     
                    _nncat = hed.attrs['nncat'].decode()
                    _halfdz = hed.attrs['halfzrad_search_cMpc']
                    _zslice = hed.attrs['slicewidth_cMpc']
                    _dist3d = bool(mgrp.attrs['use_3D_distance'])    
                        
                    _galsel = []
                    for key in ggp:
                        arn = ggp[key].attrs['array'].decode()
                        minv = ggp[key].attrs['min']
                        maxv = ggp[key].attrs['max']
                        _galsel.append((arn, minv, maxv))
                        
                    if _nncat != nncat:
                        # different component files -> expect different indices
                        _fnpart = _nncat.split('/')[-1]
                        _fnpart_tomatch = '_'.join((_fnpart.split('_'))[:-1]) + '_' + '-'.join(['index'] + _fnpart.split('_')[-1].split('-')[1:])
                        fnpart = nncat.split('/')[-1]
                        fnpart_tomatch = '_'.join((fnpart.split('_'))[:-1]) + '_' + '-'.join(['index'] + fnpart.split('_')[-1].split('-')[1:])
                        if _fnpart_tomatch != fnpart_tomatch:
                            raise RuntimeError('The different files have different neighbor catalogues')
                    if _zslice != zslice:
                        raise RuntimeError('The different files have different absorber slice widths')
                    if _halfdz != halfdz:
                        raise RuntimeError('The different files have different slice search radii')
                    if _dist3d != dist3d:
                        raise RuntimeError('The different files have different matching dimensions')
                    if set(galsel) != set(_galsel):
                        raise RuntimeError('The different files have different galaxy selections')
                
                for ionsel in mgrp:
                    for prop in mgrp[ionsel]:
                        iongrp = mgrp[ionsel][prop]['ionsel']
                        _isel = {ion: np.array(iongrp[ion]) for ion in iongrp}
                        if ionsel not in ionsels:
                            ionsels[ionsel] = {}
                        if prop not in ionsels[ionsel]:
                            ionsels[ionsel][prop] = _isel
                        else:
                            if set(ionsels[ionsel][prop].keys()) != set(_isel.keys()):
                                raise RuntimeError('Same ion selection names do not match selections (keys/ions)')
                            if not np.all([np.all(ionsels[ionsel][prop][ion] == _isel[ion]) for ion in _isel]):
                                raise RuntimeError('Same ion selection names do not match selections (ranges)')
                        
                        edge = np.array(mgrp[ionsel][prop]['edges'])
                        hist = np.array(mgrp[ionsel][prop]['hist'])
                        if np.sum(hist) == 0: # no contribution -> just skip (edges will be NaN, so that would give errors)
                            continue
                        if first:
                            elog  = bool(mgrp[ionsel][prop]['edges'].attrs['log'])
                            elogs[prop] = elog
                        else:
                            elog = elogs[prop]                                
                            _elog  = bool(mgrp[ionsel][prop]['edges'].attrs['log'])
                            if _elog != elog:
                                raise RuntimeError('log/lin edge spacing is inconsistent between histogram files')
                        
                        if ionsel not in hists:
                            hists[ionsel] = {}
                            edges[ionsel] = {}
                        if prop not in hists[ionsel]:
                            hists[ionsel][prop] = hist
                            edges[ionsel][prop] = [np.arange(hist.shape[0] + 1), edge]
                        else:
                            if hists[ionsel][prop].shape[0] != hist.shape[0]:
                                raise RuntimeError('Different files contain histograms for different neighbor numbers')
                            edges0 = np.arange(hists[ionsel][prop].shape[0] + 1)
                            hists[ionsel][prop], edges[ionsel][prop] = cu.combine_hists(hists[ionsel][prop], hist,\
                                                                       edges[ionsel][prop], [edges0, edge],\
                                                                       rtol=1e-5, atol=1e-8, add=True)
                
                first = False
                
        # slcat info
        if 'Header' not in fo:
            hed = fo.create_group('Header')
            hed.attrs.create('nncat', np.string_(nncat))
            #hed.attrs.create('dist3d', dist3d)
            hed.attrs.create('halfzrad_search_cMpc', halfdz)
            hed.attrs.create('slicewidth_cMpc', zslice)
            csm = hed.create_group('cosmopars')
            for key in cosmopars.keys():
                csm.attrs.create(key, cosmopars[key])
            ggp = hed.create_group('galsel')
            for tupn in range(len(galsel)):
                tup = galsel[tupn]
                sggp = ggp.create_group('tuple_{num}'.format(num=tupn))
                sggp.attrs.create('array', np.string_(tup[0]))
                sggp.attrs.create('min', tup[1])
                sggp.attrs.create('max', tup[2])
            igp = hed.create_group('ionsel_slcat')
            for ion in slcatsel:
                igp.create_dataset(ion, data=np.array(slcatsel[ion]))
        
        if newmatchnum:
            onum = 0
            while 'match_{num}'.format(num=onum) in fo:
                onum += 1
            print('Storing histograms as match num {num}'.format(num=onum))
        else:
            onum = matchnum[0]
        grp = fo.create_group('match_{num}'.format(num=onum))
        grp.attrs.create('use_3D_distance', dist3d)
            
        # iterate over and store selections   
        for ionsel in hists:                         
            for prop in hists[ionsel]:
                mgrn = 'match_{num}/{ionsel}/{prop}'.format(num=onum, ionsel=ionsel, prop=prop)
                if mgrn in fo: # already done and stored
                    continue
                    hists = np.array(hists)
                
                grp = fo.create_group(mgrn)
                sgp = grp.create_group('ionsel')
                if ionsels[ionsel][prop] is not None:
                    for ion in ionsels[ionsel][prop]:
                        sgp.create_dataset(ion, data=np.array(ionsels[ionsel][prop][ion]))
                dse = grp.create_dataset('edges', data=edges[ionsel][prop][1])
                dse.attrs.create('log', elog)
                dsh = grp.create_dataset('hist', data=hists[ionsel][prop])
                dsh.attrs.create('info', np.string_('axis 0: neighbor number, axis 1: histogram'))

def est3ddist(galaxy, zcomp=zuv, cosmopars=None):
    '''
    galaxy: dict with 'r' (impact parameter, pMpc), 'z' (redshift) entries
    
    returns:
    -------
    esimated 3d distance using the Hubble flow, in pMpc
    '''
    vdiff = (galaxy['z'] - zcomp) / (1. + zcomp) * c.c
    if cosmopars is None:
        _csm = None
    else:
        _csm = cosmopars.copy()
        _csm['z'] = zcomp
        _csm['a'] = 1. / (1. + zcomp)
    hf = cu.Hubble(zcomp, cosmopars=_csm)
    rlos = vdiff / hf / c.cm_per_mpc # pMpc units
    return np.sqrt(rlos**2 + galaxy['r']**2)
    
    
def plot_absenv_hist(toplot='dist2d', ionsel=None,\
                     ionsel_meas='all', histfile='auto', cumulative=True):
    '''
    toplot:      'dist2d', 'dist3d', or 'mstar' -- what to plot
    ionsel:      which ion column selections to apply
                 'det':     o6 prior + o7 detection limit
                 None:      only the sightline catalogue selection
                 otherwise: ions ('o6', 'o7', 'o8') separated by '-', e.g.
                            'o6', 'o7-o8', 'o6-o7-o8'
    ionsel_meas: for o7 and o8, which measurements to use/compare 
                (list of strings); options are
                (old): 'cie': CIE model fit
                'suv': slab model at the o6/UV redshift
                'sxr': slab model at the o7/X-ray redshift
    '''
    histfiles = {'2dmatch_sameslice': (ddir + 'savedhists_sightlinecat_z-0.1_selection1_nearest-neighbor-match_nngb-5_nsl-0.5.hdf5', 1),\
                 '2dmatch_2slice': (ddir + 'savedhists_sightlinecat_z-0.1_selection1_nearest-neighbor-match_nngb-5_nsl-1.0.hdf5', 0),\
                 '3dmatch': (ddir + 'savedhists_sightlinecat_z-0.1_selection1_nearest-neighbor-match_nngb-5_nsl-3.0.hdf5', 1),\
                 '2dmatch_sameslice_meas2': (ddir + 'savedshists_sightlinecat_z-0.1_selection2_nearest-neighbor-match_nngb-5_nsl-0.5.hdf5', 0),\
                 '2dmatch_2slice_meas2': (ddir + 'savedshists_sightlinecat_z-0.1_selection2_nearest-neighbor-match_nngb-5_nsl-1.0.hdf5', 0),\
                 }
    if cumulative:
        maxfracplot = 5e-3
        mincumulplot = 0.995
    else:
        maxfracplot = 1e-4
        mincumulplot = 0.999
    
    # axis label, name of the hdf5 groups containing the histograms
    if toplot == 'dist2d':
        prop = 'neighbor_dist_pmpc'
        xlabel = '$\\mathrm{{r}}_{{\\perp}} \\; [\\mathrm{pMpc}]$'
        if histfile == 'auto':
            hkey = '2dmatch_sameslice'
        else:
            hkey = histfile
    elif toplot == 'dist3d':
        prop = 'neighbor_dist_pmpc'
        xlabel = '$\\mathrm{{r}}_{{\\mathrm{{3D}}}} \\; [\\mathrm{pMpc}]$'
        hkey = '3dmatch'
    elif toplot == 'mstar':
        prop = 'Mstar_Msun' 
        xlabel = '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\star}} \\; [\\mathrm{{M}}_{{\\odot}}]$' 
        if histfile == 'auto':
            hkey = '2dmatch_sameslice'
        else:
            hkey = histfile
    else:
        raise ValueError('{} is not a valid toplot option'.format(toplot))
    histfile = histfiles[hkey] 
    ylabel = 'cumulative fraction of absorbers' if cumulative else \
             'fraction of absorbers'
    
    
    # names of the ion selection groups in the histogram files
    if ionsel == 'o6':
        ionsels = ['o6uvp']
    elif ionsel == 'det':
        ionsels = ['o6uvp-o7dlm']
    elif ionsel is None:
        ionsels = ['catsel']
    else:
        ions = ionsel.split('-')
        ions.sort()
        if ionsel_meas == 'all':
            ionsel_meas = ['cuv', 'cxr', 'suv', 'sxr']
        ionsel_meas.sort()
        
        o6part = 'o6uvp-' if 'o6' in ions else ''
        _ions = list(np.copy(ions))
        if 'o6' in _ions:
            _ions.remove('o6')
        ionsels = [o6part + '-'.join(['{ion}{meas}'.format(ion=ion, meas=meas)\
                                      for ion in _ions])\
                  for meas in ionsel_meas]
        meastype_ionsel = {ionsels[i]: ionsel_meas[i]\
                           for i in range(len(ionsel_meas))}
    print('Using ion selections {ionsels}'.format(ionsels=ionsels))
    
    with h5py.File(histfile[0], 'r') as hf:
        hed = hf['Header']
        cosmopars = {key: val for key, val in hed['cosmopars'].attrs.items()}
        print(list(hed.attrs.keys()))
        #dist3d = '3dmatch' in hkey
        print(hf['match_{num}'.format(num=histfile[1])].attrs.keys())
        dist3d = bool(hf['match_{num}'.format(num=histfile[1])].attrs['use_3D_distance'])
        if not dist3d == ('3dmatch' in hkey):
            raise RuntimeError('File {fn} should {nd} distance data, but does not'.format(\
                             fn=histfile[0], nd=hkey[:2]))
        zslice = hed.attrs['slicewidth_cMpc']
        zrad_search = hed.attrs['halfzrad_search_cMpc']
        galsels = []
        for tup in hed['galsel'].keys():
            galsels.append((hed['galsel'][tup].attrs['array'].decode(),\
                            hed['galsel'][tup].attrs['min'],\
                            hed['galsel'][tup].attrs['max'],\
                            ))
        selstr = {}
        if 'catsel' in ionsels:
            grp = hed['ionsel_slcat']
            _ions = sorted(list(grp.keys()))
            _minmax = {ion: np.array(grp[ion]) for ion in _ions}
            selstr['catsel'] = ' or '.join(['{ion} $\geq {_min}$'.format(\
                  ion=ild.getnicename(ion), _min=_minmax[ion][0])\
                  for ion in _ions])
             
        histedge = {}
        for _ionsel in ionsels:
            mgrn = 'match_{num}/{ionsel}/{prop}'.format(num=histfile[1], ionsel=_ionsel, prop=prop)
            grp = hf[mgrn]
            
            histedge[_ionsel] = {}
            histedge[_ionsel]['hist'] = np.array(grp['hist'])
            histedge[_ionsel]['edges'] = np.array(grp['edges'])
            #logv = bool(grp['edges'].attrs['log'])
            
            if _ionsel != 'catsel':
                igrp = grp['ionsel']
                _ions = sorted(list(igrp.keys()))
                _minmax = {ion: np.array(igrp[ion]) for ion in _ions}
                _selstr = []
                for _ion in _ions:
                    mm = _minmax[_ion]
                    stion = ild.getnicename(_ion)
                    if mm[0] == -np.inf and mm[1] == np.inf:
                        continue
                    elif mm[0] == -np.inf:
                        _selstr.append('{ion}: $< {_max:.2f}$'.format(\
                                       ion=stion, _max=mm[1]))
                    elif mm[1] == np.inf:
                        _selstr.append('{ion}: $\\geq {_min:.2f}$'.format(\
                                       ion=stion, _min=mm[0]))
                    else:
                        _selstr.append('{ion}: ${_min:.2f} \\endash {_max:.2f}$'.format(\
                                       ion=stion, _min=mm[0], _max=mm[1]))
                selstr[_ionsel] = '\n'.join(_selstr)
                
    labels = {'neighbor_dist_pmpc': '$\\mathrm{{r}}_{{\\mathrm{{3D}}}} \\; [\\mathrm{pMpc}]$' if dist3d else\
                                    '$\\mathrm{{r}}_{{\\perp}} \\; [\\mathrm{pMpc}]$',\
              'M200c_Msun': '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\mathrm{{200c}}}} \\; [\\mathrm{{M}}_{{\\odot}}]$',\
              'Mstar_Msun': '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\star}} \\; [\\mathrm{{M}}_{{\\odot}}]$',\
              }
    cumulstr = '_cumul' if cumulative else ''
    outname = mdir + 'nnhist_{prop}_{ionsel}_{histfile}{cstr}.pdf'.format(\
                             prop=prop, ionsel='_'.join(ionsels),\
                             histfile=hkey, cstr=cumulstr)   

    galselstrs = []
    for gsel in galsels:
        _str = labels[gsel[0]]
        _min = gsel[1]
        takelog = '\\log_{{10}}' in _str
        try:
            _min = float(_min)
            if takelog:
                _min = np.log10(_min)
        except ValueError:
            _min = None
        _max = gsel[2]
        try:
            _max = float(_max)
            if takelog:
                _max = np.log10(_max)
        except ValueError:
            _max = None
            
        if _min is None and _max is None:
            continue
        elif _min is None and _max is not None:
            _str = _str + ' $< {_max:.1f}$'.format(_max=_max)
        elif _min is not None and _max is None:
            _str = _str + ' $\geq {_min:.1f}$'.format(_min=_min)
        else:
            _str = _str + ' ${_min:.1f}\endash{_max:.1f}$'.format(_min=_min, _max=_max)
        galselstrs.append(_str)
    galselstr = 'galaxy selection:\n' + '\n'.join(galselstrs) + '\n'
    
    ionselstr = '$\\log_{{10}} \\mathrm{{N}} \\; [\\mathrm{{cm}}^{{-2}}]$ selection:\n'
    ionselstr = ionselstr + '\n'.join(['({count} absorbers)\n{sel}'.format(\
                                       ins=_ionsel, sel=selstr[_ionsel],\
                                       count=np.sum(histedge[_ionsel]['hist'][0], axis=0),\
                                       )\
                                       for _ionsel in ionsels]) \
                + '\n'
    zradstr = 'neighbor search:\n $\pm \\Delta z = {dz:.3f}$ cMpc\n'.format(
               dz=zrad_search)
    info = ionselstr + galselstr + zradstr
    if info[-1] == '\n':
        info = info[:-1]
    
    _linestyles = ['solid', 'dashed', 'dotted']
    ls_ionsel = {ionsels[si]: _linestyles[si] for si in range(len(ionsels))}
    
    if '3dmatch' in hkey:
        galdata_obs = sorted(galdata_meas,\
                    key=lambda x: est3ddist(x, zcomp=zuv, cosmopars=cosmopars))
    else:
        galdata_obs = sorted(galdata_meas, key=lambda x: x['r'])
    
    nngb = len(galdata_obs)
    colors_nn = {i: 'C{i}'.format(i=i%10) for i in range(nngb)}
    
    ## set up figure
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.add_subplot(1, 1, 1)
    fontsize = 12
    lw = 2
    alpha_data = 0.8
    alpha_err = 0.2
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if not cumulative:
        ax.set_yscale('log')
    ax.tick_params(which='both', direction='in', right=True, top=True,\
                   labelsize=fontsize - 1.)
    ax.minorticks_on()
    
    # plot data, probe data ranges
    ymax = -np.inf
    ymin = np.inf
    xmax = -np.inf
    xmin = np.inf
    for _ionsel in ionsels:
        hists = (histedge[_ionsel]['hist'][:nngb]).astype(np.float)
        edges =  histedge[_ionsel]['edges']
        
        hists /= np.sum(hists, axis=1)[:, np.newaxis]
        
        cumul = np.cumsum(hists, axis=1)
        ind = np.where(np.all(cumul > mincumulplot, axis=0))[0][0]
        ind = max(ind, np.where(np.any(hists >= np.max(hists) * maxfracplot, axis=0))[0][-1])
        
        xmax = max(xmax, edges[ind + 1])
        ymax = max(ymax, np.max(hists))
        ymin = min(ymin, ymax * maxfracplot) # np.min(hists[:, ind])
        xmin = min(xmin, edges[0])
        
        for nn in range(nngb):
            if cumulative:
                ax.plot(edges, np.append([0.], cumul[nn]), color=colors_nn[nn],\
                    linestyle=ls_ionsel[_ionsel], linewidth=lw)
            else:
                ax.step(edges[:-1], hists[nn], where='post', color=colors_nn[nn],\
                    linestyle=ls_ionsel[_ionsel], linewidth=lw)
    
    # add measured data:
    if prop == 'Mstar_Msun':
        for nn in range(nngb):
            xv = galdata_obs[nn]['mstar'][0]
            xvmin = galdata_obs[nn]['mstar'][0] - galdata_obs[nn]['mstar'][1]
            xvmax = galdata_obs[nn]['mstar'][0] + galdata_obs[nn]['mstar'][2]
            ax.axvline(xv, color=colors_nn[nn], linewidth=lw, alpha=alpha_data)
            ax.axvspan(xvmin, xvmax, alpha=alpha_err, color=colors_nn[nn])
            xmin = min(xmin, xvmin)
            xmax = max(xmax, xvmax)
    elif prop == 'neighbor_dist_pmpc':
        if dist3d:
            zobs = zuv
            _csm = cosmopars.copy()
            _csm['z'] = zobs
            _csm['a'] = 1. / (1. + zobs)
            # hubble flow offset within one slice
            zoff_est = 0.5 * zslice * c.cm_per_mpc * _csm['a'] \
                       * cu.Hubble(zobs, cosmopars=_csm) / c.c \
                       * (1. + zobs)
            for nn in range(nngb):
                galdata = galdata_obs[nn].copy()
                xv = est3ddist(galdata, zcomp=zobs, cosmopars=cosmopars)
                #z_orig = np.copy(galdata['z'])[()]
                xvbot = galdata['r']
                galdata['z'] -= zoff_est
                z_xmin = np.copy(galdata['z'])[()]
                xvmin = est3ddist(galdata, zcomp=zobs, cosmopars=cosmopars)
                galdata['z'] += 2. * zoff_est 
                z_xmax = np.copy(galdata['z'])[()]
                xvmax = est3ddist(galdata, zcomp=zobs, cosmopars=cosmopars)
                if z_xmin <= zobs and zobs <= z_xmax:
                    _xvmin = xvbot
                    xvmax = max(xvmin, xvmax)
                    xvmin = _xvmin
                
                ax.axvline(xv, color=colors_nn[nn], linewidth=lw, alpha=alpha_data)
                ax.axvline(xvbot, color=colors_nn[nn], linewidth=lw, alpha=alpha_data,\
                           linestyle='dashed')
                ax.axvspan(xvmin, xvmax, alpha=alpha_err, color=colors_nn[nn])
                #print(xvbot, xvmin, xvmax, xv)
                
                xmin = min(xmin, xvmin)
                xmax = max(xmax, xvmax)
                
        else:
            for nn in range(nngb):
                xv = galdata_obs[nn]['r']
                ax.axvline(xv, color=colors_nn[nn], linewidth=lw, alpha=alpha_data)
                
                xmin = min(xmin, xv)
                xmax = max(xmax, xv)
    
    xmar = 0.01 * (xmax - xmin)
    ymar = 0.01 * np.log10(ymax / ymin)
    ax.set_xlim(xmin - xmar, xmax + xmar)
    if cumulative:
        ax.set_ylim(0., 1.)
    else:
        ax.set_ylim(ymin, ymax * (1. + 10**ymar))
                
    if len(ionsels) > 1:
        handles1 = [mlines.Line2D([], [], color='black',\
                                  linewidth=lw,\
                                  linestyle=ls_ionsel[_ionsel],
                                  label=meastype_ionsel[_ionsel]) 
                    for _ionsel in ionsels]
    else:
        handles1 = []
    handles2 = [mlines.Line2D([], [], color=colors_nn[nn],\
                                  linewidth=lw,\
                                  linestyle='solid',
                                  label='nb. {nn}'.format(nn=nn + 1)) 
                    for nn in range(nngb)]
    
    if prop == 'neighbor_dist_pmpc':
        if cumulative:
            legendloc = 'lower right'
            legendanchor = (1., 0.4 + 0.07 * len(ionsels))
            legendncol = 1 #if handles1 == [] else 2
            legendframe = True
            
            infov = 'bottom'
            infoh = 'right'
            infox = 0.98
            infoy = 0.02
            infobbox = None
        else:
            legendloc = 'upper right'
            legendanchor = (1., 0.62 - 0.07 * len(ionsels))
            legendncol = 1 #if handles1 == [] else 2
            legendframe = True
            
            infov = 'top'
            infoh = 'right'
            infox = 0.98
            infoy = 0.98
            infobbox = None
        
    elif prop == 'Mstar_Msun':
        if cumulative:
            legendloc = 'upper left'
            legendanchor = (0.02, 0.98)
            legendncol = 1 if handles1 == [] else 2
            legendframe = True
            
            infov = 'bottom'
            infoh = 'right'
            infox = 0.98
            infoy = 0.02
            infobbox = dict(facecolor=(1., 1., 1., 0.5), edgecolor='gray',\
                            boxstyle='round')
        else:
            legendloc = 'upper right'
            legendanchor = (0.98, 0.98)
            legendncol = 1 if handles1 == [] else 2
            legendframe = True
            
            infov = 'bottom'
            infoh = 'left'
            infox = 0.02
            infoy = 0.02
            infobbox = dict(facecolor=(1., 1., 1., 0.5), edgecolor='gray',\
                            boxstyle='round')
    ax.legend(handles=handles1 + handles2, fontsize=fontsize - 1.,\
              loc=legendloc, bbox_to_anchor=legendanchor, ncol=legendncol,\
              frameon=legendframe)
    ax.text(infox, infoy, info, fontsize=fontsize - 1.,\
            verticalalignment=infov, horizontalalignment=infoh,\
            transform=ax.transAxes, bbox=infobbox)
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
