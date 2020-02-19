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

import make_maps_opts_locs as ol
import eagle_constants_and_units as c
import cosmo_utils as cu

# put stored files here
ddir = '/net/quasar/data2/wijers/slcat/'
mdir = '/net/luttero/data2/jussi_ton180_data/'

n_jobs = int(os.environ['OMP_NUM_THREADS']) # not shared-memory, I think, but a good indicator

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
                galsel='def', nngb=3, nsl=0.5, dist3d=False):
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
    '''
    
    # load sightline data
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
        
        coldens = {ion: np.array(fs['coldens_{ion}'.format(ion=ion)]) for ion in ions}
        xpos = np.array(fs['xpos_pmpc'])
        ypos = np.array(fs['ypos_pmpc'])
        zpos = np.array(fs['zpos_pmpc'])
        
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
        neigh_dist, neigh_ind = nnfinder.kneighbors(X=abspos,\
                                                    n_neighbors=nngb,\
                                                    return_distance=True)
        #print(neigh_dist)
        #print(neigh_ind)
        neighprops_this = {key: halodct_b[key][neigh_ind] for key in halodct_b}
        neighprops_this.update({'neighbor_dist_pmpc': neigh_dist})
        
        del nnfinder
        
        for key in neighpropdct:
            neighpropdct[key] = np.append(neighpropdct[key], neighprops_this[key], axis=0)
        # to make sure the order of the absorber arrays is maintained
        zselmax_last = zsel_abs.stop
        
    ## save the data
    outfile = slcat.split('/')[-1]
    outfile = '.'.join(outfile.split('.')[:-1])
    outfile = ddir + outfile + '_nearest-neighbor-match_nngb-{nngb}_nsl-{nsl}.hdf5'.format(nngb=nngb, nsl=nsl)
    
    with h5py.File(outfile, 'a') as fo:
        prev = fo.keys()
        if 'Header' not in prev:
            hed = fo.create_group('Header')
            csm = hed.create_group('cosmopars')
            for key in cosmopars:
                csm.attrs.create(key, cosmopars[key])
            hed.attrs.create('sightline_catalogue', np.string_(slcat))
            
            agp = fo.create_group('absorbers')
            for key in absdct:
                agp.create_dataset(key, data=absdct[key])
                if key[1:] != 'pos':
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
    
