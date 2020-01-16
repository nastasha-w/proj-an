#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:27:59 2019

@author: wijers
"""

import numpy as np
import h5py
import pandas as pd
import os
import string

import selecthalos as sh
import cosmo_utils as cu
import eagle_constants_and_units as c
import make_maps_v3_master as m3
import make_maps_opts_locs as ol

# directory for metadata files
tdir = '/net/luttero/data2/imgs/CGM/3dprof/'

defaults = {'sample': 'L0100N1504_27_Mh0p5dex_1000'}

samples = {'L0100N1504_27_Mh0p5dex_1000': sh.L0100N1504_27_Mh0p5dex_1000,\
           'L0100N1504_27_Mstar-Mh0p5dex-match_1000': sh.L0100N1504_27_Mstar_Mhbinmatch_1000}

weighttypes = {'Mass': {'ptype': 'basic', 'quantity': 'Mass'},\
               'Volume': {'ptype': 'basic', 'quantity': 'propvol'},\
               'gas':   {'ptype': 'basic', 'quantity': 'Mass', 'parttype': '0'},\
               'ISM':   {'ptype': 'basic', 'quantity': 'Mass', 'parttype': '0'},\
               'stars': {'ptype': 'basic', 'quantity': 'Mass', 'parttype': '4'},\
               'BHs':   {'ptype': 'basic', 'quantity': 'Mass', 'parttype': '5'},\
               'DM':    {'ptype': 'basic', 'quantity': 'Mass', 'parttype': '1'},\
               }
weighttypes.update({ion: {'ptype': 'Nion', 'ion': ion} for ion in\
                    ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'oxygen',\
                     'ne8', 'ne9', 'neon', 'fe17', 'iron', 'hneutralssh']}) 
for ion in ['oxygen', 'neon', 'iron']:
    weighttypes.update({'ISM-%s'%{ion}: {'ptype': 'Nion', 'ionW': ion, 'parttype': '0'},\
                        'gas-%s'%{ion}: {'ptype': 'Nion', 'ionW': ion, 'parttype': '0'},\
                        'stars-%s'%{ion}: {'ptype': 'Nion', 'ionW': ion, 'parttype': '4'},\
                        })

def dataname(samplen):
    return tdir + 'halodata_%s.txt'%(samplen)

def files(samplen, weighttype, histtype=None):
    if histtype is None or histtype == 'rprof_rho-T-nion':
        return tdir + 'filenames_%s_%s.txt'%(samplen, weighttype)
    elif histtype == 'ionmass':
        return tdir + 'filenames_%s_%s_ionmass.txt'%(samplen, weighttype)
    elif histtype == 'rprof':
        return tdir + 'filenames_%s_%s_ionmass.txt'%(samplen, weighttype)
    elif histtype.startswith('Zprof'):
        return tdir + 'filenames_%s_%s_%s.txt'%(samplen, weighttype, histtype)
    
def combine_hists(h1, h2, e1, e2, rtol=1e-5, atol=1e-8, add=True):
    '''
    add histograms h1, h2 with the same dimension, after aligning edges e1, e2
    add = True -> add histograms, return sum
    add = False -> align histograms, return padded histograms and bins
    
    e1, e2 are sequences of arrays, h1, h2 are arrays
    edgetol specifies what relative/absolute (absolute if one is zero) 
    differences between edge values are acceptable to call bin edges equal
    
    if edges are not equal along some axis, they must be on a common, equally 
    spaced grid.
    (this is meant for combining histograms run with the same float or fixed 
    array axbins options)
    '''
    if len(h1.shape) != len(h2.shape):
        raise ValueError('Can only add histograms of the same shape')
    if not (np.all(np.array(h1.shape) == np.array([len(e) - 1 for e in e1]))\
            and \
            np.all(np.array(h2.shape) == np.array([len(e) - 1 for e in e2]))\
           ):
        raise ValueError('Histogram shape does not match edges')
       
    # iterate over edges, determine overlaps
    p1 = []
    p2 = []
    es = []

    for ei in range(len(e1)):
        e1t = np.array(e1[ei])
        e2t = np.array(e2[ei])
        p1t = [None, None]
        p2t = [None, None]
        
        # if the arrays happen to be equal, it's easy
        if len(e1t) == len(e2t):
            if np.allclose(e1t, e2t, rtol=rtol, atol=atol):
                p1t = [0, 0]
                p2t = [0, 0]
                es.append(0.5 * (e1t + e2t))
                p1.append(p1t)
                p2.append(p2t)
                continue
        
        # if not, things get messy fast. Assume equal spacing (check) 
        s1t = np.diff(e1t)
        s2t = np.diff(e2t)
        if not np.allclose(s1t[0][np.newaxis], s1t):
            raise RuntimeError('Cannot deal with unequally spaced arrays that do not match (axis %i)'%(ei))
        if not np.allclose(s2t[0][np.newaxis], s2t):
            raise RuntimeError('Cannot deal with unequally spaced arrays that do not match (axis %i)'%(ei))
        if not np.isclose(np.average(s1t), np.average(s2t), atol=atol, rtol=rtol):
            raise RuntimeError('Cannot deal with differently spaced arrays (axis %i)'%(ei)) 
        st = 0.5 * (np.average(s1t) + np.average(s2t))
        if st <= 0.:
            raise RuntimeError('Cannot deal with decreasing array values (axis %i)'%(ei))
        # check if the arrays share a zero point for their scales
        if not np.isclose(((e1t[0] - e2t[0]) / st + 0.5) % 1 - 0.5, 0., atol=atol, rtol=rtol):
            raise RuntimeError('Cannot deal with arrays not on a common grid (axis %i)'%(ei))

        g0 = 0.5 * ((e1t[0] / st + 0.5) % 1. - 0.5 + (e2t[0] / st + 0.5) % 1. - 0.5)        
        # calulate indices of the array endpoints on the common grid (zero point is g0)
        e1i0 = int(np.floor((e1t[0] - g0) / st + 0.5))
        e1i1 = int(np.floor((e1t[-1] - g0) / st + 0.5))
        e2i0 = int(np.floor((e2t[0] - g0) / st + 0.5))
        e2i1 = int(np.floor((e2t[-1] - g0) / st + 0.5))
        
        # set histogram padding based on grid indices
        p1t = [None, None]
        p2t = [None, None]
        if e1i0 > e2i0:
            p1t[0] = e1i0 - e2i0
            p2t[0] = 0
        else:
            p1t[0] = 0
            p2t[0] = e2i0 - e1i0
        if e1i1 > e2i1:
            p1t[1] = 0
            p2t[1] = e1i1 - e2i1
        else:
            p1t[1] = e2i1 - e1i1
            p2t[1] = 0
        # set up new edges based on the grid, initially
        esi0 = min(e1i0, e2i0)
        esi1 = max(e1i1, e2i1)
        est = np.arange(g0 + esi0 * st, g0 + (esi1 + 0.5) * st, st)
        # overwrite with old edges (2, then 1, to give preference to the histogram 1 edges)
        # meant to avoid accumulating round-off errors through st, g0
        est[e2i0 - esi0: e2i1 + 1 - esi0] = e2t
        est[e1i0 - esi0: e1i1 + 1 - esi0] = e1t
        
        p1.append(p1t)
        p2.append(p2t)
        es.append(est)

    #print(p1)
    #print(p2)
    #print(es)
        
    h1 = np.pad(h1, mode='constant', constant_values=0, pad_width=p1)
    h2 = np.pad(h2, mode='constant', constant_values=0, pad_width=p2)
    if add:
        hs = h1 + h2
        return hs, es
    else:
        return h1, h2, es
    
def gensample(samplename=None, galaxyselector=None):
    '''
    retrieve and store metadata for a given sample
    '''
    if samplename is None:
        samplename = defaults['sample']
    if galaxyselector is None:
        galaxyselector = samples[samplename]
    halocat = galaxyselector.halocat
    galids_dct = galaxyselector.galids()
    galids_all = np.array([gid for key in galids_dct.keys() for gid in galids_dct[key]])
    fon = dataname(samplename)
    
    with h5py.File(halocat, 'r') as hc:
        cosmopars = {key: item for key, item in hc['Header/cosmopars'].attrs.items()}
        Xcom = np.array(hc['Xcom_cMpc'])
        Ycom = np.array(hc['Ycom_cMpc'])
        Zcom = np.array(hc['Zcom_cMpc'])
        R200c = np.array(hc['R200c_pkpc']) / cosmopars['a'] * 1e-3
        M200c = np.array(hc['M200c_Msun'])
        Mstar = np.array(hc['Mstar_Msun'])
        galids = np.array(hc['galaxyid'])
        
        gsel = cu.match(galids, galids_all, arr2_sorted=False, arr2_index=None) # for array 1: index of same elt. in array 2, or -1 if not in array 2
        gsel = np.where(gsel >= 0) # no match -> index set to -1

        galids = galids[gsel]
#        if not np.all(galids_all == galids): # will be triggered since orders are idfferent, but sets are the same
#            print(galids_all)
#            print(len(galids_all))
#            print(galids)
#            print(len(galids))
#            print(gsel)
#            print(len(gsel[0]))
#            print(set(galids) == set(galids_all))
#            raise RuntimeError('Something has gone wrong in the galaxy id matching')       
        Xcom = Xcom[gsel]
        Ycom = Ycom[gsel]
        Zcom = Zcom[gsel]
        R200c = R200c[gsel]
        M200c = M200c[gsel]
        Mstar = Mstar[gsel]
        
    with open(fon, 'w') as fo:
        fo.write('halocat:\t%s\n'%(halocat))
        fo.write('samplename:\t%s\n'%(samplename))
        fo.write('galaxyid\tXcom_cMpc\tYcom_cMpc\tZcom_cMpc\tR200c_cMpc\tM200c_Msun\tMstar_Msun\n')
        for gi in range(len(galids)):
            fo.write('%i\t%f\t%f\t%f\t%f\t%f\t%f\n'%(galids[gi], Xcom[gi], Ycom[gi], Zcom[gi], R200c[gi], M200c[gi], Mstar[gi]))
        
def genhists(samplename=None, rbinu='pkpc', idsel=None, weighttype='Mass',\
             logM200min=11.0, axdct='rprof_rho-T-nion'):
    '''
    generate the histograms for a given sample
    rbins: used fixed bins in pkpc or in R200c (relevant for stacking)
    axdct: axdct to use for a given weight type (names for different sets)
           'rprof_rho-T-nion': ion-weighted temperature, density, ion density (if 
           ion-weighted) as a function of radius
           'Zprof[-<elt>]': metallicity profile. Abundance of the parent 
           element for ions, otherwise or overwritten by element after '-'
           (e.g. 'Zprof' or 'Zprof-oxygen')
    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    '''
    if samplename is None:
        samplename = defaults['sample']
    fin = dataname(samplename)
    
    with open(fin, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%fin)
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
        #ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(fin, header=headlen, sep='\t', index_col='galaxyid')
    if idsel is not None:
        if isinstance(idsel, slice):
            galaxyids = np.array(galdata_all.index)[idsel]
        else:
            galaxyids = idsel
    else:
        galaxyids = np.array(galdata_all.index)
    
    if axdct == 'rprof_rho-T-nion':
        axesdct = [{'ptype': 'coords', 'quantity': 'r3D'},\
                   {'ptype': 'basic', 'quantity': 'Temperature'},\
                   {'ptype': 'basic', 'quantity': 'Density'},\
                   ]
        if weighttype in ol.elements_ion.keys():
            axesdct =  axesdct + [{'ptype': 'Niondens', 'ion': weighttype}]
        nonrbins = [0.1] * (len(axesdct) - 1)
        name_append = '_%s_snapdata'%rbinu
    elif axdct.startswith('Zprof'):
        axesdct = [{'ptype': 'coords', 'quantity': 'r3D'},\
                   ]
        parts = axdct.split('-')
        if len(parts) > 1:
            elt = parts[1]
        elif weighttype in ol.elements_ion.keys():
            elt = ol.elements_ion[weighttype]
            axdct = axdct + '-%s'%(elt)
        else:
            raise ValueError('axdct Zprof for weighttype %s needs an element specifier'%weighttype)
        axesdct =  axesdct + [{'ptype': 'basic', 'quantity': 'SmoothedElementAbundance/%s'%(string.capwords(elt))}]  
        Zbins = np.array([-np.inf] + list(np.arange(-38.0, -0.95, 0.1)) + [np.inf]) # need the -inf in there to deal with Z=0 particles properly; non-inf edges from stacks with bin=0.1 runs
        nonrbins = [Zbins] * (len(axesdct) - 1)
        name_append = '_%s_snapdata_corrZ'%rbinu
        
    with open(files(samplename, weighttype, histtype=axdct), 'w') as fdoc:
        fdoc.write('galaxyid\tfilename\tgroupname\n')
        
        for gid in galaxyids:
            R200c = galdata_all.at[gid, 'R200c_cMpc']
            Xcom = galdata_all.at[gid, 'Xcom_cMpc']
            Ycom = galdata_all.at[gid, 'Ycom_cMpc']
            Zcom = galdata_all.at[gid, 'Zcom_cMpc']
            M200 = galdata_all.at[gid, 'M200c_Msun']
            if M200 < 10**logM200min:
                continue
            
            if rbinu == 'pkpc':
                rbins = np.array([0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 350., 400., 450., 500.]) * 1e-3 * c.cm_per_mpc
                if rbins[-1] < R200c:
                    rbins = np.append(rbins, [R200c])
            else:
                rbins = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.25, 1.50, 2., 2.5, 3., 3.5, 4.]) * R200c * c.cm_per_mpc * cosmopars['a']
            cen = [Xcom, Ycom, Zcom]
            L_x, L_y, L_z = (2. * rbins[-1] / c.cm_per_mpc / cosmopars['a'],) * 3
            
            axbins =  [rbins] + nonrbins
            logax = [False] + [True] * (len(axesdct) - 1)
            
            args = (weighttypes[weighttype]['ptype'], simnum, snapnum, var, axesdct,)
            kwargs = {'simulation': 'eagle', 'excludeSFR': 'T4', 'abunds': 'Pt', 'parttype': '0',\
                      'sylviasshtables': False, 'allinR200c': True, 'mdef': '200c',\
                      'L_x': L_x, 'L_y': L_y, 'L_z': L_z, 'centre': cen, 'Ls_in_Mpc': True,\
                      'misc': None,\
                      'axbins': axbins, 'logax': logax,\
                      'name_append': name_append, 'loghist': False}
            
            kwargs_extra = weighttypes[weighttype].copy()
            del kwargs_extra['ptype']
            kwargs.update(kwargs_extra)
            
            # ion, quantity, nameonly,
            outname = m3.makehistograms_perparticle(*args, nameonly=True, **kwargs)
            
            alreadyexists = False
            if os.path.isfile(outname[0]):
                with h5py.File(outname[0]) as fo_t:
                    if outname[1] in fo_t.keys():
                        alreadyexists = True
            if alreadyexists:
                print('For galaxy %i, a histogram already exists; skipping'%(gid))
            else:
                m3.makehistograms_perparticle(*args, nameonly=False, **kwargs)
            
            fdoc.write('%i\t%s\t%s\n'%(gid, outname[0], outname[1]))

def genhists_ionmass(samplename=None, rbinu='R200c', idsel=None, weighttype='o6', logM200min=11.0):
    '''
    generate the histograms for a given sample
    rbins: used fixed bins in pkpc or in R200c (relevant for stacking)
    
    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    '''
    if samplename is None:
        samplename = defaults['sample']
    fin = dataname(samplename)
    
    with open(fin, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%fin)
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
        #ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(fin, header=headlen, sep='\t', index_col='galaxyid')
    if idsel is not None:
        if isinstance(idsel, slice):
            galaxyids = np.array(galdata_all.index)[idsel]
        else:
            galaxyids = idsel
    else:
        galaxyids = np.array(galdata_all.index)
    
    name_append = '_%s'%rbinu
    axesdct = [{'ptype': 'coords', 'quantity': 'r3D'},\
               ]
    
    with open(files(samplename, weighttype, histtype='ionmass'), 'w') as fdoc:
        fdoc.write('galaxyid\tfilename\tgroupname\n')
        
        for gid in galaxyids:
            R200c = galdata_all.at[gid, 'R200c_cMpc']
            Xcom = galdata_all.at[gid, 'Xcom_cMpc']
            Ycom = galdata_all.at[gid, 'Ycom_cMpc']
            Zcom = galdata_all.at[gid, 'Zcom_cMpc']
            M200 = galdata_all.at[gid, 'M200c_Msun']
            if M200 < 10**logM200min:
                continue
            
            if rbinu == 'pkpc':
                rbins = np.array([0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 350., 400., 450., 500.]) * 1e-3 * c.cm_per_mpc
                if rbins[-1] < R200c:
                    rbins = np.append(rbins, [R200c])
            else:
                rbins = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.25, 1.50, 2., 2.5, 3., 3.5, 4.]) * R200c * c.cm_per_mpc * cosmopars['a']
            cen = [Xcom, Ycom, Zcom]
            L_x, L_y, L_z = (2. * rbins[-1] / c.cm_per_mpc / cosmopars['a'],) * 3
            
            axbins =  [rbins] + [0.1]
            logax = [False] + [True] 
            
            bentables = False
            if weighttype in ol.elements_ion:
                if ol.elements_ion[weighttype] == 'oxygen':
                    bentables = True
            
            args = (weighttypes[weighttype]['ptype'], simnum, snapnum, var, axesdct,)
            kwargs = {'simulation': 'eagle', 'excludeSFR': 'T4', 'abunds': 'Pt', 'parttype': '0',\
                      'sylviasshtables': False, 'bensgadget2tables': bentables,\
                      'allinR200c': True, 'mdef': '200c',\
                      'L_x': L_x, 'L_y': L_y, 'L_z': L_z, 'centre': cen, 'Ls_in_Mpc': True,\
                      'misc': None,\
                      'axbins': axbins, 'logax': logax,\
                      'name_append': name_append, 'loghist': False}
            
            kwargs_extra = weighttypes[weighttype].copy()
            del kwargs_extra['ptype']
            kwargs.update(kwargs_extra)
            
            # ion, quantity, nameonly,
            outname = m3.makehistograms_perparticle(*args, nameonly=True, **kwargs)
            
            alreadyexists = False
            if os.path.isfile(outname[0]):
                with h5py.File(outname[0]) as fo_t:
                    if outname[1] in fo_t.keys():
                        alreadyexists = True
            if alreadyexists:
                print('For galaxy %i, a histogram already exists; skipping'%(gid))
            else:
                m3.makehistograms_perparticle(*args, nameonly=False, **kwargs)
            
            fdoc.write('%i\t%s\t%s\n'%(gid, outname[0], outname[1]))


def genhists_massdist(samplename=None, rbinu='pkpc', idsel=None,\
                      weighttype='gas',\
                      logM200min=11.0, axdct='rprof'):
    '''
    generate the histograms for a given sample
    rbins: used fixed bins in pkpc or in R200c (relevant for stacking)
    axdct: axdct to use for a given weight type (names for different sets)
           'rprof': total mass in each radial bin
    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    '''
    if samplename is None:
        samplename = defaults['sample']
    fin = dataname(samplename)
    
    with open(fin, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%fin)
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
        #ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(fin, header=headlen, sep='\t', index_col='galaxyid')
    if idsel is not None:
        if isinstance(idsel, slice):
            galaxyids = np.array(galdata_all.index)[idsel]
        else:
            galaxyids = idsel
    else:
        galaxyids = np.array(galdata_all.index)
    
    if axdct == 'rprof':
        axesdct = [{'ptype': 'coords', 'quantity': 'r3D'},\
                   ]
        nonrbins = []
        if weighttype.startswith('gas'):
            # gas particle min. SFR:
            # min. gas density = 57.7 * rho_mean (cosmic mean, using rho_matter * omegab to e safe) = 7.376116060910138e-30
            # SFR = m_g * A * (M_sun / pc^2)^n * (gamma / G * f_g * P) * (n - 1) / 2
            # gamma = 5/3, G = newton constant, f_g = 1 (gas fraction), P = total pressure
            # A = 1.515 × 10−4 M⊙ yr−1 kpc−2, n = 1.4 (n = 2 at nH > 10^3 cm^-3)
            
            axesdct.append({'ptype': 'basic', 'quantity': 'StarFormationRate'})
            # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
            minval = 2**-149 * c.solar_mass / c.sec_per_year 
            nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
        name_append = '_%s_snapdata'%rbinu
            
    
        
    with open(files(samplename, weighttype, histtype=axdct), 'w') as fdoc:
        fdoc.write('galaxyid\tfilename\tgroupname\n')
        
        for gid in galaxyids:
            R200c = galdata_all.at[gid, 'R200c_cMpc']
            Xcom = galdata_all.at[gid, 'Xcom_cMpc']
            Ycom = galdata_all.at[gid, 'Ycom_cMpc']
            Zcom = galdata_all.at[gid, 'Zcom_cMpc']
            M200 = galdata_all.at[gid, 'M200c_Msun']
            if M200 < 10**logM200min:
                continue
            
            if rbinu == 'pkpc':
                rbins = np.array([0., 5., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 350., 400., 450., 500.]) * 1e-3 * c.cm_per_mpc
                if rbins[-1] < R200c:
                    rbins = np.append(rbins, [R200c])
            else:
                rbins = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.25, 1.50, 2., 2.5, 3., 3.5, 4.]) * R200c * c.cm_per_mpc * cosmopars['a']
            cen = [Xcom, Ycom, Zcom]
            L_x, L_y, L_z = (2. * rbins[-1] / c.cm_per_mpc / cosmopars['a'],) * 3
            
            axbins =  [rbins] + nonrbins
            logax = [False] + [False] * (len(axesdct) - 1)
            
            args = (weighttypes[weighttype]['ptype'], simnum, snapnum, var, axesdct,)
            kwargs = {'simulation': 'eagle', 'excludeSFR': 'T4', 'abunds': 'Pt',\
                      'sylviasshtables': False, 'allinR200c': True, 'mdef': '200c',\
                      'L_x': L_x, 'L_y': L_y, 'L_z': L_z, 'centre': cen, 'Ls_in_Mpc': True,\
                      'misc': None,\
                      'axbins': axbins, 'logax': logax,\
                      'name_append': name_append, 'loghist': False}
            
            kwargs_extra = weighttypes[weighttype].copy()
            del kwargs_extra['ptype']
            kwargs.update(kwargs_extra)
            
            # ion, quantity, nameonly,
            outname = m3.makehistograms_perparticle(*args, nameonly=True, **kwargs)
            
            alreadyexists = False
            if os.path.isfile(outname[0]):
                with h5py.File(outname[0]) as fo_t:
                    if outname[1] in fo_t.keys():
                        alreadyexists = True
            if alreadyexists:
                print('For galaxy %i, a histogram already exists; skipping'%(gid))
            else:
                m3.makehistograms_perparticle(*args, nameonly=False, **kwargs)
            
            fdoc.write('%i\t%s\t%s\n'%(gid, outname[0], outname[1]))
            
        
def combhists(samplename=None, rbinu='pkpc', idsel=None, weighttype='Mass',\
              binby=('M200c_Msun', 10**np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15.])),\
              combmethod='addnormed-R200c', histtype='rprof_rho-T-nion'):
    '''
    generate the histograms for a given sample
    rbinu: used fixed bins in pkpc or in R200c (relevant for stacking)
    
    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    
    binby: column, edges tuple
           which halos to combine into one histogram
    combmethod: how to combine. options are: 
           - 'add': just add all the histograms together
           - 'addnormed-R200c': add histograms normalized by the sum of weights
              within R200c (only if rbinu is 'R200c')
           - 'addnormed-M200c': add histograms normalized by M200c 
             (only equivalent to the previous if it's a Mass-weighted 
             histogram)
           - 'addnormed-all': add histograms normalized by the sum of the 
             histogram to the outermost radial bin
    '''
    if samplename is None:
        samplename = defaults['sample']
    fdata = dataname(samplename)
    fname = files(samplename, weighttype, histtype=histtype)
    
    with open(fdata, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%(fdata))
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
        #simnum = hed.attrs['simnum']
        #snapnum = hed.attrs['snapnum']
        #var = hed.attrs['var']
        #ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(fdata, header=headlen, sep='\t', index_col='galaxyid')
    galname_all = pd.read_csv(fname, header=0, sep='\t', index_col='galaxyid')

    galids = np.array(galname_all.index) # galdata may also include non-selected haloes
    
    colsel = binby[0]
    galbins = binby[1]
    numgalbins = len(galbins) - 1
    hists = [None] * numgalbins
    edges = [None] * numgalbins
    edgedata = [None] * numgalbins
    galids_base = [None] * numgalbins
    galids_bin = [[]] * numgalbins
    bgrpns = ['%s-%s/%s_%f-%f'%(combmethod, rbinu, colsel, galbins[i], galbins[i + 1]) for i in range(numgalbins)]
    
    # construct name of summed histogram by removing position specs from a specific one
    outname = galname_all.at[galids[0], 'filename']
    pathparts = outname.split('/')
    namepart = pathparts[-1]
    ext = namepart.split('.')[-1]
    namepart = '.'.join(namepart.split('.')[:-1])
    nameparts = (namepart).split('_')
    outname = []
    for part in nameparts:
        if not (part[0] in ['x', 'y', 'z'] and '-pm' in part):
            outname.append(part)
    outname.append('galcomb')
    outname = '_'.join(outname)
    outname = '/'.join(pathparts[:-1]) + '/' +  outname + '.' + ext
    #print(outname)
    
    # axis data attributes that are allowed to differ between summed histograms
    neqlist = ['number of particles',\
               'number of particles > max value',\
               'number of particles < min value',\
               'number of particles with finite values']
    
    with h5py.File(outname, 'a') as fo:
        igrpn = galname_all.at[galids[0], 'groupname']
        
        for galid in galids:
            selval = galdata_all.at[galid, colsel]
            binind = np.searchsorted(galbins, selval, side='right') - 1
            if binind in [-1, numgalbins]: # halo/stellar mass does not fall into any of the selected ranges
                continue
            
            # retrieve data from this histogram for checks
            igrpn_temp = galname_all.at[galid, 'groupname']   
            if igrpn_temp != igrpn:
                raise RuntimeError('histogram names for galaxyid %i: %s, %i: %s did not match'%(galids[0], igrpn, galid, igrpn_temp))
            ifilen_temp = galname_all.at[galid, 'filename']   
            
            #try:
            with h5py.File(ifilen_temp, 'r') as fit:
                igrp_t = fit[igrpn_temp]
                hist_t = np.array(igrp_t['histogram'])
                if bool(igrp_t['histogram'].attrs['log']):
                    hist_t = 10**hist_t
                #wtsum_t = igrp_t['histogram'].attrs['sum of weights'] # includes stuff outside the maximum radial bin
                edges_t = [np.array(igrp_t['binedges/Axis%i'%i]) for i in range(len(hist_t.shape))]
                edgekeys_t = list(igrp_t.keys())
                edgekeys_t.remove('histogram')
                edgekeys_t.remove('binedges')
                edgedata_t = {}
                for ekey in edgekeys_t: 
                    edgedata_t[ekey] =  {akey: item for akey, item in igrp_t[ekey].attrs.items()}
                    for akey in neqlist:
                        del edgedata_t[ekey][akey]
            #except IOError:
            #    print('Failed to find file for galaxy %i'%(galid))
            #continue
                        
            # run compatibility checks, align/expand edges
            galids_bin[binind].append(galid)
            if edgedata[binind] is None:
                edgedata[binind] = edgedata_t
                galids_base[binind] = galid
            else:
                if not set(edgekeys_t) == set(edgedata[binind].keys()):
                    raise RuntimeError('Mismatch in histogram axis names for galaxyids %i, %i'%(galids_base[binind], galid))
                if not np.all([edgedata_t[ekey][akey] == edgedata[binind][akey] for akey in edgedata_t[ekey].keys()] for ekey in edgekeys_t):
                    raise RuntimeError('Mismatch in histogram axis properties for galaxyids %i, %i'%(galids_base[binind], galid))
            
            # edges are compatible: shift and combine histograms
            # radial bins: only shift if R200c units needed
            try:
                rax = edgedata_t['3Dradius']['histogram axis']
            except KeyError:
                raise KeyError('Could not retrieve histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                    
            if rbinu == 'R200c':
                R200c = galdata_all.at[galid, 'R200c_cMpc']
                R200c *= c.cm_per_mpc * cosmopars['a']              
                edges_t[rax] *= (1. / R200c)
            
            if combmethod == 'add':
                norm_t = 1.
            elif combmethod == 'addnormed-R200c':
                if rbinu != 'R200c':
                    raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                _i = np.where(np.isclose(edges_t[rax], 1.))[0]
                if len(_i) != 1:
                    raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges_t[rax])))
                _i = _i[0]
                _a = range(len(hist_t.shape))
                _s = [slice(None, None, None) for dummy in _a]
                _s[rax] = slice(None, _i, None)
                norm_t = np.sum(hist_t[tuple(_s)])
            elif combmethod == 'addnormed-M200c':
                norm_t = galdata_all.at[galid, 'M200c_Msun']
                norm_t *= c.solar_mass
            elif combmethod == 'addnormed-all':
                norm_t = np.sum(hist_t[_s])
            
            hist_t *= (1. / norm_t)
                
            if hists[binind] is None:
                hists[binind] = hist_t
                edges[binind] = edges_t 
                #print('set hists[%i] to'%binind)
                #print(hist_t)
            else:
                hists[binind], edges[binind] = combine_hists(hists[binind], hist_t, edges[binind], edges_t, rtol=1e-5, atol=1e-8)
                #print('current hist (%i) is'%binind)
                #print(hists[binind])
                
        # store the data
        # don't forget the list of galids (galids_bin, and edgedata)
        #print(hists)
        print('Histogramming finished. Saving data...')
        ogrpn = '%s/%s'%(igrpn, samplename)
        if ogrpn in fo:
            ogrp = fo[ogrpn]
        else:
            ogrp = fo.create_group(ogrpn)
        bgrps = [ogrp.create_group(name) if name not in ogrp\
                 else ogrp[name] for name in bgrpns]
        
        for bind in range(numgalbins):
            bgrp = bgrps[bind]
            hist = hists[bind]
            edge = edges[bind] 
            edged = edgedata[bind]
            galids = galids_bin[bind]
            
            try:
                bgrp.create_dataset('histogram', data=hist)
                bgrp['histogram'].attrs.create('log', False)
                
                bgrp.create_group('binedges')
                for i in range(len(edge)):
                    bgrp['binedges'].create_dataset('Axis%i'%(i), data=edge[i])
                
                for key in edged.keys():
                    bgrp.create_group(key)
                    for skey in edged[key].keys():
                        m3.saveattr(bgrp[key], skey, edged[key][skey])
                    hax = edged[key]['histogram axis']
                    bgrp[key].create_dataset('bins', data=edge[hax])
                
                bgrp.create_dataset('galaxyids', data=np.array(galids_bin[binind]))
            except RuntimeError: # datasets already existed -> delete first
                print('Overwriting group {}/{}'.format(ogrp, bgrp))
                for name in bgrp.keys():
                    del bgrp[name]
                
                bgrp.create_dataset('histogram', data=hist)
                bgrp['histogram'].attrs.create('log', False)
                
                bgrp.create_group('binedges')
                for i in range(len(edge)):
                    bgrp['binedges'].create_dataset('Axis%i'%(i), data=edge[i])
                
                for key in edged.keys():
                    bgrp.create_group(key)
                    for skey in edged[key].keys():
                        m3.saveattr(bgrp[key], skey, edged[key][skey])
                    hax = edged[key]['histogram axis']
                    bgrp[key].create_dataset('bins', data=edge[hax])
                
                bgrp.create_dataset('galaxyids', data=np.array(galids_bin[binind]))
                
    print('Saved data to file {}'.format(outname))  
    print('Main hdf5 group: {}/{}'.format(igrpn, samplename))
    print('and subgroups: ' + (', '.join(['{}'] * len(bgrpns))).format(*tuple(bgrpns)))      
    
def extracthists_ionfrac(samplename='L0100N1504_27_Mh0p5dex_1000',\
              addedges=(0.1, 1.)):
    '''
    generate the histograms for a given sample
    rbinu: used fixed bins in pkpc or in R200c (relevant for stacking)
    
    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    
    binby: column, edges tuple
           which halos to combine into one histogram
    combmethod: how to combine. options are: 
           - 'add': just add all the histograms together
           - 'addnormed-R200c': add histograms normalized by the sum of weights
              within R200c (only if rbinu is 'R200c')
           - 'addnormed-M200c': add histograms normalized by M200c 
             (only equivalent to the previous if it's a Mass-weighted 
             histogram)
           - 'addnormed-all': add histograms normalized by the sum of the 
             histogram to the outermost radial bin
    '''
    rbinu='R200c'
    outname = ol.pdir + 'ionfracs_halos_%s_%s-%s-%s_PtAb.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
    weighttypes_ion = [['oxygen', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8'], ['neon', 'ne8', 'ne9'], ['iron', 'fe17']]
    histtype = 'ionmass' 
    
    if samplename is None:
        samplename = defaults['sample']
    fdata = dataname(samplename)
    fnames_ion = {ion: files(samplename, ion, histtype=histtype) for ls in weighttypes_ion for ion in ls}
    
    with open(fdata, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%(fdata))
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
        #simnum = hed.attrs['simnum']
        #snapnum = hed.attrs['snapnum']
        #var = hed.attrs['var']
        #ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(fdata, header=headlen, sep='\t', index_col='galaxyid')
    galnames_all = {ion: pd.read_csv(fnames_ion[ion], header=0, sep='\t', index_col='galaxyid') for ion in fnames_ion}

    galids = np.array(galnames_all[weighttypes_ion[0][0]].index) # galdata may also include non-selected haloes; galnames galaxyids should match
        
    # axis data attributes that are allowed to differ between summed histograms
    neqlist = ['number of particles',\
               'number of particles > max value',\
               'number of particles < min value',\
               'number of particles with finite values']
    
    with h5py.File(outname, 'a') as fo:
        csp = fo.create_group('Header/cosmopars')
        for key in cosmopars:
            csp.attrs.create(key, cosmopars[key])
        fo.create_dataset('galaxyids', data=galids)
        
        for eltlist in weighttypes_ion:  
            savelist = np.ones((len(galids), len(eltlist) - 1), dtype=np.float) * np.NaN
            for gind in range(len(galids)):
                galid = galids[gind]
                
                tempsum = {}
                for ion in eltlist:
                    # retrieve data from this histogram for checks
                    igrpn_temp = galnames_all[ion].at[galid, 'groupname']   
                    ifilen_temp = galnames_all[ion].at[galid, 'filename']   
                    

                    with h5py.File(ifilen_temp, 'r') as fit:
                        igrp_t = fit[igrpn_temp]
                        hist_t = np.array(igrp_t['histogram'])
        
                        if bool(igrp_t['histogram'].attrs['log']):
                            hist_t = 10**hist_t
                        #wtsum_t = igrp_t['histogram'].attrs['sum of weights'] # includes stuff outside the maximum radial bin
                        edges_t = [np.array(igrp_t['binedges/Axis%i'%i]) for i in range(len(hist_t.shape))]
                        edgekeys_t = list(igrp_t.keys())
                        edgekeys_t.remove('histogram')
                        edgekeys_t.remove('binedges')
                        edgedata_t = {}
                        for ekey in edgekeys_t: 
                            edgedata_t[ekey] =  {akey: item for akey, item in igrp_t[ekey].attrs.items()}
                            for akey in neqlist:
                                del edgedata_t[ekey][akey]
                    try:
                        rax = edgedata_t['3Dradius']['histogram axis']
                    except KeyError:
                        raise KeyError('Could not retrieve histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                            
                    if rbinu == 'R200c':                    
                        R200c = galdata_all.at[galid, 'R200c_cMpc']
                        R200c *= c.cm_per_mpc * cosmopars['a']              
                        edges_t[rax] *= (1. / R200c)
                    
                    try:
                        ind1 = np.where(np.isclose(edges_t[rax], addedges[0]))[0][0]
                    except IndexError:
                        raise RuntimeError('Could not find a histogram edge matching %f for galaxy %i, ion %s'%(addedges[0], galid, ion))
                    try:
                        ind2 = np.where(np.isclose(edges_t[rax], addedges[1]))[0][0]
                    except IndexError:
                        raise RuntimeError('Could not find a histogram edge matching %f for galaxy %i, ion %s'%(addedges[1], galid, ion))
                    
                    addsel = [slice(None, None, None)] * len(edges_t)
                    addsel[rax] = slice(ind1, ind2, None) # left edge ind1 -> start from in ind1, right edge ind2 -> stop after bin ind2 - 1
                    tempsum[ion] = np.sum(hist_t[tuple(addsel)])
                    
                # store the data
                ionsums_galid = np.array([tempsum[ion] / tempsum[eltlist[0]] for ion in eltlist[1:]])
                savelist[gind, :] = ionsums_galid
                
            # don't forget the list of galids (galids_bin, and edgedata)
            #print(hists)
            egrp = fo.create_group('ionfracs_%s'%(eltlist[0]))
            egrp.attrs.create('element', np.string_(eltlist[0]))
            egrp.attrs.create('ions', np.array([np.string_(ion) for ion in eltlist[1:]]))
            egrp.create_dataset('fractions', data=savelist)

def addhalomasses_hists_ionfrac(samplename='L0100N1504_27_Mh0p5dex_1000',\
              addedges=(0.1, 1.)):
    rbinu='R200c'
    filename = ol.pdir + 'ionfracs_halos_%s_%s-%s-%s_PtAb.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
    
    fdata = dataname(samplename)
    
    with open(fdata, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%(fdata))
                else:
                    break
            elif line.startswith('halocat'):
                halocat = line.split(':')[1]
                halocat = halocat.strip()
                headlen += 1
            elif ':' in line or line == '\n':
                headlen += 1
                
    galdata_all = pd.read_csv(fdata, header=headlen, sep='\t', index_col='galaxyid')
    
    with h5py.File(filename, 'a') as fn:
        galaxyids_if = np.array(fn['galaxyids'])
        m200c = np.array(galdata_all['M200c_Msun'][galaxyids_if])
        fn.create_dataset('M200c_Msun', data=m200c)