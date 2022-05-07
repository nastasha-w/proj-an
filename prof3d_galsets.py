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
#import plot_utils as pu # for percentiles_from_histogram

# directory for metadata files
tdir = '/net/luttero/data2/imgs/CGM/3dprof/'

defaults = {'sample': 'L0100N1504_27_Mh0p5dex_1000'}

samples = {'L0100N1504_27_Mh0p5dex_1000': sh.L0100N1504_27_Mh0p5dex_1000,
           'L0100N1504_27_Mstar-Mh0p5dex-match_1000': 
               sh.L0100N1504_27_Mstar_Mhbinmatch_1000,
           'RecalL0025N0752_27_Mh0p5dex_1000': 
               sh.RecalL0025N0752_27_Mh0p5dex_1000}

weighttypes = {'Mass': {'ptype': 'basic', 'quantity': 'Mass'},
               'Volume': {'ptype': 'basic', 'quantity': 'propvol'},
               'gas':   {'ptype': 'basic', 'quantity': 'Mass', 
                         'parttype': '0'},
               'stars': {'ptype': 'basic', 'quantity': 'Mass', 
                         'parttype': '4'},
               'BHs':   {'ptype': 'basic', 'quantity': 'Mass', 
                         'parttype': '5'},
               'DM':    {'ptype': 'basic', 'quantity': 'Mass', 
                         'parttype': '1'},
               }
weighttypes.update({ion: {'ptype': 'Nion', 'ion': ion} for ion in\
                    ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8', 'oxygen',\
                     'ne8', 'ne9', 'neon', 'fe17', 'iron', 'hneutralssh']}) 
for ion in ['oxygen', 'neon', 'iron']:
    weighttypes.update({'gas-%s'%(ion): {'ptype': 'Nion', 'ion': ion, 'parttype': '0'},\
                        'stars-%s'%(ion): {'ptype': 'Nion', 'ion': ion, 'parttype': '4'},\
                        })

lines1 = ['c5r', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r', 'fe18',\
          'fe17-other1', 'fe19', 'o7r', 'o7ix', 'o7iy', 'o7f', 'o8', 'fe17',\
          'c6', 'n7']
lines2 = ['c5r', 'n6r', 'n6-actualr', 'ne9r', 'ne10', 'mg11r', 'mg12',\
          'si13r', 'fe18', 'fe17-other1', 'fe19', 'o7r', 'o7ix', 'o7iy', 'o7f',\
          'o8', 'fe17', 'c6', 'n7']
weighttypes.update({'em-{l}'.format(l=line): {'ptype': 'Luminosity',\
                    'ion': line} \
                    for line in lines2}) 
lines_PS20 = ['C  5      40.2678A', 'C  6      33.7372A', 
              'N  6      29.5343A', 'N  6      28.7870A',
              'N  7      24.7807A', 'O  7      21.6020A',
              'O  7      21.8044A', 'O  7      21.8070A',
              'O  7      22.1012A', 'O  8      18.9709A',
              'Ne 9      13.4471A', 'Ne10      12.1375A',
              'Mg11      9.16875A', 'Mg12      8.42141A',
              'Si13      6.64803A', 'Fe17      17.0510A',
              'Fe17      15.2620A', 'Fe17      16.7760A',
              'Fe17      17.0960A', 'Fe18      16.0720A',
              ]
weighttypes.update({'em-{l}'.format(l=line.replace(' ', '-')): \
                    {'ptype': 'Luminosity', 'ion': line, 'ps20tables': True,
                     'ps20depletion': False} \
                    for line in lines_PS20}) 

def dataname(samplen):
    return tdir + 'halodata_%s.txt'%(samplen)

def files(samplen, weighttype, histtype=None):
    if histtype is None or histtype == 'rprof_rho-T-nion':
        return tdir + 'filenames_%s_%s.txt'%(samplen, weighttype)
    elif histtype == 'ionmass':
        return tdir + 'filenames_%s_%s_ionmass.txt'%(samplen, weighttype)
    elif histtype == 'rprof':
        return tdir + 'filenames_%s_%s_rprof.txt'%(samplen, weighttype)
    elif histtype.startswith('Zprof'):
        return tdir + 'filenames_%s_%s_%s.txt'%(samplen, weighttype, histtype)
    else:
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

def percentiles_from_histogram_handlezeros(histogram, edgesaxis, axis=-1, 
                       percentiles=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
    '''
    get percentiles from the histogram along axis
    edgesaxis are the bin edges along that same axis
    histograms can be weighted by something: this function just solves 
    cumulative distribution == percentiles

    differs from the plot_utils version in its ability to handle zero 
    cumulative values. (outputs NaN for those)
    '''
    percentiles = np.array(percentiles)
    if not np.all(percentiles >= 0.) and np.all(percentiles <= 1.):
        raise ValueError('Input percentiles shoudl be fractions in the range [0, 1]')
    cdists = np.cumsum(histogram, axis=axis, dtype=np.float) 
    sel = list((slice(None, None, None),) * len(histogram.shape))
    sel2 = np.copy(sel)
    sel[axis] = -1
    sel2[axis] = np.newaxis
    zeroweight = cdists[tuple(sel)] == 0.
    zeroweight = zeroweight[tuple(sel2)]
    cdists /= (cdists[tuple(sel)])[tuple(sel2)] # normalised cumulative dist: divide by total along axis
    # bin-edge corrspondence: at edge 0, cumulative value is zero
    # histogram values are counts in cells -> hist bin 0 is what is accumulated between edges 0 and 1
    # cumulative sum: counts in cells up to and including the current one: 
    # if percentile matches cumsum in cell, the percentile value is it's right edges -> edge[cell index + 1]
    # effectively, if the cumsum is prepended by zeros, we get a hist bin matches edge bin matching

    oldshape1 = list(histogram.shape)[:axis] 
    oldshape2 = list(histogram.shape)[axis + 1:]
    newlen1 = int(np.prod(oldshape1))
    newlen2 = int(np.prod(oldshape2))
    axlen = histogram.shape[axis]
    cdists = cdists.reshape((newlen1, axlen, newlen2))
    zeroweight = zeroweight.reshape((newlen1, 1, newlen2))
    cdists = np.append(np.zeros((newlen1, 1, newlen2)), cdists, axis=1)
    cdists[:, -1, :] = 1. # should already be true, but avoids fp error issues

    leftarr  = cdists[np.newaxis, :, :, :] <= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    rightarr = cdists[np.newaxis, :, :, :] >= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    _zwsel = np.repeat(zeroweight, axlen + 1, axis=1) # appended zeros
    leftarr[:, _zwsel] = True
    rightarr[:, _zwsel] = True

    leftbininds = np.array([[[ np.max(np.where(leftarr[pind, ind1, :, ind2])[0]) \
                               for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
    # print leftarr.shape
    # print rightarr.shape
    rightbininds = np.array([[[np.min(np.where(rightarr[pind, ind1, :, ind2])[0]) \
                               for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
    # if left and right bins are the same, effictively just choose one
    # if left and right bins are separated by more than one (plateau edge), 
    #    this will give the middle of the plateau
    lweights = np.array([[[ (cdists[ind1, rightbininds[pind, ind1, ind2], ind2] - percentiles[pind]) \
                            / ( cdists[ind1, rightbininds[pind, ind1, ind2], ind2] - cdists[ind1, leftbininds[pind, ind1, ind2], ind2]) \
                            if rightbininds[pind, ind1, ind2] != leftbininds[pind, ind1, ind2] \
                            else 1.
                           for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
                
    outperc = lweights * edgesaxis[leftbininds] + (1. - lweights) * edgesaxis[rightbininds]
    outperc[:, zeroweight.reshape(newlen1, newlen2)] = np.NaN
    outperc = outperc.reshape((len(percentiles),) + tuple(oldshape1 + oldshape2))
    return outperc

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
    weighttype gas-nH-[element] gets input as gas for the weight itself, but 
           an nH cut is used instead of SFR to determine ISM membership later 
           on (0.1 cc)
           nHm2 instead: used 0.01 cc
           nHorSF: uses 0.1 cc cut or SF to define the ISM
    '''
    logname = files(samplename, weighttype, histtype=axdct)
    
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
            
            name_append = '_%s_snapdata_CorrPartType'%rbinu
            if 'nH' in weighttype.split('-'):
                axesdct.append({'ptype': 'Niondens', 'ion': 'hydrogen'})
                # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
                minval = 10**-1 # approx SFR threashold, but Z-independent 
                nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
                weighttype = weighttype.split('-')
                weighttype.remove('nH')
                weighttype = '-'.join(weighttype)
                logax = [False, False, True]
            elif 'nHm2' in weighttype.split('-'):
                axesdct.append({'ptype': 'Niondens', 'ion': 'hydrogen'})
                # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
                minval = 10**-2 # approx SFR threashold, but Z-independent 
                nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
                weighttype = weighttype.split('-')
                weighttype.remove('nHm2')
                weighttype = '-'.join(weighttype)
                logax = [False, False, True]
                name_append  = name_append + '_m2lim'
            elif 'nHorSF' in weighttype.split('-'):
                axesdct.append({'ptype': 'Niondens', 'ion': 'hydrogen'})
                # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
                minval = 10**-1 # approx SFR threashold, but Z-independent 
                nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
                weighttype = weighttype.split('-')
                weighttype.remove('nHorSF')
                weighttype = '-'.join(weighttype)
                
                axesdct.append({'ptype': 'basic', 'quantity': 'StarFormationRate'})
                # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
                minval = 2**-149 * c.solar_mass / c.sec_per_year 
                nonrbins.append(np.array([-np.inf, minval, np.inf]))
                
                logax = [False, False, False, True]
            else:
                axesdct.append({'ptype': 'basic', 'quantity': 'StarFormationRate'})
                # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
                minval = 2**-149 * c.solar_mass / c.sec_per_year 
                nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
                logax = [False, False, True]
                
            axesdct.append({'ptype': 'basic', 'quantity': 'Temperature', 'excludeSFR': False})
            Tbins = np.array([-np.inf, 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., np.inf])
            nonrbins.append(Tbins)
            
        else:
            logax = [False]
        
        
             
    with open(logname, 'w') as fdoc:
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
 
           
def genhists_luminositydist(samplename='L0100N1504_27_Mh0p5dex_1000',\
                            rbinu='R200c', idsel=None,\
                            weighttype=None,\
                            logM200min=10.0, axdct='nrprof'):
    '''
    generate the histograms for a given sample
    rbins: used fixed bins in pkpc or in R200c (relevant for stacking)
    axdct: 'Trprof': T profile
           'nrprof': n_H profile
           'Zrprof': Z profile
           'pds':    n_H, T, Z in coarse radial bins
           '{elt}-rprof': Smoothed Z profile for an element (for Mass- and 
           Volume weighttypes)
    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    weighttype: determines the weighting of the histogram
           em-{line}: luminosity weighting
    '''
    if axdct == 'pds':
        raise NotImplementedError('pds is just a stub option for future dev.')
    logname = files(samplename, weighttype, histtype=axdct)
    
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
        simnum = hed.attrs['simnum'].decode()
        snapnum = int(hed.attrs['snapnum'])
        var = hed.attrs['var'].decode()
        #ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(fin, header=headlen, sep='\t', index_col='galaxyid')
    if idsel is not None:
        if isinstance(idsel, slice):
            galaxyids = np.array(galdata_all.index)[idsel]
        else:
            galaxyids = idsel
    else:
        galaxyids = np.array(galdata_all.index)
    
    axesdct = [{'ptype': 'coords', 'quantity': 'r3D'},\
              ]
    nonrbins = []
    Zbins = np.array([-np.inf] + list(np.arange(-38.0, 0.05, 0.1)) + [np.inf]) 
    
    if weighttype.startswith('em'):
        # gas particle min. SFR:
        # min. gas density = 57.7 * rho_mean (cosmic mean, using rho_matter * omegab to e safe) = 7.376116060910138e-30
        # SFR = m_g * A * (M_sun / pc^2)^n * (gamma / G * f_g * P) * (n - 1) / 2
        # gamma = 5/3, G = newton constant, f_g = 1 (gas fraction), P = total pressure
        # A = 1.515 × 10−4 M⊙ yr−1 kpc−2, n = 1.4 (n = 2 at nH > 10^3 cm^-3)
        
        name_append = ''
        _axdct = axdct
        if axdct == 'nrprof':
            axesdct.append({'ptype': 'Niondens', 'ion': 'hydrogen'})
            nonrbins.append(0.1)
        elif axdct == 'Trprof':
            axesdct.append({'ptype': 'basic', 'quantity': 'Temperature'})
            nonrbins.append(0.1)
        elif axdct == 'Zrprof':
            line = '-'.join(weighttype.split('-')[1:])
            try:
                elt = ol.elements_ion[line]
            except KeyError as err: # PS20 line
                try: 
                    tab = m3.linetable_PS20(line.replace('-', ' '), 0.0, 
                                           emission=False)
                    elt = tab.element.lower()
                except ValueError as err2:
                    print('Trying to interpret {} as a PS20 line'.format(line))
                    print(err2)
                    raise err
            qty = 'SmoothedElementAbundance/{elt}'.format(elt=string.capwords(elt))
            axesdct.append({'ptype': 'basic', 'quantity': qty})
            nonrbins.append(Zbins)
        
        if axdct == 'pds':
            pass
        else:
            # log SF/nonSF gas
            # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
            minval = 2**-149 * c.solar_mass / c.sec_per_year 
            nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
            axesdct.append({'ptype': 'basic', 'quantity': 'StarFormationRate'})
            logax = [False, True, False]
    else: # Mass- or Volume-weighted
        #print('Using M/V case')
        name_append = ''
        elt = axdct.split('-')
        if len(elt) == 1:
            _axdct = axdct
        else:
            _axdct = 'Zrprof'
            elt = string.capwords(elt[0])
            #print('Using Zprof, elt {}'.format(elt)) 
        
        if _axdct == 'nrprof':
            axesdct.append({'ptype': 'Niondens', 'ion': 'hydrogen'})
            nonrbins.append(0.1)
        elif _axdct == 'Trprof':
            axesdct.append({'ptype': 'basic', 'quantity': 'Temperature'})
            nonrbins.append(0.1)
        elif _axdct == 'Zrprof':
            qty = 'SmoothedElementAbundance/{elt}'.format(elt=elt)
            #print('Using qty {}'.format(qty))
            axesdct.append({'ptype': 'basic', 'quantity': qty})
            nonrbins.append(Zbins)
        
        if _axdct == 'pds':
            pass
        else:
            # log SF/nonSF gas
            # minimum float32 value -> cgs units; much smaller than any SFR in the 12 Mpc box
            minval = 2**-149 * c.solar_mass / c.sec_per_year 
            nonrbins.append(np.array([-np.inf, minval, np.inf])) # calculate minimum SFR possible in Eagle, use as minimum bin for ISM value
            axesdct.append({'ptype': 'basic', 'quantity': 'StarFormationRate'})
            logax = [False, True, False]
             
    with open(logname, 'w') as fdoc:
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
                if axdct == 'pds':
                    pass
                else:
                    rbins = np.array([0., 5., 10., 20., 30., 40., 50., 60.,
                                      70., 80., 90., 100., 125., 150., 175.,
                                      200., 250., 300., 350., 400., 450.,
                                      500.]) \
                            * 1e-3 * c.cm_per_mpc
                    if rbins[-1] < R200c:
                        rbins = np.append(rbins, [R200c])
            else:
                if axdct == 'pds':
                    pass
                else:
                    rbins = np.array([0., 0.01, 0.02, 0.05, 0.1, 0.15, 0.2,
                                      0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                                      1.25, 1.50, 2., 2.5, 3., 3.5, 4.]) \
                            * R200c * c.cm_per_mpc * cosmopars['a']
            cen = [Xcom, Ycom, Zcom]
            L_x, L_y, L_z = (2. * rbins[-1] / c.cm_per_mpc / cosmopars['a'],) * 3
            
            axbins =  [rbins] + nonrbins
            
            args = (weighttypes[weighttype]['ptype'], simnum, snapnum, var, axesdct,)
            kwargs = {'simulation': 'eagle', 'excludeSFR': 'T4', 'abunds': 'Sm',\
                      'parttype': '0',\
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
                        # throw out any remaining incomplete Z range profiles
                        if _axdct == 'Zprof':
                            keys = list(fo_t[outname[1]].keys())
                            zkey = np.where(['ElementAbundance' in key for key in keys])[0]
                            if len(zkey) == 0:
                                # something weird here; delete
                                del fo_t[outname[1]]
                            else:
                                zkeys = np.array(keys)[zkey]
                                missing = False
                                for key in zkeys:
                                    zbins = fo_t[outname[1]][key]['bins'][:]
                                    if not (-np.inf == zbins[0] and np.inf == zbins[-1]):
                                        missing = True
                                if missing:
                                    print('Removing erroneous hist {}'.format(outname))
                                    del fo_t[outname[1]]
                                else:
                                    alreadyexists = True
                        else:
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

def extracthists_luminosity(samplename='L0100N1504_27_Mh0p5dex_1000',
              addedges=(0.0, 1.), logM200min=11.0, lineset='SBlines'):
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
    lineset: which set of lines to use. options are
           - 'SBlines': lines from Serena Bertone's tables, for paper 3
           - 'PS20lines': lines from the Ploeckinger & Schaye (2020) tables
             for paper 3
    '''
    rbinu = 'R200c'
    
    if lineset == 'SBlines':
        outname = ol.pdir + 'luminosities_halos_%s_%s-%s-%s_SmAb.hdf5'%(samplename,
                                        str(addedges[0]), str(addedges[1]), rbinu)
        weighttypes = ['em-{l}'.format(l=line.replace(' ', '-')) \
                       for line in lines2]
    elif lineset == 'PS20lines':
        base = 'luminosities_PS20_depletion-F_halos_%s_%s-%s-%s_SmAb.hdf5'
        outname = ol.pdir + base%(samplename, 
                                  str(addedges[0]), str(addedges[1]), rbinu)
        weighttypes = ['em-{l}'.format(l=line.replace(' ', '-')) \
                       for line in lines_PS20]
    histtype = 'nrprof' 
    
    if samplename is None:
        samplename = defaults['sample']
    fdata = dataname(samplename)
    fnames_line = {line: files(samplename, line, histtype=histtype) for line in weighttypes}
    
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
    galnames_all = {line: pd.read_csv(fnames_line[line], header=0, sep='\t', index_col='galaxyid')\
                    for line in fnames_line}

    galids = np.array(galnames_all[weighttypes[0]].index) # galdata may also include non-selected haloes; galnames galaxyids should match
    galids = galids[galdata_all.loc[galids, 'M200c_Msun'] > 10**logM200min]
    
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
        
        savelist = np.ones((len(galids), len(weighttypes), 2), dtype=np.float64) * np.NaN # initialize as NaN
        
        for li, line in enumerate(weighttypes):  
            
            for gind in range(len(galids)):
                galid = galids[gind]
                # retrieve data from this histogram for checks
                igrpn_temp = galnames_all[line].at[galid, 'groupname']   
                ifilen_temp = galnames_all[line].at[galid, 'filename']  
                #if line in ['n6-actualr', 'ne10']:
                #    ifilen_temp = ifilen_temp.replace('test3.5', 'test3.6')
                #else:
                #    ifilen_temp = ifilen_temp.replace('test3.6', 'test3.5')
                
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
                try:
                    sfax = edgedata_t['StarFormationRate_T4EOS']['histogram axis']
                except KeyError:
                    raise KeyError('Could not retrieve SFR axis for galaxy %i, file %s'%(galid, ifilen_temp))
                        
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
                addsel_nsf = list(np.copy(addsel))
                addsel_nsf[sfax] = slice(0, 1, None)
                addsel_sf = list(np.copy(addsel))
                addsel_sf[sfax] = slice(1, 2, None)
                tempsum_nsf = np.sum(hist_t[tuple(addsel_nsf)])
                tempsum_sf = np.sum(hist_t[tuple(addsel_sf)])
                
                savelist[gind, li, 0] = tempsum_nsf
                savelist[gind, li, 1] = tempsum_sf
                
        # don't forget the list of galids (galids_bin, and edgedata)
        #print(hists)
        egrp = fo
        lines = ['-'.join(weight.split('-')[1:]) for weight in weighttypes]
        egrp.attrs.create('lines', np.array([np.string_(line) for line in lines]))
        ds = egrp.create_dataset('luminosities', data=savelist)
        ds.attrs.create('units', np.string_('erg/s'))
        ds.attrs.create('axis0', np.string_('galaxyid'))
        ds.attrs.create('axis1', np.string_('line'))
        ds.attrs.create('axis2', np.array([np.string_('non-star-forming'),\
                                           np.string_('star-forming')]))

def extract_totweighted_luminosity(samplename='L0100N1504_27_Mh0p5dex_1000',
              addedges=(0.0, 1.), weight='Luminosity', logM200min=11.0,
              lineset='SBtables'):
    '''
    generate the histograms for a given sample
    rbinu: used fixed bins in pkpc or in R200c (relevant for stacking)
     
    weight:  'Luminosity': luminosities for the different lines
             'Mass':       gas mass (incl. the different metals for Z)
             'Volume':     gas volume (incl. the different metals for Z)
    lineset: 'SBtables' or 'PS20tables'
    '''
    rbinu = 'R200c'
    if weight == 'Luminosity':
        if lineset == 'SBlines':
            base = 'luminosity-weighted-nH-T-Z_halos_%s_%s-%s-%s_SmAb.hdf5'
            outname = ol.pdir + base%(samplename, str(addedges[0]),
                                      str(addedges[1]), rbinu)
            weighttypes = ['em-{l}'.format(l=line) for line in lines2]
        elif lineset == 'PS20lines':
            base = 'luminosity-weighted-nH-T-Z_PS20_depletion-F_halos_%s_%s-%s-%s_SmAb.hdf5'
            outname = ol.pdir + base%(samplename, 
                                      str(addedges[0]), str(addedges[1]), rbinu)
            weighttypes = ['em-{l}'.format(l=line.replace(' ', '-')) \
                           for line in lines_PS20]
        elif lineset == 'convtest':
            base = 'luminosity-weighted-nH-T-Z_SB-PS20_depletion-F_halos_%s_%s-%s-%s_SmAb.hdf5'
            outname = ol.pdir + base%(samplename, 
                                      str(addedges[0]), str(addedges[1]), rbinu)
            _lines = ['o7r', 'o8', 'Fe17      17.0510A', 'si13r']
            weighttypes = ['em-{l}'.format(l=line.replace(' ', '-')) \
                           for line in _lines]
            
        histtypes = ['Zrprof', 'nrprof', 'Trprof'] 
    elif weight in ['Mass', 'Volume']:
        outname = ol.pdir + '{weight}-weighted-nH-T-Z_halos_%s_%s-%s-%s_SmAb.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
        outname = outname.format(weight=weight.lower())
        weighttypes = [weight]
        metals = ['Carbon', 'Nitrogen', 'Oxygen', 'Neon', 'Magnesium',\
                  'Iron', 'Silicon']
        axdcts = ['{elt}-rprof'.format(elt=elt) for elt in metals]
        histtypes = ['nrprof', 'Trprof'] + axdcts 
    
    if samplename is None:
        samplename = defaults['sample']
    fdata = dataname(samplename)
    fnames_histtype_line = {histtype: {line: files(samplename, line, histtype=histtype)\
                                       for line in weighttypes}\
                            for histtype in histtypes}
    
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
    galnames_all = {histtype: {line: pd.read_csv(fnames_histtype_line[histtype][line],
                                                 header=0, sep='\t', index_col='galaxyid')\
                               for line in fnames_histtype_line[histtype]}\
                    for histtype in histtypes}

    galids = np.array(galnames_all[histtypes[0]][weighttypes[0]].index) # galdata may also include non-selected haloes; galnames galaxyids should match
    galids = galids[galdata_all.loc[galids, 'M200c_Msun'] > 10**logM200min]
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
        
        savelist = np.zeros((len(galids), len(weighttypes), 2), dtype=np.float64) / 0. # initialize as NaN
        
        for histtype in histtypes:
            print('starting {}'.format(histtype))
            savetot = histtype == 'nrprof' # one that exists for all the weights
            if savetot:
                savelist_tot = np.zeros((len(galids), len(weighttypes), 2), dtype=np.float64) / 0. # initialize as NaN
            for li, line in enumerate(weighttypes): 
                print('starting {}'.format(line))
                for gind in range(len(galids)):
                    galid = galids[gind]
                    
                    # retrieve data from this histogram for checks
                    igrpn_temp = galnames_all[histtype][line].at[galid, 'groupname']   
                    ifilen_temp = galnames_all[histtype][line].at[galid, 'filename']   
                    #if line in ['n6-actualr', 'ne10']:
                    #    ifilen_temp = ifilen_temp.replace('test3.5', 'test3.6')
                    #else:
                    #    ifilen_temp = ifilen_temp.replace('test3.6', 'test3.5')
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
                    try:
                        sfax = edgedata_t['StarFormationRate_T4EOS']['histogram axis']
                    except KeyError:
                        raise KeyError('Could not retrieve SFR axis for galaxy %i, file %s'%(galid, ifilen_temp))
                    
                    if histtype == 'Trprof':
                        axname_toaverage = 'Temperature_T4EOS'
                    elif histtype == 'nrprof':
                        axname_toaverage = 'Niondens_hydrogen_SmAb_T4EOS'
                        if axname_toaverage not in edgedata_t:
                            siontab = '_PS20-iontab-UVB-dust1-CR1-G1-shield1_depletion-F'
                            axname_toaverage = 'Niondens_hydrogen_SmAb{}_T4EOS'
                            axname_toaverage = axname_toaverage.format(siontab)
                        
                    else:
                        base = 'SmoothedElementAbundance-{elt}_T4EOS'
                        if histtype == 'Zrprof':
                            _line = '-'.join(line.split('-')[1:]) # 'em-o7r', 'em-fer17-other1'
                            try:
                                elt = ol.elements_ion[_line]
                            except KeyError as err: # PS20 line
                                try: 
                                    tab = m3.linetable_PS20(_line.replace('-', ' '), 0.0, 
                                                            emission=False)
                                    elt = tab.element.lower()
                                except ValueError as err2:
                                    print('Trying to interpret {} as a PS20 line'.format(line))
                                    print(err2)
                                    raise err
                        else:
                            elt = histtype.split('-')[0] # 'Carbon-rprof'
                        #print(elt)
                        axname_toaverage = base.format(elt=string.capwords(elt))
                        
                    try:
                        avax = edgedata_t[axname_toaverage]['histogram axis']
                        avlog = bool(edgedata_t[axname_toaverage]['log'])
                    except KeyError:
                        raise KeyError('Could not retrieve axis to average %s for galaxy %i, file %s'%(axname_toaverage, galid, ifilen_temp))
                    if rbinu == 'R200c':                    
                        R200c = galdata_all.at[galid, 'R200c_cMpc']
                        R200c *= c.cm_per_mpc * cosmopars['a']              
                        edges_t[rax] *= (1. / R200c)
                    
                    try:
                        ind1 = np.where(np.isclose(edges_t[rax], 
                                                   addedges[0]))[0][0]
                    except IndexError:
                        raise RuntimeError('Could not find a histogram edge matching %f for galaxy %i, ion %s'%(addedges[0], galid, ion))
                    try:
                        ind2 = np.where(np.isclose(edges_t[rax], 
                                                   addedges[1]))[0][0]
                    except IndexError:
                        raise RuntimeError('Could not find a histogram edge matching %f for galaxy %i, ion %s'%(addedges[1], galid, ion))
                    
                    addsel = [slice(None, None, None)] * len(edges_t)
                    addsel[rax] = slice(ind1, ind2, None) # left edge ind1 -> start from in ind1, right edge ind2 -> stop after bin ind2 - 1
                    addsel_nsf = list(np.copy(addsel))
                    addsel_nsf[sfax] = slice(0, 1, None)
                    addsel_sf = list(np.copy(addsel))
                    addsel_sf[sfax] = slice(1, 2, None)
                    tempsum_nsf = np.sum(hist_t[tuple(addsel_nsf)])
                    tempsum_sf = np.sum(hist_t[tuple(addsel_sf)])
                        
                    if savetot:
                        savelist_tot[gind, li, 0] = tempsum_nsf
                        savelist_tot[gind, li, 1] = tempsum_sf
                    
                    avedges = edges_t[avax]
                    avcens = 0.5 * (avedges[:-1] + avedges[1:])
                    if avedges[-1] == np.inf:
                        avcens[-1] = avedges[-2] + \
                                     0.5 * (avedges[-2] - avedges[-3])
                    if avlog:
                        avcens = 10**avcens
                    #sumaxes = list(range(len(hist_t.shape)))
                    #sumaxes.remove(avax)
                    #sumaxes = tuple(sumaxes)
                    shapeav = [np.newaxis] * len(hist_t.shape)
                    shapeav[avax] = slice(None, None, None)
                    shapeav = tuple(shapeav)
                    av_nsf = np.sum(hist_t[tuple(addsel_nsf)] \
                                    * avcens[shapeav]\
                                    ) / tempsum_nsf
                    av_sf  = np.sum(hist_t[tuple(addsel_sf)] \
                                    * avcens[shapeav]\
                                    ) / tempsum_sf  
                    if avlog:
                        av_nsf = np.log10(av_nsf)
                        av_sf  = np.log10(av_sf)
                    #print(tempsum_sf)
                    #print(tempsum_nsf)
                    #print(av_sf)
                    #print(av_nsf)
                    #print('')
                    savelist[gind, li, 0] = av_nsf
                    savelist[gind, li, 1] = av_sf
                    
            # don't forget the list of galids (galids_bin, and edgedata)
            #print(hists)
            if savetot:
                egrp = fo
                if weight == 'Luminosity':
                    lines = ['-'.join(weight.split('-')[1:]) \
                             for weight in weighttypes]
                    units = 'erg/s'
                else:
                    lines = weighttypes
                    if weight == 'Mass':
                        units = 'g'
                    elif weight == 'Volume':
                        units = 'cm**-3'
                egrp.attrs.create('weight', np.array([np.string_(line) \
                                                      for line in lines]))
                
                ds = egrp.create_dataset('weight_total', data=savelist_tot)
                ds.attrs.create('units', np.string_(units))
                ds.attrs.create('axis0', np.string_('galaxyid'))
                ds.attrs.create('axis1', np.string_('weight'))
                ds.attrs.create('axis2', 
                                np.array([np.string_('non-star-forming'),
                                          np.string_('star-forming')]))
        
            if histtype == 'Trprof':
                dsname = 'Temperature_T4EOS'
                units = 'log10 K'
            elif histtype == 'nrprof':
                dsname = 'Niondens_hydrogen_SmAb'
                units = 'log10 cm**-3'
            else:
                base = 'SmoothedElementAbundance-{elt}'
                units = 'log10 mass fraction'
                if histtype == 'Zrprof':
                    #line = '-'.join(weight.split('-')[1:]) # 'em-o7r', 'em-fer17-other1'
                    #elt = string.capwords(ol.elements_ion[line])
                    elt = 'parent'
                else:
                    elt = histtype.split('-')[0] # 'Carbon-rprof'
                dsname = base.format(elt=elt)
            egrp = fo
            if weight == 'Luminosity':
                lines = ['-'.join(weight.split('-')[1:]) \
                         for weight in weighttypes]
            else:
                lines = weighttypes
            #egrp.attrs.create('weight', np.array([np.string_(line) for line in lines]))
            
            ds = egrp.create_dataset(dsname, data=savelist)
            ds.attrs.create('units', np.string_(units))
            ds.attrs.create('axis0', np.string_('galaxyid'))
            ds.attrs.create('axis1', np.string_('weight'))
            ds.attrs.create('axis2', np.array([np.string_('non-star-forming'),
                                               np.string_('star-forming')]))
        
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
        
def extracthists_massdist(samplename='L0100N1504_27_Mh0p5dex_1000',\
              addedges=(0.0, 1.), nHcut=False, nHm2=False, nHorSF=False):    
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
    nHcut: use the nH limit (10^-1 cm^-3) instead of SFR to define the ISM
    nHm2:  use the nH limit (10^-2 cm^-3) instead of SFR to define the ISM
    nHorSF: use nH > 10^-1 cm^-3 OR SF to define the ISM
    '''
    
    rbinu='R200c'
    if nHcut:
        outname = ol.pdir + 'massdist-baryoncomp_halos_%s_%s-%s-%s_PtAb_nHcut.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
        weighttypes_ion = {'Mass': ['gas-nH', 'stars', 'BHs', 'DM']} 
        weighttypes_ion.update({elt: ['gas-nH-%s'%(elt), 'stars-%s'%(elt)] for elt in ['oxygen', 'neon', 'iron']})
    elif nHm2:
        outname = ol.pdir + 'massdist-baryoncomp_halos_%s_%s-%s-%s_PtAb_nHm2cut.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
        weighttypes_ion = {'Mass': ['gas-nHm2', 'stars', 'BHs', 'DM']} 
        weighttypes_ion.update({elt: ['gas-nHm2-%s'%(elt), 'stars-%s'%(elt)] for elt in ['oxygen', 'neon', 'iron']})
    elif nHorSF:
        outname = ol.pdir + 'massdist-baryoncomp_halos_%s_%s-%s-%s_PtAb_nHorSFcut.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
        weighttypes_ion = {'Mass': ['gas-nHorSF', 'stars', 'BHs', 'DM']} 
        weighttypes_ion.update({elt: ['gas-nHorSF-%s'%(elt), 'stars-%s'%(elt)] for elt in ['oxygen', 'neon', 'iron']})
    else:
        outname = ol.pdir + 'massdist-baryoncomp_halos_%s_%s-%s-%s_PtAb.hdf5'%(samplename, str(addedges[0]), str(addedges[1]), rbinu)
        weighttypes_ion = {'Mass': ['gas', 'stars', 'BHs', 'DM']} 
        weighttypes_ion.update({elt: ['gas-%s'%(elt), 'stars-%s'%(elt)] for elt in ['oxygen', 'neon', 'iron']})
    gas_tlims_add = [-np.inf, 5., 5.5, 7., np.inf]
    histtype = 'rprof'

    if samplename is None:
        samplename = defaults['sample']
    fdata = dataname(samplename)
    weights_all = [weight for key in weighttypes_ion for weight in weighttypes_ion[key]]
    fnames_wt = {weight: files(samplename, weight, histtype=histtype) for weight in weights_all}
    
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
    galnames_all = {weight: pd.read_csv(fnames_wt[weight], header=0, sep='\t', index_col='galaxyid') for weight in weights_all}

    galids = np.array(galnames_all[weighttypes_ion['Mass'][0]].index) # galdata may also include non-selected haloes; galnames galaxyids should match
        
    # axis data attributes that are allowed to differ between summed histograms
    neqlist = ['number of particles',\
               'number of particles > max value',\
               'number of particles < min value',\
               'number of particles with finite values']

    # temperature and SF/nonSF gas subset labels
    tlbl = ['T-%s-%s'%(gas_tlims_add[i], gas_tlims_add[i + 1]) for i in range(len(gas_tlims_add) - 1)]
    if nHcut or nHm2:
        slbl = ['CGM', 'ISM']
    else:
        slbl = ['nonSF', 'SF']
    if nHorSF:
        nlbl = ['lodens', 'hidens']
        
    with h5py.File(outname, 'a') as fo:
        csp = fo.create_group('Header/cosmopars')
        for key in cosmopars:
            csp.attrs.create(key, cosmopars[key])
        fo.create_dataset('galaxyids', data=galids)
        m200c = np.array(galdata_all['M200c_Msun'][galids])
        fo.create_dataset('M200c_Msun', data=m200c)
        
        for wtkey in weighttypes_ion:  
            wtlist = weighttypes_ion[wtkey]
            storeorder = list(np.copy(wtlist)) 
            for weight in wtlist:
                if 'gas' in weight:
                    if nHorSF:
                        storeorder += ['_'.join([weight, sl, nl, tl]) for sl in slbl for nl in nlbl for tl in tlbl]
                    else:
                        storeorder += ['_'.join([weight, sl, tl]) for sl in slbl for tl in tlbl]
            storeorder.sort()
            savelist = np.ones((len(galids), len(storeorder)), dtype=np.float) * np.NaN
            for gind in range(len(galids)):
                galid = galids[gind]
                
                tempsum = {}
                for wt in wtlist:
                    # retrieve data from this histogram for checks
                    try:
                        igrpn_temp = galnames_all[wt].at[galid, 'groupname']   
                        ifilen_temp = galnames_all[wt].at[galid, 'filename']   
                    except ValueError as err:
                        print('GalaxyID %s, wtkey %s, '%(galid, wtkey))
                        raise err
                        
                    with h5py.File(ifilen_temp, 'r') as fit:
                        igrp_t = fit[igrpn_temp]
                        hist_t = np.array(igrp_t['histogram'])
        
                        if bool(igrp_t['histogram'].attrs['log']):
                            hist_t = 10**hist_t
                        #wtsum_t = igrp_t['histogram'].attrs['sum of wehisttype == 'rprof'ights'] # includes stuff outside the maximum radial bin
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
                        raise KeyError('Could not retrieve 3D radius histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                            
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
                    
                    if 'gas' in wt: # extra splits on SF/non-SF and gas temperature
                        try:
                            tax = edgedata_t['Temperature_wiEOS']['histogram axis']
                        except KeyError:
                            raise KeyError('Could not retrieve Temperature histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                        tinds = np.where(np.isclose(edges_t[tax][:, np.newaxis], np.array(gas_tlims_add)[np.newaxis, :]))[0]
                        
                        if nHcut or nHm2:
                            try:
                                sax = edgedata_t['Niondens_hydrogen_PtAb_T4EOS']['histogram axis']
                            except KeyError:
                                raise KeyError('Could not retrieve nH histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                            if nHcut:
                                sinds = np.where(np.isclose(edges_t[sax], [-np.inf, 0.1, np.inf]))[0] 
                            else:
                                sinds = np.where(np.isclose(edges_t[sax], [-np.inf, 0.01, np.inf]))[0] 
                        else:
                            try:
                                sax = edgedata_t['StarFormationRate_T4EOS']['histogram axis']
                            except KeyError:
                                raise KeyError('Could not retrieve StarFormationRate histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                            sinds = np.where(np.isclose(edges_t[sax], [-np.inf, 0., np.inf]))[0] # middle value is minimally > 0., but within isclose range
                        if nHorSF:
                            try:
                                nax = edgedata_t['Niondens_hydrogen_PtAb_T4EOS']['histogram axis']
                            except KeyError:
                                raise KeyError('Could not retrieve nH histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
                            ninds = np.where(np.isclose(edges_t[nax], [-np.inf, 0.1, np.inf]))[0] 
                                              
                        addsel_base = [slice(None, None, None)] * len(edges_t)
                        addsel_base[rax] = slice(ind1, ind2, None) # left edge ind1 -> start from in ind1, right edge ind2 -> stop after bin ind2 - 1
                        if nHorSF:
                            for ti in range(len(tlbl)):
                                t1 = tinds[ti]
                                t2 = tinds[ti + 1]
                                tl = tlbl[ti]
                                for si in range(len(slbl)):
                                    s1 = sinds[si]
                                    s2 = sinds[si + 1]
                                    sl = slbl[si]
                                    for ni in range(len(nlbl)):
                                        n1 = ninds[ni]
                                        n2 = ninds[ni + 1]
                                        nl = nlbl[ni]
                                        addsel = list(np.copy(addsel_base))
                                        addsel[tax] = slice(t1, t2, None)
                                        addsel[sax] = slice(s1, s2, None)
                                        addsel[nax] = slice(n1, n2, None)
                                        storekey = '_'.join([wt, sl, nl, tl])
                                        tempsum[storekey] = np.sum(hist_t[tuple(addsel)])
                        else:
                            for ti in range(len(tlbl)):
                                t1 = tinds[ti]
                                t2 = tinds[ti + 1]
                                tl = tlbl[ti]
                                for si in range(len(slbl)):
                                    s1 = sinds[si]
                                    s2 = sinds[si + 1]
                                    sl = slbl[si]
                                    addsel = list(np.copy(addsel_base))
                                    addsel[tax] = slice(t1, t2, None)
                                    addsel[sax] = slice(s1, s2, None)
                                    storekey = '_'.join([wt, sl, tl])
                                    tempsum[storekey] = np.sum(hist_t[tuple(addsel)])
                                
                        
                    # also store total gas for conistency checks etc.
                    addsel = [slice(None, None, None)] * len(edges_t)
                    addsel[rax] = slice(ind1, ind2, None) # left edge ind1 -> start from in ind1, right edge ind2 -> stop after bin ind2 - 1
                    tempsum[wt] = np.sum(hist_t[tuple(addsel)])
                    
                # store the data
                wtsums_galid = np.array([tempsum[wt] for wt in storeorder])
                savelist[gind, :] = wtsums_galid
                
            # don't forget the list of galids (galids_bin, and edgedata)
            #print(hists)
            egrp = fo.create_group('massdist_%s'%(wtkey))
            egrp.attrs.create('categories', np.array([np.string_(wt) for wt in storeorder]))
            egrp.create_dataset('mass', data=savelist)
            egrp['mass'].attrs.create('info', np.string_('total mass in each component in the given radial range'))
            egrp['mass'].attrs.create('units', np.string_('g'))

def deletesets(filen):
    '''
    if something has gone wrong: delete the datasets in one of the listed files
    '''
    galname_all = pd.read_csv(filen, header=0, sep='\t', index_col='galaxyid')
    for gid in galname_all.index:
        filen = galname_all.at[gid, 'filename']
        groupn = galname_all.at[gid, 'groupname']
        with h5py.File(filen, 'a') as fi:
            del fi[groupn]
    
# tested output files with one set of 10 galaxies 
# (some weren't in the mass selection)
# tested percentiles with one of them, inclSFgas True option only
# tested for cumul and Temperature_T4EOS percaxis, histtypes
# nrprof and Trprof respectively
# minrad_use is untested, copied from plotting stuff
def extract_indiv_radprof(percaxis=None, samplename=None, idsel=None, 
                          weighttype='Mass', histtype='rprof_rho-T-nion',
                          binby=('M200c_Msun', 
                          10**np.array([11., 11.5, 12., 12.5, 13., 13.5, 14., 15.])),
                          percentiles=np.array([0.02, 0.1, 0.5, 0.9, 0.98]),
                          inclSFgas=True,
                          minrad_use_r200c=0.1):
    '''
    from q - 3D radius histograms weighted by w, extract w-weighted 
    percentiles of the q distribution as a function of radius
    done for each individual galaxy in the sample

    geared towards a particular luminosity-weighted sample
    other samples will need additional paramters to determine whether 
    to select or sum over certain axes.

    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    binby: only used for sample subselection and consistency checks 
           (i.e., values above and below the bin range are excluded,
            histogram axes must match.)
            some data is saved per binby group though, to save some space
    percaxis: axis to get the profile for. 'cumul' get the cumulative 
            weight profiles
    '''
    if samplename is None:
        samplename = defaults['sample']
    fdata = dataname(samplename)
    if histtype == 'cumul':
        _histtype = 'nrprof'
    else:
        _histtype = histtype
    fname = files(samplename, weighttype, histtype=_histtype)
    
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
    
    galdata_all = pd.read_csv(fdata, header=headlen, sep='\t', 
                              index_col='galaxyid')
    galname_all = pd.read_csv(fname, header=0, sep='\t', 
                              index_col='galaxyid')
    #return galdata_all, galname_all # to get input files for checking output
    if idsel is not None:
        if isinstance(idsel, slice):
            galids = np.array(galname_all.index)[idsel]
        else:
            galids = idsel
    else:
        galids = np.array(galname_all.index) # galdata may also include non-selected haloes

    colsel = binby[0]
    galbins = binby[1]
    numgalbins = len(galbins) - 1
    edgedata = [None] * numgalbins
    galids_base = [None] * numgalbins
    galids_bin = [[]] * numgalbins
    ggrpn_base = 'galaxy_{galid}'

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
    outname.append('inclSFgas' if inclSFgas else 'exclSFgas')
    outname.append('indiv-gal-rad3Dprof')
    if minrad_use_r200c is not None:
        outname.append('from-{:.2f}-R200c'.format(minrad_use_r200c))
    #outname.append('testfile') # debug without messing up previous work
    outname = '_'.join(outname)
    outname = '/'.join(pathparts[:-1]) + '/' +  outname + '.' + ext
    print('will save to: {}'.format(outname))
    
    # axis data attributes that are allowed to differ between summed histograms
    neqlist = ['number of particles',\
               'number of particles > max value',\
               'number of particles < min value',\
               'number of particles with finite values']
    print('using galaxy ids {}'.format(galids))

    with h5py.File(outname, 'a') as fo:
        # encodes data stored -> same name is a basic consistency check 
        # for the sample
        igrpn = galname_all.at[galids[0], 'groupname']
        
        for galid in galids:
            selval = galdata_all.at[galid, colsel]
            binind = np.searchsorted(galbins, selval, side='right') - 1
            if binind in [-1, numgalbins]: # halo/stellar mass does not fall into any of the selected ranges
                #print('Skipping galaxy id {}'.format(galid))
                continue
            
            # retrieve data from this histogram for checks
            igrpn_temp = galname_all.at[galid, 'groupname']   
            if igrpn_temp != igrpn:
                raise RuntimeError('histogram names for galaxyid %i: %s, %i: %s did not match'%(galids[0], igrpn, galid, igrpn_temp))
            ifilen_temp = galname_all.at[galid, 'filename']   
            
            #try:
            print('trying galaxy id {}'.format(galid))
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
                    msg = 'Mismatch in histogram axis names for galaxyids %i, %i'
                    msg = msg%(galids_base[binind], galid)
                    raise RuntimeError(msg)
                if not np.all([edgedata_t[ekey][akey] == \
                               edgedata[binind][akey] \
                               for akey in edgedata_t[ekey].keys()] \
                              for ekey in edgekeys_t):
                    msg = 'Mismatch in histogram axis properties for galaxyids %i, %i'
                    msg = msg%(galids_base[binind], galid)
                    raise RuntimeError(msg)
            
            # edges are compatible: shift and combine histograms
            # radial bins: only shift if R200c units needed
            numaxes = len(edges_t)
            sumaxes = list(range(numaxes))
            try:
                rax = edgedata_t['3Dradius']['histogram axis']
            except KeyError:
                raise KeyError('Could not retrieve histogram axis for galaxy %i, file %s'%(galid, ifilen_temp))
            if percaxis != 'cumul':
                try:
                    pax = edgedata_t[percaxis]['histogram axis']
                except KeyError:
                    raise KeyError('Could not retrieve percentile property axis {} for galaxy {}, file {}'.format(percaxis, galid, ifilen_temp))
                sumaxes.remove(pax)
            
            sumaxes.remove(rax)
            axessel = [slice(None, None, None)] * numaxes
            if not inclSFgas:
                sfaxname = 'StarFormationRate'
                try:
                    sfax = edgedata_t[sfaxname]['histogram axis']
                except KeyError:
                    _keys = edgedata_t.keys()
                    sfcandidate = [sfaxname in _key for _key in _keys]
                    if sum(sfcandidate) == 1:
                        _sfki = np.where(sfcandidate)[0][0]
                        _sfaxname = _keys[_sfki]
                        sfax = edgedata_t[_sfaxname]['histogram axis']
                    else:
                        msg = 'Could not retrieve SFR axis like {} from options' +\
                              ' {} for galaxy {}, file {}'
                        msg = msg.format(sfaxname, _keys, galid, ifilen_temp)
                        raise KeyError()
                sfi = np.where(np.isclose(edges_t[sfax]), 0.)[0][0]
                axessel[sfax] = slice(0, sfi, None)
            if len(sumaxes) > 0:
                hist_t = np.sum(hist_t[tuple(axessel)], axis=tuple(sumaxes))
            # axes in summed histogram
            if percaxis == 'cumul':
               _rax = 0
            else:
               _pax, _rax = np.argsort([pax, rax])
            if minrad_use_r200c is not None:
                redges = np.copy(edges_t[rax])
                r200c = galdata_all.at[galid, 'R200c_cMpc']
                r200c *= cosmopars['a'] * c.cm_per_mpc
                redges *= 1. / r200c
                si = np.where(np.isclose(redges, minrad_use_r200c))[0]
                if len(si) != 1:
                    msg = 'did not find edge matching {} R200c for {}'
                    msg = msg.format(minrad_use_r200c, galid)
                    raise RuntimeError(msg)
                si = si[0]
                ssel = [slice(None, None, None)] * len(hist_t.shape)
                ssel_keep = ssel.copy()
                ssel_keep[_rax] = slice(si, None, None)
                ssel_keep = tuple(ssel_keep)
                ssel_sum = ssel.copy()
                ssel_sum[_rax] = slice(None, si, None)
                ssel_sum = tuple(ssel_sum)
                ssel_appax = ssel.copy()
                ssel_appax[_rax] = np.newaxis
                ssel_appax = tuple(ssel_appax)
                _hist = np.sum(hist_t[ssel_sum], axis=_rax)[ssel_appax]
                _hist = np.append(_hist, hist_t[ssel_keep], axis=_rax)
                hist_t = _hist
                edges_t[rax] = np.append(edges_t[rax][0], edges_t[rax][si:])
            if percaxis == 'cumul':
                cumulvals = np.cumsum(hist_t)
            else:
            # shape: percentile, radial bin
                
                percs = percentiles_from_histogram_handlezeros(hist_t, 
                        edges_t[pax], axis=_pax, percentiles=percentiles)
            
            # store the data
            # don't forget the list of galids (galids_bin, and edgedata)
            #print(hists)
            print('Trying to save for galid {}'.format(galid))
            ogrpn = '%s/%s'%(percaxis, samplename)
            if ogrpn in fo:
               ogrp = fo[ogrpn]
            else:
                ogrp = fo.create_group(ogrpn)
            
            ggrpn = ggrpn_base.format(galid=galid)
            ggrp = ogrp.create_group(ggrpn) if ggrpn not in ogrp\
                     else ogrp[ggrpn] 
            if percaxis == 'cumul':
                ggrp.create_dataset('cumulative_weight', data=cumulvals)
                ggrp['cumulative_weight'].attrs.create('inclSFgas', inclSFgas)
            else:
                ggrp.create_dataset('percentiles', data=percs)
                ggrp['percentiles'].attrs.create('axis_perc', 0)
                ggrp['percentiles'].attrs.create('axis_r3D', 1)
                ggrp['percentiles'].attrs.create('inclSFgas', inclSFgas)
            ggrp.create_dataset('edges_r3D', data=edges_t[rax])
            ggrp['edges_r3D'].attrs.create('units', np.string_('cm'))
            ggrp['edges_r3D'].attrs.create('comoving', False)
        
        print('Trying to save overall/bin data')
        ogrpn = '%s/%s'%(percaxis, samplename)
        if ogrpn in fo:
            ogrp = fo[ogrpn]
        else:
            ogrp = fo.create_group(ogrpn)
        hgrp = ogrp.create_group('Header')
        cgrp = hgrp.create_group('cosmopars')
        for key in cosmopars:
            cgrp.attrs.create(key, cosmopars[key])
        hgrp.attrs.create('galaxy_data_file', np.string_(fdata))
        hgrp.attrs.create('galaxy_histogram_file_list', np.string_(fname))
        for bi in range(numgalbins):
            bgrpn = binby[0] + \
                    '_{:.2f}-{:.2f}'.format(binby[1][bi], binby[1][bi + 1])
            bgrp = ogrp.create_group(bgrpn)
            edged = edgedata[bi]
            print(bgrpn)
            print(edged)
            if edged is not None: # in test cases, a bin may be empty
                hgrp = bgrp.create_group('orig_hist_data')
                for key in edged.keys():
                    hgrp.create_group(key)
                    for skey in edged[key].keys():
                        m3.saveattr(hgrp[key], skey, edged[key][skey])
                
            bgrp.create_dataset('galaxyids', data=np.array(galids_bin[bi]))
            bgrp.create_dataset('percentiles', data=percentiles)
            
    print('Saved data to file {}'.format(outname))  
    print('Main hdf5 group: {}/{}'.format(percaxis, samplename))


def combine_indiv_radprof(percaxis=None, samplename=None, idsel=None, 
                          inclSFgas=True,
                          weighttype='Mass', histtype='rprof_rho-T-nion',
                          percentiles_in=np.array([0.02, 0.1, 0.5, 0.9, 0.98]),
                          percentiles_out=[[0.5], [0.5], 
                                           [0.02, 0.1, 0.5, 0.9, 0.98], 
                                           [0.5], [0.5]],
                          cumul_normrad_r200c=1.,
                          minrad_use_r200c=0.1):
    '''
    get percentiles of individual percentile distributions in galaxy 
    property sets. Saved in the same file as the individual profiles
    binning follows the groups set in the previous step

    idsel: project only a subset of galaxies according to the given list
           useful for testing on a few galaxies
           ! do not run in  parallel: different processes will try to write to
           the same list of output files
    percaxis: axis to get the profile for
    cumul_normrad: for cumulative profiles, normalize to the value at this
            radius before getting percentiles. None means no normalization
            beforehand.
    minrad_use_r200c: any radial bins below this edge are collapsed into one 
            bin before stacking
    '''
    # non-cumul: profile tested on Trprof, em-c5r, one halo mass bin,
    # percentile 50 of 90th percentiles 
    # generally looked ok (equal numbers > and < 50th percentile in individual
    # profiles), except in the smallest bin, where the number of values
    # equal to the median is larger then the difference between the number
    # of higher and lower values.
    # cumul: profile tested on nrprof, em-c5r, one halo mass bin, 
    # percentile 10, normalized at R200c

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
    
    galdata_all = pd.read_csv(fdata, header=headlen, sep='\t', 
                              index_col='galaxyid')
    galname_all = pd.read_csv(fname, header=0, sep='\t', 
                              index_col='galaxyid')
    
    if idsel is not None:
        if isinstance(idsel, slice):
            galids = np.array(galname_all.index)[idsel]
        else:
            galids = idsel
    else:
        galids = np.array(galname_all.index) # galdata may also include non-selected haloes

    ggrpn_base = 'galaxy_{galid}'

    # construct name of summed histogram by removing position specs from a specific one
    outname = galname_all.at[galids[0], 'filename']
    pathparts = outname.split('/')
    namepart = pathparts[-1]
    ext = namepart.split('.')[-1]
    namepart = '.'.join(namepart.split('.')[:-1])
    nameparts = (namepart).split('_')
    inname = []
    for part in nameparts:
        if not (part[0] in ['x', 'y', 'z'] and '-pm' in part):
            inname.append(part)
    inname.append('inclSFgas' if inclSFgas else 'exclSFgas')
    inname.append('indiv-gal-rad3Dprof')
    if minrad_use_r200c is not None:
        inname.append('from-{:.2f}-R200c'.format(minrad_use_r200c))
    #inname.append('testfile') # debug without messing up previous work
    inname = '_'.join(inname)
    inname = '/'.join(pathparts[:-1]) + '/' +  inname + '.' + ext
    outname = inname #.replace('indiv', 'comb')
    #print(outname)
    
    ogrpn = '/'.join([percaxis, samplename])
    with h5py.File(outname, 'a') as fo:
        # encodes data stored -> same name is a basic consistency check 
        # for the sample
        
        print('file: ', outname)
        print('main group: ', ogrpn)
        #parts = ogrpn.split('/')
        #_mark = fo 
        #for part in parts:
        #    if part in _mark.keys():
        #        print('going ok up to: ', part)
        #        _mark = _mark[part]
        #    else:
        #        print('at ', _mark)
        #        print('group ',  part, ' not in ', list(_mark.keys()))
        #        break
        mgrp = fo[ogrpn]
        
        allkeys = list(mgrp.keys())
        binkeys = {key if key != 'Header' and 'galaxy' not in key\
                   else None for key in allkeys}
        binkeys -= {None}
        binkeys = list(binkeys)
        
        for bkey in binkeys:
            bgrp = mgrp[bkey]
            if percaxis != 'cumul':
                percentiles_stored = bgrp['percentiles'][:]
            galids_bin = bgrp['galaxyids'][:]
        
            edges_ref = None
            percvals = []
            for galid in galids_bin:
                ggrpn = ggrpn_base.format(galid=galid)
                ggrp = mgrp[ggrpn]
                r200c = galdata_all.at[galid, 'R200c_cMpc']
                r200c *= cosmopars['a'] * c.cm_per_mpc
                _ed = ggrp['edges_r3D'] / r200c
                if edges_ref is None:
                    edges_ref = _ed
                elif not np.allclose(_ed, edges_ref):
                    msg = 'edges within one galaxy bin are mismatched' +\
                          'galxyids {}, {} in {}'
                    msg = msg.format(galid, galids_bin[0], bkey)
                    raise RuntimeError(msg)
                if percaxis == 'cumul':
                    if cumul_normrad_r200c is None:
                        percvals.append(ggrp['cumulative_weight'])
                    else:
                        normp = cumul_normrad_r200c
                        encledge = np.where(np.isclose(_ed, normp))[0]
                        if len(encledge) == 0:
                            msg = 'could not find a value close to {}' + \
                                  ' R200c for galaxy {}'
                            msg = msg.format(normp, galid)
                            raise RuntimeError(msg)
                        elif len(encledge) == 1:
                            ni = encledge[0]
                        else:
                            ni = np.argmin(np.abs(_ed, - normp))
                        ni = ni - 1 # cumulative value is at right bin edge
                        _cumul = ggrp['cumulative_weight'][:]
                        _cumul *= 1. / _cumul[ni]
                        percvals.append(_cumul)

                else:
                    percvals.append(ggrp['percentiles'])
                    

                
            # shape: galaxy, [percentile,] radius
            percvals = np.array(percvals) 
            galcount = percvals.shape[0]

            sgrp = bgrp.create_group('ensemble_percentiles')
            if percaxis == 'cumul':
                dsfmt = 'perc-{pout:.3f}'
                if hasattr(percentiles_out[0], '__len__'):
                    perc_out = set()
                    for _po in percentiles_out:
                        perc_out = perc_out.union(set(list(_po)))
                    perc_out = list(perc_out)
                    perc_out.sort()
                    perc_out = np.array(perc_out)
                else:
                    perc_out = np.array(percentiles_out)
                #print('percvals: ', percvals)
                #print('perc_out: ', perc_out)
                percofcumul = np.quantile(percvals, perc_out, axis=0)
                for poind, pout in enumerate(perc_out):
                    dsname = dsfmt.format(pout=pout)
                    sgrp.create_dataset(dsname, data=percofcumul[poind, :])
                    isnormed = cumul_normrad_r200c is not None
                    sgrp.attrs.create('normalized profiles', isnormed)
                    if isnormed:
                        sgrp.attrs.create('indiv. normalized at [R200c]', 
                                          cumul_normrad_r200c)
                
            else:
                dsfmt = 'perc-{pout:.3f}_of_indiv_perc-{pin:.3f}' 
                nanref = None
                for p_outind, p_in in enumerate(percentiles_in):
                    if not np.any(np.isclose(p_in, percentiles_stored)):
                        msg = 'percentile {} was not stored'.format(p_in)
                        raise RuntimeError(msg)
                    pind_in = np.argmin(np.abs(p_in - percentiles_stored))
                    _pv = percvals[:, pind_in, :]
                    ppoints_out = percentiles_out[p_outind]

                    percofperc = np.nanquantile(_pv, ppoints_out, axis=0)
                    nancount = np.sum(np.isnan(_pv), axis=0)
                
                    if nanref is None:
                        nanref = nancount
                    elif not np.all(nanref == nancount):
                        raise RuntimeError('NaN counts different for different percentiles')
                    for i, pout in enumerate(ppoints_out):
                        _data = percofperc[i, :]
                        dsname = dsfmt.format(pout=pout, pin=p_in)
                        sgrp.create_dataset(dsname, data=_data)
                sgrp.create_dataset('NaN_per_bin', data=nancount)

            sgrp.create_dataset('edges_r3D', data=edges_ref)
            sgrp['edges_r3D'].attrs.create('units', np.string_('R200c'))
            #sgrp['edges_r3D'].attrs.create('comoving', False)
            sgrp.attrs.create('galaxy_count', galcount)

    print('Saved data to file {}'.format(outname))  
    print('Main hdf5 group: '.format(ogrpn))
