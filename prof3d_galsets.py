#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:27:59 2019

@author: wijers
"""

import numpy as np
import h5py
import pandas as pd

import selecthalos as sh
import cosmo_utils as cu
import eagle_constants_and_units as c
import make_maps_v3_master as m3
import make_maps_opts_locs as ol

# directory for metadata files
tdir = '/net/luttero/data2/imgs/CGM/3dprof/'

defaults = {'sample': 'L0100N1504_27_Mh0p5dex_1000'}

samples = {'L0100N1504_27_Mh0p5dex_1000': sh.L0100N1504_27_Mh0p5dex_1000}

weighttypes = {'Mass': {'ptype': 'basic', 'quantity': 'Mass'},\
               'Volume': {'ptype': 'basic', 'quantity': 'propvol'},\
               }
weighttypes.update({ion: {'ptype': 'Nion', 'ion': ion} for ion in\
                    ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17', 'hneutralssh']}) 

def dataname(samplen):
    return tdir + 'halodata_%s.txt'%(samplen)

def files(samplen, weighttype):
    return tdir + 'filenames_%s_%s.txt'%(samplen, weighttype)

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
        
def genhists(samplename=None, rbinu='pkpc', idsel=None, weighttype='Mass', logM200min=11.0):
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
               {'ptype': 'basic', 'quantity': 'Temperature'},\
               {'ptype': 'basic', 'quantity': 'Density'},\
               ]
    if weighttype in ol.elements_ion.keys():
        axesdct =  axesdct + [{'ptype': 'Niondens', 'ion': weighttype}]
    
    with open(files(samplename, weighttype), 'w') as fdoc:
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
            
            axbins =  [rbins] + [0.1] * (len(axesdct) - 1)
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
            m3.makehistograms_perparticle(*args, nameonly=False, **kwargs)
            
            fdoc.write('%i\t%s\t%s\n'%(gid, outname[0], outname[1]))
            