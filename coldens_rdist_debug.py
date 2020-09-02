#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:11:08 2020

@author: Nastasha

Something keeps stopping my radial profiles from stamps going out as far as 
I want. This script is to figure out what. For use on cosma

"""

import os
import numpy as np
import pandas as pd
import h5py

import cosmo_utils as cu
import make_maps_opts_locs as ol
import selecthalos as sh

def get_galdata():
    fin = ol.pdir + 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    with open(fin, 'r') as fi:
        while True:
            line  = fi.readline()
            if line.startswith('halocat:'):
                hcf = line.split('\t')[-1]
                hcf = hcf.strip()
                
                if not os.path.isfile(hcf):
                    hcf = ol.pdir + hcf.split('/')[-1]
                    if not os.path.isfile(hcf):
                        raise RuntimeError('Could not find halo catalogue\n{}'.format(line))
                break
            elif line == '': # end of file
                raise RuntimeError('Could not find the halo catalogue file name')
    galdata = pd.read_csv(fin, header=2, sep='\t', index_col='galaxyid')
    with h5py.File(hcf, 'r') as fi:
        cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
    return galdata, cosmopars
    
def test_r200c(galdata, cosmopars):
    '''
    test whether cosmo_utils R200c_pkpc does what it should
    
    input: output from get_galdata
    '''
    print('Testing cosmo_utils R200c_pkpc')
    gids = np.random.choice(galdata.index, size=20, replace=False)
    
    M200c_Msun = np.array(galdata['M200c_Msun'][gids])
    R200c_pkpc = np.array(galdata['R200c_cMpc'][gids]) * 1e3 * cosmopars['a']
    
    R200c_pkpc_cu = cu.R200c_pkpc(M200c_Msun, cosmopars)
    
    res = np.allclose(R200c_pkpc, R200c_pkpc_cu, rtol=1e-4)
    if res:
        print('passed, using masses {}--{} log10 Msun'.format(\
              np.log10(np.min(M200c_Msun)), np.log10(np.max(M200c_Msun))))
    else:
        print('failed, using masses {}--{} log10 Msun'.format(\
              np.log10(np.min(M200c_Msun)), np.log10(np.max(M200c_Msun))))
        head = 'log10 M200c [Msun] \t R200c [pkpc] \t R200c [pkpc]: cosmo_utils'
        fill = '{M200c} \t {R200c} \t {R200c_cu}'
        print(head)
        for M200c, R200c, R200c_cu in zip(M200c_Msun, R200c_pkpc, R200c_pkpc_cu):
            print(fill.format(R200c=R200c, R200c_cu=R200c_cu, M200c=np.log10(M200c)))
    return res

def test_inputsettings(galdata, cosmopars):
    '''
    test the input max. radii for stamps in runhistograms
    '''
    print('Testing the assignment of max. radii to galaxyids')
    
    ## from runhistograms.py
    rmax_r200c = 4.
    
    # select 1500 halos randomly in  0.5 dex Mstar bins (trying to do everything just gives memory errors)
    print('Getting galaxy ids')
    galids_dct = sh.L0100N1504_27_Mh0p5dex_1000.galids() 
    # there seem to be issues with these bins and they aren't going to be in
    # the plots anyway
    del galids_dct['geq9.0_le9.5']
    del galids_dct['geq9.5_le10.0']
    del galids_dct['geq10.0_le10.5']
    # set minimum distance based on virial radius of halo mass bin;
    # factor 1.1 is a margin
    print('Getting halo radii')
    maxradii_mhbins = {key: 1.1 * cu.R200c_pkpc(10**14.6, cosmopars) if key == 'geq14.0'\
                            else 1.1 * cu.R200c_pkpc(10**(float(key.split('_')[1][2:])), cosmopars)\
                       for key in galids_dct} 
    #print('for debug: galids_dct:\n')
    #print(galids_dct)
    #print('\n')
    print('Matching radii to Mhalo bins...')
    allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
    gkeys = list(galids_dct.keys())
    keymatch = [gkeys[np.where([gid in galids_dct[key] for key in gkeys])[0][0]] for gid in allids]
    mindist_pkpc = rmax_r200c * np.array([maxradii_mhbins[gkey] for gkey in keymatch])
    
    ## testing part
    R200c_pkpc = galdata['R200c_cMpc'][np.array(allids)] * 1e3 * cosmopars['a']
    t1 = np.all(mindist_pkpc >= rmax_r200c * R200c_pkpc)
    if not t1:
        print('Individual galaxies are assigned too small R200c')
    # set by hand for test
    mbindata = {'geq10.0_le10.5': 10.5,\
                'geq10.5_le11.0': 11.0,\
                'geq11.0_le11.5': 11.5,\
                'geq11.5_le12.0': 12.0,\
                'geq12.0_le12.5': 12.5,\
                'geq12.5_le13.0': 13.0,\
                'geq13.0_le13.5': 13.5,\
                'geq13.5_le14.0': 14.0,\
                'geq14.0':        14.6,\
                'geq9.0_le9.5':    9.5,\
                'geq9.5_le10.0':  10.0,\
                }
    minrad_mbins = {key: rmax_r200c * cu.R200c_pkpc(10**mbindata[key], cosmopars)\
                    for key in mbindata}
    ed = np.arange(9., 14.1, 0.5)
    matchbin = {(10**ed[i], 10**ed[i + 1]): 'geq{:.1f}_le{:.1f}'.format(ed[i], ed[i+1])\
                for i in range(len(ed) - 1)}
    matchbin.update({(10**ed[-1], np.inf): 'geq{:.1f}'.format(ed[-1])})

    M200c_Msun = galdata['M200c_Msun'][np.array(allids)]    
    minR200c_bins = np.ones(len(allids)) * np.NaN
    for key in matchbin:
        sel = M200c_Msun >= key[0]
        sel &= M200c_Msun < key[1]
        minrad = minrad_mbins[matchbin[key]]
        minR200c_bins[sel] = minrad
    if np.any(np.isnan(minR200c_bins)):
        raise RuntimeError('Some galaxies were not assigned minimum test radii')
    t2 = np.all(mindist_pkpc >= minR200c_bins)
    if not t2:
        print('Galaxies are assigned R200c too small for their M200c bins')
    res = t1 & t2
    if res:
        print('test passed')
    else:
        print('test failed')
    return res
    
    
def main():
    galdata, cosmopars = get_galdata()
    
    res_cu = test_r200c(galdata, cosmopars)
    print('\n\n')
    res_rh = test_inputsettings(galdata, cosmopars)
    print('\n\n')
    
    if np.all([res_cu, res_rh]):
        print('All tests passed')
    else:
        if not res_cu:
            print('cosmo_utils.R200c_pkpc failed')
        if not res_rh:
            print('runhistograms stamp radii setting failed')
            
if __name__ == '__main__':
    main()
    
    
    


