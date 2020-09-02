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


def main():
    galdata, cosmopars = get_galdata()
    
    res_cu = test_r200c(galdata, cosmopars)
    print('\n\n')
    
    if np.all([res_cu]):
        print('All tests passed')

if __name__ == '__main__':
    main()
    
    
    


