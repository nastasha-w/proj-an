#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:18:01 2021

@author: Nastasha

pipeline for generating radial profiles from generic column density maps,
saving only the final profiles if desired
 
"""

import h5py
import numpy as np
import sys
import os

import make_maps_v3_master as m3
import coldens_rdist as crd

import pipeline_cddfs as pcd





pcd.get_names_and_pars(mapslices, *args, **kwargs)

# stamps contain Header info from the maps in their Header group
# radial profile files do not -> copy over in the pipeline

def create_stampfiles(mapslices, catname, *args, nameonly=False,
                      stampkwlist, **kwargs):
    
    
    stampkw_defaults = {'velspace': False, 'offset_los': 0.,
                        'mindist_pkpc': None, 'numsl': 1,
                        'selection': None, 'rmax_r200c': None}
    check = np.all(['selection' in kws and 'rmax_r200c' in kws \
                    for kws in stampkwlist])
    if not check:
        raise ValueError('keywords "selection" and "max_r200c" ' +\
                         'must be given in stampkwlist')
    
    nameslist, argslist, kwargslist = get_names_and_pars(mapslices,\
                                                         *args, **kwargs)
        
    already_exists = []
    _nameslist = []    
    for name, _args, _kwargs in zip(nameslist, argslist, kwargslist):
        name = name[0]
        _nameslist.append(name)
        if os.path.isfile(name):
            already_exists.append(True)
        else:
            already_exists.append(False)
            print('Creating file: {}'.format(name))
            print('-'*80)
            m3.make_map(*_args, nameonly=False, **kwargs)
            print('-'*80)
            print('\n'*3)
    
    if len(_nameslist) == 1:
        szcens = None
        filebase = _nameslist[0]
    else:
        _base = _nameslist[0]
        keyword = '{}cen'.format(kwargslist[0]['axis'])
        pathparts = _base.split('/')
        nameparts = pathparts[-1].split('_')
        index = np.where([keyword in part for part in nameparts])[0][0]
        # stamp finder still uses the old string fill format
        filebase = '_'.join(nameparts[:index] +\
                            [keyword + '%s'] +\
                            nameparts[index + 1:])
        filebase = '/'.join(pathparts[:-1] + [filebase])
        
        szcens = []
        for name in _nameslist:
            pathparts = name.split('/')
            nameparts = pathparts[-1].split('_')
            index = np.where([keyword in part for part in nameparts])[0][0]
            part = nameparts[index]
            fill = part[len(keyword):]            
            szcens.append(fill)
    
    outname = ol.pdir + 'stamps/' + 'stamps_' + filebase%('-all')
    if nameonly:
        return outname
    
    L_x = None
    npix_x = None
    rmin_r200c = np.Nan 
    maxnum = np.inf
    kw_ign = {'npix_y': None, 'logquantity': True, 'npix_y': None,
              }
    axis = kwargslist[0]['axis']
    
    
    
    
    if isinstance(stampkwlist, dict):
        stampkwlist = [stampkwlist]
    for skwargs in stampkwlist:
        _kw = kw_ign.copy()
        _kw.update(stampkw_defaults)
        _kw.update(skwargs)
        
        # not actually a keyword argument, but easier to wrap this way
        selection = _kw['selection']
        del _kw['selection']
        
        # with stamps=True, from hdf5 files:
        # following arguments are ignored
                     
        base, szcens, L_x, npix_x,\
                     rmin_r200c, ,\
                     catname,\
                     selection, maxnum, outname=None,\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=None,\
                     axis='z', velspace=False, offset_los=0., stamps=False,\
                     trackprogress=False
                     
        crd.rdists_sl_from_selection(filebase, szcens, L_x, npix_x,
                                     rmin_r200c, rmax_r200c,
                                     catname,
                                     selection, maxnum, outname=outname,
                                     axis=axis, 
                                     stamps=True,
                                     trackprogress=True, 
                                     **_kws)
    
    if deletemaps:
        for preexisting, name in zip(already_exists, _nameslist):
            if not preexisting:
                os.remove(name)
                
def makeprofiles():

    
    halocat = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'   
    galids_dct = sh.L0100N1504_27_Mh0p5dex_1000.galids()
    del galids_dct['geq9.0_le9.5']
    del galids_dct['geq9.5_le10.0']
    del galids_dct['geq10.0_le10.5']

    with h5py.File(catname, 'r') as cat:
        cosmopars = {key: item for key, item in cat['Header/cosmopars'].attrs.items()}
  
    lines = ['c5r', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r', 'fe18',\
            'fe17-other1', 'fe19', 'o7r', 'o7ix', 'o7iy', 'o7f', 'o8', 'fe17',\
            'c6', 'n7', 'n6-actualr']
    lineind = jobind - 30234
    line = lines[lineind]
    numsl = 1
  
    mapbase = 'emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_noEOS.hdf5'
    if line in ['ne10', 'n6-actualr']:
        mapbase = mapbase.replace('test3.5', 'test3.6')
    mapname = ol.ndir + mapbase.format(line=line)
             
    stampname = ol.pdir + 'stamps/' + 'stamps_%s_%islice_to-min3p5R200c_L0100N1504_27_Mh0p5dex_1000_centrals_M-ge-10p5.hdf5'%((mapname.split('/')[-1][:-5])%('-all'), numsl)
    proffile = stampname.split('/')[-1]
    proffile = ol.pdir + 'radprof/radprof_' + proffile
    
    rbins_r200c = np.arange(0., 2.51, 0.05)
    yvals_perc = [1., 5., 10., 50., 90., 95., 99.]
    #kwarg_opts = [
    #              {'runit': 'pkpc', 'ytype': 'mean', 'yvals': None,\
    #               'separateprofiles': True, 'uselogvals': False},\
    #              ]
    
    for hmkey in galids_dct:
        print('Trying halo set %s'%(hmkey))
        # get max. distance from halo mass range:
        
        # hmkeys format: 'geq10.0_le10.5' or 'geq14.0'
        # extracted out to max(R200c in bin)
        minmass_Msun = 10**(float(hmkey.split('_')[0][3:]))
        maxdist_pkpc = 3.5 * cu.R200c_pkpc(minmass_Msun * 10**0.5, cosmopars)
        # lin-log bins: lin 10 pkpc up to 100 kpc, then 0.1 dex
        rbins_log_large_pkpc = 10.**(np.arange(1., np.log10(maxdist_pkpc), 0.25))
        rbins_pkpc_large = np.append([0.], rbins_log_large_pkpc)
        
        rbins_log_small_pkpc = 10.**(np.arange(1., np.log10(maxdist_pkpc), 0.1))
        rbins_pkpc_small = np.append([0.], rbins_log_small_pkpc)
        
        #if kwargs['runit'] == 'pkpc':
        #    rbins = rbins_pkpc
        #else:
        for rbins in [rbins_pkpc_small, rbins_pkpc_large]:
            galids = np.copy(galids_dct[hmkey])
            #if kwargs['separateprofiles']:
            #    galids = galids[:10] # just a few examples, don't need the whole set
            #print('Calling getprofiles_fromstamps with:')
            #print(stampname)
            #print('rbins: {}'.format(rbins))
            #if len(galids) > 15:
            #    print('galaxyids: {} ... {}'.format(galids[:8],  galids[-7:]))
            #else:
            #    print('galaxyids: {}'.format(galids))
            #print(halocat)
            #print('out: {}'.format(outfile))
            #print('grouptag: {}'.format(hmkey))
            #print('\t '.join(['{key}: {val}'.format(key=key, val=kwargs[key])\
            #                  for key in kwargs]))
            #print('\n\n')
            crd.combineprofiles(proffile, rbins, galids,
                                runit='pkpc', ytype_in='mean', yvals_in=None,
                                ytype_out='perc',\
                                yvals_out=[1., 5., 10., 50., 90., 95., 99.],
                                uselogvals=True,
                                outfile=proffile, grptag=hmkey)