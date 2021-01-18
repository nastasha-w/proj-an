#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 16:18:01 2021

@author: Nastasha

pipeline for generating radial profiles from generic column density maps,
saving only the final profiles if desired

halo catalogue generation is not included, and names of the stamp and radial 
profile files must be given by hand. (Generic naming for selections can get 
hairy pretty fast.)

Note that the pipeline does assume that if a stamp file with the same name as
provided already exists, it contains the same data as the file you want to
give that name.
 
"""

import h5py
import numpy as np
import sys
import os

import make_maps_v3_master as m3
import make_maps_opt_locs as ol
import coldens_rdist as crd

import pipeline_cddfs as pcd


# stamps contain Header info from the maps in their Header group
# radial profile files do not -> copy over in the pipeline

def create_stampfiles(mapslices, catname, args, stampkwlist, 
                      deletemaps=False, **kwargs):
    '''
    create files containing stamps extracted from slice maps from simulations,
    creating the simulation slice maps if needed.

    Parameters
    ----------
    mapslices : int
        Number of slices to divide the simulation volume into.
    catname : string
        name of the hdf5 file containing the halo and galaxy data.
    args : tuple
        (non-keyword) arguments for make_maps_v3_master.make_maps. Center and 
        slice thickness are modified by dividing the given volume into 
        mapslices slices along the projection axis.
    stampkwlist : list-like of dictionaries
        each entry is a set of arguments to pass to 
        coldens_rdists.rdists_sl_from_selection.
        values are:
        -----------
        'galaxyid': array-like of ints
            ids of the central galaxies of the haloes to extract stamps for.
            passed to selecthalos as [('galaxyid', galaxyids)] 
        'rmax_r200c': float or array of floats
            radius of the stamps to extract in units of R200c of the parent 
            halo.
        'outname': string
            name for the stamp file. 
        'numsl': int, optional
            number of slices to combine for one stamp. The default is 1.
        'velspace': bool, optional 
            cross-match galaxies to slices based on galaxy positions in 
            velocity space. The default is False. 
        'offset_los': float, optional  
            offset galaxy positions along the line of sight by this amount 
            before cross-matching to slices (cMpc). The default is 0.
        'mindist_pkpc': float or array of floats, optional
            minimum radius of the stamps to extract (pkpc). If an array, the
            length and order should match the selection, given via galaxyids.
            The maximum of this value and rmax_r200c is used. 
            The default is 0.    
    deletemaps : bool, optional
        Delete the simulations slice maps that were not already present after 
        creating the stamp files. Note that weighted maps are ignored here. 
        The default is False.
    **kwargs : keywords
        Keyword arguments for make_maps_v3_master.make_maps. Arguments 'hdf5',
        'save', and 'nameonly' are ignored and set to the values needed for 
        the pipeline.

    Raises
    ------
    ValueError
        invalid or missing arguments in stampkwlist.

    Returns
    -------
    None.

    '''
    
    stampkw_defaults = {'velspace': False, 'offset_los': 0.,
                        'mindist_pkpc': None, 'numsl': 1,
                        'galaxyid': None, 'rmax_r200c': None,
                        'outname': None}
    
    check = np.all(['galaxyid' in kws and \
                    'rmax_r200c' in kws and \
                    'outname' in kws \
                    for kws in stampkwlist])
    if not check:
        raise ValueError('keywords "selection" and "max_r200c" ' +\
                         'must be given in stampkwlist')
    check = np.all([kws['outname'] is not None for kws in stampkwlist]) 
    if not check:
        msg = 'outname may not be None for any entry in stampkwlist'
        raise ValueError(msg)
    
    nameslist, argslist, kwargslist = pcd.get_names_and_pars(mapslices,\
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
    
    axis = kwargslist[0]['axis']
    
    # with stamps=True, from hdf5 files:
    # following arguments are ignored
    L_x = None
    npix_x = None
    rmin_r200c = np.NaN
    maxnum = np.inf
    kw_ign = {'npix_y': None, 'logquantity': True, 'npix_y': None,
              }
     
    if isinstance(stampkwlist, dict):
        stampkwlist = [stampkwlist]
    for skwargs in stampkwlist:
        _kw = kw_ign.copy()
        _kw.update(stampkw_defaults)
        _kw.update(skwargs)
        
        # not actually keyword arguments, but easier to wrap this way
        galaxyid = _kw['galaxyid']
        del _kw['galaxyid']
        rmax_r200c = _kw['rmax_r200c']
        del _kw['rmax_r200c']
        outname = _kw['outname']
        if '/' not in outname:
            outname = ol.pdir + 'stamps/' + outname
        
        crd.rdists_sl_from_selection(filebase, szcens, L_x, npix_x,
                                     rmin_r200c, rmax_r200c,
                                     catname,
                                     [('galaxyid', galaxyid)], maxnum,
                                     axis=axis, outname=outname,
                                     stamps=True,
                                     trackprogress=True, 
                                     **_kw)
    
    if deletemaps:
        for preexisting, name in zip(already_exists, _nameslist):
            if not preexisting:
                os.remove(name)

                
def create_rprofiles(mapslices, catname, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=(),
                     deletemaps=False, deletestamps=False, **kwargs):
    '''
    create radial profiles. Make slice maps and stamp files as intermediate
    products if they do not already exist

    Parameters
    ----------
    mapslices : int
        Number of slices to divide the simulation volume into.
    catname : string
        name of the hdf5 file containing the halo and galaxy data.
    args : tuple
        (non-keyword) arguments for make_maps_v3_master.make_maps. Center and 
        slice thickness are modified by dividing the given volume into 
        mapslices slices along the projection axis.
    stampkwlist : list-like of dictionaries
        each entry is a set of arguments to pass to 
        coldens_rdists.rdists_sl_from_selection.
        values are:
        -----------
        'galaxyid': array-like of ints
            ids of the central galaxies of the haloes to extract stamps for.
            passed to selecthalos as [('galaxyid', galaxyids)] 
        'rmax_r200c': float or array of floats
            radius of the stamps to extract in units of R200c of the parent 
            halo.
        'outname': string
            name for the stamp file. 
        'numsl': int, optional
            number of slices to combine for one stamp. The default is 1.
        'velspace': bool, optional 
            cross-match galaxies to slices based on galaxy positions in 
            velocity space. The default is False. 
        'offset_los': float, optional  
            offset galaxy positions along the line of sight by this amount 
            before cross-matching to slices (cMpc). The default is 0.
        'mindist_pkpc': float or array of floats, optional
            minimum radius of the stamps to extract (pkpc). If an array, the
            length and order should match the selection, given via galaxyids.
            The maximum of this value and rmax_r200c is used. 
            The default is 0.  
    rprofkwlist : list of list of dicts
            set of keywords to use for radial profile creation. The outer 
            layer of lists matches the stamp keywords to use, the inner layer
            is the set of radial profile keyowrds to use for each stamp set.
            Entries are:
            'rbins': array-like of floats       
                bin edges for statistics extraction
            'runit': string       
                unit the rbins are given in: 'pkpc' or 'R200c'. The default is 
                'pkpc'. 
            'galaxyid': array-like of ints, optional     
                galaxyids to get the profiles from. The default is all the
                galaxyids used for the stamps.
            'ytype': string
                what kind of statistic to extract; 'perc' for percentiles, 
                'fcov' for covering fractions, or 'mean' for the average. The 
                default is 'perc'.
            'yvals': float or list of floats None       
                 if ytype is 'perc' or 'fcov': the values to get the covering
                 fractions (same units as the map) or percentiles (0-100 range)
                 for (float). ignored for ytype 'mean'. The default is 50.
            'separateprofiles': bool 
                 get a profile for each galaxyid (True) or for all of them 
                 together (False). Combined profiles are only possible if this 
                 is True. The default is False.
            'grptag': string     
                 tag for the autonamed output group for the profile. Useful to
                 indicate how the galaxy ids were selected
            'uselogvals':  bool 
                 if True use log y values in the output (and interpret 
                 ytype='fcov' yvals as log). Otherwise, non-log y values are 
                 used in every case. (Just following what's in the input files 
                 is not an option.)
    combrprofkwlist : list of lists of lists of dicts, optional
        the arguments for combined statistsics for individual radial profiles.
        The outermost list layer corresponds to stamp sets, the layer inside 
        that corresponds to the initial radial profile types, and the 
        innermost layer is for different statistics for each profile set.
        Entries are:
           'ytype_out': string
                same options as 'ytype' above. The default is 'perc'.
           'yvals_out': float or list of floats or None
                same options as 'yvals' above. The default is 50.
           'galaxyids': array of floats, optional. 
                The galaxyids to combine the profiles for. The default is all
                the galaxyids used for the radial profiles.
    deletemaps : bool, optional
        Delete the simulations slice maps that were not already present after 
        creating the stamp files. Note that weighted maps are ignored here. 
        The default is False.
    deletestamps :  bool, optional
        Delete the stamp files that were not already present after 
        creating the radial profile files. The default is False.
    **kwargs : keywords
        Keyword arguments for make_maps_v3_master.make_maps. Arguments 'hdf5',
        'save', and 'nameonly' are ignored and set to the values needed for 
        the pipeline.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    defaults_rprof = {'runit': 'pkpc', 'ytype': 'perc', 'yvals': 50.,
                      'separateprofiles': False, 'uselogvals': True,
                      'outfile': None, 'grptag': None}
    
    check = np.all(['bins' in kws for _l in rprofkwlist for kws in _l])
    if not check:
        raise ValueError('"bins" must be specified for radial profiles')
    
    newstamps = []
    stampkwlist_run = []
    
    for stampkw, _rprofkwlist in zip(stampkwlist, rprofkwlist):
        stampfilen =  stampkw['outname']
        if '/' not in stampfilen:
            stampfilen = ol.pdir + 'stamps/' + stampfilen
        if not os.path.isfile(stampfilen):
            newstamps.append(stampfilen)
            stampkwlist_run.append(stampkw)
    
    if len(stampkwlist_run) > 0:
        create_stampfiles(mapslices, catname, args, stampkwlist_run, 
                          deletemaps=deletemaps, **kwargs)
    
    for stampkw, _rprofkwlist, _combkwlist in \
        zip(stampkwlist, rprofkwlist, combrprofkwlist):
        
        stampfilen =  stampkw['outname']
        if '/' not in stampfilen:
            stampfilen = ol.pdir + 'stamps/' + stampfilen
        
        for _rprofkw, __combkwlist in zip(_rprofkwlist, _combkwlist):
            rkw = defaults_rprof.copy()
            rkw.update(_rprofkw)
            if 'galaxyid' in _rprofkwlist:
                galids = _rprofkwlist['galaxyid']
                del rkw['galaxyid']
            else:
                galids = stampkw['galaxyid']
            
            if 'nameonly' in rkw:
                del rkw['nameonly']
            rbins = rkw['bins']
            
            outfilen = crd.getprofiles_fromstamps(stampfilen, rbins, galids,
                                                  nameonly=True, **rkw)
            crd.getprofiles_fromstamps(stampfilen, rbins, galids,
                                       nameonly=True, **rkw)
            # add header info to the profile file
            with h5py.File(outfilen, 'a') as _f:
                if not 'Header' in _f:
                    with h5py.File(stampfilen) as _s:
                        _s.copy('Header', _f, name='Header')
             
            if len(__combkwlist) > 0:
                if not rkw['separateprofiles']:
                    msg = 'Cannot create combined profiles without separate profiles'
                    raise ValueError(msg)
                    
                for combkw in __combkwlist:
                    _combkw = combkw.copy()
                    if 'galaxyid' in _combkw:
                        _galids = _combkw['galaxyid']
                        del _combkw['galaxyid']
                    else:
                        _galids = galids
                    
                    crd.combineprofiles(outfilen, rbins, _galids,
                            runit=rkw['runit'], ytype_in=rkw['ytype'], 
                            yvals_in=rkw['type'],
                            uselogvals=rkw['uselogvals'], outfile=outfilen, 
                            grptag=rkw['grouptag'],
                            **combkw)
    
    if deletestamps:
        for filen in newstamps:
            os.remove(filen)
        
        
    
    
    
    
    
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