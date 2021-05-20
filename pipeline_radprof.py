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

Running the pipeline for the same set of simulation map slices in different,
parallel processes will most likely cause errors, since this pipeline assumes 
each instance of itself is the only one creating this set of files, or 
stamp files from those. 
 
"""

import h5py
import numpy as np
import sys
import os

import make_maps_v3_master as m3
import make_maps_opts_locs as ol
import coldens_rdist as crd
import pipeline_cddfs as pcd
import cosmo_utils as cu
import selecthalos as sh

# on cosma; file name limits?
shortdir = '/cosma5/data/dp004/dc-wije1/t/'

all_lines_SB = ['c5r', 'n6r', 'n6-actualr', 'ne9r', 'ne10', 'mg11r', 'mg12',
                'si13r', 'fe18', 'fe17-other1', 'fe19', 'o7r', 'o7ix', 'o7iy',
                'o7f', 'o8', 'fe17', 'c6', 'n7']
all_lines_PS20 = ['C  5      40.2678A', 'C  6      33.7372A',
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

# stamps contain Header info from the maps in their Header group
# radial profile files do not -> copy over in the pipeline

def shorten_filename(outname):
    _outname = outname
    _outname = _outname.split('/')[-1]
    _outname = _outname.replace(
        'iontab-PS20-UVB-dust1-CR1-G1-shield1', 'PS20tab-def')
    _outname = _outname.replace('TSN-7.499_TAGN-8.499',
                                'Tmindef')
    _outname = _outname.replace('inclSN-nH-lt--2.0', 'lonH-in')
    _outname = shortdir + _outname
    return _outname

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
            name for the stamp file. Automatically named if not given, but 
            these names do not contain any info on the stampkw settings.
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
    stampfiles: list of strings
        Names of the stamp files in the order of stampkwlist.

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
    #check = np.all([kws['outname'] is not None for kws in stampkwlist]) 
    #if not check:
    #    msg = 'outname may not be None for any entry in stampkwlist'
    #    raise ValueError(msg)
    
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
    
    stampfiles = []
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
        if outname is None: # autoname
            outname_base = filebase.split('/')[-1]
            outname_base = ol.pdir + 'stamps/stamps_' + outname_base
            outname_base = outname.base%('-all')
            if outname_base[-5:] == '.hdf5':
                outname_base = outname_base[:-5]
            outname_base = outname_base + '_set{}.hdf5'
            i = 0
            while os.path.isfile(outname_base.format(i)):
                i += 1
            outname = outname_base.format(i)
            
        if '/' not in outname:
            outname = ol.pdir + 'stamps/' + outname
        del _kw['outname']
        
        print('Saving stamps in {}'.format(outname))
        try:
            crd.rdists_sl_from_selection(filebase, szcens, L_x, npix_x,
                                         rmin_r200c, rmax_r200c,
                                         catname,
                                         [('galaxyid', galaxyid)], maxnum,
                                         axis=axis, outname=outname,
                                         stamps=True,
                                         trackprogress=True, 
                                         **_kw)
        except IOError:# file name too long?
            _outname = shorten_filename(outname)
            crd.rdists_sl_from_selection(filebase, szcens, L_x, npix_x,
                                         rmin_r200c, rmax_r200c,
                                         catname,
                                         [('galaxyid', galaxyid)], maxnum,
                                         axis=axis, outname=_outname,
                                         stamps=True,
                                         trackprogress=True, 
                                         **_kw)
            outname = _outname
        stampfiles.append(outname)
        
    if deletemaps:
        for preexisting, name in zip(already_exists, _nameslist):
            if not preexisting:
                print('Deleting map {}'.format(name))
                os.remove(name)
    return stampfiles 
                
def create_rprofiles(mapslices, catname, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=None,
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
            is the set of radial profile keywords to use for each stamp set.
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
    combrprofkwlist : list of lists of lists of dicts or None, optional
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
    
    check = np.all(['rbins' in kws for _l in rprofkwlist for kws in _l])
    if not check:
        raise ValueError('"rbins" must be specified for radial profiles')
    
    newstamps = []
    stampkwlist_run = []
    inds_createnew = []
    for ni, stampkw in enumerate(stampkwlist):
        stampfilen =  stampkw['outname']
        if '/' not in stampfilen:
            stampfilen = ol.pdir + 'stamps/' + stampfilen
        if stampfilen is None:
            newstamps.append(stampfilen)
            stampkwlist_run.append(stampkw)
            inds_createnew.append(ni)
        elif not os.path.isfile(stampfilen):
            if os.path.isfile(shorten_filename(stampfilen)):
                continue
            newstamps.append(stampfilen)
            stampkwlist_run.append(stampkw)
            inds_createnew.append(ni)
            
    if len(stampkwlist_run) > 0:
        stampfiles_new = create_stampfiles(mapslices, catname, args,
                                           stampkwlist_run, 
                                           deletemaps=deletemaps, **kwargs)
    # store the actual stamp names with the keywords used to create them
    for newi, fulli in enumerate(inds_createnew):
        stampkwlist[fulli]['outname'] = stampfiles_new[newi]
    
    #print(stampkwlist)
    #print(rprofkwlist)
    #print(combrprofkwlist)
    
    # match rprofkwlist shape to avoid errors from zipping lists
    if combrprofkwlist is None:
        combrprofkwlist = [ [[]] * len(rprofkwlist[i]) \
                           for i in range(len(rprofkwlist))] 
        combrprofkwlist = [combrprofkwlist] * len(rprofkwlist)
    
    for stampkw, _rprofkwlist, _combkwlist in \
        zip(stampkwlist, rprofkwlist, combrprofkwlist):
        #print('making radial profiles for stamps: {}'.format(stampkw))
            
        stampfilen = stampkw['outname']
        if not os.path.isfile(stampfilen):
            stampfilen = shorten_filename(stampfilen)
        if '/' not in stampfilen:
            stampfilen = ol.pdir + 'stamps/' + stampfilen
        
        for _rprofkw, __combkwlist in zip(_rprofkwlist, _combkwlist):
            rkw = defaults_rprof.copy()
            rkw.update(_rprofkw)
            if 'galaxyid' in _rprofkw:
                galids = _rprofkw['galaxyid']
            else:
                galids = stampkw['galaxyid']
            if galids is None:
                galids = stampkw['galaxyid']
            del rkw['galaxyid']
            
            if 'nameonly' in rkw:
                del rkw['nameonly']
            rbins = rkw['rbins']
            del rkw['rbins']
            #print('making radial profiles: {}'.format(rkw))
            print('Running {tag} with outer bin {}'.format(rbins[-1],
                                                           tag=rkw['grptag']))
            print('for {} galaxies'.format(len(galids)))
            
            if not os.path.isfile(stampfilen):
                _stampfilen = shorten_filename(stampfilen)
            else:
                _stampfilen = stampfilen
            outfilen = crd.getprofiles_fromstamps(_stampfilen, rbins, galids,
                                                  nameonly=True, 
                                                  halocat=catname,
                                                  **rkw)
            #print('Saving radial profiles in {}'.format(outfilen))
            try:
                crd.getprofiles_fromstamps(_stampfilen, rbins, galids,
                                           nameonly=False, halocat=catname, 
                                           **rkw)
            except IOError:
                msg = 'Saving file as\n{new}\ninstead of\n{old}'
                print(msg.format(new=shorten_filename(outfilen), 
                                 old=outfilen))
                outfilen = shorten_filename(outfilen)
                rkw['outfile'] = outfilen
                crd.getprofiles_fromstamps(_stampfilen, rbins, galids,
                                           nameonly=False, halocat=catname, 
                                           **rkw)
                
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
                    #print('combining radial profiles: {}'.format(_combkw))
                    if 'galaxyid' in _combkw:
                        _galids = _combkw['galaxyid']
                        del _combkw['galaxyid']
                    else:
                        _galids = galids
                    if _galids is None:
                        _galids = galids
                        
                    crd.combineprofiles(outfilen, rbins, _galids,
                            runit=rkw['runit'], ytype_in=rkw['ytype'], 
                            yvals_in=rkw['yvals'],
                            uselogvals=rkw['uselogvals'], outfile=outfilen, 
                            grptag=rkw['grptag'],
                            **_combkw)
    
    if deletestamps:
        for filen in newstamps:
            print('Deleting stamps {}'.format(filen))
            os.remove(filen)
        
            
        
def getprofiles_convtest_paper3(index):
    '''
    generate the radial profiles for the paper 3 convergence tests: statistics
    of the mean radial profiles in 10 kpc -> 0.1, 0.25 dex bins

    Parameters
    ----------
    index : int
        index for the simulation box(es) (fast, 2) and line (slow) to run.

    Returns
    -------
    None.

    '''
    
    lines = ['c5r', 'n6-actualr', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12',
             'si13r', 'fe18', 'fe17-other1', 'fe19', 'o7r', 'o7iy', 'o7f',
             'o8', 'fe17', 'c6', 'n7']
    
    lineind = index // 2
    line = lines[lineind]
         
    simset = index % 2
    
    # # test version
    #if simset == 0:
    #    simnums = ['L0025N0376']
    #    varlist = ['REFERENCE']
    #    npix = [8000]
    #    centres = [[12.5] * 3]
    #    mapslices = [4]
    #    halocats = ['catalogue_RefL0025N0376_snap27_aperture30.hdf5']   
    #    stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
    #                   'outname': None, 'rmax_r200c': 2.5,
    #                   'mindist_pkpc': None, 'galaxyid': None},
    #                   ] 
    #    galids_dcts = [sh.L0025N0376_27_Mh0p5dex_1000.galids()]
        
    if simset == 0:
        simnums = ['L0100N1504']
        varlist = ['REFERENCE']
        npix = [32000]
        centres = [[50.] * 3]
        mapslices = [16]

        halocats = ['catalogue_RefL0100N1504_snap27_aperture30.hdf5'] 
        
        stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
                        'outname': None, 'rmax_r200c': 3.5,
                        'mindist_pkpc': None, 'galaxyid': None},
                       {'numsl': 2, 'offset_los': 0., 'velspace': False,
                        'outname': None, 'rmax_r200c': 3.5,
                        'mindist_pkpc': None, 'galaxyid': None},
                       ] 
        galids_dcts = [sh.L0100N1504_27_Mh0p5dex_1000.galids()]
         
    elif simset == 1:
        simnums = ['L0050N0752', 'L0025N0376',
                   'L0025N0752', 'L0025N0752']
        varlist = ['REFERENCE', 'REFERENCE',\
                   'REFERENCE', 'RECALIBRATED']
        npix = [16000, 8000, 8000, 8000]
        centres = [[25.] * 3] + [[12.5] * 3] * 3
        mapslices = [8, 4, 4, 4]
        
        halocats = ['catalogue_RefL0050N0752_snap27_aperture30.hdf5',
                    'catalogue_RefL0025N0376_snap27_aperture30.hdf5',  
                    'catalogue_RefL0025N0752_snap27_aperture30.hdf5',
                    'catalogue_RecalL0025N0752_snap27_aperture30.hdf5',
                    ]
        
        stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
                       'outname': None, 'rmax_r200c': 3.5,
                       'mindist_pkpc': None, 'galaxyid': None},
                       ]
        galids_dcts = [sh.L0050N0752_27_Mh0p5dex_1000.galids(),
                       sh.L0025N0376_27_Mh0p5dex_1000.galids(),
                       sh.L0025N0752_27_Mh0p5dex_1000.galids(),
                       sh.RecalL0025N0752_27_Mh0p5dex_1000.galids()]
        
    ptypeW = 'emission'
    snapnum = 27
    
    kwargs = {'abundsW': 'Sm', 'excludeSFRW': True, 'ptypeQ': None,
              'axis': 'z', 'periodic': True, 'kernel': 'C2',
              'log': True, 'saveres': True, 'hdf5': True,
              'simulation': 'eagle', 'ompproj': True,
              }
    
    halocats = [halocat if '/' in halocat else ol.pdir + halocat \
                for halocat in halocats]
    
    rprofkw_base = [{'rbins': None, 'runit': 'pkpc', 'galaxyid': None,
                     'ytype': 'mean', 'yvals': None, 
                     'separateprofiles': True, 'grptag': None,
                     'uselogvals': False}]
    combrprofkw_base = [{'ytype_out': 'mean', 'yvals_out': None, 
                         'galaxyid': None},
                        {'ytype_out': 'perc', 
                         'yvals_out': [1., 2., 5., 10., 25., 50., 75., 90.,
                                       95., 98., 99.], 
                         'galaxyid': None},
                        ]
    
    for simnum, var, _npix, centre, _mapslices, halocat, galids_dct in\
        zip(simnums, varlist, npix, centres, mapslices, halocats, galids_dcts):
        
        # snapshot data 
        L_x, L_y, L_z = (centre[0] * 2.,) * 3
        npix_x, npix_y = (_npix,) * 2
        args = (simnum, snapnum, centre, L_x, L_y, L_z,
                npix_x, npix_y, ptypeW)
        kwargs['var'] = var
        kwargs['ionW'] = line
        
        print('Calling create_rprofiles with')
        print(args)
        print(kwargs)
        #print(bins)
        print('mapslices: {}'.format(_mapslices))
        print(stampkwlist)
        print('\n')
        
        del galids_dct['geq9.0_le9.5']
        del galids_dct['geq9.5_le10.0']
        del galids_dct['geq10.0_le10.5']
        stampname = ol.pdir + 'stamps/' + \
                   'stamps_{mapname}_{numsl}slice_to-min{dist}R200c' +\
                   '_Mh0p5dex_1000_centrals_M-ge-10p5.hdf5'
        
        with h5py.File(halocat, 'r') as cat:
            cosmopars = {key: item for key, item \
                         in cat['Header/cosmopars'].attrs.items()}
            #r200cvals = np.array(cat['R200c_pkpc'])
            #galids = np.array(cat['galaxyid'])
        
        # assuming z-axis projection 
        _centre = np.copy(centre)
        _centre[2] = 3.125
        _args = (simnum, snapnum, _centre, L_x, L_y, L_z / float(_mapslices),
                 npix_x, npix_y, ptypeW)
        mapfilen = m3.make_map(*_args, nameonly=True, **kwargs)
        mapfilen = mapfilen[0]
        mapfilen = mapfilen.split('/')[-1]
        mapfilen.replace('zcen3.125', 'zcen-all')
        
        for i in range(len(stampkwlist)):
            dist = stampkwlist[i]['rmax_r200c']
            dist = '{}'.format(dist)
            dist = dist.replace('.', 'p')
            stampkwlist[i]['outname'] = \
                stampname.format(mapname=mapfilen[:-5], 
                                 numsl=stampkwlist[i]['numsl'],
                                 dist=dist)
            
            print('Getting halo radii')
            #radii_mhbins = {key: [r200cvals[galids == galid] \
            #                      for galid in galids_dct[key]] \
            #                for key in galids_dct}
            nonemptykeys = {key if len(galids_dct[key]) > 0 else None \
                            for key in galids_dct}
            nonemptykeys -= {None}
            nonemptykeys = list(nonemptykeys)
            #maxradii_mhbins = {key: np.max(radii_mhbins[key]) \
            #                   for key in nonemptykeys}
            maxradii_mhbins =  {hmkey: cu.R200c_pkpc(
                                       10**(float(hmkey.split('_')[0][3:]) +\
                                            0.5),
                                       cosmopars)
                                for hmkey in nonemptykeys}
            #print('for debug: galids_dct:\n')
            #print(galids_dct)
            #print('\n')
            print('Matching radii to Mhalo bins...')
            allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
            gkeys = list(galids_dct.keys())
            keymatch = [gkeys[np.where([gid in galids_dct[key] \
                                        for key in gkeys])[0][0]] \
                        for gid in allids]
            mindist_pkpc = stampkwlist[i]['rmax_r200c'] *\
                           np.array([maxradii_mhbins[gkey] \
                                     for gkey in keymatch])
            print('... done')

            stampkwlist[i]['mindist_pkpc'] = mindist_pkpc
            stampkwlist[i]['galaxyid'] = np.array(allids)
            
        rprofkwlist = [] 
        combrprofkwlist = []
        for stampi, stampkw in enumerate(stampkwlist):
            templist_rp = []
            templist_cp = []
            
            proffile = stampkw['outname'].split('/')[-1]
            proffile = ol.pdir + 'radprof/radprof_' + proffile
        
            for masstag in galids_dct:
                if len(galids_dct[masstag]) == 0:
                    continue
                print('Setting properties for {}'.format(masstag))
                for _rkw in rprofkw_base:
                    rkw = _rkw.copy()
                    rkw['galaxyid'] = galids_dct[masstag]
                    rkw['grptag'] = masstag
                    rkw['outfile'] = proffile
                    
                    # masstag format: 'geq10.0_le10.5' or 'geq14.0'
                    minmass_Msun = 10**(float(masstag.split('_')[0][3:]))
                    minmass_Msun *= 10**0.5
                    maxdist_pkpc = stampkw['rmax_r200c'] * \
                                   cu.R200c_pkpc(minmass_Msun, cosmopars)
                    print('Using maxdist_pkpc {}'.format(maxdist_pkpc))                    
                    rbins_log_large_pkpc = 10.**(\
                                np.arange(1.,np.log10(maxdist_pkpc), 0.25))
                    rbins_pkpc_large = np.append([0.], rbins_log_large_pkpc)
                    rkw1 = rkw.copy()
                    rkw1['rbins'] = rbins_pkpc_large
                    templist_rp.append(rkw1)
                    
                    rbins_log_small_pkpc = 10.**(\
                                np.arange(1., np.log10(maxdist_pkpc), 0.1))
                    rbins_pkpc_small = np.append([0.], rbins_log_small_pkpc)
                    rkw2 = rkw.copy()
                    rkw2['rbins'] = rbins_pkpc_small
                    templist_rp.append(rkw2)
                    print('Adding radial profiles up to: {}, {} pkpc'.format(
                         rbins_pkpc_large[-1], rbins_pkpc_small[-1]))
                    
                    numradd = 2
                    
                    _templist_cp = []
                    for _ckw in combrprofkw_base:
                        ckw = _ckw.copy()
                        ckw['galaxyid'] = rkw['galaxyid']
                        _templist_cp.append(ckw)
                    templist_cp = templist_cp + [_templist_cp] * numradd
                
            rprofkwlist.append(templist_rp)
            combrprofkwlist.append(templist_cp)
            
        print('Calling create_rprofiles with:')
        print('mapslices: {}'.format(_mapslices))
        print('halo catalogue: {}'.format(halocat))
        print('args (for make_map): {}'.format(args))
        print('kwargs (for make_map): {}'.format(kwargs))
        #print('stampkwlist: {}'.format(stampkwlist))
        #print('rprofkwlist: {}'.format(rprofkwlist))
        #print('combrprofkwlist: {}'.format(combrprofkwlist))
        
        create_rprofiles(_mapslices, halocat, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=combrprofkwlist,
                     deletemaps=True, deletestamps=True, **kwargs)
        
def getprofiles_paper3_PS20tables(index):
    '''
    generate the radial profiles for the paper 3 lines and other Fe L-shell
    lines using the new (PS20) tables

    Parameters
    ----------
    index : int
        index for the line (starts at zero; for command line runs, include the
        offset for this function.)

    Returns
    -------
    None.

    '''
    lines = [ 'C  5      40.2678A',
              'C  6      33.7372A',
              'N  6      29.5343A',
              'N  6      28.7870A',
              'N  7      24.7807A',
              'O  7      21.6020A',
              'O  7      21.8044A',
              'O  7      21.8070A',
              'O  7      22.1012A',
              'O  8      18.9709A',
              'Ne 9      13.4471A',
              'Ne10      12.1375A',
              'Mg11      9.16875A',
              'Mg12      8.42141A',
              'Si13      6.64803A',
              'Fe17      17.0510A',
              'Fe17      15.2620A',
              'Fe17      16.7760A',
              'Fe17      17.0960A',
              'Fe18      16.0720A',
              ]
    
    lineind = index
    line = lines[lineind]       
    simset = 0
    
    # # test version
    #if simset == 0:
    #    simnums = ['L0025N0376']
    #    varlist = ['REFERENCE']
    #    npix = [8000]
    #    centres = [[12.5] * 3]
    #    mapslices = [4]
    #    halocats = ['catalogue_RefL0025N0376_snap27_aperture30.hdf5']   
    #    stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
    #                   'outname': None, 'rmax_r200c': 2.5,
    #                   'mindist_pkpc': None, 'galaxyid': None},
    #                   ] 
    #    galids_dcts = [sh.L0025N0376_27_Mh0p5dex_1000.galids()]
        
    if simset == 0:
        simnums = ['L0100N1504']
        varlist = ['REFERENCE']
        npix = [32000]
        centres = [[50.] * 3]
        mapslices = [16]

        halocats = ['catalogue_RefL0100N1504_snap27_aperture30.hdf5'] 
        
        stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
                        'outname': None, 'rmax_r200c': 3.5,
                        'mindist_pkpc': None, 'galaxyid': None},
                       ] 
        galids_dcts = [sh.L0100N1504_27_Mh0p5dex_1000.galids()]
        
    ptypeW = 'emission'
    snapnum = 27
    
    kwargs = {'abundsW': 'Sm', 'excludeSFRW': True, 'ptypeQ': None,
              'axis': 'z', 'periodic': True, 'kernel': 'C2',
              'log': True, 'saveres': True, 'hdf5': True,
              'simulation': 'eagle', 'ompproj': True,
              'ps20tables': True, 'ps20depletion': False,
              }
    
    halocats = [halocat if '/' in halocat else ol.pdir + halocat \
                for halocat in halocats]
    
    rprofkw_base = [{'rbins': None, 'runit': 'pkpc', 'galaxyid': None,
                     'ytype': 'mean', 'yvals': None, 
                     'separateprofiles': True, 'grptag': None,
                     'uselogvals': False}]
    combrprofkw_base = [{'ytype_out': 'mean', 'yvals_out': None, 
                         'galaxyid': None},
                        {'ytype_out': 'perc', 
                         'yvals_out': [1., 2., 5., 10., 25., 50., 75., 90.,
                                       95., 98., 99.], 
                         'galaxyid': None},
                        ]
    
    for simnum, var, _npix, centre, _mapslices, halocat, galids_dct in\
        zip(simnums, varlist, npix, centres, mapslices, halocats, galids_dcts):
        
        # snapshot data 
        L_x, L_y, L_z = (centre[0] * 2.,) * 3
        npix_x, npix_y = (_npix,) * 2
        args = (simnum, snapnum, centre, L_x, L_y, L_z,
                npix_x, npix_y, ptypeW)
        kwargs['var'] = var
        kwargs['ionW'] = line
        
        print('Calling create_rprofiles with')
        print(args)
        print(kwargs)
        #print(bins)
        print('mapslices: {}'.format(_mapslices))
        print(stampkwlist)
        print('\n')
        
        del galids_dct['geq9.0_le9.5']
        del galids_dct['geq9.5_le10.0']
        del galids_dct['geq10.0_le10.5']
        stampname = ol.pdir + 'stamps/' + \
                   'stamps_{mapname}_{numsl}slice_to-min{dist}R200c' +\
                   '_Mh0p5dex_1000_centrals_M-ge-10p5.hdf5'
        
        with h5py.File(halocat, 'r') as cat:
            cosmopars = {key: item for key, item \
                         in cat['Header/cosmopars'].attrs.items()}
            #r200cvals = np.array(cat['R200c_pkpc'])
            #galids = np.array(cat['galaxyid'])
        
        # assuming z-axis projection 
        _centre = np.copy(centre)
        _centre[2] = 3.125
        _args = (simnum, snapnum, _centre, L_x, L_y, L_z / float(_mapslices),
                 npix_x, npix_y, ptypeW)
        mapfilen = m3.make_map(*_args, nameonly=True, **kwargs)
        mapfilen = mapfilen[0]
        mapfilen = mapfilen.split('/')[-1]
        mapfilen.replace('zcen3.125', 'zcen-all')
        
        for i in range(len(stampkwlist)):
            dist = stampkwlist[i]['rmax_r200c']
            dist = '{}'.format(dist)
            dist = dist.replace('.', 'p')
            stampkwlist[i]['outname'] = \
                stampname.format(mapname=mapfilen[:-5], 
                                 numsl=stampkwlist[i]['numsl'],
                                 dist=dist)
            
            print('Getting halo radii')
            #radii_mhbins = {key: [r200cvals[galids == galid] \
            #                      for galid in galids_dct[key]] \
            #                for key in galids_dct}
            nonemptykeys = {key if len(galids_dct[key]) > 0 else None \
                            for key in galids_dct}
            nonemptykeys -= {None}
            nonemptykeys = list(nonemptykeys)
            #maxradii_mhbins = {key: np.max(radii_mhbins[key]) \
            #                   for key in nonemptykeys}
            maxradii_mhbins =  {hmkey: cu.R200c_pkpc(
                                       10**(float(hmkey.split('_')[0][3:]) +\
                                            0.5),
                                       cosmopars)
                                for hmkey in nonemptykeys}
            #print('for debug: galids_dct:\n')
            #print(galids_dct)
            #print('\n')
            print('Matching radii to Mhalo bins...')
            allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
            gkeys = list(galids_dct.keys())
            keymatch = [gkeys[np.where([gid in galids_dct[key] \
                                        for key in gkeys])[0][0]] \
                        for gid in allids]
            mindist_pkpc = stampkwlist[i]['rmax_r200c'] *\
                           np.array([maxradii_mhbins[gkey] \
                                     for gkey in keymatch])
            print('... done')

            stampkwlist[i]['mindist_pkpc'] = mindist_pkpc
            stampkwlist[i]['galaxyid'] = np.array(allids)
            
        rprofkwlist = [] 
        combrprofkwlist = []
        for stampi, stampkw in enumerate(stampkwlist):
            templist_rp = []
            templist_cp = []
            
            proffile = stampkw['outname'].split('/')[-1]
            proffile = ol.pdir + 'radprof/radprof_' + proffile
        
            for masstag in galids_dct:
                if len(galids_dct[masstag]) == 0:
                    continue
                print('Setting properties for {}'.format(masstag))
                for _rkw in rprofkw_base:
                    rkw = _rkw.copy()
                    rkw['galaxyid'] = galids_dct[masstag]
                    rkw['grptag'] = masstag
                    rkw['outfile'] = proffile
                    
                    # masstag format: 'geq10.0_le10.5' or 'geq14.0'
                    minmass_Msun = 10**(float(masstag.split('_')[0][3:]))
                    minmass_Msun *= 10**0.5
                    maxdist_pkpc = stampkw['rmax_r200c'] * \
                                   cu.R200c_pkpc(minmass_Msun, cosmopars)
                    print('Using maxdist_pkpc {}'.format(maxdist_pkpc))                    
                    rbins_log_large_pkpc = 10.**(\
                                np.arange(1.,np.log10(maxdist_pkpc), 0.25))
                    rbins_pkpc_large = np.append([0.], rbins_log_large_pkpc)
                    rkw1 = rkw.copy()
                    rkw1['rbins'] = rbins_pkpc_large
                    templist_rp.append(rkw1)
                    
                    rbins_log_small_pkpc = 10.**(\
                                np.arange(1., np.log10(maxdist_pkpc), 0.1))
                    rbins_pkpc_small = np.append([0.], rbins_log_small_pkpc)
                    rkw2 = rkw.copy()
                    rkw2['rbins'] = rbins_pkpc_small
                    templist_rp.append(rkw2)
                    print('Adding radial profiles up to: {}, {} pkpc'.format(
                         rbins_pkpc_large[-1], rbins_pkpc_small[-1]))
                    
                    numradd = 2
                    
                    _templist_cp = []
                    for _ckw in combrprofkw_base:
                        ckw = _ckw.copy()
                        ckw['galaxyid'] = rkw['galaxyid']
                        _templist_cp.append(ckw)
                    templist_cp = templist_cp + [_templist_cp] * numradd
                
            rprofkwlist.append(templist_rp)
            combrprofkwlist.append(templist_cp)
            
        print('Calling create_rprofiles with:')
        print('mapslices: {}'.format(_mapslices))
        print('halo catalogue: {}'.format(halocat))
        print('args (for make_map): {}'.format(args))
        print('kwargs (for make_map): {}'.format(kwargs))
        #print('stampkwlist: {}'.format(stampkwlist))
        #print('rprofkwlist: {}'.format(rprofkwlist))
        #print('combrprofkwlist: {}'.format(combrprofkwlist))
        
        create_rprofiles(_mapslices, halocat, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=combrprofkwlist,
                     deletemaps=False, deletestamps=False, **kwargs)

def getconvtest_paper3_PS20tables(index):
    '''
    generate the radial profiles for the paper 3 lines and other Fe L-shell
    lines using the new (PS20) tables

    Parameters
    ----------
    index : int
        index for the line (starts at zero; for command line runs, include the
        offset for this function.)

    Returns
    -------
    None.

    '''
    
    # only needed for the lines actually used in the paper
    lines = [ 'Fe17      17.0510A',
              'Fe17      15.2620A',
              'Fe17      16.7760A',
              'Fe17      17.0960A',
              'Fe18      16.0720A',
              ]
    
    lineind = index
    line = lines[lineind]       
    simset = 0
    
    # # test version
    #if simset == 0:
    #    simnums = ['L0025N0376']
    #    varlist = ['REFERENCE']
    #    npix = [8000]
    #    centres = [[12.5] * 3]
    #    mapslices = [4]
    #    halocats = ['catalogue_RefL0025N0376_snap27_aperture30.hdf5']   
    #    stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
    #                   'outname': None, 'rmax_r200c': 2.5,
    #                   'mindist_pkpc': None, 'galaxyid': None},
    #                   ] 
    #    galids_dcts = [sh.L0025N0376_27_Mh0p5dex_1000.galids()]
        
    if simset == 0:
        simnums = ['L0050N0752', 'L0025N0376',
                   'L0025N0752', 'L0025N0752']
        varlist = ['REFERENCE', 'REFERENCE',\
                   'REFERENCE', 'RECALIBRATED']
        npix = [16000, 8000, 8000, 8000]
        centres = [[25.] * 3] + [[12.5] * 3] * 3
        mapslices = [8, 4, 4, 4]
        
        halocats = ['catalogue_RefL0050N0752_snap27_aperture30.hdf5',
                    'catalogue_RefL0025N0376_snap27_aperture30.hdf5',  
                    'catalogue_RefL0025N0752_snap27_aperture30.hdf5',
                    'catalogue_RecalL0025N0752_snap27_aperture30.hdf5',
                    ]
        
        stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
                       'outname': None, 'rmax_r200c': 3.5,
                       'mindist_pkpc': None, 'galaxyid': None},
                       ]
        galids_dcts = [sh.L0050N0752_27_Mh0p5dex_1000.galids(),
                       sh.L0025N0376_27_Mh0p5dex_1000.galids(),
                       sh.L0025N0752_27_Mh0p5dex_1000.galids(),
                       sh.RecalL0025N0752_27_Mh0p5dex_1000.galids()]
        
    ptypeW = 'emission'
    snapnum = 27
    
    kwargs = {'abundsW': 'Sm', 'excludeSFRW': True, 'ptypeQ': None,
              'axis': 'z', 'periodic': True, 'kernel': 'C2',
              'log': True, 'saveres': True, 'hdf5': True,
              'simulation': 'eagle', 'ompproj': True,
              'ps20tables': True, 'ps20depletion': False,
              }
    
    halocats = [halocat if '/' in halocat else ol.pdir + halocat \
                for halocat in halocats]
    
    rprofkw_base = [{'rbins': None, 'runit': 'pkpc', 'galaxyid': None,
                     'ytype': 'mean', 'yvals': None, 
                     'separateprofiles': True, 'grptag': None,
                     'uselogvals': False}]
    combrprofkw_base = [{'ytype_out': 'mean', 'yvals_out': None, 
                         'galaxyid': None},
                        {'ytype_out': 'perc', 
                         'yvals_out': [1., 2., 5., 10., 25., 50., 75., 90.,
                                       95., 98., 99.], 
                         'galaxyid': None},
                        ]
    
    for simnum, var, _npix, centre, _mapslices, halocat, galids_dct in\
        zip(simnums, varlist, npix, centres, mapslices, halocats, galids_dcts):
        
        # snapshot data 
        L_x, L_y, L_z = (centre[0] * 2.,) * 3
        npix_x, npix_y = (_npix,) * 2
        args = (simnum, snapnum, centre, L_x, L_y, L_z,
                npix_x, npix_y, ptypeW)
        kwargs['var'] = var
        kwargs['ionW'] = line
        
        print('Calling create_rprofiles with')
        print(args)
        print(kwargs)
        #print(bins)
        print('mapslices: {}'.format(_mapslices))
        print(stampkwlist)
        print('\n')
        
        del galids_dct['geq9.0_le9.5']
        del galids_dct['geq9.5_le10.0']
        del galids_dct['geq10.0_le10.5']
        stampname = ol.pdir + 'stamps/' + \
                   'stamps_{mapname}_{numsl}slice_to-min{dist}R200c' +\
                   '_Mh0p5dex_1000_centrals_M-ge-10p5.hdf5'
        
        with h5py.File(halocat, 'r') as cat:
            cosmopars = {key: item for key, item \
                         in cat['Header/cosmopars'].attrs.items()}
            #r200cvals = np.array(cat['R200c_pkpc'])
            #galids = np.array(cat['galaxyid'])
        
        # assuming z-axis projection 
        _centre = np.copy(centre)
        _centre[2] = 3.125
        _args = (simnum, snapnum, _centre, L_x, L_y, L_z / float(_mapslices),
                 npix_x, npix_y, ptypeW)
        mapfilen = m3.make_map(*_args, nameonly=True, **kwargs)
        mapfilen = mapfilen[0]
        mapfilen = mapfilen.split('/')[-1]
        mapfilen.replace('zcen3.125', 'zcen-all')
        
        for i in range(len(stampkwlist)):
            dist = stampkwlist[i]['rmax_r200c']
            dist = '{}'.format(dist)
            dist = dist.replace('.', 'p')
            stampkwlist[i]['outname'] = \
                stampname.format(mapname=mapfilen[:-5], 
                                 numsl=stampkwlist[i]['numsl'],
                                 dist=dist)
            
            print('Getting halo radii')
            #radii_mhbins = {key: [r200cvals[galids == galid] \
            #                      for galid in galids_dct[key]] \
            #                for key in galids_dct}
            nonemptykeys = {key if len(galids_dct[key]) > 0 else None \
                            for key in galids_dct}
            nonemptykeys -= {None}
            nonemptykeys = list(nonemptykeys)
            #maxradii_mhbins = {key: np.max(radii_mhbins[key]) \
            #                   for key in nonemptykeys}
            maxradii_mhbins =  {hmkey: cu.R200c_pkpc(
                                       10**(float(hmkey.split('_')[0][3:]) +\
                                            0.5),
                                       cosmopars)
                                for hmkey in nonemptykeys}
            #print('for debug: galids_dct:\n')
            #print(galids_dct)
            #print('\n')
            print('Matching radii to Mhalo bins...')
            allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
            gkeys = list(galids_dct.keys())
            keymatch = [gkeys[np.where([gid in galids_dct[key] \
                                        for key in gkeys])[0][0]] \
                        for gid in allids]
            mindist_pkpc = stampkwlist[i]['rmax_r200c'] *\
                           np.array([maxradii_mhbins[gkey] \
                                     for gkey in keymatch])
            print('... done')

            stampkwlist[i]['mindist_pkpc'] = mindist_pkpc
            stampkwlist[i]['galaxyid'] = np.array(allids)
            
        rprofkwlist = [] 
        combrprofkwlist = []
        for stampi, stampkw in enumerate(stampkwlist):
            templist_rp = []
            templist_cp = []
            
            proffile = stampkw['outname'].split('/')[-1]
            proffile = ol.pdir + 'radprof/radprof_' + proffile
        
            for masstag in galids_dct:
                if len(galids_dct[masstag]) == 0:
                    continue
                print('Setting properties for {}'.format(masstag))
                for _rkw in rprofkw_base:
                    rkw = _rkw.copy()
                    rkw['galaxyid'] = galids_dct[masstag]
                    rkw['grptag'] = masstag
                    rkw['outfile'] = proffile
                    
                    # masstag format: 'geq10.0_le10.5' or 'geq14.0'
                    minmass_Msun = 10**(float(masstag.split('_')[0][3:]))
                    minmass_Msun *= 10**0.5
                    maxdist_pkpc = stampkw['rmax_r200c'] * \
                                   cu.R200c_pkpc(minmass_Msun, cosmopars)
                    print('Using maxdist_pkpc {}'.format(maxdist_pkpc))                    
                    rbins_log_large_pkpc = 10.**(\
                                np.arange(1.,np.log10(maxdist_pkpc), 0.25))
                    rbins_pkpc_large = np.append([0.], rbins_log_large_pkpc)
                    rkw1 = rkw.copy()
                    rkw1['rbins'] = rbins_pkpc_large
                    templist_rp.append(rkw1)
                    
                    rbins_log_small_pkpc = 10.**(\
                                np.arange(1., np.log10(maxdist_pkpc), 0.1))
                    rbins_pkpc_small = np.append([0.], rbins_log_small_pkpc)
                    rkw2 = rkw.copy()
                    rkw2['rbins'] = rbins_pkpc_small
                    templist_rp.append(rkw2)
                    print('Adding radial profiles up to: {}, {} pkpc'.format(
                         rbins_pkpc_large[-1], rbins_pkpc_small[-1]))
                    
                    numradd = 2
                    
                    _templist_cp = []
                    for _ckw in combrprofkw_base:
                        ckw = _ckw.copy()
                        ckw['galaxyid'] = rkw['galaxyid']
                        _templist_cp.append(ckw)
                    templist_cp = templist_cp + [_templist_cp] * numradd
                
            rprofkwlist.append(templist_rp)
            combrprofkwlist.append(templist_cp)
            
        print('Calling create_rprofiles with:')
        print('mapslices: {}'.format(_mapslices))
        print('halo catalogue: {}'.format(halocat))
        print('args (for make_map): {}'.format(args))
        print('kwargs (for make_map): {}'.format(kwargs))
        #print('stampkwlist: {}'.format(stampkwlist))
        #print('rprofkwlist: {}'.format(rprofkwlist))
        #print('combrprofkwlist: {}'.format(combrprofkwlist))
        
        create_rprofiles(_mapslices, halocat, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=combrprofkwlist,
                     deletemaps=True, deletestamps=True, **kwargs)
        
def get_xrayabs(index):
    '''
    generate the radial profiles for some x-ray absorption lines

    Parameters
    ----------
    index : int
        index for the line (starts at zero; for command line runs, include the
        offset for this function.)

    Returns
    -------
    None.

    '''
    
    # only needed for the lines actually used in the paper
    ions = ['o7', 'o8', 'fe17', 'ne9', 'ne10', 'c5', 'c6', 'n6', 'n7']
    snapshots = [27, 19]
    
    ionind = index // len(snapshots)
    ion = ions[ionind]
    snapnum = snapshots[index % len(snapshots)]
    

    simnums = ['L0100N1504']
    varlist = ['REFERENCE']
    npix = [32000]
    centres = [[50.] * 3] 
    mapslices = [16]
        
    hc_base = 'catalogue_RefL0100N1504_snap{}_aperture30.hdf5'
    halocats = [hc_base.format(snapnum)]
        
    stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
                   'outname': None, 'rmax_r200c': 3.5,
                   'mindist_pkpc': None, 'galaxyid': None},
                   ]
    galids_dcts = [sh.L0100N1504_27_Mh0p5dex_1000.galids() \
                   if snapnum == 27 else 
                   sh.L0100N1504_19_Mh0p5dex_1000.galids() \
                   if snapnum == 19 else None]
        
    ptypeW = 'coldens'
    
    kwargs = {'abundsW': 'Pt', 'excludeSFRW': 'T4', 'ptypeQ': None,
              'axis': 'z', 'periodic': True, 'kernel': 'C2',
              'log': True, 'saveres': True, 'hdf5': True,
              'simulation': 'eagle', 'ompproj': True,
              'ps20tables': False, 'ps20depletion': False,
              }
    
    halocats = [halocat if '/' in halocat else ol.pdir + halocat \
                for halocat in halocats]
    
    rprofkw_base = [{'rbins': None, 'runit': 'R200c', 'galaxyid': None,
                     'ytype': 'mean', 'yvals': None, 
                     'separateprofiles': True, 'grptag': None,
                     'uselogvals': False},
                    {'rbins': None, 'runit': 'R200c', 'galaxyid': None,
                     'ytype': 'perc', 
                     'yvals': [1., 2., 5., 10., 25., 50., 75., 90., 95., 
                               98., 99.], 
                     'separateprofiles': False, 'grptag': None,
                     'uselogvals': False},
                    ]   
    combrprofkw_base = [{'ytype_out': 'mean', 'yvals_out': None, 
                         'galaxyid': None},
                        {'ytype_out': 'perc', 
                         'yvals_out': [1., 2., 5., 10., 25., 50., 75., 90.,
                                       95., 98., 99.], 
                         'galaxyid': None},
                        ]
    
    for simnum, var, _npix, centre, _mapslices, halocat, galids_dct in\
        zip(simnums, varlist, npix, centres, mapslices, halocats, galids_dcts):
        
        # snapshot data 
        L_x, L_y, L_z = (centre[0] * 2.,) * 3
        npix_x, npix_y = (_npix,) * 2
        args = (simnum, snapnum, centre, L_x, L_y, L_z,
                npix_x, npix_y, ptypeW)
        kwargs['var'] = var
        kwargs['ionW'] = ion
        
        print('Calling create_rprofiles with')
        print(args)
        print(kwargs)
        #print(bins)
        print('mapslices: {}'.format(_mapslices))
        print(stampkwlist)
        print('\n')
        
        del galids_dct['geq9.0_le9.5']
        del galids_dct['geq9.5_le10.0']
        del galids_dct['geq10.0_le10.5']
        stampname = ol.pdir + 'stamps/' + \
                   'stamps_{mapname}_{numsl}slice_to-min{dist}R200c' +\
                   '_Mh0p5dex_1000_centrals_M-ge-10p5.hdf5'
        
        with h5py.File(halocat, 'r') as cat:
            cosmopars = {key: item for key, item \
                         in cat['Header/cosmopars'].attrs.items()}
            #r200cvals = np.array(cat['R200c_pkpc'])
            #galids = np.array(cat['galaxyid'])
        
        # assuming z-axis projection 
        _centre = np.copy(centre)
        _centre[2] = 3.125
        _args = (simnum, snapnum, _centre, L_x, L_y, L_z / float(_mapslices),
                 npix_x, npix_y, ptypeW)
        mapfilen = m3.make_map(*_args, nameonly=True, **kwargs)
        mapfilen = mapfilen[0]
        mapfilen = mapfilen.split('/')[-1]
        mapfilen = mapfilen.replace('zcen3.125', 'zcen-all')
        
        for i in range(len(stampkwlist)):
            dist = stampkwlist[i]['rmax_r200c']
            dist = '{}'.format(dist)
            dist = dist.replace('.', 'p')
            stampkwlist[i]['outname'] = \
                stampname.format(mapname=mapfilen[:-5], 
                                 numsl=stampkwlist[i]['numsl'],
                                 dist=dist)
            
            print('Getting halo radii')
            #radii_mhbins = {key: [r200cvals[galids == galid] \
            #                      for galid in galids_dct[key]] \
            #                for key in galids_dct}
            nonemptykeys = {key if len(galids_dct[key]) > 0 else None \
                            for key in galids_dct}
            nonemptykeys -= {None}
            nonemptykeys = list(nonemptykeys)
            #maxradii_mhbins = {key: np.max(radii_mhbins[key]) \
            #                   for key in nonemptykeys}
            maxradii_mhbins =  {hmkey: cu.R200c_pkpc(
                                       10**(float(hmkey.split('_')[0][3:]) +\
                                            0.5),
                                       cosmopars)
                                for hmkey in nonemptykeys}
            #print('for debug: galids_dct:\n')
            #print(galids_dct)
            #print('\n')
            print('Matching radii to Mhalo bins...')
            allids = [gid for key in galids_dct.keys() \
                      for gid in galids_dct[key]]
            gkeys = list(galids_dct.keys())
            keymatch = [gkeys[np.where([gid in galids_dct[key] \
                                        for key in gkeys])[0][0]] \
                        for gid in allids]
            mindist_pkpc = stampkwlist[i]['rmax_r200c'] *\
                           np.array([maxradii_mhbins[gkey] \
                                     for gkey in keymatch])
            print('... done')

            stampkwlist[i]['mindist_pkpc'] = mindist_pkpc
            stampkwlist[i]['galaxyid'] = np.array(allids)
            
        rprofkwlist = [] 
        combrprofkwlist = []
        for stampi, stampkw in enumerate(stampkwlist):
            templist_rp = []
            templist_cp = []
            
            proffile = stampkw['outname'].split('/')[-1]
            proffile = ol.pdir + 'radprof/radprof_' + proffile
        
            for masstag in galids_dct:
                if len(galids_dct[masstag]) == 0:
                    continue
                print('Setting properties for {}'.format(masstag))
                for _rkw in rprofkw_base:
                    rkw = _rkw.copy()
                    rkw['galaxyid'] = galids_dct[masstag]
                    rkw['grptag'] = masstag
                    rkw['outfile'] = proffile
                    
                    # masstag format: 'geq10.0_le10.5' or 'geq14.0'
                    ##minmass_Msun = 10**(float(masstag.split('_')[0][3:]))
                    #minmass_Msun *= 10**0.5
                    #maxdist_pkpc = stampkw['rmax_r200c'] * \
                     #              cu.R200c_pkpc(minmass_Msun, cosmopars)
                    #print('Using maxdist_pkpc {}'.format(maxdist_pkpc))
                    logbins = np.arange(-2., 
                                        np.log10(stampkw['rmax_r200c']) \
                                        - 0.001, 0.05)                    
                    rbins_R200c = 10.**(logbins)
                    rbins_R200c = np.append([0.], rbins_R200c)
                    rbins_R200c = np.append(rbins_R200c, 
                                            [stampkw['rmax_r200c']])
                    rkw1 = rkw.copy()
                    rkw1['rbins'] = rbins_R200c
                    templist_rp.append(rkw1)
                    
                    numradd = 1
                    
                    _templist_cp = []
                    
                    if rkw['separateprofiles']:
                        for _ckw in combrprofkw_base:
                            ckw = _ckw.copy()
                            ckw['galaxyid'] = rkw['galaxyid']
                            _templist_cp.append(ckw)
                    templist_cp = templist_cp + [_templist_cp] * numradd
                
            rprofkwlist.append(templist_rp)
            combrprofkwlist.append(templist_cp)
            
        print('Calling create_rprofiles with:')
        print('mapslices: {}'.format(_mapslices))
        print('halo catalogue: {}'.format(halocat))
        print('args (for make_map): {}'.format(args))
        print('kwargs (for make_map): {}'.format(kwargs))
        #print('stampkwlist: {}'.format(stampkwlist))
        #print('rprofkwlist: {}'.format(rprofkwlist))
        #print('combrprofkwlist: {}'.format(combrprofkwlist))
        
        create_rprofiles(_mapslices, halocat, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=combrprofkwlist,
                     deletemaps=False, deletestamps=False, **kwargs)
        

def getprofiles_directfbtest(index):
    '''
    generate radial profiles for the paper 3 direct feedback effect tests
    statistics  of the mean radial profiles in 10 kpc -> 0.1, 0.25 dex bins

    Parameters
    ----------
    index : int
        index for the delay time (fast, 3) and line (slow, 18) to run.

    Returns
    -------
    None.

    '''
    
    plot_lines_SB = ['c5r', 'c6', 'n6-actualr', 'n7', 'o7r', 'o7iy', 'o7f', 
                     'o8', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r']
    plot_lines_PS20 = ['Fe17      15.2620A', 'Fe17      16.7760A',
                       'Fe17      17.0510A', 'Fe17      17.0960A', 
                       'Fe18      16.0720A']
    lines = plot_lines_SB + plot_lines_PS20
    
    lineind = index // 3
    line = lines[lineind]
         
    simset = index % 3
        
    simnums = ['L0100N1504']
    varlist = ['REFERENCE']
    npix = [32000]
    centres = [[50.] * 3]
    mapslices = [16]

    halocats = ['catalogue_RefL0100N1504_snap27_aperture30.hdf5'] 
    
    ptypeW = 'emission'
    snapnum = 27
    
    stampkwlist = [{'numsl': 1, 'offset_los': 0., 'velspace': False,
                    'outname': None, 'rmax_r200c': 3.5,
                    'mindist_pkpc': None, 'galaxyid': None},
                   ] 
    galids_dcts = [sh.L0100N1504_27_Mh0p5dex_1000.galids()]
    if simset == 0:
        deltat_Myr = 3.
    elif simset == 1:
        deltat_Myr = 10.
    elif simset == 2:
        deltat_Myr = 30.
    ps20tables = line in all_lines_PS20
    
    kwargs = {'abundsW': 'Sm', 'excludeSFRW': True, 'ptypeQ': None,
              'axis': 'z', 'periodic': True, 'kernel': 'C2',
              'log': True, 'saveres': True, 'hdf5': True,
              'simulation': 'eagle', 'ompproj': True,
              'excludedirectfb': True, 'deltalogT_directfb': 0.2, 
              'deltatMyr_directfb': deltat_Myr, 
              'inclhotgas_maxlognH_snfb': -2.,
              'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
              'ps20tables': ps20tables, 'ps20depletion': False,}
    
    halocats = [halocat if '/' in halocat else ol.pdir + halocat \
                for halocat in halocats]
    
    rprofkw_base = [{'rbins': None, 'runit': 'pkpc', 'galaxyid': None,
                     'ytype': 'mean', 'yvals': None, 
                     'separateprofiles': True, 'grptag': None,
                     'uselogvals': False}]
    combrprofkw_base = [{'ytype_out': 'mean', 'yvals_out': None, 
                         'galaxyid': None},
                        {'ytype_out': 'perc', 
                         'yvals_out': [1., 2., 5., 10., 25., 50., 75., 90.,
                                       95., 98., 99.], 
                         'galaxyid': None},
                        ]
    
    for simnum, var, _npix, centre, _mapslices, halocat, galids_dct in\
        zip(simnums, varlist, npix, centres, mapslices, halocats, galids_dcts):
        
        # snapshot data 
        L_x, L_y, L_z = (centre[0] * 2.,) * 3
        npix_x, npix_y = (_npix,) * 2
        args = (simnum, snapnum, centre, L_x, L_y, L_z,
                npix_x, npix_y, ptypeW)
        kwargs['var'] = var
        kwargs['ionW'] = line
        
        print('Calling create_rprofiles with')
        print(args)
        print(kwargs)
        #print(bins)
        print('mapslices: {}'.format(_mapslices))
        print(stampkwlist)
        print('\n')
        
        del galids_dct['geq9.0_le9.5']
        del galids_dct['geq9.5_le10.0']
        del galids_dct['geq10.0_le10.5']
        stampname = ol.pdir + 'stamps/' + \
                   'stamps_{mapname}_{numsl}slice_to-min{dist}R200c' +\
                   '_Mh0p5dex_1000_centrals_M-ge-10p5.hdf5'
        
        with h5py.File(halocat, 'r') as cat:
            cosmopars = {key: item for key, item \
                         in cat['Header/cosmopars'].attrs.items()}
            #r200cvals = np.array(cat['R200c_pkpc'])
            #galids = np.array(cat['galaxyid'])
        
        # assuming z-axis projection 
        _centre = np.copy(centre)
        _centre[2] = 3.125
        _args = (simnum, snapnum, _centre, L_x, L_y, L_z / float(_mapslices),
                 npix_x, npix_y, ptypeW)
        mapfilen = m3.make_map(*_args, nameonly=True, **kwargs)
        mapfilen = mapfilen[0]
        mapfilen = mapfilen.split('/')[-1]
        mapfilen.replace('zcen3.125', 'zcen-all')
        
        for i in range(len(stampkwlist)):
            dist = stampkwlist[i]['rmax_r200c']
            dist = '{}'.format(dist)
            dist = dist.replace('.', 'p')
            stampkwlist[i]['outname'] = \
                stampname.format(mapname=mapfilen[:-5], 
                                 numsl=stampkwlist[i]['numsl'],
                                 dist=dist)
            
            print('Getting halo radii')
            #radii_mhbins = {key: [r200cvals[galids == galid] \
            #                      for galid in galids_dct[key]] \
            #                for key in galids_dct}
            nonemptykeys = {key if len(galids_dct[key]) > 0 else None \
                            for key in galids_dct}
            nonemptykeys -= {None}
            nonemptykeys = list(nonemptykeys)
            #maxradii_mhbins = {key: np.max(radii_mhbins[key]) \
            #                   for key in nonemptykeys}
            maxradii_mhbins =  {hmkey: cu.R200c_pkpc(
                                       10**(float(hmkey.split('_')[0][3:]) +\
                                            0.5),
                                       cosmopars)
                                for hmkey in nonemptykeys}
            #print('for debug: galids_dct:\n')
            #print(galids_dct)
            #print('\n')
            print('Matching radii to Mhalo bins...')
            allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
            gkeys = list(galids_dct.keys())
            keymatch = [gkeys[np.where([gid in galids_dct[key] \
                                        for key in gkeys])[0][0]] \
                        for gid in allids]
            mindist_pkpc = stampkwlist[i]['rmax_r200c'] *\
                           np.array([maxradii_mhbins[gkey] \
                                     for gkey in keymatch])
            print('... done')

            stampkwlist[i]['mindist_pkpc'] = mindist_pkpc
            stampkwlist[i]['galaxyid'] = np.array(allids)
            
        rprofkwlist = [] 
        combrprofkwlist = []
        for stampi, stampkw in enumerate(stampkwlist):
            templist_rp = []
            templist_cp = []
            
            proffile = stampkw['outname'].split('/')[-1]
            proffile = ol.pdir + 'radprof/radprof_' + proffile
        
            for masstag in galids_dct:
                if len(galids_dct[masstag]) == 0:
                    continue
                print('Setting properties for {}'.format(masstag))
                for _rkw in rprofkw_base:
                    rkw = _rkw.copy()
                    rkw['galaxyid'] = galids_dct[masstag]
                    rkw['grptag'] = masstag
                    rkw['outfile'] = proffile
                    
                    # masstag format: 'geq10.0_le10.5' or 'geq14.0'
                    minmass_Msun = 10**(float(masstag.split('_')[0][3:]))
                    minmass_Msun *= 10**0.5
                    maxdist_pkpc = stampkw['rmax_r200c'] * \
                                   cu.R200c_pkpc(minmass_Msun, cosmopars)
                    print('Using maxdist_pkpc {}'.format(maxdist_pkpc))                    
                    rbins_log_large_pkpc = 10.**(\
                                np.arange(1., np.log10(maxdist_pkpc), 0.25))
                    rbins_pkpc_large = np.append([0.], rbins_log_large_pkpc)
                    rkw1 = rkw.copy()
                    rkw1['rbins'] = rbins_pkpc_large
                    templist_rp.append(rkw1)
                    
                    rbins_log_small_pkpc = 10.**(\
                                np.arange(1., np.log10(maxdist_pkpc), 0.1))
                    rbins_pkpc_small = np.append([0.], rbins_log_small_pkpc)
                    rkw2 = rkw.copy()
                    rkw2['rbins'] = rbins_pkpc_small
                    templist_rp.append(rkw2)
                    print('Adding radial profiles up to: {}, {} pkpc'.format(
                         rbins_pkpc_large[-1], rbins_pkpc_small[-1]))
                    
                    numradd = 2
                    
                    _templist_cp = []
                    for _ckw in combrprofkw_base:
                        ckw = _ckw.copy()
                        ckw['galaxyid'] = rkw['galaxyid']
                        _templist_cp.append(ckw)
                    templist_cp = templist_cp + [_templist_cp] * numradd
                
            rprofkwlist.append(templist_rp)
            combrprofkwlist.append(templist_cp)
            
        print('Calling create_rprofiles with:')
        print('mapslices: {}'.format(_mapslices))
        print('halo catalogue: {}'.format(halocat))
        print('args (for make_map): {}'.format(args))
        print('kwargs (for make_map): {}'.format(kwargs))
        #print('stampkwlist: {}'.format(stampkwlist))
        #print('rprofkwlist: {}'.format(rprofkwlist))
        #print('combrprofkwlist: {}'.format(combrprofkwlist))
        
        create_rprofiles(_mapslices, halocat, args, stampkwlist, rprofkwlist,
                     combrprofkwlist=combrprofkwlist,
                     deletemaps=True, deletestamps=True, **kwargs)
        
if __name__ == '__main__':
    index = int(sys.argv[1])
    if not 'OMP_NUM_THREADS' in os.environ:
        raise RuntimeError('OMP_NUM_THREADS environment variable needs to be set')
    
    if index >=0 and index < 36:
        getprofiles_convtest_paper3(index)
    
    elif index < 56:
        getprofiles_paper3_PS20tables(index - 36)
    
    elif index < 61:
        getconvtest_paper3_PS20tables(index - 56)
    
    elif index < 79:
        get_xrayabs(index - 61)
    
    elif index < 133:
        getprofiles_directfbtest(index - 79)
        