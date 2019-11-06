#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:54:05 2019

@author: wijers

general prescription for selecting subsets from halo catalogues and a class to
generate standardized random samples using fixed seeds; useful for e.g.
comparing distributions of different ions using the same galaxy sample
"""

import numpy as np
import scipy.optimize as spo
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

import cosmo_utils as cu
import eagle_constants_and_units as c
import make_maps_opts_locs as ol
import projection_classes as pc 


def selectone(fo, select, cosmopars, length):
    '''
    applies a selection tuple to a set of galaxy ids
    assumes standard halo catalogue format
    '''

    if select[0] == 'galaxyid':
        groupmatches = cu.match(np.array(fo['galaxyid']), np.array(select[1]), arr2_sorted=False, arr2_index=None)
        #print(groupmatches)
        return groupmatches >= 0

    elif select[0] == 'groupid':
        groupmatches = cu.match(np.array(fo['groupid']), np.array(select[1]), arr2_sorted=False, arr2_index=None)
        return groupmatches >= 0

    elif select[0] in ['X', 'Y', 'Z']:
        pos = np.array(fo['%scom_cMpc'%select[0]])
        box = cosmopars['boxsize'] / cosmopars['h']

        try: 
            marginmultiplier = select[3]
        except IndexError:
            marginmultiplier = None
        if marginmultiplier is None:
            marginmultiplier = 1.
        margin = marginmultiplier * np.array(fo['R200c_pkpc']) * 1.e-3 / cosmopars['a']

        if select[1] is not None:
            minpos = select[1]
        else:
            minpos = 0.
        if select[2] is not None:
            maxpos = select[2]
        else:
            maxpos = box
        sel_temp_ = pos - margin >= minpos
        sel_temp_ = np.logical_and(sel_temp_, pos + margin <  maxpos)
        #print('%s selected'%select[0])
        #print('selected %i out of %i halos'%(np.sum(sel_temp_), len(sel_temp_)))
        return sel_temp_
    
    elif select[0] in ['VX', 'VY', 'VZ']:
        velocities = np.array(fo['%scom_cMpc'%select[0][1]])
        velocities *= cu.c.cm_per_mpc * cosmopars['a'] * cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1.e-5
        velocities += np.array(fo['%spec_kmps'%select[0]])
        box = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * cu.c.cm_per_mpc * cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1.e-5
        velocities %= box
        sel_temp_ = np.ones(length).astype(bool)
        if select[1] is not None:
            sel_temp_ = np.logical_and(sel_temp_, velocities >= select[1])
        if select[2] is not None:
            sel_temp_ = np.logical_and(sel_temp_, velocities <  select[2])
        return sel_temp_

    else:
        if select[0][:3] == 'log':
            selstr = select[0][3:]
            selmin = 10**select[1] if select[1] is not None else None
            selmax = 10**select[2] if select[2] is not None else None
        else:
            selstr = select[0]
            selmin = select[1]
            selmax = select[2]
        sel_temp_ = np.ones(length).astype(bool)
        prop = np.array(fo[selstr])
        if select[1] is not None:
            sel_temp_ = np.logical_and(sel_temp_, prop >= selmin)
        if select[2] is not None:
            sel_temp_ = np.logical_and(sel_temp_, prop <  selmax)
        return sel_temp_
                
def selecthalos(fo, selection):
    '''
    applies a selection to an open halo catalogue hdf5 file
    selection: list of tuples 
               (name, min/None, max/None [, margin for positions])
               names and units match hdf5 groups in the halo catalogue
               multiple tuples with the same name: one OR the other
               tuples with different names: ALL criteria must be satisfied
               margins are in units of r200c for each halo 
                 (margin > 0 -> exclude more)
               velocities are rest-frame km/s
    '''
    cosmopars = {key: fo['Header/cosmopars'].attrs[key] for key in fo['Header/cosmopars'].attrs.keys()}
    galid = np.array(fo['galaxyid'])
    sel = np.ones(len(galid)).astype(bool)
    #for sl in selection:
    #    str_temp = str(sl)
    #    if len(str_temp) > 1000:
    #        print(str_temp[:1000] + '...\n')
    #    else:
    #        print(str_temp)
    selcrit = {sl[0] for sl in selection }
    #print('For debug: selcrit\n')
    #print(selcrit)
    #print('\n')
    #print('For debug: selection [0]\n')
    #print(selection[0][0])
    #print(selection[0][1][:100])
    #print('\n')
    for crit in selcrit:
        #print('crit: %s'%crit)
        #print([sl[0] for sl in selection])
        ## doesn't work for galaxyid selection, since list/array of ids is not hashable
        #sels = {sl if sl[0] == crit else None for sl in selection}
        ## use indices for set instead, put actual selection arrays in a list
        selinds = {sli if selection[sli][0] == crit else None for sli in range(len(selection))}
        selinds -= {None}
        sels = [selection[sli] for sli in selinds]
        #print(selinds)
        #print(str(sels)[:1000])
        if None in sels:
            sels.remove(None)
        sel_temp = np.zeros(len(galid)).astype(bool)
        for select in sels:
            sel_temp = np.logical_or(sel_temp, selectone(fo, select, cosmopars, len(sel_temp)))
        print('previous selection: %i halos'%(np.sum(sel)))
        sel = np.logical_and(sel, sel_temp)
        print('current selection (incl %s):\t %i halos'%(crit, np.sum(sel)))
    return sel


def gethaloselections(halocat, selections=None, names=None):
    '''
    get different selections from a catalogue using selecthalos
    '''
    if isinstance(halocat, h5py.File):
        fi = halocat
    else:
        if '/' not in halocat:
            halocat = ol.pdir + halocat
        if halocat[-5:] != '.hdf5':
            halocat = halocat + '.hdf5'
        fi = h5py.File(halocat, 'r')
    
    if selections is None:
        selections = [[]]
    if names is None:
        names = range(len(selections))
    
    galids = np.array(fi['galaxyid'])
    return {names[i]: galids[selecthalos(fi, selections[i])] for i in range(len(selections))}

def getrandomsubset(halocat, number, selections=None, names=None, seed=None, replace=False):
    '''
    seed is only applied once; this will reproduce the same random subsets only
    if all the same names (and number of names) are used
    if no replacement and number > size of selection: returns the whole 
    selection
    '''
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    
    sels = gethaloselections(halocat, selections=selections, names=names)
    keys = list(sels.keys())
    keys.sort() # same order ensures same selections for the same random seed
    if replace:
        rnd = {name: np.random.choice(sels[name], number, replace=replace) for name in keys}
    else: # may get error 
        rnd = {}
        for name in keys:
            try:
                rnd.update({name: np.random.choice(sels[name], number, replace=replace)})
            except ValueError: # sample larger than array size
                rnd.update({name: sels[name]})

    np.random.seed() # reset random seed just in case this has side-effects
    return rnd




##########
# selection from subhalo files (with a lot of input parsing and naming)        
##########
def parse_mhalo(pname, sel, mdef='m200c', aperture=30):
    if pname != 'Mhalo':
        raise ValueError('parse_mhalo only works for pname Mhalo; was called with %s'%pname)
    msun = c.solar_mass

    try:
        md = sel[3]['mdef']
    except (KeyError, IndexError): # selection doesn't include an mdef kwarg
        md = mdef
        
    if md == 'group':
        selname = 'FOF/GroupMass'
    else:
        if md[-1] == 'c':
            part1 = 'Crit'
        elif md[-1] == 'm':
            part1 = 'Mean'
        selname = 'FOF/Group_M_%s%s'%(part1, md[:-1])
    selmin = sel[1]
    selmax = sel[2]   
    
    # covert input to cgs units
    log = 'log' in sel[0]  
    if log:
        selmin = 10**selmin if selmin is not None else -np.inf
        selmax = 10**selmax if selmax is not None else np.inf
    else:
        if selmin is None:
            selmin = -np.inf
        if selmax is None:
            selmax = np.inf    
    
    selmin *= msun
    selmax *= msun
    
    if log and sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=log%s<%s'%(sel[1], md, sel[2])
    elif log and sel[2] is None:
        name_sub = '%s<=log%s'%(sel[1], md)                      
    elif log and sel[1] is None:   
        name_sub = 'log%s<%s'%(md, sel[2])
    elif sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=%s<%s'%(sel[1], md, sel[2])
    elif sel[2] is None:
        name_sub = '%s<=%s'%(sel[1], md)                      
    elif sel[1] is None:   
        name_sub = '%s<%s'%(md, sel[2])
    
    return name_sub, selname, selmin, selmax, None

def parse_subhalo(pname, sel, mdef='200c', aperture=30):
    if pname not in ['Mstar', 'MBHap', 'SFR', 'sSFR']:
         ValueError('parse_subhalo only works for pnames Mstar, MBHap, SFR, sSFR; was called with %s'%pname)
    msun = c.solar_mass
    yr   = c.sec_per_year
    
    parttype_stars = 4
    parttype_bh    = 5
    try:
        ap = sel[3]['aperture']
    except (KeyError, IndexError): # selection doesn't include an mdef kwarg
        ap = aperture
        
    if ap == 'group':
        if pname == 'Mstar':
            selname = 'Subhalo/Stars/Mass'
            multf = msun
        elif pname == 'MBHap':
            selname = 'Subhalo/BlackHoleMass'
            multf = msun
        elif pname == 'SFR':
            selname = 'Subhalo/StarFormationRate'
            multf = msun / yr
        elif pname == 'sSFR':
            selname = ('Subhalo/StarFormationRate', 'Subhalo/Stars/Mass')
            multf = 1. / yr
    else:
        basename = 'Subhalo/ApertureMeasurements/%s/%.3ikpc'%('%s', int(ap))
        if pname == 'Mstar':
            selname = (basename%('Mass'), parttype_stars)
            multf = msun
        elif pname == 'MBHap':
            selname = (basename%('Mass'), parttype_bh)
            multf = msun
        elif pname == 'SFR':
            selname = basename%('SFR')
            multf = msun / yr
        elif pname == 'sSFR':
            selname = (basename%('SFR'), basename%('Mass'), parttype_stars)
            multf = 1. / yr
    # covert input to cgs units
    selmin = sel[1]
    selmax = sel[2]
    log = 'log' in sel[0]  
    if log:
        selmin = 10**selmin if selmin is not None else -np.inf
        selmax = 10**selmax if selmax is not None else np.inf
    else:
        if selmin is None:
            selmin = -np.inf
        if selmax is None:
            selmax = np.inf
    selmin *= multf
    selmax *= multf
    
    if ap == 'group':
        md = ap
    else:
        md = '%ikpc'%(int(ap))
    if log and sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=log%s<%s'%(sel[1], md, sel[2])
    elif log and sel[2] is None:
        name_sub = '%s<=log%s'%(sel[1], md)                      
    elif log and sel[1] is None:   
        name_sub = 'log%s<%s'%(md, sel[2])
    elif sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=%s<%s'%(sel[1], md, sel[2])
    elif sel[2] is None:
        name_sub = '%s<=%s'%(sel[1], md)                      
    elif sel[1] is None:   
        name_sub = '%s<%s'%(md, sel[2])
    
    return name_sub, selname, selmin, selmax, None

def parse_position(simfile, pname, sel, mdef='200c', aperture=30):
    if pname not in ['X', 'Y', 'Z']:
        ValueError('parse_position only works for pnames X, Y, Z; was called with %s'%pname)
    
    pmpc = c.cm_per_mpc
    if pname == 'X':
        axis = 0
    elif pname == 'Y':
        axis = 1
    elif pname == 'Z':
        axis = 2
    
    margin = 0.
    md = mdef  
    if len(sel) > 3:
        val3 = sel[3]
        if isinstance(val3, dict):
            if 'mdef' in val3.keys():
                md = val3['mdef']
        else:
            margin = val3
    if len(sel) > 4:
        val4 = sel[4]
        if isinstance(val4, dict):
            if 'mdef' in val4.keys():
                md = val4['mdef']
        else:
            margin = val4
    
    selname = ('FOF/GroupCentreOfPotential', axis)
    if margin != 0.:
        if md == 'group': # set default: FOF group does not come with a radius
            md = '200c' 
        if md[-1] == 'c':
            part1 = 'Crit'
        elif md[-1] == 'm':
            part1 = 'Mean'
        marginname = 'FOF/Group_R_%s%s'%(part1, md[:-1])
    
    # covert input to cgs units
    selmin = sel[1] * pmpc * simfile.a if sel[1] is not None else -np.inf
    selmax = sel[2] * pmpc * simfile.a if sel[2] is not None else np.inf
    
    if margin == 0. and sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=cMpc<%s'%(sel[1], sel[2])
    elif margin == 0. and sel[2] is None:
        name_sub = '%s<=cMpc'%(sel[1])                      
    elif margin == 0. and sel[1] is None:   
        name_sub = 'cMpc<%s'%(sel[2])
    elif sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=cMpc<%s-pm%sR%s'%(sel[1], sel[2], margin, md)
    elif sel[2] is None:
        name_sub = '%s<=cMpc-pm%sR%s'%(sel[1], margin, md)                      
    elif sel[1] is None:   
        name_sub = 'cMpc<%s-pm%sR%s'%(sel[2], margin, md)
    
    if margin == 0.:
        return name_sub, selname, selmin, selmax, None
    else:
        return name_sub, selname, selmin, selmax, (marginname, margin)

def parse_velocity(simfile, pname, sel, mdef='200c', aperture=30):
    if pname not in ['VX', 'VY', 'VZ']:
        ValueError('parse_velocity only works for pnames VX, VY, VZ; was called with %s'%pname)
    
    kmps = 1e5
    
    if pname == 'VX':
        axis = 0
    elif pname == 'VY':
        axis = 1
    elif pname == 'VZ':
        axis = 2
    
    margin = 0.
    if len(sel) > 3:
        val3 = sel[3]
        if isinstance(val3, dict):
            pass
        else:
            margin = val3
    if len(sel) > 4:
        val4 = sel[4]
        if isinstance(val4, dict):
            pass
        else:
            margin = val4
            
    selname = ('Subhalo/CentreOfMass', 'Subhalo/Velocity', axis)
    if margin != 0.:
        margin *= kmps
    
    # covert input to cgs units
    selmin = sel[1] * kmps if sel[1] is not None else -np.inf
    selmax = sel[2] * kmps if sel[2] is not None else np.inf
    
    if margin == 0. and sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=kmps<%s'%(sel[1], sel[2])
    elif margin == 0. and sel[2] is None:
        name_sub = '%s<=kmps'%(sel[1])                      
    elif margin == 0. and sel[1] is None:   
        name_sub = 'kpms<%s'%(sel[2])
    elif sel[1] is not None and sel[2] is not None:
        name_sub = '%s<=kpms<%s-pm%s'%(sel[1], sel[2], margin / kmps)
    elif sel[2] is None:
        name_sub = '%s<=kmps-pm%s'%(sel[1], margin / kmps)                      
    elif sel[1] is None:   
        name_sub = 'kmps<%s-pm%s'%(sel[2], margin / kmps)
    
    if margin == 0.:
        return name_sub, selname, selmin, selmax, None
    else:
        return name_sub, selname, selmin, selmax, margin
    
def parse_onesel_subfind(simfile, pname, sel, mdef='200c', aperture=30):
    '''
    helper function for parse_halosel_subfind
    '''
    
    if pname == 'Mhalo':
        return parse_mhalo(pname, sel, mdef=mdef, aperture=aperture)        
    elif pname in ['Mstar', 'MBHap', 'SFR', 'sSFR']: # various subhalo properties
        return parse_subhalo(pname, sel, mdef=mdef, aperture=aperture)    
    elif pname in ['X', 'Y', 'Z']:
        return parse_position(simfile, pname, sel, mdef=mdef, aperture=aperture)
    elif pname in ['VX', 'VY', 'VZ']:
        return parse_velocity(simfile, pname, sel, mdef=mdef, aperture=aperture)
    else:
        raise ValueError('pname %s is not a valid choice'%pname)
    
def parse_halosel_subfind(simfile, halosel, mdef='200c', aperture=30):
    '''
    given a list of single halosel tuples, 
    returns 
    a dct of tuples:
        name: 'sels: 'list of tuples for each selection criterion
                      tuple: name of hdf5 group, min, max, margin/None
                      margin: (hdf5 group name, multiplier)
                      exceptions:
                          for sSFR: 1st entry is 
                                    ( SFR name, mass name, star particle index)
                                    or
                                    (mass name, SFR name)
                                    based on aperture or group masses used
                          for Mstar: 1st entry is
                                     (mass name, star particle index) if it's
                                     an aperture
                          for MBHap: 1st entry is
                                     (mass name, BH particle index)
                          positions:
                                     (name, axis)
                          velocities:
                                     (position name, peculiar velocity name, axis)
              'name': name of the selection on this criterion
    name of selection halosel. Sorts are applied to keep names consistent 
        between runs
         
    '''
    dct_out = {}
    
    names = {'Mhalo_Msun':    'Mhalo',\
             'Mhalo_logMsun': 'Mhalo',\
             'Mstar_Msun':    'Mstar',\
             'Mstar_logMsun': 'Mstar',\
             'Mbhap_Msun':    'MBHap',\
             'Mbhap_logMsun': 'MBHap',\
             'SFR_MsunPerYr': 'SFR',\
             'SFR_logMsunPerYr': 'SFR',\
             'sSFR_PerYr':    'sSFR',\
             'sSFR_logPerYr': 'sSFR',\
             'X_cMpc':        'X',\
             'Y_cMpc':        'Y',\
             'Z_cMpc':        'Z',\
             'VX_kmps':       'VX',\
             'VY_kmps':       'VY',\
             'VZ_kmps':       'VZ',\
             }
    props = {prop: set(name if names[name] == prop else None for name in names.keys()) - {None} for (key, prop) in names.items()}
    if not np.all([sel[0] in names.keys() for sel in halosel]):
        raise ValueError('One of the halo selection criteria is not allowed/implemented:\n\tallowed:\t%s\n\tgiven:\t%s'%(sorted(list(names.keys())), [sel[0] for sel in halosel]))
    
    # selections divided into sets selecting on the same properties
    names_intermediate = {name: set([sel if names[sel[0]] == names[name] else None for sel in halosel]) - {None} for name in names.keys()}
    names_intermediate = {prop: set(sel for name in props[prop] for sel in names_intermediate[name]) for prop in props.keys()}
    
    ##debug
    #print('parse_halosel_subfind: input selection -> selection quantities mapping:')
    #print('input halosel: %s'%halosel)
    #for name in names_intermediate.keys():
    #    print('%s:\t%s'%(name, names_intermediate[name]))
    #print('')
    
    ### first run: catgorize halosel into groups selecting on the same thing
    for pname in names_intermediate.keys():
        if names_intermediate[pname] == set():
            continue # no selections for this name
        dct_out[pname] = {}
        sels = list(names_intermediate[pname])
        nameparts = []
        dct_out[pname]['sels'] = []
            
        # loop over sels in each category  
        for sel in sels:
            if sel[1] is None and sel[2] is None:
                continue
            out = parse_onesel_subfind(simfile, pname, sel, mdef=mdef, aperture=aperture)
            nameparts = nameparts + [out[0]]
            dct_out[pname]['sels'] = dct_out[pname]['sels'] + [out[1:]]
        
        if len(nameparts) > 0:
            sortlist = [(dct_out[pname]['sels'][i][0], dct_out[pname]['sels'][i][1], nameparts[i]) for i in range(len(nameparts))]
            sortlist.sort()
            nameparts_sorted = [pt[2] for pt in sortlist]
            dct_out[pname]['name'] = '%s_%s'%(pname, '_'.join(nameparts_sorted))
        else:
            dct_out[pname]['name'] = None
    ## debug
    print('parse_halosel_subfind: individually parsed inputs:')
    for pname in sorted(list(dct_out.keys())):
        print('%s:\t%s'%(pname, dct_out[pname]['name']))
        print('\t%s'%(dct_out[pname]['sels']))
    print('')
    
    pnames_anysel = list(dct_out.keys())
    pnames_anysel = set(key if len(dct_out[pname]['sels']) > 0 else None for key in pnames_anysel) - {None}
    pnames_anysel = list(pnames_anysel)
    pnames_anysel.sort()
    dct_out = {key: dct_out[key] for key in pnames_anysel}
    haloselname = 'halosel_%s_endhalosel'%('_'.join([dct_out[key]['name'] for key in pnames_anysel])) # endhalosel is to make autoparsing names easier
    
    return dct_out, haloselname
            
                
                    
                    

def selecthalos_subfindfiles(simfile, halosel, mdef='200c', aperture=30, nameonly=False, testplots=False):
    '''
    selects halos from simfile according to halosel criteria
    
    halosel: list of tuples (name, min/None, max/None [, margin, kwargs])
             name options:
              'Mhalo_[log]Msun'
              'Mstar_[log]Msun'
              'Mbhap_[log]Msun' -- rough estimate, since using aperature 
                                   instead of generally desired central BH mass
              'SFR_[log]MsunPerYr' 
              'sSFR_[log]PerYr'    SFR, M* == 0 -> never selected
                                   M* == 0 -> sSFR = np.inf (set a minimum 
                                   stellar mass to exclude these)
              'X_cMpc', 'Y_cMpc', 'Z_cMpc' -- selection on group COP
                               margin is taken in units of halo radius (mdef,
                               R200c if the FOF mass is used)
              'VX_kmps', 'VY_kmps', 'VZ_kmps' -- rest-frame velocities. Same as
                               halo position selection, but this time in 
                               velocity space 
                               uses subhalo 0 COM and velocity
                               margin is in rest-frame km/s (equivalent to 
                               adjusting the boundaries)
              star and BH criteria apply to central subhalos (subhalo 0)
             
             for position and velocity selections, min > max is interpreted as 
             a region overlapping the periodic box edge, and margins also 
             account for those boundaries. Postive margins include things
             outside the selected range, negative margins only include things
             comfortably within the selected range. 'None' is interpreted as 
             the upper or lower box boundary for these properties.
    mdef:     ('200', '500', or '2500' + 'm' or 'c') or 'group'
              mass (or radius) definitions used; group means FOF or total
              (applies to FOF data)
    aperture: aperture measurent (pkpc) or 'group' for e.g. masses
              group means the subhalo total
              (applies to subhalo data)
    kwargs:   (dict) mdef and/or aperture to use for this selection. Overrides 
              general mdef, aperature for this one selection criterion          
    returns:
    ---------------
    default:
        array of group ids (FOF indices + 1) for the selected halos
    if nameonly:
        a name for the selection (string) instead of the selected array
    '''
    
    #### input check
    allowed_apertures = [1, 3, 5, 10, 20, 30, 40, 50, 70, 100, 'group']
    allowed_mdefs = ['200m', '200c', '500m', '500c', '2500m', '2500c', 'group']
    
    if aperture != 'group':
        if int(aperture) not in allowed_apertures:
            raise ValueError('selecthalos_subfindfiles: aperature %s was not one of the allowed %s'%(aperture, allowed_apertures))
    if mdef not in allowed_mdefs:
        raise ValueError('selecthalos_subfindfiles: mdef %s was not one of the allowed %s'%(mdef, allowed_mdefs))
    
    selections, outname = parse_halosel_subfind(simfile, halosel, mdef=mdef, aperture=aperture)
    
    if nameonly:
        return outname
    
    pnames = list(selections.keys())
    pnames_cat = {pname: 'FOF' if 'FOF' in selections[pname]['sels'][0][0] else \
                         'sub' if 'Subhalo' in selections[pname]['sels'][0][0] else \
                         'classerror' if not isinstance(selections[pname]['sels'][0][0], tuple) else \
                         'FOF' if 'FOF' in selections[pname]['sels'][0][0][0] else \
                         'sub' if 'Subhalo' in selections[pname]['sels'][0][0][0] else \
                         'classerror' \
                  for pname in pnames}
    
    selections_FOF = list(set(pname if pnames_cat[pname] == 'FOF' else None for pname in pnames) - {None}) 
    selections_sub = list(set(pname if pnames_cat[pname] == 'sub' else None for pname in pnames) - {None})
    selections_err = list(set(pname if pnames_cat[pname] == 'classerror' else None for pname in pnames) - {None})

    ## debug
    print('Overview of parsed input:')
    print('FOF criteria: %s'%selections_FOF)
    print('Suhalo criteria: %s'%selections_sub)
    for pname in sorted(list(selections.keys())):
        print('%s:\t %s'%(pname, selections[pname]))
    
    if len(selections_err) > 0:
        raise RuntimeError('Halo selections yielded hdf5 names not in FOF or Subhalo groups:\n\tproperties:\t%s\n\tnames:\t%s'%(selections_err, [selections[pname]['sels'][0][0] for pname in selections_err]))
    
    # generate simfile for subfind file read-in
    try:
        simfile_halos = pc.Simfile(simfile.simnum, simfile.snapnum, simfile.var, simulation=simfile.simulation, file_type='sub')
    except Exception as exp1:
        print('Error on subfind catalogue Simfile.__init__:')
        raise exp1
    
    subinds = simfile_halos.readarray('FOF/FirstSubhaloID', rawunits=True).astype(int)
    #print(subinds.dtype)
    #print(subinds)
    outsel  = np.ones(len(subinds)).astype(bool)
    
    # Mhalo, X, Y, Z
    if len(selections_FOF) > 0:
        for pname in selections_FOF:
            sel_list = selections[pname]['sels']
            subsel = np.zeros(len(outsel)).astype(bool)
            
            for sel in sel_list:
                if pname == 'Mhalo': # parsed output is (hdf5 name, min, max)
                    prop = simfile_halos.readarray(sel[0], rawunits=True)
                    conv = simfile_halos.CGSconvtot
                    minv = sel[1] / conv
                    maxv = sel[2] / conv
                    subsubsel = prop >= minv
                    subsubsel &= prop < maxv
                
                elif pname in ['X', 'Y', 'Z']: # parsed output is ((hdf5 name, axis), min, max, None / (hdf5 name margin, multiplier))
                    prop = simfile_halos.readarray(sel[0][0], rawunits=True)[:, sel[0][1]]
                    conv = simfile_halos.CGSconvtot
                    minv = sel[1] / conv
                    maxv = sel[2] / conv
                    boxsize = (simfile_halos.boxsize / simfile_halos.h * simfile.a * c.cm_per_mpc) / conv # in cgs units -> raw units
                    # +- infinity doesn't work with modulo (NaN) -> interpret as box boundaries (may introduce fp errors, but that's just too bad)
                    if minv == -np.inf:
                        minv = 0.
                    if maxv == np.inf:
                        maxv = boxsize
                    minv %= boxsize
                    maxv %= boxsize
                    #prop %= boxsize # shouldn't be necessary
                    
                    #print('minv, maxv: %s, %s'%(minv, maxv))
                    if sel[3] is not None:
                        marg = simfile_halos.readarray(sel[3][0], rawunits=True)
                        mconv = simfile_halos.CGSconvtot
                        marg *= (mconv * sel[3][1] / conv) # margin * mutliplier in prop units
                        
                        region_size = maxv - minv if maxv >= minv else maxv + boxsize - minv
                        #print('region_size: %s'%region_size)
                        if sel[3][1] < 0.: # margins might be large enough for apparent min/max values to switch place (naively selecting everything but the region center)
                            nosel = region_size <= 2. * np.abs(marg) * (1. + 1e-8)
                            if np.sum(nosel) > 0:
                                print('selecthalos_subfindfiles: for property %s, margin %s x %s is large enough that some halos are too large to be included at any position'%(pname, sel[3][1], sel[3][0]))
                        else: 
                            nosel = slice(None, 0, None)
                        if  sel[3][1] > 0.: # margins might be large enough for apparent min/max values to switch place (naively selecting everything but a small region furthest from the desired area)
                            allsel = boxsize - region_size <= 2. * marg * (1. + 1e-8)
                        else:
                            allsel = slice(None, 0, None)
                        
                        # min/max -> arrays in 0 - box size range
                        minv = (minv - marg) % boxsize
                        maxv = (maxv + marg) % boxsize
                        #print('Pos. selection minv, maxv: %s, %s'%(minv, maxv))
                        #print('allsel: %s'%allsel)
                        #print('nosel: %s'%nosel)
                        # edge overlap is now a per halo issue; exclude halos too large for the region altogether
                        selor  = minv > maxv
                        seland = np.logical_not(selor)
                        
                        subsubsel = np.zeros(len(subsel)).astype(bool)
                        subsubsel[seland] =  minv[seland] <= prop[seland]
                        subsubsel[seland] &= maxv[seland] >  prop[seland]
                        subsubsel[selor]  =  minv[selor]  <= prop[selor]
                        subsubsel[selor]  |= maxv[selor]  >  prop[selor]
                        subsubsel[allsel] = True
                        subsubsel[nosel]  = False
                        
                        if testplots:
                            bins = np.arange(0., 25.05, 0.1)
                            h = simfile_halos.h
                            plt.hist(prop / h, log=True, label='all', bins=bins, alpha=0.5)
                            plt.hist(prop[subsubsel] / h, log=True, label='selected', bins=bins, alpha=0.5)
                            plt.hist(minv[seland] / h, log=True, label='minv seland', bins=bins, histtype='step')
                            plt.hist(maxv[seland] / h, log=True, label='maxv seland', bins=bins, histtype='step')
                            plt.hist(minv[selor] / h, log=True, label='minv selor', bins=bins, histtype='step')
                            plt.hist(maxv[selor] / h, log=True, label='maxv selor', bins=bins, histtype='step')
                            plt.legend()
                            plt.show()
                        
                        #print('sum nosel: %s'%(len(subsubsel[nosel])))
                        #print('sum allsel: %s'%(len(subsubsel[allsel])))
                        #print('sum seland: %s'%(len(subsubsel[seland])))
                        #print('sum selor: %s'%(len(subsubsel[selor])))
                        #print('length subsubsel: %s'%(len(subsubsel)))
                        del marg
                        del selor
                        del seland
                        del allsel
                        del nosel
                    else:
                        if minv > maxv: # |####|max_______|min###|
                            subsubsel = prop >= minv
                            subsubsel |= prop < maxv
                        else: # |____|min######|max__|
                            subsubsel = prop >= minv
                            subsubsel &= prop < maxv
                    
                else:
                    raise RuntimeError('%s was misclassified as a FOF property'%pname)               
                
                subsel |= subsubsel
            outsel &= subsel
        del subsel
        del subsubsel
        del prop
    
    if len(selections_sub) > 0:
        # some FOF halos have no subhalos -> remove those from the selection first
        hassubhalos = simfile_halos.readarray('FOF/NumOfSubhalos', rawunits=True) > 0
        outsel &= hassubhalos
        subinds[np.logical_not(hassubhalos)] = 0 # will be a valid index, and any selection on those halos will be ignored anyway  
        del hassubhalos
        
        # get FOF subhalo 0 subhalo indices
        # apply selections to that subset of subhalos
        # logical_and with FOF selection 
        for pname in selections_sub:
            sel_list = selections[pname]['sels']
            #print(sel_list)
            subsel = np.zeros(len(outsel)).astype(bool)
            
            for sel in sel_list: 
                if pname == 'sSFR': # parsed output is (hdf5 name, min, max)
                    prop = simfile_halos.readarray(sel[0][0], rawunits=True)[subinds] # SFR
                    conv = simfile_halos.CGSconvtot
                    if len(sel[0]) == 3: # aperture mass
                        mass = simfile_halos.readarray(sel[0][1], rawunits=True)[subinds, sel[0][2]]
                        mconv = simfile_halos.CGSconvtot
                    elif len(sel[0]) == 2: # subhalo mass
                        mass = simfile_halos.readarray(sel[0][1], rawunits=True)[subinds]
                        mconv = simfile_halos.CGSconvtot
                    
                    prop /= mass # may yield NaN; in those cases, all comparisons seem to be False (includinf np.NaN == np.NaN)
                    conv /= mconv
                    
                    minv = sel[1] / conv
                    maxv = sel[2] / conv
                    subsubsel = prop >= minv
                    subsubsel &= prop < maxv
                    del mass
                    
                    if testplots:
                        minp = min(np.log10(np.min(prop[np.logical_and(np.isfinite(prop), prop > 0)])), -5.)
                        maxp = np.log10(np.max(prop[np.isfinite(prop)]))
                        num = 100
                        plt.hist(np.max([np.log10(prop), -4.9*np.ones(len(prop))], axis=0), bins=np.arange(minp, maxp + 0.5 * (maxp - minp) / float(num), (maxp - minp) / float(num)), log=True, color='blue', alpha=0.5, label='all')
                        plt.hist(np.max([np.log10(prop[subsubsel]), -4.9*np.ones(np.sum(subsubsel))], axis=0), bins=np.arange(minp, maxp + 0.5 * (maxp - minp) / float(num), (maxp - minp) / float(num)), log=True, color='orange', alpha=0.5, label='subselection sSFR')
                        plt.legend()
                        plt.xlabel('log10 sSFR (-4.9 -> no SF and/or stellar mass)')
                        plt.title('sSFR selection')
                        plt.ylabel('# subhalos (centrals)')
                        plt.axvline(np.log10(minv), linestyle='dotted')
                        plt.axvline(np.log10(maxv), linestyle='dotted')
                        plt.show()
                    
                elif pname in ['VX', 'VY', 'VZ']: #((hdf5name position, hdf5name velocity, axis), min, max, None/margin)
                    pos   = simfile_halos.readarray(sel[0][0], rawunits=True)[subinds, sel[0][2]]
                    pconv = simfile_halos.CGSconvtot
                    cosmopars = {'z': simfile_halos.z, 'a': simfile_halos.a, 'h': simfile_halos.h,\
                                 'boxsize': simfile_halos.boxsize, 'omegab': simfile_halos.omegab,\
                                 'omegam': simfile_halos.omegam, 'omegalambda': simfile_halos.omegalambda}                   
                    hf   = cu.Hubble(simfile_halos.z, cosmopars=cosmopars)
                    
                    vel  = simfile_halos.readarray(sel[0][1], rawunits=True)[subinds, sel[0][2]]
                    conv = simfile_halos.CGSconvtot
                    #print('V selection v conv: %s'%conv)
                    
                    pos *= (pconv * hf / conv)
                    #print('V selection pos min/max: %s, %s'%(np.min(pos), np.max(pos)))
                    #print('V selection vpec min/max: %s, %s'%(np.min(vel), np.max(vel)))
                    vel += pos
                    del pos
                    prop = vel
                    del vel
                    
                    if testplots:
                        plt.hist(prop, bins=200, log=True)
                        plt.xlabel('LOS velocity [rest-frame km/s]')
                        plt.show()
                        
                    boxsize = (simfile_halos.boxsize / simfile_halos.h * simfile_halos.a * c.cm_per_mpc * hf) / conv 
                    minv = sel[1] / conv
                    maxv = sel[2] / conv
                    #print('V selection boxsize: %s'%(boxsize))
                     # in cgs units
                    # +- infinity doesn't work with modulo (NaN) -> interpret as box boundaries (may introduce fp errors, but that's just too bad)
                    if minv == -np.inf:
                        minv = 0.
                    if maxv == np.inf:
                        maxv = boxsize
                    minv %= boxsize
                    maxv %= boxsize
                    prop %= boxsize
                    #print('V selection: minv, maxv: %s, %s'%(minv, maxv))
                    
                    nosel = False
                    allsel = False
                    if sel[3] is not None:
                        marg = sel[3] / conv
                        region_size = maxv - minv if maxv >= minv else maxv + boxsize - minv
                        if marg < 0.: # margins might be large enough for apparent min/max values to switch place (naively selecting everything but the region center)
                            nosel = region_size <= 2. * np.abs(marg)
                            if nosel:
                                print('selecthalos_subfindfiles: for property %s, margin %s is large enough that all halos are excluded'%(pname, sel[3]))
                            
                        if marg > 0.: # margins might be large enough for apparent min/max values to switch place (naively selecting everything but a small region furthest from the desired area)
                            allsel = boxsize - region_size <= 2. * marg
                        
                        minv -= marg
                        minv %= boxsize
                        maxv += marg
                        maxv %= boxsize
                    
                    #print('V selection using minv, maxv: %s, %s'%(minv, maxv))
                    
                    if allsel:
                        subsubsel = np.ones(len(outsel)).astype(bool)
                        #print('V selection: allsel case')
                    elif nosel:
                        subsubsel = np.zeros(len(outsel)).astype(bool)
                        #print('V selection: nosel case')
                    elif minv > maxv: # |####|max_______|min###|
                        #print('V selection: or sel case')
                        subsubsel = (prop >= minv)
                        subsubsel |= (prop < maxv)
                        #print('V selection: selected %s / %s'%(np.sum(subsubsel), len(subsubsel)))
                    else: # |____|min######|max__|
                        #print('V selection: and sel case')
                        subsubsel = (prop >= minv)
                        subsubsel &= (prop < maxv)
                            
                else: #Mstar, MBHap, SFR
                    #print(sel)
                    if isinstance(sel[0], tuple): # name, axis/index tuple
                        prop = simfile_halos.readarray(sel[0][0], rawunits=True)[subinds, sel[0][1]]
                    else: # just a name
                        prop = simfile_halos.readarray(sel[0], rawunits=True)[subinds]
                    conv = simfile_halos.CGSconvtot
                    minv = sel[1] / conv
                    maxv = sel[2] / conv
                    subsubsel = prop >= minv
                    subsubsel &= prop < maxv
                    
                subsel |= subsubsel
            outsel &= subsel
        del subsel
        del subsubsel
        del prop
    
    # add one to selected FOF indices to get group numbers
    # return group numbers
    out = np.where(outsel)[0] + 1
    return out


###############################################################################
#            Standardized sets of halos using getrandomsubset                 #
###############################################################################

class Galaxyselector:
    '''
    sets up a galaxy catalogue and selector, can be used to generate a set of 
    galaxy ids and check the selections used for them
    '''
    def __init__(self, halocat, number=None, selections=None, names=None, seed=None, replace=False):
        if selections is None:
            self.selections = [[]]
        else:
            self.selections = selections
        if names is None:
            self.names = range(len(self.selections))
        else:
            self.names = names
        self.seed = seed
        self.number = number
        self.replace = replace
        self.halocat = halocat
        if '/' not in self.halocat:
            self.halocat = ol.pdir + self.halocat
        if self.halocat[-5:] != '.hdf5':
            self.halocat = self.halocat + '.hdf5'
            
        if self.number is None:
            self.getgalid = self.getgalid_all
        else:
            self.get_galid = self.getgalid_rnd
    
    def galids(self, name=None):
        if not hasattr(self, 'galid_dct'):
            self.get_galid()
        if name is None and len(self.galid_dct) == 1:
            return self.galid_dct[0]
        elif name is None:
            return self.galid_dct
        else:
            return self.galid_dct[name]
    
    def getgalid_all(self):
        self.galid_dct = gethaloselections(self.halocat, selections=self.selections, names=self.names)
    def getgalid_rnd(self):
        self.galid_dct = getrandomsubset(self.halocat, self.number, selections=self.selections, names=self.names, seed=self.seed, replace=self.replace)


# L0025N0752Recal, snap 11. Minimum halo mass 10**8 Msun
halocat_L0025N0752Recal_11 = ol.pdir + 'catalogue_RecalL0025N0752_snap11_aperture30_inclsatellites.hdf5'

L0025N0752Recal_11_all_1000 = Galaxyselector(halocat_L0025N0752Recal_11, number=1000, seed=0)

L0025N0752Recal_11_wstars_1000 = Galaxyselector(halocat_L0025N0752Recal_11, number=1000, selections=[[('Mstar_Msun', 10**5., None)]], seed=0)
L0025N0752Recal_11_mhmin_1000 = Galaxyselector(halocat_L0025N0752Recal_11, number=1000, selections=[[('M200c_Msun', 10**9.5, None)]], seed=0)

Mst_edges = np.array([7., 7.5, 8., 8.5, 9., 9.5, 10.])
Mst_mins = list(Mst_edges)
Mst_maxs = list(Mst_edges[1:]) + [None]
Mst_sels = [[('Mstar_Msun', 10**Mst_mins[i], 10**Mst_maxs[i])] if Mst_maxs[i] is not None else\
            [('Mstar_Msun', 10**Mst_mins[i], None)]\
            for i in range(len(Mst_mins))]
Mst_names =[ 'geq%s_le%s'%(Mst_mins[i], Mst_maxs[i]) if Mst_maxs[i] is not None else\
             'geq%s'%(Mst_mins[i])\
            for i in range(len(Mst_mins))]

L0025N0752Recal_11_all_Mstar0p5dex = Galaxyselector(halocat_L0025N0752Recal_11, selections=Mst_sels, names=Mst_names, number=200, seed=0)


Mh_edges = np.array([9., 9.5, 10., 10.5, 11., 11.5, 12.])
Mh_mins = list(Mh_edges)
Mh_maxs = list(Mh_edges[1:]) + [None]
Mh_sels = [[('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i])] if Mh_maxs[i] is not None else\
            [('M200c_Msun', 10**Mh_mins[i], None)]\
            for i in range(len(Mh_mins))]
Mh_names =[ 'geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
             'geq%s'%(Mh_mins[i])\
            for i in range(len(Mst_mins))]

L0025N0752Recal_11_Mh0p5dex = Galaxyselector(halocat_L0025N0752Recal_11, selections=Mh_sels, names=Mh_names, number=100, seed=0)


# L0100N1504 snapshot 27, all centrals
halocat_L0100N1504_27 = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'

Mh_edges = np.array([9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.])
Mh_mins = list(Mh_edges)
Mh_maxs = list(Mh_edges[1:]) + [None]
Mh_sels = [[('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i])] if Mh_maxs[i] is not None else\
            [('M200c_Msun', 10**Mh_mins[i], None)]\
            for i in range(len(Mh_mins))]
Mh_names =[ 'geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
             'geq%s'%(Mh_mins[i])\
            for i in range(len(Mh_mins))]

L0100N1504_27_Mh0p5dex = Galaxyselector(halocat_L0100N1504_27, selections=Mh_sels, names=Mh_names, number=200, seed=0)
L0100N1504_27_Mh0p5dex_1000 = Galaxyselector(halocat_L0100N1504_27, selections=Mh_sels, names=Mh_names, number=1000, seed=0)
L0100N1504_27_Mh0p5dex_7000 = Galaxyselector(halocat_L0100N1504_27, selections=Mh_sels, names=Mh_names, number=7000, seed=0)
L0100N1504_27_Mh0p5dex_1000 = Galaxyselector(halocat_L0100N1504_27, selections=Mh_sels, names=Mh_names, number=1000, seed=0)
L0100N1504_27_Mh0p5dex_100 = Galaxyselector(halocat_L0100N1504_27, selections=Mh_sels, names=Mh_names, number=100, seed=0)


    

def getmstarbins(m200cbins=(10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.0),\
                 halocat=halocat_L0100N1504_27,\
                 method='maxpurefrac', plot=True):
    '''
    for a halo catalogue halocat and bins in logM200c_Msun
    get the best bins in logMstar_Msun according to the chosen method
    bins are optimized to 0.1 dex round numbers
    
    methods: 
        'edgemedian':  median stellar mass around the halo masss bin edges
        'maxpurefrac-<subopt>': optimize the stellar-mass halo-mass confusion 
                       matrix (maximally pure halo mass sample in a stellar 
                       mass bin)
                       subopts use the minimum ('min'), product ('prod') or sum 
                       ('sum') of the galaxy fraction in the stellar mass bins 
                       that fall into the corresponding halo mass bins 
    '''
    binround = 0.1
    
    with h5py.File(halocat, 'r') as cat:
        m200c = np.log10(np.array(cat['M200c_Msun']))
        mstar = np.log10(np.array(cat['Mstar_Msun']))
    
    m200cbins = np.array(m200cbins)
    expand = (np.floor(np.min(m200c) / binround) * binround, np.ceil(np.max(m200c) / binround) * binround)
    m200cinds = np.digitize(m200c, m200cbins, right=True)
    m200cbins = np.append(expand[0], m200cbins)
    m200cbins = np.append(m200cbins, expand[1])
    
    xinds = [np.where(m200cinds == i) for i in range(len(m200cbins) - 1)]
    binnedvals = [(m200c[xind], mstar[xind]) for xind in xinds]
    
    firstguess = np.array([np.median(mstar[np.abs(m200c - _bin) < 0.5 * binround]) for _bin in m200cbins])
    mstar_rev =  (np.floor(np.min(mstar) / binround) * binround, np.ceil(np.max(mstar) / binround) * binround)
    firstguess[0] = mstar_rev[0]
    firstguess[-1] = mstar_rev[-1]
    
    if method == 'edgemedian':
        outbins_mstar = firstguess
        
    elif method.startswith('maxpurefrac'):        
        def minfunc(midbins):
            #print(midbins)
            tempbins = np.copy(firstguess)
            tempbins[1:-1] = midbins
            #print(tempbins)
            if np.any(np.diff(tempbins) <= 0.): # penalty for non-monotonic bins
                return 2. * (len(tempbins) - 1)
            _xycounts = np.array([np.histogram(pair[1], bins=tempbins)[0] for pair in binnedvals]).astype(np.float)
            _xycounts /= np.sum(_xycounts, axis=0)[np.newaxis, :] # purity of halo mass sample at fixed stellar mass
            if np.any(np.isnan(_xycounts)): # no halos in a stelar mass bin ->  divide by zero
                return 2. * (len(tempbins) - 1)
            # sum opt.
            if method.endswith('sum'):
                optval = np.sum([_xycounts[i, i] for i in range(_xycounts.shape[0])])
            # product opt.
            elif method.endswith('prod'):
                optval = np.prod([_xycounts[i, i] for i in range(_xycounts.shape[0])])
            elif method.endswith('min'):
                optval = np.min([_xycounts[i, i] for i in range(_xycounts.shape[0])])
            #print(optval)
            return - 1. * optval # minimization -> make negative
       
        binfit = spo.minimize(minfunc, x0=firstguess[1:-1], method='COBYLA') # just use defaults; the exact values aren't critical
        if binfit.success:
            outbins_mstar = np.copy(firstguess)
            outbins_mstar[1:-1] = binfit.x
            print(binfit.x)
        else:
            print('Bin fitting failed:')
            print(binfit.message)
            return binfit
        
    xycounts = np.array([np.histogram(pair[1], bins=outbins_mstar)[0] for pair in binnedvals])
    
    if plot:
        xynorm = xycounts.astype(float) / np.sum(xycounts, axis=0)[np.newaxis, :]
        cmap = 'viridis'
        vmin = 0.
        vmax = 1.
        fontsize = 12
        xmin = 9.0
        ymin = 6.3
        xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
        ylabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{\star}} \; [\mathrm{M}_{\odot}]$'
        clabel = r'fraction at fixed $\mathrm{M}_{\star}$'
        
        plotbins_x = np.copy(m200cbins)
        plotbins_x[0] = max(plotbins_x[0], xmin)
        plotbins_y = np.copy(outbins_mstar)
        plotbins_y[0] = max(plotbins_y[0], ymin)
        
        fig = plt.figure(figsize=(5.5, 5.))
        grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[10., 1.])
        ax = fig.add_subplot(grid[0])
        cax = fig.add_subplot(grid[1])
        
        img = ax.pcolormesh(plotbins_x, plotbins_y, xynorm.T, cmap=cmap,\
                            vmin=vmin, vmax=vmax, rasterized=True)
        
        ax.set_xticks(m200cbins[1:])
        ax.set_yticks(outbins_mstar[1:])
        ax.set_xlim((plotbins_x[0], plotbins_x[-1]))
        ax.set_ylim((plotbins_y[0], plotbins_y[-1]))
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1, which='both')
        
        for i in range(len(plotbins_x) - 1):
            for j in range(len(plotbins_y) - 1):
                xcoord = 0.5 * (plotbins_x[i] + plotbins_x[i + 1])
                ycoord = 0.5 * (plotbins_y[j] + plotbins_y[j + 1])
                num = xycounts[i, j]
                if num > 1000:
                    rotation = 90
                else:
                    rotation = 0 
                ax.text(xcoord, ycoord, xycounts[i, j], fontsize=fontsize,\
                        color='gray', rotation = rotation,\
                        horizontalalignment='center', verticalalignment='center')
        plt.colorbar(img, cax=cax, orientation='vertical')
        cax.set_ylabel(clabel, fontsize=fontsize)
        cax.tick_params(labelsize=fontsize - 1, which='both')
        plt.show()
        
    return outbins_mstar, xycounts


### selected stellar mass bins (rounded versions of maxpurefrac-sum and -prod):
Mstar_edges = np.array([8.7,  9.7, 10.3, 10.8, 11.1, 11.3, 11.5, 11.7])
Mstar_sels = [[('Mstar_Msun', 10**Mstar_edges[i], 10**Mstar_edges[i + 1])] \
              for i in range(len(Mstar_edges) - 1)]
Mstar_names =['geq%s_le%s'%(Mstar_edges[i], Mstar_edges[i + 1]) \
              for i in range(len(Mstar_edges) - 1)]

L0100N1504_27_Mstar_Mhbinmatch_1000 = Galaxyselector(halocat_L0100N1504_27, selections=Mstar_sels, names=Mstar_names, number=1000, seed=0)

        
###############################################################################
#                                  Tests                                      #
###############################################################################

def testhaloselection_subfind(num=0):
    '''
    runs a number of tests of the halo selection scheme 
    returns plots and True/False
    '''
    
    # where to put the plots
    testdir = '/net/luttero/data2/imgs/CGM/haloselection_test/'
    
    sf_sub = pc.Simfile('L0025N0376', 19, 'REFERENCE', file_type='sub')
    # rest-frame Hubble flow = 1503.787421799456 km/s
    cmpc_to_kmps = 1503.787421799456 / 25.
    subinds = sf_sub.readarray('FOF/FirstSubhaloID', rawunits=True).astype(int)
    hassubs = sf_sub.readarray('FOF/NumOfSubhalos', rawunits=True) > 0
    subinds_sel = subinds[hassubs]
    groupnums = sf_sub.readarray('Subhalo/GroupNumber', rawunits=True).astype(int)
    subgroupnums = sf_sub.readarray('Subhalo/SubGroupNumber', rawunits=True).astype(int)
    
    deltap1 = 0.1
    #deltap2 = 0.1
    success = None
    
    if num == 0:
        dims = 1
        nd1  = 1
        name_intent = 'Mstar 10^8 -- 10^9 Msun'
        name_out = selecthalos_subfindfiles(sf_sub, [('Mstar_Msun', 10**8, 10**9)], nameonly=True)
        out = selecthalos_subfindfiles(sf_sub, [('Mstar_Msun', 10**8, 10**9)], nameonly=False)
        
        prop1 = np.log10(sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/030kpc')[:, 4] / c.solar_mass)
        min1 = 8.
        max1 = 9.
        pmin1 = 5.
        pmax1 = 13.
        xlabel = r'$\log_{10} \, M_{*} \; [\mathrm{M}_{\odot}]$'
        
        prop1_all = prop1[subinds_sel] 
        prop1_sel = prop1[subinds[out - 1]]
        target = groupnums[np.logical_and(np.logical_and(8. <= prop1, 9. > prop1), subgroupnums==0)]
    
    elif num == 1:
        dims = 1
        nd1  = 1
        name_intent = 'Mstar < 10^9.5 Msun'
        name_out = selecthalos_subfindfiles(sf_sub, [('Mstar_logMsun', None, 9.5)], nameonly=True)
        out = selecthalos_subfindfiles(sf_sub, [('Mstar_logMsun', None, 9.5)], nameonly=False)
        
        prop1 = np.log10(sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/030kpc')[:, 4] / c.solar_mass)
        min1 = -np.inf
        max1 = 9.5
        pmin1 = 5.
        pmax1 = 13.
        xlabel = r'$\log_{10} \, M_{*} \; [\mathrm{M}_{\odot}]$'
        
        prop1_all = prop1[subinds_sel] 
        prop1_sel = prop1[subinds[out - 1]]
        target = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums==0)]

    elif num == 2:
        dims = 1
        nd1  = 1
        name_intent = 'Mstar (5 kpc) < 10^9.5 Msun'
        name_out = selecthalos_subfindfiles(sf_sub, [('Mstar_logMsun', None, 9.5)], nameonly=True, aperture=5)
        out = selecthalos_subfindfiles(sf_sub, [('Mstar_logMsun', None, 9.5)], nameonly=False, aperture=5)
        
        prop1 = np.log10(sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/005kpc')[:, 4] / c.solar_mass)
        min1 = -np.inf
        max1 = 9.5
        pmin1 = 5.
        pmax1 = 13.
        xlabel = r'$\log_{10} \, M_{*} \; [\mathrm{M}_{\odot}]$'
        
        prop1_all = prop1[subinds_sel] 
        prop1_sel = prop1[subinds[out - 1]]
        target = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums==0)]
   
    elif num == 3:
        dims = 1
        nd1  = 2
        name_intent = 'M200c 10^9 -- 10^9.5, > 10^10.5 Msun'
        name_out = selecthalos_subfindfiles(sf_sub, [('Mhalo_Msun', 10**9, 10**9.5), ('Mhalo_logMsun', 10.5, None)], nameonly=True)
        out = selecthalos_subfindfiles(sf_sub, [('Mhalo_Msun', 10**9, 10**9.5), ('Mhalo_logMsun', 10.5, None)], nameonly=False)
        
        prop1 = np.log10(sf_sub.readarray('FOF/Group_M_Crit200') / c.solar_mass)
        min1 = [9., 10.5]
        max1 = [9.5, np.inf]
        pmin1 = 7.
        pmax1 = 15.
        xlabel = r'$\log_{10} \, M_{200c} \; [\mathrm{M}_{\odot}]$'
        
        prop1_all = prop1
        prop1_sel = prop1[out - 1]
        target = np.where(np.logical_or(np.logical_and(min1[0] <= prop1, max1[0] > prop1), np.logical_and(min1[1] <= prop1, max1[1] > prop1)))[0] + 1
   
    elif num == 4:
        dims = 1
        nd1  = 1
        name_intent = 'sSFR (40 kpc) >= 10^-11 yr^-1'
        name_out = selecthalos_subfindfiles(sf_sub, [('sSFR_logPerYr', -11., None)], nameonly=True, aperture=40)
        out = selecthalos_subfindfiles(sf_sub, [('sSFR_logPerYr', -11., None)], nameonly=False, aperture=40)
        #return out
        
        prop1 = np.log10((sf_sub.readarray('Subhalo/ApertureMeasurements/SFR/040kpc') * c.sec_per_year) / sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/040kpc')[:, 4])
        min1 = -11.
        max1 = np.inf
        pmin1 = -15.
        pmax1 = -5.
        xlabel = r'$\log_{10} \, \mathrm{sSFR} \; [\mathrm{yr}^{-1}]$'
        
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] 
        target = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums==0)]
         
    elif num == 5:
        dims = 1
        nd1  = 1
        name_intent = 'SFR (20 kpc) < 10^0.5 Msun yr^-1'
        name_out = selecthalos_subfindfiles(sf_sub, [('SFR_logMsunPerYr', None, 0.5)], nameonly=True, aperture=20)
        out = selecthalos_subfindfiles(sf_sub, [('SFR_logMsunPerYr', None, 0.5)], nameonly=False, aperture=20)
        #return out
        
        prop1 = np.log10(sf_sub.readarray('Subhalo/ApertureMeasurements/SFR/020kpc') * c.sec_per_year / c.solar_mass)
        min1 = -np.inf
        max1 = 0.5
        pmin1 = -5.
        pmax1 = 5.
        xlabel = r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot} \, \mathrm{yr}^{-1}]$'
        
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1 
        target = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums==0)]
        
    elif num == 6:
        dims = 1
        nd1  = 2
        name_intent = '10^6 Msun <= MBHap (30 kpc) < 10^7 Msun'
        name_out = selecthalos_subfindfiles(sf_sub, [('Mbhap_logMsun', 6., 6.8), ('Mbhap_Msun', 3.0e6, 1.0e7)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('Mbhap_logMsun', 6., 6.8), ('Mbhap_Msun', 3.0e6, 1.0e7)], nameonly=False, aperture=30, testplots=True)
        #return out
        
        prop1 = np.log10(sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/030kpc')[:, 5]/ c.solar_mass)
        min1 = [6., np.log10(3e6)]
        max1 = [6.8, 7.]
        pmin1 = 4.5
        pmax1 = 11.
        xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{BH, aperture}} \; [\mathrm{M}_{\odot}]$'
        
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1 
        target = groupnums[np.logical_and(np.logical_and(min1[0] <= prop1, max1[1] > prop1), subgroupnums==0)]
   
    elif num == 7:
        dims = 1
        nd1 = 2
        name_intent = '-1 cMpc <= Xcen < 5 cMpc or 1.2Rvir X intersects the (12, 17 range)'
        name_out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', -1., 5.), ('X_cMpc', 12., 17., 1.2)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', -1., 5.), ('X_cMpc', 12., 17., 1.2)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 0] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r200c = sf_sub.readarray('FOF/Group_R_Crit200') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect = np.logical_and(prop1 + 1.2*r200c >= 12., prop1 - 1.2*r200c < 17.)
        prop1[shouldselect] = 14.5
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = [24., 12.]
        max1 = [5.,  17.]
        pmin1 = 0.
        pmax1 = 25.
        xlabel = r'X [cMpc] (with shifts for Rvir margins)'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1 
        target = np.where(np.logical_or(np.logical_or(min1[0] <= prop1, max1[0] > prop1), shouldselect))[0] + 1
        
    elif num == 8:
        dims = 1
        nd1 = 2
        name_intent = '-1 cMpc <= Xcen < 5 cMpc or 0.7 Rvir X fully in the (12, 17 range)'
        name_out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', -1., 5.), ('X_cMpc', 12., 17., -0.7)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', -1., 5.), ('X_cMpc', 12., 17., -0.7)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 0] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r200c = sf_sub.readarray('FOF/Group_R_Crit200') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect = np.logical_and(prop1 - 0.7 * r200c >= 12., prop1 + 0.7 * r200c < 17.)
        prop1[shouldselect] = 14.5
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = [24., 12.]
        max1 = [5.,  17.]
        pmin1 = 0.
        pmax1 = 25.
        xlabel = r'X [cMpc] (with shifts for Rvir margins)'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1   
        target = np.where(np.logical_or(np.logical_or(min1[0] <= prop1, max1[0] > prop1), shouldselect))[0] + 1
        
    elif num == 9:
        dims = 1
        nd1 = 1
        name_intent = '-9.7 cMpc <= Ycen < 15. cMpc interesects 5Rvir '
        name_out = selecthalos_subfindfiles(sf_sub, [('Y_cMpc', -9.7, 15., 5.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('Y_cMpc', -9.7, 15., 5.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 1] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r200c = sf_sub.readarray('FOF/Group_R_Crit200') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect = np.logical_or(prop1 - 5. * r200c < 15., prop1 + 5. * r200c >= 15.3)
        prop1[shouldselect] = 1.5
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 15.3
        max1 = 15.0
        pmin1 = 0.
        pmax1 = 25.
        xlabel = r'Y [cMpc] (with shifts for Rvir margins)'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1  
        target = np.where(shouldselect)[0] + 1
        
    elif num == 10:
        dims = 1
        nd1 = 1
        name_intent = '-1.3 cMpc <= Zcen < -1.1 cMpc fully encloses 5 Rvir '
        name_out = selecthalos_subfindfiles(sf_sub, [('Z_cMpc', -1.3, -1.1, -5.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('Z_cMpc', -1.3, -1.1, -5.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 2] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r200c = sf_sub.readarray('FOF/Group_R_Crit200') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect = np.logical_and(prop1 - 5. * r200c >= 23.7, prop1 + 5. * r200c < 23.9)
        prop1[shouldselect] = 25.2
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 23.7
        max1 = 23.9
        pmin1 = 0.
        pmax1 = 25.5
        xlabel = r'Z [cMpc] (with shifts for Rvir margins)'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1  
        target = np.where(shouldselect)[0] + 1
        
    elif num == 11:
        dims = 1
        nd1 = 1
        name_intent = '-200. <= VY [kmps] < 500.'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 25. * cmpc_to_kmps - 200.
        max1 = 500.
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = groupnums[np.logical_and(np.logical_or(min1 <= prop1, max1 > prop1), subgroupnums==0)]
    
    elif num == 12:
        dims = 1
        nd1 = 2
        name_intent = '-200. <= VY [kmps] < 500. + 30 kmps margin, 600 <= VY [kmps] < 700 - 40 kmps margin'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., 30.), ('VY_kmps', 600., 700., -40.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., 30.), ('VY_kmps', 600., 700., -40.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = [25. * cmpc_to_kmps - 230., 640]
        max1 = [530., 660.]
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = groupnums[np.logical_and(np.logical_or(np.logical_or(min1[0] <= prop1, max1[0] > prop1), np.logical_and(min1[1] <= prop1, max1[1] > prop1)), subgroupnums==0)]
     
    elif num == 13:
        dims = 1
        nd1 = 1
        name_intent = '-200. <= VY [kmps] < 500. - 30 kmps margin'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., -30.)], nameonly=True, aperture=40)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., -30.)], nameonly=False, aperture=40)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 25. * cmpc_to_kmps - 170.
        max1 = 470.
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = groupnums[np.logical_and(np.logical_or(min1 <= prop1, max1 > prop1), subgroupnums==0)]
    
    elif num == 14:
        dims = 1
        nd1 = 1
        name_intent = '-200. <= VY [kmps] < 500. + 3000 kmps margin'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., 3000.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., 3000.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 0. 
        max1 = 25. * cmpc_to_kmps
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = groupnums[subgroupnums==0]
    
    elif num == 15:
        dims = 1
        nd1 = 1
        name_intent = '100. <= VY [kmps] < 500. + 3000 kmps margin'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 500., 3000.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 500., 3000.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 0. 
        max1 = 25. * cmpc_to_kmps
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = groupnums[subgroupnums==0]
    
    elif num == 16:
        dims = 1
        nd1 = 1
        name_intent = '-200. <= VY [kmps] < 500. - 3000 kmps margin'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., -3000.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', -200., 500., -3000.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 0. 
        max1 = 0. * cmpc_to_kmps
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = np.array([])
    
    elif num == 17:
        dims = 1
        nd1 = 1
        name_intent = '100. <= VY [kmps] < 500. - 3000 kmps margin'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 500., -3000.)], nameonly=True, aperture=30)
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 500., -3000.)], nameonly=False, aperture=30)
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        
        min1 = 0. 
        max1 = 0. * cmpc_to_kmps
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        xlabel = r'VY [kmps]'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        target = np.array([])
    
    elif num == 18:
        dims = 1
        nd1 = 1
        nd2 = 1
        name_intent = '0.1 <= X [cMpc] < 10. intersects R200c'
        name_out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', 0.1, 10., 6.)], nameonly=True, aperture=30, mdef='200c')
        out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', 0.1, 10., 6.)], nameonly=False, aperture=30, mdef='200c')
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 0] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r500m = sf_sub.readarray('FOF/Group_R_Crit200') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect =  (prop1 + 6. * r500m) >= sf_sub.boxsize / sf_sub.h + 0.1
        shouldselect |= np.logical_and((prop1 + 6. * r500m) >= 0.1  , (prop1 - 6. * r500m) < 10.)
        prop1[shouldselect] = 5.
                
        min1 = 0.1
        max1 = 10.
        pmin1 = 0.
        pmax1 = 25.
        
        #pmin2 = 7.5
        #pmax2 = 14.5
        
        xlabel = r'X [cMpc]'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1
        
        target = np.where(shouldselect)[0] + 1
        
    elif num == 19:
        dims = 1
        nd1 = 1
        nd2 = 1
        name_intent = '0.1 <= X [cMpc] < 10. intersects R500m'
        name_out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', 0.1, 10., 6.)], nameonly=True, aperture=30, mdef='500m')
        out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', 0.1, 10., 6.)], nameonly=False, aperture=30, mdef='500m')
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 0] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r500m = sf_sub.readarray('FOF/Group_R_Mean500') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect =  (prop1 + 6. * r500m) >= sf_sub.boxsize / sf_sub.h + 0.1
        shouldselect |= np.logical_and((prop1 + 6. * r500m) >= 0.1  , (prop1 - 6. * r500m) < 10.)
        prop1[shouldselect] = 5.
                
        min1 = 0.1
        max1 = 10.
        pmin1 = 0.
        pmax1 = 25.
        
        #pmin2 = 7.5
        #pmax2 = 14.5
        
        xlabel = r'X [cMpc]'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1
        
        target = np.where(shouldselect)[0] + 1
        #print(target1)
        #print(len(target1))
        #print(target2)
        #print(len(target2))
        
    ######
    # test selection on > 1 property 
    #####
    
    elif num == 20:
        dims = 2
        nd1 = 1
        nd2 = 1
        name_intent = '100. <= VY [kmps] < 500. kmps, 9.5 <= logM500m [Msun] < 11.5'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 1000., 0.), ('Mhalo_logMsun', 9.5, 11.5)], nameonly=True, aperture=30, mdef='500m')
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 1000., 0.), ('Mhalo_logMsun', 9.5, 11.5)], nameonly=False, aperture=30, mdef='500m')
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        
        prop2 = np.log10(sf_sub.readarray('FOF/Group_M_Mean500') / c.solar_mass)
        
        min1 = 100. 
        max1 = 1000.
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        
        min2 = 9.5
        max2 = 11.5
        #pmin2 = 7.5
        #pmax2 = 14.5
        
        xlabel = r'VY [kmps]'
        ylabel = r'$\log_{10} \, \mathrm{M}_{500m} \; [\mathrm{M}_{\odot}]$'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        prop2_all = prop2[hassubs]
        prop2_sel = prop2[out - 1]
        
        target1 = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums==0)]
        target2 = np.where(np.logical_and(min2 <= prop2, max2 > prop2))[0] + 1
        #print(target1)
        #print(len(target1))
        #print(target2)
        #print(len(target2))
        
        target = set(target1) & set(target2)
        target = np.array(list(target))
        target.sort()
    
    elif num == 21:
        dims = 2
        nd1 = 1
        nd2 = 1
        name_intent = '0.1 <= X [cMpc] < 10. intersects R500m, 9.5 <= logM500m [Msun] < 11.5'
        name_out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', 0.1, 10., 6.), ('Mhalo_logMsun', 9.5, 11.5)], nameonly=True, aperture=30, mdef='500m')
        out = selecthalos_subfindfiles(sf_sub, [('X_cMpc', 0.1, 10., 6.), ('Mhalo_logMsun', 9.5, 11.5)], nameonly=False, aperture=30, mdef='500m')
        
        prop1 = sf_sub.readarray('FOF/GroupCentreOfPotential')[:, 0] / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r500m = sf_sub.readarray('FOF/Group_R_Mean500') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        shouldselect =  (prop1 + 6. * r500m) >= sf_sub.boxsize / sf_sub.h + 0.1
        shouldselect |= np.logical_and((prop1 + 6. * r500m) >= 0.1  , (prop1 - 6. * r500m) < 10.)
        prop1[shouldselect] = 5.
        
        prop2 = np.log10(sf_sub.readarray('FOF/Group_M_Mean500') / c.solar_mass)
        
        min1 = 0.1
        max1 = 10.
        pmin1 = 0.
        pmax1 = 25.
        
        min2 = 9.5
        max2 = 11.5
        #pmin2 = 7.5
        #pmax2 = 14.5
        
        xlabel = r'X [cMpc]'
        ylabel = r'$\log_{10} \, \mathrm{M}_{500m} \; [\mathrm{M}_{\odot}]$'
        prop1_all = prop1
        prop1_sel = prop1[out - 1] # out -1
        prop2_all = prop2
        prop2_sel = prop2[out - 1]
        
        target1 = np.where(shouldselect)[0] + 1
        target2 = np.where(np.logical_and(min2 <= prop2, max2 > prop2))[0] + 1
        #print(target1)
        #print(len(target1))
        #print(target2)
        #print(len(target2))
        
        target = set(target1) & set(target2)
        target = np.array(list(target))
        target.sort()
    
    elif num == 22:
        dims = 2
        nd1 = 1
        nd2 = 1
        name_intent = '100. <= VY [kmps] < 500. kmps, 9.5 <= logMstar(40 kpc) [Msun] < 11.5'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 1000., 0.), ('Mstar_logMsun', 9.5, 11.5)], nameonly=True, aperture=40, mdef='500m')
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 1000., 0.), ('Mstar_logMsun', 9.5, 11.5)], nameonly=False, aperture=40, mdef='500m')
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        
        prop2 = np.log10(sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/040kpc')[:, 4] / c.solar_mass)
        
        min1 = 100. 
        max1 = 1000.
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        
        min2 = 9.5
        max2 = 11.5
        #pmin2 = 7.5
        #pmax2 = 14.5
        
        xlabel = r'VY [kmps]'
        ylabel = r'$\log_{10} \, \mathrm{M}_{*}(40\,\mathrm{pkpc}) \; [\mathrm{M}_{\odot}]$'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        prop2_all = prop2[subinds_sel]
        prop2_sel = prop2[subinds[out - 1]]
        
        target1 = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums == 0)]
        target2 = groupnums[np.logical_and(np.logical_and(min2 <= prop2, max2 > prop2), subgroupnums == 0)]
        #print(target1)
        #print(len(target1))
        #print(target2)
        #print(len(target2))
        
        target = set(target1) & set(target2)
        target = np.array(list(target))
        target.sort()
    
    # just put a bunch of stuff together here...
    elif num == 23:
        dims = 2
        nd1 = 1
        nd2 = 2
        name_intent = '100. <= VY [kmps] < 1000. kmps, 9.5 <= logM200c [Msun] < 10. or 10.5 < logM200c [Msun], Mstar (5 kpc) >= 3e6 Msun, 5. <= X < 15. or 20. <= X < 22., -5. <= Z < 5. (1 Rvir margins)'
        name_out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 1000., 0.), ('Mhalo_logMsun', 9.5, 10.), ('Mhalo_logMsun', 10.5, None),\
                                                     ('Mstar_Msun', 3e6, None), ('X_cMpc', 5., 15., 1.),\
                                                     ('X_cMpc', 20., 22., 1.), ('Z_cMpc', -5., 5., 1.)],\
                                            nameonly=True, aperture=5, mdef='200c')
        out = selecthalos_subfindfiles(sf_sub, [('VY_kmps', 100., 1000., 0.), ('Mhalo_logMsun', 9.5, 10.), ('Mhalo_logMsun', 10.5, None),\
                                                     ('Mstar_Msun', 3e6, None), ('X_cMpc', 5., 15., 1.),\
                                                     ('X_cMpc', 20., 22., 1.), ('Z_cMpc', -5., 5., 1.)],\
                                            nameonly=False, aperture=5, mdef='200c')
        
        prop1 = sf_sub.readarray('Subhalo/CentreOfMass')[:, 1] / c.cm_per_mpc / sf_sub.a * cmpc_to_kmps
        #print('min/max: %s, %s'%(np.min(prop1), np.max(prop1)))
        vel = sf_sub.readarray('Subhalo/Velocity')[:, 1] / 1e5
        prop1 += vel
        box = cmpc_to_kmps * sf_sub.boxsize / sf_sub.h 
        prop1 %= box
        
        min1 = 100. 
        max1 = 1000.
        pmin1 = 0.
        pmax1 = 25. * cmpc_to_kmps
        deltap1 = 10.
        
        prop2 = np.log10(sf_sub.readarray('FOF/Group_M_Crit200') / c.solar_mass)
        
        min2 = [9.5, 10.5]
        max2 = [10., np.inf]
        #pmin2 = 7.5
        #pmax2 = 14.5
        
        prop3 = sf_sub.readarray('Subhalo/ApertureMeasurements/Mass/005kpc')[:, 4] / c.solar_mass
        
        min3 = 3e6
        max3 = np.inf
        
        xlabel = r'VY [kmps]'
        ylabel = r'$\log_{10} \, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}]$'
        prop1_all = prop1[subinds_sel]
        prop1_sel = prop1[subinds[out - 1]] # out -1
        prop2_all = prop2[hassubs]
        prop2_sel = prop2[out - 1]
        
        prop4 = sf_sub.readarray('FOF/GroupCentreOfPotential') / c.cm_per_mpc / sf_sub.a
        #print('Min/max X: %s, %s'%(np.min(prop1), np.max(prop1)))
        r200c = sf_sub.readarray('FOF/Group_R_Crit200') / c.cm_per_mpc / sf_sub.a
        # mangle the selection coordinates to account for the margins properly
        
        shouldselect4 = np.logical_or(np.logical_and(5. - r200c <= prop4[:, 0], 15. + r200c > prop4[:, 0]), np.logical_and(20. - r200c <= prop4[:, 0], 22. + r200c > prop4[:, 0])) 
        shouldselect4 = np.logical_and(shouldselect4, np.logical_or(20. - r200c <= prop4[:, 2], 5. + r200c > prop4[:, 2]))
        
        target1 = groupnums[np.logical_and(np.logical_and(min1 <= prop1, max1 > prop1), subgroupnums==0)]
        target2 = np.where(np.logical_or(np.logical_and(min2[0] <= prop2, max2[0] > prop2), np.logical_and(min2[1] <= prop2, max2[1] > prop2)))[0] + 1
        target3 = groupnums[np.logical_and(np.logical_and(min3 <= prop3, max3 > prop3), subgroupnums==0)]
        target4 = np.where(shouldselect4)[0] + 1
        #print(target1)
        #print(len(target1))
        #print(target2)
        #print(len(target2))
        
        target = (set(target1) & set(target2)) & (set(target3) & set(target4))
        target = np.array(list(target))
        target.sort()
        
        # to test prop3 selection
        # prop2_all = np.log10(prop3[subinds_sel])
        # prop2_sel = np.log10(prop3[subinds[out - 1]])
        # min2 = np.log10(min3)
        # max2 = np.log10(max3)
        # nd2 = 1
        # ylabel = r'$\log_{10} \, \mathrm{M}_{*}(5 \, \mathrm{kpc}) \; [\mathrm{M}_{\odot}]$'
        
        # to test X / Z seleciton
        prop1_all = prop4[:, 0][hassubs]
        prop1_sel = prop4[:, 0][out - 1]
        
        min1 = [5., 20.]
        max1 = [15., 22.]
        nd1 = 2
        
        prop2_all = prop4[:, 2][hassubs]
        prop2_sel = prop4[:, 2][out - 1]
        
        min2 = 20.
        max2 = 5.
        nd2 = 1
        
        xlabel = r'X [cMpc]'
        ylabel = r'Z [cMpc]'
        
    else:
        raise ValueError('There is no test %s'%num)
        
    fontsize = 12
    if dims == 1:
        plt.figure()
        ax = plt.subplot(111)
        ax.hist(prop1_all, color='blue', alpha=0.5, log=True, label='all (centrals)', bins=np.arange(pmin1, pmax1 + deltap1, deltap1))
        ax.hist(prop1_sel, color='orange', alpha=0.5, log=True, label='selected (centrals)', bins=np.arange(pmin1, pmax1 + deltap1, deltap1))
        ax.legend(fontsize=fontsize)
        if nd1 == 1:
            color = 'C0'
            ax.axvline(min1, linestyle='dashed', color=color)
            ax.axvline(max1, linestyle='dashed', color=color)
        else:
            for i in range(nd1):
                color = 'C%i'%i
                ax.axvline(min1[i], linestyle='dashed', color=color)
                ax.axvline(max1[i], linestyle='dashed', color=color)
                 
        ax.set_title('Intended selection: %s\n'%(name_intent) + 'Recovered selection: %s'%(name_out), fontsize=fontsize)
        ax.set_ylabel('(sub)halos', fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
    
    if dims == 2:
        plt.figure()
        ax = plt.subplot(111)
        ax.scatter(prop1_all, prop2_all, color='blue', alpha=0.5, label='all (centrals)')
        ax.scatter(prop1_sel, prop2_sel, color='orange', alpha=0.5, label='all (centrals)')
        ax.legend(fontsize=fontsize)
        if nd1 == 1:
            color = 'C0'
            ax.axvline(min1, linestyle='dashed', color=color)
            ax.axvline(max1, linestyle='dashed', color=color)
        else:
            for i in range(nd1):
                color = 'C%i'%i
                ax.axvline(min1[i], linestyle='dashed', color=color)
                ax.axvline(max1[i], linestyle='dashed', color=color)
        if nd2 == 1:
            color = 'C0'
            ax.axhline(min2, linestyle='dashed', color=color)
            ax.axhline(max2, linestyle='dashed', color=color)
        else:
            for i in range(nd2):
                color = 'C%i'%i
                ax.axhline(min2[i], linestyle='dashed', color=color)
                ax.axhline(max2[i], linestyle='dashed', color=color)
                
        ax.set_title('Intended selection: %s\n'%(name_intent) + 'Recovered selection: %s'%(name_out), fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        
    plt.savefig(testdir + 'test_%i.pdf'%num, format='pdf', bbox_inches='tight')
    
    target.sort()
    if len(target) == len(out):
        success = np.all(target == out)
    else:
        success = False
        
    return success

def testhaloselection_subfind_all(showplots=False):
    num = 0
    allsucces = True
    while True:
        try:
            success = testhaloselection_subfind(num=num)
            print('Halo selection test %i: %s'%(num, success))
            allsucces &= success
            if showplots:
                plt.show()
            else:
                plt.close() # otherwise, might get errors from too many plots
        except ValueError:
            break
        num += 1
    
    return allsucces
