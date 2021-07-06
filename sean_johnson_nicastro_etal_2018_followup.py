#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:15:32 2019

@author: wijers
"""

import numpy as np
import h5py 
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines

import tol_colors as tc

import makecddfs as mc
import eagle_constants_and_units as c
import cosmo_utils as cu
#import halocatalogue as hc
#import selecthalos as sh

pdir = '/net/luttero/data2/proc/'
mdir = '/net/luttero/data2/imgs/sean_johnson_nicastro_etal_2018_followup/'


# z=0.43 absorber, snapshots 23 (z=0.37) and 24 (z=0.50) are closest 
# halo catalogues generated on cosma
#def getstats_groupcoincidence(simnum='L0100N1504', var='REFERENCE',\
#                              snaps=[23, 24], apsize=30, groupmass=5e13,\
#                              deltagroupmassdex=0.3, los=['x', 'y', 'z'],\
#                              distcut_pMpc=10., impactpar_pkpc = 200,\
#                              galmaxsSFR=10**-11.0, galminMstar=10**11.0):
#    '''
#    inputs:
#    -------------------------------------------------------------------------
#    simnum:    simulation box name (str)
#    var:       simulation subgrid variation name (str)
#    snaps:     simulation snapshots (list of ints)
#    apsize:    aperture size for M* (int, size in pkpc)
#    groupmass: mass of central group (float, Msun)
#    deltagroupmassdex: range of central group masses to consider (float, log10
#                       range)
#    los:       lines of sight to use --
#                 int n -> n random directions (sightline length = box size)
#                 list  -> specific directions. str 'x', 'y', 'z': along that 
#                          axis length-3 iterable: direction vector (will be 
#                          normalized to unity)
#    distcut_pMpc: distance cut to impose on the galaxy (float, pMpc)
#    galmaxsSFR:   (float, yr-1) 
#    galminMstar:  (float, Msun)             
#    '''
#    
#    # generate halo catalogues if they do not exist already
#    catnames = [hc.generatehdf5_censat(simnum, snap, Mhmin=0., var=var, apsize=apsize, nameonly=True) for snap in snaps]
#    for i in range(len(catnames)):
#        if not os.path.isfile(pdir + catnames[i]):
#            hc.generatehdf5_censat(simnum, snaps[i], Mhmin=0., var=var, apsize=apsize, nameonly=False)
#    
#    # open halo catalogues and get the entries needed for each
#    datadct = {}
#    for i in range(len(catnames)):
#        snap = snaps[i]
#        catname = catnames[i]
#        with h5py.File(pdir + catname, 'r') as fi:
#            cosmopars = {key: item for (key, item) in fi['Header/cosmopars'].attrs.items()}
#            mhalo = np.array(fi['M200c_Msun'])
#            mstar = np.array(fi['Mstar_Msun'])
#            ssfr  = np.array(fi['SFR_MsunPerYr']) / mstar
#            iscen = np.array(fi['SubGroupNumber']) == 0
#            gcen3 = np.array([fi['Xgroupcop_cMpc'], fi['Ygroupcop_cMpc'], fi['Zgroupcop_cMpc']]).T * cosmopars['a']
#            scen3 = np.array([fi['Xcop_cMpc'], fi['Ycop_cMpc'], fi['Zcop_cMpc']]).T * cosmopars['a']
#            vpec3 = np.array([fi['VXpec_kmps'], fi['VYpec_kmps'], fi['VZpec_kmps']]).T
#            
#            if np.all(gcen3[iscen] == scen3[iscen]):
#                print('Group centers are just subhalo 0 centers')
#            else:
#                print('Group centers match subhalo 0 centers in %i / %i cases'%(np.sum(gcen3[iscen] == scen3[iscen]) , np.sum(iscen)))
#                
#            szobs = cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * c.cm_per_mpc * gcen3 + vpec3 * 1e5 # cgs velocities
#            szobs *= 1. / c.c # local redshift
#            szobs += 1.
#            szobs *= (cosmopars['z'] + 1.)
#            szobs -= 1.
#            
#            deltazbox = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * c.cm_per_mpc * cu.Hubble(cosmopars['z'], cosmopars=cosmopars)
#            deltazbox = (1. + deltazbox/c.c) * (cosmopars['z'] + 1.) - 1.
#            
#            boxsize = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a']
#            
#            mhalomin = groupmass * 10**(1. - deltagroupmassdex)
#            mhalomax = groupmass * 10**(1. + deltagroupmassdex)
#            
#            groupsel = np.all(np.array([iscen, mhalomin <= mhalo, mhalomax > mhalo]), axis=0)
#            galsel = np.all(np.array([mstar >= galminMstar, ssfr < galmaxsSFR]), axis=0)
#            
#            grouppos3 = gcen3[groupsel]
#            groupz3   = szobs[groupsel]
#            galpos3   = scen3[galsel]
#            galz3     = scen3[galsel]
#            
#            datadct[snap] = {'boxsize':   boxsize,\
#                             'deltazbox': deltazbox,\
#                             'grouppos3': grouppos3,\
#                             'groupz3':   groupz3,\
#                             'galpos3':   galpos3,\
#                             'galz3':     galz3}
#    
#    # parse the los selection
#    losdct = {}
#    if isinstance(los, int):
#        for i in range(len(snaps)):
#            phi = np.random.uniform(low=0, high=np.pi*2, size=los)
#            costheta = np.random.uniform(low=-1, high=1, size=los)
#            theta = np.arccos(costheta)
#            x = np.sin(theta) * np.cos(phi)
#            y = np.sin(theta) * np.sin(phi)
#            z = np.cos(theta )
#            losdct[snaps[i]] = list(np.array([x, y, z]).T)
#    elif np.any([isinstance(line, str) for line in los]):
#        for i in range(len(los)):
#            if los[i] == 'x':
#                los[i] = np.array([1., 0., 0.])
#            elif los[i] == 'y':
#                los[i] = np.array([0., 1., 0.])
#            elif los[i] == 'z':
#                los[i] = np.array([0., 0., 1.])
#            else:
#                los[i] = np.array(los[i])
#                los[i] = los[i] / np.sqrt(np.sum(los[i]**2))
#    
#    # get the sample statistics for each snapshot and los direction
#    for si in range(len(snaps)):
#        boxsize   = datadct[snap]['boxsize']
#        deltazbox = datadct[snap]['deltazbox']
#        grouppos3 = datadct[snap]['grouppos3']
#        groupz3   = datadct[snap]['groupz3']
#        galpos3   = datadct[snap]['galpos3']
#        galz3     = datadct[snap]['galz3']
#        
#        galdists3 = galpos3 - grouppos3
#        galdists3 
#        
#        for li in range(len(losdct[i])):
#            vec = losdct[snaps[si]][li]
#            grouppos_los = np.sum(grouppos3 * vec[np.newaxis, :], axis=1)
#            galpos_los   = np.sum(galpos3 * vec[np.newaxis, :], axis=1)

def checksubdct_equal(dct):
    keys = dct.keys()
    if not np.all(np.array([set(dct[key].keys()) == set(dct[keys[0]].keys()) for key in keys])):
        print('Keys do not match')
        return False
    if not np.all(np.array([np.all(np.array([np.all(dct[keys[0]][sub] == dct[key][sub]) for sub in dct[keys[0]].keys() ])) for key in keys])):
        print('Entries do not match')
        return False
    return True

def setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, 
             labelright=False, right=True, labeltop=False):
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=right, top=top, axis='both',
                   which='both', color=color,
                   labelleft=labelleft, labeltop=labeltop, labelbottom = labelbottom, 
                   labelright=labelright)

def interp_fill_between(binsedges1, binsedges2):
    '''
    Takes in two binsedges (y,x) datasets, returns combined x values and interpolated y values for those points 
    assumes x values are sorted low-to-high
    '''
    x1 = binsedges1[1]
    x2 = binsedges2[1]
    y1 = binsedges1[0]
    y2 = binsedges2[0]
    allx = np.sort(np.array(list(x1) + list(x2[np.all(x1[:,np.newaxis] != x2[np.newaxis, :], axis = 0)]))) # all unique x values
    allx = allx[allx >= max(x1[0],x2[0])] # interpolate, don't extrapolate. For fill between, the full x range must match
    allx = allx[allx <= min(x1[-1],x2[-1])]
    y1all = np.interp(allx, x1, y1) # linear interpolation
    y2all = np.interp(allx, x2, y2) # linear interpolation   
    return allx, y1all, y2all

def linterpsolve(xvals, yvals, xpoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below xpoint 
    xvals, yvals: 1D arrays
    xpoint: float
    '''
    if np.all(np.diff(xvals) > 0.):
        incr = True
    elif np.all(np.diff(xvals) < 0.):
        incr = False
    else:
        print('linterpsolve only works for monotonic functions')
        return None
    ind1 = np.where(xvals <= xpoint)[0]
    ind2 = np.where(xvals >= xpoint)[0]
    #print(ind1)
    #print(ind2)
    if len(ind2) == 0 or len(ind1) == 0:
        print('xpoint is outside the bounds of xvals')
        return None
    if incr:
        ind1 = np.max(ind1)
        ind2 = np.min(ind2)
    else:
        ind1 = np.min(ind1)
        ind2 = np.max(ind2)
    #print('Indices x: %i, %i'%(ind1, ind2))
    #print('x values: lower %s, upper %s, searched %s'%(xvals[ind1], xvals[ind2], xpoint))
    if ind1 == ind2:
        ypoint = yvals[ind1]
    else:
        w = (xpoint - xvals[ind1]) / (xvals[ind2] - xvals[ind1]) #weight
        ypoint = yvals[ind2] * w + yvals[ind1] * (1. - w)
    #print('y values: lower %s, upper %s, solution: %s'%(yvals[ind1], yvals[ind2], ypoint))
    return ypoint

def find_intercepts(yvals, xvals, ypoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below ypoint 
    xvals, yvals: 1D arrays
    ypoint: float
    Does not distinguish between intersections separated by less than 2 xvals points
    '''
    if not (np.all(np.diff(xvals) < 0.) or np.all(np.diff(xvals) > 0.)):
        print('linterpsolve only works for monotonic x values')
        return None
    zerodiffs = yvals - ypoint
    leqzero = np.where(zerodiffs <= 0.)[0]
    if len(leqzero) == 0:
        return np.array([])
    elif len(leqzero) == 1:
        edges = [[leqzero[0], leqzero[0]]]
    else:
        segmentedges = np.where(np.diff(leqzero) > 1)[0] + 1
        if len(segmentedges) == 0: # one dip below zero -> edges are intercepts
            edges = [[leqzero[0], leqzero[-1]]]
        else:
            parts = [leqzero[: segmentedges[0]] if si == 0 else \
                     leqzero[segmentedges[si - 1] : segmentedges[si]] if si < len(segmentedges) else\
                     leqzero[segmentedges[si - 1] :] \
                     for si in range(len(segmentedges) + 1)]
            edges = [[part[0], part[-1]] for part in parts]
    intercepts = [[linterpsolve(zerodiffs[ed[0]-1: ed[0] + 1], xvals[ed[0]-1: ed[0] + 1], 0.),\
                   linterpsolve(zerodiffs[ed[1]: ed[1] + 2],   xvals[ed[1]: ed[1] + 2], 0.)]  \
                  if ed[0] != 0 and ed[1] != len(yvals) - 1 else \
                  [None,\
                   linterpsolve(zerodiffs[ed[1]: ed[1] + 2],   xvals[ed[1]: ed[1] + 2], 0.)] \
                  if ed[1] != len(yvals) - 1 else \
                  [linterpsolve(zerodiffs[ed[0]-1: ed[0] + 1], xvals[ed[0]-1: ed[0] + 1], 0.),\
                   None]  \
                  if ed[0] != 0 else \
                  [None, None]
                 for ed in edges]
    intercepts = [i for i2 in intercepts for i in i2]
    if intercepts[0] is None:
        intercepts = intercepts[1:]
    if intercepts[-1] is None:
        intercepts = intercepts[:-1]
    return np.array(intercepts)

def plot_o7cddfsplits_groupenv(numsl=3, minmass=5e13):
    '''
    numsl: 3 or 4
    minmass: 5e13 or 2.5e13; for numsl == 4, only 5e13 is available
    '''
    if numsl == 4:
        if minmass != 5e13:
            raise ValueError('For 4 slices, only minmass=5e13 (default) is available')
        imgname = 'cddf_coldens_o7_L0100N1504_23_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_M200c-geq-5e13-Msun.pdf'
        
        h5files = {'snap23': pdir + 'cddf_coldens_o7_L0100N1504_23_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_M200c-geq-5e13-Msun.hdf5',\
                   'snap24': pdir + 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_M200c-geq-5e13-Msun.hdf5'}
        filekeys = h5files.keys()
        
        masknames = ['nomask',\
                 '100pkpc_pos_M200c-geq-5e13-Msun',\
                 '200pkpc_pos_M200c-geq-5e13-Msun',\
                 '500pkpc_pos_M200c-geq-5e13-Msun',\
                 '1000pkpc_pos_M200c-geq-5e13-Msun',\
                 '10000pkpc_pos_M200c-geq-5e13-Msun',\
                 '100pkpc_vel_M200c-geq-5e13-Msun',\
                 '200pkpc_vel_M200c-geq-5e13-Msun',\
                 '500pkpc_vel_M200c-geq-5e13-Msun',\
                 '1000pkpc_vel_M200c-geq-5e13-Msun',\
                 '10000pkpc_vel_M200c-geq-5e13-Msun',\
                 ]
        legendnames = {'nomask': 'total',\
                 '100pkpc_pos_M200c-geq-5e13-Msun': r'$r_{\perp} < 100 \, \mathrm{pkpc}$',\
                 '200pkpc_pos_M200c-geq-5e13-Msun': r'$r_{\perp} < 200 \, \mathrm{pkpc}$',\
                 '500pkpc_pos_M200c-geq-5e13-Msun': r'$r_{\perp} < 500 \, \mathrm{pkpc}$',\
                 '1000pkpc_pos_M200c-geq-5e13-Msun': r'$r_{\perp} < 1000 \, \mathrm{pkpc}$',\
                 '10000pkpc_pos_M200c-geq-5e13-Msun': r'$r_{\perp} < 10000 \, \mathrm{pkpc}$',\
                 '100pkpc_vel_M200c-geq-5e13-Msun': None,\
                 '200pkpc_vel_M200c-geq-5e13-Msun': None,\
                 '500pkpc_vel_M200c-geq-5e13-Msun': None,\
                 '1000pkpc_vel_M200c-geq-5e13-Msun': None,\
                 '10000pkpc_vel_M200c-geq-5e13-Msun': None,\
                 }
        colors = {'nomask': 'black',\
                 '100pkpc_pos_M200c-geq-5e13-Msun': 'C4',\
                 '200pkpc_pos_M200c-geq-5e13-Msun': 'C0',\
                 '500pkpc_pos_M200c-geq-5e13-Msun': 'C2',\
                 '1000pkpc_pos_M200c-geq-5e13-Msun': 'C1',\
                 '10000pkpc_pos_M200c-geq-5e13-Msun': 'C3',\
                 '100pkpc_vel_M200c-geq-5e13-Msun': 'C4',\
                 '200pkpc_vel_M200c-geq-5e13-Msun': 'C0',\
                 '500pkpc_vel_M200c-geq-5e13-Msun': 'C2',\
                 '1000pkpc_vel_M200c-geq-5e13-Msun': 'C1',\
                 '10000pkpc_vel_M200c-geq-5e13-Msun': 'C3',\
                 }      
        linestyles = {'nomask': 'solid',\
                 '100pkpc_pos_M200c-geq-5e13-Msun': 'solid',\
                 '200pkpc_pos_M200c-geq-5e13-Msun': 'solid',\
                 '500pkpc_pos_M200c-geq-5e13-Msun': 'solid',\
                 '1000pkpc_pos_M200c-geq-5e13-Msun': 'solid',\
                 '10000pkpc_pos_M200c-geq-5e13-Msun': 'solid',\
                 '100pkpc_vel_M200c-geq-5e13-Msun': 'dashed',\
                 '200pkpc_vel_M200c-geq-5e13-Msun': 'dashed',\
                 '500pkpc_vel_M200c-geq-5e13-Msun': 'dashed',\
                 '1000pkpc_vel_M200c-geq-5e13-Msun': 'dashed',\
                 '10000pkpc_vel_M200c-geq-5e13-Msun': 'dashed',\
                 }  
        linewidths = {'nomask': 2.,\
                 '100pkpc_pos_M200c-geq-5e13-Msun': 2.,\
                 '200pkpc_pos_M200c-geq-5e13-Msun': 2.,\
                 '500pkpc_pos_M200c-geq-5e13-Msun': 2.,\
                 '1000pkpc_pos_M200c-geq-5e13-Msun': 2.,\
                 '10000pkpc_pos_M200c-geq-5e13-Msun': 2.,\
                 '100pkpc_vel_M200c-geq-5e13-Msun': 1.5,\
                 '200pkpc_vel_M200c-geq-5e13-Msun': 1.5,\
                 '500pkpc_vel_M200c-geq-5e13-Msun': 1.5,\
                 '1000pkpc_vel_M200c-geq-5e13-Msun': 1.5,\
                 '10000pkpc_vel_M200c-geq-5e13-Msun': 1.5,\
                 }  
        alphas = {'snap23': 1.,\
                  'snap24': 0.5}
    
    elif numsl == 3:
        if minmass == 5e13:
            imgname = 'cddf_coldens_o7_L0100N1504_23-24_test3.31_PtAb_C2Sm_32000pix_33.3slice_zcen-all_z-projection_T4EOS_masks_M200c-geq-5e13-Msun.pdf'
            fillstr = '5.0e13'
        elif minmass == 2.5e13:
            imgname = 'cddf_coldens_o7_L0100N1504_23-24_test3.31_PtAb_C2Sm_32000pix_33.3slice_zcen-all_z-projection_T4EOS_masks_M200c-geq-2.5e13-Msun.pdf'
            fillstr = '2.5e13'
        else:
            raise ValueError('For 3 slices, only minmass=5e13 (default) or 2.5e13 are available')
            
        h5files = {'snap23': pdir + 'cddf_coldens_o7_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_33.3333333333slice_zcen-all_z-projection_T4EOS_masks_M200c-2.5e13-5e13_environment.hdf5',\
                   'snap24': pdir + 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_33.3333333333slice_zcen-all_z-projection_T4EOS_masks_M200c-2.5e13-5e13_environment.hdf5'}
        filekeys = h5files.keys()

            
        masknames = ['nomask',\
                     '100pkpc_pos_M200c-geq-%s-Msun'%fillstr,\
                     '200pkpc_pos_M200c-geq-%s-Msun'%fillstr,\
                     '500pkpc_pos_M200c-geq-%s-Msun'%fillstr,\
                     '1000pkpc_pos_M200c-geq-%s-Msun'%fillstr,\
                     '10000pkpc_pos_M200c-geq-%s-Msun'%fillstr,\
                     '100pkpc_vel_M200c-geq-%s-Msun'%fillstr,\
                     '200pkpc_vel_M200c-geq-%s-Msun'%fillstr,\
                     '500pkpc_vel_M200c-geq-%s-Msun'%fillstr,\
                     '1000pkpc_vel_M200c-geq-%s-Msun'%fillstr,\
                     '10000pkpc_vel_M200c-geq-%s-Msun'%fillstr,\
                     ]
        legendnames = {'nomask': 'total',\
                 '100pkpc_pos_M200c-geq-%s-Msun'%fillstr: r'$r_{\perp} < 100 \, \mathrm{pkpc}$',\
                 '200pkpc_pos_M200c-geq-%s-Msun'%fillstr: r'$r_{\perp} < 200 \, \mathrm{pkpc}$',\
                 '500pkpc_pos_M200c-geq-%s-Msun'%fillstr: r'$r_{\perp} < 500 \, \mathrm{pkpc}$',\
                 '1000pkpc_pos_M200c-geq-%s-Msun'%fillstr: r'$r_{\perp} < 1000 \, \mathrm{pkpc}$',\
                 '10000pkpc_pos_M200c-geq-%s-Msun'%fillstr: r'$r_{\perp} < 10000 \, \mathrm{pkpc}$',\
                 '100pkpc_vel_M200c-geq-%s-Msun'%fillstr: None,\
                 '200pkpc_vel_M200c-geq-%s-Msun'%fillstr: None,\
                 '500pkpc_vel_M200c-geq-%s-Msun'%fillstr: None,\
                 '1000pkpc_vel_M200c-geq-%s-Msun'%fillstr: None,\
                 '10000pkpc_vel_M200c-geq-%s-Msun'%fillstr: None,\
                 }
        colors = {'nomask': 'black',\
                 '100pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'C4',\
                 '200pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'C0',\
                 '500pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'C2',\
                 '1000pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'C1',\
                 '10000pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'C3',\
                 '100pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'C4',\
                 '200pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'C0',\
                 '500pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'C2',\
                 '1000pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'C1',\
                 '10000pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'C3',\
                 }      
        linestyles = {'nomask': 'solid',\
                 '100pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'solid',\
                 '200pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'solid',\
                 '500pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'solid',\
                 '1000pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'solid',\
                 '10000pkpc_pos_M200c-geq-%s-Msun'%fillstr: 'solid',\
                 '100pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'dashed',\
                 '200pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'dashed',\
                 '500pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'dashed',\
                 '1000pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'dashed',\
                 '10000pkpc_vel_M200c-geq-%s-Msun'%fillstr: 'dashed',\
                 }  
        linewidths = {'nomask': 2.,\
                 '100pkpc_pos_M200c-geq-%s-Msun'%fillstr: 2.,\
                 '200pkpc_pos_M200c-geq-%s-Msun'%fillstr: 2.,\
                 '500pkpc_pos_M200c-geq-%s-Msun'%fillstr: 2.,\
                 '1000pkpc_pos_M200c-geq-%s-Msun'%fillstr: 2.,\
                 '10000pkpc_pos_M200c-geq-%s-Msun'%fillstr: 2.,\
                 '100pkpc_vel_M200c-geq-%s-Msun'%fillstr: 1.5,\
                 '200pkpc_vel_M200c-geq-%s-Msun'%fillstr: 1.5,\
                 '500pkpc_vel_M200c-geq-%s-Msun'%fillstr: 1.5,\
                 '1000pkpc_vel_M200c-geq-%s-Msun'%fillstr: 1.5,\
                 '10000pkpc_vel_M200c-geq-%s-Msun'%fillstr: 1.5,\
                 }  
        alphas = {'snap23': 1.,\
                  'snap24': 0.5}
        
    legend_handles = [mlines.Line2D([], [], color=colors[mask], linestyle='solid', label=legendnames[mask], linewidth=2.) for mask in masknames[:6]]
    legend_handles = legend_handles + [mlines.Line2D([], [], color='gray', linestyle='solid', label='los pos.', linewidth=2.),\
                                       mlines.Line2D([], [], color='gray', linestyle='dashed', label='los vel.', linewidth=1.5)]
    
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    
    hists = {}
    cosmopars = {}
    fcovs = {}
    dXtot = {}
    dXtotdlogN = {}
    bins = {}
    for filekey in h5files.keys():
        filename = h5files[filekey]
        with h5py.File(filename, 'r') as fi:
            bins[filekey] = np.array(fi['bins/axis_0'])
            # handle +- infinity edges for plotting; should be outside the plot range anyway
            if bins[filekey][0] == -np.inf:
                bins[filekey][0] = -100.
            if bins[filekey][-1] == np.inf:
                bins[filekey][-1] = 100.
            
            # extract number of pixels from the input filename, using naming system of make_maps
            inname = np.array(fi['input_filenames'])[0]
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            ionind = 1 + np.where(np.array([part == 'coldens' for part in parts]))[0][0]
            ion = parts[ionind]
            
            masks = masknames
    
            hists[filekey] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
            fcovs[filekey] = {mask: fi[mask].attrs['covfrac'] for mask in hists[filekey].keys()}
            
            if numsl == 4:
                examplemaskdir = 'masks/12.5/'
            elif numsl == 3:
                examplemaskdir = 'masks/50.0/'
            examplemask = fi[examplemaskdir].keys()[0]
            cosmopars[filekey] = {key: item for (key, item) in fi['%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
            dXtot[filekey] = mc.getdX(cosmopars[filekey]['z'], cosmopars[filekey]['boxsize'] / cosmopars[filekey]['h'], cosmopars=cosmopars[filekey]) * float(numpix_1sl**2)
            dXtotdlogN[filekey] = dXtot[filekey] * np.diff(bins[filekey])
    
    legend_handles = legend_handles + [mlines.Line2D([], [], color='brown', linestyle='solid', alpha=alphas[key], label=r'$z=%.2f$'%cosmopars[key]['z'], linewidth=2.) for key in filekeys] 
    
    if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
        bins = bins[filekeys[0]]
    else:
        raise RuntimeError("bins for different files don't match")
        
    fig = plt.figure(figsize=(5.5, 5.5))
    grid = gsp.GridSpec(2, 2, hspace=0.0, wspace=0.0, height_ratios=[4., 4.], width_ratios=[6., 2.])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    lax = fig.add_subplot(grid[:, 1])
    
    fontsize = 12
    if ion[0] == 'h':
        ax1.set_xlim(12.5, 23.5)
        ax1.set_ylim(-6., 1.65)
    else:
        ax1.set_xlim(14., 17.5)
        ax1.set_ylim(-6.5, 1.6)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(-5., 0.) 
    
    ax2.set_xlabel(r'$\log_{10}\, \mathrm{N}_{\mathrm{O\,VII}} \; [\mathrm{cm}^{-2}]$', fontsize=fontsize)
    ax1.set_ylabel(r'$\log_{10} \left(\, \partial^2 n \,/\, \partial \log_{10} \mathrm{N} \, \partial X  \,\right)$', fontsize=fontsize)
    ax2.set_ylabel(r'$\log_{10}$ subset CDDF / total', fontsize=fontsize)
    setticks(ax1, fontsize=fontsize, labelbottom=False)
    setticks(ax2, fontsize=fontsize)
    
    plotx = bins[:-1] + 0.5 * np.diff(bins)
    
    for mi in range(len(masknames)):
        for fk in filekeys:
            mask = masknames[mi]
            #ax.step(bins[:-1], np.log10(hists[mask] / dXtotdlogN), where='post', color=colors[mi], label=labels[mask].expandtabs())
            ax1.plot(plotx, np.log10(hists[fk][mask] / dXtotdlogN[fk]), color=colors[mask], linestyle=linestyles[mask], alpha=alphas[fk], linewidth=linewidths[mask])
            ax2.plot(plotx, np.log10(hists[fk][mask] / hists[fk]['nomask']), color=colors[mask], linestyle=linestyles[mask], alpha=alphas[fk], linewidth=linewidths[mask])
            ax2.axhline(np.log10(fcovs[fk][mask]), color=colors[mask], linestyle=linestyles[mask], alpha=alphas[fk], linewidth=0.5)
    lax.axis('off')
    lax.legend(handles=legend_handles, fontsize=fontsize - 1, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    minmassexp = int(np.floor(np.log10(minmass)))
    minmassmul = minmass / 10**minmassexp
    ax1.text(0.05, 0.05, r'absorbers close to halos with' + '\n' + r'$\mathrm{M}_{200\mathrm{c}} \geq %.1f \times 10^{%i} \, \mathrm{M}_{\odot}$'%(minmassmul, minmassexp), horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    if imgname is not None:
        plt.savefig(imgname, format='pdf', bbox_inches='tight')
        

def plot_o7cddfsplits_isolated(incl=True, numsl=3):
    '''
    incl: True -> plot CDDFs and fraction within distances to the halos
          False -> plot the complements to those: fractions outside halo 
                   environments
    '''
    if numsl == 4:
        imgname = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_isolated_from_gals_%s.pdf'
    elif numsl == 3:
        imgname = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_33.3333333333slice_zcen-all_z-projection_T4EOS_masks_isolated_from_gals_%s.pdf'
    if incl:
        imgname = imgname%('incl')
    else:
        imgname = imgname%('excl')
    
    if numsl == 4:
        h5file1 = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.7-11.2_isolation.hdf5'
        h5file2 = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.3-11.0_isolation.hdf5'
        
        h5files = {'9.3-11.0': h5file2, '9.7-11.2': h5file1}
        
        masknames ={'9.7-11.2': ['nomask',\
                                 '630pkpc_pos_logMstar-geq-9.7-Msun',\
                                 '2273pkpc_pos_logMstar-geq-11.2-Msun',\
                                 '630pkpc_vel_logMstar-geq-9.7-Msun',\
                                 '2273pkpc_vel_logMstar-geq-11.2-Msun',\
                                ],\
                    '9.3-11.0':['nomask',\
                                 '630pkpc_pos_logMstar-geq-9.3-Msun',\
                                 '2273pkpc_pos_logMstar-geq-11.0-Msun',\
                                 '630pkpc_vel_logMstar-geq-9.3-Msun',\
                                 '2273pkpc_vel_logMstar-geq-11.0-Msun',\
                                ],\
                    }
        #masknames_all = masknames['9.7-11.2'] + masknames['9.3-11.0'][1:]
        masknames_legendsample = ['nomask', '630pkpc_pos_logMstar-geq-9.3-Msun', '630pkpc_pos_logMstar-geq-9.7-Msun', '2273pkpc_pos_logMstar-geq-11.0-Msun', '2273pkpc_pos_logMstar-geq-11.2-Msun']
    
    elif numsl == 3:
        h5file1 = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_33.3333333333slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.2-9.3-11.0-11.2_isolation.hdf5'
        
        h5files = {'9.3-9.7-11.0-11.2': h5file1}
        
        masknames ={'9.3-9.7-11.0-11.2': ['nomask',\
                                 '630pkpc_pos_logMstar-geq-9.7-Msun',\
                                 '2273pkpc_pos_logMstar-geq-11.2-Msun',\
                                 '630pkpc_vel_logMstar-geq-9.7-Msun',\
                                 '2273pkpc_vel_logMstar-geq-11.2-Msun',\
                                 '630pkpc_pos_logMstar-geq-9.3-Msun',\
                                 '2273pkpc_pos_logMstar-geq-11.0-Msun',\
                                 '630pkpc_vel_logMstar-geq-9.3-Msun',\
                                 '2273pkpc_vel_logMstar-geq-11.0-Msun',\
                                 ],\
                    }
        #masknames_all = masknames['9.7-11.2'] + masknames['9.3-11.0'][1:]
        masknames_legendsample = ['nomask', '630pkpc_pos_logMstar-geq-9.3-Msun', '630pkpc_pos_logMstar-geq-9.7-Msun', '2273pkpc_pos_logMstar-geq-11.0-Msun', '2273pkpc_pos_logMstar-geq-11.2-Msun']
        
    else:
        raise ValueError('Numsl should be 3 or 4')
    
    if incl:
        legendnames = {'nomask': 'total',\
                 '2273pkpc_pos_logMstar-geq-11.2-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.2) < 2273 \, \mathrm{pkpc}$',\
                 '630pkpc_pos_logMstar-geq-9.7-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.7) < 630 \, \mathrm{pkpc}$',\
                 '2273pkpc_pos_logMstar-geq-11.0-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.0) < 2273 \, \mathrm{pkpc}$',\
                 '630pkpc_pos_logMstar-geq-9.3-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.3) < 630 \, \mathrm{pkpc}$',\
                 '2273pkpc_vel_logMstar-geq-11.2-Msun': None,\
                 '630pkpc_vel_logMstar-geq-9.7-Msun':  None,\
                 '2273pkpc_vel_logMstar-geq-11.0-Msun': None,\
                 '630pkpc_vel_logMstar-geq-9.3-Msun':  None,\
                 }
    else:
        legendnames = {'nomask': 'total',\
                 '2273pkpc_pos_logMstar-geq-11.2-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.2) > 2273 \, \mathrm{pkpc}$',\
                 '630pkpc_pos_logMstar-geq-9.7-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.7) > 630 \, \mathrm{pkpc}$',\
                 '2273pkpc_pos_logMstar-geq-11.0-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.0) > 2273 \, \mathrm{pkpc}$',\
                 '630pkpc_pos_logMstar-geq-9.3-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.3) > 630 \, \mathrm{pkpc}$',\
                 '2273pkpc_vel_logMstar-geq-11.2-Msun': None,\
                 '630pkpc_vel_logMstar-geq-9.7-Msun':  None,\
                 '2273pkpc_vel_logMstar-geq-11.0-Msun': None,\
                 '630pkpc_vel_logMstar-geq-9.3-Msun':  None,\
                 }
        
    colors = {'nomask': 'black',\
             '2273pkpc_pos_logMstar-geq-11.2-Msun': 'C0',\
             '2273pkpc_vel_logMstar-geq-11.2-Msun': 'C0',\
             '630pkpc_pos_logMstar-geq-9.7-Msun': 'C1',\
             '630pkpc_vel_logMstar-geq-9.7-Msun': 'C1',\
             '2273pkpc_pos_logMstar-geq-11.0-Msun': 'C2',\
             '2273pkpc_vel_logMstar-geq-11.0-Msun': 'C2',\
             '630pkpc_pos_logMstar-geq-9.3-Msun': 'C3',\
             '630pkpc_vel_logMstar-geq-9.3-Msun': 'C3',\
             }      
    linestyles = {'nomask': 'solid',\
             '2273pkpc_pos_logMstar-geq-11.2-Msun': 'solid',\
             '2273pkpc_vel_logMstar-geq-11.2-Msun': 'dashed',\
             '630pkpc_pos_logMstar-geq-9.7-Msun': 'solid',\
             '630pkpc_vel_logMstar-geq-9.7-Msun': 'dashed',\
             '2273pkpc_pos_logMstar-geq-11.0-Msun': 'solid',\
             '2273pkpc_vel_logMstar-geq-11.0-Msun': 'dashed',\
             '630pkpc_pos_logMstar-geq-9.3-Msun': 'solid',\
             '630pkpc_vel_logMstar-geq-9.3-Msun': 'dashed',\
             }  
    linewidths = {'nomask': 2.,\
             '2273pkpc_pos_logMstar-geq-11.2-Msun': 2.,\
             '2273pkpc_vel_logMstar-geq-11.2-Msun': 1.5,\
             '630pkpc_pos_logMstar-geq-9.7-Msun': 2.,\
             '630pkpc_vel_logMstar-geq-9.7-Msun': 1.5,\
             '2273pkpc_pos_logMstar-geq-11.0-Msun': 2.,\
             '2273pkpc_vel_logMstar-geq-11.0-Msun': 1.5,\
             '630pkpc_pos_logMstar-geq-9.3-Msun': 2.,\
             '630pkpc_vel_logMstar-geq-9.3-Msun': 1.5,\
             }  

    legend_handles = [mlines.Line2D([], [], color=colors[mask], linestyle='solid', label=legendnames[mask], linewidth=2.) for mask in masknames_legendsample]
    legend_handles_ls =  [mlines.Line2D([], [], color='gray', linestyle='solid', label='los pos.', linewidth=2.),\
                          mlines.Line2D([], [], color='gray', linestyle='dashed', label='los vel.', linewidth=1.5)] 
    
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    
    hists = {}
    cosmopars = {}
    fcovs = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for filekey in h5files.keys():
        filename = h5files[filekey]
        with h5py.File(pdir + filename, 'r') as fi:
            bins[filekey] = np.array(fi['bins/axis_0'])
            # handle +- infinity edges for plotting; should be outside the plot range anyway
            if bins[filekey][0] == -np.inf:
                bins[filekey][0] = -100.
            if bins[filekey][-1] == np.inf:
                bins[filekey][-1] = 100.
            
            # extract number of pixels from the input filename, using naming system of make_maps
            inname = np.array(fi['input_filenames'])[0]
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            ionind = 1 + np.where(np.array([part == 'coldens' for part in parts]))[0][0]
            ion = parts[ionind]
            
            masks = masknames[filekey]
    
            hists[filekey] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
            fcovs[filekey] = {mask: fi[mask].attrs['covfrac'] for mask in hists[filekey].keys()}
            
            if numsl == 4:
                examplemaskdir = 'masks/12.5/'
            elif numsl == 3:
                examplemaskdir = 'masks/50.0/'
            examplemask = fi[examplemaskdir].keys()[0]
            cosmopars[filekey] = {key: item for (key, item) in fi['%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
            dXtot[filekey] = mc.getdX(cosmopars[filekey]['z'], cosmopars[filekey]['boxsize'] / cosmopars[filekey]['h'], cosmopars=cosmopars[filekey]) * float(numpix_1sl**2)
            dztot[filekey] = mc.getdz(cosmopars[filekey]['z'], cosmopars[filekey]['boxsize'] / cosmopars[filekey]['h'], cosmopars=cosmopars[filekey]) * float(numpix_1sl**2)
            dXtotdlogN[filekey] = dXtot[filekey] * np.diff(bins[filekey])
        
    #legend_handles = legend_handles + [mlines.Line2D([], [], color='brown', linestyle='solid', alpha=alphas[key], label=r'$z=%.2f$'%cosmopars[key]['z'], linewidth=2.) for key in filekeys] 
    
    filekeys = h5files.keys()
    if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
        bins = bins[filekeys[0]]
    else:
        raise RuntimeError("bins for different files don't match")
    
    if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
        raise RuntimeError('total histograms from different files do not match')
        
    fig = plt.figure(figsize=(5.5, 5.5))
    grid = gsp.GridSpec(2, 1, hspace=0.0, wspace=0.0, height_ratios=[4., 4.])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    lax = ax1
    
    fontsize = 12
    if ion[0] == 'h':
        ax1.set_xlim(12.5, 23.5)
        ax1.set_ylim(-6., 1.65)
    else:
        ax1.set_xlim(14., 17.5)
        ax1.set_ylim(-6.5, 1.6)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0., 1.05) 
    
    ax2.set_xlabel(r'$\log_{10}\, \mathrm{N}_{\mathrm{O\,VII}} \; [\mathrm{cm}^{-2}]$', fontsize=fontsize)
    ax1.set_ylabel(r'$\log_{10} \left(\, \partial^2 n \,/\, \partial \log_{10} \mathrm{N} \, \partial X  \,\right)$', fontsize=fontsize)
    ax2.set_ylabel(r'subset CDDF / total', fontsize=fontsize)
    setticks(ax1, fontsize=fontsize, labelbottom=False)
    setticks(ax2, fontsize=fontsize)
    
    plotx = bins[:-1] + 0.5 * np.diff(bins)
    
    ax2.axhline(1., color='black', linestyle='solid', linewidth=0.5)
    for fk in filekeys:
        for mi in range(len(masknames[fk])):
            mask = masknames[fk][mi]
            if incl or mask == 'nomask':
                #ax.step(bins[:-1], np.log10(hists[mask] / dXtotdlogN), where='post', color=colors[mi], label=labels[mask].expandtabs())
                ax1.plot(plotx, np.log10(hists[fk][mask] / dXtotdlogN[fk]), color=colors[mask], linestyle=linestyles[mask], linewidth=linewidths[mask])
                ax2.plot(plotx, hists[fk][mask] / hists[fk]['nomask'], color=colors[mask], linestyle=linestyles[mask], linewidth=linewidths[mask])
                ax2.axhline(fcovs[fk][mask], color=colors[mask], linestyle=linestyles[mask], linewidth=0.5)
            else:
                ax1.plot(plotx, np.log10((hists[fk]['nomask'] - hists[fk][mask]) / dXtotdlogN[fk]), color=colors[mask], linestyle=linestyles[mask], linewidth=linewidths[mask])
                ax2.plot(plotx, (hists[fk]['nomask'] - hists[fk][mask]) / hists[fk]['nomask'], color=colors[mask], linestyle=linestyles[mask], linewidth=linewidths[mask])
                ax2.axhline(1. - fcovs[fk][mask], color=colors[mask], linestyle=linestyles[mask], linewidth=0.5)
    
    ### document fractions at logN = 15.575fk
    Nval = 15.575
    if numsl == 4:
        cddf_tot     = linterpsolve(plotx, np.log10((hists['9.7-11.2']['nomask']) / dXtotdlogN['9.7-11.2']), Nval)
        cddf_pos_9p7 = linterpsolve(plotx, np.log10((hists['9.7-11.2']['nomask'] - hists['9.7-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.7-11.2']), Nval)
        cddf_vel_9p7 = linterpsolve(plotx, np.log10((hists['9.7-11.2']['nomask'] - hists['9.7-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.7-11.2']), Nval)
        cddf_pos_9p3 = linterpsolve(plotx, np.log10((hists['9.3-11.0']['nomask'] - hists['9.3-11.0']['630pkpc_pos_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-11.0']), Nval)
        cddf_vel_9p3 = linterpsolve(plotx, np.log10((hists['9.3-11.0']['nomask'] - hists['9.3-11.0']['630pkpc_vel_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-11.0']), Nval)
        
        frac_pos_9p7 = linterpsolve(plotx, 1. - hists['9.7-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun'] / hists['9.7-11.2']['nomask'], Nval)
        frac_vel_9p7 = linterpsolve(plotx, 1. - hists['9.7-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun'] / hists['9.7-11.2']['nomask'], Nval)
        frac_pos_9p3 = linterpsolve(plotx, 1. - hists['9.3-11.0']['630pkpc_pos_logMstar-geq-9.3-Msun'] / hists['9.3-11.0']['nomask'], Nval)
        frac_vel_9p3 = linterpsolve(plotx, 1. - hists['9.3-11.0']['630pkpc_vel_logMstar-geq-9.3-Msun'] / hists['9.3-11.0']['nomask'], Nval)
    
        dXoverdz = dXtot['9.7-11.2'] / dztot['9.7-11.2']
        
    elif numsl == 3:
        cddf_tot     = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        cddf_pos_9p7 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        cddf_vel_9p7 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        cddf_pos_9p3 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        cddf_vel_9p3 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        
        frac_pos_9p7 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        frac_vel_9p7 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        frac_pos_9p3 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.3-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        frac_vel_9p3 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.3-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        print(('%s, '*4)%(frac_pos_9p7, frac_vel_9p7, frac_pos_9p3, frac_vel_9p3))
        
        dXoverdz = dXtot['9.3-9.7-11.0-11.2'] / dztot['9.3-9.7-11.0-11.2']
        
        histc_x = bins[:-1]
        histc_pos_9p7 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun']
        histc_vel_9p7 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun']
        histc_pos_9p3 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.3-Msun']
        histc_vel_9p3 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.3-Msun']
        histc_all = hists['9.3-9.7-11.0-11.2']['nomask']
        
        histc_pos_9p7 = np.cumsum(histc_pos_9p7[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        histc_vel_9p7 = np.cumsum(histc_vel_9p7[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        histc_pos_9p3 = np.cumsum(histc_pos_9p3[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        histc_vel_9p3 = np.cumsum(histc_vel_9p3[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        histc_all =  np.cumsum(histc_all[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz

        p_N_iso_pos_9p7 = linterpsolve(histc_x, histc_pos_9p7, Nval)
        p_N_iso_vel_9p7 = linterpsolve(histc_x, histc_vel_9p7, Nval)
        p_N_iso_pos_9p3 = linterpsolve(histc_x, histc_pos_9p3, Nval)
        p_N_iso_vel_9p3 = linterpsolve(histc_x, histc_vel_9p3, Nval)
        
        Nvals = [15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0] + [Nval]
        frac_cumul_pos_9p7 = {Nval: linterpsolve(histc_x, histc_pos_9p7 / histc_all, Nval) for Nval in Nvals}
        frac_cumul_vel_9p7 = {Nval: linterpsolve(histc_x, histc_vel_9p7 / histc_all, Nval) for Nval in Nvals}
        frac_cumul_pos_9p3 = {Nval: linterpsolve(histc_x, histc_pos_9p3 / histc_all, Nval) for Nval in Nvals}
        frac_cumul_vel_9p3 = {Nval: linterpsolve(histc_x, histc_vel_9p3 / histc_all, Nval) for Nval in Nvals}
        
    print('at log N(O VII) / cm^-2 = %s, contributions from regions isolated from galaxies at log M*/Msun = 9.3, 9.7 by at least 630 pkpc:'%Nval)
    print('\tlog10 CDDF (total) = %s'%cddf_tot)
    print('\tlog10 CDDF (pos, ge 9.7) = %s'%cddf_pos_9p7)
    print('\tlog10 CDDF (pos, ge 9.3) = %s'%cddf_pos_9p3)
    print('\tlog10 CDDF (vel, ge 9.7) = %s'%cddf_vel_9p7)
    print('\tlog10 CDDF (vel, ge 9.3) = %s'%cddf_vel_9p3)
    print('\tCDDF (pos, ge 9.7) / total = %s'%frac_pos_9p7)
    print('\tCDDF (pos, ge 9.3) / total = %s'%frac_pos_9p3)
    print('\tCDDF (vel, ge 9.7) / total = %s'%frac_vel_9p7)
    print('\tCDDF (vel, ge 9.3) / total = %s'%frac_vel_9p3)
    print('\tN_expected / dz (pos, ge 9.7) = %s'%p_N_iso_pos_9p7)
    print('\tN_expected / dz (pos, ge 9.3) = %s'%p_N_iso_pos_9p3)
    print('\tN_expected / dz (vel, ge 9.7) = %s'%p_N_iso_vel_9p7)
    print('\tN_expected / dz (vel, ge 9.3) = %s'%p_N_iso_vel_9p3)
    print('\t dX / dz = %s'%dXoverdz)
    
    print('\nratios of cumulative CDDF with isolation criteria to cumulative CDDF total above column densities of')
    fillstr = '\t' + '\t'.join(['%.4f'] * len(Nvals))
    print('\t\t' + fillstr%tuple(Nvals))
    print('\tpos 9.7\t' + fillstr%(tuple([frac_cumul_pos_9p7[Nval] for Nval in Nvals])))
    print('\tvel 9.7\t' + fillstr%(tuple([frac_cumul_vel_9p7[Nval] for Nval in Nvals])))
    print('\tpos 9.3\t' + fillstr%(tuple([frac_cumul_pos_9p3[Nval] for Nval in Nvals])))
    print('\tvel 9.3\t' + fillstr%(tuple([frac_cumul_vel_9p3[Nval] for Nval in Nvals])))
    
    #lax.axis('off')
    leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    lax.add_artist(leg1)
    lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    if imgname is not None:
        plt.savefig(imgname, format='pdf', bbox_inches='tight')
        
    
def plot_o7_radprof(xlog=False):
    imgname = 'fcov_rdist_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS%s.pdf'
    if xlog:
        imgname = imgname%('xlog')
    else:
        imgname = imgname%('xlin')
    #
    h5files = {'ge9.7_pos': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-9.7_posspace_stored_profiles.hdf5',\
               'ge9.7_vel': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-9.7_velspace_stored_profiles.hdf5',\
               '9.6-9.8_pos': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-9.6-9.8_posspace_stored_profiles.hdf5',\
               '9.6-9.8_vel': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-9.6-9.8_velspace_stored_profiles.hdf5',\
               'ge11.2_pos': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-11.2_posspace_stored_profiles.hdf5',\
               'ge11.2_vel': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-11.2_velspace_stored_profiles.hdf5',\
               '11.1-11.3_pos': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-11.1-11.3_posspace_stored_profiles.hdf5',\
               '11.1-11.3_vel': 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-11.1-11.3_velspace_stored_profiles.hdf5',\
               }
    
    halocat = '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap24_aperture30.hdf5'
    with h5py.File(halocat, 'r') as hc:
        cosmopars = {key: item for (key, item) in hc['Header/cosmopars'].attrs.items()}
        
    h5path = 'galset_0/pkpc_bins/binset_0/'
    
    fcovs = [15.0, 15.2, 15.5] #[14.5, 15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 16.0]
    
    colors_fcov = {14.5: 'C7',\
                   15.0: 'C4',\
                   15.1: 'C0',\
                   15.2: 'C2',\
                   15.3: 'C8',\
                   15.4: 'C1',\
                   15.5: 'C3',\
                   16.0: 'C5'}
    
    linewidths_fcov= {14.5: 1.5,\
                      15.0: 2.,\
                      15.1: 2.,\
                      15.2: 2.5,\
                      15.3: 2.,\
                      15.4: 2.,\
                      15.5: 2.,\
                      16.0: 1.5,\
                      }
    
    linestyles_fk = {'ge9.7_pos': 'dashed',\
                     'ge9.7_vel': 'dashed',\
                     '9.6-9.8_pos': 'solid',\
                     '9.6-9.8_vel': 'solid',\
                     'ge11.2_pos': 'dotted',\
                     'ge11.2_vel': 'dotted',\
                     '11.1-11.3_pos': 'dashdot',\
                     '11.1-11.3_vel': 'dashdot',\
                    }
    
    alphas_fk = {'ge9.7_pos': 1.,\
                 'ge9.7_vel': 0.5,\
                 '9.6-9.8_pos': 1.,\
                 '9.6-9.8_vel': 0.5,\
                 'ge11.2_pos': 1.,\
                 'ge11.2_vel': 0.5,\
                 '11.1-11.3_pos': 1.,\
                 '11.1-11.3_vel': 0.5,\
                 }
    
    bins = {}
    covfracs = {}
    
    for filekey in h5files.keys():
        filename = h5files[filekey]
        bins[filekey] = {}
        covfracs[filekey] = {}
        with h5py.File(pdir + 'radprof/' + filename, 'r') as fi:
            if fi[h5path.split('/')[0]].attrs['seltag'] != 'all':
                raise RuntimeError('Selected wrong galaxy set')
            bins[filekey] = np.array(fi[h5path + 'bin_edges'])
            for cov in fcovs:
                covfracs[filekey][cov] = np.array(fi[h5path + 'fcov_%s'%cov])
            
    # get whole map covering fractions
    histfile = '/net/luttero/data2/proc/cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.7-11.2_isolation.hdf5'
    with h5py.File(histfile, 'r') as hf:
        hbins = np.array(hf['bins/axis_0'])
        hist = np.array(hf['nomask/hist'])
        
        covfracs_all = {cov: np.sum(hist[hbins[:-1] >= cov]) / np.sum(hist) for cov in fcovs}
    
    # M = 4 * pi /3 R^3 rho -> R = (3 M / 4 pi rho)^(1/3)
    #rho_crit_0 = 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars['h']**2
    #rho_crit = rho_crit_0 * (cosmopars['omegam'] / cosmopars['a']**3 + cosmopars['omegalambda'])
    #rhalo_9p7_pkpc = (10**9.7 * c.solar_mass * 3. / (4. * np.pi * rho_crit))**(1./3.) / (1e-3 * c.cm_per_mpc)
    #rhalo_11p2_pkpc = (10**11.2 * c.solar_mass * 3. / (4. * np.pi * rho_crit))**(1./3.) / (1e-3 * c.cm_per_mpc)
    
    
    fig = plt.figure(figsize=(5.5, 3.))
    ax = fig.add_subplot(111)
    fontsize = 12
    ax.set_xlabel(r'$r_{\perp} \; [\mathrm{pkpc}]$', fontsize=fontsize)
    ax.set_ylabel(r'$f_{\mathrm{cov}}(N > N_{\min})$', fontsize=fontsize)
    if xlog:
        ax.set_xscale('log')
    
    for cov in fcovs:
        ax.axhline(covfracs_all[cov], color=colors_fcov[cov], linestyle='solid', linewidth=1.5, alpha=0.7)
    filekeys = h5files.keys()
    for fk in filekeys:
        xvals = bins[fk]
        xvals = xvals[:-1] + 0.5 * np.diff(xvals)
        for cov in fcovs:
            ax.plot(xvals, covfracs[fk][cov], color=colors_fcov[cov], linestyle=linestyles_fk[fk], linewidth=linewidths_fcov[cov], alpha=alphas_fk[fk])
    
    ax.set_xlim(0., 3500.)
    xlim = ax.get_xlim()
    
    #ax.axvline(rhalo_9p7_pkpc, linestyle='solid', linewidth=1.5, color='black', alpha=0.5)
    #if xlog:
    #    xpos = (np.log10(rhalo_9p7_pkpc) - np.log10(xlim[0])) / (np.log10(xlim[1]) - np.log10(xlim[0]))
    #else:
    #    xpos = (rhalo_9p7_pkpc - xlim[0]) / (xlim[1] - xlim[0])
    #ax.text(xpos, 0.99, r'$\mathrm{R}_{200\mathrm{c}}(9.7)$', fontsize=fontsize - 1, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')
    
    #ax.axvline(rhalo_11p2_pkpc, linestyle='dashdot', linewidth=1.5, color='black', alpha=0.5)
    #if xlog:
    #    xpos = (np.log10(rhalo_11p2_pkpc) - np.log10(xlim[0])) / (np.log10(xlim[1]) - np.log10(xlim[0]))
    #else:
    #    xpos = (rhalo_11p2_pkpc - xlim[0]) / (xlim[1] - xlim[0])
    #ax.text(xpos, 0.85, r'$\mathrm{R}_{200\mathrm{c}}(11.2)$', fontsize=fontsize - 1, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left')


    legend_handles = [mlines.Line2D([], [], color=colors_fcov[cov], linestyle='solid', linewidth=linewidths_fcov[cov], alpha=1., label=r'$N_{\min}=%.1f$'%cov) for cov in fcovs] 
    legend_handles_ls = [\
        mlines.Line2D([], [], color='brown', linestyle='dashed',  linewidth=2., alpha=1.,  label=r'$M_{*}\, \geq 9.7$, pos.'),\
        mlines.Line2D([], [], color='brown', linestyle='solid',   linewidth=2., alpha=1.,  label=r'$M_{*}\, 9.6 \endash 9.8$'),\
        mlines.Line2D([], [], color='brown', linestyle='dotted',  linewidth=2., alpha=1.,  label=r'$M_{*}\, \geq 11.2$'),\
        mlines.Line2D([], [], color='brown', linestyle='dashdot', linewidth=2., alpha=1.,  label=r'$M_{*}\, 11.1 \endash 11.3$'),\
        mlines.Line2D([], [], color='brown', linestyle='dashed',  linewidth=2., alpha=0.5, label=r'vel.'),\
                        ]

    #leg1 = ax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    leg2 = ax.legend(handles=legend_handles_ls + legend_handles,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False, ncol=2)
    #ax.add_artist(leg1)
    ax.add_artist(leg2)

    plt.savefig(mdir + imgname, format='pdf', bbox_inches='tight')
    
    
def plot_o7cddfsplits_isolated_thesisversion(incl=True):
    '''
    incl: True -> plot CDDFs and fraction within distances to the halos
          False -> plot the complements to those: fractions outside halo 
                   environments
    '''
    # Delta v = +- 1000 km/s, matches galaxy search range in the paper
    numsl=3
    _cs = tc.tol_cset('vibrant')
     
    if numsl == 4:
        imgname = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_isolated_from_gals_%s_thesisversion.pdf'
    elif numsl == 3:
        imgname = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_33.3333333333slice_zcen-all_z-projection_T4EOS_masks_isolated_from_gals_%s_thesisversion.pdf'
    if incl:
        imgname = imgname%('incl')
    else:
        imgname = imgname%('excl')
    
    if numsl == 4:
        h5file1 = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.7-11.2_isolation.hdf5'
        h5file2 = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.3-11.0_isolation.hdf5'
        
        h5files = {'9.3-11.0': h5file2, '9.7-11.2': h5file1}
        
        masknames ={'9.7-11.2': ['nomask',\
                                 '630pkpc_pos_logMstar-geq-9.7-Msun',\
                                 '2273pkpc_pos_logMstar-geq-11.2-Msun',\
                                 '630pkpc_vel_logMstar-geq-9.7-Msun',\
                                 '2273pkpc_vel_logMstar-geq-11.2-Msun',\
                                ],\
                    '9.3-11.0':['nomask',\
                                 '630pkpc_pos_logMstar-geq-9.3-Msun',\
                                 '2273pkpc_pos_logMstar-geq-11.0-Msun',\
                                 '630pkpc_vel_logMstar-geq-9.3-Msun',\
                                 '2273pkpc_vel_logMstar-geq-11.0-Msun',\
                                ],\
                    }
        #masknames_all = masknames['9.7-11.2'] + masknames['9.3-11.0'][1:]
        masknames_legendsample = ['nomask', '630pkpc_pos_logMstar-geq-9.3-Msun', '630pkpc_pos_logMstar-geq-9.7-Msun', '2273pkpc_pos_logMstar-geq-11.0-Msun', '2273pkpc_pos_logMstar-geq-11.2-Msun']
    
    elif numsl == 3:
        h5file1 = 'cddf_coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_33.3333333333slice_zcen-all_z-projection_T4EOS_masks_Mstar-9.2-9.3-11.0-11.2_isolation.hdf5'
        
        h5files = {'9.3-9.7-11.0-11.2': h5file1}
        
        masknames ={'9.3-9.7-11.0-11.2': ['nomask',\
                                 '630pkpc_pos_logMstar-geq-9.7-Msun',
                                 '2273pkpc_pos_logMstar-geq-11.2-Msun',
                                 '630pkpc_vel_logMstar-geq-9.7-Msun',
                                 '2273pkpc_vel_logMstar-geq-11.2-Msun',
                                 #'630pkpc_pos_logMstar-geq-9.3-Msun',
                                 #'2273pkpc_pos_logMstar-geq-11.0-Msun',
                                 #'630pkpc_vel_logMstar-geq-9.3-Msun',
                                 #'2273pkpc_vel_logMstar-geq-11.0-Msun',
                                 ],
                    }
        #masknames_all = masknames['9.7-11.2'] + masknames['9.3-11.0'][1:]
        masknames_legendsample = ['nomask', 
                                  #'630pkpc_pos_logMstar-geq-9.3-Msun', 
                                  '630pkpc_pos_logMstar-geq-9.7-Msun', 
                                  #'2273pkpc_pos_logMstar-geq-11.0-Msun', 
                                  '2273pkpc_pos_logMstar-geq-11.2-Msun']
        
    else:
        raise ValueError('Numsl should be 3 or 4')
    
    if incl:
        legendnames = {'nomask': 'total',\
                 '2273pkpc_pos_logMstar-geq-11.2-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.2) < 2273 \, \mathrm{pkpc}$',\
                 '630pkpc_pos_logMstar-geq-9.7-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.7) < 630 \, \mathrm{pkpc}$',\
                 #'2273pkpc_pos_logMstar-geq-11.0-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.0) < 2273 \, \mathrm{pkpc}$',\
                 #'630pkpc_pos_logMstar-geq-9.3-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.3) < 630 \, \mathrm{pkpc}$',\
                 '2273pkpc_vel_logMstar-geq-11.2-Msun': None,\
                 '630pkpc_vel_logMstar-geq-9.7-Msun':  None,\
                 #'2273pkpc_vel_logMstar-geq-11.0-Msun': None,\
                 #'630pkpc_vel_logMstar-geq-9.3-Msun':  None,\
                 }
    else:
        legendnames = {'nomask': 'total',\
                 '2273pkpc_pos_logMstar-geq-11.2-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.2) > 2273 \, \mathrm{pkpc}$',\
                 '630pkpc_pos_logMstar-geq-9.7-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.7) > 630 \, \mathrm{pkpc}$',\
                 #'2273pkpc_pos_logMstar-geq-11.0-Msun': r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 11.0) > 2273 \, \mathrm{pkpc}$',\
                 #'630pkpc_pos_logMstar-geq-9.3-Msun':  r'$r_{\perp}(\log \, \mathrm{M}_{*} [\mathrm{M}_{\odot}] \geq 9.3) > 630 \, \mathrm{pkpc}$',\
                 '2273pkpc_vel_logMstar-geq-11.2-Msun': None,\
                 '630pkpc_vel_logMstar-geq-9.7-Msun':  None,\
                 #'2273pkpc_vel_logMstar-geq-11.0-Msun': None,\
                 #'630pkpc_vel_logMstar-geq-9.3-Msun':  None,\
                 }
        
    colors = {'nomask': 'black',\
             '2273pkpc_pos_logMstar-geq-11.2-Msun': _cs.orange,
             '2273pkpc_vel_logMstar-geq-11.2-Msun': _cs.orange,
             '630pkpc_pos_logMstar-geq-9.7-Msun': _cs.blue,
             '630pkpc_vel_logMstar-geq-9.7-Msun': _cs.blue,
             #'2273pkpc_pos_logMstar-geq-11.0-Msun': 'C2',\
             #'2273pkpc_vel_logMstar-geq-11.0-Msun': 'C2',\
             #'630pkpc_pos_logMstar-geq-9.3-Msun': 'C3',\
             #'630pkpc_vel_logMstar-geq-9.3-Msun': 'C3',\
             }      
    linestyles = {'nomask': 'solid',\
             '2273pkpc_pos_logMstar-geq-11.2-Msun': 'solid',\
             '2273pkpc_vel_logMstar-geq-11.2-Msun': 'dashed',\
             '630pkpc_pos_logMstar-geq-9.7-Msun': 'solid',\
             '630pkpc_vel_logMstar-geq-9.7-Msun': 'dashed',\
             #'2273pkpc_pos_logMstar-geq-11.0-Msun': 'solid',\
             #'2273pkpc_vel_logMstar-geq-11.0-Msun': 'dashed',\
             #'630pkpc_pos_logMstar-geq-9.3-Msun': 'solid',\
             #'630pkpc_vel_logMstar-geq-9.3-Msun': 'dashed',\
             }  
    linewidths = {'nomask': 2.,\
             '2273pkpc_pos_logMstar-geq-11.2-Msun': 2.,\
             '2273pkpc_vel_logMstar-geq-11.2-Msun': 1.5,\
             '630pkpc_pos_logMstar-geq-9.7-Msun': 2.,\
             '630pkpc_vel_logMstar-geq-9.7-Msun': 1.5,\
             #'2273pkpc_pos_logMstar-geq-11.0-Msun': 2.,\
             #'2273pkpc_vel_logMstar-geq-11.0-Msun': 1.5,\
             #'630pkpc_pos_logMstar-geq-9.3-Msun': 2.,\
             #'630pkpc_vel_logMstar-geq-9.3-Msun': 1.5,\
             }  

    legend_handles = [mlines.Line2D([], [], color=colors[mask], linestyle='solid', 
                      label=legendnames[mask], linewidth=2.) 
                      for mask in masknames_legendsample]
    legend_handles_ls =  [mlines.Line2D([], [], color='gray', linestyle='solid', 
                          label='los pos.', linewidth=2.),\
                          mlines.Line2D([], [], color='gray', linestyle='dashed', 
                          label='los vel.', linewidth=1.5)] 
    
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    
    hists = {}
    cosmopars = {}
    fcovs = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for filekey in h5files.keys():
        filename = h5files[filekey]
        with h5py.File(pdir + filename, 'r') as fi:
            bins[filekey] = np.array(fi['bins/axis_0'])
            # handle +- infinity edges for plotting; should be outside the plot range anyway
            if bins[filekey][0] == -np.inf:
                bins[filekey][0] = -100.
            if bins[filekey][-1] == np.inf:
                bins[filekey][-1] = 100.
            
            # extract number of pixels from the input filename, using naming system of make_maps
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            ionind = 1 + np.where(np.array([part == 'coldens' for part in parts]))[0][0]
            ion = parts[ionind]
            
            masks = masknames[filekey]
    
            hists[filekey] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
            fcovs[filekey] = {mask: fi[mask].attrs['covfrac'] for mask in hists[filekey].keys()}
            
            if numsl == 4:
                examplemaskdir = 'masks/12.5/'
            elif numsl == 3:
                examplemaskdir = 'masks/50.0/'
            examplemask = fi[examplemaskdir].keys()[0]
            fpath = '%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)
            cosmopars[filekey] = {key: item for (key, item) in 
                                  fi[fpath].attrs.items()}
            dXtot[filekey] = mc.getdX(cosmopars[filekey]['z'], 
                                      cosmopars[filekey]['boxsize'] / cosmopars[filekey]['h'], 
                                      cosmopars=cosmopars[filekey]) * float(numpix_1sl**2)
            dztot[filekey] = mc.getdz(cosmopars[filekey]['z'], 
                                      cosmopars[filekey]['boxsize'] / cosmopars[filekey]['h'], 
                                      cosmopars=cosmopars[filekey]) * float(numpix_1sl**2)
            dXtotdlogN[filekey] = dXtot[filekey] * np.diff(bins[filekey])
        
    #legend_handles = legend_handles + [mlines.Line2D([], [], color='brown', linestyle='solid', alpha=alphas[key], label=r'$z=%.2f$'%cosmopars[key]['z'], linewidth=2.) for key in filekeys] 
    
    filekeys = h5files.keys()
    if np.all([np.all(bins[key] == bins[filekeys[0]]) \
               if len(bins[key]) == len(bins[filekeys[0]]) else False \
               for key in filekeys]):
        bins = bins[filekeys[0]]
    else:
        raise RuntimeError("bins for different files don't match")
    
    if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask'])\
                  for key in filekeys])):
        raise RuntimeError('total histograms from different files do not match')
        
    fig = plt.figure(figsize=(5.5, 5.5))
    grid = gsp.GridSpec(2, 1, hspace=0.0, wspace=0.0, height_ratios=[4., 4.])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    lax = ax1
    
    fontsize = 12
    if ion[0] == 'h':
        ax1.set_xlim(12.5, 23.5)
        ax1.set_ylim(-6., 1.65)
    else:
        ax1.set_xlim(14., 17.5)
        ax1.set_ylim(-6.5, 1.6)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(0., 1.05) 
    
    ax2.set_xlabel(r'$\log_{10}\, \mathrm{N}_{\mathrm{O\,VII}} \; [\mathrm{cm}^{-2}]$', fontsize=fontsize)
    ax1.set_ylabel(r'$\log_{10} \left(\, \partial^2 n \,/\, \partial \log_{10} \mathrm{N} \, \partial X  \,\right)$', fontsize=fontsize)
    ax2.set_ylabel(r'subset CDDF / total', fontsize=fontsize)
    setticks(ax1, fontsize=fontsize, labelbottom=False)
    setticks(ax2, fontsize=fontsize)
    
    plotx = bins[:-1] + 0.5 * np.diff(bins)
    
    ax2.axhline(1., color='black', linestyle='solid', linewidth=0.5)
    for fk in filekeys:
        for mi in range(len(masknames[fk])):
            mask = masknames[fk][mi]
            if incl or mask == 'nomask':
                #ax.step(bins[:-1], np.log10(hists[mask] / dXtotdlogN), where='post', color=colors[mi], label=labels[mask].expandtabs())
                ax1.plot(plotx, np.log10(hists[fk][mask] / dXtotdlogN[fk]), 
                         color=colors[mask], linestyle=linestyles[mask], 
                         linewidth=linewidths[mask])
                ax2.plot(plotx, hists[fk][mask] / hists[fk]['nomask'], color=colors[mask], 
                         linestyle=linestyles[mask], linewidth=linewidths[mask])
                ax2.axhline(fcovs[fk][mask], color=colors[mask], 
                            linestyle=linestyles[mask], linewidth=0.5)
            else:
                ax1.plot(plotx, np.log10((hists[fk]['nomask'] - hists[fk][mask]) / dXtotdlogN[fk]), 
                         color=colors[mask], linestyle=linestyles[mask], 
                         linewidth=linewidths[mask])
                ax2.plot(plotx, (hists[fk]['nomask'] - hists[fk][mask]) / hists[fk]['nomask'], 
                                 color=colors[mask], linestyle=linestyles[mask], 
                                 linewidth=linewidths[mask])
                ax2.axhline(1. - fcovs[fk][mask], color=colors[mask], 
                            linestyle=linestyles[mask], linewidth=0.5)
    
    ### document fractions at logN = 15.575fk
    Nval = 15.575
    if numsl == 4:
        cddf_tot     = linterpsolve(plotx, np.log10((hists['9.7-11.2']['nomask']) / dXtotdlogN['9.7-11.2']), Nval)
        cddf_pos_9p7 = linterpsolve(plotx, np.log10((hists['9.7-11.2']['nomask'] - hists['9.7-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.7-11.2']), Nval)
        cddf_vel_9p7 = linterpsolve(plotx, np.log10((hists['9.7-11.2']['nomask'] - hists['9.7-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.7-11.2']), Nval)
        #cddf_pos_9p3 = linterpsolve(plotx, np.log10((hists['9.3-11.0']['nomask'] - hists['9.3-11.0']['630pkpc_pos_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-11.0']), Nval)
        #cddf_vel_9p3 = linterpsolve(plotx, np.log10((hists['9.3-11.0']['nomask'] - hists['9.3-11.0']['630pkpc_vel_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-11.0']), Nval)
        
        frac_pos_9p7 = linterpsolve(plotx, 1. - hists['9.7-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun'] / hists['9.7-11.2']['nomask'], Nval)
        frac_vel_9p7 = linterpsolve(plotx, 1. - hists['9.7-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun'] / hists['9.7-11.2']['nomask'], Nval)
        #frac_pos_9p3 = linterpsolve(plotx, 1. - hists['9.3-11.0']['630pkpc_pos_logMstar-geq-9.3-Msun'] / hists['9.3-11.0']['nomask'], Nval)
        #frac_vel_9p3 = linterpsolve(plotx, 1. - hists['9.3-11.0']['630pkpc_vel_logMstar-geq-9.3-Msun'] / hists['9.3-11.0']['nomask'], Nval)
    
        dXoverdz = dXtot['9.7-11.2'] / dztot['9.7-11.2']
        
    elif numsl == 3:
        cddf_tot     = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        cddf_pos_9p7 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        cddf_vel_9p7 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        #cddf_pos_9p3 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        #cddf_vel_9p3 = linterpsolve(plotx, np.log10((hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.3-Msun']) / dXtotdlogN['9.3-9.7-11.0-11.2']), Nval)
        
        frac_pos_9p7 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        frac_vel_9p7 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        #frac_pos_9p3 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.3-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        #frac_vel_9p3 = linterpsolve(plotx, 1. - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.3-Msun'] / hists['9.3-9.7-11.0-11.2']['nomask'], Nval)
        print(('%s, '*2)%(frac_pos_9p7, frac_vel_9p7,)) #frac_pos_9p3, frac_vel_9p3))
        
        dXoverdz = dXtot['9.3-9.7-11.0-11.2'] / dztot['9.3-9.7-11.0-11.2']
        
        histc_x = bins[:-1]
        histc_pos_9p7 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.7-Msun']
        histc_vel_9p7 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.7-Msun']
        #histc_pos_9p3 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_pos_logMstar-geq-9.3-Msun']
        #histc_vel_9p3 = hists['9.3-9.7-11.0-11.2']['nomask'] - hists['9.3-9.7-11.0-11.2']['630pkpc_vel_logMstar-geq-9.3-Msun']
        histc_all = hists['9.3-9.7-11.0-11.2']['nomask']
        
        histc_pos_9p7 = np.cumsum(histc_pos_9p7[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        histc_vel_9p7 = np.cumsum(histc_vel_9p7[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        #histc_pos_9p3 = np.cumsum(histc_pos_9p3[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        #histc_vel_9p3 = np.cumsum(histc_vel_9p3[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz
        histc_all =  np.cumsum(histc_all[::-1])[::-1] / dXtotdlogN['9.3-9.7-11.0-11.2'] * dXoverdz

        p_N_iso_pos_9p7 = linterpsolve(histc_x, histc_pos_9p7, Nval)
        p_N_iso_vel_9p7 = linterpsolve(histc_x, histc_vel_9p7, Nval)
        #p_N_iso_pos_9p3 = linterpsolve(histc_x, histc_pos_9p3, Nval)
        #p_N_iso_vel_9p3 = linterpsolve(histc_x, histc_vel_9p3, Nval)
        
        Nvals = [15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16.0] + [Nval]
        frac_cumul_pos_9p7 = {Nval: linterpsolve(histc_x, histc_pos_9p7 / histc_all, Nval) for Nval in Nvals}
        frac_cumul_vel_9p7 = {Nval: linterpsolve(histc_x, histc_vel_9p7 / histc_all, Nval) for Nval in Nvals}
        #frac_cumul_pos_9p3 = {Nval: linterpsolve(histc_x, histc_pos_9p3 / histc_all, Nval) for Nval in Nvals}
        #frac_cumul_vel_9p3 = {Nval: linterpsolve(histc_x, histc_vel_9p3 / histc_all, Nval) for Nval in Nvals}
        
    print('at log N(O VII) / cm^-2 = %s, contributions from regions isolated from galaxies at log M*/Msun = 9.3, 9.7 by at least 630 pkpc:'%Nval)
    print('\tlog10 CDDF (total) = %s'%cddf_tot)
    print('\tlog10 CDDF (pos, ge 9.7) = %s'%cddf_pos_9p7)
    #print('\tlog10 CDDF (pos, ge 9.3) = %s'%cddf_pos_9p3)
    print('\tlog10 CDDF (vel, ge 9.7) = %s'%cddf_vel_9p7)
    #print('\tlog10 CDDF (vel, ge 9.3) = %s'%cddf_vel_9p3)
    print('\tCDDF (pos, ge 9.7) / total = %s'%frac_pos_9p7)
    #print('\tCDDF (pos, ge 9.3) / total = %s'%frac_pos_9p3)
    print('\tCDDF (vel, ge 9.7) / total = %s'%frac_vel_9p7)
    #print('\tCDDF (vel, ge 9.3) / total = %s'%frac_vel_9p3)
    print('\tN_expected / dz (pos, ge 9.7) = %s'%p_N_iso_pos_9p7)
    #print('\tN_expected / dz (pos, ge 9.3) = %s'%p_N_iso_pos_9p3)
    print('\tN_expected / dz (vel, ge 9.7) = %s'%p_N_iso_vel_9p7)
    #print('\tN_expected / dz (vel, ge 9.3) = %s'%p_N_iso_vel_9p3)
    print('\t dX / dz = %s'%dXoverdz)
    
    print('\nratios of cumulative CDDF with isolation criteria to cumulative CDDF total above column densities of')
    fillstr = '\t' + '\t'.join(['%.4f'] * len(Nvals))
    print('\t\t' + fillstr%tuple(Nvals))
    print('\tpos 9.7\t' + fillstr%(tuple([frac_cumul_pos_9p7[Nval] for Nval in Nvals])))
    print('\tvel 9.7\t' + fillstr%(tuple([frac_cumul_vel_9p7[Nval] for Nval in Nvals])))
    #print('\tpos 9.3\t' + fillstr%(tuple([frac_cumul_pos_9p3[Nval] for Nval in Nvals])))
    #print('\tvel 9.3\t' + fillstr%(tuple([frac_cumul_vel_9p3[Nval] for Nval in Nvals])))
    
    #lax.axis('off')
    leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', 
                      bbox_to_anchor=(0.01, 0.01), frameon=False)
    leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', 
                      bbox_to_anchor=(0.99, 0.99), frameon=False)
    lax.add_artist(leg1)
    lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    if imgname is not None:
        plt.savefig(imgname, format='pdf', bbox_inches='tight')
    
