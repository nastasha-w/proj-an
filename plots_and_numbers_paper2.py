#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:03:06 2020

@author: wijers

adapted from plothistograms_paper2.py
"""


import numpy as np
import h5py
import pandas as pd
import string
import os

datadir = '/data2/paper2/'

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.legend_handler as mlh
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.patches as mpatch

import eagle_constants_and_units as c #only use for physical constants and unit conversion!
import ion_header as ionh
import cosmo_utils as cu
import ion_line_data as ild
import plot_utils as pu


fontsize=12
ioncolors = {'o7': 'C3',\
             'o8': 'C0',\
             'o6': 'C2',\
             'ne8': 'C1',\
             'ne9': 'C9',\
             'hneutralssh': 'C6',\
             'fe17': 'C4'}

# neutral hydrogen: indicates DLA cutoff = shoulder of the distribution
approx_breaks = {'o7': 16.0,\
             'o8': 16.0,\
             'o6': 14.3,\
             'ne8': 13.7,\
             'ne9': 15.3,\
             'hneutralssh': 20.3,\
             'fe17': 15.0}

# at max. tabulated density, range where ion frac. is >= 10% of the maximum at max density
Tranges_CIE = {'o6':   (5.3, 5.8),\
               'o7':   (5.4, 6.5),\
               'o8':   (6.1, 6.8),\
               'ne8':  (5.6, 6.1),\
               'ne9':  (5.7, 6.8),\
               'fe17': (6.3, 7.0),\
               }
rho_to_nh = 0.752 / (c.atomw_H * c.u)

#retrieved with mc.getcosmopars
cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 2.220446049250313e-16}
cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}
logrhob_av_ea_28 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
logrhob_av_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )
logrhoc_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars_ea_27['z'], cosmopars=cosmopars_ea_27)**2)


mass_edges_standard = (11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0)
def add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
             orientation='vertical', clabel=None, fontsize=fontsize, aspect=10.):
    '''
    returns color bar object, color dictionary (keys: lower mass edges)
    '''
    massedges = np.array(massedges)
    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
    keys = sorted(massedges)
    colors = {keys[i]: clist[i] for i in range(len(keys))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation=orientation)
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(aspect)
    
    return cbar, colors


def T200c_hot(M200c, cosmopars):
    # checked against notes from Joop's lecture: agrees pretty well at z=0, not at z=9 (proabably a halo mass definition issue)
    _M200c = np.copy(M200c)
    _M200c *= c.solar_mass # to cgs
    rhoc = (3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars['z'], cosmopars=cosmopars)**2) # Hubble(z) will assume an EAGLE cosmology
    mu = 0.59 # about right for ionised (hot) gas, primordial
    R200c = (_M200c / (200. * rhoc))**(1./3.)
    return (mu * c.protonmass) / (3. * c.boltzmann) * c.gravity * _M200c / R200c

def R200c_pkpc(M200c, cosmopars):
    '''
    M200c: solar masses
    '''
    M200c *= c.solar_mass # to cgs
    rhoc = (3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars['z'], cosmopars=cosmopars)**2) # Hubble(z) will assume an EAGLE cosmology
    R200c = (M200c / (200. * rhoc))**(1./3.)
    return R200c / c.cm_per_mpc * 1e3 


def readin_radprof(infodct, yvals_label_ion,\
                   labels=None, ions_perlabel=None, ytype='perc',\
                   datadir=datadir, binset=0, units='R200c'):
    '''
    infodct:
        {label: <read-in data>}
        <read-in data>: {'filenames': {ion: filename},\
                         'setnames': names of the galsettags in the files to 
                                     match
                         'setfills': None or 
                                     {setname: string/string tuple to fill into
                                                filenames for different sets},\
                         }
         if using setfills, the fills and filenames should work with 
         string.format, kwargs
    yvals_label_ion:
        {label: {ion: yvals list}}
        yvals list: list of y values to use: percentile points or minimum 
        log10 / cm^-2 column densities
    labels: which labels to use (infodct keys; None -> all of them)
    ions_perlabel: {label: list of ions}
        ions to use for each label
    ytype: 'perc' (percentiles) or 'fcov' (covering fractions)

    returns dictionaries:
        yvals: {label: {ion: {setname: yvals array (cell indices)}}}
        bins:  {label: {ion: {setname: radial bins array (edge indices)}}}
        numgals: {label: {ion: {setname: size of galaxy sample}}}
        bin edges will have +- infinity already adjusted to finite values
    '''
    
    yvals = {}
    bins = {}
    numgals = {}
    
    readpath_base = '{un}_bins/binset_{bs}/{yt}_{val}'.format(un=units,\
                     bs=binset, yt=ytype, val='{}')
    readpath_bins = '{un}_bins/binset_{bs}/bins'.format(un=units,\
                     bs=binset)
    if labels is None:
        labels = list(infodct.keys())
        
    for var in labels:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        
        if ions_perlabel is None:
            ions = list(infodct[var].keys())
        else:
            ions = ions_perlabel[var]
            
        for ion in ions:
            print('Reading in data for ion {ion}'.format(ion=ion))
            filename = infodct[var]['filenames'][ion]
            goaltags = infodct[var]['setnames']
            setfills = infodct[var]['setfills']
            
            #_units   = techvars[var]['units']
            if ion not in filename:
                raise RuntimeError('File {fn} attributed to ion {ion}, mismatch'.format(fn=filename, ion=ion))
            
            if setfills is None:
                with h5py.File(datadir + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    
                    for galset in galsets:
                        ex = True
                        for val in yvals_label_ion[var][ion]:
                            readpath = readpath_base.format(val)
                            try:
                                np.array(fi[galset + '/' + readpath])
                            except KeyError:
                                ex = False
                                break
                        if ex:
                            tags[fi[galset].attrs['seltag'].decode()] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())

                    if len(tags_unread) > 0:
                        print('For file {fn}, missed the following tags:\n\t{st}'.format(fn=filename, st=tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['{}/{}'.format(tags[tag], readpath_base.format(val))]) for val in yvals_label_ion[var][ion]}
                        numgals[var][ion][tag] = len(np.array(fi['{}/galaxyid'.format(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = datadir + filename.format(**fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in yvals_label_ion[var][ion]:
                                readpath = readpath_base.format(val)
                                try:
                                     np.array(fi[galset + '/' + readpath])
                                except KeyError:
                                    ex = False
                                    break
                            
                            if ex:
                                tags[fi[galset].attrs['seltag']] = galset
                            
                        #tags_toread = {tag} &  set(tags.keys())
                        tags_unread = {tag} - set(tags.keys())
                        #print(goaltags)
                        #print(tags.keys())
                        if len(tags_unread) > 0:
                            print('For file {fn}, missed the following tags:\n\t{st}'.format(fn=filename, st=tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpath_base.format(val))]) for val in yvals_label_ion[var][ion]}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))