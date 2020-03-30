#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:49:22 2019

@author: wijers
"""

import numpy as np
import make_maps_opts_locs as ol
import h5py
import pandas as pd
import string
import os
import scipy.integrate as si
import scipy.interpolate as sint

ndir = ol.ndir
mdir = '/net/luttero/data2/imgs/CGM/misc_start/' # luttero location
pdir = ol.pdir

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

import makecddfs as mc
import eagle_constants_and_units as c #only use for physical constants and unit conversion!
import ion_header as ionh
import make_maps_v3_master as m3 # for ion balances
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

### find some rough FoF radius to R200c conversions
def get_FoF_to_200c():
    '''
    rough numerical solver
    '''
    Mhvals = 10**np.arange(11., 14.6, 0.5) * c.solar_mass
    cosmopars = cosmopars_ea_27
    Rh = cu.Rhalo(Mhvals, cosmopars=cosmopars)
        
    rho_fof_edge = (1. / 0.2)**3 * cu.rhom(cosmopars['z'], cosmopars=cosmopars)
    print(rho_fof_edge)
    
    rvals = Rh[:, np.newaxis] * np.arange(0., 5., 0.005)[np.newaxis, :]
    rprof = cu.rho_NFW(rvals, Mhvals[:, np.newaxis], delta=200, ref='rhocrit', z=0., cosmopars=cosmopars, c='Schaller15')
    
    rvals *= (1./ Rh)[:, np.newaxis]
    sols = []
    for Mhi in range(rprof.shape[0]):
        rsol = linterpsolve(rprof[Mhi, :], rvals[Mhi, :], rho_fof_edge)
        sols.append(rsol)
   
        color='C%i'%(Mhi%10)
        plt.plot(rvals[Mhi, :], rprof[Mhi, :], color=color, label='%.1f'%(np.log10(Mhvals[Mhi] / c.solar_mass)))
        plt.axvline(rsol, linestyle='dashed', color=color)
        
    plt.axhline(rho_fof_edge, linestyle='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    return sols, np.log10(Mhvals / c.solar_mass)

def get_M200c_to_500c():
    '''
    rough numerical solver
    '''
    Mhvals = 10**np.arange(11., 14.6, 0.5) * c.solar_mass
    cosmopars = cosmopars_ea_27
    Rh = cu.Rhalo(Mhvals, cosmopars=cosmopars)
        
    rvals = Rh[:, np.newaxis] * np.arange(0., 5., 0.005)[np.newaxis, :]
    rprof = cu.rho_NFW(rvals, Mhvals[:, np.newaxis], delta=200, ref='rhocrit', z=0, cosmopars=cosmopars, c='Schaller15')
    rvals_cen = rvals[:, :-1] + 0.5 * np.diff(rvals, axis=1)
    rhovals_cen = rprof[:, :-1] + 0.5 * np.diff(rprof, axis=1)
    rhovals_cen[:, 0] = rprof[:, 1] # central value is np.inf
    Mencl = np.cumsum( 4. * np.pi * np.diff(rvals, axis=1) * rvals_cen**2 * rhovals_cen, axis=1)
    Rhoencl = Mencl / ( 4. * np.pi / 3. * rvals[:, 1:]**3)
    
    rhotarget = 500. *  cu.rhocrit(0., cosmopars=cosmopars)
    rvals *= (1./ Rh)[:, np.newaxis]
    sols = []
    for Mhi in range(rprof.shape[0]):
        msol = linterpsolve(Rhoencl[Mhi, :], Mencl[Mhi, :], rhotarget)
        sols.append(np.log10(msol / c.solar_mass))
   
        color='C%i'%(Mhi%10)
        plt.plot(Rhoencl[Mhi, :], Mencl[Mhi, :], color=color, label='%.1f'%(np.log10(Mhvals[Mhi] / c.solar_mass)))
        plt.axhline(msol, linestyle='dashed', color=color)

    plt.axhline(rhotarget, linestyle='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    return sols, np.log10(Mhvals / c.solar_mass)
    
    
def setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False, left=True):
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=right, top=top, left=left, axis='both', which='both', color=color,\
                   labelleft=labelleft, labeltop=labeltop, labelbottom=labelbottom, labelright=labelright)

def checksubdct_equal(dct):
    keys = list(dct.keys())
    if not np.all(np.array([set(list(dct[key].keys())) == set(list(dct[keys[0]].keys())) for key in keys])):
        print('Keys do not match')
        return False
    if not np.all(np.array([np.all(np.array([np.all(dct[keys[0]][sub] == dct[key][sub]) for sub in dct[keys[0]].keys() ])) for key in keys])):
        print('Entries do not match')
        return False
    return True

class HandlerDashedLines(mlh.HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = ((height) / (numlines + 1)) * np.ones(xdata.shape, float)
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = mlines.Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[0] is not None:
                # seem to come out twice the input size when using dashes[1] -> fix
                legline.set_dashes([_d *0.5 for _d in dashes[1]])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def format_selection(dct, decf=1, lenf=4):
    '''
    key = thing selected by
    item = array
    (key, *item) matches selection formatting in mask generation
    '''
    fmt = '%' + str(lenf) +  '.' + str(decf) + 'f'
    
    out = []
    keys = dct.keys()
    keys.sort() # standardize order (exact ordering is arbitrary)
    for key in keys:
        if 'M200c_Msun' in key:
            minval = np.log10(dct[key][0])
            maxval = np.log10(dct[key][1])
            if np.isnan(minval) and np.isnan(maxval):
                pass
            elif np.isnan(minval):
                out.append(r'$\log_{10} \, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}] <$ %s'%(fmt%maxval))            
            elif np.isnan(maxval):
                out.append(r'%s $\leq \log_{10} \, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}]$'%(fmt%minval))
            else:
                out.append(r'%s $\leq \log_{10} \, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}] <$ %s'%(fmt%minval, fmt%maxval))
        else:
            raise NotImplementedError('Formatting for selection criterion %s is not yet implemented in format_selection'%key)
    return out
    
def plotcddfsplits_byhaloprops_old(filename, imgname=None, sortby='selection/included/M200c_Msun_set0', fontsize=fontsize):
    '''
    assumes file is structured like masked histogram outputs, using only one
    dimension
    and that the total path length used is the number of pixels in the original
    maps (from the input_filenames) times dX derived from the full box depths;
    essentially, it assumes no missing slices
    '''
    
    if '/' not in filename:
        filename = ol.pdir + filename
    if filename[-5:] != '.hdf5':
        filename = filename + '.hdf5'
        
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
            
    with h5py.File(filename, 'r') as fi:
        bins = np.array(fi['bins/axis_0'])
        # handle +- infinity edges for plotting; should be outside the plot range anyway
        if bins[0] == -np.inf:
            bins[0] = -100.
        if bins[-1] == np.inf:
            bins[-1] = 100.
        
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
        
        try:
            masks = fi['masks'].keys()   
            if sortby is not None:
                masks.sort(key=lambda ms: tuple(np.array(fi['masks/%s/%s'%(ms, sortby)])))
        except KeyError: 
            raise KeyError('File contains no masked histograms')
            masks = []
            
        maskpars = {mask: {key: item for (key, item) in fi['masks/%s/Header/parameters'%mask].attrs.items()} for mask in masks}
        mcatpars = {mask: {key: item for (key, item) in fi['masks/%s/Header'%mask].attrs.items()} for mask in masks}
        if not checksubdct_equal(maskpars):
            maskpars_equal = False
            raise Warning('The mask parameters for the different masks do not match')
        else:
            maskpars_equal = True
        if not checksubdct_equal(mcatpars):
            mcatpars_equal = False
            raise Warning('The halo catalogues for the different masks do not match') 
        else:
            mcatpars_equal = True
        maskpars = maskpars[masks[0]]
        mcatpars = mcatpars[masks[0]]
        masksels_in = {mask: {key: np.array(fi['masks/%s/selection/included/%s'%(mask, key)])\
                              for key in fi['masks/%s/selection/included'%(mask)].keys()} \
                             if 'masks/%s/selection/included'%(mask) in fi else\
                             {} \
                       for mask in masks}
        masksels_ex = {mask: {key: np.array(fi['masks/%s/selection/excluded/%s'%(mask, key)])\
                              for key in fi['masks/%s/selection/excluded'%(mask)].keys()} \
                             if 'masks/%s/selection/excluded'%(mask) in fi else\
                             {} \
                       for mask in masks}
        
        hists = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
        try:
            hists.update({'nomask': np.array(fi['nomask/hist'])})
            incltotal = True
        except KeyError:
            incltotal = False
        fcovs = {mask:fi[mask].attrs['covfrac'] for mask in hists.keys()}
        
        cosmopars = {key: item for (key, item) in fi['masks/%s/Header/cosmopars'%masks[0]].attrs.items()}
        dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
        dXtotdlogN = dXtot * np.diff(bins)

    
    maskpars_doc = ['%s:\t%s'%(key, maskpars[key]) for key in maskpars.keys()]
    maskpars_doc = '\n'.join(maskpars_doc)
    if not maskpars_equal:
        maskpars_doc = 'for mask ...%s\n'%(masks[0][-30:]) + maskpars_doc
    mcatpars_doc = ['%s: %s'%(key, mcatpars[key]) for key in mcatpars.keys()]
    mcatpars_doc = '\n'.join(mcatpars_doc)
    if not mcatpars_equal:
        mcatpars_doc = 'for catalogue ...%s\n'%(masks[0][-30:]) + mcatpars_doc
    
    maskdoc_gen = mcatpars_doc + '\n' + maskpars_doc
    
    masklabels_in = {mask: ', '.join(format_selection(masksels_in[mask], decf=1)) for mask in masks}
    masklabels_in = {mask: 'incl. ' + masklabels_in[mask] if masklabels_in[mask] != '' else '' for mask in masks}
    masklabels_ex = {mask: ', '.join(format_selection(masksels_ex[mask], decf=1)) for mask in masks}
    masklabels_ex = {mask: 'excl. ' + masklabels_ex[mask] if masklabels_in[mask] != '' else '' for mask in masks}
    
    labels = {mask: ';\t'.join([masklabels_in[mask], masklabels_ex[mask], r'$f_{\mathrm{cov}} = %.1e$'%(fcovs[mask])]) for mask in masks}
    
    fig = plt.figure(figsize=(8.5, 7.5))
    grid = gsp.GridSpec(2, 2, hspace=0.25, wspace=0.0, height_ratios=[4., 4.], width_ratios=[6., 2.])
    ax = fig.add_subplot(grid[0, 0])
    tax = fig.add_subplot(grid[0, 1])
    lax = fig.add_subplot(grid[1, :])
    if ion[0] == 'h':
        ax.set_xlim(12.5, 23.5)
        ax.set_ylim(-6., 1.65)
    else:
        ax.set_xlim(12., 17.5)
        ax.set_ylim(-6., 1.4)
    
    cmap = cm.get_cmap('rainbow')
    colors = cmap(np.arange(len(masks)) / float(max([len(masks) - 1, 1])))

    tax.axis('off')
    tax.text(0.05,0.95, maskdoc_gen.expandtabs(), fontsize=fontsize, transform=tax.transAxes, verticalalignment='top', horizontalalignment='left', wrap=True) # , bbox=dict(facecolor='white',alpha=0.3)
    
    ax.set_xlabel(r'$\log_{10}\, \mathrm{N}_\mathrm{%s} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion, mathmode=True)), fontsize=fontsize)
    ax.set_ylabel(r'$\log_{10} \left(\, \partial^2 n \,/\, \partial \log_{10} \mathrm{N} \, \partial X  \,\right)$', fontsize=fontsize)
    setticks(ax, fontsize=fontsize)
    
    plotx = bins[:-1] + 0.5 * np.diff(bins)
    for mi in range(len(masks)):
        mask = masks[mi]
        #ax.step(bins[:-1], np.log10(hists[mask] / dXtotdlogN), where='post', color=colors[mi], label=labels[mask].expandtabs())
        ax.plot(plotx, np.log10(hists[mask] / dXtotdlogN), color=colors[mi], label=labels[mask].expandtabs())
    if incltotal:
        #ax.step(bins[:-1], np.log10(np.sum(np.array([hists[mas] for mas in masks]), axis=0)/ dXtotdlogN), where='post', color='gray', linestyle='dashed', label='all subsets')
        #ax.step(bins[:-1], np.log10(hists['nomask'] / dXtotdlogN), where='post', color='black', linewidth=2, label='total')
        ax.plot(plotx, np.log10(hists['nomask'] / dXtotdlogN), color='black', linewidth=2, label='total')
        ax.plot(plotx, np.log10(np.sum(np.array([hists[mas] for mas in masks]), axis=0)/ dXtotdlogN), color='gray', linestyle='dashed', label='all subsets')
    lhandles, llabels = ax.get_legend_handles_labels()
    lax.axis('off')
    lax.legend(handles=lhandles, fontsize=fontsize - 1, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    
    if imgname is not None:
        plt.savefig(imgname, format='pdf', bbox_inches='tight')
        


def plot_cddfs(ions, fontsize=fontsize, imgname=None, techvars=[0]):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    
    if imgname is None:
        imgname = 'cddfs_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvars-%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars])))
    if '/' not in imgname:
        imgname = mdir + imgname
    if imgname[-4:] != '.pdf':
        imgname = imgname + '.pdf'


        
    if isinstance(ions, str):
        ions = [ions]
    
    ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    #clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'}
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    linewidths = {0: 2}
    
    linestyles = {0: 'solid'}
    
    alphas = {0: 1.}
    
    masknames1 = ['nomask']
    masknames = masknames1 #{0: {ion: masknames1 for ion in ions}}
    
    #legendnames_techvars = {0: r'$r_{\perp} < R_{\mathrm{200c}}$, excl., cen'}
    
#    panelwidth = 2.5
#    panelheight = 2.
#    legheight = 0.6
#    figwidth = numcols * panelwidth + 0.6 
#    figheight = numcols * panelheight + 0.2 * numcols + legheight
#    fig = plt.figure(figsize=(figwidth, figheight))
#    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.2, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=[panelheight] * numrows + [legheight])
#    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
#    cax  = fig.add_subplot(grid[:numrows, numcols])
#    lax  = fig.add_subplot(grid[numrows, :])
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(5.5, 3.0), gridspec_kw={'wspace': 0.0})
    
    hists = {}
    cosmopars = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        dXtot[var] = {}
        dztot[var] = {}
        dXtotdlogN[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                if _bins[0] == -np.inf:
                    _bins[0] = -100.
                if _bins[-1] == np.inf:
                    _bins[-1] = 100.
                bins[var][ion] = _bins
                
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
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                
                examplemaskdir = fi['masks'].keys()[0]
                examplemask = fi['masks/%s'%(examplemaskdir)].keys()[0]
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                dXtot[var][ion] = mc.getdX(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dztot[var][ion] = mc.getdz(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])

        assert checksubdct_equal(cosmopars[var])
                     
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
   #     raise RuntimeError('total histograms from different files do not match')
        
    
    ax1.set_xlim(12.0, 17.)
    ax1.set_ylim(-5.0, 2.5)
    ax2.set_xlim(12.0, 23.0)
    ax2.set_ylim(-5.0, 2.5)
    
    setticks(ax1, fontsize=fontsize, labelbottom=True, labelleft=True)
    setticks(ax2, fontsize=fontsize, labelbottom=True, labelleft=False)

    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
            
    for ionind in range(len(ions)):
        ion = ions[ionind]
        
        if ion in ['hneutralssh']:
            ax = ax2
        else:
            ax = ax1
            
        #if relative:
        #    ax.text(0.05, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        #else:
        #    ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
            
        for vi in range(len(techvars)):
            plotx = bins[var][ion]
            plotx = plotx[:-1] + 0.5 * np.diff(plotx)
            
            ax.plot(plotx, np.log10((hists[var][ion]['nomask']) / dXtotdlogN[var][ion]), color=ioncolors[ion], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=ild.getnicename(ion, mathmode=False))
            
            ylim = ax.get_ylim()
            if ion == 'o8':
                ls = 'dashed'
            else:
                ls = 'solid'
            ax.axvline(approx_breaks[ion], ylim[0], 0.3 , color=ioncolors[ion], linewidth=1.5, linestyle=ls)
    #lax.axis('off')
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    leg1 = ax2.legend(handles=handles1[:3], fontsize=fontsize, ncol=1, loc='upper right', bbox_to_anchor=(1., 1.), frameon=False)
    leg2 = ax2.legend(handles=handles1[3:] + handles2, fontsize=fontsize, ncol=1, loc='lower left', bbox_to_anchor=(0., 0.), frameon=False)
    ax2.add_artist(leg1)
    ax2.add_artist(leg2)
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')

def save_Tvir_ions(snap=27):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
    
    outdir = '/net/luttero/data2/paper2/'
    outname = 'ionbal_snap{snap}.hdf5'.format(snap=snap)   
    outname = outdir + outname
    
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    if snap == 27:
        cosmopars = cosmopars_ea_27
        
    with h5py.File(outname, 'w') as fo:
        for ion in ions: 
            bal, T, nH = m3.findiontables(ion, cosmopars['z'])
            grp = fo.create_group(ion)
            grp.create_dataset('logTK', data=T)
            grp.create_dataset('lognHcm3', data=nH)
            ds = grp.create_dataset('ionbal', data=bal)
            ds.attrs.create('axis0', np.string_('lognHcm3'))
            ds.attrs.create('axis1', np.string_('logTK'))
        

def plot_Tvir_ions(snap=27, _ioncolors=ioncolors):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
    fontsize = 12
    mdir = '/net/luttero/data2/imgs/CGM/'
    
    if snap == 27:
        cosmopars = cosmopars_ea_27
        logrhob = logrhob_av_ea_27
        logrhoc = logrhoc_ea_27
        #print(logrhob, logrhoc)
    
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] #, 'he2'
    ioncolors = _ioncolors.copy()
    
    #ioncolors.update({'he2': 'darkgoldenrod'})
    Ts = {}
    Tmaxs = {}
    nHs = {}
    bals = {}
    maxfracs = {}
    
    fracv = 0.1
    ciemargin = 1.50
    
    for ion in ions:
        bal, T, nH = m3.findiontables(ion, cosmopars['z'])
        bals[ion] = bal
        nHs[ion] = nH
        Ts[ion] = T
        indmaxfrac = np.argmax(bal[-1, :])
        maxfrac = bal[-1, indmaxfrac]
        Tmax = T[indmaxfrac]
        Tmaxs[ion] = Tmax
        
        xs = find_intercepts(bal[-1, :], T, fracv * maxfrac)
        print('Ion %s has maximum CIE fraction %.3f, at log T[K] = %.1f, %s max range is %s'%(ion, maxfrac, Tmax, fracv, str(xs)))
        maxfracs[ion] = maxfrac
        
    # neutral hydrogen
    Tvals = 10**Ts[ions[0]]
    nHvals = 10**nHs[ions[0]]
    Tgrid = np.array([[T] * len(nHvals) for T in Tvals]).flatten()
    nHgrid = np.array([nHvals] * len(Tvals)).flatten()
    bal = m3.cfh.nHIHmol_over_nH({'Temperature': Tgrid, 'nH': nHgrid}, cosmopars['z'], UVB='HM01', useLSR=False)
    bal = (bal.reshape((len(Tvals), len(nHvals)))).T
    ion = 'hneutralssh'
    bals[ion] = bal
    Ts[ion]  = np.log10(Tvals)
    nHs[ion] = np.log10(nHvals)
   
    maxcol = bal[-1, :]
    maxfrac = np.max(maxcol[np.isfinite(maxcol)])
    maxfracs[ion] = maxfrac
    #print('Ion %s has maximum CIE fraction %.3f'%(ion, maxfrac))
    xs = find_intercepts(maxcol[np.isfinite(maxcol)], np.log10(Tvals)[np.isfinite(maxcol)], fracv * maxfrac)
    print('Ion %s has maximum CIE fraction %.3f, %s max range is %s'%(ion, maxfrac, fracv, str(xs)))
        
    allions = ['hneutralssh'] + ions
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.0))
    ax.set_xlim(-8., -1.5)
    ax.set_ylim(2.5, 9.)
    
    ax.set_ylabel(r'$\log_{10} \, T \; [K]$', fontsize=fontsize)
    ax.set_xlabel(r'$\log_{10} \, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    setticks(ax, fontsize=fontsize, right=False)
    
    axy2 = ax.twinx()
    ylim = ax.get_ylim()
    axy2.set_ylim(*ylim)
    mhalos = np.arange(9.0, 15.1, 0.5)
    Tvals = np.log10(T200c_hot(10**mhalos, cosmopars))
    Tlabels = ['%.1f'%mh for mh in mhalos]
    axy2.set_yticks(Tvals)
    axy2.set_yticklabels(Tlabels)
    setticks(axy2, fontsize=fontsize, left=False, right=True, labelleft=False, labelright=True)
    axy2.minorticks_off()
    axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$', fontsize=fontsize)
    
    ax.axvline(logrhob + np.log10(rho_to_nh), 0., 0.75, color='gray', linestyle='dashed', linewidth=1.5)
    ax.axvline(logrhoc + np.log10(rho_to_nh * 200. * cosmopars['omegab'] / cosmopars['omegam']), 0., 0.75, color='gray', linestyle='solid', linewidth=1.5)
    

    for ion in allions:
        ax.contourf(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], alpha=0.1, linewidths=[3.], levels=[0.1 * maxfracs[ion], 1.])
        ax.contour(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], linewidths=[2.], levels=[0.1 * maxfracs[ion]], linestyles=['solid'])
        if ion != 'hneutralssh':
            ax.axhline(Tmaxs[ion], 0.95, 1., color=ioncolors[ion], linewidth=3.)
            
        bal = bals[ion]
        maxcol = bal[-1, :]
        diffs = bal / maxcol[np.newaxis, :]
        diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal == 0)] = 0.
        diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] = bal[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] / 1e-18
        diffs = np.abs(np.log10(diffs))
            
        mask = bal < 0.6 * fracv * maxfracs[ion] # 0.6 gets the contours to ~ the edges of the ion regions
        diffs[mask] = np.NaN

        ax.contour(nHs[ion], Ts[ion][np.isfinite(maxcol)], (diffs[:, np.isfinite(maxcol)]).T, levels=[np.log10(ciemargin)], linestyles=['solid'], linewidths=[1.], alphas=0.5, colors=ioncolors[ion])
        
    handles = [mlines.Line2D([], [], label=ild.getnicename(ion, mathmode=False), color=ioncolors[ion]) for ion in allions]
    ax.legend(handles=handles, fontsize=fontsize, ncol=3, bbox_to_anchor=(0.0, 1.0), loc='upper left', frameon=False)

    plt.savefig(mdir + 'ionbals_snap{}_HM01_ionizedmu_inclhneutralssh.pdf'.format(snap), format='pdf', bbox_inches='tight')



def plot_cddfsplits(ions, fontsize=fontsize, imgname=None, techvars=[0], relative=True):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    else:
        if relative:
            reltag = '_rel'
        else:
            reltag = '_abs'
        imgname = 'cddfsplits_byhalomass_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvars-%s%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars])), reltag)
        imgname = mdir + imgname
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    cmapname = 'rainbow'
    sumcolor = 'saddlebrown'
    totalcolor = 'black'
    if relative:
        ylabel = r'$\log_{10}$ CDDF / total'
    else:
        ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'}
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    linewidths = {0: 2}
    
    linestyles = {0: 'solid'}
    
    alphas = {0: 1.}
    
    masknames1 = ['nomask',\
                  'logM200c_Msun-9.0-9.5',\
                  'logM200c_Msun-9.5-10.0',\
                  'logM200c_Msun-10.0-10.5',\
                  'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    masknames = masknames1 #{0: {ion: masknames1 for ion in ions}}
    
    legendnames_techvars = {0: r'$r_{\perp} < R_{\mathrm{200c}}$, excl., cen'}
    
    panelwidth = 2.5
    panelheight = 2.
    legheight = 0.6
    fcovticklen = 0.035
    figwidth = numcols * panelwidth + 0.6 
    figheight = numcols * panelheight + 0.2 * numcols + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.2, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    lax  = fig.add_subplot(grid[numrows, :])
    
    
    hists = {}
    cosmopars = {}
    fcovs = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        fcovs[var] = {}
        dXtot[var] = {}
        dztot[var] = {}
        dXtotdlogN[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                if _bins[0] == -np.inf:
                    _bins[0] = -100.
                if _bins[-1] == np.inf:
                    _bins[-1] = 100.
                bins[var][ion] = _bins
                
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
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                fcovs[var][ion] = {mask: fi[mask].attrs['covfrac'] for mask in hists[var][ion].keys()}
                

                examplemaskdir = fi['masks'].keys()[0]
                examplemask = fi['masks/%s'%(examplemaskdir)].keys()[0]
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                dXtot[var][ion] = mc.getdX(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dztot[var][ion] = mc.getdz(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])

        assert checksubdct_equal(cosmopars[var])
                     
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
   #     raise RuntimeError('total histograms from different files do not match')
        
    massranges = [(float(i) for i in name.split('-')[-2:]) if name != 'nomask' else None for name in masknames]
    massranges.remove(None)
    massedges = sorted(list(set([val for rng in massranges for val in rng])))
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels = {name: float(name.split('-')[1]) + 0.5 * np.average(np.diff(massedges)) for name in masknames[1:]}
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks = sorted(masslabels.keys(), key=masslabels.__getitem__)
    colors = {_masks[i]: clist[i] for i in range(len(_masks))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    

    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        if relative:
            ax.set_ylim(-4.5, 0.1)
        else:
            ax.set_ylim(-6.0, 2.5)
        
        labelx = yi == numrows - 1
        labely = xi == 0
        setticks(ax, fontsize=fontsize, labelbottom=True, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        if relative:
            ax.text(0.05, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        else:
            ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
            
        for vi in range(len(techvars)):
            masks = sorted(masslabels.keys(), key=masslabels.__getitem__)
            for mi in range(len(masks)):
                plotx = bins[var][ion]
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)
                mask = masks[mi]
                if relative:
                    ax.plot(plotx, np.log10((hists[var][ion][mask]) / hists[var][ion]['nomask']), color='black', linestyle=linestyles[var], linewidth=linewidths[var] + 0.5, alpha=alphas[var], label=masslabels[mask])
                    ax.plot(plotx, np.log10((hists[var][ion][mask]) / hists[var][ion]['nomask']), color=colors[mask], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels[mask])
                    ax.axhline(np.log10(fcovs[var][ion][mask]), 0., fcovticklen, color=colors[mask], linestyle=linestyles[var], linewidth=max(linewidths[var] - 1., 0.5), alpha=alphas[var], zorder=-1)
                    ax.axhline(np.log10(fcovs[var][ion][mask]), 1. - fcovticklen, 1., color=colors[mask], linestyle=linestyles[var], linewidth=max(linewidths[var] - 1., 0.5), alpha=alphas[var], zorder=-1)
                else:
                    ax.plot(plotx, np.log10((hists[var][ion][mask]) / dXtotdlogN[var][ion]), color=colors[mask], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels[mask])
            halosum = np.sum(np.array([hists[var][ion][ms] for ms in masks]), axis=0)
            fcovsum = np.sum(np.array([fcovs[var][ion][ms] for ms in masks]), axis=0)
            if relative:
                ax.plot(plotx, np.log10(halosum / hists[var][ion]['nomask']), color=sumcolor, linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label='halo sum')
                ax.axhline(np.log10(fcovsum), 0., fcovticklen, color=sumcolor, linestyle=linestyles[var], linewidth=max(linewidths[var] - 1., 0.5), alpha=alphas[var], zorder=-1)
                ax.axhline(np.log10(fcovsum), 1. - fcovticklen, 1., color=sumcolor, linestyle=linestyles[var], linewidth=max(linewidths[var] - 1., 0.5), alpha=alphas[var], zorder=-1)
            else:
                ax.plot(plotx, np.log10(halosum / dXtotdlogN[var][ion]), color=sumcolor, linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label='halo sum')
                ax.plot(plotx, np.log10((hists[var][ion]['nomask']) / dXtotdlogN[var][ion]), color=totalcolor, linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels[mask])
        if relative:
            ax.axhline(0., color=totalcolor, linestyle='solid', linewidth=1.5, alpha=0.7)
            ylim = ax.get_ylim()
            ax.axvline(approx_breaks[ion], ylim[0], 0.2 , color='gray', linewidth=1.5) # ioncolors[ion]
    #lax.axis('off')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars:
        # set up the proxy artist
        subcols = list(clist) + [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var], linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    sumhandles = [mlines.Line2D([], [], color=sumcolor, linestyle='solid', label='all halos', linewidth=2.),\
                  mlines.Line2D([], [], color=totalcolor, linestyle='solid', label='total', linewidth=2.)]
    sumlabels = ['all halos', 'total']
    lax.legend(lcs + sumhandles, [legendnames_techvars[var] for var in techvars] + sumlabels,\
               handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2 * numcols, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    
    
    

def plot_radprof(ions, fontsize=fontsize, imgname=None, techvars_touse=[0], units='R200c', ytype='perc', yvals_toplot=[10., 50., 90.], highlightcrit={'techvars': [0], 'Mmin': [11.0, 12.0, 13.0]}, printnumgals=False):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    
    mdir = '/net/luttero/data2/imgs/CGM/radprof/'
    
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    else:
        imgname = 'radprof_byhalomass_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvars-%s_units-%s_%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars_touse])), units, ytype)
        imgname = mdir + imgname
        
        if ytype=='perc' and 50.0 not in yvals_toplot:
            imgname = imgname[:-4] + '_yvals-%s'%('-'.join([str(val) for val in yvals_toplot])) + '.pdf'
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    cmapname = 'rainbow'
    hatches = ['\\', '/', '|', 'o', '+', '*', '-', 'x', '.']
    #sumcolor = 'saddlebrown'
    totalcolor = 'black'
    shading_alpha = 0.35 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    if units == 'pkpc':
        xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    elif units == 'R200c':
        xlabel = r'$r_{\perp} \; [\mathrm{R}_{\mathrm{200c}}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}
    
    ion_filedct_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}

    # use only 100 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}

    ion_filedct_subsample_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}
    # use only 1000 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample2_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                 }
        
    # define used mass ranges
    Mh_edges = np.array([9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.])
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]

    galsetnames_massonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    galsetnames_offedges = {name + '_Z_off-edge-by-R200c':  galsetnames_massonly[name] for name in galsetnames_massonly.keys()}
    
    techvars = {0: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_massonly.keys()},\
                1: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_offedges.keys()},\
                2: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_massonly.keys()},\
                3: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_offedges.keys()},\
                4: {'filenames': ion_filedct_subsample_1sl, 'setnames': galsetnames_massonly.keys()},\
                5: {'filenames': ion_filedct_subsample_2sl, 'setnames': galsetnames_massonly.keys()},\
                6: {'filenames': ion_filedct_subsample2_1sl, 'setnames': galsetnames_massonly.keys()}}
    
    linewidths = {0: 1.5,\
                  1: 1.5,\
                  2: 2.5,\
                  3: 2.5,\
                  4: 1.5,\
                  5: 2.5,\
                  6: 1.5}
    
    
    linestyles = {0: 'solid',\
                  1: 'dashed',\
                  2: 'solid',\
                  3: 'dotted',\
                  4: 'solid',\
                  5: 'solid',\
                  6: 'solid'}
    
    alphas = {0: 1.,\
              1: 1.,\
              2: 1.,\
              3: 1.,\
              4: 0.4,\
              5: 0.4,\
              6: 0.6}
    
    legendnames_techvars = {0: r'1 sl., all',\
                            1: r'1 sl., off-edge',\
                            2: r'2 sl., all',\
                            3: r'2 sl., off-edge',\
                            4: r'1 sl., 100',\
                            5: r'2 sl., 100',\
                            6: r'1 sl., 1000'}
    
    readpaths = {val: '%s_bins/binset_0/%s_%s'%(units, ytype, val) for val in yvals_toplot}
    readpath_bins = '/'.join((readpaths[readpaths.keys()[0]]).split('/')[:-1]) + '/bin_edges'
    print(readpaths)
    panelwidth = 2.5
    panelheight = 2.
    legheight = 1.2
    if ytype == 'perc':
        wspace = 0.2
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + 0.6 
    figheight = numcols * panelheight + 0.2 * numcols + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    lax  = fig.add_subplot(grid[numrows, :])
    
    
    yvals = {}
    #cosmopars = {}
    #fcovs = {}
    #dXtot = {}
    #dztot = {}
    #dXtotdlogN = {}
    bins = {}
    numgals = {}
    
    for var in techvars_touse:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var]['filenames'][ion]
            goaltags = techvars[var]['setnames']
            
            if ion not in filename:
                raise RuntimeError('File %s attributed to ion %s, mismatch'%(filename, ion))
            
            with h5py.File(ol.pdir + 'radprof/' + filename, 'r') as fi:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                galsets = fi.keys()
                tags = {} 
                for galset in galsets:
                    ex = True
                    for val in readpaths.keys():
                        try:
                            temp = np.array(fi[galset + '/' + readpaths[val]])
                        except KeyError:
                            ex = False
                            break
                    
                    if ex:
                        tags[fi[galset].attrs['seltag']] = galset
                    
                tags_toread = set(goaltags) &  set(tags.keys())
                tags_unread = set(goaltags) - set(tags.keys())
                #print(goaltags)
                #print(tags.keys())
                if len(tags_unread) > 0:
                    print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                
                for tag in tags_toread:
                    _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                    # handle +- infinity edges for plotting; should be outside the plot range anyway
                    if _bins[0] == -np.inf:
                        _bins[0] = -100.
                    if _bins[-1] == np.inf:
                        _bins[-1] = 100.
                    bins[var][ion][tag] = _bins
                    
                    # extract number of pixels from the input filename, using naming system of make_maps
                    
                    yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                    numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
                    
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
    #    raise RuntimeError('total histograms from different files do not match')
    if printnumgals:
       print('tech vars: 0 = 1 slice, all, 1 = 1 slice, off-edge, 2 = 2 slices, all, 3 = 2 slices, off-edge')
       print('\n')
       
       for ion in ions:
           for var in techvars_touse:
               tags = techvars[var]['setnames']
               if var in [0, 2]:
                   tags = sorted(tags, key=galsetnames_massonly.__getitem__)
               else:
                   tags = sorted(tags, key=galsetnames_offedges.__getitem__)
               print('%s, var %s:'%(ion, var))
               print('\n'.join(['%s\t%s'%(tag, numgals[var][ion][tag]) for tag in tags]))
               print('\n')
       return numgals
        
    massranges = [sel[1:] for sel in Mh_sels]
    #print(massranges)
    massedges = sorted(list(set([np.log10(val) for rng in massranges for val in rng])))
    #print(massedges)
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels1 = {name: tuple(np.log10(np.array(galsetnames_massonly[name][1:]))) for name in galsetnames_massonly.keys()}
    masslabels2 = {name: tuple(np.log10(np.array(galsetnames_offedges[name][1:]))) for name in galsetnames_offedges.keys()}
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks1 = sorted(masslabels1.keys(), key=masslabels1.__getitem__)
    colors = {_masks1[i]: clist[i] for i in range(len(_masks1))}
    _masks2 = sorted(masslabels2.keys(), key=masslabels2.__getitem__)
    colors.update({_masks2[i]: clist[i] for i in range(len(_masks2))})
    #del _masks
    masslabels_all = masslabels1
    masslabels_all.update(masslabels2)
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    

    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ytype == 'perc':
            if ion[0] == 'h':
                ax.set_ylim(12.0, 21.0)
            #else:
            #    ax.set_ylim(11.5, 17.0)
            elif ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.25, 16.25)
            elif ion == 'o6':
                ax.set_ylim(11., 15.)
            elif ion == 'ne9':
                ax.set_ylim(12., 16.)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(0., 1.)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        

        ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        
        hatchind = 0
        for vi in range(len(techvars_touse)):
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                if highlightcrit is not None: #highlightcrit={'techvars': [0], 'Mmin': [10.0, 12.0, 14.0]}
                    matched = True
                    if 'techvars' in highlightcrit.keys():
                        matched &= var in highlightcrit['techvars']
                    if 'Mmin' in highlightcrit.keys():
                        matched &= np.min(np.abs(masslabels_all[tag][0] - np.array(highlightcrit['Mmin']))) <= 0.01
                    if matched:
                        yvals_toplot_temp = yvals_toplot
                    else:
                        yvals_toplot_temp = [yvals_toplot[0]] if len(yvals_toplot) == 1 else [yvals_toplot[1]]
                else:
                    yvals_toplot_temp = yvals_toplot
                
                
                if len(yvals_toplot_temp) == 3:
                    yval = yvals_toplot_temp[0]
                    try:                      
                        ploty1 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    yval = yvals_toplot_temp[2]
                    try:                      
                        ploty2 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    # according to stackexchange, this is the only way to set the hatch color in matplotlib 2.0.0 (quasar); does require the same color for all hatches...
                    plt.rcParams['hatch.color'] = (0.5, 0.5, 0.5, alphas[var] * shading_alpha,) #mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,)
                    ax.fill_between(plotx, ploty1, ploty2, color=(0., 0., 0., 0.), hatch=hatches[hatchind], facecolor=mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,), edgecolor='face', linewidth=0.0)
                    ax.fill_between(plotx, ploty1, ploty2, color=colors[tag], alpha=alphas[var] * shading_alpha, label=masslabels_all[tag])
                    
                    hatchind += 1
                    yvals_toplot_temp = [yvals_toplot_temp[1]]
                    
                if len(yvals_toplot_temp) == 1:
                    yval = yvals_toplot_temp[0]
                    try:
                        ploty = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val))
                        continue
                    if yval == 50.0: # only highlight the medians
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
                    else:
                        patheff = []
                    ax.plot(plotx, ploty, color=colors[tag], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels_all[tag], path_effects=patheff)
                
        
        if ytype == 'perc':
            #ax.axhline(0., color=totalcolor, linestyle='solid', linewidth=1.5, alpha=0.7)
            #xlim = ax.get_xlim()
            ax.axhline(approx_breaks[ion], 1. - 0.2, 1., color='gray', linewidth=1.5) # ioncolors[ion]
    #lax.axis('off')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = list(clist) #+ [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var], linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    #sumhandles = [mlines.Line2D([], [], color=sumcolor, linestyle='solid', label='all halos', linewidth=2.),\
    #              mlines.Line2D([], [], color=totalcolor, linestyle='solid', label='total', linewidth=2.)]
    #sumlabels = ['all halos', 'total']
    lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse], handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2 * numcols, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    


def plotfofonlycddfs(ion, relative=False):
    '''
    Note: all haloes line with masks (brown dashed) is for all haloes with 
          M200c > 10^9 Msun, while the solid line is for all FoF+200c gas at 
          any M200c
    '''
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    outname = mdir + 'split_FoF-M200c_proj_%s'%ion
    if relative:
        outname = outname + '_rel'
    outname = outname + '.pdf'
    
    ions = ['o7', 'o8', 'o6', 'ne8', 'fe17', 'ne9', 'hneutralssh']
    medges = np.arange(9., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_all = {key: [ol.pdir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    
    if ion not in ions:
        raise ValueError('Ion must be one of %s'%ions)
        
    filenames_this = filenames_all[ion]
    masses_proj = ['none'] + list(medges)
    filedct = {masses_proj[i]: filenames_this[i] for i in range(len(filenames_this))} 
    
    masknames =  ['nomask',\
                  'logM200c_Msun-9.0-9.5',\
                  'logM200c_Msun-9.5-10.0',\
                  'logM200c_Msun-10.0-10.5',\
                  'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for pmass in masses_proj:
        dct_fofcddf[pmass] = {}
        try:
            with h5py.File(filedct[pmass]) as fi:
                try:
                    bins = np.array(fi['bins/axis_0'])
                except KeyError as err:
                    print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                    raise err
                    
                dct_fofcddf[pmass]['bins'] = bins
                
                inname = np.array(fi['input_filenames'])[0]
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                numpix_1sl.remove(None)
                numpix_1sl = int(list(numpix_1sl)[0][:-3])
                print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                for mmass in masses_proj[1:]:
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    mask_examples = {key: item for (key, item) in grp.attrs.items()}
                    del mask_examples['covfrac']
                    example_key = mask_examples.keys()[0] # 'mask_<slice center>'
                    example_mask = mask_examples[example_key] # '<dir path><mask file name>'
                    path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                    #print(path)
                    cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
        
                    dct_fofcddf[pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
                # use cosmopars from the last read mask
                mmass = 'none'
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                dXtotdlogN = dXtot * np.diff(bins)
                dct_fofcddf[pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
            
        except IOError as err:
            print('Failed to read in %s; stated error:'%filedct[pmass])
            print(err)
         
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'}
    
    file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
    dct_totcddf = {}
    with h5py.File(file_allproj) as fi:
        try:
            bins = np.array(fi['bins/axis_0'])
        except KeyError as err:
            print('While trying to load bins in file %s\n:'%(file_allproj))
            raise err
            
        dct_totcddf['bins'] = bins
        
        inname = np.array(fi['input_filenames'])[0]
        inname = inname.split('/')[-1] # throw out directory path
        parts = inname.split('_')

        numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
        numpix_1sl.remove(None)
        numpix_1sl = int(list(numpix_1sl)[0][:-3])
        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
        
        for mmass in masses_proj[1:]:
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            mask_examples = {key: item for (key, item) in grp.attrs.items()}
            del mask_examples['covfrac']
            example_key = mask_examples.keys()[0] # 'mask_<slice center>'
            example_mask = mask_examples[example_key] # '<dir path><mask file name>'
            path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
            cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
            dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
        
            dct_totcddf[mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
        # use cosmopars from the last read mask
        mmass = 'none'
        grp = fi[maskdct[mmass]]
        hist = np.array(grp['hist'])
        covfrac = grp.attrs['covfrac']
        # recover cosmopars:
        dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
        dXtotdlogN = dXtot * np.diff(bins)
        dct_totcddf[mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    cmapname = 'rainbow'
    #sumcolor = 'saddlebrown'
    #totalcolor = 'black'
    if relative:
        ylabel = r'$\log_{10}$ CDDF / total'
    else:
        ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'masks for haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) + [np.inf]
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels = {name: name + 0.5 * np.average(np.diff(massedges)) for name in masses_proj[1:]}
    
    numcols = 4
    numrows = 4
    panelwidth = 2.5
    panelheight = 2.
    legheight = 1.
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = False
        numrows_fig = numrows + 1
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols - 1
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numcols * panelheight + 0.2 * numcols + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=0.0, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(masses_proj) + 1)]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    
    
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks = sorted(masslabels.keys(), key=masslabels.__getitem__)
    colors = {_masks[i]: clist[i] for i in range(len(_masks))}
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    linewidth = 2.
    alpha = 1.
    
    for massind in range(len(masses_proj) + 1):
        xi = massind % numcols
        yi = massind // numcols
        if massind == 0:
            pmass = masses_proj[massind]
        elif massind == 1:
            pmass = 'all halos'
        else:
            pmass = masses_proj[massind - 1]
        ax = axes[massind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        if relative:
            ax.set_ylim(-4.5, 1.)
        else:
            ax.set_ylim(-6.0, 2.5)
        
        labelx = yi == numrows - 1 or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.0, foreground="b"), mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"), mppe.Normal()]
        
        if pmass == 'none':
            ptext = 'masks vs halo-only'
            if relative:
                divby = dct_totcddf['none']['cddf']
            else:
                divby = 1. 
        
            for pmass in masses_proj[1:]:
                _lw = linewidth
                _pe = patheff
                
                bins = dct_totcddf['bins']
                plotx = bins[:-1] + 0.5 * np.diff(bins)
                ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=_pe, linewidth=_lw)
                
                bins = dct_fofcddf[pmass]['bins']
                plotx = bins[:-1] + 0.5 * np.diff(bins)
                ax.plot(plotx, np.log10(dct_fofcddf[pmass]['none']['cddf'] / divby), color=colors[pmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
                
            _lw = linewidth
            _pe = patheff
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)

        elif pmass == 'all halos':
            ptext = 'all halo gas'
            if relative:
                divby = dct_fofcddf['none']['none']['cddf']
            else:
                divby = 1. 

            for mmass in masses_proj[1:]:
                bins = dct_fofcddf[mmass]['bins']
                plotx = bins[:-1] + 0.5 * np.diff(bins)
                ax.plot(plotx, np.log10(dct_fofcddf['none'][mmass]['cddf'] / divby), color=colors[mmass], linestyle='solid', alpha=alpha, path_effects=patheff)
            
            bins = dct_fofcddf['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf['none']['none']['cddf'] / divby), color=colors['none'], linestyle='solid', alpha=alpha, path_effects=patheff, linewidth=linewidth)
            
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(np.sum([dct_totcddf[mass]['cddf'] for mass in masses_proj[1:]], axis=0) / divby), color=colors['allhalos'], linestyle='dashed', alpha=alpha, path_effects=patheff_thick, linewidth=linewidth + 0.5)
            
            bins = dct_fofcddf['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(np.sum([dct_fofcddf['none'][mass]['cddf'] for mass in masses_proj[1:]], axis=0) / divby), color=colors['allhalos'], linestyle='solid', alpha=alpha, path_effects=patheff_thick, linewidth=linewidth + 0.5)
        
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=patheff, linewidth=linewidth)
            
        else:
            if pmass == 14.0:
                ptext = r'$ > %.1f$'%pmass # \log_{10} \, \mathrm{M}_{\mathrm{200c}} \, / \, \mathrm{M}_{\odot}
            else:
                ptext = r'$ %.1f \emdash %.1f$'%(pmass, pmass + 0.5) # \leq \log_{10} \, \mathrm{M}_{\mathrm{200c}} \, / \, \mathrm{M}_{\odot} <
            
            if relative:
                divby = dct_fofcddf[pmass]['none']['cddf']
            else:
                divby = 1.
            
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=patheff)
                
            for mmass in masses_proj[1:]:
                if mmass == pmass:
                    _pe = patheff_thick
                    _lw = linewidth + 0.5
                else:
                    _pe = patheff
                    _lw = linewidth
                    
                bins = dct_fofcddf[pmass]['bins']
                plotx = bins[:-1] + 0.5 * np.diff(bins)
                ax.plot(plotx, np.log10(dct_fofcddf[pmass][mmass]['cddf'] / divby), color=colors[mmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
            bins = dct_fofcddf[pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[pmass]['none']['cddf'] / divby), color=colors['none'], linestyle='solid', alpha=alpha, path_effects=patheff)
        
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=patheff_thick, linewidth=linewidth + 0.5)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=patheff, linewidth=linewidth)
            
        if relative:
            ax.text(0.05, 0.95, ptext, horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        else:
            ax.text(0.95, 0.95, ptext, horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
            
    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid', 'dashed']:
        subcols = list(clist) + [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    sumhandles = [mlines.Line2D([], [], color=colors['none'], linestyle='solid', label='FoF no mask', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='solid', label='total', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='solid', label=r'all FoF+200c gas', linewidth=2.),\
                  ]
    sumlabels = ['FoF+200c, no mask', 'all gas, no mask', r'mask: all haloes $> 9.0$']
    lax.legend(lcs + sumhandles, ['FoF+200c, with mask', 'all gas, with mask'] + sumlabels,\
               handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize,\
               ncol=ncol_legend, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(outname, format='pdf')


def quicklook_particlehist(h5name, group=None, plotlog=None):
    
    if '/' not in h5name:
        h5name = ol.ndir + h5name
    if h5name[-5:] != '.hdf5':
        h5name = h5name + '.hdf5'
    
    with h5py.File(h5name, 'r') as fi:
        keys = list(fi.keys())
        keys.remove('Header')
        keys.remove('Units')
        if len(keys) > 1:
            print('Which group do you want to plot?')
            for i in range(len(keys)):
                print('%i\t%s'%(i, keys[i]))
            choice = input('Please enter name or index: ')
            choice = choice.strip()
            try:
                ci = int(choice)
                gn = keys[ci]
            except ValueError:
                gn = choice
            if gn not in keys:
                raise ValueError('group "%s" is not one of the options'%(gn))
        else:
            gn = keys[0]
        rgrp = fi[gn]
        
        # extract hsitogram data
        hist = np.array(rgrp['histogram'])
        histlog = rgrp['histogram'].attrs['log']
        if plotlog is None:
            plotlog = histlog
        sumin = rgrp['histogram'].attrs['sum of weights']
        if histlog:
            hist = 10**hist
        sumout = np.sum(hist)
        missingfrac = 1. - sumin / sumout
        
        axdata = {}
        axns = list(rgrp.keys())
        axns.remove('histogram')
        axns.remove('binedges')
        for axn in axns:
            sgrp = rgrp[axn]
            axi = sgrp.attrs['histogram axis']
            axlog = sgrp.attrs['log']
            npartfin = sgrp.attrs['number of particles with finite values']
            nparttot = sgrp.attrs['number of particles']
            npartexclmin = sgrp.attrs['number of particles < min value']
            npartexclmax = sgrp.attrs['number of particles > max value']
            bins = np.array(sgrp['bins'])
            
            partselstr = '' 
            if npartfin < nparttot and not (bins[0] == -np.inf and bins[-1] == np.inf): # only finite data values are checked
                partselstr = partselstr + '%s / %s values not finite'%(nparttot - npartfin, nparttot)
                if axlog and bins[0] == -np.inf:
                    partselstr = partselstr + ', -inf counted in hist'
            if npartexclmin > 0:
                partselstr = partselstr + '\n%s / %s finte values excluded (too low)'%(npartexclmin, npartfin)
            if npartexclmax > 0:
                partselstr = partselstr + '\n%s / %s finte values excluded (too high)'%(npartexclmax, npartfin)
          
            if partselstr == '':
                partselstr = None
            elif  partselstr[0] == '\n':
                partselstr = partselstr[1:]
                
            if partselstr is not None:
                partselstr = '%s:\n'%axn + partselstr + '\n'
                
            if bins[0] == -np.inf:
                bins[0] = 2. * bins[1] - bins[2]
            if bins[-1] == np.inf:
                bins[-1] = 2. * bins[-2] - bins[-3]
                
            axdata[axi] = {'bins': bins, 'log': axlog, 'sel': partselstr, 'name': axn}
            
    # set up plot grid based on number of dimensions
    ndim = len(hist.shape)
    
    panelwidth = 2.5
    panelheight = 2.5
    caxwidth = 1.
    textheight = 2.
    fontsize = 12
    cmap = 'gist_yarg'
    
    cext = 'neither'
    if plotlog:
        vmin = np.log10(np.min(hist[hist > 0]))
        vmax = np.log10(np.max(hist[hist > 0]))
        clabel = 'log10 weight'
        if vmax - vmin > 10.:
            vmin = vmax - 10.
            cext = 'min'
    else:
        vmin = np.min(hist[hist > 0])
        vmax = np.max(hist)
        clabel = 'weight'
        
    fig = plt.figure(figsize=(ndim * panelwidth + caxwidth, ndim * panelheight + textheight))
    maingrid = gsp.GridSpec(ndim + 1, ndim + 1, hspace=0.05, wspace=0.05, height_ratios=[panelheight] * ndim + [textheight], width_ratios=[panelwidth] * ndim + [caxwidth])
    cax = fig.add_subplot(maingrid[:ndim , ndim])
    tax = fig.add_subplot(maingrid[ndim, :])
    
    titleax = fig.add_subplot(maingrid[:, :])
    title = h5name.split('/')[-1] + '\n%s'%gn
    titleax.text(0.5, 1.01, title, fontsize=fontsize, verticalalignment='bottom', horizontalalignment='center', transform=titleax.transAxes)
    titleax.axis('off')
    
    for ci in range(ndim):
        for ri in range(ndim):
            if ci > ri: # above the diagonal
                continue
            ax = fig.add_subplot(maingrid[ri, ci])
            hax = ci
            hay = ri
            datax = axdata[hax]
            datay = axdata[hay]
            
            labelx = ri == ndim - 1
            labely = ci == 0 and ri > 0
            labelright = ci == ri
    
            if labelx:    
                label = datax['name']
                if datax['log']:
                    label = 'log10 ' + label
                ax.set_xlabel(label, fontsize=fontsize)
            if labely:    
                label = datay['name']
                if datay['log']:
                    label = 'log10 ' + label
                ax.set_ylabel(label, fontsize=fontsize)
            ax.tick_params(direction='in', which='both', top=True, right=True,\
                           labelbottom=labelx, labelleft=labely, labelright=labelright,\
                           labelsize=fontsize - 1)
                    
            if ci == ri: # histogram plot
                sumtup = range(ndim)
                sumtup.remove(hax)
                sumtup = tuple(sumtup)
                hist_t = np.sum(hist, axis=sumtup)
                if plotlog:
                    hist_t = np.log10(hist_t)
                ax.step(datax['bins'], np.append(hist_t, [np.NaN]), where='post')
            
            else:
                sumtup = range(ndim)
                sumtup.remove(hax)
                sumtup.remove(hay)
                sumtup = tuple(sumtup)
                hist_t = np.sum(hist, axis=sumtup)
                if plotlog:
                    hist_t = np.log10(hist_t)
                img = ax.pcolormesh(datax['bins'], datay['bins'], hist_t.T, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
    
    plt.colorbar(img, cax=cax, orientation='vertical', extend=cext)
    cax.set_aspect(10.)
    cax.set_ylabel(clabel, fontsize=fontsize)
    cax.tick_params(which='both', labelsize=fontsize - 1.)
    
    addtext = 'missing weight fraction: %.5f\n'%missingfrac
    for i in range(ndim):
        if axdata[i]['sel'] is not None:
            addtext = addtext + axdata[i]['sel']
        
    tax.text(0.05, 0.7, addtext, fontsize=fontsize, verticalalignment='top', horizontalalignment='left', transform=tax.transAxes)
    tax.axis('off')
    
    tparts = title.split('\n')
    imgname = tparts[0][:-5] + '_' + tparts[1]
    mdir = '/net/luttero/data2/imgs/histograms_basic/'
    plt.savefig(mdir + imgname + '.pdf', format='pdf', bbox_inches='tight')
    

def plotpd_by_halo_subcat(ion):
    
    datafile_dir = '/net/quasar/data2/wijers/temp/'
    if ion == 'Mass':
        datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_T4EOS.hdf5'
        datafile = datafile_dir + datafile_base%(ion)
    else:
        datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_PtAb_T4EOS.hdf5'
        datafile = datafile_dir + datafile_base%('Nion_%s'%ion)
    
    outname = '/net/luttero/data2/imgs/histograms_basic/' + 'overview_phasediagram_%s_L0100N1504_27_T4EOS.pdf'%(ion)

    tlabel = r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$'
    dlabel = r'$\log_{10} \, \mathrm{n}_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$'
    
    
    with h5py.File(datafile, 'r') as fi:
        groupname = 'Temperature_T4EOS_Density_T4EOS_M200c_halo_allinR200c_subhalo_category'
        tname = 'Temperature_T4EOS'
        dname = 'Density_T4EOS'
        sname = 'subhalo_category'
        hname = 'M200c_halo_allinR200c'
        
        mgrp = fi[groupname]       
        tax = mgrp[tname].attrs['histogram axis']
        tbins = np.array(mgrp['%s/bins'%(tname)])
        dax = mgrp[dname].attrs['histogram axis']
        dbins = np.array(mgrp['%s/bins'%(dname)]) + np.log10(rho_to_nh)      
        sax = mgrp[sname].attrs['histogram axis']
        sbins = np.array(mgrp['%s/bins'%(sname)])
        hax = mgrp[hname].attrs['histogram axis']
        hbins = np.array(mgrp['%s/bins'%(hname)])
        
        hist = np.array(mgrp['histogram'])
        if mgrp['histogram'].attrs['log']:
            hist = 10**hist
        total_in = mgrp['histogram'].attrs['sum of weights']
        print('Histogramming recovered %f of input weights'%(np.sum(hist) / total_in))
        
        cosmopars = {key: item for key, item in fi['Header/cosmopars'].attrs.items()}
        
    slabels = ['cen.', 'sat.', 'unb.']
    hlabels = [r'$ %.1f \emdash %.1f$'%(hbins[i] - np.log10(c.solar_mass), hbins[i + 1] - np.log10(c.solar_mass)) if i > 1 and hbins[i + 1] < np.inf else\
               r'$> %.1f$'%(hbins[i] - np.log10(c.solar_mass)) if hbins[i + 1] == np.inf else \
               r'$< %.1f$'%(hbins[i + 1] - np.log10(c.solar_mass)) if i == 1  else \
               'IGM' \
               for i in range(len(hbins) - 1)]
    
    cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    cmap.set_under(cmap(0.))
        
    panelwidth = 2.0
    panelheight = 2.0
    #caxwidth = 1.
    textheight = 2.
    fontsize = 12
    cmap = 'gist_yarg'
    plotlog=True
    
    
    if ion == 'Mass':
        clabel = 'Mass fraction'
    else:
        clabel = r'$\mathrm{%s}$ fraction'%(ild.getnicename(ion, mathmode=True))       
    cext = 'neither'
    if plotlog:
        vmin = np.log10(np.min(hist[hist > 0]) / total_in)
        vmax = np.log10(np.max(hist[hist > 0]) / total_in)
        clabel = r'$\log_{10}$ ' + clabel
        if vmax - vmin > 10.:
            vmin = vmax - 10.
            cext = 'min'
    else:
        vmin = np.min(hist[hist > 0])
        vmax = np.max(hist)
    
    numhorz = (1 + len(hbins) - 1) // 2
    numvert = 2 * (len(sbins))
        
    fig = plt.figure(figsize=(numhorz * panelwidth, numvert * panelheight + textheight))
    maingrid = gsp.GridSpec(ncols=numhorz, nrows=numvert + 1, hspace=0.0, wspace=0.0, height_ratios=[textheight] + [panelheight] * numvert, width_ratios=[panelwidth] * numhorz)
    cax = fig.add_subplot(maingrid[0, numhorz - 2:])
    titleax = fig.add_subplot(maingrid[0, :-2])
    
    title = 'phase diagram for %s'%ion
    titleax.text(0.5, 0.1, title, fontsize=fontsize + 2, verticalalignment='bottom', horizontalalignment='center', transform=titleax.transAxes)
    titleax.axis('off')
    
    for ci in range(0, numhorz, 1):
        for ri in range(1, numvert + 1, 1):
            si = (ri - 1) // 2
            hi = ci + ((ri - 1) % 2) * numhorz
            if hi > len(hlabels):
                continue
            
            ax = fig.add_subplot(maingrid[ri, ci])
            
            labelx = ri == numvert
            labely = ci == 0
            labelright = False
    
            if labelx:    
                ax.set_xlabel(dlabel, fontsize=fontsize)
            if labely:    
                ax.set_ylabel(tlabel, fontsize=fontsize)
            ax.tick_params(direction='in', which='both', top=True, right=True,\
                           labelbottom=labelx, labelleft=labely, labelright=labelright,\
                           labelsize=fontsize - 1)
            
            sellabel_t = ''
            
            hist_t = np.copy(hist)
            sumtup = range(4)
            sumtup.remove(dax)
            sumtup.remove(tax)
            if hi == 0:
                pass
            else:
                hsel = [slice(None, None, None)] * 4
                hsel[hax] = slice(hi - 1, hi, None)
                hist_t = hist_t[tuple(hsel)] # along hax, size of array is 1, but axis still exists
                if len(sellabel_t) > 0: 
                    sellabel_t = sellabel_t + '\n'
                sellabel_t  = sellabel_t + hlabels[hi - 1]
            if si == 0:
                pass
            else:
                ssel = [slice(None, None, None)] * 4
                ssel[sax] = slice(si - 1, si, None)
                hist_t = hist_t[tuple(ssel)] # along hax, size of array is 1, but axis still exists
                if len(sellabel_t) > 0: 
                    sellabel_t = sellabel_t + '\n'
                sellabel_t  = sellabel_t + slabels[si - 1]
            sumtup = tuple(sumtup)
            hist_t = np.sum(hist_t, axis=sumtup)
            total_t = np.sum(hist_t)
            
            if total_t == 0.:
                ax.axis('off')
                continue
            
            hist_t /= total_in
            if plotlog:
                hist_t = np.log10(hist_t)
            
            
            if tax > dax:
                hist_t = hist_t.T
            img = ax.pcolormesh(dbins, tbins, hist_t, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
            
            if hi > 0:
                T200c_lo = T200c_hot(10**(hbins[hi - 1]) / c.solar_mass, cosmopars=cosmopars)
                T200c_hi = T200c_hot(10**(hbins[hi]) / c.solar_mass, cosmopars=cosmopars)
                if T200c_lo < 10**(ax.get_ylim()[1]) and T200c_lo > 10**(ax.get_ylim()[0]):
                    ax.axhline(np.log10(T200c_lo), linestyle='dotted', linewidth=1.5, color='C0')
                if T200c_hi < 10**(ax.get_ylim()[1]) and T200c_hi > 10**(ax.get_ylim()[0]):
                    ax.axhline(np.log10(T200c_hi), linestyle='dotted', linewidth=1.5, color='C0')
                
            ax.text(0.02, 0.98, sellabel_t, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, fontsize=fontsize)
            ax.text(0.98, 0.78, '%.2e'%(total_t / total_in), verticalalignment='top', horizontalalignment='right', transform=ax.transAxes, fontsize=fontsize)
    
    plt.colorbar(img, cax=cax, orientation='horizontal', extend=cext)
    cax.set_aspect(1./10.)
    cax.set_xlabel(clabel, fontsize=fontsize)
    cax.tick_params(which='both', labelsize=fontsize - 1.)
    cax.xaxis.set_label_position('top')
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')   


def plotfracs_by_halo_subcat(ions=['Mass', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'], first='halo', fmt='pdf'):
    '''
    first: group mass bins by halo mass or subhalo catgory first
    !!! calculated from particle data -> IGM contributions will be wrong !!!
    '''
    
    datafile_dct = {}
    for ion in ions:
        datafile_dir = '/net/quasar/data2/wijers/temp/'
        if ion == 'Mass':
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_T4EOS.hdf5'
            datafile = datafile_dir + datafile_base%(ion)
        else:
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_PtAb_T4EOS.hdf5'
            datafile = datafile_dir + datafile_base%('Nion_%s'%ion)
        datafile_dct[ion] = datafile
    
    outname = '/net/luttero/data2/imgs/histograms_basic/' + 'barchart_halosubcat_L0100N1504_27_T4EOS_%s-first.%s'%(first, fmt)
    
    data_dct = {}
    for ion in ions:
        datafile = datafile_dct[ion]
        with h5py.File(datafile, 'r') as fi:
            groupname = 'Temperature_T4EOS_Density_T4EOS_M200c_halo_allinR200c_subhalo_category'
            tname = 'Temperature_T4EOS'
            dname = 'Density_T4EOS'
            sname = 'subhalo_category'
            hname = 'M200c_halo_allinR200c'
            
            mgrp = fi[groupname]       
            tax = mgrp[tname].attrs['histogram axis']
            #tbins = np.array(mgrp['%s/bins'%(tname)])
            dax = mgrp[dname].attrs['histogram axis']
            #dbins = np.array(mgrp['%s/bins'%(dname)]) + np.log10(rho_to_nh)      
            sax = mgrp[sname].attrs['histogram axis']
            sbins = np.array(mgrp['%s/bins'%(sname)])
            hax = mgrp[hname].attrs['histogram axis']
            hbins = np.array(mgrp['%s/bins'%(hname)])
            
            hist = np.array(mgrp['histogram'])
            if mgrp['histogram'].attrs['log']:
                hist = 10**hist
            hist = np.sum(hist, axis=(dax, tax))
            total_in = mgrp['histogram'].attrs['sum of weights']
            # print('Histogramming recovered %f of input weights'%(np.sum(hist) / total_in))
            
            #cosmopars = {key: item for key, item in fi['Header/cosmopars'].attrs.items()}
            
            data_dct[ion] = {'hist': hist, 'hbins': hbins, 'sbins': sbins, 'total': total_in}
            
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'        
    ylabel = 'fraction'
    xlabels = [r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)) if ion != 'Mass' else ion for ion in ions] 
    slabels = ['cen.', 'sat.', 'unb.']
    alphas = {'cen.': 1.0, 'sat.': 0.4, 'unb.': 0.7}
    
    
    cmapname = 'rainbow'
    nigmcolor = 'saddlebrown'
    igmcolor = 'gray'   
    print(hbins - np.log10(c.solar_mass))
    namededges = hbins[2:-1] - np.log10(c.solar_mass) # first two are -np.inf, and < single particle mass, last bin is empty (> 10^15 Msun)
    print(namededges)
    
    clist = cm.get_cmap(cmapname, len(namededges) - 2)(np.linspace(0., 1., len(namededges) - 2))
    clist = np.append(clist, [mpl.colors.to_rgba('firebrick')], axis=0)
    #print(clist)
    colors = {hi : clist[hi - 2] for hi in range(2, 2 + len(namededges) - 1)}
    #print(colors)
    colors[1] = mpl.colors.to_rgba(nigmcolor)
    colors[0] = mpl.colors.to_rgba(igmcolor)
    colors[len(hbins) - 1] = mpl.colors.to_rgba('magenta') # shouldn't actaully be used
    #print(colors)

    fig = plt.figure(figsize=(5.5, 3.))
    maingrid = gsp.GridSpec(ncols=2, nrows=2, hspace=0.0, wspace=0.05, height_ratios=[0.7, 4.3], width_ratios=[5., 1.])
    cax = fig.add_subplot(maingrid[1, 1])
    lax = fig.add_subplot(maingrid[0, :])
    ax = fig.add_subplot(maingrid[1, 0])
    
    print(namededges)
    cmap = mpl.colors.ListedColormap(clist)
    cmap.set_under(nigmcolor)
    #cmap.set_over('magenta')
    norm = mpl.colors.BoundaryNorm(namededges, cmap.N)
    print(len(clist))
    print(cmap.N)
    print(len(namededges))
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append([0.], namededges, axis=0),\
                                ticks=namededges,\
                                spacing='proportional', extend='min',\
                                orientation='vertical')
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)
    
    if first == 'halo':
        range1 = range(len(hbins) - 2)
        range2 = range(len(sbins) - 1)
    else:
        range1 = range(len(sbins) - 1)
        range2 = range(len(hbins) - 2)
    
    
    bottom = np.zeros(len(ions))
    barwidth = 0.9
    xvals = np.arange(len(ions))
    
    for ind1 in range1:
        for ind2 in range2:
            if first == 'halo':
                hind = ind1
                sind = ind2
            else:
                hind = ind2
                sind = ind1
                
            alpha = alphas[slabels[sind]]
            color = mpl.colors.to_rgba(colors[hind], alpha=alpha)
            sel = (hind, sind) if hax < sax else (sind, hind)
            
            vals = np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            
            ax.bar(xvals, vals, barwidth, bottom=bottom, color=color)
            bottom += vals
            
    setticks(ax, fontsize, top=False)
    ax.xaxis.set_tick_params(which='both', length=0.) # get labels without showing the ticks        
    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels)
    for label in ax.get_xmajorticklabels():
        label.set_rotation(45)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    legelts = [mpatch.Patch(facecolor=igmcolor, label='IGM')] + \
              [mpatch.Patch(facecolor=mpl.colors.to_rgba('tan', alpha=alphas[slabel]), label=slabel) for slabel in slabels]
    lax.legend(handles=legelts, ncol=4, fontsize=fontsize, bbox_to_anchor=(0.5, 0.05), loc='lower center')
    lax.axis('off')
    
    plt.savefig(outname, format=fmt, bbox_inches='tight')


def plot3Dprof_overview(weighttype='Mass'):
    '''
    plot: cumulative profile of weight, [ion number density profile], 
          rho profile, T profile
    rows show different halo mass ranges
    '''
    outdir = '/net/luttero/data2/imgs/CGM/3dprof/'
    outname = outdir + 'overview_radprof_L0100N1504_27_Mh0p5dex_1000_%s.pdf'%(weighttype)
    
    fontsize = 12
    cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    cmap.set_under(cmap(0.))
    percentiles = [0.05, 0.50, 0.95]
    linestyles = ['dashed', 'solid', 'dashed']
    
    rbinu = 'R200c'
    combmethod = 'addnormed-R200c'
    #binq = 'M200c_Msun'
    binqn = r'$\mathrm{M}_{\mathrm{200c}} \, [\mathrm{M}_{\odot}]$' 
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
    wname = ild.getnicename(weighttype, mathmode=True) if weighttype in ol.elements_ion.keys() else \
            r'\mathrm{Mass}' if weighttype == 'Mass' else \
            r'\mathrm{Volume}' if weighttype == 'Volume' else \
            None
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%(wname, wname) 
                }
    clabel = r'$\log_{10} \, \left\langle %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}}) \right\rangle \, / \,$'%(wname, wname) + 'bin size'
    
    if weighttype in ol.elements_ion.keys():
        filename = ol.ndir + 'particlehist_Nion_%s_L0100N1504_27_test3.4_PtAb_T4EOS_galcomb.hdf5'%(weighttype)
        nprof = 4
        title = r'$\mathrm{%s}$ and $\mathrm{%s}$-weighted profiles'%(wname, wname)
        tgrpn = '3Dradius_Temperature_T4EOS_Density_T4EOS_Niondens_%s_PtAb_T4EOS_R200c_snapdata'%(weighttype)
        axns  = {'r3d':  '3Dradius',\
                'T':    'Temperature_T4EOS',\
                'rho':  'Density_T4EOS',\
                'nion': 'Niondens_%s_PtAb_T4EOS'%(weighttype),\
                }
        axnl = ['nion', 'T', 'rho']
    else:
        if weighttype == 'Volume':
            filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.4_T4EOS_galcomb.hdf5'%('propvol')
        else:
            filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.4_T4EOS_galcomb.hdf5'%(weighttype)
        nprof = 3
        title = r'%s and %s-weighted profiles'%(weighttype, weighttype)
        tgrpn = '3Dradius_Temperature_T4EOS_Density_T4EOS_R200c_snapdata'
        axns = {'r3d':  '3Dradius',\
                'T':    'Temperature_T4EOS',\
                'rho':  'Density_T4EOS',\
                }
        axnl = ['T', 'rho']
        
    file_galsin = '/net/luttero/data2/imgs/CGM/3dprof/filenames_L0100N1504_27_Mh0p5dex_1000_%s.txt'%(weighttype)
    file_galdata = '/net/luttero/data2/imgs/CGM/3dprof/halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    
    # generated randomly once
    #galids_per_bin = {11.0: [13074219,  3802158,  3978003,  3801075, 13588395, 11396298, 8769997,  12024375, 12044831, 12027193],\
    #                  11.5: [11599169, 11148475, 10435177,  9938601, 10198004,  9626515, 10925515, 10472334, 13823711, 11382071],\
    #                  12.0: [17795988,  8880354, 18016380,  8824646,  8976542,  8948515, 8593530,   9225418, 18167602,  8991644],\
    #                  12.5: [16907882, 16565965, 15934507, 15890726, 16643442, 16530723, 8364907,  14042157, 14837489, 14195766],\
    #                  13.0: [20009129, 20309020, 19987958, 19462909, 20474648, 19615775, 19488333, 19975482, 20519792, 19784480],\
    #                  13.5: [18816265, 18781590, 19634930, 18961507, 18927203, 19299051, 6004915,  20943533, 18849993, 21059563],\
    #                  14.0: [19701410, 10705995, 14978116, 21986362, 21109761, 21242351, 21573587, 21730536, 21379522],\
    #                 }
    galids_per_bin = {11.0: [13074219,  3802158,  3978003],\
                      11.5: [11599169, 11148475, 10435177],\
                      12.0: [17795988,  8880354, 18016380],\
                      12.5: [16907882, 16565965, 15934507],\
                      13.0: [20009129, 20309020, 19987958],\
                      13.5: [18816265, 18781590, 19634930],\
                      14.0: [21242351, 10705995, 21242351],\
                     }
    
    mgrpn = 'L0100N1504_27_Mh0p5dex_1000/%s-%s'%(combmethod, rbinu)
    
    # read in data: stacked histograms -> process to plottables
    hists_main = {}
    edges_main = {}
    galids_main = {}
    
    with h5py.File(filename, 'r') as fi:
        grp = fi[tgrpn + '/' + mgrpn]
        sgrpns = list(grp.keys())
        massbins = [grpn.split('_')[-1] for grpn in sgrpns]    
        massbins = [[np.log10(float(val)) for val in binn.split('-')] for binn in massbins]
        
        for mi in range(len(sgrpns)):
            mkey = massbins[mi][0]
            
            grp_t = grp[sgrpns[mi]]
            hist = np.array(grp_t['histogram'])
            if bool(grp_t['histogram'].attrs['log']):
                hist = 10**hist
            
            edges = {}
            axes = {}
            for axn in axns:
               edges[axn] = np.array(grp_t[axns[axn] + '/bins'])
               if not bool(grp_t[axns[axn]].attrs['log']):
                   edges[axn] = np.log10(edges[axn])
               axes[axn] = grp_t[axns[axn]].attrs['histogram axis']  
            
            edges_main[mkey] = {}
            hists_main[mkey] = {}
            
            # apply normalization consisent with stacking method
            if rbinu == 'pkpc':
                edges['r3d'] += np.log10(c.cm_per_mpc * 1e-3)
            
            if combmethod == 'addnormed-R200c':
                if rbinu != 'R200c':
                    raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                if len(_i) != 1:
                    raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                _i = _i[0]
                _a = range(len(hist.shape))
                _s = [slice(None, None, None) for dummy in _a]
                _s[axes['r3d']] = slice(None, _i, None)
                norm_t = np.sum(hist[tuple(_s)])
            hist *= (1. / norm_t)
            
            for pt in axnl:
                rax = axes['r3d']
                yax = axes[pt]
                
                edges_r = np.copy(edges['r3d'])
                edges_y = np.copy(edges[pt])
                
                hist_t = np.copy(hist)
                
                # deal with edge units (r3d is already in R200c units if R200c-stacked)
                if edges_r[0] == -np.inf: # reset centre bin position
                    edges_r[0] = 2. * edges_r[1] - edges_r[2] 
                if pt == 'rho':
                    edges_y += np.log10(rho_to_nh)
                    
                sax = range(len(hist_t.shape))
                sax.remove(rax)
                sax.remove(yax)
                hist_t = np.sum(hist_t, axis=tuple(sax))
                if yax < rax:
                    hist_t = hist_t.T
                #hist_t /= (np.diff(edges_r)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
                
                hists_main[mkey][pt] = hist_t
                edges_main[mkey][pt] = [edges_r, edges_y]
                #print(hist_t.shape)
            
            # add in cumulative plot for the weight
            hist_t = np.copy(hist)
            sax = range(len(hist_t.shape))
            sax.remove(rax)
            hist_t = np.sum(hist_t, axis=tuple(sax))
            hist_t = np.cumsum(hist_t)
            hists_main[mkey]['weight'] = hist_t
            edges_main[mkey]['weight'] = [edges_r[1:]]
                
            
            galids_main[mkey] = np.array(grp_t['galaxyids'])
    
    # read in data: individual galaxies
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', index_col='galaxyid')
    galname_all = pd.read_csv(file_galsin, header=0, sep='\t', index_col='galaxyid')
    
    hists_single = {}
    edges_single = {}
    
    for mbin in galids_per_bin:
        galids = galids_per_bin[mbin]
        for galid in galids:
            filen = galname_all.at[galid, 'filename']
            grpn = galname_all.at[galid, 'groupname']
            if rbinu == 'R200c':
                Runit = galdata_all.at[galid, 'R200c_cMpc'] * c.cm_per_mpc * cosmopars['a']
            else:
                Runit = c.cm_per_mpc * 1e-3 #pkpc
            
            with h5py.File(filen, 'r') as fi:
                grp_t = fi[grpn]
                    
                hist = np.array(grp_t['histogram'])
                if bool(grp_t['histogram'].attrs['log']):
                    hist = 10**hist
                    
                edges = {}
                axes = {}
                
                for axn in axns:
                   edges[axn] = np.array(grp_t[axns[axn] + '/bins'])
                   if axn == 'r3d':
                       edges[axn] *= (1./ Runit)
                   if not bool(grp_t[axns[axn]].attrs['log']):
                       edges[axn] = np.log10(edges[axn])
                   axes[axn] = grp_t[axns[axn]].attrs['histogram axis']          
        
                if combmethod == 'addnormed-R200c':
                    if rbinu != 'R200c':
                        raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                    _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                    if len(_i) != 1:
                        raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                    _i = _i[0]
                    _a = range(len(hist.shape))
                    _s = [slice(None, None, None) for dummy in _a]
                    _s[axes['r3d']] = slice(None, _i, None)
                    norm_t = np.sum(hist[tuple(_s)])
                
                hist *= (1. / norm_t)
                
                hists_single[galid] = {}
                edges_single[galid] = {}
                
                for pt in axnl:
                    rax = axes['r3d']
                    yax = axes[pt]
                    
                    edges_r = np.copy(edges['r3d'])
                    edges_y = np.copy(edges[pt])
                    
                    hist_t = np.copy(hist)
                    
                    # deal with edge units (r3d is already in R200c units if R200c-stacked)
                    if edges_r[0] == -np.inf: # reset centre bin position
                        edges_r[0] = 2. * edges_r[1] - edges_r[2] 
                    if edges_y[0] == -np.inf: # reset centre bin position
                        edges_y[0] = 2. * edges_y[1] - edges_y[2]
                    if edges_y[-1] == np.inf: # reset centre bin position
                        edges_y[-1] = 2. * edges_y[-2] - edges_y[-3]
                    if pt == 'rho':
                        edges_y += np.log10(rho_to_nh)
                        
                    sax = range(len(hist_t.shape))
                    sax.remove(rax)
                    sax.remove(yax)
                    hist_t = np.sum(hist_t, axis=tuple(sax))
                    if yax < rax:
                        hist_t = hist_t.T
                    #hist_t /= (np.diff(edges_r)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
                    
                    hists_single[galid][pt] = hist_t
                    edges_single[galid][pt] = [edges_r, edges_y]
                
                # add in cumulative plot for the weight
                hist_t = np.copy(hist)
                sax = range(len(hist_t.shape))
                sax.remove(rax)
                hist_t = np.sum(hist_t, axis=tuple(sax))
                hist_t = np.cumsum(hist_t)
                hists_single[galid]['weight'] = hist_t
                edges_single[galid]['weight'] = [edges_r[1:]]
            
    # set up plot grid
    panelwidth = 3.
    panelheight = 3.
    toplabelheight = 0.5
    caxwidth = 0.5
    nmassbins = len(hists_main)
    
    fig = plt.figure(figsize=(nmassbins * panelwidth + caxwidth, nprof * panelheight + toplabelheight))
    grid = gsp.GridSpec(nrows=nprof + 1, ncols=nmassbins + 1, hspace=0.0, wspace=0.0, width_ratios=[panelwidth] * nmassbins + [caxwidth], height_ratios=[toplabelheight] + [panelheight] * nprof )
    axes = np.array([[fig.add_subplot(grid[yi + 1, xi]) for xi in range(nmassbins)] for yi in range(nprof)])
    cax  = fig.add_subplot(grid[1:, nmassbins])
    laxes = [fig.add_subplot(grid[0, xi]) for xi in range(nmassbins)]
    
    vmax = np.log10(np.max([np.max([np.max(hists_main[mkey][axn]) for axn in axnl]) for mkey in hists_main]))
    vmin = np.log10(np.min([np.min([np.min(hists_main[mkey][axn]) for axn in axnl]) for mkey in hists_main]))
    vmin = max(vmin, vmax - 7.)
    
    massmins = sorted(list(hists_main.keys()))

    linewidth = 1.
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.5, foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
     
    fig.suptitle(title, fontsize=fontsize + 2)
    
    
    for mi in range(nmassbins):
        ind = np.where(np.array(massbins)[:, 0] == massmins[mi])[0][0]
        text = r'$%.1f \, \endash \, %.1f$'%(massbins[ind][0], massbins[ind][1]) #r'$\log_{10} \,$' + binqn + r': 
        
        ax = laxes[mi]
        ax.text(0.5, 0.1, text, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='center', verticalalignment='bottom')
        ax.axis('off') 
        
    for mi in range(nmassbins):
        for ti in range(nprof):
            # where are we
            ax = axes[ti, mi]
            labelx = ti == nprof - 1
            labely = mi == 0
            
            if ti == 0:
                yq = 'weight'
            else:
                yq = axnl[ti - 1]
            mkey = massmins[mi]
            
            # set up axis
            setticks(ax, top=True, left=True, labelleft=labely, labelbottom=labelx, fontsize=fontsize)
            if labelx:
                ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            if labely:
                ax.set_ylabel(axlabels[yq], fontsize=fontsize)
            
            # plot stacked histogram
            edges_r = edges_main[mkey][yq][0] 
            if yq != 'weight':
                edges_y = edges_main[mkey][yq][1]
                hist = hists_main[mkey][yq]
                
                img, _1, _2 = pu.add_2dplot(ax, hist, [edges_r, edges_y], toplotaxes=(0, 1),\
                              log=True, usepcolor=True, pixdens=True,\
                              cmap=cmap, vmin=vmin, vmax=vmax, zorder=-2)
                perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles))
                mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                
                for pi in range(len(percentiles)):
                    ax.plot(mid_r, perclines[pi], color='white',\
                            linestyle=linestyles[pi], alpha=1.,\
                            path_effects=patheff_thick, linewidth=linewidth + 1)
                
                mmatch = np.array(list(galids_per_bin.keys()))
                ind = np.where(np.isclose(mkey, mmatch))[0][0]
                galids = galids_per_bin[mmatch[ind]]
                
                for galidi in range(len(galids)):
                    galid = galids[galidi]
                    color = 'C%i'%(galidi % 10)
                    
                    hist = hists_single[galid][yq]
                    edges_r = edges_single[galid][yq][0]
                    edges_y = edges_single[galid][yq][1]
                    
                    perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles))
                    mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                    
                    for pi in range(len(percentiles)):
                        ax.plot(mid_r, perclines[pi], color=color,\
                                linestyle=linestyles[pi], alpha=1.,\
                                path_effects=patheff, linewidth=linewidth,\
                                zorder = -1)
            else:
                hist = hists_main[mkey][yq]
                
                ax.plot(edges_r, np.log10(hist), color='black',\
                            linestyle='solid', alpha=1.,\
                            path_effects=None, linewidth=linewidth + 1.5)
                
                mmatch = np.array(list(galids_per_bin.keys()))
                ind = np.where(np.isclose(mkey, mmatch))[0][0]
                galids = galids_per_bin[mmatch[ind]]
                
                for galidi in range(len(galids)):
                    galid = galids[galidi]
                    color = 'C%i'%(galidi % 10)
                    
                    hist = hists_single[galid][yq]
                    edges_r = edges_single[galid][yq][0]
                    
                    ax.plot(edges_r, np.log10(hist), color=color,\
                            linestyle='solid', alpha=1.,\
                            path_effects=None, linewidth=linewidth + 0.5,\
                            zorder=-1)
    # color bar 
    pu.add_colorbar(cax, img=img, vmin=vmin, vmax=vmax, cmap=cmap,\
                    clabel=clabel, fontsize=fontsize, orientation='vertical',\
                    extend='min')
    cax.set_aspect(10.)
    
    # sync y limits on plots
    for yi in range(nprof):
        ylims = np.array([axes[yi, mi].get_ylim() for mi in range(nmassbins)])
        miny = np.min(ylims[:, 0])
        maxy = np.max(ylims[:, 1])
        # intended for ion number densities
        miny = max(miny, maxy - 17.)
        [[axes[yi, mi].set_ylim(miny, maxy) for mi in range(nmassbins)]]
    for xi in range(nmassbins):
        xlims = np.array([axes[i, xi].get_xlim() for i in range(nprof)])
        minx = np.min(xlims[:, 0])
        maxx = np.max(xlims[:, 1])
        [axes[i, xi].set_xlim(minx, maxx) for i in range(nprof)]
    
    plt.savefig(outname, format='pdf', box_inches='tight')


def plot_3dprof_overview_wZ(weighttype='Mass'):
    '''
    Overviews from stored 2d profiles. No single-agalaxy comparisons, though
    '''
    
    file_in = '/net/luttero/data2/imgs/CGM/3dprof/hists3d_forplot_to2d-1dversions_minr-0.05.hdf5'
    file_out = '/net/luttero/data2/imgs/CGM/3dprof/overview_radprof_L0100N1504_27_Mh0p5dex_1000_wZ_%s.pdf'%(weighttype)
    percentiles = [0.05, 0.50, 0.95]
    linestyles = ['dashed', 'solid', 'dashed']
    cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    wname = weighttype if weighttype not in ol.elements_ion.keys() else ild.getnicename(weighttype, mathmode=True)
    clabel = r'$\log_{10} \mathrm{%s}(r) \, / \, \mathrm{%s}(< \mathrm{R}_{\mathrm{200c}})$ / bin size'%(wname, wname)
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}_{\mathrm{H}}} \; [\mathrm{cm}^{-3}]$',\
                'nion': r'$\log_{10} \, \mathrm{n}_{\mathrm{ion}}} \; [\mathrm{cm}^{-3}]$',\
                'Z_oxygen': '[O / H]',\
                'Z_neon': '[Ne / H]',\
                'Z_iron': '[Fe / H]',\
                'weight': r'$\log_{10} \mathrm{%s}(<r) \, / \, \mathrm{%s}(< \mathrm{R}_{\mathrm{200c}})$'%(wname, wname)
                }
    wname = weighttype if weighttype not in ol.elements_ion.keys() else ild.getnicename(weighttype, mathmode=False)
    title = '%s and %s-weighted profiles'%(wname, wname)
    hists_main = {}
    edges_main = {}
    hists_norm = {}
    with h5py.File(file_in) as fi:
        mgrp = fi[weighttype]
        massbins = np.array(fi['mass_bins'])
        mass_keys = [key.decode() for key in np.array(fi['mass_keys'])]
        axnl = list(mgrp[mass_keys[0]].keys())
        for mkey in mass_keys:
            edges_main[mkey] = {}
            hists_main[mkey] = {}
            hists_norm[mkey] = {}
            for axn in axnl:
                hists_main[mkey][axn] = np.array(mgrp['%s/%s/hist'%(mkey, axn)])
                if axn != 'weight':
                    edges_main[mkey][axn] = [np.array(mgrp['%s/%s/edges_0'%(mkey, axn)]),\
                                             np.array(mgrp['%s/%s/edges_1'%(mkey, axn)]) ]
                    hists_norm[mkey][axn] = hists_main[mkey][axn] / (np.diff(edges_main[mkey][axn][0])[:, np.newaxis] * np.diff(edges_main[mkey][axn][1])[np.newaxis, :])
                else:
                    edges_main[mkey][axn] = [np.array(mgrp['%s/%s/edges'%(mkey, axn)])]
                if axn.startswith('Z_'):
                    solar = ol.solar_abunds_ea[axn[2:]]
                    edges_main[mkey][axn][1] -= np.log10(solar)
                    
    nprof = len(hists_main[mass_keys[0]].keys())
    # set up plot grid
    panelwidth = 3.
    panelheight = 3.
    toplabelheight = 0.5
    caxwidth = 0.5
    nmassbins = len(hists_main.keys())
    
    fig = plt.figure(figsize=(nmassbins * panelwidth + caxwidth, nprof * panelheight + toplabelheight))
    grid = gsp.GridSpec(nrows=nprof + 1, ncols=nmassbins + 1, hspace=0.0, wspace=0.0, width_ratios=[panelwidth] * nmassbins + [caxwidth], height_ratios=[toplabelheight] + [panelheight] * nprof )
    axes = np.array([[fig.add_subplot(grid[yi + 1, xi]) for xi in range(nmassbins)] for yi in range(nprof)])
    cax  = fig.add_subplot(grid[1:, nmassbins])
    laxes = [fig.add_subplot(grid[0, xi]) for xi in range(nmassbins)]
    
    vmax = np.log10(np.max([np.max([np.max(hists_norm[mkey][axn]) for axn in hists_norm[mkey].keys()]) for mkey in hists_norm]))
    vmin = np.log10(np.min([np.min([np.min(hists_norm[mkey][axn]) for axn in hists_norm[mkey].keys()]) for mkey in hists_norm]))
    vmin = max(vmin, vmax - 7.)
    
    massmins = sorted([float(key) for key in hists_main.keys()])

    linewidth = 1.
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.5, foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
     
    fig.suptitle(title, fontsize=fontsize + 2)
    
    
    for mi in range(nmassbins):
        ind = np.where(np.array(massbins)[:, 0] == massmins[mi])[0][0]
        text = r'$%.1f \, \endash \, %.1f$'%(massbins[ind][0], massbins[ind][1]) #r'$\log_{10} \,$' + binqn + r': 
        
        ax = laxes[mi]
        ax.text(0.5, 0.1, text, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='center', verticalalignment='bottom')
        ax.axis('off') 
        
    for mi in range(nmassbins):
        for ti in range(nprof):
            # where are we
            ax = axes[ti, mi]
            labelx = ti == nprof - 1
            labely = mi == 0
            
            if ti == 0:
                yq = 'weight'
            else:
                yq = axnl[ti - 1]
            mkey = str(massmins[mi])
            
            # set up axis
            setticks(ax, top=True, left=True, labelleft=labely, labelbottom=labelx, fontsize=fontsize)
            if labelx:
                ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            if labely:
                ax.set_ylabel(axlabels[yq], fontsize=fontsize)
            
            # plot stacked histogram
            edges_r = edges_main[mkey][yq][0] 
            if yq != 'weight':
                edges_y = edges_main[mkey][yq][1]
                hist = hists_main[mkey][yq]
                
                img, _1, _2 = pu.add_2dplot(ax, hist, [edges_r, edges_y], toplotaxes=(0, 1),\
                              log=True, usepcolor=True, pixdens=True,\
                              cmap=cmap, vmin=vmin, vmax=vmax, zorder=-2)
                perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles))
                mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                
                for pi in range(len(percentiles)):
                    ax.plot(mid_r, perclines[pi], color='white',\
                            linestyle=linestyles[pi], alpha=1.,\
                            path_effects=patheff_thick, linewidth=linewidth + 1)
                
            else:
                hist = hists_main[mkey][yq]
                
                ax.plot(edges_r, np.log10(hist), color='black',\
                            linestyle='solid', alpha=1.,\
                            path_effects=None, linewidth=linewidth + 1.5)
    # color bar 
    pu.add_colorbar(cax, img=img, vmin=vmin, vmax=vmax, cmap=cmap,\
                    clabel=clabel, fontsize=fontsize, orientation='vertical',\
                    extend='min')
    cax.set_aspect(10.)
    
    # sync y limits on plots
    for yi in range(nprof):
        ylims = np.array([axes[yi, mi].get_ylim() for mi in range(nmassbins)])
        miny = np.min(ylims[:, 0])
        maxy = np.max(ylims[:, 1])
        # intended for ion number densities
        miny = max(miny, maxy - 10.)
        [[axes[yi, mi].set_ylim(miny, maxy) for mi in range(nmassbins)]]
    for xi in range(nmassbins):
        xlims = np.array([axes[i, xi].get_xlim() for i in range(nprof)])
        minx = np.min(xlims[:, 0])
        maxx = np.max(xlims[:, 1])
        [axes[i, xi].set_xlim(minx, maxx) for i in range(nprof)]
    
    plt.savefig(file_out, format='pdf', box_inches='tight')
    
    
def plotsubsamplediffs_NEW(sample=(3, 6), allsamples=True):
    if allsamples:
        smpfill = ''
    else:
        smpfill = '_lesssamples'
    if sample == 3:
        dfilen = '/net/luttero/data2/specwizard_data/sample3_coldens_EW_subsamples.hdf5'
        fileout = '/net/luttero/data2/specwizard_data/sample3_specwizard-NEW_wbparfit_wsubsamples%s.pdf'%(smpfill)
    elif sample == (3, 6):
        dfilen = '/net/luttero/data2/specwizard_data/sample3-6_coldens_EW_subsamples.hdf5'
        fileout = '/net/luttero/data2/specwizard_data/sample3-6_specwizard-NEW_wbparfit_wsubsamples%s.pdf'%(smpfill)
        
    percentiles = [5., 50., 95.]
    logNspacing = 0.2
    linemin = 10
    ncols = 3
    ylabel = r'$\log_{10} \, \mathrm{EW} \; [\mathrm{m\AA}]$'
    xlabel = r'$\log_{10} \, \mathrm{N}(\mathrm{%s}) \; [\mathrm{cm}^{-2}]$'
    fontsize = 12
    linewidth = 1.5
    
    uselines = {'o7': ild.o7major,\
                'o8': ild.o8doublet,\
                'o6': ild.o6major,\
                'ne8': ild.ne8major,\
                'ne9': ild.ne9major,\
                'fe17': ild.fe17major,\
                }
    Nminmax =  {'o7':   (15.5, 18.3),\
                'o8':   (15.8, 17.5),\
                'o6':   (14.2, 17.0),\
                'ne9':  (15.7, 17.4),\
                'ne8':  (14.7, 16.4),\
                'fe17': (14.9, 16.6),\
                }
    logEWminmax = {'o7':   (0.8, 1.6),\
                   'o8':   (1.0, 1.7),\
                   'o6':   (2.0, 3.0),\
                   'ne9':  (0.6, 1.3),\
                   'ne8':  (2.1, 2.8),\
                   'fe17': (0.6, 1.3),\
                   }
    
    data = {}
    with h5py.File(dfilen, 'r') as fi:
        selections = list(fi.keys())
        toremove = [] 
        for _sel in selections:
            if _sel.decode().startswith('Header'):
                toremove.append(_sel)
                continue
            data[_sel] = {}
            for ionk in fi[_sel].keys():
                #print('%s/%s'%(selection, ionk))
                if ionk.startswith('specnum'):
                    continue
                grp = fi['%s/%s'%(_sel, ionk)]
                logN = np.array(grp['logN_cmm2'])
                EW = np.array(grp['EWrest_A'])
                blin = np.array(grp.attrs['bparfit_cmps_linEW'])[0] * 1e-5
                blog = np.array(grp.attrs['bparfit_cmps_logEW'])[0] * 1e-5
                ion = ionk.split('_')[0]
                data[_sel][ion] = {'logN': logN, 'EW': EW,\
                                   'blin': blin, 'blog': blog}
    for sel in toremove:
        selections.remove(sel)
          
    ions = {val for sub in data for val in data[sub]}
    ions = list(ions)
    ions.sort()
    nions = len(ions)
    
    selkeys = selections
    colors = {key: ioncolors[key.split('_')[0]] if key.split('_')[0] in ioncolors else 'black' for key in selkeys}
    #{selkeys[i]: 'C%i'%(i % 10) for i in range(len(selkeys))}
    _linestyles = ['solid'] * len(selkeys)
    linestyles = {selkeys[i]: _linestyles[i] for i in range(len(selkeys))}
    
    panelwidth = 2.5
    panelheight = 2.5
    legheight = 1.5
    wspace = 0.5
    hspace = 0.5
    nrows = nions // ncols
    
    fig = plt.figure(figsize=(panelwidth * ncols + wspace * (ncols - 1), panelheight * nrows + hspace * nrows + legheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows + 1, wspace=wspace, hspace=hspace)   
    axes = [fig.add_subplot(grid[ioni // ncols, ioni % ncols]) for ioni in range(nions)]
    lax = fig.add_subplot(grid[nrows, :])

    for ioni in range(nions):
        ax = axes[ioni]
        ion = ions[ioni]
        ax.set_xlabel(xlabel%(ild.getnicename(ion, mathmode=True)), fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        
        for key in selkeys:
            logN = data[key][ion]['logN']
            EW   = np.log10(data[key][ion]['EW']) + 3.
            if not (allsamples or ion in key or 'full' in key):
                continue
 
            _linewidth = linewidth  
            if ioni == 0:
                label = key
            else:
                label = None
            minN = np.min(logN) 
            maxN = np.max(logN)
            minN -= 1e-7 * np.abs(minN)
            maxN += 1e-7 * np.abs(maxN)
            bminN = np.floor(minN / logNspacing) * logNspacing
            bmaxN = np.ceil(maxN / logNspacing) * logNspacing
            Nbins = np.arange(bminN, bmaxN + 0.5 * logNspacing, logNspacing)
            Ncens = Nbins[:-1] + 0.5 * np.diff(Nbins)
            
            bininds = np.array(np.digitize(logN, Nbins))
            EWs_bin = [EW[np.where(bininds == i + 1)] for i in range(len(Ncens))]
            Ns_bin  = [logN[np.where(bininds == i + 1)] for i in range(len(Ncens))]
            percvals = np.array([np.percentile(EWs, percentiles) \
                                    if len(EWs) > 0 else \
                                    np.ones(len(percentiles)) * np.NaN
                                    for EWs in EWs_bin])
            
            whereplot = np.where(np.array([len(_EW) >= linemin for _EW in EWs_bin]))[0]
            plotmin = whereplot[0]
            plotmax = whereplot[-1]
            ax.fill_between(Ncens[plotmin : plotmax + 1],\
                            percvals[plotmin : plotmax + 1, 0],\
                            percvals[plotmin : plotmax + 1, 2],\
                            color=colors[key], alpha=0.1)
            ax.plot(Ncens[plotmin : plotmax + 1],\
                            percvals[plotmin : plotmax + 1, 1],\
                            color=colors[key], linestyle=linestyles[key],\
                            label=label, linewidth=_linewidth)
            
            Ns_sc = [val for bi in range(plotmin) + range(plotmax + 1, len(Ncens)) for val in Ns_bin[bi]]
            EWs_sc = [val for bi in range(plotmin) + range(plotmax + 1, len(Ncens)) for val in EWs_bin[bi]]
            ax.scatter(Ns_sc, EWs_sc, color=colors[key], alpha=0.1)
            
            blin = data[key][ion]['blin']
            blog = data[key][ion]['blog']
            
            lines = uselines[ion]
            EWslin = ild.linflatcurveofgrowth_inv(10**Ncens, blin * 1e5, lines)
            EWslog = ild.linflatcurveofgrowth_inv(10**Ncens, blog * 1e5, lines)
            if isinstance(lines, ild.SpecLine): 
                EWsthin = ild.lingrowthcurve_inv(10**Ncens, lines)
            else:
                EWsthin = np.sum([ild.lingrowthcurve_inv(10**Ncens, lines.speclines[lkey])\
                                 for lkey in lines.speclines],axis=0)
            ax.plot(Ncens, np.log10(EWsthin) + 3., color='gray', linestyle='dotted', label='opt. thin')
            ax.plot(Ncens, np.log10(EWslin) + 3., color=colors[key], label='lin. fit', linestyle='dashed')
            ax.plot(Ncens, np.log10(EWslog) + 3., color=colors[key], label='log fit', linestyle='dashdot')
            
            ax.set_xlim(Nminmax[ion])
            ax.set_ylim(logEWminmax[ion])
    lcs = []
    line = [[(0, 0)]]    
    # set up the proxy artist
    for ls in ['dashed', 'dashdot']:
        subcols = [mpl.colors.to_rgba(colors[key]) for key in colors]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    selhandles = [mlines.Line2D([], [], color=colors[key], linestyle='solid', label=key.split('_')[0] + ' sample') 
                  for key in colors]
    sellabels = [key.split('_')[0] + ' sample' for key in colors]
    linhandle = [mlines.Line2D([], [], color='gray', linestyle='dotted', label='opt. thin') ]
    linlabel = ['opt. thin']
    lcs_labels = ['lin. fit', 'log fit']
    lax.legend(selhandles + lcs + linhandle, sellabels + lcs_labels + linlabel,\
               handler_map={type(lc): HandlerDashedLines()},\
               fontsize=fontsize, ncol=ncols, loc='upper center', bbox_to_anchor=(0.5, 1.))
    lax.axis('off')
    
    plt.savefig(fileout, format='pdf', bbox_inches='tight')       
            
 
def plotdiffs_bparglobalperc(bparfit='bpar_global_perc_set1.txt'):
    fontsize=12
    fdir = '/net/luttero/data2/specwizard_data/'
    df = pd.read_csv(fdir + bparfit, sep='\t')
    
    colors = {50: 'red', 100: 'green', 200: 'blue'}
    yoff1 = {50: 0.015, 100: 0.0, 200: -0.015}
    markers = {0.1: 'o', 0.15: 'p', 0.2: '^'}
    yoff2  = {0.1: 0.01, 0.15: 0.0, 0.2: -0.01}
    sizes  = {5.0: 15, 50.0: 30, 95.0: 45}
    ionpos = {'o6': 0.9, 'ne8': 0.8, 'o7': 0.7, 'ne9': 0.6, 'o8': 0.5, 'fe17': 0.4}
    ylim = (0.35, 0.95)

    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(nrows=2, ncols=1, hspace=0.3, height_ratios=[4.0, 2.0])
    ax = fig.add_subplot(grid[0])
    lax = fig.add_subplot(grid[1])
    
    for rowi in df.index:
        ion  = df.at[rowi, 'ion']
        perc = df.at[rowi, 'percentile']
        #bs   = df.at['binsize_logN', rowi]
        bn   = df.at[rowi, 'maxinbin']
        mn   = df.at[rowi, 'minlogEWdiff']
        bval = df.at[rowi, 'b_kmps']
        
        pos = ionpos[ion] + yoff1[bn] + yoff2[mn]
        ax.scatter(bval, pos, s=sizes[perc], c=colors[bn], marker=markers[mn], alpha=0.3)
    
    ax.tick_params(axis='x', which='both', top=True, labelsize=fontsize-1, direction='in')
    ax.minorticks_on()
    ax.set_xlabel(r'$b \; [\mathrm{km} \, \mathrm{s}^{-1}]$', fontsize=fontsize)
    
    ax.tick_params(axis='y', which='both', right=False, labelsize=fontsize)
    ax.set_ylim(ylim)
    tickpos = [ionpos[ion] for ion in ionpos]
    tickpos.sort()
    ticklabels = [ild.getnicename(ion) for ion in [''.join([ion if ionpos[ion] == pos else '' for ion in ionpos]) for pos in tickpos]]
    
    ax.set_yticks(tickpos)
    ax.set_yticklabels(ticklabels)
    ax.tick_params(which='minor', axis='y', left=False, right=False)
    
    mklist = list(markers.keys())
    mklist.sort
    markers_handles = [mlines.Line2D([], [], color='black', marker=markers[minN], linestyle='None', markersize=sizes[50.] / 5.) \
               for minN in mklist] 
    markers_labels = [r'min. $\Delta \, \log_{10} \, \mathrm{EW} = %.2f$'%(minN) for minN in mklist]
    
    cllist = list(colors.keys())
    cllist.sort()
    colors_handles = [mlines.Line2D([], [], color=colors[cl], marker='d', linestyle='None', markersize=sizes[50.] / 5.) \
               for cl in cllist] 
    colors_labels = [r'max. sl. in bin = %i'%(cl) for cl in cllist]
    
    szlist = list(sizes.keys())
    szlist.sort()
    sizes_handles = [mlines.Line2D([], [], color='black', marker='d', linestyle='None', markersize=sizes[sz] / 5.) \
               for sz in szlist] 
    sizes_labels = [r'percentile %.0f'%(sz) for sz in szlist]
    
    lax.axis('off')
    lax.legend(markers_handles + colors_handles + sizes_handles,\
               markers_labels  + colors_labels  + sizes_labels,\
               fontsize=fontsize, loc='upper center', ncol=2)
    
    plt.savefig(fdir + bparfit[:-4] + '.pdf', format='pdf', bbox_inches='tight')    



def plot_NEW_dw_diffs(fontsize=fontsize,\
                      EWdiff=False, deltaEW_at_newEW=False,\
                      deltaEW_at_oldEW=False,\
                      usevwindows=False): 
    '''
    EWdiff:  plot the EW difference as a function of column density
    deltaEW_at_newEW: plot the EW difference (color) as a function of the DW
                      N, EW
    deltaEW_at_oldEW: plot the EW difference (color) as a function of the 
                      Gaussian N, EW
    usevwindows:      make those plots using total N, EW or the paper2 vwindows
    '''
    
    mdir = '/net/luttero/data2/specwizard_data/voigt_gaussian_diffs_paper2ions/'
    if EWdiff:
        outname = 'EWdiff_same_sightline_for_Gaussian_N'
    elif deltaEW_at_newEW:       
        outname = 'EWdiff_for_Gaussian_N-EW'
    else:
        outname = 'EWdiff_for_Voigt_N-EW'
    if usevwindows:
        outname = mdir + outname + '_vwindows-paper2.pdf'
    else:
        outname = mdir + outname + '_100cMpc.pdf'
    
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    datafile = '/net/luttero/data2/specwizard_data/' +\
               'sample3-6_coldens_EW_vwindows_subsamples.hdf5'
    
    uselines = {'o7': ild.o7major,\
                'o8': ild.o8doublet,\
                'o6': ild.o6major,\
                'ne8': ild.ne8major,\
                'ne9': ild.ne9major,\
                'fe17': ild.fe17major,\
                }
    Nminmax =  {'o7':   (14.5, 18.3),\
                'o8':   (14.8, 17.5),\
                'o6':   (12.8, 17.0),\
                'ne9':  (14.9, 17.4),\
                'ne8':  (13.5, 16.4),\
                'fe17': (14.2, 16.6),\
                }
    logEWminmax = {'o7':   (0.0, 1.9),\
                   'o8':   (0.0, 1.7),\
                   'o6':   (1.0, 3.2),\
                   'ne9':  (0.0, 1.3),\
                   'ne8':  (1.3, 2.8),\
                   'fe17': (0.0, 1.4),\
                   }
    Tmax_CIE = {'o6':   10**5.5,\
            'ne8':  10**5.8,\
            'o7':   10**5.9,\
            'ne9':  10**6.2,\
            'o8':   10**6.4,\
            'fe17': 10**6.7,\
            }
    
    bvals_CIE = {ion: np.sqrt(2. * c.boltzmann * Tmax_CIE[ion] / \
                              (ionh.atomw[string.capwords(ol.elements_ion[ion])] * c.u)) \
                      * 1e-5
                 for ion in Tmax_CIE}
    # v windows
    if usevwindows:
        bfits_gs = {'o6': 28.,\
                    'o7': 82.,\
                    'o8': 112.,\
                    'ne8': 37.,\
                    'ne9': 81.,\
                    'fe17': 90.,\
                    }
        bfits_vt = {'o6': 28.,\
                    'o7': 83.,\
                    'o8': 125.,\
                    'ne8': 37.,\
                    'ne9': 82.,\
                    'fe17': 92.,\
                    }
    # full sightlines
    else:
        bfits_gs = {'o6': 34.,\
                 'o7': 90.,\
                 'o8': 158.,\
                 'ne8': 41.,\
                 'ne9': 90.,\
                 'fe17': 102.,\
                 }
        bfits_gs = {'o6': 34.,\
                 'o7': 91.,\
                 'o8': 168.,\
                 'ne8': 41.,\
                 'ne9': 91.,\
                 'fe17': 105.,\
                 }
    if deltaEW_at_newEW:
        bfits = bfits_vt
    else:
        bfits = bfits_gs
        
    # half range sizes
    vwindows_ion = {'o6': 600.,\
                    'ne8': 600.,\
                    'o7': 1600.,\
                    'o8': 1600.,\
                    'fe17': 1600.,\
                    'ne9': 1600,\
                    }
    samplegroups_ion = {ion: '{ion}_selection'.format(ion=ion) \
                              for ion in vwindows_ion}
    
    with h5py.File(datafile, 'r') as df:
        coldens_gs = {}
        EWs_gs = {}
        coldens_vt = {}
        EWs_vt = {}
        vwlabels = {}
        for ion in ions:
            samplegroup = samplegroups_ion[ion] + '/'
            if usevwindows:
                vwindow = vwindows_ion[ion]
                vwlabels[ion] = vwindow
            else:
                vwindow = cosmopars_ea_27['boxsize'] * cosmopars_ea_27['a'] \
                          / cosmopars_ea_27['h'] * c.cm_per_mpc *\
                          cu.Hubble(cosmopars_ea_27['z'],\
                                    cosmopars=cosmopars_ea_27) \
                          * 1e-5
                vwlabels[ion] = vwindow
            if usevwindows:
                spath_gs = 'vwindows_maxtau/Deltav_{dv:.3f}/'.format(dv=vwindow)
                spath_vt = 'vwindows_maxtau_dw/Deltav_{dv:.3f}/'.format(dv=vwindow)
                
                epath_gs = spath_gs + 'EW/'
                cpath_gs = spath_gs + 'coldens/'
                epath_vt = spath_vt + 'EW/'
                cpath_vt = spath_vt + 'coldens/'
                
                epath_gs = samplegroup + epath_gs
                cpath_gs = samplegroup + cpath_gs
                epath_vt = samplegroup + epath_vt
                cpath_vt = samplegroup + cpath_vt
            else:
                epath_gs = samplegroup + 'EW_tot/'
                epath_vt = samplegroup + 'EW_tot_dw/'
                cpath_gs = samplegroup + 'coldens_tot/'
                cpath_vt = samplegroup + 'coldens_tot/'
                
            #print(cpath_gs + ion)
            #print(epath_gs + ion)
            #print(cpath_vt + ion)
            #print(epath_vt + ion)
            coldens_gs[ion] = np.array(df[cpath_gs + ion])
            EWs_gs[ion] = np.array(df[epath_gs + ion])
            coldens_vt[ion] = np.array(df[cpath_vt + ion])
            EWs_vt[ion] = np.array(df[epath_vt + ion])
    
    bvals_indic = [10., 20., 50., 100., 200.]
    percentiles = [2., 10., 50., 90., 98.]
    logNspacing = 0.1
    linemin = 50
    ncols = 3
    if EWdiff:
        ylabel = '$\\log_{{10}} \\, \\mathrm{{EW}}_{{\\mathrm{{voigt}}}} \\, / \\, \\mathrm{{EW}}_{{\\mathrm{{gauss}}}}$'
    else:
        ylabel = r'$\log_{10} \, \mathrm{EW} \; [\mathrm{m\AA}]$'
    clabel = '$\\log_{{10}} \\, \\mathrm{{EW}}_{{\\mathrm{{voigt}}}} \\, / \\, \\mathrm{{EW}}_{{\\mathrm{{gauss}}}}$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    bbox = {'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'none'}
    
    alpha = 0.3
    size_data = 3.
    linewidth = 2.
    path_effects = [mppe.Stroke(linewidth=linewidth, foreground="black"),\
                    mppe.Stroke(linewidth=linewidth - 0.5)]
    color_data = ['gray', 'dimgray']
    color_bbkg = 'C5'
    color_lin  = 'forestgreen'
    color_cie  = 'orange'
    color_fit  = 'blue'
    ls_data    = 'solid'
    ls_bbkg    = 'dotted'
    ls_lin     = 'dotted'
    ls_cie     = 'dashed'
    ls_fit     = 'dashdot'
    ftllabel = 'best-fit $b$'
    cielabel = '$b(\\mathrm{{T}}_{{\\max, \\mathrm{{CIE}}}})$'
    linlabel = 'lin. COG'
    bkglabel = 'var. $b \\; [\\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}]$'    
    
    nions = len(ions)
    
    panelwidth = 2.8
    panelheight = 2.8
    legheight = 1.0
    wspace = 0.25
    hspace = 0.2
    nrows = nions // ncols
    
    fig = plt.figure(figsize=(panelwidth * ncols + wspace * (ncols - 1),\
                              panelheight * nrows + hspace * (nrows - 1) +\
                              legheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows + 1, wspace=wspace, hspace=hspace, height_ratios=[panelheight] * nrows + [legheight])   
    axes = [fig.add_subplot(grid[ioni // ncols, ioni % ncols]) for ioni in range(nions)]
    lax = fig.add_subplot(grid[nrows, :])
    
    for ioni in range(nions):
        ax = axes[ioni]
        ion = ions[ioni]
        
        pu.setticks(ax, fontsize)
        if ioni % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if ioni // ncols == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        
        axlabel1 = '$\\Delta v = \\pm {dv:.0f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'
        axlabel2 = '$\\mathrm{{{ion}}}$'
        axlabel1 = axlabel1.format(dv=0.5 * vwlabels[ion])
        axlabel2 = axlabel2.format(ion=ild.getnicename(ion, mathmode=True))
        ax.text(0.03, 0.97, axlabel1, fontsize=fontsize - 1,\
                transform=ax.transAxes, verticalalignment='top',\
                horizontalalignment='left',\
                bbox=bbox)
        ax.text(0.03, 0.85, axlabel2, fontsize=fontsize,\
                transform=ax.transAxes, verticalalignment='top',\
                horizontalalignment='left',\
                bbox=None)
        if EWdiff:
            ax.set_xlim(*Nminmax[ion])
            
            N = coldens_gs[ion]
            delta = np.log10(EWs_vt[ion]) - np.log10(EWs_gs[ion]) 
            Nmin = np.min(N)
            Nmax = np.max(N)
            Nbmin = np.floor(Nmin / logNspacing) * logNspacing
            Nbmax = np.ceil(Nmax / logNspacing) * logNspacing
            Nbins = np.arange(Nbmin, Nbmax + 0.5 * logNspacing, logNspacing)
            Nbinc = Nbins + 0.5 * logNspacing
            Nbinc = np.append([Nbinc[0] - 0.5 * logNspacing], Nbinc)
            
            pvals, outliers, pinds =  pu.get_perc_and_points(N, delta, Nbins,\
                            percentiles=percentiles,\
                            mincount_x=linemin,\
                            getoutliers_y=True, getmincounts_x=True,\
                            x_extremes_only=True)
            pvals = pvals.T
            
            nrange = len(percentiles) // 2
            plotmed  = len(percentiles) % 2
            
            for i in range(nrange):
                pmin = pvals[i][pinds]
                pmax = pvals[len(pvals) - 1 - i][pinds]
                ax.fill_between(Nbinc[pinds], pmin, pmax,\
                                color=color_data[i], alpha=alpha)
            if plotmed:
                pmed = pvals[nrange]
                ax.plot(Nbinc, pmed, linestyle=ls_data,\
                        linewidth=linewidth, color=color_data[0],\
                        path_effects=path_effects)
            ax.scatter(outliers[0], outliers[1], c=color_data[0], alpha=alpha,\
                       s=size_data)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ymax = np.max(delta[N >= xlim[0]])
            ymin = np.max(delta[N >= xlim[0]])
            if ylim[-1] > ymax * 1.5:
                ylim = (min(ylim[0], - ymin * 1.5, ymin * 1.5, -0.05 * ymax),\
                        1.5 * ymax)
            
            yr = ylim[1] - ylim[0]
            xr = xlim[1] - xlim[0]
            label = True
            for bval in bvals_indic:
                if label:
                    _label = bkglabel
                else:
                    _label = None
                _vals = ild.linflatdampedcurveofgrowth_inv(10**Nbinc, bval * 1e5, uselines[ion]) \
                        / ild.linflatcurveofgrowth_inv_faster(10**Nbinc, bval * 1e5, uselines[ion])
                _vals = np.log10(_vals)
                ax.plot(Nbinc, _vals, linestyle=ls_bbkg, color=color_bbkg,\
                    linewidth=linewidth, label=_label, zorder=-1)
                label=False
                
                ## plots are already quit busy, so leave out b labels in the 
                ## cramped top region
                if _vals[-1] < ylim[1] and _vals[-1] > ylim[0]:
                    rot = np.tan((_vals[-2] - _vals[-5]) / (Nbinc[-2] - Nbinc[-5])\
                                 * xr / yr)
                    rot *= 180. / np.pi # rad -> deg
                    ax.text(xlim[1] - 0.015 * xr, _vals[-1] - 0.02 * yr,\
                            '{:.0f}'.format(bval),\
                            horizontalalignment='right', verticalalignment='top',\
                            color=color_bbkg, fontsize=fontsize - 2, zorder=-1,\
                            rotation=rot)
            ax.set_ylim(*ylim)
            
        else:    
            fitlabel = 'fit: $b = {b:.0f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'.format(b=bfits[ion])
            ax.text(0.97, 0.03, fitlabel, fontsize=fontsize - 1,\
                    transform=ax.transAxes, verticalalignment='bottom',\
                    horizontalalignment='right', bbox=bbox)
            
            ax.set_xlim(*Nminmax[ion])
            ax.set_ylim(*logEWminmax[ion])
            
            if deltaEW_at_newEW:
                N = coldens_vt[ion]
                EW = np.log10(EWs_vt[ion]) + 3. # Angstrom -> mA
                indf = ild.linflatcurveofgrowth_inv_faster
            elif deltaEW_at_oldEW:
                N = coldens_gs[ion]
                EW = np.log10(EWs_gs[ion]) + 3. # Angstrom -> mA
                indf = ild.linflatdampedcurveofgrowth_inv
            Nmin = np.min(N)
            Nmax = np.max(N)
            Nbmin = np.floor(Nmin / logNspacing) * logNspacing
            Nbmax = np.ceil(Nmax / logNspacing) * logNspacing
            Nbins = np.arange(Nbmin, Nbmax + 0.5 * logNspacing, logNspacing)
            Nbinc = Nbins + 0.5 * logNspacing
            Nbinc = np.append([Nbinc[0] - 0.5 * logNspacing], Nbinc)
            
            delta = np.log10(EWs_vt[ion]) - np.log10(EWs_gs[ion]) 
            img_c = ax.scatter(N, EW, c=delta, alpha=alpha,\
                               s=size_data)
            # add color bar in axes:
            axpos = ax.get_position()
            cax   = fig.add_axes([axpos.x0 + 0.25 * axpos.width,\
                                  axpos.y0 + 0.15 * axpos.height,\
                                  axpos.width * 0.7, axpos.height * 0.07])
            cbar = plt.colorbar(img_c, cax=cax, orientation='horizontal')
            cax.set_xlabel(clabel, fontsize=fontsize - 1)
            cax.xaxis.set_label_position('top') 
            
            linvals = ild.lingrowthcurve_inv(10**Nbinc, uselines[ion])
            linvals = np.log10(linvals) + 3
            ax.plot(Nbinc, linvals, linestyle=ls_lin, color=color_lin,\
                    linewidth=linewidth, label=linlabel)
            
            cievals = indf(10**Nbinc, bvals_CIE[ion] * 1e5, uselines[ion])
            cievals = np.log10(cievals) + 3
            ax.plot(Nbinc, cievals, linestyle=ls_cie, color=color_cie,\
                    linewidth=linewidth, label=cielabel, path_effects=path_effects)
            
            fitvals = indf(10**Nbinc, bfits[ion] * 1e5, uselines[ion])
            fitvals = np.log10(fitvals) + 3
            ax.plot(Nbinc, fitvals, linestyle=ls_fit, color=color_fit,\
                    linewidth=linewidth, label=ftllabel, path_effects=path_effects)
            
            label = True
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            for bval in bvals_indic:
                if label:
                    _label = bkglabel
                else:
                    _label = None
                _vals = indf(10**Nbinc, bval * 1e5, uselines[ion])
                _vals = np.log10(_vals) + 3
                ax.plot(Nbinc, _vals, linestyle=ls_bbkg, color=color_bbkg,\
                    linewidth=linewidth, label=_label, zorder=-1)
                label=False
                
                ## plots are already quit busy, so leave out b labels in the 
                ## cramped top region
                if _vals[-1] < ylim[1] and _vals[-1] > ylim[0]:
                    rot = np.tan((_vals[-2] - _vals[-5]) / (Nbinc[-2] - Nbinc[-5])\
                                 * xr / yr)
                    rot *= 180. / np.pi # rad -> deg
                    ax.text(xlim[1] - 0.015 * xr, _vals[-1] - 0.02 * yr,\
                            '{:.0f}'.format(bval),\
                            horizontalalignment='right', verticalalignment='top',\
                            color=color_bbkg, fontsize=fontsize - 2, zorder=-1,\
                            rotation=rot)
                
            dfn = '/net/luttero/data2/paper2/' + 'EWdiffs_dampingwings.hdf5'
            with h5py.File(dfn, 'r') as df:
                grp = df[ion]
                Nsample = np.array(grp['logNcm2'])
                EWgrid_lnf = np.log10(np.array(grp['EW_Angstrom_gaussian'])) + 3.
                EWgrid_dmp = np.log10(np.array(grp['EW_Angstrom_voigt'])) + 3.
                
            dEW = EWgrid_dmp - EWgrid_lnf
            
            xpoints = np.tile(Nsample, EWgrid_lnf.shape[0])
            if deltaEW_at_newEW:
                ypoints = EWgrid_lnf.flatten()
            else:
                ypoints = EWgrid_dmp.flatten()
            zpoints = dEW.flatten()
           
            EWpoints = np.linspace(ylim[0], ylim[1], 200)
            gridpoints = (Nsample[np.newaxis, :], EWpoints[:,None])
            diffvals = sint.griddata((xpoints, ypoints), zpoints, gridpoints,\
                                   method='linear')
            contours = ax.contour(Nsample, EWpoints, np.abs(diffvals),\
                                  levels=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2],\
                                  colors='red', zorder=2)
            ax.clabel(contours, inline=1, fontsize=fontsize - 2)
            
    print('Used b values {} km/s (low-to-high -> bottom to top in plot)'.format(bvals_indic))        
    lax.axis('off')
    hnd, lab = axes[0].get_legend_handles_labels()
    lax.legend(handles=hnd, labels=lab, fontsize=fontsize, loc='lower center',\
                         ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(outname, format='pdf', bbox_inches='tight')    
    
    
def plotconfmatrix_mstarmhalo(halocat='/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5',\
                 method='rounded1'):
    outname = '/net/luttero/data2/imgs/CGM/radprof/confusion_matrix_mstarcen_mhalo_%s-bins.pdf'%(method)
    
    #binround = 0.1
    m200cbins = np.array([-np.inf, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.0, 14.6])
    if method == 'edgemedian':
        mstarbins = np.array([     -np.inf,  7.0101724,  7.8082123,  8.664759 ,  9.578973 , 10.291372 ,\
                                10.716454 , 10.952257 , 11.219421 , 11.433342 , 11.7      ], dtype=np.float32)
    elif method == 'maxpurefrac-sum':
        mstarbins = np.array([   -np.inf,  7.099232,  7.939403,  8.682909,  9.680908, 10.303451,\
                              10.787625, 11.092827, 11.309834, 11.522154, 11.7     ],dtype=np.float32)
    elif method == 'maxpurefrac-prod':
        mstarbins = np.array([   -np.inf,  7.072308 ,  7.8946085,  8.725354 ,  9.675405 ,\
                              10.338363 , 10.790792 , 11.088814 , 11.331073 , 11.533567 , 11.7], dtype=np.float32)
    elif method == 'maxpurefrac-min':
        mstarbins = np.array([   -np.inf,  7.1493073,  7.8008804,  8.597658 ,  9.566083 ,
                              10.321982 , 10.776766 , 11.013796 , 11.289852 , 11.511742 , 11.7], dtype=np.float32)
    elif method == 'rounded1':
        mstarbins = np.array([   -np.inf,  7.1,  7.9,  8.7,  9.7, 10.3, 10.8, 11.1, 11.3, 11.5, 11.7], dtype=np.float32)
    elif method == 'rounded2':
        mstarbins = np.array([   -np.inf,  7.1,  7.8,  8.6,  9.6, 10.3, 10.8, 11.0, 11.3, 11.5, 11.7], dtype=np.float32)
        
    with h5py.File(halocat, 'r') as cat:
        m200c = np.log10(np.array(cat['M200c_Msun']))
        mstar = np.log10(np.array(cat['Mstar_Msun']))
    
    #m200cbins = np.array(m200cbins)
    #expand = (np.floor(np.min(m200c) / binround) * binround, np.ceil(np.max(m200c) / binround) * binround)
    #m200cbins = np.append(expand[0], m200cbins)
    #m200cbins = np.append(m200cbins, expand[1])
        
    xycounts, xe, ye = np.histogram2d(m200c, mstar, bins=[m200cbins, mstarbins])

    xynorm = xycounts.astype(float) / np.sum(xycounts, axis=0)[np.newaxis, :]
    cmap = 'viridis'
    vmin = 0.
    vmax = 1.
    fontsize = 12
    xmin = 9.2
    ymin = 6.6
    xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    ylabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{\star}} \; [\mathrm{M}_{\odot}]$'
    clabel = r'fraction at fixed $\mathrm{M}_{\star}$'
    
    plotbins_x = np.copy(m200cbins)
    plotbins_x[0] = max(plotbins_x[0], xmin)
    plotbins_y = np.copy(mstarbins)
    plotbins_y[0] = max(plotbins_y[0], ymin)
    
    fig = plt.figure(figsize=(5., 5.5))
    grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[10., 1.])
    ax = fig.add_subplot(grid[0])
    cax = fig.add_subplot(grid[1])
    
    img = ax.pcolormesh(plotbins_x, plotbins_y, xynorm.T, cmap=cmap,\
                        vmin=vmin, vmax=vmax, rasterized=True)
    
    ax.set_xticks(m200cbins[1:])
    ax.set_yticks(mstarbins[1:])
    ax.set_xlim((plotbins_x[0], plotbins_x[-1]))
    ax.set_ylim((plotbins_y[0], plotbins_y[-1]))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1, which='both')
    for label in ax.get_xmajorticklabels():
        label.set_rotation(45)

    ylim = ax.get_ylim() 
    offset = 0.025 * (ylim[1] - ylim[0])    
    for i in range(len(plotbins_x) - 1):
        for j in range(len(plotbins_y) - 1):
            xcoord = 0.5 * (plotbins_x[i] + plotbins_x[i + 1])
            ycoord = 0.5 * (plotbins_y[j] + plotbins_y[j + 1]) - 0.1 * offset
            num = int(xycounts[i, j])
            os = 0
            #if num > 1000:
            #    rotation = 90
            #else:
            #    rotation = 0
            if (num >= 1000 or xycounts[max(i - 1, 0), j] >= 1000 or xycounts[min(i + 1, xycounts.shape[0] - 1), j] >= 1000) or\
                (num >= 100 and xycounts[max(i - 1, 0), j] >= 100 and xycounts[min(i + 1, xycounts.shape[0] - 1), j] >= 100):
                os = offset if i % 2 else -1 * offset
            ax.text(xcoord, ycoord + os, num, fontsize=fontsize - 1.,\
                    color='white', rotation=0.,\
                    horizontalalignment='center', verticalalignment='center', 
                    path_effects = [mppe.Stroke(linewidth=1.5, foreground='black'), mppe.Normal()] )
    plt.colorbar(img, cax=cax, orientation='vertical')
    cax.set_ylabel(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1, which='both')
    cax.set_aspect(10.)
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    

def savetables_bensgadget2_cie():
    cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777,\
                   'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307,\
                   'z': 0.10063854175996956}
    cosmopars = cosmopars_ea_27
    ions = ['o{n}'.format(n=n) for n in range(1, 9)]

    outname = '/net/luttero/data2/paper2/' + \
              'cietables_oxygen_bensgadget2_z-{z}.hdf5'.format(**cosmopars)    
    with h5py.File(outname, 'w') as fo:
        hed = fo.create_group('Header')
        csm = hed.create_group('cosmopars')
        for key in cosmopars:
            csm.attrs.create(key, cosmopars[key])
        for ion in ions:
            logionbal, lognHcm3, logTK = m3.findiontables_bensgadget2(ion, cosmopars['z'])
            cievals = logionbal[-1, :]
            nval = lognHcm3[-1]
            
            igrp = fo.create_group(ion)
            igrp.create_dataset('logTK', data=logTK)
            igrp.create_dataset('logionbal', data=cievals)
        hed.attrs.create('info', np.string_('log ion fractions at max. tabulated nH and redshift given by cosmopars'))    
        hed.attrs.create('lognHcm3', nval)    
            

def plot_ionfracs_firstlook(addedges=(0.1, 1.), var='focus'):
    '''
    var: 'focus' for o6, o7, o8, ne8, ne9, fe17
         'oxygen' for all the oxygen species
    '''
    fontsize = 12
    
    filename_in = ol.pdir + 'ionfracs_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb.hdf5'%(str(addedges[0]), str(addedges[1]))
    outname = '/net/luttero/data2/imgs/CGM/3dprof/' + 'ionfracs_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s.pdf'%(str(addedges[0]), str(addedges[1]), var)
    m200cbins = np.array(list(np.arange(11., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])
    percentiles = [10., 50., 90.]
    alpha = 0.3
    lw = 2
    xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    ylabel = r'CGM ion fraction'
    
    iondata = {}
    with h5py.File(filename_in, 'r') as fd:
        cosmopars = {key: item for key, item in fd['Header/cosmopars'].attrs.items()}
        m200cvals = np.log10(np.array(fd['M200c_Msun']))
        fkeys = list(fd.keys())
        for key in ['Header', 'M200c_Msun', 'galaxyids']:
            if key in fkeys:
                fkeys.remove(key)
        for key in fkeys:
            grp = fd[key]
            _ions = grp.attrs['ions']
            try: # only one ion
                _ions = [_ions.decode()]
            except: # list/array of ions
                _ions = [_ion.decode() for _ion in _ions]
            allfracs = np.array(grp['fractions'])
            for ii in range(len(_ions)):
                iondata[_ions[ii]] = allfracs[:, ii]
            
        basesel = np.all(np.array([np.isfinite(iondata[ion]) for ion in iondata]), axis=0) # issues from a low-mass halo: probably a very small metal-free system
        m200cvals = m200cvals[basesel]
        for ion in iondata:
            iondata[ion] = iondata[ion][basesel]
        
    bininds = np.digitize(m200cvals, m200cbins)
    bincens = m200cbins[:-1] + 0.5 * np.diff(m200cbins)
    #bincens[-1] = np.median(m200cvals[bininds == len(m200cbins) - 1])
    #print(bincens)
    T200cvals = T200c_hot(10**bincens, cosmopars)
    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = grid = gsp.GridSpec(ncols=2, nrows=2, hspace=0.0, wspace=0.1, width_ratios=[5., 1.], height_ratios=[0.7, 2.])
    ax  = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[0, 0])
    lax = fig.add_subplot(grid[:, 1])
    
    extracolors = {'o1': 'navy', 'o2': 'skyblue', 'o3': 'olive', 'o4': 'darksalmon', 'o5': 'darkred'}
    if var == 'focus':
        plotions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    elif var == 'oxygen':
        plotions = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    setticks(ax, fontsize=fontsize) 
    ax2.set_ylabel('CIE fraction', fontsize=fontsize)
    if var == 'focus':
        ax.set_yscale('log')
        ax2.set_yscale('log')
        ax.set_ylim(1e-4, 0.7)
        ax2.set_ylim(1e-4, 1.3)
    else:
        ax.set_ylim(0., 1.)
        ax2.set_ylim(0., 1.)
    prev_halo = np.zeros(len(bincens))
    prev_cie = np.zeros(len(bincens))
    
    for ion in plotions:
        _iondata = iondata[ion]
        _color = ioncolors[ion] if ion in ioncolors else extracolors[ion]
        
        if var == 'focus':
            percvals = np.array([np.percentile(_iondata[bininds == i], percentiles) for i in range(1, len(m200cbins))]).T
            ax.plot(bincens, percvals[1], label=r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)), color=_color, linewidth=lw)
            ax.fill_between(bincens, percvals[0], percvals[2], color=_color, alpha=alpha)
            tablevals = m3.find_ionbal(cosmopars['z'], ion, {'logT': np.log10(T200cvals), 'lognH': np.ones(len(T200cvals)) * 6.}) # extreme nH -> highest tabulated values used
            ax2.plot(np.log10(T200cvals), tablevals, color=_color, linewidth=lw)  
        else:
            avgs = np.array([np.average(_iondata[bininds == i]) for i in range(1, len(m200cbins))])
            tablevals = m3.find_ionbal_bensgadget2(cosmopars['z'], ion, {'logT': np.log10(T200cvals), 'lognH': np.ones(len(T200cvals)) * 6.}) # extreme nH -> highest tabulated values used
            
            ax.fill_between(bincens, prev_halo, prev_halo + avgs, color=_color, label=r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)))
            prev_halo += avgs
            ax2.fill_between(np.log10(T200cvals), prev_cie, prev_cie + tablevals, color=_color)
            prev_cie += tablevals
    
    # set T ticks
    mlim = 10**np.array(ax.get_xlim())
    tlim = np.log10(T200c_hot(mlim, cosmopars))
    ax2.set_xlim(tuple(tlim))
    ax2.set_xlabel(r'$\log_{10} \, \mathrm{T}_{\mathrm{200c}} \; [\mathrm{K}]$', fontsize=fontsize)
    setticks(ax2, fontsize=fontsize, labelbottom=False, labeltop=True)
    ax2.xaxis.set_label_position('top') 
    
    handles, lables = ax.get_legend_handles_labels()
    
    if var == 'focus':
        legelts = [mpatch.Patch(facecolor='gray', alpha=alpha, label='%.1f %%'%(percentiles[2] - percentiles[0]))] + \
                  [mlines.Line2D([], [], color='gray', label='median')]
    else:
        legelts = []
    lax.legend(handles=handles + legelts, ncol=1, fontsize=fontsize, bbox_to_anchor=(0.02, 0.98), loc='upper left')
    lax.axis('off')
    
    plt.savefig(outname, format='pdf', box_inches='tight')
    
    
def plot_masscontr_firstlook(addedges=(0.0, 1.), var='Mass'):
    '''
    var: 'Mass' for total mass
         'oxygen', 'neon', or 'iron' for metal mass
    '''
    fontsize = 12
    
    filename_in = ol.pdir + 'massdist-baryoncomp_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb.hdf5'%(str(addedges[0]), str(addedges[1]))
    outname = '/net/luttero/data2/imgs/CGM/3dprof/' + 'masscontr_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s.pdf'%(str(addedges[0]), str(addedges[1]), var)
    m200cbins = np.array(list(np.arange(11., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])
    percentiles = [10., 50., 90.]
    alpha = 0.3
    lw = 2
    xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    ylabel = r'mass fraction'
    
    if var == 'Mass':
        groupname = 'massdist_Mass'
        catcol = {'BHs': ['BHs'],\
                  'DM': ['DM'],\
                  'gas': ['gas'],\
                  'stars': ['stars'],\
                  'ISM': ['gas_SF_T--inf-5.0', 'gas_SF_T-5.0-5.5',\
                          'gas_SF_T-5.5-7.0', 'gas_SF_T-7.0-inf'],\
                  r'CGM $<5.5$': ['gas_nonSF_T--inf-5.0', 'gas_nonSF_T-5.0-5.5'],\
                  r'CGM $5.5 \endash 7$': ['gas_nonSF_T-5.5-7.0'],\
                  r'CGM $> 7$': ['gas_nonSF_T-7.0-inf']}
        addcol = {'total': ['BHs', 'gas', 'stars', 'DM'],\
                  'CGM': [r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$'],\
                  'baryons': ['BHs', 'gas', 'stars'],\
                  'gas-subsum': ['ISM', r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$']}
    else:
        groupname = 'massdist_%s'%(var)
        catcol = {'gas': ['gas-%s'%(var)],\
                  'stars': ['stars-%s'%(var)],\
                  'ISM': ['gas-%s_SF_T--inf-5.0'%(var), 'gas-%s_SF_T-5.0-5.5'%(var),\
                          'gas-%s_SF_T-5.5-7.0'%(var), 'gas-%s_SF_T-7.0-inf'%(var)],\
                  r'CGM $<5.5$': ['gas-%s_nonSF_T--inf-5.0'%(var), 'gas-%s_nonSF_T-5.0-5.5'%(var)],\
                  r'CGM $5.5 \endash 7$': ['gas-%s_nonSF_T-5.5-7.0'%(var)],\
                  r'CGM $> 7$': ['gas-%s_nonSF_T-7.0-inf'%(var)]}
        addcol = {'total': ['gas', 'stars'],\
                  'CGM': [r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$'],\
                  'gas-subsum': ['ISM', r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$']}
        
    with h5py.File(filename_in, 'r') as fd:
        cosmopars = {key: item for key, item in fd['Header/cosmopars'].attrs.items()}
        m200cvals = np.log10(np.array(fd['M200c_Msun']))
        grp = fd[groupname]
        arr_all = np.array(grp['mass'])
        collabels = list(grp.attrs['categories'])
        collabels = np.array([lab.decode() for lab in collabels])
        catind = {key: np.array([np.where(collabels == subn)[0][0] for subn in catcol[key]]) \
                       for key in catcol}
        massdata = {key: np.sum(arr_all[:, catind[key]], axis=1) for key in catcol}
        _sumdata = {key: np.sum([massdata[subkey] for subkey in addcol[key]], axis=0) for key in addcol}
        massdata.update(_sumdata)
        # check
        if not np.all(np.isclose(massdata['gas'], massdata['gas-subsum'])):
            raise RuntimeError('The gas subcategory masses do not add up to the total gas mass for all halos')
        
        #basesel = np.all(np.array([np.isfinite(iondata[ion]) for ion in iondata]), axis=0) # issues from a low-mass halo: probably a very small metal-free system
        #m200cvals = m200cvals[basesel]
        #for ion in iondata:
        #    iondata[ion] = iondata[ion][basesel]
        
    bininds = np.digitize(m200cvals, m200cbins)
    bincens = m200cbins[:-1] + 0.5 * np.diff(m200cbins)
    #bincens[-1] = np.median(m200cvals[bininds == len(m200cbins) - 1])
    #print(bincens)
    #T200cvals = T200c_hot(10**bincens, cosmopars)
    
    fig = plt.figure(figsize=(5.5, 6.))
    grid = grid = gsp.GridSpec(ncols=1, nrows=2, hspace=0.0, wspace=0.1, height_ratios=[4., 2.])
    ax  = fig.add_subplot(grid[0, 0])
    #ax2 = fig.add_subplot(grid[0, 0])
    lax = fig.add_subplot(grid[1, 0])
    
    colors = {'BHs': 'black',\
              'DM':  'gray',\
              'gas': 'C1',\
              'stars': 'C8',\
              'ISM': 'C3',\
              'CGM': 'C0',\
              'baryons': 'gray',\
              r'CGM $<5.5$': 'C9',\
              r'CGM $5.5 \endash 7$': 'C4',\
              r'CGM $> 7$': 'C6'}
    linestyles = {'BHs': 'solid',\
              'DM':  'solid',\
              'gas': 'solid',\
              'stars': 'solid',\
              'baryons': 'solid',\
              'BHs': 'solid',\
              'ISM': 'solid',\
              'CGM': 'solid',\
              r'CGM $<5.5$': 'dotted',\
              r'CGM $5.5 \endash 7$': 'solid',\
              r'CGM $> 7$': 'dotted'}
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    setticks(ax, fontsize=fontsize) 

    ax.set_yscale('log')
    if var == 'Mass':
        ax.set_ylim(1e-4, 0.2)
    else:
        ax.set_ylim(2e-3, 1.)
    #if 'DM' in catcol:
    #    fdm = 1. - cosmopars['omegab'] / cosmopars['omegam']
    #    ax.axhline(fdm, linewidth=lw, color=colors['DM'], label=r'$1 - \Omega_{\mathrm{b}} \,/\, \Omega_{\mathrm{m}}$')
    if var == 'Mass':
        fb = cosmopars['omegab'] / cosmopars['omegam']
        ax.axhline(fb, linewidth=lw, linestyle='dashed', color=colors['DM'], label=r'$\Omega_{\mathrm{b}} \,/\, \Omega_{\mathrm{m}}$')
        
    for label in massdata:
        if label in ['DM', 'total', 'gas-subsum', 'gas', r'CGM $<5.5$', r'CGM $> 7$', 'BHs']:
            continue
        _massdata = massdata[label] / massdata['total']
        _color = colors[label]
        
        percvals = np.array([np.percentile(_massdata[bininds == i], percentiles) for i in range(1, len(m200cbins))]).T
        ax.plot(bincens, percvals[1], label=label, color=_color, linewidth=lw, linestyle=linestyles[label])
        if label not in ['baryons', r'CGM $5.5 \endash 7$']:
            ax.fill_between(bincens, percvals[0], percvals[2], color=_color, alpha=alpha)
    
    handles, lables = ax.get_legend_handles_labels()
    

    legelts = [mpatch.Patch(facecolor='tan', alpha=alpha, label='%.1f %%'%(percentiles[2] - percentiles[0]))] + \
              [mlines.Line2D([], [], color='tan', label='median')]

    lax.legend(handles=handles + legelts, ncol=3, fontsize=fontsize, bbox_to_anchor=(0.5, 0.6), loc='upper center')
    lax.axis('off')
    
    plt.savefig(outname, format='pdf', box_inches='tight')
    
    
def plot_conv_cddfs(ion='ne9', comp='pixres', rel=True):
    '''
    ion: 'ne9' or 'fe17'
    comp: 'pixres', 'slices', 'boxres', 'boxsize'
    rel: True/False: plot relative to default
    '''    
    datadir = ol.pdir
    outdir = '/net/luttero/data2/imgs/cddfs_nice/'
    if rel:
        srel = 'rel'
        ylabel = r'$\left(\mathrm{d}n(\mathrm{N}) \,/\, \mathrm{d}X \right) \,/\,${}'
    else:
        srel = 'abs'
        r'$\partial^2n \,/\, \partial \log_{10} \mathrm{N} \, \partial X$'
    imgname = outdir + 'cddf_conv_test_{}_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_z-projection_T4EOS_{}.pdf'.format(ion, comp, srel)
    
    if comp == 'pixres':
        title = 'pixel size'
        basefile = {'3.125 ckpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-2_offset-0_resreduce-1.hdf5'.format(ion)}
        compfiles = {'6.25 ckpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-2_offset-0_resreduce-2.hdf5'.format(ion),\
                     '12.5 ckpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-2_offset-0_resreduce-4.hdf5'.format(ion),\
                     '25 ckpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-2_offset-0_resreduce-8.hdf5'.format(ion),\
                     }
        yrel = (0.1, 1.5)
    elif comp == 'slices':
        title = 'slice thickness'
        basefile = {'6.25 cMpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-2_offset-0_resreduce-1.hdf5'.format(ion)}
        compfiles = {'3.125 cMpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-1_offset-0_resreduce-1.hdf5'.format(ion),\
                     '12.5 cMpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-4_offset-0_resreduce-1.hdf5'.format(ion),\
                     '25 cMpc': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-8_offset-0_resreduce-1.hdf5'.format(ion),\
                     }
        yrel = (0.5, 2.)
    elif comp == 'boxsize':
        title = 'box size'
        basefile = {'L100N1504': 'cddf_coldens_{}_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_3.125slice_zcen-all_z-projection_T4EOS_add-2_offset-0_resreduce-1.hdf5'.format(ion)}
        compfiles = {'L050N0752': 'cddf_coldens_{}_L0050N0752_27_test3.4_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_add-1_offset-0_resreduce-1.hdf5'.format(ion),\
                     'L025N0376': 'cddf_coldens_{}_L0025N0376_27_test3.4_PtAb_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_add-1_offset-0_resreduce-1.hdf5'.format(ion),\
                     }
        yrel = (0.05, 20.)
    elif comp == 'boxres':
        title = 'sim. resolution'
        basefile = {'L025N0376-Ref': 'cddf_coldens_{}_L0025N0376_27_test3.4_PtAb_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_add-1_offset-0_resreduce-1.hdf5'.format(ion)}
        compfiles = {'L025N0752-Ref': 'cddf_coldens_{}_L0025N0752_27_test3.4_PtAb_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_add-1_offset-0_resreduce-1.hdf5'.format(ion),\
                     'L025N0752-Recal': 'cddf_coldens_{}_L0025N0752RECALIBRATED_27_test3.4_PtAb_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_add-1_offset-0_resreduce-1.hdf5'.format(ion),\
                     }
        yrel = (0.05, 30.)
    basekey = list(basefile.keys())[0]
    if rel:
        ylabel = r'$\left(\partial n(\mathrm{N}) \,/\, \partial X \right) \,/\,$' + basekey
    else:
        ylabel = r'$\partial^2n \,/\, \partial \log_{10} \mathrm{N} \, \partial X$'
        
    kw_hist = 'histogram'
    kw_edge = 'edges'
    
    fig = plt.figure(figsize=(5.5, 5))
    ax = fig.add_subplot(1, 1, 1)
    fontsize = 12
    
    ax.set_xlabel('$\\log_{{10}} \\, \\mathrm{{N}}(\mathrm{{{}}}) \; [\\mathrm{{cm}}^{{-2}}]$'.format(ild.getnicename(ion, mathmode=True)), fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_yscale('log')
    ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=fontsize - 1.)
    ax.minorticks_on()
    ax.set_xlim(12., 17.)
    if rel:
        ax.set_ylim(*yrel)
    else:
        ax.set_ylim(1e-6, 2e2)
    
    compfiles.update(basefile)
    plotlines = {}
    for key in compfiles:
        with h5py.File(datadir + compfiles[key], 'r') as fi:
            try:
                hist = np.array(fi[kw_hist])
                edges = np.array(fi[kw_edge])
                if edges[0] == -np.inf:
                    edges[0] = 2. * edges[1] - edges[2]
                if edges[-1] == np.inf:
                    edges[-1] = 2. * edges[-2] - edges[-3]
                diff = np.diff(edges)
                dX = fi['Header'].attrs['dX']
                
                cens = edges[:-1] + 0.5 * diff
                vals = hist / diff / dX
                
                plotlines[key] = {'x': cens, 'y': vals}
            except KeyError as err:
                print('Error arose for file {}'.format(compfiles[key]))
                raise err
    keys = sorted(list(compfiles.keys()), key=lambda st: st.split(' ')[0])
    for key in keys:
        xv = plotlines[key]['x']
        yv = np.copy(plotlines[key]['y'])
        if rel:
            yv = yv / plotlines[basekey]['y']
        ax.plot(xv, yv, linewidth=2, label=key)
    
    ax.legend(fontsize=fontsize, loc='lower left')
    ax.text(0.98, 0.98, ild.getnicename(ion, mathmode=False), fontsize=fontsize,\
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes)
    
    plt.savefig(imgname, format='pdf', box_inches='tight')
        
###############################################################################
#                  nice plots for the paper: simplified                       #
###############################################################################
    
#### CDDFs

def plot_cddfs_nice(ions=None, fontsize=fontsize, imgname=None, techvars=[0]):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    if ions is None:
        ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']

    if imgname is None:
        imgname = 'cddfs_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvars-%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars])))
    if '/' not in imgname:
        imgname = mdir + imgname
    if imgname[-4:] != '.pdf':
        imgname = imgname + '.pdf'


        
    if isinstance(ions, str):
        ions = [ions]
    
    ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    #clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    linewidths = {0: 2}
    
    linestyles = {0: 'solid'}
    
    alphas = {0: 1.}
    
    masknames1 = ['nomask']
    masknames = masknames1 #{0: {ion: masknames1 for ion in ions}}
    
    #legendnames_techvars = {0: r'$r_{\perp} < R_{\mathrm{200c}}$, excl., cen'}
    
#    panelwidth = 2.5
#    panelheight = 2.
#    legheight = 0.6
#    figwidth = numcols * panelwidth + 0.6 
#    figheight = numcols * panelheight + 0.2 * numcols + legheight
#    fig = plt.figure(figsize=(figwidth, figheight))
#    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.2, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=[panelheight] * numrows + [legheight])
#    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
#    cax  = fig.add_subplot(grid[:numrows, numcols])
#    lax  = fig.add_subplot(grid[numrows, :])
    
    fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.0), gridspec_kw={'wspace': 0.0})
    
    hists = {}
    cosmopars = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        dXtot[var] = {}
        dztot[var] = {}
        dXtotdlogN[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                if _bins[0] == -np.inf:
                    _bins[0] = -100.
                if _bins[-1] == np.inf:
                    _bins[-1] = 100.
                bins[var][ion] = _bins
                
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
                
                masks = masknames
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                
                examplemaskdir = list(fi['masks'].keys())[0]
                examplemask = list(fi['masks/%s'%(examplemaskdir)].keys())[0]
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                dXtot[var][ion] = mc.getdX(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dztot[var][ion] = mc.getdz(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])

        assert checksubdct_equal(cosmopars[var])
                     
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
   #     raise RuntimeError('total histograms from different files do not match')
        
    
    ax1.set_xlim(12.0, 17.)
    ax1.set_ylim(-4.05, 2.5)
    #ax2.set_xlim(12.0, 23.0)
    #ax2.set_ylim(-5.0, 2.5)
    
    setticks(ax1, fontsize=fontsize, labelbottom=True, labelleft=True)
    #setticks(ax2, fontsize=fontsize, labelbottom=True, labelleft=False)

    ax1.set_xlabel(xlabel, fontsize=fontsize)
    #ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
            
    for ionind in range(len(ions)):
        ion = ions[ionind]
        
        #if ion in ['hneutralssh']:
        #    ax = ax2
        #else:
        ax = ax1
            
        #if relative:
        #    ax.text(0.05, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        #else:
        #    ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
            
        for vi in range(len(techvars)):
            plotx = bins[var][ion]
            plotx = plotx[:-1] + 0.5 * np.diff(plotx)
            
            ax.plot(plotx, np.log10((hists[var][ion]['nomask']) / dXtotdlogN[var][ion]), color=ioncolors[ion], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=ild.getnicename(ion, mathmode=False))
            
            ylim = ax.get_ylim()
            if ion == 'o8':
                ls = 'dashed'
            else:
                ls = 'solid'
            ax.axvline(approx_breaks[ion], ylim[0], 0.05 , color=ioncolors[ion], linewidth=2., linestyle=ls)
    #lax.axis('off')
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    #handles2, labels2 = ax2.get_legend_handles_labels()
    leg1 = ax1.legend(handles=handles1, fontsize=fontsize, ncol=1, loc='lower left', bbox_to_anchor=(0., 0.), frameon=False)
    #leg2 = ax2.legend(handles=handles1[3:] + handles2, fontsize=fontsize, ncol=1, loc='lower left', bbox_to_anchor=(0., 0.), frameon=False)
    ax1.add_artist(leg1)
    #ax2.add_artist(leg2)
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    


def plot_Tvir_ions_nice(snap=27, _ioncolors=ioncolors):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
    fontsize = 12
    mdir = '/net/luttero/data2/imgs/CGM/'
    
    if snap == 27:
        cosmopars = cosmopars_ea_27
        logrhob = logrhob_av_ea_27
        logrhoc = logrhoc_ea_27
        #print(logrhob, logrhoc)
    elif snap == 23:
        cosmopars = cosmopars_ea_27
        cosmopars['a'] = 0.665288  # eagle wiki
        cosmopars['z'] = 1. / cosmopars['a'] - 1. 
        logrhob = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars['h']**2 * cosmopars['omegab'] / cosmopars['a']**3 )
        
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] #, 'he2'
    ioncolors = _ioncolors.copy()
    
    #ioncolors.update({'he2': 'darkgoldenrod'})
    Ts = {}
    Tmaxs = {}
    nHs = {}
    bals = {}
    maxfracs = {}
    
    fracv = 0.1
    ciemargin = 1.50
    
    for ion in ions:
        bal, T, nH = m3.findiontables(ion, cosmopars['z'])
        bals[ion] = bal
        nHs[ion] = nH
        Ts[ion] = T
        indmaxfrac = np.argmax(bal[-1, :])
        maxfrac = bal[-1, indmaxfrac]
        Tmax = T[indmaxfrac]
        Tmaxs[ion] = Tmax
        
        xs = find_intercepts(bal[-1, :], T, fracv * maxfrac)
        print('Ion %s has maximum CIE fraction %.3f, at log T[K] = %.1f, %s max range is %s'%(ion, maxfrac, Tmax, fracv, str(xs)))
        maxfracs[ion] = maxfrac
        
    # neutral hydrogen
    #Tvals = 10**Ts[ions[0]]
    #nHvals = 10**nHs[ions[0]]
    #Tgrid = np.array([[T] * len(nHvals) for T in Tvals]).flatten()
    #nHgrid = np.array([nHvals] * len(Tvals)).flatten()
    #bal = m3.cfh.nHIHmol_over_nH({'Temperature': Tgrid, 'nH': nHgrid}, cosmopars['z'], UVB='HM01', useLSR=False)
    #bal = (bal.reshape((len(Tvals), len(nHvals)))).T
    #ion = 'hneutralssh'
    #bals[ion] = bal
    #Ts[ion]  = np.log10(Tvals)
    #nHs[ion] = np.log10(nHvals)
   
    #maxcol = bal[-1, :]
    #maxfrac = np.max(maxcol[np.isfinite(maxcol)])
    #maxfracs[ion] = maxfrac
    #print('Ion %s has maximum CIE fraction %.3f'%(ion, maxfrac))
    #xs = find_intercepts(maxcol[np.isfinite(maxcol)], np.log10(Tvals)[np.isfinite(maxcol)], fracv * maxfrac)
    #print('Ion %s has maximum CIE fraction %.3f, %s max range is %s'%(ion, maxfrac, fracv, str(xs)))
        
    allions = ions #['hneutralssh'] + 
    
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(5.5, 10.), gridspec_kw={'hspace': 0.})
    ax1.set_xlim(-8., -1.5)
    ax1.set_ylim(3.4, 7.65)
    ax2.set_xlim(-8., -1.5)
    ax2.set_ylim(3.4, 7.65)
    axions = {1: ['o6', 'o7', 'o8'], 2: ['ne8', 'ne9', 'fe17']}
    
    ax1.set_ylabel(r'$\log_{10} \, T \; [K]$', fontsize=fontsize)
    ax2.set_ylabel(r'$\log_{10} \, T \; [K]$', fontsize=fontsize)
    ax2.set_xlabel(r'$\log_{10} \, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    setticks(ax1, fontsize=fontsize, right=False, labelbottom=False)
    setticks(ax2, fontsize=fontsize, right=False)
    
    ax1.axvline(logrhob + np.log10(rho_to_nh), 0., 0.85, color='gray', linestyle='dashed', linewidth=1.5)
    ax2.axvline(logrhob + np.log10(rho_to_nh), 0., 0.85, color='gray', linestyle='dashed', linewidth=1.5)
    #ax.axvline(logrhoc + np.log10(rho_to_nh * 200. * cosmopars['omegab'] / cosmopars['omegam']), 0., 0.75, color='gray', linestyle='solid', linewidth=1.5)
    

    for ax, axi in zip([ax1, ax2], [1, 2]):
        for ion in axions[axi]:
            ax.contourf(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], alpha=0.1, linewidths=[3.], levels=[0.1 * maxfracs[ion], 1.])
            ax.contour(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], linewidths=[2.], levels=[0.1 * maxfracs[ion]], linestyles=['solid'])
        for ion in allions:
            ax.axhline(Tmaxs[ion], 0.95, 1., color=ioncolors[ion], linewidth=3.)
            
        #bal = bals[ion]
        #maxcol = bal[-1, :]
        #diffs = bal / maxcol[np.newaxis, :]
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal == 0)] = 0.
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] = bal[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] / 1e-18
        #diffs = np.abs(np.log10(diffs))
            
        #mask = bal < 0.6 * fracv * maxfracs[ion] # 0.6 gets the contours to ~ the edges of the ion regions
        #diffs[mask] = np.NaN

        #ax.contour(nHs[ion], Ts[ion][np.isfinite(maxcol)], (diffs[:, np.isfinite(maxcol)]).T, levels=[np.log10(ciemargin)], linestyles=['solid'], linewidths=[1.], alphas=0.5, colors=ioncolors[ion])

        axy2 = ax.twinx()
        ylim = ax.get_ylim()
        axy2.set_ylim(*ylim)
        mhalos = np.arange(9.0, 15.1, 0.5)
        Tvals = np.log10(T200c_hot(10**mhalos, cosmopars))
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_yticks(Tvals)
        axy2.set_yticklabels(Tlabels)
        setticks(axy2, fontsize=fontsize, left=False, right=True, labelleft=False, labelright=True)
        axy2.minorticks_off()
        axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$', fontsize=fontsize)
    
        handles = [mlines.Line2D([], [], label=ild.getnicename(ion, mathmode=False), color=ioncolors[ion]) for ion in axions[axi]]
        ax.legend(handles=handles, fontsize=fontsize, ncol=3, bbox_to_anchor=(0.0, 1.0), loc='upper left', frameon=False)

    plt.savefig(mdir + 'ionbals_snap{}_HM01_ionizedmu.pdf'.format(snap), format='pdf', bbox_inches='tight')
    
def plot_Tvir_ions_nice_talkversion(snap=27, _ioncolors=ioncolors, num=0):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    num: 0 -> empty frame, 6 -> all ions
    '''
    fontsize = 12
    mdir = '/home/wijers/Documents/papers/cgm_xray_abs/talk_figures/'
    
    if snap == 27:
        cosmopars = cosmopars_ea_27
        logrhob = logrhob_av_ea_27
        logrhoc = logrhoc_ea_27
        #print(logrhob, logrhoc)
    
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'][:num] #, 'he2'
    ioncolors = _ioncolors.copy()
    
    #ioncolors.update({'he2': 'darkgoldenrod'})
    Ts = {}
    Tmaxs = {}
    nHs = {}
    bals = {}
    maxfracs = {}
    
    fracv = 0.1
    ciemargin = 1.50
    
    for ion in ions:
        bal, T, nH = m3.findiontables(ion, cosmopars['z'])
        bals[ion] = bal
        nHs[ion] = nH
        Ts[ion] = T
        indmaxfrac = np.argmax(bal[-1, :])
        maxfrac = bal[-1, indmaxfrac]
        Tmax = T[indmaxfrac]
        Tmaxs[ion] = Tmax
        
        xs = find_intercepts(bal[-1, :], T, fracv * maxfrac)
        print('Ion %s has maximum CIE fraction %.3f, at log T[K] = %.1f, %s max range is %s'%(ion, maxfrac, Tmax, fracv, str(xs)))
        maxfracs[ion] = maxfrac
        
    # neutral hydrogen
    #Tvals = 10**Ts[ions[0]]
    #nHvals = 10**nHs[ions[0]]
    #Tgrid = np.array([[T] * len(nHvals) for T in Tvals]).flatten()
    #nHgrid = np.array([nHvals] * len(Tvals)).flatten()
    #bal = m3.cfh.nHIHmol_over_nH({'Temperature': Tgrid, 'nH': nHgrid}, cosmopars['z'], UVB='HM01', useLSR=False)
    #bal = (bal.reshape((len(Tvals), len(nHvals)))).T
    #ion = 'hneutralssh'
    #bals[ion] = bal
    #Ts[ion]  = np.log10(Tvals)
    #nHs[ion] = np.log10(nHvals)
   
    #maxcol = bal[-1, :]
    #maxfrac = np.max(maxcol[np.isfinite(maxcol)])
    #maxfracs[ion] = maxfrac
    #print('Ion %s has maximum CIE fraction %.3f'%(ion, maxfrac))
    #xs = find_intercepts(maxcol[np.isfinite(maxcol)], np.log10(Tvals)[np.isfinite(maxcol)], fracv * maxfrac)
    #print('Ion %s has maximum CIE fraction %.3f, %s max range is %s'%(ion, maxfrac, fracv, str(xs)))
        
    allions = ions #['hneutralssh'] + 
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(5.5, 3.), gridspec_kw={'wspace': 0.})
    ax1.set_xlim(-8., -1.0)
    ax1.set_ylim(3.4, 7.3)
    ax2.set_xlim(-8., -1.0)
    ax2.set_ylim(3.4, 7.3)
    axions = {1: ['o6', 'o7', 'o8'], 2: ['ne8', 'ne9', 'fe17']}
    
    ax1.set_ylabel(r'$\log_{10} \, \mathrm{T} \; [K]$', fontsize=fontsize)
    ax1.set_xlabel(r'$\log_{10} \, n(\mathrm{H}) \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    ax2.set_xlabel(r'$\log_{10} \, n(\mathrm{H}) \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    setticks(ax1, fontsize=fontsize, right=False)
    setticks(ax2, fontsize=fontsize, right=False, labelleft=False)
    
    ax1.axvline(logrhob + np.log10(rho_to_nh), 0., 1., color='gray', linestyle='dashed', linewidth=1.5)
    ax2.axvline(logrhob + np.log10(rho_to_nh), 0., 1., color='gray', linestyle='dashed', linewidth=1.5)
    #ax.axvline(logrhoc + np.log10(rho_to_nh * 200. * cosmopars['omegab'] / cosmopars['omegam']), 0., 0.75, color='gray', linestyle='solid', linewidth=1.5)
    

    for ax, axi in zip([ax1, ax2], [1, 2]):
        for ion in axions[axi]:
            if ion in ions:
                ax.contourf(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], alpha=0.1, linewidths=[3.], levels=[0.1 * maxfracs[ion], 1.])
                ax.contour(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], linewidths=[2.], levels=[0.1 * maxfracs[ion]], linestyles=['solid'])
        for ion in ions:
            ax.axhline(Tmaxs[ion], 0.95, 1., color=ioncolors[ion], linewidth=3.)
            
        #bal = bals[ion]
        #maxcol = bal[-1, :]
        #diffs = bal / maxcol[np.newaxis, :]
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal == 0)] = 0.
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] = bal[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] / 1e-18
        #diffs = np.abs(np.log10(diffs))
            
        #mask = bal < 0.6 * fracv * maxfracs[ion] # 0.6 gets the contours to ~ the edges of the ion regions
        #diffs[mask] = np.NaN

        #ax.contour(nHs[ion], Ts[ion][np.isfinite(maxcol)], (diffs[:, np.isfinite(maxcol)]).T, levels=[np.log10(ciemargin)], linestyles=['solid'], linewidths=[1.], alphas=0.5, colors=ioncolors[ion])

        axy2 = ax.twinx()
        ylim = ax.get_ylim()
        axy2.set_ylim(*ylim)
        mhalos = np.arange(9.0, 14.6, 0.5)
        Tvals = np.log10(T200c_hot(10**mhalos, cosmopars))
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_yticks(Tvals)
        if axi == 2:
            axy2.set_yticklabels(Tlabels)
            setticks(axy2, fontsize=fontsize, left=False, right=True, labelleft=False, labelright=True)
            axy2.minorticks_off()
            axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$', fontsize=fontsize)
        else:
            setticks(axy2, fontsize=fontsize, left=False, right=True, labelleft=False, labelright=False)
            axy2.minorticks_off()
        handles = [mlines.Line2D([], [], label=ild.getnicename(ion, mathmode=False), color=ioncolors[ion], linewidth=3.) for ion in axions[axi]]
        ax.legend(handles=handles, fontsize=fontsize, ncol=1,\
                  bbox_to_anchor=(1.03, 0.0), loc='lower right',\
                  frameon=False, handlelength=1.5)
    
    # only right-hand plot

    plt.savefig(mdir + 'ionbals_snap27_HM01_ionizedmu_num%i.pdf'%(num), format='pdf', bbox_inches='tight')
    
## CDDFsplits: using Fof-only projections instead of masks
def plotcddfsplits_fof(relative=False):
    '''
    paper plot: FoF-only projections vs. all gas
    '''
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] #, 'hneutralssh'
    
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    outname = mdir + 'split_FoF-M200c_proj_%s'%('-'.join(ions))
    if relative:
        outname = outname + '_rel'
    outname = outname + '.pdf'
    
    medges = np.arange(11., 14.1, 0.5) #np.arange(9., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_all = {key: [ol.pdir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    
    #if ion not in ions:
    #    raise ValueError('Ion must be one of %s'%ions)
    
    masses_proj = ['none'] + list(medges)
    filenames_ion = {ion: filenames_all[ion] for ion in ions}  
    filedct = {ion: {masses_proj[i]: filenames_ion[ion][i] for i in range(len(filenames_ion[ion]))} for ion in ions}
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for ion in ions:
        dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['bins/axis_0'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                        raise err
                        
                    dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['input_filenames'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    for mmass in masses_proj[1:]:
                        grp = fi[maskdct[mmass]]
                        hist = np.array(grp['hist'])
                        covfrac = grp.attrs['covfrac']
                        # recover cosmopars:
                        mask_examples = {key: item for (key, item) in grp.attrs.items()}
                        del mask_examples['covfrac']
                        example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                        example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                        path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                        #print(path)
                        cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                        dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                        dXtotdlogN = dXtot * np.diff(bins)
            
                        dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                    
                    # use cosmopars from the last read mask
                    mmass = 'none'
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
         
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    dct_totcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_totcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_totcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            for mmass in masses_proj[1:]:
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                mask_examples = {key: item for (key, item) in grp.attrs.items()}
                del mask_examples['covfrac']
                example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                dXtotdlogN = dXtot * np.diff(bins)
            
                dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
            # use cosmopars from the last read mask
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    cmapname = 'rainbow'
    #sumcolor = 'saddlebrown'
    #totalcolor = 'black'
    if relative:
        ylabel = r'$\log_{10}$ CDDF / total'
    else:
        ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'gas from haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) + [np.inf]
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels = {name: name + 0.5 * np.average(np.diff(massedges)) for name in masses_proj[1:]}
    
    numcols = 3
    numrows = 2
    panelwidth = 2.5
    panelheight = 2.5
    spaceh = 0.2
    legheight = 1.
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = True
        numrows_fig = numrows + 1
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols
        legindstart = 0
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numrows * panelheight + spaceh * numrows + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=spaceh, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks = sorted(masslabels.keys(), key=masslabels.__getitem__)
    colors = {_masks[i]: clist[i] for i in range(len(_masks))}
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    linewidth = 2.
    alpha = 1.
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]
        #if massind == 0:
        #    pmass = masses_proj[massind]
        #elif massind == 1:
        #    pmass = 'all halos'
        #else:
        #    pmass = masses_proj[massind - 1]
        #ax = axes[massind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        if relative:
            ax.set_ylim(-4.5, 1.)
        else:
            ax.set_ylim(-4.1, 2.5)
        
        labelx = yi == numrows - 1 #or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        setticks(ax, fontsize=fontsize, labelbottom=True, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.0, foreground="b"), mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"), mppe.Normal()]
        
        
        if relative:
            divby = dct_totcddf[ion]['none']['cddf']
        else:
            divby = 1. 
                    
        for pmass in masses_proj[1:]:
            _lw = linewidth
            _pe = patheff
            
            #bins = dct_totcddf['bins']
            #plotx = bins[:-1] + 0.5 * np.diff(bins)
            #ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
            # CDDF for projected mass, no mask
            bins = dct_fofcddf[ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[ion][pmass]['none']['cddf'] / divby), color=colors[pmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
        _lw = linewidth
        _pe = patheff
        # total CDDF
        bins = dct_totcddf[ion]['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax.plot(plotx, np.log10(dct_totcddf[ion]['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
        
        # all halo gas CDDF
        bins = dct_fofcddf[ion]['none']['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax.plot(plotx, np.log10(dct_fofcddf[ion]['none']['none']['cddf'] / divby), color=colors['allhalos'], linestyle='dashed', alpha=alpha, path_effects=patheff, linewidth=linewidth)

        text = r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True))
        if relative:
            ax.text(0.05, 0.05, text, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
        else:
            ax.text(0.95, 0.95, text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize)            

    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid']:
        subcols = list(clist) + [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    sumhandles = [#mlines.Line2D([], [], color=colors['none'], linestyle='solid', label='FoF no mask', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='solid', label='all gas', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='dashed', label=r'all halo gas', linewidth=2.),\
                  ]
    sumlabels = ['all gas', r'all halo gas']
    lax.legend(lcs + sumhandles, ['halo gas'] + sumlabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=ncol_legend, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')


def plotcddfsplits_fof_talkversion(relative=False, fmt='pdf', ion='o6', num=0, halostart='high'):
    '''
    Note: all haloes line with masks (brown dashed) is for all haloes with 
          M200c > 10^9 Msun, while the solid line is for all FoF+200c gas at 
          any M200c
    ion: which one to plot
    num: 0 -> frame only, 1 -> CDDF main, 2 -> add total halos, then add in 
         halos starting at <halostart> masses; 
         9 -> whole figure
    '''
    #ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] #, 'hneutralssh'
    ions = [ion]
    fontsize = 16
    
    mdir = '/home/wijers/Documents/papers/cgm_xray_abs/talk_figures/'
    outname = mdir + 'split_FoF-M200c_proj_%s_halostart-%s_num%i'%(ion, halostart, num)
    if relative:
        outname = outname + '_rel'
    outname = outname + '.%s'%fmt
    
    medges_all = np.arange(11., 14.1, 0.5) #np.arange(9., 14.1, 0.5)
    if halostart == 'high':
        medges = medges_all[2 + len(medges_all) - num:]
    else:
        medges = medges_all[: min(num - 3, 0)]
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_all = {key: [ol.pdir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    cosmopars = cosmopars_ea_27
    
    #if ion not in ions:
    #    raise ValueError('Ion must be one of %s'%ions)
    
    masses_proj = ['none'] + list(medges)
    masses_proj_all = ['none'] + list(medges_all)
    filenames_ion = {ion: filenames_all[ion] for ion in ions}  
    filedct = {ion: {masses_proj[i]: filenames_ion[ion][i] for i in range(len(filenames_ion[ion]))} for ion in ions}
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj_all[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for ion in ions:
        dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['bins/axis_0'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                        raise err
                        
                    dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['input_filenames'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    for mmass in masses_proj[1:]:
                        grp = fi[maskdct[mmass]]
                        hist = np.array(grp['hist'])
                        covfrac = grp.attrs['covfrac']
                        # recover cosmopars:
                        mask_examples = {key: item for (key, item) in grp.attrs.items()}
                        del mask_examples['covfrac']
                        example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                        example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                        path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                        #print(path)
                        cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                        dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                        dXtotdlogN = dXtot * np.diff(bins)
            
                        dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                    
                    # use cosmopars from the last read mask
                    mmass = 'none'
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
         
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    dct_totcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_totcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_totcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            for mmass in masses_proj[1:]:
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                mask_examples = {key: item for (key, item) in grp.attrs.items()}
                del mask_examples['covfrac']
                example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                dXtotdlogN = dXtot * np.diff(bins)
            
                dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
            # use cosmopars from the last read mask
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    #sumcolor = 'saddlebrown'
    #totalcolor = 'black'
    if relative:
        ylabel = r'$\log_{10}$ CDDF / total'
    else:
        ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    figwidth = 5.5 
    figheight = 5.
    spaceh = 0.33
    numcols = 1
    numrows = 1
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(nrows=2, ncols=2, hspace=spaceh, wspace=0.0, width_ratios=[5., 1.], height_ratios=[5., 1.])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    lax  = fig.add_subplot(grid[numrows, :])
    
    cbar, colors = add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
                    orientation='vertical', clabel=clabel, fontsize=fontsize, aspect=10.)
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    linewidth = 2.
    alpha = 1.
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]
        #if massind == 0:
        #    pmass = masses_proj[massind]
        #elif massind == 1:
        #    pmass = 'all halos'
        #else:
        #    pmass = masses_proj[massind - 1]
        #ax = axes[massind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        if relative:
            ax.set_ylim(-4.5, 1.)
        else:
            ax.set_ylim(-4.1, 2.5)
        
        labelx = yi == numrows - 1 #or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        setticks(ax, fontsize=fontsize, labelbottom=True, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.0, foreground="b"), mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"), mppe.Normal()]
               
        if relative:
            divby = dct_totcddf[ion]['none']['cddf']
        else:
            divby = 1. 
                    
        for pmass in masses_proj[1:]:
            _lw = linewidth
            _pe = patheff
            
            #bins = dct_totcddf['bins']
            #plotx = bins[:-1] + 0.5 * np.diff(bins)
            #ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
            # CDDF for projected mass, no mask
            bins = dct_fofcddf[ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[ion][pmass]['none']['cddf'] / divby), color=colors[pmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
        _lw = linewidth
        _pe = patheff
        # total CDDF
        if num > 0:
            bins = dct_totcddf[ion]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[ion]['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
        
        # all halo gas CDDF
        if num > 1:
            bins = dct_fofcddf[ion]['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[ion]['none']['none']['cddf'] / divby), color=colors['allhalos'], linestyle='dashed', alpha=alpha, path_effects=patheff, linewidth=linewidth)

        text = r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True))
        if relative:
            ax.text(0.05, 0.05, text, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
        else:
            ax.text(0.95, 0.95, text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize)            

    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid']:
        subcols = [colors[val] for val in np.arange(11., 14.1, 0.5)] + [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    sumhandles = [#mlines.Line2D([], [], color=colors['none'], linestyle='solid', label='FoF no mask', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='solid', label='total', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='dashed', label=r'all FoF+200c gas', linewidth=2.),\
                  ]
    sumlabels = ['all gas', r'all halo gas']
    lax.legend(lcs + sumhandles, ['halo gas'] + sumlabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(outname, format=fmt, bbox_inches='tight')
    
## CDDFsplits: comparing Fof-only projections and masks
def plotcddfs_fofvsmask(ion, relative=False):
    '''
    Note: all haloes line with masks (brown dashed) is for all haloes with 
          M200c > 10^9 Msun, while the solid line is for all FoF+200c gas at 
          any M200c
    '''
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    outname = mdir + 'split_FoF-M200c_proj_%s'%ion
    if relative:
        outname = outname + '_rel'
    outname = outname + '.pdf'
    
    ions = ['o7', 'o8', 'o6', 'ne8', 'fe17', 'ne9', 'hneutralssh']
    medges = np.arange(11., 14.1, 0.5) #np.arange(11., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_all = {key: [ol.pdir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    
    if ion not in ions:
        raise ValueError('Ion must be one of %s'%ions)
        
    filenames_this = filenames_all[ion]
    masses_proj = ['none'] + list(medges)
    filedct = {masses_proj[i]: filenames_this[i] for i in range(len(filenames_this))} 
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for pmass in masses_proj:
        dct_fofcddf[pmass] = {}
        try:
            with h5py.File(filedct[pmass]) as fi:
                try:
                    bins = np.array(fi['bins/axis_0'])
                except KeyError as err:
                    print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                    raise err
                    
                dct_fofcddf[pmass]['bins'] = bins
                
                inname = np.array(fi['input_filenames'])[0]
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                numpix_1sl.remove(None)
                numpix_1sl = int(list(numpix_1sl)[0][:-3])
                print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                for mmass in masses_proj[1:]:
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    mask_examples = {key: item for (key, item) in grp.attrs.items()}
                    del mask_examples['covfrac']
                    example_key = mask_examples.keys()[0] # 'mask_<slice center>'
                    example_mask = mask_examples[example_key] # '<dir path><mask file name>'
                    path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                    #print(path)
                    cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
        
                    dct_fofcddf[pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
                # use cosmopars from the last read mask
                mmass = 'none'
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                dXtotdlogN = dXtot * np.diff(bins)
                dct_fofcddf[pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
            
        except IOError as err:
            print('Failed to read in %s; stated error:'%filedct[pmass])
            print(err)
         
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'}
    
    file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
    dct_totcddf = {}
    with h5py.File(file_allproj) as fi:
        try:
            bins = np.array(fi['bins/axis_0'])
        except KeyError as err:
            print('While trying to load bins in file %s\n:'%(file_allproj))
            raise err
            
        dct_totcddf['bins'] = bins
        
        inname = np.array(fi['input_filenames'])[0]
        inname = inname.split('/')[-1] # throw out directory path
        parts = inname.split('_')

        numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
        numpix_1sl.remove(None)
        numpix_1sl = int(list(numpix_1sl)[0][:-3])
        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
        
        for mmass in masses_proj[1:]:
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            mask_examples = {key: item for (key, item) in grp.attrs.items()}
            del mask_examples['covfrac']
            example_key = mask_examples.keys()[0] # 'mask_<slice center>'
            example_mask = mask_examples[example_key] # '<dir path><mask file name>'
            path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
            cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
            dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
        
            dct_totcddf[mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
        # use cosmopars from the last read mask
        mmass = 'none'
        grp = fi[maskdct[mmass]]
        hist = np.array(grp['hist'])
        covfrac = grp.attrs['covfrac']
        # recover cosmopars:
        dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
        dXtotdlogN = dXtot * np.diff(bins)
        dct_totcddf[mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    cmapname = 'rainbow'
    #sumcolor = 'saddlebrown'
    #totalcolor = 'black'
    if relative:
        ylabel = r'$\log_{10}$ CDDF / total'
    else:
        ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'masks for haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) + [np.inf]
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels = {name: name + 0.5 * np.average(np.diff(massedges)) for name in masses_proj[1:]}
    
    numcols = 3
    numrows = 3
    panelwidth = 2.5
    panelheight = 2.
    legheight = 1.
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = True
        numrows_fig = numrows + 1
        legindstart = 0
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numcols * panelheight + 0.2 * numcols + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=0.0, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(masses_proj) + 1)]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    
    
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks = sorted(masslabels.keys(), key=masslabels.__getitem__)
    colors = {_masks[i]: clist[i] for i in range(len(_masks))}
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    linewidth = 2.
    alpha = 1.
    
    for massind in range(len(masses_proj) + 1):
        xi = massind % numcols
        yi = massind // numcols
        if massind == 0:
            pmass = masses_proj[massind]
        elif massind == 1:
            pmass = 'all halos'
        else:
            pmass = masses_proj[massind - 1]
        ax = axes[massind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        if relative:
            ax.set_ylim(-4.5, 1.)
        else:
            ax.set_ylim(-6.0, 2.5)
        
        labelx = yi == numrows - 1 or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely and yi == 1: 
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.0, foreground="b"), mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"), mppe.Normal()]
        
        if pmass == 'none':
            ptext = 'mask split'
            if relative:
                divby = dct_totcddf['none']['cddf']
            else:
                divby = 1. 
        
            for pmass in masses_proj[1:]:
                _lw = linewidth
                _pe = patheff
                
                bins = dct_totcddf['bins']
                plotx = bins[:-1] + 0.5 * np.diff(bins)
                ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=_pe, linewidth=_lw)
                
                #bins = dct_fofcddf[pmass]['bins']
                #plotx = bins[:-1] + 0.5 * np.diff(bins)
                #ax.plot(plotx, np.log10(dct_fofcddf[pmass]['none']['cddf'] / divby), color=colors[pmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
                
            _lw = linewidth
            _pe = patheff
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)

        elif pmass == 'all halos':
            ptext = 'all halo gas'
            if relative:
                divby = dct_fofcddf['none']['none']['cddf']
            else:
                divby = 1. 

            #for mmass in masses_proj[1:]:
            #    bins = dct_fofcddf[mmass]['bins']
            #    plotx = bins[:-1] + 0.5 * np.diff(bins)
            #    ax.plot(plotx, np.log10(dct_fofcddf['none'][mmass]['cddf'] / divby), color=colors[mmass], linestyle='solid', alpha=alpha, path_effects=patheff)
            
            bins = dct_fofcddf['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf['none']['none']['cddf'] / divby), color=colors['none'], linestyle='solid', alpha=alpha, path_effects=patheff, linewidth=linewidth)
            
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(np.sum([dct_totcddf[mass]['cddf'] for mass in masses_proj[1:]], axis=0) / divby), color=colors['allhalos'], linestyle='dashed', alpha=alpha, path_effects=patheff_thick, linewidth=linewidth + 0.5)
            
            bins = dct_fofcddf['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(np.sum([dct_fofcddf['none'][mass]['cddf'] for mass in masses_proj[1:]], axis=0) / divby), color=colors['allhalos'], linestyle='solid', alpha=alpha, path_effects=patheff_thick, linewidth=linewidth + 0.5)
        
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=patheff, linewidth=linewidth)
            
        else:
            if pmass == 14.0:
                ptext = r'$ > %.1f$'%pmass # \log_{10} \, \mathrm{M}_{\mathrm{200c}} \, / \, \mathrm{M}_{\odot}
            else:
                ptext = r'$ %.1f \emdash %.1f$'%(pmass, pmass + 0.5) # \leq \log_{10} \, \mathrm{M}_{\mathrm{200c}} \, / \, \mathrm{M}_{\odot} <
            
            if relative:
                divby = dct_fofcddf[pmass]['none']['cddf']
            else:
                divby = 1.
            
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=patheff)
                
            #for mmass in masses_proj[1:]:
            #    if mmass == pmass:
                    #_pe = patheff_thick
                    #_lw = linewidth + 0.5
            #    else:
            mmass = pmass
            _pe = patheff
            _lw = linewidth
            
            bins = dct_fofcddf[pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[pmass][mmass]['cddf'] / divby), color=colors[mmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
            bins = dct_fofcddf[pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[pmass]['none']['cddf'] / divby), color=colors['none'], linestyle='solid', alpha=alpha, path_effects=patheff)
        
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=patheff_thick, linewidth=linewidth + 0.5)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=patheff, linewidth=linewidth)
            
        if relative:
            ax.text(0.05, 0.95, ptext, horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        else:
            ax.text(0.97, 0.97, ptext, horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
            
    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid', 'dashed']:
        subcols = list(clist) + [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    sumhandles = [mlines.Line2D([], [], color=colors['none'], linestyle='solid', label='FoF no mask', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='solid', label='total', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='solid', label=r'all FoF+200c gas', linewidth=2.),\
                  ]
    sumlabels = ['FoF+200c, no mask', 'all gas, no mask', r'mask: all haloes $> 11.0$']
    lax.legend(lcs + sumhandles, ['FoF+200c, with mask', 'all gas, with mask'] + sumlabels,\
               handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize,\
               ncol=ncol_legend, loc='upper center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(outname, format='pdf')



def plotfracs_by_halo(ions=['Mass', 'oxygen', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'], fmt='pdf'):
    '''
    first: group mass bins by halo mass or subhalo catgory first
    '''
    
    ### corrections checked against total ion omegas in particle data only:
    #                                histogram for this plot    particle / snap box totals  
    #Particle data / total for hneutralssh: 0.9960255976557024  0.9960400162241149
    #Particle data / total for Mass: 0.26765323008070446        0.2676554816256265 
    #Particle data / total for o6:   0.27129005157978364        0.2712953355974472
    #Particle data / total for ne8:  0.2764794130850087         0.2764820025783056
    #Particle data / total for o7:   0.4821431746085879         0.48214487331886124
    #Particle data / total for ne9:  0.5315625247371946         0.5315598048201217
    #Particle data / total for o8:   0.3538290133287435         0.3538293076657853         
    #Particle data / total for fe17: 0.674329330727548          0.6743306068203926
    # 
    
    ## get total ion masses -> ion numbers from box_statistcis.py -> calcomegas
    # needed since the halo fractions come from histograms using particle data,
    # which excludes IGM contributions
    # omega data comes from snapshots
    omega_to_g_L100_27 = 2.0961773946142324e+50
    #Particle abundances, SF gas at 10^4 K
    Omega_gas = 0.056292501227365246
    Omega_oxygen_gas = 3.494829254263584e-05
    Omega_neon_gas   = 4.802734584639758e-06
    Omega_iron_gas   = 5.074204751697753e-06
    Omega_o6_gas   = 2.543315324105566e-07
    Omega_o7_gas   = 6.2585354699292046e-06
    Omega_o8_gas   = 7.598227615977929e-06
    Omega_ne8_gas  = 1.2563290036620843e-07
    Omega_ne9_gas  = 1.5418221011797857e-06
    Omega_fe17_gas = 6.091088105543567e-07
    Omega_hneutralssh_gas = 0.00023774499538436282
    
    total_nions = {'o6': Omega_o6_gas,\
                   'o7': Omega_o7_gas,\
                   'o8': Omega_o8_gas,\
                   'ne8': Omega_ne8_gas,\
                   'ne9': Omega_ne9_gas,\
                   'fe17': Omega_fe17_gas,\
                   'hneutralssh': Omega_hneutralssh_gas,\
                   'oxygen': Omega_oxygen_gas,\
                   'iron': Omega_iron_gas,\
                   'neon': Omega_neon_gas,\
                   'Mass': Omega_gas}
    
    total_nions = {ion: total_nions[ion] * omega_to_g_L100_27 / (ionh.atomw[string.capwords(ol.elements_ion[ion])] * c.u) \
                        if ion in ol.elements_ion else \
                        total_nions[ion] * omega_to_g_L100_27 / (ionh.atomw[string.capwords(ion)] * c.u) \
                        if string.capwords(ion) in ionh.atomw else
                        total_nions[ion] * omega_to_g_L100_27 \
                   for ion in total_nions     
                   }
    
    datafile_dct = {}
    for ion in ions:
        datafile_dir = '/net/quasar/data2/wijers/temp/'
        if ion == 'Mass':
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_T4EOS.hdf5'
            datafile = datafile_dir + datafile_base%(ion)
        elif ion in ol.elements_ion:
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_PtAb_T4EOS.hdf5'
            datafile = datafile_dir + datafile_base%('Nion_%s'%ion)
        else:
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_PtAb_T4EOS.hdf5'
            datafile = datafile_dir + datafile_base%('Nion_%s'%ion)
        datafile_dct[ion] = datafile
    
    outname = '/net/luttero/data2/imgs/histograms_basic/' + 'barchart_halomass_L0100N1504_27_T4EOS.%s'%(fmt)
    
    data_dct = {}
    for ion in ions:
        datafile = datafile_dct[ion]
        with h5py.File(datafile, 'r') as fi:
            try: # Mass, ion species histograms
                groupname = 'Temperature_T4EOS_Density_T4EOS_M200c_halo_allinR200c_subhalo_category'
                tname = 'Temperature_T4EOS'
                dname = 'Density_T4EOS'
                sname = 'subhalo_category'
                hname = 'M200c_halo_allinR200c'
                
                mgrp = fi[groupname]       
                tax = mgrp[tname].attrs['histogram axis']
                #tbins = np.array(mgrp['%s/bins'%(tname)])
                dax = mgrp[dname].attrs['histogram axis']
                #dbins = np.array(mgrp['%s/bins'%(dname)]) + np.log10(rho_to_nh)      
                sax = mgrp[sname].attrs['histogram axis']
                sbins = np.array(mgrp['%s/bins'%(sname)])
                hax = mgrp[hname].attrs['histogram axis']
                hbins = np.array(mgrp['%s/bins'%(hname)])
                
                hist = np.array(mgrp['histogram'])
                if mgrp['histogram'].attrs['log']:
                    hist = 10**hist
                hist = np.sum(hist, axis=(dax, tax, sax))
                total_in = mgrp['histogram'].attrs['sum of weights']
                # print('Histogramming recovered %f of input weights'%(np.sum(hist) / total_in))
            except KeyError: # element histograms
                groupname = 'M200c_halo_allinR200c_subhalo_category'
                sname = 'subhalo_category'
                hname = 'M200c_halo_allinR200c'
                
                mgrp = fi[groupname]         
                sax = mgrp[sname].attrs['histogram axis']
                sbins = np.array(mgrp['%s/bins'%(sname)])
                hax = mgrp[hname].attrs['histogram axis']
                hbins = np.array(mgrp['%s/bins'%(hname)])
                
                hist = np.array(mgrp['histogram'])
                if mgrp['histogram'].attrs['log']:
                    hist = 10**hist
                hist = np.sum(hist, axis=(sax,))
                total_in = mgrp['histogram'].attrs['sum of weights']
                
            #cosmopars = {key: item for key, item in fi['Header/cosmopars'].attrs.items()}
            total_allpart = total_nions[ion]
            print('Particle data / total for %s: %s'%(ion, total_in / total_allpart))
            
            hist[0] += total_allpart - total_in # add snap/particle data difference to IGM contribution
            data_dct[ion] = {'hist': hist, 'hbins': hbins, 'sbins': sbins, 'total': total_allpart}
            
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'        
    ylabel = 'fraction'
    xlabels = [r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)) if ion in ol.elements_ion else ion for ion in ions] 
    #slabels = ['cen.', 'sat.', 'unb.']
    #alphas = {'cen.': 1.0, 'sat.': 0.4, 'unb.': 0.7}
    
    cmapname = 'rainbow'
    nigmcolor = 'saddlebrown'
    igmcolor = 'gray'   
    print(hbins - np.log10(c.solar_mass))
    namededges = hbins[2:-1] - np.log10(c.solar_mass) # first two are -np.inf, and < single particle mass, last bin is empty (> 10^15 Msun)
    print(namededges)

    mmin = 11.
    indmin = np.argmin(np.abs(namededges - mmin))
    plotedges = namededges[indmin:]
    print(indmin)
    print(plotedges)

    clist = cm.get_cmap(cmapname, len(plotedges) - 2)(np.linspace(0., 1., len(plotedges) - 2))
    clist = np.append(clist, [mpl.colors.to_rgba('firebrick')], axis=0)
    #print(clist)
    colors = {hi + 2: clist[hi - indmin] for hi in range(indmin, indmin + len(plotedges) - 1)}
    print(colors)
    colors[1] = mpl.colors.to_rgba(nigmcolor)
    colors[0] = mpl.colors.to_rgba(igmcolor)
    colors[len(hbins) - 1] = mpl.colors.to_rgba('magenta') # shouldn't actaully be used
    #print(colors)

    fig = plt.figure(figsize=(5.5, 4.))
    maingrid = gsp.GridSpec(ncols=2, nrows=2, hspace=0.0, wspace=0.05, height_ratios=[0.7, 4.3], width_ratios=[5., 1.])
    cax = fig.add_subplot(maingrid[1, 1])
    lax = fig.add_subplot(maingrid[0, :])
    ax = fig.add_subplot(maingrid[1, 0])
    
    #print(namededges)
    cmap = mpl.colors.ListedColormap(clist)
    cmap.set_under(nigmcolor)
    #cmap.set_over('magenta')
    norm = mpl.colors.BoundaryNorm(plotedges, cmap.N)
    print(len(clist))
    print(cmap.N)
    print(len(plotedges))
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append([0.], plotedges, axis=0),\
                                ticks=plotedges,\
                                spacing='proportional', extend='min',\
                                orientation='vertical')
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)
    
    bottom = np.zeros(len(ions))
    barwidth = 0.9
    xvals = np.arange(len(ions))
    
    #print(data_dct['o6']['hist'].shape)
    #print(hbins - np.log10(c.solar_mass))
    #print(plotedges)
    for ind1 in range(len(hbins) - 2):  # last bin: M200c > 15, is empty 
        print(hbins[ind1] - np.log10(c.solar_mass))    
        alpha = 1.
        if ind1 >= indmin + 2 or ind1 == 0:
            color = mpl.colors.to_rgba(colors[ind1], alpha=alpha)
        else:
            color = nigmcolor
        sel = (ind1,) 
        
        if ind1 > 0 and ind1 < indmin + 2:
            if ind1 == 1: # start addition for nigm bins
                vals = np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            else:
                vals += np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            if ind1 == indmin + 1: # end of addition; plot
                ax.bar(xvals, vals, barwidth, bottom=bottom, color=color)
                bottom += vals
        else:
            vals = np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            
            ax.bar(xvals, vals, barwidth, bottom=bottom, color=color)
            bottom += vals
            
    setticks(ax, fontsize, top=False)
    ax.xaxis.set_tick_params(which='both', length=0.) # get labels without showing the ticks        
    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels)
    for label in ax.get_xmajorticklabels():
        label.set_rotation(45)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    legelts = [mpatch.Patch(facecolor=igmcolor, label='IGM')] #+ \
              #[mpatch.Patch(facecolor=mpl.colors.to_rgba('tan', alpha=alphas[slabel]), label=slabel) for slabel in slabels]
    lax.legend(handles=legelts, ncol=4, fontsize=fontsize, bbox_to_anchor=(0.5, 0.05), loc='lower center')
    lax.axis('off')
    
    plt.savefig(outname, format=fmt, bbox_inches='tight')


def plot_radprof_limited(ions=None, fontsize=fontsize, imgname=None):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    techvars_touse=[0, 7]
    units='R200c'
    ytype='perc'
    yvals_toplot=[10., 50., 90.]
    highlightcrit={'techvars': [0]} 
    plotcrit = {0: None,\
                7: {'o6':   {'Mmin': [11., 12.,  14.]},\
                    'o7':   {'Mmin': [11., 12.5, 14.]},\
                    'o8':   {'Mmin': [11., 13.,  14.]},\
                    'ne8':  {'Mmin': [11., 12.,  14.]},\
                    'ne9':  {'Mmin': [11., 13.,  14.]},\
                    'fe17': {'Mmin': [11., 13.,  14.]},\
                    },\
                }
    #plotcrit = {0: None,\
    #            7: {'o6':   {'Mmin': np.arange(11., 14.1, 0.5)},\
    #                'o7':   {'Mmin': np.arange(11., 14.1, 0.5)},\
    #                'o8':   {'Mmin': np.arange(11., 14.1, 0.5)},\
    #                'ne8':  {'Mmin': np.arange(11., 14.1, 0.5)},\
    #                'ne9':  {'Mmin': np.arange(11., 14.1, 0.5)},\
    #                'fe17': {'Mmin': np.arange(11., 14.1, 0.5)},\
    #                },\
    #            }
    printnumgals=False
    
    mdir = '/net/luttero/data2/imgs/CGM/radprof/'
    if ions is None:
        ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    else:
        imgname = 'radprof_byhalomass_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvars-%s_units-%s_%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars_touse])), units, ytype)
        imgname = mdir + imgname
        
        if ytype=='perc' and 50.0 not in yvals_toplot:
            imgname = imgname[:-4] + '_yvals-%s'%('-'.join([str(val) for val in yvals_toplot])) + '.pdf'
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    cmapname = 'rainbow'
    #hatches = ['\\', '/', '|', 'o', '+', '*', '-', 'x', '.']
    #sumcolor = 'saddlebrown'
    totalcolor = 'black'
    shading_alpha = 0.45 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    if units == 'pkpc':
        xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    elif units == 'R200c':
        xlabel = r'$r_{\perp} \; [\mathrm{R}_{\mathrm{200c}}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}
    
    ion_filedct_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}

    # use only 100 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}

    ion_filedct_subsample_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}
    # use only 1000 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample2_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                 }
    
    ion_filedct_1sl_binfofonly = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o6':   'rdist_coldens_o6_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o7':   'rdist_coldens_o7_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           }
        
    # define used mass ranges
    Mh_edges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    Mh_names_1sl_binfofonly = ['geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
                              'geq%s'%(Mh_mins[i])\
                              for i in range(len(Mh_mins))]

    galsetnames_massonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    galsetnames_offedges = {name + '_Z_off-edge-by-R200c':  galsetnames_massonly[name] for name in galsetnames_massonly.keys()}
    galsetnames_1sl_binfofonly = {name: sel for name, sel in zip(Mh_names_1sl_binfofonly, Mh_sels)}
    
    fills_filedct_fofonly = {Mh_names_1sl_binfofonly[i]: 'Mhalo_%.1f<=log200c<%.1f'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else \
                                                         'Mhalo_%.1f<=log200c'%(Mh_mins[i]) \
                             for i in range(len(Mh_mins))}
    
    techvars = {0: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                1: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_offedges.keys(), 'setfills': None},\
                2: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                3: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_offedges.keys(), 'setfills': None},\
                4: {'filenames': ion_filedct_subsample_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                5: {'filenames': ion_filedct_subsample_2sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                6: {'filenames': ion_filedct_subsample2_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                7: {'filenames': ion_filedct_1sl_binfofonly, 'setnames': galsetnames_1sl_binfofonly.keys(), 'setfills': fills_filedct_fofonly},\
                }
    
    linewidths = {0: 1.5,\
                  1: 1.5,\
                  2: 2.5,\
                  3: 2.5,\
                  4: 1.5,\
                  5: 2.5,\
                  6: 1.5,\
                  7: 1.}
       
    linestyles = {0: 'solid',\
                  1: 'dashed',\
                  2: 'solid',\
                  3: 'dotted',\
                  4: 'solid',\
                  5: 'solid',\
                  6: 'solid',\
                  7: 'dashed',\
                  }
    
    alphas = {0: 1.,\
              1: 1.,\
              2: 1.,\
              3: 1.,\
              4: 0.4,\
              5: 0.4,\
              6: 0.6,\
              7: 1.}
    
    legendnames_techvars = {0: 'all gas',\
                            # 0: r'1 sl., all',\
                            1: r'1 sl., off-edge',\
                            2: r'2 sl., all',\
                            3: r'2 sl., off-edge',\
                            4: r'1 sl., 100',\
                            5: r'2 sl., 100',\
                            6: r'1 sl., 1000',\
                            7: r'FoF gas only',\
                            }
    
    readpaths = {val: '%s_bins/binset_0/%s_%s'%(units, ytype, val) for val in yvals_toplot}
    readpath_bins = '/'.join((readpaths[list(readpaths.keys())[0]]).split('/')[:-1]) + '/bin_edges'
    print(readpaths)
    panelwidth = 2.5
    panelheight = 2.
    legheight = 0.9
    cwidth = 0.6
    if ytype == 'perc':
        wspace = 0.2
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + cwidth + wspace * numcols
    figheight = numcols * panelheight + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace, width_ratios=[panelwidth] * numcols + [cwidth], height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if len(techvars_touse) > 1:
        lax  = fig.add_subplot(grid[numrows, :])
    
    yvals = {}
    #cosmopars = {}
    #fcovs = {}
    #dXtot = {}
    #dztot = {}
    #dXtotdlogN = {}
    bins = {}
    numgals = {}
    
    for var in techvars_touse:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var]['filenames'][ion]
            goaltags = techvars[var]['setnames']
            setfills = techvars[var]['setfills']
            
            if ion not in filename:
                raise RuntimeError('File %s attributed to ion %s, mismatch'%(filename, ion))
            
            if setfills is None:
                with h5py.File(ol.pdir + 'radprof/' + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    for galset in galsets:
                        ex = True
                        for val in readpaths.keys():
                            try:
                                temp = np.array(fi[galset + '/' + readpaths[val]])
                            except KeyError:
                                ex = False
                                break
                        
                        if ex:
                            tags[fi[galset].attrs['seltag']] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())
                    #print(goaltags)
                    #print(tags.keys())
                    if len(tags_unread) > 0:
                        print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = ol.pdir + 'radprof/' + filename%(fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in readpaths.keys():
                                try:
                                    temp = np.array(fi[galset + '/' + readpaths[val]])
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
                            print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
    #    raise RuntimeError('total histograms from different files do not match')
    if printnumgals:
       print('tech vars: 0 = 1 slice, all, 1 = 1 slice, off-edge, 2 = 2 slices, all, 3 = 2 slices, off-edge')
       print('\n')
       
       for ion in ions:
           for var in techvars_touse:
               tags = techvars[var]['setnames']
               if var in [0, 2]:
                   tags = sorted(tags, key=galsetnames_massonly.__getitem__)
               else:
                   tags = sorted(tags, key=galsetnames_offedges.__getitem__)
               print('%s, var %s:'%(ion, var))
               print('\n'.join(['%s\t%s'%(tag, numgals[var][ion][tag]) for tag in tags]))
               print('\n')
       return numgals
        
    massranges = [sel[1:] for sel in Mh_sels]
    #print(massranges)
    massedges = sorted(list(set([np.log10(val) for rng in massranges for val in rng])))
    #print(massedges)
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels1 = {name: tuple(np.log10(np.array(galsetnames_massonly[name][1:]))) for name in galsetnames_massonly.keys()}
    masslabels2 = {name: tuple(np.log10(np.array(galsetnames_offedges[name][1:]))) for name in galsetnames_offedges.keys()}
    masslabels3 = {name: tuple(np.log10(np.array(galsetnames_1sl_binfofonly[name][1:]))) for name in galsetnames_1sl_binfofonly.keys()}
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks1 = sorted(masslabels1.keys(), key=masslabels1.__getitem__)
    colors = {_masks1[i]: clist[i] for i in range(len(_masks1))}
    _masks2 = sorted(masslabels2.keys(), key=masslabels2.__getitem__)
    colors.update({_masks2[i]: clist[i] for i in range(len(_masks2))})
    _masks3 = sorted(masslabels3.keys(), key=masslabels3.__getitem__)
    colors.update({_masks3[i]: clist[i] for i in range(len(_masks3))})
    #del _masks
    masslabels_all = masslabels1
    masslabels_all.update(masslabels2)
    masslabels_all.update(masslabels3)
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ytype == 'perc':
            if ion[0] == 'h':
                ax.set_ylim(12.0, 21.0)
            #else:
            #    ax.set_ylim(11.5, 17.0)
            elif ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o6':
                ax.set_ylim(10.8, 14.8)
            elif ion == 'ne9':
                ax.set_ylim(11.9, 15.9)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(0., 1.)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        

        ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        
        #hatchind = 0
        for vi in range(len(techvars_touse) -1, -1, -1): # backwards to get lowest techvars on top
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                if plotcrit[var] is not None:
                    mlist = plotcrit[var][ion]['Mmin']
                    match = np.min(np.abs(masslabels_all[tag][0] - np.array(mlist)[:, np.newaxis])) <= 0.01
                    if not match:
                        continue
                    
                if highlightcrit is not None: #highlightcrit={'techvars': [0], 'Mmin': [10.0, 12.0, 14.0]}
                    matched = True
                    _highlightcrit = highlightcrit
                    _highlightcrit['Mmin'] = \
                        12.0 if ion == 'o6' else \
                        12.0 if ion == 'ne8' else \
                        12.5 if ion == 'o7' else \
                        13.0 if ion == 'ne9' else \
                        13.0 if ion == 'o8' else \
                        13.0 if ion == 'fe17' else \
                        np.inf 
                    if 'techvars' in highlightcrit.keys():
                        matched &= var in _highlightcrit['techvars']
                    if 'Mmin' in highlightcrit.keys():
                        matched &= np.min(np.abs(masslabels_all[tag][0] - np.array(_highlightcrit['Mmin']))) <= 0.01
                    if matched:
                        yvals_toplot_temp = yvals_toplot
                    else:
                        yvals_toplot_temp = [yvals_toplot[0]] if len(yvals_toplot) == 1 else [yvals_toplot[1]]
                else:
                    yvals_toplot_temp = yvals_toplot
                
                
                if len(yvals_toplot_temp) == 3:
                    yval = yvals_toplot_temp[0]
                    try:                      
                        ploty1 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    yval = yvals_toplot_temp[2]
                    try:                      
                        ploty2 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    # according to stackexchange, this is the only way to set the hatch color in matplotlib 2.0.0 (quasar); does require the same color for all hatches...
                    #plt.rcParams['hatch.color'] = (0.5, 0.5, 0.5, alphas[var] * shading_alpha,) #mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,)
                    #ax.fill_between(plotx, ploty1, ploty2, color=(0., 0., 0., 0.), hatch=hatches[hatchind], facecolor=mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,), edgecolor='face', linewidth=0.0)
                    ax.fill_between(plotx, ploty1, ploty2, color=colors[tag], alpha=alphas[var] * shading_alpha, label=masslabels_all[tag])
                    
                    #hatchind += 1
                    yvals_toplot_temp = [yvals_toplot_temp[1]]
                    
                if len(yvals_toplot_temp) == 1:
                    yval = yvals_toplot_temp[0]
                    try:
                        ploty = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val))
                        continue
                    if yval == 50.0: # only highlight the medians
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
                    else:
                        patheff = []
                    ax.plot(plotx, ploty, color=colors[tag], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels_all[tag], path_effects=patheff)
                
        
        if ytype == 'perc':
            #ax.axhline(0., color=totalcolor, linestyle='solid', linewidth=1.5, alpha=0.7)
            #xlim = ax.get_xlim()
            ax.axhline(approx_breaks[ion], 0., 0.1, color='gray', linewidth=1.5, zorder=-1) # ioncolors[ion]
    #lax.axis('off')
        #ax.axvline(1.5, linestyle='dotted', linewidth=1., color='gray')
        #ax.axvline(1.6, linestyle='dotted', linewidth=1., color='black')
        ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = list(clist) #+ [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var], linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    #sumhandles = [mlines.Line2D([], [], color=sumcolor, linestyle='solid', label='all halos', linewidth=2.),\
    #              mlines.Line2D([], [], color=totalcolor, linestyle='solid', label='total', linewidth=2.)]
    #sumlabels = ['all halos', 'total']
    if len(techvars_touse) > 1:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse], handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2 * numcols, loc='lower center', bbox_to_anchor=(0.5, 0.))
        lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    

def plot_radprof_talkversion(ion='o6', fontsize=16, fmt='pdf', num=0):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    num: 0 -> plot frame, 1 -> first profile (high-mass), 7 -> last profile
         8 -> add detection limits
    '''
    techvars_touse=[0]
    units='R200c'
    ytype='perc'
    yvals_toplot=[50.]
    highlightcrit=None 
    printnumgals=False
    
    # ~HST-COS for O VI, Ne VIII, Athena estimate (rough early version) for X-ray
    detectcrit = {'o6': 13.5,\
                  'ne8': 13.5,\
                  'o7': 15.5,\
                  'o8': 15.7,\
                  'ne9': 15.5,\
                  'fe17': 14.9,\
                  }
    detectinstrument = {'o6': 'HST-COS',\
                        'ne8': 'HST-COS',\
                        'o7': 'Athena X-IFU',\
                        'o8': 'Athena X-IFU',\
                        'ne9': 'Athena X-IFU',\
                        'fe17': 'Athena X-IFU',\
                        }
    
    mdir = '/home/wijers/Documents/papers/cgm_xray_abs/talk_figures/'
    ions = [ion]
    
    imgname = 'radprof_byhalomass_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvar-%s_units-%s_%s_num%i.%s'%(ion, techvars_touse[0], units, ytype, num, fmt)
    imgname = mdir + imgname
    
    if ytype=='perc' and 50.0 not in yvals_toplot:
        imgname = imgname[:-4] + '_yvals-%s'%('-'.join([str(val) for val in yvals_toplot])) + '.{}'.format(fmt)
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    cmapname = 'rainbow'
    #hatches = ['\\', '/', '|', 'o', '+', '*', '-', 'x', '.']
    #sumcolor = 'saddlebrown'
    totalcolor = 'black'
    shading_alpha = 0.45 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    if units == 'pkpc':
        xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    elif units == 'R200c':
        xlabel = r'$r_{\perp} \; [\mathrm{R}_{\mathrm{200c}}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}
    
    ion_filedct_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}

    # use only 100 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}

    ion_filedct_subsample_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}
    # use only 1000 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample2_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                 }
    
    ion_filedct_1sl_binfofonly = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o6':   'rdist_coldens_o6_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o7':   'rdist_coldens_o7_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           }
        
    # define used mass ranges
    Mh_edges_all = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins_all = list(Mh_edges_all)
    Mh_maxs_all = list(Mh_edges_all[1:]) + [None]
    
    #Mh_edges = Mh_edges_all[min(len(Mh_edges_all) - num, 0):]
    Mh_mins = Mh_mins_all[max(len(Mh_edges_all) - num, 0):]
    Mh_maxs = Mh_maxs_all[max(len(Mh_edges_all) - num, 0):]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_sels_all = [('M200c_Msun', 10**Mh_mins_all[i], 10**Mh_maxs_all[i]) if Mh_maxs_all[i] is not None else\
                   ('M200c_Msun', 10**Mh_mins_all[i], np.inf)\
                   for i in range(len(Mh_mins_all))]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names_all =['logM200c_Msun_geq%s_le%s'%(Mh_mins_all[i], Mh_maxs_all[i]) if Mh_maxs_all[i] is not None else\
                   'logM200c_Msun_geq%s'%(Mh_mins_all[i])\
                   for i in range(len(Mh_mins_all))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    Mh_names_1sl_binfofonly_all = ['geq%s_le%s'%(Mh_mins_all[i], Mh_maxs_all[i]) if Mh_maxs_all[i] is not None else\
                                   'geq%s'%(Mh_mins_all[i])\
                                   for i in range(len(Mh_mins_all))]
    Mh_names_1sl_binfofonly = ['geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
                              'geq%s'%(Mh_mins[i])\
                              for i in range(len(Mh_mins))]
    galsetnames_massonly_all = {name: sel for name, sel in zip(Mh_names_all, Mh_sels_all)}
    galsetnames_offedges_all = {name + '_Z_off-edge-by-R200c':  galsetnames_massonly_all[name] for name in galsetnames_massonly_all.keys()}
    galsetnames_1sl_binfofonly_all = {name: sel for name, sel in zip(Mh_names_1sl_binfofonly_all, Mh_sels_all)}
    galsetnames_massonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    galsetnames_offedges = {name + '_Z_off-edge-by-R200c':  galsetnames_massonly[name] for name in galsetnames_massonly.keys()}
    galsetnames_1sl_binfofonly = {name: sel for name, sel in zip(Mh_names_1sl_binfofonly, Mh_sels)}
    
    fills_filedct_fofonly = {Mh_names_1sl_binfofonly[i]: 'Mhalo_%.1f<=log200c<%.1f'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else \
                                                         'Mhalo_%.1f<=log200c'%(Mh_mins[i]) \
                             for i in range(len(Mh_mins))}
    
    techvars = {0: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                1: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_offedges.keys(), 'setfills': None},\
                2: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                3: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_offedges.keys(), 'setfills': None},\
                4: {'filenames': ion_filedct_subsample_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                5: {'filenames': ion_filedct_subsample_2sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                6: {'filenames': ion_filedct_subsample2_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                7: {'filenames': ion_filedct_1sl_binfofonly, 'setnames': galsetnames_1sl_binfofonly.keys(), 'setfills': fills_filedct_fofonly},\
                }
    
    linewidths = {0: 1.5,\
                  1: 1.5,\
                  2: 2.5,\
                  3: 2.5,\
                  4: 1.5,\
                  5: 2.5,\
                  6: 1.5,\
                  7: 1.}
       
    linestyles = {0: 'solid',\
                  1: 'dashed',\
                  2: 'solid',\
                  3: 'dotted',\
                  4: 'solid',\
                  5: 'solid',\
                  6: 'solid',\
                  7: 'dashed',\
                  }
    
    alphas = {0: 1.,\
              1: 1.,\
              2: 1.,\
              3: 1.,\
              4: 0.4,\
              5: 0.4,\
              6: 0.6,\
              7: 1.}
    
    legendnames_techvars = {0: 'all gas',\
                            # 0: r'1 sl., all',\
                            1: r'1 sl., off-edge',\
                            2: r'2 sl., all',\
                            3: r'2 sl., off-edge',\
                            4: r'1 sl., 100',\
                            5: r'2 sl., 100',\
                            6: r'1 sl., 1000',\
                            7: r'FoF gas only',\
                            }
    
    readpaths = {val: '%s_bins/binset_0/%s_%s'%(units, ytype, val) for val in yvals_toplot}
    readpath_bins = '/'.join((readpaths[list(readpaths.keys())[0]]).split('/')[:-1]) + '/bin_edges'
    print(readpaths)
    if ytype == 'perc':
        wspace = 0.05
        xlim = (0.019, 3.1)
    else:
        wspace = 0.05
        xlim = (0.019, 3.1)
    #fcovticklen = 0.035
    figwidth = 5.5
    figheight = 5.
    numrows = 1
    numcols = 1
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace, width_ratios=[5., 1.], height_ratios=[5., 1.])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if len(techvars_touse) > 1:
        raise ValueError('Showing at most one tcchvar in these plots')
    
    yvals = {}
    #cosmopars = {}
    #fcovs = {}
    #dXtot = {}
    #dztot = {}
    #dXtotdlogN = {}
    bins = {}
    numgals = {}
    
    for var in techvars_touse:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var]['filenames'][ion]
            goaltags = techvars[var]['setnames']
            setfills = techvars[var]['setfills']
            
            if ion not in filename:
                raise RuntimeError('File %s attributed to ion %s, mismatch'%(filename, ion))
            
            if setfills is None:
                with h5py.File(ol.pdir + 'radprof/' + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    for galset in galsets:
                        ex = True
                        for val in readpaths.keys():
                            try:
                                temp = np.array(fi[galset + '/' + readpaths[val]])
                            except KeyError:
                                ex = False
                                break
                        
                        if ex:
                            tags[fi[galset].attrs['seltag'].decode()] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())
                    #print(goaltags)
                    #print(tags.keys())
                    if len(tags_unread) > 0:
                        print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = ol.pdir + 'radprof/' + filename%(fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in readpaths.keys():
                                try:
                                    temp = np.array(fi[galset + '/' + readpaths[val]])
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
                            print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
           
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
    #    raise RuntimeError('total histograms from different files do not match')
    if printnumgals:
       print('tech vars: 0 = 1 slice, all, 1 = 1 slice, off-edge, 2 = 2 slices, all, 3 = 2 slices, off-edge')
       print('\n')
       
       for ion in ions:
           for var in techvars_touse:
               tags = techvars[var]['setnames']
               if var in [0, 2]:
                   tags = sorted(tags, key=galsetnames_massonly.__getitem__)
               else:
                   tags = sorted(tags, key=galsetnames_offedges.__getitem__)
               print('%s, var %s:'%(ion, var))
               print('\n'.join(['%s\t%s'%(tag, numgals[var][ion][tag]) for tag in tags]))
               print('\n')
       return numgals
        
    massranges = [sel[1:] for sel in Mh_sels_all]
    #print(massranges)
    massedges = sorted(list(set([np.log10(val) for rng in massranges for val in rng])))
    #print(massedges)
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels1 = {name: tuple(np.log10(np.array(galsetnames_massonly_all[name][1:]))) for name in galsetnames_massonly_all.keys()}
    masslabels2 = {name: tuple(np.log10(np.array(galsetnames_offedges_all[name][1:]))) for name in galsetnames_offedges_all.keys()}
    masslabels3 = {name: tuple(np.log10(np.array(galsetnames_1sl_binfofonly_all[name][1:]))) for name in galsetnames_1sl_binfofonly_all.keys()}
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks1 = sorted(masslabels1.keys(), key=masslabels1.__getitem__)
    colors = {_masks1[i]: clist[i] for i in range(len(_masks1))}
    _masks2 = sorted(masslabels2.keys(), key=masslabels2.__getitem__)
    colors.update({_masks2[i]: clist[i] for i in range(len(_masks2))})
    _masks3 = sorted(masslabels3.keys(), key=masslabels3.__getitem__)
    colors.update({_masks3[i]: clist[i] for i in range(len(_masks3))})
    #del _masks
    masslabels_all = masslabels1
    masslabels_all.update(masslabels2)
    masslabels_all.update(masslabels3)
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ytype == 'perc':
            if ion[0] == 'h':
                ax.set_ylim(12.0, 21.0)
            #else:
            #    ax.set_ylim(11.5, 17.0)
            elif ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o6':
                ax.set_ylim(10.8, 14.8)
            elif ion == 'ne9':
                ax.set_ylim(11.9, 15.9)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(0., 1.)
        ax.set_xlim(*xlim)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        

        ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        
        #hatchind = 0
        for vi in range(len(techvars_touse) -1, -1, -1): # backwards to get lowest techvars on top
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                if highlightcrit is not None: #highlightcrit={'techvars': [0], 'Mmin': [10.0, 12.0, 14.0]}
                    matched = True
                    _highlightcrit = highlightcrit
                    _highlightcrit['Mmin'] = \
                        12.0 if ion == 'o6' else \
                        12.0 if ion == 'ne8' else \
                        12.5 if ion == 'o7' else \
                        13.0 if ion == 'ne9' else \
                        13.0 if ion == 'o8' else \
                        13.0 if ion == 'fe17' else \
                        np.inf 
                    if 'techvars' in highlightcrit.keys():
                        matched &= var in _highlightcrit['techvars']
                    if 'Mmin' in highlightcrit.keys():
                        matched &= np.min(np.abs(masslabels_all[tag][0] - np.array(_highlightcrit['Mmin']))) <= 0.01
                    if matched:
                        yvals_toplot_temp = yvals_toplot
                    else:
                        yvals_toplot_temp = [yvals_toplot[0]] if len(yvals_toplot) == 1 else [yvals_toplot[1]]
                else:
                    yvals_toplot_temp = yvals_toplot
                
                
                if len(yvals_toplot_temp) == 3:
                    yval = yvals_toplot_temp[0]
                    try:                      
                        ploty1 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    yval = yvals_toplot_temp[2]
                    try:                      
                        ploty2 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    # according to stackexchange, this is the only way to set the hatch color in matplotlib 2.0.0 (quasar); does require the same color for all hatches...
                    #plt.rcParams['hatch.color'] = (0.5, 0.5, 0.5, alphas[var] * shading_alpha,) #mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,)
                    #ax.fill_between(plotx, ploty1, ploty2, color=(0., 0., 0., 0.), hatch=hatches[hatchind], facecolor=mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,), edgecolor='face', linewidth=0.0)
                    ax.fill_between(plotx, ploty1, ploty2, color=colors[tag], alpha=alphas[var] * shading_alpha, label=masslabels_all[tag])
                    
                    #hatchind += 1
                    yvals_toplot_temp = [yvals_toplot_temp[1]]
                    
                if len(yvals_toplot_temp) == 1:
                    yval = yvals_toplot_temp[0]
                    try:
                        ploty = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val))
                        continue
                    if yval == 50.0: # only highlight the medians
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
                    else:
                        patheff = []
                    ax.plot(plotx, ploty, color=colors[tag], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels_all[tag], path_effects=patheff)
        # add detection indicator
        if num > 7:
            Nval = detectcrit[ion]
            ax.axhline(Nval, linestyle='dotted', color='gray')
            #ylims = ax.get_ylim()
            #ax.text(1., (Nval - ylims[0]) / (ylims[1] - ylims[0]), detectinstrument[ion],\
            #        fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom',\
            #        horizontalalignment='right', color='gray')
                
        if ytype == 'perc':
            #ax.axhline(0., color=totalcolor, linestyle='solid', linewidth=1.5, alpha=0.7)
            #xlim = ax.get_xlim()
            ax.axhline(approx_breaks[ion], 0., 0.1, color='gray', linewidth=1.5, zorder=-1) # ioncolors[ion]
    #lax.axis('off')
        ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = list(clist) #+ [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var], linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    #sumhandles = [mlines.Line2D([], [], color=sumcolor, linestyle='solid', label='all halos', linewidth=2.),\
    #              mlines.Line2D([], [], color=totalcolor, linestyle='solid', label='total', linewidth=2.)]
    #sumlabels = ['all halos', 'total']
    if len(techvars_touse) > 1:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse], handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2 * numcols, loc='lower center', bbox_to_anchor=(0.5, 0.))
        lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format=fmt, bbox_inches='tight')
    

def plot_radprof_zev(ions=None, fontsize=fontsize, imgname=None):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    techvars_touse=[0, 8]
    units='R200c'
    ytype='perc'
    yvals_toplot=[10., 50., 90.]
    highlightcrit={'techvars': [0]} 
    printnumgals=False
    
    mdir = '/net/luttero/data2/imgs/CGM/radprof/'
    if ions is None:
        ions = ['o7', 'o8', 'ne8'] #['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    
    if imgname is not None:
        if '/' not in imgname:
            imgname = mdir + imgname
        if imgname[-4:] != '.pdf':
            imgname = imgname + '.pdf'
    else:
        imgname = 'radprof_byhalomass_%s_L0100N1504_27-23_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_techvars-%s_units-%s_%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars_touse])), units, ytype)
        imgname = mdir + imgname
        
        if ytype=='perc' and 50.0 not in yvals_toplot:
            imgname = imgname[:-4] + '_yvals-%s'%('-'.join([str(val) for val in yvals_toplot])) + '.pdf'
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    cmapname = 'rainbow'
    #hatches = ['\\', '/', '|', 'o', '+', '*', '-', 'x', '.']
    #sumcolor = 'saddlebrown'
    totalcolor = 'black'
    shading_alpha = 0.45 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    if units == 'pkpc':
        xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    elif units == 'R200c':
        xlabel = r'$r_{\perp} \; [\mathrm{R}_{\mathrm{200c}}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}
    
    ion_filedct_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5'}

    # use only 100 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}

    ion_filedct_subsample_2sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne9': 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'ne8': 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o8': 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o7': 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'o6': 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5',\
                                 'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals_stored_profiles.hdf5'}
    # use only 1000 galaxies (random selection) per mass bin -> compare
    ion_filedct_subsample2_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                  'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals_stored_profiles.hdf5',\
                                 }
    
    ion_filedct_1sl_binfofonly = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o6':   'rdist_coldens_o6_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o7':   'rdist_coldens_o7_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           }
    
    ion_filedct_1sl_snap23 = {'ne8': 'rdist_coldens_ne8_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                              'o7': 'rdist_coldens_o7_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EO_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                              'o8': 'rdist_coldens_o8_L0100N1504_23_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EO_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                              }
        
    # define used mass ranges
    Mh_edges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    Mh_names_1sl_binfofonly = ['geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
                              'geq%s'%(Mh_mins[i])\
                              for i in range(len(Mh_mins))]

    galsetnames_massonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    galsetnames_offedges = {name + '_Z_off-edge-by-R200c':  galsetnames_massonly[name] for name in galsetnames_massonly.keys()}
    galsetnames_1sl_binfofonly = {name: sel for name, sel in zip(Mh_names_1sl_binfofonly, Mh_sels)}
    
    fills_filedct_fofonly = {Mh_names_1sl_binfofonly[i]: 'Mhalo_%.1f<=log200c<%.1f'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else \
                                                         'Mhalo_%.1f<=log200c'%(Mh_mins[i]) \
                             for i in range(len(Mh_mins))}
    
    techvars = {0: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                1: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_offedges.keys(), 'setfills': None},\
                2: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                3: {'filenames': ion_filedct_2sl, 'setnames': galsetnames_offedges.keys(), 'setfills': None},\
                4: {'filenames': ion_filedct_subsample_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                5: {'filenames': ion_filedct_subsample_2sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                6: {'filenames': ion_filedct_subsample2_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                7: {'filenames': ion_filedct_1sl_binfofonly, 'setnames': galsetnames_1sl_binfofonly.keys(), 'setfills': fills_filedct_fofonly},\
                8: {'filenames': ion_filedct_1sl_snap23, 'setnames': galsetnames_1sl_binfofonly.keys(), 'setfills': None}
                }
    
    linewidths = {0: 1.5,\
                  1: 1.5,\
                  2: 2.5,\
                  3: 2.5,\
                  4: 1.5,\
                  5: 2.5,\
                  6: 1.5,\
                  7: 1.,\
                  8: 2.,\
                  }
       
    linestyles = {0: 'solid',\
                  1: 'dashed',\
                  2: 'solid',\
                  3: 'dotted',\
                  4: 'solid',\
                  5: 'solid',\
                  6: 'solid',\
                  7: 'dashed',\
                  8: 'dashed',\
                  }
    
    alphas = {0: 1.,\
              1: 1.,\
              2: 1.,\
              3: 1.,\
              4: 0.4,\
              5: 0.4,\
              6: 0.6,\
              7: 1.,\
              8: 1.}
    
    legendnames_techvars = {0: 'all gas, z=0.1',\
                            # 0: r'1 sl., all',\
                            1: r'1 sl., off-edge',\
                            2: r'2 sl., all',\
                            3: r'2 sl., off-edge',\
                            4: r'1 sl., 100',\
                            5: r'2 sl., 100',\
                            6: r'1 sl., 1000',\
                            7: r'FoF gas only',\
                            8: r'all gas, z=0.5'}
    
    readpaths = {val: '%s_bins/binset_0/%s_%s'%(units, ytype, val) for val in yvals_toplot}
    readpath_bins = '/'.join((readpaths[readpaths.keys()[0]]).split('/')[:-1]) + '/bin_edges'
    print(readpaths)
    panelwidth = 2.5
    panelheight = 3.
    legheight = 1.3
    cwidth = 0.6
    if ytype == 'perc':
        wspace = 0.2
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + cwidth + wspace * numcols
    figheight = numrows * panelheight + legheight
    print('{}, {}'.format(figwidth, figheight))
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace, width_ratios=[panelwidth] * numcols + [cwidth], height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if len(techvars_touse) > 1:
        lax  = fig.add_subplot(grid[numrows, :])
    
    yvals = {}
    #cosmopars = {}
    #fcovs = {}
    #dXtot = {}
    #dztot = {}
    #dXtotdlogN = {}
    bins = {}
    numgals = {}
    
    for var in techvars_touse:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var]['filenames'][ion]
            goaltags = techvars[var]['setnames']
            setfills = techvars[var]['setfills']
            
            if ion not in filename:
                raise RuntimeError('File %s attributed to ion %s, mismatch'%(filename, ion))
            
            if setfills is None:
                with h5py.File(ol.pdir + 'radprof/' + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    for galset in galsets:
                        ex = True
                        for val in readpaths.keys():
                            try:
                                temp = np.array(fi[galset + '/' + readpaths[val]])
                            except KeyError:
                                ex = False
                                break
                        
                        if ex:
                            tags[fi[galset].attrs['seltag']] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())
                    #print(goaltags)
                    #print(tags.keys())
                    if len(tags_unread) > 0:
                        print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = ol.pdir + 'radprof/' + filename%(fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in readpaths.keys():
                                try:
                                    np.array(fi[galset + '/' + readpaths[val]])
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
                            print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[val])]) for val in readpaths.keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
    #    raise RuntimeError('total histograms from different files do not match')
    if printnumgals:
       print('tech vars: 0 = 1 slice, all, 1 = 1 slice, off-edge, 2 = 2 slices, all, 3 = 2 slices, off-edge')
       print('\n')
       
       for ion in ions:
           for var in techvars_touse:
               tags = techvars[var]['setnames']
               if var in [0, 2]:
                   tags = sorted(tags, key=galsetnames_massonly.__getitem__)
               else:
                   tags = sorted(tags, key=galsetnames_offedges.__getitem__)
               print('%s, var %s:'%(ion, var))
               print('\n'.join(['%s\t%s'%(tag, numgals[var][ion][tag]) for tag in tags]))
               print('\n')
       return numgals
        
    massranges = [sel[1:] for sel in Mh_sels]
    #print(massranges)
    massedges = sorted(list(set([np.log10(val) for rng in massranges for val in rng])))
    #print(massedges)
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels1 = {name: tuple(np.log10(np.array(galsetnames_massonly[name][1:]))) for name in galsetnames_massonly.keys()}
    masslabels2 = {name: tuple(np.log10(np.array(galsetnames_offedges[name][1:]))) for name in galsetnames_offedges.keys()}
    masslabels3 = {name: tuple(np.log10(np.array(galsetnames_1sl_binfofonly[name][1:]))) for name in galsetnames_1sl_binfofonly.keys()}
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks1 = sorted(masslabels1.keys(), key=masslabels1.__getitem__)
    colors = {_masks1[i]: clist[i] for i in range(len(_masks1))}
    _masks2 = sorted(masslabels2.keys(), key=masslabels2.__getitem__)
    colors.update({_masks2[i]: clist[i] for i in range(len(_masks2))})
    _masks3 = sorted(masslabels3.keys(), key=masslabels3.__getitem__)
    colors.update({_masks3[i]: clist[i] for i in range(len(_masks3))})
    #del _masks
    masslabels_all = masslabels1
    masslabels_all.update(masslabels2)
    masslabels_all.update(masslabels3)
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ytype == 'perc':
            if ion[0] == 'h':
                ax.set_ylim(12.0, 21.0)
            #else:
            #    ax.set_ylim(11.5, 17.0)
            elif ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o6':
                ax.set_ylim(10.8, 14.8)
            elif ion == 'ne9':
                ax.set_ylim(11.9, 15.9)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(0., 1.)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        

        ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        
        #hatchind = 0
        for vi in range(len(techvars_touse) -1, -1, -1): # backwards to get lowest techvars on top
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                if highlightcrit is not None: #highlightcrit={'techvars': [0], 'Mmin': [10.0, 12.0, 14.0]}
                    matched = True
                    _highlightcrit = highlightcrit
                    _highlightcrit['Mmin'] = \
                        12.0 if ion == 'o6' else \
                        12.0 if ion == 'ne8' else \
                        12.5 if ion == 'o7' else \
                        13.0 if ion == 'ne9' else \
                        13.0 if ion == 'o8' else \
                        13.0 if ion == 'fe17' else \
                        np.inf 
                    if 'techvars' in highlightcrit.keys():
                        matched &= var in _highlightcrit['techvars']
                    if 'Mmin' in highlightcrit.keys():
                        matched &= np.min(np.abs(masslabels_all[tag][0] - np.array(_highlightcrit['Mmin']))) <= 0.01
                    if matched:
                        yvals_toplot_temp = yvals_toplot
                    else:
                        yvals_toplot_temp = [yvals_toplot[0]] if len(yvals_toplot) == 1 else [yvals_toplot[1]]
                else:
                    yvals_toplot_temp = yvals_toplot
                
                
                if len(yvals_toplot_temp) == 3:
                    yval = yvals_toplot_temp[0]
                    try:                      
                        ploty1 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    yval = yvals_toplot_temp[2]
                    try:                      
                        ploty2 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                    # according to stackexchange, this is the only way to set the hatch color in matplotlib 2.0.0 (quasar); does require the same color for all hatches...
                    #plt.rcParams['hatch.color'] = (0.5, 0.5, 0.5, alphas[var] * shading_alpha,) #mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,)
                    #ax.fill_between(plotx, ploty1, ploty2, color=(0., 0., 0., 0.), hatch=hatches[hatchind], facecolor=mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,), edgecolor='face', linewidth=0.0)
                    ax.fill_between(plotx, ploty1, ploty2, color=colors[tag], alpha=alphas[var] * shading_alpha, label=masslabels_all[tag])
                    
                    #hatchind += 1
                    yvals_toplot_temp = [yvals_toplot_temp[1]]
                    
                if len(yvals_toplot_temp) == 1:
                    yval = yvals_toplot_temp[0]
                    try:
                        ploty = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val))
                        continue
                    if yval == 50.0: # only highlight the medians
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
                    else:
                        patheff = []
                    ax.plot(plotx, ploty, color=colors[tag], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels_all[tag], path_effects=patheff)
                
        
        if ytype == 'perc':
            #ax.axhline(0., color=totalcolor, linestyle='solid', linewidth=1.5, alpha=0.7)
            #xlim = ax.get_xlim()
            ax.axhline(approx_breaks[ion], 0., 0.1, color='gray', linewidth=1.5, zorder=-1) # ioncolors[ion]
    #lax.axis('off')
        ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = list(clist) #+ [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var], linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    #sumhandles = [mlines.Line2D([], [], color=sumcolor, linestyle='solid', label='all halos', linewidth=2.),\
    #              mlines.Line2D([], [], color=totalcolor, linestyle='solid', label='total', linewidth=2.)]
    #sumlabels = ['all halos', 'total']
    if len(techvars_touse) > 1:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse], handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2 * numcols, loc='lower center', bbox_to_anchor=(0.5, 0.))
        lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    

def plotcddfsplits_fof_zev(relative=False):
    '''
    paper plot: FoF-only projections vs. all gas at z=0.1 and z=0.5
    '''
    ions = ['ne8', 'o7', 'o8'] #, 'hneutralssh'
    
    mdir = '/net/luttero/data2/imgs/CGM/cddfsplits/'
    outname = mdir + 'split_FoF-M200c_proj_z0p1-vs-0p5_%s'%('-'.join(ions))
    if relative:
        outname = outname + '_rel'
    outname = outname + '.pdf'
    
    medges = np.arange(11., 14.1, 0.5) #np.arange(9., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all_s27 = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_s27 = {key: [ol.pdir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all_s27[key]] for key in prefilenames_all_s27.keys()}
        
    filenames_s23 = {key: [ol.pdir + 'cddf_coldens_%s_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_add-1_offset-0_resreduce-1.hdf5'%(key, halofill) for halofill in halofills]
                 for key in ions}
    #if ion not in ions:
    #    raise ValueError('Ion must be one of %s'%ions)
    
    masses_proj = ['none'] + list(medges)
    filenames_s27_ion = {ion: filenames_s27[ion] for ion in ions}  
    filedct_s27 = {ion: {masses_proj[i]: filenames_s27_ion[ion][i] for i in range(len(filenames_s27_ion[ion]))} for ion in ions}
    filenames_s23_ion = {ion: filenames_s23[ion] for ion in ions}  
    filedct_s23 = {ion: {masses_proj[i]: filenames_s23_ion[ion][i] for i in range(len(filenames_s27_ion[ion]))} for ion in ions}
    #print(filedct_s23['o8'].keys())
    #print(filedct_s27['o8'].keys())

    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    cosmopars_snap = {}
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    filedct = filedct_s27
    for ion in ions:
        dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['bins/axis_0'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[ion][pmass]))
                        raise err
                        
                    dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['input_filenames'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    for mmass in masses_proj[1:]:
                        grp = fi[maskdct[mmass]]
                        hist = np.array(grp['hist'])
                        covfrac = grp.attrs['covfrac']
                        # recover cosmopars:
                        mask_examples = {key: item for (key, item) in grp.attrs.items()}
                        del mask_examples['covfrac']
                        example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                        example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                        path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                        #print(path)
                        cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                        cosmopars_snap[27] = cosmopars
                        dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                        dXtotdlogN = dXtot * np.diff(bins)
                        
                        dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                    
                    # use cosmopars from the last read mask
                    mmass = 'none'
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
         
    dct_fofcddf = {27: dct_fofcddf}
    _dct_fofcddf = {}
    filedct = filedct_s23
    for ion in ions:
        _dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            _dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['edges'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[ion][pmass]))
                        raise err
                    if bins[0] == -np.inf:
                        bins[0] = 2. * bins[1] - bins[2]
                    if bins[-1] == np.inf:
                        bins[-2] = 2. * bins[-2] - bins[-3]
                    
                    _dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['Header/filenames_in'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
                    cosmopars_snap[23] = cosmopars
                    
                    # use cosmopars from the last read mask
                    mmass = 'none'
                    grp = fi
                    hist = np.array(grp['histogram'])
                    covfrac = 1. #grp.attrs['covfrac']
                    # recover cosmopars:
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    _dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
    
    dct_fofcddf.update({23: _dct_fofcddf})
    
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      #'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    dct_totcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_totcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_totcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            for mmass in masses_proj[1:]:
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                mask_examples = {key: item for (key, item) in grp.attrs.items()}
                del mask_examples['covfrac']
                example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                dXtotdlogN = dXtot * np.diff(bins)
            
                dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
            # use cosmopars from the last read mask
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    dct_totcddf = {27: dct_totcddf}
    
    totfiles_snap23 = {'o8': ol.pdir + 'cddf_coldens_o8_L0100N1504_23_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.hdf5',\
                       'o7': ol.pdir + 'cddf_coldens_o7_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.hdf5',\
                       'ne8': ol.pdir + 'cddf_coldens_ne8_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.hdf5',\
                       }
    dct_totcddf[23] = {}
    for ion in totfiles_snap23:
        dct_totcddf[23][ion] = {'none': {}}
        with h5py.File(totfiles_snap23[ion], 'r') as fi:
            print(totfiles_snap23[ion])
            print(fi.keys())
            cosmopars = cosmopars_snap[23]
            
            covfrac = 1. #grp.attrs['covfrac']
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
            
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            hist = np.array(fi['masks_0/hist'])
            bins = np.array(fi['bins/axis_0'])
            if bins[0] == -np.inf:
                bins[0] = 2. * bins[1] - bins[2]
            if bins[-1] == np.inf:
                bins[-2] = 2. * bins[-2] - bins[-3]
                        
            dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            
            dct_totcddf[23][ion]['bins'] = bins           
            dct_totcddf[23][ion]['none'] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    cmapname = 'rainbow'
    #sumcolor = 'saddlebrown'
    #totalcolor = 'black'
    if relative:
        ylabel = r'$\log_{10}$ CDDF / total'
    else:
        ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'gas from haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) + [np.inf]
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    masslabels = {name: name + 0.5 * np.average(np.diff(massedges)) for name in masses_proj[1:]}
    
    numcols = 3
    numrows = 2
    panelwidth = 2.5
    panelheight = 2.5
    spaceh = 0.0
    legheight = 1.8
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = True
        numrows_fig = numrows + 1
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols
        legindstart = 0
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numrows * panelheight + spaceh * numrows + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=spaceh, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(2 * len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    clist = cm.get_cmap(cmapname, len(massedges) - 1)(np.linspace(0., 1.,len(massedges) - 1))
    _masks = sorted(masslabels.keys(), key=masslabels.__getitem__)
    colors = {_masks[i]: clist[i] for i in range(len(_masks))}
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges[:-1], cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massedges,\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    linewidth = 2.
    alpha = 1.
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax_top = axes[ionind]
        ax_bot = axes[ionind + 3]
        #if massind == 0:
        #    pmass = masses_proj[massind]
        #elif massind == 1:
        #    pmass = 'all halos'
        #else:
        #    pmass = masses_proj[massind - 1]
        #ax = axes[massind]

        if ion[0] == 'h':
            ax_top.set_xlim(12.0, 23.0)
            ax_bot.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax_top.set_xlim(12., 16.)
            ax_bot.set_xlim(12., 16.)
        elif ion == 'o7':
            ax_top.set_xlim(13.25, 17.25)
            ax_bot.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax_top.set_xlim(13., 17.)
            ax_bot.set_xlim(13., 17.)
        elif ion == 'o6':
            ax_top.set_xlim(12., 16.)
            ax_bot.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax_top.set_xlim(12.5, 16.5)
            ax_bot.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax_top.set_xlim(11.5, 15.5)
            ax_bot.set_xlim(11.5, 15.5)
            
        if relative:
            ax_top.set_ylim(-4.5, 1.)
            ax_bot.set_ylim(-4.5, 1.)
        else:
            ax_top.set_ylim(-4.1, 2.5)
            ax_bot.set_ylim(-4.1, 2.5)
            
        labelx = yi == numrows - 1 #or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        setticks(ax_top, fontsize=fontsize, labelbottom=False, labelleft=labely)
        setticks(ax_bot, fontsize=fontsize, labelbottom=True, labelleft=labely)
        ax_bot.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax_bot.set_ylabel(ylabel, fontsize=fontsize)
            ax_top.set_ylabel(ylabel, fontsize=fontsize)
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.0, foreground="b"), mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"), mppe.Normal()]
        
        
        if relative:
            divby = dct_totcddf[ion]['none']['cddf']
        else:
            divby = 1. 
                    
        for pi in range(1, len(masses_proj)):
            pmass = masses_proj[pi]
            _lw = linewidth
            _pe = patheff
            
            #bins = dct_totcddf['bins']
            #plotx = bins[:-1] + 0.5 * np.diff(bins)
            #ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass], linestyle='dashed', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
            # CDDF for projected mass, no mask
            if bool(pi%2):
                ax = ax_top
            else:
                ax = ax_bot
            bins = dct_fofcddf[27][ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[27][ion][pmass]['none']['cddf'] / divby), color=colors[pmass], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
            # snap 23
            bins = dct_fofcddf[23][ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[23][ion][pmass]['none']['cddf'] / divby), color=colors[pmass], linestyle='dotted', alpha=alpha, path_effects=_pe, linewidth=_lw)
            
        _lw = linewidth
        _pe = patheff
        # total CDDF
        bins = dct_totcddf[27][ion]['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_top.plot(plotx, np.log10(dct_totcddf[27][ion]['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)
        ax_bot.plot(plotx, np.log10(dct_totcddf[27][ion]['none']['cddf'] / divby), color=colors['total'], linestyle='solid', alpha=alpha, path_effects=_pe, linewidth=_lw)

        bins = dct_totcddf[23][ion]['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_top.plot(plotx, np.log10(dct_totcddf[23][ion]['none']['cddf'] / divby), color=colors['total'], linestyle='dotted', alpha=alpha, path_effects=_pe, linewidth=_lw)
        ax_bot.plot(plotx, np.log10(dct_totcddf[23][ion]['none']['cddf'] / divby), color=colors['total'], linestyle='dotted', alpha=alpha, path_effects=_pe, linewidth=_lw)
        
        # all halo gas CDDF
        bins = dct_fofcddf[27][ion]['none']['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_bot.plot(plotx, np.log10(dct_fofcddf[27][ion]['none']['none']['cddf'] / divby), color=colors['allhalos'], linestyle='dashed', alpha=alpha, path_effects=patheff, linewidth=linewidth)
        
        bins = dct_fofcddf[23][ion]['none']['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_bot.plot(plotx, np.log10(dct_fofcddf[23][ion]['none']['none']['cddf'] / divby), color=colors['allhalos'], linestyle='dashdot', alpha=alpha, path_effects=patheff, linewidth=linewidth)

        text = r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True))
        if relative:
            ax.text(0.05, 0.05, text, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=fontsize)
        else:
            ax.text(0.95, 0.95, text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize)            

    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid', 'dotted']:
        subcols = list(clist) + [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
    
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    sumhandles = [#mlines.Line2D([], [], color=colors['none'], linestyle='solid', label='FoF no mask', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='solid', label='all gas, $z=0.1$', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='dotted', label='all gas, $z=0.5$', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='dashed', label=r'all halo gas, $z=0.1$', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='dashdot', label=r'all halo gas, $z=0.5$', linewidth=2.),\
                  ]
    sumlabels = ['all gas, $z=0.1$', 'all gas, $z=0.5$', 'all halo gas, $z=0.1$', 'all halo gas, $z=0.5$']
    lax.legend(lcs + sumhandles, ['halo gas, $z=0.1$', 'halo gas, $z=0.5$'] + sumlabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=ncol_legend, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
    
    
def plot_NEW():
    #dfilen = '/net/luttero/data2/specwizard_data/sample3_coldens_EW_subsamples.hdf5'
    dfilen = '/net/luttero/data2/specwizard_data/sample3-6_coldens_EW_subsamples.hdf5'
    percentiles = [5., 50., 95.]
    logNspacing = 0.1
    linemin = 20
    ncols = 3
    ylabel = r'$\log_{10} \, \mathrm{EW} \; [\mathrm{m\AA}]$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    fontsize = 12
    uselines = {'o7': ild.o7major,\
                'o8': ild.o8doublet,\
                'o6': ild.o6major,\
                'ne8': ild.ne8major,\
                'ne9': ild.ne9major,\
                'fe17': ild.fe17major,\
                }
    Nminmax =  {'o7':   (14.5, 18.3),\
                'o8':   (14.8, 17.5),\
                'o6':   (12.8, 17.0),\
                'ne9':  (14.9, 17.4),\
                'ne8':  (13.5, 16.4),\
                'fe17': (14.2, 16.6),\
                }
    logEWminmax = {'o7':   (0.0, 1.8),\
                   'o8':   (0.0, 1.7),\
                   'o6':   (1.0, 3.2),\
                   'ne9':  (0.0, 1.3),\
                   'ne8':  (1.3, 2.8),\
                   'fe17': (0.0, 1.3),\
                   }
    # T vals from plot_Tvir_ions
    #Ion o6   has maximum CIE fraction 0.196, at log T[K] = 5.5, 0.1 max range is [5.29739276 5.75294118]
    #Ion ne8  has maximum CIE fraction 0.233, at log T[K] = 5.8, 0.1 max range is [5.6208641  6.14552226]
    #Ion o7   has maximum CIE fraction 0.994, at log T[K] = 5.9, 0.1 max range is [5.40228139 6.4949395 ]
    #Ion ne9  has maximum CIE fraction 0.981, at log T[K] = 6.2, 0.1 max range is [5.71844286 6.80842876]
    #Ion o8   has maximum CIE fraction 0.448, at log T[K] = 6.4, 0.1 max range is [6.10214446 6.80841602]
    #Ion fe17 has maximum CIE fraction 0.491, at log T[K] = 6.7, 0.1 max range is [6.30526704 7.02535685]

    Tmax_CIE = {'o6':   10**5.5,\
                'ne8':  10**5.8,\
                'o7':   10**5.9,\
                'ne9':  10**6.2,\
                'o8':   10**6.4,\
                'fe17': 10**6.7,\
                }
    bvals_CIE = {ion: np.sqrt(2. * c.boltzmann * Tmax_CIE[ion] / \
                              (ionh.atomw[string.capwords(ol.elements_ion[ion])] * c.u)) \
                      * 1e-5
                 for ion in Tmax_CIE}
    # informed by ~ b par global percentiles
    bvals_indic = {'o7': [45., 230., 20.],\
                   'o8': [80., 320.],\
                   'o6': [20., 120., 5.],\
                   'ne8': [25., 90.],\
                   'ne9': [50., 230.],\
                   'fe17': [50, 180.],\
                   }
    bvals = {ion: [bvals_CIE[ion]] + bvals_indic[ion] for ion in bvals_CIE}
    

    data = {}
    with h5py.File(dfilen, 'r') as fi:
        #selections = list(fi.keys())
        #selections.remove('Header')
        selection = 'full_sample'
        
        subsel = list(fi[selection].keys())
        torem = []
        for sub in subsel:
            if sub.startswith('specnum'):
                torem.append(sub)
        for sub in torem:
            subsel.remove(sub)
        #print(subsel)    
        for ionk in subsel:
            grp = fi['%s/%s'%(selection, ionk)]
            #print(grp.keys())
            logN = np.array(grp['logN_cmm2'])
            EW = np.array(grp['EWrest_A'])
            blin = np.array(grp.attrs['bparfit_cmps_linEW'])[0] * 1e-5
            blog = np.array(grp.attrs['bparfit_cmps_logEW'])[0] * 1e-5
            ion = ionk.split('_')[0]
            data[ion] = {'logN': logN, 'EW': EW,\
                         'blin': blin, 'blog': blog}
                
    ions = list(data.keys())
    ions.sort()
    nions = len(ions)
    
    linestyles = {'med': 'solid',\
                  'out': 'solid',\
                  'EWfit_lin': 'dashed',\
                  'EWfit_log': 'dashed',\
                  'linCOG': 'dotted',\
                  'b_indic': (0.0, [4., 1., 1., 1.])} # [9.6, 2.4, 1.6, 2.4]
    colors = {'data': 'gray',\
              'med':  'green',\
              'out':  'yellowgreen',\
              'EWfit_lin': 'blue',\
              'EWfit_log': 'cyan',\
              'linCOG': 'cadetblue',\
              'b_indic': ['blue', 'red', 'firebrick', 'lightcoral']}
    kws_sublegends = {'handlelength': 1.8,\
                      'handletextpad': 0.5,\
                      'columnspacing': 0.7,\
                      }
    alpha_data = 0.05
    size_data = 10.
    linewidth = 2.
    path_effects = [mppe.Stroke(linewidth=linewidth, foreground="black"), mppe.Stroke(linewidth=linewidth - 0.5)]
    
    panelwidth = 2.8
    panelheight = 2.8
    legheight = 0.5
    wspace = 0.25
    hspace = 0.2
    nrows = nions // ncols
    
    fig = plt.figure(figsize=(panelwidth * ncols + wspace * (ncols - 1), panelheight * nrows + hspace * nrows + legheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows + 1, wspace=wspace, hspace=hspace, height_ratios=[panelheight] * nrows + [legheight])   
    axes = [fig.add_subplot(grid[ioni // ncols, ioni % ncols]) for ioni in range(nions)]
    lax = fig.add_subplot(grid[nrows, :])

    outdata = {}
    for ioni in range(nions):
        ax = axes[ioni]
        ion = ions[ioni]
        if ioni // ncols == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if ioni % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        axlabel = r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True))
        ax.text(0.05, 0.95, axlabel, fontsize=fontsize,\
                verticalalignment='top', horizontalalignment='left',\
                transform=ax.transAxes)
        
        logN = data[ion]['logN']
        EW   = np.log10(data[ion]['EW']) + 3.
        minN = np.min(logN) 
        maxN = np.max(logN)
        minN -= 1e-7 * np.abs(minN)
        maxN += 1e-7 * np.abs(maxN)
        bminN = np.floor(minN / logNspacing) * logNspacing
        bmaxN = np.ceil(maxN / logNspacing) * logNspacing
        Nbins = np.arange(bminN, bmaxN + 0.5 * logNspacing, logNspacing)
        Ncens = Nbins[:-1] + 0.5 * np.diff(Nbins)
        
        bininds = np.array(np.digitize(logN, Nbins))
        EWs_bin = [EW[np.where(bininds == i + 1)] for i in range(len(Ncens))]
        #Ns_bin  = [logN[np.where(bininds == i + 1)] for i in range(len(Ncens))]
        percvals = np.array([np.percentile(EWs, percentiles) \
                                if len(EWs) > 0 else \
                                np.ones(len(percentiles)) * np.NaN
                                for EWs in EWs_bin])
        
        whereplot = np.where(np.array([len(_EW) >= linemin for _EW in EWs_bin]))[0]
        plotmin = whereplot[0]
        plotmax = whereplot[-1]
        #ax.fill_between(Ncens[plotmin : plotmax + 1],\
        #                percvals[plotmin : plotmax + 1, 0],\
        #                percvals[plotmin : plotmax + 1, 2],\
        #                color=colors[key], alpha=0.1)
        ax.scatter(logN, EW, color=colors['data'], alpha=alpha_data, s=size_data, rasterized=True)
        ax.plot(Ncens[plotmin : plotmax + 1],\
                        percvals[plotmin : plotmax + 1, 1],\
                        color=colors['med'], linestyle=linestyles['med'],\
                        label=None, linewidth=linewidth, path_effects=path_effects)
        ax.plot(Ncens[plotmin : plotmax + 1],\
                        percvals[plotmin : plotmax + 1, 0],\
                        color=colors['out'], linestyle=linestyles['out'],\
                        label=None, linewidth=linewidth, path_effects=path_effects)
        ax.plot(Ncens[plotmin : plotmax + 1],\
                        percvals[plotmin : plotmax + 1, 2],\
                        color=colors['out'], linestyle=linestyles['out'],\
                        label=None, linewidth=linewidth, path_effects=path_effects)
        
        outdata[ion] = {'Ncens': Ncens, 'Nbins': Nbins,\
                        'bincount': np.array([len(_EW) for _EW in EWs_bin]),\
                        
                        'percvals_logmA': percvals, 'percentiles': percentiles}
        #Ns_sc = [val for bi in range(plotmin) + range(plotmax + 1, len(Ncens)) for val in Ns_bin[bi]]
        #EWs_sc = [val for bi in range(plotmin) + range(plotmax + 1, len(Ncens)) for val in EWs_bin[bi]]
        #ax.scatter(Ns_sc, EWs_sc, color=colors[key], alpha=0.1)
        
        blin = data[ion]['blin']
        blog = data[ion]['blog']
        
        lines = uselines[ion]
        #EWslin = ild.linflatcurveofgrowth_inv(10**Ncens, blin * 1e5, lines)
        EWslog = ild.linflatcurveofgrowth_inv(10**Ncens, blog * 1e5, lines)
        if isinstance(lines, ild.SpecLine): 
            EWsthin = ild.lingrowthcurve_inv(10**Ncens, lines)
        else:
            EWsthin = np.sum([ild.lingrowthcurve_inv(10**Ncens, lines.speclines[lkey])\
                             for lkey in lines.speclines],axis=0)
        label = '%.0f'%blin
        ax.plot(Ncens, np.log10(EWsthin) + 3., color=colors['linCOG'], linestyle=linestyles['linCOG'], path_effects=path_effects)
        #ax.plot(Ncens, np.log10(EWslin) + 3., color=colors['EWfit_lin'], label=None, linestyle=linestyles['EWfit_lin'])
        ax.plot(Ncens, np.log10(EWslog) + 3., color=colors['EWfit_log'], linestyle=linestyles['EWfit_log'], label=label, path_effects=path_effects)
        
        bvals_this = bvals[ion]
        for bi in range(len(bvals_this)):
            bval = bvals_this[bi]
            EWs = np.log10(ild.linflatcurveofgrowth_inv(10**Ncens, bval * 1e5, lines)) + 3.
            label = '%.0f'%bval
            ax.plot(Ncens, EWs, linestyle=linestyles['b_indic'], color=colors['b_indic'][bi],\
                    linewidth=linewidth, label=label, path_effects=path_effects)
        
        hnd, lab = ax.get_legend_handles_labels()
        if len(lab) > 4:
            ncol_lr = 2
            bbta = (1.04, -0.04)
        else:
            ncol_lr = 1
            bbta = (1.02, -0.02)
        leg1 = ax.legend(hnd[:1], lab[:1], fontsize=fontsize, loc='upper left',\
                         ncol=1, frameon=False, bbox_to_anchor=(-0.015, 0.91),\
                         **kws_sublegends)
        leg2 = ax.legend(hnd[1:], lab[1:], fontsize=fontsize, loc='lower right',\
                         frameon=False, bbox_to_anchor=bbta,\
                         ncol=ncol_lr, **kws_sublegends)
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        
        ax.set_xlim(Nminmax[ion])
        ax.set_ylim(logEWminmax[ion])
        pu.setticks(ax, fontsize=fontsize)
        
    lcs = []
    line = [[(0, 0)]]    
    # set up the proxy artist    
    for ls in [linestyles['b_indic']]:
        subcols = [mpl.colors.to_rgba(color) for color in colors['b_indic']]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    #selhandles = [mlines.Line2D([], [], color=colors[key], linestyle='solid', label=key.split('_')[0] + ' sample') 
    #              for key in colors]
    #sellabels = [key.split('_')[0] + ' sample' for key in colors]
    handles1 = [mlines.Line2D([], [], color=colors['med'], linestyle=linestyles['med'], label='median', path_effects=path_effects),\
                mlines.Line2D([], [], color=colors['out'], linestyle=linestyles['out'], label='%.0f %%'%(percentiles[2] - percentiles[0]), path_effects=path_effects),\
                mlines.Line2D([], [], color=colors['linCOG'], linestyle=linestyles['linCOG'], label='opt. thin', path_effects=path_effects),\
                mlines.Line2D([], [], color=colors['EWfit_log'], linestyle=linestyles['EWfit_log'], label='best-fit b', path_effects=path_effects),\
                mlines.Line2D([], [], color=colors['b_indic'][0], linestyle=linestyles['b_indic'], label=r'$b(T_{\max, \mathrm{CIE}})$', path_effects=path_effects),\
                ]
    labels1 = ['median', '%.0f %%'%(percentiles[2] - percentiles[0]), 'opt. thin', r'best-fit $b$', r'$b(T_{\max, \mathrm{CIE}})$']
    lax.legend(handles1 + lcs, labels1 + [r'var. $b \; [\mathrm{km} \, \mathrm{s}^{-1}]$'],\
               handler_map={type(lc): HandlerDashedLines()},\
               fontsize=fontsize, ncol=ncols, loc='upper center', bbox_to_anchor=(0.5, 0.6),\
               handlelength=1.8)
    lax.axis('off')
    
    plt.savefig('/net/luttero/data2/specwizard_data/sample3-6_specwizard-NEW_wbparfit_fullsample.pdf', format='pdf', bbox_inches='tight')
    #return outdata

def savedata_plotform_3dhists(minrshow=0.05, minrshow_kpc=None):
    '''
    minrshow: minimum radius in R200c units
    minrshow_kpc: additional criterion: minimum units in kpc, calculated from
                  R200c at the halfway mass for the mass bin
    '''
    weighttypes = ['Mass', 'Volume', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    outdir = '/net/luttero/data2/imgs/CGM/3dprof/'
    minrstr = str(minrshow)
    if minrshow_kpc is not None:
        minrstr += '-and-{}-kpc'.format(minrshow_kpc)
    outfile = 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrstr)
    
    hists_main = {}
    edges_main = {}
    edges_rmin = {}
    galids_main = {}  
    lminrshow = np.log10(minrshow)
    hists_rmin = {}
    hists_rmin_norm = {}
    for _wt in weighttypes:
        if _wt in ol.elements_ion.keys():
            filename = ol.ndir + 'particlehist_Nion_%s_L0100N1504_27_test3.4_PtAb_T4EOS_galcomb.hdf5'%(_wt)
            tgrpn = '3Dradius_Temperature_T4EOS_Density_T4EOS_Niondens_%s_PtAb_T4EOS_R200c_snapdata'%(_wt)
            axns  = {'r3d':  '3Dradius',\
                    'T':    'Temperature_T4EOS',\
                    'rho':  'Density_T4EOS',\
                    'nion': 'Niondens_%s_PtAb_T4EOS'%(_wt),\
                    }
            axnl = ['nion', 'T', 'rho']
            
            elts_Z = [ol.elements_ion[_wt]]
                        
        else:
            if _wt == 'Volume':
                filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.4_T4EOS_galcomb.hdf5'%('propvol')
            else:
                filename = ol.ndir + 'particlehist_%s_L0100N1504_27_test3.4_T4EOS_galcomb.hdf5'%(_wt)
            tgrpn = '3Dradius_Temperature_T4EOS_Density_T4EOS_R200c_snapdata'
            axns = {'r3d':  '3Dradius',\
                    'T':    'Temperature_T4EOS',\
                    'rho':  'Density_T4EOS',\
                    }
            axnl = ['T', 'rho']
            elts_Z = ['oxygen', 'neon', 'iron']
            
        tgrpns_Z = ['3Dradius_SmoothedElementAbundance-%s_T4EOS_R200c_snapdata_corrZ'%(string.capwords(elt)) for elt in elts_Z]
        axns_Z = [{'r3d': '3Dradius',\
                   'Z_%s'%(elt): 'SmoothedElementAbundance-%s_T4EOS'%(string.capwords(elt))} \
                  for elt in elts_Z]
        axnl_Z = [['Z_%s'%(elt)] for elt in elts_Z]
        
        hists_main[_wt] = {}
        edges_main[_wt] = {}
        edges_rmin[_wt] = {}
        galids_main[_wt] = {}
        hists_rmin[_wt] = {}
        hists_rmin_norm[_wt] = {}
        
        rbinu = 'R200c'
        combmethod = 'addnormed-R200c'
        mgrpn = 'L0100N1504_27_Mh0p5dex_1000/%s-%s'%(combmethod, rbinu)
        
        with h5py.File(filename, 'r') as fi:
            for _tgrpn, _axns, _axnl in zip([tgrpn] + tgrpns_Z, [axns] + axns_Z, [axnl] + axnl_Z):
                print(_tgrpn, _axns, _axnl)
                grp = fi[_tgrpn + '/' + mgrpn]
                sgrpns = list(grp.keys())
                massbins = [grpn.split('_')[-1] for grpn in sgrpns]    
                massbins = [[np.log10(float(val)) for val in binn.split('-')] for binn in massbins]
                
                for mi in range(len(sgrpns)):
                    mkey = massbins[mi][0]
                    if minrshow_kpc is not None:
                        _minr_massbin = np.log10(minrshow_kpc / R200c_pkpc(10**mkey, cosmopars_ea_27))
                    else:
                        _minr_massbin = -np.inf
                    _minr_massbin = max(_minr_massbin, lminrshow)
                    print('Using minimum radius {minr} * R200c for mass bin {minm}-{maxm}'.format(minr=_minr_massbin, minm=mkey, maxm=massbins[mi][1]))
                    grp_t = grp[sgrpns[mi]]
                    hist = np.array(grp_t['histogram'])
                    if bool(grp_t['histogram'].attrs['log']):
                        hist = 10**hist
                    
                    edges = {}
                    axes = {}
                    for axn in _axns:
                       edges[axn] = np.array(grp_t[_axns[axn] + '/bins'])
                       if not bool(grp_t[_axns[axn]].attrs['log']):
                           edges[axn] = np.log10(edges[axn])
                       axes[axn] = grp_t[_axns[axn]].attrs['histogram axis']  
                    
                    # create empty dicts once mass keys are known, but don't erase previously stored ones (other _tgrpn)
                    if mkey not in edges_main[_wt].keys(): 
                        edges_main[_wt][mkey] = {}
                        edges_rmin[_wt][mkey] = {}
                        hists_main[_wt][mkey] = {}
                        hists_rmin[_wt][mkey] = {}
                        hists_rmin_norm[_wt][mkey] = {}
                    
                    # apply normalization consisent with stacking method
                    if rbinu == 'pkpc':
                        edges['r3d'] += np.log10(c.cm_per_mpc * 1e-3)
                    
                    if combmethod == 'addnormed-R200c':
                        if rbinu != 'R200c':
                            raise ValueError('The combination method addnormed-R200c only works with rbin units R200c')
                        _i = np.where(np.isclose(edges['r3d'], 0.))[0]
                        if len(_i) != 1:
                            raise RuntimeError('For addnormed-R200c combination, no or multiple radial edges are close to R200c:\nedges [R200c] were: %s'%(str(edges['r3d'])))
                        _i = _i[0]
                        _a = range(len(hist.shape))
                        _s = [slice(None, None, None) for dummy in _a]
                        _s[axes['r3d']] = slice(None, _i, None)
                        norm_t = np.sum(hist[tuple(_s)])
                    hist *= (1. / norm_t)
                    
                    for pt in _axnl:
                        rax = axes['r3d']
                        yax = axes[pt]
                        
                        edges_r = np.copy(edges['r3d'])
                        edges_y = np.copy(edges[pt])
                        
                        hist_t = np.copy(hist)
                        
                        # deal with edge units (r3d is already in R200c units if R200c-stacked)
                        if edges_r[0] == -np.inf: # reset centre bin position
                            edges_r[0] = 2. * edges_r[1] - edges_r[2] 
                        if edges_y[0] == -np.inf: # reset centre bin position
                            edges_y[0] = 2. * edges_y[1] - edges_y[2]
                        if edges_y[-1] == np.inf: # reset centre bin position
                            edges_y[-1] = 2. * edges_y[-2] - edges_y[-3]
                        if pt == 'rho':
                            edges_y += np.log10(rho_to_nh)
                            
                        sax = list(range(len(hist_t.shape)))
                        sax.remove(rax)
                        sax.remove(yax)
                        hist_t = np.sum(hist_t, axis=tuple(sax))
                        if yax < rax:
                            hist_t = hist_t.T
                        #hist_t /= (np.diff(edges_r)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
                        
                        hists_main[_wt][mkey][pt] = hist_t
                        edges_main[_wt][mkey][pt] = [edges_r, edges_y]
                        #print(hist_t.shape)
                        
                        # sum everything up to minrshow (keep 1 column < minrshow)
                        try:
                            rminind = np.where(np.isclose(edges_r, _minr_massbin))[0][0]
                        except IndexError:
                            rminind = np.argmax(_minr_massbin < edges_r)
                        sl_sum = slice(0, rminind, None)
                        sl_new = slice(rminind - 1, None, None)
                        firstcol = np.sum(hist_t[sl_sum, :], axis=0)
                        hist_tt = np.copy(hist_t[sl_new, :])
                        hist_tt[0, :] = firstcol
                        edges_r_tt = np.copy(edges_r[sl_new])
                        edges_r_tt[0] = edges_r[0]
                        
                        hists_rmin[_wt][mkey][pt] = hist_tt
                        edges_rmin[_wt][mkey][pt] = [edges_r_tt, edges_y]
                        
                        # normalize histogram to bin size
                        hists_rmin_norm[_wt][mkey][pt] = hist_tt / (np.diff(edges_r_tt)[:, np.newaxis] * np.diff(edges_y)[np.newaxis, :])
            
                        
                    # add in cumulative plot for the weight (from rho-T-nion histograms)
                    if 'Smoothed' not in _tgrpn:
                        hist_t = np.copy(hist)
                        sax = list(range(len(hist_t.shape)))
                        sax.remove(rax)
                        hist_t = np.sum(hist_t, axis=tuple(sax))
                        hist_t = np.cumsum(hist_t)
                        hists_main[_wt][mkey]['weight'] = hist_t
                        edges_main[_wt][mkey]['weight'] = edges_r[1:]
                        
                    if mkey in galids_main[_wt]:
                        if not np.all(np.array(grp_t['galaxyids']) == galids_main[_wt][mkey]):
                            raise RuntimeError('Different histograms have different galaxy sets in the same mass bin: %s, %s'%(_wt, mkey))
                    else:
                        galids_main[_wt][mkey] = np.array(grp_t['galaxyids'])
                
    with h5py.File(outdir + outfile, 'w') as fo:
        masskeys = list(edges_main[weighttypes[0]].keys())
        fo.create_dataset('mass_keys', data=np.array([np.string_('%.1f'%_mkey) for _mkey in masskeys]))
        fo.create_dataset('weights', data=np.array([np.string_(_wt) for _wt in weighttypes]))
        fo.create_dataset('mass_bins', data=np.array(massbins))
        fo.create_dataset('histogram_types', data=np.array([np.string_(name) for name in ['rho', 'T', 'nion', 'weight', 'Z_oxygen', 'Z_neon', 'Z_iron']]))
        units = {'r3d': np.string_('log10 r3d / R200c'),\
                 'T':   np.string_('log10 T / K'),\
                 'rho': np.string_('log10 nH / cm**-3, from rho with X=0.752'),\
                 'nion': np.string_('log10 nion / cm**-3'),\
                 'Z_oxygen': np.string_('log10 mass fraction'),\
                 'Z_iron': np.string_('log10 mass fraction'),\
                 'Z_neon': np.string_('log10 mass fraction'),\
                 }
        
        for _wt in weighttypes:
            for mkey in masskeys:
                print(list(edges_main[_wt][mkey].keys()))
                for htype in edges_main[_wt][mkey].keys():
                    grp = fo.create_group('%s/%.1f/%s'%(_wt, mkey, htype))
                    if htype == 'weight':
                        grp.create_dataset('hist', data=hists_main[_wt][mkey][htype])
                        grp.create_dataset('edges', data=edges_main[_wt][mkey][htype])
                        grp['edges'].attrs.create('units', units['r3d'])
                        grp['hist'].attrs.create('units', np.string_('cumulative enclosed fraction rel. to R200c'))
                    else:
                        grp.create_dataset('hist', data=hists_main[_wt][mkey][htype])
                        grp.create_dataset('edges_0', data=edges_main[_wt][mkey][htype][0])
                        grp.create_dataset('edges_1', data=edges_main[_wt][mkey][htype][1])
                        grp['edges_0'].attrs.create('units', units['r3d'])
                        grp['edges_1'].attrs.create('units', units[htype])
                        
                        grp.create_dataset('hist_rmin', data=hists_rmin[_wt][mkey][htype])
                        grp.create_dataset('edges_rmin_0', data=edges_rmin[_wt][mkey][htype][0])
                        grp.create_dataset('edges_rmin_1', data=edges_rmin[_wt][mkey][htype][1])
                        grp['edges_rmin_0'].attrs.create('units', units['r3d'])
                        grp['edges_rmin_1'].attrs.create('units', units[htype])
                        
                        grp.create_dataset('hist_rmin_norm', data=hists_rmin_norm[_wt][mkey][htype])
                        grp['hist_rmin_norm'].attrs.create('units', np.string_('cumulative enclosed fraction rel. to R200c / Delta edges_0 / Delta edges_1'))
                        
                
def plot3Dprof_niceversion(variation, minrshow=0.05, Mhrange=(12., 12.5), weighttype='Mass', ionset='all'):
    '''
    variation: 'cumul' - cumulative mass and ion fraction plots for each halo mass
               'focus-Mass' - plot of T, nH weighted by ion weight in Mhrange
                              ionset: use all ions, or a listed subset
               not implemented 'focus-weight' - plot T, nH for different masses weighted by weighttype
    '''
    outdir = '/net/luttero/data2/imgs/CGM/3dprof/'
    weighttypes = ['Mass', 'Volume', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
        
    fontsize = 12
    cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    cmap.set_under(cmap(0.))
    percentiles = [0.05, 0.50, 0.95]
    linestyles = ['dashed', 'solid', 'dashed']
    
    rbinu = 'R200c'
    combmethod = 'addnormed-R200c'
    #binq = 'M200c_Msun'
    binqn = r'$\mathrm{M}_{\mathrm{200c}} \, [\mathrm{M}_{\odot}]$' 
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
    wnames = {weighttype: r'\mathrm{%s}'%(ild.getnicename(weighttype, mathmode=True)) if weighttype in ol.elements_ion.keys() else \
                          r'\mathrm{Mass}' if weighttype == 'Mass' else \
                          r"\mathrm{'Vol.'}" if weighttype == 'Volume' else \
                          None \
              for weighttype in weighttypes}
    
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'\log_{10} \, \mathrm{Z} \, / \, \mathrm{Z}_{\odot}',\
                #'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%('q', 'q') 
                }
    clabel = r'$\log_{10} \, \left\langle %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}}) \right\rangle \, / \,$'%('q', 'q') + 'bin size'
    
    saveddata = outdir + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrshow)
    if not os.path.isfile(saveddata):
        savedata_plotform_3dhists(minrshow=minrshow)
                
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1., foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
    cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    cmap.set_under(cmap(0.))
    
    with h5py.File(saveddata, 'r') as df:
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        massbins = np.array(df['mass_bins'])
        
        if variation == 'cumul':
            outname = outdir + 'cumul_radprofs_L0100N1504_27_Mh0p5dex_1000.pdf'
            
            figsize = (11., 3.) # full-page figure
            masskeys.sort()
            masskeys = masskeys[1::2]
            nmasses = len(masskeys)
            
            xlim = (np.log10(minrshow), np.log10(4.2)) # don't care too much about the inner details, extracted out to 4 * R200c
            ylim = (-3.9, 2.1)
    
            if nmasses != 3:
                raise RuntimeError('Cumulative profile plot is set up for 3 mass bins; found %i'%nmasses)
            ncols = 4
            nrows = 1
            
            fig = plt.figure(figsize=figsize)
            grid = gsp.GridSpec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0.0, width_ratios=[1.] * ncols, height_ratios=[1.] * nrows )
            axes = np.array([fig.add_subplot(grid[mi // ncols, mi % ncols]) for mi in range(nmasses)])
            lax  = fig.add_subplot(grid[nrows - 1, -1])
            
            colors = ioncolors.copy()
            colors.update({'Mass': 'black', 'Volume': 'gray'})
            linestyle_kw = {'Mass': {'linestyle': 'solid'},\
                            'Volume': {'linestyle': 'solid'},\
                            'o6':   {'dashes': [6, 2]},\
                            'o7':   {'dashes': [3, 1]},\
                            'o8':   {'dashes': [1, 1]},\
                            'ne8':  {'dashes': [6, 2, 3, 2]},\
                            'ne9':  {'dashes': [6, 2, 1, 2]},\
                            'fe17': {'dashes': [3, 1, 1, 1]},\
                            }
            
            for mi in range(nmasses):
                mkey = masskeys[mi]
                binind = np.where(np.array(massbins)[:, 0] == float(mkey))[0][0]
                ax = axes[mi]
                
                # add mass range indicator            
                text = r'$%.1f \, \endash \, %.1f$'%(massbins[binind][0], massbins[binind][1]) #r'$\log_{10} \,$' + binqn + r':            
                ax.text(0.05, 0.95, text, fontsize=fontsize, transform=ax.transAxes,\
                        horizontalalignment='left', verticalalignment='top')
                
                # axis labels and ticks
                labelx = (mi // ncols == nrows - 1) or (mi // ncols == nrows - 2 and nmasses % ncols > 0 and  nmasses % ncols <= mi % ncols)
                labely = mi % ncols == 0
                
                # set up axis labels and ticks
                setticks(ax, top=True, left=True, labelleft=labely, labelbottom=labelx, fontsize=fontsize)
                if labelx:
                    ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
                if labely:
                    ax.set_ylabel(r'$\log_{10} \, \mathrm{q}(<r) \, / \, \mathrm{q}(< \mathrm{R}_{\mathrm{200c}})$', fontsize=fontsize)
                
                # plot the data
                yq = 'weight'
                for weight in weighttypes:
                    hist = np.array(df['%s/%s/%s/hist'%(weight, mkey, yq)])
                    edges_r = np.array(df['%s/%s/%s/edges'%(weight, mkey, yq)])
                    #print(hist)
                    #print(edges_r)
                    if np.any(np.isnan(hist)) or np.any(np.isnan(edges_r)):
                        print('Got NaN values for %s, %s'%(weight, mkey))
                        print('edges: %s'%edges_r)
                        print('hist.: %s'%hist)
                    
                    ax.plot(edges_r, np.log10(hist), color=colors[weight],\
                                alpha=1.,\
                                path_effects=None, linewidth=linewidth + 0.5,\
                                label=r'$%s$'%(wnames[weight]),\
                                **linestyle_kw[weight])
                    ax.set_xlim(xlim)
                    
            # add legend
            handles, labels = axes[0].get_legend_handles_labels()
            leg = lax.legend(handles=handles, fontsize=fontsize, loc='upper left',\
                       bbox_to_anchor=(0.01, 0.80), ncol=2,\
                       handlelength=2.5, handletextpad=0.5, columnspacing=1.0,\
                       frameon=True)
            leg.set_title(r'quantity $q$', prop={'size': fontsize})
            lax.axis('off')
            
            # sync y lim ranges
            #ylims = np.array([_ax.get_ylim() for _ax in axes])
            #y0 = np.min(ylims[:, 0])
            #y1 = np.max(ylims[:, 1])
            [_ax.set_ylim(ylim) for _ax in axes]
            
            plt.savefig(outname, format='pdf', bbox_inches='tight')
         
        elif variation == 'focus-Mass':
            if isinstance(ionset, (list, np.array, tuple)):
                weighttypes_panel = list(ionset)
                outname = outdir + 'focus_radprofs_ions-%s_M200c-%.1f-%.1f_L0100N1504_27_Mh0p5dex_1000.pdf'%(('-'.join(sorted(weighttypes_panel)),) + tuple(Mhrange))
            elif ionset is None or ionset == 'all':
                weighttypes_panel = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] 
                outname = outdir + 'focus_radprofs_ions_M200c-%.1f-%.1f_L0100N1504_27_Mh0p5dex_1000.pdf'%(Mhrange)
                
            weighttypes_overlay = ['Mass', 'Volume']
            colors_overlay = {'Mass': 'turquoise',\
                              'Volume': 'yellowgreen',\
                              'ion': 'lightsalmon'}
            color_Tcie = 'red'
            color_Tvir = 'blue'
            linestyle_Tindic = 'dotted'
            percentiles = [5., 50., 95.]
            percstyles = ['dashed', 'solid', 'dashed']
            
            
            
            mind = np.where(np.isclose([float(_mk) for _mk in masskeys], Mhrange[0]))[0][0]
            mkey = masskeys[mind]
            binind = np.where(np.isclose(np.array(massbins)[:, 0], Mhrange[0]))[0][0]
            masstext = r'$%.1f \, \endash \, %.1f$'%(massbins[binind][0], massbins[binind][1])
            
            if len(weighttypes_panel) > 3:
                figsize = (11., 11.) # full-page figure
            else:
                figsize = (11., 6.)
            nions = len(weighttypes_panel)
            ncols = 3
            nrows = ((nions - 1) // ncols + 1) * 2
                    
            xlim = (np.log10(minrshow), np.log10(4.0)) # don't care too much about the inner details, extracted out to 4 * R200c
            #ylim = (-3.9, 2.1)
    
                
            fig = plt.figure(figsize=figsize)
            grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=0.0, wspace=0.0, width_ratios=[1.] * ncols + [0.5], height_ratios=[1.] * nrows )
            axes = np.array([[fig.add_subplot(grid[xi, yi]) for xi in range(nrows)] for yi in range(ncols)])
            if nrows > 2:
                lax  = fig.add_subplot(grid[0:2, ncols])
                cax =  fig.add_subplot(grid[2:, ncols])
                lbbox = (0.02, 0.78)
                caspect = 10.
            else:
                lax  = fig.add_subplot(grid[0:1, ncols])
                cax =  fig.add_subplot(grid[1:, ncols])
                lbbox = (0.02, 0.99)
                caspect = 7.
            hists_rmin_norm = {ion: {axn: np.array(df['%s/%s/%s/hist_rmin_norm'%(ion, mkey, axn)]) for axn in ['rho', 'T']} for ion in weighttypes}
            hists_rmin = {ion: {axn: np.array(df['%s/%s/%s/hist_rmin'%(ion, mkey, axn)]) for axn in ['rho', 'T']} for ion in weighttypes}
            edges0_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_0'%(ion, mkey, axn)]) for axn in ['rho', 'T']} for ion in weighttypes}
            edges1_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_1'%(ion, mkey, axn)]) for axn in ['rho', 'T']} for ion in weighttypes}
            
            vmax = np.log10(np.max([[np.max(hists_rmin_norm[ion][axn]) for axn in ['rho', 'T']] for ion in weighttypes_panel]))
            vmin = np.log10(np.min([[np.min(hists_rmin_norm[ion][axn]) for axn in ['rho', 'T']] for ion in weighttypes_panel]))
            vmin = max(vmin, vmax - 7.)
            
            for ii in range(nions):
                xi = ii % ncols
                yi = ii // ncols
                axnh = axes[xi, 2 * yi]
                axt  = axes[xi, 2 * yi + 1]
                ion = weighttypes_panel[ii]
                
                labelx = yi == nrows // 2 - 1
                labely = xi == 0
                
                # set up axes
                setticks(axnh, top=True, left=True, labelleft=labely, labelbottom=False, fontsize=fontsize)
                setticks(axt,  top=True, left=True, labelleft=labely, labelbottom=labelx, fontsize=fontsize)
                if labelx:
                    axt.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
                if labely:
                    axt.set_ylabel(axlabels['T'], fontsize=fontsize)
                    axnh.set_ylabel(axlabels['rho'], fontsize=fontsize)
                axt.set_xlim(xlim)
                axnh.set_xlim(xlim)
                
                # nH plot
                for ax, yq in zip([axnh, axt], ['rho', 'T']):
                    ax.text(0.95, 0.95, r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)),\
                            fontsize=fontsize, transform=ax.transAxes,\
                            verticalalignment='top', horizontalalignment='right')
                    
                    edges_r = edges0_rmin[ion][yq]
                    edges_y = edges1_rmin[ion][yq]
                    hist = hists_rmin[ion][yq]
                    hist_n = hists_rmin_norm[ion][yq]
                    
                    # pixdens False: already normalized
                    img, _1, _2 = pu.add_2dplot(ax, hist_n, [edges_r, edges_y], toplotaxes=(0, 1),\
                                  log=True, usepcolor=True, pixdens=False,\
                                  cmap=cmap, vmin=vmin, vmax=vmax, zorder=-2)
                    
                    perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles) / 100.)
                    mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                    
                    for pi in range(len(percentiles)):
                        ax.plot(mid_r, perclines[pi], color=colors_overlay['ion'],\
                                linestyle=percstyles[pi], alpha=1.,\
                                path_effects=patheff_thick, linewidth=linewidth + 1)
                    
                    for wt in weighttypes_overlay:
                        hist = hists_rmin[wt][yq]
                        edges_r = edges0_rmin[wt][yq]
                        edges_y = edges1_rmin[wt][yq]
                        perclines = pu.percentiles_from_histogram(hist, edges_y, axis=1, percentiles=np.array(percentiles) / 100.)
                        mid_r = edges_r[:-1] + 0.5 * np.diff(edges_r)
                         
                        for pi in range(len(percentiles)):
                            ax.plot(mid_r, perclines[pi], color=colors_overlay[wt],\
                                    linestyle=percstyles[pi], alpha=1.,\
                                    path_effects=patheff, linewidth=linewidth + 0.5)
                
                axt.axhline(Tranges_CIE[ion][0], color=color_Tcie, linestyle=linestyle_Tindic, linewidth=linewidth)
                axt.axhline(Tranges_CIE[ion][1], color=color_Tcie, linestyle=linestyle_Tindic, linewidth=linewidth)
                axt.axhline(np.log10(T200c_hot(10**massbins[binind][0], cosmopars)),\
                            color=color_Tvir, linestyle=linestyle_Tindic, linewidth=linewidth )
                axt.axhline(np.log10(T200c_hot(10**massbins[binind][1], cosmopars)),\
                            color=color_Tvir, linestyle=linestyle_Tindic, linewidth=linewidth )
            # color bar 
            pu.add_colorbar(cax, img=img, vmin=vmin, vmax=vmax, cmap=cmap,\
                            clabel=clabel, fontsize=fontsize, orientation='vertical',\
                            extend='min')
            cax.set_aspect(caspect)
            cax.tick_params(labelsize=fontsize - 1)
            
            # legend
            typehandles = [mlines.Line2D([], [], color=colors_overlay[key], label='%s-weighted'%(key),\
                                         path_effects=patheff_thick if key=='ion' else patheff,\
                                         linewidth=linewidth + 1 if key=='ion' else linewidth + 0.5,\
                                         ) for key in colors_overlay]
            perchandles = [mlines.Line2D([], [], color='gray', label='median', linewidth=linewidth + 0.75,\
                                         linestyle=percstyles[1]),\
                           mlines.Line2D([], [], color='gray', label='%s %%'%(percentiles[2] - percentiles[0]),\
                                         linewidth=linewidth + 0.75,\
                                         linestyle=percstyles[0]),\
                          ]
            thandles = [mlines.Line2D([], [],  color=color_Tcie, linestyle=linestyle_Tindic, linewidth=linewidth,\
                                      label='CIE range'),\
                        mlines.Line2D([], [],  color=color_Tvir, linestyle=linestyle_Tindic, linewidth=linewidth,\
                                      label=r'$\mathrm{T}_{\mathrm{200c}}$ range'),\
                        ]
            #lax.text(0.1, 0.95, masstext, horizontalalignment='left',\
            #         verticalalignment='top', fontsize=fontsize,\
            #         transform=lax.transAxes)
            lax.legend(handles=typehandles + perchandles + thandles, fontsize=fontsize,\
                       loc='upper left', bbox_to_anchor=lbbox,\
                       frameon=True, ncol=1)
            lax.axis('off')
            
            # sync y limits on plots
            ylims = np.array([axes[xi, 2 * yi].get_ylim() for xi in range(ncols) for yi in range(nrows // 2)])
            y0 = np.min(ylims[:, 0])
            y1 = np.max(ylims[:, 1])
            [axes[xi, 2 * yi].set_ylim((y0, y1)) for xi in range(ncols) for yi in range(nrows // 2)]
             
            ylims = np.array([axes[xi, 2 * yi + 1].get_ylim() for xi in range(ncols) for yi in range(nrows // 2)])
            y0 = np.min(ylims[:, 0])
            y1 = np.max(ylims[:, 1])
            [axes[xi, 2 * yi + 1].set_ylim((y0, y1)) for xi in range(ncols) for yi in range(nrows // 2)]
                
            plt.savefig(outname, format='pdf', box_inches='tight')

def plot3Dprof_haloprop(minrshow=0.05, minrshow_kpc=15., Zshow='oxygen'):
    '''
    mass- and Volume-weighted rho, T, Z profiles for different halo masses
    in R200c units, stacked weighting each halo by 1 / weight in R200c
    '''
    
    outdir = '/net/luttero/data2/imgs/CGM/3dprof/'
    outname = 'profiles_3d_halo_rho_T_Z-%s_median.pdf'%Zshow
    weighttypes = ['Mass', 'Volume']
    elts_Z = [Zshow] #['oxygen', 'neon', 'iron']
    solarZ = ol.solar_abunds_ea
    massslice = slice(None, None, 2)
        
    fontsize = 12
    #cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    #cmap.set_under(cmap(0.))
    percentiles = [50.]
    linestyles = {'Volume': 'solid',\
                  'Mass': 'dashed'}
    
    if len(elts_Z) > 1:
        alphas = {'oxygen': 1.0,\
                  'neon':   0.7,\
                  'iron':   0.4,\
                  }
    else:
        alphas = {elts_Z[0]: 1.0}
    
    #rbinu = 'R200c'
    #combmethod = 'addnormed-R200c'
    #binq = 'M200c_Msun'
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
   
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \, / \, \mathrm{Z}_{\odot}$',\
                #'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%('q', 'q') 
                }
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\mathrm{200c}}]$'
    
    minrstr = str(minrshow)
    if minrshow_kpc is not None:
        minrstr += '-and-{}-kpc'.format(minrshow_kpc)
    saveddata = outdir + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrstr)
    if not os.path.isfile(saveddata):
        savedata_plotform_3dhists(minrshow=minrshow, minrshow_kpc=minrshow_kpc)
                
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1., foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
    
    fig = plt.figure(figsize=(10., 3.5))
    grid = gsp.GridSpec(nrows=1, ncols=6, hspace=0.0, wspace=0.0,\
                        width_ratios=[1., 0.3, 1., 0.3, 1., 0.25],\
                        bottom=0.15, left=0.05)
    axes = np.array([fig.add_subplot(grid[0, 2*i]) for i in range(3)])
    #lax = fig.add_subplot(grid[1, :2])
    cax = fig.add_subplot(grid[0, 5])
    
    # set up color bar (sepearte to leave white spaces for unused bins)
    massedges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.])
    cmapname = 'rainbow'
    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
    clist[1::2] = np.array([1., 1., 1., 1.])
    keys = sorted(massedges)
    colordct = {keys[i]: clist[i] for i in range(len(keys))}
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
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)
    
    #cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
    #                               orientation='vertical', clabel=clabel, fontsize=fontsize, aspect=8.)
    axplot = {'T': 0,\
              'rho': 1,\
              'Z_oxygen': 2,\
              'Z_neon': 2,\
              'Z_iron': 2,\
              }

    with h5py.File(saveddata, 'r') as df:
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        massbins = np.array(df['mass_bins'])
        # use every other mass bin for legibility
        massbins = sorted(massbins, key=lambda x: x[0])
        massbins = massbins[massslice]
        
        for key in axplot.keys():
            axi = axplot[key]
            ax = axes[axi]
            setticks(ax, top=True, left=True, labelleft=True, labelbottom=True, fontsize=fontsize)
            #ax.set_ylabel(axlabels[key.split('_')[0]], fontsize=fontsize)
            #ax.text(0.98, 0.98, axlabels[key.split('_')[0]], fontsize=fontsize,\
            #        transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
            
            ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, /\, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            ax.set_ylabel(axlabels[key.split('_')[0]], fontsize=fontsize)
            
        for Mhrange in massbins:
            mind = np.where(np.isclose([float(_mk) for _mk in masskeys], Mhrange[0]))[0][0]
            mkey = masskeys[mind]
            #binind = np.where(np.isclose(np.array(massbins)[:, 0], Mhrange[0]))[0][0]

            typelist = ['rho', 'T'] + ['Z_%s'%(elt) for elt in elts_Z]
            hists_rmin = {ion: {axn: np.array(df['%s/%s/%s/hist_rmin'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            edges0_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_0'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            edges1_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_1'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            
            color = colordct[float(mkey)]
            for axn in typelist:
                for ion in weighttypes:
                    if axn.startswith('Z_'):
                        alpha = alphas[axn[2:]]
                    else:
                        alpha = 1.
                    _hist = hists_rmin[ion][axn]
                    _e0 = edges0_rmin[ion][axn]
                    _e0 = _e0[:-1] + 0.5 * np.diff(_e0) 
                    _e0[0] = 2. * _e0[1] - _e0[2] # was already adjusted; no need to extend the plot too far to low r
                    _e1 = edges1_rmin[ion][axn]
                    perclines = pu.percentiles_from_histogram(_hist, _e1, axis=1, percentiles=np.array(percentiles) / 100.)
                    if axn.startswith('Z_'):
                        perclines -= np.log10(solarZ[axn[2:]])
                    axes[axplot[axn]].plot(_e0, perclines[0], color=color,\
                        alpha=alpha, linestyle=linestyles[ion],\
                        linewidth=linewidth, path_effects=patheff)
    
    masskeys = sorted(masskeys, key=lambda x: float(x))
    medges = masskeys[massslice]
    #print(masskeys)
    #print(medges)
    for ed in medges:
        ind = np.where([ed == mkey for mkey in masskeys])[0][0]
        tval1 = np.log10(T200c_hot(10**float(ed), cosmopars))
        axes[axplot['T']].axhline(tval1,\
                                  color=colordct[float(ed)], zorder=-1,\
                                  linestyle='dotted')
        if ind + 1 < len(masskeys):
            ed2 = masskeys[ind + 1]
            tval2 = np.log10(T200c_hot(10**float(ed2), cosmopars))
            axes[axplot['T']].axhline(tval2,\
                                  color=colordct[float(ed)], zorder=-1,\
                                  linestyle='dotted')
    
    # legend
    typehandles = [mlines.Line2D([], [], linestyle=linestyles[key],\
                                 label='%s-weighted'%(key),\
                                 path_effects=patheff,\
                                 linewidth=linewidth,\
                                 color = 'gray'
                                 ) for key in weighttypes]
    if len(elts_Z) > 1:
        thandles = [mlines.Line2D([], [], linestyle='solid',\
                                     label=key,\
                                     path_effects=patheff,\
                                     linewidth=linewidth,\
                                     color='gray',\
                                     alpha=alphas[key],\
                                     ) for key in elts_Z]
    else:
        thandles = []
    handles=typehandles + thandles 
    #lax.text(0.1, 0.95, masstext, horizontalalignment='left',\
    #         verticalalignment='top', fontsize=fontsize,\
    #         transform=lax.transAxes)
    axes[2].legend(handles=handles, fontsize=fontsize,\
               loc='lower left', bbox_to_anchor=(0.0, 0.0),\
               frameon=False, ncol=1) #ncol=min(4, len(handles))
    #lax.axis('off')
        
    plt.savefig(outdir + outname, format='pdf', box_inches='tight')
      

def plot3Dprof_haloprop_talkversion(minrshow=0.05, Zshow='oxygen', num=0, fmt='pdf'):
    '''
    mass- and Volume-weighted rho, T, Z profiles for different halo masses
    in R200c units, stacked weighting each halo by 1 / weight in R200c
    
    num: 0 -> empty, 1-7: adding volume-weighted, 8-14: adding mass-weighted
    '''
    
    outdir = '/home/wijers/Documents/papers/cgm_xray_abs/talk_figures/'
    outname = 'profiles_3d_halo_rho_T_Z-%s_median_num%i.%s'%(Zshow, num, fmt)
    weighttypes = ['Volume', 'Mass']
    elts_Z = [Zshow] #['oxygen', 'neon', 'iron']
    solarZ = ol.solar_abunds_ea
        
    fontsize = 16
    #cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    #cmap.set_under(cmap(0.))
    percentiles = [50.]
    linestyles = {'Volume': 'solid',\
                  'Mass': 'dashed'}
    
    if len(elts_Z) > 1:
        alphas = {'oxygen': 1.0,\
                  'neon':   0.7,\
                  'iron':   0.4,\
                  }
    else:
        alphas = {elts_Z[0]: 1.0}
    
    #rbinu = 'R200c'
    #combmethod = 'addnormed-R200c'
    #binq = 'M200c_Msun'
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
   
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \, / \, \mathrm{Z}_{\odot}$',\
                #'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%('q', 'q') 
                }
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\mathrm{200c}}]$'
    
    saveddata = '/net/luttero/data2/imgs/CGM/3dprof/' + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrshow)
    #if not os.path.isfile(saveddata):
    #    savedata_plotform_3dhists(minrshow=minrshow)
                
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1., foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
    
    fig = plt.figure(figsize=(10., 3.5))
    grid = gsp.GridSpec(nrows=1, ncols=4, wspace=0.27, width_ratios=[3., 3., 3., 0.5], bottom=0.2)
    axes = np.array([fig.add_subplot(grid[0, i]) for i in range(3)])
    #lax = fig.add_subplot(grid[1, :2])
    cax = fig.add_subplot(grid[0, 3])
    cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
                                   orientation='vertical', clabel=clabel, fontsize=fontsize, aspect=8.)
    axplot = {'T': 0,\
              'rho': 1,\
              'Z_oxygen': 2,\
              'Z_neon': 2,\
              'Z_iron': 2,\
              }
    xlim = (-0.95, 0.7)
    ylim = {'T': (3.75, 7.9),\
            'rho': (-6.8, -1.),\
            'Z_oxygen': (-4.4, 0.7),\
            'Z_neon': (-4.4, 0.7),\
            'Z_iron': (-4.4, 0.7),\
            }

    with h5py.File(saveddata, 'r') as df:
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        masskeys = sorted(masskeys, key=float)
        massbins = np.array(df['mass_bins'])
        sortkeys = np.argsort(massbins[:, 0])
        massbins = massbins[sortkeys]
        
        for key in axplot.keys():
            axi = axplot[key]
            ax = axes[axi]
            setticks(ax, top=True, left=True, labelleft=True, labelbottom=True, fontsize=fontsize)
            #ax.set_ylabel(axlabels[key.split('_')[0]], fontsize=fontsize)
            ax.text(0.98, 0.98, axlabels[key.split('_')[0]], fontsize=fontsize,\
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')
            
            ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, /\, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim[key])
            
        for Mhrange in massbins:
            mind = np.where(np.isclose([float(_mk) for _mk in masskeys], Mhrange[0]))[0][0]
            mkey = masskeys[mind]
            #binind = np.where(np.isclose(np.array(massbins)[:, 0], Mhrange[0]))[0][0]

            typelist = ['rho', 'T'] + ['Z_%s'%(elt) for elt in elts_Z]
            hists_rmin = {ion: {axn: np.array(df['%s/%s/%s/hist_rmin'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            edges0_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_0'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            edges1_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_1'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            
            color = colordct[float(mkey)]
            for axi in range(len(typelist)):
                axn = typelist[axi]
                for ioni in range(len(weighttypes)):
                    ion = weighttypes[ioni]
                    if len(massbins) - mind - 1 + len(massbins) * ioni < num: # mass bins high-to-low, Volume, then Mass
                        if axn.startswith('Z_'):
                            alpha = alphas[axn[2:]]
                        else:
                            alpha = 1.
                        _hist = hists_rmin[ion][axn]
                        _e0 = edges0_rmin[ion][axn]
                        _e0 = _e0[:-1] + 0.5 * np.diff(_e0) 
                        _e1 = edges1_rmin[ion][axn]
                        perclines = pu.percentiles_from_histogram(_hist, _e1, axis=1, percentiles=np.array(percentiles) / 100.)
                        if axn.startswith('Z_'):
                            perclines -= np.log10(solarZ[axn[2:]])
                        axes[axplot[axn]].plot(_e0, perclines[0], color=color,\
                            alpha=alpha, linestyle=linestyles[ion],\
                            linewidth=linewidth, path_effects=patheff)
        
    medges = sorted([float(key) for key in masskeys])
    for edi in range(len(medges)):
        ed = medges[edi]
        if len(medges) - edi - 1 < num:
            axes[axplot['T']].axhline(np.log10(T200c_hot(10**float(ed), cosmopars)),\
                                      color=colordct[float(ed)], zorder=-1,\
                                      linestyle='dotted')
    
    # legend
    typehandles = [mlines.Line2D([], [], linestyle=linestyles[key],\
                                 label='%s'%(key),\
                                 path_effects=patheff,\
                                 linewidth=linewidth,\
                                 color = 'gray'
                                 ) for key in weighttypes]
    if len(elts_Z) > 1:
        thandles = [mlines.Line2D([], [], linestyle='solid',\
                                     label=key,\
                                     path_effects=patheff,\
                                     linewidth=linewidth,\
                                     color='gray',\
                                     alpha=alphas[key],\
                                     ) for key in elts_Z]
    else:
        thandles = []
    handles=typehandles + thandles 
    #lax.text(0.1, 0.95, masstext, horizontalalignment='left',\
    #         verticalalignment='top', fontsize=fontsize,\
    #         transform=lax.transAxes)
    axes[2].legend(handles=handles, fontsize=fontsize,\
               loc='lower left', bbox_to_anchor=(0.0, 0.0),\
               frameon=False, ncol=1) #ncol=min(4, len(handles))
    #lax.axis('off')
        
    plt.savefig(outdir + outname, format=fmt, box_inches='tight')

      
def plot3Dprof_ionw(minrshow=0.05, ions=('o6', 'o7', 'o8'), axnl=('rho', 'T', 'Z')):
    '''
    ion-weighted rho, T, Z profiles for different halo masses
    in R200c units, stacked weighting each halo by 1 / weight in R200c
    if Mass or Volume is specified, oxygen Z values are used
    '''
    
    outdir = '/net/luttero/data2/imgs/CGM/3dprof/'
    weighttypes = list(ions)
    axnl = list(axnl)
    outname = 'profiles_3d_halo_%s_%s_median.pdf'%('-'.join(axnl), '-'.join(weighttypes))
    solarZ = ol.solar_abunds_ea
        
    fontsize = 12
    #cmap = truncate_colormap(cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7, n=-1)
    #cmap.set_under(cmap(0.))
    percentiles = [50.]
    alpha = 1.
    
    #rbinu = 'R200c'
    #combmethod = 'addnormed-R200c'
    #binq = 'M200c_Msun'
    cosmopars = {'a': 0.9085634947881763,\
                 'boxsize': 67.77,\
                 'h': 0.6777,\
                 'omegab': 0.0482519,\
                 'omegalambda': 0.693,\
                 'omegam':  0.307,\
                 'z': 0.10063854175996956,\
                 } # avoid having to read in the halo catalogue just for this; copied from there
    
   
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \, / \, \mathrm{Z}_{\odot}$',\
                #'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%('q', 'q') 
                }
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\mathrm{200c}}]$'
    
    saveddata = outdir + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrshow)
    if not os.path.isfile(saveddata):
        savedata_plotform_3dhists(minrshow=minrshow)
                
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    patheff_thick = [mppe.Stroke(linewidth=linewidth + 1., foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
       
    figwidth = 11.
    numions = len(weighttypes)
    numpt   = len(axnl)
    ncols = min(3, numions)
    nrows = ((numions - 1) // ncols + 1) * numpt
    cwidth = 0.5
    wspace = 0.0
    panelwidth = (figwidth - cwidth- wspace * ncols) / ncols
    panelheight = panelwidth
    figheight =  panelheight * nrows
    
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=0.0, wspace=wspace, width_ratios=[panelwidth] * ncols + [cwidth], height_ratios=[1.] * nrows)
    axes = np.array([[fig.add_subplot(grid[ii // ncols + ti, ii % ncols]) for ti in range(numpt)] for ii in range(numions)])
    #lax = fig.add_subplot(grid[1, :2])
    cax = fig.add_subplot(grid[:min(nrows, 2), ncols])
    cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
                                   orientation='vertical', clabel=clabel, fontsize=fontsize, aspect=10.)

    with h5py.File(saveddata, 'r') as df:
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        massbins = np.array(df['mass_bins'])
        
        for ii in range(numions):
            for ti in range(numpt):
                ion = ions[ii]
                axn = axnl[ti]
                ax = axes[ii, ti]
                
                labelleft = ii % ncols == 0
                labelbottom = (numions - ii <= ncols and ti == numpt - 1)
                
                setticks(ax, top=True, left=True, right=True, labelleft=labelleft, labelbottom=labelbottom, fontsize=fontsize)
                if labelleft:
                    ax.set_ylabel(axlabels[axn], fontsize=fontsize)
                if labelbottom:
                    ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, /\, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
        
                for Mhrange in massbins:
                    mind = np.where(np.isclose([float(_mk) for _mk in masskeys], Mhrange[0]))[0][0]
                    mkey = masskeys[mind]
                    #binind = np.where(np.isclose(np.array(massbins)[:, 0], Mhrange[0]))[0][0]
        
                    if axn == 'Z':
                        if ion in ol.elements_ion.keys():
                            elt = ol.elements_ion[ion]
                        else:
                            elt = 'oxygen' #default
                        _axn = 'Z_{}'.format(elt)
                    else:
                        _axn = axn
                    if ion in ol.elements_ion.keys():
                        name = ild.getnicename(ion, mathmode=True)
                    else:
                        name = ion
                    ax.text(0.95, 0.95, r'$\mathrm{%s}$-weighted'%(name), fontsize=fontsize,\
                            transform=ax.transAxes, horizontalalignment='right',\
                            verticalalignment='top', fontweight='normal')
                    
                    _hist = np.array(df['%s/%s/%s/hist_rmin'%(ion, mkey, _axn)]) 
                    _e0 = np.array(df['%s/%s/%s/edges_rmin_0'%(ion, mkey, _axn)])
                    _e1 = np.array(df['%s/%s/%s/edges_rmin_1'%(ion, mkey, _axn)]) 
                    _e0 = _e0[:-1] + 0.5 * np.diff(_e0)   
                    
                    alpha = 1.
                    color = colordct[float(mkey)]
                    perclines = pu.percentiles_from_histogram(_hist, _e1, axis=1, percentiles=np.array(percentiles) / 100.)
                    if axn == 'Z':
                        perclines -= np.log10(solarZ[elt])
                    ax.plot(_e0, perclines[0], color=color,\
                        alpha=alpha, linestyle='solid',\
                        linewidth=linewidth, path_effects=patheff)
                    
                    if axn == 'T':
                        ax.axhline(np.log10(T200c_hot(10**float(mkey), cosmopars)),\
                                                  color=colordct[float(mkey)], zorder=-1,\
                                                  linestyle='dotted')
                        
                        if ion in ol.elements_ion:
                            ax.axhline(Tranges_CIE[ion][0],\
                                       color='black', zorder=-1,\
                                       linestyle='dotted')
                            ax.axhline(Tranges_CIE[ion][1],\
                                       color='black', zorder=-1,\
                                       linestyle='dotted')
    # sync y axes:
    for ti in range(numpt):
        ylims = np.array([ax.get_ylim() for ax in axes[:, ti]])
        y0 = np.min(ylims[:, 0])
        y1 = np.max(ylims[:, 1])
        [ax.set_ylim(y0, y1) for ax in axes[:, ti]]
        
    plt.savefig(outdir + outname, format='pdf', box_inches='tight')
            

def plot_radprof_mstar(ions=None, var='main', fontsize=fontsize):
    '''
    var:  main (just the Mstar bins) or appendix (lots more info)
    main paper: main-fcov-break for covering fractions at the CDDF break
                main-fcov-obs   for covering fractions at obs. limits     
    appendix:
        - Mhalo plots (mvir -> pkpc)
        - Mhalo subsample of Mstar (mvir -> pkpc)
        - Mhalo subsample of Mstar (pkpc stacked)
        - Mstar sample (pkpc stacked)
    '''
    xlim = None
    if var == 'main':
        techvars_touse = [3]
        highlightcrit = {'techvars': [3]} 
        ytype='perc'
        yvals_toplot=[10., 50., 90.]
        printnumgals=False
    elif var == 'main-fcov':
        techvars_touse = [3]
        highlightcrit = None
        ytype='fcov'
        yvals_toplot = {'o6':   [14.3, 13.5],\
                        'ne8':  [13.7, 13.5],\
                        'o7':   [16.0, 15.5],\
                        'ne9':  [15.3, 15.5],\
                        'o8':   [16.0, 15.7],\
                        'fe17': [15.0, 14.9]}
        printnumgals=False
    elif var == 'main-fcov-break':
        techvars_touse = [3]
        highlightcrit = None
        ytype='fcov'
        yvals_toplot = {'o6':   [14.3],\
                        'ne8':  [13.7],\
                        'o7':   [16.0],\
                        'ne9':  [15.3],\
                        'o8':   [16.0],\
                        'fe17': [15.0]}
        printnumgals=False
        xlim = (1., 1.5e3)
    elif var == 'main-fcov-obs':
        techvars_touse = [3]
        highlightcrit = None
        ytype='fcov'
        yvals_toplot = {'o6':   [13.5],\
                        'ne8':  [13.5],\
                        'o7':   [15.5],\
                        'ne9':  [15.5],\
                        'o8':   [15.7],\
                        'fe17': [14.9]}
        printnumgals=False
    elif var == 'appendix':
        techvars_touse = range(4)
        highlightcrit = {'techvars': []} # busy enough
        ytype='perc'
        yvals_toplot=[10., 50., 90.]
        printnumgals=False
    
    mdir = '/net/luttero/data2/imgs/CGM/radprof/'
    if ions is None:
        ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    
    imgname = 'radprof_bystellarmass_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6.25slice_zcen-all_%s_techvars-%s_%s.pdf'%('-'.join(sorted(ions)), var, '-'.join(sorted([str(tvar) for tvar in techvars_touse])), ytype)
    imgname = mdir + imgname        
    if (ytype=='perc' and 50.0 not in yvals_toplot):
        imgname = imgname[:-4] + '_yvals-%s'%('-'.join([str(val) for val in yvals_toplot])) + '.pdf'
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    cmapname = 'rainbow'
    #hatches = ['\\', '/', '|', 'o', '+', '*', '-', 'x', '.']
    #sumcolor = 'saddlebrown'
    totalcolor = 'black'
    shading_alpha = 0.45 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\star} \; [\mathrm{M}_{\odot}]$'
    linestyles_fcov = ['solid', 'dashed', 'dotted', 'dotdash']
    cosmopars = cosmopars_ea_27
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_Mhalo = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         #'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         }
    
    ion_filedct_Mstar = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         #'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         }

    # define used mass ranges
    Mh_edges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    galsetnames_hmassonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    
    Ms_edges = np.array([8.7, 9.7, 10.3, 10.8, 11.1, 11.3, 11.5, 11.7])
    Ms_mins = list(Ms_edges[:-1])
    Ms_maxs = list(Ms_edges[1:]) 
    Ms_base = ['geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_names = ['logMstar_Msun_1000_geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_sels = [('logMstar_Msun', Ms_mins[i], Ms_maxs[i]) if Ms_maxs[i] is not None else\
               ('logMstar_Msun', Ms_mins[i], np.inf)\
               for i in range(len(Ms_mins))]
    galsetnames_smass = {name: sel for name, sel in zip(Ms_names, Ms_sels)}
    
    matchvals_Mstar_Mhalo = {'geq8.7_le9.7':   (11.0, 11.5),\
                             'geq9.7_le10.3':  (11.5, 12.0),\
                             'geq10.3_le10.8': (12.0, 12.5),\
                             'geq10.8_le11.1': (12.5, 13.0),\
                             'geq11.1_le11.3': (13.0, 13.5),\
                             'geq11.3_le11.5': (13.5, 14.0),\
                             'geq11.5_le11.7': (14.0, 14.6),\
                             }
    Ms_hmatch_names = ['logMstar_Msun_1000_%s_M200c-%.1f-%.1f-sub'%(key, matchvals_Mstar_Mhalo[key][0], matchvals_Mstar_Mhalo[key][1]) for key in Ms_base]
    galsetnames_smass_hmatch = {name: sel for name, sel in zip(Ms_hmatch_names, Ms_sels)}
    
    # to put R200c-binned values into pkpc plots
    R200c_toassume_vals = [11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25]    
    R200c_toassume = {name: val for name, val in zip(Ms_hmatch_names, R200c_toassume_vals)}
    R200c_toassume.update({name: val for name, val in zip(Mh_names, R200c_toassume_vals)})
    
    techvars = {0: {'filenames': ion_filedct_Mhalo, 'setnames': list(galsetnames_hmassonly.keys()), 'setfills': None, 'units': 'R200c'},\
                1: {'filenames': ion_filedct_Mstar, 'setnames': Ms_hmatch_names, 'setfills': None, 'units': 'R200c'},\
                2: {'filenames': ion_filedct_Mstar, 'setnames': Ms_hmatch_names, 'setfills': None, 'units': 'pkpc'},\
                3: {'filenames': ion_filedct_Mstar, 'setnames': Ms_names, 'setfills': None, 'units': 'pkpc'},\
                }
    
    linewidths = {0: 2.5,\
                  1: 1.5,\
                  2: 1.5,\
                  3: 2.5,\
                  }
       
    linestyles = {0: 'dashed',\
                  1: 'dotted',\
                  2: 'dashdot',\
                  3: 'solid',\
                  }
    
    alphas = {0: 1.,\
              1: 1.,\
              2: 1.,\
              3: 1.,\
              }
    
    legendnames_techvars = {0: r'$\mathrm{M}_{\mathrm{200c}}$, $\mathrm{R}_{\mathrm{200c}}$-stacked',\
                            1: r'$\mathrm{M}_{\mathrm{200c}} + \mathrm{M}_{\star}$, $\mathrm{R}_{\mathrm{200c}}$-stacked',\
                            2: r'$\mathrm{M}_{\mathrm{200c}} + \mathrm{M}_{\star}$, pkpc-stacked',\
                            3: r'$\mathrm{M}_{\star}$, pkpc-stacked',\
                            }
    
    if isinstance(yvals_toplot, dict):
        readpaths = {tv: {ion: {val: '%s_bins/binset_0/%s_%s'%(techvars[tv]['units'], ytype, val) for val in yvals_toplot[ion]} for ion in ions} for tv in techvars}
        readpath_bins = {tv: '/'.join((readpaths[tv][ions[0]][yvals_toplot[ions[0]][0]]).split('/')[:-1]) + '/bin_edges' for tv in techvars}
    else:
        readpaths = {tv: {ion: {val: '%s_bins/binset_0/%s_%s'%(techvars[tv]['units'], ytype, val) for val in yvals_toplot} for ion in ions} for tv in techvars}
        readpath_bins = {tv: '/'.join((readpaths[tv][ions[0]][yvals_toplot[0]]).split('/')[:-1]) + '/bin_edges' for tv in techvars}
        
    print(readpaths)
    panelwidth = 2.5
    panelheight = 2.
    if len(techvars_touse) > 1:
        legheight = 1.3
    else:
        legheight = 0.0
    cwidth = 0.6
    if ytype == 'perc':
        wspace = 0.2
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + cwidth + wspace * numcols
    figheight = numcols * panelheight + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace,\
                        width_ratios=[panelwidth] * numcols + [cwidth],\
                        height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if len(techvars_touse) > 1:
        lax  = fig.add_subplot(grid[numrows, :])
    
    yvals = {}
    #cosmopars = {}
    #fcovs = {}
    #dXtot = {}
    #dztot = {}
    #dXtotdlogN = {}
    bins = {}
    numgals = {}
    
    for var in techvars_touse:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var]['filenames'][ion]
            goaltags = techvars[var]['setnames']
            setfills = techvars[var]['setfills']
            
            #_units   = techvars[var]['units']
            if ion not in filename:
                raise RuntimeError('File %s attributed to ion %s, mismatch'%(filename, ion))
            
            if setfills is None:
                with h5py.File(ol.pdir + 'radprof/' + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    for galset in galsets:
                        ex = True
                        for val in readpaths[var][ion].keys():
                            try:
                                temp = np.array(fi[galset + '/' + readpaths[var][ion][val]])
                            except KeyError:
                                ex = False
                                break
                        
                        if ex:
                            tags[fi[galset].attrs['seltag'].decode()] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())
                    #print(goaltags)
                    #print(tags.keys())
                    if len(tags_unread) > 0:
                        print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins[var]])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[var][ion][val])]) for val in readpaths[var][ion].keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = ol.pdir + 'radprof/' + filename%(fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in readpaths[var][ion].keys():
                                try:
                                    temp = np.array(fi[galset + '/' + readpaths[var][ion][val]])
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
                            print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[var][ion][val])]) for val in readpaths[var][ion].keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
    ## checks: will fail with e.g. halo-only projection, though
    #filekeys = h5files.keys()
    #if np.all([np.all(bins[key] == bins[filekeys[0]]) if len(bins[key]) == len(bins[filekeys[0]]) else False for key in filekeys]):
    #    bins = bins[filekeys[0]]
    #else:
    #    raise RuntimeError("bins for different files don't match")
    
    #if not np.all(np.array([np.all(hists[key]['nomask'] == hists[filekeys[0]]['nomask']) for key in filekeys])):
    #    raise RuntimeError('total histograms from different files do not match')
    if printnumgals:
       print('tech vars: 0 = 1 slice, all, 1 = 1 slice, off-edge, 2 = 2 slices, all, 3 = 2 slices, off-edge')
       print('\n')
       
       for ion in ions:
           for var in techvars_touse:
               tags = techvars[var]['setnames']
               if var in [0, 2]:
                   tags = sorted(tags, key=galsetnames_hmassonly.__getitem__)
               print('%s, var %s:'%(ion, var))
               print('\n'.join(['%s\t%s'%(tag, numgals[var][ion][tag]) for tag in tags]))
               print('\n')
       return numgals
   
    # assumes matching numbers of Mstar, Mh bins
    hmassranges = [sel[1:] for sel in Mh_sels]
    smassranges = [sel[1:] for sel in Ms_sels]
    #print(massranges)
    hmassedges = sorted(list(set([np.log10(val) for rng in hmassranges for val in rng])))
    smassedges = sorted(list(set([val for rng in smassranges for val in rng])))
    #print(massedges)
    if hmassedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        hmassedges[-1] = 2. * hmassedges[-2] - hmassedges[-3]
    hmasslabels1 = {name: tuple(np.log10(np.array(galsetnames_hmassonly[name][1:]))) for name in galsetnames_hmassonly.keys()}
    smasslabels2 = {name: tuple(galsetnames_smass_hmatch[name][1:]) for name in galsetnames_smass_hmatch.keys()}
    smasslabels3 = {name: tuple(galsetnames_smass[name][1:]) for name in galsetnames_smass.keys()}
    
    clist = cm.get_cmap(cmapname, len(smassedges) - 1)(np.linspace(0., 1., len(smassedges) - 1))
    _hmasks1 = sorted(hmasslabels1.keys(), key=hmasslabels1.__getitem__)
    colors = {_hmasks1[i]: clist[i] for i in range(len(_hmasks1))}
    _smasks2 = sorted(smasslabels2.keys(), key=smasslabels2.__getitem__)
    colors.update({_smasks2[i]: clist[i] for i in range(len(_smasks2))})
    _smasks3 = sorted(smasslabels3.keys(), key=smasslabels3.__getitem__)
    colors.update({_smasks3[i]: clist[i] for i in range(len(_smasks3))})
    #del _masks
    masslabels_all = hmasslabels1
    masslabels_all.update(smasslabels2)
    masslabels_all.update(smasslabels3)

    #print(clist)
    cmap = mpl.colors.ListedColormap(clist)
    #cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(smassedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=smassedges,\
                                ticks=smassedges,\
                                spacing='proportional', extend='neither',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(9.)
    
    #print(clist)
    
    # annotate color bar with sample size per bin
    #if indicatenumgals:
    #    ancolor = 'black'
    #    for tag in masslabels.keys():
    #        ypos = masslabels[tag]
    #        xpos = 0.5
    #        cax.text(xpos, (ypos - massedges[0]) / (massedges[-2] - massedges[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
        
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]
        if isinstance(yvals_toplot, dict):
            _yvals_toplot = yvals_toplot[ion]
        else:
            _yvals_toplot = yvals_toplot

        if ytype == 'perc':
            if ion[0] == 'h':
                ax.set_ylim(12.0, 21.0)
            #else:
            #    ax.set_ylim(11.5, 17.0)
            elif ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o6':
                ax.set_ylim(10.8, 14.8)
            elif ion == 'ne9':
                ax.set_ylim(11.9, 15.9)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(10**-1.95, 10**0.3)
            ax.set_yscale('log')
        
        if xlim is not None:
            ax.set_xlim(*xlim)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        if ytype == 'perc':
            ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        else:
            ax.text(0.05, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='left', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        #hatchind = 0
        for vi in range(len(techvars_touse)):
            tags = techvars[techvars_touse[vi]]['setnames']
            _units = techvars[techvars_touse[vi]]['units']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                
                try:
                    plotx = bins[var][ion][tag]
                    if _units == 'R200c': # plots are in pkpc -> need to convert to pkpc in some reasonable way
                        basesize = R200c_pkpc(10**R200c_toassume[tag], cosmopars)
                        plotx *= basesize
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                if highlightcrit is not None: #highlightcrit={'techvars': [0], 'Mmin': [10.0, 12.0, 14.0]}
                    matched = True
                    _highlightcrit = highlightcrit
                    _highlightcrit['Mmin'] = \
                        10.3 if ion == 'o6' else \
                        10.3 if ion == 'ne8' else \
                        10.8 if ion == 'o7' else \
                        11.1 if ion == 'ne9' else \
                        11.1 if ion == 'o8' else \
                        11.1 if ion == 'fe17' else \
                        np.inf 
                    if 'techvars' in highlightcrit.keys():
                        matched &= var in _highlightcrit['techvars']
                    if 'Mmin' in highlightcrit.keys():
                        matched &= np.min(np.abs(masslabels_all[tag][0] - np.array(_highlightcrit['Mmin']))) <= 0.01
                    if matched:
                        yvals_toplot_temp = _yvals_toplot
                    else:
                        yvals_toplot_temp = [_yvals_toplot[0]] if len(_yvals_toplot) == 1 else [_yvals_toplot[1]]
                else:
                    yvals_toplot_temp = _yvals_toplot
                
                
                if ytype == 'perc':
                    if len(yvals_toplot_temp) == 3:
                        yval = yvals_toplot_temp[0]
                        try:                      
                            ploty1 = yvals[var][ion][tag][yval]
                        except KeyError:
                            print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                        yval = yvals_toplot_temp[2]
                        try:                      
                            ploty2 = yvals[var][ion][tag][yval]
                        except KeyError:
                            print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val)) 
                        # according to stackexchange, this is the only way to set the hatch color in matplotlib 2.0.0 (quasar); does require the same color for all hatches...
                        #plt.rcParams['hatch.color'] = (0.5, 0.5, 0.5, alphas[var] * shading_alpha,) #mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,)
                        #ax.fill_between(plotx, ploty1, ploty2, color=(0., 0., 0., 0.), hatch=hatches[hatchind], facecolor=mpl.colors.to_rgb(colors[tag]) + (alphas[var] * shading_alpha,), edgecolor='face', linewidth=0.0)
                        ax.fill_between(plotx, ploty1, ploty2, color=colors[tag], alpha=alphas[var] * shading_alpha, label=masslabels_all[tag])
                        
                        #hatchind += 1
                        yvals_toplot_temp = [yvals_toplot_temp[1]]
                        
                    if len(yvals_toplot_temp) == 1:
                        yval = yvals_toplot_temp[0]
                        try:
                            ploty = yvals[var][ion][tag][yval]
                        except KeyError:
                            print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, val))
                            continue
                        if yval == 50.0: # only highlight the medians
                            patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
                        else:
                            patheff = []
                        ax.plot(plotx, ploty, color=colors[tag], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=masslabels_all[tag], path_effects=patheff)

                elif ytype == 'fcov':
                    for yi in range(len(yvals_toplot_temp)):
                        linestyle = linestyles_fcov[yi]
                        yval = yvals_toplot_temp[yi]
                        ploty = yvals[var][ion][tag][yval]
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
                        ax.plot(plotx, ploty, color=colors[tag], linestyle=linestyle, linewidth=linewidths[var], alpha=alphas[var], label=masslabels_all[tag], path_effects=patheff)
                        
                    
        if ytype == 'perc':
            #ax.axhline(0., color=totalcolor, linestyle='solid', linewidth=1.5, alpha=0.7)
            #xlim = ax.get_xlim()
            ax.axhline(approx_breaks[ion], 0., 0.1, color='gray', linewidth=1.5, zorder=-1) # ioncolors[ion]
        if ytype == 'fcov': # add value labels
            linewidth = np.average([linewidths[var] for var in techvars_touse])
            alpha = np.average([alphas[var] for var in techvars_touse])
            patheff = [] #[mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
            color = 'gray'
            if len(yvals_toplot_temp) > 1:
                legend_handles = [mlines.Line2D([], [], linewidth=linewidth,\
                                                alpha=alpha, color=color,\
                                                path_effects=patheff,\
                                                label='%.1f'%yvals_toplot_temp[yi],\
                                                linestyle=linestyles_fcov[yi]) \
                                  for yi in range(len(yvals_toplot_temp))]
                ax.legend(handles=legend_handles, fontsize=fontsize - 1,\
                          loc='upper right', bbox_to_anchor=(1.02, 1.02),\
                          columnspacing=1.2, handletextpad=0.5,\
                          frameon=False, ncol=2)
            else:
                panellabel = '$ > %.1f$'%yvals_toplot_temp[yi]
                ax.text(0.95, 0.95, panellabel, fontsize=fontsize, \
                        verticalalignment='top', horizontalalignment='right',\
                        transform=ax.transAxes)
    #lax.axis('off')
        ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = list(clist) #+ [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var], linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    #sumhandles = [mlines.Line2D([], [], color=sumcolor, linestyle='solid', label='all halos', linewidth=2.),\
    #              mlines.Line2D([], [], color=totalcolor, linestyle='solid', label='total', linewidth=2.)]
    #sumlabels = ['all halos', 'total']
    if len(techvars_touse) > 1:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse], handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 0.))
        lax.axis('off')
    #leg1 = lax.legend(handles=legend_handles, fontsize=fontsize-1, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)
    #leg2 = lax.legend(handles=legend_handles_ls,fontsize=fontsize-1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
    #lax.add_artist(leg1)
    #lax.add_artist(leg2)
    #ax1.text(0.02, 0.05, r'absorbers close to galaxies at $z=0.37$', horizontalalignment='left', verticalalignment='bottom', transform=ax1.transAxes, fontsize=fontsize)
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    

def getcovfrac_total_halo(minr_pkpc, maxr_r200c):
    '''
    for an ion, get the fraction of the total halo covering fraction (<maxr) 
    and the total cddf contribution (< minr, <maxr) for absorber above the 
    cddf break, assuming no halo overlaps 
    '''
    fcovset = 'break'
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    
    halocat='/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    # rounded1 set used in the paper
    mstarbins = np.array([   -np.inf,  7.1,  7.9,  8.7,  9.7, 10.3, 10.8, 11.1, 11.3, 11.5, 11.7], dtype=np.float32)
    m200cbins = np.array([-np.inf, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.0, 14.6])
    with h5py.File(halocat, 'r') as cat:
        m200c = np.log10(np.array(cat['M200c_Msun']))
        mstar = np.log10(np.array(cat['Mstar_Msun']))
    
    #m200cbins = np.array(m200cbins)
    #expand = (np.floor(np.min(m200c) / binround) * binround, np.ceil(np.max(m200c) / binround) * binround)
    #m200cbins = np.append(expand[0], m200cbins)
    #m200cbins = np.append(m200cbins, expand[1])
    

    if fcovset == 'break':
        techvars_touse = [3]
        ytype='fcov'
        yvals_toplot = {'o6':   [14.3],\
                        'ne8':  [13.7],\
                        'o7':   [16.0],\
                        'ne9':  [15.3],\
                        'o8':   [16.0],\
                        'fe17': [15.0]}
    elif fcovset == 'obs':
        techvars_touse = [3]
        ytype='fcov'
        yvals_toplot = {'o6':   [13.5],\
                        'ne8':  [13.5],\
                        'o7':   [15.5],\
                        'ne9':  [15.5],\
                        'o8':   [15.7],\
                        'fe17': [14.9]}

    cosmopars = cosmopars_ea_27
    # up to 2.5 Rvir / 500 pkpc

    ion_filedct_Mstar = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals_fullrdist_stored_profiles.hdf5',\
                         #'hneutralssh': 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                         }

    # define used mass ranges
    Mh_edges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    galsetnames_hmassonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    
    Ms_edges = np.array([8.7, 9.7, 10.3, 10.8, 11.1, 11.3, 11.5, 11.7])
    Ms_mins = list(Ms_edges[:-1])
    Ms_maxs = list(Ms_edges[1:]) 
    Ms_base = ['geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_names = ['logMstar_Msun_1000_geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_sels = [('logMstar_Msun', Ms_mins[i], Ms_maxs[i]) if Ms_maxs[i] is not None else\
               ('logMstar_Msun', Ms_mins[i], np.inf)\
               for i in range(len(Ms_mins))]
    galsetnames_smass = {name: sel for name, sel in zip(Ms_names, Ms_sels)}
    
    matchvals_Mstar_Mhalo = {'geq8.7_le9.7':   (11.0, 11.5),\
                             'geq9.7_le10.3':  (11.5, 12.0),\
                             'geq10.3_le10.8': (12.0, 12.5),\
                             'geq10.8_le11.1': (12.5, 13.0),\
                             'geq11.1_le11.3': (13.0, 13.5),\
                             'geq11.3_le11.5': (13.5, 14.0),\
                             'geq11.5_le11.7': (14.0, 14.6),\
                             }
    Ms_hmatch_names = ['logMstar_Msun_1000_%s_M200c-%.1f-%.1f-sub'%(key, matchvals_Mstar_Mhalo[key][0], matchvals_Mstar_Mhalo[key][1]) for key in Ms_base]
    galsetnames_smass_hmatch = {name: sel for name, sel in zip(Ms_hmatch_names, Ms_sels)}
    
    # to put R200c-binned values into pkpc plots
    R200c_toassume_vals = [11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25]    
    R200c_toassume = {name: val for name, val in zip(Ms_hmatch_names, R200c_toassume_vals)}
    R200c_toassume.update({name: val for name, val in zip(Mh_names, R200c_toassume_vals)})
    
    techvars = {3: {'filenames': ion_filedct_Mstar, 'setnames': Ms_names, 'setfills': None, 'units': 'pkpc'},\
                }
    
    if isinstance(yvals_toplot, dict):
        readpaths = {tv: {ion: {val: '%s_bins/binset_0/%s_%s'%(techvars[tv]['units'], ytype, val) for val in yvals_toplot[ion]} for ion in ions} for tv in techvars}
    else:
        readpaths = {tv: {ion: {val: '%s_bins/binset_0/%s_%s'%(techvars[tv]['units'], ytype, val) for val in yvals_toplot} for ion in ions} for tv in techvars}
    readpath_bins = {tv: '/'.join((readpaths[tv][ions[0]][yvals_toplot[ions[0]][0]]).split('/')[:-1]) + '/bin_edges' for tv in techvars}
    print(readpaths)

    
    yvals = {}
    #cosmopars = {}
    #fcovs = {}
    #dXtot = {}
    #dztot = {}
    #dXtotdlogN = {}
    bins = {}
    numgals = {}
    
    for var in techvars_touse:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var]['filenames'][ion]
            goaltags = techvars[var]['setnames']
            setfills = techvars[var]['setfills']
            
            #_units   = techvars[var]['units']
            if ion not in filename:
                raise RuntimeError('File %s attributed to ion %s, mismatch'%(filename, ion))
            
            if setfills is None:
                with h5py.File(ol.pdir + 'radprof/' + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    for galset in galsets:
                        ex = True
                        for val in readpaths[var][ion].keys():
                            try:
                                temp = np.array(fi[galset + '/' + readpaths[var][ion][val]])
                            except KeyError:
                                ex = False
                                break
                        
                        if ex:
                            tags[fi[galset].attrs['seltag'].decode()] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())
                    #print(goaltags)
                    #print(tags.keys())
                    if len(tags_unread) > 0:
                        print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins[var]])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[var][ion][val])]) for val in readpaths[var][ion].keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = ol.pdir + 'radprof/' + filename%(fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in readpaths[var][ion].keys():
                                try:
                                    np.array(fi[galset + '/' + readpaths[var][ion][val]])
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
                            print('For file %s, missed the following tags:\n\t%s'%(filename, tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpaths[var][ion][val])]) for val in readpaths[var][ion].keys()}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))

    npix_rminmax = {}
    # size of one pixel (pkpc)
    pixsize = (100. / 32000.) * 1e3 * cosmopars['a']
    for ionind in range(len(ions)):   
        ion = ions[ionind]
        var = techvars_touse[0]
        tags = techvars[var]['setnames']
        # example tag 'logMstar_Msun_1000_geq%.1f_le%.1f'
        tags = sorted(tags, key=lambda x: float(x.split('_')[3][3:]))
        #print(tags)
        npix_rminmax[ion] = {}
        
        for ti in range(len(tags)): # mass bins
            #print(tag)
            tag = tags[ti]
            npix_rminmax[ion][tag] = {}         
            try:
                rvals = np.array(bins[var][ion][tag])
            except KeyError: # dataset not read in
                print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                continue
            rtag = '_'.join(tag.split('_')[-2:])
            basesize = R200c_pkpc(10**(matchvals_Mstar_Mhalo[rtag][0] + 0.25), cosmopars)
            maxr_pkpc = maxr_r200c * basesize
            fcovs = np.array(yvals[var][ion][tag][yvals_toplot[ion][0]])
            
            npix_r = fcovs * np.pi * (rvals[1:]**2 - rvals[:-1]**2 ) / pixsize**2 # fraction * annulus surface area / pixel size    
            npix_inr = np.cumsum(npix_r)
            npix_inrmin = pu.linterpsolve(rvals[1:], npix_inr, minr_pkpc)
            npix_inrmax = pu.linterpsolve(rvals[1:], npix_inr, maxr_pkpc)
            
            npix_rminmax[ion][tag].update({'maxr_pkpc': maxr_pkpc,\
                        'npix_perhalo_inrmin': npix_inrmin,\
                        'npix_perhalo_inrmax': npix_inrmax})       
    
    ##### read in total cddfs
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'}
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    masknames1 = ['nomask']
    masknames = masknames1 #{0: {ion: masknames1 for ion in ions}}

    hists = {}
    cosmopars = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        dXtot[var] = {}
        dztot[var] = {}
        dXtotdlogN[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                if _bins[0] == -np.inf:
                    _bins[0] = -100.
                if _bins[-1] == np.inf:
                    _bins[-1] = 100.
                bins[var][ion] = _bins
                
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
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                #print('ion %s: sum = %f'%(ion, np.sum(hists[var][ion]['nomask'])))
                
                examplemaskdir = fi['masks'].keys()[0]
                examplemask = fi['masks/%s'%(examplemaskdir)].keys()[0]
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                dXtot[var][ion] = mc.getdX(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dztot[var][ion] = mc.getdz(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])

        assert checksubdct_equal(cosmopars[var])
    
    npix_overlim_all = {}
    var_tot = 0
    totpix = 16 * 32000**2 # checked hist sum
    for ion in ions:
        lim = yvals_toplot[ion][0]
        edges = bins[var_tot][ion]
        counts = hists[var_tot][ion]['nomask']
        cumulcounts = np.cumsum(counts[::-1])[::-1]
        npix_overlim_all[ion] = pu.linterpsolve(edges[:-1], cumulcounts, lim)
        print('{ion} fcov total: {fcov}'.format(ion=ion, fcov=npix_overlim_all[ion] / totpix))
    print('\n')
    
    overview = {}
    for ion in ions:
        tags = list(npix_rminmax[ion].keys())
        minmaxv = np.array([[float(x.split('_')[3][3:]), float(x.split('_')[4][2:])] for x in tags])
        _ind = np.argsort(minmaxv[:, 0])
        tags = np.array(tags)[_ind]
        minmaxv = minmaxv[_ind, :]
        overview[ion] = {'limval': yvals_toplot[ion][0]}
        
        #print('For ion {ion}, absorbers > {val}'.format(ion=ion, val=yvals_toplot[ion]))
        for ti in range(len(tags)):
            Mmin, Mmax = minmaxv[ti]
            tag = tags[ti]
            numgals = np.sum(np.logical_and(mstar >= Mmin, mstar < Mmax))
            
            inmin_over_inmax = npix_rminmax[ion][tag]['npix_perhalo_inrmin'] /\
                               npix_rminmax[ion][tag]['npix_perhalo_inrmax']
            inmax_over_total = npix_rminmax[ion][tag]['npix_perhalo_inrmax'] \
                               * numgals / npix_overlim_all[ion]
            rmax = npix_rminmax[ion][tag]['maxr_pkpc']
            
            overview[ion][Mmin] = {'Mmin': Mmin, 'Mmax': Mmax, 'rmax': rmax,\
                                   'inmin_over_inmax': inmin_over_inmax,\
                                   'inmax_over_total': inmax_over_total}
            #print('In Mstar range {Mmin:4.1f}-{Mmax:4.1f}:'.format(Mmin=Mmin, Mmax=Mmax))
            #print('    < {rmin:6.3f} pkpc / < {rmax:6.3f} pkpc: {corefrac}'.format(rmin=minr_pkpc, rmax=rmax, corefrac=inmin_over_inmax))
            #print('    < {rmax:6.3f} pkpc / total: {halofrac}'.format(rmax=rmax, halofrac=inmax_over_total))
            #print('{num} galaxies'.format(num=numgals))
        #print('\n')
    #print([set(overview[ion].keys()) == set(overview[ions[0]].keys()) for ion in ions])
    
    ### print overview table
    print('For an inner radius of {minr} pkpc'.format(minr=minr_pkpc))
    print('Core contribution / total halo')
    topstr = '$\\mathrm{{M}}_{{\\star}}$' +\
             ' & $\\mathrm{{r}}_{{\\perp, \\max}}$ &' +\
             ' & '.join(['{ion}'.format(ion=ild.getnicename(ion)) for ion in ions]) + ' \\\\'
    topstr2 = '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\odot}}$' +\
              ' & $\\mathrm{{pkpc}}$ &' +\
              ' & '.join(['$>{yval:.1f}$'.format(yval=yvals_toplot[ion][0]) for ion in ions]) + ' \\\\'
    fillstr = '{Mmin:.1f}--{Mmax:.1f} & {rmax:.0f} & ' + ' & '.join(['{{{ist}:.4f}}'.format(ist=ion) for ion in ions]) + ' \\\\'
    totstr = 'total & \t & ' +  ' & '.join(['{{{ist}:.4f}}'.format(ist=ion) for ion in ions]) + ' \\\\'
    Mvals = list(overview[ions[0]].keys())
    Mvals.remove('limval')
    stion = ions[0]
    print(topstr)
    print(topstr2)
    for Mmin in sorted(Mvals):
        print(fillstr.format(Mmin=overview[stion][Mmin]['Mmin'],\
                             Mmax=overview[stion][Mmin]['Mmax'],\
                             rmax=overview[stion][Mmin]['rmax'],\
                             **{ion: overview[ion][Mmin]['inmin_over_inmax'] for ion in ions}))
    print('\n')
    print('Halo contribution / total')
    print(topstr)
    print(topstr2)
    for Mmin in sorted(Mvals):
        print(fillstr.format(Mmin=overview[stion][Mmin]['Mmin'],\
                             Mmax=overview[stion][Mmin]['Mmax'],\
                             rmax=overview[stion][Mmin]['rmax'],\
                             **{ion: overview[ion][Mmin]['inmax_over_total'] for ion in ions}))
    print(totstr.format(**{ion: np.sum([overview[ion][Mmin]['inmax_over_total'] for Mmin in Mvals]) for ion in ions}))
    

def get_dNdz_halos(limset='break'):   
    '''
    get number of absorbers >N for the different halo sets from masked and
    FoF-only CDDFs. Threasholds N are set by 
    limset: 'break' (CDDF break) or 'obs' (estimated observation limits)
    '''

    if limset == 'break':
        lims = {'o6':   [14.3],\
                        'ne8':  [13.7],\
                        'o7':   [16.0],\
                        'ne9':  [15.3],\
                        'o8':   [16.0],\
                        'fe17': [15.0]}
    elif limset == 'obs':
        lims = {'o6':   [13.5],\
                        'ne8':  [13.5],\
                        'o7':   [15.5],\
                        'ne9':  [15.5],\
                        'o8':   [15.7],\
                        'fe17': [14.9]}

    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    medges = np.arange(11., 14.1, 0.5) #np.arange(11., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}   
    filenames_all = {key: [ol.pdir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    
    masses_proj = ['none'] + list(medges)
    filedct = {ion: {masses_proj[i]: filenames_all[ion][i] for i in range(len(filenames_all[ion]))} for ion in ions} 
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for ion in ions:
        dct_fofcddf[ion] = {}
        # FoF CDDFs: only without masks
        pmass = 'none'
        dct_fofcddf[ion] = {}
        try:
            with h5py.File(filedct[ion][pmass]) as fi:
                try:
                    bins = np.array(fi['bins/axis_0'])
                except KeyError as err:
                    print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                    raise err
                    
                dct_fofcddf[ion]['bins'] = bins
                
                inname = np.array(fi['input_filenames'])[0]
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                numpix_1sl.remove(None)
                numpix_1sl = int(list(numpix_1sl)[0][:-3])
                print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                for mmass in masses_proj[1:]:
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    mask_examples = {key: item for (key, item) in grp.attrs.items()}
                    del mask_examples['covfrac']
                    example_key = mask_examples.keys()[0] # 'mask_<slice center>'
                    example_mask = mask_examples[example_key] # '<dir path><mask file name>'
                    path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                    #print(path)
                    cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                    dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    #dXtotdlogN = dXtot * np.diff(bins)
        
                    dct_fofcddf[ion][mmass] = {'cddf': hist / dXtot, 'covfrac': covfrac}
                
                # use cosmopars from the last read mask
                mmass = 'none'
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                dztot = mc.getdz(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                #dXtotdlogN = dXtot * np.diff(bins)
                dct_fofcddf[ion][mmass] = {'cddf': hist / dztot, 'covfrac': covfrac}
            
        except IOError as err:
            print('Failed to read in %s; stated error:'%filedct[pmass])
            print(err)
         
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': ol.pdir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  ol.pdir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  ol.pdir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   ol.pdir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   ol.pdir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   ol.pdir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'hneutralssh': ol.pdir + 'cddf_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'}
    
    dct_maskcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_maskcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_maskcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0]
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
        
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            for mmass in masses_proj[1:]:
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                mask_examples = {key: item for (key, item) in grp.attrs.items()}
                del mask_examples['covfrac']
                example_key = mask_examples.keys()[0] # 'mask_<slice center>'
                example_mask = mask_examples[example_key] # '<dir path><mask file name>'
                path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                dXtot = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                #dXtotdlogN = dXtot * np.diff(bins)
            
                dct_maskcddf[ion][mmass] = {'cddf': hist / dXtot, 'covfrac': covfrac}
            # use cosmopars from the last read mask
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dztot = mc.getdz(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            #dXtotdlogN = dXtot * np.diff(bins)
            dct_maskcddf[ion][mmass] = {'cddf': hist / dztot, 'covfrac': covfrac}
    
    results = {}
    for ion in ions:
       results[ion] = {}
       bins_mask = dct_maskcddf[ion]['bins']
       bins_fof  = dct_fofcddf[ion]['bins']
       
       for mmass in masses_proj:
           cumul_fof =  np.cumsum(dct_fofcddf[ion][mmass]['cddf'][::-1])[::-1]
           cumul_mask = np.cumsum(dct_maskcddf[ion][mmass]['cddf'][::-1])[::-1]
           
           val_fof  = pu.linterpsolve(bins_fof[:-1], cumul_fof, lims[ion][0])
           val_mask = pu.linterpsolve(bins_mask[:-1], cumul_mask, lims[ion][0])
           fcov_mask = dct_maskcddf[ion][mmass]['covfrac']
           
           results[ion][mmass] = {'fof': val_fof,\
                                 'mask': val_mask,\
                                 'fcov': fcov_mask}
    strings_mmass = {'none': 'total'}
    strings_mmass.update({mmass: '{Mmin:.1f}--{Mmax:.1f}'.format(Mmin=mmass, Mmax=mmass + 0.5) \
                                 if mmass < 13.9 else \
                                 '$>{Mmin:.1f}$'.format(Mmin=mmass) \
                                 for mmass in masses_proj[1:]})
    
    ionstr = {ion: '\\ion{{{}}}{{{}}}'.format(ild.getnicename(ion).split(' ')[0], \
                                              string.lower(ild.getnicename(ion).split(' ')[1])) \
              for ion in ions}
    
    ### print overview table
    print('FoF-only CDDFs (total = all haloes)')
    print('dn(>N, halo) / dz')
    topstr = '$\\mathrm{{M}}_{{200\\mathrm{{c}}}} $ & ' + \
             ' & '.join(['{ion}'.format(ion=ionstr[ion]) for ion in ions]) + ' \\\\'
    topstr2 = '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\odot}}$ & ' +\
              ' & '.join(['$>{yval:.1f}$'.format(yval=lims[ion][0]) for ion in ions]) + ' \\\\'
    fillstr = '{massst} & ' + ' & '.join(['{{{ist}:.3f}}'.format(ist=ion) for ion in ions]) + ' \\\\'
    #resstr = 'total &  ' +  ' & '.join(['{{{ist}:.4f}}'.format(ist=ion) for ion in ions]) + ' \\\\'

    print(topstr)
    print(topstr2)
    print('\\hline')
    for Mmin in masses_proj:
        print(fillstr.format(massst=strings_mmass[Mmin],\
                             **{ion: results[ion][Mmin]['fof'] for ion in ions}))
    print('\n')
    print('R200c mask CDDFs (total = all gas)')
    print(topstr)
    print(topstr2)
    print('\\hline')
    for Mmin in masses_proj:
        print(fillstr.format(massst=strings_mmass[Mmin],\
                             **{ion: results[ion][Mmin]['mask'] for ion in ions}))    
    print('\n')
    print('R200c mask CDDFs as a fraction of all gas')
    print(topstr)
    print(topstr2)
    print('\\hline')
    for Mmin in masses_proj[1:]:
        print(fillstr.format(massst=strings_mmass[Mmin],\
                             **{ion: results[ion][Mmin]['mask'] / results[ion]['none']['mask'] for ion in ions}))
    print(fillstr.format(massst='halo total',\
                             **{ion: np.sum([results[ion][Mmin]['mask'] for Mmin in masses_proj[1:]]) / results[ion]['none']['mask'] for ion in ions}))
    
    print('\n')
    print('FoF all halos / total CDDF')
    print(topstr)
    print(topstr2)
    print('\\hline')
    print(fillstr.format(massst='all FoF / all gas',\
                             **{ion: results[ion]['none']['fof'] / results[ion]['none']['mask'] for ion in ions}))