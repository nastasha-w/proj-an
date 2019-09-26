#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:49:22 2019

@author: wijers
"""

import numpy as np
import make_maps_opts_locs as ol
import h5py
import scipy

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
import loadnpz_and_plot as lnp
import make_maps_v3_master as m3 # for ion balances
import simfileclone as sfc # for cooling contours
import cosmo_utils as cu
import ion_line_data as ild


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

rho_to_nh = 0.752 / (c.atomw_H * c.u)

#retrieved with mc.getcosmopars
cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 2.220446049250313e-16}
cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}
logrhob_av_ea_28 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
logrhob_av_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )
logrhoc_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars_ea_27['z'], cosmopars=cosmopars_ea_27)**2)

def T200c_hot(M200c, cosmopars):
    # checked against notes from Joop's lecture: agrees pretty well at z=0, not at z=9 (proabably a halo mass definition issue)
    M200c *= c.solar_mass # to cgs
    rhoc = (3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars['z'], cosmopars=cosmopars)**2) # Hubble(z) will assume an EAGLE cosmology
    mu = 0.59 # about right for ionised (hot) gas, primordial
    R200c = (M200c / (200. * rhoc))**(1./3.)
    return (mu * c.protonmass) / (3. * c.boltzmann) * c.gravity * M200c/R200c

def R200c_pkpc(M200c, cosmopars):
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

def setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False, left=True):
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=right, top=top, left=left, axis='both', which='both', color=color,\
                   labelleft=labelleft, labeltop=labeltop, labelbottom=labelbottom, labelright=labelright)

def checksubdct_equal(dct):
    keys = dct.keys()
    if not np.all(np.array([set(dct[key].keys()) == set(dct[keys[0]].keys()) for key in keys])):
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
                legline.set_dashes(dashes[1])
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

    plt.savefig(mdir + 'ionbals_snap27_HM01_ionizedmu.pdf', format='pdf', bbox_inches='tight')



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
        
    massranges = [(float(i) for i in name.split('-')[-2:]) if name is not 'nomask' else None for name in masknames]
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
    lax.legend(lcs + sumhandles, [legendnames_techvars[var] for var in techvars] + sumlabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=2 * numcols, loc='lower center', bbox_to_anchor=(0.5, 0.))
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
    lax.legend(lcs + sumhandles, ['FoF+200c, with mask', 'all gas, with mask'] + sumlabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=ncol_legend, loc='lower center', bbox_to_anchor=(0.5, 0.))
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
            choice = raw_input('Please enter name or index: ')
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


def plotfracs_by_halo_subcat(ions=['Mass', 'hneutralssh', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'], first='halo'):
    '''
    first: group mass bins by halo mass or subhalo catgory first
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
    
    outname = '/net/luttero/data2/imgs/histograms_basic/' + 'barchart_halosubcat_L0100N1504_27_T4EOS_%s-first.pdf'%(first)
    
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
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    

###############################################################################
#                  nice plots for the paper: simplified                       #
###############################################################################