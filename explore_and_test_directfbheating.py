#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:03:38 2021

@author: Nastasha
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import h5py

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.patches as mpatch
import matplotlib.ticker as ticker

import tol_colors as tc

import projection_classes as pc
import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu
import make_maps_v3_master as m3
import test_comp_tables_SP20 as tct

mdir = '/net/luttero/data1/line_em_abs/v3_master_tests/exclude_direct_fb/'
mapdir = mdir + 'maps/'
m3.ol.ndir = mapdir


tnow_key = 'PartType0/Temperature'                                                                                                                                   
tmax_key = 'PartType0/MaximumTemperature'                                                                                                                            
amax_key = 'PartType0/AExpMaximumTemperature' 
dens_key = 'PartType0/Density'
    
labels = {tnow_key: tnow_key.split('/')[-1] + ' [K]',
          tmax_key: tmax_key.split('/')[-1] + ' [K]',
          amax_key: 'a(Tmax)',
          dens_key: 'nH [cm**-3]'
          }

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
plot_lines_SB = ['c5r', 'c6', 'n6-actualr', 'n7', 'o7r', 'o7iy', 'o7f', 'o8',
                 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r']
plot_lines_PS20 = ['Fe17      15.2620A', 'Fe17      16.7760A',
                   'Fe17      17.0510A', 'Fe17      17.0960A', 
                   'Fe18      16.0720A']
lines_paper = plot_lines_SB + plot_lines_PS20
siontab = '_iontab-PS20-UVB-dust1-CR1-G1-shield1_depletion-F'

def set_simfile(simnum, snapnum, var='REFERENCE'):
    global simfile
    simfile = pc.Simfile(simnum, snapnum, var)

def read_simdata():
    global dens, tnow, tmax, amax
    dens = simfile.readarray(dens_key, rawunits=False)
    tnow = simfile.readarray(tnow_key, rawunits=True)
    tmax = simfile.readarray(tmax_key, rawunits=True)
    amax = simfile.readarray(amax_key, rawunits=True)
    dens *= 0.752 / (c.atomw_H * c.u)

def plot_tmaxhist():
    name = mdir + 'tmax_hist_{simnum}_{snap}_{var}.pdf'
    name = name.format(simnum=simfile.simnum, snap=simfile.snapnum, 
                       var=simfile.var)
    vals = np.log10(tmax)
    minv = np.min(vals)
    maxv = np.max(vals)
    delta = 0.01
    bins = np.arange(minv, maxv + 1.01 * delta, delta)
    plt.hist(vals, bins=bins, log=True, alpha=0.3)
    plt.grid(b=True)
    plt.axvline(7.5, color='red', zorder=0, linewidth=1.)
    plt.axvline(8.5, color='red', zorder=0, linewidth=1.)
    plt.xlabel('log10 ' + labels[tmax_key])
    plt.ylabel('SPH particle count')
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_tmax_amax():
    name = mdir + 'tmax_amax_hist_{simnum}_{snap}_{var}.pdf'
    name = name.format(simnum=simfile.simnum, snap=simfile.snapnum, 
                       var=simfile.var)
    xvals = np.log10(tmax)
    minx = np.min(xvals)
    maxx = np.max(xvals)
    delta = 0.05
    xbins = np.arange(minx, maxx + 1.01 * delta, delta)
    
    yvals = amax
    miny = np.min(yvals)
    maxy = np.max(yvals)
    delta = 0.01
    ybins = np.arange(miny, maxy + 1.01 * delta, delta)
    
    vals, tmax_bin, amax_bin = np.histogram2d(xvals, yvals, 
                                              bins=[xbins, ybins])
    plt.pcolormesh(xbins, ybins, np.log10(vals.T))
    plt.grid(b=True)
    plt.axvline(7.5, color='red', zorder=0, linewidth=1.)
    plt.axvline(8.5, color='red', zorder=0, linewidth=1.)
    plt.xlabel('log10 ' + labels[tmax_key])
    plt.ylabel(labels[amax_key])
    plt.colorbar()
    
    plt.title('log10 SPH particle count')
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_tcorr():
    name = mdir + 'tmax_tnow_hist_{simnum}_{snap}_{var}.pdf'
    name = name.format(simnum=simfile.simnum, snap=simfile.snapnum, 
                       var=simfile.var)
    
    xvals = np.log10(tnow)
    minx = np.min(xvals)
    maxx = np.max(xvals)
    
    yvals = np.log10(tmax)
    miny = np.min(yvals)
    maxy = np.max(yvals)
    
    eq0 = max(minx, miny)
    eq1 = min(maxx, maxy)
    subsample = 100
    
    plt.scatter(xvals[::subsample], yvals[::subsample], c=amax[::subsample], 
                alpha=0.2, s=3, rasterized=True)
    plt.axhline(7.5, color='red', zorder=0)                                                                                                                                        
    plt.axhline(8.5, color='red', zorder=0)   
    plt.plot((eq0, eq1), (eq0, eq1), color='black', zorder=0)
    plt.grid(b=True)
    plt.colorbar()
    
    plt.xlabel('log10 ' + labels[tnow_key])
    plt.ylabel('log10 ' + labels[tmax_key])
    sf = ', subsample factor {}'.format(subsample)
    plt.title('color: ' + labels[amax_key] + sf)
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_phasediagram_acut(deltat_myr):
    name = mdir + 'phasediagram_amaxseldiff_{simnum}_{snap}_{var}'+\
                  '_deltat-{deltat}-Myr.pdf'
    name = name.format(simnum=simfile.simnum, snap=simfile.snapnum, 
                       var=simfile.var, deltat=deltat_myr)
    
    xvals = np.log10(dens)
    minx = np.min(xvals)
    maxx = np.max(xvals)
    delta = 0.1
    xbins = np.arange(minx, maxx + 1.01 * delta, delta)
    
    yvals = np.log10(tnow)
    miny = np.min(yvals)
    maxy = np.max(yvals)
    delta = 0.05
    ybins = np.arange(miny, maxy + 1.01 * delta, delta)    
    
    anow = simfile.a
    deltat = deltat_myr * c.sec_per_megayear
    time_now = cu.t_expfactor(anow, cosmopars=simfile)
    acut = cu.expfactor_t(time_now - deltat, cosmopars=simfile)  
    sel = amax > acut 
    print('selected {} / {} particles'.format(np.sum(sel), len(sel)))
    
    hist_all, xb, yb = np.histogram2d(xvals, yvals, bins=[xbins, ybins])
    hist_exc, xb, yb = np.histogram2d(xvals[sel], yvals[sel], 
                                      bins=[xbins, ybins])
    loghist = np.log10(hist_all)
    levels = np.arange(0, np.ceil(np.max(loghist)), 1)
    xc = 0.5 * (xbins[:-1] + xbins[1:])
    yc = 0.5 * (ybins[:-1] + ybins[1:])
    plt.contour(xc, yc, loghist.T, levels=levels, colors='black')
    plt.pcolormesh(xbins, ybins, np.log10(hist_exc.T))
    plt.grid(b=True)
    plt.colorbar()
    
    plt.xlabel('log10 ' + labels[dens_key])
    plt.ylabel('log10 ' + labels[tnow_key])
    title = 'contours: all SPH particles (factor of 10 steps),'+\
            ' colors: excluded gas'
    plt.title(title)
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    
def plot_phasediagram_selcut(deltat_myr):
    name = mdir + 'phasediagram_fbseldiff_{simnum}_{snap}_{var}' +\
                  '_deltat-{deltat}-Myr.pdf'
    name = name.format(simnum=simfile.simnum, snap=simfile.snapnum, 
                       var=simfile.var, deltat=deltat_myr)
    
    xvals = np.log10(dens)
    minx = np.min(xvals)
    maxx = np.max(xvals)
    delta = 0.1
    xbins = np.arange(minx, maxx + 1.01 * delta, delta)
    
    yvals = np.log10(tnow)
    miny = np.min(yvals)
    maxy = np.max(yvals)
    delta = 0.05
    ybins = np.arange(miny, maxy + 1.01 * delta, delta)    
    
    anow = simfile.a
    deltat = deltat_myr * c.sec_per_megayear
    time_now = cu.t_expfactor(anow, cosmopars=simfile)
    acut = cu.expfactor_t(time_now - deltat, cosmopars=simfile)
    tcut = 7.499
    
    sel = amax > acut 
    sel &= tmax >= 10**tcut
    print('selected {} / {} particles'.format(np.sum(sel), len(sel)))
    
    hist_all, xb, yb = np.histogram2d(xvals, yvals, bins=[xbins, ybins])
    hist_exc, xb, yb = np.histogram2d(xvals[sel], yvals[sel], 
                                      bins=[xbins, ybins])
    loghist = np.log10(hist_all)
    levels = np.arange(0, np.ceil(np.max(loghist)), 1)
    xc = 0.5 * (xbins[:-1] + xbins[1:])
    yc = 0.5 * (ybins[:-1] + ybins[1:])
    plt.contour(xc, yc, loghist.T, levels=levels, colors='black')
    plt.pcolormesh(xbins, ybins, np.log10(hist_exc.T))
    plt.grid(b=True)
    plt.colorbar()
    
    plt.xlabel('log10 ' + labels[dens_key])
    plt.ylabel('log10 ' + labels[tnow_key])
    title = 'contours: all SPH particles (factor of 10 steps),'+\
            ' colors: excluded gas'
    plt.title(title)
    plt.savefig(name, bbox_inches='tight')
    plt.close()
    
def plot_all():
    '''
    run all plots. doesn't work too well for the 100 Mpc volume.

    Returns
    -------
    None.

    '''
    plot_tmaxhist()
    plot_tmax_amax()
    plot_tcorr()
    t_myr = np.array([0., 1., 2., 3., 5., 10., 20., 30., 50., 1e2, 1e3, 1e4])
    for t in t_myr:
        plot_phasediagram_acut(t)
        plot_phasediagram_selcut(t)
    
def run_phasediagrams_LMweighted(index, checkindex=False):
    _lines = lines_paper
    weights = ['Mass'] + _lines
    htypes = ['all', 'halo']
    slow = len(weights)
    _weight = weights[index % slow]
    htype = htypes[index // slow]
    if checkindex:
        print(_weight, htype)
        return _weight, htype
    
    m3.ol.ndir = mdir + 'data/'
    
    simnum = 'L0100N1504'
    snapnum = 27
    var = 'REFERENCE'
    PS20tab = _weight in all_lines_PS20
    
    if _weight == 'Mass':
        ptype = 'basic'
        kwargs = {'quantity': _weight}
    else:
        ptype = 'Luminosity'
        kwargs = {'ion': _weight}
        
    if htype == 'halo':
        axesdct = [{'ptype': 'halo', 'quantity': 'Mass'}]
        axbins = [np.array([-np.inf, 0., c.solar_mass] +\
                       list(10**(np.arange(9., 15.5, 0.5)) * c.solar_mass) +\
                       [np.inf])
                  ]
        logax = [False]
    else:
        axesdct = []
        axbins = []
        logax = []
    
    # # SFR bins
    # minval = 2**-149 * c.solar_mass / c.sec_per_year 
    # # calculate minimum SFR possible in Eagle, 
    # # use as minimum bin for ISM value
    # axbins.append(np.array([-np.inf, minval, np.inf])) 
    # axesdct.append({'ptype': 'basic', 'quantity': 'StarFormationRate'})
    # logax = logax + [False]
    
    #t_myr = np.array([0., 1., 2., 3., 5., 10., 20., 30., 50., 1e2, 2e2, 
    #                  3e2, 1e3, 1e4])
    # second go at it
    t_myr = np.array([0., 1., 2., 3., 5., 10., 30., 1e2, 1e4])
    t_myr.sort()
    _simfile = pc.Simfile(simnum, snapnum, var)
    anow = _simfile.a
    deltat = t_myr * c.sec_per_megayear
    time_now = cu.t_expfactor(anow, cosmopars=_simfile)
    acut = cu.expfactor_t(time_now - deltat, cosmopars=_simfile)  
    abins = np.array([-np.inf] + list(acut[::-1]) + [np.inf])
    #tmaxbins = [-np.inf, 7.499, 8.499, np.inf]
    # second go
    tmaxbins = [-np.inf, 7.499, 7.6, 7.7, 7.8, 7.9, 8., 8.499, 
                8.6, 8.7, 8.8, 8.9, 9.0, np.inf]
    baseaxes = [{'ptype': 'basic', 'quantity': 'Temperature'},
                {'ptype': 'basic', 'quantity': 'Density'},
                {'ptype': 'basic', 'quantity': 'AExpMaximumTemperature'},
                {'ptype': 'basic', 'quantity': 'MaximumTemperature'},
                ]
    basebins = [0.1, 0.1, abins, tmaxbins]
    baselog = [True, True, False, True]
    
    axesdct = axesdct + baseaxes
    axbins = axbins + basebins
    logax = logax + baselog

    name_append = '_set2'
    
    args = (ptype, simnum, snapnum, var, axesdct)
    _kwargs = dict(simulation='eagle',
                   excludeSFR='T4', abunds='Sm', parttype='0',
                   axbins=axbins,
                   sylviasshtables=False, bensgadget2tables=False,
                   ps20tables=PS20tab, ps20depletion=False,
                   allinR200c=True, mdef='200c',
                   L_x=None, L_y=None, L_z=None, centre=None, Ls_in_Mpc=True,
                   misc=None,
                   name_append=name_append, logax=logax, loghist=False,
                   )
    _kwargs.update(kwargs)
    kwargs = _kwargs
    filen, grpn = m3.makehistograms_perparticle(*args, nameonly=True, **kwargs)
    done = False
    if os.path.isfile(filen):
        with h5py.File(filen) as fi:
            if grpn in fi:
                done = True
    if not done:
        m3.makehistograms_perparticle(*args, nameonly=False, **kwargs)

def readin_phasediagrams_LMweighted(filename, setnum=1):
    '''
    read in the hdf5 histogram data for phase information plots with halo mass
    and direct feedback info

    Parameters
    ----------
    filename : str
        name of the hdf5 histogram file.

    Returns
    -------
    dct_all : dct
        dictionary containing the histogram ('hist'), axis bin edges ('bins'),
        and which axes in the histogram they correspond to ('axes'). Contains
        the histogram for all gas in the simulation.
    dct_hm : dct
        same as dct_all, but includes an axis for the halo mass, and only 
        particles in haloes.
    cosmopars : dct
        dictionary containing cosmological parameters

    '''
    if '/' not in filename:
        filename = mdir + 'data/' + filename
    
    edgekeys = {'M200c': 'M200c_halo_allinR200c',
                'Tmax': 'MaximumTemperature_T4EOS',
                'Tnow': 'Temperature_T4EOS',
                'dens': 'Density_T4EOS',
                'amax': 'AExpMaximumTemperature_T4EOS'}
    
    with h5py.File(filename, 'r') as f:
        if setnum == 1:
            add_all = ''
            add_hm = ''
        else:
            add_all = '_set{}'.format(setnum)
            add_hm = ''
        key_hm = 'M200c_halo_allinR200c_Temperature_T4EOS_Density_T4EOS' + \
                 '_AExpMaximumTemperature_T4EOS_MaximumTemperature_T4EOS' +\
                 add_hm
        key_all = 'Temperature_T4EOS_Density_T4EOS_AExpMaximumTemperature' + \
                  '_T4EOS_MaximumTemperature_T4EOS' + add_all
        dct_all = {}
        dct_hm = {}
        for gkey, dct in zip([key_hm, key_all], [dct_hm, dct_all]):
            grp = f[gkey]
            dct['bins'] = {}
            dct['axes'] = {}
            for ekey in edgekeys:
                if edgekeys[ekey] in grp:
                    egrp = grp[edgekeys[ekey]]
                    bins = egrp['bins'][:]
                    axis = egrp.attrs['histogram axis']
                    dct['bins'][ekey] = bins
                    dct['axes'][ekey] = axis
            dct['hist'] = grp['histogram'][:]
            if bool(grp['histogram'].attrs['log']):
                dct['hist'] = 10**dct['hist']
        cosmopars = {key: val for key, val \
                          in f['Header/cosmopars'].attrs.items()}
    return dct_all, dct_hm, cosmopars

def deltat_from_acut(acut, cosmopars):
    time_now = cu.t_expfactor(cosmopars['a'], cosmopars=cosmopars)
    time_cut = cu.t_expfactor(acut, cosmopars=cosmopars) 
    deltat = time_now - time_cut
    deltat *= 1. / c.sec_per_megayear
    return deltat
    
def plot_phasediagram_selcut_fromsaved(*args, weight_fn='Mass',
                                       weightname='Mass [g]'):
    
    dct_all = args[0]
    dct_hm = args[1]
    cosmopars = args[2]
    
    snecut = 7.499
    agncut = 8.499
    massbins_check = np.arange(10.5, 14.6, 0.5)
    
    outdir = mdir + 'phasediagrams/'
    outbase = 'phasedigram_L0100N1504_27_{weight}_selcuts_{massrange}.pdf'
    fontsize = 12
    
    for mi, mbin in enumerate(massbins_check):
        mrange = 'M200c-{:.1f}-{:.1f}'.format(mbin, mbin + 0.5) \
                 if mi < len(massbins_check) - 2 else \
                 'M200c-{:.1f}-inf'.format(mbin) \
                 if mi == len(massbins_check) - 2 else \
                 'all-gas'
        outname = outdir + outbase.format(weight=weight_fn, massrange=mrange)
        
        if mi == len(massbins_check) - 1:
            dct = dct_all
            hist = np.copy(dct['hist'])
            hax = 100
        else:
            dct = dct_hm
            hax = dct['axes']['M200c']
            hbins = dct['bins']['M200c']
            hi = np.where(np.isclose(hbins, 10**mbin * c.solar_mass))[0][0]
            hsel = [slice(None, None, None)] * len(dct['hist'].shape)
            hsel[hax] = slice(hi, hi + 1, None)
            hist = np.sum(dct['hist'][tuple(hsel)], axis=hax)
            
        tnow_ax = dct['axes']['Tnow']
        dens_ax = dct['axes']['dens']
        tnow_bins = dct['bins']['Tnow']
        dens_bins = dct['bins']['dens'] + np.log10(0.752 / (c.atomw_H * c.u))
        
        tnow_c = 0.5 * (tnow_bins[:-1] + tnow_bins[1:])
        dens_c = 0.5 * (dens_bins[:-1] + dens_bins[1:])
        
        tmax_ax = dct['axes']['Tmax']
        tmax_bins = dct['bins']['Tmax']
        
        amax_ax = dct['axes']['amax']
        deltat_bins = deltat_from_acut(dct['bins']['amax'], cosmopars) 
        deltat_bins[0] = 0.
        
        if tnow_ax > hax:
            tnow_ax = tnow_ax - 1
        if dens_ax > hax:
            dens_ax = dens_ax - 1
        if tmax_ax > hax:
            tmax_ax = tmax_ax - 1
        if amax_ax > hax:
            amax_ax = amax_ax - 1
        
        total = np.sum(hist, axis=(tmax_ax, amax_ax))
        if tnow_ax > dens_ax:
            total = total.T
        
        vmax = np.log10(np.max(total))
        vmin = np.log10(np.min(total[total > 0.]))
        levels = np.arange(vmin, vmax, 1.)
        if len(levels) > 5:
            levels = np.linspace(max(vmin + 1, vmax - 6), vmax - 1, 5)
        cmap = cm.get_cmap('viridis')

        numtbins = len(deltat_bins) - 1
        
        panelwidth = 2.5
        ncols = 4
        nrows = (numtbins - 1) // ncols + 1
        width = ncols * panelwidth
        height_ratios = [panelwidth * 1. / 0.75] + [panelwidth] * (2 * nrows)
        height = sum(height_ratios)
        
        fig = plt.figure(figsize=(width, height))
        grid = gsp.GridSpec(nrows=1 + 2 * nrows, ncols=ncols, 
                            hspace=0.0, wspace=0.0, 
                            height_ratios=height_ratios, top=0.95)
        axes1 = [fig.add_subplot(grid[1 + i // ncols, i % ncols]) \
                 for i in range(numtbins)]
        axes2 = [fig.add_subplot(grid[1 + nrows + i // ncols, i % ncols]) \
                 for i in range(numtbins)]
            
        _ax = fig.add_subplot(grid[0, :])
        _ax.axis('off')
        _l, _b, _w, _h = (_ax.get_position()).bounds
        hmargin = _h * 0.25
        __h = _h - hmargin
        pwidth = __h * height / width
        cwidth = 0.3 * pwidth
        wmargin = 0.1 * pwidth
        twidth = _w - (pwidth + cwidth + 5. * wmargin)
        
        totax = fig.add_axes([_l, _b + hmargin, pwidth, __h])
        cax = fig.add_axes([_l + pwidth + wmargin, _b + hmargin, cwidth, __h])
        tax = fig.add_axes([_l + pwidth + cwidth + 5. * wmargin, 
                            _b + hmargin, twidth, __h])
        tax.axis('off')
        title = 'weight: {}\n'.format(weightname) +\
                'distribution of the weighted quantity in\nphase space, '+\
                'using different minimum \n past maximum temperature cuts:\n'+\
                'for SNe and AGN feedback (top)\nand AGN only (bottom)\n'+\
                'and different maximum times since\nthat maximum was attained'
        tax.text(0., 1., title, fontsize=fontsize, transform=tax.transAxes,
                 horizontalalignment='left', verticalalignment='top')
        
        cutlabel = '$\\Delta \\, \\mathrm{{t}} \\,/\\, \\mathrm{{Myr}} <'+\
                   '{deltat:.0f}$'
        img = totax.pcolormesh(dens_bins, tnow_bins, np.log10(total), 
                               cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(img, cax=cax, extend='neither', 
                            orientation='vertical', aspect=15.)   
        cset = totax.contour(dens_c, tnow_c, np.log10(total), levels=levels,
                             colors='black')
        cbar.add_lines(cset)
        cax.set_ylabel('$\\log_{10}$ weight', fontsize=fontsize)
        cax.tick_params(labelsize=fontsize - 1.)
        
        for Tcut, axset in zip([snecut, agncut], [axes1, axes2]):
            for ti, deltat in enumerate(deltat_bins[:-1]):
                ax = axset[ti]
                ax.contour(dens_c, tnow_c, np.log10(total), levels=levels,
                           colors='black')
                label = cutlabel.format(deltat=deltat, Tcut=Tcut)
                ax.text(0.98, 0.98, label, fontsize=fontsize, 
                        transform=ax.transAxes, horizontalalignment='right',
                        verticalalignment='top')
                sel = [slice(None, None, None)] * len(hist.shape)
                tmax_ci = np.where(np.isclose(Tcut, tmax_bins))[0][0]
                sel[tmax_ax] = slice(tmax_ci, None, None)
                sel[amax_ax] = slice(ti, None, None)
                subtot = np.sum(hist[tuple(sel)], axis=(amax_ax, tmax_ax))
                if tnow_ax > dens_ax:
                    subtot = subtot.T
                ax.pcolormesh(dens_bins, tnow_bins, np.log10(subtot), 
                              cmap=cmap, vmin=vmin, vmax=vmax)
                
                pu.setticks(ax, fontsize=fontsize - 1, labelbottom=False, 
                            labelleft=False)
                ax.grid(b=True)
        labelax = fig.add_subplot(grid[1:, :], frameon=False)
        labelax.tick_params(labelcolor='none', top=False, bottom=False, 
                            left=False, right=False)
        xlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\,'+\
                 ' [\\mathrm{cm}^{-3}]$'
        ylabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
        labelax.set_xlabel(xlabel, fontsize=fontsize)
        labelax.set_ylabel(ylabel, fontsize=fontsize)
        totax.set_xlabel(xlabel, fontsize=fontsize)
        totax.set_ylabel(ylabel, fontsize=fontsize)
        pu.setticks(totax, fontsize=fontsize - 1)
        for i in range(numtbins):
            if i >= numtbins - ncols:
                axes1[i].tick_params(labelbottom=True)
                axes2[i].tick_params(labelbottom=True)
            if i % ncols == 0:
                axes1[i].tick_params(labelleft=True)
                axes2[i].tick_params(labelleft=True)
                
        plt.savefig(outname, bbox_inches='tight')

def plot_phasediagram2_selcut_fromsaved(*args, weight_fn='Mass',
                                        weightname='Mass [g]', agnlim=False):
    dct = args[0]
    #dct_hm = args[1]
    if len(args) == 3:
        cosmopars = args[2]
    else:
        cosmopars = args[1]
    
    snecut = 7.499
    agncut = 8.499
    
    outdir = mdir + 'phasediagrams/'
    outbase = 'phasedigram_L0100N1504_27_{weight}_selcuts_Tmaxvar_{fbc}.pdf'
    fontsize = 12
    
    fbc = 'agn' if agnlim else 'sne'
    outname = outdir + outbase.format(weight=weight_fn, fbc=fbc)
        
    hist = np.copy(dct['hist'])
            
    tnow_ax = dct['axes']['Tnow']
    dens_ax = dct['axes']['dens']
    tnow_bins = dct['bins']['Tnow']
    dens_bins = dct['bins']['dens'] + np.log10(0.752 / (c.atomw_H * c.u))
    
    tnow_c = 0.5 * (tnow_bins[:-1] + tnow_bins[1:])
    dens_c = 0.5 * (dens_bins[:-1] + dens_bins[1:])
    
    tmax_ax = dct['axes']['Tmax']
    tmax_bins = dct['bins']['Tmax']
    
    amax_ax = dct['axes']['amax']
    deltat_bins = deltat_from_acut(dct['bins']['amax'], cosmopars) 
    deltat_bins[0] = deltat_from_acut(0., cosmopars)
        
    total = np.sum(hist, axis=(tmax_ax, amax_ax))
    if tnow_ax > dens_ax:
        total = total.T
        
    vmax = np.log10(np.max(total))
    vmin = np.log10(np.min(total[total > 0.]))
    levels = np.arange(vmin, vmax, 1.)
    if len(levels) > 5:
        levels = np.linspace(max(vmin + 1, vmax - 6), vmax - 1, 5)
    cmap = cm.get_cmap('viridis')

    numtbins = len(deltat_bins) - 1
    snei0 = np.where(np.isclose(snecut, tmax_bins))[0][0]
    agni0 = np.where(np.isclose(agncut, tmax_bins))[0][0]
    numbins_tmax_sne = agni0 - snei0
    numbins_tmax_agn = len(tmax_bins) - agni0 - 1
    if agnlim:
        numbins_tmax = numbins_tmax_agn
        tmax_i0 = agni0
    else:
        numbins_tmax = numbins_tmax_sne
        tmax_i0 = snei0
        
    panelwidth = 2.
    ncols = numbins_tmax
    nrows = numtbins 
    width = ncols * panelwidth
    height_ratios = [panelwidth * 1. / 0.75] + [panelwidth] * (nrows)
    height = sum(height_ratios)
    
    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=1 + nrows, ncols=ncols, 
                        hspace=0.0, wspace=0.0, 
                        height_ratios=height_ratios, top=0.95)
    axes = [[fig.add_subplot(grid[1 + i, j]) \
             for j in range(numbins_tmax)] for i in range(numtbins)]
        
    _ax = fig.add_subplot(grid[0, :])
    _ax.axis('off')
    _l, _b, _w, _h = (_ax.get_position()).bounds
    hmargin = _h * 0.25
    __h = _h - hmargin
    pwidth = __h * height / width
    cwidth = 0.3 * pwidth
    wmargin = 0.1 * pwidth
    twidth = _w - (pwidth + cwidth + 5. * wmargin)
    
    totax = fig.add_axes([_l, _b + hmargin, pwidth, __h])
    cax = fig.add_axes([_l + pwidth + wmargin, _b + hmargin, cwidth, __h])
    tax = fig.add_axes([_l + pwidth + cwidth + 5. * wmargin, 
                        _b + hmargin, twidth, __h])
    tax.axis('off')
    title = 'weight: {}\n'.format(weightname) +\
            'distribution of the weighted quantity in\nphase space, '+\
            'using different minimum \n past maximum temperature cuts\n'+\
            'and different maximum times since\nthat maximum was attained'
    tax.text(0., 1., title, fontsize=fontsize, transform=tax.transAxes,
             horizontalalignment='left', verticalalignment='top')
    
    cutlabel_deltat = '$\\Delta \\, \\mathrm{{t}} \\,/\\, \\mathrm{{Myr}} <'+\
                      '{deltat:.0f}$'
    #cutlabel_tmax = '$\\log_{{10}} \\, \\mathrm{{T}}_{{\\mathrm{{max}}}} '+\
    #                '\\,/\\, \\mathrm{{K}} = {tmin:.1f} \\emdash {tmax:.1f}$'
    cutlabel_tmax = '$\\mathrm{{T}}_{{\\mathrm{{max}}}}: '+\
                    '{tmin:.1f} \\endash {tmax:.1f}$'
    
    img = totax.pcolormesh(dens_bins, tnow_bins, np.log10(total), 
                           cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(img, cax=cax, extend='neither', 
                        orientation='vertical', aspect=15.)   
    cset = totax.contour(dens_c, tnow_c, np.log10(total), levels=levels,
                         colors='black')
    cbar.add_lines(cset)
    cax.set_ylabel('$\\log_{10}$ weight', fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1.)
    
    for tmax_i1 in range(1, numbins_tmax + 1):
        for ti, deltat in enumerate(deltat_bins[:-1]):
            ax = axes[ti][tmax_i1 - 1]
            ax.contour(dens_c, tnow_c, np.log10(total), levels=levels,
                       colors='black')
            label = cutlabel_deltat.format(deltat=deltat)
            ax.text(0.98, 0.98, label, fontsize=fontsize, 
                    transform=ax.transAxes, horizontalalignment='right',
                    verticalalignment='top')
            label = cutlabel_tmax.format(tmin=tmax_bins[tmax_i0],
                                         tmax=tmax_bins[tmax_i0 + tmax_i1])
            ax.text(0.98, 0.02, label, fontsize=fontsize, 
                    transform=ax.transAxes, horizontalalignment='right',
                    verticalalignment='bottom')
            sel = [slice(None, None, None)] * len(hist.shape)
            sel[tmax_ax] = slice(tmax_i0, tmax_i0 + tmax_i1, None)
            sel[amax_ax] = slice(ti, None, None)
            subtot = np.sum(hist[tuple(sel)], axis=(amax_ax, tmax_ax))
            if tnow_ax > dens_ax:
                subtot = subtot.T
            ax.pcolormesh(dens_bins, tnow_bins, np.log10(subtot), 
                          cmap=cmap, vmin=vmin, vmax=vmax)
            
            pu.setticks(ax, fontsize=fontsize - 1, labelbottom=False, 
                        labelleft=False)
            ax.grid(b=True)
            if not agnlim:
                ax.axvline(-2., color='red', linestyle='dotted', linewidth=1.)
                ax.axhline(tmax_bins[tmax_i0] - 0.1, color='red',
                           linestyle='dotted', linewidth=1.)
    labelax = fig.add_subplot(grid[1:, :], frameon=False)
    labelax.tick_params(labelcolor='none', top=False, bottom=False, 
                        left=False, right=False)
    xlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\,'+\
             ' [\\mathrm{cm}^{-3}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    totax.set_xlabel(xlabel, fontsize=fontsize)
    totax.set_ylabel(ylabel, fontsize=fontsize)
    pu.setticks(totax, fontsize=fontsize - 1)
    for i in range(numtbins):
        axes[i][0].tick_params(labelleft=True)      
    for j in range(numbins_tmax):
        axes[numtbins - 1][j].tick_params(labelbottom=True)
            
    plt.savefig(outname, bbox_inches='tight')
        
def plot_all_phaseinfo_cuts():
    weights = ['Mass'] + lines_paper
    filebase = 'particlehist_{weight}_L0100N1504_27_test3.7_SmAb_T4EOS.hdf5'
    for weight in weights:
        if weight == 'Mass':
            wfill = weight
            wname = 'Mass [g]'
            wt = weight
        elif weight in all_lines_PS20:
            wfill = 'Luminosity_{wt}{it}'.format(wt=weight.replace(' ', '-'),
                                                 it=siontab)
            _wt = weight
            while '  ' in _wt:
                _wt = _wt.replace('  ', ' ')
            wname = 'L {} [erg/s]'.format(_wt)
            wt = 'Luminosity_{}'.format(weight.replace(' ', '-'))
        elif weight in all_lines_SB:
            wfill = 'Luminosity_{wt}'.format(wt=weight.replace(' ', '-'))
            _wt = weight
            while '  ' in _wt:
                _wt = _wt.replace('  ', ' ')
            wname = 'L {} [erg/s]'.format(_wt)
            wt = 'Luminosity_{}'.format(weight)
        filen = mdir + 'data/' + filebase.format(weight=wfill)
        if weight == 'Mass':
            filen = filen.replace('_SmAb', '')
        
        args = readin_phasediagrams_LMweighted(filen)
        plot_phasediagram_selcut_fromsaved(*args, weight_fn=wt, 
                                           weightname=wname)
        plt.close('all')

def plot_nHdist_selcut_fromsaved(*args, weight_fn='Mass',
                                 weightname='Mass [g]'):
    
    dct_all = args[0]
    dct_hm = args[1]
    cosmopars = args[2]
    for key in ['amax', 'Tmax']:
        if not np.allclose(dct_all['bins'][key], dct_hm['bins'][key]):
            msg = 'Bins in halo and all gas histograms should match (key: {})'
            raise ValueError(msg.format(key))
    deltat_bins = deltat_from_acut(dct_all['bins']['amax'], cosmopars) 
    deltat_bins[0] = deltat_from_acut(0., cosmopars)
    
    snecut = 7.499
    agncut = 8.499
    massbins_check = np.arange(10.5, 14.6, 0.5)
    
    outdir = mdir
    outbase = 'nHdist_L0100N1504_27_{weight}_selcuts.pdf'
    fontsize = 12
    ls_agn = 'dashed'
    ls_sne = 'solid'
    ls_all = 'dotted'
    
    panelwidth = 5.
    ncols = 2
    nrows = (len(massbins_check) - 1) // ncols + 1
    width = ncols * panelwidth
    height_ratios = [0.4 * panelwidth] + [0.4 * panelwidth] * (2 * nrows)
    height = sum(height_ratios)
    
    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=1 + 2 * nrows, ncols=ncols, 
                        hspace=0.0, wspace=0.0, 
                        height_ratios=height_ratios, top=0.95)
    axes_cumul = [fig.add_subplot(grid[1 + 2 * (i // ncols), i % ncols]) \
                  for i in range(len(massbins_check))]
    axes_diff  = [fig.add_subplot(grid[2 + 2 * (i // ncols), i % ncols]) \
                  for i in range(len(massbins_check))]
        
    _ax = fig.add_subplot(grid[0, :])
    _ax.axis('off')
    _l, _b, _w, _h = (_ax.get_position()).bounds
    hmargin = _h * 0.05
    __h = _h - hmargin
    #eqwidth = __h * height / width
    cwidth = 0.49 * _w
    wmargin = 0.02 * _w
    lwidth = cwidth
    twidth = _w - (max(lwidth, cwidth) + wmargin)
    
    cax = fig.add_axes([_l, _b + hmargin, cwidth, __h * 0.3])
    lax = fig.add_axes([_l, _b + hmargin + __h * 0.7, lwidth, __h * 0.3])
    tax = fig.add_axes([_l + max(lwidth, cwidth) + wmargin , 
                        _b + hmargin, twidth, __h])
    tax.axis('off')
    title = 'weight: {}\n'.format(weightname) +\
            'distribution of the weighted quantity with\ndensity, '+\
            'using different minimum past\nmaximum temperature cuts:\n'+\
            'for SNe and AGN feedback (all fb) and AGN\nonly (AGN fb) '+\
            'and different maximum times\nsince that maximum was attained'
    tax.text(0., 0., title, fontsize=fontsize, transform=tax.transAxes,
             horizontalalignment='left', verticalalignment='bottom')
    
    handles = [mlines.Line2D((), (), color='gray', label=label, 
                             linestyle=ls)\
               for ls, label in zip([ls_sne, ls_agn, ls_all], 
                                    ['all fb', 'AGN fb', 'total'])]
    lax.legend(handles=handles, fontsize=fontsize, loc='upper center',
               ncol=3, bbox_to_anchor=(0.5, 1.))
    lax.axis('off')
    clabel = 'max $\\Delta \\, \\mathrm{{t}} \\,/\\, \\mathrm{{Myr}}$ since fb'
    tedges = np.array(deltat_bins[:-1])
    clist = tc.tol_cmap('rainbow_discrete', 
                        lut=len(tedges))(np.linspace(0.,  1., len(tedges)))
    keys = sorted(tedges)
    colors = {keys[i]: clist[len(clist) - 1 - i] for i in range(len(keys))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist)
    #cmap.set_over(clist[-1])
    bounds = np.linspace(0., 1., len(tedges) + 1)
    ticks = 0.5 * (bounds[:-1] + bounds[1:])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    ticklabels = ['{:.0f}'.format(te) for te in tedges]
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                     norm=norm,
                                     boundaries=bounds,
                                     ticks=ticks,
                                     spacing='uniform', extend='neither',
                                     orientation='horizontal')

    cbar.set_label(clabel, fontsize=fontsize)
    cax.xaxis.set_label_position('top')
    cax.tick_params(labelsize=fontsize - 1, labelbottom=False, bottom=False,
                    labeltop=True, top=True, rotation=45.)
    cax.set_xticklabels(labels=ticklabels)
    cax.set_aspect(0.1)
    
    labelax = fig.add_subplot(grid[1:, :], frameon=False)
    labelax.tick_params(labelcolor='none', top=False, bottom=False, 
                        left=False, right=False)
    xlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\,'+\
             ' [\\mathrm{cm}^{-3}]$'
    ylabel_diff = 'weight / $\\Delta \\, \\log_{10} \\mathrm{n}_{\\mathrm{H}}$'
    ylabel_cumul = 'weight ($ > \\log_{10} \\mathrm{n}_{\\mathrm{H}}$)'
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    
    max_cumul = -np.inf
    max_diff = -np.inf
    for mi, mbin in enumerate(massbins_check):
        mrange = 'M200c: {:.1f}-{:.1f}'.format(mbin, mbin + 0.5) \
                 if mi < len(massbins_check) - 2 else \
                 'M200c: > {:.1f}'.format(mbin) \
                 if mi == len(massbins_check) - 2 else \
                 'all gas'     
        
        if mi == len(massbins_check) - 1:
            dct = dct_all
            tnow_ax = dct['axes']['Tnow']
            hist = np.sum(dct['hist'], axis=tnow_ax)
            hax = 100
        else:
            dct = dct_hm
            hax = dct['axes']['M200c']
            hbins = dct['bins']['M200c']
            tnow_ax = dct['axes']['Tnow']
            hi = np.where(np.isclose(hbins, 10**mbin * c.solar_mass))[0][0]
            hsel = [slice(None, None, None)] * len(dct['hist'].shape)
            hsel[hax] = slice(hi, hi + 1, None)
            hist = np.sum(dct['hist'][tuple(hsel)], axis=(hax, tnow_ax))
            
        dens_ax = dct['axes']['dens']
        dens_bins = dct['bins']['dens'] + np.log10(0.752 / (c.atomw_H * c.u))
        dens_c = 0.5 * (dens_bins[:-1] + dens_bins[1:])
        dens_w =  np.diff(dens_bins)
        
        tmax_ax = dct['axes']['Tmax']
        tmax_bins = dct['bins']['Tmax']
        amax_ax = dct['axes']['amax']
        
        if dens_ax > hax:
            dens_ax = dens_ax - 1
        if dens_ax >= tnow_ax:
            dens_ax = dens_ax - 1
        if tmax_ax > hax:
            tmax_ax = tmax_ax - 1
        if tmax_ax >= tnow_ax:
            tmax_ax = tmax_ax - 1
        if amax_ax > hax:
            amax_ax = amax_ax - 1
        if amax_ax >= tnow_ax: 
            amax_ax = amax_ax - 1
        
        total = np.sum(hist, axis=(tmax_ax, amax_ax))
        
        axi = (mi + 1) % len(massbins_check)
        ax_diff = axes_diff[axi]
        ax_cumul = axes_cumul[axi]
        if axi % ncols == 0:
            ax_cumul.set_ylabel(ylabel_cumul, fontsize=fontsize)
            ax_diff.set_ylabel(ylabel_diff, fontsize=fontsize)
        
        #ax_diff.bar(dens_c, np.log10(total / dens_w), bottom=None, 
        #            align='center', width=dens_w, color='none', 
        #            edgecolor='black', linestyle=ls_all)
        y_diff = np.log10(total / dens_w)
        max_diff = max(max_diff, np.max(y_diff))
        ax_diff.plot(dens_c, y_diff, color='black', linestyle=ls_all)
        y_cumul = np.log10(np.append(np.cumsum(total[::-1])[::-1], [0.]))
        max_cumul = max(max_cumul, y_cumul[0])
        ax_cumul.plot(dens_bins, y_cumul, color='black', 
                      linestyle=ls_all)
        
        ax_cumul.text(0.02, 0.02, mrange, fontsize=fontsize, 
                      transform=ax_cumul.transAxes, 
                      horizontalalignment='left',
                      verticalalignment='bottom')
        
        for Tcut, ls in zip([snecut, agncut], [ls_sne, ls_agn]):
            for ti, deltat in enumerate(deltat_bins[:-1]):            
                sel = [slice(None, None, None)] * len(hist.shape)
                tmax_ci = np.where(np.isclose(Tcut, tmax_bins))[0][0]
                sel[tmax_ax] = slice(tmax_ci, None, None)
                sel[amax_ax] = slice(ti, None, None)
                subtot = np.sum(hist[tuple(sel)], axis=(amax_ax, tmax_ax))
                
                #ax_diff.bar(dens_c, np.log10(subtot / dens_w), bottom=None, 
                #            align='center', width=dens_w, color='none', 
                #            edgecolor=colors[deltat], linestyle=ls)
                ax_diff.plot(dens_c, np.log10(subtot / dens_w),
                             color=colors[deltat], linestyle=ls)
                y_cumul = np.append(np.cumsum(subtot[::-1])[::-1], [0.])
                ax_cumul.plot(dens_bins, np.log10(y_cumul), 
                              color=colors[deltat], linestyle=ls)
                
                pu.setticks(ax_cumul, fontsize=fontsize - 1, 
                            labelbottom=False, 
                            labelleft=axi % ncols == 0)
                pu.setticks(ax_diff, fontsize=fontsize - 1, 
                            labelbottom=axi >= len(massbins_check) - ncols, 
                            labelleft=axi % ncols == 0)
                ax_cumul.grid(b=True)
                ax_diff.grid(b=True)
    
    # sync axis ranges
    yrs = [ax.get_ylim() for ax in axes_diff]
    ymin = min([yr[0]for yr in yrs])
    #ymax = max([yr[1]for yr in yrs])
    ymax = max_diff + 0.2
    ymin = max(ymin, ymax - 10.)
    [ax.set_ylim(ymin, ymax) for ax in axes_diff]
    yrs = [ax.get_ylim() for ax in axes_cumul]
    ymin = min([yr[0]for yr in yrs])
    #ymax = max([yr[1]for yr in yrs])
    ymax = max_cumul + 0.2
    ymin = max(ymin, ymax - 10.)
    [ax.set_ylim(ymin, ymax) for ax in axes_cumul]
    xrs = [ax.get_xlim() for ax in np.append(axes_diff, axes_cumul)]
    xmin = min([xr[0]for xr in xrs])
    xmax = max([xr[1]for xr in xrs])
    [ax.set_xlim(xmin, xmax) for ax in np.append(axes_diff, axes_cumul)]
    
    outname = outdir + outbase.format(weight=weight_fn)        
    plt.savefig(outname, bbox_inches='tight')
    
def plot_all_nH_cuts():
    weights = ['Mass'] + lines_paper
    filebase = 'particlehist_{weight}_L0100N1504_27_test3.7_SmAb_T4EOS.hdf5'
    for weight in weights:
        if weight == 'Mass':
            wfill = weight
            wname = 'Mass [g]'
            wt = weight
        elif weight in all_lines_PS20:
            wfill = 'Luminosity_{wt}{it}'.format(wt=weight.replace(' ', '-'),
                                                 it=siontab)
            _wt = weight
            while '  ' in _wt:
                _wt = _wt.replace('  ', ' ')
            wname = 'L {} [erg/s]'.format(_wt)
            wt = 'Luminosity_{}'.format(weight.replace(' ', '-'))
        elif weight in all_lines_SB:
            wfill = 'Luminosity_{wt}'.format(wt=weight.replace(' ', '-'))
            _wt = weight
            while '  ' in _wt:
                _wt = _wt.replace('  ', ' ')
            wname = 'L {} [erg/s]'.format(_wt)
            wt = 'Luminosity_{}'.format(weight)
        filen = mdir + 'data/' + filebase.format(weight=wfill)
        if weight == 'Mass':
            filen = filen.replace('_SmAb', '')
        
        args = readin_phasediagrams_LMweighted(filen)
        plot_nHdist_selcut_fromsaved(*args, weight_fn=wt, 
                                           weightname=wname)
        plt.close('all')
        
def run_maps():
    
    simnum = 'L0025N0376'
    snapnum = 27
    centre = [12.5, 12.5, 3.125]
    L_x, L_y, L_z = (25., 25., 6.25)
    npix_x, npix_y = (400, ) * 2 
    ptypeW = 'basic'
    
    # small differnces, most likely, so turn off ompproj for exact comparisons
    args = (simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, ptypeW)
    kwargs_default = {'ionW': None, 'abundsW': None, 'quantityW': 'Mass',
                      'ionQ': None, 'abundsQ': None, 'ptypeQ': 'basic',
                      'quantityQ': 'Temperature',
                      'excludeSFRW': 'T4', 'excludeSFRQ': 'T4', 
                      'parttype': '0', 'var': 'REFERENCE', 'axis': 'z',
                      'log': True, 'ompproj': False, 'hdf5': True, 
                      'saveres': True, 'periodic': False,
                      }
    kwargs_iter = [{'deltatMyr_directfb': 10., 
                    'inclhotgas_maxlognH_snfb': -2.,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': False, 'deltalogT_directfb': 0.2,
                    'quantityQ': 'Temperature'},
                   {'deltatMyr_directfb': 10., 
                    'inclhotgas_maxlognH_snfb': -2.,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': False, 'deltalogT_directfb': 0.2,
                    'quantityQ': 'Density'},
                   {'deltatMyr_directfb': 10., 
                    'inclhotgas_maxlognH_snfb': -2.,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': True, 'deltalogT_directfb': 0.2,
                    'quantityQ': 'Temperature'},
                   {'deltatMyr_directfb': 10., 
                    'inclhotgas_maxlognH_snfb': -2.,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': True, 'deltalogT_directfb': 0.2,
                    'quantityQ': 'Density'},
                   {'deltatMyr_directfb': 10., 
                    'inclhotgas_maxlognH_snfb': -2.,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': True, 'deltalogT_directfb': 0.5,
                    'quantityQ': 'Temperature'},
                   {'deltatMyr_directfb': 100., 
                    'inclhotgas_maxlognH_snfb': -2.,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': True, 'deltalogT_directfb': 0.2,
                    'quantityQ': 'Temperature'},
                   {'deltatMyr_directfb': 10., 
                    'inclhotgas_maxlognH_snfb': -np.inf,
                    'logTK_agnfb': 8.499, 'logTK_snfb': 7.499,
                    'excludedirectfb': True, 'deltalogT_directfb': 0.2,
                    'quantityQ': 'Density'},
                   ]
    for _kwargs in kwargs_iter:
        kwargs = kwargs_default.copy()
        kwargs.update(_kwargs)
        m3.make_map(*args, **kwargs)

def compare_maps():
    path = mdir + 'maps/'
    outdir = mdir + 'mapcomp/'
    # fn_old_dens = path + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7' + \
    #               '_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_' + \
    #               'y3.125-pm6.25_z-projection_master.hdf5'
    # fn_old_mass = path + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_'+\
    #               'zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS'+\
    #               '_master.hdf5'
    # fn_old_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_'+\
    #               'test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_'+\
    #               'y3.125-pm6.25_z-projection_master.hdf5'
     
    # fn_new_dens = path + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7' + \
    #               '_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_' + \
    #               'y3.125-pm6.25_z-projection.hdf5'
    # fn_new_mass = path + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_'+\
    #               'zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS'+\
    #               '.hdf5'
    # fn_new_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_'+\
    #               'test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_'+\
    #               'y3.125-pm6.25_z-projection.hdf5'
    
    # fn_def_dens = path + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_'+\
    #               'C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125'+\
    #               '-pm6.25_z-projection_exclfb_TSN-7.499_TAGN-8.499_'+\
    #               'Trng-0.2_10.0-Myr_inclSN-nH-lt--2.0.hdf5'
    # fn_def_mass = path + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice'+\
    #               '_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_'+\
    #               'T4EOS_exclfb_TSN-7.499_TAGN-8.499_Trng-0.2_10.0-Myr_'+\
    #               'inclSN-nH-lt--2.0.hdf5'
    # fn_def_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_'+\
    #               'test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25'+\
    #               '_y3.125-pm6.25_z-projection_exclfb_TSN-7.499_TAGN-8.499'+\
    #               '_Trng-0.2_10.0-Myr_inclSN-nH-lt--2.0.hdf5'
                  
    # fn_nonHcut_dens = path + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_'+\
    #               'test3.7_C2Sm'+\
    #               '_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_'+\
    #               'z-projection_exclfb_TSN-7.499_TAGN-8.499_Trng-0.2_'+\
    #               '10.0-Myr.hdf5'
    # fn_nonHcut_mass = path + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_'+\
    #               '6.25slice_'+\
    #               'zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS'+\
    #               '_exclfb_TSN-7.499_TAGN-8.499_Trng-0.2_10.0-Myr.hdf5'            
    
    # fn_lTrng_mass = path + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice'+\
    #               '_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS'+\
    #               '_exclfb_TSN-7.499_TAGN-8.499_Trng-0.5_10.0-Myr'+\
    #               '_inclSN-nH-lt--2.0.hdf5'
    # fn_lTrng_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_'+\
    #               'test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_'+\
    #               'y3.125-pm6.25_z-projection_exclfb_TSN-7.499_TAGN-8.499'+\
    #               '_Trng-0.5_10.0-Myr_inclSN-nH-lt--2.0.hdf5'
                  
    # fn_ldeltat_mass = path + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_'+\
    #               '6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_'+\
    #               'z-projection_T4EOS_exclfb_TSN-7.499_TAGN-8.499_'+\
    #               'Trng-0.2_100.0-Myr_inclSN-nH-lt--2.0.hdf5'
    # fn_ldeltat_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_'+\
    #               'test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_'+\
    #               'y3.125-pm6.25_z-projection_exclfb_TSN-7.499_TAGN-8.499'+\
    #               '_Trng-0.2_100.0-Myr_inclSN-nH-lt--2.0.hdf5'
                  
    fn_old_dens = path + 'Density_T4EOS_Mass_T4EOS_L0025N0376_27_test3.7' + \
                  '_C2Sm_400pix_6.25slice_zcen3.125' + \
                  '_z-projection_master.hdf5'
    fn_old_mass = path + 'Mass_L0025N0376_27_test3.7_C2Sm_400pix_6.25slice_'+\
                  'zcen3.125_z-projection_T4EOS'+\
                  '_master.hdf5'
    fn_old_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0025N0376_27_'+\
                  'test3.7_C2Sm_400pix_6.25slice_zcen3.125'+\
                  '_z-projection_master.hdf5'
     
    fn_new_dens = path + 'Density_T4EOS_Mass_T4EOS_L0025N0376_27_test3.7' + \
                  '_C2Sm_400pix_6.25slice_zcen3.125' + \
                  '_z-projection.hdf5'
    fn_new_mass = path + 'Mass_L0025N0376_27_test3.7_C2Sm_400pix_6.25slice_'+\
                  'zcen3.125_z-projection_T4EOS'+\
                  '.hdf5'
    fn_new_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0025N0376_27_'+\
                  'test3.7_C2Sm_400pix_6.25slice_zcen3.125'+\
                  '_z-projection.hdf5'
    
    fn_def_dens = path + 'Density_T4EOS_Mass_T4EOS_L0025N0376_27_test3.7_'+\
                  'C2Sm_400pix_6.25slice_zcen3.125'+\
                  '_z-projection_exclfb_TSN-7.499_TAGN-8.499_'+\
                  'Trng-0.2_10.0-Myr_inclSN-nH-lt--2.0.hdf5'
    fn_def_mass = path + 'Mass_L0025N0376_27_test3.7_C2Sm_400pix_6.25slice'+\
                  '_zcen3.125_z-projection_'+\
                  'T4EOS_exclfb_TSN-7.499_TAGN-8.499_Trng-0.2_10.0-Myr_'+\
                  'inclSN-nH-lt--2.0.hdf5'
    fn_def_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0025N0376_27_'+\
                  'test3.7_C2Sm_400pix_6.25slice_zcen3.125'+\
                  '_z-projection_exclfb_TSN-7.499_TAGN-8.499'+\
                  '_Trng-0.2_10.0-Myr_inclSN-nH-lt--2.0.hdf5'
                  
    fn_nonHcut_dens = path + 'Density_T4EOS_Mass_T4EOS_L0025N0376_27_'+\
                  'test3.7_C2Sm'+\
                  '_400pix_6.25slice_zcen3.125_'+\
                  'z-projection_exclfb_TSN-7.499_TAGN-8.499_Trng-0.2_'+\
                  '10.0-Myr.hdf5'
    fn_nonHcut_mass = path + 'Mass_L0025N0376_27_test3.7_C2Sm_400pix_'+\
                  '6.25slice_'+\
                  'zcen3.125_z-projection_T4EOS'+\
                  '_exclfb_TSN-7.499_TAGN-8.499_Trng-0.2_10.0-Myr.hdf5'            
    
    fn_lTrng_mass = path + 'Mass_L0025N0376_27_test3.7_C2Sm_400pix_6.25slice'+\
                  '_zcen3.125_z-projection_T4EOS'+\
                  '_exclfb_TSN-7.499_TAGN-8.499_Trng-0.5_10.0-Myr'+\
                  '_inclSN-nH-lt--2.0.hdf5'
    fn_lTrng_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0025N0376_27_'+\
                  'test3.7_C2Sm_400pix_6.25slice_zcen3.125'+\
                  '_z-projection_exclfb_TSN-7.499_TAGN-8.499'+\
                  '_Trng-0.5_10.0-Myr_inclSN-nH-lt--2.0.hdf5'
                  
    fn_ldeltat_mass = path + 'Mass_L0025N0376_27_test3.7_C2Sm_400pix_'+\
                  '6.25slice_zcen3.125_'+\
                  'z-projection_T4EOS_exclfb_TSN-7.499_TAGN-8.499_'+\
                  'Trng-0.2_100.0-Myr_inclSN-nH-lt--2.0.hdf5'
    fn_ldeltat_temp = path + 'Temperature_T4EOS_Mass_T4EOS_L0025N0376_27_'+\
                  'test3.7_C2Sm_400pix_6.25slice_zcen3.125'+\
                  '_z-projection_exclfb_TSN-7.499_TAGN-8.499'+\
                  '_Trng-0.2_100.0-Myr_inclSN-nH-lt--2.0.hdf5'
    
        
    tct.compare_maps(fn_new_mass, fn_old_mass, 
                     imgname=outdir + 'mapcomp_new-old_Mass.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_new_dens, fn_old_dens, 
                     imgname=outdir + 'mapcomp_new-old_Density.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_new_temp, fn_old_temp, 
                     imgname=outdir + 'mapcomp_new-old_Temperature.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_def_mass, fn_new_mass, 
                     imgname=outdir + 'mapcomp_excl-default_Mass.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_def_dens, fn_new_dens, 
                     imgname=outdir + 'mapcomp_excl-default_Density.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_def_temp, fn_new_temp, 
                     imgname=outdir + 'mapcomp_excl-default_Temperature.pdf', 
                     mapmin=None, diffmax=None)
    
    tct.compare_maps(fn_nonHcut_mass, fn_def_mass, 
                     imgname=outdir + 'mapcomp_excl-default-nonHcut_Mass.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_nonHcut_dens, fn_def_dens, 
                     imgname=outdir + \
                             'mapcomp_excl-default-nonHcut_Density.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_ldeltat_mass, fn_def_mass, 
                     imgname=outdir + 'mapcomp_excl-default-ldeltat_Mass.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_ldeltat_temp, fn_def_temp, 
                     imgname=outdir + \
                             'mapcomp_excl-default-ldeltat_Temperature.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_lTrng_mass, fn_def_mass, 
                     imgname=outdir + 'mapcomp_excl-default-lTrng_Mass.pdf', 
                     mapmin=None, diffmax=None)
    tct.compare_maps(fn_lTrng_temp, fn_def_temp, 
                     imgname=outdir + \
                             'mapcomp_excl-default-lTrng_Temperature.pdf', 
                     mapmin=None, diffmax=None)
    
if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == 'runpdhists':
        print('running phase diagram hists')
        checkindex = '--checkindex' in args
        index = int(args[1])
        run_phasediagrams_LMweighted(index, checkindex=checkindex)
        
    else:
        if len(args) not in [2, 3]:
            msg = '2 or 3 arguments should be provided:'+\
                  ' simnum, snapnum [, var]'
            raise ValueError(msg)
        simnum = args[0]
        snapnum = int(args[1])
        if len(args) > 2:
            var = args[2]
        else:
            var = 'REFERENCE'
        set_simfile(simnum, snapnum, var)
        read_simdata()
        plot_all()
    