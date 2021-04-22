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

import projection_classes as pc
import eagle_constants_and_units as c
import cosmo_utils as cu
import make_maps_v3_master as m3

mdir = '/net/luttero/data1/line_em_abs/v3_master_tests/exclude_direct_fb/'

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
    _weight = weights[index // slow]
    htype = htypes[index % slow]
    if checkindex:
        return _weight, htype
    
    m3.ol.ndir = mdir + 'data/'
    
    simnum = 'L0012N0188'
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
    
    t_myr = np.array([0., 1., 2., 3., 5., 10., 20., 30., 50., 1e2, 2e2, 
                      3e2, 1e3, 1e4])
    t_myr = t_myr.sort()
    _simfile = pc.Simfile(simnum, snapnum, var)
    anow = _simfile.a
    deltat = t_myr * c.sec_per_megayear
    time_now = cu.t_expfactor(anow, cosmopars=_simfile)
    acut = cu.expfactor_t(time_now - deltat, cosmopars=_simfile)  
    abins = np.array([-np.inf] + list(acut[::-1]) + [np.inf])
    tmaxbins = [-np.inf, 7.499, 8.499, np.inf]
    
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

    args = (ptype, simnum, snapnum, var, axesdct)
    _kwargs = dict(simulation='eagle',
                   excludeSFR='T4', abunds='Sm', parttype='0',
                   axbins=axbins,
                   sylviasshtables=PS20tab, bensgadget2tables=False,
                   ps20tables=True, ps20depletion=False,
                   allinR200c=True, mdef='200c',
                   L_x=None, L_y=None, L_z=None, centre=None, Ls_in_Mpc=True,
                   misc=None,
                   name_append=None, logax=logax, loghist=False,
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
    