#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:03:38 2021

@author: Nastasha
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

import projection_classes as pc
import eagle_constants_and_units as c
import cosmo_utils as cu

mdir = '/data1/line_em_abs/v3_master_tests/exclude_direct_fb/'

tnow_key = 'PartType0/Temperature'                                                                                                                                   
tmax_key = 'PartType0/MaximumTemperature'                                                                                                                            
amax_key = 'PartType0/AExpMaximumTemperature' 
dens_key = 'PartType0/Density'
    
labels = {tnow_key: tnow_key.split('/')[-1] + ' [K]',
          tmax_key: tmax_key.split('/')[-1] + ' [K]',
          amax_key: 'a(Tmax)',
          dens_key: 'nH [cm**-3]'
          }

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
    
    
def plot_all():
    plot_tmaxhist()
    plot_tmax_amax()
    plot_tcorr()
    t_myr = np.array([0., 1., 2., 3., 5., 10., 20., 30., 50., 1e2, 1e3, 1e4])
    for t in t_myr:
        plot_phasediagram_acut(t)
        plot_phasediagram_selcut(t)
        
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in [2, 3]:
        msg = '2 or 3 arguments should be provided: simnum, snapnum [, var]'
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
    