#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
plot or do quick checks on results from fire_maps
'''

import numpy as np
import h5py
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.patches as mpatch
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

import tol_colors as tc

import make_maps_opts_locs as ol
import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu

def hasnan_map(filen):
    with h5py.File(filen, 'r') as f:
        if 'map' not in f:
            print('skipping {}'.format(filen))
            return False # don't flag anything in a file loop
        map_ = f['map'][:]
        if np.any(np.isnan(map_)):
            print('NaN values in map {}'.format(filen))
            return True
        else:
            return False

def checkdir_nanmap(dirn):
    filens = os.listdir(dirn)
    filens = [filen for filen in filens if filen.endswith('.hdf5')]
    anynan = False
    for filen in filens:
        anynan |= hasnan_map(dirn + filen)
    return anynan

def get_rval_massmap(filen, units='pkpc'):
    '''
    get radius and map quantity matched arrays

    Input:
    ------
    filen: str
        file name including full path
    units: {'pkpc', 'Rvir', 'cm'}
        units for the radius (impact parameter)

    Returns:
    --------
    radii: float array (1D)
        impact parameters
    map values: float array (1D) 
        map values, matching impact parameters 
    '''
    with h5py.File(filen, 'r') as f:
        _map = f['map'][:]
        shape = _map.shape
        xinds, yinds = np.indices(shape).astype(np.float32)
        # centered on halo
        xcen = 0.5 * float(shape[0])
        ycen = 0.5 * float(shape[1])
        dpix2 = (xinds + 0.5 - xcen)**2 + (yinds + 0.5 - ycen)**2
        dpix = np.sqrt(dpix2)
        
        pixssize_pkpc = f['Header/inputpars'].attrs['pixsize_pkpc']
        if units == 'pkpc':
            dpix *= pixssize_pkpc
        elif units == 'Rvir':
            # using Imran's shrinking spheres method
            rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
            dpix *= pixssize_pkpc * c.cm_per_mpc * 1e-3 / rvir_cm
        elif units == 'cm':
            dpix *= pixssize_pkpc * c.cm_per_mpc * 1e-3
    return dpix.flatten(), _map.flatten()

def get_profile_massmap(filen, rbins, rbin_units='pkpc',
                        profiles=[]):
    '''
    get values with impact parameter from maps. If multiple files
    are given, averages, percentiles, etc. are taken over the full
    sample, weighted by map pixel 

    Parameters:
    -----------
    filen: str or iterable of strings
        name(s) of the file(s) containing the map (with full path)
    rbins: array-like of floats
        impact parameter bin edges
    rbin_units: {'pkpc', 'Rvir', 'cm'}
        units used for the rbins
    profiles: iterable of strings
        which profiles to extract. Returned in order of input
        options are:
            'av-lin' (linear average), 
            'av-log' (log-space average),
            'perc-<float>' (percentile, values 0 -- 1)
            'min' (minimum)
            'max' (maximum)
            'fcov-<float>' (fraction of values >= given value)
    Returns:
    --------
    profiles: each an array of floats
        the profile values in each radial bin
    '''
    
    # single filename (string) or iterable of file names
    if isinstance(filen, type('')):
        filens = [filen]
    else:
        filens = filen
    first = True
    for _filen in filens:
        with h5py.File(_filen, 'r') as f:
            _islog = bool(f['map'].attrs['log'])
        rvals, mvs = get_rval_massmap(_filen, units=rbin_units)
        
        # searchsorted index 0 means value < rbins[0], 
        # index len(rbins) means values > rbins[-1] 
        rinds = np.searchsorted(rbins, rvals) - 1 
        _mvs_by_bin = [mvs[rinds == i] for i in range(len(rbins))]
        _mvs_by_bin = _mvs_by_bin[:-1]
        if first:
            mvs_by_bin = _mvs_by_bin
            islog = _islog
            first = False
        else:
            mvs_by_bin = [np.append(mvs, _mvs) \
                          for mvs, _mvs in zip(mvs_by_bin, _mvs_by_bin)]
            # not handling this for now since all maps should be log
            if _islog != islog:
                msg = 'Different files have log vs. non-log map values'
                raise RuntimeError(msg)

    out = []
    for prof in profiles:
        if prof == 'av-lin':
            if islog:
                _pr = np.log10(np.array([np.average(10**_mvs) \
                                         for _mvs in mvs_by_bin]))
            else:
                _pr = np.array([np.average(_mvs) \
                                for _mvs in mvs_by_bin])
        elif prof == 'av-log':
            if islog:
                _pr = np.array([np.average(_mvs) \
                                for _mvs in mvs_by_bin])
            else:
                _pr = 10**np.array([np.average(np.log10(_mvs)) \
                                    for _mvs in mvs_by_bin])
        elif prof == 'max':
            _pr = np.array([np.max(_mvs) for _mvs in mvs_by_bin])
        elif prof == 'max':
            _pr = np.array([np.min(_mvs) for _mvs in mvs_by_bin])
        elif prof.startswith('perc-'):
            pv = float(prof.split('-')[-1])
            _pr = np.array([np.quantile(_mvs, pv) if len(_mvs) > 0 \
                            else np.NaN for _mvs in mvs_by_bin])
        elif prof.startswith('fcov-'):
            pv = float(prof.split('-')[-1])
            _pr = np.array([np.sum(_mvs >= pv) / float(len(_mvs)) \
                            for _mvs in mvs_by_bin])
        out.append(_pr)
    return out
         


def plot_halomasscheck(halofile, checkfile, imgname=None):
    '''
    compare the halo masses from the AHF/halo_0000_smooth.dat
    file and direct calculation of all mass within Rvir
    of the halo center Xc, Yc, Zc 

    Parameters:
    -----------
    halofile: str
        the halo_0000_smooth.dat file with full path
    checkfile: str
        file containing the calculated halo masses
    imgname: str
        name of the file to store the output image to
    
    Output:
    -------
    None
    '''

    checkdat = pd.read_csv(checkfile, sep='\t')
    ahfdat = pd.read_csv(halofile, sep='\t')
    # from log file of /projects/b1026/snapshots/metal_diffusion/m12i_res7100/
    # checks
    hconst =  0.702 

    fig = plt.figure(figsize=(11., 4.))
    grid = gsp.GridSpec(nrows=1, ncols=3, hspace=0.0, wspace=0.3, 
                        width_ratios=[1., 1., 1.])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fontsize = 12
    colors = tc.tol_cset('bright')
    size = 10.

    ax = axes[0]
    masslabel = 'Mvir [Msun]'
    xlabel = 'AHF ' + masslabel
    ylabel = masslabel + ' from AHF center and Rvir'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(which='both', direction='in', labelsize=fontsize-1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    xv = np.array(checkdat['Mvir_AHF_Msun'])
    yv = np.array(checkdat['Mvir_sum_Msun'])
    minv = min(np.min(xv), np.min(yv))
    maxv = max(np.max(xv), np.max(yv))
    ax.plot([minv, maxv], [minv, maxv], color='gray', linestyle='dotted')
    ax.scatter(xv, yv, c=colors[0], s=size)

    ax = axes[1]
    xlabel = 'log (1 + redshift)'
    ylabel = masslabel
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(which='both', direction='in', labelsize=fontsize-1)
    ax.set_yscale('log')
    ahf_z = np.array(ahfdat['redshift'])
    _sort = np.argsort(ahf_z)
    ahf_z = ahf_z[_sort]
    ahf_mvir = np.array(ahfdat['Mvir'])[_sort] / hconst
    ahf_xv = np.log10(ahf_z + 1.)
    flat = np.diff(ahf_mvir) == 0.
    sectionbreaks = list(np.where(np.diff(flat) != 0)[0] + 1) 
    if flat[0]:
        sectionbreaks = [0] + sectionbreaks
    if flat[-1]:
        sectionbreaks = sectionbreaks + [len(flat)]
    sectionbreaks = np.array(sectionbreaks)
    sections = sectionbreaks.reshape((len(sectionbreaks) // 2, 2))
    flatx = [ahf_xv[sect[0]: sect[1] + 1] for sect in sections]
    flaty = [ahf_mvir[sect[0]: sect[1] + 1] for sect in sections]
    ax.plot(ahf_xv, ahf_mvir, color=colors[0])
    for _x, _y in zip(flatx, flaty):
        ax.plot(_x, _y, color='black')
    xv = np.log10(1. + checkdat['redshift'])
    ax.scatter(xv, checkdat['Mvir_AHF_Msun'], 
               color=colors[0], label='AHF mass', s=size)
    ax.scatter(xv, checkdat['Mvir_sum_Msun'], 
               color=colors[1], label='sum < AHF Rvir', s=size)
    ax.legend(fontsize=fontsize)
    ax.set_title('black: AHF halo mass is exactly flat', fontsize=fontsize)

    ax = axes[2]
    xlabel = 'log (1 + redshift)'
    ylabel = 'log abs ([M(< Rvir)] - [AHF]) / [AHF]'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(which='both', direction='in', labelsize=fontsize-1)
    vahf = np.array(checkdat['Mvir_AHF_Msun'])
    vsum = np.array(checkdat['Mvir_sum_Msun'])
    yv = np.log10(np.abs((vsum - vahf) / vahf))
    xv = np.log10(1. + np.array(checkdat['redshift']))
    plt.scatter(xv, yv, c=colors[0], s=size)

    if imgname is not None:
        plt.savefig(imgname, bbox_inches='tight')

def runhalomasschecks(opt=1):
    checkdir = '/projects/b1026/nastasha/tests/start_fire/AHF_unit_tests/'
    if opt == 1:
        checkfile = checkdir +  'metal_diffusion__m12i_res7100.txt'
        halofile = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/halo/ahf/halo_00000_smooth.dat'
        imgname = checkdir + 'metal_diffusion__m11i_res7100_AHF-vs-sum.pdf'
        plot_halomasscheck(halofile, checkfile, imgname=imgname)
    else:
        raise ValueError('opt={} is not a valid option'.format(opt))








def quicklook_massmap(filen, savename=None, mincol=None):
    '''
    quick plot of the mass map in the file
    '''

    with h5py.File(filen, 'r') as f:
        map = f['map'][:]
        vmin = f['map'].attrs['minfinite']
        vmax = f['map'].attrs['max']

        box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
        cosmopars = {key: val for key, val in \
                     f['Header/inputpars/cosmopars'].attrs.items()}
        #print(cosmopars)
        if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
            rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
            rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
        elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
            rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
            rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
        xax = f['Header/inputpars'].attrs['Axis1']
        yax = f['Header/inputpars'].attrs['Axis2']
        box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
        extent = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                  -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])

    if mincol is None:
        cmap = 'viridis'
    else:
        cmap = pu.paste_cmaps(['gist_yarg', 'viridis'], [vmin, mincol, vmax])
    extend = 'neither' if np.min(map) == vmin else 'min'

    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(nrows=1, ncols=2, hspace=0.1, wspace=0.0, 
                        width_ratios=[10., 1.])
    ax = fig.add_subplot(grid[0, 0]) 
    cax = fig.add_subplot(grid[0, 1])
    fontsize = 12
    
    xlabel = ['X', 'Y', 'Z'][xax] + ' [pkpc]'
    ylabel = ['X', 'Y', 'Z'][yax] + ' [pkpc]'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title('Mass map centered on halo')

    img = ax.imshow(map.T, origin='lower', interpolation='nearest', vmin=vmin,
                    vmax=vmax, cmap=cmap, extent=extent)
    ax.tick_params(axis='both', labelsize=fontsize-1)
    plt.colorbar(img, cax=cax, extend=extend, orientation='vertical') 

    patches = [mpatch.Circle((0., 0.), rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5)
    ax.add_collection(collection)
    ax.text(1.05 * 2**-0.5 * rvir_pkpc, 1.05 * 2**-0.5 * rvir_pkpc, 
            '$R_{\\mathrm{vir}}$',
            color='red', fontsize=fontsize)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def checkcenter_massmap(filen_template, savename=None, mincol=None,
                        center_simunits=None, Rvir_simunits=None):
    '''
    quick plot of the mass map in the file
    '''

    filens = {ax: filen_template.format(ax=ax) for ax in ['x', 'y', 'z']}
    
    fig = plt.figure(figsize=(8., 8.))
    grid = gsp.GridSpec(nrows=2, ncols=2, hspace=0.2, wspace=0.2, 
                        width_ratios=[1., 1.])
    axes = {}
    axes['z'] = fig.add_subplot(grid[0, 0]) 
    axes['y'] = fig.add_subplot(grid[0, 1])
    axes['x'] = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[1, 1])
    fontsize = 12

    axlabels = ['{} [sim. units: ckpc/h]'. format(_ax) for _ax in 'XYZ']
    
    massmaps = {}
    extents = {}
    xlabels = {}
    ylabels = {}
    xinds = {}
    yinds = {}
    vmin = np.inf
    vmax = -np.inf
    for ax in axes:
        filen = filens[ax]
        with h5py.File(filen, 'r') as f:
            _map = f['map'][:]
            vmin = min(vmin, f['map'].attrs['minfinite'])
            vmax = max(vmax, f['map'].attrs['max'])
            
            # error in file creation -- no actual conversion to cm
            region_simunits = f['Header/inputpars'].attrs['mapped_region_cm']
            #coords_to_CGS = f['Header/inputpars'].attrs['coords_toCGS']
            # region_simunits = region_cm / coords_to_CGS

            #box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
            cosmopars = {key: val for key, val in \
                        f['Header/inputpars/cosmopars'].attrs.items()}
            _ax1 = f['Header/inputpars'].attrs['Axis1']
            _ax2 = f['Header/inputpars'].attrs['Axis2']
            _ax3 = f['Header/inputpars'].attrs['Axis3']
            if _ax3 == 2:
                xax = _ax1
                yax = _ax2
            elif _ax3 == 0:
                xax = _ax2
                yax = _ax1
                _map = _map.T
            elif _ax3 == 1:
                xax = _ax2
                yax = _ax1
                _map = _map.T
           
            extent = (region_simunits[xax][0], region_simunits[xax][1],
                      region_simunits[yax][0], region_simunits[yax][1])
            massmaps[ax] = _map
            extents[ax] = extent
            xlabels[ax] = axlabels[xax]
            ylabels[ax] = axlabels[yax]
            xinds[ax] = xax
            yinds[ax] = yax
    print('redshift: ', cosmopars['z'])

    if mincol is None:
        cmap = 'viridis'
    else:
        cmap = pu.paste_cmaps(['gist_yarg', 'viridis'], [vmin, mincol, vmax])
    extend = 'neither' if np.min(map) == vmin else 'min'
    
    for axn in axes:
        ax = axes[axn]
        ax.set_xlabel(xlabels[axn], fontsize=fontsize)
        ax.set_ylabel(ylabels[axn], fontsize=fontsize)

        img = ax.imshow(massmaps[axn].T, origin='lower', 
                        interpolation='nearest', vmin=vmin,
                        vmax=vmax, cmap=cmap, extent=extents[axn])
        ax.tick_params(axis='both', labelsize=fontsize-1)
        
        if center_simunits is not None:
            _cen = [center_simunits[xinds[axn]], center_simunits[yinds[axn]]]
            ax.scatter([_cen[0]], [_cen[1]], marker='.', color='red',
                        s=10)
            if Rvir_simunits is not None:
                patches = [mpatch.Circle(_cen, Rvir_simunits)]
                collection = mcol.PatchCollection(patches)
                collection.set(edgecolor=['red'], facecolor='none', 
                               linewidth=1.5)
                ax.add_collection(collection)
                ax.text(1.05 * 2**-0.5 * Rvir_simunits, 
                        1.05 * 2**-0.5 * Rvir_simunits, 
                        '$R_{\\mathrm{vir}}$',
                        color='red', fontsize=fontsize)
    
    cbar = plt.colorbar(img, cax=cax, extend=extend, orientation='horizontal',
                        aspect=10)
    clabel = 'surface density $[\\log_{10} \\mathrm{g}\\,\\mathrm{cm}^{-2}]$'
    cax.set_xlabel(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize-1)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')


def run_checkcenter_massmap(index, center=None, rvir=None,
                            masstype='gas'):
    outdir = '/projects/b1026/nastasha/tests/start_fire/map_tests/'
    cen = center
    mincols = {'gas': -5.,
               'DM': -5.,
               'stars': -7.,
               'BH': None}
    if index == 0:
        dirpath = '/projects/b1026/snapshots/fire3/m13h206_m3e5/' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000/' 
        simname = 'm13h206_m3e5__' + \
                  'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'                     
        snapnum = 27  
        outfilen_template = 'mass_pt{pt}_{sc}_snap{sn}_axis-{ax}_' + \
                            'wholezoom_v1.hdf5'
        _temp = outdir + outfilen_template 
        mapfilens = {'gas': _temp.format(pt=0, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),
                     'DM': _temp.format(pt=1, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),
                     'stars': _temp.format(pt=4, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),
                     'BH': _temp.format(pt=5, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),                            
                    }
        cen =  [48414.20743443, 49480.35333529, 48451.20700497]
    mapfile_template = mapfilens[masstype]
    
    checkcenter_massmap(mapfile_template, savename=None, 
                        mincol=mincols[masstype],
                        center_simunits=cen, Rvir_simunits=rvir)
    

def masstest_map(filens):
    '''
    files for all parttypes
    '''
    
    # mass in g will overflow 
    enclmass = np.float64(0.)
    for filen in filens:
        with h5py.File(filen, 'r') as f:
            map = 10**f['map'][:]

            box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
            cosmopars = {key: val for key, val in \
                         f['Header/inputpars/cosmopars'].attrs.items()}
            #print(cosmopars)
            halopath = 'Header/inputpars/halodata'
            rvir_ckpcoverh = f[halopath].attrs['Rvir_ckpcoverh']
            mvir_msunoverh = np.float64(f[halopath].attrs['Mvir_Msunoverh'])
            pixsize_pkpc = f['Header/inputpars'].attrs['pixsize_pkpc']
            rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
            xax = f['Header/inputpars'].attrs['Axis1']
            yax = f['Header/inputpars'].attrs['Axis2']
            box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
            xcminmax = (-0.5 * box_pkpc[xax] + 0.5 * pixsize_pkpc, 
                        0.5 * box_pkpc[xax] - 0.5 * pixsize_pkpc)
            ycminmax = (-0.5 * box_pkpc[yax] + 0.5 * pixsize_pkpc,
                        0.5 * box_pkpc[yax] - 0.5 * pixsize_pkpc)
            npix_x = map.shape[0]
            npix_y = map.shape[1]
            pixdist2_pkpc = np.linspace(xcminmax[0], xcminmax[1], npix_x)**2 +\
                            np.linspace(ycminmax[0], ycminmax[1], npix_y)**2
            mapsel = pixdist2_pkpc < rvir_pkpc**2
            pixarea_cm = (pixsize_pkpc * 1e-3 * c.cm_per_mpc)**2
            partmass = np.float64(np.sum(map[mapsel])) * pixarea_cm
            enclmass += partmass
    halomass_g = mvir_msunoverh * c.solar_mass / cosmopars['h']
    print('Found Mvir (AHF) = {:.3e} g'.format(halomass_g))
    print('Found enclosed mass in projection = {:.3e} g'.format(enclmass))

def ionbal_test(filens, simlabel=None):
    '''
    for a series of files containing the tables and data for a full
    ionisation series: plot the table and interpolated ion balances
    and get a sense of whether the differences are ok
    Also get the total element fraction, to check if it makes sense
    with depletion.
    '''
    if simlabel is None:
        simlabel = 'testhalo1-m13h206_m3e5'

    tot_iontab = None
    logTtab = None
    lognHtab = None

    logTsim = None
    lognHsim = None
    Zsim = None
    tot_ionsim = None

    target_Z = None
    delta_Z = None
    
    redshift = None
    ions = []

    imgdir = '/'.join(filens[0].split('/')[:-1]) + '/'
    _imgname = 'ionbal-test_{ion}_depletion-{dep}_Z-{met}_z-{z:.1f}{lt}_{simname}.pdf'

    for filen in filens:
        with h5py.File(filen, 'r') as f:
            # title info
            cosmopars = {key: val for key, val in \
                         f['Header/cosmopars'].attrs.items()}
            if redshift is None:
                redshift = cosmopars['z']
            elif not redshift == cosmopars['z']:
                msg = 'Input files have different redshifts; found in {}'
                raise ValueError(msg.format(filen))
            _delta_Z = f['Header'].attrs['delta_Z']
            if delta_Z is None:
                delta_Z = _delta_Z
            elif not delta_Z == _delta_Z:
                msg = 'Input files have different delta_Z; found in {}'
                raise ValueError(msg.format(filen))
            _target_Z = f['Header'].attrs['target_Z']
            if target_Z is None:
                target_Z = _target_Z
            elif not target_Z == _target_Z:
                msg = 'Input files have different target_Z; found in {}'
                raise ValueError(msg.format(filen))
            ion = f['Header'].attrs['ion'].decode()
            ps20depletion = bool(f['Header'].attrs['ps20depletion'])
            ions.append(ion)
            if 'lintable' in f['Header'].attrs:
                dolintable = True
                lintable = bool(f['Header'].attrs['lintable']) 
                ltlabel = '_lintable-{}'.format(lintable)
            else:
                dolintable = False
                ltlabel = ''
  
            # table data
            if logTtab is None:
                logTtab = f['iontab_data/logT_K'][:]
            elif not np.all(logTtab == f['iontab_data/logT_K'][:]):
                msg = 'Input files have different table log T bins; found in {}'
                raise ValueError(msg.format(filen))
            if lognHtab is None:
                lognHtab = f['iontab_data/lognH_cm**-3'][:]
            elif not np.all(lognHtab == f['iontab_data/lognH_cm**-3'][:]):
                msg = 'Input files have different table log nH bins; found in {}'
                raise ValueError(msg.format(filen))
            iontab = f['iontab_data/ionbal_T_nH'][:].T
            if tot_iontab is None:
                tot_iontab = iontab
            else:
                tot_iontab += iontab 
            
            # sim data
            if logTsim is None:
                logTsim = np.log10(f['simulation_data/T_K'])
            elif not np.all(logTsim == np.log10(f['simulation_data/T_K'])):
                msg = 'Input files have different simulation log T values; found in {}'
                raise ValueError(msg.format(filen))
            if lognHsim is None:
                lognHsim = np.log10(f['simulation_data/nH_cm**-3'])
            elif not np.all(lognHsim == np.log10(f['simulation_data/nH_cm**-3'])):
                msg = 'Input files have different simulation log nH values; found in {}'
                raise ValueError(msg.format(filen))
            if Zsim is None:
                Zsim = f['simulation_data/metallicity_abs_mass_frac'][:]
            elif not np.all(Zsim == f['simulation_data/metallicity_abs_mass_frac'][:]):
                msg = 'Input files have different simulation Z values; found in {}'
                raise ValueError(msg.format(filen))
            ionsim = f['simulation_data/ionbal'][:]
            if tot_ionsim is None:
                tot_ionsim = ionsim
            else:
                tot_ionsim += ionsim
        
        outname = imgdir + _imgname.format(simname=simlabel, ion=ion, 
                                           dep=ps20depletion,
                                           met=target_Z, z=redshift,
                                           lt=ltlabel)
            
        title = '{ion} PS20 table at z={z:.2f}, Z={met:.1e} vs. interp. FIRE data, dust depl. {dep}' 
        title = title.format(ion=ion, dep=ps20depletion, z=redshift, met=target_Z)
        if dolintable:
            title = title + ', lin. table {}'.format(lintable)
        
        fig = plt.figure(figsize=(13., 4.))
        grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.5, 
                            width_ratios=[1., 0.1, 1., 0.1, 1.],
                            left=0.05, right=0.95)
        axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
        fontsize = 12
        cmap = 'viridis'
        size = 10.
        vmin = -10
        vmax = 0.

        fig.suptitle(title, fontsize=fontsize)

        ax = axes[0]
        cax = axes[1]

        xedges = lognHtab[:-1] - 0.5 * np.diff(lognHtab)
        xend = [lognHtab[-1] - 0.5 * (lognHtab[-1] - lognHtab[-2]),
                lognHtab[-1] + 0.5 * (lognHtab[-1] - lognHtab[-2])
                ]
        xedges = np.append(xedges, xend)
        yedges = logTtab[:-1] - 0.5 * np.diff(logTtab)
        yend = [logTtab[-1] - 0.5 * (logTtab[-1] - logTtab[-2]),
                logTtab[-1] + 0.5 * (logTtab[-1] - logTtab[-2])
                ]
        yedges = np.append(yedges, yend)
        img = ax.pcolormesh(xedges, yedges, np.log10(iontab.T), cmap=cmap, 
                            rasterized=True, vmin=vmin, vmax=vmax)
        plt.colorbar(img, cax=cax)
        cax.set_ylabel('log ion fraction', fontsize=fontsize)

        ax.scatter(lognHsim, logTsim, s=size, c=np.log10(ionsim),
                   edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax,
                   rasterized=True)
        ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                      fontsize=fontsize)
        xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                 ' [\\mathrm{cm}^{-3}]$'
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
        cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                        right=False, left=True)
        cax.yaxis.set_label_position('left')
        
        ax = axes[2]
        cax = axes[3]
        
        Tinds = np.argmin(np.abs(logTsim[:, np.newaxis] 
                                 - logTtab[np.newaxis, :]), axis=1)
        nHinds = np.argmin(np.abs(lognHsim[:, np.newaxis] 
                                 - lognHtab[np.newaxis, :]), axis=1)
        closest_gridtosim = iontab[(nHinds, Tinds)]
        dz = Zsim - target_Z
        vmin = -1. * delta_Z
        vmax = delta_Z
        
        delta_sim = closest_gridtosim - ionsim
        xvals_sim = np.random.uniform(low=0.0, high=1.0, size=len(logTsim))

        delta_tab1 = (iontab[1:, :] - iontab[:-1, :]).flatten()
        delta_tab2 = (iontab[:, 1:] - iontab[:, :-1]).flatten()
        xvals_tab1 = np.random.uniform(low=0.0, high=1.0, size=len(delta_tab1))
        xvals_tab2 = np.random.uniform(low=0.0, high=1.0, size=len(delta_tab2))

        ax.scatter(xvals_tab1, delta_tab1, s=0.5*size, color='black', 
                   label='$\\Delta$ table grid', rasterized=True)
        ax.scatter(xvals_tab1, -1. * delta_tab1, s=0.5*size, color='black',
                   rasterized=True)
        ax.scatter(xvals_tab2, delta_tab2, s=0.5*size, color='black',
                   rasterized=True)
        ax.scatter(xvals_tab2, -1. * delta_tab2, s=0.5*size, color='black',
                   rasterized=True)

        img = ax.scatter(xvals_sim, delta_sim, s=size, c=dz,
                         edgecolor='black', cmap='RdBu', vmin=vmin, vmax=vmax,
                         label='sim - table', rasterized=True)
        plt.colorbar(img, cax=cax, extend='neither')
        ax.set_ylabel('difference with nearest table value', 
                      fontsize=fontsize)
        #cax.set_ylabel('simulation Z - table Z', fontsize=fontsize,
        #               horizontalalignment='center', verticalalignment='center',
        #               x=0.5, y=0.5)
        cax.text(0.5, 0.5, 'simulation Z - table Z', fontsize=fontsize,
                 horizontalalignment='center', verticalalignment='center',
                 rotation='vertical', transform=cax.transAxes)
        #cax.yaxis.set_label_coords(0.5, 0.5)
        cax.tick_params(labelsize=fontsize - 3, labelright=False, labelleft=True,
                        right=False, left=True)
        #cax.yaxis.set_label_position('left')
        ax.legend(fontsize=fontsize - 1)
        ax.tick_params(labelbottom=False, bottom=False)

        ax = axes[4]

        nbins = 100
        maxv = np.max(np.abs(delta_tab1))
        maxv = max(maxv, np.max(np.abs(delta_tab2)))
        maxv = max(maxv, np.max(np.abs(delta_sim)))
        bins = np.linspace(-1. * maxv, maxv, nbins)
        
        tabvals = np.append(delta_tab1, -1. * delta_tab1)
        tabvals = np.append(tabvals, delta_tab2)
        tabvals = np.append(tabvals, -1. * delta_tab2)

        ax.set_yscale('log')
        ax.hist(tabvals, bins=bins, histtype='step', color='black',
                label='$\\Delta$ table grid', align='mid', density=True)
        ax.hist(delta_sim, bins=bins, histtype='step', color='blue',
                label='sim - table', linestyle='dashed', align='mid', 
                density=True)

        ax.set_xlabel('difference with nearest table value', 
                      fontsize=fontsize)
        ax.set_ylabel('probability density', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 1)
        
        plt.savefig(outname, bbox_inches='tight')
    
    outname = imgdir + _imgname.format(simname=simlabel, ion='-'.join(ions), 
                                       dep=ps20depletion,
                                       met=target_Z, z=redshift,
                                       lt=ltlabel)
        
    title = 'sum of {ion} PS20 tables at z={z:.2f}, Z={met:.1e} vs. ' + \
            'interp. FIRE data, dust depl. {dep}' 
    title = title.format(ion=', '.join(ions), dep=ps20depletion, 
                         z=redshift, met=target_Z)
    if dolintable:
        title = title + ', lin. table {}'.format(lintable)
    
    fig = plt.figure(figsize=(13., 4.))
    grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.5, 
                        width_ratios=[1., 0.1, 1., 0.1, 1.],
                        left=0.05, right=0.95)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    fontsize = 12
    cmap = 'RdBu'
    size = 10.
        
    vmax = np.max(np.abs(tot_ionsim - 1.))
    vmax = max(vmax, np.max(np.abs(tot_iontab - 1.)))
    vmin = 1. - vmax
    vmax = 1. + vmax

    fig.suptitle(title, fontsize=fontsize)

    ax = axes[0]
    cax = axes[1]

    xedges = lognHtab[:-1] - 0.5 * np.diff(lognHtab)
    xend = [lognHtab[-1] - 0.5 * (lognHtab[-1] - lognHtab[-2]),
            lognHtab[-1] + 0.5 * (lognHtab[-1] - lognHtab[-2])
            ]
    xedges = np.append(xedges, xend)
    yedges = logTtab[:-1] - 0.5 * np.diff(logTtab)
    yend = [logTtab[-1] - 0.5 * (logTtab[-1] - logTtab[-2]),
            logTtab[-1] + 0.5 * (logTtab[-1] - logTtab[-2])
            ]
    yedges = np.append(yedges, yend)
    img = ax.pcolormesh(xedges, yedges, tot_iontab.T, cmap=cmap, 
                        rasterized=True, vmin=vmin, vmax=vmax)
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('ion fraction sum', fontsize=fontsize)

    ax.scatter(lognHsim, logTsim, s=size, c=tot_ionsim,
                edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax,
                rasterized=True)
    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
    
    ax = axes[2]
    cax = axes[3]
    
    Tinds = np.argmin(np.abs(logTsim[:, np.newaxis] 
                                - logTtab[np.newaxis, :]), axis=1)
    nHinds = np.argmin(np.abs(lognHsim[:, np.newaxis] 
                                - lognHtab[np.newaxis, :]), axis=1)
    closest_gridtosim = tot_iontab[(nHinds, Tinds)]
    dz = Zsim - target_Z
    vmin = -1. * delta_Z
    vmax = delta_Z
    
    delta_sim = closest_gridtosim - tot_ionsim
    xvals_sim = np.random.uniform(low=0.0, high=1.0, size=len(logTsim))

    delta_tab1 = (tot_iontab[1:, :] - tot_iontab[:-1, :]).flatten()
    delta_tab2 = (tot_iontab[:, 1:] - tot_iontab[:, :-1]).flatten()
    xvals_tab1 = np.random.uniform(low=0.0, high=1.0, size=len(delta_tab1))
    xvals_tab2 = np.random.uniform(low=0.0, high=1.0, size=len(delta_tab2))

    ax.scatter(xvals_tab1, delta_tab1, s=0.5*size, color='black', 
                label='$\\Delta$ table grid', rasterized=True)
    ax.scatter(xvals_tab1, -1. * delta_tab1, s=0.5*size, color='black',
                rasterized=True)
    ax.scatter(xvals_tab2, delta_tab2, s=0.5*size, color='black',
                rasterized=True)
    ax.scatter(xvals_tab2, -1. * delta_tab2, s=0.5*size, color='black',
                rasterized=True)

    img = ax.scatter(xvals_sim, delta_sim, s=size, c=dz,
                     edgecolor='black', cmap='RdBu', vmin=vmin, vmax=vmax,
                     label='sim - table', rasterized=True)
    plt.colorbar(img, cax=cax, extend='neither')
    ax.set_ylabel('difference with nearest table value', 
                fontsize=fontsize)
    #cax.set_ylabel('simulation Z - table Z', fontsize=fontsize,
    #               horizontalalignment='center', verticalalignment='center',
    #               x=0.5, y=0.5)
    cax.text(0.5, 0.5, 'simulation Z - table Z', fontsize=fontsize,
                horizontalalignment='center', verticalalignment='center',
                rotation='vertical', transform=cax.transAxes)
    #cax.yaxis.set_label_coords(0.5, 0.5)
    cax.tick_params(labelsize=fontsize - 3, labelright=False, labelleft=True,
                    right=False, left=True)
    #cax.yaxis.set_label_position('left')
    ax.legend(fontsize=fontsize - 1)
    ax.tick_params(labelbottom=False, bottom=False)

    ax = axes[4]

    nbins = 100
    maxv = np.max(np.abs(delta_tab1))
    maxv = max(maxv, np.max(np.abs(delta_tab2)))
    maxv = max(maxv, np.max(np.abs(delta_sim)))
    bins = np.linspace(-1. * maxv, maxv, nbins)
    
    tabvals = np.append(delta_tab1, -1. * delta_tab1)
    tabvals = np.append(tabvals, delta_tab2)
    tabvals = np.append(tabvals, -1. * delta_tab2)

    ax.set_yscale('log')
    ax.hist(tabvals, bins=bins, histtype='step', color='black',
            label='$\\Delta$ table grid', align='mid', density=True)
    ax.hist(delta_sim, bins=bins, histtype='step', color='blue',
            label='sim - table', linestyle='dashed', align='mid', 
            density=True)

    ax.set_xlabel('difference with nearest table value', 
                    fontsize=fontsize)
    ax.set_ylabel('probability density', fontsize=fontsize)
    ax.legend(fontsize=fontsize - 1)
    
    plt.savefig(outname, bbox_inches='tight')

    # sum test tables
    _imgname = 'ionbal-test_tablesum_{ion}_depletion-{dep}_Z-{met}_z-{z:.1f}{lt}.pdf'
    outname = imgdir + _imgname.format(simname=simlabel, ion=ion, 
                                       dep=ps20depletion,
                                       met=target_Z, z=redshift,
                                       lt=ltlabel)

    title = 'sum of {ion} PS20 tables at z={z:.2f}, Z={met:.1e} vs. 1.0 dust depl. {dep}' 
    title = title.format(ion=', '.join(ions), dep=ps20depletion, 
                         z=redshift, met=target_Z)
    
    fig = plt.figure(figsize=(13., 4.))
    grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.5, 
                        width_ratios=[1., 0.1, 1., 0.1, 1.],
                        left=0.05, right=0.95)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    fontsize = 12
    cmap = 'RdBu'
    size = 10.
        
    vmax = np.max(np.abs(tot_ionsim - 1.))
    vmax = max(vmax, np.max(np.abs(tot_iontab - 1.)))
    vmin = 1. - vmax
    vmax = 1. + vmax

    fig.suptitle(title, fontsize=fontsize)

    ax = axes[0]
    cax = axes[1]

    xedges = lognHtab[:-1] - 0.5 * np.diff(lognHtab)
    xend = [lognHtab[-1] - 0.5 * (lognHtab[-1] - lognHtab[-2]),
            lognHtab[-1] + 0.5 * (lognHtab[-1] - lognHtab[-2])
            ]
    xedges = np.append(xedges, xend)
    yedges = logTtab[:-1] - 0.5 * np.diff(logTtab)
    yend = [logTtab[-1] - 0.5 * (logTtab[-1] - logTtab[-2]),
            logTtab[-1] + 0.5 * (logTtab[-1] - logTtab[-2])
            ]
    yedges = np.append(yedges, yend)
    img = ax.pcolormesh(xedges, yedges, tot_iontab.T, cmap=cmap, 
                        rasterized=True, vmin=vmin, vmax=vmax)
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('ion fraction sum', fontsize=fontsize)

    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
    
    ax = axes[2]
    cax = axes[3]
    
    delta_tab1 = (tot_iontab - 1.).flatten()
    xvals_tab1 = np.random.uniform(low=0.0, high=1.0, size=len(delta_tab1))
    c_tab1 = np.repeat(lognHtab[:, np.newaxis], tot_iontab.shape[1], axis=1)
    vmid = 1.5
    vmin = np.min(lognHtab)
    vmax = np.max(lognHtab)
    tv = (vmid - vmin) / (vmax - vmin) 
    cmap = pu.paste_cmaps(['plasma', 'viridis'], [vmin, vmid, vmax], 
                          [(0., tv), (tv, 1.)])

    img = ax.scatter(xvals_tab1, delta_tab1, s=0.5*size, c=c_tab1, 
                     rasterized=True, cmap=cmap)

    plt.colorbar(img, cax=cax, extend='neither')
    ax.set_ylabel('table sum - 1.', fontsize=fontsize)
    #cax.set_ylabel('simulation Z - table Z', fontsize=fontsize,
    #               horizontalalignment='center', verticalalignment='center',
    #               x=0.5, y=0.5)
    clabel = '$\\log_{10}\\,\\mathrm{n}_{\\mathrm{H}}\\;[\\mathrm{cm}^{-3}]$'
    cax.text(0.5, 0.5, clabel, fontsize=fontsize,
                horizontalalignment='center', verticalalignment='center',
                rotation='vertical', transform=cax.transAxes)
    #cax.yaxis.set_label_coords(0.5, 0.5)
    cax.tick_params(labelsize=fontsize - 3, labelright=False, labelleft=True,
                    right=False, left=True)
    #cax.yaxis.set_label_position('left')
    ax.legend(fontsize=fontsize - 1)
    ax.tick_params(labelbottom=False, bottom=False)

    ax = axes[4]

    nbins = 100
    maxv = np.max(np.abs(delta_tab1))
    bins = np.linspace(-1. * maxv, maxv, nbins)
    
    tabvals = delta_tab1

    ax.set_yscale('log')
    ax.hist(tabvals, bins=bins, histtype='step', color='black',
            label=None, align='mid', density=True)

    ax.set_xlabel('table sum - 1.', 
                    fontsize=fontsize)
    ax.set_ylabel('probability density', fontsize=fontsize)
    ax.legend(fontsize=fontsize - 1)
    
    plt.savefig(outname, bbox_inches='tight')

    outname = imgdir + _imgname.format(simname=simlabel, ion='-'.join(ions), 
                                       dep=ps20depletion,
                                       met=target_Z, z=redshift,
                                       lt=ltlabel)

    # sum test interp
    _imgname = 'ionbal-test_interp-sum_{ion}_depletion-{dep}_Z-{met}'+\
               '_z-{z:.1f}{lt}_{simname}.pdf'
    outname = imgdir + _imgname.format(simname=simlabel, ion='-'.join(ions), 
                                       dep=ps20depletion,
                                       met=target_Z, z=redshift,
                                       lt=ltlabel)
        
    title = 'sum of interpolated {ion} PS20 tables at z={z:.2f}, '+ \
            'Z={met:.1e} from FIRE data, dust depl. {dep}' 
    title = title.format(ion=', '.join(ions), dep=ps20depletion, 
                         z=redshift, met=target_Z)
    if dolintable:
        title = title + ', lin. table {}'.format(lintable)
    
    fig = plt.figure(figsize=(13., 4.))
    grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.5, 
                        width_ratios=[1., 0.1, 1., 0.1, 1.],
                        left=0.05, right=0.95)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    fontsize = 12
    cmap = 'RdBu'
    size = 10.
        
    vmax = np.max(np.abs(tot_ionsim - 1.))
    vmax = max(vmax, np.max(np.abs(tot_iontab - 1.)))
    vmin = 1. - vmax
    vmax = 1. + vmax

    fig.suptitle(title, fontsize=fontsize)

    ax = axes[0]
    cax = axes[1]

    img = ax.scatter(lognHsim, logTsim, s=size, c=tot_ionsim,
                     edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax,
                     rasterized=True)
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('ion fraction sum', fontsize=fontsize)

    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
    
    ax = axes[2]
    cax = axes[3]
    
    delta_sim = tot_ionsim - 1.
    xvals_sim = np.random.uniform(low=0.0, high=1.0, size=len(logTsim))
    vmid = 1.5
    vmin = np.min(lognHsim)
    vmax = np.max(lognHsim)
    tv = (vmid - vmin) / (vmax - vmin) 
    if vmid >= vmax:
        cmap = 'plasma'
    else:
        cmap = pu.paste_cmaps(['plasma', 'viridis'], [vmin, vmid, vmax], 
                              [(0., tv), (tv, 1.)])

    img = ax.scatter(xvals_sim, delta_sim, s=size, c=lognHsim,
                     edgecolor='black', cmap=cmap,
                     label='interp. sum - 1.', rasterized=True)
    plt.colorbar(img, cax=cax, extend='neither')
    ax.set_ylabel('interp. sum - 1.', fontsize=fontsize)
    #cax.set_ylabel('simulation Z - table Z', fontsize=fontsize,
    #               horizontalalignment='center', verticalalignment='center',
    #               x=0.5, y=0.5)
    cax.text(0.5, 0.5, clabel, fontsize=fontsize,
                horizontalalignment='center', verticalalignment='center',
                rotation='vertical', transform=cax.transAxes)
    #cax.yaxis.set_label_coords(0.5, 0.5)
    cax.tick_params(labelsize=fontsize - 3, labelright=False, labelleft=True,
                    right=False, left=True)
    #cax.yaxis.set_label_position('left')
    ax.legend(fontsize=fontsize - 1)
    ax.tick_params(labelbottom=False, bottom=False)

    ax = axes[4]

    nbins = 100
    maxv = np.max(np.abs(delta_sim))
    bins = np.linspace(-1. * maxv, maxv, nbins)

    ax.set_yscale('log')
    ax.hist(delta_sim, bins=bins, histtype='step', color='blue',
            label=None, linestyle='dashed', align='mid', 
            density=True)

    ax.set_xlabel('interp. sim - 1.', fontsize=fontsize)
    ax.set_ylabel('probability density', fontsize=fontsize)
    ax.legend(fontsize=fontsize - 1)
    
    plt.savefig(outname, bbox_inches='tight')


def run_ionbal_tests(index):
    # laptop
    ddir = '/Users/nastasha/ciera/tests/fire_start/ionbal_tests/'
    simname = 'm13h206_m3e5__m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1'+\
              '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    pre = 'ionbal_test_PS20'
    ftmp = ['{pre}_{ion}_depletion-False_Z-0.01_snap045{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-False_Z-0.0001_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-False_Z-0.01_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.0001_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.01_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.01_snap045{lt}_{simname}.hdf5',
            ]
    _index = index % 6
    lti = index // 6
    if lti == 0:
        lt = ''
    elif lti == 1:
        lt = '_lintable-False'
    elif lti == 2:
        lt = '_lintable-True'
    ions = ['O{}'.format(i) for i in range(1, 10)]
    filens = [ddir + ftmp[_index].format(pre=pre, simname=simname, 
                                         ion=ion, lt=lt) \
              for ion in ions]
    ionbal_test(filens)

def test_tablesum_direct(ztargets = [0., 1., 3.], element='oxygen'):
    '''
    forget interpolations, do the maps look right on their own.
    '''
    fn = '/Users/nastasha/phd/tables/ionbal/lines_sp20/'+\
         'UVB_dust1_CR1_G1_shield1.hdf5'
    outdir = '/Users/nastasha/ciera/tests/fire_start/ionbal_tests/'
    savename = outdir + 'sumtest-nodepl_z-{z:.3f}_{elt}_tables-direct.pdf'

    with h5py.File(fn, 'r') as f:
        mgrp = f['Tdep/IonFractions']
        eltkeys = list(mgrp.keys())
        eltgrp = [key if element.lower() in key else None\
                  for key in eltkeys]
        eltgrp = list(set(eltgrp))
        eltgrp.remove(None)
        eltgrp = mgrp[eltgrp[0]]
        # redshift, T, Z, nH, ion
        ztab = f['TableBins/RedshiftBins'][:]
        logT = f['TableBins/TemperatureBins'][:]
        met = f['TableBins/MetallicityBins'][:]
        lognH = f['TableBins/DensityBins'][:]
        nHcut = 1.0
        nHcuti = np.min(np.where(lognH >= nHcut)[0])
        fontsize = 12
        nHedges = lognH[:-1] - 0.5 * np.diff(lognH)
        nHend = [lognH[-1] - 0.5 * (lognH[-1] - lognH[-2]),
                 lognH[-1] + 0.5 * (lognH[-1] - lognH[-2])
                 ]
        nHedges = np.append(nHedges, nHend)
        Tedges = logT[:-1] - 0.5 * np.diff(logT)
        Tend = [logT[-1] - 0.5 * (logT[-1] - logT[-2]),
                logT[-1] + 0.5 * (logT[-1] - logT[-2])
                ]
        Tedges = np.append(Tedges, Tend)
        
        for ztar in ztargets:
            Zstart = 0 if element in ['hydrogen', 'helium'] else 1

            ncols = 4
            nz = len(met) - Zstart
            nrows = ((nz - 1) // ncols + 1) * 2
            width = 11.
            width_ratios = [1.] * ncols + [0.1]
            wspace = 0.3
            panelwidth = width / (sum(width_ratios) + ncols * wspace)
            hspace = 0.4
            height = (hspace * (nrows - 1.) + nrows) * panelwidth 

            fig = plt.figure(figsize=(width, height))
            grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=hspace,
                                wspace=wspace, 
                                width_ratios=width_ratios,
                                top=0.95, bottom=0.05)
            cax = fig.add_subplot(grid[:2, ncols])
            pdaxes = [fig.add_subplot(grid[2 * (i // 4), i % 4]) \
                      for i in range(nz)]
            haxes =  [fig.add_subplot(grid[2 * (i // 4) + 1, i % 4]) \
                       for i in range(nz)]

            zi = np.argmin(np.abs(ztab - ztar))
            zval = ztab[zi]
            _savename = savename.format(z=zval, elt=element)
            title = 'Sum of {elt} ions at z={z:.3f}'
            fig.suptitle(title.format(elt=element, z=zval), fontsize=fontsize)
            # T, Z, nH
            tabsum = np.sum(10**eltgrp[zi, :, Zstart:, :, :], axis=3)
            vmax = np.max(np.abs(tabsum - 1.))
            vmin = 1. - vmax
            vmax = 1. + vmax
            cmap = 'RdBu'
            #hbins = np.linspace(vmin, vmax, 100)

            nHlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                      '[\\mathrm{cm}^{-3}]$'
            Tlabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
            flabel = 'sum of ion fractions'

            # first value is Z = 0.0 -> no metal ions 
            for _Zi, _Z in enumerate(met[Zstart:]):
                label = '$ \\log_{{10}}Z/Z_{{\\odot}}$\n={:.1f}'.format(_Z)
                pdax = pdaxes[_Zi]
                hax = haxes[_Zi]
                pdax.text(0.05, 0.95, label, fontsize=fontsize - 2, 
                          horizontalalignment='left', verticalalignment='top',
                          transform=pdax.transAxes)
                hax.text(0.05, 0.95, label, fontsize=fontsize - 2, 
                          horizontalalignment='left', verticalalignment='top',
                          transform=hax.transAxes)
                pdax.set_xlabel(nHlabel, fontsize=fontsize)
                hax.set_xlabel(flabel, fontsize=fontsize)
                if _Zi % ncols == 0: 
                    pdax.set_ylabel(Tlabel, fontsize=fontsize)
                    hax.set_ylabel('probability density', fontsize=fontsize)

                img = pdax.pcolormesh(nHedges, Tedges, 
                                      tabsum[:, _Zi, :], 
                                      vmin=vmin, vmax=vmax, cmap=cmap)
                pdax.axvline(nHcut, color='black', linestyle='dashed')

                nHhi = (tabsum[:, _Zi, nHcuti:]).flatten()
                nHlo = (tabsum[:, _Zi, :nHcuti]).flatten()
                
                hax.set_yscale('log')
                dmax = np.max(np.abs(nHhi - 1.))
                dmax = max(dmax, np.max(np.abs(nHlo - 1.)))
                hmin = 1. - dmax
                hmax = 1 + dmax
                hbins = np.linspace(hmin, hmax, 100)
                
                histlo, _ = np.histogram(nHlo, bins=hbins, density=True)
                histhi, _ = np.histogram(nHhi, bins=hbins, density=True)
                hax.set_yscale('log')
                hax.step(hbins[:-1], histlo, where='post', 
                         label='log nH >= {:.1f}'.format(nHcut),
                         color='blue')
                hax.step(hbins[:-1], histhi, where='post', 
                         label='log nH >= {:.1f}'.format(nHcut),
                         color='black', linestyle='dashed')

                #hv1, _, _ = hax.hist(nHlo, bins=hbins, density=True, 
                #                     label='log nH >= {:.1f}'.format(nHcut),
                #                     color='blue', align='mid', 
                #                     histtype='step')
                #hv2, _, _ = hax.hist(nHhi, bins=hbins, density=True, 
                #                     label='log nH < {:.1f}'.format(nHcut),
                #                     color='black', linestyle='dashed', 
                #                     align='mid', histtype='step')
                ymin = min(np.min(histlo[histlo > 0.]), 
                           np.min(histhi[histhi > 0.]))
                _ymin, _ymax = hax.get_ylim()
                hax.set_ylim((0.7 * ymin, _ymax))

                if _Zi == 0:
                    hax.legend(fontsize=fontsize - 2)
            
            plt.colorbar(img, cax=cax)
            cax.set_ylabel(flabel, fontsize=fontsize)
            plt.savefig(_savename, bbox_inches='tight')

def test_lin_option_ionbal(filen_old, filen_new_log, filen_new_lin):

    imgdir = '/'.join(filen_old.split('/')[:-1]) + '/'
    imgname_oldnew = 'ionbal_comp_test_ps20-logtable_ps20-old_{ion}' + \
                     '_{sim}_z-{z:.2f}_logZ-{met:.2f}_depletion-{dep}.pdf'
    imgname_linlog = 'ionbal_comp_test_ps20-log-lintable' + \
                     '_{sim}_z-{z:.2f}_logZ-{met:.2f}_depletion-{dep}.pdf'
    simlabel = 'testhalo1-m13h206_m3e5'

    logTsim = None
    lognHsim = None
    Zsim = None

    target_Z = None
    delta_Z = None
    ion = None
    ps20depletion = None
    
    redshift = None

    dct_old = {}
    dct_new_log = {}
    dct_new_lin = {}

    for filen, dct in zip([filen_old, filen_new_log, filen_new_lin],
                          [dct_old, dct_new_log, dct_new_lin]):
        with h5py.File(filen, 'r') as f:
            # title info
            cosmopars = {key: val for key, val in \
                        f['Header/cosmopars'].attrs.items()}
            if redshift is None:
                redshift = cosmopars['z']
            elif not redshift == cosmopars['z']:
                msg = 'Input files have different redshifts; found in {}'
                raise ValueError(msg.format(filen))
            _delta_Z = f['Header'].attrs['delta_Z']
            if delta_Z is None:
                delta_Z = _delta_Z
            elif not delta_Z == _delta_Z:
                msg = 'Input files have different delta_Z; found in {}'
                raise ValueError(msg.format(filen))
            _target_Z = f['Header'].attrs['target_Z']
            if target_Z is None:
                target_Z = _target_Z
            elif not target_Z == _target_Z:
                msg = 'Input files have different target_Z; found in {}'
                raise ValueError(msg.format(filen))
            _ion = f['Header'].attrs['ion'].decode()
            if ion is None:
                ion = _ion
            elif not ion == _ion:
                msg = 'Input files have different ion; found in {}'
                raise ValueError(msg.format(filen))
            _ps20depletion = bool(f['Header'].attrs['ps20depletion'])
            if ps20depletion is None:
                ps20depletion = _ps20depletion
            elif not ps20depletion == _ps20depletion:
                msg = 'Input files have different ps20depletion; found in {}'
                raise ValueError(msg.format(filen))
            if 'lintable' in f['Header'].attrs:
                dct['lintable'] = bool(f['Header'].attrs['lintable']) 
            else:
                dct['lintable'] = None
            
            # sim data
            if logTsim is None:
                logTsim = np.log10(f['simulation_data/T_K'])
            elif not np.all(logTsim == np.log10(f['simulation_data/T_K'])):
                msg = 'Input files have different simulation log T values; found in {}'
                raise ValueError(msg.format(filen))
            if lognHsim is None:
                lognHsim = np.log10(f['simulation_data/nH_cm**-3'])
            elif not np.all(lognHsim == np.log10(f['simulation_data/nH_cm**-3'])):
                msg = 'Input files have different simulation log nH values; found in {}'
                raise ValueError(msg.format(filen))
            if Zsim is None:
                Zsim = f['simulation_data/metallicity_abs_mass_frac'][:]
            elif not np.all(Zsim == f['simulation_data/metallicity_abs_mass_frac'][:]):
                msg = 'Input files have different simulation Z values; found in {}'
                raise ValueError(msg.format(filen))
            dct['ionsim'] = f['simulation_data/ionbal'][:]
    
    # new log vs. old version
    title = 'interpolated {ion} PS20 tables at z={z:.2f}, '+ \
            'Z={met:.1e} from FIRE data, dust depl. {dep}' 
    title = title.format(ion=ion, dep=ps20depletion, 
                         z=redshift, met=target_Z)
    
    fig = plt.figure(figsize=(13., 4.))
    grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.5, 
                        width_ratios=[1., 0.1, 1., 0.1, 1.],
                        left=0.05, right=0.95)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    fontsize = 12
    cmap = 'viridis'
    size = 10.
    
    ionbal_old = dct_old['ionsim']
    ionbal_new_log = dct_new_log['ionsim']
    ionbal_new_lin = dct_new_lin['ionsim']
    vmax = 0.
    vmin = -10.

    fig.suptitle(title, fontsize=fontsize)

    ax = axes[0]
    cax = axes[1]
    img = ax.scatter(lognHsim, logTsim, s=size, c=np.log10(ionbal_old),
                     edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax,
                     rasterized=True)
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('$\\log_{10}$ ion fraction', fontsize=fontsize)

    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
    ax.text(0.05, 0.05, 'Old table interp.,\nlog space',
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    ax = axes[2]
    cax = axes[3]
    delta_oldnew = np.log10(ionbal_new_log) - np.log10(ionbal_old)
    if np.all(ionbal_new_log == ionbal_old):
        print('Old, new log maps are same to fp precision')
        vmax = 1e-49
        vmin = -1. * vmax
        dohist = False
    else:
        vmax = np.max(np.abs(delta_oldnew))
        vmin = -1. * vmax
        dohist = True

    img = ax.scatter(lognHsim, logTsim, s=size, 
                     c=delta_oldnew,
                     edgecolor='black', cmap='RdBu', vmin=vmin, vmax=vmax,
                     rasterized=True)
    plt.colorbar(img, cax=cax)
    clabel = '$\\Delta_{\\mathrm{new, old}} \\, \\log_{10}$ ion fraction'
    cax.set_ylabel(clabel, fontsize=fontsize)

    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
 
    ax = axes[4]
    
    if dohist: 
        nbins = 100
        maxv = np.max(np.abs(delta_oldnew))
        bins = np.linspace(-1. * maxv, maxv, nbins)

        ax.set_yscale('log')
        ax.hist(delta_oldnew, bins=bins, histtype='step', color='blue',
                label=None, linestyle='dashed', align='mid', 
                density=True)

        ax.set_xlabel(clabel, fontsize=fontsize)
        ax.set_ylabel('probability density', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 1)
    else:
        ax.text(0.5, 0.5, 'Old, new log maps are\nthe same to fp precision',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(left=False, bottom=False, labelleft=False, 
                       labelbottom=False, which='both')
        
    
    outname = imgdir + imgname_oldnew.format(sim=simlabel, dep=ps20depletion,
                                             z=redshift, met=np.log10(target_Z),
                                             ion=ion)
    plt.savefig(outname, bbox_inches='tight')

    # new log vs. new lin
    title = 'interpolated {ion} PS20 tables at z={z:.2f}, '+ \
            'Z={met:.1e} from FIRE data, dust depl. {dep}' 
    title = title.format(ion=ion, dep=ps20depletion, 
                         z=redshift, met=target_Z)
    
    fig = plt.figure(figsize=(13., 4.))
    grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.5, 
                        width_ratios=[1., 0.1, 1., 0.1, 1.],
                        left=0.05, right=0.95)
    axes = [fig.add_subplot(grid[0, i]) for i in range(5)]
    fontsize = 12
    cmap = 'viridis'
    size = 10.
    
    vmax = 0.
    vmin = -10.

    fig.suptitle(title, fontsize=fontsize)

    ax = axes[0]
    cax = axes[1]
    img = ax.scatter(lognHsim, logTsim, s=size, c=np.log10(ionbal_new_lin),
                     edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax,
                     rasterized=True)
    plt.colorbar(img, cax=cax)
    cax.set_ylabel('$\\log_{10}$ ion fraction', fontsize=fontsize)

    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
    ax.text(0.05, 0.05, 'New table interp.,\nlin space',
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    ax = axes[2]
    cax = axes[3]
    delta_linlog = np.log10(ionbal_new_lin) - np.log10(ionbal_new_log)
    if np.all(ionbal_new_log == ionbal_new_lin):
        print('New log, lin maps are the same to fp precision')
        vmax = 1e-49
        dohist = False
    else:
        vmax = np.max(np.abs(delta_linlog[np.isfinite(delta_linlog)]))
        extend = 'neither'
        if vmax > 1.:
            vmax = 1.
            extend = 'both'
        vmin = -1. * vmax
        dohist = True
    img = ax.scatter(lognHsim, logTsim, s=size, 
                     c=delta_linlog,
                     edgecolor='black', cmap='RdBu', vmin=vmin, vmax=vmax,
                     rasterized=True)
    plt.colorbar(img, cax=cax, extend=extend)
    clabel = '$\\Delta_{\\mathrm{lin, log}} \\, \\log_{10}$ ion fraction'
    cax.set_ylabel(clabel, fontsize=fontsize)

    ax.set_ylabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                    fontsize=fontsize)
    xlabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                ' [\\mathrm{cm}^{-3}]$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
    cax.tick_params(labelsize=fontsize - 1, labelright=False, labelleft=True,
                    right=False, left=True)
    cax.yaxis.set_label_position('left')
 
    ax = axes[4]
    
    if dohist:
        nbins = 100
        maxv = np.max(np.abs(delta_linlog[np.isfinite(delta_linlog)]))
        bins = np.linspace(-1. * maxv, maxv, nbins)

        ax.set_yscale('log')
        ax.hist(delta_linlog, bins=bins, histtype='step', color='blue',
                label=None, linestyle='dashed', align='mid', 
                density=True)

        ax.set_xlabel(clabel, fontsize=fontsize)
        ax.set_ylabel('probability density', fontsize=fontsize)
        ax.legend(fontsize=fontsize - 1)
    else:
        ax.text(0.5, 0.5, 'New lin, log maps are\nthe same to fp precision',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(left=False, bottom=False, labelleft=False, 
                       labelbottom=False, which='both')

    outname = imgdir + imgname_linlog.format(sim=simlabel, dep=ps20depletion,
                                             z=redshift, met=np.log10(target_Z),
                                             ion=ion)
    plt.savefig(outname, bbox_inches='tight')
    
def run_test_lin_option_ionbal(index):
     # laptop
    ddir = '/Users/nastasha/ciera/tests/fire_start/ionbal_tests/'
    simname = 'm13h206_m3e5__m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1'+\
              '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    pre = 'ionbal_test_PS20'
    ftmp = ['{pre}_{ion}_depletion-False_Z-0.01_snap045{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-False_Z-0.0001_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-False_Z-0.01_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.0001_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.01_snap027{lt}_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.01_snap045{lt}_{simname}.hdf5',
            ]
    ions = ['O{}'.format(i) for i in range(1, 10)]
    lts = ['', '_lintable-False', '_lintable-True']

    ioni = index % len(ions)
    vari = index // len(ions)
    
    filen_old, filen_new_log, filen_new_lin = \
        (ddir + ftmp[vari].format(pre=pre, simname=simname, 
                                  ion=ions[ioni], lt=lt) \
         for lt in lts)
    test_lin_option_ionbal(filen_old, filen_new_log, filen_new_lin)


def test_tablesum_interpolate_to_tabulated(ztargets = [0., 1., 3.], 
                                           element='oxygen'):
    '''
    Ok, can it get the right values when interpolating to the
    listed values then.
    '''
    fn = '/Users/nastasha/phd/tables/ionbal/lines_sp20/'+\
         'UVB_dust1_CR1_G1_shield1.hdf5'
    outdir = '/Users/nastasha/ciera/tests/fire_start/ionbal_tests/'
    savename = outdir + 'sumtest-nodepl_z-{z:.3f}_{elt}_tables-direct.pdf'

    with h5py.File(fn, 'r') as f:
        mgrp = f['Tdep/IonFractions']
        eltkeys = list(mgrp.keys())
        eltgrp = [key if element.lower() in key else None\
                  for key in eltkeys]
        eltgrp = list(set(eltgrp))
        eltgrp.remove(None)
        eltgrp = mgrp[eltgrp[0]]
        # redshift, T, Z, nH, ion
        ztab = f['TableBins/RedshiftBins'][:]
        logT = f['TableBins/TemperatureBins'][:]
        met = f['TableBins/MetallicityBins'][:]
        lognH = f['TableBins/DensityBins'][:]
        nHcut = 1.0
        nHcuti = np.min(np.where(lognH >= nHcut)[0])
        fontsize = 12
        nHedges = lognH[:-1] - 0.5 * np.diff(lognH)
        nHend = [lognH[-1] - 0.5 * (lognH[-1] - lognH[-2]),
                 lognH[-1] + 0.5 * (lognH[-1] - lognH[-2])
                 ]
        nHedges = np.append(nHedges, nHend)
        Tedges = logT[:-1] - 0.5 * np.diff(logT)
        Tend = [logT[-1] - 0.5 * (logT[-1] - logT[-2]),
                logT[-1] + 0.5 * (logT[-1] - logT[-2])
                ]
        Tedges = np.append(Tedges, Tend)
        
        for ztar in ztargets:
            Zstart = 0 if element in ['hydrogen', 'helium'] else 1

            ncols = 4
            nz = len(met) - Zstart
            nrows = ((nz - 1) // ncols + 1) * 2
            width = 11.
            width_ratios = [1.] * ncols + [0.1]
            wspace = 0.3
            panelwidth = width / (sum(width_ratios) + ncols * wspace)
            hspace = 0.4
            height = (hspace * (nrows - 1.) + nrows) * panelwidth 

            fig = plt.figure(figsize=(width, height))
            grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=hspace,
                                wspace=wspace, 
                                width_ratios=width_ratios,
                                top=0.95, bottom=0.05)
            cax = fig.add_subplot(grid[:2, ncols])
            pdaxes = [fig.add_subplot(grid[2 * (i // 4), i % 4]) \
                      for i in range(nz)]
            haxes =  [fig.add_subplot(grid[2 * (i // 4) + 1, i % 4]) \
                       for i in range(nz)]

            zi = np.argmin(np.abs(ztab - ztar))
            zval = ztab[zi]
            _savename = savename.format(z=zval, elt=element)
            title = 'Sum of {elt} ions at z={z:.3f}'
            fig.suptitle(title.format(elt=element, z=zval), fontsize=fontsize)
            # T, Z, nH
            tabsum = np.sum(10**eltgrp[zi, :, Zstart:, :, :], axis=3)
            vmax = np.max(np.abs(tabsum - 1.))
            vmin = 1. - vmax
            vmax = 1. + vmax
            cmap = 'RdBu'
            #hbins = np.linspace(vmin, vmax, 100)

            nHlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                      '[\\mathrm{cm}^{-3}]$'
            Tlabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
            flabel = 'sum of ion fractions'

            # first value is Z = 0.0 -> no metal ions 
            for _Zi, _Z in enumerate(met[Zstart:]):
                label = '$ \\log_{{10}}Z/Z_{{\\odot}}$\n={:.1f}'.format(_Z)
                pdax = pdaxes[_Zi]
                hax = haxes[_Zi]
                pdax.text(0.05, 0.95, label, fontsize=fontsize - 2, 
                          horizontalalignment='left', verticalalignment='top',
                          transform=pdax.transAxes)
                hax.text(0.05, 0.95, label, fontsize=fontsize - 2, 
                          horizontalalignment='left', verticalalignment='top',
                          transform=hax.transAxes)
                pdax.set_xlabel(nHlabel, fontsize=fontsize)
                hax.set_xlabel(flabel, fontsize=fontsize)
                if _Zi % ncols == 0: 
                    pdax.set_ylabel(Tlabel, fontsize=fontsize)
                    hax.set_ylabel('probability density', fontsize=fontsize)

                img = pdax.pcolormesh(nHedges, Tedges, 
                                      tabsum[:, _Zi, :], 
                                      vmin=vmin, vmax=vmax, cmap=cmap)
                pdax.axvline(nHcut, color='black', linestyle='dashed')

                nHhi = (tabsum[:, _Zi, nHcuti:]).flatten()
                nHlo = (tabsum[:, _Zi, :nHcuti]).flatten()
                
                hax.set_yscale('log')
                dmax = np.max(np.abs(nHhi - 1.))
                dmax = max(dmax, np.max(np.abs(nHlo - 1.)))
                hmin = 1. - dmax
                hmax = 1 + dmax
                hbins = np.linspace(hmin, hmax, 100)
                
                histlo, _ = np.histogram(nHlo, bins=hbins, density=True)
                histhi, _ = np.histogram(nHhi, bins=hbins, density=True)
                hax.set_yscale('log')
                hax.step(hbins[:-1], histlo, where='post', 
                         label='log nH >= {:.1f}'.format(nHcut),
                         color='blue')
                hax.step(hbins[:-1], histhi, where='post', 
                         label='log nH >= {:.1f}'.format(nHcut),
                         color='black', linestyle='dashed')

                #hv1, _, _ = hax.hist(nHlo, bins=hbins, density=True, 
                #                     label='log nH >= {:.1f}'.format(nHcut),
                #                     color='blue', align='mid', 
                #                     histtype='step')
                #hv2, _, _ = hax.hist(nHhi, bins=hbins, density=True, 
                #                     label='log nH < {:.1f}'.format(nHcut),
                #                     color='black', linestyle='dashed', 
                #                     align='mid', histtype='step')
                ymin = min(np.min(histlo[histlo > 0.]), 
                           np.min(histhi[histhi > 0.]))
                _ymin, _ymax = hax.get_ylim()
                hax.set_ylim((0.7 * ymin, _ymax))

                if _Zi == 0:
                    hax.legend(fontsize=fontsize - 2)
            
            plt.colorbar(img, cax=cax)
            cax.set_ylabel(flabel, fontsize=fontsize)
            plt.savefig(_savename, bbox_inches='tight')


def plot_radprof_m12i_CR_comp(smallrange=True):
    '''
    Rough comparison to Ji, Chan, et al. (2020)
    '''
    if smallrange:
        rbins = np.linspace(0., 0.8, 16)
        yranges = {'si4': (10.8, 15.1), 'n5': (10.9, 14.7), 
                   'o6': (13., 15.3), 'ne8': (12.8, 15.1)}
        rcens = 0.5 * (rbins[:-1] + rbins[1:])
    else:
        rbins = np.append([-3.5], np.linspace(-1., 0.3, 14))
        rbins = 10**rbins
        rcens = 0.5 * (rbins[:-1] + rbins[1:])
        yranges = {'si4': (6., 15.1), 'n5': (6., 14.7), 
                   'o6': (11., 15.3), 'ne8': (10.4, 14.7)}
        
    axions = {'si4': 0, 'n5': 1, 'o6': 2, 'ne8': 3}
    snapcolors = {277: 'green', 600: 'black'}
    snaplabels = {277: 'z=1', 600: 'z=0'}
    ions = list(axions.keys())
    snapshots = list(snaplabels.keys())
    fdir = '/Users/nastasha/ciera/tests/fire_start/map_tests/'
    filens = ['coldens_n5_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap277_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_n5_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap600_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_ne8_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap277_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_ne8_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap600_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_o6_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap277_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_o6_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap600_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_si4_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap277_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              'coldens_si4_m12i_noAGNfb_CR-diff-coeff-690_FIRE-2_snap600_shrink-sph-cen_BN98_2rvir_v1.hdf5',
              ]
    filens = [fdir + filen for filen in filens]

    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(nrows=2, ncols=2, hspace=0.3, wspace=0.5)
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    fontsize = 12
    
    title = 'm12i with CRs, linear average and full\nrange column densities, z-projection'
    fig.suptitle(title, fontsize=fontsize)

    for filen in filens:   
        rd, cd = get_rval_massmap(filen, units='Rvir')
        rinds = np.searchsorted(rbins, rd) - 1
        cd_by_bin = [cd[rinds == i] for i in range(len(rbins))]
        cd_av = np.log10(np.array([np.average(10**cds) for cds in cd_by_bin]))
        cd_range = np.array([[np.min(cds), np.max(cds)] for cds in cd_by_bin])
        cd_range[cd_range == -np.inf] = -100.

        ion = ions[np.where(['coldens_{}'.format(ion) in filen \
                   for ion in ions])[0][0]] 
        snapnum = snapshots[np.where(['snap{}'.format(snap) in filen \
                             for snap in snapshots])[0][0]] 
        print(ion, snapnum)
        ax = axes[axions[ion]]
        snaplabel = snaplabels[snapnum]
        color = snapcolors[snapnum]

        ax.plot(rcens, cd_av[:-1], color=color, label=snaplabel)
        ax.fill_between(rcens, cd_range[:-1, 0], cd_range[:-1, 1], color=color,
                        alpha=0.3)
    for ion in ions:
        print(ion)
        print(axions[ion])
        ax = axes[axions[ion]]
        ax.set_xlabel('$r_{\perp} \\; [\\mathrm{R}_{\\mathrm{vir}}]$', 
                      fontsize=fontsize)
        ax.set_ylabel('$\\log_{10} \\, \\mathrm{N} \\; [\\mathrm{cm}^{-2}]$',
                      fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 1, direction='in', which='both')
        ax.set_ylim(yranges[ion])
        if not smallrange:
            ax.set_xscale('log')
        ax.text(0.05, 0.05, ion, fontsize=fontsize,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

    axes[axions[ions[0]]].legend(fontsize=fontsize)
    outname = fdir + 'radprof_coldens_ji-chan-etal-2020_comp_m12i' + \
                     '_noAGNfb_CR-diff-coeff-690_FIRE-2'
    outname = outname + '_smallrad.pdf' if smallrange else \
              outname + '_largerad.pdf'
    plt.savefig(outname, bbox_inches='tight')


def test_ionsum_and_Z_maps():
    
    fdir = '/Users/nastasha/ciera/tests/fire_start/map_tests/'
    ionb = 'coldens_{ion}_m13h206_m3e5__m13h206_m3e5_MHDCRspec1_fire3' + \
           '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5' +\
           '_fcr1e-3_vw3000_snap27_shrink-sph-cen_BN98_2rvir_v1.hdf5'
    ionfiles = [fdir + ionb.format(ion='O{}'.format(i)) for i in range(1, 10)]
    eltfile = fdir + ionb.format(ion='Oxygen')
    massfile = fdir + ionb.format(ion='gas-mass')
    fdir_h = '/Users/nastasha/ciera/tests/fire_start/hist_tests/'
    histZfile = fdir_h + 'hist_Oxygen_by_Mass_0-1-2Rvir_m13h206_m3e5__' + \
                'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021' +\
                '_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap27_' + \
                'shrink-sph-cen_BN98_2rvir_v1.hdf5'
    
    outfilen = fdir + 'O-sum-and-Z-frac-test_m13h206_m3e5_MHDCRspec1_fire3' +\
           '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5' +\
           '_fcr1e-3_vw3000_snap27_shrink-sph-cen_BN98_2rvir_v1.pdf'

    ionmaps = []
    ion_mapext = []
    eltmass = c.atomw_O * c.u
    
    width_ratios = [1.] * 4 + [0.2]
    fig = plt.figure(figsize=(11., 11.))
    grid = gsp.GridSpec(nrows=4, ncols=5, hspace=0.05, wspace=0.05,
                        width_ratios=width_ratios)
    coordsax = fig.add_subplot(grid[:4, :4])
    ionaxes = [fig.add_subplot(grid[i // 4, i % 4]) for i in range(9)]
    ionsumax = fig.add_subplot(grid[2, 1])
    elttotax = fig.add_subplot(grid[2, 2])
    deltaionsumax = fig.add_subplot(grid[2, 3])
    massax = fig.add_subplot(grid[3, 0])
    metax = fig.add_subplot(grid[3, 1])
    
    # slightly smaller panel to add axes labels and stuff
    _histax = fig.add_subplot(grid[3, 2])
    _histax.set_axis_off()
    _histax.tick_params(which='both', labelleft=False, labelbottom=False,
                        left=False, bottom=False)
    hpos = _histax.get_position()
    fmarx = 0.3
    fmary = 0.2
    histax = fig.add_axes([hpos.x0 + fmarx * hpos.width, 
                           hpos.y0 + fmary * hpos.height,
                           hpos.width * (1. - fmarx), 
                           hpos.height * (1. - fmary)])

    _dhistax = fig.add_subplot(grid[3, 3])
    _dhistax.set_axis_off()
    _dhistax.tick_params(which='both', labelleft=False, labelbottom=False,
                         left=False, bottom=False)
    hpos = _dhistax.get_position()
    fmarx = 0.3
    fmary = 0.2
    dhistax = fig.add_axes([hpos.x0 + fmarx * hpos.width, 
                            hpos.y0 + fmary * hpos.height,
                            hpos.width * (1. - fmarx), 
                            hpos.height * (1. - fmary)])
    

    fontsize = 12
    

    cmap_cd = 'afmhot'
    cmap_gas = 'viridis'
    cmap_Z = 'plasma'
    cmap_delta = 'RdBu'

    cax_i = fig.add_subplot(grid[0, 4])
    cax_Z = fig.add_subplot(grid[1, 4])
    cax_delta = fig.add_subplot(grid[2, 4])
    cax_gas = fig.add_subplot(grid[3, 4])

    coordsax.spines['right'].set_visible(False)
    coordsax.spines['top'].set_visible(False)
    coordsax.spines['left'].set_visible(False)
    coordsax.spines['bottom'].set_visible(False)
    coordsax.tick_params(which='both', labelbottom=False, labelleft=False,
                         left=False, bottom=False)
    coordsax.set_xlabel('X [pkpc]', fontsize=fontsize, labelpad=20.)
    coordsax.set_ylabel('Y [pkpc]', fontsize=fontsize, labelpad=40.)

    patheff_text = [mppe.Stroke(linewidth=2.0, foreground="white"),
                    mppe.Stroke(linewidth=0.4, foreground="black"),
                    mppe.Normal()]  

    vmin_i = np.inf
    vmax_i = -np.inf
    for ionf in ionfiles:
        with h5py.File(ionf, 'r') as f:
            #print(ionf)
            _map = f['map'][:]
            ionmaps.append(_map)
            vmin = f['map'].attrs['minfinite']
            vmax = f['map'].attrs['max']
            vmin_i = min(vmin, vmin_i)
            vmax_i = max(vmax, vmax_i)

            box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
            cosmopars = {key: val for key, val in \
                        f['Header/inputpars/cosmopars'].attrs.items()}
            #print(cosmopars)
            if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
                rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
                rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
            elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
                rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
            xax = f['Header/inputpars'].attrs['Axis1']
            yax = f['Header/inputpars'].attrs['Axis2']
            box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
            extent = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                      -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])
            ion_mapext.append(extent)

    with h5py.File(eltfile, 'r') as f:
        map_elt = f['map'][:]
        vmin = f['map'].attrs['minfinite']
        vmax = f['map'].attrs['max']
        vmin_i = min(vmin, vmin_i)
        vmax_i = max(vmax, vmax_i)

        box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
        cosmopars = {key: val for key, val in \
                    f['Header/inputpars/cosmopars'].attrs.items()}
        #print(cosmopars)
        if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
            rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
            rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
        elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
            rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
            rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
        xax = f['Header/inputpars'].attrs['Axis1']
        yax = f['Header/inputpars'].attrs['Axis2']
        box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
        extent_elt = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                      -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])
    
    with h5py.File(massfile, 'r') as f:
        map_mass = f['map'][:]
        vmin = f['map'].attrs['minfinite']
        vmax = f['map'].attrs['max']
        vmin_m = min(vmin, vmin_i)
        vmax_m = max(vmax, vmax_i)

        box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
        cosmopars = {key: val for key, val in \
                     f['Header/inputpars/cosmopars'].attrs.items()}
        #print(cosmopars)
        if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
            rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
            rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
        elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
            rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
            rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
        xax = f['Header/inputpars'].attrs['Axis1']
        yax = f['Header/inputpars'].attrs['Axis2']
        box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
        extent_mass = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                       -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])
        
    
    _vmin_i = max(vmin_i, vmax_i - 10.)
    extlow_i = 'neither' if _vmin_i >= vmin_i else 'min'
    vtrans = 12.5
    if vtrans > _vmin_i and vtrans < vmax_i:
        _cmap_cd = pu.paste_cmaps(['gist_yarg', cmap_cd], 
        [_vmin_i, vtrans, vmax_i])    
    else:
        _cmap_cd = cmap_cd
    patheff_circ = [mppe.Stroke(linewidth=2.0, foreground="white"),
                    mppe.Stroke(linewidth=1.5, foreground="black"),
                    mppe.Normal()]  

    isum = np.zeros(ionmaps[0].shape, dtype=ionmaps[0].dtype)
    for ii, (imap, iext) in enumerate(zip(ionmaps, ion_mapext)):
        ax = ionaxes[ii]
        ion = 'O{}'.format(ii + 1)
        cen = (0.5 * (iext[0] + iext[1]), 0.5 * (iext[2] + iext[3]))
        ynum = ii % 4 == 0
        ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=False,
                       labelleft=ynum, direction='out')

        img_i = ax.imshow(imap.T, origin='lower', interpolation='nearest',
                          extent=iext, vmin=_vmin_i, vmax=vmax_i, 
                          cmap=_cmap_cd)
        ax.text(0.05, 0.95, ion, fontsize=fontsize,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, color='blue', 
                path_effects=patheff_text)
        patches = [mpatch.Circle(cen, rvir_pkpc)]
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=['blue'], facecolor='none', linewidth=1.5,
                       linestyle='dashed', path_effects=patheff_circ)
        ax.add_collection(collection)
        if ii == 0:
            ax.text(1.05 * 2**-0.5 * rvir_pkpc, 1.05 * 2**-0.5 * rvir_pkpc, 
                    '$R_{\\mathrm{vir}}$',
                    color='blue', fontsize=fontsize,
                    path_effects=patheff_text)
        isum += 10**imap
    
    plt.colorbar(img_i, cax=cax_i, extend=extlow_i, orientation='vertical')
    cax_i.set_ylabel('$\\log_{10} \\, \\mathrm{N} \\; [\\mathrm{cm}^{-2}]$',
                     fontsize=fontsize)

    isum = np.log10(isum)
    ax = ionsumax
    ion = 'ion sum'
    cen = (0.5 * (iext[0] + iext[1]), 0.5 * (iext[2] + iext[3]))
    ynum = False
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=False,
                   labelleft=ynum, direction='out')

    ax.imshow(isum.T, origin='lower', interpolation='nearest',
              extent=iext, vmin=_vmin_i, vmax=vmax_i, 
              cmap=_cmap_cd)
    ax.text(0.05, 0.95, ion, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='blue', 
            path_effects=patheff_text)
    patches = [mpatch.Circle(cen, rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['blue'], facecolor='none', linewidth=1.5,
                    linestyle='dashed', path_effects=patheff_circ)
    ax.add_collection(collection)
    
    ax = elttotax
    ion = 'all O'
    cen = (0.5 * (iext[0] + iext[1]), 0.5 * (iext[2] + iext[3]))
    ynum = False
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=False,
                   labelleft=ynum, direction='out')
    #print(map_elt)
    ax.imshow(map_elt.T, origin='lower', interpolation='nearest',
              extent=extent_elt, vmin=_vmin_i, vmax=vmax_i, 
              cmap=_cmap_cd)
    ax.text(0.05, 0.95, ion, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='blue', 
            path_effects=patheff_text)
    patches = [mpatch.Circle(cen, rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['blue'], facecolor='none', linewidth=1.5,
                    linestyle='dashed', path_effects=patheff_circ)
    ax.add_collection(collection)
     
    ax = deltaionsumax
    _map = isum - map_elt
    maxd = np.max(np.abs(_map[np.isfinite(_map)]))
    if np.any(np.abs(_map) > maxd):
        extend = 'both'
    else:
        extend = 'neither'
    ion = 'ion sum - all O'
    cen = (0.5 * (iext[0] + iext[1]), 0.5 * (iext[2] + iext[3]))
    ynum = False
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=False,
                   labelleft=ynum, direction='out')
    img_delta = ax.imshow(_map.T, origin='lower', interpolation='nearest',
                          extent=extent_elt, vmin=-maxd, vmax=maxd, 
                          cmap=cmap_delta)
    ax.text(0.05, 0.95, ion, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black', 
            path_effects=patheff_text)
    patches = [mpatch.Circle(cen, rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['black'], facecolor='none', linewidth=1.5,
                    linestyle='dashed', path_effects=patheff_circ)
    ax.add_collection(collection)
    
    plt.colorbar(img_delta, cax=cax_delta, extend=extend, orientation='vertical')
    cax_delta.set_ylabel('$\\Delta \\, \\log_{10} \\, \\mathrm{N}$',
                         fontsize=fontsize)

    ax = dhistax
    ax.set_xlabel('$\\Delta \\, \\log_{10} \\, \\mathrm{N}$',
                  fontsize=fontsize)
    ax.set_ylabel('# pixels', fontsize=fontsize)
    ax.hist(_map.flatten(), bins=100, log=True, histtype='stepfilled',
            color='blue')
    ax.text(0.05, 0.95, 'ion sum - all O', fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black', 
            path_effects=patheff_text)
    ax.set_xlim(-1.05 * maxd, 1.05 * maxd)
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=True,
                   labelleft=True, direction='in')
    ax.tick_params(axis='x', which='both', rotation=45.)
    
    ax = massax
    _map = map_mass
    extend = 'neither'
    ion = 'gas'
    cen = (0.5 * (extent_mass[0] + extent_mass[1]), 
           0.5 * (extent_mass[2] + extent_mass[3]))
    ynum = True
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=True,
                   labelleft=ynum, direction='out')
    img_mass = ax.imshow(_map.T, origin='lower', interpolation='nearest',
                          extent=extent_mass, cmap=cmap_gas)
    ax.text(0.05, 0.95, ion, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='red', 
            path_effects=patheff_text)
    patches = [mpatch.Circle(cen, rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5,
                    linestyle='dashed', path_effects=patheff_circ)
    ax.add_collection(collection)

    plt.colorbar(img_mass, cax=cax_gas, extend=extend, orientation='vertical')
    cax_gas.set_ylabel('$\\log_{10} \\, \\Sigma \\; ' + \
                         '[\\mathrm{g}\\,\\mathrm{cm}^{-2}]$',
                         fontsize=fontsize)
    
    ax = metax
    _map = map_elt + np.log10(eltmass) - map_mass
    extend = 'neither'
    ion = 'O mass frac.'
    cen = (0.5 * (extent_mass[0] + extent_mass[1]), 
           0.5 * (extent_mass[2] + extent_mass[3]))
    ynum = False
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=True,
                   labelleft=ynum, direction='out')
    img_Z = ax.imshow(_map.T, origin='lower', interpolation='nearest',
                      extent=extent_mass, cmap=cmap_Z)
    ax.text(0.05, 0.95, ion, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='red', 
            path_effects=patheff_text)
    patches = [mpatch.Circle(cen, rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5,
                    linestyle='dashed', path_effects=patheff_circ)
    ax.add_collection(collection)

    plt.colorbar(img_Z, cax=cax_Z, extend=extend, orientation='vertical')
    cax_Z.set_ylabel('$\\log_{10} \\, \\mathrm{Z}$', fontsize=fontsize)
        
    with h5py.File(histZfile, 'r') as f:
        hist = f['histogram/histogram'][:]
        hist -= np.log10(c.solar_mass)
        xvals = f['axis_1/bins'][:]
    
    ax = histax
    ynum = True
    ax.tick_params(axis='both', labelsize=fontsize-1, labelbottom=True,
                   labelleft=ynum, direction='in')
    label = 'O/M hist.'
    ax.text(0.05, 0.95, label, fontsize=fontsize,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black', 
            path_effects=patheff_text)
    xlabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{O}} \\,/\\,' +\
             '\\mathrm{M}_{\\mathrm{gas}}$'
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel('$\\log_{10} \\, \\mathrm{M} \\; [\\mathrm{M}_{\\odot}]$',
                  fontsize=fontsize)
    
    _hist = np.empty((hist.shape[0], hist.shape[1] + 1), dtype=hist.dtype)
    _hist[:, :-1] = hist
    _hist[:, -1] = -np.inf
    maxy = np.max(_hist)
    miny = np.min(_hist[np.isfinite(_hist)])
    _hist[_hist == -np.inf] = -100.

    ax.step(xvals, _hist[0, :], color='black', linewidth=2., where='post')
    ax.step(xvals, _hist[1, :], color='blue', linewidth=1.5, 
            linestyle='dashed', where='post')
    ax.text(0.05, 0.70, '$< \\mathrm{R}_{\\mathrm{vir}}$', 
            fontsize=fontsize - 2,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='black', 
            path_effects=patheff_text)
    ax.text(0.05, 0.80, '$1 \\endash 2 \\, \\mathrm{R}_{\\mathrm{vir}}$', 
            fontsize=fontsize - 2,
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, color='blue', 
            path_effects=patheff_text)
    ax.set_ylim(miny * 0.95, maxy * 1.1)
    
    plt.savefig(outfilen, bbox_inches='tight')

def readfile_outputtimes(path):
    if path.endswith('output'):
        path = path[:-6]
    if not path.endswith('/'):
        path = path + '/'
    targetfile = path + 'snapshot_scale-factors.txt'
    if not os.path.isfile(targetfile):
        raise RuntimeError('No file {} found'.format(targetfile))
    with open(targetfile, 'r') as f:
        aopts = f.read()
    aopts = (aopts.strip()).split('\n')
    aopts = np.array([float(aopt) for aopt in aopts])
    zopts = 1. / aopts - 1.
    return zopts
        
def plotsnaps_m13noBH():
    #basedir = '/scratch3/01799/phopkins/fire3_suite_done/'
    basedir = '/Users/nastasha/ciera/fire_data/'
    
    # noBH m13s from Lindsey's spreadsheet 
    checkpaths = ['m13h002_m3e5/m13h002_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h007_m3e5/m13h007_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h029_m3e5/m13h029_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h113_m3e5/m13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h206_m3e5/m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h206_m3e5/m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct142021_crdiffc1_sdp1e10_gacc31_fa0.5',
                  'm13h206_m3e5/m13h206_m3e5_MHD_fire3_fireBH_Sep052021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h206_m3e5/m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h206_m3e5/m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct142021_crdiffc1_sdp1e10_gacc31_fa0.5',
                  'm13h206_m3e5/m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e10_gacc31_fa0.5_fcr3e-3_vw3000',
                  'm13h223_m3e5/m13h223_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  'm13h236_m3e5/m13h236_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                  ]
    data = {}
    vals = []
    title = 'snapshot redshifts: m13, m3e5, fire3_fireBH, sdp1e10_gacc31_fa0.5'
    for path in checkpaths:
        val = readfile_outputtimes(basedir + path)
        key = path.split('/')[-1]
        # shorten
        key = key.replace('m13', '')
        key = key.replace('_m3e5', '')
        key = key.replace('_fire3_fireBH', '')
        key = key.replace('_sdp1e10_gacc31_fa0.5', '')
        if len(key) > 30:
            tot = len(key)
            opts = np.where([char == '_' for char in key])[0]
            splitopt = np.argmin(np.abs(opts - 0.5 * tot))
            splitpoint = opts[splitopt]
            key = key[:splitpoint] + '\n' + key[splitpoint:]
        data.update({key: val})
        vals.append(val)
    
    commonvals = [val if np.all([val in vals[i] for i in range(len(vals))])\
                  else None for val in vals[0]]
    normalvals = [val if np.sum([val in vals[i] for i in range(len(vals))]) \
                         > 0.5 * len(vals)\
                  else None for val in vals[3]]
    while None in commonvals:
        commonvals.remove(None)
    while None in normalvals:
        normalvals.remove(None)
    
    fontsize = 12
    fig = plt.figure(figsize=(10., 5))
    ax = fig.add_axes([0.33, 0.1, 0.62, 0.8])
    fig.suptitle(title, fontsize=fontsize)
    keys = list(data.keys())
    keys.sort()
    yvals = np.arange(len(keys)) + 0.5
    xzeroval = 0.01
    for key, yval in zip(keys, yvals):
        xvals = data[key]
        if xvals[0] <= 0.:
            xvals[0] = xzeroval
        xall = [val in normalvals for val in xvals]
        colors = np.ones((len(xvals), 4)) * np.array([0.0, 0.0, 0.0, 1.0])
        colors[np.logical_not(xall)] = np.array([0.0, 1.0, 0.0, 0.4])
        ax.scatter(xvals, [yval] * len(xvals), c=colors, marker='o',
                   s=15)
        oddvals = np.where(np.logical_not(xall))[0]
        for ind in oddvals:
            xv = xvals[ind]
            if ind > 0:
                if xv == xvals[ind - 1]:
                    st = st + '={}'.format(int)
                else:
                    st = '{}'.format(ind)
            else:   
                st = '{}'.format(ind)
            ax.text(xv, yval, st,
                    horizontalalignment='left' if ind % 4 > 1 else 'right',
                    verticalalignment='bottom' if ind % 2 else 'top')
        numsnaps = len(xvals)
        ax.text(11., yval, '({})'.format(numsnaps),
                horizontalalignment='left', verticalalignment='center')
    for xv in commonvals:
        ax.axvline(xv, color='gray', alpha=0.3)
    ax.set_yticks(yvals, labels=keys)
    ax.tick_params(left=True, bottom=True, labelsize=fontsize - 1,
                   direction='in', which='both')
    ax.set_xlabel('redshift', fontsize=fontsize)
    ax.text(xzeroval, yvals[0], '$z=0$', fontsize=fontsize - 1,
            horizontalalignment='left', verticalalignment='center')
    ax.set_xlim(0.8 * xzeroval, 17.)
    ax.set_xscale('log')

    plt.savefig('/Users/nastasha/ciera/tests/fire_start/' + \
                'snaptimes_m13_noBH_sample_Lindsey.pdf')

def plotcomp_mass_ion_BH_noBH(fnmass_noBH='', fnion_noBH='',
                              fnmass_BH='', fnion_BH='',
                              outname='', title=None):
    mapn = {'mass_noBH': fnmass_noBH,
            'ion_noBH': fnion_noBH,
            'mass_BH': fnmass_BH,
            'ion_BH': fnion_BH}
    mapkeys = ['mass_noBH', 'mass_BH',
               'ion_noBH', 'ion_BH']
    examplekey_BH = 'mass_BH'
    examplekey_noBH = 'mass_noBH'

    examplekey_ion = 'ion_noBH'
    examplekey_mass = 'mass_noBH'
    maps = {}
    vmins = {}
    vmaxs = {}
    extents = {}
    rvirs = {}
    mvirs = {}
    if title is None:
        filens = {}
        redshifts = {}
    for key in mapn:
        with h5py.File(mapn[key], 'r') as f:
            map = f['map'][:]
            vmin = f['map'].attrs['minfinite']
            vmax = f['map'].attrs['max']

            box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
            cosmopars = {key: val for key, val in \
                        f['Header/inputpars/cosmopars'].attrs.items()}
            #print(cosmopars)
            if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
                rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
                rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
            elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
                rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
            xax = f['Header/inputpars'].attrs['Axis1']
            yax = f['Header/inputpars'].attrs['Axis2']
            box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
            extent = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                      -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])
            maps[key] = map
            vmins[key] = vmin
            vmaxs[key] = vmax
            extents[key] = extent
            rvirs[key] = rvir_pkpc
            mvirs[key] = f['Header/inputpars/halodata'].attrs['Mvir_g']
            if title is None:
                filen = f['Header/inputpars'].attrs['snapfiles'][0].decode()
                filen = filen.split('/')
                if 'output' in filen:
                    filen = filen[-3]
                else:
                    filen = filen[-2]
                filens[key] = filen
                redshifts[key] = f['Header/inputpars/cosmopars'].attrs['z']
            if key == examplekey_ion:
                pathn = 'Header/inputpars/maptype_args_dict'
                ion_used = f[pathn].attrs['ion'].decode()

    vmin_mass = min([vmins[key] if 'mass' in key else np.inf for key in mapn])
    vmax_mass = max([vmaxs[key] if 'mass' in key else -np.inf for key in mapn])
    cmap_mass = 'viridis'
    mincol_ion = 13.
    vmin_ion = min([vmins[key] if 'ion' in key else np.inf for key in mapn])
    vmax_ion = max([vmaxs[key] if 'ion' in key else -np.inf for key in mapn])
    if vmin_ion < mincol_ion and vmax_ion > mincol_ion:
        cmap_ion = pu.paste_cmaps(['gist_yarg', 'plasma'], 
                                  [vmin_ion, mincol_ion, vmax_ion])
    else:
        cmap_ion = 'plasma'

    ncols = 2
    nrows = 2
    hspace = 0.2
    wspace = 0.35
    panelsize = 2.5
    if title is None:
        tspace = 1.2
    else:
        tspace = 0.5
    caxw = 0.12 * panelsize
    width_ratios = [panelsize] * ncols + [caxw]
    height_ratios = [panelsize] * nrows
    
    width = sum(width_ratios) * (1. + ncols / (ncols + 1.) * wspace)
    height = sum(height_ratios) * (1. + (nrows - 1)/ (nrows) * hspace) + tspace
    tfrac = 1. - tspace / height
    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1,
                        hspace=hspace, wspace=hspace, 
                        width_ratios=width_ratios,
                        height_ratios=height_ratios,
                        top=tfrac)
    axes = [fig.add_subplot(grid[i // 2, i%ncols]) for i in range(len(mapn))]
    cax_mass = fig.add_subplot(grid[0, 2])
    cax_ion = fig.add_subplot(grid[1, 2])
    fontsize = 12
    
    if title is None:
        if np.allclose(redshifts[examplekey_ion], 
                       [redshifts[key] for key in mapkeys]):
            title = ['redshift: {z:.3f}, ion: {ion}'.format(\
                     z=redshifts[examplekey_ion], ion=ion_used)]
        else:
            print('files have different redshifts: {}'.format(redshifts))
            title = []
        for key, pre in zip([examplekey_noBH, examplekey_BH],
                            ['left (no BH): ', 'right (BH): ']):
            filen = pre + filens[key]
            maxline = 70
            if len(filen) > maxline:
                splitopts = np.where([char == '_' for char in filen])[0]
                origlen = len(filen)
                filen = filen.split('_')
                rest = origlen
                splitlist=[]
                last = 0
                while rest > maxline:
                    if last == 0:
                        start = 0
                    else:
                        start = splitopts[last]
                    cur = last + sum(splitopts[last + 1:] \
                                     <= start + maxline)
                    if cur == last:
                        msg = 'file name splitting for title failed for {}'
                        raise RuntimeError(msg.format('_'.join(filen)))
                    splitlist.append(cur)
                    rest = rest - (splitopts[cur] - splitopts[last])
                    last = cur
                #print(splitopts)
                #print(splitlist)
                #print(origlen)
                for i in splitlist:
                    filen[i] = filen[i] + '\n'
                filen = '_'.join(filen)
            title.append(filen)
        title = '\n'.join(title)

    xlabel = ['X', 'Y', 'Z'][xax] + ' [pkpc]'
    ylabel = ['X', 'Y', 'Z'][yax] + ' [pkpc]'
    fig.suptitle(title, fontsize=fontsize)

    imgs = {}
    for axi, (ax, mapkey) in enumerate(zip(axes, mapkeys)):
        labelx = axi >= len(mapn) - ncols
        labely = axi % ncols == 0
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize-2)
        
        if 'ion' in mapkey:
            vmin = vmin_ion
            vmax = vmax_ion
            cmap = cmap_ion
            axtitle = 'ion, '
        elif 'mass' in mapkey:
            vmin = vmin_mass
            vmax = vmax_mass
            cmap = cmap_mass
            axtitle = 'gas, '
        if '_noBH' in mapkey:
            axtitle = axtitle + 'BH off'
        else:
            axtitle = axtitle + 'BH on'
        mvir_logmsun = np.log10(mvirs[mapkey] / c.solar_mass)
        masspart = '$\\mathrm{{M}}_\\mathrm{{vir}}=10^{{{logm_msun:.1f}}}'+\
                   '\\; \\mathrm{{M}}_{{\\odot}}$'
        masspart = masspart.format(logm_msun=mvir_logmsun)
        axtitle = masspart + '\n' + axtitle
        
        imgs[mapkey] = ax.imshow(maps[mapkey].T, origin='lower', 
                                 interpolation='nearest', vmin=vmin,
                                 vmax=vmax, cmap=cmap, extent=extent)

        patches = [mpatch.Circle((0., 0.), rvirs[mapkey])]
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5)
        ax.add_collection(collection)
        if axi == 0:
            ax.text(1.05 * 2**-0.5 * rvirs[mapkey], 
                    1.05 * 2**-0.5 * rvirs[mapkey], 
                    '$R_{\\mathrm{vir}}$',
                    color='red', fontsize=fontsize)
        ax.text(0.02, 0.98, axtitle, fontsize=fontsize - 1, color='red',
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)

    plt.colorbar(imgs[examplekey_ion], cax=cax_ion, 
                 extend='neither', orientation='vertical')
    cax_ion.set_ylabel('$\\log_{10} \\, \\mathrm{N} \\; [\\mathrm{cm}]^{-2}$',
                        fontsize=fontsize) 
    plt.colorbar(imgs[examplekey_mass], cax=cax_mass, 
                 extend='neither', orientation='vertical') 
    clabel_mass = '$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}' + \
                  ' \\; [\\mathrm{g} \\, \\mathrm{cm}]^{-2}$'
    cax_mass.set_ylabel(clabel_mass, fontsize=fontsize) 
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def plotcomps_mass_ion_BH_noBH(mapset=2):
    outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/maps_BH_noBH/'
    templatetype = 'all'
    if mapset == 2:
        mdir = '/Users/nastasha/ciera/sim_maps/fire/set2_BH_noBH/'
        template_m12i = 'coldens_{qt}_m12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_{bh}_gacc31_fa0.5_snap50_shrink-sph-cen_BN98_2rvir_v1.hdf5'
        template_m13h206 = 'coldens_{qt}_m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_{bh}_gacc31_fa0.5_snap50_shrink-sph-cen_BN98_2rvir_v1.hdf5' 
        bh_on = 'sdp1e-4'
        bh_off = 'sdp1e10'
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'N5', 'C2', 'Si2', 'Fe2', 'Mg2', 'Mg10']

        templates = {'m12i': template_m12i,
                     'm13h206': template_m13h206}
        labels_tpl = {'m12i': 'm12i_m6e4, MHDCRspec1, fireCR0, crdiffc690, gacc31_fa0.5',
                      'm13h206': 'm13h206_m3e5, MHDCRspec1, fireCR0, crdiffc690, gacc31_fa0.5'}
        title_tpl= 'Gas and {ion} columns, z=0.5, {template}'
        _outname = 'mapcomp_BH_noBH_set2_{template}_snap50_gas_{ion}.pdf'
        tempkeys = list(template.keys()) 
    elif mapset == 3:
        mdir = '/Users/nastasha/ciera/sim_maps/fire/set3_BH_noBH/'
        template_m12i = 'coldens_{{qt}}_m12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_{{bh}}_gacc31_fa0.5_snap{snap}_shrink-sph-cen_BN98_2rvir_v1.hdf5'
        template_m13h206 = 'coldens_{{qt}}_m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_{{bh}}_gacc31_fa0.5_snap{snap}_shrink-sph-cen_BN98_2rvir_v1.hdf5' 
        bh_on = 'sdp1e-4'
        bh_off = 'sdp1e10'
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        snaps = [45, 49, 51, 60]
        
        templates = {'m12i_snap{}'.format(snap): \
                     template_m12i.format(snap=snap) for snap in snaps}
        templates.update({'m13h206_snap{}'.format(snap): \
                          template_m13h206.format(snap=snap) \
                          for snap in snaps})
        keys_m12i = ['m12i_snap{}'.format(snap) for snap in snaps]
        keys_m13h206 = ['m13h206_snap{}'.format(snap) for snap in snaps]
        labels_tpl = {key: key + ', MHDCRspec1, fireCR0, crdiffc690, gacc31_fa0.5'\
                      for key in keys_m12i}
        labels_tpl.update({key: key + ', MHDCRspec1, fireCR0, crdiffc690, gacc31_fa0.5'\
                           for key in keys_m13h206})              
        title_tpl= 'Gas and {ion} columns, {template}'
        _outname = 'mapcomp_BH_noBH_set3_{template}_gas_{ion}.pdf'
        tempkeys = keys_m12i + keys_m13h206 
    elif mapset == 4:
        mdir = '/Users/nastasha/ciera/sim_maps/fire/set4_BH_noBH/'
        template = 'coldens_{{qt}}_{ic}_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_{{bh}}_gacc31_fa0.5_snap{snap}_shrink-sph-cen_BN98_2rvir_v2.hdf5'
        bh_on = 'sdp2e-4'
        bh_off = 'sdp1e10'
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        snaps = [500, 258, 186]
        ics = ['m12m', 'm12f']
        
        templates = {'{}_snap{}'.format(ic, snap): \
                     template.format(snap=snap, ic=ic) \
                     for snap in snaps for ic in ics}
        keys = ['{}_snap{}'.format(ic, snap) for snap in snaps for ic in ics]
        labels_tpl = {key: key + ', m7e3, Sep182021_hr_crdiffc690, gacc31_fa0.5'\
                      for key in keys}          
        title_tpl= 'Gas and {ion} columns, {template}'
        _outname = 'mapcomp_BH_noBH_set4_{template}_gas_{ion}.pdf'
        tempkeys = keys
    elif mapset == 5:
        mdir = '/Users/nastasha/ciera/sim_maps/fire/set5_BH_noBH/'
        template_noBH = 'coldens_{{qt}}_{ic}_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5_snap{snap}_shrink-sph-cen_BN98_2rvir_v2.hdf5'
        template_BH = 'coldens_{{qt}}_{ic}_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}_shrink-sph-cen_BN98_2rvir_v2.hdf5'
        bh_on = 'sdp1e-4'
        bh_off = 'sdp1e10'
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        snaps = [45, 50, 60]
        ics = ['m13h206', 'm13h007']
        
        templates_BH = {'{}_snap{}'.format(ic, snap): \
                        template_BH.format(snap=snap, ic=ic) \
                        for snap in snaps for ic in ics}
        templates_noBH = {'{}_snap{}'.format(ic, snap): \
                         template_noBH.format(snap=snap, ic=ic) \
                         for snap in snaps for ic in ics}
        keys = ['{}_snap{}'.format(ic, snap) for snap in snaps for ic in ics]   
        title_tpl = None
        templatetype = 'BHnoBH'
        _outname = 'mapcomp_BH_noBH_set5_{template}_gas_{ion}.pdf'
        tempkeys = keys

    elif mapset == 6:
        mdir = '/Users/nastasha/ciera/sim_maps/fire/set6_BH_noBH/'
        template_noBH = 'coldens_{{qt}}_{ic}_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5_snap{snap}_shrink-sph-cen_BN98_2rvir_v2.hdf5'
        template_BH = 'coldens_{{qt}}_{ic}_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}_shrink-sph-cen_BN98_2rvir_v2.hdf5'
        bh_on = 'sdp1e-4'
        bh_off = 'sdp1e10'
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        snaps = [45, 50]
        ics = ['m13h002']
        
        templates_BH = {'{}_snap{}'.format(ic, snap): \
                        template_BH.format(snap=snap, ic=ic) \
                        for snap in snaps for ic in ics}
        templates_noBH = {'{}_snap{}'.format(ic, snap): \
                         template_noBH.format(snap=snap, ic=ic) \
                         for snap in snaps for ic in ics}
        keys = ['{}_snap{}'.format(ic, snap) for snap in snaps for ic in ics]        
        title_tpl = None
        templatetype = 'BHnoBH'
        _outname = 'mapcomp_BH_noBH_set5_{template}_gas_{ion}.pdf'
        tempkeys = keys

    for templatekey in tempkeys:
        for ion in qts_ion:
            if templatetype == 'all':
                filetemplate = mdir + templates[templatekey]
                fnmass_noBH = filetemplate.format(qt=qt_mass, bh=bh_off)
                fnion_noBH = filetemplate.format(qt=ion, bh=bh_off)
                fnmass_BH = filetemplate.format(qt=qt_mass, bh=bh_on)
                fnion_BH = filetemplate.format(qt=ion, bh=bh_on)
            elif templatetype == 'BHnoBH':
                filetemplate_BH = mdir + templates_BH[templatekey]
                filetemplate_noBH = mdir + templates_noBH[templatekey]
                fnmass_noBH = filetemplate_noBH.format(qt=qt_mass)
                fnion_noBH = filetemplate_noBH.format(qt=ion)
                fnmass_BH = filetemplate_BH.format(qt=qt_mass)
                fnion_BH = filetemplate_BH.format(qt=ion)
            
            outname = outdir + _outname.format(template=templatekey, ion=ion)

            if title_tpl is None:
                title = None
            else:
                title = title_tpl.format(ion=ion, 
                                         template=labels_tpl[templatekey])
                if len(title) > 50:
                    splitopts = np.where([char == ',' for char in title])[0]
                    optchoice = np.argmin(np.abs(splitopts - 0.5 * len(title)))
                    splitind = splitopts[optchoice]
                    title = (title[:splitind + 1]).strip() + '\n' +\
                            (title[splitind + 1:]).strip()
            #print(fnmass_noBH)
            #print(fnion_noBH)
            #print(fnmass_BH)
            #print(fnion_BH)
            #print(outname)
            #print(title)
            plotcomp_mass_ion_BH_noBH(fnmass_noBH=fnmass_noBH, 
                                      fnion_noBH=fnion_noBH,
                                      fnmass_BH=fnmass_BH, 
                                      fnion_BH=fnion_BH,
                                      outname=outname, 
                                      title=title)

def plotcomp_mass_ion_physmodels(fns_mass, fns_ion, model_labels,
                                 outname='', title=None):
    mapn = {'mass_{}'.format(lab): fns_mass[i] \
            for i, lab in enumerate(model_labels)}
    mapn.update({'ion_{}'.format(lab): fns_ion[i] \
                 for i, lab in enumerate(model_labels)})
    mapkeys = ['mass_{}'.format(lab) for lab in model_labels]
    mapkeys += ['ion_{}'.format(lab) for lab in model_labels]
    examplekeys_model = ['mass_{}'.format(lab) for lab in model_labels]
    examplekey_ion = 'ion_{}'.format(model_labels[0])
    examplekey_mass = 'mass_{}'.format(model_labels[0])
    maps = {}
    vmins = {}
    vmaxs = {}
    extents = {}
    rvirs = {}
    mvirs = {}
    if title is None:
        filens = {}
        redshifts = {}
    for key in mapn:
        with h5py.File(mapn[key], 'r') as f:
            map = f['map'][:]
            vmin = f['map'].attrs['minfinite']
            vmax = f['map'].attrs['max']

            box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
            cosmopars = {key: val for key, val in \
                        f['Header/inputpars/cosmopars'].attrs.items()}
            #print(cosmopars)
            if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
                rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
                rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
            elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
                rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
            xax = f['Header/inputpars'].attrs['Axis1']
            yax = f['Header/inputpars'].attrs['Axis2']
            box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
            extent = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                      -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])
            maps[key] = map
            vmins[key] = vmin
            vmaxs[key] = vmax
            extents[key] = extent
            rvirs[key] = rvir_pkpc
            mvirs[key] = f['Header/inputpars/halodata'].attrs['Mvir_g']
            if title is None:
                filen = f['Header/inputpars'].attrs['snapfiles'][0].decode()
                filen = filen.split('/')
                if 'output' in filen:
                    filen = filen[-3]
                else:
                    filen = filen[-2]
                filens[key] = filen
                redshifts[key] = f['Header/inputpars/cosmopars'].attrs['z']
            if key == examplekey_ion:
                pathn = 'Header/inputpars/maptype_args_dict'
                ion_used = f[pathn].attrs['ion'].decode()

    vmin_mass = min([vmins[key] if 'mass' in key else np.inf for key in mapn])
    vmax_mass = max([vmaxs[key] if 'mass' in key else -np.inf for key in mapn])
    cmap_mass = 'viridis'
    if ion_used == 'Ne8':
        mincol_ion = 13.8
    elif ion_used == 'O6':
        mincol_ion = 13.5
    elif ion_used == 'H1':
        mincol_ion = 13.6
    else: # just take a stab at it
        mincol_ion = 13.5
    vmin_ion = min([vmins[key] if 'ion' in key else np.inf for key in mapn])
    vmax_ion = max([vmaxs[key] if 'ion' in key else -np.inf for key in mapn])
    if vmin_ion < mincol_ion and vmax_ion > mincol_ion:
        cmap_ion = pu.paste_cmaps(['gist_yarg', 'plasma'], 
                                  [vmin_ion, mincol_ion, vmax_ion])
    else:
        cmap_ion = 'plasma'

    ncols = len(model_labels)
    nrows = 2
    hspace = 0.2
    wspace = 0.35
    panelsize = 2.5
    if title is None:
        tspace = 1.2
    else:
        tspace = 0.5
    caxw = 0.12 * panelsize
    width_ratios = [panelsize] * ncols + [caxw]
    height_ratios = [panelsize] * nrows
    
    width = sum(width_ratios) * (1. + ncols / (ncols + 1.) * wspace)
    height = sum(height_ratios) * (1. + (nrows - 1)/ (nrows) * hspace) + tspace
    tfrac = 1. - tspace / height
    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1,
                        hspace=hspace, wspace=hspace, 
                        width_ratios=width_ratios,
                        height_ratios=height_ratios,
                        top=tfrac)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) \
            for i in range(len(mapn))]
    cax_mass = fig.add_subplot(grid[0, ncols])
    cax_ion = fig.add_subplot(grid[1, ncols])
    fontsize = 12
    
    if title is None:
        if np.allclose(redshifts[examplekey_ion], 
                       [redshifts[key] for key in mapkeys]):
            title = ['redshift: {z:.3f}, ion: {ion}'.format(\
                     z=redshifts[examplekey_ion], ion=ion_used)]
        else:
            print('files have different redshifts: {}'.format(redshifts))
            title = []
        for key, pre in zip(examplekeys_model,
                            ['left to right (1): '] + \
                            ['({})'.format(i + 2) \
                             for i in range(len(examplekeys_model) - 1)]):
            filen = pre + filens[key]
            maxline = 30 * ncols
            if len(filen) > maxline:
                splitopts = np.where([char == '_' for char in filen])[0]
                origlen = len(filen)
                filen = filen.split('_')
                rest = origlen
                splitlist=[]
                last = 0
                while rest > maxline:
                    if last == 0:
                        start = 0
                    else:
                        start = splitopts[last]
                    cur = last + sum(splitopts[last + 1:] \
                                     <= start + maxline)
                    if cur == last:
                        msg = 'file name splitting for title failed for {}'
                        raise RuntimeError(msg.format('_'.join(filen)))
                    splitlist.append(cur)
                    rest = rest - (splitopts[cur] - splitopts[last])
                    last = cur
                #print(splitopts)
                #print(splitlist)
                #print(origlen)
                for i in splitlist:
                    filen[i] = filen[i] + '\n'
                filen = '_'.join(filen)
            title.append(filen)
        title = '\n'.join(title)

    xlabel = ['X', 'Y', 'Z'][xax] + ' [pkpc]'
    ylabel = ['X', 'Y', 'Z'][yax] + ' [pkpc]'
    fig.suptitle(title, fontsize=fontsize)

    imgs = {}
    for axi, (ax, mapkey) in enumerate(zip(axes, mapkeys)):
        labelx = axi >= len(mapn) - ncols
        labely = axi % ncols == 0
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize-2)
        
        if 'ion' in mapkey:
            vmin = vmin_ion
            vmax = vmax_ion
            cmap = cmap_ion
            axtitle = 'ion, '
        elif 'mass' in mapkey:
            vmin = vmin_mass
            vmax = vmax_mass
            cmap = cmap_mass
            axtitle = 'gas, '
        modeli = axi % len(model_labels)
        axtitle = axtitle + model_labels[modeli]
        mvir_logmsun = np.log10(mvirs[mapkey] / c.solar_mass)
        masspart = '$\\mathrm{{M}}_\\mathrm{{vir}}=10^{{{logm_msun:.1f}}}'+\
                   '\\; \\mathrm{{M}}_{{\\odot}}$'
        masspart = masspart.format(logm_msun=mvir_logmsun)
        axtitle = masspart + '\n' + axtitle
        
        imgs[mapkey] = ax.imshow(maps[mapkey].T, origin='lower', 
                                 interpolation='nearest', vmin=vmin,
                                 vmax=vmax, cmap=cmap, extent=extents[mapkey])

        patches = [mpatch.Circle((0., 0.), rvirs[mapkey])]
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5)
        ax.add_collection(collection)
        if axi == 0:
            ax.text(1.05 * 2**-0.5 * rvirs[mapkey], 
                    1.05 * 2**-0.5 * rvirs[mapkey], 
                    '$R_{\\mathrm{vir}}$',
                    color='red', fontsize=fontsize)
        ax.text(0.02, 0.98, axtitle, fontsize=fontsize - 1, color='red',
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes)

    plt.colorbar(imgs[examplekey_ion], cax=cax_ion, 
                 extend='neither', orientation='vertical')
    cax_ion.set_ylabel('$\\log_{10} \\, \\mathrm{N} \\; [\\mathrm{cm}]^{-2}$',
                        fontsize=fontsize) 
    plt.colorbar(imgs[examplekey_mass], cax=cax_mass, 
                 extend='neither', orientation='vertical') 
    clabel_mass = '$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}' + \
                  ' \\; [\\mathrm{g} \\, \\mathrm{cm}]^{-2}$'
    cax_mass.set_ylabel(clabel_mass, fontsize=fontsize) 
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def plotcomps_mass_ion_physmodel(mapset='clean_set1'):
    outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/maps_clean_set1/'
    if mapset == 'clean_set1':
        mdir = '/Users/nastasha/ciera/sim_maps/fire/clean_set1/'
        tpl_m12_noBH = 'coldens_{qt}_{ic}_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5'
        tpl_m12_AGN_CR = 'coldens_{qt}_{ic}_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap50_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5'
        tpl_m12_AGN_noCR = 'coldens_{qt}_{ic}_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5'
        tpl_m13_noBH = 'coldens_{qt}_{ic}_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5_snap50_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        tpl_m13_AGN_CR = 'coldens_{qt}_{ic}_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap50_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        tpl_m13_AGN_noCR_h113 = 'coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        tpl_m13_AGN_noCR_h206 = 'coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        model_labels = ['no BH', 'AGN, no CR', 'AGN + CR']
        ics_m12 = ['m12f'] # bug: 'm12m'
        ics_m13 = ['m13h206', 'm13h113']
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'Mg10', 'H1']
        h1_imstr = '_ionfrac-fromsim'
        
        fnss_mass_m12 = [[mdir + tpl_m12_noBH.format(ic=ic, qt=qt_mass, 
                                                     im=''),
                          mdir + tpl_m12_AGN_noCR.format(ic=ic, qt=qt_mass, 
                                                         im=''),
                          mdir + tpl_m12_AGN_CR.format(ic=ic, qt=qt_mass, 
                                                       im=''),
                          ] \
                         for ic in ics_m12 for ion in qts_ion]
        fnss_mass_m13 = [[mdir + tpl_m13_noBH.format(ic=ic, qt=qt_mass, 
                                                     im=''),
                          mdir + tpl_m13_AGN_noCR_h206.format(ic=ic, 
                                qt=qt_mass, im='')\
                            if 'h206' in ic else
                            mdir + tpl_m13_AGN_noCR_h113.format(ic=ic, 
                                qt=qt_mass, im=''),
                          mdir + tpl_m13_AGN_CR.format(ic=ic, qt=qt_mass, 
                                                       im=''),
                          ] \
                         for ic in ics_m13 for ion in qts_ion]
        fnss_ion_m12 = [[mdir + tpl_m12_noBH.format(ic=ic, qt=ion, 
                              im=h1_imstr if ion == 'H1' else ''),
                         mdir + tpl_m12_AGN_noCR.format(ic=ic, qt=ion, 
                              im=h1_imstr if ion == 'H1' else ''),
                         mdir + tpl_m12_AGN_CR.format(ic=ic, qt=ion, 
                              im=h1_imstr if ion == 'H1' else ''),
                         ]
                         for ic in ics_m12 for ion in qts_ion]
        fnss_ion_m13 = [[mdir + tpl_m13_noBH.format(ic=ic, qt=ion, 
                             im=h1_imstr if ion == 'H1' else ''),
                         mdir + tpl_m13_AGN_noCR_h206.format(ic=ic, 
                              qt=ion, im=h1_imstr if ion == 'H1' else '')\
                            if 'h206' in ic else
                            mdir + tpl_m13_AGN_noCR_h113.format(ic=ic, 
                                qt=ion, im=h1_imstr if ion == 'H1' else ''),
                         mdir + tpl_m13_AGN_CR.format(ic=ic, qt=ion, 
                             im=h1_imstr if ion == 'H1' else ''),
                         ] \
                         for ic in ics_m13 for ion in qts_ion]
        fnss_mass = fnss_mass_m12 + fnss_mass_m13
        fnss_ion = fnss_ion_m12 + fnss_ion_m13
        model_labelss = [model_labels] * len(fnss_mass)

        ttpl_m12 = ('noBH: m7e3, AGN-noCR: m7e3, sdp2e-4,'
                    ' AGN-CR: m6e4, sdp1e-4, fcr1e-3_vw3000') 
        ttpl_m13h113 = ('noBH: m3e5, AGN-noCR: m3e4, sdp1e-4,'
                        ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
        ttpl_m13h206 = ('noBH: m3e5, AGN-noCR: m3e4, sdp3e-4,'
                        ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
        title_tpl= 'Gas and {ion} columns, {ic} z=0.5, {ttpl}'
        titles = [title_tpl.format(ic=ic, ion=ion, 
                                   ttpl=ttpl_m12 if 'm12' in ic else 
                                        ttpl_m13h206 if 'm13h206' in ic else\
                                        ttpl_m13h113)\
                  for ic in ics_m12 + ics_m13 for ion in qts_ion]

        _outname = 'mapcomp_noBH_AGN_AGNCR_set1_{ic}_z0p5_gas_{ion}.pdf'
        outnames = [_outname.format(ic=ic, ion=ion) \
                    for ic in ics_m12 + ics_m13 for ion in qts_ion]
    
    if mapset == 'clean_set2':
        outdir = ('/Users/nastasha/ciera/projects_lead/fire3_ionabs/'
                  'maps_clean_set2/')
        mdir = '/Users/nastasha/ciera/sim_maps/fire/clean_set2/'
        tpl_m12_noBH =          ('coldens_{qt}_{ic}_m7e3_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp1e10'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m12_AGN_CR =        ('coldens_{qt}_{ic}_m6e4_MHDCRspec1_fire3'
                                 '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                                 '_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m12_AGN_noCR =      ('coldens_{qt}_{ic}_m7e3_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp2e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m13_noBH =          ('coldens_{qt}_{ic}_m3e5_MHD_fire3'
                                 '_fireBH_Sep182021_crdiffc690_sdp1e10'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_CR =        ('coldens_{qt}_{ic}_m3e5_MHDCRspec1_fire3'
                                 '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                                 '_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_noCR_h113 = ('coldens_{qt}_{ic}_m3e4_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp1e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_noCR_h206 = ('coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH'
                                 '_Sep182021_hr_crdiffc690_sdp3e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        model_labels = ['no BH', 'AGN, no CR', 'AGN + CR']
        ics_m12 = ['m12f'] # bug: 'm12m'
        ics_m13 = ['m13h206', 'm13h113']
        qt_mass = 'gas-mass'
        qts_ion = ['O6', 'Ne8', 'Mg10', 'H1']
        h1_imstr = '_ionfrac-fromsim'
        snaps_sr = [49, 48, 47, 46, 45]
        snaps_hr =  [240, 224, 210, 197, 186]
        redshifts = [0.6, 0.7, 0.8, 0.9, 1.0]
        
        fnss_mass_m12 = [[mdir + tpl_m12_noBH.format(ic=ic, qt=qt_mass, 
                                                     im='', snap=snhr),
                          mdir + tpl_m12_AGN_noCR.format(ic=ic, qt=qt_mass, 
                                                         im='', snap=snhr),
                          mdir + tpl_m12_AGN_CR.format(ic=ic, qt=qt_mass, 
                                                       im='', snap=snsr),
                          ] \
                         for ic in ics_m12 for ion in qts_ion \
                         for snhr, snsr in zip(snaps_hr, snaps_sr)]
        fnss_mass_m13 = [[mdir + tpl_m13_noBH.format(ic=ic, qt=qt_mass, 
                                                     im='', snap=snsr),
                          mdir + tpl_m13_AGN_noCR_h206.format(ic=ic, 
                                qt=qt_mass, im='', snap=snhr)\
                            if 'h206' in ic else
                            mdir + tpl_m13_AGN_noCR_h113.format(ic=ic, 
                                qt=qt_mass, im='', snap=snhr),
                          mdir + tpl_m13_AGN_CR.format(ic=ic, qt=qt_mass, 
                                                       im='', snap=snsr),
                          ] \
                         for ic in ics_m13 for ion in qts_ion\
                         for snhr, snsr in zip(snaps_hr, snaps_sr)]
        fnss_ion_m12 = [[mdir + tpl_m12_noBH.format(ic=ic, qt=ion, 
                              im=h1_imstr if ion == 'H1' else '',
                              snap=snhr),
                         mdir + tpl_m12_AGN_noCR.format(ic=ic, qt=ion, 
                              im=h1_imstr if ion == 'H1' else '',
                              snap=snhr),
                         mdir + tpl_m12_AGN_CR.format(ic=ic, qt=ion, 
                              im=h1_imstr if ion == 'H1' else '',
                              snap=snsr),
                         ]
                         for ic in ics_m12 for ion in qts_ion\
                         for snhr, snsr in zip(snaps_hr, snaps_sr)]
        fnss_ion_m13 = [[mdir + tpl_m13_noBH.format(ic=ic, qt=ion, 
                             im=h1_imstr if ion == 'H1' else '',
                             snap=snsr),
                         mdir + tpl_m13_AGN_noCR_h206.format(ic=ic, 
                              qt=ion, im=h1_imstr if ion == 'H1' else '',
                              snap=snhr)\
                            if 'h206' in ic else
                            mdir + tpl_m13_AGN_noCR_h113.format(ic=ic, 
                                qt=ion, im=h1_imstr if ion == 'H1' else '',
                                snap=snhr),
                         mdir + tpl_m13_AGN_CR.format(ic=ic, qt=ion, 
                             im=h1_imstr if ion == 'H1' else '',
                             snap=snsr),
                         ] \
                         for ic in ics_m13 for ion in qts_ion\
                         for snhr, snsr in zip(snaps_hr, snaps_sr)]
        fnss_mass = fnss_mass_m12 + fnss_mass_m13
        fnss_ion = fnss_ion_m12 + fnss_ion_m13
        model_labelss = [model_labels] * len(fnss_mass)

        ttpl_m12 = ('noBH: m7e3, AGN-noCR: m7e3, sdp2e-4,'
                    ' AGN-CR: m6e4, sdp1e-4, fcr1e-3_vw3000') 
        ttpl_m13h113 = ('noBH: m3e5, AGN-noCR: m3e4, sdp1e-4,'
                        ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
        ttpl_m13h206 = ('noBH: m3e5, AGN-noCR: m3e4, sdp3e-4,'
                        ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
        title_tpl= 'Gas and {ion} columns, {ic} z={z:.1f}, {ttpl}'
        titles = [title_tpl.format(ic=ic, ion=ion, z=z,
                                   ttpl=ttpl_m12 if 'm12' in ic else 
                                        ttpl_m13h206 if 'm13h206' in ic else\
                                        ttpl_m13h113)\
                  for ic in ics_m12 + ics_m13 for ion in qts_ion\
                  for z in redshifts]

        _outname = 'mapcomp_noBH_AGN_AGNCR_{ic}_z{z}_gas_{ion}.pdf'
        outnames = [_outname.format(ic=ic, ion=ion, 
                                    z=('{:.1f}'.format(z)).replace('.', 'p'))\
                    for ic in ics_m12 + ics_m13 for ion in qts_ion \
                    for z in redshifts]

    for fns_mass, fns_ion, model_labels, title, outname in \
            zip(fnss_mass, fnss_ion, model_labelss, titles, outnames):    
        outname = outdir + outname

        if title is not None:
            if len(title) > 70:
                splitopts = np.where([char == ',' for char in title])[0]
                optchoice = np.argmin(np.abs(splitopts - 0.5 * len(title)))
                splitind = splitopts[optchoice]
                title = (title[:splitind + 1]).strip() + '\n' +\
                        (title[splitind + 1:]).strip()
        plotcomp_mass_ion_physmodels(fns_mass, 
                                        fns_ion, 
                                        model_labels,
                                        outname=outname, 
                                        title=title)


def compare_profiles_BHnoBH(bhfiles, nobhfiles,
                            legendlab_bhfs, legendlab_nobhfs,
                            rbins_pkpc, title=None, 
                            ylabel=None, ax=None,
                            outname=None):
    fontsize = 12
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.))
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{r}_{\\perp}$ [pkpc]', fontsize=fontsize)

    ls_bh = 'solid'
    ls_nobh = 'dashed'
    lw_med = 1.
    lw_av = 2.5
    alpha_range = 0.3
    colors = tc.tol_cset('bright')

    rcens = 0.5 * (rbins_pkpc[1:] + rbins_pkpc[:-1])
    
    for fi, (bhfn, nobhfn, bhlab, nobhlab) in \
            enumerate(zip(bhfiles, nobhfiles, 
                          legendlab_bhfs, legendlab_nobhfs)):
        with h5py.File(bhfn, 'r') as f:  
            rvir_bh_pkpc = f['Header/inputpars/halodata'].attrs['Rvir_cm']
            rvir_bh_pkpc /= (c.cm_per_mpc * 1e-3)
        with h5py.File(nobhfn, 'r') as f:  
            rvir_nobh_pkpc = f['Header/inputpars/halodata'].attrs['Rvir_cm']
            rvir_nobh_pkpc /= (c.cm_per_mpc * 1e-3)
        
        bh_av, bh_med, bh_90, bh_10 = get_profile_massmap(
            bhfn, rbins_pkpc, rbin_units='pkpc', 
            profiles=['av-lin', 'perc-0.5', 'perc-0.9', 'perc-0.1'])
        
        nb_av, nb_med, nb_90, nb_10 = get_profile_massmap(
            nobhfn, rbins_pkpc, rbin_units='pkpc', 
            profiles=['av-lin', 'perc-0.5', 'perc-0.9', 'perc-0.1'])
        
        color_bh = colors[2 * fi]
        color_nobh = colors[2 * fi + 1]
        ax.plot(rcens, bh_av, color=color_bh, linestyle=ls_bh, 
                linewidth=lw_av, label=bhlab)
        ax.plot(rcens, nb_av, color=color_nobh, linestyle=ls_nobh, 
                linewidth=lw_av, label=nobhlab)
        ax.plot(rcens, bh_med, color=color_bh, linestyle=ls_bh, 
                linewidth=lw_med)
        ax.plot(rcens, nb_med, color=color_nobh, linestyle=ls_nobh, 
                linewidth=lw_med)
        ax.fill_between(rcens, bh_10, bh_90, color=color_bh, 
                        alpha=alpha_range, linestyle=ls_bh,
                        linewidth=0.5)
        ax.fill_between(rcens, nb_10, nb_90, color=color_nobh, 
                        alpha=alpha_range, linestyle=ls_nobh,
                        linewidth=0.5)
        # indicate Rvir on the curves
        yp_bh_av = pu.linterpsolve(rcens, bh_av, rvir_bh_pkpc)
        yp_bh_med = pu.linterpsolve(rcens, bh_med, rvir_bh_pkpc)
        ax.scatter([rvir_bh_pkpc] * 2, [yp_bh_av, yp_bh_med], marker='o',
                   c=color_bh, s=15)
        yp_nb_av = pu.linterpsolve(rcens, nb_av, rvir_nobh_pkpc)
        yp_nb_med = pu.linterpsolve(rcens, nb_med, rvir_nobh_pkpc)
        ax.scatter([rvir_nobh_pkpc] * 2, [yp_nb_av, yp_nb_med], marker='o',
                   c=color_nobh, s=15)

    handles2, labels = ax.get_legend_handles_labels()
    handles1 = [mlines.Line2D((), (), linewidth=lw_med, linestyle=ls_bh,
                              label='med., w/ BH', color='black'),
                mlines.Line2D((), (), linewidth=lw_med, linestyle=ls_nobh,
                              label='med., no BH', color='black'),
                mlines.Line2D((), (), linewidth=lw_av, linestyle=ls_bh,
                              label='mean, w/ BH', color='black'),
                mlines.Line2D((), (), linewidth=lw_av, linestyle=ls_nobh,
                              label='mean, no BH', color='black'),
                mpatch.Patch(label='perc. 10-90', linewidth=0.5, 
                             color='black', alpha=alpha_range),
                mlines.Line2D((), (), linestyle=None, marker='o',
                              label='$\\mathrm{R}_{\\mathrm{vir}}$', 
                              color='black', markersize=5)
                ]
    ax.legend(handles=handles1 + handles2, fontsize=fontsize)
    ax.set_xscale('log')

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')
        
def compare_profilesets(mapset, compmode):

    if mapset == 4:
        outdir = ('/Users/nastasha/ciera/projects_lead/'
                  'fire3_ionabs/profiles_BH_noBH/')
        qts = ['gas-mass', 'Mg10', 'Ne8', 'O6', 'N5', 'Mg2']
        snaps = [500, 258, 186]
        bhmodes = {'nobh': 'sdp1e10',
                   'bh': 'sdp2e-4'}
        ics = ['m12f', 'm12m']
        
        fdir = '/Users/nastasha/ciera/sim_maps/fire/set4_BH_noBH/'
        filetemp = ('coldens_{qt}_{ic}_m7e3_MHD_fire3_fireBH_Sep182021'
                    '_hr_crdiffc690_{bh}_gacc31_fa0.5_snap{snap}_'
                    'shrink-sph-cen_BN98_2rvir_v2.hdf5')
        
        if compmode == 'indiv':
            filesets_bh = [[fdir + filetemp.format(bh=bhmodes['bh'], 
                                                   snap=snap,
                                                   ic=ic, qt=qt)] \
                            for ic in ics for snap in snaps for qt in qts]
            filesets_nobh = [[fdir + filetemp.format(bh=bhmodes['nobh'], 
                                                     snap=snap,
                                                     ic=ic, qt=qt)] \
                            for ic in ics for snap in snaps for qt in qts]
            ttemp = ('set 4, {qt}, snapshot {snap}, {ic}_m7e3,\n'
                     'MHD_fire3_fireBH_Sep18202_hr_crdiffc690, gacc31_fa0.5'
                     )
            titles = [ttemp.format(snap=snap, ic=ic, qt=qt)\
                      for ic in ics for snap in snaps for qt in qts]
            lls_bh = [['BH']] * len(filesets_bh)
            lls_nobh = [['no BH']] * len(filesets_bh)
            rbinss = [np.linspace(0., 300., 50) if 'm12' in ic else \
                      np.linspace(0., 600., 50) if 'm13' in ic else \
                      None \
                      for ic in ics for snap in snaps for qt in qts]
            mlabel = ('$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}'
                      ' \\, [\\mathrm{g} \\, \\mathrm{cm}^2]$')
            ilabel = ('$\\log_{{10}} \\, \\mathrm{{N}}(\\mathrm{{{ion}}})'
                      ' \\, [\\mathrm{{cm}}^2]$')
            ylabels = [mlabel if 'mass' in qt else\
                       ilabel.format(ion=qt) \
                       for ic in ics for snap in snaps for qt in qts]
            outtemp = 'rprofcomp_set4_{ic}_{qt}_snap{snap}_BHnoBH.pdf'
            outnames = [outdir + outtemp.format(ic=ic, qt=qt, snap=snap) \
                       for ic in ics for snap in snaps for qt in qts]

    if mapset in [5, 6]:
        outdir = ('/Users/nastasha/ciera/projects_lead/'
                  'fire3_ionabs/profiles_BH_noBH/')
        qts = ['gas-mass', 'Mg10', 'Ne8', 'O6', 'N5', 'Mg2']
        if mapset == 5:
           snaps = [60, 50, 45]
           ics = ['m13h007', 'm13h206']
        else:
            snaps = [50, 45]
            ics = ['m13h002']        
        fdir = '/Users/nastasha/ciera/sim_maps/fire/set{}_BH_noBH/'
        fdir = fdir.format(mapset)
        filetemp_nobh = ('coldens_{qt}_{ic}_m3e5_MHD_fire3_fireBH_Sep182021'
                       '_crdiffc690_sdp1e10_gacc31_fa0.5_snap{snap}_shrink'
                       '-sph-cen_BN98_2rvir_v2.hdf5')
        filetemp_bh = ('coldens_{qt}_{ic}_m3e5_MHDCRspec1_fire3_fireBH'
                         '_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5'
                         '_fcr1e-3_vw3000_snap{snap}_shrink-sph-cen_BN98'
                         '_2rvir_v2.hdf5')
        
        if compmode == 'indiv':
            filesets_bh = [[fdir + filetemp_bh.format(snap=snap,
                                                      ic=ic, qt=qt)] \
                            for ic in ics for snap in snaps for qt in qts]
            filesets_nobh = [[fdir + filetemp_nobh.format(snap=snap,
                                                          ic=ic, qt=qt)] \
                            for ic in ics for snap in snaps for qt in qts]
            ttemp = ('set {set_}, {qt}, snapshot {snap}, {ic}_m3e5,\n'
                     'no BH: MHD_fire3_fireBH_Sep182021_crdiffc690,'
                     ' gacc31_fa0.5\n'
                     'BH: MHDCRspec1, fireCR1_Oct252021_crdiffc1_sdp1e-4,'
                     ' fcr1e-3_vw3000'
                     )
            titles = [ttemp.format(set_=mapset, snap=snap, ic=ic, qt=qt)\
                      for ic in ics for snap in snaps for qt in qts]
            lls_bh = [['BH']] * len(filesets_bh)
            lls_nobh = [['no BH']] * len(filesets_bh)
            rbinss = [np.linspace(0., 300., 50) if 'm12' in ic else \
                      np.linspace(0., 700., 50) if 'm13' in ic else \
                      None \
                      for ic in ics for snap in snaps for qt in qts]
            mlabel = ('$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}'
                      ' \\, [\\mathrm{g} \\, \\mathrm{cm}^2]$')
            ilabel = ('$\\log_{{10}} \\, \\mathrm{{N}}(\\mathrm{{{ion}}})'
                      ' \\, [\\mathrm{{cm}}^2]$')
            ylabels = [mlabel if 'mass' in qt else\
                       ilabel.format(ion=qt) \
                       for ic in ics for snap in snaps for qt in qts]
            outtemp = 'rprofcomp_set4_{ic}_{qt}_snap{snap}_BHnoBH.pdf'
            outnames = [outdir + outtemp.format(ic=ic, qt=qt, snap=snap) \
                        for ic in ics for snap in snaps for qt in qts]

    for fs_bh, fs_nobh, ll_bh, ll_nobh, \
            rbins_pkpc, title, ylabel, outname in \
            zip(filesets_bh, filesets_nobh, lls_bh, lls_nobh,
                rbinss, titles, ylabels, outnames):
        compare_profiles_BHnoBH(fs_bh, fs_nobh, ll_bh, ll_nobh,
                                rbins_pkpc, title=title, 
                                ylabel=ylabel, ax=None,
                                outname=outname)

def compare_profiles_physmodels(modelfilelists,
                                legendlablists,
                                physmodel_labels,
                                rbins_pkpc, title=None, 
                                ylabel=None, ax=None,
                                outname=None, domean=True,
                                doscatter=True, colors=None):
    '''
    modelfilelists, legendlablists: lists of lists, shapes should match
    inner index: physics model differences
    outer index: different versions with that physics model 
                 (e.g., different ics, redshifts)
    '''
    fontsize = 12
    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.))
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel('$\\mathrm{r}_{\\perp}$ [pkpc]', fontsize=fontsize)
    
    if len(modelfilelists[0]) <= 4:
        lss = ['solid', 'dashed', 'dashdot', 'dashdot']
    else:
        lss = ['solid'] * len(modelfilelists[0])
    if domean:
        lw_med = 1.
        lw_av = 2.5
    else:
        lw_med = 2.
        lw_av = 0.5
    alpha_range = 0.3
    if colors is None:
        colors = tc.tol_cset('bright')

    rcens = 0.5 * (rbins_pkpc[1:] + rbins_pkpc[:-1])
    #print(modelfilelists)
    for vi, (modelfiles, leglabels) in \
             enumerate(zip(modelfilelists, legendlablists)):
        #print(modelfiles)
        for pi, (modelf, ls, leglab) in \
                enumerate(zip(modelfiles, lss, leglabels)):
            #print(modelf)
            with h5py.File(modelf, 'r') as f:  
                rvir_bh_pkpc = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                rvir_bh_pkpc /= (c.cm_per_mpc * 1e-3)
        
            av, med, p90, p10 = get_profile_massmap(
                modelf, rbins_pkpc, rbin_units='pkpc', 
                profiles=['av-lin', 'perc-0.5', 'perc-0.9', 'perc-0.1'])
            
            color = colors[len(modelfiles) * vi + pi]
            if domean:
                ax.plot(rcens, av, color=color, linestyle=ls, 
                        linewidth=lw_av, label=leglab)
            ax.plot(rcens, med, color=color, linestyle=ls, 
                    linewidth=lw_med, 
                    label=None if domean else leglab)
            if doscatter:
                ax.fill_between(rcens, p10, p90, color=color, 
                                alpha=alpha_range, linestyle=ls,
                                linewidth=0.5)
            # indicate Rvir on the curves
            yp_bh_med = pu.linterpsolve(rcens, med, rvir_bh_pkpc)
            if domean:
                yp_bh_av = pu.linterpsolve(rcens, av, rvir_bh_pkpc)
                ax.scatter([rvir_bh_pkpc] * 2, [yp_bh_av, yp_bh_med],
                            marker='o', c=color, s=60)
            else:
                ax.scatter([rvir_bh_pkpc], [yp_bh_med],
                            marker='o', c=color, s=60)

    handles3, labels = ax.get_legend_handles_labels()
    handles1 = [mlines.Line2D((), (), linewidth=lw_med, linestyle='solid',
                            label='med.', color='black'),
                mlines.Line2D((), (), linestyle=None, marker='o',
                            label='$\\mathrm{R}_{\\mathrm{vir}}$', 
                            color='black', markersize=5)
                ]
    if domean:
        handles1 = [mlines.Line2D((), (), linewidth=lw_av, linestyle='solid',
                                  label='mean', color='black')] + handles1 
    if doscatter:
        handles1 = handles1 + \
                   [mpatch.Patch(label='perc. 10-90', linewidth=0.5, 
                                 color='black', alpha=alpha_range)]
    if len(modelfilelists) > 1: # more than one line per phys model
        handles2 = [mlines.Line2D((), (), linewidth=1.7, linestyle=ls,
                                  label=lab, color='black')\
                    for ls, lab in zip(lss, physmodel_labels)]
    else:
        handles2 = []
    ax.legend(handles=handles1 + handles2 + handles3, fontsize=fontsize)
    ax.set_xscale('log')

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def compare_profilesets_physmodel(mapset='clean_set1', var='zev'):

    if mapset == 'clean_set1':
        outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/profiles_clean_set1/'
        
        mdir = '/Users/nastasha/ciera/sim_maps/fire/clean_set1/'
        tpl_m12_noBH = 'coldens_{qt}_{ic}_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5'
        tpl_m12_AGN_CR = 'coldens_{qt}_{ic}_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap50_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5'
        tpl_m12_AGN_noCR = 'coldens_{qt}_{ic}_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5'
        tpl_m13_noBH = 'coldens_{qt}_{ic}_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5_snap50_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        tpl_m13_AGN_CR = 'coldens_{qt}_{ic}_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000_snap50_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        tpl_m13_AGN_noCR_h113 = 'coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        tpl_m13_AGN_noCR_h206 = 'coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5_snap258_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5' 
        model_labels = ['no BH', 'AGN, no CR', 'AGN + CR']
        ics_m12 = ['m12f'] # bug: 'm12m'
        ics_m13 = ['m13h206', 'm13h113']
        qts = ['gas-mass', 'O6', 'Ne8', 'Mg10', 'H1']
        h1_imstr = '_ionfrac-fromsim'
        
        fnss_m12 = [[[mdir + tpl_m12_noBH.format(ic=ic, qt=ion, 
                          im=h1_imstr if ion == 'H1' else ''),
                      mdir + tpl_m12_AGN_noCR.format(ic=ic, qt=ion,
                          im=h1_imstr if ion == 'H1' else ''),
                      mdir + tpl_m12_AGN_CR.format(ic=ic, qt=ion, 
                          im=h1_imstr if ion == 'H1' else ''),
                      ]] \
                    for ic in ics_m12 for ion in qts]
        fnss_m13 = [[[mdir + tpl_m13_noBH.format(ic=ic, qt=ion, 
                          im=h1_imstr if ion == 'H1' else ''),
                      mdir + tpl_m13_AGN_noCR_h206.format(ic=ic, 
                          qt=ion, im=h1_imstr if ion == 'H1' else '')\
                         if 'h206' in ic else
                         mdir + tpl_m13_AGN_noCR_h113.format(ic=ic, 
                             qt=ion, im=h1_imstr if ion == 'H1' else ''),
                      mdir + tpl_m13_AGN_CR.format(ic=ic, qt=ion, 
                          im=h1_imstr if ion == 'H1' else ''),
                     ]] \
                     for ic in ics_m13 for ion in qts]
        fnsss = fnss_m12 + fnss_m13
        model_labelss = [[model_labels]] * len(fnsss)
        physmodel_labelss = [model_labels] * len(fnsss)

        rbinss = [np.linspace(0., 300., 50) if 'm12' in ic else \
                  np.linspace(0., 700., 50) if 'm13' in ic else \
                  None \
                  for ic in ics_m12 + ics_m13 for qt in qts]
        mlabel = ('$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}'+\
                  ' \\, [\\mathrm{g} \\, \\mathrm{cm}^2]$')
        ilabel = ('$\\log_{{10}} \\, \\mathrm{{N}}(\\mathrm{{{ion}}})'+\
                  ' \\, [\\mathrm{{cm}}^2]$')
        ylabels = [mlabel if 'mass' in qt else\
                   ilabel.format(ion=qt) \
                   for ic in  ics_m12 + ics_m13 for qt in qts]
        outtemp = 'rprofcomp_clean_set1_{ic}_{qt}_z0p5_noBH_AGNCR_AGNnoCR.pdf'
        outnames = [outdir + outtemp.format(ic=ic, qt=qt) \
                    for ic in ics_m12 + ics_m13 for qt in qts]

        ttpl_m12 = ('noBH-noCR: m7e3, AGN-noCR: m7e3, sdp2e-4,'
                    ' AGN-CR: m6e4, sdp1e-4, fcr1e-3_vw3000') 
        ttpl_m13h113 = ('noBH-noCR: m3e5, AGN-noCR: m3e4, sdp1e-4,'
                        ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
        ttpl_m13h206 = ('noBH-noCR: m3e5, AGN-noCR: m3e4, sdp3e-4,'
                        ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
        title_tpl= '{ion} profiles, {ic} z=0.5, {ttpl}'
        titles = [title_tpl.format(ic=ic, ion=ion, 
                                   ttpl=ttpl_m12 if 'm12' in ic else 
                                        ttpl_m13h206 if 'm13h206' in ic else\
                                        ttpl_m13h113)\
                  for ic in ics_m12 + ics_m13 for ion in qts]
        
        
        _outname = 'rprof_noBH_AGN_AGNCR_set1_{ic}_z0p5_gas_{ion}_nomean.pdf'
        outnames = [_outname.format(ic=ic, ion=ion) \
                    for ic in ics_m12 + ics_m13 for ion in qts]
        domean = False
    if mapset == 'clean_set1-2':
        outdir = ('/Users/nastasha/ciera/projects_lead/fire3_ionabs/'
                  'profiles_clean/')
        mdir1 = '/Users/nastasha/ciera/sim_maps/fire/clean_set1/'
        mdir2 = '/Users/nastasha/ciera/sim_maps/fire/clean_set2/'
        tpl_m12_noBH =          ('coldens_{qt}_{ic}_m7e3_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp1e10'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m12_AGN_CR =        ('coldens_{qt}_{ic}_m6e4_MHDCRspec1_fire3'
                                 '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                                 '_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m12_AGN_noCR =      ('coldens_{qt}_{ic}_m7e3_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp2e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m13_noBH =          ('coldens_{qt}_{ic}_m3e5_MHD_fire3'
                                 '_fireBH_Sep182021_crdiffc690_sdp1e10'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_CR =        ('coldens_{qt}_{ic}_m3e5_MHDCRspec1_fire3'
                                 '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                                 '_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_noCR_h113 = ('coldens_{qt}_{ic}_m3e4_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp1e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_noCR_h206 = ('coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH'
                                 '_Sep182021_hr_crdiffc690_sdp3e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        model_labels = ['no BH', 'AGN, no CR', 'AGN + CR']
        ics_m12 = ['m12f'] # bug: 'm12m'
        ics_m13 = ['m13h206', 'm13h113']
        qts = ['gas-mass', 'O6', 'Ne8', 'Mg10', 'H1']
        h1_imstr = '_ionfrac-fromsim'
        snaps_sr = [50, 49, 48, 47, 46, 45]
        snaps_hr =  [258, 240, 224, 210, 197, 186]
        redshifts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        mlabel = ('$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}'+\
                  ' \\, [\\mathrm{g} \\, \\mathrm{cm}^2]$')
        ilabel = ('$\\log_{{10}} \\, \\mathrm{{N}}(\\mathrm{{{ion}}})'+\
                  ' \\, [\\mathrm{{cm}}^2]$')
        
        if var == 'model':
            ttpl_m12 = ('noBH: m7e3, AGN-noCR: m7e3, sdp2e-4,'
                       ' AGN-CR: m6e4, sdp1e-4, fcr1e-3_vw3000') 
            ttpl_m13h113 = ('noBH: m3e5, AGN-noCR: m3e4, sdp1e-4,'
                            ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 
            ttpl_m13h206 = ('noBH: m3e5, AGN-noCR: m3e4, sdp3e-4,'
                            ' AGN-CR: m3e5, sdp1e-4, fcr1e-3_vw3000') 

            fnss_m12 = [[[tpl_m12_noBH.format(ic=ic, qt=qt, 
                              im=h1_imstr if qt == 'H1' else '', 
                              snap=snhr),
                          tpl_m12_AGN_noCR.format(ic=ic, qt=qt, 
                              im=h1_imstr if qt == 'H1' else '', 
                              snap=snhr),
                          tpl_m12_AGN_CR.format(ic=ic, qt=qt, 
                              im=h1_imstr if qt == 'H1' else '', 
                              snap=snsr),
                          ]] \
                          for ic in ics_m12 for qt in qts \
                          for snhr, snsr in zip(snaps_hr, snaps_sr)]
            fnss_m13 = [[[tpl_m13_noBH.format(ic=ic, qt=qt,
                              im=h1_imstr if qt == 'H1' else '',
                              snap=snsr),
                          tpl_m13_AGN_noCR_h206.format(ic=ic,
                              qt=qt, im=h1_imstr if qt == 'H1' else '',
                              snap=snhr)\
                            if 'h206' in ic else
                            tpl_m13_AGN_noCR_h113.format(ic=ic, 
                                qt=qt, im=h1_imstr if qt == 'H1' else '',
                                snap=snhr),
                          tpl_m13_AGN_CR.format(ic=ic, qt=qt, 
                                im=h1_imstr if qt == 'H1' else '', 
                                snap=snsr),
                         ]] \
                         for ic in ics_m13 for qt in qts\
                         for snhr, snsr in zip(snaps_hr, snaps_sr)]
            fnsss = fnss_m12 + fnss_m13
            fnsss = [[[mdir1 + fn if 'snap50' in fn or 'snap258' in fn else\
                       mdir2 + fn for fn in lst] for lst in _] for _ in fnsss]

            model_labelss = [[model_labels]] * len(fnsss)
            physmodel_labelss = [model_labels] * len(fnsss)

            rbinss = [np.linspace(0., 300., 50) if 'm12' in ic else \
                      np.linspace(0., 600., 50) if 'm13' in ic else \
                      None \
                      for ic in ics_m12 + ics_m13 for qt in qts\
                      for snhr, snsr in zip(snaps_hr, snaps_sr)]
            ylabels = [mlabel if 'mass' in qt else\
                       ilabel.format(ion=qt) \
                       for ic in  ics_m12 + ics_m13 for qt in qts\
                       for snhr, snsr in zip(snaps_hr, snaps_sr)]
            title_tpl= '{ion} profiles, {ic} z={z:.1f}, {ttpl}'
            titles = [title_tpl.format(ic=ic, ion=ion, z=z,
                                       ttpl=ttpl_m12 if 'm12' in ic else 
                                            ttpl_m13h206 if 'm13h206' in ic else\
                                            ttpl_m13h113)\
                      for ic in ics_m12 + ics_m13 for ion in qts\
                      for z in redshifts]
            
            outtemp = ('rprofcomp_clean_set1_set2_{ic}_{qt}'
                       '_z{z}_noBH_AGNCR_AGNnoCR.pdf')
            outnames = [outtemp.format(ic=ic, qt=qt,
                            z=('{:.1f}'.format(z)).replace('.', 'p')) \
                        for ic in ics_m12 + ics_m13 for qt in qts \
                        for z in redshifts]
            domean = False

    for fnss, model_labels, physmodel_labels, title, outname, \
            ylabel, rbins_pkpc in \
            zip(fnsss, model_labelss, physmodel_labelss, titles, outnames, 
                ylabels, rbinss):    
        outname = outdir + outname

        if title is not None:
            if len(title) > 70:
                splitopts = np.where([char == ',' for char in title])[0]
                optchoice = np.argmin(np.abs(splitopts - 0.5 * len(title)))
                splitind = splitopts[optchoice]
                title = (title[:splitind + 1]).strip() + '\n' +\
                        (title[splitind + 1:]).strip()
        compare_profiles_physmodels(fnss, model_labels,
                                    physmodel_labels,
                                    rbins_pkpc, title=title, 
                                    ylabel=ylabel, ax=None,
                                    outname=outname, domean=domean)

def compare_physmodel_z_panels(modelfiles_per_panel,
                               legendlablists_per_panel,
                               physmodel_labels_per_panel,
                               titles_per_panel,
                               ensemble_legend_labels,
                               rbins_pkpc_per_panel,
                               figtitle=None, ylabel=None,
                               outname=None):
    '''
    It's assumed modelfiles_per_panel and legendlablists_per_panel,
    and physmodel_labels_per_panel are lists of lists of strings.
    ensemble_legend_labels is a list of strings.
    '''
    nmodels = len(modelfiles_per_panel)
    ncols = min(3, nmodels + 1)
    nrows = (nmodels + 1) // 3 + 1
    panelsize = 2.5
    hspace = 0.25
    wspace = 0.25
    height_ratios = [panelsize] * nrows
    width_ratios = [panelsize] * ncols
    height = sum(height_ratios) * (1. + hspace * (ncols - 1.))
    width = sum(width_ratios) * (1. + wspace * (nrows - 1.))
    
    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols, 
                        hspace=hspace, wspace=wspace, 
                        width_ratios=width_ratios, 
                        height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) \
            for i in range(nmodels + 1)]
    li = ncols - (nmodels + 1) % ncols
    lax = fig.add_subplot(grid[nrows - 1, -1 * li:])

    fontsize = 12
    nlines = len(modelfiles_per_panel[0])
    _colors = tc.tol_cmap('rainbow_discrete', nlines)
    colors = _colors(np.linspace(1. / nlines, 1. - 1. / nlines, nlines))
    # zip/list comprehension issues or something when using 
    # tol_cmap outputs directly
    colors = [mcolors.to_rgb(col) for col in colors]
    modelcolors = tc.tol_cset('bright')

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize)

    for axi, (modelfiles, legendlablists, physmodel_labels, title, 
              rbins_pkpc, ax) in \
            enumerate(zip(modelfiles_per_panel, legendlablists_per_panel, 
                      physmodel_labels_per_panel, titles_per_panel, 
                      rbins_pkpc_per_panel, axes)):
        if axi % ncols == 0:
            _ylabel = ylabel
        else:
            _ylabel = None
        compare_profiles_physmodels([modelfiles],
                                    [legendlablists],
                                    physmodel_labels,
                                    rbins_pkpc, title=None, 
                                    ylabel=_ylabel, ax=ax,
                                    outname=None, domean=False,
                                    doscatter=False, colors=colors)
        ax.set_title(title, fontsize=fontsize)
        # legend in its own panel
        if axi == 0:
            handles2, labels2 = ax.get_legend_handles_labels()
        ax.get_legend().remove() # remove annoying empty rectangle

        if axi >= nmodels + 1 - ncols:
            ax.set_xlabel('$\\mathrm{r}_{\\perp} \\, [\\mathrm{pkpc}]$',
                          fontsize=fontsize)
        else:
            ax.set_xlabel(None)
        
        av, med, p90, p10 = get_profile_massmap(
                modelfiles, rbins_pkpc, rbin_units='pkpc', 
                profiles=['av-lin', 'perc-0.5', 'perc-0.9', 'perc-0.1'])
        rcens = 0.5 * (rbins_pkpc[:-1] + rbins_pkpc[1:])
        ax.plot(rcens, med, color='black', linewidth=2, linestyle='solid')
        ax.fill_between(rcens, p10, p90, color='black', alpha=0.3)

        c_last = modelcolors[axi]
        axes[nmodels].plot(rcens, med, color=c_last, linewidth=2, 
                           linestyle='solid', 
                           label=ensemble_legend_labels[axi])
        axes[nmodels].fill_between(rcens, p10, p90, color=c_last, alpha=0.3)
    
    if nmodels % ncols == 0:
        axes[nmodels].set_ylabel(ylabel, fontsize=fontsize)
    axes[nmodels].set_xlabel('$\\mathrm{r}_{\\perp} \\, [\\mathrm{pkpc}]$',
                             fontsize=fontsize)
    axes[nmodels].tick_params(direction='in', which='both', 
                              labelsize=fontsize - 1)
    axes[nmodels].legend(fontsize=fontsize, bbox_to_anchor=(1.05, 0.00),
                         loc='lower left')
    axes[nmodels].set_xscale('log')
    axes[nmodels].set_title('ensemble profiles')
    
    # sync y ranges
    ylims = [ax.get_ylim() for ax in axes]
    ymin = min([ylim[0] for ylim in ylims])
    if 'gas-mass' not in modelfiles_per_panel[0][0]:
        ymin = max(ymin, 10.)
    ymax = max([ylim[1] for ylim in ylims])    
    [ax.set_ylim((ymin, ymax)) for ax in axes]

    lax.axis('off')
    line = [[(0, 0)]]
    lc = mcol.LineCollection(line * len(colors), linestyle='solid', 
                             linewidth=2., colors=colors)
    handles1 = [lc,
                mlines.Line2D((), (), linestyle='solid', linewidth=2,
                              label='all med.', color='black'),
                mpatch.Patch(label='all perc. 10-90', linewidth=0.5, 
                             color='black', alpha=0.3),
                mlines.Line2D((), (), linestyle=None, marker='o',
                              label='$\\mathrm{R}_{\\mathrm{vir}}$', 
                              color='gray', markersize=5)
                ]
    labels1 = ['median', 'all med.', 'all perc. 10-90', 
               '$\\mathrm{R}_{\\mathrm{vir}}$']
    lax.legend(handles1 + handles2, labels1 + labels2, 
               fontsize=fontsize, ncols=2,
               handler_map={type(lc): pu.HandlerDashedLines()},
               bbox_to_anchor=(0.95, 0.95), loc='upper right')

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def compare_physmodel_z_mapsets(mapset='clean_set1-2'):

    if mapset == 'clean_set1-2':
        outdir = ('/Users/nastasha/ciera/projects_lead/fire3_ionabs/'
                  'profiles_clean/')
        mdir1 = '/Users/nastasha/ciera/sim_maps/fire/clean_set1/'
        mdir2 = '/Users/nastasha/ciera/sim_maps/fire/clean_set2/'
        tpl_m12_noBH =          ('coldens_{qt}_{ic}_m7e3_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp1e10'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m12_AGN_CR =        ('coldens_{qt}_{ic}_m6e4_MHDCRspec1_fire3'
                                 '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                                 '_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m12_AGN_noCR =      ('coldens_{qt}_{ic}_m7e3_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp2e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5')
        tpl_m13_noBH =          ('coldens_{qt}_{ic}_m3e5_MHD_fire3'
                                 '_fireBH_Sep182021_crdiffc690_sdp1e10'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_CR =        ('coldens_{qt}_{ic}_m3e5_MHDCRspec1_fire3'
                                 '_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                                 '_gacc31_fa0.5_fcr1e-3_vw3000_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_noCR_h113 = ('coldens_{qt}_{ic}_m3e4_MHD_fire3'
                                 '_fireBH_Sep182021_hr_crdiffc690_sdp1e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        tpl_m13_AGN_noCR_h206 = ('coldens_{qt}_{ic}_m3e4_MHD_fire3_fireBH'
                                 '_Sep182021_hr_crdiffc690_sdp3e-4'
                                 '_gacc31_fa0.5_snap{snap}'
                                 '_shrink-sph-cen_BN98_2rvir{im}_v2.hdf5') 
        model_labels = ['no BH', 'AGN, no CR', 'AGN + CR']
        ics_m12 = ['m12f'] # bug: 'm12m'
        ics_m13 = ['m13h206', 'm13h113']
        qts = ['gas-mass', 'O6', 'Ne8', 'Mg10', 'H1']
        h1_imstr = '_ionfrac-fromsim'
        # low to high z -> colored blue to red
        snaps_sr = [50, 49, 48, 47, 46, 45]
        snaps_hr =  [258, 240, 224, 210, 197, 186]
        redshifts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        mlabel = ('$\\log_{10} \\, \\Sigma_{\\mathrm{gas}}'+\
                  ' \\, [\\mathrm{g} \\, \\mathrm{cm}^2]$')
        ilabel = ('$\\log_{{10}} \\, \\mathrm{{N}}(\\mathrm{{{ion}}})'+\
                  ' \\, [\\mathrm{{cm}}^2]$')
        
        ics = ics_m12 + ics_m13
        ensemble_legend_labels = ['all noBH', 'all AGN-noCR', 'all AGN-CR']
        titles_per_panel_m12  = ['noBH: m7e3', 'AGN-noCR: m7e3, sdp2e-4',
                                 'AGN-CR:\nm6e4, sdp1e-4, fcr1e-3_vw3000'] 
        titles_per_panel_m13h113 = ['noBH: m3e5', 'AGN-noCR: m3e4, sdp1e-4',
                                    'AGN-CR:\nm3e5, sdp1e-4, fcr1e-3_vw3000'] 
        titles_per_panel_m13h206 = ['noBH: m3e5', 'AGN-noCR: m3e4, sdp3e-4',
                                    'AGN-CR:\nm3e5, sdp1e-4, fcr1e-3_vw3000']

        fnss_m12 = [[[tpl_m12_noBH.format(ic=ic, qt=qt, 
                            im=h1_imstr if qt == 'H1' else '', 
                            snap=snhr)\
                      for snhr in snaps_hr],
                     [tpl_m12_AGN_noCR.format(ic=ic, qt=qt, 
                            im=h1_imstr if qt == 'H1' else '', 
                            snap=snhr)\
                      for snhr in snaps_hr],
                     [tpl_m12_AGN_CR.format(ic=ic, qt=qt, 
                            im=h1_imstr if qt == 'H1' else '', 
                            snap=snsr)\
                      for snsr in snaps_sr],
                     ] \
                    for ic in ics_m12 for qt in qts]
        fnss_m13 = [[[tpl_m13_noBH.format(ic=ic, qt=qt,
                            im=h1_imstr if qt == 'H1' else '',
                            snap=snsr)\
                      for snsr in snaps_sr],
                     [tpl_m13_AGN_noCR_h206.format(ic=ic,
                            qt=qt, im=h1_imstr if qt == 'H1' else '',
                            snap=snhr)\
                      if 'h206' in ic else \
                      tpl_m13_AGN_noCR_h113.format(ic=ic, 
                            qt=qt, im=h1_imstr if qt == 'H1' else '',
                            snap=snhr) \
                      for snhr in snaps_hr],
                     [tpl_m13_AGN_CR.format(ic=ic, qt=qt, 
                            im=h1_imstr if qt == 'H1' else '', 
                            snap=snsr)\
                      for snsr in snaps_sr],
                     ] \
                     for ic in ics_m13 for qt in qts]
        fnsss = fnss_m12 + fnss_m13
        fnsss = [[[mdir1 + fn if 'snap50' in fn or 'snap258' in fn else\
                   mdir2 + fn for fn in lst] for lst in _] for _ in fnsss]

        legendlabss = [[['z={:.1f}'.format(z) for z in redshifts]\
                        if i == 0 else [None] * len(redshifts) \
                        for i in range(len(fnsss[0]))] \
                       for ic in ics for qt in qts]
        physmodel_labelss = [[['z={:.1f}'.format(z) for z in redshifts]
                              for i in range(len(fnsss[0]))] \
                             for ic in ics_m12 + ics_m13 for qt in qts]
        
        rbinss = [[np.linspace(0., 300., 50) if 'm12' in ic else \
                   np.linspace(0., 600., 50) if 'm13' in ic else \
                   None] * len(fnsss[0]) \
                  for ic in ics for qt in qts]
        ylabels = [mlabel if 'mass' in qt else\
                   ilabel.format(ion=qt) \
                   for ic in  ics for qt in qts]
        title_tpl= '{qt} profiles, {ic}'
        figtitles = [title_tpl.format(ic=ic, qt=qt)\
                      for ic in ics for qt in qts]
        ensemble_legend_labelss = [ensemble_legend_labels \
                                   for ic in ics for qt in qts]
        ptitles_list = [titles_per_panel_m12 if 'm12' in ic else\
                        titles_per_panel_m13h113 if ic == 'm13h113' else\
                        titles_per_panel_m13h206 if ic == 'm13h206' else\
                        None\
                        for ic in ics for qt in qts]
        outtemp = ('rprofcomp_clean_set1_set2_{ic}_{qt}'
                   '_z0p5-1_noBH_AGNCR_AGNnoCR.pdf')
        outnames = [outtemp.format(ic=ic, qt=qt) \
                    for ic in ics for qt in qts]

    for fnss, physmodel_labels, title, outname, ptitles,\
            ylabel, rbins_pkpc, legendlabs, enslabs in \
            zip(fnsss, physmodel_labelss, figtitles, outnames, ptitles_list,
                ylabels, rbinss, legendlabss, ensemble_legend_labelss):    
        outname = outdir + outname

        if title is not None:
            if len(title) > 70:
                splitopts = np.where([char == ',' for char in title])[0]
                optchoice = np.argmin(np.abs(splitopts - 0.5 * len(title)))
                splitind = splitopts[optchoice]
                title = (title[:splitind + 1]).strip() + '\n' +\
                        (title[splitind + 1:]).strip()
        compare_physmodel_z_panels(fnss,
                                  legendlabs,
                                  physmodel_labels,
                                  ptitles,
                                  enslabs,
                                  rbins_pkpc,
                                  figtitle=title, ylabel=ylabel,
                                  outname=outname)

def check_h1maps():
    '''
    specific sanity test comparing total H, H I from the PS20 tables,
    and H I from the FIRE NeutralHydrogenAbundance field
    '''

    mapdir = '/Users/nastasha/ciera/tests/fire_start/h1_sim_test/'
    mapf_Htot = ('coldens_Hydrogen_m12m_m7e3_MHD_fire3_fireBH_Sep182021_hr'
                 '_crdiffc690_sdp1e10_gacc31_fa0.5_snap500_shrink-sph-cen'
                 '_BN98_2rvir_v2.hdf5')
    mapf_H1sim = ('coldens_H1-sim_m12m_m7e3_MHD_fire3_fireBH_Sep182021_hr'
                  '_crdiffc690_sdp1e10_gacc31_fa0.5_snap500_shrink-sph-cen'
                  '_BN98_2rvir_v2.hdf5')
    mapf_H1PS20 = ('coldens_H1-PS20_m12m_m7e3_MHD_fire3_fireBH_Sep182021_hr'
                   '_crdiffc690_sdp1e10_gacc31_fa0.5_snap500_shrink-sph-cen'
                   '_BN98_2rvir_v2.hdf5')

    maps = {}
    extents = {}
    rvirs = {}
    vmin = np.inf
    vmax = -np.inf
    mapkeys = ['htot', 'h1sim', 'h1ps20']
    maptitles = {'htot': '$\\mathrm{N}(\\mathrm{H})$',
                 'h1sim': '$\\mathrm{N}(\\mathrm{H I})$, FIRE',
                 'h1ps20': '$\\mathrm{N}(\\mathrm{H I})$, PS20 table',
                 }
    fractitles = {'h1sim': ('$\\mathrm{N}(\\mathrm{H I}) \\,/\\,'
                           ' \\mathrm{N}(\\mathrm{H})$, FIRE'), 
                  'h1ps20': ('$\\mathrm{N}(\\mathrm{H I}) \\,/\\,'
                             ' \\mathrm{N}(\\mathrm{H})$, PS20 tables'),
                  }

    for mapkey, mapf in zip(mapkeys, 
                            [mapf_Htot, mapf_H1sim, mapf_H1PS20]):
        with h5py.File(mapdir + mapf, 'r') as f:
            _map = f['map'][:]
            _vmin = f['map'].attrs['minfinite']
            _vmax = f['map'].attrs['max']
            print(mapf, _vmin, _vmax)

            box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
            cosmopars = {key: val for key, val in \
                        f['Header/inputpars/cosmopars'].attrs.items()}
            #print(cosmopars)
            if 'Rvir_ckpcoverh' in f['Header/inputpars/halodata'].attrs:
                rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
                rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
            elif 'Rvir_cm' in f['Header/inputpars/halodata'].attrs:
                rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                rvir_pkpc = rvir_cm / (c.cm_per_mpc * 1e-3)
            xax = f['Header/inputpars'].attrs['Axis1']
            yax = f['Header/inputpars'].attrs['Axis2']
            box_pkpc = box_cm / (1e-3 * c.cm_per_mpc)
            extent = (-0.5 * box_pkpc[xax], 0.5 * box_pkpc[xax],
                      -0.5 * box_pkpc[yax], 0.5 * box_pkpc[yax])
            
            maps[mapkey] = _map
            extents[mapkey] = extent
            rvirs[mapkey] = rvir_pkpc
            vmin = min(vmin, _vmin)
            vmax = max(vmax, _vmax)
    
    mincol = 13.6
    if mincol > vmin and mincol < vmax:
        cmap = pu.paste_cmaps(['gist_yarg', 'viridis'], [vmin, mincol, vmax])
    else:
        cmap = 'virdis'
    anyzero = np.min([np.min(maps[key]) for key in maps]) == vmin
    extend = 'neither' if anyzero else 'min'

    fig = plt.figure(figsize=(8.5, 5.))
    grid = gsp.GridSpec(nrows=2, ncols=5, hspace=0.2, wspace=0.2, 
                        width_ratios=[1., 5., 5., 5., 1.])
    mapaxes = [fig.add_subplot(grid[0, 1+ i]) for i in range(3)]
    mapcax = fig.add_subplot(grid[0, 4])
    fracaxes = [fig.add_subplot(grid[1, 2 + i]) for i in range(2)]
    fraccax = fig.add_subplot(grid[1, 4])
    diffax = fig.add_subplot(grid[1, 1])
    diffcax = fig.add_subplot(grid[1, 0])
    fontsize = 12
    
    # should be same for all three maps
    xlabel = ['X', 'Y', 'Z'][xax] + ' [pkpc]'
    ylabel = ['X', 'Y', 'Z'][yax] + ' [pkpc]'

    for mi, mapkey in enumerate(mapkeys):
        ax = mapaxes[mi]
        if mi == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(maptitles[mapkey], fontsize=fontsize)

        img = ax.imshow(maps[mapkey].T, origin='lower', 
                       interpolation='nearest', 
                       vmin=vmin, vmax=vmax, cmap=cmap, 
                       extent=extents[mapkey]
                        )
        ax.tick_params(axis='both', labelsize=fontsize-1,
                       labelleft=mi == 0, labelbottom=False)

        patches = [mpatch.Circle((0., 0.), rvir_pkpc)]
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5)
        ax.add_collection(collection)
        if mi == 0:
            ax.text(1.05 * 2**-0.5 * rvir_pkpc, 1.05 * 2**-0.5 * rvir_pkpc, 
                   '$R_{\\mathrm{vir}}$',
                    color='red', fontsize=fontsize)
    plt.colorbar(img, cax=mapcax, extend=extend, orientation='vertical') 
    mapcax.set_ylabel('$\\log_{10} \\, \\mathrm{N} \\; [\\mathrm{cm}^{-2}]$',
                      fontsize=fontsize) 

    fmaps = {'h1sim': maps['h1sim'] - maps['htot'],
             'h1ps20': maps['h1ps20'] - maps['htot']
             }
    fvmin = np.min([np.min(fmaps[fkey][np.isfinite(fmaps[fkey])])\
                    for fkey in fmaps])
    fvmax = np.min([np.max(fmaps[fkey][np.isfinite(fmaps[fkey])])\
                    for fkey in fmaps])
    anyzero = np.min([np.min(fmaps[fkey]) for fkey in fmaps]) == fvmin
    extend = 'neither' if anyzero else 'min'
    fcmap = 'afmhot'
    
    for mi, mapkey in enumerate(['h1sim', 'h1ps20']):
        ax = fracaxes[mi]
        ax.set_xlabel(xlabel, fontsize=fontsize)
        #if mi == 0:
        #    ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(fractitles[mapkey], fontsize=fontsize)

        img = ax.imshow(fmaps[mapkey].T, origin='lower', 
                        interpolation='nearest', 
                        vmin=fvmin, vmax=fvmax, cmap=fcmap, 
                        extent=extents[mapkey]
                        )
        ax.tick_params(axis='both', labelsize=fontsize-1, 
                       labelleft=False, labelbottom=True)

        patches = [mpatch.Circle((0., 0.), rvir_pkpc)]
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5)
        ax.add_collection(collection)
    plt.colorbar(img, cax=fraccax, extend=extend, orientation='vertical')
    fraccax.set_ylabel('$\\log_{10} \\, \\mathrm{N}(\\mathrm{H I}) \\,/\\,'
                       ' \\mathrm{N}(\\mathrm{H})$', fontsize=fontsize) 
    
    ax = diffax
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_title('FIRE / PS20', fontsize=fontsize)
    diffmap = fmaps['h1sim'] - fmaps['h1ps20']
    vmax = np.max(np.abs(diffmap[np.isfinite(diffmap)]))
    vmin = -1. * vmax
    img = ax.imshow(diffmap.T, origin='lower', 
                    interpolation='nearest', 
                    vmin=vmin, vmax=vmax, cmap='RdBu', 
                    extent=extents['h1sim']
                    )
    ax.tick_params(axis='both', labelsize=fontsize-1,
                   labelleft=False)
    patches = [mpatch.Circle((0., 0.), rvir_pkpc)]
    collection = mcol.PatchCollection(patches)
    collection.set(edgecolor=['red'], facecolor='none', linewidth=1.5)
    ax.add_collection(collection)
    plt.colorbar(img, cax=diffcax, extend='neither', orientation='vertical')
    diffcax.set_ylabel('$\\Delta \\, \\log_{10} \\,'
                       ' \\mathrm{N}(\\mathrm{H I})$', fontsize=fontsize) 
    diffcax.yaxis.set_label_position('left')
    diffcax.yaxis.tick_left()

    outdir = '/Users/nastasha/ciera/tests/fire_start/h1_sim_test/'
    savename = outdir + 'mapcomp_H_H1-sim_H1-PS20.pdf'
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')


def plot3dprof_weightvalgrid(filen_template, fillkw_weights, fillkw_vals,
                             figtitle=None, outname=None):
    
    rholab = ('$\\log_{10}\\,\\rho \\; '
              '[\\mathrm{H}(X=0.752) \\, \\mathrm{cm}^{-3}]$')
    tlab = '$\\log_{10}\\,\\mathrm{T} \\; [\\mathrm{K}]$'
    zolab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{O} \\;'
             ' [\\mathrm{Z}_{\\mathrm{O}, \\odot}]$')
    znelab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Ne} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Ne}, \\odot}]$')
    zmglab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Mg} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Mg}, \\odot}]$')
    ylabopts = {('sim-direct', 'Density'): rholab,
                ('sim-direct', 'Temperature'): tlab,
                ('sim-direct', 'ElementAbundance/Oxygen'): zolab,
                ('sim-direct', 'ElementAbundance/Neon'): znelab,
                ('sim-direct', 'ElementAbundance/Magnesium'): zmglab,
               }
    rhonorm = c.atomw_H * c.u / 0.752
    tnorm = 1.
    zonorm = ol.solar_abunds_ea['oxygen']
    znenorm = ol.solar_abunds_ea['neon']
    zmgnorm = ol.solar_abunds_ea['magnesium']
    ynormopts = {('sim-direct', 'Density'): rhonorm,
                 ('sim-direct', 'Temperature'): tnorm,
                 ('sim-direct', 'ElementAbundance/Oxygen'): zonorm,
                 ('sim-direct', 'ElementAbundance/Neon'): znenorm,
                 ('sim-direct', 'ElementAbundance/Magnesium'): zmgnorm,
                }
    wlabopts = {'Mass': 'Mass-weighted',
                'Volume': 'Volume-weighted',
                ('ion', 'O6'): 'O VI-weighted',
                ('ion', 'Ne8'): 'Ne VIII-weighted',
                ('ion', 'Mg10'): 'Mg X-weighted',
                ('ion', 'H1'): 'H I-weighted',
                }
    hists = []
    hists_raw = []
    rvals = []
    yvals = []
    ykeys = []
    wkeys = []
    rvirs = []
    runits = None
    for vdict in fillkw_vals:
        _hists = []
        _rvals = []
        _yvals = []
        _ykeys = []
        _wkeys = []
        _rvirs = []
        _hists_raw = []
        for wdict in fillkw_weights:
            _tdict = wdict.copy()
            _tdict.update(vdict)
            filen = filen_template.format(**_tdict)
            with h5py.File(filen, 'r') as f:
                rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                rv = f['axis_0/bins'][:]
                _runits = f['axis_0'].attrs['units'].decode()
                yv = f['axis_1/bins'][:]
                ykey = f['axis_1'].attrs['qty_type'].decode()
                if ykey == 'sim-direct':
                    _path = 'axis_1/qty_type_args'
                    _key = 'field'
                    ykey = (ykey, f[_path].attrs[_key].decode())
                elif ykey == 'ion':
                    _path = 'axis_1/qty_type_args'
                    _key = 'ion'
                    ykey = (ykey, f[_path].attrs[_key].decode())
                elif ykey == 'Metal':
                    _path = 'axis_1/qty_type_args'
                    _key = 'element'
                    ykey = (ykey, f[_path].attrs[_key].decode())
                hist_raw = f['histogram/histogram'][:]
                wkey = f['histogram'].attrs['weight_type'].decode()
                if wkey == 'ion':
                    _path = 'histogram/weight_type_args'
                    _key = 'ion'
                    wkey = (wkey, f[_path].attrs[_key].decode())
                elif wkey == 'Metal':
                    _path = 'histogram/weight_type_args'
                    _key = 'element'
                    wkey = (wkey, f[_path].attrs[_key].decode())
               
                deltax = np.diff(rv)
                if yv[0] == -np.inf:
                    yv[0] = 2. * yv[1] - yv[2]
                if yv[-1] == np.inf:
                    yv[-1] = 2. * yv[-2] - yv[-3]
                deltay = np.diff(yv)
                hist_norm = hist_raw - np.log10(np.sum(10**hist_raw))
                hist_norm = hist_norm - np.log10(deltax)[:, np.newaxis]
                hist_norm = hist_norm - np.log10(deltay)[np.newaxis, :]
                
            if runits is not None:
                if _runits != runits:
                    raise ValueError('mismatch in r3D units')
            if _runits == 'pkpc':
                rvir = rvir_cm / (c.cm_per_mpc * 1e-3)
            elif _runits == 'Rvir':
                rvir = 1.
            
            _hists_raw.append(hist_raw)
            _hists.append(hist_norm)
            _rvals.append(rv)
            _yvals.append(yv)
            _ykeys.append(ykey)
            _wkeys.append(wkey)
            _rvirs.append(rvir)
        hists_raw.append(_hists_raw)
        hists.append(_hists)
        rvals.append(_rvals)
        yvals.append(_yvals)
        ykeys.append(_ykeys)
        wkeys.append(_wkeys)
        rvirs.append(_rvirs)
    print(ykeys)
    print(wkeys)

    if runits == 'Rvir':
        xlabel = '$\\mathrm{r} \\; [\\mathrm{R}_{\\mathrm{vir}}]$'
        markrvir = False
    else:
        xlabel = '$\\mathrm{r} \\; [\\mathrm{pkpc}]$'
        markrvir = True

    panelsize = 2.
    caxwidth = 0.5
    ncols = len(fillkw_weights)
    nrows = len(fillkw_vals)
    wspace = 0.
    hspace = 0. 
    width_ratios = [panelsize] * ncols + [caxwidth]
    height_ratios = [panelsize] * nrows
    width = sum(width_ratios) \
            * (1. + (len(width_ratios) - 1.) / len(width_ratios) * wspace)
    height = sum(height_ratios) \
             * (1. + (len(height_ratios) - 1.) / len(height_ratios) * hspace)

    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=hspace, 
                        wspace=wspace, width_ratios=width_ratios,
                        height_ratios=height_ratios)
    axes = [[fig.add_subplot(grid[i, j]) \
             for j in range(ncols)] for i in range(nrows)]
    cax = fig.add_subplot(grid[:2, ncols])
    fontsize = 12

    cmap = 'gist_yarg'
    vmin = min([min([np.min(hist[np.isfinite(hist)]) \
                     for hist in _])\
                for _ in hists])
    vmax = max([max([np.max(hist) for hist in _]) for _ in hists])

    if not np.all([np.all([key == colkeys[0] for key in colkeys])\
                   for colkeys in ykeys]):
        print('ykeys:\n', ykeys)
        raise RuntimeError('y-axis quantities differ in same row')
    if not np.all([np.all([wkeys[0][i] == colkeys[i] for i in range(ncols)])\
                   for colkeys in wkeys]):
        print('wkeys:\n', wkeys)
        raise RuntimeError('weights differ in same column')
    
    for ri in range(nrows):
        for ci in range(ncols):
            hist_raw = hists_raw[ri][ci]
            hist = hists[ri][ci]
            ykey = ykeys[ri][ci]
            wkey = wkeys[ri][ci]
            rvir = rvirs[ri][ci]
            rv = rvals[ri][ci]
            yv = yvals[ri][ci]
            ax = axes[ri][ci]
            
            yv = yv - np.log10(ynormopts[ykey])
            if ri == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                labelx = True
            else:
                labelx = False
            if ri == 0:
                ax.set_title(wlabopts[wkey], fontsize=fontsize)
            if ci == 0:
                ax.set_ylabel(ylabopts[ykey], fontsize=fontsize)
                labely = True
            else:
                labely = False
            ax.tick_params(which='both', direction='in', top=True,
                           left=True, bottom=True, right=True,
                           labelbottom=labelx, labelleft=labely,
                           labelsize=fontsize - 1.)
            img = ax.pcolormesh(rv, yv, hist.T, shading='flat', 
                                rasterized=True, cmap=cmap, vmin=vmin,
                                vmax=vmax)
            
            pv = np.array([0.1, 0.5, 0.9])
            plo, pmed, phi = pu.percentiles_from_histogram(10**hist_raw, yv, 
                                                           axis=1,
                                                           percentiles=pv)
            rc = 0.5 * (rv[1:] + rv[:-1])
            ax.plot(rc, pmed, color='red', linestyle='dotted', linewidth=1.)
            ax.plot(rc, plo, color='red', linestyle='dashed', linewidth=1.)
            ax.plot(rc, phi, color='red', linestyle='dashed', linewidth=1.)

            if markrvir:
                yp_med = pu.linterpsolve(rc, pmed, rvir)
                ax.scatter([rvir], [yp_med], marker='o', c='red', s=10)
        
        ylims = [axes[ri][i].get_ylim() for i in range(ncols)]
        ymin = min([ylim[0] for ylim in ylims])
        ymax = max([ylim[1] for ylim in ylims])
        [axes[ri][i].set_ylim((ymin, ymax)) for i in range(ncols)]
    
    plt.colorbar(img, cax=cax)
    clabel = ('$\\log_{10} \\; \\Delta(\\mathrm{weight}) \\, / \\,'
              '\\mathrm{total}\\,\\mathrm{weight} \\, / \\,'
              '\\Delta(\\mathrm{r}) \\, / \\, \\Delta(\\log_{10}\\, y)$')
    cax.set_ylabel(clabel, fontsize=fontsize)

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize)

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')
            
def plotset3dprof_weightvalgrid(fileset):
    if fileset == 'test1':
        filedir = '/Users/nastasha/ciera/profiles/fire/clean_set1_set2/'
        filen_template = ('hist_{yq}_r3D_by_{wq}_m13h206_m3e5_MHD_fire3'
                          '_fireBH_Sep182021_crdiffc690_sdp1e10'
                          '_gacc31_fa0.5_snap50_bins1_v1.hdf5')
        filen_templates = [filedir + filen_template]
        fillkw_valss = [[{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                         {'yq': 'Oxygen'},
                         ]]
        fillkw_weightss = [[{'wq': 'Mass'}]]
        outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/3dprof/'
        outnames = [outdir + 'test1_plot_3dprof.pdf']
        figtitles = ['m13h206 noBH, z=0.5']
    if fileset == 'clean_set1':
        filedir = '/Users/nastasha/ciera/profiles/fire/clean_set1_set2/'
        filen_template = ('hist_{{yq}}_r3D_by_{{wq}}_{simname}'
                          '_snap{snap}_bins1_v1.hdf5')
        simnames = [('m13h113_m3e5_MHDCRspec1_fire3_fireBH'
                     '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                     '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h113_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e-4'
                      '_gacc31_fa0.5'),
                     ('m13h113_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h206_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp3e-4'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m12f_m6e4_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp2e-4'
                      '_gacc31_fa0.5'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ]
        simlabs = ['m13h113 AGN-CR', 
                   'm13h113 AGN-noCR',
                   'm13h113 noBH',
                   'm13h206 AGN-CR',
                   'm13h206 AGN-noCR',
                   'm13h206 noBH',
                   'm12f AGN-CR',
                   'm12f AGN-noCR',
                   'm12f noBH',
                   ]
        srst = ['m13h113_m3e5', 'm13h206_m3e5', 'm12f_m6e4']
        hrst = ['m13h113_m3e4', 'm13h206_m3e4', 'm12f_m7e3']
        snaps_sr = [45, 46, 47, 48, 49, 50]
        snaps_hr = [186, 197, 210, 224, 240, 258]
        redshifts = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        simsnaplen = len(snaps_hr + snaps_hr) * len(simnames)
        fillkw_valss = [[{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                         {'yq': 'Oxygen'},
                         {'yq': 'Magnesium'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Oxygen'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Magnesium'},
                        ]
                       ] * simsnaplen
        fillkw_weightss = [[{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'H1'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'Ne8'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'O6'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'Mg10'},
                           ],
                          ] * simsnaplen
        weightsvals = ['M_V_H1', 'M_V_Ne8', 'M_V_O6', 'M_V_Mg10']
        filen_templates = \
            [filedir + filen_template.format(
                simname=sim,
                snap=snhr if np.any([sim.startswith(st) for st in hrst])\
                     else snsr if np.any([sim.startswith(st) for st in srst])\
                     else None)\
             for sim in simnames for snhr, snsr in zip(snaps_hr, snaps_sr)\
             for i in range(len(weightsvals))]
        outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/3dprof/'
        outtemplate = 'prof3D_indivdist_{wv}_{simlab}_z{z}.pdf'
        outnames = [outdir + outtemplate.format(
                        simlab=simlab.replace(' ', '_'),
                        z=f'{z:.1f}'.replace('.', 'p'),
                        wv=wv)\
                    for simlab in simlabs for z in redshifts \
                    for wv in weightsvals]
        figtitles = [f'{simlab}, z={z:.1f}'\
                     for simlab in simlabs for z in redshifts\
                     for i in range(len(weightsvals))]
    
    for filen_template, fillkw_weights, fillkw_vals, figtitle, outname \
            in zip(filen_templates, fillkw_weightss, fillkw_valss,
                   figtitles, outnames):
        print(fillkw_weights)
        print(fillkw_vals)
        plot3dprof_weightvalgrid(filen_template, fillkw_weights, fillkw_vals,
                                 figtitle=figtitle, outname=outname)


def plot3dprof_zev_weightvalgrid(filen_template, fillkw_weights, fillkw_vals,
                                 fillkw_z, figtitle=None, outname=None):
    
    rholab = ('$\\log_{10}\\,\\rho \\; '
              '[\\mathrm{H}(X=0.752) \\, \\mathrm{cm}^{-3}]$')
    tlab = '$\\log_{10}\\,\\mathrm{T} \\; [\\mathrm{K}]$'
    zolab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{O} \\;'
             ' [\\mathrm{Z}_{\\mathrm{O}, \\odot}]$')
    znelab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Ne} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Ne}, \\odot}]$')
    zmglab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Mg} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Mg}, \\odot}]$')
    ylabopts = {('sim-direct', 'Density'): rholab,
                ('sim-direct', 'Temperature'): tlab,
                ('sim-direct', 'ElementAbundance/Oxygen'): zolab,
                ('sim-direct', 'ElementAbundance/Neon'): znelab,
                ('sim-direct', 'ElementAbundance/Magnesium'): zmglab,
               }
    rhonorm = c.atomw_H * c.u / 0.752
    tnorm = 1.
    zonorm = ol.solar_abunds_ea['oxygen']
    znenorm = ol.solar_abunds_ea['neon']
    zmgnorm = ol.solar_abunds_ea['magnesium']
    ynormopts = {('sim-direct', 'Density'): rhonorm,
                 ('sim-direct', 'Temperature'): tnorm,
                 ('sim-direct', 'ElementAbundance/Oxygen'): zonorm,
                 ('sim-direct', 'ElementAbundance/Neon'): znenorm,
                 ('sim-direct', 'ElementAbundance/Magnesium'): zmgnorm,
                }
    wlabopts = {'Mass': 'Mass-weighted',
                'Volume': 'Volume-weighted',
                ('ion', 'O6'): 'O VI-weighted',
                ('ion', 'Ne8'): 'Ne VIII-weighted',
                ('ion', 'Mg10'): 'Mg X-weighted',
                ('ion', 'H1'): 'H I-weighted',
                }
    nz = len(fillkw_z)
    hists_raw = []
    rvals = []
    yvals = []
    ykeys = []
    wkeys = []
    rvirs = []
    zvals = []
    runits = None
    for vdict in fillkw_vals:
        _rvals = []
        _yvals = []
        _ykeys = []
        _wkeys = []
        _rvirs = []
        _zvals = []
        _hists_raw = []
        for wdict in fillkw_weights:
            _tdict = wdict.copy()
            _tdict.update(vdict)
            __rvals = []
            __yvals = []
            __ykeys = []
            __wkeys = []
            __rvirs = []
            __zvals = []
            __hists_raw = []
            for zdict in fillkw_z:
                __tdict = _tdict.copy()
                __tdict.update(zdict)
                filen = filen_template.format(**__tdict)
                with h5py.File(filen, 'r') as f:
                    rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                    rv = f['axis_0/bins'][:]
                    _runits = f['axis_0'].attrs['units'].decode()
                    yv = f['axis_1/bins'][:]
                    ykey = f['axis_1'].attrs['qty_type'].decode()
                    if ykey == 'sim-direct':
                        _path = 'axis_1/qty_type_args'
                        _key = 'field'
                        ykey = (ykey, f[_path].attrs[_key].decode())
                    elif ykey == 'ion':
                        _path = 'axis_1/qty_type_args'
                        _key = 'ion'
                        ykey = (ykey, f[_path].attrs[_key].decode())
                    elif ykey == 'Metal':
                        _path = 'axis_1/qty_type_args'
                        _key = 'element'
                        ykey = (ykey, f[_path].attrs[_key].decode())
                    hist_raw = f['histogram/histogram'][:]
                    wkey = f['histogram'].attrs['weight_type'].decode()
                    if wkey == 'ion':
                        _path = 'histogram/weight_type_args'
                        _key = 'ion'
                        wkey = (wkey, f[_path].attrs[_key].decode())
                    elif wkey == 'Metal':
                        _path = 'histogram/weight_type_args'
                        _key = 'element'
                        wkey = (wkey, f[_path].attrs[_key].decode())
                    redshift = f['Header/cosmopars'].attrs['z']
                    
                    # adjust for plotting purposes
                    if yv[0] == -np.inf:
                        yv[0] = 2. * yv[1] - yv[2]
                    if yv[-1] == np.inf:
                        yv[-1] = 2. * yv[-2] - yv[-3]
                    
                if runits is not None:
                    if _runits != runits:
                        raise ValueError('mismatch in r3D units')
                if _runits == 'pkpc':
                    rvir = rvir_cm / (c.cm_per_mpc * 1e-3)
                elif _runits == 'Rvir':
                    rvir = 1.
                
                __hists_raw.append(hist_raw)
                __rvals.append(rv)
                __yvals.append(yv)
                __ykeys.append(ykey)
                __wkeys.append(wkey)
                __rvirs.append(rvir)
                __zvals.append(redshift)
            _hists_raw.append(__hists_raw)
            _rvals.append(__rvals)
            _yvals.append(__yvals)
            _ykeys.append(__ykeys)
            _wkeys.append(__wkeys)
            _rvirs.append(__rvirs)
            _zvals.append(__zvals)
        hists_raw.append(_hists_raw)
        rvals.append(_rvals)
        yvals.append(_yvals)
        ykeys.append(_ykeys)
        wkeys.append(_wkeys)
        rvirs.append(_rvirs)
        zvals.append(_zvals)
    #print(ykeys)
    #print(wkeys)

    if runits == 'Rvir':
        xlabel = '$\\mathrm{r} \\; [\\mathrm{R}_{\\mathrm{vir}}]$'
        markrvir = False
    else:
        xlabel = '$\\mathrm{r} \\; [\\mathrm{pkpc}]$'
        markrvir = True

    panelsize = 2.
    caxwidth = 0.5
    ncols = len(fillkw_weights)
    nrows = len(fillkw_vals)
    wspace = 0.
    hspace = 0. 
    width_ratios = [panelsize] * ncols + [caxwidth]
    height_ratios = [panelsize] * nrows
    width = sum(width_ratios) \
            * (1. + (len(width_ratios) - 1.) / len(width_ratios) * wspace)
    height = sum(height_ratios) \
             * (1. + (len(height_ratios) - 1.) / len(height_ratios) * hspace)

    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=hspace, 
                        wspace=wspace, width_ratios=width_ratios,
                        height_ratios=height_ratios)
    axes = [[fig.add_subplot(grid[i, j]) \
             for j in range(ncols)] for i in range(nrows)]
    cax = fig.add_subplot(grid[0, ncols])
    fontsize = 12

    if not np.all([np.all([np.all([key == rowkeys[0][0] \
                                   for key in key_list])\
                           for key_list in rowkeys])\
                   for rowkeys in ykeys]):
        print('ykeys:\n', ykeys)
        raise RuntimeError('y-axis quantities differ in same row')
    if not np.all([np.all([np.all([wkeys[0][i][0] == colkeys[i][j] \
                                   for j in range(nz)])\
                           for i in range(ncols)])\
                   for colkeys in wkeys]):
        print('wkeys:\n', wkeys)
        raise RuntimeError('weights differ in same column')
    if not np.all([np.all([np.allclose(zvals[0][0], zvals[j][i])\
                           for i in range(ncols)])\
                   for j in range(nrows)]):
        print('zvals:\n', zvals)
        raise RuntimeError('redshift values differ between panels')
    
    _colors = tc.tol_cmap('rainbow_discrete', nz)
    zticks = np.linspace(0.5 / nz, 1. - 0.5 / nz, nz)
    colors = _colors(zticks)
    # zip/list comprehension issues or something when using 
    # tol_cmap outputs directly
    colors = [mcolors.to_rgb(col) for col in colors]
    
    norm = mcolors.Normalize(vmin=0., vmax=1.)
    plt.colorbar(mcm.ScalarMappable(cmap=_colors, norm=norm), cax=cax, 
                 extend='neither', orientation='vertical')
    cax.set_ylabel('redshift', fontsize=fontsize)
    cax.set_yticks(zticks)
    cax.set_yticklabels([f'{z:.1f}' for z in zvals[0][0]])
    cax.tick_params(labelsize=fontsize - 1.)

    for ri in range(nrows):
        for ci in range(ncols):
            ax = axes[ri][ci]
            ykey = ykeys[ri][ci][0] # already checked equal in panel
            wkey = wkeys[ri][ci][0] # already checked equal in panel
            _rvirs = rvirs[ri][ci]
            _rvals = rvals[ri][ci]
            _yvals = yvals[ri][ci]
            _hists_raw = hists_raw[ri][ci]
            
            _yvals = [yv - np.log10(ynormopts[ykey]) for yv in _yvals]
            if ri == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                labelx = True
            else:
                labelx = False
            if ri == 0:
                ax.set_title(wlabopts[wkey], fontsize=fontsize)
            if ci == 0:
                ax.set_ylabel(ylabopts[ykey], fontsize=fontsize)
                labely = True
            else:
                labely = False
            ax.tick_params(which='both', direction='in', top=True,
                           left=True, bottom=True, right=True,
                           labelbottom=labelx, labelleft=labely,
                           labelsize=fontsize - 1.)
            
            pv = np.array([0.1, 0.5, 0.9])
            rc_all = None
            plos = []
            pmeds = []
            phis = []
            for rvir, rv, yv, hist_raw, color in\
                    zip(_rvirs, _rvals, _yvals, _hists_raw, colors):
                plo, pmed, phi = pu.percentiles_from_histogram(10**hist_raw, 
                                     yv, axis=1, percentiles=pv)
                rc = 0.5 * (rv[1:] + rv[:-1])
                ax.plot(rc, pmed, color=color, linestyle='solid', 
                        linewidth=2.)
                ax.plot(rc, plo, color=color, linestyle='dashed', 
                        linewidth=0.7)
                ax.plot(rc, phi, color=color, linestyle='dashed', 
                        linewidth=0.7)

                if markrvir:
                    yp_med = pu.linterpsolve(rc, pmed, rvir)
                    ax.scatter([rvir], [yp_med], marker='o', c=color, s=15)
                if rc_all is None:
                    rc_all = rc
                elif not np.allclose(rc_all, rc):
                    msg = 'radial bins mismatched between redshifts'
                    raise RuntimeError(msg)
                plos.append(plo)
                pmeds.append(pmed)
                phis.append(phi)
            plos = np.array(plos)
            pmeds = np.array(pmeds)
            phis = np.array(phis)
            plo_med = np.quantile(plos, 0.5, axis=0)
            phi_med = np.quantile(phis, 0.5, axis=0)
            pmed_med = np.quantile(pmeds, 0.5, axis=0)

            ax.plot(rc_all, pmed_med, color='black', linestyle='solid', 
                    linewidth=2.)
            ax.fill_between(rc_all, plo_med, phi_med, color='black', 
                            alpha=0.3)
            ax.set_xscale('log')

        ylims = [axes[ri][i].get_ylim() for i in range(ncols)]
        ymin = min([ylim[0] for ylim in ylims])
        ymax = max([ylim[1] for ylim in ylims])
        [axes[ri][i].set_ylim((ymin, ymax)) for i in range(ncols)]

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize)

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def plotset3dprof_zev_weightvalgrid(fileset):
    if fileset == 'clean_set1-2':
        filedir = '/Users/nastasha/ciera/profiles/fire/clean_set1_set2/'
        filen_template = ('hist_{{yq}}_r3D_by_{{wq}}_{simname}'
                          '_snap{{snap}}_bins1_v1.hdf5')
        simnames = [('m13h113_m3e5_MHDCRspec1_fire3_fireBH'
                     '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                     '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h113_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e-4'
                      '_gacc31_fa0.5'),
                     ('m13h113_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h206_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp3e-4'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m12f_m6e4_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp2e-4'
                      '_gacc31_fa0.5'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ]
        simlabs = ['m13h113 AGN-CR', 
                   'm13h113 AGN-noCR',
                   'm13h113 noBH',
                   'm13h206 AGN-CR',
                   'm13h206 AGN-noCR',
                   'm13h206 noBH',
                   'm12f AGN-CR',
                   'm12f AGN-noCR',
                   'm12f noBH',
                   ]
        srst = ['m13h113_m3e5', 'm13h206_m3e5', 'm12f_m6e4']
        hrst = ['m13h113_m3e4', 'm13h206_m3e4', 'm12f_m7e3']
        snaps_sr = [50, 49, 48, 47, 46, 45]
        snaps_hr = [258, 240, 224, 210, 197, 186]
        #redshifts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        simlen = len(simnames)
        fillkw_valss = [[{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                         {'yq': 'Oxygen'},
                         {'yq': 'Magnesium'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Oxygen'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Magnesium'},
                        ]
                       ] * simlen
        fillkw_weightss = [[{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'H1'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'Ne8'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'O6'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'Mg10'},
                           ],
                          ] * simlen
        weightsvals = ['M_V_H1', 'M_V_Ne8', 'M_V_O6', 'M_V_Mg10']
        fillkw_zs = [[{'snap': snhr} \
                           if np.any([sim.startswith(st) for st in hrst])\
                       else {'snap': snsr} \
                           if np.any([sim.startswith(st) for st in srst])\
                       else None
                      for snhr, snsr in zip(snaps_hr, snaps_sr)] \
                     for sim in simnames for i in range(len(weightsvals))]
        filen_templates = \
            [filedir + filen_template.format(simname=sim)\
             for sim in simnames for i in range(len(weightsvals))]
        outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/3dprof/'
        outtemplate = 'prof3D_zev_{wv}_{simlab}.pdf'
        outnames = [outdir + outtemplate.format(
                        simlab=simlab.replace(' ', '_'),
                        wv=wv)\
                    for simlab in simlabs for wv in weightsvals]
        figtitles = [simlab for simlab in simlabs \
                     for i in range(len(weightsvals))]
    
    for filen_template, fillkw_weights, fillkw_vals, fillkw_z, figtitle, \
            outname in zip(filen_templates, fillkw_weightss, fillkw_valss,
                           fillkw_zs, figtitles, outnames):
        plot3dprof_zev_weightvalgrid(filen_template, fillkw_weights, 
                                     fillkw_vals, fillkw_z, figtitle=figtitle, 
                                     outname=outname)


def plot3dprof_wcomp_zvalgrid(filen_template, fillkw_weights, fillkw_vals,
                              fillkw_z, figtitle=None, outname=None):
    
    rholab = ('$\\log_{10}\\,\\rho \\; '
              '[\\mathrm{H}(X=0.752) \\, \\mathrm{cm}^{-3}]$')
    tlab = '$\\log_{10}\\,\\mathrm{T} \\; [\\mathrm{K}]$'
    zolab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{O} \\;'
             ' [\\mathrm{Z}_{\\mathrm{O}, \\odot}]$')
    znelab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Ne} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Ne}, \\odot}]$')
    zmglab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Mg} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Mg}, \\odot}]$')
    ylabopts = {('sim-direct', 'Density'): rholab,
                ('sim-direct', 'Temperature'): tlab,
                ('sim-direct', 'ElementAbundance/Oxygen'): zolab,
                ('sim-direct', 'ElementAbundance/Neon'): znelab,
                ('sim-direct', 'ElementAbundance/Magnesium'): zmglab,
               }
    rhonorm = c.atomw_H * c.u / 0.752
    tnorm = 1.
    zonorm = ol.solar_abunds_ea['oxygen']
    znenorm = ol.solar_abunds_ea['neon']
    zmgnorm = ol.solar_abunds_ea['magnesium']
    ynormopts = {('sim-direct', 'Density'): rhonorm,
                 ('sim-direct', 'Temperature'): tnorm,
                 ('sim-direct', 'ElementAbundance/Oxygen'): zonorm,
                 ('sim-direct', 'ElementAbundance/Neon'): znenorm,
                 ('sim-direct', 'ElementAbundance/Magnesium'): zmgnorm,
                }
    wlabopts = {'Mass': 'Mass-wtd',
                'Volume': 'Volume-wtd',
                ('ion', 'O6'): 'O VI-wtd',
                ('ion', 'Ne8'): 'Ne VIII-wtd',
                ('ion', 'Mg10'): 'Mg X-wtd',
                ('ion', 'H1'): 'H I-wtd',
                }
    cset = tc.tol_cset('vibrant')
    wcolors = {'Mass': cset.blue,
               'Volume': cset.black,
               ('ion', 'O6'): cset.orange,
               ('ion', 'Ne8'): cset.teal,
               ('ion', 'Mg10'): cset.cyan,
               ('ion', 'H1'): cset.red,
               }
    wstyles = {'Mass': 'shade',
               'Volume': 'shade',
               ('ion', 'O6'): 'lines',
               ('ion', 'Ne8'): 'lines',
               ('ion', 'Mg10'): 'lines',
               ('ion', 'H1'): 'lines',
               }
    # for different element abundances in the same plot
    ycolors = {('sim-direct', 'Density'): None,
               ('sim-direct', 'Temperature'): None,
               ('sim-direct', 'ElementAbundance/Oxygen'): None,
               ('sim-direct', 'ElementAbundance/Neon'): None,
               ('sim-direct', 'ElementAbundance/Magnesium'): None,
               }
    hists_raw = []
    rvals = []
    yvals = []
    ykeys = []
    wkeys = []
    rvirs = []
    zvals = []
    runits = None
    for vdict in fillkw_vals:
        _rvals = []
        _yvals = []
        _ykeys = []
        _wkeys = []
        _rvirs = []
        _zvals = []
        _hists_raw = []
        for zdict in fillkw_z:
            _tdict = zdict.copy()
            _tdict.update(vdict)
            __rvals = []
            __yvals = []
            __ykeys = []
            __wkeys = []
            __rvirs = []
            __zvals = []
            __hists_raw = []
            for wdict in fillkw_weights:
                __tdict = _tdict.copy()
                __tdict.update(wdict)
                filen = filen_template.format(**__tdict)
                with h5py.File(filen, 'r') as f:
                    rvir_cm = f['Header/inputpars/halodata'].attrs['Rvir_cm']
                    rv = f['axis_0/bins'][:]
                    _runits = f['axis_0'].attrs['units'].decode()
                    yv = f['axis_1/bins'][:]
                    ykey = f['axis_1'].attrs['qty_type'].decode()
                    if ykey == 'sim-direct':
                        _path = 'axis_1/qty_type_args'
                        _key = 'field'
                        ykey = (ykey, f[_path].attrs[_key].decode())
                    elif ykey == 'ion':
                        _path = 'axis_1/qty_type_args'
                        _key = 'ion'
                        ykey = (ykey, f[_path].attrs[_key].decode())
                    elif ykey == 'Metal':
                        _path = 'axis_1/qty_type_args'
                        _key = 'element'
                        ykey = (ykey, f[_path].attrs[_key].decode())
                    hist_raw = f['histogram/histogram'][:]
                    wkey = f['histogram'].attrs['weight_type'].decode()
                    if wkey == 'ion':
                        _path = 'histogram/weight_type_args'
                        _key = 'ion'
                        wkey = (wkey, f[_path].attrs[_key].decode())
                    elif wkey == 'Metal':
                        _path = 'histogram/weight_type_args'
                        _key = 'element'
                        wkey = (wkey, f[_path].attrs[_key].decode())
                    redshift = f['Header/cosmopars'].attrs['z']
                    
                    # adjust for plotting purposes
                    if yv[0] == -np.inf:
                        yv[0] = 2. * yv[1] - yv[2]
                    if yv[-1] == np.inf:
                        yv[-1] = 2. * yv[-2] - yv[-3]
                    
                if runits is not None:
                    if _runits != runits:
                        raise ValueError('mismatch in r3D units')
                if _runits == 'pkpc':
                    rvir = rvir_cm / (c.cm_per_mpc * 1e-3)
                elif _runits == 'Rvir':
                    rvir = 1.
                
                __hists_raw.append(hist_raw)
                __rvals.append(rv)
                __yvals.append(yv)
                __ykeys.append(ykey)
                __wkeys.append(wkey)
                __rvirs.append(rvir)
                __zvals.append(redshift)
            _hists_raw.append(__hists_raw)
            _rvals.append(__rvals)
            _yvals.append(__yvals)
            _ykeys.append(__ykeys)
            _wkeys.append(__wkeys)
            _rvirs.append(__rvirs)
            _zvals.append(__zvals)
        hists_raw.append(_hists_raw)
        rvals.append(_rvals)
        yvals.append(_yvals)
        ykeys.append(_ykeys)
        wkeys.append(_wkeys)
        rvirs.append(_rvirs)
        zvals.append(_zvals)
    #print(ykeys)
    #print(wkeys)

    if runits == 'Rvir':
        xlabel = '$\\mathrm{r} \\; [\\mathrm{R}_{\\mathrm{vir}}]$'
        markrvir = False
    else:
        xlabel = '$\\mathrm{r} \\; [\\mathrm{pkpc}]$'
        markrvir = True

    panelsize = 2.
    legheight = 1.5
    ncols = len(fillkw_z)
    nrows = len(fillkw_vals)
    nw = len(fillkw_weights)
    wspace = 0.
    hspace = 0. 
    width_ratios = [panelsize] * ncols
    height_ratios = [panelsize] * nrows + [legheight]
    width = sum(width_ratios) \
            * (1. + (len(width_ratios) - 1.) / len(width_ratios) * wspace)
    height = sum(height_ratios) \
             * (1. + (len(height_ratios) - 1.) / len(height_ratios) * hspace)

    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows + 1, ncols=ncols, hspace=hspace, 
                        wspace=wspace, width_ratios=width_ratios,
                        height_ratios=height_ratios)
    axes = [[fig.add_subplot(grid[i, j]) \
             for j in range(ncols)] for i in range(nrows)]
    lax = fig.add_subplot(grid[nrows, :])
    fontsize = 12

    if not np.all([np.all([np.all([key == rowkeys[0][0] \
                                   for key in key_list])\
                           for key_list in rowkeys])\
                   for rowkeys in ykeys]):
        print('ykeys:\n', ykeys)
        raise RuntimeError('y-axis quantities differ in same row')
    if not np.all([np.all([np.all([np.isclose(zvals[0][i][0], colz[i][j]) \
                                   for j in range(nw)])\
                           for i in range(ncols)])\
                   for colz in zvals]):
        print('zvals:\n', zvals)
        raise RuntimeError('redshift values differ in same column')
    if not np.all([np.all([np.all([w1 == w2 for w1, w2 in\
                                   zip(wkeys[0][0], wkeys[j][i])]) \
                           for i in range(ncols)])\
                   for j in range(nrows)]):
        print('wkeys:\n', wkeys)
        raise RuntimeError('weights differ between panels')
    
    styleargs_lines = {'med': {'linewidth': 2., 'linestyle': 'solid'},
                       'sct': {'linewidth': 1., 'linestyle': 'dashed'},
                       }
    styleargs_shade = {'med': {'linewidth': 2., 'linestyle': 'solid'},
                       'sct': {'alpha': 0.2},
                      }
    for ri in range(nrows):
        for ci in range(ncols):
            ax = axes[ri][ci]
            ykey = ykeys[ri][ci][0] # already checked equal in panel, row
            zval = zvals[ri][ci][0] # already checked equal in panel, column
            _wkeys = wkeys[ri][ci]
            _rvirs = rvirs[ri][ci]
            _rvals = rvals[ri][ci]
            _yvals = yvals[ri][ci]
            _hists_raw = hists_raw[ri][ci]
            
            _yvals = [yv - np.log10(ynormopts[ykey]) for yv in _yvals]
            if ri == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                labelx = True
            else:
                labelx = False
            if ri == 0:
                ax.set_title(f'$z = {zval:.1f}$', fontsize=fontsize)
            if ci == 0:
                ax.set_ylabel(ylabopts[ykey], fontsize=fontsize)
                labely = True
            else:
                labely = False
            ax.tick_params(which='both', direction='in', top=True,
                           left=True, bottom=True, right=True,
                           labelbottom=labelx, labelleft=labely,
                           labelsize=fontsize - 1.)
            ax.set_xscale('log')

            pv = np.array([0.1, 0.5, 0.9])
            for rvir, rv, yv, hist_raw, wkey in\
                    zip(_rvirs, _rvals, _yvals, _hists_raw, _wkeys):
                plo, pmed, phi = pu.percentiles_from_histogram(10**hist_raw, 
                                     yv, axis=1, percentiles=pv)
                pstyle = wstyles[wkey]
                color = ycolors[ykey]
                if color is None:
                    color = wcolors[wkey]
                rc = 0.5 * (rv[1:] + rv[:-1])
                if pstyle == 'lines':
                    ax.plot(rc, pmed, color=color, **styleargs_lines['med'],
                            label='median, ' + wlabopts[wkey])
                    ax.plot(rc, plo, color=color, **styleargs_lines['sct'],
                            label='10, 90%, ' + wlabopts[wkey])
                    ax.plot(rc, phi, color=color, **styleargs_lines['sct'])
                elif pstyle == 'shade':
                    ax.plot(rc, pmed, color=color, **styleargs_shade['med'],
                            label='median, ' + wlabopts[wkey])
                    ax.fill_between(rc, plo, phi, color=color, 
                                    **styleargs_shade['sct'],
                                    label='10-90%, ' + wlabopts[wkey])
                if markrvir:
                    yp_med = pu.linterpsolve(rc, pmed, rvir)
                    ax.scatter([rvir], [yp_med], marker='o', c=color, s=15)
            if ykey[0] == 'sim-direct' and \
                    ykey[1].startswith('ElementAbundance'):
                ymin, ymax = ax.get_ylim()
                ymin = max(ymin, -10.)
                ax.set_ylim(ymin, ymax)

        ylims = [axes[ri][i].get_ylim() for i in range(ncols)]
        ymin = min([ylim[0] for ylim in ylims])
        ymax = max([ylim[1] for ylim in ylims])
        [axes[ri][i].set_ylim((ymin, ymax)) for i in range(ncols)]
    handles, _ = axes[0][0].get_legend_handles_labels()
    lax.set_axis_off()
    lax.legend(handles=handles, fontsize=fontsize, 
               ncol=int(np.floor(0.67 * ncols)),
               loc='upper center', bbox_to_anchor=(0.5, 0.65))

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize)

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def plotset3dprof_zev_weightvalgrid(fileset):
    if fileset == 'clean_set1-2':
        filedir = '/Users/nastasha/ciera/profiles/fire/clean_set1_set2/'
        filen_template = ('hist_{{yq}}_r3D_by_{{wq}}_{simname}'
                          '_snap{{snap}}_bins1_v1.hdf5')
        simnames = [('m13h113_m3e5_MHDCRspec1_fire3_fireBH'
                     '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                     '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h113_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e-4'
                      '_gacc31_fa0.5'),
                     ('m13h113_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h206_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp3e-4'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m12f_m6e4_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp2e-4'
                      '_gacc31_fa0.5'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ]
        simlabs = ['m13h113 AGN-CR', 
                   'm13h113 AGN-noCR',
                   'm13h113 noBH',
                   'm13h206 AGN-CR',
                   'm13h206 AGN-noCR',
                   'm13h206 noBH',
                   'm12f AGN-CR',
                   'm12f AGN-noCR',
                   'm12f noBH',
                   ]
        srst = ['m13h113_m3e5', 'm13h206_m3e5', 'm12f_m6e4']
        hrst = ['m13h113_m3e4', 'm13h206_m3e4', 'm12f_m7e3']
        snaps_sr = [50, 49, 48, 47, 46, 45]
        snaps_hr = [258, 240, 224, 210, 197, 186]
        #redshifts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        simlen = len(simnames)
        fillkw_valss = [[{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                         {'yq': 'Oxygen'},
                         {'yq': 'Magnesium'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                         {'yq': 'Oxygen'},
                         {'yq': 'Magnesium'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Oxygen'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Magnesium'},
                        ]
                       ] * simlen
        fillkw_weightss = [[{'wq': 'Mass'},
                            {'wq': 'Volume'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'H1'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'Ne8'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'O6'},
                           ],
                           [{'wq': 'Mass'},
                            {'wq': 'Volume'},
                            {'wq': 'Mg10'},
                           ],
                          ] * simlen
        weightsvals = ['M_V', 'M_V_H1', 'M_V_Ne8', 'M_V_O6', 'M_V_Mg10']
        fillkw_zs = [[{'snap': snhr} \
                           if np.any([sim.startswith(st) for st in hrst])\
                       else {'snap': snsr} \
                           if np.any([sim.startswith(st) for st in srst])\
                       else None
                      for snhr, snsr in zip(snaps_hr, snaps_sr)] \
                     for sim in simnames for i in range(len(weightsvals))]
        filen_templates = \
            [filedir + filen_template.format(simname=sim)\
             for sim in simnames for i in range(len(weightsvals))]
        outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/3dprof/'
        outtemplate = 'prof3D_wcomp_{wv}_{simlab}.pdf'
        outnames = [outdir + outtemplate.format(
                        simlab=simlab.replace(' ', '_'),
                        wv=wv)\
                    for simlab in simlabs for wv in weightsvals]
        figtitles = [simlab for simlab in simlabs \
                     for i in range(len(weightsvals))]
    
    for filen_template, fillkw_weights, fillkw_vals, fillkw_z, figtitle, \
            outname in zip(filen_templates, fillkw_weightss, fillkw_valss,
                           fillkw_zs, figtitles, outnames):
        plot3dprof_wcomp_zvalgrid(filen_template, fillkw_weights, 
                                  fillkw_vals, fillkw_z, figtitle=figtitle, 
                                  outname=outname)
    
def plot3dprof_wcomp_weightvalgrid(filen_template, fillkw_weights, 
                                   fillkw_vals, fillkw_z, 
                                   figtitle=None, outname=None):
    '''
    fillkw_weights is a list of lists of weights:
    outer list -> columns
    inner lists -> weights in each column
    '''
    
    rholab = ('$\\log_{10}\\,\\rho \\; '
              '[\\mathrm{H}(X=0.752) \\, \\mathrm{cm}^{-3}]$')
    tlab = '$\\log_{10}\\,\\mathrm{T} \\; [\\mathrm{K}]$'
    zolab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{O} \\;'
             ' [\\mathrm{Z}_{\\mathrm{O}, \\odot}]$')
    znelab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Ne} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Ne}, \\odot}]$')
    zmglab = ('$\\log_{10}\\,\\mathrm{Z}_\\mathrm{Mg} \\;'
              ' [\\mathrm{Z}_{\\mathrm{Mg}, \\odot}]$')
    ylabopts = {('sim-direct', 'Density'): rholab,
                ('sim-direct', 'Temperature'): tlab,
                ('sim-direct', 'ElementAbundance/Oxygen'): zolab,
                ('sim-direct', 'ElementAbundance/Neon'): znelab,
                ('sim-direct', 'ElementAbundance/Magnesium'): zmglab,
               }
    rhonorm = c.atomw_H * c.u / 0.752
    tnorm = 1.
    zonorm = ol.solar_abunds_ea['oxygen']
    znenorm = ol.solar_abunds_ea['neon']
    zmgnorm = ol.solar_abunds_ea['magnesium']
    ynormopts = {('sim-direct', 'Density'): rhonorm,
                 ('sim-direct', 'Temperature'): tnorm,
                 ('sim-direct', 'ElementAbundance/Oxygen'): zonorm,
                 ('sim-direct', 'ElementAbundance/Neon'): znenorm,
                 ('sim-direct', 'ElementAbundance/Magnesium'): zmgnorm,
                }
    wlabopts = {'Mass': 'Mass-wtd',
                'Volume': 'Volume-wtd',
                ('ion', 'O6'): 'O VI-wtd',
                ('ion', 'Ne8'): 'Ne VIII-wtd',
                ('ion', 'Mg10'): 'Mg X-wtd',
                ('ion', 'H1'): 'H I-wtd',
                }
    cset = tc.tol_cset('vibrant')
    wcolors = {'Mass': cset.blue,
               'Volume': cset.black,
               ('ion', 'O6'): cset.orange,
               ('ion', 'Ne8'): cset.teal,
               ('ion', 'Mg10'): cset.cyan,
               ('ion', 'H1'): cset.red,
               }
    wstyles = {'Mass': 'shade',
               'Volume': 'shade',
               ('ion', 'O6'): 'lines',
               ('ion', 'Ne8'): 'lines',
               ('ion', 'Mg10'): 'lines',
               ('ion', 'H1'): 'lines',
               }
    # for different element abundances in the same plot
    ycolors = {('sim-direct', 'Density'): None,
               ('sim-direct', 'Temperature'): None,
               ('sim-direct', 'ElementAbundance/Oxygen'): None,
               ('sim-direct', 'ElementAbundance/Neon'): None,
               ('sim-direct', 'ElementAbundance/Magnesium'): None,
               }
    hists_raw = []
    rvals = []
    yvals = []
    ykeys = []
    wkeys = []
    rvirs = []
    zvals = []
    runits = None
    for vdict in fillkw_vals:
        _rvals = []
        _yvals = []
        _ykeys = []
        _wkeys = []
        _rvirs = []
        _zvals = []
        _hists_raw = []
        for wdict_list in fillkw_weights:
            __rvals = []
            __yvals = []
            __ykeys = []
            __wkeys = []
            __rvirs = []
            __zvals = []
            __hists_raw = []
            for wdict in wdict_list:
                _tdict = wdict.copy()
                _tdict.update(vdict)
                ___rvals = []
                ___yvals = []
                ___ykeys = []
                ___wkeys = []
                ___rvirs = []
                ___zvals = []
                ___hists_raw = []
                for zdict in fillkw_z:
                    __tdict = _tdict.copy()
                    __tdict.update(zdict)
                    __tdict.update(wdict)
                    filen = filen_template.format(**__tdict)
                    with h5py.File(filen, 'r') as f:
                        hdatpath = 'Header/inputpars/halodata'
                        rvir_cm = f[hdatpath].attrs['Rvir_cm']
                        rv = f['axis_0/bins'][:]
                        _runits = f['axis_0'].attrs['units'].decode()
                        yv = f['axis_1/bins'][:]
                        ykey = f['axis_1'].attrs['qty_type'].decode()
                        if ykey == 'sim-direct':
                            _path = 'axis_1/qty_type_args'
                            _key = 'field'
                            ykey = (ykey, f[_path].attrs[_key].decode())
                        elif ykey == 'ion':
                            _path = 'axis_1/qty_type_args'
                            _key = 'ion'
                            ykey = (ykey, f[_path].attrs[_key].decode())
                        elif ykey == 'Metal':
                            _path = 'axis_1/qty_type_args'
                            _key = 'element'
                            ykey = (ykey, f[_path].attrs[_key].decode())
                        hist_raw = f['histogram/histogram'][:]
                        wkey = f['histogram'].attrs['weight_type'].decode()
                        if wkey == 'ion':
                            _path = 'histogram/weight_type_args'
                            _key = 'ion'
                            wkey = (wkey, f[_path].attrs[_key].decode())
                        elif wkey == 'Metal':
                            _path = 'histogram/weight_type_args'
                            _key = 'element'
                            wkey = (wkey, f[_path].attrs[_key].decode())
                        redshift = f['Header/cosmopars'].attrs['z']
                        
                        # adjust for plotting purposes
                        if yv[0] == -np.inf:
                            yv[0] = 2. * yv[1] - yv[2]
                        if yv[-1] == np.inf:
                            yv[-1] = 2. * yv[-2] - yv[-3]
                        
                        if runits is not None:
                            if _runits != runits:
                                raise ValueError('mismatch in r3D units')
                        if _runits == 'pkpc':
                            rvir = rvir_cm / (c.cm_per_mpc * 1e-3)
                        elif _runits == 'Rvir':
                            rvir = 1.
                    ___hists_raw.append(hist_raw)
                    ___rvals.append(rv)
                    ___yvals.append(yv)
                    ___ykeys.append(ykey)
                    ___wkeys.append(wkey)
                    ___rvirs.append(rvir)
                    ___zvals.append(redshift)
                __hists_raw.append(___hists_raw)
                __rvals.append(___rvals)
                __yvals.append(___yvals)
                __ykeys.append(___ykeys)
                __wkeys.append(___wkeys)
                __rvirs.append(___rvirs)
                __zvals.append(___zvals)
            _hists_raw.append(__hists_raw)
            _rvals.append(__rvals)
            _yvals.append(__yvals)
            _ykeys.append(__ykeys)
            _wkeys.append(__wkeys)
            _rvirs.append(__rvirs)
            _zvals.append(__zvals)
        hists_raw.append(_hists_raw)
        rvals.append(_rvals)
        yvals.append(_yvals)
        ykeys.append(_ykeys)
        wkeys.append(_wkeys)
        rvirs.append(_rvirs)
        zvals.append(_zvals)
    #print(ykeys)
    #print(wkeys)

    if runits == 'Rvir':
        xlabel = '$\\mathrm{r} \\; [\\mathrm{R}_{\\mathrm{vir}}]$'
        markrvir = False
    else:
        xlabel = '$\\mathrm{r} \\; [\\mathrm{pkpc}]$'
        markrvir = True

    panelsize = 2.
    legheight = 2.
    ncols = len(fillkw_weights)
    nrows = len(fillkw_vals)
    wspace = 0.
    hspace = 0. 
    width_ratios = [panelsize] * ncols
    height_ratios = [panelsize] * nrows + [legheight]
    width = sum(width_ratios) \
            * (1. + (len(width_ratios) - 1.) / len(width_ratios) * wspace)
    height = sum(height_ratios) \
             * (1. + (len(height_ratios) - 1.) / len(height_ratios) * hspace)

    fig = plt.figure(figsize=(width, height))
    grid = gsp.GridSpec(nrows=nrows + 1, ncols=ncols, hspace=hspace, 
                        wspace=wspace, width_ratios=width_ratios,
                        height_ratios=height_ratios)
    axes = [[fig.add_subplot(grid[i, j]) \
             for j in range(ncols)] for i in range(nrows)]
    lax = fig.add_subplot(grid[nrows, :])
    fontsize = 12

    if not np.all([np.all([np.all([np.all([key == rowkeys[0][0][0] \
                                           for key in _key_list])\
                                   for _key_list in key_list])\
                           for key_list in rowkeys])\
                   for rowkeys in ykeys]):
        print('ykeys:\n', ykeys)
        raise RuntimeError('y-axis quantities differ in same row')
    if not np.all([np.all([np.all([np.all([_w1 == _w2 for _w1, _w2 in \
                                           zip(w1, w2)]) \
                                   for w1, w2 in\
                                   zip(wkeys[0][i], wkeys[j][i])]) \
                           for i in range(ncols)])\
                   for j in range(nrows)]):
        print('wkeys:\n', wkeys)
        raise RuntimeError('weights differ in same column')
    if not np.all([np.all([np.all([np.all([np.isclose(zvals[0][0][0][zi], 
                                                      zvals[i][j][wi][zi]) \
                                           for zi in \
                                           range(len(zvals[i][j][wi]))])\
                                   for wi in range(len(zvals[i][j]))])\
                           for j in range(ncols)])\
                   for i in range(nrows)]):
        print('zvals:\n', zvals)
        raise RuntimeError('redshift values between panels or weights')
    
    styleargs_lines = {'med': {'linewidth': 2., 'linestyle': 'solid'},
                       'sct': {'linewidth': 1., 'linestyle': 'dashed'},
                       'sct_med': {'elinewidth': 1., 'linestyle': 'none'},
                       }
    styleargs_shade = {'med': {'linewidth': 2., 'linestyle': 'solid'},
                       'sct': {'alpha': 0.2},
                       'sct_med': {'elinewidth': 1., 'linestyle': 'none'},
                      }
    wkeys_unique = [key for kl1 in wkeys for kl2 in kl1 for kl3 in kl2\
                    for key in kl3]
    wkeys_unique = list(set(wkeys_unique))
    linedone_wkey = {wkey: [False] * 3 for wkey in wkeys_unique}
    for ri in range(nrows):
        for ci in range(ncols):
            ax = axes[ri][ci]
            ykey = ykeys[ri][ci][0][0] # already checked equal in panel, row
            zval = zvals[ri][ci][0] # already checked equal in panels
            _wkeys = wkeys[ri][ci]
            _rvirs = rvirs[ri][ci]
            _rvals = rvals[ri][ci]
            _yvals = yvals[ri][ci]
            _hists_raw = hists_raw[ri][ci]
            
            _yvals = [yv - np.log10(ynormopts[ykey]) for yv in _yvals]
            if ri == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=fontsize)
                labelx = True
            else:
                labelx = False
            if ci == 0:
                ax.set_ylabel(ylabopts[ykey], fontsize=fontsize)
                labely = True
            else:
                labely = False
            ax.tick_params(which='both', direction='in', top=True,
                           left=True, bottom=True, right=True,
                           labelbottom=labelx, labelleft=labely,
                           labelsize=fontsize - 1.)
            ax.set_xscale('log')
            ax.grid(visible=True)

            pv = np.array([0.1, 0.5, 0.9])
            zlab = f'$z={min(zval):.1f} \\endash {max(zval):.1f}$'
            rvir_min = np.inf
            rvir_max = -np.inf
            rc_all = None
            nw = len(_wkeys)
            for wi, (__rvir, __rv, __yv, __hist_raw, __wkey) in\
                    enumerate(zip(_rvirs, _rvals, _yvals, _hists_raw, 
                                  _wkeys)):
                plos = []
                phis = []
                pmeds = []
                rc_all = None
                for rvir, rv, yv, hist_raw, wkey in \
                        zip(__rvir, __rv, __yv, __hist_raw, __wkey):
                    rvir_min = min(rvir_min, rvir)
                    rvir_max = max(rvir_max, rvir)
                    plo, pmed, phi = pu.percentiles_from_histogram(
                        10**hist_raw, yv, axis=1, percentiles=pv)
                    plos.append(plo)
                    phis.append(phi)
                    pmeds.append(pmed)
                    rc = 0.5 * (rv[1:] + rv[:-1])
                    if rc_all is None:
                        rc_all = rc
                    elif not np.allclose(rc_all, rc):
                        msg = 'r values mismatched in same panel, weight'
                        raise RuntimeError(msg)
                plos = np.array(plos)
                phis = np.array(phis)
                pmeds = np.array(pmeds)
                
                medmed = np.median(pmeds, axis=0)
                maxmed = np.max(pmeds, axis=0)
                minmed = np.min(pmeds, axis=0)
                mscthi = np.median(phis, axis=0)
                msctlo = np.median(plos, axis=0)
                deltahi = maxmed - medmed
                deltalo = medmed - minmed

                pstyle = wstyles[wkey]
                color = ycolors[ykey]
                if color is None:
                    color = wcolors[wkey]
                medlab = 'med. of ' + zlab + ' med., ' + wlabopts[wkey]
                sctlab_lines = 'med. of ' + zlab + ' 10, 90%, ' +\
                                wlabopts[wkey]
                sctlab_shade = 'med. of ' + zlab + ' 10-90%, ' +\
                                wlabopts[wkey]
                sct_medlab = zlab + ' med. range, ' + wlabopts[wkey]
                if pstyle == 'lines':
                    if linedone_wkey[wkey][0]:
                        label = None
                    else:
                        label = medlab
                        linedone_wkey[wkey][0] = True
                    ax.plot(rc_all, medmed, color=color, label=label,
                            **styleargs_lines['med'])
                    if linedone_wkey[wkey][1]:
                        label = None
                    else:
                        label = sctlab_lines
                        linedone_wkey[wkey][1] = True
                    ax.plot(rc_all, mscthi, color=color, label=label,
                            **styleargs_lines['sct'])
                    ax.plot(rc_all, msctlo, color=color, 
                            **styleargs_lines['sct'])
                    if linedone_wkey[wkey][2]:
                        label = None
                    else:
                        label = sct_medlab
                        linedone_wkey[wkey][2] = True
                    ax.errorbar(rc_all, medmed, color=color, label=label,
                                errorevery=(wi, nw), xerr=None, 
                                yerr=(deltalo, deltahi),
                                **styleargs_lines['sct_med'])
                elif pstyle == 'shade':
                    if linedone_wkey[wkey][0]:
                        label = None
                    else:
                        label = medlab
                        linedone_wkey[wkey][0] = True
                    ax.plot(rc, medmed, color=color, label=label,
                            **styleargs_shade['med'])
                    if linedone_wkey[wkey][1]:
                        label = None
                    else:
                        label = sctlab_shade
                        linedone_wkey[wkey][1] = True
                    ax.fill_between(rc, mscthi, msctlo, color=color, 
                                    **styleargs_shade['sct'],
                                    label=label)
                    if linedone_wkey[wkey][2]:
                        label = None
                    else:
                        label = sct_medlab
                        linedone_wkey[wkey][2] = True
                    ax.errorbar(rc_all, medmed, color=color, label=label,
                                errorevery=(wi, nw), xerr=None, 
                                yerr=(deltalo, deltahi),
                                **styleargs_shade['sct_med'])
                if markrvir:
                    yp_med = pu.linterpsolve(rc, medmed, rvir)
                    ax.scatter([rvir], [yp_med], marker='o', c=color, s=15)
            if ykey[0][0] == 'sim-direct' and \
                    ykey[0][1].startswith('ElementAbundance'):
                ymin, ymax = ax.get_ylim()
                ymin = max(ymin, -10.)
                ax.set_ylim(ymin, ymax)

        ylims = [axes[ri][i].get_ylim() for i in range(ncols)]
        ymin = min([ylim[0] for ylim in ylims])
        ymax = max([ylim[1] for ylim in ylims])
        [axes[ri][i].set_ylim((ymin, ymax)) for i in range(ncols)]
    handles = []
    for j in range(ncols):
        _handles, _ = axes[0][j].get_legend_handles_labels()
        handles = handles + _handles
    lax.set_axis_off()
    lax.legend(handles=handles, fontsize=fontsize - 1., 
               ncol=int(np.floor(0.5 * ncols)),
               loc='upper center', bbox_to_anchor=(0.5, 0.65))

    if figtitle is not None:
        fig.suptitle(figtitle, fontsize=fontsize)

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight')

def plotset3dprof_wcomp_weightvalgrid(fileset):
    if fileset == 'clean_set1-2':
        filedir = '/Users/nastasha/ciera/profiles/fire/clean_set1_set2/'
        filen_template = ('hist_{{yq}}_r3D_by_{{wq}}_{simname}'
                          '_snap{{snap}}_bins1_v1.hdf5')
        simnames = [('m13h113_m3e5_MHDCRspec1_fire3_fireBH'
                     '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                     '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h113_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e-4'
                      '_gacc31_fa0.5'),
                     ('m13h113_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m13h206_m3e4_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp3e-4'
                      '_gacc31_fa0.5'),
                     ('m13h206_m3e5_MHD_fire3_fireBH'
                      '_Sep182021_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ('m12f_m6e4_MHDCRspec1_fire3_fireBH'
                      '_fireCR1_Oct252021_crdiffc1_sdp1e-4'
                      '_gacc31_fa0.5_fcr1e-3_vw3000'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp2e-4'
                      '_gacc31_fa0.5'),
                     ('m12f_m7e3_MHD_fire3_fireBH'
                      '_Sep182021_hr_crdiffc690_sdp1e10'
                      '_gacc31_fa0.5'),
                     ]
        simlabs = ['m13h113 AGN-CR', 
                   'm13h113 AGN-noCR',
                   'm13h113 noBH',
                   'm13h206 AGN-CR',
                   'm13h206 AGN-noCR',
                   'm13h206 noBH',
                   'm12f AGN-CR',
                   'm12f AGN-noCR',
                   'm12f noBH',
                   ]
        srst = ['m13h113_m3e5', 'm13h206_m3e5', 'm12f_m6e4']
        hrst = ['m13h113_m3e4', 'm13h206_m3e4', 'm12f_m7e3']
        snaps_sr = [50, 49, 48, 47, 46, 45]
        snaps_hr = [258, 240, 224, 210, 197, 186]
        #redshifts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        simlen = len(simnames)
        fillkw_valss = [[{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                         {'yq': 'Oxygen'},
                         {'yq': 'Magnesium'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Neon'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Magnesium'},
                        ],
                        [{'yq': 'Density'},
                         {'yq': 'Temperature'},
                         {'yq': 'Oxygen'},
                        ]
                       ] * simlen
        fillkw_weightss = [[[{'wq': 'Mass'},
                             {'wq': 'Volume'},
                            ],
                            [{'wq': 'Mass'},
                             {'wq': 'Volume'},
                             {'wq': 'H1'},
                            ],
                           ], 
                           [[{'wq': 'Mass'},
                             {'wq': 'Volume'}
                            ],
                            [{'wq': 'Volume'},
                             {'wq': 'Ne8'},
                            ],
                           ],
                           [[{'wq': 'Mass'},
                             {'wq': 'Volume'}
                            ],
                            [{'wq': 'Volume'},
                             {'wq': 'Mg10'},
                            ],
                           ],
                           [[{'wq': 'Mass'},
                             {'wq': 'Volume'}
                            ],
                            [{'wq': 'Volume'},
                             {'wq': 'O6'},
                            ],
                           ],
                          ] * simlen
        weightsvals = ['M_V_H1', 'M_V_Ne8', 'M_V_O6', 'M_V_Mg10']
        fillkw_zs = [[{'snap': snhr} \
                           if np.any([sim.startswith(st) for st in hrst])\
                       else {'snap': snsr} \
                           if np.any([sim.startswith(st) for st in srst])\
                       else None
                      for snhr, snsr in zip(snaps_hr, snaps_sr)] \
                     for sim in simnames for i in range(len(weightsvals))]
        filen_templates = \
            [filedir + filen_template.format(simname=sim)\
             for sim in simnames for i in range(len(weightsvals))]
        outdir = '/Users/nastasha/ciera/projects_lead/fire3_ionabs/3dprof/'
        outtemplate = 'prof3D_wcomp_zcomb_{wv}_{simlab}.pdf'
        outnames = [outdir + outtemplate.format(
                        simlab=simlab.replace(' ', '_'),
                        wv=wv)\
                    for simlab in simlabs for wv in weightsvals]
        figtitles = [simlab for simlab in simlabs \
                     for i in range(len(weightsvals))]
    
    for filen_template, fillkw_weights, fillkw_vals, fillkw_z, figtitle, \
            outname in zip(filen_templates, fillkw_weightss, fillkw_valss,
                           fillkw_zs, figtitles, outnames):
        plot3dprof_wcomp_weightvalgrid(filen_template, fillkw_weights, 
                                  fillkw_vals, fillkw_z, figtitle=figtitle, 
                                  outname=outname)


