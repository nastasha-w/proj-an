#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
plot or do quick checks on results from fire_maps
'''

import numpy as np
import h5py
import pandas as pd
import os

import tol_colors as tc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.patches as mpatch
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe

import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu


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
    
        





    

