#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
plot or do quick checks on results from fire_maps
'''

from unicodedata import normalize
import numpy as np
import h5py
import pandas as pd

import tol_colors as tc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.patches as mpatch
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe

import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu


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
        rvir_ckpcoverh = f['Header/inputpars/halodata'].attrs['Rvir_ckpcoverh']
        xax = f['Header/inputpars'].attrs['Axis1']
        yax = f['Header/inputpars'].attrs['Axis2']
        rvir_pkpc = rvir_ckpcoverh * cosmopars['a'] / cosmopars['h']
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
    _imgname = 'ionbal-test_{ion}_depletion-False_Z-0.01_snap045_{simname}.pdf'

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
        
        outname = imgdir + _imgname.format(sim=simlabel, ion=ion)
            
        title = '{ion} PS20 table at z={z:.2f} vs. interp., dust depl. {dep}' 
        title = title.format(ion=ion, dep=ps20depletion, z=redshift)
        
        fig = plt.figure(figsize=(11., 4.))
        grid = gsp.GridSpec(nrows=1, ncols=5, hspace=0.0, wspace=0.3, 
                        width_ratios=[1., 0.1, 1., 0.1, 1.])
        axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
        fontsize = 12
        cmap = 'viridis'
        size = 20.
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
                   edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel('$\\log \\, \\mathrm{T} \\; [\\mathrm{K]}$', 
                      fontsize=fontsize)
        ylabel = '$\\log \\, \\mathrm{n}_{\\mathrm{H}} \\;' + \
                 ' [\\mathrm{cm}^{-3}]$'
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(which='both', axis='both', labelsize=fontsize - 1)
        cax.tick_params(labelsize=fontsize - 1)

        
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
                   label='$\\Delta$ table grid')
        ax.scatter(xvals_tab1, -1. * delta_tab1, s=0.5*size, color='black')
        ax.scatter(xvals_tab2, delta_tab2, s=0.5*size, color='black')
        ax.scatter(xvals_tab2, -1. * delta_tab2, s=0.5*size, color='black')

        img = ax.scatter(xvals_sim, delta_sim, s=size, c=dz,
                         edgecolor='black', cmap=cmap, vmin=vmin, vmax=vmax,
                         label='sim - table')
        plt.colorbar(img, cax=cax, extend='neither')
        ax.set_ylabel('difference with nearest table value', 
                      fontsize=fontsize)
        cax.set_ylabel('simulation Z - table Z', fontsize=fontsize)
        ax.legend(fontsize=fontsize)

        ax = axes[4]

        nbins = 100
        maxv = np.max(np.abs(delta_tab1))
        maxv = max(maxv, np.abs(delta_tab2))
        maxv = max(maxv, np.abs(delta_sim))
        bins = np.linspace(-1. * maxv, maxv, nbins)
        
        tabvals = np.append(delta_tab1, -1. * delta_tab1)
        tabvals = np.append(tabvals, delta_tab2)
        tabvals = np.append(tabvals, -1. * delta_tab2)
        ax.hist(tabvals, bins=bins, log=True, histtype='step', color='black',
                label='$\\Delta$ table grid', align='mid', density=True)
        ax.hist(tabvals, bins=bins, log=True, histtype='step', color='blue',
                label='sim - table', linestyle='dashed', align='mid', 
                density=True)

        ax.set_xlabel('difference with nearest table value', 
                      fontsize=fontsize)
        ax.set_ylabel('probability density', fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        
        plt.savefig(outname, bbox_inches='tight')
    


        




            

def run_ionbal_tests(index):
    # laptop
    ddir = '/Users/nastasha/ciera/tests/fire_start/ionbal_tests/'
    simname = 'm13h206_m3e5__m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1'+\
              '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    pre = 'ionbal_test_PS20'
    ftmp = ['{pre}_{ion}_depletion-False_Z-0.01_snap045_{simname}.hdf5',
            '{pre}_{ion}_depletion-False_Z-0.0001_snap027_{simname}.hdf5',
            '{pre}_{ion}_depletion-False_Z-0.01_snap045_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.0001_snap027_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.01_snap027_{simname}.hdf5',
            '{pre}_{ion}_depletion-True_Z-0.01_snap045_{simname}.hdf5',
            ]
    ions = ['O{}'.format(i) for i in range(1, 10)]
    filens = [ddir + ftmp[index].format(pre=pre, simname=simname, ion=ion) \
              for ion in ions]
    ionbal_test(filens)