#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from xml.dom.expatbuilder import CDATA_SECTION_NODE
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

    fig = plt.figure(figsize=(11., 3.))
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
    ylabel = 'log abs ([M(< Rvir)] - [AHF M]) / [AHF M]'
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
        map = f['map']
        vmin = f['map'].attrs['minfinite']
        vmax = f['map'].attrs['max']

        box_cm = f['Header/inputpars'].attrs['diameter_used_cm']
        cosmopars = {key: val for key, val in \
                     f['Header/inputpars/cosmopars'].items()}
        rvir_ckpc = f['Header/inputpars/halodata'].attrs['Rvir_ckpc']
        xax = f['Header/inputpars'].attrs['Axis1']
        yax = f['Header/inputpars'].attrs['Axis2']
        rvir_pkpc = rvir_ckpc * cosmopars['a']
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
    ax.text(2**-0.5 * rvir_pkpc, 2**-0.5 * rvir_pkpc, '$R_{\\mathrm{vir}}$',
            color='red', fontsize=fontsize)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
