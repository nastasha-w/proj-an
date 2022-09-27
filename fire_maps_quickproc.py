#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.patches as mpatch
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe

import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu

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
