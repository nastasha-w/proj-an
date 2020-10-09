#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: wijers

calculate the nearest-neighbour galaxy distribution
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

halocat =  '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5'
mmin = 12.0
mmax = 12.5
mmin_all = 11.0 # ignore lower-mass haloes entirely

mdir = '/net/luttero/data2/imgs/jukka-toni/'

def diffmatrix(pos1, pos2, wrap=None):
    '''
    returns matrix of pos1 - pos2, including periodic boundary
    effects if wrap is not None
    
    parameters:
    -----------
    pos1:  positions (numpy array, 1d)
    pos2:  positions (same units as pos1, numpy array, 1d)
    wrap:  period (float), or None for no wrapping

    returns:
    --------
    numpy array (2d): axis 0 -> pos1 varies
                      axis 1 -> pos2 varies
    '''
    diff = pos1[:, np.newaxis] - pos2[np.newaxis, :]
    if wrap is not None:
        diff += 0.5 * wrap
        diff %= wrap
        diff -= 0.5 * wrap
    return diff

with h5py.File(halocat, 'r') as df:
    logM = np.log10(df['M200c_Msun'][:]) 
    xpos = df['Xcom_cMpc'][:]
    ypos = df['Ycom_cMpc'][:]
    zpos = df['Zcom_cMpc'][:]

    cosmopars = {key: val for key, val in df['Header/cosmopars'].attrs.items()}
    rvir = df['R200c_pkpc'][:] * 1e-3 / cosmopars['a']
    wrap = cosmopars['boxsize'] / cosmopars['h']

# apply global selection
_sel = logM >= mmin_all
xpos = xpos[_sel]
ypos = ypos[_sel]
zpos = zpos[_sel]
rvir = rvir[_sel]
logM = logM[_sel]
del _sel

# get separations
msel = np.logical_and(logM >= mmin, logM < mmax)
xd = diffmatrix(xpos[msel], xpos, wrap=wrap)
yd = diffmatrix(ypos[msel], ypos, wrap=wrap)
zd = diffmatrix(zpos[msel], zpos, wrap=wrap)
diff = np.sqrt(xd**2 + yd**2 + zd**2)
diff_zproj = np.sqrt(xd**2 + yd**2)
diff_yproj = np.sqrt(xd**2 + zd**2)
diff_xproj = np.sqrt(yd**2 + zd**2)

# same galaxy -> set difference to np.inf so minima aren't 'contaminated'
diff_zproj[diff == 0.] = np.inf
diff_yproj[diff == 0.] = np.inf
diff_xproj[diff == 0.] = np.inf
diff[diff == 0.] = np.inf

def plotdiffs():
    fig, axes = plt.subplots(ncols=2, nrows=2,\
                gridspec_kw={'hspace': 0.35, 'wspace': 0.35, 'top': 0.9})
    fontsize = 10
    minsep_proj = 0.0

    title = 'Nearest haloes to $\\log_{{10}} \\mathrm{{M}}\\,/\\, \\mathrm{{M}}_{{\\odot}} = {mmin:.1f} \\endash {mmax:.1f}$ haloes\nin a {boxsize:.1f} cMpc EAGLE volume, $z={z:.1f}$'
    pdflabel = '$\\mathrm{d} \\mathrm{N} \\,/\\, \\mathrm{d} \\log_{10} \\mathrm{D}$'
    fig.suptitle(title.format(mmin=mmin, mmax=mmax, boxsize=wrap,\
                              z=cosmopars['z']),\
                 fontsize=fontsize)
    lkwargs = {'mmin': mmin, 'mmax': mmax, 'mmin_all': mmin_all}
    sel1 = {'sel': msel, 'label': '${mmin:.1f} \\endash {mmax:.1f}$'.format(**lkwargs)}
    sel2 = {'sel': logM >= mmin,\
            'label': '$\\geq {mmin:.1f}$'.format(**lkwargs)}
    sel3 = {'sel': logM >= mmin_all,\
            'label': '$\\geq {mmin_all:.1f}$'.format(**lkwargs)}
    sels = [sel1, sel2, sel3]

    ax = axes[0, 0]
    for sel in sels:
        nearest = np.min(diff[:, sel['sel']], axis=1)
        _min = np.log10(np.min(nearest))
        _max = np.log10(np.max(nearest))
        bins = np.logspace(_min * 0.99, _max * 1.01, 20)
        hist, bins = np.histogram(nearest, bins=bins)
        hist = hist / (np.sum(hist) * np.diff(np.log10(bins)))
        hist = np.append(hist, 0.)
        ax.step(bins, hist, label=sel['label'])
    ax.set_xlabel('nearest halo distance [cMpc]',\
                  fontsize=fontsize)
    ax.set_ylabel(pdflabel, fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=fontsize)
        
    ax = axes[0, 1]
    for sel in sels:
        nearest = np.min(diff[:, sel['sel']], axis=1)
        nearest /= rvir[msel]
        _min = np.log10(np.min(nearest))
        _max = np.log10(np.max(nearest))
        bins = np.logspace(_min * 0.99, _max * 1.01, 20)
        hist, bins = np.histogram(nearest, bins=bins)
        hist = hist / (np.sum(hist) * np.diff(np.log10(bins)))
        hist = np.append(hist, 0.)
        ax.step(bins, hist, label=sel['label'])
    ax.set_xlabel('nearest halo distance [$\\mathrm{R}_{\\mathrm{200c}}$]', fontsize=fontsize)
    ax.set_ylabel(pdflabel, fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.legend(fontsize=fontsize)

    ax = axes[1, 0]
    for si, sel in enumerate(sels):
        for sample, label, ls in zip([diff_xproj, diff_yproj, diff_zproj],\
                                     ['x-axis', 'y-axis', 'z-axis'],\
                                     ['solid', 'dashed', 'dotted']):
            nearest = np.min(sample[:, sel['sel']], axis=1)
            _min = np.log10(np.min(nearest))
            _max = np.log10(np.max(nearest))
            if _min < np.log10(minsep_proj):
                _min = np.log10(minsep_proj)
                nearest[nearest < minsep_proj] = minsep_proj
            bins = np.logspace(_min * 0.99, _max * 1.01, 20)
            hist, bins = np.histogram(nearest, bins=bins)
            hist = hist / (np.sum(hist) * np.diff(np.log10(bins)))
            hist = np.append(hist, 0.)
            _label = label if si == 0 else None
            ax.step(bins, hist, label=_label, linestyle=ls,\
                    color='C{}'.format(si))
    ax.set_xlabel('nearest projected halo [cMpc]',\
                  fontsize=fontsize)
    ax.set_ylabel(pdflabel, fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=fontsize)

    ax = axes[1, 1]
    for si, sel in enumerate(sels):
        for sample, label, ls in zip([diff_xproj, diff_yproj, diff_zproj],\
                                     ['x-axis', 'y-axis', 'z-axis'],\
                                     ['solid', 'dashed', 'dotted']):
            nearest = np.min(sample[:, sel['sel']], axis=1)
            nearest /= rvir[msel]
            _min = np.log10(np.min(nearest))
            _max = np.log10(np.max(nearest))
            if _min < np.log10(minsep_proj):
                _min = np.log10(minsep_proj)
                nearest[nearest < minsep_proj] = minsep_proj
            bins = np.logspace(_min * 0.99, _max * 1.01, 20)
            hist, bins = np.histogram(nearest, bins=bins)
            hist = hist / (np.sum(hist) * np.diff(np.log10(bins)))
            hist = np.append(hist, 0.)
            _label = label if si == 0 else None
            ax.step(bins, hist, label=_label, linestyle=ls,\
                    color='C{}'.format(si))
    ax.set_xlabel('nearest projected halo [$\\mathrm{R}_{\\mathrm{200c}}$]',\
                  fontsize=fontsize)
    ax.set_ylabel(pdflabel, fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.legend(fontsize=fontsize)

    axargs = {'direction': 'in', 'labelsize': fontsize - 1,\
              'which': 'both', 'top': True, 'right': True}
    [[ax.tick_params(**axargs) for ax in _] for _ in axes]

    outname = 'galaxy_separations_from_{mmin:.1f}-{mmax:.1f}_to_{mmin_all:.1f}.eps'
    outname = outname.format(**lkwargs)
    plt.savefig(mdir + outname, format='eps')

if __name__ == '__main__':
    plotdiffs()    

