#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd

import matplotlib.pyplot as plt

import tol_colors as tc
import eagle_constants_and_units as c


wdir = '/Users/nastasha/ciera/projects_lead/tdist_groups/data/'
mdir = '/Users/nastasha/ciera/projects_lead/tdist_groups/imgs/'
min_mass_msun = 1e13

fndir_laptop = '/Users/nastasha/phd/data/paper3/3dprof/'
fn_halo = 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
fn_mass = 'filenames_L0100N1504_27_Mh0p5dex_1000_Mass_Trprof.txt'
fn_vol = 'filenames_L0100N1504_27_Mh0p5dex_1000_Volume_Trprof.txt'

# run on laptop, copy from quasar
# xargs -a list.txt scp -t new_folder
def make_copylist_groups():
    copyfiles = []
    copylistn = wdir + 'copylist_trdist_groups_quasar.txt'

    halodat = pd.read_csv(fndir_laptop + fn_halo, header=2, index_col='galaxyid', sep='\t')
    gids = halodat.index[halodat['M200c_Msun'] >= min_mass_msun]
    gids = np.array(gids)
    massfn = pd.read_csv(fndir_laptop + fn_mass, sep='\t', index_col='galaxyid')
    copyfiles += list(massfn['filename'][gids])
    volfn = pd.read_csv(fndir_laptop + fn_vol, sep='\t', index_col='galaxyid')
    copyfiles += list(volfn['filename'][gids])

    print('directory: ', '/'.join(copyfiles[0].split('/')[:-1]))
    copyfiles = [filen.split('/')[-1] for filen in copyfiles]
    outdat = '\n'.join(copyfiles)
    text_file = open(copylistn, "w")
    text_file.write(outdat)
    text_file.close()

# R200c bins I used: [0.   0.01 0.02 0.05 0.1  0.15 0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.   1.25 1.5  2.   2.5  3.   3.5  4.  ]
def plotdist_group(galid, ax=None, weight='mass', inclsf=False, r200cranges=[(0.1, 1.), (0.1, 0.2), (0.4, 0.6), (0.9, 1.)],
                   savename=None):
    halodat = pd.read_csv(fndir_laptop + fn_halo, header=2, index_col='galaxyid', sep='\t')
    mass_msun = halodat['M200c_Msun'][galid] 
    halorad_cmpc = halodat['R200c_cMpc'][galid]

    if weight == 'mass':
        fn = pd.read_csv(fndir_laptop + fn_mass, sep='\t', index_col='galaxyid')
    elif weight == 'volume':
        fn = pd.read_csv(fndir_laptop + fn_vol, sep='\t', index_col='galaxyid')
    histfn = fn['filename'][galid]
    histfn = wdir + histfn.split('/')[-1] # quasar path -> laptop path
    histgn = fn['groupname'][galid]

    with h5py.File(histfn, 'r') as f:
        aexp = f['Header/cosmopars'].attrs['a']
        halorad_cm = halorad_cmpc * aexp * c.cm_per_mpc

        rbins_cm = f[histgn]['3Dradius/bins'][:]
        rbins_r200c = rbins_cm / halorad_cm
        rind = f[histgn]['3Dradius'].attrs['histogram axis']
        rsels = []
        for tup in r200cranges:
            ilo = np.where(np.isclose(rbins_r200c, tup[0]))[0]
            if len(ilo) == 1:
                ilo = ilo[0]
            else:
                msg = 'No unique matching bin edge for {} R200c in {}'
                raise RuntimeError(msg.format(tup[0], rbins_r200c))
            ihi = np.where(np.isclose(rbins_r200c, tup[1]))[0]
            if len(ihi) == 1:
                ihi = ihi[0]
            else:
                msg = 'No unique matching bin edge for {} R200c in {}'
                raise RuntimeError(msg.format(tup[1], rbins_r200c))
            rsels.append(slice(ilo, ihi, None))
            
        sfind = f[histgn]['StarFormationRate_T4EOS'].attrs['histogram axis']
        sfsel = slice(None, None, None) if inclsf else slice(0, 1, None)
        
        logTbins = f[histgn]['Temperature_T4EOS/bins'][:]
        #tind = f[histgn]['Temperature_T4EOS'].attrs['histogram axis']
        
        hist = f[histgn]['histogram'][:]
        hlog = bool(f[histgn]['histogram'].attrs['log'])
        if hlog:
            hist = 10**hist
        
    subhists = []
    sumaxes = (rind, sfind)
    for rsel in rsels:
        if len(hist.shape) != 3:
            msg = 'Unexpected histogram dimension {}'
            raise RuntimeError(msg.format(len(hist.shape)))
        sel = [slice(None, None, None)] * len(hist.shape)
        sel[sfind] = sfsel
        sel[rind] = rsel
        sel = tuple(sel)
        _shist = np.sum(hist[sel], axis=sumaxes)
        subhists.append(_shist)
    
    # if ax is given, assume the save is handled by the wrapper routine
    dosave = ax is None
    if dosave:
        if savename is None:
            savename = mdir + 'halo_examples/' + '{}_galaxy_{}.pdf'
            savename = savename.format(weight, galid)
        elif '/' not in savename:
            savename = mdir + 'halo_examples/' + savename
        if '.' not in savename:
            savename = savename + '.pdf'
    if ax is None:
        fig = plt.figure(figsize=(5.5, 5.))
        _ax = fig.add_subplot(1, 1, 1)
    else:
        _ax = ax
    
    fontsize = 12
    colors = tc.tol_cset('bright')

    _ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True,
                   left=True, bottom=True,
                   axis='both', which='both',
                   labelleft=True, labeltop=False,
                   labelbottom=True, labelright=False)
    title = 'galaxy {}, $\\mathrm{{M}}_{{200c}} = 10^{{{:.1f}}} ' +\
            '\\; \\mathrm{{M}}_{{\\odot}}$'
    title = title.format(galid, np.log10(mass_msun))
    _ax.set_title(title, fontsize=fontsize)
    xlabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
    _ax.set_xlabel(xlabel, fontsize=fontsize)
    yfill = 'M' if weight == 'mass' else \
            'V' if weight == 'volume' else \
            ''
    ylabel = '$\\log_{{10}} \\, \\Delta \\mathrm{{{yq}}} \\,/\\,' + \
             ' \\mathrm{{{yq}}} \\,/\\, \\Delta \\log_{{10}} \\mathrm{{T}}$'
    ylabel = ylabel.format(yq=yfill)
    _ax.set_ylabel(ylabel, fontsize=fontsize)
 
    for ci, (subhist, redges) in enumerate(zip(subhists, r200cranges)):
        color = colors[ci]
        label = '${:.1f} \\emdash {:.1f} \\; \\mathrm{{R}}_{{\\mathrm{{200c}}}}$'
        label = label.format(*tuple(redges))
        subhist /= np.sum(subhist)
        subhist /= np.diff(logTbins)
        subhist = np.append(subhist, [0])
        _ax.step(logTbins, np.log10(subhist), where='pre', color=color, label=label,
                 linewidth=1.5)
    _ax.legend(fontsize=fontsize)

    if dosave:
        plt.savefig(savename, format=savename.split('.')[-1], bbox_inches='tight')

def plotdist_group_examples(samplesize, weight='mass', inclsf=False, 
                            r200cranges=[(0.1, 1.), (0.1, 0.2), (0.4, 0.6), (0.9, 1.)]):
    halodat = pd.read_csv(fndir_laptop + fn_halo, header=2, index_col='galaxyid', sep='\t')
    gids = halodat.index[halodat['M200c_Msun'] >= min_mass_msun]
    gids = np.array(gids)
    np.random.seed(0)
    galids = np.random.choice(gids, replace=False, size=samplesize)
    for galid in galids:
        plotdist_group(galid, ax=None, weight=weight, inclsf=inclsf, 
                       r200cranges=r200cranges, savename=None)
        






        






    

