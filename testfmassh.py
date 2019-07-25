#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:26:12 2018

@author: wijers
"""

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib as mpl

import calcfmassh as cfh
reload(cfh) # testing -> keep up to date

## input test values
# randomish values in about the right range of T and nH
testTvals  = 10**np.arange(2.05, 9.1, 0.31)
testnHvals = 10**np.arange(-8.34, 4.56, 0.27)
testeosvals = slice(None, None, None) # assume all gas is on the EOS for testing purposes
zvals = np.arange(0.11, 5.0, 0.29)
useLSR = False
UVBs   = ['HM01', 'HM12']
ions   = ['h1ssh', 'hmolssh']

# T varies fast, nH varies slow
testTgrid  = np.repeat(testTvals[np.newaxis, :], len(testnHvals), axis=0).flatten()
testnHgrid = np.repeat(testnHvals[:, np.newaxis], len(testTvals), axis=1).flatten()
dct = {'Temperature': testTgrid, 'nH': testnHgrid, 'eos': testeosvals}

def runtestgrids():
    outgrids = {ion: {UVB: np.zeros((len(zvals), len(testnHvals), len(testTvals)))/0. for UVB in UVBs} for ion in ions} # start with NaN grid
    for genind in range(len(zvals) * len(ions) * len(UVBs)):
        zind = genind // (len(ions) * len(UVBs))
        iind = genind // len(UVBs) - zind * len(ions)
        uind = genind % len(UVBs)  
        z   = zvals[zind]
        ion = ions[iind]
        UVB = UVBs[uind]
        
        h1hmolfrac = cfh.nHIHmol_over_nH(dct, z, UVB=UVB, useLSR=useLSR)
        if ion == 'h1ssh':
            h1hmolfrac *= (1. - cfh.rhoHmol_over_rhoH(dct))
        elif ion == 'hmolssh':
            h1hmolfrac *= cfh.rhoHmol_over_rhoH(dct)
    
        outgrids[ions[iind]][UVBs[uind]][zind, :, :] = h1hmolfrac.reshape(len(testnHvals), len(testTvals))

    return outgrids

def runtestgrids_oneredshift(z):
    outgrids = {ion: {UVB: np.zeros((len(testnHvals), len(testTvals)))/0. for UVB in UVBs} for ion in ions} # start with NaN grid
    for genind in range(len(ions) * len(UVBs)):
        iind = genind // len(UVBs)
        uind = genind % len(UVBs)  
        ion = ions[iind]
        UVB = UVBs[uind]
        
        h1hmolfrac = cfh.nHIHmol_over_nH(dct, z, UVB=UVB, useLSR=useLSR)
        if ion == 'h1ssh':
            h1hmolfrac *= (1. - cfh.rhoHmol_over_rhoH(dct, EOS='eagle'))
        elif ion == 'hmolssh':
            h1hmolfrac *= cfh.rhoHmol_over_rhoH(dct, EOS='eagle')
    
        outgrids[ions[iind]][UVBs[uind]] = h1hmolfrac.reshape(len(testnHvals), len(testTvals))
    return outgrids
        
def savetestgrids(outgrids, z=None):
    with h5py.File('/net/luttero/data1/line_em_abs/v3_master_tests/h1ssh_tests/tests_rahmati2013_ssh_z0_debug4.hdf5', 'w') as out:
        hed = out.create_group('Header')
        if z is None:
            hed.create_dataset('Temperature_ax2', data=testTvals)
            hed['Temperature_ax2'].attrs.create('units', 'K')
            hed.create_dataset('HydrogenNumberDensity_ax1', data=testnHvals)
            hed['HydrogenNumberDensity_ax1'].attrs.create('units', 'cm^-3')
            hed.create_dataset('redshift_ax0', data=zvals)
            hed.attrs.create('star-forming', 'all')
        else:
            hed.create_dataset('Temperature_ax1', data=testTvals)
            hed['Temperature_ax1'].attrs.create('units', 'K')
            hed.create_dataset('HydrogenNumberDensity_ax0', data=testnHvals)
            hed['HydrogenNumberDensity_ax0'].attrs.create('units', 'cm^-3')
            hed.create_dataset('redshift', data=z)
            hed.attrs.create('star-forming', 'all')
        for UVB in UVBs:
            for ion in ions:
                dsname = 'massfraction-%s_%s-UVB'%(ion, UVB)
                out.create_dataset(dsname, data=outgrids[ion][UVB])
   
def add_colorbar(ax, img=None, vmin=None, vmax=None, cmap=None, clabel=None,\
                 newax=False, extend='neither', fontsize=12., orientation='vertical'):
    if img is None:
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, extend=extend, orientation=orientation)
    else:
        cbar = mpl.colorbar.Colorbar(ax, img, extend=extend, orientation=orientation)
    ax.tick_params(labelsize=fontsize - 1.)
    if clabel is not None:
        cbar.set_label(clabel,fontsize=fontsize)
        
def comparegrids_z0(plottotalgrids=True):
    path = '/net/luttero/data1/line_em_abs/v3_master_tests/h1ssh_tests/'
    shape = (len(testnHvals), len(testTvals))
    with h5py.File(path + 'tests_rahmati2013_ssh_z0_debug4.hdf5') as n,\
         h5py.File(path + 'Yannick_HI_test_noSF.hdf5') as yt,\
         h5py.File(path + 'Yannick_HI_test_SFR-1p0_18Dec18.hdf5') as yh2:
        ftot_yh2 = np.array(yh2['f_Neutral']).reshape(shape) / 0.752
        fhi_yh2  = np.array(yh2['f_HI_BR06']).reshape(shape) / 0.752
        fh2_yh2  = np.array(yh2['f_H2_BR06']).reshape(shape) / 0.752
        
        ftot_yt = np.array(yt['f_Neutral']).reshape(shape)
        fhi_yt  = np.array(yt['f_HI']).reshape(shape)
        fh2_yt  = np.array(yt['f_H2']).reshape(shape)
        
        fhi_n  = np.array(n['massfraction-h1ssh_HM01-UVB']).reshape(shape) 
        fh2_n  = np.array(n['massfraction-hmolssh_HM01-UVB']).reshape(shape) 
        ftot_n = fhi_n + fh2_n
        
    print('Total neutral fraction check:')
    print("Max diff Yannick's methods: %f"%(np.max(np.abs(ftot_yh2 - ftot_yt))))
    print("Max diff Yannick's no H2/Nastasha's methods: %f"%(np.max(np.abs(ftot_yh2 - ftot_n))))
    print("Max diff Yannick's H2/Nastasha's methods: %f"%(np.max(np.abs(ftot_yt - ftot_n))))
    
    print('')
    print("Check if I've understood the fraction ratios:")
    print("Max diff Yannick's no H2 HI and total neutral: %f"%(np.max(np.abs(ftot_yt - fhi_yt))))
    print("Max diff Yannick's no H2 H2 and zero: %f"%(np.max(fh2_yt)))
    print("Max diff Yannick's H2-method HI + H2 and neutral: %f"%(np.max(np.abs(ftot_yh2 - fhi_yh2 - fh2_yh2))))
    
    print('')
    print("Check max differences in species:")
    print("Max diff Yannick's H2-method and Nastasha's HI: %f"%(np.max(np.abs(fhi_yh2 - fhi_n))))
    print("Max diff Yannick's H2-method and Nastasha's H2: %f"%(np.max(np.abs(fh2_yh2 - fh2_n))))
    print("Max log ratio Yannick's H2-method and Nastasha's HI: %f"%(np.max(np.abs(np.log10(fhi_yh2/fhi_n)))))
    print("Max log ratio Yannick's H2-method and Nastasha's H2: %f"%(np.max(np.abs(np.log10(fh2_yh2/fh2_n)))))
    
    ## plot differences
    # total grid plots

    cmap = 'viridis'   
    fontsize = 12
    Tlabel = r'$\log_{10}\, T \; [\mathrm{K}]$'
    nHlabel = r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$'
    
    if plottotalgrids:
        imgs = np.log10(np.array([[ftot_yt, fhi_yt, fh2_yt], [ftot_yh2, fhi_yh2, fh2_yh2], [ftot_n, fhi_n, fh2_n]]))
        methods = [r'YB no $H_2$', r'YB with $H_2$', 'NW']
        species = ['neutral', r'$\mathrm{HI}$', r'$H_2$']
        plt.figure(figsize=(6.5, 6.5*3.0/3.3))
        grid = gsp.GridSpec(3, 4, height_ratios=[1., 1., 1.], width_ratios=[1., 1., 1., 0.3], hspace=0.17, wspace=0.05, top=0.95, bottom=0.05, left=0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
        mainaxes = [[plt.subplot(grid[xi, yi]) for xi in range(3)] for yi in range(3)]
        caxes =    [ plt.subplot(grid[xi, 3])  for xi in range(3)]
        vmins = np.array([np.min(imgs[:, i, :, :][np.isfinite(imgs[:, i, :, :])]) for i in range(3)])
        vmaxs = np.max(imgs, axis=(0, 2, 3))
        
        Tdiff  = np.average(np.diff(np.log10(testTvals)))
        nHdiff = np.average(np.diff(np.log10(testnHvals)))
        Tmin  = np.log10(testTvals[0])   - 0.5 * Tdiff
        Tmax  = np.log10(testTvals[-1])  + 0.5 * Tdiff
        nHmin = np.log10(testnHvals[0])  - 0.5 * nHdiff
        nHmax = np.log10(testnHvals[-1]) + 0.5 * nHdiff
        
        for rowind in range(3):
            for colind in range(3):            
                ax = mainaxes[colind][rowind]
                yticks = colind == 0 
                xticks = rowind == 2
                if yticks:
                    ax.set_ylabel(Tlabel, fontsize=fontsize)
                if xticks:
                    ax.set_xlabel(nHlabel, fontsize=fontsize)
                ax.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=yticks, labelbottom=xticks, which='both')
                ax.set_title('%s: %s'%(species[rowind], methods[colind]), fontsize=fontsize)
                img = ax.imshow(imgs[colind, rowind, :, :].T, origin='lower', interpolation='nearest', extent=(nHmin, nHmax, Tmin, Tmax), cmap=cmap, vmin=vmins[rowind], vmax=vmaxs[rowind])
                ax.set_aspect((nHmax - nHmin)/(Tmax - Tmin), adjustable='box-forced') 
            cax = caxes[rowind]
            add_colorbar(cax, img=img, clabel='$\log_{10}$ %s fraction'%species[rowind], newax=False, extend='neither', fontsize=fontsize, orientation='vertical')
            ax.tick_params(labelsize=fontsize - 1)
            cax.set_aspect(15.)
        
        plt.savefig(path + 'fractionplots_redshift0_NWdebug4_YBbugfix2.pdf', format='pdf', bbox_inches='tight')
        
def plotdiffs_z0(plottotalgrids=True, vmax=1.):
    path = '/net/luttero/data1/line_em_abs/v3_master_tests/h1ssh_tests/'
    shape = (len(testnHvals), len(testTvals))
    with h5py.File(path + 'tests_rahmati2013_ssh_z0_debug4.hdf5') as n,\
         h5py.File(path + 'Yannick_HI_test_noSF.hdf5') as yt,\
         h5py.File(path + 'Yannick_HI_test_SFR-1p0_18Dec18.hdf5') as yh2:
        ftot_yh2 = np.array(yh2['f_Neutral']).reshape(shape) / 0.752
        fhi_yh2  = np.array(yh2['f_HI_BR06']).reshape(shape) / 0.752
        fh2_yh2  = np.array(yh2['f_H2_BR06']).reshape(shape) / 0.752
        
        ftot_yt = np.array(yt['f_Neutral']).reshape(shape)
        fhi_yt  = np.array(yt['f_HI']).reshape(shape)
        fh2_yt  = np.array(yt['f_H2']).reshape(shape)
        
        fhi_n  = np.array(n['massfraction-h1ssh_HM01-UVB']).reshape(shape) 
        fh2_n  = np.array(n['massfraction-hmolssh_HM01-UVB']).reshape(shape) 
        ftot_n = fhi_n + fh2_n
        
    print('Total neutral fraction check:')
    print("Max diff Yannick's methods: %f"%(np.max(np.abs(ftot_yh2 - ftot_yt))))
    print("Max diff Yannick's no H2/Nastasha's methods: %f"%(np.max(np.abs(ftot_yh2 - ftot_n))))
    print("Max diff Yannick's H2/Nastasha's methods: %f"%(np.max(np.abs(ftot_yt - ftot_n))))
    
    print('')
    print("Check if I've understood the fraction ratios:")
    print("Max diff Yannick's no H2 HI and total neutral: %f"%(np.max(np.abs(ftot_yt - fhi_yt))))
    print("Max diff Yannick's no H2 H2 and zero: %f"%(np.max(fh2_yt)))
    print("Max diff Yannick's H2-method HI + H2 and neutral: %f"%(np.max(np.abs(ftot_yh2 - fhi_yh2 - fh2_yh2))))
    
    print('')
    print("Check max differences in species:")
    print("Max diff Yannick's H2-method and Nastasha's HI: %f"%(np.max(np.abs(fhi_yh2 - fhi_n))))
    print("Max diff Yannick's H2-method and Nastasha's H2: %f"%(np.max(np.abs(fh2_yh2 - fh2_n))))
    print("Max log ratio Yannick's H2-method and Nastasha's HI: %f"%(np.max(np.abs(np.log10(fhi_yh2/fhi_n)))))
    print("Max log ratio Yannick's H2-method and Nastasha's H2: %f"%(np.max(np.abs(np.log10(fh2_yh2/fh2_n)))))
    
    ## plot differences
    # total grid plots

    cmap = 'RdBu'   
    fontsize = 12
    Tlabel = r'$\log_{10}\, T \; [\mathrm{K}]$'
    nHlabel = r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$'
    
    if plottotalgrids:
        imgs = np.log10(np.array([[ftot_yt / ftot_yh2, fhi_yt / fhi_yh2, fh2_yt / fh2_yh2], [ftot_yh2 / ftot_n, fhi_yh2 / fhi_n, fh2_yh2 / fh2_n], [ftot_yt / ftot_n, fhi_yt / fhi_n, fh2_yt / fh2_n]]))
        methods = [r'YB no $H_2$ / YB with $H2$', r'YB with $H_2$ / NW', r' YB no $H_2$ / NW']
        species = ['neutral', r'$\mathrm{HI}$', r'$H_2$']
        plt.figure(figsize=(6.5, 6.5*3.0/3.3))
        grid = gsp.GridSpec(3, 4, height_ratios=[1., 1., 1.], width_ratios=[1., 1., 1., 0.3], hspace=0.17, wspace=0.05, top=0.95, bottom=0.05, left=0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
        mainaxes = [[plt.subplot(grid[xi, yi]) for xi in range(3)] for yi in range(3)]
        caxes =    [ plt.subplot(grid[xi, 3])  for xi in range(3)]
        #vmins = np.array([np.min(np.abs(imgs[:, i, :, :][np.isfinite(imgs[:, i, :, :])])) for i in range(3)])
        vmaxs = np.min(np.array([np.max(np.abs(imgs), axis=(0, 2, 3)), (vmax,) * imgs.shape[1] ]), axis=0)
        vmins = -1. * vmaxs 
        
        Tdiff  = np.average(np.diff(np.log10(testTvals)))
        nHdiff = np.average(np.diff(np.log10(testnHvals)))
        Tmin  = np.log10(testTvals[0])   - 0.5 * Tdiff
        Tmax  = np.log10(testTvals[-1])  + 0.5 * Tdiff
        nHmin = np.log10(testnHvals[0])  - 0.5 * nHdiff
        nHmax = np.log10(testnHvals[-1]) + 0.5 * nHdiff
        
        for rowind in range(3):
            for colind in range(3):            
                ax = mainaxes[colind][rowind]
                yticks = colind == 0 
                xticks = rowind == 2
                if yticks:
                    ax.set_ylabel(Tlabel, fontsize=fontsize)
                if xticks:
                    ax.set_xlabel(nHlabel, fontsize=fontsize)
                ax.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=yticks, labelbottom=xticks, which='both')
                ax.set_title('%s: %s'%(species[rowind], methods[colind]), fontsize=fontsize)
                img = ax.imshow(imgs[colind, rowind, :, :].T, origin='lower', interpolation='nearest', extent=(nHmin, nHmax, Tmin, Tmax), cmap=cmap, vmin=vmins[rowind], vmax=vmaxs[rowind])
                ax.set_aspect((nHmax - nHmin)/(Tmax - Tmin), adjustable='box-forced') 
            cax = caxes[rowind]
            add_colorbar(cax, img=img, clabel='$\log_{10}$ %s fraction'%species[rowind], newax=False, extend='neither', fontsize=fontsize, orientation='vertical')
            ax.tick_params(labelsize=fontsize - 1)
            cax.set_aspect(15.)
        
        plt.savefig(path + 'fractionplots_redshift0_differences_NWdebug4_YBbugfix2.pdf', format='pdf', bbox_inches='tight')

def compareversiongrids_z0(file1, file2, plottotalgrids=True):
    path = '/net/luttero/data1/line_em_abs/v3_master_tests/h1ssh_tests/'
    shape = (len(testnHvals), len(testTvals))
    with h5py.File(path + file1) as new,\
         h5py.File(path + file2) as old:
        
        fhi_n  = np.array(new['massfraction-h1ssh_HM01-UVB']).reshape(shape) 
        fh2_n  = np.array(new['massfraction-hmolssh_HM01-UVB']).reshape(shape) 
        ftot_n = fhi_n + fh2_n
        
        fhi_o  = np.array(old['massfraction-h1ssh_HM01-UVB']).reshape(shape) 
        fh2_o  = np.array(old['massfraction-hmolssh_HM01-UVB']).reshape(shape)
        ftot_o = fhi_o + fh2_o
        
    print('Total neutral fraction check:')
    print("Max diff methods: %f"%(np.max(np.abs(ftot_n - ftot_o))))
    
    print('')
    print("Check max differences in species:")
    print("Max diff HI: %f"%(np.max(np.abs(fhi_n - fhi_o))))
    print("Max diff H2: %f"%(np.max(np.abs(fh2_n - fh2_o))))
    print("Max log ratio HI: %f"%(np.max(np.abs(np.log10(fhi_n/fhi_o)))))
    print("Max log ratio H2: %f"%(np.max(np.abs(np.log10(fh2_n/fh2_o)))))
    
    ## plot differences
    # total grid plots

    cmap = 'viridis'   
    fontsize = 12
    Tlabel = r'$\log_{10}\, T \; [\mathrm{K}]$'
    nHlabel = r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$'
    
    if plottotalgrids:
        imgs = np.log10(np.array([[ftot_n, fhi_n, fh2_n], [ftot_o, fhi_o, fh2_o]]))
        files = [file1[:-5], file2[:-5]]
        species = ['neutral', r'$\mathrm{HI}$', r'$H_2$']
        plt.figure(figsize=(6.5, 6.5*3.0/2.3))
        grid = gsp.GridSpec(3, 3, height_ratios=[1., 1., 1.], width_ratios=[1., 1., 0.3], hspace=0.17, wspace=0.05, top=0.95, bottom=0.05, left=0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
        mainaxes = [[plt.subplot(grid[xi, yi]) for xi in range(3)] for yi in range(2)]
        caxes =    [ plt.subplot(grid[xi, 2])  for xi in range(3)]
        vmins = np.array([np.min(imgs[:, i, :, :][np.isfinite(imgs[:, i, :, :])]) for i in range(3)])
        vmaxs = np.max(imgs, axis=(0, 2, 3))
        
        Tdiff  = np.average(np.diff(np.log10(testTvals)))
        nHdiff = np.average(np.diff(np.log10(testnHvals)))
        Tmin  = np.log10(testTvals[0])   - 0.5 * Tdiff
        Tmax  = np.log10(testTvals[-1])  + 0.5 * Tdiff
        nHmin = np.log10(testnHvals[0])  - 0.5 * nHdiff
        nHmax = np.log10(testnHvals[-1]) + 0.5 * nHdiff
        
        for rowind in range(3):
            for colind in range(2):            
                ax = mainaxes[colind][rowind]
                yticks = colind == 0 
                xticks = rowind == 2
                if yticks:
                    ax.set_ylabel(Tlabel, fontsize=fontsize)
                if xticks:
                    ax.set_xlabel(nHlabel, fontsize=fontsize)
                ax.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=yticks, labelbottom=xticks, which='both')
                ax.set_title('%s: %s'%(species[rowind], files[colind]), fontsize=fontsize)
                img = ax.imshow(imgs[colind, rowind, :, :].T, origin='lower', interpolation='nearest', extent=(nHmin, nHmax, Tmin, Tmax), cmap=cmap, vmin=vmins[rowind], vmax=vmaxs[rowind])
                ax.set_aspect((nHmax - nHmin)/(Tmax - Tmin), adjustable='box-forced') 
            cax = caxes[rowind]
            add_colorbar(cax, img=img, clabel='$\log_{10}$ %s fraction'%species[rowind], newax=False, extend='neither', fontsize=fontsize, orientation='vertical')
            ax.tick_params(labelsize=fontsize - 1)
            cax.set_aspect(15.)
        
        plt.savefig(path + 'versioncomp_%s-%s.pdf'%(file1[:-5], file2[:-5]), format='pdf', bbox_inches='tight')