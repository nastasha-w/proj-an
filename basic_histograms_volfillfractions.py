#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:30:57 2018

@author: wijers
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from matplotlib.animation import FuncAnimation
import matplotlib as mpl

import eagle_constants_and_units as c
import make_maps_v3_master as m3
import make_maps_opts_locs as ol

pdir = ol.pdir
overf = 1.e25

simnum = 'L0100N1504'
var = 'REFERENCE'
snapnum = 27
filename = pdir + 'mass_volume_fractions_by_T.hdf5'

logTbins = np.array([-100., 2.] + list(np.arange(2.5, 8.55, 0.1)) + [9., 9.5, 100.])
Tbins = 10**logTbins

def gethistogram():
    simfile = m3.Simfile(simnum, snapnum, var, file_type=ol.file_type, simulation='eagle')
    
    T = simfile.readarray('PartType0/Temperature', rawunits=False)
    # T4SFR
    sfr = simfile.readarray('PartType0/StarFormationRate')
    T[sfr > 0 ] = 10**4
    
    mass = simfile.readarray('PartType0/Mass', rawunits=True)
    massc = simfile.CGSconvtot
    density = simfile.readarray('PartType0/Density', rawunits=True)
    densc = simfile.CGSconvtot
    
    volume = mass / density
    volc = massc / densc
    
    vhist, vedgesT = np.histogram(T, bins=Tbins, weights=volume, density=False)
    mhist, medgesT = np.histogram(T, bins=Tbins, weights=mass, density=False)
    
    vtot = np.sum(volume)
    mtot = np.sum(mass)
    
    vsfr = np.sum(volume[sfr>0])
    msfr = np.sum(volume[sfr>0])
    
    with h5py.File(filename, 'w') as fo:
        fo.create_dataset('V_histogram_by_T', data=vhist)
        fo.create_dataset('m_histogram_by_T', data=mhist)
        fo.create_dataset('V_histogram_T_edges', data=vedgesT)
        fo.create_dataset('m_histogram_T_edges', data=medgesT)
        fo.create_dataset('V_total_gas', data=vtot)
        fo.create_dataset('m_total_gas', data=mtot)
        fo.create_dataset('V_total_SFgas', data=vsfr)
        fo.create_dataset('m_total_SFgas', data=msfr)
        units = fo.create_group('units')
        units.attrs.create('V_to_cgs_proper', volc)
        units.attrs.create('M_to_cgs', massc)
        units.attrs.create('T_to_cgs', 1.)
        hed = fo.create_group('Header')
        hed.attrs.create('SF_gas', '10^4 K')
        hed.attrs.create('simulation', 'eagle')
        hed.attrs.create('variation', var)
        hed.attrs.create('snapshot', snapnum)
        hed.attrs.create('box', simnum)
        hed.attrs.create('a', simfile.a)
        hed.attrs.create('h', simfile.h)
        hed.attrs.create('z', simfile.z)
        hed.attrs.create('omegam', simfile.omegam)
        hed.attrs.create('omegab', simfile.omegab)
        hed.attrs.create('omegalambda', simfile.omegalambda)

def plothistograms_mVfracs_by_T():
    with h5py.File(filename) as fi:
        TedgesV = np.log10(np.array(fi['V_histogram_T_edges']))
        Tedgesm = np.log10(np.array(fi['m_histogram_T_edges']))
        histV = np.array(fi['V_histogram_by_T'])
        histm = np.array(fi['m_histogram_by_T'])
        totV = np.array(fi['V_total_gas'])
        totm = np.array(fi['m_total_gas'])
        #sfV = np.array(fi['V_total_SFgas'])
        #sfm = np.array(fi['m_total_SFgas'])
        Vconv = fi['units'].attrs['V_to_cgs_proper']
        print('Total volume: %s cm^3'%(totV*Vconv))
    
    fig, ax = plt.subplots(1,1)
    fontsize=12
    vcol = 'blue'
    mcol = 'red'
    
    histV /= totV * np.diff(TedgesV)
    histm /= totm * np.diff(Tedgesm)
    ax.step(TedgesV[:-1], histV, color=vcol, linewidth=2, label='Volume fraction', where='post')
    ax.step(Tedgesm[:-1], histm, color=mcol, linewidth=2, label='Mass fraction', linestyle='dashed', where='post')
    #ax.axhline(sfV / totV, color=vcol, linewidth=1, label='star-forming gas')
    #ax.axhline(sfm / totm, color=mcol, linewidth=1, label='star-forming gas', linestyle='dashed')
    
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(r'$\log_{10}\, T \; [K]$', fontsize=fontsize)
    ax.set_ylabel(r'$ \mathrm{total}^{-1} \; \mathrm{d} (M \mathrm{\;or\;} V) \,/\, \mathrm{d} \log_{10} T$', fontsize=fontsize)
    ax.set_title('Gas temperature distribution in EAGLE L0100N1504, $z=0.1$')
    ax.set_yscale('log')
    ax.set_xlim(2.6, 8.1)
    ax.set_ylim(1.e-4, 2.)
    ax.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=True, labelbottom=True, which='both')
    ax.minorticks_on()
    
    plt.savefig('/net/luttero/data2/imgs/gas_state/Thist_by_mass_volume_L0100N1504_27_T4EOS.pdf', format='pdf', box_inches='tight')
        

def gethistograms_phase_ev(simnum = 'L0012N0188',\
    snapnums = np.array([8, 10, 12, 15, 19, 22, 23, 24, 25, 26, 27, 28]),\
    var = 'REFERENCE'):
    filename = pdir + '%s%s_phase_diagrams_by_mass_zev.hdf5'%(simnum, var)
    
    logTbins = np.array([-100., 2.] + list(np.arange(2.5, 8.55, 0.1)) + [9., 9.5, 100.])
    Tbins = 10**logTbins
    
    lognHbins = np.array([-100., -9.] + list(np.arange(-8.7, 4.05, 0.1)) + [4.5, 5., 100.])
    nHbins = 10**lognHbins
    
    logdelbins = np.array([-100., -3.] + list(np.arange(-2.5, 10.05, 0.1)) + [10.5, 100.])
    delbins = 10**logdelbins
    
    with h5py.File(filename, 'w') as fo:    
        hed = fo.create_group('Header')
        hed.attrs.create('SF_gas', '10^4 K')
        hed.attrs.create('abundance', 'ElementAbundance/Hydrogen')
        hed.attrs.create('simulation', 'eagle')
        hed.attrs.create('variation', var)
        hed.create_dataset('snapshots', data=snapnums)
        hed.attrs.create('box', simnum)
    
    for snap in snapnums:
        print('Calculating snapshot %i histogram'%snap)
        simfile = m3.Simfile(simnum, snap, var, file_type=ol.file_type, simulation='eagle')
    
        T = simfile.readarray('PartType0/Temperature', rawunits=False)
        sfr = simfile.readarray('PartType0/StarFormationRate')
        T[sfr > 0 ] = 10**4
        
        mass = simfile.readarray('PartType0/Mass', rawunits=True)
        massc = simfile.CGSconvtot
        mtot = np.sum(mass)
        msfr = np.sum(mass[sfr>0])
        
        del sfr
        
        density = simfile.readarray('PartType0/Density', rawunits=True)
        densc = simfile.CGSconvtot
        nH = simfile.readarray('PartType0/ElementAbundance/Hydrogen', rawunits=True)
        nHc = simfile.CGSconvtot
        nH *= density
        nH *= nHc * densc / (c.atomw_H * c.u) 
        
        rhob = 3. / (8. * np.pi * c.gravity) * c.hubble**2 \
              * simfile.h**2 * simfile.omegab / simfile.a**3 
        density /= rhob # density -> overdensity
        density *= densc
        #print np.min(density), np.max(density)
        
        #print density.shape, nH.shape, T.shape, mass.shape
        histdel, (edgesdel, edgesT) = np.histogramdd((density, T), bins=[delbins, Tbins], weights=mass)
        histnH,  (edgesnH,  edgesT) = np.histogramdd((nH,      T), bins=[nHbins,  Tbins], weights=mass)
        
        print('Saving snapshot %i histogram'%snap)
        with h5py.File(filename, 'a') as fo:
            grp = fo.create_group('snap%i'%snap)
            grp.create_dataset('m_histogram_by_delta_T', data=histdel)
            grp.create_dataset('m_histogram_by_nH_T', data=histnH)
            grp.create_dataset('m_histogram_T_edges', data=edgesT)
            grp.create_dataset('m_histogram_delta_edges', data=edgesdel)
            grp.create_dataset('m_histogram_nH_edges', data=edgesnH)
            grp.create_dataset('m_total_gas', data=mtot)
            grp.create_dataset('m_total_SFgas', data=msfr)
            units = grp.create_group('units')
            units.attrs.create('M_to_cgs', massc)
            units.attrs.create('nH_to_cgs', 1.)
            units.attrs.create('delta', 'rho / (Omega_b x rho_crit)_z')
            units.attrs.create('T_to_cgs', 1.)
            cosmo = grp.create_group('cosmopars')
            cosmo.attrs.create('a', simfile.a)
            cosmo.attrs.create('h', simfile.h)
            cosmo.attrs.create('z', simfile.z)
            cosmo.attrs.create('omegam', simfile.omegam)
            cosmo.attrs.create('omegab', simfile.omegab)
            cosmo.attrs.create('omegalambda', simfile.omegalambda)

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
        
def plothistograms_zev(filename):
    mdir = '/home/wijers/Documents/proposals/sean_johnson_HST_midcycle_2019_spring/'
    outname = filename.split('/')[-1]
    outname = outname.split('.')[:-1]
    outname = '.'.join(outname)
    outname = '%s%s.gif'%(mdir, outname)
    
    with h5py.File(filename) as fi:
        snapnums = np.array(fi['Header/snapshots'])
        delhists = {snap: np.array(fi['snap%i/%s'%(snap, 'm_histogram_by_delta_T')]) / np.array(fi['snap%i/%s'%(snap, 'm_total_gas')]) for snap in snapnums}
        nHhists  = {snap: np.array(fi['snap%i/%s'%(snap, 'm_histogram_by_nH_T')]) / np.array(fi['snap%i/%s'%(snap, 'm_total_gas')]) for snap in snapnums}
        Tgrids = {snap: np.array(fi['snap%i/%s'%(snap, 'm_histogram_T_edges')]) for snap in snapnums}
        nHgrids = {snap: np.array(fi['snap%i/%s'%(snap, 'm_histogram_nH_edges')]) for snap in snapnums}
        deltagrids = {snap: np.array(fi['snap%i/%s'%(snap, 'm_histogram_delta_edges')]) for snap in snapnums}
        redshifts ={snap: (fi['snap%i/%s'%(snap, 'cosmopars')]).attrs['z'] for snap in snapnums}
        
    fig = plt.figure(figsize=(5.5, 3.))
    grid = gsp.GridSpec(1, 3, height_ratios=[1.], width_ratios=[5., 5., 1.], wspace=0.05, hspace=0.25, top=0.95, bottom=0.17, left=0.12, right=0.88)
    ax1 = plt.subplot(grid[0], facecolor='white') 
    ax2 = plt.subplot(grid[1])
    cax = plt.subplot(grid[2])
    fontsize = 12
    vmax = max([max([np.max(delhists[snap]), np.max(nHhists[snap])]) for snap in snapnums]) / 0.1**2
    vmin = vmax * 1.e-7
    cmap = 'gist_yarg'
    
    ax1.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=True, labelbottom=True, which='both')
    ax1.minorticks_on()
    ax2.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=False, labelbottom=True, which='both')
    ax2.minorticks_on()
    ax1.set_xlabel(r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    ax2.set_xlabel(r'$\log_{10}\, \rho \,/\, (\Omega_b(z) \times \rho_c(z))$', fontsize=fontsize)
    ax1.set_ylabel(r'$\log_{10}\, T \; [K]$', fontsize=fontsize)
    
    add_colorbar(cax, img=None, vmin=np.log10(vmin), vmax=np.log10(vmax), cmap=cmap,\
                 clabel=r'$\log_{10}\,$mass fraction$\,/\, \mathrm{dex}^{2}$',\
                 newax=False, extend='min', fontsize=12., orientation='vertical')
    
    def update(snap):
        delhist = delhists[snap]
        nHhist  = nHhists[snap]
        Tgrid   = np.log10(Tgrids[snap])
        nHgrid  = np.log10(nHgrids[snap])
        deltagrid = np.log10(deltagrids[snap])
        
        # edges are log(0) and infinity -> adjust
        Tgrid[0]  = Tgrid[1]  - 0.1
        Tgrid[-1] = Tgrid[-2] + 0.1
        nHgrid[0]  = nHgrid[1]  - 0.1
        nHgrid[-1] = nHgrid[-2] + 0.1
        deltagrid[0]  = deltagrid[1]  - 0.1
        deltagrid[-1] = deltagrid[-2] + 0.1
        
        deldens = delhist / (np.diff(deltagrid)[:, np.newaxis] * np.diff(Tgrid)[np.newaxis, :])
        nHdens  = nHhist  / (np.diff(nHgrid)[:, np.newaxis]    * np.diff(Tgrid)[np.newaxis, :])
        
        ax1.pcolormesh(nHgrid,    Tgrid, np.log10(nHdens).T,  cmap=cmap, vmin=np.log10(vmin), vmax=np.log10(vmax))
        ax2.pcolormesh(deltagrid, Tgrid, np.log10(deldens).T, cmap=cmap, vmin=np.log10(vmin), vmax=np.log10(vmax))
        
        ax1.text(0.95, 0.95, r'$z=%.2f$'%(redshifts[snap]), fontsize=fontsize,\
                horizontalalignment='right', verticalalignment='top',\
                transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=1.))
    
    anim = FuncAnimation(fig, update, frames=snapnums, interval=300)
    anim.save(outname, dpi=80, writer='imagemagick')
    
    #plt.savefig('/net/luttero/data2/imgs/gas_state/Thist_by_mass_volume_L0100N1504_27_T4EOS.pdf', format='pdf', box_inches='tight')