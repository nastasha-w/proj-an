
# quest modules:
# module load python/anaconda3.6

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import readin_fire_data as rfd
import make_maps_v3_master as m3
import plot_utils as pu

# snapshot 50: redshift 0.5; hopefully enough to make a-scaling errors clear
firedata_test = '/projects/b1026/snapshots/AGN_suite/fiducial_jet/m12i_res57000/output/snapshot_050.hdf5'
firedata_params = '/projects/b1026/snapshots/AGN_suite/fiducial_jet/m12i_res57000/params.txt-usedvalues'
ddir = '/projects/b1026/nastasha/tests/start_fire/'


def make_simple_phasediagram():
    '''
    first test: just read in the data for a histogram
    '''
    
    snap = rfd.Firesnap(firedata_test, firedata_params)

    temperature = snap.readarray_emulateEAGLE('PartType0/Temperature')
    tempconv = snap.toCGS    
    temperature = np.log10(temperature * tempconv)
    density = snap.readarray_emulateEAGLE('PartType0/Density')
    densconv = snap.toCGS
    density = np.log10(density * densconv)
    
    binsize = 0.05
    tempmin = np.min(temperature)
    tempmax = np.max(temperature)
    densmin = np.min(density)
    densmax = np.max(density)
    
    mint = np.floor(tempmin / binsize) * binsize
    maxt = np.ceil(tempmax / binsize) * binsize
    axbins_t = np.arange(mint, maxt + binsize/2., binsize)
    mind = np.floor(densmin / binsize) * binsize
    maxd = np.ceil(densmax / binsize) * binsize
    axbins_d = np.arange(mind, maxd + binsize/2., binsize)
    
    mass = snap.readarray_emulateEAGLE('PartType0/Mass')
    massconv = snap.toCGS
    # actually converting causes an overflow
    
    hist, xbins, ybins = np.histogram2d(density, temperature, 
                                        bins=[axbins_d, axbins_t], 
                                        weights=mass)
    
    fn = 'phasediagram_rhoT_firesnap.hdf5'
    with h5py.File(ddir + fn, 'w') as f:
        f.create_dataset('hist', data=hist)
        munit = '{} g per histogram bin'.format(massconv)
        f['hist'].attrs.create('units', np.string_(munit))
        f['hist'].attrs.create('toCGS', massconv) 
        f['hist'].attrs.create('dimension0', np.string_('logdensity'))                                   
        f['hist'].attrs.create('dimension1', np.string_('logtemperature'))
        f.create_dataset('logdensity', data=axbins_d)
        f['logdensity'].attrs.create('units', np.string_('g * cm**-3'))
        f.create_dataset('logtemperature', data=axbins_t)
        f['logtemperature'].attrs.create('units', np.string_('K'))
        
    hfrac = snap.readarray_emulateEAGLE('PartType0/ElementAbundance/Hydrogen')
    hconv = snap.toCGS # 1.0
    hfrac *= hconv # most values ~ 0.7 --0.75
    
    hdens = density + np.log10(hfrac / (rfd.uf.c.atomw_H * rfd.uf.c.u))
    hdensmin = np.min(hdens)
    hdensmax = np.max(hdens)
    minnh = np.floor(hdensmin / binsize) * binsize
    maxnh = np.ceil(hdensmax / binsize) * binsize
    axbins_nh = np.arange(minnh, maxnh + binsize/2., binsize)
    
    hist, xbins, ybins = np.histogram2d(hdens, temperature, 
                                        bins=[axbins_nh, axbins_t], 
                                        weights=mass)
    
    fn = 'phasediagram_nHT_firesnap.hdf5'                                 
    with h5py.File(ddir + fn, 'w') as f:
        f.create_dataset('hist', data=hist)
        munit = '{} g per histogram bin'.format(massconv)
        f['hist'].attrs.create('units', np.string_(munit))
        f['hist'].attrs.create('toCGS', massconv) 
        f['hist'].attrs.create('dimension0', np.string_('loghydrogendensity'))                                   
        f['hist'].attrs.create('dimension1', np.string_('logtemperature'))
        f.create_dataset('loghydrogendensity', data=axbins_nh)
        f['loghydrogendensity'].attrs.create('units', np.string_('cm**-3'))
        f.create_dataset('logtemperature', data=axbins_t)
        f['logtemperature'].attrs.create('units', np.string_('K'))
        
def make_simple_surfdensplot():
    snap = rfd.Firesnap(firedata_test, firedata_params)
    
    coords = snap.readarray_emulateEAGLE('PartType0/Coordinates')
    coordconv = snap.toCGS
    lsmooth = snap.readarray_emulateEAGLE('PartType0/SmoothingLength')
    lconv = snap.toCGS
    mass = snap.readarray_emulateEAGLE('PartType0/Mass')
    massconv = snap.toCGS
    density = snap.readarray_emulateEAGLE('PartType0/Density')
    densconv = snap.toCGS
    # checked lconv == coordconv

    NumPart = len(lsmooth)
    boxunits = rfd.uf.c.cm_per_mpc * snap.cosmopars.a / snap.cosmopars.h
    boxsize = snap.cosmopars.boxsize * boxunits / coordconv
    box3 = [boxsize] * 3
    periodic = False
    Axis1 = 0
    Axis2 = 1
    Axis3 = 2
    Ls = [10.e3, 10.e3, 10.e3] # ckpc/h
    # median particle coordinates (no particle weights)
    centre = [29576.388948049986, 30730.297831713087, 32117.89832612393] 
    dct = {'coords': coords, 'lsmooth': lsmooth, 'qW': mass, 'qQ': density}
    npix_x = 800
    npix_y = 800
    tree = False
    ompproj = False
    kernel = 'C2'
    
    m3.translate(dct, 'coords', centre, boxsize, periodic)
    massmap, densmap = m3.project(NumPart, Ls, Axis1, Axis2, Axis3, box3, 
                                  periodic, npix_x, npix_y, kernel, dct, tree,
                                  ompproj=ompproj, projmin=None, projmax=None)
    massmap *= (massconv / lconv**2)
    densmap *= densconv
    
    fn = 'mass_density_maps_firesnap.hdf5'
    with h5py.File(ddir + fn, 'w') as f:
        f.create_dataset('massmap', data=massmap)
        f['massmap'].attrs.create('units', np.string_('g * cm**-2'))
        f.create_dataset('densmap_massweighted', data=densmap)
        f['densmap_massweighted'].attrs.create('units', np.string_('g * cm**-3'))
        grp = f.create_group('Header/inputpars')
        grp.attrs.create('firesnap', np.string_(firedata_test))
        grp.attrs.create('simparfile', np.string_(firedata_params))
        grp.attrs.create('codeunits_length_cgs', lconv)
        grp.attrs.create('centre_codeunits', np.array(centre))
        grp.attrs.create('Ls_codeunits', np.array(Ls))
        grp.attrs.create('Axis1', Axis1)
        grp.attrs.create('Axis2', Axis2)
        grp.attrs.create('Axis3', Axis3)
        grp.attrs.create('npix_x', npix_y)
        grp.attrs.create('npix_y', npix_y)
        grp.attrs.create('kernel', np.string_(kernel))
        cosmodct = snap.cosmopars.getdct()
        grp = f.create_group('Header/cosmopars')
        for key in cosmodct:
            grp.attrs.create(key, cosmodct[key])
            
def plot_simple_phasediagrams(plottype='firesnap'):
    filen_rho = 'phasediagram_rhoT_firesnap.hdf5'
    filen_nH = 'phasediagram_nHT_firesnap.hdf5'
    
    rho_to_nH = 0.752 / (rfd.uf.c.atomw_H * rfd.uf.c.u)
    
    with h5py.File(ddir + filen_rho, 'r') as f:
        hist_rho = f['hist'][:]
        hist_rho_toCGS = f['hist'].attrs['toCGS']
        
        rho = f['logdensity'][:] + np.log10(rho_to_nH)
        temp_rho = f['logtemperature'][:]

    with h5py.File(ddir + filen_nH, 'r') as f:
        hist_nH = f['hist'][:]
        hist_nH_toCGS = f['hist'].attrs['toCGS']
        
        nH = f['loghydrogendensity'][:] 
        temp_nH = f['logtemperature'][:]
    
    cmap = 'viridis'
    fontsize = 12
    contourlevels = [0.99, 0.9, 0.5]
    contourstyles = ['dotted', 'dashed', 'solid']
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(11., 3.5))
    
    _hist_rho = np.log10(hist_rho / np.diff(rho)[:, np.newaxis] \
                     / np.diff(temp_rho)[np.newaxis, :])
    _hist_rho += np.log10(hist_rho_toCGS)
    _hist_nH = np.log10(hist_nH / np.diff(nH)[:, np.newaxis] \
                     / np.diff(temp_nH)[np.newaxis, :])
    _hist_nH += np.log10(hist_nH_toCGS)
    
    ax1.set_title('$\\rho \\rightarrow \\mathrm{n}_{\\mathrm{H}}$ assuming $X=0.752$',
                  fontsize=fontsize)
    pu.setticks(ax1, fontsize - 1.)
    ax1.set_xlabel('$\\log_{10} \\, \\rho \\; [\\mathrm{H} \\, \\mathrm{cm}^{-3}, X=0.752]$',
                   fontsize=fontsize)
    ax1.set_ylabel('$\\log_{10}$ T [K]', fontsize=fontsize)
    ax1.pcolormesh(rho, temp_rho, _hist_rho.T, cmap=cmap)
    pu.add_2dhist_contours(ax1, hist_rho, [rho, temp_rho], [0, 1],
                           mins=None, maxs=None, histlegend=False, 
                           fraclevels=True, levels=contourlevels, legend=True, 
                           dimlabels=None, legendlabel=None,
                           legendlabel_pre=None, shiftx=0., shifty=0., 
                           dimshifts=None, colors=['red'] * len(contourlevels),
                           linestyles=contourstyles, linewidth=1.)
    handles = [mlines.Line2D((), (), label='{:.1f}%'.format(level*100), 
                             color='red', linewidth=1., linestyle=ls)\
               for level, ls in zip(contourlevels, contourstyles)]
    leg = ax1.legend(fontsize=fontsize - 1., handlelength=1.5,
                     loc='upper left')
    leg.set_title("enclosed mass", prop = {'size': fontsize - 1.}) 
    
    ax1.set_title('$\\mathrm{n}_{\\mathrm{H}}$ from FIRE X',
                  fontsize=fontsize)
    pu.setticks(ax1, fontsize - 1.)
    ax1.set_xlabel('$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{-3}$',
                   fontsize=fontsize)
    ax1.set_ylabel('$\\log_{10}$ T [K]', fontsize=fontsize)
    ax1.pcolormesh(nH, temp_nH, _hist_nH.T, cmap=cmap)
    pu.add_2dhist_contours(ax1, hist_nH, [nH, temp_nH], [0, 1],
                           mins=None, maxs=None, histlegend=False, 
                           fraclevels=True, levels=contourlevels, legend=True, 
                           dimlabels=None, legendlabel=None,
                           legendlabel_pre=None, shiftx=0., shifty=0., 
                           dimshifts=None, colors=['red'] * len(contourlevels),
                           linestyles=contourstyles, linewidth=1.)
    
    ax3.set_title('comparison', fontsize=fontsize)
    pu.setticks(ax3, fontsize - 1.)
    ax3.set_xlabel('$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{-3}$',
                   fontsize=fontsize)
    ax1.set_ylabel('$\\log_{10}$ T [K]', fontsize=fontsize)
    pu.add_2dhist_contours(ax3, hist_rho, [rho, temp_rho], [0, 1],
                           mins=None, maxs=None, histlegend=False, 
                           fraclevels=True, levels=contourlevels, legend=True, 
                           dimlabels=None, legendlabel='encl. mass fraction',
                           legendlabel_pre=None, shiftx=0., shifty=0., 
                           dimshifts=None, colors=['C0'] * len(contourlevels),
                           linestyles=contourstyles, linewidth=1.5)
    pu.add_2dhist_contours(ax3, hist_nH, [nH, temp_nH], [0, 1],
                           mins=None, maxs=None, histlegend=False, 
                           fraclevels=True, levels=contourlevels, legend=True, 
                           dimlabels=None, legendlabel=None,
                           legendlabel_pre=None, shiftx=0., shifty=0., 
                           dimshifts=None, colors=['C1'] * len(contourlevels),
                           linestyles=contourstyles, linewidth=1.)
    handles = [mlines.Line2D((), (), label='sim. X', color='C1', 
                              linewidth=1., linestyle='solid'),
               mlines.Line2D((), (), label='$X=0.752$', color='C0', 
                              linewidth=1., linestyle='solid')]     
    ax3.legend(handles=handles, fontsize=fontsize, loc='upper right')
    