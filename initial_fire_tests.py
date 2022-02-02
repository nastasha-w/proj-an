
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

# TO TEST FIRE BRANCH UNITS:
# - coordinate selections: region (EAGLE), ppp (EAGLE with some hacks? BAHAMAS?), ppv (EAGLE)
# - OnEquationOfState -> SFR: check same T4/excl. selections
# luminosity_calc_halpha_fromSFR: conv. factor uses EAGLE raw units -> check conv. factor in FIRE units

# TODO updates:
# FIRE h1 option to use own tracked fraction


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
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, 
                                        figsize=(11., 3.5),
                                        gridspec_kw={'wspace': 0.4})
    
    _hist_rho = np.log10(hist_rho / np.diff(rho)[:, np.newaxis] \
                     / np.diff(temp_rho)[np.newaxis, :])
    _hist_rho += np.log10(hist_rho_toCGS)
    _hist_nH = np.log10(hist_nH / np.diff(nH)[:, np.newaxis] \
                     / np.diff(temp_nH)[np.newaxis, :])
    _hist_nH += np.log10(hist_nH_toCGS)
    
    vmin = min([np.min(_hist_rho[np.isfinite(_hist_rho)]), 
                np.min(_hist_nH[np.isfinite(_hist_nH)])])
    vmax = max([np.max(_hist_rho), np.max(_hist_nH)])
    clabel_rho = '$\\log_{10} \\, \\partial^2\\mathrm{M} \\,/\\,'+\
                ' \\partial \\log_{10} \\rho \\, ' +\
                ' \\partial \\log_{10} \\mathrm{T} \\;' +\
                '[\\mathrm{g}]$'
    clabel_nH = '$\\log_{10} \\, \\partial^2\\mathrm{M} \\,/\\,'+\
                ' \\partial \\log_{10} \\mathrm{n}_{\\mathrm{H}} \\, ' +\
                ' \\partial \\log_{10} \\mathrm{T} \\;' +\
                '[\\mathrm{g}]$'            
                
    ax1.set_title('$\\rho \\rightarrow \\mathrm{n}_{\\mathrm{H}}$ assuming $X=0.752$',
                  fontsize=fontsize)
    pu.setticks(ax1, fontsize - 1.)
    ax1.set_xlabel('$\\log_{10} \\, \\rho \\; [\\mathrm{H} \\, \\mathrm{cm}^{-3}, X=0.752]$',
                   fontsize=fontsize)
    ax1.set_ylabel('$\\log_{10}$ T [K]', fontsize=fontsize)
    img = ax1.pcolormesh(rho, temp_rho, _hist_rho.T, cmap=cmap, 
                         vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax1, label=clabel_rho)
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
    leg = ax1.legend(handles=handles, fontsize=fontsize - 1., handlelength=1.5,
                     loc='upper right')
    leg.set_title("encl. mass", prop = {'size': fontsize - 1.}) 
    
    ax2.set_title('$\\mathrm{n}_{\\mathrm{H}}$ from FIRE X',
                  fontsize=fontsize)
    pu.setticks(ax2, fontsize - 1.)
    ax2.set_xlabel('$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{-3}$',
                   fontsize=fontsize)
    ax2.set_ylabel('$\\log_{10}$ T [K]', fontsize=fontsize)
    img = ax2.pcolormesh(nH, temp_nH, _hist_nH.T, cmap=cmap,
                         vmin=vmin, vmax=vmax)
    plt.colorbar(img, ax=ax2, label=clabel_nH)
    pu.add_2dhist_contours(ax2, hist_nH, [nH, temp_nH], [0, 1],
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
    ax3.set_ylabel('$\\log_{10}$ T [K]', fontsize=fontsize)
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
    handles = [mlines.Line2D((), (), label='$X=0.752$', color='C0', 
                              linewidth=1., linestyle='solid'),
               mlines.Line2D((), (), label='sim. X', color='C1', 
                              linewidth=1., linestyle='solid'),
               ]     
    ax3.legend(handles=handles, fontsize=fontsize - 1., loc='upper right')
    
    plt.savefig(ddir + 'phasediagram_firesnap.pdf', format='pdf', 
                bbox_inches='tight')
                
def plot_simple_surfdensplot(plottype='firesnap'):
    fn = 'mass_density_maps_firesnap.hdf5'
    
    with h5py.File(ddir + fn, 'r') as f:
        mass = f['massmap'][:]
        dens_mass = f['densmap_massweighted'][:]
        
        xax = f['Header/inputpars'].attrs['Axis1']
        yax = f['Header/inputpars'].attrs['Axis2'] 
        center = f['Header/inputpars'].attrs['centre_codeunits']
        Ls = f['Header/inputpars'].attrs['Ls_codeunits']
        codelength_cgs = f['Header/inputpars'].attrs['codeunits_length_cgs']
        aexp = f['Header/cosmopars'].attrs['a']
        extent = [center[xax] - 0.5 * Ls[xax], 
                  center[xax] + 0.5 * Ls[xax],
                  center[yax] - 0.5 * Ls[yax], 
                  center[yax] + 0.5 * Ls[yax],
                  ]
    extent = [coord * codelength_cgs / rfd.uf.c.cm_per_mpc / aexp \
              for coord in extent]
    extent = tuple(extent)
    xlabel = ['X', 'Y', 'Z'][xax] + ' [cMpc]'  
    ylabel = ['X', 'Y', 'Z'][yax] + ' [cMpc]'

    clabel_mass = '$\\log_{10} \\Sigma_{\\mathrm{gas}} \\;' +\
                  ' [\\mathrm{H} \\, \\mathrm{cm}^{-2}, X=0.752]$'
    clabel_dens = '$\\log_{10} \\rho_{\\mathrm{gas}} \\;' +\
                  ' [\\mathrm{H} \\, \\mathrm{cm}^{-3}, X=0.752]$'
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(11., 5.),
                                   gridspec_kw={'wspace': 0.4})
    fontsize = 12
    rho_to_nH = 0.752 / (rfd.uf.c.atomw_H * rfd.uf.c.u)
    
    _massmap = np.log10(mass.T) + np.log10(rho_to_nH)
    vmin = np.min(_massmap[np.isfinite(_massmap)])
    vmax = np.max(_massmap)
    cmap_mass = pu.paste_cmaps(['gist_gray', 'viridis'], [vmin, 17., vmax], 
                               trunclist=[[0.0, 0.7], [0., 1.]], 
                               transwidths=None)
    img = ax1.imshow(_massmap, 
                     extent=extent, origin='lower', 
                     interpolation='nearest', cmap=cmap_mass)
    plt.colorbar(img, ax=ax1, label=clabel_mass)
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_title('Gas surface density ($\\mathrm{N}_{\\mathrm{H}}$ units)', 
                  fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize - 1.)
    
    _densmap = np.log10(dens_mass.T) + np.log10(rho_to_nH)
    vmind = np.min(_densmap[np.isfinite(_densmap)])
    vmaxd = np.max(_densmap)
    edges = [float(vmind), -7., float(vmaxd)]
    cmap_dens = pu.paste_cmaps(['gist_yarg', 'viridis'], [vmind, -7., vmaxd], 
                               trunclist=[[0.3, 1.], [0., 1.]], 
                               transwidths=None)
    img = ax2.imshow(_densmap, 
                     extent=extent, origin='lower', 
                     interpolation='nearest', cmap=cmap_dens)
    plt.colorbar(img, ax=ax2, label=clabel_dens)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    ax2.set_ylabel(ylabel, fontsize=fontsize)
    ax2.set_title('Mass-weighted gas density ($\\mathrm{n}_{\\mathrm{H}}$ units)', 
                  fontsize=fontsize)
    ax2.tick_params(labelsize=fontsize - 1.)
    
    plt.savefig(ddir + 'massmap_firesnap.pdf', format='pdf',
                bbox_inches='tight')
                
    