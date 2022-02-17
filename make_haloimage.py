#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:25:12 2020

@author: Nastasha
"""

import numpy as np
import h5py
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.patches as mpatch
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.cm as cm
import matplotlib.ticker as ticker

import make_maps_v3_master as m3
import eagle_constants_and_units as c
import make_maps_opts_locs as ol


halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'

mdir = '/net/luttero/data2/imgs/pretty/'

def selecthalo(logM200c, _halocat=halocat, margin=0.05, randomseed=None):
    '''
    selects a random halo in the target mass range and prints basic data about
    it
    
    Parameters
    ----------
    logM200c : float [log10 Msun]
        target halo mass M200c.
    _halocat : string, optional
        The name of the file containing the halo data. If no path is included,
        the file is assumed to be in make_maps_opts_locs.pdir
        The default is halocat.
    margin : float, optional
        the maximum difference with the target halo mass allowed (in log10 
        Msun). A rondom halo is selected within this range.
        The default is 0.05.
    randomseed: int, optional
        seed for the random number generator.
        The default is None.

    Raises
    ------
    RuntimeError
        If there is no halo in the selected mass range.

    Returns
    -------
    galid: string
        the galaxy ID (EAGLE online halo catalogue)
    m200c: float
        the mass M200c of the selected halo (log10 Msun)
    centre: list of 3 floats
        the halo centre (central galaxy centre of mass) in cMpc
    R200:
        the halo size R200c in cMpc
    '''
    
    if '/' not in _halocat:
        _halocat = ol.pdir + _halocat
    with h5py.File(_halocat, 'r') as hc:
        boxdata = {key: val for key, val in hc['Header'].attrs.items()}
        cosmopars = {key: val for key, val in\
                     hc['Header/cosmopars'].attrs.items()}
        m200c = np.log10(hc['M200c_Msun'][:])
        galid = hc['galaxyid'][:]
        close = np.where(np.abs(m200c - logM200c) < margin)[0]
        if len(close) < 1:
            raise RuntimeError('No haloes in the selected mass range')
        np.random.seed(seed=randomseed)
        ind = np.random.choice(close)
        m200c = m200c[ind]
        galid = galid[ind]
        cenx = hc['Xcom_cMpc'][ind]
        ceny = hc['Ycom_cMpc'][ind]
        cenz = hc['Zcom_cMpc'][ind]
        R200 = hc['R200c_pkpc'][ind] * 1e-3 / cosmopars['a'] # to cMpc
        
    print('Selected galaxy {galid} with log10 M200c / Msun {m200c}'.format(\
          galid=galid, m200c=m200c))
    print('Center [{}, {}, {}] cMpc, R200c {R200c} cMpc'.format(\
          cenx, ceny, cenz, R200c=R200))
    print(boxdata)
    return galid, m200c, [cenx, ceny, cenz], R200

def getimgs(cen, size, sizemargin=2., imgtype='CV'):
    '''
    get images of a halo in a number of properties (see listed arguments)

    Parameters
    ----------
    cen : list of 3 floats
        halo centre [cMpc].
    size : float
        halo size [cMpc].
    sizemargin : float, optional
        radius of the projected cube in units of the size. The default is 2..
    imgtype: string, optional
        which images to make. 
        'CV' is the original temperature, density, O VII absorption, 
        O VII emission plot.
        'SM' is the density, metallicity plot for Smita Mathur
    Returns
    -------
    names : list of strings
        the names of the files containing the images.

    '''
    simnum = 'L0100N1504'
    snapnum = 27
    centre = cen
    L_z = size * sizemargin * 2. # radius
    L_x = L_y = L_z
    npix_x = 400
    npix_y = npix_x
    firstargs = (simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y)
    
    kwargs_all = {'excludeSFRW': 'T4',\
                  'excludeSFRQ': 'T4',\
                  'var': 'REFERENCE',\
                  'axis': 'z',\
                  'periodic': False,\
                  'saveres': True,\
                  'hdf5': True,\
                  'ompproj': True}
    if imgtype == 'CV':
        argsets = [[('basic',), {'quantityW': 'Mass', 'ptypeQ': 'basic',
                                 'quantityQ': 'Temperature'}],
                   [('basic',), {'quantityW': 'Mass', 'ptypeQ': 'basic',
                                 'quantityQ': 'Density'}],
                   [('coldens',), {'ionW': 'o7', 'abundsW': 'Pt'}],
                   [('emission',), {'ionW': 'o7r', 'abundsW': 'Sm'}],
                ]
    elif imgtype == 'SM':
         argsets = [[('basic',), {'quantityW': 'Mass', 'ptypeQ': 'basic',
                                  'quantityQ': 'Density'}],
                    [('basic',), {'quantityW': 'Mass', 'ptypeQ': 'basic',
                                  'quantityQ': 'Metallicity'}],
                ]
    else:
        raise ValueError('{} is not a valid imgtype option'.format(imgtype))

    names = []
    for argset in argsets:
        args = firstargs + argset[0]
        kwargs = kwargs_all.copy()
        kwargs.update(argset[1])
        
        name = m3.make_map(*args, nameonly=True, **kwargs)
        inclq = False
        if isinstance(name, tuple):
            if name[1] is None:
                name = name[0]
            else:
                inclq = True
                names.append(name[0])
                names.append(name[1])
                if not (os.path.isfile(name[0]) and os.path.isfile(name[1])):
                    m3.make_map(*args, nameonly=False, **kwargs)
        if not inclq:
            names.append(name)
            if not os.path.isfile(name): 
                m3.make_map(*args, nameonly=False, **kwargs)
                
    print('Done creating images:')
    print(names)
    return names    

def plotimgs(names, R200c, M200c, galid, imgtype='CV'):
    
    while None in names:
        names.remove(None)
    
    fontsize = 12
    if imgtype == 'CV':
         maptypes = ['Density', 'Temperature', 'coldens_o7', 'emission_o7r']
    elif imgtype == 'SM':
        maptypes = ['Density', 'Metallicity']
    ncols = min(len(maptypes), 4)
    nrows = (len(maptypes) - 1) // ncols + 1
    figwidth = 11. * float(ncols) / 4.
    
    panelwidth = figwidth / ncols
    panelheight = panelwidth
    cheight = 0.5
    height_ratios = [panelheight, cheight] * nrows
    
    figheight = sum(height_ratios)
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows * 2, hspace=0.0, wspace=0.0,\
                        width_ratios=[panelwidth] * ncols,\
                        height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[2 * (i // ncols), i % ncols]) \
            for i in range(len(maptypes))]
    caxes = [fig.add_subplot(grid[2 * (i // ncols) + 1, i % ncols]) \
            for i in range(len(maptypes))]
    
    for mi, mt in enumerate(maptypes):
        ax = axes[mi]
        _cax = caxes[mi]
        _cax.axis('off')
        _l, _b, _w, _h = (_cax.get_position()).bounds
        margin = panelwidth * 0.07 / figwidth
        cax = fig.add_axes([_l + margin, _b,\
                            _w - 2.* margin, _h])
        
        
        match = [(name.split('/')[-1]).startswith(mt) for name in names]
        match = np.where(match)[0]
        if len(match) == 0:
            raise RuntimeError('{} map not found for names {}'.format(\
                                mt, names))
        elif len(match) > 1:
            fns = [names[i] for i in match]
            if not np.all([fn == fns[0] for fn in fns]):
                raise RuntimeError('{} map has mltiple options: {}'.format(\
                                    mt, fns))
            fn = fns[0]
        else:
            fn = names[match[0]]
        
        with h5py.File(fn, 'r') as mf:
            _map = mf['map'][:]
            _min = mf['map'].attrs['minfinite']
            _max = mf['map'].attrs['max']
            cosmopars = {key: val for key, val \
                         in mf['Header/inputpars/cosmopars'].attrs.items()}
            log = bool(mf['Header/inputpars'].attrs['log'])
            if not log:
                _map = np.log10(_map)
            
            axis = mf['Header/inputpars'].attrs['axis'].decode()
            if axis == 'z':
                l0 = 'x'
                l1 = 'y'
            elif axis == 'x':
                l0 = 'y'
                l1 = 'z'
            elif axis == 'y':
                l0 = 'z'
                l1 = 'x'
            _l0 = mf['Header/inputpars'].attrs['L_{ax}'.format(ax=l0)]
            _l1 = mf['Header/inputpars'].attrs['L_{ax}'.format(ax=l1)]
            #pixsize_0_cMpc = _l0 / float(mf['Header/inputpars'].attrs('npix_x'))
            #pixsize_1_cMpc = _l1 / float(mf['Header/inputpars'].attrs('npix_y'))
            
        if mt == 'Mass':
            clabel = '$\\log_{10} \\Sigma_{\\mathrm{gas}} \\; [\\mathrm{M}_{\\mathrm{\\odot}} \\,/\\, \\mathrm{pkpc}^{2}]$'
            vmin = -np.inf
            vmax = np.inf
            cmap = cm.get_cmap('viridis')
            units = c.solar_mass / (c.cm_per_mpc * 1e-3)**2
        elif mt == 'Temperature':
            clabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
            vmin = -np.inf
            vmax = np.inf
            cmap = cm.get_cmap('plasma')
            units = 1.     
        elif mt == 'Density':
            clabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{-3}]$'
            vmin = -np.inf
            vmax = np.inf
            cmap = cm.get_cmap('viridis')
            units = c.atomw_H * c.u / 0.752
        elif mt == 'Metallicity':
            clabel = '$\\log_{10} \\, \\mathrm{Z} \\; [\\mathrm{Z}_{\\odot}]$'
            vmin = -2.5
            vmax = np.inf
            cmap = cm.get_cmap('magma')
            units = ol.Zsun_ea
        elif mt == 'coldens_o7':
            clabel = '$\\log_{10} \\, \\mathrm{N}(\mathrm{O\\,VII}) \\; [\\mathrm{cm}^{-2}]$'
            vmin = 14.5
            vmax = np.inf
            cmap = cm.get_cmap('magma')          
            units = 1.
        elif mt == 'emission_o7r':
            clabel = '$\\log_{10} \\, \\mathrm{SB}(\mathrm{O\\,VII \, r}) \\; [\\mathrm{ph} \\,/\\,\\mathrm{s} \\, \\mathrm{cm}^{2} \\mathrm{sr}]$'
            vmin = -2.5
            vmax = np.inf
            cmap = cm.get_cmap('inferno')
            
            units = 1.
            
        _map -= np.log10(units)
        _min -= np.log10(units)
        _max -= np.log10(units)
        vmin = max(vmin, _min)
        vmax = min(vmax, _max)
        extent = (-0.5 * _l0, 0.5*_l0, -0.5*_l1, 0.5*_l1)
        cmap.set_under(cmap(0.))
        cmap.set_over(cmap(1.))
        if _min < vmin or np.any(np.logical_not(np.isfinite(_map))):
            if _max > vmax:
                extend = 'both'
            else:
                extend = 'min'
        elif _max > vmax:
            extend = 'max'
        else:
            extend = 'neither'
            
        img = ax.imshow(_map.T, origin='lower', interpolation='nearest',
                        extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
            
        cax.tick_params(labelsize=fontsize - 1)
        cax.set_aspect(0.1)
        #locator = ticker.MaxNLocator(nbins=5)
        plt.colorbar(img, cax=cax, extend=extend, orientation='horizontal',
                     aspect=0.05) #ticks=locator, 
        cax.set_xlabel(clabel, fontsize=fontsize)
        
        ax.tick_params(left=False, bottom=False, labelbottom=False,
                       labelleft=False)

        patches = [mpatch.Circle((0., 0.), R200c)] # x, y axes only
        patheff = [mppe.Stroke(linewidth=1.2, foreground="black"),
                   mppe.Stroke(linewidth=0.7, foreground="white"),
                   mppe.Normal()] 
        collection = mcol.PatchCollection(patches)
        collection.set(edgecolor='white', facecolor='none', linewidth=0.7,
                       path_effects=patheff)    
        ax.add_collection(collection)
        
        patheff_text = [mppe.Stroke(linewidth=2.0, foreground="white"),
                        mppe.Stroke(linewidth=0.4, foreground="black"),
                        mppe.Normal()]  
        if mi == 0:
            ax.text(0.05, 0.95,\
                    '$\\log_{{10}} \\mathrm{{M}}_{{\\mathrm{{200c}}}} / \\mathrm{{M}}_{{\\odot}} = {M200c:.1f}$'.format(M200c=M200c),\
                    fontsize=fontsize, verticalalignment='top',
                    horizontalalignment='left', transform=ax.transAxes,
                    path_effects=patheff_text)
        if mi == 1:
            ax.text(2.**-0.5 * R200c, 2.**-0.5 * R200c,
                    '$\\mathrm{R}_{\\mathrm{200c}}$',
                    fontsize=fontsize, verticalalignment='bottom',
                    horizontalalignment='left',
                    path_effects=patheff_text)
                 
            xlim = ax.get_xlim()
            lenline = 250. * 1e-3 / cosmopars['a'] / (xlim[1] - xlim[0])
            _text = '250 pkpc'
            
            ax.plot([0.1, 0.1 + lenline], [0.05, 0.05],
                    color='white', linewidth=0.7, path_effects=patheff,
                    transform=ax.transAxes)
            ax.text(0.1 + 0.5 * lenline, 0.06, _text,
                    fontsize=fontsize, path_effects=patheff_text,
                    transform=ax.transAxes, verticalalignment='bottom',
                    horizontalalignment='center')
    
    if imgtype == 'CV':
        outname = 'galaxy{}_nH_T_o7_o7r.eps'.format(galid)
    elif imgtype == 'SM':
        outname = 'galaxy{}_nH_Z.eps'.format(galid)
    plt.savefig(mdir + outname, format='eps', bbox_inches='tight')

if __name__ == '__main__':
    args = sys.argv
    imgtype = 'SM'

    if imgtype == 'CV':
        sizemargin = 2.
    elif imgtype == 'SM':
        sizemargin = 1.5
    m200_tar = float(sys.argv[1])
    if len(sys.argv) > 2:
        randomseed = int(sys.argv[2])
    else:
        randomseed = 0
    out = selecthalo(m200_tar, _halocat=halocat, margin=0.05,\
                     randomseed=randomseed)
    galid, m200c, cen, R200 = out
    
    filens = getimgs(cen, R200, sizemargin=sizemargin, imgtype=imgtype)
    plotimgs(filens, R200, m200c, galid, imgtype=imgtype)
    print('Made image for galaxy {}'.format(galid))
    print('Saved in {}'.format(mdir))
    
    
