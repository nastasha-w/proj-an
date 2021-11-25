
import numpy as np
import string
import h5py
import os

import matplotlib.pyplot as plt

import make_maps_v3_master as m3

basepath = '/Users/Nastasha/ciera/tests/make_maps_select/'
ndir_newbranch = basepath + 'maps_select_branch/'
ndir_master = basepath + 'maps_old_master/'
mdir = basepath + 'imgs/'

def printinfo(pathobj):
     if not isinstance(pathobj, h5py._hl.dataset.Dataset):
         print('keys and datasets:')
         for key in pathobj:
             print(f'{key}:\t {pathobj[key]}')
     else:
         print(pathobj)
         print(pathobj[(slice(None, 2, None),) * len(pathobj.shape)])
     print('\nattributes:')
     for key, val in pathobj.attrs.items():
         print(f'{key}:\t {val}')

def runmaps_selcheck(mode='new'):
    simnum = 'L0012N0188'
    snapnum = 27
    centre = [3.125, 3.125, 3.125]
    L_x, L_y, L_z = (6.25, ) * 3
    npix_x = 400
    npix_y = 400
    
    args_all = (simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y)
    kwargs_all = {'excludeSFRW': 'T4', 'excludeSFRQ': 'T4', 
                  'parttype': '0', 'periodic': False, 'saveres': True,
                  'ompproj': False, 'hdf5': True}
    
    args_grid = [('basic',), ('basic',), ('emission',),  
                 ('emission',), ('coldens',)]
    kwargs_grid = [{'quantityW': 'Mass', 'ptypeQ': 'basic', 
                    'quantityQ': 'Temperature'},
                   {'quantityW': 'Mass', 'ptypeQ': 'basic', 
                    'quantityQ': 'Density'},
                   {'ionW': 'o8', 'ptypeQ': 'basic',
                    'quantityQ': 'Density'},
                   {'ionW': 'o8', 'abundsW': 'Sm'},
                   {'ionW': 'o6', 'abundsW': 'Pt'},
                   ]
    
    # do multiple selections work together correctly? Also: SFR settings passed through?
    selects_rhoT = [[({'ptype': 'basic', 'quantity': 'Temperature'}, None, 1e5),
                     ({'ptype': 'basic', 'quantity': 'Density'}, None, 1e-29)],
                    [({'ptype': 'basic', 'quantity': 'Temperature'}, None, 1e5),
                     ({'ptype': 'basic', 'quantity': 'Density'}, 1e-29, None)],
                    [({'ptype': 'basic', 'quantity': 'Temperature'}, 1e5, None),
                     ({'ptype': 'basic', 'quantity': 'Density'}, None, 1e-29)],
                    [({'ptype': 'basic', 'quantity': 'Temperature'}, 1e5, None),
                     ({'ptype': 'basic', 'quantity': 'Density'}, 1e-29, None)]]
    # Do all None/value combinations work right?
    selects_rho = [[({'ptype': 'basic', 'quantity': 'Density'}, None, 1e-29)],
                   [({'ptype': 'basic', 'quantity': 'Density'}, 1e-29, 1e-27)],
                   [({'ptype': 'basic', 'quantity': 'Density'}, 1e-27, None)],
                   [({'ptype': 'basic', 'quantity': 'Density'}, None, None)]
                   ]
    # qualitative: does a Lumdens selection work?
    # o8 K-alpha: volume average is 10**37.5 erg/s/cMpc**3 -> ~10**-36 erg/s/cm**3
    selects_Lo8 = [[({'ptype': 'Lumdens', 'ion': 'o8', 'abunds': 'Pt'}, None, 1e-33)],
                   [({'ptype': 'Lumdens', 'ion': 'o8', 'abunds': 'Pt'}, 1e-33, None)]]
    # qualitative: does an Niondens selection work?
    # for O VI: nH ~ 1e-4 cm**-3, nO/nH ~ 1e-5, O6/O ~ 0.01
    selects_no6 = [[({'ptype': 'Niondens', 'ion': 'o6', 'excludeSFR': 'T4'}, None, 1e-11)],
                   [({'ptype': 'Niondens', 'ion': 'o6', 'excludeSFR': 'T4'}, 1e-11, None)]]

    selects_grid = [selects_rhoT, selects_rhoT, selects_rho, selects_Lo8, selects_no6]

    for _kwargs, _args, _sels in zip(kwargs_grid, args_grid, selects_grid):
        kwargs = kwargs_all.copy()
        kwargs.update(_kwargs)
        args = args_all + _args
        
        if mode == 'old':
            m3.ol.ndir = ndir_master
            m3.make_map(*args, nameonly=False, **kwargs)
            
        elif mode == 'new':
            m3.ol.ndir = ndir_newbranch
            for _select in _sels + [None]:
                fns = m3.make_map(*args, nameonly=True, select=_select, 
                                  **kwargs)
                if fns[1] is None:
                    run = not os.path.isfile(fns[0])
                else:
                    run = not (os.path.isfile(fns[0]) and \
                               os.path.isfile(fns[1]))
                if run:
                     m3.make_map(*args, nameonly=False, select=_select, 
                                 **kwargs)



def readmapdata(filen):
    with h5py.File(filen, 'r') as _f:
        arr1 = _f['map'][:]
        axis1 = _f['Header/inputpars'].attrs['axis'].decode()
        _axes = ['x', 'y', 'z']
        if axis1 == 'z':
            xax = 0
            yax = 1
        elif axis1 == 'x':
            xax = 1
            yax = 2
        elif axis1 == 'y':
            xax = 2
            yax = 1
        L0 = _f['Header/inputpars'].attrs['L_{}'.format(_axes[xax])]
        L1 = _f['Header/inputpars'].attrs['L_{}'.format(_axes[yax])]
        centre = _f['Header/inputpars'].attrs['centre']
        extent1 = (centre[xax] - 0.5 * L0, centre[xax] + 0.5 * L0,
                   centre[yax] - 0.5 * L1, centre[yax] + 0.5 * L1)
        xname1 = string.capwords(_axes[xax]) + ' [cMpc]'
        yname1 = string.capwords(_axes[yax]) + ' [cMpc]'
    return arr1, extent1, (xname1, yname1)

def comparemaps(file1, file2, vmin=None, diffmax=None, 
                outname=mdir + 'comp.pdf',
                fontsize=12, clabel=''):
    '''
    compare 2 maps that should be the same or very similar
    '''
    with h5py.File(file1, 'r') as _f:
        arr1 = _f['map'][:]
        axis1 = _f['Header/inputpars'].attrs['axis'].decode()
        _axes = ['x', 'y', 'z']
        if axis1 == 'z':
            xax = 0
            yax = 1
        elif axis1 == 'x':
            xax = 1
            yax = 2
        elif axis1 == 'y':
            xax = 2
            yax = 1
        L0 = _f['Header/inputpars'].attrs['L_{}'.format(_axes[xax])]
        L1 = _f['Header/inputpars'].attrs['L_{}'.format(_axes[yax])]
        centre = _f['Header/inputpars'].attrs['centre']
        extent1 = (centre[xax] - 0.5 * L0, centre[xax] + 0.5 * L0,
                   centre[yax] - 0.5 * L1, centre[yax] + 0.5 * L1)
        xname1 = string.capwords(_axes[xax]) + ' [cMpc]'
        yname1 = string.capwords(_axes[yax]) + ' [cMpc]'
        
    with h5py.File(file2, 'r') as _f:
        arr2 = _f['map'][:]
        axis2 = _f['Header/inputpars'].attrs['axis'].decode()
        _axes = ['x', 'y', 'z']
        if axis2 == 'z':
            xax = 0
            yax = 1
        elif axis2 == 'x':
            xax = 1
            yax = 2
        elif axis2 == 'y':
            xax = 2
            yax = 1
        L0 = _f['Header/inputpars'].attrs['L_{}'.format(_axes[xax])]
        L1 = _f['Header/inputpars'].attrs['L_{}'.format(_axes[yax])]
        centre = _f['Header/inputpars'].attrs['centre']
        extent2 = (centre[xax] - 0.5 * L0, centre[xax] + 0.5 * L0,
                   centre[yax] - 0.5 * L1, centre[yax] + 0.5 * L1)
        xname2 = string.capwords(_axes[xax]) + ' [cMpc]'
        yname2 = string.capwords(_axes[yax]) + ' [cMpc]'
    
    if not np.allclose(extent1, extent2) and xname1 == xname2 and \
        yname1 == yname2 and arr2.shape == arr1.shape:
        msg = 'The regions or pixels in the two maps do not match'
        raise RuntimeError(msg)

    Vmin = min(np.min(arr1[np.isfinite(arr1)]), 
               np.min(arr2[np.isfinite(arr2)]))
    if vmin is not None:
       Vmin = max(vmin, Vmin)
    Vmax = max(np.max(arr1[np.isfinite(arr1)]), 
               np.max(arr2[np.isfinite(arr2)]))
    diff = arr1 - arr2
    if diffmax is not None:
        maxdiff = diffmax
    else:    
        maxdiff = np.max(np.abs(diff)[np.isfinite(np.abs(diff))])
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, 
                                                           sharex=False, 
                                                           sharey=False)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=None, hspace=None)

    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.imshow(arr1.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest') 
    ax1.set_title('test run', fontsize=fontsize)
    ax1.set_xlabel(xname1, fontsize=fontsize)
    ax1.set_ylabel(yname1, fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax1)
    cax1 = div.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(img, cax=cax1)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2.tick_params(labelsize=fontsize)
    ax2.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax2.imshow(arr2.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest') 
    ax2.set_title('check run',fontsize=fontsize)
    ax2.set_xlabel(xname2, fontsize=fontsize)
    ax2.set_ylabel(yname2, fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax2)
    cax2 = div.append_axes("right",size="5%",pad=0.1)
    cbar2 = plt.colorbar(img, cax=cax2)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax3.tick_params(labelsize=fontsize)
    ax3.patch.set_facecolor('black')
    img = ax3.imshow((diff).T,origin='lower', cmap=cm.get_cmap('RdBu'), 
                     vmin = -maxdiff, vmax=maxdiff, interpolation='nearest') 
    ax3.set_title('test - check', fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax3)
    cax3 = div.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(img, cax=cax3)
    cbar3.solids.set_edgecolor("face")
    cbar3.ax.set_ylabel(r'$\Delta$' + clabel, fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)
    
    ax6.set_title('test - check')
    ax6.hist(np.ndarray.flatten(diff[np.isfinite(diff)]), log=True, bins=50)
    ax6.set_ylabel('number of pixels', fontsize = fontsize)
    ax6.set_xlabel(r'$\Delta$' + clabel, fontsize=fontsize)

    
    arrbins = np.arange(Vmin, Vmax + 0.0001 * (Vmax-Vmin) / float(nbins), 
                       (Vmax - Vmin) / float(nbins))
    diffbins = np.arange(-maxdiff, maxdiff + 0.0002 * maxdiff / float(nbins), 
                         2. * maxdiff / float(nbins))

    ax4.tick_params(labelsize=fontsize)
    ax4.hist(arr1[np.isfinite(arr1)],bins = arrbins, log=True )    
    ax4.set_title('test run',fontsize=fontsize)
    ax4.set_xlabel(clabel,fontsize=fontsize)
    ax4.set_ylabel('columns',fontsize=fontsize)
    
    ax5.tick_params(labelsize=fontsize)
    ax5.hist(arr2[np.isfinite(arr2)],bins = arrbins, log=True )    
    ax5.set_title('check run',fontsize=fontsize)
    ax5.set_xlabel(clabel)
    ax5.set_ylabel('columns',fontsize=fontsize)
    
    fig.tight_layout()
    plt.savefig(name, format='pdf')
    
    
def comparepartmaps(partfiles, totfile, vmin=None, diffmax=None, 
                    outname=mdir + 'comp.pdf',
                    fontsize=12, clabel=''):
    '''
    compare 2 maps that should be the same or very similar
    '''
    map_tot, extent_tot, xname_tot, yname_tot = readmapdata(totfile)
    
    
    if not np.allclose(extent1, extent2) and xname1 == xname2 and \
        yname1 == yname2 and arr2.shape == arr1.shape:
        msg = 'The regions or pixels in the two maps do not match'
        raise RuntimeError(msg)

    Vmin = min(np.min(arr1[np.isfinite(arr1)]), 
               np.min(arr2[np.isfinite(arr2)]))
    if vmin is not None:
       Vmin = max(vmin, Vmin)
    Vmax = max(np.max(arr1[np.isfinite(arr1)]), 
               np.max(arr2[np.isfinite(arr2)]))
    diff = arr1 - arr2
    if diffmax is not None:
        maxdiff = diffmax
    else:    
        maxdiff = np.max(np.abs(diff)[np.isfinite(np.abs(diff))])
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, 
                                                           sharex=False, 
                                                           sharey=False)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, 
                        wspace=None, hspace=None)

    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.imshow(arr1.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest') 
    ax1.set_title('test run', fontsize=fontsize)
    ax1.set_xlabel(xname1, fontsize=fontsize)
    ax1.set_ylabel(yname1, fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax1)
    cax1 = div.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(img, cax=cax1)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2.tick_params(labelsize=fontsize)
    ax2.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax2.imshow(arr2.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest') 
    ax2.set_title('check run',fontsize=fontsize)
    ax2.set_xlabel(xname2, fontsize=fontsize)
    ax2.set_ylabel(yname2, fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax2)
    cax2 = div.append_axes("right",size="5%",pad=0.1)
    cbar2 = plt.colorbar(img, cax=cax2)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax3.tick_params(labelsize=fontsize)
    ax3.patch.set_facecolor('black')
    img = ax3.imshow((diff).T,origin='lower', cmap=cm.get_cmap('RdBu'), 
                     vmin = -maxdiff, vmax=maxdiff, interpolation='nearest') 
    ax3.set_title('test - check', fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax3)
    cax3 = div.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(img, cax=cax3)
    cbar3.solids.set_edgecolor("face")
    cbar3.ax.set_ylabel(r'$\Delta$' + clabel, fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)
    
    ax6.set_title('test - check')
    ax6.hist(np.ndarray.flatten(diff[np.isfinite(diff)]), log=True, bins=50)
    ax6.set_ylabel('number of pixels', fontsize = fontsize)
    ax6.set_xlabel(r'$\Delta$' + clabel, fontsize=fontsize)

    
    arrbins = np.arange(Vmin, Vmax + 0.0001 * (Vmax-Vmin) / float(nbins), 
                       (Vmax - Vmin) / float(nbins))
    diffbins = np.arange(-maxdiff, maxdiff + 0.0002 * maxdiff / float(nbins), 
                         2. * maxdiff / float(nbins))

    ax4.tick_params(labelsize=fontsize)
    ax4.hist(arr1[np.isfinite(arr1)],bins = arrbins, log=True )    
    ax4.set_title('test run',fontsize=fontsize)
    ax4.set_xlabel(clabel,fontsize=fontsize)
    ax4.set_ylabel('columns',fontsize=fontsize)
    
    ax5.tick_params(labelsize=fontsize)
    ax5.hist(arr2[np.isfinite(arr2)],bins = arrbins, log=True )    
    ax5.set_title('check run',fontsize=fontsize)
    ax5.set_xlabel(clabel)
    ax5.set_ylabel('columns',fontsize=fontsize)
    
    fig.tight_layout()
    plt.savefig(name, format='pdf')