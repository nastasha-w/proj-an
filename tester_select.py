
import numpy as np
import string
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import make_maps_v3_master as m3

basepath = '/Users/Nastasha/ciera/tests/make_maps_select/'
ndir_newbranch = basepath + 'maps_select_branch/'
ndir_master = basepath + 'maps_old_master/'
mdir = basepath + 'imgs/'

label_no6 = '$\\log_{10} \\, \\mathrm{N}(\\mathrm{O\\,VI}) \\; [\\mathrm{cm}^{-2}]$'
label_emo8 = '$\\log_{10} \\, \\mathrm{SB}(\\mathrm{O\\,VIII}) \\;'+ \
             '[\\mathrm{ph}\\,\\mathrm{s}^{-1}\\mathrm{sr}^{-1}\\mathrm{cm}^{-2}]$'
label_rho_by_o8 = '$\\log_{10} \\, \\langle\\rho\\rangle_{\\mathrm{L}(\\mathrm{O\\,VIII})} \\; [\\mathrm{g} \\, \\mathrm{cm}^{-3}]$'
label_mass = '$\\log_{10} \\, \\Sigma \\; [\\mathrm{g} \\, \\mathrm{cm}^{-2}]$'
label_t_by_mass = '$\\log_{10} \\, \\langle\\mathrm{T}\\rangle_{\\mathrm{M}} \\; [\\mathrm{K}]$'
label_rho_by_mass = '$\\log_{10} \\, \\langle\\rho\\rangle_{\\mathrm{M}} \\; [\\mathrm{g} \\, \\mathrm{cm}^{-3}]$'



def printinfo(pathobj):
     if not isinstance(pathobj, h5py._hl.dataset.Dataset):
         print('keys and datasets:')
         for key in pathobj:
             print('{key}:\t {val}'.format(key=key, val=pathobj[key]))
     else:
         print(pathobj)
         print(pathobj[(slice(None, 2, None),) * len(pathobj.shape)])
     print('\nattributes:')
     for key, val in pathobj.attrs.items():
         print('{key}:\t {val}'.format(key=key, val=val))

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
    # multiple selections: NO
    selects_rhoT = [[({'ptype': 'basic', 'quantity': 'Temperature'}, None, 1e5),
                     ({'ptype': 'basic', 'quantity': 'Density'}, None, 1e-29)],
                    [({'ptype': 'basic', 'quantity': 'Temperature'}, None, 1e5),
                     ({'ptype': 'basic', 'quantity': 'Density'}, 1e-29, None)],
                    [({'ptype': 'basic', 'quantity': 'Temperature'}, 1e5, None),
                     ({'ptype': 'basic', 'quantity': 'Density'}, None, 1e-29)],
                    [({'ptype': 'basic', 'quantity': 'Temperature'}, 1e5, None),
                     ({'ptype': 'basic', 'quantity': 'Density'}, 1e-29, None)]]
    # Do all None/value combinations work right? -> (None, None) does
    # the selection looks ok, but the largest bin is empty. Some weirdness
    # in the weighted densities.
    selects_rho = [[({'ptype': 'basic', 'quantity': 'Density'}, None, 1e-29)],
                   [({'ptype': 'basic', 'quantity': 'Density'}, 1e-29, 1e-28)],
                   [({'ptype': 'basic', 'quantity': 'Density'}, 1e-28, None)],
                   [({'ptype': 'basic', 'quantity': 'Density'}, None, None)]
                   ]
    # qualitative: does a Lumdens selection work? -> seems to
    # o8 K-alpha: volume average is 10**37.5 erg/s/cMpc**3 -> ~10**-36 erg/s/cm**3
    selects_Lo8 = [[({'ptype': 'Lumdens', 'ion': 'o8', 'abunds': 'Pt'}, None, 1e-37)],
                   [({'ptype': 'Lumdens', 'ion': 'o8', 'abunds': 'Pt'}, 1e-37, None)]]
    # qualitative: does an Niondens selection work?
    # for O VI: nH ~ 1e-4 cm**-3, nO/nH ~ 1e-5, O6/O ~ 0.01 -> seems to
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
    return arr1, extent1, xname1, yname1

def comparemaps(file1, file2, vmin=None, diffmax=None, 
                outname=mdir + 'comp.pdf',
                fontsize=12, clabel='', nbins=50):
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
        maxdiff = max(maxdiff, 1e-7)
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, 
                                                           sharex=False, 
                                                           sharey=False,
                                                           figsize=(11., 7.))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.92, 
                        wspace=None, hspace=None)
    title = 'test: \t{}\ncheck: \t{}'
    fig.suptitle(title.format(file1, file2), fontsize=fontsize - 2)
    
    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.imshow(arr1.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest',
                     extent=extent1) 
    ax1.set_title('test run', fontsize=fontsize)
    ax1.set_xlabel(xname1, fontsize=fontsize)
    ax1.set_ylabel(yname1, fontsize=fontsize)
    #div = axgrid.make_axes_locatable(ax1)
    #cax1 = div.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(img, ax=ax1, fraction=0.05, pad=0.05, shrink=0.6)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2.tick_params(labelsize=fontsize)
    ax2.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax2.imshow(arr2.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest',
                     extent=extent1) 
    ax2.set_title('check run',fontsize=fontsize)
    ax2.set_xlabel(xname2, fontsize=fontsize)
    ax2.set_ylabel(yname2, fontsize=fontsize)
    #div = axgrid.make_axes_locatable(ax2)
    #cax2 = div.append_axes("right",size="5%",pad=0.1)
    cbar2 = plt.colorbar(img, ax=ax2, fraction=0.05, pad=0.05, shrink=0.6)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax3.tick_params(labelsize=fontsize)
    ax3.patch.set_facecolor('black')
    img = ax3.imshow((diff).T,origin='lower', cmap=cm.get_cmap('RdBu'), 
                     vmin = -maxdiff, vmax=maxdiff, interpolation='nearest',
                     extent=extent1) 
    ax3.set_title('test - check', fontsize=fontsize)
    #div = axgrid.make_axes_locatable(ax3)
    #cax3 = div.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(img, ax=ax3, fraction=0.05, pad=0.05, shrink=0.6)
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
    plt.savefig(outname, format='pdf')
    
    
def comparepartmaps(partfiles, totfile, partlabels,
                    partweights=None,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'comp.pdf',
                    fontsize=12, clabel='', nbins=100):
    '''
    compare a set of maps that to one they should add up to
    '''
    map_tot, extent_tot, xname_tot, yname_tot = readmapdata(totfile)
    part_map_ext_xname_yname = [readmapdata(pf) for pf in partfiles]
    if not np.all([np.allclose(extent_tot, _extent) and xname_tot == _xname \
                   and yname_tot == _yname and map_tot.shape == _map.shape \
                   for _map, _extent, _xname, _yname in part_map_ext_xname_yname]):
        msg = 'The regions or pixels in the different maps do not match'
        raise RuntimeError(msg)
        
    if partweights is not None:
        weights_map_ext_xname_yname = [readmapdata(pf) for pf in partweights]
        if not np.all([np.allclose(extent_tot, _extent) and xname_tot == _xname \
                   and yname_tot == _yname and map_tot.shape == _map.shape \
                   for _map, _extent, _xname, _yname in \
                   weights_map_ext_xname_yname]):
            msg = 'The regions or pixels in the different weight maps do' +\
                  ' not match'
            raise RuntimeError(msg)
    
    if partweights is None:
        sumarr = np.log10(np.sum([10**_part[0] for _part in \
                                  part_map_ext_xname_yname], axis=0))
    else:
        wsum = np.log10(np.sum([10**_part[0] for _part in \
                                weights_map_ext_xname_yname], axis=0))
        sumarr = np.log10(np.sum([10**(_part[0] + _wt[0]) for _part, _wt in \
                                  zip(part_map_ext_xname_yname,
                                      weights_map_ext_xname_yname)], axis=0))
        sumarr -= wsum
        
    Vmin = min([np.min(map_tot[np.isfinite(map_tot)]),
                np.min(sumarr[np.isfinite(sumarr)])] + \
               [np.min(_part[0][np.isfinite(_part[0])] \
                if np.any(np.isfinite(_part[0])) else np.inf) \
                for _part in part_map_ext_xname_yname])
    if vmin is not None:
       Vmin = max(vmin, Vmin)
    Vmax = max([np.max(map_tot[np.isfinite(map_tot)]),
                np.max(sumarr[np.isfinite(sumarr)])] + \
               [np.max(_part[0][np.isfinite(_part[0])] \
                if np.any(np.isfinite(_part[0])) else -np.inf) \
                for _part in part_map_ext_xname_yname])
    
    diff = sumarr - map_tot
    if diffmax is not None:
        maxdiff = diffmax
    else:    
        maxdiff = np.max(np.abs(diff)[np.isfinite(np.abs(diff))])
    maxdiff = max(maxdiff, 1e-7)
       
    
    nparts = len(part_map_ext_xname_yname)
    _ncols = min(max(3, nparts), 5)
    _nrows = (nparts  - 1) // _ncols + 2
    fig, axes = plt.subplots(nrows=2 * _nrows, ncols=_ncols, 
                            sharex=False, sharey=False,
                            figsize=(11., 11.))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.85, 
                        wspace=None, hspace=None)
    topaxes = axes[::2, :].flatten()
    bottomaxes = axes[1::2, :].flatten()
    title = 'test: \t{}\nparts: \t{}'
    fn2 = '\n'.join([part.split('/')[-1] for part in partfiles])
    fig.suptitle(title.format(totfile.split('/')[-1], fn2), fontsize=fontsize)
    
    arrbins = np.arange(Vmin, Vmax + 0.0001 * (Vmax-Vmin) / float(nbins), 
                       (Vmax - Vmin) / float(nbins))
    diffbins = np.arange(-maxdiff, maxdiff + 0.0002 * maxdiff / float(nbins), 
                         2. * maxdiff / float(nbins))
                         
    counter = 0
    for axt, axb, (_map, _ext, _xname, _yname) in \
        zip(topaxes[:nparts], bottomaxes[:nparts], part_map_ext_xname_yname):
        
        axt.tick_params(labelsize=fontsize)
        axt.patch.set_facecolor(cm.get_cmap('viridis')(0.))
        img = axt.imshow(_map.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                         vmin=Vmin, vmax=Vmax, interpolation='nearest',
                         extent=_ext) 
        axt.set_title(partlabels[counter], fontsize=fontsize)
        axt.set_xlabel(_xname, fontsize=fontsize)
        axt.set_ylabel(_yname, fontsize=fontsize)
        cbar = plt.colorbar(img, ax=axt, fraction=0.05, pad=0.05, shrink=0.6)
        cbar.solids.set_edgecolor("face")
        cbar.ax.set_ylabel(clabel, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)    
        
        axb.tick_params(labelsize=fontsize)
        axb.hist(_map[np.isfinite(_map)], bins=arrbins, log=True)    
        axb.set_title(partlabels[counter], fontsize=fontsize)
        axb.set_xlabel(clabel, fontsize=fontsize)
        axb.set_ylabel('pixels', fontsize=fontsize)
        counter += 1
        
    ax1 = topaxes[-3]
    ax2 = topaxes[-2]
    ax3 = topaxes[-1]
    ax4 = bottomaxes[-3]
    ax5 = bottomaxes[-2]
    ax6 = bottomaxes[-1]
    
    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.imshow(sumarr.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest') 
    ax1.set_title('sum', fontsize=fontsize)
    ax1.set_xlabel(xname_tot, fontsize=fontsize)
    ax1.set_ylabel(yname_tot, fontsize=fontsize)
    cbar1 = plt.colorbar(img, ax=ax1, fraction=0.05, pad=0.05, shrink=0.6)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2.tick_params(labelsize=fontsize)
    ax2.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax2.imshow(map_tot.T,origin='lower', cmap=cm.get_cmap('viridis'), 
                     vmin=Vmin, vmax=Vmax, interpolation='nearest') 
    ax2.set_title('check',fontsize=fontsize)
    ax2.set_xlabel(xname_tot, fontsize=fontsize)
    ax2.set_ylabel(yname_tot, fontsize=fontsize)
    cbar2 = plt.colorbar(img, ax=ax2, fraction=0.05, pad=0.05, shrink=0.6)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax3.tick_params(labelsize=fontsize)
    ax3.patch.set_facecolor('black')
    img = ax3.imshow((diff).T,origin='lower', cmap=cm.get_cmap('RdBu'), 
                     vmin = -maxdiff, vmax=maxdiff, interpolation='nearest') 
    ax3.set_title('sum - check', fontsize=fontsize)
    cbar3 = plt.colorbar(img, ax=ax3, fraction=0.05, pad=0.05, shrink=0.6)
    cbar3.solids.set_edgecolor("face")
    cbar3.ax.set_ylabel(r'$\Delta$' + clabel, fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)
    
    ax6.set_title('sum - check')
    ax6.hist(np.ndarray.flatten(diff[np.isfinite(diff)]), log=True, 
             bins=diffbins)
    ax6.set_ylabel('number of pixels', fontsize=fontsize)
    ax6.set_xlabel(r'$\Delta$' + clabel, fontsize=fontsize)

    ax4.tick_params(labelsize=fontsize)
    ax4.hist(sumarr[np.isfinite(sumarr)], bins=arrbins, log=True)    
    ax4.set_title('sum', fontsize=fontsize)
    ax4.set_xlabel(clabel, fontsize=fontsize)
    ax4.set_ylabel('pixels', fontsize=fontsize)
    
    ax5.tick_params(labelsize=fontsize)
    ax5.hist(map_tot[np.isfinite(map_tot)], bins=arrbins, log=True)    
    ax5.set_title('check',fontsize=fontsize)
    ax5.set_xlabel(clabel)
    ax5.set_ylabel('pixels', fontsize=fontsize)
    
    fig.tight_layout()
    plt.savefig(outname, format='pdf')
    
def comparemapsets():
    colo6 = 'coldens_o6_L0012N0188_27_test3.7_PtAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS.hdf5'
    rho_by_o8 = 'Density_T4EOS_emission_o8_SmAb_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection.hdf5'
    rho_by_o8_nosel = 'Density_T4EOS_emission_o8_SmAb_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Density_T4EOS_min-None_max-None_endpartsel.hdf5'
    rho_by_mass = 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection.hdf5'
    emo8 = 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS.hdf5'
    emo8_nosel = 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Density_T4EOS_min-None_max-None_endpartsel.hdf5'
    mass = 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS.hdf5'
    t_by_mass = 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection.hdf5'
    
    comparemaps(ndir_newbranch + colo6, ndir_master + colo6, 
                outname=mdir + 'samecheck_oldversion_coldens_o6.pdf', 
                clabel=label_no6)
    comparemaps(ndir_newbranch + emo8, ndir_master + emo8,
                outname=mdir + 'samecheck_oldversion_emission_o8.pdf', 
                clabel=label_emo8)
    comparemaps(ndir_newbranch + mass, ndir_master + mass,
                outname=mdir + 'samecheck_oldversion_mass.pdf', 
                clabel='$\\log_{10} \\, \\Sigma \\; [\\mathrm{g} \\, \\mathrm{cm}^{-2}]$')
    comparemaps(ndir_newbranch + t_by_mass, ndir_master + t_by_mass, 
                outname=mdir + 'samecheck_oldversion_mass_weighted_temperature.pdf', 
                clabel='$\\log_{10} \\, \\langle\\mathrm{T}\\rangle_{\\mathrm{M}} \\; [\\mathrm{K}]$')
    comparemaps(ndir_newbranch + rho_by_mass, ndir_master + rho_by_mass, 
                outname=mdir + 'samecheck_oldversion_mass_weighted_density.pdf', 
                clabel='$\\log_{10} \\, \\langle\\rho\\rangle_{\\mathrm{M}} \\; [\\mathrm{g} \\, \\mathrm{cm}^{-3}]$')
    comparemaps(ndir_newbranch + rho_by_o8, ndir_master + rho_by_o8, 
                outname=mdir + 'samecheck_oldversion_emission_o8_weighted_density.pdf', 
                clabel='$\\log_{10} \\, \\langle\\rho\\rangle_{\\mathrm{L}(\\mathrm{O\\,VIII})} \\; [\\mathrm{g} \\, \\mathrm{cm}^{-3}]$')
    comparemaps(ndir_newbranch + rho_by_o8, ndir_newbranch + rho_by_o8_nosel, 
                outname=mdir + 'samecheck_emptysel_emission_o8_weighted_density.pdf', 
                clabel='$\\log_{10} \\, \\langle\\rho\\rangle_{\\mathrm{L}(\\mathrm{O\\,VIII})} \\; [\\mathrm{g} \\, \\mathrm{cm}^{-3}]$')
    comparemaps(ndir_newbranch + emo8, ndir_newbranch + emo8_nosel, 
                outname=mdir + 'samecheck_emptysel_emission_o8.pdf', 
                clabel=label_emo8)
                
    partfiles_colo6 = [ndir_newbranch + 'coldens_o6_L0012N0188_27_test3.7_PtAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Niondens_o6_PtAb_T4EOS_min-None_max-1e-11_endpartsel.hdf5',
                       ndir_newbranch + 'coldens_o6_L0012N0188_27_test3.7_PtAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Niondens_o6_PtAb_T4EOS_min-1e-11_max-None_endpartsel.hdf5']
    labels_colo6 = ['$n(\\mathrm{O\\,VI}) <  1e-11 \\; \mathrm{cm}^{-3}$',
                    '$n(\\mathrm{O\\,VI}) >  1e-11 \\; \mathrm{cm}^{-3}$']
    
    comparepartmaps(partfiles_colo6, ndir_newbranch + colo6, labels_colo6,
                    partweights=None,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_coldens_o6.pdf',
                    fontsize=12, clabel=label_no6)
                    
    partfiles_emo8_o8 = [ndir_newbranch + 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Lumdens_o8_PtAb_T4EOS_min-None_max-1e-37_endpartsel.hdf5',
                         ndir_newbranch + 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Lumdens_o8_PtAb_T4EOS_min-1e-37_max-None_endpartsel.hdf5']
    labels_emo8_o8 = ['$L(\\mathrm{O\\,VIII})/\\mathrm{V} <  1e-37 \\; \\mathrm{erg}\\,\\mathrm{s}^{-1}\\mathrm{cm}^{-3}$',
                      '$L(\\mathrm{O\\,VIII})/\\mathrm{V} >  1e-37 \\; \\mathrm{erg}\\,\\mathrm{s}^{-1}\\mathrm{cm}^{-3}$']
    
    comparepartmaps(partfiles_emo8_o8, ndir_newbranch + emo8, labels_emo8_o8,
                    partweights=None,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_emission_o8_splitby_lumdens_o8.pdf',
                    fontsize=12, clabel=label_emo8)  
                    
    partfiles_emo8_rho =  [ndir_newbranch + 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                           ndir_newbranch + 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Density_T4EOS_min-1e-29_max-1e-28_endpartsel.hdf5',
                           ndir_newbranch + 'emission_o8_L0012N0188_27_test3.7_SmAb_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Density_T4EOS_min-1e-28_max-None_endpartsel.hdf5',
                           ]
    labels_emo8_rho = ['$\\rho  < 1e-29 \\mathrm{g}\\,\\mathrm{cm}^{-3}$',
                       '$1e-29 < \\rho \\; [\\mathrm{g}\\,\\mathrm{cm}^{-3}]$ < 1e-28',
                       '$\\rho  > 1e-28 \\mathrm{g}\\,\\mathrm{cm}^{-3}$'
                       ]              
    comparepartmaps(partfiles_emo8_rho, ndir_newbranch + emo8, labels_emo8_rho,
                    partweights=None,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_emission_o8_splitby_density.pdf',
                    fontsize=12, clabel=label_emo8)  
    
    partfiles_rho_by_emo8 = [ndir_newbranch + 'Density_T4EOS_emission_o8_SmAb_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                             ndir_newbranch + 'Density_T4EOS_emission_o8_SmAb_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Density_T4EOS_min-1e-29_max-1e-28_endpartsel.hdf5',
                             ndir_newbranch + 'Density_T4EOS_emission_o8_SmAb_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Density_T4EOS_min-1e-28_max-None_endpartsel.hdf5',
                             ]
    
    comparepartmaps(partfiles_rho_by_emo8, ndir_newbranch + rho_by_o8, labels_emo8_rho,
                    partweights=partfiles_emo8_rho,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_emission_o8_weighted_density_splitby_density.pdf',
                    fontsize=12, clabel=label_rho_by_o8)                 
    
    partfiles_mass = [ndir_newbranch + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-100000.0_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                      ndir_newbranch + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Temperature_T4EOS_min-100000.0_max-None_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                      ndir_newbranch + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Temperature_T4EOS_min-None_max-100000.0_Density_T4EOS_min-1e-29_max-None_endpartsel.hdf5',
                      ndir_newbranch + 'Mass_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_T4EOS_partsel_Temperature_T4EOS_min-100000.0_max-None_Density_T4EOS_min-1e-29_max-None_endpartsel.hdf5',
                      ]

    labels_mass = ['$\\mathrm{T} < 1e5 \\;\\mathrm{K}, \\mathrm{rho} < 1e-29 \\;\\mathrm{g}\\,\\mathrm{cm}^{-3}$',
                   '$\\mathrm{T} > 1e5 \\;\\mathrm{K}, \\mathrm{rho} < 1e-29 \\;\\mathrm{g}\\,\\mathrm{cm}^{-3}$',
                   '$\\mathrm{T} < 1e5 \\;\\mathrm{K}, \\mathrm{rho} > 1e-29 \\;\\mathrm{g}\\,\\mathrm{cm}^{-3}$',
                   '$\\mathrm{T} > 1e5 \\;\\mathrm{K}, \\mathrm{rho} > 1e-29 \\;\\mathrm{g}\\,\\mathrm{cm}^{-3}$'
                   ]
    
    comparepartmaps(partfiles_mass, ndir_newbranch + mass, labels_mass,
                    partweights=None,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_mass_splitby_density_temperature.pdf',
                    fontsize=12, clabel=label_mass)  
                    
    partfiles_t_by_mass = [ndir_newbranch + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-None_max-100000.0_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                           ndir_newbranch + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-100000.0_max-None_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                           ndir_newbranch + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-None_max-100000.0_Density_T4EOS_min-1e-29_max-None_endpartsel.hdf5',
                           ndir_newbranch + 'Temperature_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-100000.0_max-None_Density_T4EOS_min-1e-29_max-None_endpartsel.hdf5',
                           ]
        
    comparepartmaps(partfiles_t_by_mass, ndir_newbranch + t_by_mass, labels_mass,
                    partweights=partfiles_mass,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_mass_weighted_temperature_splitby_density_temperature.pdf',
                    fontsize=12, clabel=label_t_by_mass)
                    
    partfiles_rho_by_mass = [ndir_newbranch + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-None_max-100000.0_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                             ndir_newbranch + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-100000.0_max-None_Density_T4EOS_min-None_max-1e-29_endpartsel.hdf5',
                             ndir_newbranch + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-None_max-100000.0_Density_T4EOS_min-1e-29_max-None_endpartsel.hdf5',
                             ndir_newbranch + 'Density_T4EOS_Mass_T4EOS_L0012N0188_27_test3.7_C2Sm_400pix_6.25slice_zcen3.125_x3.125-pm6.25_y3.125-pm6.25_z-projection_partsel_Temperature_T4EOS_min-100000.0_max-None_Density_T4EOS_min-1e-29_max-None_endpartsel.hdf5',
                             ]
        
    comparepartmaps(partfiles_rho_by_mass, ndir_newbranch + rho_by_mass, labels_mass,
                    partweights=partfiles_mass,
                    vmin=None, diffmax=None, 
                    outname=mdir + 'sumcheck_mass_weighted_density_splitby_density_temperature.pdf',
                    fontsize=12, clabel=label_rho_by_mass)  