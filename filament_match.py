#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:35:03 2020

matches 2D map sets to filament maps in Toni's compressed 
(filament coords only) hdf5 format
"""

import numpy as np
import h5py
import os, sys
import matplotlib.pyplot as plt

import make_maps_v3_master as m3
import makecddfs as mc

# to supress print statements from make_map when getting names
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

filmapdir = '/net/luttero/data2/filament_maps_toni/'
defaultmapname = 'filament_map_v1.hdf5'


def getfilmapdata(mapname=defaultmapname):
    hf = h5py.File(filmapdir + mapname, 'r')
    hat = {key: item.decode() if isinstance(item, bytes) else item for (key, item) in hf['Header'].attrs.items()}
    # all stored as strings -> convert appropraite values to float for checks
    hat['BoxSize'] = float(hat['BoxSize'])
    hat['Luminosity_cut'] = float(hat['Luminosity_cut'])
    hat['Redshift'] = float(hat['Redshift'])
    return hat
    
def getfilmappix(filmapdata, mapname=defaultmapname, npix=256, check=False):
    hf = h5py.File(filmapdir + mapname, 'r')
    coords = np.array(hf['Coordinates/Coordinates']) # large
    size = filmapdata['BoxSize'] # assume same units
    
    # convert coords to pixel indices
    # coordinates are centres, so int() should be a reliable round-off method
    pixsize = size / npix
    pixcoords = (coords / pixsize).astype(np.int)
    
    if check:
        backcoords = (pixcoords.astype(np.float) + 0.5) * pixsize
        if not np.all(np.isclose(backcoords, coords, rtol=1e-3)):
            raise RuntimeError('Back calculation of pixels to coordinates failed; check if the number of pixels is correct\n' + \
                               'Used: {npix}**3 pixels for filament map {mapname}, box size {size} cMpc')
            
    return pixcoords
    
def getcolmapnames(mapset=None):
    '''
    args, kwargs for make_map. nameonly is ignored
    '''
    
    if mapset is None:
        mapset = 'o7_L100_snap28_filres'
    if mapset == 'o7_L100_snap28_filres':
        numslices = 256
        simnum = 'L0100N1504'
        snapnum = 28
        centre = [50., 50., 50.]
        L_x = 100.
        L_y = 100.
        L_z = 100.
        npix_x = 4096
        npix_y = 4096
        ptypeW = 'coldens'
        kwargs = {'ionW': 'o7',\
                  'abundsW': 'Pt',\
                  'ptypeQ': None,\
                  'excludeSFRW': 'T4',\
                  'parttype': '0',\
                  'var': 'REFERENCE',\
                  'axis': 'z',\
                  'log': True,\
                  'simulation': 'eagle',\
                  'numslices': None,\
                  'hdf5': True,\
                  }   
    elif mapset == 'o8_L100_snap28_filres':
        numslices = 256
        simnum = 'L0100N1504'
        snapnum = 28
        centre = [50., 50., 50.]
        L_x = 100.
        L_y = 100.
        L_z = 100.
        npix_x = 4096
        npix_y = 4096
        ptypeW = 'coldens'
        kwargs = {'ionW': 'o8',\
                  'abundsW': 'Pt',\
                  'ptypeQ': None,\
                  'excludeSFRW': 'T4',\
                  'parttype': '0',\
                  'var': 'REFERENCE',\
                  'axis': 'z',\
                  'log': True,\
                  'simulation': 'eagle',\
                  'numslices': None,\
                  'hdf5': True,\
                  }
    else:
        raise ValueError(f'Mapset {mapset} is not implemented')
        
    centres = np.array([centre] * numslices)
    axis = kwargs['axis']
    if axis == 'z':
        L_z = L_z / np.float(numslices) # split into numslices slices along projection axis
        centres[:, 2] = centre[2] - (numslices + 1.) * L_z / 2. + np.arange(1, numslices + 1) * L_z  
    if axis == 'x':
        L_x = L_x / np.float(numslices) # split into numslices slices along projection axis
        centres[:, 0] = centre[0] - (numslices + 1.) * L_z / 2. + np.arange(1, numslices + 1) * L_x 
    if axis == 'y':
        L_y = L_y / np.float(numslices) # split into numslices slices along projection axis
        centres[:, 1] = centre[1] - (numslices + 1.) * L_y / 2. + np.arange(1, numslices + 1) * L_y
        
    argss = [(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, ptypeW) for centre in centres]
    with HiddenPrints(): 
        names = [m3.make_map(*args, nameonly=True, **kwargs)[0] for args in argss]
    
    namex = np.array([os.path.isfile(name) for name in names])
    if not np.all(namex):
        print('Failed to find the following files:\n' + '\n\t'.join(list(np.array(names)[np.logical_not(namex)])))
        raise RuntimeError('Some or all of the needed column density map files do no exist')
    return names

def compatcheck(colmapnames, filmapname=defaultmapname):
    fdata = getfilmapdata(mapname=filmapname)
    zcheck = fdata['Redshift']
    boxcheck = fdata['BoxSize']
    
    # with all the file opening, just stop checking at the first mismatch
    for name in colmapnames:
        with h5py.File(name, 'r') as hf:
            _z = hf['Header/inputpars/cosmopars'].attrs['z']
            _bs = hf['Header/inputpars/cosmopars'].attrs['boxsize'] / hf['Header/inputpars/cosmopars'].attrs['h']
            if not (np.isclose(zcheck, _z) and np.isclose(boxcheck, _bs)):
                print(f'Mismatch for:\ncolmap: {name}\nfilmap: {filmapname}\n z, box = {zcheck}, {boxcheck} (filaments) vs. {_z}, {_bs} (col. dens.)')
                return False
    return True

def getmatchedhist(filmapname=defaultmapname, colmapset=None, npix_filmap=256,\
                   bins=None, show_filmaps=False):
    '''
    assumes slice 0 starts at coordinate 0 along the los axis
    '''
    if bins == None:
        bins = np.array([-np.inf] + list(np.arange(-28., 25.01, 0.05)) + [np.inf])
    
    colmapnames = getcolmapnames(mapset=colmapset)
    filmapdata = getfilmapdata(mapname=filmapname)
    check = compatcheck(colmapnames, filmapname=filmapname)
    if not check:
        raise RuntimeError('Filament map and column denisity map redshift and/or box size do not match')
    
    # get 3D mask at filament map resolution
    filpix = getfilmappix(filmapdata, mapname=filmapname, npix=npix_filmap, check=True)
    mask_all = np.zeros((npix_filmap,) * 3, dtype=bool)
    mask_all[tuple(filpix.T)] = True
    
    # get grid size data and los axis
    _nameex = colmapnames[0]
    with h5py.File(_nameex) as _f:
        shape = _f['map'].shape
        saxis = _f['Header/inputpars'].attrs['axis'].decode()
        if saxis == 'x':
            axis = 0
        elif saxis == 'y':
            axis = 1
        elif saxis == 'z':
            axis = 2
        else:
            raise RuntimeError('Column density map did not have projection axis x, y, or z, but {saxis}')
    if shape[0] % npix_filmap != 0 or shape[1] % npix_filmap != 0:
        raise NotImplementedError('As written, getmatchedhist only deals with column density maps with an integer mutliple of the filament map pixels')
    
    # loop over column density maps and get histograms
    outhist_in = None
    outhist_all = None
    
    for name in colmapnames:
        with h5py.File(name, 'r') as _f:
            # determine the los position of the slice and extract the corresponding 2d mask
            if bool(_f['Header/inputpars'].attrs['LsinMpc']):
                conv = 1.
            else:
                conv = 1. / _f['Header/inputpars/cosmopars'].attrs['h']
            loscen = np.array(_f['Header/inputpars'].attrs['centre'])[axis] * conv
            #size = _f['Header/inputpars/cosmopars'].attrs['boxsize'] / _f['Header/inputpars/cosmopars'].attrs['h']
            slicesize =  _f['Header/inputpars'].attrs[f'L_{saxis}'] * conv
            cosmopars = {key: val for key, val in _f['Header/inputpars/cosmopars'].attrs.items()}
            lospos = int(loscen / slicesize)
            sel = [slice(None, None, None)] * 3
            sel[axis] = lospos
            slice_mask_lo = mask_all[tuple(sel)]
        
            
            # expand the mask to the col. map resolution (tested)
            slice_mask = np.zeros((npix_filmap, shape[0] // npix_filmap, npix_filmap, shape[1] // npix_filmap), dtype=bool)
            sel_lo = np.where(slice_mask_lo)
            slice_mask[sel_lo[0], :, sel_lo[1], :] = True
            slice_mask = slice_mask.reshape(shape)
            
            if show_filmaps and lospos % 20 == 0: # I might want to check a few of these, but not 256
                plt.imshow(slice_mask.T, origin='lower', interpolation='nearest', cmap='gist_gray')
                plt.show()
            
            colmap = np.array(_f['map'])
            
            if outhist_all is None:
                outhist_all, _bins = np.histogram(colmap.flatten(), bins=bins)
            else:
                _hist, _bins = np.histogram(colmap.flatten(), bins=bins)
                outhist_all += _hist
            if not np.all(_bins == bins):
                raise RuntimeError(f'In/out bins mismatch: {bins}, {_bins}')
            if outhist_in is None:
                outhist_in, _bins = np.histogram((colmap[slice_mask]).flatten(), bins=bins)
            else:
                _hist, _bins = np.histogram((colmap[slice_mask]), bins=bins)
                outhist_in += _hist
            if not np.all(_bins == bins):
                raise RuntimeError(f'In/out bins mismatch: {bins}, {_bins}')
            
    out = {'outhist_in': outhist_in,\
           'outhist_all': outhist_all,\
           'bins': bins,\
           'colmapnames': colmapnames,\
           'filmapname': filmapname,\
           'filmapdata': filmapdata,\
           'axname': saxis,\
           'cosmopars': cosmopars,\
           }
    return out
            

def pipeline_getsavehist(filmapname=defaultmapname, colmapset=None, npix_filmap=256,\
                   bins=None, show_filmaps=False):
    # run the histogram pipeline
    data = getmatchedhist(filmapname=filmapname, colmapset=colmapset, npix_filmap=256,\
                   bins=bins, show_filmaps=show_filmaps)
    # autoname
    axname = data['axname']
    axmatch = f'{axname}cen'
    namepart_colmap = data['colmapnames'][0]
    namepart_colmap = namepart_colmap.split('/')[-1] # remove path
    namepart_colmap = '.'.join(namepart_colmap.split('.')[:-1]) # remove file extension
    parts = namepart_colmap.split('_')
    parts = [axmatch + '-all' if axmatch in part else part for part in parts] # [x-z]cen#### -> [x-z]cen-all
    namepart_colmap = '_'.join(parts)
    
    namepart_filmap = '.'.join(filmapname.split('.')[:-1]) # remove file extension
    
    savename = filmapdir + f'histogram_filaments_{namepart_colmap}_{namepart_filmap}.hdf5'
    
    with h5py.File(savename, 'w') as fo:
        hed = fo.create_group('Header')
        hed.attrs.create('info', np.string_('Column density histogram of the maps in colmapset (histogram_all) and the parts of those maps in the filament map (histogram_fil)'))
        hed.attrs.create('filament_map', np.string_(filmapdir + data['filmapname']))
        _fd = data['filmapdata']
        for key in _fd:
            if isinstance(_fd[key], str):
                hed.attrs.create(key, np.string_(_fd[key]))
            else:
                hed.attrs.create(key, _fd[key])
        hed.attrs.create('numpix_filament_map', npix_filmap)
        csm = hed.create_group('cosmopars')
        for key in data['cosmopars']:
            csm.attrs.create(key, data['cosmopars'][key])
        
        cd = fo.create_dataset('colmapset', data=np.array(data['colmapnames'], dtype=np.string_))
        cd.attrs.create('info', np.string_('column density maps used for these histograms'))
        
        fo.create_dataset('edges', data=data['bins'])
        fo.create_dataset('histogram_all', data=data['outhist_all'])
        fo.create_dataset('histogram_fil', data=data['outhist_in'])
        

def plotcddf_filpart(histname='o7_L100_snap28_filres'):
    if histname == 'o7_L100_snap28_filres':
        fn = 'histogram_filaments_coldens_o7_L0100N1504_28_test3.4_PtAb_C2Sm_4096pix_0.390625slice_zcen-all_z-projection_T4EOS_filament_map_v1.hdf5'
        xlabel = r'$\log_{10} \, \mathrm{N}(\mathrm{O\,VII}) \; [\mathrm{cm}^{-2}]$'
    ylabel = r'$\log_{10} \; \partial^2 n \,/\, \partial \log_{10} \mathrm{N} \, \partial X$'
    # retrieve filament map data
    with h5py.File(filmapdir + fn, 'r') as fi:
        histogram_all = np.array(fi['histogram_all'])
        histogram_fil = np.array(fi['histogram_fil'])
        edges = np.array(fi['edges'])
        
        input_files = np.array(fi['colmapset'])
        input_ex = input_files[0].decode()
        with h5py.File(input_ex, 'r') as fm:
            cosmopars = {key: val for key, val in fm['Header/inputpars/cosmopars'].attrs.items()}
            n2pix_map = fm['Header/inputpars'].attrs['npix_x'] * fm['Header/inputpars'].attrs['npix_y']
            axname = fm['Header/inputpars'].attrs['axis'].decode()
            slice_length = fm['Header/inputpars'].attrs[f'L_{axname}']
            if not bool(fm['Header/inputpars'].attrs[f'LsinMpc']):
                slice_length /= cosmopars['h']
            boxsize = cosmopars['boxsize'] / cosmopars['h']
        numcounts_target = int(boxsize / slice_length + 0.5) * n2pix_map
        if edges[0] == -np.inf:
            if numcounts_target != np.sum(histogram_all):
                print('The number of histograms does not match expectations from pixel size and slice thicknesss')
        else:
            print(f'The total histogram contains {np.sum(histogram_all) / numcounts_target} of the input pixels')
        print(f'The filaments contain {np.sum(histogram_fil) / np.sum(histogram_all)} of the counted pixels')
        dXtot = mc.getdX(cosmopars['z'], boxsize, cosmopars=cosmopars) * n2pix_map
    
    dlogN = np.diff(edges)
    if edges[0] == -np.inf:
        edges[0] = 2. * edges[1] - edges[2]
    if edges[-1] == np.inf:
        edges[-1] = 2 * edges[-2] - edges[-3]
    cens = edges[:-1] + 0.5 * dlogN
    
    # get 6.25 cMpc cddfs for comparison
    alsoplot = {}
    if histname == 'o7_L100_snap28_filres':
        defcddfg = 'cddf_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
        defcddffile = '/net/luttero/data2/paper1/CDDFs.hdf5'
        
        with h5py.File(defcddffile, 'r') as fi:
            grp = fi[defcddfg]
            _hist = np.array(grp['dn_absorbers_dNdX'])
            _edges = np.array(grp['left_edges'])
            _diff = np.diff(_edges)
            _diff = np.append(_diff, _diff[-1])
            _cens = _edges + 0.5 * _diff
            _dN_over_dlogN = np.log(10.) * 10**(_cens) # like in makecddfs calculations
            _histogram = _hist * _dN_over_dlogN
        alsoplot['all, 6.25 cMpc, high-res'] = {'x': _cens, 'y': _histogram}

    imgname = filmapdir + histname + '.pdf'
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.add_subplot(1, 1, 1)
    
    fontsize = 12
    linewidth = 2
    ax.set_xlim(12., 16.5)
    ax.set_ylim(-3.5, 2.7)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=fontsize - 1.)
    ax.minorticks_on()
    
    ax.plot(cens, np.log10(histogram_all / dXtot / dlogN), linewidth=linewidth, linestyle='solid', label=f'all, {slice_length:.2f} cMpc')
    ax.plot(cens, np.log10(histogram_fil / dXtot / dlogN), linewidth=linewidth, linestyle='dashed', label=f'filaments, {slice_length:.2f} cMpc')
    for key in alsoplot:
        ax.plot(alsoplot[key]['x'], np.log10(alsoplot[key]['y']), linewidth=linewidth, linestyle='dotted', label=key)
  
    ax.legend(fontsize=fontsize, loc='lower left')
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    