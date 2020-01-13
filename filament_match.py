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
    return names

def compatcheck(colmapnames, filmapname=defaultmapname):
    fdata = getfilmapdata(mapname=filmapname)
    zcheck = fdata['Redshift']
    boxcheck = fdata['BoxSize']
    
    # with all the file opening, just stop checking at the first mismatch
    for name in colmapnames:
        with h5py.File(name, 'r') as hf:
            _z = hf['Header/cosmopars'].attrs('z')
            _bs = hf['Header/cosmopars'].attrs('boxsize') / hf['Header/cosmopars'].attrs('h')
            if not (np.isclose(zcheck, _z) and np.isclose(boxcheck, _bs)):
                print(f'Mismatch for:\ncolmap: {name}\nfilmap: {filmapname}\n z, box = {zcheck}, {boxcheck} (filaments) vs. {_z}, {_bs} (col. dens.)')
                return False
    return True

def getmatchedhist(filmapname=defaultmapname, colmapset=None, npix_filmap=256,\
                   bins=None, show_filmaps=False):
    
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
        saxis = _f['Header'].attrs['axis'].decode()
        if saxis == 'x':
            axis = 0
        elif saxis == 'y':
            axis = 1
        elif saxis == 'z':
            axis = 2
        else:
            raise RuntimeError('Column density map did not have projection axis x, y, or z')
    if shape[0] % npix_filmap != 0 or shape[1] % npix_filmap != 0:
        raise NotImplementedError('As written, getmatchedhist only deals with column density maps with an integer mutliple of the filament map pixels')
    
    # loop over column density maps and get histograms
    outhist_in = None
    outhist_all = None
    
    for name in colmapnames:
        with h5py.File(name, 'r') as _f:
            # determine the los position of the slice and extract the corresponding 2d mask
            if bool(_f['Header'].attrs['LsinMpc']):
                conv = 1.
            else:
                conv = 1. / _f['Header/cosmopars'].attrs['h']
            loscen = np.array(_f['Header'].attrs['centre'])[axis] * conv
            size = _f['Header/cosmopars'].attrs['boxsize'] / _f['Header/cosmopars'].attrs['h']
            lospos = int(loscen / size)
            sel = [slice(None, None, None)] * 3
            sel[axis] = lospos
            slice_mask_lo = mask_all[tuple(sel)]
            
            # expand the mask to the col. map resolution (tested)
            slice_mask = np.zeros((npix_filmap, shape[0] // npix_filmap, npix_filmap, shape[1] // npix_filmap), dtype=bool)
            sel_lo = np.where(slice_mask_lo)
            slice_mask[sel_lo[0], :, sel_lo[1], :] = True
            slice_mask = slice_mask.reshape(shape)
            
            if show_filmaps and loscen < 10: # I might want to check a few of these, but not 256
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
            

    
    
    
    
    