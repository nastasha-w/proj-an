#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt


import ion_line_data as ild
import plot_utils as pu

ddir = '/Users/nastasha/phd/data/plotdata_paper2/'

ions = ['o7', 'o8']
fn_o8 = 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'
fn_o7 = 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'

z = 0.1
# in mA
lims = {'o8': [1., 3.],
        'o7': [1., 3.]}
Nlims = {}
for ion in ions:
    Nlims[ion] = {}
    if ion == 'o8':
        width = 112e5
        line = ild.o8doublet
    elif ion == 'o7':
        width = 83e5
        line = ild.o7major
    
    for lim in lims[ion]:
        def solveN(logN):
            ew = ild.linflatdampedcurveofgrowth_inv(10**logN, width, line)
            return np.abs(ew * 1e3 * (1. + z) - lim)
        Nlim = so.minimize(solveN, 15., method='Powell')
        Nlims[ion][lim] = Nlim.x[0]
print(Nlims)

print('Using observed EW lims of {} mA at z={}'.format(lims, z))
for ion, fn in zip(['o7', 'o8'], [fn_o7, fn_o8]):
    with h5py.File(ddir + fn, 'r') as _f:
        Nedges = _f['bins/axis_0'][:]
        counts = _f['nomask']['hist'][:]
    
    if Nedges[-1] == np.inf and counts[-1] == 0.:
        Nedges = Nedges[:-1]
        counts = counts[:-1]
    Ncens = 10**(0.5 * (Nedges[1:] + Nedges[:-1]))
    Nion_rel = counts * Ncens
    Nion_cumul = np.cumsum(Nion_rel)
    Nion_cumul /= Nion_cumul[-1]
    edges_cumul = Nedges[1:]

    _lims = lims[ion]
    _Nlims = Nlims[ion]
    msg = 'fraction of {ion} above {Nlim} ({lim} mA): {frac}'
    for lim in _lims:
        Nlim = _Nlims[lim]
        fracabove = 1. - pu.linterpsolve(edges_cumul, Nion_cumul, Nlim)
        print(msg.format(ion=ion, Nlim=Nlim, lim=lim, frac=fracabove))
    
    
