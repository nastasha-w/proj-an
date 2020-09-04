#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 17:43:02 2020

@author: wijers


Notes:
------
soxs contains files with info on foregrounds and backgrounds, and instrument
backgrounds (Athena/Hitomi). (I'd have to look up the units though.)
-> from spectrum from hdf5 docs:
  If the file is HDF5, it must have one array dataset, named "spectrum", which 
  is the spectrum in units of photons cm−2 s−1 keV−1, and two scalar datasets, 
  "emin" and "emax", which are the minimum and maximum energies in keV.

check: how is a spectrum in spectrum in ph / (cm2 keV s) converted to a SB in
the spatial model 
energies and coordinates are generated separately: suggests spectrum represents
the total flux, SB = total / angular size
indeed, the total fluxes agree

"""


import soxs
import numpy as np

import make_maps_opts_locs as ol
import eagle_constants_and_units as c

# energy units: keV
emin = 0.1
emax = 12.
res = 0.0002 # X-IFU res. = 2.5 eV. 
# Doesn't nicely resolve the line width, but below instrument resolution, that probably won't matter 
nbins = int((emax - emin) / res + 1.)
ebins = np.linspace(emin, emax, nbins)
emid = 0.5 * (ebins[:-1] + ebins[1:])

# line parameters
line_center = ol.line_eng_ion['o8'] / c.ev_to_erg * 1e-3
line_width = 100. * 1e5 / c.c * line_center # Gaussian b
inv_sigma = 1. / line_width      
norm = 5. # units: "photon/(cm**2*s*keV)"   

# single gaussian line spectrum
gauss = np.exp(-0.5 * (emid - line_center)**2 * inv_sigma**2)
gauss *= norm / np.sum(gauss)
spec = soxs.Spectrum(ebins, gauss)

# galactic absorption
NH = 0.02 # units: 10^22 cm**-2 # check values from Sarah's paper?
spec.apply_foreground_absorption(NH, model="tbabs", redshift=0.0)

# spatial model: uniform SB disk -> this is actuall a rectangular model
# implement a radialfunctionmodel instead (other radial profiles are 
# subclasses implementing this anyway)
ra0 = 30. # degrees
dec0 = 40. # degrees
fov = 2. # arcmin # ~300 pkpc at z=0.1 
disk = soxs.spatial.FillFOVModel(ra0, dec0, fov)

## simput creation: (large values)
exp_time = (500.0, "ks")
area = (3.0, "m**2")
phlist = soxs.PhotonList.from_models('pt_src', spec, disk, exp_time, area)
