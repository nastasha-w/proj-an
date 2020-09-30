#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:19:56 2020

@author: wijers

calculate minimum surface brightnesses for different instruments
"""

import numpy as np
#import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.special import erf
# sherpa: for reading in and using rmf data here. Requires python >=3.5
from sherpa.astro.data import DataRMF

import eagle_constants_and_units as c 
import make_maps_opts_locs as ol

import matplotlib.pyplot as plt

arcmin2 = 1. * (np.pi / (180. * 60.))**2 # 1 arcmin**2 in steradian

ddir_xifu = ol.dir_instrumentfiles + 'athena_x-ifu/cost-constrained_2020-09-28/'
filename_resp_xifu = 'responses/XIFU_CC_BASELINECONF_2018_10_10'
filename_bkg_xifu  = 'backgrounds/TotalBKG1arcmin2.pha'

def nsigma(sb, bkg, solidangle, aeff, deltat):
    '''
    detection significance (based on signal/noise), assuming no systematic 
    errors 
    
    parameters:
    -----------
    sb:         surface brightness (photons / cm2 / s / sr)
    bkg:        (equivalent) surface brightness of backgrounds 
                (astro/local/instrumental; photons / cm2 / s / sr)
    solidangle: extraction area for the observation (sr)
    aeff:       effective area (cm2)
    deltat:     observing time (s)
    '''
    return sb * (solidangle * aeff * deltat / (sb + bkg))**0.5

def minsb(nsigma, bkg_rate, counts_norm1, deltat_times_solidangle):
    '''
    minimum surface brightness for detection
    
    parameters:
    -----------
    nsigma:       detection significance required (sigma) 
    bkg_rate:     background rate (counts / s / sr)
                  (astro/local/instrumental)
    counts_norm1: count rate (counts / s / sr) for an input rate of 
                  1 photon / s / cm**2 / sr           
    solidangle:   extraction area for the observation (sr)
    deltat:       observing time (s)
    
    returns:
    --------
    minimum surface brightness (photons / cm2 / s / sr)
    '''
    
    deltat_times_solidangle *= arcmin2
    sb = nsigma**2 / (2. * deltat_times_solidangle * counts_norm1)
    sb *= (1. + (1. + bkg_rate * deltat_times_solidangle / nsigma**2)**0.5)
    return sb

class Responses:
    '''
    given the .arf and .rmf files, stores the response file properties needed
    to estimate the detection limits for an instrument
    '''
    
    def __init__(self, arf_fn, rmf_fn=None):
        '''
        parameters:
        ------
        arf_fn:  the name of the .arf file ('.arf' is appended if not included)
        rmf_fn:  the name of the .rmf file (default: arf file name with the\
                  extension replaced)
        
        '''
        
        self.arf_fn = arf_fn
        if self.arf_fn[-4:] != '.arf':
            self.arf_fn = self.arf_fn + '.arf'
        if rmf_fn is None:
            self.rmf_fn = self.arf_fn[:-4] + '.rmf'
        else:
            self.rmf_fn = rmf_fn
        
        with fits.open(self.arf_fn) as _hdu:
            self.E_lo_arf = _hdu[1].data['ENERG_LO'] # keV
            self.E_hi_arf = _hdu[1].data['ENERG_HI'] # keV
            self.aeff = _hdu[1].data['SPECRESP'] # cm**2
        self.E_cen_arf = 0.5 * (self.E_lo_arf + self.E_hi_arf)
        
        self.get_Aeff = interp1d(self.E_cen_arf, self.aeff, kind='linear',\
                                 copy=True, fill_value=0.)
        
        self.rmf = self.get_rmf(self.rmf_fn)
        self.check_compat()
        
        
    # from a quick test, it seems to conserve normalization (the arf handles 
    # that part), and seems to indeed have a ~2.5 eV FWHM at lower energies 
    def get_rmf(self, filename):
        '''
        get a sherpa.astro.data.DataRMF object from a fits .rmf file 
        '''
        with fits.open(filename) as _hdu:
            self.channel_rmf = _hdu[1].data['CHANNEL'] 
            self.E_min_rmf = _hdu[1].data['E_min']  #KeV
            self.E_max_rmf = _hdu[1].data['E_max']
            
            self.E_lo_rmf = _hdu[2].data['ENERG_LO'] # same as E_min in checked file
            self.E_hi_rmf = _hdu[2].data['ENERG_HI'] # same as E_max in checked file
            _n_grp =  _hdu[2].data['N_GRP']
            _f_chan = _hdu[2].data['F_CHAN']
            _n_chan = _hdu[2].data['N_CHAN']
            _matrix = _hdu[2].data['MATRIX']
            _header_dct = {key: val for key, val in _hdu[2].header.items()}
            # name, detchans, energ_lo, energ_hi, n_grp, f_chan, n_chan, matrix,\
            # offset=1, e_min=None, e_max=None, header=None, ethresh=None
            _rmf_container = DataRMF(filename, len(self.channel_rmf),\
                                     self.E_lo_rmf,\
                                     self.E_hi_rmf,\
                                     _n_grp, _f_chan, _n_chan,\
                                     _matrix.flatten(),\
                                     e_min=self.E_min_rmf,\
                                     e_max=self.E_max_rmf,\
                                     header=_header_dct)
        return _rmf_container
    
    def check_compat(self):
        '''
        check whether the rmf internal and rmf/arf energy ranges match up as 
        expected. The sherpa rmf container may handle different rmf ranges, 
        but arf/rmf channels definitely should match for the way they are 
        combined here.
        '''
        if not (np.all(self.E_min_rmf == self.E_lo_arf) and\
                np.all(self.E_max_rmf == self.E_hi_arf) and\
                np.all(self.E_min_rmf == self.E_lo_rmf) and\
                np.all(self.E_max_rmf == self.E_hi_rmf)):
            raise RuntimeError('Energy channels in the .arf and .rmf files do not match')

    def get_outspec(self, inspec):
        '''
        parameters:
        -----------
        inspec: input spectrum in units photons / cm**2 / s / sr, in bins 
                matching the energy chanels E_lo_arf and E_hi_arf
                numpy array, float
        
        returns:
        --------
        recorded count rate spectrum, excluding backgrounds, no noise added
        units counts / s / sr, in the spectrum channels
        numpy array, float
        '''
        _out = self.rmf.apply_rmf(inspec * self.aeff)
        return _out


def getdata_xifu():
    '''
    get effective area and backgrounds from the X-IFU
    
    1 arcmin / extentded source backgrounds just scale linearly with Aeff, 
    so scaling with effective area is simple
    
    backgrounds are in counts/s/keV/arcmin2 for the extended source,
    counts/s/keV with a 5 arcsec extraction radius for point sources
    
    internal backgrounds are assumed to be a constant 0.005 cts/ cm2 / s / keV
    
    returns:
    --------
    Responses: Responses object using the X-IFU arf amd rmf 
    get_bkg:   function that takes the detector channel and returns 
               the background (counts / cm**2/ s / sr)
               from scipy.interpolate.interp1d
    '''
    
    resp = Responses(ddir_xifu + filename_resp_xifu)
    
    ## gives the same Aeff curve with emid = 0.5 * (egrid_lo + egrid_hi)
    ## as the .aeff file
    #with fits.open(ddir + filename_resp + '.arf') as hdu:
    #    E_lo_arf = hdu[1].data['ENERG_LO'] # keV
    #    E_hi_arf = hdu[1].data['ENERG_HI'] # keV
    #    area = hdu[1].data['SPECRESP'] # cm**2
    #E_cen_arf = 0.5 * (E_lo_arf + E_hi_arf)
    
    #df = pd.read_csv(ddir + filename_resp + '.aeff', header=None,\
    #                 names=['E_keV', 'Aeff_cm2'], index=0)
    #get_Aeff = interp1d(df.index, df['Aeff_cm2'], kind='linear', copy=True,\
    #                    fill_value=0.)
    
    # documentation: bkg is for 1 arcmin**2, backgrounds scale with 
    # solid angle \propto detector area
    extr_area_file = 1. * (np.pi / (180. * 60.))**2 # 1 arcmin**2 -> steradian 
    with fits.open(ddir_xifu + filename_bkg_xifu) as hdu:
        channel_bkg = hdu[1].data['CHANNEL'] 
        rate_bkg = hdu[1].data['RATE']  # counts / s / channel
    
    get_bkg = interp1d(channel_bkg, rate_bkg / extr_area_file, kind='linear',\
                       copy=True, fill_value=np.NaN)
    
    return resp, get_bkg

def getminSB_grid(E_rest, linewidth_kmps=100., z=0.0,\
                  nsigma=5., area_texp=1e5, instrument='athena-xifu',\
                  extr_range=2.5):
    '''
    calculate the minimum surface brightness (photons / cm**2 / s / sr) of a 
    Gaussian emission line for detection
    
    parameters:
    -----------
    E_rest:         rest-frame energy (keV); float or 1D-array of floats
    linewidth_kmps: width of the gaussian line (b parameter); km/s
    z:              redshift of the lines (float)
    nsigma:         required detection significance (sigma; float)
    area_texp:      solid angle to extract the emission from (arcmin**2) 
                    times the exposure time (s); float
    instrument:     'athena-xifu'
    extr_range:     range around the input energy to extract the counts
                    float: range in eV (will be rounded to whole channels)
                    int:   number of channels
        
    returns:
    --------
    _minSB:         array of minimum SB values (photons / cm**2 / s / sr)
    E_pos:          redshifted energies of the lines
    '''
    
    if instrument == 'athena-xifu':
        resp, get_bkg = getdata_xifu()
    else:
        raise ValueError('{} is not a valid instrument option'.format(instrument))
    
    # setup input spectra
    if not hasattr(E_rest, '__len__'):
        E_rest = np.array([E_rest])
    
    grid_emin = resp.E_lo_arf 
    grid_emax = resp.E_hi_arf
    if np.all(grid_emin[1:] == grid_emax[:-1]):
        grid = np.append(grid_emin, grid_emax[-1])
    else:
        raise RuntimeError('E_lo_arf and E_hi_erf grids do not match')
    # line profile \propto exp(-(Delta v / b)**2)
    # erf to integrate properly over the response spacing for narrow lines
    E_pos = E_rest / (1. + z)
    E_width = E_pos * linewidth_kmps * 1e5 / c.c
    specs_norm1 = 0.5 * (1. + erf((E_pos[:, np.newaxis] - grid[np.newaxis, :])\
                                    / E_width[:, np.newaxis]))
    specs_norm1 = specs_norm1[:, :-1] - specs_norm1[:, 1:]
    
    # get count spectra
    counts_norm1 = np.array([resp.get_outspec(spec) for spec in specs_norm1])
    #channels = resp.channel_rmf
    bkg = get_bkg(resp.channel_rmf)
    E_lo = resp.E_lo_rmf
    E_hi = resp.E_hi_rmf
    E_cen = 0.5 * (E_lo + E_hi)
    
    if isinstance(extr_range, int):
        cenchan = np.argmin(np.abs(E_pos[:, np.newaxis] - E_cen[np.newaxis, :]), axis=1)
        offset = extr_range // 2
        if extr_range % 2 == 1:
            ranges = [slice(cen - offset, cen + offset + 1) for cen in cenchan]
        else:
            ranges = [slice(cen - offset, cen + offset) if\
                      np.abs(E_cen[cen - offset] - tar) <  np.abs(E_cen[cen + offset] - tar) else\
                      slice(cen - offset - 1, cen + offset - 1)\
                      for tar, cen in zip(E_pos, cenchan)]
    else:
        extr_range *= 1e-3 # eV to keV
        mins = np.argmin(np.abs(E_cen[np.newaxis, :]  - E_pos[:, np.newaxis] + extr_range), axis=1)
        maxs = np.argmin(np.abs(E_cen[np.newaxis, :]  - E_pos[:, np.newaxis] - extr_range), axis=1)
        ranges = [slice(_min, _max + 1) for _min, _max in zip(mins, maxs)]
    
    counts_norm1_extr = np.array([np.sum(counts[_slice]) for counts, _slice in zip(counts_norm1, ranges)])
    bkg_extr = np.array([np.sum(bkg[_slice]) for _slice in ranges])

    # extract the min. SB
    area_texp *= arcmin2
    _minsb = minsb(nsigma, bkg_extr, counts_norm1_extr, area_texp)
    
    # check: plot in/out spectra
    for li in range(len(E_rest)):
        plt.plot(E_cen, _minsb[li] * specs_norm1[li] * resp.aeff * area_texp,\
                 label='min. det. input spectrum (using Aeff)')
        plt.plot(E_cen, _minsb[li] * counts_norm1[li], label='min. det count spectrum')
        plt.plot(E_cen[ranges[li]], _minsb[li] * counts_norm1[li][ranges[li]],\
                 linestyle='dotted', label='extracted min. det count spectrum')
        plt.axvline(E_pos, label='line energy (redshift)')
        plt.legend()
        plt.show()
    return _minsb
    