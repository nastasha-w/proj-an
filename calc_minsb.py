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
# sherpa: for reading in and using rmf data here. Requires python >=3.5
from sherpa.astro.data import DataRMF

import eagle_constants_and_units as c 
import make_maps_opts_locs as ol

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

def minsb(nsigma, bkg, solidangle, aeff, deltat):
    '''
    minimum surface brightness for detection
    
    parameters:
    -----------
    nsigma:     detection significance required (sigma) 
    bkg:        (equivalent) surface brightness of backgrounds 
                (astro/local/instrumental; photons / cm2 / s / sr)
    solidangle: extraction area for the observation (sr)
    aeff:       effective area (cm2)
    deltat:     observing time (s)
    
    returns:
    --------
    minimum surface brightness (photons / cm2 / s / sr)
    '''
    
    _c2 = nsigma**2 / (solidangle * aeff * deltat)
    sbmin = 0.5 * (_c2 + (_c2 + 4 * _c2**0.5 * bkg)**0.5)
    return sbmin

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