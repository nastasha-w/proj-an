#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:19:56 2020

@author: wijers

calculate minimum surface brightnesses for different instruments
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d

import eagle_constants_and_units as c 
import make_maps_opts_locs as ol

def nsigma(sb, bkg, solidangle, aeff, deltat):
    '''
    detection significance (based on signal/noise), assuming no systematic 
    errors
    
    input:
    ------
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
    
    input:
    ------
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
    get_Aeff:  function that takes in line energy (observed, keV) and returns 
               effective area (cm**2), from scipy.interpolate.interp1d
    get_bkg:   function that takes in line energy (observed, keV) and returns 
               the background (counts / cm**2/ s / sr)
               from scipy.interpolate.interp1d
    
    '''
    ddir = ol.dir_instrumentfiles + 'athena_x-ifu/cost-constrained_2020-09-28/'
    filename_resp = 'responses/XIFU_CC_BASELINECONF_2018_10_10'
    filename_bkg  = 'backgrounds/TotalBKG1arcmin2.pha'
    
    ## gives the same Aeff curve with emid = 0.5 * (egrid_lo + egrid_hi)
    #with fits.open(ddir + filename + '.arf') as hdu:
    #    egrid_lo = hdu[1].data['ENERG_LO'] # keV
    #    egrid_hi = hdu[1].data['ENERG_HI'] # keV
    #    area = hdu[1].data['SPECRESP'] # cm**2

    df = pd.read_csv(ddir + filename_resp + '.aeff', header=None,\
                     names=['E_keV', 'Aeff_cm2'], index=0)
    get_Aeff = interp1d(df.index, df['Aeff_cm2'], kind='linear', copy=True,\
                        fill_value=0.)

    extr_area_file = 1. * (np.pi / (180. * 60.))**2 # 1 arcmin**2 -> steradian 
    with fits.open(ddir + filename_bkg) as hdu:
        channel = hdu[1].data['CHANNEL'] 
        rate = hdu[1].data['RATE']  # counts / s / channel
    
    # get channel to energy mapping
    with fits.open(ddir + filename_resp + '.rmf') as hdu:
        channel_ref = hdu[1].data['CHANNEL'] 
        E_min = hdu[1].data['E_min']  #KeV
        E_max = hdu[1].data['E_max']
    E_cen = 0.5 * (E_min + E_max)
    
    if not np.all(channel == channel_ref): # would only really catch different numbers of channels
        raise RuntimeError('RMF and background file seem to contain different channels')
    
    get_bkg = interp1d(E_cen, rate / extr_area_file, kind='linear', copy=True,\
                        fill_value=np.NaN)
    
    return get_Aeff, get_bkg