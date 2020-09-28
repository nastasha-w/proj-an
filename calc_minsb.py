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
    '''
    ddir = ol.dir_instrumentfiles + 'athena_x-ifu/cost-constrained_2020-09-28/'
    filename = 'XIFU_CC_BASELINECONF_2018_10_10'
    

    with fits.open(ddir + filename + '.arf') as hdu:
    	egrid_lo = hdu[1].data[“ENERG_LO”]
    	egrid_up = hdu[1].data[“ENERG_UP”]
    	area = hdu[1].data[“SPECRESP”]

    df = pd.readcsv(ddir + filename + '.aeff')
    
    