#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:19:56 2020

@author: wijers

calculate minimum surface brightnesses for different instruments
sherpa package, including xspec, requires python >= 3.5 and HEADAS 
environment variable set

getting xspec to work with sherpa just did not work. The regular sherpa
parts seem to though. -> use xspec model table instead

# luttero:
# setenv HEADAS /software/heasoft/current/x86_64-pc-linux-gnu-libc2.17
# # flex package is only installed for me (--user)
# Python 3.7.9, pip3 explicitly
# sherpa setup.cfg, xspec section
#xspec_version = 12.10.1
#xspec_lib_dirs = /software/heasoft/current/x86_64-pc-linux-gnu-libc2.17/lib
#xspec_include_dirs = /software/heasoft/current/x86_64-pc-linux-gnu-libc2.17/include
#xspec_libraries = XSFunctions XSModel XSUtil XS hdsp_6.25
##cfitsio_lib_dirs = None
#cfitsio_libraries = cfitsio
##ccfits_lib_dirs = None
#ccfits_libraries = CCfits_2.5
##wcslib_lib_dirs = None
#wcslib_libraries = wcs-5.19.1
##gfortran_lib_dirs = None
##gfortran_libraries = gfortran
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.special import erf
# sherpa: for reading in and using rmf data here. Requires python >=3.5
#from sherpa.astro import xspec
from sherpa.astro.data import DataRMF


import eagle_constants_and_units as c 
import make_maps_opts_locs as ol
import tol_colors as tc

import matplotlib.pyplot as plt

arcmin2 = 1. * (np.pi / (180. * 60.))**2 # 1 arcmin**2 in steradian

## paths
# save images to:
mdir = '/net/luttero/data2/imgs/paper3/minSB/'

# response files and backgrounds:
ddir_xifu = ol.dir_instrumentfiles + 'athena_x-ifu/cost-constrained_2020-09-28/'
filename_resp_xifu = 'responses/XIFU_CC_BASELINECONF_2018_10_10'
filename_bkg_xifu  = 'backgrounds/TotalBKG1arcmin2.pha'

ddir_galabs_xifu = mdir
filename_galabs_xifu = 'norm_1ph_per_keVcm2s_wabs-0.018.txt'

# .arf files from SOXS; may need updating
full_lynx_mucal_hires_rmf = ol.dir_instrumentfiles + 'lynx/mucal/xrs_mucal_0.3eV.rmf'
full_lynx_mucal_hires_arf = '/net/luttero/data2/soxs_responses/xrs_mucal_3x10_0.3eV.arf'
full_lynx_mucal_main_rmf = ol.dir_instrumentfiles + 'lynx/mucal/xrs_mucal_3.0eV.rmf'
full_lynx_mucal_main_arf = '/net/luttero/data2/soxs_responses/xrs_mucal_3x10_3.0eV.arf'

# multiple background files
basedir_bkg_lynx = ol.dir_instrumentfiles + 'lynx/mucal/'
lynx_mucal_hires_bkg_cxb = 'bg_cxb_lxm.fits'
lynx_mucal_main_bkg_cxb = 'bg_cxb_lxmmain.fits'
lynx_mucal_hires_bkg_gal = 'bg_gal_lxm.fits'
lynx_mucal_main_bkg_gal = 'bg_gal_lxmmain.fits'
lynx_mucal_hires_bkg_part = 'bg_part_lxm.fits'
lynx_mucal_main_bkg_part = 'bg_part_lxmmain.fits'

# simple set for XRISM
ddir_xrism_resolve = ol.dir_instrumentfiles + 'xrism/resolve/specfiles_v002/'
filename_arf_xrism_resolve = 'resolve_flt_spec_noGV_20190611.arf' # for a constant SB over pi*5*5 arcmin area (circle), not 3x3 arcmin2 FOV
filename_bkg_nonxray_xrism_resolve = 'resolve_h5ev_2019a_rslnxb.pha'
filename_rmf_xrism_resolve = 'resolve_h5ev_2019a.rmf'
filename_bkg_xray_xrism_resolve = mdir + 'bkgmodel_astro_aurora_2020-10-20_resolve_h5ev_2019a.fak'
area_resolve_xraybkg = 1. * arcmin2
area_resolve_arf = np.pi * 5.**2 * arcmin2
texp_resolve_xraykg = 1e6

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
    sb *= (1. + (1. + 4. * bkg_rate * deltat_times_solidangle / nsigma**2)**0.5)
    return sb

class Responses:
    '''
    given the .arf and .rmf files, stores the response file properties needed
    to estimate the detection limits for an instrument
    '''
    
    def __init__(self, arf_fn, rmf_fn=None, setmin_E_keV=1e-10):
        '''
        parameters:
        ------
        arf_fn:  the name of the .arf file ('.arf' is appended if not included)
        rmf_fn:  the name of the .rmf file (default: arf file name with the
                 extension replaced)
        setmin_E_keV: the minimum energy to assume if ENERG_LO is zero; 
                 a value of 0. will cause a sherpa error 
        '''
        self.setmin_E_keV = setmin_E_keV
        
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
        #if self.E_lo_arf[0] == 0.:
        #    self.E_lo_arf[0] = self.setmin_E_keV
        
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
            self.channel_rmf = _hdu['EBOUNDS'].data['CHANNEL'] 
            self.E_min_rmf = _hdu['EBOUNDS'].data['E_min']  #KeV
            self.E_max_rmf = _hdu['EBOUNDS'].data['E_max']
            
            # different files use different fits extension names for this
            keyopts = ['MATRIX', 'SPECRESP MATRIX']
            for key in keyopts:
                if key in _hdu:
                    break
            if key not in _hdu:
                raise RuntimeError('Could not find the response matrix extension in file {}'.format(filename))
                
            self.E_lo_rmf = _hdu[key].data['ENERG_LO'] # same as E_min in checked file
            self.E_hi_rmf = _hdu[key].data['ENERG_HI'] # same as E_max in checked file
            #if self.E_lo_rmf[0] == 0.:
            #    self.E_lo_rmf[0] = self.setmin_E_keV
            _n_grp =  _hdu[key].data['N_GRP']
            _f_chan = _hdu[key].data['F_CHAN']
            _n_chan = _hdu[key].data['N_CHAN']
            _matrix = _hdu[key].data['MATRIX']
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
                                     ethresh=self.setmin_E_keV,\
                                     header=_header_dct)
        return _rmf_container
    
    def check_compat(self):
        '''
        check whether the rmf internal and rmf/arf energy ranges match up as 
        expected. The sherpa rmf container may handle different rmf ranges, 
        but arf/rmf channels definitely should match for the way they are 
        combined here.
        '''
        if not (np.allclose(self.E_lo_rmf, self.E_lo_arf) and\
                np.allclose(self.E_hi_rmf,self.E_hi_arf)):
            #np.all(self.E_min_rmf == self.E_lo_rmf) and\
            #np.all(self.E_max_rmf == self.E_hi_rmf)):
            raise RuntimeError('Energy channels in the .arf and .rmf files do not match')

    def get_outspec(self, inspec):
        '''
        Parameters
        ----------
        inspec: input spectrum in units photons / cm**2 / s / sr, in bins 
                matching the energy chanels E_lo_arf and E_hi_arf
                numpy array, float
        
        Returns
        -------
        recorded count rate spectrum, excluding backgrounds, no noise added
        units counts / s / sr, in the spectrum channels
        numpy array, float
        '''
        _out = self.rmf.apply_rmf(inspec * self.aeff)
        return _out
    

class InstrumentModel:
    '''
    class containing instrument repsonses and backgrounds, and functions to 
    calculate minimum observable SBs from there
    '''
    
    def __init__(self, instrument='athena-xifu',\
                 rmf_fn=None, arf_fn=None, bkg_fn=None):
        '''
        set up the response files and background for an instrument. Defaults 
        for the responses and backgrounds are used if not given.
        
        Parameters
        ----------
        instrument:    name of the instrument (sets defaults and info like the 
                       solid angle covered by the background model)
                       string, options: 'athena-xifu'
                                        'lynx-lxm-main' 
                                        'lynx-lxm-uhr' -> Ultra High Resolution
                                        'xrism-resolve'
                                        
        rmf_fn:        .rmf response file name 
                       (string, incl. full path and file extension, or None for
                       instrument default)
        arf_fn:        .arf response file name 
                       (string, incl. full path and file extension, or None for
                       instrument default)
        bkg_fn:        background levels file name 
                       (string, incl. full path and file extension, or None for
                       instrument default)
                       
        instrument is not checked for consistency with the response files and
        background; galactic absorption uses the x-ifu paper model: wabs 
        '''
        self.instrument = instrument
        
        if self.instrument == 'athena-xifu':
            if rmf_fn is None:
                self.rmf_fn = ddir_xifu + filename_resp_xifu + '.rmf'
            else:
                self.rmf_fn = rmf_fn
            if arf_fn is None:
                self.arf_fn = ddir_xifu + filename_resp_xifu + '.arf'
            else:
                self.arf_fn = arf_fn
            self.responses = Responses(arf_fn=self.arf_fn, rmf_fn=self.rmf_fn)
            
            if bkg_fn is None:
                self.bkg_fn = ddir_xifu + filename_bkg_xifu
            self.extr_area_file_sr = 1. * arcmin2 #* (np.pi / (180. * 60.))**2 # 1 arcmin**2 -> steradian 
            
            with fits.open(self.bkg_fn) as hdu:
                self.channel_bkg = hdu[1].data['CHANNEL'] 
                self.rate_bkg = hdu[1].data['RATE']  # counts / s / channel
            
            self.get_bkg = interp1d(self.channel_bkg,\
                                    self.rate_bkg / self.extr_area_file_sr,\
                                    kind='linear', copy=True,\
                                    fill_value=np.NaN)
                
            self.get_galabs = self._getgalabs_xifu()
        
        elif instrument.startswith('lynx'):
            if instrument == 'lynx-lxm-uhr':
                self.rmf_fn = full_lynx_mucal_hires_rmf
                self.arf_fn = full_lynx_mucal_hires_arf
                
                self.bkg_fn1 = basedir_bkg_lynx + lynx_mucal_hires_bkg_cxb
                self.bkg_fn2 = basedir_bkg_lynx + lynx_mucal_hires_bkg_gal
                self.bkg_fn3 = basedir_bkg_lynx + lynx_mucal_hires_bkg_part
                self.extr_area_file_sr = 1. * arcmin2 
                
            elif instrument == 'lynx-lxm-main':
                self.rmf_fn = full_lynx_mucal_main_rmf
                self.arf_fn = full_lynx_mucal_main_arf
                
                self.bkg_fn1 = basedir_bkg_lynx + lynx_mucal_main_bkg_cxb
                self.bkg_fn2 = basedir_bkg_lynx + lynx_mucal_main_bkg_gal
                self.bkg_fn3 = basedir_bkg_lynx + lynx_mucal_main_bkg_part
                self.extr_area_file_sr = 25. * arcmin2 
                
            if rmf_fn is not None:
                self.rmf_fn = rmf_fn
            if arf_fn is not None:
                self.arf_fn = arf_fn
            self.responses = Responses(arf_fn=self.arf_fn, rmf_fn=self.rmf_fn)
            
            if bkg_fn is not None:
                self.bkg_fn1 = bkg_fn.format(comp='cxb')
                self.bkg_fn2 = bkg_fn.format(comp='gal')
                self.bkg_fn3 = bkg_fn.format(comp='part')
            
            _bkg = np.zeros(len(self.responses.channel_rmf), dtype=np.float)
            for fn in [self.bkg_fn1, self.bkg_fn2, self.bkg_fn3]:
                with fits.open(self.bkg_fn1) as hdu:
                    texp_s = hdu[1].header['EXPOSURE']
                    channels = hdu[1].data['PHA']
                    bc = np.bincount(channels,\
                                     minlength=self.responses.channel_rmf[-1] + 1)
                    bc = bc[self.responses.channel_rmf[0]:]
                    _bkg += bc.astype(float) / texp_s 
            
            self.get_bkg = interp1d(self.responses.channel_rmf,\
                                    _bkg / self.extr_area_file_sr,\
                                    kind='linear', copy=True,\
                                    fill_value=np.NaN)
            # placeholder!
            self.get_galabs = self._getgalabs_xifu()
            
        elif self.instrument == 'xrism-resolve':
            if rmf_fn is None:
                self.rmf_fn = ddir_xrism_resolve + filename_rmf_xrism_resolve
            else:
                self.rmf_fn = rmf_fn
            if arf_fn is None:
                self.arf_fn = ddir_xrism_resolve + filename_arf_xrism_resolve
            else:
                self.arf_fn = arf_fn
            self.responses = Responses(arf_fn=self.arf_fn, rmf_fn=self.rmf_fn)
            
            if bkg_fn is None:
                self.bkg_fn1 = ddir_xrism_resolve + filename_bkg_nonxray_xrism_resolve
                self.extr_area_fn1_sr = area_resolve_arf
                
                self.bkg_fn2 = filename_bkg_xray_xrism_resolve
                self.extr_area_fn2_sr = area_resolve_xraybkg
            else:
                raise ValueError('Non-standard background files are not an option for XRISM Resolve given extraction area issues')
            
            _bkg = np.zeros(len(self.responses.channel_rmf), dtype=np.float)
            with fits.open(self.bkg_fn1) as hdu:
                self.channel_bkg = hdu[1].data['CHANNEL'] 
                self.rate_bkg = hdu[1].data['COUNTS']  # counts / channel, no rate data
                texp = hdu[1].header['EXPOSURE'] # in s
                _bkg += self.rate_bkg / self.extr_area_fn1_sr / texp
                
            with fits.open(self.bkg_fn2) as hdu:
                self.channel_bkg = hdu[1].data['CHANNEL'] 
                self.rate_bkg = hdu[1].data['RATE']  # counts / s / channel
                _bkg += self.rate_bkg / self.extr_area_fn2_sr        
            
            self.get_bkg = interp1d(self.responses.channel_rmf,\
                                    _bkg,\
                                    kind='linear', copy=True,\
                                    fill_value=np.NaN)
                
            self.get_galabs = self._getgalabs_xifu()
        
        else:
            raise ValueError('{} is not a valid or implemented instrument'.format(self.instrument))
            #self.rmf_fn = rmf_fn
            #self.arf_fn = arf_fn
            #self.bkg_fn = bkg_fn
            #self.responses = Responses(arf_fn=self.arf_fn, rmf_fn=self.rmf_fn)
        
        self.setup_Egrid()
        
    def setup_Egrid(self):
        if np.all(self.responses.E_lo_arf[1:] == self.responses.E_hi_arf[:-1]):
            self.Egrid = np.append(self.responses.E_lo_arf,\
                                   self.responses.E_hi_arf[-1])

        else:
            raise RuntimeError('E_lo_arf and E_hi_arf grids do not match')
            
    def getminSB_grid(self, E_rest, linewidth_kmps=100., z=0.0,\
                      nsigma=5., area_texp=1e5, extr_range=2.5,\
                      incl_galabs=False):
        '''
        calculate the minimum surface brightness (photons / cm**2 / s / sr) of 
        a Gaussian emission line for detection
        
        Parameters
        ----------
        E_rest:         rest-frame energy (keV); float or 1D-array of floats
        linewidth_kmps: width of the gaussian line (b parameter); km/s
        z:              redshift of the lines (float)
        nsigma:         required detection significance (sigma; float)
        area_texp:      solid angle to extract the emission from (arcmin**2) 
                        times the exposure time (s); float
        extr_range:     range around the input energy to extract the counts
                        float: range in eV (will be rounded to whole channels)
                        int:   number of channels
        incl_galabs:    include the effect of absorption by our Galaxy (bool)
                        (minimum surface brightnesses at the instrument are 
                        adjusted for galactic absorption to estimate the
                        required intrinsic (but redshifted) flux)
            
        Returns
        -------
        _minSB:         array of minimum SB values (photons / cm**2 / s / sr)
        E_pos:          redshifted energies of the lines
        '''
                
        # setup input spectra
        if not hasattr(E_rest, '__len__'):
            E_rest = np.array([E_rest])
        
        # line profile \propto exp(-(Delta v / b)**2)
        # erf to integrate properly over the response spacing for narrow lines
        E_pos = E_rest / (1. + z)
        E_width = E_pos * linewidth_kmps * 1e5 / c.c
        specs_norm1 = 0.5 * (1. + erf((E_pos[:, np.newaxis] -\
                                       self.Egrid[np.newaxis, :])\
                                      / E_width[:, np.newaxis]))
        specs_norm1 = specs_norm1[:, :-1] - specs_norm1[:, 1:]
        if not np.allclose(np.sum(specs_norm1, axis=1), 1.):
            msg = 'Spectra not normalized to 1 over {} -- {} keV (obs)'
            msg = msg.format(self.responses.E_lo_arf[0], self.responses.E_hi_arf[-1])
            raise RuntimeError(msg)
            
        #print(np.sum(specs_norm1, axis=1))
        E_cen = 0.5 * (self.responses.E_lo_arf + self.responses.E_hi_arf)
        if incl_galabs:
            absfrac = self.get_galabs(E_cen)
            specs_norm1 *= absfrac[np.newaxis, :]
        
        # get count spectra
        counts_norm1 = np.array([self.responses.get_outspec(spec)\
                                 for spec in specs_norm1])
        
        #channels = resp.channel_rmf
        bkg = self.get_bkg(self.responses.channel_rmf)
        
        if isinstance(extr_range, int):
            cenchan = np.argmin(np.abs(E_pos[:, np.newaxis] -\
                                       E_cen[np.newaxis, :]),\
                                axis=1)
            offset = extr_range // 2
            if extr_range % 2 == 1:
                ranges = [slice(cen - offset, cen + offset + 1)\
                          for cen in cenchan]
            else:
                ranges = [slice(cen - offset, cen + offset) if\
                          np.abs(E_cen[cen - offset] - tar) < \
                          np.abs(E_cen[cen + offset] - tar) else\
                          slice(cen - offset - 1, cen + offset - 1)\
                          for tar, cen in zip(E_pos, cenchan)]
        else:
            extr_range *= 1e-3 # eV to keV
            mins = np.argmin(np.abs(self.responses.E_lo_rmf[np.newaxis, :]\
                                    - E_pos[:, np.newaxis] + extr_range),\
                             axis=1)
            maxs = np.argmin(np.abs(self.responses.E_hi_rmf[np.newaxis, :]\
                                    - E_pos[:, np.newaxis] - extr_range),\
                             axis=1)
            ranges = [slice(_min, _max) for _min, _max in zip(mins, maxs)]
        
        counts_norm1_extr = np.array([np.sum(counts[_slice])\
                            for counts, _slice in zip(counts_norm1, ranges)])
        bkg_extr = np.array([np.sum(bkg[_slice]) for _slice in ranges])
        #print(counts_norm1_extr)
        #print(bkg_extr)
        # extract the min. SB
        _minsb = minsb(nsigma, bkg_extr, counts_norm1_extr, area_texp)
        
        ## check: plot in/out spectra
        #for li in range(len(E_rest)):
        #    plt.plot(E_cen, _minsb[li] * specs_norm1[li] * resp.aeff * area_texp * arcmin2,\
        #             label='min. det. input spectrum (using Aeff)')
        #    plt.plot(E_cen, _minsb[li] * counts_norm1[li] * area_texp * arcmin2, label='min. det count spectrum')
        #    plt.plot(E_cen, bkg * area_texp * arcmin2, label='background')
        #    plt.plot(E_cen[ranges[li]], _minsb[li] * counts_norm1[li][ranges[li]] * area_texp * arcmin2,\
        #             linestyle='dotted', label='extracted min. det count spectrum')
        #    plt.axvline(E_pos[li], label='line energy (redshift)')
        #    plt.legend()
        #    plt.show()
        return _minsb
        
    def _getgalabs_xifu(self):
        '''
        from the  McCammon et al. (2002) diffuse X-ray background model, using
        wabs (xspec model) for the galactic  absorption, with a hydrogen 
        column of 1.8e20 cm**-2 (0.018 parameter value)

        minimum energy (~0.1 keV) is above X-IFU minimum, but high enough, 
        really
        
        Returns
        -------
        get_galabs:  function that returns the absorber flux fraction at the
                     input energy (keV) (scipy interp1d object)

        '''
        _fn = ddir_galabs_xifu + filename_galabs_xifu
        _kwargs = {'header': None,\
                   'names': ['E_keV', 'DeltaE_keV', 'absfrac'],\
                   'skiprows': 3,\
                   'sep': ' '}
        _df = pd.read_csv(_fn, **_kwargs)
        get_galabs = interp1d(_df['E_keV'], _df['absfrac'],\
                              kind='linear', copy=True,\
                              fill_value=(1e-10, 1.), bounds_error=False)
        return get_galabs
        
        
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
    extr_area_file = 1. * arcmin2 #* (np.pi / (180. * 60.))**2 # 1 arcmin**2 -> steradian 
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
    #print(np.sum(specs_norm1, axis=1))
    
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
        mins = np.argmin(np.abs(E_lo[np.newaxis, :]  - E_pos[:, np.newaxis] + extr_range), axis=1)
        maxs = np.argmin(np.abs(E_hi[np.newaxis, :]  - E_pos[:, np.newaxis] - extr_range), axis=1)
        ranges = [slice(_min, _max + 1) for _min, _max in zip(mins, maxs)]
    
    counts_norm1_extr = np.array([np.sum(counts[_slice]) for counts, _slice in zip(counts_norm1, ranges)])
    bkg_extr = np.array([np.sum(bkg[_slice]) for _slice in ranges])
    #print(counts_norm1_extr)
    #print(bkg_extr)
    # extract the min. SB
    _minsb = minsb(nsigma, bkg_extr, counts_norm1_extr, area_texp)
    
    ## check: plot in/out spectra
    #for li in range(len(E_rest)):
    #    plt.plot(E_cen, _minsb[li] * specs_norm1[li] * resp.aeff * area_texp * arcmin2,\
    #             label='min. det. input spectrum (using Aeff)')
    #    plt.plot(E_cen, _minsb[li] * counts_norm1[li] * area_texp * arcmin2, label='min. det count spectrum')
    #    plt.plot(E_cen, bkg * area_texp * arcmin2, label='background')
    #    plt.plot(E_cen[ranges[li]], _minsb[li] * counts_norm1[li][ranges[li]] * area_texp * arcmin2,\
    #             linestyle='dotted', label='extracted min. det count spectrum')
    #    plt.axvline(E_pos[li], label='line energy (redshift)')
    #    plt.legend()
    #    plt.show()
    return _minsb

#def get_galabs():
#    '''
#    from the  McCammon et al. (2002) diffuse X-ray background model, based on
#    high galactic latitude observations
#    '''
#    nH = 0.018 # units 10^22 cm^-2
#    md = xspec.XSwabs('galabs')
#    md.nH.set(nH)
#    Egrid = np.linespace(0.1, 12.5, 1000)
#    vals = md(Egrid[:-1], Egrid[1:]) 
#    return Egrid, vals

def explorepars_omegat_extr(instrument):
    
    extr_ranges = {'athena-xifu': [4, 7, 10],\
                   'lynx-lxm-main': [3, 5, 7],\
                   'lynx-lxm-uhr': [17, 33, 50],\
                   'xrism-resolve': [8, 10, 12],\
                   }
    # 0.3 - 2 keV
    delta_E_chan = {'lynx-lxm-main': 0.6,\
                    'lynx-lxm-uhr': 0.06,\
                    'athena-xifu': 0.36,\
                    'xrism-resolve': 0.5}
    extr = extr_ranges[instrument]
    omegat = [1e5, 1e6, 1e7]
    label = '{omegat:.0e} am2*s, {extr:.1f} eV'
    title = 'Varying $\\Delta \\Omega \\times \\mathrm{t}_{\\mathrm{exp}}$ and the $\\Delta$E range for line measurment'
    fontsize = 12
    
    kwargs_extr = [{'color': 'red'}, {'color': 'orange'}, {'color': 'green'},\
                   {'color': 'blue'}]
    kwargs_omegat = [{'linestyle': 'dotted'}, {'linestyle': 'dashed'},\
                     {'linestyle': 'solid'}]
    Egrid = np.linspace(0.3, 2.0, 10)
    im = InstrumentModel(instrument=instrument)
    if instrument == 'lynx-lxm-uhr':
        _max = im.responses.E_hi_arf[-1]
        Egrid = np.arange(0.3, _max - 0.05, 0.01)
    
    for kw1, _extr in zip(kwargs_extr, extr):
        for kw2, _omegat in zip(kwargs_omegat, omegat):
            kwargs = kw1.copy()
            kwargs.update(kw2)
            y = im.getminSB_grid(Egrid, area_texp=_omegat, extr_range=_extr,\
                                 linewidth_kmps=100.)
            
            _xtrl = _extr * delta_E_chan[instrument] if isinstance(_extr, int)\
                    else _extr
            plt.plot(Egrid, np.log10(y), label=label.format(omegat=_omegat,\
                                                            extr=_xtrl),\
                     **kwargs)
    plt.title(title, fontsize=fontsize)
    plt.xlabel('line energy (keV)', fontsize=fontsize)
    plt.ylabel('log10 min. SB [photons / s / cm**2 / sr]', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.savefig(mdir + 'minSB_{instr}_varying_omegatexp_spec-extr.pdf'.format(instr=instrument), bbox_inches='tight')        

def plot_minSB():
    names = ['athena-xifu', 'lynx-lxm-main', 'lynx-lxm-uhr']#, 'xrism-resolve']
    labels = {'athena-xifu': 'X-IFU',\
              'lynx-lxm-main': 'LXM-main',\
              'lynx-lxm-uhr': 'LXM-UHR',\
              'xrism-resolve': 'XRISM-R',\
              }
    cset = tc.tol_cset('vibrant')
    colors = {'athena-xifu': cset.blue,\
              'lynx-lxm-main': cset.orange,\
              'lynx-lxm-uhr': cset.red,\
              'xrism-resolve': cset.teal}
    extr_ranges = {'athena-xifu': [1.5, 2.5, 3.5],\
                   'lynx-lxm-main': [2.0, 3.0, 4.0],\
                   'lynx-lxm-uhr': [1.0, 2.0, 3.0],\
                   'xrism-resolve': [4.0, 5.0, 6.0],\
                   }
    alphas = [1., 0.8, 0.4]
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.gca()
    fontsize = 12
    exptimes = [1e5, 1e6, 1e7]
    linestyles = ['solid', 'dashed', 'dotted']
    addl = '{omegat:.0e} am2 s, $\\Delta$E = {deltae:.1f} eV'
    
    for isn in names:
        ins = InstrumentModel(instrument=isn)
        if isn == 'lynx-lxm-uhr':
            Egrid = np.arange(0.1, 0.9, 0.01)
        else:
            Egrid = np.arange(0.1, 3., 0.01)
        
        for et, ls in zip(exptimes, linestyles):
            for erng, alpha in zip(extr_ranges[isn], alphas):
                aeff = ins.getminSB_grid(Egrid, linewidth_kmps=100., z=0.0,\
                              nsigma=5., area_texp=et, extr_range=erng,\
                              incl_galabs=False)
                label = labels[isn] + addl.format(omegat=et, deltae=erng)
                ax.plot(Egrid, aeff, label=label, color=colors[isn],\
                        linestyle=ls, alpha=alpha)
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E [keV]', fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{A}_{\\mathrm{eff}} \\; [\\mathrm{cm}^{2}]$',\
                  fontsize=fontsize)
    ax.legend(fontsize=fontsize-2)
    

def plot_Aeff_galabs():
    names = ['athena-xifu', 'lynx-lxm-main', 'lynx-lxm-uhr', 'xrism-resolve']
    labels = {'athena-xifu': 'X-IFU',\
              'lynx-lxm-main': 'LXM-main',\
              'lynx-lxm-uhr': 'LXM-UHR',\
              'xrism-resolve': 'XRISM-R',\
              }
    cset = tc.tol_cset('vibrant')
    colors = {'athena-xifu': cset.blue,\
              'lynx-lxm-main': cset.orange,\
              'lynx-lxm-uhr': cset.red,\
              'xrism-resolve': cset.teal}
    
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.gca()
    fontsize = 12
    
    for isn in names:
        ins = InstrumentModel(instrument=isn)
        Egrid = 0.5 * (ins.Egrid[:-1] + ins.Egrid[1:]) 
        
        aeff = ins.responses.get_Aeff(Egrid)
        label = labels[isn] 
        ax.plot(Egrid, aeff, label=label, color=colors[isn], linewidth=2)
         
        if isn == 'athena-xifu':
            absfrac = ins.get_galabs(Egrid)
            ax.plot(Egrid, absfrac * 1e4, color='black',\
                    label='wabs * 1e4 $\\mathrm{cm}^{2}$')
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E [keV]', fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{A}_{\\mathrm{eff}} \\; [\\mathrm{cm}^{2}]$',\
                  fontsize=fontsize)
    xlim = ax.get_xlim()
    ax.set_xlim(0.1, xlim[1])
    ax.set_ylim(1., 3e4)
    ax.legend(fontsize=fontsize)

