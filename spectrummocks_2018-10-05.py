#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 19:24:59 2018

@author: wijers
"""
import numpy as np
import ctypes as ct

import eagle_constants_and_units as cu

# c function used to downsample an array into a smaller array, where the number
# of bins in the second array does not need to divide the number in the first
# bin-based, so assumes the value at index one represents a cell between edges
# at 0 and 1, rather than assuming that this represents a sample of a function
# at edge 0
execfile_downsample_grid = './gridcoarser.so'

## some line parameters
# units = Angstrom (hence 1.e-8 factors in calculations)
lambda_rest = {'o7':        21.60169,\
               'o8':        18.9689,\
               'o8major':   18.9671,\
               'o8minor':   18.9725,\
               'h1':        1156.6987178884301,\
               'c3':        977.0201,\
               'c4major':   1548.2041,\
               'c5major':   40.2678,\
               'c6major':   33.7342,\
               'n6':        28.7875,\
               'n7major':   24.7792,\
               'ne8major':  770.409,\
               'ne9':       13.4471,\
               'o4':        787.711,\
               'o5':        629.730,\
               'o6major':   1031.9261,\
               'si3':       1206.500,\
               'si4major':  1393.76018,\
               'fe17major': 15.0140     }
fosc ={'o7':      0.696,\
       'o8':      0.416,\
       'o8major': 0.277,\
       'o8minor': 0.139,\
       'h1':      0.5644956 } 

# functions

def blazarflux(Eoutkev, Etotcgs=7.5e-12, Eminkev=2., Emaxkev=10., Gammaphot=2):
    '''
    dNphot/dE = A * E^-Gammaphot
    A set by total cgs energy flux (erg/cm2/s) from Emin to Emax
    '''
    kev_to_erg = cu.c.ev_to_erg * 1.e3 # erg/kev
    if Gammaphot == 2:
        A = Etotcgs / np.log(Emaxkev / Eminkev)
    else:
        A = (2 - Gammaphot) * Etotcgs / ((Emaxkev * kev_to_erg)**(2-Gammaphot) - (Eminkev * kev_to_erg)**(2-Gammaphot))
    Nphotcgs = A * (Eoutkev * kev_to_erg)**(-1 * Gammaphot) # photons / cm2 / s / erg
    return Nphotcgs * kev_to_erg # photons/ cm2 / s / keV

def getphotonspec(spectrum, vsize_rf_cmps, z, ion, exposuretime_s, Aeff_cm2, **kwargs):
    eng_ion_kev = cu.c.c * cu.c.planck / (lambda_rest[ion] * 1.e-8 * (1. + z)) / (1.e3 * cu.c.ev_to_erg) #(observed wavelength)
    pixsize_kev = vsize_rf_cmps / cu.c.c * eng_ion_kev
    photons_per_pix = blazarflux(eng_ion_kev, **kwargs) * pixsize_kev * exposuretime_s * Aeff_cm2
    photonspec = spectrum * photons_per_pix
    return photonspec, photons_per_pix

def getnoise(photonspec):
    return np.sqrt(photonspec)

def downsample(array,outpixels,average=True,axis=0):
    '''
    interpolates array to a smaller array of size <outpixels>, along <axis>
    axis may be an iterable of axes
    average: True -> use average of input pixels for the new pixels, False -> use the sum
      in both cases, the portion assigned to each new pixel is old-new overlap/old size
    regular grid spacing is assumed (otherwise, see combine_specwiz_makemaps.py for a more general routine)
    
    calls the c function in gridcoarser.c, using make_maps_opts_locs.py for where to find it 
    '''
    inshape  = np.array(array.shape)
    outshape = list(inshape)
    outshape = np.array(outshape)
    outshape[axis] = outpixels
    outarray = np.zeros(outshape,dtype = np.float32)

    cfile = execfile_downsample_grid

    acfile = ct.CDLL(cfile)
    redfunc = acfile.reducearray

    redfunc.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(np.prod(inshape),)),\
                           np.ctypeslib.ndpointer( dtype=ct.c_int, shape=(len(inshape),) ),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=tuple(outshape)),
                           np.ctypeslib.ndpointer(dtype=ct.c_int, shape=(len(outshape),) ), \
                           ct.POINTER(ct.c_int),\
                           ct.POINTER(ct.c_int),\
                           ct.POINTER(ct.c_int),\
                        ]

    # argument conversion. input array as flattend, 1D array

    res = redfunc((array.astype(np.float32)).flatten(),\
               inshape.astype(np.int32),\
               outarray,\
               outshape.astype(np.int32),\
               ct.byref(ct.c_int(len(inshape))),\
               ct.byref(ct.c_int(0)),\
               ct.byref(ct.c_int(average))\
              )
    #print(outarray.shape)
    return np.reshape(outarray,outshape)

def smoothspectrum(spectrum, vvals_rf_kmps, ion, z, fwhm_ev, pix_per_fwhm=2.):
    eng_ion_ev = cu.c.c * cu.c.planck / (lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg) # redshifted energy -> want observed energy given rf velocity
    pixsize_orig_ev = eng_ion_ev * np.average(np.diff(vvals_rf_kmps)) * 1.e5 / cu.c.c # vvals: km/s -> cm/s
    fwhm_pixunits = fwhm_ev / pixsize_orig_ev
    numsample_gaussian = (int(np.ceil(7. * fwhm_pixunits)) / 2) * 2 + 1 # 7 x hwhm away already has a 16 dex difference in y value 
    if len(spectrum) < numsample_gaussian:
        numsample_gaussian = (len(spectrum)/ 2) * 2 - 1
    gaussian_x = (np.arange(numsample_gaussian) - numsample_gaussian / 2).astype(np.float)
    gaussian_y = np.exp(-1*gaussian_x**2 * 4.*np.log(2.)/fwhm_pixunits**2)
    gaussian_y /= np.sum(gaussian_y)
    gaussian_y = np.append(gaussian_y, np.zeros(len(spectrum) - numsample_gaussian))
    spectrum_smoothed = np.fft.irfft(np.fft.rfft(spectrum) * np.fft.rfft(gaussian_y))
    spectrum_smoothed = np.roll(spectrum_smoothed, -1 * (numsample_gaussian / 2)) # center gets shifted by offset between zero index and peak of the gaussian
    downsample_pix = int(np.round(float(len(spectrum)) * pix_per_fwhm / fwhm_pixunits, 0))
    spectrum_downsampled = downsample(spectrum_smoothed, downsample_pix)
    vsize_pix_rf_kmps = np.average(np.diff(vvals_rf_kmps)) * float(len(spectrum)) / float(len(spectrum_downsampled))
    print('Downsample factor %s'%(float(len(spectrum)) / float(len(spectrum_downsampled))))
    return spectrum_downsampled, vsize_pix_rf_kmps


def getstandardspectra(vvals_rf_kmps, spectrum, ion, z, instrument='Arcus'):
    '''
    Instrument: Arcus or Athena
    ion: o7 or o8
    vvals_rf_kmps: pixel velocity values (rest-frame, km/s, as in specwizard VHubble_KmpS)
    z: redshift
    '''
    # from latest (2018 - 07 - 31) Athena science requirements 
    if instrument == 'Athena':
        Aeff_cm2 = 1.05e4 # there are a few values mentioned; this seems like a reasonable guess (2nd entry for 1 keV; also the one on the X-IFU main website)
        fwhm_ev = 2.5 # at 1 keV max value (website says 2.5 eV though, so I'm not sure now, also in 2018 paper. In earlier papers, 1.5 eV seemed to be a goal, 2.5 the requirement)
        exposuretime_s = 50.0e3
        # parameters used for the weak-line sensitivity limit
        Etotcgs = 1.0e-11
        Eminkev = 2.
        Emaxkev = 10. 
        Gammaphot = 1.8 
    # from fig. 1 table in Smith, Abraham, et al. 
    elif instrument == 'Arcus':
        eng_ion_ev = cu.c.c * cu.c.planck / (lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
        if lambda_rest[ion] * (1. + z) > 21.6:
            fwhm_ev = eng_ion_ev/2500. # spec. res. 2500 at 21.6 - 28 A
        else:
            fwhm_ev = eng_ion_ev/2000. # spec. res. 2000 at 16 - 21.6 A
            
        Aeff_cm2 = 500.
        exposuretime_s = 500.0e3
        Etotcgs = 7.5e-12
        Eminkev = 2.
        Emaxkev = 10. 
        Gammaphot = 2 # kind of a guess
    
    smallerspec, pixsize_rf_kmps = smoothspectrum(spectrum, vvals_rf_kmps, ion, z, fwhm_ev)
    photonspec, photons_per_pix = getphotonspec(smallerspec, pixsize_rf_kmps*1e5, z, ion, exposuretime_s, Aeff_cm2, Etotcgs=Etotcgs, Eminkev=Eminkev, Emaxkev=Emaxkev, Gammaphot=Gammaphot)
    noise = getnoise(photonspec)
    normphotonspec = photonspec / photons_per_pix
    normphotonnoise = noise / photons_per_pix
    return normphotonspec, normphotonnoise, pixsize_rf_kmps