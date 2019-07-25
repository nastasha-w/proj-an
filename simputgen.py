"""
Created on Tue Jan  8 09:42:00 2019

@author: Nastasha
"""

import numpy as np
import astropy as ap
import soxs 
import h5py

import make_maps_opts_locs as ol
import plotspecwizard as ps

# soxs spectrum can be generated from numpy arrays!
# arguments: bin edges (keV), flux (photons/cm^2/s/keV)
# has attributes ebins, emid, and flux, so can do math with that

specfile = ol.sdir + 'sample3/spec.snap_027_z000p101.0.hdf5'
odir = '/net/luttero/data2/xray_mocks/simputfiles/'

# Emin, Emax for X-IFU is 0.2 - 12. keV
def generate_spectrum(specnum=9282, Emin=0.2, Emax=12., spacing_keV=0.00025,\
                      ions=['o7', 'o8', 'fe17', 'ne9'],\
                      specfile=specfile,\
                      n_H22_mw=0.02, n_H22_bl=0.02, z_bl=1.,
                      **blazarparams):
    '''         
    Still bugs somewhere!!             
    Blazar spectrum parameters: normalisation: flux in erg/s/cm^2 between Emin 
                                               and Emax 
                                               Etotcgs=7.5e-12, 
                                               EminkeV=2., 
                                               EmaxkeV=10. 
                                PL index:      Gammaphot=2.
    '''    
    ebins_keV, flux_ph_cm2skeV = ps.get_absorbed_spectrum(specnum, ions,\
                                     inname=specfile,\
                                     blazarparams=blazarparams,\
                                     savedir_img=odir,\
                                     Eminincl_keV=Emin, Emaxincl_keV=Emax, spacing_keV=0.00025)
    # set bin edges (spectrum is flux density; slight changes to centers shouldn't mess things up too badly or create artifacts)
    #ebins_keV = np.zeros(len(emid_keV) + 1) # assuming constant bin sizes at the edges
    #ebins_keV[1:-1] = emid_keV[:-1] + 0.5*np.diff(emid_keV)
    #ebins_keV[0]    = 1.5 * emid_keV[0] - 0.5 * emid_keV[1]
    #ebins_keV[-1]   = 1.5 * emid_keV[-1] - 0.5 * emid_keV[-2]

    spec = soxs.Spectrum(ebins_keV, flux_ph_cm2skeV)
    
    ## foregrond and blazar local absorption (10^22 cm^-2)
    spec.apply_foreground_absorption(n_H22_mw, model="tbabs", redshift=0.0)
    spec.apply_foreground_absorption(n_H22_bl, model="tbabs", redshift=z_bl)
    
    return spec
    
#### write the output (point source)
## Create the spatial model
def save_photon_list(spec, specnum,\
                     ra=0., dec=0.,\
                     exp_time=(500.0, "ks"), area=(3.0, "m**2")):
    pt_src = soxs.PointSourceModel(ra, dec)

    ## Create the photon list
    phlist = soxs.PhotonList.from_models('sample3_spectrum%i'%specnum, spec, pt_src, exp_time, area)
    simput_prefix = "basic_phlist_sample3_spectrum%i"%specnum
    cat = soxs.simput.SimputCatalog(simput_prefix, phlist)
    cat.write_catalog()
    #try:
    #    phlist.write_photon_list(simput_prefix, append=False)
    #except:
    #    print('Saving photon list failed')
    return cat
    
def save_simput_spectrum(spec, ra=0., dec=0.,\
                         Eminfnorm=2., Emaxfnorm=10.,\
                         namebase='multi_sample3_spectrum%i', simputdir=None,\
                         specnum=9282):
    '''
    Based on Sarah Walsh's simput writing script: simput_all.py
    '''
    outname = 'simput_%s.fits'%namebase%specnum   
    if simputdir is None:
        simputdir = './'
    outname = simputdir + outname 
    flux_ph, flux_eng = spec.get_flux_in_band(Eminfnorm, Emaxfnorm)    
    
    #source catalog columns, formats and units
    column_src=['SRC_ID', 'SRC_NAME', 'RA', 'DEC', 'E_MIN', 'E_MAX', 'FLUX', 'SPECTRUM', 'IMAGE', 'TIMING']
    formats_src=['J', '32A', 'E', 'E', 'E', 'E', 'E', '32A', '32A', '60A']
    units_src=['','','deg','deg','keV','keV','erg/s/cm**2','','','']  
    a_src = [[1], ['spec%i'%specnum], [np.float32(ra)], [np.float32(dec)], [np.float32(Eminfnorm)], [np.float32(Emaxfnorm)], [np.float32(flux_eng)],\
                  ['[SPECTRUM,1][#row==1]'],['NULL'],['NULL']]
    columns_src=[ap.io.fits.Column(name=column_src[i],\
                                    format=formats_src[i],\
                                    unit=units_src[i],\
                                    array=a_src[i]) \
                for i in range(len(column_src))
                ]

    cols_src = ap.io.fits.ColDefs(columns_src)
    hdu_src = ap.io.fits.BinTableHDU().from_columns(cols_src)
    
    #extension header mandatory keywords
    header_src=['HDUCLASS', 'HDUCLAS1', 'HDUVERS', 'EXTNAME', 'RADECSYS', 'EQUINOX']
    keywords_src=['HEASARC/SIMPUT', 'SRC_CAT', '1.1.0', 'SRC_CAT', 'FK5', 2000]

    hdr_src = hdu_src.header
    for i in range(len(header_src)):
	   hdr_src[header_src[i]] = keywords_src[i]
    
    ## source catalogue

    flux_spec = spec.flux
    energy_spec = spec.emid
    #print flux_spec.shape, energy_spec.shape
    
    a_spec = [(energy_spec.astype(np.float32)).reshape(1, len(energy_spec)), flux_spec.astype(np.float32).reshape(1, len(energy_spec))]
    
    #source catalog columns, formats and units
    column_spec=['ENERGY', 'FLUXDENSITY']
    formats_spec=['%iE'%(len(energy_spec)), '%iE'%(len(flux_spec))]
    units_spec=['keV','photon/s/cm**2/keV']
    
    #creating the source catalog
    
    columns_spec = [ap.io.fits.Column(name=column_spec[i],\
                                      format=formats_spec[i],\
                                      unit=units_spec[i],\
                                      array=a_spec[i]) \
                    for i in range(len(column_spec)) ]
    
    cols_spec = ap.io.fits.ColDefs(columns_spec)
    hdu_spec = ap.io.fits.BinTableHDU().from_columns(cols_spec)
    
    #extension header mandatory keywords
    header_spec=['HDUCLASS', 'HDUCLAS1', 'HDUVERS', 'EXTNAME', 'EXTVERS']
    keywords_spec=['HEASARC/SIMPUT', 'SPECTRUM', '1.1.0', 'SPECTRUM', '1']
    
    #reating the extension header
    hdr_spec = hdu_spec.header
    for i in range(len(header_spec)):
    	  hdr_spec[header_spec[i]] = keywords_spec[i]
    
    #creating an empty primary header and HDU lists
    primary_hdu = ap.io.fits.PrimaryHDU()
    hdul = ap.io.fits.HDUList([primary_hdu, hdu_src, hdu_spec])
    try:
        hdul.writeto(outname, overwrite=True)
    except TypeError:
        hdul.writeto(outname)
    
    return outname

def make_simput(specnum, ions=['o7', 'o8', 'fe17', 'ne9'], specfile=specfile,\
                Emin=0.2, Emax=12., spacing_keV=0.00025, \
                ra=0., dec=0., Eminfnorm=2., Emaxfnorm=10.,\
                namebase='multi_sample3_spectrum%i', simputdir=odir,\
                n_H22_mw=0.02, n_H22_bl=0.02, z_bl=0.2, **blazarparams):
    '''                      
    Blazar spectrum parameters: normalisation: flux in erg/s/cm^2 between Emin 
                                               and Emax 
                                               Etotcgs=7.5e-12, 
                                               EminkeV=2., 
                                               EmaxkeV=10. 
                                PL index:      Gammaphot=2.
    simput file name is  'simput_%s.fits'%namebase%specnum 
    ''' 
    spec = generate_spectrum(specnum=specnum, Emin=Emin, Emax=Emax, ions=ions,\
                      specfile=specfile, spacing_keV=spacing_keV, 
                      n_H22_mw=n_H22_mw, n_H22_bl=n_H22_bl, z_bl=z_bl,
                      **blazarparams)
                      
    save_simput_spectrum(spec, ra=ra, dec=dec,\
                         Eminfnorm=Eminfnorm, Emaxfnorm=Emaxfnorm,\
                         namebase=namebase, simputdir=simputdir,\
                         specnum=specnum)


def save_simput_lineimg(img, spec, pixsize_deg, Elow_keV=None, Ehigh_keV=None, racen=0., deccen=0.,\
                         namebase='basic_lineimg_%s', name='test'):
    '''
    Based on Sarah Walsh's simput writing script: simput_all.py
    
    img: ax0 -> RA, ax1 -> DEC, flux in photons / cm2 / s / steradian
    racen and deccen are image center coordinates (deg)
    '''
    outname = 'simput_%s.fits'%namebase%name  
    
    # get total image flux
    if Elow_keV is None:
        Elow_keV = spec.ebins
    flux_ph, flux_eng = spec.get_flux_in_band(Elow_keV, Ehigh_keV)
    eng_per_photon = flux_eng / flux_ph
    flux_tot = np.sum(img) *(pixsize_deg * np.pi / 180.)**2 # surface brightness -> photons /cm2 /s
    flux_tot *= eng_per_photon # photons /cm2 /s -> erg / cm2 / s
    
    ### source catalog columns, formats and units
    column_src=['SRC_ID', 'SRC_NAME', 'RA', 'DEC', 'E_MIN', 'E_MAX', 'FLUX', 'SPECTRUM', 'IMAGE', 'TIMING']
    formats_src=['J', '32A', 'E', 'E', 'E', 'E', 'E', '32A', '32A', '60A']
    units_src=['','','deg','deg','keV','keV','erg/s/cm**2','','','']  
    a_src = [[1], ['lineim'], [np.float32(racen)], [np.float32(deccen)], [np.float32(Elow_keV)], [np.float32(Ehigh_keV)], [np.float32(flux_tot)],\
                  ['[SPECTRUM,1][#row==1]'],['[IMAGE,1]'],['NULL']]
    columns_src=[ap.io.fits.Column(name=column_src[i],\
                                    format=formats_src[i],\
                                    unit=units_src[i],\
                                    array=a_src[i]) \
                for i in range(len(column_src))
                ]

    cols_src = ap.io.fits.ColDefs(columns_src)
    hdu_src = ap.io.fits.BinTableHDU().from_columns(cols_src)
    
    #extension header mandatory keywords
    header_src=['HDUCLASS', 'HDUCLAS1', 'HDUVERS', 'EXTNAME', 'RADECSYS', 'EQUINOX']
    keywords_src=['HEASARC/SIMPUT', 'SRC_CAT', '1.1.0', 'SRC_CAT', 'FK5', 2000]

    hdr_src = hdu_src.header
    for i in range(len(header_src)):
	   hdr_src[header_src[i]] = keywords_src[i]
    
    ## spectrum extension
      
    flux_spec = spec.flux
    energy_spec = spec.emid
    #print flux_spec.shape, energy_spec.shape
    
    a_spec = [(energy_spec.astype(np.float32)).reshape(1, len(energy_spec)), flux_spec.astype(np.float32).reshape(1, len(energy_spec))]
    
    #spectrum columns, formats and units
    column_spec=['ENERGY', 'FLUXDENSITY']
    formats_spec=['%iE'%(len(energy_spec)), '%iE'%(len(flux_spec))]
    units_spec=['keV','photon/s/cm**2/keV']
    
    columns_spec = [ap.io.fits.Column(name=column_spec[i],\
                                      format=formats_spec[i],\
                                      unit=units_spec[i],\
                                      array=a_spec[i]) \
                    for i in range(len(column_spec)) ]
    
    cols_spec = ap.io.fits.ColDefs(columns_spec)
    hdu_spec = ap.io.fits.BinTableHDU().from_columns(cols_spec)
    
    #extension header mandatory keywords
    header_spec=['HDUCLASS', 'HDUCLAS1', 'HDUVERS', 'EXTNAME', 'EXTVERS']
    keywords_spec=['HEASARC/SIMPUT', 'SPECTRUM', '1.1.0', 'SPECTRUM', '1']
    
    #creating the extension header
    hdr_spec = hdu_spec.header
    for i in range(len(header_spec)):
    	  hdr_spec[header_spec[i]] = keywords_spec[i]    
    
    
    
    ### image extension
    
    
    
    ### image extension
    # in astropy.io.fits, axis0 = y, axis1 = x, [0, 0] = top left
    # im my images, axis0=x
    hdu_img  = ap.io.fits.ImageHDU(data=(img[::-1, :]).T)    
    
    #extension header mandatory keywords + WCS header keywords
    #CRVALi is overwritten by source RA, DEC in SRC_CAT
    # add IMGROTA and IMGSCAL keywords to SRC_CAT to rotate, scale the image 
    header_img= ['HDUCLASS', 'HDUCLAS1', 'HDUVERS',  'EXTNAME', 'EXTVERS',\
                 'RADECSYS', 'EQUINOX',  'WCSAXIS',\
                 'CUNIT1',   'CUNIT2',   'CTYPE1',   'CTYPE2',\
                 'CRVAL1',   'CRVAL2',   'CDELT1',   'CDElT2',\
                 'CRPIX1',               'CRPIX2']
    keywords_img=['HEASARC/SIMPUT', 'IMAGE', '1.1.0', 'IMAGE', '1',\
                 'FK5',      2000,       2,\
                 'deg',      'deg',      'RA---TAN', 'DEC--TAN',\
                 racen,         deccen,  pixsize_deg, pixsize_deg,\
                 float(img.shape[0]) / 2. + 1., float(img.shape[0]) / 2. + 1.]
    
    #creating the extension header
    hdr_img = hdu_img.header
    for i in range(len(header_img)):
    	  hdr_img[header_img[i]] = keywords_img[i]   
    
    #creating an empty primary header and HDU lists
    primary_hdu = ap.io.fits.PrimaryHDU()
    hdul = ap.io.fits.HDUList([primary_hdu, hdu_src, hdu_spec, hdu_img])
    try:
        hdul.writeto(outname, overwrite=True)
    except TypeError:
        hdul.writeto(outname)
    
    return outname
    
def make_simput_lineimg(imgfile, Eline_keV, fwhm_line_keV,\
                        redshift, pixsize_deg,\
                        Elow_keV=None, Ehigh_keV=None, racen=0., deccen=0.,\
                        sel=None, key=None, \
                        namebase='basic_lineimg_%s', name=None):
    if name is None:
        name = imgfile.split('/')[-1] # remove directory path
        name = '.'.join(name.split('.')[:-1]) # remove file extension name
        if sel is not None:
            name = name + '_indices-%s'%(sel)
    
    if Elow_keV is None:
        Elow_keV = (Eline_keV - 10. * fwhm_line_keV ) * (1. + redshift)
    if Ehigh_keV is None:
        Ehigh_keV = (Eline_keV + 10. * fwhm_line_keV ) * (1. + redshift)
    dE = fwhm_line_keV / 10.
    
    Ebins = np.arange(Elow_keV, Ehigh_keV + dE, dE)
    flux  = np.zeros(len(Ebins) - 1)
    spec = soxs.spectra.spectrum(Ebins, flux)
    # line amplitude is arbitrary: flux normalisation comes from the image
    spec.add_emission_line(Eline_keV * (1. + redshift),\
                           fwhm_line_keV * (1. + redshift),\
                           (1., 'photon/cm**2/s'),
                           line_type='gaussian')
    if sel is None:
        sel = (slice(None, None, None), slice(None, None, None))
    if imgfile[-4:] == '.npz':
        tmp = np.load(imgfile)
        if key is None:
            try:
               img = tmp['arr_0']
            except KeyError:   
               key = tmp.keys()[0]
               print('Using image %s from file %s'%(imgfile, key))
               img = tmp[key]
        else:
            img = tmp[key]
        img = img[sel]
            
    elif imgfile[-5:] == '.hdf5':
        tmp = h5py.File(imgfile, 'r')
        ds = tmp[key]
        totshape = ds.shape

        if sel[0].start is None:
            start0 = 0
        else:
            start0 = sel[0].start
        if sel[1].start is None:
            start1 = 1
        else:
            start1 = sel[1].start

        if sel[0].stop is None:
            stop0 = totshape[0]
        else:
            stop0 = sel[0].stop
        if sel[1].stop is None:
            stop1 = totshape[1]
        else:
            stop1 = sel[1].stop
        
        if sel[0].step is None:
            step0 = 0
        else:
            step0 = sel[0].step
        if sel[1].step is None:
            step1 = 1
        else:
            step1 = sel[1].step
        
        outshape = ((stop0 - start0 + step0 - 1) // step0, (stop1 - start1 + step1 - 1) // step1)
        img = np.emtpy(outshape)
        ds.read_direct(img, source_sel=sel)
        
    save_simput_lineimg(img, spec, pixsize_deg,\
                        Elow_keV=Elow_keV, Ehigh_keV=Ehigh_keV,\
                        racen=racen, deccen=racen,\
                        namebase=namebase, name=name)
    tmp.close()