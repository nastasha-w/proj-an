#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:08:45 2019

@author: wijers
"""

import numpy as np
import scipy as sp
import astropy.convolution as conv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import string
import fnmatch 
import os
import h5py

import make_maps_opts_locs as ol
import eagle_constants_and_units as cu
import make_maps_v3_master as m3
import makecddfs as mc

radians = 1.
degrees = np.pi/180.
arcmin = 1./60. * degrees 
arcsec = 1./3600. * degrees

def get_kernel(fwhm, z, length, numpix, numsig=10):
    '''
    fwhm: in arcsec
    length: in cMpc
    '''
    numsig = int(numsig)
    comdist = m3.comoving_distance_cm(z)
    longlen = float(length) * cu.cm_per_mpc  
    fwhm *= arcsec
    
    if comdist > longlen/2.: # even at larger values, the projection along z-axis = projection along sightline approximation will break down
        adist = comdist/(1.+z)
    else: 
        adist = longlen/2./(1.+z) 
    image_angle_per_pixel = longlen / (1. + z) / numpix * 1. / adist * radians # these functions work in units of angles
    #print image_angle_per_pixel
    gauss_sigma = fwhm / image_angle_per_pixel / (2. * np.sqrt(2. * np.log(2)))  # fhwm in radians -> sigma in pixel size units
    #print gauss_sigma
    kernel_pix = 2 * int(np.ceil(numsig * gauss_sigma)) + 1
    #print kernel_pix
    basegrid = np.indices((kernel_pix, kernel_pix)) - kernel_pix // 2
    #print basegrid
    kernel = 1. / (2. * np.pi * gauss_sigma**2 ) * np.exp(-1 * (basegrid[0]**2 + basegrid[1]**2) / (2. * gauss_sigma**2))
    return kernel
    
def convolve_image_kernel(imgdct, fwhm, z, length, numpix, numsig=10):
    try:
        img = 10**(imgdct['arr_0'])
    except KeyError:
        img = 10**(imgdct[imgdct.keys()[0]])
    kernel = get_kernel(fwhm, z, length, numpix, numsig=numsig)
    kernel /= np.sum(kernel)
     
    convimg = conv.convolve_fft(img, kernel, boundary='wrap', allow_huge=True)
    convimg[convimg < 0.] = 0. # possible due to noise -> fix to avoid trouble when taking log
    print np.sum(img), np.sum(convimg)
    return convimg

def convolve_image_npz_to_npz(filename, fwhm, numsig=10, sidelength='box'):
    '''
    10 sigma already means ~20 OOM in kernel value; should be enough
    '''

    if '/' not in filename:
        filename = ol.ndir + filename
    img = np.load(filename)
    simdata = mc.get_simdata_from_outputname(filename)
    numpix = simdata['numpix']
    cosmopars = mc.getcosmopars(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
    if sidelength == 'box':
        sidelength = cosmopars['boxsize'] / cosmopars['h']
    
    outimg = convolve_image_kernel(img, fwhm, cosmopars['z'], sidelength, numpix, numsig=numsig)

    outname = filename.split('/')[-1]
    if outname[-4:] == '.npz':
        outname = outname[:-4]
    outname = outname + '_kernelconv_fwhm-%.2farcsec_extent-%isigma'%(fwhm, numsig)
    
    outimg = np.log10(outimg)
    minf = np.min(outimg[np.isfinite(outimg)])
    maxf = np.max(outimg)
    np.savez(ol.ndir + outname, arr_0=outimg, minfinite=minf, max=maxf)

def addminmax(filename):
    fil = np.load(ol.ndir + filename)
    if 'minfinite' in fil.keys() and 'max' in fil.keys():
        return None
    img = fil['arr_0']
    minf = np.min(img[np.isfinite(img)])
    maxf = np.max(img)
    np.savez(ol.ndir + filename, arr_0=img, minfinite=minf, max=maxf)

def emission_est_from_sfrsd(filename, eff, sidelength='box'):
    # sfrsd: units g / s / cm^2
    simdata = mc.get_simdata_from_outputname(filename)
    simfile = m3.Simfile(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
    ion = filename.split('/')[-1]
    ion = ion.split('_')[1]
    print ion
    if sidelength == 'box':
        sidelength = simfile.boxsize / simfile.h
    
    # Ls, axis: only cares about x and y dimension of image -> just put in Ls as sidelength and select first 2
    # vardict: only iuse to access simfile
    vardict = m3.Vardict(simfile, '0', [])
    sd_to_sb = m3.lumninosty_to_Sb(vardict, (sidelength,)*3, 0, 1 , 2, simdata['numpix'], simdata['numpix'], 'o7r') \
               / m3.Nion_to_coldens(vardict, (sidelength,)*3, 0, 1 , 2, simdata['numpix'], simdata['numpix'])
    
    sfr_to_SNe_efficiency = 8.73e15 # erg g−1, for fth=1: used in the EAGLE simulations, assuming all stars at >10^6 Msun go off as 10^51 erg SNe and a Chabrier IMF     
    if ion == 'o7r':
        total_to_emission_efficiency = 0.01 # half of max O7 emission / (cooling - heating)
    elif ion == 'o8':
        total_to_emission_efficiency = 0.035 # half of max O8 emission / (cooling - heating)
    return sd_to_sb * sfr_to_SNe_efficiency * total_to_emission_efficiency * eff

def emission_est_from_lit(filename, sidelength='box', method='mc+02'):
    ## D. McCammon, R. Almy, E. Apodaca et al. 2002 (mc+02):
    #  O VII and O VIII (and C VI) line detections over ~ 1 sr
    #  "The field of viewwas centered at l= 90deg, b= +60deg, a region chosen
    # to betypical of high latitudes for the 500–1000 eV region by avoiding 
    # Loop I, the North Polar Spur, and other features thought to be due to 
    # supernova remnants and superbubbles.About 38% of this general 
    # high-latitude background is known to be produced by distant AGNs 
    # (Hasinger et al.1993), but the source and emission mechanism of the
    # remainder is unknown. In the 100–300 eV range, this part ofthe  sky  
    # includes some of  the brightest high-latitude enhancements. These 
    # apparently are partly due to extensions of the local hot bubble in these 
    # directions but may also be produced by patches of hot gas in the halo 
    # (Kuntz &Snowden 2000). The field of view and path of a 360deg scan 
    # through the earth to evaluate background are shown in Figure 6.
    ## Henley & Shelton (2013, series includes 2010, 2012)
    # exclude sightlines at |b| < 30 deg from their sample since they want halo
    # data. They model solar wind charge exchange and the local bubble, but not
    # the disk/ISM in general, it seems.
    #"Yoshino et al. (2009) found a tight correlation between the observed Ovii
    # and Oviiiintensities in their sample of spectra, with a non-zero “floor” 
    # to the Ovii emission, leading them to conclude that their spectra 
    # included a uniform local component with Ovii and Oviii intensities of∼2 
    # and∼0 photons  cm−2s−1sr−1, respectively.They subsequently used a 
    # foreground model with these oxygenintensities to obtain their halo
    # measurements." 
    # "the energy  available  from SNe (8×1038erg s−1kpc−2; e.g., Yao et al. 
    # 2009)." 
    simdata = mc.get_simdata_from_outputname(filename)
    simfile = m3.Simfile(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
    ion = filename.split('/')[-1]
    ion = ion.split('_')[1]
    print ion
    if sidelength == 'box':
        sidelength = simfile.boxsize / simfile.h
        
    if method == 'mc+02':
        if ion == 'o7r':
            total_intensity = 4.8 # +- 0.8 photons/ cm2 /s /sr, whole triplet
    
        elif ion == 'o8':
            intensity = 1.6  # +- 0.4 photons/ cm2 /s /sr
    


def plot_lineem_over_Lambda(z, ion='o7r'):
    '''
    get ratio of o7r radiative losses over total cooling 
    z: redshift
    for solar metallicities
    '''
    
    lines, logtk, lognh = m3.findemtables(ol.elements_ion[ion], z)
    #logtk, lognh = (np.arange(2., 8.5, 0.25), np.arange(-8., 2., 0.2))
    #lines = np.ones((len(logtk), len(lognh), ol.line_nos_ion['o7r'] + 1))
    lambda_ion = lines[:, :, ol.line_nos_ion[ion]] # axes: T, nH
    print('ok so far')
    dct = {}
    Tvals = logtk
    nHvals = lognh
    # grid shape: nH, T
    Tgrid = (np.array((Tvals,)*len(nHvals))).flatten()
    nHgrid = np.array([(rho,)*len(Tvals) for rho in nHvals]).flatten()
    dct['metallicity'] = np.ones(len(Tgrid))*0.0129
    dct['helium']      = np.ones(len(Tgrid))*0.28055534
    dct['logT'] = Tgrid
    dct['lognH'] = nHgrid
    dct['Density'] = 10**nHgrid / 0.70649785 * m3.ionh.atomw['Hydrogen'] * cu.u
    print('starting on the cooling rates')
    lambda_rad = m3.find_coolingrates(z, dct, method='total_metals')
    
    lambda_rad = np.log10(lambda_rad.reshape(len(nHvals), len(Tvals)))
    lambda_ion = lambda_ion.T
    ionfrac = lambda_ion - lambda_rad

    fig = plt.figure(figsize=(7.5, 3.5))
    grid = gsp.GridSpec(2, 3, width_ratios=[1., 1., 1.], height_ratios=[5., 1.], wspace=0.20, hspace=0.35)
    ax1, ax2, ax3 = tuple([plt.subplot(grid[0, i]) for i in range(3)]) 
    cax1, cax2, cax3 = tuple([plt.subplot(grid[1, i]) for i in range(3)]) 
    
    fontsize = 12
    vdynrange = 10.
    cmap = 'viridis'
    o7max = np.max(lambda_ion[np.isfinite(lambda_ion)])
    totmax = np.max(lambda_rad[np.isfinite(lambda_rad)])
    fracmax = np.max(ionfrac[np.isfinite(ionfrac)])
    ionlabels = {'o7r': 'O \, VII \, r',\
                 'o8':  'O \, VIII '}
    xlabel = r'$\log_{10} \, n_H \; [\mathrm{cm}^{-2}]$'
    ylabel = r'$\log_{10} \, T \; [\mathrm{K}]$'
    lumlabel = r'$\log_{10} \, \Lambda \, / \, n_H^2 \; [\mathrm{erg}\, \mathrm{s}^{-1} \mathrm{cm}^{3}]$'
    lumratlabel = r'$\log_{10} \, \Lambda_{%s} \, / \, \Lambda_{\mathrm{rad}}$'%(ionlabels[ion])
    
    
    img1 = ax1.pcolormesh(nHvals, Tvals, lambda_ion.T, cmap=cmap, vmax=o7max, vmin=o7max-vdynrange)
    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_title('O VII r luminosity', fontsize=fontsize)
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both',\
                   labelleft=True, labeltop=False, labelbottom=True, labelright=False)
    
    plt.colorbar(img1, cax=cax1, orientation='horizontal', extend='min')
    cax1.set_aspect(1./12.)
    cax1.set_xlabel(lumlabel, fontsize=fontsize)
    cax1.tick_params(labelsize=fontsize-1)
    
    img2 = ax2.pcolormesh(nHvals, Tvals, lambda_rad.T, cmap=cmap, vmax=totmax, vmin=totmax-vdynrange)
    ax2.set_xlabel(xlabel, fontsize=fontsize)
    #ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax2.set_title(r'cooling $-$ heating', fontsize=fontsize)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both',\
                   labelleft=False, labeltop=False, labelbottom=True, labelright=False)
    
    plt.colorbar(img2, cax=cax2, orientation='horizontal', extend='min')
    cax2.set_aspect(1./12.)
    cax2.set_xlabel(lumlabel, fontsize=fontsize)
    cax2.tick_params(labelsize=fontsize-1)
    
    img3 = ax3.pcolormesh(nHvals, Tvals, ionfrac.T, cmap=cmap, vmax=fracmax, vmin=fracmax-vdynrange)
    ax3.set_xlabel(xlabel, fontsize=fontsize)
    #ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax3.text(0.95, 0.05, 'max: %.4f'%(fracmax), fontsize=fontsize-1, horizontalalignment='right', verticalalignment='bottom', transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.3))
    ax3.set_title('luminosity fraction', fontsize=fontsize)
    ax3.minorticks_on()
    ax3.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both',\
                   labelleft=False, labeltop=False, labelbottom=True, labelright=False)
    
    plt.colorbar(img3, cax=cax3, orientation='horizontal', extend='min')
    cax3.set_aspect(1./12.)
    cax3.set_xlabel(lumratlabel, fontsize=fontsize)
    cax3.tick_params(labelsize=fontsize-1)
    plt.savefig('/net/luttero/data2/imgs/ion_tables/luminosity_fraction_%s_Zsolar_z%.2f.pdf'%(ion, z), format='pdf')
    
    return ionfrac

def addimgs(basename, basesfrname, efficiencies, fwhms=[5., 15., 60.], numsigs=[10, 10, 10]):
    efficiencies = np.array(efficiencies)
    
    if basename[-4:] == '.npz':
        basename = basename[:-4]
    if basesfrname[-4:] == '.npz':
        basesfrname = basesfrname[:-4]
        
    parts = basename.split('/')[-1]
    parts = parts.split('_')
    
    smoothext = '_kernelconv_fwhm-%.2farcsec_extent-%isigma'
    savenames = {(fwhms[fwhmi], eff): basename + '_plus_SFRtoSNetoEmOverLambda_eff-%.1e'%(eff) + smoothext%(fwhms[fwhmi], numsigs[fwhmi]) for fwhmi in range(len(fwhms)) for eff in efficiencies}
    savenames.update({(0., eff): basename + '_plus_SFRtoSNetoEmOverLambda_eff-%.1e'%(eff) for eff in efficiencies})
    loadnames = {fwhms[fwhmi]: (basename + smoothext%(fwhms[fwhmi], numsigs[fwhmi]) + '.npz', basesfrname + smoothext%(fwhms[fwhmi], numsigs[fwhmi]) + '.npz') for fwhmi in range(len(fwhms)) for eff in efficiencies}
    loadnames.update({0.: (basename + '.npz', basesfrname + '.npz')})
    
    toteff =  {eff: emission_est_from_sfrsd(basename, eff, sidelength='box') for eff in efficiencies}
    
    keys = savenames.keys()
    print keys
    keys0 = list(set(key[0] for key in keys))
    keys1 = list(set(key[1] for key in keys))
    for key0 in keys0:
        emarr = 10**np.load(ol.ndir + loadnames[key0][0])['arr_0']
        sfarr = 10**np.load(ol.ndir + loadnames[key0][1])['arr_0']
        for key1 in keys1:
            eff   = toteff[key1]
            outarr = eff * sfarr + emarr
            outarr = np.log10(outarr)
            minf = np.min(outarr[np.isfinite(outarr)])
            maxf = np.max(outarr)
            np.savez(ol.ndir + savenames[(key0, key1)], arr_0=outarr, minfinite=minf, max=maxf)
        
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str    
    
def compareplots_L50slice(filenames, labels, sel=None, savename=None,\
                          halocat=None, galaxyid=None, radext_r200c=2.,\
                          dataloc='upper left', ion=None, efficiency=None,\
                          vmin=None, vmax=None):
    '''
    sel needs to be specified if no galaxyid is given
    assumed proijection along z-axis for a 50 cMpc^2, 6.25 cMpc slice 
    '''
    # 50 Mpc box at z=0.1: 0.115 rad or 6.59 degrees size on sky
    mdir = '/net/luttero/data2/imgs/CGM/psf_effects_xray_emission/'
    nsub = len(filenames)
    Ltot = 50.
    llcorner = [0., 0.] # projection plane coordinates of the lower left corner
    ccolor = 'blue'
    
    if halocat is not None and galaxyid is not None:
        if '/' not in halocat:
            halocat = ol.pdir + halocat
        if halocat[-5:] != '.hdf5':
            halocat = halocat + '.hdf5'
        cat = h5py.File(halocat, 'r')
        cosmopars = {key: cat['Header/cosmopars'].attrs[key] for key in cat['Header/cosmopars'].attrs.keys()}
        ids = np.array(cat['galaxyid'])
        galind = np.where(ids == galaxyid)[0][0]
        m200 = np.array(cat['M200c_Msun'])[galind]
        r200 = np.array(cat['R200c_pkpc'])[galind] / cosmopars['a'] / 1.e3 
        xcen = np.array(cat['Xcop_cMpc'][galind])
        ycen = np.array(cat['Ycop_cMpc'][galind])
        
        comdist = m3.comoving_distance_cm(cosmopars['z'])
        longlen = float(Ltot) * cu.cm_per_mpc  
        
        
    files = [np.load(ol.ndir + filename) for filename in filenames]
    fig = plt.figure(figsize=(5. * nsub + 1., 5.))
    grid = gsp.GridSpec(1, len(filenames) + 1, width_ratios=list((5.,) * len(filenames)) + [1.], wspace=0.05, hspace=0.0)
    axes = tuple([plt.subplot(grid[0, i]) for i in range(nsub)])
    cax = plt.subplot(grid[0, nsub])
    
    vdynrange = 10.
    cmap = 'cubehelix_r'
    fontsize = 12.
    
    if vmax is None:
        try:
            vmax = max([fil['max'][()] for fil in files])
        except KeyError:
            del files
            for fil in filenames:
                addminmax(fil)
            files = [np.load(ol.ndir + filename) for filename in filenames]
            vmax = max([fil['max'][()] for fil in files])
    if vmin is None:
        vmin = vmax - vdynrange
        
    for i in range(nsub):
        labelleft = i == 0
        ylabel = i == 0
        ax = axes[i]
        fil = files[i]
        
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize - 1, direction='in', right=True, top=True, axis='both', which='both',\
                       labelleft=labelleft, labeltop=False, labelbottom=True, labelright=False)
        ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        if ylabel:
            ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        ax.text(0.05, 0.95, labels[i], fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
        
        totimg = fil['arr_0']
        npix = totimg.shape[0]
        
        if halocat is not None:
            if comdist > longlen/2.: # even at larger values, the projection along z-axis = projection along sightline approximation will break down
                adist = comdist/(1.+cosmopars['z'])
            else: 
                adist = longlen/2./(1.+cosmopars['z']) 
            image_angle_per_pixel = longlen * cosmopars['a'] / npix * 1. / adist * radians # these functions work in units of angles
            
            pixsize = Ltot / float(npix)
            arcminsize = pixsize / (image_angle_per_pixel / (15. * arcsec))
            #print arcminsize
            
            imgext = np.array([[xcen - radext_r200c * r200,\
                                xcen + radext_r200c * r200],\
                               [ycen - radext_r200c * r200,\
                                ycen + radext_r200c * r200]])
            imgext[0] -= llcorner[0]
            imgext[1] -= llcorner[1]
            imgext /= pixsize        
            sel = np.round(imgext, 0).astype(np.int)
            sel = (slice(sel[0][0], sel[0][1], None), slice(sel[1][0], sel[1][1], None))
            
            imgext *= pixsize
            imgext[0] += llcorner[0]
            imgext[1] += llcorner[1]
            extent = imgext.flatten()
        else:
            extent = (sel[0].start if sel[0].start is not None else 0,\
                  sel[0].stop  if sel[0].stop  is not None else npix,\
                  sel[1].start if sel[1].start is not None else 0,\
                  sel[1].stop  if sel[1].stop  is not None else npix )
            extent = np.array(extent, dtype=np.float)
            extent *= Ltot / float(npix)
        
        img = ax.imshow(totimg[sel].T, interpolation='nearest', origin='lower',\
                        cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
        
        if halocat is not None:
            circle = plt.Circle((xcen, ycen), r200, color=ccolor, fill=False)
            ax.add_artist(circle)
            
            if ion == 'o7r':
                ionname = r'$\mathrm{O\, VII}, 21.60\,\mathrm{\AA}$'
            elif ion == 'o8':
                ionname = r'$\mathrm{O\, VIII}, 18.97\,\mathrm{\AA}$'
            else:
                ionname = ''  
            if efficiency == None:
                effname = ''
            elif efficiency == 0.:
                effname = 'No ISM emission'
            else:
                effname = r'ISM: $\epsilon_{\mathrm{SNe}, \mathrm{rad}} =' + latex_float(efficiency) + r'$'
            zname = r'$z = %.1f$'%(cosmopars['z'])
            text0 = '\n'.join([ionname, zname, effname])
            
            galaxyname = 'galaxy %s'%galaxyid
            halomassname = r'$M_{200c} = %.1e \, \mathrm{M}_{\odot}$'%m200
            text1 = '\n'.join([galaxyname, halomassname])
            
            if i == 0:
                ax.text(0.95, 0.95, text0, fontsize=fontsize, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
            if i == 1:
                ax.text(0.95, 0.95, text1, fontsize=fontsize, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
            if i == 2:
                ax.text(xcen +  r200 / 2.**0.5, ycen - r200 / 2.**0.5, r'$R_{200c}$', fontsize=fontsize, horizontalalignment='left', verticalalignment='top', color=ccolor)
                ax.plot([xcen + 0.* r200 - 0.5 * arcminsize, xcen + 0.* r200 + 0.5 * arcminsize], [ycen - 1.85 * r200,  ycen - 1.85 * r200], color=ccolor)
                ax.text(xcen + 0.* r200, ycen - 1.8 * r200, '15 arcsec', fontsize=fontsize, horizontalalignment='center', verticalalignment='bottom', color=ccolor)
                
    plt.colorbar(img, cax=cax, orientation='vertical', extend='min')
    cax.set_aspect(12.)
    cax.set_ylabel(r'$\log_{10} \,  \mathrm{photons} \;  \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}$', fontsize=fontsize)
    cax.tick_params(labelsize=fontsize-1)
    
    if savename is not None:
        plt.savefig(mdir + 'halozooms_different_fwhm/' + savename + '.pdf', format='pdf', bbox_inches='tight')


def compareplots_L50slice_set(o7base, o8base, halocat, galaxyid, vmin=-4.5, vmax=4.5):
    if 'snap19' in halocat:
        simname = 'EAGLE-Ref-L100_z-1.0'
    elif 'snap27' in halocat:
        simname = 'EAGLE-Ref-L50_z-0.1'
    
    # files in order: efficiencies 0, 1, 1e-2, 1e-4, for each: fwhm = none, 15., 5., 60.
    o7files = fnmatch.filter( next(os.walk(ol.ndir))[2], o7base)  
    o7files.sort()
    o8files = fnmatch.filter( next(os.walk(ol.ndir))[2], o8base)  
    o8files.sort()
    
    efficiencies = [0., 1., 1.e-2, 1.e-4]
    effsls = {0: slice(0, 4, None),\
              1: slice(4, 8, None),\
              1.e-2: slice(8, 12, None),\
              1.e-4: slice(12, 16, None)}
    effnames = {0: 'noISM',\
              1: 'ISM-eff-1',\
              1.e-2: 'ISM-eff-1e-02',\
              1.e-4: 'ISM-eff-1e-04'}
    
    psforder = np.array([0, 2, 1, 3])
    labels = ['no psf', 'fwhm: 5 arcsec', 'fwhm: 15 arcsec', 'fwhm: 60 arcsec']
    
    
    for eff in efficiencies:
        compareplots_L50slice(np.array(o7files[effsls[eff]])[psforder], labels,\
                                 savename='emission_o7r_%s_Z-axis_%s_galaxy-%s'%(simname, effnames[eff], galaxyid),\
                                 halocat=halocat,\
                                 galaxyid=galaxyid, ion='o7r', efficiency=eff,\
                                 vmin=vmin, vmax=vmax)
        compareplots_L50slice(np.array(o8files[effsls[eff]])[psforder], labels,\
                                 savename='emission_o8_%s_Z-axis_%s_galaxy-%s'%(simname, effnames[eff], galaxyid),\
                                 halocat=halocat,\
                                 galaxyid=galaxyid, ion='o8', efficiency=eff,\
                                 vmin=vmin, vmax=vmax)
        
def percentiles_from_histogram(histogram, edgesaxis, axis=-1, percentiles=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
    '''
    get percentiles from the histogram along axis
    edgesaxis are the bin edges along that same axis
    histograms can be weighted by something: this function just solves 
    cumulative distribution == percentiles
    '''
    cdists = np.cumsum(histogram, axis=axis, dtype=np.float) 
    sel = list((slice(None, None, None),)*len(histogram.shape))
    sel2 = np.copy(sel)
    sel[axis] = -1
    sel2[axis] = np.newaxis
    cdists /= (cdists[tuple(sel)])[tuple(sel2)] # normalised cumulative dist: divide by total along axis
    # bin-edge corrspondence: at edge 0, cumulative value is zero
    # histogram values are counts in cells -> hist bin 0 is what is accumulated between edges 0 and 1
    # cumulative sum: counts in cells up to and including the current one: 
    # if percentile matches cumsum in cell, the percentile value is it's rigtht edges -> edge[cell index + 1]
    # effectively, if the cumsum is prepended by zeros, we get a hist bin matches edge bin matching

    oldshape1 = list(histogram.shape)[:axis] 
    oldshape2 = list(histogram.shape)[axis+1:]
    newlen1 = int(np.prod(oldshape1))
    newlen2 = int(np.prod(oldshape2))
    axlen = histogram.shape[axis]
    cdists = cdists.reshape((newlen1, axlen, newlen2))
    cdists = np.append(np.zeros((newlen1, 1, newlen2)), cdists, axis=1)
    cdists[:, -1, :] = 1. # should already be true, but avoids fp error issues

    leftarr  = cdists[np.newaxis, :, :, :] <= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    rightarr = cdists[np.newaxis, :, :, :] >= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    
    leftbininds = np.array([[[ np.max(np.where(leftarr[pind, ind1, :, ind2])[0]) \
                               for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
    # print leftarr.shape
    # print rightarr.shape
    rightbininds = np.array([[[np.min(np.where(rightarr[pind, ind1, :, ind2])[0]) \
                               for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
    # if left and right bins are the same, effictively just choose one
    # if left and right bins are separated by more than one (plateau edge), 
    #    this will give the middle of the plateau
    lweights = np.array([[[ (cdists[ind1, rightbininds[pind, ind1, ind2], ind2] - percentiles[pind]) \
                            / ( cdists[ind1, rightbininds[pind, ind1, ind2], ind2] - cdists[ind1, leftbininds[pind, ind1, ind2], ind2]) \
                            if rightbininds[pind, ind1, ind2] != leftbininds[pind, ind1, ind2] \
                            else 1.
                           for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
                
    outperc = lweights * edgesaxis[leftbininds] + (1. - lweights) * edgesaxis[rightbininds]
    outperc = outperc.reshape((len(percentiles),) + tuple(oldshape1 + oldshape2))
    return outperc

def comparehistograms(ion, snap, efficiency, savename):
    if snap == 27:
        if ion == 'o8':
            base = 'emission_o8_L0050N0752_27_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_z-projection_noEOS_%s_Density*.npz'
        elif ion == 'o7r':
            base = 'emission_o7r_L0050N0752_27_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_z-projection_noEOS_%s_Density*.npz'
        cosmopars_ea = mc.getcosmopars('L0050N0752',snap,'REFERENCE',file_type = 'snap',simulation = 'eagle')
        logrhob_av_ea = np.log10( 3./(8.*np.pi*cu.gravity)*cu.hubble**2 * cosmopars_ea['h']**2 * cosmopars_ea['omegab'] / cosmopars_ea['a']**3 )
    elif snap == 19:
        if ion == 'o8':
            base = 'emission_o8_L0100N1504_19_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_x25.0-pm50.0_y25.0-pm50.0_z-projection_noEOS_%s_Density*.npz'
        elif ion == 'o7r':
            base = 'emission_o7r_L0100N1504_19_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_x25.0-pm50.0_y25.0-pm50.0_z-projection_noEOS_%s_Density*.npz'
        cosmopars_ea = mc.getcosmopars('L0100N1504',snap,'REFERENCE',file_type = 'snap',simulation = 'eagle')
        logrhob_av_ea = np.log10( 3./(8.*np.pi*cu.gravity)*cu.hubble**2 * cosmopars_ea['h']**2 * cosmopars_ea['omegab'] / cosmopars_ea['a']**3 )
        
    effbase = 'eff-'
    convbase = 'fwhm-%.2farcsec'
    kernels = [5., 15., 60.]
    
    if efficiency > 0:
        base = base%('*' + effbase + '%.1e'%efficiency + '*')
        #print base
        files = fnmatch.filter(next(os.walk(ol.pdir))[2], base)
        #print files
    else:
        base = base%('*')
        files = fnmatch.filter(next(os.walk(ol.pdir))[2], base)
        files = list(set([fil if effbase not in fil else None for fil in files]))
        files.remove(None)
    
    file_noconv = files[np.where(['fwhm' not in fil for fil in files])[0][0]]
    files_kernels = [files[np.where([convbase%kernel in fil for fil in files])[0][0]] for kernel in kernels]
    files = [file_noconv] + files_kernels
    titles = ['No kernel'] + ['FWHM: %.0f arcsec'%kernel for kernel in kernels]
    print('using files:')
    print('\n'.join(files))
    
    mdir = '/net/luttero/data2/imgs/CGM/psf_effects_xray_emission/'
    nsub = len(files)
    
    
        
    lfiles = [np.load(ol.pdir + filename) for filename in files]
    fig = plt.figure(figsize=(5. * nsub + 1., 5.))
    grid = gsp.GridSpec(1, nsub + 1, width_ratios=list((5.,) * nsub) + [1.], wspace=0.05, hspace=0.0)
    axes = tuple([plt.subplot(grid[0, i]) for i in range(nsub)])
    cax = plt.subplot(grid[0, nsub])
    
    vdynrange = 10.
    cmap = 'gist_yarg'
    fontsize = 12.
    sbrange = (-6., 4.0)
    deltarange = (-2.5, 7.5) 
    percentiles = np.array([0.05, 0.5, 0.95])
    perclabels_wipsf = ['wi psf, %.0f %%'%(100*perc) for perc in percentiles]
    perclabels_nopsf = ['no psf, %.0f %%'%(100*perc) for perc in percentiles]
    perclabels_nopsf[0] = None
    perclabels_nopsf[2] = None
    perclinestyles = ['dashed', 'solid', 'dashed']
    color_nopsf = 'red'
    color_wipsf = 'limegreen'
    
    
    
    hists = [fil['bins'] for fil in lfiles]
    # extreme edges are +- infinity -> reset. those bins should be empty or unimportant anyway
    rhoedges = [fil['edges'][1]  - logrhob_av_ea for fil in lfiles]
    emedges  = [np.array([fil['edges'][0][1] - 5.] + list(fil['edges'][0][1:-1]) + [fil['edges'][0][-2] + 5.]) for fil in lfiles]
    percpos_em = [percentiles_from_histogram(hists[i], rhoedges[i], axis=1, percentiles=percentiles) for i in range(nsub)]
    percpos_de = [percentiles_from_histogram(hists[i], emedges[i],  axis=0, percentiles=percentiles) for i in range(nsub)]
    emcenters  = [ed[:-1] + 0.5 * np.diff(ed) for ed in emedges]
    decenters  = [ed[:-1] + 0.5 * np.diff(ed) for ed in rhoedges]
    maxind_showperc_em = [np.max(np.where(np.sum(hist, axis=1) >= 10)[0]) + 1 for hist in hists]
    maxind_showperc_de = [np.max(np.where(np.sum(hist, axis=0) >= 10)[0]) + 1 for hist in hists]
    minind_showperc_em = [np.min(np.where(np.sum(hist, axis=1) >= 10)[0]) + 1 for hist in hists]
    minind_showperc_de = [np.min(np.where(np.sum(hist, axis=0) >= 10)[0]) + 1 for hist in hists]
    hists    = [hists[i] / (np.diff(rhoedges[i])[np.newaxis, :] * np.diff(emedges[i])[:, np.newaxis]) for i in range(nsub)]
    vmax = np.log10(np.max([np.max(hist) for hist in hists]))
    vmin = vmax - vdynrange
    
    #return hists, rhoedges, emedges

    
    
    for i in range(nsub):
        labelleft = i == 0
        ylabel = i == 0
        dolegend = i == 0
        ax = axes[i]
        fil = files[i]
        
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize - 1, direction='in', right=True, top=True, axis='both', which='both',\
                       labelleft=labelleft, labeltop=False, labelbottom=True, labelright=False)
        ax.set_xlabel(r'$\log_{10}\, \delta + 1$', fontsize=fontsize)
        ax.set_xlim(*deltarange)
        ax.set_ylim(*sbrange)
        if ylabel:
            ax.set_ylabel(r'$\log_{10} \, \mathrm{SB}\; \mathrm{photons}\, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}$', fontsize=fontsize)
        ax.set_title(titles[i], fontsize=fontsize)
        
        img = ax.pcolormesh(rhoedges[i], emedges[i], np.log10(hists[i]),\
                        cmap=cmap, vmin=vmin, vmax=vmax)
        for j in range(len(percentiles)):
            ax.plot(decenters[i][minind_showperc_de[i]:maxind_showperc_de[i]],\
                percpos_de[i][j][minind_showperc_de[i]:maxind_showperc_de[i]],\
                linestyle=perclinestyles[j], color=color_wipsf, label=perclabels_wipsf[j])
            ax.plot(decenters[0][minind_showperc_de[0]:maxind_showperc_de[0]],\
                percpos_de[0][j][minind_showperc_de[0]:maxind_showperc_de[0]],\
                linestyle=perclinestyles[j], color=color_nopsf, label=perclabels_nopsf[j])
        if dolegend:
            ax.legend(fontsize=fontsize, loc='upper left')
            
    plt.colorbar(img, cax=cax, orientation='vertical', extend='min')
    cax.set_aspect(12.)
    cax.set_ylabel(r'$\log_{10} f_{\mathrm{sky}}$ with $SB > 0$ (arbitrary units)', fontsize=fontsize)
    cax.tick_params(labelsize=fontsize-1)
    
    if savename is not None:
        plt.savefig(mdir + savename + '.pdf', format='pdf', bbox_inches='tight')