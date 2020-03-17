#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:03:06 2020

@author: wijers

adapted from plothistograms_paper2.py
"""


import numpy as np
import h5py
import pandas as pd
import string
import os
import fnmatch
import ctypes as ct

datadir = '/net/luttero/data2/paper2/'
mdir    = '/net/luttero/data2/imgs/CGM/plots_paper2/'

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.patches as mpatch

import make_maps_opts_locs as ol
import eagle_constants_and_units as c #only use for physical constants and unit conversion!
import ion_header as ionh
import cosmo_utils as cu
import ion_line_data as ild
import plot_utils as pu

# avoid issues with interpolation of tables:
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'


fontsize=12
ioncolors = {'o7': 'C3',\
             'o8': 'C0',\
             'o6': 'C2',\
             'ne8': 'C1',\
             'ne9': 'C9',\
             'hneutralssh': 'C6',\
             'fe17': 'C4'}

# neutral hydrogen: indicates DLA cutoff = shoulder of the distribution
approx_breaks = {'o7': 16.0,\
             'o8': 16.0,\
             'o6': 14.3,\
             'ne8': 13.7,\
             'ne9': 15.3,\
             'hneutralssh': 20.3,\
             'fe17': 15.0}

# at max. tabulated density, range where ion frac. is >= 10% of the maximum at max density
Tranges_CIE = {'o6':   (5.3, 5.8),\
               'o7':   (5.4, 6.5),\
               'o8':   (6.1, 6.8),\
               'ne8':  (5.6, 6.1),\
               'ne9':  (5.7, 6.8),\
               'fe17': (6.3, 7.0),\
               }
Tmax_CIE = {'o6':   10**5.5,\
            'ne8':  10**5.8,\
            'o7':   10**5.9,\
            'ne9':  10**6.2,\
            'o8':   10**6.4,\
            'fe17': 10**6.7,\
            }
 
rho_to_nh = 0.752 / (c.atomw_H * c.u)

#retrieved with mc.getcosmopars
cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777,\
                   'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307,\
                   'z': 2.220446049250313e-16}
cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777,\
                   'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307,\
                   'z': 0.10063854175996956}
cosmopars_ea_23 = {'a': 0.6652884960735025,\
                   'boxsize': 67.77,\
                   'h': 0.6777,\
                   'omegab': 0.0482519,\
                   'omegalambda': 0.693,\
                   'omegam': 0.307,\
                   'z': 0.5031073074342141,\
                   }
logrhob_av_ea_28 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
logrhob_av_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )
logrhoc_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars_ea_27['z'], cosmopars=cosmopars_ea_27)**2)

mass_edges_standard = (11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0)
# from get_binmedians
medians_mmin_standardedges = {11.: 11.1941690445,\
                              11.5: 11.6978940964,\
                              12.: 12.2030181885,\
                              12.5: 12.6984605789,\
                              13.: 13.1768951416,\
                              13.5: 13.6665382385,\
                              14.: 14.235991478,\
                              }
minrshow_R200c = 0.1 # right edge of smallest radial bin to show in 3D plots

def add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
             orientation='vertical', clabel=None, fontsize=fontsize, aspect=10.):
    '''
    returns color bar object, color dictionary (keys: lower mass edges)
    '''
    massedges = np.array(massedges)
    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
    keys = sorted(massedges)
    colors = {keys[i]: clist[i] for i in range(len(keys))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation=orientation)
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(aspect)
    
    return cbar, colors


def T200c_hot(M200c, cosmopars):
    # checked against notes from Joop's lecture: agrees pretty well at z=0, not at z=9 (proabably a halo mass definition issue)
    _M200c = np.copy(M200c)
    _M200c *= c.solar_mass # to cgs
    rhoc = (3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars['z'], cosmopars=cosmopars)**2) # Hubble(z) will assume an EAGLE cosmology
    mu = 0.59 # about right for ionised (hot) gas, primordial
    R200c = (_M200c / (200. * rhoc))**(1./3.)
    return (mu * c.protonmass) / (3. * c.boltzmann) * c.gravity * _M200c / R200c

def R200c_pkpc(M200c, cosmopars):
    '''
    M200c: solar masses
    '''
    M200c *= c.solar_mass # to cgs
    rhoc = (3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars['z'], cosmopars=cosmopars)**2) # Hubble(z) will assume an EAGLE cosmology
    R200c = (M200c / (200. * rhoc))**(1./3.)
    return R200c / c.cm_per_mpc * 1e3 

### ion tables for the oxygen split plot
def parse_ionbalfiles_bensgadget2(filename, ioncol=None):
    '''
    returns a temperature-density table from the ascii files
    separated from the main retrieval function findiontables_bensgadget2
    since these ascii files have some messy specifics to deal with 

    table data returned is (log10 balance, lognHcm3, logTK)
    balance is lognH x logT
    '''
    
    ## deal with the ascii format -> pandas dataframe
    # first line has some issues: parse explicitly and pass as arguments to read_csv
    with open(filename, 'r') as fi:
        head = fi.readline()
    if head[0] == '#':
       head = head[1:]
       
    # spacing around column names is inconsistent; split -> strip produces a bunch of empty strings in the list
    columns = head.split(' ')
    columns = [column.strip() for column in columns]
    while '' in columns:
        columns.remove('')
    # last 'column' is a redshift indicator (format 'redshift= <#.######>')
    zcol = ['redshift' in column for column in columns]
    if np.any(zcol):
        zinds = np.where(zcol)[0]
        zfilename = float(filename.split('/')[-1][2:7]) * 1e-4
        for zi in zinds:
            #rcol = columns[zi]
            zcol = float(columns[zi + 1])
            if not np.isclose(zcol, zfilename, atol=2e-4, rtol=1e-5):
                raise RuntimeError('redshift value mismatch for file %s: %s from file name, %s in file'%(filename, zfilename, zcol))
            columns.remove(columns[zi +1])
            columns.remove(columns[zi])
                
    # get the table column name
    if ioncol is not None:
        elt, num = ild.get_elt_state(ioncol)
        elt = string.capwords(elt)
        snum = ild.arabic_to_roman[num]
        columnname = elt[:9 - len(snum)] + snum
        usecols = ['Hdens', 'Temp', columnname]
    else:
        usecols = None
    
    ## use pandas to read in the file
    #print(columns)
    #print(usecols)
    df = pd.read_csv(filename, header=None, names=columns, usecols=usecols, sep='  ', comment='#', index_col=['Hdens', 'Temp'])
    if ioncol is None:
        return df

    # reshape tables: since logT, lognH values are exactly the same across 
    # rows/columns, not just fp close, pandas can deal with this easily        
    df = pd.pivot_table(df, values=columnname, index=['Hdens'], columns=['Temp'])
    ionbal = np.array(df)
    logTK = np.array(df.columns)
    lognHcm3 = np.array(df.index)
    
    return ionbal, lognHcm3, logTK


def findiontables_bensgadget2(ion, z):
    '''
    gets ion balance tables at z by interpolating Ben Oppenheimer's ascii 
    ionization tables made for gagdet-2 analysis
    
    note: the directory is set in opts_locs, but the file name pattern is 
    hard-coded
    '''
    # from Ben's tables, using HM01 UV bkg,
    # files are ascii, contain ionisation fraction of a species for rho, T
    # different files -> different z
    
    
    # search for the right files
    pattern = 'lt[0-9][0-9][0-9][0-9][0-9]f100_i31'
    # determined with ls -l and manual inspection of exmaples that these smaller 
    # files only contain data for low densities.   
    # in order to be able to interpolate, use only the complete files
    files_excl = ['lt01006f100_i31',\
                  'lt04675f100_i31',\
                  'lt10530f100_i31',\
                  'lt18710f100_i31',\
                  'lt30170f100_i31',\
                  'lt68590f100_i31',\
                  'lt94790f100_i31',\
                  ]
    zsel = slice(2, 7, None)
    znorm = 1e-4
    tabledir = ol.dir_iontab_ben_gadget2

    files = fnmatch.filter(next(os.walk(tabledir))[2], pattern)
    #print(files)
    for filen in files_excl:
        if filen in files:
            files.remove(filen)
    files_zs = [float(fil[zsel]) * znorm for fil in files]
    files = {files_zs[i]: files[i] for i in range(len(files))}

    zs = np.sort(np.array(files_zs))
    zind2 = np.searchsorted(zs, z)
    if zind2 == 0:
        if np.isclose(z, zs[0], atol=1e-3, rtol=1e-3): 
            zind1 = zind2 # just use the lowest z if it's close enough
        else:
            raise RuntimeError('Requested redshift %s is outside the tabulated range %s-%s'%(z, zs[0], zs[-1]))
    elif zind2 == len(zs):
        if np.isclose(z, zs[-1], atol=1e-3, rtol=1e-3): 
            zind2 -= 1 # just use the highest z if it's close enough
            zind1 = zind2
        else:
            raise RuntimeError('Requested redshift %s is outside the tabulated range %s-%s'%(z, zs[0], zs[-1]))
    else:
        zind1 = zind2 - 1
    
    z1 = zs[zind1]
    z2 = zs[zind2]
    if z1 == z2:
        w1 = 1.
        w2 = 0.
    else:
        w1 = (z - z2) / (z1 - z2)
        w2 = 1. - w1
    file1 = tabledir + files[z1]
    file2 = tabledir + files[z2]   
    
    if z1 == z2:
        ionbal, lognHcm3, logTK = parse_ionbalfiles_bensgadget2(file1, ioncol=ion)
    else:
        ionbal1, lognHcm31, logTK1 = parse_ionbalfiles_bensgadget2(file1, ioncol=ion)
        ionbal2, lognHcm32, logTK2 = parse_ionbalfiles_bensgadget2(file2, ioncol=ion)
        if not (np.all(logTK1 == logTK2) and np.all(lognHcm31 == lognHcm32)):
            raise RuntimeError('Density and temperature values used for the closest two tables do not match:\
                               \n%s\n%s\nused for redshifts %s, %s around desired %s'%(file1, file2, z1, z2, z))
        logTK = logTK1 #np.average([logTK1, logTK2], axis=0)
        lognHcm3 = lognHcm31 #np.average([lognHcm31, lognHcm32], axis=1)
        logionbal = np.log10(w1 * 10**ionbal1 + w2 * 10**ionbal2)

    return logionbal, lognHcm3, logTK


def find_ionbal_bensgadget2(z, ion, dct_nH_T):
    table_zeroequiv = 10**-9.99999
    
    # compared to the line emission files, the order of the nH, T indices in the balance tables is switched
    lognH = dct_nH_T['lognH']
    logT  = dct_nH_T['logT']
    logionbal, lognH_tab, logTK_tab = findiontables_bensgadget2(ion,z) #(np.array([[0.,0.],[0.,1.],[0.,2.]]), np.array([0.,1.,2.]), np.array([0.,1.]) )
    NumPart = len(lognH)
    inbalance = np.zeros(NumPart, dtype=np.float32)

    if len(logT) != NumPart:
        raise ValueError('find_ionbal_bensgadget2: lognH and logT should have the same length')

    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too
    # ion balance tables are density x temperature x redshift

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong, \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK_tab)*len(lognH_tab),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognH_tab),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK_tab),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(lognH.astype(np.float32),\
               logT.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten((10**logionbal).astype(np.float32)),\
               lognH_tab.astype(np.float32),\
               ct.c_int(len(lognH_tab)),\
               logTK_tab.astype(np.float32),\
               ct.c_int(len(logTK_tab)), \
               inbalance \
              )

    print("-------------- C interpolation function output finished ----------------------\n")

    if res != 0:
        raise RuntimeError('find_ionbal_bensgadget2: Something has gone wrong in the C function: output %s. \n'%str(res))
        
    inbalance[inbalance == table_zeroequiv] = 0.
    return inbalance



def readin_radprof(infodct, yvals_label_ion,\
                   labels=None, ions_perlabel=None, ytype='perc',\
                   datadir=datadir, binset=0, units='R200c'):
    '''
    infodct:
        {label: <read-in data>}
        <read-in data>: {'filenames': {ion: filename},\
                         'setnames': names of the galsettags in the files to 
                                     match
                         'setfills': None or 
                                     {setname: string/string tuple to fill into
                                                filenames for different sets},\
                         }
         if using setfills, the fills and filenames should work with 
         string.format, kwargs
    yvals_label_ion:
        {label: {ion: yvals list}}
        yvals list: list of y values to use: percentile points or minimum 
        log10 / cm^-2 column densities
    labels: which labels to use (infodct keys; None -> all of them)
    ions_perlabel: {label: list of ions}
        ions to use for each label
    ytype: 'perc' (percentiles) or 'fcov' (covering fractions)

    returns dictionaries:
        yvals: {label: {ion: {setname: yvals array (cell indices)}}}
        bins:  {label: {ion: {setname: radial bins array (edge indices)}}}
        numgals: {label: {ion: {setname: size of galaxy sample}}}
        bin edges will have +- infinity already adjusted to finite values
    '''
    
    yvals = {}
    bins = {}
    numgals = {}
    
    readpath_base = '{un}_bins/binset_{bs}/{yt}_{val}'.format(un=units,\
                     bs=binset, yt=ytype, val='{}')
    readpath_bins = '{un}_bins/binset_{bs}/bin_edges'.format(un=units,\
                     bs=binset)
    if labels is None:
        labels = list(infodct.keys())
        
    for var in labels:
        yvals[var] = {}
        #cosmopars[var] = {}
        #fcovs[var] = {}
        #dXtot[var] = {}
        #dztot[var] = {}
        #dXtotdlogN[var] = {}
        bins[var] = {}
        numgals[var] = {}
        
        if ions_perlabel is None:
            ions = list(infodct[var]['filenames'].keys())
        else:
            ions = ions_perlabel[var]
            
        for ion in ions:
            print('Reading in data for ion {ion}'.format(ion=ion))
            filename = infodct[var]['filenames'][ion]
            goaltags = infodct[var]['setnames']
            setfills = infodct[var]['setfills']
            
            #_units   = techvars[var]['units']
            if ion not in filename:
                raise RuntimeError('File {fn} attributed to ion {ion}, mismatch'.format(fn=filename, ion=ion))
            
            if setfills is None:
                with h5py.File(datadir + filename, 'r') as fi:
                    bins[var][ion] = {}
                    yvals[var][ion] = {}
                    numgals[var][ion] = {}
                    galsets = fi.keys()
                    tags = {} 
                    
                    for galset in galsets:
                        ex = True
                        for val in yvals_label_ion[var][ion]:
                            readpath = readpath_base.format(val)
                            try:
                                np.array(fi[galset + '/' + readpath])
                            except KeyError:
                                ex = False
                                break
                        if ex:
                            tags[fi[galset].attrs['seltag'].decode()] = galset
                        
                    tags_toread = set(goaltags) &  set(tags.keys())
                    tags_unread = set(goaltags) - set(tags.keys())

                    if len(tags_unread) > 0:
                        print('For file {fn}, missed the following tags:\n\t{st}'.format(fn=filename, st=tags_unread))
                    
                    for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['{}/{}'.format(tags[tag], readpath_base.format(val))]) for val in yvals_label_ion[var][ion]}
                        numgals[var][ion][tag] = len(np.array(fi['{}/galaxyid'.format(tags[tag])]))
            else:
                bins[var][ion] = {}
                yvals[var][ion] = {}
                numgals[var][ion] = {}
                for tag in goaltags:
                    fill = setfills[tag]                    
                    #print('Using %s, %s, %s'%(var, ion, tag))
                    fn_temp = datadir + filename.format(**fill)
                    #print('For ion %s, tag %s, trying file %s'%(ion, tag, fn_temp))
                    with h5py.File(fn_temp, 'r') as fi:                       
                        galsets = fi.keys()
                        tags = {} 
                        for galset in galsets:
                            ex = True
                            for val in yvals_label_ion[var][ion]:
                                readpath = readpath_base.format(val)
                                try:
                                     np.array(fi[galset + '/' + readpath])
                                except KeyError:
                                    ex = False
                                    break
                            
                            if ex:
                                tags[fi[galset].attrs['seltag']] = galset
                            
                        #tags_toread = {tag} &  set(tags.keys())
                        tags_unread = {tag} - set(tags.keys())
                        #print(goaltags)
                        #print(tags.keys())
                        if len(tags_unread) > 0:
                            print('For file {fn}, missed the following tags:\n\t{st}'.format(fn=filename, st=tags_unread))
                        
                        #for tag in tags_toread:
                        _bins = np.array(fi[tags[tag] + '/' + readpath_bins])
                        # handle +- infinity edges for plotting; should be outside the plot range anyway
                        if _bins[0] == -np.inf:
                            _bins[0] = -100.
                        if _bins[-1] == np.inf:
                            _bins[-1] = 100.
                        bins[var][ion][tag] = _bins
                        
                        # extract number of pixels from the input filename, using naming system of make_maps
                        
                        yvals[var][ion][tag] = {val: np.array(fi['%s/%s'%(tags[tag], readpath_base.format(val))]) for val in yvals_label_ion[var][ion]}
                        numgals[var][ion][tag] = len(np.array(fi['%s/galaxyid'%(tags[tag])]))
    return yvals, bins, numgals      




###############################################################################
############################## Paper plots ####################################
###############################################################################
                     
################################ CDDFs ########################################                        
                     
# CDDFs, total, no split
def plot_cddfs_nice(fontsize=fontsize, imgname=None):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    linewidths, transparancies, and linestyles indicate different technical 
      variations in assigning pixels to halos
      
    technical variations: - incl. everything in halos (no overlap exclusion)
                          - use slices containing any part of the halo (any part of cen +/- R200c within slice)
                          - use slices containing only the whole halo (all of cen +/- R200c within slice)
                          - use halo range only projections
                          - use halo mass only projections with the masking variations
    '''
    print('Total CDDFs snapshot 27; o7 and o8 ticks offset for legibility')
    
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    techvars = [0]
    
    if imgname is None:
        imgname = 'cddfs_total_{ions}_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6p25slice_zcen-all.pdf'.format(ions='-'.join(sorted(ions)))
    if '/' not in imgname:
        imgname = mdir + imgname
    if imgname[-4:] != '.pdf':
        imgname = imgname + '.pdf'

    if isinstance(ions, str):
        ions = [ions]
    
    ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    #clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    ion_filedct_excl_1R200c_cenpos = {'fe17': datadir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  datadir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  datadir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   datadir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   datadir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   datadir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    linewidths = {0: 2}
    
    linestyles = {0: 'solid'}
    
    alphas = {0: 1.}
    
    masknames1 = ['nomask']
    masknames = masknames1 #{0: {ion: masknames1 for ion in ions}}
       
    fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.0), gridspec_kw={'wspace': 0.0})
    
    hists = {}
    cosmopars = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        dXtot[var] = {}
        dztot[var] = {}
        dXtotdlogN[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                if _bins[0] == -np.inf:
                    _bins[0] = -100.
                if _bins[-1] == np.inf:
                    _bins[-1] = 100.
                bins[var][ion] = _bins
                
                # extract number of pixels from the input filename, using naming system of make_maps
                inname = np.array(fi['input_filenames'])[0].decode()
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                numpix_1sl.remove(None)
                numpix_1sl = int(list(numpix_1sl)[0][:-3])
                print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                ionind = 1 + np.where(np.array([part == 'coldens' for part in parts]))[0][0]
                ion = parts[ionind]
                
                masks = masknames
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                
                examplemaskdir = list(fi['masks'].keys())[0]
                examplemask = list(fi['masks/%s'%(examplemaskdir)].keys())[0]
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                dXtot[var][ion] = cu.getdX(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dztot[var][ion] = cu.getdz(cosmopars[var][ion]['z'], cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'], cosmopars=cosmopars[var][ion]) * float(numpix_1sl**2)
                dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])

                        
    ax1.set_xlim(12.0, 17.)
    ax1.set_ylim(-4.05, 2.5)
    
    pu.setticks(ax1, fontsize=fontsize, labelbottom=True, labelleft=True)

    ax1.set_xlabel(xlabel, fontsize=fontsize)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
            
    for ionind in range(len(ions)):
        ion = ions[ionind]
        ax = ax1

        for vi in range(len(techvars)):
            plotx = bins[var][ion]
            plotx = plotx[:-1] + 0.5 * np.diff(plotx)
            
            ax.plot(plotx, np.log10((hists[var][ion]['nomask']) / dXtotdlogN[var][ion]), color=ioncolors[ion], linestyle=linestyles[var], linewidth=linewidths[var], alpha=alphas[var], label=ild.getnicename(ion, mathmode=False))
            
            ylim = ax.get_ylim()
            if ion == 'o8':
                vbreak = approx_breaks[ion] + 0.02
            elif ion == 'o7':
                vbreak = approx_breaks[ion] - 0.02
            else:
                vbreak = approx_breaks[ion] 
            ax.axvline(vbreak, ylim[0], 0.05 ,\
                       color=ioncolors[ion], linewidth=2., linestyle='solid')
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    leg1 = ax1.legend(handles=handles1, fontsize=fontsize, ncol=1,\
                      loc='lower left', bbox_to_anchor=(0., 0.), frameon=False)
    ax1.add_artist(leg1)
   
    plt.savefig(imgname, format='pdf', bbox_inches='tight')


def plotcddfsplits_fof():
    '''
    paper plot: FoF-only projections vs. all gas
    '''
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] #, 'hneutralssh'
    
    outname = mdir + 'split_FoF-M200c_proj_%s'%('-'.join(ions))
    outname = outname + '.pdf'
    
    medges = np.arange(11., 14.1, 0.5) #np.arange(9., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_all = {key: [datadir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] +\
                           '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'\
                           for fn in prefilenames_all[key]] \
                     for key in prefilenames_all.keys()}
    
    masses_proj = ['none'] + list(medges)
    filenames_ion = {ion: filenames_all[ion] for ion in ions}  
    filedct = {ion: {masses_proj[i]: filenames_ion[ion][i] for i in range(len(filenames_ion[ion]))} for ion in ions}
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for ion in ions:
        dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['bins/axis_0'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                        raise err
                        
                    dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['input_filenames'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    cosmopars = cosmopars_ea_27
                    
                    mmass = 'none'
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'],\
                                     cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
         
            
    ## read in all gas CDDF
    ion_filedct_excl_1R200c_cenpos = {'fe17': datadir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  datadir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  datadir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   datadir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   datadir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   datadir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    dct_totcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_totcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_totcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            cosmopars = cosmopars_ea_27
            
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'gas from haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) + [np.inf]
        
    numcols = 3
    numrows = 2
    panelwidth = 2.5
    panelheight = 2.5
    spaceh = 0.2
    legheight = 1.
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = True
        numrows_fig = numrows + 1
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols
        legindstart = 0
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numrows * panelheight + spaceh * numrows + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=spaceh, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    cbar, colors = add_cbar_mass(cax, fontsize=fontsize, aspect=9.,\
                                 clabel=clabel)
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    
    linewidth = 2.
    alpha = 1.
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        ax.set_ylim(-4.1, 2.5)
        
        labelx = yi == numrows - 1 #or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        pu.setticks(ax, fontsize=fontsize, labelbottom=True, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
       
        divby = 1. 
                    
        for pmass in masses_proj[1:]:
            _lw = linewidth
            _pe = patheff
                       
            # CDDF for projected mass, no mask
            bins = dct_fofcddf[ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[ion][pmass]['none']['cddf'] / divby),\
                    color=colors[pmass], linestyle='solid', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)
            
        _lw = linewidth
        _pe = patheff
        # total CDDF
        bins = dct_totcddf[ion]['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax.plot(plotx, np.log10(dct_totcddf[ion]['none']['cddf'] / divby),\
                color=colors['total'], linestyle='solid', alpha=alpha,\
                path_effects=_pe, linewidth=_lw)
        
        # all halo gas CDDF
        bins = dct_fofcddf[ion]['none']['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax.plot(plotx, np.log10(dct_fofcddf[ion]['none']['none']['cddf'] / divby),\
                color=colors['allhalos'], linestyle='dashed',\
                alpha=alpha, path_effects=patheff, linewidth=linewidth)

        text = r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True))
        ax.text(0.95, 0.95, text,\
                horizontalalignment='right', verticalalignment='top',\
                transform=ax.transAxes, fontsize=fontsize)            

    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid']:
        subcols = [colors[ed] for ed in massedges[:-1]] +\
                  [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
        
    sumhandles = [mlines.Line2D([], [], color=colors['total'],\
                                linestyle='solid', label='all gas',\
                                linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'],\
                                linestyle='dashed', label=r'all halo gas',\
                                linewidth=2.),\
                  ]
    sumlabels = ['all gas', r'all halo gas']
    lax.legend(lcs + sumhandles, ['halo gas'] + sumlabels,\
               handler_map={type(lc): pu.HandlerDashedLines()},\
               fontsize=fontsize, ncol=ncol_legend,\
               loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')


def plotcddfsplits_fof_zev():
    '''
    paper plot: FoF-only projections vs. all gas at z=0.1 and z=0.5
    '''
    ions = ['ne8', 'o7', 'o8']
    outname = mdir + 'split_FoF-M200c_proj_z0p1-vs-0p5_%s'%('-'.join(ions))
    outname = outname + '.pdf'
    
    medges = np.arange(11., 14.1, 0.5) #np.arange(9., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all_s27 = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_s27 = {key: [datadir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] +\
                           '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5'\
                           for fn in prefilenames_all_s27[key]] for key in prefilenames_all_s27.keys()}
        
    filenames_s23 = {key: [datadir + 'cddf_coldens_%s_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel_add-1_offset-0_resreduce-1.hdf5'%(key, halofill) \
                           for halofill in halofills]
                 for key in ions}

    
    masses_proj = ['none'] + list(medges)
    filenames_s27_ion = {ion: filenames_s27[ion] for ion in ions}  
    filedct_s27 = {ion: {masses_proj[i]: filenames_s27_ion[ion][i] for i in range(len(filenames_s27_ion[ion]))} for ion in ions}
    filenames_s23_ion = {ion: filenames_s23[ion] for ion in ions}  
    filedct_s23 = {ion: {masses_proj[i]: filenames_s23_ion[ion][i] for i in range(len(filenames_s27_ion[ion]))} for ion in ions}

    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    cosmopars_snap = {}
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    filedct = filedct_s27
    for ion in ions:
        dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['bins/axis_0'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[ion][pmass]))
                        raise err
                        
                    dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['input_filenames'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    cosmopars = cosmopars_ea_27
                    
                    # use cosmopars from the last read mask
                    mmass = 'none'
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
         
    dct_fofcddf = {27: dct_fofcddf}
    _dct_fofcddf = {}
    filedct = filedct_s23
    for ion in ions:
        _dct_fofcddf[ion] = {} 
        for pmass in masses_proj:
            _dct_fofcddf[ion][pmass] = {}
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    try:
                        bins = np.array(fi['edges'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[ion][pmass]))
                        raise err
                    if bins[0] == -np.inf:
                        bins[0] = 2. * bins[1] - bins[2]
                    if bins[-1] == np.inf:
                        bins[-2] = 2. * bins[-2] - bins[-3]
                    
                    _dct_fofcddf[ion][pmass]['bins'] = bins
                    
                    inname = np.array(fi['Header/filenames_in'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    if numpix_1sl != 32000: # expected for standard CDDFs
                        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
                    cosmopars_snap[23] = cosmopars
                    
                    mmass = 'none'
                    grp = fi
                    hist = np.array(grp['histogram'])
                    covfrac = 1. #grp.attrs['covfrac']
                    dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
                    _dct_fofcddf[ion][pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
    
    dct_fofcddf.update({23: _dct_fofcddf})
    
            
    ## read in ttotal cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': datadir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  datadir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  datadir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   datadir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   datadir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   datadir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    dct_totcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_totcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_totcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
    
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            cosmopars = cosmopars_ea_27
            # use cosmopars from the last read mask
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            dct_totcddf[ion][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    dct_totcddf = {27: dct_totcddf}
    
    totfiles_snap23 = {'o8': datadir + 'cddf_coldens_o8_L0100N1504_23_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.hdf5',\
                       'o7': datadir + 'cddf_coldens_o7_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.hdf5',\
                       'ne8': datadir + 'cddf_coldens_ne8_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.hdf5',\
                       }
    dct_totcddf[23] = {}
    for ion in totfiles_snap23:
        dct_totcddf[23][ion] = {'none': {}}
        with h5py.File(totfiles_snap23[ion], 'r') as fi:
            cosmopars = cosmopars_snap[23]
            
            covfrac = 1. #grp.attrs['covfrac']
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
            
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            hist = np.array(fi['masks_0/hist'])
            bins = np.array(fi['bins/axis_0'])
            if bins[0] == -np.inf:
                bins[0] = 2. * bins[1] - bins[2]
            if bins[-1] == np.inf:
                bins[-2] = 2. * bins[-2] - bins[-3]
                        
            dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
            
            dct_totcddf[23][ion]['bins'] = bins           
            dct_totcddf[23][ion]['none'] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'gas from haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) + [np.inf]
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    #masslabels = {name: name + 0.5 * np.average(np.diff(massedges)) for name in masses_proj[1:]}
    
    numcols = 3
    numrows = 2
    panelwidth = 2.5
    panelheight = 2.5
    spaceh = 0.0
    legheight = 1.8
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = True
        numrows_fig = numrows + 1
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols
        legindstart = 0
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numrows * panelheight + spaceh * numrows + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=spaceh, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(2 * len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    cbar, colors = add_cbar_mass(cax, fontsize=fontsize, aspect=9.,\
                                 clabel=clabel)
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    
    linewidth = 2.
    alpha = 1.
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        #yi = ionind // numcols
        ion = ions[ionind]
        ax_top = axes[ionind]
        ax_bot = axes[ionind + 3]

        if ion[0] == 'h':
            ax_top.set_xlim(12.0, 23.0)
            ax_bot.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax_top.set_xlim(12., 16.)
            ax_bot.set_xlim(12., 16.)
        elif ion == 'o7':
            ax_top.set_xlim(13.25, 17.25)
            ax_bot.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax_top.set_xlim(13., 17.)
            ax_bot.set_xlim(13., 17.)
        elif ion == 'o6':
            ax_top.set_xlim(12., 16.)
            ax_bot.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax_top.set_xlim(12.5, 16.5)
            ax_bot.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax_top.set_xlim(11.5, 15.5)
            ax_bot.set_xlim(11.5, 15.5)
            

        ax_top.set_ylim(-4.1, 2.5)
        ax_bot.set_ylim(-4.1, 2.5)
            
        #labelx = yi == numrows - 1 #or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        pu.setticks(ax_top, fontsize=fontsize, labelbottom=False, labelleft=labely)
        pu.setticks(ax_bot, fontsize=fontsize, labelbottom=True, labelleft=labely)
        ax_bot.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax_bot.set_ylabel(ylabel, fontsize=fontsize)
            ax_top.set_ylabel(ylabel, fontsize=fontsize)
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        
        divby = 1. 
                    
        for pi in range(1, len(masses_proj)):
            pmass = masses_proj[pi]
            _lw = linewidth
            _pe = patheff
            
            # CDDF for projected mass, no mask
            if bool(pi%2):
                ax = ax_top
            else:
                ax = ax_bot
            bins = dct_fofcddf[27][ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[27][ion][pmass]['none']['cddf'] / divby),\
                    color=colors[pmass], linestyle='solid', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)
            
            # snap 23
            bins = dct_fofcddf[23][ion][pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[23][ion][pmass]['none']['cddf'] / divby),\
                    color=colors[pmass], linestyle='dotted', alpha=alpha,
                    path_effects=_pe, linewidth=_lw)
            
        _lw = linewidth
        _pe = patheff
        # total CDDF
        bins = dct_totcddf[27][ion]['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_top.plot(plotx, np.log10(dct_totcddf[27][ion]['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='solid', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)
        ax_bot.plot(plotx, np.log10(dct_totcddf[27][ion]['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='solid', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)

        bins = dct_totcddf[23][ion]['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_top.plot(plotx, np.log10(dct_totcddf[23][ion]['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='dotted',\
                    alpha=alpha, path_effects=_pe, linewidth=_lw)
        ax_bot.plot(plotx, np.log10(dct_totcddf[23][ion]['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='dotted', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)
        
        # all halo gas CDDF
        bins = dct_fofcddf[27][ion]['none']['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_bot.plot(plotx, np.log10(dct_fofcddf[27][ion]['none']['none']['cddf'] / divby),\
                    color=colors['allhalos'], linestyle='dashed', alpha=alpha, path_effects=patheff,\
                    linewidth=linewidth)
        
        bins = dct_fofcddf[23][ion]['none']['bins']
        plotx = bins[:-1] + 0.5 * np.diff(bins)
        ax_bot.plot(plotx, np.log10(dct_fofcddf[23][ion]['none']['none']['cddf'] / divby),\
                    color=colors['allhalos'], linestyle='dashdot', alpha=alpha,\
                    path_effects=patheff, linewidth=linewidth)

        text = r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True))
        ax.text(0.95, 0.95, text, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize)            

    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid', 'dotted']:
        subcols = [colors[ed] for ed in massedges[:-1]] +\
                  [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)
    
    # create the legend
    sumhandles = [mlines.Line2D([], [], color=colors['total'], linestyle='solid', label='all gas, $z=0.1$', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'], linestyle='dotted', label='all gas, $z=0.5$', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='dashed', label=r'all halo gas, $z=0.1$', linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'], linestyle='dashdot', label=r'all halo gas, $z=0.5$', linewidth=2.),\
                  ]
    sumlabels = ['all gas, $z=0.1$', 'all gas, $z=0.5$',\
                 'all halo gas, $z=0.1$', 'all halo gas, $z=0.5$']
    lax.legend(lcs + sumhandles, ['halo gas, $z=0.1$', 'halo gas, $z=0.5$'] + sumlabels,\
               handler_map={type(lc): pu.HandlerDashedLines()}, fontsize=fontsize,\
               ncol=ncol_legend, loc='lower center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')

    plt.savefig(outname, format='pdf', bbox_inches='tight')

# cddfsplits, FoF, mask, appendix, split method  
def plotcddfs_fofvsmask(ion):
    '''
    Note: all haloes line with masks (brown dashed) is for all haloes with 
          M200c > 10^9 Msun, while the solid line is for all FoF+200c gas at 
          any M200c
    '''

    outname = mdir + 'split_FoF-M200c_proj_%s'%ion
    outname = outname + '.pdf'
    
    ions = ['o7', 'o8', 'o6', 'ne8', 'fe17', 'ne9']
    medges = np.arange(11., 14.1, 0.5) #np.arange(11., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}
    
    filenames_all = {key: [datadir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    
    if ion not in ions:
        raise ValueError('Ion must be one of %s'%ions)
        
    filenames_this = filenames_all[ion]
    masses_proj = ['none'] + list(medges)
    filedct = {masses_proj[i]: filenames_this[i] for i in range(len(filenames_this))} 
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  #'logM200c_Msun-10.0-10.5',\
                  #'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for pmass in masses_proj:
        dct_fofcddf[pmass] = {}
        try:
            with h5py.File(filedct[pmass]) as fi:
                try:
                    bins = np.array(fi['bins/axis_0'])
                except KeyError as err:
                    print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                    raise err
                    
                dct_fofcddf[pmass]['bins'] = bins
                
                inname = np.array(fi['input_filenames'])[0]
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                numpix_1sl.remove(None)
                numpix_1sl = int(list(numpix_1sl)[0][:-3])
                print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                for mmass in masses_proj[1:]:
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    mask_examples = {key: item for (key, item) in grp.attrs.items()}
                    del mask_examples['covfrac']
                    example_key = mask_examples.keys()[0] # 'mask_<slice center>'
                    example_mask = mask_examples[example_key] # '<dir path><mask file name>'
                    path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                    #print(path)
                    cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                    dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                    dXtotdlogN = dXtot * np.diff(bins)
        
                    dct_fofcddf[pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
                
                # use cosmopars from the last read mask
                mmass = 'none'
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
                dXtotdlogN = dXtot * np.diff(bins)
                dct_fofcddf[pmass][mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
            
        except IOError as err:
            print('Failed to read in %s; stated error:'%filedct[pmass])
            print(err)
         
            
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': datadir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  datadir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  datadir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   datadir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   datadir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   datadir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    
    file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
    dct_totcddf = {}
    with h5py.File(file_allproj) as fi:
        try:
            bins = np.array(fi['bins/axis_0'])
        except KeyError as err:
            print('While trying to load bins in file %s\n:'%(file_allproj))
            raise err
            
        dct_totcddf['bins'] = bins
        
        inname = np.array(fi['input_filenames'])[0]
        inname = inname.split('/')[-1] # throw out directory path
        parts = inname.split('_')

        numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
        numpix_1sl.remove(None)
        numpix_1sl = int(list(numpix_1sl)[0][:-3])
        print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
        
        for mmass in masses_proj[1:]:
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            mask_examples = {key: item for (key, item) in grp.attrs.items()}
            del mask_examples['covfrac']
            example_key = mask_examples.keys()[0] # 'mask_<slice center>'
            example_mask = mask_examples[example_key] # '<dir path><mask file name>'
            path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
            cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
            dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
            dXtotdlogN = dXtot * np.diff(bins)
        
            dct_totcddf[mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
        # use cosmopars from the last read mask
        mmass = 'none'
        grp = fi[maskdct[mmass]]
        hist = np.array(grp['hist'])
        covfrac = grp.attrs['covfrac']
        # recover cosmopars:
        dXtot = cu.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) * float(numpix_1sl**2)
        dXtotdlogN = dXtot * np.diff(bins)
        dct_totcddf[mmass] = {'cddf': hist / dXtotdlogN, 'covfrac': covfrac}
    
    ylabel = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} \mathrm{N} \, \partial X \right)$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    clabel = r'masks for haloes with $\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    massedges = list(medges) 
    
    numcols = 3
    numrows = 3
    panelwidth = 2.5
    panelheight = 2.
    legheight = 1.
    #fcovticklen = 0.035
    if numcols * numrows - len(massedges) - 2 >= 2: # put legend in lower right corner of panel region
        seplegend = False
        legindstart = numcols - (numcols * numrows - len(medges) - 2)
        legheight = 0.
        numrows_fig = numrows
        ncol_legend = (numcols - legindstart) - 1
        height_ratios=[panelheight] * numrows 
    else:
        seplegend = True
        numrows_fig = numrows + 1
        legindstart = 0
        height_ratios=[panelheight] * numrows + [legheight]
        ncol_legend = numcols
    
    figwidth = numcols * panelwidth + 0.6 
    figheight = numcols * panelheight + 0.2 * numcols + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows_fig, numcols + 1, hspace=0.0, wspace=0.0, width_ratios=[panelwidth] * numcols + [0.6], height_ratios=height_ratios)
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(masses_proj) + 1)]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if seplegend:
        lax  = fig.add_subplot(grid[numrows, :])
    else:
        lax = fig.add_subplot(grid[numrows - 1, legindstart:])
    
    
    
    cbar, colors = add_cbar_mass(cax=cax, clabel=clabel, fontsize=fontsize,\
                                 aspect=9.)
    colors['none'] = 'gray' # add no mask label for plotting purposes
    colors['total'] = 'black'
    colors['allhalos'] = 'brown'
    linewidth = 2.
    alpha = 1.
    
    for massind in range(len(masses_proj) + 1):
        xi = massind % numcols
        yi = massind // numcols
        if massind == 0:
            pmass = masses_proj[massind]
        elif massind == 1:
            pmass = 'all halos'
        else:
            pmass = masses_proj[massind - 1]
        ax = axes[massind]

        if ion[0] == 'h':
            ax.set_xlim(12.0, 23.0)
        elif ion == 'fe17':
            ax.set_xlim(12., 16.)
        elif ion == 'o7':
            ax.set_xlim(13.25, 17.25)
        elif ion == 'o8':
            ax.set_xlim(13., 17.)
        elif ion == 'o6':
            ax.set_xlim(12., 16.)
        elif ion == 'ne9':
            ax.set_xlim(12.5, 16.5)
        elif ion == 'ne8':
            ax.set_xlim(11.5, 15.5)
            
        ax.set_ylim(-6.0, 2.5)
        
        labelx = yi == numrows - 1 or (yi == numrows - 2 and numcols * yi + xi > len(masses_proj) + 1) 
        labely = xi == 0
        pu.setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=labely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely and yi == 1: 
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
        patheff_thick = [mppe.Stroke(linewidth=linewidth + 1.0, foreground="b"), mppe.Stroke(linewidth=linewidth + 0.5, foreground="w"), mppe.Normal()]
        
        if pmass == 'none':
            ptext = 'mask split'
            divby = 1. 
        
            for pmass in masses_proj[1:]:
                _lw = linewidth
                _pe = patheff
                
                bins = dct_totcddf['bins']
                plotx = bins[:-1] + 0.5 * np.diff(bins)
                ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby),\
                        color=colors[pmass], linestyle='dashed', alpha=alpha,\
                        path_effects=_pe, linewidth=_lw)
                
            _lw = linewidth
            _pe = patheff
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='solid', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)

        elif pmass == 'all halos':
            divby = 1. 

            bins = dct_fofcddf['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf['none']['none']['cddf'] / divby),\
                    color=colors['none'], linestyle='solid', alpha=alpha,\
                    path_effects=patheff, linewidth=linewidth)
            
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(np.sum([dct_totcddf[mass]['cddf'] for mass in masses_proj[1:]], axis=0) / divby),\
                    color=colors['allhalos'], linestyle='dashed', alpha=alpha,\
                    path_effects=patheff_thick, linewidth=linewidth + 0.5)
            
            bins = dct_fofcddf['none']['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(np.sum([dct_fofcddf['none'][mass]['cddf'] for mass in masses_proj[1:]], axis=0) / divby),\
                    color=colors['allhalos'], linestyle='solid', alpha=alpha,\
                    path_effects=patheff_thick, linewidth=linewidth + 0.5)
        
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='solid', alpha=alpha,\
                    path_effects=patheff, linewidth=linewidth)
            
        else:
            if pmass == 14.0:
                ptext = r'$ > %.1f$'%pmass # \log_{10} \, \mathrm{M}_{\mathrm{200c}} \, / \, \mathrm{M}_{\odot}
            else:
                ptext = r'$ %.1f \emdash %.1f$'%(pmass, pmass + 0.5) # \leq \log_{10} \, \mathrm{M}_{\mathrm{200c}} \, / \, \mathrm{M}_{\odot} <
            divby = 1.
            
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby), color=colors[pmass],\
                    linestyle='dashed', alpha=alpha, path_effects=patheff)
                
            mmass = pmass
            _pe = patheff
            _lw = linewidth
            
            bins = dct_fofcddf[pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[pmass][mmass]['cddf'] / divby),\
                    color=colors[mmass], linestyle='solid', alpha=alpha,\
                    path_effects=_pe, linewidth=_lw)
            
            bins = dct_fofcddf[pmass]['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_fofcddf[pmass]['none']['cddf'] / divby),\
                    color=colors['none'], linestyle='solid',\
                    alpha=alpha, path_effects=patheff)
        
            bins = dct_totcddf['bins']
            plotx = bins[:-1] + 0.5 * np.diff(bins)
            ax.plot(plotx, np.log10(dct_totcddf[pmass]['cddf'] / divby),\
                    color=colors[pmass], linestyle='dashed', alpha=alpha,\
                    path_effects=patheff_thick, linewidth=linewidth + 0.5)
            ax.plot(plotx, np.log10(dct_totcddf['none']['cddf'] / divby),\
                    color=colors['total'], linestyle='solid', alpha=alpha,\
                    path_effects=patheff, linewidth=linewidth)
            
        ax.text(0.97, 0.97, ptext,\
                horizontalalignment='right', verticalalignment='top',\
                fontsize=fontsize, transform=ax.transAxes)
            
    lcs = []
    line = [[(0, 0)]]
    
    # set up the proxy artist
    for ls in ['solid', 'dashed']:
        subcols = [colors[ed] for ed in massedges] + \
        [mpl.colors.to_rgba(colors['allhalos'], alpha=alpha)]
        subcols = np.array(subcols)
        subcols[:, 3] = 1. # alpha value
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=ls, linewidth=linewidth, colors=subcols)
        lcs.append(lc)

    sumhandles = [mlines.Line2D([], [], color=colors['none'],\
                                linestyle='solid', label='FoF no mask',\
                                linewidth=2.),\
                  mlines.Line2D([], [], color=colors['total'],\
                                linestyle='solid', label='total',\
                                linewidth=2.),\
                  mlines.Line2D([], [], color=colors['allhalos'],\
                                linestyle='solid', label=r'all FoF+200c gas',\
                                linewidth=2.),\
                  ]
    sumlabels = ['FoF+200c, no mask', 'all gas, no mask', r'mask: all haloes $> 11.0$']
    lax.legend(lcs + sumhandles, ['FoF+200c, with mask', 'all gas, with mask'] + sumlabels,\
               handler_map={type(lc): pu.HandlerDashedLines()}, fontsize=fontsize,\
               ncol=ncol_legend, loc='upper center', bbox_to_anchor=(0.5, 0.))
    lax.axis('off')
   
    plt.savefig(outname, format='pdf')

############################## 2d profiles ####################################

# Halo mass, 2dprof, 2d profiles, R200c, virial radius, normalized,
# median, scatter, column density profles
def plot_radprof_limited(fontsize=fontsize):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
    '''
    techvars_touse=[0, 7]
    units='R200c'
    ytype='perc'
    yvals_toplot=[10., 50., 90.]
    plotcrit = {0: None,\
                7: {'o6':   {'Mmin': [11., 12.,  14.]},\
                    'o7':   {'Mmin': [11., 12.5, 14.]},\
                    'o8':   {'Mmin': [11., 13.,  14.]},\
                    'ne8':  {'Mmin': [11., 12.,  14.]},\
                    'ne9':  {'Mmin': [11., 13.,  14.]},\
                    'fe17': {'Mmin': [11., 13.,  14.]},\
                    },\
                }
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    
    imgname = 'radprof_byhalomass_%s_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6p25slice_zcen-all_techvars-%s_units-%s_%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars_touse])), units, ytype)
    imgname = mdir + imgname
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    shading_alpha = 0.45 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    if units == 'pkpc':
        xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    elif units == 'R200c':
        xlabel = r'$r_{\perp} \; [\mathrm{R}_{\mathrm{200c}}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_1sl = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       }
   
    ion_filedct_1sl_binfofonly = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_{hs}_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_{hs}_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_{hs}_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o6':   'rdist_coldens_o6_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_{hs}_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o7':   'rdist_coldens_o7_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_{hs}_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_halosel_{hs}_allinR200c_endhalosel_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                           }
        
    # define used mass ranges
    Mh_edges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    Mh_names_1sl_binfofonly = ['geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
                              'geq%s'%(Mh_mins[i])\
                              for i in range(len(Mh_mins))]

    galsetnames_massonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    galsetnames_1sl_binfofonly = {name: sel for name, sel in zip(Mh_names_1sl_binfofonly, Mh_sels)}
    
    fills_filedct_fofonly = {Mh_names_1sl_binfofonly[i]: {'hs': 'Mhalo_%.1f<=log200c<%.1f'%(Mh_mins[i], Mh_maxs[i])\
                                                                if Mh_maxs[i] is not None else \
                                                                'Mhalo_%.1f<=log200c'%(Mh_mins[i])} \
                             for i in range(len(Mh_mins))}
    
    techvars = {0: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                7: {'filenames': ion_filedct_1sl_binfofonly, 'setnames': galsetnames_1sl_binfofonly.keys(), 'setfills': fills_filedct_fofonly},\
                }
    
    linewidths = {0: 1.5,\
                  7: 1.,\
                  }       
    linestyles = {0: 'solid',\
                  7: 'dashed',\
                  }   
    alphas = {0: 1.,\
              7: 1.}   
    legendnames_techvars = {0: 'all gas',\
                            7: r'FoF gas only',\
                            }
    panelwidth = 2.5
    panelheight = 2.
    legheight = 0.9
    cwidth = 0.6
    if ytype == 'perc':
        wspace = 0.2
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + cwidth + wspace * numcols
    figheight = numcols * panelheight + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace, width_ratios=[panelwidth] * numcols + [cwidth], height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if len(techvars_touse) > 1:
        lax  = fig.add_subplot(grid[numrows, :])
    
    yvals_label_ion = {0: {ion: yvals_toplot for ion in ions},\
                       7: {ion: [yvals_toplot[1]] for ion in ions},\
                       }
    yvals, bins, numgals = readin_radprof(techvars, yvals_label_ion,\
                   labels=None, ions_perlabel=None, ytype=ytype,\
                   datadir=datadir, binset=0, units=units)
    
        
    cbar, colors = add_cbar_mass(cax=cax, fontsize=fontsize, clabel=clabel,\
                                 aspect=9.)
        
    masslabels1 = {name: tuple(np.log10(np.array(galsetnames_massonly[name][1:])))\
                   for name in galsetnames_massonly.keys()}
    masslabels2 = {name: tuple(np.log10(np.array(galsetnames_1sl_binfofonly[name][1:])))\
                   for name in galsetnames_1sl_binfofonly.keys()}
    masslabels_all = masslabels1
    masslabels_all.update(masslabels2)
    masslabels_all = {key: masslabels_all[key][0] for key in masslabels_all}
    massedges = sorted([masslabels_all[key] for key in masslabels_all])
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]

        if ytype == 'perc':
            if ion[0] == 'h':
                ax.set_ylim(12.0, 21.0)
            #else:
            #    ax.set_ylim(11.5, 17.0)
            elif ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o6':
                ax.set_ylim(10.8, 14.8)
            elif ion == 'ne9':
                ax.set_ylim(11.9, 15.9)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(0., 1.)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        pu.setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        

        ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        
        #hatchind = 0
        for vi in range(len(techvars_touse) -1, -1, -1): # backwards to get lowest techvars on top
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                if plotcrit[var] is not None:
                    mlist = plotcrit[var][ion]['Mmin']
                    match = np.any(np.isclose(masslabels_all[tag], np.array(mlist)))
                    if not match:
                        continue
                    
                if var == 0:
                    matchval = \
                        12.0 if ion == 'o6' else \
                        12.0 if ion == 'ne8' else \
                        12.5 if ion == 'o7' else \
                        13.0 if ion == 'ne9' else \
                        13.0 if ion == 'o8' else \
                        13.0 if ion == 'fe17' else \
                        np.inf 
                    matched = np.isclose(masslabels_all[tag], matchval)
                    if matched:
                        yvals_toplot_temp = yvals_label_ion[var][ion]
                    else:
                        yvals_toplot_temp = [yvals_label_ion[var][ion][0]] \
                                            if len(yvals_label_ion[var][ion]) == 1 else\
                                            [yvals_label_ion[var][ion][1]]
                else:
                    yvals_toplot_temp = yvals_label_ion[var][ion]
                                
                if len(yvals_toplot_temp) == 3:
                    yval = yvals_toplot_temp[0]
                    try:                      
                        ploty1 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval)) 
                    yval = yvals_toplot_temp[2]
                    try:                      
                        ploty2 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval)) 
                    ax.fill_between(plotx, ploty1, ploty2,\
                                    color=colors[masslabels_all[tag]],\
                                    alpha=alphas[var] * shading_alpha,\
                                    label=masslabels_all[tag])
                    yvals_toplot_temp = [yvals_toplot_temp[1]]
                    
                if len(yvals_toplot_temp) == 1:
                    yval = yvals_toplot_temp[0]
                    try:
                        ploty = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval))
                        continue
                    if yval == 50.0: # only highlight the medians
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"),\
                                   mppe.Stroke(linewidth=linewidths[var], foreground="w"),\
                                   mppe.Normal()]
                    else:
                        patheff = []
                    ax.plot(plotx, ploty,\
                            color=colors[masslabels_all[tag]],\
                            linestyle=linestyles[var],\
                            linewidth=linewidths[var], alpha=alphas[var],\
                            label=masslabels_all[tag], path_effects=patheff)
                
        
        if ytype == 'perc':
            ax.axhline(approx_breaks[ion], 0., 0.1, color='gray', linewidth=1.5, zorder=-1) # ioncolors[ion]
        ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = list(colors[ed] for ed in massedges) #+ [mpl.colors.to_rgba(sumcolor, alpha=alphas[var])]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[var],\
                                 linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    # create the legend
    if len(techvars_touse) > 1:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse],\
                   handler_map={type(lc): pu.HandlerDashedLines()},\
                   fontsize=fontsize, ncol=2 * numcols,\
                   loc='lower center', bbox_to_anchor=(0.5, 0.))
        lax.axis('off')
    plt.savefig(imgname, format='pdf', bbox_inches='tight')

# Mstar, stellar mass radial profiles, 2d profiles, impact parameters
# percentiles, covering fractions, CDDF breaks, detection limits, Athena X-IFU
# detectablility, M*
def plot_radprof_mstar(var='main', fontsize=fontsize, lowmass=False):
    '''
    var:        main: percenitles in M* bins
                main-fcov-break for covering fractions at the CDDF break
                main-fcov-obs   for covering fractions at obs. limits   
    lowmass:    include M* < 10^9 Msun galaxies
                using this option will produce some errors, since the masses 
                are split over two files, and the read-in function will try to 
                find all masses in both
    '''
    xlim = None
    units = 'pkpc'
    if var == 'main':
        highlightcrit = 'maxcol'
        ytype='perc'
        yvals_toplot=[10., 50., 90.]
    elif var == 'main-fcov-break':
        highlightcrit = None
        ytype='fcov'
        yvals_toplot = {'o6':   [14.3],\
                        'ne8':  [13.7],\
                        'o7':   [16.0],\
                        'ne9':  [15.3],\
                        'o8':   [16.0],\
                        'fe17': [15.0]}
    elif var == 'main-fcov-obs':
        highlightcrit = None
        ytype='fcov'
        yvals_toplot = {'o6':   [13.5],\
                        'ne8':  [13.5],\
                        'o7':   [15.5],\
                        'ne9':  [15.5],\
                        'o8':   [15.7],\
                        'fe17': [14.9]}
    if lowmass:
        techvars_touse = [0, 1]
    else:
        techvars_touse = [0]
    
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    if lowmass:
        slm = '_incl-lowmass'
    else:
        slm = ''
    
    imgname = 'radprof_bystellarmass_L0100N1504_27_PtAb_C2Sm_32000pix_T4EOS_6p25slice_zcen-all_{var}{lm}.pdf'.format(var=var, lm=slm)
    imgname = mdir + imgname        
    print('Using y value selection (log10 Nmin/cm2 or percentiles): {}'.format(yvals_toplot))
    
    numcols = 3
    numrows = (len(ions) - 1) // numcols + 1
    
    cmapname = 'rainbow'
    shading_alpha = 0.45 
    if ytype == 'perc':
        ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    elif ytype == 'fcov':
        ylabel = 'covering fraction'
    xlabel = r'$r_{\perp} \; [\mathrm{pkpc}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\star} \; [\mathrm{M}_{\odot}]$'
    linestyles_fcov = ['solid', 'dashed', 'dotted', 'dotdash']

    ion_filedct_Mstar = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         }
    ion_filedct_Mstar_lowmass = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         }

    # define used mass ranges
    Ms_edges = np.array([9.0, 9.5, 10., 10.5, 11., 11.7])
    Ms_mins = list(Ms_edges[:-1])
    Ms_maxs = list(Ms_edges[1:]) 
    #Ms_base = ['geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_names = ['logMstar_Msun_1000_geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_sels = [('logMstar_Msun', Ms_mins[i], Ms_maxs[i]) if Ms_maxs[i] is not None else\
               ('logMstar_Msun', Ms_mins[i], np.inf)\
               for i in range(len(Ms_mins))]
    galsetnames_smass = {name: sel for name, sel in zip(Ms_names, Ms_sels)}  
    
    techvars = {0: {'filenames': ion_filedct_Mstar, 'setnames': Ms_names, 'setfills': None},\
                }
    if lowmass:
        Ms_edges_lm = np.array([8.0, 8.5, 9.0])
        Ms_mins_lm = list(Ms_edges_lm[:-1])
        Ms_maxs_lm = list(Ms_edges_lm[1:]) 
        #Ms_base = ['geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
        Ms_names_lm = ['logMstar_Msun_1000_geq%.1f_le%.1f'%(smin, smax)\
                       for smin, smax in zip(Ms_mins_lm, Ms_maxs_lm)]
        Ms_sels_lm = [('logMstar_Msun', Ms_mins_lm[i], Ms_maxs_lm[i]) 
                        if Ms_maxs_lm[i] is not None else\
                      ('logMstar_Msun', Ms_mins_lm[i], np.inf)\
                      for i in range(len(Ms_mins_lm))]
        galsetnames_smass_lm = {name: sel for name, sel in zip(Ms_names_lm, Ms_sels_lm)}
        
        galsetnames_smass.update(galsetnames_smass_lm)
        
        Ms_edges = np.append(Ms_edges_lm[:-1], Ms_edges)
        techvars.update({1: {'filenames': ion_filedct_Mstar_lowmass,\
                             'setnames': Ms_names_lm, 'setfills': None}})
       
    linewidths = {0: 2.,\
                  1: 2.}     
    linestyles = {0: 'solid',\
                  1: 'solid'}    
    alphas = {0: 1., 1: 1.}
    
    legendnames_techvars = {0: r'$\mathrm{M}_{\star}$, pkpc-stacked',\
                            1: r'$\mathrm{M}_{\star}$, pkpc-stacked',\
                            }
    
    if not isinstance(yvals_toplot, dict):
        yvals_toplot = {ion: yvals_toplot for ion in ion_filedct_Mstar}
    
    yvals, bins, numgals = readin_radprof(techvars,\
                           {tv: yvals_toplot for tv in techvars},\
                           labels=None, ions_perlabel=None, ytype=ytype,\
                           datadir=datadir, binset=0, units=units)
  
    addlegend = (len(techvars_touse) > 1 and not lowmass) \
                or len(techvars_touse) > 2
    panelwidth = 2.5
    panelheight = 2.
    if addlegend:
        legheight = 1.3
    else:
        legheight = 0.0
    cwidth = 0.6
    if ytype == 'perc':
        wspace = 0.2
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + cwidth + wspace * numcols
    figheight = numcols * panelheight + legheight
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(numrows + 1, numcols + 1, hspace=0.0, wspace=wspace,\
                        width_ratios=[panelwidth] * numcols + [cwidth],\
                        height_ratios=[panelheight] * numrows + [legheight])
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(len(ions))]
    cax  = fig.add_subplot(grid[:numrows, numcols])
    if addlegend:
        lax  = fig.add_subplot(grid[numrows, :])
    
  
    cbar, colors = add_cbar_mass(cax, cmapname=cmapname,\
                                   massedges=Ms_edges[:-1],\
                                   orientation='vertical', clabel=clabel,\
                                   fontsize=fontsize, aspect=9.)
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        ax = axes[ionind]
        if isinstance(yvals_toplot, dict):
            _yvals_toplot = yvals_toplot[ion]
        else:
            _yvals_toplot = yvals_toplot

        if ytype == 'perc':
            if ion == 'fe17':
                ax.set_ylim(11.5, 15.5)
            elif ion == 'o7':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o8':
                ax.set_ylim(12.5, 16.5)
            elif ion == 'o6':
                ax.set_ylim(10.8, 14.8)
            elif ion == 'ne9':
                ax.set_ylim(11.9, 15.9)
            elif ion == 'ne8':
                ax.set_ylim(10.5, 14.5)
        
        elif ytype == 'fcov':
            ax.set_ylim(10**-1.95, 10**0.3)
            ax.set_yscale('log')
        
        if xlim is not None:
            ax.set_xlim(*xlim)
        
        labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
        labely = xi == 0
        if wspace == 0.0:
            ticklabely = xi == 0
        else:
            ticklabely = True
        pu.setticks(ax, fontsize=fontsize, labelbottom=labelx, labelleft=ticklabely)
        if labelx:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        
        if ytype == 'perc':
            ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False),\
                    horizontalalignment='right', verticalalignment='top',\
                    fontsize=fontsize, transform=ax.transAxes)
        else:
            ax.text(0.05, 0.95, ild.getnicename(ion, mathmode=False),\
                    horizontalalignment='left', verticalalignment='top',\
                    fontsize=fontsize, transform=ax.transAxes)
        #hatchind = 0
        for vi in range(len(techvars_touse)):
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=galsetnames_smass.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                mkey = galsetnames_smass[tag][1]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                # adjust bins based on log scale, not lin
                plotx = np.log10(plotx)
                plotx[0] = 2. * plotx[1] - plotx[2] # innermost value is 0 -> -np.inf
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)
                plotx = 10**plotx
                
                if highlightcrit == 'maxcol': #highlightcrit={'techvars': [0], 'Mmin': [10.0, 12.0, 14.0]}
                    Mmin_tomatch = \
                        10.5 if ion == 'o6' else \
                        10.5 if ion == 'ne8' else \
                        10.5 if ion == 'o7' else \
                        11.0 if ion == 'ne9' else \
                        11.0 if ion == 'o8' else \
                        11.0 if ion == 'fe17' else \
                        np.inf                     
                    matched = np.min(np.abs(galsetnames_smass[tag][1] - Mmin_tomatch)) <= 0.01
                    if matched:
                        yvals_toplot_temp = _yvals_toplot
                    else:
                        yvals_toplot_temp = [_yvals_toplot[0]] if len(_yvals_toplot) == 1 else [_yvals_toplot[1]]
                else:
                    yvals_toplot_temp = _yvals_toplot
                
                
                if ytype == 'perc':
                    if len(yvals_toplot_temp) == 3:
                        yval = yvals_toplot_temp[0]
                        try:                      
                            ploty1 = yvals[var][ion][tag][yval]
                        except KeyError:
                            print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval)) 
                        yval = yvals_toplot_temp[2]
                        try:                      
                            ploty2 = yvals[var][ion][tag][yval]
                        except KeyError:
                            print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval))                         
                        ax.fill_between(plotx, ploty1, ploty2, color=colors[mkey],\
                                        alpha=alphas[var] * shading_alpha,\
                                        label=galsetnames_smass[tag])
                        
                        yvals_toplot_temp = [yvals_toplot_temp[1]]
                        
                    if len(yvals_toplot_temp) == 1:
                        yval = yvals_toplot_temp[0]
                        try:
                            ploty = yvals[var][ion][tag][yval]
                        except KeyError:
                            print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval))
                            continue
                        if yval == 50.0: # only highlight the medians
                            patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"),\
                                       mppe.Stroke(linewidth=linewidths[var], foreground="w"),\
                                       mppe.Normal()]
                        else:
                            patheff = []
                        ax.plot(plotx, ploty, color=colors[mkey],\
                                linestyle=linestyles[var],\
                                linewidth=linewidths[var], alpha=alphas[var],\
                                label=galsetnames_smass[tag],\
                                path_effects=patheff)

                elif ytype == 'fcov':
                    for yi in range(len(yvals_toplot_temp)):
                        linestyle = linestyles_fcov[yi]
                        yval = yvals_toplot_temp[yi]
                        ploty = yvals[var][ion][tag][yval]
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"),\
                                   mppe.Stroke(linewidth=linewidths[var], foreground="w"),\
                                   mppe.Normal()]
                        ax.plot(plotx, ploty, color=colors[mkey],\
                                linestyle=linestyle,\
                                linewidth=linewidths[var], alpha=alphas[var],\
                                label=galsetnames_smass[tag],\
                                path_effects=patheff)
                                           
        if ytype == 'perc':
            ax.axhline(approx_breaks[ion], 0., 0.1, color='gray',\
                       linewidth=1.5, zorder=-1) 
        if ytype == 'fcov': # add value labels
            linewidth = np.average([linewidths[var] for var in techvars_touse])
            alpha = np.average([alphas[var] for var in techvars_touse])
            patheff = [] #[mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"), mppe.Stroke(linewidth=linewidths[var], foreground="w"), mppe.Normal()]
            color = 'gray'
            if len(yvals_toplot_temp) > 1:
                legend_handles = [mlines.Line2D([], [], linewidth=linewidth,\
                                                alpha=alpha, color=color,\
                                                path_effects=patheff,\
                                                label='%.1f'%yvals_toplot_temp[yi],\
                                                linestyle=linestyles_fcov[yi]) \
                                  for yi in range(len(yvals_toplot_temp))]
                ax.legend(handles=legend_handles, fontsize=fontsize - 1,\
                          loc='upper right', bbox_to_anchor=(1.02, 1.02),\
                          columnspacing=1.2, handletextpad=0.5,\
                          frameon=False, ncol=2)
            else:
                panellabel = '$ > %.1f$'%yvals_toplot_temp[yi]
                ax.text(0.95, 0.95, panellabel, fontsize=fontsize, \
                        verticalalignment='top', horizontalalignment='right',\
                        transform=ax.transAxes)
        ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = [colors[mkey] for mkey in \
                   sorted([galsetnames_smass[tag][1] for tag in tags])] # tags is sorted
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        lc = mcol.LineCollection(line * len(subcols),\
                                 linestyle=linestyles[var], \
                                 linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
        
    if addlegend:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse],\
                   handler_map={type(lc): pu.HandlerDashedLines()},\
                   fontsize=fontsize, ncol=2, loc='lower center',\
                   bbox_to_anchor=(0.5, 0.))
        lax.axis('off')

    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    
    
def plot_radprof_zev(fontsize=fontsize):
    '''
    ions in different panels
    colors indicate different halo masses (from a rainbow color bar)
      
    techvars: 0 = 1 slice, z=0.1, all gas, all haloes
              8 = 1 slice, z=0.1, all gas, all haloes
    '''
    techvars_touse = [0, 8]
    units = 'R200c'
    ytype = 'perc'
    yvals_toplot = [10., 50., 90.]
    ions = ['o7', 'o8', 'ne8'] #['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    
    imgname = 'radprof_byhalomass_%s_L0100N1504_27-23_PtAb_C2Sm_32000pix_T4EOS_6p25slice_zcen-all_techvars-%s_units-%s_%s.pdf'%('-'.join(sorted(ions)), '-'.join(sorted([str(var) for var in techvars_touse])), units, ytype)
    imgname = mdir + imgname
        
    if isinstance(ions, str):
        ions = [ions]
    if len(ions) <= 3:
        numrows = 1
        numcols = len(ions)
    elif len(ions) == 4:
        numrows = 2
        numcols = 2
    else:
        numrows = (len(ions) - 1) // 3 + 1
        numcols = 3
    
    shading_alpha = 0.45 
    ylabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    xlabel = r'$r_{\perp} \; [\mathrm{R}_{\mathrm{200c}}]$'
    clabel = r'$\log_{10}\, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    # up to 2.5 Rvir / 500 pkpc
    ion_filedct_1sl = {#'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       #'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o8':   'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       #'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                       }
    
    ion_filedct_1sl_snap23 = {'ne8': 'rdist_coldens_ne8_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                              'o7': 'rdist_coldens_o7_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EO_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                              'o8': 'rdist_coldens_o8_L0100N1504_23_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EO_1slice_to-100-pkpc-or-3-R200c_M200c-0p5dex-7000_centrals_stored_profiles.hdf5',\
                              }
        
    # define used mass ranges
    Mh_edges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.]) # 9., 9.5, 10., 10.5
    Mh_mins = list(Mh_edges)
    Mh_maxs = list(Mh_edges[1:]) + [None]
    Mh_sels = [('M200c_Msun', 10**Mh_mins[i], 10**Mh_maxs[i]) if Mh_maxs[i] is not None else\
               ('M200c_Msun', 10**Mh_mins[i], np.inf)\
               for i in range(len(Mh_mins))]
    # different runs of galaxy mass ranges are named differently.
    # variable names are just examples of where those names are applied
    # matched to attributes in the hdf5 files, since group names are just 
    # numbered
    Mh_names =['logM200c_Msun_geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
               'logM200c_Msun_geq%s'%(Mh_mins[i])\
               for i in range(len(Mh_mins))]
    Mh_names_1sl_binfofonly = ['geq%s_le%s'%(Mh_mins[i], Mh_maxs[i]) if Mh_maxs[i] is not None else\
                              'geq%s'%(Mh_mins[i])\
                              for i in range(len(Mh_mins))]

    galsetnames_massonly = {name: sel for name, sel in zip(Mh_names, Mh_sels)}
    galsetnames_1sl_binfofonly = {name: sel for name, sel in zip(Mh_names_1sl_binfofonly, Mh_sels)}
    
    masslabels_all = {name: tuple(np.log10(np.array(galsetnames_massonly[name][1:])))\
                      for name in galsetnames_massonly.keys()}
    masslabels_all.update({name: tuple(np.log10(np.array(galsetnames_1sl_binfofonly[name][1:]))) \
                           for name in galsetnames_1sl_binfofonly.keys()})
        
    techvars = {0: {'filenames': ion_filedct_1sl, 'setnames': galsetnames_massonly.keys(), 'setfills': None},\
                8: {'filenames': ion_filedct_1sl_snap23, 'setnames': galsetnames_1sl_binfofonly.keys(), 'setfills': None},\
                }
    
    linewidths = {0: 2.,\
                  8: 2.,\
                  }
       
    linestyles = {0: 'solid',\
                  8: 'dotted',\
                  }
    
    alphas = {0: 1.,\
              8: 1.}
    
    legendnames_techvars = {0: 'all gas, z=0.1',\
                            8: 'all gas, z=0.5'}

    panelwidth = 2.5
    panelheight = 2.5
    legheight = 1.3
    cwidth = 0.6
    if ytype == 'perc':
        wspace = 0.22
    else:
        wspace = 0.0
    #fcovticklen = 0.035
    figwidth = numcols * panelwidth + cwidth + wspace * numcols
    figheight = 2 * numrows * panelheight + legheight
    #print('{}, {}'.format(figwidth, figheight))
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(2 * numrows + 1, numcols + 1, hspace=0.0,\
                        wspace=wspace,\
                        width_ratios=[panelwidth] * numcols + [cwidth],\
                        height_ratios=[panelheight] * numrows * 2 + [legheight],\
                        bottom=0.05)
    axes = [[fig.add_subplot(grid[(i // numcols) * 2 + j, i % numcols]) for j in range(2)]for i in range(len(ions))]
    cax  = fig.add_subplot(grid[: 2 * numrows, numcols])
    if len(techvars_touse) > 1:
        lax  = fig.add_subplot(grid[2 * numrows, :])
    yvals = {}
    bins = {}
    numgals = {}
    
    yvals, bins, numgals = \
        readin_radprof(techvars, {label: {ion: yvals_toplot \
                                      for ion in ions} for label in techvars},\
                   labels=None, ions_perlabel=None, ytype='perc',\
                   datadir=datadir, binset=0, units='R200c')
        
    massranges = [sel[1:] for sel in Mh_sels]
    massedges = sorted(list(set([np.log10(val) for rng in massranges for val in rng])))
    if massedges[-1] == np.inf: # used for setting the color bar -> just need some dummy value higher than the last one
        massedges[-1] = 2. * massedges[-2] - massedges[-3]
    
    cbar, colordct = add_cbar_mass(cax, cmapname='rainbow',
                                  orientation='vertical', clabel=clabel,\
                                  fontsize=fontsize, aspect=9.)
    
    for ionind in range(len(ions)):
        xi = ionind % numcols
        yi = ionind // numcols
        ion = ions[ionind]
        _axes = axes[ionind]
        
        for axi in range(2):
            ax = _axes[axi]
            if ytype == 'perc':
                if ion == 'fe17':
                    ax.set_ylim(11.5, 15.5)
                elif ion == 'o7':
                    ax.set_ylim(12.5, 16.5)
                elif ion == 'o8':
                    ax.set_ylim(12.5, 16.5)
                elif ion == 'o6':
                    ax.set_ylim(10.8, 14.8)
                elif ion == 'ne9':
                    ax.set_ylim(11.9, 15.9)
                elif ion == 'ne8':
                    ax.set_ylim(10.5, 14.5)
            
            elif ytype == 'fcov':
                ax.set_ylim(0., 1.)
            
            labelx = (yi == numrows - 1 or (yi == numrows - 2 and (yi + 1) * numcols + xi > len(ions) - 1)) # bottom plot in column
            labelx = labelx and axi == 1
            labely = xi == 0
            if wspace == 0.0:
                ticklabely = xi == 0
            else:
                ticklabely = True
            pu.setticks(ax, fontsize=fontsize, labelbottom=labelx,\
                        labelleft=ticklabely)
            if labelx:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            if labely:
                ax.set_ylabel(ylabel, fontsize=fontsize)
            
            ax.text(0.95, 0.95, ild.getnicename(ion, mathmode=False), horizontalalignment='right', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
        

        for vi in range(len(techvars_touse) -1, -1, -1): # backwards to get lowest techvars on top
            tags = techvars[techvars_touse[vi]]['setnames']
            tags = sorted(tags, key=masslabels_all.__getitem__)
            var = techvars_touse[vi]
            for ti in range(len(tags)):
                tag = tags[ti]
                si  = ti % 2
                ax = _axes[si]
                color = colordct[np.round(masslabels_all[tag][0], 1)]
                
                try:
                    plotx = bins[var][ion][tag]
                except KeyError: # dataset not read in
                    print('Could not find techvars %i, ion %s, tag %s'%(var, ion, tag))
                    continue
                plotx = plotx[:-1] + 0.5 * np.diff(plotx)

                # decide whther to plot the 80% range:
                plottv = 0
                plotmm = 12.0 if ion == 'o6' else \
                         12.0 if ion == 'ne8' else \
                         12.5 if ion == 'o7' else \
                         13.0 if ion == 'ne9' else \
                         13.0 if ion == 'o8' else \
                         13.0 if ion == 'fe17' else \
                         np.inf 
                plotscatter = var == plottv
                plotscatter &= np.isclose(masslabels_all[tag][0], plotmm)
                if plotscatter:
                    yvals_toplot_temp = yvals_toplot
                else:
                    yvals_toplot_temp = [yvals_toplot[0]] if len(yvals_toplot) == 1 else [yvals_toplot[1]]
                                
                if len(yvals_toplot_temp) == 3:
                    yval = yvals_toplot_temp[0]
                    try:                      
                        ploty1 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval)) 
                    yval = yvals_toplot_temp[2]
                    try:                      
                        ploty2 = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval))    
                    ax.fill_between(plotx, ploty1, ploty2,\
                                    color=color,\
                                    alpha=alphas[var] * shading_alpha,\
                                    label=masslabels_all[tag])

                    yvals_toplot_temp = [yvals_toplot_temp[1]]
                    
                if len(yvals_toplot_temp) == 1:
                    yval = yvals_toplot_temp[0]
                    try:
                        ploty = yvals[var][ion][tag][yval]
                    except KeyError:
                        print('Failed to read in %s - %s - %s -%s'%(var, ion, tag, yval))
                        continue
                    if yval == 50.0: # only highlight the medians
                        patheff = [mppe.Stroke(linewidth=linewidths[var] + 0.5, foreground="b"),\
                                   mppe.Stroke(linewidth=linewidths[var], foreground="w"),\
                                   mppe.Normal()]
                    else:
                        patheff = []
                    ax.plot(plotx, ploty, color=color,\
                            linestyle=linestyles[var],\
                            linewidth=linewidths[var], alpha=alphas[var],\
                            label=masslabels_all[tag], path_effects=patheff)
        for ax in _axes:
            if ytype == 'perc':
                ax.axhline(approx_breaks[ion], 0., 0.1, color='gray', linewidth=1.5, zorder=-1) # ioncolors[ion]
            ax.set_xscale('log')
    
    lcs = []
    line = [[(0, 0)]]
    for var in techvars_touse:
        # set up the proxy artist
        subcols = [colordct[key] for key in sorted(list(colordct.keys()))]
        subcols = np.array(subcols)
        subcols[:, 3] = alphas[var]
        lc = mcol.LineCollection(line * len(subcols),\
                                 linestyle=linestyles[var],\
                                 linewidth=linewidths[var], colors=subcols)
        lcs.append(lc)
    if len(techvars_touse) > 1:
        lax.legend(lcs, [legendnames_techvars[var] for var in techvars_touse],\
                   handler_map={type(lc): pu.HandlerDashedLines()},\
                   fontsize=fontsize, ncol=2 * numcols,\
                   loc='lower center', bbox_to_anchor=(0.5, 0.))
        lax.axis('off')

    plt.savefig(imgname, format='pdf', bbox_inches='tight')
    
    
############################## 3d profiles ####################################

# cumulative 3D radial profiles, mass, volume, ions
def plot3Dprof_cumulative(minrshow=minrshow_R200c, ionset='all'):
    '''
    minrshow just sets the read-in file here; limits are determined by xlim
    '''
    weighttypes = ['Mass', 'Volume', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']        
    fontsize = 12
    linewidth = 2.
    
    wnames = {weighttype: r'\mathrm{%s}'%(ild.getnicename(weighttype, mathmode=True)) if weighttype in ol.elements_ion.keys() else \
                          r'\mathrm{Mass}' if weighttype == 'Mass' else \
                          r"\mathrm{Vol.}" if weighttype == 'Volume' else \
                          None \
              for weighttype in weighttypes}
    
    saveddata = datadir + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrshow)

    with h5py.File(saveddata, 'r') as df:
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        massbins = np.array(df['mass_bins'])
        
        outname = mdir + 'cumul_radprofs_L0100N1504_27_Mh0p5dex_1000.pdf'
        
        figsize = (11., 3.) 
        masskeys.sort()
        masskeys = masskeys[1::2]
        nmasses = len(masskeys)
        
        xlim = (-1.1442460376113135, np.log10(4.2)) # don't care too much about the inner details, extracted out to 4 * R200c
        ylim = (-3.0, 2.1)

        if nmasses != 3:
            raise RuntimeError('Cumulative profile plot is set up for 3 mass bins; found %i'%nmasses)
        ncols = 4
        nrows = 1
        
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0.0, width_ratios=[1.] * ncols, height_ratios=[1.] * nrows )
        axes = np.array([fig.add_subplot(grid[mi // ncols, mi % ncols]) for mi in range(nmasses)])
        lax  = fig.add_subplot(grid[nrows - 1, -1])
        
        colors = ioncolors.copy()
        colors.update({'Mass': 'black', 'Volume': 'gray'})
        linestyle_kw = {'Mass': {'linestyle': 'solid'},\
                        'Volume': {'linestyle': 'solid'},\
                        'o6':   {'dashes': [6, 2]},\
                        'o7':   {'dashes': [3, 1]},\
                        'o8':   {'dashes': [1, 1]},\
                        'ne8':  {'dashes': [6, 2, 3, 2]},\
                        'ne9':  {'dashes': [6, 2, 1, 2]},\
                        'fe17': {'dashes': [3, 1, 1, 1]},\
                        }
            
        for mi in range(nmasses):
            mkey = masskeys[mi]
            binind = np.where(np.array(massbins)[:, 0] == float(mkey))[0][0]
            ax = axes[mi]
            
            # add mass range indicator            
            text = r'$%.1f \, \endash \, %.1f$'%(massbins[binind][0], massbins[binind][1]) #r'$\log_{10} \,$' + binqn + r':            
            ax.text(0.05, 0.95, text, fontsize=fontsize, transform=ax.transAxes,\
                    horizontalalignment='left', verticalalignment='top')
            
            # axis labels and ticks
            labelx = (mi // ncols == nrows - 1) or (mi // ncols == nrows - 2 and nmasses % ncols > 0 and  nmasses % ncols <= mi % ncols)
            labely = mi % ncols == 0
            
            # set up axis labels and ticks
            pu.setticks(ax, top=True, labelleft=labely,\
                        labelbottom=labelx, fontsize=fontsize)
            if labelx:
                ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, / \, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            if labely:
                ax.set_ylabel(r'$\log_{10} \, \mathrm{q}(<r) \, / \, \mathrm{q}(< \mathrm{R}_{\mathrm{200c}})$', fontsize=fontsize)
            
            # plot the data
            yq = 'weight'
            for weight in weighttypes:
                hist = np.array(df['%s/%s/%s/hist'%(weight, mkey, yq)])
                edges_r = np.array(df['%s/%s/%s/edges'%(weight, mkey, yq)])
                if np.any(np.isnan(hist)) or np.any(np.isnan(edges_r)):
                    print('Got NaN values for %s, %s'%(weight, mkey))
                    print('edges: %s'%edges_r)
                    print('hist.: %s'%hist)
                
                if weight == 'Volume': # plot actual volume, not particle volume
                    ax.plot(edges_r, 3. * edges_r,\
                            color=colors[weight],\
                            alpha=1.,\
                            path_effects=None, linewidth=linewidth,\
                            label=r'$%s$'%(wnames[weight]),\
                            **linestyle_kw[weight])
                else:
                    ax.plot(edges_r, np.log10(hist), color=colors[weight],\
                                alpha=1.,\
                                path_effects=None, linewidth=linewidth,\
                                label=r'$%s$'%(wnames[weight]),\
                                **linestyle_kw[weight])
                ax.set_xlim(xlim)
                
        # add legend
        handles, labels = axes[0].get_legend_handles_labels()
        leg = lax.legend(handles=handles, fontsize=fontsize, loc='upper left',\
                   bbox_to_anchor=(0.01, 0.80), ncol=2,\
                   handlelength=2.5, handletextpad=0.5, columnspacing=1.0,\
                   frameon=True)
        leg.set_title(r'quantity $q$', prop={'size': fontsize})
        lax.axis('off')
        
        [_ax.set_ylim(ylim) for _ax in axes]
        
        plt.savefig(outname, format='pdf', bbox_inches='tight')
         

# halo temperature, metallicity, density, rho, nH, Z, T,
# mass-weighted, volume-weighted, 3D
def plot3Dprof_haloprop(minrshow=minrshow_R200c, minrshow_kpc=None,\
                        Zshow='oxygen'):
    '''
    mass- and Volume-weighted rho, T, Z profiles for different halo masses
    in R200c units, stacked weighting each halo by 1 / weight in R200c
    '''
    
    outdir = mdir
    outname = 'profiles_3d_halo_rho_T_Z-%s_median.pdf'%Zshow
    weighttypes = ['Mass', 'Volume']
    elts_Z = [Zshow] #['oxygen', 'neon', 'iron']
    solarZ = ol.solar_abunds_ea
    massslice = slice(None, None, 2) # subset of halo masses to plot
        
    fontsize = 12
    percentiles = [50.]
    linestyles = {'Volume': 'solid',\
                  'Mass': 'dashed'}
    
    if len(elts_Z) > 1:
        alphas = {'oxygen': 1.0,\
                  'neon':   0.7,\
                  'iron':   0.4,\
                  }
    else:
        alphas = {elts_Z[0]: 1.0}
    
    cosmopars = cosmopars_ea_27
   
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \, / \, \mathrm{Z}_{\odot}$',\
                #'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%('q', 'q') 
                }
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\mathrm{200c}}]$'
    
    minrstr = str(minrshow)
    if minrshow_kpc is not None:
        minrstr += '-and-{}-kpc'.format(minrshow_kpc)
    saveddata = datadir + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrstr)
                
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
    #patheff_thick = [mppe.Stroke(linewidth=linewidth + 1., foreground="black"), mppe.Stroke(linewidth=linewidth + 1., foreground="w"), mppe.Normal()]
    
    fig = plt.figure(figsize=(10., 3.5))
    grid = gsp.GridSpec(nrows=1, ncols=6, hspace=0.0, wspace=0.0,\
                        width_ratios=[1., 0.3, 1., 0.3, 1., 0.25],\
                        bottom=0.15, left=0.05)
    axes = np.array([fig.add_subplot(grid[0, 2*i]) for i in range(3)])
    cax = fig.add_subplot(grid[0, 5])
    
    # set up color bar (separate to leave white spaces for unused bins)
    massedges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.])
    cmapname = 'rainbow'
    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
    clist[1::2] = np.array([1., 1., 1., 1.])
    keys = sorted(massedges)
    colordct = {keys[i]: clist[i] for i in range(len(keys))}
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)
    
    #cbar, colordct = add_cbar_mass(cax, cmapname='rainbow', massedges=mass_edges_standard,\
    #                               orientation='vertical', clabel=clabel, fontsize=fontsize, aspect=8.)
    axplot = {'T': 0,\
              'rho': 1,\
              'Z_oxygen': 2,\
              'Z_neon': 2,\
              'Z_iron': 2,\
              }

    with h5py.File(saveddata, 'r') as df:
        # read in mass bins
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        massbins = np.array(df['mass_bins'])
        # use every other mass bin for legibility
        massbins = sorted(massbins, key=lambda x: x[0])
        massbins = massbins[massslice]
        
        for key in axplot.keys():
            axi = axplot[key]
            ax = axes[axi]
            pu.setticks(ax, top=True, labelleft=True, labelbottom=True,\
                        fontsize=fontsize)
            
            ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, /\, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
            ax.set_ylabel(axlabels[key.split('_')[0]], fontsize=fontsize)
            
        for Mhrange in massbins:
            mind = np.where(np.isclose([float(_mk) for _mk in masskeys], Mhrange[0]))[0][0]
            mkey = masskeys[mind]

            typelist = ['rho', 'T'] + ['Z_%s'%(elt) for elt in elts_Z]
            hists_rmin = {ion: {axn: np.array(df['%s/%s/%s/hist_rmin'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            edges0_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_0'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            edges1_rmin = {ion: {axn: np.array(df['%s/%s/%s/edges_rmin_1'%(ion, mkey, axn)]) for axn in typelist} for ion in weighttypes}
            
            color = colordct[float(mkey)]
            for axn in typelist:
                for ion in weighttypes:
                    if axn.startswith('Z_'):
                        alpha = alphas[axn[2:]]
                    else:
                        alpha = 1.
                    _hist = hists_rmin[ion][axn]
                    _e0 = edges0_rmin[ion][axn]
                    _e0 = _e0[:-1] + 0.5 * np.diff(_e0) 
                    _e0[0] = 2. * _e0[1] - _e0[2] # was already adjusted; no need to extend the plot too far to low r
                    _e1 = edges1_rmin[ion][axn]
                    perclines = pu.percentiles_from_histogram(_hist, _e1, axis=1, percentiles=np.array(percentiles) / 100.)
                    if axn.startswith('Z_'):
                        perclines -= np.log10(solarZ[axn[2:]])
                    axes[axplot[axn]].plot(_e0, perclines[0], color=color,\
                        alpha=alpha, linestyle=linestyles[ion],\
                        linewidth=linewidth, path_effects=patheff)
    
    masskeys = sorted(masskeys, key=lambda x: float(x))
    medges = masskeys[massslice]
    for ed in medges:
        #ind = np.where([ed == mkey for mkey in masskeys])[0][0]
        #tval1 = np.log10(T200c_hot(10**float(ed), cosmopars))
        m_med = medians_mmin_standardedges[float(ed)]
        tval1 = np.log10(T200c_hot(10**m_med, cosmopars))
        axes[axplot['T']].axhline(tval1,\
                                  color=colordct[float(ed)], zorder=-1,\
                                  linestyle='dotted')
        #if ind + 1 < len(masskeys):
        #    ed2 = masskeys[ind + 1]
        #    tval2 = np.log10(T200c_hot(10**float(ed2), cosmopars))
        #    axes[axplot['T']].axhline(tval2,\
        #                          color=colordct[float(ed)], zorder=-1,\
        #                          linestyle='dotted')
    
    # legend
    typehandles = [mlines.Line2D([], [], linestyle=linestyles[key],\
                                 label='%s-weighted'%(key),\
                                 path_effects=patheff,\
                                 linewidth=linewidth,\
                                 color = 'gray'
                                 ) for key in weighttypes]
    if len(elts_Z) > 1:
        thandles = [mlines.Line2D([], [], linestyle='solid',\
                                     label=key,\
                                     path_effects=patheff,\
                                     linewidth=linewidth,\
                                     color='gray',\
                                     alpha=alphas[key],\
                                     ) for key in elts_Z]
    else:
        thandles = []
    handles=typehandles + thandles 
    axes[2].legend(handles=handles, fontsize=fontsize,\
               loc='lower left', bbox_to_anchor=(0.0, 0.0),\
               frameon=False, ncol=1) #ncol=min(4, len(handles))
    
    print(axes[0].get_xlim())
    plt.savefig(outdir + outname, format='pdf', box_inches='tight')
    
# halo temperature, density, metallicity, rho, nH, T, Z
# ion-weighted
def plot3Dprof_ionw(minrshow=minrshow_R200c, ions=('o6', 'o7', 'o8'),\
                    axnl=('rho', 'T', 'Z')):
    '''
    ion-weighted rho, T, Z profiles for different halo masses
    in R200c units, stacked weighting each halo by 1 / weight in R200c
    axnl: what to show in which panel (top to bottom)
    ions: which ion to use for ion-weighted values (left to right)
    '''
    
    outdir = mdir
    weighttypes = list(ions)
    axnl = list(axnl)
    outname = 'profiles_3d_halo_%s_%s_median.pdf'%('-'.join(axnl), '-'.join(weighttypes))
    solarZ = ol.solar_abunds_ea
    msel = slice(None, None, 2)
    mnsel = slice(1, None, 2)
        
    fontsize = 12
    percentiles = [50.]
    alpha = 1.
    
    cosmopars = cosmopars_ea_27
    
   
    axlabels = {'T': r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$',\
                'rho': r'$\log_{10} \, \mathrm{n}(\mathrm{H}) \; [\mathrm{cm}^{-3}]$',\
                'Z': r'$\log_{10} \, \mathrm{Z} \, / \, \mathrm{Z}_{\odot}$',\
                #'nion': r'$\log_{10} \, \mathrm{n}(\mathrm{%s}) \; [\mathrm{cm}^{-3}]$'%(wname),\
                'weight': r'$\log_{10} \, %s(< r) \,/\, %s(< \mathrm{R}_{\mathrm{200c}})$'%('q', 'q') 
                }
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\mathrm{200c}}]$'
    
    saveddata = datadir + 'hists3d_forplot_to2d-1dversions_minr-%s.hdf5'%(minrshow)
                
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"), mppe.Stroke(linewidth=linewidth, foreground="w"), mppe.Normal()]
       
    figwidth = 11.
    numions = len(weighttypes)
    numpt   = len(axnl)
    ncols = min(3, numions)
    nrows = ((numions - 1) // ncols + 1) * numpt
    cwidth = 0.5
    wspace = 0.0
    panelwidth = (figwidth - cwidth - wspace * ncols) / ncols
    panelheight = 0.8 * panelwidth
    figheight =  panelheight * nrows
        
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols + 1, hspace=0.0,\
                        wspace=wspace,\
                        width_ratios=[panelwidth] * ncols + [cwidth],\
                        height_ratios=[1.] * nrows,\
                        bottom=0.07, top=0.95)
    axes = np.array([[fig.add_subplot(grid[ii // ncols + ti, ii % ncols])\
                      for ti in range(numpt)] for ii in range(numions)])
    cax = fig.add_subplot(grid[:min(nrows, 2), ncols])
    
    
    massedges = np.array([11., 11.5, 12., 12.5, 13., 13.5, 14.])
    cmapname = 'rainbow'
    
    clist = cm.get_cmap(cmapname, len(massedges))(np.linspace(0.,  1., len(massedges)))
    clist[mnsel] = np.array([1., 1., 1., 1.])
    keys = sorted(massedges)
    colordct = {keys[i]: clist[i] for i in range(len(keys))}
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)

    with h5py.File(saveddata, 'r') as df:
        masskeys = [_str.decode() for _str in np.array(df['mass_keys'])]
        massbins = np.array(df['mass_bins'])
        massbins = sorted(massbins, key=lambda x: x[0])
        massbins = massbins[msel]
        
        for ii in range(numions):
            for ti in range(numpt):
                ion = ions[ii]
                axn = axnl[ti]
                ax = axes[ii, ti]
                
                labelleft = ii % ncols == 0
                labelbottom = (numions - ii <= ncols and ti == numpt - 1)
                
                pu.setticks(ax, top=True, right=True, labelleft=labelleft,\
                            labelbottom=labelbottom, fontsize=fontsize)
                if labelleft:
                    ax.set_ylabel(axlabels[axn], fontsize=fontsize)
                if labelbottom:
                    ax.set_xlabel(r'$\log_{10} \, \mathrm{r} \, /\, \mathrm{R}_{\mathrm{200c}}$', fontsize=fontsize)
        
                for Mhrange in massbins:
                    mind = np.where(np.isclose([float(_mk) for _mk in masskeys], Mhrange[0]))[0][0]
                    mkey = masskeys[mind]
        
                    if axn == 'Z':
                        if ion in ol.elements_ion.keys():
                            elt = ol.elements_ion[ion]
                        else:
                            elt = 'oxygen' #default
                        _axn = 'Z_{}'.format(elt)
                    else:
                        _axn = axn
                    if ion in ol.elements_ion.keys():
                        name = ild.getnicename(ion, mathmode=True)
                    else:
                        name = ion
                    ax.text(0.95, 0.95, r'$\mathrm{%s}$-weighted'%(name), fontsize=fontsize,\
                            transform=ax.transAxes, horizontalalignment='right',\
                            verticalalignment='top', fontweight='normal')
                    
                    _hist = np.array(df['%s/%s/%s/hist_rmin'%(ion, mkey, _axn)]) 
                    _e0 = np.array(df['%s/%s/%s/edges_rmin_0'%(ion, mkey, _axn)])
                    _e1 = np.array(df['%s/%s/%s/edges_rmin_1'%(ion, mkey, _axn)]) 
                    _e0[0] = 2. * _e0[1] - _e0[2] # don't want the innermost in in the original extraction
                    _e0 = _e0[:-1] + 0.5 * np.diff(_e0)   
                    
                    alpha = 1.
                    color = colordct[float(mkey)]
                    perclines = pu.percentiles_from_histogram(_hist, _e1, axis=1, percentiles=np.array(percentiles) / 100.)
                    if axn == 'Z':
                        perclines -= np.log10(solarZ[elt])
                    ax.plot(_e0, perclines[0], color=color,\
                        alpha=alpha, linestyle='solid',\
                        linewidth=linewidth, path_effects=patheff)
                    
                    if axn == 'T':
                        mval = medians_mmin_standardedges[float(mkey)]
                        ax.axhline(np.log10(T200c_hot(10**mval, cosmopars)),\
                                                  color=colordct[float(mkey)],\
                                                  zorder=-1,\
                                                  linestyle='dotted')
                        
                        if ion in ol.elements_ion:
                            ax.axhline(Tranges_CIE[ion][0],\
                                       color='black', zorder=-1,\
                                       linestyle='dotted')
                            ax.axhline(Tranges_CIE[ion][1],\
                                       color='black', zorder=-1,\
                                       linestyle='dotted')
    # sync y axes:
    for ti in range(numpt):
        ylims = np.array([ax.get_ylim() for ax in axes[:, ti]])
        y0 = np.min(ylims[:, 0])
        y1 = np.max(ylims[:, 1])
        [ax.set_ylim(y0, y1) for ax in axes[:, ti]]
        
    plt.savefig(outdir + outname, format='pdf', box_inches='tight')
    
    
########################### halo mass splits ##################################

# bar chart, CGM, IGM, halo mass, halo mass contributions, baryon distribution
def plotfracs_by_halo(ions=['Mass', 'oxygen', 'o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'], fmt='pdf'):
    '''
    first: group mass bins by halo mass or subhalo catgory first
    '''
    
    ### corrections checked against total ion omegas in particle data only:
    #                                histogram for this plot    particle / snap box totals  
    #Particle data / total for hneutralssh: 0.9960255976557024  0.9960400162241149
    #Particle data / total for Mass: 0.26765323008070446        0.2676554816256265 
    #Particle data / total for o6:   0.27129005157978364        0.2712953355974472
    #Particle data / total for ne8:  0.2764794130850087         0.2764820025783056
    #Particle data / total for o7:   0.4821431746085879         0.48214487331886124
    #Particle data / total for ne9:  0.5315625247371946         0.5315598048201217
    #Particle data / total for o8:   0.3538290133287435         0.3538293076657853         
    #Particle data / total for fe17: 0.674329330727548          0.6743306068203926
    # 
    
    ## get total ion masses -> ion numbers from box_statistcis.py -> calcomegas
    # needed since the halo fractions come from histograms using particle data,
    # which excludes IGM contributions
    # omega data comes from snapshots
    omega_to_g_L100_27 = 2.0961773946142324e+50
    #Particle abundances, SF gas at 10^4 K
    Omega_gas = 0.056292501227365246
    Omega_oxygen_gas = 3.494829254263584e-05
    Omega_neon_gas   = 4.802734584639758e-06
    Omega_iron_gas   = 5.074204751697753e-06
    Omega_o6_gas   = 2.543315324105566e-07
    Omega_o7_gas   = 6.2585354699292046e-06
    Omega_o8_gas   = 7.598227615977929e-06
    Omega_ne8_gas  = 1.2563290036620843e-07
    Omega_ne9_gas  = 1.5418221011797857e-06
    Omega_fe17_gas = 6.091088105543567e-07
    Omega_hneutralssh_gas = 0.00023774499538436282
    
    total_nions = {'o6': Omega_o6_gas,\
                   'o7': Omega_o7_gas,\
                   'o8': Omega_o8_gas,\
                   'ne8': Omega_ne8_gas,\
                   'ne9': Omega_ne9_gas,\
                   'fe17': Omega_fe17_gas,\
                   'hneutralssh': Omega_hneutralssh_gas,\
                   'oxygen': Omega_oxygen_gas,\
                   'iron': Omega_iron_gas,\
                   'neon': Omega_neon_gas,\
                   'Mass': Omega_gas}
    
    total_nions = {ion: total_nions[ion] * omega_to_g_L100_27 / (ionh.atomw[string.capwords(ol.elements_ion[ion])] * c.u) \
                        if ion in ol.elements_ion else \
                        total_nions[ion] * omega_to_g_L100_27 / (ionh.atomw[string.capwords(ion)] * c.u) \
                        if string.capwords(ion) in ionh.atomw else
                        total_nions[ion] * omega_to_g_L100_27 \
                   for ion in total_nions     
                   }
    
    datafile_dct = {}
    for ion in ions:
        if ion == 'Mass':
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_T4EOS.hdf5'
            datafile = datadir + datafile_base%(ion)
        elif ion in ol.elements_ion:
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_PtAb_T4EOS.hdf5'
            datafile = datadir + datafile_base%('Nion_%s'%ion)
        else:
            datafile_base = 'particlehist_%s_L0100N1504_27_test3.4_PtAb_T4EOS.hdf5'
            datafile = datadir + datafile_base%('Nion_%s'%ion)
        datafile_dct[ion] = datafile
    
    outname = mdir + 'barchart_halomass_L0100N1504_27_T4EOS.%s'%(fmt)
    
    data_dct = {}
    for ion in ions:
        datafile = datafile_dct[ion]
        with h5py.File(datafile, 'r') as fi:
            try: # Mass, ion species histograms
                groupname = 'Temperature_T4EOS_Density_T4EOS_M200c_halo_allinR200c_subhalo_category'
                tname = 'Temperature_T4EOS'
                dname = 'Density_T4EOS'
                sname = 'subhalo_category'
                hname = 'M200c_halo_allinR200c'
                
                mgrp = fi[groupname]       
                tax = mgrp[tname].attrs['histogram axis']
                #tbins = np.array(mgrp['%s/bins'%(tname)])
                dax = mgrp[dname].attrs['histogram axis']
                #dbins = np.array(mgrp['%s/bins'%(dname)]) + np.log10(rho_to_nh)      
                sax = mgrp[sname].attrs['histogram axis']
                sbins = np.array(mgrp['%s/bins'%(sname)])
                #hax = mgrp[hname].attrs['histogram axis']
                hbins = np.array(mgrp['%s/bins'%(hname)])
                
                hist = np.array(mgrp['histogram'])
                if mgrp['histogram'].attrs['log']:
                    hist = 10**hist
                hist = np.sum(hist, axis=(dax, tax, sax))
                total_in = mgrp['histogram'].attrs['sum of weights']
                # print('Histogramming recovered %f of input weights'%(np.sum(hist) / total_in))
            except KeyError: # element histograms
                groupname = 'M200c_halo_allinR200c_subhalo_category'
                sname = 'subhalo_category'
                hname = 'M200c_halo_allinR200c'
                
                mgrp = fi[groupname]         
                sax = mgrp[sname].attrs['histogram axis']
                sbins = np.array(mgrp['%s/bins'%(sname)])
                #hax = mgrp[hname].attrs['histogram axis']
                hbins = np.array(mgrp['%s/bins'%(hname)])
                
                hist = np.array(mgrp['histogram'])
                if mgrp['histogram'].attrs['log']:
                    hist = 10**hist
                hist = np.sum(hist, axis=(sax,))
                total_in = mgrp['histogram'].attrs['sum of weights']
                
            #cosmopars = {key: item for key, item in fi['Header/cosmopars'].attrs.items()}
            total_allpart = total_nions[ion]
            print('Particle data / total for %s: %s'%(ion, total_in / total_allpart))
            
            hist[0] += total_allpart - total_in # add snap/particle data difference to IGM contribution
            data_dct[ion] = {'hist': hist, 'hbins': hbins, 'sbins': sbins, 'total': total_allpart}
            
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'        
    ylabel = 'fraction'
    xlabels = [r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)) if ion in ol.elements_ion else ion for ion in ions] 

    # custom color map: standard rainbow + extra for a higher mass bin    
    cmapname = 'rainbow'
    nigmcolor = 'saddlebrown'
    igmcolor = 'gray'   
    #print(hbins - np.log10(c.solar_mass))
    namededges = hbins[2:-1] - np.log10(c.solar_mass) # first two are -np.inf, and < single particle mass, last bin is empty (> 10^15 Msun)
    #print(namededges)

    mmin = 11.
    indmin = np.argmin(np.abs(namededges - mmin))
    plotedges = namededges[indmin:]

    clist = cm.get_cmap(cmapname, len(plotedges) - 2)(np.linspace(0., 1., len(plotedges) - 2))
    clist = np.append(clist, [mpl.colors.to_rgba('firebrick')], axis=0)
    colors = {hi + 2: clist[hi - indmin] for hi in range(indmin, indmin + len(plotedges) - 1)}
    colors[1] = mpl.colors.to_rgba(nigmcolor)
    colors[0] = mpl.colors.to_rgba(igmcolor)
    colors[len(hbins) - 1] = mpl.colors.to_rgba('magenta') # shouldn't actaully be used


    fig = plt.figure(figsize=(5.5, 4.))
    maingrid = gsp.GridSpec(ncols=2, nrows=2, hspace=0.0, wspace=0.05, height_ratios=[0.7, 4.3], width_ratios=[5., 1.])
    cax = fig.add_subplot(maingrid[1, 1])
    lax = fig.add_subplot(maingrid[0, :])
    ax = fig.add_subplot(maingrid[1, 0])
    
    cmap = mpl.colors.ListedColormap(clist)
    cmap.set_under(nigmcolor)
    norm = mpl.colors.BoundaryNorm(plotedges, cmap.N)

    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append([0.], plotedges, axis=0),\
                                ticks=plotedges,\
                                spacing='proportional', extend='min',\
                                orientation='vertical')
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(8.)
    
    bottom = np.zeros(len(ions))
    barwidth = 0.9
    xvals = np.arange(len(ions))
    
    for ind1 in range(len(hbins) - 2):  # last bin: M200c > 15, is empty 
        #print(hbins[ind1] - np.log10(c.solar_mass))    
        alpha = 1.
        if ind1 >= indmin + 2 or ind1 == 0:
            color = mpl.colors.to_rgba(colors[ind1], alpha=alpha)
        else:
            color = nigmcolor
        sel = (ind1,) 
        
        if ind1 > 0 and ind1 < indmin + 2:
            if ind1 == 1: # start addition for nigm bins
                vals = np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            else:
                vals += np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            if ind1 == indmin + 1: # end of addition; plot
                ax.bar(xvals, vals, barwidth, bottom=bottom, color=color)
                bottom += vals
        else:
            vals = np.array([data_dct[ion]['hist'][sel] / data_dct[ion]['total'] for ion in ions])
            
            ax.bar(xvals, vals, barwidth, bottom=bottom, color=color)
            bottom += vals
            
    pu.setticks(ax, fontsize, top=False)
    ax.xaxis.set_tick_params(which='both', length=0.) # get labels without showing the ticks        
    ax.set_xticks(xvals)
    ax.set_xticklabels(xlabels)
    for label in ax.get_xmajorticklabels():
        label.set_rotation(45)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    legelts = [mpatch.Patch(facecolor=igmcolor, label='IGM')]
    lax.legend(handles=legelts, ncol=4, fontsize=fontsize, bbox_to_anchor=(0.5, 0.05), loc='lower center')
    lax.axis('off')
    
    plt.savefig(outname, format=fmt, bbox_inches='tight')

# baryon split, baryon contributions, halo mass buildup, halo metals, 
# metals split
# makes two of the paper plots: the halo mass and oxygen decompositions
def plot_masscontr_halo(addedges=(0.0, 1.), var='Mass',\
                        nHcut=False, nHm2=False, nHorSF=False):
    '''
    addedges: radial regions to consider; (0.0, 1.0) or (0.0, 2.0), units R200c
    var: 'Mass' for total mass
         'oxygen', 'neon', or 'iron' for metal mass
    nHcut: CGM/ISM split based on nH </> 10**-1 cm**-3
           otherwise: based on SFR =/> 0.
           False used from examination of var=Mass plots: in low-mass haloes, 
           the nH cut excludes quite some SF gas, which should be ISM. The SF
           cut will, however, exclude some dense, but low-Z gas.
    nHm2:  same as nHcut, but uses a lower density cut of 10**-2 cm**-3. This 
           will include some gas more typically called CGM in the ISM bin
    nHorSF: ISM is defined as gas that meets the nH > 10**-1 cm**-3 limit or 
           is star-forming; compromise definition that tries to inlcude 
           everything typically called ISM
    '''
    print('lines are medians, shaded regions are central 80% (only shown for ISM, stars, CGM)')
    
    fontsize = 12
    
    if np.sum([nHcut, nHm2, nHorSF]) > 1:
        print('Set at most one option controlling the ISM definition to True')
    
    if nHcut:
        print('Defining the ISM as nH > 10**-1 cm**-3')
        filename_in = datadir + 'massdist-baryoncomp_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_nHcut.hdf5'%(str(addedges[0]), str(addedges[1]))
        outname = 'masscontr_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s_nHcut'%(str(addedges[0]), str(addedges[1]), var)
    elif nHm2:
        print('Defining the ISM as nH > 10**-2 cm**-3')
        filename_in = datadir + 'massdist-baryoncomp_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_nHm2cut.hdf5'%(str(addedges[0]), str(addedges[1]))
        outname = 'masscontr_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s_nHm2cut'%(str(addedges[0]), str(addedges[1]), var)
    elif nHorSF:
        print('Defining the ISM as nH > 10**-1 cm**-3 or SFR > 0')
        filename_in = datadir + 'massdist-baryoncomp_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_nHorSFcut.hdf5'%(str(addedges[0]), str(addedges[1]))
        outname = 'masscontr_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s_nHorSFcut'%(str(addedges[0]), str(addedges[1]), var)
    else:
        print('Defining the ISM as SFR > 0')
        filename_in = datadir + 'massdist-baryoncomp_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb.hdf5'%(str(addedges[0]), str(addedges[1]))
        outname = 'masscontr_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s'%(str(addedges[0]), str(addedges[1]), var)
    outname = outname.replace('.', 'p')
    outname = mdir + outname + '.pdf'
    m200cbins = np.array(list(np.arange(11., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])
    percentiles = [10., 50., 90.]
    alpha = 0.3
    lw = 2
    xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    ylabel = '{var} fraction'.format(var=var.lower() + (' mass' if var != 'Mass' else ''))
    
    if nHcut or nHm2:
        cgmlab = 'CGM'
        ismlabs = ['ISM']
    elif nHorSF:
        dlolab = 'lodens'
        dhilab = 'hidens'
        nsflab = 'nonSF'
        ysflab = 'SF'
        ismlabs = [ysflab + '_' + dlolab,\
                   ysflab + '_' + dhilab,\
                   nsflab + '_' + dhilab]
        cgmlab = nsflab + '_' + dlolab
    else:
        cgmlab = 'nonSF'
        ismlabs = ['SF']
    if nHm2:
        ctag = '-nHm2'
    elif nHorSF:
        ctag = '-nHorSF'
    else:
        ctag = ''
    if var == 'Mass':
        groupname = 'massdist_Mass'
        catcol = {'BHs': ['BHs'],\
                  'DM': ['DM'],\
                  'gas': ['gas{ctag}'],\
                  'stars': ['stars'],\
                  'ISM': ['gas{ctag}_{ism}_T--inf-5.0',\
                          'gas{ctag}_{ism}_T-5.0-5.5',\
                          'gas{ctag}_{ism}_T-5.5-7.0',\
                          'gas{ctag}_{ism}_T-7.0-inf'],\
                  r'CGM $<5.5$': ['gas{ctag}_{cgm}_T--inf-5.0',\
                                  'gas{ctag}_{cgm}_T-5.0-5.5'],\
                  r'CGM $5.5 \endash 7$': ['gas{ctag}_{cgm}_T-5.5-7.0'],\
                  r'CGM $> 7$': ['gas{ctag}_{cgm}_T-7.0-inf']}
        addcol = {'total': ['BHs', 'gas', 'stars', 'DM'],\
                  'CGM': [r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$'],\
                  'baryons': ['BHs', 'gas', 'stars'],\
                  'gas-subsum': ['ISM', r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$']}
    else:
        groupname = 'massdist_{var}'.format(var=var)
        catcol = {'gas': ['gas{ctag}-{var}'],\
                  'stars': ['stars-{var}'],\
                  'ISM': ['gas{ctag}-{var}_{ism}_T--inf-5.0',\
                          'gas{ctag}-{var}_{ism}_T-5.0-5.5',\
                          'gas{ctag}-{var}_{ism}_T-5.5-7.0',\
                          'gas{ctag}-{var}_{ism}_T-7.0-inf'],\
                  r'CGM $<5.5$': ['gas{ctag}-{var}_{cgm}_T--inf-5.0',\
                                  'gas{ctag}-{var}_{cgm}_T-5.0-5.5'],\
                  r'CGM $5.5 \endash 7$': ['gas{ctag}-{var}_{cgm}_T-5.5-7.0'],\
                  r'CGM $> 7$': ['gas{ctag}-{var}_{cgm}_T-7.0-inf']}
        addcol = {'total': ['gas', 'stars'],\
                  'CGM': [r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$'],\
                  'gas-subsum': ['ISM', r'CGM $<5.5$', r'CGM $5.5 \endash 7$', r'CGM $> 7$']}
    # add ISM formatting iteration separately, otherwise other values are duplicated
    catcol = {key: [val.format(ism='{ism}', cgm=cgmlab, var=var, ctag=ctag) \
                    for val in catcol[key]] \
                    for key in catcol}
    catcol['ISM'] = [val.format(ism=ismlab) for ismlab in ismlabs \
                     for val in catcol['ISM']]    

    with h5py.File(filename_in, 'r') as fd:
        cosmopars = {key: item for key, item in fd['Header/cosmopars'].attrs.items()}
        m200cvals = np.log10(np.array(fd['M200c_Msun']))
        grp = fd[groupname]
        arr_all = np.array(grp['mass'])
        collabels = list(grp.attrs['categories'])
        collabels = np.array([lab.decode() for lab in collabels])
        catind = {key: np.array([np.where(collabels == subn)[0][0] for subn in catcol[key]]) \
                       for key in catcol}
        massdata = {key: np.sum(arr_all[:, catind[key]], axis=1) for key in catcol}
        _sumdata = {key: np.sum([massdata[subkey] for subkey in addcol[key]], axis=0) for key in addcol}
        massdata.update(_sumdata)
        # check
        if not np.all(np.isclose(massdata['gas'], massdata['gas-subsum'])):
            return massdata
            print(massdata)
            raise RuntimeError('The gas subcategory masses do not add up to the total gas mass for all halos')

    print(m200cvals)
    print(m200cbins)    
    bininds = np.digitize(m200cvals, m200cbins)
    bincens = m200cbins[:-1] + 0.5 * np.diff(m200cbins)
    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = grid = gsp.GridSpec(ncols=1, nrows=2, hspace=0.1, wspace=0.0,\
                               height_ratios=[3.8, 1.2], top=0.95, bottom=0.05)
    ax  = fig.add_subplot(grid[0, 0])
    lax = fig.add_subplot(grid[1, 0])
    
    colors = {'BHs': 'black',\
              'DM':  'gray',\
              'gas': 'C1',\
              'stars': 'C8',\
              'ISM': 'C3',\
              'CGM': 'C0',\
              'baryons': 'gray',\
              r'CGM $<5.5$': 'C9',\
              r'CGM $5.5 \endash 7$': 'C4',\
              r'CGM $> 7$': 'C6'}
    linestyles = {'BHs': 'solid',\
              'DM':  'solid',\
              'gas': 'solid',\
              'stars': 'solid',\
              'baryons': 'solid',\
              'BHs': 'solid',\
              'ISM': 'solid',\
              'CGM': 'solid',\
              r'CGM $<5.5$': 'dotted',\
              r'CGM $5.5 \endash 7$': 'solid',\
              r'CGM $> 7$': 'dotted'}
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    pu.setticks(ax, fontsize=fontsize) 

    ax.set_yscale('log')
    if var == 'Mass':
        ax.set_ylim(1e-4, 0.2)
    else:
        ax.set_ylim(2e-3, 1.)
    #if 'DM' in catcol:
    #    fdm = 1. - cosmopars['omegab'] / cosmopars['omegam']
    #    ax.axhline(fdm, linewidth=lw, color=colors['DM'], label=r'$1 - \Omega_{\mathrm{b}} \,/\, \Omega_{\mathrm{m}}$')
    if var == 'Mass':
        fb = cosmopars['omegab'] / cosmopars['omegam']
        ax.axhline(fb, linewidth=lw, linestyle='dashed', color=colors['DM'], label=r'$\Omega_{\mathrm{b}} \,/\, \Omega_{\mathrm{m}}$')
        
    for label in massdata:
        if label in ['DM', 'total', 'gas-subsum', 'gas', r'CGM $<5.5$', r'CGM $> 7$', 'BHs']:
            continue
        _massdata = massdata[label] / massdata['total']
        _color = colors[label]
        
        percvals = np.array([np.percentile(_massdata[bininds == i], percentiles) for i in range(1, len(m200cbins))]).T
        ax.plot(bincens, percvals[1], label=label, color=_color, linewidth=lw, linestyle=linestyles[label])
        if label not in ['baryons', r'CGM $5.5 \endash 7$']:
            ax.fill_between(bincens, percvals[0], percvals[2], color=_color, alpha=alpha)
    
    handles, lables = ax.get_legend_handles_labels()
    
    #legelts = [mpatch.Patch(facecolor='tan', alpha=alpha, label='%.1f %%'%(percentiles[2] - percentiles[0]))] + \
    #          [mlines.Line2D([], [], color='tan', label='median')]

    lax.legend(handles=handles, ncol=3, fontsize=fontsize, bbox_to_anchor=(0.5, 0.6), loc='upper center')
    lax.axis('off')
    
    plt.savefig(outname, format='pdf', box_inches='tight')
    
# oxygen split, oxygen fractions, ion fractions, halo ion fractions, 
# CGM ion fractions 
def plot_ionfracs_halos(addedges=(0.1, 1.), var='focus'):
    '''
    addedges: radial regions to consider; (min, max) units R200c
              min: 0.0, or 0.1, max: 1.0 or 2.0
    var: 'focus' for o6, o7, o8, ne8, ne9, fe17
         'oxygen' for all the oxygen species
    '''
    fontsize = 12
    
    filename_in = datadir + 'ionfracs_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb.hdf5'%(str(addedges[0]), str(addedges[1]))
    outname = 'ionfracs_halos_L0100N1504_27_Mh0p5dex_1000_%s-%s-R200c_PtAb_%s'%(str(addedges[0]), str(addedges[1]), var)
    outname = outname.replace('.', 'p')
    outname = mdir + outname + '.pdf' 
    m200cbins = np.array(list(np.arange(11., 13.05, 0.1)) + [13.25, 13.5, 13.75, 14.0, 14.6])
    percentiles = [10., 50., 90.]
    alpha = 0.3
    lw = 2
    xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    ylabel = r'CGM ion fraction'
    
    iondata = {}
    with h5py.File(filename_in, 'r') as fd:
        cosmopars = {key: item for key, item in fd['Header/cosmopars'].attrs.items()}
        m200cvals = np.log10(np.array(fd['M200c_Msun']))
        fkeys = list(fd.keys())
        for key in ['Header', 'M200c_Msun', 'galaxyids']:
            if key in fkeys:
                fkeys.remove(key)
        for key in fkeys:
            grp = fd[key]
            _ions = grp.attrs['ions']
            try: # only one ion
                _ions = [_ions.decode()]
            except: # list/array of ions
                _ions = [_ion.decode() for _ion in _ions]
            allfracs = np.array(grp['fractions'])
            for ii in range(len(_ions)):
                iondata[_ions[ii]] = allfracs[:, ii]
            
        basesel = np.all(np.array([np.isfinite(iondata[ion]) for ion in iondata]), axis=0) # issues from a low-mass halo: probably a very small metal-free system
        m200cvals = m200cvals[basesel]
        for ion in iondata:
            iondata[ion] = iondata[ion][basesel]
        
    bininds = np.digitize(m200cvals, m200cbins)
    bincens = m200cbins[:-1] + 0.5 * np.diff(m200cbins)
    #bincens[-1] = np.median(m200cvals[bininds == len(m200cbins) - 1])
    #print(bincens)
    T200cvals = T200c_hot(10**bincens, cosmopars)
    
    if var == 'focus':
        rt = 0.87
    else:
        rt = 0.90
    fig = plt.figure(figsize=(5.5, 5.))
    grid = grid = gsp.GridSpec(ncols=2, nrows=2, hspace=0.0, wspace=0.0,\
                               width_ratios=[5., 1.], height_ratios=[0.7, 2.],\
                               left=0.15, right=rt)
    ax  = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[0, 0])
    lax = fig.add_subplot(grid[:, 1])
    
    extracolors = {'o1': 'navy',\
                   'o2': 'skyblue',\
                   'o3': 'olive',\
                   'o4': 'darksalmon',\
                   'o5': 'darkred'}
    if var == 'focus':
        plotions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    elif var == 'oxygen':
        plotions = ['o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    pu.setticks(ax, fontsize=fontsize) 
    ax2.set_ylabel('$\mathrm{CIE}(\\mathrm{{T}} = \\mathrm{{T}}_{{\\mathrm{{200c}}}})$ fraction', fontsize=fontsize)
    if var == 'focus':
        ax.set_yscale('log')
        ax2.set_yscale('log')
        ax.set_ylim(1e-4, 0.7)
        ax2.set_ylim(1e-4, 1.3)
    else:
        ax.set_ylim(0., 1.)
        ax2.set_ylim(0., 1.)
    prev_halo = np.zeros(len(bincens))
    prev_cie = np.zeros(len(bincens))
    
    for ion in plotions:
        _iondata = iondata[ion]
        _color = ioncolors[ion] if ion in ioncolors else extracolors[ion]
        
        if var == 'focus':
            percvals = np.array([np.percentile(_iondata[bininds == i], percentiles) for i in range(1, len(m200cbins))]).T
            ax.plot(bincens, percvals[1], label=r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)), color=_color, linewidth=lw)
            ax.fill_between(bincens, percvals[0], percvals[2], color=_color, alpha=alpha)
            tablevals = cu.find_ionbal(cosmopars['z'], ion, {'logT': np.log10(T200cvals), 'lognH': np.ones(len(T200cvals)) * 6.}) # extreme nH -> highest tabulated values used
            ax2.plot(np.log10(T200cvals), tablevals, color=_color, linewidth=lw)  
        else:
            avgs = np.array([np.average(_iondata[bininds == i]) for i in range(1, len(m200cbins))])
            tablevals = find_ionbal_bensgadget2(cosmopars['z'], ion, {'logT': np.log10(T200cvals), 'lognH': np.ones(len(T200cvals)) * 6.}) # extreme nH -> highest tabulated values used
            
            ax.fill_between(bincens, prev_halo, prev_halo + avgs, color=_color, label=r'$\mathrm{%s}$'%(ild.getnicename(ion, mathmode=True)))
            prev_halo += avgs
            ax2.fill_between(np.log10(T200cvals), prev_cie, prev_cie + tablevals, color=_color)
            prev_cie += tablevals
    
    # set T ticks
    mlim = 10**np.array(ax.get_xlim())
    tlim = np.log10(T200c_hot(mlim, cosmopars))
    ax2.set_xlim(tuple(tlim))
    ax2.set_xlabel(r'$\log_{10} \, \mathrm{T}_{\mathrm{200c}} \; [\mathrm{K}]$', fontsize=fontsize)
    pu.setticks(ax2, fontsize=fontsize, labelbottom=False, labeltop=True)
    ax2.xaxis.set_label_position('top') 
    
    if var == 'oxygen': # resolve y tick label overlap
        #yticklab = ax2.get_yticklabels(minor=False)
        yticks =  ax2.get_yticks(minor=False)
        #newticklab = [tick.get_text() for tick in yticklab[1:]]
        ax2.set_yticks(yticks[1:], minor=False)
        #ax2.set_yticklabels(newticklab, minor=False) # tick labels are empty: formatted only when rendered?
    handles, lables = ax.get_legend_handles_labels()
    
    if var == 'focus':
        #legelts = [mpatch.Patch(facecolor='gray', alpha=alpha, label='%.1f %%'%(percentiles[2] - percentiles[0]))] + \
        #          [mlines.Line2D([], [], color='gray', label='median')]
        handles = handles
    else:
        handles = handles[::-1]
    lax.legend(handles=handles, ncol=1, fontsize=fontsize,\
               bbox_to_anchor=(0.02, 1.0), loc='upper left')
    lax.axis('off')
    
    plt.savefig(outname, format='pdf', box_inches='tight')
    
################################ Misc #########################################

# ionbal, ion balance, ion tables, ionization tables, Tvir, virial temperature
def plot_Tvir_ions_nice(snap=27, _ioncolors=ioncolors):
    '''
    contour plots for ions balances + shading for halo masses at different Tvir
    '''
    fontsize = 12
    
    if snap == 27:
        cosmopars = cosmopars_ea_27
        logrhob = logrhob_av_ea_27

    elif snap == 23:
        cosmopars = cosmopars_ea_27
        cosmopars['a'] = 0.665288  # eagle wiki
        cosmopars['z'] = 1. / cosmopars['a'] - 1. 
        logrhob = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars['h']**2 * cosmopars['omegab'] / cosmopars['a']**3 )
        
    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17'] #, 'he2'
    ioncolors = _ioncolors.copy()
    
    #ioncolors.update({'he2': 'darkgoldenrod'})
    Ts = {}
    Tmaxs = {}
    nHs = {}
    bals = {}
    maxfracs = {}
    
    fracv = 0.1
    
    filen = datadir + 'ionbal_snap{snap}.hdf5'.format(snap=snap) 
    with h5py.File(filen, 'r') as fi:
        for ion in ions:          
            bal = np.array(fi[ion]['ionbal'])
            T = np.array(fi[ion]['logTK'])
            bals[ion] = bal
            nHs[ion] = np.array(fi[ion]['lognHcm3'])
            Ts[ion] = T
            indmaxfrac = np.argmax(bal[-1, :])
            maxfrac = bal[-1, indmaxfrac]
            Tmax = T[indmaxfrac]
            Tmaxs[ion] = Tmax
        
            xs = pu.find_intercepts(bal[-1, :], T, fracv * maxfrac)
            print('Ion %s has maximum CIE fraction %.3f, at log T[K] = %.1f, %s max range is %s'%(ion, maxfrac, Tmax, fracv, str(xs)))
            maxfracs[ion] = maxfrac
            
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(5.5, 10.), gridspec_kw={'hspace': 0.})
    ax1.set_xlim(-8., -1.5)
    ax1.set_ylim(3.4, 7.65)
    ax2.set_xlim(-8., -1.5)
    ax2.set_ylim(3.4, 7.65)
    axions = {1: ['o6', 'o7', 'o8'], 2: ['ne8', 'ne9', 'fe17']}
    
    ax1.set_ylabel(r'$\log_{10} \, T \; [K]$', fontsize=fontsize)
    ax2.set_ylabel(r'$\log_{10} \, T \; [K]$', fontsize=fontsize)
    ax2.set_xlabel(r'$\log_{10} \, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    pu.setticks(ax1, fontsize=fontsize, right=False, labelbottom=False)
    pu.setticks(ax2, fontsize=fontsize, right=False)
    
    ax1.axvline(logrhob + np.log10(rho_to_nh), 0., 0.85, color='gray', linestyle='dashed', linewidth=1.5)
    ax2.axvline(logrhob + np.log10(rho_to_nh), 0., 0.85, color='gray', linestyle='dashed', linewidth=1.5)
    #ax.axvline(logrhoc + np.log10(rho_to_nh * 200. * cosmopars['omegab'] / cosmopars['omegam']), 0., 0.75, color='gray', linestyle='solid', linewidth=1.5)
    

    for ax, axi in zip([ax1, ax2], [1, 2]):
        for ion in axions[axi]:
            ax.contourf(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], alpha=0.1, linewidths=[3.], levels=[0.1 * maxfracs[ion], 1.])
            ax.contour(nHs[ion], Ts[ion], bals[ion].T, colors=ioncolors[ion], linewidths=[2.], levels=[0.1 * maxfracs[ion]], linestyles=['solid'])
        for ion in ions:
            ax.axhline(Tmaxs[ion], 0.95, 1., color=ioncolors[ion], linewidth=3.)
            
        #bal = bals[ion]
        #maxcol = bal[-1, :]
        #diffs = bal / maxcol[np.newaxis, :]
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal == 0)] = 0.
        #diffs[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] = bal[np.logical_and(maxcol[np.newaxis, :] == 0, bal != 0)] / 1e-18
        #diffs = np.abs(np.log10(diffs))
            
        #mask = bal < 0.6 * fracv * maxfracs[ion] # 0.6 gets the contours to ~ the edges of the ion regions
        #diffs[mask] = np.NaN

        #ax.contour(nHs[ion], Ts[ion][np.isfinite(maxcol)], (diffs[:, np.isfinite(maxcol)]).T, levels=[np.log10(ciemargin)], linestyles=['solid'], linewidths=[1.], alphas=0.5, colors=ioncolors[ion])

        axy2 = ax.twinx()
        ylim = ax.get_ylim()
        axy2.set_ylim(*ylim)
        mhalos = np.arange(9.0, 15.1, 0.5)
        Tvals = np.log10(T200c_hot(10**mhalos, cosmopars))
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_yticks(Tvals)
        axy2.set_yticklabels(Tlabels)
        pu.setticks(axy2, fontsize=fontsize, left=False, right=True, labelleft=False, labelright=True)
        axy2.minorticks_off()
        axy2.set_ylabel(r'$\log_{10} \, \mathrm{M_{\mathrm{200c}}} (T_{\mathrm{200c}}) \; [\mathrm{M}_{\odot}]$', fontsize=fontsize)
    
        handles = [mlines.Line2D([], [], label=ild.getnicename(ion, mathmode=False), color=ioncolors[ion]) for ion in axions[axi]]
        ax.legend(handles=handles, fontsize=fontsize, ncol=3, bbox_to_anchor=(0.0, 1.0), loc='upper left', frameon=False)

    plt.savefig(mdir + 'ionbals_snap{}_HM01_ionizedmu.pdf'.format(snap), format='pdf', bbox_inches='tight')


# column density equivalent width, coldens, N, EW, N-EW, cog, curve of growth
# b, bpar
def plot_NEW(fontsize=fontsize):    
    outname = mdir + 'coldens_EW_sample3-6_ionselsamples_L0100N1504_27_T4EOS.pdf'
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    datafile = datadir + 'sample3-6_coldens_EW_vwindows_subsamples.hdf5'
    
    uselines = {'o7': ild.o7major,\
                'o8': ild.o8doublet,\
                'o6': ild.o6major,\
                'ne8': ild.ne8major,\
                'ne9': ild.ne9major,\
                'fe17': ild.fe17major,\
                }
    Nminmax =  {'o7':   (14.5, 18.3),\
                'o8':   (14.8, 17.5),\
                'o6':   (12.8, 17.0),\
                'ne9':  (14.9, 17.4),\
                'ne8':  (13.5, 16.4),\
                'fe17': (14.2, 16.6),\
                }
    logEWminmax = {'o7':   (0.0, 1.9),\
                   'o8':   (0.0, 1.7),\
                   'o6':   (1.0, 3.2),\
                   'ne9':  (0.0, 1.3),\
                   'ne8':  (1.3, 2.8),\
                   'fe17': (0.0, 1.4),\
                   }
   
    bvals_CIE = {ion: np.sqrt(2. * c.boltzmann * Tmax_CIE[ion] / \
                              (ionh.atomw[string.capwords(ol.elements_ion[ion])] * c.u)) \
                      * 1e-5
                 for ion in Tmax_CIE}
    # v windows
    bfits = {'o6': 28.,\
             'o7': 82.,\
             'o8': 112.,\
             'ne8': 37.,\
             'ne9': 81.,\
             'fe17': 90.,\
             }
    # full sightlines
    #bfits = {'o6': 34.,\
    #         'o7': 90.,\
    #         'o8': 158.,\
    #         'ne8': 41.,\
    #         'ne9': 90.,\
    #         'fe17': 102.,\
    #         }
    # half range sizes
    vwindows_ion = {'o6': 600.,\
                    'ne8': 600.,\
                    'o7': 1600.,\
                    'o8': 1600.,\
                    'fe17': 1600.,\
                    'ne9': 1600,\
                    }
    samplegroups_ion = {ion: '{ion}_selection'.format(ion=ion) \
                              for ion in vwindows_ion}
    
    with h5py.File(datafile, 'r') as df:
        coldens = {}
        EWs = {}
        for ion in ions:
            samplegroup = samplegroups_ion[ion] + '/'
            vwindow = vwindows_ion[ion]
            
            if vwindow is None:
                epath = 'EW_tot/'
                cpath = 'coldens_tot/'
            else:
                spath = 'vwindows_maxtau/Deltav_{dv:.3f}/'.format(dv=vwindow)
                epath = spath + 'EW/'
                cpath = spath + 'coldens/'
            epath = samplegroup + epath
            cpath = samplegroup + cpath
            
            coldens[ion] = np.array(df[cpath + ion])
            EWs[ion] = np.array(df[epath + ion])
    
    bvals_indic = [10., 20., 50., 100., 200.]
    percentiles = [2., 10., 50., 90., 98.]
    logNspacing = 0.1
    linemin = 50
    ncols = 3
    ylabel = r'$\log_{10} \, \mathrm{EW} \; [\mathrm{m\AA}]$'
    xlabel = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    bbox = {'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'none'}
    
    alpha = 0.3
    size_data = 3.
    linewidth = 2.
    path_effects = [mppe.Stroke(linewidth=linewidth, foreground="black"), mppe.Stroke(linewidth=linewidth - 0.5)]
    color_data = ['gray', 'dimgray']
    color_bbkg = 'C5'
    color_lin  = 'forestgreen'
    color_cie  = 'orange'
    color_fit  = 'blue'
    ls_data    = 'solid'
    ls_bbkg    = 'dotted'
    ls_lin     = 'dotted'
    ls_cie     = 'dashed'
    ls_fit     = 'dashdot'
    ftllabel = 'best-fit $b$'
    cielabel = '$b(\\mathrm{{T}}_{{\\max, \\mathrm{{CIE}}}})$'
    linlabel = 'lin. COG'
    bkglabel = 'var. $b \\; [\\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}]$'    
    
    nions = len(ions)
    
    panelwidth = 2.8
    panelheight = 2.8
    legheight = 1.0
    wspace = 0.25
    hspace = 0.2
    nrows = nions // ncols
    
    fig = plt.figure(figsize=(panelwidth * ncols + wspace * (ncols - 1),\
                              panelheight * nrows + hspace * (nrows - 1) +\
                              legheight))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows + 1, wspace=wspace, hspace=hspace, height_ratios=[panelheight] * nrows + [legheight])   
    axes = [fig.add_subplot(grid[ioni // ncols, ioni % ncols]) for ioni in range(nions)]
    lax = fig.add_subplot(grid[nrows, :])
    
    for ioni in range(nions):
        ax = axes[ioni]
        ion = ions[ioni]
        
        pu.setticks(ax, fontsize)
        if ioni % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if ioni // ncols == nrows - 1:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        
        axlabel1 = '$\\Delta v = \\pm {dv:.0f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'
        axlabel2 = '$\\mathrm{{{ion}}}$'
        axlabel1 = axlabel1.format(dv=0.5 * vwindows_ion[ion])
        axlabel2 = axlabel2.format(ion=ild.getnicename(ion, mathmode=True))
        ax.text(0.03, 0.97, axlabel1, fontsize=fontsize - 1,\
                transform=ax.transAxes, verticalalignment='top',\
                horizontalalignment='left',\
                bbox=bbox)
        ax.text(0.03, 0.85, axlabel2, fontsize=fontsize,\
                transform=ax.transAxes, verticalalignment='top',\
                horizontalalignment='left',\
                bbox=None)
        
        fitlabel = 'fit: $b = {b:.0f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'.format(b=bfits[ion])
        ax.text(0.97, 0.03, fitlabel, fontsize=fontsize - 1,\
                transform=ax.transAxes, verticalalignment='bottom',\
                horizontalalignment='right', bbox=bbox)
        
        ax.set_xlim(*Nminmax[ion])
        ax.set_ylim(*logEWminmax[ion])
        
        N = coldens[ion]
        EW = np.log10(EWs[ion]) + 3. # Angstrom -> mA
        Nmin = np.min(N)
        Nmax = np.max(N)
        Nbmin = np.floor(Nmin / logNspacing) * logNspacing
        Nbmax = np.ceil(Nmax / logNspacing) * logNspacing
        Nbins = np.arange(Nbmin, Nbmax + 0.5 * logNspacing, logNspacing)
        Nbinc = Nbins + 0.5 * logNspacing
        Nbinc = np.append([Nbinc[0] - 0.5 * logNspacing], Nbinc)
        
        pvals, outliers, pinds =  pu.get_perc_and_points(N, EW, Nbins,\
                        percentiles=percentiles,\
                        mincount_x=linemin,\
                        getoutliers_y=True, getmincounts_x=True,\
                        x_extremes_only=True)
        pvals = pvals.T
        
        nrange = len(percentiles) // 2
        plotmed  = len(percentiles) % 2
        
        for i in range(nrange):
            pmin = pvals[i][pinds]
            pmax = pvals[len(pvals) - 1 - i][pinds]
            ax.fill_between(Nbinc[pinds], pmin, pmax,\
                            color=color_data[i], alpha=alpha)
        if plotmed:
            pmed = pvals[nrange]
            ax.plot(Nbinc, pmed, linestyle=ls_data,\
                    linewidth=linewidth, color=color_data[0],\
                    path_effects=path_effects)
        ax.scatter(outliers[0], outliers[1], c=color_data[0], alpha=alpha,\
                   s=size_data)
        
        linvals = ild.lingrowthcurve_inv(10**Nbinc, uselines[ion])
        linvals = np.log10(linvals) + 3
        ax.plot(Nbinc, linvals, linestyle=ls_lin, color=color_lin,\
                linewidth=linewidth, label=linlabel)
        
        cievals = ild.linflatcurveofgrowth_inv_faster(10**Nbinc,\
                                               bvals_CIE[ion] * 1e5,\
                                               uselines[ion])
        cievals = np.log10(cievals) + 3
        ax.plot(Nbinc, cievals, linestyle=ls_cie, color=color_cie,\
                linewidth=linewidth, label=cielabel, path_effects=path_effects)
        
        fitvals = ild.linflatcurveofgrowth_inv_faster(10**Nbinc,\
                                               bfits[ion] * 1e5,\
                                               uselines[ion])
        fitvals = np.log10(fitvals) + 3
        ax.plot(Nbinc, fitvals, linestyle=ls_fit, color=color_fit,\
                linewidth=linewidth, label=ftllabel, path_effects=path_effects)
        
        label = True
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xr = xlim[1] - xlim[0]
        yr = ylim[1] - ylim[0]
        for bval in bvals_indic:
            if label:
                _label = bkglabel
            else:
                _label = None
            _vals = ild.linflatcurveofgrowth_inv_faster(10**Nbinc,\
                                               bval * 1e5,\
                                               uselines[ion])
            _vals = np.log10(_vals) + 3
            ax.plot(Nbinc, _vals, linestyle=ls_bbkg, color=color_bbkg,\
                linewidth=linewidth, label=_label, zorder=-1)
            label=False
            
            ## plots are already quit busy, so leave out b labels in the 
            ## cramped top region
            if _vals[-1] < ylim[1]:
                rot = np.tan((_vals[-2] - _vals[-5]) / (Nbinc[-2] - Nbinc[-5])\
                             * xr / yr)
                rot *= 180. / np.pi # rad -> deg
                ax.text(xlim[1] - 0.015 * xr, _vals[-1] - 0.02 * yr,\
                        '{:.0f}'.format(bval),\
                        horizontalalignment='right', verticalalignment='top',\
                        color=color_bbkg, fontsize=fontsize - 2, zorder=-1,\
                        rotation=rot)
            #else:
            #    indcross = np.where(_vals < ylim[1])[0][-1]
            #    xpos = Nbinc[indcross]
            #    ax.text(xpos + 0.02 * xr, ylim[1], '{:.0f}'.format(bval),\
            #            horizontalalignment='left', verticalalignment='top')
            
            
    print('Used b values {} km/s (low-to-high -> bottom to top in plot)'.format(bvals_indic))        
    lax.axis('off')
    hnd, lab = axes[0].get_legend_handles_labels()
    lax.legend(handles=hnd, labels=lab, fontsize=fontsize, loc='lower center',\
                         ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.0))

    plt.savefig(outname, format='pdf', bbox_inches='tight')    

# b, b parameter, 
def plotbpar():
    '''
    best fit b parameters (log EW) as a function of velocity window size
    '''
    datafile = datadir + 'bparfit_data.txt'
    outname = mdir + 'sample3-6_vwindows_effect_bparfit.pdf'
    logfit = True
    if logfit:
        print('Using log EW fits')
    else:
        print('Using (non-log) EW fits')
    
    cosmopars = cosmopars_ea_27
    z = cosmopars['z']
    boxvel = 100. * c.cm_per_mpc / (1. + z) *\
             cu.Hubble(z, cosmopars=cosmopars) * 1e-5
    #set_boxvel = 0.5 * boxvel
    
    
    data = pd.read_csv(datafile, sep='\t', header=0)
    data['ion'] = data['ion'].astype('category')
    data['sightline selection'] = data['sightline selection'].astype('category')
    data.loc[data['Delta v [full, rest-frame, km/s]'] == np.inf, \
             'Delta v [full, rest-frame, km/s]'] = boxvel
    # log/lin fits, ion sample selections
    plotdata = data.loc[data['fit log EW'] == logfit]
    plotdata = plotdata.loc[plotdata['sightline selection'] != 'full_sample']
    plotdata = plotdata.pivot(columns='ion',\
                              index='Delta v [full, rest-frame, km/s]',
                              values='best-fit b [km/s]')
    
    vwindows_ion = {'o6': 600.,\
                    'ne8': 600.,\
                    'o7': 1600.,\
                    'o8': 1600.,\
                    'fe17': 1600.,\
                    'ne9': 1600,\
                    }    
    bvals_CIE = {ion: np.sqrt(2. * c.boltzmann * Tmax_CIE[ion] / \
                              (ionh.atomw[string.capwords(ol.elements_ion[ion])] * c.u)) \
                      * 1e-5
                 for ion in Tmax_CIE}
    
    fontsize = 12
    
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.),\
                           gridspec_kw={'bottom': 0.15})
    ax.set_xlabel('$\\Delta v \\; [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$ (half of total range)', fontsize=fontsize)
    ax.set_ylabel('best-fit $b \\; [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$', fontsize=fontsize)
    size = 30
    marker = 'o'
    alpha = 1.
    
    for ion in plotdata.columns:
        xv = plotdata[ion].index * 0.5 # full width -> half width
        yv = np.array(plotdata[ion])
        label = ild.getnicename(ion, mathmode=False)
        ax.plot(xv, yv, color=ioncolors[ion], linewidth=2, alpha=alpha,\
                label=label)
        ax.scatter(0.5 * vwindows_ion[ion], plotdata[ion][vwindows_ion[ion]],\
                   alpha=alpha, s=size, marker=marker, color=ioncolors[ion])
        ax.scatter(0.5 * boxvel, plotdata[ion][boxvel],\
                   alpha=alpha, s=size, marker='o', edgecolor=ioncolors[ion],\
                   facecolor='none')
        ax.axhline(bvals_CIE[ion], 0.0, 0.07, linestyle='dotted', linewidth=2,\
                   color=ioncolors[ion])
        print(ion)
        print('whole box: b={b} km/s'.format(b=plotdata[ion][boxvel]))
        print('Delta v={dv} km/s: b={b} km/s'.format(dv=vwindows_ion[ion] * 0.5,\
              b=plotdata[ion][vwindows_ion[ion]]))
    #leg_ions = [mlines.Line2D([], [], color=ioncolors[ion], linestyle='None',\
    #                      markersize = 0.1 * size_ionsel,\
    #                      label=ion, marker='o')  \
    #            for ion in ioncolors]
    xlim = list(ax.get_xlim())
    if xlim[0] < 0.:
        xlim[0] = 0.
    ax.set_xlim(*tuple(xlim))
    ylim = list(ax.get_ylim())
    ylim[0] = 0.
    ax.set_ylim(*tuple(ylim))
    
    pu.setticks(ax, fontsize=fontsize)
    #xticks = ax.get_xticks()
    #if set_boxvel in xticks: # just change the label
    #    xlabels = ['{:.0f}'.format(tick) for tick in xticks]
    #    xlabels[np.where(xticks == set_boxvel)[0][0]] = '100 cMpc'
    #else: # insert the label, remove neighboring ticks
    #    insind = np.searchsorted(xticks, set_boxvel)
    #    xticks = np.array(list(xticks[:insind]) + [set_boxvel] + list(xticks[insind: ]))
    #    xlabels = ['{:.0f}'.format(tick) for tick in xticks]
    #    xlabels[insind - 1] = ''
    #    xlabels[insind] = '100 cMpc'
    #    if len(xlabels) > insind + 1:
    #        xlabels[insind + 1] = ''
    #ax.set_xticks(xticks)
    #ax.set_xticklabels(xlabels)    
    
    ax.legend(ncol=3, fontsize=fontsize - 1, loc='upper left', frameon=False)

    plt.savefig(outname, format='pdf', bbox_inches='tight')  

# confusion matrix, stellar mass bins
def plotconfmatrix_mstarmhalo(fontsize=fontsize):
    outname = mdir + 'confusion_matrix_mstar_mhalo_bins.pdf'
    
    halocat = datadir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    #binround = 0.1
    m200cbins = np.array([-np.inf, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14.0, 14.6])
    mstarbins = np.array([-np.inf, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.6])
        
    with h5py.File(halocat, 'r') as cat:
        m200c = np.log10(np.array(cat['M200c_Msun']))
        mstar = np.log10(np.array(cat['Mstar_Msun']))
    
    #m200cbins = np.array(m200cbins)
    #expand = (np.floor(np.min(m200c) / binround) * binround, np.ceil(np.max(m200c) / binround) * binround)
    #m200cbins = np.append(expand[0], m200cbins)
    #m200cbins = np.append(m200cbins, expand[1])
        
    xycounts, xe, ye = np.histogram2d(m200c, mstar, bins=[m200cbins, mstarbins])

    xynorm = xycounts.astype(float) / np.sum(xycounts, axis=0)[np.newaxis, :]
    cmap = 'viridis'
    vmin = 0.
    vmax = 1.
    fontsize = 12
    xmin = 9.0
    ymin = 7.5
    xlabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    ylabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{\star}} \; [\mathrm{M}_{\odot}]$'
    clabel = r'fraction at fixed $\mathrm{M}_{\star}$'
    
    plotbins_x = np.copy(m200cbins)
    plotbins_x[0] = max(plotbins_x[0], xmin)
    plotbins_y = np.copy(mstarbins)
    plotbins_y[0] = max(plotbins_y[0], ymin)
    
    fig = plt.figure(figsize=(5.5, 5.0))
    grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[10., 1.],\
                        bottom=0.15, left=0.15, right=0.87)
    ax = fig.add_subplot(grid[0])
    cax = fig.add_subplot(grid[1])
    
    img = ax.pcolormesh(plotbins_x, plotbins_y, xynorm.T, cmap=cmap,\
                        vmin=vmin, vmax=vmax, rasterized=True)
    
    ax.set_xticks(m200cbins[1:])
    ax.set_yticks(mstarbins[1:])
    ax.set_xlim((plotbins_x[0], plotbins_x[-1]))
    ax.set_ylim((plotbins_y[0], plotbins_y[-1]))
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1, which='both')
    for label in ax.get_xmajorticklabels():
        label.set_rotation(45)

    ylim = ax.get_ylim() 
    offset = 0.025 * (ylim[1] - ylim[0])    
    for i in range(len(plotbins_x) - 1):
        for j in range(len(plotbins_y) - 1):
            xcoord = 0.5 * (plotbins_x[i] + plotbins_x[i + 1])
            ycoord = 0.5 * (plotbins_y[j] + plotbins_y[j + 1]) - 0.1 * offset
            num = int(xycounts[i, j])
            ofs = 0
            #if num > 1000:
            #    rotation = 90
            #else:
            #    rotation = 0
            if (num >= 1000) or\
               ((xycounts[max(i - 1, 0), j] >= 1000 or xycounts[min(i + 1, xycounts.shape[0] - 1), j] >= 1000) and num > 10) or\
               (xycounts[max(i - 1, 0), j] >= 10000 or xycounts[min(i + 1, xycounts.shape[0] - 1), j] >= 10000) or \
               (num >= 100 and (xycounts[max(i - 1, 0), j] >= 100 or xycounts[min(i + 1, xycounts.shape[0] - 1), j] >= 100)):
                ofs = offset if i % 2 else -1 * offset
            ax.text(xcoord, ycoord + ofs, num, fontsize=fontsize - 1.,\
                    color='white', rotation=0.,\
                    horizontalalignment='center', verticalalignment='center', 
                    path_effects = [mppe.Stroke(linewidth=1.5, foreground='black'), mppe.Normal()] )
    plt.colorbar(img, cax=cax, orientation='vertical')
    cax.set_ylabel(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1, which='both')
    cax.set_aspect(10.)
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
    
############################# get numbers #####################################
    
def get_binmedians(binarr='M200c_Msun', binedges=mass_edges_standard,\
                   takelogarr=True):
    '''
    get median M200c in M200c bins
    '''
    catfn = datadir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    
    with h5py.File(catfn, 'r') as fc:
        arr = fc[binarr]
        if takelogarr:
            arr = np.log10(arr)
        arr = np.sort(arr)
        edgeinds = np.searchsorted(arr, binedges)
        edgeinds = np.append([0], edgeinds)
        edgeinds = np.append(edgeinds, [len(arr)])
        binedges = [-np.inf] + list(binedges) + [np.inf]
        print(binedges)
        print(edgeinds)
        
        for bi in range(len(binedges) - 1):
            _slice = slice(edgeinds[bi], edgeinds[bi + 1])
            _range = (binedges[bi], binedges[bi + 1])
            _med = np.median(arr[_slice])
            print('{qty}: {rng} -> median {med}'.format(qty=binarr,\
                  rng=_range, med=_med))

def calc_deltav_xray(width=2.5, z=0.1):
    '''
    for a width in eV, calculate width in rest-frame km/s for the X-ray ions
    '''
    wls = {'o7': ild.o7major.lambda_angstrom,\
           'o8': (ild.o8minor.lambda_angstrom * ild.o8minor.fosc + \
                  ild.o8major.lambda_angstrom * ild.o8major.fosc) / \
                 (ild.o8minor.fosc + ild.o8major.fosc),\
           'ne9': ild.ne9major.lambda_angstrom,\
           'fe17': ild.fe17major.lambda_angstrom,\
          }
    res = {ion: width * c.ev_to_erg / (1. + z) / \
                 (c.c * c.planck / (wls[ion] * 1e-8)) \
                  * c.c / 1e5 \
                 for ion in wls}
    print('Wavelenghts (A)')
    print(wls)
    print('Delta v (km/s)')
    print(res)


def get_dNdz_halos(limset='break'):   
    '''
    get number of absorbers >N for the different halo sets from masked and
    FoF-only CDDFs. Threasholds N are set by 
    limset: 'break' (CDDF break) or 'obs' (estimated observation limits)
    '''

    if limset == 'break':
        lims = {'o6':   [14.3],\
                'ne8':  [13.7],\
                'o7':   [16.0],\
                'ne9':  [15.3],\
                'o8':   [16.0],\
                'fe17': [15.0]}
    elif limset == 'obs':
        lims = {'o6':   [13.5],\
                'ne8':  [13.5],\
                'o7':   [15.5],\
                'ne9':  [15.5],\
                'o8':   [15.7],\
                'fe17': [14.9]}

    ions = ['o6', 'ne8', 'o7', 'ne9', 'o8', 'fe17']
    medges = np.arange(10., 14.1, 0.5) #np.arange(11., 14.1, 0.5)
    halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
    prefilenames_all = {key: ['coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in ions}   
    filenames_all = {key: [datadir + 'cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in prefilenames_all[key]] for key in prefilenames_all.keys()}
    
    masses_proj = ['none'] + list(medges)
    filedct = {ion: {masses_proj[i]: filenames_all[ion][i] for i in range(len(filenames_all[ion]))} for ion in ions} 
    
    masknames =  ['nomask',\
                  #'logM200c_Msun-9.0-9.5',\
                  #'logM200c_Msun-9.5-10.0',\
                  'logM200c_Msun-10.0-10.5',\
                  'logM200c_Msun-10.5-11.0',\
                  'logM200c_Msun-11.0-11.5',\
                  'logM200c_Msun-11.5-12.0',\
                  'logM200c_Msun-12.0-12.5',\
                  'logM200c_Msun-12.5-13.0',\
                  'logM200c_Msun-13.0-13.5',\
                  'logM200c_Msun-13.5-14.0',\
                  'logM200c_Msun-14.0-inf',\
                  ]
    maskdct = {masses_proj[i]: masknames[i] for i in range(len(masknames))}
    
    ## read in cddfs from halo-only projections
    dct_fofcddf = {}
    for ion in ions:
        dct_fofcddf[ion] = {}
        # FoF CDDFs: only without masks
        mmass = 'none'
        dct_fofcddf[ion] = {}
        for pmass in masses_proj:
            try:
                with h5py.File(filedct[ion][pmass]) as fi:
                    #print(filedct[ion][pmass])
                    try:
                        bins = np.array(fi['bins/axis_0'])
                    except KeyError as err:
                        print('While trying to load bins in file %s\n:'%(filedct[pmass]))
                        raise err
                        
                    dct_fofcddf[ion]['bins'] = bins
                    
                    inname = np.array(fi['input_filenames'])[0].decode()
                    inname = inname.split('/')[-1] # throw out directory path
                    parts = inname.split('_')
            
                    numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                    numpix_1sl.remove(None)
                    numpix_1sl = int(list(numpix_1sl)[0][:-3])
                    print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                    
                    grp = fi[maskdct[mmass]]
                    hist = np.array(grp['hist'])
                    covfrac = grp.attrs['covfrac']
                    # recover cosmopars:
                    cosmopars = cosmopars_ea_27
                    dztot = cu.getdz(cosmopars['z'],\
                                     cosmopars['boxsize'] / cosmopars['h'],\
                                     cosmopars=cosmopars) *\
                            float(numpix_1sl**2)
                    #dXtotdlogN = dXtot * np.diff(bins)
                    dct_fofcddf[ion][pmass] = {'cddf': hist / dztot, 'covfrac': covfrac}
            
            except IOError as err:
                print('Failed to read in %s; stated error:'%filedct[pmass])
                print(err)
                
    ## read in split cddfs from total ion projections
    ion_filedct_excl_1R200c_cenpos = {'fe17': datadir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  datadir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  datadir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   datadir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   datadir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   datadir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    
    dct_maskcddf = {}
    for ion in ions:
        file_allproj = ion_filedct_excl_1R200c_cenpos[ion]
        dct_maskcddf[ion] = {}
        with h5py.File(file_allproj) as fi:
            #print(file_allproj)
            try:
                bins = np.array(fi['bins/axis_0'])
            except KeyError as err:
                print('While trying to load bins in file %s\n:'%(file_allproj))
                raise err
                
            dct_maskcddf[ion]['bins'] = bins
            
            inname = np.array(fi['input_filenames'])[0].decode()
            inname = inname.split('/')[-1] # throw out directory path
            parts = inname.split('_')
        
            numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
            numpix_1sl.remove(None)
            numpix_1sl = int(list(numpix_1sl)[0][:-3])
            print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
            
            for mmass in masses_proj[1:]:
                grp = fi[maskdct[mmass]]
                hist = np.array(grp['hist'])
                covfrac = grp.attrs['covfrac']
                # recover cosmopars:
                mask_examples = {key: item for (key, item) in grp.attrs.items()}
                del mask_examples['covfrac']
                example_key = list(mask_examples.keys())[0] # 'mask_<slice center>'
                example_mask = mask_examples[example_key].decode() # '<dir path><mask file name>'
                path = 'masks/%s/%s/Header/cosmopars'%(example_key[5:], example_mask.split('/')[-1])
                cosmopars = {key: item for (key, item) in fi[path].attrs.items()}
                dXtot = cu.getdX(cosmopars['z'],\
                                 cosmopars['boxsize'] / cosmopars['h'],\
                                 cosmopars=cosmopars) * \
                        float(numpix_1sl**2)
                #dXtotdlogN = dXtot * np.diff(bins)
            
                dct_maskcddf[ion][mmass] = {'cddf': hist / dXtot, 'covfrac': covfrac}
            # use cosmopars from the last read mask
            mmass = 'none'
            grp = fi[maskdct[mmass]]
            hist = np.array(grp['hist'])
            covfrac = grp.attrs['covfrac']
            # recover cosmopars:
            dztot = cu.getdz(cosmopars['z'],\
                             cosmopars['boxsize'] / cosmopars['h'],\
                             cosmopars=cosmopars) *\
                    float(numpix_1sl**2)
            #dXtotdlogN = dXtot * np.diff(bins)
            dct_maskcddf[ion][mmass] = {'cddf': hist / dztot, 'covfrac': covfrac}
    
    results = {}
    for ion in ions:
       results[ion] = {}
       bins_mask = dct_maskcddf[ion]['bins']
       bins_fof  = dct_fofcddf[ion]['bins']
       
       for mmass in masses_proj:
           cumul_fof =  np.cumsum(dct_fofcddf[ion][mmass]['cddf'][::-1])[::-1]
           val_fof  = pu.linterpsolve(bins_fof[:-1], cumul_fof, lims[ion][0])
           results[ion][mmass] = {'fof': val_fof}
               
           cumul_mask = np.cumsum(dct_maskcddf[ion][mmass]['cddf'][::-1])[::-1]
           val_mask = pu.linterpsolve(bins_mask[:-1], cumul_mask, lims[ion][0])
           fcov_mask = dct_maskcddf[ion][mmass]['covfrac']
           
           results[ion][mmass].update({'mask': val_mask, 'fcov': fcov_mask})

    strings_mmass = {'none': 'total'}
    strings_mmass.update({mmass: '{Mmin:.1f}--{Mmax:.1f}'.format(Mmin=mmass, Mmax=mmass + 0.5) \
                                 if mmass < 13.9 else \
                                 '$>{Mmin:.1f}$'.format(Mmin=mmass) \
                                 for mmass in masses_proj[1:]})
    
    ionstr = {ion: '\\ion{{{}}}{{{}}}'.format(ild.getnicename(ion).split(' ')[0], \
                                              (ild.getnicename(ion).split(' ')[1]).lower()) \
              for ion in ions}
    
    ### print overview table
    print('FoF-only CDDFs (total = all haloes)')
    print('dn(>N, halo) / dz')
    topstr = '$\\mathrm{{M}}_{{200\\mathrm{{c}}}} $ & ' + \
             ' & '.join(['{ion}'.format(ion=ionstr[ion]) for ion in ions]) + ' \\\\'
    topstr2 = '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\odot}}$ & ' +\
              ' & '.join(['$>{yval:.1f}$'.format(yval=lims[ion][0]) for ion in ions]) + ' \\\\'
    fillstr = '{massst} & ' + ' & '.join(['{{{ist}:.3f}}'.format(ist=ion) for ion in ions]) + ' \\\\'
    #resstr = 'total &  ' +  ' & '.join(['{{{ist}:.4f}}'.format(ist=ion) for ion in ions]) + ' \\\\'

    print(topstr)
    print(topstr2)
    print('\\hline')
    for Mmin in masses_proj:
        print(fillstr.format(massst=strings_mmass[Mmin],\
                             **{ion: results[ion][Mmin]['fof'] for ion in ions}))
    print('\n')
    print('R200c mask CDDFs (total = all gas)')
    print(topstr)
    print(topstr2)
    print('\\hline')
    for Mmin in masses_proj:
        print(fillstr.format(massst=strings_mmass[Mmin],\
                             **{ion: results[ion][Mmin]['mask'] for ion in ions}))    
    print('\n')
    print('R200c mask CDDFs as a fraction of all gas')
    print(topstr)
    print(topstr2)
    print('\\hline')
    for Mmin in masses_proj[1:]:
        print(fillstr.format(massst=strings_mmass[Mmin],\
                             **{ion: results[ion][Mmin]['mask'] / results[ion]['none']['mask'] for ion in ions}))
    print(fillstr.format(massst='halo total',\
                             **{ion: np.sum([results[ion][Mmin]['mask'] for Mmin in masses_proj[1:]]) \
                                     / results[ion]['none']['mask'] for ion in ions}))
    
    print('\n')
    print('FoF all halos / total CDDF')
    print(topstr)
    print(topstr2)
    print('\\hline')
    print(fillstr.format(massst='all FoF / all gas',\
                             **{ion: results[ion]['none']['fof'] \
                                / results[ion]['none']['mask'] for ion in ions}))
    

def getcovfrac_total_halo(minr_pkpc, maxr_r200c):
    '''
    for an ion, get the fraction of the total halo covering fraction (<maxr) 
    and the total cddf contribution (< minr, <maxr) for absorber above the 
    cddf break, assuming no halo overlaps 
    
    '''
    fcovset = 'break'
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    
    halocat='/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    # rounded1 set used in the paper
    Ms_edges = np.array([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.7], dtype=np.float32)

    with h5py.File(halocat, 'r') as cat:
        r200c = np.array(cat['R200c_pkpc'])
        mstar = np.log10(np.array(cat['Mstar_Msun']))
        minds = np.digitize(mstar, Ms_edges)
        rmeds = [np.median(r200c[minds == i]) for i in range(1, len(Ms_edges))] # digitize 0 -> < first bin
        
    if fcovset == 'break':
        ytype='fcov'
        yvals_toplot = {'o6':   [14.3],\
                        'ne8':  [13.7],\
                        'o7':   [16.0],\
                        'ne9':  [15.3],\
                        'o8':   [16.0],\
                        'fe17': [15.0]}
    elif fcovset == 'obs':
        ytype='fcov'
        yvals_toplot = {'o6':   [13.5],\
                        'ne8':  [13.5],\
                        'o7':   [15.5],\
                        'ne9':  [15.5],\
                        'o8':   [15.7],\
                        'fe17': [14.9]}

    cosmopars = cosmopars_ea_27

    ion_filedct_Mstar = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex_centrals_fullrdist_stored_profiles.hdf5',\
                         }
    ion_filedct_Mstar_lowmass = {'fe17': 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne9':  'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'ne8':  'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'o8':   'rdist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'o7':   'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         'o6':   'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-99p-3R200c_Mstar-0p5dex-lowmass_centrals_fullrdist_stored_profiles.hdf5',\
                         }

    # define used mass ranges
    Ms_mins = list(Ms_edges[:-1])
    Ms_maxs = list(Ms_edges[1:]) 
    #Ms_base = ['geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins, Ms_maxs)]
    Ms_names_hm = ['logMstar_Msun_1000_geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins[2:], Ms_maxs[2:])]
    Ms_names_lm = ['logMstar_Msun_1000_geq%.1f_le%.1f'%(smin, smax) for smin, smax in zip(Ms_mins[:2], Ms_maxs[:2])]
    Ms_names = Ms_names_lm + Ms_names_hm
    #Ms_sels = [('logMstar_Msun', Ms_mins[i], Ms_maxs[i]) if Ms_maxs[i] is not None else\
    #           ('logMstar_Msun', Ms_mins[i], np.inf)\
    #           for i in range(len(Ms_mins))]
    #galsetnames_smass = {name: sel for name, sel in zip(Ms_names, Ms_sels)}
        
    # R200c to assume for each M* bin   
    R200c_toassume = {name: val for name, val in zip(Ms_names, rmeds)}
    
    techvars = {0: {'filenames': ion_filedct_Mstar, 'setnames': Ms_names_hm,\
                    'setfills': None, 'units': 'pkpc'},\
                1: {'filenames': ion_filedct_Mstar_lowmass,\
                    'setnames': Ms_names_lm,\
                    'setfills': None, 'units': 'pkpc'},\
                }
           
    yvals_label_ion = {tv: yvals_toplot for tv in techvars}
    yvals, bins, numgals = readin_radprof(techvars, yvals_label_ion,\
                   labels=None, ions_perlabel=None, ytype=ytype,\
                   datadir=datadir, binset=0, units='pkpc')
    
    # techvars here are just different files for different masses -> combine
    _yvals = yvals[0].copy()
    [_yvals[ion].update(yvals[1][ion]) for ion in _yvals]
    yvals = _yvals
    
    _bins = bins[0].copy()
    [_bins[ion].update(bins[1][ion]) for ion in _bins]
    bins = _bins
  
    npix_rminmax = {}
    # size of one pixel (pkpc)
    pixsize = (100. / 32000.) * 1e3 * cosmopars['a']
    for ionind in range(len(ions)):   
        ion = ions[ionind]
        tags = Ms_names
        # example tag 'logMstar_Msun_1000_geq%.1f_le%.1f'
        tags = sorted(tags, key=lambda x: float(x.split('_')[3][3:]))
        #print(tags)
        npix_rminmax[ion] = {}
        
        for ti in range(len(tags)): # mass bins
            #print(tag)
            tag = tags[ti]
            npix_rminmax[ion][tag] = {}         
            try:
                rvals = np.array(bins[ion][tag])
            except KeyError: # dataset not read in
                print('Could not find ion %s, tag %s'%(ion, tag))
                continue
            #rtag = '_'.join(tag.split('_')[-2:])
            basesize = R200c_toassume[tag]
            maxr_pkpc = maxr_r200c * basesize
            fcovs = np.array(yvals[ion][tag][yvals_toplot[ion][0]])
            
            npix_r = fcovs * np.pi * (rvals[1:]**2 - rvals[:-1]**2 ) / pixsize**2 # fraction * annulus surface area / pixel size    
            npix_inr = np.cumsum(npix_r)
            print(rvals)
            print(minr_pkpc)
            print(maxr_pkpc)
            npix_inrmin = pu.linterpsolve(rvals[1:], npix_inr, minr_pkpc)
            npix_inrmax = pu.linterpsolve(rvals[1:], npix_inr, maxr_pkpc)
            
            npix_rminmax[ion][tag].update({'maxr_pkpc': maxr_pkpc,\
                        'npix_perhalo_inrmin': npix_inrmin,\
                        'npix_perhalo_inrmax': npix_inrmax})       
    
    ##### read in total cddfs
    ion_filedct_excl_1R200c_cenpos = {'fe17': datadir + 'cddf_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne9':  datadir + 'cddf_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'ne8':  datadir + 'cddf_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o8':   datadir + 'cddf_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o7':   datadir + 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      'o6':   datadir + 'cddf_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5',\
                                      }
    
    techvars = {0: ion_filedct_excl_1R200c_cenpos}
    
    masknames = ['nomask']

    hists = {}
    cosmopars = {}
    dXtot = {}
    dztot = {}
    dXtotdlogN = {}
    bins = {}
    
    for var in techvars:
        hists[var] = {}
        cosmopars[var] = {}
        dXtot[var] = {}
        dztot[var] = {}
        dXtotdlogN[var] = {}
        bins[var] = {}
        for ion in ions:
            print('Reading in data for ion %s'%ion)
            filename = techvars[var][ion]
            with h5py.File(filename, 'r') as fi:
                _bins = np.array(fi['bins/axis_0'])
                # handle +- infinity edges for plotting; should be outside the plot range anyway
                if _bins[0] == -np.inf:
                    _bins[0] = -100.
                if _bins[-1] == np.inf:
                    _bins[-1] = 100.
                bins[var][ion] = _bins
                
                # extract number of pixels from the input filename, using naming system of make_maps
                inname = np.array(fi['input_filenames'])[0].decode()
                inname = inname.split('/')[-1] # throw out directory path
                parts = inname.split('_')
        
                numpix_1sl = set(part if 'pix' in part else None for part in parts) # find the part of the name needed: '...pix'
                numpix_1sl.remove(None)
                numpix_1sl = int(list(numpix_1sl)[0][:-3])
                print('Using %i pixels per side for the sample size'%numpix_1sl) # needed for the total path length
                
                ionind = 1 + np.where(np.array([part == 'coldens' for part in parts]))[0][0]
                ion = parts[ionind]
                
                masks = masknames
        
                hists[var][ion] = {mask: np.array(fi['%s/hist'%mask]) for mask in masks}
                #print('ion %s: sum = %f'%(ion, np.sum(hists[var][ion]['nomask'])))
                
                examplemaskdir = list(fi['masks'].keys())[0]
                examplemask = fi['masks/%s'%(examplemaskdir)].keys()[0].decode()
                cosmopars[var][ion] = {key: item for (key, item) in fi['masks/%s/%s/Header/cosmopars/'%(examplemaskdir, examplemask)].attrs.items()}
                dXtot[var][ion] = cu.getdX(cosmopars[var][ion]['z'],\
                     cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'],\
                     cosmopars=cosmopars[var][ion]) *\
                     float(numpix_1sl**2)
                dztot[var][ion] = cu.getdz(cosmopars[var][ion]['z'], \
                     cosmopars[var][ion]['boxsize'] / cosmopars[var][ion]['h'],\
                     cosmopars=cosmopars[var][ion]) *\
                     float(numpix_1sl**2)
                dXtotdlogN[var][ion] = dXtot[var][ion] * np.diff(bins[var][ion])
    
    npix_overlim_all = {}
    var_tot = 0
    totpix = 16 * 32000**2 # checked hist sum
    for ion in ions:
        lim = yvals_toplot[ion][0]
        edges = bins[var_tot][ion]
        counts = hists[var_tot][ion]['nomask']
        cumulcounts = np.cumsum(counts[::-1])[::-1]
        npix_overlim_all[ion] = pu.linterpsolve(edges[:-1], cumulcounts, lim)
        print('{ion} fcov total: {fcov}'.format(ion=ion, fcov=npix_overlim_all[ion] / totpix))
    print('\n')
    
    overview = {}
    for ion in ions:
        tags = list(npix_rminmax[ion].keys())
        minmaxv = np.array([[float(x.split('_')[3][3:]), float(x.split('_')[4][2:])] for x in tags])
        _ind = np.argsort(minmaxv[:, 0])
        tags = np.array(tags)[_ind]
        minmaxv = minmaxv[_ind, :]
        overview[ion] = {'limval': yvals_toplot[ion][0]}
        
        #print('For ion {ion}, absorbers > {val}'.format(ion=ion, val=yvals_toplot[ion]))
        for ti in range(len(tags)):
            Mmin, Mmax = minmaxv[ti]
            tag = tags[ti]
            numgals = np.sum(np.logical_and(mstar >= Mmin, mstar < Mmax))
            
            inmin_over_inmax = npix_rminmax[ion][tag]['npix_perhalo_inrmin'] /\
                               npix_rminmax[ion][tag]['npix_perhalo_inrmax']
            inmax_over_total = npix_rminmax[ion][tag]['npix_perhalo_inrmax'] \
                               * numgals / npix_overlim_all[ion]
            rmax = npix_rminmax[ion][tag]['maxr_pkpc']
            
            overview[ion][Mmin] = {'Mmin': Mmin, 'Mmax': Mmax, 'rmax': rmax,\
                                   'inmin_over_inmax': inmin_over_inmax,\
                                   'inmax_over_total': inmax_over_total}
            #print('In Mstar range {Mmin:4.1f}-{Mmax:4.1f}:'.format(Mmin=Mmin, Mmax=Mmax))
            #print('    < {rmin:6.3f} pkpc / < {rmax:6.3f} pkpc: {corefrac}'.format(rmin=minr_pkpc, rmax=rmax, corefrac=inmin_over_inmax))
            #print('    < {rmax:6.3f} pkpc / total: {halofrac}'.format(rmax=rmax, halofrac=inmax_over_total))
            #print('{num} galaxies'.format(num=numgals))
        #print('\n')
    #print([set(overview[ion].keys()) == set(overview[ions[0]].keys()) for ion in ions])
    
    ### print overview table
    print('For an inner radius of {minr} pkpc'.format(minr=minr_pkpc))
    print('Core contribution / total halo')
    topstr = '$\\mathrm{{M}}_{{\\star}}$' +\
             ' & $\\mathrm{{r}}_{{\\perp, \\max}}$ &' +\
             ' & '.join(['{ion}'.format(ion=ild.getnicename(ion)) for ion in ions]) + ' \\\\'
    topstr2 = '$\\log_{{10}} \\, \\mathrm{{M}}_{{\\odot}}$' +\
              ' & $\\mathrm{{pkpc}}$ &' +\
              ' & '.join(['$>{yval:.1f}$'.format(yval=yvals_toplot[ion][0]) for ion in ions]) + ' \\\\'
    fillstr = '{Mmin:.1f}--{Mmax:.1f} & {rmax:.0f} & ' + ' & '.join(['{{{ist}:.4f}}'.format(ist=ion) for ion in ions]) + ' \\\\'
    totstr = 'total & \t & ' +  ' & '.join(['{{{ist}:.4f}}'.format(ist=ion) for ion in ions]) + ' \\\\'
    Mvals = list(overview[ions[0]].keys())
    Mvals.remove('limval')
    stion = ions[0]
    print(topstr)
    print(topstr2)
    for Mmin in sorted(Mvals):
        print(fillstr.format(Mmin=overview[stion][Mmin]['Mmin'],\
                             Mmax=overview[stion][Mmin]['Mmax'],\
                             rmax=overview[stion][Mmin]['rmax'],\
                             **{ion: overview[ion][Mmin]['inmin_over_inmax'] for ion in ions}))
    print('\n')
    print('Halo contribution / total')
    print(topstr)
    print(topstr2)
    for Mmin in sorted(Mvals):
        print(fillstr.format(Mmin=overview[stion][Mmin]['Mmin'],\
                             Mmax=overview[stion][Mmin]['Mmax'],\
                             rmax=overview[stion][Mmin]['rmax'],\
                             **{ion: overview[ion][Mmin]['inmax_over_total'] for ion in ions}))
    print(totstr.format(**{ion: np.sum([overview[ion][Mmin]['inmax_over_total'] for Mmin in Mvals]) for ion in ions}))