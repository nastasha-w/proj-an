#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:48:47 2020

@author: Nastasha
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import string

import eagle_constants_and_units as c
import cosmo_utils as cu
import make_maps_opts_locs as ol
import plot_utils as pu
import ion_line_data as ild

datadir_TOP = '/Users/Nastasha/phd/tables/opacity_project/'

def parse_linename_lambda(name):
    '''
    extract wavelength (A) from cloudy line names
    
    default is Angstrom (A), m means micron, c means cm (hazy 7.02 pg 137)
    '''
    wlpart = name.split(' ')[-1]
    if wlpart[-1] == 'A':
        return float(wlpart[:-1])
    elif wlpart[-1] == 'm':
        return float(wlpart[:-1]) * 1e4 #1 micron = 1e-6 m = 1e4 A
    elif wlpart[-1] == 'c':
        return float(wlpart[:-1]) * 1e8 #(1 cm = 1e8 A)

def parse_linename_ion(name):
    '''
    extract ion (element abbreviation, number tuple) from cloudy line names
    '''
    eltpart = name[:2].strip()
    stagepart = int(name[2:4])
    return (eltpart, stagepart)

def ion_from_line(line):
    '''
    get ion in 'o7', 'fe17' format from line in 'o7r', 'o8', 'fe17-other1' 
    format
    '''
    # match the ion
    ion = ''
    while not line[0].isdigit():
        ion += line[0]
        line = line[1:]
    while line[0].isdigit():
        ion += line[0]
        if len(line) == 1:
            break
        line = line[1:]
    return ion

def findbrightlines(z, keVmin=0.3, logemmin=-25.5, Tmin=5., Tmax=7.,\
                    rangemin=0.1):
    '''
    find bright lines usind redshift z tables
    
    brightness is calculated for CIE, so z shouldn't actually matter
    
    inputs:
    -------
    z         redshift (float)
    eVmin     minimum line energy in keV (float)
    logemmin  minimum max line emissitvity (float, log10 erg cm3)
    Tmin      minimum CIE peak temperature (float, log10 K)
    Tmax      maximum CIE peak temperature (float, log10 K)
    rangemin  fraction of the maxmimum emissivity to take the T range over
    outputs:
    --------
    list of lines and list of indices selected
    prints a LaTeX-ready table of the lines, their wavelengths, and 
      temperature ranges
    '''
    
    zname = ol.zopts[0]
    
    elements = ['carbon', 'nitrogen', 'oxygen', 'neon',\
                'magnesium', 'silicon', 'iron']
    # ion, wavelength, energy, max. em, Tmax, Trange
    topline = '\\begin{tabular}{l l l l l c}'
    header = 'ion & $\\lambda$ & E &' + \
             '$\\max \\, \\Lambda \,\\mathrm{n}_\\mathrm{H}^{-2} \\mathrm{V}^{-1}$' +\
             ' & $\\mathrm{T}_{\\mathrm{peak}}$ & $\\mathrm{T}_{0.1 \\times \\mathrm{peak}}$ \\\\'
    header2 = ' & $\\Ang$ & $\\log_{10} \\, \\mathrm{erg} \\, \\mathrm{cm}^{3}$' + \
              ' & keV & $\\log_{10}$~K & \\ $\\log_{10}$~K \\\\'
    print(topline)
    print('\\hline')
    print(header)
    print(header2)          
    print('\\hline')
    
    fillstr = '{ion} & {wl} & {E} & {maxem} & {Tmax} & {Trange} \\\\'
    for elt in elements:
        tablefilename = ol.dir_emtab%zname + elt + '.hdf5'
        with h5py.File(tablefilename, "r") as tablefile:
            lines = [line.decode() for line in tablefile['lambda'][:]] 
            wls_A = np.array([parse_linename_lambda(line) for line in lines]) 
        wls_keV = c.planck * c.c / (wls_A * 1e-8) / (c.ev_to_erg * 1e3)
        lsel = wls_keV >= keVmin
        wls_A = wls_A[lsel]
        wls_keV = wls_keV[lsel]
        lines = np.array(lines)[lsel]
        # table dimensions: T, nH, line
        emdenssq, logTK, lognHcm3 = cu.findemtables(elt, z) 
        emdenssq = emdenssq[:, :, lsel]
        emdenssq_cie = emdenssq[:, -1, :]
        del emdenssq
        
        nHused = lognHcm3[-1]
        print('Using log nH / cm3 = {nH:.2f} for {elt}'.format(nH=nHused,\
              elt=elt))
        
        lsel2 = np.any(emdenssq_cie >= logemmin, axis=0)
        emdenssq_cie = emdenssq_cie[:, lsel2]
        lines = lines[lsel2]
        wls_A = wls_A[lsel2]
        wls_keV = wls_keV[lsel2]
        
        Tmis = np.argmax(emdenssq_cie, axis=0)
        Tmaxs = logTK[Tmis]
        lsel3 = np.logical_and(Tmaxs >= Tmin, Tmaxs <= Tmax)
        
        emdenssq_cie = emdenssq_cie[:, lsel3]
        lines = lines[lsel3]
        wls_A = wls_A[lsel3]
        wls_keV = wls_keV[lsel3]
        Tmis = Tmis[lsel3]
        Tmaxs = Tmaxs[lsel3]
        
        emmins = emdenssq_cie[Tmis, np.arange(emdenssq_cie.shape[1])] + np.log10(rangemin)
        Tranges = [pu.find_intercepts(emdenssq_cie[:, i], logTK, emmins[i])\
                   for i in range(emmins.shape[0])]
        
        numdig_wl = 4
        for li in range(len(lines)):
            # ion, wavelength, energy, max. em, Tmax, Trange
            elt_num = parse_linename_ion(lines[li])
            ionname = '\\ion{{{elt}}}{{{num}}}'.format(elt=elt_num[0],\
                              num=(ild.arabic_to_roman[elt_num[1]]).lower())
            wl_A = wls_A[li]
            toround = (numdig_wl) - int(np.log10(wl_A) + 1.) # int applies floor, 4 sig. digits
            wlname = str(np.round(wl_A, toround))
            if len(wlname) > numdig_wl + 1 and wlname[-2:] == '.0':
                wlname = wlname[:-2]
            if '.' in wlname and len(wlname) < numdig_wl + 1:
                wlname = wlname + '0' * (numdig_wl + 1 - len(wlname))
            if wlname.startswith('0.') and len(wlname) < numdig_wl + 2:
                wlname = wlname + '0' * (numdig_wl + 2 - len(wlname))
                
            E_keV = wls_keV[li]
            toround = (numdig_wl) - int(np.log10(E_keV) + 1.) # int applies floor, 4 sig. digits
            Ename = str(np.round(E_keV, toround))
            if len(Ename) > numdig_wl + 1 and Ename[-2:] == '.0':
                Ename = Ename[:-2]
            if '.' in Ename and len(Ename) < numdig_wl + 1:
                Ename = Ename + '0' * (numdig_wl + 1 - len(Ename))
            if Ename.startswith('0.') and len(Ename) < numdig_wl + 2:
                Ename = Ename + '0' * (numdig_wl + 2 - len(Ename))
                
            maxem = str(np.round(emdenssq_cie[Tmis[li], li], 1))
            
            Tpeak = str(Tmaxs[li]) # 1 sig. dig.
            
            Trange = '{mi:.1f}--{ma:.1f}'.format(mi=Tranges[li][0],\
                      ma=Tranges[li][1])
            
            print(fillstr.format(ion=ionname, wl=wlname, E=Ename, maxem=maxem,\
                                 Tmax=Tpeak, Trange=Trange))
    print('\\hline')    
    print('\\end{tabular}')
    

def printiondata(ion, z=0.0, rangemin=0.1):
    '''
    print line data for a given ion lines using redshift z tables
    
    brightness is calculated for CIE, so z shouldn't actually matter
    
    inputs:
    -------
    ion       ion to get the line data for
    z         redshift (float), used to get table data (shouldn't matter)
    eVmin     minimum line energy in keV (float)
    logemmin  minimum max line emissitvity (float, log10 erg cm3)
    Tmin      minimum CIE peak temperature (float, log10 K)
    Tmax      maximum CIE peak temperature (float, log10 K)
    rangemin  fraction of the maxmimum emissivity to take the T range over
    
    outputs:
    --------
    list of lines and list of indices selected
    prints a LaTeX-ready table of the lines, their wavelengths, and 
      temperature ranges
    '''
    
    zname = ol.zopts[0]
    
    elt = ol.elements_ion[ion]
    # ion, wavelength, energy, max. em, Tmax, Trange
    topline = '\\begin{tabular}{l l l l l l c}'
    header = 'ion & $\\lambda$ & E & wavenumber & ' + \
             '$\\max \\, \\Lambda \,\\mathrm{n}_\\mathrm{H}^{-2} \\mathrm{V}^{-1}$' +\
             ' & $\\mathrm{T}_{\\mathrm{peak}}$ & $\\mathrm{T}_{0.1 \\times \\mathrm{peak}}$ \\\\'
    header2 = ' & $\\Ang$ & keV & cm$^{-1}$ & $\\log_{10} \\, \\mathrm{erg} \\, \\mathrm{cm}^{3}$' + \
              ' & $\\log_{10}$~K & \\ $\\log_{10}$~K \\\\'
    print(topline)
    print('\\hline')
    print(header)
    print(header2)          
    print('\\hline')
    
    fillstr = '{ion} & {wl} & {E} & {wn} & {maxem} & {Tmax} & {Trange} \\\\'
    
    tablefilename = ol.dir_emtab%zname + elt + '.hdf5'
    with h5py.File(tablefilename, "r") as tablefile:
        lines = [line.decode() for line in tablefile['lambda'][:]] 
        wls_A = np.array([parse_linename_lambda(line) for line in lines]) 
    # match the ion
    eltpart = ''
    ionnum = ion
    while not ionnum[0].isdigit():
        eltpart = eltpart + ionnum[0]
        ionnum = ionnum[1:]
    eltpart = string.capwords(eltpart)
    ionmatch = eltpart + ' ' * (2 - len(eltpart) + 2 - len(ionnum)) + ionnum

    lsel = np.array([line.startswith(ionmatch + ' ') for line in lines])
    wls_A = wls_A[lsel]
    lines = np.array(lines)[lsel]
    wls_keV = c.planck * c.c / (wls_A * 1e-8) / (c.ev_to_erg * 1e3)
    wns_icm = 1. / (wls_A * 1e-8)    
    
    # table dimensions: T, nH, line
    emdenssq, logTK, lognHcm3 = cu.findemtables(elt, z) 
    emdenssq = emdenssq[:, :, lsel]
    emdenssq_cie = emdenssq[:, -1, :]
    del emdenssq
    
    nHused = lognHcm3[-1]
    print('Using log nH / cm3 = {nH:.2f} for {elt}'.format(nH=nHused,\
          elt=elt))
    
    Tmis = np.argmax(emdenssq_cie, axis=0)
    Tmaxs = logTK[Tmis]
        
    emmins = emdenssq_cie[Tmis, np.arange(emdenssq_cie.shape[1])] + np.log10(rangemin)
    Tranges = [pu.find_intercepts(emdenssq_cie[:, i], logTK, emmins[i])\
               for i in range(emmins.shape[0])]
        
    numdig_wl = 4
    for li in range(len(lines)):
        # ion, wavelength, energy, max. em, Tmax, Trange
        elt_num = parse_linename_ion(lines[li])
        ionname = '\\ion{{{elt}}}{{{num}}}'.format(elt=elt_num[0],\
                          num=(ild.arabic_to_roman[elt_num[1]]).lower())
        wl_A = wls_A[li]
        toround = (numdig_wl) - int(np.log10(wl_A) + 1.) # int applies floor, 4 sig. digits
        wlname = str(np.round(wl_A, toround))
        if len(wlname) > numdig_wl + 1 and wlname[-2:] == '.0':
            wlname = wlname[:-2]
        if '.' in wlname and len(wlname) < numdig_wl + 1:
            wlname = wlname + '0' * (numdig_wl + 1 - len(wlname))
        if wlname.startswith('0.') and len(wlname) < numdig_wl + 2:
            wlname = wlname + '0' * (numdig_wl + 2 - len(wlname))
            
        E_keV = wls_keV[li]
        toround = (numdig_wl) - int(np.log10(E_keV) + 1.) # int applies floor, 4 sig. digits
        Ename = str(np.round(E_keV, toround))
        if len(Ename) > numdig_wl + 1 and Ename[-2:] == '.0':
            Ename = Ename[:-2]
        if '.' in Ename and len(Ename) < numdig_wl + 1:
            Ename = Ename + '0' * (numdig_wl + 1 - len(Ename))
        if Ename.startswith('0.') and len(Ename) < numdig_wl + 2:
            Ename = Ename + '0' * (numdig_wl + 2 - len(Ename))
        
        wn_icm = wns_icm[li]
        toround = (numdig_wl) - int(np.log10(wn_icm) + 1.) # int applies floor, 4 sig. digits
        wnname = str(np.round(wn_icm, toround))
        if len(wnname) > numdig_wl + 1 and wnname[-2:] == '.0':
            wnname = wnname[:-2]
        if '.' in wnname and len(wnname) < numdig_wl + 1:
            wnname = wnname + '0' * (numdig_wl + 1 - len(wnname))
        if wnname.startswith('0.') and len(wnname) < numdig_wl + 2:
            wnname = wnname + '0' * (numdig_wl + 2 - len(wnname))
            
        maxem = str(np.round(emdenssq_cie[Tmis[li], li], 1))
        
        Tpeak = str(Tmaxs[li]) # 1 sig. dig.
        
        Trange = '{mi:.1f}--{ma:.1f}'.format(mi=Tranges[li][0],\
                  ma=Tranges[li][1])
        
        print(fillstr.format(ion=ionname, wl=wlname, E=Ename, maxem=maxem,\
                             Tmax=Tpeak, Trange=Trange, wn=wnname))
    print('\\hline')    
    print('\\end{tabular}')


def print_linetable(z, lines=None, rangemin=0.1, latex=True):
    '''
    print line data using redshift z tables
    
    brightness is calculated for CIE, so z shouldn't actually matter
    
    inputs:
    -------
    z         redshift (float)
    lines     list of lines. Default list (None) is from Chartlotte's thesis
    rangemin  fraction of the maxmimum emissivity to take the T range over
    latex     print as a latex table (bool).
    
    outputs:
    --------
    list of lines and list of indices selected
    prints a LaTeX-ready table of the lines, their wavelengths, and 
      temperature ranges
    '''
    
    zname = ol.zopts[0]
    if lines is None:
        lines = ['c5r', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r',\
          'fe17', 'fe17-other1', 'fe18', 'fe19', 'o7r', 'o7ix', 'o7iy', 'o7f',\
          'o8', 'c6', 'n7']
    
    elements = list(set(ol.elements_ion[line] for line in lines))
    eltnums = {'carbon': 6, 'nitrogen': 7, 'oxygen': 8, 'neon': 10,\
               'magnesium': 12, 'silicon': 14, 'iron': 26}
    elements.sort(key=eltnums.get)
    
    # ion, wavelength, energy, max. em, Tmax, Trange
    if latex:
        topline = '\\begin{tabular}{l l l l l c}'
        header = 'ion & $\\lambda$ & E &' + \
                 '$\\max \\, \\Lambda \,\\mathrm{{n}}_\\mathrm{{H}}^{{-2}} \\mathrm{V}^{{-1}}$' +\
                 ' & $\\mathrm{{T}}_{{\\mathrm{{peak}}}}$ & $\\mathrm{{T}}_{{{rmin} \\times \\mathrm{{peak}}}}$ \\\\'
        header = header.format(rmin=rangemin)
        header2 = ' & $\\Ang$ & keV & $\\log_{10} \\, \\mathrm{erg} \\, \\mathrm{cm}^{3}$' + \
                  ' & $\\log_{10}$~K & \\ $\\log_{10}$~K \\\\'
        print(topline)
        print('\\hline')
        print(header)
        print(header2)          
        print('\\hline')
        
        fillstr = '{ion} & {wl} & {E} & {maxem} & {Tmax} & {Trange} \\\\'
    else:
        header = 'ion \t wavelength \t E \t' + \
                 ' max Lambda * nH**-2 * V**-1 ' +\
                 '\t T(peak) \t T({rmin} * peak)'.format(rmin=rangemin)
        header2 = ' \t A \t keV \t log10 erg * cm**-3 * s**-1' + \
                  ' \t log10 K \t log10 K'
        print(header)
        print(header2)          
        fillstr = '{ion} \t {wl} \t {E} \t {maxem} \t {Tmax} \t {Trange}'
    
    for elt in elements:
        tablefilename = ol.dir_emtab%zname + elt + '.hdf5'
        with h5py.File(tablefilename, "r") as tablefile:
            lines_all = [line.decode() for line in tablefile['lambda'][:]] 
            wls_A = np.array([parse_linename_lambda(line) for line in lines_all]) 
        wls_keV = c.planck * c.c / (wls_A * 1e-8) / (c.ev_to_erg * 1e3)
        
        lines_elt = np.array(lines)[np.array([ol.elements_ion[line] == elt\
                                              for line in lines])]
        lsel = np.array([ol.line_nos_ion[line] for line in lines_elt])
        
        wls_A = wls_A[lsel]
        wls_keV = wls_keV[lsel]
        _lines = np.array(lines_all)[lsel]
        # table dimensions: T, nH, line
        emdenssq, logTK, lognHcm3 = cu.findemtables(elt, z) 
        emdenssq = emdenssq[:, :, lsel]
        emdenssq_cie = emdenssq[:, -1, :]
        del emdenssq
        
        nHused = lognHcm3[-1]
        print('Using log nH / cm3 = {nH:.2f} for {elt}'.format(nH=nHused,\
              elt=elt))
        
        Tmis = np.argmax(emdenssq_cie, axis=0)
        Tmaxs = logTK[Tmis]
        
        emmins = emdenssq_cie[Tmis, np.arange(emdenssq_cie.shape[1])] + np.log10(rangemin)
        Tranges = [pu.find_intercepts(emdenssq_cie[:, i], logTK, emmins[i])\
                   for i in range(emmins.shape[0])]
        
        numdig_wl = 4
        if latex:
            ionfmt = '\\ion{{{elt}}}{{{num}}}'
        else:
            ionfmt = '{elt} {num}'
        for li in range(len(_lines)):
            # ion, wavelength, energy, max. em, Tmax, Trange
            elt_num = parse_linename_ion(_lines[li])
            ionname = ionfmt.format(elt=elt_num[0],\
                              num=(ild.arabic_to_roman[elt_num[1]]).lower())
            wl_A = wls_A[li]
            toround = (numdig_wl) - int(np.log10(wl_A) + 1.) # int applies floor, 4 sig. digits
            wlname = str(np.round(wl_A, toround))
            if len(wlname) > numdig_wl + 1 and wlname[-2:] == '.0':
                wlname = wlname[:-2]
            if '.' in wlname and len(wlname) < numdig_wl + 1:
                wlname = wlname + '0' * (numdig_wl + 1 - len(wlname))
            if wlname.startswith('0.') and len(wlname) < numdig_wl + 2:
                wlname = wlname + '0' * (numdig_wl + 2 - len(wlname))
                
            E_keV = wls_keV[li]
            toround = (numdig_wl) - int(np.log10(E_keV) + 1.) # int applies floor, 4 sig. digits
            Ename = str(np.round(E_keV, toround))
            if len(Ename) > numdig_wl + 1 and Ename[-2:] == '.0':
                Ename = Ename[:-2]
            if '.' in Ename and len(Ename) < numdig_wl + 1:
                Ename = Ename + '0' * (numdig_wl + 1 - len(Ename))
            if Ename.startswith('0.') and len(Ename) < numdig_wl + 2:
                Ename = Ename + '0' * (numdig_wl + 2 - len(Ename))
                
            maxem = str(np.round(emdenssq_cie[Tmis[li], li], 1))
            
            Tpeak = str(Tmaxs[li]) # 1 sig. dig.
            
            Trange = '{mi:.1f}--{ma:.1f}'.format(mi=Tranges[li][0],\
                      ma=Tranges[li][1])
            
            print(fillstr.format(ion=ionname, wl=wlname, E=Ename, maxem=maxem,\
                                 Tmax=Tpeak, Trange=Trange))
    if latex:
        print('\\hline')    
        print('\\end{tabular}')


def print_linetable_minimal(z, lines=None, latex=False):
    '''
    print line data (ions and energies only) using redshift z tables
    
    inputs:
    -------
    z         redshift (float)
    lines     list of lines. Default list (None) is from Chartlotte's thesis
    latex     print as a latex table (bool).
    
    outputs:
    --------
    list of lines and list of indices selected
    prints a LaTeX-ready table of the lines, their wavelengths, and 
      temperature ranges
    '''
    
    zname = ol.zopts[0]
    if lines is None:
        lines = ['c5r', 'n6r', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r',\
          'fe17', 'fe17-other1', 'fe18', 'fe19', 'o7r', 'o7ix', 'o7iy', 'o7f',\
          'o8', 'c6', 'n7']
    
    elements = list(set(ol.elements_ion[line] for line in lines))
    eltnums = {'carbon': 6, 'nitrogen': 7, 'oxygen': 8, 'neon': 10,\
               'magnesium': 12, 'silicon': 14, 'iron': 26}
    elements.sort(key=eltnums.get)
    
    # ion, wavelength, energy, max. em, Tmax, Trange
    if latex:
        topline = '\\begin{tabular}{l l l l l c}'
        header = 'ion & $\\mathrm{E}_{\\mathrm{rest}}$ &' + \
                 '& $\\mathrm{E}_{\\mathrm{obs}}$ \\\\'
        header2 = ' & keV & keV \\\\'
        print(topline)
        print('\\hline')
        print(header)
        print(header2)          
        print('\\hline')
        
        fillstr = '{ion} & {Erest} & {Eobs} \\\\'
    else:
        header = 'ion \t E_rest \t E_obs'
        header2 = ' \t keV \t keV'
        print(header)
        print(header2)          
        fillstr = '{ion} \t {Erest} \t {Eobs}'
    
    for elt in elements:
        tablefilename = ol.dir_emtab%zname + elt + '.hdf5'
        with h5py.File(tablefilename, "r") as tablefile:
            lines_all = [line.decode() for line in tablefile['lambda'][:]] 
            wls_A = np.array([parse_linename_lambda(line) for line in lines_all]) 
        wls_keV = c.planck * c.c / (wls_A * 1e-8) / (c.ev_to_erg * 1e3)
        
        lines_elt = np.array(lines)[np.array([ol.elements_ion[line] == elt\
                                              for line in lines])]
        lsel = np.array([ol.line_nos_ion[line] for line in lines_elt])
        
        wls_keV = wls_keV[lsel]
        _lines = np.array(lines_all)[lsel]
        
        numdig_wl = 4
        if latex:
            ionfmt = '\\ion{{{elt}}}{{{num}}}'
        else:
            ionfmt = '{elt} {num}'
        for li in range(len(_lines)):
            # ion, wavelength, energy, max. em, Tmax, Trange
            elt_num = parse_linename_ion(_lines[li])
            ionname = ionfmt.format(elt=elt_num[0],\
                              num=(ild.arabic_to_roman[elt_num[1]]).lower())

            E_keV = wls_keV[li]
            toround = (numdig_wl) - int(np.log10(E_keV) + 1.) # int applies floor, 4 sig. digits
            Ename = str(np.round(E_keV, toround))
            if len(Ename) > numdig_wl + 1 and Ename[-2:] == '.0':
                Ename = Ename[:-2]
            if '.' in Ename and len(Ename) < numdig_wl + 1:
                Ename = Ename + '0' * (numdig_wl + 1 - len(Ename))
            if Ename.startswith('0.') and len(Ename) < numdig_wl + 2:
                Ename = Ename + '0' * (numdig_wl + 2 - len(Ename))
                
            Eobs_keV = wls_keV[li] / (1. + z)
            toround = (numdig_wl) - int(np.log10(Eobs_keV) + 1.) # int applies floor, 4 sig. digits
            Eobsname = str(np.round(Eobs_keV, toround))
            if len(Eobsname) > numdig_wl + 1 and Eobsname[-2:] == '.0':
                Eobsname = Eobsname[:-2]
            if '.' in Ename and len(Ename) < numdig_wl + 1:
                Eobsname = Eobsname + '0' * (numdig_wl + 1 - len(Eobsname))
            if Eobsname.startswith('0.') and len(Eobsname) < numdig_wl + 2:
                Eobsname = Eobsname + '0' * (numdig_wl + 2 - len(Eobsname))
                
            print(fillstr.format(ion=ionname, Erest=Ename, Eobs=Eobsname))
    if latex:
        print('\\hline')    
        print('\\end{tabular}')
        
 
def readin_TOPfile(filen, NZ, NE):
    '''
    read in the data from filen, return rows matching NE and NZ
    
    input:
    ------
    filen:  string, name of the file
    NZ:     atom numbers to match (int), corresponding to 
    NE:     electron numbers to match (NE, NZ pairs are matched)
    '''
    columns_use = ['NZ', 'NE', 'iSLP', 'jSLP', 'iLV', 'jLV', 'gF', 'WL(A)']
    # 'i' is kind of useless
    # WL(A) and gF are both useful for cross-matching level2.dat and TOP  
    
    # select the NE values that might go with the NZ in this file (one per element)
    _NZ_NE = [(NZ[i], NE[i]) for i in range(len(NZ))]
    _NZ_NE = list(set(_NZ_NE))
    NZvals = list(set([_e[0] for _e in _NZ_NE]))
    elts = [ild.elt_atomnumber[_NZ] for _NZ in NZvals]
    esel = np.array([elt in filen for elt in elts])
    if not np.any(esel):
        raise ValueError('file {} does not contain any element {}'.format(filen, NZvals))
    
    _NZ = np.array(NZvals)[esel][0]
    _NE = set(_ze[1] if _ze[0] == _NZ else None for _ze in _NZ_NE)
    _NE -= {None}
    _NE = np.array(list(_NE))
    
    start_sepions = '# ions:'
    start_commentline = ' ===='
    kwargs_readcsv = {'header': 0,\
                      'usecols': columns_use,\
                      'skiprows': [0, 2],\
                      'skipfooter': 1,\
                      'delim_whitespace': True,\
                      'engine':'python',\
                      }
    with open(filen) as _f:
        line1 = _f.readline()
        if line1.startswith(start_sepions):
            iterate_ions = True
        elif line1.startswith(start_commentline):
            iterate_ions = False
        else:
            raise RuntimeError('file {} in unrecognized format:\n{}'.format(filen, line1))

    if iterate_ions:
        first = True
        base = '.'.join(filen.split('.')[:-1]) + '{NE}.txt'
        for __NE in _NE:
            _filen = base.format(NE=__NE)
            if first:
                df = pd.read_csv(_filen, **kwargs_readcsv)
                if not np.all(df['NZ'] == _NZ):
                    raise ValueError('Error retrieving data: file {filen} contained data on elements other than {NZ}'.format(\
                             filen=_filen, NZ=_NZ))
                if not np.all(df['NE'] == __NE):
                    raise ValueError('Error retrieving data: file {filen} contained data on ion stages other than {NE}'.format(\
                             filen=_filen, NE=__NE))
                first = False
            else:
                _df = pd.read_csv(_filen, **kwargs_readcsv)
                if not np.all(_df['NZ'] == _NZ):
                    raise ValueError('Error retrieving data: file {filen} contained data on elements other than {NZ}'.format(\
                             filen=_filen, NZ=_NZ))
                if not np.all(_df['NE'] == __NE):
                    raise ValueError('Error retrieving data: file {filen} contained data on ion stages other than {NE}'.format(\
                             filen=_filen, NE=__NE))
                df = pd.concat([df, _df], ignore_index=True, copy=False)
    else:
        df = pd.read_csv(filen, **kwargs_readcsv)
        if not np.all(df['NZ'] == _NZ):
            raise ValueError('Error retrieving data: file {filen} contained data on elements other than {NZ}'.format(\
                             filen=filen, NZ=_NZ))
        df = df[np.any(df['NE'][:, np.newaxis] == _NE[np.newaxis, :], axis=1)]
    return df

def get_TOP_transitions(ions, wavelen_A, wavelen_tol=1e-4):
    '''
    retrieve the transitions from the Opacity Project database corresponding 
    to a set of ions
    
    Not matching gF for now, since the CLOUDY tables give either gF or A for
    different lines. Can still be used as a by-eye tiebreaker
    
    input:
    ------
    ions:           list/array of strings; ion names in 'o7', 'fe17' format
    wavelen_A:      wavelengths of the transitions (list/array of floats)
    wavelen_tol:    the relative tolerance for matching wavelengths
    
    output:
    -------
    transitions:   list of strings containing the transitions in TOP 
                   spectroscopic format
                   if multiple matches are found within in wavenumber tolerance,
                   the list will contain a tuple of those transitions
    '''
    
    elt_stage = [ild.get_elt_state(ion) for ion in ions]
    atomnums = [ild.atomnumber_elt[es[0]] for es in elt_stage]
    # TOP lists transitions by how many ions are left, not ion stage
    electronnums = [atomnums[i] + 1 - es[1] for i, es in enumerate(elt_stage)] 
    
    eltlist = list(set([es[0] for es in elt_stage]))
    
    ## read in files for the different elements:
    filen_base = datadir_TOP + 'transitions_{elt}.txt'

    first = True
    for elt in eltlist:
        filen = filen_base.format(elt=elt)
        if first:
            linedata = readin_TOPfile(filen, atomnums, electronnums)
            first = False
        else:
            _ld = readin_TOPfile(filen, atomnums, electronnums)
            linedata = pd.concat([linedata, _ld], ignore_index=True, copy=False)
    
    ## cross-match NE, NZ, lambda
    matches = []
    first = True
    for wl, NZ, NE, ion in zip(wavelen_A, atomnums, electronnums, ions):
        match = np.isclose(wl, linedata['WL(A)'], rtol=wavelen_tol) &\
                (linedata['NZ'] == NZ) & (linedata['NE'] == NE)
        wmatch = np.where(match)[0]
        matches.append(wmatch)
        if first:
            alldata = linedata.loc[match].copy()
            alldata['WL(A)_in'] = np.ones(len(wmatch)) * wl
            alldata['ion'] = [ion] * len(wmatch)
            first = False
        else:
            _df = linedata.loc[match].copy()
            _df['WL(A)_in'] = np.ones(len(wmatch)) * wl
            _df['ion'] = [ion] * len(wmatch)
            alldata = pd.concat([alldata, _df], ignore_index=True, copy=False)
        
    ## get electron configs for the matched lines
    filen = datadir_TOP + 'econfig.txt'
    with open(filen) as _f:
        _f.readline()
        head = _f.readline()
        breaks = [0]
        spaceprev = True
        for i in range(len(head)):
            if head[i] == ' ': 
                if not spaceprev:
                    breaks.append(i)
                spaceprev = True
            else:
                spaceprev = False
        breaks[-1] = len(head)  
    colspecs = [(breaks[i] + 1, breaks[i+1]) for i in range(len(breaks) - 1)]      
    edata = pd.read_fwf(filen, colspecs=colspecs,\
                        skiprows=[0, 2], skipfooter=1)
    
    alldata = pd.merge(alldata, edata, on=['NE', 'NZ', 'iLV', 'iSLP'],\
                       how='left')
    alldata = pd.merge(alldata, edata, left_on=['NE', 'NZ', 'jLV', 'jSLP'],\
                       right_on=['NE', 'NZ', 'iLV', 'iSLP'],\
                       how='left', suffixes=('-i', '-j'))
    del alldata['i-i']
    del alldata['i-j']
    del alldata['iSLP-j']
    del alldata['iLV-j']
    renamef = lambda x: 'jCONF' if x == 'iCONF-j'\
                        else x[:-2] if x[-2:] in ['-i', '-j']\
                        else x
    alldata = alldata.rename(axis='columns', mapper=renamef)
    return alldata
        
def get_transition_data(lines):
    
    ions = [ion_from_line(line) for line in lines]
    
    zname = ol.zopts[0]
    elts = [ol.elements_ion[line] for line in lines]
    cloudy_inds = np.array([ol.line_nos_ion[line] for line in lines])
    
    cloudy_lines = np.empty(len(lines), dtype='<U20')
    cloudy_wls   = np.NaN * np.ones(len(lines))
    eltlist, ioninds = np.unique(elts, return_inverse=True)
    
    for eltn, elt in enumerate(eltlist):
        tablefilename = ol.dir_emtab%zname + elt + '.hdf5'
        _inds = cloudy_inds[ioninds == eltn]
        
        with h5py.File(tablefilename, "r") as tablefile:
            _lines = np.array([line.decode() for line in tablefile['lambda']])[_inds]
            _wls_A = np.array([parse_linename_lambda(line) for line in _lines])
        cloudy_lines[ioninds == eltn] = _lines
        cloudy_wls[ioninds == eltn] = _wls_A
    print(cloudy_lines)
    opdata = get_TOP_transitions(ions, cloudy_wls, wavelen_tol=1e-2)    
    inmatch = np.array([np.where(opdata.loc[i, 'WL(A)_in'] == cloudy_wls)[0][0]\
                      for i in range(len(opdata))])
    opdata['CLOUDY_name'] = cloudy_lines[inmatch]
    print(opdata)


