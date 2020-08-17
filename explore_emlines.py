#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:48:47 2020

@author: Nastasha
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import string

import eagle_constants_and_units as c
import cosmo_utils as cu
import make_maps_opts_locs as ol
import plot_utils as pu
import ion_line_data as ild

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




