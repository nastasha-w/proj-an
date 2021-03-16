# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:16:50 2017

@author: wijers

Calculate line emission or ion column densities from EAGLE simulation outputs,
project these quantities or EAGLE output quantities onto a 2D grid

supports line emission and column density from ions/lines in
make_maps_opts_locs,
and can calculate both total quantities in a column along one of the coordinate
axes, and average quatities weighted by one of the supported quantities

projections along more general directions are not supported.

Uses the v3 version of HsmlAndProject, which allows the projection of larger
numbers of particles than before, and fixes an issue with the minimum
smoothing length in the case of non-square pixels

print and integer divisions changed to work with python3.
Python3 runs are, however, UNTESTED

TODO:
    - test wishlisting
    - implement generic selections using wishlistingstex2 = np.array([stex, 'hello'], dtype='<U10')[0]
      (can be basic only, initially)
"""

version = 3.7 # matches corresponding make_maps version
# for 3.4:
# naming of outputs updated to be more sensible, e.g. cares less about
# projection axis;
# sacrifices naming equal to previous versions
# for 3.5:
# changed metallicity scaling calculation for line emission

###############################################################################
###############################################################################
#############################                    ##############################
#############################       README       ##############################
#############################                    ##############################
###############################################################################
###############################################################################

# This is a make_maps version that tries to incorporate all the other
# make_maps_v3 versions. It still uses make_maps_opts_locs for small tables and
# lists, and file locations.
#
# The main function is make_map, but a lot of the functionality written out in
# previous versions in split into functions here: emission and column density
# calculations, for example, are done per particle (luminosity and number of
# ions, respectively), and have a column area / conversion to surface
# brightness done afterwards. These functions can be used separately to
# calculate e.g. phase diagrams weighted by ion number etc.
#
# Note for these purposes that read-in and deletion of variables is usually
# done via the Vardict (`variable dictionay') class.
# (And at a minimum through Simfile, which at
# present is a thin wrapper for read_eagle_files.readfile, but is meant to
# be expandible to similar simulations, e.g. Hydrangea/C-EAGLE or OWLS, without
# changing the rest of the code too much.)
#
# a Vardict object keeps track of the particle properties read in or calculated
# (e.g. temperature or luminosity). Of course, for single use, this is just
# a somewhat overcomplicted way to store variables. However, it is useful for
# combining more than one function, without having to change the function
# depending on what else you want to calculate (or fill it with if statements).
#
# Vardict simply stores a wishlist, and uses delif to remove a variable only if
# it is not on the wishlist. The variable 'last' can be set to True to force
# deletion, which is easy for the last function in a calculation. Note that
# the generation and updating of the wishlist is not trivial; this just allows
# this part to be independent of e.g. emission calculation, reducing the chance
# of errors cropping up due to unnecessary fiddling with functions.
#
# In order for this to work properly, a consistent naming of variables is
# important. Variables read in from EAGLE directly are named by their hdf5
# path, without the PartType#/ .  Derived quantities have the following names:
#
# logT        for log10 Temperature [K]
# lognH       for log10 hydrogen number density [cm^-3], proper volume density
# propvol     for volume [cm^-3], proper
# luminosity  for emission line luminosity; conversion to cgs stored in Vardict
# coldens     for column density; conversion to cgs stored in Vardict
# eos         for particles on the equation of state (boolean)
# Lambda_over_nH2 cooling rate/nH^2 [erg cm^3/s]
# tcool       cooling time (internal energy/cooling rate) [s]
# nH          hydrogen number density [cm^-3]
# logZ        log10 metallicity, mass fraction
# halomass    halo mass of the halo containing the particle 
#             (0. -> outside a halo)
# subhalocat  category for subhalos: 0.5 -> central, 1.5 -> satellite, 
#             2.5 -> unbound (gets binned into bins 0, 1, 2)
# ipropvol   1 / proper volume (can be used a a weight in 1D histograms to get 
#             number densities) 
# r3D         3D radius (centre point not specified)
# note that these variables do not always mean exactly the same thing:
# for example lognH  will e.g. depend on the hydrogen number density used.
# Take this into account in wishlist generation and calculation order.
###############################################################################
###############################################################################


################################
#       imports (not all)      #
################################

import os
import fnmatch
import numpy as np
import ctypes as ct
import string
import h5py
import numbers as num # for instance checking
import sys
import pandas as pd
import scipy.interpolate as spint 

import make_maps_opts_locs as ol
import projection_classes as pc
import eagle_constants_and_units as c
import ion_header as ionh
import calcfmassh as cfh # not always needed, but only needs numpy and does not cost much to import
#import halocatalogue as hc
import selecthalos as sh
import cosmo_utils as cu
import ion_line_data as ild # for functions to manipulate element/ion names

##########################
#      functions 1       #
##########################

#### cosmological basics

def comoving_distance_cm(z, simfile=None): # assumes Omega_k = 0
    if z < 1e-8:
        print('Using 0 comoving distance from z. \n')
        return 0.
    if simfile is None:
        # use cosmological parameters for EAGLE from its constants and units file
        hpar = c.hubbleparam
        omega0 = c.omega0
        omegalambda = c.omegalambda
    else:
        # use cosmological parameters for the simulation file (read in in Simfile.__init__ from hdf5 output files)
        hpar = simfile.h # use hubble parameter from the simulation; c.hubble is 100 km/s/Mpc
        z = simfile.z    # override input z by the value for the used snapshot
        omega0 = simfile.omegam
        omegalambda = simfile.omegalambda

    def integrand(zi):
        return (omega0 * (1. + zi)**3 + omegalambda)**0.5
    zi_arr = np.arange(0, z + 0.5 * z / 512., z / 512.)
    com = np.trapz(1. / integrand(zi_arr), x=zi_arr)
    return com * c.c / (c.hubble * hpar)

def ang_diam_distance_cm(z,simfile=None):
    if simfile is None:
        return comoving_distance_cm(z,simfile)/(1.+z)
    else:
        return comoving_distance_cm(simfile.z,simfile)/(1.+simfile.z)

def lum_distance_cm(z,simfile=None):
    if simfile is None:
        return comoving_distance_cm(z,simfile)*(1.+z)
    else:
        return comoving_distance_cm(simfile.z,simfile)*(1.+simfile.z)

def Hubble(z,simfile=None):
    if simfile is None:
        # use cosmological parameters for EAGLE from its constants and units file
        hpar = c.hubbleparam
        omega0 = c.omega0
        omegalambda = c.omegalambda
    else:
        # use cosmological parameters for the simulation file (read in in Simfile.__init__ from hdf5 output files)
        hpar = simfile.h # use hubble parameter from the simulation; c.hubble is 100 km/s/Mpc
        z = simfile.z    # override input z by the value for the used snapshot
        omega0 = simfile.omegam
        omegalambda = simfile.omegalambda

    return (c.hubble*hpar)*(omega0*(1.+z)**3 + omegalambda)**0.5

def solidangle(alpha,beta):
    '''
    calculates the solid angle of a rectangular pixel with half angles alpha 
    and beta defining the rectangle
    
    input:
    ------
    alpha:  0.5 * pix_length_1/D_A, 
    beta:   0.5 * pix_length_2/D_A
    
    returns:
    --------
    solid angle in steradians
    
    '''
    #from www.mpia.de/~mathar/public/mathar20051002.pdf
    # citing  A. Khadjavi, J. Opt. Soc. Am. 58, 1417 (1968).
    # stored in home/papers
    # using the exact formula, with alpha = beta,
    # the python exact formula gives zero for alpha = beta < 10^-3--10^-4
    # assuming the pixel sizes are not so odd that the exact formula is needed in one direction and gives zero for the other,
    # use the Taylor expansion to order 4
    # testing the difference between the Taylor and python exact calculations shows that
    # for square pixels, 10**-2.5 is a reasonable cut-off
    # for rectangular pixels, the cut-off seems to be needed in both values
    if alpha < 10**-2.5 or beta < 10**-2.5:
        return 4*alpha*beta - 2*alpha*beta*(alpha**2+beta**2)
    else:
        return 4*np.arccos(((1+alpha**2 +beta**2)/((1+alpha**2)*(1+beta**2)))**0.5)



#### emission/asorption table finding and interpolation

def readstrdata(filen, separator=None, headerlength=1):
    # input: file name, charachter separating columns, number of lines at the top to ignore
    # separator None means any length of whitespace is considered a separator
    # only for string data

    data = open(filen,'r')
    array = []
    # skip header:
    for i in range(headerlength):
        data.readline()
    for line in data:
        line = line.strip() # remove '\n'
        columns = line.split(separator)
        columns = [str(col) for col in columns]
        array.append(columns)
    return np.array(array)


# finds emission tables for element and interpolates them to zcalc if needed and possible
def findemtables(element, zcalc):

    #### checks and setup

    if not element in ol.elements:
        print("There will be an error somewhere: %s is not included or misspelled. \n" % element)

    if zcalc < 0. and zcalc > 1e-4:
        zcalc = 0.0
        zname = ol.zopts[0]
        interp = False

    elif zcalc in ol.zpoints:
        # only need one table
        zname = ol.zopts[ol.zpoints.index(zcalc)]
        interp = False

    elif zcalc <= ol.zpoints[-1]:
        # linear interpolation between two tables
        zarray = np.asarray(ol.zpoints)
        zname1 = ol.zopts[len(zarray[zarray < zcalc]) - 1]
        zname2 = ol.zopts[-len(zarray[zarray > zcalc])]
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n")


    #### read in the tables; interpolate tables in z if needed and possible

    if not interp:
        tablefilename = ol.dir_emtab%zname + element + '.hdf5'
        tablefile = h5py.File(tablefilename, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK =     np.array(tablefile.get('logt'),dtype=np.float32)
        logrhocm3 = np.array(tablefile.get('logd'),dtype=np.float32)
        lines =     np.array(tablefile.get('lines'),dtype=np.float32)


        tablefile.close()

    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        tablefilename1 = ol.dir_emtab%zname1 + element + '.hdf5'
        tablefile1 = h5py.File(tablefilename1, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK1 =     np.array(tablefile1.get('logt'),dtype=np.float32)
        logrhocm31 = np.array(tablefile1.get('logd'),dtype=np.float32)
        lines1 =     np.array(tablefile1.get('lines'),dtype=np.float32)

        tablefile1.close()

        tablefilename2 = ol.dir_emtab%zname2 + element + '.hdf5'
        tablefile2 = h5py.File(tablefilename2, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK2 =     np.array(tablefile2.get('logt'),dtype=np.float32)
        logrhocm32 = np.array(tablefile2.get('logd'),dtype=np.float32)
        lines2 =     np.array(tablefile2.get('lines'),dtype=np.float32)

        tablefile2.close()

        if (np.all(logTK1 == logTK2) and np.all(logrhocm31 == logrhocm32)):
            print("interpolating 2 emission tables")
            lines = 1./(float(zname2)-float(zname1)) * ( (float(zname2)-zcalc)*lines1 + (zcalc-float(zname1))*lines2 )
            logTK = logTK1
            logrhocm3 = logrhocm31
        else:
            print("Temperature and density ranges of the two interpolation z tables don't match. \n")
            print("Using nearest z table in stead.")
            if abs(zcalc - float(zname1)) < abs(zcalc - float(zname2)):
                logTK = logTK1
                logrhocm3 = logrhocm31
                lines = lines1
            else:
                logTK = logTK2
                logrhocm3 = logrhocm32
                lines = lines2

    return lines, logTK, logrhocm3


# calculate emission using C function (interpolator)
def find_emdenssq(z, elt, dct_nH_T, lineind):

    p_emtable, logTK, lognHcm3 = findemtables(elt,z)
    emtable = p_emtable[:,:,lineind]
    lognH = dct_nH_T['lognH']
    logT = dct_nH_T['logT']
    NumPart = len(lognH)
    inlogemission = np.zeros(NumPart,dtype=np.float32)


    if len(logT) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d
    # ion balance tables are temperature x density x line no.
    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

    # argument conversion

    res = interpfunction(logT.astype(np.float32),\
               lognH.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(emtable.astype(np.float32)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               inlogemission \
              )

    print("-------------- C interpolation function output finished ----------------------\n")

    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None

    return inlogemission



def findiontables(ion,z):
    # README in dir_iontab:
    # files are hdf5, contain ionisation fraction of a species for rho, T, z


    #### checks and setup

    if not ion in ol.ions:
        print("There will be an error somewhere: %s is not included or misspelled. \n" % ion)

    tablefilename = ol.dir_iontab %ion + '.hdf5'
    tablefile = h5py.File(tablefilename, "r")
    logTK =   np.array(tablefile.get('logt'),dtype=np.float32)
    lognHcm3 =   np.array(tablefile.get('logd'),dtype=np.float32)
    ziopts = np.array(tablefile.get('redshift'),dtype=np.float32) # monotonically incresing, first entry is zero
    balance_d_t_z = np.array(tablefile.get('ionbal'),dtype=np.float32)
    tablefile.close()

    if z < 0.:
        z = 0.0
        zind = 0
        interp = False

    elif z in ziopts:
        # only need one table
        zind = np.argwhere(z == ziopts)
        interp = False

    elif z <= ziopts[-1]:
        # linear interpolation between two tables
        zind1 = np.sum(ziopts < z) - 1
        zind2 = -np.sum(ziopts > z)
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n")


    #### read in the tables; interpolate tables in z if needed and possible

    if not interp:
        balance = np.squeeze(balance_d_t_z[:,:,zind]) # for some reason, extra dimensions are tacked on

    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        balance1 = balance_d_t_z[:,:,zind1]
        balance2 = balance_d_t_z[:,:,zind2]

        print("interpolating 2 ion balance tables")
        balance = 1./( ziopts[zind2] - ziopts[zind1]) * ( (ziopts[zind2]-z)*balance1 + (z-ziopts[zind1])*balance2 )

    return balance, logTK, lognHcm3

def find_ionbal(z, ion, dct_nH_T):

    # compared to the line emission files, the order of the nH, T indices in the balance tables is switched
    lognH = dct_nH_T['lognH']
    logT  = dct_nH_T['logT']
    balance, logTK, lognHcm3 = findiontables(ion,z) #(np.array([[0.,0.],[0.,1.],[0.,2.]]), np.array([0.,1.,2.]), np.array([0.,1.]) )
    NumPart = len(lognH)
    inbalance = np.zeros(NumPart,dtype=np.float32)


    if len(logT) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too
    # ion balance tables are density x temperature x redshift

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong, \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(lognH.astype(np.float32),\
               logT.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(balance.astype(np.float32)),\
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               inbalance \
              )

    print("-------------- C interpolation function output finished ----------------------\n")

    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None

    return inbalance

def findiontables_sylviassh(ion, z):
    # from Sylvia's tables, using HM12 UV bkg, dust, no cosmic rays or insterstellar radiation field
    # files are hdf5, contain ionisation fraction of a species for rho, T, z
    # for Hmol: returns not mass ratio
    #### checks and setup

    if not ion in ol.elements_ion.keys():
        print("There will be an error somewhere: %s is not included or misspelled. \n" % ion)
    if ion in ['hmolssh', 'h1ssh', 'h1', 'h2']:
        eltname = '/Tdep/HydrogenFractionsVol'
        if ion == 'hmolssh':
            ionind = 2
        elif ion in ['h1ssh', 'h1']:
            ionind = 0
        elif ion == 'h2':
            ionind = 1
    elif ion == 'hneutralssh':
        print("Sorry, total neutral hydrogen is not an option right now with Sylvia's tables")
        raise ValueError("hneutralssh from Sylvia's tables is not currently implemented")
    else:
        if ion[-3:] == 'ssh':
            ion = ion[:-3] # remove 'ssh'
        eltname = ol.elements_ion[ion]
        eltname = ol.eltdct_to_ct[eltname] # sulfur -> Sulphur
        eltname = eltname[0].lower() + eltname[1:]  # lower case first letter
        ionnum = ''
        i = len(ion) - 1
        while (ion[i].isdigit() and i >= 0):
            ionnum = ion[i] + ionnum
            i -= 1
        ionnum = int(ionnum)
        ionind = ionnum - 1

    tablefilename = ol.iontab_sylvia_ssh
    with h5py.File(tablefilename, "r") as tablefile:
        logTK    = np.array(tablefile['TableBins/TemperatureBins'], dtype=np.float32)  # log10 K
        lognHcm3 = np.array(tablefile['TableBins/DensityBins'], dtype=np.float32) # log10 cm^-3
        logZ     = np.array(tablefile['TableBins/MetallicityBins'], dtype=np.float32) # log10 mass fraction
        ziopts   = np.array(tablefile['TableBins/RedshiftBins'], dtype=np.float32) # monotonically incresing, first entry is zero
        if '/' not in eltname:
            balancegrp = tablefile['Tdep/IonFractionsVol']
            grpnames = balancegrp.keys()
            grpname = [name if eltname in name else None for name in grpnames]
            grpname = list(set(grpname))
            grpname.remove(None)
            grpname = grpname[0]
            eltname = 'Tdep/IonFractionsVol/%s'%grpname
        tab_z_T_Z_nH = np.array(tablefile[eltname], dtype=np.float32)[:, :, :, :, ionind] # z, T, Z, nH, ion

    if z < 0.:
        z = 0.0
        zind = 0
        interp = False

    elif z in ziopts:
        # only need one table
        zind = np.argwhere(z == ziopts)
        interp = False

    elif z <= ziopts[-1]:
        # linear interpolation between two tables
        zind1 = np.sum(ziopts < z) - 1
        zind2 = -1 * np.sum(ziopts > z)
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n")


    #### read in the tables; interpolate tables in z if needed and possible

    if not interp:
        balance = np.squeeze(tab_z_T_Z_nH[zind, :, :, :]) # for some reason, extra dimensions are tacked on

    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        balance1 = tab_z_T_Z_nH[zind1, :, :, :]
        balance2 = tab_z_T_Z_nH[zind2, :, :, :]

        print("interpolating 2 emission tables")
        balance = 1. / ( ziopts[zind2] - ziopts[zind1]) *\
                  ( (ziopts[zind2] - z) * balance1 \
                   + (z - ziopts[zind1]) * balance2 )

    return balance, logTK, logZ, lognHcm3

def find_ionbal_sylviassh(z, ion, dct_logT_logZ_lognH):

    # compared to the line emission files, the order of the nH, T indices in the balance tables is switched
    balance, logTK, logZabs, lognHcm3 = findiontables_sylviassh(ion, z) #(np.array([[0.,0.],[0.,1.],[0.,2.]]), np.array([0.,1.,2.]), np.array([0.,1.]) )
    lognH = dct_logT_logZ_lognH['lognH']
    logT  = dct_logT_logZ_lognH['logT']
    logZ  = dct_logT_logZ_lognH['logZ']
    NumPart = len(lognH)
    inbalance = np.zeros(NumPart, dtype=np.float32)

    if len(logT) != NumPart or len(logZ) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator; works for non-emission stuff too
    # retrieved ion balance tables are T x Z x nH

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               ct.c_longlong , \
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(logZabs)*len(lognHcm3),)), \
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logZabs),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(logT.astype(np.float32),\
                         logZ.astype(np.float32),\
                         lognH.astype(np.float32),\
                         ct.c_longlong(NumPart),\
                         np.ndarray.flatten(balance.astype(np.float32)),\
                         logTK.astype(np.float32),\
                         ct.c_int(len(logTK)), \
                         logZabs.astype(np.float32),\
                         ct.c_int(len(logZabs)), \
                         lognHcm3.astype(np.float32),\
                         ct.c_int(len(lognHcm3)),\
                         inbalance \
                         )

    print("-------------- C interpolation function output finished ----------------------\n")

    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None

    return 10**inbalance

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


class linetable_PS20:
    '''
    class for storing data from the Ploeckinger & Schaye (2020) ion balance 
    and line emission tables. 
    
    Methods include functions for interpolating the ion balance and emissvitiy 
    tables in logT, logZ, and lognH, as well as the dust depletion and assumed
    abundance of the given element for a total metallicity value. These table
    interpolations omit the log Z / solarZ = -50.0 values in the tables,
    because these somewhat arbitrary zero values are unsuitable for 
    interpolation.
    
    This does mean that line emission (proportional to the element fraction)
    should be rescaled to the right metallicity (or better: specific element
    abundance) after the interpolation step.
    
    A single instance is for one emission or absorption line. This does mean 
    the table bins and dust table may be stored twice if multiple objects are 
    in use. The instance is also only valid for one redshift.
    '''
    
    def parse_ionname(self):
        '''
        retrieve the element name and ionization stage by interpreting the 
        ion (or emission line) string. Assumes that emission lines match the
        format of the IdentifierLines dataset in the emission line tables 
        (and are not blends), and that absorption lines are speciefied as e.g.
        'o7', 'fe17', or 'Si 4', possibly followed by something other than 
        digits
        
        Returns
        -------
        None.

        '''
        if self.emission: # lines are formatted as in the IdentifierLines dataset
            try:
                self.elementshort = self.ion[:2].strip()
                self.ionstage = int(self.ion[2:4])
                msg = 'Interpreting {line} as coming from the {elt} {stage} ion'
                msg = msg.format(line=self.ion, elt=self.elementshort,
                                 stage=self.ionstage)
                print(msg)
            except:
                msg = 'Failed to parse "{}" as an emission line'
                raise ValueError(msg.format(self.ion))
        else: # ions are '<elt><stage>....'
            try:
                if self.ion == 'hmolssh':
                    self.elementshort = 'H'
                    # get a useful error, I hope
                    self.ionstage = 'invalid index for hmolssh'   
                else:
                    self.elementshort = ''
                    self.ionstage = ''
                    i = 0
                    while not self.ion[i].isdigit():
                        i += 1             
                    self.elementshort = string.capwords(self.ion[:i])
                    self.elementshort = self.elementshort.strip()
                    while self.ion[i].isdigit():
                        self.ionstage = self.ionstage + self.ion[i]
                        i += 1
                        if i == len(self.ion):
                            break
                    self.ionstage = int(self.ionstage)
            except:
                msg = 'Failed to parse "{}" as an ion'
                raise ValueError(msg.format(self.ion))
            msg = 'Interpreting {ion} as the {elt} {stage} ion'
            msg = msg.format(ion=self.ion, elt=self.elementshort,
                             stage=self.ionstage)
            print(msg)
            
    def getmetadata(self):
        '''
        get the solar metallicity (self.solarZ),
        the abundance of the selected element as a function of log Z / solarZ 
        (self.numberfraction_Z),
        and the element mass in atomic mass units
      
        Raises
        ------
        ValueError
            No abundance data is available for self.elementshort

        Returns
        -------
        None.

        '''
        self.table_logzero = -50.0
        
        with h5py.File(self.ionbalfile, 'r') as f:
            # get element abundances (n_i / n_H) for tabulated log Z / Z_solar
            self.numberfractions_Z_elt = f['TotalAbundances'][:, :]
            elts = [elt.decode() for elt in f['ElementNamesShort'][:]]
            if self.elementshort not in elts:
                msg = 'Data for element {elt} is not available'
                msg = msg.format(elt=self.elementshort)
                raise ValueError(msg)
            self.eltind = np.where([self.elementshort == elt \
                                    for elt in elts])[0][0]
            # element number fraction n_i / n_H as a function of metallicity
            # in the tables
            self.numberfraction_Z = np.copy(\
                self.numberfractions_Z_elt[:, self.eltind])
            
            self.element = f['ElementNames'][self.eltind].decode().strip()
            self.elementmass_u = f['ElementMasses'][self.eltind]
            
            # solar Z for scaling
            self.solarZ = f['SolarMetallicity'][0]
    
    def __init__(self, ion, z, emission=False, vol=True,
                 ionbalfile=ol.iontab_sylvia_ssh,
                 emtabfile=ol.emtab_sylvia_ssh):
        '''
        Parameters
        ----------
        ion: string
            ion or emission line to get tables for; emission line names should
            match an entry in the IdentifierLines dataset in the emission line
            table. Ion names should follow the format 
            '<element abbreviation><ionization stage>[other stuff]',
            e.g. 'o7' for the O^{6+} / O VII ion, 'Fe17-other', etc.
            other options are 'hmolssh' and 'h1ssh'. Note that 'h2' is ionized
            hydrogen (H^+), not molecular hydrogen (H_2)
        z: float
            redshift. Must be within the tabulated range.
        emission: bool
            get emission tables (True) or absorption tables (False)
            if emission is True, the absorption table methods for the ion 
            producing the line are also available. Using emission=True for an
            invalid emission line name will produce an error though.
        vol: bool
            Use quantities for the last Cloudy zone (True) or column-averaged
            quantities (False). The column option is only available for the
            hydrogen species.
        ionbalfile: string
            the file (including directory path) containing the ion balance 
            data. This is also needed for emission calculations, since some of
            the metadata is not contained in the emission file.
        emtabfile: string
            the file (including directory path) containing the emissivity 
            data. Only used when calculating emission data. It is assumed 
            table options, like inclusion (or not) of shielding, cosmic rays,
            dust, and stellar radiation, are the same for both tables.
            
        Returns
        -------
        a linetable_PS20 object.
        '''
        self.ion = ion
        self.z = z
        self.ionbalfile = ionbalfile
        self.emtabfile = emtabfile
        self.emission = emission
        self.vol = vol
        
        self.parse_ionname()
        self.getmetadata()
    
    def __str__(self):
        _str = '{obj}: {ion} {emabs} at z={z:.3f} using {vol} data from' +\
               ' {iontab} and {emtab}'
        if self.emission:
            emabs = 'emission'
        else:
            emabs = 'ion fraction'
        if self.vol:
            vc = 'Vol'
        else:
            vc = 'Col'
        _str = _str.format(obj=self.__class__, ion=self.ion, emabs=emabs,
                           z=self.z, vol=vc, 
                           iontab=self.ionbalfile.split('/')[-1],
                           emtab=self.emtabfile.split('/')[-1])
        return _str
    
    def __repr__(self):
        _str = '{obj} instance: interpolate Ploeckinger & Schaye (2020) tables\n'
        _str += 'ion:\t {ion}, interpreted as (coming from) the {elt} {stage} ion\n'
        _str += 'z:\t {z}\n'
        _str += 'vol:\t {vol}\n'
        _str += 'emission:\t {emission}\n'
        _str += 'emtabfile:\t {emtabfile}\n'
        _str += 'ionbalfile:\t {ionbalfile}\n'
        _str = _str.format(obj=self.__class__, ion=self.ion,
                           elt=self.elementshort, stage=self.ionstage,
                           z=self.z, vol=self.vol, emission=self.emission,
                           emtabfile=self.emtabfile,
                           ionbalfile=self.ionbalfile)
        return _str
    
    def find_ionbal(self, dct_T_Z_nH, log=False):
        '''
        retrieve the interpolated ion balance values for the input particle 
        density, temperature, and metallicity
        
        The lowest metallicity bin is ignored since all ion fractions are 
        tabulated as zero there.

        Parameters
        ----------
        dct_T_Z_nH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].
        log: bool
            return log ion balance if True
        Returns
        -------
        float array
            ion balances: ion mass / gas phase parent element mass.

        '''
        if not hasattr(self, 'iontable_T_Z_nH'):
            self.findiontable()
        
        res = self.interpolate_3Dtable(dct_T_Z_nH,
                                       self.iontable_T_Z_nH[:, :, :])
        if not log:
            res = 10**res
        return res
        
    def find_logemission(self, dct_T_Z_nH):
        '''
        retrieve the interpolated emission values for the input particle 
        density, temperature, and metallicity

        Parameters
        ----------
        dct_T_Z_nH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].

        Returns
        -------
        float array
            log line emission per unit volume: log10 erg / s / cm**3 
            (the non-log emission values may cause overflows)
        '''
        if not hasattr(self, 'emtable_T_Z_nH'):
            self.findemtable()
        return self.interpolate_3Dtable(dct_T_Z_nH, self.emtable_T_Z_nH)
    
    def find_depletion(self, dct_T_Z_nH):
        '''
        retrieve the interpolated dust depletion values for the input particle 
        density, temperature, and metallicity

        Parameters
        ----------
        dct_T_Z_nH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].

        Returns
        -------
        float array
            dust depletion: fraction of element locked in dust
        '''
        if not hasattr(self, 'depletiontable_T_Z_nH'):
            self.finddepletiontable()
        return 10**self.interpolate_3Dtable(dct_T_Z_nH,
                                            self.depletiontable_T_Z_nH)
    
    def find_assumedabundance(self, dct_Z, log=False):
        '''
        retrieve the interpolated assumed parent element abundance values for 
        the input particle metallicity

        Parameters
        ----------
        dct_Z: dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
        log: bool
            return log abundances if True
        Returns
        -------
        float array
            element abundance n_i / n_H for the parent element of the line or
            ion assumed at the input metallicity
            
        '''
        
        if not hasattr(self, 'logZsol'):
            self.findiontable()
        logZabs = self.logZsol + np.log10(self.solarZ)
        # edge values outside range: matches use of edge values in T, Z, nH
        # table interpolation
        edgevals = (self.numberfraction_Z[1], self.numberfraction_Z[-1])
        self.abunds_interp = spint.interp1d(logZabs, self.numberfraction_Z[1:],
                                            kind='linear', axis=-1, copy=True,
                                            bounds_error=False,
                                            fill_value=edgevals)
        res = self.abunds_interp(dct_Z['logZ'])
        if not log:
            res = 10**res
        return res
        
    def findiontable(self):
        if self.vol:
            vc = 'Vol'
        else:
            vc = 'Col'
            
        if self.ion in ['hmolssh', 'h1ssh', 'h1', 'h2']:
            tablepath = '/Tdep/HydrogenFractions{vc}'.format(vc=vc)
            if self.ion == 'hmolssh':
                ionind = 2
            elif self.ion in ['h1ssh', 'h1']:
                ionind = 0
            elif self.ion == 'h2':
                ionind = 1
        elif self.ion == 'hneutralssh':
            msg = "hneutralssh fractions from the PS20 tables are not" + \
                  "currently implemented"
            raise NotImplementedError(msg)
        else:
            if not self.vol:
                msg = 'Column quantities are only available for hydrogen'
                raise ValueError(msg)
            ionind = self.ionstage - 1 
            tablepath = 'Tdep/IonFractions/{eltnum:02d}{eltname}'
            tablepath = tablepath.format(eltnum=self.eltind,
                                         eltname=self.element.lower())
            print('Using table {}'.format(tablepath))
            
        with h5py.File(self.ionbalfile, "r") as tablefile:
            self.logTK     = tablefile['TableBins/TemperatureBins'][:] 
            self.lognHcm3  = tablefile['TableBins/DensityBins'][:] 
            self.logZsol   = tablefile['TableBins/MetallicityBins'][1:]
            self.redshifts = tablefile['TableBins/RedshiftBins'][:] 
            
            if self.z < self.redshifts[0] or self.z > self.redshifts[-1]:
                msg = 'Desired redshift {z} is outside the tabulated range '+\
                      + '{zmin} - {zmax}'
                msg = msg.format(z=self.z, zmin=self.redshifts[0], 
                                 zmax=self.reshifts[-1])
                raise ValueError(msg)
            zi_lo = np.min(np.where(self.z <= self.redshifts)[0])
            zi_hi = np.max(np.where(self.z >= self.redshifts)[0])            
            
            tableg = tablefile[tablepath] #  z, T, Z, nH, ion
             # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Ion
            if zi_lo == zi_hi:
                self.iontable_T_Z_nH =\
                    tableg[zi_lo, :, 1:, :, ionind]
            else:
                msg = 'Linearly interpolating ion balance table ' +\
                      'values in redshift'
                print(msg)
                z_lo = self.redshifts[zi_lo]
                z_hi = self.redshifts[zi_hi]
                
                self.iontable_T_Z_nH =\
                    (z_hi - self.z) / (z_hi - z_lo) * \
                        tableg[zi_lo, :, 1:, :, ionind] + \
                    (self.z - z_lo) / (z_hi - z_lo) * \
                        tableg[zi_hi, :, 1:, :, ionind]

    def findemtable(self):
        if self.vol:
            vc = 'Vol'
        else:
            vc = 'Col'
            
        with h5py.File(self.emtabfile, 'r') as f:
            lineid = f['IdentifierLines'][:]
            lineid = np.array([line.decode() for line in lineid])
            match = [self.ion == line for line in lineid]
            li = np.where(match)[0][0]
            
            emg = f['Tdep/Emissivities{vc}'.format(vc=vc)] 
            # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Line
            self.redshifts = f['TableBins/RedshiftBins'][:]
            self.logTK = f['TableBins/TemperatureBins'][:]
            self.lognHcm3 = f['TableBins/DensityBins'][:]
            self.logZsol = f['TableBins/MetallicityBins'][1:] # -50. = primordial
            
            zi_lo = np.min(np.where(self.z <= self.redshifts)[0])
            zi_hi = np.max(np.where(self.z >= self.redshifts)[0])            
            
            if self.z < self.redshifts[0] or self.z > self.redshifts[-1]:
                msg = 'Desired redshift {z} is outside the tabulated range '+\
                      + '{zmin} - {zmax}'
                msg = msg.format(z=self.z, zmin=self.redshifts[0], 
                                 zmax=self.reshifts[-1])
                raise ValueError(msg) 
             # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Line
            if zi_lo == zi_hi:
                self.emtable_T_Z_nH = emg[zi_lo, :, 1:, :, li]
            else:
                z_lo = self.redshifts[zi_lo]
                z_hi = self.redshifts[zi_hi]
                self.emtable_T_Z_nH =\
                    (z_hi - self.z) / (z_hi - z_lo) *\
                        emg[zi_lo, :, 1:, :, li] + \
                    (self.z - z_lo) / (z_hi - z_lo) *\
                        emg[zi_hi, :, 1:, :, li]
        
    def finddepletiontable(self):    
        with h5py.File(self.ionbalfile, 'r') as f:        
            deplg = f['Tdep/Depletion'] 
            # z, T, Z, nH, element
            self.redshifts = f['TableBins/RedshiftBins'][:]
            self.logTK = f['TableBins/TemperatureBins'][:]
            self.lognHcm3 = f['TableBins/DensityBins'][:]
            self.logZsol = f['TableBins/MetallicityBins'][1:] # -50. = primordial
             
            zi_lo = np.min(np.where(self.z <= self.redshifts)[0])
            zi_hi = np.max(np.where(self.z >= self.redshifts)[0])              
            
            if self.z < self.redshifts[0] or self.z > self.redshifts[-1]:
                msg = 'Desired redshift {z} is outside the tabulated range '+\
                      '{zmin} - {zmax}'
                msg = msg.format(z=self.z, zmin=self.redshifts[0], 
                                 zmax=self.reshifts[-1])
                raise ValueError(msg) 
             # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: element
            if zi_lo == zi_hi:
                self.depletiontable_T_Z_nH = \
                    deplg[zi_lo, :, 1:, :, self.eltind]
            else:
                z_lo = self.redshifts[zi_lo]
                z_hi = self.redshifts[zi_hi]
                self.depletiontable_T_Z_nH =\
                    (z_hi - self.z) / (z_hi - z_lo) *\
                        deplg[zi_lo, :, 1:, :, self.eltind] + \
                    (self.z - z_lo) / (z_hi - z_lo) *\
                        deplg[zi_hi, :, 1:, :, self.eltind]
        
        
    def interpolate_3Dtable(self, dct_logT_logZ_lognH, table):
        '''
        retrieve the table values for the input particle density, temperature,
        and metallicity

        Parameters
        ----------
        dct_logT_logZ_lognH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].

        Raises
        ------
        ValueError
            input arrays have different lengths
            or the requested redshift is outside the tabulated range
            or the input table shape doesn't match logTK, logZsol, lognHcm3.

        Returns
        -------
        1D float array
            the fraction interpolated table values

        '''
        if len(table.shape) != 3:
            msg = 'Interpolation is for 3 dimensional tables only, ' + \
                'not shape {}'.format(table.shape)
            raise ValueError(msg)
        
        expected_tableshape = (len(self.logTK), 
                               len(self.logZsol), 
                               len(self.lognHcm3)) 
        if table.shape != expected_tableshape: 
            msg  = 'Table shape {} did not match expected {}'
            msg = msg.format(table.shape, expected_tableshape)
            raise ValueError(msg)
            
        logT  = dct_logT_logZ_lognH['logT']
        logZ  = dct_logT_logZ_lognH['logZ']
        lognH = dct_logT_logZ_lognH['lognH']
        
        NumPart = len(lognH)
        inbalance = np.zeros(NumPart, dtype=np.float32)    
        if len(logT) != NumPart or len(logZ) != NumPart:
            raise ValueError('lognH, logZ, and logT  should have the same length')
    
        # need to compile with some extra options to get this to work: make -f make_emission_only
        print("------------------- C interpolation function output --------------------------\n")
        cfile = ol.c_interpfile
    
        acfile = ct.CDLL(cfile)
        interpfunction = acfile.interpolate_3d 
        # just a linear interpolator; works for non-emission stuff too
        # retrieved ion balance tables are T x Z x nH
    
        type_partarray = np.ctypeslib.ndpointer(dtype=ct.c_float, 
                                                          shape=(NumPart,))
        tablesize = np.prod(table.shape)
        type_table = np.ctypeslib.ndpointer(dtype=ct.c_float,
                                            shape=(tablesize,))
        
        interpfunction.argtypes = [type_partarray,
                                   type_partarray,
                                   type_partarray,
                                   ct.c_longlong, 
                                   type_table,
                                   np.ctypeslib.ndpointer(dtype=ct.c_float,
                                       shape=self.logTK.shape),
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=ct.c_float,
                                       shape=self.logZsol.shape),
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=ct.c_float,
                                       shape=self.lognHcm3.shape), 
                                   ct.c_int,
                                   type_partarray]
    
        logZabs = self.logZsol + np.log10(self.solarZ)
        res = interpfunction(logT.astype(np.float32),
                             logZ.astype(np.float32),
                             lognH.astype(np.float32),
                             ct.c_longlong(NumPart),
                             np.ndarray.flatten(table.astype(np.float32)),
                             self.logTK.astype(np.float32),
                             ct.c_int(len(self.logTK)), 
                             logZabs.astype(np.float32),
                             ct.c_int(len(logZabs)), 
                             self.lognHcm3.astype(np.float32),
                             ct.c_int(len(self.lognHcm3)),
                             inbalance,
                             )
    
        print("-------------- C interpolation function output finished ----------------------\n")
    
        if res != 0:
            print('Something has gone wrong in the C function: output %s. \n',str(res))
            return None
    
        return inbalance


### cooling tables -> cooling rates.

def getcoolingtable(tablefile, per_elt=True):
    '''
    retrieves the per element or total metal tables from Wiersma, Schaye,
    & Smith 2009 given an hdf5 file and whether to get the per element data or
    the total metals data
    a full file is ~1.2 MB, so reading in everything before processing
    shouldn't be too hard on memory
    '''

    cooldct = {} # to hold the read-in data

    # Metal-free cooling tables are on a T, nH, Hefrac grid
    lognHcm3   =   np.log10(np.array(tablefile.get('Metal_free/Hydrogen_density_bins'),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
    logTK      =   np.log10(np.array(tablefile.get('Metal_free/Temperature_bins'),dtype=np.float32)) # stored as T [K], but log space even intervals
    Hemassfrac =   np.array(tablefile.get('Metal_free/Helium_mass_fraction_bins'),dtype=np.float32)

    lambda_over_nH2_mf = np.array(tablefile.get('Metal_free/Net_Cooling'),dtype=np.float32)

    cooldct['Metal_free'] = {'lognHcm3': lognHcm3, 'logTK': logTK, 'Hemassfrac': Hemassfrac, 'Lambda_over_nH2': lambda_over_nH2_mf}
    cooldct['Metal_free']['mu'] = np.array(tablefile.get('Metal_free/Mean_particle_mass'),dtype=np.float32)

    # electron density table and the solar table to scale it by
    cooldct['Electron_density_over_n_h'] = {}
    cooldct['Electron_density_over_n_h']['solar'] = np.array(tablefile.get('Solar/Electron_density_over_n_h'),dtype=np.float32)
    cooldct['Electron_density_over_n_h']['solar_logTK'] = np.log10(np.array(tablefile.get('Solar/Temperature_bins'),dtype=np.float32))
    cooldct['Electron_density_over_n_h']['solar_lognHcm3'] = np.log10(np.array(tablefile.get('Solar/Hydrogen_density_bins'),dtype=np.float32))
    cooldct['Electron_density_over_n_h']['table'] = np.array(tablefile.get('Metal_free/Electron_density_over_n_h'),dtype=np.float32) # same bins as the metal-free cooling tables

    # solar abundance data
    elts = np.array(tablefile.get('/Header/Abundances/Abund_names')) # list of element names (capital letters)
    abunds = np.array(tablefile.get('/Header/Abundances/Solar_number_ratios'),dtype=np.float32)
    cooldct['solar_nfrac'] = {elts[ind]: abunds[ind] for ind in range(len(elts))}
    cooldct['solar_mfrac'] = {'total_metals': 0.0129} # from Rob Wiersma's IDL routine compute_cooling_Z.pro, documentation. Solar mass fraction

    eltsl = list(elts)
    eltsl.remove('Hydrogen')
    eltsl.remove('Helium')
    # per-element or total cooling rates
    if per_elt:
        for elt in eltsl: # will get NaN values for Helium, Hydrogen if these are not removed from the list
            cooldct[elt] = {}
            cooldct[elt]['logTK']    = np.log10(np.array(tablefile.get('%s/Temperature_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
            cooldct[elt]['lognHcm3'] = np.log10(np.array(tablefile.get('%s/Hydrogen_density_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
            cooldct[elt]['Lambda_over_nH2'] = np.array(tablefile.get('%s/Net_Cooling'%(elt)),dtype=np.float32)  # stored as nH [cm^-3], but log space even intervals

    else:
        elt = 'Total_Metals'
        cooldct[elt] = {}
        cooldct[elt]['logTK']    = np.log10(np.array(tablefile.get('%s/Temperature_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
        cooldct[elt]['lognHcm3'] = np.log10(np.array(tablefile.get('%s/Hydrogen_density_bins'%(elt)),dtype=np.float32))  # stored as nH [cm^-3], but log space even intervals
        cooldct[elt]['Lambda_over_nH2'] = np.array(tablefile.get('%s/Net_Cooling'%(elt)),dtype=np.float32)  # stored as nH [cm^-3], but log space even intervals

    return cooldct

def findcoolingtables(z, method='per_element'):
    '''
    gets the per element cooling tables from Wiersema, Schaye, & Smith 2009,
    does linear redshift interpolation to the selected z value if required

    methods: 'per_element' or 'total_metals'. See readme file with the cooling
    tables, or Wiersema, Schaye & Smith 2009, eq 4
    (basically, the difference is that total_metals uses all metals and assumes
    solar element abundance ratios)
    '''
    wdir = ol.dir_coolingtab
    szopts = readstrdata(wdir + 'redshifts.dat', headerlength = 1) # file has list of redshifts for which there are tables
    szopts = szopts.T[0] # 2d->1d array
    zopts = np.array([float(sz) for sz in szopts])
    #print(zopts)
    #print(szopts)
    if method == 'per_element':
        perelt = True
    elif method == 'total_metals':
        perelt = False

    if z < 0. and z > -1.e-4:
        z = 0.0
        zind = 0
        interp = False

    elif z in zopts:
        # only need one table
        zind = np.argwhere(z == zopts)[0,0]
        interp = False

    elif z <= zopts[-1]:
        # linear interpolation between two tables
        zind1 = np.sum(zopts < z) - 1
        zind2 = -1 * np.sum(zopts > z)
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n")


    if not interp:
        tablefilename = wdir + 'z_%s.hdf5'%(szopts[zind])
        tablefile = h5py.File(tablefilename, "r")
        tabdct_out = getcoolingtable(tablefile, per_elt = perelt)
        tablefile.close()

    else: # get both cooling tables, interpolate in z
        #print()
        tablefilename1 = wdir + 'z_%s.hdf5'%(szopts[zind1])
        tablefile1 = h5py.File(tablefilename1, "r")
        tabdct1 = getcoolingtable(tablefile1, per_elt = perelt)
        tablefile1.close()

        tablefilename2 = wdir + 'z_%s.hdf5'%(szopts[zind2])
        tablefile2 = h5py.File(tablefilename2, "r")
        tabdct2 = getcoolingtable(tablefile2, per_elt = perelt)
        tablefile2.close()

        tabdct_out = {}

        keys = list(tabdct1.keys())

        # metal-free: if interpolation grid match (they should), interpolate
        # the tables in z. Electron density tables have the same grid points
        # separate from the other because it contains helium
        if (np.all(tabdct1['Metal_free']['lognHcm3']   == tabdct2['Metal_free']['lognHcm3']) and\
            np.all(tabdct1['Metal_free']['logTK']      == tabdct2['Metal_free']['logTK']) ) and\
            np.all(tabdct1['Metal_free']['Hemassfrac'] == tabdct2['Metal_free']['Hemassfrac' ]):
            tabdct_out['Metal_free'] = {}
            tabdct_out['Electron_density_over_n_h'] = {}
            tabdct_out['Metal_free']['lognHcm3']   = tabdct2['Metal_free']['lognHcm3']
            tabdct_out['Metal_free']['logTK']      = tabdct2['Metal_free']['logTK']
            tabdct_out['Metal_free']['Hemassfrac'] = tabdct2['Metal_free']['Hemassfrac']
            tabdct_out['Metal_free']['Lambda_over_nH2'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Metal_free']['Lambda_over_nH2'] +\
                  (z-zopts[zind1])*tabdct2['Metal_free']['Lambda_over_nH2']   )
            tabdct_out['Electron_density_over_n_h']['table'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Electron_density_over_n_h']['table'] +\
                  (z-zopts[zind1])*tabdct2['Electron_density_over_n_h']['table']   )
            tabdct_out['Metal_free']['mu'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Metal_free']['mu'] +\
                  (z-zopts[zind1])*tabdct2['Metal_free']['mu']   )
        else:
            print('Failed to interpolate Metal_free tables due to mismatch in interpolation grids')

        #interpolate solar electron density grids
        if (np.all(tabdct1['Electron_density_over_n_h']['solar_logTK']    == tabdct2['Electron_density_over_n_h']['solar_logTK'] ) and\
            np.all(tabdct1['Electron_density_over_n_h']['solar_lognHcm3'] == tabdct2['Electron_density_over_n_h']['solar_lognHcm3']) ):
            if 'Electron_density_over_n_h' not in tabdct_out.keys():
                tabdct_out['Electron_density_over_n_h'] = {}

            tabdct_out['Electron_density_over_n_h']['solar_logTK']    = tabdct2['Electron_density_over_n_h']['solar_logTK']
            tabdct_out['Electron_density_over_n_h']['solar_lognHcm3'] = tabdct2['Electron_density_over_n_h']['solar_lognHcm3']

            tabdct_out['Electron_density_over_n_h']['solar'] =\
                1./(zopts[zind2]-zopts[zind1]) *\
                ( (zopts[zind2]-z)*tabdct1['Electron_density_over_n_h']['solar'] +\
                  (z-zopts[zind1])*tabdct2['Electron_density_over_n_h']['solar']   )
        else:
            print('Failed to interpolate Solar tables due to mismatch in interpolation grids')

        if tabdct1['solar_nfrac'] == tabdct2['solar_nfrac']:
            tabdct_out['solar_nfrac'] = tabdct2['solar_nfrac']
        else:
            print('Failed to assign solar number fraction list due to mismatch between tables')
        if tabdct1['solar_mfrac'] == tabdct2['solar_mfrac']:
            tabdct_out['solar_mfrac'] = tabdct2['solar_mfrac']
        else:
            print('Failed to assign mass number fraction list due to mismatch between tables')

         # we've just done these:
        keys.remove('Metal_free')
        keys.remove('Electron_density_over_n_h')
        keys.remove('solar_nfrac')
        keys.remove('solar_mfrac')
        # total_metals and the elements all work the same way
        for key in keys:
            if (np.all(tabdct1[key]['lognHcm3']   == tabdct2[key]['lognHcm3']) and\
                np.all(tabdct1[key]['logTK']      == tabdct2[key]['logTK']) ):
                tabdct_out[key] = {}
                tabdct_out[key]['lognHcm3']   = tabdct2[key]['lognHcm3']
                tabdct_out[key]['logTK']      = tabdct2[key]['logTK']
                tabdct_out[key]['Lambda_over_nH2'] =\
                    1./(zopts[zind2]-zopts[zind1]) *\
                    ( (zopts[zind2]-z)*tabdct1[key]['Lambda_over_nH2'] +\
                      (z-zopts[zind1])*tabdct2[key]['Lambda_over_nH2']   )
            else:
                print('Failed to interpolate %s tables due to mismatch in interpolation grids'%key)

    return tabdct_out


def find_coolingrates(z, dct, method='per_element', **kwargs):
    '''
    !! Comparison of cooling times to Wiersma, Schaye, & Smith (2009), where
    the used tables come from and which the calculations should match, shows
    that cooling contours might be off by ~0.1-0.2 dex (esp. their fig. 4,
    where the intersecting lines mean differences are more visible than they
    would be inother plots)


    arguments:
    z:      redshift (used to find the radiation field)
            if Vardict is used, the redshift it contains is used for unit
            conversions
    dct:    dictionary: should contain
                lognH [cm^-3] hydrogen number density, log10
                logT [K] temperature, log10
                mass fraction per element: dictionary
                    element name (lowercase, sulfur): mass fraction
                Density [cm^-3] density
                    (needed to get number density from mass fraction)
            or Vardict instance: kwargs must include
                T4EOS (True/False; excluding SF gas should be done beforehand
                via selections)
                hab    if lognH is not read in already:
                    'SmoothedElementAbundance/Hydrogen',
                    'ElementAbundance/Hydrogen',
                    or hydrogen mass fraction (float)
                abunds ('Sm', 'Pt', or dct of float values [mass fraction,
                       NOT solar])
                       in case of a dictionary, all lowercase or all uppercase
                       element names should both work; tested on uppercase
                       spelling: Sulfur, not Sulpher
                last   (boolean, default True): delete all vardict entries used
                       except the final answer. (Otherwise, entries in the
                       wishlist are saved, but the other are still deleted.)
    method: 'per_element' or 'total_metals'
            if dct is a dictionary, element abundances should have helium in
            both cases, only 'metallicity' and 'helium' if the 'total_metals'
            method is used

    returns:
    cooling rate Lambda/n_H^2 [erg/s cm^3] for the particles for which
    the data was supplied
    '''

    if method == 'per_element':
        elts_geq_he = list(ol.eltdct_to_ct.keys())
        elts_geq_he.remove('hydrogen')
        elts_geq_h = list(np.copy(elts_geq_he))
        elts_geq_he.remove('helium')
    elif method == 'total_metals':
        elts_geq_h = ['helium', 'metallicity']
        elts_geq_he = ['metallicity']
    delafter_abunds = False # if abundances are overwritten with custom values, delete after we're done to avoid confusion with the EAGLE output values (same names)

    if isinstance(dct, pc.Vardict):
        vard = True
        partdct = dct.particle
        if 'last' in kwargs.keys():
            last = kwargs['last']
        else:
            last = True

        eltab_base = kwargs['abunds']
        if not (isinstance(eltab_base, str) or isinstance(eltab_base, dict)): # tuple like in make_maps?
            eltab_base = eltab_base[0]
        if isinstance(eltab_base,str):
            if eltab_base == 'Sm' or 'SmoothedElementAbundance' in eltab_base: # example abundance is accepted
                eltab_base = 'SmoothedElementAbundance/%s'
            elif eltab_base == 'Pt' or 'ElementAbundance' == eltab_base[:16]: # example abundance is accepted
                eltab_base = 'ElementAbundance/%s'
            else:
                print('eltab value %s is not a valid option'%eltab_base)
                return -1
        elif isinstance(eltab_base, dict):
            delafter_abunds = True
            eltab_base =  'ElementAbundance/%s'
            try: # allow uppercase or lowercase element names
                dct.particle.update({eltab_base%string.capwords(elt): kwargs['abunds'][elt] for elt in elts_geq_he})
            except KeyError:
                dct.particle.update({eltab_base%string.capwords(elt): kwargs['abunds'][string.capwords(elt)] for elt in elts_geq_he})

        if not dct.isstored_part('logT'):
            dct.getlogT(last=last,logT = kwargs['T4EOS'])
        # modify wishlist for our purposes
        wishlist_old = list(np.copy(dct.wishlist))
        dct.wishlist = list(set(dct.wishlist + ['Density']))
        if not dct.isstored_part('lognH'):
            if 'last' in kwargs.keys():
                skwargs = kwargs.copy()
                del skwargs['last']
            else:
                skwargs = kwargs
            dct.getlognH(last=False,**skwargs)
        if not dct.isstored_part('Density'): # if we already had lognH, might need to read in Density explicitly
            dct.readif('Density')
        NumPart = len(dct.particle['lognH'])

    else:
        vard = False
        NumPart = len(dct['lognH'])
        partdct = dct
        # some value checking: array lengths
        if not np.all(np.array([ len(dct['logT'])==NumPart, len(dct['Density'])==NumPart])):
            print("Lengths of lognH (%i), logT (%i) and Density (%i) should match, but don't"%(NumPart, len(dct['logT']), len(dct['Density'])))
        if not np.all(np.array( [ True if not hasattr(dct[elt], '__len__') else\
                                  len(dct[elt])==NumPart or len(dct[elt])==1\
                                  for elt in elts_geq_h ] )):
            print("Element mass fractions must be numbers or arrays of length 1 or matching the logT, lognH and Density")
            return -1
        eltab_base =  'ElementAbundance/%s'
        elts_to_update = list(np.copy(elts_geq_h))
        if 'metallicity' in elts_to_update:
            elts_to_update.remove('metallicity')
        dct.update({eltab_base%(string.capwords(elt)): dct[elt] for elt in elts_to_update}) # allows retrieval of values to be independent of dictionary/Vardict use

    cooldct = findcoolingtables(z, method = method)
    lambda_over_nH2 = np.zeros(NumPart,dtype=np.float32)

    # do the per-element cooling interpolation and number (per_element) or mass (total_metals) fraction rescaling
    if method == 'per_element':
        for elt in elts_geq_he:
            incool = np.zeros(NumPart,dtype=np.float32)
            logTK = cooldct[ol.eltdct_to_ct[elt]]['logTK']
            lognHcm3 = cooldct[ol.eltdct_to_ct[elt]]['lognHcm3']
            table = cooldct[ol.eltdct_to_ct[elt]]['Lambda_over_nH2'] # temperature x density

            # need to compile with some extra options to get this to work: make -f make_emission_only
            print("------------------- C interpolation function output --------------------------\n")
            cfile = ol.c_interpfile

            acfile = ct.CDLL(cfile)
            interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too

            interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                                   ct.c_longlong , \
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                                   ct.c_int,\
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                                   ct.c_int,\
                                   np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


            res = interpfunction(partdct['logT'].astype(np.float32),\
                       partdct['lognH'].astype(np.float32),\
                       ct.c_longlong(NumPart),\
                       np.ndarray.flatten(table.astype(np.float32)),\
                       logTK.astype(np.float32),\
                       ct.c_int(len(logTK)), \
                       lognHcm3.astype(np.float32),\
                       ct.c_int(len(lognHcm3)),\
                       incool \
                      )

            print("-------------- C interpolation function output finished ----------------------\n")

            if res != 0:
                print('Something has gone wrong in the C function: output %s. \n',str(res))
                return -2

            # rescale by ni/nh / (ni/nh)_solar; ni = rho*massfraction_i/mass_i
            if vard:
                dct.readif(eltab_base%string.capwords(elt))
            incool *= partdct[eltab_base%string.capwords(elt)]
            if vard:
                dct.delif(eltab_base%string.capwords(elt), last=last)
            incool /= (ionh.atomw[string.capwords(elt)] * c.u)
            incool /= cooldct['solar_nfrac'][ol.eltdct_to_ct[elt]] #partdct[eltab_base%('Helium')].astype(np.float32)scale by ni/nH / (ni/nH)_solar

            lambda_over_nH2 += incool


        lambda_over_nH2  *= partdct['Density'] # ne/nh / (ne/nh)_solar * element-indepent part of ni/nh / (ni/nh)_solar ( = density / nH)
        if vard:
            cgsfd = dct.CGSconv['Density']
            if cgsfd != 1.:
                lambda_over_nH2  *= cgsfd
        lambda_over_nH2 /= 10**partdct['lognH']
    # end of per element

    elif method == 'total_metals':
        logTK = cooldct['Total_Metals']['logTK']
        lognHcm3 = cooldct['Total_Metals']['lognHcm3']
        table = cooldct['Total_Metals']['Lambda_over_nH2'] # temperature x density

        # need to compile with some extra options to get this to work: make -f make_emission_only
        print("------------------- C interpolation function output --------------------------\n")
        cfile = ol.c_interpfile

        acfile = ct.CDLL(cfile)
        interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too

        interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                               ct.c_longlong , \
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                               ct.c_int,\
                               np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


        res = interpfunction(partdct['logT'].astype(np.float32),\
                   partdct['lognH'].astype(np.float32),\
                   ct.c_longlong(NumPart),\
                   np.ndarray.flatten(table.astype(np.float32)),\
                   logTK.astype(np.float32),\
                   ct.c_int(len(logTK)), \
                   lognHcm3.astype(np.float32),\
                   ct.c_int(len(lognHcm3)),\
                   lambda_over_nH2 \
                  )

        print("-------------- C interpolation function output finished ----------------------\n")

        if res != 0:
            print('Something has gone wrong in the C function: output %s. \n',str(res))
            return -2

        # rescale by ni/nh / (ni/nh)_solar; ni = rho*massfraction_i/mass_i
        if vard:
            eltab = kwargs['abunds']
            if eltab == 'Pt':
                metkey = 'Metallicity'
                dct.readif(metkey)
            elif eltab == 'Sm':
                metkey = 'SmoothedMetallicity'
                dct.readif(metkey)
            else: #dictionary
                metkey = 'metallicity'
                if 'metallicity' in eltab.keys():
                    partdct[metkey] = eltab[metkey]
                else:
                    partdct[metkey] = eltab['Metallicity']
        else: #dictionary
            metkey = 'metallicity'
        if vard:
            dct.readif(metkey)
        lambda_over_nH2 *= partdct[metkey]
        if vard:
            dct.delif(metkey, last=last)
        lambda_over_nH2 /= cooldct['solar_mfrac']['total_metals'] #scale by mi/mH / (mi/mH)_solar (average particle mass is unknown -> cannot get particle density)
    # end of total metals

    # restore wishlist,vardict to pre-call version; clean-up
    if vard:
        eltab = kwargs['abunds']
        if (isinstance(eltab, num.Number) or isinstance(eltab, dict)): # not the actual EAGLE values were used
            if method == 'total_metals' and metkey in partdct.keys(): # may have already been caught by delif
                del partdct[metkey]
        if 'Density' not in wishlist_old and 'Density' in dct.wishlist:
            dct.wishlist.remove('Density')
        dct.delif('Density',last=last)
        if delafter_abunds: # abundances were not the EAGLE output values
            for elt in elts_geq_he:
                if dct.isstored_part(eltab_base%string.capwords(elt)):
                    dct.delif(eltab_base%string.capwords(elt), last=True)


    ## finish rescaling the cooling rates: ne/nh / (ne/nh)_solar
    incool = np.zeros(NumPart,dtype=np.float32)
    fHe      = cooldct['Metal_free']['Hemassfrac']
    lognHcm3 = cooldct['Metal_free']['lognHcm3']
    logTK    = cooldct['Metal_free']['logTK']
    table    = cooldct['Electron_density_over_n_h']['table'] #  fHe x temperature x density

    # get helium number fractions
    if vard:
        dct.readif(eltab_base%('Helium'))
        if not hasattr(partdct[eltab_base%('Helium')], '__len__'): # single-number abundance, while a full array is needed for the C function
            partdct[eltab_base%('Helium')] = partdct[eltab_base%('Helium')]*np.ones(NumPart, dtype=np.float32)

    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3)*len(fHe),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(partdct[eltab_base%('Helium')].astype(np.float32),\
               partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               fHe.astype(np.float32),\
               ct.c_int(len(fHe)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               incool \
              )

    print("-------------- C interpolation function output finished ----------------------\n")
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return -2

    lambda_over_nH2 *= incool

    # 2d interpolation for solar values
    incool = np.zeros(NumPart,dtype=np.float32)

    logTK    = cooldct['Electron_density_over_n_h']['solar_logTK']
    lognHcm3 = cooldct['Electron_density_over_n_h']['solar_lognHcm3']
    table    = cooldct['Electron_density_over_n_h']['solar'] # temperature x density

    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_2d # just a linear interpolator; works for non-emission stuff too

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               incool \
              )
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return -2
    lambda_over_nH2 /= incool


    ## add the metal-free cooling
    incool = np.zeros(NumPart,dtype=np.float32)
    fHe      = cooldct['Metal_free']['Hemassfrac']
    lognHcm3 = cooldct['Metal_free']['lognHcm3']
    logTK    = cooldct['Metal_free']['logTK']
    table    = cooldct['Metal_free']['Lambda_over_nH2'] # fHe x temperature x density


    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3)*len(fHe),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(partdct[eltab_base%('Helium')].astype(np.float32),\
               partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               fHe.astype(np.float32),\
               ct.c_int(len(fHe)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               incool \
              )

    print("-------------- C interpolation function output finished ----------------------\n")

    lambda_over_nH2 += incool
    del incool

    if vard:
        dct.delif('lognH', last=last)
        dct.delif('logT',  last=last)
        dct.delif(eltab_base%('Helium'), last=last)
        dct.add_part('Lambda_over_nH2', lambda_over_nH2)

    return lambda_over_nH2



def find_coolingtimes(z,dct, method = 'per_element', **kwargs):
    '''
    !! Comparison to Wiersma, Schaye, & Smith (2009), where the used tables
    come from and which the calculations should match, shows that cooling
    contours might be off by ~0.1-0.2 dex (esp. their fig. 4, where the
    intersecting lines mean differences are more visible than they would be in
    other plots)

    arguments: see find_coolingrates

    returns internal energy / cooling rate
    negative values -> cooling times
    postive values -> heating times
    '''
    # cooling tables have mean particle mass as a function of temperature, density, and helium fraction (mu)


    if isinstance(dct, pc.Vardict):
        vard = True
        partdct = dct.particle
        if 'last' in kwargs.keys():
            last = kwargs['last']
        else:
            last = True
        if not dct.isstored_part('logT'):
            dct.getlogT(last=last,logT = kwargs['T4EOS'])
        # modify wishlist for our purposes
        wishlist_old = list(np.copy(dct.wishlist))
        dct.wishlist = list(set(dct.wishlist +  ['Density', 'lognH', 'logT']))
        if not dct.isstored_part('lognH'):
            if 'last' in kwargs.keys():
                skwargs = kwargs.copy()
                del skwargs['last']
            else:
                skwargs = kwargs
            dct.getlognH(last=False,**skwargs)
        NumPart = len(dct.particle['lognH'])

        eltab_base = kwargs['abunds']
        delafter_abunds = False
        if not (isinstance(eltab_base, str) or isinstance(eltab_base, dict)): # make_maps-style tuple: none of that nonsense here
            eltab_base = eltab_base[0]
        if isinstance(eltab_base,str):
            if eltab_base == 'Sm' or 'SmoothedElementAbundance' in eltab_base: # example abundance is accepted
                eltab_base = 'SmoothedElementAbundance/%s'
            elif eltab_base == 'Pt' or 'ElementAbundance' == eltab_base[:16]: # example abundance is accepted
                eltab_base = 'ElementAbundance/%s'
            else:
                print('eltab value %s is not a valid option'%eltab_base)
                return -1
        elif isinstance(eltab_base, dict):
            delafter_abunds = True
            eltab_base =  'ElementAbundance/%s'
            if 'Helium' in kwargs['abunds']:
                hekey = 'Helium'
            else:
                hekey = 'helium'
            dct.particle.update({eltab_base%('Helium'): kwargs['abunds'][hekey] * np.ones(NumPart, dtype=np.float32)})
        dct.wishlist += [eltab_base%('Helium')]

    else:
        vard = False
        NumPart = len(dct['lognH'])
        partdct = dct
        # some value checking: array lengths
        if not np.all(np.array([ len(dct['logT'])==NumPart, len(dct['Density'])==NumPart])):
            print("Lengths of lognH (%i), logT (%i) and Density (%i) should match, but don't"%(NumPart, len(dct['logT']), len(dct['Density'])))
        dct['ElementAbundance/Helium'] = dct['helium']
        eltab_base =  'ElementAbundance/%s'

    Lambda = find_coolingrates(z,dct, method=method, **kwargs)
    Lambda *= 10**(2*partdct['lognH'])



    # get mu
    cooldct = findcoolingtables(z, method = method)
    mu = np.zeros(NumPart,dtype=np.float32)
    fHe      = cooldct['Metal_free']['Hemassfrac']
    lognHcm3 = cooldct['Metal_free']['lognHcm3']
    logTK    = cooldct['Metal_free']['logTK']
    table    = cooldct['Metal_free']['mu'] # fHe x temperature x density

    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_3d # just a linear interpolator

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe)*len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(fHe),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]


    res = interpfunction(partdct[eltab_base%('Helium')].astype(np.float32),\
               partdct['logT'].astype(np.float32),\
               partdct['lognH'].astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(table.astype(np.float32)),\
               fHe.astype(np.float32),\
               ct.c_int(len(fHe)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               mu \
              )

    print("-------------- C interpolation function output finished ----------------------\n")
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return -2
    if vard:
        dct.delif('Lambda_over_nH2', last=last)

    tcool =  partdct['Density']/mu*(10**partdct['logT']) #internal energy; mu is metal-free, so there will be some small errors in n when there are metals present
    cgc = (1.5*c.boltzmann/c.u)
    if vard:
        cgc *= dct.CGSconv['Density'] # to CGS units
    tcool *= cgc # to CGS units
    del mu

    tcool /= Lambda # tcool = Uint/Lambda_over_V
    del Lambda

    # clean up wishlist, add end result to vardict
    if vard:
        if 'Density' not in wishlist_old and 'Density' in dct.wishlist:
            dct.wishlist.remove('Density')
            dct.delif('Density', last=last)
        if eltab_base%('Helium') not in wishlist_old and eltab_base%('Helium') in dct.wishlist:
            dct.wishlist.remove(eltab_base%('Helium'))
            dct.delif(eltab_base%('Helium'), last=last)
        if 'lognH' not in wishlist_old and 'lognH' in dct.wishlist:
            dct.wishlist.remove('lognH')
            dct.delif('lognH', last=last)
        if 'logT' not in wishlist_old and 'logT' in dct.wishlist:
            dct.wishlist.remove('logT')
            dct.delif('logT', last=last)
        if delafter_abunds: # abundances were not the EAGLE output values
            if dct.isstored_part(eltab_base%('Helium')):
                dct.delif(eltab_base%('Helium'), last=True)
        dct.add_part('tcool', tcool)
    return tcool



def getBenOpp1chemabundtables(vardict,excludeSFR,eltab,hab,ion,last=True,updatesel=True,misc=None):
    # ion names used here and in table naming -> ions in ChemicalAbundances table
    print('Getting ion balance from simulation directly (BenOpp1)')
    iontranslation = {'c2':  'CarbonII',\
                      'c3':  'CarbonIII',\
                      'c4':  'CarbonIV',\
                      'h1':  'HydrogenI',\
                      'mg2': 'MagnesiumII',\
                      'ne8': 'NeonVIII',\
                      'n5':  'NitrogenV',\
                      'o6':  'OxygenVI',\
                      'o7':  'OxygenVII',\
                      'o8':  'OxygenVIII',\
                      'si2': 'SiliconII',\
                      'si3': 'SiliconIII',\
                      'si4': 'SiliconIV'}
    mass_over_h = {'hydrogen':  1.,\
                   'carbon':    c.atomw_C/c.atomw_H,\
                   'magnesium': c.atomw_Mg/c.atomw_H,\
                   'neon':      c.atomw_Ne/c.atomw_H,\
                   'nitrogen':  c.atomw_N/c.atomw_H,\
                   'oxygen':    c.atomw_O/c.atomw_H,\
                   'silicon':   c.atomw_Si/c.atomw_H}
    if ion not in iontranslation.keys():
        print('ChemicalAbundances tables are not available ')
    # store chemical abundances as ionfrac, to match what is used otherwise
    # chemical abundance arrays may contain NaN values. Set those to zero.
    vardict.readif('ChemicalAbundances/%s'%(iontranslation[ion]),region = 'auto',rawunits = True,out =False, setsel = None,setval = None)
    if ion != 'h1':
        vardict.readif(eltab, region='auto', rawunits=True, out=False, setsel=None, setval=None)
        vardict.readif(hab, region='auto', rawunits=True, out=False, setsel=None, setval=None)
        vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])] /=\
            vardict.particle[eltab]/(mass_over_h[ol.elements_ion[ion]]*vardict.particle[hab]) # convert num. dens. rel to hydrogen to num dens. rel. to total element (eltab and hab are mass fractions)

        vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])][vardict.particle[eltab]==0] = 0. #handle /0 errors properly: no elements -> no ions of that element
        vardict.add_part('ionfrac',vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])])
        # correct zero values: NaN values -> zero, infinite values (element abundance is zero) -> zero
        vardict.particle['ionfrac'][np.isnan(vardict.particle['ionfrac'])] = 0.
        vardict.particle['ionfrac'][np.isinf(vardict.particle['ionfrac'])] = 0.

    else:
        vardict.add_part('ionfrac',vardict.particle['ChemicalAbundances/%s'%(iontranslation[ion])])
    vardict.delif('ChemicalAbundances/%s'%(iontranslation[ion]),last=True) # got mangled -> remove to avoid confusion
    if hab != eltab: # we still need eltab later
        vardict.delif(hab, last=last)



#### i/o processing for projections and particle selection

def translate(old_dct, old_nm, centre, boxsize, periodic):

    if type(boxsize) == float: # to handle velocity space slicing with the correct periodicity
        boxsize = (boxsize,)*3

    if not periodic:
        print('Translating particle positions: (%.2f, %.2f, %.2f) -> (0, 0, 0) Mpc' \
          % (centre[0], centre[1], centre[2]))
    else:
        print('Translating particle positions: (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f) Mpc' \
          % (centre[0], centre[1], centre[2], 0.5*boxsize[0], 0.5*boxsize[1], 0.5*boxsize[2]))

    # translates old coordinates into coordinates wrt centre
    # taking into account the periodicity of the box
    # for periodic boundary conditions, the coordinates must be translated into the [0, BoxSize] range for compatibility with the C function

    # change arrays in place and use dict to avoid copying and save memory
    # docs and test: broadcasting checks trailing dimensions first, so boxsize/center operation should work even if coordinate array has 3 elements

    centre = np.array(centre)
    boxsize = np.array(boxsize)
    if not periodic:
#        old_dct[old_nm][:,0] = (old_dct[old_nm][:,0] - centre[0] + 0.5*boxsize[0])%boxsize[0] - 0.5*boxsize[0]
#        old_dct[old_nm][:,1] = (old_dct[old_nm][:,1] - centre[1] + 0.5*boxsize[1])%boxsize[1] - 0.5*boxsize[1]
#        old_dct[old_nm][:,2] = (old_dct[old_nm][:,2] - centre[2] + 0.5*boxsize[2])%boxsize[2] - 0.5*boxsize[2]
        old_dct[old_nm] += 0.5*boxsize - centre
        old_dct[old_nm] %= boxsize
        old_dct[old_nm] -= 0.5*boxsize

    if periodic:
#        old_dct[old_nm][:,0] = (old_dct[old_nm][:,0] - centre[0] + 0.5*boxsize[0])%boxsize[0]
#        old_dct[old_nm][:,1] = (old_dct[old_nm][:,1] - centre[1] + 0.5*boxsize[1])%boxsize[1]
#        old_dct[old_nm][:,2] = (old_dct[old_nm][:,2] - centre[2] + 0.5*boxsize[2])%boxsize[2]
        old_dct[old_nm] += 0.5*boxsize - centre
        old_dct[old_nm] %= boxsize

    old_dct[old_nm] = old_dct[old_nm].astype(np.float32)
    return None



def nameoutput(vardict, ptypeW, simnum, snapnum, version, kernel,
               npix_x, L_x, L_y, L_z, centre, BoxSize, hconst,
               excludeSFRW, excludeSFRQ, velcut, sylviasshtables, bensgadget2tables,
               ps20tables, ps20depletion,
               axis, var, abundsW, ionW, parttype, ptypeQ, abundsQ, ionQ, quantityW, quantityQ,
               simulation, LsinMpc, halosel, kwargs_halosel, misc, hdf5):
    # some messiness is hard to avoid, but it's contained
    # Ls and centre have not been converted to Mpc when this function is called

    # box and axis
    zcen = ''
    xypos = ''
    if LsinMpc:
        Lunit = ''
        hfac = 1.
    else:
        Lunit = '-hm1'
        hfac = hconst
    if axis == 'z':
        if L_z*hfac < BoxSize * hconst**-1:
            zcen = '_zcen%s%s' %(str(centre[2]),Lunit)
        if L_x*hfac < BoxSize * hconst**-1 or L_y*hfac < BoxSize * hconst**-1:
            xypos = '_x%s-pm%s%s_y%s-pm%s%s' %(str(centre[0]), str(L_x), 
                                               Lunit, 
                                               str(centre[1]), str(L_y),
                                               Lunit)
        sLp = str(L_z)

    elif axis == 'y':
        if L_y*hfac < BoxSize * hconst**-1:
            zcen = '_ycen%s%s' % (str(centre[1]),Lunit)
        if L_x*hfac < BoxSize * hconst**-1 or L_z*hfac < BoxSize * hconst**-1:
            xypos = '_z%s-pm%s%s_x%s-pm%s%s' %(str(centre[2]), str(L_z), 
                                               Lunit,
                                               str(centre[0]), str(L_x),
                                               Lunit)
        sLp = str(L_y)

    elif axis == 'x':
        if L_x*hfac < BoxSize * hconst**-1:
            zcen = '_xcen%s%s' % (str(centre[0]),Lunit)
        if L_y*hfac < BoxSize * hconst**-1 or L_z*hfac < BoxSize * hconst**-1:
            xypos = '_y%s-pm%s%s_z%s-pm%s%s' %(str(centre[1]), str(L_y), 
                                               Lunit, 
                                               str(centre[2]), str(L_z), 
                                               Lunit)
        sLp = str(L_x)


    axind = '_%s-projection' %axis

    # EOS particle handling
    if excludeSFRW == True:
        SFRindW = '_noEOS'
    elif excludeSFRW == False:
        SFRindW = '_wiEOS'
    elif excludeSFRW == 'T4':
        SFRindW = '_T4EOS'
    elif excludeSFRW == 'from':
        SFRindW = '_fromSFR'
    elif excludeSFRW == 'only':
        SFRindW = '_onlyEOS'

    if excludeSFRQ == True:
        SFRindQ = '_noEOS'
    elif excludeSFRQ == False:
        SFRindQ = '_wiEOS'
    elif excludeSFRQ == 'T4':
        SFRindQ = '_T4EOS'
    elif excludeSFRQ == 'from':
        SFRindQ = '_fromSFR'
    elif excludeSFRQ == 'only':
        SFRindQ = '_onlyEOS'

    if sylviasshtables and ptypeW == 'coldens':
        iontableindW = '_iontab-sylviasHM12shh'
    elif bensgadget2tables and ptypeW == 'coldens':
        iontableindW = '_iontab-bensgagdet2'
    elif ps20tables and ptypeW in ['coldens', 'emission']:
        iontableindW = '_iontab-PS20'
        iontab = ol.iontab_sylvia_ssh.split('/')[-1]
        iontab = iontab[:-5] # remove '.hdf5'
        iontab = iontab.replace('_', '-')
        #iontableindW = iontableindW + '-' + 'iontab'
        if ps20depletion:
            iontableindW += '_depletion-T'
        else:
            iontableindW += '_depletion-F'
    else:
        iontableindW = ''
    if sylviasshtables and ptypeQ == 'coldens':
        iontableindQ = '_iontab-sylviasHM12shh'
    elif bensgadget2tables and ptypeQ == 'coldens':
        iontableindQ = '_iontab-bensgagdet2'
    elif ps20tables and ptypeQ in ['coldens', 'emission']:
        iontableindQ = '_iontab-PS20-'
        iontab = ol.iontab_sylvia_ssh.split('/')[-1]
        iontab = iontab[:-5] # remove '.hdf5'
        iontab = iontab.replace('_', '-')
        iontableindQ = iontableindQ + '-' + 'iontab'
        if ps20depletion:
            iontableindQ += '_depletion-T'
        else:
            iontableindQ += '_depletion-F'
    else:
        iontableindQ = ''
        
    # abundances
    if ptypeW in ['coldens', 'emission']:
        if abundsW[0] not in ['Sm','Pt']:
            sabundsW = '%smassfracAb'%str(abundsW[0])
        else:
            sabundsW = abundsW[0] + 'Ab'
        if type(abundsW[1]) == float:
            sabundsW = sabundsW + '-%smassfracHAb'%str(abundsW[1])
        elif abundsW[1] != abundsW[0]:
            sabundsW = sabundsW + '-%smassfracHAb'%abundsW[1]

    if ptypeQ in ['coldens', 'emission']:
        if abundsQ[0] not in ['Sm','Pt']:
            sabundsQ = str(abundsQ[0]) + 'massfracAb'
        else:
            sabundsQ = abundsQ[0] + 'Ab'
        if type(abundsQ[1]) == float:
            sabundsQ = sabundsQ + '-%smassfracHAb'%str(abundsQ[1])
        elif abundsQ[1] != abundsQ[0]:
            sabundsQ = sabundsQ + '-%sHAb'%abundsQ[1]


    # miscellaneous: ppv/ppp box, simulation, particle type
    if velcut == True:
        vind = '_velocity-sliced'
    elif isinstance(velcut, tuple):
        if velcut[0] == 0.:
            vind = '_velocity_pm%s-kmps'%(str(velcut[1]))
        else:
            if velcut[0]<0:
                vind0 = str(velcut[0])
            else:
                vind0 = '+%s'%(str(velcut[0]))
            vind = '_velocity_%s-pm%s-kmps'%(vind0, str(velcut[1]))
    else:
        vind = ''

    if var != 'REFERENCE':
        ssimnum = str(simnum) +var
    else:
        ssimnum = str(simnum)
    if simulation == 'bahamas':
        ssimnum = 'BA-%s'%ssimnum
    elif simulation == 'eagle-ioneq':
        ssimnum = 'EA-ioneq-%s'%ssimnum
    elif simulation == 'c-eagle-hydrangea':
        ssimnum = 'CEH-%s'%ssimnum

    if parttype != '0':
        sparttype = '_PartType%s'%parttype
    else:
        sparttype = ''

    #avoid / in file names
    if ptypeW == 'basic':
        squantityW = quantityW
        squantityW = squantityW.replace('/','-')
    if ptypeQ == 'basic':
        squantityQ = quantityQ
        squantityQ = squantityQ.replace('/','-')
    
    # halo selections
    if halosel is not None:
        halostr = '_' + selecthaloparticles(vardict, halosel, nameonly=True,
                                            last=False, **kwargs_halosel)
        if 'label' in kwargs_halosel.keys():
            if kwargs_halosel['label'] is not None:
                halostr = '_halosel-%s-endhalosel'%kwargs_halosel['label']
    else:
        halostr = ''
        
    # putting it together: ptypeQ = None is set to get resfile for W
    if ptypeQ is None: #output outputW name
        if ptypeW == 'coldens' or ptypeW == 'emission':
            base = '{ptype}_{ion}{iontab}_{sim}_{snap}_test{ver}_{abunds}' + \
                   '_{kernel}Sm_{npix}pix_{depth}slice'
            base = base.format(ptype=ptypeW, ion=ionW.replace(' ', '-'),
                               iontab=iontableindW,
                               sim=ssimnum, snap=snapnum, ver=str(version),
                               abunds=sabundsW, kernel=kernel, npix=npix_x,
                               depth=sLp)
            resfile = ol.ndir + base + zcen + xypos + axind + SFRindW +\
                      halostr + vind

        elif ptypeW == 'basic':
            base = '{qW}{parttype}_{sim}_{snap}_test{ver}' + \
                   '_{kernel}Sm_{npix}pix_{depth}slice'
            base = base.format(qW=squantityW, parttype=sparttype, sim=ssimnum, 
                               snap=snapnum, ver=str(version), kernel=kernel, 
                               npix=npix_x, depth=sLp)
            resfile = ol.ndir + base + zcen + xypos + axind + SFRindW +\
                      halostr + vind

    if ptypeQ is not None: # naming for quantityQ output
        qty_base = '{ptype}_{ion}_{abunds}{iontab}{sfgas}'
        if ptypeQ == 'basic':
            squantityQ = squantityQ + SFRindQ
        else:
            squantityQ = qty_base.format(ptype=ptypeQ, 
                                         ion=ionQ.replace(' ', '-'),
                                         abunds=sabundsQ, iontab=iontableindQ,
                                         sfgas=SFRindQ)
        if ptypeW == 'basic':
            squantityW = squantityW + SFRindW
        else:
            squantityW = qty_base.format(ptype=ptypeW, 
                                         ion=ionW.replace(' ', '-'),
                                         abunds=sabundsW, iontab=iontableindW,
                                         sfgas=SFRindW)
        base = '{qQ}_{qW}{parttype}_{sim}_{snap}_test{ver}_{abunds}' + \
               '_{kernel}Sm_{npix}pix_{depth}slice'
        base = base.format(qQ=squantityQ, qW=squantityW, parttype=sparttype,
                            sim=ssimnum, snap=snapnum, ver=str(version),
                            kernel=kernel, npix=npix_x, depth=sLp)
        resfile = ol.ndir + base + zcen + xypos + axind + halostr + vind


    #if misc is not None:
    #    # key-value expansion means names are not strictly deterministic for misc options, but the naming should be unambiguous, so that's ok
    #    namelist = [(key,misc[key]) for key in misc.keys()]
    #    namelist = ['%s-%s'%(str(pair[0]),str(pair[1])) for pair in namelist]
    #    misctail = '_%s'%('_'.join(namelist))
    #else:
    #    misctail = ''

    if misc is not None: # if if if : if we want to use chemical abundances from Ben' Oppenheimer's recal variations
        if 'usechemabundtables' in misc:
            if misc['usechemabundtables'] == 'BenOpp1':
                resfile = resfile + '_BenOpp1-chemtables'
    if hdf5:
        resfile = resfile + '.hdf5'
    #resfile = resfile + misctail
    if ptypeQ == None:
        print('saving W result to: '+resfile+'\n')
    else:
        print('saving Q result to: '+resfile+'\n')
    return resfile



def inputcheck(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y,
         ptypeW,
         ionW, abundsW, quantityW,
         ionQ, abundsQ, quantityQ, ptypeQ,
         excludeSFRW, excludeSFRQ, parttype,
         theta, phi, psi,
         sylviasshtables, bensgadget2tables,
         ps20tables, ps20depletion,
         var, axis, log, velcut,
         periodic, kernel, saveres,
         simulation, LsinMpc,
         select, misc, ompproj, numslices, halosel, kwargs_halosel,
         hdf5, override_simdatapath):

    '''
    Checks the input to make_map();
    This is not an exhaustive check; it does handle the default/auto options
    return numbers are not ordered; just search <return ##>
    '''
    # max used number: 55

    # basic type and valid option checks
    if not isinstance(var, str):
        print('%s should be a string.\n'%var)
        return 1
    if kernel not in ol.kernel_list:
        print('%s is not a kernel option. Options are: \n' % kernel)
        print(ol.kernel_list)
        return 2
    if axis not in ['x','y','z']:
        print('Axis must be "x", "y", or "z".')
        return 11
    if (theta,psi,phi) != (0.,0.,0.):
        print('Warning: rotation is not implemented in this code!\n  Using zero rotation version')
    if type(periodic) != bool:
        print('periodic should be True or False.\n')
        return 12
    if not isinstance(log, bool):
        print('log should be True or False.\n')
        return 13
    if not (isinstance(numslices, int) or numslices is None):
        print('numslices should be an integer or None.\n')
        return 37
    if not isinstance(sylviasshtables, bool):
        print('sylviasshtables should be True or False.\n')
        return 35
    elif sylviasshtables:
        if not (ptypeW == 'coldens' or ptypeQ == 'coldens'):
            print('Warning: setting for ion tables will not be used in this calculation; e.g. self-shielded emission is not available')
        if ionW == 'hneutralssh' or ionQ == 'hneutralssh':
            print("Neutral hydrogen is not currenty available from Sylvia's tables")
            return 36
    if not isinstance(bensgadget2tables, bool):
        print('bensgadget2tables should be True or False.\n')
        return 46
    elif bensgadget2tables:
        if not (ptypeW == 'coldens' or ptypeQ == 'coldens'):
            print('Warning: setting for ion tables will not be used in this calculation; only absorption is available')
        if ionW == 'hneutralssh' or ionQ == 'hneutralssh':
            print("Neutral hydrogen is not currenty available from Ben's gadget 2 tables")
            return 47
    if not isinstance(ps20tables, bool):
        print('ps20tables should be True or False.\n')
        return 48  
    if ps20tables:
        if not os.path.isfile(ol.iontab_sylvia_ssh):
            print('PS20 table {} was not found'.format(ol.iontab_sylvia_ssh))
            return 52
    if ps20tables and (ptypeQ == 'emission' or ptypeW == 'emission'):
        if not os.path.isfile(ol.emtab_sylvia_ssh):
            print('PS20 emission table {} was not found'.format(ol.emtab_sylvia_ssh))
            return 53
        iontab = ol.iontab_sylvia_ssh.split('/')[-1]
        iontab = iontab[:-5]
        emtab = ol.emtab_sylvia_ssh.split('/')[-1]
        emtab = emtab[:-5]
        if emtab != iontab + '_lines':
            print('PS20 emission and absorption tables do not match:')
            print(ol.emtab_sylvia_ssh)
            print(ol.iontab_sylvia_ssh)
            return 51
    if not isinstance(ps20depletion, bool):
        print('ps20depletion should be True or False.\n')
        return 49
    if sylviasshtables + bensgadget2tables + ps20tables > 1:
        print('Cannot use more than one of sylviasshtables, sp20tables, and bensgadget2tables; choose one')
        return 50
    if not isinstance(saveres, bool):
        print('saveres should be True or False.\n')
        return 14
    if (not isinstance(velcut, bool)) and not isinstance(velcut, num.Number):
        numtuple = False
        if hasattr(velcut, '__len__'):
            if len(velcut) == 2:
                if isinstance(velcut[0], num.Number) and isinstance(velcut[1], num.Number):
                    numtuple = True
                    if velcut[1] < 0:
                        velcut = list(velcut)
                        velcut[1] = -1*velcut[1]
        if numtuple:
            velcut = tuple(velcut)
        else:
            print('velcut should be True or False, a number, or a (number, number) tuple.\n')
            return 15
    elif not isinstance(velcut, bool): # it's a number
        if velcut < 0:
            velcut = -1*velcut
        velcut = (0, velcut)
    if not isinstance(hdf5, bool):
        print('hdf5 should be a boolean.\n')
        return 44
    if override_simdatapath == False:
        override_simdatapath = None
    if not (isinstance(override_simdatapath, str) or override_simdatapath is None):
        print('override_simdatapath should be None or a string containing specifying directory containing the snapshot.../, groups.../, etc. directories')
        return 45
    if not isinstance(snapnum, int):
        print('snapnum should be an integer.\n')
        return 21
    if (not isinstance(centre[0], num.Number)) or (not isinstance(centre[0], num.Number)) or (not isinstance(centre[0], num.Number)):
        print('centre should contain 3 floats')
        return 29
    else:
        centre = [float(centre[0]), float(centre[1]), float(centre[2])]
    if not isinstance(ompproj, bool):
        if ompproj == 1:
            ompproj = True
        elif ompproj == 0:
            ompproj = False
        else:
            print('ompproj should be a boolean or 0 or 1')
            return 32
    if halosel is not None:
        if isinstance(halosel, tuple):
            if isinstance(halosel[0], tuple): # tuple of tuples -> list of tuples
                halosel = list(halosel)
            else: # single tuple -> enclose in list to match expected format
                halosel = [halosel]
        elif hasattr(halosel, '__iter__'): # some iterable -> convert to list
            halosel = list(halosel)
        else:
            print('Halosel should be a list of tuples')
            return 38
    if kwargs_halosel is None:
        kwargs_halosel = {}
    else:
        allowed_kwargs = ['aperture', 'mdef', 'exclsatellites', 'allinR200c', 'label']
        if not np.all([key in allowed_kwargs for key in kwargs_halosel.keys()]):
            print('allowed kwargs_halosel are %s'%allowed_kwargs)
            print('input kwargs_halosel were %s'%list(kwargs_halosel.keys()))
            return 39
        if 'aperture' in kwargs_halosel.keys():
            if not isinstance(kwargs_halosel['aperture'], num.Number):
                print('kwargs_halosel: "aperture" must be a number (int); was %s'%(kwargs_halosel['aperture']))
                return 40     
        if 'exclsatellites' in kwargs_halosel.keys():
            if not isinstance(kwargs_halosel['exclsatellites'], bool):
                print('kwargs_halosel: "exclsatellties" must be a boolean; was %s'%(kwargs_halosel['exclsatellites']))
                return 41
        if 'allinR200' in kwargs_halosel.keys():
            if not isinstance(kwargs_halosel['allinR200'], bool):
                print('kwargs_halosel: "allinR200" must be a boolean; was %s'%(kwargs_halosel['allinR200']))
                return 42   
        if 'mdef' in kwargs_halosel.keys():
            if not isinstance(kwargs_halosel['mdef'], str):
                print('kwargs_halosel: "mdef" must be a string; was %s'%(kwargs_halosel['mdef']))
                return 43   
        if 'label' in kwargs_halosel.keys():
            if not isinstance(kwargs_halosel['label'], str):
                print('kwargs_halosel: "label" must be a string; was %s'%(kwargs_halosel['label']))
                return 43   
            
    if simulation not in ['eagle', 'bahamas', 'Eagle', 'Bahamas', 'EAGLE',
                          'BAHAMAS', 'eagle-ioneq', 'c-eagle-hydrangea', 'CE',
                          'hydrangea', 'CEH']:
        print('Simulation %s is not a valid choice; should be "eagle", "eagle-ioneq", "c-eagle-hydrangea" or "bahamas"'%str(simulation))
        return 30
    elif simulation == 'Eagle' or simulation == 'EAGLE':
        simulation = 'eagle'
        print('Preferred form of simulation names is all lowercase (%s)'%simulation)
    elif simulation == 'Bahamas' or simulation == 'BAHAMAS':
        simulation = 'bahamas'
        print('Preferred form of simulation names is all lowercase (%s)'%simulation)
    elif simulation in ['c-eagle-hydrangea', 'CE', 'hydrangea', 'CEH']:
        simulation = 'c-eagle-hydrangea'

    if (simulation in ['eagle', 'eagle-ioneq', 'c-eagle-hydrangea']) and LsinMpc is None:
        LsinMpc = True
    elif simulation == 'bahamas' and LsinMpc is None:
        LsinMpc = False

    if simulation in ['eagle', 'eagle-ioneq', 'bahamas'] and not isinstance(simnum, str):
        print('simnum should be a string')
        return 22
    elif simulation == 'c-eagle-hydrangea' and not isinstance(simnum, int):
        if isinstance(simnum, str):
            if simnum.isdigit():
                simnum = int(simnum)
            else:
                print('simnum should be a string integer or integer')
                return 22
        else:
            print('simnum should be a string integer or integer')
            return 22
    if simulation == 'eagle' and (len(simnum) != 10 or simnum[0] != 'L' or simnum[5] != 'N'):
        print('incorrect simnum format %s; should be L####N#### for eagle\n'%simnum)
        return 23
    elif simulation == 'bahamas' and (simnum[0] != 'L' or simnum[4] != 'N'):
        print('incorrect simnum format %s; should be L*N* for bahamas\n'%simnum)
        return 31
    elif simulation == 'eagle-ioneq' and simnum != 'L0025N0752':
        print('For eagle-ioneq, only L0025N0752 is avilable')
        return 33

    centre = [float(centre[0]),float(centre[1]),float(centre[2])]
    if not (isinstance(L_x, num.Number) and isinstance(L_y, num.Number) and isinstance(L_z, num.Number)):
        print('L_x, L_y, and L_z should be floats')
        return 24
    L_x, L_y, L_z = (float(L_x),float(L_y),float(L_z))
    if (not isinstance(npix_x, int)) or (not isinstance(npix_y, int)) or npix_x < 1 or npix_y < 1:
        print('npix_x, npix_y should be positive integers')
        return 25

    # combination-dependent checks
    if var == 'auto':
        if simnum == 'L0025N0752' and simulation != 'eagle-ioneq':
            var = 'RECALIBRATED'
        else:
            var = 'REFERENCE'

    if misc is not None: # if if if : if we want to use chemical abundances from Ben' Oppenheimer's recal variations
        if 'usechemabundtables' in misc:
            if misc['usechemabundtables'] == 'BenOpp1':
                if simulation != 'eagle-ioneq':
                    print('chemical abundance tables are only avaiable for the eagle-ioneq simulation')
                    return 34
                if ptypeW == 'coldens':
                    if 'Sm' in abundsW:
                        print('chemical abundance tables are only for particle abundances')
                        return 34
                    elif abundsW in ['auto', None]:
                        abundsW = 'Pt'
                if ptypeQ == 'coldens':
                    if 'Sm' in abundsQ:
                        print('chemical abundance tables are only for particle abundances')
                        return 34
                    elif abundsQ in ['auto', None]:
                        abundsQ = 'Pt'


    iseltQ, iseltW = (False, False)

    if ptypeW not in ['emission', 'coldens', 'basic']:
        print('ptypeW should be one of emission, coldens, or basic (str).\n')
        return 3
    elif ptypeW in ['emission', 'coldens']:
        parttype = '0'
        if ionW in ol.elements_ion.keys():
            iseltW = False
        elif ionW in ol.elements and ptypeW == 'coldens':
            iseltW = True
        elif ps20tables:
            try:
                linetable_PS20(ionW, 0.0, emission=ptypeW=='emission')
            except ValueError as err:
                print(err)
                print('Invalid PS20 ion {}'.format(ionW))
                return 55
            iseltW = False
        else:
            print('%s is an invalid ion option for ptypeW %s\n'%(ionW,ptypeW))
            return 26
        if not isinstance(abundsW, (list, tuple, np.ndarray)):
            abundsW = [abundsW,'auto']
        else:
            abundsW = list(abundsW) # tuple element assigment is not allowed, sometimes needed
        if abundsW[0] not in ['Sm','Pt','auto']:
            if not isinstance(abundsW[0], num.Number):
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 4
            elif iseltW:
                abundsW[0] = abundsW[0] * ol.solar_abunds_ea[ionW]
            else:
                abundsW[0] = abundsW[0] * ol.solar_abunds_ea[ol.elements_ion[ionW]]
        elif abundsW[0] == 'auto':
            if ptypeW == 'emission':
                abundsW[0] = 'Sm'
            else:
                abundsW[0] = 'Pt'
        if abundsW[1] not in ['Sm','Pt','auto']:
            if not isinstance(abundsW[1], num.Number):
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 30
        elif abundsW[1] == 'auto':
            if isinstance(abundsW[0], num.Number):
                abundsW[1] = 0.752 # if element abundance is fixed, use primordial hydrogen abundance
            else:
                abundsW[1] = abundsW[0]
    else:
        if quantityW is None:
            print('For pytpeW basic, quantityW must be specified.\n')
            return 5
        elif not isinstance(quantityW, str):
            print('quantityW must be a string.\n')
            return 6


    if ptypeQ not in ['emission', 'coldens', 'basic', None]:
        print('ptypeQ should be one of emission, coldens, basic (str), or None.\n')
        return 7

    elif ptypeQ in ['emission','coldens']:
        parttype = '0'
        if ionQ in ol.elements_ion.keys():
            iseltQ = False
        elif ionQ in ol.elements and ptypeQ == 'coldens':
            iseltQ = True
        elif ps20tables:
            try:
                linetable_PS20(ionQ, 0.0, emission=ptypeQ=='emission')
            except ValueError as err:
                print(err)
                print('Invalid PS20 ion {}'.format(ionQ))
                return 55
            iseltQ = False
        else:
            print('%s is an invalid ion option for ptypeQ %s\n'%(ionQ,ptypeQ))
            return 8

        if not isinstance(abundsQ, (list, tuple, np.ndarray)):
            abundsQ = [abundsQ, 'auto']
        else:
            abundsQ = list(abundsQ)
        if abundsQ[0] not in ['Sm','Pt','auto']:
            if not isinstance(abundsQ[0], num.Number):
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 9
            elif iseltQ:
                abundsQ[0] = abundsQ[0] * ol.solar_abunds_ea[ionQ]
            else:
                abundsQ[0] = abundsQ[0] * ol.solar_abunds_ea[ol.elements_ion[ionQ]]
        elif abundsQ[0] == 'auto':
            if ptypeQ == 'emission':
                abundsQ[0] = 'Sm'
            else:
                abundsQ[0] = 'Pt'
        if abundsQ[1] not in ['Sm','Pt','auto']:
            if not isinstance(abundsQ[1], num.Number):
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 28
        elif abundsQ[1] == 'auto':
            if isinstance(abundsQ[0], num.Number):
                abundsQ[1] = 0.752 # if element abundance is fixed, use primordial hydrogen abundance
            else:
                abundsQ[1] = abundsQ[0]

    elif (not isinstance(quantityQ, str)) and quantityQ is not None:
        print('quantityQ must be a string or None.\n')
        return 27


    if ptypeW == 'basic' or ptypeQ == 'basic':
        if parttype not in ['0','1','4','5']: # parttype only matters if it is used
            if parttype in [0,1,4,5]:
                parttype = str(parttype)
            else:
                print('parttype should be "0", "1", "4", or "5" (str).\n')
                return 16

    if excludeSFRW not in [True,False,'T4','only']:
        if excludeSFRW != 'from':
            print('Invalid option for excludeSFRW: %s'%excludeSFRW)
            return 17
        elif not (ptypeW == 'emission' and ionW == 'halpha'):
            excludeSFRW = 'only'
            print('Unless calculation is for halpha emission, fromSFR will default to onlySFR.\n')

    if ptypeQ is not None:
        if (excludeSFRW in [False,'T4']) and (excludeSFRQ not in [False, 'T4']):
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFRW,excludeSFRQ))
            return 18
        elif excludeSFRW in ['from','only']:
            if excludeSFRQ not in ['from', 'only']:
                print('ExcludeSFR options %s and %s are not compatible'%(excludeSFRW,excludeSFRQ))
                return 19
            elif excludeSFRQ == 'from' and not (ptypeQ == 'emission' and ionQ == 'halpha'):
                excludeSFRQ = 'only'
                print('Unless calculation is for halpha emission, fromSFR will default to onlySFR.\n')

        elif excludeSFRW != excludeSFRQ and excludeSFRW == True:
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFRW,excludeSFRQ))
            return 20

    if parttype != '0': #EOS is only relevant for parttype 0 (gas)
        excludeSFRW = False
        excludeSFRQ = False

    # if nothing has gone wrong, return all input, since setting quantities in functions doesn't work on global variables
    return 0, iseltW, iseltQ, simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype,\
         theta, phi, psi, \
         sylviasshtables, bensgadget2tables,\
         ps20tables, ps20depletion,\
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc, misc, ompproj, numslices,\
         halosel, kwargs_halosel, hdf5, override_simdatapath



def partselect_pos(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype='0'): # this is handled by read_eagle; the hash tables are sufficient to specify the region

    '''
    Uses the read_eagle hash tables to select a region to be used on read-ins
    '''
    hconst = simfile.h
    BoxSize = simfile.boxsize
    if Ls[Axis1] >= BoxSize * hconst**-1 and Ls[Axis2] >= BoxSize * hconst**-1:
        if Ls[Axis3] >= BoxSize * hconst**-1:
            region = None
        else:
            region = np.array([0., BoxSize, 0., BoxSize, 0., BoxSize])
            region[[2*Axis3, 2*Axis3+1]] = [(centre[Axis3] - Ls[Axis3] * 0.5) * hconst, (centre[Axis3] + Ls[Axis3] * 0.5) * hconst]

    else:
        region = np.array([0., BoxSize, 0., BoxSize, 0., BoxSize])
        region[[2*Axis3, 2*Axis3+1]] = [(centre[Axis3] - Ls[Axis3] * 0.5) * hconst, (centre[Axis3] + Ls[Axis3] * 0.5) * hconst]
        lsmooth = simfile.readarray('PartType%s/SmoothingLength'%parttype, rawunits=True, region=region)
        margin = np.max(lsmooth)
        del lsmooth # read it in again later, if it's needed again
        region[[2*Axis2, 2*Axis2+1]] = [(centre[Axis2] - Ls[Axis2] * 0.5) * hconst - margin ,(centre[Axis2] + Ls[Axis2] * 0.5) * hconst + margin]
        region[[2*Axis1, 2*Axis1+1]] = [(centre[Axis1] - Ls[Axis1] * 0.5) * hconst - margin ,(centre[Axis1] + Ls[Axis1] * 0.5) * hconst + margin]
    print('Selecting region %s [cMpc/h]'%(region))
    return region


#def partselect_vel_region(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype = '0'):  # obsolete
#    '''
#    returns a region for particle read-in (as does partselect_pos),
#    and the velocity coordinates along the projection direction
#    '''
#    hconst = simfile.h
#    BoxSize = simfile.boxsize
#    z = simfile.z
#    hf = Hubble(z,simfile=simfile)
#
#    # full selection in position directions (Axis0, Axis1)
#    Ls_pos = np.copy(Ls)
#    Ls_pos[Axis3] = BoxSize
#    region = partselect_pos(simfile, centre, Ls_pos, Axis1, Axis2, Axis3, parttype=parttype)
#    print('partselect vel region : %s'%region)
#
#    # further selection: only need velocity in the projection direction
#    velp = simfile.readarray('PartType%s/Velocity'%parttype, rawunits=True,region=region)[:,Axis3]
#    vconv = simfile.a **simfile.a_scaling * (simfile.h ** simfile.h_scaling) * simfile.CGSconversion
#    maxdzvelp = np.max(np.abs(velp))*vconv/hf/(hconst**-1*simfile.a*c.unitlength_in_cm) #convert gadget velocity to cgs velocity to cgs position to gadget comoving coordinate units
#    del velp # read in again with new region in ppv_selselect_coordsgen
#
#    region[[2*Axis3, 2*Axis3+1]] = [ (centre[Axis3]-Ls[Axis3]/2.)*hconst - maxdzvelp ,(centre[Axis3]+Ls[Axis3]/2.)*hconst + maxdzvelp]
#    print('Velocity space selection spatial region: %s'%str(region))
#
#    return region


def ppv_selselect_coordsgen(centre, Ls, Axis1, Axis2, Axis3, periodic, vardict, velcut=True): # changed to only need region input
    '''
    separated into a function, but needs to be used carefully
    given position-equivalent velocity space selection,
    partselect_vel_region output,
    adds coordinates, box3, etc. in position-position-velocity space to vardict
    '''
    # vrf = H(z)*dphys + vpec,rest-frame
    # vobs = (1+z)*vrf
    userawv = isinstance(velcut, tuple)

    hconst = vardict.simfile.h
    BoxSize = vardict.simfile.boxsize
    hf = Hubble(vardict.simfile.z,simfile=vardict.simfile)

    # coords are stored in float64, but I don't need that precision here. (As long as particles end up in about the same slice, it should be ok. It's converted to float32 for projection anyway)
    vardict.readif('Coordinates', rawunits=True) # hubble is also in cgs; ~1e24 cm length coordinates shouldn't overflow
    coords = vardict.particle['Coordinates']
    conv = vardict.CGSconv['Coordinates']
    vardict.add_part('coords_cMpc-vel',coords) # position in gagdet units
    del coords
    vardict.delif('Coordinates', last=True)

    vardict.readif('Velocity', rawunits=True, region=vardict.region)
    vconv = vardict.CGSconv['Velocity']
    vardict.particle['Velocity'] = vardict.particle['Velocity'][:,Axis3] * vconv # velocity (cgs)

    boxvel = BoxSize*conv*hf
    vardict.particle['coords_cMpc-vel'][:,Axis3] *= conv*hf # position (gadget) -> position (cgs, proper) -> hubble flow (cgs)
    vardict.particle['coords_cMpc-vel'][:,Axis3] += vardict.particle['Velocity'] # positive velocity = direction of position increase: both away from observer
    vardict.delif('Velocity',last=True)
    vardict.particle['coords_cMpc-vel'][:,Axis3] %= boxvel

    # avoid fancy indexing to avoid array copies
    vardict.particle['coords_cMpc-vel'][:,Axis1] *= hconst**-1
    vardict.particle['coords_cMpc-vel'][:,Axis2] *= hconst**-1
    BoxSize = BoxSize * hconst**-1

    Ls[Axis3] = Ls[Axis3] / BoxSize * boxvel
    centre[Axis3] = centre[Axis3] / BoxSize * boxvel
    if userawv:
        Ls[Axis3] = velcut[1] * 1e5 # km/s -> cm/s
        centre[Axis3] += velcut[0] * 1e5 # km/s -> cm/s, add to hubble flow
    box3 = list((BoxSize,)*3)
    box3[Axis3] = boxvel

    translate(vardict.particle, 'coords_cMpc-vel', centre, box3, periodic)
    # This selection can miss particles at the edge due to numerical errors
    # this is shown by testing on a whole box projection with Readfileclone,
    # with a box designed to create these edges cases
    #
    # using >=  and <= runs the risk of double-counting particles when slicing
    # the box, however, so this is not an ideal solution either.
    #
    # tests on the 12.5 cMpc box showed no loss of particles on the whole box
    # selection
    if periodic:
        sel = pc.Sel({'arr': 0.5*boxvel - 0.5*Ls[Axis3] <= vardict.particle['coords_cMpc-vel'][:,Axis3]})
        sel.comb({'arr':  0.5*boxvel + 0.5*Ls[Axis3] >  vardict.particle['coords_cMpc-vel'][:,Axis3]})
    else:
        sel = pc.Sel({'arr': -0.5*Ls[Axis3] <= vardict.particle['coords_cMpc-vel'][:,Axis3]})
        sel.comb({'arr':   0.5*Ls[Axis3] >  vardict.particle['coords_cMpc-vel'][:,Axis3]})

    vardict.add_box('box3', box3)
    vardict.overwrite_box('centre',centre)
    vardict.overwrite_box('Ls',Ls)
    vardict.update(sel)


def ppp_selselect_coordsadd(vardict, centre, Ls, periodic, Axis1, Axis2, Axis3, keepcoords=True): # changed to only need region input
    '''
    separated into a function, but needs to be used carefully
    adds coordinates, box3, etc. in position space to vardict
    (depending on wishlist setting in vardict)
    periodic refers to the project and is only used in setting the target
    range in coordinate translation. The coordinates are always taken to be
    periodic
    axis: axis along which to project later: no lsmooth margin here
    '''

    BoxSize = vardict.simfile.boxsize / vardict.simfile.h # cMpc

    # coords are stored in float64, but I don't need that precision here. (As long as particles end up in about the same slice, it should be ok. It's converted to float32 for projection anyway)
    vardict.readif('Coordinates', rawunits=True, region=vardict.region)
    coords = vardict.particle['Coordinates']
    vardict.delif('Coordinates', last=True) # units get changed -> delete to avoid issues later on
    vardict.readif('SmoothingLength',rawunits=True)
    lmax= np.max(vardict.particle['SmoothingLength'])
    vardict.delif('SmoothingLength')
    hconst = vardict.simfile.h
    lmax /= hconst
    coords /= hconst # convert to Mpc from Mpc/h
    box3 = list((BoxSize,)*3)

    vardict.add_part('coords_cMpc-vel',coords)
    del coords
    translate(vardict.particle, 'coords_cMpc-vel', centre, box3, periodic)

    doselection = np.array([0,1,2])[np.array(Ls) < BoxSize]
    # no lsmooth margin in projection direction
    # centering depends on periodicity
    sel = pc.Sel()
    for axind in doselection:
        if axind == Axis1 or axind == Axis2:
            margin = 0.5*Ls[axind] + lmax
        elif axind == Axis3:
            margin = 0.5*Ls[axind]
        else:
            print('Error in particle selection by position: axis %i does not match any of Axis1 (%i), Axis2 (%i) or Axis3 (%i)'%(axind, Axis1, Axis2, Axis3))
            return None

        if periodic:
            sel.comb({'arr': 0.5*BoxSize - margin <= vardict.particle['coords_cMpc-vel'][:,axind]})
            sel.comb({'arr': 0.5*BoxSize + margin >  vardict.particle['coords_cMpc-vel'][:,axind]})
        else:
            sel.comb({'arr': -1*margin <= vardict.particle['coords_cMpc-vel'][:,axind]})
            sel.comb({'arr':    margin >  vardict.particle['coords_cMpc-vel'][:,axind]})

    vardict.add_box('box3',box3)
    vardict.add_box('centre',centre)
    vardict.add_box('Ls',Ls)
    vardict.delif('coords_cMpc-vel',last=not keepcoords)
    vardict.update(sel)


##### Does not work because online catalogue group ids do not match particle
##### data group numbers
#def selecthaloparticles(vardict, halosel, aperture=30,\
#                        exclsatellites=False, allinR200=True,\
#                        nameonly=False, last=True):
#    '''
#    select only the particle in halos satifying the criteria in halosel
#    assumes online halo catalogues Group IDs match GroupNumbers in particle 
#    data  
#    
#    input:
#    ------------
#    vardict:    Vardict instance to apply the selection to
#    halosel:    halo selection criteria as used in selecthalos.py: selecthalos
#                list of (name, min/None, max/None [, margin]) tuples
#                margins are in units of R200c (default if None is 1.), and only
#                apply to 'X', 'Y', or 'Z' selections (on halo position)
#                tuples with the same name give different allowed ranges (OR)
#                then criteria for the sets of tuples with different names must
#                all be satified (and)
#                names must match halo catalogue entries
#    aperture:  aperture in which to get e.g. stellar masses 
#                (default: 30 [pkpc])
#    exclsatellites: halo list applies to centrals/FOF main halos only
#                exclsatellites determines whether SubGroupNumber !=0 gas 
#                (belonging to subhalos) is explicitly excluded (True) or 
#    allinR200:  include particles inside R200c but not in the FOF group
#    
#    nameonly:   just check input and output the name
#    
#    output:
#    ------------
#    if nameonly: string describing the selection
#    else:        array of galaxy ids of the selected halos/galaxies
#    modifies the selection in vardict to include only particles meeting the 
#    halo selection
#    '''
#    #### check input
#    simnum = vardict.simfile.simnum
#    snap   = vardict.simfile.snapnum
#    var    = vardict.simfile.var
#    simulation = vardict.simfile.simulation
#    
#    if vardict.simfile.filetype != 'particles':
#        raise ValueError('selecthaloparticles only works with particle files, not input filetype %s'%vardict.simfile.filetype)
#    if simulation != 'eagle':
#        raise ValueError('selecthaloparticles only works for eagle simulations, not input simulation %s'%simulation)
#    
#    if nameonly:
#        if len(halosel) > 0:
#            selind_incl = sorted(halosel)
#            selind_incl = '_halo-selection_' + '_'.join(['-'.join(str(val) for val in tup) for tup in selind_incl])
#        else:
#            selind_incl = '_halos'
#        
#        if exclsatellites:
#            satind = '_exclsatellites'
#        else:
#            satind = ''
#        
#        if allinR200:
#            hind = '_R200c'
#        else:
#            hind = '_FOF'
#        
#        apind = '_aperture-%s-pkpc'%aperture
#        
#        return selind_incl + hind + apind + satind
#    
#    #### check if halo catalogue exists (with Mhmin = 0.), otherwise, generate it
#    varind = hc.getvar(var)
#    halocatname = ol.pdir + 'catalogue_%s%s_snap%i_aperture%i.hdf5'%(varind, simnum, snap, aperture)
#
#    runselection = True
#    if os.path.isfile(halocatname):
#        with h5py.File(halocatname, 'a') as hcf:
#            Mhmin = hcf['Header'].attrs['Mhalo_min_Msun']
#            if Mhmin <= 0.:
#                runselection = False
#                
#    if runselection:            
#        hc.generatehdf5_centrals(simnum, snap, Mhmin=0., var=var, apsize=aperture)
#    
#    with h5py.File(halocatname, 'r') as hfc:
#        sel = sh.selecthalos(hfc, halosel) # boolean array with selection to apply to arrays in halo catalogue file
#        galaxyids = np.array(hfc['galaxyid'])[sel]
#        groupids  = np.array(hfc['groupid'])[sel]
#        #allids = np.array(hfc['groupid'])
#    
#    vardict.readif('GroupNumber', rawunits=True)
#    if allinR200:
#        vardict.particle['GroupNumber'] = np.abs(vardict.particle['GroupNumber']) # GroupNumber < 0: within R200 of halo |GroupNumber|, but not in FOF group
#    if exclsatellites:
#        groupmax = np.max(vardict.particle['GroupNumber']) + 1
#        vardict.readif('SubGroupNumber', rawunits=True)
#        vardict.particle['GroupNumber'] -= groupmax * vardict.particle['SubGroupNumber'] # SubgroupNumber > 0 -> GroupNumber becomes < 0 and particle will not be selected
#        vardict.delif('SubGroupNumber', last=last)    
#    
#    #anymatch = np.any([gid in allids for gid in vardict.particle['GroupNumber']])
#    # print('Any particle IDs match any catlogue ids: %s'%anymatch) False in initial bug test
#    #print(vardict.particle['GroupNumber'])
#    groupmatches = match(vardict.particle['GroupNumber'], groupids, arr2_sorted=False, arr2_index=None)
#    vardict.delif('GroupNumber', last=True) # possily modified, so delete
#    
#    sel = pc.Sel({'arr': groupmatches >= 0})
#    vardict.update(sel)
#    return galaxyids

'''
halosel:   only include particles belonging to FOF halos meeting these 
               selection criteria. See documentation of selecthaloparticles for
               format, or sh.selecthalos_subfindfiles (sh = selecthalos.py) 
               for more details
               default: None -> no selection on halo membership
    kwargs_halosel: kwargs for selecthalos 
                aperture: aperture in which to get e.g. stellar masses 
                  (default: 30 [pkpc])
                mdef: halo mass definition to use (default: '200c')
                exclsatellites: halo list applies to centrals/FOF main halos 
                  only; exclsatellites determines whether SubGroupNumber !=0 
                  gas (belonging to subhalos) is explicitly excluded (True) 
                  or not 
                allinR200: include particles inside R200c but not in the FOF 
                  group
'''
    
def selecthaloparticles(vardict, halosel, nameonly=False, last=True, **kwargs):
    '''
    vardict:   instance for particle data, used to generate simfile for subfind
               files & selection is applied to this
    halosel:   criteria by which to select halos (see documentation for 
               selecthalos.py -> selecthalos_subfindfiles, or from here:
               sh.selecthalos_subfindfiles)
    nameonly:  just run the input checks and return the name of the selection
    last:      delete any arrays read in for this function (applies to the 
               particle data; FOF/Subfind data is assumed not to be of further
               use and deleted)
    
    used kwargs: (any others are just ignored)
    aperture:  aperture in which to measure subhalo properties for the 
               selection (default 30 [pkpc])
    mdef:      halo mass definition for the selection (default '200c')
    exlcsatellites: only use particles with SubGroupNumber 0 (default False)
    allinR200c: include unbound halo particles within R200c (GroupNumber < 0)
               (default True)
               
    returns:
        list of groupnumbers used
    modifies the particle selection in vardict 
    '''
    
    if vardict.simfile.filetype != 'particles':
        raise ValueError('selecthaloparticles only works with particle files, not input filetype %s'%vardict.simfile.filetype)
    if vardict.simfile.simulation != 'eagle':
        raise ValueError('selecthaloparticles only works for eagle simulations, not input simulation %s'%vardict.simfile.simulation)
    # validity of mdef, aperture, halosel are checked in selecthalos.py (sh)
    if 'mdef' in kwargs.keys():
        mdef = kwargs['mdef']
    else:
        mdef = '200c'
    if 'aperture' in kwargs.keys():
        aperture = kwargs['aperture']
    else:
        aperture = 30
    if 'exclsatellites' in kwargs.keys():
        exclsatellites = kwargs['exclsatellites']
    else:
        exclsatellites = False 
    if 'allinR200c' in kwargs.keys():
        allinR200c = kwargs['allinR200c']
    else:
        allinR200c = True
    kwargs_allowed = {'mdef', 'aperture', 'exclsatellites', 'allinR200c', 'label'}
    if not set(kwargs.keys()).issubset(kwargs_allowed):
        print('Warning: kwargs %s in selecthalosparticles are ignored'%(set(kwargs.keys()) - kwargs_allowed))
        
    simfile_sf = pc.Simfile(vardict.simfile.simnum, vardict.simfile.snapnum, vardict.simfile.var, file_type='sub', simulation=vardict.simfile.simulation)
    groupnums = sh.selecthalos_subfindfiles(simfile_sf, halosel, mdef=mdef, aperture=aperture, nameonly=nameonly)
    if nameonly:
        #split off '_endhalosel' 
        groupnums = groupnums.split('_')
        name1 = '_'.join(groupnums[:-1])
        name2 = '_' + groupnums[-1]
        if exclsatellites:
            satind = '_exclsatellites'
        else:
            satind = ''
        
        if allinR200c:
            hind = '_allinR200c'
        else:
            hind = '_FOF'
        return name1 + hind + satind + name2
    ## debug:
    try:
        vardict.readif('GroupNumber', rawunits=True)
    except IOError as err:
        print('The mysterious error has arisen; status data:')
        print('trying to read data from: %s'%(vardict.simfile.readfile))
        print('vardict status: read in %s'%(', '.join([key for key in vardict.particle])))
        if len(vardict.particle) == 0:
            # try if it's a general file problem:
            vardict.readif('Mass', rawunits=True)
        raise err
        
    if allinR200c:
        vardict.particle['GroupNumber'] = np.abs(vardict.particle['GroupNumber']) # GroupNumber < 0: within R200 of halo |GroupNumber|, but not in FOF group
    if exclsatellites:
        # subgroupnumber -> True/False array: unbound gas is too close to overflowing int32 type, subtracting max * subgroupnumber may be risky
        vardict.readif('SubGroupNumber', rawunits=True)
        vardict.particle['GroupNumber'][vardict.particle['SubGroupNumber'] > 0] = -1
        vardict.delif('SubGroupNumber', last=last)    
    
    groupnums.sort()
    groupmatches = cu.match(vardict.particle['GroupNumber'], groupnums, arr2_sorted=True, arr2_index=None)
    vardict.delif('GroupNumber', last=True) # possily modified, so delete

    sel = pc.Sel({'arr': groupmatches >= 0})
    vardict.update(sel)
    
    return groupnums


##### small helper functions for the main projection routine

def get_eltab_names(abunds,iselt,ion): #assumes
    if abunds[0] == 'Sm':
        if not iselt:
            eltab = 'SmoothedElementAbundance/%s' %string.capwords(ol.elements_ion[ion])
        else:
            eltab = 'SmoothedElementAbundance/%s' %string.capwords(ion)
    elif abunds[0] =='Pt': # auto already set in inputcheck
        if not iselt:
            eltab = 'ElementAbundance/%s' %string.capwords(ol.elements_ion[ion])
        else:
            eltab = 'ElementAbundance/%s' %string.capwords(ion)
    else:
        eltab = abunds[0] #float

    if abunds[1] == 'Sm':
        hab = 'SmoothedElementAbundance/Hydrogen'
    elif abunds[1] =='Pt': # auto already set in inputcheck
        hab = 'ElementAbundance/Hydrogen'
    else:
        hab = abunds[1] #float

    return eltab, hab





#################################
# main functions, using classes #
#################################


def luminosity_calc(vardict, excludeSFR, eltab, hab, ion,\
                    last=True, updatesel=True, ps20tables=False, 
                    ps20depletion=True):
    '''
    Calculate the per particle luminosity of an emission line (ion)
    vardict should already contain the particle selection for which to
    calculate the luminosities
    last and updatesel defaults set for single use

    At this stage, it only matters if excludeSFR is 'T4' or something else;
    EOS pre-selection has already been done.
    eltab and hab can either be a (PartType#/ excluded) hdf5 path or an element
    mass fraction (float).

    outputs SPH particle line luminosities in erg/s *1e10,
    and the 1e10 conversion factor back to CGS (prevents risk of float32
    overflow)
    '''
    print('Calculating particle luminosities...')

    if isinstance(eltab, str):
        vardict.readif(eltab, rawunits=True)
        if updatesel and (ol.elements_ion[ion] not in ['hydrogen', 'helium']):
            vardict.update(vardict.particle[eltab] > 0.)

    if not vardict.isstored_part('propvol'):
        vardict.readif('Density', rawunits=True)
        vardict.readif('Mass', rawunits=True)
        vardict.add_part('propvol', (vardict.particle['Mass'] /\
                                     vardict.particle['Density']) *
                         (vardict.CGSconv['Mass'] / vardict.CGSconv['Density']))
        vardict.delif('Mass',last=last)
    
    if len(vardict.particle['propvol']) > 0:
        print('Min, max, median of particle volume [cgs]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['propvol']), np.max(vardict.particle['propvol']), np.median(vardict.particle['propvol'])))
    else:
        print('No particles in current selection')

    if not vardict.isstored_part('lognH'):
        vardict.readif('Density', rawunits=True)
        if isinstance(hab, str):
            vardict.readif(hab, rawunits=True)
            vardict.add_part('lognH', np.log10(vardict.particle[hab]) +\
                                      np.log10(vardict.particle['Density']) +\
                                      np.log10(vardict.CGSconv['Density'] / (c.atomw_H * c.u)) )
        else:
            vardict.add_part('lognH', np.log10(vardict.particle['Density']) +\
                                      np.log10(vardict.CGSconv['Density'] * hab / (c.atomw_H * c.u)) )
        vardict.delif('Density',last=last)

    if len(vardict.particle['lognH']) > 0:
        print('Min, max, median of particle log10 nH [cgs]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['lognH']),\
               np.max(vardict.particle['lognH']),\
               np.median(vardict.particle['lognH'])) )

    if not vardict.isstored_part('logT'):
        if excludeSFR == 'T4':
            vardict.readif('OnEquationOfState', rawunits=True)
            vardict.add_part('eos', vardict.particle['OnEquationOfState'] > 0.)
            vardict.delif('OnEquationOfState', last=last)
            vardict.readif('Temperature', rawunits=True,\
                           setsel=vardict.particle['eos'], setval = 1e4)
            vardict.delif('eos', last=last)
        else:
            vardict.readif('Temperature', rawunits=True)
        vardict.add_part('logT', np.log10(vardict.particle['Temperature']))
        vardict.delif('Temperature', last=last)
    
    if len(vardict.particle['logT']) > 0:
        print('Min, max, median of particle log temperature [K]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['logT']),\
               np.max(vardict.particle['logT']),\
               np.median(vardict.particle['logT'])))

    if ps20tables:
        table = linetable_PS20(ion, vardict.simfile.z, emission=True)
        if 'logZ' not in vardict.particle.keys():
            if isinstance(eltab, str):
                if 'SmoothedElementAbundance' in eltab:
                    vardict.readif('SmoothedMetallicity', rawunits=True) # dimensionless
                    vardict.add_part('logZ', np.log10(vardict.particle['SmoothedMetallicity']))
                    vardict.delif('SmoothedMetallicity', last=last)
                elif 'ElementAbundance' in eltab:
                    vardict.readif('Metallicity', rawunits=True) # dimensionless
                    vardict.add_part('logZ', np.log10(vardict.particle['Metallicity']))
                    vardict.delif('Metallicity', last=last)
            else:
                _logZ = np.ones(len(vardict.particle['lognH']))
                _logZ *= np.log10(table.solarZ)
                vardict.add_part('logZ', _logZ)
                del _logZ
        # zero metallicity leads to interpolation errors; 
        # if hydrogen or helium, set minimum to > -np.inf 
        # any value < -50. -> minimum table value used
        minlz = np.min(vardict.particle['logZ'])
        print('Minimum logZ found: {}'.format(minlz))
        dellz = False
        if minlz == -np.inf:
            print('Adjusting minimum logZ to small finite value')
            vardict.particle['logZ'][vardict.particle['logZ'] == -np.inf] = -100.
            dellz = True        
        luminosity = table.find_logemission(vardict.particle)        
        vardict.delif('logT', last=last) 
        vardict.delif('lognH', last=last)
                
        if not ps20depletion:
            # luminosity in table = (1 - depletion) * luminosity_if_all_elt_in_gas
            # so divide by depleted fraction to get undepleted emission
            luminosity -= np.log10(1. - table.find_depletion(vardict.particle))
        # rescale to the correct /element/ abundance
        if ol.elements_ion[ion] == 'hydrogen': # no rescaling if hydrogen
            vardict.delif('logZ', last=(last or dellz)) 
        else:
            luminosity -= table.find_assumedabundance(vardict.particle,
                                                      log=True)
            vardict.delif('logZ', last=(last or dellz))
            if isinstance(hab, str):
                vardict.readif(hab, rawunits=True)
                hmfrac = vardict.particle[hab]
                if eltab != hab:
                    vardict.delif(hab, last=last)
            else:
                hmfrac = hab
            if isinstance(eltab, str):
                vardict.readif(eltab, rawunits=True)
                emfrac = vardict.particle[eltab]
                vardict.delif(eltab, last=last)
            else:
                emfrac = eltab
            zscale = emfrac / hmfrac
            del emfrac
            del hmfrac
            zscale *= ionh.atomw['Hydrogen'] / table.elementmass_u
            luminosity += np.log10(zscale)
        luminosity += np.log10(vardict.particle['propvol'])
        vardict.delif('propvol',last=last)
        if len(luminosity) > 0:
            maxL = np.max(luminosity)
            # attempt to prevent overflow in exponentiation + projection
            rescale = 34. - maxL
        else:
            rescale = 0.
        luminosity = 10.0**(luminosity - rescale)
        CGSconv = 10**rescale
    
    else: # not ps20tables
        lineind = ol.line_nos_ion[ion]
        vardict.add_part('emdenssq', find_emdenssq(vardict.simfile.z,\
                                                   ol.elements_ion[ion],\
                                                   vardict.particle,\
                                                   lineind))
        if len(vardict.particle['emdenssq']) > 0:
            print('Min, max, median of particle emdenssq: %.5e %.5e %.5e' \
                % (np.min(vardict.particle['emdenssq']),\
                   np.max(vardict.particle['emdenssq']),\
                   np.median(vardict.particle['emdenssq'])))
        vardict.delif('logT', last=last)
    
        # for agreement with Cosmoplotter
        # also: using SPH_KERNEL_GADGET; check what EAGLE uses!!
        #lowZ = eltabund < 10**-15
        #eltabund[lowZ] = 0.
        if ol.elements_ion[ion] == 'hydrogen': # no rescaling if hydrogen
            zscale = 1.
        else:
            if isinstance(hab, str):
                vardict.readif(hab, rawunits=True)
                hmfrac = vardict.particle[hab]
                if eltab != hab:
                    vardict.delif(hab, last=last)
            else:
                hmfrac = hab
            if isinstance(eltab, str):
                vardict.readif(eltab, rawunits=True)
                emfrac = vardict.particle[eltab]
                vardict.delif(eltab, last=last)
            else:
                emfrac = eltab
            zscale = emfrac / hmfrac
            del emfrac
            del hmfrac
            zscale *= ionh.atomw['Hydrogen'] / \
                      ionh.atomw[string.capwords(ol.elements_ion[ion])]
            zscale /= ol.solar_abunds_sb[ol.elements_ion[ion]]
        # using units of 10**-10 * CGS, to make sure overflow of float32 does not occur in C
        # (max is within 2-3 factors of 10 of float32 overflow in one simulation)
        luminosity = zscale * 10**(vardict.particle['emdenssq'] +\
                                   2. * vardict.particle['lognH'] +\
                                   np.log10(vardict.particle['propvol']) - 10.)
        CGSconv = 1e10
        vardict.delif('lognH',last=last)
        vardict.delif('emdenssq',last=last)
        vardict.delif('propvol',last=last)

    if len(luminosity) > 0:
        print('Min, max, median of particle luminosity [1e10 cgs]: %.5e %.5e %.5e' \
            % (np.min(luminosity), np.max(luminosity), np.median(luminosity)))
    print('  done.\n')
    return luminosity, CGSconv # array, cgsconversion


def luminosity_to_Sb(vardict, Ls, Axis1, Axis2, Axis3, npix_x, npix_y, ion,
                     ps20tables=False):
    '''
    converts cgs luminosity (erg/s) to cgs surface brightness
    (photons/s/cm2/steradian)
    ion needed because conversion depends on the line energy
    
    Parameters
    ----------
    vardict: Vardict instance 
        used to get cosmological parameters
    Ls: array of floats            
        the dimensions of the projected box: Ls[0] is the full extent along 
        the x axis, Ls[1] along y, Ls[2] is along z (diameter, not radius)
    Axis1, Axis2: int  
        axes perpendicular to the line of sight (0=x, 1=y, 2=z
    Axis3: int 
        axis along the line of sight (int)
    npix_x, npix_y: int
        number of pixels along Axis1 and Axis2, respectively 
    ion: str           
        name for the line to get the conversion for, as used in 
        make_maps_opts_locs
    ps20tables: bool    
        if True, the ion refers to a PS20 table line (get energy from the 
        line name, not make_maps_opnts_locs)
    In Ls, the the indices match the simulation axes every time: indices 0, 1, 
    and 2 always correspond to the X, Y, and Z axes respectively.
    Axis1, Axis2, and Axis3 and as used as indices for Ls. Axis1 and Axis2, 
    are used with Ls and npix_<x/y> to determine the dimensions of each pixel, 
    and Axis3 is used to impose a minimum comoving distance of half the extent
    along the line of sight.

    returns:
    --------
    a number (float) which can be multiplied by the emission line luminosity 
    in a pixel in erg/s to get the surface brightness in 
    photons/cm^2/s/steradian            
        '''
    zcalc = vardict.simfile.z
    comdist = comoving_distance_cm(zcalc, simfile=vardict.simfile)
    longlen = max(Ls) * 0.5 * c.cm_per_mpc
    if comdist > longlen: # even at larger values, the projection along z-axis = projection along sightline approximation will break down
        ldist = comdist * (1. + zcalc) # luminosity distance
        adist = comdist / (1. + zcalc) # angular size distance
    else:
        ldist = longlen * (1. + zcalc)
        adist = longlen / (1. + zcalc)

    # conversion (x, y are axis placeholders and may actually represent different axes in the simulation, as with numpix_x, and numpix_y)
    halfangle_x = 0.5 * Ls[Axis1] / (1. + zcalc) / npix_x * c.cm_per_mpc / adist
    halfangle_y = 0.5 * Ls[Axis2] / (1. + zcalc) / npix_y * c.cm_per_mpc / adist

    #solidangle = 2*np.pi*(1-np.cos(2.*halfangle_x))
    #print("solid angle per pixel: %f" %solidangle)
    # the (1+z) is not so much a correction to the line energy as to the luminosity distance:
    # the 1/4 pi dL^2 luminosity -> flux conversion is for a broad spectrum and includes energy flux decrease to to redshifting
    # multiplying by (1+z) compensates for this: the number of photons does not change from redshifting
    if ps20tables:
        # units: Å (A), cm (c), and μm (m)
        _wl = (ion[4:]).strip()
        _unit = _wl[-1]
        wl = float(_wl[:-1])
        unit = 1. if _unit == 'c' else 1e-8 if _unit == 'A' \
              else 1e-4 if _unit == 'm' else np.NaN
        wl_cm = wl * unit
        eng_erg = c.planck * c.c / wl_cm
    else:
        eng_erg = ol.line_eng_ion[ion]
    return 1. / (4 * np.pi * ldist**2) * (1. + zcalc) / eng_erg *\
           1. / solidangle(halfangle_x, halfangle_y)



def Nion_calc(vardict, excludeSFR, eltab, hab, ion, sylviasshtables=False,
              last=True, updatesel=True, misc=None, bensgadget2tables=False,
              ps20tables=False, ps20depletion=True):
    '''
    When using sylviasshtables (deprecated) or ps20tables, 
    smoothed/particle/fixed metallicities match the choice for element 
    abundance. If the element abundance is a float value, solar metallicity is
    assumed for ps20tables, and eltab / solar eltab * solar Z is used for 
    syvliasshtables. (This is modified for ps20tables because it doesn't work
    for hydrogen and helium, and metallicity dependences are small anyway.)
    '''
    ionbal_from_outputs = False
    if misc is not None:
        if 'usechemabundtables' in misc:
            if misc['usechemabundtables'] == 'BenOpp1':
                ionbal_from_outputs = True

    if isinstance(eltab, str):
        vardict.readif(eltab, rawunits=True)
        if updatesel and (ol.elements_ion[ion] not in ['hydrogen', 'helium']):
            vardict.update(vardict.particle[eltab] > 0.)

    if not ionbal_from_outputs: # if not misc option for getting ionfrac from Ben Oppenheimer's modified RECAL-L0025N0752 runs with non-equilibrium ion fractions
        if not vardict.isstored_part('lognH'):
            vardict.readif('Density', rawunits=True)
            if isinstance(hab, str):
                vardict.readif(hab,rawunits = True)
                vardict.add_part('lognH', np.log10(vardict.particle[hab]) + np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density'] / (c.atomw_H * c.u)) )
                if eltab != hab:
                    vardict.delif(hab, last=last)
            else:
                vardict.add_part('lognH', np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density'] * hab / (c.atomw_H * c.u)) )
            vardict.delif('Density', last=last)
        
        if len(vardict.particle['lognH']) > 0:
            print('Min, max, median of particle log10 nH [cgs]: %.5e %.5e %.5e' \
                % (np.min(vardict.particle['lognH']), np.max(vardict.particle['lognH']), np.median(vardict.particle['lognH'])) )
        else:
            print('No particles in current selection')

        if not vardict.isstored_part('logT'):
            if excludeSFR == 'T4':
                vardict.readif('OnEquationOfState', rawunits=True)
                vardict.add_part('eos', vardict.particle['OnEquationOfState'] > 0.)
                vardict.delif('OnEquationOfState', last=last)
                vardict.readif('Temperature', rawunits=True,\
                               setsel=vardict.particle['eos'], setval = 1e4)
                vardict.delif('eos',last=last)
            else:
                vardict.readif('Temperature', rawunits=True)
            vardict.add_part('logT', np.log10(vardict.particle['Temperature']))
            vardict.delif('Temperature',last=last)
        if len(vardict.particle['logT']) > 0:
            print('Min, max, median of particle log temperature [K]: %.5e %.5e %.5e' \
                % (np.min(vardict.particle['logT']), np.max(vardict.particle['logT']), np.median(vardict.particle['logT'])))
        if sylviasshtables or ps20tables:
            if ps20tables:
                table = linetable_PS20(ion, vardict.simfile.z)
                print('Using table for z={.3f}'.format(vardict.simfile.z))
            if 'logZ' not in vardict.particle.keys():
                if isinstance(eltab, str):
                    if 'SmoothedElementAbundance' in eltab:
                        vardict.readif('SmoothedMetallicity', rawunits=True) # dimensionless
                        vardict.add_part('logZ', np.log10(vardict.particle['SmoothedMetallicity']))
                        vardict.delif('SmoothedMetallicity', last=last)
                    elif 'ElementAbundance' in eltab:
                        vardict.readif('Metallicity', rawunits=True) # dimensionless
                        vardict.add_part('logZ', np.log10(vardict.particle['Metallicity']))
                        vardict.delif('Metallicity', last=last)
                elif sylviasshtables:
                    vardict.add_part('logZ', np.ones(len(vardict.particle['lognH'])) *\
                                     np.log10(eltab / ol.solar_abunds_ea[ol.elements_ion[ion]] *\
                                              ol.Zsun_sylviastables))
                else:
                    _logZ = np.ones(len(vardict.particle['lognH']))
                    _logZ *= np.log10(table.solarZ)
                    vardict.add_part('logZ', _logZ)
                    del _logZ
                    
                    # invert logZ - element abundance relation
                    
                # zero metallicity leads to interpolation errors; 
                # if hydrogen or helium, set minimum to > -np.inf 
                # any value < -50. -> minimum table value used
                minlz = np.min(vardict.particle['logZ'])
                print('Minimum logZ found: {}'.format(minlz))
                dellz = False
                if minlz == -np.inf:
                    print('Adjusting minimum logZ to small finite value')
                    vardict.particle['logZ'][vardict.particle['logZ'] == -np.inf] = -100.
                    dellz = True
            if sylviasshtables:
                vardict.add_part('ionfrac', find_ionbal_sylviassh(vardict.simfile.z,
                                                                  ion, vardict.particle))
            elif ps20tables:
                _ionfrac = table.find_ionbal(vardict.particle, log=False)
                if ps20depletion:
                    _ionfrac *= (1. - table.find_depletion(vardict.particle))
                vardict.add_part('ionfrac', _ionfrac)
                del _ionfrac
                
            if not isinstance(eltab, str):
                vardict.delif('logZ', last=True)
            else:
                vardict.delif('logZ', last=dellz)
        elif bensgadget2tables: # same data as the default tables from Serena Bertone
            vardict.add_part('ionfrac', find_ionbal_bensgadget2(vardict.simfile.z, 
                                                                ion, vardict.particle))
        else:
            vardict.add_part('ionfrac', find_ionbal(vardict.simfile.z, ion, 
                                                    vardict.particle))
        vardict.delif('lognH', last=last)
        vardict.delif('logT', last=last)

    # get ion balance; misc option for en Oppenheimer's ion balamce tables in L0025N0752 modified recal
    else:
        # gets ionfrac entry
        getBenOpp1chemabundtables(vardict,excludeSFR,eltab,hab,ion,last=last,updatesel=updatesel,misc=misc)

    vardict.readif('Mass', rawunits=True)

    if isinstance(eltab, str):
        Nion = vardict.particle[eltab] * vardict.particle['ionfrac'] *\
            vardict.particle['Mass']
        vardict.delif('ionfrac',last=last)
        vardict.delif('Mass',last=last)
        vardict.delif(eltab,last=last)
    else:
        Nion = eltab * vardict.particle['ionfrac'] * vardict.particle['Mass']
        vardict.delif('ionfrac',last=last)
        vardict.delif('Mass',last=last)
    
    if ps20tables:
        ionmass = table.elementmass_u
    else:
        ionmass = ionh.atomw[string.capwords(ol.elements_ion[ion])]
    if ion == 'hmolssh':
        ionmass *= 2
    to_cgs_numdens = vardict.CGSconv['Mass'] / (ionmass * c.u)

    return Nion, to_cgs_numdens # array, cgsconversion


def Nelt_calc(vardict,excludeSFR,eltab,hab,ion,last=True,updatesel=True,
              ps20tables=False, ps20depletion=True):
    '''
    ps20tables and ps20depletion mean the element column is calculated with
    the dust depletion subtracted. ps20tables is ignored if ps20tables is 
    ignored ps20depletion is False. hab is also only used for depletion 
    calculations.
    '''
    

    if isinstance(eltab, str):
        vardict.readif(eltab, rawunits=True)
        if updatesel and (ion not in ['hydrogen', 'helium']):
            vardict.update(vardict.particle[eltab] > 0.)

    vardict.readif('Mass', rawunits=True)

    if isinstance(eltab, str):
        Nelt = vardict.particle[eltab]*vardict.particle['Mass']
        vardict.delif('Mass',last=last)
        vardict.delif(eltab,last=last)
    else:
        Nelt = eltab*vardict.particle['Mass']
        vardict.delif('Mass',last=last)
    
    if ps20tables and ps20depletion:
        if not vardict.isstored_part('lognH'):
            vardict.readif('Density', rawunits=True)
            if isinstance(hab, str):
                vardict.readif(hab,rawunits = True)
                vardict.add_part('lognH', np.log10(vardict.particle[hab]) + np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density'] / (c.atomw_H * c.u)) )
                if eltab != hab:
                    vardict.delif(hab, last=last)
            else:
                vardict.add_part('lognH', np.log10(vardict.particle['Density']) + np.log10( vardict.CGSconv['Density'] * hab / (c.atomw_H * c.u)) )
            vardict.delif('Density', last=last)
        
        if len(vardict.particle['lognH']) > 0:
            print('Min, max, median of particle log10 nH [cgs]: %.5e %.5e %.5e' \
                % (np.min(vardict.particle['lognH']), np.max(vardict.particle['lognH']), np.median(vardict.particle['lognH'])) )
        else:
            print('No particles in current selection')

        if not vardict.isstored_part('logT'):
            if excludeSFR == 'T4':
                vardict.readif('OnEquationOfState', rawunits=True)
                vardict.add_part('eos', vardict.particle['OnEquationOfState'] > 0.)
                vardict.delif('OnEquationOfState', last=last)
                vardict.readif('Temperature', rawunits=True,\
                               setsel=vardict.particle['eos'], setval = 1e4)
                vardict.delif('eos',last=last)
            else:
                vardict.readif('Temperature', rawunits=True)
            vardict.add_part('logT', np.log10(vardict.particle['Temperature']))
            vardict.delif('Temperature',last=last)
        if len(vardict.particle['logT']) > 0:
            print('Min, max, median of particle log temperature [K]: %.5e %.5e %.5e' \
                % (np.min(vardict.particle['logT']), np.max(vardict.particle['logT']), np.median(vardict.particle['logT'])))
        
        # linetable object needs an actual ion
        dummyion = ild.element_to_abbr[ion]
        dummyion = dummyion.lower() + '1'
        table = linetable_PS20(dummyion, vardict.simfile.z)
        print('Using table for z={.3f}'.format(vardict.simfile.z))
        if 'logZ' not in vardict.particle.keys():
            if isinstance(eltab, str):
                if 'SmoothedElementAbundance' in eltab:
                    vardict.readif('SmoothedMetallicity', rawunits=True) # dimensionless
                    vardict.add_part('logZ', np.log10(vardict.particle['SmoothedMetallicity']))
                    vardict.delif('SmoothedMetallicity', last=last)
                elif 'ElementAbundance' in eltab:
                    vardict.readif('Metallicity', rawunits=True) # dimensionless
                    vardict.add_part('logZ', np.log10(vardict.particle['Metallicity']))
                    vardict.delif('Metallicity', last=last)
            else:
                _logZ = np.ones(len(vardict.particle['lognH']))
                _logZ *= np.log10(table.solarZ)
                vardict.add_part('logZ', _logZ)
                del _logZ
        Nelt *= (1. - table.find_depletion(vardict.particle))
        vardict.delif('lognH', last=last)
        vardict.delif('logT', last=last)
        vardict.delif('logZ', last=last)
        ionmass = table.elementmass_u
    else:
        ionmass = ionh.atomw[string.capwords(ion)]
    to_cgs_numdens = vardict.CGSconv['Mass'] / (ionmass * c.u)
    return Nelt, to_cgs_numdens


def Nion_calc_ssh(vardict, excludeSFR, hab, ion, last=True, updatesel=True, misc=None):
    '''
    Rahmati et al 2013 HI and H2 (molecular)
    best to use with face-value EOS temperatures, I think; LSR comes from a 
    pressure scaling
    '''
    if misc is None:
        useLSR = False
        UVB = 'HM01'
    else:
        if 'useLSR' in misc.keys():
            useLSR = misc['useLSR']
        else:
            useLSR = False
        if 'UVB' in misc.keys():
            UVB = misc['UVB']
        else:
            UVB = 'HM01'

    if isinstance(hab, str):
        vardict.readif(hab, rawunits=True)

    if not vardict.isstored_part('nH'):
        vardict.readif('Density', rawunits=True)
        #print('Number of particles in use: %s'%str(np.sum(vardict.readsel.val)))
        if isinstance(hab, str):
            vardict.readif(hab,rawunits = True)
            vardict.add_part('nH', vardict.particle[hab] * vardict.particle['Density'] * vardict.CGSconv['Density'] / (c.atomw_H * c.u) )
            #print('Min, max, median of particle Density: %.5e %.5e %.5e' \
            #    % (np.min(vardict.particle['Density']), np.max(vardict.particle['Density']), np.median(vardict.particle['Density'])))
        else:
            vardict.add_part('nH', vardict.particle['Density'] * vardict.CGSconv['Density'] * (hab / (c.atomw_H * c.u)) )
        vardict.delif('Density',last=last)
    if len(vardict.particle['nH']) > 0:
        print('Min, max, median of particle nH [cgs]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['nH']), np.max(vardict.particle['nH']), np.median(vardict.particle['nH'])) )
    else:
        print('No particles in current selection')

    if not vardict.isstored_part('Temperature'):
        if excludeSFR == 'T4':
            vardict.readif('OnEquationOfState', rawunits=True)
            vardict.add_part('eos', vardict.particle['OnEquationOfState'] > 0.)
            vardict.delif('OnEquationOfState',last=last)
            vardict.readif('Temperature', rawunits=True, setsel=vardict.particle['eos'],setval = 1e4)
            vardict.delif('eos', last=last)
        else:
            vardict.readif('Temperature', rawunits=True)
    if len(vardict.particle['Temperature']) > 0:
        print('Min, max, median of particle temperature [K]: %.5e %.5e %.5e' \
            % (np.min(vardict.particle['Temperature']), np.max(vardict.particle['Temperature']), np.median(vardict.particle['Temperature'])))

    h1hmolfrac = cfh.nHIHmol_over_nH(vardict.particle, vardict.simfile.z, UVB=UVB, useLSR=useLSR)
    if ion == 'h1ssh':
        if not vardict.isstored_part('eos'):
            vardict.readif('StarFormationRate')
            vardict.add_part('eos', vardict.particle['StarFormationRate'] > 0.)
            vardict.delif('StarFormationRate')
        h1hmolfrac *= (1. - cfh.rhoHmol_over_rhoH(vardict.particle))
        vardict.delif('eos')
    elif ion == 'hmolssh':
        if not vardict.isstored_part('eos'):
            vardict.readif('StarFormationRate')
            vardict.add_part('eos', vardict.particle['StarFormationRate'] > 0.)
            vardict.delif('StarFormationRate')
        h1hmolfrac *= cfh.rhoHmol_over_rhoH(vardict.particle)
        vardict.delif('eos')
    else: # 'hneutralssh' -> want just the total
        pass
    vardict.delif('Temperature',last=last)
    vardict.delif('nH', last=last)
    vardict.readif('Mass',rawunits=True)

    if isinstance(hab, str):
        vardict.readif(hab,rawunits = True) # may not be saved if nH was already there
        Nion = vardict.particle[hab]*h1hmolfrac*vardict.particle['Mass']
        del h1hmolfrac
        vardict.delif('Mass',last=last)
        vardict.delif(hab,last=last)
    else:
        Nion = hab*h1hmolfrac*vardict.particle['Mass']
        del h1hmolfrac
        vardict.delif('Mass',last=last)

    ionmass = ionh.atomw[string.capwords(ol.elements_ion[ion])]
    if ion == 'hmolssh': # kind of a combination for total neutral, but total h column in neutral form is more useful than working out some weighted average
        ionmass *= 2.
    to_cgs_numdens = vardict.CGSconv['Mass'] / (ionmass * c.u)

    return Nion, to_cgs_numdens # array, cgsconversion


def Nion_to_coldens(vardict, Ls, Axis1, Axis2, Axis3, npix_x, npix_y):
    afact = vardict.simfile.a
    area = (Ls[Axis1] / np.float32(npix_x)) * (Ls[Axis2] / np.float32(npix_y))\
            * c.cm_per_mpc ** 2 * afact**2
    return 1. / area


def luminosity_calc_halpha_fromSFR(vardict,excludeSFR,last=True,updatesel=True):

    if not vardict.isstored_part('eos'):
        vardict.readif('OnEquationOfState', rawunits=True)
        vardict.add_part('eos',vardict.particle['OnEquationOfState']> 0.)
        vardict.delif('OnEquationOfState',last=last)
    if updatesel:
        vardict.update(vardict.particle['eos'])
        vardict.readif('StarFormationRate', rawunits=True)
    else:
        vardict.readif('StarFormationRate', rawunits=True,setsel= not vardict.particle['eos'],setval=0.)
    vardict.delif('eos',last=last)
    convtolum = 1./(5.37e-42)
    return vardict.particle['StarFormationRate'], convtolum # array, cgsconversion


def readbasic(vardict, quantity, excludeSFR, last=True, **kwargs):
    '''
    for some derived quantities, certain keywords are required
    '''
    # Temperature: requires setting T=1e4 K for EOS particles depending on excludeSFR setting
    if quantity == 'Temperature':
        if excludeSFR == 'T4':
            if not vardict.isstored_part('eos'):
                vardict.readif('OnEquationOfState',rawunits=True)
                vardict.add_part('eos',vardict.particle['OnEquationOfState'] > 0.)
                vardict.delif('OnEquationOfState',last=last)
            vardict.readif('Temperature',rawunits=True,setsel = vardict.particle['eos'],setval = 1e4)
            vardict.delif('eos',last=last)
        else:
            vardict.readif('Temperature',rawunits=True)

    # Mass is not actually stored for DM: just use ones, and DM mass from file
    elif vardict.parttype == '1' and quantity == 'Mass':
        vardict.readif('ParticleIDs', rawunits=True)
        vardict.add_part('Mass', np.ones((vardict.particle['ParticleIDs'].shape[0],)))
        vardict.CGSconv['Mass'] = vardict.simfile.particlemass_DM_g
        vardict.delif('ParticleIDs', last=last)
        
    # derived properties with vardict read-in method
    elif quantity == 'lognH':
        vardict.getlognH(last=last,**kwargs)
    elif quantity == 'logT':
        vardict.getlogT(last=last,logT=(excludeSFR == 'T4')) # excludeSFR setting determines wheather to use T4 or not
    elif quantity == 'propvol':
        vardict.getpropvol(last=last)
    # default case: standard simulation quantity read-in
    elif quantity in ['propvol', 'ipropvol']:
        vardict.readif('Mass', rawunits=True)
        vardict.readif('Density', rawunits=True)
        if quantity == 'propvol': 
            vardict.add_part('propvol', vardict.particle['Mass'] / vardict.particle['Density'])
            vardict.CGSconv['propvol'] = vardict.CGSconv['Mass'] / vardict.CGSconv['Density']
        elif quantity == 'ipropvol':
            vardict.add_part('ipropvol',  vardict.particle['Density'] / vardict.particle['Mass'])
            vardict.CGSconv['ipropvol'] = vardict.CGSconv['Density'] / vardict.CGSconv['Mass'] 
        vardict.delif('Mass', last=last)
        vardict.delif('Density', last=last)
    else:
        vardict.readif(quantity,rawunits=True)



def project(NumPart, Ls, Axis1, Axis2, Axis3, box3, periodic, npix_x, npix_y,\
            kernel, dct, tree, ompproj=True, projmin=None, projmax=None):
    '''
    input:
    --------------------------------------------------------------------------
    - NumPart: number of SPH particles to project (int)
    - Ls:      dimensions (diameter) of the box to project (same units as 
               coordinates)
               length 3, indexable
    - Axis<i>: for Ls and coordinates, these variables control which axis is 
               the the projection axis (Axis3), and the orientation of the 
               other two. For a z-projection with a resulting array 
               (X index, Y index), use Axis1=0, Axis2=1, Axis3=2
    - box3:    the dimensions of the parent box (same units as Ls)
               length 3, indexable
    - periodic: is the projected region (perpendicular to the line of sight) a
               full slice of a periodic simulation (gas distributions at the 
               edges should be wrapped around the box), or a smaller part of 
               the simulation (gas contributions outside the projected region 
               edges should be ignored)
    - npix_x,y: how many pixels to use in the Axis1 and Axis2 directions, 
               respectively. Note that the minimum smoothing length is set to 
               the pixel diagonal, so very non-square pixels won't actually add
               much resolution in the higher-resolution direction.
               integers
    - kernel:  what shape to assume for the gas distribution respresented by a 
               single SPH particle 
               'C2' or 'gadget'
    - dct must be a dictionary containing arrays 
      'coords', 'lsmooth', 'qW', 'qQ' (prevents copying of large arrays)
      o 'coords': coordinates, (Numpart, 3) array. Coordinates should be 
                  transformed so that the projected region is a 
                  [-Ls[0] / 2., Ls[0] / 2.,\
                   -Ls[1] / 2., Ls[1] / 2.,\
                   -Ls[2] / 2., Ls[2] / 2. ]
                  box (if not periodic) or
                  [0., Ls[0], 0., Ls[1], 0., Ls[2]] if it is.
                  (The reason for this assumption in the periodic case is that 
                  it makes it easy to determine when something needs to be 
                  wrapped around the edge, and for the non-periodic case, it 
                  allows the code to ignore periodic conditions even though the
                  simulations are periodic and the selected region could 
                  therefore in principle require wrapping.)
      o 'lsmooth': gas smoothing lengths (same units as coords)
      o 'qW':     the array containing the particle property to directly,
                  project, and to weight qQ by
      o 'qQ':     the array to get a qW-weighted average for in each pixel
    - projmin, projmax: maximum coordinate values in projection direction
                  (override default values in Ls; I put this in for a specific
                  application)
              
    returns:
    --------------------------------------------------------------------------
    (ResultW, ResultQ) : tuple of npix_x, npix_y arrays (float32)
    - ResultW: qW projected onto the grid. The array contains the sum of qW 
               contributions to each pixel, not a qW surface density.
               the sums of ResultW and qW shoudl be the same to floating-point 
               errors when projecting a whole simulation, but apparent mass 
               loss in the projection may occur when projecting smaller 
               regions, where some particles in the qW array are (partially) 
               outside the projected region
    - ResultQ: qW-weighted average of qQ in each pixel
    '''

    # positions [Mpc / cm/s], kernel sizes [Mpc] and input quantities
    # a quirk of HsmlAndProject is that it only works for >= 100 particles. Pad with zeros if less.
    if NumPart >=100:
        pos = dct['coords'].astype(np.float32)
        Hsml = dct['lsmooth'].astype(np.float32)
        qW = dct['qW'].astype(np.float32)
        qQ = dct['qQ'].astype(np.float32)

    else:
        qQ = np.zeros((100,), dtype=np.float32)
        qQ[:NumPart] = dct['qQ'].astype(np.float32)
        qW = np.zeros((100,), dtype=np.float32)
        qW[:NumPart] = dct['qW'].astype(np.float32)
        Hsml = np.zeros((100,), dtype=np.float32)
        Hsml[:NumPart] = dct['lsmooth'].astype(np.float32)
        pos = np.ones((100,3), dtype=np.float32) * 1e8  #should put the particles outside any EAGLE projection region
        pos[:NumPart,:] = dct['coords'].astype(np.float32)
        NumPart = 100

    # ==============================================
    # Putting everything in right format for C routine
    # ==============================================

    print('\n--- Calling findHsmlAndProject ---\n')

    # define edges of the map wrt centre coordinates [Mpc]
    # in the periodic case, the C function expects all coordinates to be in the [0, BoxSize] range (though I don't think it actually reads Xmin etc. in for this)
    # these need to be defined wrt the 'rotated' axes, e.g. Zmin, Zmax are always the min/max along the projection direction
    if not periodic: # 0-centered
        Xmin = -0.5 * Ls[Axis1]
        Xmax =  0.5 * Ls[Axis1]
        Ymin = -0.5 * Ls[Axis2]
        Ymax =  0.5 * Ls[Axis2]
        if projmin is None:
            Zmin = -0.5 * Ls[Axis3]
        else:
            Zmin = projmin
        if projmax is None:
            Zmax = 0.5 * Ls[Axis3]
        else:
            Zmax = projmax

    else: # half box centered (BoxSize used for x-y periodic boundary conditions)
        Xmin, Ymin = (0.,) * 2
        Xmax, Ymax = (box3[Axis1], box3[Axis2])
        if projmin is None:
            Zmin = 0.5 * (box3[Axis3] - Ls[Axis3])
        else:
            Zmin = projmin
        if projmax is None:
            Zmax = 0.5 * (box3[Axis3] + Ls[Axis3])
        else:
            Zmax = projmax

    BoxSize = box3[Axis1]

    # maximum kernel size [Mpc] (modified from Marijke's version)
    Hmax = 0.5 * min(Ls[Axis1],Ls[Axis2]) # Axis3 might be velocity; whole different units, so just ignore

    # arrays to be filled with resulting maps
    ResultW = np.zeros((npix_x, npix_y)).astype(np.float32)
    ResultQ = np.zeros((npix_x, npix_y)).astype(np.float32)

    # input arrays for C routine (change in c_pos <-> change in pos)
    c_pos = pos[:,:]
    c_Hsml = Hsml[:]
    c_QuantityW = qW[:]
    c_QuantityQ = qQ[:]
    c_ResultW = ResultW[:,:]
    c_ResultQ = ResultQ[:,:]

    # check if HsmlAndProject changes
    print('Total quantity W in: %.5e' % (np.sum(c_QuantityW)))
    print('Total quantity Q in: %.5e' % (np.sum(c_QuantityQ)))

    # path to shared library
    if ompproj:
        sompproj = '_omp'
    else:
        sompproj = ''
    if tree:
        # in v3, projection can use more particles than c_int max,
        # but the tree building cannot
        if not ct.c_int(NumPart).value == NumPart:
            print(' ***         Warning         ***\n\nNumber of particles %i overflows C int type.\n This will likely cause the tree building routine in HsmlAndProjcet_v3 to fail.\nSee notes on v3 version.\n\n*****************************\n')
        if periodic:
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_%s_perbc%s.so' %(kernel, sompproj)
        else:
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_%s%s.so' %(kernel, sompproj)
    else:
        if periodic:
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_notree_%s_perbc%s.so' %(kernel, sompproj)
        else:
            lib_path = ol.hsml_dir + 'HsmlAndProject_v3_notree_%s%s.so' %(kernel, sompproj)

    print('Using projection file: %s \n' % lib_path)
    # load the library
    my_library = ct.CDLL(lib_path)

    # set the parameter types (numbers with ctypes, arrays with ndpointers)
    my_library.findHsmlAndProject.argtypes = [ct.c_long,
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,3)),
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_float,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_int,
                                  ct.c_float,
                                  ct.c_double,
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(npix_x,npix_y)),
                                  np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(npix_x,npix_y))]

    # set the return type
    my_library.findHsmlAndProject.restype = None

    print('----------')

    # call the findHsmlAndProject C routine
    my_library.findHsmlAndProject(ct.c_long(NumPart),   # number of particles in map
                                  c_pos,                # positions wrt to centre (NumPart, 3)
                                  c_Hsml,               # SPH kernel
                                  c_QuantityW,          # quantity to be mapped by projection (or weighting for average)
                                  c_QuantityQ,          # quantity to be mapped by averaging
                                  ct.c_float(Xmin),     # left edge of map
                                  ct.c_float(Xmax),     # right edge of map
                                  ct.c_float(Ymin),     # bottom edge of map
                                  ct.c_float(Ymax),     # top edge of map
                                  ct.c_float(Zmin),     # near edge of map
                                  ct.c_float(Zmax),     # far edge of map
                                  ct.c_int(npix_x),     # number of pixels in x direction
                                  ct.c_int(npix_y),     # number of pixels in y direction
                                  ct.c_int(ol.desngb),  # number of neightbours for SPH interpolation
                                  ct.c_int(Axis1),      # horizontal axis (x direction)
                                  ct.c_int(Axis2),      # vertical axis (y direction)
                                  ct.c_int(Axis3),      # projection axis (z direction)
                                  ct.c_float(Hmax),     # maximum allowed smoothing kernel
                                  ct.c_double(BoxSize), # size of simulation box
                                  c_ResultW,            # RESULT: map of projected QuantityW (npix_x, npix_y)
                                  c_ResultQ)            # RESULT: map of projected QuantityQ weighted by QuantityW (npix_x, npix_y)

    print('----------')

    # check if mapped quantities conserved (but first one also counts some particles outside the map actually)
    print('Total quantity W in:  %.5e' % (np.sum(c_QuantityW)))
    print('Total quantity W out: %.5e' % (np.sum(ResultW)))
    print('Total quantity Q in:  %.5e' % (np.sum(c_QuantityQ)))
    print('Total quantity Q out: %.5e' % (np.sum(ResultQ)))

    return ResultW, ResultQ





########################################################################################################################





#### tables of particle properties needed for different functions, subfunctions requiring further quantites, and setting for wishlist quantities (T4EOS (bool) for Temperature and logT, hab for lognH)

# thin wrapper for functions with some extra information for wishlisting
class DocumentedFunction:
    '''
    function: the function to call
    subfunctions: list of other DocumentedFunction instances
    neededquantities: Vardict.particle entries the function uses (excl. only
    needed by subfunctions)
    settings: e.g. T4EOS, hab: quantities that determine the value of the
              needed quantities
    '''
    def __init__(self,function, subfunctions, neededquantities, neededsettings):
        self.function = function
        self.subfunctions = subfunctions
        self.needed = neededquantities
        self.reqset = neededsettings

    def __call__(self,*args,**kwargs):
        return self.function(*args,**kwargs)

def getwishlist(funcname,**kwargs):
    # key words: eltab, hab (str, float, or None (unspecified) values),
    # matches output of settings, so these can be re-input as kwargs for subfunctions
    '''
    Outputs:
    - the raw and derived quantities a function uses, specifying
      subfunctions' outputs, but not their internal requiremtnt,
    - the names of subfunctions used to get derived quanitites
      (required format 'get<output quantity label in vardict.particle>')
      list is in order of calling, so that settings reflect the last saved
      version
    - settings: options for quantities used in the subfunctions in case of
      ambiguities.
        Temperature, logT: bool - T4EOS used (True) or not (False)
        eltab, hab: str or float, as produced by get_eltab_names
      this is used when the vardict.particle key for a quantity does not
      reflect this information, but the values it stores depend on it: e.g.
      eltab and hab labels contain this information, but the derived lognH
      does not. In normal calculations, this is useful, since the rest of e.g.
      a luminosity calculation is independent of these settings, but in saving
      quantities, it could cause the wrong quantity to be used in a second
      calculation

    !!! If a function uses different settings internally, the wishlist
    combination routine can easily fail; see that function for notes on how to
    handle functions like that !!!

    '''

    # set kwargs defaults:
    if not 'eltab' in kwargs.keys():
        kwargs.update({'eltab': None})
    if not 'hab' in kwargs.keys():
        kwargs.update({'hab': None})
    if not 'Temperature' in kwargs.keys():
        kwargs.update({'Temperature': None})
    if not 'logT' in kwargs.keys():
        kwargs.update({'logT': None})

    if funcname == 'luminosity_calc':
        if kwargs['eltab'] is None or kwargs['logT']is None or kwargs['hab'] is None:
            print('eltab, hab, and logT must be specified to generate the wishlist for luminosity_calc.')
            return None
        else:
            subs = ['getpropvol', 'getlognH', 'getlogT']
            needed = ['lognH','logT','propvol']
            settings = {'logT': kwargs['logT'], 'hab': kwargs['hab'], 'eltab': kwargs['eltab']}
            if isinstance(kwargs['eltab'],str):
                needed += [kwargs['eltab']]
            return (needed, subs, settings)

    elif funcname == 'Nion_calc':
        if kwargs['eltab'] is None or kwargs['logT']is None or kwargs['hab'] is None:
            print('eltab, hab, and logT must be specified to generate the wishlist for Nion_calc.')
        else:
            subs = ['getlognH', 'getlogT']
            needed = ['lognH','logT','Mass']
            settings = {'logT': kwargs['logT'], 'hab': kwargs['hab'], 'eltab': kwargs['eltab']}
            if isinstance(kwargs['eltab'],str):
                needed += [kwargs['eltab']]
            return (needed, subs, settings)

    elif funcname == 'Nelt_calc':
        if kwargs['eltab'] is None:
            print('eltab must be specified to generate the wishlist for Nelt_calc.')
        else:
            subs = []
            needed = ['Mass']
            settings = {'eltab': kwargs['eltab']}
            if isinstance(kwargs['eltab'],str):
                needed += [kwargs['eltab']]
            return (needed, subs, settings)

    elif funcname == 'luminosity_calc_halpha_fromSFR':
        subs = []
        needed = ['eos','StarFormationRate'] # OnEquationOfState is (so far) only ever used to get eos
        settings = {}
        return (needed,subs,settings)

    elif funcname == 'getlognH':
        if kwargs['hab'] is None:
            print('hab must be specified to generate the wishlist for getlognH.')
        else:
            subs = []
            needed = ['Density']
            settings = {'hab': kwargs['hab']}
            if isinstance(kwargs['hab'],str):
                needed += [kwargs['hab']]
            return (needed,subs,settings)

    elif funcname == 'getlogT':
        if kwargs['logT'] is None:
            print('logT must be specified to generate the wishlist for getlogT.')
        subs = []
        needed = ['Temperature']
        if kwargs['logT']:
            needed += ['OnEquationOfState']
        settings = {'Temperature': kwargs['logT']}
        return (needed, subs, settings)

    elif funcname == 'getpropvol':
        subs = []
        needed = ['Mass','Density']
        settings = {}
        return (needed,subs,settings)

    else:
        print('No entry in getwishlist for function %s'%funcname)
        return None

# checks a set of functions and returns needed, subs, settings for the whole lot, assuming the settings agree between all the subs
def getsubswishlist(subs,settings):
    neededsubs = set()
    subsubs = set()
    settingssubs = {}
    for sub in subs:
        neededsub, subsub, settingssub = getwishlist(sub,**settings)
        neededsubs |= set(neededsub)
        subsubs |= set(subsub )
        settingssubs.update(settingssub)
    return neededsubs, list(subsubs), settingssubs

# settings dependencies for subfunction outputs:
settingsdep = {\
'hab': {'lognH'},\
}

def removesettingsdiffs(overlap,settings1,settings2):
    '''
    removes anything from overlap with a settings discrepancy
    returns new overlap
    '''
    settingskeys12 = set(settings1.keys())&set(settings2.keys()) # settings relevant for both functions, with updated settings2
    settingseq = {key: settings1[key] == settings2[key] for key in settingskeys12} # check where settings agree
    # if a quantity is needed in both functions, but with different settings, saving it will only cause trouble
    for key in settingseq.keys():
        if settingseq[key] == False:
            overlap.remove(key)
            if key in settingsdep:
                overlap -= settingsdep[key]
    return overlap

# wishlist given getwishlist output for the first and second functions to run
def combwishlist(neededsubssettingsfirst,neededsubssettingssecond):
    '''
    !!! May fail if a function or (nested) subfunction uses different settings
    internally !!!
    (Probably best not to try so save anything from something like that anyway;
    do not include that quantity in the getwishlist needed/subs list, and set
    the relevant setting to None. Then any wishlisting will not include these
    quantities.)
    '''
    used1, subs1, settings1 = neededsubssettingsfirst
    needed2, subs2, settings2 = neededsubssettingssecond

    ## want to examine the whole tree of quantities used by function1 -> loop over nested subfunctions

    nextsubs = subs1
    used1 = set(used1)
    ## again, assumes first function uses the same settings throughout
    while len(nextsubs) > 0: # while there is a next layer of subfunctions
        usedsub, nextsubs, settingssub = getsubswishlist(nextsubs,settings1)
        settings1.update(settingssub) # add next layer of settings to everything function 1 uses
        used1 |= usedsub # add next layer of used quantities to everything function 1 uses



    needed12 = set(needed2)&used1 # stuff for second function that the first already gets
    needed12 = removesettingsdiffs(needed12,settings1,settings2)

    # loop over nested subfunctions: if a subfunction result is not saved, check if a quantity it uses should be
    nextsubs = subs2
    while len(nextsubs) > 0: # loops over all nested subfunctions
        # checks which of the subfunctions don't have their outcomes on the wishlist already
        nextsubproducts = set([nextsub[3:] for nextsub in nextsubs]) # remove 'get' from the function name
        nextsubproducts -= needed12 # get the subfunction products that are not already stored
        nextsubs = ['get%s'%nextproduct for nextproduct in nextsubproducts] # gets the subfunctions whose products are not already stored, so those whose ingredients we want to keep in function 1 gets them

        # gets the requirements for the selected subfunctions and any additiona settings
        neededsub, nextsubs, settingssub = getsubswishlist(nextsubs,settings2)
        settings2.update(settingssub) # add next layer of settings to everything function 2 needs
        needed12 |= neededsub&used1 # add next layer of used quantities to everything function 2 needs
        needed12 = removesettingsdiffs(needed12,settings1,settings2)

    return list(needed12)


def saveattr(grp, name, val):
    '''
    meant for relatively simple cases; do not apply to e.g. selection tuples 
    with mixed types
    '''
    if sys.version.split('.')[0] == '3':
        def isstr(x):
            return isinstance(x, str)
    elif sys.version.split('.')[0] == '2':
        def isstr(x):
            return isinstance(x, basestring)
    else:
        raise RuntimeError('Only python versions 2 and 3 are supported')
        
    if isinstance(val, dict):
        subgrp = grp.create_group(name)
        for key in val.keys():
            saveattr(subgrp, key, val[key])
    elif isstr(val):
        grp.attrs.create(name, np.string_(val))
    elif hasattr(val, '__len__'):
        valt = np.array(val)
        if np.any([isstr(x) for x in valt.flatten()]): # store all values as strings in any value is a string
            valt = valt.astype(np.string_)
        grp.attrs.create(name, valt)
    elif val is None:
        grp.attrs.create(name, np.string_('None'))
    else:
        grp.attrs.create(name, val)
        
def savemap_hdf5(hdf5name, projmap, minval, maxval,
         simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y,
         ptypeW,
         ionW, abundsW, quantityW,
         ionQ, abundsQ, quantityQ, ptypeQ,
         excludeSFRW, excludeSFRQ, parttype,
         theta, phi, psi,
         sylviasshtables, bensgadget2tables,
         ps20tables, ps20depletion,
         var, axis, log, velcut,
         periodic, kernel, saveres,
         simulation, LsinMpc, misc, ompproj, numslices,
         halosel, kwargs_halosel, cosmopars, override_simdatapath, groupnums):
    '''
    save projmap, minval, maxval with npzname and the processed input 
    parameters
    groupnums is only stored if halosel is not None
    '''
    
    #print('save hdf5 function called')
    with h5py.File(hdf5name, 'w') as fh:
        hed = fh.create_group('Header/inputpars')
        # relatively simple cases
        saveattr(hed, 'simnum', simnum)
        saveattr(hed, 'snapnum', snapnum)
        saveattr(hed, 'centre', centre)
        saveattr(hed, 'L_x', L_x)
        saveattr(hed, 'L_y', L_y)
        saveattr(hed, 'L_z', L_z)
        saveattr(hed, 'npix_x', npix_x)
        saveattr(hed, 'npix_y', npix_y)
        saveattr(hed, 'var', var)
        
        saveattr(hed, 'ptypeW', ptypeW)
        saveattr(hed, 'ionW', ionW)
        saveattr(hed, 'abundsW', abundsW)
        saveattr(hed, 'quantityW', quantityW)
        saveattr(hed, 'excludeSFRW', excludeSFRW)
        
        saveattr(hed, 'ptypeQ', ptypeQ)
        saveattr(hed, 'ionQ', ionQ)
        saveattr(hed, 'abundsQ', abundsQ)
        saveattr(hed, 'quantityQ', quantityQ)
        saveattr(hed, 'excludeSFRQ', excludeSFRQ)
        
        saveattr(hed, 'parttype', parttype)
        saveattr(hed, 'var', var)
        saveattr(hed, 'axis', axis)
        saveattr(hed, 'log', log)
        saveattr(hed, 'velcut', velcut)
        saveattr(hed, 'periodic', periodic)
        saveattr(hed, 'kernel', kernel)
        saveattr(hed, 'simulation', simulation)
        saveattr(hed, 'LsinMpc', LsinMpc)
        saveattr(hed, 'ompproj', ompproj)
        saveattr(hed, 'numslices', numslices)
        saveattr(hed, 'sylviasshtables', sylviasshtables)
        saveattr(hed, 'bensgadget2tables', bensgadget2tables)
        saveattr(hed, 'ps20tables', ps20tables)
        saveattr(hed, 'ps20depletion', ps20depletion)
        saveattr(hed, 'theta', theta)
        saveattr(hed, 'phi', phi)
        saveattr(hed, 'psi', psi)
        saveattr(hed, 'override_simdatapath', override_simdatapath)
        
        saveattr(hed, 'cosmopars', cosmopars)
        
        saveattr(hed, 'make_maps_opts_locs.emtab_sylvia_ssh', 
                 str(ol.emtab_sylvia_ssh))
        saveattr(hed, 'make_maps_opts_locs.iontab_sylvia_ssh', 
                 str(ol.iontab_sylvia_ssh))
        
        hsel = hed.create_group('halosel')
        saveattr(hsel, 'kwargs_halosel', kwargs_halosel)
        if halosel is None:
            saveattr(hsel, 'any_selection', False)
        else:
            saveattr(hsel, 'any_selection', True)
            selection_element_counter = 0
            for selection_element in halosel:
                sgrp = hsel.create_group('tuple_%i'%selection_element_counter)
                for tuple_index in range(len(selection_element)):
                    val = selection_element[tuple_index]
                    if isinstance(val, dict):
                        for key in val.keys():
                            saveattr(sgrp, key, val[key])
                    else:
                        saveattr(sgrp, 'tuple_index_%i'%tuple_index, val)
                selection_element_counter += 1
            hsel.create_dataset('groupnums', data=groupnums)
        
        mgrp = hed.create_group('misc')
        if misc is None:
            saveattr(mgrp, 'any_values', False)
        else:
            saveattr(mgrp, 'any_values', True)
        saveattr(mgrp, 'dict', misc)
        
        #fh['Header'].attrs.create('string encoding', np.string_(encoding)) # record how to get from bytes back to python strings

        # main map save
        ds_map = fh.create_dataset('map', data=projmap)
        ds_map.attrs.create('max', maxval)
        ds_map.attrs.create('minfinite', minval)
        
##########################################################################################

def make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW=None, abundsW='auto', quantityW=None,
         ionQ=None, abundsQ='auto', quantityQ=None, ptypeQ=None,
         excludeSFRW=False, excludeSFRQ=False, parttype='0',
         theta=0.0, phi=0.0, psi=0.0, \
         sylviasshtables=False, bensgadget2tables=False,
         ps20tables=False, ps20depletion=True,
         var='auto', axis='z',log=True, velcut=False,\
         periodic=True, kernel='C2', saveres=False,\
         simulation='eagle', LsinMpc=None,\
         select=None, misc=None, halosel=None, kwargs_halosel=None,\
         ompproj=False, nameonly=False, numslices=None, hdf5=False,\
         override_simdatapath=None):

    """
    make a map of an integral quantity (W), and optionally, a local quantity 
    weighted by an integral quantity (Q), from a numerical simulation
    
    Parameters
    ----------
    
    ---------------------
    simulation to analyse
    ---------------------
    simnum:    string
        which simulation volume to use. For Eagle, the format is 'L####N####'  
    var: string
        which feedback/physics variation to use: e.g., 'REFERENCE',
        'RECALIBRATED'. Should match the value in the directory path for the
        target simulation. The default is 'auto'; this means 'REFERENCE' for 
        all Eagle volumes but L0025N0752, where the default means 
        'RECALIBRATED'.
    snapnum: int
        the number of the simulation snapshot to use
    simulation: string
        which simulation set to use, e.g. 'eagle' or 'bahamas' 
        (or 'eagle-ioneq'). The default is 'eagle'.
    override_simdatapath: None or a string 
        if this isn't None, the sring specifies the directory containing
        e.g. the snapshot.../, group.../ directories for non-standard file 
        organisations where the format assumed in make_maps_opts_locs.py, 
        projection_classes.py, and read_eagle_files.py will fail at higher 
        levels than this.
        ! Beware: this overrides the simnum and var options, so ideally, set 
        these paths in a wrapper scripts that ensures that the files and 
        assumed simulation parameters agree
        
    ------------------------------------------
     what to project (SPH particle selection)
    ------------------------------------------
    centre: list-like or 3 floats
        centre of the box to select particles from.
    L_x (y,z): float
        total length of the region to project in the x, (y,z) direction.
    LsinMpc: bool or None
        L_x,y,z and centre are given in Mpc (True) or Mpc/h (False). 
        If None, this is set to True for EAGLE, and False for BAHAMAS. The
        default None.
   
    theta, psi, phi: float
        angles to rotate coordinates before projecting. The defaults are 0.
        UNIMPLEMENTED after modifications to Marijke Segers' make_maps
    axis:  string; options: 'x', 'y', 'z'
        axis to project along. The default is 'z'.       
    velcut: bool, float, or (float, float) tuple
        Select particles along the line of sight by velocity (Hubble flow +
        peculiar) instead of position. Options are:
        True:  uses the Hubble flow equivalent of the region given in position
               space (centre, L_x, L_y, L_z) along the projection direction.
        False: no velocity information is used; particles are selected only on
               position
        float: particles with velocities within +/- the input value 
               (proper km/s) of the Hubble flow velocity at given centre are
               used. Particles are chosen only from the position-selected 
               region. (Meaning there is a dual position/velocity selection
               along the projection direction.) The values is assumed to be/
               set to a positive value.    
        (float, float): the first value is a velocity offset relative to the
               hubble flow at the centre, the second is as above. Velocities 
               are rest-frame km/s.
        The default is False.
    npix_x, npix_y: int, >0   
        number of pixels to use in the projection in the x and y
        directions. File naming only uses the number of x pixels, and the 
        minimum smoothing length uses the pixel diagonal, so using non-square 
        pixels will not improve the resolution along one direction by much.
        The x and y directions correspond to the axes after projection. If the 
        projection direction is 'z', they corresond to the simulation x and y
        axes. For 'x' projections, npix_x -> simulation y, npix_y -> 
        simulation z, and for 'y' projections, npix_x -> simulation z, npix_y
        -> simulation x
    nameonly: bool
        if True, don't do any projection, just return the npz/hdf5 file names
        you would get using the same parameters (assuming savres=True).
        The default is False. 
    numslices: int or None
        if not None, cut the projection region into numslices slices along the
        projection axis, and project and save each separately. If the 
        projection axis selection is on velocity, this slicing will also be in 
        velocity space. (This is useful for thin slices, or velocity space, 
        when data for many particles will loaded anyway.)
    halosel: list of tuples or None  
        if not None, only include particles belonging to FOF halos meeting 
        these selection criteria. See documentation of selecthaloparticles for
        the format, or sh.selecthalos_subfindfiles (sh = selecthalos.py) for 
        more details.
        The default is None, meaning there is no selection on halo membership.
        Note that an empty selection will include all halo particles, not all 
        particles: halosel=[] gives different results from halosel=None.
    kwargs_halosel: kwargs for selecthalos 
        aperture: int
            aperture in which to get e.g. stellar masses (physical kpc). The
            default is 30.
        mdef: string
            The halo mass definition to use. The default is '200c'.
        exclsatellites: bool
            Only halo particles belonging to centrals/FOF main halos are 
            included. exclsatellites determines whether SubGroupNumber !=0 gas
            (belonging to subhalos) is explicitly excluded (True). Note that 
            this means unbound gas (e.g. at the edges of the halo) is also 
            excluded. The default is False.
        allinR200c: bool
            include particles inside R200c but not in the FOF group. The 
            default is True.
        label: string or omit
            replace the automatic name for the halo selection with this label 
            (useful for more complicated selections which otherwise produce 
             'filename too long' IOErrors)
            recommended to use with the hdf5 saving option, which will 
            save the exact selection used. Otherwise, the exact parameter 
            documentation relies on external files/notes.
                
    The chosen region is assumed to be a continuous block. 

    -----------------
     quantities to project
    -----------------
    two quantities can be projected: W and Q
    W is projected directly: the output map is the sum of the particle
        contributions in each grid cell
    for Q, a W-weighted average is calculated in each cell
    Parameters describing what quantities to calculate have W/Q versions, that
        do the same thing, but for the different quantities. For the Q options,
        None can be used for all if no weighted average is desired.

    ptypeW/ptypeQ: str    
        the category of quantity to project. Options are 'basic', 'emission',
        'coldens', and for ptypeQ, None.
        'basic' means a quantity stored in the EAGLE output, and None for 
        pytypeQ means no weighted average map is computed, only the intergral
        quantity map. For ptypeQ, the default is None.
    ionW/ionQ: str    
        ion/element for which to calculated the column density (ptype 
        'coldens') or ion/line of which to calculated the emission (ptype 
        'emission'). For the standard tables, the options are given in 
        make_maps_opts_locs.py. The argument is required for ptype[W/Q] 
        options 'emission' and 'coldens'; for the ptype option 'basic', the 
        ion option is ignored.
        see make_maps_opts_locs.py for options       
    quantityW/quantityQ: str
        the quantity from the EAGLE output to project. This should be the path
        in the hdf5 file starting after 'PartType#/'. The argument is required 
        for ptype option 'basic'; for ptype options 'emission' and 'coldens', 
        the option is ignored.
    parttype: str
        the particle type to project:
        '0': gas
        '1': DM !! DM mass projection only work for Eagle 
                  (Simfile particle mass read-in), and will fail for the full
                  100 cMpc volume due to counter int/long int issues in the 
                  smoothing length calculation.
        '4': Stars
        '5': BHs
        The argument is required for ptype option 'basic'. For ptype options 
        'emission' and 'coldens', the option is ignored. The default is '0'.
    -------------------
     technical choices
    -------------------
    abundsW/abundsQ: str, float, or tuple(option, option)  
        type of SPH element abundance to use in 'coldens' or 'emission' 
        calculations. If one option is given, smoothed/particle abundances are
        used for both nH and element abundances.
        A float means a fixed element abundance in eagle solar units (see 
        make_maps_opts_locs). For emission and coldens, the primordial 
        hydrogen abundance is then used to calculate lognH. 'Sm' means use 
        smoothed abudances, 'Pt' means particle abundances.
        If a tuple is given, the first element (index 0) is for the element 
        abundance, and the second for hydrogen (lognH calculation for emission 
        and absorption). The float option for logNh here is the (absolute) 
        mass fraction.
        'auto' means use the smoothed abundance for ptype 'emission', and 
        particle for 'coldens'. As a second tuple argument, 'auto' means use
        the same as option for hydrogen as the elemen (and a primordial 
        hydrogen mass fraction if a float value is given).
        options are 'Sm', 'Pt', 'auto', float, or a 2-tuple of these.
        For emission from hydrogen, only the hydrogen abundance matters; the 
        element abundance is not used in this case. Mixing element and 
        hydrogen settings is, in any case, not recommended. 
        The default is 'auto'.
    kernel: str   
        smoothing kernel to use in projections. The options are 'C2' and 
        'gadget'. The default is 'C2'.
        See HsmlAndProject for other (unimplemented) options
    periodic:  bool
        use periodic boundary conditions (not along projection axis). Always 
        set True if you're using the whole box perpendicular to the projection
        axis and False otherwise.
    excludeSFRW/excludeSFRQ: str or bool
        how to handle particle on the equation of state (star-forming gas).
        The options are:
        True:  exclude EOS particles
        False: include EOS particles at face temperature
        'T4':  include EOS particles at T = 1e4 K
        'only': include only EOS particles
        'from': use only EOS particles or calculate halpha emission from the 
                star formation rate (ptype 'emission', currently only for ion 
                'halpha')
        Since Q and W must use the same particles, only False and T4
        or from and only can be combined with each other.
    misc: dct or None     
        intended for one-off uses without messing with the rest of the code.
        Used in nameoutput with simple key-value naming. No checks in 
        inputcheck: typically single-funciton modifications; checks are done 
        there. The default is None.
    ompproj: bool  
        use the OpenMP implementation of the C projection routine. Faster, but
        not necessary for small box/region tests. Do set OMP_NUM_THREADS in 
        the shell to restrict the number of threads if you use the 
        multithreading implementation on a shared system.
    sylviasshtables: bool, DEPRECATED
        Use Sylvia's tables to calculate ion fractions. These assume an HM12 
        UV/X-ray background and use a newer Cloudy version, compared to 
        EAGLE's HM01 and older Cloudy cooling (Wiersma et al. 2009). However, 
        these do include self-shielding, which is not included in the EAGLE 
        cooling tales, but is needed for realistic low ion properties.
        !! The location of these tables in make_maps_opts_locs.py has been 
        overwritten with the location of the Ploeckinger & Schaye (2020) 
        tables: the final, published version for which the original ssh tables
        were a work in progress. The default is False
    ps20tables: bool
        Use the Ploeckinger & Schaye (2020) tables to calculate ion fractions 
        or line emission. These assume an FG20 UV/X-ray background and use a 
        newer Cloudy version, compared to EAGLE's HM01 and older Cloudy 
        cooling (Wiersma et al. 2009). However, these do include 
        self-shielding and other processes important for ISM gas and low ions, 
        which is not included in the EAGLE cooling tales, but are needed for 
        realistic low ion properties. They also do not contain a bug in the
        Fe L-shell emission lines that is present in the default tables. The
        variation of Ploeckinger & Schaye table can be adapted by setting 
        iontab_sylvia_ssh and emtab_sylvia_ssh in make_maps_opts_locs.py. 
        The default is False.
    ps20depletion: bool
        Include the effects of dust depletion on ion/element content of the 
        gas or element emission. The value is only used if ps20tables is True.
        The default is True, but this might fail if a 'dust0' table is used. 
    bensgadget2tables: bool
        use Ben Oppenheimer's tables made for work on Gadegt-2 simulations to 
        calculate ion fractions; these are made under the same assumptions 
        (HM01 UV/X-ray background, solar Z), but with a newer cloudy version, 
        than the default tables from Serena Bertone that are consistent with 
        Eagle cooling. These tables exist for a limited number of ions, but 
        include O I - VIII.
               
    --------------
    output control
    --------------
    hdf5: bool      
        save output to an hdf5 file instead of .npz (also document input 
        parameters).The default is False for backward compatibilty, but True
        is strongly recommended.
    nameonly: bool
        instead of calculating the full maps, just return the names of the 
        files you would get if saveres=True, with the same parameters. The
        default is False.
    saveres: bool 
        save the output maps to .npz or .hdf5 files. The default is False. 
    log: bool      
        return (and save) log10 of the projected values. The default is True,
        and is strongly recommended to prevent float32 overflows in the 
        output.
        
    Modify make_maps_opts_locs.py for locations of interpolation files (c),
    projection routine (c), ion balance and emission tables (hdf5) and write
    locations.
    
    ----------------------
     dct options for misc
    ----------------------
    'usechemabundtables': 'BenOpp1'
                           use Ben Oppenheimer's ChemicalAbundances tables
                           ion inclusion is checked, whether the simulation
                           contains these tables is not
                           only for simulation='eagle-ioneq', var='REFERENCE'
                           or 'ssh', simunum='L0025N0752', limited snapshots
                           only useful for column densities
                           NOT YET IMPLEMENTED CORRECTLY
    'useLSR':              bool: use local stellar radiation estimate for
                           HI/Hmolecular in self-shielding model
                           (Rahmati, Schaye, Pawlik, Raicevic 2013, equation 7)
                           default: False
    'UVB':                 X-ray/UV background to use; only for self-shielded
                           HI/Hmolecular model
                           (Rahmati, Pawlik, Raicevic, Schaye 2013)
                           default: 'HM01'
    Returns
    -------
    default: 2D array of projected emission/ions/etc.
        tuple of (integral quantity map, weighted average map or None)
    if nameonly: name of the file you would get from if saveres=True with 
        the same parameters. Tuple of (integral quantity file, weighted 
        average file or None).
    Optionally (saveres), creates an .hdf5 or .npz file (hdf5) containing the 
    2D array (naming is automatic).
    
    modify make_maps_opts_locs for locations of interpolation files (c),
    projection routine (c), ion balance and emission tables (hdf5) and write
    locations
    """
    ########################
    #   setup and checks   #
    ########################

    # Must come first! (including 'auto' option handling)
    res = inputcheck(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y,
         ptypeW,
         ionW, abundsW, quantityW,
         ionQ, abundsQ, quantityQ, ptypeQ,
         excludeSFRW, excludeSFRQ, parttype,
         theta, phi, psi,
         sylviasshtables, bensgadget2tables,
         ps20tables, ps20depletion,
         var, axis, log, velcut,
         periodic, kernel, saveres,
         simulation, LsinMpc,
         select, misc, ompproj, numslices, halosel, kwargs_halosel,
         hdf5, override_simdatapath)
    if isinstance(res, int):
        raise ValueError("inputcheck returned error code %i"%res)

    iseltW, iseltQ, simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW, abundsW, quantityW,\
         ionQ, abundsQ, quantityQ, ptypeQ,\
         excludeSFRW, excludeSFRQ, parttype,\
         theta, phi, psi, \
         sylviasshtables, bensgadget2tables,\
         ps20tables, ps20depletion,\
         var, axis, log, velcut,\
         periodic, kernel, saveres,\
         simulation, LsinMpc, misc, ompproj, numslices,\
         halosel, kwargs_halosel, hdf5, override_simdatapath = res[1:]

    print('Processed input:')
    print((':\t%s\t'.join(['simnum', 'snapnum', 'simulation', 'var', 'parttype', '']))\
                          %(simnum,   snapnum,   simulation,   var,   parttype))
    print((':\t%s\t'.join(['centre', 'L_x', 'L_y', 'L_z', 'axis', 'velcut', 'LsinMpc', '']))\
                          %(centre,   L_x,   L_y,   L_z,   axis,   velcut,   LsinMpc))
    print((':\t%s\t'.join(['npix_x', 'npix_y', 'kernel', 'numslices', 'periodic', '']))\
                          %(npix_x,   npix_y,   kernel,   numslices,   periodic))
    print((':\t%s\t'.join(['theta', 'phi', 'psi,', '']))\
                          %(theta, phi, psi))
    print((':\t%s\t'.join(['ptypeW', 'ionW', 'abundsW', 'quantityW', 'excludeSFRW', 'iseltW', '']))\
                          %(ptypeW,   ionW,   abundsW,   quantityW,   excludeSFRW,   iseltW))
    print((':\t%s\t'.join(['ptypeQ', 'ionQ', 'abundsQ', 'quantityQ', 'excludeSFRQ', 'iseltQ', '']))\
                          %(ptypeQ,   ionQ,   abundsQ,   quantityQ,   excludeSFRQ,   iseltQ))
    print((':\t%s\t'.join(['sylviasshtables', 'bensgadget2tables', 'ps20tables', 'ps20depletion', '']))\
                          %(sylviasshtables,   bensgadget2tables,   ps20tables,   ps20depletion))
    print((':\t%s\t'.join(['log', 'saveres', 'ompproj', 'hdf5', '']))\
                          %(log,   saveres,   ompproj,   hdf5))
    print((':\t%s\t'.join(['halosel', 'kwargs_halosel', '']))\
                          %(halosel,  kwargs_halosel))
    print((':\t%s\t'.join(['override_simdatapath', '']))\
                          %(override_simdatapath))
    print('misc:\t%s'%misc)
    print('\n')




    ##### Wishlist generation: preventing doing calculations twice

    wishlist = ['coords_cMpc-vel']
    if ptypeQ is not None:
        if ptypeQ == 'basic' and quantityQ != 'Temperature': #temperature checks come later
            wishlist += [quantityQ]
        elif (ptypeQ=='basic' and quantityQ =='Temperature') and ((excludeSFRQ == 'T4' and excludeSFRW == 'T4') or (excludeSFRQ != 'T4' and excludeSFRW != 'T4')):
            wishlist += ['Temperature']
        elif (ptypeQ == 'emission' and excludeSFRQ != 'from') or ptypeQ == 'coldens': # element abundances to be added when the right option is determined below
            wishlist += ['Density','Mass']
        elif ptypeQ == 'emission' and excludeSFRQ == 'from':
            wishlist += ['StarFormationRate']
        else:
            wishlist += []

    # since different function require smoothed or particle abundances:
    # does some imput checking
    # set eltabW/Q, habW/Q: input for read_eagle element retrieval
    # set iseltW/Q (for the coldens case): calculate an ion or element coldens
    # updates the wishlist to include all EAGLE output quantities desired.

    if ptypeW in ['emission', 'coldens']:
        eltabW, habW = get_eltab_names(abundsW, iseltW, ionW)
    else:
        eltabW, habW = (None, None)

    if ptypeQ in ['emission', 'coldens'] and excludeSFRQ != 'from':
        eltabQ, habQ = get_eltab_names(abundsQ, iseltQ, ionQ)
        wishlist += [habQ, eltabQ]
    else:
        eltabQ, habQ = (None, None)

    if (ptypeQ == 'emission' and excludeSFRQ != 'from') or ptypeQ == 'coldens': # element abundances to be added when the right option is determined below
        wishlist += ['Density', 'Mass']
    elif ptypeQ == 'emission' and excludeSFRQ == 'from':
        wishlist += ['StarFormationRate']
    else:
        wishlist = []

    #### more advanced wishlist options:
    if (ptypeW in ['emission','coldens'] and excludeSFRW != 'from') and \
       (ptypeQ in ['emission','coldens'] and excludeSFRQ != 'from'):
        if habQ == habW: #same abundance choice
            wishlist.append('lognH')
            # only needed to get lognH
            wishlist.remove('Density')
            wishlist.remove(habQ)
        if eltabQ == eltabW and (sylviasshtables or ps20tables) and ptypeW == 'coldens': #same abundance choice
            wishlist.append('logZ')
        if ptypeW == 'emission' and ptypeQ == 'emission':
            wishlist.append('propvol')
            if habW == habQ:
                wishlist.remove('Mass')

    if (ptypeQ in ['emission','coldens'] and excludeSFRQ != 'from') and ((excludeSFRQ == 'T4' and excludeSFRW == 'T4') or (excludeSFRQ != 'T4' and excludeSFRW != 'T4')):
        # if Q needs temperature and W is not using a different version (T4 vs. not T4)
        if ptypeW in ['coldens', 'emission'] and excludeSFRW !='from':
            wishlist.append('logT')
        else:
            wishlist.append('Temperature') # don't want to save the temperature when logT is all we need

    if (ptypeW == 'coldens' and ionW in ['h1ssh', 'hmolssh']) or (ptypeQ == 'coldens' and ionQ in ['h1ssh', 'hmolssh']):
        wishlist.append('eos')

    #### set data file, setup axis handling
    if halosel is not None:
        simfile = pc.Simfile(simnum, snapnum, var, simulation=simulation,
                             file_type='particles',
                             override_filepath=override_simdatapath)
    else: # use default: snapshot data
        simfile = pc.Simfile(simnum, snapnum, var, simulation=simulation,
                             override_filepath=override_simdatapath)

    if axis == 'x':
        Axis1 = 1
        Axis2 = 2
        Axis3 = 0
    elif axis == 'y':
        Axis1 = 2
        Axis2 = 0
        Axis3 = 1
    else:
        Axis1 = 0
        Axis2 = 1
        Axis3 = 2

    #name output files (W file name is independent of Q options)
    # done before conversion to Mpc to preserve nice filenames in Mpc/h units
    
    vardict_temp = pc.Vardict(simfile, parttype, []) #argument needed in selecthalos; only the naming and some input checks are called here, only simfile properties are used for those
    resfile = nameoutput(vardict_temp, ptypeW, simnum, snapnum, version, 
                         kernel,
                         npix_x, L_x, L_y, L_z, centre, simfile.boxsize, 
                         simfile.h,
                         excludeSFRW, excludeSFRQ, velcut, sylviasshtables, 
                         bensgadget2tables, ps20tables, ps20depletion,                          
                         axis, var, abundsW, ionW, parttype, None, 
                         abundsQ, ionQ, quantityW, quantityQ,
                         simulation, LsinMpc, halosel, kwargs_halosel, 
                         misc, hdf5)
    if ptypeQ !=None:
        resfile2 = nameoutput(vardict_temp, ptypeW, simnum, snapnum, version, 
                              kernel,
                              npix_x, L_x, L_y, L_z, centre, simfile.boxsize,
                              simfile.h,
                              excludeSFRW, excludeSFRQ, velcut, sylviasshtables, 
                              bensgadget2tables,  ps20tables, ps20depletion,
                              axis, var, abundsW, ionW, parttype, 
                              ptypeQ, abundsQ, ionQ, quantityW, quantityQ,
                              simulation, LsinMpc, halosel, kwargs_halosel,
                              misc, hdf5)
    del vardict_temp
    # just get the file name for a set of parameters
    if nameonly:
        if ptypeQ is None:
            return resfile, None
        else:
            return resfile, resfile2

    Ls = np.array([L_x,L_y,L_z])
    # make sure centre, Ls are in Mpc (other option in Mpc/h)
    if not LsinMpc:
        Ls /= simfile.h
        centre = np.array(centre) / simfile.h
    else:
        centre = np.array(centre)
    centre_cMpc = np.copy(centre)

    ####################################
    # read-in and quantity calculation #
    ####################################


    if simfile.region_supported:
        if velcut==True: # in practice, velocity region selection return the whole box in the projection direction anyway
            # region, hubbleflow_cgs = partselect_vel_region(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype = parttype)
            region = partselect_pos(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype=parttype)
            if region is not None:
                region[[2*Axis3, 2*Axis3+1]] = [ 0.,simfile.boxsize]
        else: # None means use the whole box; velcut tuple -> subselect from region
            region = partselect_pos(simfile, centre, Ls, Axis1, Axis2, Axis3, parttype=parttype)
        # initiate vardict: set centre, Ls, box3, and coords if we're working in ppv space
        if np.all(region == np.array([0.,simfile.boxsize,0.,simfile.boxsize,0.,simfile.boxsize])):
            region = None # Don't bother imposing a region if everything is selected; will be the case for whole box slices.
    else:
        region = None

    vardict_WQ = pc.Vardict(simfile, parttype, wishlist, region=region, readsel=pc.Sel())
    vardict_WQ.add_box('centre', centre)
    vardict_WQ.add_box('Ls', Ls)

    # cases where Sels need to be used for read-in coordinate selection
    # else sets box3 for region selection only case
    if (velcut==True or isinstance(velcut, tuple)) and simfile.region_supported:
        #print('Starting with %i particles'%np.sum(vardict_WQ.readsel.val))
        if isinstance(velcut, tuple): # do exact selection along z axis after region choice
            ppp_selselect_coordsadd(vardict_WQ, centre, Ls, periodic, Axis1, Axis2, Axis3, keepcoords=False)
        ppv_selselect_coordsgen(centre, Ls, Axis1, Axis2, Axis3, periodic, vardict_WQ, velcut=velcut)
    elif not vardict_WQ.simfile.region_supported and velcut==False: #set region by setting readsel
        ppp_selselect_coordsadd(vardict_WQ, centre, Ls, periodic, Axis1, Axis2, Axis3)
    elif not vardict_WQ.simfile.region_supported and (velcut==True or isinstance(velcut, tuple)): # ppv_coordselect uses previously set readsels
        Ls_temp = np.copy(Ls)
        if velcut == True: # if tuple, we want to maintain the initial spatial selection along the projection axis
            Ls_temp[Axis3] = simfile.boxsize / simfile.h
        ppp_selselect_coordsadd(vardict_WQ, centre, Ls_temp, periodic, Axis1, Axis2, Axis3, keepcoords=False) # does the Axis1, Axis2 selection, and Axis3 if pos+vel selection
        if vardict_WQ.readsel.seldef:
            print('%i particles left after ppp'%(np.sum(vardict_WQ.readsel.val)))
        else:
            print('All particles left after ppp')
        del Ls_temp
        ppv_selselect_coordsgen(centre, Ls, Axis1, Axis2, Axis3, periodic, vardict_WQ, velcut=velcut) # does the Axis3 selection
        if vardict_WQ.readsel.seldef:
            print('%i particles left after ppv'%(np.sum(vardict_WQ.readsel.val)))
        else:
            print('All particles left after ppv')
    else:
        box3 = (simfile.boxsize*simfile.h**-1,)*3
        vardict_WQ.add_box('box3',box3)

    # apply halo selection after the more read-in efficient region selection
    if halosel is not None:
        groupnums = selecthaloparticles(vardict_WQ, halosel, nameonly=False,
                                        last=False, **kwargs_halosel) 
    
    # excludeSFR handling: use np.logical_not on selection array
    # this is needed for all calculations, so might as well do it here
    # only remaining checks on excludeSFR are for 'T4' and 'from'

    if excludeSFRW in ['from','only']: # only select EOS particles; difference in only in the emission calculation
        vardict_WQ.readif('OnEquationOfState',rawunits =True)
        eossel = pc.Sel({'arr': vardict_WQ.particle['OnEquationOfState'] > 0.})
        vardict_WQ.delif('OnEquationOfState')
        vardict_WQ.update(eossel) #should significantly reduce memory impact of coordinate storage
        del eossel

    elif excludeSFRW == True: # only select non-EOS particles
        vardict_WQ.readif('OnEquationOfState',rawunits =True)
        eossel = pc.Sel({'arr': vardict_WQ.particle['OnEquationOfState'] <= 0.})
        vardict_WQ.delif('OnEquationOfState')
        vardict_WQ.update(eossel) #will have less impact on coordinate storage
        del eossel
    # False and T4 require no up-front or general particle selection, just one instance in the temperature read-in


    # calculate the quantities to project: save outside vardict (and no links in it) to prevent modification by the next calculation
    if ptypeQ is None:
        last = True
    else:
        last = False
    if ptypeW == 'basic':
        readbasic(vardict_WQ, quantityW, excludeSFRW, last=last)
        qW  = vardict_WQ.particle[quantityW]
        multipafterW = Nion_to_coldens(vardict_WQ, Ls, Axis1, Axis2, Axis3, npix_x, npix_y) *\
                       vardict_WQ.CGSconv[quantityW]
            
    elif ptypeW == 'coldens' and not iseltW:
        if ionW in ['h1ssh', 'hmolssh', 'hneutralssh'] and \
            not (sylviasshtables or bensgadget2tables):
            qW, multipafterW = Nion_calc_ssh(vardict_WQ, excludeSFRW, habW, 
                                             ionW, last=True, updatesel=True, 
                                             misc=misc)
        else:
            qW, multipafterW = Nion_calc(vardict_WQ, excludeSFRW, eltabW, 
                                         habW, ionW,
                                         sylviasshtables=sylviasshtables,
                                         bensgadget2tables=bensgadget2tables,
                                         last=last, updatesel=True, misc=misc,
                                         ps20tables=ps20tables, 
                                         ps20depletion=ps20depletion)
        multipafterW *= Nion_to_coldens(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                        npix_x, npix_y)
    elif ptypeW == 'coldens' and iseltW:
        qW, multipafterW = Nelt_calc(vardict_WQ, excludeSFRW, eltabW, ionW,
                                     last=last, updatesel=True,
                                     ps20tables=ps20tables, 
                                     ps20depletion=ps20depletion)
        multipafterW *= Nion_to_coldens(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                        npix_x, npix_y)

    elif ptypeW == 'emission' and excludeSFRW != 'from':
        qW, multipafterW = luminosity_calc(vardict_WQ, excludeSFRW, eltabW,
                                           habW, ionW, last=last, 
                                           updatesel=True,
                                           ps20tables=ps20tables, 
                                           ps20depletion=ps20depletion)
        multipafterW *= luminosity_to_Sb(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                         npix_x, npix_y, ionW,
                                         ps20tables=ps20tables)
    elif ptypeW == 'emission' and excludeSFRW == 'from':
        if ionW == 'halpha':
            qW, multipafterW = luminosity_calc_halpha_fromSFR(vardict_WQ,
                                                              excludeSFRW, 
                                                              last=last, 
                                                              updatesel=True)
        multipafterW *= luminosity_to_Sb(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                         npix_x, npix_y, ionW, 
                                         ps20tables=ps20tables)


    if ptypeQ == 'basic':
        readbasic(vardict_WQ, quantityQ, excludeSFRQ, last=True)
        qQ  = vardict_WQ.particle[quantityQ]
        multipafterQ = vardict_WQ.CGSconv[quantityQ]
        
    elif ptypeQ == 'coldens' and not iseltQ:
        if ionQ in ['h1ssh', 'hmolssh', 'hneutralssh'] and not (sylviasshtables or bensgadget2tables):
            qQ, multipafterQ = Nion_calc_ssh(vardict_WQ, excludeSFRQ, habQ, 
                                             ionQ, last=True, updatesel=False, 
                                             misc=misc)
        else:
            qQ, multipafterQ = Nion_calc(vardict_WQ, excludeSFRQ, eltabQ, 
                                         habQ, ionQ,
                                         sylviasshtables=sylviasshtables, 
                                         bensgadget2tables=bensgadget2tables,
                                         ps20tables=ps20tables, 
                                         ps20depletion=ps20depletion,
                                         last=True, updatesel=False, misc=misc)
        multipafterQ *= Nion_to_coldens(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                        npix_x, npix_y)
    elif ptypeQ == 'coldens' and iseltQ:
        qQ, multipafterQ = Nelt_calc(vardict_WQ, excludeSFRQ, eltabQ, ionQ,
                                     last=True, updatesel=False,
                                     ps20tables=ps20tables, 
                                     ps20depletion=ps20depletion)
        multipafterQ *= Nion_to_coldens(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                        npix_x, npix_y)

    elif ptypeQ == 'emission' and excludeSFRQ != 'from':
        qQ, multipafterQ = luminosity_calc(vardict_WQ, excludeSFRQ, eltabQ,
                                           habQ, ionQ, last=True, 
                                           updatesel=False,
                                           ps20tables=ps20tables, 
                                           ps20depletion=ps20depletion)
        multipafterQ *= luminosity_to_Sb(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                         npix_x, npix_y, ionW,
                                         ps20tables=ps20tables)
    elif ptypeQ == 'emission' and excludeSFRQ == 'from':
        if ionQ == 'halpha':
            qQ, multipafterQ = luminosity_calc_halpha_fromSFR(vardict_WQ,
                                                              excludeSFRQ, 
                                                              last=True, 
                                                              updatesel=False)
        multipafterQ *= luminosity_to_Sb(vardict_WQ, Ls, Axis1, Axis2, Axis3,
                                         npix_x, npix_y, ionW, 
                                         ps20tables=ps20tables)

    if velcut == False:
        vardict_WQ.readif('Coordinates',rawunits=True)
        vardict_WQ.add_part('coords_cMpc-vel', 
                            vardict_WQ.particle['Coordinates'] * simfile.h**-1)
        vardict_WQ.delif('Coordinates',last=True) # essentially, force delete
        translate(vardict_WQ.particle,'coords_cMpc-vel', 
                  vardict_WQ.box['centre'], vardict_WQ.box['box3'], periodic)

    NumPart = vardict_WQ.particle['coords_cMpc-vel'].shape[0]
    if parttype == '0':
        lsmooth = simfile.readarray('PartType%s/SmoothingLength'%parttype, 
                                    rawunits=True, 
                                    region=vardict_WQ.region)[vardict_WQ.readsel.val] * \
                  simfile.h**-1
        tree = False
    elif parttype == '1': # DM: has a physically reasonable smoothing length, but it is not in the output files
        lsmooth = np.zeros(NumPart)
        tree = True
    elif parttype == '4' or parttype=='5':
        lsmooth = 0.5 * np.ones(NumPart) * Ls[Axis1] / float(npix_x)
        tree = False
        
    # gets saved with the hdf5 option
    cosmopars = {'omegam': vardict_WQ.simfile.omegam,\
                 'omegalambda': vardict_WQ.simfile.omegalambda,\
                 'omegab': vardict_WQ.simfile.omegab,
                 'a': vardict_WQ.simfile.a,\
                 'z': vardict_WQ.simfile.z,\
                 'h': vardict_WQ.simfile.h,\
                 'boxsize': vardict_WQ.simfile.boxsize}

    # prevents largest particle values from overflowing float32 (extra factor 1e4 is a rough attempt to prevent overflow in the projection)
    try:
        maxlogW = np.log10(np.max(qW))
    except ValueError: # qW has length zero
        maxlogW = 0.
    overfW = (int(np.ceil(maxlogW))+4) // 38
    qW = qW*10**(-overfW*38)
    multipafterW *= 10**(overfW*38)
    if ptypeQ is not None:
        try:
            overfQ = int(np.ceil(np.log10(np.max(qQ)) + maxlogW)+4) // 38 - overfW
        except ValueError:
            overfQ = 0
        qQ = qQ*10**(-overfQ*38)
        multipafterQ *= 10**(overfQ*38)
    else:
        qQ = np.zeros(qW.shape,dtype=np.float32)

    if numslices is None:
        projdict = {'lsmooth': lsmooth, 
                    'coords': vardict_WQ.particle['coords_cMpc-vel'],
                    'qW': qW, 
                    'qQ': qQ}
        resultW, resultQ = project(NumPart, vardict_WQ.box['Ls'],
                                   Axis1, Axis2, Axis3, vardict_WQ.box['box3'],
                                   periodic, npix_x, npix_y, kernel, 
                                   projdict, tree, ompproj=ompproj)


        if log: # strongly recommended: log values should fit into float32 just fine, e.g. non-log cgs Mass overflows float32
            resultW = np.log10(resultW) + np.log10(multipafterW)
            if ptypeQ is not None:
                resultQ = np.log10(resultQ) + np.log10(multipafterQ)
        else:
            resultW *= multipafterW
            if ptypeQ is not None:
                resultQ *= multipafterQ
        if saveres:
            resW = resultW.astype(np.float32)
            try:
                minW = np.min(resW[np.isfinite(resW)])
            except ValueError: # nothing in the map at all
                minW = np.NaN
            maxW = np.max(resW)
            if not LsinMpc:
                centre_save = centre * simfile.h
            else:
                centre_save = centre
            if halosel is None:
                if hdf5:
                    #print('should be saving hdf5 file now')
                    savemap_hdf5(resfile, resW, minW, maxW,
                                 simnum, snapnum, centre_save, L_x, L_y, L_z, 
                                 npix_x, npix_y, 
                                 ptypeW,
                                 ionW, abundsW, quantityW,
                                 ionQ, abundsQ, quantityQ, ptypeQ,
                                 excludeSFRW, excludeSFRQ, parttype,
                                 theta, phi, psi,
                                 sylviasshtables, bensgadget2tables,
                                 ps20tables, ps20depletion,
                                 var, axis, log, velcut,
                                 periodic, kernel, saveres,
                                 simulation, LsinMpc, misc, ompproj, numslices,
                                 halosel, kwargs_halosel, cosmopars, 
                                 override_simdatapath, None)
                else:
                    np.savez(resfile, arr_0=resW, minfinite=minW, max=maxW)
            else:
                if hdf5:
                    #print('should be saving hdf5 file now')
                    savemap_hdf5(resfile, resW, minW, maxW,
                                 simnum, snapnum, centre_save, L_x, L_y, L_z, 
                                 npix_x, npix_y,
                                 ptypeW,
                                 ionW, abundsW, quantityW,
                                 ionQ, abundsQ, quantityQ, ptypeQ,
                                 excludeSFRW, excludeSFRQ, parttype,
                                 theta, phi, psi,
                                 sylviasshtables, bensgadget2tables,
                                 ps20tables, ps20depletion,
                                 var, axis, log, velcut,
                                 periodic, kernel, saveres,
                                 simulation, LsinMpc, misc, ompproj, numslices,
                                 halosel, kwargs_halosel, cosmopars, 
                                 override_simdatapath, groupnums)
                else:
                    np.savez(resfile, arr_0=resW, minfinite=minW, max=maxW,
                             groupnums=groupnums)
            del resW
            if ptypeQ is not None:
                resQ = resultQ.astype(np.float32)
                try:
                    minQ= np.min(resQ[np.isfinite(resQ)])
                except ValueError:
                    minQ = np.NaN
                maxQ = np.max(resQ)
                if halosel is None:
                    if hdf5:
                        savemap_hdf5(resfile2, resQ, minQ, maxQ,
                                     simnum, snapnum, centre, L_x, L_y, L_z, 
                                     npix_x, npix_y, 
                                     ptypeW,
                                     ionW, abundsW, quantityW,
                                     ionQ, abundsQ, quantityQ, ptypeQ,
                                     excludeSFRW, excludeSFRQ, parttype,
                                     theta, phi, psi,
                                     sylviasshtables, bensgadget2tables,
                                     ps20tables, ps20depletion,
                                     var, axis, log, velcut,
                                     periodic, kernel, saveres,
                                     simulation, LsinMpc, misc, ompproj, numslices,
                                     halosel, kwargs_halosel, cosmopars, 
                                     override_simdatapath, None)
                    else:
                        np.savez(resfile2, arr_0=resQ, minfinite=minQ, max=maxQ)
                else:
                    if hdf5:
                        savemap_hdf5(resfile2, resQ, minQ, maxQ,
                                     simnum, snapnum, centre, L_x, L_y, L_z, 
                                     npix_x, npix_y,
                                     ptypeW,
                                     ionW, abundsW, quantityW,
                                     ionQ, abundsQ, quantityQ, ptypeQ,
                                     excludeSFRW, excludeSFRQ, parttype,
                                     theta, phi, psi, 
                                     sylviasshtables, bensgadget2tables,
                                     ps20tables, ps20depletion,
                                     var, axis, log, velcut,
                                     periodic, kernel, saveres,
                                     simulation, LsinMpc, misc, ompproj, numslices,
                                     halosel, kwargs_halosel, cosmopars, 
                                     override_simdatapath, groupnums)
                    else:
                        np.savez(resfile2, arr_0=resQ, minfinite=minQ, max=maxQ, groupnums=groupnums)
                del resQ
            print('results saved to file')
    else: #numslices is not None
        resultW = []
        resultQ = []
        for sliceind in range(numslices):
            sliceind += 1 # 1-start
            Ls_temp = np.copy(vardict_WQ.box['Ls'])
            L_proj = Ls_temp[Axis3] / float(numslices)
            Ls_temp[Axis3] = L_proj

            if periodic:
                c_proj = 0.5 * vardict_WQ.box['box3'][Axis3]
            else:
                c_proj = 0.
            c_proj = c_proj - (numslices + 1.) * L_proj * 0.5 + sliceind * L_proj

            projmin = c_proj - L_proj / 2.
            projmax = c_proj + L_proj / 2.
            print('box, Ls overall: %s, %s'%(vardict_WQ.box['box3'], vardict_WQ.box['Ls']))
            print('projmin %s, projmax %s'%( projmin, projmax))

            # name output for single slice:
            if saveres:
                if not LsinMpc:
                    centre_temp = centre_cMpc * simfile.h
                else:
                    centre_temp = np.copy(centre_cMpc)
                if Axis3 == 0:
                    L_x_temp = L_x / float(numslices)
                    L_y_temp = L_y
                    L_z_temp = L_z
                    centre_temp[Axis3] = centre_temp[Axis3] - (numslices + 1.)*L_x_temp/2. + sliceind*L_x_temp
                elif Axis3 == 1:
                    L_x_temp = L_x
                    L_y_temp = L_y / float(numslices)
                    L_z_temp = L_z
                    centre_temp[Axis3] = centre_temp[Axis3] - (numslices + 1.)*L_y_temp/2. + sliceind*L_y_temp
                elif Axis3 == 2:
                    L_x_temp = L_x
                    L_y_temp = L_y
                    L_z_temp = L_z / float(numslices)
                    centre_temp[Axis3] = centre_temp[Axis3] - (numslices + 1.)*L_z_temp/2. + sliceind*L_z_temp

                subresfile = nameoutput(ptypeW, simnum, snapnum, version, kernel,
                         npix_x, L_x_temp, L_y_temp, L_z_temp, centre_temp, simfile.boxsize, simfile.h,
                         excludeSFRW, excludeSFRQ, velcut, sylviasshtables, bensgadget2tables,
                         ps20tables, ps20depletion,
                         axis, var, abundsW, ionW, parttype, None, abundsQ, ionQ, quantityW, quantityQ,
                         simulation, LsinMpc, misc)
                print('Saving W result to %s'%subresfile)
                if ptypeQ !=None:
                    subresfile2 = nameoutput(ptypeW, simnum, snapnum, version, kernel,
                              npix_x, L_x_temp, L_y_temp, L_z_temp, centre_temp, simfile.boxsize, simfile.h,
                              excludeSFRW, excludeSFRQ, velcut, sylviasshtables, bensgadget2tables,
                              ps20tables, ps20depletion,
                              axis, var, abundsW, ionW, parttype, ptypeQ, abundsQ, ionQ, quantityW, quantityQ,
                              simulation, LsinMpc, misc)
                    print('Saving Q result to %s'%subresfile2)

            projdict = {'lsmooth': lsmooth, 
                        'coords': vardict_WQ.particle['coords_cMpc-vel'],
                        'qW': qW, 
                        'qQ':qQ }
            subresultW, subresultQ = project(NumPart, Ls_temp, 
                                             Axis1, Axis2, Axis3, 
                                             vardict_WQ.box['box3'], periodic, 
                                             npix_x, npix_y, kernel, projdict, 
                                             tree, ompproj=ompproj, 
                                             projmin=projmin, projmax=projmax)

            if log: # strongly recommended: log values should fit into float32 just fine, e.g. non-log cgs Mass overflows float32
                subresultW = np.log10(subresultW) + np.log10(multipafterW)
                if ptypeQ is not None:
                    subresultQ = np.log10(subresultQ) + np.log10(multipafterQ)
            else:
                subresultW *= multipafterW
                if ptypeQ is not None:
                    subresultQ *= multipafterQ
            if saveres:
                resW = subresultW.astype(np.float32)
                try:
                    minW = np.min(resW[np.isfinite(resW)])
                except ValueError:
                    minW = np.NaN
                maxW = np.max(resW)
                if halosel is None:
                    if hdf5:
                        savemap_hdf5(subresfile, resW, minW, maxW,
                                     simnum, snapnum, centre_temp, 
                                     L_x_temp, L_y_temp, L_z_temp, 
                                     npix_x, npix_y, 
                                     ptypeW,
                                     ionW, abundsW, quantityW,
                                     ionQ, abundsQ, quantityQ, ptypeQ,
                                     excludeSFRW, excludeSFRQ, parttype,
                                     theta, phi, psi, 
                                     sylviasshtables, bensgadget2tables,
                                     ps20tables, ps20depletion,
                                     var, axis, log, velcut,
                                     periodic, kernel, saveres,
                                     simulation, LsinMpc, misc, ompproj, numslices,
                                     halosel, kwargs_halosel, cosmopars, 
                                     override_simdatapath, None)
                    else:
                        np.savez(subresfile, arr_0=resW, minfinite=minW, max=maxW)
                else:
                    if hdf5:
                        savemap_hdf5(subresfile, resW, minW, maxW,
                                     simnum, snapnum, centre_temp, 
                                     L_x_temp, L_y_temp, L_z_temp,  
                                     npix_x, npix_y, 
                                     ptypeW,
                                     ionW, abundsW, quantityW,
                                     ionQ, abundsQ, quantityQ, ptypeQ,
                                     excludeSFRW, excludeSFRQ, parttype,
                                     theta, phi, psi, 
                                     sylviasshtables, bensgadget2tables,
                                     ps20tables, ps20depletion,
                                     var, axis, log, velcut,
                                     periodic, kernel, saveres,
                                     simulation, LsinMpc, misc, ompproj, numslices,
                                     halosel, kwargs_halosel, cosmopars, 
                                     override_simdatapath, groupnums)
                    else:
                        np.savez(subresfile, arr_0=resW, minfinite=minW, 
                                 max=maxW, groupnums=groupnums)
                del resW
                if ptypeQ is not None:
                    resQ = subresultQ.astype(np.float32)
                    try:
                        minQ = np.min(resQ[np.isfinite(resQ)])
                    except ValueError:
                        minQ = np.NaN
                    maxQ = np.max(resQ)
                    if halosel is None:
                        if hdf5:
                            savemap_hdf5(subresfile2, resQ, minQ, maxQ,
                                         simnum, snapnum, centre_temp, 
                                         L_x_temp, L_y_temp, L_z_temp,  
                                         npix_x, npix_y, 
                                         ptypeW,
                                         ionW, abundsW, quantityW,
                                         ionQ, abundsQ, quantityQ, ptypeQ,
                                         excludeSFRW, excludeSFRQ, parttype,
                                         theta, phi, psi, 
                                         sylviasshtables, bensgadget2tables,
                                         ps20tables, ps20depletion,
                                         var, axis, log, velcut,
                                         periodic, kernel, saveres,
                                         simulation, LsinMpc, misc, ompproj, numslices,
                                         halosel, kwargs_halosel, cosmopars, 
                                         override_simdatapath, None)
                        else:
                            np.savez(subresfile2, arr_0=resQ, minfinite=minQ, max=maxQ)
                    else:
                        if hdf5:
                            savemap_hdf5(subresfile2, resQ, minQ, maxQ,
                                         simnum, snapnum, centre_temp, 
                                         L_x_temp, L_y_temp, L_z_temp,  
                                         npix_x, npix_y,
                                         ptypeW,
                                         ionW, abundsW, quantityW,
                                         ionQ, abundsQ, quantityQ, ptypeQ,
                                         excludeSFRW, excludeSFRQ, parttype,
                                         theta, phi, psi,
                                         sylviasshtables, bensgadget2tables,
                                         ps20tables, ps20depletion,
                                         var, axis, log, velcut,
                                         periodic, kernel, saveres,
                                         simulation, LsinMpc, misc, ompproj, numslices,
                                         halosel, kwargs_halosel, cosmopars, 
                                         override_simdatapath, groupnums)
                        else:
                            np.savez(subresfile2, arr_0=resQ, minfinite=minQ, 
                                     max=maxQ, groupnums=groupnums)
                    del resQ
                print('results saved to file')
            resultW = resultW + [subresultW]
            resultQ = resultQ + [subresultQ]

    return resultW, resultQ


######################################
###### per-particle histograms #######
######################################

def gethalomass(vardict, mdef='200c', allinR200c=True):
    '''
    kwargs:
        mdef
        allinR200c
    returns: vardict groups 'logMh_Msun' (-np.inf for no halo)
    
    '''
    if vardict.simfile.filetype != 'particles':
        raise ValueError('selecthaloparticles only works with particle files, not input filetype %s'%vardict.simfile.filetype)
    if vardict.simfile.simulation != 'eagle':
        raise ValueError('selecthaloparticles only works for eagle simulations, not input simulation %s'%vardict.simfile.simulation)
    # validity of mdef, aperture, halosel are checked in selecthalos.py (sh)
    if mdef == 'group':
        dsname = 'FOF/GroupMass'
    else:
        if not (mdef[-1] in ['c', 'm'] and mdef[:-1] in ['200', '500', '2500']):
            raise ValueError('Option %s for mdef is invalid'%mdef)
        if mdef[-1] == 'c':
            part1 = 'Crit'
        elif mdef[-1] == 'm':
            part1 = 'Mean'
        dsname = 'FOF/Group_M_%s%s'%(part1, mdef[:-1])
        
    if not isinstance(allinR200c, bool):
        raise ValueError('allinR200c should be a boolean')
        
    simfile_sf = pc.Simfile(vardict.simfile.simnum, vardict.simfile.snapnum, vardict.simfile.var, file_type='sub', simulation=vardict.simfile.simulation)
    # 2-step mass conversion because converting directly to cgs causes float32 to overflow
    masses_halo = simfile_sf.readarray(dsname, rawunits=True)
    mconv = simfile_sf.CGSconvtot / c.solar_mass
    masses_halo *= mconv
    
    vardict.readif('GroupNumber', rawunits=True)
    if allinR200c:
        vardict.particle['GroupNumber'] = np.abs(vardict.particle['GroupNumber']) # GroupNumber < 0: within R200 of halo |GroupNumber|, but not in FOF group
    vardict.particle['GroupNumber'] -= 1 # go from group labels (1-start) to indices (0-start)
    
    parentmasses = np.ones(len(vardict.particle['GroupNumber'])) * np.NaN
    #print(vardict.particle['GroupNumber'].dtype) -> int32, so max value allowed is 2**32
    outsideany = np.logical_or(vardict.particle['GroupNumber'] < 0, vardict.particle['GroupNumber'] == 2**30 - 1)
    parentmasses[outsideany] = 0.
    parentmasses[np.logical_not(outsideany)] = masses_halo[vardict.particle['GroupNumber'][np.logical_not(outsideany)]]
    
    vardict.delif('GroupNumber', last=True) # changed values -> remove
    vardict.add_part('halomass', parentmasses)
    vardict.CGSconv['halomass'] = c.solar_mass
    #print('Min/max halo masses: %s (%s) / %s'%(np.min(parentmasses), np.min(parentmasses[parentmasses > 0]), np.max(parentmasses)))
    
def getsubhaloclass(vardict):
    '''
    kwargs:
        mdef
        allinR200c
    returns: vardict groups 'logMh_Msun' (-np.inf for no halo)
    
    '''
    if vardict.simfile.filetype != 'particles':
        raise ValueError('selecthaloparticles only works with particle files, not input filetype %s'%vardict.simfile.filetype)
    if vardict.simfile.simulation != 'eagle':
        raise ValueError('selecthaloparticles only works for eagle simulations, not input simulation %s'%vardict.simfile.simulation)
    
    vardict.readif('SubGroupNumber', rawunits=True)
    vardict.particle['subhalocat'] = np.ones(len(vardict.particle['SubGroupNumber'])) * np.NaN
    vardict.particle['subhalocat'][vardict.particle['SubGroupNumber'] == 0] = 0.5
    vardict.particle['subhalocat'][vardict.particle['SubGroupNumber'] >  0] = 1.5
    vardict.particle['subhalocat'][vardict.particle['SubGroupNumber'] == 2**30] = 2.5
    vardict.delif('SubGroupNumber', last=True) 

def get3ddist(vardict, cen, last=True, trustcoords=False):
    '''
    trustcoords: trust 'Coordinates' entry in vardict to be in cMpc units and
                 CGSconv to reflect that
    '''
    if not trustcoords: 
        vardict.delif('Coordinates', last=True)
    if 'Coordinates' not in vardict.particle:
        vardict.readif('Coordinates', rawunits=True)
        vardict.particle['Coordinates'] *= (1. / vardict.simfile.h)
        vardict.CGSconv['Coordinates'] *= vardict.simfile.h
        
    if not np.all(cen == 0.): # translation step will often have been made before in region selection -> no need to repeat
        translate(vardict.particle, 'Coordinates', cen, np.array((vardict.simfile.boxsize / vardict.simfile.h,) *3), False) # non-periodic -> centered on cen
    radii = np.sqrt(np.sum(vardict.particle['Coordinates']**2, axis=1))
    vardict.add_part('r3D', radii)
    vardict.CGSconv['r3D'] = vardict.CGSconv['Coordinates']
    vardict.delif('Coordinates', last=last)


def namehistogram_perparticle(ptype, simnum, snapnum, var, simulation,
                              L_x, L_y, L_z, centre, LsinMpc, BoxSize, hconst,
                              excludeSFR,
                              abunds, ion, parttype, quantity,
                              sylviasshtables, bensgadget2tables,
                              ps20tables, ps20depletion,
                              misc):
    # some messiness is hard to avoid, but it's contained
    # Ls and centre have not been converted to Mpc when this function is called

    # box, if there is a box selection
    if not np.all([L_x is None, L_y is None, L_z is None, centre is None]):
        if LsinMpc:
            Lunit = ''
            hfac = 1.
        else:
            Lunit = '-hm1'
            hfac = hconst
        poss = []
        if L_x*hfac < BoxSize * hconst**-1:
            poss += ['x%s-pm%s%s'%(str(centre[0]), str(L_x), Lunit)]
        if L_y*hfac < BoxSize * hconst**-1:
            poss += ['y%s-pm%s%s'%(str(centre[1]), str(L_y), Lunit)]
        if L_z*hfac < BoxSize * hconst**-1:
            poss += ['z%s-pm%s%s'%(str(centre[2]), str(L_z), Lunit)]
        boxstring = '_'+'_'.join(poss)
    else:
        boxstring = ''

    # EOS particle handling
    if excludeSFR == True:
        SFRind = '_noEOS'
    elif excludeSFR == False:
        SFRind = '_wiEOS'
    elif excludeSFR == 'T4':
        SFRind = '_T4EOS'
    elif excludeSFR == 'from':
        SFRind = '_fromSFR'
    elif excludeSFR == 'only':
        SFRind = '_onlyEOS'

    # abundances
    if ptype in ['Nion', 'Niondens', 'Luminosity', 'Lumdens']:
        if abunds[0] not in ['Sm','Pt']:
            sabunds = '{}massfracAb'.format(str(abunds[0]))
        else:
            sabunds = abunds[0] + 'Ab'
        if isinstance(abunds[1], num.Number):
            sabunds = sabunds + '-{}massfracHAb'.format(str(abunds[1]))
        elif abunds[1] != abunds[0]:
            sabunds = sabunds + '-{}massfracHAb'.format(abunds[1])


    if var != 'REFERENCE':
        ssimnum = simnum + var
    else:
        ssimnum = simnum
    if simulation == 'bahamas':
        ssimnum = 'BA-{}'.format(ssimnum)
    if simulation == 'eagle-ioneq':
        ssimnum = 'EA-ioneq-{}'.format(simnum)

    if parttype != '0':
        sparttype = '_PartType{}'.quantity(parttype)
    else:
        sparttype = ''

    #avoid / in file names
    if ptype == 'basic':
        squantity = quantity
        squantity = squantity.replace('/','-')
    
    iontableind = ''
    if ptype in ['Nion', 'Niondens']:
        if sylviasshtables:
            iontableind = '_iontab-sylviasHM12shh'
        elif bensgadget2tables:
            iontableind = '_iontab-bensgagdet2'
    if ps20tables and ptype in ['Nion', 'Niondens', 'Luminosity', 'Lumdens']: 
        iontableind = '_iontab-PS20-'
        iontab = ol.iontab_sylvia_ssh.split('/')[-1]
        iontab = iontab[:-5] # remove '.hdf5'
        iontab = iontab.replace('_', '-')
        iontableind = iontableind + '-' + 'iontab'
        if ps20depletion:
            iontableind += '_depletion-T'
        else:
            iontableind += '_depletion-F'
        
    if ptype in ['Nion', 'Niondens', 'Luminosity', 'Lumdens']:
        base = 'particlehist_{ptype}_{ion}{parttype}{iontab}' + \
               '_{sim}_{snap}_test{ver}_{abunds}'
        base = base.format(ptype=ptype, ion=ion.replace(' ', '-'),
                           parttype=sparttype, iontab=iontableind, sim=simnum,
                           snap=snapnum, ver=str(version), abunds=sabunds)
        resfile = ol.ndir + base + boxstring + SFRind
    elif ptype == 'basic':
        base = 'particlehist_{qty}{parttype}_{sim}_{snap}_test{ver}'
        base = base.format(qty=squantity, parttype=sparttype, sim=simnum,
                           snap=snapnum, ver=str(version))
        resfile = ol.ndir + base + boxstring + SFRind
    elif ptype in ['halo', 'coords']:
        base = 'particlehist_{ptype}-{quantity}{parttype}' +\
               '_{sim}_{snap}_test{ver}'
        base = base.format(ptype=ptype, quantity=quantity, 
                           parttype=sparttype, sim=simnum, snap=snapnum,
                           ver=str(version))
        resfile = ol.ndir + base + boxstring + SFRind
        
    if misc is not None:
        miscind = '_'+'_'.join(['{}-{}'.format(key, misc[key]) \
                                for key in misc.keys()])
        resfile = resfile + miscind

    resfile = resfile + '.hdf5'
    print('saving result to: '+resfile+'\n')
    return resfile

def namehistogram_perparticle_axis(dct):
    '''
    dct should contain all the axesdct entires (with defaults included)
    '''
    ptype = dct['ptype']
    excludeSFR = dct['excludeSFR']
    if 'misc' in dct.keys():
        misc = dct['misc']
    else:
        misc = None

    if excludeSFR == True:
        SFRind = '_noEOS'
    elif excludeSFR == False:
        SFRind = '_wiEOS'
    elif excludeSFR == 'T4':
        SFRind = '_T4EOS'
    elif excludeSFR == 'from':
        SFRind = '_fromSFR'
    elif excludeSFR == 'only':
        SFRind = '_onlyEOS'

    if ptype in ['Luminosity', 'Lumdens', 'Nion', 'Niondens']:
        parttype = dct['parttype']
        if parttype != '0':
            sparttype = '_PartType%s'%parttype
        else:
            sparttype = ''
        abunds = dct['abunds']
        if abunds[0] not in ['Sm','Pt']:
            sabunds = '%smassfracAb'%str(abunds[0])
        else:
            sabunds = abunds[0] + 'Ab'
        if isinstance(abunds[1], num.Number):
            sabunds = sabunds + '-%smassfracHAb'%str(abunds[1])
        elif abunds[1] != abunds[0]:
            sabunds = sabunds + '-%smassfracHAb'%abunds[1]
        stables = ''
        if dct['sylviasshtables']:
            stables = '_iontab-sylviasHM12shh'
        elif dct['bensgadget2tables']:
            stables = '_iontab-bensgagdet2'
        elif dct['ps20tables']: 
            iontableind = '_PS20-iontab'
            iontab = ol.iontab_sylvia_ssh.split('/')[-1]
            iontab = iontab[:-5] # remove '.hdf5'
            iontab = iontab.replace('_', '-')
            #iontableind = iontableind + '-' + 'iontab'
            if dct['ps20depletion']:
                iontableind += '_depletion-T'
            else:
                iontableind += '_depletion-F'
            stables = iontableind
        axname = '%s_%s%s_%s%s' %(ptype, dct['ion'], sparttype, sabunds, 
                                  stables) +\
                 SFRind

    elif ptype == 'basic':
        parttype = dct['parttype']
        if parttype != '0':
            sparttype = '_PartType%s'%parttype
        else:
            sparttype = ''
        squantity = dct['quantity']
        squantity = squantity.replace('/','-')
        axname = '%s%s'%(squantity, sparttype) + SFRind

    elif ptype == 'halo':
        if dct['quantity'] == 'Mass':
            if dct['allinR200c']: 
                inclind = '_allinR200c'
            else:
                inclind = '_FoFonly'
            axname = 'M%s_halo'%(dct['mdef']) + inclind
        elif dct['quantity'] == 'subcat':
            axname = 'subhalo_category'
            
    elif ptype == 'coords':
        if dct['quantity'] == 'r3D':
            axname = '3Dradius'

    if misc is not None:
        miscind = '_'+'_'.join(['%s-%s'%(key, misc[key]) for key in misc.keys()])
        axname = axname + miscind

    return axname

def check_particlequantity(dct, dct_defaults, parttype, simulation):
    '''
    dct: ptype, excludeSFR, abunds, ion, parttype, quantity, misc
    dct_defaults: same entries, use to set defaults in dct
    '''
    # largest int used : 47
    if 'ptype' in dct:
        ptype = dct['ptype']
    else:
        raise ValueError('in check_particlequantity: each quantity dict must have "ptype" specified')
    if 'excludeSFR' in dct:
        excludeSFR = dct['excludeSFR']
    elif 'excludeSFR' in dct_defaults:
        excludeSFR = dct_defaults['excludeSFR']
        dct['excludeSFR'] = excludeSFR
    
    if ptype not in ['Nion', 'Niondens', 'Luminosity', 'Lumdens', 'basic', 'halo', 'coords']:
        print('ptype should be one of Nion, Niondens, Luminosity, Lumdens, basic, halo, coords (str).\n')
        return 3
    elif ptype in ['Nion', 'Niondens', 'Luminosity', 'Lumdens']:
        if 'ion' not in dct.keys():
            print('For ptype %s, an ion must be specified'%(ptype))
            return 37
        else:
            ion = dct['ion']
        if 'abunds' in dct.keys():
            abunds = dct['abunds']
        elif 'abunds' in dct_defaults.keys():
            abunds = dct_defaults['abunds']
        else:
            abunds = None
        if ion in ol.elements_ion.keys():
            iselt = False
            parttype = '0'
        elif ion in ol.elements and ptype in ['Nion', 'Niondens']:
            iselt = True
            if parttype not in ['0', '4', 0, 4]:
                print('Element masses are only available for gas and stars')
                return 47
            else:
                parttype = str(parttype)
        else:
            print('%s is an invalid ion option for ptype %s\n'%(ion,ptype))
            return 8
        if not isinstance(abunds, (list, tuple, np.ndarray)):
            abunds = [abunds, 'auto']
        else:
            abunds = list(abunds) # tuple element assigment is not allowed, sometimes needed
        if abunds[0] not in ['Sm','Pt','auto']:
            if not isinstance(abunds[0], num.Number):
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 4
            elif iselt:
                abunds[0] = abunds[0] * ol.solar_abunds_ea[ion]
            else:
                abunds[0] = abunds[0] * ol.solar_abunds_ea[ol.elements_ion[ion]]
        elif abunds[0] == 'auto':
            if ptype in ['Luminosity', 'Lumdens']:
                abunds[0] = 'Sm'
            else:
                abunds[0] = 'Pt'
        if abunds[1] not in ['Sm','Pt','auto']:
            if not isinstance(abunds[1], num.Number):
                print('Abundances must be either smoothed ("Sm") or particle ("Pt") abundances, automatic ("auto"), or a solar units abundance (float)')
                return 30
        elif abunds[1] == 'auto':
            if isinstance(abunds[0], num.Number):
                abunds[1] = 0.752 # if element abundance is fixed, use primordial hydrogen abundance
            else:
                abunds[1] = abunds[0]
        dct['abunds'] = tuple(abunds)
        abunds = tuple(abunds)
        
        
    else: # ptype == basic or halo
        if 'quantity' not in dct.keys():
            print('For ptypes basic, halo, coords, quantity must be specified.\n')
            return 5
        quantity = dct['quantity']
        if not isinstance(quantity, str):
            print('quantity must be a string.\n')
            return 6
        if ptype == 'halo':
            if quantity not in ['Mass', 'subcat']:
                print('For ptype halo, the options are Mass and subcat')
                return 38
        elif ptype == 'coords':
            if quantity not in ['r3D']:
                print('For ptype coords, the option is r3D')
                return 41
        if parttype not in ['0','1','4','5']: # parttype only matters if it is used
            if parttype in [0, 1, 4, 5]:
                parttype = str(parttype)
            else:
                print('parttype should be "0", "1", "4", or "5" (str).\n')
                return 16


    if excludeSFR not in [True, False, 'T4', 'only']:
        if excludeSFR != 'from':
            print('Invalid option for excludeSFR: %s'%excludeSFR)
            return 17
        elif not (ptype in ['Luminosity', 'Lumdens'] and ion == 'halpha'):
            excludeSFR = 'only'
            print('Unless calculation is for halpha emission, fromSFR will default to onlySFR.\n')
    if 'excludeSFR' in dct_defaults.keys():
        excludeSFR_def = dct_defaults['excludeSFR']
        if (excludeSFR in [False,' T4']) and (excludeSFR_def not in [False, 'T4']):
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFR, excludeSFR_def))
            return 18
        elif excludeSFR in ['from', 'only'] and excludeSFR_def not in ['from', 'only']:
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFR,excludeSFR_def))
            return 19
        elif excludeSFR != excludeSFR_def and excludeSFR == True:
            print('ExcludeSFR options %s and %s are not compatible'%(excludeSFR, excludeSFR_def))
            return 20

    if parttype != '0': #EOS is only relevant for parttype 0 (gas)
        excludeSFR = False
    dct['excludeSFR'] = excludeSFR

    if 'misc' in dct.keys(): # if if if : if we want to use chemical abundances from Ben' Oppenheimer's recal variations
        misc = dct['misc']
        if misc is not None:
            if 'usechemabundtables' in misc:
                if misc['usechemabundtables'] == 'BenOpp1':
                    if simulation != 'eagle-ioneq':
                        print('chemical abundance tables are only avaiable for the eagle-ioneq simulation')
                        return 34
                    if ptype in ['Nion', 'Niondens']:
                        if 'Sm' in abunds:
                            print('chemical abundance tables are only for particle abundances')
                            return 34
                        elif abunds in ['auto', None]:
                            abunds = 'Pt'
            if ptype in ['Nion', 'Niondens'] and ion in ['h1ssh', 'hmolssh' 'hneutralssh']:
                if 'UVB' in misc:
                    if misc['UVB'] not in cfh.phototables.keys():
                        print('Invalid option for misc -> UVB')
                        return 35
                if 'useLSR' in misc:
                    if not isinstance(misc['useLSR'], bool):
                        print('misc -> useLSR should be a boolean')
                        return 36
                    
    if 'mdef' not in dct:
        dct['mdef'] = dct_defaults['mdef']
    if 'allinR200c' not in dct:
        dct['allinR200c'] = dct_defaults['allinR200c']
        
    if not (dct['mdef'] == 'group' or (dct['mdef'][-1] in ['c', 'm'] and dct['mdef'][:-1] in ['200', '500', '2500'])):
        print('mdef option %s is invalid'%(dct['mdef']))
        return 39
        
    if not isinstance(dct['allinR200c'], bool):
        print('allinR200c should be True or False')
        return 40
    # table set checks
    if ptype in ['Nion', 'Niondens']:
        if 'sylviasshtables' in dct.keys():
            sylviasshtables = dct['sylviasshtables']
        else:
            sylviasshtables = dct_defaults['sylviasshtables']
        if 'bensgadget2tables' in dct.keys():
            bensgadget2tables = dct['bensgadget2tables']
        else:
            bensgadget2tables = dct_defaults['bensgadget2tables']
        if 'ps20tables' in dct.keys():
            ps20tables = dct['ps20tables']
        else:
            ps20tables = dct_defaults['ps20tables']
        if 'ps20depletion' in dct.keys():
            ps20depletion = dct['ps20depletion']
        else:
            ps20depletion = dct_defaults['ps20depletion']
        if not isinstance(sylviasshtables, bool):
            print('sylviasshtables should be True or False')
            return 42
        if not isinstance(bensgadget2tables, bool):
            print('bensgadget2tables should be True or False')
            return 43
        if not isinstance(ps20tables, bool):
            print('ps20tables should be True or False')
            return 47
        if not isinstance(ps20depletion, bool):
            print('ps20depletion should be True or False')
            return 48
        if sylviasshtables + bensgadget2tables + ps20tables > 1:
            print('only one table set of sylviasshtables and bensgadget2tables can be used')
            return 44
        if sylviasshtables and ion == 'hneutralssh':
            print("Neutral hydrogen is not currenty available from Sylvia's tables")
            return 45
        if bensgadget2tables and ion not in ol.ion_list_bensgadget2tables:
            print("%s is not available from Ben's gadget 2 tables"%(ion))
            return 46
        dct['sylviasshtables'] = sylviasshtables
        dct['bensgadget2tables'] = bensgadget2tables
        dct['ps20tables'] = ps20tables
        dct['ps20depletion'] = ps20depletion
    elif ptype in ['Luminosity', 'Lumdens']:
        dct['sylviasshtables'] = False
        dct['bensgadget2tables'] = False
        if 'ps20tables' in dct.keys():
            ps20tables = dct['ps20tables']
        else:
            ps20tables = dct_defaults['ps20tables']
        if 'ps20depletion' in dct.keys():
            ps20depletion = dct['ps20depletion']
        else:
            ps20depletion = dct_defaults['ps20depletion']
        if not isinstance(ps20tables, bool):
            print('ps20tables should be True or False')
            return 47
        if not isinstance(ps20depletion, bool):
            print('ps20depletion should be True or False')
            return 48
    else:
        dct['sylviasshtables'] = False
        dct['bensgadget2tables'] = False
        dct['ps20tables'] = False
        dct['ps20depletion'] = False
                
    dct['parttype'] = parttype
    
    # make sure there is something to check for dct keys (might be None or useless)
    for key in dct_defaults.keys():
        if key not in dct.keys():
            dct[key] = dct_defaults[key]
    return dct, parttype

def inputcheck_particlehist(ptype, simnum, snapnum, var, simulation,
                              L_x, L_y, L_z, centre, LsinMpc,
                              excludeSFR, abunds, ion, parttype, quantity,
                              axesdct, axbins, allinR200c, mdef,
                              sylviasshtables, bensgadget2tables, 
                              ps20tables, ps20depletion,
                              misc):

    '''
    Checks the input to make_map();
    This is not an exhaustive check; it does handle the default/auto options
    return numbers are not ordered; just search <return ##>
    '''
    # max used number: 53

    # basic type and valid option checks
    if not isinstance(var, str):
        print('%s should be a string.\n'%var)
        return 1
    if not isinstance(snapnum, int):
        print('snapnum should be an integer.\n')
        return 21
    if not isinstance(simnum, str):
        print('simnum should be a string')
        return 22
    
    if np.any([dct['ptype'] == 'coords' for dct in axesdct]) and centre is None:
        print('For ptype coords, a centre must be specified')
        return 40
    
    if not isinstance(sylviasshtables, bool):
        print('syvliasshtables should be True or False')
        return 41
    elif sylviasshtables and not np.any([ptype in ['Nion', 'Niondens']] + [_dct['ptype'] in ['Nion', 'Niondens'] for _dct in axesdct]):
        print('Warning: the option sylviasshtables only applies to ion numbers or densities; it will be ignored altogether here')
        return 42
    if not isinstance(bensgadget2tables, bool):
        print('bensgadget2tables should be True or False')
        return 43
    elif bensgadget2tables and not np.any([ptype in ['Nion', 'Niondens']] + [_dct['ptype'] in ['Nion', 'Niondens'] for _dct in axesdct]):
        print('Warning: the option bensgadget2tables only applies to ion numbers or densities; it will be ignored altogether here')
        return 44
    if not isinstance(ps20tables, bool):
        print('ps20tables should be True or False')
        return 46
    if ps20tables and ptype in ['Luminosity', 'Lumdens', 'Nion', 'Niondens']:
        if not os.path.isfile(ol.iontab_sylvia_ssh):
            print('PS20 table {} was not found'.format(ol.iontab_sylvia_ssh))
            return 52
    if ps20tables and ptype in ['Luminosity', 'Lumdens']:
        if not os.path.isfile(ol.emtab_sylvia_ssh):
            print('PS20 emission table {} was not found'.format(ol.emtab_sylvia_ssh))
            return 53
        iontab = ol.iontab_sylvia_ssh.split('/')[-1]
        iontab = iontab[:-5]
        emtab = ol.emtab_sylvia_ssh.split('/')[-1]
        emtab = emtab[:-5]
        if emtab != iontab + '_lines':
            print('PS20 emission and absorption tables do not match:')
            print(ol.emtab_sylvia_ssh)
            print(ol.iontab_sylvia_ssh)
            return 51
    if not isinstance(ps20depletion, bool):
        print('ps20depletion should be True or False')
        return 47
    if bensgadget2tables + sylviasshtables + ps20tables > 1:
        print('Only one table set of bensgadget2tables, ps20tables and sylviasshtables can be used')
        return 45    
    
    if not (L_x is None and L_y is None and L_z is None and centre is None):
        if (not isinstance(centre[0], num.Number)) or (not isinstance(centre[1], num.Number)) or (not isinstance(centre[2], num.Number)):
            print('centre should contain 3 floats')
            return 29
        centre = [float(centre[0]),float(centre[1]),float(centre[2])]
        if (not isinstance(L_x, num.Number)) or (not isinstance(L_y, num.Number)) or (not isinstance(L_z, num.Number)):
            print('L_x, L_y, and L_z should be floats')
            return 24
        L_x, L_y, L_z = (float(L_x),float(L_y),float(L_z))

    if simulation not in ['eagle', 'bahamas', 'Eagle', 'Bahamas', 'EAGLE', 'BAHAMAS', 'eagle-ioneq']:
        print('Simulation %s is not a valid choice; should be "eagle", "eagle-ioneq" or "bahamas"'%str(simulation))
        return 30
    elif simulation == 'Eagle' or simulation == 'EAGLE':
        simulation = 'eagle'
        print('Preferred form of simulation names is all lowercase (%s)'%simulation)
    elif simulation == 'Bahamas' or simulation == 'BAHAMAS':
        simulation = 'bahamas'
        print('Preferred form of simulation names is all lowercase (%s)'%simulation)

    if LsinMpc not in [True, False, None]:
        print('%s is an invalid option for LsinMpc')
        return 39
    if (simulation == 'eagle' or simulation == 'eagle-ioneq') and LsinMpc is None:
        LsinMpc = True
    elif simulation == 'bahamas' and LsinMpc is None:
        LsinMpc = False

    if simulation == 'eagle' and (len(simnum) != 10 or simnum[0] != 'L' or simnum[5] != 'N'):
        print('incorrect simnum format %s; should be L####N#### for eagle\n'%simnum)
        return 23
    elif simulation == 'bahamas' and (simnum[0] != 'L' or simnum[4] != 'N'):
        print('incorrect simnum format %s; should be L*N* for bahamas\n'%simnum)
        return 31
    elif simulation == 'eagle-ioneq' and simnum != 'L0025N0752':
        print('For eagle-ioneq, only L0025N0752 is avilable')
        return 33

    # combination-dependent checks
    if var == 'auto':
        if simnum == 'L0025N0752' and simulation != 'eagle-ioneq':
            var = 'RECALIBRATED'
        else:
            var = 'REFERENCE'

    dct_defaults = {'ptype': ptype, 'excludeSFR': excludeSFR, 
                    'abunds': abunds,
                    'ion': ion, 'parttype': parttype, 'quantity': quantity,
                    'misc': misc, 'allinR200c': allinR200c, 'mdef': mdef,
                    'sylviasshtables': sylviasshtables, 
                    'bensgadget2tables': bensgadget2tables,
                    'ps20tables': ps20tables, 'ps20depletion': ps20depletion}
    dct_defaults, parttype = check_particlequantity(dct_defaults, {},
                                                    parttype, simulation)
    axesdct = [check_particlequantity(dct, dct_defaults, parttype, 
                                      simulation)[0] 
               for dct in axesdct]
    if np.any(np.array([isinstance(dct, int) for dct in axesdct])):
        print('Error in one of the axis particle properties')
        return 38


    # if nothing has gone wrong, return all input, since setting quantities in functions doesn't work on global variables
    return (0, dct_defaults['ptype'], simnum, snapnum, var, simulation,
            L_x, L_y, L_z, centre, LsinMpc, dct_defaults['excludeSFR'], 
            dct_defaults['abunds'], dct_defaults['ion'], 
            dct_defaults['parttype'], dct_defaults['quantity'],
            axesdct, axbins, dct_defaults['allinR200c'], dct_defaults['mdef'],
            dct_defaults['sylviasshtables'], dct_defaults['bensgadget2tables'],
            dct_defaults['ps20tables'],  dct_defaults['ps20depletion'], misc)



# TODO: ps20tables=False, ps20depletion=True,
def getparticledata(vardict, ptype, excludeSFR, abunds, ion, quantity,
                    sylviasshtables=False, bensgadget2tables=False,
                    ps20tables=False, ps20depletion=True,
                    last=True, updatesel=False, misc=None, mdef='200c', 
                    allinR200c=True):
    '''
    just copied bits from make_map
    '''
            
    iselt = False
    if ion in ol.elements and ptype in ['Nion', 'Niondens']:
        iselt = True
    if ptype in ['Nion', 'Niondens', 'Luminosity', 'Lumdens']:
        eltab, hab = get_eltab_names(abunds, iselt, ion)

    if excludeSFR in ['from', 'only']: # only select EOS particles; difference in only in the emission calculation
        vardict.readif('OnEquationOfState', rawunits=True)
        eossel = pc.Sel({'arr': vardict.particle['OnEquationOfState'] > 0.})
        vardict.delif('OnEquationOfState')
        vardict.update(eossel) #should significantly reduce memory impact of coordinate storage
        del eossel

    elif excludeSFR == True: # only select non-EOS particles
        vardict.readif('OnEquationOfState',rawunits =True)
        eossel = pc.Sel({'arr': vardict.particle['OnEquationOfState'] <= 0.})
        vardict.delif('OnEquationOfState')
        vardict.update(eossel) #will have less impact on coordinate storage
        del eossel
    # False and T4 require no up-front or general particle selection, just one instance in the temperature read-in

    last = last
    if ptype == 'basic':
        readbasic(vardict, quantity, excludeSFR, last=last)
        q = vardict.particle[quantity]
        multipafter =  vardict.CGSconv[quantity]
    
    elif ptype == 'halo':
        if quantity == 'Mass':
            gethalomass(vardict, mdef=mdef, allinR200c=allinR200c)
            q = vardict.particle['halomass']
            multipafter = vardict.CGSconv['halomass']
            vardict.delif('halomass', last=last)
        elif quantity == 'subcat':
            getsubhaloclass(vardict)
            q = vardict.particle['subhalocat']
            vardict.delif('subhalocat', last=last)
            multipafter = 1.

    elif ptype in ['Nion', 'Niondens'] and not iselt:
        if ion in ['h1ssh', 'hmolssh', 'hneutralssh'] and \
           not (sylviasshtables or bensgadget2tables):
            q, multipafter = Nion_calc_ssh(vardict, excludeSFR, hab, ion, 
                                           last=last, updatesel=updatesel, 
                                           misc=misc)
            if ptype == 'Niondens':
                readbasic(vardict, 'ipropvol', excludeSFR, last=last)
                q *= vardict.particle['ipropvol'] 
                multipafter *= vardict.CGSconv['ipropvol']
        else:
            q, multipafter = Nion_calc(vardict, excludeSFR, eltab, hab, ion, 
                                       last=last,
                                       sylviasshtables=sylviasshtables, 
                                       bensgadget2tables=bensgadget2tables,
                                       ps20tables=ps20tables, 
                                       ps20depletion=ps20depletion,
                                       updatesel=updatesel, misc=misc)
            if ptype == 'Niondens':
                readbasic(vardict, 'ipropvol', excludeSFR, last=last)
                q *= vardict.particle['ipropvol'] 
                multipafter *= vardict.CGSconv['ipropvol']
    elif ptype in ['Nion', 'Niondens'] and iselt:
        q, multipafter = Nelt_calc(vardict, excludeSFR, eltab, ion, last=last,
                                   updatesel=updatesel, ps20tables=ps20tables, 
                                   ps20depletion=ps20depletion,)
        if ptype == 'Niondens':
            readbasic(vardict, 'ipropvol', excludeSFR, last=last)
            q *= vardict.particle['ipropvol'] 
            multipafter *= vardict.CGSconv['ipropvol']
    elif ptype in ['Luminosity', 'Lumdens'] and excludeSFR != 'from':
        q, multipafter = luminosity_calc(vardict, excludeSFR, eltab, hab, ion, 
                                         last=last, updatesel=updatesel, 
                                         ps20tables=ps20tables, 
                                         ps20depletion=ps20depletion)
        if ptype == 'Lumdens':
            readbasic(vardict, 'ipropvol', excludeSFR, last=last)
            q *= vardict.particle['ipropvol'] 
            multipafter *= vardict.CGSconv['ipropvol']
    elif ptype in ['Luminosity', 'Lumdens'] and excludeSFR == 'from':
        if ion == 'halpha':
            q, multipafter = luminosity_calc_halpha_fromSFR(vardict, 
                                                            excludeSFR, 
                                                            last=last, 
                                                            updatesel=updatesel)
            if ptype == 'Lumdens':
                readbasic(vardict, 'ipropvol', excludeSFR, last=last)
                q *= vardict.particle['ipropvol'] 
                multipafter *= vardict.CGSconv['ipropvol']
        else:
            raise ValueError('Invalid option excludeSFR=from for ion other than halpha')
    
    elif ptype == 'coords':
        if quantity == 'r3D':
            # coordinates should have been centred in region selection, in cMpc units
            get3ddist(vardict, np.array([0., 0., 0.]), last=last, 
                      trustcoords=True)
            q = vardict.particle['r3D']
            multipafter = vardict.CGSconv['r3D']
        else:
            raise ValueError('Invalid quantity option %s for ptype %s'%(quantity, ptype))
    else:
        raise ValueError('Invalid ptype option %s'%(ptype))
        return None
    
    if 'ipropvol' in vardict.particle.keys():
        vardict.delif('ipropvol', last=last)
        
    return q, multipafter



def makehistograms_perparticle(ptype, simnum, snapnum, var, _axesdct,
                               simulation='eagle',
                               excludeSFR=False, abunds=None, ion=None, 
                               parttype='0', quantity=None, axbins=0.2,
                               sylviasshtables=False, bensgadget2tables=False,
                               ps20tables=False, ps20depletion=True,
                               allinR200c=True, mdef='200c',\
                               L_x=None, L_y=None, L_z=None, centre=None, 
                               Ls_in_Mpc=True,
                               misc=None,
                               name_append=None, logax=True, loghist=False,
                               nameonly=False):
    '''
    only does a few very specific caluclations in current implementation

    arguments same as make_map:
    ---------------------------
    simnum
    snapnum
    var
    simulation
    excludeSFR
    abunds
    ion
    parttype
    quantity
    sylviasshtables DEPRECATED (only available as a choice to apply to all 
                                weights/axes)
    ps20tables (only available as a choice to apply to all weights/axes)
    ps20depletion (only available as a choice to apply to all weights/axes)
    bensgadget2tables (only available as a choice to apply to all weights/axes)
    misc
    L_x, L_Y, L_z, centre, Ls_in_Mpc: not currently implemented beyond input
        check and autonaming
        
    argument similar to make_map:
    -----------------------------
    ptype, allinR200c, mdef
        but:
         - pytpe 'halo' is an option, 
           with 'Mass' and 'subcat' quantities  
           'Mass' group particles by parent halo mass (no halo -> halo mass 0.)
           'subcat' divides particles into central, satellite, and unbound 
           classes
         - instead of 'coldens' and 'emission', 
           the ion/emission types are 'Nion' and 'Luminosity' (for total number
           of ions or luminosity of the particle, to be used as weights)
           or 'Niondens' and 'Lumdens' (for the volume density of emission or 
           ions, to be used on axes to gauge contributions to surface 
           brightness and column density)
         - ptype 'coords' is an option,
           with quantity 'r3D': the 3D radial distance to centre (must be 
           specified)
           
    not in make_map:
    ----------------
    _axesdct: list of dictionaries for each hist: the axes of the histogram
        entires are (the non-None elements of) ptype, exlcudeSFR, abunds, ion, 
        parttype, quantity, mdef, allinR200c, (defaults are same as general/
        weight values)
        note that bins are always (log) cgs, including for e.g. halo mass
        to get nice bin values in other units (e.g. solar masses, virial radii)
        specify the bins based on those values in cgs units
    logax: boolean, or array of booleans matching axesdct
           take log values of a property for the histogram (bins are assumed 
           to apply to the selected value type, and are not transformed based
           on logax)
    loghist: store log histogram values or not (bool)
    axbins: bins to use along each axis
            int -> number of bins, float -> bin size; applied to all (finite) 
            values between data min and max, with bin edges placed so 0. would 
            be an edge if included in the data range in the float case
            list/array: use those bin edges
            if a single value is given, that is used for the bin size/number
            along each axis. An iterable is interpreted single values for each
            axis, so to use the same specified in edges for each axis, use
            [[<value 1>, ..., <value N>]] * <number of dimensions>
    name_append: append this name to the group in the hdf5 file. Useful if 
            redoing a histogram with e.g. different binning, since autonaming
            only accounts for the properties of the weight and axis data.
    nameonly: return file name, group name tuple
    
    
    Note: if halo properties are needed, the histogramming will be done with
    the particle data files, which excludes particles not included in any halo
    
    TODO: wishlisting implementation: avoid double read-ins (currently only
    indirectly done for coords-r3D and region selection)
    '''
    axesdct = [_dct.copy() for _dct in _axesdct]
    
    res = inputcheck_particlehist(ptype, simnum, snapnum, var, simulation,
                              L_x, L_y, L_z, centre, Ls_in_Mpc,
                              excludeSFR, abunds, ion, parttype, quantity,
                              axesdct, axbins, allinR200c, mdef,
                              sylviasshtables, bensgadget2tables,
                              ps20tables, ps20depletion,
                              misc)
    if isinstance(res, int):
        print('Input error %i'%res)
        raise ValueError('inputcheck returned code %i'%res)
    if hasattr(logax, '__len__'):
        logax = np.array(logax)
        if not np.all([isinstance(val, (bool, np.bool_)) for val in logax]):
            raise ValueError('All logax values must be True or False')
    elif not isinstance(logax, bool)  :
        raise ValueError('logax should be True or False, or a list of booleans matching axesdct')
    else:
        logax = np.array([logax] * len(axesdct))
    if not isinstance(loghist, bool):
        raise ValueError('loghist should be True or False')

    pytpe, simnum, snapnum, var, simulation,\
    L_x, L_y, L_z, centre, LsinMpc,\
    exlcudeSFR, abunds, ion, parttype, quantity,\
    axesdct, axbins, allinR200c, mdef,\
    sylviasshtables, bensgadget2tables,\
    ps20tables, ps20depletion,\
    misc = res[1:]

    print('Processed input for makehstograms_perparticle:')
    print('general:')
    print('parttype: \t%s \tsimnum: \t%s snapnum: \t%s \tvar: \t%s \tsimulation: \t%s'%(parttype, simnum, snapnum, var, simulation))
    print('L_x: \t%s \tL_y: \t%s \tL_z: \t%s \tcentre: \t%s \tLs_in_Mpc: \t%s'%(L_x, L_y, L_z, centre, Ls_in_Mpc))
    print('loghist: \t%s \tnameonly: \t%s \tname_append: \t%s'%(loghist, nameonly, name_append))
    fillstr_particleprop = 'ptype: \t%s \texcludeSFR: \t%s \tabunds: \t%s '+\
        '\tion: \t%s \tquantity: \t%s\n\tsylviasshtables: \t%s '+\
        '\tbensgadget2tables: \t%s \tps20tables: \t%s \tps20depletion: \t%s,'+\
        '\tallinR200c: \t%s\tmdef: \t%s'
    print('histogram weight:')
    print(fillstr_particleprop%(ptype, excludeSFR, abunds, ion, quantity,
                                sylviasshtables, bensgadget2tables, 
                                ps20tables, ps20depletion, allinR200c, mdef))
    print('misc: %s'%(misc))
    print('histogram axes:')
    for axi in range(len(axesdct)):
        dct_temp = axesdct[axi]
        print('axis %i'%axi)
        print(fillstr_particleprop%(dct_temp['ptype'], dct_temp['excludeSFR'],
                                    dct_temp['abunds'], dct_temp['ion'],
                                    dct_temp['quantity'], dct_temp['sylviasshtables'],
                                    dct_temp['bensgadget2tables'], 
                                    dct_temp['ps20tables'], dct_temp['ps20depletion'],
                                    dct_temp['allinR200c'],
                                    dct_temp['mdef']))
        print('\taxbin: \t%s \tlogax: \t%s'%(axbins[axi] if hasattr(axbins, '__getitem__') else axbins, logax[axi]))
    
    useparticledata = ptype == 'halo'
    useparticledata = useparticledata or np.any([dct_sub['ptype'] == 'halo' \
                                                 for dct_sub in axesdct])
    
    if useparticledata:
        simfile = pc.Simfile(simnum, snapnum, var, file_type='particles', 
                             simulation=simulation)
        print('Using particle data')
    else:
        simfile = pc.Simfile(simnum, snapnum, var, file_type='snap', 
                             simulation=simulation)
        print('Using snapshot data')
    vardict = pc.Vardict(simfile, parttype, [], region=None, readsel=None) # important: vardict.region is set later, so don't read in anything before that
    
    # apply region selection if needed; don't use the make_maps methods here, 
    # since those account for smoothing length margins that are unnecessary for
    # this 
    if not (L_x is None and L_y is None and L_z is None and centre is None):
        Ls = np.array([L_x, L_y, L_z])
        if not Ls_in_Mpc:
            Ls_hunits = np.copy(Ls)
            Ls /= vardict.simfile.h
        else:
            Ls_hunits = np.copy(Ls)
            Ls_hunits *= vardict.simfile.h
        if vardict.simfile.region_supported:
            # select region; units for this are cMpc/h
            if np.all(Ls_hunits >= vardict.simfile.boxsize):
                region = None
            else:
                hconst = vardict.simfile.h
                # selectio is periodic -> don't have to worry about edge overlaps
                region = [centre[0] * hconst - 0.5 * Ls_hunits[0],\
                          centre[0] * hconst + 0.5 * Ls_hunits[0],\
                          centre[1] * hconst - 0.5 * Ls_hunits[1],\
                          centre[1] * hconst + 0.5 * Ls_hunits[1],\
                          centre[2] * hconst - 0.5 * Ls_hunits[2],\
                          centre[2] * hconst + 0.5 * Ls_hunits[2],\
                          ]
                vardict.region = region
                print(vardict.region)
        # apply precise selection to particle centres
        BoxSize = vardict.simfile.boxsize / vardict.simfile.h # cMpc
        box3 = list((BoxSize,)*3)
        
        # coords are stored in float64, but I don't need that precision here. 
        vardict.readif('Coordinates', rawunits=True)
        vardict.particle['Coordinates'] *= (1. / vardict.simfile.h)
        vardict.CGSconv['Coordinates'] *= vardict.simfile.h
        #print(vardict.particle['Coordinates'])
        #print(vardict.CGSconv['Coordinates'])
        translate(vardict.particle, 'Coordinates', centre, box3, False) # periodic = False -> centre on centre
        
        doselection = np.array([0,1,2])[Ls < BoxSize] # no selection needed if the selection dimension is the whole box (with an fp margin for h conversions)
        if len(doselection) > 0:
            sel = pc.Sel()
            for axind in doselection:
                margin = 0.5 * Ls[axind]
                sel.comb({'arr': -1. * margin <= vardict.particle['Coordinates'][:, axind]})
                sel.comb({'arr':       margin >  vardict.particle['Coordinates'][:, axind]})
        vardict.update(sel)
        #print(len(vardict.particle['Coordinates']))
        #print('sel length, num. True: %s, %s'%(len(sel.val), np.sum(sel.val)))
        
        keepcoords = np.any([sub['ptype'] == 'coords' for sub in axesdct]) # keep the centred coordinates if they're needed for r3D later
        if not keepcoords:
            vardict.delif('Coordinates', last=keepcoords)
        vardict.add_box('box3', box3)
        vardict.overwrite_box('centre',centre)
        vardict.overwrite_box('Ls',Ls)
    
    outfilename = namehistogram_perparticle(ptype, simnum, snapnum, var, simulation,\
                              L_x, L_y, L_z, centre, Ls_in_Mpc, simfile.boxsize, simfile.h, excludeSFR,\
                              abunds, ion, parttype, quantity,\
                              sylviasshtables, bensgadget2tables,\
                              misc)
    axnames = [namehistogram_perparticle_axis(dct) for dct in axesdct]
    groupname = '_'.join(axnames) 
    if name_append is not None:
        groupname = groupname + name_append
    
    if nameonly:
        return outfilename, groupname

    with h5py.File(outfilename, 'a') as outfile:
        if groupname in outfile.keys():
            raise RuntimeError('This histogram already seems to exist; specify name_append to get a new unique name')
        else:
            group = outfile.create_group(groupname)
    
        if 'Header' not in outfile.keys():
            hed = outfile.create_group('Header')
            hed.attrs.create('simnum', np.string_(simnum))
            hed.attrs.create('snapnum', snapnum)
            hed.attrs.create('var', np.string_(var))
            hed.attrs.create('simulation', np.string_(simulation))
            hed.attrs.create('make_maps_opts_locs.emtab_sylvia_ssh', 
                             np.string_(str(ol.emtab_sylvia_ssh)))
            hed.attrs.create('make_maps_opts_locs.iontab_sylvia_ssh', 
                             np.string_(str(ol.iontab_sylvia_ssh)))
            csm = hed.create_group('cosmopars')
            csm.attrs.create('a', simfile.a)
            csm.attrs.create('z', simfile.z)
            csm.attrs.create('h', simfile.h)
            csm.attrs.create('omegab', simfile.omegab)
            csm.attrs.create('omegam', simfile.omegam)
            csm.attrs.create('omegalambda', simfile.omegalambda)
            csm.attrs.create('boxsize', simfile.boxsize)
        
        if 'Units' not in outfile.keys():
            ung = outfile.create_group('Units')
            ung.attrs.create('all', np.string_('cgs, proper lengths'))
            ung.attrs.create('emission', np.string_('photons (s**-1 * cm**-3 or s**-1)'))
    
        axdata = []
        axbins_touse = []
        for axind in range(len(axesdct)): #loop: large memory use, most likely, in each part
            dct_t = axesdct[axind]
            ptype_t = dct_t['ptype']
            excludeSFR_t = dct_t['excludeSFR']
            if 'ion' in dct_t.keys():
                ion_t = dct_t['ion']
            else:
                ion_t = None
            if 'abunds' in dct_t.keys():
                abunds_t = dct_t['abunds']
            else:
                abunds_t = None
            if 'quantity' in dct_t.keys():
                quantity_t = dct_t['quantity']
            else:
                quantity_t = None
            if 'misc' in dct_t.keys():
                misc_t = dct_t['misc']
            else:
                misc_t = None
            if 'mdef' in dct_t.keys():
                mdef_t = dct_t['mdef']
            else:
                mdef_t = None
            if 'allinR200c' in dct_t.keys():
                allinR200c_t = dct_t['allinR200c']
            else:
                allinR200c_t = None
            if 'sylviasshtables' in dct_t.keys():
                sylviasshtables_t = dct_t['sylviasshtables']
            else:
                sylviasshtables_t = None
            if 'bensgadget2tables' in dct_t.keys():
                bensgadget2tables_t = dct_t['bensgadget2tables']
            else:
                bensgadget2tables_t = None
            if 'ps20tables' in dct_t.keys():
                ps20tables_t = dct_t['ps20tables']
            else:
                ps20tables_t = None
            if 'ps20depletion' in dct_t.keys():
                ps20depletion_t = dct_t['ps20depletion']
            else:
                ps20depletion_t = None
            logax_t = logax[axind]
            axdata_t, multipafter_t = getparticledata(vardict, ptype_t, excludeSFR_t, abunds_t,\
                                                     ion_t, quantity_t,\
                                                     sylviasshtables=sylviasshtables_t,\
                                                     bensgadget2tables=bensgadget2tables_t,\
                                                     ps20tables=ps20tables_t, 
                                                     ps20depletion=ps20depletion_t,
                                                     last=True,\
                                                     updatesel=False, misc=misc_t, mdef=mdef_t,\
                                                     allinR200c=allinR200c_t)
            
            if logax_t:
                axdata_t = np.log10(axdata_t) + np.log10(multipafter_t)
            else:
                axdata_t *= multipafter_t 
            print('ptype_t, quantity_t, ion_t: %s, %s, %s'%(ptype_t, quantity_t, ion_t))
            #print(axdata_t)
            min_t = np.min(axdata_t[np.isfinite(axdata_t)])
            max_t = np.max(axdata_t[np.isfinite(axdata_t)])
    
            grp = group.create_group(axnames[axind])
            grp.attrs.create('number of particles', len(axdata_t))
            grp.attrs.create('number of particles with finite values', int(np.sum(np.isfinite(axdata_t))) )
            grp.attrs.create('histogram axis', axind)
            grp.attrs.create('log', logax_t)
            
            if ptype_t == 'halo' and quantity_t == 'subcat': # override input binning for standard indices
                if logax_t:
                    bins = np.log10([0.1, 1., 2., 3.])
                else:
                    bins = np.array([0., 1., 2., 3.])
                axbins_t = bins
                if axbins_t[-1] < max_t:
                    numgtr = np.sum(axdata_t[np.isfinite(axdata_t)] > axbins_t[-1])
                else:
                    numgtr = 0
                if axbins_t[0] > min_t:
                    numltr = np.sum(axdata_t[np.isfinite(axdata_t)] < axbins_t[0])
                else:
                    numltr = 0
                grp.attrs.create('number of particles > max value', numgtr)
                grp.attrs.create('number of particles < min value', numltr)
                grp.attrs.create('info', np.string_('indices correspond to subhalo categories'))
                grp.attrs.create('index 0', np.string_('central galaxy and bound to halo'))
                grp.attrs.create('index 1', np.string_('bound to satellities'))
                grp.attrs.create('index 2', np.string_('no subgroup membership'))     
                
            elif axbins is not None:
                if hasattr(axbins, '__len__'):
                    axbins_t = axbins[axind]
                else:
                    axbins_t = axbins
                if isinstance(axbins_t, int): # number of bins
                    axbins_t = np.linspace(min_t * ( 1. - 1.e-7), max_t * (1. + 1.e-7), axbins_t + 1)
                    grp.attrs.create('number of particles > max value', 0)
                    grp.attrs.create('number of particles < min value', 0)
                elif isinstance(axbins_t, num.Number): # something float-like: spacing
                    # search for round values up to spacing below min and above max
                    minbin = np.floor(min_t / axbins_t) * axbins_t
                    maxbin = np.ceil(max_t / axbins_t) * axbins_t
                    axbins_t = np.arange(minbin, maxbin + axbins_t/2., axbins_t)
                    grp.attrs.create('number of particles > max value', 0)
                    grp.attrs.create('number of particles < min value', 0)
                else: # list/array of bin edges
                    #print(axbins_t)
                    #print(min_t)
                    if axbins_t[-1] < max_t:
                        numgtr = np.sum(axdata_t[np.isfinite(axdata_t)] > axbins_t[-1])
                    else:
                        numgtr = 0
                    if axbins_t[0] > min_t:
                        numltr = np.sum(axdata_t[np.isfinite(axdata_t)] < axbins_t[0])
                    else:
                        numltr = 0
                    grp.attrs.create('number of particles > max value', numgtr)
                    grp.attrs.create('number of particles < min value', numltr)
            else:
                raise ValueError('axbins must be specified')
                
            axbins_t = np.array(axbins_t)
            grp.create_dataset('bins', data=axbins_t)
            saveattr(grp, 'ptype', ptype_t)
            saveattr(grp, 'excludeSFR', excludeSFR_t)
            saveattr(grp, 'ion', ion_t)
            saveattr(grp, 'quantity', quantity_t)
            saveattr(grp, 'sylviasshtables', sylviasshtables_t)
            saveattr(grp, 'bensgadget2tables', bensgadget2tables_t)
            saveattr(grp, 'ps20tables', ps20tables_t)
            saveattr(grp, 'ps20depletion', ps20depletion_t)
            saveattr(grp, 'misc', misc_t)
            saveattr(grp, 'mdef', mdef_t)
            saveattr(grp, 'allinR200c', allinR200c_t)
            if isinstance(abunds_t, tuple):
                saveattr(grp, 'abunds', 'tuple')
                saveattr(grp, 'abunds0', abunds_t[0])
                saveattr(grp, 'abunds1', abunds_t[1])
            else:
                saveattr(grp, 'abunds', abunds_t)

            axbins_touse += [axbins_t]
            axdata += [axdata_t]
            del axdata_t
                   
        weight, multipafter_w = getparticledata(vardict, ptype, excludeSFR, 
                                                abunds, ion, quantity,
                                                sylviasshtables=sylviasshtables, 
                                                bensgadget2tables=bensgadget2tables,
                                                ps20tables=ps20tables, 
                                                ps20depletion=ps20depletion,
                                                last=True, updatesel=False, 
                                                misc=None, 
                                                allinR200c=allinR200c, 
                                                mdef=mdef)
        maxw = np.max(weight)
        lenw = len(weight)
        # rescale for fp precision and overflow avoidance
        resce = np.ceil(np.log10(maxw) + np.log10(lenw)) - 37 # max float32 between 1e38 and 1e39
        weight *= 10**(-1 * resce)
        multipafter_w *= 10**resce # will restore any overflow in the end, unless log values are stored 
        
        # loop to prevent memory from running out (histogramdd issue)
        maxperslice = 752**3 // 8
        if len(weight) < maxperslice:
            hist, edges = np.histogramdd(axdata, weights=weight, bins=axbins_touse)
        else:
            lentot = len(weight)
            numperslice = maxperslice
            slices = [slice(i * numperslice, min((i + 1) * numperslice, lentot), None) for i in range((lentot - 1) // numperslice + 1)]
            for slind in range(len(slices)):
                axdata_temp = [data[slices[slind]] for data in axdata]
                hist_temp, edges_temp = np.histogramdd(axdata_temp, weights=weight[slices[slind]], bins=axbins_touse)
                if slind == 0 :
                    hist = hist_temp
                    edges = edges_temp
                else:
                    hist += hist_temp
                    if not np.all(np.array([np.all(edges[i] == edges_temp[i]) for i in range(len(edges))])):
                        outfile.close()
                        raise RuntimeError('Error: edges mismatch in histogramming loop (slind = %i)'%slind)
                        
                        return None
        wsum = np.sum(weight) * multipafter_w
        if loghist:
            hist = np.log10(hist) + np.log10(multipafter_w)
        else:
            hist *= multipafter_w
        group.create_dataset('histogram', data=hist)
        group['histogram'].attrs.create('sum of weights', wsum) # should be equal to sum of the histogram, but it's good to check
        group['histogram'].attrs.create('log', loghist)
        bingrp = group.create_group('binedges')
        saveattr(group, 'L_x', L_x)
        saveattr(group, 'L_y', L_y)
        saveattr(group, 'L_z', L_z)
        saveattr(group, 'centre', centre)
        saveattr(group, 'Ls_in_Mpc', Ls_in_Mpc)
        saveattr(group, 'ptype', ptype)
        saveattr(group, 'excludeSFR', excludeSFR)
        saveattr(group, 'ion', ion)
        saveattr(group, 'quantity', quantity)
        saveattr(group, 'sylviasshtables', sylviasshtables)
        saveattr(group, 'bensgadget2tables', bensgadget2tables)
        saveattr(group, 'ps20tables', ps20tables)
        saveattr(group, 'ps20depletion', ps20depletion)
        saveattr(group, 'misc', misc)
        saveattr(group, 'mdef', mdef)
        saveattr(group, 'allinR200c', allinR200c)
        
        for i in range(len(edges)):
            bingrp.create_dataset('Axis%i'%(i), data=edges[i])

    return hist, axbins_touse