#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:01:03 2019

@author: wijers
"""

import numpy as np
import string
import h5py

import make_maps_v3_master as m3
import projection_classes as pc
import make_maps_opts_locs as ol
import eagle_constants_and_units as cu
import ion_line_data as ild
import selecthalos as sh

def calcomegas(simnum, snap, var):
    outfile = ol.ndir + 'omega_oxygen_data_%s_snap%i_%s.txt'%(simnum, snap, var)
    out = open(outfile, 'w')
    
    simfile = pc.Simfile(simnum, snap, var, file_type='snap', simulation='eagle')
    
    Hz = m3.Hubble(simfile.z, simfile=simfile)
    rhocrit =  3. / (8. * np.pi * cu.gravity) * Hz**2   
    simvolume = (simfile.boxsize * simfile.a / simfile.h * cu.cm_per_mpc)**3
    
    # black holes
    bhmass = simfile.readarray('PartType5/Mass', rawunits=True)
    bhmass_tocgs = simfile.CGSconvtot
    bhmass_tot = np.sum(bhmass) * bhmass_tocgs
    out.write('Omega_BH: \t%s\n'%(str(bhmass_tot / rhocrit / simvolume)))
    
    del bhmass
    
    # stars    
    starmass = simfile.readarray('PartType4/Mass', rawunits=True)
    starmass_tocgs = simfile.CGSconvtot
    starmass_tot = np.sum(starmass) * starmass_tocgs
    out.write('Omega_stars: \t%s\n'%(str(starmass_tot / rhocrit / simvolume)))
    
    starOxygenmass = simfile.readarray('PartType4/ElementAbundance/Oxygen', rawunits=True)
    starOxfrac_tocgs = simfile.CGSconvtot
    starOxygenmass *=  starmass
    starOxygenmass_tot = np.sum(starOxygenmass) * starOxfrac_tocgs * starmass_tocgs
    out.write('Omega_oxygen_stars: \t%s\n'%(str(starOxygenmass_tot / rhocrit / simvolume)))
    
    # gas
    gasmass = simfile.readarray('PartType0/Mass', rawunits=True)
    gasmass_tocgs = simfile.CGSconvtot
    gasmass_tot = np.sum(gasmass) * gasmass_tocgs
    out.write('Omega_gas: \t%s\n'%(str(gasmass_tot / rhocrit / simvolume)))

    gasOxygenmass = simfile.readarray('PartType0/ElementAbundance/Oxygen', rawunits=True)
    gasOxfrac_tocgs = simfile.CGSconvtot
    gasOxygenmass *=  gasmass
    gasOxygenmass_tot = np.sum(gasOxygenmass) * gasOxfrac_tocgs * gasmass_tocgs
    out.write('Omega_oxygen_gas: \t%s\n'%(str(gasOxygenmass_tot / rhocrit / simvolume)))

    # temperature (SF gas  -> 10^4 K)
    temp  = simfile.readarray('PartType0/Temperature', rawunits=True)
    temp_tocgs = simfile.CGSconvtot
    temp *= temp_tocgs
    sfr   = simfile.readarray('PartType0/StarFormationRate', rawunits=True)
    temp[sfr > 0.] = 10**4.
    del sfr 
    temp = np.log10(temp)
    
    # hydrogen number density
    hfrac = simfile.readarray('PartType0/ElementAbundance/Hydrogen', rawunits=True)
    dens  = simfile.readarray('PartType0/Density', rawunits=True)
    dens_tocgs = simfile.CGSconvtot
    dens *= hfrac * (dens_tocgs / (cu.atomw_H * cu.u))
    dens = np.log10(dens)
    
    gaso6mass = m3.find_ionbal(simfile.z, 'o6', {'logT': temp, 'lognH': dens})
    gaso6mass *= gasOxygenmass
    gaso6mass_tot = np.sum(gaso6mass) * gasOxfrac_tocgs * gasmass_tocgs
    out.write('Omega_o6_gas: \t%s\n'%(str(gaso6mass_tot / rhocrit / simvolume)))
    del gaso6mass
    
    gaso7mass = m3.find_ionbal(simfile.z, 'o7', {'logT': temp, 'lognH': dens})
    gaso7mass *= gasOxygenmass
    gaso7mass_tot = np.sum(gaso7mass) * gasOxfrac_tocgs * gasmass_tocgs
    out.write('Omega_o7_gas: \t%s\n'%(str(gaso7mass_tot / rhocrit / simvolume)))
    del gaso7mass
    
    gaso8mass = m3.find_ionbal(simfile.z, 'o8', {'logT': temp, 'lognH': dens})
    gaso8mass *= gasOxygenmass
    gaso8mass_tot = np.sum(gaso8mass) * gasOxfrac_tocgs * gasmass_tocgs
    out.write('Omega_o8_gas: \t%s\n'%(str(gaso8mass_tot / rhocrit / simvolume)))
    del gaso8mass
    
    out.write('Omega_b_nominal (z=0): \t%s\n'%(simfile.omegab))
    omegab_at_z = 3. / (8. * np.pi * cu.gravity)* cu.hubble**2 * simfile.h**2 * simfile.omegab / simfile.a**3 / rhocrit
    out.write('Omega_b_nominal (z): \t%s\n'%omegab_at_z)
    out.close()

def getbins(minval, maxval, mincare, maxcare, pre=None, post=None, numdec=1, binsize=None, inclinf=True):
    '''
    binsize overrides numdec if set
    '''
    if binsize is None:
        size = 10.**(-1. * numdec)
        rmin = np.round(minval, numdec)
        rmax = np.round(maxval, numdec)
    else: # assume edge values should be chosen such that 0 is one of them, if extended sufficently
        size = binsize
        rmin = np.floor(minval / size) * size 
        rmax = np.ceil(maxval / size) * size 
    if rmin >= minval:
        rmin -= size
    if rmax <= maxval:
        rmax += size
            
    inclpre = False
    inclpost = False
    if rmin < mincare:
        rmin = mincare
        inclpre = True
    if rmax > maxcare:
        rmax = maxcare
        inclpost = True
    
    cen = list(np.arange(rmin, rmax + 0.5 * size, size))
    
    if inclpost and post is not None:
        cen = cen + list(post)
    if inclpre and pre is not None:
        cen = list(pre) + cen
    
    if inclinf:
        if cen[0] != -np.inf:
            cen = [-np.inf] + cen
        if cen[-1] != np.inf:
            cen = cen + [np.inf]
    
    return np.array(cen)
    
    
    
def calc_gas_props(simnum, snapnum, var='REFERENCE', simulation='eagle', ions=None, bindec=1, binsize=None):

    simfile = pc.Simfile(simnum, snapnum, var, file_type='snap', simulation='eagle')
    
    # temperature (SF gas  -> 10^4 K)
    temp  = simfile.readarray('PartType0/Temperature', rawunits=True)
    temp_tocgs = simfile.CGSconvtot
    temp *= temp_tocgs
    sfr   = simfile.readarray('PartType0/StarFormationRate', rawunits=True)
    temp[sfr > 0.] = 10**4.
    del sfr 
    temp = np.log10(temp)
    
    # hydrogen number density
    hfrac = simfile.readarray('PartType0/ElementAbundance/Hydrogen', rawunits=True)
    dens  = simfile.readarray('PartType0/Density', rawunits=True)
    dens_tocgs = simfile.CGSconvtot
    dens *= hfrac * (dens_tocgs / (cu.atomw_H * cu.u))
    dens = np.log10(dens)
    
    # Metallicity
    Zsm  = simfile.readarray('PartType0/SmoothedMetallicity', rawunits=True)
    Zsm_tocgs = simfile.CGSconvtot
    if Zsm_tocgs != 1.:
        Zsm *= Zsm_tocgs
    Zsm = np.log10(Zsm)
    
    name = 'gashistogram_%s%s_%s_PtAb_T4EOS.hdf5'%(simnum, var, snapnum)
    outfile = h5py.File(ol.ndir + name, 'a')
    
    if 'Header' not in outfile.keys():
        hd = outfile.create_group('Header')
        cm = hd.create_group('cosmopars')
        cm.attrs.create('h', simfile.h)
        cm.attrs.create('a', simfile.a)
        cm.attrs.create('z', simfile.z)
        cm.attrs.create('boxsize', simfile.boxsize)
        cm.attrs.create('omegam', simfile.omegam)
        cm.attrs.create('omegab', simfile.omegab)
        cm.attrs.create('omegalambda', simfile.omegalambda)
        hd.attrs.create('info', 'gas nH, T, SmoothedZ weighted by gas, element, ion masses')
        un = hd.create_group('units')
        un.attrs.create('Temperature', 'log10 K, SF gas at 10^4 K')
        un.attrs.create('nH', 'hydrogen number density, PtAb, log10 cm^-3')
        un.attrs.create('SmoothedMetallicity', 'log10 gas mass fraction')
        un.attrs.create('histograms', 'mass-weighted, arbitrary units')
    
    if 'edges' not in outfile.keys():
        ed = outfile.create_group('edges')
        ed.create_dataset('axorder', data=np.array(['nH', 'Temperature', 'SmoothedMetallicity']))
        
        Tmin = np.min(temp)
        Tmax = np.max(temp)
        Tbins = getbins(Tmin, Tmax, 2., np.inf, pre=[0., 1.], post=None, numdec=bindec, binsize=binsize, inclinf=True)
        
        nHmin = np.min(dens)
        nHmax = np.max(dens)
        nHbins = getbins(nHmin, nHmax, -8.5, np.inf, pre=[-9.5, -9.], post=None, numdec=bindec, binsize=binsize, inclinf=True)
        
        Zmin = np.min(dens)
        Zmax = np.max(dens)
        Zbins = getbins(Zmin, Zmax, -4., np.inf, pre=[-6., -5.5, -5., -4.5], post=None, numdec=bindec, binsize=binsize, inclinf=True)
        
        ed.create_dataset('Temperature', data=Tbins)
        ed.create_dataset('nH', data=nHbins)
        ed.create_dataset('SmoothedMetallicity', data=Zbins)
    else:
        Tbins = np.array(outfile['edges/Temperature'])
        nHbins = np.array(outfile['edges/nH'])
        Zbins = np.array(outfile['edges/SmoothedMetallicity'])
    
    if 'histograms' not in outfile.keys():
        hs = outfile.create_group('histograms')
    else:
        hs = outfile['histograms']
        
    gasmass = simfile.readarray('PartType0/Mass', rawunits=True)
    if 'Mass' not in hs.keys():
        hist, edges = np.histogramdd([dens, temp, Zsm], bins=[nHbins, Tbins, Zbins], weights=gasmass)
        hs.create_dataset('Mass', data=hist)
    
    if ions is not None:
        elements = set([ild.elements_ion[ion] for ion in ions])
        iondct = {elt: [ion if ild.elements_ion[ion] == elt else None for ion in ions] for elt in elements}
        iondct = {elt: list(set(iondct[elt]) - set([None])) for elt in iondct.keys()}
        
        for element in iondct.keys():
            if element in hs.keys() and np.all([ion in hs.keys() for ion in iondct[element]]): # already have all these ones -> no need to recalculate
                continue 
            eltmass = simfile.readarray('PartType0/ElementAbundance/%s'%(string.capwords(element)), rawunits=True)
            eltmass *=  gasmass
            
            if element not in hs.keys():
                hist, edges = np.histogramdd([dens, temp, Zsm], bins=[nHbins, Tbins, Zbins], weights=eltmass)
                hs.create_dataset(element, data=hist)
    
            for ion in iondct[element]:
                if ion in hs.keys():
                    continue
                ionmass = m3.find_ionbal(simfile.z, ion, {'logT': temp, 'lognH': dens})
                ionmass *= eltmass
                hist, edges = np.histogramdd([dens, temp, Zsm], bins=[nHbins, Tbins, Zbins], weights=ionmass)
                hs.create_dataset(ion, data=hist)
    
    outfile.close()
    


def calc_halo_stats(simnum, snapnum, var='REFERENCE', simulation='eagle', ions=None, bindec=1, binsize=None):
    '''
    just a stub; DO NOT USE
    '''
    mdef = '200c'
    
    halomassbins = np.array([-np.inf] + list(np.arange(9., 14.1, 0.5)) + [np.inf])
    # subgroup catgories: subgroupnumber 0, subgroupnumber for satellite, subgroupnumber 2^32 (unbound)
    halosels = [[('M200c_logMsun', halomassbins[i], halomassbins[i + 1])] for i in range(len(halomassbins) - 1)]
    # particle data + FoF 
    simfile = pc.Simfile(simnum, snapnum, var, file_type='particle', simulation='eagle')
    simfile_sf = pc.Simfile(simfile.simnum, simfile.snapnum, simfile.var, file_type='sub', simulation=simfile.simulation)
    # groupnum 2^32 -> no group
    groupnums = {binind: sh.selecthalos_subfindfiles(simfile_sf, halosels[binind], mdef=mdef, aperture=30, nameonly=False) for binind in range(len(halosels))}
    
    if simfile.var == 'REFERENCE':
        vind = 'Ref'
    elif simfile.var == 'RECALIBRATED':
        vind = 'Recal'
    else:
        vind = simfile.var
        
    filename = 'halohist_box_%s%s_%s_by_M%s.hdf5'%(vind, simnum, snapnum, mdef)
    filename = ol.pdir + filename
    # set up header
    with h5py.File(filename, 'a') as fo:
        if 'Header' not in fo:
            hed = fo.create_group('Header')
            hed.attrs.create('simnum', np.string_(simnum))
            hed.attrs.create('snapnum', snapnum)
            hed.attrs.create('var', np.string_(simfile.var))
            csm = hed.create_group('cosmopars')
            csm.attrs.create('boxsize', simfile.boxsize)
            csm.attrs.create('a', simfile.a)
            csm.attrs.create('h', simfile.h)
            csm.attrs.create('z', simfile.z)
            csm.attrs.create('omegam', simfile.omegam)
            csm.attrs.create('omegab', simfile.omegab)
            csm.attrs.create('omegalambda', simfile.omegalambda)
        
    # set up the things to get histograms of
    parttype='0'
    weights = [('basic', 'Mass'), ('basic', 'propvol')]
    if ions is not None:
        weights = weights + ions
    hists = ['lognH', 'logT', 'SmoothedMetallicity']
    
    # temperature (SF gas  -> 10^4 K)
    temp  = simfile.readarray('PartType0/Temperature', rawunits=True)
    temp_tocgs = simfile.CGSconvtot
    temp *= temp_tocgs
    sfr   = simfile.readarray('PartType0/StarFormationRate', rawunits=True)
    temp[sfr > 0.] = 10**4.
    del sfr 
    temp = np.log10(temp)
    
    # hydrogen number density
    hfrac = simfile.readarray('PartType0/ElementAbundance/Hydrogen', rawunits=True)
    dens  = simfile.readarray('PartType0/Density', rawunits=True)
    dens_tocgs = simfile.CGSconvtot
    dens *= hfrac * (dens_tocgs / (cu.atomw_H * cu.u))
    dens = np.log10(dens)
    
    # Metallicity
    Zsm  = simfile.readarray('PartType0/SmoothedMetallicity', rawunits=True)
    Zsm_tocgs = simfile.CGSconvtot
    if Zsm_tocgs != 1.:
        Zsm *= Zsm_tocgs
    Zsm = np.log10(Zsm)
    
    name = 'gashistogram_withhaloinfo_%s%s_%s_PtAb_T4EOS.hdf5'%(simnum, var, snapnum)
    outfile = h5py.File(ol.ndir + name, 'a')
    
    if 'Header' not in outfile.keys():
        hd = outfile.create_group('Header')
        cm = hd.create_group('cosmopars')
        cm.attrs.create('h', simfile.h)
        cm.attrs.create('a', simfile.a)
        cm.attrs.create('z', simfile.z)
        cm.attrs.create('boxsize', simfile.boxsize)
        cm.attrs.create('omegam', simfile.omegam)
        cm.attrs.create('omegab', simfile.omegab)
        cm.attrs.create('omegalambda', simfile.omegalambda)
        hd.attrs.create('info', 'gas nH, T, SmoothedZ weighted by gas, element, ion masses')
        un = hd.create_group('units')
        un.attrs.create('Temperature', 'log10 K, SF gas at 10^4 K')
        un.attrs.create('nH', 'hydrogen number density, PtAb, log10 cm^-3')
        un.attrs.create('SmoothedMetallicity', 'log10 gas mass fraction')
        un.attrs.create('histograms', 'mass-weighted, arbitrary units')
    
    if 'edges' not in outfile.keys():
        ed = outfile.create_group('edges')
        ed.create_dataset('axorder', data=np.array(['nH', 'Temperature', 'SmoothedMetallicity']))
        
        Tmin = np.min(temp)
        Tmax = np.max(temp)
        Tbins = getbins(Tmin, Tmax, 2., np.inf, pre=[0., 1.], post=None, numdec=bindec, binsize=binsize, inclinf=True)
        
        nHmin = np.min(dens)
        nHmax = np.max(dens)
        nHbins = getbins(nHmin, nHmax, -8.5, np.inf, pre=[-9.5, -9.], post=None, numdec=bindec, binsize=binsize, inclinf=True)
        
        Zmin = np.min(dens)
        Zmax = np.max(dens)
        Zbins = getbins(Zmin, Zmax, -4., np.inf, pre=[-6., -5.5, -5., -4.5], post=None, numdec=bindec, binsize=binsize, inclinf=True)
        
        ed.create_dataset('Temperature', data=Tbins)
        ed.create_dataset('nH', data=nHbins)
        ed.create_dataset('SmoothedMetallicity', data=Zbins)
    else:
        Tbins = np.array(outfile['edges/Temperature'])
        nHbins = np.array(outfile['edges/nH'])
        Zbins = np.array(outfile['edges/SmoothedMetallicity'])
    
    if 'histograms' not in outfile.keys():
        hs = outfile.create_group('histograms')
    else:
        hs = outfile['histograms']
        
    gasmass = simfile.readarray('PartType0/Mass', rawunits=True)
    if 'Mass' not in hs.keys():
        hist, edges = np.histogramdd([dens, temp, Zsm], bins=[nHbins, Tbins, Zbins], weights=gasmass)
        hs.create_dataset('Mass', data=hist)
    
    if ions is not None:
        elements = set([ild.elements_ion[ion] for ion in ions])
        iondct = {elt: [ion if ild.elements_ion[ion] == elt else None for ion in ions] for elt in elements}
        iondct = {elt: list(set(iondct[elt]) - set([None])) for elt in iondct.keys()}
        
        for element in iondct.keys():
            if element in hs.keys() and np.all([ion in hs.keys() for ion in iondct[element]]): # already have all these ones -> no need to recalculate
                continue 
            eltmass = simfile.readarray('PartType0/ElementAbundance/%s'%(string.capwords(element)), rawunits=True)
            eltmass *=  gasmass
            
            if element not in hs.keys():
                hist, edges = np.histogramdd([dens, temp, Zsm], bins=[nHbins, Tbins, Zbins], weights=eltmass)
                hs.create_dataset(element, data=hist)
    
            for ion in iondct[element]:
                if ion in hs.keys():
                    continue
                ionmass = m3.find_ionbal(simfile.z, ion, {'logT': temp, 'lognH': dens})
                ionmass *= eltmass
                hist, edges = np.histogramdd([dens, temp, Zsm], bins=[nHbins, Tbins, Zbins], weights=ionmass)
                hs.create_dataset(ion, data=hist)
    
    outfile.close()