#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:32:41 2018

@author: wijers
"""
import numpy as np
import make_maps_opts_locs as ol
import h5py
import scipy

ndir = ol.ndir
mdir = '/net/luttero/data2/imgs/whim_xray_paper1/' # luttero location
pdir = ol.pdir

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines

import makecddfs as mc
import eagle_constants_and_units as c #only use for physical constants and unit conversion!
import loadnpz_and_plot as lnp
import make_maps_v3_master as m3 # for ion balances
import simfileclone as sfc # for cooling contours

from plothistograms_filenames import  *


def savehist_compressed_as_hdf5(openhdf5group, hist, saveaxes, axsumtuple=None):
    '''
    hist: npzldif object
    saveaxes: axes to not immediately sum over
    axsumtuple: (axis, index list) for this axis, the total and sums up to, between, and after the given indices are stored
    axis from axsumtupl should be in the saveaxes list
    '''
    try:
        saveaxes = list(saveaxes)
    except TypeError:
        saveaxes = [saveaxes]
    summedaxes = tuple(list( set(range(len(hist['edges'])))-set(saveaxes) )) # get axes to sum over
    saveaxes = sorted(saveaxes)
    
    grp = openhdf5group
    grp.attrs.create('file_origin', hist.filename)
    grp.attrs.create('numcounts_total', hist.npt)
    
    if axsumtuple is None:
        if len(summedaxes) == 0:
            tosave = hist['bins']
        else:
            tosave = np.sum(hist['bins'], axis=summedaxes)  
        dimensions = hist['dimension'][np.array(saveaxes)]
        edges = hist['edges'][np.array(saveaxes)]
        grp.create_dataset('hist_all', data=tosave)
        grp.create_dataset('dimension', data=dimensions)
        grp['dimension'].attrs.create('info', 'name of quantity histogrammed along each axis')
        for i in range(len(dimensions)):
            grp.create_dataset('edges_axis%i'%i, data=edges[i])
    else:
        if axsumtuple[0] in saveaxes: # remove from saveaxes, redo summedaxes calc
            saveaxes.remove(axsumtuple[0])
            summedaxes = tuple(list( set(range(len(hist['edges'])))-set(saveaxes) )) # get axes to sum over
            saveaxes = list(saveaxes)
            saveaxes = sorted(saveaxes)
        if len(summedaxes) == 0:
            tosave = hist['bins']
        else:
            tosave = np.sum(hist['bins'], axis=summedaxes)  
        # save data on sum over all bins
        dimensions = hist['dimension'][saveaxes]
        edges = hist['edges'][saveaxes]
        grp.create_dataset('hist_all', data=tosave)
        grp.create_dataset('dimension', data=dimensions)
        grp['dimension'].attrs.create('info', 'name of quantity histogrammed along each axis')
        for i in range(len(dimensions)):
            grp.create_dataset('edges_axis%i'%i, data=edges[i])
        
        # get the subset histograms
        slice_base = list((slice(None, None, None),)*len(hist['edges']))
        axforpartsum = axsumtuple[0]
        axpartinds = axsumtuple[1]
        slices_part = [tuple(slice_base[:axforpartsum] + [slice(None, axpartinds[i], None)] + slice_base[axforpartsum + 1:]) if i == 0 else\
                       tuple(slice_base[:axforpartsum] + [slice(axpartinds[i - 1], None, None)] + slice_base[axforpartsum + 1:]) if i == len(axpartinds) else\
                       tuple(slice_base[:axforpartsum] + [slice(axpartinds[i - 1], axpartinds[i], None)] + slice_base[axforpartsum + 1:])\
                       for i in range(len(axpartinds) + 1)]
        
        slicenames = [hist['dimension'][axforpartsum] + '_min-%s_to_%s'%(hist['edges'][axforpartsum][0], hist['edges'][axforpartsum][axpartinds[i]]) if i == 0 else\
                      hist['dimension'][axforpartsum] + '_%s_to_max-%s'%(hist['edges'][axforpartsum][axpartinds[i-1]], hist['edges'][axforpartsum][-1]) if i == len(axpartinds) else\
                      hist['dimension'][axforpartsum] + '_%s_to_%s'%(hist['edges'][axforpartsum][axpartinds[i-1]], hist['edges'][axforpartsum][axpartinds[i]])\
                      for i in range(len(axpartinds) + 1)]
        attrs_store = [{'minval': hist['edges'][axforpartsum][0], 'maxval': hist['edges'][axforpartsum][axpartinds[i]], 'minincl': True, 'maxincl': False} if i == 0 else\
                 {'minval': hist['edges'][axforpartsum][axpartinds[i-1]], 'maxval': hist['edges'][axforpartsum][-1], 'minincl': False, 'maxincl': True} if i == len(axpartinds) else\
                 {'minval': hist['edges'][axforpartsum][axpartinds[i-1]], 'maxval': hist['edges'][axforpartsum][axpartinds[i]], 'minincl': False, 'maxincl': False}\
                 for i in range(len(axpartinds) + 1)]
        for i in range(len(axpartinds) + 1):
            grp.create_dataset(slicenames[i], data=np.sum(hist['bins'][slices_part[i]], axis=summedaxes))
            for key in attrs_store[i].keys():
                grp[slicenames[i]].attrs.create(key, attrs_store[i][key])
    return None
        
# load ion balance table (z=0. matches snapshots)
o7_ib, logTK_ib, lognHcm3_ib = m3.findiontables('o7',0.0)
o8_ib, logTK_ib, lognHcm3_ib = m3.findiontables('o8',0.0)


# conversions and cosmology
cosmopars_ea = mc.getcosmopars('L0100N1504',28,'REFERENCE',file_type = 'snap',simulation = 'eagle')
cosmopars_ba = mc.getcosmopars('L400N1024',32,'REFERENCE',file_type = 'snap',simulation = 'bahamas')
cosmopars_ea_27 = mc.getcosmopars('L0100N1504',27,'REFERENCE',file_type = 'snap',simulation = 'eagle')

o7_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o7',cosmopars_ea_27['z'])
o8_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o8',cosmopars_ea_27['z'])
o6_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o6',cosmopars_ea_27['z'])
ne8_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('ne8',cosmopars_ea_27['z'])

# both in cgs, using primordial abundance from EAGLE
rho_to_nh = 0.752/(c.atomw_H*c.u)

logrhocgs_ib = lognHcm3_ib - np.log10(rho_to_nh)

logrhob_av_ea = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ea['h']**2 * cosmopars_ea['omegab'] ) 
logrhob_av_ba = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ba['h']**2 * cosmopars_ba['omegab'] )
logrhob_av_ea_27 = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )


def savehistograms(hdf5name):
    with h5py.File('/net/luttero/data2/paper1/%s.hdf5'%hdf5name, 'a') as outfile:
        ### auxilliary data
        
        # cosmological parameters
        grp = outfile.create_group('cosmopars_eagle')
        cosmopars_ea_28 = mc.getcosmopars('L0100N1504', 28, 'REFERENCE', file_type='snap', simulation='eagle')
        cosmopars_ea_27 = mc.getcosmopars('L0100N1504', 27, 'REFERENCE', file_type='snap', simulation='eagle')
        grp_sub = grp.create_group('snap27')
        for key in cosmopars_ea_27.keys():
            grp_sub.attrs.create(key, cosmopars_ea_27[key])
        grp_sub = grp.create_group('snap28')
        for key in cosmopars_ea_28.keys():
            grp_sub.attrs.create(key, cosmopars_ea_28[key])
        del grp_sub
        
        # ion balance tables
        grp = outfile.create_group('ionbal_o7_snap27')
        o7_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o7',cosmopars_ea_27['z'])
        grp.create_dataset('ionbal', data=o7_ib_27)
        grp.create_dataset('log10_temperature_K', data=logTK_ib)
        grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        grp = outfile.create_group('ionbal_o8_snap27')
        o8_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o8',cosmopars_ea_27['z'])
        grp.create_dataset('ionbal', data=o8_ib_27)
        grp.create_dataset('log10_temperature_K', data=logTK_ib)
        grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        grp = outfile.create_group('ionbal_o7_snap28')
        o7_ib_28, logTK_ib, lognHcm3_ib = m3.findiontables('o7',cosmopars_ea_28['z'])
        grp.create_dataset('ionbal', data=o7_ib_28)
        grp.create_dataset('log10_temperature_K', data=logTK_ib)
        grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        grp = outfile.create_group('ionbal_o8_snap28')
        o8_ib_28, logTK_ib, lognHcm3_ib = m3.findiontables('o8',cosmopars_ea_28['z'])
        grp.create_dataset('ionbal', data=o8_ib_28)
        grp.create_dataset('log10_temperature_K', data=logTK_ib)
        grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        # cooling times at different metallicities (solar element ratios)
        simfile = sfc.Simfileclone(sfc.Zvar_rhoT_z0)
        vardict = m3.Vardict(simfile, 0, []) # Mass should not be read in, all other entries should be cleaned up

        logTvals   = np.log10(sfc.Tvals)
        logrhovals = np.log10(sfc.rhovals * m3.c.unitdensity_in_cgs)
        lognHvals_sol  = logrhovals + np.log10(sfc.dct_sol['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H)) 
        lognHvals_0p1  = logrhovals + np.log10( sfc.dct_0p1['Hydrogen']    / (m3.c.u*m3.c.atomw_H) )
        lognHvals_pri  = logrhovals + np.log10(sfc.dct_pri['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H))
    
        abundsdct_0p1 = sfc.dct_0p1
        # in simflieclone.Zvar_rhoT_z0, smoothed metallicites are primoridial and particle metallicites are solar
        outshape = (len(logrhovals), len(logTvals))
        tcool_perelt_sol = (m3.find_coolingtimes(vardict.simfile.z, vardict, method = 'per_element', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
        tcool_perelt_0p1 = (m3.find_coolingtimes(vardict.simfile.z, vardict, method = 'per_element', T4EOS=False, hab=abundsdct_0p1['Hydrogen'], abunds=abundsdct_0p1, last=False)).reshape(outshape)
        tcool_perelt_pri = (m3.find_coolingtimes(vardict.simfile.z, vardict, method = 'per_element', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
               
        grp = outfile.create_group('cooling_times_snap28')
        grp.create_group('units')
        grp['units'].attrs.create('Temperature', 'log10 K')
        grp['units'].attrs.create('Hydrogen number density nH', 'log10 cm^-3')
        grp['units'].attrs.create('Radiative cooling timescale tcool', 's')
        grp.attrs.create('info', 'subgroups have cooling times for different gas metallicities Z; element ratios are solar')

        grp_sub = grp.create_group('Z_solar')
        grp_sub.create_dataset('axis0_lognH', data=lognHvals_sol)
        grp_sub.create_dataset('axis1_logT', data=logTvals)
        grp_sub.create_dataset('tcool', data=tcool_perelt_sol)
        
        grp_sub = grp.create_group('Z_0p1solar')
        grp_sub.create_dataset('axis0_lognH', data=lognHvals_0p1)
        grp_sub.create_dataset('axis1_logT', data=logTvals)
        grp_sub.create_dataset('tcool', data=tcool_perelt_0p1)
        
        grp_sub = grp.create_group('Z_primordial')
        grp_sub.create_dataset('axis0_lognH', data=lognHvals_pri)
        grp_sub.create_dataset('axis1_logT', data=logTvals)
        grp_sub.create_dataset('tcool', data=tcool_perelt_pri)
        
        del grp_sub
        
        ### main histograms
        
        # o7, o8 by rho, T, snap28       
        hist_o7 = ea4o7
        hist_o8 = ea4o8
        lims_o7_T = [30, 35, 40, 45, 50] # add 5 for 10^2.5 K
        lims_o8_T = [30, 35, 40, 45, 50] # add 5 for 10^2.5 K
        lims_o7_rho = [33, 43, 53, 63] # densities -30.4 (~rhoav snap 28), + 1 dex, add 16 for rho = 10^-33 g/cm^3
        lims_o8_rho = [21, 31, 41, 51] # densities -30.4 (~rhoav snap 28) + 1 dex, 
        
        grp = outfile.create_group('coldens_o7_snap28_in_Tbins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_o7, saveaxes=2, axsumtuple=(1, lims_o7_T))
        
        grp = outfile.create_group('coldens_o7_snap28_in_rhobins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 g/cm^3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_o7, saveaxes=2, axsumtuple=(0, lims_o7_rho))
        
        grp = outfile.create_group('coldens_o8_snap28_in_Tbins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column_density_N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_o8, saveaxes=2, axsumtuple=(1, lims_o8_T))
        
        grp = outfile.create_group('coldens_o8_snap28_in_rhobins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 g/cm^3')
        grp_sub.attrs.create('Column_density_N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_o8, saveaxes=2, axsumtuple=(0, lims_o8_rho))
        
        del grp_sub
        del lims_o7_T
        del lims_o7_rho
        del lims_o8_T
        del lims_o8_rho
        
        ## o7, o7 by smoothed oxygen abundance
        hist = eafOSmo78
        lims_fO_o7 = [36, 41, 46, 51, 56]
        lims_fO_o8 = [36, 41, 46, 51, 56]
        
        grp = outfile.create_group('coldens_o7_snap28_in_Sm-fObins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('fO', 'mass fraction')
        grp_sub.attrs.create('fO_solar', ol.solar_abunds['oxygen'])
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist, saveaxes=0, axsumtuple=(2, lims_fO_o7))

        grp = outfile.create_group('coldens_o8_snap28_in_Sm-fObins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('fO', 'mass fraction')
        grp_sub.attrs.create('fO_solar', ol.solar_abunds['oxygen'])
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist, saveaxes=1, axsumtuple=(3, lims_fO_o8))
        del hist
        del lims_fO_o7
        del lims_fO_o8
        # o7, o8 by rho and T, snap28
        #lims = [44, 64, 74, 84, 94] # log10 N /cm^-2 = 12, 14, 15, 16, 17
        lims = [49, 59, 69, 79, 89] # log10 N /cm^-2 = 12, 14, 15, 16, 17

        grp = outfile.create_group('density_temperature_by_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Density', 'log10 g/cm^3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_o7, saveaxes=(0, 1), axsumtuple=(2, lims))
        
        grp = outfile.create_group('density_temperature_by_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Density', 'log10 g/cm^3')
        grp_sub.attrs.create('Column_density_N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_o8, saveaxes=(0, 1), axsumtuple=(2, lims))

        del lims
        
        # phase diagrams (just copy from npz files) keys: hist, logT, lognH
        mass = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_mass-weighted_normed.npz')    
        ox = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_O-mass-weighted_normed.npz')
        o7 = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_O7-mass-weighted_normed.npz')
        o8 = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_O8-mass-weighted_normed.npz')
        grp = outfile.create_group('Gas_histograms')
        
        grp_sub = grp.create_group('density_temperature_mass-histogram_snap28')
        dct = mass
        grp_sub.create_dataset('histogram', data=dct['hist'])
        grp_sub.create_dataset('axis0_log10_temperature_K', data=dct['logT'])
        grp_sub.create_dataset('axis1_log10_nH_cm-3', data=dct['lognH'])
        
        grp_sub = grp.create_group('density_temperature_oxygen-mass-histogram_snap28')
        dct = ox
        grp_sub.create_dataset('histogram', data=dct['hist'])
        grp_sub.create_dataset('axis0_log10_temperature_K', data=dct['logT'])
        grp_sub.create_dataset('axis1_log10_nH_cm-3', data=dct['lognH'])
        
        grp_sub = grp.create_group('density_temperature_o7-mass-histogram_snap28')
        dct = o7
        grp_sub.create_dataset('histogram', data=dct['hist'])
        grp_sub.create_dataset('axis0_log10_temperature_K', data=dct['logT'])
        grp_sub.create_dataset('axis1_log10_nH_cm-3', data=dct['lognH'])
        
        grp_sub = grp.create_group('density_temperature_o8-mass-histogram_snap28')
        dct = o8
        grp_sub.create_dataset('histogram', data=dct['hist'])
        grp_sub.create_dataset('axis0_log10_temperature_K', data=dct['logT'])
        grp_sub.create_dataset('axis1_log10_nH_cm-3', data=dct['lognH'])
        
        del dct
        del grp_sub
        
        # rho, T, fO, by O6, O7, O8 snap28       
        hist_rho_o78 = earhoo78
        hist_T_o78   = eaTo78 
        hist_fO_o78  = eafOo78 
        # same NOion edges for all three histograms
        lims_No7 = [76, 86, 96, 106, 116] 
        lims_No8 = [49, 59, 69, 79, 89] 
        
        grp = outfile.create_group('temperature_by_o7_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o78, saveaxes=(2, 3), axsumtuple=(1, lims_No8))
        
        grp = outfile.create_group('temperature_by_o7_o8_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o78, saveaxes=(2, 3), axsumtuple=(0, lims_No7))
        
        grp = outfile.create_group('density_by_o7_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o78, saveaxes=(2, 3), axsumtuple=(1, lims_No8))
        
        grp = outfile.create_group('density_by_o7_o8_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o78, saveaxes=(2, 3), axsumtuple=(0, lims_No7))
        
        grp = outfile.create_group('oxygenmassfraction_by_o7_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o78, saveaxes=(2, 3), axsumtuple=(1, lims_No8))
        
        grp = outfile.create_group('oxygenmassfraction_by_o7_o8_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o78, saveaxes=(2, 3), axsumtuple=(0, lims_No7))
        
        del lims_No7
        del lims_No8
        del grp_sub
        
        hist_rho_o67 = earhoo67_28
        hist_T_o67   = eaTo67_28 
        hist_fO_o67  = eafOo67_28
        lims_No6 = [26, 36, 46, 56, 66] 
        lims_No7 = [76, 86, 96, 106, 116] 
        
        grp = outfile.create_group('temperature_by_o6_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o67, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('temperature_by_o6_o7_snap28_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o67, saveaxes=(2, 3), axsumtuple=(0, lims_No6))
        
        grp = outfile.create_group('density_by_o6_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o67, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('density_by_o6_o7_snap28_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o67, saveaxes=(2, 3), axsumtuple=(0, lims_No6))
        
        grp = outfile.create_group('oxygenmassfraction_by_o6_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o67, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('oxygenmassfraction_by_o6_o7_snap28_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o67, saveaxes=(2, 3), axsumtuple=(0, lims_No6))
        
        del lims_No7
        del lims_No6
        del grp_sub
        
        # rho, T by Ne8, O7 snap 28
        
        hist_rho_o7ne8 = earhoo7ne8 
        hist_T_o7ne8 = eaTo7ne8
        lims_No7 = [76, 86, 96, 106, 116] 
        
        grp = outfile.create_group('temperature_by_ne8_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o7ne8, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('density_by_ne8_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o7ne8, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        del grp_sub
        
        # ion correlations: ne8, o6 with o7 and o8
        # o6, o7, o8
        lims = [41, 51, 61, 71, 81]
        grp = outfile.create_group('No7_No8_byNo6_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3o678, saveaxes=(1, 2), axsumtuple=(0, lims))
        # ne8, o7, o8
        lims = [30, 40, 50, 60, 70]
        grp = outfile.create_group('No7_No8_byNne8_in_Nne8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3ne8o78, saveaxes=(1, 2), axsumtuple=(0, lims))
        
        del lims
        del grp_sub
        
        # ion correlations: ne8, o6 with o7, o8
        # o6, o7, o8
        #ea3o678_16 
        #ea3ne8o78_16
        
        grp = outfile.create_group('No6_No7')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3o678_16, saveaxes=(0, 1))
        
        grp = outfile.create_group('No6_No8')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3o678_16, saveaxes=(0, 2))

        # ne8, o7, o8        
        grp = outfile.create_group('Nne8_No7')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3ne8o78_16, saveaxes=(0, 1))
        
        grp = outfile.create_group('Nne8_No8')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3ne8o78_16, saveaxes=(0, 2))
        
        grp = outfile.create_group('No7_No8')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3ne8o78_16, saveaxes=(1, 2))
        
        # neutral hydrogen, o7, o8
        grp = outfile.create_group('NHneutral_No7')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3_heutralo78, saveaxes=(0, 1))
        
        grp = outfile.create_group('NHneutral_No8')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3_heutralo78, saveaxes=(0, 2))


def savehistograms_lynxpds(hdf5name):
    with h5py.File('/home/wijers/Documents/papers/lynx_white_paper_ben/%s.hdf5'%hdf5name, 'a') as outfile:
        ### auxilliary data
        
        # cosmological parameters
        grp = outfile.create_group('cosmopars_eagle')
        #cosmopars_ea_28 = mc.getcosmopars('L0100N1504', 28, 'REFERENCE', file_type='snap', simulation='eagle')
        cosmopars_ea_27 = mc.getcosmopars('L0100N1504', 27, 'REFERENCE', file_type='snap', simulation='eagle')
        grp_sub = grp.create_group('snap27')
        for key in cosmopars_ea_27.keys():
            grp_sub.attrs.create(key, cosmopars_ea_27[key])
        #grp_sub = grp.create_group('snap28')
        #for key in cosmopars_ea_28.keys():
        #    grp_sub.attrs.create(key, cosmopars_ea_28[key])
        #del grp_sub
        
        # ion balance tables
        grp = outfile.create_group('ionbal_o7_snap27')
        o7_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o7',cosmopars_ea_27['z'])
        grp.create_dataset('ionbal', data=o7_ib_27)
        grp.create_dataset('log10_temperature_K', data=logTK_ib)
        grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        grp = outfile.create_group('ionbal_o8_snap27')
        o8_ib_27, logTK_ib, lognHcm3_ib = m3.findiontables('o8',cosmopars_ea_27['z'])
        grp.create_dataset('ionbal', data=o8_ib_27)
        grp.create_dataset('log10_temperature_K', data=logTK_ib)
        grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        #grp = outfile.create_group('ionbal_o7_snap28')
        #o7_ib_28, logTK_ib, lognHcm3_ib = m3.findiontables('o7',cosmopars_ea_28['z'])
        #grp.create_dataset('ionbal', data=o7_ib_28)
        #grp.create_dataset('log10_temperature_K', data=logTK_ib)
        #grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        #grp = outfile.create_group('ionbal_o8_snap28')
        #o8_ib_28, logTK_ib, lognHcm3_ib = m3.findiontables('o8',cosmopars_ea_28['z'])
        #grp.create_dataset('ionbal', data=o8_ib_28)
        #grp.create_dataset('log10_temperature_K', data=logTK_ib)
        #grp.create_dataset('log10_nH_cm-3', data=lognHcm3_ib)
        
        ### main histograms
        
        
        # rho, T, fO, by O6, O7, O8 snap28       
        hist_rhoT_o6 = ea3o6 
        
        # same NOion edges for all three histograms
        lims_No7 = [76, 86, 96, 106, 116] 
        lims_No8 = [49, 59, 69, 79, 89] 
        
        grp = outfile.create_group('temperature_by_o7_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o78, saveaxes=(2, 3), axsumtuple=(1, lims_No8))
        
        grp = outfile.create_group('temperature_by_o7_o8_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o78, saveaxes=(2, 3), axsumtuple=(0, lims_No7))
        
        grp = outfile.create_group('density_by_o7_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o78, saveaxes=(2, 3), axsumtuple=(1, lims_No8))
        
        grp = outfile.create_group('density_by_o7_o8_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o78, saveaxes=(2, 3), axsumtuple=(0, lims_No7))
        
        grp = outfile.create_group('oxygenmassfraction_by_o7_o8_snap28_in_No8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o78, saveaxes=(2, 3), axsumtuple=(1, lims_No8))
        
        grp = outfile.create_group('oxygenmassfraction_by_o7_o8_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o78, saveaxes=(2, 3), axsumtuple=(0, lims_No7))
        
        del lims_No7
        del lims_No8
        del grp_sub
        
        hist_rho_o67 = earhoo67_28
        hist_T_o67   = eaTo67_28 
        hist_fO_o67  = eafOo67_28
        lims_No6 = [26, 36, 46, 56, 66] 
        lims_No7 = [76, 86, 96, 106, 116] 
        
        grp = outfile.create_group('temperature_by_o6_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o67, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('temperature_by_o6_o7_snap28_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o67, saveaxes=(2, 3), axsumtuple=(0, lims_No6))
        
        grp = outfile.create_group('density_by_o6_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o67, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('density_by_o6_o7_snap28_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o67, saveaxes=(2, 3), axsumtuple=(0, lims_No6))
        
        grp = outfile.create_group('oxygenmassfraction_by_o6_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o67, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('oxygenmassfraction_by_o6_o7_snap28_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_fO_o67, saveaxes=(2, 3), axsumtuple=(0, lims_No6))
        
        del lims_No7
        del lims_No6
        del grp_sub
        
        # rho, T by Ne8, O7 snap 28
        
        hist_rho_o7ne8 = earhoo7ne8 
        hist_T_o7ne8 = eaTo7ne8
        lims_No7 = [76, 86, 96, 106, 116] 
        
        grp = outfile.create_group('temperature_by_ne8_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Temperature', 'log10 K')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_T_o7ne8, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        grp = outfile.create_group('density_by_ne8_o7_snap28_in_No7bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Density', 'log10 cm^-3')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, hist_rho_o7ne8, saveaxes=(2, 3), axsumtuple=(1, lims_No7))
        
        del grp_sub
        
        # ion correlations: ne8, o6 with o7 and o8
        # o6, o7, o8
        lims = [41, 51, 61, 71, 81]
        grp = outfile.create_group('No7_No8_byNo6_in_No6bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3o678, saveaxes=(1, 2), axsumtuple=(0, lims))
        # ne8, o7, o8
        lims = [30, 40, 50, 60, 70]
        grp = outfile.create_group('No7_No8_byNne8_in_Nne8bins')
        grp_sub = grp.create_group('units')
        grp_sub.attrs.create('Column density N', 'log10 cm^-2')
        savehist_compressed_as_hdf5(grp, ea3ne8o78, saveaxes=(1, 2), axsumtuple=(0, lims))
        
        del lims
        del grp_sub
        
        # ion correlations: ne8, o6 with o7, o8
        # o6, o7, o8
        #ea3o678_16 
        #ea3ne8o78_16

        
        
ionlabels = {'o6':  'O\, VI',\
             'o7':  'O\, VII',\
             'o8':  'O\, VIII',\
             'ne8': 'Ne\, VIII',\
             'c4':  'C\, IV',\
             'h1':  'H\, I',\
             }

def T200c_hot(M200c, z):
    # checked against notes from Joop's lecture: agrees pretty well at z=0, not at z=9 (proabably a halo mass definition issue)
    M200c *= m3.c.solar_mass # to cgs
    rhoc = (3./(8.*np.pi*c.gravity)* m3.Hubble(z)**2) # Hubble(z) will assume an EAGLE cosmology
    mu = 0.59 # about right for ionised (hot) gas, primordial
    R200c = (M200c/(200*rhoc))**(1./3.)
    return (mu * m3.c.protonmass) / (3. * m3.c.boltzmann) * m3.c.gravity * M200c/R200c
    
## settings/choices
fontsize=12

def handleinfedges(hist, setmin=-100., setmax=100.):
    for ei in range(len(hist['edges'])):
        if hist['edges'][ei][0] == -np.inf:
            hist['edges'][ei][0] = setmin
        if hist['edges'][ei][-1] == np.inf:
            hist['edges'][ei][-1] = setmax

def handleinfedges_dct(edges, setmin=-100., setmax=100.):
    for ei in edges.keys():
        if edges[ei][0] == -np.inf:
            edges[ei][0] = setmin
        if edges[ei][-1] == np.inf:
            edges[ei][-1] = setmax


def getlabel(hist,axis):
    if 'o8' in hist['dimension'][axis] or 'O8' in hist['dimension'][axis]:
        ionlab = 'O\,VIII'
    elif 'o7' in hist['dimension'][axis] or 'O7' in hist['dimension'][axis]:
        ionlab = 'O\,VII'
    elif 'o6' in hist['dimension'][axis] or 'O6' in hist['dimension'][axis]:
        ionlab = 'O\,VI'
    elif 'ne8' in hist['dimension'][axis] or 'Ne8' in hist['dimension'][axis]:
        ionlab = 'Ne\,VIII'
    elif 'hneutral' in hist['dimension'][axis] or 'Hneutral' in hist['dimension'][axis]:
        ionlab = 'H\, I + H_{2}'
    else: 
        ionlab = None
    
    if 'Temperature' in hist['dimension'][axis]:
        return r'$\log_{10} T \, [K], N_{\mathrm{%s}} \mathrm{-weighted}$'%ionlab
    elif 'Density' in hist['dimension'][axis]:
        return r'$\log_{10} \rho \, [\mathrm{g}\,\mathrm{cm}^{-3}], N_{\mathrm{%s}} \mathrm{-weighted}$'%ionlab 
    elif 'OxygenMassFrac_w_Mass' in hist['dimension'][axis]:
        return r'$\log_{10} f_{\mathrm{mass}}(\mathrm{O})$'
    elif 'OxygenMassFrac' in hist['dimension'][axis] and ionlab is not None:
        return r'$\log_{10} f_{\mathrm{N\, %s}}(\mathrm{O})$'%(ionlab)
    elif 'NO6' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{O\,VI}} \, [\mathrm{cm}^{-2}]$'
    elif 'NO7' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{O\,VII}} \, [\mathrm{cm}^{-2}]$'
    elif 'NO8' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{O\,VIII}} \, [\mathrm{cm}^{-2}]$'
    elif 'NNe8' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{Ne\,VIII}} \, [\mathrm{cm}^{-2}]$'
    elif 'NHneutral' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{H \, I} + \mathrm{H}_{2}} \, [\mathrm{cm}^{-2}]$'
    elif 'NH' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{H}} \, [\mathrm{cm}^{-2}]$'    
    else:
        print('No label found for axis %i: %s'%(axis, hist['dimension'][axis]))
	return None

## from stackexchange
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def simple_colormap(color, alpha=1.):
    rgba = mpl.colors.to_rgba(color, alpha=alpha)
    cvals = np.array([list(rgba[:3]) + [0.], rgba])
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'simple_%s'%(color), cvals)
    return new_cmap

normmax_bone = 0.35
normmax_gray = 0.7

bone_m = truncate_colormap(plt.get_cmap('bone_r'), maxval=normmax_bone)
gray_m = truncate_colormap(plt.get_cmap('gist_gray_r'), maxval=normmax_gray)

def linterpsolve(xvals, yvals, xpoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below xpoint 
    xvals, yvals: 1D arrays
    xpoint: float
    '''
    if np.all(np.diff(xvals) > 0.):
        incr = True
    elif np.all(np.diff(xvals) < 0.):
        incr = False
    else:
        print('linterpsolve only works for monotonic functions')
        return None
    ind1 = np.where(xvals <= xpoint)[0]
    ind2 = np.where(xvals >= xpoint)[0]
    #print(ind1)
    #print(ind2)
    if len(ind2) == 0 or len(ind1) == 0:
        print('xpoint is outside the bounds of xvals')
        return None
    if incr:
        ind1 = np.max(ind1)
        ind2 = np.min(ind2)
    else:
        ind1 = np.min(ind1)
        ind2 = np.max(ind2)
    #print('Indices x: %i, %i'%(ind1, ind2))
    #print('x values: lower %s, upper %s, searched %s'%(xvals[ind1], xvals[ind2], xpoint))
    if ind1 == ind2:
        ypoint = yvals[ind1]
    else:
        w = (xpoint - xvals[ind1]) / (xvals[ind2] - xvals[ind1]) #weight
        ypoint = yvals[ind2] * w + yvals[ind1] * (1. - w)
    #print('y values: lower %s, upper %s, solution: %s'%(yvals[ind1], yvals[ind2], ypoint))
    return ypoint

def find_intercepts(yvals, xvals, ypoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below ypoint 
    xvals, yvals: 1D arrays
    ypoint: float
    Does not distinguish between intersections separated by less than 2 xvals points
    '''
    if not (np.all(np.diff(xvals) < 0.) or np.all(np.diff(xvals) > 0.)):
        print('linterpsolve only works for monotonic x values')
        return None
    zerodiffs = yvals - ypoint
    leqzero = np.where(zerodiffs <= 0.)[0]
    if len(leqzero) == 0:
        return np.array([])
    elif len(leqzero) == 1:
        edges = [[leqzero[0], leqzero[0]]]
    else:
        segmentedges = np.where(np.diff(leqzero) > 1)[0] + 1
        if len(segmentedges) == 0: # one dip below zero -> edges are intercepts
            edges = [[leqzero[0], leqzero[-1]]]
        else:
            parts = [leqzero[: segmentedges[0]] if si == 0 else \
                     leqzero[segmentedges[si - 1] : segmentedges[si]] if si < len(segmentedges) else\
                     leqzero[segmentedges[si - 1] :] \
                     for si in range(len(segmentedges) + 1)]
            edges = [[part[0], part[-1]] for part in parts]
    intercepts = [[linterpsolve(zerodiffs[ed[0]-1: ed[0] + 1], xvals[ed[0]-1: ed[0] + 1], 0.),\
                   linterpsolve(zerodiffs[ed[1]: ed[1] + 2],   xvals[ed[1]: ed[1] + 2], 0.)]  \
                  if ed[0] != 0 and ed[1] != len(yvals) - 1 else \
                  [None,\
                   linterpsolve(zerodiffs[ed[1]: ed[1] + 2],   xvals[ed[1]: ed[1] + 2], 0.)] \
                  if ed[1] != len(yvals) - 1 else \
                  [linterpsolve(zerodiffs[ed[0]-1: ed[0] + 1], xvals[ed[0]-1: ed[0] + 1], 0.),\
                   None]  \
                  if ed[0] != 0 else \
                  [None, None]
                 for ed in edges]
    intercepts = [i for i2 in intercepts for i in i2]
    if intercepts[0] is None:
        intercepts = intercepts[1:]
    if intercepts[-1] is None:
        intercepts = intercepts[:-1]
    return np.array(intercepts)

def add_2dplot(ax,hist3d,toplotaxes,log=True,usepcolor = False, pixdens = False, shiftx = 0., shifty = 0., **kwargs):
    # hist3d can be a histogram of any number >=2 of dimensions
    # like in plot1d, get the number of axes from the length of the edges array
    # usepcolor: if edges arrays are not equally spaced, imshow will get the ticks wrong
    summedaxes = tuple(list( set(range(len(hist3d['edges'])))-set(toplotaxes) )) # get axes to sum over
    toplotaxes= list(toplotaxes)
    #toplotaxes.sort()
    axis1, axis2 = tuple(toplotaxes)
    # sum over non-plotted axes
    if len(summedaxes) == 0:
        imgtoplot = hist3d['bins']
    else:
        imgtoplot = np.sum(hist3d['bins'],axis=summedaxes)
    
    
    if pixdens:
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(hist3d['edges'][toplotaxes[0]]), np.diff(hist3d['edges'][toplotaxes[1]]) ] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])

        if axis1 > axis2:
            imgtoplot = imgtoplot.T
        imgtoplot /= normmatrix
        if axis1 > axis2:
            imgtoplot = imgtoplot.T
        del normmatrix
        
    if log:
        imgtoplot = np.log10(imgtoplot)
    # transpose plot if axes not in standard order; normally, need to use transposed array in image
    if axis1 < axis2:
        imgtoplot = imgtoplot.T
    if usepcolor:
        img = ax.pcolormesh(hist3d['edges'][axis1] + shiftx, hist3d['edges'][axis2] + shifty, imgtoplot, **kwargs)
    else:
        img = ax.imshow(imgtoplot,origin='lower',interpolation='nearest',extent=(hist3d['edges'][axis1][0]+shiftx, hist3d['edges'][axis1][-1]+shiftx, hist3d['edges'][axis2][0]+shifty, hist3d['edges'][axis2][-1]+shifty),**kwargs)
    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
    else:
        vmin = np.min(imgtoplot[np.isfinite(imgtoplot)])
    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
    else:
        vmax = np.max(imgtoplot[np.isfinite(imgtoplot)])
    return img, vmin, vmax

def getminmax2d(hist3d, axis=None, log=True, pixdens=False): 
    # axis = axis to sum over; None -> don't sum over any axes 
    # now works for histgrams of general dimensions
    if axis is None:
        imgtoplot = hist3d['bins']
    else:
        imgtoplot = np.sum(hist3d['bins'],axis=axis)
    if pixdens:
        if axis is None:
            naxis = range(len(hist3d['edges']))
        else:
            if not hasattr(axis, '__len__'):
                saxis = [axis]
            else:
                saxis = axis
            naxis = list(set(range(len(hist3d['edges']))) - set(saxis)) # axes not to sum over
        naxis.sort() 
        numdims = len(naxis)
        binsizes = [np.diff(hist3d['edges'][axisi]) for axisi in naxis] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])
        imgtoplot /= normmatrix
        del normmatrix
    finite = np.isfinite(imgtoplot)
    if log:
        imin = np.min(imgtoplot[np.logical_and(finite, imgtoplot > 0) ])
        imax = np.max(imgtoplot[np.logical_and(finite, imgtoplot > 0) ])
        imin = np.log10(imin)
        imax = np.log10(imax)
    else:
        imin = np.min(imgtoplot[np.isfinite(imgtoplot)])
        imax = np.max(imgtoplot[np.isfinite(imgtoplot)])
    return imin, imax

def add_1dplot(ax,hist3d,axis1,log=True,**kwargs):
    # edges is an object array, contains array for each dimension -> number of dimesions is length of edges
    # modified so this works for any dimension of histogram
    chosenaxes = list(set(range(len(hist3d['edges'])))-set([axis1])) # get axes to sum over: all except axis1
    if len(chosenaxes) == 0:
        bins = hist3d['bins']
    else: # if histogram has more than 1 dimension, sum over the others
        bins = np.sum(hist3d['bins'],axis=tuple(chosenaxes))
    # plot the histogram on ax 	
    ax.step(hist3d['edges'][axis1][:-1],bins,where = 'post',**kwargs)
    if log:
        ax.set_yscale('log')

def add_colorbar(ax,img=None,vmin=None,vmax=None,cmap=None,clabel=None,newax=False,extend='neither',fontsize=fontsize,orientation='vertical'):
    if img is None:
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,extend=extend,orientation=orientation)
    elif newax:
        div = axgrid.make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.2)
        cbar = mpl.colorbar.Colorbar(cax,img,extend=extend)
    else:
        cbar = mpl.colorbar.Colorbar(ax,img,extend=extend,orientation=orientation)
    ax.tick_params(labelsize=fontsize-2)
    if clabel is not None:
        cbar.set_label(clabel,fontsize=fontsize)

def add_ax2rho(ax,xory = 'x',fontsize=fontsize, labelax = True):

    if xory == 'x':
        ax2 = ax.twiny()
        old_ticklocs = ax.get_xticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
	
	# use same spacing and number of ticks, but start at first integer value in the new units
    new_lim = old_xlim + np.log10(rho_to_nh)
    numticks = len(old_ticklocs)
    newticks = np.ceil(new_lim[0]) + np.array(old_ticklocs) - old_ticklocs[0]
    newticks = np.round(newticks,2)
    newticklabels = [str(int(tick)) if int(tick)== tick else str(tick) for tick in newticks]
   	
	#print old_ticklocs
    print newticklabels
        #ax2.set_xticks(np.round(old_ticklocs + np.log10(rho_to_nh),2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        #ax2.set_xticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)]) 
    ax2.set_xticks(newticks - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
    ax2.set_xticklabels(newticklabels)    
    if labelax:         
        ax2.set_xlabel(r'$\log_{10} n_H \, [\mathrm{cm}^{-3}], f_H = 0.752$',fontsize=fontsize)
        ax2.set_xlim(old_xlim)
        ax2.set_ylim(old_ylim)
    else:
        ax2 = ax.twinx()
        old_ticklocs = ax.get_yticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
        ax2.set_yticks(np.round(old_ticklocs + np.log10(rho_to_nh),2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        ax2.set_yticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)])        
        if labelax:
            ax2.set_ylabel(r'$\log_{10} n_H \, [\mathrm{cm}^{-3}], f_H = 0.752$',fontsize=fontsize)
        ax2.set_xlim(old_xlim)
        ax2.set_ylim(old_ylim)
    ax2.tick_params(labelsize=fontsize,axis='both')
    return ax2

def add_rhoavx(ax,onlyeagle=False,eacolor='lightgray',bacolor='gray',userhob_ea = None, **kwargs):
    if onlyeagle:
        ealabel = r'$\overline{\rho_b}$'
    else:
        ealabel = r'EAGLE $\overline{\rho_b}$'
	 
    if userhob_ea is None:
        rhob_ea = logrhob_av_ea
    else:
        rhob_ea = userhob_ea
    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 1
    ax.axvline(x=rhob_ea,ymin=0.,ymax=1.,color=eacolor, label=ealabel, **kwargs)
    if not onlyeagle:
        ax.axvline(x=logrhob_av_ba,ymin=0.,ymax=1.,color=bacolor,linewidth=1,label = r'BAHAMAS $\overline{\rho_b}$')

def add_fsolx(ax, ion, color = 'gray', label = 'solar'):
    value = np.log10(ol.solar_abunds[ol.elements_ion[ion]])
    ax.axvline(x=value,ymin=0.,ymax=1.,color=color,linewidth=1,label = label)

def add_fsoly(ax, ion, color = 'gray', label = 'solar fraction'):
    value = np.log10(ol.solar_abunds[ol.elements_ion[ion]])
    ax.axhline(y=value,xmin=0.,xmax=1.,color=color,linewidth=1,label = label) 

def add_collsional_ionbal(ax,ib = None,**kwargs):
    if ib is None:
        ib = o7_ib
    ax.plot(logTK_ib,ib[-1,:],**kwargs)
    ax.set_ylabel('fraction of O atoms in O VII state',fontsize=fontsize)

def add_ionbal_contours(ax,legend=True,ib=None, levels = None,  colors = None , reset_lim = True, legend_labelsonly=False, **kwargs):
    if ib is None:
        ib = o7_ib
    if reset_lim:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    logrho, logT = np.meshgrid(logrhocgs_ib, logTK_ib)
    if levels is None:        
        levels = [1e-3,3e-2, 0.3, 0.9]
    if colors is None:
        colors = ['mediumvioletred','magenta','orchid','palevioletred']
    if 'fontsize' in kwargs.keys():
        fontsize = kwargs['fontsize']
        del kwargs['fontsize']
    else:
        fontsize = None
    contours = ax.contour(logrho, logT, ib.T, levels, colors = colors, **kwargs)
    if reset_lim:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    # make a legend to avoid crowding plot region
    if not legend_labelsonly:
        for i in range(len(levels)):
            contours.collections[i].set_label('%.0e'%levels[i])
    if legend:
        legend = ax.legend(loc='lower right', fontsize=fontsize)
        legend.set_title(r'$f_{\mathrm{O VII}}, f_H=0.752$', prop={'size':fontsize})
    #ax.clabel(contours, inline=True, fontsize=fontsize, inline_spacing = 0,manual = [(-28.,6.7),(-26.,6.5),(-24.,6.3),(-27.,5.5),(-28.5,4.5),(-25.,5.2)])

def add_ionbal_img(ax, ib=None, **kwargs):
    # this is intended as a background: keep the limits of the data plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # easy to change to options if I want o8 or something later
    if ib is None:
        ionbal = o7_ib
    else:
        ionbal = ib
    logT = logTK_ib
    logrho = logrhocgs_ib

    # color bar handling (defaults etc. are handled here -> remove from kwargs before passing to imshow)
    if 'cmap' in kwargs:
        cmap = mpl.cm.get_cmap(kwargs['cmap'])
        del kwargs['cmap'] 
    else: # default color map
        cmap = mpl.cm.get_cmap('gist_gray')
    cmap.set_under(cmap(0))
    ax.set_facecolor(cmap(0))
    
    # to get extents, we need bin edges rather than centres; tables have evenly spaced logrho, logT
    logrho_diff = np.average(np.diff(logrho))
    logrho_ib_edges = np.array(list(logrho - logrho_diff/2.) \
                                  + [logrho[-1] + logrho_diff/2.])
    logT_diff = np.average(np.diff(logT))				  
    logT_ib_edges = np.array(list(logT - logT_diff/2.) \
                                  + [logT[-1] + logT_diff/2.])
				  
    #img = ax.imshow(np.log10(ionbal.T),origin='lower',interpolation='nearest',extent=(logrho_ib_edges[0],logrho_ib_edges[-1],logT_ib_edges[0],logT_ib_edges[-1]),cmap=cmap,**kwargs)
    img = ax.pcolormesh(logrho_ib_edges, logT_ib_edges, np.log10(ionbal.T), cmap=cmap, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')
    
    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
    else:
        vmin = np.min(ionbal[np.isfinite(ionbal)])
    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
    else:
        vmax = np.max(ionbal[np.isfinite(ionbal)])
    return img, vmin, vmax

    
def add_2dhist_contours(ax,hist,toplotaxes,mins= None, maxs=None, histlegend=True, fraclevels=True, levels=None, legend = True, dimlabels=None,legendlabel = None, legendlabel_pre = None, shiftx = 0., shifty = 0., dimshifts=None, **kwargs):
    '''
    colors, linestyles: through kwargs
    othersmin and othersmax should be indices along the corresponding histogram axes
    assumes xlim, ylim are already set as desired
    dimlabels can be used to override (long) dimension labels from hist
    '''
    # get axes to sum over; preserve order of other axes to match limits
    
        
    summedaxes = range(len(hist['edges']))
    summedaxes.remove(toplotaxes[0])
    summedaxes.remove(toplotaxes[1])
    
    # handle some defaults
    bins = hist['bins']
    edges = hist['edges']
    
    #print('min/max per edge: %s'%str([(mins[i], maxs[i], len(edges[i])) for i in [0,1,2]]))
    #print('mins: %s, maxs: %s'%(mins, maxs))
    
    if dimlabels is None:
        dimlabels = [getlabel(hist,axis) for axis in range(len(edges))]    
    if mins is None:
        mins= (None,)*len(edges)
    if maxs is None:
        maxs = (None,)*len(edges)
    if dimshifts is None:
        dimshifts = (0.,) * len(edges)
	
    # get the selection of min/maxs and apply the selection, put axes in the desired order
    sels = [slice(mins[i],maxs[i],None) for i in range(len(edges))]
    sels = tuple(sels)
    
    if len(summedaxes) >0:
        binsum = np.sum(bins[sels],axis=tuple(summedaxes))
    else:
        binsum = bins[sels]
    if toplotaxes[0] > toplotaxes[1]:
        binsum = binsum.transpose()
    #print('min/max binsum: %.4e, %.4e'%(np.min(binsum),np.max(binsum)))
    
    binfrac = np.sum(binsum)/np.sum(bins) # fraction of total bins selected
    # add min < dimension_quantity < max in legend label
    if legendlabel is None:
        labelparts = [r'%.1f $<$ %s $<$ %.1f, '%(edges[i][mins[i]] + dimshifts[i], dimlabels[i], edges[i][maxs[i]] + dimshifts[i]) if (mins[i] is not None and maxs[i] is not None) else\
                      r'%.1f $<$ %s, '%(edges[i][mins[i]] + dimshifts[i], dimlabels[i])                  if (mins[i] is not None and maxs[i] is None)     else\
		      r'%s $<$ %.1f, '%(dimlabels[i], edges[i][maxs[i]] + dimshifts[i])                  if (mins[i] is None and maxs[i] is not None)     else\
		      '' for i in range(len(edges))] #no label if there is no selection on that dimension
        legendlabel = ''.join(labelparts)
        # add percentage of total histogram selected
        if legendlabel[-2:] == ', ':
            legendlabel = legendlabel[:-2] + ': '
        legendlabel += '%.1f%%'%(100.*binfrac) 

    if legendlabel_pre is not None:
        legendlabel = legendlabel_pre + legendlabel
    
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    if levels is None:
        if fraclevels:
            levels = [1., 0.9, 0.5] # enclosed fractions for each level (approximate)
        else:
	        levels = [1e-3,3e-2,0.1,0.5]

    if fraclevels: # assumes all levels are between 0 and 1
        binsum = binsum/np.sum(binsum) # redo normalisation for the smaller dataset
        #print('min/max binsum: %.4e, %.4e'%(np.min(binsum),np.max(binsum)))
        
        # for sorting, normialise bins by bin size: peak finding depends on density, should not favour larger bins
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(hist['edges'][toplotaxes[0]]), np.diff(hist['edges'][toplotaxes[1]]) ] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])

        binsumcopy = binsum.copy() # copy to rework
        bindens    = binsumcopy/normmatrix
        bindensflat= bindens.copy().reshape(np.prod(bindens.shape)) # reshape creates views; argsorting later will mess up the array we need for the plot
        binsumcopy = binsumcopy.reshape(np.prod(binsumcopy.shape))
        binsumcopy = binsumcopy[np.argsort(bindensflat)] # get all histogram values in order of histogram density (low to high)
        
        binsumcopy = np.flipud(binsumcopy) # flip to high-to-low
        cumul = np.cumsum(binsumcopy) # add values high-to-low 
        wherelist = [[(np.where(cumul<=level))[0],(np.where(cumul>=level))[0]] for level in levels] # list of max-lower and min-higher indices

        ### made for using bin counts -> binsumcopy is ordered y its own values
	    # sort out list: where arrays may be empty -> levels outside 0,1 range, probabaly
	    # set value level 0 for level == 1. -> just want everything (may have no cumulative values that large due to fp errors)
	    # if all cumulative values are too high (maxmimum bin has too high a fraction), set to first cumulative value (=max bin value)
	    # otherwise: interpolate values, or use overlap
	    #print binsumcopy, binsumcopy.shape
	    #print cumul
	    #print wherelist
	    #return levels, cumul, binsumcopy, wherelist
        if np.all(normmatrix == normmatrix[0,0]): # all bins are the same size
            valslist = [cumul[0]  if  wherelist[i][0].shape == (0,) else\
	                    0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		                np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
                                               np.array([ binsumcopy[wherelist[i][0][-1]], binsumcopy[wherelist[i][1][0]] ]) )[0]\
		                for i in range(len(levels))]
            pltarr = binsum
        else: # find a reasonable interpolation of bindens in stead; need to plot the contours in binsdens as well, in this case
            bindensflat.sort() # to match cumul array indices: sort, then make high to low
            bindensflat = bindensflat[::-1]
            valslist = [bindensflat[0]  if  wherelist[i][0].shape == (0,) else\
	                    0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		                np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
		                                                 np.array([ bindensflat[wherelist[i][0][-1]], bindensflat[wherelist[i][1][0]] ]))[0]\
		                for i in range(len(levels))]
            pltarr = bindens
        #print('min/max bindens: %.4e, %f, min/max flat: %.4e, %f'%(np.min(bindens),np.max(bindens),np.min(bindensflat),np.max(bindensflat)))
        #print('binsum shape: %s, bindens shape: %s, normmatrix shape: %s,  x: %i, y: %i'%(str(binsum.shape),str(bindens.shape), str(normmatrix.shape), len(hist['edges'][toplotaxes[0]]), len(hist['edges'][toplotaxes[1]])))
        #print('wherelist: %s'%wherelist)
        #plt.subplot(2,1,1)
        #plt.pcolor(hist['edges'][toplotaxes[0]], hist['edges'][toplotaxes[1]], np.log10(binsum.T), vmin = np.min( np.log10(binsum.T)[np.isfinite(np.log10(binsum.T))]))
        #plt.colorbar()
        #plt.subplot(2,1,2)
        #plt.pcolor(hist['edges'][toplotaxes[0]], hist['edges'][toplotaxes[1]], np.log10(bindens.T), vmin = np.min( np.log10(bindens.T)[np.isfinite(np.log10(bindens.T))]))
        #plt.colorbar()
        #plt.show()
        
        del normmatrix
        del binsumcopy
        del binsum
        del bindens
        del bindensflat
        #for i in range(len(levels)):
        #    if not (wherelist[i][0].shape == (0,) or wherelist[i][1].shape == (0,)):
	    #        print('interpolating (%f, %f) <- index %i and (%f, %f)  <- index %i to %f'\
	    #	 %(cumul[wherelist[i][0][-1]],binsumcopy[wherelist[i][0][-1]],wherelist[i][0][-1],\
	    #          cumul[wherelist[i][1][0]], binsumcopy[wherelist[i][1][0]], wherelist[i][1][0],\
    	#	   levels[i]) )
        #print(np.all(np.diff(binsumcopy)>=0.))
        uselevels = np.copy(valslist)
        # check for double values; fudge slightly if levels are the same
        anyequal = np.array([np.array(valslist) == val for val in valslist])
        if np.sum(anyequal) > len(valslist): # any levels equal to a *different* level
            eqvals = [np.where(anyequal[ind])[0] for ind in range(len(valslist))] # what levels any value is equal to
            eqgroups = set([tuple(list(eq)) for eq in eqvals]) # get the sets of unique values
            eqgroups = list(eqgroups)
            fudgeby = 1.e-8
            grouplist = [(np.where(np.array([ind in group for group in eqgroups]))[0])[0] for ind in range(len(valslist))] # which group is each uselevel index in
            groupindlist = [(np.where(ind == np.array(eqgroups[grouplist[ind]]))[0])[0] for ind in range(len(valslist))] # which group index corresponds to a goven uselevel index
            addto = [[valslist[group[0]]*fudgeby*ind for ind in range(len(group))] for group in eqgroups] #add nothing for single-element groups
                            
            valslist = [uselevels[ind] + addto[grouplist[ind]][groupindlist[ind]] for ind in range(len(valslist))]
            print('Desired cumulative fraction levels were %s; using value levels %s fudged from %s'%(levels, valslist, uselevels))
            uselevels = valslist
        else:
            print('Desired cumulative fraction levels were %s; using value levels %s'%(levels,uselevels))
    else:
        uselevels=levels
    
    removezerolevelprops = False
    if len(uselevels) > 1:
        if uselevels[0] ==uselevels[1]:
            uselevels = uselevels[1:]
            removezerolevelprops = True
            
    #print binsum, binsum.shape
    if 'linestyles' in kwargs:        
        linestyles = kwargs['linestyles']
    else:
        linestyles = [] # to not break the legend search
    
    if removezerolevelprops: # a duplicate level was kicked out -> remove properties for that level
        if 'linestyles' in kwargs.keys():
            kwargs['linestyles'] = kwargs['linestyles'][1:]
        if 'colors' in kwargs.keys():
            kwargs['colors'] = kwargs['colors'][1:]
            
    # get pixel centres from edges
    centres0 = edges[toplotaxes[0]][:-1] + shiftx + 0.5*np.diff(edges[toplotaxes[0]]) 
    centres1 = edges[toplotaxes[1]][:-1] + shifty + 0.5*np.diff(edges[toplotaxes[1]])
    contours = ax.contour(centres0, centres1, pltarr.T, uselevels, **kwargs)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    # make a legend to avoid crowding plot region
    #for i in range(len(levels)):
    #    contours.collections[i].set_label('%.0e'%levels[i])
    # color only legend; get a solid line in the legend
    
    #ax.tick_params(labelsize=fontsize,axis='both')
    if 'solid' in linestyles:
        contours.collections[np.where(np.array(linestyles)=='solid')[0][0]].set_label(legendlabel)
    else: # just do the first one
        contours.collections[0].set_label(legendlabel)
    if histlegend:
        ax.legend(loc='lower right',title=r'$f_{\mathrm{O VII}}, f_H=0.752$')
    

def dummyformatter(x, pos):
    return ''
dummyformatter = plt.FuncFormatter(dummyformatter)

def setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False):
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=right, top=top, axis='both', which='both', color=color,\
                   labelleft=labelleft, labeltop=labeltop, labelbottom = labelbottom, labelright=labelright)


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



##########################################
####### BAHAMAS/EAGLE COMPARISONS ########
##########################################

def plotrhohists(fixz=False):
    if fixz:
        ba = h3ba_fz
        eahi = h3eahi_fz
        tail = '_fixz'
    else:
        ba = h3ba
        eahi = h3eahi
        eami = h3eami
        tail = ''
    fig, ax = plt.subplots(nrows=1,ncols=1)
    add_1dplot(ax,ba,0,color='blue',label='BAHAMAS')
    add_1dplot(ax,eahi,0,color='red',label='EAGLE, 32k pix')
    if not fixz:
        add_1dplot(ax,eami,0,color='orange',label='EAGLE, 5600 pix',linestyle = 'dashed')
    ax.set_xlabel(getlabel(h3ba,0),fontsize=fontsize)
    ax.set_ylabel('fraction of pixels',fontsize=fontsize)
    add_rhoavx(ax)
    ax.legend(fontsize=fontsize)
    ax2 = add_ax2rho(ax)
    setticks(ax, fontsize, top=False, labeltop=False)
    setticks(ax2, fontsize, top=True, labeltop=True, bottom=False, left=False, right=False, labelright=False)
    plt.savefig(mdir + 'bahamas_comp/rho_histograms%s.pdf'%tail,format = 'pdf',bbox_inches='tight')

def plotThists(fixz=False):
    if fixz:
        ba = h3ba_fz
        eahi = h3eahi_fz
        tail = '_fixz'
    else:
        ba = h3ba
        eahi = h3eahi
        eami = h3eami
        tail = ''
    fig, ax = plt.subplots(nrows=1,ncols=1)
    add_1dplot(ax,ba,1,color='blue',label='BAHAMAS')
    add_1dplot(ax,eahi,1,color='red',label='EAGLE, 32k pix')
    if not fixz:
        add_1dplot(ax,eami,1,color='orange',label='EAGLE, 5600 pix',linestyle = 'dashed')
    ax.set_xlabel(getlabel(h3ba,1),fontsize=fontsize)
    ax.set_ylabel('fraction of pixels',fontsize=fontsize)

    ax2 = ax.twinx()
    add_collsional_ionbal(ax2,color='gray',linestyle='dotted',label='coll. ion. eq.')
    ax2.set_ylim(1e-9,None)
    ax2.set_yscale('log')

    ax.set_xlim(3.,8.)
    ax2.set_xlim(ax.get_xlim())

    ax.legend(fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    setticks(ax, fontsize, right=False, labelright=False)
    setticks(ax2, fontsize, top=False, bottom=False, left=False, right=True, labelright=True)
    plt.savefig(mdir + 'bahamas_comp/T_histograms%s.pdf'%tail,format = 'pdf',bbox_inches='tight')

def plotNO7hists(usehhists=False, fixz=False):
    fig, ax = plt.subplots(nrows=1,ncols=1)
    if usehhists:
        add_1dplot(ax,h3bah,0,color='blue',label='BAHAMAS, z')
        add_1dplot(ax,h3eahih,0,color='red',label='EAGLE, xyz, 32k pix')
        add_1dplot(ax,h3eamih,0,color='orange',label='EAGLE, xyz, 5600 pix',linestyle = 'dashed')
        ax.set_xlabel(getlabel(h3bah,0),fontsize=fontsize)
        ax.set_ylabel('fraction of pixels',fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_xlim(7.,None)
        setticks(ax, fontsize)
        plt.savefig(mdir + 'NO7_histograms_from_o7-hydrogen.png',format = 'png',bbox_inches='tight')
    elif fixz:
        add_1dplot(ax,h3ba_fz,2,color='blue',label='BAHAMAS, z')
        add_1dplot(ax,h3eahi_fz,2,color='red',label='EAGLE, 32k pix')
        ax.set_xlabel(getlabel(h3ba_fz,2),fontsize=fontsize)
        ax.set_ylabel('fraction of pixels',fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_xlim(13.0,None)
        setticks(ax, fontsize)
        plt.savefig(mdir + 'bahamas_comp/NO7_histograms_fixz.pdf',format = 'pdf',bbox_inches='tight')
    else: # no fixz, NO7-NH histograms
        add_1dplot(ax,h3ba,2,color='blue',label='BAHAMAS')
        add_1dplot(ax,h3eahi,2,color='red',label='EAGLE, 32k pix')
        add_1dplot(ax,h3eami,2,color='orange',label='EAGLE, 5600 pix',linestyle = 'dashed')
        ax.set_xlabel(getlabel(h3ba,2),fontsize=fontsize)
        ax.set_ylabel('fraction of pixels',fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_xlim(7.,None)
        setticks(ax, fontsize)
        plt.savefig(mdir + 'bahamas_comp/NO7_histograms.pdf',format = 'pdf',bbox_inches='tight')
    
def plotNHhists():
    fig, ax = plt.subplots(nrows=1,ncols=1)
    add_1dplot(ax,h3bah,1,color='blue',label='BAHAMAS, z')
    add_1dplot(ax,h3eahih,1,color='red',label='EAGLE, xyz, 32k pix')
    add_1dplot(ax,h3eamih,1,color='orange',label='EAGLE, xyz, 5600 pix',linestyle = 'dashed')
    ax.set_xlabel(getlabel(h3bah,1),fontsize=fontsize)
    ax.set_ylabel('fraction of pixels',fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    setticks(ax, fontsize)
    #ax.set_xlim(7.,None)
    plt.savefig(mdir + 'NH_histograms.png',format = 'png',bbox_inches='tight')
        


def subplotrhoT(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ionballegend=True, fontsize=None, **kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(0,1),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax2 = add_ax2rho(ax,xory = 'x', fontsize=fontsize)
    add_rhoavx(ax)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.15))
    # set the plot to be square
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()  
    add_ionbal_contours(ax2,label='O7 fractions',legend=ionballegend, fontsize=fontsize) # will generally reset ax limits -> reset
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim1)  
    ax2.set_ylim(ylim1)
    ax2.set_xlim(xlim1)
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    ax2.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    setticks(ax, fontsize, top=False)
    setticks(ax2, fontsize, left=False, right=False, labelright=False, bottom=False, top=True, labeltop=True)
    if mainlegend:
        ax.legend(loc= 'lower right', fontsize=fontsize)
    return img, vmin, vmax

def plotrhoT(fixz=False):
    if fixz:
        ba = h3ba_fz
        eahi = h3eahi_fz
        tail = '_fixz'
        fig = plt.figure(figsize=(13.,5.))
        grid = gsp.GridSpec(1,3,width_ratios=[5.,5.,1.])
        ax1, ax2, cax = tuple(plt.subplot(grid[i]) for i in range(3)) 
    else:
        ba = h3ba
        eahi = h3eahi
        eami = h3eami
        tail = ''
        fig = plt.figure(figsize=(16.,5.))
        grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.])
        ax1, ax2, ax3, cax = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(ba, 2)
    imin2,imax2 = getminmax2d(eahi, 2)
    if fixz:
        vmin = min([imin1,imin2])
        vmax = max([imax1,imax2])
    else:
        imin3,imax3 = getminmax2d(eami, 2)
        vmin = min([imin1,imin2,imin3])
        vmax = max([imax1,imax2,imax3])
    fontsize = 16
    
    cmap = 'nipy_spectral'
    img, vmin, vmax = subplotrhoT(ax1, ba ,title= 'BAHAMAS',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ionballegend=False, fontsize=fontsize)
    subplotrhoT(ax2, eahi ,title= 'EAGLE, 32k pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ionballegend=True, fontsize=fontsize)
    if not fixz:
        subplotrhoT(ax3, eami ,title= 'EAGLE, 5600 pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ionballegend=False, fontsize=fontsize)
    
    add_colorbar(cax,img=img,clabel=r'$\log_{10}$ fraction of pixels', fontsize=fontsize)
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'bahamas_comp/phase_diagrams%s.pdf'%tail,format = 'pdf',bbox_inches='tight')



def subplotrhoNO7(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ionballegend=True,ymin=None,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(0,2),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax2 = add_ax2rho(ax,xory = 'x')
    add_rhoavx(ax)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.15))
    # set the plot to be square
    if ymin is not None:
        ax.set_ylim(ymin,ax.get_ylim()[1])
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim() 
    #print ylim1 
    #ax.set_xlim(xlim1)
    #ax.set_ylim(ylim1)  
    ax2.set_ylim(ylim1)
    ax2.set_xlim(xlim1)
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    ax2.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    setticks(ax, fontsize, top=False)
    setticks(ax2, fontsize, left=False, right=False, labelright=False, bottom=False, top=True, labeltop=True)
    if mainlegend:
        ax.legend(loc= 'lower right')
    return img, vmin, vmax

def plotrhoNO7(fixz=False):
    if fixz:
        ba = h3ba_fz
        eahi = h3eahi_fz
        tail = '_fixz'
        fig = plt.figure(figsize=(13.,5.))
        grid = gsp.GridSpec(1,3,width_ratios=[5.,5.,1.])
        ax1, ax2, cax = tuple(plt.subplot(grid[i]) for i in range(3)) 
        ymin = 13.
    else:
        ba = h3ba
        eahi = h3eahi
        eami = h3eami
        tail = ''
        fig = plt.figure(figsize=(16.,5.))
        grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.])
        ax1, ax2, ax3, cax = tuple(plt.subplot(grid[i]) for i in range(4))  
        ymin = 7.
            
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(ba, 1)
    imin2,imax2 = getminmax2d(eahi, 1)
    if fixz:
        vmin = min([imin1,imin2])
        vmax = max([imax1,imax2])
    else:
        imin3,imax3 = getminmax2d(eami, 1)
        vmin = min([imin1,imin2,imin3])
        vmax = max([imax1,imax2,imax3])
    fontsize = 16
    
    # set minimum column density
    cmap = 'nipy_spectral'
 
    img, vmin, vmax = subplotrhoNO7(ax1, ba ,title= 'BAHAMAS',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ymin=ymin)
    subplotrhoNO7(ax2, eahi ,title= 'EAGLE, 32k pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    if not fixz:
        subplotrhoNO7(ax3, eami ,title= 'EAGLE, 5600 pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    
    add_colorbar(cax,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'bahamas_comp/rho_NO7%s.pdf'%tail, format='pdf', bbox_inches='tight')


def subplotTNO7(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ionballegend=True,ymin=None,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(1,2),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.15))
    # set the plot to be square
    if ymin is not None:
        ax.set_ylim(ymin,ax.get_ylim()[1])
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim() 
    #print ylim1 
    #ax.set_xlim(xlim1)
    #ax.set_ylim(ylim1)  
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    if mainlegend:
        ax.legend(loc= 'lower right')
    setticks(ax, fontsize)
    return img, vmin, vmax

def plotTNO7(fixz=False):
    if fixz:
        ba = h3ba_fz
        eahi = h3eahi_fz
        tail = '_fixz'
        fig = plt.figure(figsize=(13.,5.))
        grid = gsp.GridSpec(1,3,width_ratios=[5.,5.,1.])
        ax1, ax2, cax = tuple(plt.subplot(grid[i]) for i in range(3)) 
        ymin = 13.0
    else:
        ba = h3ba
        eahi = h3eahi
        eami = h3eami
        tail = ''
        fig = plt.figure(figsize=(16.,5.))
        grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.])
        ax1, ax2, ax3, cax = tuple(plt.subplot(grid[i]) for i in range(4))  
        ymin = 7.
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(ba, 0)
    imin2,imax2 = getminmax2d(eahi, 0)
    if fixz:
        vmin = min([imin1,imin2])
        vmax = max([imax1,imax2])
    else:
        imin3,imax3 = getminmax2d(eami, 0)
        vmin = min([imin1,imin2,imin3])
        vmax = max([imax1,imax2,imax3])       
    
    # set minimum column density
    cmap = 'nipy_spectral'
 
    img, vmin, vmax = subplotTNO7(ax1, ba ,title= 'BAHAMAS',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ymin=ymin)
    subplotTNO7(ax2, eahi ,title= 'EAGLE, 32k pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    if not fixz:
        subplotTNO7(ax3, eami ,title= 'EAGLE, 5600 pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    
    add_colorbar(cax,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'bahamas_comp/T_NO7%s.pdf'%tail,format = 'pdf',bbox_inches='tight')


def subplotNHNO7(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ymin=None,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(1,0),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,0),fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.02))
    # set the plot to be square
    if ymin is not None:
        ax.set_ylim(ymin,ax.get_ylim()[1])
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim() 
    #print ylim1 
    #ax.set_xlim(xlim1)
    #ax.set_ylim(ylim1)  
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    if mainlegend:
        ax.legend(loc= 'lower right')
    return img, vmin, vmax

def plotNHNO7():
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.],wspace=0.5,hspace=0.5)
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(h3bah)
    imin2,imax2 = getminmax2d(h3eahih)
    imin3,imax3 = getminmax2d(h3eamih)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    #ymin=None
    ymin=7.
    cmap = 'nipy_spectral'
 
    img, vmin, vmax = subplotNHNO7(ax1, h3bah ,title= 'BAHAMAS, z-projection',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ymin=ymin)
    subplotNHNO7(ax2, h3eahih ,title= 'EAGLE, 32k pix, xyz-average',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    subplotNHNO7(ax3, h3eamih ,title= 'EAGLE, 5600 pix, xyz-average',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'NH_NO7.png',format = 'png',bbox_inches='tight')




def plotrhoT_eaba_byNO7(slidemode = False,fontsize=14):
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(2)) 
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3, ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(3))
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(h3eahi['edges'][0][0],h3eahi['edges'][0][-1])
    ax1.set_ylim(h3eahi['edges'][1][0],h3eahi['edges'][1][-1])
    ax1.set_ylabel(getlabel(h3eahi,1),fontsize=fontsize)
    ax1.set_xlabel(getlabel(h3eahi,0),fontsize=fontsize)
    
    # square plot
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (None,None,r'$N_{O VII}$')
    
    img, vmin, vmax = add_ionbal_img(ax1,cmap='gist_gray',vmin=-7.)
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10} \, n_{O VII} \,/ n_{O},\, f_H = 0.752$',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
    
    #add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,None), histlegend=False, fraclevels=True,\
    #                        levels=[0.99999], linestyles = ['dashdot'],colors = ['saddlebrown'],dimlabels = dimlabels)
    # EAGLE stnd-res
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')			    
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
	
	
    # EAGLE ba-res			    
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['firebrick','firebrick','firebrick'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')			    
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
    #                            levels=fraclevels, linestyles = linestyles,colors = ['chocolate','chocolate','chocolate'],dimlabels = dimlabels,\
    #		    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['goldenrod','goldenrod','goldenrod'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['darkgreen','darkgreen','darkgreen'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['darkblue','darkblue','darkblue'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
			    
    # BAHAMAS			    
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['lightcoral','lightcoral','lightcoral'],dimlabels = dimlabels,\
         		     legendlabel_pre = 'BA, ')			    
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['sandybrown','sandybrown','sandybrown'],dimlabels = dimlabels,\
    		    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['yellow','yellow','yellow'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['lime','lime','lime'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['dodgerblue','dodgerblue','dodgerblue'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orchid','orchid','orchid'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')		    
			    	  
    #set average density indicator
    ax12 = add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    
    if not slidemode:	    
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_ea-hi-ba_by_NO7.png',format = 'png',bbox_inches='tight') 
    else:
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_ea-hi-ba_by_NO7_slide.png',format = 'png',bbox_inches='tight')



def subplotrhoT_eaba_splitNO7(ax, hist, bins_sub, edges, toplotaxes, xlim = None, ylim = None, vmin = None, vmax = None, cmap = 'viridis',\
                                      dotoplabel = False, dobottomlabel = False, doylabel = False, subplotlabel = None, logrhob = None, ionlab = '', fontsize=fontsize):
    '''
    do the subplot, and sort out the axes
    rhoav is set for snapshot 28
    '''
    ax.set_facecolor(mpl.cm.get_cmap(cmap)(0.))
    img = ax.pcolormesh(edges[toplotaxes[0]], edges[toplotaxes[1]], bins_sub.T, cmap=cmap, vmin=vmin, vmax=vmax)
       
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlim is not None:
        ax.set_ylim(ylim)
        
    # square plot
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    #set average density indicator (also used for bahamas here)
    add_rhoavx(ax,onlyeagle=True,eacolor='darksalmon', userhob_ea = logrhob)
    ax12 = add_ax2rho(ax, fontsize=fontsize, labelax=False) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white',\
                   labelleft = doylabel, labeltop = False, labelbottom = dobottomlabel, labelright = False)
    ax.tick_params(length =7., width = 1., which = 'major', axis='both')
    ax.tick_params(length =4., width = 1., which = 'minor', axis='both')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 6, prune='lower'))
    
    ax12.minorticks_on()
    ax12.tick_params(labelsize=fontsize,direction = 'in', right = False, left = False, top = True, bottom = False, axis='both',\
                     which = 'both', color = 'white', labelleft = False, labeltop = dotoplabel, labelbottom = False, labelright = False)
    ax12.tick_params(length =7., width = 1., which = 'major', axis='both')
    ax12.tick_params(length =4., width = 1., which = 'minor', axis='both')
    ax12.spines['right'].set_color('white')
    ax12.spines['left'].set_color('white')
    ax12.spines['top'].set_color('white')
    ax12.spines['bottom'].set_color('white')
    
    add_ionbal_contours(ax12,label='O7 fractions',legend=False, fontsize=fontsize)
    
    # axis labels: somewhat abbreviated
    if doylabel:
        ax.set_ylabel(r'$\log_{10} T \, [K], N_{\mathrm{%s}}$-wtd'%ionlab, fontsize=fontsize)
    if dobottomlabel:
        ax.set_xlabel(r'$\log_{10} \rho \, [\mathrm{cm}^{-3}], N_{\mathrm{%s}}$-wtd'%ionlab, fontsize=fontsize)
    if dotoplabel:
        ax12.set_xlabel(r'$\sim \log_{10} n_H \, [\mathrm{cm}^{-3}]$', fontsize=fontsize) # the full f_H = 0.752 label is too long 
    if subplotlabel is not None:
        ax.text(0.92,0.92,subplotlabel[2][:-2],fontsize=fontsize-1, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3), color='white')
        ax.text(0.92,0.08,subplotlabel[3],fontsize=fontsize-1, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3), color='white')
    return img, ax12

def plotrhoT_eaba_splitNO7(fixz=False):
    '''
    plot rho/T histograms, split into different subplots for different column density ranges
    '''
    if fixz:
        ba = h3ba_fz  
        eahi = h3eahi_fz
        tail = '_fixz'
        
        limsN_ea = [6, 11, 16, 21, 26] # log10N = 14., 14.5, 15., 15.5, 16. 
        limsN_ba = [6, 11, 16, 21, 26] # log10N = 14., 14.5, 15., 15.5, 16. 
        
        numx = len(limsN_ea) + 1
        numy = 2
        hists = [ba, eahi]
        limsgen = [limsN_ba, limsN_ea]
        titles = ['BAHAMAS', 'EAGLE']
        
    else:
        ba = h3ba
        eahi = h3eahi
        eami = h3eami
        tail = ''
        
        limsN_ea = [250, 270, 280, 290, 300]      # log10N = 12, 14, 15, 16, 17
        limsN_ba = [250, 270, 280, 290, 300] 
        
        numx = len(limsN_ea) + 1
        numy = 3
        hists = [ba, eahi, eami]
        limsgen = [limsN_ba, limsN_ea, limsN_ea]
        titles = ['BAHAMAS', 'EAGLE 32k pix', r'EAGLE $5600$ pix']

    logrhob_ea = logrhob_av_ea
    logrhob_ba = logrhob_av_ba
    ionlab = 'O \, VII'
    logrhobs = [logrhob_ba, logrhob_ea, logrhob_ea]
        
    fontsize = 16    
    cmap = 'nipy_spectral'    
    legendloc= 'upper left'
    legend_bbox_to_anchor=(0.0,0.7)
    
 
    # set up figure and plot; assumes numy >=2 scale numx, width by *0.81 to get nice plot in plt.show()
    fig = plt.figure(figsize=(3.*numx +1. ,4.*numy)) # figsize: width, height
 
    grid = gsp.GridSpec(numy+1, numx, wspace=0.0, hspace=0.05, top=0.95, bottom=0.05, left=0.05, height_ratios=list((4.,)*numy)+[2.]) # gridspec: nrows, ncols
    mainaxes = [[fig.add_subplot(grid[yi,xi]) for yi in range(numy)] for xi in range(numx)] # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[numy, :2]) 
    legax = fig.add_subplot(grid[numy, 2:])    

    
    subplotdct = {}
    subplotdct['labels'] = []
    subplotdct['bins']   = []
    subplotdct['edges']  = []
    
    for hind in range(len(hists)):
        lims = limsgen[hind]
        hist = hists[hind]
        
        edges = hist['edges']
        bins = hist['bins']
        
        # selections
        mins = [(None, None, lims[xi -1]) if (xi > 0) else\
                (None, None, None)\
                for xi in range(numx)]    
        maxs = [(None, None, lims[xi]) if (xi < numx-1) else\
                (None, None, None)\
                for xi in range(numx)] 
 
        # put in min/max values for each dimension
        slices = [tuple([\
                           slice(mins[xi][dim],maxs[xi][dim],None)\
                           for dim in range(len(edges))\
                           ])\
                    for xi in range(numx)]

        # apply the selection to get the histogram for each subplot (assume all histograms have the same dimension/variable layout )
        toplotaxes = (0,1)
        summedaxes = range(len(edges))
        summedaxes.remove(toplotaxes[0])
        summedaxes.remove(toplotaxes[1])
        dimlabels = (None, None,r'$N_{\mathrm{%s}}$'%ionlab) # for range labels per plot
    
        subplotbins = [np.sum(bins[slices[xi]], axis=tuple(summedaxes)) for xi in range(numx)]
    
        if toplotaxes[0] > toplotaxes[1]:
            subplotbins = [sub.transpose() for sub in subplotbins]
    
        # set up labels for the fO/N ranges: (copied from 2d hist contours)
        # want .2f on abundances, .1f on column densities -> use different format strings
        fmtstrings = [[r'%.2f $<$ %s $<$ %.2f, ', r'%.2f $<$ %s $<$ %.2f, ', r'%.1f $<$ %s $<$ %.1f, '],\
                      [r'%.2f $<$ %s, ',          r'%.2f $<$ %s, ',          r'%.1f $<$ %s, ',        ],\
                      [r'%s $<$ %.2f, ',          r'%s $<$ %.2f, ',          r'%s $<$ %.1f, ',        ],\
                      ['',                        '',                        '',                        '']]
        #no label if there is no selection on that dimension
        labels =[ [\
                    (fmtstrings[0][i])%(edges[i][mins[xi][i]], dimlabels[i], edges[i][maxs[xi][i]]) if (mins[xi][i] is not None and maxs[xi][i] is not None) else\
                    (fmtstrings[1][i])%(edges[i][mins[xi][i]], dimlabels[i])                        if (mins[xi][i] is not None and maxs[xi][i] is None)     else\
                    (fmtstrings[2][i])%(dimlabels[i], edges[i][maxs[xi][i]])                        if (mins[xi][i] is None and maxs[xi][i] is not None)     else\
                    (fmtstrings[3][i])\
                          for i in range(len(edges))\
                          ]\
                for xi in range(numx)]
    
        # add in the fraction labels  
        binfracs = [np.sum(bins_sub) for bins_sub in subplotbins]   
        labels = [labels[xi] + ['%.1e'%(binfracs[xi])] for xi in range(numx)]
    
        subplotdct['labels'] += [labels]
        subplotdct['edges']  += [edges]
        subplotdct['bins']   += [subplotbins]
    
    # set plot ranges
    # set np.inf as min for subplot if all values zero ->  get min of nonzero values
    vmin = np.log10(min([min([ np.min(subplotdct['bins'][yi][xi][subplotdct['bins'][yi][xi] > 0]) if np.any(subplotdct['bins'][yi][xi] > 0) else\
                              np.inf\
                              for yi in range(numy)]) for xi in range(numx)]))
    vmax = np.log10(max([max([ np.max(subplotdct['bins'][yi][xi])  for yi in range(numy)]) for xi in range(numx)]))
    
    xlim = (-32., max([edges[0][-1] for edges in subplotdct['edges']]))
    ylim = (3., 8.)
    
    for flatind in range(numx*numy):
        
        hind = flatind/numx
        Nind = flatind%numx
        doylabel      = (Nind == 0)
        dotoplabel    = (hind == 0)
        dobottomlabel = (hind == numy-1)
        
        ax = mainaxes[Nind][hind]        
        imgsub, ax12sub = subplotrhoT_eaba_splitNO7(ax, hist, np.log10(subplotdct['bins'][hind][Nind]), subplotdct['edges'][hind], toplotaxes, 
                                      xlim=xlim, ylim=ylim, vmin=vmin, vmax=vmax, cmap=cmap,\
                                      dotoplabel=dotoplabel, dobottomlabel=dobottomlabel, doylabel=doylabel, subplotlabel=subplotdct['labels'][hind][Nind], logrhob=logrhobs[hind],
                                      fontsize=fontsize, ionlab=ionlab)
        if flatind==0:
            img = imgsub
            ax12 = ax12sub #for contour labels
        if Nind == numx-1:
            ax.text(1.05, 0.05, titles[hind], fontsize=fontsize, horizontalalignment='left', verticalalignment = 'bottom', transform=ax.transAxes, rotation='vertical')
 
    add_colorbar(cax, img=img, clabel=r'$\log_{10}$ fraction of sightlines', fontsize=fontsize, orientation='horizontal') #extend = 'min'
    cax.set_aspect(0.1)
    cax.tick_params(labelsize=fontsize, axis='both')
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = mainaxes[0][0].get_legend_handles_labels()
    handles_contours, labels_contours = ax12.get_legend_handles_labels()
    labels_contours = [  r'$f_{%s} = %s$'%(ionlab, val) for val in labels_contours]
    #level_legend_handles = [mlines.Line2D([], [], color='red', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    legax.legend(handles=handles_contours + handles_subs, labels=labels_contours + labels_subs, fontsize=fontsize,ncol=3,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    legax.axis('off')
    
    plt.savefig(mdir + 'bahamas_comp/phase_diagrams_by_NO7-log10cmi2_EA-L0100N1504_28_A-L400N1024_PtAb_32000pix_100p0slice_z-cen-all%s.pdf'%tail ,format = 'pdf',bbox_inches='tight') 
    
    
##############################################
########        EAGLE data          ##########
##############################################        

def plotrhoT_byNO7(slidemode = False):
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(2)) 
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3, ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(3))
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(h3eahi['edges'][0][0],h3eahi['edges'][0][-1])
    ax1.set_ylim(h3eahi['edges'][1][0],h3eahi['edges'][1][-1])
    ax1.set_ylabel(getlabel(h3eahi,1),fontsize=fontsize)
    ax1.set_xlabel(getlabel(h3eahi,0),fontsize=fontsize)
    
    # square plot
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (None,None,r'$N_{O VII}$')
    
    img, vmin, vmax = add_ionbal_img(ax1,cmap='gist_gray',vmin=-7.)
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10} \, n_{O VII} \,/ n_{O},\, f_H = 0.752$',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
    
    #add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,None), histlegend=False, fraclevels=True,\
    #                        levels=[0.99999], linestyles = ['dashdot'],colors = ['saddlebrown'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels)
			    
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'],dimlabels = dimlabels)	  
    #set average density indicator
    ax12 = add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    
    if not slidemode:	    
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_by_NO7.png',format = 'png',bbox_inches='tight') 
    else:
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_by_NO7_slide.png',format = 'png',bbox_inches='tight')


def plotfOhists_o78_28_16sl(fontsize=12):
    fig, (ax1, ax2) = plt.subplots(figsize = (10.,5.), nrows= 1, ncols =2,sharey = True)
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')

    add_1dplot(ax1,ea4o7,3,log=True,color = 'red', label = 'O VII -weighted')
    add_1dplot(ax2,ea4o8,3,log=True,color = 'blue', label = 'O VIII -weighted')
    
    ax1.set_xlabel(getlabel(ea4o7,3),fontsize=fontsize)
    ax2.set_xlabel(getlabel(ea4o8,3),fontsize=fontsize)
    ax1.set_ylabel('fraction of sightlines')

    add_fsolx(ax1, 'o7', color = 'gray', label = 'solar')
    add_fsolx(ax2, 'o8', color = 'gray', label = 'solar')
    
    ax1.legend(fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    
    ax1.set_xlim(-7,-1.2)
    ax2.set_xlim(-7,-1.2)
    
    #fig.tight_layout()
    fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    plt.savefig(mdir + 'fO-histograms_w_o7-o8_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png')


def plotfO_vs_N_o78_28_16sl(fontsize=12):
    fig = plt.figure(figsize=(11.,5.))
    grid = gsp.GridSpec(1,3,width_ratios=[5.,5.,1.],wspace=0.35,hspace=0.5)
    ax1, ax2, ax3 = tuple(plt.subplot(grid[i], facecolor = 'black') for i in range(3)) 
    
    # set all vmin, vmax to 
    imin1,imax1 = getminmax2d(ea4o7, axis=(0,1), log=True)
    imin2,imax2 = getminmax2d(ea4o8, axis=(0,1), log=True)
    vmin = min([imin1,imin2])
    vmax = max([imax1,imax2])
    
    # set minimum column density
    
    cmap = 'nipy_spectral'
    
    img, vmin, vmax = add_2dplot(ax1,ea4o7,(3,2),log=True,usepcolor = True, vmin = vmin, vmax = vmax, cmap=cmap)
    add_2dplot(ax2,ea4o8,(3,2),log=True,usepcolor = True, vmin = vmin, vmax = vmax, cmap=cmap)
    
    add_colorbar(ax3,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax3.set_aspect(10., adjustable='box-forced')
    
    
    ax1.set_xlabel(getlabel(ea4o7,3),fontsize=fontsize)
    ax2.set_xlabel(getlabel(ea4o8,3),fontsize=fontsize)
    ax1.set_ylabel(getlabel(ea4o7,2),fontsize=fontsize)
    ax2.set_ylabel(getlabel(ea4o8,2),fontsize=fontsize)

    add_fsolx(ax1, 'o7', color = 'gray', label = 'solar')
    add_fsolx(ax2, 'o8', color = 'gray', label = 'solar')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'white')
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'white')

    ax1.legend(fontsize=fontsize, loc = 'upper left')
    
    ax1.set_xlim(-6.,-1.15)
    ax1.set_ylim(11.,17.7)
    ax2.set_xlim(-6.,-1,15)
    ax2.set_ylim(11.,17.7)
    
    #fig.tight_layout()
    fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    plt.savefig(mdir + 'fO-N-histograms_w_o7-o8_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png')
    

def plotrho_fO_by_N_o78_28_16sl(ion = 'o7', fontsize=12, slidemode =False):
    
    if ion == 'o7':
        hist = ea4o7
        ionlab = 'O VII'
    elif ion == 'o8':
        hist = ea4o8
        ionlab = 'O VIII'
    
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'black') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'black')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(-32.,hist['edges'][0][-1])
    ax1.set_ylim(-5.,hist['edges'][3][-1])
    ax1.set_ylabel(getlabel(hist,3),fontsize=fontsize)
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
        
   
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (0,3),log=True, usepcolor = True, vmin = -8., cmap='gist_gray')
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10}$ fraction of pixels',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (None,None,r'$N_{%s}$'%ionlab)
    
    if ion == 'o7' or 'o8':
        lims = [44,64,74,84,94]

    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,None,None), maxs=(None,None,lims[0],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,lims[0],None), maxs=(None,None,lims[1],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,lims[1],None), maxs=(None,None,lims[2],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,lims[2],None), maxs=(None,None,lims[3],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,lims[3],None), maxs=(None,None,lims[4],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,lims[4],None), maxs=(None,None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
    #set average density indicator
    add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    
    ax12.minorticks_on()
    ax12.tick_params(labelsize=fontsize,direction = 'in', right = False, left = False, top = True, bottom = False, axis='both', which = 'both', color = 'white')
    
    # set solar oxygen fraction reference    
    ax1.set_xlabel(getlabel(ea4o7,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(ea4o7,3),fontsize=fontsize)
    add_fsoly(ax1, 'o7', color = 'indianred', label = 'solar fraction')
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'fO-N-histograms_by_%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png'%ion ,format = 'png',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'fO-N-histograms_by_%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all_slide.png'%ion,format = 'png',bbox_inches='tight')
    

def plot_o678corr_27_16sl(zoom = False):
    hist = ea3o678
    
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.],wspace = 0.4)
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(hist, 2)
    imin2,imax2 = getminmax2d(hist, 1)
    imin3,imax3 = getminmax2d(hist, 0)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    #ymin=7.
    cmap = 'nipy_spectral'
    
    img, vmin, vmax = add_2dplot(ax1,hist,(0,1),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap)
    add_2dplot(ax2,hist,(0,2),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap)
    add_2dplot(ax3,hist,(1,2),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap)
    
   
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax2.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax2.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax3.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax3.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    
    if zoom:
        ax1.set_xlim(8.,ax1.get_xlim()[1])
        ax2.set_xlim(4.,ax2.get_xlim()[1])
        ax3.set_xlim(8.,ax3.get_xlim()[1])
        
        ax1.set_ylim(8.,ax1.get_ylim()[1])
        ax2.set_ylim(8.,ax2.get_ylim()[1])
        ax3.set_ylim(8.,ax3.get_ylim()[1])
    
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    ax3.minorticks_on()
    ax3.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    if zoom:
        plt.savefig(mdir + 'coldens_o6-o7-o8_correlations_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all_zoom.png',format = 'png',bbox_inches='tight')
    else:
        plt.savefig(mdir + 'coldens_o6-o7-o8_correlations_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all.png',format = 'png',bbox_inches='tight')


def plot_o678corr_27_16sl_diag():
    hist = ea3o678
    
    fig = plt.figure(figsize=(5.5, 3.))
    grid = gsp.GridSpec(1,3,width_ratios=[5.,5.,1.],wspace = 0.4)
    ax1, ax2, ax4 = tuple(plt.subplot(grid[i]) for i in range(3)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    vmin = 0.
    vmax = 1.
    xlim = (12.5, 16.6)
    ylim = (13., None)
    
    # set minimum column density
    #ymin=7.
    cmap = 'gist_gray'
    
    bins = hist['bins']
    bins_o67 = np.sum(bins, axis=2)
    bins_o68 = np.sum(bins, axis=1)
    
    bins_o67 = bins_o67 / np.sum(bins_o67, axis=1)[:, np.newaxis] # fraction of O7 absorbers in an O7 bin at fixed O6 column density
    bins_o67 = np.cumsum(bins_o67[:, ::-1], axis=1)[:, ::-1] # fraction of O7 absorbers with N > value in an O7 bin at fixed o6 column density
    bins_o68 = bins_o68 / np.sum(bins_o68, axis=1)[:, np.newaxis]
    bins_o68 = np.cumsum(bins_o68[:, ::-1], axis=1)[:, ::-1] 
    img = ax1.pcolormesh(hist['edges'][0], hist['edges'][1], bins_o67.T, cmap=cmap, vmin=vmin, vmax=vmax)
    img = ax2.pcolormesh(hist['edges'][0], hist['edges'][2], bins_o68.T, cmap=cmap, vmin=vmin, vmax=vmax)
   
    ax1.set_xlabel(r'$\log_{10}\, N_{O \, VI} \; [\mathrm{cm}^{-2}]$',fontsize=fontsize)
    ax1.set_ylabel(r'$\log_{10}\, N_{O \, VII} \; [\mathrm{cm}^{-2}]$',fontsize=fontsize)
    ax2.set_xlabel(r'$\log_{10}\, N_{O \, VI} \; [\mathrm{cm}^{-2}]$',fontsize=fontsize)
    ax2.set_ylabel(r'$\log_{10}\, N_{O \, VIII} \; [\mathrm{cm}^{-2}]$',fontsize=fontsize)

    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both')
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both')
    
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_ylim(ylim)

    add_colorbar(ax4, img=img, clabel=r'$f \left( > N \,|\, N_{\mathrm{O \, VI}} \right)$', fontsize=fontsize)
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'coldens_o6-o7-o8_correlations_diagnostic_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all_zoom.pdf',format = 'pdf',bbox_inches='tight')
    
        
        
def plot_ne8o78corr_27_16sl(zoom = False):
    hist = ea3ne8o78_16
    
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.],wspace = 0.4)
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(hist, 2)
    imin2,imax2 = getminmax2d(hist, 1)
    imin3,imax3 = getminmax2d(hist, 0)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    #ymin=7.
    cmap = 'nipy_spectral'
    
    img, vmin, vmax = add_2dplot(ax1,hist,(0,1),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap)
    add_2dplot(ax2,hist,(0,2),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap)
    add_2dplot(ax3,hist,(1,2),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap)
    
   
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax2.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax2.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax3.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax3.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    
    if zoom:
        ax1.set_xlim(8.,ax1.get_xlim()[1])
        ax2.set_xlim(4.,ax2.get_xlim()[1])
        ax3.set_xlim(8.,ax3.get_xlim()[1])
        
        ax1.set_ylim(8.,ax1.get_ylim()[1])
        ax2.set_ylim(8.,ax2.get_ylim()[1])
        ax3.set_ylim(8.,ax3.get_ylim()[1])
    
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    ax3.minorticks_on()
    ax3.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    if zoom:
        plt.savefig(mdir + 'coldens_ne8-o7-o8_correlations_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all_zoom.png',format = 'png',bbox_inches='tight')
    else:
        plt.savefig(mdir + 'coldens_ne8-o7-o8_correlations_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all.png',format = 'png',bbox_inches='tight')

def plot_hneutralo78corr_28_16sl(zoom = False):
    hist = ea3_heutralo78
    # deal with - np.inf edges
    if hist['edges'][0][0] == -np.inf:
        hist['edges'][0][0] = -100.
    if hist['edges'][1][0] == -np.inf:
        hist['edges'][1][0] = -100.
        
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.],wspace = 0.4)
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(hist, axis=2, pixdens=True)
    imin2,imax2 = getminmax2d(hist, axis=1, pixdens=True)
    imin3,imax3 = getminmax2d(hist, axis=0, pixdens=True)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    #ymin=7.
    cmap = 'nipy_spectral'
    textcolor = 'white'
    
    img, vmin, vmax = add_2dplot(ax1,hist,(0,1),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap, pixdens=True)
    add_2dplot(ax2,hist,(0,2),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap, pixdens=True)
    add_2dplot(ax3,hist,(1,2),log=True,usepcolor = True,vmin=vmin,vmax=vmax,cmap = cmap, pixdens=True)
    
    percs = np.array([0.01, 0.10, 0.50, 0.90, 0.99])
    colors = ('gray',)*len(percs)
    linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    
    perc1 = percentiles_from_histogram(np.sum(hist['bins'], axis=2), hist['edges'][1], axis=1, percentiles=percs)
    perc2 = percentiles_from_histogram(np.sum(hist['bins'], axis=1), hist['edges'][2], axis=1, percentiles=percs)
    perc3 = percentiles_from_histogram(np.sum(hist['bins'], axis=0), hist['edges'][2], axis=1, percentiles=percs)
    
    cens1 = hist['edges'][0][:-1] + 0.5 * np.diff(hist['edges'][0]) 
    cens2 = cens1
    cens3 = hist['edges'][1][:-1] + 0.5 * np.diff(hist['edges'][1]) 
    
    for pind in range(len(percs)):
        ax1.plot(cens1, perc1[pind], color=colors[pind], linestyle=linestyles[pind], label=r'$%.0f$ %%'%(percs[pind]*100.))
        ax2.plot(cens2, perc2[pind], color=colors[pind], linestyle=linestyles[pind], label=r'$%.0f$ %%'%(percs[pind]*100.))
        ax3.plot(cens3, perc3[pind], color=colors[pind], linestyle=linestyles[pind], label=r'$%.0f$ %%'%(percs[pind]*100.))
    ax3.legend(fontsize=fontsize, loc='lower right', bbox_to_anchor=(0.95, 0.05))
    
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax2.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax2.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax3.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax3.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    
    if zoom:
        ax1.set_xlim(11.,hist['edges'][0][-2])
        ax2.set_xlim(11.,hist['edges'][0][-2])
        ax3.set_xlim(11.,hist['edges'][1][-2])
        
        ax1.set_ylim(12.,hist['edges'][1][-2])
        ax2.set_ylim(12.,hist['edges'][2][-2])
        ax3.set_ylim(11.,hist['edges'][2][-2])
    
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = textcolor)
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = textcolor)
    ax3.minorticks_on()
    ax3.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = textcolor)
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of sightlines $\mathrm{dex}^{-2}$')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    if zoom:
        plt.savefig(mdir + 'coldens_hneutral-o7-o8_correlations_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all_zoom.png',format = 'png',bbox_inches='tight')
    else:
        plt.savefig(mdir + 'coldens_hneutral-o7-o8_correlations_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_T4EOS_zcen-all.png',format = 'png',bbox_inches='tight')
        
        
def plotrho_fO_by_T_o78_28_16sl(ion = 'o7', fontsize=12, slidemode =False):
    
    if ion == 'o7':
        hist = ea4o7
        ionlab = 'O VII'
    elif ion == 'o8':
        hist = ea4o8
        ionlab = 'O VIII'
    
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'black') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'black')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(-32.,hist['edges'][0][-1])
    ax1.set_ylim(-5.,hist['edges'][3][-1])
    ax1.set_ylabel(getlabel(hist,3),fontsize=fontsize)
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
        
   
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (0,3),log=True, usepcolor = True, vmin = -8., cmap='gist_gray')
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10}$ fraction of sightlines',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (None,'T',r'$N_{%s}$'%ionlab)
    
    if ion == 'o7' or 'o8':
        lims = [10,20,30,40,50] # log10 T [K] = 3., 4, 5, 6, 7

    add_2dhist_contours(ax1,hist,(0,3),mins= (None,None,None,None), maxs=(None,lims[0],None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,lims[0],None,None), maxs=(None,lims[1],None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,lims[1],None,None), maxs=(None,lims[2],None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,lims[2],None,None), maxs=(None,lims[3],None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,lims[3],None,None), maxs=(None,lims[4],None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,3),mins= (None,lims[4],None,None), maxs=(None,None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
    #set average density indicator
    add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    
    ax12.minorticks_on()
    ax12.tick_params(labelsize=fontsize,direction = 'in', right = False, left = False, top = True, bottom = False, axis='both', which = 'both', color = 'white')
    
    # set solar oxygen fraction reference    
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,3),fontsize=fontsize)
    add_fsoly(ax1, 'o7', color = 'indianred', label = 'solar fraction')
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'fO-N-histograms_by_T_coldens_%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png'%ion ,format = 'png',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'fO-N-histograms_by_T_coldens_%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png'%ion,format = 'png',bbox_inches='tight')
    




def plot_cor_o678_27_16sl(fontsize=12, slidemode =False):
    
    hist = ea3o678_16
    ionlab = 'O VI'
        
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'black') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'black')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range (max  = 0.05 + max included in hist)
    ax1.set_xlim(9.,17.8)
    ax1.set_ylim(9.,17.2)
    ax1.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
        
   
    # plot backgorund and colorbar: total distribution
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (1,2),log=True, usepcolor = True, vmin = -8., cmap='bone')
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10}$ fraction of sightlines',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (r'$N_{%s}$'%ionlab,None,None)

    lims = [77,97,107,117,127]

    add_2dhist_contours(ax1,hist,(1,2),mins= (None,None,None), maxs=(lims[0], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[0], None, None), maxs=(lims[1], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[1], None, None), maxs=(lims[2], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[2], None, None), maxs=(lims[3], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[3], None, None), maxs=(lims[4], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[4], None, None), maxs=(None, None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
       
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'white')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'coldens_o7-o8_by_o6_L0100N1504_27_PtAb_C2Sm_32000pix_6.25slice_zcen-all_T4EOS.png' ,format = 'png',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_o7-o8_by_o6_L0100N1504_27_PtAb_C2Sm_32000pix__6.25slice_zcen-all_T4EOS_slide.png',format = 'png',bbox_inches='tight')

def subplot_phasediagram_grid_by_N_fO(ax, hist, bins_sub, bins_all, edges, toplotaxes, xlim = None, ylim = None, vmin = None, vmax = None, cmap = 'viridis',\
                                      dotoplabel = False, dobottomlabel = False, doylabel = False, subplotlabel = None, logrhob = None, ionlab = ''):
    '''
    do the subplot, and sort out the axes
    rhoav is set for snapshot 28
    '''
    img = ax.pcolormesh(edges[toplotaxes[0]], edges[toplotaxes[1]], bins_sub.T, cmap = cmap, vmin = vmin, vmax = vmax)
    
        
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlim is not None:
        ax.set_ylim(ylim)
        
    # square plot
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    #set average density indicator
    add_rhoavx(ax,onlyeagle=True,eacolor='darksalmon', userhob_ea = logrhob)
    ax12 = add_ax2rho(ax,fontsize=fontsize,labelax = False) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white',\
                   labelleft = doylabel, labeltop = False, labelbottom = dobottomlabel, labelright = False)
    ax.tick_params(length =7., width = 1., which = 'major')
    ax.tick_params(length =4., width = 1., which = 'minor')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 6, prune='lower'))
    
    ax12.minorticks_on()
    ax12.tick_params(labelsize=fontsize,direction = 'in', right = False, left = False, top = True, bottom = False, axis='both',\
                     which = 'both', color = 'white', labelleft = False, labeltop = dotoplabel, labelbottom = False, labelright = False)
    ax12.tick_params(length =7., width = 1., which = 'major')
    ax12.tick_params(length =4., width = 1., which = 'minor')
    ax12.spines['right'].set_color('white')
    ax12.spines['left'].set_color('white')
    ax12.spines['top'].set_color('white')
    ax12.spines['bottom'].set_color('white')
    
    # axis labels: somewhat abbreviated
    if doylabel:
        ax.set_ylabel(r'$\log_{10} T \, [K], N_{\mathrm{%s}}$-wtd'%ionlab,fontsize=fontsize)
    if dobottomlabel:
        ax.set_xlabel(r'$\log_{10} \rho \, [\mathrm{cm}^{-3}], N_{\mathrm{%s}}$-wtd'%ionlab,fontsize=fontsize)
    if dotoplabel:
        ax12.set_xlabel(r'$\sim \log_{10} n_H \, [\mathrm{cm}^{-3}]$', fontsize = fontsize) # the full f_H = 0.752 label is too long 
    if subplotlabel is not None:
        ax.text(0.92,0.92,subplotlabel[4],fontsize=fontsize-1, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
        ax.text(0.92,0.23,subplotlabel[3][:-2],fontsize=fontsize-1, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
        ax.text(0.92,0.08,subplotlabel[2][:-2],fontsize=fontsize-1, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    return img

     
def plot_phasediagram_grid_by_N_fO(ion = 'o7'):
    '''
    for 16 slices: plot grid of rho-T sightline histograms for N, fO ranges
    '''

    if ion == 'o7':
        hist = ea4o7
        ionlab = 'O VII'
        limsN = [44,64,74,84,94]      # log10N = 12, 14, 15, 16, 17
        limsf = [25, 45, 54, 62, 70]  # log10 fO/solar ~ -4, -2, -1.1, -0.3, 0.5
        logrhob = logrhob_av_ea # eagle 28
    elif ion == 'o8':
        hist = ea4o8
        ionlab = 'O VIII'
        limsN = [44,64,74,84,94]      # log10N = 12, 14, 15, 16, 17
        limsf = [25, 45, 54, 62, 70]  # log10 fO/solar ~ -4, -2, -1.1, -0.3, 0.5
        logrhob = logrhob_av_ea # eagle 28
    logfsol = np.log10(ol.solar_abunds[ol.elements_ion[ion]])
    
    bins = hist['bins']
    edges = hist['edges']
    edges[3] -= logfsol # rescale values to log10 solar
    
    # set up ranges and selections
    numx = len(limsN) + 1
    numy = len(limsf) + 1
    
    mins = [[(None, None, limsN[xi -1], limsf[yi - 1]) if (xi > 0 and yi > 0) else\
             (None, None, limsN[xi -1], None)          if (xi > 0) else\
             (None, None, None,         limsf[yi - 1]) if (yi > 0) else\
             (None, None, None,         None)\
             for yi in range(numy)] for xi in range(numx)]
    
    maxs = [[(None, None, limsN[xi], limsf[yi]) if (xi < numx-1 and yi < numy-1) else\
             (None, None, limsN[xi], None)      if (xi < numx-1) else\
             (None, None, None,      limsf[yi]) if (yi < numy-1) else\
             (None, None, None,      None)\
             for yi in range(numy)] for xi in range(numx)] 
    # put in min/max values for each dimension
    slices = [[ tuple([\
                       slice(mins[xi][yi][dim],maxs[xi][yi][dim],None)\
                       for dim in range(len(edges))\
                       ])\
                for yi in range(numy)] for xi in range(numx)]
    

    # apply the slection to get the histogram for each subplot
    toplotaxes = (0,1)
    summedaxes = range(len(hist['edges']))
    summedaxes.remove(toplotaxes[0])
    summedaxes.remove(toplotaxes[1])

    subplotbins = [[ np.sum(bins[ slices[xi][yi] ], axis=tuple(summedaxes))\
                    for yi in range(numy)] for xi in range(numx)]

    if toplotaxes[0] > toplotaxes[1]:
        subplotbins = [[sub.transpose() for sub in ys] for ys in subplotbins]

        
    # set up labels for the fO/N ranges: (copied from 2d hist contours)
    dimlabels = (None, None,r'$N_{\mathrm{%s}}$'%ionlab,r'$f_{O}$') # for range labels per plot
    # want .2f on abundances, .1f on column densities -> use different format strings
    fmtstrings = [[r'%.2f $<$ %s $<$ %.2f, ', r'%.2f $<$ %s $<$ %.2f, ', r'%.1f $<$ %s $<$ %.1f, ', r'%.2f $<$ %s $<$ %.2f, '],\
                  [r'%.2f $<$ %s, ',          r'%.2f $<$ %s, ',          r'%.1f $<$ %s, ',          r'%.2f $<$ %s, '],\
                  [r'%s $<$ %.2f, ',          r'%s $<$ %.2f, ',          r'%s $<$ %.1f, ',          r'%s $<$ %.2f, '],\
                  ['',                        '',                        '',                        '']]
    #no label if there is no selection on that dimension
    labels =[[      [\
                      (fmtstrings[0][i])%(edges[i][mins[xi][yi][i]],dimlabels[i],edges[i][maxs[xi][yi][i]]) if (mins[xi][yi][i] is not None and maxs[xi][yi][i] is not None) else\
                      (fmtstrings[1][i])%(edges[i][mins[xi][yi][i]],dimlabels[i])                           if (mins[xi][yi][i] is not None and maxs[xi][yi][i] is None)     else\
		               (fmtstrings[2][i])%(dimlabels[i],edges[i][maxs[xi][yi][i]])                          if (mins[xi][yi][i] is None and maxs[xi][yi][i] is not None)     else\
		               (fmtstrings[3][i])\
                      for i in range(len(edges))\
                      ]\
             for yi in range(numy)] for xi in range(numx)]

    binfracs = [[np.sum(bins_sub) for bins_sub in ys] for ys in subplotbins]

    # add in the fraction labels     
    labels = [[ labels[xi][yi] + [ '%.1e'%(binfracs[xi][yi]) ]
             for yi in range(numy)] for xi in range(numx)]
    
    
    # set up figure and plot; assumes numy >=2 scale numx, width by *0.81 to get nice plot in plt.show()
    fontsize = 14
    fig = plt.figure(figsize=(3.*numx +1. ,3.*numy*0.94)) # figsize: width, height
 
    grid = gsp.GridSpec(numy,numx +1,width_ratios=list((3.,)*numx)+[1.],wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05) # grispec: nrows, ncols
    mainaxes = [[fig.add_subplot(grid[yi,xi]) for yi in range(numy)] for xi in range(numx)] # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:2, numx]) 
    legax = fig.add_subplot(grid[2:, numx])    

    linestyles = ['solid', 'dashed', 'dotted']
    legendloc= 'upper left'
    legend_bbox_to_anchor=(0.0,0.95)
    fraclevels = [0.99,0.50,0.10] 
    
    xlim = (-32., edges[0][-1])
    ylim = (3., 8.)
    # set np.inf as min for subplot if all values zero ->  get min of nonzero values
    vmin = np.log10(min([min([ np.min(subplotbins[xi][yi][subplotbins[xi][yi] > 0]) if np.any(subplotbins[xi][yi] > 0) else\
                              np.inf\
                              for yi in range(numy)]) for xi in range(numx)]))
    vmax = np.log10(max([max([ np.max(subplotbins[xi][yi])  for yi in range(numy)]) for xi in range(numx)]))
    
    for flatind in range(numx*numy):
        
        xi = flatind/numx
        yi = flatind%numx
        doylabel      = (xi == 0)
        dotoplabel    = (yi == 0)
        dobottomlabel = (yi == numy-1)
        imgsub = subplot_phasediagram_grid_by_N_fO(mainaxes[xi][yi], hist, np.log10(subplotbins[xi][yi]), bins, edges, toplotaxes, xlim = xlim, ylim = ylim, vmin = vmin, vmax = vmax, cmap = 'viridis',\
                                      dotoplabel = dotoplabel, dobottomlabel = dobottomlabel, doylabel = doylabel, subplotlabel = labels[xi][yi], logrhob = logrhob, ionlab = ionlab)
        if flatind ==0:
            img = imgsub
    
    add_colorbar(cax,img=img,clabel=r'$\log_{10}$ fraction of sightlines',fontsize=fontsize) #extend = 'min'
    cax.set_aspect(10.)
    cax.tick_params(labelsize=fontsize,axis='both')
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = mainaxes[0][0].get_legend_handles_labels()
    #level_legend_handles = [mlines.Line2D([], [], color='red', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    level_legend_handles = []
    legax.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=1,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    legax.axis('off')
    
    if ion == 'o7':
        plt.savefig(mdir + 'phase_diagrams_by_NO7-log10cmi2_by_fmassO-w-NO7-log10solar_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png' ,format = 'png',bbox_inches='tight') 
    if ion == 'o8':
        plt.savefig(mdir + 'phase_diagrams_by_NO8-log10cmi2_by_fmassO-w-NO8-log10solar_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.png' ,format = 'png',bbox_inches='tight') 
     
        
        


def compare_ionisation_bands():
    fig, ax = plt.subplots(ncols = 1, nrows = 1)
    levels = [.05, 0.5] #[1e-3,3e-2, 0.3, 0.9]
    linestyles = [ 'dashed', 'solid'] #, 'dashdot', 'dotted']
    fontsize = 12
    
    add_ionbal_contours(ax,legend=False,ib=o7_ib_27, levels = levels,  colors = list(('red',)*4), linestyles = linestyles, reset_lim = False)
    add_ionbal_contours(ax,legend=False,ib=o8_ib_27, levels = levels,  colors = list(('blue',)*4), linestyles = linestyles, reset_lim = False)    
    add_ionbal_contours(ax,legend=False,ib=o6_ib_27, levels = levels,  colors = list(('green',)*4), linestyles = linestyles, reset_lim = False)
    add_ionbal_contours(ax,legend=False,ib=ne8_ib_27, levels = levels,  colors = list(('orange',)*4), linestyles = linestyles, reset_lim = False)
    
    ax.set_ylabel(r'$\log_{10} T \, [K]$', fontsize =fontsize)
    ax.set_xlabel(r'$\log_{10} \rho \, [\mathrm{g} \,\mathrm{cm}^{-3}], f_H = 0.752$', fontsize=fontsize)
    ax.set_title('Ion balance at $z=0.1$, HM01 photoionisation', fontsize=fontsize)
    
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1e'%(levels[i])) for i in range(len(levels))]
    ion_legend_handles = [mlines.Line2D([], [], color='green', linestyle = 'solid', label='O VI'),\
                          mlines.Line2D([], [], color='red', linestyle = 'solid', label='O VII'),\
                          mlines.Line2D([], [], color='blue', linestyle = 'solid', label='O VIII'),\
                          mlines.Line2D([], [], color='orange', linestyle = 'solid', label='Ne VIII') ]
    ax.legend(handles=level_legend_handles + ion_legend_handles, fontsize=fontsize, ncol=2, loc='lower right') #,ncol=1,loc='lower right',bbox_to_anchor=(0.98, 0.02), framealpha = 0., edgecolor = 'lightgray'
    
    plt.savefig(mdir + 'ion_balance_snap27_ne8_o678.png')
    
def compare_ionisation_bands_o6c4_z0p352():
    fig, ax = plt.subplots(ncols = 1, nrows = 1)
    levels = [.05, 0.5] #[1e-3,3e-2, 0.3, 0.9]
    linestyles = [ 'dashed', 'solid'] #, 'dashdot', 'dotted']
    fontsize = 12
    
    z=0.352 # snap 20 C-EAGLE comparison
    
    h1_ib, logTKh1_ib, lognHcm3h1_ib = m3.findiontables('h1',z)
    c4_ib, logTKc4_ib, lognHcm3c4_ib = m3.findiontables('c4',z)
    o6_ib, logTKo6_ib, lognHcm3o6_ib = m3.findiontables('o6',z)
    o7_ib, logTKo7_ib, lognHcm3o7_ib = m3.findiontables('o7',z)
    o8_ib, logTKo8_ib, lognHcm3o8_ib = m3.findiontables('o8',z)
    
    add_ionbal_contours(ax,legend=False,ib=h1_ib, levels = levels,  colors = list(('red',)*4), linestyles = linestyles, reset_lim = False)
    add_ionbal_contours(ax,legend=False,ib=c4_ib, levels = levels,  colors = list(('orange',)*4), linestyles = linestyles, reset_lim = False)
    add_ionbal_contours(ax,legend=False,ib=o6_ib, levels = levels,  colors = list(('green',)*4), linestyles = linestyles, reset_lim = False)
    add_ionbal_contours(ax,legend=False,ib=o7_ib, levels = levels,  colors = list(('blue',)*4), linestyles = linestyles, reset_lim = False)
    add_ionbal_contours(ax,legend=False,ib=o8_ib, levels = levels,  colors = list(('purple',)*4), linestyles = linestyles, reset_lim = False)    
    #add_ionbal_contours(ax,legend=False,ib=ne8_ib, levels = levels,  colors = list(('orange',)*4), linestyles = linestyles, reset_lim = False)
    
    ax.set_ylabel(r'$\log_{10} T \, [K]$', fontsize =fontsize)
    ax.set_xlabel(r'$\log_{10} \rho \, [\mathrm{g} \,\mathrm{cm}^{-3}], f_H = 0.752$', fontsize=fontsize)
    ax.set_title('Ion balance at $z=0.1$, HM01 photoionisation', fontsize=fontsize)
    
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1e'%(levels[i])) for i in range(len(levels))]
    ion_legend_handles = [mlines.Line2D([], [], color='red', linestyle = 'solid', label='H I (opt. thin)'),\
                          mlines.Line2D([], [], color='orange', linestyle = 'solid', label='C IV') ,\
                          mlines.Line2D([], [], color='green', linestyle = 'solid', label='O VI'),\
                          mlines.Line2D([], [], color='blue', linestyle = 'solid', label='O VII'),\
                          mlines.Line2D([], [], color='purple', linestyle = 'solid', label='O VIII'),\
                          ]
    ax.legend(handles=level_legend_handles + ion_legend_handles, fontsize=fontsize, ncol=2, loc='lower right') #,ncol=1,loc='lower right',bbox_to_anchor=(0.98, 0.02), framealpha = 0., edgecolor = 'lightgray'
    
    plt.savefig(mdir + 'ion_balance_snap20-CEH_h1nossh-c4-o678.png')
    
def plot_ionpeaks(z=0., ions=('o6', 'o7', 'o8', 'ne8')):
    # reshift zero ion balance peak for different densities
    scmap = 'nipy_spectral'
    fontsize = 12
    
    ibs = [m3.findiontables(ion, z) for ion in ions] # ib, T, nH
    if not (np.all(np.array([ibs[0][2] == ibs[i][2] for i in range(len(ibs))])) and\
            np.all(np.array([ibs[0][1] == ibs[i][1] for i in range(len(ibs))]))):
        raise RuntimeError('number density ranges for different ions do not match')
    vmin = min([np.min(ib[2]) for ib in ibs])
    vmax = max([np.max(ib[2]) for ib in ibs])
    colorvals = (ibs[0][2] - vmin)/(vmax-vmin)
    
    fig, axes = plt.subplots(ncols=len(ions) + 1, nrows=1, gridspec_kw = {'width_ratios': [5.]*len(ions) + [1.]}, figsize = (5. * len(ions) + 1, 5.))
    cmap = mpl.cm.get_cmap(scmap)
    
    # close to interesting densities 
    nH_cuts_approx = [np.argmin(np.abs(ibs[0][2] - nH)) for nH in [-6., -5., -4., -3.]]# densities -30.4, -29. (-28.9), -28. (-27.9), -26.5 (-26.4)
    nH_cut_colors = ['saddlebrown', 'peru', 'burlywood', 'violet']

    specialrhovalscounter = 0
    for ind in range(len(logrhocgs_ib)):
        if ind not in nH_cuts_approx:
            color = cmap(colorvals[ind])
            linewidth = 1
        else:
            color = nH_cut_colors[specialrhovalscounter]
            specialrhovalscounter +=1
            linewidth = 2
        for axi in range(len(ions)):
            axes[axi].plot(ibs[axi][1], ibs[axi][0][ind], color=color, linewidth=linewidth)
    for axi in range(len(ions)):
        ax = axes[axi]

        ax.tick_params(labelsize=fontsize-1,direction='in',top=True,right=True, which = 'both')
        ax.minorticks_on()
        ax.set_xlabel(r'$\log_{10} T \, [K]$', fontsize = fontsize)
        if axi == 0:
            ax.set_ylabel(r'ion fraction', fontsize = fontsize)
        peak = np.max(ibs[axi][0][-1])        
        ax.axhline(0.1*peak, color='gray', linewidth = 1)
        ax.set_xlim(3.,7.5)
        
        Tcuts_0p1 = find_intercepts(ibs[axi][0][-1], ibs[axi][1], 0.1 * peak)
        Tcuts_0p01 = find_intercepts(ibs[axi][0][-1], ibs[axi][1], 0.01 * peak)
        for T in list(Tcuts_0p1) + list(Tcuts_0p01):
            ax.axvline(T, linestyle='dotted', color='black')
        
        ax.text(0.95, 0.95, r'$\mathrm{%s}$'%(ionlabels[ions[axi]]) ,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
        ax.text(0.95, 0.85, 'coll. max:' ,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes)
        ax.text(0.95, 0.80, '%.3f'%peak ,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes)
    
   
    # manual color bar to add line without issues
    axes[-1].tick_params(labelsize=fontsize-1,direction='out',top=False,right=True, bottom = False, left = False, which = 'both', labeltop = False, labelbottom=False, labelleft = False, labelright = True)
    axes[-1].minorticks_on()
    axes[-1].yaxis.set_label_position("right")

    axes[-1].imshow(colorvals[:,np.newaxis], extent = (0,1,vmin,vmax), origin='lower', cmap=scmap)
    cxlim = axes[-1].get_xlim()
    cylim = axes[-1].get_ylim()
    axes[-1].set_aspect((10*cxlim[1]-cxlim[0])/(cylim[1]-cylim[0]), adjustable='box-forced')
    axes[-1].set_ylabel(r'$\log_{10} n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    # indicate density cuts in the color bar
    for ind in range(len(nH_cuts_approx)):
        axes[-1].axhline(ibs[0][2][nH_cuts_approx[ind]], color=nH_cut_colors[ind], linewidth=2)
    
    plt.savefig(mdir + 'ion_balance_%s_cloudy_HM01_z%.1f.pdf'%('-'.join(ions), z), format = 'pdf',bbox_inches='tight')
    
    
    
    
def plotOabund_mass_vs_ionweighted_o78(slices = 16):
    
    if slices == 16:
        histo7 = eafOo7
        histo8 = eafOo8
    elif slices == 1:
        histo7 = eafOo7_100
        histo8 = eafOo8_100
    
    fig, axes = plt.subplots(ncols = 4, nrows = 2, gridspec_kw = {'width_ratios': [5.,5.,5.,1.]}, figsize = (16.,10.))
    fontsize = 12.
    vmin = -8.
    cmap = 'nipy_spectral'
    
    vmax_o7 = np.max(np.array([getminmax2d(histo7, axis=0, log=True), getminmax2d(histo7, axis=1, log=True), getminmax2d(histo7, axis=2, log=True)]))
    vmax_o8 = np.max(np.array([getminmax2d(histo8, axis=0, log=True), getminmax2d(histo8, axis=1, log=True), getminmax2d(histo8, axis=2, log=True)]))

    xlims = np.array([ [(-7., histo7['edges'][1][-1]), (-7., histo7['edges'][1][-1]), (-7., histo7['edges'][2][-1]), None],\
                       [(-7., histo8['edges'][1][-1]), (-7., histo8['edges'][1][-1]), (-7., histo8['edges'][2][-1]), None]])
    ylims = np.array([ [(-7., histo7['edges'][2][-1]), (8., histo7['edges'][0][-1]), (8., histo7['edges'][0][-1]), None],\
                       [(-7., histo8['edges'][2][-1]), (8., histo8['edges'][0][-1]), (8., histo8['edges'][0][-1]), None]])
        
    for axind in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]:
        if xlims[axind] is not None:
            xlim = xlims[axind]
            axes[axind].set_xlim(*tuple(xlims[axind]))
        else:
            xlim = axes[axind].get_xlim()
        if ylims[axind] is not None:
            ylim = ylims[axind]
            axes[axind].set_ylim(*tuple(ylims[axind]))
        else:
            ylim = axes[axind].get_ylim()
        axes[axind].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')
    
    
    img_o7, vmin, vmax_o7 = add_2dplot(axes[0,0],histo7, (1,2),log=True, usepcolor = True, vmin = vmin, vmax = vmax_o7, cmap=cmap)
    axes[0,0].set_xlabel(getlabel(histo7,1), fontsize=fontsize)
    axes[0,0].set_ylabel(getlabel(histo7,2), fontsize=fontsize)
    
    add_2dplot(axes[0,1],histo7, (1,0),log=True, usepcolor = True, vmin = vmin, vmax = vmax_o7, cmap=cmap)
    axes[0,1].set_xlabel(getlabel(histo7,1), fontsize=fontsize)
    axes[0,1].set_ylabel(getlabel(histo7,0), fontsize=fontsize)
    
    add_2dplot(axes[0,2],histo7, (2,0),log=True, usepcolor = True, vmin = vmin, vmax = vmax_o7, cmap=cmap)
    axes[0,2].set_xlabel(getlabel(histo7,2), fontsize=fontsize)
    axes[0,2].set_ylabel(getlabel(histo7,0), fontsize=fontsize)
    
    add_colorbar(axes[0,3],img=img_o7, clabel= r'$\log_{10}$ fraction of sightlines' ,newax=False,extend='min',fontsize=fontsize,orientation='vertical')
    axes[0,3].set_aspect(10*(xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')
    
    
    img_o8, vmin, vmax_o8 = add_2dplot(axes[1,0],histo8, (1,2),log=True, usepcolor = True, vmin = vmin, vmax = vmax_o8, cmap=cmap)
    axes[1,0].set_xlabel(getlabel(histo8,1), fontsize=fontsize)
    axes[1,0].set_ylabel(getlabel(histo8,2), fontsize=fontsize)
    
    add_2dplot(axes[1,1],histo8, (1,0),log=True, usepcolor = True, vmin = vmin, vmax = vmax_o8, cmap=cmap)
    axes[1,1].set_xlabel(getlabel(histo8,1), fontsize=fontsize)
    axes[1,1].set_ylabel(getlabel(histo8,0), fontsize=fontsize)
    
    add_2dplot(axes[1,2],histo8, (2,0),log=True, usepcolor = True, vmin = vmin, vmax = vmax_o8, cmap=cmap)
    axes[1,2].set_xlabel(getlabel(histo8,2), fontsize=fontsize)
    axes[1,2].set_ylabel(getlabel(histo8,0), fontsize=fontsize)
    
    add_colorbar(axes[1,3],img=img_o8, clabel= r'$\log_{10}$ fraction of sightlines' ,newax=False,extend='min',fontsize=fontsize,orientation='vertical')
    axes[1,3].set_aspect(10*(xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')
    
    add_fsolx(axes[(0,0)], 'o7', color = 'salmon', label = 'solar')
    add_fsoly(axes[(0,0)], 'o7', color = 'salmon', label = None)
    add_fsolx(axes[(0,1)], 'o7', color = 'salmon', label = 'solar')
    add_fsolx(axes[(0,2)], 'o7', color = 'salmon', label = 'solar')
    
    add_fsolx(axes[(1,0)], 'o8', color = 'salmon', label = 'solar')
    add_fsoly(axes[(1,0)], 'o8', color = 'salmon', label = 'solar')
    add_fsolx(axes[(1,1)], 'o8', color = 'salmon', label = 'solar')
    add_fsolx(axes[(1,2)], 'o8', color = 'salmon', label = 'solar')
    
    axes[(0,0)].legend(loc = 'upper left', fontsize = fontsize)
    
    if slices == 1:	    
        plt.savefig(mdir + 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_Oabund_by_mass_No78.png', format = 'png',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Oabund_by_mass_No78.png', format = 'png',bbox_inches='tight')
        
        
def plotOabund_mass_vs_ionweighted_by_o78(slices = 16, ion = 'o7', slidemode = False):
    
    if slices == 16:
        if ion == 'o7':
            hist = eafOo7
            ionlab = 'O VII'
        else:
            hist = eafOo8
            ionlab = 'O VIII'
    elif slices == 1:
        if ion == 'o7': 
            hist = eafOo7_100
            ionlab = 'O VII'
        else: 
            hist = eafOo8_100
            ionlab = 'O VIII'

    vmin = -8.
    cmap = 'bone'
    fontsize =12.
    
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'black') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'black')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(-8.,hist['edges'][2][-1])
    ax1.set_ylim(-8.,hist['edges'][1][-1])
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax1.set_xlabel(getlabel(hist,2),fontsize=fontsize)
    
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
        
   
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (2,1),log=True, usepcolor = True, vmin = vmin, cmap=cmap)
    # add colorbar

    add_colorbar(ax2,img=img,clabel=r'$\log_{10}$  fraction of sightlines', extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (r'$N_{%s}$'%ionlab, None, None)
    
    # col.dens. 12, 14, 15, 16, 17,
    if ion == 'o7' and slices == 16:
        lims = [71,91,101,111,121]
    elif ion == 'o7' and slices == 1:
        lims = [34,54,64,74,84]
    if ion == 'o8' and slices == 16:
        lims = [44,64,74,84,94]
    elif ion == 'o8' and slices == 1:
        lims = [26,46,56,66,76]

    add_2dhist_contours(ax1,hist,(2,1),mins= (None,None,None), maxs=(lims[0],None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(2,1),mins= (lims[0],None, None), maxs=(lims[1],None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(2,1),mins= (lims[1],None, None), maxs=(lims[2],None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(2,1),mins= (lims[2],None, None), maxs=(lims[3],None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(2,1),mins= (lims[3],None, None), maxs=(lims[4],None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(2,1),mins= (lims[4],None, None), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
    #set solar indicator
    add_fsolx(ax1, ion, color = 'salmon', label = 'solar')
    add_fsoly(ax1, ion, color = 'salmon', label = None)
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if slidemode:
        send = '_slide'
    else:
        send = ''
        
    if slices == 1:	    
        plt.savefig(mdir + 'coldens_%s_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_Oabund_mass-%s_by_N%s%s.png'%(ion,ion,ion,send), format = 'png',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_%s_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Oabund_by_mass-%s_by_N%s%s.png'%(ion,ion,ion,send), format = 'png',bbox_inches='tight')
        


        

def implot_regions(xpix, ypix):
    '''
    plot N, rho, T, Z maps for o7 and o8 in one-slice zoom regions
    '''
    fontsize =12.
    mdir = '/net/luttero/data2/imgs/coldens_maps_nice/'
    
    o7 = np.load('/net/luttero/data2/temp/coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_T4EOS.npz', mmap_mode = 'r')
    o8 = np.load('/net/luttero/data2/temp/coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_T4EOS.npz', mmap_mode = 'r')
    
    rho_o7 = np.load('/net/luttero/data2/temp/Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen3.125_z-projection.npz', mmap_mode = 'r')
    rho_o8 = np.load('/net/luttero/data2/temp/Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen3.125_z-projection.npz', mmap_mode = 'r')
    
    T_o7 = np.load('/net/luttero/data2/temp/Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen3.125_z-projection.npz', mmap_mode = 'r')
    T_o8 = np.load('/net/luttero/data2/temp/Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen3.125_z-projection.npz', mmap_mode = 'r')
    
    fO_o7 = np.load('/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen3.125_z-projection.npz', mmap_mode = 'r')
    fO_o8 = np.load('/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen3.125_z-projection.npz', mmap_mode = 'r')
    
    sel = (slice(*xpix),slice(*ypix))
    extent = (xpix[0]/32000.*100., xpix[1]/32000.*100., ypix[0]/32000.*100., ypix[1]/32000.*100.)
 
    o7im     = o7['arr_0'][sel]
    o8im     = o8['arr_0'][sel]
    rho_o7im = rho_o7['arr_0'][sel]
    rho_o8im = rho_o8['arr_0'][sel]
    T_o7im   = T_o7['arr_0'][sel]
    T_o8im   = T_o8['arr_0'][sel]
    fO_o7im  = fO_o7['arr_0'][sel]
    fO_o8im  = fO_o8['arr_0'][sel]
    
    fig = plt.figure(figsize = (22.,7.5))
    grid = gsp.GridSpec(2,8,width_ratios=[5., 1., 5., 1., 5., 1., 5., 1.], height_ratios =  [5., 5.], wspace=0.35,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in 2*np.arange(4)] for xi in range(2)]) # in mainaxes: x = column, y = row
    caxes = [fig.add_subplot(grid[0:2, ind]) for ind in 2*np.arange(4) +1 ] 
    
    vmin_N = 12.
    vmax_N = 17.
    vmin_rho = -32.
    vmax_rho = -23.
    vmin_T = 4.
    vmax_T = 8.
    vmin_fO = -5.
    vmax_fO = -1.5
    
    cmapN = 'cubehelix'
    cmap_rho = 'plasma'
    cmap_T = 'inferno'
    cmap_fO = 'magma'
    
    for flatind in range(8):
        xind = flatind%4
        yind = flatind/4
        ax = mainaxes[yind, xind]
        
        doylabel = xind == 0
        dobottomlabel = yind == 1
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'white',\
                   labelleft = doylabel, labeltop = False, labelbottom = dobottomlabel, labelright = False)
        ax.tick_params(length =7., width = 1., which = 'major')
        ax.tick_params(length =4., width = 1., which = 'minor')
        if doylabel:
            ax.set_ylabel('Y [cMpc]', fontsize = fontsize)
        if dobottomlabel:
            ax.set_xlabel('X [cMpc]', fontsize = fontsize)
    
    for ind in range(4):
        caxes[ind].tick_params(labelsize=fontsize, axis='both', which = 'both')

    i0 = mainaxes[0,0].imshow(o7im.T, origin = 'lower', interpolation='nearest', vmin = vmin_N, vmax = vmax_N, extent = extent, cmap = cmapN)   
    mainaxes[1,0].imshow(o8im.T, origin = 'lower', interpolation='nearest', vmin = vmin_N, vmax = vmax_N, extent = extent, cmap = cmapN)
    
    i1 = mainaxes[0,1].imshow(rho_o7im.T, origin = 'lower', interpolation='nearest', vmin = vmin_rho, vmax = vmax_rho, extent = extent, cmap = cmap_rho)   
    mainaxes[1,1].imshow(rho_o8im.T, origin = 'lower', interpolation='nearest', vmin = vmin_rho, vmax = vmax_rho, extent = extent, cmap = cmap_rho)
    
    i2 = mainaxes[0,2].imshow(T_o7im.T, origin = 'lower', interpolation='nearest', vmin = vmin_T, vmax = vmax_T, extent = extent, cmap = cmap_T)   
    mainaxes[1,2].imshow(T_o8im.T, origin = 'lower', interpolation='nearest', vmin = vmin_T, vmax = vmax_T, extent = extent, cmap = cmap_T)
    
    i3 = mainaxes[0,3].imshow(fO_o7im.T, origin = 'lower', interpolation='nearest', vmin = vmin_fO, vmax = vmax_fO, extent = extent, cmap = cmap_fO)   
    mainaxes[1,3].imshow(fO_o8im.T, origin = 'lower', interpolation='nearest', vmin = vmin_fO, vmax = vmax_fO, extent = extent, cmap = cmap_fO)
    
    mainaxes[0,0].text(0.95,0.95,'O VII',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    mainaxes[1,0].text(0.95,0.95,'O VIII',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    
    add_colorbar(caxes[0],img=i0,vmin=None,vmax=None,cmap=None,clabel=r'$\log_{10} N \, [\mathrm{cm}^{-2}]$',newax=False,extend='both',fontsize=fontsize,orientation='vertical')
    add_colorbar(caxes[1],img=None, vmin=vmin_rho -logrhob_av_ea ,vmax=vmax_rho -logrhob_av_ea,cmap=cmap_rho,clabel=r'$\log_{10} \delta$',newax=False,extend='both',fontsize=fontsize,orientation='vertical')
    add_colorbar(caxes[2],img=i2,vmin=None,vmax=None,cmap=None,clabel=r'$\log_{10} T \, [\mathrm{K}]$',newax=False,extend='both',fontsize=fontsize,orientation='vertical')
    add_colorbar(caxes[3],img=None,vmin= vmin_fO - np.log10(ol.solar_abunds['oxygen']),vmax= vmax_fO - np.log10(ol.solar_abunds['oxygen']),cmap=cmap_fO,clabel=r'$\log_{10} Z \, [Z_{\odot}]$',newax=False,extend='both',fontsize=fontsize,orientation='vertical')

    plt.savefig(mdir+ 'coldens_rho_T_fO_ionwieghted_T4EOS_o78_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen3.125_z-projection_T4EOS_xpix-%i-%i_ypix-%i-%i.pdf'%(xpix[0],xpix[1],ypix[0],ypix[1]), bbox_inches = 'tight', format = 'pdf')

##############################################################################
############# stuff for in the paper #########################################
##############################################################################
        
def plotrho_T_by_N_o78_28_16sl(ion='o7', fontsize=12, slidemode =False):
    
    if ion == 'o6':
        hist = ea3o6
        ionlab = 'O VI'
        o6_ib, logTKo6_ib, lognHcm3o6_ib = m3.findiontables('o6',0.0)
        ib = o6_ib
        rhoax = 1
        Tax = 2
        Nax = 0
    elif ion == 'o7':
        hist = ea4o7
        ionlab = 'O\,VII'
        ib = o7_ib
        rhoax = 0
        Tax = 1
        Nax = 2
    elif ion == 'o8':
        hist = ea4o8
        ionlab = 'O\,VIII'
        ib = o8_ib
        rhoax = 0
        Tax = 1
        Nax = 2
    elif ion == 'ne8':
        hist = ea3ne8
        ne8_ib, logTKne8_ib, lognHcm3ne8_ib = m3.findiontables('ne8',0.0)
        ib = ne8_ib
        ionlab = 'Ne\, VIII'
        rhoax = 1
        Tax = 2
        Nax = 0
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(5.5, 7.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.], wspace=0.0, hspace=0.25)
        ax1 = plt.subplot(grid[0,0], facecolor = 'white') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'white')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(-32.,hist['edges'][rhoax][-1])
    ax1.set_ylim(3.,8.)
    ax1.set_ylabel(getlabel(hist,Tax),fontsize=fontsize)
    ax1.set_xlabel(getlabel(hist,rhoax),fontsize=fontsize)
    
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
        
   
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_ionbal_img(ax1, cmap='gist_gray_r', vmin=-5., ib=ib)
    # add colorbar

    add_colorbar(ax2,img=img,clabel=r'$\log_{10} \, n_{%s} \,/ n_{O},\, f_H = 0.752$'%(ionlab), extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = [None, None, None]
    dimlabels[Nax] = r'$N_{%s}$'%ionlab
    
    if ion == 'o7' or ion == 'o8':
        lims = [44,64,74,84,94]
    elif ion == 'o6':
        lims = [34, 44, 54, 64, 74]
    elif ion == 'ne8':
        lims = [34, 44, 54, 64, 74]
        
    maxs = [list((None,) * (len(hist['edges']))) for i in range(len(lims) + 1)]
    maxs = [tuple(maxs[i][:Nax] + [lims[i]] + maxs[i][Nax+1:]) if i < len(lims) else \
            tuple(maxs[i]) \
            for i in range(len(lims) + 1) ]
    mins = [list((None,) * (len(hist['edges']))) for i in range(len(lims) + 1)]
    mins = [tuple(mins[i][:Nax] + [lims[i-1]] + mins[i][Nax+1:]) if i > 0 else \
            tuple(mins[i]) \
            for i in range(len(lims) + 1) ]
    Ncolors = ['red', 'orange', 'gold', 'green', 'dodgerblue', 'darkviolet']
    
    for i in range(len(lims) + 1):
        add_2dhist_contours(ax1,hist,(rhoax, Tax), mins=mins[i], maxs=maxs[i], histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles=linestyles, colors=(Ncolors[i],)*len(linestyles) ,dimlabels=dimlabels,\
			        legendlabel_pre = None)			    
#    add_2dhist_contours(ax1,hist,(rhoax, Tax),mins= (None,None,lims[0],None), maxs=(None,None,lims[1],None), histlegend=False, fraclevels=True,\
#                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
#			    legendlabel_pre = None)
#    add_2dhist_contours(ax1,hist,(rhoax, Tax),mins= (None,None,lims[1],None), maxs=(None,None,lims[2],None), histlegend=False, fraclevels=True,\
#                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
#			    legendlabel_pre = None)
#    add_2dhist_contours(ax1,hist,(rhoax, Tax),mins= (None,None,lims[2],None), maxs=(None,None,lims[3],None), histlegend=False, fraclevels=True,\
#                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
#			    legendlabel_pre = None)
#    add_2dhist_contours(ax1,hist,(rhoax, Tax),mins= (None,None,lims[3],None), maxs=(None,None,lims[4],None), histlegend=False, fraclevels=True,\
#                            levels=fraclevels, linestyles = linestyles,colors = ['dodgerblue','dodgerblue','dodgerblue'],dimlabels = dimlabels,\
#			    legendlabel_pre = None)
#    add_2dhist_contours(ax1,hist,(rhoax, Tax),mins= (None,None,lims[4],None), maxs=(None,None,None,None), histlegend=False, fraclevels=True,\
#                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
#			    legendlabel_pre = None)
	
    #set average density indicator
    add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
	## add cooling contours
    # calculate cooling time, plot contours
    simfile = sfc.Simfileclone(sfc.Zvar_rhoT_z0)
    vardict = m3.Vardict(simfile, 0, []) # Mass should not be read in, all other entries should be cleaned up

    logTvals   = np.log10(sfc.Tvals)
    logrhovals = np.log10(sfc.rhovals * m3.c.unitdensity_in_cgs)
    lognHvals_sol  = logrhovals + np.log10( sfc.dct_sol['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    #lognHvals_0p1  = logrhovals + np.log10( sfc.dct_0p1['Hydrogen']    / (m3.c.u*m3.c.atomw_H) )
    lognHvals_pri  = logrhovals + np.log10( sfc.dct_pri['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) )
    
    #logdeltavals = logrhovals - np.log10( ( 3./(8.*np.pi*m3.c.gravity)*m3.c.hubble**2 * m3.c.hubbleparam**2 * m3.c.omegabaryon) )
        
    #abundsdct_0p1 = sfc.dct_0p1
    # in simflieclone.Zvar_rhoT_z0, smoothed metallicites are primoridial and particle metallicites are solar
    outshape = (len(logrhovals), len(logTvals))
    tcool_perelt_sol = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    #tcool_perelt_0p1 = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab=abundsdct_0p1['Hydrogen'], abunds=abundsdct_0p1, last=False)).reshape(outshape)
    tcool_perelt_pri = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
    
    tH = 1./m3.Hubble(0.)
    levels_tcool = [0.1*tH, tH]
    levels_tcool_labels = [r'$t_c = 0.1 t_H$', r'$t_c = t_H$']
    linestyles_tcool = ['dashdot', 'solid']
    colors_tcool = ['darkcyan', 'olive']
    colors_tcool_labels = [r'$t_c(Z=Z_{\odot})$', r'$t_c(Z=0)$']
    
    contourdct = {'sol':{'x': lognHvals_sol, 'y': logTvals, 'z': np.abs(tcool_perelt_sol),\
                         'kwargs': {'colors': colors_tcool[0], 'levels':levels_tcool, 'linestyles':linestyles_tcool}},\
                  'pri':{'x': lognHvals_pri, 'y': logTvals, 'z': np.abs(tcool_perelt_pri),\
                         'kwargs': {'colors': colors_tcool[1], 'levels':levels_tcool, 'linestyles':linestyles_tcool}} }
    #'0p1':{'x': lognHvals_0p1, 'y': logTvals, 'z': np.abs(tcool_perelt_0p1),\
    #                     'kwargs': {'colors': colors_tcool[1], 'levels':levels_tcool, 'linestyles':linestyles_tcool}},\
    if contourdct is not None:
        for key in contourdct.keys():
            toplot = contourdct[key]
            ax1.contour(toplot['x'] - np.log10(rho_to_nh), toplot['y'], toplot['z'].T, **(toplot['kwargs']))
            
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'black')
    
    ax12.minorticks_on()
    ax12.tick_params(labelsize=fontsize,direction = 'in', right = False, left = False, top = True, bottom = False, axis='both', which = 'both', color = 'black')
    
    # set solar oxygen fraction reference    
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    handles_tcool_linestyles = [mlines.Line2D([], [], color='gray', linestyle =linestyles_tcool[i], label=levels_tcool_labels[i]) for i in range(len(levels_tcool))]
    handles_tcool_colors = [mlines.Line2D([], [], color=colors_tcool[i], linestyle = 'solid', label=colors_tcool_labels[i]) for i in range(len(colors_tcool))]
    
    ax3.legend(handles=handles_subs + level_legend_handles + handles_tcool_linestyles + handles_tcool_colors,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'phase_diagrams_by_N%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all.pdf'%ion ,format = 'pdf',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'phase_diagrams_by_N%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all_slide.png'%ion,format = 'png',bbox_inches='tight')
        


def plotrho_T_by_N_o78_28_16sl_simplified(ion = 'o7', fontsize=12, slidemode =False):
    
    if ion == 'o7':
        hist = ea4o7
        ionlab = 'O VII'
        ib = o7_ib
    elif ion == 'o8':
        hist = ea4o8
        ionlab = 'O VIII'
        ib = o8_ib
        
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,6.))
        grid = gsp.GridSpec(1,2,width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'black') 
        ax2 = plt.subplot(grid[0,1])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=20
        fig = plt.figure(figsize=(8.,6.))
        grid = gsp.GridSpec(1,2,width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'black') 
        ax2 = plt.subplot(grid[0,1])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(-32.,hist['edges'][0][-1])
    ax1.set_ylim(2.9,8.1)
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
        
   
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_ionbal_img(ax1,cmap='gist_gray',vmin=-5., ib=ib)
    # add colorbar

    add_colorbar(ax2,img=img,clabel=r'$\log_{10} \, n_{%s} \,/ n_{O},\, f_H = 0.752$'%(ionlab), extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50] 
    linestyles = ['dashed','solid']
    dimlabels = (None,None,r'$N_{%s}$'%ionlab)
    
    if ion == 'o7' or ion == 'o8':
        lims = [44,64,74,84,94]

    add_2dhist_contours(ax1,hist,(0,1),mins= (None,None,None,None), maxs=(None,None,lims[0],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(0,1),mins= (None,None,lims[0],None), maxs=(None,None,lims[1],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['saddlebrown','saddlebrown','saddlebrown'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,1),mins= (None,None,lims[1],None), maxs=(None,None,lims[2],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,1),mins= (None,None,lims[2],None), maxs=(None,None,lims[3],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkslategray', 'darkslategray', 'darkslategray'] ,dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,1),mins= (None,None,lims[3],None), maxs=(None,None,lims[4],None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(0,1),mins= (None,None,lims[4],None), maxs=(None,None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
    #set average density indicator
    #add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon', fontsize=fontsize)
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = False, axis='both', which = 'both', color = 'white')
    
    ax12.minorticks_on()
    ax12.tick_params(labelsize=fontsize,direction = 'in', right = False, left = False, top = True, bottom = False, axis='both', which = 'both', color = 'white')
    
    # set solar oxygen fraction reference    
    ax1.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    
    if ion == 'o7':
        ax1.text(0.05,0.95,r'O VII: $\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'white', transform=ax1.transAxes) # bbox=dict(facecolor='white',alpha=0.3)
        ax1.text(-30.0,3.5,r'$< %.0f$'%(hist['edges'][2][lims[0]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'red',  bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-29.3,4.0,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[0]], hist['edges'][2][lims[1]] ),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'saddlebrown', bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-28.5,5.0,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[1]], hist['edges'][2][lims[2]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'green', bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-28.0,5.5,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[2]], hist['edges'][2][lims[3]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'darkslategray',  bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-27.2,6.5,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[3]], hist['edges'][2][lims[4]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', color = 'blue',  bbox=dict(facecolor='moccasin',alpha=0.6)) 
        ax1.text(-25.0,6.5,r'$ >%.0f$'%(hist['edges'][2][lims[4]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', color = 'purple',  bbox=dict(facecolor='moccasin',alpha=0.6)) 
    
    if ion == 'o8':
        ax1.text(0.05,0.95,r'O VIII: $\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'white', transform=ax1.transAxes) # bbox=dict(facecolor='white',alpha=0.3)
        ax1.text(-30.0,3.5,r'$< %.0f$'%(hist['edges'][2][lims[0]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'red',  bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-29.1,4.5,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[0]], hist['edges'][2][lims[1]] ),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'saddlebrown', bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-28.7,5.2,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[1]], hist['edges'][2][lims[2]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'green', bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-28.0,5.7,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[2]], hist['edges'][2][lims[3]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'darkslategray',  bbox=dict(facecolor='moccasin',alpha=0.6))
        ax1.text(-26.9,6.2,r'$%.0f$ - $%.0f$'%(hist['edges'][2][lims[3]], hist['edges'][2][lims[4]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'blue',  bbox=dict(facecolor='moccasin',alpha=0.6)) 
        ax1.text(-24.7,6.3,r'$ >%.0f$'%(hist['edges'][2][lims[4]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'purple',  bbox=dict(facecolor='moccasin',alpha=0.6)) 
    
    # set up legend in ax below main figure
    #handles_subs, labels_subs = ax1.get_legend_handles_labels()
    #level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    #ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    #ax3.axis('off')	
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    

    
    if not slidemode:	    
        plt.savefig(mdir + 'phase_diagrams_by_N%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all_simplified.pdf'%ion ,format = 'pdf',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'phase_diagrams_by_N%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all_simplified_slide.png'%ion,format = 'png',bbox_inches='tight',dpi=300)
        
        

def plot_cor_o678_27_1sl(fontsize=12, slidemode=False):
    
    hist = ea3o678
    ionlab = 'O\,VI'
        
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(5.5,5.5))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'white') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'white')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range (max  = 0.05 + max included in hist)
    #ax1.set_xlim(8.,16.25)
    #ax1.set_ylim(9.,17.25)
    ax1.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,2),fontsize=fontsize)
        
   
    # plot backgorund and colorbar: total distribution
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (1,2),log=True, usepcolor = True, vmin = -8., cmap='bone_r')
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10}$ fraction of sightlines',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (r'$N_{%s}$'%ionlab,None,None)

    lims = [31,51,61,71,81]

    add_2dhist_contours(ax1,hist,(1,2),mins= (None,None,None), maxs=(lims[0], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[0], None, None), maxs=(lims[1], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[1], None, None), maxs=(lims[2], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[2], None, None), maxs=(lims[3], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[3], None, None), maxs=(lims[4], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[4], None, None), maxs=(None, None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'black')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'coldens_o7-o8_by_o6_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS.pdf' ,format = 'pdf',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_o7-o8_by_o6_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_slide.png',format = 'png',bbox_inches='tight')


def subplot_mass_phasediagram(ax, bins_sub, logrho, logT, contourdct=None, xlim = None, ylim = None, vmin = None, vmax = None, cmap = 'viridis',\
                              dobottomlabel = False, doylabel = False, subplotlabel = None, subplotind = None, logrhob = None,\
                              fraclevels = None, linestyles = None, fontsize = fontsize, subplotindheight = 0.92):
    '''
    do the subplot, and sort out the axes
    rhoav is set for snapshot 28
    '''
    img = ax.pcolormesh(logrho, logT, np.log10(bins_sub.T), cmap = cmap, vmin = vmin, vmax = vmax)

    if contourdct is not None:
        for key in contourdct.keys():
            toplot = contourdct[key]
            ax.contour(toplot['x'], toplot['y'], toplot['z'].T, **(toplot['kwargs']))
                
    if fraclevels is not None:
        add_2dhist_contours(ax, {'bins': bins_sub, 'edges': [logrho, logT]},(0,1),mins= None, maxs=None, histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['fuchsia','fuchsia','fuchsia'],dimlabels = (None, None),\
			                 legendlabel = 'mass')	
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlim is not None:
        ax.set_ylim(ylim)
        
    # square plot
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    #set average density indicator
    add_rhoavx(ax,onlyeagle=True,eacolor='gray',userhob_ea = logrhob + np.log10(rho_to_nh), linewidth=2)
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'white',\
                   labelleft = doylabel, labeltop = False, labelbottom = dobottomlabel, labelright = False)
    ax.tick_params(length =7., width = 1., which = 'major')
    ax.tick_params(length =4., width = 1., which = 'minor')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 6, prune='lower'))
    
    ## plot temperature floor
    #ax.plot([-5.,-5.],[ 1., np.log10(8000.)], color = 'red')
    #ax.plot([-5.,-1.],[np.log10(8000.), np.log10(8000.)], color = 'red')
    #ax.plot([-1.,4.],[np.log10(8000.), np.log10( 8000.*(1.e4/1.e-1)**(1./3.) )], color = 'red')
    
    
    
    # axis labels: somewhat abbreviated
    if doylabel:
        ax.set_ylabel(r'$\log_{10} T \, [K]$',fontsize=fontsize)
    if dobottomlabel:
        ax.set_xlabel(r'$\log_{10} n_H \, [\mathrm{cm}^{-3}]$',fontsize=fontsize)
    if subplotlabel is not None:
        ax.text(0.92,0.92 ,subplotlabel,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, color = 'white') # bbox=dict(facecolor='white',alpha=0.3)
    if subplotind is not None:
        ax.text(0.92, subplotindheight,subplotind,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes, color = 'white')
    return img


def plot_mass_phase_diagrams_28():

    # just hard-code loading: these histograms are small and pre-normalised
    # keys: hist, logT, lognH. hist = array: logT x lognH
    
    import simfileclone as sfc
    
    mass = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_mass-weighted_normed.npz')    
    ox = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_O-mass-weighted_normed.npz')
    o7 = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_O7-mass-weighted_normed.npz')
    o8 = dictfromnpz('/net/luttero/data2/temp/phase_diagram_v3_L0100N1504_28_PtAb_T4EOS_O8-mass-weighted_normed.npz')
    logrhob = logrhob_av_ea # z=0

    # calculate cooling time, plot contours
    simfile = sfc.Simfileclone(sfc.Zvar_rhoT_z0)
    vardict = m3.Vardict(simfile, 0, []) # Mass should not be read in, all other entries should be cleaned up

    logTvals   = np.log10(sfc.Tvals)
    logrhovals = np.log10(sfc.rhovals * m3.c.unitdensity_in_cgs)
    lognHvals_sol  = logrhovals + np.log10( sfc.dct_sol['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    #lognHvals_0p1  = logrhovals + np.log10( sfc.dct_0p1['Hydrogen']    / (m3.c.u*m3.c.atomw_H) )
    lognHvals_pri  = logrhovals + np.log10( sfc.dct_pri['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) )
    
    #logdeltavals = logrhovals - np.log10( ( 3./(8.*np.pi*m3.c.gravity)*m3.c.hubble**2 * m3.c.hubbleparam**2 * m3.c.omegabaryon) )
        
    #abundsdct_0p1 = sfc.dct_0p1
    # in simflieclone.Zvar_rhoT_z0, smoothed metallicites are primoridial and particle metallicites are solar
    outshape = (len(logrhovals), len(logTvals))
    tcool_perelt_sol = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    #tcool_perelt_0p1 = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab=abundsdct_0p1['Hydrogen'], abunds=abundsdct_0p1, last=False)).reshape(outshape)
    tcool_perelt_pri = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
    
    tH = 1./m3.Hubble(0.)
    levels_tcool = [0.1*tH, tH]
    levels_tcool_labels = [r'$0.1 t_H$', r'$t_H$']
    linestyles_tcool = ['dashdot', 'solid']
    colors_tcool = ['red', 'cyan']
    colors_tcool_labels = [r'$Z=Z_{\odot}$', r'$Z=0$']
    
    contourdct = {'sol':{'x': lognHvals_sol, 'y': logTvals, 'z': np.abs(tcool_perelt_sol),\
                         'kwargs': {'colors': colors_tcool[0], 'levels':levels_tcool, 'linestyles':linestyles_tcool}},\
                  'pri':{'x': lognHvals_pri, 'y': logTvals, 'z': np.abs(tcool_perelt_pri),\
                         'kwargs': {'colors': colors_tcool[1], 'levels':levels_tcool, 'linestyles':linestyles_tcool}} }
    #'0p1':{'x': lognHvals_0p1, 'y': logTvals, 'z': np.abs(tcool_perelt_0p1),\
    #                     'kwargs': {'colors': colors_tcool[1], 'levels':levels_tcool, 'linestyles':linestyles_tcool}},\
    fontsize = 20
    fig = plt.figure(figsize=(9.5 ,7.99)) # figsize: width, height
    numx=2
    numy=2
 
    grid = gsp.GridSpec(2,3,width_ratios=[4.,4.,1.], height_ratios = [4.,4.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05) # grispec: nrows, ncols
    mainaxes = [[fig.add_subplot(grid[yi,xi]) for yi in range(numy)] for xi in range(numx)] # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[0:2, numx]) 
    #legax = fig.add_subplot(grid[numy, :])    

    linestyles = ['dotted', 'dashed', 'solid']
    #legendloc= 'lower center'
    #legend_bbox_to_anchor=(0.5,0.0)
    fraclevels = [0.99,0.50,0.10] 
    #ncol_legend = 2
    
    xlim = (mass['lognH'][0], mass['lognH'][-1])
    ylim = (mass['logT'][0], mass['logT'][-1])
    
    subplotbins = [[mass['hist'].T, o7['hist'].T],[ox['hist'].T, o8['hist'].T]]
    subplotT = [[mass['logT'],o7['logT']],[ox['logT'],o8['logT']]]
    subplotnH = [[mass['lognH'],o7['lognH']],[ox['lognH'],o8['lognH']]]
    subplotinds = [['(a)','(c)'],['(b)','(d)']]
    subplotlabels = [['mass', 'O VII mass'],['O mass','O VIII mass']]
    # set np.inf as min for subplot if all values zero ->  get min of nonzero values
    vmin = -10.
    #min([min([ np.min(subplotbins[xi][yi][np.isfinite(subplotbins[xi][yi])]) if np.any(np.isfinite(subplotbins[xi][yi])) else\
    #                          np.inf\
    #                          for yi in range(numy)]) for xi in range(numx)])
    vmax = max([max([ np.max(subplotbins[xi][yi][np.isfinite(subplotbins[xi][yi])])  for yi in range(numy)]) for xi in range(numx)])
    
    for flatind in range(numx*numy):
        
        xi = flatind/numx
        yi = flatind%numx
        doylabel    = (xi == 0)
        doxlabel    = (yi == numy-1)
        if doxlabel:
            subplotindheight = 0.45
        else:
            subplotindheight = 0.45
        imgsub = subplot_mass_phasediagram(mainaxes[xi][yi], subplotbins[xi][yi], subplotnH[xi][yi], subplotT[xi][yi], contourdct=contourdct,\
                              xlim = xlim, ylim = ylim, vmin = vmin, vmax = vmax, cmap = 'viridis',\
                              dobottomlabel = doxlabel, doylabel = doylabel, subplotlabel = subplotlabels[xi][yi], subplotind = subplotinds[xi][yi], logrhob = logrhob,\
                              fraclevels = fraclevels, linestyles = linestyles, fontsize = fontsize, subplotindheight = subplotindheight)
        if flatind ==0:
            img = imgsub

    add_colorbar(cax,img=img,clabel=r'$\log_{10}$ mass fraction',fontsize=fontsize, extend = 'min')
    cax.set_aspect(15.)
    cax.tick_params(labelsize=fontsize,axis='both')

    # get edges in nH units from rho units: plot sightline enclosed fractions
    edges_o7 = ea4o7['edges']
    edges_o7[0] = edges_o7[0].copy() + np.log10(rho_to_nh)
    ea4o7_nHunits = {'bins': ea4o7['bins'], 'edges': edges_o7}
    add_2dhist_contours(mainaxes[0][1], ea4o7_nHunits, (0,1), mins= None, maxs=None, histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['chocolate','chocolate','chocolate'],dimlabels = (None, None),\
			                 legendlabel = 'sightl.')
    edges_o8 = ea4o8['edges']
    edges_o8[0] = edges_o8[0].copy() + np.log10(rho_to_nh)
    ea4o8_nHunits = {'bins': ea4o8['bins'], 'edges': edges_o8}
    
    add_2dhist_contours(mainaxes[1][1], ea4o8_nHunits, (0,1), mins= None, maxs=None, histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['chocolate','chocolate','chocolate'],dimlabels = (None, None),\
			                 legendlabel = 'sightls.')

    
    

    
    # set up legends
    handles_tcool_colors = [mlines.Line2D([], [], color=colors_tcool[i], linestyle = 'solid', label=colors_tcool_labels[i]) for i in range(len(colors_tcool))]
    leg3 = mainaxes[0][0].legend(handles=handles_tcool_colors, fontsize=fontsize,ncol=1,loc='lower right',bbox_to_anchor=(0.995, 0.16), framealpha = 0., edgecolor = 'lightgray')
    plt.setp(leg3.get_texts(), color='white')
    
    handles_tcool_linestyles = [mlines.Line2D([], [], color='tan', linestyle =linestyles_tcool[i], label=levels_tcool_labels[i]) for i in range(len(levels_tcool))]
    leg4 = mainaxes[1][0].legend(handles=handles_tcool_linestyles, fontsize=fontsize,ncol=1,loc='lower right',bbox_to_anchor=(0.995, 0.16), framealpha = 0., edgecolor = 'lightgray')
    plt.setp(leg4.get_texts(), color='white')
    
    handles_subs, labels_subs = mainaxes[0][1].get_legend_handles_labels()
    leg1 = mainaxes[0][1].legend(handles=handles_subs, fontsize=fontsize,ncol=1,loc='lower right',bbox_to_anchor=(0.995, 0.02), framealpha = 0., edgecolor = 'lightgray')
    plt.setp(leg1.get_texts(), color='white')
    
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%%'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    leg2 = mainaxes[1][1].legend(handles=level_legend_handles, fontsize=fontsize,ncol=1,loc='lower right',bbox_to_anchor=(0.995, 0.02), framealpha = 0., edgecolor = 'lightgray')
    plt.setp(leg2.get_texts(), color='white')
    #legax.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncol_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    #legax.axis('off')
    
 
    plt.savefig(mdir + 'phase_diagram_by_mass_v2_L0100N1504_28_PtAb_T4SFR.pdf' ,format = 'pdf',bbox_inches='tight') 


def plotcddf_decomposed_by_rho(relative = False, slidemode = False):
    
    hist_o7 = ea4o7
    hist_o8 = ea4o8
    logrhob_av = logrhob_av_ea
    
    lims_o7_rho = [33, 43, 53, 63] # densities -30.4 (~rhoav snap 28), + 1 dex 
    #lims_o7_T   = [20, 30, 34, 43, 50] # Temperatures 4, 5, 5.4, 6.3, 7
    
    lims_o8_rho = [21, 31, 41, 51] # densities -30.4 (~rhoav snap 28) + 1 dex
    #lims_o8_T = [20, 30, 34, 39, 50] # Temperatures 4, 5., 5.4, 5.9, 7 
    
    if slidemode:
        fontsize=14
        sfi_a = ''
        sfi_b = ''
    else:
        fontsize=12
        sfi_a = '(a)'
        sfi_b = '(b)'
    dXtot = mc.getdX(cosmopars_ea['z'],100.,cosmopars=cosmopars_ea)*32000**2
    
    fig = plt.figure(figsize=(5.5,4.2))
    grid = gsp.GridSpec(2,2,width_ratios=[5.,5.], height_ratios = [4.,3.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0,1]] for xi in [0]]) # in mainaxes: x = column, y = row
    lax = fig.add_subplot(grid[1,0:2]) 
    
    #fig, axes = plt.subplots(ncols=2, nrows=4, sharex = True, sharey=True,)
    
    edges_o7 = hist_o7['edges'][2]
    edges_o8 = hist_o8['edges'][2]
    
    rhobins_o7 = [np.sum(hist_o7['bins'][:,:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:lims_o7_rho[0],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][lims_o7_rho[0]:lims_o7_rho[1],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][lims_o7_rho[1]:lims_o7_rho[2],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][lims_o7_rho[2]:lims_o7_rho[3],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][lims_o7_rho[3]:,:,:,:],axis=(0,1,3))
                 ]
    
    if slidemode:
        labels_rho_o7 = ['total',\
                     r'$ \delta <%.1f $'%( hist_o7['edges'][0][lims_o7_rho[0]] -logrhob_av ),\
                     r'$ %.1f < \delta < %.1f $'%( hist_o7['edges'][0][lims_o7_rho[0]] -logrhob_av, hist_o7['edges'][0][lims_o7_rho[1]] -logrhob_av ),\
                     r'$ %.1f < \delta < %.1f $'%( hist_o7['edges'][0][lims_o7_rho[1]] -logrhob_av, hist_o7['edges'][0][lims_o7_rho[2]] -logrhob_av ),\
                     r'$ %.1f < \delta < %.1f $'%( hist_o7['edges'][0][lims_o7_rho[2]] -logrhob_av, hist_o7['edges'][0][lims_o7_rho[3]] -logrhob_av ),\
                     r'$ %.1f < \delta $'%( hist_o7['edges'][0][lims_o7_rho[3]] -logrhob_av )\
                    ]
    else:
        labels_rho_o7 = ['total',\
                     r'$ \delta <%.2f $'%( hist_o7['edges'][0][lims_o7_rho[0]] -logrhob_av ),\
                     r'$ %.2f < \delta < %.2f $'%( hist_o7['edges'][0][lims_o7_rho[0]] -logrhob_av, hist_o7['edges'][0][lims_o7_rho[1]] -logrhob_av ),\
                     r'$ %.2f < \delta < %.2f $'%( hist_o7['edges'][0][lims_o7_rho[1]] -logrhob_av, hist_o7['edges'][0][lims_o7_rho[2]] -logrhob_av ),\
                     r'$ %.2f < \delta < %.2f $'%( hist_o7['edges'][0][lims_o7_rho[2]] -logrhob_av, hist_o7['edges'][0][lims_o7_rho[3]] -logrhob_av ),\
                     r'$ %.2f < \delta $'%( hist_o7['edges'][0][lims_o7_rho[3]] -logrhob_av )\
                    ]
        
    if relative:
        binsedges_rho_o7 = [[np.array(list(bins/(rhobins_o7[0])) + [0.]), edges_o7] for bins in rhobins_o7] #-> dN/dlogNdX
        ylim_o7 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.35, 0.05)
        subtitleloc = (0.22,0.13)
    else:
        binsedges_rho_o7 = [[hist_o8.npt * np.array(list(bins/np.diff(edges_o7)/dXtot) + [0.]), edges_o7] for bins in rhobins_o7] #-> dN/dlogNdX 
        binsedges_rho_o7 = [[np.max(np.array([np.log10(be[0]), np.array((-100.,)*len(be[0])) ]), axis=0) , be[1]] for be in binsedges_rho_o7] # log, with zero -> -100 to get nicer cutoffs in the plot
        ylim_o7 = (-6.5,2.5)
        ylabel = r'$\log_{10}\left( \partial^2 n \,/\, \partial \log_{10} N \, \partial X \right)$'
        subfigindloc = (0.95,0.75)
        subtitleloc = None
                  
    
    colors_rho_o7 = ['black', 'red', 'orange', 'green', 'blue','purple']
    
    lnp.cddfsubplot1(axes[0,0], binsedges_rho_o7,subtitle = 'O VII',subfigind = sfi_a,\
                     xlabel=r'$\log_{10} N_{\mathrm{O\, VII}} \, [\mathrm{cm}^{-2}]$', ylabel = ylabel,\
                     colors = colors_rho_o7, labels = None, linestyles=None, fontsize =fontsize,\
                     xlim = (11.5,17.9),ylim=ylim_o7,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = False,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
    
    rhobins_o8 = [np.sum(hist_o8['bins'][:,:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:lims_o8_rho[0],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][lims_o8_rho[0]:lims_o8_rho[1],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][lims_o8_rho[1]:lims_o8_rho[2],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][lims_o8_rho[2]:lims_o8_rho[3],:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][lims_o8_rho[3]:,:,:,:],axis=(0,1,3))
                 ]
    if relative:
        binsedges_rho_o8 = [[np.array(list(bins/(rhobins_o8[0])) + [0.]), edges_o8] for bins in rhobins_o8] #-> dN/dlogNdX
        ylim_o8 = (1.e-2, 1.05)
        subfigindloc = (0.35, 0.05)
        subtitleloc = (0.22,0.13)
    else:
        binsedges_rho_o8 = [[hist_o8.npt * np.array(list(bins/np.diff(edges_o8)/dXtot) + [0.]), edges_o8] for bins in rhobins_o8] #-> dN/dlogNdX 
        binsedges_rho_o8 = [[np.max(np.array([np.log10(be[0]), np.array((-100.,)*len(be[0])) ]), axis=0) , be[1]] for be in binsedges_rho_o8]
        ylim_o8 = (-6.5,2.5)
        subfigindloc = (0.95,0.72)
     
                  
    labels_rho_o8 = ['total',\
                     r'$ \delta <%.2f $'%( hist_o8['edges'][0][lims_o8_rho[0]] -logrhob_av ),\
                     r'$ %.2f < \delta < %.2f $'%( hist_o8['edges'][0][lims_o8_rho[0]] -logrhob_av, hist_o8['edges'][0][lims_o8_rho[1]] -logrhob_av ),\
                     r'$ %.2f < \delta < %.2f $'%( hist_o8['edges'][0][lims_o8_rho[1]] -logrhob_av, hist_o8['edges'][0][lims_o8_rho[2]] -logrhob_av ),\
                     r'$ %.2f < \delta < %.2f $'%( hist_o8['edges'][0][lims_o8_rho[2]] -logrhob_av, hist_o8['edges'][0][lims_o8_rho[3]] -logrhob_av ),\
                     r'$ %.2f < \delta $'%( hist_o8['edges'][0][lims_o8_rho[3]] -logrhob_av )\
                    ]
    colors_rho_o8 = colors_rho_o7
    
    lnp.cddfsubplot1(axes[0,1], binsedges_rho_o8,subtitle = 'O VIII',subfigind = sfi_b,\
                     xlabel=r'$\log_{10} N_{\mathrm{O\, VIII}} \, [\mathrm{cm}^{-2}]$',ylabel = None,\
                     colors = colors_rho_o8, labels = None, linestyles=None, fontsize =fontsize,\
                     xlim = (11.5,17.9),ylim=ylim_o8,xticklabels=False,yticklabels=False,\
                     legendloc = None,legend_ncol=None,legend_title = r'$\log_{10} \rho \, [\mathrm{g}\, \mathrm{cm}^{-3}]$',\
                     ylog = False,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
    
    #legend_handles_rho_o8 = [mlines.Line2D([], [], color=colors_rho_o8[i], linestyle = 'solid', label=labels_rho_o8[i]) for i in range(len(rhobins_o8))]
    #legend_rho_o8 = axes[1,1].legend(handles = legend_handles_rho_o8, fontsize = fontsize, ncol =2)
    #legend_rho_o8.set_title(r'$\log_{10} \rho \, [\mathrm{g}\, \mathrm{cm}^{-3}]$',  prop = {'size': fontsize})
    
    legend_handles_rho_o7 = [mlines.Line2D([], [], color=colors_rho_o7[i], linestyle = 'solid', label=labels_rho_o7[i]) for i in range(len(rhobins_o7))]
    legend_rho_o7 = lax.legend(handles = legend_handles_rho_o7, fontsize = fontsize, ncol =2, loc = 'upper center', bbox_to_anchor=(0.5, 0.75))
    legend_rho_o7.set_title(r'$\log_{10} \delta$ ($= \rho_{N\, \mathrm{ion}}\, /\,\overline{\rho_b}$)',  prop = {'size': fontsize})
    lax.axis('off')

    # turn the y axis labels off for realzies
    axes[0][1].yaxis.set_major_formatter(dummyformatter)
    axes[0][1].yaxis.set_minor_formatter(dummyformatter)
    
    if relative:
        send = '_relative'
    else:
        send = ''
    
    if slidemode:
        plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_rho%s.png'%send, bbox_inches = 'tight', format = 'png', dpi=300)
    else:
        plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_rho%s.pdf'%send, bbox_inches = 'tight', format = 'pdf')
  
def plotcddf_decomposed_by_T(relative = False, slidemode = False):
    hist_o7 = ea4o7
    hist_o8 = ea4o8
    
    edges_o7 = hist_o7['edges'][2]
    edges_o8 = hist_o8['edges'][2]
    
    if slidemode:
        fontsize = 14
        sfi_a = ''
        sfi_b = ''
    else:
        fontsize=12
        sfi_a = '(a)'
        sfi_b = '(b)'
    dXtot = mc.getdX(cosmopars_ea['z'],100.,cosmopars=cosmopars_ea)*32000**2
    
    
    #lims_o7_rho = [33, 47, 57, 72] # densities -30.4 (~rhoav snap 28), -29, -28, -26.5
    if slidemode:
        lims_o7_T  = [34, 45]
        Tbins_o7 =   [np.sum(hist_o7['bins'][:,:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,:lims_o7_T[0],:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[0]:lims_o7_T[1],:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[1]:,:,:],axis=(0,1,3))\
                 ]
        labels_T_o7 = ['total',\
                     r'$<%s$'%str(hist_o7['edges'][1][lims_o7_T[0]]),\
                     r'$%s$-$%s$'%(str(hist_o7['edges'][1][lims_o7_T[0]]), str(hist_o7['edges'][1][lims_o7_T[1]])),\
                     r'$>%s$'%(str(hist_o7['edges'][1][lims_o7_T[1]]))\
                    ]
        colors_T_o7 = ['black', 'red', 'green', 'blue']
    else:
        lims_o7_T   = [30, 34, 45, 50] # Temperatures 5, 5.4, 6.5, 7
        Tbins_o7 =   [np.sum(hist_o7['bins'][:,:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,:lims_o7_T[0],:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[0]:lims_o7_T[1],:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[1]:lims_o7_T[2],:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[2]:lims_o7_T[3],:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[3]:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o7['bins'][:,lims_o7_T[0]:lims_o7_T[3],:,:],axis=(0,1,3))\
                 ]
        labels_T_o7 = ['total',\
                     r'$<%s$'%str(hist_o7['edges'][1][lims_o7_T[0]]),\
                     r'$%s$-$%s$'%(str(hist_o7['edges'][1][lims_o7_T[0]]), str(hist_o7['edges'][1][lims_o7_T[1]])),\
                     r'$%s$-$%s$'%(str(hist_o7['edges'][1][lims_o7_T[1]]), str(hist_o7['edges'][1][lims_o7_T[2]])),\
                     r'$%s$-$%s$'%(str(hist_o7['edges'][1][lims_o7_T[2]]), str(hist_o7['edges'][1][lims_o7_T[3]])),\
                     r'$>%s$'%(str(hist_o7['edges'][1][lims_o7_T[3]])),\
                     r'$%s$-$%s$'%(str(hist_o7['edges'][1][lims_o7_T[0]]), str(hist_o7['edges'][1][lims_o7_T[3]])),\
                    ]
        colors_T_o7 = ['black', 'red', 'orange', 'green', 'blue', 'purple', 'gray']
    
    
    #lims_o8_rho = [21, 35, 45, 60] # densities -30.4 (~rhoav snap 28), -29, -28, -26.5
    if slidemode:
        lims_o8_T = [34, 41, 48]
        Tbins_o8 =   [np.sum(hist_o8['bins'][:,:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,:lims_o8_T[0],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[0]:lims_o8_T[1],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[1]:lims_o8_T[2],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[2]:,:,:],axis=(0,1,3))\
                 ]
        labels_T_o8 = ['total',\
                     r'$<%s$'%str(hist_o8['edges'][1][lims_o8_T[0]]),\
                     r'$%s$-$%s$'%(str(hist_o8['edges'][1][lims_o8_T[0]]), str(hist_o8['edges'][1][lims_o8_T[1]])),\
                     r'$%s$-$%s$'%(str(hist_o8['edges'][1][lims_o8_T[1]]), str(hist_o8['edges'][1][lims_o8_T[2]])),\
                     r'$>%s$'%(str(hist_o8['edges'][1][lims_o8_T[2]])),\
                    ]
        colors_T_o8 = ['black', 'red', 'orange', 'green', 'blue']
    else:
        lims_o8_T = [30, 34, 41, 48, 50] # Temperatures  5., 5.4, 6.1, 6.8, 7 
        Tbins_o8 =   [np.sum(hist_o8['bins'][:,:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,:lims_o8_T[0],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[0]:lims_o8_T[1],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[1]:lims_o8_T[2],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[2]:lims_o8_T[3],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[3]:lims_o8_T[4],:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[4]:,:,:],axis=(0,1,3)),\
                  np.sum(hist_o8['bins'][:,lims_o8_T[0]:lims_o8_T[4],:,:],axis=(0,1,3))\
                 ]
        labels_T_o8 = ['total',\
                     r'$<%s$'%str(hist_o8['edges'][1][lims_o8_T[0]]),\
                     r'$%s$-$%s$'%(str(hist_o8['edges'][1][lims_o8_T[0]]), str(hist_o8['edges'][1][lims_o8_T[1]])),\
                     r'$%s$-$%s$'%(str(hist_o8['edges'][1][lims_o8_T[1]]), str(hist_o8['edges'][1][lims_o8_T[2]])),\
                     r'$%s$-$%s$'%(str(hist_o8['edges'][1][lims_o8_T[2]]), str(hist_o8['edges'][1][lims_o8_T[3]])),\
                     r'$%s$-$%s$'%(str(hist_o8['edges'][1][lims_o8_T[3]]), str(hist_o8['edges'][1][lims_o8_T[4]])),\
                     r'$>%s$'%(str(hist_o8['edges'][1][lims_o8_T[4]])),\
                     r'$%s$-$%s$'%(hist_o8['edges'][1][lims_o8_T[0]], str(hist_o8['edges'][1][lims_o8_T[4]])),\
                    ]
        colors_T_o8 = ['black', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'gray']
    
    
    fig = plt.figure(figsize=(6.,5.))
    grid = gsp.GridSpec(2,2,width_ratios=[5.,5.], height_ratios = [4.,4.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0,1]] for xi in [0,1]]) # in mainaxes: x = column, y = row
    
    
    if relative:
        binsedges_T_o7 = [[np.array(list(bins/(Tbins_o7[0])) + [0.]), edges_o7] for bins in Tbins_o7] #-> dN/dlogNdX
        ylim_o7 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.92,0.70)
        subtitleloc = (0.92,0.90)
    else:
        binsedges_T_o7 = [[hist_o7.npt*np.array(list(bins/np.diff(edges_o7)/dXtot) + [0.]), edges_o7] for bins in Tbins_o7] #-> dN/dlogNdX 
        binsedges_T_o7 = [[np.max(np.array([np.log10(be[0]), np.array((-100.,)*len(be[0])) ]), axis=0) , be[1]] for be in binsedges_T_o7] # log, with zero -> -100 to get nicer cutoffs in the plot
        ylim_o7 = (-6.5,2.5)
        ylabel = r'$\log_{10}\left( \partial^2 n \,/\, \partial \log_{10} N \, \partial X \right)$'
        subfigindloc = (0.12,0.05)
        subtitleloc = None
    
                  
    
    
    
    lnp.cddfsubplot1(axes[0,0], binsedges_T_o7,subtitle = 'O VII',subfigind = sfi_a,\
                     xlabel=r'$\log_{10} N_{\mathrm{O VII}} \, [\mathrm{cm}^{-2}]$', ylabel=ylabel,\
                     colors = colors_T_o7, labels = None, linestyles=None, fontsize =fontsize,\
                     xlim = (11.5,17.9),ylim=ylim_o7,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = False,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
 
    
    if relative:
        binsedges_T_o8 = [[np.array(list(bins/(Tbins_o8[0])) + [0.]), edges_o8] for bins in Tbins_o8] #-> dN/dlogNdX
        ylim_o8 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.15,0.70)
        subtitleloc = (0.22,0.90)
    else:
        binsedges_T_o8 = [[hist_o8.npt*np.array(list(bins/np.diff(edges_o8)/dXtot) + [0.]), edges_o8] for bins in Tbins_o8] #-> dN/dlogNdX 
        binsedges_T_o8 = [[np.max(np.array([np.log10(be[0]), np.array((-100.,)*len(be[0])) ]), axis=0) , be[1]] for be in binsedges_T_o8] # log, with zero -> -100 to get nicer cutoffs in the plot
        ylim_o8 = (-6.5,2.5)
        ylabel = r'$\log_{10}\left( \partial^2 n \,/\, \partial \log_{10} N \, \partial X \right)$'
        subfigindloc = (0.12,0.05)
        subtitleloc = None
    
    
    lnp.cddfsubplot1(axes[0,1], binsedges_T_o8,subtitle = 'O VIII',subfigind = sfi_b,\
                     xlabel=r'$\log_{10} N_{\mathrm{O VIII}} \, [\mathrm{cm}^{-2}]$',ylabel = None,\
                     colors = colors_T_o8, labels=None, linestyles=None, fontsize =fontsize,\
                     xlim = (11.5,17.9),ylim=ylim_o8,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = False, subfigindloc=subfigindloc, takeylog=False, steppost=True, subtitleloc=subtitleloc)
    
    
    legend_handles_T_o7 = [mlines.Line2D([], [], color=colors_T_o7[i], linestyle = 'solid', label=labels_T_o7[i]) for i in range(len(Tbins_o7))]
    legend_T_o7 = axes[1,0].legend(handles = legend_handles_T_o7, fontsize = fontsize, ncol = 2, loc = 'upper center', bbox_to_anchor=(0.5, 0.80), columnspacing = 0.5)
    legend_T_o7.set_title(r'$\log_{10} T \, [\mathrm{K}]$',  prop = {'size': fontsize})
    axes[1,0].axis('off')
     
    legend_handles_T_o8 = [mlines.Line2D([], [], color=colors_T_o8[i], linestyle = 'solid', label=labels_T_o8[i]) for i in range(len(Tbins_o8))]
    legend_T_o8 = axes[1,1].legend(handles = legend_handles_T_o8, fontsize = fontsize, ncol=2, loc = 'upper center', bbox_to_anchor=(0.5, 0.80), columnspacing = 0.5)
    legend_T_o8.set_title(r'$\log_{10} T \, [\mathrm{K}]$',  prop = {'size': fontsize})
    axes[1,1].axis('off')
    
    # turn the y axis labels off for realzies
    axes[0][1].yaxis.set_major_formatter(dummyformatter)
    axes[0][1].yaxis.set_minor_formatter(dummyformatter)

    if relative:
        send = '_relative'
    else:
        send = ''
    if slidemode:
        plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_T%s.png'%send, bbox_inches = 'tight', format = 'png', dpi=300)
    else:
        plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_T%s.pdf'%send, bbox_inches = 'tight', format = 'pdf')
    

def plotcddf_decomposed_by_rho_T_ionsource(relative = False):
    hist_o7 = ea4o7
    hist_o8 = ea4o8
    logrhob_av = logrhob_av_ea
    
    edges_o7 = hist_o7['edges'][2]
    edges_o8 = hist_o8['edges'][2]
    
    fontsize=12
    dXtot = mc.getdX(cosmopars_ea['z'],100.,cosmopars=cosmopars_ea)*32000**2
    
    

    
    ## o7 rho, T limits; more complicated 2d selections -> set by hand    
    #lims_o7_rho = [33, 47, 57, 72] # densities -30.4 (~rhoav snap 28), -29, -28, -26.5
    #lims_o7_T   = [30, 34, 45, 50] # Temperatures 5, 5.4, 6.5, 7
    # rho, T, N, Z
    total_o7 = np.sum(hist_o7['bins'][:,:,:,:],axis=(0,1,3))
    label_total_o7 = 'total'
    photo_o7 = np.sum(hist_o7['bins'][:47,:,:,:],axis=(0,1,3)) + np.sum(hist_o7['bins'][47:,:30,:,:],axis=(0,1,3))  # T < 10^5 K or rho < 10^-29 g/cm3 : definitely photoionised
    label_photo_o7 = r'$\delta < %s$ or $T < %s$'%('%.2f'%(hist_o7['edges'][0][47] -logrhob_av), str(hist_o7['edges'][1][30]))
    whim_o7 = np.sum(hist_o7['bins'][:,30:50,:,:],axis=(0,1,3))  #  10^5 K < T < 10^7 K : Cen Ostriker 1999 WHIM def.
    label_whim_o7 = r'$%s < T < %s$'%(str(hist_o7['edges'][1][30]), str(hist_o7['edges'][1][50]))
    collis_o7 = np.sum(hist_o7['bins'][57:,34:45,:,:],axis=(0,1,3))   # 10^5.4 K < T < 10^6.5 K, rho > 10^-28 g/cm3: collisional peak, collisionally ionised regime
    label_collis_o7 = r'$%s < T < %s$, $ %s < \delta$'%(str(hist_o7['edges'][1][34]), str(hist_o7['edges'][1][45]), '%.2f'%(hist_o7['edges'][0][57] -logrhob_av)) 
    trans_o7 = np.sum(hist_o7['bins'][47:57,30:,:,:],axis=(0,1,3))  # region between photo- and collisional ionisation:  10^-29 g/cm3 < rho < 10^-28 g/cm3 and T > 10^5 K 
    label_trans_o7 = r'$%s < T$, $%s < \delta < %s$'%(str(hist_o7['edges'][1][30]), '%.2f'%(hist_o7['edges'][0][47] -logrhob_av), '%.2f'%(hist_o7['edges'][0][57] -logrhob_av))
    collis_lo_o7 = np.sum(hist_o7['bins'][57:,30:34,:,:],axis=(0,1,3))  # 10^5 K < T < 10^5.4 K, rho > 10^-28 g/cm3: outside peak, collisionally ionised regime
    label_collis_lo_o7 = r'$%s < T < %s$, $\delta < %s$'%(str(hist_o7['edges'][1][30]), str(hist_o7['edges'][1][34]), '%.2f'%(hist_o7['edges'][0][57] -logrhob_av)) 
    collis_hi_o7 = np.sum(hist_o7['bins'][57:,45:,:,:],axis=(0,1,3)) # T > 10^6.5 K, rho > 10^-28 g/cm3: outside peak, collisionally ionised regime
    label_collis_hi_o7 = r'$%s < T$, $\delta < %s$'%(str(hist_o7['edges'][1][45]), '%.2f'%(hist_o7['edges'][0][57] -logrhob_av)) 

    total_check_o7 = photo_o7 + collis_o7 + collis_lo_o7 + collis_hi_o7 + trans_o7
    diff_o7 = total_check_o7-total_o7
    match_o7 = np.all(np.isfinite(total_o7) == np.isfinite(total_check_o7))
    print('Total check o7: %.2e (should be zero or fp difference), %s (should be true)'%(np.max(diff_o7[np.isfinite(diff_o7)]), match_o7))
    
    bins_o7 =   [total_o7,       photo_o7,       trans_o7,       collis_o7,       collis_lo_o7,       collis_hi_o7,       whim_o7]
    labels_o7 = [label_total_o7, label_photo_o7, label_trans_o7, label_collis_o7, label_collis_lo_o7, label_collis_hi_o7, label_whim_o7]
    
    ## o8 rho, T limits
    #lims_o8_rho = [21, 35, 45, 60] # densities -30.4 (~rhoav snap 28), -29, -28, -26.5
    #lims_o8_T = [30, 34, 41, 48, 50] # Temperatures  5., 5.4, 6.1, 6.8, 7 
    total_o8 = np.sum(hist_o8['bins'][:,:,:,:],axis=(0,1,3))
    label_total_o8 = 'total'
    photo_o8 = np.sum(hist_o8['bins'][:35,:,:,:],axis=(0,1,3)) + np.sum(hist_o8['bins'][35:,:34,:,:],axis=(0,1,3))  # T < 10^5 K or rho < 10^-29 g/cm3 : definitely photoionised
    label_photo_o8 = r'$\delta < %s$ or $T < %s$'%('%.2f'%(hist_o8['edges'][0][35] -logrhob_av), str(hist_o8['edges'][1][34]))
    whim_o8 = np.sum(hist_o8['bins'][:,30:50,:,:],axis=(0,1,3))  #  10^5 K < T < 10^7 K : Cen Ostriker 1999 WHIM def.
    label_whim_o8 = r'$%s < T < %s$'%(str(hist_o8['edges'][1][30]), str(hist_o8['edges'][1][50]))
    collis_o8 = np.sum(hist_o8['bins'][45:,41:48,:,:],axis=(0,1,3))   # 10^6.1 K < T < 10^6.8 K, rho > 10^-28 g/cm3: collisional peak, collisionally ionised regime
    label_collis_o8 = r'$%s < T < %s$, $%s < \delta$'%(str(hist_o8['edges'][1][41]), str(hist_o8['edges'][1][48]), '%.2f'%(hist_o8['edges'][0][45] -logrhob_av)) 
    trans_o8 = np.sum(hist_o8['bins'][35:45,34:,:,:],axis=(0,1,3))   # region between photo- and collisional ionisation:  10^-29 g/cm3 < rho < 10^-28 g/cm3 and T > 10^5.4 K
    label_trans_o8 = r'$%s < T$, $%s < \delta < %s$'%(str(hist_o8['edges'][1][34]),'%.2f'%(hist_o8['edges'][0][35] -logrhob_av), '%.2f'%(hist_o8['edges'][0][45] -logrhob_av))
    collis_lo_o8 = np.sum(hist_o8['bins'][45:,34:41,:,:],axis=(0,1,3))  # 10^5.4 K < T < 10^6.1 K,  rho > 10^-28 g/cm3: outside peak, collisionally ionised regime
    label_collis_lo_o8 = r'$%s < T < %s$, $%s < \delta$'%(str(hist_o8['edges'][1][34]), str(hist_o8['edges'][1][41]), '%.2f'%(hist_o8['edges'][0][45] -logrhob_av)) 
    collis_hi_o8 = np.sum(hist_o8['bins'][45:,48:,:,:],axis=(0,1,3)) # T > 10^6.8 K,  rho > 10^-28 g/cm3: outside peak, collisionally ionised regime
    label_collis_hi_o8 = r'$%s < T$, $%s < \delta$'%(str(hist_o8['edges'][1][48]), '%.2f'%(hist_o8['edges'][0][45] -logrhob_av)) 
    
    total_check_o8 = photo_o8 + collis_o8 + collis_lo_o8 + collis_hi_o8 + trans_o8
    diff_o8 = total_check_o8-total_o8
    match_o8 = np.all(np.isfinite(total_o8) == np.isfinite(total_check_o8))
    print('Total check o8: %i (should be zero), %s (should be true)'%(np.max(diff_o8[np.isfinite(diff_o8)]), match_o8))
    
    bins_o8 =   [total_o8,       photo_o8,       trans_o8,       collis_o8,       collis_lo_o8,       collis_hi_o8,       whim_o8]
    labels_o8 = [label_total_o8, label_photo_o8, label_trans_o8, label_collis_o8, label_collis_lo_o8, label_collis_hi_o8, label_whim_o8]
    
    
    fontsize=12
    dXtot = mc.getdX(cosmopars_ea['z'],100.,cosmopars=cosmopars_ea)*32000**2
    
    fig = plt.figure(figsize=(6.,5.5))
    grid = gsp.GridSpec(2,2,width_ratios=[5.,5.], height_ratios = [5.,6.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0,1]] for xi in [0,1]]) # in mainaxes: x = column, y = row
    
    
    if relative:
        binsedges_o7 = [[np.array(list(bins/(bins_o7[0])) + [0.]), edges_o7] for bins in bins_o7] #-> dN/dlogNdX
        ylim_o7 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.15,0.75)
        subtitleloc = (0.22,0.95)
    else:
        binsedges_o7 = [[hist_o7.npt*np.array(list(bins/np.diff(edges_o7)/dXtot) + [0.]), edges_o7] for bins in bins_o7] #-> dN/dlogNdX 
        ylim_o7 = None
        ylabel = r'$\mathrm{d} n \,/\, \mathrm{d} \log_{10} N \, \mathrm{d} X$'
        subfigindloc = (0.95,0.75)
        subtitleloc = None
    
                                      
    colors_o7 = ['black', 'red', 'gold', 'green', 'orange', 'purple', 'gray']
    
    lnp.cddfsubplot1(axes[0,0], binsedges_o7,subtitle = 'O VII',subfigind = '(a)',\
                     xlabel=r'$\log_{10} N_{\mathrm{O VII}} \, [\mathrm{cm}^{-2}]$', ylabel = ylabel,\
                     colors = colors_o7, labels = None, linestyles=None, fontsize =fontsize,\
                     xlim = (9.,18.),ylim=ylim_o7,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = True,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
 
    if relative:
        binsedges_o8 = [[np.array(list(bins/(bins_o8[0])) + [0.]), edges_o8] for bins in bins_o8] #-> dN/dlogNdX
        ylim_o8 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.15,0.75)
        subtitleloc = (0.22,0.95)
    else:
        binsedges_o8 = [[hist_o8.npt*np.array(list(bins/np.diff(edges_o8)/dXtot) + [0.]), edges_o8] for bins in bins_o8] #-> dN/dlogNdX 
        ylim_o8 = None
        ylabel = r'$\mathrm{d} n \,/\, \mathrm{d} \log_{10} N \, \mathrm{d} X$'
        subfigindloc = (0.95,0.75)
        subtitleloc = None
    
                 

    colors_o8 = ['black', 'red', 'gold', 'green', 'orange', 'purple', 'gray']
    
    lnp.cddfsubplot1(axes[0,1], binsedges_o8,subtitle = 'O VIII',subfigind = '(b)',\
                     xlabel=r'$\log_{10} N_{\mathrm{O VIII}} \, [\mathrm{cm}^{-2}]$',ylabel = None,\
                     colors = colors_o8, labels = None, linestyles=None, fontsize =fontsize,\
                     xlim = (9.,18.),ylim=ylim_o8,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = True,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
    
    
    legend_handles_o7 = [mlines.Line2D([], [], color=colors_o7[i], linestyle = 'solid', label=labels_o7[i]) for i in range(len(bins_o7))]
    legend_o7 = axes[1,0].legend(handles = legend_handles_o7, fontsize = fontsize, ncol = 1, loc = 'upper center', bbox_to_anchor=(0.5, 0.80))
    legend_o7.set_title(r'$\log_{10} T \, [\mathrm{K}]$, $\log_{10} \delta$',  prop = {'size': fontsize})
    axes[1,0].axis('off')
     
    legend_handles_o8 = [mlines.Line2D([], [], color=colors_o8[i], linestyle = 'solid', label=labels_o8[i]) for i in range(len(bins_o8))]
    legend_o8 = axes[1,1].legend(handles = legend_handles_o8, fontsize = fontsize, ncol=1, loc = 'upper center', bbox_to_anchor=(0.5, 0.80))
    legend_o8.set_title(r'$\log_{10} T \, [\mathrm{K}]$, $\log_{10} \delta$',  prop = {'size': fontsize})
    axes[1,1].axis('off')
    
    # turn the y axis labels off for realzies
    axes[0][1].yaxis.set_major_formatter(dummyformatter)
    axes[0][1].yaxis.set_minor_formatter(dummyformatter)

    if relative:
        send = '_relative'
    else:
        send = ''
        
    plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_rho_T%s.pdf'%send, bbox_inches = 'tight', format = 'pdf')
    


    
def fmt(selection, values, fmts, axnames):
    '''
    list of tuples: slice(*selection[i]) should be a valid 1d selection
    does not factor in strides =/= 1
    '''
    
    fmt_lo = r'$ %s < %s $'
    fmt_c  = r'$ %s < %s < %s $'
    fmt_hi = r'$ %s < %s $'
    
    fmts_lo = [fmt_lo%(axnames[ind], fmts[ind]) for ind in range(len(selection)) ]
    fmts_c =  [fmt_c%(fmts[ind], axnames[ind], fmts[ind]) for ind in range(len(selection)) ]
    fmts_hi = [fmt_hi%(fmts[ind], axnames[ind]) for ind in range(len(selection)) ]

    realsels = np.array([sel != (None,)*len(sel) for sel in selection])
    selinds = np.where(realsels)[0]
    #print(selinds)
    #for ind in selinds:
    #    print(values[ind][slice(*selection[ind])])

    
    fmtstring=[\
             fmts_lo[ind]%(values[ind][selection[ind][1]]) if selection[ind][0] is None else\
             fmts_hi[ind]%(values[ind][selection[ind][0]]) if selection[ind][1] is None else\
             fmts_c[ind]%(values[ind][selection[ind][0]], values[ind][selection[ind][1]])\
             for ind in selinds\
            ]    

    return ', '.join(fmtstring)
    
def plotcddf_decomposed_by_rho_T_breakfinder(relative = False):
    hist_o7 = ea4o7
    hist_o8 = ea4o8
    logrhob_av = logrhob_av_ea
    
    fmts_dim  = ['%.2f', '%.1f', '%s', '%s']
    names_dim = ['\delta', 'T', 'N', 'Z']
    
    edges_o7 = hist_o7['edges'][2]
    edges_o8 = hist_o8['edges'][2]
    
    fontsize=12
    dXtot = mc.getdX(cosmopars_ea['z'],100.,cosmopars=cosmopars_ea)*32000**2
    
    
    ## o7 rho, T limits; more complicated 2d selections -> set by hand    
    #lims_o7_rho = [33, 47, 57, 72] # densities -30.4 (~rhoav snap 28), -29, -28, -26.5
    #lims_o7_T   = [30, 34, 45, 50] # Temperatures 5, 5.4, 6.5, 7
    # rho, T, N, Z
    lims_o7_T = [34, 45]
    lims_o7_rho = [33, 43, 53, 56, 63]
    
    slices_o7_T = [(None, lims_o7_T[ind]) if ind == 0 else\
                   (lims_o7_T[ind -1], None) if ind == len(lims_o7_T) else\
                   (lims_o7_T[ind -1], lims_o7_T[ind])\
                   for ind in range(len(lims_o7_T) + 1)]

    slices_o7_rho = [(None, lims_o7_rho[ind]) if ind == 0 else\
                   (lims_o7_rho[ind -1], None) if ind == len(lims_o7_rho) else\
                   (lims_o7_rho[ind -1], lims_o7_rho[ind])\
                   for ind in range(len(lims_o7_rho) + 1)]
    
    slices_o7 = [(srho, sT, (None,), (None,)) for sT in slices_o7_T for srho in slices_o7_rho] # rho is the fast index
    
    ls_base_o7 = np.array(['dotted', 'solid', 'dashed', 'dotdash'])[:len(slices_o7_T)]
    linestyles_o7 = [style for style in ls_base_o7 for dummy in range(len(slices_o7_rho))]
    cs_base_o7 = np.array(['red', 'orange', 'gold', 'green', 'blue', 'purple', 'fuchsia', 'brown'])[:len(slices_o7_rho)]
    colors_o7 = [color for dummy in range(len(slices_o7_T)) for color in cs_base_o7]
    legsel_o7 = list(np.arange(len(slices_o7_rho))) + list( (1 + np.arange(len(slices_o7_T)-1))*len(slices_o7_rho)) #which to include in the legend: all fast index values at slow 0, then all slow index values >0 at fast 0
    #print(legsel_o7)
    
    
    bins_o7 = [np.sum(hist_o7['bins'][tuple([slice(*sub) for sub in sel])], axis = (0,1,3)) for sel in slices_o7]
    labels_o7 = [fmt(sel, [hist_o7['edges'][0]  - logrhob_av, hist_o7['edges'][1], hist_o7['edges'][2], hist_o7['edges'][3]], fmts_dim, names_dim) for sel in slices_o7]    
    
    
    total_o7 = np.sum(hist_o7['bins'][:,:,:,:],axis=(0,1,3))
    total_check_o7 = np.sum(np.array(bins_o7), axis=0)
    diff_o7 = total_check_o7-total_o7
    match_o7 = np.all(np.isfinite(total_o7) == np.isfinite(total_check_o7))
    print('Total check o7: %.2e (should be zero or fp difference), %s (should be true)'%(np.max(diff_o7[np.isfinite(diff_o7)]), match_o7))
    
    bins_o7 = [total_o7] + bins_o7
    labels_o7 = ['total'] + labels_o7
    colors_o7 = ['black'] + colors_o7
    linestyles_o7 = ['solid'] + linestyles_o7
    legsel_o7 = [0] + list(np.array(legsel_o7)+1)
    
    #print(len(bins_o7))
    #print(labels_o7)
    #print(colors_o7)
    #print(linestyles_o7)


    ## o8 rho, T limits
    #lims_o8_rho = [21, 35, 45, 60] # densities -30.4 (~rhoav snap 28), -29, -28, -26.5
    #lims_o8_T = [30, 34, 41, 48, 50] # Temperatures  5., 5.4, 6.1, 6.8, 7 
    lims_o8_T = [41, 48]
    lims_o8_rho = [21, 31, 41, 44, 51]
    
    slices_o8_T = [(None, lims_o8_T[ind]) if ind == 0 else\
                   (lims_o8_T[ind -1], None) if ind == len(lims_o8_T) else\
                   (lims_o8_T[ind -1], lims_o8_T[ind])\
                   for ind in range(len(lims_o8_T) + 1)]

    slices_o8_rho = [(None, lims_o8_rho[ind]) if ind == 0 else\
                   (lims_o8_rho[ind -1], None) if ind == len(lims_o8_rho) else\
                   (lims_o8_rho[ind -1], lims_o8_rho[ind])\
                   for ind in range(len(lims_o8_rho) + 1)]
    
    slices_o8 = [(srho, sT, (None,), (None,)) for sT in slices_o8_T for srho in slices_o8_rho] # T is the fast index
    #print(slices_o8)
    
    ls_base_o8 = np.array(['dotted', 'solid', 'dashed', 'dotdash'])[:len(slices_o8_T)]
    linestyles_o8 = [style for style in ls_base_o8 for dummy in range(len(slices_o8_rho))]
    cs_base_o8 = np.array(['red', 'orange', 'gold', 'green', 'blue', 'purple', 'fuchsia', 'brown'])[:len(slices_o8_rho)]
    colors_o8 = [color for dummy in range(len(slices_o8_T)) for color in cs_base_o8]
    legsel_o8 = list(np.arange(len(slices_o8_rho))) + list( (1 + np.arange(len(slices_o8_T)-1))*len(slices_o8_rho)) #which to include in the legend: all fast index values at slow 0, then all slow index values >0 at fast 0
    
    bins_o8 = [np.sum(hist_o8['bins'][tuple([slice(*sub) for sub in sel])], axis = (0,1,3)) for sel in slices_o8]
    labels_o8 = [fmt(sel, [hist_o8['edges'][0]  - logrhob_av, hist_o8['edges'][1], hist_o8['edges'][2], hist_o8['edges'][3]], fmts_dim, names_dim) for sel in slices_o8]    
    
    
    total_o8 = np.sum(hist_o8['bins'][:,:,:,:],axis=(0,1,3))
    total_check_o8 = np.sum(np.array(bins_o8), axis=0)
    diff_o8 = total_check_o8-total_o8
    match_o8 = np.all(np.isfinite(total_o8) == np.isfinite(total_check_o8))
    print('Total check o8: %.2e (should be zero or fp difference), %s (should be true)'%(np.max(diff_o8[np.isfinite(diff_o8)]), match_o8))
    
    bins_o8 = [total_o8] + bins_o8
    labels_o8 = ['total'] + labels_o8
    colors_o8 = ['black'] + colors_o8
    linestyles_o8 = ['solid'] + linestyles_o8
    legsel_o8 = [0] + list(np.array(legsel_o8)+1)
    
    
    fontsize=12
    dXtot = mc.getdX(cosmopars_ea['z'],100.,cosmopars=cosmopars_ea)*32000**2
    
    fig = plt.figure(figsize=(6.,5.5))
    grid = gsp.GridSpec(2,2,width_ratios=[5.,5.], height_ratios = [5.,6.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0,1]] for xi in [0,1]]) # in mainaxes: x = column, y = row
    
    
    if relative:
        binsedges_o7 = [[np.array(list(bins/(bins_o7[0])) + [0.]), edges_o7] for bins in bins_o7] #-> dN/dlogNdX
        ylim_o7 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.15,0.75)
        subtitleloc = (0.22,0.95)
    else:
        binsedges_o7 = [[hist_o7.npt*np.array(list(bins/np.diff(edges_o7)/dXtot) + [0.]), edges_o7] for bins in bins_o7] #-> dN/dlogNdX 
        ylim_o7 = None
        ylabel = r'$\mathrm{d} n \,/\, \mathrm{d} \log_{10} N \, \mathrm{d} X$'
        subfigindloc = (0.95,0.75)
        subtitleloc = None
    
    
    lnp.cddfsubplot1(axes[0,0], binsedges_o7,subtitle = 'O VII',subfigind = '(a)',\
                     xlabel=r'$\log_{10} N_{\mathrm{O VII}} \, [\mathrm{cm}^{-2}]$', ylabel = ylabel,\
                     colors = colors_o7, labels = None, linestyles=linestyles_o7, fontsize =fontsize,\
                     xlim = (9.,18.),ylim=ylim_o7,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = True,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
 
    if relative:
        binsedges_o8 = [[np.array(list(bins/(bins_o8[0])) + [0.]), edges_o8] for bins in bins_o8] #-> dN/dlogNdX
        ylim_o8 = (1.e-2, 1.05)
        ylabel = 'fraction of absorbers'
        subfigindloc = (0.15,0.75)
        subtitleloc = (0.22,0.95)
    else:
        binsedges_o8 = [[hist_o8.npt*np.array(list(bins/np.diff(edges_o8)/dXtot) + [0.]), edges_o8] for bins in bins_o8] #-> dN/dlogNdX 
        ylim_o8 = None
        ylabel = r'$\mathrm{d} n \,/\, \mathrm{d} \log_{10} N \, \mathrm{d} X$'
        subfigindloc = (0.95,0.75)
        subtitleloc = None
    
                 
    lnp.cddfsubplot1(axes[0,1], binsedges_o8,subtitle = 'O VIII',subfigind = '(b)',\
                     xlabel=r'$\log_{10} N_{\mathrm{O VIII}} \, [\mathrm{cm}^{-2}]$',ylabel = None,\
                     colors = colors_o8, labels = None, linestyles=linestyles_o8, fontsize =fontsize,\
                     xlim = (9.,18.),ylim=ylim_o8,xticklabels=False,yticklabels=True,\
                     legendloc = None,legend_ncol=None,\
                     ylog = True,subfigindloc=subfigindloc,takeylog=False, steppost = True, subtitleloc = subtitleloc)
    
    
    legend_handles_o7 = [mlines.Line2D([], [], color=colors_o7[i], linestyle = linestyles_o7[i], label=labels_o7[i]) for i in legsel_o7]
    legend_o7 = axes[1,0].legend(handles = legend_handles_o7, fontsize = fontsize, ncol = 1, loc = 'upper center', bbox_to_anchor=(0.5, 0.80))
    legend_o7.set_title(r'$\log_{10} T \, [\mathrm{K}]$, $\log_{10} \delta$',  prop = {'size': fontsize})
    axes[1,0].axis('off')
     
    legend_handles_o8 = [mlines.Line2D([], [], color=colors_o8[i], linestyle = linestyles_o7[i], label=labels_o8[i]) for i in legsel_o8]
    legend_o8 = axes[1,1].legend(handles = legend_handles_o8, fontsize = fontsize, ncol=1, loc = 'upper center', bbox_to_anchor=(0.5, 0.80))
    legend_o8.set_title(r'$\log_{10} T \, [\mathrm{K}]$, $\log_{10} \delta$',  prop = {'size': fontsize})
    axes[1,1].axis('off')
    
    # turn the y axis labels off for realzies
    axes[0][1].yaxis.set_major_formatter(dummyformatter)
    axes[0][1].yaxis.set_minor_formatter(dummyformatter)

    if relative:
        send = '_relative'
    else:
        send = ''
        
    plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_rho_T_blocks%s.pdf'%send, bbox_inches = 'tight', format = 'pdf')
    


def plotcddf_AGNeffect_decomposed(kind='T'):
    
    hist_o7ref = ea4o7_ref50
    hist_o8ref = ea4o8_ref50 
    hist_o7noagn = ea4o7_noagn50
    hist_o8noagn = ea4o8_noagn50
    
    hists = {'o7ref' :  hist_o7ref,\
             'o7noagn': hist_o7noagn,\
             'o8ref':   hist_o8ref,\
             'o8noagn': hist_o8noagn}
    
    handleinfedges(hist_o7ref, setmin=-100., setmax=100.)
    handleinfedges(hist_o8ref, setmin=-100., setmax=100.)
    handleinfedges(hist_o7noagn, setmin=-100., setmax=100.)
    handleinfedges(hist_o8noagn, setmin=-100., setmax=100.)
    
    nax = 0
    dax = 1
    tax = 2
    zax = 3
    
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'magenta']
    colors2 = ['lightsalmon', 'tan', 'limegreen', 'cyan', 'mediumorchid', 'pink']
    
    if kind == 'T':
        splitax = tax
        sumaxes = (dax, zax)
        splitedges = [5.0, 5.5, 6., 6.5, 7.]
    elif kind == 'delta':
        splitax = dax
        sumaxes = (tax, zax)
        splitedges = [0.0, 1.0, 2.0, 3.0, 4.0]
    elif kind == 'Z':
        splitax = zax
        sumaxes = (dax, tax)
        splitedges = [-1.5, -1.0, -0.5, 0.0, 0.5]
        
    legtitles = {dax: r'$\log_{10}\,\delta$',\
                 tax: r'$\log_{10}\, \mathrm{T} \; [\mathrm{K}]$',\
                 zax: r'$[\mathrm{O} \, / \, \mathrm{H}]$'}
    o7label = r'$\log_{10} \, N_{\mathrm{O\, VII}}  \; [\mathrm{cm}^{-2}]$'
    o8label = r'$\log_{10} \, N_{\mathrm{O\, VIII}} \; [\mathrm{cm}^{-2}]$'
    ylabel  = r'$\log_{10} \left( \partial^2 n \, / \, \partial \log_{10} N \, \partial X \right)$'
    ylim = (-6.5,2.5)
    subfigindloc = (0.95, 0.87)
    
    convs = {nax: 1., dax: 10.**logrhob_av_ea, tax: 1., zax: ol.solar_abunds['oxygen']}

    #dimlabels = (labels[0], labels[1], labels[2], labels[3])
    dimshifts = (-1.*np.log10(convs[i]) for i in range(4))
    edgeinds = {key: [np.argmin(np.abs(splitedges[i] - hists[key]['edges'][splitax] + np.log10(convs[splitax]))) for i in range(len(splitedges))] for key in hists.keys()}
    mins = {key: [None] + edgeinds[key] for key in edgeinds.keys()}
    maxs = {key: edgeinds[key] + [None] for key in edgeinds.keys()}

    
    fontsize=12
    sfi_a = ''
    sfi_b = ''
    dXtot = mc.getdX(0.0, 50., cosmopars=cosmopars_ea) * 16000**2
    
    fig = plt.figure(figsize=(5.5,4.2))
    grid = gsp.GridSpec(2, 2, width_ratios=[5.,5.], height_ratios=[4.,3.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0,1]] for xi in [0]]) # in mainaxes: x = column, y = row
    lax = fig.add_subplot(grid[1, 0:2]) 
    
    #fig, axes = plt.subplots(ncols=2, nrows=4, sharex = True, sharey=True,)
        
    saxes = list(sumaxes) + [splitax]
    saxes.sort()
    saxes = tuple(saxes)
    blankslices = [slice(None, None, None)] * 4
    sels = {key: [tuple(blankslices[:splitax] + [slice(mins[key][i], maxs[key][i], None)] + blankslices[splitax + 1:])\
                  for i in range(len(mins[key]))]
            for key in mins.keys()}

    bins = {key: [np.sum(hists[key]['bins'][:,:,:,:], axis=saxes)] + \
                 [np.sum(hists[key]['bins'][sels[key][i]], axis=saxes) \
                  for i in range(len(sels[key]))] \
            for key in hists.keys()}

    splitedges = {key: hists[key]['edges'][splitax] - np.log10(convs[splitax]) for key in hists.keys()}
    leglabels = {key: [r'$%.1f \endash %.1f$'%(splitedges[key][mn], splitedges[key][mx]) if (mn is not None and mx is not None) else\
                       r'$ < %.1f$'%(splitedges[key][mx]) if mx is not None else\
                       r'$ > %.1f$'%(splitedges[key][mn]) if mn is not None else\
                       'total'\
                       for (mn, mx) in zip(mins[key], maxs[key])] \
                       for key in mins.keys()}

    binsedges = {key: [[hists[key].npt * np.array(list(bn / np.diff(hists[key]['edges'][nax]) / dXtot) + [0.]), hists[key]['edges'][nax]] for bn in bins[key]] for key in hists.keys()} #-> dN/dlogNdX 
    binsedges = {key: [[np.max(np.array([np.log10(be[0]), np.array((-100.,)*len(be[0])) ]), axis=0) , be[1]] for be in binsedges[key]] for key in binsedges.keys()} # log, with zero -> -100 to get nicer cutoffs in the plot
    

    colors_ = ['black'] + colors[:len(mins['o7ref']) + 1]
    colors_ = colors_ + ['gray'] + colors2[:len(mins['o7ref']) + 1]
    linestyles = ['solid'] * (len(mins['o7ref']) + 1) + ['solid'] * (len(mins['o7ref']) + 1)
    
    lnp.cddfsubplot1(axes[0,0], binsedges['o7ref'] + binsedges['o7noagn'], subtitle='O VII',subfigind=sfi_a,\
                     xlabel=o7label, ylabel=ylabel,\
                     colors=colors_, labels=['total, Ref'] + leglabels['o7ref'] + ['total, NoAGN'] + leglabels['o7noagn'], linestyles=linestyles, fontsize=fontsize,\
                     xlim=(11.5,17.9), ylim=ylim, xticklabels=True, yticklabels=True,\
                     legendloc = None, legend_ncol=None, dolegend=False,\
                     ylog = False, subfigindloc=subfigindloc, takeylog=False, steppost=True, subtitleloc=None)
    
    lnp.cddfsubplot1(axes[0,1], binsedges['o8ref']+ binsedges['o8noagn'], subtitle='O VIII',subfigind=sfi_b,\
                     xlabel=o8label, ylabel=None,\
                     colors=colors_, labels=['total, Ref'] + leglabels['o8ref'] + ['total, NoAGN'] + leglabels['o8noagn'], linestyles=linestyles, fontsize=fontsize,\
                     xlim=(11.5,17.9), ylim=ylim, xticklabels=True, yticklabels=False,\
                     legendloc=None, legend_ncol=None, legend_title=None, dolegend=False, \
                     ylog=False, subfigindloc=subfigindloc,takeylog=False, steppost=True, subtitleloc=None)
    
    #legend_handles_rho_o8 = [mlines.Line2D([], [], color=colors_rho_o8[i], linestyle = 'solid', label=labels_rho_o8[i]) for i in range(len(rhobins_o8))]
    #legend_rho_o8 = axes[1,1].legend(handles = legend_handles_rho_o8, fontsize = fontsize, ncol =2)
    #legend_rho_o8.set_title(r'$\log_{10} \rho \, [\mathrm{g}\, \mathrm{cm}^{-3}]$',  prop = {'size': fontsize})
    
    #legend_handles = [mlines.Line2D([], [], color=colors[i], linestyle='solid', label=leglabels['o7ref'][i]) for i in range(len(leglabels['o7ref']))]
    #legend_handles = [mlines.Line2D([], [], color='black', linestyle='solid', label='total') for i in range(len(leglabels['o7ref']))] + legend_handles 
    #legend_handles2 = [mlines.Line2D([], [], color='gray', linestyle=ls, label=lb) for (ls, lb) in zip(['solid', 'dotted'], ['Ref', 'NoAGN'])]
    lh = axes[0][0].get_legend_handles_labels()
    legend_rho_o7 = lax.legend(handles=lh[0], fontsize = fontsize, ncol=2, loc = 'upper center', bbox_to_anchor=(0.5, 0.75))
    legend_rho_o7.set_title(legtitles[splitax],  prop = {'size': fontsize})
    lax.axis('off')

    # turn the y axis labels off for realzies
    axes[0][1].yaxis.set_major_formatter(dummyformatter)
    axes[0][1].yaxis.set_minor_formatter(dummyformatter)
    
    plt.savefig(mdir+ 'cddfs_coldens_o7-o8_L0050N0752Ref-NoAGN_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_by_%s.pdf'%(kind), bbox_inches='tight', format='pdf')
    
    
def plotgasproperties_AGNeffect(kind='nH-T'):
    
    hist_o7ref = ea4o7_ref50
    hist_o8ref = ea4o8_ref50 
    hist_o7noagn = ea4o7_noagn50
    hist_o8noagn = ea4o8_noagn50
    
    hists = {'o7ref' :  hist_o7ref,\
             'o7noagn': hist_o7noagn,\
             'o8ref':   hist_o8ref,\
             'o8noagn': hist_o8noagn}
    
    handleinfedges(hist_o7ref, setmin=-100., setmax=100.)
    handleinfedges(hist_o8ref, setmin=-100., setmax=100.)
    handleinfedges(hist_o7noagn, setmin=-100., setmax=100.)
    handleinfedges(hist_o8noagn, setmin=-100., setmax=100.)
    
    nax = 0
    dax = 1
    tax = 2
    zax = 3
    
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'deeppink']
    colors2 = ['lightsalmon', 'tan', 'limegreen', 'cyan', 'mediumorchid', 'hotpink']
    
    if kind == 'nH-T':
        plotaxes = (dax, tax)
        sumaxes = (nax, zax)
    elif kind == 'nH-Z':
        plotaxes = (dax, zax)
        sumaxes = (nax, tax)
    elif kind == 'T-Z':
        plotaxes = (tax, zax)
        sumaxes = (nax, dax)
    splitax = nax
    splitedges = [12.5, 13.5, 14.5, 15.5, 16.5]
    legtitles = {nax: r'$\log_{10}\, N \; [\mathrm{cm}^{-2}]$',\
                 dax: r'$\log_{10}\, n_{\mathrm{H}, \mathrm{ion}} $',\
                 tax: r'$\log_{10}\, \mathrm{T}_{\mathrm{ion}} \; [\mathrm{K}]$',\
                 zax: r'$[\mathrm{O}_{\mathrm{ion}} \, / \, \mathrm{H}]$'}
    
    lims = {nax: (12.5, 17.9), dax: (-8.5, 0.5), tax: (3., 8.), zax: (-4., 1.)}
    
    convs = {nax: 1., dax: rho_to_nh, tax: 1., zax: 1./ol.solar_abunds['oxygen']}

    #dimlabels = (labels[0], labels[1], labels[2], labels[3])
    dimshifts = (-1.*np.log10(convs[i]) for i in range(4))
    edgeinds = {key: [np.argmin(np.abs(splitedges[i] - hists[key]['edges'][splitax] + np.log10(convs[splitax]))) for i in range(len(splitedges))] for key in hists.keys()}
    mins = {key: [None] + edgeinds[key] for key in edgeinds.keys()}
    maxs = {key: edgeinds[key] + [None] for key in edgeinds.keys()}

    
    fontsize=12
    sfi_a = ''
    sfi_b = ''
    dXtot = mc.getdX(0.0, 50., cosmopars=cosmopars_ea) * 16000**2
    
    fig = plt.figure(figsize=(7.5, 9.0))
    grid = gsp.GridSpec(3, 2, width_ratios=[5., 5.], height_ratios=[4., 4., 3.], wspace=0.0,hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0,1]] for xi in [0, 1]]) # in mainaxes: x = column, y = row
    lax = fig.add_subplot(grid[2, 0:2]) 
    
    #fig, axes = plt.subplots(ncols=2, nrows=4, sharex = True, sharey=True,)
        
    saxes = list(sumaxes) 
    saxes.sort()
    saxes = tuple(saxes)
    blankslices = [slice(None, None, None)] * 4
    sels = {key: [tuple(blankslices[:splitax] + [slice(mins[key][i], maxs[key][i], None)] + blankslices[splitax + 1:])\
                  for i in range(len(mins[key]))]
            for key in mins.keys()}

    bins = {key: [np.sum(hists[key]['bins'][:,:,:,:], axis=saxes)] + \
                 [np.sum(hists[key]['bins'][sels[key][i]], axis=saxes) \
                  for i in range(len(sels[key]))] \
            for key in hists.keys()}

    splitedges = {key: hists[key]['edges'][splitax] - np.log10(convs[splitax]) for key in hists.keys()}
    leglabels = {key: [r'$%.1f \endash %.1f$'%(splitedges[key][mn], splitedges[key][mx]) if (mn is not None and mx is not None) else\
                       r'$ < %.1f$'%(splitedges[key][mx]) if mx is not None else\
                       r'$ > %.1f$'%(splitedges[key][mn]) if mn is not None else\
                       'total'\
                       for (mn, mx) in zip(mins[key], maxs[key])] \
                       for key in mins.keys()}

    if plotaxes[0] > plotaxes[1]:
        ax1 = plotaxes[0]
        ax0 = plotaxes[1]
        toplotaxes = (1, 0)
    else:
        ax1 = plotaxes[1]
        ax0 = plotaxes[0]
        toplotaxes = (0, 1)
    #binsedges = {key: [[bn / (np.diff(hists[key]['edges'][ax0][:, np.newaxis]) * np.diff(hists[key]['edges'][ax1][np.newaxis, :])) for bn in bins[key]] for key in hists.keys()}
    
    colors_ = ['black'] + colors[:len(mins['o7ref']) + 1]
    colors2_ = ['gray'] + colors2[:len(mins['o7ref']) + 1]
    linestyles_tot = ['dotted', 'dashed', 'dashdot', 'solid']
    levels_tot     = [0.9999, 0.99, 0.9, 0.5]
    linestyles_sub = ['dashed', 'solid']
    levels_sub     = [0.99, 0.5]
    
    for axi in range(4):
        yi = axi // 2
        xi = axi % 2
        ax = axes[yi, xi]
        if xi == 0:
            ion = 'o7'
            doylabel = True
        elif xi == 1:
            ion = 'o8'
            doylabel = False
        if yi == 0:
            doxlabel = False
            
            add_2dhist_contours_simple(ax, bins['%sref'%ion][0], [hists['%sref'%ion]['edges'][ax0], hists['%sref'%ion]['edges'][ax1]], toplotaxes=toplotaxes,\
                        fraclevels=True, levels=levels_tot, legendlabel='total, Ref',\
                        shiftx=np.log10(convs[plotaxes[0]]), shifty=np.log10(convs[plotaxes[1]]),
                        colors=[colors_[0]] * len(levels_tot), linestyles=linestyles_tot)
            add_2dhist_contours_simple(ax, bins['%snoagn'%ion][0], [hists['%snoagn'%ion]['edges'][ax0], hists['%snoagn'%ion]['edges'][ax1]], toplotaxes=toplotaxes,\
                        fraclevels=True, levels=levels_tot, legendlabel='total, NoAGN',\
                        shiftx=np.log10(convs[plotaxes[0]]), shifty=np.log10(convs[plotaxes[1]]),
                        colors=[colors2_[0]] * len(levels_tot), linestyles=linestyles_tot)
        elif yi == 1:
            doxlabel=True
            
            for i in range(len(sels['o7ref'])):
                add_2dhist_contours_simple(ax, bins['%sref'%ion][i + 1], [hists['%sref'%ion]['edges'][ax0], hists['%sref'%ion]['edges'][ax1]], toplotaxes=toplotaxes,\
                        fraclevels=True, levels=levels_sub, legendlabel=leglabels['%sref'%ion][i],\
                        shiftx=np.log10(convs[plotaxes[0]]), shifty=np.log10(convs[plotaxes[1]]),
                        colors=[colors_[i + 1]] * len(levels_sub), linestyles=linestyles_sub)
                add_2dhist_contours_simple(ax, bins['%snoagn'%ion][i + 1], [hists['%snoagn'%ion]['edges'][ax0], hists['%snoagn'%ion]['edges'][ax1]], toplotaxes=toplotaxes,\
                        fraclevels=True, levels=levels_sub, legendlabel=leglabels['%snoagn'%ion][i],\
                        shiftx=np.log10(convs[plotaxes[0]]), shifty=np.log10(convs[plotaxes[1]]),
                        colors=[colors2_[i + 1]] * len(levels_sub), linestyles=linestyles_sub)
        
        setticks(ax, fontsize, color='black', labelbottom=doxlabel, top=True, labelleft=doylabel, right=True, labeltop=False)
        ax.set_xlim(*lims[plotaxes[0]])
        ax.set_ylim(*lims[plotaxes[1]])
        if doylabel:
            ax.set_ylabel(legtitles[plotaxes[1]], fontsize=fontsize)
        if doxlabel:
            ax.set_xlabel(legtitles[plotaxes[0]], fontsize=fontsize)
    #legend_handles_rho_o8 = [mlines.Line2D([], [], color=colors_rho_o8[i], linestyle = 'solid', label=labels_rho_o8[i]) for i in range(len(rhobins_o8))]
    #legend_rho_o8 = axes[1,1].legend(handles = legend_handles_rho_o8, fontsize = fontsize, ncol =2)
    #legend_rho_o8.set_title(r'$\log_{10} \rho \, [\mathrm{g}\, \mathrm{cm}^{-3}]$',  prop = {'size': fontsize})
    
    legend_handles = [mlines.Line2D([], [], color=colors[i], linestyle='solid', label=leglabels['o7ref'][i]) for i in range(len(leglabels['o7ref']))]
    legend_handles_a = [mlines.Line2D([], [], color='black', linestyle='solid', label='total, Ref')]
    legend_handles_2 = [mlines.Line2D([], [], color=colors2[i], linestyle='solid', label=leglabels['o7noagn'][i]) for i in range(len(leglabels['o7noagn']))]
    legend_handles_2a = [mlines.Line2D([], [], color='gray', linestyle='solid', label='total, NoAGN')]
    legend_handles_styles = [mlines.Line2D([], [], color='brown', linestyle=linestyles_tot[i], label='%.2f %%'%(100. * levels_tot[i])) for i in range(len(levels_tot))]
    #legend_handles = [mlines.Line2D([], [], color='black', linestyle='solid', label='total') for i in range(len(leglabels['o7ref']))] + legend_handles 
    #legend_handles2 = [mlines.Line2D([], [], color='gray', linestyle=ls, label=lb) for (ls, lb) in zip(['solid', 'dotted'], ['Ref', 'NoAGN'])]
    #lh = axes[0][0].get_legend_handles_labels()
    legend = lax.legend(handles= legend_handles_a + legend_handles + legend_handles_2a + legend_handles_2 + legend_handles_styles, fontsize = fontsize, ncol=3, loc = 'upper center', bbox_to_anchor=(0.5, 0.75))
    legend.set_title(legtitles[splitax],  prop = {'size': fontsize})
    lax.axis('off')

    # turn the y axis labels off for realzies
    #axes[0][1].yaxis.set_major_formatter(dummyformatter)
    #axes[0][1].yaxis.set_minor_formatter(dummyformatter)
    
    plt.savefig(mdir+ 'PDs_coldens_o7-o8_L0050N0752Ref-NoAGN_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_by_%s.pdf'%(kind), bbox_inches='tight', format='pdf')
    
def plotgasproperties_noproj_AGNeffect(weight='Mass'):
    
    with h5py.File(ol.ndir + 'gashistogram_L0050N0752EagleVariation_NoAGN_28_PtAb_T4EOS.hdf5', 'r') as fo:
        hist_noagn = np.array(fo['histograms/%s'%weight])
        
        nedges_noagn = np.array(fo['edges/nH'])
        tedges_noagn = np.array(fo['edges/Temperature'])
        zedges_noagn = np.array(fo['edges/SmoothedMetallicity'])
        
        nax_noagn = np.where(np.array(fo['edges/axorder']) == 'nH')[0][0] 
        tax_noagn = np.where(np.array(fo['edges/axorder']) == 'Temperature')[0][0]
        zax_noagn = np.where(np.array(fo['edges/axorder']) == 'SmoothedMetallicity')[0][0]
        
        edges_noagn = {nax_noagn: nedges_noagn, tax_noagn: tedges_noagn, zax_noagn: zedges_noagn}

    with h5py.File(ol.ndir + 'gashistogram_L0050N0752REFERENCE_28_PtAb_T4EOS.hdf5', 'r') as fo:
        hist_ref = np.array(fo['histograms/%s'%weight])
        
        nedges_ref = np.array(fo['edges/nH'])
        tedges_ref = np.array(fo['edges/Temperature'])
        zedges_ref = np.array(fo['edges/SmoothedMetallicity'])
        
        nax_ref = np.where(np.array(fo['edges/axorder']) == 'nH')[0][0] 
        tax_ref = np.where(np.array(fo['edges/axorder']) == 'Temperature')[0][0]
        zax_ref = np.where(np.array(fo['edges/axorder']) == 'SmoothedMetallicity')[0][0]
        
        edges_ref = {nax_ref: nedges_ref, tax_ref: tedges_ref, zax_ref: zedges_ref}  

    handleinfedges_dct(edges_noagn)
    handleinfedges_dct(edges_ref)
    
    if not np.all([nax_noagn == nax_ref, tax_noagn == tax_ref, zax_noagn == zax_ref]):
        raise RuntimeError('Axes for Ref, NoAGN did not match')
    nax = nax_noagn
    tax = tax_noagn
    zax = zax_noagn
    
    axlabels = {nax: r'$\log_{10} \, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$',\
                tax: r'$\log_{10} \, T \; [\mathrm{K}]$',\
                zax: r'$\log_{10} \, Z \; [Z_{\odot}]$'}
    shortlabels = {nax: r'\log_{10} \, n_{\mathrm{H}}',\
                   tax: r'\log_{10} \, T',\
                   zax: r'\log_{10} \, Z'}
    
    lims = {nax: (-9.5, 4.), tax: (2.4, 8.5), zax: (-5., 1.2)}
    convs = {nax: 1., tax: 1., zax: ol.Zsun_sylviastables}

    plotaxes = [(nax, tax), (zax, nax), (tax, zax)]
    sumaxis  = [zax, tax, nax]
    ax0s = [min(axs) for axs in plotaxes] 
    ax1s = [max(axs) for axs in plotaxes]
    toplotaxes = [(0, 1) if axs[0] < axs[1] else (1, 0) for axs in plotaxes]
    
    color_noagn = 'magenta'
    color_ref = 'limegreen'
    
    linestyles = ['dotted', 'dashed', 'dashdot', 'solid']
    levels    = [0.9999, 0.99, 0.9, 0.5]
    
    colors_noagn = [color_noagn] * len(levels)
    colors_ref   = [color_ref]   * len(levels)
    cmap = 'gist_yarg'
    
    fontsize=12
    
    fig = plt.figure(figsize=(11.0, 9.0))
    grid = gsp.GridSpec(4, 4, width_ratios=[5., 5., 5., 1.], height_ratios=[2., 5., 5., 2.], wspace=0.35, hspace=0.0, top=0.95, bottom = 0.05, left = 0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([[fig.add_subplot(grid[xi,yi]) for yi in [0, 1, 2]] for xi in [0, 1, 2]]) # in mainaxes: x = column, y = row
    lax = fig.add_subplot(grid[3, 0:4])
    cax = fig.add_subplot(grid[1:3, 3])
    
    #fig, axes = plt.subplots(ncols=2, nrows=4, sharex = True, sharey=True,)
        
    bins = [[np.sum(hist_ref, axis=sumaxis[i])   / (np.diff(edges_ref[ax0s[i]])[:, np.newaxis]   *  np.diff(edges_ref[ax1s[i]])[np.newaxis, :])  / np.sum(hist_ref),\
             np.sum(hist_noagn, axis=sumaxis[i]) / (np.diff(edges_noagn[ax0s[i]])[:, np.newaxis] *  np.diff(edges_noagn[ax1s[i]])[np.newaxis, :]) / np.sum(hist_noagn)\
             ] for i in range(len(plotaxes))]
    bins1d = [[np.sum(hist_ref,   axis=(sumaxis[i], plotaxes[i][1])) / np.diff(edges_ref[plotaxes[i][0]])   / np.sum(hist_ref),\
               np.sum(hist_noagn, axis=(sumaxis[i], plotaxes[i][1])) / np.diff(edges_noagn[plotaxes[i][0]]) / np.sum(hist_noagn)\
              ] for i in range(len(plotaxes))]
              # 
              # 
    vmax = np.log10(max([np.max(np.array(bn)) for col in bins for bn in col]))
    vmin = vmax - 8.
    
    
    for axi in range(9):
        yi = axi // 3
        xi = axi % 3
        ax = axes[yi, xi]
        
        plotax = plotaxes[xi]
        ax0    = ax0s[xi]
        ax1    = ax1s[xi]
        shortlabel = shortlabels[plotax[0]]
        toplotax = toplotaxes[xi]
        #ax.text(0.05, 0.05, 'starting', transform=ax.transAxes)
        
        if yi == 0:
            ref   = np.log10(bins1d[xi][0])
            noagn = np.log10(bins1d[xi][1])
            
            ax.step(edges_ref[plotax[0]][:-1] - np.log10(convs[plotax[0]]),   np.max([ref, np.ones(len(ref)) * -100.], axis=0), color=color_ref, linestyle='solid', where='post')
            ax.step(edges_noagn[plotax[0]][:-1] - np.log10(convs[plotax[0]]), np.max([noagn, np.ones(len(noagn)) * -100.], axis=0), color=color_noagn, linestyle='solid', where='post')
            
            maxv = max([np.max(ref), np.max(noagn)]) + 0.2
            minv = maxv - 6.
            
            ax.set_xlim(*lims[plotax[0]])
            ax.set_ylim(minv, maxv)
            setticks(ax, fontsize, labelbottom=False, labelleft=True)
            ax.set_ylabel(r'$\log_{10} \left( \, \mathrm{d} f_{\mathrm{M}} \, / \, \mathrm{d}' + shortlabel + r'\, \right)$', fontsize=fontsize)
            

        elif yi == 1:
            bins_  = bins[xi][0]
            if toplotax[0] < toplotax[1]:
                bins_ = bins_.T
            img = ax.pcolormesh(edges_ref[plotax[0]] - np.log10(convs[plotax[0]]), edges_ref[plotax[1]] - np.log10(convs[plotax[1]]), np.log10(bins_), cmap=cmap, vmin=vmin, vmax=vmax)
            
            add_2dhist_contours_simple(ax, bins[xi][0], [edges_ref[ax0], edges_ref[ax1]], toplotaxes=toplotax,\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=-np.log10(convs[plotax[0]]), shifty=-np.log10(convs[plotax[1]]),
                        colors=colors_ref, linestyles=linestyles)
            add_2dhist_contours_simple(ax, bins[xi][1], [edges_noagn[ax0], edges_noagn[ax1]], toplotaxes=toplotax,\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=-np.log10(convs[plotax[0]]), shifty=-np.log10(convs[plotax[1]]),
                        colors=colors_noagn, linestyles=linestyles)
           
            ax.text(0.05,0.95, 'Reference', fontsize=fontsize, horizontalalignment='left', verticalalignment='top', color='black', transform=ax.transAxes) # bbox=dict(facecolor='white',alpha=0.3)
            ax.set_ylabel(axlabels[plotax[1]], fontsize=fontsize)
            setticks(ax, fontsize, labelbottom=False, labelleft=True)
            ax.set_xlim(*lims[plotax[0]])
            ax.set_ylim(*lims[plotax[1]])
           
        elif yi == 2:
            bins_  = bins[xi][1]
            if toplotax[0] < toplotax[1]:
                bins_ = bins_.T
            img = ax.pcolormesh(edges_noagn[plotax[0]] - np.log10(convs[plotax[0]]), edges_noagn[plotax[1]] - np.log10(convs[plotax[1]]), np.log10(bins_), cmap=cmap, vmin=vmin, vmax=vmax)
            
            add_2dhist_contours_simple(ax, bins[xi][0], [edges_ref[ax0], edges_ref[ax1]], toplotaxes=toplotax,\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=-np.log10(convs[plotax[0]]), shifty=-np.log10(convs[plotax[1]]),
                        colors=colors_ref, linestyles=linestyles)
            add_2dhist_contours_simple(ax, bins[xi][1], [edges_noagn[ax0], edges_noagn[ax1]], toplotaxes=toplotax,\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=-np.log10(convs[plotax[0]]), shifty=-np.log10(convs[plotax[1]]),
                        colors=colors_noagn, linestyles=linestyles)
            
            ax.text(0.05,0.95, 'NoAGN', fontsize=fontsize, horizontalalignment='left', verticalalignment='top', color='black', transform=ax.transAxes) # bbox=dict(facecolor='white',alpha=0.3)
            ax.set_xlabel(axlabels[plotax[0]], fontsize=fontsize)
            ax.set_ylabel(axlabels[plotax[1]], fontsize=fontsize)
            setticks(ax, fontsize, labelbottom=True, labelleft=True)
            ax.set_xlim(*lims[plotax[0]])
            ax.set_ylim(*lims[plotax[1]])

    #legend_handles_rho_o8 = [mlines.Line2D([], [], color=colors_rho_o8[i], linestyle = 'solid', label=labels_rho_o8[i]) for i in range(len(rhobins_o8))]
    #legend_rho_o8 = axes[1,1].legend(handles = legend_handles_rho_o8, fontsize = fontsize, ncol =2)
    #legend_rho_o8.set_title(r'$\log_{10} \rho \, [\mathrm{g}\, \mathrm{cm}^{-3}]$',  prop = {'size': fontsize})
    
    legend_handles_col = [mlines.Line2D([], [], color=color_ref, linestyle='solid', label='Reference'),\
                          mlines.Line2D([], [], color=color_noagn, linestyle='solid', label='NoAGN') ]
    legend_handles_lns = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.2f %%'%(levels[i] * 100.)) for i in range(len(levels))]
    #legend_handles_a = [mlines.Line2D([], [], color='black', linestyle='solid', label='total, Ref')]
    #legend_handles_2 = [mlines.Line2D([], [], color=colors2[i], linestyle='solid', label=leglabels['o7noagn'][i]) for i in range(len(leglabels['o7noagn']))]
    #legend_handles_2a = [mlines.Line2D([], [], color='gray', linestyle='solid', label='total, NoAGN')]
    #legend_handles_styles = [mlines.Line2D([], [], color='brown', linestyle=linestyles_tot[i], label='%.2f %%'%(100. * levels_tot[i])) for i in range(len(levels_tot))]
    #legend_handles = [mlines.Line2D([], [], color='black', linestyle='solid', label='total') for i in range(len(leglabels['o7ref']))] + legend_handles 
    #legend_handles2 = [mlines.Line2D([], [], color='gray', linestyle=ls, label=lb) for (ls, lb) in zip(['solid', 'dotted'], ['Ref', 'NoAGN'])]
    #lh = axes[0][0].get_legend_handles_labels()
    # legend = lax.legend()
    lax.legend(handles= legend_handles_col + legend_handles_lns, fontsize=fontsize, ncol=3, loc = 'upper center', bbox_to_anchor=(0.5, 0.5))
    #legend.set_title(legtitles[splitax],  prop = {'size': fontsize})
    lax.axis('off')
    
    add_colorbar(cax, img=img, vmin=vmin, vmax=vmax, cmap=cmap,\
                 clabel=r'$\log_{10}\, f_{\mathrm{M}} \,/\, \mathrm{pix. size} \; [\mathrm{dex}^{-2}]$',\
                 newax=False, extend='min',fontsize=fontsize, orientation='vertical')
    cax.tick_params(labelsize=fontsize)
    # turn the y axis labels off for realzies
    #axes[0][1].yaxis.set_major_formatter(dummyformatter)
    #axes[0][1].yaxis.set_minor_formatter(dummyformatter)
    
    plt.savefig(mdir+ 'PDs_gasmass_L0050N0752Ref-NoAGN_28_PtAb_T4EOS_by_%s.pdf'%(weight), bbox_inches='tight', format='pdf')
    
def ploto78diffs_Nvsdiff():
    '''
    left: No7 vs ib o7 (from o7) / ib o7 (from o8), contours: No8
    right: No8 vs ib o8 (from o7) / ib o8 (from o8), contours: No7
    '''

    hist = eaibreldiffo78byo78
    rescalef = 0.
    xlim = (10., 17.7)
    ylim = (-3.,5.)
    diffo7label = r'$\log_{10}\, f_{\mathrm{O\,VII}}(\mathrm{O\,VII})\, /\, f_{\mathrm{O\,VII}}(\mathrm{O\,VIII})$'
    diffo8label = r'$\log_{10}\, f_{\mathrm{O\,VIII}}(\mathrm{O\,VII})\, /\, f_{\mathrm{O\,VIII}}(\mathrm{O\,VIII})$'
    No7label = r'$\log_{10}\, N_{\mathrm{O\,VII}} \, [\mathrm{cm}^{-2}]$'
    No8label = r'$\log_{10}\, N_{\mathrm{O\,VIII}} \, [\mathrm{cm}^{-2}]$'
    difflabel = r'$\log_{10}\, f_{\mathrm{ion}}(\mathrm{O\,VII})\, /\, f_{\mathrm{ion}}(\mathrm{O\,VIII})$'
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    pixdens = True
        
    vmin = -6.
    vmin1, vmax1 = getminmax2d(hist, axis=(1,3), log=True, pixdens=True) # axis is summed over; 
    vmin2, vmax2 = getminmax2d(hist, axis=(0,2), log=True, pixdens=True)
    vmax = max(vmax1, vmax2)
    vmin = vmax - 7.
    cmap = 'bone_r'
    fontsize = 20.
    sfi_a = '(a)'
    sfi_b = '(b)'
    textcolor = 'black'
    
    labelyax2 = False

    fig = plt.figure(figsize=(15.,10.))
    grid = gsp.GridSpec(2,3,height_ratios=[8.,2.],width_ratios=[7.,7.,1.],wspace=0.0)
    ax1 = plt.subplot(grid[0,0], facecolor=mpl.cm.get_cmap(cmap)(0.)) 
    ax2 = plt.subplot(grid[0,1], facecolor=mpl.cm.get_cmap(cmap)(0.)) 
    cax = plt.subplot(grid[0,2])
    lax = plt.subplot(grid[1,0:2])
    #lax2 = plt.subplot(grid[1,1])       
    ncols_legend = 4
    legendloc = 9
    legend_bbox_to_anchor = (0.5,0.95)
    #if slidemode: # ax3 for legend right of plot
    #    fontsize=14
    #    fig = plt.figure(figsize=(12.,6.))
    #    grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
    #    ax3 = plt.subplot(grid[0,0]) 
    #    ax1 = plt.subplot(grid[0,1], facecolor = 'black')
    #    ax2 = plt.subplot(grid[0,2])
    #    ncols_legend = 1 
    #    legendloc= 1
    #    legend_bbox_to_anchor=(0.95,1.)

    
    # account for small size of the plot -> larger ticks
    ax1.tick_params(length =7., width = 1., which = 'major', axis='both')
    ax1.tick_params(length =4., width = 1., which = 'minor', axis='both')
    ax2.tick_params(length =7., width = 1., which = 'major', axis='both')
    ax2.tick_params(length =4., width = 1., which = 'minor', axis='both') 
    cax.tick_params(length =7., width = 1., which = 'major', axis='both')
    cax.tick_params(length =4., width = 1., which = 'minor', axis='both') 
   
    # plot background and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (0,2),log=True, usepcolor = True, vmin = vmin, vmax=vmax, cmap=cmap, shiftx=rescalef, shifty=rescalef, pixdens=pixdens)
    img, vmin, vmax = add_2dplot(ax2,hist, (1,3),log=True, usepcolor = True, vmin = vmin, vmax=vmax, cmap=cmap, shiftx=rescalef, shifty=rescalef, pixdens=pixdens)

    # add colorbar
    if pixdens:
        clabel = r'$\log_{10} \mathrm{sightlines} \, \mathrm{dex}^{-2}$'
    else:
        clabel = r'$\log_{10}$  fraction of sightlines'
    add_colorbar(cax,img=img,clabel=clabel, extend = 'min',fontsize=fontsize)
    cax.set_aspect(10.)
    cax.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    dimlabels = (r'$N_{O\,VII}$', r'$N_{O\,VIII}$', None,None)
    
    # col.dens. 12, 14, 15, 16, 17,
    #if ion == 'o7' and slices == 16:
    lims_o7 = [71,91,101,111,121]
    #elif ion == 'o7' and slices == 1:
    #    lims = [34,54,64,74,84]
    #if ion == 'o8' and slices == 16:
    lims_o8 = [44,64,74,84,94]
    #elif ion == 'o8' and slices == 1:
    #    lims = [26,46,56,66,76]
    
    projaxes1 = (0,2)
    projaxes2 = (1,3)
    colors_b = ['red', 'orange', 'gold', 'green', 'blue', 'darkviolet'] 
    
    selection_o7 = [[(None, None, None, None), (lims_o7[i], None, None, None)] if i ==0 else\
                    [(lims_o7[i-1], None, None, None), (None, None, None, None)] if i == len(lims_o7) else\
                    [(lims_o7[i-1], None, None, None), (lims_o7[i], None, None, None)]\
                    for i in range(len(lims_o7)+1)]
    selection_o8 = [[(None, None, None, None), (None,lims_o8[i], None, None)] if i ==0 else\
                    [(None, lims_o8[i-1], None, None), (None, None, None, None)] if i == len(lims_o8) else\
                    [(None, lims_o8[i-1], None, None), (None, lims_o8[i], None, None)]\
                    for i in range(len(lims_o8)+1)]
    colors = [(color,)*len(fraclevels) for color in colors_b]

    for i in range(len(lims_o7)+1):
        add_2dhist_contours(ax1, hist, projaxes1, mins=selection_o8[i][0], maxs=selection_o8[i][1], histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles, colors=colors[i], dimlabels=dimlabels,\
			                 legendlabel_pre=None, shiftx=rescalef, shifty=rescalef)			 
    for i in range(len(lims_o8)+1):
        add_2dhist_contours(ax2, hist, projaxes2, mins=selection_o7[i][0], maxs=selection_o7[i][1], histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles, colors=colors[i], dimlabels=dimlabels,\
			                 legendlabel_pre=None, shiftx=rescalef, shifty=rescalef)			
    
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right=True, top=True, axis='both', which = 'both', color=textcolor)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right=True, top=True, labelleft=labelyax2, axis='both', which = 'both', color=textcolor)
    ax1.spines['right'].set_color(textcolor)
    ax1.spines['left'].set_color(textcolor)
    ax1.spines['top'].set_color(textcolor)
    ax1.spines['bottom'].set_color(textcolor)
    ax2.spines['right'].set_color(textcolor)
    ax2.spines['left'].set_color(textcolor)
    ax2.spines['top'].set_color(textcolor)
    ax2.spines['bottom'].set_color(textcolor)
    
    linestyle_diffs = ['solid', 'dashed', 'dotted']
    color_diffs = 'dodgerblue'
    

    labels_diffs = ['equal', r'$\pm 0.5$ dex', r'$\pm 1$ dex']
    pm = [0.5,1.]
    

    ax1.axhline(0., color=color_diffs, linestyle=linestyle_diffs[0])
    ax1.axhline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
    ax1.axhline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
    ax1.axhline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
    ax1.axhline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
    ax2.axhline(0., color=color_diffs, linestyle=linestyle_diffs[0])
    ax2.axhline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
    ax2.axhline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
    ax2.axhline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
    ax2.axhline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
        
    # set aspect and axis limits (gets messed up if done efore tick label resetting)
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
    ax1.set_xlabel(No7label,fontsize=fontsize)
    ax1.set_ylabel(difflabel,fontsize=fontsize)
    ax2.set_xlabel(No8label,fontsize=fontsize)    
    if labelyax2:
        ax2.set_ylabel(diffo8label,fontsize=fontsize) 
    
    # remove rightmost tick label where it's a problem (prune by hand because MaxNLocator changes all the label locations)
    # important: after limit setting, before aspect ratio setting
    if False:
        # only major ticks
        old_ticklocs = ax1.get_xticks() #array	
        old_ticklocs_min = ax1.get_xticks(minor=True) #array	
        ax1.set_xticks(old_ticklocs[:-1]) 
        ax1.set_xticks(old_ticklocs_min, minor=True)
        ax1.set_xlim(*xlim)
        
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()
    ax2.set_aspect((xlim2[1]-xlim2[0])/(ylim2[1]-ylim2[0]), adjustable='box-forced')
    
    # subfig indices
    ax1.text(0.95,0.05,sfi_a, fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax1.transAxes, color=textcolor)
    ax2.text(0.95,0.05,sfi_b, fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax2.transAxes, color=textcolor)
    # subfig titles
    subtitleloc = (0.05, 0.95)
    subtitle_o7 = r'$\mathrm{O\, VII}\, f_{\mathrm{ion}},  N_{\mathrm{O\, VIII}}$ contours'
    subtitle_o8 = r'$\mathrm{O\, VIII}\, f_{\mathrm{ion}}, N_{\mathrm{O\, VII}}$ contours'
    ax1.text(subtitleloc[0],subtitleloc[1],subtitle_o7,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes, bbox=dict(facecolor='white',alpha=0.3), color=textcolor)
    ax2.text(subtitleloc[0],subtitleloc[1],subtitle_o8,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax2.transAxes, bbox=dict(facecolor='white',alpha=0.3), color=textcolor)
    # set labels manually: if edges match for O7 and O8, show one legend, and just leave out the fractions (they're on other plots already)
    edgesmatch = np.all(hist['edges'][0][np.array(lims_o7)] == hist['edges'][1][np.array(lims_o8)])
    if edgesmatch:
        cutq = 'N'
        sublabels = [r'$%.1f < %s$'%(hist['edges'][0][lims_o7[bind-1]], cutq) if bind == len(lims_o7) else\
                     r'$%s < %.1f$'%(cutq, hist['edges'][0][lims_o7[bind]]) if bind == 0 else\
                     r'$%.1f < %s < %.1f$'%(hist['edges'][0][lims_o7[bind-1]], cutq, hist['edges'][0][lims_o7[bind]])\
                     for bind in range(len(lims_o7)+1)]
        handles_subs = [mlines.Line2D([], [], color=colors_b[i], linestyle='solid', label=sublabels[i]) for i in range(len(lims_o7)+1)]
    else:
        print('Warning: column density edges do not match for the different ions')
        handles_subs1, labels_subs1 = ax1.get_legend_handles_labels()     
        handles_subs2, labels_subs2 = ax2.get_legend_handles_labels()
        handles_subs = handles_subs1 + handles_subs2
        #labels_subs  = labels_subs1 + labels_subs2
    # set up legend in ax below main figure
    
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]    
    diffs_legend_handles = [mlines.Line2D([], [], color=color_diffs, linestyle = linestyle_diffs[i], label=labels_diffs[i]) for i in range(len(linestyle_diffs))]
    lax.legend(handles=handles_subs + level_legend_handles + diffs_legend_handles, fontsize=fontsize, ncol=ncols_legend, loc=legendloc, bbox_to_anchor=legend_bbox_to_anchor)
    lax.axis('off')
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')

    plt.savefig(mdir + 'ion_diffs/coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_N_vs_ibreldiff.pdf', format = 'pdf',bbox_inches='tight')
    

def plotiondiffs_Nvsdiff(kind):
    '''
    left: No7 vs ib o7 (from o7) / ib o7 (from o8), contours: No8
    right: No8 vs ib o8 (from o7) / ib o8 (from o8), contours: No7
    
    assumes axes 0, 1, 2, 3 are log column densities of ions 1, 2, quantities to compare for ions 1, 2
    (some vairable names assume ion1 = o7, ion2 = o8)
    '''
    ## ion-specific quantities
    
    # col.dens 12, 13, 14, 15, 16
    lims_o6_27 = [34, 44, 54, 64, 74]
    lims_o6_28 = [21, 31, 41, 51, 61]
    lims_ne8_28 = [34, 44, 54, 64, 74]
    
    # col. dens 12, 13, 14, 15, 16
    lims_o8_27 = [33, 43, 53, 63, 73]
    # col.dens. 12, 14, 15, 16, 17,
    #if ion == 'o7' and slices == 16:
    lims_o7_28 = [71,91,101,111,121]
    #elif ion == 'o7' and slices == 1:
    #    lims = [34,54,64,74,84]
    #if ion == 'o8' and slices == 16:
    lims_o8_28 = [44,64,74,84,94]
    #elif ion == 'o8' and slices == 1:
    #    lims = [26,46,56,66,76]
    
    # length should be sufficient for the number of bins
    colors_b = ['red', 'orange', 'gold', 'green', 'blue', 'darkviolet'] 
    color_backtrace = 'saddlebrown'
    ## some defaults
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    pixdens = True
    vdynrange = 10.
    xlim = None
    ylim = None
    labelyax2 = True
    labelxax1 = False
    qtylabel2 = ''
    prunex1 = False
    pruney1 = False
    # rescalef: plotted quantity (axes 2,3) is rescaled in plots (for plotting in non-stored units, e.g. solar metallicity, n_H)
    # N_vs_qty: plot quantity against column density (otherwise: quantity against quantity)
    # isdiff: qty is a difference 
    
    # snap 28 O7 vs O8
    if kind == 'eaibreldiffo78byo78':
        hist = eaibreldiffo78byo78
        rescalef = 0.
        xlim = (10., 17.7)
        ylim = (-2.,4.)
        qtylabel = r'$\log_{{10}}\, f_{{\mathrm{{{0}}}}}(\mathrm{{O\,VII}})\, /\, f_{{\mathrm{{{0}}}}}(\mathrm{{O\,VIII}})$'
        qtylabel_short = r'f_{{\mathrm{{{0}}}}}'
        name = 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_N_vs_ibreldiff'
        ion1 = 'o7'
        ion2 = 'o8'
        vdynrange = 9.
        limsion1 = lims_o7_28
        limsion2 = lims_o8_28
        N_vs_qty = True
        isdiff = True
        pruney1 = True
    elif kind == 'eaTo78':
        hist = eaTo78
        rescalef = 0.
        xlim = (3., 7.2)
        ylim = xlim
        name = 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_T_by_o7_vs_by_o8'
        qtylabel = r'$\log_{{10}}\, T_{{\mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        qtylabel_short = r'$T_{{\mathrm{{{0}}}}}$'
        ion1 = 'o7'
        ion2 = 'o8'
        vdynrange = 6.
        limsion1 = lims_o7_28
        limsion2 = lims_o8_28
        N_vs_qty = False
        isdiff = False
        fraclevels = [0.99,0.50] 
        linestyles = ['dashed','solid']
    elif kind == 'earhoo78':
        hist = earhoo78
        rescalef = np.log10(rho_to_nh)
        xlim = (-8., 0.)
        ylim = xlim
        name = 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_rho_by_o7_vs_by_o8'
        qtylabel_short = r'$n_{{H, \mathrm{{{0}}}}}$'
        qtylabel = r'$\log_{{10}}\, n_{{H, \mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        ion1 = 'o7'
        ion2 = 'o8'
        vdynrange = 6.
        limsion1 = lims_o7_28
        limsion2 = lims_o8_28
        N_vs_qty = False
        isdiff = False
        pruney1 = True
        fraclevels = [0.99,0.50] 
        linestyles = ['dashed','solid']
    elif kind == 'eafOo78':
        hist = eafOo78
        rescalef = -1*np.log10(ol.solar_abunds['oxygen'])
        xlim = (-1.5, 0.8)
        ylim = xlim
        name = 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_fO_by_o7_vs_by_o8'
        qtylabel_short = r'$f_{{O, \mathrm{{{0}}}}}'
        qtylabel = r'$\log_{{10}}\, f_{{O, \mathrm{{{0}}}}}\, [\odot]$'
        ion1 = 'o7'
        ion2 = 'o8'
        vdynrange = 6.
        limsion1 = lims_o7_28
        limsion2 = lims_o8_28
        N_vs_qty = False
        isdiff = False
        fraclevels = [0.99,0.50] 
        linestyles = ['dashed','solid']
    elif kind == 'eafO-Smo78':
        hist = eafOSmo78
        rescalef = -1*np.log10(ol.solar_abunds['oxygen'])
        xlim = (-1.5, 0.8)
        ylim = xlim
        name = 'coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Sm-fO_by_o7_vs_by_o8'
        qtylabel_short = r'$f_{{O, \mathrm{{{0}}}}}'
        qtylabel = r'$\log_{{10}}\, f_{{O, \mathrm{{{0}}}}}\, [\odot]$'
        ion1 = 'o7'
        ion2 = 'o8'
        vdynrange = 6.
        limsion1 = [5, 25, 35, 45, 55]
        limsion2 = [5, 25, 35, 45, 55]
        N_vs_qty = False
        isdiff = False
        fraclevels = [0.99,0.50] 
        linestyles = ['dashed','solid']
    # snap 27 O6 vs O8
    elif kind == 'earhoo68_27':
        hist = earhoo68_27
        rescalef = np.log10(rho_to_nh)
        xlim = (-8., 0.)
        ylim = xlim
        name = 'Density_w_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_rhodiff'
        qtylabel_short = r'$n_{{H, \mathrm{{{0}}}}}$'
        qtylabel = r'$\log_{{10}}\, n_{{H, \mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        ion1 = 'o6'
        ion2 = 'o8'
        limsion1 = lims_o6_27
        limsion2 = lims_o8_27
        N_vs_qty = False
        isdiff = False
    elif kind == 'eaTo68_27':
        hist = eaTo68_27
        rescalef = 0.
        xlim = (3., 8.)
        ylim = xlim
        name = 'Temperature_w_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel = r'$\log_{{10}}\, T_{{\mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        qtylabel_short = r'$T_{{\mathrm{{{0}}}}}$'
        ion1 = 'o6'
        ion2 = 'o8'
        limsion1 = lims_o6_27
        limsion2 = lims_o8_27
        N_vs_qty = False
        isdiff = False
    elif kind == 'eaTo6ne8':
        hist = eaTo6ne8 
        rescalef = 0.
        xlim = (3., 8.)
        ylim = xlim
        name = 'Temperature_w_coldens_o6-ne8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel = r'$\log_{{10}}\, T_{{\mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        qtylabel_short = r'$T_{{\mathrm{{{0}}}}}$'
        ion1 = 'ne8'
        ion2 = 'o6'
        limsion1 = lims_ne8_28
        limsion2 = lims_o6_28
        N_vs_qty = False
        isdiff = False
    elif kind == 'eaTo7ne8':
        hist = eaTo7ne8 
        rescalef = 0.
        xlim = (3., 8.)
        ylim = xlim
        name = 'Temperature_w_coldens_o7-ne8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel = r'$\log_{{10}}\, T_{{\mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        qtylabel_short = r'$T_{{\mathrm{{{0}}}}}$'
        ion1 = 'ne8'
        ion2 = 'o7'
        limsion1 = lims_ne8_28
        limsion2 = lims_o7_28
        N_vs_qty = False
        isdiff = False
    elif kind == 'eaTo8ne8':
        hist = eaTo8ne8 
        rescalef = 0.
        xlim = (3., 8.)
        ylim = xlim
        name = 'Temperature_w_coldens_o8-ne8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel = r'$\log_{{10}}\, T_{{\mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        qtylabel_short = r'$T_{{\mathrm{{{0}}}}}$'
        ion1 = 'ne8'
        ion2 = 'o8'
        limsion1 = lims_ne8_28
        limsion2 = lims_o8_28
        N_vs_qty = False
        isdiff = False
    elif kind == 'earhoo6ne8':
        hist = earhoo6ne8 
        rescalef = np.log10(rho_to_nh)
        xlim = (-8., 0.)
        ylim = xlim
        name = 'Density_w_coldens_o6-ne8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel_short = r'$n_{{H, \mathrm{{{0}}}}}$'
        qtylabel = r'$\log_{{10}}\, n_{{H, \mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        ion1 = 'ne8'
        ion2 = 'o6'
        limsion1 = lims_ne8_28
        limsion2 = lims_o6_28
        N_vs_qty = False
        isdiff = False
    elif kind == 'earhoo7ne8':
        hist = earhoo7ne8 
        rescalef = np.log10(rho_to_nh)
        xlim = (-8., 0.)
        ylim = xlim
        name = 'Density_w_coldens_o7-ne8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel_short = r'$n_{{H, \mathrm{{{0}}}}}$'
        qtylabel = r'$\log_{{10}}\, n_{{H, \mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        ion1 = 'ne8'
        ion2 = 'o7'
        limsion1 = lims_ne8_28
        limsion2 = lims_o7_28
        N_vs_qty = False
        isdiff = False
    elif kind == 'earhoo8ne8':
        hist = earhoo8ne8 
        rescalef = np.log10(rho_to_nh)
        xlim = (-8., 0.)
        ylim = xlim
        name = 'Density_w_coldens_o8-ne8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_Tdiff'
        qtylabel_short = r'$n_{{H, \mathrm{{{0}}}}}$'
        qtylabel = r'$\log_{{10}}\, n_{{H, \mathrm{{{0}}}}}\, [\mathrm{{cm}}^{{-3}}]$'
        ion1 = 'ne8'
        ion2 = 'o8'
        limsion1 = lims_ne8_28
        limsion2 = lims_o8_28
        N_vs_qty = False
        isdiff = False
        
    ion1label = ionlabels[ion1]
    ion2label = ionlabels[ion2]
    Nlabel_base = r'$\log_{{10}} N_{{\mathrm{{{0}}}}} \, [\mathrm{{cm}}^{{-2}}]$'
    Nion1label = Nlabel_base.format(ion1label)
    Nion2label = Nlabel_base.format(ion2label)
    Nionlabel = Nlabel_base.format('ion')
    qtylabel1 = qtylabel.format(ion1label)
    qtylabel2 = qtylabel.format(ion2label)
    qtylabel  = qtylabel.format('ion')
    
    cmap = 'bone_r'
    fontsize = 12.
    sfi_a = '(a)'
    sfi_b = '(b)'
    textcolor = 'black'

    fig = plt.figure(figsize=(5.5, 10.))
    grid = gsp.GridSpec(3, 2, height_ratios=[6., 6., 3.5],width_ratios=[8.,1.],wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0,0], facecolor=mpl.cm.get_cmap(cmap)(0.)) 
    ax2 = plt.subplot(grid[1,0], facecolor=mpl.cm.get_cmap(cmap)(0.)) 
    cax = plt.subplot(grid[0:2,1])
    lax = plt.subplot(grid[2,:])
    #lax2 = plt.subplot(grid[1,1])       
    ncols_legend = 3
    legendloc = 'lower center'
    legend_bbox_to_anchor = (0.5, -0.1)
    #if slidemode: # ax3 for legend right of plot
    #    fontsize=14
    #    fig = plt.figure(figsize=(12.,6.))
    #    grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
    #    ax3 = plt.subplot(grid[0,0]) 
    #    ax1 = plt.subplot(grid[0,1], facecolor = 'black')
    #    ax2 = plt.subplot(grid[0,2])
    #    ncols_legend = 1 
    #    legendloc= 1
    #    legend_bbox_to_anchor=(0.95,1.)

    
    # account for small size of the plot -> larger ticks
    #ax1.tick_params(length =7., width = 1., which = 'major', axis='both')
    #ax1.tick_params(length =4., width = 1., which = 'minor', axis='both')
    #ax2.tick_params(length =7., width = 1., which = 'major', axis='both')
    #ax2.tick_params(length =4., width = 1., which = 'minor', axis='both') 
    #cax.tick_params(length =7., width = 1., which = 'major', axis='both')
    #cax.tick_params(length =4., width = 1., which = 'minor', axis='both') 
   
    # plot background and colorbar: total distribution
    if N_vs_qty:
        projaxes1 = (0, 2)
        projaxes2 = (1, 3)
        projaxes_c1 = (1, 3)
        projaxes_c2 = (0, 2)
    else:
        projaxes1 = (2, 3)
        projaxes2 = (2, 3)
        projaxes_c1 = (0, 1)
        projaxes_c2 = (0, 1)
        
    vmin1, vmax1 = getminmax2d(hist, axis=projaxes_c1, log=True, pixdens=pixdens) # axis is summed over; 
    vmin2, vmax2 = getminmax2d(hist, axis=projaxes_c2, log=True, pixdens=pixdens)
    vmax = max(vmax1, vmax2)
    vmin = max(vmax - vdynrange, min(vmin1, vmin2))
    img, vmin, vmax = add_2dplot(ax1, hist, projaxes1, log=True, usepcolor = True, vmin = vmin, vmax=vmax, cmap=cmap, shiftx=rescalef, shifty=rescalef, pixdens=pixdens)
    img, vmin, vmax = add_2dplot(ax2, hist, projaxes2, log=True, usepcolor = True, vmin = vmin, vmax=vmax, cmap=cmap, shiftx=rescalef, shifty=rescalef, pixdens=pixdens)

    # add colorbar
    if pixdens:
        clabel = r'$\log_{10}\, \mathrm{sightline\, fraction} \, \mathrm{dex}^{-2}$'
    else:
        clabel = r'$\log_{10}$ fraction of sightlines'
    add_colorbar(cax, img=img, clabel=clabel, extend='min', fontsize=fontsize)
    cax.set_aspect(15.)
    cax.tick_params(labelsize=fontsize, axis='both')
   
    # plot contour levels for column density subsets
    dimlabels = (r'$N_{\mathrm{%s}}$'%ion1label, r'$N_{\mathrm{%s}}$'%ion2label, None,None)
    
    
    selection_ion1 = [[(None, None, None, None), (limsion1[i], None, None, None)] if i ==0 else\
                    [(limsion1[i-1], None, None, None), (None, None, None, None)] if i == len(limsion1) else\
                    [(limsion1[i-1], None, None, None), (limsion1[i], None, None, None)]\
                    for i in range(len(limsion1)+1)]
    selection_ion2 = [[(None, None, None, None), (None,limsion2[i], None, None)] if i ==0 else\
                    [(None, limsion2[i-1], None, None), (None, None, None, None)] if i == len(limsion2) else\
                    [(None, limsion2[i-1], None, None), (None, limsion2[i], None, None)]\
                    for i in range(len(limsion2)+1)]
    colors = [(color,)*len(fraclevels) for color in colors_b]

    if N_vs_qty:
        selection_ax1 = selection_ion2
        selection_ax2 = selection_ion1
    else:
        selection_ax1 = selection_ion1
        selection_ax2 = selection_ion2
    for i in range(len(selection_ax1)):
        add_2dhist_contours(ax1, hist, projaxes1, mins=selection_ax1[i][0], maxs=selection_ax1[i][1], histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles, colors=colors[i], dimlabels=dimlabels,\
			                 legendlabel_pre=None, shiftx=rescalef, shifty=rescalef, linewidth=2)             
    for i in range(len(selection_ax2)):
        add_2dhist_contours(ax2, hist, projaxes2, mins=selection_ax2[i][0], maxs=selection_ax2[i][1], histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles, colors=colors[i], dimlabels=dimlabels,\
			                 legendlabel_pre=None, shiftx=rescalef, shifty=rescalef, linewidth=2)	
    if kind == 'eaibreldiffo78byo78':
        hist_temp = np.sum(hist['bins'], axis=projaxes_c1)
        yax = projaxes1[1]
        percentiles =  0.5  + np.append( - np.array(fraclevels) / 2., np.array(fraclevels)[::-1] / 2.)  #fraclevels = [0.99,0.50,0.10] 
        linestyles_perc = np.append(np.array(linestyles), np.array(linestyles)[::-1])
        yvals = percentiles_from_histogram(hist_temp, hist['edges'][yax], axis=1, percentiles=percentiles)
        xvals = hist['edges'][projaxes1[0]]
        xvals = xvals[:-1] + 0.5 * np.diff(xvals)
        where = np.where(np.sum(hist_temp, axis=1) >= 0)
        for pind in range(len(percentiles)):
            ax1.plot(xvals[where], yvals[pind][where], linestyle=linestyles_perc[pind], color=color_backtrace)
        
        hist_temp = np.sum(hist['bins'], axis=projaxes_c2)
        yax = projaxes2[1]
        yvals = percentiles_from_histogram(hist_temp, hist['edges'][yax], axis=1, percentiles=percentiles)
        xvals = hist['edges'][projaxes2[0]]
        xvals = xvals[:-1] + 0.5 * np.diff(xvals)
        where = np.where(np.sum(hist_temp, axis=1) >= 0)
        for pind in range(len(percentiles)):
            ax2.plot(xvals[where], yvals[pind][where], linestyle=linestyles_perc[pind], color=color_backtrace)
            
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right=True, top=True, labelbottom=labelxax1, axis='both', which = 'both', color=textcolor)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right=True, top=True, labelleft=labelyax2, axis='both', which = 'both', color=textcolor)
    ax1.spines['right'].set_color(textcolor)
    ax1.spines['left'].set_color(textcolor)
    ax1.spines['top'].set_color(textcolor)
    ax1.spines['bottom'].set_color(textcolor)
    ax2.spines['right'].set_color(textcolor)
    ax2.spines['left'].set_color(textcolor)
    ax2.spines['top'].set_color(textcolor)
    ax2.spines['bottom'].set_color(textcolor)
    
    linestyle_diffs = ['solid', 'dashed', 'dotted']
    color_diffs = 'dodgerblue'
    

    labels_diffs = ['equal', r'$\pm 0.5$ dex', r'$\pm 1$ dex']
    pm = [0.5,1.]
    
    # set x, y limits if not specified
    if xlim is None:
        xlim1 = ax1.get_xlim()
        xlim2 = ax2.get_xlim()
        xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
    if ylim is None:
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    
    if N_vs_qty:
        if isdiff:
            ax1.axhline(0., color=color_diffs, linestyle=linestyle_diffs[0])
            ax1.axhline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.axhline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.axhline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax1.axhline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.axhline(0., color=color_diffs, linestyle=linestyle_diffs[0])
            ax2.axhline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.axhline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.axhline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.axhline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
        # else: no sensible difference lines to plot
        
        # set aspect and axis limits (gets messed up if done efore tick label resetting)
        if labelxax1:
            ax1.set_xlabel(Nion1label, fontsize=fontsize)
            ax2.set_xlabel(Nion2label, fontsize=fontsize)
        else:
            ax2.set_xlabel(Nionlabel, fontsize=fontsize)      
        if labelyax2:
            ax2.set_ylabel(qtylabel2, fontsize=fontsize) 
            ax1.set_ylabel(qtylabel1, fontsize=fontsize)
        else:
            ax1.set_ylabel(qtylabel, fontsize=fontsize)
    else:
        if isdiff: # mark differences on each axis
            ax1.axhline(0., color=color_diffs, linestyle=linestyle_diffs[0])
            ax1.axhline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.axhline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.axhline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax1.axhline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.axhline(0., color=color_diffs, linestyle=linestyle_diffs[0])
            ax2.axhline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.axhline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.axhline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.axhline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            
            ax1.axvline(0., color=color_diffs, linestyle=linestyle_diffs[0])
            ax1.axvline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.axvline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.axvline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax1.axvline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.axvline(0., color=color_diffs, linestyle=linestyle_diffs[0])
            ax2.axvline(pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.axvline(-1.*pm[0], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.axvline(pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.axvline(-1.*pm[1], color=color_diffs, linestyle=linestyle_diffs[2])
        
        else: # plot equality lines
            ax1.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], color=color_diffs, linestyle=linestyle_diffs[0])
            ax1.plot([xlim[0], xlim[1]],[xlim[0] + pm[0], xlim[1] + pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.plot([xlim[0], xlim[1]],[xlim[0] - pm[0], xlim[1] - pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
            ax1.plot([xlim[0], xlim[1]],[xlim[0] + pm[1], xlim[1] + pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
            ax1.plot([xlim[0], xlim[1]],[xlim[0] - pm[1], xlim[1] - pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
            
            ax2.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], color=color_diffs, linestyle=linestyle_diffs[0])
            ax2.plot([xlim[0], xlim[1]],[xlim[0] + pm[0], xlim[1] + pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.plot([xlim[0], xlim[1]],[xlim[0] - pm[0], xlim[1] - pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
            ax2.plot([xlim[0], xlim[1]],[xlim[0] + pm[1], xlim[1] + pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
            ax2.plot([xlim[0], xlim[1]],[xlim[0] - pm[1], xlim[1] - pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
            
        # set aspect and axis limits (gets messed up if done efore tick label resetting)
        ax2.set_xlabel(qtylabel1,fontsize=fontsize)
        ax1.set_ylabel(qtylabel2,fontsize=fontsize)
        if labelxax1:
            ax1.set_xlabel(qtylabel1,fontsize=fontsize)    
        if labelyax2:
            ax2.set_ylabel(qtylabel2,fontsize=fontsize) 
             
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
        
    # remove rightmost tick label where it's a problem (prune by hand because MaxNLocator changes all the label locations)
    # important: after limit setting, before aspect ratio setting
    if prunex1:
        # only major ticks
        old_ticklocs = ax1.get_xticks() #array	
        old_ticklocs_min = ax1.get_xticks(minor=True) #array	
        ax1.set_xticks(old_ticklocs[:-1]) 
        ax1.set_xticks(old_ticklocs_min, minor=True)
        ax1.set_xlim(*xlim)
    if pruney1:
        # only major ticks
        old_ticklocs = ax1.get_yticks() #array	
        old_ticklocs_min = ax1.get_yticks(minor=True) #array	
        ax1.set_yticks(old_ticklocs[1:]) 
        ax1.set_yticks(old_ticklocs_min, minor=True)
        ax1.set_ylim(*ylim)
        
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()
    ax2.set_aspect((xlim2[1]-xlim2[0])/(ylim2[1]-ylim2[0]), adjustable='box-forced')
    
    # subfig indices
    ax1.text(0.95, 0.05, sfi_a, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, color=textcolor)
    ax2.text(0.95, 0.05, sfi_b, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, color=textcolor)
    # subfig titles
    subtitleloc = (0.05, 0.95)
    if N_vs_qty:
        if not labelyax2:
            subtitle1 = r'$%s,  N_{\mathrm{%s}}$ contours'%(qtylabel_short%ion1label, ion2label)
            subtitle2 = r'$%s,  N_{\mathrm{%s}}$ contours'%(qtylabel_short%ion2label, ion1label)
        elif not labelxax1:
            subtitle1 = r'$\mathrm{ion} = \mathrm{%s},  N_{\mathrm{%s}}$ contours'%(ion1label, ion2label)
            subtitle2 = r'$\mathrm{ion} = \mathrm{%s},  N_{\mathrm{%s}}$ contours'%(ion2label, ion1label)
        else:
            subtitle1 = r'$N_{\mathrm{%s}}$ contours'%(ion2label)
            subtitle2 = r'$N_{\mathrm{%s}}$ contours'%(ion1label)
    else:
        subtitle1 = r'$N_{\mathrm{%s}}$ contours'%(ion1label)
        subtitle2 = r'$N_{\mathrm{%s}}$ contours'%(ion2label)
    ax1.text(subtitleloc[0], subtitleloc[1], subtitle1,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes, bbox=dict(facecolor='white',alpha=0.3), color=textcolor)
    ax2.text(subtitleloc[0], subtitleloc[1], subtitle2,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax2.transAxes, bbox=dict(facecolor='white',alpha=0.3), color=textcolor)
    # set labels manually: if edges match for O7 and O8, show one legend, and just leave out the fractions (they're on other plots already)
    edgesmatch = np.all(hist['edges'][0][np.array(limsion1)] == hist['edges'][1][np.array(limsion2)])
    if edgesmatch:
        cutq = 'N'
        sublabels = [r'$%.1f < %s$'%(hist['edges'][0][limsion1[bind-1]], cutq) if bind == len(limsion1) else\
                     r'$%s < %.1f$'%(cutq, hist['edges'][0][limsion1[bind]]) if bind == 0 else\
                     r'$%.1f < %s < %.1f$'%(hist['edges'][0][limsion1[bind-1]], cutq, hist['edges'][0][limsion1[bind]])\
                     for bind in range(len(limsion1)+1)]
        handles_subs = [mlines.Line2D([], [], color=colors_b[i], linestyle='solid', label=sublabels[i]) for i in range(len(limsion1)+1)]
    else:
        print('Warning: column density edges do not match for the different ions')
        handles_subs1, labels_subs1 = ax1.get_legend_handles_labels()     
        handles_subs2, labels_subs2 = ax2.get_legend_handles_labels()
        handles_subs = handles_subs1 + handles_subs2
        #labels_subs  = labels_subs1 + labels_subs2
    # set up legend in ax below main figure
    
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% encl.'%(100.*fraclevels[i])) for i in range(len(fraclevels))]    
    diffs_legend_handles = [mlines.Line2D([], [], color=color_diffs, linestyle = linestyle_diffs[i], label=labels_diffs[i]) for i in range(len(linestyle_diffs))]
    lax.legend(handles=handles_subs + level_legend_handles + diffs_legend_handles, fontsize=fontsize, ncol=ncols_legend, loc=legendloc, bbox_to_anchor=legend_bbox_to_anchor)
    lax.axis('off')
    
    #fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')

    plt.savefig(mdir + 'ion_diffs/%s.pdf'%(name), format = 'pdf',bbox_inches='tight')
    
    
def plot_cor_ne8o78_27_1sl(fontsize=12, slidemode =False):
    
    hist = ea3ne8o78
    ionlab = 'Ne\,VIII'
        
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(5.5,5.5))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1 = plt.subplot(grid[0,0], facecolor = 'white') 
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=18
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3 = plt.subplot(grid[0,0]) 
        ax1 = plt.subplot(grid[0,1], facecolor = 'white')
        ax2 = plt.subplot(grid[0,2])
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range (max  = 0.05 + max included in hist)
    #ax1.set_xlim(8.,16.25)
    #ax1.set_ylim(9.,17.25)
    ax1.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax1.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    

    # plot backgorund and colorbar: total distribution
    # plot backgorund and colorbar: total distribution
    img, vmin, vmax = add_2dplot(ax1,hist, (1,2),log=True, usepcolor = True, vmin = -8., cmap='bone_r')
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10}$ fraction of sightlines',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (r'$N_{%s}$'%ionlab,None,None)

    lims = [30,40,50,60,70]

    add_2dhist_contours(ax1,hist,(1,2),mins= (None,None,None), maxs=(lims[0], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[0], None, None), maxs=(lims[1], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[1], None, None), maxs=(lims[2], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[2], None, None), maxs=(lims[3], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[3], None, None), maxs=(lims[4], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax1,hist,(1,2),mins= (lims[4], None, None), maxs=(None, None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
     # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'black')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'coldens_o7-o8_by_ne8_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS.pdf' ,format = 'pdf',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_o7-o8_by_ne8_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_slide.png',format = 'png',bbox_inches='tight')


def plot_cor_ne8o78_27_1sl_simplified(fontsize=14, slidemode =False):
    
    hist = ea3ne8o78
    ionlab = 'Ne VIII'
    
    fig, ax  = plt.subplots(ncols=1,nrows=1)    

        
    if slidemode: # ax3 for legend right of plot
        fontsize=18
    else:
        pass

    # set up x-y extents from data range (max  = 0.05 + max included in hist)
    #ax1.set_xlim(8.,16.25)
    #ax1.set_ylim(9.,17.25)
    ax.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50] 
    linestyles = ['dashed','solid']
    dimlabels = (r'$N_{%s}$'%ionlab,None,None)

    lims = [30,40,50,60,70]

    add_2dhist_contours(ax,hist,(1,2),mins= (None,None,None), maxs=(lims[0], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[0], None, None), maxs=(lims[1], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[1], None, None), maxs=(lims[2], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['brown','brown','brown'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[2], None, None), maxs=(lims[3], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[3], None, None), maxs=(lims[4], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[4], None, None), maxs=(None, None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
     # square plot; set up axis 1 frame
    ax.set_xlim(11.,17.8)
    ax.set_ylim(11.,17.8)
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax.text(0.05,0.95,r'Ne VIII: $\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'black', transform=ax.transAxes) # bbox=dict(facecolor='white',alpha=0.3)
    ax.text(12.6,12.3,r'$< %.0f$'%(hist['edges'][0][lims[0]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'red')
    ax.text(13.3,13.0,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[0]], hist['edges'][0][lims[1]] ),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'orange')
    ax.text(14.4,14.0,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[1]], hist['edges'][0][lims[2]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'brown')
    ax.text(15.1,15.8,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[2]], hist['edges'][0][lims[3]]),fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', color = 'green')
    ax.text(16.5,16.4,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[3]], hist['edges'][0][lims[4]]),fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', color = 'blue')
    ax.text(17.5,17.0,r'$ >%.0f$'%(hist['edges'][0][lims[4]]),fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', color = 'purple') 
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'coldens_o7-o8_by_ne8_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_simplified.pdf' ,format = 'pdf',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_o7-o8_by_ne8_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_simpliefied_slide.png',format = 'png',bbox_inches='tight',dpi=300)




def plot_cor_o678_27_1sl_simplified(fontsize=14, slidemode =True):
    
    hist = ea3o678
    ionlab = 'O VI'
    
    fig, ax  = plt.subplots(ncols=1,nrows=1)    

        
    if slidemode: # ax3 for legend right of plot
        fontsize=18
    else:
        pass

    # set up x-y extents from data range (max  = 0.05 + max included in hist)
    #ax1.set_xlim(8.,16.25)
    #ax1.set_ylim(9.,17.25)
    ax.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
   
    # plot contour levels for column density subsets
    fraclevels = [0.99,0.50] 
    linestyles = ['dashed','solid']
    dimlabels = (r'$N_{%s}$'%ionlab,None,None)

    lims = [31,51,61,71,81]

    add_2dhist_contours(ax,hist,(1,2),mins= (None,None,None), maxs=(lims[0], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = None)			    
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[0], None, None), maxs=(lims[1], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[1], None, None), maxs=(lims[2], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['brown','brown','brown'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[2], None, None), maxs=(lims[3], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[3], None, None), maxs=(lims[4], None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = None)
    add_2dhist_contours(ax,hist,(1,2),mins= (lims[4], None, None), maxs=(None, None, None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'], dimlabels = dimlabels,\
			    legendlabel_pre = None)
	
	
     # square plot; set up axis 1 frame
    ax.set_xlim(11.,17.8)
    ax.set_ylim(11.,17.8)
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax.text(0.05,0.95,r'O VI: $\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'black', transform=ax.transAxes) # bbox=dict(facecolor='white',alpha=0.3)
    ax.text(11.7,11.8,r'$< %.0f$'%(hist['edges'][0][lims[0]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'red')
    ax.text(13.3,13.0,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[0]], hist['edges'][0][lims[1]] ),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'orange')
    ax.text(14.4,14.0,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[1]], hist['edges'][0][lims[2]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'brown')
    ax.text(15.1,15.8,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[2]], hist['edges'][0][lims[3]]),fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', color = 'green')
    ax.text(16.1,16.4,r'$%.0f$ - $%.0f$'%(hist['edges'][0][lims[3]], hist['edges'][0][lims[4]]),fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', color = 'blue')
    ax.text(16.5,14.7,r'$ >%.0f$'%(hist['edges'][0][lims[4]]),fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', color = 'purple') 
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    fig.tight_layout()
    #fig.suptitle('Ref-L0100N1504 $z= 0.0$, $6.25\, \mathrm{cMpc}$ sightlines')
    
    if not slidemode:	    
        plt.savefig(mdir + 'coldens_o7-o8_by_o6_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_simplified.pdf' ,format = 'pdf',bbox_inches='tight') 
    else:
        plt.savefig(mdir + 'coldens_o7-o8_by_o6_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_simpliefied_slide.png',format = 'png',bbox_inches='tight',dpi=300)
        

######## For Sowgat's proposal (2018)
def median_from_hist(hist1d, edges):
    if np.all(hist1d==0.):
        return np.NaN
    frac = 0.5
    cumuledges = edges[1:] # cumsum = number up to right edge
    cumul = np.cumsum(hist1d).astype(np.float)
    cumul /= float(cumul[-1]) # normalise by total
    if np.all(cumul > frac): # everything is in the first bin
        return 0.5*(cumuledges[0]+cumuledges[1])
    i0 = np.max(np.where(cumul <= frac)[0]) # last index leq
    i1 = np.min(np.where(cumul >= frac)[0]) # first index geq
    if cumul[i0] == cumul[i1]: # cumulative distribution plateaus exaclty at 0.5: i1 and i0 are plateau start and end indices OR i0 == i1, same edge element 
        w = 0.5 # we want the middle of the plateau
    else:
        w = (frac - cumul[i0])/(cumul[i1] - cumul[i0])
    return cumuledges[i0]*(1-w) + cumuledges[i1]*w
    
def plot_NNe8_vs_NO7():

    fileh = np.load('/net/luttero/data2/proc/hist_coldens_ne8-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.npz')
    bins = fileh['bins']
    edges = fileh['edges']
    toplot = np.log10(np.sum(bins, axis=2)/float(16*32000**2))
    
    fontsize = 16.
    cmap = 'cubehelix_r'
    vmin = -6.
    xlim = (14.5, 17.)
    ylim = (11.6, 15.5)
    specialval_o7 = np.log10(5.2e15) # Nicastro et al N_{O VII} for the lower-z system 
    centres_o7 = edges[1][:-1] + 0.5*np.diff(edges[1])
    # axis 0: Ne8, axis 1: O7

    # get median values for ne8 at each o7
    bins = 10**toplot
    cumul = np.cumsum(bins, axis=0)
    cumul = cumul / cumul[-1, :] # normalise to total in o7 bin
    median = np.array([median_from_hist(bins[:, o7i], edges[0]) for o7i in range(len(edges[1]) -1)])
    median_at_specialval_o7 = np.interp(specialval_o7, centres_o7, median)
    
    ploto7min = np.min(np.where(edges[1] >= xlim[0])[0]) - 1
    ploto7max = np.max(np.where(edges[1] <= xlim[1])[0]) + 1
    plotne8min = np.min(np.where(edges[0] >= ylim[0])[0]) - 1
    plotne8max = np.max(np.where(edges[0] <= ylim[1])[0]) + 1
    vmax = np.max(toplot[plotne8min:plotne8max, ploto7min:ploto7max])
    #print plotxmin, plotxmax, plotymin, plotymax, vmax
    #print np.max(toplot[plotymin:plotymax, plotxmin:plotxmax]), np.max(toplot[plotymin:plotymax, :]), np.max(toplot[:, plotxmin:plotxmax]), np.max(toplot[plotymin:, plotxmin:plotxmax]), np.max(toplot[:plotymax, plotxmin:plotxmax]), np.max(toplot[plotymin:plotymax, plotxmin:]), np.max(toplot[plotymin:plotymax, :plotxmax]), np.max(toplot)
    #print edges[1][plotxmin:plotxmax+1], edges[0][plotymin:plotymax+1]
    #return toplot[plotxmin:plotxmax, plotymin:plotymax]
    #plotxmin = 0
    #plotxmax = len(edges[1])
    #plotymin = 0
    #plotymax = len(edges[0])
    #vmax = np.max(toplot[plotxmin:plotxmax, plotymin:plotymax])
    
    fig, (ax, lax) = plt.subplots(ncols=2, nrows=1, gridspec_kw={'width_ratios':[10.,1.], 'wspace':0.})
    img = ax.pcolormesh(edges[1][ploto7min:ploto7max+1], edges[0][plotne8min:plotne8max+1], toplot[plotne8min:plotne8max, ploto7min:ploto7max], vmin=vmin, vmax=vmax, cmap=cmap)
    ax.plot(centres_o7, median, label='median', color='red', linestyle='solid', linewidth=3)
    ax.legend(fontsize=fontsize, loc='lower right')
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ymax = (median_at_specialval_o7 - ylim[0]) / (ylim[1] - ylim[0])
    xmax = (specialval_o7 - xlim[0]) / (xlim[1] - xlim[0])
    #print xmax, ymax, median_at_specialval_o7, ylim[1]-ylim[0], specialval_o7, (xlim[1]-xlim[0])
    ax.axhline(y=median_at_specialval_o7, xmin=0., xmax=xmax, color='red', linestyle='dashed', linewidth=3) 
    ax.axvline(x=specialval_o7, ymin=0., ymax=ymax, color='red', linestyle='dashed', linewidth=3) #
    ax.set_ylabel(r'$\log_{10} N({\mathrm{Ne \, VIII}}) \, [\mathrm{cm^{-2}}]$', fontsize=fontsize)
    ax.set_xlabel(r'$\log_{10} N({\mathrm{O \, VII}} \, [\mathrm{cm^{-2}}]$', fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',color='black')
    ax.text(0.05, 0.95, r'EAGLE $100\, \mathrm{Mpc}$ simulation', fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax.transAxes)

    add_colorbar(lax, img=img, clabel= r'$\log_{10}$ fraction of sightlines' ,newax=False, extend='min',fontsize=fontsize,orientation='vertical')
    lax.tick_params(labelsize=fontsize)
    lax.set_aspect(15.)
     
    plt.savefig(mdir + 'coldens_o7_vs_ne8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.pdf' ,format = 'pdf',bbox_inches='tight')
    return None

def getfracgeq(No7=None, Nne8=None):
    if No7 is None:
        No7 = 5.2e15
    if Nne8 is None:
        Nne8 = 10**13.4
    o7 = np.log10(No7)
    ne8 = np.log10(Nne8)
    
    fileh = np.load('/net/luttero/data2/proc/hist_coldens_ne8-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.npz')
    bins = fileh['bins']
    edges = fileh['edges']
    edges_ne8 = edges[0]
    edges_o7 = edges[1]
    ax_ne8 = 0
    ax_o7 = 1
    bins = np.sum(bins, axis=2) # sum over O8 column densities
    
    if hasattr(o7, '__len__'):
        i0 = np.max(np.where(edges_o7 <= o7[0])[0]) # last bin edge leq first column density
        i1 = np.min(np.where(edges_o7 >= o7[1])[0]) # first bin edge geq last column density
        if i0 +1 < i1: # upper and lower edges in different bins
            w0 = (edges_o7[i0+1] - o7[0])/(edges_o7[i0 + 1] - edges_o7[i0])  # fraction of bin to use (one if lower limit is a bin edge)
            w1 = (o7[1] - edges_o7[i1 - 1])/(edges_o7[i1] - edges_o7[i1 - 1]) 
            weights = np.array([w0] + list(np.ones(max(0, i1-i0-2))) + [w1]) # weigh fractions in different bins by bins size: evenly spaced, but get the right edge contributions
            subset_o7 = bins[:, i0:i1]*weights
            subset_o7 = np.sum(subset_o7, axis=ax_o7)
        elif i0 + 1 == i1: # upper and lower edges in same bin (bin i0)
            subset_o7 = bins[:, i0]
        elif i0 == i1: # upper and lower edges are the same
            print('The same upper and lower N_O7 are not a range')
            return None
        else:
            print('Unexpected case (o7 range)')
            return None
    else: # o7 is a scalar
        i0 = np.max(np.where(edges_o7 <= o7)[0]) # last index leq
        i1 = np.min(np.where(edges_o7 >= o7)[0])
        if i0 == i1: # o7 value is right on bin edge
            subset_o7 = bins[:, i0 - 1:i0 + 1] # take the average of the two adjacent bins
            subset_o7 = np.sum(subset_o7, axis=ax_o7)
        elif i0 +1 == i1: # o7 value falls into one bin -> just use that bin
            subset_o7 = bins[:, i0]
        else:
            print('Unexpected case (o7)')
            return None
    i0_ne8 = np.max(np.where(edges_ne8 <= ne8)[0]) # last index leq
    i1_ne8 = np.min(np.where(edges_ne8 >= ne8)[0])
    
    if i0_ne8 == i1_ne8: # falls on bin edge
        return float(np.sum(subset_o7[i0_ne8:]))/float(np.sum(subset_o7)) # fraction geq cutoff edge
    elif i0 +1 == i1: # o7 value falls into one bin -> intepolate fraction in that bin
        add_edgebin = float(subset_o7[i0_ne8]) * (ne8 - edges_ne8[i0_ne8]) / (edges_ne8[i1_ne8] - edges_ne8[i0_ne8])
        return (float(np.sum(subset_o7[i1_ne8:])) + add_edgebin)/float(np.sum(subset_o7)) # fraction geq cutoff edge
    else:
        print('Unexpected case (ne8)')
        return None
    
    
            
            
            
    
    
def check_ionrat_ciediff():
    z = 0.355
    o7_ib, logTKo7_ib, lognHcm3o7_ib = m3.findiontables('o7',z)
    ne8_ib, logTKne8_ib, lognHcm3ne8_ib = m3.findiontables('ne8',z)
    # axis 0: nH, axis1: T
    if not (np.all(logTKo7_ib == logTKne8_ib) and np.all(lognHcm3o7_ib == lognHcm3ne8_ib )):
        print('T or nH arrays do not match')
        return None
    else:
        logTK = logTKo7_ib
        lognH = lognHcm3o7_ib
    il_o7 = 'O\, VII'
    il_ne8 = 'Ne\, VIII'
    fontsize = 12
    
    nHi_cie = len(lognH) - 1
    nH_cie = lognH[nHi_cie]
    nHi_nem5 = np.argmin(np.abs(lognH - (-5.))) # approximate Nicastro et al. ne = 1e-5 with nH = 1e-5
    nH_nem5 = lognH[nHi_nem5]
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(9., 4.), gridspec_kw={'wspace': 0.3})
    fig.suptitle(r'Comparison of the Ne VIII to O VII abundance ratios at fixed $[\mathrm{Ne}/\mathrm{O}]$', fontsize=fontsize)
 
    ax1.set_xlabel(r'$\log_{10} T\,[\mathrm{K}]$', fontsize=fontsize)
    ax1.set_ylabel(r'$ \log_{10}$ ion fraction', fontsize=fontsize)
    ax1.text(0.05, 0.95, r'HM01 UV/X-ray bkg at $z=0.355$', fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes)
    ax1.text(0.05, 0.87, r'ion fractions: $\mathrm{%s}/ \mathrm{Ne}$, $\mathrm{%s} / \mathrm{O}$'%(il_ne8, il_o7), fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes)
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',color='black')
    ax1.set_xlim(5., 7.5)   
    ax1.set_ylim(-11.9, 2.8)
     
    ax1.plot(logTK, np.log10(ne8_ib[nHi_cie, :]), label=r'$\mathrm{%s}, CIE: n_H = 10^{%.1f}\, \mathrm{cm}^{-3}$'%(il_ne8, nH_cie))
    ax1.plot(logTK, np.log10(o7_ib[nHi_cie, :]), label=r'$\mathrm{%s}, CIE: n_H = 10^{%.1f}\, \mathrm{cm}^{-3}$'%(il_o7, nH_cie))
    ax1.plot(logTK, np.log10(ne8_ib[nHi_nem5, :]), label=r'$\mathrm{%s}, n_H = 10^{%.1f}\, \mathrm{cm}^{-3}$'%(il_ne8, nH_nem5)) 
    ax1.plot(logTK, np.log10(o7_ib[nHi_nem5, : ]), label=r'$\mathrm{%s}, n_H = 10^{%.1f}\, \mathrm{cm}^{-3}$'%(il_o7, nH_nem5))
 
    ax1.legend(fontsize=fontsize, loc='lower right')
    
    ax2.set_xlabel(r'$\log_{10} T\,[\mathrm{K}]$', fontsize=fontsize)
    ax2.set_ylabel(r'$ \log_{10} \mathrm{%s} / \mathrm{%s}$'%(il_ne8, il_o7), fontsize=fontsize)
    ax2.text(0.05, 0.95, r'HM01 UV/X-ray bkg at $z=0.355$', fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax2.transAxes)
    ax2.text(0.05, 0.87, r'assuming $N(\mathrm{Ne}) =  N(\mathrm{O})$', fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax2.transAxes)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',color='black')
    ax2.set_xlim(5., 7.5)  
    ax2.set_ylim(-3.9, 0.6)
    
    ax2.plot(logTK, np.log10(ne8_ib[nHi_cie, :]/o7_ib[nHi_cie, :]), label=r'CIE: $n_H = 10^{%.1f}\, \mathrm{cm}^{-3}$'%(nH_cie))
    ax2.plot(logTK, np.log10(ne8_ib[nHi_nem5, :]/o7_ib[nHi_nem5, : ]), label=r'$n_H = 10^{%.1f}\, \mathrm{cm}^{-3}$'%(nH_nem5))

    ylim = ax2.get_ylim()
    ax2.set_ylim(-4., ylim[1])    
    ax2.legend(fontsize=fontsize, loc='lower right')
    
    plt.savefig(mdir + 'HSTproposal_impact_of_nH_on_predicted_Ne8_to_O7_ratios.pdf',format = 'pdf',bbox_inches='tight')
    
###############################################################################
##### O6 vs O8 absorption properties: Alexis Finoguenov and Jussi Ahoranta ####
###############################################################################    

def gaussian2(mux, sigmax, muy, sigmay, xs, ys):
    return  1. / (2. * np.pi * sigmax * sigmay) * np.exp( -1 * (xs - mux)**2 / (2. * sigmax**2) - (ys - muy)**2 / (2. * sigmay**2))

def gaussian(mux, sigmax, xs):
    return  1. / ((2. * np.pi)**0.5 * sigmax) * np.exp(-1 * (xs - mux)**2 / (2. * sigmax**2))

def add_2dhist_contours_simple(ax, hist, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=None, legendlabel=None,\
                        shiftx=0., shifty=0., **kwargs):
    '''
    colors, linestyles: through kwargs
    toplotaxes: order determines whether axes are transposed or not
    '''
    # transpose axes if required
    binsum = np.copy(hist)
    if toplotaxes[0] > toplotaxes[1]:
        binsum = binsum.T
        edges = [edges[1], edges[0]]
         
    if levels is None:
        if fraclevels:
            levels = [1., 0.9, 0.5] # enclosed fractions for each level (approximate)
        else:
	        levels = [1e-3, 3e-2, 0.1, 0.5]

    if fraclevels: # assumes all levels are between 0 and 1
        binsum = binsum/np.sum(binsum) # redo normalisation for the smaller dataset
        #print('min/max binsum: %.4e, %.4e'%(np.min(binsum),np.max(binsum)))
        
        # for sorting, normialise bins by bin size: peak finding depends on density, should not favour larger bins
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(edges[0]), np.diff(edges[1])] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])
        
        binsumcopy = binsum.copy() # copy to rework
        bindens    = binsumcopy/normmatrix
        bindensflat= bindens.copy().reshape(np.prod(bindens.shape)) # reshape creates views; argsorting later will mess up the array we need for the plot
        binsumcopy = binsumcopy.reshape(np.prod(binsumcopy.shape))
        binsumcopy = binsumcopy[np.argsort(bindensflat)] # get all histogram values in order of histogram density (low to high)
        
        binsumcopy = np.flipud(binsumcopy) # flip to high-to-low
        cumul = np.cumsum(binsumcopy) # add values high-to-low 
        wherelist = [[(np.where(cumul<=level))[0],(np.where(cumul>=level))[0]] for level in levels] # list of max-lower and min-higher indices

        ### made for using bin counts -> binsumcopy is ordered y its own values
	    # sort out list: where arrays may be empty -> levels outside 0,1 range, probabaly
	    # set value level 0 for level == 1. -> just want everything (may have no cumulative values that large due to fp errors)
	    # if all cumulative values are too high (maxmimum bin has too high a fraction), set to first cumulative value (=max bin value)
	    # otherwise: interpolate values, or use overlap
        if np.all(normmatrix == normmatrix[0,0]): # all bins are the same size
            valslist = [cumul[0]  if  wherelist[i][0].shape == (0,) else\
	                    0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		                np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
                                               np.array([ binsumcopy[wherelist[i][0][-1]], binsumcopy[wherelist[i][1][0]] ]) )[0]\
		                for i in range(len(levels))]
            pltarr = binsum
        else: # find a reasonable interpolation of bindens in stead; need to plot the contours in binsdens as well, in this case
            bindensflat.sort() # to match cumul array indices: sort, then make high to low
            bindensflat = bindensflat[::-1]
            valslist = [bindensflat[0]  if  wherelist[i][0].shape == (0,) else\
	                    0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		                np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
		                                                 np.array([ bindensflat[wherelist[i][0][-1]], bindensflat[wherelist[i][1][0]] ]))[0]\
		                for i in range(len(levels))]
            pltarr = bindens        
        del normmatrix
        del binsumcopy
        del binsum
        del bindens
        del bindensflat
        #for i in range(len(levels)):
        #    if not (wherelist[i][0].shape == (0,) or wherelist[i][1].shape == (0,)):
	    #        print('interpolating (%f, %f) <- index %i and (%f, %f)  <- index %i to %f'\
	    #	 %(cumul[wherelist[i][0][-1]],binsumcopy[wherelist[i][0][-1]],wherelist[i][0][-1],\
	    #          cumul[wherelist[i][1][0]], binsumcopy[wherelist[i][1][0]], wherelist[i][1][0],\
    	#	   levels[i]) )
        #print(np.all(np.diff(binsumcopy)>=0.))
        uselevels = np.copy(valslist)
        # check for double values; fudge slightly if levels are the same
        anyequal = np.array([np.array(valslist) == val for val in valslist])
        if np.sum(anyequal) > len(valslist): # any levels equal to a *different* level
            eqvals = [np.where(anyequal[ind])[0] for ind in range(len(valslist))] # what levels any value is equal to
            eqgroups = set([tuple(list(eq)) for eq in eqvals]) # get the sets of unique values
            eqgroups = list(eqgroups)
            fudgeby = 1.e-8
            grouplist = [(np.where(np.array([ind in group for group in eqgroups]))[0])[0] for ind in range(len(valslist))] # which group is each uselevel index in
            groupindlist = [(np.where(ind == np.array(eqgroups[grouplist[ind]]))[0])[0] for ind in range(len(valslist))] # which group index corresponds to a goven uselevel index
            addto = [[valslist[group[0]]*fudgeby*ind for ind in range(len(group))] for group in eqgroups] #add nothing for single-element groups
                            
            valslist = [uselevels[ind] + addto[grouplist[ind]][groupindlist[ind]] for ind in range(len(valslist))]
            print('Desired cumulative fraction levels were %s; using value levels %s fudged from %s'%(levels, valslist, uselevels))
            uselevels = valslist
        else:
            print('Desired cumulative fraction levels were %s; using value levels %s'%(levels,uselevels))
    else:
        uselevels=levels
    
    removezerolevelprops = False
    if uselevels[0] == uselevels[1]:
        uselevels = uselevels[1:]
        removezerolevelprops = True
            
    #print binsum, binsum.shape
    if 'linestyles' in kwargs:        
        linestyles = kwargs['linestyles']
    else:
        linestyles = [] # to not break the legend search
    
    if removezerolevelprops: # a duplicate level was kicked out -> remove properties for that level
        if 'linestyles' in kwargs.keys():
            kwargs['linestyles'] = kwargs['linestyles'][1:]
        if 'colors' in kwargs.keys():
            kwargs['colors'] = kwargs['colors'][1:]
            
    # get pixel centres from edges
    centres0 = edges[0][:-1] + shiftx + 0.5*np.diff(edges[0]) 
    centres1 = edges[1][:-1] + shifty + 0.5*np.diff(edges[1])
    contours = ax.contour(centres0, centres1, pltarr.T, uselevels, **kwargs)
    # make a legend to avoid crowding plot region
    #for i in range(len(levels)):
    #    contours.collections[i].set_label('%.0e'%levels[i])
    # color only legend; get a solid line in the legend
    
    #ax.tick_params(labelsize=fontsize,axis='both')
    if 'solid' in linestyles:
        contours.collections[np.where(np.array(linestyles)=='solid')[0][0]].set_label(legendlabel)
    else: # just do the first one
        contours.collections[0].set_label(legendlabel)

def subplot_absorbercomps_o68(ax, hist, edges, P, vmin=None, vmax=None, title=None, fontsize=12, xlabels=True, ylabels=True):
    xlim = (-6.7, -1.)
    ylim = xlim
    xlabel = (r'$\log_{10}\, n_H\; [\mathrm{cm}^{-3}], \mathrm{O\, VI}$')
    ylabel = (r'$\log_{10}\, n_H\; [\mathrm{cm}^{-3}], \mathrm{O\, VIII}$')
    cmap = 'gist_gray_r'
    levels = [0.999, 0.99, 0.90, 0.50]
    
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=xlabels, labelleft=ylabels, labelright=False, labeltop=False, color='black')
    ax.minorticks_on()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabels:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_aspect('equal')
    
    img = ax.pcolormesh(edges[0], edges[1], np.log10(hist.T), vmin=vmin, vmax=vmax, cmap=cmap)
    add_2dhist_contours_simple(ax, hist, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=0., shifty=0., colors=('magenta',)*4, linestyles=['dotted', 'dashed', 'dashdot', 'solid'])
    ax.plot([edges[0][0], edges[0][-1]], [edges[0][0], edges[0][-1]], color='dodgerblue')
    ax.plot([edges[0][0], edges[0][-1]], [edges[0][0] - 1., edges[0][-1] - 1.], color='dodgerblue', linestyle='dotted')
    ax.text(0.95, 0.05, r'$P = $%.1e'%(P), fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    return img

def getPmaxdiff(hist, edges, logdiff):
    '''
    P(log nH, O VI - log nH, O VIII >= logdiff) 
    assumes edges match and logdiff is an integer multiple of the edge spacing
    '''
    epsilon = 1.e-2
    
    edgeso6 = edges[0][:-1] + 0.5 * np.diff(edges[0])
    edgeso8 = edges[1][:-1] + 0.5 * np.diff(edges[1])
    diffs = edgeso6[:, np.newaxis] - edgeso8[np.newaxis, :]
    w1 = np.where(diffs >= logdiff + epsilon)
    wh = np.where(np.abs(diffs - logdiff) <= epsilon)
    P = np.sum(hist[w1]) + 0.5 * np.sum(hist[wh]) 
    return P

def plotabsorbercomps_o68():
    mu_logNo8 =15.5 
    sigma_logNo8 = 0.3 
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    measrhodiff = 1.
    
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = mdir + 'ion_diffs/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_matched_to_obs.pdf'
    
    hist = earhoo68_27 # dimensions: ['NO6', 'NO8', 'Density_w_NO6', 'Density_w_NO8']
    edges_No6 = hist['edges'][0]
    edges_No8 = hist['edges'][1]
    edges_rhoNo6 = hist['edges'][2]
    edges_rhoNo8 = hist['edges'][3]
    edges_rhoNo6 += np.log10(rho_to_nh)
    edges_rhoNo8 += np.log10(rho_to_nh)
    histvals = hist['bins']
    
    # 1 sigma `bucket':
    minmaxinds_o6_b1s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + sigma_logNo6)))]
    minmaxinds_o8_b1s = [np.argmin(np.abs(edges_No8 - (mu_logNo8 - sigma_logNo8))), np.argmin(np.abs(edges_No8 - (mu_logNo8 + sigma_logNo8)))]
    
    minmaxinds_o6_b2s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 2 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 2 * sigma_logNo6)))]
    minmaxinds_o8_b2s = [np.argmin(np.abs(edges_No8 - (mu_logNo8 - 2 * sigma_logNo8))), np.argmin(np.abs(edges_No8 - (mu_logNo8 + 2 * sigma_logNo8)))]
    
    minmaxinds_o6_b5s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 5 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 5 * sigma_logNo6)))]
    minmaxinds_o8_b5s = [np.argmin(np.abs(edges_No8 - (mu_logNo8 - 5 * sigma_logNo8))), np.argmin(np.abs(edges_No8 - (mu_logNo8 + 5 * sigma_logNo8)))]
    
    #print minmaxinds_o6_b1s, minmaxinds_o8_b1s, minmaxinds_o6_b2s, minmaxinds_o8_b2s, minmaxinds_o6_b5s, minmaxinds_o8_b5s
    
    centres_No6 = edges_No6[:-1] + 0.5 * np.diff(edges_No6)
    centres_No8 = edges_No8[:-1] + 0.5 * np.diff(edges_No8)
    
    minmax_o6_b1s = [edges_No6[minmaxinds_o6_b1s[0]], edges_No6[minmaxinds_o6_b1s[1]]] 
    minmax_o8_b1s = [edges_No6[minmaxinds_o8_b1s[0]], edges_No6[minmaxinds_o8_b1s[1]]] 
    minmax_o6_b2s = [edges_No6[minmaxinds_o6_b2s[0]], edges_No6[minmaxinds_o6_b2s[1]]] 
    minmax_o8_b2s = [edges_No6[minmaxinds_o8_b2s[0]], edges_No6[minmaxinds_o8_b2s[1]]] 
    
    #print minmax_o6_b1s, minmax_o8_b1s, minmax_o6_b2s, minmax_o8_b2s
    
    hist_b1s = np.sum(histvals[slice(minmaxinds_o6_b1s[0], minmaxinds_o6_b1s[1], None), slice(minmaxinds_o8_b1s[0], minmaxinds_o8_b1s[1], None), :, :], axis=(0, 1))
    hist_b2s = np.sum(histvals[slice(minmaxinds_o6_b2s[0], minmaxinds_o6_b2s[1], None), slice(minmaxinds_o8_b2s[0], minmaxinds_o8_b2s[1], None), :, :], axis=(0, 1))
    print np.sum(hist_b1s), np.sum(hist_b2s)
    hist_b1s /= np.sum(hist_b1s)
    hist_b2s /= np.sum(hist_b2s)
    
    hist_b5sel = histvals[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None), slice(minmaxinds_o8_b5s[0], minmaxinds_o8_b5s[1], None), :, :]
    print np.sum(hist_b5sel)
    gaussweights = gaussian2(mu_logNo6, sigma_logNo6, mu_logNo8, sigma_logNo8, centres_No6[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None), np.newaxis], centres_No8[np.newaxis, slice(minmaxinds_o8_b5s[0], minmaxinds_o8_b5s[1], None)])
    
    hist_bay = np.sum(hist_b5sel * gaussweights[:, :, np.newaxis, np.newaxis], axis=(0,1))
    hist_bay /= np.sum(hist_bay)
    hist_gss = np.copy(hist_b5sel)
    hist_gss /= np.sum(hist_gss, axis=(2, 3))[:, :, np.newaxis, np.newaxis] # divide out column density weighting -> 'flat weighting'
    hist_gss[np.isnan(hist_gss)] = 0. # deal with 0./0. cases 
    hist_gss *= gaussweights[:, :, np.newaxis, np.newaxis]
    hist_gss = np.sum(hist_gss, axis=(0, 1))
    hist_gss /= np.sum(hist_gss)
    
    
    fig = plt.figure(figsize=(5.5, 5.5))
    grid = gsp.GridSpec(2, 2, width_ratios=[1., 1.], wspace=0.10, hspace=0.15, top=0.95, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[yi,xi]) for yi in range(2)] for xi in range(2)]) # in mainaxes: x = column, y = row
    
    title_b1s = r'$\pm 1 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$,'%(minmax_o6_b1s[0], minmax_o6_b1s[1]) + '\n' + r'$N_{\mathrm{O\, VIII}}:  %.1f \endash %.1f$'%(minmax_o8_b1s[0], minmax_o8_b1s[1])
    title_b2s = r'$\pm 2 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$,'%(minmax_o6_b2s[0], minmax_o6_b2s[1]) + '\n' + r'$N_{\mathrm{O\, VIII}}:  %.1f \endash %.1f$'%(minmax_o8_b2s[0], minmax_o8_b2s[1])
    title_bay = r'$\pm 5 \sigma$, CDDF $\times$ gaussian'
    title_gss = r'$\pm 5 \sigma$, gaussian'
    
    P_b1s = getPmaxdiff(hist_b1s, [edges_rhoNo6, edges_rhoNo8], measrhodiff)
    P_b2s = getPmaxdiff(hist_b2s, [edges_rhoNo6, edges_rhoNo8], measrhodiff)
    P_bay = getPmaxdiff(hist_bay, [edges_rhoNo6, edges_rhoNo8], measrhodiff)
    P_gss = getPmaxdiff(hist_gss, [edges_rhoNo6, edges_rhoNo8], measrhodiff)

    subplot_absorbercomps_o68(mainaxes[0, 0], hist_b1s, [edges_rhoNo6, edges_rhoNo8], P_b1s, vmin=vmin, vmax=vmax, title=title_b1s, fontsize=fontsize, xlabels=False)
    subplot_absorbercomps_o68(mainaxes[1, 0], hist_b2s, [edges_rhoNo6, edges_rhoNo8], P_b2s, vmin=vmin, vmax=vmax, title=title_b2s, fontsize=fontsize, xlabels=False, ylabels=False)
    subplot_absorbercomps_o68(mainaxes[0, 1], hist_bay, [edges_rhoNo6, edges_rhoNo8], P_bay, vmin=vmin, vmax=vmax, title=title_bay, fontsize=fontsize)
    subplot_absorbercomps_o68(mainaxes[1, 1], hist_gss, [edges_rhoNo6, edges_rhoNo8], P_gss, vmin=vmin, vmax=vmax, title=title_gss, fontsize=fontsize, ylabels=False)
    
    plt.savefig(name, format='pdf', bbox_inches='tight')

def subplot_absorbercomps_o8ne9(ax, hist, edges, xyerr, histc=None, vmin=None, vmax=None, title=None, fontsize=12, xlabels=True, ylabels=True, cmap=None, takeloghist=True):
    xlim = (11.7, 17.)
    ylim = (10.5, 16.5)
    xlabel = (r'$\log_{10}\, N_\mathrm{O\, VIII} \; [\mathrm{cm}^{-2}]$')
    ylabel = (r'$\log_{10}\, N_\mathrm{Ne\, IX} \; [\mathrm{cm}^{-2}]$')
    if cmap is None:
        cmap = 'gist_gray_r'
    if histc is None:
        histc = hist
    levels = [0.999, 0.99, 0.90, 0.50]
    
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=xlabels, labelleft=ylabels, labelright=False, labeltop=False, color='black')
    ax.minorticks_on()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabels:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_aspect('equal')
    
    if takeloghist:
        hist = np.log10(hist)
    hist = np.ma.masked_where(np.isnan(hist), hist)
    img = ax.pcolormesh(edges[0], edges[1], hist.T, vmin=vmin, vmax=vmax, cmap=cmap)
    add_2dhist_contours_simple(ax, histc, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=0., shifty=0., colors=('magenta',)*len(levels), linestyles=['dotted', 'dashed', 'dashdot', 'solid'])
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0], edges[0][-1]], color='dodgerblue')
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0] - 1., edges[0][-1] - 1.], color='dodgerblue', linestyle='dotted')
    ax.errorbar([xyerr[0]], xyerr[1], yerr=xyerr[3], xerr=xyerr[2], fmt='.', color='black', label='obs.', zorder=5, capsize=2)
    return img
  

def plotabsorbercomps_o678ne9():
    hist = eao678ne9_16
    
    mu_logNo8 = 15.5 
    sigma_logNo8 = 0.3 
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    mu_logNne9 = 15.4
    sigma_logNne9 = 0.3
    ul_logNo7 = 14.9
    
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = mdir + 'ion_diffs/hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_for_observations_comparison.pdf'
    
    edges_No6 = hist['edges'][0]
    edges_No7 = hist['edges'][1]
    edges_No8 = hist['edges'][2]
    edges_Nne9 = hist['edges'][3]
    histvals = hist['bins']
    
    # 1 sigma `bucket':
    minmaxinds_o6_b1s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + sigma_logNo6)))]    
    minmaxinds_o6_b2s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 2 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 2 * sigma_logNo6)))]
    minmaxinds_o6_b5s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 5 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 5 * sigma_logNo6)))]
    maxind_o7 = np.argmin(np.abs(edges_No7 - ul_logNo7))
    
    #print minmaxinds_o6_b1s, minmaxinds_o8_b1s, minmaxinds_o6_b2s, minmaxinds_o8_b2s, minmaxinds_o6_b5s, minmaxinds_o8_b5s
    
    centres_No6 = edges_No6[:-1] + 0.5 * np.diff(edges_No6)
    centres_No8 = edges_No8[:-1] + 0.5 * np.diff(edges_No8)
    centres_Nne9 = edges_Nne9[:-1] + 0.5 * np.diff(edges_Nne9)
        
    minmax_o6_b1s = [edges_No6[minmaxinds_o6_b1s[0]], edges_No6[minmaxinds_o6_b1s[1]]] 
    minmax_o6_b2s = [edges_No6[minmaxinds_o6_b2s[0]], edges_No6[minmaxinds_o6_b2s[1]]] 
    max_o7 = edges_No7[maxind_o7]
    #print minmax_o6_b1s, minmax_o8_b1s, minmax_o6_b2s, minmax_o8_b2s
    
    hist_b1s = np.sum(histvals[slice(minmaxinds_o6_b1s[0], minmaxinds_o6_b1s[1], None), ...], axis=(0, 1))
    hist_b2s = np.sum(histvals[slice(minmaxinds_o6_b2s[0], minmaxinds_o6_b2s[1], None), ...], axis=(0, 1))
    hist_b1s_ulo7 = np.sum(histvals[slice(minmaxinds_o6_b1s[0], minmaxinds_o6_b1s[1], None), slice(None, maxind_o7, None), ...], axis=(0, 1))
    hist_b2s_ulo7 = np.sum(histvals[slice(minmaxinds_o6_b2s[0], minmaxinds_o6_b2s[1], None), slice(None, maxind_o7, None), ...], axis=(0, 1))
    
    print np.sum(hist_b1s), np.sum(hist_b2s), np.sum(hist_b1s_ulo7), np.sum(hist_b2s_ulo7)
    hist_b1s /= np.sum(hist_b1s)
    hist_b2s /= np.sum(hist_b2s)
    
    hist_b5sel = histvals[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None), :, :, :]
    hist_b5sel_ulo7 = histvals[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None), :, :, :]
    print np.sum(hist_b5sel), np.sum(hist_b5sel_ulo7)
    gaussweights = gaussian(mu_logNo6, sigma_logNo6, centres_No6[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None)])
    
    hist_bay = np.sum(hist_b5sel * gaussweights[:, np.newaxis, np.newaxis, np.newaxis], axis=(0, 1))
    hist_bay /= np.sum(hist_bay)
    hist_gss = np.copy(hist_b5sel)
    hist_gss /= np.sum(hist_gss, axis=(2, 3))[:, :, np.newaxis, np.newaxis] # divide out column density weighting -> 'flat weighting'
    hist_gss[np.isnan(hist_gss)] = 0. # deal with 0./0. cases 
    hist_gss *= gaussweights[:, np.newaxis, np.newaxis, np.newaxis]
    hist_gss = np.sum(hist_gss, axis=(0, 1))
    hist_gss /= np.sum(hist_gss)
    
    
    fig = plt.figure(figsize=(5.5, 8.))
    grid = gsp.GridSpec(3, 2, width_ratios=[1., 1.], wspace=0.10, hspace=0.25, top=0.95, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[yi,xi]) for yi in range(3)] for xi in range(2)]) # in mainaxes: x = column, y = row
    
    title_b1s = r'$\pm 1 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$'%(minmax_o6_b1s[0], minmax_o6_b1s[1])
    title_b2s = r'$\pm 2 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$'%(minmax_o6_b2s[0], minmax_o6_b2s[1])
    title_b1s_ulo7 = r'$\pm 1 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$,'%(minmax_o6_b1s[0], minmax_o6_b1s[1]) + '\n' + r'$N_{\mathrm{O\, VII}} < %.1f$'%(max_o7)
    title_b2s_ulo7 = r'$\pm 2 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$,'%(minmax_o6_b2s[0], minmax_o6_b2s[1]) + '\n' + r'$N_{\mathrm{O\, VII}} < %.1f$'%(max_o7)
    title_bay = r'$N_{\mathrm{O\, VI}} \pm 5 \sigma$, CDDF $\times$ gaussian'
    title_gss = r'$N_{\mathrm{O\, VI}} \pm 5 \sigma$, gaussian'
    
    subplot_absorbercomps_o8ne9(mainaxes[0, 0], hist_b1s, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title=title_b1s,\
                                fontsize=fontsize, xlabels=False)
    subplot_absorbercomps_o8ne9(mainaxes[1, 0], hist_b2s, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title=title_b2s,\
                                fontsize=fontsize, xlabels=False, ylabels=False)
    subplot_absorbercomps_o8ne9(mainaxes[0, 1], hist_b1s_ulo7, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title=title_b1s_ulo7,\
                                fontsize=fontsize, xlabels=False)
    subplot_absorbercomps_o8ne9(mainaxes[1, 1], hist_b2s_ulo7, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title=title_b2s_ulo7,\
                                fontsize=fontsize, xlabels=False, ylabels=False)
    subplot_absorbercomps_o8ne9(mainaxes[0, 2], hist_bay, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title=title_bay,\
                                fontsize=fontsize)
    subplot_absorbercomps_o8ne9(mainaxes[1, 2], hist_gss, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title=title_gss,\
                                fontsize=fontsize, ylabels=False)
    
    plt.savefig(name, format='pdf', bbox_inches='tight')
    

def plotabsorbercomps_o678ne9_withONefractions(histtype='o6-2sigma', ionfrac='oxygen-over-neon', percentiles=(0.05, 0.5, 0.95)):
    '''
    histtype: 'o6-1sigma', 'o6-2sigma', 'o6-1sigma_o7-ul', 
              'o6-1sigma_o7-ul', 'o6-cddf-w-gauss', 'o6-gauss'
    ionfrac:  'oxygen', 'neon', 'oxygen-over-neon'
    '''
    hist = eao678ne9wfONe_16
    
    mu_logNo8 = 15.5 
    sigma_logNo8 = 0.3 
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    mu_logNne9 = 15.4
    sigma_logNne9 = 0.3
    ul_logNo7 = 14.9
    
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = mdir + 'ion_diffs/hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and weighted_massfrac_ONe_Pt_%s_%s.pdf'%(ionfrac, histtype)
    
    edges_No6 = hist['edges'][0]
    edges_No7 = hist['edges'][1]
    edges_No8 = hist['edges'][2]
    edges_Nne9 = hist['edges'][3]
    edges_fObyo8 = hist['edges'][4]
    edges_fNebyNe9 = hist['edges'][5]
    histvals = hist['bins']
    
    # 1 sigma `bucket':
    minmaxinds_o6_b1s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + sigma_logNo6)))]    
    minmaxinds_o6_b2s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 2 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 2 * sigma_logNo6)))]
    minmaxinds_o6_b5s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 5 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 5 * sigma_logNo6)))]
    maxind_o7 = np.argmin(np.abs(edges_No7 - ul_logNo7))
    
    #print minmaxinds_o6_b1s, minmaxinds_o8_b1s, minmaxinds_o6_b2s, minmaxinds_o8_b2s, minmaxinds_o6_b5s, minmaxinds_o8_b5s
    
    centres_No6 = edges_No6[:-1] + 0.5 * np.diff(edges_No6)
    centres_No8 = edges_No8[:-1] + 0.5 * np.diff(edges_No8)
    centres_Nne9 = edges_Nne9[:-1] + 0.5 * np.diff(edges_Nne9)
        
    minmax_o6_b1s = [edges_No6[minmaxinds_o6_b1s[0]], edges_No6[minmaxinds_o6_b1s[1]]] 
    minmax_o6_b2s = [edges_No6[minmaxinds_o6_b2s[0]], edges_No6[minmaxinds_o6_b2s[1]]] 
    max_o7 = edges_No7[maxind_o7]
    #print minmax_o6_b1s, minmax_o8_b1s, minmax_o6_b2s, minmax_o8_b2s
    
    ### basic histograms: number of absorbers in various column density selections
    sel = list((slice(None, None, None),)*len(hist['edges']))
    if 'o6-1sigma' in histtype:
        sel[0] = slice(minmaxinds_o6_b1s[0], minmaxinds_o6_b1s[1], None)
    elif 'o6-2sigma' in histtype:
        sel[0] = slice(minmaxinds_o6_b2s[0], minmaxinds_o6_b2s[1], None)
    if 'o7-ul' in histtype:
        sel[1] = slice(None, maxind_o7, None)
        
    if 'gauss' not in histtype:
        abshist = np.sum(histvals[tuple(sel)], axis=(0, 1, 4, 5))
        stemp = np.sum(abshist)
        print np.min(abshist), np.max(abshist)
        print('Number of included absorbers: %i'%stemp)
        abshist /= stemp
        weights = 1.
    
    elif histtype in ['o6-cddf-w-gauss', 'o6-gauss']:
        sel[0] = slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None)
        hist_b5sel = np.sum(histvals[tuple(sel)], axis=(4,5))
        #hist_b5sel_ulo7 = np.sum(histvals[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None), ...], axis=(4,5))
        gaussweights = gaussian(mu_logNo6, sigma_logNo6, centres_No6[slice(minmaxinds_o6_b5s[0], minmaxinds_o6_b5s[1], None)])
        
        if histtype == 'o6-cddf-w-gauss':
            weights = gaussweights[:, np.newaxis, np.newaxis, np.newaxis]
            hist_bay = np.sum(hist_b5sel * weights, axis=(0, 1))
            hist_bay /= np.sum(hist_bay)
            abshist = hist_bay
        elif histtype == 'o6-gauss':
            hist_gss = np.copy(hist_b5sel)
            weights = 1. / np.sum(hist_gss, axis=(2, 3))[:, :, np.newaxis, np.newaxis] # divide out column density weighting for o8, ne9 -> 'flat weighting'
            weights[np.isinfinite(weights)] = 0. # deal with 1./0. cases 
            weights *= gaussweights[:, np.newaxis, np.newaxis, np.newaxis]
            abshist = np.sum(hist_gss * weights, axis=(0, 1))
            abshist /= np.sum(abshist)
    
    if ionfrac == 'oxygen':
        ionax = 4
        solar = ol.solar_abunds[ionfrac]
        clabel = r'$\log_{10}\, f_{\mathrm{O}} \; [\mathrm{\odot}]$'
    elif ionfrac == 'neon':
        ionax = 5
        solar = ol.solar_abunds[ionfrac]
        clabel = r'$\log_{10}\, f_{\mathrm{Ne}} \; [\mathrm{\odot}]$'
        
    if ionfrac in ['oxygen', 'neon']:
        sumaxes = [0, 1, 4, 5] # display axes are 2 and 3, ignore the others
        sumaxes.remove(ionax)
        percentiles = np.array(percentiles)
        cmapp = 'viridis'
        
        dists  = np.sum(histvals[sel] * weights, axis=tuple(sumaxes)) # abshist, except not summed over the element abundance axes
        edges  = hist['edges'][ionax]
        
        totals = np.sum(dists, axis=2)
        dists /= totals[:, :, np.newaxis] # distributions normalised to one
        cdists = np.cumsum(dists, axis=2) # will end at 1 absent fp errors, start at contribution of the first pixel (should typically be zero at the measured column densities)
        #plt.step(hist['edges'][ionax][:-1], cdists[20, 20, :], where='post')
        #plt.show()

        valinds = \
        [[ np.searchsorted(cdists[xi, yi], percentiles)
            for yi in range(cdists.shape[1])]\
            for xi in range(cdists.shape[0])]
        valinds = np.array(valinds)
        valinds = np.array([valinds, valinds + 1])
        valinds[valinds > cdists.shape[2]] = cdists.shape[2]  # percentiles beyond last bin (fp error) -> just use last edges value
        
        #print 'edges: ', edges.shape, '  cdists: ', cdists.shape, '  valinds: ', valinds.shape, valinds.dtype
        interpweights = \
        [[[  (edges[valinds[1, xi, yi, pi]] - cdists[xi, yi][max(valinds[0, xi, yi, pi], cdists.shape[2] - 1)] ) \
           / (edges[valinds[1, xi, yi, pi]] - edges[valinds[0, xi, yi, pi]]) \
           for pi in range(len(percentiles))] \
           for yi in range(cdists.shape[1])]\
           for xi in range(cdists.shape[0])]
        interpweights = np.array(interpweights)
        interpweights[np.isinf(interpweights)] = 1.
        percentilevals = valinds[0].astype(np.float)
        percentilevals = edges[valinds[0]]  #* interpweights + edges[valinds[1]] * (1. - interpweights)
        percentilevals[abshist <= 0.] = np.NaN # mask values with no absorbers: currently set to mimum value
    
        percentilevals -= np.log10(solar)
        pmin = np.min(percentilevals[np.isfinite(percentilevals)])
        pmax = np.max(percentilevals[np.isfinite(percentilevals)])
        print pmin, pmax
    
    elif ionfrac == 'oxygen-over-neon':
        oxax = 4
        neax = 5
        solar = ol.solar_abunds['oxygen'] / ol.solar_abunds['neon']
        clabel = r'$\log_{10}\, \mathrm{O} \,/\, \mathrm{Ne} \; [\mathrm{\odot}]$'
        cmapp = 'coolwarm'
        
        sumaxes = [0, 1, 4, 5] # display axes are 2 and 3, ignore the others
        sumaxes.remove(oxax)
        sumaxes.remove(neax)
        percentiles = np.array(percentiles)
        
        dists  = np.sum(histvals[sel] * weights, axis=tuple(sumaxes)) # abshist, except not summed over the element abundance axes
        oxedges = hist['edges'][oxax]
        needges = hist['edges'][neax]
        
        # construct oxygen / neon binning from oxygen and neon bins
        oxcens = oxedges[:-1] + 0.5 * np.diff(oxedges)
        necens = needges[:-1] + 0.5 * np.diff(needges)
        oxovernevals = oxcens[:, np.newaxis] - necens[np.newaxis, :]
        if oxax > neax:
            oxovernevals = oxovernevals.T
        oxovernevals_forbins = list(set(np.round(oxovernevals.flatten(), 1)))
        oxovernevals_forbins.sort()
        if np.min(oxovernevals) < np.min(oxovernevals_forbins):
            oxovernevals_forbins = [np.round(np.min(oxovernevals) - 0.1, 1)] + oxovernevals_forbins
        if np.max(oxovernevals) > np.max(oxovernevals_forbins):
            oxovernevals_forbins =  oxovernevals_forbins + [np.round(np.max(oxovernevals) + 0.1, 1)]
        oxovernebins =  [1.5 * oxovernevals_forbins[0] - 0.5 * oxovernevals_forbins[1]] +\
                        list(oxovernevals_forbins[1:] - 0.5 * np.diff(oxovernevals_forbins)) + \
                        [1.5 * oxovernevals_forbins[-1] - 0.5 * oxovernevals_forbins[-2]]
        oxovernebins = np.array(oxovernebins)
        #print np.diff(oxovernevals_forbins), np.diff(oxovernebins) 
        oxovernebinmatch = np.digitize(oxovernevals, oxovernebins) - 1 # searchsorted-type results: index 1 means between edges 0 and 1 -> bin 0
        
        # bin ox, ne 2d slices into 1D ratio bins
        udists = [[[ np.sum(dists[xi, yi][np.where(oxovernebinmatch == bi)])\
                 for bi in range(len(oxovernebins) - 1)]\
                 for yi in range(dists.shape[1])]\
                 for xi in range(dists.shape[0])]
        udists = np.array(udists)
        edges = oxovernebins
        
        # now treat udists the same as the o8, ne8, element arrays for the O, Ne mass fraction plots
        totals = np.sum(udists, axis=2)
        udists /= totals[:, :, np.newaxis] # distributions normalised to one
        cdists = np.cumsum(udists, axis=2) # will end at 1 absent fp errors, start at contribution of the first pixel (should typically be zero at the measured column densities)
        #plt.step(hist['edges'][ionax][:-1], cdists[20, 20, :], where='post')
        #plt.show()

        valinds = \
        [[ np.searchsorted(cdists[xi, yi], percentiles)
            for yi in range(cdists.shape[1])]\
            for xi in range(cdists.shape[0])]
        valinds = np.array(valinds)
        valinds = np.array([valinds, valinds + 1])
        valinds[valinds > cdists.shape[2]] = cdists.shape[2]  # percentiles beyond last bin (fp error) -> just use last edges value
        
        interpweights = \
        [[[  (edges[valinds[1, xi, yi, pi]] - cdists[xi, yi][max(valinds[0, xi, yi, pi], cdists.shape[2] - 1)] ) \
           / (edges[valinds[1, xi, yi, pi]] - edges[valinds[0, xi, yi, pi]]) \
           for pi in range(len(percentiles))] \
           for yi in range(cdists.shape[1])]\
           for xi in range(cdists.shape[0])]
        interpweights = np.array(interpweights)
        interpweights[np.isinf(interpweights)] = 1.
        percentilevals = valinds[0].astype(np.float)
        percentilevals = edges[valinds[0]]  #* interpweights + edges[valinds[1]] * (1. - interpweights)
        percentilevals[abshist <= 0.] = np.NaN # mask values with no absorbers: currently set to mimum value
        
        percentilevals -= np.log10(solar)
        pmin = np.min(percentilevals[np.isfinite(percentilevals)])
        pmax = np.max(percentilevals[np.isfinite(percentilevals)])
        pmax = max(np.abs(pmin), np.abs(pmax))
        pmin = -1. * pmax
        print pmin, pmax
    
    
    ### oxygen/neon plots: (weighted) percentiles for each element in different histograms 
    
    
    fig = plt.figure(figsize=(5.5, 5.5))
    grid = gsp.GridSpec(2, 3, width_ratios=[1., 1., 0.3], wspace=0.10, hspace=0.15, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[yi,xi]) for yi in range(2)] for xi in range(2)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:, 2])
                  
    if histtype == 'o6-1sigma':
        title = r'$\pm 1 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$'%(minmax_o6_b1s[0], minmax_o6_b1s[1])
    elif histtype == 'o6-2sigma':
        title = r'$\pm 2 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$'%(minmax_o6_b2s[0], minmax_o6_b2s[1])
    elif histtype == 'o6-1sigma_o7-ul':
        title = r'$\pm 1 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$,'%(minmax_o6_b1s[0], minmax_o6_b1s[1]) + '  ' + r'$N_{\mathrm{O\, VII}} < %.1f$'%(max_o7)
    elif histtype ==  'o6-2sigma_o7-ul':
        title = '$\pm 2 \sigma$; $N_{\mathrm{O\, VI}}: %.1f \endash %.1f$,'%(minmax_o6_b2s[0], minmax_o6_b2s[1]) + '  ' + r'$N_{\mathrm{O\, VII}} < %.1f$'%(max_o7)
    elif histtype == 'o6-cddf-w-gauss':
        title = '$N_{\mathrm{O\, VI}} \pm 5 \sigma$, CDDF $\times$ gaussian'
    elif histtype == 'o6-gauss':
        title = r'$N_{\mathrm{O\, VI}} \pm 5 \sigma$, gaussian'
    
    subplot_absorbercomps_o8ne9(mainaxes[0, 0], abshist, [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=vmin, vmax=vmax, title='absorbers',\
                                fontsize=fontsize, xlabels=False)
    img = subplot_absorbercomps_o8ne9(mainaxes[1, 0], percentilevals[:, :, 0], [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=pmin, vmax=pmax, title='percentile %.3f'%percentiles[0],\
                                cmap=cmapp, histc=abshist, takeloghist=False,\
                                fontsize=fontsize, xlabels=False, ylabels=False)
    subplot_absorbercomps_o8ne9(mainaxes[0, 1], percentilevals[:, :, 1], [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=pmin, vmax=pmax, title='percentile %.3f'%percentiles[1],\
                                cmap=cmapp, histc=abshist, takeloghist=False,\
                                fontsize=fontsize, xlabels=True)
    subplot_absorbercomps_o8ne9(mainaxes[1, 1], percentilevals[:, :, 2], [edges_No8, edges_Nne9],\
                                [mu_logNo8, mu_logNne9, sigma_logNo8, sigma_logNne9],\
                                vmin=pmin, vmax=pmax, title='percentile %.3f'%percentiles[2],\
                                cmap=cmapp, histc=abshist, takeloghist=False,\
                                fontsize=fontsize, xlabels=True, ylabels=False)

    add_colorbar(cax, img=img ,vmin=pmin, vmax=pmax, cmap=cmapp,\
                 clabel=clabel, newax=False, extend='neither',\
                 fontsize=fontsize, orientation='vertical')
    plt.suptitle(title, fontsize=fontsize)
    
    plt.savefig(name, format='pdf', bbox_inches='tight')
    
###############################################################################
#### Paper versions of plots with Jussi & Alexis' O8/Ne9 counterpart to O6 ####
###############################################################################
def subplot_coldenscomps_o8ne9(ax, hist, edges, histc=None,\
                                vmin=None, vmax=None, title=None, fontsize=12,\
                                levels=(0.999, 0.99, 0.90, 0.50),\
                                colors=None, linestyles=None,\
                                xlabels=True, ylabels=True, cmap=None,\
                                takeloghist=True, plotslab=True, plotcie=True):
    mu_logNo8_slab = 15.5 
    sigma_logNo8_slab = 0.2 
    mu_logNne9_slab = 15.4
    sigma_logNne9_slab_up = 0.1
    sigma_logNne9_slab_down = 0.2
    
    mu_logNo8_cie = 15.4
    sigma_logNo8_cie = 0.2
    mu_logNne9_cie = 14.9
    sigma_logNne9_cie = 0.2

    xlim = (11.7, 17.)
    ylim = (11., 16.5)
    xlabel = (r'$\log_{10}\, N(\mathrm{O\, VIII}) \; [\mathrm{cm}^{-2}]$')
    ylabel = (r'$\log_{10}\, N(\mathrm{Ne\, IX}) \; [\mathrm{cm}^{-2}]$')
    if cmap is None:
        cmap = gray_m
    if histc is None:
        histc = hist
    
    ax.tick_params(labelsize=fontsize - 1, direction='in', right=True, top=True, axis='both', which='both', labelbottom=xlabels, labelleft=ylabels, labelright=False, labeltop=False, color='black')
    ax.minorticks_on()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabels:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_aspect('equal')
    
    if takeloghist:
        hist = np.log10(hist)
    hist = np.ma.masked_where(np.isnan(hist), hist)
    img = ax.pcolormesh(edges[0], edges[1], hist.T, vmin=vmin, vmax=vmax, cmap=cmap)
    add_2dhist_contours_simple(ax, histc, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=levels, linestyles=linestyles, colors=colors, legendlabel=None,\
                        shiftx=0., shifty=0.)
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0], edges[0][-1]], color='dodgerblue')
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0] - 1., edges[0][-1] - 1.], color='dodgerblue', linestyle='dotted')
    if plotslab:
        ax.errorbar([mu_logNo8_slab], [mu_logNne9_slab], yerr=([sigma_logNne9_slab_down], [sigma_logNne9_slab_up]), xerr=[sigma_logNo8_slab], fmt='.', color='black', label='slab', zorder=5, capsize=3, elinewidth=2, capthick=2)
    if plotcie:
        ax.errorbar([mu_logNo8_cie], [mu_logNne9_cie], yerr=[sigma_logNne9_cie], xerr=[sigma_logNo8_cie], fmt='.', color='blue', label='CIE', zorder=5, capsize=3, elinewidth=2, capthick=2)
    return img


def subplot_Tcomps_o8ne9(ax, hist, edges, histc=None,\
                         vmin=None, vmax=None, title=None, fontsize=12,\
                         levels=(0.999, 0.99, 0.90, 0.50),\
                         colors=None, linestyles=None,\
                         xlabels=True, ylabels=True, cmap=None,\
                         takeloghist=True):

    xlim = (3., 8.)
    ylim = (3., 8.)
    xlabel = (r'$\log_{10}\, T(\mathrm{O\, VI}) \; [\mathrm{K}]$')
    ylabel = (r'$\log_{10}\, T(\mathrm{O\, VIII}) \; [\mathrm{K}]$')
    if cmap is None:
        cmap = gray_m
    if histc is None:
        histc = hist
    
    ax.tick_params(labelsize=fontsize - 1, direction='in', right=True, top=True, axis='both', which='both', labelbottom=xlabels, labelleft=ylabels, labelright=False, labeltop=False, color='black')
    ax.minorticks_on()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if xlabels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabels:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    ax.set_aspect('equal')
    
    if takeloghist:
        hist = np.log10(hist)
    hist = np.ma.masked_where(np.isnan(hist), hist)
    img = ax.pcolormesh(edges[0], edges[1], hist.T, vmin=vmin, vmax=vmax, cmap=cmap)
    add_2dhist_contours_simple(ax, histc, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=levels, linestyles=linestyles, colors=colors, legendlabel=None,\
                        shiftx=0., shifty=0.)
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0], edges[0][-1]], color='dodgerblue')
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0] - 1., edges[0][-1] - 1.], color='dodgerblue', linestyle='dotted')
    return img

def subplot_PD_o8ne9(ax, hist, edges, histc=None,\
                         vmin=None, vmax=None, title=None, fontsize=12,\
                         levels=(0.999, 0.99, 0.90, 0.50),\
                         colors=None, linestyles=None,\
                         xlabels=True, ylabels=True, cmap=None,\
                         takeloghist=True):

    xlabel = (r'$\log_{10}\, n_{\mathrm{H}}(\mathrm{O\, VI}) \; [\mathrm{K}]$')
    ylabel = (r'$\log_{10}\, T(\mathrm{O\, VI}) \; [\mathrm{K}]$')
    if cmap is None:
        cmap = gray_m
    if histc is None:
        histc = hist
    
    ax.tick_params(labelsize=fontsize - 1, direction='in', right=True, top=True, axis='both', which='both', labelbottom=xlabels, labelleft=ylabels, labelright=False, labeltop=False, color='black')
    ax.minorticks_on()
    if xlabels:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabels:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    #ax.set_aspect('equal')
    
    if takeloghist:
        hist = np.log10(hist)
    hist = np.ma.masked_where(np.isnan(hist), hist)
    img = ax.pcolormesh(edges[0], edges[1], hist.T, vmin=vmin, vmax=vmax, cmap=cmap)
    add_2dhist_contours_simple(ax, histc, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=levels, linestyles=linestyles, colors=colors, legendlabel=None,\
                        shiftx=0., shifty=0.)
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0], edges[0][-1]], color='dodgerblue')
    #ax.plot([edges[0][0], edges[0][-1]], [edges[0][0] - 1., edges[0][-1] - 1.], color='dodgerblue', linestyle='dotted')
    return img

def plotabsorbercomps_o678ne9corr():
    '''
    
    '''
    hist = eao678ne9_16
    
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    
    mu_logNo8_slab = 15.5 
    sigma_logNo8_slab = 0.2 
    mu_logNne9_slab = 15.4
    sigma_logNne9_slab_up = 0.1
    sigma_logNne9_slab_down = 0.2
    ul_logNo7_slab = 15.5
    
    mu_logNo8_cie = 15.4
    sigma_logNo8_cie = 0.2
    mu_logNne9_cie = 14.9
    sigma_logNne9_cie = 0.2
    mu_logNo7_cie = 14.8
    sigma_logNo7_cie = 0.2
    
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = '/home/wijers/Documents/papers/jussi_alexis_o8ne9_counterpart_to_o6/' + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3x_PtAb_C2Sm_32000pix_6p25slice_zcen-all_z-projection_T4EOS.eps'
    
    edges_No6 = hist['edges'][0]
    edges_No7 = hist['edges'][1]
    edges_No8 = hist['edges'][2]
    edges_Nne9 = hist['edges'][3]
    histvals = hist['bins']
    
    # 1 sigma `bucket':
    minmaxinds_o6_b1s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + sigma_logNo6)))]    
    minmaxinds_o6_b2s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 2 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 2 * sigma_logNo6)))]
    #minmaxinds_o6_b5s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 5 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 5 * sigma_logNo6)))]
    maxind_o7slab = np.argmin(np.abs(edges_No7 - ul_logNo7_slab))
    minmaxinds_o7cie_b1s = [np.argmin(np.abs(edges_No7 - (mu_logNo7_cie - sigma_logNo7_cie))),     np.argmin(np.abs(edges_No7 - (mu_logNo7_cie + sigma_logNo7_cie)))]    
    minmaxinds_o7cie_b2s = [np.argmin(np.abs(edges_No7 - (mu_logNo7_cie - 2 * sigma_logNo7_cie))), np.argmin(np.abs(edges_No7 - (mu_logNo7_cie + 2 * sigma_logNo7_cie)))]
    #minmaxinds_o7cie_b5s = [np.argmin(np.abs(edges_No7 - (mu_logNo7_cie - 5 * sigma_logNo7_cie))), np.argmin(np.abs(edges_No7 - (mu_logNo7_cie + 5 * sigma_logNo7_cie)))]
    
    #print minmaxinds_o6_b1s, minmaxinds_o8_b1s, minmaxinds_o6_b2s, minmaxinds_o8_b2s, minmaxinds_o6_b5s, minmaxinds_o8_b5s
    
    centres_No6 = edges_No6[:-1] + 0.5 * np.diff(edges_No6)
    centres_No8 = edges_No8[:-1] + 0.5 * np.diff(edges_No8)
    centres_Nne9 = edges_Nne9[:-1] + 0.5 * np.diff(edges_Nne9)
        
    minmax_o6_b1s = [edges_No6[minmaxinds_o6_b1s[0]], edges_No6[minmaxinds_o6_b1s[1]]] 
    minmax_o6_b2s = [edges_No6[minmaxinds_o6_b2s[0]], edges_No6[minmaxinds_o6_b2s[1]]] 
    max_o7slab = edges_No7[maxind_o7slab]
    minmax_o7cie_b1s = [edges_No7[minmaxinds_o7cie_b1s[0]], edges_No6[minmaxinds_o7cie_b1s[1]]] 
    minmax_o7cie_b2s = [edges_No7[minmaxinds_o7cie_b2s[0]], edges_No6[minmaxinds_o7cie_b2s[1]]] 
    #print minmax_o6_b1s, minmax_o8_b1s, minmax_o6_b2s, minmax_o8_b2s
    
    ### basic histograms: number of absorbers in various column density selections
    #sel = list((slice(None, None, None),)*len(hist['edges']))
    o6sels = ['o6_1sigma', 'o6_2sigma']
    o7sels = ['o7none', 'o7slab_ul', 'o7cie_1sigma', 'o7cie_2sigma']
    sels_1ax = {'o6_1sigma': slice(minmaxinds_o6_b1s[0], minmaxinds_o6_b1s[1], None),\
                'o6_2sigma': slice(minmaxinds_o6_b2s[0], minmaxinds_o6_b2s[1], None),\
                'o7none': slice(None, None, None),\
                'o7slab_ul': slice(None, maxind_o7slab, None),\
                'o7cie_1sigma': slice(minmaxinds_o7cie_b1s[0], minmaxinds_o7cie_b1s[1], None),\
                'o7cie_2sigma': slice(minmaxinds_o7cie_b2s[0], minmaxinds_o7cie_b2s[1], None),\
                }
    titles_1ax = {'o6_1sigma': r'$\mathrm{O\, VI}: %.1f \endash %.1f$'%(minmax_o6_b1s[0], minmax_o6_b1s[1]),\
                  'o6_2sigma': r'$\mathrm{O\, VI}: %.1f \endash %.1f$'%(minmax_o6_b2s[0], minmax_o6_b2s[1]),\
                  'o7none': '',\
                  'o7slab_ul': r'$\mathrm{O\, VII} < %.1f$'%(max_o7slab),\
                  'o7cie_1sigma': r'$\mathrm{O\, VII}: %.1f \endash %.1f$'%(minmax_o7cie_b1s[0], minmax_o7cie_b1s[1]),\
                  'o7cie_2sigma': r'$\mathrm{O\, VII}: %.1f \endash %.1f$'%(minmax_o7cie_b2s[0], minmax_o7cie_b2s[1]),\
                 }
    
    normhists = [[ np.sum(histvals[sels_1ax[o6sel], sels_1ax[o7sel], ...], axis=(0, 1)) \
                  /(np.diff(edges_No8)[:, np.newaxis] * np.diff(edges_Nne9)[np.newaxis, :])  for o7sel in o7sels] for o6sel in o6sels]
    clabel = r'$\log_{10} \left( f_{\mathrm{absorbers}} \,/\, (\Delta \log_{10}N)^2 \right)$'
    vmax = np.log10(max([np.max(hi) for his in normhists for hi in his]))
    vmin = vmax - 8.
    
    fig = plt.figure(figsize=(5.5, 13.))
    grid = gsp.GridSpec(5, 2, height_ratios=[1., 1., 1., 1., 0.3], wspace=0.0, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[yi, xi]) for yi in range(4)] for xi in range(2)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[4, 0])
    lax = fig.add_subplot(grid[4, 1])               
    
    edges =[edges_No8, edges_Nne9]
    levels = (0.999, 0.99, 0.90, 0.50)
    colors=  ['red', 'orange', 'green', 'purple'] #('orange',)*len(levels)
    linestyles= ('solid',) * len(levels) # ['dotted', 'dashed', 'dashdot', 'solid']
    for i in range(len(o6sels) * len(o7sels)):
        xi = i % len(o6sels)
        yi = i // len(o6sels)
        ax = mainaxes[xi, yi]
        o6sel = o6sels[xi]
        o7sel = o7sels[yi]
        bins = normhists[xi][yi]
        title = '\n'.join([titles_1ax[o6sel], titles_1ax[o7sel]])
        ax.text(0.95, 0.05, title, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
        img = subplot_coldenscomps_o8ne9(ax, bins, edges, histc=None,\
                                vmin=vmin, vmax=vmax, title=None, fontsize=12,\
                                levels=levels, colors=colors, linestyles=linestyles,\
                                ylabels=(xi == 0), xlabels=(yi == len(o7sels) - 1),\
                                cmap=None,\
                                takeloghist=True, plotslab=True, plotcie=True)
    

    add_colorbar(cax, img=img ,vmin=vmin, vmax=vmax,\
                 clabel=clabel, newax=False, extend='min',\
                 fontsize=fontsize, orientation='horizontal')
    cax.tick_params(labelsize=fontsize - 1, axis='both', which='both')
    cax.set_aspect(1./10.)
    lax.axis('off')
    handles_subs, labels_subs = mainaxes[0][0].get_legend_handles_labels()
    #level_legend_handles = [mlines.Line2D([], [], color='red', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    level_legend_handles = [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], label='%.2f%% enclosed'%(100.*levels[i])) for i in range(len(levels))]
    lax.legend(handles=handles_subs + level_legend_handles, fontsize=fontsize, ncol=2, loc='upper center', bbox_to_anchor=(0.7, 0.5))
    
    plt.savefig(name, format='eps', bbox_inches='tight')
    


def getlims(edges, cen, sigma=None, numsig=1.):
    if sigma is None: # dummy selection
        return (None, None)
    elif sigma == 'ul': # upper limit
        maxval = np.argmin(np.abs(edges - cen))
        return (None, maxval)
    elif sigma == 'll': # lower limit
        minval = np.argmin(np.abs(edges - cen))
        return (minval, None)
    elif hasattr(sigma, '__len__'): #tuple : sigma -, sigma +
        minval = np.argmin(np.abs(edges - (cen - numsig * sigma[0])))
        maxval = np.argmin(np.abs(edges - (cen + numsig * sigma[1])))
        return (minval, maxval)
    else:
        minval = np.argmin(np.abs(edges - (cen - numsig * sigma)))
        maxval = np.argmin(np.abs(edges - (cen + numsig * sigma)))
        return (minval, maxval)

def plotTcomps_o678ne9corr(sigma=2., model='cie', seltype='1ion'):
    '''
    paperversion: ignores input sigma, model
    '''
    

    if seltype == '2ion':
        selectionstoshow = [[('No6', sigma), ('No7%s'%model, sigma)], [('No6', sigma), ('No8%s'%model, sigma)],  [('No6', sigma), ('Nne9%s'%model, sigma)],\
                            [('No7%s'%model, sigma), ('No8%s'%model, sigma)], [('No7%s'%model, sigma), ('Nne9%s'%model, sigma)], [('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 3)
        sellabel = '%s-%isigma-2ion'%(model, sigma)
    elif seltype == '1ion':
        selectionstoshow = [[('No6', sigma)], [('No7%s'%model, sigma)], [('No8%s'%model, sigma)],  [('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-1ion'%(model, sigma)
    elif seltype == '3ion':
        selectionstoshow = [[('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma)], [('No6', sigma), ('No7%s'%model, sigma), ('Nne9%s'%model, sigma)],\
                            [('No6', sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)], [('No7%s'%model, sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-3ion'%(model, sigma)
    elif seltype == 'addions':
        selectionstoshow = [[('No6', sigma)], [('No6', sigma), ('No7%s'%model, sigma)], [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma)],\
                            [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-addions'%(model, sigma)
    elif seltype == 'paperversion':
        sigma = 1.
        model = 'slab'
        selectionstoshow = [[('No6', sigma)], [('No6', sigma), ('No7%s'%model, sigma)], [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma)],\
                            [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-addions_paperversion'%(model, sigma)
    
    
    hist = eao678ne9wTo68_16
    handleinfedges(hist, setmin=-51., setmax=50.)
    
    o6ax  = 0
    o7ax  = 1
    o8ax  = 2
    ne9ax = 3
    To6ax = 4
    To8ax = 5
    
    edges_No6 = hist['edges'][o6ax]
    edges_No7 = hist['edges'][o7ax]
    edges_No8 = hist['edges'][o8ax]
    edges_Nne9 = hist['edges'][ne9ax]
    edges_To8 = hist['edges'][To8ax]
    edges_To6 = hist['edges'][To6ax]
    histvals = hist['bins']
    
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    
    mu_logNo8_slab = 15.5 
    sigma_logNo8_slab = 0.2 
    mu_logNne9_slab = 15.4
    sigma_logNne9_slab_up = 0.1
    sigma_logNne9_slab_down = 0.2
    ul_logNo7_slab = 15.5
    
    mu_logNo8_cie = 15.4
    sigma_logNo8_cie = 0.2
    mu_logNne9_cie = 14.9
    sigma_logNne9_cie = 0.2
    mu_logNo7_cie = 14.8
    sigma_logNo7_cie = 0.2
    
    mu_T_hot = 6.48
    sigma_T_hot = 0.05
    mu_T_warm = 5.51
    sigma_T_warm_up =  0.24
    sigma_T_warm_down =   0.24
    
    measvals = {'No6':      [mu_logNo6, sigma_logNo6],\
                'No7slab':  [ul_logNo7_slab, 'ul'],\
                'No7cie':   [mu_logNo7_cie,   sigma_logNo7_cie],\
                'No8slab':  [mu_logNo8_slab,  sigma_logNo8_slab],\
                'No8cie':   [mu_logNo8_cie,   sigma_logNo8_cie],\
                'Nne9slab': [mu_logNne9_slab, (sigma_logNne9_slab_down, sigma_logNne9_slab_up)],\
                'Nne9cie':  [mu_logNne9_cie,  sigma_logNne9_cie],\
                }
    edges =    {'No6': edges_No6,\
                'No7': edges_No7,\
                'No8': edges_No8,\
                'Nne9': edges_Nne9,\
                }
    seltoion = {'No6':      'No6',\
                'No7slab':  'No7',\
                'No7cie':   'No7',\
                'No8slab':  'No8',\
                'No8cie':   'No8',\
                'Nne9slab': 'Nne9',\
                'Nne9cie':  'Nne9',\
                }
    Nlabels = {'No6':  '\mathrm{O\,VI}',\
               'No7':  '\mathrm{O\,VII}',\
               'No8':  '\mathrm{O\,VIII}',\
               'Nne9': '\mathrm{Ne\,IX}',\
               }
    axids = {  'No6':  o6ax,\
               'No7':  o7ax,\
               'No8':  o8ax,\
               'Nne9': ne9ax,\
               }
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = '/home/wijers/Documents/papers/jussi_alexis_o8ne9_counterpart_to_o6/' + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3x_PtAb_C2Sm_32000pix_6p25slice_zcen-all_z-projection_T4EOS_To6-To8_%s.eps'%sellabel
    
   
    lims = {(key, numsig): getlims(edges[seltoion[key]], measvals[key][0], sigma=measvals[key][1], numsig=numsig)\
            for key in measvals.keys() for numsig in [1., 2.]}
    
    
    sels_1ax = {key: slice(lims[key][0], lims[key][1], None) for key in lims.keys()}
    
    titles_1ax = {key: r'$%s: %.1f \endash %.1f$'%(Nlabels[seltoion[key[0]]],\
                                                  edges[seltoion[key[0]]][lims[key][0]],\
                                                  edges[seltoion[key[0]]][lims[key][1]])\
                            if lims[key][0] is not None and lims[key][1] is not None else\
                       r'$%s: < %.1f$'%(Nlabels[seltoion[key[0]]],\
                                       edges[seltoion[key[0]]][lims[key][1]])\
                            if lims[key][0] is None and lims[key][1] is not None else\
                       r'$%s: > %.1f$'%(Nlabels[seltoion[key[0]]],\
                                       edges[seltoion[key[0]]][lims[key][0]])\
                            if lims[key][0] is not None and lims[key][1] is None else\
                       ''\
                 for key in lims.keys()}
    
    
    titles = ['\n'.join([titles_1ax[key] for key in panelsel]) for panelsel in selectionstoshow]
    selections = []
    for selection in selectionstoshow:
        sel = list((slice(None, None, None),) * 6)
        for key in selection:
            sel[axids[seltoion[key[0]]]] = sels_1ax[key]
        selections.append(sel)
        
    normhists = [np.sum(histvals[tuple(sel)], axis=(o6ax, o7ax, o8ax, ne9ax)) \
                      /(np.diff(edges_To8)[:, np.newaxis] * np.diff(edges_To6)[:, np.newaxis]) \
                 for sel in selections]
    
    clabel = r'$\log_{10} \left( f_{\mathrm{abs.}} \,/\, (\Delta \log_{10}T)^2 \right)$'
    xlabel = (r'$\log_{10}\, T(\mathrm{O\, VI}) \; [\mathrm{K}]$')
    ylabel = (r'$\log_{10}\, T(\mathrm{O\, VIII}) \; [\mathrm{K}]$')
    vmax = np.log10(max([np.max(hi) for his in normhists for hi in his]))
    vmin = vmax - 8.
    
    fig = plt.figure(figsize=(2.5*shape[1] + 0.3, 2.5*shape[0] + 0.3))
    grid = gsp.GridSpec(shape[0] + 1, shape[1] + 1,\
                        height_ratios=list((1.,) * shape[0]) + [0.3],\
                        width_ratios=list((1.,) * shape[1]) + [0.3],\
                        wspace=0.0, hspace=0.0, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    labelax = fig.add_subplot(grid[:shape[0], :shape[1]])
    mainaxes = np.array([[fig.add_subplot(grid[xi, yi]) for yi in range(shape[1])] for xi in range(shape[0])]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:shape[0], shape[1]])
    lax = fig.add_subplot(grid[shape[0], :])               
    
    
    levels = (0.999, 0.99, 0.90, 0.50)
    colors=  ['firebrick', 'darkorange', 'yellowgreen', 'paleturquoise'] #('orange',)*len(levels)
    linestyles= ('solid',) * len(levels) # ['dotted', 'dashed', 'dashdot', 'solid']
    for i in range(np.prod(shape)):
        xi = i % shape[1]
        yi = i // shape[1]
        ax = mainaxes[yi, xi]
        bins = normhists[i][2:-1, 2:-1]
        edges =[edges_To6[2:-1], edges_To8[2:-1]]
        print('x: %f -- %f, y: %f -- %f'%(edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]))
        title = titles[i]
        ax.set_xlim(3., 8.)    
        ax.set_ylim(3., 8.) 
        ax.text(0.95, 0.05, title, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes) #, bbox=dict(facecolor='white',alpha=0.3)
        img = subplot_Tcomps_o8ne9(ax, bins, edges, histc=None,\
                         vmin=vmin, vmax=vmax, title=None, fontsize=fontsize,\
                         levels=levels,\
                         colors=colors, linestyles=linestyles,\
                         xlabels=(yi == shape[0] - 1), ylabels=(xi==0), cmap=None,\
                         takeloghist=True)
        ax.errorbar([mu_T_warm], [mu_T_hot], xerr=([sigma_T_warm_down], [sigma_T_warm_up]), yerr=[sigma_T_hot], fmt='.', color='blue', label='2T CIE', zorder=5, capsize=3, elinewidth=2, capthick=2)
        #if yi == 0:
        #    ax.set_ylabel('$\log_{10}\, T_{\mathrm{O\,VIII}} \; [\mathrm{K}]$', fontsize=fontsize)
        #if xi == shape[1] - 1:
        #    ax.set_xlabel('$\log_{10}\, T_{\mathrm{O\,VI}} \; [\mathrm{K}]$', fontsize=fontsize)
        ax.set_xlim(3.1, 8.)    
        ax.set_ylim(3.1, 8.)  
        ax.plot([0., 10.], [0., 10.], color='brown', linestyle='dashed')
        ax.set_xlabel('')
        ax.set_ylabel('')
     # hide tick and tick label of the big axes
    labelax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    labelax.grid(False)
    labelax.spines['top'].set_color('none')
    labelax.spines['bottom'].set_color('none')
    labelax.spines['left'].set_color('none')
    labelax.spines['right'].set_color('none')
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    
    add_colorbar(cax, img=img ,vmin=vmin, vmax=vmax,\
                 clabel=clabel, newax=False, extend='min',\
                 fontsize=fontsize, orientation='vertical')
    cax.tick_params(labelsize=fontsize - 1, axis='both', which='both')
    cax.set_aspect(12.)
    lax.axis('off')
    handles_subs, labels_subs = mainaxes[0][0].get_legend_handles_labels()
    #level_legend_handles = [mlines.Line2D([], [], color='red', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    level_legend_handles = [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], label='%.1f%% enclosed'%(100.*levels[i])) for i in range(len(levels))]
    lax.legend(handles=level_legend_handles + handles_subs, fontsize=fontsize, ncol=shape[1] + 1, loc='upper center', bbox_to_anchor=(0.5, 0.2))
    
    plt.savefig(name, format='eps', bbox_inches='tight')
    

def ploto6pd_o678ne9corr(sigma=2., model='cie', seltype='1ion', zsel=None, sellabel=None, xlim=None, ylim=None, addciediff=False):
    '''
    paperversion: ignores input sigma, model, zsel
    
    seltype: '1ion', '2ion', '3ion', 'addions': use with model, sigma
             or custom selection (used in all panels), list of ion-model, sigma tuples 
                (main sigma and model keywords are ignored in this case)
             or list of lists of custom selections: applied to different panels in row and columns
    zsel: (min, max) tuple (set one to None for just upper/lower limits) or list of those 
    '''
    insellabel = sellabel
    
    if seltype is None: # selection will be fully set by zsel
        selectionstoshow = []
        shape = (None, None)
        sellabel = ''
    elif isinstance(seltype, list): # selection is a list of list of lists of (<ion>[model], sigma) tuples; to be combined with different zsels in different panels
        if isinstance(seltype[0], list): # selection is given: outermost layer of lists: varies row, then column, then list of selection tuples per panel
            selectionstoshow = [sel for row in seltype for sel in row]
            shape = (len(seltype[0]), len(seltype))
            sellabel = sellabel # naming the selection in every panel will only be confusing
        else: # list contains tuples: selection for ones panel, other panels selections etc. will come from zsel
            selectionstoshow = seltype
            shape = (None, None)
            sellabel = '_'.join(['%s-%isigma'%(sel[0], sel[1]) for sel in seltype])
    elif seltype == '1ion':
        selectionstoshow = [[('No6', sigma)], [('No7%s'%model, sigma)], [('No8%s'%model, sigma)],  [('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-1ion'%(model, sigma)
    elif seltype == '2ion':
        selectionstoshow = [[('No6', sigma), ('No7%s'%model, sigma)], [('No6', sigma), ('No8%s'%model, sigma)],  [('No6', sigma), ('Nne9%s'%model, sigma)],\
                            [('No7%s'%model, sigma), ('No8%s'%model, sigma)], [('No7%s'%model, sigma), ('Nne9%s'%model, sigma)], [('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 3)
        sellabel = '%s-%isigma-2ion'%(model, sigma)
    elif seltype == '3ion':
        selectionstoshow = [[('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma)], [('No6', sigma), ('No7%s'%model, sigma), ('Nne9%s'%model, sigma)],\
                            [('No6', sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)], [('No7%s'%model, sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-3ion'%(model, sigma)
    elif seltype == 'addions':
        selectionstoshow = [[('No6', sigma)], [('No6', sigma), ('No7%s'%model, sigma)], [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma)],\
                            [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-addions'%(model, sigma)
    elif seltype == 'paperversion':
        sigma = 1.
        model = 'slab'
        selectionstoshow = [[('No6', sigma)], [('No6', sigma), ('No7%s'%model, sigma)], [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma)],\
                            [('No6', sigma), ('No7%s'%model, sigma), ('No8%s'%model, sigma), ('Nne9%s'%model, sigma)]]
        shape = (2, 2)
        sellabel = '%s-%isigma-addions_paperversion'%(model, sigma)
    
    
    hist = eao678ne9_o6wrhoTfOSm_16
    handleinfedges(hist, setmin=-80., setmax=20.)
    
    o6ax    = 0
    o7ax    = 1
    o8ax    = 2
    ne9ax   = 3
    rhoo6ax = 4
    To6ax   = 5
    fOo6ax  = 6
    
    edges_No6 = hist['edges'][o6ax]
    edges_No7 = hist['edges'][o7ax]
    edges_No8 = hist['edges'][o8ax]
    edges_Nne9 = hist['edges'][ne9ax]
    edges_rhoo6 = hist['edges'][rhoo6ax] + np.log10(rho_to_nh)
    edges_To6 = hist['edges'][To6ax]
    edges_fOo6 = hist['edges'][fOo6ax] - np.log10(ol.solar_abunds['oxygen'])
    histvals = hist['bins']
    
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    
    mu_logNo8_slab = 15.5 
    sigma_logNo8_slab = 0.2 
    mu_logNne9_slab = 15.4
    sigma_logNne9_slab_up = 0.1
    sigma_logNne9_slab_down = 0.2
    ul_logNo7_slab = 15.5
    
    mu_logNo8_cie = 15.4
    sigma_logNo8_cie = 0.2
    mu_logNne9_cie = 14.9
    sigma_logNne9_cie = 0.2
    mu_logNo7_cie = 14.8
    sigma_logNo7_cie = 0.2
    
    mu_T_hot = 6.48
    sigma_T_hot = 0.05
    mu_T_warm = 5.51
    sigma_T_warm_up = 0.24
    sigma_T_warm_down =  0.24
    
    measvals = {'No6':      [mu_logNo6, sigma_logNo6],\
                'No7slab':  [ul_logNo7_slab, 'ul'],\
                'No7cie':   [mu_logNo7_cie,   sigma_logNo7_cie],\
                'No8slab':  [mu_logNo8_slab,  sigma_logNo8_slab],\
                'No8cie':   [mu_logNo8_cie,   sigma_logNo8_cie],\
                'Nne9slab': [mu_logNne9_slab, (sigma_logNne9_slab_down, sigma_logNne9_slab_up)],\
                'Nne9cie':  [mu_logNne9_cie,  sigma_logNne9_cie],\
                }
    edges =    {'No6': edges_No6,\
                'No7': edges_No7,\
                'No8': edges_No8,\
                'Nne9': edges_Nne9,\
                'fOo6': edges_fOo6,\
                }
    seltoion = {'No6':      'No6',\
                'No7slab':  'No7',\
                'No7cie':   'No7',\
                'No8slab':  'No8',\
                'No8cie':   'No8',\
                'Nne9slab': 'Nne9',\
                'Nne9cie':  'Nne9',\
                }
    
    Nlabels = {'No6':  '\mathrm{O\,VI}',\
               'No7':  '\mathrm{O\,VII}',\
               'No8':  '\mathrm{O\,VIII}',\
               'Nne9': '\mathrm{Ne\,IX}',\
               'fOo6': '\log_{10} \, \mathrm{O}',\
               }
    axids = {  'No6':  o6ax,\
               'No7':  o7ax,\
               'No8':  o8ax,\
               'Nne9': ne9ax,\
               'fOo6': fOo6ax,\
               }
   
    
    if zsel is not None:
        if isinstance(zsel[0], list):
            if not np.all(shape == (None,None)):
                raise NotImplementedError('Sets of selections for Z and ions simulateously are not currently supported')
            shape = (len(zsel[0]), len(zsel))
            zsel = [sel for row in zsel for sel in row]
            keysigmacennumsigs = ['zsel_%i'%i for i in range(len(zsel))]
            seltoion.update({key: 'fOo6' for key in keysigmacennumsigs})
            
            measvals.update({keysigmacennumsigs[i]:\
                             [None, None] if zsel[i][0] is None and zsel[i][1] is None else\
                             [zsel[i][1], 'ul'] if zsel[i][0] is None else \
                             [zsel[i][0], 'll'] if zsel[i][1] is None else \
                             [(zsel[i][0] + zsel[i][1]) / 2., (zsel[i][1] - zsel[i][0]) / 2.] \
                             for i in range(len(zsel))} )
            
            numsig = 1.
            selectionstoshow = [selectionstoshow + [(zsel_key, numsig)] for zsel_key in keysigmacennumsigs]
        else: # one z selection, add selection to each panel
            zsel_key = 'fOo6'
            seltoion['fOo6'] = 'fOo6'
            if zsel[0] is None:
                if zsel[1] is None:
                    sigma  = None
                    cen    = None
                    numsig = None
                sigma  = 'ul'
                cen    = zsel[1]
                numsig = None
            elif zsel[1] is None:
                sigma  = 'll'
                cen    = zsel[0]
                numsig = None
            else:
                cen    = (zsel[0] + zsel[1]) / 2.
                sigma  = cen - zsel[0]
                numsig = 1.
            # add to list of things to select on (will also calculate 2 sigma limits, but those are not used)
            measvals.update({zsel_key: [cen, sigma]})
            selectionstoshow = [sel + [(zsel_key, numsig)] for sel in selectionstoshow]
        sellabel = sellabel + insellabel
        
    lims = {(key, numsig): getlims(edges[seltoion[key]], measvals[key][0], sigma=measvals[key][1], numsig=numsig)\
            for key in measvals.keys() for numsig in [1., 2.]}
    
    sels_1ax = {key: slice(lims[key][0], lims[key][1], None) for key in lims.keys()}

    titles_1ax = {key: r'$%s: %.1f \endash %.1f$'%(Nlabels[seltoion[key[0]]],\
                                                  edges[seltoion[key[0]]][lims[key][0]],\
                                                  edges[seltoion[key[0]]][lims[key][1]])\
                            if lims[key][0] is not None and lims[key][1] is not None else\
                       r'$%s: < %.1f$'%(Nlabels[seltoion[key[0]]],\
                                       edges[seltoion[key[0]]][lims[key][1]])\
                            if lims[key][0] is None and lims[key][1] is not None else\
                       r'$%s: > %.1f$'%(Nlabels[seltoion[key[0]]],\
                                       edges[seltoion[key[0]]][lims[key][0]])\
                            if lims[key][0] is not None and lims[key][1] is None else\
                       ''\
                 for key in lims.keys()}
    
    titles = ['\n'.join([titles_1ax[key] for key in panelsel]) for panelsel in selectionstoshow]
    selections = []
    for selection in selectionstoshow:
        sel = list((slice(None, None, None),) * 7)
        for key in selection:
            sel[axids[seltoion[key[0]]]] = sels_1ax[key]
        selections.append(sel)
        
    normhists = [np.sum(histvals[tuple(sel)], axis=(o6ax, o7ax, o8ax, ne9ax, fOo6ax)) \
                      /(np.diff(edges_rhoo6)[:, np.newaxis] * np.diff(edges_To6)[np.newaxis, :]) \
                 for sel in selections]
    
    if addciediff:
        o6frac, T_ib, nh_ib = m3.findiontables('o6', 0.1)
        cie = o6frac[-1, :]
        ciediff = np.abs(np.log10(o6frac / cie[np.newaxis, :]))
        levels_ciediff = np.log10(1. + np.array([0.1, 0.2, 0.5, 1.]))
        colors_ciediff = ['green', 'gold', 'orange', 'red']
        linestyles_ciediff = ['dashed'] * len(levels_ciediff)
    
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = '/home/wijers/Documents/papers/jussi_alexis_o8ne9_counterpart_to_o6/' + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3x_PtAb_C2Sm_32000pix_6p25slice_zcen-all_z-projection_T4EOS_PD_o6_%s.eps'%sellabel
    
    clabel = r'$\log_{10} \left( f_{\mathrm{abs.}} \,/\, \Delta \log_{10}T \, \Delta \log_{10} n_{\mathrm{H}}\right)$'
    xlabel = (r'$\log_{10}\, n_{\mathrm{H}}(\mathrm{O\, VI}) \; [\mathrm{cm}^{-3}]$')
    ylabel = (r'$\log_{10}\, T(\mathrm{O\, VI}) \; [\mathrm{K}]$')
    vmax = np.log10(max([np.max(hi) for his in normhists for hi in his]))
    vmin = vmax - 8.
    
    fig = plt.figure(figsize=(2.5*shape[1] + 0.3, 2.5*shape[0] + 0.6))
    grid = gsp.GridSpec(shape[0] + 1, shape[1] + 1,\
                        height_ratios=list((1.,) * shape[0]) + [0.6],\
                        width_ratios=list((1.,) * shape[1]) + [0.3],\
                        wspace=0.0, hspace=0.0, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    labelax = fig.add_subplot(grid[:shape[0], :shape[1]])
    mainaxes = np.array([[fig.add_subplot(grid[xi, yi]) for yi in range(shape[1])] for xi in range(shape[0])]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:shape[0], shape[1]])
    lax = fig.add_subplot(grid[shape[0], :])               
    
    
    levels = (0.999, 0.99, 0.90, 0.50)
    colors=  ['firebrick', 'darkorange', 'yellowgreen', 'paleturquoise'] #('orange',)*len(levels)
    linestyles= ('solid',) * len(levels) # ['dotted', 'dashed', 'dashdot', 'solid']
    
    if xlim is None:
        xlim = (-8., -1.)
    if ylim is None:
        ylim = (2.5, 8.) 
    for i in range(np.prod(shape)):
        xi = i % shape[1]
        yi = i // shape[1]
        ax = mainaxes[yi, xi]
        bins = normhists[i][2:-1, 2:-1]
        edges =[edges_rhoo6[2:-1], edges_To6[2:-1]]
        print('x: %f -- %f, y: %f -- %f'%(edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]))
        title = titles[i]
        ax.text(0.95, 0.05, title, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes) #, bbox=dict(facecolor='white',alpha=0.3)
        img = subplot_PD_o8ne9(ax, bins, edges, histc=None,\
                         vmin=vmin, vmax=vmax, title=None, fontsize=fontsize,\
                         levels=levels,\
                         colors=colors, linestyles=linestyles,\
                         xlabels=(yi == shape[0] - 1), ylabels=(xi==0), cmap=None,\
                         takeloghist=True)
        #ax.errorbar([mu_T_warm], [mu_T_hot], xerr=([sigma_T_warm_down], [sigma_T_warm_up]), yerr=[sigma_T_hot], fmt='.', color='blue', label='2T CIE', zorder=5, capsize=3, elinewidth=2, capthick=2)
        #if yi == 0:
        #    ax.set_ylabel('$\log_{10}\, T_{\mathrm{O\,VIII}} \; [\mathrm{K}]$', fontsize=fontsize)
        #if xi == shape[1] - 1:
        #    ax.set_xlabel('$\log_{10}\, T_{\mathrm{O\,VI}} \; [\mathrm{K}]$', fontsize=fontsize)
        #ax.set_xlim(3.1, 8.)    
        #ax.set_ylim(3.1, 8.)  
        #ax.plot([0., 10.], [0., 10.], color='brown', linestyle='dashed')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(*xlim)    
        ax.set_ylim(*ylim) 
        
        if addciediff:
            ax.contour(nh_ib, T_ib, ciediff.T, levels=levels_ciediff, colors=colors_ciediff, linestyles=linestyles_ciediff)
     # hide tick and tick label of the big axes
    labelax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    labelax.grid(False)
    labelax.spines['top'].set_color('none')
    labelax.spines['bottom'].set_color('none')
    labelax.spines['left'].set_color('none')
    labelax.spines['right'].set_color('none')
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    
    add_colorbar(cax, img=img ,vmin=vmin, vmax=vmax,\
                 clabel=clabel, newax=False, extend='min',\
                 fontsize=fontsize, orientation='vertical')
    cax.tick_params(labelsize=fontsize - 1, axis='both', which='both')
    cax.set_aspect(12.)
    lax.axis('off')
    handles_subs, labels_subs = mainaxes[0][0].get_legend_handles_labels()
    #level_legend_handles = [mlines.Line2D([], [], color='red', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    level_legend_handles = [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], label='%.1f%% enclosed'%(100.*levels[i])) for i in range(len(levels))]
    if addciediff:
        level_legend_handles = level_legend_handles + \
            [mlines.Line2D([], [], color=colors_ciediff[i], linestyle=linestyles_ciediff[i], label=r'$\Delta \mathrm{CIE} = $%.0f%%'%(100. * (10**levels_ciediff[i] - 1.))) for i in range(len(levels_ciediff))]
    lax.legend(handles=level_legend_handles + handles_subs, fontsize=fontsize, ncol=shape[1] + 1, loc='upper center', bbox_to_anchor=(0.5, 0.2))
    
    plt.savefig(name, format='eps', bbox_inches='tight')
    
    
    
def plotabsorbercomps_o678ne9corr_paperversion():
    '''
    
    '''
    hist = eao678ne9_16
    
    mu_logNo6 = 13.263 
    sigma_logNo6 = 0.11
    
    mu_logNo8_slab = 15.5 
    sigma_logNo8_slab = 0.2 
    mu_logNne9_slab = 15.4
    sigma_logNne9_slab_up = 0.1
    sigma_logNne9_slab_down = 0.2
    ul_logNo7_slab = 15.5
    
    mu_logNo8_cie = 15.4
    sigma_logNo8_cie = 0.2
    mu_logNne9_cie = 14.9
    sigma_logNne9_cie = 0.2
    mu_logNo7_cie = 14.8
    sigma_logNo7_cie = 0.2
    
    fontsize = 12
    vmin = -10.
    vmax = 0.
    name = '/home/wijers/Documents/papers/jussi_alexis_o8ne9_counterpart_to_o6/' + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3x_PtAb_C2Sm_32000pix_6p25slice_zcen-all_z-projection_T4EOS_paperversion.eps'
    
    edges_No6 = hist['edges'][0]
    edges_No7 = hist['edges'][1]
    edges_No8 = hist['edges'][2]
    edges_Nne9 = hist['edges'][3]
    histvals = hist['bins']
    
    # 1 sigma `bucket':
    minmaxinds_o6_b1s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + sigma_logNo6)))]    
    minmaxinds_o6_b2s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 2 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 2 * sigma_logNo6)))]
    #minmaxinds_o6_b5s = [np.argmin(np.abs(edges_No6 - (mu_logNo6 - 5 * sigma_logNo6))), np.argmin(np.abs(edges_No6 - (mu_logNo6 + 5 * sigma_logNo6)))]
    maxind_o7slab = np.argmin(np.abs(edges_No7 - ul_logNo7_slab))
    minmaxinds_o7cie_b1s = [np.argmin(np.abs(edges_No7 - (mu_logNo7_cie - sigma_logNo7_cie))),     np.argmin(np.abs(edges_No7 - (mu_logNo7_cie + sigma_logNo7_cie)))]    
    minmaxinds_o7cie_b2s = [np.argmin(np.abs(edges_No7 - (mu_logNo7_cie - 2 * sigma_logNo7_cie))), np.argmin(np.abs(edges_No7 - (mu_logNo7_cie + 2 * sigma_logNo7_cie)))]
    #minmaxinds_o7cie_b5s = [np.argmin(np.abs(edges_No7 - (mu_logNo7_cie - 5 * sigma_logNo7_cie))), np.argmin(np.abs(edges_No7 - (mu_logNo7_cie + 5 * sigma_logNo7_cie)))]
    
    #print minmaxinds_o6_b1s, minmaxinds_o8_b1s, minmaxinds_o6_b2s, minmaxinds_o8_b2s, minmaxinds_o6_b5s, minmaxinds_o8_b5s
    
    centres_No6 = edges_No6[:-1] + 0.5 * np.diff(edges_No6)
    centres_No8 = edges_No8[:-1] + 0.5 * np.diff(edges_No8)
    centres_Nne9 = edges_Nne9[:-1] + 0.5 * np.diff(edges_Nne9)
        
    minmax_o6_b1s = [edges_No6[minmaxinds_o6_b1s[0]], edges_No6[minmaxinds_o6_b1s[1]]] 
    minmax_o6_b2s = [edges_No6[minmaxinds_o6_b2s[0]], edges_No6[minmaxinds_o6_b2s[1]]] 
    max_o7slab = edges_No7[maxind_o7slab]
    minmax_o7cie_b1s = [edges_No7[minmaxinds_o7cie_b1s[0]], edges_No6[minmaxinds_o7cie_b1s[1]]] 
    minmax_o7cie_b2s = [edges_No7[minmaxinds_o7cie_b2s[0]], edges_No6[minmaxinds_o7cie_b2s[1]]] 
    #print minmax_o6_b1s, minmax_o8_b1s, minmax_o6_b2s, minmax_o8_b2s
    
    ### basic histograms: number of absorbers in various column density selections
    #sel = list((slice(None, None, None),)*len(hist['edges']))
    o6sels = ['o6_1sigma'] # , 'o6_2sigma']
    o7sels = ['o7none', 'o7slab_ul'] #, 'o7cie_1sigma', 'o7cie_2sigma']
    sels_1ax = {'o6_1sigma': slice(minmaxinds_o6_b1s[0], minmaxinds_o6_b1s[1], None),\
                'o6_2sigma': slice(minmaxinds_o6_b2s[0], minmaxinds_o6_b2s[1], None),\
                'o7none': slice(None, None, None),\
                'o7slab_ul': slice(None, maxind_o7slab, None),\
                'o7cie_1sigma': slice(minmaxinds_o7cie_b1s[0], minmaxinds_o7cie_b1s[1], None),\
                'o7cie_2sigma': slice(minmaxinds_o7cie_b2s[0], minmaxinds_o7cie_b2s[1], None),\
                }
    titles_1ax = {'o6_1sigma': r'$\mathrm{O\, VI}: %.1f \endash %.1f$'%(minmax_o6_b1s[0], minmax_o6_b1s[1]),\
                  'o6_2sigma': r'$\mathrm{O\, VI}: %.1f \endash %.1f$'%(minmax_o6_b2s[0], minmax_o6_b2s[1]),\
                  'o7none': '',\
                  'o7slab_ul': r'$\mathrm{O\, VII}: < %.1f$'%(max_o7slab),\
                  'o7cie_1sigma': r'$\mathrm{O\, VII}: %.1f \endash %.1f$'%(minmax_o7cie_b1s[0], minmax_o7cie_b1s[1]),\
                  'o7cie_2sigma': r'$\mathrm{O\, VII}: %.1f \endash %.1f$'%(minmax_o7cie_b2s[0], minmax_o7cie_b2s[1]),\
                 }
    
    normhists = [[ np.sum(histvals[sels_1ax[o6sel], sels_1ax[o7sel], ...], axis=(0, 1)) \
                  /(np.diff(edges_No8)[:, np.newaxis] * np.diff(edges_Nne9)[np.newaxis, :])  for o7sel in o7sels] for o6sel in o6sels]
    clabel = r'$\log_{10} \left( f_{\mathrm{abs.}} \,/\, (\Delta \log_{10}N)^2 \right)$'
    vmax = np.log10(max([np.max(hi) for his in normhists for hi in his]))
    vmin = vmax - 8.
    
    fig = plt.figure(figsize=(5.5, 1.5/2.2 * 5.5))
    grid = gsp.GridSpec(2, 3, height_ratios=[1., 0.3], width_ratios=[1., 1., 0.2], wspace=0.0, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    labelax = fig.add_subplot(grid[len(o6sels) - 1, :2])
    mainaxes = np.array([[fig.add_subplot(grid[yi, xi]) for yi in range(1)] for xi in range(2)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[0, 2])
    lax = fig.add_subplot(grid[1, :])               
    
    edges =[edges_No8, edges_Nne9]
    levels = (0.999, 0.99, 0.90, 0.50)
    colors=  ['firebrick', 'darkorange', 'yellowgreen', 'paleturquoise'] #('orange',)*len(levels)
    linestyles= ('solid',) * len(levels) # ['dotted', 'dashed', 'dashdot', 'solid']
    for i in range(len(o6sels) * len(o7sels)):
        xi = i % len(o6sels)
        yi = i // len(o6sels)
        ax = mainaxes[yi, xi]
        o6sel = o6sels[xi]
        o7sel = o7sels[yi]
        bins = normhists[xi][yi]
        title = '\n'.join([titles_1ax[o7sel], titles_1ax[o6sel]])
        while title[-1:] == '\n':
            title = title[:-1]
        while title[0] == '\n':
            title = title[1:]
        ax.text(0.95, 0.05, title, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes) # , bbox=dict(facecolor='white',alpha=0.3)
        img = subplot_coldenscomps_o8ne9(ax, bins, edges, histc=None,\
                                vmin=vmin, vmax=vmax, title=None, fontsize=12,\
                                levels=levels, colors=colors, linestyles=linestyles,\
                                ylabels=(yi == 0), xlabels=(xi == len(o6sels) - 1),\
                                cmap=None,\
                                takeloghist=True, plotslab=True, plotcie=True)
        ax.set_xlim(11.85, 17.45)
        ax.set_xlabel('')
    
     # hide tick and tick label of the big axes
    labelax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off', labelleft=False)
    labelax.grid(False)
    labelax.spines['top'].set_color('none')
    labelax.spines['bottom'].set_color('none')
    labelax.spines['left'].set_color('none')
    labelax.spines['right'].set_color('none')
    labelax.set_xlabel(r'$\log_{10}\, N(\mathrm{O\, VIII}) \; [\mathrm{cm}^{-2}]$', fontsize=fontsize)

    add_colorbar(cax, img=img ,vmin=vmin, vmax=vmax,\
                 clabel=clabel, newax=False, extend='min',\
                 fontsize=fontsize, orientation='vertical')
    cax.tick_params(labelsize=fontsize - 1, axis='both', which='both')
    cax.set_aspect(10.)
    lax.axis('off')
    handles_subs, labels_subs = mainaxes[0][0].get_legend_handles_labels()
    #level_legend_handles = [mlines.Line2D([], [], color='red', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    level_legend_handles = [mlines.Line2D([], [], color=colors[i], linestyle=linestyles[i], label='%.1f%% enclosed'%(100.*levels[i])) for i in range(len(levels))]
    lax.legend(handles=handles_subs + level_legend_handles, fontsize=fontsize, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.9))
    
    plt.savefig(name, format='eps', bbox_inches='tight')



###############################################################################
########### plots for Z measurement differences (w/ Maryam Arabsalmani) #######
###############################################################################

def plotZmeascomps(fontsize=fontsize):
    
    hist_h1   = ea25Zmeas_h1   
    hist_mass = ea25Zmeas_mass 
    hist_sfr  = ea25Zmeas_sfr 
    handleinfedges(hist_h1, setmin=-100., setmax=100.)
    handleinfedges(hist_sfr, setmin=-100., setmax=100.)
    handleinfedges(hist_mass, setmin=-100., setmax=100.)
    aexp = 0.498972 
    
    massconv = c.solar_mass / (c.cm_per_mpc / 1.e3)**2
    masslabel = r'$\log_{10} \, \Sigma_{\mathrm{gas}}\; [\mathrm{M}_{\odot} \, \mathrm{pkpc}^{-2}]$'
    
    h1conv = 1.
    h1label = r'$\log_{10} \, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$'
    
    sfrconv = c.solar_mass / (c.cm_per_mpc / 1.e3)**2 / c.sec_per_year
    sfrlabel = r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot}\, \mathrm{pkpc}^{-2} \mathrm{yr}^{-1}]$'
    
    Zconv = ol.Zsun_sylviastables
    Zlabel = r'$\log_{10} \, Z_{\mathrm{%s} \; [Z_{\odot}]}$'
    Zmasslabel = Zlabel%'mass'
    Zsfrlabel  = Zlabel%'SFR'
    Zh1label   = Zlabel%'H\,I'
    
    Zmax = 1
    Zhax = 2
    Zsax = 3
    
    labels = {Zmax: Zmasslabel, Zhax: Zh1label, Zsax: Zsfrlabel}
    
    fig = plt.figure(figsize=(10.5, 3.))
    grid = gsp.GridSpec(1, 4, height_ratios=[1.], width_ratios=[1., 1., 1., 0.2], wspace=0.35, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[yi, xi]) for yi in range(1)] for xi in range(3)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[0, 3])
    
    sumaxes = [(0, Zsax), (0, Zhax), (0, Zmax)]
    vmaxs = [getminmax2d(hist_h1, axis=i, log=True, pixdens=True)[1] for i in sumaxes]
    vmax = max(vmaxs)
    vmin = vmax - 8.
    xlim = (-8., 1.)
    ylim = xlim 
    
    cmap = 'gist_yarg'
    
    for i in range(3):
        ax = mainaxes[i, 0]
        sumaxis = sumaxes[i]
        plotaxes = list(set(np.arange(4)) - set(sumaxis))
        plotaxes.sort()
        plotaxes=tuple(plotaxes)
        imgminmax = add_2dplot(ax, hist_h1, plotaxes, log=True, usepcolor=True, pixdens=True, shiftx=-1.*np.log10(Zconv), shifty=-1.*np.log10(Zconv), cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel(labels[plotaxes[0]], fontsize=fontsize)
        ax.set_ylabel(labels[plotaxes[1]], fontsize=fontsize)
        setticks(ax, fontsize, color='black', bottom=True, top=True, left=True, labelright=False, right=True, labeltop=False)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.plot(xlim, xlim, color='dodgerblue', linestyle='solid', linewidth=1.)
    
    add_colorbar(cax,img=imgminmax[0],vmin=vmin,vmax=vmax,cmap=cmap,clabel=r'$\log_{10}\, \mathrm{absorber\, fraction} / \left(\Delta \log_{10} Z\right)^2$',newax=False,extend='min',fontsize=fontsize,orientation='vertical')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    
    plt.savefig('/net/luttero/data2/imgs/Zmeascomps/Z_vs_Z_nocuts_L0025N0376_19_PtAb_C2Sm_6.25slice_8000pix_T4EOS.pdf', format='pdf', bbox_inches='tight')
    
def plotZmeascomps_by_measq(fontsize=fontsize, measured='mass', simset='L0025N0376_19', cvar=False):
    
    #aexp = 0.498972 
    
    massconv = c.solar_mass / (c.cm_per_mpc / 1.e3)**2
    masslabel = r'$\log_{10} \, \Sigma_{\mathrm{gas}}\; [\mathrm{M}_{\odot} \, \mathrm{pkpc}^{-2}]$'
    h1conv = 1.
    h1label = r'$\log_{10} \, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$'
    sfrconv = c.solar_mass / (c.cm_per_mpc / 1.e3)**2 / c.sec_per_year
    sfrlabel = r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot}\, \mathrm{pkpc}^{-2} \mathrm{yr}^{-1}]$'
    
    if simset == 'L0025N0376_19':
        basename  = 'Z_vs_Z_meascuts_%s_L0025N0376_19_PtAb_C2Sm_6.25slice_8000pix_T4EOS.pdf'  
    elif simset == 'L0025N0752Recal_19_SmZ':
        basename  = 'Z_vs_Z_meascuts_%s_L0025N0752Recal_19_SmZ_PtAb_C2Sm_3.125slice_10000pix_T4EOS.pdf'  
    elif simset == 'L0025N0752Recal_19_SmZ_hn':
        basename  = 'Z_vs_Z_meascuts_usinghneutralssh_%s_L0025N0752Recal_19_SmZ_PtAb_C2Sm_3.125slice_10000pix_T4EOS.pdf'  
            
    if measured == 'mass':
        if simset == 'L0025N0376_19':
            hist = ea25Zmeas_mass 
        elif simset == 'L0025N0752Recal_19_SmZ':
            hist = ea25RecZmeas_mass
        elif simset == 'L0025N0752Recal_19_SmZ_hn':
             hist = ea25RecZmeas_hn_mass       
        handleinfedges(hist, setmin=-100., setmax=100.)
        measconv = massconv
        measlabel = masslabel
        contourbins = [4., 5., 6., 7., 8., 9.]
        levels = [0.90]
        linestyles = ['solid']
    elif measured == 'h1':
        if simset == 'L0025N0376_19':
            hist = ea25Zmeas_h1   
        elif simset == 'L0025N0752Recal_19_SmZ':
            hist = ea25RecZmeas_h1 
        elif simset == 'L0025N0752Recal_19_SmZ_hn':
            hist = ea25RecZmeas_hn_h1  
        handleinfedges(hist, setmin=-100., setmax=100.)
        measconv = h1conv
        measlabel = h1label
        if cvar:
            contourbins = [17., 19., 20., 20.3, 20.4, 20.5, 20.6, 20.7, 21.]
        else:
            contourbins = [12., 14., 15., 17., 19., 20., 21.]
        levels = [0.90]
        linestyles = ['solid']
    elif measured == 'sfr':
        if simset == 'L0025N0376_19':
            hist  = ea25Zmeas_sfr   
        elif simset == 'L0025N0752Recal_19_SmZ':
            hist = ea25RecZmeas_sfr
        elif simset == 'L0025N0752Recal_19_SmZ_hn':
            hist = ea25RecZmeas_hn_sfr  

        handleinfedges(hist, setmin=-100., setmax=100.)
        measconv = sfrconv
        measlabel= sfrlabel
        contourbins = [-5., -4., -3., -2., -1., 0.]
        levels = [0.90]
        linestyles = ['solid']
        
    Zconv = ol.Zsun_sylviastables
    Zlabel = r'$\log_{10} \, Z_{\mathrm{%s} \; [Z_{\odot}]}$'
    Zmasslabel = Zlabel%'mass'
    Zsfrlabel  = Zlabel%'SFR'
    Zh1label   = Zlabel%'H\,I'
    
    Zmax = 1
    Zhax = 2
    Zsax = 3
    
    labels = {Zmax: Zmasslabel, Zhax: Zh1label, Zsax: Zsfrlabel}
    
    fig = plt.figure(figsize=(10.5, 5.5))
    grid = gsp.GridSpec(2, 4, height_ratios=[1., 1.], width_ratios=[1., 1., 1., 0.2], wspace=0.35, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([[fig.add_subplot(grid[yi, xi]) for yi in range(1)] for xi in range(3)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[0, 3])
    lax = fig.add_subplot(grid[1, :])
    
    sumaxes = [(0, Zsax), (0, Zhax), (0, Zmax)]
    vmaxs = [getminmax2d(hist, axis=i, log=True, pixdens=True)[1] for i in sumaxes]
    vmax = max(vmaxs)
    vmin = vmax - 8.
    xlim = (-8., 1.)
    ylim = xlim 
    
    cmap = 'gist_yarg'
    colors = ['saddlebrown', 'maroon', 'red', 'orange', 'gold', 'forestgreen', 'lime', 'cyan', 'blue', 'purple', 'magenta']
    dimlabels = (measlabel, labels[1], labels[2], labels[3])
    dimshifts = (-1.*np.log10(measconv),) +  (-1.*np.log10(Zconv),) * 3
    edgeinds = [np.argmin(np.abs(contourbins[i] - hist['edges'][0] + np.log10(measconv))) for i in range(len(contourbins))]
    mins = [(edge,) + (None,) * 3 for edge in [None] + edgeinds]
    maxs = [(edge,) + (None,) * 3 for edge in edgeinds + [None]]

    for i in range(3):
        ax = mainaxes[i, 0]
        sumaxis = sumaxes[i]
        plotaxes = list(set(np.arange(4)) - set(sumaxis))
        plotaxes.sort()
        plotaxes=tuple(plotaxes)
        imgminmax = add_2dplot(ax, hist, plotaxes, log=True, usepcolor=True, pixdens=True, shiftx=-1.*np.log10(Zconv), shifty=-1.*np.log10(Zconv), cmap=cmap, vmin=vmin, vmax=vmax)
        
        for j in range(len(mins)):
            add_2dhist_contours(ax, hist, plotaxes, mins=mins[j], maxs=maxs[j],\
                                histlegend=False, fraclevels=True, levels=levels, legend=False,\
                                dimlabels=dimlabels, legendlabel=None, legendlabel_pre=None,\
                                shiftx=-1.*np.log10(Zconv), shifty=-1.*np.log10(Zconv), dimshifts=dimshifts,\
                                colors=[colors[j]] * len(levels), linestyles=linestyles)
        
        ax.set_xlabel(labels[plotaxes[0]], fontsize=fontsize)
        ax.set_ylabel(labels[plotaxes[1]], fontsize=fontsize)
        setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False)
        
        handles_subs, labels_subs = ax.get_legend_handles_labels()
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        ax.plot(xlim, xlim, color='dodgerblue', linestyle='solid', linewidth=1.)
    
    add_colorbar(cax,img=imgminmax[0],vmin=vmin,vmax=vmax,cmap=cmap,clabel=r'$\log_{10}\, \mathrm{absorber\, fraction} / \left(\Delta \log_{10} Z\right)^2$',newax=False,extend='min',fontsize=fontsize,orientation='vertical')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    
    handles_encl = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.0f %%'%(levels[i]*100.)) for i in range(len(levels))]
    lax.legend(handles=handles_subs + handles_encl, fontsize=fontsize, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.75))
    lax.axis('off')
    
    outname = basename%measured
    if cvar:
        outname = outname[:-4] + '_cvar' + outname[-4:]
    plt.savefig('/net/luttero/data2/imgs/Zmeascomps/' + outname, format='pdf', bbox_inches='tight')


def plotZmeasdiffs_by_measq(fontsize=fontsize, measured='mass', simset='L0025N0376_19'):
    
    aexp = 0.498972 
    
    massconv = c.solar_mass / (c.cm_per_mpc / 1.e3)**2
    masslabel = r'$\log_{10} \, \Sigma_{\mathrm{gas}}\; [\mathrm{M}_{\odot} \, \mathrm{pkpc}^{-2}]$'
    h1conv = 1.
    h1label = r'$\log_{10} \, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$'
    sfrconv = c.solar_mass / (c.cm_per_mpc / 1.e3)**2 / c.sec_per_year
    sfrlabel = r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot}\, \mathrm{pkpc}^{-2} \mathrm{yr}^{-1}]$'
    
    Zconv = ol.Zsun_sylviastables
    Zlabel = r'$\log_{10} \, Z_{\mathrm{%s}} \; [Z_{\odot}]$'
    Zdifflabel = r'$\log_{10} \, Z_{\mathrm{%s}} \,/\, Z_{\mathrm{%s}}$'
    Zmasslabel = Zlabel%'mass'
    Zsfrlabel  = Zlabel%'SFR'
    Zh1label   = Zlabel%'H\,I'
    
    if simset == 'L0025N0376_19':
        addname = ''
        sim = 'L0025N0376_19_PtAb_C2Sm_6.25slice_8000pix_T4EOS'
    elif simset == 'L0025N0376_19_SmZ':
        addname = '_SmoothedZ'
        sim = 'L0025N0376_19_PtAb_C2Sm_6.25slice_8000pix_T4EOS'
    elif simset == 'L0025N0376_19_SmZ_hn':
        addname = '_SmoothedZ_hneutral'  
        sim = 'L0025N0376_19_PtAb_C2Sm_6.25slice_8000pix_T4EOS'
    elif simset == 'L0025N0752Recal_19_SmZ':
        addname = '_SmoothedZ'
        sim = 'L0025N0752RECALIBRATED_19_C2Sm_10000pix_3.125slice'
    elif simset == 'L0025N0752Recal_19_SmZ_hn':
        addname = '_SmoothedZ_hneutral'  
        sim = 'L0025N0752RECALIBRATED_19_C2Sm_10000pix_3.125slice'

    if measured == 'mass':
        if simset == 'L0025N0376_19':
            hist = ea25Zdiff_mass 
        elif simset == 'L0025N0376_19_SmZ':
            hist = ea25SmZdiff_mass 
        elif simset == 'L0025N0376_19_SmZ_hn':
            hist = ea25SmZdiff_mass_hn 
        elif simset == 'L0025N0752Recal_19_SmZ':
            hist = ea25RecSmZdiff_mass
        elif simset == 'L0025N0752Recal_19_SmZ_hn':
            hist = ea25RecSmZdiff_mass_hn
        handleinfedges(hist, setmin=-100., setmax=100.)
        measconv  = massconv
        measlabel = masslabel
        Zmeaslabel = Zmasslabel
        d1label   = Zdifflabel%('H\,I', 'mass')
        d2label   = Zdifflabel%('SFR', 'mass')
        contourbins = [4., 5., 6., 7., 8., 9.]
        levels = [0.90]
        linestyles = ['solid']
        qtlim  = (4., 8.5)
    elif measured == 'h1':
        if simset == 'L0025N0376_19':
            hist = ea25Zdiff_h1 
        elif simset == 'L0025N0376_19_SmZ':
            hist = ea25SmZdiff_h1 
        elif simset == 'L0025N0376_19_SmZ_hn':
            hist = ea25SmZdiff_h1_hn 
        elif simset == 'L0025N0752Recal_19_SmZ':
            hist = ea25RecSmZdiff_h1
        elif simset == 'L0025N0752Recal_19_SmZ_hn':
            hist = ea25RecSmZdiff_h1_hn
        handleinfedges(hist, setmin=-100., setmax=100.)
        measconv = h1conv
        measlabel = h1label
        Zmeaslabel = Zh1label
        d1label   = Zdifflabel%('mass', 'H\,I')
        d2label   = Zdifflabel%('SFR', 'H\,I')
        levels = [0.90]
        linestyles = ['solid']
        qtlim  = (14.5, 22.5)
    elif measured == 'sfr':
        if simset == 'L0025N0376_19':
            hist  = ea25Zdiff_sfr 
        elif simset == 'L0025N0376_19_SmZ':
            hist  = ea25SmZdiff_sfr
        elif simset == 'L0025N0376_19_SmZ_hn':
            hist  = ea25SmZdiff_sfr_hn 
        elif simset == 'L0025N0752Recal_19_SmZ':
            hist = ea25RecSmZdiff_sfr
        elif simset == 'L0025N0752Recal_19_SmZ_hn':
            hist = ea25RecSmZdiff_sfr_hn
        handleinfedges(hist, setmin=-100., setmax=100.)
        measconv = sfrconv
        measlabel= sfrlabel
        Zmeaslabel = Zsfrlabel
        d1label   = Zdifflabel%('mass', 'SFR')
        d2label   = Zdifflabel%('H\,I', 'SFR')
        contourbins = [-5., -4., -3., -2., -1., 0.]
        levels = [0.90]
        linestyles = ['solid']
        qtlim = (-7., -0.5)
    Zmax = 1
    d1ax = 2
    d2ax = 3
    
    labels = {0: measlabel, Zmax: Zmeaslabel, d1ax: d1label,   d2ax: d2label  }
    convs  = {0: measconv,  Zmax: Zconv,      d1ax: 1.,        d2ax: 1.       }
    lims   = {0: qtlim,     Zmax: (-4., 1.),  d1ax: (-3., 3.), d2ax: (-3., 3.)}
    
    fig = plt.figure(figsize=(10.5, 5.5))
    grid = gsp.GridSpec(2, 4, height_ratios=[1., 1.], width_ratios=[1., 1., 1., 0.2], wspace=0.35, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([fig.add_subplot(grid[yi, xi]) for (yi, xi) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:, 3])
    lax = fig.add_subplot(grid[1, 0])
    
    sumaxes =  [(d1ax, d2ax), (Zmax, d2ax), (Zmax, d1ax), (0,    d2ax), (0,    d1ax)]
    plotaxes = [(0,    Zmax), (0,    d1ax), (0,    d2ax), (Zmax, d1ax), (Zmax, d2ax)]
    vmaxs = [getminmax2d(hist, axis=i, log=True, pixdens=True)[1] for i in sumaxes]
    vmax = max(vmaxs)
    vmin = vmax - 6.
    
    cmap = 'gist_yarg'
    colors = ['saddlebrown', 'maroon', 'red', 'orange', 'gold', 'forestgreen', 'lime', 'cyan', 'blue', 'purple', 'magenta']
    dimlabels = (measlabel, labels[1], labels[2], labels[3])
    dimshifts = (-1.*np.log10(measconv), -1.*np.log10(Zconv), 0., 0.)
    edgeinds = [np.argmin(np.abs(contourbins[i] - hist['edges'][0] + np.log10(measconv))) for i in range(len(contourbins))]
    mins = [(edge,) + (None,) * 3 for edge in [None] + edgeinds]
    maxs = [(edge,) + (None,) * 3 for edge in edgeinds + [None]]

    percarr = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    linestyles_perc = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    colors_perc = ['magenta'] *len(percarr)

    for i in range(5):
        ax = mainaxes[i]
        sumaxis = sumaxes[i]
        plotaxes = list(set(np.arange(4)) - set(sumaxis))
        plotaxes.sort()
        plotaxes = tuple(plotaxes)
        imgminmax = add_2dplot(ax, hist, plotaxes, log=True, usepcolor=True, pixdens=True, shiftx=-1.*np.log10(convs[plotaxes[0]]), shifty=-1.*np.log10(convs[plotaxes[1]]), cmap=cmap, vmin=vmin, vmax=vmax)
        
        #for j in range(len(mins)):
        #    add_2dhist_contours(ax, hist, plotaxes, mins=mins[j], maxs=maxs[j],\
        #                        histlegend=False, fraclevels=True, levels=levels, legend=False,\
        #                        dimlabels=dimlabels, legendlabel=None, legendlabel_pre=None,\
        #                        shiftx=-1.*np.log10(convs[plotaxes[0]]), shifty=-1.*np.log10(convs[plotaxes[1]]), dimshifts=dimshifts,\
        #                        colors=[colors[j]] * len(levels), linestyles=linestyles)
        #
        #
        
        subhist = np.sum(hist['bins'], axis=sumaxis)
        percentiles = percentiles_from_histogram(subhist, hist['edges'][plotaxes[1]], axis=1, percentiles=percarr)
        cens_ax0 = hist['edges'][plotaxes[0]]
        cens_ax0 = cens_ax0[:-1] + 0.5 * np.diff(cens_ax0)
        plotcrit = np.sum(subhist, axis=1) * hist.npt >= 10.
        plotsl   = slice(np.where(plotcrit)[0][0], np.where(plotcrit)[0][-1] + 1, None)
        for pi in range(len(percentiles)):
            ax.plot(cens_ax0[plotsl] - np.log10(convs[plotaxes[0]]), percentiles[pi, plotsl]- np.log10(convs[plotaxes[1]]), linestyle=linestyles_perc[pi], color=colors_perc[pi], label='%.0f %%'%(percarr[pi] * 100.))
        
        handles_subs, labels_subs = ax.get_legend_handles_labels()
        
        ax.set_xlabel(labels[plotaxes[0]], fontsize=fontsize)
        ax.set_ylabel(labels[plotaxes[1]], fontsize=fontsize)
        setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False)
        
        
        ax.set_xlim(*lims[plotaxes[0]])
        ax.set_ylim(*lims[plotaxes[1]])
        
        if plotaxes[1] in [d1ax, d2ax]:
            ax.plot(lims[plotaxes[0]], [0., 0.], color='dodgerblue', linestyle='solid', linewidth=1.)
    
    add_colorbar(cax, img=imgminmax[0], vmin=vmin, vmax=vmax, cmap=cmap, \
                 clabel=r'$\log_{10}\, \mathrm{absorber\, fraction} / \mathrm{pix. size} \; [\mathrm{dex}^{-2}] $',
                 newax=False, extend='min', fontsize=fontsize, orientation='vertical')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    
    #handles_encl = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.0f %%'%(levels[i]*100.)) for i in range(len(levels))]
    handles_encl = []
    lax.legend(handles=handles_subs + handles_encl, fontsize=fontsize, ncol=2, loc='upper right', bbox_to_anchor=(0.95, 0.9))
    lax.axis('off')
    
    plt.savefig('/net/luttero/data2/imgs/Zmeascomps/Z_vs_Z_diffplots_%s_%s%s.pdf'%(measured, sim, addname), format='pdf', bbox_inches='tight')
    
def plot_Zcompmeasures():
    fontsize = 12.
    
    hist = ea25Zmass_meascomp
    
    msax = 0
    h1ax = 1
    sfax = 2
    zmax = 3
    
    labels = {msax: r'$\log_{10} \, \Sigma_{\mathrm{gas}}\; [\mathrm{M}_{\odot} \, \mathrm{pkpc}^{-2}]$',\
              h1ax: r'$\log_{10} \, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$',\
              sfax: r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot}\, \mathrm{pkpc}^{-2} \mathrm{yr}^{-1}]$',\
              zmax: r'$\log_{10} \, Z_{\mathrm{mass}} \; [Z_{\odot}]$'}
    
    convs = {msax: c.solar_mass / (c.cm_per_mpc / 1.e3)**2,\
             h1ax: 1.,\
             sfax: c.solar_mass / (c.cm_per_mpc / 1.e3)**2 / c.sec_per_year,\
             zmax: ol.Zsun_sylviastables}
    
    lims = {msax: (4., 8.5), h1ax: (11., 22.5), sfax:(-7., -0.5), zmax: (-8., 1.)}
    
    fig = plt.figure(figsize=(8.5, 2.5))
    grid = gsp.GridSpec(1, 4, width_ratios=[1., 1., 1., 0.2], wspace=0.45, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([fig.add_subplot(grid[xi]) for xi in range(3)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:, 3])
    
    sumaxes =  [(sfax, zmax), (msax, zmax), (msax, sfax)]
    plotaxes = [(h1ax, msax), (h1ax, sfax), (h1ax, zmax)]
    vmaxs = [getminmax2d(hist, axis=i, log=True, pixdens=True)[1] for i in sumaxes]
    vmax = max(vmaxs)
    vmin = vmax - 6.
    
    cmap = 'gist_yarg'
    #colors = ['saddlebrown', 'maroon', 'red', 'orange', 'gold', 'forestgreen', 'lime', 'cyan', 'blue', 'purple', 'magenta']
    #dimlabels = (labels[0], labels[1], labels[2], labels[3])
    #dimshifts = (-1.*np.log10(convs[i]) for i in range(4))
    #edgeinds = [np.argmin(np.abs(contourbins[i] - hist['edges'][0] + np.log10(measconv))) for i in range(len(contourbins))]
    #mins = [(edge,) + (None,) * 3 for edge in [None] + edgeinds]
    #maxs = [(edge,) + (None,) * 3 for edge in edgeinds + [None]]

    percarr = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    linestyles_perc = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    colors_perc = ['magenta'] *len(percarr)

    for i in range(3):
        ax = mainaxes[i]
        sumaxis = sumaxes[i]
        plotaxis = plotaxes[i]
        plotaxis = tuple(plotaxis)
        imgminmax = add_2dplot(ax, hist, plotaxis, log=True, usepcolor=True, pixdens=True, shiftx=-1.*np.log10(convs[plotaxis[0]]), shifty=-1.*np.log10(convs[plotaxis[1]]), cmap=cmap, vmin=vmin, vmax=vmax)
        
        #for j in range(len(mins)):
        #    add_2dhist_contours(ax, hist, plotaxes, mins=mins[j], maxs=maxs[j],\
        #                        histlegend=False, fraclevels=True, levels=levels, legend=False,\
        #                        dimlabels=dimlabels, legendlabel=None, legendlabel_pre=None,\
        #                        shiftx=-1.*np.log10(convs[plotaxes[0]]), shifty=-1.*np.log10(convs[plotaxes[1]]), dimshifts=dimshifts,\
        #                        colors=[colors[j]] * len(levels), linestyles=linestyles)
        #
        #
        
        subhist = np.sum(hist['bins'], axis=sumaxis)
        if plotaxis[0] > plotaxis[1]:
            subhist=subhist.T
        percentiles = percentiles_from_histogram(subhist, hist['edges'][plotaxis[1]], axis=1, percentiles=percarr)
        cens_ax0 = hist['edges'][plotaxis[0]]
        cens_ax0 = cens_ax0[:-1] + 0.5 * np.diff(cens_ax0)
        plotcrit = np.sum(subhist, axis=1) * hist.npt >= 10.
        plotsl   = slice(np.where(plotcrit)[0][0], np.where(plotcrit)[0][-1] + 1, None)
        for pi in range(len(percentiles)):
            ax.plot(cens_ax0[plotsl] - np.log10(convs[plotaxis[0]]), percentiles[pi, plotsl]- np.log10(convs[plotaxis[1]]), linestyle=linestyles_perc[pi], color=colors_perc[pi], label='%.0f %%'%(percarr[pi] * 100.))
        
        handles_subs, labels_subs = ax.get_legend_handles_labels()
        
        ax.set_xlabel(labels[plotaxis[0]], fontsize=fontsize)
        ax.set_ylabel(labels[plotaxis[1]], fontsize=fontsize)
        setticks(ax, fontsize, color='black', bottom=True, top=True, left=True, labelright=False, right=True, labeltop=False)
        
        
        ax.set_xlim(*lims[plotaxis[0]])
        ax.set_ylim(*lims[plotaxis[1]])
        
    add_colorbar(cax, img=imgminmax[0], vmin=vmin, vmax=vmax, cmap=cmap, \
                 clabel=r'$\log_{10}\, \mathrm{absorber\, fraction} / \mathrm{pix. size} \; [\mathrm{dex}^{-2}] $',
                 newax=False, extend='min', fontsize=fontsize, orientation='vertical')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    
    #handles_encl = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.0f %%'%(levels[i]*100.)) for i in range(len(levels))]
    handles_encl = []
    mainaxes[1].legend(handles=handles_subs + handles_encl, fontsize=fontsize - 1, ncol=1, loc='upper left', bbox_to_anchor=(0.01, 0.99))
    
    plt.savefig('/net/luttero/data2/imgs/Zmeascomps/coldens_h1ssh_vs_mass_sfr_Zmass_L0025N0376_19_PtAb_C2Sm_6.25slice_8000pix_T4EOS.pdf', format='pdf', bbox_inches='tight')


def plot_Zmeas_basecomp(hn=False):
    fontsize = 12.
    
    if hn:
        hist = ea25RecZmeas_basecomp_hn
        outname = 'Mass_hneutralssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.pdf'
    else:
        hist = ea25RecZmeas_basecomp
        outname = 'Mass_h1ssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.pdf'

    msax = 0
    h1ax = 1
    sfax = 2

    labels = {msax: r'$\log_{10} \, \Sigma_{\mathrm{gas}}\; [\mathrm{M}_{\odot} \, \mathrm{pkpc}^{-2}]$',\
              h1ax: r'$\log_{10} \, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$',\
              sfax: r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot}\, \mathrm{pkpc}^{-2} \mathrm{yr}^{-1}]$',\
              }
    
    convs = {msax: c.solar_mass / (c.cm_per_mpc / 1.e3)**2,\
             h1ax: 1.,\
             sfax: c.solar_mass / (c.cm_per_mpc / 1.e3)**2 / c.sec_per_year,\
            }
    
    lims = {msax: (4., 9.0), h1ax: (11., 23.5), sfax:(-8., 0.5)}
    
    fig = plt.figure(figsize=(10.5, 3.5))
    grid = gsp.GridSpec(1, 4, width_ratios=[1., 1., 1., 0.2], wspace=0.45, hspace=0.25, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([fig.add_subplot(grid[xi]) for xi in range(3)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[:, 3])
    
    sumaxes =  [(sfax,), (h1ax,), (msax,)]
    plotaxes = [(msax, h1ax), (msax, sfax), (h1ax, sfax)]
    vmaxs = [getminmax2d(hist, axis=i, log=True, pixdens=True)[1] for i in sumaxes]
    vmax = max(vmaxs)
    vmin = vmax - 8.
    
    cmap = 'gist_yarg'
    #colors = ['saddlebrown', 'maroon', 'red', 'orange', 'gold', 'forestgreen', 'lime', 'cyan', 'blue', 'purple', 'magenta']
    #dimlabels = (labels[0], labels[1], labels[2], labels[3])
    #dimshifts = (-1.*np.log10(convs[i]) for i in range(4))
    #edgeinds = [np.argmin(np.abs(contourbins[i] - hist['edges'][0] + np.log10(measconv))) for i in range(len(contourbins))]
    #mins = [(edge,) + (None,) * 3 for edge in [None] + edgeinds]
    #maxs = [(edge,) + (None,) * 3 for edge in edgeinds + [None]]

    percarr = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    linestyles_perc = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    colors_perc = ['magenta'] *len(percarr)

    for i in range(3):
        ax = mainaxes[i]
        sumaxis = sumaxes[i]
        plotaxis = plotaxes[i]
        plotaxis = tuple(plotaxis)
        imgminmax = add_2dplot(ax, hist, plotaxis, log=True, usepcolor=True, pixdens=True, shiftx=-1.*np.log10(convs[plotaxis[0]]), shifty=-1.*np.log10(convs[plotaxis[1]]), cmap=cmap, vmin=vmin, vmax=vmax)
        
        #for j in range(len(mins)):
        #    add_2dhist_contours(ax, hist, plotaxes, mins=mins[j], maxs=maxs[j],\
        #                        histlegend=False, fraclevels=True, levels=levels, legend=False,\
        #                        dimlabels=dimlabels, legendlabel=None, legendlabel_pre=None,\
        #                        shiftx=-1.*np.log10(convs[plotaxes[0]]), shifty=-1.*np.log10(convs[plotaxes[1]]), dimshifts=dimshifts,\
        #                        colors=[colors[j]] * len(levels), linestyles=linestyles)
        #
        #
        
        subhist = np.sum(hist['bins'], axis=sumaxis)
        if plotaxis[0] > plotaxis[1]:
            subhist=subhist.T
        percentiles = percentiles_from_histogram(subhist, hist['edges'][plotaxis[1]], axis=1, percentiles=percarr)
        cens_ax0 = hist['edges'][plotaxis[0]]
        cens_ax0 = cens_ax0[:-1] + 0.5 * np.diff(cens_ax0)
        plotcrit = np.sum(subhist, axis=1) * hist.npt >= 10.
        plotsl   = slice(np.where(plotcrit)[0][0], np.where(plotcrit)[0][-1] + 1, None)
        for pi in range(len(percentiles)):
            ax.plot(cens_ax0[plotsl] - np.log10(convs[plotaxis[0]]), percentiles[pi, plotsl]- np.log10(convs[plotaxis[1]]), linestyle=linestyles_perc[pi], color=colors_perc[pi], label='%.0f %%'%(percarr[pi] * 100.))
        
        handles_subs, labels_subs = ax.get_legend_handles_labels()
        
        ax.set_xlabel(labels[plotaxis[0]], fontsize=fontsize)
        ax.set_ylabel(labels[plotaxis[1]], fontsize=fontsize)
        setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False)
        
        
        ax.set_xlim(*lims[plotaxis[0]])
        ax.set_ylim(*lims[plotaxis[1]])
        
    add_colorbar(cax, img=imgminmax[0], vmin=vmin, vmax=vmax, cmap=cmap, \
                 clabel=r'$\log_{10}\, \mathrm{absorber\, fraction} / \mathrm{pix. size} \; [\mathrm{dex}^{-2}] $',
                 newax=False, extend='min', fontsize=fontsize, orientation='vertical')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    
    #handles_encl = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.0f %%'%(levels[i]*100.)) for i in range(len(levels))]
    handles_encl = []
    mainaxes[2].legend(handles=handles_subs + handles_encl, fontsize=fontsize - 1, ncol=1, loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False)
    
    plt.savefig('/net/luttero/data2/imgs/Zmeascomps/' + outname, format='pdf', bbox_inches='tight')

#################################################
##### O VII distributions with O VI priors ######
#################################################

def ploto7cddf_witho6priors(fontsize=fontsize, cumul=True):
    outdir = '/home/wijers/Documents/proposals/nasa_archive_max_o6_followup_o7_search/'
    if cumul:
        outname = 'coldens_o7_given_o6_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.pdf'
    else:
        outname = 'cddf_o7_given_o6_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.pdf'
    dfilen = '/net/luttero/data2/paper1/histograms.hdf5'
    datagroup = 'No6_No7'
    
    o6_minvals = [None, 12.5, 13., 13.5, 14.0]
    o6_maxvals = [12.5, 13., 13.5, 14.0, None]
    colors = ['mediumorchid', 'dodgerblue', 'green', 'orange', 'red']
    colors = {i: colors[i] for i in range(len(o6_minvals))}
    colors['all'] = 'gray'
    
    # data from '/net/luttero/data2/proc/hist_coldens_o6-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_hires-8.npz'
    with h5py.File(dfilen, 'r') as df:
        grp = df[datagroup]
        
        dimension = np.array(grp['dimension'])
        o6ax = np.where(dimension == 'NO6')[0][0]
        o7ax = np.where(dimension == 'NO7')[0][0]
        
        o6bins = np.array(grp['edges_axis%i'%o6ax])
        o7bins = np.array(grp['edges_axis%i'%o7ax])
        
        hist = np.array(grp['hist_all'])
        #nct = grp.attrs['numcounts_total']
        L_z = 6.25 # from original file name
        
        cosmopars = {key: item for (key, item) in df['cosmopars_eagle/snap27'].attrs.items()}
    
    o6sels = {'all': slice(None, None, None)}
    mininds = [np.argmin(np.abs(o6bins - val)) if val is not None else None for val in o6_minvals]
    maxinds = [np.argmin(np.abs(o6bins - val)) if val is not None else None for val in o6_maxvals]
    o6sels.update({i: slice(mininds[i], maxinds[i], None) for i in range(len(o6_minvals))}) # minimum values -> include bins starting where left edge = minmimum value
    base = [slice(None, None, None)] * len(hist.shape)
    sumaxes = range(len(hist.shape))
    sumaxes.remove(o7ax)
    o6sels = {minval: tuple(base[:o6ax] + [o6sels[minval]] + base[o6ax + 1:]) for minval in o6sels.keys()}
  
    hists = {minv: np.sum(hist[o6sels[minv]], axis=tuple(sumaxes)) for minv in o6sels.keys()} 
    
    dzpp = mc.getdz(cosmopars['z'], L_z, cosmopars=cosmopars) # histogram is already normalized to the total number of pixels
    fig = plt.figure(figsize=(4.5, 3.))
    ax = fig.add_subplot(111)
    
    if cumul:
        chists = {minv: np.cumsum(hists[minv][::-1])[::-1] for minv in hists.keys()} # sum of everything >= left edge
        plotx = o7bins[:-1]
        ax.set_ylabel(r'$\log_{10} \, \mathrm{d} n(> \mathrm{N}) \,/\, \mathrm{d} z$', fontsize=fontsize)
        ax.set_xlim(12.9, 16.5)
        ax.set_ylim(-4.7, 2.)
    else:
        chists = {minv: hists[minv] / np.diff(o7bins) for minv in hists.keys()} # sum of everything >= left edge
        plotx = o7bins[:-1] + 0.5 * np.diff(o7bins)
        ax.set_ylabel(r'$\log_{10} \, \partial^2 n \,/\, \partial \log_{10} N \, \partial z \; [\mathrm{cm}^2]$', fontsize=fontsize)
        ax.set_xlim(12.9, 16.5)
        ax.set_ylim(-4.7, 2.)
    
    setticks(ax, fontsize)
    ax.set_xlabel(r'$\log_{10} \, \mathrm{N}(\mathrm{O\, VII}) \; [\mathrm{cm}^{-2}]$', fontsize=fontsize)
   
    
    for minv in sorted(chists.keys()):
        if minv == 'all':
            label = 'all'
            color = 'gray'
        else:
            label = 'all' if o6sels[minv][o6ax].stop is None and o6sels[minv][o6ax].start is None else\
                    r'$ > %.1f$'%(o6bins[o6sels[minv][o6ax].start]) if o6sels[minv][o6ax].stop is None else\
                    r'$ < %.1f$'%(o6bins[o6sels[minv][o6ax].stop]) if o6sels[minv][o6ax].start is None else\
                    r'$%.1f \endash %.1f$'%(o6bins[o6sels[minv][o6ax].start], o6bins[o6sels[minv][o6ax].stop])
            color = colors[minv]
        ax.plot(plotx, np.max(np.array([np.log10(chists[minv] / dzpp), -100.*np.ones(len(chists[minv]))]), axis=0), color=color, label=label)
    
    legend = ax.legend(fontsize=fontsize - 1, ncol=2, loc='lower left', frameon=True)
    legend.set_title(r'$\log_{10}\, \mathrm{N}(\mathrm{O\, VI}) \; [\mathrm{cm}^{-2}]$', prop={'size': fontsize - 1})
    
    print('saving ' + outname)
    plt.savefig(outdir + outname, format='pdf', bbox_inches='tight')
    