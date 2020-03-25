#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:22:42 2020

@author: wijers

based on specwiz_proc.py: 
for analysis of a large set of specwizard sightlines in one go
adapted to add velocity shifts, and leave out some of the stuff fromm older 
projects
"""

## imports
import numpy as np
import h5py
import os
import scipy.optimize as spo
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.cm as cm

import cosmo_utils as cu
import eagle_constants_and_units as c
import ion_line_data as ild
import make_maps_opts_locs as ol

import plot_utils as pu

## defaults
sdir = ol.sdir
ldir = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/'
ndir = ol.ndir
pdir = ol.pdir
mdir = ol.mdir

# based on resolved/unresolved multiplets, strongest lines
ion_default_lines = {'o6': (ild.o6major,),\
                     'o7': (ild.o7major,),\
                     'o8': (ild.o8major, ild.o8minor),\
                     'ne8': (ild.ne8major,),\
                     'ne9': (ild.ne9major,),\
                     'fe17': (ild.fe17major,),\
                     }

# extract groups containing spectra
# spectrum files contained in groups Spectrum<number>
# loading from luttero takes a while on quasar; 
# using a local file copy does help, but it's still not fast
class SpecSet:
    def __init__(self, filename, getall=False, ionlines=ion_default_lines):
        '''
        filename: name of the file containing the specwizard output
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        getall:   get total column densities and EWs
        '''
        if filename[:len(sdir)] == sdir or '/' in filename:
            self.filename = filename
        else:
            self.filename = sdir + filename
        self.specfile = h5py.File(self.filename, 'r')
        self.specgroups = np.array(self.specfile.keys())
        self.isspec = np.array(['Spectrum' in group for group in self.specgroups])
        self.numspecs = np.sum(self.isspec)
        if not self.numspecs > 0:
            raise ValueError('The specified file {filen} contains no spectra'.format(filen=self.filename))
        self.specgroups = self.specgroups[self.isspec] 
        self.specgroups = np.array(sorted(self.specgroups,\
                                          key=lambda name: int(name[8:])))
        del self.isspec
        self.ionlines = ionlines
        
        self.cosmopars = \
            {'boxsize':     self.specfile['/Header'].attrs.get('BoxSize'),\
             'h':           self.specfile['/Header'].attrs.get('HubbleParam'),\
             'a':           self.specfile['/Header'].attrs.get('ExpansionFactor'),\
             'z':           self.specfile['/Header'].attrs.get('Redshift'),\
             'omegam':      self.specfile['/Header'].attrs.get('Omega0'),\
             'omegalambda': self.specfile['/Header'].attrs.get('OmegaLambda'),\
             'omegab':      self.specfile['/Header'].attrs.get('OmegaBaryon') }
        
        ## set general sightline and box properties
        # positions in gagdet units = 1/h cMpc
        self.positions = np.array([[
                     self.specfile[specgroup].attrs['X-position'],\
                     self.specfile[specgroup].attrs['Y-position']\
                                   ] for specgroup in self.specgroups]) 
        # convert to cMpc
        self.positions *= 1. / self.cosmopars['h']
        self.slicelength = self.cosmopars['boxsize'] / self.cosmopars['h'] #in cMpc
        # observed Delta z, since the comoving slice length is used
        self.deltaredshift_obs = cu.Hubble(self.cosmopars['z'],\
                                       cosmopars=self.cosmopars) \
                            / cu.c.c * self.slicelength * cu.c.cm_per_mpc 
        
        self.numspecpix = self.specfile['VHubble_KMpS'].shape[0]

        # set dicts to add to per ion
        self.spectra = {}
        self.tau     = {}
        self.coldens = {}
        self.EW      = {}
        self.nion    = {}
        self.posmassw= {}
        self.posionw = {}
        self.veltauw = {}
        self.tau_base     = {}
        self.base_spectra = {}     
        self.aligned = {}
        
        # if the file has no lines of sight, it's useless anyway
        self.ions = np.array(self.specfile['Spectrum0'].keys()) 
        self.ision = np.array(['RealSpaceMass' not in group for group in self.ions])
        self.ions = self.ions[self.ision]
        del self.ision
        self.aligned = {ion: False for ion in self.ions}
 
        self.dataoptions = {\
             'posmassw':\
             {'pecvel':      'RealSpaceMassWeighted/LOSPeculiarVelocity_KMpS',\
              'zmassfrac':   'RealSpaceMassWeighted/MetalMassFraction',\
              'overdensity': 'RealSpaceMassWeighted/OverDensity',\
              'temperature': 'RealSpaceMassWeighted/Temperature_K'},\
             'ion':\
             {'flux':        'Flux',\
              'logcoldens':  'LogTotalIonColumnDensity',\
              'tau':         'OpticalDepth'},\
             'posionw':\
             {'pecvel':      'RealSpaceNionWeighted/LOSPeculiarVelocity_KMpS',\
              'nion':        'RealSpaceNionWeighted/Nion_CM3',\
              'overdensity': 'RealSpaceNionWeighted/OverDensity',\
              'temperature': 'RealSpaceNionWeighted/Temperature_K'},\
             'veltauw':\
             {'pecvel':      'RealSpaceNionWeighted/LOSPeculiarVelocity_KMpS',\
              'nion':        'RealSpaceNionWeighted/Nion_CM3',\
              'overdensity': 'RealSpaceNionWeighted/OverDensity',\
              'temperature': 'RealSpaceNionWeighted/Temperature_K'}\
             }

        self.getlinedata_in()
        if getall:
            self.getall()
    
    def getall(self, dions='all'):
        self.getspectra(dions=dions)
        self.getcoldenstot(dions=dions)
        self.getEWtot(dions=dions)
        
    def getlinedata_in(self):
        # header also contains unused lines: only the first listed line is used
        # for each ion
        _ions_in = np.array(self.specfile['Header'].attrs['Ions'])
        self.ions_in, _inds = np.unique(_ions_in, return_index=True) 
        self.fosc_in = np.array(self.specfile['Header'].attrs['Transitions_Oscillator_Strength'])[_inds]
        self.lambda_in = np.array(self.specfile['Header'].attrs['Transitions_Rest_Wavelength'])[_inds]
        self.linedata_in = {self.ions_in[i]: {'lambda_rest': self.lambda_in[i],\
                                              'fosc': self.fosc_in[i]}\
                            for i in range(len(self.ions))}
                 
    def getcoldenstot(self, dions='all'):

       # take total column density from file
        if dions == 'all':
            dions = self.ions
        self.coldens.update({ion: \
                               np.array([ \
                                 self.specfile['{grpn}/{ion}/LogTotalIonColumnDensity'.format(ion=ion, grpn=specgrp)][()] \
                               for specgrp in self.specgroups])\
                             for ion in dions})
    
    def getspectra(self, dions='all'):
        '''
        dions:    ions to get EWs for
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        spectra are periodic: multiple lines here are intended for overlapping
        multiplets
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        #print(dions)
        self.velperpix_kmps = np.average(np.diff(np.array(self.specfile['VHubble_KMpS'])))
        
        for ion in dions:
            if ion not in self.tau_base.keys():
                self.getquantity('tau', 'ion', dions=[ion])
            fosc_in = self.linedata_in[ion]['fosc']
            lambda_r_in = self.linedata_in[ion]['lambda_rest'] # Angstrom units
            
            fosc_use = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambda_r_use = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            
            veldiff_at_z_kmps = [(lambda_r - lambda_r_in) \
                                  / lambda_r_in * cu.c.c * 1.e-5 
                                 for lambda_r in lambda_r_use]
            pixshifts = [int(veldiff / self.velperpix_kmps + 0.5) \
                         for veldiff in veldiff_at_z_kmps]
            rescalef = (fosc_use * lambda_r_use**2) / \
                       (fosc_in * lambda_r_in**2)
            # rescale, shift copies of tau
            if not (np.allclose(rescalef, 1.) and len(rescalef) == 1):
                tau = np.copy(self.tau_base[ion])[:, :, np.newaxis]
                tau = tau * rescalef[np.newaxis, np.newaxis, :] # cannot do in place with extra dimension
                for li in range(len(pixshifts)):
                    shift = pixshifts[li]
                    tau[:, :, li] = np.roll(tau[:, :, li], shift, axis=1)
                tau = np.sum(tau, axis=2)
            else:
                tau = np.copy(self.tau_base[ion])
            # save results
            self.tau.update({ion: tau})
            self.spectra.update({ion: np.exp(-1 * tau)})
            self.aligned[ion] = False
    
    def alignmaxtau(self, dions='all'):
        '''
        shift all spectra so max. tau is at the central pixel
        if multiple maxima have the exact same value, the first instance in 
        the default zero position is used (argmax default)
        self.centerpix is the pixel index the max values get shifted to
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        
        self.centerpix = self.numspecpix // 2
        for ion in dions:
            if ion not in self.tau:
                self.getspectra(dions=[ion])
            tau = self.tau[ion]
            maxpos = np.argmax(tau, axis=1) 
            rollargs = self.centerpix - maxpos
            for i in range(tau.shape[0]):
                tau[i, :] = np.roll(tau[i, :], rollargs[i])
            self.tau[ion] = tau
            self.spectra.update({ion: np.exp(-1 * tau)})
            self.aligned[ion] = True
            
    def getEWtot(self, dions='all'):
        '''
        dions:    ions to get EWs for
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        #print(dions)
        for ion in dions:
            if ion not in self.spectra.keys():
                self.getspectra(dions=[ion])
            
            foscs = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambdas = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            lambda_eff = np.sum(foscs  * lambdas) / np.sum(foscs)
            
            # EW = \int dlamdba (1-flux) = (Delta lambda) - (Delta lambda)/N * sum_i=1^N F_normalised(i)  
            self.EW[ion] = 1. - np.sum(self.spectra[ion], axis=1) / float(self.spectra[ion].shape[1])
            self.EW[ion] *= self.deltaredshift_obs / (self.cosmopars['z'] + 1.) * lambda_eff # convert absorbed flux fraction to rest-frame EW
    
    def getcoldens_EW_vwindow(self, deltav_rest_kmps, dions='all'):    
        '''
        dions:            ions to get EWs for
        deltav_rest_kmps: velocity interval (total = 2x maximum offset from max
                          tau). Units: rest-frame km/s
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        
        # make sure the stuff we need exists
        toalign = []
        togetcd = []
        for ion in dions:
            if not self.aligned[ion]:
                toalign.append(ion)
            if ion not in self.coldens:
                togetcd.append(ion)
        self.alignmaxtau(dions=toalign)
        self.getcoldenstot(dions=togetcd)
        
        # set up selection ranges
        numpix = int(deltav_rest_kmps / self.velperpix_kmps + 0.5)
        if numpix >= self.numspecpix:
            print('Velocity interval {vel} is larger than the box size'.format(vel=deltav_rest_kmps))
            print('using the whole sightline, without repetitions')
            vsel = slice(None, None, None)
            numpix =  self.numspecpix
        else:
            vsel = slice(self.centerpix - numpix // 2,\
                         self.centerpix - numpix // 2 + numpix,\
                         None)
        pathfrac = float(numpix) / float(self.numspecpix)
        
        # iterate over ions: select region, calculate N, EW
        vkey = deltav_rest_kmps
        if not hasattr(self, 'vwindow_EW'):
            self.vwindow_EW = {vkey: {}}
            self.vwindow_coldens = {vkey: {}}
        else:
            self.vwindow_EW.update({vkey: {}})
            self.vwindow_coldens.update({vkey: {}})
        for ion in dions:           
            foscs = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambdas = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            lambda_eff = np.sum(foscs  * lambdas) / np.sum(foscs)
            
            self.vwindow_EW[vkey][ion] = 1. - np.sum(self.spectra[ion][:, vsel], axis=1) / float(numpix)
            # convert absorbed flux fraction to rest-frame EW
            self.vwindow_EW[vkey][ion] *= pathfrac * self.deltaredshift_obs / (self.cosmopars['z'] + 1.) * lambda_eff
            
            # N \propto tau
            self.vwindow_coldens[vkey][ion] = np.sum(self.tau[ion][:, vsel], axis=1) \
                                              / np.sum(self.tau[ion][:, :], axis=1)            
            self.vwindow_coldens[vkey][ion] *= 10**self.coldens[ion]
            self.vwindow_coldens[vkey][ion] = np.log10(self.vwindow_coldens[vkey][ion])
            
    def save_specdata(self, filename):
        '''
        filename should include the directory
        
        save N, EW totals and windows, 
        as well as used ionlines, cosmopars, specwizard file
        only new vwindow and total values are added
        '''
        
        if os.path.isfile(filename):
            createnew = False
        else:
            createnew = True
        with h5py.File(filename, 'a') as fo:
            if createnew:
                hed = fo.create_group('Header')
                csm = hed.create_group('cosmopars')
                for key in self.cosmopars:
                    csm.attrs.create(key, self.cosmopars[key])
                hed.create_dataset('specgroups', data=self.specgroups.astype(np.string_))
                hed.attrs.create('filename_specwizard', np.string_(self.specfile))
                # used lines
                inl = hed.create_group('ionlines')
                for ion in self.ionlines:
                    sgp = inl.create_group(ion)
                    for li in range(len(self.ionlines[ion])):
                        s2grp = sgp.create_group('line_{}'.format(li))
                        ild.savelinedata(s2grp, self.ionlines[ion][li])
            
            if 'coldens_tot' not in fo.keys():
                grp = fo.create_group('coldens_tot')
                for ion in self.coldens:
                    grp.create_dataset(ion, data=self.coldens[ion])
                grp.attrs.create('info', np.string_('column density [log10 cm^-2] for each ion'))
            if 'EW_tot' not in fo.keys():
                grp = fo.create_group('EW_tot')
                for ion in self.EW:
                    grp.create_dataset(ion, data=self.EW[ion])
                grp.attrs.create('info', np.string_('equivalent width [Angstrom, rest-frame] for each ion'))
            
            if not hasattr(self, 'vwindow_EW'):
                # nothing left to save
                return
                
            if 'vwindows_maxtau' not in fo.keys():
                vgp = fo.create_group('vwindows_maxtau')
                vgp.attrs.create('info', np.string_('column density and EW in rest-frame velocity windows Deltav (+- 0.5 * Deltav) [km/s] around the maximum-optical-depth pixel'))
            else:
                vgp = fo['vwindows_maxtau']
            for deltav in self.vwindow_EW:
                vname = 'Deltav_{dv:.3f}'.format(dv=deltav)
                if vname in vgp.keys(): # already stored
                    continue
                svgp = vgp.create_group(vname)
                svgp.attrs.create('Deltav_rf_kmps', deltav)
                cvgp = svgp.create_group('coldens')
                cvgp.attrs.create('info', np.string_('column density [log10 cm^-2] for each ion'))
                evgp = svgp.create_group('EW')
                evgp.attrs.create('info', np.string_('equivalent width [Angstrom, rest-frame] for each ion'))
                
                for ion in self.vwindow_EW[deltav]:
                    evgp.create_dataset(ion, data=self.vwindow_EW[deltav][ion])
                    cvgp.create_dataset(ion, data=self.vwindow_coldens[deltav][ion])
        
    def getnion(self, dions='all'): # ion number density in cm^3 in each pixel: 
        if dions == 'all':
            dions = self.ions
        self.nion.update({ion: np.array([self.specfile['Spectrum%i/%s/RealSpaceNionWeighted/NIon_CM3'%(specnum,ion)][()] for specnum in range(self.numspecs)]) for ion in dions})
            
    def getspectra_base(self, dions='all'):
        if dions == 'all':
            dions = self.ions
        self.spectra_base.update({ion: np.array([self.specfile['Spectrum%i/%s/Flux'%(specnum,ion)][()] for specnum in range(self.numspecs)]) for ion in dions})


    def getquantity(self, name, cat, dions='all'):
        '''
        reads the quantity <name>, in the weighting category <cat>, into
        the appropriate dictionary for the ions <dions> (if applicable)
        '''
        if cat not in self.dataoptions.keys():
            raise ValueError('Cat options are %s, not %s. No values retrieved.'%(str(self.dataoptions.keys()), cat))
        elif name not in self.dataoptions[cat].keys():
            raise ValueError('Name options are %s, not %s. No values retrieved.'%(str(self.dataoptions[cat].keys()), name)) 

        elif cat == 'posmassw':
            if name == 'nion':
                self.getnion(dions=dions)
            else:
                self.posmassw.update({name : np.array([np.array(self.specfile['{grn}/{dn}'.format(grn=specgroup, dn=self.dataoptions[cat][name])]) \
                                                       for specgroup in self.specgroups])})
        elif cat == 'ion':
            if name == 'flux': # we have a function for this one
                self.getspectra_base(dions=dions) 
            elif name == 'logcoldens': # and for this one
                self.getcoldens(dions=dions)
            else: # name == 'tau'
                self.tau_base.update({ion: np.array([np.array(self.specfile['{grn}/{ion}/{dn}'.format(grn=specgroup, ion=ion, dn=self.dataoptions[cat][name])]) \
                                                     for specgroup in self.specgroups]) for ion in dions})
        else:
            if cat == 'posionw':
                self.basedict = self.posionw
            elif cat == 'veltauw':
                self.basedict = self.veltauw
            else:
                raise ValueError('{cat} is not a valid cat option. No values retrieved.'.format(cat=cat))
            if name not in self.basedict.keys():
                self.basedict[name] = {}
            for ion in dions:      
                self.basedict[name].update({ion : np.array([np.array(self.specfile['{grn}/{ion}/{dn}'.format(grn=specgroup, ion=ion, dn=self.dataoptions[cat][name])]) \
                             for specgroup in self.specgroups]) for ion in dions})
            del self.basedict           


def combine_sample_NEW(samples=(3, 6)):
    '''
    applies ion-selected subset selection, and merges total selections
    only adds new data
    Note: the files are not checked for compatibility
    
    adapted from specwiz_proc for v window N-EW files
    '''
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17'] # only o8 doublet is expected to be unresolved -> rest is fine to use single lines
    if samples[0] == 3:
        filen0 = '/net/luttero/data2/specwizard_data/sample3_coldens_EW_vwindows.hdf5'
        fn_samplesel0 = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'
        ionselgrpn0 = {'o7': 'file0',\
                       'o8': 'file1',\
                       'o6': 'file2',\
                       }
    if samples[1] == 6:
        filen1 = '/net/luttero/data2/specwizard_data/sample6_coldens_EW_vwindows.hdf5'
        fn_samplesel1 = '/net/luttero/data2/specwizard_data/sample6/los_sample6_ne8-ne9-fe17_L0100N1504_data.hdf5'
        ionselgrpn1 = {'ne8': 'file0',\
                       'ne9': 'file1',\
                       'fe17': 'file2',\
                       }
    if samples == (3, 6):
        outfilen = '/net/luttero/data2/specwizard_data/sample3-6_coldens_EW_vwindows_subsamples.hdf5'
    
    with h5py.File(filen0, 'r') as f0,\
         h5py.File(filen1, 'r') as f1,\
         h5py.File(outfilen, 'a') as fo:

        # Header info: copy per file
        if 'Header_sample%s'%samples[0] not in fo:
            f0.copy('Header', fo, name='Header_sample%s'%samples[0])
        if 'Header_sample%s'%samples[1] not in fo:
            f1.copy('Header', fo, name='Header_sample%s'%samples[1])
        
        # combine the full samples: compare integer pixel values -> can just check equality
        grpn_full = 'full_sample'        
        with h5py.File(fn_samplesel0, 'r') as ft:
            pixels0 = np.array(ft['Selection/selected_pixels_allions'])
        with h5py.File(fn_samplesel1, 'r') as ft:
            pixels1 = np.array(ft['Selection/selected_pixels_allions'])
        eqgrid = np.all(pixels0[:, np.newaxis, :] == pixels1[np.newaxis, :, :] , axis=2)
        keep1  = np.logical_not(np.any(eqgrid, axis=0))
        specnums_1in0part = np.array([np.where(eqgrid[i])[0][0] if np.any(eqgrid[i]) else -1 for i in range(eqgrid.shape[0])])
        specnums_1in1part = np.where(keep1)[0]
        specnums_0in0part = np.arange(eqgrid.shape[0])
        specnums_0in1part = np.array([np.where(eqgrid[:, j])[0][0] if np.any(eqgrid[:, j]) else -1 for j in range(eqgrid.shape[1])])
        print('Overlap between files: %i sightlines'%(eqgrid.shape[1] - len(specnums_1in1part)))
        
        specnums_file0 = np.append(specnums_0in0part, specnums_0in1part[keep1])
        specnums_file1 = np.append(specnums_1in0part, specnums_1in1part)
        
        if grpn_full in fo:
            grpf = fo[grpn_full]
        else:
            grpf = fo.create_group(grpn_full)
            grpf.create_dataset('specinds_sample%i'%samples[0], data=specnums_file0)
            grpf['specinds_sample%i'%samples[0]].attrs.create('info', np.string_('specind -1 means the sightline is not present in the file'))
            grpf.create_dataset('specinds_sample%i'%samples[1], data=specnums_file1)
            grpf['specinds_sample%i'%samples[1]].attrs.create('info', np.string_('specind -1 means the sightline is not present in the file'))
        
        if 'EW_tot' not in grpf:
            for ion in ions:
                ewn = 'EW_tot'
                cdn = 'coldens_tot'
                
                Ns0 = np.array(f0['{path}/{ion}'.format(path=cdn, ion=ion)])
                EW0 = np.array(f0['{path}/{ion}'.format(path=ewn, ion=ion)])
                Ns1 = np.array(f1['{path}/{ion}'.format(path=cdn, ion=ion)])
                EW1 = np.array(f1['{path}/{ion}'.format(path=ewn, ion=ion)])
                
                Ns = np.append(Ns0, Ns1[keep1])
                EW = np.append(EW0, EW1[keep1])
                
                grpf.create_dataset('{path}/{ion}'.format(path=ewn, ion=ion),\
                                          data=EW)                
                grpf.create_dataset('{path}/{ion}'.format(path=cdn, ion=ion),\
                                          data=Ns)                    
            attrs_ewn = {key: val for key, val in f0[ewn].attrs.items()}
            attrs_cdn = {key: val for key, val in f0[cdn].attrs.items()}
            for key in attrs_ewn:
                grpf['EW_tot'].attrs.create(key, attrs_ewn[key])
            for key in attrs_cdn:
                grpf['coldens_tot'].attrs.create(key, attrs_ewn[key])
                
        vwn = 'vwindows_maxtau'
        if vwn in f0 and vwn in f1:
            if vwn in grpf:
                gvw = grpf[vwn]
            else:
                gvw = grpf.create_group(vwn)
                _attrs = {key: val for key, val in f0[vwn].attrs.items()}
                for key in _attrs: # just Delta v def. info
                    gvw.attrs.create(key, _attrs[key])
                    
            vkeys0 = set(f0[vwn].keys())
            vkeys1 = set(f1[vwn].keys())
            vkeys = vkeys0 & vkeys1 # only common elements
            
            for vkey in vkeys:
                if vkey in gvw: # already copied
                    continue
                dv0 = f0['{path}/{dv}'.format(path=vwn, dv=vkey)].attrs['Deltav_rf_kmps']
                dv1 = f1['{path}/{dv}'.format(path=vwn, dv=vkey)].attrs['Deltav_rf_kmps']
                if not np.isclose(dv0, dv1): # Delta v's don't quite match
                    continue
                gdv = gvw.create_group(vkey)
                gdv.attrs.create('Deltav_rf_kmps', 0.5 * (dv0 + dv1))
                
                for ion in ions:
                    ewn = 'EW'
                    cdn = 'coldens'
                    
                    Ns0 = np.array(f0['{path}/{dv}/{qty}/{ion}'.format(path=vwn, qty=cdn, dv=vkey, ion=ion)])
                    EW0 = np.array(f0['{path}/{dv}/{qty}/{ion}'.format(path=vwn, qty=ewn, dv=vkey, ion=ion)])
                    Ns1 = np.array(f1['{path}/{dv}/{qty}/{ion}'.format(path=vwn, qty=cdn, dv=vkey, ion=ion)])
                    EW1 = np.array(f1['{path}/{dv}/{qty}/{ion}'.format(path=vwn, qty=ewn, dv=vkey, ion=ion)])
                    
                    Ns = np.append(Ns0, Ns1[keep1])
                    EW = np.append(EW0, EW1[keep1])
                    
                    gdv.create_dataset('{qty}/{ion}'.format(qty=ewn, ion=ion),\
                                              data=EW)                   
                    gdv.create_dataset('{qty}/{ion}'.format(qty=cdn, ion=ion),\
                                              data=Ns)
                    
                attrs_ewn = {key: val for key, val in f0['{path}/{dv}/{qty}'.format(path=vwn, qty=ewn, dv=vkey)].attrs.items()}
                attrs_cdn = {key: val for key, val in f0['{path}/{dv}/{qty}'.format(path=vwn, qty=cdn, dv=vkey)].attrs.items()}
                for key in attrs_ewn:
                    gdv[ewn].attrs.create(key, attrs_ewn[key])
                for key in attrs_cdn:
                    gdv[cdn].attrs.create(key, attrs_cdn[key]) 
        
        # ion-selected samples
        for ion in ions:
            iname = '{ion}_selection'.format(ion=ion)
            if ion in ionselgrpn0:
                ionselgrpn = ionselgrpn0[ion]
                fn_samplesel = fn_samplesel0 
                fi = f0
                pix_all = pixels0
            elif ion in ionselgrpn1:
                ionselgrpn = ionselgrpn1[ion]
                fn_samplesel = fn_samplesel1 
                fi = f1
                pix_all = pixels1
            else:
                print('No selection recorded for ion {ion}; skipping'.format(ion=ion))
                continue
            
            with h5py.File(fn_samplesel, 'r') as fs:
                _g = fs['Selection/{ign}'.format(ign=ionselgrpn)]
                # simple check: not bulletproof, but should catch simple all-gas map mismatches
                mapfile = _g.attrs['filename'].decode()
                mapfile = mapfile.split('/')[-1]
                mapparts = mapfile.split('_')
                if ion not in mapparts:
                    print('Skippping ion {ion}: map file {mapfile} seems to be wrong'.format(ion=ion, mapfile=mapfile))
                    continue
                pix_sub = np.array(_g['selected_pixels_thision'])
            
            # indices in pix_all where both coordinates match some pixel in pix_sub
            eqgrid = np.any(np.all(pix_sub[:, np.newaxis, :] == pix_all[np.newaxis, :, :] , axis=2), axis=0)
            subinds = np.where(eqgrid)
            print('For ion {ion}: {num} sightlines'.format(ion=ion, num=len(subinds[0])))
            
            if iname in fo:
                grpi = fo[iname]
            else:
                grpi = fo.create_group(iname)
                grpi.attrs.create('selection_filename', np.string_(fn_samplesel))
                grpi.attrs.create('selection_groupname', np.string_(ionselgrpn))
                grpi.create_dataset('selected_specinds', data=subinds[0])
            
            if 'EW_tot' not in grpi:
                for ion in ions:
                    ewn = 'EW_tot'
                    cdn = 'coldens_tot'
                    
                    Ns = np.array(fi['{path}/{ion}'.format(path=cdn, ion=ion)])[subinds]
                    EW = np.array(fi['{path}/{ion}'.format(path=ewn, ion=ion)])[subinds]
                    
                    grpi.create_dataset('{path}/{ion}'.format(path=ewn, ion=ion),\
                                              data=EW)                
                    grpi.create_dataset('{path}/{ion}'.format(path=cdn, ion=ion),\
                                              data=Ns)                    
                attrs_ewn = {key: val for key, val in fi[ewn].attrs.items()}
                attrs_cdn = {key: val for key, val in fi[cdn].attrs.items()}
                for key in attrs_ewn:
                    grpi['EW_tot'].attrs.create(key, attrs_ewn[key])
                for key in attrs_cdn:
                    grpi['coldens_tot'].attrs.create(key, attrs_ewn[key])
                    
            vwn = 'vwindows_maxtau'
            if vwn in fi:
                if vwn in grpi:
                    gvw = grpi[vwn]
                else:
                    gvw = grpi.create_group(vwn)
                    _attrs = {key: val for key, val in fi[vwn].attrs.items()}
                    for key in _attrs: # just Delta v def. info
                        gvw.attrs.create(key, _attrs[key])
                        
                vkeys = set(fi[vwn].keys()) 
                for vkey in vkeys:
                    if vkey in gvw: # already copied
                        continue
                    dv = fi['{path}/{dv}'.format(path=vwn, dv=vkey)].attrs['Deltav_rf_kmps']
                    gdv = gvw.create_group(vkey)
                    gdv.attrs.create('Deltav_rf_kmps', dv)
                    
                    for ion in ions:
                        ewn = 'EW'
                        cdn = 'coldens'
                        
                        Ns = np.array(fi['{path}/{dv}/{qty}/{ion}'.format(path=vwn, qty=cdn, dv=vkey, ion=ion)])[subinds]
                        EW = np.array(fi['{path}/{dv}/{qty}/{ion}'.format(path=vwn, qty=ewn, dv=vkey, ion=ion)])[subinds]
                        
                        gdv.create_dataset('{qty}/{ion}'.format(qty=ewn, ion=ion),\
                                                  data=EW)                   
                        gdv.create_dataset('{qty}/{ion}'.format(qty=cdn, ion=ion),\
                                                  data=Ns)
                        
                    attrs_ewn = {key: val for key, val in fi['{path}/{dv}/{qty}'.format(path=vwn, qty=ewn, dv=vkey)].attrs.items()}
                    attrs_cdn = {key: val for key, val in fi['{path}/{dv}/{qty}'.format(path=vwn, qty=cdn, dv=vkey)].attrs.items()}
                    for key in attrs_ewn:
                        gdv[ewn].attrs.create(key, attrs_ewn[key])
                    for key in attrs_cdn:
                        gdv[cdn].attrs.create(key, attrs_cdn[key]) 
            
def fitbpar(datafile, vwindow=None,\
            ions=('o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17'),\
            fitlogEW=True,\
            samplegroup=None):
    '''
    fit b parameters to the merged and selected N-EW file sets
    '''
    ionls = {'o8': ild.o8doublet,\
             'o7': ild.o7major,\
             'o6': ild.o6major,\
             'ne8': ild.ne8major,\
             'ne9': ild.ne9major,\
             'fe17': ild.fe17major,\
             }
    bstart = 100. * 1e5 # cm/s
    
    with h5py.File(datafile, 'r') as df:
        if samplegroup is None:
            samplegroup = 'full_sample/'
        elif samplegroup[-1] != '/':
            samplegroup = samplegroup + '/'
            
        if vwindow is None:
            epath = 'EW_tot/'
            cpath = 'coldens_tot/'
        else:
            spath = 'vwindows_maxtau/Deltav_{dv:.3f}/'.format(dv=vwindow)
            epath = spath + 'EW/'
            cpath = spath + 'coldens/'
        epath = samplegroup + epath
        cpath = samplegroup + cpath
        
        coldens = {}
        EWs = {}
        for ion in ions:
            #try:
            coldens[ion] = np.array(df[cpath + ion])
            EWs[ion] = np.array(df[epath + ion])
            #except KeyError as err:
            #    print('For vwindow {vw}, ion {ion}, samplegroup {sg}'.format(vw=vwindow, ion=ion, sg=samplegroup))
            #    print('Tried paths {pc}, {pe}'.format(pc=cpath + ion,\
            #          pe=epath + ion))
            #    raise err
    res = {}    
    for ion in ions:
        print('fitting ion {ion}'.format(ion=ion))
        N = 10**coldens[ion]
        if fitlogEW:
            EW = np.log10(EWs[ion])
        else:
            EW = EWs[ion]
        
        _ion = ionls[ion]
        def lossfunc(b):
            if b <= 0.: # back off from that area!
                return np.inf
            EWres = ild.linflatcurveofgrowth_inv_faster(N, b, _ion)
            if fitlogEW:
                EWres = np.log10(EWres)
            return np.sum((EWres - EW)**2)
        
        optres = spo.minimize(lossfunc, x0=bstart, method='COBYLA', tol=1e4,\
                              options={'rhobeg': 2e6})
         
        if optres.success:
            bfit = optres.x * 1e-5 # to km/s
            res[ion] = bfit
            print('Best fit for {ion}: {fit}'.format(ion=ion, fit=bfit))
        else:
            print('b parameter fitting failed:')
            print(optres.message)
            return optres
    return res

def fitbpar_paper2():
    '''
    call fitpar for some interesting paper 2 values 
    
    (the overhead from file read-in is minimal compared to the fitting itself)
    '''
    datafile = '/net/luttero/data2/specwizard_data/sample3-6_coldens_EW_vwindows_subsamples.hdf5'
    outfile = '/net/luttero/data2/paper2/bparfit_data.txt'
    
    # to test the general dependence on window size
    vwindows_all = [50.] + list(np.arange(100, 6400, 100.)) + [None]
    vwindows_ion = {'o6': 600.,\
                    'ne8': 600.,\
                    'o7': 1000.,\
                    'o8': 1000.,\
                    'fe17': 1000.,\
                    'ne9': 1000,\
                    }
    vwindows_ion = {ion: [vwindows_ion[ion]] +  vwindows_all \
                    if vwindows_ion[ion] not in vwindows_all else\
                    vwindows_all \
                    for ion in vwindows_ion}
    samplegroups_ion = {ion: ['{ion}_selection'.format(ion=ion)]\
                              for ion in vwindows_ion}
    
    fillstring = '{ion}\t{dv}\t{selection}\t{EWlog}\t{fitval}\n'
    topstring = fillstring.format(ion='ion', dv='Delta v [full, rest-frame, km/s]',\
                                  selection='sightline selection',\
                                  EWlog='fit log EW',\
                                  fitval='best-fit b [km/s]')
    with open(outfile, 'w') as fo:
        fo.write(topstring)
        for ion in vwindows_ion:
            vwindows = vwindows_ion[ion]
            samplegroups = samplegroups_ion[ion]
            for vwindow in vwindows:
                for samplegroup in samplegroups:
            
                    res_log = fitbpar(datafile, vwindow=vwindow,\
                          ions=[ion],\
                          fitlogEW=True,\
                          samplegroup=samplegroup)
                    res_lin = fitbpar(datafile, vwindow=vwindow,\
                          ions=[ion],\
                          fitlogEW=False,\
                          samplegroup=samplegroup)
                    
                    if vwindow is None:
                        fvwindow = np.inf
                    else:
                        fvwindow = vwindow
                    fo.write(fillstring.format(ion=ion, dv=fvwindow,\
                                               selection=samplegroup,\
                                               EWlog=True,\
                                               fitval=res_log[ion]))
                    fo.write(fillstring.format(ion=ion, dv=fvwindow,\
                                               selection=samplegroup,\
                                               EWlog=False,\
                                               fitval=res_lin[ion])) 

def plotbpar_paper2():
    datafile = '/net/luttero/data2/paper2/bparfit_data.txt'
    outname = '/net/luttero/data2/specwizard_data/sample3-6_vwindows_effect_bparfit.pdf'
    z =  0.10063854175996956
    boxvel = 100. * c.cm_per_mpc / (1. + z) * cu.Hubble(z) * 1e-5
    set_boxvel = 0.5 * boxvel
    
    data = pd.read_csv(datafile, sep='\t', header=0)
    
    ioncolors = {'o7': 'C3',\
                 'o8': 'C0',\
                 'o6': 'C2',\
                 'ne8': 'C1',\
                 'ne9': 'C9',\
                 'hneutralssh': 'C6',\
                 'fe17': 'C4'}
    vwindows_ion = {'o6': 600.,\
                    'ne8': 600.,\
                    'o7': 1000.,\
                    'o8': 1000.,\
                    'fe17': 1000.,\
                    'ne9': 1000,\
                    }
    
    fontsize = 12
    
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$\\Delta v [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$ (half of total range)', fontsize=fontsize)
    ax.set_ylabel('best-fit $b [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$', fontsize=fontsize)
    size_allsel = 10
    size_ionsel = 30
    marker_log = '*'
    marker_lin = 'o'
    alpha = 0.3
    
    for ind in data.index:
        _data = data.iloc[ind]
        ion = _data['ion']
        color = ioncolors[ion]
        if _data['fit log EW']:
            marker = marker_log
        else:
            marker = marker_lin
        if ion in _data['sightline selection']:
            size = size_ionsel
        else:
            size = size_allsel
            
        dv  = _data['Delta v [full, rest-frame, km/s]'] * 0.5
        if dv == np.inf:
            dv = set_boxvel
        bfit = _data['best-fit b [km/s]']
        
        ax.scatter([dv], [bfit], c=color, s=size, marker=marker, alpha=alpha)
        
        if size == size_ionsel:
            if dv == vwindows_ion[ion] * 0.5:
                if marker == marker_lin:
                    print('{ion:4} lin fit pm {dv:4.0f} km/s: {bfit:3.0f}'.format(ion=ion, dv=dv, bfit=bfit))
                else:
                    print('{ion:4} log fit pm {dv:4.0f} km/s: {bfit:3.0f}'.format(ion=ion, dv=dv, bfit=bfit))
            elif dv == set_boxvel and marker == marker_log:
                print('{ion:4} log fit pm {dv:4.0f} km/s: {bfit:3.0f}'.format(ion=ion, dv=0.5 * boxvel, bfit=bfit))
            
    # legend
    leg_markers = [mlines.Line2D([], [], color='gray', linestyle='None',\
                          markersize= 0.2 * size, label=label, marker=marker)  \
                   for marker, size, label in \
                   zip([marker_log,  marker_lin],\
                       [size_ionsel, size_ionsel],\
                       ['log fit', 'lin. fit'])\
                   ]
    leg_ions = [mlines.Line2D([], [], color=ioncolors[ion], linestyle='None',\
                          markersize = 0.1 * size_ionsel,\
                          label=ion, marker='o')  \
                for ion in ioncolors]
    xlim = list(ax.get_xlim())
    if xlim[0] < 0.:
        xlim[0] = 0.
    ax.set_xlim(*tuple(xlim))
    
    xticks = ax.get_xticks()
    if set_boxvel in xticks: # just change the label
        xlabels = ['{:.0f}'.format(tick) for tick in xticks]
        xlabels[np.where(xticks == set_boxvel)[0][0]] = '100 cMpc'
    else: # insert the label, remove neighboring ticks
        insind = np.searchsorted(xticks, set_boxvel)
        xticks = np.array(list(xticks[:insind - 1]) + [set_boxvel] + list(xticks[insind + 1: ]))
        xlabels = ['{:.0f}'.format(tick) for tick in xticks]
        xlabels[insind - 1] = '100 cMpc'
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)    
    
    ax.legend(handles=leg_markers + leg_ions, ncol=3, fontsize=fontsize - 1)

    plt.savefig(outname, format='pdf', bbox_inches='tight')         

def plot_NEW(specset, ions, vwindows=None, savename=None):

    alpha = 0.2    
    percentiles = (10., 50., 90.)
    
    xlabel = '$\\log_{{10}}\\, \\mathrm{{N}}_{{\\mathrm{{{ion}}}}} \\; [\\mathrm{{cm}}^{{-2}}]$'
    ylabel = '$\\mathrm{{EW}}  \; [\\mathrm{{\\AA}}]$'
    fontsize = 12
    
    numions = len(ions)
    numcols = min(3, numions)
    numrows = (numions - 1) // numcols + 1
    
    panelheight = 3.
    panelwidth = 3.
    hspace = 0.3
    wspace = 0.3
    figw = panelwidth * numcols + hspace * (numcols - 1)
    figh = panelheight * numrows + wspace * (numrows - 1)
    
    
    fig = plt.figure(figsize=(figw, figh))
    grid = gsp.GridSpec(nrows=numrows, ncols=numcols, hspace=hspace,\
                        wspace=wspace)    
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(numions)]

    if vwindows is None:
        dolegend = False
        vwindows = [None]
    else:
        dolegend = True
            
    for ii in range(numions):
        ion = ions[ii]
        ax = axes[ii]
        
        if ii % numcols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if numions - ii <= numcols:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.set_yscale('log')
        
        ax.text(0.95, 0.05, ild.getnicename(ion), fontsize=fontsize,\
                horizontalalignment='right', verticalalignment='bottom',\
                transform=ax.transAxes)
        
        minc = np.inf
        maxc = -np.inf
        for vi in range(len(vwindows)):
            deltav = vwindows[vi]
            color = 'C{}'.format(vi%10)
            
            if deltav is None:
                lhandle = '{_len:.0f} cMpc'.format(_len=specset.slicelength)
                coldens = specset.coldens[ion]
                EW      = specset.EW[ion]
            else:
                lhandle = '${dv:.0f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'.format(dv=deltav)
                coldens = specset.vwindow_coldens[deltav][ion]
                EW      = specset.vwindow_EW[deltav][ion]
            #ax.scatter(coldens, EW, label=lhandle, alpha=alpha, s=5)
            cmin = np.min(coldens)
            cmax = np.max(coldens)
            cbinsize = 0.1
            cbmin = np.floor(cmin / cbinsize) * cbinsize
            cbmax = np.ceil(cmax / cbinsize) * cbinsize
            cbins = np.arange(cbmin, cbmax + 0.5 * cbinsize, cbinsize)
            cbincen = np.append(cbins[0] - 0.5 * cbinsize, cbins + 0.5 * cbinsize)
            cperc, coutliers, cmincount = pu.get_perc_and_points(\
                        coldens, EW, cbins,\
                        percentiles=percentiles,\
                        mincount_x=20,\
                        getoutliers_y=False, getmincounts_x=True,\
                        x_extremes_only=True)  
            
            numperc = len(percentiles)
            mcc = slice(cmincount[0], cmincount[-1] + 1, 1)
            #print(cperc.shape)
            #print(cbincen.shape)
            for i in range(numperc // 2):
                p1 = cperc[:, i][mcc]
                p2 = cperc[:, numperc - 1 - i][mcc]
                ax.fill_between(cbincen[mcc], p1, p2,\
                                  alpha=alpha, color=color)
            if bool(numperc % 2):
                i = numperc // 2                
                m = cperc[:, i][mcc]
                ax.plot(cbincen[mcc], m, color=color, label=lhandle)
            ax.scatter(coutliers[0], coutliers[1], color=color, alpha=alpha,\
                       s=3) 
            maxc = max(np.max(coldens), maxc)
            minc = min(np.min(coldens), minc)
        cvals = 10**np.linspace(minc, maxc, 100)
        EWvals = ild.lingrowthcurve_inv(cvals, specset.ionlines[ion])
        ax.plot(np.log10(cvals), EWvals, linestyle='dashed', linewidth=2,\
                color='black', label='opt. thin')         
        if dolegend:
            ax.legend(fontsize=fontsize - 1., loc='upper left',\
                      bbox_to_anchor=(0.0, 1.0), frameon=False)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_NEW_fracs(specset, ions, vwindows=None, savename=None):
    
    alpha = 0.3    
    percentiles = (2.5, 10., 50., 90., 97.5)
    
    xlabel_c = '$\\log_{{10}}\\, \\mathrm{{N}}(\\mathrm{{{ion}}}) \\; [\\mathrm{{cm}}^{{-2}}]$'
    xlabel_e = '$\\mathrm{{EW}}(\\mathrm{{{ion}}})  \; [\\mathrm{{\\AA}}]$'
    ylabel = '$\\Delta v$ / tot.'
    fontsize = 11
    
    numions = len(ions)
    numcols = min(3, numions)
    numrows = ((numions - 1) // numcols + 1) * 2
    
    panelheight = 1.5
    panelwidth = 3.
    hspace = 0.25
    wspace = 0.2
    figw = panelwidth * numcols + hspace * (numcols - 1)
    figh = panelheight * numrows + wspace * (numrows - 1)
    
    
    fig = plt.figure(figsize=(figw, figh))
    grid = gsp.GridSpec(nrows=numrows, ncols=numcols, hspace=hspace,\
                        wspace=wspace)    
    axes = [[fig.add_subplot(grid[2 * (ii // numcols) + ti, ii % numcols]) for ti in range(2)] for ii in range(numions)]

    if vwindows is None:
        dolegend = False
        vwindows = [None]
    else:
        dolegend = True
            
    for ii in range(numions):
        ion = ions[ii]
        ax_c = axes[ii][0]
        ax_e = axes[ii][1]
        
        if ii % numcols == 0:
            ax_c.set_ylabel(ylabel, fontsize=fontsize)
            ax_e.set_ylabel(ylabel, fontsize=fontsize)

        ax_c.text(0.05, 0.95, xlabel_c.format(ion=ild.getnicename(ion, mathmode=True)),\
                  fontsize=fontsize, transform=ax_c.transAxes,\
                  horizontalalignment='left', verticalalignment='top')
        ax_e.text(0.05, 0.95, xlabel_e.format(ion='\\mathrm{{{}}}'.format(ild.getnicename(ion, mathmode=True))),\
                  fontsize=fontsize, transform=ax_e.transAxes,\
                  horizontalalignment='left', verticalalignment='top')
        ax_c.tick_params(which='both', direction='in', top=True, right=True,\
                         labelsize=fontsize - 1)
        ax_c.minorticks_on()
        ax_c.set_yscale('log')
        ax_e.tick_params(which='both', direction='in', top=True, right=True,\
                         labelsize=fontsize - 1)
        ax_e.minorticks_on()
        ax_e.set_yscale('log')
        ax_e.set_xscale('log')
        
        #ax_e.text(0.95, 0.95, ild.getnicename(ion), fontsize=fontsize,\
        #          horizontalalignment='right', verticalalignment='top',\
        #          transform=ax_e.transAxes)
        
        base_c = specset.coldens[ion]
        base_e = specset.EW[ion]
        for vi in range(len(vwindows)):
            deltav = vwindows[vi]
            color = 'C{}'.format(vi%10)
    
            if deltav is None:
                lhandle = '{_len:.1f} cMpc'.format(_len=specset.slicelength)
                coldens = specset.coldens[ion]
                EW      = specset.EW[ion]
            else:
                lhandle = '${dv:.1f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'.format(dv=deltav)
                coldens = specset.vwindow_coldens[deltav][ion]
                EW      = specset.vwindow_EW[deltav][ion]
                
            cdata =  10**(coldens - base_c)
            edata = EW / base_e
            #ax_c.scatter(base_c, 10**(coldens - base_c), label=lhandle, alpha=alpha, s=5)
            #ax_e.scatter(base_e, EW / base_e, label=lhandle, alpha=alpha, s=5)
            
            cmin = np.min(coldens)
            cmax = np.max(coldens)
            cbinsize = 0.1
            cbmin = np.floor(cmin / cbinsize) * cbinsize
            cbmax = np.ceil(cmax / cbinsize) * cbinsize
            cbins = np.arange(cbmin, cbmax + 0.5 * cbinsize, cbinsize)
            cbincen = np.append(cbins[0] - 0.5 * cbinsize, cbins + 0.5 * cbinsize)
            cperc, coutliers, cmincount = pu.get_perc_and_points(\
                        coldens, cdata, cbins,\
                        percentiles=percentiles,\
                        mincount_x=10,\
                        getoutliers_y=True, getmincounts_x=True,\
                        x_extremes_only=True)  
            
            EW = np.log10(EW)
            emin = np.min(EW)
            emax = np.max(EW)
            ebinsize = 0.1
            ebmin = np.floor(emin / ebinsize) * ebinsize
            ebmax = np.ceil(emax / ebinsize) * ebinsize
            ebins = np.arange(ebmin, ebmax + 0.5 * ebinsize, ebinsize)
            ebincen = np.append(ebins[0] - 0.5 * ebinsize, ebins + 0.5 * ebinsize)
            eperc, eoutliers, emincount = pu.get_perc_and_points(\
                        EW, edata, ebins,\
                        percentiles=percentiles,\
                        mincount_x=10,\
                        getoutliers_y=True, getmincounts_x=True,\
                        x_extremes_only=True) 
            EW = 10**EW
            ebincen = 10**ebincen
            
            numperc = len(percentiles)
            mcc = slice(cmincount[0], cmincount[-1] + 1, 1)
            mce = slice(emincount[0], emincount[-1] + 1, 1)
            #print(cperc.shape)
            #print(cbincen.shape)
            for i in range(numperc // 2):
                p1 = cperc[:, i][mcc]
                p2 = cperc[:, numperc - 1 - i][mcc]
                ax_c.fill_between(cbincen[mcc], p1, p2,\
                                  alpha=alpha, color=color)
                
                p1 = eperc[:, i][mce]
                p2 = eperc[:, numperc - 1 - i][mce]
                ax_e.fill_between(ebincen[mce], p1, p2,\
                                  alpha=alpha, color=color)
            if bool(numperc % 2):
                i = numperc // 2
                
                m = cperc[:, i]
                ax_c.plot(cbincen, m, color=color)
                
                m = eperc[:, i]
                ax_e.plot(ebincen, m, color=color)
            
            ax_c.scatter(coutliers[0], coutliers[1], label=lhandle,\
                         color=color, s=3, alpha=alpha)
            ax_e.scatter(10**eoutliers[0], eoutliers[1], label=lhandle,\
                         color=color, s=3, alpha=alpha)
            
        ax_c.axhline(1., linestyle='dashed', linewidth=2,\
                color='black', label=None)  
        ax_e.axhline(1., linestyle='dashed', linewidth=2,\
                color='black', label=None) 
        if dolegend:
            ax_c.legend(fontsize=fontsize - 1, loc='lower right',\
                      bbox_to_anchor=(1.0, 0.0), frameon=True)
        fig.suptitle('percentiles: {}'.format(percentiles), fontsize=fontsize)
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
        
def plot_N_EW_growth(ion, numlines=100):
    '''
    try to see is the large Delta v `jump' is caused by N or EW features; if so
    there's a bug somewhere
    '''
    fontsize = 12
    datadir = '/net/luttero/data2/paper2/'
    mdir = '/net/luttero/data2/specwizard_data/bugcheck_bpar_deltav/'
    outname = mdir + 'bugcheck_effect_deltav_{ion}_coldens_EW_sample3-6_ionselsamples_L0100N1504_27_T4EOS.pdf'.format(ion=ion)
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

    samplegroups_ion = {ion: '{ion}_selection'.format(ion=ion) \
                              for ion in ions}
    
    with h5py.File(datafile, 'r') as df:
        cosmopars = {key: val for key, val in df['Header_sample3/cosmopars'].attrs.items()}
        boxvel = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * \
                 c.cm_per_mpc * \
                 cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1e-5
        coldens = {}
        EWs = {}
        
        samplegroup = samplegroups_ion[ion] + '/'
        vgrp = df[samplegroup + 'vwindows_maxtau']
        vkeys = list(vgrp.keys())
        subsample = slice(None, None, None)
        for vkey in [None] + vkeys:
            if vkey is None:
                epath = 'EW_tot/'
                cpath = 'coldens_tot/'
                dv = boxvel
            else:
                spath = 'vwindows_maxtau/{vk}/'.format(vk=vkey)
                epath = spath + 'EW/'
                cpath = spath + 'coldens/'
                dv = df[samplegroup + spath[:-1]].attrs['Deltav_rf_kmps']
            epath = samplegroup + epath
            cpath = samplegroup + cpath
            
            coldens[dv] = np.array(df[cpath + ion])[subsample]
            EWs[dv] = np.array(df[epath + ion])[subsample]
            
            if vkey is None: 
                # set up subsampling based on sample size
                numtot = len(coldens[dv])
                if numtot > numlines:
                    subsample = np.random.choice(numtot, size=numlines,\
                                                 replace=False)                   
                    coldens[dv] = np.array(df[cpath + ion])[subsample]
                    EWs[dv] = np.array(df[epath + ion])[subsample]
    
    dvs = sorted(coldens.keys())
    _cds = np.array([coldens[dv] for dv in dvs])
    _ews = np.array([EWs[dv] for dv in dvs])
    
    ylabel_e = r'$\log_{10} \, \mathrm{EW} \; [\mathrm{m\AA}]$'
    ylabel_n = r'$\log_{10} \, \mathrm{N} \; [\mathrm{cm}^{-2}]$'
    xlabel   = r'$\Delta \, v \; [\mathrm{km}\, \mathrm{s}^{-1}]$'
    title = 'growth of N, EW with $\\Delta v$ for {ion}'.format(\
                        ion=ild.getnicename(ion, mathmode=False),\
                        )
    panelwidth = 3.5
    panelheight = 2.8
    wspace = 0.25
    hspace = 0.0
    ncols = 1
    nrows = 2
    
    fig = plt.figure(figsize=(panelwidth * ncols + wspace * (ncols - 1),\
                              panelheight * nrows + hspace * (nrows - 1)))
    grid = gsp.GridSpec(ncols=ncols, nrows=nrows, wspace=wspace, hspace=hspace)   
    nax = fig.add_subplot(grid[0, 0])
    eax = fig.add_subplot(grid[1, 0])
    
    nax.set_ylabel(ylabel_n, fontsize=fontsize)
    eax.set_ylabel(ylabel_e, fontsize=fontsize)
    eax.set_xlabel(xlabel, fontsize=fontsize)
    fig.suptitle(title, fontsize=fontsize)
    
    nax.plot(dvs, _cds, linewidth=1., alpha=0.5)
    eax.plot(dvs, np.log10(_ews) + 3., linewidth=1., alpha=0.5)

    plt.savefig(outname, format='pdf', bbox_inches='tight')    

    
def plot_absorber_locations(ion):
    '''
    seems like there's a set of jumps at specific column densities for these 
    ions -> check if they're probing the same structure
    '''
    fontsize = 12
    datadir = '/net/luttero/data2/specwizard_data/'
    mdir = '/net/luttero/data2/specwizard_data/bugcheck_bpar_deltav/'
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
    jumpdN  = {'o6':   0.2,\
               'o7':   0.2,\
               'o8':   0.3,\
               'ne8':  0.2,\
               'ne9':  0.2,\
               'fe17': 0.2,\
               }

    samplegroups_ion = {ion: '{ion}_selection'.format(ion=ion) \
                              for ion in ions}
    
    with h5py.File(datafile, 'r') as df:
        cosmopars = {key: val for key, val in df['Header_sample3/cosmopars'].attrs.items()}
        boxvel = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * \
                 c.cm_per_mpc * \
                 cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1e-5
        boxsize = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] 
        coldens = {}
        EWs = {}
        
        samplegroup = samplegroups_ion[ion] + '/'
        vgrp = df[samplegroup + 'vwindows_maxtau']
        vkeys = list(vgrp.keys())
        subsample = slice(None, None, None)
        
        sgrp = df[samplegroup[:-1]]
        selfilen = sgrp.attrs['selection_filename'].decode()
        selfnum  = sgrp.attrs['selection_groupname'].decode()
        
        for vkey in [None] + vkeys:
            if vkey is None:
                epath = 'EW_tot/'
                cpath = 'coldens_tot/'
                dv = boxvel
            else:
                spath = 'vwindows_maxtau/{vk}/'.format(vk=vkey)
                epath = spath + 'EW/'
                cpath = spath + 'coldens/'
                dv = df[samplegroup + spath[:-1]].attrs['Deltav_rf_kmps']
            epath = samplegroup + epath
            cpath = samplegroup + cpath
            
            coldens[dv] = np.array(df[cpath + ion])[subsample]
            EWs[dv] = np.array(df[epath + ion])[subsample]
    
    with h5py.File(selfilen, 'r') as sf:
        pixels = np.array(sf['Selection/{fn}/selected_pixels_thision'.format(fn=selfnum)]).astype(np.float32)
        coords = (pixels + 0.5) / 32000. * boxsize
    
    v1 = boxvel
    v2 = 5000.
    deltaN = jumpdN[ion]
    jumpsel = coldens[v1] - coldens[v2] > deltaN
    others  = np.logical_not(jumpsel)
        
    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(ncols=2, nrows=1, wspace=0.05, width_ratios=[5., 0.5])   
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    
    ax.set_xlabel('X [cMpc]', fontsize=fontsize)
    ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
    title = '{ion}: sightlines with $ \\log_{{10}} \\mathrm{{N}}({v1:.0f}) - \\log_{{10}} \\mathrm{{N}}({v2:.0f}) > {deltaN}$'.format(\
             ion=ild.getnicename(ion), v1=v1, v2=v2, deltaN=deltaN)
    fig.suptitle(title, fontsize=fontsize)
    
    cmap = cm.get_cmap('viridis')
    cdvals = coldens[boxvel]
    cdmin = np.min(cdvals)
    cdmax = np.max(cdvals)
    cdnorm = (cdvals - cdmin) / (cdmax - cdmin)
    
    ax.scatter(*tuple(coords[others].T), c=cdnorm[others], s=10, marker='o',\
               alpha=0.3, edgecolor='none')
    ax.scatter(*tuple(coords[jumpsel].T), c=cdnorm[jumpsel], s=50, marker='*',\
               alpha=0.5, edgecolor='black')
    
    norm = mpl.colors.Normalize(vmin=cdmin, vmax=cdmax)
    tomap = cm.ScalarMappable(norm=norm, cmap=cmap)
    tomap.set_array(np.linspace(cdmin, cdmax, 2))
    cbar = plt.colorbar(tomap, cax=cax, orientation='vertical', extend='neither')
    #cbar.set_aspect(10.)
    cax.set_ylabel('$\\log_{{10}} \\, \\mathrm{{N}} \\; [\\mathrm{{cm}}^{{-2}}]$, 100 cMpc', fontsize=fontsize)

    outname = mdir + 'bugcheck_effect_jump_locations_{ion}_{v1:.0f}-{v2:.0f}-{deltaN}_coldens_EW_sample3-6_ionselsamples_L0100N1504_27_T4EOS.pdf'.format(ion=ion, v1=v1, v2=v2, deltaN=deltaN)
    plt.savefig(outname, format='pdf', bbox_inches='tight')    
    
    
def fitbpar_jumpeffect():
    '''
    fit b parameters to subsets of the sightlines to see what's driving the 
    'jump' across the simulation box from max tau
    
    save as hdf5 -> save subsample N, EW data and indices
    '''
    outfile = '/net/luttero/data2/specwizard_data/bugcheck_bpar_deltav/' + \
                      'bparfits_sightline_subsamples_set2.hdf5'
                      
    datafile = '/net/luttero/data2/specwizard_data/sample3-6_coldens_EW_vwindows_subsamples.hdf5'
    vwindows = list(np.arange(3400., 6400., 200.)) + [None]
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    fitlogEW = True
    samplegroups = {ion: '{ion}_selection'.format(ion=ion) for ion in ions}
    
    # jump identification criteria:
    jv1 = 5000.
    jv2 = 6200.
    jdns = [0.025, 0.05, 0.1, 0.2, 0.4] # Delta log10 N
    # sightline selection general
    colmin = {'o6': 13.,\
              'o7': 15.,\
              'o8': 15.5,\
              'ne8': 14.,\
              'ne9': 15.,\
              'fe17': 14.5,\
              } 
    ionls = {'o8': ild.o8doublet,\
             'o7': ild.o7major,\
             'o6': ild.o6major,\
             'ne8': ild.ne8major,\
             'ne9': ild.ne9major,\
             'fe17': ild.fe17major,\
             }
    bstart = 100. * 1e5 # cm/s
    
    # read in the data
    coldens = {}
    EWs = {}
    with h5py.File(datafile, 'r') as df:  
        cosmopars = {key: val for key, val in df['Header_sample3/cosmopars'].attrs.items()}
        boxvel = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * \
                 c.cm_per_mpc * \
                 cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1e-5
                 
        for ion in ions:
            _mincd = colmin[ion]
            coldens[ion] = {}
            EWs[ion] = {}
            
            for vwindow in vwindows:
                samplegroup = samplegroups[ion]
                if samplegroup is None:
                    samplegroup = 'full_sample/'
                elif samplegroup[-1] != '/':
                    samplegroup = samplegroup + '/'
                    
                if vwindow is None:
                    epath = 'EW_tot/'
                    cpath = 'coldens_tot/'
                    vkey = boxvel
                else:
                    spath = 'vwindows_maxtau/Deltav_{dv:.3f}/'.format(dv=vwindow)
                    epath = spath + 'EW/'
                    cpath = spath + 'coldens/'
                    vkey = vwindow
                epath = samplegroup + epath
                cpath = samplegroup + cpath

                _cd = np.array(df[cpath + ion])
                _ew = np.array(df[epath + ion])
                coldens[ion][vkey] = _cd
                EWs[ion][vkey] = _ew
    
    # get the selections for each ion
    with h5py.File(outfile, 'a') as fo:
        hed = fo.create_group('Header')
        hed.attrs.create('input file', np.string_(datafile))
        hed.attrs.create('info', np.string_('using ion-selected subsamples as a basis for each ion'))
        csm = hed.create_group('cosmopars')
        for key in cosmopars:
            csm.attrs.create(key, cosmopars[key])
        
        for ion in ions:
            colsel = coldens[ion][boxvel] >= colmin[ion]
            jumpsels = {jdn: np.logical_and(coldens[ion][jv2] - coldens[ion][jv1] < jdn,\
                                      colsel) for jdn in jdns}                      
            igrp = fo.create_group(ion)
            sgrp = igrp.create_group('selection')
            sgrp.attrs.create('v1_rf_kmps', jv1)
            sgrp.attrs.create('v2_rf_kmps', jv2)
            sgrp.create_dataset('Delta log10 N', data=np.array(jdns))
            sgrp.attrs.create('min log10 N cm**-2', colmin[ion])
            sgrp.create_dataset('coldens_selection', data=colsel)
            jsgrp = sgrp.create_group('coldens_jump_selection')
            for jdn in jdns:
                jsgrp.create_dataset('{:.3f}'.format(jdn), data=jumpsels[jdn])
            igrp.create_dataset('coldens_total_all', data=coldens[ion][boxvel])
            igrp.create_dataset('ew_total_all', data=EWs[ion][boxvel])
            
            bvals = []            
            for selection in [slice(None, None, None), colsel] + \
                [jumpsels[jdn] for jdn in jdns]:
                _bvals = {}
                vwds = list(coldens[ion].keys())
                for vwindow in vwds:                   
                    N = 10**coldens[ion][vwindow][selection]
                    EW = EWs[ion][vwindow][selection]
                    if fitlogEW:
                        EW = np.log10(EW)
                        
                    _ion = ionls[ion]
                    def lossfunc(b):
                        if b <= 0.: # back off from that area!
                            return np.inf
                        EWres = ild.linflatcurveofgrowth_inv_faster(N, b, _ion)
                        if fitlogEW:
                            EWres = np.log10(EWres)
                        return np.sum((EWres - EW)**2)
                    
                    optres = spo.minimize(lossfunc, x0=bstart, method='COBYLA', tol=1e4,\
                                          options={'rhobeg': 2e6})
                     
                    if optres.success:
                        bfit = optres.x * 1e-5 # to km/s
                        print('Best fit for {ion}: {fit}'.format(ion=ion, fit=bfit))
                        _bvals[vwindow] = bfit
                    else:
                        print('b parameter fitting failed:')
                        print(optres.message)
                        return optres
                print('Collating b values')
                vvals = sorted(list(_bvals.keys()))
                _bvals = [_bvals[key] for key in vvals]                
                bvals.append(_bvals)
            print('Saving ion data')
            igrp.create_dataset('Delta v kmps', data=np.array(vvals))
            ds = igrp.create_dataset('best-fit b kmps', data=np.array(bvals))
            ds.attrs.create('info', np.string_('index 0: samples - all, coldens, coldens and jumps; index 1: Delta v'))
            
def plot_bpareffect_jumpsel(ion):

    datafile = '/net/luttero/data2/specwizard_data/bugcheck_bpar_deltav/' + \
                      'bparfits_sightline_subsamples_set2.hdf5'
    outname = '.'.join(datafile.split('.')[:-1]) + '_{ion}.pdf'.format(ion=ion)
    
    fig = plt.figure(figsize=(7., 5.))
    grid = gsp.GridSpec(nrows=2, ncols=2, hspace=0.35, wspace=0.35,\
                        width_ratios=[5., 3.])
    fontsize = 12
    title = 'effect of N-jump selections for {ion}'.format(ion=ild.getnicename(ion))
    fig.suptitle(title, fontsize=fontsize)
    
    bax = fig.add_subplot(grid[:, 0])
    nax = fig.add_subplot(grid[0, 1])
    eax = fig.add_subplot(grid[1, 1])
    
    bax.set_xlabel('$\\Delta \\, v \\; [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$', fontsize=fontsize)
    bax.set_ylabel('best-fit $b \\; [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$', fontsize=fontsize)
    pu.setticks(bax, fontsize)
    nax.set_xlabel('$\\log_{{10}} \\, \\mathrm{{N}} \\; [\\mathrm{{cm}}^{{-2}}]$', fontsize=fontsize)
    nax.set_ylabel('number of sightlines', fontsize=fontsize)
    pu.setticks(nax, fontsize)
    eax.set_xlabel('$\\log_{{10}} \\, \\mathrm{{EW}} \\; [\\mathrm{{m\\AA}}]$', fontsize=fontsize)
    eax.set_ylabel('number of sightlines', fontsize=fontsize)
    pu.setticks(eax, fontsize)
    
    kwargs_ls  = [{'linestyle': 'solid'}, {'linestyle': 'dashed'}] + \
                 [{'dashes': [6, 2]}, {'dashes': [3, 1]}, {'dashes': [1, 1]},\
                  {'dashes': [6, 2, 3, 2]}, {'dashes': [6, 2, 1, 2]},\
                  {'dashes': [3, 1, 1, 1]}]
    
    with h5py.File(datafile, 'r') as df:
        grp = df[ion]
        sgrp = grp['selection']
        jdns = np.array(sgrp['Delta log10 N'])
        csel = np.array(sgrp['coldens_selection'])
        cjsel = {key: np.array(sgrp['coldens_jump_selection']['{:.3f}'.format(key)])\
                 for key in jdns}
        colmin = sgrp.attrs['min log10 N cm**-2']
        
        sels = [slice(None, None, None), csel] + [cjsel[key] for key in jdns]
        labels = ['all', '$\\log_{{10}}\\mathrm{{N}} > {cmin:.1f}$'.format(cmin=colmin)] + \
                 ['$\\Delta \\log_{{10}} \\mathrm{{N}} < {dn}$'.format(dn=jdn)\
                  for jdn in jdns]
        colors = ['gray', 'black'] +\
                 ['C{}'.format(i % 10) for i in range(len(jdns))]
                 
        ew = np.log10(np.array(grp['ew_total_all'])) + 3.
        cd = np.array(grp['coldens_total_all'])
        dv = np.array(grp['Delta v kmps'])
        bv = np.array(grp['best-fit b kmps'])
        
    cdmin = np.floor(np.min(cd) / 0.1) * 0.1
    cdmax = np.ceil(np.max(cd) / 0.1) * 0.1
    cdbins = np.arange(cdmin, cdmax + 0.05, 0.1)
    ewmin = np.floor(np.min(ew) / 0.1) * 0.1
    ewmax = np.ceil(np.max(ew) / 0.1) * 0.1
    ewbins = np.arange(ewmin, ewmax + 0.05, 0.1)
    
    for i in range(len(sels)):
        color = colors[i]
        label = labels[i]
        sel = sels[i]
        kwargs = kwargs_ls[i]
        
        hist, edges = np.histogram(ew[sel], bins=ewbins)
        ploty = np.array([0] + [count for val in hist for count in [val] * 2] + [0])
        plotx = np.array([ed for val in edges for ed in [val] * 2])
        eax.plot(plotx, ploty, color=color,  **kwargs)
        
        hist, edges = np.histogram(cd[sel], bins=cdbins)
        ploty = np.array([0] + [count for val in hist for count in [val] * 2] + [0])
        plotx = np.array([ed for val in edges for ed in [val] * 2])
        nax.plot(plotx, ploty, color=color, **kwargs)
        
        bax.plot(dv, bv[i], color=color, label=label, **kwargs)
    
    bax.legend(fontsize=fontsize)
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
    
def selectsamples_jumpeffect():
    '''
    use jump criteria from set1, set 2 plots that show 'growth as usual' 
    between 5000 and 6200 km/s Delta v (total) to select jump and control 
    samples for the ions with issues
    
    save as hdf5 -> save spectrum files and numbers in specwizard output files
                    and identifiers in N/EW vs. Delta v files
    '''
    infostr = 'selected sightlines by change in column density between ' + \
              'jump_v1 and jump_v2. Control samples are randomly selected ' +\
              'for each jummp sightline from a sample with ' + \
              '(1) no/small N jumps, and' + \
              '(2) a column density difference with the jump sightline ' + \
              'smaller than Ntol_match (log cm**-2). ' + \
              'The two control samples are generated by matching column ' + \
              'densities at Deltav_match_control1 (control1), ' + \
              'and in the full sightline (control2). ' + \
              'The same sightline may occur more than once as a control.'
    
    outfile = '/net/luttero/data2/specwizard_data/bugcheck_bpar_deltav/' + \
                      'sample_selection1.hdf5'
                      
    datafile = '/net/luttero/data2/specwizard_data/sample3-6_coldens_EW_vwindows_subsamples.hdf5'

    #ions_all = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    ions = ['o8', 'fe17', 'ne9']
    #samplegroups = {ion: '{ion}_selection'.format(ion=ion) for ion in ions}
    
    # jump identification criteria:
    jv1 = 5000.
    jv2 = 6200.
    vc1 = 1600. # matching the max tau absorber
    Ntol_match = 0.05
    # sightline selection general (total column densities)
    colmin = {'o8': 15.5,\
              'ne9': 15.,\
              'fe17': 14.5,\
              } 
    # min. jump for the jump sample, max. jump for the control sample
    jumpcrit = {'o8': 0.2,\
                'fe17': 0.1,\
                'ne9': 0.2,\
                }
    controlcrit = {'o8': 0.1,\
                   'fe17': 0.05,\
                   'ne9': 0.1,\
                   }
    
    with h5py.File(datafile, 'r') as df,\
         h5py.File(outfile, 'w') as fo:
        # header info copy
        dkeys = df.keys()
        for key in dkeys:
            if 'Header' in key:
                df.copy(key, fo)
        for ion in ions:
            ogrp = fo.create_group('{ion}_sample'.format(ion=ion))
            
            grp = df['{ion}_selection'.format(ion=ion)]
            sample = grp.attrs['selection_filename'].decode()
            sample = sample.split('/')[-1] # file name
            sample = sample.split('_')
            sample = np.array(sample)[np.array(['sample' in part \
                                                for part in sample])]
            
            ogrp.attrs.create('jump_v1', jv1)
            ogrp.attrs.create('jump_v2', jv2)
            ogrp.attrs.create('mindiff_N_jump', jumpcrit[ion])
            ogrp.attrs.create('maxdiff_N_controls', controlcrit[ion])
            ogrp.attrs.create('min_total_N_included', colmin[ion])
            ogrp.attrs.create('Deltav_match_control1', vc1)
            ogrp.attrs.create('Ntol_match_controls', Ntol_match)
            ogrp.attrs.create('info', np.string_(infostr))
            ogrp.attrs.create('specwizard_sample', np.string_(sample))
            ogrp.attrs.create('vwindow_data_file', np.string_(datafile))
            
            specnums = np.array(grp['selected_specinds'])
            cdtot = np.array(grp['coldens_tot/{ion}'.format(ion=ion)])
            cdcen = np.array(grp['Deltav_{dv:.3f}/coldens/{ion}'.format(dv=vc1, ion=ion)])
            cdjv1 = np.array(grp['Deltav_{dv:.3f}/coldens/{ion}'.format(dv=jv1, ion=ion)])
            cdjv2 = np.array(grp['Deltav_{dv:.3f}/coldens/{ion}'.format(dv=jv2, ion=ion)])
            
            gensel = cdtot >= colmin[ion]
            jumpinds = np.where(np.logical_and(gensel, cdjv2 - cdjv1 >= jumpcrit[ion]))[0]
            control_candidate_inds = np.where(cdjv2 - cdjv1 <= controlcrit[ion])[0]
            
            specnums_jump = specnums[jumpinds]
            
            control1_inds = [control_candidate_inds[\
                                np.random.choice(np.where(\
                                np.abs(cdcen[j] - cdcen[control_candidate_inds]) <= Ntol_match\
                                )[0])\
                                ]
                             for j in jumpinds]
            control1_inds = np.array(control1_inds)
            specnums_control1 = specnums[control1_inds]
            
            control2_inds = [control_candidate_inds[\
                                np.random.choice(np.where(\
                                np.abs(cdtot[j] - cdtot[control_candidate_inds]) <= Ntol_match\
                                )[0])\
                                ]
                             for j in jumpinds]
            control2_inds = np.array(control2_inds)
            specnums_control2 = specnums[control2_inds]
            
            ogrp.create_dataset('indices_jump_in_ion_sample', data=jumpinds)
            ogrp.create_dataset('specnums_jump', data=specnums_jump)
            ogrp.create_dataset('indices_control1_in_ion_sample', data=control1_inds)
            ogrp.create_dataset('specnums_control1', data=specnums_control1)
            ogrp.create_dataset('indices_control2_in_ion_sample', data=control2_inds)
            ogrp.create_dataset('specnums_control2', data=specnums_control2)
            
            print('Samples generated for ion {ion}'.format(ion=ion))
            print('Jump sample contains {} sightlines'.format(len(jumpinds)))
            print('control1 sample contains {} unique sightlines'.format(len(np.unique(specnums_control1))))
            print('control2 sample contains {} unique sightlines'.format(len(np.unique(specnums_control2))))
            
def savedata_jumpeffect():
    '''
    store spectra, x/y positions, max tau positions, and cross-reff'd galaxies
    for the jump-selected and control sample sightlines
    '''
    pass    
