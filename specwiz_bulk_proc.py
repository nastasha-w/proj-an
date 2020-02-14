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
import ctypes as ct

import comso_utils as cu
import ion_line_data as ild
import make_maps_opts_locs as ol

## defaults
sdir = ol.sdir
ldir = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/'
ndir = ol.ndir
pdir = ol.pdir
mdir = ol.mdir

ion_default_lines = {'o6': (ild.o6major,),\
                     'o7': (ild.o7major,),\
                     'o8': (ild.o8major, ild.o8minor),\
                     'ne8': (ild.ne8major,),\
                     'ne9': (ild.ne9major,),\
                     'fe17': (ild.fe17major,),\
                     }
# extract groups containing spectra
# spectrum files contained in groups Spectrum<number>

class SpecSet:
    def __init__(self, filename, getall=False, ionlines=ion_default_lines):
        '''
        filename: name of the file containing the specwizard output
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        getall:   get total column densities and EWs
        '''
        if filename[:len(sdir)] == sdir:
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
                     self.specfile['Spectrum%i'%specnum].attrs['X-position'],\
                     self.specfile['Spectrum%i'%specnum].attrs['Y-position']\
                                   ] for specnum in range(self.numspecs)]) 
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
        self.ions_in = np.array(self.specfile['Header'].attrs['Ions'])
        self.fosc_in = np.array(self.specfile['Header'].attrs['Transitions_Oscillator_Strength'])
        self.lambda_in = np.array(self.specfile['Header'].attrs['Transitions_Rest_Wavelength'])
        self.linedata_in = {self.ions_in[i]: {'lambda_rest': self.lambda_in[i],\
                                              'fosc': self.fosc_in[i]}\
                            for i in range(len(self.ions))}
                 
    def getcoldenstot(self, dions='all'):

       # take total column density from file
        if dions == 'all':
            dions = self.ions
        self.coldens.update({ion: \
                               np.array([ \
                                 self.specfile['{grpn}/{ion}/LogTotalIonColumnDensity'.format(grpn=specgrp)][()] \
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
            lambda_r_in = self.linedata[ion]['lambda_rest'] # Angstrom units
            
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
            if not np.allclose(rescalef, 1.):
                tau = np.copy(self.tau[ion])[:, :, np.newaxis]
                tau *= rescalef[np.newaxis, np.newaxis, :]
                for li in range(len(pixshifts)):
                    shift = pixshifts[li]
                    tau[:, :, li] = np.roll(tau[:, :, li], shift, axis=1)
                tau = np.sum(tau, axis=2)
            
            # save results
            self.tau.update({ion: tau})
            self.spectra.update({ion, np.exp(-1 * tau)})
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
            self.spectra.update({ion, np.exp(-1 * tau)})
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
            self.EW[ion] *= self.deltaredshift * lambda_eff # convert absorbed flux fraction to EW
            # convert to rest-frame EW
            self.EW[ion] *= 1. / (self.redshift + 1.)
    
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
        
        numpix = int(deltav_rest_kmps / self.velperpix_kmps + 0.5)
        if numpix >= self.numspecpix:
            print('Velocity interval {vel} is larger than the box size'.format(deltav_rest_kmps))
            print('using the whole sightline, without repetitions')
            vsel = slice(None, None, None)
            numpix =  self.numspecpix
        else:
            vsel = slice(self.centerpix - numpix // 2,\
                         self.centerpix - numpix // 2 + numpix,\
                         None)
        pathfrac = float(numpix) / float(self.numspecpix)
        
        vkey = deltav_rest_kmps
        self.vwindow_EW = {vkey: {}}
        self.vwindow_coldens = {vkey: {}}
        for ion in dions:
            if not self.aligned(ion):
                self.alignmaxtau(dions=[ion])
            if ion not in self.coldens:
                self.getcoldenstot(dions=[ion])
            
            foscs = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambdas = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            lambda_eff = np.sum(foscs  * lambdas) / np.sum(foscs)
            
            self.vwindow_EW[vkey][ion] = 1. - np.sum(self.spectra[ion][:, vsel], axis=1) / float(numpix)
            # convert absorbed flux fraction to EW
            self.vwindow_EW[vkey][ion] *= pathfrac * self.deltaredshift * lambda_eff
            # convert to rest-frame EW
            self.vwindow_EW[vkey][ion] *= 1. / (self.redshift + 1.)
            
            # N \propto tau
            self.vwindow_coldens[vkey][ion] = np.sum(self.tau[ion][:, vsel], axis=1) \
                                              / np.sum(self.tau[ion][:, :], axis=1)            
            self.vwindow_coldens[vkey][ion] *= 10**self.coldens[ion]
            self.vwindow_coldens[vkey][ion] = np.log10(self.vwindow_coldens[vkey][ion])
            
    def save_specdata(self):
        pass
        '''
        save N, EW totals and windows, 
        as well as used ionlines, cosmopars, positions, etc.
        '''
        
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
            print('Cat options are %s, not %s. No values retrieved.'%(str(self.dataoptions.keys()),cat))
            return
        elif name not in self.dataoptions[cat].keys():
            print('Name options are %s, not %s. No values retrieved.'%(str(self.dataoptions[cat].keys()),name)) 
            return

        elif cat == 'posmassw':
            if name == 'nion':
                self.getnion(dions=dions)
            else:
                self.posmassw.update({name : np.array([np.array(self.specfile['Spectrum%i/%s'%(specnum,self.dataoptions[cat][name])]) for specnum in range(self.numspecs)])})
        elif cat == 'ion':
            if name == 'flux': # we have a function for this one
                self.getspectra_base(dions=dions) 
            elif name == 'logcoldens': # and for this one
                self.getcoldens(dions=dions)
            else: # name == 'tau'
                self.tau_base.update({ion: np.array([np.array(self.specfile['Spectrum%i/%s/%s'%(specnum, ion, self.dataoptions[cat][name])]) for specnum in range(self.numspecs)]) for ion in dions})
        else:
            if cat == 'posionw':
                self.basedict = self.posionw
            elif cat == 'veltauw':
                self.basedict = self.veltauw
            else:
                print('%s is not a valis cat option. No values retrieved.'%cat)
                return
            if name not in self.basedict.keys():
                self.basedict[name] = {}
            for ion in dions:      
                self.basedict[name].update({ion : np.array([np.array(self.specfile['Spectrum%i/%s/%s'%(specnum,ion,self.dataoptions[cat][name])]) for specnum in range(self.numspecs)]) for ion in dions})
            del self.basedict           


