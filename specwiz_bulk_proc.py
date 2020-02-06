#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:22:42 2020

@author: wijers

based on specwiz_proc.py: 
for analysis of a large set of specwizard sightlines in one go
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

class Specout:
    def __init__(self, filename, getall=False):
        '''
        sets the hdf5 file self.specfile, and extracts and calculates some stuff
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

    def getlinedata_in(self):
        self.ions_in = np.array(self.specfile['Header'].attrs['Ions'])
        self.fosc_in = np.array(self.specfile['Header'].attrs['Transitions_Oscillator_Strength'])
        self.lambda_in = np.array(self.specfile['Header'].attrs['Transitions_Rest_Wavelength'])
        self.linedata_in = {self.ions_in[i]: {'lambda_rest': self.lambda_in[i],\
                                              'fosc': self.fosc_in[i]}\
                            for i in range(len(self.ions))}
                 
    def getcoldens_tot(self, dions='all'):

       # take total column density from file
        if dions == 'all':
            dions = self.ions
        self.coldens.update({ion: \
                               np.array([ \
                                 self.specfile['{grpn}/{ion}/LogTotalIonColumnDensity'.format(grpn=specgrp)][()] \
                               for specgrp in self.specgroups])\
                             for ion in dions})
    
    def getspectra(self, dions='all', ionlines=ion_default_lines):
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
            
            fosc_use = np.array([specl.fosc for specl in ionlines[ion]])
            lambda_r_use = np.array([specl.lambda_angstrom for specl in ionlines[ion]])
            
            veldiff_at_z_kmps = [(lambda_r - lambda_r_in) \
                                  / lambda_r_in * cu.c.c * 1.e-5 
                                 for lambda_r in lambda_r_use]
            pixshifts = [int(veldiff / self.velperpix_kmps + 0.5) \
                         for veldiff in veldiff_at_z_kmps]
            rescalef = (fosc_use * lambda_r_use**2) / \
                       (fosc_in * lambda_rest_in**2)
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
    
    def alignmaxtau(self, dions='all', ionlines=ion_default_lines):
        '''
        shift all spectra so max. tau is at the central pixel
        if multiple maxima have the exact same value, the first instance in 
        the default zero position is used (argmax default)
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        
        self.centerpix = self.numspecpix // 2
        for ion in dions:
            if ion not in self.tau:
                self.getspectra(dions=[ion], ionlines=ion_default_lines)
            tau = self.tau[ion]
            maxpos = np.argmax(tau, axis=1) 
            rollargs = self.centerpix - maxpos
            for i in range(tau.shape[0]):
                tau[i, :] = np.roll(tau[i, :], rollargs[i])
            self.tau[ion] = tau
            self.spectra.update({ion, np.exp(-1 * tau)})
            self.aligned[ion] = True
            
    def getEW_tot(self, dions='all', ionlines=ion_default_lines):
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
                self.getspectra(dions=[ion], ionlines=ion_default_lines)
            
            foscs = np.array([specl.fosc for specl in ionlines[ion]])
            lambdas = np.array([specl.lambda_angstrom for specl in ionlines[ion]])
            lambda_eff = np.sum(foscs  * lambdas) / np.sum(foscs)
            
            # EW = \int dlamdba (1-flux) = (Delta lambda) - (Delta lambda)/N * sum_i=1^N F_normalised(i)  
            self.EW[ion] = 1. - np.sum(self.spectra[ion], axis=1) / float(self.spectra[ion].shape[1])
            self.EW[ion] *= self.deltaredshift * lambda_eff # convert absorbed flux fraction to EW
            # convert to rest-frame EW
            self.EW[ion] *= 1. / (self.redshift + 1.)
    
    def getEW_vwindow(self, dions='all', ionlines=ion_default_lines):    
        '''
        dions:    ions to get EWs for
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
            
                    
    def getslices(self, slices,offset=0.,posvals=None, posvalperiod = None):
        '''
        Returns list of slice objects to most closely retrieve a slicing into
          <slices (int)> slices
        offset is in slice depth units (int), e.g. slices = 2, offset = 0.5 
          gives 2 slices centered at the edge of the box and at half the box

        when offset != 0., the list is returned with <slices> + 1 slice 
          instances, where first and last are understood to be one box slice 
          cut in two
        '''
        
        # make sure there is at most one slice (part) before offset, and offset > 0
        offset = offset%1.
        
        self.sliceleftedges = np.round(float(self.numspecpix) * (np.arange(slices)+offset)/float(slices),0).astype(int,copy=False)
        outlist = [slice(self.sliceleftedges[i],self.sliceleftedges[i+1],None) for i in range(slices-1)]
        outlist += [slice(self.sliceleftedges[-1],self.numspecpix,None)]
        if offset != 0.:
            outlist = [slice(0,self.sliceleftedges[0],None)] + outlist
        return outlist


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



    def getall(self, dions='all'):
        self.getspectra(dions=dions)
        self.getcoldens(dions=dions)
        self.getEW(dions=dions)
        #self.getnion(dions=dions)

    # corrects o8 spectrum to what you would get from using the o8 doublet with both f_osc in stead of just the larger f_osc component
    # just writes over any previous spectra -> alos produces the correct o8 spectrum
    # correct EW if that was present
    def correct_o8(self):
        # get o8 optical depth spectrum
        self.tauspec_o8 = np.array([np.array(self.specfile['Spectrum%i/o8/OpticalDepth'%specnum]) for specnum in range(self.numspecs)])
        # get pixel velocity along the los (average to reduce single-pixel fp errors); velocities start at zero
        self.velperpix_kmps = np.average(np.diff(np.array(self.specfile['VHubble_KMpS'])))
        # redshift between o8major and o8minor at z=0: measured velocity difference = total redshift * light speed * (km/s / cm/s)
        self.lambdamajz = lambda_rest['o8major'] #* (1.+self.redshift)
        self.lambdaminz = lambda_rest['o8minor'] #* (1.+self.redshift)    
        self.veldiff_at_z_kmps = (self.lambdaminz-self.lambdamajz)/lambda_rest['o8major'] * cu.c.c *1.e-5 
        # what to shift the o8major array by after rescaling to get to o8minor velocity
        self.pixshift_majtomin = int(round(self.veldiff_at_z_kmps/self.velperpix_kmps,0))

        #from Lan & Fukugita 2017: linear EW  = f lambda^2/1.13e20 * Nion; lin. approx for tau is same, but fot tau it is exact
        # -> rescale tau for minor contriution to tau_minor = tau_major * (f_min * lambda_min**2)/(f_maj * lambda_maj**2)   
        self.tauspec_o8_minor = self.tauspec_o8 * (fosc['o8minor'] * lambda_rest['o8minor']**2)/(fosc['o8major'] * lambda_rest['o8major']**2)
        # shift the absorption by the appropriate number of pixels for the wavelength difference 
        self.tauspec_o8_minor = np.roll(self.tauspec_o8_minor,self.pixshift_majtomin,axis=1) # 0 axis -> spectrum number, 1 axis -> spectrum for each sightline
        #optical depths add up
        self.tauspec_o8 += self.tauspec_o8_minor

        self.spectra[u'o8'] = np.exp(-1*self.tauspec_o8)
        self.tauspec_o8
        self.tauspec_o8_minor
        
        # correct corresponding EW: redo calculation with corrected spectrum
        if 'o8' in self.EW.keys():
            self.getEW(dions = ['o8'])