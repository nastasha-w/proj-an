#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:25:00 2019

@author: wijers
"""

import numpy as np

import make_maps_v3_master as m3
import make_maps_opts_locs as ol
import eagle_constants_and_units as c
import eagle_constants_and_units as cu

cosmopars_eagle_z0 = cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 2.220446049250313e-16}
cosmopars_eagle_z0p1 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}
#logrhob_av_ea_28 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
#logrhob_av_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )

def getrhoc(z, cosmopars=None):
    return 3. / (8. * np.pi * c.gravity) * cu.Hubble(z, cosmopars=cosmopars)**2










class Mockhalo:
    
    def __init__(self, M200c_Msun, z=0, cosmopars=cosmopars_eagle_z0, baryonfraction=1.):
        '''
        input z overwrites cosmopars z, a
        set basic paramaters for the halo
        '''
        # things set constant for now
        self.deltadef = 200.
        self.mu = 0.59 # about right for ionised (hot) gas, primordial
        
        
        self.comsopars = cosmopars
        self.cosmopars['z'] = z
        self.cosmopars['a'] = 1. / (1. + self.cosmopars['z'])
        self.baryonfraction = baryonfraction
        
        self.mass = M200c_Msun * c.solar_mass      
        self.massDM = self.mass * (1 - self.cosmopars['omegabaryon'] / self.cosmopars['omegam'] * baryonfraction)
        self.massgas = self.mass * (1 - self.cosmopars['omegabaryon'] / self.cosmopars['omegam'] * baryonfraction)
        
        self.rhoc = getrhoc(self.cosmopars['z'], cosmopars=self.cosmopars) # Hubble(z) will assume an EAGLE cosmology
        self.r200c = (self.mass / (self.deltadef * self.rhoc))**(1./3.)
        
        self.concentration()
        self.T200c_hot()
    
    def get_thermprof(self, rsample):
        '''
        not even in HSE; literally just NFW gas at Tvir
        '''
        self.getradprof_rho(rsample)
        self.T_radprof = self.Tvir * np.ones(len(self.rsample))
        
        self.P_radprof = self.rho_radprof / self.mu * c.boltzmann * self.T_radprof # no molecules
        
    def concentration(self, mode='trivial'):
        if mode == 'trivial':
            self.conc = 5.
        self.conc_mode = mode
    
    def T200c_hot(self):
    # checked against notes from Joop's lecture: agrees pretty well at z=0, not at z=9 (proabably a halo mass definition issue)
        self.Tvir = (self.mu * c.protonmass) / (3. * c.boltzmann) * c.gravity * self.mass / self.r200c
        
    def getradprof_rho(self, rsample):
        self.rs = self.r200c / self.conc
        self.deltac = self.deltadef / 3. * self.conc**3 / (np.log(1. + self.conc) - self.conc / (1. + self.conc))  # Ludlow et al. 2013 formula for NFW
        
        self.rsample = rsample
        self.rhotot_radprof = self.rhoc * self.deltac / (self.rsample / self.rs * (1. + self.rsample / self.rs)**2)
        self.rho_radprof = self.rhotot_radprof * self.massgas / self.mass
        self.rhoDM_radprof = self.rhotot_radprof - self.rho_radprof
        
    def getemission(self, dct_lognH_logT, line):
        self.lineind = ol.line_nos_ion[line]
        self.emdens = 10**m3.find_emdenssq(self.cosmopars['z'], ol.elements_ion[line], dct_lognH_logT, self.lineind) # luminosity = this * n_H^2 * volume * elt. abundance / solar 