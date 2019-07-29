#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:25:00 2019

@author: wijers
"""

import numpy as np
import matplotlib.pyplot as plt

import make_maps_v3_master as m3
import make_maps_opts_locs as ol
import eagle_constants_and_units as c
import cosmo_utils as cu

cosmopars_eagle_z0 = cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 2.220446049250313e-16}
cosmopars_eagle_z0p1 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}
#logrhob_av_ea_28 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
#logrhob_av_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )

rho_to_nh = 0.752 / (c.atomw_H * c.u)
deg2 = (np.pi / 180.)**2
arcsec2 = deg2 / 60.**4
arcmin2 = deg2 / 60.**2


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
        
        
        self.cosmopars = cosmopars
        self.cosmopars['z'] = z
        self.cosmopars['a'] = 1. / (1. + self.cosmopars['z'])
        self.baryonfraction = baryonfraction
        
        self.mass = M200c_Msun * c.solar_mass      
        self.massDM = self.mass * (1 - self.cosmopars['omegab'] / self.cosmopars['omegam'] * baryonfraction)
        self.massgas = self.mass * (1 - self.cosmopars['omegab'] / self.cosmopars['omegam'] * baryonfraction)
        
        self.rhoc = getrhoc(self.cosmopars['z'], cosmopars=self.cosmopars) # Hubble(z) will assume an EAGLE cosmology
        self.r200c = (self.mass / (self.deltadef * self.rhoc))**(1./3.)
        
        self.concentration()
        self.T200c_hot()
        
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
    
    def getthermprof(self, rsample):
        '''
        not even in HSE; literally just NFW gas at Tvir
        '''
        self.getradprof_rho(rsample)
        self.T_radprof = self.Tvir * np.ones(len(self.rsample))
        
        self.P_radprof = self.rho_radprof / self.mu * c.boltzmann * self.T_radprof # no molecules
        self.HSE = False
        
    def getemission(self, dct_lognH_logT, line):
        self.lineind = ol.line_nos_ion[line]
        self.emdens = 10**m3.find_emdenssq(self.cosmopars['z'], ol.elements_ion[line], dct_lognH_logT, self.lineind) # luminosity = this * n_H^2 * volume * elt. abundance / solar 
        
    
    def solveHSE(self, mode='NFW'):
        '''
        mode: NFW -> assume gas follows DM, solve for temperature
              Tvir -> assume gas is at Tvir, solve for density
        get pressure 
        no iteration, just assumes gas follows DM
        '''
        self.HSE = False
        self.HSEmode = None
        raise NotImplementedError('HSE application is not written in yet')
    
    def calc_emissionprof(self, rbins=1000, mode='oversimplified', Z=0.3, lines=[]):
        '''
        get emission per unit volume radial profile
        Z = metallicity in solar units (assuming solar ratios)
        '''
        self.rsample = np.linspace(0., self.r200c, rbins)
        if mode == 'oversimplified':
            self.getthermprof(self.rsample)
        else:
            self.getthermprof(self.rsample)
            self.solveHSE(mode=mode)
        self.emissionmode = mode
        
        self.nH_radprof = self.rho_radprof * 0.752 / (c.atomw_H * c.u) # technically for primoridal gas, but should be good enough
        self.emtot = np.zeros(len(self.rsample))
        for line in lines:
            self.getemission({'lognH': np.log10(self.nH_radprof), 'logT': np.log10(self.T_radprof)}, line=line) 
            self.emdens /= ol.line_eng_ion[line] # convert to photons
            self.emtot += self.emdens
            del self.emdens
        self.emdens_radprof = self.emtot * Z * self.nH_radprof**2
        del self.emtot
        self.emdens_lines = lines
    
    
    def emcol_to_SB(self):
         self.zdist = self.cosmopars['z']
         self.comdist = cu.comoving_distance_cm(self.cosmopars['z'], cosmopars=self.cosmopars)
         if self.comdist <= 0.:
             self.comdist = 1. * c.cm_per_mpc
         # even at larger values, the projection along z-axis = projection along sightline approximation will break down
         self.ldist = self.comdist * (1. + self.zdist) # luminosity distance
         self.adist = self.comdist / (1. + self.zdist) # angular size distance
         # half angles spanned by an emission element: emission is a column quantity here, so per cm^2
         self.halfangle_x = 1. / self.adist
         self.halfangle_y = 1. / self.adist

         #solidangle = 2*np.pi*(1-np.cos(2.*halfangle_x))
         #print("solid angle per pixel: %f" %solidangle)
         # the (1+z) is not so much a correction to the line energy as to the luminosity distance:
         # the 1/4 pi dL^2 luminosity -> flux conversion is for a broad spectrum and includes energy flux decrease to to redshifting
         # multiplying by (1+z) compensates for this: the number of photons does not change from redshifting
         self.emcol_to_SB_factor = 1. / (4 * np.pi * self.ldist**2) * (1. + self.zdist)  * 1. / cu.solidangle(self.halfangle_x, self.halfangle_y) # extra factor 1 + z is to account fo the different between photon and energy luminosity scalings

    def calc_SBprof(self, rbins=1000, mode='oversimplified', Z=0.3, lines=[]):
        if hasattr(self, 'emdens_radprof'):
            if set(self.emdens_lines) == set(lines) and rbins == len(self.rsample):
                pass
            else:
                self.calc_emissionprof(rbins=rbins, mode=mode, Z=Z, lines=lines)
        else:
            self.calc_emissionprof(rbins=rbins, mode=mode, Z=Z, lines=lines)
        
        # use rbins as impact parameter bins too
        self.bvals = np.copy(self.rsample)[:-1] # last bin will have r^2 - b^2 = 0
        if not np.max(np.abs(np.diff(np.diff(self.rsample)))) < 1e-7 * self.rsample[-1]: # sample equally spaced
            raise RuntimeError("rsample is assumed to be equally spaced, but isn't")  
        # ignore central bin: NFW density -> emission diverges there
        if self.rsample[0] == 0.:
            self.rsample_temp = self.rsample[1:]
            self.emdens_radprof_temp = self.emdens_radprof[1:]
        else:
            self.rsample_temp = self.rsample
            self.emdens_radprof_temp = self.emdens_radprof
            
        self.emcol = np.array([\
                               2. * np.sum(\
                                    self.emdens_radprof_temp[self.rsample_temp > b] *
                                    self.rsample_temp[self.rsample_temp  > b ] / \
                                    np.sqrt(self.rsample_temp[self.rsample_temp > b]**2 - b**2) \
                                    * np.average(np.diff(self.rsample_temp)) )\
                               for b in self.bvals])
        del self.rsample_temp
        del self.emdens_radprof_temp
        self.emcol_to_SB()
        ## emcol: luminosity / cm^2 emitting
        self.SB_cgs = self.emcol * self.emcol_to_SB_factor
    
    def plot_SBprof(self, fontsize=12, **kwargs):
        '''
        kwargs are for plotting
        '''
        plt.plot(self.bvals / (1e-3 * c.cm_per_mpc), self.SB_cgs, **kwargs) #  * arcmin2,
        plt.xlabel(r'$r_{\perp} \; [\mathrm{pkpc}]$', fontsize=fontsize)
        plt.ylabel(r'$\mathrm{SB} \; [\mathrm{photons}\, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{arcmin}^{-2}]$')
        plt.yscale('log')
        plt.title(r'$\mathrm{M}_{\mathrm{200c}} = %.2e \mathrm{M}_{\odot}, z = %.2f$'%(self.mass / c.solar_mass, self.cosmopars['z']))


def ploto7trend(z=0.1, masses=[]):
    plt.figure()
    fontsize=12
    mhs = []
    for mass in masses:
        mh = Mockhalo(10**mass, z=0.1)
        mh.calc_SBprof(lines=['o7r', 'o7ix', 'o7iy', 'o7f'])
        mh.plot_SBprof(label=mass, fontsize=fontsize)
        mhs.append(mh)
    #plt.xscale('log')
    plt.title('O VII triplet trend with halo mass')
    plt.ylabel(r'$\mathrm{SB} \; [\mathrm{photons}\, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}]$', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.gca().tick_params(which='both', direction='in', labelsize=fontsize-1)
    plt.savefig('/home/wijers/Documents/papers/aurora_white_paper_wide-field-xray/veryiffymodel_o7triplet_halomasstrend_linx.pdf', format='pdf', bbox_inches='tight')
    
    plt.figure()
    for mi in range(len(masses)):
        mass = masses[mi]
        plt.plot(mhs[mi].rsample / (1e-3 * c.cm_per_mpc), mhs[mi].nH_radprof, label=mass)
    plt.xlabel(r'$r_{\mathrm{3D}} \; [\mathrm{pkpc}]$', fontsize=fontsize)
    plt.ylabel(r'$\mathrm{n}_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(fontsize=fontsize)
    plt.title('Assmued radial density profiles as a funciton of halo mass', fontsize=fontsize)
    plt.savefig('/home/wijers/Documents/papers/aurora_white_paper_wide-field-xray/veryiffymodel_radial_density_profiles_halomasstrend.pdf', format='pdf', bbox_inches='tight')

def ploto8trend(z=0.1, masses=[]):
    plt.figure()
    fontsize=12
    mhs = []
    for mass in masses:
        mh = Mockhalo(10**mass, z=0.1)
        mh.calc_SBprof(lines=['o8'])
        mh.plot_SBprof(label=mass, fontsize=fontsize)
        mhs.append(mh)
    #plt.xscale('log')
    plt.title('O VIII trend with halo mass')
    plt.ylabel(r'$\mathrm{SB} \; [\mathrm{photons}\, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}]$', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.gca().tick_params(which='both', direction='in', labelsize=fontsize-1)
    plt.savefig('/home/wijers/Documents/papers/aurora_white_paper_wide-field-xray/veryiffymodel_o8_halomasstrend_linx.pdf', format='pdf', bbox_inches='tight')
    
#    plt.figure()
#    for mi in range(len(masses)):
#        mass = masses[mi]
#        plt.plot(mhs[mi].rsample / (1e-3 * c.cm_per_mpc), mhs[mi].nH_radprof, label=mass)
#    plt.xlabel(r'$r_{\mathrm{3D}} \; [\mathrm{pkpc}]$', fontsize=fontsize)
#    plt.ylabel(r'$\mathrm{n}_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.legend(fontsize=fontsize)
#    plt.title('Assmued radial density profiles as a funciton of halo mass', fontsize=fontsize)
#    plt.savefig('/home/wijers/Documents/papers/aurora_white_paper_wide-field-xray/veryiffymodel_radial_density_profiles_halomasstrend.pdf', format='pdf', bbox_inches='tight')