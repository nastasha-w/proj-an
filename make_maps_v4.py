#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:10:09 2019

@author: wijers


settings: must be string, bool, int, float, or tuple of those

keys (str): options:
------------------------------------------------------------------------------
includeSF  'all', 'no', 'only': all gas, no SF gas, or only SF gas 
           respectively (does not hnadle selection, but checked against some 
           other options for efficiency since this selection is commonly used 
           and impacts whether prescriptions to handle star-forming gas make 
           sense)
T4         boolean: set SF gas to 10^4 K (else outputs taken at face value)
ibmethod   'sbtables', 'sptables', 'bdofromsim', 'rahmati2013', 'fromSFR': how 
           to calculate or get ion fractions - interpolate tables by Serena
           Bertone (HM01, old Cloudy version, but consistent with eagle 
           cooling) or Sylvia Ploeckinger (HM12, latest Cloudy version in Jan
           2019, includes self-shielding), get directly from the simulation as
           in Ben Oppenheimer's zooms, or calculate analytically using Ali 
           Rahmati's 2013 methods
emmethod   'sbtables', 'fromSFR': how to get gas emissivities
H2method   'BR06' (different options may come later)
ab_<element> 'Sm', 'Pt' or float value: use smoothed abundances, particle 
           abundances, or a fixed mass fraction given by the floating-point 
           value (solar units)
coolmethod 'per_element' or 'total_metals'

quantitiy names:
------------------------------------------------------------------------------
most simulation quantities: just the name of the quantity (after PartType#/),
                  e.g. Temperature, StarFormationRate
Z :               metallicity (total); smoothed, particle, or float depending
                  on settings
eltab_<element> : ElementAbundance/ or SmooothedElementAbundance/ or fixed 
                  (absolute mass fraction, not in solar units)
Nion_<ion>      : number of ions in that particle (for column densities)
Lline_<line>    : luminosity of that particle (for surface brightnesses)
eos             : (boolean) - is SFR > 0 ?
logT            : log10 temperature [K]
nH              : hydrogen number density
lognH           : log10 nH [cm^-3]
volume          : mass / density
Lambda_over_nH2 : cooling rate/nH^2 [erg cm^3/s]
tcool           : cooling time (internal energy/cooling rate) [s]
logZ            : log10 Z mass fraction
"""

import numpy as np
import numbers
import string

import iondata as iond
import ion_header as ionh
import eagle_constants_and_units as cu
import make_maps_opts_locs as ol

settings_options = {\
    'includeSF': {'all', 'no', 'only'},\
    'T4'       : {True, False},\
    'ibmethod' : {'sbtables', 'sptables', 'bdofromsim', 'rahmati2013', 'fromSFR'},\
    'emmethod' : {'sbtables', 'fromSFR'},\
    'H2method' : {'BR06'},\
    'coolmethod': {'per_element', 'total_metals'},\
    }

# not necessarily an exhaustive check, but should save a bit of coding time
# for abundances and listed settings, does option and type checking and removes
# simple conflicts. Does not check settings against
# ignores unlisted settings
 
def check_eltab_setting(key, **settings):
    if isinstance(settings[key], numbers.Number):
        if not (0. <= settings[key] and 1. >= settings[key]):
            raise ValueError('in get_reqsettings_this for %s: element and metal abundances are mass fractions and should be between 0 and 1'%(key))
        else: # valid choice
           return True
    elif settings[key] in ['Sm', 'Pt']:
        return True
    else:
        return False
                    
def check_settings(**settings):
    keys = settings.keys()
    
    eltab_keys = [key if (key == 'Z' or key[:6] == 'eltab_') else None]
    if len(eltab_keys) > 0:
        if None in eltab_keys:
            eltab_keys = list(set(eltab_keys))
            eltab_keys.remove(None)
        eltab_check = np.all([check_eltab_setting(key) for key in eltab_keys])
    else:
        eltab_check = True
        
    other_keys = set(keys) - set(eltab_keys)
    if len(other_keys) > 0:
        other_check = np.all([settings[key] in settings_options[key] for setting in settings])
    else:
        other_check = True 
    
    if not (other_check and eltab_check):
        return False
    
    # check if stuff requiring SFR include SF gas, and stuff ignoring SF gas
    # is not putting effort into checking/modfifying SF gas
    if 'T4' in keys:
        if not 'includeSF' in keys:
            return False
        elif settings['includeSF'] == 'no' and settings['T4']:
            return False
        else:
            pass
    if 'ibmethod' in keys:
        if not 'includeSF' in keys:
            return False
        elif settings['includeSF'] == 'no' and settings['ibmethod'] == 'fromSFR':
            return False
        else:
            pass
    if 'emmethod' in keys:
        if not 'includeSF' in keys:
            return False
        elif settings['includeSF'] == 'no' and settings['emmethod'] == 'fromSFR':
            return False
        else:
            pass
    
    return True
                    
                    
                    
                    


class ParticleQuantity:
    '''
    mostly dummy class, just sets all the required properties
    read and delete functions are input into reading quantities to allow 
    coupling to e.g. Vardict, or Simfile directly 
    '''
    def init(self, name, derived, **settings):
        self.name = name
        self.settings = settings
        self.derived = derived
        self.get_reqprops()
        self.get_reqsettings()
        if not np.all(setting in self.settings.keys() for setting in self.reqsettings):
            raise ValueError('settings %s do not contain all required settings for %s'%(self.name, self.settings))
        self.zerodiag = None # if this is zero, then the particle quantity will be: useful for preselection by Vardict
        
    def __repr__(self):
        return 'ParticleQuantity %s; derived: %s,\nsettings: %s'%(self.name, self.derived, self.settings)
    
    def __str__(self):
        return 'ParticleQuantity %s'%(self.name)
    
    def get_reqprops(self):
        self.get_reqprops_names()
        self.reqprops = {key: ParticleQuantity(key, self.settings) for key in self.reqprops_names}
    
    def get_reqsettings_all(self):
        '''
        retrieves all required settings recursively from required properties
        '''
        if not hasattr(self, 'reqprops'):
            self.get_reqprops()
        self.get_reqsettings_this()
        self.reqsettings = self.reqsettings_this | set([req for key in self.reqprops.keys() for req in self.reqprops[key].reqsettings])
    
    def issame(self, pq):
        try:
            if self.name == pq.name:
                if self.reqsettings == pq.reqsettings:
                    return np.all([np.all(np.array(self.settings[key]) == np.array(pq.settings[key])) for key in self.reqsettings])
                else:
                    return False
            else:
                return False
        except AttributeError:
            print('Warning: %s is not a ParticleQuantity'%pq)
            return False
        
    def cleanup(self):
        try:
            del self.arr
        except AttributeError:
            print('%s did not have an array read in'%(self))
        try:
            del self.tocgs
        except: # it doesn't really matter if one float is kept accidentally
            pass

    def require(self, ret=False, savelist=None, readf=None, delf=None):
        if self.derived:
            if not hasattr(self, 'arr'):
                self.get(ret=ret, savelist=savelist, readf=readf, delf=delf)
    
    # overwrite in subclasses
    def get_reqsettings_this(self):
        '''
        Should get required settings, and check whether the required input 
        settings are ok (present and non-conflicting)
        issues with required object settings are handled in those
        '''
        raise NotImplementedError('get_reqsettings_this method in %s is not set'%str(self)) 
        
    def get(self, ret=False, savelist=None, readf=None, delf=None):
        '''
        should calculate or read in the particle property, and set self.arr, 
        self.tocgs
        '''
        raise NotImplementedError('calc method in %s is not set'%str(self)) 

        
    def get_reqprops_names(self):
        raise NotImplementedError('get_reqprops_names method in %s is not set'%str(self)) 
    
   


class BasicParticleQuantity(ParticleQuantity):
    '''
    Stuff that's read in directly from the output; note that if T4 is set,
    temperature is a derived property
    '''
    def init(self, name, derived, **settings):
        ParticleQuantity.__init__(self, name, False, **settings)
        self.settings = settings
        self.derived = derived
        self.get_readkey()

    def get_reqprops_names(self):
        self.reqprops_names = set() # no prior info needed for something that is read in directly
    
    def get_reqsettings_this(self):
        # temperature 
        if self.name == 'Temperature':
            self.reqsettings = {'includeSF'}
            if 'includeSF' not in self.settings.keys():
                raise ValueError('settings %s do not contain all required settings for %s'%(self.name, self.settings))
            # includeSF valid, T4 setting matters
            elif self.settings['includeSF'] in ['all', 'only']:
                self.reqsettings |= 'T4'
                if 'T4' not in self.settings.keys():
                    raise ValueError('settings %s do not contain all required settings for %s'%(self.name, self.settings))
                else:
                    if not isinstance(self.settings['T4'], bool):
                        raise ValueError('Invalid option for includeSF %s'%self.settings['T4'])
                    elif self.settings['T4']:
                        raise ValueError('the temperature with SF gas at 10^4 K is a derived quantity, do not use a BasicParticleQuantity for this')
                    else:
                        pass
            # includeSF valid, T4 doesn't matter
            elif self.settings['includeSF'] == 'no':
                self.settings['T4'] = False # if we're not going to use SF gas, it's a waste of time to check which gas to set to 10^4 K
            else:
                raise ValueError('Invalid option for includeSF %s'%self.settings['includeSF'])
        
        # abundances
        if self.name == 'Z':
            self.reqsettings = {'ab_Z'}
        elif self.name[:6] == 'eltab_':
            self.reqsettings = {self.name[3:]}
            
        if self.name == 'Z' or self.name[:6] == 'eltab_':
            self.absetting = self.settings[self.reqsettings[0]]
            if isinstance(self.absetting, numbers.Number):
                if not (0 <= self.absetting and 1 >= self.absetting):
                    raise ValueError('in get_reqsettings_this for %s: element and metal abundances are mass fractions and should be between 0 and 1'%(self.name))
                else: # valid choice
                    self.absetting = float(self.absetting)
            elif self.absetting in ['Sm', 'Pt']:
                pass
            else:
                raise ValueError('in get_reqsettings_this for %s: abundances should be "Sm", "Pt", or a mass fraction (float)'%(self.name))
        
        # other directly read-in quantities
        else:
            self.reqsettings = set()
            
            
    def get(self, ret=False, savelist=None, readf=None, delf=None):
        if self.name == 'Z' or self.name[:6] == 'eltab_':
            if isinstance(self.absetting, float):
                self.arr = self.absetting
                self.tocgs = 1.
                self.readkey = None
            elif self.name == 'Z':
                if self.absetting == 'Sm':
                    self.readkey = 'SmoothedMetallicity'
                elif self.absetting == 'Pt':
                    self.readkey = 'Metallicity'
                else:
                    raise ValueError()
            elif self.name[:6] == 'eltab_':
                if self.absetting == 'Sm':
                    self.readkey = 'SmoothedElementAbundance/'
            else:
                raise ValueError()
        else:
            self.readkey = self.name
        if self.readkey is not None:
            self.arr, self.tocgs = readf(self.readkey)
        if ret:
            return self.arr, self.tocgs
        
        includeSF  'all', 'no', 'only': all gas, no SF gas, or only SF gas 
           respectively (does not hnadle selection, but checked against some 
           other options for efficiency since this selection is commonly used 
           and impacts whether prescriptions to handle star-forming gas make 
           sense)
T4         boolean: set SF gas to 10^4 K (else outputs taken at face value)
fromSFR    boolean: derive a property from the star formation rate
ibmethod   'sbtables', 'sptables', 'bdofromsim', 'rahmati2013': how to
           calculate or get ion fractions - interpolate tables by Serena
           Bertone (HM01, old Cloudy version, but consistent with eagle 
           cooling) or Sylvia Ploeckinger (HM12, latest Cloudy version in Jan
           2019, includes self-shielding), get directly from the simulation as
           in Ben Oppenheimer's zooms, or calculate analytically using Ali 
           Rahmati's 2013 methods
H2method   'BR06' (different options may come later)
ab_<element> 'Sm', 'Pt' or float value: use smoothed abundances, particle 
           abundances, or a fixed mass fraction given by the floating-point 
           value (solar units)
coolmethod 'per_element' or 'total_metals'
        
def getParticleQuantity(name, settings):
    '''
    returns the appropriate ParticleQuantity subclass
    '''
    
    return ParticleQuantity(name, derived, settings)