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

import sys
import os

import numpy as np
import numbers
import string

import iondata as iond
import ion_header as ionh
import eagle_constants_and_units as cu
import make_maps_opts_locs as ol



version = (0, 0, 0)
####################
# simple utilities #
####################

if sys.version.split('.')[0] == '3':
    def isstr(x):
        return isinstance(x, str)
elif sys.version.split('.')[0] == '2':
    def isstr(x):
        return isinstance(x, basestring)

def isnumber(x): # apparently, bools in python are an int subclass
    return isinstance(x, numbers.Number) and not isinstance(x, bool)



settings_options = {\
    'includeSF': {'all', 'no', 'only'},\
    'T4SF'     : {True, False},\
    'ibmethod' : {'sbtables', 'ps20tables', 'bdofromsim', 'rahmati2013', 'fromSFR'},\
    'emmethod' : {'sbtables', 'fromSFR', 'ps20tables'},\
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
    def init(self, ptype, subtype, derived, **settings):
        self.ptype
        self.subtype
        self.settings = settings
        self.derived = derived
        self.get_reqprops()
        self.get_reqsettings()
        if not np.all(setting in self.settings.keys() for setting in self.reqsettings):
            raise ValueError('settings %s do not contain all required settings for %s'%(self.ptype, self.settings))
        self.zerodiag = None # if this is zero, then the particle quantity will be: useful for preselection by Vardict
        
    def __repr__(self):
        return 'ParticleQuantity %s -- %s,\nsettings: %s'%(self.ptype, self.subtype, self.settings)
    
    def __str__(self):
        return 'ParticleQuantity %s: %s'%(self.ptype, self.subtype)
    
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
            if self.ptype == pq.ptype and self.subtype == pq.subtype:
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



class SimulationBox:
    '''
    lightweight class holding parameters for a box region within a simulation
    checks parameters and rotates vectors
    
    TODO: add enclosing box calculation for e.g., readregion
    
    UNTESTED!!!
    '''
    
    def __init__(self, center, diameter, boxsize, rotmatrix=None, 
                 periodic=False, subslices=1, axis=2, useLOSredshift=False):
        '''
        

        Parameters
        ----------
        center : array-like of floats
            center of the selected volume, in non-rotated coordinates. 
        diameter : float or array-like of floats
            total size of the selected volume along each *rotated* axis.
        boxsize : float
            size of the periodic simulation volume. Set to a very large value 
            if the simulation is not periodic (incl. e.g., outflow boundaries)
        rotmatrix : array-like of floats or None, optional
            orthonormal matrix containing the rotated coordinate axes in the
            simulation coordinate frame. Each new axis has fixed index 1,
            varying index 0 in the array.
            The default is None, meaning no rotation is desired.
        periodic : bool, optional
            does the selected volume span the periodic volume orthogonal to 
            the projection axis? Note that all simulations are assumed to be 
            periodic on large scales. This just affects edge wrapping in 
            projections. The default is False.
        subslices : int, optional
            During the projection, should the volume be sub-divided into 
            orthogonal to the projection axis? 1 measn no, larger values set 
            the number of sub-slices. The default is 1.
        axis : int, optional
            the index of the rotated axis to project along. The default is 2.
        useLOSredshift : bool, optional
            along the line of sight axis, use . The default is False.
        
        units for center diameter, and boxsize must match.
        
        Returns
        -------
        SimulationBox object

        '''
        # pre-sets
        self.dimension = 3
        self.coordinate_type = float
        
        # parameters
        self.center =center
        self.diameter = diameter
        self.boxsize = boxsize
        self.rotmatrix = rotmatrix
        self.periodic = periodic
        self.subslices = subslices
        self.axis = axis
        self.useLOSredshift = useLOSredshift
        self.checkparams()
        
    def checkparams(self)
        errmsg_boxsize = 'SimulationBox: boxsize should be a float > 0. Was {}'
        if not isnumber(self.boxsize):
            raise ValueError(errmsg_boxsize.format(self.boxsize))
        elif self.boxsize <= 0:
            raise ValueError(errmsg_boxsize.format(self.boxsize))

        errmsg_center = 'SimulationBox: center should be a'+\
                        ' list of ' + str(self.dimension) + ' floats. Was {}'
        if not hasttr(self.center, '__len__'):
            raise ValueError(errmsg_center.format(self.center))
        elif len(self.center) != self.dimension:
            raise ValueError(errmsg_center.format(self.center))
        elif not np.all([isnumber(x) for x in self.center]):
            raise ValueError(errmsg_center.format(self.center))
        else:
            self.center = np.asarray(self.center) % self.boxsize
            if len(self.center.shape) != 1:
                raise ValueError(errmsg_center.format(self.center))
        
        errmsg_diameter = 'SimulationBox: diameter should be a single float'+\
                          ' > 0 or list of ' + str(self.dimension) + \
                          ' floats. Was {}'
        if isnumber(self.diameter):
            if self.diameter < 0:
                raise ValueError(errmsg_diameter.format(self.diameter))
            self.diameter = self.diameter * \
                            np.ones((self.dimension,), 
                                    dtype=self.coordinate_type)          
        elif not hasttr(self.diameter, '__len__'):
            raise ValueError(errmsg_diameter.format(self.diameter))
        elif len(self.diameter) != self.dimension:
            raise ValueError(errmsg_diameter.format(self.diameter))
        elif not np.all([isnumber(l) for l in self.diameter]):
            raise ValueError(errmsg_diameter.format(self.diameter))
        elif not np.all([l > 0 l in self.diameter]):
            raise ValueError(errmsg_diameter.format(self.diameter))
        elif np.any([l > self.boxsize for l in self.diameter]):
            msg = 'SimulationBox: input diameter {} is larger than the box {}'
            raise ValueError(msg.format(self.diameter, self.boxsize))
        else:
            self.diameter = np.asarray(self.diameter)
        
        errmsg_rotmatrix = 'SimulationBox: rotmatrix should be None or a ' +\
                           + str(self.dimension) + 'x' + str(self.dimension)+\
                           'orthonormal matrix of floats'
        if rotmatrix is not None:
            if not hasattr(self.rotmatrix, 'shape'):
                self.rotmatrix = np.asarray(self.rotmatrix, 
                                            dtype=self.coordinate_type)
                
            if not np.allclose(np.matmul(self.rotmatrix, self.rotmatrix.T),
                               np.diag((1,) * self.dimension)):
                raise ValueError(errmsg_rotmatrix)
        
        errmsg_axis = 'SimulationBox: axis should be an integer between' + \
                      ' 0 and ' + str(self.dimension) + '. Was {}'
        if not isinstance(self.axis, int):
            raise ValueError(errmsg_axis.format(self.axis))
        elif not (self.axis >= 0 and self.axis < self.dimension):
            raise ValueError(errmsg_axis.format(self.axis))
        
        if not isinstance(self.periodic, bool):
            raise ValueError('SimulationBox: periodic should be True or False')
        self.nonprojaxes = set(range(self.dimension)) - {self.axis}
        if np.any([self.boxsize <= self.diameter[i] for i in nonprojaxes]):
            if not periodic:
                msg = 'For diameter {} and box size {}, projection should' +\ 
                      'be periodic.'
                raise ValueError(msg)
            if self.periodic and self.rotmatrix is not None:
                if not np.allclose(self.rotmatrix, 
                                   np.diag(1.,) * self.dimension):
                    msg = 'Boundary wrapping for projections will fail' + \
                          ' with rotations'
                    raise ValueError(msg)
                    
        if not isinstance(self.useLOSredshift, bool):
            msg = 'SimulationBox: useLOSredshift should be True or False'
            raise ValueError(msg)    
    
    def rotatecoords(self, coords, key=None, coordaxis=1):
        _coords = coords if key is None else coords[key]
        if _coords.shape[coordaxis] != self.dimension:
            'Dimension {} of rotation matrix is incompatible with '
            'coordinates shape {}, axis {}'
            raise ValueError(msg.format(self.dimension, _coords.shape, 
                                        coordaxis))
        if key is not None:
            coords[key] = np.tensordot(self.rotmatrix, _coords, 
                                axes=([1], [coordaxis]))
        else:
            return np.tensordot(self.rotmatrix, _coords, 
                                axes=([1], [coordaxis]))
    
    def getlosval(self, coords, key=None, coordaxis=1):
        _coords = coords if key is None else coords[key]
        if _coords.shape[coordaxis] != self.dimension:
            'Dimension {} of rotation matrix is incompatible with '
            'coordinates shape {}, axis {}'
            raise ValueError(msg.format(self.dimension, _coords.shape, 
                                        coordaxis))
        if key is not None:
            coords[key] = np.tensordot(self.rotmatrix[:, axis], _coords, 
                                axes=([0], [coordaxis]))
        else:
            return np.tensordot(self.rotmatrix[:, axis], _coords, 
                                axes=([0], [coordaxis]))




