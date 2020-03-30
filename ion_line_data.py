#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:42:09 2019

@author: wijers

general-use atom line data and column density / optical depth / EW functions
objects for single-line and single-ion data, pandas table for more general stuff 

to add a line:
    - add ion: element to elements_ion if the ion is new
    - add a specline object for the entry (variable name  = line name, must be unique)
    - add (to) an IonLines object
    - add the specline object to the majorlines or otherlines list
    
Has some general functions for hdf5 read-in and saving of line data, but 
dependencies are limited to python standards and constants_and_units  
"""
import numpy as np
import pandas as pd
import scipy.integrate as si
from scipy.special import wofz # fadeeva function; used for getting voight profiles
import string
import h5py

import eagle_constants_and_units as c

elements_ion = {'c1': 'carbon', 'c2': 'carbon', 'c3': 'carbon', 'c4': 'carbon', 'c5': 'carbon', 'c6': 'carbon',\
             'fe2': 'iron', 'fe3': 'iron', 'fe17': 'iron', \
             'h1': 'hydrogen', 'h2': 'hydrogen', 'lyalpha': 'hydrogen', 'halpha': 'hydrogen', 'h1ssh': 'hydrogen', 'hmolssh': 'hydrogen', 'hneutralssh': 'hydrogen',\
             'he1': 'helium', 'he2': 'helium',\
             'mg1': 'magnesium', 'mg2': 'magnesium',\
             'n2': 'nitrogen', 'n3': 'nitrogen', 'n4': 'nitrogen', 'n5': 'nitrogen', 'n6': 'nitrogen', 'n7': 'nitrogen',\
             'ne8': 'neon', 'ne9': 'neon', 'ne10': 'neon',\
             'o1': 'oxygen', 'o3': 'oxygen', 'o4': 'oxygen', 'o5': 'oxygen', 'o6': 'oxygen', 'o7': 'oxygen', 'o8': 'oxygen',\
             'o7r': 'oxygen',\
             's5': 'sulfur',\
             'si2': 'silicon', 'si3': 'silicon', 'si4': 'silicon', 'si13': 'silicon'}
element_to_abbr = {\
                   'hydrogen':  'H',\
                   'helium':    'He',\
                   'carbon':    'C',\
                   'iron':      'Fe',\
                   'magnesium': 'Mg',\
                   'nitrogen':  'N',\
                   'neon':      'Ne',\
                   'oxygen':    'O',\
                   'sulfur':    'S',\
                   'silicon':   'Si',\
                   }
abbr_to_element = {element_to_abbr[key]: key for key in element_to_abbr.keys()}

roman_to_arabic = {'I':      1,\
                   'II':     2,\
                   'III':    3,\
                   'IV':     4,\
                   'V':      5,\
                   'VI':     6,\
                   'VII':    7,\
                   'VIII':   8,\
                   'IX':     9,\
                   'X':      10,\
                   'XI':     11,\
                   'XII':    12,\
                   'XIII':   13,\
                   'XIV':    14,\
                   'XV':     15,\
                   'XVI':    16,\
                   'XVII':   17,\
                   'XVIII':  18,\
                   'XIX':    19,\
                   'XX':     20,\
                   'XXI':    21,\
                   'XXII':   22,\
                   'XXIII':  23,\
                   'XXIV':   24,\
                   'XXV':    25,\
                   'XXVI':   26}
arabic_to_roman = {roman_to_arabic[key]: key for key in roman_to_arabic.keys()}

def getnicename(ion, mathmode=False):
    if 'ssh' in ion:
        if ion == 'hneutralssh':
            if mathmode:
                return '\mathrm{H\,I} + \mathrm{H}_2'
            else:
                return 'H I + H2'
        elif ion == 'h1ssh':
            ion = 'h1'
        elif ion == 'hmolssh':
            if mathmode:
                return '\mathrm{H}_2'
            else:
                return 'H2'
    eltpart = ''
    pos = 0
    while not ion[pos].isdigit():
        eltpart = eltpart + ion[pos]
        pos += 1
    ionpart = ion[pos:]
    if mathmode:
        return eltpart.capitalize() + '\,' + arabic_to_roman[int(ionpart)]
    else:
        return eltpart.capitalize() + ' ' + arabic_to_roman[int(ionpart)]

def get_elt_state(ion):
    '''
    input:
    ------
    ion: (str) chemical element abreviation followed by a number (1 = neutral)
    
    output:
    -------
    2-tuple (str, int): chemical name (lowercase), ionization state 
                        (1 = neutral)
    '''
    if ion.endswith('ssh'): # this is a special case for tables anyway
        raise ValueError('get_elt_state only works for <chemical abbreviation><integer>-formatted ions, not custom names for self-shielded species')
    _elt = ion
    _state = ''
    while (_elt[-1]).isdigit():
        _state = _elt[-1] + _state
        _elt = _elt[:-1]

    elt = abbr_to_element[string.capwords(_elt)]
    state = int(_state)
    return elt, state

# using api.types. to get it to work on cosma (different pandas version)
# and using list specifier because this dtype setup seems to only be possible by another route entirely in pandas 0.19 on luttero
try:
    ion_type = pd.api.types.CategoricalDtype(categories=elements_ion.keys(), ordered=None)
    element_type = pd.api.types.CategoricalDtype(categories=set([elements_ion[key] for key in elements_ion.keys()]), ordered=None)
    usecatdtype = True
except AttributeError:
    ion_list = elements_ion.keys()
    element_list = list(set([elements_ion[key] for key in elements_ion.keys()]))
    usecatdtype=False

class SpecLine:
    def __init__(self, name, parention, lambda_angstrom, fosc,\
                 combination=None, Atrans=np.NaN):
        '''
        input:
        ------
        name:             (str) name of the line
        parention:        ion within which this is a transition
                          (str, e.g. 'o8', 'fe17') 
        lambda_angstrom:  wavelength of the line in Angstrom (float)
        fosc:             oscillator strength (float)
        combination:      None: just a single line
                          tuple of ion names: this line is actually a 
                          multiplet of the lines in the tuple
        Atrans:           transition probability (float, s**-1) 
                          optional, if you want to calculate the curve of 
                          growth including damping wings
        '''
        self.name = name
        self.ion = parention
        self.lambda_angstrom = lambda_angstrom
        self.fosc = fosc
        self.Atrans = Atrans
        if combination is None:
            self.single = True
        else:
            self.single = False
            self.components = tuple(combination)

    def __str__(self):
        if self.single:
            return 'SpecLine %s'%(self.name)
        else:
            return 'SpecLine %s, combines %s'%(self.name, ', '.join([component.name for component in self.components]))
    def __repr__(self):
        if self.single:
            return 'SpecLine %s from %s: lambda=%.4f A, fosc=%.3e'%(self.name, self.ion, self.lambda_angstrom, self.fosc)  
        else:
            return 'SpecLine %s from %s: lambda=%.4f A, fosc=%.3e\n\tcombines %s'%(self.name, self.ion, self.lambda_angstrom, self.fosc, ', '.join([component.name for component in self.components]))   
    
    def getseries(self, ismajor):
        if not self.single:
            raise TypeError("series conversion is not supported for combination lines")
        else:
            return pd.Series({'ion':     self.ion,\
                              'lambda_angstrom': self.lambda_angstrom,\
                              'fosc':    self.fosc,\
                              'element': elements_ion[self.ion],\
                              'major':   ismajor})

class IonLines:
    '''
    Can be used for all lines logged for an ion, or for a specific multiplet
    '''
    def __init__(self, name, parentelement, speclines, specwizarddefaultline):
        self.name = name
        self.element = parentelement
        self.speclines = {specline.name: specline for specline in speclines}
        self.major = specwizarddefaultline

def savelinedata(openhdf5group, specline):
    '''
    saves line data as attributes in openhdf5group
    recursively saves components in a 'linecomponents' subgroup
    '''
    openhdf5group.attrs.create('linename',       np.string_(specline.name))
    openhdf5group.attrs.create('ion',            np.string_(specline.ion))
    openhdf5group.attrs.create('wavelength [A]', specline.lambda_angstrom)
    openhdf5group.attrs.create('fosc',           specline.fosc)
    if not specline.single:
        cgrp = openhdf5group.create_group('linecomponents')
        for component in specline.components:
            ccgroup = cgrp.create_group(component.name)
            savelinedata(ccgroup, component)

def readlinedata(hdf5group):
    '''
    return a SpecLine object read in from a savelinedata-format hdf5 group 
    '''
    name    = hdf5group.attrs['linename'].decode()
    ion     = hdf5group.attrs['ion'].decode()
    wavelen = hdf5group.attrs['wavelength [A]']
    fosc    = hdf5group.attrs['fosc']
    if 'linecomponents' in hdf5group.keys():
        subgrp = hdf5group['linecomponents']
        components = [readlinedata(subgrp[key]) for key in subgrp.keys()]
        out = SpecLine(name, ion, wavelen, fosc, combination=tuple(components))
    else:
        out = SpecLine(name, ion, wavelen, fosc)
    return out

def saveionlines(openhdf5group, ionlines):
    '''
    saves line data as attributes in openhdf5group
    recursively saves components in a 'linecomponents' subgroup
    '''
    openhdf5group.attrs.create('linesetname',    np.string_(ionlines.name))
    openhdf5group.attrs.create('element',        np.string_(ionlines.element))
    mgrp = openhdf5group.create_group('major')
    savelinedata(mgrp, ionlines.major)
    lgrp = openhdf5group.create_group('lines')
    for name in ionlines.speclines.keys():
        sgrp = lgrp.create_group(name)
        savelinedata(sgrp, ionlines.speclines[name])

def readionlines(hdf5group):
    '''
    return a IonLines object read in from a saveionlines-format hdf5 group 
    '''
    name     = hdf5group.attrs['linesetname'].decode()
    element  = hdf5group.attrs['element'].decode()
    major    = readlinedata(hdf5group['major'])
    lgrp     = hdf5group['lines']
    lines    = {key: readlinedata(lgrp[key]) for key in lgrp.keys()}
    return IonLines(name, element, lines, major)

def savelinetable(openhdf5group, linetable):
    '''
    save linetable or a subset of it
    '''
    openhdf5group.attrs.create('info', 'line data for ions; major means the one used for the specwizard spectrum')
    for column in linetable.columns:
        dtype_df = linetable.dtypes[column]
        if isinstance(dtype_df, pd.CategoricalDtype):
            dtype_df = str
        openhdf5group.create_dataset(column, data=np.array(linetable[column]).astype(dtype_df))
    openhdf5group.create_dataset('linenames', data=np.array(linetable.index).astype(str))
    
def readlinetable(hdf5group):
    columns = {name: np.array(hdf5group[name]) for name in ['element', 'fosc', 'ion', 'lambda_angstrom', 'major']}
    index   = np.array(hdf5group['linenames'])
    table = pd.DataFrame(data=columns, index=index, columns=None, dtype=None, copy=False)
    if usecatdtype:
        table = table.astype({'ion':     ion_type,\
                      'lambda_angstrom': np.float,\
                      'fosc':    np.float,\
                      'element': element_type,\
                      'major':   bool})
    else:
        table = table.astype({
                      'lambda_angstrom': np.float,\
                      'fosc':    np.float,\
                      'major':   bool})
        table["ion"] = table["ion"].astype('category', categories=ion_list, ordered=False)
        table["element"] = table["element"].astype('category', categories=element_list, ordered=False)
    return table

# O VIII
o8major    = SpecLine('o8major',   'o8', 18.9671, 0.277, Atrans=2.57e12)   
o8minor    = SpecLine('o8minor',   'o8', 18.9725, 0.139, Atrans=2.58e12)
o8combo    = SpecLine('o8combo',   'o8', 18.9689, 0.416, combination=(o8major, o8minor))
o8_2major  = SpecLine('o8_2major', 'o8', 16.0055, 0.0527)
o8_2minor  = SpecLine('o8_2minor', 'o8', 16.0067, 0.0263)
o8_3major  = SpecLine('o8_3major', 'o8', 15.1765, 9.67E-03)
o8_3minor  = SpecLine('o8_3minor', 'o8', 15.1760, 1.93E-02)
o8lines    = IonLines('o8', 'oxygen', (o8major, o8minor, o8_2major, o8_2minor), o8major)
o8doublet  = IonLines('o8', 'oxygen', (o8major, o8minor), o8major)

# O VII
o7major    = SpecLine('o7major',   'o7', 21.6019, 0.696, Atrans=3.32e12)
o7r        = o7major
o7_2       = SpecLine('o7_2',      'o7', 18.6284, 0.146)
o7_3       = SpecLine('o7_3',      'o7', 17.7683, 5.52E-02)
o7_4       = SpecLine('o7_4',      'o7', 17.3960, 2.68E-02)
o7_5       = SpecLine('o7_5',      'o7', 17.2000, 1.51E-02)
o7lines    = IonLines('o7', 'oxygen', (o7major, o7_2), o7major)

# O VI
o6major    = SpecLine('o6major',   'o6', 1031.9261, 0.13250, Atrans=4.17e8)   
o6minor    = SpecLine('o6minor',   'o6', 1037.6167, 0.06580)
o6_3major  = SpecLine('o6_3major', 'o6', 150.1246, 8.84E-02)
o6_3minor  = SpecLine('o6_3minor', 'o6', 150.0893, 1.77E-01)
o6_4major  = SpecLine('o6_4major', 'o6', 115.8301, 2.47E-02)
o6_4minor  = SpecLine('o6_4minor', 'o6', 115.8215, 4.94E-02)
o6_5major  = SpecLine('o6_5major', 'o6', 104.8130, 2.13E-02)
o6_5minor  = SpecLine('o6_5minor', 'o6', 104.8130, 1.06E-02)
o6_6major  = SpecLine('o6_6major', 'o6', 99.6880, 1.13E-02)
o6_6minor  = SpecLine('o6_6minor', 'o6', 99.6880, 5.64E-03)
o6_2major  = SpecLine('o6_2major', 'o6', 22.0189, 0.351)
o6_2minor  = SpecLine('o6_2minor', 'o6', 22.0205, 0.174)
o6_7major  = SpecLine('o6_7major', 'o6', 21.7891, 1.99E-02)
o6_7minor  = SpecLine('o6_7minor', 'o6', 21.7881, 3.71E-02)
o6_8major  = SpecLine('o6_8major', 'o6', 19.3791, 6.54E-02)
o6_8minor  = SpecLine('o6_8minor', 'o6', 19.3789, 3.27E-02)
o6_9major  = SpecLine('o6_9major', 'o6', 19.1805, 1.26E-02)
o6_9minor  = SpecLine('o6_9minor', 'o6', 19.1798, 2.53E-02)
o6_10major = SpecLine('o6_10major', 'o6', 18.5870, 2.85E-02)
o6_10minor = SpecLine('o6_10minor', 'o6', 18.5869, 1.43E-02)

o6lines    = IonLines('o6', 'oxygen', (o6major, o6minor, o6_2major, o6_2minor), o6major)
o6doublet  = IonLines('o6', 'oxygen', (o6major, o6minor), o6major)               

# Fe XVII
fe17major  = SpecLine('fe17major',   'fe17', 15.0140, 2.72, Atrans=2.70e13)   
fe17minor  = SpecLine('fe17minor',   'fe17', 15.2610, 0.614) 
fe17_3     = SpecLine('fe17_3',      'fe17', 17.0550, 1.30E-01) 
fe17_4     = SpecLine('fe17_4',      'fe17', 16.7800, 1.14E-01) 
fe17_5     = SpecLine('fe17_5',      'fe17', 15.4560, 1.09E-02) 
fe17_6     = SpecLine('fe17_6',      'fe17', 13.8920, 3.47E-02) 
fe17_7     = SpecLine('fe17_7',      'fe17', 13.8260, 2.90E-01) 
fe17_8     = SpecLine('fe17_8',      'fe17', 12.6820, 2.02E-02) 
fe17_9     = SpecLine('fe17_9',      'fe17', 12.5220, 1.36E-02) 
fe17_10    = SpecLine('fe17_10',     'fe17', 12.2640, 2.85E-01) 
fe17_11    = SpecLine('fe17_11',     'fe17', 11.2509, 1.63E-01) 
fe17_12    = SpecLine('fe17_12',     'fe17', 11.1307, 1.26E-01) 
fe17_13    = SpecLine('fe17_13',     'fe17', 11.0250, 1.33E-02) 
fe17_14    = SpecLine('fe17_14',     'fe17', 11.0007, 9.52E-02) 
fe17_15    = SpecLine('fe17_15',     'fe17', 10.7709, 9.91E-02) 
fe17_16    = SpecLine('fe17_16',     'fe17', 10.6580, 5.87E-02) 
fe17_17    = SpecLine('fe17_17',     'fe17', 10.5014, 6.88E-02) 
fe17_18    = SpecLine('fe17_18',     'fe17', 10.3927, 3.81E-02) 
fe17_19    = SpecLine('fe17_19',     'fe17', 10.1023, 4.52E-02) 
fe17_20    = SpecLine('fe17_20',     'fe17', 9.6837,  2.60E-02) 
fe17_21    = SpecLine('fe17_21',     'fe17', 9.4514, 1.66E-02) 
fe17_22    = SpecLine('fe17_22',     'fe17', 1.7236, 1.03E-01) 
fe17lines  = IonLines('fe17', 'iron', (fe17major, fe17minor), fe17major)
fe17doublet= IonLines('fe17', 'iron', (fe17major, fe17minor), fe17major)                  

# C V
c5major    = SpecLine('c5major',   'c5', 40.2678, 0.648)   
c5_2       = SpecLine('c5_2',      'c5', 34.9728, 0.141)
c5_3       = SpecLine('c5_3',      'c5', 33.4262, 5.36E-02)
c5_4       = SpecLine('c5_4',      'c5', 32.7542, 2.61E-02)
c5_5       = SpecLine('c5_5',      'c5', 32.3998, 1.48E-02)

# C VI
c6major    = SpecLine('c6major',   'c6', 33.7342, 0.277)   
c6minor    = SpecLine('c6minor',   'c6', 33.7396, 0.139)
c6_2major  = SpecLine('c6_2major', 'c6', 28.4652, 0.0527)
c6_2minor  = SpecLine('c6_2minor', 'c6', 28.4663, 0.0263)
c6_3major  = SpecLine('c6_3major', 'c6', 26.9901, 9.67E-03)
c6_3minor  = SpecLine('c6_3minor', 'c6', 26.9896, 1.93E-02)
c6lines    = IonLines('c6', 'carbon', (c6major, c6minor, c6_2major, c6_2minor), c6major)
c6doublet  = IonLines('c6', 'carbon', (c6major, c6minor), c6major)

# N VII
n7major    = SpecLine('n7major',   'n7', 24.7792, 0.277)   
n7minor    = SpecLine('n7minor',   'n7', 24.7846, 0.139)
n7_2major  = SpecLine('n7_2major', 'n7', 20.9106, 2.63E-02)   
n7_2minor  = SpecLine('n7_2minor', 'n7', 20.9095, 5.27E-02)
n7_3major  = SpecLine('n7_3major', 'n7', 19.8261, 9.67E-03)   
n7_3minor  = SpecLine('n7_3minor', 'n7', 19.8257, 1.93E-02)
n7lines    = IonLines('n7', 'nitrogen', (n7major, n7minor), n7major)
n7doublet  = IonLines('n7', 'nitrogen', (n7major, n7minor), n7major)

# N VI
n6major    = SpecLine('n6major',   'n6', 28.7875, 0.675)  
n6_2       = SpecLine('n6_2',      'n6', 24.9000, 1.44E-01)
n6_3       = SpecLine('n6_3',      'n6', 23.7710, 5.46E-02)
n6_4       = SpecLine('n6_4',      'n6', 23.2770, 2.66E-02)
n6_5       = SpecLine('n6_5',      'n6', 23.0238, 1.50E-02)
n6lines    = IonLines('n6', 'nitrogen', (n6major,), n6major)      

# N V major, minor different in specwizard and Jelle's data -- using specwizard versions
n5major    = SpecLine('n5major',   'n5', 1238.821, 0.156000)  
n5minor    = SpecLine('n5minor',   'n5', 1242.804, 0.0770)
n5_2major  = SpecLine('n5_2major', 'n5', 209.3076, 7.96E-02)  
n5_2minor  = SpecLine('n5_2minor', 'n5', 209.2742, 1.59E-01)
n5_3major  = SpecLine('n5_3major', 'n5', 162.5644, 2.28E-02)  
n5_3minor  = SpecLine('n5_3minor', 'n5', 162.5557, 4.57E-02)
n5_4major  = SpecLine('n5_4major', 'n5', 147.4273, 9.99E-03)  
n5_4minor  = SpecLine('n5_4minor', 'n5', 147.4238, 2.00E-02)
n5_5major  = SpecLine('n5_5major', 'n5', 140.3581, 5.31E-03)  
n5_5minor  = SpecLine('n5_5minor', 'n5', 140.3563, 1.06E-02)
n5_6major  = SpecLine('n5_6major', 'n5', 209.3076, 7.96E-02)  
n5_6minor  = SpecLine('n5_6minor', 'n5', 209.2742, 1.59E-01)
n5_7major  = SpecLine('n5_7major', 'n5', 29.4150, 2.14E-01)  
n5_7minor  = SpecLine('n5_7minor', 'n5', 29.4135, 4.31E-01)
n5_8major  = SpecLine('n5_8major', 'n5', 29.1289, 2.06E-02)  
n5_8minor  = SpecLine('n5_8minor', 'n5', 29.1279, 3.91E-02)


# Ne VIII
ne8major   = SpecLine('ne8major',  'ne8', 770.409, 0.103, Atrans=5.79e8) 
ne8minor   = SpecLine('ne8minor',  'ne8', 780.324, 0.0505) 
ne8_2major = SpecLine('ne8_2major','ne8', 88.0790, 2.01E-01) 
ne8_2minor = SpecLine('ne8_2minor','ne8', 88.1170, 1.00E-01) 
ne8_3major = SpecLine('ne8_3major','ne8', 67.3818, 5.41E-02) 
ne8_3minor = SpecLine('ne8_3minor','ne8', 67.3818, 2.71E-02) 
ne8_4major = SpecLine('ne8_4major','ne8', 60.7958, 2.30E-02) 
ne8_4minor = SpecLine('ne8_4minor','ne8', 60.7958, 1.15E-02) 
ne8_5major = SpecLine('ne8_5major','ne8', 57.7471, 1.21E-02) 
ne8_5minor = SpecLine('ne8_5minor','ne8', 57.7471, 6.04E-03)
ne8_6major = SpecLine('ne8_6major','ne8', 13.6553, 1.87E-01) 
ne8_6minor = SpecLine('ne8_6minor','ne8', 13.6533, 3.81E-01)

# Ne IX
ne9major   = SpecLine('ne9major',   'ne9', 13.4471, 0.724, Atrans=8.90e12)   
ne9_2      = SpecLine('ne9_2',      'ne9', 11.5466, 1.49E-01)
ne9_3      = SpecLine('ne9_3',      'ne9', 11.0003, 5.61E-02)
ne9_4      = SpecLine('ne9_4',      'ne9', 10.7643, 2.71E-02)
ne9_5      = SpecLine('ne9_5',      'ne9', 10.6426, 1.53E-02)
ne9lines   = IonLines('ne9', 'neon', (ne9major,), ne9major)      

# Ne X
ne10major   = SpecLine('ne10major', 'ne10', 12.1321, 2.77E-01) 
ne10minor   = SpecLine('ne10minor', 'ne10', 12.1375, 1.39E-01) 
ne10_2major = SpecLine('ne10_2major', 'ne10', 10.2396, 2.63E-02) 
ne10_2minor = SpecLine('ne10_2minor', 'ne10', 10.2385, 5.27E-02) 
ne10_3major = SpecLine('ne10_3major', 'ne10', 9.7085, 9.67E-03) 
ne10_3minor = SpecLine('ne10_3minor', 'ne10', 9.7080, 1.93E-02) 

lambda_rest = {
               'h1':        1156.6987178884301,\
               'c3':        977.0201,\
               'c4major':   1548.2041,\
               'c5major':   40.2678,\
               'ne8major':  770.409,\
               'o4':        787.711,\
               'o5':        629.730,\
               'si3':       1206.500,\
               'si4major':  1393.76018,\
               'lyalpha':   1215.67,\
               }
fosc ={
       'h1':        0.5644956,\
       'lyalpha':   0.4162,\
       } 
        
majorlines = [o8major,\
              o7major,\
              o6major,\
              fe17major,\
              c6major,\
              c5major,\
              n7major,\
              n6major,\
              n5major,\
              ne9major,\
              ne8major,\
              ne10major,\
              ]

otherlines = [c6minor,\
              c6_2major,\
              c6_2minor,\
              c6_3major,\
              c6_3minor,\
              c5_2,\
              c5_3,\
              c5_4,\
              c5_5,\
              n7minor,\
              n7_2major,\
              n7_2minor,\
              n7_3major,\
              n7_3minor,\
              n6_2,\
              n6_3,\
              n6_4,\
              n6_5,\
              n5minor,\
              n5_2major,\
              n5_2minor,\
              n5_3major,\
              n5_3minor,\
              n5_4major,\
              n5_4minor,\
              n5_5major,\
              n5_5minor,\
              n5_6major,\
              n5_6minor,\
              n5_7major,\
              n5_7minor,\
              n5_8major,\
              n5_8minor,\
              o8minor,\
              o8_2major,\
              o8_2minor,\
              o8_3major,\
              o8_3minor,\
              o7_2,\
              o7_3,\
              o7_4,\
              o7_5,\
              o6minor,\
              o6_2major,\
              o6_2minor,\
              o6_3major,\
              o6_3minor,\
              o6_4major,\
              o6_4minor,\
              o6_5major,\
              o6_5minor,\
              o6_6major,\
              o6_6minor,\
              o6_7major,\
              o6_7minor,\
              o6_8major,\
              o6_8minor,\
              o6_9major,\
              o6_9minor,\
              o6_10major,\
              o6_10minor,\
              ne8minor,\
              ne8minor,\
              ne8_2major,\
              ne8_2minor,\
              ne8_3major,\
              ne8_3minor,\
              ne8_4major,\
              ne8_4minor,\
              ne8_5major,\
              ne8_5minor,\
              ne8_6major,\
              ne8_6minor,\
              ne9_2,\
              ne9_3,\
              ne9_4,\
              ne9_5,\
              ne10minor,\
              ne10_2minor,\
              ne10_2major,\
              ne10_3minor,\
              ne10_3major,\
              fe17minor,\
              fe17_3,\
              fe17_4,\
              fe17_5,\
              fe17_6,\
              fe17_7,\
              fe17_8,\
              fe17_9,\
              fe17_10,\
              fe17_11,\
              fe17_12,\
              fe17_13,\
              fe17_14,\
              fe17_15,\
              fe17_16,\
              fe17_17,\
              fe17_18,\
              fe17_19,\
              fe17_20,\
              fe17_21,\
              fe17_22,\
             ]

alllines = {line.name: line.getseries(True) for line in majorlines}
alllines.update({line.name: line.getseries(False) for line in otherlines })
linetable = pd.DataFrame(alllines)
linetable = linetable.transpose()
if usecatdtype:
    linetable = linetable.astype({'ion':     ion_type,\
                      'lambda_angstrom': np.float,\
                      'fosc':    np.float,\
                      'element': element_type,\
                      'major':   bool})
else:
    linetable = linetable.astype({\
                      'lambda_angstrom': np.float,\
                      'fosc':    np.float,\
                      'major':   bool})
    linetable["ion"] = linetable["ion"].astype('category', categories=ion_list, ordered=False)
    linetable["element"] = linetable["element"].astype('category', categories=element_list, ordered=False)
# a few utility functions/variables for linedata
def get_major(ion):
    return linetable.loc[np.logical_and(linetable['ion'] == ion, linetable['major'])].index[0]

def get_linenames(ion):
    return list(linetable.loc[linetable['ion'] == ion].index)

ions = set(linetable['ion'])

#from Lan & Fukugita 2017, citing e.g. B.T. Draine, 2011: Physics of the interstellar and intergalactic medium
# ew in angstrom, Nion in cm^-2
# Lan and Fukugita use rest-frame wavelengths; possibly better match using redshifted wavelengths. (deltaredshift uses the same value as in coldens hist. calc., then EW useslambda_rest*deltaredshift)
# 1.13e20 comes from (np.pi*4.80320427e-10**2/9.10938215e-28/2.99792458e10**2 / 1e8)**-1 = ( np.pi*e[statcoulomb]**2/(m_e c**2) *Angstrom/cm )**-1

def lingrowthcurve(ew, ion):
    if isinstance(ion, str):
        lambda_rest = linetable['lambda_angstrom'][ion]
        fosc = linetable['fosc'][ion]
    else:
        try:
            lambda_rest = ion.lambda_angstrom
            fosc = ion.fosc
        except AttributeError:
            lambda_rest = ion.major.lambda_angstrom
            fosc = ion.major.fosc
    return (np.pi* c.electroncharge**2 / (c.electronmass * c.c**2) *1e-8)**-1 /(fosc * lambda_rest**2) * ew
  
def lingrowthcurve_inv(Nion, ion):
    '''
    returns EW in A
    '''
    if isinstance(ion, str):
        lambda_rest = linetable['lambda_angstrom'][ion]
        fosc = linetable['fosc'][ion]
    elif hasattr(ion, '__len__'): # tuple, list or something
        return np.sum([lingrowthcurve_inv(Nion, _ion) for _ion in ion], axis=0)
    else:
        try:
            lambda_rest = ion.lambda_angstrom
            fosc = ion.fosc
        except AttributeError:
            lines = ion.speclines.keys()
            return np.sum([lingrowthcurve_inv(Nion, ion.speclines[line]) for line in lines], axis=0)
    return Nion * (fosc * lambda_rest**2) * (np.pi* c.electroncharge**2 / (c.electronmass * c.c**2) * 1e-8)

def linflatcurveofgrowth_inv(Nion, b, ion):
    '''
    equations from zuserver2.star.ucl.ac.uk/~idh/PHAS2112/Lectures/Current/Part4.pdf
    b in cm/s
    Nion in cm^-2
    out: EW in Angstrom
    '''
    # central optical depth; 1.13e20,c and pi come from comparison to the linear equation
    #print('lambda_rest[ion]: %f'%lambda_rest[ion])
    #print('fosc[ion]: %f'%fosc[ion])
    #print('Nion: ' + str(Nion))
    #print('b: ' + str(b))
    if not hasattr(Nion, '__len__'):
        Nion = np.array([Nion])
    
    if ion == 'o8_assingle': # backwards compatibiltiy
        ion = o8combo
            
    if hasattr(ion, 'major') or (ion in elements_ion.keys()): # ion or IonLines, not line
        if hasattr(ion, 'major'):
            #fosc_m   = ion.major.fosc
            wavelen_m = ion.major.lambda_angstrom
            lines = ion.speclines.keys()
            fosc    = {line: ion.speclines[line].fosc for line in lines}
            wavelen = {line: ion.speclines[line].lambda_angstrom for line in lines}
        else:
            ionlines = linetable.loc[linetable['ion'] == ion]
            lines    = ionlines.index 
            mline     = ionlines.loc[ionlines['major']]
            #fosc_m    = mline['fosc'][0]
            wavelen_m = mline['lambda_angstrom'][0]
            dct  = ionlines.to_dict()
            fosc = dct['fosc']
            wavelen = dct['lambda_angstrom']
            
        tau0s = np.array([(np.pi**0.5 * c.electroncharge**2 / (c.electronmass * c.c) * 1e-8) * wavelen[line] * fosc[line] * Nion / b for line in lines]).T
        xoffsets = (c.c / b)* (np.array([wavelen[line] for line in lines]) - wavelen_m) / wavelen_m # it shouldn't matter relative to which the offset is taken
        #print(tau0s)
        #print(xoffsets)     
        prefactor = wavelen_m / c.c * b # just use the average here
        # absorption profiles are multiplied to get total absorption
        
        integral = np.array([si.quad(lambda x: 1- np.exp(np.sum(-taus*np.exp(-1*(x-xoffsets)**2),axis=0)),-np.inf,np.inf) for taus in tau0s])

    else:
        if hasattr(ion, 'fosc'):
            fosc  = ion.fosc
            wavelen = ion.lambda_angstrom
        else:
            line     = linetable.loc[ion]
            fosc     = line['fosc']
            wavelen  = line['lambda_angstrom']
            
        tau0 = (np.pi**0.5* c.electroncharge**2 / (c.electronmass * c.c) *1e-8) * wavelen * fosc * Nion / b
        prefactor = wavelen / c.c * b
        #def integrand(x):
        #    1- np.exp(-tau0*np.exp(-1*x**2))
        integral = np.array([si.quad(lambda x: 1. - np.exp(-tau*np.exp(-1*x**2)), -np.inf, np.inf) for tau in tau0])

    if np.max(integral[:,1]/integral[:,0]) > 1e-5:
        print('Warning: check integration errors in linflatcurveofgrowth_inv')
    return prefactor * integral[:, 0]

def linflatcurveofgrowth_inv_faster(Nion, b, ion):
    '''
    equations from zuserver2.star.ucl.ac.uk/~idh/PHAS2112/Lectures/Current/Part4.pdf
    b in cm/s
    Nion in cm^-2
    out: EW in Angstrom
    
    more approx integration, should be fast enough for fitting
    '''
    # central optical depth; 1.13e20,c and pi come from comparison to the linear equation
    #print('lambda_rest[ion]: %f'%lambda_rest[ion])
    #print('fosc[ion]: %f'%fosc[ion])
    #print('Nion: ' + str(Nion))
    #print('b: ' + str(b))
    if not hasattr(Nion, '__len__'):
        Nion = np.array([Nion])
    
    if ion == 'o8_assingle': # backwards compatibiltiy
        ion = o8combo
    # tweak for precision/speed tradeoff
    xsample = np.arange(-12., 12.005, 0.1)
    
    if hasattr(ion, 'major') or (ion in elements_ion.keys()): # ion or IonLines, not line
        if hasattr(ion, 'major'):
            #fosc_m   = ion.major.fosc
            wavelen_m = ion.major.lambda_angstrom
            lines = ion.speclines.keys()
            fosc    = {line: ion.speclines[line].fosc for line in lines}
            wavelen = {line: ion.speclines[line].lambda_angstrom for line in lines}
        else:
            ionlines = linetable.loc[linetable['ion'] == ion]
            lines    = ionlines.index 
            mline     = ionlines.loc[ionlines['major']]
            #fosc_m    = mline['fosc'][0]
            wavelen_m = mline['lambda_angstrom'][0]
            dct  = ionlines.to_dict()
            fosc = dct['fosc']
            wavelen = dct['lambda_angstrom']
        
        # axis 0: input Nion, axis 1: multiplet component
        tau0s = np.array([(np.pi**0.5 * c.electroncharge**2 / (c.electronmass * c.c) * 1e-8) * wavelen[line] * fosc[line] * Nion / b for line in lines]).T
        xoffsets = (c.c / b) * (np.array([wavelen[line] for line in lines]) - wavelen_m) / wavelen_m # it shouldn't matter relative to which the offset is taken
        #print(tau0s)
        #print(xoffsets)     
        prefactor = wavelen_m / c.c * b # just use the average here
        # absorption profiles are multiplied to get total absorption
        # x is the scale over which the exponential changes:
        # axis 0: Nion, integration, axis 1: lines, axis 2: integration x
        integrand = 1 - np.exp(\
                          np.sum(-tau0s[:, :, np.newaxis] *\
                                 np.exp(-1 * (xsample[np.newaxis, np.newaxis, :] \
                                              - xoffsets[np.newaxis, :, np.newaxis])**2),\
                                 axis=1)\
                               )
        integral = si.simps(integrand, x=xsample, axis=1)

    else:
        if hasattr(ion, 'fosc'):
            fosc  = ion.fosc
            wavelen = ion.lambda_angstrom
        else:
            line     = linetable.loc[ion]
            fosc     = line['fosc']
            wavelen  = line['lambda_angstrom']
            
        tau0 = (np.pi**0.5* c.electroncharge**2 / (c.electronmass * c.c) *1e-8) * wavelen * fosc * Nion / b
        prefactor = wavelen / c.c * b
        #def integrand(x):
        #    1- np.exp(-tau0*np.exp(-1*x**2))
        integrand = 1 - np.exp(\
                              -tau0[:, np.newaxis] *\
                                 np.exp(-1 * (xsample[np.newaxis, :])**2)
                               )
        integral = si.simps(integrand, x=xsample, axis=1)

    return prefactor * integral[:]

def linflatdampedcurveofgrowth_inv(Nion, b, ion):
    '''
    equations from zuserver2.star.ucl.ac.uk/~idh/PHAS2112/Lectures/Current/Part4.pdf
    b in cm/s
    Nion in cm^-2
    out: EW in Angstrom
    
    '''
    # central optical depth; 1.13e20,c and pi come from comparison to the linear equation
    #print('lambda_rest[ion]: %f'%lambda_rest[ion])
    #print('fosc[ion]: %f'%fosc[ion])
    #print('Nion: ' + str(Nion))
    #print('b: ' + str(b))
    if not hasattr(Nion, '__len__'):
        Nion = np.array([Nion])
    
    if ion == 'o8_assingle': # backwards compatibiltiy
        ion = o8combo
    
    # multiple lines in one go
    if hasattr(ion, 'major') or (ion in elements_ion.keys()): # ion or IonLines, not line
        if hasattr(ion, 'major'):
            #fosc_m   = ion.major.fosc
            wavelen_m = ion.major.lambda_angstrom
            lines = ion.speclines.keys()
            fosc    = {line: ion.speclines[line].fosc for line in lines}
            wavelen = {line: ion.speclines[line].lambda_angstrom for line in lines}
            atrans  = {line: ion.speclines[line].Atrans for line in lines}
        else:
            ionlines = linetable.loc[linetable['ion'] == ion]
            lines    = ionlines.index 
            mline     = ionlines.loc[ionlines['major']]
            #fosc_m    = mline['fosc'][0]
            wavelen_m = mline['lambda_angstrom'][0]
            dct  = ionlines.to_dict()
            fosc = dct['fosc']
            wavelen = dct['lambda_angstrom']
            atrans = dct['Atrans']
        wavelen_m = np.sum([fosc[line] * wavelen[line] for line in lines]) \
                   / np.sum([fosc[line] for line in lines]) 
        
        # tweak for precision/speed tradeoff; units: frequency (s**-1)
        nucen = c.c / wavelen_m
        nus = c.c / np.array([wavelen[line] for line in lines])
        xoffsets = nus - nucen
        
        # axis 0: input Nion, axis 1: multiplet component
        sigma = b / (wavelen_m * 1e-8 * 2.**0.5) # gaussian sigma from b parameter
        hwhm_cauchy = np.array([atrans[line] for line in lines]) / (4. * np.pi)
        
        snus = np.sort(nus)
        xo_min = np.min(np.diff(np.sort(snus)))
        xo_max = snus[-1] - snus[0]
        delta_rel = 0.05 * sigma
        delta_rel_cen = min(0.1 * xo_min, delta_rel)
        range_hires = (2. * np.floor((snus[0] - nucen) / delta_rel_cen) * delta_rel_cen,\
                       2. * np.ceil((snus[-1] - nucen) / delta_rel_cen) * delta_rel_cen)
        diff_max = 300. * np.max(hwhm_cauchy) + 20. * sigma + xo_max
        print('dw EW calc: x samplesize ~ {}'.format(2. * diff_max / delta_rel +\
                                                   (range_hires[1] - range_hires[0]) / delta_rel_cen))
        xsample_lo = np.arange(range_hires[0] - 0.5 * delta_rel_cen, -diff_max, -delta_rel)[::-1]
        xsample_hi = np.arange(range_hires[1] + 0.5 * delta_rel_cen, diff_max, delta_rel)
        xsample_mi = np.arange(range_hires[0], range_hires[1] + 0.5 * delta_rel_cen, delta_rel_cen)
        xsample = np.append(xsample_lo, xsample_mi)
        xsample = np.append(xsample, xsample_hi)
        
        # axis 0: Nion, axis 1: lines, axis 2: integration x
        z_in = (xsample[np.newaxis, np.newaxis, :]  \
                 - xoffsets[np.newaxis, :, np.newaxis] \
                 + hwhm_cauchy[np.newaxis, : , np.newaxis] * 1j) \
                / (sigma * 2.**0.5)       
        vps = np.real(wofz(z_in)) / (sigma * (2. * np.pi)**0.5)
        # norm * rework from fadeeva function 
        norms = np.array([np.pi * c.electroncharge**2 / (c.electronmass * c.c) *\
                          fosc[line] * Nion \
                          for line in lines]).T            
        tau = np.sum(vps * norms[:, :, np.newaxis], axis=1) # total opt. depth
        prefactor = (wavelen_m)**2 / c.c * 1e-8 # just use the average here
        
        integrand = 1 - np.exp(-1. * tau)
        integral = si.simps(integrand, x=xsample, axis=1)

    else: # singlet line
        if hasattr(ion, 'fosc'):
            fosc  = ion.fosc
            wavelen = ion.lambda_angstrom
            atrans  = ion.Atrans
        else:
            line     = linetable.loc[ion]
            fosc     = line['fosc']
            wavelen  = line['lambda_angstrom']
            atrans   = line['Atrans']
        
        sigma = b / (wavelen * 1e-8 * 2.**0.5) # gaussian sigma from b parameter
        hwhm_cauchy = atrans / (4. * np.pi)
             
        delta_rel = 0.05 * sigma
        diff_max = 300. * hwhm_cauchy + 20. * sigma
        xsample = np.arange(-diff_max, diff_max + 0.5 * delta_rel, delta_rel)
        
        z_in = (xsample + hwhm_cauchy * 1j) \
                / (sigma * 2.**0.5)       
        vps = np.real(wofz(z_in)) / (sigma * (2. * np.pi)**0.5) 
        #print(si.simps(vps, x=xsample)) # checked: this works out to ~1
        
        # axis 0: Nion, axis 1: integration x
        norm = np.pi * c.electroncharge**2 / (c.electronmass * c.c) *\
               fosc * Nion             
        tau = vps[np.newaxis, :] * norm[:, np.newaxis] # total opt. depth
        prefactor = (wavelen)**2 / c.c * 1e-8  
        
        integrand = 1 - np.exp(-1. * tau)
        integral = si.simps(integrand, x=xsample, axis=1)

    return prefactor * integral[:]


        
        
def nion_ppv_from_tauv(tau,ion):
    '''
    Same reference as linflatcurveofgrowth; tau here is tau_v (result of specwizard projection), not tau_nu or tau_lambda
    returns nion in units of cm^-2 (rest-frame km/s)^-1: conversion to line of 
    sight distance depends on cosmological parameters.
    '''
    if hasattr(ion, 'major') or (ion in elements_ion.keys()): # ion or IonLines, not line
        if hasattr(ion, 'major'):
            fosc    = ion.major.fosc
            wavelen = ion.major.lambda_angstrom
        else:
            ionlines = linetable.loc[linetable['ion'] == ion]
            mline   = ionlines.loc[ionlines['major']]
            fosc    = mline['fosc'][0]
            wavelen = mline['lambda_angstrom'][0]
    else:
        if hasattr(ion, 'fosc'):
            fosc  = ion.fosc
            wavelen = ion.lambda_angstrom
        else:
            line     = linetable.loc[ion]
            fosc     = line['fosc']
            wavelen  = line['lambda_angstrom']
    return (np.pi * c.electroncharge**2 / (c.electronmass * c.c**2))**-1  * 1. / (fosc * wavelen**2 * 1.e-8**2) * tau 
    # tau at that pixel -> total Nion represented there 
    # (really Nion*normailised spectrum, but we've got the full tau spectra, so no need to factor that out)


def testbparfit_faster():
    bvals = np.arange(15., 300., 5.) * 1e5 
    csample = 10**np.arange(12., 17., 0.1) 
    testdoub = o8doublet
    testsing = o7major
    
    allgood = True
    print('Testing multiplet {}'.format(testdoub))
    for bval in bvals:
         fvals = linflatcurveofgrowth_inv_faster(csample, bval, testdoub)
         ovals = linflatcurveofgrowth_inv(csample, bval, testdoub)
         if not np.allclose(fvals, ovals):
             print('Issue for b={b}'.format(b=bval))
             allgood = False
             
    print('Testing singlet {}'.format(testsing))
    for bval in bvals:
         fvals = linflatcurveofgrowth_inv_faster(csample, bval, testsing)
         ovals = linflatcurveofgrowth_inv(csample, bval, testsing)
         if not np.allclose(fvals, ovals):
             print('Issue for b={b}'.format(b=bval))
             allgood = False
    
    if allgood:
        print('Test succeeded!')
        
def getEWdiffgrid():
    outfile = '/net/luttero/data2/paper2/' + \
              'EWdiffs_dampingwings.hdf5'
    
    Nsample = np.linspace(12., 20., 200)
    bsample = np.arange(2., 500.5, 1.) * 1e5
    uselines = {'o7': o7major,\
                'o8': o8doublet,\
                'o6': o6major,\
                'ne8': ne8major,\
                'ne9': ne9major,\
                'fe17': fe17major,\
                }
    with h5py.File(outfile, 'w') as fo:
        for ion in uselines:
            line = uselines[ion]
            grp = fo.create_group(ion)
            lgrp = grp.create_group('lines')
            if hasattr(line, 'major'):
                saveionlines(lgrp, line)
            else:
                savelinedata(lgrp, line)
            EWgrid_lnf = np.array([linflatcurveofgrowth_inv_faster(10**Nsample,\
                                                                       b,\
                                                             uselines[ion])
                                  for b in bsample])
            EWgrid_dmp = np.array([linflatdampedcurveofgrowth_inv(10**Nsample,\
                                                                       b,\
                                                             uselines[ion])
                                  for b in bsample])
            
            grp.create_dataset('logNcm2', data=Nsample)
            grp.create_dataset('b_cmps', data=bsample)
            grp.create_dataset('EW_Angstrom_gaussian', data=EWgrid_lnf)
            grp.create_dataset('EW_Angstrom_voigt', data=EWgrid_dmp)
