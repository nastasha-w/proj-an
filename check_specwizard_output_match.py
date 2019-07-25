#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:30:18 2018

@author: wijers

For testing if specwizard outputs from different systems/versions give the same results
"""
import numpy as np
import h5py

tolerance = 1e-6

def comparearrays(grp1, grp2, name, verbose=False):
    try:
        arr1 = np.array(grp1[name])
    except KeyError:
        print('File 1 does not contain array %s'%name)
        return False
    try:
        arr2 = np.array(grp2[name])
    except KeyError:
        print('File 2 does not contain array %s'%name)
        return False 
    if arr1.shape != arr2.shape:
        print('Files 1 and 2 have different shapes %s and %s for array %s'%(arr1.shape, arr2.shape, name))
        return False
    else:
        exactmatch = np.all(arr1 == arr2)
        if exactmatch:
            if verbose:
                print('Exact match for array %s'%(name))
            return True
        else:
            reldiffs = arr1 / arr2
            absdiffs = arr1 - arr2
            # switch from relative to absolute differences if one of the array values is close to zero
            reldiffs = np.abs(reldiffs[np.logical_and(np.abs(arr1) > tolerance, np.abs(arr2) > tolerance)] - 1.)
            absdiffs = np.abs(absdiffs[np.logical_or(np.abs(arr1) <= tolerance, np.abs(arr2) <= tolerance)])
            if len(reldiffs) > 0:
                maxreldiff = np.max(reldiffs)
            else:
                maxreldiff = 0.
            if len(absdiffs) > 0: 
                maxabsdiff = np.max(absdiffs)
            else:
                maxabsdiff = 0.
            if maxreldiff <= tolerance and maxabsdiff <= tolerance:
                if verbose:
                    print('Match within tolerance for array %s'%(name))
                return True
            else:
                print('Mismatch within tolerance %s for array %s: differences %s, %s'%(tolerance, name, maxreldiff, maxabsdiff))
                return False

def compareattributes(grp1, grp2, name, verbose=False, exceptions=None):
    try:
        keys1 = grp1[name].attrs.keys()
    except KeyError:
        print('File 1 does not contain array %s'%name)
        return False
    try:
        keys2 = grp2[name].attrs.keys()
    except KeyError:
        print('File 2 does not contain array %s'%name)
        return False
    if set(keys1) != set(keys2):
        print('Attribute keys for %s do not match'%(name))
        return False
    else:
        if exceptions is not None:
            for exception in exceptions:
                keys1.remove(exception)
        keysmatch = np.array([np.all(np.array(grp1[name].attrs[key] == grp2[name].attrs[key])) for key in keys1])
        allmatch = np.all(keysmatch)
        if not allmatch:
            print('Attributes for %s do not all match'%(name))
            for ind in range(len(keys1)):
                if not keysmatch[ind]:
                    print('%s:\t %s,\t%s'%(key, grp1[name].attrs[key], grp2[name].attrs[key]))
            return False
        else:
            if verbose:
                print('Attributes for %s match'%(name))
            return True

def comparegroups(file1, file2, name, verbose=False, exceptions=None):
    '''
    Recursive function: compares a group and all its subgroups
    '''
    if exceptions is not None:
        if name in exceptions.keys():
            exception = exceptions[name]
        else:
            exception = None
    else:
        exception = None
    attrsmatch = compareattributes(file1, file2, name, verbose=verbose, exceptions=exception)
    # retrieves full paths for groups, dataset names only for datasets
    subs1 = file1[name].keys()
    subs2 = file2[name].keys()
    datasets1 = set([sub if file1[name].get(sub, getclass=True) == h5py.Dataset else\
                 None for sub in subs1])
    datasets2 = set([sub if file2[name].get(sub, getclass=True) == h5py.Dataset else\
                 None for sub in subs2])
    if datasets1 != datasets2:
        print('%s contains different datasets in the two files'%name)
        return False
    else:
        if None in datasets1:
            datasets1.remove(None)
        dsmatch = np.all(np.array([comparearrays(file1, file2, name + '/' + ds, verbose=verbose) for ds in datasets1])) 
        dsmatch &= np.all(np.array([compareattributes(file1, file2, name + '/' + ds, verbose=verbose) for ds in datasets1]))
    subgroups1 = set([sub if file1[name].get(sub, getclass=True) == h5py.Group else\
                 None for sub in subs1])
    subgroups2 = set([sub if file2[name].get(sub, getclass=True) == h5py.Group else\
                 None for sub in subs2])
    if subgroups1 != subgroups2:
        print('%s contains different subgroups in the two files'%name)
        return False
    else:
        if None in subgroups1:
            subgroups1.remove(None)
        subsmatch = np.all(np.array([comparegroups(file1, file2, name + '/' + subname, verbose=verbose, exceptions=exceptions) for subname in subgroups1])) 
    if dsmatch and attrsmatch and subsmatch:
        return True
    else:
        return False
    
def compare_specwizard_outputs(filen1, filen2, verbose=False):
    '''
    Intended for output matching the first n spectra; matching other spectra
    would require matching e.g. positions rather than spectrum numbers
    
    specwizardruntimeparameters may differ: outputdir, los_coordinates_file, SpectrumFile
    '''
    with h5py.File(filen1, 'r') as file1,\
         h5py.File(filen2, 'r') as file2:
        # don't compare 'Projection': that will only match is all the sightlines are the same
        notspectrumgroups = ['Constants',\
                             'Header',\
                             'Parameters',\
                             'Units']
        # these atributes are allowed to differ
        exceptions = {'Parameters/SpecWizardRuntimeParameters': ['outputdir', 'los_coordinates_file', 'SpectrumFile']}
        match_notspectra = np.all(np.array([comparegroups(file1, file2, groupn, verbose=verbose, exceptions=exceptions) for groupn in notspectrumgroups]))
        match_vhubble = comparearrays(file1, file2, 'VHubble_KMpS', verbose=verbose)
        
        spectrumgroups_file1 = [int(group[8:]) if 'Spectrum' in group else None for group in file1.keys()]
        spectrumgroups_file2 = [int(group[8:]) if 'Spectrum' in group else None for group in file2.keys()]
        spectrumgroups_both = list(set(spectrumgroups_file1) & set(spectrumgroups_file2))
        spectrumgroups_both.remove(None)
        spectrumgroups_both.sort()
        if verbose:
            if np.all(np.diff(np.array(spectrumgroups_both)) < 1.1):
                print('Both files contain spectra %s - %s; comparing these'%(min(spectrumgroups_both), max(spectrumgroups_both)))
            else:
                print('Both files contain spectra %s; comparing these'%spectrumgroups_both)
        match_spectra = np.all(np.array([comparegroups(file1, file2, 'Spectrum%i'%groupn, verbose=verbose) for groupn in spectrumgroups_both]))
        
        if verbose:
            print('\n')
        print('Velocity bins match: %s'%match_vhubble)
        print('Projection and simulation parameters match: %s'%match_notspectra)
        print('Corresponding spectra match: %s'%match_spectra)
        
    return match_vhubble and match_notspectra and match_spectra