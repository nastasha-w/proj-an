#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:08:35 2020

@author: wijers

Check whether the eagle output files are ok
args:    simnum,    snapnum, variation, filetype in that order
         simnum:    e.g. 'L0100N1504'
         snapnum:   snapshot/snipshot number (integer)
         variation: e.g. 'REFERENCE', 'RECALIBRATED'
         filetype: 'snap', 'snip', 'particles', 'particles_snip', 'group',
                   'group_snip', 'sub', 'sub_snip'
options: --basic (default): check file read-in
         --full: check for missing and corrupted files as well
         --skipbasic: skip the simple read-in check (this takes a while for
                      large boxes)
"""

import h5py 
import glob
import sys

import projection_classes as pc

def parse_commandline():
    kwargs = {}   
    args = sys.argv
    #if sys.version.split('.')[0] == '3':
    #    args = [arg.decode() for arg in args]
    
    simnum, snapnum, variation, filetype = args[1:5]
    snapnum = int(snapnum)
    kwargs['simnum'] = simnum
    kwargs['snapnum'] = snapnum
    kwargs['variation'] = variation
    kwargs['file_type'] = filetype
    
    if '--basic' in args:
        kwargs['full'] = False
    elif '--full' in args:
        kwargs['full'] = True
    else:
        kwargs['full'] = False
    if '--skipbasic' in args:
        kwargs['basic'] = False
    else:
        kwargs['basic'] = True     
    return kwargs

def initial_check(simfile):
    '''
    just try reading in an array
    '''
    if simfile.filetype in ['snap', 'snip', 'particles', 'particles_snip']:
        readarr = 'PartType0/Mass'
    elif simfile.filetype in [ 'group', 'group_snip', 'sub', 'sub_snip']:
        readarr = 'FOF/GroupLength'
    try:
        simfile.readarray(readarr)
    except Exception as err:
        print('Array {arr} read-in failed; error was:'.format(arr=readarr))
        raise err

def check_files(simfile):
    '''
    Check if the individual files seem ok
    '''
    
    matchfiles = glob.glob(simfile.filenamebase + '*.hdf5')
    firsttry = matchfiles[0]
    with h5py.File(firsttry, 'r') as _f:
        numfiles = _f['Header'].attrs['NumFilesPerSnapshot']
    
    missing = []
    if len(matchfiles) != numfiles:
        print('Files are missing: {num} out of {tot}'.format(num=numfiles - len(matchfiles),\
                                                             tot=numfiles))
        base = simfile.filenamebase + '{}.hdf5'
        for i in range(numfiles):
            ft = base.format(i)
            if ft not in matchfiles:
                print('Missing {}'.format(ft))
                missing.append(ft)
    
    fails = []
    for filen in matchfiles:
        try:
            with h5py.File(filen, 'r'):
                pass # just need to see if it's a valid hdf5 file
        except Exception as err:
            print('File {} generated an error:'.format(filen))
            print(err)
            fails.append(filen)
    
    return {'corrupted': fails, 'missing': missing}

def main():
    kwargs = parse_commandline()
    simfile = pc.Simfile(kwargs['simnum'], kwargs['snapnum'],\
                         kwargs['variation'], file_type=kwargs['file_type'])
    
    if kwargs['basic']:
        try:
            initial_check(simfile)
            print('passed initial check')
            return 0
        except Exception as err:
            print('failed inital check:')
            print(err)
        
    if kwargs['full']:
        fails = check_files(simfile)
        return fails
    

if __name__ == '__main__':
    main()
        
            

    