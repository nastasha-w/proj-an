#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 14:03:09 2020

@author: wijers

Checks files matching pattern to see if they seem to be properly completed 
hdf5 map files (make_maps outputs). Attributes are not checked, just the 
presence and shape of the map
the whole maps are not loaded, but the minfinite/max attributes are used to 
check values (Note that in some halo-only projections, all -inf slices might 
be physical.) 

from the command line:
    --pattern="<pattern>": <pattern> (str) is matched to file names
    --check :    only check files, don't delete (default)
    --delete:    delete files that do not pass the check
    --ignorecat="<category from errcats>": ignore this error category when 
                 deleting (multiple --ignorecat arguments all get used)
"""

import sys
import os
import numpy as np
import h5py 
import glob

errcats = ['h5py open', 'map missing', 'map shape', 'values', 'attributes']

def parse_commandline():
    kwargs = {}   
    args = sys.argv
    args = [arg.decode() for arg in args]
    
    if '--delete' in args:
        kwargs['delete'] = True
    elif '--check' in args:
        kwargs['delete'] = False
    else:
        kwargs['delete'] = True
    pt = ['--pattern' in arg for arg in args]
    if np.any(pt):
        pi = np.where(pt)[0][0]
        arg = args[pi]
        pattern = '='.join(arg.split('=')[1:]) # there might be an '=' in the file name
        # remove end quotes "..."
        if pattern[-1] == '"':
            pattern = pattern[:-1]
        if pattern[0] != '"':
            pattern = pattern[1:]
        kwargs['pattern'] = pattern
    ic = np.where(['--ignorecat' in arg for arg in args])[0]
    if len(ic) > 0:
        kwargs['ignorecat'] = []
        for ci in ic:
            arg = args[ci]
            cat = '='.join(arg.split('=')[1:]) # there might be an '=' in the file name
            # remove end quotes "..."
            if cat[-1] == '"':
                cat = cat[:-1]
            if cat[0] != '"':
                cat = cat[1:]
            kwargs['ignorecat'].append(cat)     
    return kwargs

def check_mapfiles(_pattern):
    '''
    Searches for files matching pattern
    returns a list of these files that seem to have gone wrong somehow
    '''
    files = glob.glob(_pattern)
    print('Found {num} files; checking...'.format(num=len(files)))
    fails = {cat: [] for cat in errcats}
    
    for filen in files:
        # explicit checks for values, presence of 'map'
        # otherwise, try-except checks
        try:
            with h5py.File(filen, 'r') as _f:
                if 'map' not in _f.keys():
                    print('No map stored in file {}'.format(filen))
                    fails['map missing'].append(filen)
                    continue
                shape = _f['map'].shape
                inshape = (_f['Header/inputpars'].attrs['npix_x'], _f['Header/inputpars'].attrs['npix_y'])
                if shape != inshape:
                    print('In file {}, input shape {} did not match stored map shape {}'.format(filen, inshape, shape))
                    fails['map shape'].append(filen)
                    continue
                _min = _f['map'].attrs['minfinite']
                _max = _f['map'].attrs['max']
                if _min == np.inf or _max == -np.inf or _max == np.NaN:
                    print('In file {}, NaNs were present or no values were finite: max was {}, min finite was {}'.format(filen, _max, _min))
                    fails['values'].append(filen)
                
        except OSError as err:
            print('Failed to open file {}'.format(filen))
            print('OSError:' + str(err))
            fails['h5py open'].append(filen)
        except IOError as err:
            print('Failed to open file {}'.format(filen))
            print('IOError:' + str(err))
            fails['h5py open'].append(filen)
        except KeyError as err:
            print('Attributes missing in file {}'.format(filen))
            print('KeyError:' + str(err))
            fails['attributes'].append(filen)
            
    for cat in errcats:
        print('{:4} files failed on {}'.format(len(fails[cat]), cat)) 
    return fails

def remove_mapfiles(faillist, errcats='all'):
    if errcats == 'all':
        errcats = faillist.keys()
    for cat in errcats:
        for filen in faillist[cat]:
            os.command('rm {}'.format(filen))

if __name__ == '__main__':
    kwargs = parse_commandline()
    fails = check_mapfiles(kwargs['pattern'])
    if kwargs['delete']:
        print('Deleting...')
        if 'ignorecat' in kwargs:
            delcats = list(np.copy(errcats))
            for cat in kwargs['ignorecat']:
                delcats.remove(cat)
        remove_mapfiles(fails, errcats=delcats)
        print('done')
        