#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# first exploration of FIRE data files, from a quick look at one or two of 
# them probably will be replaced by pre-written code by others at some point,
# or based on actually reading the docs

###### Notes/Todo ######
# - find out where to get omega_baryon
# - find out box size units

# quest location FIRE data: /projects/b1026/snapshots
# typical tests: /projects/b1026/snapshots/metal_diffusion/m12i_res57000
# but that seems to be empty now
# try one of the m12s for these tests

import h5py
import os
import numpy as np

import units_fire as uf
import make_maps_opts_locs as ol

# can add cases for python 2/3
def isstr(object): # should be python 2/3 robust
    return isinstance(object, type(''))
def isbstr(object):
    return isinstance(object, bytes) 

# setup from the internets
class FieldNotFoundError(Exception):
    def __init__(self, *args):
        if len(args) > 0:
            self.message = args[0]
        else:
            self.message = None
    def __str__(self):
        if self.message is not None:
            return 'FieldNotFoundError: {0} '.format(self.message)
        else:
            return 'FieldNotFoundError'

atomnumber_to_name = {1:  'Hydrogen',
                      2:  'Helium',
                      3:  'Lithium',
                      4:  'Beryllium',
                      5:  'Boron',
                      6:  'Carbon',
                      7:  'Nitrogen',
                      8:  'Oxygen',
                      9:  'Fluorine',
                      10: 'Neon',
                      11: 'Sodium',
                      12: 'Magnesium',
                      13: 'Aluminum',
                      14: 'Silicon',
                      15: 'Phosphorus',
                      16: 'Sulfur',
                      17: 'Chlorine',
                      18: 'Argon',
                      19: 'Potassium',
                      20: 'Calcium',
                      26: 'Iron',
                      }
name_to_atomnumber = {atomnumber_to_name[key]: key \
                      for key in atomnumber_to_name}
standard_atomno_indices = [0, 2, 6, 7, 8, 10, 12, 14, 16, 20, 26]     
gamma_gas = 5. / 3.                 
                      

class Cosmopars:
    def __init__(self, dct):
        self.a = dct['a']
        self.z = dct['z']
        self.h = dct['h']
        self.omegam = dct['omegam']
        self.omegalambda = dct['omegalambda']      
        self.omegab = dct['omegab']
        self.boxsize = dct['boxsize']
    def getdct(self):
        dct = {}
        dct['a'] = self.a
        dct['z'] = self.z 
        dct['h'] = self.h
        dct['omegam'] = self.omegam
        dct['omegalambda'] = self.omegalambda 
        dct['omegab'] = self.omegab 
        dct['boxsize'] = self.boxsize
        return dct
        
class Firesnap:
    def __init__(self, basename, parameterfile=None):
        '''
        Parameters:
        -----------
        basename: str
            name of the snapshot file (including the full directory path). 
            For multiple/split-output files, excluding the '.##.hdf5' end of 
            the file is ok.
        parameterfile: str
            name of the parameter file (including the full directory path) used
            for the simulation we're using the snapshot of.
        Returns:
        --------
        Firesnap object, for reading in datasets and attributes from FIRE snapshots 
        '''
        
        done = False
        if os.path.isfile(basename):
            parts = basename.split('.')
            # see if it's a snapshot_###.##.hdf5 name or snapshot_###.hdf5
            try: 
               int(parts[-2])
               basename = '.'.join(parts[:-2])
            except ValueError:
               self.firstfilen = basename
               self.filens = [self.firstfilen]
               self.numfiles = 1
               done = True
        if not done:
            if basename.endswith('.'):
                basename = basename[:-1]
            self.firstfilen = basename + '.0.hdf5'
            if not os.path.isfile(self.firstfilen):
                msg = 'Based on the name {bf}, the expected file {ff}' + \
                      ' was not found'
                raise ValueError(msg.format(bf=basename, 
                                 ff=self.firstfilen))
            with h5py.File(self.firstfilen, 'r') as f:
                self.numfiles = f['Header'].attrs['NumFilesPerSnapshot']
            self.filens = [basename + '.{num}.hdf5'.format(num=i) \
                           for i in range(self.numfiles)]
        
        self.parfilen = parameterfile
        if self.parfilen is not None:
            self.units = uf.Units(self.firstfilen, self.parfilen)
        else:
            self.units = uf.Units(self.firstfilen)
            
        # for quick attributes access
        self.ff = h5py.File(self.firstfilen, 'r')   
        self.find_cosmopars()
        
    def readattr(self, path, attribute):
        '''
        read in an attribute from the snapshot (first file)
        '''
        val = self.ff[path].attrs[attribute]  
        if isbstr(val):
            val = val.decode()
        return val
    
    def find_cosmopars(self):
        cdct = {}
        cdct['z'] = self.ff['Header'].attrs['Redshift'] 
        cdct['a'] = self.ff['Header'].attrs['Time']
        a_z = 1. / (1. + cdct['z'])
        if not np.isclose(cdct['a'], a_z):
            print('Warning: Time value inconsistent with expansion factor a:')
            print('Time: {t}, 1 / (1 + z): {a}'.format(t=cdct['a'], a=a_z))
            print('Using 1 / (1 + z)')
            cdct['a'] = a_z
        del a_z
        
        if self.parfilen is None:
            # FIRE-2 and FIRE-3 use different keys for the Omegas
            tosearch = self.ff['Header'].attrs
            # Omega_matter
            if 'Omega0' in tosearch:
                cdct['omegam'] = tosearch['Omega0']
            elif 'Omega_Matter' in tosearch:
                cdct['omegam'] = tosearch['Omega_Matter']
            else:
                raise KeyError('Could not find a value for omegam')
            # Omega_Lambda
            if 'OmegaLambda' in tosearch:
                cdct['omegalambda'] = tosearch['OmegaLambda']
            elif 'Omega_Lambda' in tosearch:
                cdct['omegalambda'] = tosearch['Omega_Lambda']
            else:
                raise KeyError('Could not find a value for omegalambda')
            # Omega_baryon
            if 'OmegaBaryon' in tosearch:
                cdct['omegab'] = tosearch['OmegaBaryon']
            elif 'Omega_Baryon' in tosearch:
                cdct['omegab'] = tosearch['Omega_Baryon']
            else:
                print('Warning: did not find a value for omegab, using NaN')
                cdct['omegab'] = np.NaN
            # other
            cdct['h'] = self.ff['Header'].attrs['HubbleParam'] 
            cdct['boxsize'] = self.ff['Header'].attrs['BoxSize'] 
        else:
            pardict = self._cosmopars_from_parameterfile()
            cdct.update(pardict)
        # for compatibility with EAGLE: cMpc/h
        cdct['boxsize'] *= self.units.codelength_cm / uf.c.cm_per_mpc \
                           / self.units.a * self.units.HubbleParam
        self.cosmopars = Cosmopars(cdct)
    
    def _cosmopars_from_parameterfile(self):
        partdict = {}
        targetnum = 5
        with open(self.parfilen, 'r') as _f:
            for line in _f:
                if line.startswith('Omega0'):
                    partdict['omegam'] = float(line.split()[1])
                elif line.startswith('OmegaLambda'):
                    partdict['omegalambda'] = float(line.split()[1])
                elif line.startswith('OmegaBaryon'):
                    partdict['omegab'] = float(line.split()[1])  
                elif line.startswith('HubbleParam'):
                    partdict['h'] = float(line.split()[1])
                elif line.startswith('BoxSize'):
                    partdict['boxsize'] = float(line.split()[1])
                if len(partdict) == targetnum:
                    break
        if not len(partdict) == targetnum:
            msg = 'Could not find all time-independent cosmopars in the' + \
                  ' parameterfile'
            raise RuntimeError(msg)
        return partdict
    
    # read-in and subsampling tested
    def readarray(self, path, subsample=1, errorflag=np.nan, subindex=None):
        '''
        read in an array from the snapshot file
        note that subsample read-ins are slow
        sets a .toCGS attribute: multiply array by this to get (physical)
        cgs units.
        
        Parameters:
        -----------
        path: str
            the full path to the dataset in the hdf5 file            
        subsample: int
            one in <subsample> values is read in, by simply taking values at
            this interval
        errorflag: array type
            set initial values for the array, before copying stored values
            ignored if all data is in a single file
        subindex: int or None
            specific index to read in if the array is 2D instead of 1D 
            (e.g., only read in the X coordinates -> subindex=0)

        Returns:
        --------
        the desired array 

        Errors:
        -------
        FieldNotFoundError: the desired field was not present in (any) file
        '''
        self.toCGS = np.NaN # overwrite and old values to avoid undetected errors
        # just let h5py handle it
            
        if self.numfiles == 1:
            if path not in self.ff:
                raise FieldNotFoundError
            sel = slice(None, None, subsample)
            if subindex is not None:
                sel = (sel, subindex)
            arr = self.ff[path][sel]
        
        # needs a bit more planning to mimic the slice object result
        # (don't want to introduce dependencies on how files are split)
        else:
            parttypeindex = int(path.split('/')[0][-1]) 
            numpart = self.ff['Header'].attrs['NumPart_Total'][parttypeindex] 
            arrsize = (numpart - 1) // subsample + 1
            if numpart == 0:
                msg = 'No particles of type {} present in files {}'
                raise IOError(msg.format(parttypeindex, self.filens))
            # lowest non-overflow value for int data, 
            # 'nan', 'na', or 'n' for string/bytes data (depends on max. 
            # string length)
            
            start = 0
            combindex = 0
            array_init = False
            for filen in self.filens:
                print(filen)
                with h5py.File(filen, 'r') as f:
                    npt = 'NumPart_ThisFile'
                    sublen_tot = f['Header'].attrs[npt][parttypeindex]
                    if sublen_tot == 0:
                        continue
                    if not array_init:
                        dtype = f[path].dtype
                        if subindex is None:
                            shape = (arrsize,) + f[path].shape[1:]
                        else:
                            shape = (arrsize,)
                        # empty means values will not stand out if 'filled in' wrong
                        arr = np.empty(shape=shape, dtype=dtype)
                        arr[:] = errorflag 
                        print('Array shape: ', arr.shape)
                        array_init = True
                    ds = f[path]
                    print('size on file: ', sublen_tot)
                    # in combined array indices: 
                    # next multiple of subsample - first element in subarray
                    subsel_offset = (-1 * start) % subsample 
                    subsel = slice(subsel_offset, None, subsample)
                    if subindex is not None:
                        subsel = (subsel, subindex)
                    
                    numsel = (sublen_tot - 1 - subsel_offset) // subsample + 1
                    totsel = slice(combindex, combindex + numsel, None)
                    print('selected: ', numsel)
                    
                    ## test on quest, snapshot 600 (z=0)
                    # /projects/b1026/snapshots/metal_diffusion/m12i_res7100
                    # /output
                    # timeit for readin (PartType4/Masses, subsample=5)
                    # using only 10 trials to keep the time manageable
                    # ds.read_direct(arr, source_sel=subsel, dest_sel=totsel):
                    # -> 888.8572574020363
                    # arr[totsel] = ds[subsel]:
                    # -> 961.0126019851305
                    
                    ds.read_direct(arr, source_sel=subsel, dest_sel=totsel)
                    #arr[totsel] = ds[subsel]
                    
                    start += sublen_tot
                    combindex += numsel
            if not array_init:
                # evidently, the field wasn't in any file
                raise FieldNotFoundError(('Field {} not found'.format(path)))
        self.toCGS = self.units.getunits(path)
        return arr
    
    def readarray_emulateEAGLE(self, field, subsample=1, errorflag=np.nan):
        '''
        Read in an array and set to toCGS attribute. Includes conversions from 
        field names in EAGLE and calculation of gas temperature

        note, 'PartType0/Pressure' returns only the /Thermal/ pressure
        '''
        self.toCGS = np.NaN # overwrite and old values to avoid undetected errors
        # Metals: field names match, but structure is different
        if 'Metallicity' in field:
            if 'Smoothed' in field:
                msg = 'Warning: smoothed abundances are unavailable in FIRE' +\
                      '; using non-smoothed values'
                print(msg)
                field = field.replace('Smoothed', '')
            if 'name_to_atomnumber' in self.ff['Header'].attrs:
                atomnos = self.ff['Header'].attrs['name_to_atomnumber']
                index = np.where([atomno == 0 for atomno in atomnos])[0][0]
            else:
                # assume “standard” runs with METAL_SPECIES_COOLING enabled
                index = np.where([atomno == 0 for atomno \
                                  in standard_atomno_indices])[0][0]
            out = self.readarray(field, subsample=subsample, 
                                 errorflag=errorflag, subindex=index)
            self.toCGS = self.units.getunits(field)
            return out
            
        elif 'ElementAbundance' in field:
            if 'Smoothed' in field:
                msg = 'Warning: smoothed abundances are unavailable in FIRE' +\
                      '; using non-smoothed values'
                print(msg)
                field = field.replace('Smoothed', '')
            element = field.split('/')[-1]
            if element == 'Hydrogen':
                _fieldstart = '/'.join(field.split('/')[:-1])
                he = self.readarray_emulateEAGLE(_fieldstart + '/Helium', 
                                                 subsample=1, errorflag=np.nan)
                _fieldstart = '/'.join(field.split('/')[:-2])
                me = self.readarray_emulateEAGLE(_fieldstart + '/Metallicity', 
                                                 subsample=1, errorflag=np.nan)
                hfrac = 1. - he - me
                del he
                del me
                return hfrac
                
            if 'name_to_atomnumber' in self.ff['Header'].attrs:
                atomnos = self.ff['Header'].attrs['name_to_atomnumber']
                atomno = name_to_atomnumber[element]
                index = np.where([_atomno == atomno for _atomno \
                                  in atomnos])[0][0]
            else:
                atomnos = standard_atomno_indices
                atomno = name_to_atomnumber[element]
                index = np.where([_atomno == atomno for _atomno \
                                  in standard_atomno_indices])[0][0]
            parttypestr = field.split('/')[0]
            field = parttypestr + '/Metallicity'
            out = self.readarray(field, subsample=subsample, 
                                 errorflag=errorflag, subindex=index)
            self.toCGS = self.units.getunits(field)
            return out
        # lots of fields are just the same
        else:
            try: 
                self.toCGS = self.units.getunits(field)
                return self.readarray(field, subsample=subsample, 
                                      errorflag=errorflag)   
            except (FieldNotFoundError, uf.UnitsNotFoundError) as err:
                # same stuff, different name
                if field.endswith('Mass'): # Mass in EAGLE = Masses in FIRE
                    _field = field + 'es'
                    self.toCGS = self.units.getunits(_field)
                    return self.readarray(_field, subsample=subsample, 
                                          errorflag=errorflag)
                # temperature: need to calculate instead of read in
                elif field == 'PartType0/Temperature':
                    hekey = 'PartType0/ElementAbundance/Helium'
                    hefrac = self.readarray_emulateEAGLE(hekey, 
                                                         subsample=subsample,
                                                         errorflag=errorflag)
                    etoh = self.readarray('PartType0/ElectronAbundance',
                                           subsample=subsample, 
                                           errorflag=errorflag)  
                    print(etoh)
                    print(hefrac)                      
                    yhe = hefrac / (4. * (1. - hefrac))             
                    mu = (1. + 4. * yhe) / ( 1. + yhe + etoh)
                    print(mu)
                    del etoh
                    del hefrac
                    mean_molecular_weight = mu * uf.c.atomw_H * uf.c.u
                    temperature = self.readarray('PartType0/InternalEnergy',
                                                 subsample=subsample, 
                                                 errorflag=errorflag)
                    uconv = self.units.getunits('PartType0/InternalEnergy')
                    scalar = uconv * (gamma_gas - 1.) / uf.c.boltzmann 
                    print(scalar)
                    print(mean_molecular_weight)
                    temperature *= scalar * mean_molecular_weight 
                    self.toCGS = 1. 
                    # do the conversion: matches expected units from EAGLE
                    # and an extra scalar multiplication doesn't cost much
                    return temperature
                elif field == 'PartType0/Pressure':
                    # !! Thermal Pressure only !!
                    # P = n * k_B * T
                    # n = Density / mu (density = mu * n_part by definition)
                    # T = mu * (gamma-1) * InternalEnergy / k_B
                    # so P = Density / mu * k_B * mu * (gamma - 1) * InternalEnergy / k_B
                    # so P = Density * (gamma - 1) * InternalEnergy
                    pressure = self.readarray('PartType0/InternalEnergy',
                                                 subsample=subsample, 
                                                 errorflag=errorflag)
                    # internal energy per unit mass
                    _toCGS = self.units.getunits('PartType0/InternalEnergy')
                    _toCGS *= (gamma_gas - 1.)
                    pressure *= self.readarray('PartType0/Density',
                                                 subsample=subsample, 
                                                 errorflag=errorflag)
                    _toCGS *= self.units.getunits('PartType0/Density')
                    self.toCGS = _toCGS
                    return pressure
                else:
                    raise err
                  
def get_Firesnap(path, snapnum, filetype='snap'):
    '''
    return a FireSnap object, with the parameterfile and snapshot file 
    in the given path.
    
    The path should contain the parameter file, and the snapshots directly
    or a subdirectory called 'output' containing the snapshot file(s). This
    function tries different file names and subdirectories in order.
    
    Parameters:
    -----------
    path: str
        the path containing the parameter file and snapshot(s)
        the path may be relative to make_maps_opts_locs.simdir_fire 
    snapnum: int
        the snapshot number
    filetype: str, optional
        stub; for now this only works for snapshots, but it can be
        adapted to read return a halo file reader.
        The default is 'snap'.
    
    ''' 
    opts_filetype = ['snap']
    if filetype not in opts_filetype:
        raise ValueError('filetype should be one of {}'.format(opts_filetype))
    prefix = ol.simdir_fire
    if not os.path.isdir(path):
        if not os.path.isdir(prefix + path):
            msg = 'Could not find a directory {} or {}'
            raise ValueError(msg.format(path, prefix + path))
        else:
            path = prefix + path
    if path[-1] != '/':
        path = path + '/'
    
    opts_pardir = ['', 'output/']
    opts_parfile = ['params.txt-usedvalues',
                    'parameters-usedvalues',  
                    'params.txt']
    parameterfile = None
    for subdir in opts_pardir:
        for opt in opts_parfile:
            if os.path.isfile(path + subdir + opt):
                parameterfile = path + subdir + opt
                break
    if parameterfile is None:
        msg = 'Could not find a parameter file {} in {}'
        dirs = [path + _d for _d in opts_pardir]
        print(msg.format(opts_parfile, dirs))
        print('Using parameterfile=None for FireSnap')
    
    opts_snapdir = ['', 'output/']
    # parts with the snapshot number in there
    opts_snapfile = ['snapshot_{snap:03d}.hdf5',
                     'snapshot_{snap:03d}.0.hdf5',
                     'snapdir_{snap:03d}/snapshot_{snap:03d}.hdf5',
                     'snapdir_{snap:03d}/snapshot_{snap:03d}.0.hdf5',
                     ]
    opts_snapfile = [filen.format(snap=snapnum) for filen in opts_snapfile]
    
    basename = ''
    for subdir in opts_snapdir:
        for snapfile in opts_snapfile:
            opt = path + subdir + snapfile
            if os.path.isfile(opt):
                basename = opt
                break
    if basename == '':
        msg = 'Could not find a snapshot file {} in {}'
        dirs = [path + _d for _d in opts_snapdir]
        raise RuntimeError(msg.format(opts_snapfile, dirs))
    
    firesnap = Firesnap(basename, parameterfile=parameterfile)
    msg = 'Using parameterfile {}, (1st) snapshot {}'
    print(msg.format(parameterfile, basename))
    return firesnap
     
def findclosestz_snap(path, redshift):
    '''
    Utility function for picking snapshots.

    Parameters:
    -----------
    path: str
        where to look for the snapshot list value. Same options as
        the get_Firesnap path.
    redshift: float
        which redshift value to try to match
    
    Returns:
    --------
    snapnum: int
        the number of the closest matching snapshot
    zval: float
        the redshift of the closest matching snapshot
    '''
    if path.endswith('output'):
        path = path[:-6]
    if not path.endswith('/'):
        path = path + '/'
    targetfile = path + 'snapshot_scale-factors.txt'
    if not os.path.isfile(targetfile):
        raise RuntimeError('No file {} found'.format(targetfile))
    with open(targetfile, 'r') as f:
        aopts = f.read()
    aopts = (aopts.strip()).split('\n')
    aopts = np.array([float(aopt) for aopt in aopts])
    zopts = 1. / aopts - 1.
    snapnum = np.argmin(np.abs(zopts - redshift))
    zval = zopts[snapnum]
    return snapnum, zval
    
        