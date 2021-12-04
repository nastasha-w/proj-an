#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:03:53 2019

@author: wijers
"""

import numpy as np
import h5py

import make_maps_opts_locs as ol
import eagle_constants_and_units as c

# simulation read-in files might not always be present. There will be errors
# down the line if this fails despite being needed, though
try:
    import read_eagle_files as eag
except ImportError:
    print('Warning: failed to import read_eagle_files')
try:
    import read_eagle_files_noneq_noregion as refn #modified read_bahamas_files
except ImportError:
    print('Warning: failed to import read_eagle_files_noneq_noregion')
try:
    import read_bahamas_files as bhm #also has OWLS option
except ImportError:
    print('Warning: failed to import read_bahamas_files')
try:
    import read_eagle_files_ceh_noregion as rce
except ImportError:
    print('Warning: failed to import read_eagle_files_ceh_noregion')
    
#################
#    classes    #
#################

class Simfile:
    '''
    Presently, a thin wrapper for read_eagle_files.readfile
    intended to possibly expand read-in to e.g. C-EAGLE/Hydrangea, OWLS, BAHAMAS:
    contain more or less the same data (so that no rewrite of the calculations is needed),
    but have different read-in options (regions or just selections) and libraries

    in case of C-EAGLE/Hydrangea: simfile is the halo number (int), var is ignored

    anything beyond raw data passing is handled by Vardict.
    if necessary, e.g. tables or separate files can be used to set e.g., a, h, scalings
    for non-EAGLE outputs
    if region handling is not included, Vardict and the selections should be given a case
    or option to handle that, rather than using consecutive selections on each array
    
    DM particle mass is only read in in Eagle for now, in case the units are 
    different in other gagdet variants
    '''
    def readarray_eagle(self, name, region=None, rawunits=False):
        arr = self.readfile.read_data_array(name, gadgetunits=rawunits, suppress=False,region=region)
        self.a_scaling = self.readfile.a_scaling
        self.h_scaling = self.readfile.h_scaling
        self.CGSconversion = self.readfile.CGSconversion
        self.CGSconvtot = self.a**self.a_scaling * self.h**self.h_scaling * self.CGSconversion
        return arr
    def readarray_bahamas(self, name, region=None, rawunits=False): #region is useless here
        if region is not None:
            print('Warning (readarray_bahamas): region selection will not have any effect')
        arr = self.readfile.read_var(name, gadgetunits=rawunits, verbose=True)
        # CGS conversion should be safe to just take from the first file
        self.CGSconvtot = self.readfile.convert_cgs(name, 0, verbose=True)
        self.a_scaling = self.readfile.a_scaling
        self.h_scaling = self.readfile.h_scaling
        self.CGSconversion = self.readfile.CGSconversion
        return arr

    def __init__(self,simnum, snapnum, var, file_type=ol.file_type, simulation='eagle', override_filepath=None):
        self.simnum = simnum
        self.snapnum = snapnum
        self.var = var
        self.simulation = simulation
        self.filetype = file_type
        
        if (override_filepath is not None) and (simulation != 'eagle'):
            raise NotImplementedError('The filepath override option is only implemented for Eagle.\
                                      To implement for other simulations, check the read_..._files py used for that simulation.\
                                      It may be neceassary to add an option to not add "/data/" to the file path, like in read_eagle_files.py.\
                                      Then, choose this option and set the file path for the simulation depending on these options.\
                                      ')
        if simulation == 'eagle':
            if override_filepath is None:
                simdir  = ol.simdir_eagle%simnum + '/' + var
                add_data_to_model_dir = True
            else:
                simdir = override_filepath
                add_data_to_model_dir = False
            self.readfile = eag.read_eagle_file(simdir, file_type, snapnum, gadgetunits=True, suppress=False, add_data_to_model_dir=add_data_to_model_dir)
            # pass down readfile properties for reference
            self.boxsize = self.readfile.boxsize
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1. / self.a - 1. # rounded off in read_eagle_files for some reason...
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.fname
            try:
                self.hdf5file = h5py.File( self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File( self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.particlemass_DM_g = np.array(self.hdf5file['Header'].attrs['MassTable'])[1] * \
                                      self.hdf5file['Units'].attrs['UnitMass_in_g'] * \
                                      self.h**-1 # checked units with Matthieu 
            self.hdf5file.close()

            self.region_supported = True
            self.readarray = self.readarray_eagle

        elif simulation == 'eagle-ioneq':
            simdir  = ol.simdir_eagle_noneq
            if var == 'REFERENCE':
                ioneq_str = 'ioneq_'
            elif var == 'ssh':
                ioneq_str = 'ioneq_SS_'
            else:
                raise ValueError('var option %s is not valid for eagle-ioneq'%var)
            print('EAGLE ioneq simulation: %s, %s, %s, %s'%(simdir, file_type, snapnum, ioneq_str))
            self.readfile = refn.Gadget(simdir, file_type, snapnum,\
                                        gadgetunits=True, suppress=False,\
                                        sim='EAGLE-IONEQ', ioneq_str=ioneq_str,\
                                        add_data_dir = False)
            # pass down readfile properties for reference
            self.boxsize = self.readfile.boxsize
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1. / self.a - 1.
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.filename
            try:
                self.hdf5file = h5py.File( self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File( self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.hdf5file.close()

            self.region_supported = False
            self.readarray = self.readarray_bahamas # modified read_bahamas_files has same methods

        elif simulation == 'bahamas':
            simdir = ol.simdir_bahamas%simnum
            self.readfile = bhm.Gadget(simdir, file_type, snapnum,\
                                       gadgetunits=True, verbose=True,\
                                       sim='BAHAMAS')
            self.boxsize = self.readfile.boxsize
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1. / self.a - 1.
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.filename
            try:
                self.hdf5file = h5py.File( self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File( self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.hdf5file.close()

            self.region_supported = False
            self.readarray = self.readarray_bahamas
        elif simulation == 'c-eagle-hydrangea':
            # here, var is interpreted as the halo number (formatting taken care of, just input an integer)
            simdir = ol.simdir_ceaglehydrangea%(simnum)
            self.readfile = rce.Gadget(simdir, file_type, snapnum,\
                                       sim='C-EAGLE', gadgetunits=True,\
                                       suppress=False)
            # pass down readfile properties for reference
            self.boxsize = self.readfile.boxsize
            self.h = self.readfile.h
            self.a = self.readfile.a
            self.z = 1. / self.a - 1.
            # omegam and omegalambda are not retrieved by read_eagle, but are needed to get the hubble parameter H(z)
            # try cases are extracted from read_eagle files and read_bahamas_files
            self.filenamebase = self.readfile.filename
            try:
                self.hdf5file = h5py.File(self.filenamebase+"0.hdf5", 'r' )
            except:
                self.hdf5file = h5py.File(self.filenamebase+"hdf5", 'r' )
            self.omegam = self.hdf5file['Header'].attrs['Omega0']
            self.omegalambda = self.hdf5file['Header'].attrs['OmegaLambda']
            self.omegab = self.hdf5file['Header'].attrs['OmegaBaryon']
            self.hdf5file.close()
            if simnum in ol.halos_ceagle:
                self.region_supported = False
            elif simnum in ol.halos_hydrangea:
                self.region_supported = False # for now
            self.readarray = self.readarray_bahamas

        else:
            raise ValueError('Simulation %s is not supported.'%simulation)
    
    def get_cosmopars(self):
        return {'a': self.a,\
                'z': self.z,\
                'h': self.h,\
                'boxsize': self.boxsize,\
                'omegalambda': self.omegalambda,\
                'omegab': self.omegab,\
                'omegam': self.omegam,\
                }
class Sel:
    '''
    an array mask for 1D arrays (selections):
    default value is slice(None,None,None), can be set as a boolean array
    can be combined with other 1D masks through logical_and

    just here as a conveniet way to initalise particle selection for unknown
    array sizes

    key: if input is a dict containing the array, key to use to retrieve it
    note: if a sel instance is input, no deep copy is made - its arrays can be
         modified in-function
    '''
    # asarray uses copy=False
    def __init__(self,arr=None,key='arr'):
        if isinstance(arr,np.ndarray):
            self.seldef = True
            self.val = np.asarray(arr,dtype=bool)
        elif isinstance(arr,dict):
            if key in arr.keys():
                self.seldef = True
                self.val = np.asarray(arr[key],dtype=bool)
            elif len(arr.keys()) ==1: # must the the one element, right?
                key_old = key
                key = list(arr.keys())[0]
                if key_old is not None:
                    print('Warning: invalid key %s for dictionary input to Sel.__init__; using only key %s in stead.'%(str(key_old), str(key)))
                self.seldef = True
                self.val = np.asarray(arr[key],dtype=bool)
            else:
                print('Error: invalid key %s for dictionary input to Sel.__init__')
        else: # default setting
            self.seldef = False
            self.val = slice(None,None,None)
    def __str__(self):
        return 'Sel(%s, seldef=%s)'%(str(self.val),str(self.seldef))

    def comb(self,arr,key=None): # for combining two selections of the same length into the first

        if isinstance(arr,Sel):
            if arr.seldef and self.seldef: # if arr is a non-default Sel instance, just use the array value
                self.val &= arr.val
            elif not self.seldef:
                self.val = arr.val
                self.seldef = arr.seldef
            else: #arr.seldef, self.seldef == False
                pass # Sel instance is unaltered

        elif isinstance(arr,np.ndarray): # if the instance in non-default, combine with any arr value, otherwise replace the instance
            if self.seldef:
                self.val &= np.asarray(arr,dtype=bool)   #self.val  = np.logical_and(self.val,arr)
            else:
                self.val = np.asarray(arr,dtype=bool)
            self.seldef = True

        elif isinstance(arr,dict):
            if key in arr.keys():
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val &= np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel): # might as well allow a Sel from a dictionary; contained array is not copied on call, so recursion won't waste memory
                    self.comb(arr[key])
                else:
                    print('Error (Sel.comb): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            elif len(arr.keys()) ==1: # must the the one element, right?
                key_old = key
                key = list(arr.keys())[0]
                if key_old is not None:
                    print('Warning (Sel.comb): invalid key %s for dictionary input to Sel.comb; using only key %s in stead.'%(str(key_old), str(key)))
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val &= np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel):
                    self.comb(arr[key])
                else:
                    print('Error (Sel.comb): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            else:
                print('Error (Sel.comb): invalid key %s for dictionary input to Sel.comb; Sel instance unmodified'%str(key))

        else:
            print('Error (Sel.comb): arr must be a Sel instance, compatible array, or dict containing such an array')


    def refine(self,arr,key=None): # comb for when the second array is a selection of the first's True indices
    #        if isinstance(arr,Sel) and arr.seldef: # if arr is a non-default Sel instance, just use the array value; non-initalised Sel does nothing
    #            self.refine(arr.val)
    #        elif not isinstance(arr,Sel): # if the instance is non-default, combine with any arr value, otherwise replace the instance
    #            if self.seldef:
    #                self.val[self.val] = arr
    #            else:
    #               self.val = arr
    #            self.seldef = True

        if isinstance(arr, Sel):
            if arr.seldef and self.seldef: # if arr is a non-default Sel instance, just use the array value
                self.val[self.val] = arr.val
            elif not self.seldef:
                self.val = arr.val
                self.seldef = arr.seldef
            else: #arr.seldef, self.seldef == False
                pass # Sel instance is unaltered

        elif isinstance(arr,np.ndarray): # if the instance in non-default, combine with any arr value, otherwise replace the instance
            if self.seldef:
                self.val[self.val] = np.asarray(arr,dtype=bool)   #self.val  = np.logical_and(self.val,arr)
            else:
                self.val = np.asarray(arr,dtype=bool)
            self.seldef = True

        elif isinstance(arr,dict):
            if key in arr.keys():
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val[self.val] = np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel): # might as well allow a Sel from a dictionary; contained array is not copied on call, so recursion won't waste memory
                    self.refine(arr[key])
                else:
                    print('Error (Sel.refine): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            elif len(arr.keys()) ==1: # must the the one element, right?
                key_old = key
                key = list(arr.keys())[0]
                if key_old is not None:
                    print('Warning (Sel.refine): invalid key %s for dictionary input to Sel.comb; using only key %s in stead.'%(str(key_old), str(key)))
                if isinstance(arr[key],np.ndarray):
                    if not self.seldef:
                        self.val = np.asarray(arr[key],dtype=bool)
                        self.seldef = True
                    else:
                        self.val[self.val] = np.asarray(arr[key],dtype=bool)
                elif isinstance(arr[key],Sel): # might as well allow a Sel from a dictionary; contained array is not copied on call, so recursion won't waste memory
                    self.refine(arr[key])
                else:
                    print('Error (Sel.refine): can not combine Sel instance with object %s; Sel instance unmodified'%type(arr[key]))
            else:
                print('Error (Sel.refine): invalid key %s for dictionary input to Sel.refine; Sel instance unmodified'%str(key))

        else:
            print('Error (Sel.comb): arr must be a Sel instance, compatible array, or dict containing such an array')

    def reset(self):
        self.seldef = False
        self.val = slice(None,None,None)



class Vardict:
    '''
    Stores read-in and derived variables in a dictionary, and keeps track of
        what will still be needed in the next calculation
    In the next calculation, then tracks what still needs to be read in, and
        what is still stored
    wishlist: variables desired after the calculation (depends on both W and Q,
        so construction is done separately); list
    parttype: PartType as in the EAGLE output (see project docstring); string
    simfile: Simfile instance - tells the function where to find its
        data
    sel:      selection relative to total stored array in region: used on read-ins
    '''

    def __init__(self, simfile, parttype, wishlist, region=None, readsel=None):
        self.simfile = simfile
        self.parttype = parttype
        self.wishlist = wishlist
        self.particle = {} # name:var dictionary of particle attributes
        self.box = {}      # name:var dictionary of box attributes
        self.CGSconv = {}  #name:units dictionary (stores factor by which to multiply corresponding array to get CGS)
        # convenient to store as default
        self.region = region
        if readsel is None:
            self.readsel = Sel()
        elif isinstance(readsel,Sel):
            self.readsel = readsel
        else:
            self.readsel = Sel(readsel)
        #self.reportmemuse()

    def delif(self, name, last=False):
        '''
        deletes a variable if it is not on a list of variables to save,
        and removes corresponding dictionary entry, always if it is the last use
        only works on particle quantities;
        box quantities are not worth the trouble of deleting
        '''
        #self.reportmemuse()
        if name not in self.wishlist or last:
            del self.particle[name] # remove dictionary element

    def isstored_box(self, name):
        return name in self.box.keys()

    def isstored_part(self, name):
        return name in self.particle.keys()

    def readif(self, name, region='auto', rawunits=False, out=False, setsel=None, setval=None):  # reads in name (of parttype) from EAGLE if it is not already read in (on lst),
        '''
        reads in data that is not already read in, using a dictionary to trak read-in particles
        name:     name of EAGLE output to read in (not including 'PartType/') (str)
        parttype: parttype of array to read in (str)
        region, gadgetunits: as in read_data_array
        sel :     a Sel instance
        cnvfact:  save cgs conversion factor to this variable unless set to None
        out:      saves result to the variable if not None
        setsel and setval: sets values in mask setsel to setval if setval is not None
            !! cnvfact is not stored in Vardict
        '''
        #self.reportmemuse()
        if region == 'auto':
            self.region_temp = self.region
        else:
            self.region_temp = region
        sel = self.readsel
        if not self.isstored_part(name):
            if 'Coordinates' in name:
                self.particle[name] = self.simfile.readarray('PartType%s/%s' %(self.parttype, name), rawunits=True, region=self.region_temp).astype(np.float32)[sel.val, :]
            else:
                self.particle[name] = self.simfile.readarray('PartType%s/%s' %(self.parttype, name), rawunits=True, region=self.region_temp)[sel.val] #coordinates are always needed, so these will not be read in this way; other arrays are 1D
            self.CGSconv[name] = self.simfile.a ** self.simfile.a_scaling * (self.simfile.h ** self.simfile.h_scaling) * self.simfile.CGSconversion
            if not rawunits: # do CGS conversion here since read_eagle_files does not seem to modify the array in place
                self.particle[name] *= self.CGSconv[name]
                self.CGSconv[name] = 1.
        if setval != None:
            self.particle[name][setsel] = setval
        if out:
            return self.particle[name]
        del self.region_temp
        
    def add_part(self,name,var):
        #self.reportmemuse()
        if name in self.particle.keys():
            print('Warning: variable <%s> will be overwritten by Vardict.add_part' %name)
        self.particle[name] = var

    def add_box(self,name,var):
        #self.reportmemuse()
        if name in self.box.keys():
            print('Warning: variable <%s> will be overwritten by Vardict.add_box' %name)
        self.box[name] = var

    def update(self, selname, key=None):
        '''
        updates stored particle property arrays to only contain only the new
        selection elements, and updates the read-in selection
        '''
        #self.reportmemuse()
        if not isinstance(selname, Sel) and not isinstance(selname, dict):
            selname = Sel({'arr': selname}, 'arr') # if input is a numpy array, don't do a second copy for comb/refine calls
        if selname.seldef and self.readsel.seldef:
            if selname.val.shape == self.readsel.val.shape:
                self.tempsel = selname.val[self.readsel.val]
                for name in self.particle.keys():
                    self.particle[name] = (self.particle[name])[self.tempsel,...]
                del self.tempsel
                self.readsel.comb(selname)
                print('\nCalled Vardict.readsel.comb\n')
            #elif selname.val.shape[0] == self.readsel.val.shape[0]: # if keys()[0] happens to be coordinates or something, we just want to match the zero index
            #    for name in self.particle.keys():
            #        self.particle[name] = (self.particle[name])[selname.val,...]
            #    self.readsel.refine(selname)
            
            else: # assume selname is a selection of the readsel True entries
                for name in self.particle.keys():
                    self.particle[name] = (self.particle[name])[selname.val,...]
                self.readsel.refine(selname)
                print('\nCalled Vardict.readsel.refine\n')
                
        elif selname.seldef:
            for name in self.particle.keys():
                self.particle[name] = (self.particle[name])[selname.val,...]
            self.readsel.refine(selname)
            print('\nCalled Vardict.readsel.refine\n')
        else:
            pass # selname is undefined; no update necessary
            print('\nNo readsel update necessary\n')
            
    def overwrite_part(self, name, var):
        #self.reportmemuse()
        self.particle[name] = var

    def overwrite_box(self, name, var):
        #self.reportmemuse()
        self.box[name] = var

    ## functions to get specific derived particle properties; wishlist generation counts on naming being get<name of property to store>
    # note: each function 'cleans up' all other quantites it uses! adjust wishlists in lumonisity etc. calculation to save quantities needed later on
    def getlognH(self, last=True, **kwargs):
        #self.reportmemuse()
        if not 'hab' in kwargs.keys():
            print('hab must be specified to calculate lognH')
            return None
        self.readif('Density', rawunits=True)
        if type(kwargs['hab']) == str:
            self.readif(kwargs['hab'],rawunits = True)
            self.add_part('lognH', np.log10(self.particle[kwargs['hab']]) + np.log10(self.particle['Density']) + np.log10(self.CGSconv['Density'] / (c.atomw_H * c.u)) )
            self.delif(kwargs['hab'], last=last)
        else:
            self.add_part('lognH', np.log10(self.particle['Density']) + np.log10(self.CGSconv['Density'] * kwargs['hab'] / (c.atomw_H * c.u)) )
        self.delif('Density', last=last)
        self.CGSconv['lognH'] = 0.

    def getpropvol(self,last=True):
        #self.reportmemuse()
        self.readif('Density', rawunits=True)
        self.readif('Mass', rawunits=True)
        self.add_part('propvol', (self.particle['Mass'] / self.particle['Density']) * (self.CGSconv['Mass'] / self.CGSconv['Density']))
        self.delif('Mass',last=last)
        self.delif('Density',last=last)
        self.CGSconv['propvol'] = 1.

    def getlogT(self,last=True,**kwargs):
        #self.reportmemuse()
        if kwargs['T4']:
            self.readif('OnEquationOfState',rawunits=True)
            self.add_part('eos',self.particle['OnEquationOfState'] > 0.)
            self.delif('OnEquationOfState',last=last)
            self.readif('Temperature',rawunits=True,setsel = self.particle['eos'],setval = 1e4)
            self.delif('eos',last=last)
        else:
            self.readif('Temperature',rawunits=True)
        self.add_part('logT',np.log10(self.particle['Temperature']))
        self.delif('Temperature',last=last)
        self.CGSconv['logT'] = 0.

    def reportmemuse(self):
        '''
        return the memory used in Vardict by Vardict.particle and Vardict.sel
        note: this is an approximation, meant for debugging and optimisation
              help
        '''
        self.keys = self.particle.keys()
        self.sizes = [self.particle[key].nbytes / 1024.**2 for key in self.keys]
        if self.readsel.seldef:
            self.selsize = self.readsel.val.nbytes / 1024.**2
            self.sellen  = len(self.readsel.val)
        else:
            self.selsize = 0.
            self.sellen  = 0
        self.strbase = '%s:\t %.3e MB,\t %.e3 elements'
        self.out = '\n'.join([self.strbase%(key, self.particle[key].nbytes / 1024.**2, float(len(self.particle[key]))) for key in self.keys])
        self.out = self.out + '\n' + self.strbase%('readsel', self.selsize, float(self.sellen))
        print('\n-------------------------------')
        print('Approximate current memory use by Vardict instance:')
        print(self.out)
        print('total approximate vardict memory use: %.3e MB'%(float(sum(self.sizes) + self.selsize)))
        print('-------------------------------\n')

        del self.strbase
        del self.out
        del self.selsize
        del self.sellen
        del self.keys
        del self.sizes

#class Readfileclone: # for testing purposes: contains properties and grids mimicking read_eagle_files readfiles objects
#    def __init__(self, z=0., coords='auto', vel='auto', boxsize=10. / c.hubbleparam**-1, lsmooth = 1.1/c.hubbleparam**-1):
#        self.z = z
#        self.hub = Hubble(self.z)
#        self.h = c.hubbleparam
#        self.a = 1. / (1. + self.z)
#        self.boxsize = boxsize
#
#        if coords == 'auto': # 10x10x10 grid of halfway cell centres, scaled to box size
#            numcens = 10
#            coordsbase = np.indices((numcens,)*3)[0]
#            coordsbase = (np.asarray(coordsbase,dtype=np.float)+0.5) / float(numcens)
#            self.Coordinates = np.empty((np.prod(coordsbase.shape),3))
#            self.Coordinates[:,0] = np.ndarray.flatten(coordsbase)*self.boxsize
#            self.Coordinates[:,1] = np.ndarray.flatten(np.swapaxes(coordsbase,0,1))*self.boxsize
#            self.Coordinates[:,2] = np.ndarray.flatten(np.swapaxes(coordsbase,0,2))*self.boxsize
#            del coordsbase
#        else:
#            self.Coordinates = coords
#
#        if vel == 'auto':
#            self.Velocity = np.empty(self.Coordinates.shape)
#            self.Velocity[:,0] = 100. #km/s
#            self.Velocity[:,1] = -50. #km/s
#            self.Velocity[:,2] = 150. #km/s
#        elif vel.shape == (3,):
#            self.Velocity = np.empty(self.Coordinates.shape)
#            self.Velocity[:,0] = vel[0] #km/s
#            self.Velocity[:,1] = vel[1] #km/s
#            self.Velocity[:,2] = vel[2] #km/s
#        else:
#            self.Velocity = vel
#
#        self.SmoothingLength = lsmooth*np.ones(self.Coordinates.shape[0])
#        self.readarray = self.read_data_array
#
#    def read_data_array(self, name, gadgetunits=False, region=None, suppress=False):
#        # select correct entry, and set conversion factors
#        if 'Coordinates' in name:
#            out = np.copy(self.Coordinates)
#            if not gadgetunits:
#                out *= self.h**-1*self.a*c.unitlength_in_cm
#            self.a_scaling = 1
#            self.h_scaling = -1
#            self.CGSconversion = c.unitlength_in_cm
#
#        elif 'Velocity' in name:
#            out = np.copy(self.Velocity)
#            if not gadgetunits:
#                out *= self.a**0.5*c.unitvelocity_in_cm_per_s
#            self.a_scaling = 0.5
#            self.h_scaling = 0
#            self.CGSconversion = c.unitvelocity_in_cm_per_s
#
#        elif 'SmoothingLength' in name:
#            out = np.copy(self.SmoothingLength)
#            if not gadgetunits:
#                out *= self.h**-1*self.a*c.unitlength_in_cm
#            self.a_scaling = 1
#            self.h_scaling = -1
#            self.CGSconversion = c.unitlength_in_cm
#
#        else:
#            print('No handling for %s has been implemented'%name)
#
#        #region handling
#        self.mask = Sel()
#        if region is not None:
#            self.mask.comb(region[0] <= self.Coordinates[:,0])
#            self.mask.comb(region[1] >  self.Coordinates[:,0])
#            self.mask.comb(region[2] <= self.Coordinates[:,1])
#            self.mask.comb(region[3] >  self.Coordinates[:,1])
#            self.mask.comb(region[4] <= self.Coordinates[:,2])
#            self.mask.comb(region[5] >  self.Coordinates[:,2])
#        if len(out.shape) == 2:
#            return out[self.mask.val,:]
#        else:
#            return out[self.mask.val]
#
#    def readarray(self,name,region=None,rawunits=False):
#        return self.read_data_array(name, gadgetunits=rawunits, region=region, suppress=False)