#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:12:53 2018

@author: wijers
"""

#import random
import h5py 
import makecddfs as mc
#import os
import numpy as np
from scipy.stats import poisson as ps
from scipy.stats import binom as binom
import sys #to try to figure out what is blowing up memory use: sys.getsizeof (not recursive)
import matplotlib.pyplot as plt

import make_maps_opts_locs as ol
pdir = ol.pdir
ndir = ol.ndir 
mdir = ol.mdir

##### trying to figure out memory use issues

def gettotalsize(obj):
    '''
    recursively find size of lists and arrays
    array -> sys.getsizeof size includes size of subs
    list  -> sys.getsizeof only depends on list length
    '''
    if isinstance(obj, list): # do recursion
        totalsize = sum([gettotalsize(sub) for sub in obj])
        totalsize += sys.getsizeof(obj)
    elif isinstance(obj, np.ndarray):
        if len(obj.shape) <= 1:
            totalsize = sys.getsizeof(obj)
        else: 
            totalsize = sum([gettotalsize(sub) for sub in obj])
    else: # should only be single basic types, like floats or booleans 
        totalsize = sys.getsizeof(obj)
    return totalsize
####

def getrandompix(numpix,number):
    if number > numpix**2:
        print("Cannot choose %i different sightlines from %i total"%(number,numpix**2))
        return None
    xlenyplusy = np.random.choice(numpix**2, number, replace=False) # chooses number from np.arange(numpix**2) uniformly, without replacement
    xs = xlenyplusy/numpix
    ys = xlenyplusy%numpix
    inds = np.array([xs,ys]).T
    return inds

def getlendata(dct,cosmopars, L_z):
    '''
    L_z: summed slice thicknesses in cMpc
    '''
    dct['dx_allfills'] = mc.getdX(cosmopars['z'],L_z,cosmopars=cosmopars) 
    dct['dz_allfills'] = mc.getdz(cosmopars['z'],L_z,cosmopars=cosmopars)
    return None
 
def getoutname(namebase,fills):
    dirtree = namebase.split('/')
    namebase = dirtree[-1] # throw out the file tree: this is added on later: pdir
    
    if isinstance(fills[0],str):
        lenfill = 1
    elif not hasattr(fills[0],'__len__'): #single float value or something
        lenfill = 1
    else: # if fills[0] is a string, length will retun the string length
        lenfill = len(fills[0])
    
    outname = namebase%(('-all',)*lenfill)
    # strip off file extension
    if outname[-4:] == '.npz':
        outname = outname[:-4]
    elif outname[-6:] == '.npz%s': 
        outname = outname[:-6]
    elif outname[-8:] == '.npz-all': 
        outname = outname[:-8]
    if outname[-5:] == '.hdf5':
        outname = outname[:-5]

    outname = 'cddfs_' + outname + '_dXvar'
    
    outname = outname + '.hdf5' 
    return outname

def generate_dX_cddfs(namebase, fills, dXs, numdXs, colmin, colmax, numbins, outname = None):
    '''
    dXs:    target values of dX
    numdXs: number of realisations for each dX value
    
    assumes all slices in a box are used to calculate L_z
    '''

    simdata = mc.get_simdata_from_outputname(namebase%(fills[0])) # simnum, snapnum, var (same for all fills, or at least it should be)
    cosmopars = mc.getcosmopars(simdata['simnum'],simdata['snapnum'],simdata['var'],file_type = 'snap',simulation = simdata['simulation']) # h , omegam, omegalambda, boxsize, a, z
    
    L_z = cosmopars['boxsize']/cosmopars['h'] /float(len(fills)) # boxsize in cMpc/h -> cMpc, add/len(fills) gives fraction of whole box in one slice/slab
    L_z_all = cosmopars['boxsize']/cosmopars['h']

    getlendata(cosmopars, cosmopars, L_z_all) # modify cosmpars to include dX, dz lengths

    if outname is None:
        outname = getoutname(namebase,fills)
    if '/' not in outname:
        outname = pdir + outname
    if '/' not in namebase:
        namebase = ndir + namebase
        
    print('Storing output in %s'%(outname))
    
    if isinstance(numdXs, int):
        numdXs = list((numdXs,)*len(dXs))
    elif len(numdXs) != len(dXs):
        print('Number of dX choices does not match length of list of realisations of each dX')
        return None

    numcddfs = sum(numdXs)
    cddfarr = np.zeros((numcddfs,numbins)) # initialise zero array for cddfs; add to each in each loop over the file name
    leftedges = np.zeros((numcddfs,numbins))
    numpixselect = [int(np.round(dX/cosmopars['dx_allfills'])) for dX in dXs]

    filenames = [namebase%fill for fill in fills]
    arr0 = np.load(filenames[0])['arr_0']

    dXs_true = [num*cosmopars['dx_allfills'] for num in numpixselect]
    dzs_true = [num*cosmopars['dz_allfills'] for num in numpixselect]
    print('Selecting pixels...')
    pixselection = [[getrandompix(arr0.shape[0],numpixselect[dxind]) for realisation in range(numdXs[dxind])] for dxind in range(len(dXs))]
    # convert to list of tuples selector format
    pixselection = [[[tuple(inds) for inds in pixselection[dXind][realisation].T] for realisation in range(numdXs[dxind])] for dXind in range(len(dXs))]
    print('done')
    outputfile = h5py.File(outname,'a') #create if it does not exist, r/w open if it does
    
    # set up header: cosmopars info + file name. Differs from earlier instance only if the number of fills differs
    firstindextoadd = 0
    if outputfile.__contains__('Header'): # file already existed; check compatibility
        header = outputfile['Header']
        storedfiles = header.attrs.get('files_used')
        if set(storedfiles) != set(filenames):
            print('Error: file already exists with a different set of fills. Set a non-default filename.')
            return None
        else:
            alreadypresent = outputfile.keys()
            cddfs = [group for group in alreadypresent if 'cddf' in group]
            inds = [int(cddfname[4:]) for cddfname in cddfs]
            if inds == []:
                firstindextoadd = 0
            else:
                firstindextoadd = max(inds) + 1
            del inds
            del cddfs
            del alreadypresent
    else:
        header = outputfile.create_group('Header')
        for key in cosmopars.keys():
            header.attrs.create(key, cosmopars[key], shape=None, dtype=None)
        header.attrs.create('coherent_sightline_length_cMpc', L_z)
        header.attrs.create('files_used',filenames)
 
    # to get dX index from cddf index
    numcddfsatend = np.cumsum(numdXs)

    # loop over the slices to get the actual cddfs
    print('Starting cddf file loop...')
    firstflag=True
    for filen in filenames:
        print('Processing file %s'%filen)
        if filen == filenames[0]:
            data = arr0
        else:
            data = np.load(filen)['arr_0']
        print('file loaded')
            
        for cddfind in range(numcddfs):
            dXsind = np.min(np.where(cddfind < numcddfsatend)[0]) 
            subind = subind = cddfind - numcddfsatend[dXsind-1]*(dXsind!=0) # if the dX index is zero, dXind-1 will get the last index (wrong); should subtract zero
            print('Doing sample %i of dX = %.2f'%(subind, dXs[dXsind]))
            print('getting subsample')
            inputdata = data[pixselection[dXsind][subind]]
            print('calling make_coldens_hist')
            hist, edges = mc.make_coldens_hist({'arr': inputdata}, L_z_all, cosmopars['z'], bins = numbins, colmin = colmin,colmax = colmax,save = None, cosmopars=cosmopars,verbose=True) # use total L_z since results are added without dividing
           
            cddfarr[cddfind] += hist
            if firstflag:
                leftedges[cddfind] = edges
            elif not np.all(leftedges[cddfind] == edges):
                    print('Bin edge mismatch for cddf %i, file %s.'%(cddfind,filen))
                    return None               
                    
        if filen == filenames[0]:
            del arr0
        del data
        firstflag = False
    print('done')
        
    # save histograms and other relevant data
    print('Saving data...')
    for cddfind in range(numcddfs):
        print('Saving cddf %i'%cddfind)
        group = outputfile.create_group('cddf%i'%(cddfind+firstindextoadd))
        
        dXsind = np.min(np.where(cddfind < numcddfsatend)[0]) 
        subind = cddfind - numcddfsatend[dXsind-1]*(dXsind!=0)

        group.attrs.create('dX_true', dXs_true[dXsind])
        group.attrs.create('dz_true', dzs_true[dXsind])
        group.attrs.create('dX_target', dXs[dXsind])
        group.create_dataset('selected_pixels',data = np.array(pixselection[dXsind][subind]))
        group.create_dataset('leftedges_log10N_cmi2', data = leftedges[cddfind])
        group.create_dataset('dNumber_dNdX', data = cddfarr[cddfind])
    
    outputfile.close()
    print('data saved')
    return None


###### analyse statisitics of the cddfs
def hist_from_cddf(cddf_bins, cddf_edges, cosmopars, numpix):
    hist = mc.cddf_over_pixcount(cosmopars['z'],cosmopars['boxsize']/cosmopars['h'], numpix, cddf_edges, cosmopars=cosmopars)*cddf_bins
    return np.round(hist,0).astype(int) 




# reads in hdf5 file of dX variations, does processing
class cddf_dXvar:
    def getcosmopars(self):
        '''
        Assumes self.file is set (done by __init__)
        sets self.cosmopars: h, a, z, boxsize, omegalambda, omegam
        '''
        self.cosmopars = {}
        self.attrs_temp = self.file['/Header'].attrs  
        self.cosmopars['h'] = self.attrs_temp.get('h')
        self.cosmopars['a'] = self.attrs_temp.get('a')
        self.cosmopars['z'] = self.attrs_temp.get('z')
        self.cosmopars['omegam'] = self.attrs_temp.get('omegam')
        self.cosmopars['omegalambda'] = self.attrs_temp.get('omegalambda')
        self.cosmopars['boxsize'] = self.attrs_temp.get('boxsize')
        del self.attrs_temp

    def getcddfgroups(self):
        '''
        Assumes self.file is set (done by __init__)
        sets self.dX_target, self.dX_true, self.dz_true, self.cddfglens (arrays, length = number of unique target dX lengths)
             self.cddfgroups (list of lists, outer length = same as above: names of cddf groups with different target lengths)   
        '''
        self.keys_temp = self.file['/'].keys() #list  
        self.keys_temp = [key for key in self.keys_temp if 'cddf' in key] # select cddf... groups
        self.dXvals_target_temp = [self.file['/%s'%key].attrs.get('dX_target') for key in self.keys_temp] # get the different unique dX values
        self.dX_target = np.sort(np.array(list(set(self.dXvals_target_temp)))) # sort unique values small - large
        self.cddfgroups = [[self.keys_temp[i] for i in range(len(self.keys_temp)) if self.dXvals_target_temp[i] == dX ] for dX in self.dX_target] # list of cddf groups for each target dX 

        self.numcddfs = len(self.keys_temp)
        self.cddfglens = np.array([len(cddfgroup) for cddfgroup in self.cddfgroups])
        if self.numcddfs != np.sum(self.cddfglens):
            print('An error has occurred in cddf grouping into same dX target lists')
            return None
        self.dX_true = np.array([self.file['/%s'%(keys[0])].attrs.get('dX_true') for keys in self.cddfgroups])
        self.dz_true = np.array([self.file['/%s'%(keys[0])].attrs.get('dz_true') for keys in self.cddfgroups])
          
        del self.keys_temp
        del self.dXvals_target_temp
       

    def __init__(self, filename):
        '''
        init: opens hdf5 file  
        '''
        if '/' not in filename:
            filename = pdir + filename
        self.file = h5py.File(filename,'r')
        self.filename = filename
        self.getcosmopars()
        self.getcddfgroups()

    def getcddfs(self):
        '''
        retrieves bins, edges, numpix info for each cddf
        '''
        self.cddfdata = {}
        self.keys_temp = [key for group in self.cddfgroups for key in group] # flat list here: all cddfs
        self.cddfdata = {key:\
                             {'bins':  np.array(self.file['/%s/dNumber_dNdX'%(key)]),\
                              'edges': np.array(self.file['/%s/leftedges_log10N_cmi2'%(key)]),\
                              'numpix': self.file['/%s/selected_pixels'%(key)].shape[1] } \
                         for key in self.keys_temp}
        del self.keys_temp


    def gethists(self):
        if not hasattr(self,'cddfdata'):
            self.getcddfs()

        # assume all have same bins
        self.edges_temp = self.cddfdata[self.cddfgroups[0][0]]['edges']
        self.binsize_logN = (self.edges_temp[-1] - self.edges_temp[0])/float(len(self.edges_temp)-1)
        self.logcoldens_midbins = self.edges_temp + 0.5*self.binsize_logN
        self.dN = np.log(10.)*10**(self.logcoldens_midbins)*self.binsize_logN

        for key in self.cddfdata.keys(): 
            self.dXind = np.where(np.array([key in group for group in self.cddfgroups]))[0][0]            
            self.cddfdata[key]['hist'] = self.cddfdata[key]['bins']*self.dN*self.dX_true[self.dXind]
        del self.dXind
        #if not np.all(np.array([ np.sum(self.cddfdata[key]['hist']) == self.cddfdata[key]['numpix'] for key in self.cddfdata.keys()])):
        #    print('Error in retrieving histograms from cddfs: number of pixels is not as expected')
        #WRONG: total hist counts = number of pixels (total) * number of slices * fraction of pixels with finite (non-zero) values
        
    def add_total_cddf(self,cddffile):
      
        if '/' not in cddffile:
            cddffile = pdir + cddffile

        self.cddf_total = {}
        if cddffile[-4:] == '.npz':
            self.tempfile = np.load(cddffile)
            self.cddf_total['bins'] = self.tempfile['bins']
            self.cddf_total['edges'] = self.tempfile['logedges']
        
            self.cddf_total            
            self.simdata = mc.get_simdata_from_outputname(cddffile)
        
            self.cddf_total['hist'] = mc.cddf_over_pixcount(self.cosmopars['z'],self.cosmopars['boxsize']/self.cosmopars['h'],self.simdata['numpix']**2, self.cddf_total['edges'], cosmopars=self.cosmopars)*self.cddf_total['bins']

            self.cddf_total['hist'] = np.round(self.cddf_total['hist'],0).astype(int)
            del self.tempfile

    def close(self):
        h5py.close(self.file)

    # plot cddf ratios and percentiles
    # assumes bins in all cddfs including total match
    def getperc(self, percentiles = None, prop = 'nominal'):
        '''
        percentiles: which percentiles to get
        prop: what to get the percentages on. 'nominal' -> histograms, 'cumul' -> cumulative histograms 
        '''
        if not hasattr(self,'cddf_total'):
            print('You need to add a total cddf before you can caluclate this (add_total_cddf(<filename.npz>))')
            return None
        
        if prop == 'nominal':
            self.propkey = 'hist'
            self.propkey_target = 'expval'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles'
        elif prop == 'cumul':
            self.propkey = 'hist_cumul'
            self.propkey_target = 'expval_cumul'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles_cumul'
        elif prop == 'rebin':
            self.propkey = 'hist_rebin'
            self.propkey_target = 'expval_rebin'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles_rebin'
        elif prop == 'cumul_rebin':
            self.propkey = 'hist_cumul_rebin'
            self.propkey_target = 'expval_cumul_rebin'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles_cumul_rebin'
        else:
            print('Invalid prop option %s'%prop)
            return None

        if not hasattr(self,'percentiles'):
            self.percentiles = {}
        if percentiles is None:
            self.percentiles[self.perckey_store] = np.array([0.05,0.15,0.50,0.85,0.95]) # median and 70%, 90% central intervals
        else:
            self.percentiles[self.perckey_store] = np.array(percentiles)

        # start at 0./0.: Nan array
        self.baseline_temp = self.cddf_total[self.propkey]
        if not hasattr(self, 'poissonexp'):
            self.poissonexp = {'numrat' : np.zeros(len(self.dX_target))/0.,\
                           self.propkey_target : np.zeros((len(self.dX_target),len(self.baseline_temp)))/0.,\
                           self.perckey_target: np.zeros((len(self.dX_target), len(self.percentiles[self.perckey_store]), len(self.baseline_temp)))/0.}
        else:
            self.poissonexp[self.propkey_target] =  np.zeros((len(self.dX_target),len(self.baseline_temp)))/0.
            self.poissonexp[self.perckey_target] = np.zeros((len(self.dX_target), len(self.percentiles[self.perckey_store]), len(self.baseline_temp)))/0.
        for dXind in range(len(self.dX_target)):
            
            self.poissonexp['numrat'][dXind] = float(self.cddfdata[self.cddfgroups[dXind][0]]['numpix'])/float(self.simdata['numpix']**2) # subsample number of pixels / total number of pixels
            self.poissonexp[self.propkey_target][dXind] = self.cddf_total[self.propkey]*self.poissonexp['numrat'][dXind] # expectation value for the subsample, given sample size
        
            self.poissonexp[self.perckey_target][dXind] = ps.ppf((self.percentiles[self.perckey_store])[:, np.newaxis], self.poissonexp[self.propkey_target][dXind, np.newaxis, :]) # percentile, bin array (float type, but returns are integer values)

        del self.baseline_temp
        del self.propkey
        del self.propkey_target
        del self.perckey_store
        del self.perckey_target



    def getcumul(self, rtl = True, prop = 'nominal'):
        '''
        rtl: right-to-left, i.e. from dy/dx to y(>=x) (otherwise, gets y(<= x))
        prop: what to get the cumulative distribution of (nominal only, but may include rebin later)
        '''
        if prop == 'nominal':
            self.propkey = 'hist'
            self.propkey_target = 'hist_cumul'
        elif prop == 'rebin':
            self.propkey = 'hist_rebin'
            self.propkey_target = 'hist_cumul_rebin'
        else:
            print('Invalid prop option %s'%prop)
            return None

        #log what was used
        if not hasattr(self, 'cumul_rtl'):
            self.cumul_rtl = {} 
        self.cumul_rtl[self.propkey_target] = rtl

        # cumulative on total cddf
        if hasattr(self,'cddf_total'):
            if rtl: 
                self.cddf_total[self.propkey_target] = np.cumsum((self.cddf_total[self.propkey])[::-1])[::-1]
            else:
                self.cddf_total[self.propkey_target] = np.cumsum((self.cddf_total[self.propkey]))

        # cumulative on dXs
        if rtl: 
            for key in self.cddfdata.keys():            
                self.cddfdata[key][self.propkey_target] = np.cumsum((self.cddfdata[key][self.propkey])[::-1])[::-1]
        else: 
            for key in self.cddfdata.keys():            
                self.cddfdata[key][self.propkey_target] = np.cumsum(self.cddfdata[key][self.propkey])
 

    def getactualperc(self, prop = None):
        '''
        For all calculated percentile postions, calculate the actual percentiles for that position (if expectation values are small, this may differ significantly)
        For percentiles <= 0.50, calculate the fraction in strictly smaller values
        For percentiles >  0.50, calculate the fraction in strictly larger values 
        '''
        # generate percentiles if we didn't have them yet
        if not hasattr(self, 'percentiles'):
            if prop is not None:
                self.getperc(percentiles = None, prop = prop)
            else:
                self.getperc(percentiles = None, prop = 'nominal')
                self.getperc( percentiles = None, prop = 'cumul')

        # which distributions do we have available
        self.props = []
        if prop is None:
            if 'expval' in self.percentiles.keys(): 
                self.props += ['nominal']
            if 'expval_cumul' in self.percentiles.keys():
                self.props += ['cumul']
            if 'expval_rebin' in self.percentiles.keys(): 
                self.props += ['rebin']
            if 'expval_cumul_rebin' in self.percentiles.keys():
                self.props += ['cumul_rebin']
        else:
            self.props += [prop]           
       
        # loop over available distributions and calculate actual percentiles attached to the approximate percentile bins
        for prop in self.props:
            if prop == 'nominal':
                self.propkey = 'hist'
                self.propkey_target = 'expval'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles'
            elif prop == 'cumul':
                self.propkey = 'hist_cumul'
                self.propkey_target = 'expval_cumul'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul'
            elif prop == 'rebin':
                self.propkey = 'hist_rebin'
                self.propkey_target = 'expval_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_rebin'
            elif prop == 'cumul_rebin':
                self.propkey = 'hist_cumul_rebin'
                self.propkey_target = 'expval_cumul_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul_rebin'
   
            self.perckey_true = self.perckey_approx + '_truepercvals'
    
            
            # scipy.stats objects: .sf -> fraction in larger bins (percentile > 0.5 case), .cdf -> fraction in <= bins (call on bin-1 to get percentile <=0.5 case: fraction in < bins)
            self.poissonexp[self.perckey_true] = np.array(\
                 [ ps.cdf(self.poissonexp[self.perckey_approx][:,percind,:] - 1, self.poissonexp[self.propkey_target])   \
                   if self.percentiles[self.perckey_store][percind] <=0.50 else \
                   ps.sf(self.poissonexp[self.perckey_approx][:,percind,:], self.poissonexp[self.propkey_target]) \
                   for percind in range(len(self.percentiles[self.perckey_store]))
                  ]) #-> percentile, dX, bin order
            # put in order conforming to other arrays (e.g. percentiles):
            self.poissonexp[self.perckey_true] = np.transpose(self.poissonexp[self.perckey_true],(1,0,2))
            del self.propkey_target 
            del self.perckey_store
            del self.perckey_approx
            del self.perckey_true


    def getnuminpercbin(self, prop=None):
        if not hasattr(self,'percentiles'):
            self.getactualperc(prop = prop) #will handle percentile generation as well

        # which distributions do we have available: just take self.props from getactualperc
        if prop is not None:
            self.props = [prop]
        
        if not hasattr(self,'poissontest'):
            self.poissontest = {}
        
        for prop in self.props:
            if prop == 'nominal':
                self.propkey = 'hist'
                self.propkey_target = 'expval'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles'
            elif prop == 'cumul':
                self.propkey = 'hist_cumul'
                self.propkey_target = 'expval_cumul'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul'
            elif prop == 'rebin':
                self.propkey = 'hist_rebin'
                self.propkey_target = 'expval_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_rebin'
            elif prop == 'cumul_rebin':
                self.propkey = 'hist_cumul_rebin'
                self.propkey_target = 'expval_cumul_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul_rebin'
            
            # parse percentiles same as for ranges: number <  value for percentiles <=0.5, number  > value for percentiles > 0.5
            self.perc_temp = self.percentiles[self.perckey_store]
            self.dole = self.perc_temp <= 0.5
            # shape of target array:
            self.poissontest[self.perckey_approx] = np.zeros((len(self.dX_target), len(self.percentiles[self.perckey_store]), len(self.cddfdata[self.cddfgroups[0][0]][self.propkey])))/0.                          
            
            for dXind in range(len(self.dX_target)):
                self.cddfchunk = np.array([self.cddfdata[key][self.propkey] for key in self.cddfgroups[dXind] ]) # cddfkey, hist count array
                         
                self.poissontest[self.perckey_approx][dXind] = np.array([\
                    np.sum(self.cddfchunk[:,:] < self.poissonexp[self.perckey_approx][np.newaxis,dXind,percind,:],axis=0)\
                    if self.dole[percind] else\
                    np.sum(self.cddfchunk[:,:] > self.poissonexp[self.perckey_approx][np.newaxis,dXind,percind,:],axis=0)\
                    for percind in range(len(self.percentiles[self.perckey_store]))   ])

                 #newaxis  in poissonexp matches cddf index axis in cddfchunk
            del self.propkey_target
            del self.perckey_store 
            del self.perckey_approx
            del self.propkey
            del self.dole
            del self.perc_temp
            del self.cddfchunk


    def rebin(self,factor=10):

        self.rebin_factor = factor
        if len(self.cddf_total['hist'])%factor != 0:
            print('Cannot rebin by this factor: must divide number of bins %i'%len(self.cddf_total['hist']))

        
        self.propkey = 'hist'
        self.propkey_target = 'expval'
        self.perckey_store = self.propkey_target
        self.perckey_approx = 'percentiles'
        self.propkey_new = 'hist_rebin'

        self.cddf_total[self.propkey_new] = np.sum(self.cddf_total[self.propkey].reshape(len(self.cddf_total[self.propkey])/self.rebin_factor, self.rebin_factor), axis =1 )
        self.cddf_total['edges_rebin'] = self.cddf_total['edges'][::self.rebin_factor]

        
        for key in self.cddfdata.keys():            
            self.cddfdata[key][self.propkey_new] =  np.sum(self.cddfdata[key][self.propkey].reshape(len(self.cddf_total[self.propkey])/self.rebin_factor, self.rebin_factor), axis =1 )
                
        if 'hist_cumul' in self.cddf_total.keys(): # get cumulative rebins while we're at it
            self.getcumul(rtl = self.cumul_rtl['hist_cumul'], prop = 'rebin')


    def getpercpos(self,prop=None, percentiles=None):
        if percentiles is None:
            self.percvals_test = np.array([10.,50.,90.])
        else:
            self.percvals_test = np.array(percentiles)

        if not hasattr(self,'percentiles_test'):
            self.percentiles_test = {}

        if prop is not None:
            self.props = [prop]
        
        for prop in self.props:
            if prop == 'nominal':
                self.propkey = 'hist'
                self.propkey_target = 'expval'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles'
            elif prop == 'cumul':
                self.propkey = 'hist_cumul'
                self.propkey_target = 'expval_cumul'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul'
            elif prop == 'rebin':
                self.propkey = 'hist_rebin'
                self.propkey_target = 'expval_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_rebin'
            elif prop == 'cumul_rebin':
                self.propkey = 'hist_cumul_rebin'
                self.propkey_target = 'expval_cumul_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul_rebin'
            
            # parse percentiles same as for ranges: number <  value for percentiles <=0.5, number  > value for percentiles > 0.5
            self.perc_temp = self.percvals_test
            # shape of target array:
            self.percentiles_test[self.perckey_approx] = np.zeros((len(self.dX_target), len(self.perc_temp), len(self.cddfdata[self.cddfgroups[0][0]][self.propkey])))/0.                          
            
            for dXind in range(len(self.dX_target)):
                self.cddfchunk = np.array([self.cddfdata[key][self.propkey] for key in self.cddfgroups[dXind] ]) # cddfkey, hist count array
                         
                self.percentiles_test[self.perckey_approx][dXind] = np.percentile(self.cddfchunk, self.perc_temp*100, axis=0) # expects percentiles in 0-100 range

                 #newaxis  in poissonexp matches cddf index axis in cddfchunk
            del self.propkey_target
            del self.perckey_store 
            del self.perckey_approx
            del self.propkey
            del self.perc_temp
            del self.cddfchunk

    def doitall(self):
        if not hasattr(self,'cddf_total'):
            print('Add total cddf first to run the statistical analysis')
            return None
        self.gethists()
        self.getcumul()
        self.rebin()
        self.getcumul(prop='rebin')
        self.getperc(prop = 'nominal')
        self.getperc(prop = 'cumul')
        self.getperc(prop = 'cumul_rebin')
        self.getperc(prop = 'rebin')
        self.getactualperc()
        self.getnuminpercbin()
        self.getpercpos()



# emulates cddf_dXvar class, but histograms are generated from cddf_total histograms
class cddf_dXvar_mocksamples:
     
    def getmocks(self, numperdX, dXinds = None):
        # generate and update cddfgroups catalogue   
        if dXinds is not None: 
            self.dXs_temp = self.dX_target[dXinds]
            if not hasattr(self.dXs_temp,'__len__'): #dXs are numbers, so strings shouldn't be an issue
                self.dXs_temp = np.array([self.dXs_temp])
                self.numdXinds = 1
            else:
                self.numdXinds = len(dXinds)
        else:
            self.dXs_temp = self.dX_target
            self.numdXinds = len(self.dX_target)

        if not hasattr(self,'cddfgroups'):
            self.cddfgroups = [[] for dXind in range(len(self.dX_target))] # room to add as many cddfs as in the original cddf_dXvar instance
            self.numcddfs = 0 

        self.cddfgroups_new =  [['cddf%i'%(self.numcddfs + dXind*numperdX + i) for i in range(numperdX)] for dXind in range(len(self.dXs_temp))]
        
        if dXinds is None:
            self.iter_temp = range(self.numdXinds)
        elif hasattr(dXinds, '__len__'):
            self.iter_temp = dXinds 
        else:
            self.iter_temp = [dXinds]
        
        for i in range(len(self.iter_temp)):
            self.cddfgroups[self.iter_temp[i]] += self.cddfgroups_new[i]

        self.numcddfs += self.numdXinds * numperdX

        
        # generate mock catalogues for the new cddfs
        if not hasattr(self,'cddfdata'):
            self.cddfdata = {}
        # cddf with total counts
        self.sample_temp = np.zeros(len(self.cddf_total['hist_cumul']) + 1,dtype=np.int)

        #searchsorted needs ascending order
        if self.rtl:
            self.sample_temp[:-1] = self.cddf_total['hist_cumul'][::-1]
        else:
            self.sample_temp[:-1] = self.cddf_total['hist_cumul']
        self.sample_temp[-1]  = self.totcounts
     
        for ind in range(self.numdXinds * numperdX):
            self.newdXind = ind/numperdX
            self.subind   = ind%numperdX
            self.cddfkey  = self.cddfgroups_new[self.newdXind][self.subind]
            self.dXind = self.iter_temp[self.newdXind]

            self.cddfdata[self.cddfkey] = {}
            self.cddfdata[self.cddfkey]['edges'] = self.cddf_total['edges']
            
            # actually get the subsample
            self.randints = np.random.choice(self.totcounts,self.numcounts_dX[self.dXind]) # choose (all different) random numbers for the survey size
            self.randinds = np.searchsorted(self.sample_temp, self.randints, side = 'left') # left side: bin includes n counts -> n/total goes in that bin, not the next
            self.randinds = self.randinds[self.randinds < len(self.sample_temp) -1] # max indices are for the 'column density too low' bin (regardless of ltr/rtl order, it's just stuck on at the end of the cumulative distribution)

            self.cddfdata[self.cddfkey]['hist'] = np.histogram(self.randinds, bins = np.arange(-0.5, len(self.sample_temp)-1.4, 1.))[0] # count bin indices -> histogram. Exclude index last = bins with zero column density
            self.cddfdata[self.cddfkey]['hist'] = self.cddfdata[self.cddfkey]['hist'][::-1] #histogram from descending order bins (ltr) to ascending rtl array (because the bins were flipped in the used parent cumulative distribution)            
            if self.rtl:          
                self.cddfdata[self.cddfkey]['hist_cumul'] = np.cumsum((self.cddfdata[self.cddfkey]['hist'])[::-1])[::-1]
            else:           
                self.cddfdata[self.cddfkey]['hist_cumul'] = np.cumsum(self.cddfdata[self.cddfkey]['hist'])

        del self.numdXinds
        del self.iter_temp
        del self.cddfgroups_new
        del self.newdXind 
        del self.subind   
        del self.cddfkey 
        del self.dXind
        del self.randints
        del self.randinds


    def readinhdf5(self,filename):
        '''
        like mock generation, but reads stored cddfs in stead of generating new random ones (so the plots don't change every time I make them)
        '''
        # initialise properties if not present (readin in __init__)

        if not hasattr(self,'cddfgroups'):
            self.cddfgroups = [[] for dXind in range(len(self.dX_target))] # room to add as many cddfs as in the original cddf_dXvar instance
            self.numcddfs = 0 

        if not hasattr(self,'cddfdata'):
            self.cddfdata = {}

        self.file = h5py.File(filename,'r')
        if self.orig.filename != self.file['/header'].attrs.get('filename_cxv'):
            print('Warning: using mock generated from file %s in mock oject for file %s'%(self.file['/header'].attrs.get('filename_cxv'), self.orig.filename))

        self.rtl = self.file['/header'].attrs.get('cumulative_right-to-left')

        
        self.cddfgroups_new = [ list(np.array(self.file['/cddfgroups']['%i'%(ind)])) for ind in range(len(self.dX_target))]
        self.numcddfs_new = sum(len(grp) for grp in self.cddfgroups_new)
        self.edges_temp = np.array(self.file['/header']['edges'])

        # read in the cddf groups. parse and adjust names if some cddfs are already present
        self.flatkeys_temp = [key for group in self.cddfgroups_new for key in group]
        for hdkey in self.flatkeys_temp:
            self.hdind = int(hdkey[4:])
            self.newind = self.hdind + self.numcddfs
            
            self.cddfdata['cddf%i'%self.newind] = {}
            self.cddfdata['cddf%i'%self.newind]['edges'] = self.edges_temp         
            self.cddfdata['cddf%i'%self.newind]['hist'] = np.array(self.file[hdkey]['hist'])

            if self.rtl:          
                self.cddfdata['cddf%i'%self.newind]['hist_cumul'] = np.cumsum((self.cddfdata['cddf%i'%self.newind]['hist'])[::-1])[::-1]
            else:           
                self.cddfdata['cddf%i'%self.newind]['hist_cumul'] = np.cumsum(self.cddfdata['cddf%i'%self.newind]['hist'])

        del self.flatkeys_temp
        del self.edges_temp

        # catalogue read-in cddfs (renumber from new offset if needed)
        self.cddfgroups_new  = [['cddf%i'%(int(key[4:]) + self.numcddfs) for key in group] for group in self.cddfgroups_new]

        for grpind in range(len(self.cddfgroups)):
            self.cddfgroups[grpind] += self.cddfgroups_new[grpind] 
        self.numcddfs += self.numcddfs_new

        del self.numcddfs_new
        del self.cddfgroups_new
        

    def __init__(self, cxv, numperdX, depth, dXinds = None, rtl = True, filename = None):
        '''
        depth = number of measured column densites per selected pixel (needed to go from cddf to histogram)
        '''  
        if not hasattr(cxv, 'cddf_total'):
            print('You need to add a total cddf before you can use it to generate mock samples')

        self.orig          = cxv
        self.cddf_total    = cxv.cddf_total
        self.dN            = cxv.dN
        self.binsize_logN  = cxv.binsize_logN
        self.rtl           = rtl   
        self.dX_target     = cxv.dX_target
        self.dX_true       = cxv.dX_true
        self.dz_true       = cxv.dz_true
        self.numcounts_dX  = np.array([cxv.cddfdata[cddfgroup[0]]['numpix'] for cddfgroup in cxv.cddfgroups]) * depth
        self.totcounts     = cxv.simdata['numpix']**2 * depth   # for proper sampling, we need to include the zero column density pixels excluded in the cddfs -> know how many pixels went in in the first place
        
        if self.rtl: 
            self.cddf_total['hist_cumul'] = np.cumsum((self.cddf_total['hist'])[::-1])[::-1]
        else:
            self.cddf_total['hist_cumul'] = np.cumsum((self.cddf_total['hist']))
        
        if filename is None:
            self.getmocks(numperdX, dXinds = dXinds)  
        else:
            if '/' not in filename:
                filename = pdir + filename
            self.readinhdf5(filename)

            # read in cddfs from file
    def getperc(self, percentiles = None, prop = 'nominal'):
        self.orig.getperc(percentiles=percentiles, prop=prop)            
        self.percentiles = self.orig.percentiles
        self.poissonexp = self.orig.poissonexp


    def getperc_binom(self, percentiles = None, prop = 'nominal'):
        '''
        percentiles: which percentiles to get
        prop: what to get the percentages on. 'nominal' -> histograms, 'cumul' -> cumulative histograms 
        uses binomial distribution 
        '''
        if not hasattr(self,'cddf_total'):
            print('You need to add a total cddf before you can caluclate this (add_total_cddf(<filename.npz>))')
            return None
        
        if prop == 'nominal':
            self.propkey = 'hist'
            self.propkey_target = 'expval'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles'
        elif prop == 'cumul':
            self.propkey = 'hist_cumul'
            self.propkey_target = 'expval_cumul'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles_cumul'
        elif prop == 'rebin':
            self.propkey = 'hist_rebin'
            self.propkey_target = 'expval_rebin'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles_rebin'
        elif prop == 'cumul_rebin':
            self.propkey = 'hist_cumul_rebin'
            self.propkey_target = 'expval_cumul_rebin'
            self.perckey_store = self.propkey_target
            self.perckey_target = 'percentiles_cumul_rebin'
        else:
            print('Invalid prop option %s'%prop)
            return None

        if not hasattr(self,'percentiles_binom'):
            self.percentiles_binom = {}
        if percentiles is None:
            self.percentiles_binom[self.perckey_store] = np.array([0.05,0.15,0.50,0.85,0.95]) # median and 70%, 90% central intervals
        else:
            self.percentiles_binom[self.perckey_store] = np.array(percentiles)

        # start at 0./0.: Nan array
        self.baseline_temp = self.cddf_total[self.propkey]
        if not hasattr(self, 'binomexp'):
            self.binomexp = {'numrat' : np.zeros(len(self.dX_target))/0.,\
                           self.propkey_target : np.zeros((len(self.dX_target),len(self.baseline_temp)))/0.,\
                           self.perckey_target: np.zeros((len(self.dX_target), len(self.percentiles_binom[self.perckey_store]), len(self.baseline_temp)))/0.}
        else:
            self.binomexp[self.propkey_target] =  np.zeros((len(self.dX_target),len(self.baseline_temp)))/0.
            self.binomexp[self.perckey_target] = np.zeros((len(self.dX_target), len(self.percentiles_binom[self.perckey_store]), len(self.baseline_temp)))/0.
        for dXind in range(len(self.dX_target)):
            
            self.binomexp['numrat'][dXind] = float(self.numcounts_dX[dXind])/float(self.totcounts) # subsample number of pixels / total number of pixels 
            self.binomexp[self.propkey_target][dXind] = self.cddf_total[self.propkey]*self.binomexp['numrat'][dXind] # expectation value for the subsample, given sample size
        
            self.binomexp[self.perckey_target][dXind] = binom.ppf((self.percentiles_binom[self.perckey_store])[:, np.newaxis], self.numcounts_dX[dXind, np.newaxis, np.newaxis], self.binomexp[self.propkey_target][dXind, np.newaxis, :]/(self.numcounts_dX[dXind, np.newaxis, np.newaxis]).astype(np.float)) # percentile, bin array (float type, but returns are integer values). input: percentiles, n, p

        del self.baseline_temp
        del self.propkey
        del self.propkey_target
        del self.perckey_store
        del self.perckey_target


    def savemocks(self,filename):
        if '/' not in filename:
            filename = pdir + filename
        if filename[-5:] != '.hdf5':
            filename = filename + '.hdf5'
        self.savefile = h5py.File(filename,'w')
        
        # header data
        header = self.savefile.create_group('header')
        header.attrs.create('filename_cxv', self.orig.filename)
        header.attrs.create('cumulative_right-to-left', self.rtl)
        header.attrs.create('numpix_total_mocks', self.numcounts_dX)
        header.attrs.create('numpix_total_projection', self.totcounts)
        header.attrs.create('dX_target', self.dX_target)
        header.create_dataset('edges', data = self.cddf_total['edges']) # so we don't have to store this for every cddf

        # cddfgroups
        groups = self.savefile.create_group('cddfgroups')
        for grpind in range(len(self.cddfgroups)):
            groups.create_dataset('%i'%grpind, data = np.array(self.cddfgroups[grpind]))

        # cddfdata
        for key in self.cddfdata.keys():
            cddf = self.savefile.create_group(key)
            cddf.create_dataset('hist',data = self.cddfdata[key]['hist'])
        
        self.savefile.close()

    def getactualperc(self, prop = None):
        '''
        For all calculated percentile postions, calculate the actual percentiles for that position (if expectation values are small, this may differ significantly)
        For percentiles <= 0.50, calculate the fraction in strictly smaller values
        For percentiles >  0.50, calculate the fraction in strictly larger values 

        just copy from original: expectation values are the same
        '''
        self.orig.getactualperc(prop=prop)
        self.trueperckeys = [key for key in self.orig.poissonexp.keys() if ('_truepercvals' in key) ]
        for key in self.trueperckeys:
            self.poissonexp[key] = self.orig.poissonexp[key]
        del self.trueperckeys

    def rebin(self,factor=None):
        if factor is None:
            if hasattr(self.orig, 'rebin_factor'):
                self.rebin_factor = self.orig.rebin_factor
            else:
                sel.rebin_factor = 10
        else:
            self.rebin_factor = factor
        if len(self.cddf_total['hist'])%self.rebin_factor != 0:
            print('Cannot rebin by this factor: must divide number of bins %i'%len(self.cddf_total['hist']))

        
        self.propkey = 'hist'
        self.propkey_target = 'expval'
        self.perckey_store = self.propkey_target
        self.perckey_approx = 'percentiles'
        self.propkey_new = 'hist_rebin'

        self.cddf_total[self.propkey_new] = np.sum(self.cddf_total[self.propkey].reshape(len(self.cddf_total[self.propkey])/self.rebin_factor, self.rebin_factor), axis =1 )
        self.cddf_total['edges_rebin'] = self.cddf_total['edges'][::self.rebin_factor]

        
        for key in self.cddfdata.keys():            
            self.cddfdata[key][self.propkey_new] =  np.sum(self.cddfdata[key][self.propkey].reshape(len(self.cddf_total[self.propkey])/self.rebin_factor, self.rebin_factor), axis =1 )
                
        if 'hist_cumul' in self.cddf_total.keys(): # get cumulative rebins while we're at it
            if self.rtl: 
                for key in self.cddfdata.keys():         
                    self.cddfdata[key]['hist_cumul_rebin'] = np.cumsum((self.cddfdata[key]['hist_rebin'])[::-1])[::-1]
            else:           
                for key in self.cddfdata.keys():         
                    self.cddfdata[key]['hist_cumul_rebin'] = np.cumsum((self.cddfdata[key]['hist_rebin']))



    def getnuminpercbin(self, prop=None):
        if not hasattr(self,'percentiles'):
            self.getactualperc(prop = prop) #will handle percentile generation as well

        # which distributions do we have available: just take self.props from getactualperc
        if prop is not None:
            self.props = [prop]
        else:
            self.props = self.orig.props
        
        if not hasattr(self,'poissontest'):
            self.poissontest = {}
        
        for prop in self.props:
            if prop == 'nominal':
                self.propkey = 'hist'
                self.propkey_target = 'expval'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles'
            elif prop == 'cumul':
                self.propkey = 'hist_cumul'
                self.propkey_target = 'expval_cumul'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul'
            elif prop == 'rebin':
                self.propkey = 'hist_rebin'
                self.propkey_target = 'expval_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_rebin'
            elif prop == 'cumul_rebin':
                self.propkey = 'hist_cumul_rebin'
                self.propkey_target = 'expval_cumul_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul_rebin'
            
            # parse percentiles same as for ranges: number <  value for percentiles <=0.5, number  > value for percentiles > 0.5
            self.perc_temp = self.percentiles[self.perckey_store]
            self.dole = self.perc_temp <= 0.5
            # shape of target array:
            self.poissontest[self.perckey_approx] = np.zeros((len(self.dX_target), len(self.percentiles[self.perckey_store]), len(self.cddfdata[self.cddfgroups[0][0]][self.propkey])))/0.                          
            
            for dXind in range(len(self.dX_target)):
                self.cddfchunk = np.array([self.cddfdata[key][self.propkey] for key in self.cddfgroups[dXind] ]) # cddfkey, hist count array
                         
                self.poissontest[self.perckey_approx][dXind] = np.array([\
                    np.sum(self.cddfchunk[:,:] < self.poissonexp[self.perckey_approx][np.newaxis,dXind,percind,:],axis=0)\
                    if self.dole[percind] else\
                    np.sum(self.cddfchunk[:,:] > self.poissonexp[self.perckey_approx][np.newaxis,dXind,percind,:],axis=0)\
                    for percind in range(len(self.percentiles[self.perckey_store]))   ])

                 #newaxis  in poissonexp matches cddf index axis in cddfchunk
            del self.propkey_target
            del self.perckey_store 
            del self.perckey_approx
            del self.propkey
            del self.dole
            del self.perc_temp
            del self.cddfchunk

    def getnuminpercbin_subsample(self,size, prop = None):
        '''
        get per bins and cumulative statistics for subsamples of the mock sample
        size is intended to match that of the full sightline sample, so poisson expectations are left out
        creates some highly nested dicts, since different dX values may come with different sample sizes -> different number of subsamples
        self.poissontest_sub[int size]['percentiles' or 'percentiles_cumul'][int dXind] = np.ndarray: #percentiles x # subsamples x #col. dens. bins
        '''
        if not hasattr(self,'percentiles'):
            self.getactualperc(prop = prop) #will handle percentile generation as well

        # which distributions do we have available: just take self.props from getactualperc
        if prop is not None:
            self.props = [prop]
        else:
            self.props = self.orig.props
        
        if not hasattr(self,'poissontest_sub'):
            self.poissontest_sub = {}

        self.numsubs = [len(self.cddfgroups[i])/size for i in range(len(self.cddfgroups))]
        # initialise/update dict structure
        if size not in self.poissontest_sub.keys():
            self.poissontest_sub.update({size: {}})
        if 'percentiles' not in self.poissontest_sub[size].keys():
            self.poissontest_sub[size].update( {'percentiles': {dXind:{} for dXind in range(len(self.dX_target))} })
        if 'percentiles_cumul' not in self.poissontest_sub[size].keys():
            self.poissontest_sub[size].update( {'percentiles_cumul': {dXind:{} for dXind in range(len(self.dX_target))} })
        if 'percentiles_rebin' not in self.poissontest_sub[size].keys():
            self.poissontest_sub[size].update( {'percentiles_rebin': {dXind:{} for dXind in range(len(self.dX_target))} })
        if 'percentiles_cumul_rebin' not in self.poissontest_sub[size].keys():
            self.poissontest_sub[size].update( {'percentiles_cumul_rebin': {dXind:{} for dXind in range(len(self.dX_target))} })

        for dXind in range(len(self.dX_target)):
            for prop in self.props:
                if prop == 'nominal':
                    self.propkey = 'hist'
                    self.propkey_target = 'expval'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles'
                elif prop == 'cumul':
                    self.propkey = 'hist_cumul'
                    self.propkey_target = 'expval_cumul'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles_cumul'
                elif prop == 'rebin':
                    self.propkey = 'hist_rebin'
                    self.propkey_target = 'expval_rebin'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles_rebin'
                elif prop == 'cumul_rebin':
                    self.propkey = 'hist_cumul_rebin'
                    self.propkey_target = 'expval_cumul_rebin'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles_cumul_rebin'
            
                # parse percentiles same as for ranges: number <  value for percentiles <=0.5, number  > value for percentiles > 0.5
                self.perc_temp = self.percentiles[self.perckey_store]
                self.dole = self.perc_temp <= 0.5
                # shape of target array: #percentiles x #subsamples x #col. dens. bins
                self.poissontest_sub[size][self.perckey_approx][dXind] = np.zeros((len(self.perc_temp), self.numsubs[dXind], len(self.cddfdata[self.cddfgroups[0][0]][self.propkey])))/0.                          
            
                self.cddfchunk = np.array([self.cddfdata[key][self.propkey] for key in self.cddfgroups[dXind] ]) # cddfkey, hist count array for dXind
                         
                self.poissontest_sub[size][self.perckey_approx][dXind] = np.array([[\
                        np.sum(self.cddfchunk[sampleind*size:(sampleind+1)*size,:] < self.poissonexp[self.perckey_approx][np.newaxis,dXind,percind,:],axis=0)\
                        if self.dole[percind] else\
                        np.sum(self.cddfchunk[sampleind*size:(sampleind+1)*size,:] > self.poissonexp[self.perckey_approx][np.newaxis,dXind,percind,:],axis=0)\
                    for sampleind in range(self.numsubs[dXind])   ]\
                    for percind in range(len(self.percentiles[self.perckey_store])) ])

                 #newaxis  in poissonexp matches cddf index axis in cddfchunk
        del self.propkey_target
        del self.perckey_store 
        del self.perckey_approx
        del self.propkey
        del self.dole
        del self.perc_temp
        del self.cddfchunk



    def getpercpos(self,prop=None, percentiles=None):
        if percentiles is None:
            if hasattr(self.orig,'percvals_test'):
                self.percvals_test = self.orig.percvals_test
            else:
                self.percvals_test = np.array([10.,50.,90.])
        else:
            self.percvals_test = np.array(percentiles)

        if not hasattr(self,'percentiles_test'):
            self.percentiles_test = {}

        if prop is not None:
            self.props = [prop]
        else:
            self.props = []
            if 'hist' in self.cddfdata[self.cddfdata.keys()[0]].keys():
                self.props += ['nominal']
            if 'hist_cumul' in self.cddfdata[self.cddfdata.keys()[0]].keys():
                self.props += ['cumul']
            if 'hist_rebin' in self.cddfdata[self.cddfdata.keys()[0]].keys():
                self.props += ['rebin']
            if 'hist_cumul_rebin' in self.cddfdata[self.cddfdata.keys()[0]].keys():
                self.props += ['cumul_rebin']
                  
        for prop in self.props:
            if prop == 'nominal':
                self.propkey = 'hist'
                self.propkey_target = 'expval'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles'
            elif prop == 'cumul':
                self.propkey = 'hist_cumul'
                self.propkey_target = 'expval_cumul'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul'
            elif prop == 'rebin':
                self.propkey = 'hist_rebin'
                self.propkey_target = 'expval_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_rebin'
            elif prop == 'cumul_rebin':
                self.propkey = 'hist_cumul_rebin'
                self.propkey_target = 'expval_cumul_rebin'
                self.perckey_store = self.propkey_target
                self.perckey_approx = 'percentiles_cumul_rebin'
            
            # parse percentiles same as for ranges: number <  value for percentiles <=0.5, number  > value for percentiles > 0.5
            self.perc_temp = self.percvals_test
            # shape of target array:
            self.percentiles_test[self.perckey_approx] = np.zeros((len(self.dX_target), len(self.perc_temp), len(self.cddfdata[self.cddfgroups[0][0]][self.propkey])))/0.                          
            
            for dXind in range(len(self.dX_target)):
                self.cddfchunk = np.array([self.cddfdata[key][self.propkey] for key in self.cddfgroups[dXind] ]) # cddfkey, hist count array
                         
                self.percentiles_test[self.perckey_approx][dXind] = np.percentile(self.cddfchunk, self.perc_temp*100, axis=0) # expects percentiles in 0-100 range

                 #newaxis  in poissonexp matches cddf index axis in cddfchunk
            del self.propkey_target
            del self.perckey_store 
            del self.perckey_approx
            del self.propkey
            del self.perc_temp
            del self.cddfchunk


    def getpercpos_subsample(self,size, prop = None, percentiles = None):
        '''
        get per bins and cumulative statistics for subsamples of the mock sample
        size is intended to match that of the full sightline sample, so poisson expectations are left out
        creates some highly nested dicts, since different dX values may come with different sample sizes -> different number of subsamples
        self.poissontest_sub[int size]['percentiles' or 'percentiles_cumul'][int dXind] = np.ndarray: #percentiles x # subsamples x #col. dens. bins
        '''
        if percentiles is None:
            if hasattr(self.orig,'percvals_test'):
                self.percvals_test = self.orig.percvals_test
            else:
                self.percvals_test = np.array([10.,50.,90.])
        else:
            self.percvals_test = np.array(percentiles)


        # which distributions do we have available: just take self.props from getactualperc
        if prop is not None:
            self.props = [prop]
        else:
            self.props = self.orig.props
        
        if not hasattr(self,'percvals_test_sub'):
            self.percvals_test_sub = {}

        self.numsubs = [len(self.cddfgroups[i])/size for i in range(len(self.cddfgroups))]
        # initialise/update dict structure
        if size not in self.percvals_test_sub.keys():
            self.percvals_test_sub.update({size: {}})
        if 'percentiles' not in self.percvals_test_sub[size].keys():
            self.percvals_test_sub[size].update( {'percentiles': {dXind:{} for dXind in range(len(self.dX_target))} })
        if 'percentiles_cumul' not in self.percvals_test_sub[size].keys():
            self.percvals_test_sub[size].update( {'percentiles_cumul': {dXind:{} for dXind in range(len(self.dX_target))} })
        if 'percentiles_rebin' not in self.percvals_test_sub[size].keys():
            self.percvals_test_sub[size].update( {'percentiles_rebin': {dXind:{} for dXind in range(len(self.dX_target))} })
        if 'percentiles_cumul_rebin' not in self.percvals_test_sub[size].keys():
            self.percvals_test_sub[size].update( {'percentiles_cumul_rebin': {dXind:{} for dXind in range(len(self.dX_target))} })

        for dXind in range(len(self.dX_target)):
            for prop in self.props:
                if prop == 'nominal':
                    self.propkey = 'hist'
                    self.propkey_target = 'expval'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles'
                elif prop == 'cumul':
                    self.propkey = 'hist_cumul'
                    self.propkey_target = 'expval_cumul'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles_cumul'
                elif prop == 'rebin':
                    self.propkey = 'hist_rebin'
                    self.propkey_target = 'expval_rebin'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles_rebin'
                elif prop == 'cumul_rebin':
                    self.propkey = 'hist_cumul_rebin'
                    self.propkey_target = 'expval_cumul_rebin'
                    self.perckey_store = self.propkey_target
                    self.perckey_approx = 'percentiles_cumul_rebin'
            
                # parse percentiles same as for ranges: number <  value for percentiles <=0.5, number  > value for percentiles > 0.5
                self.perc_temp = self.percvals_test
                # shape of target array: #percentiles x #subsamples x #col. dens. bins
                self.percvals_test_sub[size][self.perckey_approx][dXind] = np.zeros((len(self.perc_temp), self.numsubs[dXind], len(self.cddfdata[self.cddfgroups[0][0]][self.propkey])))/0.                          
            
                self.cddfchunk = np.array([self.cddfdata[key][self.propkey] for key in self.cddfgroups[dXind] ]) # cddfkey, hist count array for dXind
                         
                self.percvals_test_sub[size][self.perckey_approx][dXind] = np.array([[\
                        np.percentile(self.cddfchunk[sampleind*size:(sampleind+1)*size,:], self.perc_temp[percind]*100 ,axis=0)\
                    for sampleind in range(self.numsubs[dXind])   ]\
                    for percind in range(len(self.perc_temp)) ]) # expects percentiles in 0-100 range

                 #newaxis  in poissonexp matches cddf index axis in cddfchunk
        del self.propkey_target
        del self.perckey_store 
        del self.perckey_approx
        del self.propkey
        del self.perc_temp
        del self.cddfchunk


    def doitall(self):
        self.rebin()
        self.getperc(percentiles = None, prop = 'nominal')
        self.getperc(percentiles = None, prop = 'cumul')
        self.getperc(percentiles = None, prop = 'rebin')
        self.getperc(percentiles = None, prop = 'cumul_rebin')
        self.getactualperc(prop = None)
        self.getnuminpercbin(prop=None)
        self.getnuminpercbin_subsample(100, prop = None)
        self.getpercpos(prop=None, percentiles=None)
        self.getpercpos_subsample(100, prop = None)



def plotperc(cxv, dXind, sampleind = None, ax=None, xlim = (11.,17.5), ylim = (-4.,4.5), dolegend = True, legendloc = 0, doxlabel = True, doylabel = True, cumul = False, background = True, foreground=True, alpha = 0.5):
    '''
    plots (cddf_subsample - expected) / sigma(expected) on top of expected median and percentile ranges
    assumes percentiles are default
    '''
    fontsize=12
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
    if cumul:
        perckey = 'percentiles_cumul'
        expkey = 'expval_cumul'
        propkey = 'hist_cumul'
    else:
        perckey = 'percentiles'
        expkey = 'expval'
        propkey = 'hist'
    
    first = True
    # plot expected ranges and median; percentiles are offset by 0.5 for clearer in/out display
    if background:
        ax.fill_between(cxv.cddf_total['edges'],\
                    (cxv.poissonexp[perckey][dXind,0] - cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp[expkey][dXind]),\
                    (cxv.poissonexp[perckey][dXind,-1] - cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp[expkey][dXind]),\
                    color = 'dodgerblue', alpha = 0.5, label = 'expected 90%', step='post')
        ax.fill_between(cxv.cddf_total['edges'],\
                    (cxv.poissonexp[perckey][dXind,1] - cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp[expkey][dXind]),\
                    (cxv.poissonexp[perckey][dXind,-2] - cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp[expkey][dXind]),\
                    color = 'blue', alpha = 0.5, label = 'expected 70%', step = 'post')
        ax.step(cxv.cddf_total['edges'],\
                    (cxv.poissonexp[perckey][dXind,2] - cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp[expkey][dXind]),\
                    color = 'red', label = 'expected median', where = 'post', linewidth=2)
        ax.step(cxv.cddf_total['edges'],\
                    ( -1.* cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp['expval'][dXind]) - 0.1,\
                    color = 'navy', label = 'zero - 0.1', where = 'post', linewidth=1)

    # plot normalised cddfs
    if foreground:
        toplot = np.array(cxv.cddfgroups[dXind]) # array allows more complicated index selections
        if sampleind is not None:
            toplot = toplot[sampleind]
            if toplot[0] == 'c': #string 'cddf..'
                toplot = [toplot]

        for key in toplot:
            if first: 
                label = 'dX = %.2e'%cxv.dX_true[dXind]
            else:
                label = None
            ax.hlines((cxv.cddfdata[key][propkey].astype(np.float) - cxv.poissonexp[expkey][dXind])/np.sqrt(cxv.poissonexp[expkey][dXind]),\
                cxv.cddfdata[key]['edges'], cxv.cddfdata[key]['edges'] + cxv.binsize_logN, color = 'black', linewidth = 2, label = label, alpha = alpha, zorder = 10)
            first = False

    # axes
    if doxlabel:
        ax.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$', fontsize = fontsize)
    if doylabel:
        if cumul: 
            ax.set_ylabel(r'$(\mathrm{num}(>N) - \langle \mathrm{num}(>N) \rangle_{\mathrm{Psn}}) / \sqrt{\langle \mathrm{num}(>N) \rangle_{\mathrm{Psn}}}$', fontsize = fontsize)
        else: 
            ax.set_ylabel(r'$(\mathrm{num} - \langle \mathrm{num} \rangle_{\mathrm{Psn}}) / \sqrt{\langle \mathrm{num} \rangle_{\mathrm{Psn}}}$', fontsize = fontsize)
    ax.tick_params(labelsize=fontsize-1,direction='in',top=True,right=True, which = 'both')
    ax.minorticks_on()

    if dolegend:
        ax.legend(loc = legendloc, fontsize=fontsize)

    if xlim is not None:
        ax.set_xlim(*tuple(xlim))
    if ylim is not None:
        zerovals =  -1.* cxv.poissonexp['expval'][dXind]/np.sqrt(cxv.poissonexp['expval'][dXind]) # plot y values corresponding to zero counts
        zeromin = np.min(zerovals[np.isfinite(zerovals)]) 
        if ylim[0] < zeromin:
            ax.set_ylim((zeromin-0.2,ylim[1]))
        else:
            ax.set_ylim(*tuple(ylim))


def plotperc_separate(cxv, dXind, shape = (5,2), cumul = False):
    fig, axes = plt.subplots(ncols = shape[0], nrows = shape[1], sharex = True, sharey=True, figsize = (shape[0]*3.5,shape[1]*3.5), gridspec_kw = {'wspace': 0., 'hspace': 0.})
    dolegend = True
    for i in range(len(cxv.cddfgroups[dXind])):
        yind = i/(shape[0])
        xind = i%(shape[0])
        doxlabel = (yind == shape[1]-1)
        doylabel = (xind == 0)
        #print('x: %s, y: %s'%(doxlabel,doylabel))
        plotperc(cxv, dXind, sampleind = i, ax = axes[yind, xind], dolegend = dolegend, legendloc = 'lower right', doxlabel = doxlabel, doylabel = doylabel, cumul = cumul)
        dolegend = False


def plotperc_alldX(cxv,cumul = False,alpha = 0.5):
    ndX = len(cxv.dX_target)
    fig, axes = plt.subplots(ncols = ndX, nrows = 1, sharex = True, sharey=True, figsize = (ndX*3.5, 3.5), gridspec_kw = {'wspace': 0., 'hspace': 0.})

    for dXind in range(ndX):
        dolegend = (dXind == 0)
        plotperc(cxv, dXind, sampleind = None, ax=axes[dXind], xlim = (11.,17.5), ylim = (-4.,4.5), dolegend = dolegend, legendloc = 0, doxlabel = True, doylabel = dolegend, cumul = cumul, background = True, foreground=True, alpha = alpha)
        axes[dXind].set_title('dX = %.2e'%(cxv.dX_true[dXind]))
             

def plotintervalmatch(cxv, dXind, percind, cumul = False, ax = None, xlim = None, ylim = None, dolegend = True, legendloc = None, doylabel = True, mocksubs = False, colors = None, showpoisson=True, zorder = None):
    fontsize=12
    if ax is None:
        fig, ax = plt.subplots(nrows = 1, ncols = 1)

    if hasattr(percind, '__len__'):
        interval = True
    else:
        interval = False

    if not hasattr(dXind, '__len__'):
        dXind  = [dXind]
    elif dXind == 'all':
        dXind = range(len(cxv.dX_target))

    if cumul:
        tperckey = 'percentiles_cumul_truepercvals'
        measkey  = 'percentiles_cumul'
        perckey  = 'expval_cumul'
    else:
        tperckey = 'percentiles_truepercvals'
        measkey  = 'percentiles'
        perckey  = 'expval'
    
    edges = cxv.cddf_total['edges']

    if interval:
        trueperc = 1 - (cxv.poissonexp[tperckey][:,percind[0]] + cxv.poissonexp[tperckey][:,percind[1]]) # assume interval[0] < 0.5, interval[1] > 0.5
        targetp = cxv.percentiles[perckey][percind[1]] - cxv.percentiles[perckey][percind[0]]
        if mocksubs == False:
            measperc = np.array([float(len(cxv.cddfgroups[di])) for di in dXind])[:,np.newaxis] - (cxv.poissontest[measkey][:,percind[0]] + cxv.poissontest[measkey][:,percind[1]]) # total cddfs in each dX bin - number more extreme than each percentile end
        else: #mocksubs is assumed to be the size value (key in poissontest_sub)
            measperc = np.array([float(mocksubs) for di in range(len(cxv.dX_target))])[:,np.newaxis,np.newaxis] - \
                       np.array([\
                         cxv.poissontest_sub[mocksubs][measkey][ind][percind[0]] + cxv.poissontest_sub[mocksubs][measkey][ind][percind[1]]\
                         for ind in  range(len(cxv.dX_target))])
        labelpart = r'fraction in $\sim %.2f$ percentile range'%(cxv.percentiles[perckey][percind[1]] - cxv.percentiles[perckey][percind[0]] )
    else:
        trueperc = cxv.poissonexp[tperckey][:,percind] 
        targetp = cxv.percentiles[perckey][percind]
        if mocksubs == False:
            measperc = cxv.poissontest[measkey][:,percind]
        else:
            measperc = np.array([ cxv.poissontest_sub[mocksubs][measkey][ind][percind]  for ind in range(len(cxv.dX_target))])
        if targetp <= 0.5:
            labelpart = r'fraction below percentile $\sim %.2f$' %(cxv.percentiles[perckey][percind])
        else:
            labelpart = r'fraction above percentile $\sim %.2f$' %(cxv.percentiles[perckey][percind])
    # go from number to fraction
    print(measperc.shape)
    if mocksubs == False:
        measperc = measperc.astype(np.float)/np.array([float(len(cxv.cddfgroups[di])) for di in dXind])[:,np.newaxis]
    else:
        measperc = measperc.astype(np.float)/np.array([float(mocksubs) for di in range(len(cxv.dX_target))])[:,np.newaxis,np.newaxis]
    print(measperc.shape)
   
    if colors is None:
        colors = ['purple', 'blue', 'green', 'orange', 'red']
    
    for ind in dXind:
        if mocksubs == False:
            if showpoisson:
                ax.step(edges,trueperc[ind],where='post', linestyle = 'dashed', color = colors[ind], label = 'dX = %.2e, Poisson'%(cxv.dX_true[ind]),zorder=zorder)
            ax.step(edges,measperc[ind],where='post', linestyle = 'solid', color = colors[ind], label = 'dX = %.2e, mock'%(cxv.dX_true[ind]),zorder=zorder, linewidth = 1)
        else:
            if showpoisson:
                ax.step(edges,trueperc[ind],where='post', linestyle = 'dashed', color = colors[ind], label = 'dX = %.2e, Poisson'%(cxv.dX_true[ind]),zorder=zorder)
            first = True
            for sampleind in range(measperc[ind].shape[0]):
                if first:
                    label =  'dX = %.2e, mock'%(cxv.dX_true[ind])
                else:
                    label = None
                ax.step(edges,measperc[ind,sampleind],where='post', linestyle = 'solid', color = colors[ind], label = label,zorder=zorder)
                #ax.hlines(measperc[ind,sampleind], edges, edges + cxv.binsize_logN, linestyle = 'solid', color = colors[ind], label = label,zorder=zorder, linewidth = 2, alpha = 0.5)
                first = False

    ax.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)
    if doylabel:
        ax.set_ylabel(labelpart,fontsize=fontsize)

    ax.tick_params(labelsize=fontsize-1,direction='in',top=True,right=True, which = 'both')
    ax.minorticks_on()

    if dolegend:
        ax.legend(loc = legendloc, fontsize=fontsize)

    if xlim is not None:
        ax.set_xlim(*tuple(xlim))
    if ylim is not None:
        ax.set_ylim(*tuple(ylim))


def plotintervalmatch_dXseparate(cxv, percind, cumul = False,mocksubs = False, cxv_mock = None):
    ndX = len(cxv.dX_target)
    fig, axes = plt.subplots(ncols = ndX, nrows = 1, sharex = True, sharey=True, figsize = (ndX*4.5, 4.5), gridspec_kw = {'wspace': 0., 'hspace': 0.})

    if mocksubs != False:
        for dXind in range(ndX):
            dolegend = False
            plotintervalmatch(cxv_mock, dXind, percind, cumul = cumul, ax=axes[dXind], xlim = None, ylim = None, dolegend = dolegend, legendloc = None, doylabel = dolegend, colors = ['mediumorchid','cyan','limegreen','darksalmon','pink'], showpoisson = False, zorder = 0, mocksubs = mocksubs)
        axes[dXind].set_title('dX = %.2e'%(cxv.dX_true[dXind]))
        
    for dXind in range(ndX):
        dolegend = (dXind == 0)
        plotintervalmatch(cxv, dXind, percind, cumul = cumul, ax=axes[dXind], xlim = None, ylim = None, dolegend = dolegend, legendloc = None, doylabel = dolegend,zorder=1)
        axes[dXind].set_title('dX = %.2e'%(cxv.dX_true[dXind]))



def plotpercentilehists(cxv,cxv_mock, cumul = False, rebin = True, mocksize =100, add_binom = False, add_poisson = True):
    ndX = len(cxv.dX_target)
    fig, axes = plt.subplots(ncols = ndX, nrows = 1, sharex = True, sharey=False, figsize = (ndX*4.5, 4.5), gridspec_kw = {'wspace': 0., 'hspace': 0.})

    if not cumul and not rebin:
        propkey = 'hist'
        propkey_target = 'expval'
        perckey_store = propkey_target
        perckey_approx = 'percentiles'
    elif cumul and not rebin:
        propkey = 'hist_cumul'
        propkey_target = 'expval_cumul'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_cumul'
    elif not cumul and rebin:
        propkey = 'hist_rebin'
        propkey_target = 'expval_rebin'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_rebin'      
    elif cumul and rebin:
        propkey = 'hist_cumul_rebin'
        propkey_target = 'expval_cumul_rebin'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_cumul_rebin'
    perckey_target = perckey_approx

    if rebin:
        edges = cxv.cddf_total['edges_rebin']
    else:
        edges = cxv.cddf_total['edges']
    if cumul:
        where = 'pre'
    else:
        where = 'post'
    

    for dXind in range(ndX):
        ax = axes[dXind]
        ax.set_xlabel(r'$\log_{10} N \, \mathrm{cm}^{-2}$')
        ax.set_title('dX = %.2e'%(cxv.dX_true[dXind]))
        ax.set_yscale('log')

        first = True
        for sampleind in range(len(cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][0])):
            if first:
                labels = ['10%, cddf random','50%, cddf random','90%, cddf random']
            else:
                labels = [None,None,None]
            ax.step(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][0][sampleind], where=where, color = 'cyan', alpha = 0.2, label = labels[0], linewidth = 3)        
            ax.step(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][1][sampleind], where=where, color = 'limegreen', alpha = 0.2, label = labels[1], linewidth = 3)   
            ax.step(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][2][sampleind], where=where, color = 'lightcoral', alpha = 0.2, label = labels[2], linewidth = 3)   
            first = False

        if add_binom:
            ax.step(edges, cxv_mock.binomexp[perckey_target][dXind][0], where=where, color = 'midnightblue', linestyle = 'dotted', label = '10%, binom. exp')
            ax.step(edges, cxv_mock.binomexp[perckey_target][dXind][1], where=where, color = 'darkslategray', linestyle = 'dotted', label = '50%, binom. exp')
            ax.step(edges, cxv_mock.binomexp[perckey_target][dXind][2], where=where, color = 'maroon', linestyle = 'dotted', label = '90%, binom. exp')
            #ax.step(edges, cxv_mock.binomexp[propkey_target][dXind], where=where, color = 'darkslategray', linestyle = 'dotted', label = 'binom. exp. val.')

        if add_poisson:
            ax.step(edges, cxv_mock.poissonexp[perckey_target][dXind][0], where=where, color = 'midnightblue', linestyle = 'dashdot', label = '10%, Psn. exp')
            ax.step(edges, cxv_mock.poissonexp[perckey_target][dXind][1], where=where, color = 'darkslategray', linestyle = 'dashdot', label = '50%, Psn. exp')
            ax.step(edges, cxv_mock.poissonexp[perckey_target][dXind][2], where=where, color = 'maroon', linestyle = 'dashdot', label = '90%, Psn. exp')
            #ax.step(edges, cxv_mock.poissonexp[propkey_target][dXind], where=where, color = 'darkslategray', linestyle = 'dashdot', label = 'Psn. exp. val.')

        ax.step(edges, cxv.percentiles_test[perckey_approx][dXind][0], where=where, color = 'blue', linestyle = 'dashed', label = '10%, mock survey')
        ax.step(edges, cxv.percentiles_test[perckey_approx][dXind][1], where=where, color = 'green', linestyle = 'dashed', label = '50%, mock survey')
        ax.step(edges, cxv.percentiles_test[perckey_approx][dXind][2], where=where, color = 'red', linestyle = 'dashed', label = '90%, mock survey')
        
       
        
        if dXind == 0:
            ax.set_ylabel('Counts/bin')
            ax.legend()


##################################
###        paper plots         ###
##################################

datas = {} # avoid recalculating the same stuff for different plots -> save data, mock objects in this dictionary

def getdata_for_percentile_hists(coherent_mock_file, cddf_file, random_mock_file, percentiles = np.array([0.10, 0.50, 0.90]), rebin_factor = 10, cxv = None, mockslices = 16):
    '''
    mock file has mockslices number of slices
    assumes size 100 mock subsamples
    '''
    if cxv is None:
        cxv = cddf_dXvar(coherent_mock_file)
    cxv.add_total_cddf(cddf_file)
    # get histograms, rebinned, cumulative
    cxv.gethists() # histograms from cddfs 
    cxv.rebin(factor=rebin_factor) # 0.05 -> 0.5 dex bins   
    cxv.getcumul()
    cxv.rebin()
    cxv.getcumul(prop='rebin')
    # get expected Poisson distribution (get binomial from mocks)
    cxv.getperc(percentiles = percentiles, prop = 'nominal')
    cxv.getperc(percentiles = percentiles, prop = 'cumul')
    cxv.getperc(percentiles = percentiles, prop = 'cumul_rebin')
    cxv.getperc(percentiles = percentiles, prop = 'rebin')
    # get percentiles for the mock surveys
    cxv.getpercpos(percentiles = percentiles, prop = 'nominal')
    cxv.getpercpos(percentiles = percentiles, prop = 'cumul')
    cxv.getpercpos(percentiles = percentiles, prop = 'rebin')
    cxv.getpercpos(percentiles = percentiles, prop = 'cumul_rebin')

    cxvm = cddf_dXvar_mocksamples(cxv,0,mockslices,filename = random_mock_file)    
    cxvm.rebin(factor=rebin_factor) # 0.05 -> 0.5 dex bins   
    # Poisson, binomial expectation values for percentile locations
    cxvm.getperc(percentiles = percentiles, prop = 'nominal')
    cxvm.getperc(percentiles = percentiles, prop = 'cumul')
    cxvm.getperc(percentiles = percentiles, prop = 'rebin')
    cxvm.getperc(percentiles = percentiles, prop = 'cumul_rebin')
    cxvm.getperc_binom(percentiles = percentiles, prop = 'nominal')
    cxvm.getperc_binom(percentiles = percentiles, prop = 'cumul')
    cxvm.getperc_binom(percentiles = percentiles, prop = 'rebin')
    cxvm.getperc_binom(percentiles = percentiles, prop = 'cumul_rebin')
    # get percentiles for the mock sample and the size 100 subsamples
    cxvm.getpercpos(prop=None, percentiles=percentiles)
    cxvm.getpercpos_subsample(100, prop = 'nominal')
    cxvm.getpercpos_subsample(100, prop = 'cumul')
    cxvm.getpercpos_subsample(100, prop = 'rebin')
    cxvm.getpercpos_subsample(100, prop = 'cumul_rebin')

    return cxv, cxvm

def save_data_for_percentile_hist(open_hdf5_group, dat, mock, mocksize=100):
    grp = open_hdf5_group
    
    grp.create_group('cosmopars')
    for key in dat.cosmopars.keys():
        grp['cosmopars'].attrs.create(key, dat.cosmopars[key])
    grp.create_dataset('dX_target', data=dat.dX_target)
    grp.create_dataset('dX_true', data=dat.dX_true)
    grp.create_dataset('dz_true', data=dat.dz_true)
    grp.create_dataset('left_edges', data=dat.cddf_total['edges'])
    grp.create_dataset('left_edges_rebin', data=dat.cddf_total['edges_rebin'])
   
    histtypes = ['histogram', 'histogram_rebin', 'cumulative', 'cumulative_rebin']
    for histtype in histtypes:
        grp_sub = grp.create_group(histtype)
        if histtype == 'histogram':
            perckey_approx = 'percentiles'
        elif histtype == 'cumulative':
            perckey_approx = 'percentiles_cumul'
        elif histtype == 'histogram_rebin':
            perckey_approx = 'percentiles_rebin'      
        elif histtype == 'cumulative_rebin':
            perckey_approx = 'percentiles_cumul_rebin'
        perckey_target = perckey_approx

    # cxv_mock.percvals_test_sub: mocksize, type, dX value index, percentile index, sample index (same number of variations for each)
        dsname = 'random_sample_distribution'
        dataset = np.array([mock.percvals_test_sub[mocksize][perckey_approx][dXind] for dXind in range(len(mock.percvals_test_sub[mocksize][perckey_approx]))])
        grp_sub.create_dataset(dsname, data=dataset)
        grp_sub[dsname].attrs.create('dimension', np.array(['dX_index', 'percentile_index', 'random_sample_index', 'column_density_bin']))
        
        dsname = 'binomial_expectation'
        dataset = mock.binomexp[perckey_target]
        grp_sub.create_dataset(dsname, data=dataset)
        grp_sub[dsname].attrs.create('dimension', np.array(['dX_index', 'percentile_index', 'column_density_bin']))
       
        dsname = 'poisson_expectation'
        dataset = mock.poissonexp[perckey_target]
        grp_sub.create_dataset(dsname, data=dataset)
        grp_sub[dsname].attrs.create('dimension', np.array(['dX_index', 'percentile_index', 'column_density_bin']))
        
        dsname = 'mock_sample_distribution'
        dataset = dat.percentiles_test[perckey_approx]
        grp_sub.create_dataset(dsname, data=dataset)
        grp_sub[dsname].attrs.create('dimension', np.array(['dX_index', 'percentile_index', 'column_density_bin']))
            
def save_data_for_percentile_hists(name='mock_surveys'):
    percentiles = [0.10, 0.50, 0.90]
    rebin_factor = 10
    mocksize = 100 # mock survey property, not analysis variable
    
    o7mockname = 'cddfs_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar.hdf5'
    o7cddfname = 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
    o7randname = 'cddfs_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar_randomfromcddf_mocks.hdf5'
    o8mockname = 'cddfs_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar.hdf5'
    o8cddfname = 'cddf_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
    o8randname = 'cddfs_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar_randomfromcddf_mocks.hdf5'
    o7dat, o7mock =  getdata_for_percentile_hists(o7mockname, o7cddfname,  o7randname, percentiles=percentiles, rebin_factor=rebin_factor)
    o8dat, o8mock =  getdata_for_percentile_hists(o8mockname, o8cddfname,  o8randname, percentiles = percentiles, rebin_factor = rebin_factor)
    
    with h5py.File('/net/luttero/data2/paper1/%s.hdf5'%(name), 'a') as outfile:
        hed = outfile.create_group('Header')
        hed.attrs.create('info', 'mock survey CDDFs generated two ways: randomly from the total CDDF (random) and random pixels, but same pixels in each slice (mock)')
        hed.attrs.create('rebin', 'larger column density bins than in CDDFs')
        hed.attrs.create('cumulative', 'cumulative distribution (high-to-low)')
        hed.create_dataset('percentiles', data=np.array(percentiles))
        hed.attrs.create('rebin_factor', rebin_factor)
        hed.attrs.create('number_of_surveys_per_size', mocksize)
        
        grp_o7 = outfile.create_group('o7')
        save_data_for_percentile_hist(grp_o7, o7dat, o7mock, mocksize=mocksize)
        grp_o8 = outfile.create_group('o8')
        save_data_for_percentile_hist(grp_o8, o8dat, o8mock, mocksize=mocksize)
            


def subplotpercentilehists(ax, cxv, cxv_mock, dXind, cumul = False, rebin = True, mocksize =100, add_binom = False, add_poisson = True, xlabel = None, ylabel = None, subtitle = None, subfigindloc = None, subfigind = None, ylog = True, fontsize=12, xlim = None, ylim = None, dolegend = True):

    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize)
    if ylog:
        ax.set_yscale('log', nonposy='clip')
    ax.tick_params(labelsize=fontsize-1,direction='in',top=True,right=True, which = 'both')
    ax.minorticks_on()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
  
    if subtitle is not None:
        ax.text(0.95,0.95,subtitle,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    if subfigind is not None:
        if subfigindloc is None:
            ax.text(0.85,0.05,subfigind,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)
        else: 
            ax.text(subfigindloc[0],subfigindloc[1],subfigind,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)

    if not cumul and not rebin:
        propkey = 'hist'
        propkey_target = 'expval'
        perckey_store = propkey_target
        perckey_approx = 'percentiles'
    elif cumul and not rebin:
        propkey = 'hist_cumul'
        propkey_target = 'expval_cumul'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_cumul'
    elif not cumul and rebin:
        propkey = 'hist_rebin'
        propkey_target = 'expval_rebin'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_rebin'      
    elif cumul and rebin:
        propkey = 'hist_cumul_rebin'
        propkey_target = 'expval_cumul_rebin'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_cumul_rebin'
    perckey_target = perckey_approx

    if rebin:
        edges = cxv.cddf_total['edges_rebin']
    else:
        edges = cxv.cddf_total['edges']
    if cumul:
        where = 'pre'
    else:
        where = 'post'
    



    first = True
    for sampleind in range(len(cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][0])):
        if first:
            labels = ['0.1, rnd','0.5, rnd','0.9, rnd']
        else:
            labels = [None,None,None]
        ax.plot(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][0][sampleind], color = 'cyan', alpha = 0.2, label = labels[0], linewidth = 3)        
        ax.plot(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][1][sampleind], color = 'limegreen', alpha = 0.2, label = labels[1], linewidth = 3)   
        ax.plot(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][2][sampleind], color = 'lightcoral', alpha = 0.2, label = labels[2], linewidth = 3)   
        first = False

    if add_binom:
        ax.plot(edges, cxv_mock.binomexp[perckey_target][dXind][0], color = 'midnightblue', linestyle = 'dotted', label = '10%, binom. exp')
        ax.plot(edges, cxv_mock.binomexp[perckey_target][dXind][1], color = 'darkslategray', linestyle = 'dotted', label = '50%, binom. exp')
        ax.plot(edges, cxv_mock.binomexp[perckey_target][dXind][2], color = 'maroon', linestyle = 'dotted', label = '90%, binom. exp')
        #ax.step(edges, cxv_mock.binomexp[propkey_target][dXind], where=where, color = 'darkslategray', linestyle = 'dotted', label = 'binom. exp. val.')

    if add_poisson:
        ax.plot(edges, cxv_mock.poissonexp[perckey_target][dXind][0], color = 'midnightblue', linestyle = 'dashdot', label = '10%, Psn. exp')
        ax.plot(edges, cxv_mock.poissonexp[perckey_target][dXind][1], color = 'darkslategray', linestyle = 'dashdot', label = '50%, Psn. exp')
        ax.plot(edges, cxv_mock.poissonexp[perckey_target][dXind][2], color = 'maroon', linestyle = 'dashdot', label = '90%, Psn. exp')
            #ax.step(edges, cxv_mock.poissonexp[propkey_target][dXind], where=where, color = 'darkslategray', linestyle = 'dashdot', label = 'Psn. exp. val.')

    ax.plot(edges, cxv.percentiles_test[perckey_approx][dXind][0], color = 'blue', linestyle = 'dashed', label = '0.1, mock')
    ax.plot(edges, cxv.percentiles_test[perckey_approx][dXind][1], color = 'green', linestyle = 'dashed', label = '0.5, mock')
    ax.plot(edges, cxv.percentiles_test[perckey_approx][dXind][2], color = 'red', linestyle = 'dashed', label = '0.9, mock')

    #ax.scatter(edges, cxv.percentiles_test[perckey_approx][dXind][0], color = 'blue', linestyle = 'dashed', label = None, zorder = 5)
    #ax.scatter(edges, cxv.percentiles_test[perckey_approx][dXind][1], color = 'green', linestyle = 'dashed', label = None, zorder = 5)
    #ax.scatter(edges, cxv.percentiles_test[perckey_approx][dXind][2], color = 'red', linestyle = 'dashed', label = None, zorder = 5)
        
    if dolegend:   
        ax.legend(fontsize=fontsize)

    
def plotdXvar_percentile_hists(recalc = False):
    '''
    saved as cddfs_coldens_o7-o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dX-1-10-100_percentiles_10-50-90.png
    '''

    percentiles = [0.10, 0.50, 0.90]
    rebin_factor = 10
    
    if 'o7' in datas and not recalc:
        o7dat, o7mock = datas['o7']
    else:
        o7dat, o7mock =  getdata_for_percentile_hists('cddfs_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar.hdf5', 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz',  'cddfs_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar_randomfromcddf_mocks.hdf5', percentiles = percentiles, rebin_factor = rebin_factor)
        datas['o7'] = (o7dat, o7mock)

    if 'o8' in datas and not recalc:
        o8dat, o8mock = datas['o8']
    else:
        o8dat, o8mock =  getdata_for_percentile_hists('cddfs_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar.hdf5', 'cddf_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz',  'cddfs_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar_randomfromcddf_mocks.hdf5', percentiles = percentiles, rebin_factor = rebin_factor)
        datas['o8'] = (o8dat, o8mock)
   

    ### make the plots (this is a temporary version)
    # cumulative plots are best plotted just as points, I think
    rebin = False
    cumul = True   
    fontsize=12 
    xlim = (11.8,16.6)
    #ylim = (0.3,1e4)

    fig, axes = plt.subplots(ncols=2, nrows = 3, sharex = True, sharey = 'row', gridspec_kw = {'wspace': 0., 'hspace': 0.}, figsize=(5.5,8.))

    subplotpercentilehists(axes[0][0], o7dat, o7mock, 1, cumul = cumul, rebin = rebin, mocksize =100, add_binom = False, add_poisson = False, xlabel = None, ylabel = r'absorbers$\,(>N)$', subtitle = r'O VII, $\mathrm{d}X = 1.0$', subfigindloc = (0.95,0.70), subfigind = '(a)', ylog = True, fontsize=fontsize, xlim = xlim, ylim = (0.3,3.e2), dolegend = False)

    subplotpercentilehists(axes[1][0], o7dat, o7mock, 2, cumul = cumul, rebin = rebin, mocksize =100, add_binom = False, add_poisson = False, xlabel = None, ylabel = r'absorbers$\,(>N)$', subtitle = r'$\mathrm{d}X = 10$', subfigindloc = (0.95,0.70), subfigind = '(c)', ylog = True, fontsize=fontsize, xlim = xlim, ylim = (0.3,3.e3), dolegend = False)

    subplotpercentilehists(axes[2][0], o7dat, o7mock, 3, cumul = cumul, rebin = rebin, mocksize =100, add_binom = False, add_poisson = False, xlabel = r'$\log_{10} N \, [\mathrm{cm}^{-2}]$', ylabel = r'absorbers$\,(>N)$', subtitle = r'$\mathrm{d}X = 100$', subfigindloc = (0.95,0.70), subfigind = '(e)', ylog = True, fontsize=fontsize, xlim = xlim, ylim = (0.3,3.e4), dolegend = False)

    subplotpercentilehists(axes[0][1], o8dat, o8mock, 1, cumul = cumul, rebin = rebin, mocksize =100, add_binom = False, add_poisson = False, xlabel = None, ylabel = None, subtitle = r'O VIII, $\mathrm{d}X =1.0$', subfigindloc = (0.95,0.70), subfigind = '(b)', ylog = True, fontsize=fontsize, xlim = xlim, ylim = (0.3,3.e2), dolegend = False)

    subplotpercentilehists(axes[1][1], o8dat, o8mock, 2, cumul = cumul, rebin = rebin, mocksize =100, add_binom = False, add_poisson = False, xlabel = None, ylabel = None, subtitle = r'$\mathrm{d}X = 10$', subfigindloc = (0.95,0.70), subfigind = '(d)', ylog = True, fontsize=fontsize, xlim = xlim, ylim = (0.3,3.e3), dolegend = False)

    subplotpercentilehists(axes[2][1], o8dat, o8mock, 3, cumul = cumul, rebin = rebin, mocksize =100, add_binom = False, add_poisson = False, xlabel = r'$\log_{10} N \, [\mathrm{cm}^{-2}]$', ylabel = None, subtitle = r'$\mathrm{d}X = 100$', subfigindloc = (0.95,0.70), subfigind = '(f)', ylog = True, fontsize=fontsize, xlim = xlim, ylim = (0.3,3.e4), dolegend = False)

    # add legend: divide over subplots to avoid data overlap
    handles_subs, labels_subs = axes[0][0].get_legend_handles_labels()
    handles_subs = np.array(handles_subs)
    labels_subs = np.array(labels_subs)

    axes[1][0].legend(handles=list(handles_subs[np.array([0,3])]), fontsize=fontsize,ncol=1)
    axes[2][0].legend(handles=list(handles_subs[np.array([1,4])]), fontsize=fontsize,ncol=1)
    axes[2][1].legend(handles=list(handles_subs[np.array([2,5])]), fontsize=fontsize,ncol=1)


    #plotpercentilehists(o7dat,o7mock, cumul = True, rebin = True, mocksize =100, add_binom = True, add_poisson = True)
    #plotpercentilehists(o8dat,o8mock, cumul = True, rebin = True, mocksize =100, add_binom = True, add_poisson = True)
    
    plt.savefig(mdir + 'cddfs_coldens_o7-o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dX-1-10-100_percentiles_10-50-90.png', format = 'png',bbox_inches = 'tight')    



def subplotpercentilehists_rel(ax, cxv, cxv_mock, dXind, cumul = False, rebin = True, mocksize =100, add_binom = False, add_poisson = True, xlabel = None, ylabel = None, subtitle = None, subfigindloc = None, subfigind = None, ylog = True, fontsize=12, xlim = None, ylim = None, dolegend = True, yticklabels = True):

    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize)
    if ylog:
        ax.set_yscale('log', nonposy='clip')
    ax.tick_params(labelsize=fontsize-1,direction='in',top=True,right=True, which = 'both')
    ax.minorticks_on()

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if not yticklabels:
        ax.tick_params(axis='both',labelleft=False)
  
    if subtitle is not None:
        ax.text(0.05,0.95,subtitle,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    if subfigind is not None:
        if subfigindloc is None:
            ax.text(0.05,0.70,subfigind,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax.transAxes)
        else: 
            ax.text(subfigindloc[0],subfigindloc[1],subfigind,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)

    if not cumul and not rebin:
        propkey = 'hist'
        propkey_target = 'expval'
        perckey_store = propkey_target
        perckey_approx = 'percentiles'
    elif cumul and not rebin:
        propkey = 'hist_cumul'
        propkey_target = 'expval_cumul'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_cumul'
    elif not cumul and rebin:
        propkey = 'hist_rebin'
        propkey_target = 'expval_rebin'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_rebin'      
    elif cumul and rebin:
        propkey = 'hist_cumul_rebin'
        propkey_target = 'expval_cumul_rebin'
        perckey_store = propkey_target
        perckey_approx = 'percentiles_cumul_rebin'
    perckey_target = perckey_approx

    if rebin:
        edges = cxv.cddf_total['edges_rebin']
    else:
        edges = cxv.cddf_total['edges']
    if cumul:
        where = 'pre'
    else:
        where = 'post'
    

    refvals = cxv_mock.binomexp[perckey_target][dXind]

    first = True
    for sampleind in range(len(cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][0])):
        if first:
            labels = ['0.1, rnd','0.5, rnd','0.9, rnd']
        else:
            labels = [None,None,None]
        ax.plot(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][0][sampleind]/refvals[0], color = 'cyan', alpha = 0.2, label = labels[0], linewidth = 3)        
        ax.plot(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][1][sampleind]/refvals[1], color = 'limegreen', alpha = 0.2, label = labels[1], linewidth = 3)   
        ax.plot(edges, cxv_mock.percvals_test_sub[mocksize][perckey_approx][dXind][2][sampleind]/refvals[2], color = 'lightcoral', alpha = 0.2, label = labels[2], linewidth = 3)   
        first = False

    if add_binom:
        ax.plot(edges, cxv_mock.binomexp[perckey_target][dXind][0]/refvals[0], color = 'midnightblue', linestyle = 'dotted', label = '0.1, bn')
        ax.plot(edges, cxv_mock.binomexp[perckey_target][dXind][1]/refvals[1], color = 'darkslategray', linestyle = 'dotted', label = '0.5, bn')
        ax.plot(edges, cxv_mock.binomexp[perckey_target][dXind][2]/refvals[2], color = 'maroon', linestyle = 'dotted', label = '0.9, bn')
        #ax.step(edges, cxv_mock.binomexp[propkey_target][dXind], where=where, color = 'darkslategray', linestyle = 'dotted', label = 'binom. exp. val.')

    if add_poisson:
        ax.plot(edges, cxv_mock.poissonexp[perckey_target][dXind][0]/refvals[0], color = 'midnightblue', linestyle = 'dashdot', label = '0.5, Psn')
        ax.plot(edges, cxv_mock.poissonexp[perckey_target][dXind][1]/refvals[1], color = 'darkslategray', linestyle = 'dashdot', label = '0.5, Psn')
        ax.plot(edges, cxv_mock.poissonexp[perckey_target][dXind][2]/refvals[2], color = 'maroon', linestyle = 'dashdot', label = '0.9, Psn')
            #ax.step(edges, cxv_mock.poissonexp[propkey_target][dXind], where=where, color = 'darkslategray', linestyle = 'dashdot', label = 'Psn. exp. val.')

    ax.plot(edges, cxv.percentiles_test[perckey_approx][dXind][0]/refvals[0], color = 'blue', linestyle = 'dashed', label = '0.1, mock')
    ax.plot(edges, cxv.percentiles_test[perckey_approx][dXind][1]/refvals[1], color = 'green', linestyle = 'dashed', label = '0.5, mock')
    ax.plot(edges, cxv.percentiles_test[perckey_approx][dXind][2]/refvals[2], color = 'red', linestyle = 'dashed', label = '0.9, mock')

    #ax.scatter(edges, cxv.percentiles_test[perckey_approx][dXind][0], color = 'blue', linestyle = 'dashed', label = None, zorder = 5)
    #ax.scatter(edges, cxv.percentiles_test[perckey_approx][dXind][1], color = 'green', linestyle = 'dashed', label = None, zorder = 5)
    #ax.scatter(edges, cxv.percentiles_test[perckey_approx][dXind][2], color = 'red', linestyle = 'dashed', label = None, zorder = 5)
        
    if dolegend:   
        ax.legend(fontsize=fontsize)

def logformatter1(x, pos):
    """The two args are the value and tick position.
    very, very customized for the retive plots """
    label = '%.1f' % (x)
    if label[-2:] == '.0':
        label = label[:-2]
    if label in ['0.6', '0.8', '0.9']: # remove ticks 'by hand'
        label = ''
    return label

def logformatter2(x, pos):
    """The two args are the value and tick position.
    very, very customized for the retive plots """
    label = '%.1f' % (x)
    if label[-2:] == '.0':
        label = label[:-2]
    return label

def plotdXvar_percentile_hists_rel(recalc = False):
    '''
    saved as cddfs_coldens_o7-o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dX-1-10-100_percentiles_10-50-90_relative.png
    '''

    percentiles = [0.10, 0.50, 0.90]
    rebin_factor = 10
    
    if 'o7' in datas and not recalc:
        o7dat, o7mock = datas['o7']
    else:
        o7dat, o7mock =  getdata_for_percentile_hists('cddfs_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar.hdf5', 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz',  'cddfs_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar_randomfromcddf_mocks.hdf5', percentiles = percentiles, rebin_factor = rebin_factor)
        datas['o7'] = (o7dat, o7mock)

    if 'o8' in datas and not recalc:
        o8dat, o8mock = datas['o8']
    else:
        o8dat, o8mock =  getdata_for_percentile_hists('cddfs_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar.hdf5', 'cddf_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz',  'cddfs_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dXvar_randomfromcddf_mocks.hdf5', percentiles = percentiles, rebin_factor = rebin_factor)
        datas['o8'] = (o8dat, o8mock)
   

    ### make the plots (this is a temporary version)
    # cumulative plots are best plotted just as points, I think
    rebin = False
    cumul = True   
    fontsize=12 
    xlim = (11.8,16.6)
    #ylim = (0.3,1e4)

    fig, axes = plt.subplots(ncols=2, nrows = 3, sharex = True, sharey = 'row', gridspec_kw = {'wspace': 0., 'hspace': 0.}, figsize=(5.5,8.))

    formatter1 = plt.FuncFormatter(logformatter1)
    formatter2 = plt.FuncFormatter(logformatter2)
    axes[0][0].set_yscale('log')
    axes[0][1].set_yscale('log')
    axes[1][0].set_yscale('log')
    axes[1][1].set_yscale('log')
    axes[2][0].set_yscale('log')
    axes[2][1].set_yscale('log')
    axes[0][0].yaxis.set_major_formatter(formatter1)
    axes[0][0].yaxis.set_minor_formatter(formatter1)
    axes[1][0].yaxis.set_major_formatter(formatter1)
    axes[1][0].yaxis.set_minor_formatter(formatter1)
    axes[2][0].yaxis.set_major_formatter(formatter2)
    axes[2][0].yaxis.set_minor_formatter(formatter2)

    subplotpercentilehists_rel(axes[0][0], o7dat, o7mock, 1, cumul = cumul, rebin = rebin, mocksize =100, add_binom = True, add_poisson = True, xlabel = None, ylabel = r'absorbers$\,(>N)$ / bn', subtitle = r'O VII, $\mathrm{d}X = 1.0$', subfigindloc = None, subfigind = '(a)', ylog = False, fontsize=fontsize, xlim = xlim, ylim = (0.45,2.1), dolegend = False)

    subplotpercentilehists_rel(axes[1][0], o7dat, o7mock, 2, cumul = cumul, rebin = rebin, mocksize =100, add_binom = True, add_poisson = True, xlabel = None, ylabel = r'absorbers$\,(>N)$ / bn', subtitle = r'O VII, $\mathrm{d}X = 10$', subfigindloc = None, subfigind = '(c)', ylog = False, fontsize=fontsize, xlim = xlim, ylim = (0.45,2.1), dolegend = False)

    subplotpercentilehists_rel(axes[2][0], o7dat, o7mock, 3, cumul = cumul, rebin = rebin, mocksize =100, add_binom = True, add_poisson = True, xlabel = r'$\log_{10} N \, [\mathrm{cm}^{-2}]$', ylabel = r'absorbers$\,(>N)$ / bn', subtitle = r'O VII, $\mathrm{d}X = 100$', subfigindloc = None, subfigind = '(e)', ylog = False, fontsize=fontsize, xlim = xlim, ylim = (0.8,1./0.8), dolegend = False)

    subplotpercentilehists_rel(axes[0][1], o8dat, o8mock, 1, cumul = cumul, rebin = rebin, mocksize =100, add_binom = True, add_poisson = True, xlabel = None, ylabel = None, subtitle = r'O VIII, $\mathrm{d}X =1.0$', subfigindloc = None, subfigind = '(b)', ylog = False, fontsize=fontsize, xlim = xlim, ylim = None, dolegend = False, yticklabels=False)

    subplotpercentilehists_rel(axes[1][1], o8dat, o8mock, 2, cumul = cumul, rebin = rebin, mocksize =100, add_binom = True, add_poisson = True, xlabel = None, ylabel = None, subtitle = r'O VIII, $\mathrm{d}X = 10$', subfigindloc = None, subfigind = '(d)', ylog = False, fontsize=fontsize, xlim = xlim, ylim = None, dolegend = False, yticklabels=False)

    subplotpercentilehists_rel(axes[2][1], o8dat, o8mock, 3, cumul = cumul, rebin = rebin, mocksize =100, add_binom = True, add_poisson = True, xlabel = r'$\log_{10} N \, [\mathrm{cm}^{-2}]$', ylabel = None, subtitle = r'O VIII, $\mathrm{d}X = 100$', subfigindloc = None, subfigind = '(f)', ylog = False, fontsize=fontsize, xlim = xlim, ylim = None, dolegend = False, yticklabels=False)

    # add legend: divide over subplots to avoid data overlap
    handles_subs, labels_subs = axes[0][0].get_legend_handles_labels()
    handles_subs = np.array(handles_subs)
    labels_subs = np.array(labels_subs)

    axes[1][0].legend(handles=list(handles_subs[np.array([0,1,2])]), fontsize=fontsize,ncol=1, loc = 'lower left' )
    axes[1][1].legend(handles=list(handles_subs[np.array([3,4,5])]), fontsize=fontsize,ncol=1, loc = 'lower left' )
    axes[2][0].legend(handles=list(handles_subs[np.array([6,7,8])]), fontsize=fontsize,ncol=1, loc = 'lower left' )
    axes[2][1].legend(handles=list(handles_subs[np.array([9,10,11])]), fontsize=fontsize,ncol=1, loc = 'lower left' )
 
    axes[0][1].tick_params(axis='both', which = 'both', labelleft=False, labelright = False)
    axes[1][1].tick_params(axis='both', which = 'both', labelleft=False, labelright = False)
    axes[2][1].tick_params(axis='both', which = 'both', labelleft=False, labelright = False)

    

    #plotpercentilehists(o7dat,o7mock, cumul = True, rebin = True, mocksize =100, add_binom = True, add_poisson = True)
    #plotpercentilehists(o8dat,o8mock, cumul = True, rebin = True, mocksize =100, add_binom = True, add_poisson = True)
    
    plt.savefig(mdir + 'cddfs_coldens_o7-o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_dX-1-10-100_percentiles_10-50-90_relative.png', format = 'png',bbox_inches = 'tight')  
    
    
