'''
For post-processing specwizard output, and the make_maps output to compare to
'''
from __future__ import print_function #to allow printing without newlines

import numpy as np 
import h5py 
import scipy.integrate as si
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import cosmo_utils as cu
import loadnpz_and_plot as lnp
import make_maps_opts_locs as ol
import ctypes as ct





sdir = ol.sdir
ldir = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/'
ndir = ol.ndir
pdir = ol.pdir
mdir = ol.mdir

# in Angstrom; 
#   multiplet values from Verner, Verner, and Ferland (1996) for o8; better match to curve of growth using larger doublet component fosc = 0.277
#   specwizard values for o7    
#   h1 sum of oscillator strengths from specwizard, fosc-weighted average wavelength 

# same as data in specwizard atomic constants (modules)
# except O7 He-alpha triplet: r from there, i, f from Serena Bertone's table, 
#                             (wavelength), confirmed by Jelle Kaastra's teaching notes
#                             https://personal.sron.nl/~kaastra/leiden2017/lnotes_part5.pdf:
#                             Finally, because the oscillator strength for the
#                             intercombination and forbidden line are much 
#                             smaller than for the resonance line, if one 
#                             observes the triplet in ab- sorption, one usually 
#                             only sees the resonance line. Only for high 
#                             nuclear charge (like iron, Z = 26) the 
#                             intercombination line also starts contributing to 
#                             the absorption.
#                             http://adsabs.harvard.edu/full/1967ApJ...148..573E:
#                             intercombination line oscillator strength (small)
#                             https://www.aanda.org/articles/aa/pdf/2015/07/aa26324-15.pdf:
#                             pg 4, citing NIST database: i, f lines negligible compared to
#                             r line
#        O6 lines ar 21, 22 mA from Jelle Kaastra's list
# 'major' indicates the line used in specwizard for multiplets -> basis for the rescaling
# lyman alpha: https://books.google.nl/books?id=FycJvKHyiwsC&pg=PA83&redir_esc=y#v=onepage&q&f=false, 
#@INPROCEEDINGS{2016ASSL..423..187M,
#       author = {{Mortlock}, Daniel},
#        title = "{Quasars as Probes of Cosmological Reionization}",
#     keywords = {Physics, Astrophysics - Cosmology and Nongalactic Astrophysics},
#    booktitle = {Understanding the Epoch of Cosmic Reionization: Challenges and Progress},
#         year = "2016",
#       editor = {{Mesinger}, Andrei},
#       series = {Astrophysics and Space Science Library},
#       volume = {423},
#        month = "Jan",
#        pages = {187},
#          doi = {10.1007/978-3-319-21957-8_7},
#archivePrefix = {arXiv},
#       eprint = {1511.01107},
# primaryClass = {astro-ph.CO},
#       adsurl = {https://ui.adsabs.harvard.edu/\#abs/2016ASSL..423..187M},
#      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
#}

lambda_rest = {\
               'o8':        18.9689,\
               'o8major':   18.9671,\
               'o8minor':   18.9725,\
               'o8_2major': 16.0055,\
               'o8_2minor': 16.0067,\
               'h1':        1156.6987178884301,\
               'c3':        977.0201,\
               'c4major':   1548.2041,\
               'c5major':   40.2678,\
               'c6major':   33.7342,\
               'n6':        28.7875,\
               'n7major':   24.7792,\
               'n7minor':   24.7846,\
               'ne8major':  770.409,\
               'ne9':       13.4471,\
               'o4':        787.711,\
               'o5':        629.730,\
               'o6major':   1031.9261,\
               'o6minor':   1037.6167,\
               'o6_2major': 22.0189,\
               'o6_2minor': 22.0205,\
               'si3':       1206.500,\
               'si4major':  1393.76018,\
               'fe17major': 15.0140,\
               'fe17minor': 15.2610,\
               'c6major':   33.7342,\
               'c6minor':   33.7396,\
               'c6_2major': 28.4652,\
               'c6_2minor': 28.4663, \
               'o7major':   21.6019,\
               'o7':        21.6019,\
               'o7_2':      18.6284,\
               'lyalpha':   1215.67,\
               }
fosc ={'o7':        0.696,\
       'o7major':   0.696,\
       'o7_2':      0.146,\
       'o8':        0.416,\
       'o8major':   0.277,\
       'o8minor':   0.139,\
       'o8_2major': 0.0527,\
       'o8_2minor': 0.0263,\
       'fe17major': 2.72,\
       'fe17minor': 0.614,\
       'c6major':   0.277,\
       'c6minor':   0.139,\
       'c6_2major': 0.0527,\
       'c6_2minor': 0.0263, \
       'o6_major':  0.13250,\
       'o6_minor':  0.06580,\
       'o6_2major': 0.351,\
       'o6_2minor': 0.174,\
       'n6':        0.675,\
       'n7major':   0.277,\
       'n7minor':   0.139,\
       'ne9':       0.724,\
       'h1':        0.5644956,\
       'lyalpha':   0.4162,\
       } 
        

  
multip = {'o8': ['o8major', 'o8minor', 'o8_2major', 'o8_2minor'],\
          'fe17': ['fe17major', 'fe17minor'],\
          'n7': ['n7major', 'n7minor'],\
          'c6': ['c6major', 'c6minor', 'c6_2major', 'c6_2minor'],\
          'o7': ['o7major', 'o7_2'],\
          }

#from Lan & Fukugita 2017, citing e.g. B.T. Draine, 2011: Physics of the interstellar and intergalactic medium
# ew in angstrom, Nion in cm^-2
# Lan and Fukugita use rest-frame wavelengths; possibly better match using redshifted wavelengths. (deltaredshift uses the same value as in coldens hist. calc., then EW useslambda_rest*deltaredshift)
# 1.13e20 comes from (np.pi*4.80320427e-10**2/9.10938215e-28/2.99792458e10**2 / 1e8)**-1 = ( np.pi*e[statcoulomb]**2/(m_e c**2) *Angstrom/cm )**-1
def lingrowthcurve(ew,ion):
    if ion not in fosc.keys():
        ion = ion + 'major'
    return (np.pi* cu.c.electroncharge**2 /(cu.c.electronmass*cu.c.c**2) *1e-8)**-1 * ew/(fosc[ion]*lambda_rest[ion]**2)  
def lingrowthcurve_inv(Nion,ion):
    if ion not in fosc.keys():
        ion = ion + 'major'
    return Nion * (fosc[ion]*lambda_rest[ion]**2) * (np.pi* cu.c.electroncharge**2 /(cu.c.electronmass*cu.c.c**2)*1e-8)

def linflatcurveofgrowth_inv(Nion,b,ion):
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
            
    if ion in multip.keys():
        tau0s = np.array([(np.pi**0.5* cu.c.electroncharge**2 /(cu.c.electronmass*cu.c.c) *1e-8) *lambda_rest[line]*fosc[line]*Nion/b for line in multip[ion]]).T
        xoffsets = (cu.c.c/b)* (np.array([lambda_rest[line] for line in multip[ion]]) - lambda_rest[multip[ion][0]])/lambda_rest[ion] # it shouldn't matter reltive to which the offset is taken
        #print(tau0s)
        #print(xoffsets)     
        prefactor = lambda_rest[ion]/cu.c.c*b # just use the average here
        # absorption profiles are multiplied to get total absorption
        
        integral = np.array([si.quad(lambda x: 1- np.exp(np.sum(-taus*np.exp(-1*(x-xoffsets)**2),axis=0)),-np.inf,np.inf) for taus in tau0s])

    else:
        if ion == 'o8_assingle':
            ion = 'o8'
        if ion not in fosc.keys():
            ion = ion + 'major'
        tau0 = (np.pi**0.5* cu.c.electroncharge**2 /(cu.c.electronmass*cu.c.c) *1e-8) *lambda_rest[ion]*fosc[ion]*Nion/b
        prefactor = lambda_rest[ion]/cu.c.c*b
        #def integrand(x):
        #    1- np.exp(-tau0*np.exp(-1*x**2))
        integral = np.array([si.quad(lambda x: 1- np.exp(-tau*np.exp(-1*x**2)),-np.inf,np.inf) for tau in tau0])

    if np.max(integral[:,1]/integral[:,0]) > 1e-5:
        print('Warning: check integration errors in linflatcurveofgrowth_inv')
    return prefactor*integral[:,0]

def Nion_from_tauv(tau,ion):
    '''
    Same reference as linflatcurveofgrowth; tau here is tau_v (result of specwizard projection), not tau_nu or tau_lambda
    '''
    if ion not in fosc.keys():
        ion = ion + 'major'
    return (np.pi* cu.c.electroncharge**2 /(cu.c.electronmass*cu.c.c) * fosc[ion])**-1  * 1. / (lambda_rest[ion] * 1.e-8) * tau 
    # tau at that pixel -> total Nion represented there 
    # (really Nion*normailised spectrum, but we've got the full tau spectra, so no need to factor that out)
    
hcosm =  0.6777

# meant for comparing parameter files
def comparetxt(name1,name2):
    file1 = open(name1,'r')
    file2 = open(name2,'r')
    lines1 = sum(1 for line in open(name1))
    lines2 = sum(1 for line in open(name2))
    if lines1 != lines2:
        print('Files are of different length:\n %s has %i\n %s has %i'%(name1,lines1,name2,lines2))
    lines = max(lines1,lines2)
    for i in range(lines):
        line1 = file1.readline()
        line2 = file2.readline()
        if not np.all(line1 == line2):
            print('difference in line %i:\n%s%s'%(i,line1,line2))

 



# extract groups containing spectra
# spectrum files contained in groups Spectrum<number>

class Specout:
    def __init__(self, filename, getall=True):
        '''
        sets the hdf5 file self.specfile, and extracts and calculates some stuff
        '''
        if filename[:len(sdir)] == sdir:
            self.filename = filename
        else:
            self.filename = sdir + filename
        self.specfile = h5py.File(self.filename,'r')
        self.specgroups = np.array(self.specfile.keys())
        self.isspec = np.array(['Spectrum' in group for group in self.specgroups])
        self.numspecs = np.sum(self.isspec)
        self.specgroup = self.specgroups[self.isspec] 
        del self.isspec
        
        self.cosmopars = {'boxsize':     self.specfile['/Header'].attrs.get('BoxSize'),\
                          'h':           self.specfile['/Header'].attrs.get('HubbleParam'),\
                          'a':           self.specfile['/Header'].attrs.get('ExpansionFactor'),\
                          'z':           self.specfile['/Header'].attrs.get('Redshift'),\
                          'omegam':      self.specfile['/Header'].attrs.get('Omega0'),\
                          'omegalambda': self.specfile['/Header'].attrs.get('OmegaLambda'),\
                          'omegab':      self.specfile['/Header'].attrs.get('OmegaBaryon') }
        
        # set general sightline and box properties
        self.positions = np.array([np.array([self.specfile['Spectrum%i'%specnum].attrs['X-position'], self.specfile['Spectrum%i'%specnum].attrs['Y-position']]) for specnum in range(self.numspecs)]) # in gagdet units = 1/h cMpc
        self.redshift = self.specfile['Header'].attrs['Redshift']
        self.slicelength = self.specfile['Header'].attrs['BoxSize']/hcosm # in cMpc
        self.deltaredshift = cu.Hubble(self.redshift)/cu.c.c*self.slicelength*cu.c.cm_per_mpc # observed Delta z, since the comoving slice length is used

        self.numspecpix = self.specfile['VHubble_KMpS'].shape[0]

        # set dicts to add to per ion
        self.spectra = {}
        self.coldens = {}
        self.EW      = {}
        self.nion    = {}
        self.posmassw= {}
        self.posionw = {}
        self.veltauw = {}
        self.tau     = {}

        self.ions = np.array(self.specfile['Spectrum0'].keys()) # if the file has no lines of sight, it's useless anyway
        self.ision = np.array(['RealSpaceMass' not in group for group in self.ions])
        self.ions = self.ions[self.ision]
        del self.ision
 
        self.dataoptions = {\
             'posmassw':\
             {'pecvel':      'RealSpaceMassWeighted/LOSPeculiarVelocity_KMpS',\
              'zmassfrac':   'RealSpaceMassWeighted/MetalMassFraction',\
              'overdensity': 'RealSpaceMassWeighted/OverDensity',\
              'temperature': 'RealSpaceMassWeighted/Temperature_K'},\
             'ion':\
             {'flux':        'Flux',\
              'logcoldens':  'LogTotalIonColumnDensity',\
              'tau':         'OpticalDepth'},\
             'posionw':\
             {'pecvel':      'RealSpaceNionWeighted/LOSPeculiarVelocity_KMpS',\
              'nion':        'RealSpaceNionWeighted/Nion_CM3',\
              'overdensity': 'RealSpaceNionWeighted/OverDensity',\
              'temperature': 'RealSpaceNionWeighted/Temperature_K'},\
             'veltauw':\
             {'pecvel':      'RealSpaceNionWeighted/LOSPeculiarVelocity_KMpS',\
              'nion':        'RealSpaceNionWeighted/Nion_CM3',\
              'overdensity': 'RealSpaceNionWeighted/OverDensity',\
              'temperature': 'RealSpaceNionWeighted/Temperature_K'}\
             }

        if getall:
            self.getall()

    def getspectra(self,dions='all'):
        if dions == 'all':
            dions = self.ions
        doo8 = False
        # o8 is a doublet. remove all from dions, then do a separate version with a correction
        if 'o8' in dions:
            dions = list(dions)
            while 'o8' in dions:
                dions.remove('o8')
            dions = np.array(dions)
            doo8 = True
        if len(dions) > 0: # if dions was just o8, no need to risk errors, just leave it
            self.spectra.update({ion : np.array([np.array(self.specfile['Spectrum%i/%s/Flux'%(specnum,ion)]) for specnum in range(self.numspecs)]) for ion in dions})
        if doo8:
            self.correct_o8()
            

             
    def getcoldens(self, dions='all', slices=1, offset=0., realspace=False, recalc_nion=False):
        '''
        recalc_nion can force a calcultion with the desired realspace setting
        '''
        if slices == 1: # take total column density from file
            if dions == 'all':
                dions = self.ions
            self.coldens.update({ion: np.array([self.specfile['Spectrum%i/%s/LogTotalIonColumnDensity'%(specnum,ion)][()] for specnum in range(self.numspecs)]) for ion in dions})
        elif not isinstance(slices,int) or not isinstance(offset,int):
            print('slices keyword must have an integer value')
        else: # get column densities from nion values in closest approx. to slices
            slicelist = self.getslices(slices,offset=offset,posvals=None, posvalperiod = None)
            if not set(dions).issubset(self.nion.keys()) or recalc_nion:
                for ion in dions:
                    if not ion in self.nion.keys:
                        self.getnion(dions = [ion],realspace=realspace)   
            # add up densities * slicelength/(1+z) (= proper box size) * pixels in slice/ total pixels (= fraction of box the slice uses)
            # slice.indices(arraysize) gets the arguments (start, stop, step) of slice (using the array size to set defaults)
            self.tempdict = {ion: np.log10(np.array([\
                np.sum(self.nion[ion][:,slc])*self.slicelength*cu.c.cm_per_mpc/(self.redshift+1.) *\
                (slc.indices(self.numspecpix)[1] - slc.indices(self.numspecpix)[0])/slc.indices(self.numspecpix)[2] *1./self.numspecpix\
                for slc in slicelist]))  for ion in dions}
            if len(slicelist) == slices +1: # first and last entries are two parts of the same slice -> add them
                self.tempdict = {ion: np.array(list(self.tempdict[ion][:,1:-2]) + [np.log10(10**self.tempdict[ion][:,0] + 10**self.tempdict[ion][:,-1])]) for ion in dions}
            elif len(slicelist) != slices:
                print('Error in slice list generation: %s for %i slices, %f offset'%(str(slicelist),slices,offset))

            if (slices,offset) in self.coldens.keys():
                self.coldens[(slices,offset)].update(self.tempdict)
            else:
                self.coldens[(slices,offset)] = self.tempdict

            
    def getEW(self, dions='all',slices = 1, offset = 0.):
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions,str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        #print(dions)
        for ion in dions:
            if ion not in self.spectra.keys():
                self.getspectra([ion])
            if ion not in lambda_rest.keys():
                ionk = ion + 'major'
            else:
                ionk = ion
            if slices == 1:
                # EW = \int dlamdba (1-flux) = (Delta lambda) - (Delta lambda)/N * sum_i=1^N F_normalised(i)  
                self.EW[ion] = 1. - np.sum(self.spectra[ion], axis=1) / float(self.spectra[ion].shape[1])
                self.EW[ion] *= self.deltaredshift * lambda_rest[ionk] # convert absorbed flux fraction to EW
                # convert to rest-frame EW
                self.EW[ion] *= 1. / (self.redshift + 1.)
            else: 
                slicelist = self.getslices(slices,offset=offset,posvals=None, posvalperiod = None)
                # EW = \int dlamdba (1-flux) = (Delta lambda) - (Delta lambda)/N * sum_i=1^N F_normalised(i)  
                # this gets Npix_slice/Npix_total *( 1 - average flux loss per pixel)
                # later multiplication by total Delta lambda recovers the correct slice EW 
                self.temp = np.array([\
                    float( self.spectra[ion][:,slc].shape[1] - np.sum(self.spectra[ion][:,slc],axis=1) )/float(self.spectra[ion].shape[1])\
                    for slc in slicelist ])
               
                if len(slicelist) == slices +1: # first and last entries are two parts of the same slice -> add them
                    self.temp = np.array( list(self.temp[:,1:-2]) + [self.temp[:,0] + self.temp[:,-1]] )

                self.temp *= self.deltaredshift*lambda_rest[ion] # convert absorbed flux fraction to EW
                # convert to rest-frame EW
                self.temp /= (self.redshift+1.)
                if (slices,offset) in self.EW.keys():
                    self.EW[(slices,offset)].update({ion,self.temp})
                else:
                    self.EW[(slices,offset)] = {ion,self.temp}

    def getslices(self, slices,offset=0.,posvals=None, posvalperiod = None):
        '''
        Returns list of slice objects to most closely retrieve a slicing into
          <slices (int)> slices
        offset is in slice depth units (int), e.g. slices = 2, offset = 0.5 
          gives 2 slices centered at the edge of the box and at half the box

        when offset != 0., the list is returned with <slices> + 1 slice 
          instances, where first and last are understood to be one box slice 
          cut in two
        '''
        
        # make sure there is at most one slice (part) before offset, and offset > 0
        offset = offset%1.
        
        self.sliceleftedges = np.round(float(self.numspecpix) * (np.arange(slices)+offset)/float(slices),0).astype(int,copy=False)
        outlist = [slice(self.sliceleftedges[i],self.sliceleftedges[i+1],None) for i in range(slices-1)]
        outlist += [slice(self.sliceleftedges[-1],self.numspecpix,None)]
        if offset != 0.:
            outlist = [slice(0,self.sliceleftedges[0],None)] + outlist
        return outlist


    def getnion(self, dions = 'all',realspace=True): # ion number density in cm^3 in each pixel: 
        if dions == 'all':
            dions = self.ions
        if realspace:
            try:
                self.nion.update({ion: np.array([self.specfile['Spectrum%i/%s/RealSpaceNionWeighted/NIon_CM3'%(specnum,ion)][()] for specnum in range(self.numspecs)]) for ion in dions})
            except KeyError: # 'frugal' projection without nion-weighted quantities -> derive nion from tau, using inverse of calculation in specwizard
                print('Warning: calculating nion from tau (velocity space) in stead of getting real space values.')
                realspace=False
        if not realspace: 
            for ion in dions:
                self.tauspec = np.array([np.array(self.specfile['Spectrum%i/%s/OpticalDepth'%(specnum,ion)]) for specnum in range(self.numspecs)])
                uion = ion
                if ion == 'o8': # only the highest fosc line is used in the specwizard tau calculation -> need to use the same properties to 
                                # convert back to the correct Nion 
                    uion= 'o8major'
                self.nion.update({ion: Nion_from_tau(self.tauspec,uion)})


    def getquantity(self,name,cat,dions='all'):
        '''
        reads the quantity <name>, in the weighting category <cat>, into
        the appropriate dictionary for the ions <dions> (if applicable)
        '''
        if cat not in self.dataoptions.keys():
            print('Cat options are %s, not %s. No values retrieved.'%(str(self.dataoptions.keys()),cat))
            return
        elif name not in self.dataoptions[cat].keys():
            print('Name options are %s, not %s. No values retrieved.'%(str(self.dataoptions[cat].keys()),name)) 
            return

        elif cat == 'posmassw':
            if name == 'nion':
                self.getnion(dions=dions)
            else:
                self.posmassw.update({name : np.array([np.array(self.specfile['Spectrum%i/%s'%(specnum,self.dataoptions[cat][name])]) for specnum in range(self.numspecs)])})
        elif cat == 'ion':
            if name == 'flux': # we have a function for this one
                self.getspectra(dions=dions) 
            elif name == 'logcoldens': # and for this one
                self.getcoldens(dions=dions)
            else: # name == 'tau'
                self.tau.update({ion: np.array([np.array(self.specfile['Spectrum%i/%s/%s'%(specnum,ion,self.dataoptions[cat][name])]) for specnum in range(self.numspecs)]) for ion in dions})
        else:
            if cat == 'posionw':
                self.basedict = self.posionw
            elif cat == 'veltauw':
                self.basedict = self.veltauw
            else:
                print('%s is not a valis cat option. No values retrieved.'%cat)
                return
            if name not in self.basedict.keys():
                self.basedict[name] = {}
            for ion in dions:      
                self.basedict[name].update({ion : np.array([np.array(self.specfile['Spectrum%i/%s/%s'%(specnum,ion,self.dataoptions[cat][name])]) for specnum in range(self.numspecs)]) for ion in dions})
            del self.basedict           



    def getall(self, dions='all'):
        self.getspectra(dions=dions)
        self.getcoldens(dions=dions)
        self.getEW(dions=dions)
        #self.getnion(dions=dions)

    # corrects o8 spectrum to what you would get from using the o8 doublet with both f_osc in stead of just the larger f_osc component
    # just writes over any previous spectra -> alos produces the correct o8 spectrum
    # correct EW if that was present
    def correct_o8(self):
        # get o8 optical depth spectrum
        self.tauspec_o8 = np.array([np.array(self.specfile['Spectrum%i/o8/OpticalDepth'%specnum]) for specnum in range(self.numspecs)])
        # get pixel velocity along the los (average to reduce single-pixel fp errors); velocities start at zero
        self.velperpix_kmps = np.average(np.diff(np.array(self.specfile['VHubble_KMpS'])))
        # redshift between o8major and o8minor at z=0: measured velocity difference = total redshift * light speed * (km/s / cm/s)
        self.lambdamajz = lambda_rest['o8major'] #* (1.+self.redshift)
        self.lambdaminz = lambda_rest['o8minor'] #* (1.+self.redshift)    
        self.veldiff_at_z_kmps = (self.lambdaminz-self.lambdamajz)/lambda_rest['o8major'] * cu.c.c *1.e-5 
        # what to shift the o8major array by after rescaling to get to o8minor velocity
        self.pixshift_majtomin = int(round(self.veldiff_at_z_kmps/self.velperpix_kmps,0))

        #from Lan & Fukugita 2017: linear EW  = f lambda^2/1.13e20 * Nion; lin. approx for tau is same, but fot tau it is exact
        # -> rescale tau for minor contriution to tau_minor = tau_major * (f_min * lambda_min**2)/(f_maj * lambda_maj**2)   
        self.tauspec_o8_minor = self.tauspec_o8 * (fosc['o8minor'] * lambda_rest['o8minor']**2)/(fosc['o8major'] * lambda_rest['o8major']**2)
        # shift the absorption by the appropriate number of pixels for the wavelength difference 
        self.tauspec_o8_minor = np.roll(self.tauspec_o8_minor,self.pixshift_majtomin,axis=1) # 0 axis -> spectrum number, 1 axis -> spectrum for each sightline
        #optical depths add up
        self.tauspec_o8 += self.tauspec_o8_minor

        self.spectra[u'o8'] = np.exp(-1*self.tauspec_o8)
        self.tauspec_o8
        self.tauspec_o8_minor
        
        # correct corresponding EW: redo calculation with corrected spectrum
        if 'o8' in self.EW.keys():
            self.getEW(dions = ['o8'])




class Coldensmap:
    '''
    slicecen and sidelength in cMpc, internally converted to gadget units
    periodic: if True, assumes all positions are modulo side length;
              if False, only gets matching values, etc. for sightlines in the 
              range. Nearmatch values will still be periodic, so these may be 
              off quite a bit at the edges
    '''
    def __init__(self,filename, slicecen, sidelength, red = 1, periodic = False): 
        if isinstance(filename,str): # actual filename -> load the file
            if not '/' in filename:
                self.filename = ndir + filename
            else:
                self.filename = filename
            if red == 1 :
                self.pixvals = np.load(self.filename)['arr_0']
            else:
                self.pixvals = lnp.imreduce(np.load(self.filename)['arr_0'], red)
            self.slicename = filename.split('/')[-1]
            self.comps = self.slicename[:-4].split('_')
            self.pixind = np.where(np.array(['pix' in comp for comp in self.comps]))[0][0]
            self.numpix = int(self.comps[self.pixind][:-3])/red
            
 
        elif isinstance(filename,np.ndarray): # array is already loaded -> use loaded file
            self.filename = None
            if red == 1 :
                self.pixvals = filename
            else:
                self.pixvals = lnp.imreduce(filename, red)
            self.numpix = self.pixvals.shape[0] #assumed square image

        self.periodic = periodic

        print('File should contain %i pixels per side'%self.numpix)
        self.slicecen = np.array(slicecen)*hcosm
        self.sidelength = sidelength*hcosm

    def getpixindsvals(self,specout, periodic=None):
        if periodic is None:
            periodic = self.periodic
        # choose closest to pixel centre
        if isinstance(specout,Specout):
            self.pixinds = np.round((specout.positions - self.slicecen + self.sidelength/2.)/self.sidelength * float(self.numpix) -0.5,0).astype(int)
        else: # iterable of specouts assumed
            self.pixinds = []
            for spec in specout:
                self.pixinds += list(np.round((spec.positions - self.slicecen + self.sidelength/2.)/self.sidelength * float(self.numpix) -0.5,0).astype(int))
            self.pixinds = np.array(self.pixinds)

        # 8 pixels around the closest
        self.pixinds_extra = np.zeros((self.pixinds.shape[0],8,2),dtype=int)
        self.pixinds_extra[:,0,:] = self.pixinds + np.array([1,0])
        self.pixinds_extra[:,1,:] = self.pixinds + np.array([0,1])
        self.pixinds_extra[:,2,:] = self.pixinds + np.array([-1,0])
        self.pixinds_extra[:,3,:] = self.pixinds + np.array([0,-1])
        
        self.pixinds_extra[:,4,:] = self.pixinds + np.array([1,1])
        self.pixinds_extra[:,5,:] = self.pixinds + np.array([1,-1])
        self.pixinds_extra[:,6,:] = self.pixinds + np.array([-1,1])
        self.pixinds_extra[:,7,:] = self.pixinds + np.array([-1,-1])        
        
        # Will prevent errors even if outside box bounds!
        if periodic:
            self.pixinds %= self.numpix
            self.pixinds_extra %= self.numpix
            self.mask = slice(None,None,None)
            self.numsel = self.pixinds.shape[0]
        else: # x and y pixel indices must be in range(0,numpix)
            self.mask =  np.all(self.pixinds >= 0, axis = 1)
            self.mask &= np.all(self.pixinds < self.numpix, axis = 1)
            self.numsel = np.sum(self.mask)

        self.coldens_match = self.pixvals[list(self.pixinds[self.mask,:].T)]
        self.coldens_nearmatch = np.zeros((self.numsel,8))
        for i in range(8):
            self.coldens_nearmatch[:,i] = self.pixvals[list(self.pixinds_extra[self.mask][:,i,:].T)]

    def getpixregions(self,shape,specout,periodic=None): 
        if periodic is None:
            periodic = self.periodic
        if isinstance(specout,Specout):
            self.numsightlines = specout.numspecs
            self.scens = specout.positions
        else: # iterable of specouts assumed
            self.numsightlines = 0
            self.scens = []
            for spec in specout:
                self.numsightlines += spec.numspecs
                self.scens += list(spec.positions)
            self.scens = np.array(self.scens)
        # initialise to invalid value; if region overlap non-periodic edges, this seems like a sensible default
        self.pixregions = np.empty((self.numsightlines,shape[0],shape[1]),dtype=np.float32)
        self.pixregions[...] = np.NaN        

        # find appropriate pixel ranges; center on edge if even number of pixels, pixel middle if odd
        self.lengthperpixel = self.sidelength/float(self.numpix)
        self.x0y0 = self.slicecen - self.sidelength/2. # lower left coordinates of the [0,0] pixel
        self.scens_pixcoords = (self.scens - self.x0y0)/self.lengthperpixel # sightlines centre in pixel index coordinates

        shape = np.array(shape)
        self.pix00 = np.round(self.scens_pixcoords - 0.5*(shape%2),0).astype(np.int) - shape/2 # pixel index coordinates -> nearest pixel index
        self.pixranges = np.empty((self.numsightlines,4),dtype=np.int) # fill in with pixel index x0, x1, y0, y1
        self.extents = np.empty((self.numsightlines,4),dtype=np.float) # fill in with physical coordinates x0, x1, y0, y1

        # converted to int by setting pixranges dtype to int above
        self.pixranges[:,0] = self.pix00[:,0]
        self.pixranges[:,1] = self.pix00[:,0] + int(shape[0])
        self.pixranges[:,2] = self.pix00[:,1]
        self.pixranges[:,3] = self.pix00[:,1] + int(shape[1]) 
        self.sels = np.array([(slice(self.pixranges[i,0],self.pixranges[i,1],None),slice(self.pixranges[i,2],self.pixranges[i,3],None)) for i in range(self.numsightlines)]) # object array won't be very efficient, but allows nice slicing for periodic case      
        self.extents[:,0:2] = self.pixranges[:,0:2]*self.lengthperpixel + self.x0y0[0]          
        self.extents[:,2:] = self.pixranges[:,2:]*self.lengthperpixel + self.x0y0[1]

        self.pixregions =  np.array([ self.pixvals[tuple(self.sels[i])] for i in range(self.numsightlines) ]) 

        # find problematic regions (overlap edges), and do shift image and selections as appropriate;
        # similar to what I used in coldens_rdist.py
        # do not modify pixranges or extents (net), since these may be needed to refind the location
        # to cover corners and full sides: shift x, redo regions (1), shift x back -> shift y, redo regions (2) -> shift x, redo regions (3), shift x,y back
        # +---+---------+---+
        # |1,3|    2    |1,3|
        # +---+---------+---+
        # | 1 |         | 1 |
        # |-4-|         |-4-|
        # |   |         |   |
        # +---+---------+---+
        # |1,3|    2    |1,3|
        # +---+---------+---+
        # if the x shift were only done after the y shift, the regions around -4- would be on the edge for the y and xy-shifts 
        # so we really need all 3 shifts (x,y,xy) to get every region away from the edge in some configuration

        # if the y shift is only done after the x shift       
        if periodic: 
            # np.where -> (np.array[ <selected indices> ],)
            # reset these arrays to the right shape
            self.hasxoverlap = np.where(np.any([self.pixranges[:,0]<0,self.pixranges[:,1]>= self.numpix],axis=0))
            self.hasyoverlap = np.where(np.any([self.pixranges[:,2]<0,self.pixranges[:,3]>= self.numpix],axis=0)) 
            self.pixregions[self.hasxoverlap] =  np.ones(tuple(shape))*np.NaN
            self.pixregions[self.hasyoverlap] =  np.ones(tuple(shape))*np.NaN
            ## shift x
            # region overlaps left or right edge -> region/sightline index in hasxoverlap
            self.xshift = self.numpix/2 # shift by half the box -> only way too large regions should be a problem in both shifts

            # shift pixvals, pixranges by half the box (x)
            self.pixvals = np.roll(self.pixvals,self.xshift,axis=0)
            self.pixranges[:,0:2] += self.xshift
            self.pixranges %= self.numpix

            # redo region selection for regions with x overlap
            self.sels[self.hasxoverlap] = np.array([(slice(self.pixranges[i,0],self.pixranges[i,1],None),slice(self.pixranges[i,2],self.pixranges[i,3],None)) for i in self.hasxoverlap[0]])
            self.pixregions[self.hasxoverlap] =  [self.pixvals[tuple(self.sels[i])] for i in self.hasxoverlap[0]] 

            ## shift x back
            self.pixvals = np.roll(self.pixvals,-1*self.xshift,axis=0)
            self.pixranges[:,0:2] -= self.xshift
            self.pixranges %= self.numpix

            ## repeat for y 
            # region overlaps left or right edge -> region/sightline index in hasxoverlap
            self.yshift = self.numpix/2 # shift by half the box -> only way too large regions should be a problem in both shifts

            # shift pixvals, pixranges by half the box (y)
            self.pixvals = np.roll(self.pixvals,self.yshift,axis=1)
            self.pixranges[:,2:4] += self.yshift
            self.pixranges %= self.numpix

            # redo region selection for regions with x overlap
            self.sels[self.hasyoverlap] = np.array([(slice(self.pixranges[i,0],self.pixranges[i,1],None),slice(self.pixranges[i,2],self.pixranges[i,3],None)) for i in self.hasyoverlap[0]])
            self.pixregions[self.hasyoverlap] =  [self.pixvals[tuple(self.sels[i])] for i in self.hasyoverlap[0]] 

            ## shift x again; only redo where needed, or some new corners will be wrong again
            self.hasxyoverlap = (np.array(list(set(self.hasyoverlap[0]) & set(self.hasxoverlap[0]))),) # looking for corner elements -> x and y overlaps
            # region overlaps left or right edge -> region/sightline index in hasxoverlap
            self.xshift = self.numpix/2 # shift by half the box -> only way too large regions should be a problem in both shifts

            # shift pixvals, pixranges by half the box (x)
            self.pixvals = np.roll(self.pixvals,self.xshift,axis=0)
            self.pixranges[:,0:2] += self.xshift
            self.pixranges %= self.numpix

            # redo region selection for regions with x overlap
            self.sels[self.hasxyoverlap] = np.array([(slice(self.pixranges[i,0],self.pixranges[i,1],None),slice(self.pixranges[i,2],self.pixranges[i,3],None)) for i in self.hasxyoverlap[0]])
            self.pixregions[self.hasxyoverlap] = [self.pixvals[tuple(self.sels[i])] for i in self.hasxyoverlap[0]] 


            ## shift x,y back to original settings (x,y)
            self.pixvals = np.roll(self.pixvals,-1*self.xshift,axis=0)
            self.pixvals = np.roll(self.pixvals,-1*self.yshift,axis=1)
            self.pixranges[:,0:2] -= self.xshift
            self.pixranges[:,2:4] -= self.yshift
            self.pixranges %= self.numpix

            del self.hasxoverlap
            del self.hasyoverlap
            del self.hasxyoverlap

        del self.sels #confusing mess of selections in shifted and non-shifted coordinates now, 

class Coldensmapslices:
    '''
    Basically just as set of coldensmaps for each slice,
    with wrappers for the Coldensmap functions
    so far, only the __init__ properties of the slices need to match, but that may change
    meant to combine with sliced version of EW and column density values in Specout 
    '''       
    def __init__(self,filename, fills, slicecen, sidelength, red = 1, periodic = False):
        if isinstance(filename,str):
            if '/' not in filename:
                filename = ndir + filename
            self.mapdict = {}
            for fill in fills:
                 self.mapdict.update({fill: Coldensmap(filename%fill, slicecen, sidelength, red = red, periodic = periodic)})
            self.numpix = self.mapdict[fills[0]].numpix        
            self.fills = fills
        else:
            print('Only filenames (string) are currently implemented as an option.')
            return None

    def getpixindsvals(self,specout,periodic=None):
         for fill in self.fills:
             print('Doing %s'%fill)
             self.mapdict[fill].getpixindsvals(specout,periodic=periodic)

    def getpixregions(self,shape,specout,periodic=None):
         for fill in self.fills:
             print('Doing %s'%fill)
             self.mapdict[fill].getpixregions(shape,specout,periodic=periodic) 
        
#test1   = Specout('spec_L0100N1504_o7_test1-los_test1.snap_027_z000p101.0.hdf5')
#test2_2 = Specout('spec_L0100N1504_o7_test2-los_test2_run2.snap_027_z000p101.0.hdf5') 
#test3_2 = Specout('spec_L0100N1504_o7_test3-los_test3_run2.snap_027_z000p101.0.hdf5')

#map2 = Coldensmap('coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_1000pix_6.25slice_zcen-sum_x40.83125-pm0.3125_y2.675-pm0.3125_z-projection_T4EOS_totalbox.npz',np.array([40.83125,2.675]),0.3125)
#map3 = Coldensmap('coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_1000pix_6.25slice_zcen-sum_x2.246875-pm0.3125_y73.7875-pm0.3125_z-projection_T4EOS_totalbox.npz',np.array([2.246875,73.7875]),0.3125)



## get the best-fit b parameter for some (set of) specouts 
def bparfit(specouts, ions='all', EWlog=True, coldenslog=True, cbounds=(None,None), spcm_cd_sp=False, spcm_cd_mp=False, **kwargs):
    '''
    b in cm/s throughout
    fit the earlier linflatcurveofgrowth to specouts for the best-fit  parameter
    spcm_cd_sp and spcm_cd_mp: input is/are Spcm, not specout, use specwizard or 2d map column densities
    '''
    # kwargs: e.g. bounds for scipy.optimize.curve_fit
    out = {}
    if ions == 'all':
        try:
            ions = specouts.ions
        except AttributeError: # it was a list of specouts
            ions = specouts[0].ions

    for ion in ions:
        if not hasattr(specouts, '__len__'): # we've only got one
            if not spcm_cd_sp and not spcm_cd_mp:
                xs = specouts.coldens[ion]
                ys = specouts.EW[ion]
            elif spcm_cd_sp:
                xs = specouts.cd_sp
                ys = specouts.EW
            else:
                xs = specouts.cd_mp
                ys = specouts.EW           

        else: # Spcm not yet implemented
            # find how many spetra we're dealing with, in each specout
            numspecs = [specout.numspecs for specout in specouts]
            numspecstot = sum(numspecs)
            # get start/stop indices for each spectrum in final total array
            specstarts = [sum(numspecs[:i]) for i in range(len(numspecs))]
            specstarts += [numspecstot]
            # set up NaN arrays -> should be clear if something has gone wrong filling them
            xs = np.ones(numspecstot)*np.NaN
            ys = np.ones(numspecstot)*np.NaN
            # fill total x/y arrays with values form each specout
            for i in range(len(specouts)):
                xs[specstarts[i]:specstarts[i+1]] = specouts[i].coldens[ion]
                ys[specstarts[i]:specstarts[i+1]] = specouts[i].EW[ion]

        # depending on whether or not to fit in log space, the fit function and input values need to be modified  
        if not coldenslog:
            xs = 10**xs
        if EWlog:
            ys = np.log10(ys)
     
        if not coldenslog and not EWlog:
            def fitfunc(x,b):
               return linflatcurveofgrowth_inv(x,b,ion)    
        elif coldenslog and not EWlog:
            def fitfunc(x,b):                  
                return linflatcurveofgrowth_inv(10**x,b,ion)
        elif not coldenslog and EWlog:
            def fitfunc(x,b):
                return np.log10(linflatcurveofgrowth_inv(x,b,ion))
        elif coldenslog and EWlog:
            def fitfunc(x,b):
                return np.log10(linflatcurveofgrowth_inv(10**x,b,ion))
        # inital guess from visual fit:  ~ 100 km/s. 0 < b < speed of light is physically required  
        # outputs optimal, covariance
        out[ion] = sp.optimize.curve_fit(fitfunc, xs, ys, p0 = 100*1e5,bounds=(np.array([0]),np.array([3e5*1e5])),**kwargs)        

    return out

## downsampling tests:
# since random-ish [::9] downsampling seems to work well (~ promille errors), and even [::81] usually gives <~ 10% errors
# try more sophisticated add and average downsampling
def downsample(array,outpixels,average=True,axis=0):
    '''
    interpolates array to a smaller array of size <outpixels>, along <axis>
    axis may be an iterable of axes
    average: True -> use average of input pixels for the new pixels, False -> use the sum
      in both cases, the portion assigned to each new pixel is old-new overlap/old size
    regular grid spacing is assumed (otherwise, see combine_specwiz_makemaps.py for a more general routine)
    
    calls the c function in gridcoarser.c, using make_maps_opts_locs.py for where to find it 
    '''
    inshape  = np.array(array.shape)
    outshape = list(inshape)
    outshape = np.array(outshape)
    outshape[axis] = outpixels
    outarray = np.zeros(outshape,dtype = np.float32)

    cfile = ol.c_gridcoarser

    acfile = ct.CDLL(cfile)
    redfunc = acfile.reducearray

    redfunc.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(np.prod(inshape),)),\
                           np.ctypeslib.ndpointer( dtype=ct.c_int, shape=(len(inshape),) ),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=tuple(outshape)),
                           np.ctypeslib.ndpointer(dtype=ct.c_int, shape=(len(outshape),) ), \
                           ct.POINTER(ct.c_int),\
                           ct.POINTER(ct.c_int),\
                           ct.POINTER(ct.c_int),\
                        ]

    # argument conversion. input array as flattend, 1D array

    res = redfunc((array.astype(np.float32)).flatten(),\
               inshape.astype(np.int32),\
               outarray,\
               outshape.astype(np.int32),\
               ct.byref(ct.c_int(len(inshape))),\
               ct.byref(ct.c_int(0)),\
               ct.byref(ct.c_int(average))\
              )
    print(outarray.shape)
    return np.reshape(outarray,outshape)



# specwizard vs. make_maps values: logN-logN plot (relative=False) or logN-ratio plot (relative=True) 
def coldenscomp(specout,coldensmap,ion,title = None, xlabel = True, ylabel = True, legend = True, ax = None, relative = False, alpha=1.):
    if ax is None:
        ax = plt.subplot(111)
        
    if not relative:
        ax.plot(specout.coldens[ion],specout.coldens[ion],linestyle='dashed',color='gray',label = 'equal',alpha=alpha)
        ax.scatter(specout.coldens[ion][coldensmap.mask],coldensmap.coldens_nearmatch[:,4],color='red',label='diagonal neigbour pixels',alpha=alpha)
        for i in range(5,8,1):
            ax.scatter(specout.coldens[ion][coldensmap.mask],coldensmap.coldens_nearmatch[:,i],color='red',label=None,alpha=alpha)
    
        ax.scatter(specout.coldens[ion][coldensmap.mask],coldensmap.coldens_nearmatch[:,0],color='green',label='neigbour pixels',alpha=alpha)
        for i in range(1,4,1):
            ax.scatter(specout.coldens[ion][coldensmap.mask],coldensmap.coldens_nearmatch[:,i],color='green',label=None,alpha=alpha)
        ax.scatter(specout.coldens[ion][coldensmap.mask],coldensmap.coldens_match,color='blue',label='nearest pixel',alpha=alpha)
    else:
        ax.plot(specout.coldens[ion][coldensmap.mask],np.ones(len(specout.coldens[ion][coldensmap.mask])),linestyle='dashed',color='gray',label = 'equal')
        ax.scatter(specout.coldens[ion][coldensmap.mask],10**coldensmap.coldens_nearmatch[:,4]/10**specout.coldens[ion][coldensmap.mask],color='red',label='diagonal neigbour pixels',alpha=alpha)
        for i in range(5,8,1):
            ax.scatter(specout.coldens[ion][coldensmap.mask],10**coldensmap.coldens_nearmatch[:,i]/10**specout.coldens[ion][coldensmap.mask],color='red',label=None,alpha=alpha)
    
        ax.scatter(specout.coldens[ion][coldensmap.mask],10**coldensmap.coldens_nearmatch[:,0]/10**specout.coldens[ion][coldensmap.mask],color='green',label='neigbour pixels',alpha=alpha)
        for i in range(1,4,1):
            ax.scatter(specout.coldens[ion][coldensmap.mask],10**coldensmap.coldens_nearmatch[:,i]/10**specout.coldens[ion][coldensmap.mask],color='green',label=None,alpha=alpha)
        ax.scatter(specout.coldens[ion][coldensmap.mask],10**coldensmap.coldens_match/(10**specout.coldens[ion][coldensmap.mask]),color='blue',label='nearest pixel',alpha=alpha)
        #ax.set_yscale('log')

    if title is None:
        ax.set_title(r'%s column density $[\log_{10} \mathrm{cm}^{-2}]$'%ion)
    else:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel('Specwizard value')
    if ylabel and not relative:
        ax.set_ylabel('Projection value')
    if ylabel and relative:
        ax.set_ylabel('Proj. value / Specwizard value')
    if legend:
        ax.legend()

# specwizard vs. make_maps values: logN-logN plots (relative=False) or logN-ratio plot (relative=True) 
# uses coldenscomp
# varies used specout by row, coldensmap by column
# outer-level list -> rows and columns (subplots)
# inner-level list -> different datasets in the same plot (length in specouts and coldensmaps must match here; no labels for the different datasets)
#                     assumed to be different resolutions in subplot titles
def coldenscomps(specouts,coldensmaps,ion, specress = ['standard', 'x3', 'x9'], mapress = ['standard', 'x3', 'x9'],relative=False,title=None):
    nrows = len(coldensmaps)
    ncols = len(specouts)

    for row in np.arange(nrows,dtype=np.int):
        for col in np.arange(ncols,dtype=np.int):
            ax = plt.subplot(ncols,nrows,1+row+col*nrows)
            if col == ncols-1:
                xlabel = True
            else:
                xlabel = False
            if row == 0:
                ylabel = True
            else:
                ylabel = False
            if title is None: # use default title
                title = 'spec. %s res., proj. %s res.'%(specress[col],mapress[row])
   
            if row ==0 and col == 0:
                legend = True
            else:
                legend = False
            if not isinstance(specouts[col],(list,np.ndarray,tuple)):           
                coldenscomp(specouts[col],coldensmaps[row],ion,title = title, xlabel = xlabel, ylabel = ylabel, legend = legend, ax = ax,relative=relative)
            else: # specouts, coldensmaps are lists of lists: plot all second-level lists together
                  # specouts and coldensmaps should have matching order and sightlines
                for i in range(len(specouts[col])):
                    if i == 0:
                        stitle  = title
                        sxlabel = xlabel
                        sylabel = ylabel
                        slegend = legend
                    else:
                        stitle  = None
                        sxlabel = None
                        sylabel = None
                        slegend = False
                    coldenscomp(specouts[col][i],coldensmaps[row][i],ion,title = stitle, xlabel = sxlabel, ylabel = sylabel, legend = slegend, ax = ax,relative=relative)
    plt.show()

# compare specouts for the same sightlines, relative to the highest-res gi and cubic spline kernel plots
# specouts assumed to be in order of increasing resolution 
# can also do EWs in stead of column densities (ews = True)
# separate: do gi/cuic in separate plots, and a highest-res comparison in a third
def specconv(specs, specs_gi, ion, speclabels, specgilabels, colors = ['red','orange','green','blue','purple'],ews=False,separate=False):
    numnom = len(specs)
    numgi =  len(specs_gi)

    if separate:
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    else:
        fig, (ax1,ax2) = plt.subplots(1,2, sharex=True, sharey='row')

    if ews:
        norm = (specs[numnom-1]).EW[ion]
    else:
        norm = (specs[numnom-1]).coldens[ion]
    ax1.set_ylabel('ratio')
    if ews:
        ax1.set_xlabel(r'$EW \, [\mathrm{\AA}]$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    else:
        ax1.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$')
    ax1.plot(norm,norm/norm,color = colors[numnom-1], label = speclabels[numnom-1])
    if ews:
        for i in range(numnom-1):
            ax1.scatter(norm,(specs[i]).EW[ion]/norm,color = colors[i], label = speclabels[i],marker = 'o', s=30./(i+1))    
        if not separate:
            for i in range(numgi):
                ax1.scatter(norm,(specs_gi[i]).EW[ion]/norm,color = colors[i], label = specgilabels[i],marker = '*', s=30./(i+1) )
    else:
        for i in range(numnom-1):
            ax1.scatter(norm,(specs[i]).coldens[ion]/norm,color = colors[i], label = speclabels[i],marker = 'o', s=30./(i+1))    
        if not separate:
            for i in range(numgi):
                ax1.scatter(norm,(specs_gi[i]).coldens[ion]/norm,color = colors[i], label = specgilabels[i],marker = '*', s=30./(i+1) )
    ax1.legend()
    ax1.set_title('convergence rel. to highest-res. cubic spline kernel')
    
    if ews:
        norm = (specs_gi[numgi-1]).EW[ion]
    else:
        norm = (specs_gi[numgi-1]).coldens[ion]
    if ews:
        ax2.set_xlabel(r'$EW \, [\mathrm{\AA}]$')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        ax2.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$')
    ax2.plot(norm,norm/norm,color = colors[numgi-1], label = specgilabels[numgi-1])
    if ews:
        if not separate:
            for i in range(numnom):
                ax2.scatter(norm,(specs[i]).EW[ion]/norm,color = colors[i], label = speclabels[i],marker = 'o', s=30./(i+1))    
        for i in range(numgi-1):
            ax2.scatter(norm,(specs_gi[i]).EW[ion]/norm,color = colors[i], label = specgilabels[i],marker = '*', s=30./(i+1) )
    else:
        if not separate:
            for i in range(numnom):
                ax2.scatter(norm,(specs[i]).coldens[ion]/norm,color = colors[i], label = speclabels[i],marker = 'o',s=30./(i+1))    
        for i in range(numgi-1):
            ax2.scatter(norm,(specs_gi[i]).coldens[ion]/norm,color = colors[i], label = specgilabels[i],marker = '*',s=30./(i+1))
    ax2.legend()
    ax2.set_title('convergence rel. to highest-res. gaussian integrated kernel')

    if separate:
        ax3.set_title('highest-res. cubic vs. gaussian-integrated kernels')

        if ews:
            minmax = [min(np.min(specs[numnom-1].EW[ion]),np.min(specs_gi[numgi-1].EW[ion])),max(np.max(specs[numnom-1].EW[ion]),np.max(specs_gi[numgi-1].EW[ion]))]
            ax3.plot(minmax,[1,1],color='gray',linestyle='dashed',label='equal' )
            ax3.scatter(specs[numnom-1].EW[ion],specs[numnom-1].EW[ion]/specs_gi[numgi-1].EW[ion])

            ax3.set_xlabel(r'%s EW $[\mathrm{AA}]$'%speclabels[numnom-1])
            ax3.set_ylabel('%s EW / %s EW '%(speclabels[numnom-1],specgilabels[numgi-1]))
            ax3.set_xscale('log')
            ax3.set_yscale('log')
        else:
            minmax = [min(np.min(specs[numnom-1].coldens[ion]),np.min(specs_gi[numgi-1].coldens[ion])),max(np.max(specs[numnom-1].coldens[ion]),np.max(specs_gi[numgi-1].coldens[ion]))]
            ax3.plot(minmax,[0,0],color='gray',linestyle='dashed',label='equal',alpha=0.5)
            ax3.scatter(specs[numnom-1].coldens[ion],specs[numnom-1].coldens[ion] - specs_gi[numgi-1].coldens[ion])
            ax3.set_xlabel(r'%s $\log_{10} N \,[\mathrm{cm}^{-2}]$'%speclabels[numnom-1])
            ax3.set_ylabel(r'$\log_{10}$ N %s / N %s'%(speclabels[numnom-1],specgilabels[numgi-1]))


# coldens vs. EW for the sightlines in specout; reltaive -> relative to linear COG
# specout can be an iterable of specouts; color and label can be as well then
def curveofgrowth(specout,ion,label=None,color='gray',speccolors = ['red','orange','green','blue','red','orange','green'],markers = ['o','o','o','o','*','*','*'],linref=True,relative=False,alpha=1.,legend=True):

    plt.yscale('log')
    if label is None:
        label = 'specwizard curve of growth'

    if isinstance(specout,Specout):
        if relative:
            plt.scatter(specout.coldens[ion],specout.EW[ion]/lingrowthcurve_inv(10**specout.coldens[ion],ion),label = label,s=10,alpha=alpha)
        else:
            plt.scatter(specout.coldens[ion],specout.EW[ion],label = labels,s=10,alpha=alpha)
        xs = specout.EW[ion]

    else: # assume list of specouts
        if markers is None:
            markers = list(('o',)*len(specout))
        if speccolors is None:
            speccolors = ['brown','firebrick','red','orange','gold','lime','green','cyan','blue','blueviolet','purple','magenta']
        xs = []
        if isinstance(label,str) or label is None: # avoid getting the same legend entry for each spectrum unless desired
            labels = list((None,)*len(specout))
            labels[0] = label
        else:
            labels = label
        for i in range(len(specout)):
            if relative:
                plt.scatter(specout[i].coldens[ion],specout[i].EW[ion]/lingrowthcurve_inv(10**specout[i].coldens[ion],ion),label = labels[i], color=speccolors[i], marker=markers[i], s=10, alpha = alpha)
            else:
                plt.scatter(specout[i].coldens[ion],specout[i].EW[ion],label = labels[i], color=speccolors[i], marker=markers[i], s=10, alpha=alpha)
            xs += list(specout[i].coldens[ion])
        xs = np.array(xs)

    if linref:
        if relative:
            plt.plot(xs,np.ones(len(xs)),color=color, label = 'theoretical linear curve of growth')
        else:
            plt.plot(xs,lingrowthcurve_inv(10**xs,ion),color=color, label = 'theoretical linear curve of growth')
    if legend:
        plt.legend()
    if relative:
        plt.ylabel(r'$EW / \mathrm{linear COG}$')
        plt.yscale('log')
    else:
        plt.ylabel(r'$\log_{10} EW\, [\mathrm{\AA}]$')        
    plt.xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$')
    plt.title(r'%s curve of growth: specwizard results, singlet $f_{\mathrm{osc}}$'%ion)



def plotdevdev(spec1,norm1,spec2,norm2,ion,ew1=False,ew2=False,ax=None,xylabels=['',''],reflines = False, **kwargs):
    '''
    Plot deviation vs. deviation: spec1 from norm1 vs. spec2 from norm2
    deviations are relative to norm, plotted on log scales
    ew<i>: is i an equivalent width (otherwise column density)
    '''
    if ax is None: # set up axis, otherwise use the given one
        fig, ax = plt.subplots(1,1)

    ax.set_yscale('log')
    ax.set_xscale('log')

    if ew1:
        xdata = spec1.EW[ion]/norm1.EW[ion]
        ax.set_xlabel(r'%s EW deviation'%xylabels[0])
    else:
        xdata = 10**(spec1.coldens[ion] - norm1.coldens[ion])
        ax.set_xlabel(r'%s $N$ deviation'%xylabels[0])

    if ew2:
        ydata = spec2.EW[ion]/norm2.EW[ion]
        ax.set_ylabel(r'%s EW deviation'%xylabels[1])
    else:
        ydata = 10**(spec2.coldens[ion] - norm2.coldens[ion])
        ax.set_ylabel(r'%s $N$ deviation'%xylabels[1])
   
    minmax = [min(np.min(xdata),np.min(ydata)),max(np.max(xdata),np.max(ydata))] 
     
    ax.scatter(xdata,ydata,**kwargs)
    if reflines:
        # ax.plot(minmax,minmax,color='gray',linestyle='dashed') 
        ax.axhline(1, color='gray',linestyle='solid')
        ax.axvline(1, color='gray',linestyle='solid')

def griddevdev(shape,spec1,norm1,spec2,norm2,ion,ew1=None,ew2=None,xylabels=None,reflines=None,colors=None,alpha=None,label=None):

    fig, axes = plt.subplots(*shape)

    for i in range(shape[0]):
        for j in range(shape[1]):  
            print('%i,%i'%(i, j))
            spec1_sub = spec1[i][j]
            norm1_sub = norm1[i][j]
            spec2_sub = spec2[i][j]
            norm2_sub = norm2[i][j]
            plotdevdev(spec1_sub,norm1_sub,spec2_sub,norm2_sub,ion,ew1=ew1[i][j],ew2=ew2[i][j],ax=axes[i][j],xylabels=xylabels[i][j],reflines = reflines[i][j], color = colors[i][j], alpha = alpha[i][j], label = label[i][j])
            axes[i,j].legend()

def griddevdev_1(sp_i81xstnd,sp_i9xstnd,sp_stnd,sp_9xstnd,sp_i81xstnd_gi,sp_i9xstnd_gi,sp_stnd_gi,sp_9xstnd_gi):
    '''
                          i81x stnd        i9x stnd       stnd
    EW dev   gi vs cubic   
    EW dev vs N dev cubic
    EW dev gi vs N dev cubic  
    '''
    griddevdev((3,3),\
               ((sp_i81xstnd,sp_i9xstnd,sp_stnd),)*3,\
               ((sp_9xstnd,)*3,)*3,\
               ((sp_i81xstnd_gi, sp_i9xstnd_gi, sp_stnd_gi),(sp_i81xstnd,sp_i9xstnd,sp_stnd,),(sp_i81xstnd_gi,sp_i9xstnd_gi,sp_stnd_gi)),\
               ((sp_9xstnd_gi,sp_9xstnd,sp_9xstnd_gi),)*3,\
               'o7',\
               ew1=((True,)*3,(False,)*3,(False,)*3),\
               ew2=((True,)*3,)*3,\
               xylabels=((['cubic','gauss. int.'],)*3,(['cubic','cubic'],)*3,(['cubic','gauss. int.'],)*3),\
               reflines = ((True,)*3,)*3,\
               colors = (('blue','red','green'),)*3,\
               alpha = ((0.5,)*3,)*3,\
               label = (('1/81x stnd / 9x stnd','1/9x stnd / 9x stnd','stnd / 9x stnd'),)*3 )
             
# absorption spectra in specout, labeled with col. dens. from specout 
# multiple ions from one specout instance can be overplotted
# slice selects which spectra to use from specout
def plotspectra(specout,ions,name='spectest.png',slicelines=None,spectraslice=slice(None,None,None),cols=5,rows=10,quantity = 'spectra'):
    #slicefilebase = 'coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen%s_T4SFR.npz'
    #fills = [str(i) for i in np.arange(100./32.,100.,100./16.)]

    depth = 100.
    srows = rows/2    
    fontsize = 11

    if ions == 'all':
         ions = specout.ions

    shape = (rows,cols) # last two shape values are the image dimensions
    figsize = (1.*(shape[0]),2.5*(shape[1]))
    fig, axes = plt.subplots(*shape, sharex=True, sharey='row',figsize=figsize)

    # turn off axis ticks and space between and around subplots
    #for ax in np.ndarray.flatten(axes):
        #ax.minorticks_off()
        #ax.xaxis.set_major_locator(plt.NullLocator()) # turns off x axis ticks
        #ax.yaxis.set_major_locator(plt.NullLocator())   
        #ax.set_aspect(1., adjustable='box-forced')
    fig.subplots_adjust(left=0., bottom=0., right=1., top=0.95, wspace=0., hspace=0.)
    
    textoffset = {ions[i]: 0.2*i for i in range(len(ions))}
    textoffset_horiz = {ions[i]: 0.2*i for i in range(len(ions))}
    colorlist = ['blue','red','green','cyan','magenta','orange','brown','gray','purple']
    linestylelist = ['solid','dashed','dotted','dotted','dotted']
    colors = {ions[i]: colorlist[i] for i in range(len(ions))}
    linestyles = {ions[i]: linestylelist[i] for i in range(len(ions))}
    legendtext = ''

    for ion in ions:
        speclen = specout.spectra[ion].shape[1]
        specx = depth*np.arange(speclen)/float(speclen)
        
    # plots: same colormap, Vmin, Vmax through each column
        for row in range(rows):
            for col in range(cols):
                ax = axes[row,col]
                #ax.patch.set_facecolor(cm.get_cmap(colmaps[col])(0.))
                #img = ax.imshow((ims[row,col]).T,origin='lower', cmap=cm.get_cmap(colmaps[col]), vmin = Vmins[col], vmax=Vmaxs[col],interpolation='nearest')   
                ax.plot(specx,specout.spectra[ion][spectraslice][row*cols + col],color=colors[ion],linestyle = linestyles[ion], label = ion)
                #ax.set_yscale('log')
                if slicelines is not None:
                    for x in (np.arange(slicelines-1) + 1.)*depth/float(slicelines):
                        ax.axvline(x=x,ymin=0.,ymax=1.,color='lightgray',linewidth=1)
                ax.text(0.90,0.05 + textoffset[ion],'%.2f'%specout.coldens[ion][spectraslice][row*cols + col],fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes,color=colors[ion])
                ax.text(0.05,0.05 + textoffset[ion],'%.2e'%specout.EW[ion][spectraslice][row*cols + col],fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax.transAxes,color=colors[ion])
        axes[0,0].text(0.0 + textoffset_horiz[ion],1.,ion,fontsize=fontsize + 1, horizontalalignment = 'left', verticalalignment = 'top', transform=ax.transAxes,color=colors[ion])
        print('%s: %s'%(colors[ion],ion))
        legendtext += ', %s: %s'%(colors[ion],ion)
    #print(legendtext)
    fig.suptitle(r'Specwizard absorption spectra with EW ($\mathrm{\AA}$, left) and column density ($\log_{10} \mathrm{cm}^{-2}$, right)' + legendtext)

    
    plt.savefig(mdir + name,format = 'png',bbox_inches='tight')
    plt.close()


# made to compare grid1c and test10 spectra of the same sightlines; also works for different resolution spectra 
def plotspectra_loscomp(specouts,ion,name='spectest.png',slicelines=None,spectraslice=slice(None,None,None),cols=5,rows=10,quantity = 'spectra',specoutnames = None,subheight=1.,subwidth=2.5):
    #slicefilebase = 'coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen%s_T4SFR.npz'
    #fills = [str(i) for i in np.arange(100./32.,100.,100./16.)]

    depth = 100.
    srows = rows/2    
    fontsize = 11

    if isinstance(specouts, Specout): # if single item in stead of a list
        specouts = [specouts]
    if specoutnames is None:
        specoutnames = ['spec%i'%(i+1) for i in range(len(specouts))]

    shape = (rows,cols) # last two shape values are the image dimensions
    figsize = (subheight*(shape[0]),subwidth*(shape[1]))
    fig, axes = plt.subplots(*shape, sharex=True, sharey='row',figsize=figsize)

    # turn off axis ticks and space between and around subplots
    #for ax in np.ndarray.flatten(axes):
        #ax.minorticks_off()
        #ax.xaxis.set_major_locator(plt.NullLocator()) # turns off x axis ticks
        #ax.yaxis.set_major_locator(plt.NullLocator())   
        #ax.set_aspect(1., adjustable='box-forced')
    fig.subplots_adjust(left=0., bottom=0., right=1., top=0.95, wspace=0., hspace=0.)
    
    textoffset = {specoutnames[i]: 0.2*i for i in range(len(specouts))}
    textoffset_horiz = {specoutnames[i]: 0.2*i for i in range(len(specouts))}
    colorlist = ['blue','red','green','cyan','magenta','orange','brown','gray','purple']
    linestylelist = ['solid','dashed','dashdot','dotted','dotted']
    colors = {specoutnames[i]: colorlist[i] for i in range(len(specouts))}
    linestyles = {specoutnames[i]: linestylelist[i] for i in range(len(specouts))}
    legendtext = ''

    for specind in range(len(specouts)):
        specout = specouts[specind]
        speclen = specout.spectra[ion].shape[1]
        specx = depth*np.arange(speclen)/float(speclen)
        
    # plots: same colormap, Vmin, Vmax through each column
        for row in range(rows):
            for col in range(cols):
                ax = axes[row,col]
                #ax.patch.set_facecolor(cm.get_cmap(colmaps[col])(0.))
                #img = ax.imshow((ims[row,col]).T,origin='lower', cmap=cm.get_cmap(colmaps[col]), vmin = Vmins[col], vmax=Vmaxs[col],interpolation='nearest')   
                ax.plot(specx,specout.spectra[ion][spectraslice][row*cols + col],color=colors[specoutnames[specind]],linestyle = linestyles[specoutnames[specind]], label = specoutnames[specind])
                #ax.set_yscale('log')
                if slicelines is not None:
                    for x in (np.arange(slicelines-1) + 1.)*depth/float(slicelines):
                        ax.axvline(x=x,ymin=0.,ymax=1.,color='lightgray',linewidth=1)
                ax.text(0.90,0.05 + textoffset[specoutnames[specind]],'%.2f'%specout.coldens[ion][spectraslice][row*cols + col],fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes,color=colors[specoutnames[specind]])
                ax.text(0.05,0.05 + textoffset[specoutnames[specind]],'%.2e'%specout.EW[ion][spectraslice][row*cols + col],fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax.transAxes,color=colors[specoutnames[specind]])
        axes[0,0].text(0.0 + textoffset_horiz[specoutnames[specind]],1.,specoutnames[specind],fontsize=fontsize + 1, horizontalalignment = 'left', verticalalignment = 'top', transform=ax.transAxes,color=colors[specoutnames[specind]])
        print('%s: %s'%(colors[specoutnames[specind]],specoutnames[specind]))
        legendtext += ', %s: %s'%(colors[specoutnames[specind]],specoutnames[specind])
    #print(legendtext)
    fig.suptitle(r'Specwizard absorption spectra with EW ($\mathrm{\AA}$, left) and column density ($\log_{10} \mathrm{cm}^{-2}$, right)' + legendtext)

    
    plt.savefig(mdir + name,format = 'png',bbox_inches='tight')
    plt.close()


def plotsightlineinfo(specouts,ion,index,fig = None, axes=None,specoutnames = None,\
                      titletext = '', depthrange = None, plotpix = None, colorlist=None, linestylelist=None, addtext=True):
    '''
    plots a column of: 
    - a plot of the spectrum with column density and EW indicated, as in the many-sightline plots
    - a plot of the ion number density   \
    - a plot of the overdensity           > position space          
    - and a plot of the temperature      /
    '''

    if not hasattr(specouts,'__len__'):
        specouts = [specouts] # make sure specout is a list of specouts

    dotightlayout = False
    if axes is None:
        fig, axes = plt.subplots(4,1,sharex=False,figsize=(12.5,9.)) 
        fig.subplots_adjust(top=0.95,bottom=0.)
        dotightlayout = True
    elif len(axes) < 4:
        print('Length of axes is too small')
        return None
    
    ax1 = axes[0]
    ax2 = axes[1]
    #ax3 = axes[2]
    ax4 = axes[2]
    ax5 = axes[3]

    fontsize = 11

    textoffset = {specoutnames[i]: 0.2*i for i in range(len(specouts))}
    textoffset_horiz = {specoutnames[i]: 0.2*i for i in range(len(specouts))}
    if colorlist is None:
        colorlist = ['blue','red','green','cyan','magenta','orange','brown','gray','purple']
    if linestylelist is None: 
        linestylelist = ['solid','dashed','dashdot','dotted','dotted']
    colors = {specoutnames[i]: colorlist[i] for i in range(len(specouts))}
    linestyles = {specoutnames[i]: linestylelist[i] for i in range(len(specouts))}

    # set up axis scales and labels
    ax1.set_xlabel('V [km/s]', fontsize=fontsize)
    #ax2.set_xlabel('Z [cMpc]', fontsize=fontsize)
    #ax3.set_xlabel('Z [cMpc]', fontsize=fontsize)
    #ax4.set_xlabel('Z [cMpc]', fontsize=fontsize)
    ax5.set_xlabel('Z [cMpc]', fontsize=fontsize)
    
    ax1.set_ylabel('normalised flux', fontsize=fontsize)
    ax2.set_ylabel(r'$n_{\mathrm{%s}}\, [\mathrm{cm^{-3}}]$'%ion, fontsize=fontsize)    
    #ax3.set_ylabel(r'$\delta_{\rho}$, $n_{\mathrm{%s}}$-weighted'%ion, fontsize=fontsize)
    ax4.set_ylabel(r'$T [K]$, $n_{\mathrm{%s}}$-weighted'%ion, fontsize=fontsize)
    ax5.set_ylabel(r'$v_{\mathrm{pec, los}} \, [\mathrm{km}/\mathrm{s}]$, $n_{\mathrm{%s}}$-weighted'%ion, fontsize=fontsize)

    ax2.set_yscale('log')
    #ax3.set_yscale('log')
    ax4.set_yscale('log')

    ax1.tick_params(direction='in')
    ax2.tick_params(direction='in', labelbottom=False)
    #ax3.tick_params(direction='in', labelbottom=False)
    ax4.tick_params(direction='in', labelbottom=False)
    ax5.tick_params(direction='in')

    legendtext = ''
    for specind in range(len(specouts)):
        specout = specouts[specind]
        if plotpix is None:
            speclen = specout.spectra[ion].shape[1]
        else:
            speclen = plotpix
        specx = specout.slicelength*np.arange(speclen)/float(speclen) # x values for spectrum (left edges, position space)
        if depthrange is not None:
            possel = np.all(np.array([specx > depthrange[0], specx < depthrange[1]]),axis=0)
        else:
            possel = slice(None,None,None)
        legendtext += ', %s: %s'%(colors[specoutnames[specind]],specoutnames[specind])

        # spectrum plot
        vlos = np.array(specout.specfile['VHubble_KMpS'])
        spectrum = specout.spectra[ion][index]
        if plotpix is not None:
            vlos = downsample(vlos,plotpix,average=True)
            spectrum = downsample(spectrum,plotpix, average=True)
        
        ax1.plot(vlos,spectrum,color=colors[specoutnames[specind]],linestyle = linestyles[specoutnames[specind]], label = specoutnames[specind])
        if addtext:
            ax1.text(0.90,0.05 + textoffset[specoutnames[specind]],'%.2f'%specout.coldens[ion][index],fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax1.transAxes,color=colors[specoutnames[specind]])
            ax1.text(0.05,0.05 + textoffset[specoutnames[specind]],'%.2e'%specout.EW[ion][index],fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax1.transAxes,color=colors[specoutnames[specind]])

        # nion, T, vpec, Overdensity plots. possel is defined on the reduced array
        nion = np.array(specout.specfile['Spectrum%i/%s/RealSpaceNionWeighted/NIon_CM3'%(index,ion)])
        #rho  = np.array(specout.specfile['Spectrum%i/%s/RealSpaceNionWeighted/OverDensity'%(index,ion)])
        tmp  = np.array(specout.specfile['Spectrum%i/%s/RealSpaceNionWeighted/Temperature_K'%(index,ion)])
        vpec = np.array(specout.specfile['Spectrum%i/%s/RealSpaceNionWeighted/LOSPeculiarVelocity_KMpS'%(index,ion)])
        if plotpix is not None: # include ion number weighting the the weighted quantities
            nion_red = downsample(nion,plotpix,average=True)
            #rho  = downsample(nion*rho,plotpix,average=True)/nion_red
            tmp  = downsample(nion*tmp,plotpix,average=True)/nion_red
            vpec = downsample(nion*vpec,plotpix,average=True)/nion_red
            nion = nion_red

        ax2.plot(specx[possel], nion[possel],color=colors[specoutnames[specind]],linestyle = linestyles[specoutnames[specind]], label = specoutnames[specind])
        #ax3.plot(specx[possel], rho[possel],color=colors[specoutnames[specind]],linestyle = linestyles[specoutnames[specind]], label = specoutnames[specind])
        ax4.plot(specx[possel], tmp[possel],color=colors[specoutnames[specind]],linestyle = linestyles[specoutnames[specind]], label = specoutnames[specind])
        ax5.plot(specx[possel], vpec[possel],color=colors[specoutnames[specind]],linestyle = linestyles[specoutnames[specind]], label = specoutnames[specind])

    if addtext:
        fig.suptitle( titletext + r'with EW ($\mathrm{\AA}$, left) and column density ($\log_{10} \mathrm{cm}^{-2}$, right)' + legendtext)

    # some pixels get zero temperature. Set the minimum shown temperature to 10^2.5 K
    # set the minimum shown ion number density to a fraction of max
    tlim = ax4.get_ylim()
    ax4.set_ylim(10**2.5,tlim[1])

    #overdenslim = ax3.get_ylim()
    #ax3.set_ylim(1e-2,overdenslim[1])

    nionlim = ax2.get_ylim()
    ax2.set_ylim(nionlim[1]/1e5,nionlim[1])

    if dotightlayout:
        fig.tight_layout()



# compares projected column denisties to corresponding centre/edge/corner sightlines from specwizard 
def compvalues_fullslice(specout, coldensmap, ion, name = mdir + 'specwiz_projection_comparions_test3.png',fontsize=11,colmap='viridis', title = 'region comparison test1, index 46',clabel = r'$\log_{10} N_{\mathrm{O VII}}\, [\mathrm{cm}^{-2}]$'):
    
    pixvals = coldensmap.coldens_match
    coldens_specwizard = specout.coldens[ion]
    positions_specwizard = specout.positions

    plot_box = (np.min(positions_specwizard[:,0]),\
                np.max(positions_specwizard[:,0]),\
                np.min(positions_specwizard[:,1]),\
                np.max(positions_specwizard[:,1]) )
    print(str(plot_box))
    pixcen = [np.round(((plot_box[1]+plot_box[0])/2.)/sidelength * float(numpix),0).astype(int), np.round(((plot_box[3]+plot_box[2])/2.)/sidelength * float(numpix),0).astype(int)]
    numpixsel = np.round((plot_box[1]-plot_box[0])/sidelength*numpix,0).astype(int) 
    print('pixcen: %s, numpixsel: %s'%(str(pixcen),str(numpixsel)))

    numpixsel += 2
    plot_box2 = (plot_box[0] - sidelength/numpix, plot_box[1] + sidelength/numpix, plot_box[2] - sidelength/numpix, plot_box[3] + sidelength/numpix)

    selection_proj = ( slice(pixcen[0] - numpixsel/2,pixcen[0] + (numpixsel+1)/2,1) ,slice(pixcen[1] - numpixsel/2,pixcen[1] + (numpixsel+1)/2,1) )
    print(str(selection_proj))
    im_proj = pixvals[selection_proj]

    Vmin = min(np.min(im_proj),np.min(coldens_specwizard))
    Vmax = min(np.max(im_proj),np.max(coldens_specwizard))

    coldens_sw_normed = (coldens_specwizard - Vmin)/(Vmax-Vmin)
      
    fig = plt.figure(figsize = (5.5, 5.)) # large size just as a trick to get higher resolution
    ax = plt.subplot(111)   
    ax.set_xlabel(r'X [cMpc/h]',fontsize=fontsize)
    ax.set_ylabel(r'Y [cMpc/h]',fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize)
    ax.patch.set_facecolor(cm.get_cmap(colmap)(0.)) # sets background color to lowest color map value 

    img = ax.imshow(im_proj.T, extent=plot_box2, origin='lower', cmap=cm.get_cmap(colmap), vmin = Vmin, vmax=Vmax, interpolation='nearest',aspect='auto')
    plt.title(title,fontsize=fontsize)

    div = axgrid.make_axes_locatable(ax)
    cax = div.append_axes("right",size="5%",pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel(r'%s' % (clabel), fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    

    ax.scatter(positions_specwizard[:,0],positions_specwizard[:,1], color = cm.get_cmap(colmap)(coldens_sw_normed),edgecolor='black',linewidth=1,s=50)
    # save PDF figure
    fig.tight_layout()
    #fig.subplots_adjust(right=0.88)
    plt.savefig(name,format = 'png')
    plt.close()
    return 0

def compvalues_subslice(coldensmap, specout, ion, name = mdir + 'specwiz_projection_comparions_test2_hiresproj.png',fontsize=11,colmap='viridis', title = 'region comparison test1, index 45',clabel = r'$\log_{10} N_{\mathrm{O VII}}\, [\mathrm{cm}^{-2}]$'):

    pixvals = coldensmap.pixvals
    coldens_specwizard = specout.coldens[ion]
    positions_specwizard = specout.positions
    
    plot_box = (np.min(positions_specwizard[:,0]),np.max(positions_specwizard[:,0]),np.min(positions_specwizard[:,1]),np.max(positions_specwizard[:,1]))

    pixvals_lowerleftcoords = coldensmap.slicecen - coldensmap.sidelength/2.

    plotpixcen = [np.round(((plot_box[1]+plot_box[0])/2. - coldensmap.slicecen[0] + coldensmap.sidelength/2.)/coldensmap.sidelength * float(coldensmap.numpix) -0.5,0).astype(int),\
                  np.round(((plot_box[3]+plot_box[2])/2. - coldensmap.slicecen[1] + coldensmap.sidelength/2.)/coldensmap.sidelength * float(coldensmap.numpix) -0.5,0).astype(int)]
    numpixsel = np.round((plot_box[1]-plot_box[0])/coldensmap.sidelength*coldensmap.numpix,0).astype(int)
    numpixsel += 18

    # left edge of min pixel, right edge of max (so +1 in 2nd, 4th lines) 
    # 0   1   2   3   4   5   6  edge coordinates [sidelength/numpix]
    # -------------------------  
    # | 0 | 1 | 2 | 3 | 4 | 5 |  pixel numbers (array indices)
    # -------------------------
    plot_box2 = (pixvals_lowerleftcoords[0]  + float(plotpixcen[0] - numpixsel/2)*coldensmap.sidelength/coldensmap.numpix,\
                 pixvals_lowerleftcoords[0]  + float(plotpixcen[0] + numpixsel/2 +1)*coldensmap.sidelength/coldensmap.numpix,\
                 pixvals_lowerleftcoords[1]  + float(plotpixcen[1] - numpixsel/2)*coldensmap.sidelength/coldensmap.numpix,\
                 pixvals_lowerleftcoords[1]  + float(plotpixcen[1] + numpixsel/2 +1)*coldensmap.sidelength/coldensmap.numpix )
    selection_proj = ( slice(plotpixcen[0] - numpixsel/2, plotpixcen[0] + (numpixsel)/2 +1,1),\
                       slice(plotpixcen[1] - numpixsel/2, plotpixcen[1] + (numpixsel)/2 +1,1) )

    im_proj = pixvals[selection_proj]

    Vmin = min(np.min(im_proj),np.min(coldens_specwizard))
    Vmax = min(np.max(im_proj),np.max(coldens_specwizard))

    coldens_sw_normed = (coldens_specwizard - Vmin)/(Vmax-Vmin)
      
    fig = plt.figure(figsize = (5.5, 5.)) # large size just as a trick to get higher resolution
    ax = plt.subplot(111)   
    ax.set_xlabel(r'X [cMpc/h]',fontsize=fontsize)
    ax.set_ylabel(r'Y [cMpc/h]',fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize)
    ax.patch.set_facecolor(cm.get_cmap(colmap)(0.)) # sets background color to lowest color map value 

    img = ax.imshow(im_proj.T, extent=plot_box2, origin='lower', cmap=cm.get_cmap(colmap), vmin = Vmin, vmax=Vmax, interpolation='nearest',aspect='auto')
    plt.title(title,fontsize=fontsize)

    div = axgrid.make_axes_locatable(ax)
    cax = div.append_axes("right",size="5%",pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel(r'%s' % (clabel), fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    

    ax.scatter(positions_specwizard[:,0],positions_specwizard[:,1], color = cm.get_cmap(colmap)(coldens_sw_normed),edgecolor='black',linewidth=1,s=50)
    # save PDF figure
    fig.tight_layout()
    #fig.subplots_adjust(right=0.88)
    plt.savefig(name,format = 'png')
    plt.close()
    return 0

# plots the pixel region from coldensmap for a sightlinenumber (counted in the selection) using imshow,
# and the sightlines from specout in the plotted region using colored points 
def plot_pixregions(ax,coldensmap,specout,ion,sightlinenumber,maxdexdiff=2.,fontsize=12.,colmap='viridis',xlabel = r'X [cMpc/h]', ylabel = r'Y [cMpc/h]',xticks=True,yticks=True, selection = slice(None,None,None)):

    # general image setup
    ax.patch.set_facecolor(cm.get_cmap(colmap)(0.))
    ax.minorticks_off()
    ax.tick_params(labelsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize)
    if not xticks:
        ax.xaxis.set_major_locator(plt.NullLocator())
    if not yticks:
        ax.yaxis.set_major_locator(plt.NullLocator())
    

    im = (coldensmap.pixregions[selection])[sightlinenumber] # use '2-part' selection since pixregions becomes an object array at the edges
    extent = (coldensmap.extents[selection])[sightlinenumber,:]
    # selection is slightly complicated, 
    # but ensures that if more than one sightline is in the region, all are plotted
    # (includes non-slected sightlines)
    sightlinesel = np.all(np.array([extent[0] < specout.positions[:,0],\
                                    extent[1] > specout.positions[:,0],\
                                    extent[2] < specout.positions[:,1],\
                                    extent[3] > specout.positions[:,1] ]),axis = 0)
    if not (sightlinesel[selection])[sightlinenumber]:
        print('Something has gone wrong in the sightline/region selection: sightlines %i is not in the corresponding projected region.'%sightlinenumber)
    sightlines = specout.coldens[ion][sightlinesel]
    sightlinepositions = specout.positions[sightlinesel]

    immax = np.max(im)
    slmax = np.max(sightlines)
    vmax = max(immax,slmax)
    vmin = vmax - maxdexdiff
    coldens_sw_normed = (sightlines - vmin)/(vmax-vmin)

    img = ax.imshow(im.T,origin='lower', cmap=cm.get_cmap(colmap),\
                    vmin = vmin, vmax=vmax, interpolation='nearest',\
                    extent = extent, aspect='auto')
    ax.scatter(sightlinepositions[:,0],sightlinepositions[:,1], color = cm.get_cmap(colmap)(coldens_sw_normed),edgecolor='black',linewidth=1,s=50)
    ax.text(0.95,0.05,'%.2f'%np.max(sightlines),fontsize=fontsize, color= 'orangered',\
            horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)
    ax.text(0.95,0.05 + float(fontsize)/100.,'%.2f'%np.max(im),fontsize=fontsize, color='orangered',\
            horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)
    return 0

# plots the pixel regions from coldensmap in the selection using imshow,
# and the sightlines from specout in the plotted regions using colored points
# sizescale = size of a single subplot, shape is (nrows, ncolumns) for the subplots (an extra row is added for scale indication etc.)
# selection determines which sightline regions (Spectrum#) to use
def collageplot_compvalues(coldensmap,specout,ion,shape,maxdexdiff=2.,sizescale=2.,fontsize=12.,colmap='viridis',selection = slice(None,None,None),\
                           clabel=r'$\log_{10} N \, [\mathrm{cm}^{-2}]$',\
                           title = 'Total hydrogen column density in specwizard and projection',\
                           imname = 'specwiz_collageplot_compvalues_test.png'):
    
    ncols = shape[1]
    nrows = shape[0]
    if ncols*nrows != specout.numspecs and selection == slice(None,None,None):
        print('Plotting %i panels, but %i sightlines available; might fail.'%(ncols*nrows,specout.numspecs))
    if ncols < 3:
        print('There will be some index errors when plotting the bottom (label) row')

    topedge = 0.95
    fig = plt.figure(figsize=(sizescale*ncols/topedge,sizescale*nrows+1.))
    grid = gsp.GridSpec(nrows+1,ncols,height_ratios=list((sizescale,)*nrows)+[1.])
    grid.update(left=0., bottom=0., right=1., top=topedge, wspace=0., hspace=0.)

    fig.suptitle(title)

    axes = np.empty((nrows+1,ncols),dtype=object)
    for col in range(ncols):
        for row in range(nrows+1):
            axes[row,col] = plt.subplot(grid[ncols*row+col])    
    # turn off axis ticks and space between and around subplots
    #for ax in np.ndarray.flatten(axes[nrows,:]):  
    #    ax.set_aspect(0.1, adjustable='box-forced')
    #for ax in np.ndarray.flatten(axes[:nrows,:]):
    #    ax.minorticks_off()
    #    ax.xaxis.set_major_locator(plt.NullLocator()) # turns off x axis ticks
    #    ax.yaxis.set_major_locator(plt.NullLocator())   
    #    ax.set_aspect(1., adjustable='box-forced')

    # plots the slected regions and sightlines
    for col in range(ncols):
        for row in range(nrows):
            ax = axes[row,col]
            sightlinenumber = row*ncols+col
            plot_pixregions(ax,coldensmap,specout,ion,sightlinenumber,maxdexdiff=maxdexdiff,fontsize=fontsize,colmap=colmap,xlabel=None,ylabel=None,xticks=False,yticks=False,selection=selection) 
    
    # put color map and legend in; getting the ticks right was a pain in the neck
    # try other tick setting methods at your own risk
    ax = axes[nrows,0]
    ax.set_aspect(0.1, adjustable='box-forced')
    cmap = mpl.cm.get_cmap(colmap)
    norm = mpl.colors.Normalize(vmin=-1*maxdexdiff,vmax=0.)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='horizontal',ticks=np.arange(0.,-1*maxdexdiff*1.001,-0.5))
    ax.tick_params(labelsize=fontsize-2)
    if clabel != None:
        cbar.set_label('%s'%clabel,fontsize=fontsize)
                
    ax = axes[nrows,1]
    ax.arrow(0.,0.8,1.,0.,length_includes_head=True,fc='black', head_width = 0.05, transform=ax.transAxes)
    ax.arrow(1.,0.8,-1.,0.,length_includes_head=True,fc='black', head_width = 0.05, transform=ax.transAxes)
    ax.text(0.5,0.75,'%.2f ckpc'%((coldensmap.extents[0,1]-coldensmap.extents[0,0])*1e3/hcosm), color = 'black', fontsize = fontsize,\
            horizontalalignment='center', verticalalignment = 'top', transform=ax.transAxes)
    
    ax = axes[nrows,2]
    ax.text(0.95,0.05,'max sightline value',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes,color='black')
    ax.text(0.95,0.05 + 1.6*float(fontsize)/100.,'max projected value',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes,color='black')
    ax.text(0.0,0.05 + 3*1.6*float(fontsize)/100.,'text in plots indicates',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax.transAxes,color='black')
    ax.text(0.0,0.05 + 2*1.6*float(fontsize)/100.,'maximum %s:'%clabel,fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax.transAxes,color='black')

    # turn off axis borders, labels, etc. on bottom row
    for col in range(1,ncols):
        ax = axes[nrows, col]
        ax.axis('off')
    plt.savefig(mdir + imname,format = 'png',bbox_inches='tight')
    plt.close()
    

def plot_percentiles_outliers(ax, xs, ys, percentiles = [5,25,50,75,95], nminbin = 20, plotyoutliers = True, binsize = 0.2, label = None, color = 'blue'):
    # bin and get percentiles (5,25,50,75,95)
     xbins = np.arange(0.999*np.min(xs), np.max(xs)*1.001+0.1*0.999,binsize)
     bininds = np.digitize(xs,xbins)
     ypercents = np.array([np.percentile(ys[bininds==i],percentiles) for i in range(1,len(xbins))])
     ypercents = ypercents.T
     xbins_mid = xbins[:-1] + np.diff(xbins)/2.
     
     # plot the points: outside central 5-95% range, in bins with < nminbin points
     # select bins with too few points, and the points they contain
     xbins_count =  np.array([ len(np.where(bininds==i)[0]) for i in range(1,len(xbins)) ])
     bininds_lowcount = xbins_count <  nminbin
     bininds_ok = xbins_count >= nminbin
     #print(bininds_lowcount)
     #print(bininds_ok)
     if np.any(bininds_lowcount):
         outliers_frombin = np.array([ bininds == i for i in np.arange(1,len(xbins))[bininds_lowcount] ] ) # check for each data point index if it is in a low count bin
         outliers_frombin = np.any(outliers_frombin,axis=0) # select data point if it is in any low count bin  
     else: # if there are no low count bins, the initial list will be empty -> will get the wrong shape
         outliers_frombin = np.zeros((len(xs),),dtype=bool) 
      

     # get outliers from percentiles (just overplotting would also show points 'sticking out' of the end bins)
     isoutlier = np.zeros(len(xs),dtype=bool)
     for i in range(1,len(xbins)): # do percentile selection
         isoutlier[bininds == i] = np.any( np.array([ys[bininds == i] < ypercents[0][i-1],ys[bininds == i] > ypercents[-1][i-1]]) ,axis=0)
     #print(isoutlier)
     #print(outliers_frombin)
     isoutlier = np.any(np.array([isoutlier,outliers_frombin]),axis=0) # add bin selection        

     #plot in this order 
     # note that ok bins selection will not create gaps is low bins numbers occur away from the ends
     # use isoutlier as selection in scatter to get all points outside percentiles, outliers_frombin to get only low bin count points
     if plotyoutliers:
         ax.scatter(xs[isoutlier],ys[isoutlier],color=color,s=5, label= label)
     else:
         ax.scatter(xs[outliers_frombin],ys[outliers_frombin],color=color,s=5, label = label)
     ax.fill_between(xbins_mid[bininds_ok],ypercents[0,bininds_ok],ypercents[-1,bininds_ok],facecolor=color,alpha=0.2, label = None)
     ax.fill_between(xbins_mid[bininds_ok],ypercents[1,bininds_ok],ypercents[-2,bininds_ok],facecolor=color,alpha=0.2, label = None)
     ax.plot(xbins_mid,ypercents[2],color=color,label=None)
    

def plot_EWconv_wi_gi(cubics, gaussians, ion, reslabels = None, fontsize=14):
     # percentile stuff from crdists
     # cubics and gaussians: list of Specouts for which to get the EW convergence, ordered low -> high res.
     fig, axes = plt.subplots(1,3,figsize=(15.,5.))
     ax1 = axes[0]
     ax2 = axes[1]
     ax3 = axes[2]
     colorlist = ['red', 'green', 'blue', 'purple']

     # get the data out of the specout instances
     raw1 = [spec.EW[ion] for spec in cubics]
     x1 = raw1[-1]
     ys1 = [r1/x1 for r1 in raw1[:-1]]  
     raw2 = [spec.EW[ion] for spec in gaussians]
     x2 = raw2[-1]
     ys2 = [r2/x2 for r2 in raw2[:-1]] 
     
     x1 = np.log10(x1)
     x2 = np.log10(x2)
     
     for i in range(len(ys1)):
         plot_percentiles_outliers(ax1, x1, ys1[i], percentiles = [5,25,50,75,95], nminbin = 20, plotyoutliers = True, binsize = 0.3, label = reslabels[i], color = colorlist[i])
     ax1.plot([np.min(x1),np.max(x1)],[1,1],linestyle='dashed',color='gray',label='equal')
     ax1.legend(fontsize=fontsize, title = 'median, 25-75%, 5-95%')
     ax1.get_legend().get_title().set_fontsize(fontsize)
     ax1.set_yscale('log')

     for i in range(len(ys2)):
         plot_percentiles_outliers(ax2, x2, ys2[i], percentiles = [5,25,50,75,95], nminbin = 20, plotyoutliers = True, binsize = 0.3, label = reslabels[i], color = colorlist[i])
     ax2.plot([np.min(x2),np.max(x2)],[1,1],linestyle='dashed',color='gray',label='equal')
     ax2.legend(fontsize=fontsize, title = 'median, 25-75%, 5-95%')
     ax2.get_legend().get_title().set_fontsize(fontsize)
     ax2.set_yscale('log')

     plot_percentiles_outliers(ax3, x2, x1-x2, percentiles = [5,25,50,75,95], nminbin = 20, plotyoutliers = True, binsize = 0.3, label = reslabels[-1], color = colorlist[-1])
     ax3.plot([np.min(x2),np.max(x2)],[0,0],linestyle='dashed',color='gray',label='equal')
     ax3.legend(fontsize=fontsize, title = 'median, 25-75%, 5-95%')
     ax3.get_legend().get_title().set_fontsize(fontsize)

     ax1.set_xlabel(r'$\log_{10} EW_{%s} \, [\mathrm{\AA}]$'%(ion),fontsize=fontsize)
     ax1.set_ylabel(r'$EW_{%s} / EW_{%s}$(%s)'%(ion,ion,reslabels[-1]),fontsize=fontsize)
     ax1.set_title('Cubic spline kernels',fontsize=fontsize)

     ax2.set_xlabel(r'$\log_{10} EW_{%s} \, [\mathrm{\AA}]$'%(ion),fontsize=fontsize)
     ax2.set_ylabel(r'$EW_{%s} / EW_{%s}$(%s)'%(ion,ion,reslabels[-1]),fontsize=fontsize)
     ax2.set_title('Gaussian, integrated kernels',fontsize=fontsize)

     ax3.set_xlabel(r'$\log_{10} EW_{%s} \, [\mathrm{\AA}]$, %s, gaussian'%(ion,reslabels[-1]),fontsize=fontsize)
     ax3.set_ylabel(r'$\log_{10} EW_{%s}$, cubic spline / gaussian'%(ion),fontsize=fontsize)
     ax3.set_title('Gaussian, integrated vs cubic spline kernels',fontsize=fontsize)

     ax1.tick_params(labelsize=fontsize,axis='both')
     ax2.tick_params(labelsize=fontsize,axis='both')
     ax3.tick_params(labelsize=fontsize,axis='both')

     #ax1.set_ylim(0.2,2)
     #ax2.set_ylim(0.9,1.02)

     #plt.show()
     plt.savefig(mdir + 'specwiz_EW_convergence_sntd-gi_test4_test10-los_o8_standard-9x-i9x-i81x-stnd-projres_separate_distributions.png',format = 'png',bbox_inches='tight')


def getNEW_wsubsamples_multiion():
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17'] # only o8 doublet is expected to be unresolved -> rest is fine to use single lines
    filen = '/net/luttero/data2/specwizard_data/sample3/spec.snap_027_z000p101.0.hdf5'
    sfilen = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'
    outfilen = '/net/luttero/data2/specwizard_data/sample3_coldens_EW_subsamples.hdf5'
    
    so = Specout(filen, getall=False)
    so.getEW(dions=ions)
    so.getcoldens(dions=ions)

    ionselgrpn = {'o7': 'file0',\
                  'o8': 'file1',\
                  'o6': 'file2',\
                  }
    
    selections = {}
    with h5py.File(sfilen, 'r') as fs:
        numpix = fs['Header'].attrs['numpix']
        specpos = so.positions / so.cosmopars['boxsize'] * numpix
        for ion in ionselgrpn:
            sgrp = fs['Selection/%s'%(ionselgrpn[ion])]
            selpos = np.array(sgrp['selected_pixels_thision'])
            
            indsclosest = ( (selpos[:, np.newaxis, 0] - specpos[np.newaxis, :, 0])**2 +\
                            (selpos[:, np.newaxis, 1] - specpos[np.newaxis, :, 1])**2 ).argmin(axis=1)
            selections[ion] = indsclosest
    
    with h5py.File(outfilen, 'w') as fo:
        hed = fo.create_group('Header')
        hed.attrs.create('numpix_map', numpix)
        hed.attrs.create('filename_spectra', np.string_(filen))
        hed.attrs.create('filename_sample_selection', np.string_(sfilen))
        cg = hed.create_group('cosmopars')
        for key in so.cosmopars:
            cg.attrs.create(key, so.cosmopars[key])
        for sion in selections:
            grp = fo.create_group('%s_selection'%sion)
            sel = selections[sion]
            for ion in ions:
                sgrp = grp.create_group('%s_data'%ion)
                sgrp.create_dataset('logN_cmm2', data=so.coldens[ion][sel])
                sgrp.create_dataset('EWrest_A', data=so.EW[ion][sel])
                
        grp = fo.create_group('full_sample')
        for ion in ions:
            sgrp = grp.create_group('%s_data'%ion)
            sgrp.create_dataset('logN_cmm2', data=so.coldens[ion])
            sgrp.create_dataset('EWrest_A', data=so.EW[ion])
                

    
    


# ----------------------- MERGE OUTPUT FILES FROM 'PARALLEL' RUNS: only sightlines differ -----------------------
# groups in specwizard hdf5 files that contain non-l.o.s.-dependent parameters
# check attributes ('Parameters' has no attributes, only subgroups):
compgroups = ['Constants', 'Header', 'Header/ModifyMetallicityParameters', 'Parameters/ChemicalElements','Parameters/SpecWizardRuntimeParameters', 'Units']
# check array
comparrs = ['VHubble_KMpS']
# ignore parameters difference by default
ignorediffs = ['los_coordinates_file', 'NumberOfSpectra', 'SpectrumFile', 'outputdir'] # all in SpecWizardRuntimeParameters

def mergespecout(file1,file2,outname = None, ignoredifferences = [], trusthdf5parnumspec = False):
    '''
    inputs:
    ----------
    ignoredifferences: parameters in hdf5 files which are allowed to differ 
      between file1 and file2, besides line of sight differences 
    trusthdf5parnumspec: trust the 
      ['Parameters/SpecWizardRuntimeParameters'].attrs['NumberOfSpectra']
      parameter to describe the actual number of spectra in the file
      (may not be true if a file ended due to an error or abort partway 
      through)

    outputs: 
    ----------
    new hdf5 file merged from file1, file2 (file1 and file2 are not altered or
      removed), assuming these contain different sightlines, but are otherwise 
      the same
      spectrum numbering puts file1 first
    found_differences: differences in hdf5 files that are not due to different 
      lines of sight
      can be input to ignoreddifferences in a second run if not a problem 
    '''
    if outname is None:
        outname = sdir + 'merged__%s__%s.hdf5'%(file1[:-4],file2[:-4])
    if not '/' in outname:
        outname = sdir + outname
    if not '/' in file1:
        file1 = sdir + file1
    if not '/' in file2:
        file2 = sdir + file2

    # attributes: specfile (h5py.File), specgroups (specfile.keys containing spectra), numspecs
    # collects too little info to worry about getting info twice
    spec1 = Specout(file1,getall=False)
    spec2 = Specout(file2,getall=False)
    
    # check whether parameters other than lines of sight and number of spectra all agree 
    # (will include some unused stuff like metallicity modifications, etc.) 
    print('Checking compatibility of input files...', end = '')
    found_differences = [] 
    ignoredifferences += ignorediffs
    domerge = True

    if trusthdf5parnumspec:
        numspec1 = spec1.specfile['Projection'].attrs['nspec']
        numspec2 = spec2.specfile['Projection'].attrs['nspec']
    else: # retrieved from explicitly counting Spectrum* groups in files
        numspec1 = spec1.numspecs
        numspec2 = spec2.numspecs

    for group in compgroups:
        # items -> list of (name, value) tuples
        attrs1 = spec1.specfile[group].attrs.items()
        attrs2 = spec2.specfile[group].attrs.items()
        # doesn't matter what it's sorted by extacly, just make sure the attribute orders are the same
        attrs1.sort(key = lambda x: x[0])
        attrs2.sort(key = lambda x: x[0])
        attrs1 = np.asarray(attrs1,dtype=object) #object array is not efficient, but array used just for convenience anyway
        attrs2 = np.asarray(attrs2,dtype=object)
             
        if attrs1[:,0].shape != attrs2[:,0].shape:
            print('File merge failed due to attribute number mismatch in group :'%sgroup)
            print('%s contains attributes %s'%(file1,str(attrs1[:,0])))
            print('%s contains attributes %s'%(file2,str(attrs2[:,0])))
            domerge = False
            continue
        elif np.all(attrs1 == attrs2):
            #print('%s: match!'%group)
            # everything matches, nothing to do
            continue   
        elif not np.all(attrs1[:,0]==attrs2[:,0]): # files do not have the same attributes
            print('File merge failed due to attribute mismatch in group :'%sgroup)
            print('%s contains attributes %s'%(file1,str(attrs1[:,0])))
            print('%s contains attributes %s'%(file2,str(attrs2[:,0])))
            domerge = False
            continue
        # array-all is because some attribute values are arrays; 
        # list workaround is due to issue:
        # DeprecationWarning: elementwise != comparison failed; this will raise an error in the future.
        #!/cosma/local/Python/2.7.3/bin/python

        attrdiffs = np.array([np.array(attrs1[i,1]!=attrs2[i,1]).all() for i in range(len(attrs1[:,1]))])
        #print(str(attrdiffs))
        #return attrs1, attrs2
        #print(str(attrs2[:,1]))
        #print(str(attrs1[:,0]))
        #print(str(attrs2[:,0]))
        #print(str(attrdiffs))
        tempdiffs = list(attrs1[:,0][attrdiffs])
        found_differences += tempdiffs
        if not set(tempdiffs).issubset(ignoredifferences):
            domerge = False
            # do not stop looking for differences yet

    for arrname in comparrs:
        arr1 = np.array(spec1.specfile[arrname])
        arr2 = np.array(spec2.specfile[arrname])
        if not np.all(arr1==arr2):
            if arrname not in ignoredifferences:
                print('File merge failed due to array mismatch in array: %s'%arrname)
                print('%s contains array %s'%(file1,str(arr1)))
                print('%s contains array %s'%(file1,str(arr2)))
                domerge = False
            found_differences.append(arrname)

    if not domerge: # attribute differences were found that were not allowed
        print('File merge failed due to non-exempted attribute or array difference')
        return found_differences
    print('done')
                    
    # do the actual file merger; coordfiles are a tuple of los coordinate files in the order they are included as spectra
    ## setup: attributes to do special cases for (expected to differ between different los files)
    print('Starting file merger...')

    numspecsout = numspec1 + numspec2

    coords1 = spec1.specfile['Parameters/SpecWizardRuntimeParameters'].attrs['los_coordinates_file']
    coords2 = spec2.specfile['Parameters/SpecWizardRuntimeParameters'].attrs['los_coordinates_file']  
    # list(str) returns a list of characters
    if isinstance(coords1, np.ndarray):
        coords1 = list(coords1)
    else:
        coords1 = [coords1] 
    if isinstance(coords2, np.ndarray):
        coords2 = list(coords2)
    else:
        coords2 = [coords2] 
    coordfiles = tuple(coords1+coords2)
    #print('coordfiles: %s'%str(coordfiles))
 
    nspec1 = spec1.specfile['Parameters/SpecWizardRuntimeParameters'].attrs['NumberOfSpectra']
    nspec2 = spec2.specfile['Parameters/SpecWizardRuntimeParameters'].attrs['NumberOfSpectra']
    if isinstance(nspec1, np.ndarray):
        nspec1 = list(nspec1)
    else:
        nspec1 = [nspec1] 
    if isinstance(nspec2, np.ndarray):
        nspec2 = list(nspec2)
    else:
        nspec2 = [nspec2] 
    nspecout = tuple(nspec1+nspec2)
    #print('nspecout: %s'%str(nspecout))
    
    outputdir1 = spec1.specfile['Parameters/SpecWizardRuntimeParameters'].attrs['outputdir']
    outputdir2 = spec2.specfile['Parameters/SpecWizardRuntimeParameters'].attrs['outputdir']
    if isinstance(outputdir1, np.ndarray):
        outputdir1 = list(outputdir1)
    else:
        outputdir1 = [outputdir1] 
    if isinstance(outputdir2, np.ndarray):
        outputdir2 = list(outputdir2)
    else:
        outputdir2 = [outputdir2] 
    outputdirout = tuple(outputdir1+outputdir2)
    #print('outputdirout: %s'%str(outputdirout))

    # initialise output hdf5 file
    outfile = h5py.File(outname,'w')
   
    # copy attributes from file1; overwrite at the end: 'los_coordinates_file' and 'NumberOfSpectra'
    toplevelparents = []
    for group in compgroups:
        # outfile.create_group(group)
        # using parent to avoid double/nested copies, hopefully
        parent = group.split('/')
        if len(parent) == 1: # group is top-level
            #print('parent: %s, group: %s'%(parent[0],group))
            spec1.specfile.copy(group,outfile)
            toplevelparents.append(group)
        else: # group is sub of a top-level group (but if top-level has no attributes, it may not be listed)
            parent = parent[0]
            #print('parent: %s, group: %s'%(parent,group))
            if parent not in toplevelparents:            
                spec1.specfile.copy(parent,outfile)
                toplevelparents.append(parent)
            else:
                continue #we've already copied this one

    # set standard exempted parameters to reasonable values for the merged file: tuple of input files or total merged file properties 
    outfile['Parameters/SpecWizardRuntimeParameters'].attrs['NumberOfSpectra'] = nspecout
    outfile['Parameters/SpecWizardRuntimeParameters'].attrs['los_coordinates_file'] = coordfiles
    outfile['Parameters/SpecWizardRuntimeParameters'].attrs['SpectrumFile'] = outname
    outfile['Parameters/SpecWizardRuntimeParameters'].attrs['outputdir'] = outputdirout

    # handle projection file: append file2 arrays to file1 arrays
    # let nspec in 'Projection' give the in-order tuple of numers of sightlines from los files; 
    # in normal files, this information is the same as 'Parameters/SpecWizardRuntimeParameters' -> 'NumberOfSpectra',
    # so no information is lost this way

    def outfromappend(out,in1,in2,group,dset,numspec1,numspec2,axis=0): #numbyspec option means use in1 only up to spec1.numspecs
        # appends along axis 0 by default
        #print(group)
        #print(dset)
        #print(group+'/'+dset)
        #print('numspec1:%i , numspec2:%i'%(numspec1,numspec2))
        #print(str(in1[group+'/'+dset]))
        shape1 = np.array(in1[group+'/'+dset].shape)
        shape2 = np.array(in1[group+'/'+dset].shape)
        if axis >= len(shape1) or not np.all(shape1[:axis] == shape2[:axis]) or not np.all(shape1[axis+1:] == shape2[axis+1:]):
            print('Cannot append arrays along axis %i for dataset shapes %s, %s'%(axis,str(shape1),str(shape2)))
 
        shapeout = np.empty(len(shape1),dtype=np.int)       
        shapeout[axis] = numspec1 + numspec2
        #shapeout[axis] = shape1[axis] + shape2[axis]
        shapeout[:axis] = shape1[:axis]
        shapeout[axis+1:] = shape1[axis+1:]
        
        dtypeout = in1[group+'/'+dset].dtype
        
        out[group].create_dataset(dset,shapeout,dtype=dtypeout)

        sels1 = list((slice(None,None,None),)*len(shape1))
        sels1[axis] = slice(None,numspec1,None)
        sels1=tuple(sels1)
        sels2 = list((slice(None,None,None),)*len(shape1))
        sels2[axis] = slice(numspec1,None,None)
        sels2=tuple(sels2)
        sels2in = list((slice(None,None,None),)*len(shape1))
        sels2in[axis] = slice(None,numspec2,None)
        sels2in=tuple(sels2in)
        #print('sels1:%s, sels2:%s, sels2in:%s'%(str(sels1),str(sels2),str(sels2in)))
        #print('out shape:%s, in shape:%s'%(str(out[group+'/'+dset].shape), in1[group+'/'+dset].shape))
        out[group+'/'+dset][sels1] = in1[group+'/'+dset][sels1]
        out[group+'/'+dset][sels2] = in2[group+'/'+dset][sels2in]
    print('attributes done...')

    group = 'Projection'

    outfile.create_group(group)
    nspec1 = spec1.specfile[group].attrs['nspec']
    nspec2 = spec2.specfile[group].attrs['nspec']
    if isinstance(nspec1, tuple):
        nspec1 = list(nspec1)
    else:
        nspec1 = [nspec1] 
    if isinstance(nspec2, tuple):
        nspec2 = list(nspec2)
    else:
        nspec2 = [nspec2] 
    nspecout = tuple(nspec1+nspec2)
    outfile[group].attrs['nspec'] = numspecsout
    
    if 'ncontr' in spec1.specfile[group].keys() and 'ncontr' in spec2.specfile[group].keys(): 
        outfromappend(outfile,spec1.specfile,spec2.specfile,group,'ncontr',numspec1,numspec2)
    else:
        print('Projection/ncontr not included in merged file, since not present in both input files')
    outfromappend(outfile,spec1.specfile,spec2.specfile,group,'x_fraction_array',numspec1,numspec2)
    outfromappend(outfile,spec1.specfile,spec2.specfile,group,'y_fraction_array',numspec1,numspec2)
    
    for arrname in comparrs:
        outfile.copy(spec1.specfile[arrname], arrname)
    print('small datasets done...')


    print('starting spectrum group copy')
    for i in range(numspec1):
        group = 'Spectrum%i'%i
        #outfile.create_group(group)
        spec1.specfile.copy(group,outfile)
        print('.', end='')
    for i in range(numspec2):
        groupin = 'Spectrum%i'%i
        groupout = 'Spectrum%i'%(i+numspec1)
        #outfile.create_group(groupout)
        spec1.specfile.copy(groupin,outfile,name=groupout)
        print('.', end='')
    print('done')
    outfile.close()
    print('File merge succesful.')
    return found_differences
