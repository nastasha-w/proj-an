'''
plot spectra etc. from specwizard outputs. Read in single spectra, unlike in specwiz_proc, since this is very slow on large samples.
'''

import numpy as np 
import h5py 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import ctypes as ct
#import numbers #useful for type checks
import scipy
#import scipy.signal as scisig

import cosmo_utils as cu
import make_maps_opts_locs as ol


sdir = ol.sdir
ndir = ol.npzdir
pdir = ol.pdir
mdir = ol.mdir

import specwiz_proc_minimal as sp
import ion_line_data as ild

lambda_rest = ild.linetable['lambda_angstrom']
fosc = ild.linetable['fosc']

def getspecname(ion, qty = None):
    '''
    parse abbreviated spectrum quantities
    if no abbreviations are matched, the iput names themselves are used
    so just using the names form the hdf5 file is always ok
    '''
    if ion in  ['mass', 'Mass']:
        ion = 'RealSpaceMassWeighted'
    
    if qty is None:
        qty = 'Flux'
    else:
        qty = qty.split('/')
        if len(qty) == 1:
            if qty[0] in  ['Nion', 'coldens']:
                qty[0] = 'LogTotalIonColumnDensity'
            elif qty[0] == 'tau':
                qty[0] = 'OpticalDepth'
            elif qty[0] == 'vpec':
                qty[0] = 'LOSPeculiarVelocity_KMpS'
            elif qty[0] in ['zfrac', 'metalfrac', 'fz', 'fmetal']:
                qty[0] = 'MetalMassFraction'
        else:
            if qty[0] in ['realw', 'posw', 'realnionw']:
                qty[0] = 'RealSpaceNionWeighted'
            elif qty[0] in ['tauw', 'zw', 'ztauw']:
                qty[0] = 'RedshiftSpaceOpticalDepthWeighted'
            if qty[1] == 'vpec':
                qty[1] = 'LOSPeculiarVelocity_KMpS'
        qty = '/'.join(qty)
    return '%s/%s'%(ion,qty)


class Sightline:
    '''
    Like specwiz_proc's Specout, but just one sightline. Can be bundeled into a small group
    '''
    def __init__(self, specfile, specnum, ofile=None):
        '''
        ofile: open hdf5 file, can come from an enclosing SpecSample class
        '''
        self.filename = specfile
        self.specnum = specnum
        self.getscalardata()
        self.spectra = {}
        self.ofile = ofile
        
    def getscalardata(self):
        self.file = h5py.File(self.filename, 'r')
        # box and cosmology
        self.cosmopars = {'boxsize':     self.file['/Header'].attrs.get('BoxSize'),\
                          'h':           self.file['/Header'].attrs.get('HubbleParam'),\
                          'a':           self.file['/Header'].attrs.get('ExpansionFactor'),\
                          'z':           self.file['/Header'].attrs.get('Redshift'),\
                          'omegam':      self.file['/Header'].attrs.get('Omega0'),\
                          'omegalambda': self.file['/Header'].attrs.get('OmegaLambda'),\
                          'omegab':      self.file['/Header'].attrs.get('OmegaBaryon') }
        # redshift space (z=0 observer frame)
        self.slicelength = self.cosmopars['boxsize']/self.cosmopars['h'] # in cMpc
        self.slicelength_pcm = self.slicelength * self.cosmopars['a'] * cu.c.cm_per_mpc
        self.deltaredshift = cu.Hubble(self.cosmopars['z'], cosmopars=self.cosmopars) / cu.c.c * self.slicelength * cu.c.cm_per_mpc
        # position
        self.fracpos = np.array([self.file['/Projection/x_fraction_array'][self.specnum], self.file['/Projection/y_fraction_array'][self.specnum]])
        self.file.close()

    def pixind(self,npix):
        return (np.round(npix*self.fracpos - 0.5,0)).astype(int)

    def getspectrum(self, ion, qty=None, corr=False):
        '''
        ofile: open file; can be input from a specsample to avoid multiple 
        open files at once
        retrieves spectra from specwizard output hdf5 file from specwizard
        '''
        if self.ofile is None:
            self.file = h5py.File(self.filename, 'r')
            self.tempfile = self.file
        else:
            self.tempfile = self.ofile
        if corr and (ion=='o8' and qty in ['Flux', None]):
            self.correct_o8()
            key = 'o8/FluxCorr'
        else:
            key = getspecname(ion, qty=qty)
            print(key)
            self.spectra[key] = np.array(self.tempfile['Spectrum%i/%s'%(self.specnum, key)])
        
            # overwrtitten and handled by correct_o8 if that was called 
            if self.ofile is None:
                self.file.close()
            del self.tempfile
        return self.spectra[key]
               
    def correct_o8(self, verbose = True):  ## still just the specwiz_proc version
        if self.ofile is None:
            self.file = h5py.File(self.filename, 'r')
            self.tempfile = self.file
        else:
            self.tempfile = self.ofile
        if not hasattr(self, 'corrinfo'):
            self.corrinfo = {}
        self.corrinfo['o8'] = {}
        # get o8 optical depth spectrumspec.snap_027_z000p101.0.hdf5
        self.spectra['o8/FluxCorr'] = np.array(self.tempfile['Spectrum%i/o8/OpticalDepth'%self.specnum])
        # get pixel velocity along the los (average to reduce single-pixel fp errors); velocities start at zero
        self.velperpix_kmps = np.average(np.diff(np.array(self.tempfile['VHubble_KMpS'])))
        # redshift between o8major and o8minor at z=0: measured velocity difference = total redshift * light speed * (km/s / cm/s)
        # velocity stored in specwizard output is local/proper velocity -> wavelength difference should be rest-frame
        self.lambdamajz = lambda_rest['o8major'] # * (1.+self.cosmopars['z'])
        self.lambdaminz = lambda_rest['o8minor'] # * (1.+self.cosmopars['z'])    
        self.veldiff_at_z_kmps = (self.lambdaminz-self.lambdamajz) / lambda_rest['o8major'] * cu.c.c *1.e-5 
        # what to shift the o8major array by after rescaling to get to o8minor velocity
        if verbose:
            print('Line shift error: %f km/s out of %f'%(round(self.veldiff_at_z_kmps/self.velperpix_kmps,0)*self.velperpix_kmps - self.veldiff_at_z_kmps, self.veldiff_at_z_kmps))
        self.pixshift_majtomin = int(round(self.veldiff_at_z_kmps/self.velperpix_kmps,0))
        self.corrinfo['o8'].update({'veldiff_kmps': [self.veldiff_at_z_kmps], 'pixdiff': [self.pixshift_majtomin]})
        #from Lan & Fukugita 2017: linear EW  = f lambda^2/1.13e20 * Nion; lin. approx for tau is same, but fot tau it is exact
        # -> rescale tau for minor contriution to tau_minor = tau_major * (f_min * lambda_min**2)/(f_maj * lambda_maj**2)   
        self.tauspec_o8_minor = self.spectra['o8/FluxCorr'] * (fosc['o8minor'] * lambda_rest['o8minor']**2)/(fosc['o8major'] * lambda_rest['o8major']**2)
        # shift the absorption by the appropriate number of pixels for the wavelength difference 
        self.tauspec_o8_minor = np.roll(self.tauspec_o8_minor,self.pixshift_majtomin) 
        #optical depths add up
        self.spectra['o8/FluxCorr'] += self.tauspec_o8_minor
        self.spectra['o8/FluxCorr'] = np.exp(-1.*self.spectra['o8/FluxCorr'])
    
        if self.ofile is None:
            self.file.close()

        del self.tempfile
        del self.pixshift_majtomin
        del self.veldiff_at_z_kmps
        del self.tauspec_o8_minor
        del self.lambdamajz
        del self.lambdaminz
        # keep velperpix_kmps: seems generally useful, should not cause errors

    def get_multiplet_flux(self, ion, periodic=True, verbose=False, lineincl='all'):
        '''
        Only does shifts in velocity space -> only suited for e.g. doublets, 
        not well-separated lines
        end product will always contain the original line from specwizard
        '''
        if self.ofile is None:
            self.file = h5py.File(self.filename, 'r')
            self.tempfile = self.file
        else:
            self.tempfile = self.ofile

        if not hasattr(self, 'corrinfo'):
            self.corrinfo = {}            
        self.corrinfo[ion] = {}
        self.velperpix_kmps = np.average(np.diff(np.array(self.tempfile['VHubble_KMpS'])))        
        
        if len(ild.get_linenames(ion)) <= 1: # no corrections to be made: same as regular flux spectrum
            self.spectra['%s/FluxCorr'%ion] = np.array(self.tempfile['Spectrum%i/%s/Flux'%(self.specnum, ion)])
            self.corrinfo[ion].update({'veldiff_kmps': np.array([]), 'pixdiff': np.array([]), 'v0ind': 0})
            if verbose:
                print('Ion %s has only one line listed; returning the spectrum for that line'%ion)
        else:
            self.spectra['%s/FluxCorr'%ion] = np.array(self.tempfile['Spectrum%i/%s/OpticalDepth'%(self.specnum, ion)])
            # get pixel velocity along the los (average to reduce single-pixel fp errors); velocities start at zero
            # redshift between o8major and other lines at z=0: measured velocity difference = total redshift * light speed * (km/s / cm/s)
            # velocity stored in specwizard output is local/proper velocity -> wavelength difference should be rest-frame
            self.lambdamaj = ild.linetable.loc[ild.get_major(ion), 'lambda_angstrom'] # * (1.+self.cosmopars['z'])
            if lineincl == 'all':
                self.linekeys = ild.getlinenames(ion)
            else:
                self.linekeys = lineincl
            self.major = ild.get_major(ion)
            if self.major in self.linekeys:
                self.linekeys.remove(self.major)
            self.lambdas = np.array(ild.linetable.loc[self.linekeys, 'lambda_angstrom'], dtype=np.float)
            self.veldiff_kmps = (self.lambdas - self.lambdamaj) / self.lambdamaj * cu.c.c *1.e-5 
            # what to shift the o8major array by after rescaling to get to o8minor velocity
            self.pixshift_majtomin = (np.round(self.veldiff_kmps / self.velperpix_kmps, 0)).astype(int)
            if verbose:
                print('Line shift error: %s km/s out of %s'\
                      %(self.pixshift_majtomin * self.velperpix_kmps - self.veldiff_kmps, self.veldiff_kmps))
            self.corrinfo[ion].update({'veldiff_kmps': self.veldiff_kmps, 'pixdiff': self.pixshift_majtomin, 'v0ind': 0})
            #from Lan & Fukugita 2017: linear EW  = f lambda^2/1.13e20 * Nion; lin. approx for tau is same, but for tau it is exact
            # -> rescale tau for minor contribution to tau_minor = tau_major * (f_min * lambda_min**2)/(f_maj * lambda_maj**2)   

            self.tauspecs_temp = [self.spectra['%s/FluxCorr'%ion] \
                                   * (ild.linetable.loc[self.linekeys[ind], 'fosc'] * ild.linetable.loc[self.linekeys[ind], 'lambda_angstrom']**2) \
                                   / (ild.linetable.loc[self.major, 'fosc']      * ild.linetable.loc[self.major, 'lambda_angstrom']**2) \
                                   for ind in range(len(self.linekeys)) ]
                # shift the absorption by the appropriate number of pixels for the wavelength difference 
            if periodic:
                self.tauspecs_temp = np.array([np.roll(self.tauspecs_temp[ind], self.pixshift_majtomin[ind]) \
                                               for ind in range(len(self.linekeys)) ])                       
            else:
                # interpolate onto common grid in wavelength space; 
                # this may matter for ions with more widely separated lines
                self.psmin = min(0, np.min(self.pixshift_majtomin))
                self.psmax = max(0, np.max(self.pixshift_majtomin))
                # pad all tau spectra with zeros to line them up properly
                self.spectra['%s/FluxCorr'%ion] = np.append(np.zeros(np.abs(self.psmin)), self.spectra['%s/FluxCorr'%ion])
                self.spectra['%s/FluxCorr'%ion] = np.append(self.spectra['%s/FluxCorr'%ion], np.zeros(self.psmax))
                self.corrinfo[ion].update({'v0ind': np.abs(self.psmin)})
                self.tauspecs_temp = [np.append(np.zeros(self.pixshift_majtomin[ind] - self.psmin),\
                                                         self.tauspecs_temp[ind]) \
                                      for ind in range(len(self.linekeys))]
                self.tauspecs_temp = [np.append(self.tauspecs_temp[ind],\
                                                np.zeros(self.psmax - self.pixshift_majtomin[ind])) \
                                      for ind in range(len(self.linekeys))]
                self.tauspecs_temp = np.array(self.tauspecs_temp)                         
                del self.psmin
                del self.psmax
            # add up optical depths, convert to normalised flux
            self.spectra['%s/FluxCorr'%ion] += np.sum(self.tauspecs_temp, axis=0)
            self.spectra['%s/FluxCorr'%ion] = np.exp(-1.*self.spectra['%s/FluxCorr'%ion])
 
            del self.tauspecs_temp
            del self.tempfile
            del self.pixshift_majtomin
            del self.veldiff_kmps
            del self.lambdamaj
            del self.lambdas
            del self.linekeys
    
        if self.ofile is None:
            self.file.close()
            
        
        # keep velperpix_kmps: seems generally useful, should not cause errors (same for all ions/spectra)
    
    def get_multiion_flux(self, ions,\
                          verbose=False, lineincl='all', name=None,\
                          space='wavelength', grid=None,\
                          default_oversample_factor=5.):
        '''
        input:
        ----------
        ions:     list of ion names (just e.g. 'o7', no major, minor etc.)
        name:     what to call the output spectrum in the multiline dictionary
                  default name is index in order of creation (see output)
        lineincl: 'all', or dct {ion: line list or 'all'} -- which lines to 
                  include for each ion
        space:    'wavelength' or 'energy' -- output spectra in wavelength 
                  space [A] or in energy space [keV]
        grid:     numpy array in chosen output space and units to which to 
                  interpolate the output spectra. 
        default_oversample_factor: divides the the mininum spacing in the input
                  spectra to get the output grid spacing if grid is not 
                  specified (grid=None), otherwise ignored
        output:
        ----------
        nothing returned.
        multiion flux spectrum (wavelength/energy space) is saved in 
        self.spectra.multiline[name]['flux' and ('lambda_A' or 'E_keV')],
        in rest-frame units
        lines used in the spectrum in 
        self.spectra.multiline[name]['lines']
        
        note: 
        ----------
        interpolation is necessary because spectra are generally not spaced by 
        the same amount in energy/wavelength space when they are on the same 
        velocity grid
        
        when looking at well-separated lines for one ion, use this instead of 
        get_multiline_flux to avoid errors due to absence of spacing scalings
        in velocity space in that function
        
        if the list of lines for an ion is specified, the 'main' ion (the one
        for which the original spectrum was made) will only be included if it 
        was on the list        
        '''
        
        if self.ofile is None:
            self.file = h5py.File(self.filename, 'r')
            self.ofile = self.file
            self.close_ofile = True    
        else:
            self.close_ofile = False
            
        if lineincl == 'all':
            lineincl = {ion: 'all' for ion in ions}
        
        if 'multiline' not in self.spectra.keys():
            self.spectra['multiline'] = {}
        if name is None:
            name = 'multiline_%i'%(len(self.spectra['multiline'].keys()))
        else:
            name = 'multiline_%s'%(name)
        self.spectra[name] = {}
        
        self.vvals_kmps = np.array(self.ofile['VHubble_KMpS'])
        self.velperpix_kmps = np.average(np.diff(self.vvals_kmps)) 
        # get original: all dict for each ion
        self.origs = [ild.get_major(ion) for ion in ions]
        self.targets = [lineincl[ion] if lineincl[ion] != 'all'\
                        else ild.get_linenames(ion) \
                        for ion in ions]
        # avoid modifying line lists accidentally
        self.targets = {ions[i]: list(np.copy(self.targets[i]))\
                        for i in range(len(ions))}
        self.targets_r = {tar: ion for ion in ions for tar in self.targets[ion]}
        self.origs   = {ions[i]: self.origs[i] for i in range(len(ions))}
        #print 'origs: ', self.origs.keys()
        #print 'targets: ', self.targets.keys()
        #print 'targets_r: ', self.targets_r.keys() 
        
        self.tauspecs_orig = {ion: np.array(self.ofile['Spectrum%i/%s/OpticalDepth'%(self.specnum, ion)]) for ion in ions}
        
        self.tauspecs_target = {tar: self.tauspecs_orig[self.targets_r[tar]]
                                   * (fosc[tar]                 * lambda_rest[tar]**2) \
                                   / (fosc[self.origs[self.targets_r[tar]]] * lambda_rest[self.origs[self.targets_r[tar]]]**2) \
                                for tar in self.targets_r.keys()}
            
        if space == 'wavelength':
            spacename = 'lambda_A'
            if grid is None:
                self.gmin = min([lambda_rest[tar] for tar in self.targets_r.keys()])
                self.gmax = max([lambda_rest[tar] for tar in self.targets_r.keys()]) \
                            * (1. + self.velperpix_kmps * 1e5 / cu.c.c \
                                    * len(self.tauspecs_target[self.targets_r.keys()[0]]))
                self.spacing =  self.gmin\
                               * self.velperpix_kmps * 1e5 / cu.c.c \
                               / float(default_oversample_factor)                        
            
                grid = np.arange(self.gmin * 0.99, self.gmax * 1.01 + self.spacing, self.spacing)
                del self.gmin
                del self.gmax
                del self.spacing
            self.grids = {tar: lambda_rest[tar] \
                               * (1. +  self.vvals_kmps * 1.e5 / cu.c.c) \
                          for tar in self.targets_r.keys()}
        elif space == 'energy':
            spacename = 'E_keV'
            if grid is None:
                self.gmin = min([lambda_rest[tar] for tar in self.targets_r.keys()])
                self.gmax = max([lambda_rest[tar] for tar in self.targets_r.keys()]) \
                            * (1. + self.velperpix_kmps * 1e5 / cu.c.c \
                                    * len(self.tauspecs_target[self.targets_r.keys()[0]]))
                self.gmin = cu.c.c * cu.c.planck / (self.gmin * 1.e-8) / (cu.c.ev_to_erg * 1e3)
                self.gmax = cu.c.c * cu.c.planck / (self.gmax * 1.e-8) / (cu.c.ev_to_erg * 1e3)
                self.spacing =  self.gmax\
                               * self.velperpix_kmps * 1e5 / cu.c.c \
                               / float(default_oversample_factor)                        
            
                grid = np.arange(self.gmax * 0.99, self.gmin * 1.01 + self.spacing, self.spacing)
                del self.gmin
                del self.gmax
                del self.spacing
                
            self.grids = {tar: cu.c.c * cu.c.planck / (lambda_rest[tar] * 1.e-8) / (cu.c.ev_to_erg * 1e3) \
                               / (1. +  self.vvals_kmps * 1.e5 / cu.c.c) \
                          for tar in self.targets_r.keys()}        
        else:
            raise ValueError('%s is not an option for keyword argument space'%space)
        
        self.fluxes = [scipy.interpolate.griddata(self.grids[tar],\
                                                  self.tauspecs_target[tar],\
                                                  grid,\
                                                  method='linear',\
                                                  fill_value=0.)
                       for tar in self.targets_r.keys()]
        self.fluxes = np.array(self.fluxes)
        self.flux = np.sum(self.fluxes, axis=0)
        self.flux = np.exp(-1. * self.flux)
        
#        self.v0inds = {ion: self.corrinfo[ion]['v0ind'] for ion in ions}
#        self.lambda_rests = {ion: sp.lambda_rest['%smajor'%ion]\
#                                     if '%smajor'%ion in sp.lambda_rest.keys() \
#                                  else sp.lambda_rest[ion] \
#                                  for ion in ions}
#        # self.velperpix_kmps set in multiplet calculation       
#        
#        # set up grid to interpolate spectra to: minimal spacing
#        self.fluxes = {ion: self.spectra['%s/FluxCorr'%ion] for ion in ions}
#        self.lambda_vals = {ion: (1. + (np.arange(len(self.fluxes[ion])) - self.v0inds[ion]) \
#                                       * self.velperpix_kmps * 1e5 / cu.c.c) \
#                                 * self.lambda_rests[ion] \
#                            for ion in ions}     
#        self.lambda_range = [min([self.lambda_vals[ion][0] for ion in ions]), max([self.lambda_vals[ion][-1] for ion in ions])]
#        self.lambda_diff = min([np.min(np.diff(self.lambda_vals[ion])) for ion in ions]) / 2.
#        self.lambda_grid = np.arange(self.lambda_range[0], self.lambda_range[1] + self.lambda_diff, self.lambda_diff)
#        
#        
#        self.fluxes_grid = [scipy.interpolate.griddata(self.lambda_vals[ion],\
#                                                            self.fluxes[ion],\
#                                                            self.lambda_grid,\
#                                                            method='linear',\
#                                                            fill_value=1.) \
#                            for ion in ions]
#        self.fluxes_grid = np.array(self.fluxes_grid)
#        self.flux = np.prod(self.fluxes_grid, axis=0)
        self.spectra[name]['flux'] = self.flux
        self.spectra[name][spacename] = grid
        self.spectra[name]['lines'] = self.targets_r.keys() 
           
        if self.close_ofile:
            self.file.close()
            self.ofile = None
        del self.close_ofile
        #del self.v0inds
        #del self.lambda_rests   
        #del self.lambda_vals
        #del self.lambda_range
        #del self.lambda_diff
        #del self.lambda_grid
        #del self.fluxes_grid
        del self.flux
        del self.fluxes
        del self.vvals_kmps
        del self.tauspecs_orig
        del self.tauspecs_target
        del self.targets
        del self.targets_r
        del self.origs
        del self.grids
        # keep velperpix_kmps: seems generally useful, should not cause errors
        
        
    def getEW(self, ion, corr=False, vrange=None, vvals=None):
        '''
        returns EWs in mA; make sure any corrected fluxes match the last used
        set of lines
        uses the wavelength of only one line (major line), so only a good idea
        for single lines or close doublets
        '''
        if corr:
            key = '%s/FluxCorr'%ion
            if key not in self.spectra.keys():
                self.get_multiplet_flux(ion)
            if vrange is not None:
                try:
                    vrange[1] += max(max(self.corrinfo[ion]['veldiff_kmps']), 0.) 
                    vrange[1] %= (vvals[-1] + np.average(np.diff(vvals)))  
                except ValueError: # will not be set if correction was true, but only the major line was used
                    pass
                try:
                    vrange[0] += min(min(self.corrinfo[ion]['veldiff_kmps']), 0.) 
                    vrange[0] %= (vvals[-1] + np.average(np.diff(vvals)))  
                except ValueError:
                    pass
        else:
            key = '%s/Flux'%ion
            if key not in self.spectra.keys():
                self.getspectrum(ion)
        
        
        EW = self.spectra[key] # normalised flux
        if vrange is not None:
            if vrange[0] <= vrange[1]:
                vsel = np.logical_and(vvals >= vrange[0], vvals < vrange[1])
            else:
                vsel = np.logical_or(vvals >= vrange[0], vvals < vrange[1])
            EW = EW[vsel]
            zfrac = float(np.sum(vsel))/float(len(vsel))
            EW = 1. - np.sum(EW)/float(np.sum(vsel)) # total absorbed fraction
            try:
                EW *= self.deltaredshift * zfrac * lambda_rest[ion]
            except KeyError:
                EW *= self.deltaredshift * zfrac * lambda_rest[ild.get_major(ion)]
        else:        
            EW = 1. - np.sum(EW)/float(len(EW)) # total absorbed fraction
            try:
                EW *= self.deltaredshift*lambda_rest[ion] # convert absorbed flux fraction to EW
            except KeyError:
                EW *= self.deltaredshift*lambda_rest[ild.get_major(ion)]
        EW /= (self.cosmopars['z'] + 1.) # convert to rest-frame EW
        return EW * 1e3 # A -> mA
    
    def getcoldens_from_nion(self, ion, prange=None, pvals=None):
        key = getspecname(ion, 'realnionw/NIon_CM3')
        if key not in self.spectra.keys():
            self.getspectrum(ion)       
        nion = self.spectra[key] # normalised flux
        if prange is not None:
            if prange[0] <= prange[1]:
                psel = np.logical_and(pvals >= prange[0], pvals < prange[1])
            else:
                psel = np.logical_or(pvals >= prange[0], pvals < prange[1])
            nion = nion[psel]
        Nion = np.sum(nion) * self.slicelength * cu.c.cm_per_mpc / (1. + self.cosmopars['z']) / float(len(pvals))  # integrate nion over *proper* length
        return Nion

    def getcoldens_from_tau(self, ion, vvals, vrange=None):
        key = getspecname(ion, 'tau')
        if key not in self.spectra.keys():
            self.getspectrum(ion, 'tau')       
        tau = self.spectra[key] # normalised flux
        # uses only the strongest line -> take 'major' value, if available
        majkey =  '%smajor'%ion
        if majkey in sp.lambda_rest.keys():
            ionkey = majkey
        else:
            ionkey = ion
        #tau_to_nion = (np.pi* cu.c.electroncharge**2 /(cu.c.electronmass*cu.c.c**2))**-1 * 1./(sp.fosc[ionkey]*(sp.lambda_rest[ionkey] * 1.e-8)**2) # 1e-8 = Angstrom/cm
        #nion = tau * tau_to_nion
        nion = sp.Nion_from_tau(tau, ionkey)
        if vrange is not None:
            if vrange[0] <= vrange[1]:
                vsel = np.logical_and(vvals >= vrange[0], vvals < vrange[1])
            else:
                vsel = np.logical_or(vvals >= vrange[0], vvals < vrange[1])
            nion = nion[vsel]
            nion *=  1./cu.Hubble(self.cosmopars['z'], self.cosmopars) * 1.e-5  # times dp/dv (p in pcm, v in pkm/s)
        # nion spectrum is in v units -> integrate over v    
        Nion = np.sum(nion) * self.slicelength * cu.c.cm_per_mpc/ (1. + self.cosmopars['z']) / float(len(vvals)) # integrate nion over *proper* length [cgs]; dx/pixel = length per pixel 
        return Nion
        
    def getcomponents(self, ion, vvals, decr_lim=0.01, vsep_edge_group=200., nion_decr_lim=1e-6, vmargin_psearch=2., vmargin_psearch_min_kmps=200., mintol_group_sep=0.05, saveto_file=None):
        '''
        Find vspace absorption systems and corresponding real space nion 
        spectrum absorbers
        single-line values used

        decr_lim:        flux decrement threashold for what is considered 
                         part of an absorber
        vsep_edge_group: maximum velocity difference between components to be
                         considered one group (rest-frame, km/s)
        nion_decr_lim:   similar to decr_lim but for nion; value is relative to
                         absorption group max
        vmargin_psearch: multiplier for the tau-weighted peculiar velocity to
                         define region in which to search for counterparts in
                         position space
        saveto_file:     if given, open hdf5 file to which to save the results
        
        mintol_group_sep:to separate components that overlap with 
                         decr_lim * max decrement, but not within decr_lim, the 
                         closest pixels within 0.05 times the minimum decrement 
                         between the decr_lim edges become the new edges
        '''
        # set up dictionary to store outcomes
        if not hasattr(self, 'componentdct'):
            self.componentdct = {}
        self.componentdct[ion] = {}
        if saveto_file is not None:
            self.propskey = 'FluxDecrementLimit%s_MaxVSepAbsorptionSystem%s-pKmpS_NionDecrementLimit%s_VMarginPSearch%s_MinVMarginPSearch%s-pKmpS_MinToleranceGroupSeparation%s'%(decr_lim, vsep_edge_group, nion_decr_lim, vmargin_psearch, vmargin_psearch_min_kmps, mintol_group_sep)
            self.speckey  = 'Spectrum%i'%(self.specnum)
            self.fullkey  = 'SpectrumComponents/%s/%s'%(self.propskey, self.speckey)
            self.unitkey  = 'SpectrumComponents/Units'
            if self.fullkey not in saveto_file.keys():
                self.group = saveto_file.create_group(self.fullkey)
            else:
                self.group = saveto_file[self.fullkey]
            if self.unitkey not in saveto_file.keys():
                self.unitgroup = saveto_file.create_group(self.unitkey)
                self.unitgroup.attrs.create('SliceIndicesVelocitySpace', 'indices of the first and 1 + last elements in each absorption system in velocity space'.encode('utf8'))
                self.unitgroup.attrs.create('SystemEdgesVelocitySpace', 'minimum and maximum velocity (observed km/s) of each absorption system in velocity space'.encode('utf8'))

        # get the spectra and x values needed
        self.vsep_edge_group = vsep_edge_group # vvals are local velocity, which is what should be used between redshifts for consistency
        #self.pvals = vvals*1.e5 # rest-frame in km/s -> rest-frame velocity in cm/s
        #self.pvals /= cu.Hubble(self.cosmopars['z'],cosmopars=sample.cosmopars) * cu.c.cm_per_mpc # pMpc 
        #self.pvals *= (1. + self.cosmopars['z']) # cMpc
        self.fluxkey = getspecname(ion, 'Flux')
        if self.fluxkey not in self.spectra.keys():
            self.getspectrum(ion, 'Flux')
        self.flux = self.spectra[self.fluxkey]
        self.nionkey = getspecname(ion,'realnionw/NIon_CM3')
        if self.nionkey not in self.spectra.keys():
            self.getspectrum(ion, 'realnionw/NIon_CM3')
        self.nion = self.spectra[self.nionkey]
        self.vtaukey = getspecname(ion,'tauw/vpec')
        if self.vtaukey not in self.spectra.keys():
            self.getspectrum(ion, 'tauw/vpec')
        self.vtau = self.spectra[self.vtaukey]
        self.vionkey = getspecname(ion,'realnionw/vpec')
        if self.vionkey not in self.spectra.keys():
            self.getspectrum(ion, 'realnionw/vpec')
        self.vion = self.spectra[self.vionkey]

        ### get velocity-space components
        self.pixsize_v = np.average(np.diff(vvals))
        self.pixsep_edge_group = self.vsep_edge_group/self.pixsize_v # max number of pixels between components considered one absorber        
        self.absorber_indices = np.where(self.flux <= 1. - decr_lim)[0]
        if len(self.absorber_indices) > 2: # any absorbers found with > 1 pixel
            self.absorber_largeseps = np.where(np.diff(self.absorber_indices) > self.pixsep_edge_group)[0] # i in here -> absorber indices i, i+1 separated more than cutoff
            if len(self.absorber_largeseps) > 0: # >1 component
                # add 1 to left edges -> can use slicing to get in-absorber values
                self.edgeinds_v = [[np.min(self.absorber_indices), self.absorber_indices[self.absorber_largeseps[self.sepind]] + 1] if self.sepind == 0 else\
                                   [self.absorber_indices[self.absorber_largeseps[self.sepind - 1] + 1], np.max(self.absorber_indices) + 1] if self.sepind == len(self.absorber_largeseps) else\
                                   [self.absorber_indices[self.absorber_largeseps[self.sepind - 1] + 1], self.absorber_indices[self.absorber_largeseps[self.sepind]] + 1]\
                                   for self.sepind in range(len(self.absorber_largeseps) + 1)]
                del self.sepind
            else: # only one component -> just use min/max found values (same as 1 pixel case)
                self.edgeinds_v = [[np.min(self.absorber_indices), np.max(self.absorber_indices) + 1]]
            del self.absorber_largeseps
        elif len(self.absorber_indices) > 0: # one component of just one pixel. works for both  
            self.edgeinds_v = [[np.min(self.absorber_indices), np.max(self.absorber_indices) + 1]]
        else:
            self.edgeinds_v = []
            print('Sightline %i has no absorbers at decrement threashold %.2e'%(self.specnum, decr_lim))
            self.componentdct[ion] = {'numsys': 0}
            if saveto_file is not None:
                self.group.attrs.create('NumAbsorptionSystems', 0)
            return None
        del self.absorber_indices
        # check if the first and last systems wrap around the edge, and are in fact one system
        if len(self.edgeinds_v) > 1:
            if self.edgeinds_v[0][0] - (self.edgeinds_v[-1][1] - (len(vvals) +1)) <= self.pixsep_edge_group:
                self.edgeinds_v[0][0] = self.edgeinds_v[-1][0]
                self.edgeinds_v = self.edgeinds_v[:-1] 
        elif len(self.edgeinds_v) == 1:
            # the whole spectrum is just one group
            if self.edgeinds_v[0][0] - (self.edgeinds_v[0][1] - (len(vvals) +1)) <= self.pixsep_edge_group:
                self.edgeinds_v = [[0, len(vvals)]] 
        self.numsys = len(self.edgeinds_v)
        self.egdeinds_v = np.array(self.edgeinds_v)
        print(self.edgeinds_v)
        self.edgeinds_v_new = []
        # extend found components to region enclosing a decrement limit of decr_lim * max decrement in component (this does not exclude overlap between absorbers, which may occur if the max absorption is clode to the limit)
        if self.numsys > 0:
            for self.sysind in range(self.numsys):
                self.edgeinds_v_c = self.edgeinds_v[self.sysind]
                # extend max:
                self.flux_temp = np.roll(self.flux, len(vvals) - 2 - self.edgeinds_v_c[1]) # last index corresponds to the last pixel in system at decr_lim (excludes buffer pixel, hence the -2 in stead of -1)
                if self.edgeinds_v_c[0] <= self.edgeinds_v_c[1]:
                    self.sellen = self.edgeinds_v_c[1] - self.edgeinds_v_c[0] # number of pixels in selection (excluding edgesinds_v[1] = buffer pixel)
                else:
                    self.sellen = self.edgeinds_v_c[1] + len(vvals) - self.edgeinds_v_c[0]
                self.fluxmin = np.min(self.flux_temp[-1*self.sellen:])
                self.decr_lim_temp = decr_lim*(1.-self.fluxmin) # insure same fraction of total absorption for the cutoff in every absorber: decr. lim relative to local minimum flux
                self.newmaxdiff = np.min(np.where(self.flux_temp >  1. - self.decr_lim_temp)[0]) # after roll, first index with less absorption after decr_lim cutoff = first index in array for which this holds. last included = first - 1  
                self.flux_temp = np.roll(self.flux_temp, len(vvals) - self.sellen) # roll first pixel at initial cutoff to first index in array
                self.newmindiff = len(vvals) - 1 - np.max(np.where(self.flux_temp >  1. - self.decr_lim_temp)[0]) 
                self.edgeinds_v_c[0] -= self.newmindiff
                self.edgeinds_v_c[1] += self.newmaxdiff
                self.edgeinds_v_c = np.array(self.edgeinds_v_c) % len(vvals)
                self.edgeinds_v_new += [self.edgeinds_v_c]
            del self.sellen
            del self.fluxmin
            del self.decr_lim_temp
            del self.newmaxdiff
            del self.newmindiff
            del self.flux_temp
            del self.edgeinds_v_c
            # check for and handle new component overlaps
            if self.numsys > 1:
                # see notes with sketches for edge overlaps; only checks overlap with adjacent systems, since this will determine the cut-off
                # overlap: if both overlap periodic edge, they must overlap each other. 
                # if one overlaps the periodic edge and the other does not, at least one of the starts must be smaller than one of the ends (this is sufficient because the edge overlaps constrain the relative positions)
                # if neither overlaps the periodic edge, each start must come before the other end (if one starts after the other ends, the systems cannot overlap since the start/end ordering is not affected by periodicity)
                self.sysinds_overlap = [self.sysind if self.edgeinds_v_new[self.sysind][1] < self.edgeinds_v_new[self.sysind][0]\
                                                       and self.edgeinds_v_new[(self.sysind + 1) % self.numsys][1] < self.edgeinds_v_new[(self.sysind + 1) %self.numsys][0] else\
                                        self.sysind if (self.edgeinds_v_new[self.sysind][1] > self.edgeinds_v_new[(self.sysind + 1) % self.numsys][0]\
                                                        or self.edgeinds_v_new[(self.sysind + 1) % self.numsys][1] > self.edgeinds_v_new[self.sysind][0])\
                                                       and np.logical_xor(self.edgeinds_v_new[self.sysind][1] >= self.edgeinds_v_new[self.sysind][0],\
                                                                          self.edgeinds_v_new[(self.sysind + 1) % self.numsys][1] >= self.edgeinds_v_new[(self.sysind + 1) % self.numsys][0]) else\
                                        self.sysind if self.edgeinds_v_new[self.sysind][1] > self.edgeinds_v_new[(self.sysind + 1) % self.numsys][0]\
                                                       and self.edgeinds_v_new[(self.sysind + 1) % self.numsys][1] > self.edgeinds_v_new[self.sysind][0]\
                                                       and np.logical_and(self.edgeinds_v_new[self.sysind][1] >= self.edgeinds_v_new[self.sysind][0],\
                                                                          self.edgeinds_v_new[(self.sysind + 1) % self.numsys][1] >= self.edgeinds_v_new[(self.sysind + 1) % self.numsys][0]) else\
                                        None\
                                        for self.sysind in range(self.numsys)]
                del self.sysind
                self.sysinds_overlap = set(self.sysinds_overlap)
                self.sysinds_overlap -= {None} # remove dummy entries
                for self.sysind0 in self.sysinds_overlap: # cutoff is where the minimum between the edges is (to a deviation of mintol_group_sep)
                    self.sysind1 = (self.sysind0 + 1) % self.numsys
                    self.orig_edge01 = self.edgeinds_v[self.sysind0][1]
                    self.orig_edge10 = self.edgeinds_v[self.sysind1][0]
                    if self.orig_edge10 >= self.orig_edge01: # space between original systems does not overlap the edge -> neither will the new narrower selection
                        self.maxflux = np.max(self.flux[self.orig_edges01 : self.orig_edges10])
                        self.cutval = 1. - (1. - self.maxflux) * (1. + mintol_group_sep) 
                        self.newedge01 = np.min(np.where(np.logical_and(self.flux > self.cutval, np.arange(len(vvals)) >= self.orig_edge01 - 1))[0]) # first value after the edge where the flux is decrement is below the cutoff
                        self.newedge10 = np.max(np.where(np.logical_and(self.flux > self.cutval, np.arange(len(vvals)) <= self.orig_edge10))[0]) + 1
                    else: # periodic boundary overlap in original space -> new values may or may not overlap the edge
                        self.maxflux = max(np.max(self.flux[self.orig_edges01:]), np.max(self.flux[:self.orig_edges10]))
                        self.cutval = 1. - (1. - self.maxflux) * (1. + mintol_group_sep) 
                        self.possiblepos = np.where(self.flux > self.cutval)[0]
                        self.newedge01ind = np.argmin((self.possiblepos - self.orig_edge01 -1) % len(vvals))
                        self.newedge01 = self.possiblepos[self.newedge01ind] % len(vvals)
                        self.newedge10ind = np.argmin((self.orig_edge10 - self.possiblepos) % len(vvals))
                        self.newedge10 = (self.possiblepos[self.newedge10ind] + 1) % len(vvals)
                    
                    self.edgeinds_v_new[self.sysind0][1] = self.newedge01
                    self.edgeinds_v_new[self.sysind1][0] = self.newedge10
                    
                    del self.possiblepos
                    del self.newedge01ind
                    del self.newedge10ind
                    del self.maxflux
                    del self.cutval
                    del self.sysind0
                    del self.sysind1
                    del self.orig_edge01
                    del self.orig_edge10
                del self.sysinds_overlap
                # end of if numsys > 1 (overlap check and handling)
            self.edgeinds_v = np.array(self.edgeinds_v_new)
            del self.edgeinds_v_new                
            self.edgevals_v = [[vvals[self.inds[0]], vvals[self.inds[1] - 1]] for self.inds in self.edgeinds_v]
            del self.inds
            print(self.edgeinds_v)
        # end of if numsys > 0 (edge reset based on max flux decrement in system)
        # record velocity-only data
        if saveto_file is not None:
            self.group.attrs.create('NumAbsorptionSystems', self.numsys)
            self.group.attrs.create('SpectrumLength', len(vvals))
            self.group.create_dataset('SliceIndicesVelocitySpace', data=self.edgeinds_v)
            self.group.create_dataset('SystemEdgesVelocitySpace', data=self.edgevals_v)
        self.componentdct[ion]['numsys']     = self.numsys
        self.componentdct[ion]['edgeinds_v'] = self.edgeinds_v
        self.componentdct[ion]['edgevals_v'] = self.edgevals_v
        
        ### get position-space equivalents
        # mark position space places by velocities: that's the vpec unit anyway
        if self.numsys > 0:
            self.edgeinds_p = []
            self.edgevals_p = []
            for self.sysind in range(self.numsys):
                self.edgev_c = self.edgevals_v[self.sysind]
                self.edgei_c = self.edgeinds_v[self.sysind]
                # stored values are pixel centres; for inclusion, we want pixel edges 
                self.edgev_c[0] -= 0.5 * self.pixsize_v
                self.edgev_c[1] += 0.5 * self.pixsize_v
                # get min/max peculiar velocity in the absorption system (velocity space)
                if self.edgei_c[0] <= self.edgei_c[1]:
                    self.vtaumin = np.min(self.vtau[self.edgei_c[0]:self.edgei_c[1]]) * vmargin_psearch
                    self.vtaumax = np.max(self.vtau[self.edgei_c[0]:self.edgei_c[1]]) * vmargin_psearch 
                else: # wraps around edge: different selection
                    self.vtaumin = min(np.min(self.vtau[self.edgei_c[0]:]), np.min(self.vtau[:self.edgei_c[1]])) * vmargin_psearch
                    self.vtaumax = max(np.max(self.vtau[self.edgei_c[0]:]), np.max(self.vtau[:self.edgei_c[1]])) * vmargin_psearch
                # impose min/max v search area
                self.vtaumin = min(self.vtaumin, -1.*vmargin_psearch_min_kmps)
                self.vtaumax = max(self.vtaumax, vmargin_psearch_min_kmps)
                # convert margins to pixels
                self.vtaumin = np.floor(self.vtaumin / self.pixsize_v)
                self.vtaumax = np.ceil(self.vtaumax / self.pixsize_v)
                self.pi_min = (self.edgei_c[0] + self.vtaumin) % len(vvals) # vtaumin < 0, generally
                self.pi_max = (self.edgei_c[1] + self.vtaumax) % len(vvals)
                # select indices for initial position space selection (max still has the `margin pixel' for direct selection plug-in)
                # and check where they end up in velocity space 
                # note that p selection may overlap the periodic boundary even if the v selection does not
                if self.pi_min <= self.pi_max:
                    self.pinds_init = np.round(np.arange(self.pi_min, self.pi_max-0.5, 1.), 0).astype(np.float) # all indices in intial selection
                    self.pinds = np.copy(self.pinds_init)
                    self.pinds += self.vion[int(np.floor(self.pi_min)):int(np.ceil(self.pi_max))] / self.pixsize_v # corresponding velocity space position (in pixel units)
                    self.pinds %= len(vvals) # just to be sure, in case of large peculiar velocities
                else: # selection wraps around edge
                    self.pinds_init = np.round(np.arange(self.pi_min, len(vvals) + self.pi_max - 0.5, 1.), 0).astype(np.float) # all indices in intial selection
                    self.pinds = np.copy(self.pinds_init)
                    self.pinds += np.array(list(self.vion[int(np.floor(self.pi_min)):]) + list(self.vion[:int(np.ceil(self.pi_max))])) / self.pixsize_v # corresponding velocity space position (in pixel units)
                    self.pinds %= len(vvals) # important!
                # select the pixels that end up in the absorber region in velocity space
                if self.edgei_c[0] <= self.edgei_c[1]:
                    self.pinds = self.pinds_init[np.logical_and(self.pinds < self.edgei_c[1] - 0.5, self.pinds >= self.edgei_c[0] - 0.5)] # 0.5 accounts for edges/center differences, edgei_c 1 has a +1 margin from the last actual included
                else:
                    self.pinds = self.pinds_init[np.logical_or(self.pinds < self.edgei_c[1] - 0.5, self.pinds >= self.edgei_c[0] - 0.5)]
                self.pinds %= len(vvals) # pinds_init were not `modularized'
                # apply the ion density selection
                self.pinds = np.round(self.pinds, 0).astype(np.int) #float coordinates in pixel units -> int indices
                print(self.pinds)
                self.maxnion = np.max(self.nion[self.pinds])
                self.minnion = self.maxnion * nion_decr_lim
                self.pinds = self.pinds[self.nion[self.pinds] >= self.minnion]
                if np.min(self.pinds) < self.pi_min and np.max(self.pinds) > self.pi_max - 1: # initial selection wrapped around the edges, and the selected pixels fall into both parts:
                    self.edgeinds_p += [np.array([np.min(self.pinds[self.pinds>=self.pi_min]), np.max(self.pinds[self.pinds<=self.pi_max-1]) + 1])]
                else: 
                    self.edgeinds_p += [np.array([np.min(self.pinds), np.max(self.pinds) + 1])]
                self.edgevals_p += [ (np.array(self.edgeinds_p[self.sysind]).astype(np.float) + np.array([0.5, -0.5])) / float(len(vvals)) * self.slicelength ]
            del self.pinds_init     
            del self.pinds
            del self.sysind
            del self.edgev_c
            del self.edgei_c
            del self.vtaumin
            del self.vtaumax
            del self.maxnion
            del self.minnion
            del self.pi_min
            del self.pi_max
        
            self.edgeinds_p = np.array(self.edgeinds_p)
            self.edgevals_p = np.array(self.edgevals_p)
            if saveto_file is not None:
                self.group.create_dataset('SliceIndicesPositionSpace', data=self.edgeinds_p)
                self.group.create_dataset('SystemEdgesPositionSpace', data=self.edgevals_p)
            self.componentdct[ion]['edgeinds_p'] = self.edgeinds_p
            self.componentdct[ion]['edgevals_p'] = self.edgevals_p
            del self.edgeinds_p
            del self.edgevals_p
        else: # no systems -> nothing to do
            pass
        del self.edgeinds_v
        del self.edgevals_v
        del self.numsys
        # clean-up from way at the start of the calculation
        del self.vion
        del self.vionkey
        del self.vtau
        del self.vtaukey
        del self.vsep_edge_group
        del self.pixsep_edge_group
        del self.pixsize_v
        del self.nion
        del self.nionkey
        del self.flux
        del self.fluxkey
        
        return self.componentdct[ion]
        
class SpecSample:
    '''
    contains sightlines, allows operations on multiple sightlines (sort of like a transposed version of specwiz_proc)
    not very efficient for large numbers of spectra; use specwiz_proc there (gets all spectra; slow read-in, but numpy arrays for all spectra at once)
    '''
    def __init__(self, filename, specnums=None):
        self.filename = filename
        self.sldict = {}
        self.getscalardata()
        if specnums is not None:
            self.addspecs(specnums)
        

    def getscalardata(self):
        self.file = h5py.File(self.filename, 'r')
        # box and cosmology
        self.cosmopars = {'boxsize':     self.file['/Header'].attrs.get('BoxSize'),\
                          'h':           self.file['/Header'].attrs.get('HubbleParam'),\
                          'a':           self.file['/Header'].attrs.get('ExpansionFactor'),\
                          'z':           self.file['/Header'].attrs.get('Redshift'),\
                          'omegam':      self.file['/Header'].attrs.get('Omega0'),\
                          'omegalambda': self.file['/Header'].attrs.get('OmegaLambda'),\
                          'omegab':      self.file['/Header'].attrs.get('OmegaBaryon') }
        # redshift space
        self.slicelength = self.cosmopars['boxsize']/self.cosmopars['h'] # in cMpc
        self.deltaredshift = cu.Hubble(self.cosmopars['z']) / cu.c.c*self.slicelength * cu.c.cm_per_mpc # Delta z (observed)
        # position
        self.fracpos = np.array([self.file['/Projection/x_fraction_array'], self.file['/Projection/y_fraction_array']])
        # ok, this is not a scalar, but we don't wan to get it for each spectrum
        self.velocity = np.array(self.file['VHubble_KMpS'])

    def addspecs(self, specnums):
        self.sldict = {specnum: Sightline(self.filename, specnum, ofile=self.file) for specnum in specnums}
    
    def getspectra(self, ion, qty=None, corr=False):
        return np.array([self.sldict[slkey].getspectrum(ion, qty=qty, corr=corr) for slkey in self.sldict.keys()])

    def matchspectra(self, tomatch):
        self.pos = np.array([np.array(self.file['Projection/x_fraction_array']), np.array(self.file['Projection/y_fraction_array'])]).T # array of x,y positions in this file
        self.tomatchpos = np.array([tomatch.sldict[key].fracpos for key in tomatch.sldict.keys()])
        self.matchmatrix = np.all(self.pos[np.newaxis,:,:]==self.tomatchpos[:,np.newaxis,:],axis=2)
        self.specnums = np.where(np.any(self.matchmatrix,axis=0))[0]
        self.addspecs(self.specnums)

    def close(self):
        self.file.close()
        
# main sample for specwizard-make-maps matching for the EWs and a subsample fixing a bug where T4EOS was not actually applied
#subsample = SpecSample(sdir + 'sample3_sub/spec.snap_027_z000p101.0.hdf5', specnums = np.arange(512))
# sample = SpecSample(sdir+'sample3/spec.snap_027_z000p101.0.hdf5')
#sample.matchspectra(subsample)
ionlabels = {'c3':   'C \, III', 'c4':   'C \, IV',    'c5':   'C \, V',\
             'c6':   'C \, VI',  'fe17': 'Fe \, XVII', 'n6':   'N \, VI',\
             'n7':   'N \, VII', 'ne8':  'Ne \, VIII', 'ne9':  'Ne \, IX',\
             'o4':   'O \, IV',  'o5':   'O \, V',     'o6':   'O \, VI',\
             'o7':   'O \, VII', 'o8':   'O \, VIII',  'si3':  'Si \, III',\
             'si4':  'Si \, IV'}
ions = ['c3', 'c4', 'c5', 'c6', 'fe17', 'n6', 'n7', 'ne8', 'ne9', 'o4', 'o5', 'o6', 'o7', 'o8', 'si3', 'si4']

#subsample_subset = SpecSample(sdir + 'sample3_sub/spec.snap_027_z000p101.0.hdf5', specnums=[50, 60, 124, 152, 178, 227])
#sample_subset = SpecSample(sdir+'sample3/spec.snap_027_z000p101.0.hdf5')
#sample_subset.matchspectra(subsample_subset)
#sample_sub = SpecSample(sdir+'sample3_sub_deltacorrtest/spec.snap_027_z000p101.0.hdf5')

def blazarflux(Eoutkev, Etotcgs=7.5e-12, Eminkev=2., Emaxkev=10., Gammaphot=2):
    '''
    dNphot/dE = A * E^-Gammaphot
    A set by total cgs energy flux (erg/cm2/s) from Emin to Emax
    does not handle Gammaphot = 1 case
    '''
    kev_to_erg = cu.c.ev_to_erg * 1.e3 # erg/kev
    if Gammaphot == 2:
        A = Etotcgs / np.log(Emaxkev / Eminkev)
    else:
        A = (2 - Gammaphot) * Etotcgs / ((Emaxkev * kev_to_erg)**(2-Gammaphot) - (Eminkev * kev_to_erg)**(2-Gammaphot))
    Nphotcgs = A * (Eoutkev * kev_to_erg)**(-1 * Gammaphot) # photons / cm2 / s / erg
    return Nphotcgs * kev_to_erg # photons/ cm2 / s / keV

def getphotonspec(spectrum, vsize_rf_cmps, z, ion, exposuretime_s, Aeff_cm2, **kwargs):
    try:
        eng_ion_kev = cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z)) / (1.e3 * cu.c.ev_to_erg) #(observed wavelength)
    except KeyError:
        eng_ion_kev = cu.c.c * cu.c.planck / (sp.lambda_rest['%smajor'%ion] * 1.e-8 * (1. + z)) / (1.e3 * cu.c.ev_to_erg) #(observed wavelength)
    pixsize_kev = vsize_rf_cmps / cu.c.c * eng_ion_kev
    photons_per_pix = blazarflux(eng_ion_kev, **kwargs) * pixsize_kev * exposuretime_s * Aeff_cm2
    photonspec = spectrum * photons_per_pix
    return photonspec, photons_per_pix

def getnoise(photonspec):
    return np.sqrt(photonspec)

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
    return np.reshape(outarray, outshape)

def smoothspectrum(spectrum, vvals_rf_kmps, ion, z, fwhm_ev, pix_per_fwhm=2.):
    try:
        eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg) # redshifted energy -> want observed energy given rf velocity 
    except KeyError:
        eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest['%smajor'%ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg) # redshifted energy -> want observed energy given rf velocity 
    pixsize_orig_ev = eng_ion_ev * np.average(np.diff(vvals_rf_kmps)) * 1.e5 / cu.c.c # vvals: km/s -> cm/s
    fwhm_pixunits = fwhm_ev / pixsize_orig_ev
    numsample_gaussian = (int(np.ceil(7. * fwhm_pixunits)) / 2) * 2 + 1 # 7 x hwhm away already has a 16 dex difference in y value 
    if len(spectrum) < numsample_gaussian:
        numsample_gaussian = (len(spectrum)/ 2) * 2 - 1
    gaussian_x = (np.arange(numsample_gaussian) - numsample_gaussian / 2).astype(np.float)
    gaussian_y = np.exp(-1*gaussian_x**2 * 4.*np.log(2.)/fwhm_pixunits**2)
    gaussian_y /= np.sum(gaussian_y)
    gaussian_y = np.append(gaussian_y, np.zeros(len(spectrum) - numsample_gaussian))
    spectrum_smoothed = np.fft.irfft(np.fft.rfft(spectrum) * np.fft.rfft(gaussian_y))
    spectrum_smoothed = np.roll(spectrum_smoothed, -1 * (numsample_gaussian / 2)) # center gets shifted by offset between zero index and peak of the gaussian
    downsample_pix = int(np.round(float(len(spectrum)) * pix_per_fwhm / fwhm_pixunits, 0))
    spectrum_downsampled = downsample(spectrum_smoothed, downsample_pix)
    vsize_pix_rf_kmps = np.average(np.diff(vvals_rf_kmps)) * float(len(spectrum)) / float(len(spectrum_downsampled))
    print('Downsample factor %s'%(float(len(spectrum)) / float(len(spectrum_downsampled))))
    return spectrum_downsampled, vsize_pix_rf_kmps


def getsurveyspectra(vvals_rf_kmps, spectrum, ion, z, instrument='Arcus'):
    '''
    Instrument: Arcus or Athena
    ion: o7 or o8
    '''
    # from latest (2018 - 07 - 31) Athena science requirements 
    if instrument == 'Athena':
        Aeff_cm2 = 1.05e4 # there are a few values mentioned; this seems like a reasonable guess (2nd entry for 1 keV; also the one on the X-IFU main website)
        fwhm_ev = 2.1 # Barret et al. 2018, Fig. 3, + Fabrizio Nicastro's response matrix data from the team. at 1 keV max value (website says 2.5 eV though, so I'm not sure now, also in 2018 paper. In earlier papers, 1.5 eV seemed to be a goal, 2.5 the requirement)
        exposuretime_s = 50.0e3
        # parameters used for the weak-line sensitivity limit
        Etotcgs = 1.0e-11
        Eminkev = 2.
        Emaxkev = 10. 
        Gammaphot = 1.8 
    # from fig. 1 table in Smith, Abraham, et al. 
    elif instrument == 'Arcus':
        try:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
            lambda_rest = sp.lambda_rest[ion]
        except KeyError:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest['%smajor'%ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
            lambda_rest = sp.lambda_rest['%smajor'%ion]
        if lambda_rest * (1. + z) > 21.6:
            fwhm_ev = eng_ion_ev/2500. # spec. res. 2500 at 21.6 - 28 A
        else:
            fwhm_ev = eng_ion_ev/2000. # spec. res. 2000 at 16 - 21.6 A
            
        Aeff_cm2 = 250.  # Fabrizio Nicastro
        exposuretime_s = 500.0e3
        Etotcgs = 7.5e-12
        Eminkev = 2.
        Emaxkev = 10. 
        Gammaphot = 2 # kind of a guess
    
    smallerspec, pixsize_rf_kmps = smoothspectrum(spectrum, vvals_rf_kmps, ion, z, fwhm_ev)
    photonspec, photons_per_pix = getphotonspec(smallerspec, pixsize_rf_kmps*1e5, z, ion, exposuretime_s, Aeff_cm2, Etotcgs=Etotcgs, Eminkev=Eminkev, Emaxkev=Emaxkev, Gammaphot=Gammaphot)
    noise = getnoise(photonspec)
    normphotonspec = photonspec / photons_per_pix
    normphotonnoise = noise / photons_per_pix
    return normphotonspec, normphotonnoise, pixsize_rf_kmps

def getstandardspectra(vvals_rf_kmps, spectrum, ion, z, instrument='Arcus', pix_per_fwhm=2.):
    '''
    Instrument: Arcus or Athena
    ion: o7 or o8
    '''
    
    exposuretime_s = 100.0e3
    # parameters used for the weak-line sensitivity limit for Athena
    Etotcgs = 1.0e-11
    Eminkev = 2.
    Emaxkev = 10. 
    Gammaphot = 1.8 
    if instrument == 'Athena':
        # from latest (2018 - 07 - 31) Athena science requirements 
        Aeff_cm2 = 1.05e4 # there are a few values mentioned; this seems like a reasonable guess (2nd entry for 1 keV; also the one on the X-IFU main website)
        fwhm_ev = 2.1 # Barret et al. 2018, Fig. 3, + Fabrizio Nicastro's response matrix data from the team. at 1 keV max value (website says 2.5 eV though, so I'm not sure now, also in 2018 paper. In earlier papers, 1.5 eV seemed to be a goal, 2.5 the requirement)
        
    # from fig. 1 table in Smith, Abraham, et al. 
    elif instrument == 'Arcus':
        try:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
            lambda_rest = sp.lambda_rest[ion]
        except KeyError:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest['%smajor'%ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
            lambda_rest = sp.lambda_rest['%smajor'%ion]
        if lambda_rest * (1. + z) > 21.6:
            fwhm_ev = eng_ion_ev/2500. # spec. res. 2500 at 21.6 - 28 A
        else:
            fwhm_ev = eng_ion_ev/2000. # spec. res. 2000 at 16 - 21.6 A
            
        Aeff_cm2 = 250. # Fabrizio Nicastro
    
    elif instrument == 'Lynx':
        try:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
        except KeyError:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest['%smajor'%ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)

        fwhm_ev = eng_ion_ev/5000.
        Aeff_cm2 = 4.e3    
    
    elif instrument == 'proposed-FN-JK':
        try:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
        except KeyError:
            eng_ion_ev = cu.c.c * cu.c.planck / (sp.lambda_rest['%smajor'%ion] * 1.e-8 * (1. + z)) / (cu.c.ev_to_erg)
        
        fwhm_ev = eng_ion_ev/10000.
        Aeff_cm2 = 1.5e3 
    
    smallerspec, pixsize_rf_kmps = smoothspectrum(spectrum, vvals_rf_kmps, ion, z, fwhm_ev, pix_per_fwhm=pix_per_fwhm)
    photonspec, photons_per_pix = getphotonspec(smallerspec, pixsize_rf_kmps*1e5, z, ion, exposuretime_s, Aeff_cm2, Etotcgs=Etotcgs, Eminkev=Eminkev, Emaxkev=Emaxkev, Gammaphot=Gammaphot)
    noise = getnoise(photonspec)
    normphotonspec = photonspec / photons_per_pix
    normphotonnoise = np.array([normphotonspec - noise / photons_per_pix, normphotonspec + noise / photons_per_pix])
    return photonspec, normphotonnoise, pixsize_rf_kmps, photons_per_pix

    
def select_random_sightlines_by_coldens(ion, lognmin, lognmax, number, losfilename):
    '''
    returns <number> random sightlines from the uniform selection in ion
    with a column density in the lognmin, lognmax range (log10 cm^-2)
    '''
    ofile = h5py.File(losfilename, 'r')
    infiles = ofile['Header'].attrs['input_files']
    fileind = np.where([ion in infile for infile in infiles])[0][0]
    groupname = 'file%i'%(fileind)
    print('Using group %s'%groupname)
    # select pixels in ion sample based on given values
    ionsample = np.array(ofile['Selection/%s/selected_values_thision'%groupname])
    ionsampleinds = np.where(np.logical_and(ionsample >= lognmin, ionsample < lognmax))[0]
    ionsampleindschoice = np.random.choice(ionsampleinds, size=number, replace=False)
    pixelschoice = np.array(ofile['Selection/%s/selected_pixels_thision'%groupname])[ionsampleindschoice]
    # match pixels to toal sample to retrieve total sample indices
    pixelsall = np.array(ofile['Selection/selected_pixels_allions'])
    indsall = np.where(np.any(np.all(pixelsall[:, np.newaxis, :] == pixelschoice[np.newaxis, :, :], axis=2), axis=1))[0]
    return indsall

def getselections_by_coldens(dct, number, losfilename):
    '''
    dct: ion: array (numbins x 2): mins, maxs 
    number: per ion per bin
    '''
    return {ion: [select_random_sightlines_by_coldens(ion, dct[ion][ind][0], dct[ion][ind][1], number, losfilename) for ind in range(len(dct[ion]))] for ion in dct.keys()}

def checkT4EOS_sample3sub_coldenscomp(sample_in=None, subsample_in=None):
# 16 ions
    if sample_in is None:
        sample = sample
        samplename = 'sample3'
    else:
        sample = sample = SpecSample(sdir+sample_in)
        samplename = sample_in.split('/')[0]
    if subsample_in is None:
        subsample = subsample
        subsample_name = 'subsample3'
    else:
        subsample = SpecSample(sdir + subsample_in, specnums = np.arange(512))
        subsamplename = subsample_in.split('/')[0]
    sample.matchspectra(subsample)

    # 16 ions
    fig, axes = plt.subplots(nrows = 4, ncols =4, figsize = (12.,10.))
    fontsize = 12
    ions = ['c3', 'c4', 'c5', 'c6', 'fe17', 'n6', 'n7', 'ne8', 'ne9', 'o4', 'o5', 'o6', 'o7', 'o8', 'si3', 'si4']
    clist = ['red','orange','gold','green','blue', 'purple', 'cyan', 'fuchsia', 'maroon', 'brown', 'olive', 'midnightblue', 'teal', 'chocolate', 'line', 'darksalmon']
    if subsample_in is None and sample_in is None:
        fig.suptitle(r'Column densities in subsample3: with and without ISM gas at $10^4 \, \mathrm{K}$',fontsize=fontsize+2)
    else:
        fig.suptitle(r'Column densities in %s (sb) compared to %s (sa)'%(subsamplename, samplename),fontsize=fontsize+2)
    
    for ion in ions:
        subsample.getspectra(ion,'coldens')
        sample.getspectra(ion,'coldens')

    keys = np.where(sample.matchmatrix) # 2nd array = indices for sample, 1st array = corresponding indices for subsample; indices  = specnums
    speckeys = [getspecname(ion, 'coldens') for ion in ions] 
    # spectra with extreme values for the differences in each ion
    inds_maxdiff = [np.argmax(np.abs(\
                                     np.array([ subsample.sldict[numkey].spectra[speckey] for numkey in keys[0] ]) -\
                                     np.array([    sample.sldict[numkey].spectra[speckey] for numkey in keys[1] ])\
                                     )) for speckey in speckeys]
    # got: [60, 50, 227, 227, 227, 227, 227, 227, 227, 227, 178, 152, 227, 124, 60, 60]
    inds_maxdiff = list(set(inds_maxdiff)) #unique values only
    inds_maxdiff.sort() # for consistency between runs
    colors = np.array(list(('gray',)*len(subsample.sldict)),dtype = '|S10') # set dtype to avoid cutting off longer color names later
    colors[np.array(inds_maxdiff)] = np.array(clist)[:len(inds_maxdiff)]
    
    if subsample_in is None and sample_in is None:
        xlab = r'$\log_{10}\, N_{\mathrm{T4EOS}} \, [\mathrm{cm^{-2}}]$' #r'$\log_{10}\, N_{\mathrm{%s}} \, [\mathrm{cm^{-2}}]$'%(ionlab)
        ylab = r'$\log_{10}\, N_{\mathrm{T4EOS}}\,/\,N_{\mathrm{T, fv}}$' #r'$\log_{10}\, N_{\mathrm{%s}, \mathrm{T4EOS}}\,/\,N_{\mathrm{%s}, \mathrm{T, fv}}$'%(ionlab,ionlab)
    else:
         xlab = r'$\log_{10}\, N, \mathrm{sb} \, [\mathrm{cm^{-2}}]$'
         ylab = r'$\log_{10}\, N_{\mathrm{sb}}\,/\,N_{\mathrm{sa}}$'

    for ind in range(16):
        ion = ions[ind]
        ax = axes[ind/4,ind%4]
        ionlab = ionlabels[ion]
        

        # retrieve ion spectra; don't just use the output lists for SpecSample col. dens. values, since the orders for the two samples may not match there      
        speckey = getspecname(ion, 'coldens')
        coldens_sub = np.array([subsample.sldict[numkey].spectra[speckey] for numkey in keys[0]])
        coldens_main = np.array([sample.sldict[numkey].spectra[speckey] for numkey in keys[1]])

        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', color = 'black')

        ax.axhline(0., linestyle='dashed', color='black')
        ax.scatter(coldens_sub,coldens_sub-coldens_main, alpha = 0.3, c=colors)
        if ind/4 == 3:
            ax.set_xlabel(xlab, fontsize=fontsize)
        if ind%4 == 0:
            ax.set_ylabel(ylab, fontsize=fontsize)
        ax.text(0.05,0.05, r'$\mathrm{%s}$'%ionlab, fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
    if subsample_in is None and sample_in is None:
        plt.savefig(mdir + 'coldens_specwizard_sample3-facevalueT_vs_subsample3-T4EOS.pdf' ,format = 'pdf',bbox_inches='tight')
    else:
        plt.savefig(mdir + 'coldens_specwizard_%s_vs_%s.pdf'%(samplename, subsamplename) ,format = 'pdf',bbox_inches='tight')
           

def checkT4EOS_sample3sub_spectrumcomp(specnum_sub, sample_in=None, subsample_in=None):
    '''
     position values are actually pMpc
    '''
# 16 ions
    if sample_in is None:
        sample =  sample
        samplename = 'sample3'
    else:
        sample = sample = SpecSample(sdir+sample_in)
        samplename = sample_in.split('/')[0]
    if subsample_in is None:
        subasample = subsample
        subsample_name = 'subsample3'
    else:
        subsample = SpecSample(sdir + subsample_in, specnums = np.arange(512))
        subsamplename = subsample_in.split('/')[0]
    sample.matchspectra(subsample)

    fig, axes = plt.subplots(nrows = 17, ncols =6, figsize = (25.,10.), gridspec_kw = {'top': 0.95, 'bottom':0.05, 'left':0.05, 'right':0.95, 'hspace':0.0, 'wspace':0.15, 'width_ratios': [1.,1.,1.,1.,1.,1.], 'height_ratios':[1.5]+list((1.,)*16)})
    fontsize = 12
    ions = ['c3', 'c4', 'c5', 'c6', 'fe17', 'n6', 'n7', 'ne8', 'ne9', 'o4', 'o5', 'o6', 'o7', 'o8', 'si3', 'si4']
    clist = ['red','orange','gold','green','blue', 'purple', 'cyan', 'fuchsia', 'maroon']
    #fig.suptitle(r'spectra in subsample3: with and without ISM gas at $10^4 \, \mathrm{K}$',fontsize=fontsize+2)

    keys = np.where(sample.matchmatrix) # 2nd array = indices for sample, 1st array = corresponding indices for subsample; indices  = specnums
    speckeys = [getspecname(ion, 'coldens') for ion in ions] 
    key_sub  = keys[0][specnum_sub]
    key_main = keys[1][specnum_sub]
     
    # got inds_maxdiff: [60, 50, 227, 227, 227, 227, 227, 227, 227, 227, 178, 152, 227, 124, 60, 60]
    inds_maxdiff_colormatch = {50: 'red', 60:'orange', 124: 'gold', 152: 'green', 178: 'blue', 227: 'purple'}
    
    if key_sub in inds_maxdiff_colormatch.keys():
        color = inds_maxdiff_colormatch[key_sub]
    else:
        color = 'gray'

    if sample_in is None and subsample_in is None:
        fig.suptitle(r'Spectrum %i in subsample3: with and without ISM gas at $10^4 \, \mathrm{K}$'%(key_sub), fontsize=fontsize+2, color=color)
    else:
        fig.suptitle(r'Spectrum %i in %s, compared to %s'%(key_sub, subsamplename, samplename), fontsize=fontsize+2, color=color)

    vvals_sub = subsample.velocity
    vvals_main = sample.velocity
    zvals_sub = vvals_sub*1.e5/cu.Hubble(subsample.cosmopars['z'],cosmopars=subsample.cosmopars) / cu.c.cm_per_mpc
    zvals_main = vvals_main*1.e5/cu.Hubble(sample.cosmopars['z'],cosmopars=sample.cosmopars) / cu.c.cm_per_mpc

    for ion in ions:
        subsample.getspectra(ion,'coldens')
        sample.getspectra(ion,'coldens')
        subsample.getspectra(ion,'Flux')
        sample.getspectra(ion,'Flux')
        subsample.getspectra(ion,'tauw/Temperature_K')
        sample.getspectra(ion,'tauw/Temperature_K')
        subsample.getspectra(ion,'realnionw/Temperature_K')
        sample.getspectra(ion,'realnionw/Temperature_K')
        subsample.getspectra(ion,'realnionw/NIon_CM3')
        sample.getspectra(ion,'realnionw/NIon_CM3')
        subsample.getspectra(ion,'realnionw/vpec')
        sample.getspectra(ion,'realnionw/vpec')

    # what to plot by spectrum quantity
    spectypes = ['Flux', 'tauw/Temperature_K', 'realnionw/NIon_CM3',  'realnionw/Temperature_K', 'realnionw/vpec']
    velsp = [True, True, False, False, False] 
    takelog = [False, True, True, True, False]
    typekeys = [[getspecname(ion, spectype) for spectype in spectypes] for ion in ions]
    xlabels = ['v [km/s]',       'v [km/s]', 'z [cMpc]', 'z [cMpc]', 'z [cMpc]']
    ylabels = ['normalised flux', r'$\log_{10} T_{\tau} \, [\mathrm{K}]$', r'$\log_{10} n_{\mathrm{ion}} \, [\mathrm{cm}^{-3}]$', r'$\log_{10} T_{\mathrm{ion}} \, [\mathrm{K}]$', r'$v_{z, \mathrm{ion}} \, [\mathrm{km}\, \mathrm{s}^{-1}]$']

    # set up limit trackers for setting x/y limits consistently
    maxlims_T = [np.inf, -np.inf] # to set both temperatures to the same range, across all ions
    maxlims_v = [np.inf, -np.inf] # to set all ion velocities to the same range
    maxinds_x = [np.inf, -np.inf] # to set positions and velocities to the same range (Hubble flow conversion, i.e. match indices)
    # ion densities: just set to some dynamic range (different ions will differ by abundance as well as line strength)
    dynrang_d = 5.
    # fraction of max absorption/ion density that we want to include for each ion (fixed-size margins also applied)
    # assumes main/sub z and v values are the same
    edgedef_l = 0.2
    vmargin = 250 # margin on x ranges defined by absorption fractions in km/s
 
    mcol = 'midnightblue'
    scol = 'darkgoldenrod'

    ## make the plots, and keep track of min/max ranges needed later
    for genind in range(16*5):
        ionind = genind/5
        typeind = genind%5
        #print('ionind: %i, typeind: %i, typekeys shape: (%i, %i)'%(ionind,typeind, len(typekeys), len(typekeys[ionind])))
        ax = axes[ionind + 1, typeind]
        typekey = typekeys[ionind][typeind]
          
        if velsp[typeind]:
            xsub = vvals_sub
            xmain = vvals_main
        else:
            xsub = zvals_sub
            xmain = zvals_main
        if takelog[typeind]:
            ysub = np.log10(subsample.sldict[key_sub].spectra[typekeys[ionind][typeind]])
            ymain = np.log10(sample.sldict[key_main].spectra[typekeys[ionind][typeind]])
        else:
            ysub = subsample.sldict[key_sub].spectra[typekeys[ionind][typeind]]
            ymain = sample.sldict[key_main].spectra[typekeys[ionind][typeind]]

        doylabels = False
        if ionind == 15:
            ax.set_xlabel(xlabels[typeind], fontsize=fontsize)
            doylabels = True
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both', labelbottom = doylabels, color = 'black')
        ax.plot(xsub,  ysub,  color = scol)
        ax.plot(xmain, ymain, color = mcol)

        # do limit checking and accounting
        if 'Temperature' in typekeys[ionind][typeind]:
            maxlims_T[0] = min(maxlims_T[0], np.min(ysub[np.isfinite(ysub)]), np.min(ymain[np.isfinite(ymain)]))
            maxlims_T[1] = max(maxlims_T[1], np.max(ysub[np.isfinite(ysub)]), np.max(ymain[np.isfinite(ymain)]))

        if 'PeculiarVelocity' in typekeys[ionind][typeind]:
            maxlims_v[0] = min(maxlims_v[0], np.min(ysub), np.min(ymain))
            maxlims_v[1] = max(maxlims_v[1], np.max(ysub), np.max(ymain))

        if 'NIon_CM3' in typekeys[ionind][typeind]:
            maxval = max(np.max(ysub), np.max(ymain))
            ax.set_ylim(maxval-dynrang_d, maxval)

            # where is the ion density above edgedef_l * max value? (plotted values are log -> take log of threashold for comparison)
            incl = np.where(np.any( np.array([ysub >= np.log10(edgedef_l)+np.max(ysub), ymain >= np.log10(edgedef_l)+np.max(ymain) ]), axis=0))[0]
            maxinds_x[0] = min(maxinds_x[0], np.min(incl))
            maxinds_x[1] = max(maxinds_x[1], np.max(incl))

        if 'Flux' in typekeys[ionind][typeind]:
            # where is the flux decrement > edgedef_l * max value?
            incl = np.where(np.any( np.array([(1.-ysub) >= edgedef_l*np.max(1.-ysub), (1.-ymain) >= edgedef_l*np.max(1.-ymain) ]), axis=0))[0]
            maxinds_x[0] = min(maxinds_x[0], np.min(incl))
            maxinds_x[1] = max(maxinds_x[1], np.max(incl))
    
    ## set z/v range margin (no wrap-around here)
    indsmargin = int(np.ceil( vmargin/(vvals_sub[-1]-vvals_sub[0])*(len(vvals_sub)-1) ))
    maxinds_x[0] = max(maxinds_x[0]-indsmargin, 0) 
    maxinds_x[1] = min(maxinds_x[1]+indsmargin, len(vvals_sub)-1)
    
    ## loop back over the plots and apply the x/y ranges we just got
    for genind in range(16*5):
        ionind = genind/5
        typeind = genind%5
        #print('ionind: %i, typeind: %i, typekeys shape: (%i, %i)'%(ionind,typeind, len(typekeys), len(typekeys[ionind])))
        ax = axes[ionind + 1, typeind]
        typekey = typekeys[ionind][typeind]

        if velsp[typeind]:
            xsub = vvals_sub
            xmain = vvals_main
        else:
            xsub = zvals_sub
            xmain = zvals_main
        ax.set_xlim(xsub[maxinds_x[0]],xsub[maxinds_x[1]])

        if 'Temperature' in typekeys[ionind][typeind]:
            ax.set_ylim(maxlims_T[0],maxlims_T[1])

        if 'PeculiarVelocity' in typekeys[ionind][typeind]:
            ax.set_ylim(maxlims_v[0],maxlims_v[1])

            

    ## set the ion information in the rightmost plot
    for ionind in range(16):
        ax = axes[ionind +1, 5]
        ax.axis('off')
        ion = ions[ionind]
       
        coldens_sub = subsample.sldict[key_sub].spectra[getspecname(ion,'coldens')]   
        coldens_main = sample.sldict[key_main].spectra[getspecname(ion,'coldens')]
     
        if ion + 'major' in lambda_rest.keys():
            lambdakey = ion + 'major'
        else:
            lambdakey = ion

        EW_sub = subsample.sldict[key_sub].spectra[getspecname(ion,'Flux')] # normalised flux
        EW_sub = 1. - np.sum(EW_sub)/float(len(EW_sub)) # total absorbed fraction
        EW_sub *= subsample.deltaredshift*lambda_rest[lambdakey] # convert absorbed flux fraction to EW
        EW_sub /= (subsample.cosmopars['z']+1.) # convert to rest-frame EW

        EW_main = sample.sldict[key_main].spectra[getspecname(ion,'Flux')] # normalised flux
        EW_main = 1. - np.sum(EW_main)/float(len(EW_main)) # total absorbed fraction
        EW_main *= sample.deltaredshift*lambda_rest[lambdakey] # convert absorbed flux fraction to EW
        EW_main /= (sample.cosmopars['z']+1.) # convert to rest-frame EW

        ax.text(0.,0.,    r'$\mathrm{%s}$'%(ionlabels[ion]), fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
        ax.text(0.25, 0., '$%.1f$  $%.1f$  $%.2f$  $%.2f$'%(coldens_sub, coldens_main, np.log10(EW_sub), np.log10(EW_main)), fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')
    
    # top right 'legend panel'
    ax = axes[0,5]
    ax.axis('off')
    if sample_in is None and subsample_in is None:
        ax.text(0.25,0.35,r'$N$: $\log_{10} \mathrm{cm}^{-2}$, $EW$: $\log_{10} \mathrm{\AA}$' , fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')   
        ax.text(0.25, 0., r'$N_\mathrm{T4}$    $N_\mathrm{Tfv}$    $EW_\mathrm{T4}$   $EW_\mathrm{Tfv}$' , fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left') 
        ax.text(0.0,1., r'SF at $10^4 \, \mathrm{K}$ (T4)', color=scol, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
        ax.text(0.45,1., r'SF at $T_{\mathrm{out}}$ (Tfv)', color=mcol, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
    else:
        ax.text(0.25,0.35,r'$N$: $\log_{10} \mathrm{cm}^{-2}$, $EW$: $\log_{10} \mathrm{\AA}$' , fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left')   
        ax.text(0.25, 0., r'$N_\mathrm{sa}$    $N_\mathrm{sb}$    $EW_\mathrm{sa}$   $EW_\mathrm{sb}$' , fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left') 
        ax.text(0.0,1., r'%s (sb)'%subsamplename, color=scol, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
        ax.text(0.45,1., r'%s (sa)'%samplename, color=mcol, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left') 
    # top row: y axis labels
    for typeind in range(5):
        ax = axes[0,typeind]
        ax.axis('off')
        ax.text(0.5,0.15, ylabels[typeind] , fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='center', bbox=dict(facecolor='white',alpha=0.3))
    
    if sample_in is None and subsample_in is None:
        plt.savefig(mdir + 'spectra_specwizard_sample3-facevalueT_vs_subsample3-T4EOS_flux-T-nion-vpec_spectrum%i.pdf'%(key_sub) ,format = 'pdf',bbox_inches='tight')
    else:
        plt.savefig(mdir + 'spectra_specwizard_%s_vs_%s_flux-T-nion-vpec_spectrum%i.pdf'%(samplename, subsamplename, key_sub) ,format = 'pdf',bbox_inches='tight')


def test_speccor(filename, specnum, ions):
    specp = Sightline(filename, specnum)
    speco = Sightline(filename, specnum)
    fontsize=12
    colors = ['orange', 'gold', 'green', 'cyan', 'purple', 'magenta']
    for ion in ions:
        flux_orig = specp.getspectrum(ion, qty='Flux', corr=False)
        specp.get_multiplet_flux(ion, periodic=True, verbose=True, lineincl='all')
        speco.get_multiplet_flux(ion, periodic=False, verbose=True, lineincl='all')
        flux_p = specp.spectra['%s/FluxCorr'%ion]
        flux_o = speco.spectra['%s/FluxCorr'%ion]
        voffpix = speco.corrinfo[ion]['v0ind']   
        deltav  = speco.velperpix_kmps
        try:
            xvals_p = sp.lambda_rest[ion] * (1. + np.arange(len(flux_orig)) * deltav * 1e5 / cu.c.c  )
            xvals_o = sp.lambda_rest[ion] * (1. + (np.arange(len(flux_o)) - voffpix) * deltav * 1e5 / cu.c.c )
        except KeyError:
            xvals_p = sp.lambda_rest['%smajor'%ion] * (1. + np.arange(len(flux_orig)) * deltav * 1e5 / cu.c.c  )
            xvals_o = sp.lambda_rest['%smajor'%ion] * (1. + (np.arange(len(flux_o)) - voffpix) * deltav * 1e5 / cu.c.c )
        if ion in sp.multip.keys():
            lambdas = [sp.lambda_rest[i] for i in sp.multip[ion]]
            fosc    = [sp.fosc[i] for i in sp.multip[ion]]
        elif ion in sp.lambda_rest.keys():
            lambdas = [sp.lambda_rest[ion]]
            fosc    = [sp.fosc[ion]]
        else:
            lambdas = [sp.lambda_rest['%smajor'%ion]]
            fosc    = [sp.fosc['%smajor'%ion]]
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(xvals_p, flux_orig, label='original spectrum', color='black')
        ax.plot(xvals_p, flux_p,    label='periodic',          color='blue', linestyle='dashed')
        ax.plot(xvals_o, flux_o,    label='non-periodic',      color='red',  linestyle='dashdot')
        for i in range(len(lambdas)):
            ax.axvline(lambdas[i], label=r'$%.3f \; \mathrm{m \AA}$-line, $f=%.3f$'%(lambdas[i], fosc[i]), linestyle='dotted', color=colors[i])
        ax.legend(fontsize=fontsize, loc='upper left', bbox_to_anchor=(1.05, 0.95))
        ax.set_xlabel(r'$\lambda \; [\mathrm{m\AA}]$', fontsize=fontsize)
        ax.set_ylabel(r'norm. flux', fontsize=fontsize)   
        
    return specp, speco

def get_multiion_spectra(filename, specnum, ions, space='energy', save=False, emin=None, emax=None, savedir=None, savename=None):
    '''
    emin, emax: min, max enrgy of lines to include [keV]
    saving works, but files get very large (can be ~1/3 GB)
    '''
    sl = Sightline(filename, specnum)
    lines = {ion: ild.get_linenames(ion) for ion in ions}
    if emin is not None or emax is not None:
        if emin is not None:
            energies = {ion: [cu.c.c * cu.c.planck / (lambda_rest[line] * 1.e-8) / (cu.c.ev_to_erg * 1e3) for line in lines[ion]] for ion in lines.keys()}
            lines = {ion: list(np.array(lines[ion])[np.array(energies[ion]) >= emin]) for ion in ions}
        if emax is not None:
            energies = {ion: [cu.c.c * cu.c.planck / (lambda_rest[line] * 1.e-8) / (cu.c.ev_to_erg * 1e3) for line in lines[ion]] for ion in lines.keys()}
            lines = {ion: list(np.array(lines[ion])[np.array(energies[ion]) <= emax]) for ion in ions}
    sl.get_multiion_flux(ions, verbose=True, space=space, lineincl=lines)
    for ion in ions:
        if len(lines[ion]) > 0:
            sl.get_multiion_flux([ion], verbose=True, space=space, name=ion, lineincl={ion: lines[ion]})

    alllines = [line for ion in ions for line in lines[ion]]
    
    if save:
        if savename is None:
            savename = 'spectrum%s_multiion'%specnum
        if savename[-5:] != '.hdf5':
            savename = savename + '.hdf5'
        if savedir is None:
            savedir = '/net/luttero/data2/xray_mocks/input_spectra/'
        if '/' not in savename:
            savename = savedir + savename
        
        with h5py.File(savename, 'a') as out:
            linegroup = out.create_group('linedata')
            ild.savelinetable(linegroup, ild.linetable.loc[alllines])
            if space == 'wavelength':
                xkey = 'lambda_A'
            else:
                xkey = 'E_keV'
            fullgroup = out.create_group('full_spectrum')
            fullgroup.create_dataset(xkey,  data=sl.spectra['multiline_0'][xkey])
            fullgroup.create_dataset('flux',  data=sl.spectra['multiline_0']['flux'])
            iongroup = out.create_group('ion_spectra')
            for ion in ions:
                if len(lines[ion]) > 0:
                    subgroup = iongroup.create_group(ion)
                    subgroup.create_dataset(xkey,  data=sl.spectra['multiline_%s'%ion][xkey])
                    subgroup.create_dataset('flux',  data=sl.spectra['multiline_%s'%ion]['flux'])
            header = out.create_group('Header')
            header.attrs.create('ions', ions)
            header.attrs.create('Emin_keV', emin)
            header.attrs.create('Emax_keV', emax)
            header.attrs.create('specwizard_file', filename)
            header.attrs.create('specnum', specnum)
    return sl    
              
    
def plot_multiline(sl=None, filename=None, specnum=None, ions=None, returnobj=False, space='energy', lineindic=True, savedir=None, emin=None, emax=None, close=True):
    if sl is None:
        sl = get_multiion_spectra(filename, specnum, ions, returnobj=True, space=space, save=False, emin=emin, emax=emax, savedir=None, savename=None)
    else:
        specnum = sl.specnum
        filename = sl.filename
        ions = sl.spectra.keys()
        ions = [ion if 'multiline_' in ion else None for ion in ions]
        ions = list(set(ions))
        ions.remove('multiline_0')
        try:
            ions.remove(None)
        except ValueError:
            pass
        ions = [ion[10:] for ion in ions]
            
    fontsize = 12
    ylim = (0., 1.02)
    colors = ['firebrick', 'red', 'indianred', 'orange', 'gold', 'limegreen', 'forestgreen', 'cyan', 'blue', 'midnightblue', 'purple', 'magenta', 'pink']    
    if space == 'wavelength':
        xkey = 'lambda_A'
    else:
        xkey = 'E_keV'
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10., 3.))
    fig.suptitle('Spectrum %i'%specnum, fontsize=fontsize)
    xv = sl.spectra['multiline_0'][xkey]
    ax.plot(xv, sl.spectra['multiline_0']['flux'], linestyle='solid', color='black')
    
    if lineindic:
        for ind in range(len(ions)):
            
            color = colors[ind]
            ion = ions[ind]
            
            xvals_o = sl.spectra['multiline_%s'%ion][xkey]
            flux_o  = sl.spectra['multiline_%s'%ion]['flux']
            #deltav  = sl.velperpix_kmps
            #try:
            #    xvals_o = sp.lambda_rest['%smajor'%ion] * (1. + (np.arange(len(flux_o)) - voffpix) * deltav * 1e5 / cu.c.c )
            #except KeyError:
            #    xvals_o = sp.lambda_rest[ion] * (1. + (np.arange(len(flux_o)) - voffpix) * deltav * 1e5 / cu.c.c )
            lines = sl.spectra['multiline_%s'%ion]['lines']
            lambdas = [lambda_rest[i] for i in lines]
            foscs    = [fosc[i] for i in lines]
        
            ax.plot(xvals_o, flux_o, label='%s spectrum'%ion, color=color, linestyle='dashed')

            for i in range(len(lambdas)):
                ax.axvline(lambdas[i], 0., 0.7/ylim[1], label=r'$%.3f \; \mathrm{\AA}$, $f=%.3f$'%(lambdas[i], foscs[i]), linestyle='dotted', color=color)

    ax.legend(fontsize=fontsize-2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    ax.set_ylim(*ylim)
    pad = 0.05 * (xv[-1] - xv[0])
    ax.set_xlim(xv[0] - pad, xv[-1] + pad)  
    if space == 'wavelength':
        ax.set_xlabel(r'$\lambda_{\mathrm{rest}} \; [\mathrm{\AA}]$', fontsize=fontsize)
    elif space == 'energy':
        ax.set_xlabel(r'$E_{\mathrm{rest}} \; [\mathrm{keV}]$', fontsize=fontsize)        
    ax.set_ylabel(r'norm. flux', fontsize=fontsize)  

    sample = filename.split('/')[-2]
    if savedir is None:
        savedir = './'
    plt.savefig(savedir + 'multiline_spectrum%i_%s.pdf'%(specnum, sample), format='pdf', bbox_inches='tight')
    if close:
        plt.close()
    
    if returnobj:
        return sl

def plot_multiline_vpanels(specnum, lines, filename=None, vedges=None, savedir=None, savename=None):
    '''
    ions: list of lists, ion names matching ion_lin_data
    
    '''
    fontsize=12
    
    if savedir is None:
        savedir = '/net/luttero/data2/xray_mocks/input_spectra/'
    if savename is None:
        savename = 'spectrum_%i'%(specnum)
    if savename[-4:] != '.pdf':
        savename = savename + '.pdf'
    if '/' not in savename:
        savename = savedir + savename
    if filename is None:
        filename='/net/luttero/data2/specwizard_data/sample3/spec.snap_027_z000p101.0.hdf5'
    
    sl = Sightline(filename, specnum)
    ions  = [ild.linetable.loc[line[0], 'ion'] for line in lines]
    
    nlines = len(lines)
    fig = plt.figure(figsize=(10., 2.*nlines))
    grid = gsp.GridSpec(nlines, 1, hspace=0., top=0.95, bottom=0.05, left=0.05, right=0.95)
    axes = np.array([plt.subplot(grid[gi]) for gi in range(nlines)])
    
    axes[-1].set_xlabel(r'$v_{\mathrm{rest}} \, [\mathrm{km} \, \mathrm{s}^{-1}]$', fontsize=fontsize)
    axes[int((nlines - 1) / 2)].set_ylabel('norm. flux', fontsize=fontsize)
    
    for ioni in range(nlines):
        ion = ions[ioni]
        lineset = lines[ioni]
        sl.get_multiplet_flux(ion, periodic=True, verbose=False, lineincl=list(np.copy(lineset)))
        ax = axes[ioni]

        yvals = sl.spectra['%s/FluxCorr'%ion]
        xvals = np.arange(len(yvals)) / float(len(yvals)) * sl.deltaredshift * sl.cosmopars['a'] * cu.c.c / 1.e5
        
        ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=(ioni == nlines -1), labelleft=True, labelright=False, color='black')
        ax.plot(xvals, yvals)
        ax.set_xlim(xvals[0], 2. * xvals[-1] - xvals[-2])
        ylim = ax.get_ylim()
        ylim_new = (ylim[0] - (ylim[1] - ylim[0]) * 0.4, ylim[1] + (ylim[1] - ylim[0]) * 0.3)
        ax.set_ylim(*ylim_new)
        
        ax.text(0.98, 0.05, ild.getnicename(ion), fontsize=fontsize, transform=ax.transAxes,\
                    verticalalignment='bottom', horizontalalignment='right')
        
        for li in range(len(lineset)):
            xpos = 0.02 + 0.15 * li
            ypos = 0.05
            label = r'$\lambda = %.4f \, \mathrm{\AA}$'%(ild.linetable.loc[lineset[li], 'lambda_angstrom']) \
                    + '\n' + \
                    r'$f_{\mathrm{osc}} = %.3f$'%(ild.linetable.loc[lineset[li], 'fosc'])
            ax.text(xpos, ypos, label, fontsize=fontsize, transform=ax.transAxes,\
                    verticalalignment='bottom', horizontalalignment='left') # bbox=dict(facecolor='white',alpha=0.3)
        print(vedges)
        if vedges is not None:
            EWs_mA = [sl.getEW(ion, corr=True, vrange=list(np.copy(vrange)), vvals=xvals) for vrange in vedges]
            xlim = ax.get_xlim()
            for vi in range(len(vedges)):
                vset = vedges[vi]
                ax.axvline(vset[0], color='gray', linestyle='dotted')
                ax.axvline(vset[1], color='gray', linestyle='dotted')
                ax.text((vset[0] - xlim[0]) / (xlim[1] - xlim[0]), 0.95, r'$%.2f \, \mathrm{m\AA}$'%(EWs_mA[vi]),\
                        fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left' )
        print(vedges)
    fig.suptitle('Spectrum %i'%specnum, fontsize=fontsize)        
    plt.savefig(savename, format='pdf', bbox_inches='tight')
    
# sample3_sub(_deltacorr)
sightlines_xsel = {\
    0:    {'vrange': [(2700, 3250), (6350, 200 )], 'prange': [(32., 51. ), (89., 3.)]},\
    1:    {'vrange': [(2150, 2650), (5400, 50  )], 'prange': [(33.7, 37. ), (73.6, 0.5)]},\
    #27:   {'vrange': [(2970, 3230), (3950, 4420)], 'prange': [(41., 43.6), (61.4, 68.8)]},\ # sample3 data
    27:   {'vrange': [(4800, 5150)], 'prange': [(70.6, 77.)]}
    }   


    
def getplotvals(xvals, yvals, rng, yonly=False):
    '''
    deals with ranges that overlap a periodic boundary; returns selected x and 
    y ranges, with y shifted if necessary
    '''
    if np.all(np.array(rng) == np.array((None, None))):
        rng = (-np.inf, np.inf)
    if rng[1] > rng[0]: 
        xsel = np.logical_and(xvals >= rng[0], xvals <= rng[1])
        cycle = False
    else:
        #xsel = np.logical_or(xvals <= rng[0], xvals >= rng[1])
        cycle = True 
        xsel1 = xvals >= rng[0]
        xsel2 = xvals <= rng[1]
    
    if cycle: # selection range crosses periodic boundary -> need to do some extra work
        yvals_1 = yvals[xsel1]
        yvals_2 = yvals[xsel2]
        yvals_out = np.append(yvals_1, yvals_2)

        xvals_1 = xvals[xsel1] - (xvals[-1] + np.average(np.diff(xvals))) # impose periodic boundary
        xvals_2 = xvals[xsel2]
        xvals_out = np.append(xvals_1, xvals_2)
        #print(len(xvals), len(yvals_o7), len(yvals_o8))
    else:
        yvals_out = yvals[xsel]
        xvals_out = xvals[xsel]
    if yonly:
        return yvals_out
    else:
        return xvals_out, yvals_out

def add_xranges(ax, ranges, range_, ioncolors=None, linestyle='dotted', lineind=None, period=None):
    if ioncolors is None:
        ioncolors = list(('gray',)*range.shape[0])
    if lineind is None: # all lines
        for gind in range(np.prod(ranges.shape)):
            ionind = gind / (ranges.shape[1]*ranges.shape[2]) # slowest index
            sysind = gind / ranges.shape[2] - ranges.shape[1]*ionind # middle index
            endind = gind % ranges.shape[2] # fastest index
            ax.axvline(ranges[ionind, sysind, endind], color=ioncolors[ionind], linestyle=linestyle)       
    else:
         for gind in range(ranges.shape[0]*ranges.shape[2]):
            ionind = gind / ranges.shape[2] # slowest index
            endind = gind % ranges.shape[2] # fastest index
            if period is not None:
                if not ranges[ionind, lineind, endind] <= range_[lineind, 1] and ranges[ionind, lineind, endind] >= range_[lineind, 1]:
                    ax.axvline(ranges[ionind, lineind, endind] - period, color=ioncolors[ionind], linestyle=linestyle)
                else:
                    ax.axvline(ranges[ionind, lineind, endind], color=ioncolors[ionind], linestyle=linestyle)
            else:        
                ax.axvline(ranges[ionind, lineind, endind], color=ioncolors[ionind], linestyle=linestyle)

def subplotset(axes, nlines, xtot, ytot, xrange_, xranges, ions, ioncolors, syncminmax=True, maxdynrange=None, fontsize=12.):
    '''
    For plotting a row of different zooms of the same values
    '''
    minmax = [np.inf, -np.inf]
    for sysind in range(nlines):
        ax = axes[sysind]
        xrange_sys = xrange_[sysind]
        xranges_sys = xranges[:, sysind]
        
        vals_sub = getplotvals(xtot, ytot[0], xrange_sys) # xsub, ysub
        vals_sub = list(vals_sub)
        vals_sub += [getplotvals(xtot, ytot[ionind], xrange_sys, yonly=True) for ionind in range(1, len(ions))]    
        xvals_sub = vals_sub[0]
        yvals_sub = vals_sub[1:]
        
        for ionind in range(len(ions)):
            ax.plot(xvals_sub, yvals_sub[ionind], color=ioncolors[ionind])
        add_xranges(ax, xranges, xrange_, ioncolors=ioncolors, lineind=sysind, period=xtot[-1]+np.average(np.diff(xtot)) )
    
        ax.minorticks_on()
        
        if sysind == 0:
            labelright = False
            labelleft = True
        elif not syncminmax:
            labelright = True
            labelleft = False
        else:
            labelright = False
            labelleft = False
        ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=labelleft, labelright=labelright, color='black')
        
        # for min/max: only require y value inclusion in the main components for each ion
          
        for ionind in range(len(ions)):
            yvals_sub += [getplotvals(xtot, ytot[ionind], xranges_sys[ionind], yonly=True) for ionind in range(len(ions))]   
        if syncminmax:
            minmax[0] = min(minmax[0], min([np.min(yvals[np.isfinite(yvals)]) for yvals in yvals_sub]))
            minmax[1] = max(minmax[1], max([np.max(yvals[np.isfinite(yvals)]) for yvals in yvals_sub]))
    if syncminmax:
        if maxdynrange is not None:
            minmax[0] = max(minmax[0], minmax[1]-maxdynrange)
        for sysind in range(nlines):
            axes[sysind].set_ylim((minmax[0], minmax[1]))
        
def plotspectrum_o78(sightline, sample_in=None, auxdata = None):   
    '''
    for a given sightline: plot o7 and o8 spectra,
    -------------- total
    Athena mock
    Arcus mock
    spectrum
    ------------- per system
    spectrum (zoom)
    T (vspace)
    overdensity (vspace)
    
    nion (pspace)
    T (pspace)
    overdensity (pspace)
    vpec (pspace)
    '''
    # set per sightline data (customised)
    #if sightline in sightlines_xsel.keys():
    #    vrange = sightlines_xsel[sightline]['vrange']
    #    prange = sightlines_xsel[sightline]['prange']
    #else:
    #    prange = [(None, None)]
    #    vrange = [(None, None)]
    if sample_in is None:
        sample = sample
        samplename = 'sample3'
    elif isinstance(sample_in, SpecSample):
        sample = sample_in
        if sightline not in sample.sldict.keys():
            sample.addspecs(np.array([sightline]))
    else:
        sample = SpecSample(sdir + sample_in, specnums = np.array([sightline]))
        samplename = sample_in.split('/')[0]
    sl = sample.sldict[sightline]
    ions = ['o7', 'o8']

    vvals = sample.velocity # km/s
    pvals = vvals*1.e5 # rest-frame velocity in km/s ->  rest-frame velocity in cm/s
    pvals /= cu.Hubble(sample.cosmopars['z'],cosmopars=sample.cosmopars) * cu.c.cm_per_mpc # pMpc (local hubble flow means local size)
    pvals *= (1. + sl.cosmopars['z']) # convert to cMpc
    print(max(pvals))
    
    # retrieve compnents (TODO: add auxfile handling once I have a parameter set nailed down)
    #dclim = 0.01
    #vsep = 500.
    #iondclim = 1.e-5
    #vmar = 2.
    pmar_plot = 0.2  #cMpc
    vmar_plot = 200. # km/s
    includeions = []
    for ion in ions:
        #print ion
        sl.getcomponents(ion, vvals, saveto_file=None)
        if sl.componentdct[ion]['numsys'] == 0:
            print('No components for ion %s in sightline %i'%(ion, sightline))
        else:
            includeions += [ion]
    if len(includeions) == 0:
        print('No components found')
        return None
    
    # get the max component range (for setting the plot range) for the different ions
    pranges = [sl.componentdct[ion]['edgevals_p'] for ion in includeions] # trying to access edgevals_p will give an error if there are no components
    if not np.all(np.diff(np.array([len(prange) for prange in pranges])) == 0): # check that the number of components is the same for all included ions
        print('Different ions have different numbers of components in sightline %i; skipping'%sightline)
        return None
    pranges = np.array([[prange if prange[0] <= prange[1] else\
                        np.array([prange[0] - sl.slicelength, prange[1] + sl.slicelength])\
                        for prange in pranges_ion]\
                        for pranges_ion in pranges])
    prange = np.array([np.min(pranges[:, :, 0], axis=0), np.max(pranges[:, :, 1], axis=0)]).T % sl.slicelength
    prange[:, 0] -= pmar_plot
    prange[:, 1] += pmar_plot
    prange %= sl.slicelength
    pranges %= sl.slicelength
    
    maxv = vvals[-1] + np.average(np.diff(vvals)) # periodicity in velocity space
    vranges = [sl.componentdct[ion]['edgevals_v'] for ion in includeions] # trying to access edgevals_p will give an error if there are no components
    print(vranges)
    vranges = np.array([[vrange if vrange[0] <= vrange[1] else\
                        np.array([vrange[0] - maxv, vrange[1] + maxv])\
                        for vrange in vranges_ion]\
                        for vranges_ion in vranges])
    print(vranges)
    vrange = np.array([np.min(vranges[:, :, 0], axis=0), np.max(vranges[:, :, 1], axis=0)]).T % maxv    
    print(vrange)
    vrange[:, 0] -= vmar_plot
    vrange[:, 1] += vmar_plot
    vrange %= maxv
    vranges %= maxv
    
    print(vrange)
    print(vranges)
    print(maxv)
    print(prange)
    print(pranges)
    print(sl.slicelength)
    
    # set up grid and axes
    nlines = len(prange)
    fig = plt.figure(figsize = (7*nlines,11.))
    grid = gsp.GridSpec(3, 1, height_ratios=[4., 3., 4.], hspace=0.17, top = 0.95, bottom = 0.05, left= 0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    grid_ts = gsp.GridSpecFromSubplotSpec(4, 1, subplot_spec=grid[0], hspace=0.0)
    grid_vs = gsp.GridSpecFromSubplotSpec(3, nlines, subplot_spec=grid[1], hspace=0.0, wspace=0.1)
    grid_ps = gsp.GridSpecFromSubplotSpec(4, nlines, subplot_spec=grid[2], hspace=0.0, wspace=0.1)
    axes_ts = np.empty((4,), dtype=object)
    axes_vs = np.empty((3, nlines), dtype=object)
    axes_ps = np.empty((4, nlines), dtype=object)
    for gi in range(4):
        axes_ts[gi] = plt.subplot(grid_ts[gi])
    for gi in range(3*nlines):
        ri = gi/nlines
        ci = gi%nlines
        axes_vs[ri, ci] = plt.subplot(grid_vs[ri, ci])
    for gi in range(4*nlines):
        ri = gi/nlines
        ci = gi%nlines
        axes_ps[ri, ci] = plt.subplot(grid_ps[ri, ci])
        
    #fig, axes = plt.subplots(nrows=9, ncols=nlines, squeeze=False, , gridspec_kw = {'top': 0.95, 'bottom':0.05, 'left':0.05, 'right':0.95, 'hspace':0.25, 'wspace':0.02})
    fontsize = 12 
    clist = ['red','blue']

    for ion in includeions:
        sl.getspectrum(ion,'coldens')
        sl.getspectrum(ion,'Flux')
        sl.getspectrum(ion,'tauw/Temperature_K')
        sl.getspectrum(ion,'tauw/OverDensity')
        sl.getspectrum(ion,'realnionw/Temperature_K')
        sl.getspectrum(ion,'realnionw/NIon_CM3')
        sl.getspectrum(ion,'realnionw/vpec')
        sl.getspectrum(ion,'realnionw/OverDensity')
    sl.getspectrum('o8','Flux', corr=True)
    
    ##### plot the different spectra (just hard-code the positions and make them easy to switch)
    vlabel = 'v [km/s]'
    plabel = 'z [cMpc]'
    
    halfpixsize_v = 0.5*np.average(np.diff(vvals))
    ## Athena mock: ion-specific
    ax = axes_ts[0]
    ax.text(0.05,0.05, r'ATHENA mock', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, labeltop=True, color='black')
    ax.set_xlabel(vlabel, fontsize=fontsize)
    ax.xaxis.set_label_position('top') # ylabel at top of axis
    
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    normphotonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Athena')
    normphotonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Athena')
    yvals_o7 = np.append(normphotonspec_o7, np.array([normphotonspec_o7[0], normphotonspec_o7[0] ])) 
    noise_o7 = np.append(normphotonnoise_o7, np.array([normphotonnoise_o7[0],normphotonnoise_o7[0] ]))
    yvals_o7_lo = yvals_o7 - noise_o7
    yvals_o7_hi = yvals_o7 + noise_o7
    xvals_o7 = np.arange(len(normphotonspec_o7) + 2) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    yvals_o8 = np.append(normphotonspec_o8, np.array([normphotonspec_o8[0], normphotonspec_o8[0]]))
    noise_o8 = np.append(normphotonnoise_o8, np.array([normphotonnoise_o8[0], normphotonnoise_o8[0]]))
    yvals_o8_lo = yvals_o8 - noise_o8
    yvals_o8_hi = yvals_o8 + noise_o8
    xvals_o8 = np.arange(len(normphotonspec_o8) + 2) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    
    ax.plot(xvals_o7, np.ones(len(xvals_o7)), color='gray', linestyle='solid') # indicate continuum level
    ax.step(xvals_o7, yvals_o7, where='post', color=clist[0])
    ax.fill_between(xvals_o7, yvals_o7_lo, yvals_o7_hi, step='post', color=clist[0], alpha=0.2 )    
    ax.step(xvals_o8, yvals_o8, where='post', color=clist[1], linestyle='dashed')
    ax.fill_between(xvals_o8, yvals_o8_lo, yvals_o8_hi, step='post', color=clist[1], alpha=0.2 )
    maxx = max([xvals_o7[-2] + halfpixsize_v, xvals_o8[-2] + halfpixsize_v])
    
    ## Arcus mock: ion-specific
    ax = axes_ts[1]
    ax.text(0.05,0.05, r'Arcus mock', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, color='black')
    
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    normphotonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Arcus')
    normphotonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Arcus')
    yvals_o7 = np.append(normphotonspec_o7, np.array([normphotonspec_o7[0], normphotonspec_o7[0] ])) 
    noise_o7 = np.append(normphotonnoise_o7, np.array([normphotonnoise_o7[0],normphotonnoise_o7[0] ]))
    yvals_o7_lo = yvals_o7 - noise_o7
    yvals_o7_hi = yvals_o7 + noise_o7
    xvals_o7 = np.arange(len(normphotonspec_o7) + 2) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    yvals_o8 = np.append(normphotonspec_o8, np.array([normphotonspec_o8[0], normphotonspec_o8[0]]))
    noise_o8 = np.append(normphotonnoise_o8, np.array([normphotonnoise_o8[0], normphotonnoise_o8[0]]))
    yvals_o8_lo = yvals_o8 - noise_o8
    yvals_o8_hi = yvals_o8 + noise_o8
    xvals_o8 = np.arange(len(normphotonspec_o8) + 2) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    
    ax.plot(xvals_o7, np.ones(len(xvals_o7)), color='gray', linestyle='solid') # indicate continuum level
    ax.step(xvals_o7, yvals_o7, where='post', color=clist[0])
    ax.fill_between(xvals_o7, yvals_o7_lo, yvals_o7_hi, step='post', color=clist[0], alpha=0.2 )    
    ax.step(xvals_o8, yvals_o8, where='post', color=clist[1], linestyle='dashed')
    ax.fill_between(xvals_o8, yvals_o8_lo, yvals_o8_hi, step='post', color=clist[1], alpha=0.2 )
    maxx = max([maxx, xvals_o7[-2] + halfpixsize_v, xvals_o8[-2] + halfpixsize_v])
    
    ## Ideal spectrum
    ax = axes_ts[2]
    xvals = vvals
    yvals  = [sl.spectra[getspecname(ion, 'Flux')] for ion in includeions]
    if 'o8' in includeions:
        yvals_o8d = sl.spectra['o8/FluxCorr']

    for ionind in range(len(includeions)):
        ax.plot(xvals, yvals[ionind], color=clist[ionind])
    if 'o8' in includeions:
        ax.plot(xvals, yvals_o8d, color=clist[1], linestyle = 'dashed')
    add_xranges(ax, vranges, vrange, ioncolors=clist)
           
    ax.set_ylabel('norm. flux', fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, color='black')
    maxx = max(maxx, vvals[-1])
    
    ## Nion spectrum (in vspace equivalent units)
    ax = axes_ts[3]
    xvals = pvals
    print(min(pvals), max(pvals))
    yvals  = [np.log10(sl.spectra[getspecname(ion, 'realnionw/NIon_CM3')]) for ion in includeions]
    for ionind in range(len(includeions)):
        ax.plot(xvals, yvals[ionind], color=clist[ionind])
    add_xranges(ax, pranges, prange, ioncolors=clist)
           
    ax.set_ylabel(r'$n_{\mathrm{ion}\, [\mathrm{cm}^{-3}]}$', fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=True, labelleft=True, labelright=False, color='black')
    ax.set_xlabel(plabel, fontsize=fontsize)
    
     # sync x ranges of top 4 plots (might deviate a bit due to different pixels )
    axes_ts[0].set_xlim(0, maxx)
    axes_ts[1].set_xlim(0, maxx)
    axes_ts[2].set_xlim(0, maxx)
    axes_ts[3].set_xlim(0, maxx * pvals[-1] / vvals[-1])
    
    ## spectrum zooms
    xvals_tot = vvals
    yvals_tot  = [sl.spectra[getspecname(ion, 'Flux')] for ion in includeions]
    if 'o8' in includeions:
        yvals_o8d_tot = sl.spectra['o8/FluxCorr']
    axes_vs[0, 0].set_ylabel('norm. flux', fontsize=fontsize)
    subplotset(axes_vs[0, :], nlines, xvals_tot, yvals_tot, vrange, vranges, includeions, clist, syncminmax=False, maxdynrange=None, fontsize=fontsize)
    # add o8 doublet
    if 'o8' in includeions:
        for sysind in range(nlines):
            xvals_sub, yvals_o8d_sub = getplotvals(xvals_tot, yvals_o8d_tot, vrange[sysind])
            axes_vs[0, sysind].plot(xvals_sub, yvals_o8d_sub, color=clist[1], linestyle = 'dashed')

    ## tau-weighted temperatures
    xvals_tot = vvals
    yvals_tot  = [np.log10(sl.spectra[getspecname(ion, 'tauw/Temperature_K')]) for ion in includeions]
    axes_vs[1, 0].set_ylabel(r'$\log_{10} T_{\tau} \, [\mathrm{K}]$', fontsize=fontsize)
    subplotset(axes_vs[1, :], nlines, xvals_tot, yvals_tot, vrange, vranges, includeions, clist, syncminmax=True, maxdynrange=None, fontsize=fontsize)
         
    ## tau-weighted densities
    xvals_tot = vvals
    yvals_tot  = [np.log10(sl.spectra[getspecname(ion, 'tauw/OverDensity')]) for ion in includeions]
    axes_vs[2, 0].set_ylabel(r'$\log_{10}(1 + \delta_{\tau})$', fontsize=fontsize)
    subplotset(axes_vs[2, :], nlines, xvals_tot, yvals_tot, vrange, vranges, includeions, clist, syncminmax=True, maxdynrange=5., fontsize=fontsize)
    for ax in axes_vs[2, :]:
        ax.set_xlabel(vlabel, fontsize=fontsize)
        ax.tick_params(labelbottom=True)

    ## nion zooms:
    xvals_tot = pvals
    yvals_tot  = [np.log10(sl.spectra[getspecname(ion, 'realnionw/NIon_CM3')]) for ion in includeions]
    axes_ps[0, 0].set_ylabel(r'$\log_{10} n_{\mathrm{ion}} \, [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    subplotset(axes_ps[0, :], nlines, xvals_tot, yvals_tot, prange, pranges, includeions, clist, syncminmax=True, maxdynrange=6., fontsize=fontsize)
    
    ## nion-weighted temperature:
    xvals_tot = pvals
    yvals_tot  = [np.log10(sl.spectra[getspecname(ion, 'realnionw/Temperature_K')]) for ion in includeions]
    axes_ps[1, 0].set_ylabel(r'$\log_{10} T \, [\mathrm{K}]$', fontsize=fontsize)
    subplotset(axes_ps[1, :], nlines, xvals_tot, yvals_tot, prange, pranges, includeions, clist, syncminmax=True, maxdynrange=None, fontsize=fontsize)
    
    ## nion-weighted density:
    xvals_tot = pvals
    yvals_tot  = [np.log10(sl.spectra[getspecname(ion, 'realnionw/OverDensity')]) for ion in includeions]
    axes_ps[2, 0].set_ylabel(r'$\log_{10}(1 + \delta)$', fontsize=fontsize)
    subplotset(axes_ps[2, :], nlines, xvals_tot, yvals_tot, prange, pranges, includeions, clist, syncminmax=True, maxdynrange=5., fontsize=fontsize)
    
    ## nion-weighted velocity:
    xvals_tot = pvals
    yvals_tot  = [sl.spectra[getspecname(ion, 'realnionw/vpec')] for ion in includeions]
    axes_ps[3, 0].set_ylabel(r'$v_{\mathrm{pec}} \, [\mathrm{km}\, \mathrm{s}^{-1}]$', fontsize=fontsize)
    subplotset(axes_ps[3, :], nlines, xvals_tot, yvals_tot, prange, pranges, includeions, clist, syncminmax=False, maxdynrange=None, fontsize=fontsize)
    for ax in axes_ps[3, :]:
        ax.set_xlabel(plabel, fontsize=fontsize)
        ax.tick_params(labelbottom=True)
    ### add info and labels
    
    # general info: col. dens. and EW for each ion
    coldens = [sl.spectra[getspecname(ion,'coldens')] for ion in includeions]   
    EWs = []
    coldens_tau_sys = []
    coldens_nion_sys = []
    EWs_sys = []
    bbox = dict(facecolor='white', alpha=0.3)
    for ionind in range(len(includeions)):
        ion = includeions[ionind]
        if ion == 'o8':
            corr = True
        else:
            corr = False
        EWs += [sl.getEW(ion, corr=corr, vrange=None, vvals=None)]
        coldens_nion_sys += [[sl.getcoldens_from_nion(ion, prange=prng, pvals=pvals) for prng in pranges[ionind]]]
        coldens_tau_sys += [[sl.getcoldens_from_tau(ion, vvals, vrange=vrng) for vrng in vranges[ionind]]]
        EWs_sys +=  [[sl.getEW(ion, corr=corr, vrange=vrng, vvals=vvals) for vrng in vranges[ionind]]] 
       
        # add info to relevant plots
   
        ypos = 0.05 + ionind/float(len(includeions))
        print(ypos)
        axes_ts[2].text(0.01, ypos, r'$EW = %.2e \, \mathrm{m\AA}$'%(EWs[ionind]/1.e-3), fontsize=fontsize, transform=axes_ts[2].transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=bbox, color=clist[ionind])
        axes_ts[3].text(0.01, ypos, r'$\log_{10}N = %.1f \, \mathrm{cm}^{-2}$'%(coldens[ionind]), fontsize=fontsize, transform=axes_ts[3].transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=bbox, color=clist[ionind])
        for sysind in range(len(vrange)):
            axes_vs[0, sysind].text(0.01, ypos, r'$EW = %.2f \, \mathrm{m\AA}$'%(EWs_sys[ionind][sysind]/1.e-3), fontsize=fontsize, transform=axes_vs[0, sysind].transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=bbox,color=clist[ionind])
            axes_vs[0, sysind].text(0.99, ypos, r'$\log_{10}N = %.1f \, \mathrm{cm}^{-2}$'%(np.log10(coldens_tau_sys[ionind][sysind])), fontsize=fontsize, transform=axes_vs[0, sysind].transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=bbox,color=clist[ionind])
            axes_ps[0, sysind].text(0.99, ypos, r'$\log_{10}N = %.1f \, \mathrm{cm}^{-2}$'%(np.log10(coldens_nion_sys[ionind][sysind])), fontsize=fontsize, transform=axes_ps[0, sysind].transAxes, verticalalignment='bottom', horizontalalignment='right', bbox=bbox,color=clist[ionind])
 
    legend_handles = [mlines.Line2D([], [], color=clist[ionind], linestyle = 'solid', label=r'$\mathrm{%s}$'%(ionlabels[includeions[ionind]])) for ionind in range(len(includeions))]
    ncols = nlines
    if 'o8' in includeions:
        legend_handles += [mlines.Line2D([], [], color=clist[1], linestyle = 'dashed', label=r'$\mathrm{%s}$d'%(ionlabels['o8']))]
        ncols += 1
    axes_ts[0].legend(handles=legend_handles, fontsize=fontsize, ncol=ncols, loc='lower center', bbox_to_anchor=(0.5, 0.01))
    
    #ax = axes[0, 0]
    #ax.text(0.,0.5,    r'$\mathrm{%s}: \log_{10}N = %.1f \, \mathrm{cm}^{-2}, \log_{10} EW = %.2e \, \mathrm{m\AA}$'%(ionlabels['o7'], coldens_o7, EW_o7/1.e-3), fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
    #ax.text(0.,0.0,    r'$\mathrm{%s}: \log_{10}N = %.1f \, \mathrm{cm}^{-2}, \log_{10} EW = %.2e \, \mathrm{m\AA}$'%(ionlabels['o8'], coldens_o8, EW_o8/1.e-3), fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
 
    
#    # what to plot by spectrum quantity
#    spectypes = [None, None, 'Flux', 'tauw/Temperature_K', 'tauw/OverDensity', 'realnionw/NIon_CM3',  'realnionw/Temperature_K', 'realnionw/OverDensity', 'realnionw/vpec']
#    velsp = [True, True, True, True, True, False, False, False, False] 
#    takelog = [False, False, False, True, True, True, True, True, False]
#    typekeys = [[getspecname(ion, spectype) if spectype is not None else\
#                None  for spectype in spectypes] for ion in ions]
#    ylabels = ['flux [photons]', 'flux [photons]', 'norm. flux', r'$\log_{10} T_{\tau} \, [\mathrm{K}]$', r'$\log_{10} \delta_{\tau}$', r'$\log_{10} n_{\mathrm{ion}} \, [\mathrm{cm}^{-3}]$', r'$\log_{10} T_{\mathrm{ion}} \, [\mathrm{K}]$', r'$\log_{10} \delta_{\mathrm{ion}}$', r'$v_{z, \mathrm{ion}} \, [\mathrm{km}\, \mathrm{s}^{-1}]$']
#
#    # set up limit trackers for setting x/y limits consistently
#    maxlims_T = [np.inf, -np.inf] # to set both temperatures to the same range, across all ions
#    maxlims_v = [np.inf, -np.inf] # to set all ion velocities to the same range
#    maxinds_x = [np.inf, -np.inf] # to set positions and velocities to the same range (Hubble flow conversion, i.e. match indices)
#    # (ion) densities: just set to some maximum dynamic range (different ions will differ by abundance as well as line strength)
#    dynrang_d = 5.
#    maxlims_delta = [np.inf, -np.inf]
#    # fraction of max absorption/ion density that we want to include for each ion (fixed-size margins also applied)
#    # assumes main/sub z and v values are the same
#    edgedef_l = 0.2
#    vmargin = 250 # margin on x ranges defined by absorption fractions in km/s
 
#    ## make the plots, and keep track of min/max ranges needed later
#    for genind in range(9*nlines):
#        lineind = genind/9
#        typeind = genind%9
#        #print('ionind: %i, typeind: %i, typekeys shape: (%i, %i)'%(ionind,typeind, len(typekeys), len(typekeys[ionind])))
#        ax = axes[typeind, lineind]
#        typekey = typekeys[0][typeind] # just for e.g. temperature or flux; actual per-ion retrieval is done explicitly
#        
#        do_o8d = False  
#        if typekey is None:
#            ax.text(0.05,0.05,'TODO', fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'bottom', transform=ax.transAxes)
#            continue
#        if velsp[typeind]:
#            xvals = vvals
#            rng = vrange[lineind]
#        else:
#            xvals = pvals
#            rng = prange[lineind]
#        if rng == (None, None):
#            rng = (-np.inf, np.inf)
#        if rng[1] > rng[0]: 
#            xsel = np.logical_and(xvals >= rng[0], xvals <= rng[1])
#            cycle = False
#        else:
#            #xsel = np.logical_or(xvals <= rng[0], xvals >= rng[1])
#            cycle = True 
#            xsel1 = xvals >= rng[0]
#            xsel2 = xvals <= rng[1]
#        if takelog[typeind]:
#            yvals_o7 = np.log10(sl.spectra[typekeys[0][typeind]])
#            yvals_o8 = np.log10(sl.spectra[typekeys[1][typeind]])
#            if typekeys[1][typeind] == 'o8/Flux':
#                yvals_o8d = np.log10(sl.spectra['o8/FluxCorr'])
#                do_o8d = True
#        else:
#            yvals_o7 = sl.spectra[typekeys[0][typeind]]
#            yvals_o8 = sl.spectra[typekeys[1][typeind]]
#            if typekeys[1][typeind] == 'o8/Flux':
#                yvals_o8d = sl.spectra['o8/FluxCorr']
#                do_o8d = True
#
#        if cycle: # selection range crosses periodic boundary -> need to do some extra work
#            yvals_o7_1 = yvals_o7[xsel1]
#            yvals_o7_2 = yvals_o7[xsel2]
#            yvals_o7 = np.append(yvals_o7_1, yvals_o7_2)
#            yvals_o8_1 = yvals_o8[xsel1]
#            yvals_o8_2 = yvals_o8[xsel2]
#            yvals_o8 = np.append(yvals_o8_1, yvals_o8_2)
#            if do_o8d:
#                yvals_o8d_1 = yvals_o8d[xsel1]
#                yvals_o8d_2 = yvals_o8d[xsel2]
#                yvals_o8d = np.append(yvals_o8d_1, yvals_o8d_2)
#            xvals_1 = xvals[xsel1] - (xvals[-1] + np.average(np.diff(xvals))) # impose periodic boundary
#            xvals_2 = xvals[xsel2]
#            xvals = np.append(xvals_1, xvals_2)
#            #print(len(xvals), len(yvals_o7), len(yvals_o8))
#        else:
#            yvals_o7 = yvals_o7[xsel]
#            yvals_o8 = yvals_o8[xsel]
#            if do_o8d:
#                yvals_o8d = yvals_o8d[xsel]
#            xvals    = xvals[xsel]
#
#        doxlabels = False
#        doylabels = False
#        labelright = False
#        if typeind == 8:
#            ax.set_xlabel(plabel, fontsize=fontsize)
#            doxlabels = True
#        if typeind == 4:
#            ax.set_xlabel(vlabel, fontsize=fontsize)
#            doxlabels = True
#        if lineind == 0:
#            doylabels = True
#        elif lineind == nlines-1 or 'Nion' in typekeys[0][typeind] or 'Flux' in typekeys[0][typeind]:
#            labelright = True
#
#        ax.minorticks_on()
#        ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=doxlabels, labelleft=doylabels, labelright=labelright, color='black')
#        ax.plot(xvals, yvals_o7, color=clist[0])
#        ax.plot(xvals, yvals_o8, color=clist[1])
#        if do_o8d:
#           ax.plot(xvals, yvals_o8d, color=clist[1], linestyle = 'dashed')
#        if doylabels:
#           ax.set_ylabel(ylabels[typeind], fontsize=fontsize)
#
#        # do limit checking and accounting
#        if 'Temperature' in typekeys[0][typeind]:
#            maxlims_T[0] = min(maxlims_T[0], np.min(yvals_o7[np.isfinite(yvals_o7)]), np.min(yvals_o8[np.isfinite(yvals_o8)]))
#            maxlims_T[1] = max(maxlims_T[1], np.max(yvals_o7[np.isfinite(yvals_o7)]), np.max(yvals_o8[np.isfinite(yvals_o8)]))
#
#        if 'PeculiarVelocity' in typekeys[0][typeind]:
#            maxlims_v[0] = min(maxlims_v[0], np.min(yvals_o7), np.min(yvals_o8))
#            maxlims_v[1] = max(maxlims_v[1], np.max(yvals_o7), np.max(yvals_o8))
#
#        if 'OverDensity' in typekeys[0][typeind]:
#            maxlims_delta[0] = min(maxlims_delta[0], np.min(yvals_o7), np.min(yvals_o8))
#            maxlims_delta[1] = max(maxlims_delta[1], np.max(yvals_o7), np.max(yvals_o8))
#
#        if 'NIon_CM3' in typekeys[0][typeind]:
#            maxval = max(np.max(yvals_o7), np.max(yvals_o8))
#            ax.set_ylim(maxval-dynrang_d, maxval)
#
#            # handled by prange auto 
#            ## where is the ion density above edgedef_l * max value? (plotted values are log -> take log of threashold for comparison)
#            #incl = np.where(np.any( np.array([yvals_o7 >= np.log10(edgedef_l)+np.max(yvals_o7), yvals_o8 >= np.log10(edgedef_l)+np.max(yvals_o8) ]), axis=0))[0]
#            #maxinds_x[0] = min(maxinds_x[0], np.min(incl))
#            #maxinds_x[1] = max(maxinds_x[1], np.max(incl))
#
#        # handled by vrange auto
#        if 'Flux' in typekeys[0][typeind]:
#            # where is the flux decrement > edgedef_l * max value?
#            if vrange[lineind] == (None, None):
#                incl = np.where(np.any( np.array([(1.-yvals_o7) >= edgedef_l*np.max(1.-yvals_o7), (1.-yvals_o8) >= edgedef_l*np.max(1.-yvals_o8) ]), axis=0))[0]
#                maxinds_x[0] = min(maxinds_x[0], np.min(incl))
#                maxinds_x[1] = max(maxinds_x[1], np.max(incl))
#    
#    ## set z/v range margin (no wrap-around here)
#    if vrange[lineind] == (None, None):
#        indsmargin = int(np.ceil( vmargin/(vvals[-1]-vvals[0])*(len(vvals)-1) ))
#        maxinds_x[0] = max(maxinds_x[0]-indsmargin, 0) 
#        maxinds_x[1] = min(maxinds_x[1]+indsmargin, len(vvals)-1)
#    
#    ## loop back over the plots and apply the x/y ranges we just got
#    for genind in range(9*nlines):
#        lineind = genind/9
#        typeind = genind%9
#        #print('ionind: %i, typeind: %i, typekeys shape: (%i, %i)'%(ionind,typeind, len(typekeys), len(typekeys[ionind])))
#        ax = axes[typeind, lineind]
#        typekey = typekeys[0][typeind]
#
#        if velsp[typeind]:
#            xvals = vvals
#        else:
#            xvals = pvals
#        if np.all(vrange[lineind] == (None, None)):
#            ax.set_xlim(xvals[maxinds_x[0]], xvals[maxinds_x[1]])
#        else:
#            if velsp[typeind]:
#                rng = vrange[lineind]
#            else:
#                rng = prange[lineind]
#            if rng[1] > rng[0]:
#                ax.set_xlim(*rng)
#            else:
#                ax.set_xlim(rng[0] - (xvals[-1] + np.average(np.diff(xvals))), rng[1])
#
#        if typekey is not None:
#            if 'Temperature' in typekey:
#                ax.set_ylim(maxlims_T[0],maxlims_T[1])
#            elif 'PeculiarVelocity' in typekey:
#                ax.set_ylim(maxlims_v[0],maxlims_v[1])
#            elif 'OverDensity' in typekey:
#                ax.set_ylim(max(maxlims_delta[1] - dynrang_d, maxlims_delta[0]), maxlims_delta[1])

    plt.show()
    #plt.savefig(mdir + 'spectra_specwizard_%s_template_spectrum%i.pdf'%(samplename, sightline) ,format = 'pdf',bbox_inches='tight')

def niceformat_mm(number, dec=2):
    exp = int(np.floor(np.log10(number)))
    coef = number / 10**exp
    if dec > 0:
        numstr = str(round(coef, dec))
    else:
        numstr = str(int(round(coef, dec)))
    if exp == 0:
        str_out = numstr
    elif exp == 1 and dec > 1:
        str_out = str(round(coef*10, dec - 1))
    elif exp == 1 and dec == 1:
        str_out = str(int(round(coef*10, dec - 1)))
    elif exp == -1:
        str_out = str(round(coef/10, dec + 1))
    else:
        str_out = r'%s \times 10^{%i}'%(numstr, exp)
    return str_out

def plotspectrum_o78_mocksonly(sightline, sample_in=None, auxdata=None, nameadd=''):
    '''
    plots: Arcus mock, Athena mock, ideal spectrum
    '''
    
    if sample_in is None:
        sample = SpecSample(sdir + 'sample3/spec.snap_027_z000p101.0.hdf5', specnums = np.array([sightline]))
        samplename = 'sample3'
    elif isinstance(sample_in, SpecSample):
        sample = sample_in
        if sightline not in sample.sldict.keys():
            sample.addspecs(np.array([sightline]))
    else:
        sample = SpecSample(sdir + sample_in, specnums = np.array([sightline]))
        samplename = sample_in.split('/')[0] # directory above file
    if sightline not in sample.sldict.keys():
        sample.addspecs([sightline])
    sl = sample.sldict[sightline]
    ions = ['o7', 'o8']

    vvals = sample.velocity # km/s
    pvals = vvals*1.e5 # rest-frame velocity in km/s ->  rest-frame velocity in cm/s
    pvals /= cu.Hubble(sample.cosmopars['z'],cosmopars=sample.cosmopars) * cu.c.cm_per_mpc # pMpc (local hubble flow means local size)
    pvals *= (1. + sl.cosmopars['z']) # convert to cMpc
    
    fig = plt.figure(figsize = (11., 5.))
    grid = gsp.GridSpec(5, 1, height_ratios=[1., 1., 1., 0.15, 0.7], hspace=0.0, top = 0.95, bottom = 0.05, left= 0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    axes = np.empty((3,), dtype=object)
    
    for gi in range(3):
        axes[gi] = plt.subplot(grid[gi])
    lax = plt.subplot(grid[4])
      
    #fig, axes = plt.subplots(nrows=9, ncols=nlines, squeeze=False, , gridspec_kw = {'top': 0.95, 'bottom':0.05, 'left':0.05, 'right':0.95, 'hspace':0.25, 'wspace':0.02})
    fontsize = 12 
    clist = ['red','blue']

    for ion in ions:
        sl.getspectrum(ion,'coldens')
        sl.getspectrum(ion,'Flux')
    sl.getspectrum('o8','Flux', corr=True)
    
    ##### plot the different spectra 
    vlabel = 'v [km/s], rest-frame'
    
    halfpixsize_v = 0.5*np.average(np.diff(vvals)) # to get the right velocity position on the binned spectra 
    ## Athena mock: ion-specific
    ax = axes[0]
    ax.text(0.02,0.07, r'ATHENA mock', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
    ax.text(0.98,0.07, r'(a)', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right')
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=False, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, labeltop=False, color='black')
    
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    normphotonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Athena')
    normphotonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Athena')
    yvals_o7 = np.append(normphotonspec_o7, np.array([normphotonspec_o7[0], normphotonspec_o7[0] ])) 
    noise_o7 = np.append(normphotonnoise_o7, np.array([normphotonnoise_o7[0],normphotonnoise_o7[0] ]))
    yvals_o7_lo = yvals_o7 - noise_o7
    yvals_o7_hi = yvals_o7 + noise_o7
    xvals_o7 = np.arange(len(normphotonspec_o7) + 2) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    yvals_o8 = np.append(normphotonspec_o8, np.array([normphotonspec_o8[0], normphotonspec_o8[0]]))
    noise_o8 = np.append(normphotonnoise_o8, np.array([normphotonnoise_o8[0], normphotonnoise_o8[0]]))
    yvals_o8_lo = yvals_o8 - noise_o8
    yvals_o8_hi = yvals_o8 + noise_o8
    xvals_o8 = np.arange(len(normphotonspec_o8) + 2) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    
    ax.plot(xvals_o7, np.ones(len(xvals_o7)), color='gray', linestyle='solid') # indicate continuum level
    ax.step(xvals_o7, yvals_o7, where='post', color=clist[0])
    ax.fill_between(xvals_o7, yvals_o7_lo, yvals_o7_hi, step='post', color=clist[0], alpha=0.2 )    
    ax.step(xvals_o8, yvals_o8, where='post', color=clist[1], linestyle='dashed')
    ax.fill_between(xvals_o8, yvals_o8_lo, yvals_o8_hi, step='post', color=clist[1], alpha=0.2 )
    maxx = max([xvals_o7[-2] + halfpixsize_v, xvals_o8[-2] + halfpixsize_v])
    
    ## Arcus mock: ion-specific
    ax = axes[1]
    ax.text(0.02,0.07, r'Arcus mock', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
    ax.text(0.98,0.07, r'(b)', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right')
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, color='black')
    ax.set_ylabel('Flux / continuum', fontsize=fontsize, labelpad=fontsize)
    
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    normphotonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Arcus')
    normphotonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Arcus')
    yvals_o7 = np.append(normphotonspec_o7, np.array([normphotonspec_o7[0], normphotonspec_o7[0] ])) 
    noise_o7 = np.append(normphotonnoise_o7, np.array([normphotonnoise_o7[0],normphotonnoise_o7[0] ]))
    yvals_o7_lo = yvals_o7 - noise_o7
    yvals_o7_hi = yvals_o7 + noise_o7
    xvals_o7 = np.arange(len(normphotonspec_o7) + 2) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    yvals_o8 = np.append(normphotonspec_o8, np.array([normphotonspec_o8[0], normphotonspec_o8[0]]))
    noise_o8 = np.append(normphotonnoise_o8, np.array([normphotonnoise_o8[0], normphotonnoise_o8[0]]))
    yvals_o8_lo = yvals_o8 - noise_o8
    yvals_o8_hi = yvals_o8 + noise_o8
    xvals_o8 = np.arange(len(normphotonspec_o8) + 2) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    
    ax.plot(xvals_o7, np.ones(len(xvals_o7)), color='gray', linestyle='solid') # indicate continuum level
    ax.step(xvals_o7, yvals_o7, where='post', color=clist[0])
    ax.fill_between(xvals_o7, yvals_o7_lo, yvals_o7_hi, step='post', color=clist[0], alpha=0.2 )    
    ax.step(xvals_o8, yvals_o8, where='post', color=clist[1], linestyle='dashed')
    ax.fill_between(xvals_o8, yvals_o8_lo, yvals_o8_hi, step='post', color=clist[1], alpha=0.2 )
    maxx = max([maxx, xvals_o7[-2] + halfpixsize_v, xvals_o8[-2] + halfpixsize_v])
    
    ## Ideal spectrum
    ax = axes[2]
    ax.set_xlabel(vlabel, fontsize=fontsize)
    ax.text(0.98,0.07, r'(c)', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right')
    
    xvals = vvals
    yvals  = [sl.spectra[getspecname(ion, 'Flux')] for ion in ions]
    if 'o8' in ions:
        yvals_o8d = sl.spectra['o8/FluxCorr']

    for ionind in range(len(ions)):
        ax.plot(xvals, yvals[ionind], color=clist[ionind])
    if 'o8' in ions:
        ax.plot(xvals, yvals_o8d, color=clist[1], linestyle = 'dashed')
           
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=True, labelleft=True, labelright=False, color='black')
    maxx = max(maxx, vvals[-1])
    
    # sync x ranges
    for ax in axes:
        ax.set_xlim(0, maxx)
    
    # general info: col. dens. and EW for each ion
    lax.axis('off')
    coldens = [sl.spectra[getspecname(ion,'coldens')] for ion in ions]   
    EWs = []
    bbox = dict(facecolor='white', alpha=0.3)
    for ionind in range(len(ions)):
        ion = ions[ionind]
        if ion == 'o8':
            corr = True
        else:
            corr = False
        EWs += [sl.getEW(ion, corr=corr, vrange=None, vvals=None)]
        # add info to relevant plots  
        ypos = 0.05 + 0.35*ionind
        lax.text(0.00, ypos, r'$EW = %s\, \mathrm{m\AA}$'%(niceformat_mm(EWs[ionind]/1.e-3)), fontsize=fontsize, transform=lax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=None, color=clist[ionind])
        lax.text(0.20, ypos, r'$\log_{10}N = %.1f \, \mathrm{cm}^{-2}$'%(coldens[ionind]), fontsize=fontsize, transform=lax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=None, color=clist[ionind])

    legend_handles = [mlines.Line2D([], [], color=clist[ionind], linestyle = 'solid', label=r'$\mathrm{%s}$'%(ionlabels[ions[ionind]])) for ionind in range(len(ions))]
    ncols = len(ions)
    if 'o8' in ions:
        legend_handles += [mlines.Line2D([], [], color=clist[1], linestyle = 'dashed', label=r'$\mathrm{%s}$d'%(ionlabels['o8']))]
        ncols += 1
    lax.legend(handles=legend_handles, fontsize=fontsize, ncol=2, loc='lower right', bbox_to_anchor=(1.0, 0.0))
       
    plt.savefig(mdir+ '%s_spectrum%i_ideal_and_arcus_athena_standardmocks%s.pdf'%(samplename, sightline, nameadd), bbox_inches = 'tight', format = 'pdf') 

def plotspectra_o78_mocksonly(sightlinedct, sample_in=None, auxdata=None, nameadds_dct=None):
    '''
    calls plotspectrum_o78_mocksonly, uses dct from getselections_by_coldens 
    and names for the cdens bins in naming 
    !! spectrum data from standardspectra is hard-coded! change if parameters change. 
    '''
    for ion in sightlinedct.keys():
        for cind in range(len(sightlinedct[ion])):
            for sightline in sightlinedct[ion][cind]:
                if nameadds_dct is not None:
                    nameadd = '_%s_%s'%(ion, nameadds_dct[ion][cind])
                else:
                    nameadd = ''
                plotspectrum_o78_mocksonly(sightline, sample_in=None, auxdata=None, nameadd=nameadd)
                plt.close() # avoid going over max number of plots

def savemocks_o78(sightline, hdf5name):  
    '''
    standardspectrum instrument data to save is hard-coded here, not retieved!
    '''
    outfile = h5py.File('/net/luttero/data2/paper1/%s.hdf5'%hdf5name, 'a')
    grp = outfile.create_group('mockspectra_sample3/Spectrum%i'%sightline)
    
    sample = SpecSample('/net/luttero/data2/specwizard_data/sample3/spec.snap_027_z000p101.0.hdf5', specnums=[sightline])
    sl = sample.sldict[sightline]
    includeions = ['o7', 'o8']

    vvals = sample.velocity # km/s
    pvals = vvals*1.e5 # rest-frame velocity in km/s ->  rest-frame velocity in cm/s
    pvals /= cu.Hubble(sample.cosmopars['z'],cosmopars=sample.cosmopars) * cu.c.cm_per_mpc # pMpc (local hubble flow means local size)
    pvals *= (1. + sl.cosmopars['z']) # convert to cMpc
    print(max(pvals))
    grp.create_dataset('vrest_kmps', data=vvals)
    grp.create_dataset('lospos_cMpc', data=pvals)
    
    #fig, axes = plt.subplots(nrows=9, ncols=nlines, squeeze=False, , gridspec_kw = {'top': 0.95, 'bottom':0.05, 'left':0.05, 'right':0.95, 'hspace':0.25, 'wspace':0.02})

    for ion in includeions:
        sl.getspectrum(ion,'coldens')
        sl.getspectrum(ion,'Flux')
    sl.getspectrum('o8','Flux', corr=True)
        
    halfpixsize_v = 0.5*np.average(np.diff(vvals))
    ## Ideal spectrum
    yvals_o7 = sl.spectra[getspecname('o7', 'Flux')]
    yvals_o8 = sl.spectra[getspecname('o8', 'Flux')]
    yvals_o8d = sl.spectra['o8/FluxCorr']
    grp.create_dataset('flux_o7', data=yvals_o7)
    grp.create_dataset('flux_o8oneline', data=yvals_o8)
    grp.create_dataset('flux_o8doublet', data=yvals_o8d)
    
    grp.attrs.create('log10_N_cm^-2_o7', sl.spectra[getspecname('o7','coldens')])
    grp.attrs.create('log10_N_cm^-2_o8', sl.spectra[getspecname('o8','coldens')])
    grp.attrs.create('EWrest_A_o7', sl.getEW('o7', corr=False, vrange=None, vvals=None))
    grp.attrs.create('EWrest_A_o8doublet', sl.getEW('o8', corr=True, vrange=None, vvals=None))
    
    ## Athena mock: ion-specific    
    grp_ath = grp.create_group('Athena_mocks')
    grp_ath.attrs.create('exposure_time_ks', 100.)
    grp_ath.attrs.create('Aeff_cm2', 1.05e4)
    grp_ath.attrs.create('fwhm_eV', 2.1)
    grp_ath.attrs.create('spectrum_Emintomax_cgs', 1.0e-11)
    grp_ath.attrs.create('spectrum_Emin_keV', 2.)
    grp_ath.attrs.create('spectrum_Emax_keV', 10.)
    grp_ath.attrs.create('spectrum_photonindex', 1.8)
    
    grp_o7 = grp_ath.create_group('o7')
    grp_o8 = grp_ath.create_group('o8doublet')
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    photonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7, photons_per_pix_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Athena')
    photonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8, photons_per_pix_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Athena')
    xvals_o7 = np.arange(len(photonspec_o7)) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    xvals_o8 = np.arange(len(photonspec_o8)) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    grp_o7.create_dataset('photoncounts', data=photonspec_o7)
    grp_o7.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o7))
    grp_o7.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o7)
    grp_o7.create_dataset('normednoise_lohi', data=normphotonnoise_o7)
    grp_o7.create_dataset('pixcenters_vrest_kmps', data=xvals_o7)
    grp_o8.create_dataset('photoncounts', data=photonspec_o8)
    grp_o8.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o8))
    grp_o8.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o8)
    grp_o8.create_dataset('normednoise_lohi', data=normphotonnoise_o8)
    grp_o8.create_dataset('pixcenters_vrest_kmps', data=xvals_o8)
    
    ## Arcus mock: ion-specific    
    grp_arc = grp.create_group('Arcus_mocks')
    grp_arc.attrs.create('exposure_time_ks', 100.)
    grp_arc.attrs.create('Aeff_cm2', 250.)
    grp_arc.attrs.create('spectral_resolution_above_21p6A', 2500.)
    grp_arc.attrs.create('spectral_resolution_below_21p6A', 2000.)
    grp_arc.attrs.create('spectrum_Emintomax_cgs', 1.0e-11)
    grp_arc.attrs.create('spectrum_Emin_keV', 2.)
    grp_arc.attrs.create('spectrum_Emax_keV', 10.)
    grp_arc.attrs.create('spectrum_photonindex', 1.8)
    
    grp_o7 = grp_arc.create_group('o7')
    grp_o8 = grp_arc.create_group('o8doublet')
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    photonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7, photons_per_pix_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Arcus')
    photonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8, photons_per_pix_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Arcus')
    xvals_o7 = np.arange(len(photonspec_o7)) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    xvals_o8 = np.arange(len(photonspec_o8)) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    grp_o7.create_dataset('photoncounts', data=photonspec_o7)
    grp_o7.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o7))
    grp_o7.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o7)
    grp_o7.create_dataset('normednoise_lohi', data=normphotonnoise_o7)
    grp_o7.create_dataset('pixcenters_vrest_kmps', data=xvals_o7)
    grp_o8.create_dataset('photoncounts', data=photonspec_o8)
    grp_o8.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o8))
    grp_o8.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o8)
    grp_o8.create_dataset('normednoise_lohi', data=normphotonnoise_o8)
    grp_o8.create_dataset('pixcenters_vrest_kmps', data=xvals_o8)
    
    ## Lynx mock: ion-specific    
    grp_arc = grp.create_group('Lynx_mocks')
    grp_arc.attrs.create('exposure_time_ks', 100.)
    grp_arc.attrs.create('Aeff_cm2', 4.e3)
    grp_arc.attrs.create('spectral_resolution', 5000.)
    grp_arc.attrs.create('spectrum_Emintomax_cgs', 1.0e-11)
    grp_arc.attrs.create('spectrum_Emin_keV', 2.)
    grp_arc.attrs.create('spectrum_Emax_keV', 10.)
    grp_arc.attrs.create('spectrum_photonindex', 1.8)
    
    grp_o7 = grp_arc.create_group('o7')
    grp_o8 = grp_arc.create_group('o8doublet')
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    photonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7, photons_per_pix_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Lynx')
    photonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8, photons_per_pix_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Lynx')
    xvals_o7 = np.arange(len(photonspec_o7)) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    xvals_o8 = np.arange(len(photonspec_o8)) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    grp_o7.create_dataset('photoncounts', data=photonspec_o7)
    grp_o7.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o7))
    grp_o7.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o7)
    grp_o7.create_dataset('normednoise_lohi', data=normphotonnoise_o7)
    grp_o7.create_dataset('pixcenters_vrest_kmps', data=xvals_o7)
    grp_o8.create_dataset('photoncounts', data=photonspec_o8)
    grp_o8.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o8))
    grp_o8.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o8)
    grp_o8.create_dataset('normednoise_lohi', data=normphotonnoise_o8)
    grp_o8.create_dataset('pixcenters_vrest_kmps', data=xvals_o8)
    
    outfile.close()
    

def savemocks_o78_proposal_JKFN(sightline, hdf5name):  
    '''
    standardspectrum instrument data to save is hard-coded here, not retieved!
    '''
    outfile = h5py.File('/home/wijers/Documents/papers/esa2050_jelle_and_fabrizio/%s.hdf5'%hdf5name, 'a')
    grp = outfile.create_group('mockspectra_sample3/Spectrum%i'%sightline)
    
    sample = SpecSample('/net/luttero/data2/specwizard_data/sample3/spec.snap_027_z000p101.0.hdf5', specnums=[sightline])
    sl = sample.sldict[sightline]
    includeions = ['o7', 'o8']

    vvals = sample.velocity # km/s
    pvals = vvals*1.e5 # rest-frame velocity in km/s ->  rest-frame velocity in cm/s
    pvals /= cu.Hubble(sample.cosmopars['z'],cosmopars=sample.cosmopars) * cu.c.cm_per_mpc # pMpc (local hubble flow means local size)
    pvals *= (1. + sl.cosmopars['z']) # convert to cMpc
    print(max(pvals))
    grp.create_dataset('vrest_kmps', data=vvals)
    grp.create_dataset('lospos_cMpc', data=pvals)
    
    #fig, axes = plt.subplots(nrows=9, ncols=nlines, squeeze=False, , gridspec_kw = {'top': 0.95, 'bottom':0.05, 'left':0.05, 'right':0.95, 'hspace':0.25, 'wspace':0.02})

    for ion in includeions:
        sl.getspectrum(ion,'coldens')
        sl.getspectrum(ion,'Flux')
    sl.getspectrum('o8','Flux', corr=True)
        
    halfpixsize_v = 0.5*np.average(np.diff(vvals))
    ## Ideal spectrum
    yvals_o7 = sl.spectra[getspecname('o7', 'Flux')]
    yvals_o8 = sl.spectra[getspecname('o8', 'Flux')]
    yvals_o8d = sl.spectra['o8/FluxCorr']
    grp.create_dataset('flux_o7', data=yvals_o7)
    grp.create_dataset('flux_o8oneline', data=yvals_o8)
    grp.create_dataset('flux_o8doublet', data=yvals_o8d)
    
    grp.attrs.create('log10_N_cm^-2_o7', sl.spectra[getspecname('o7','coldens')])
    grp.attrs.create('log10_N_cm^-2_o8', sl.spectra[getspecname('o8','coldens')])
    grp.attrs.create('EWrest_A_o7', sl.getEW('o7', corr=False, vrange=None, vvals=None) * 1e-3)
    grp.attrs.create('EWrest_A_o8doublet', sl.getEW('o8', corr=True, vrange=None, vvals=None) * 1e-3)
    
    grp.attrs.create('position_cMpc', np.array(sl.fracpos) * sl.slicelength)
    
    ## Athena mock: ion-specific    
    grp_ath = grp.create_group('Athena_mocks')
    grp_ath.attrs.create('exposure_time_ks', 100.)
    grp_ath.attrs.create('Aeff_cm2', 1.05e4)
    grp_ath.attrs.create('fwhm_eV', 2.1)
    grp_ath.attrs.create('spectrum_Emintomax_cgs', 1.0e-11)
    grp_ath.attrs.create('spectrum_Emin_keV', 2.)
    grp_ath.attrs.create('spectrum_Emax_keV', 10.)
    grp_ath.attrs.create('spectrum_photonindex', 1.8)
    
    grp_o7 = grp_ath.create_group('o7')
    grp_o8 = grp_ath.create_group('o8doublet')
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    photonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7, photons_per_pix_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='Athena', pix_per_fwhm=3.)
    photonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8, photons_per_pix_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='Athena', pix_per_fwhm=3.)
    xvals_o7 = np.arange(len(photonspec_o7)) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    xvals_o8 = np.arange(len(photonspec_o8)) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    grp_o7.create_dataset('photoncounts', data=photonspec_o7)
    grp_o7.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o7))
    grp_o7.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o7)
    grp_o7.create_dataset('normednoise_lohi', data=normphotonnoise_o7)
    grp_o7.create_dataset('pixcenters_vrest_kmps', data=xvals_o7)
    grp_o8.create_dataset('photoncounts', data=photonspec_o8)
    grp_o8.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o8))
    grp_o8.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o8)
    grp_o8.create_dataset('normednoise_lohi', data=normphotonnoise_o8)
    grp_o8.create_dataset('pixcenters_vrest_kmps', data=xvals_o8)
    
    
    ## proposed instrument mock: ion-specific    
    grp_arc = grp.create_group('prop_mocks')
    grp_arc.attrs.create('exposure_time_ks', 100.)
    grp_arc.attrs.create('Aeff_cm2', 1.5e3)
    grp_arc.attrs.create('spectral_resolution', 10000.)
    grp_arc.attrs.create('spectrum_Emintomax_cgs', 1.0e-11)
    grp_arc.attrs.create('spectrum_Emin_keV', 2.)
    grp_arc.attrs.create('spectrum_Emax_keV', 10.)
    grp_arc.attrs.create('spectrum_photonindex', 1.8)
    
    grp_o7 = grp_arc.create_group('o7')
    grp_o8 = grp_arc.create_group('o8doublet')
    spectrum_o8d = sl.spectra['o8/FluxCorr']
    spectrum_o7  = sl.spectra['o7/Flux']
    photonspec_o7, normphotonnoise_o7, pixsize_rf_kmps_o7, photons_per_pix_o7 =  getstandardspectra(vvals, spectrum_o7, 'o7', sl.cosmopars['z'], instrument='proposed-FN-JK', pix_per_fwhm=2.)
    photonspec_o8, normphotonnoise_o8, pixsize_rf_kmps_o8, photons_per_pix_o8 =  getstandardspectra(vvals, spectrum_o8d, 'o8', sl.cosmopars['z'], instrument='proposed-FN-JK', pix_per_fwhm=2.)
    xvals_o7 = np.arange(len(photonspec_o7)) * pixsize_rf_kmps_o7 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    xvals_o8 = np.arange(len(photonspec_o8)) * pixsize_rf_kmps_o8 - halfpixsize_v #correct for centre offset between value -> edge spectra and value -> bin downsample to bin edges
    grp_o7.create_dataset('photoncounts', data=photonspec_o7)
    grp_o7.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o7))
    grp_o7.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o7)
    grp_o7.create_dataset('normednoise_lohi', data=normphotonnoise_o7)
    grp_o7.create_dataset('pixcenters_vrest_kmps', data=xvals_o7)
    grp_o8.create_dataset('photoncounts', data=photonspec_o8)
    grp_o8.create_dataset('photoncounts_poisson', data=np.random.poisson(photonspec_o8))
    grp_o8.attrs.create('unabsorbed_photons_per_pix', photons_per_pix_o8)
    grp_o8.create_dataset('normednoise_lohi', data=normphotonnoise_o8)
    grp_o8.create_dataset('pixcenters_vrest_kmps', data=xvals_o8)
    
    outfile.close()

def savemocks_o78_totxt(sightlines):  
    '''
    format for Fabrizio Nicastro's further processing
    '''
    
    outdir = '/home/wijers/Documents/papers/esa2050_jelle_and_fabrizio/'
    basename = 'sample3_spectrum%i_%s.txt'
    genname = outdir + 'metadata3.txt'
    fg = open(genname, 'a+')
       
    sample = SpecSample('/net/luttero/data2/specwizard_data/sample3/spec.snap_027_z000p101.0.hdf5', specnums=sightlines)
    
    for sightline in sightlines:
        outname = outdir + basename%(sightline, '%s')
        sl = sample.sldict[sightline]
        includeions = ['o7', 'o8']
        fo_o7 = open(outname%'o7', 'w')
        fo_o8 = open(outname%'o8d', 'w')
        
        vvals = sample.velocity # km/s
        #pvals = vvals*1.e5 # rest-frame velocity in km/s ->  rest-frame velocity in cm/s
        #pvals /= cu.Hubble(sample.cosmopars['z'],cosmopars=sample.cosmopars) * cu.c.cm_per_mpc # pMpc (local hubble flow means local size)
        #pvals *= (1. + sl.cosmopars['z']) # convert to cMpc
        #print (max(pvals))
        #grp.create_dataset('vrest_kmps', data=vvals)
        #grp.create_dataset('lospos_cMpc', data=pvals)
        
        #fig, axes = plt.subplots(nrows=9, ncols=nlines, squeeze=False, , gridspec_kw = {'top': 0.95, 'bottom':0.05, 'left':0.05, 'right':0.95, 'hspace':0.25, 'wspace':0.02})
    
        for ion in includeions:
            sl.getspectrum(ion,'coldens')
            sl.getspectrum(ion,'Flux')
        sl.getspectrum('o8','Flux', corr=True)
            
        #halfpixsize_v = 0.5*np.average(np.diff(vvals))
        ## Ideal spectrum
        yvals_o7 = sl.spectra[getspecname('o7', 'Flux')]
        #yvals_o8 = sl.spectra[getspecname('o8', 'Flux')]
        yvals_o8d = sl.spectra['o8/FluxCorr']
        
        lambdas_o7 = sp.lambda_rest['o7'] * (1 + sl.cosmopars['z']) * (1. + vvals / (cu.c.c / 1.e5))
        lambdas_o8 = sp.lambda_rest['o8'] * (1 + sl.cosmopars['z']) * (1. + vvals / (cu.c.c / 1.e5))
        
        # log sightline metadata:
        fg.write('\n')
        fg.write('sample 3, Spectrum %i :\n'%sightline)
        logN_o7 =  sl.spectra[getspecname('o7','coldens')]
        logN_o8 =  sl.spectra[getspecname('o8','coldens')]
        EW_o7  = sl.getEW('o7', corr=False, vrange=None, vvals=None)
        EW_o8d = sl.getEW('o8', corr=True, vrange=None, vvals=None)
        
        fg.write('log N_{O VII, tot} [cm^-2] = %s\n'%logN_o7)
        fg.write('log N_{O VII, tot} [cm^-2] = %s\n'%logN_o8)
        fg.write('EW_{O VII} [A, rest-frame] = %s\n'%EW_o7)
        fg.write('EW_{O VIII doublet} [A, rest-frame] = %s\n'%EW_o8d)
        
        for li in range(len(lambdas_o7)):
            fo_o7.write('%s\t%s\n'%(lambdas_o7[li], yvals_o7[li]))
        
        for li in range(len(lambdas_o8)):
            fo_o8.write('%s\t%s\n'%(lambdas_o8[li], yvals_o8d[li]))
        
        fo_o7.close()
        fo_o8.close()
        fg.flush()
    fg.close()
    
def get_absorbed_spectrum(specnum, ions, inname=sdir + 'sample3/spec.snap_027_z000p101.0.hdf5',\
                          blazarparams={},\
                          savedir_img=None,\
                          Eminincl_keV=0.2, Emaxincl_keV=12., spacing_keV=0.00025):
    '''
    Blazar spectrum parameters: normalisation: flux in erg/s/cm^2 between Emin 
                                               and Emax 
                                               Etotcgs=7.5e-12, 
                                               EminkeV=2., 
                                               EmaxkeV=10. 
                                PL index:      Gammaphot=2.
    spacing_keV: spectral resolution (units: keV, default: 0.25 eV) 
    returns: bin edges between which spectrum is evaluated 
             (keV, observer frame), 
             spectrum in photons/ cm2 / s / keV   
             calls plot_multiion -> supplies plot of input normalised spectrum                      

    note: input spectra are periodic, output spectra are not, so some sort of 
          wrapping to deal with cut-in-half absorption lines will probably be 
          necessary down the line somewhere
    '''
    # simput ascii spectrum
    # 
    
    #sample = SpecSample(inname)
    #try:
    #    speclist = list(specnum)
    #except TypeError:
    #    speclist = [specnum]
    #sample.addspecs(speclist)
    #sample.getspectra(ion, qty='Flux', corr=True)
    #spectrum = sample.sldict[specnum].spectra[getspecname(ion, 'Flux')]
    #vvals = sample.velocity * 1.e5 # km/s -> cgs cm/s
    # z = sample.cosmopars['z']
    #Evals =  cu.c.c * cu.c.planck / (sp.lambda_rest[ion] * 1.e-8 * (1. + z) * (1. +  vvals / cu.c.c)) / (cu.c.ev_to_erg * 1e3) # total redshift = cosmo redshift * doppler redshift

    #interm_minspacing_keV = 0.025e-3 # to sample the lines at sufficient uniform resolution before binning to higher

    sl = plot_multiline(inname, specnum, ions, space='energy', returnobj=True, savedir=savedir_img)
    z = sl.cosmopars['z']
    Evals = (sl.spectra['multiline_0']['E_keV'] / (1. + z)) # apply redshift
    
    # order low-to-high energy
    spectrum = sl.spectra['multiline_0']['flux']  
    
    # set up initial high-res grid -> downgrade by reshape-sum later
    diff = np.average(np.diff(Evals))
    init_spacing = diff # min(diff, interm_minspacing_keV) / 3.
    numoversample = max(int(np.floor(spacing_keV / init_spacing)), 1)
    spacing_interm_keV = init_spacing # spacing_keV / float(numoversample)
    if 0.9 * Eminincl_keV < Evals[0]:
        Evals_temp = np.append(np.arange(Evals[0] - init_spacing, 0.9 * Eminincl_keV - init_spacing, -1 * init_spacing)[::-1], Evals)
    else:
        Evals_temp = Evals
    if 1.1 * Emaxincl_keV > Evals[-1]:
        Evals_temp = np.append(Evals_temp, np.arange(Evals[-1] + init_spacing, 1.1 * Emaxincl_keV + init_spacing, init_spacing))
    Evals_interm = Evals_temp #np.arange(Eminincl_keV - 0.5 * spacing_interm_keV, Emaxincl_keV + spacing_keV + spacing_interm_keV, spacing_interm_keV)
    if len(Evals_interm) % numoversample != 0:
        Evals_interm = np.append(Evals_interm, Evals_interm[-1] + spacing_interm_keV * (1. + np.arange(numoversample - len(Evals_interm) % numoversample)) )
    #Evals_out = np.arange(Eminincl_keV, Emaxincl_keV + np.abs(diff), np.abs(diff))
    #print Evals_out
    
    #absspec_out = scipy.interpolate.griddata(Evals, spectrum, Evals_out, method='cubic')
    minind = np.max(np.where(Evals_interm <= np.min(Evals))[0])
    maxind = np.min(np.where(Evals_interm >= np.max(Evals))[0])
    #print minind, maxind    
    wheremult = slice(minind, maxind + 1, None)
    basespectrum = blazarflux(Evals_interm, **blazarparams)
  
    basespectrum[wheremult] *= spectrum
    
    
    # problem: even at 0.25 eV resolution, velocity resolution is ~150 km/s 
    # at 0.5 keV, so lines might be missed entirely by interpolating instead of
    # binning
    
    # oversample spectrum by factor numoversample, then average flux overdensity onto the output grid
    #spectrum_interm = scipy.interpolate.griddata(Evals_temp, basespectrum,\
    #                                          Evals_interm, method='linear'
    spectrum_out = np.sum(basespectrum.reshape(len(basespectrum) // numoversample , numoversample), axis=1) / float(numoversample)
    Evals_out = Evals_interm[0] - 0.5 * spacing_interm_keV + spacing_interm_keV * float(numoversample) * np.arange(len(basespectrum) // numoversample + 1)    
                                             
    return Evals_out, spectrum_out


## test whether we need damping wings
    
def plotdiff_damped_vs_gaussian_ions():
    outname = '/net/luttero/data2/specwizard_data/damping_wing_check_paper2ions.pdf'
    fig, (ax, lax) = plt.subplots(nrows=1, ncols=2,\
         gridspec_kw={'width_ratios': [1., 0.2]})
    fontsize=12
    
    ax.set_xlabel('$\\log_{{10}} \\, \\mathrm{{N}} \\; \\mathrm{{cm}}^{{-2}}$',\
                  fontsize=fontsize)
    ax.set_ylabel('$\\log_{{10}}$ damping wing EW / gaussian EW', fontsize=fontsize)
    
    Nvals = np.arange(12., 20.05, 0.1)
    bvals = np.array([5., 10., 20., 50., 100., 200.])
    dashes = [[6, 2], [3, 1], [1, 1],\
              [6, 2, 3, 2], [6, 2, 1, 2], [3, 1, 1, 1]]
    lines = [ild.o6major, ild.o7major, ild.o8doublet,\
             ild.ne8major, ild.ne9major, ild.fe17major]
    colors  = {'o7': 'C3',\
             'o8': 'C0',\
             'o6': 'C2',\
             'ne8': 'C1',\
             'ne9': 'C9',\
             'hneutralssh': 'C6',\
             'fe17': 'C4'}
    thick = {'o6': 5.,\
             'o7': 50.,\
             'o8': 50.,\
             'ne8': 20.,\
             'ne9': 50.,\
             'fe17': 50.,\
             }
    offset = 1.
    offset_step = 1.
    ax.tick_params(which='both', labelsize=fontsize-1, direction='in',\
                   right=True, left=True)
    ax.minorticks_on()
    
    for line in lines:
        try:
            ion = line.ion
        except AttributeError:
            ion = line.major.ion
        color = colors[ion]
        for bi in range(len(bvals)):
            bval = bvals[bi]
            pattern = dashes[bi]
            dmp = ild.linflatdampedcurveofgrowth_inv(10**Nvals, bval * 1e5, line)
            lnf = ild.linflatcurveofgrowth_inv_faster(10**Nvals, bval * 1e5, line)
            lw = 2.5 if thick[ion] == bval else 1.5
            ax.plot(Nvals, np.log10(dmp/lnf) + offset, dashes=pattern,\
                    color=color, linewidth=lw)
            
        ax.axhline(offset, linestyle='solid', color=color)
        ax.text(Nvals[0], offset, ild.getnicename(ion),
                fontsize=fontsize, color=color, verticalalignment='bottom',
                horizontalalignment='left')
        offset += offset_step
    
    handles = [mlines.Line2D([], [], color='gray', dashes=dashes[bi],\
                             label='{b:.0f}'.format(b=bvals[bi]))
               for bi in range(len(bvals))]
    leg = lax.legend(handles=handles, fontsize=fontsize, ncol=1,\
              loc='upper left', bbox_to_anchor=(0.05, 0.95))
    leg.set_title('b [km/s]', prop={'size': fontsize})
    lax.axis('off')
    
    plt.savefig(outname, bbox_inches='tight', format='pdf')