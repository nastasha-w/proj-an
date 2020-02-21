#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 17:22:42 2020

@author: wijers

based on specwiz_proc.py: 
for analysis of a large set of specwizard sightlines in one go
adapted to add velocity shifts, and leave out some of the stuff fromm older 
projects
"""

## imports
import numpy as np
import h5py
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp

import cosmo_utils as cu
import ion_line_data as ild
import make_maps_opts_locs as ol

## defaults
sdir = ol.sdir
ldir = '/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/'
ndir = ol.ndir
pdir = ol.pdir
mdir = ol.mdir

# based on resolved/unresolved multiplets, strongest lines
ion_default_lines = {'o6': (ild.o6major,),\
                     'o7': (ild.o7major,),\
                     'o8': (ild.o8major, ild.o8minor),\
                     'ne8': (ild.ne8major,),\
                     'ne9': (ild.ne9major,),\
                     'fe17': (ild.fe17major,),\
                     }

# extract groups containing spectra
# spectrum files contained in groups Spectrum<number>
# loading from luttero takes a while on quasar; 
# using a local file copy does help, but it's still not fast
class SpecSet:
    def __init__(self, filename, getall=False, ionlines=ion_default_lines):
        '''
        filename: name of the file containing the specwizard output
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        getall:   get total column densities and EWs
        '''
        if filename[:len(sdir)] == sdir or '/' in filename:
            self.filename = filename
        else:
            self.filename = sdir + filename
        self.specfile = h5py.File(self.filename, 'r')
        self.specgroups = np.array(self.specfile.keys())
        self.isspec = np.array(['Spectrum' in group for group in self.specgroups])
        self.numspecs = np.sum(self.isspec)
        if not self.numspecs > 0:
            raise ValueError('The specified file {filen} contains no spectra'.format(filen=self.filename))
        self.specgroups = self.specgroups[self.isspec] 
        self.specgroups = np.array(sorted(self.specgroups,\
                                          key=lambda name: int(name[8:])))
        del self.isspec
        self.ionlines = ionlines
        
        self.cosmopars = \
            {'boxsize':     self.specfile['/Header'].attrs.get('BoxSize'),\
             'h':           self.specfile['/Header'].attrs.get('HubbleParam'),\
             'a':           self.specfile['/Header'].attrs.get('ExpansionFactor'),\
             'z':           self.specfile['/Header'].attrs.get('Redshift'),\
             'omegam':      self.specfile['/Header'].attrs.get('Omega0'),\
             'omegalambda': self.specfile['/Header'].attrs.get('OmegaLambda'),\
             'omegab':      self.specfile['/Header'].attrs.get('OmegaBaryon') }
        
        ## set general sightline and box properties
        # positions in gagdet units = 1/h cMpc
        self.positions = np.array([[
                     self.specfile[specgroup].attrs['X-position'],\
                     self.specfile[specgroup].attrs['Y-position']\
                                   ] for specgroup in self.specgroups]) 
        # convert to cMpc
        self.positions *= 1. / self.cosmopars['h']
        self.slicelength = self.cosmopars['boxsize'] / self.cosmopars['h'] #in cMpc
        # observed Delta z, since the comoving slice length is used
        self.deltaredshift_obs = cu.Hubble(self.cosmopars['z'],\
                                       cosmopars=self.cosmopars) \
                            / cu.c.c * self.slicelength * cu.c.cm_per_mpc 
        
        self.numspecpix = self.specfile['VHubble_KMpS'].shape[0]

        # set dicts to add to per ion
        self.spectra = {}
        self.tau     = {}
        self.coldens = {}
        self.EW      = {}
        self.nion    = {}
        self.posmassw= {}
        self.posionw = {}
        self.veltauw = {}
        self.tau_base     = {}
        self.base_spectra = {}     
        self.aligned = {}
        
        # if the file has no lines of sight, it's useless anyway
        self.ions = np.array(self.specfile['Spectrum0'].keys()) 
        self.ision = np.array(['RealSpaceMass' not in group for group in self.ions])
        self.ions = self.ions[self.ision]
        del self.ision
        self.aligned = {ion: False for ion in self.ions}
 
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

        self.getlinedata_in()
        if getall:
            self.getall()
    
    def getall(self, dions='all'):
        self.getspectra(dions=dions)
        self.getcoldenstot(dions=dions)
        self.getEWtot(dions=dions)
        
    def getlinedata_in(self):
        # header also contains unused lines: only the first listed line is used
        # for each ion
        _ions_in = np.array(self.specfile['Header'].attrs['Ions'])
        self.ions_in, _inds = np.unique(_ions_in, return_index=True) 
        self.fosc_in = np.array(self.specfile['Header'].attrs['Transitions_Oscillator_Strength'])[_inds]
        self.lambda_in = np.array(self.specfile['Header'].attrs['Transitions_Rest_Wavelength'])[_inds]
        self.linedata_in = {self.ions_in[i]: {'lambda_rest': self.lambda_in[i],\
                                              'fosc': self.fosc_in[i]}\
                            for i in range(len(self.ions))}
                 
    def getcoldenstot(self, dions='all'):

       # take total column density from file
        if dions == 'all':
            dions = self.ions
        self.coldens.update({ion: \
                               np.array([ \
                                 self.specfile['{grpn}/{ion}/LogTotalIonColumnDensity'.format(ion=ion, grpn=specgrp)][()] \
                               for specgrp in self.specgroups])\
                             for ion in dions})
    
    def getspectra(self, dions='all'):
        '''
        dions:    ions to get EWs for
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        spectra are periodic: multiple lines here are intended for overlapping
        multiplets
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        #print(dions)
        self.velperpix_kmps = np.average(np.diff(np.array(self.specfile['VHubble_KMpS'])))
        
        for ion in dions:
            if ion not in self.tau_base.keys():
                self.getquantity('tau', 'ion', dions=[ion])
            fosc_in = self.linedata_in[ion]['fosc']
            lambda_r_in = self.linedata_in[ion]['lambda_rest'] # Angstrom units
            
            fosc_use = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambda_r_use = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            
            veldiff_at_z_kmps = [(lambda_r - lambda_r_in) \
                                  / lambda_r_in * cu.c.c * 1.e-5 
                                 for lambda_r in lambda_r_use]
            pixshifts = [int(veldiff / self.velperpix_kmps + 0.5) \
                         for veldiff in veldiff_at_z_kmps]
            rescalef = (fosc_use * lambda_r_use**2) / \
                       (fosc_in * lambda_r_in**2)
            # rescale, shift copies of tau
            if not (np.allclose(rescalef, 1.) and len(rescalef) == 1):
                tau = np.copy(self.tau_base[ion])[:, :, np.newaxis]
                tau = tau * rescalef[np.newaxis, np.newaxis, :] # cannot do in place with extra dimension
                for li in range(len(pixshifts)):
                    shift = pixshifts[li]
                    tau[:, :, li] = np.roll(tau[:, :, li], shift, axis=1)
                tau = np.sum(tau, axis=2)
            else:
                tau = np.copy(self.tau_base[ion])
            # save results
            self.tau.update({ion: tau})
            self.spectra.update({ion: np.exp(-1 * tau)})
            self.aligned[ion] = False
    
    def alignmaxtau(self, dions='all'):
        '''
        shift all spectra so max. tau is at the central pixel
        if multiple maxima have the exact same value, the first instance in 
        the default zero position is used (argmax default)
        self.centerpix is the pixel index the max values get shifted to
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        
        self.centerpix = self.numspecpix // 2
        for ion in dions:
            if ion not in self.tau:
                self.getspectra(dions=[ion])
            tau = self.tau[ion]
            maxpos = np.argmax(tau, axis=1) 
            rollargs = self.centerpix - maxpos
            for i in range(tau.shape[0]):
                tau[i, :] = np.roll(tau[i, :], rollargs[i])
            self.tau[ion] = tau
            self.spectra.update({ion: np.exp(-1 * tau)})
            self.aligned[ion] = True
            
    def getEWtot(self, dions='all'):
        '''
        dions:    ions to get EWs for
        ionlines: ion: tuple of lines dict. lines are ion_line_data.SpecLine 
                  instances
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        #print(dions)
        for ion in dions:
            if ion not in self.spectra.keys():
                self.getspectra(dions=[ion])
            
            foscs = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambdas = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            lambda_eff = np.sum(foscs  * lambdas) / np.sum(foscs)
            
            # EW = \int dlamdba (1-flux) = (Delta lambda) - (Delta lambda)/N * sum_i=1^N F_normalised(i)  
            self.EW[ion] = 1. - np.sum(self.spectra[ion], axis=1) / float(self.spectra[ion].shape[1])
            self.EW[ion] *= self.deltaredshift_obs / (self.cosmopars['z'] + 1.) * lambda_eff # convert absorbed flux fraction to rest-frame EW
    
    def getcoldens_EW_vwindow(self, deltav_rest_kmps, dions='all'):    
        '''
        dions:            ions to get EWs for
        deltav_rest_kmps: velocity interval (total = 2x maximum offset from max
                          tau). Units: rest-frame km/s
        '''
        if dions == 'all':
            dions = self.ions
        elif isinstance(dions, str): # single ion input as a string in stead of a length-1 iterable
            dions = [dions]
        
        # make sure the stuff we need exists
        toalign = []
        togetcd = []
        for ion in dions:
            if not self.aligned[ion]:
                toalign.append(ion)
            if ion not in self.coldens:
                togetcd.append(ion)
        self.alignmaxtau(dions=toalign)
        self.getcoldenstot(dions=togetcd)
        
        # set up selection ranges
        numpix = int(deltav_rest_kmps / self.velperpix_kmps + 0.5)
        if numpix >= self.numspecpix:
            print('Velocity interval {vel} is larger than the box size'.format(vel=deltav_rest_kmps))
            print('using the whole sightline, without repetitions')
            vsel = slice(None, None, None)
            numpix =  self.numspecpix
        else:
            vsel = slice(self.centerpix - numpix // 2,\
                         self.centerpix - numpix // 2 + numpix,\
                         None)
        pathfrac = float(numpix) / float(self.numspecpix)
        
        # iterate over ions: select region, calculate N, EW
        vkey = deltav_rest_kmps
        if not hasattr(self, 'vwindow_EW'):
            self.vwindow_EW = {vkey: {}}
            self.vwindow_coldens = {vkey: {}}
        else:
            self.vwindow_EW.update({vkey: {}})
            self.vwindow_coldens.update({vkey: {}})
        for ion in dions:           
            foscs = np.array([specl.fosc for specl in self.ionlines[ion]])
            lambdas = np.array([specl.lambda_angstrom for specl in self.ionlines[ion]])
            lambda_eff = np.sum(foscs  * lambdas) / np.sum(foscs)
            
            self.vwindow_EW[vkey][ion] = 1. - np.sum(self.spectra[ion][:, vsel], axis=1) / float(numpix)
            # convert absorbed flux fraction to rest-frame EW
            self.vwindow_EW[vkey][ion] *= pathfrac * self.deltaredshift_obs / (self.cosmopars['z'] + 1.) * lambda_eff
            
            # N \propto tau
            self.vwindow_coldens[vkey][ion] = np.sum(self.tau[ion][:, vsel], axis=1) \
                                              / np.sum(self.tau[ion][:, :], axis=1)            
            self.vwindow_coldens[vkey][ion] *= 10**self.coldens[ion]
            self.vwindow_coldens[vkey][ion] = np.log10(self.vwindow_coldens[vkey][ion])
            
    def save_specdata(self, filename):
        '''
        filename should include the directory
        
        save N, EW totals and windows, 
        as well as used ionlines, cosmopars, specwizard file
        only new vwindow and total values are added
        '''
        
        if os.path.isfile(filename):
            createnew = False
        else:
            createnew = True
        with h5py.File(filename, 'a') as fo:
            if createnew:
                hed = fo.create_group('Header')
                csm = hed.create_group('cosmopars')
                for key in self.cosmopars:
                    csm.attrs.create(key, self.cosmopars[key])
                hed.create_dataset('specgroups', data=self.specgroups.astype(np.string_))
                hed.attrs.create('filename_specwizard', self.specfile)
                # used lines
                inl = hed.create_group('ionlines')
                for ion in self.ionlines:
                    sgp = inl.create_group(ion)
                    for li in range(len(self.ionlines[ion])):
                        s2grp = sgp.create_group('line_{}'.format(li))
                        ild.savelinedata(s2grp, self.ionlines[ion][li])
            
            if 'coldens_tot' not in fo.keys():
                grp = fo.create_group('coldens_tot')
                for ion in self.coldens:
                    grp.create_dataset(ion, data=self.coldens[ion])
                grp.attrs.create('info', np.string_('column density [log10 cm^-2] for each ion'))
            if 'EW_tot' not in fo.keys():
                grp = fo.create_group('EW_tot')
                for ion in self.EW:
                    grp.create_dataset(ion, data=self.EW[ion])
                grp.attrs.create('info', np.string_('equivalent width [Angstrom, rest-frame] for each ion'))
            
            if 'vwindows_maxtau' not in fo.keys():
                vgp = fo.create_group('vwindows_maxtau')
                vgp.attrs.create('info', np.string_('column density and EW in rest-frame velocity windows Deltav (+- 0.5 * Deltav) [km/s] around the maximum-optical-depth pixel'))
            for deltav in self.vwindow_EW:
                vname = 'Deltav_{v:.3f}'.format(deltav)
                if vname in vgp.keys(): # already stored
                    continue
                svgp = vgp.create_group(vname)
                svgp.attrs.create('Deltav_rf_kmps', deltav)
                cvgp = svgp.create_group('coldens')
                cvgp.attrs.create('info', np.string_('column density [log10 cm^-2] for each ion'))
                evgp = svgp.create_group('EW')
                evgp.attrs.create('info', np.string_('equivalent width [Angstrom, rest-frame] for each ion'))
                
                for ion in self.vwindow_EW[deltav]:
                    evgp.create_dataset(ion, data=self.vwindow_EW[deltav][ion])
                    cvgp.create_dataset(ion, data=self.vwindow_coldens[deltav][ion])
        
    def getnion(self, dions='all'): # ion number density in cm^3 in each pixel: 
        if dions == 'all':
            dions = self.ions
        self.nion.update({ion: np.array([self.specfile['Spectrum%i/%s/RealSpaceNionWeighted/NIon_CM3'%(specnum,ion)][()] for specnum in range(self.numspecs)]) for ion in dions})
            
    def getspectra_base(self, dions='all'):
        if dions == 'all':
            dions = self.ions
        self.spectra_base.update({ion: np.array([self.specfile['Spectrum%i/%s/Flux'%(specnum,ion)][()] for specnum in range(self.numspecs)]) for ion in dions})


    def getquantity(self, name, cat, dions='all'):
        '''
        reads the quantity <name>, in the weighting category <cat>, into
        the appropriate dictionary for the ions <dions> (if applicable)
        '''
        if cat not in self.dataoptions.keys():
            raise ValueError('Cat options are %s, not %s. No values retrieved.'%(str(self.dataoptions.keys()), cat))
        elif name not in self.dataoptions[cat].keys():
            raise ValueError('Name options are %s, not %s. No values retrieved.'%(str(self.dataoptions[cat].keys()), name)) 

        elif cat == 'posmassw':
            if name == 'nion':
                self.getnion(dions=dions)
            else:
                self.posmassw.update({name : np.array([np.array(self.specfile['{grn}/{dn}'.format(grn=specgroup, dn=self.dataoptions[cat][name])]) \
                                                       for specgroup in self.specgroups])})
        elif cat == 'ion':
            if name == 'flux': # we have a function for this one
                self.getspectra_base(dions=dions) 
            elif name == 'logcoldens': # and for this one
                self.getcoldens(dions=dions)
            else: # name == 'tau'
                self.tau_base.update({ion: np.array([np.array(self.specfile['{grn}/{ion}/{dn}'.format(grn=specgroup, ion=ion, dn=self.dataoptions[cat][name])]) \
                                                     for specgroup in self.specgroups]) for ion in dions})
        else:
            if cat == 'posionw':
                self.basedict = self.posionw
            elif cat == 'veltauw':
                self.basedict = self.veltauw
            else:
                raise ValueError('{cat} is not a valid cat option. No values retrieved.'.format(cat=cat))
            if name not in self.basedict.keys():
                self.basedict[name] = {}
            for ion in dions:      
                self.basedict[name].update({ion : np.array([np.array(self.specfile['{grn}/{ion}/{dn}'.format(grn=specgroup, ion=ion, dn=self.dataoptions[cat][name])]) \
                             for specgroup in self.specgroups]) for ion in dions})
            del self.basedict           


def plot_NEW(specset, ions, vwindows=None, savename=None):

    alpha = 0.05    
    
    xlabel = '$\\log_{{10}}\\, \\mathrm{{N}}_{{\\mathrm{{{ion}}}}} \\; [\\mathrm{{cm}}^{{-2}}]$'
    ylabel = '$\\mathrm{{EW}}  \; [\\mathrm{{\\AA}}]$'
    fontsize = 12
    
    numions = len(ions)
    numcols = min(3, numions)
    numrows = (numions - 1) // numcols + 1
    
    panelheight = 3.
    panelwidth = 3.
    hspace = 0.3
    wspace = 0.3
    figw = panelwidth * numcols + hspace * (numcols - 1)
    figh = panelheight * numrows + wspace * (numrows - 1)
    
    
    fig = plt.figure(figsize=(figw, figh))
    grid = gsp.GridSpec(nrows=numrows, ncols=numcols, hspace=hspace,\
                        wspace=wspace)    
    axes = [fig.add_subplot(grid[i // numcols, i % numcols]) for i in range(numions)]

    if vwindows is None:
        dolegend = False
        vwindows = [None]
    else:
        dolegend = True
            
    for ii in range(numions):
        ion = ions[ii]
        ax = axes[ii]
        
        if ii % numcols == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if numions - ii <= numcols:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.set_yscale('log')
        
        ax.text(0.95, 0.05, ild.getnicename(ion), fontsize=fontsize,\
                horizontalalignment='right', verticalalignment='bottom',\
                transform=ax.transAxes)
        
        minc = np.inf
        maxc = -np.inf
        for deltav in vwindows:
            if deltav is None:
                lhandle = '{_len:.1f} cMpc'.format(_len=specset.slicelength)
                coldens = specset.coldens[ion]
                EW      = specset.EW[ion]
            else:
                lhandle = '${dv:.1f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'.format(dv=deltav)
                coldens = specset.vwindow_coldens[deltav][ion]
                EW      = specset.vwindow_EW[deltav][ion]
            ax.scatter(coldens, EW, label=lhandle, alpha=alpha, s=5)
            maxc = max(np.max(coldens), maxc)
            minc = min(np.min(coldens), minc)
        cvals = 10**np.linspace(minc, maxc, 100)
        EWvals = ild.lingrowthcurve_inv(cvals, specset.ionlines[ion])
        ax.plot(np.log10(cvals), EWvals, linestyle='dashed', linewidth=2,\
                color='black', label='opt. thin')         
        if dolegend:
            ax.legend(fontsize=fontsize, loc='upper left',\
                      bbox_to_anchor=(0.0, 1.0), frameon=False)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')

def plot_NEW_fracs(specset, ions, vwindows=None, savename=None):
    
    alpha = 0.05    
    
    xlabel_c = '$\\log_{{10}}\\, \\mathrm{{N}}(\\mathrm{{{ion}}}) \\; [\\mathrm{{cm}}^{{-2}}]$'
    xlabel_e = '$\\mathrm{{EW}}(\\mathrm{{{ion}}})  \; [\\mathrm{{\\AA}}]$'
    ylabel = '$\\Delta v$ / tot.'
    fontsize = 11
    
    numions = len(ions)
    numcols = min(3, numions)
    numrows = ((numions - 1) // numcols + 1) * 2
    
    panelheight = 1.5
    panelwidth = 3.
    hspace = 0.25
    wspace = 0.2
    figw = panelwidth * numcols + hspace * (numcols - 1)
    figh = panelheight * numrows + wspace * (numrows - 1)
    
    
    fig = plt.figure(figsize=(figw, figh))
    grid = gsp.GridSpec(nrows=numrows, ncols=numcols, hspace=hspace,\
                        wspace=wspace)    
    axes = [[fig.add_subplot(grid[2 * (ii // numcols) + ti, ii % numcols]) for ti in range(2)] for ii in range(numions)]

    if vwindows is None:
        dolegend = False
        vwindows = [None]
    else:
        dolegend = True
            
    for ii in range(numions):
        ion = ions[ii]
        ax_c = axes[ii][0]
        ax_e = axes[ii][1]
        
        if ii % numcols == 0:
            ax_c.set_ylabel(ylabel, fontsize=fontsize)
            ax_e.set_ylabel(ylabel, fontsize=fontsize)

        ax_c.text(0.05, 0.95, xlabel_c.format(ion=ild.getnicename(ion, mathmode=True)),\
                  fontsize=fontsize, transform=ax_c.transAxes,\
                  horizontalalignment='left', verticalalignment='top')
        ax_e.text(0.05, 0.95, xlabel_e.format(ion='\\mathrm{{{}}}'.format(ild.getnicename(ion, mathmode=True))),\
                  fontsize=fontsize, transform=ax_e.transAxes,\
                  horizontalalignment='left', verticalalignment='top')
        ax_c.tick_params(which='both', direction='in', top=True, right=True,\
                         labelsize=fontsize - 1)
        ax_c.minorticks_on()
        ax_c.set_yscale('log')
        ax_e.tick_params(which='both', direction='in', top=True, right=True,\
                         labelsize=fontsize - 1)
        ax_e.minorticks_on()
        ax_e.set_yscale('log')
        ax_e.set_xscale('log')
        
        #ax_e.text(0.95, 0.95, ild.getnicename(ion), fontsize=fontsize,\
        #          horizontalalignment='right', verticalalignment='top',\
        #          transform=ax_e.transAxes)
        
        base_c = specset.coldens[ion]
        base_e = specset.EW[ion]
        for deltav in vwindows:
            if deltav is None:
                lhandle = '{_len:.1f} cMpc'.format(_len=specset.slicelength)
                coldens = specset.coldens[ion]
                EW      = specset.EW[ion]
            else:
                lhandle = '${dv:.1f} \\, \\mathrm{{km}}\\,\\mathrm{{s}}^{{-1}}$'.format(dv=deltav)
                coldens = specset.vwindow_coldens[deltav][ion]
                EW      = specset.vwindow_EW[deltav][ion]
            ax_c.scatter(base_c, 10**(coldens - base_c), label=lhandle, alpha=alpha, s=5)
            ax_e.scatter(base_e, EW / base_e, label=lhandle, alpha=alpha, s=5)
            
        ax_c.axhline(1., linestyle='dashed', linewidth=2,\
                color='black', label=None)  
        ax_e.axhline(1., linestyle='dashed', linewidth=2,\
                color='black', label=None) 
        if dolegend:
            ax_c.legend(fontsize=fontsize, loc='lower left',\
                      bbox_to_anchor=(0.0, 0.0), frameon=False)
    
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')