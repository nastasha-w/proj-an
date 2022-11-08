#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ctypes as ct
import string
import h5py
import scipy.interpolate as spint 

import make_maps_opts_locs as ol

class linetable_PS20:
    '''
    class for storing data from the Ploeckinger & Schaye (2020) ion balance 
    and line emission tables. 
    
    Methods include functions for interpolating the ion balance and emissvitiy 
    tables in logT, logZ, and lognH, as well as the dust depletion and assumed
    abundance of the given element for a total metallicity value. These table
    interpolations omit the log Z / solarZ = -50.0 values in the tables,
    because these somewhat arbitrary zero values are unsuitable for 
    interpolation.
    
    This does mean that line emission (proportional to the element fraction)
    should be rescaled to the right metallicity (or better: specific element
    abundance) after the interpolation step.
    
    A single instance is for one emission or absorption line. This does mean 
    the table bins and dust table may be stored twice if multiple objects are 
    in use. The instance is also only valid for one redshift.
    '''
    
    def parse_ionname(self):
        '''
        retrieve the element name and ionization stage by interpreting the 
        ion (or emission line) string. Assumes that emission lines match the
        format of the IdentifierLines dataset in the emission line tables 
        (and are not blends), and that absorption lines are speciefied as e.g.
        'o7', 'fe17', or 'Si 4', possibly followed by something other than 
        digits
        
        Returns
        -------
        None.

        '''
        if self.emission: # lines are formatted as in the IdentifierLines dataset
            try:
                self.elementshort = self.ion[:2].strip()
                self.ionstage = int(self.ion[2:4])
                msg = 'Interpreting {line} as coming from the {elt} {stage} ion'
                msg = msg.format(line=self.ion, elt=self.elementshort,
                                 stage=self.ionstage)
                print(msg)
            except:
                msg = 'Failed to parse "{}" as an emission line'
                raise ValueError(msg.format(self.ion))
        else: # ions are '<elt><stage>....'
            try:
                if self.ion == 'hmolssh':
                    self.elementshort = 'H'
                    # get a useful error, I hope
                    self.ionstage = 'invalid index for hmolssh'   
                else:
                    self.elementshort = ''
                    self.ionstage = ''
                    i = 0
                    while not self.ion[i].isdigit():
                        i += 1             
                    self.elementshort = string.capwords(self.ion[:i])
                    self.elementshort = self.elementshort.strip()
                    while self.ion[i].isdigit():
                        self.ionstage = self.ionstage + self.ion[i]
                        i += 1
                        if i == len(self.ion):
                            break
                    self.ionstage = int(self.ionstage)
            except:
                msg = 'Failed to parse "{}" as an ion'
                raise ValueError(msg.format(self.ion))
            msg = 'Interpreting {ion} as the {elt} {stage} ion'
            msg = msg.format(ion=self.ion, elt=self.elementshort,
                             stage=self.ionstage)
            print(msg)
            
    def getmetadata(self):
        '''
        get the solar metallicity (self.solarZ),
        the abundance of the selected element as a function of log Z / solarZ 
        (self.numberfraction_Z),
        and the element mass in atomic mass units
      
        Raises
        ------
        ValueError
            No abundance data is available for self.elementshort

        Returns
        -------
        None.

        '''
        self.table_logzero = -50.0
        
        with h5py.File(self.ionbalfile, 'r') as f:
            # get element abundances (n_i / n_H) for tabulated log Z / Z_solar
            self.numberfractions_Z_elt = f['TotalAbundances'][:, :]
            elts = [elt.decode() for elt in f['ElementNamesShort'][:]]
            if self.elementshort not in elts:
                msg = 'Data for element {elt} is not available'
                msg = msg.format(elt=self.elementshort)
                raise ValueError(msg)
            self.eltind = np.where([self.elementshort == elt \
                                    for elt in elts])[0][0]
            # element number fraction n_i / n_H as a function of metallicity
            # in the tables
            self.numberfraction_Z = np.copy(\
                self.numberfractions_Z_elt[:, self.eltind])
            
            self.element = f['ElementNames'][self.eltind].decode().strip()
            self.elementmass_u = f['ElementMasses'][self.eltind]
            
            # solar Z for scaling
            self.solarZ = f['SolarMetallicity'][0]
    
    def __init__(self, ion, z, emission=False, vol=True,
                 ionbalfile=ol.iontab_sylvia_ssh,
                 emtabfile=ol.emtab_sylvia_ssh,
                 lintable=False):
        '''
        Parameters
        ----------
        ion: string
            ion or emission line to get tables for; emission line names should
            match an entry in the IdentifierLines dataset in the emission line
            table. Ion names should follow the format 
            '<element abbreviation><ionization stage>[other stuff]',
            e.g. 'o7' for the O^{6+} / O VII ion, 'Fe17-other', etc.
            other options are 'hmolssh' and 'h1ssh'. Note that 'h2' is ionized
            hydrogen (H^+), not molecular hydrogen (H_2)
        z: float
            redshift. Must be within the tabulated range.
        emission: bool
            get emission tables (True) or absorption tables (False)
            if emission is True, the absorption table methods for the ion 
            producing the line are also available. Using emission=True for an
            invalid emission line name will produce an error though.
        vol: bool
            Use quantities for the last Cloudy zone (True) or column-averaged
            quantities (False). The column option is only available for the
            hydrogen species.
        ionbalfile: string
            the file (including directory path) containing the ion balance 
            data. This is also needed for emission calculations, since some of
            the metadata is not contained in the emission file.
        emtabfile: string
            the file (including directory path) containing the emissivity 
            data. Only used when calculating emission data. It is assumed 
            table options, like inclusion (or not) of shielding, cosmic rays,
            dust, and stellar radiation, are the same for both tables.
        lintable: bool
            if True, interpolate tables linearly in linear space 
            (e.g., ion fraction) 
            instead of linearly in log space (e.g., log ion fraction)
            Note that whether log values are returned in controlled
            separately.
        Returns
        -------
        a linetable_PS20 object.
        '''
        self.ion = ion
        self.z = z
        self.ionbalfile = ionbalfile
        self.emtabfile = emtabfile
        self.emission = emission
        self.vol = vol
        self.lintable = lintable
        
        self.parse_ionname()
        self.getmetadata()
    
    def __str__(self):
        _str = '{obj}: {ion} {emabs} at z={z:.3f} using {vol} data from' +\
               ' {iontab} and {emtab}, to be interpolated linearly in '+ \
               '{linlog} space'
        if self.emission:
            emabs = 'emission'
        else:
            emabs = 'ion fraction'
        if self.vol:
            vc = 'Vol'
        else:
            vc = 'Col'
        linlog = 'linear' if self.lintable else 'log'
        _str = _str.format(obj=self.__class__, ion=self.ion, emabs=emabs,
                           z=self.z, vol=vc, 
                           iontab=self.ionbalfile.split('/')[-1],
                           emtab=self.emtabfile.split('/')[-1],
                           linlog=linlog)
        return _str
    
    def __repr__(self):
        _str = '{obj} instance: interpolate Ploeckinger & Schaye (2020) tables\n'
        _str += 'ion:\t {ion}, interpreted as (coming from) the {elt} {stage} ion\n'
        _str += 'z:\t {z}\n'
        _str += 'vol:\t {vol}\n'
        _str += 'emission:\t {emission}\n'
        _str += 'emtabfile:\t {emtabfile}\n'
        _str += 'ionbalfile:\t {ionbalfile}\n'
        _str += 'lintable:\t {lintable}\n'
        _str = _str.format(obj=self.__class__, ion=self.ion,
                           elt=self.elementshort, stage=self.ionstage,
                           z=self.z, vol=self.vol, emission=self.emission,
                           emtabfile=self.emtabfile,
                           ionbalfile=self.ionbalfile,
                           lintable=self.lintable)
        return _str
    
    def find_ionbal(self, dct_T_Z_nH, log=False):
        '''
        retrieve the interpolated ion balance values for the input particle 
        density, temperature, and metallicity
        
        The lowest metallicity bin is ignored since all ion fractions are 
        tabulated as zero there.

        Parameters
        ----------
        dct_T_Z_nH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].
        log: bool
            return log ion balance if True
        Returns
        -------
        float array
            ion balances: ion mass / gas phase parent element mass.

        '''
        if not hasattr(self, 'iontable_T_Z_nH'):
            self.findiontable()
        
        res = self.interpolate_3Dtable(dct_T_Z_nH,
                                       self.iontable_T_Z_nH[:, :, :])
        if (not self.lintable) and (not log):
            res = 10**res
        elif self.lintable and log:
            res = np.log10(res)
        return res
        
    def find_logemission(self, dct_T_Z_nH):
        '''
        retrieve the interpolated emission values for the input particle 
        density, temperature, and metallicity

        Parameters
        ----------
        dct_T_Z_nH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].

        Returns
        -------
        float array
            log line emission per unit volume: log10 erg / s / cm**3 
            (the non-log emission values may cause overflows)
        '''
        if not hasattr(self, 'emtable_T_Z_nH'):
            self.findemtable()
        res = self.interpolate_3Dtable(dct_T_Z_nH, self.emtable_T_Z_nH)
        if self.lintable:
            res = np.log10(res)
        return res
    
    def find_depletion(self, dct_T_Z_nH):
        '''
        retrieve the interpolated dust depletion values for the input particle 
        density, temperature, and metallicity

        Parameters
        ----------
        dct_T_Z_nH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].

        Returns
        -------
        float array
            dust depletion: fraction of element locked in dust
        '''
        if not hasattr(self, 'depletiontable_T_Z_nH'):
            self.finddepletiontable()
        res = self.interpolate_3Dtable(dct_T_Z_nH,
                                       self.depletiontable_T_Z_nH)
        if not self.lintable:
            res = 10**res
        return res
    
    def find_assumedabundance(self, dct_Z, log=False):
        '''
        retrieve the interpolated assumed parent element abundance values for 
        the input particle metallicity

        Parameters
        ----------
        dct_Z: dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
        log: bool
            return log abundances if True
        Returns
        -------
        float array
            element abundance n_i / n_H for the parent element of the line or
            ion assumed at the input metallicity
            
        '''
        
        if not hasattr(self, 'logZsol'):
            self.findiontable()
        logZabs = self.logZsol + np.log10(self.solarZ)
        # edge values outside range: matches use of edge values in T, Z, nH
        # table interpolation
        edgevals = (self.numberfraction_Z[1], self.numberfraction_Z[-1])
        self.abunds_interp = spint.interp1d(logZabs, self.numberfraction_Z[1:],
                                            kind='linear', axis=-1, copy=True,
                                            bounds_error=False,
                                            fill_value=edgevals)
        res = self.abunds_interp(dct_Z['logZ'])
        if not log:
            res = 10**res
        return res
        
    def findiontable(self):
        if self.vol:
            vc = 'Vol'
        else:
            vc = 'Col'
            
        if self.ion in ['hmolssh', 'h1ssh', 'h1', 'h2']:
            tablepath = '/Tdep/HydrogenFractions{vc}'.format(vc=vc)
            if self.ion == 'hmolssh':
                ionind = 2
            elif self.ion in ['h1ssh', 'h1']:
                ionind = 0
            elif self.ion == 'h2':
                ionind = 1
        elif self.ion == 'hneutralssh':
            msg = "hneutralssh fractions from the PS20 tables are not" + \
                  "currently implemented"
            raise NotImplementedError(msg)
        else:
            if not self.vol:
                msg = 'Column quantities are only available for hydrogen'
                raise ValueError(msg)
            ionind = self.ionstage - 1 
            tablepath = 'Tdep/IonFractions/{eltnum:02d}{eltname}'
            tablepath = tablepath.format(eltnum=self.eltind,
                                         eltname=self.element.lower())
            print('Using table {}'.format(tablepath))
            
        with h5py.File(self.ionbalfile, "r") as tablefile:
            self.logTK     = tablefile['TableBins/TemperatureBins'][:] 
            self.lognHcm3  = tablefile['TableBins/DensityBins'][:] 
            self.logZsol   = tablefile['TableBins/MetallicityBins'][1:]
            self.redshifts = tablefile['TableBins/RedshiftBins'][:] 
            
            if self.z < self.redshifts[0] or self.z > self.redshifts[-1]:
                msg = 'Desired redshift {z} is outside the tabulated range '+\
                      + '{zmin} - {zmax}'
                msg = msg.format(z=self.z, zmin=self.redshifts[0], 
                                 zmax=self.reshifts[-1])
                raise ValueError(msg)
            zi_lo = np.min(np.where(self.z <= self.redshifts)[0])
            zi_hi = np.max(np.where(self.z >= self.redshifts)[0])            
            
            tableg = tablefile[tablepath] #  z, T, Z, nH, ion
             # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Ion
            if zi_lo == zi_hi:
                self.iontable_T_Z_nH =\
                    tableg[zi_lo, :, 1:, :, ionind]
                if self.lintable:
                    self.iontable_T_Z_nH  = 10**self.iontable_T_Z_nH 
            else:
                msg = 'Linearly interpolating ion balance table ' +\
                      'values in redshift'
                print(msg)
                z_lo = self.redshifts[zi_lo]
                z_hi = self.redshifts[zi_hi]
                
                if self.lintable:
                    self.iontable_T_Z_nH =\
                        (z_hi - self.z) / (z_hi - z_lo) * \
                            10**tableg[zi_lo, :, 1:, :, ionind] + \
                        (self.z - z_lo) / (z_hi - z_lo) * \
                            10**tableg[zi_hi, :, 1:, :, ionind]
                else:
                    self.iontable_T_Z_nH =\
                        (z_hi - self.z) / (z_hi - z_lo) * \
                            tableg[zi_lo, :, 1:, :, ionind] + \
                        (self.z - z_lo) / (z_hi - z_lo) * \
                            tableg[zi_hi, :, 1:, :, ionind]

    def findemtable(self):
        if self.vol:
            vc = 'Vol'
        else:
            vc = 'Col'
            
        with h5py.File(self.emtabfile, 'r') as f:
            lineid = f['IdentifierLines'][:]
            lineid = np.array([line.decode() for line in lineid])
            match = [self.ion == line for line in lineid]
            li = np.where(match)[0][0]
            
            emg = f['Tdep/Emissivities{vc}'.format(vc=vc)] 
            # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Line
            self.redshifts = f['TableBins/RedshiftBins'][:]
            self.logTK = f['TableBins/TemperatureBins'][:]
            self.lognHcm3 = f['TableBins/DensityBins'][:]
            self.logZsol = f['TableBins/MetallicityBins'][1:] # -50. = primordial
            
            zi_lo = np.min(np.where(self.z <= self.redshifts)[0])
            zi_hi = np.max(np.where(self.z >= self.redshifts)[0])            
            
            if self.z < self.redshifts[0] or self.z > self.redshifts[-1]:
                msg = 'Desired redshift {z} is outside the tabulated range '+\
                      + '{zmin} - {zmax}'
                msg = msg.format(z=self.z, zmin=self.redshifts[0], 
                                 zmax=self.reshifts[-1])
                raise ValueError(msg) 
             # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Line
            if zi_lo == zi_hi:
                self.emtable_T_Z_nH = emg[zi_lo, :, 1:, :, li]
                if self.lintable:
                    self.emtable = 10**self.emtable
            else:
                z_lo = self.redshifts[zi_lo]
                z_hi = self.redshifts[zi_hi]

                if self.lintable:
                    self.emtable_T_Z_nH =\
                    (z_hi - self.z) / (z_hi - z_lo) *\
                        10**emg[zi_lo, :, 1:, :, li] + \
                    (self.z - z_lo) / (z_hi - z_lo) *\
                        10**emg[zi_hi, :, 1:, :, li]
                else:
                    self.emtable_T_Z_nH =\
                        (z_hi - self.z) / (z_hi - z_lo) *\
                            emg[zi_lo, :, 1:, :, li] + \
                        (self.z - z_lo) / (z_hi - z_lo) *\
                            emg[zi_hi, :, 1:, :, li]
        
    def finddepletiontable(self):    
        with h5py.File(self.ionbalfile, 'r') as f:        
            deplg = f['Tdep/Depletion'] 
            # z, T, Z, nH, element
            self.redshifts = f['TableBins/RedshiftBins'][:]
            self.logTK = f['TableBins/TemperatureBins'][:]
            self.lognHcm3 = f['TableBins/DensityBins'][:]
            self.logZsol = f['TableBins/MetallicityBins'][1:] # -50. = primordial
             
            zi_lo = np.min(np.where(self.z <= self.redshifts)[0])
            zi_hi = np.max(np.where(self.z >= self.redshifts)[0])              
            
            if self.z < self.redshifts[0] or self.z > self.redshifts[-1]:
                msg = 'Desired redshift {z} is outside the tabulated range '+\
                      '{zmin} - {zmax}'
                msg = msg.format(z=self.z, zmin=self.redshifts[0], 
                                 zmax=self.reshifts[-1])
                raise ValueError(msg) 
             # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: element
            if zi_lo == zi_hi:
                self.depletiontable_T_Z_nH = \
                    deplg[zi_lo, :, 1:, :, self.eltind]
                if self.lintable:
                    self.depletiontable_T_Z_nH = 10**self.depletiontable_T_Z_nH
            else:
                z_lo = self.redshifts[zi_lo]
                z_hi = self.redshifts[zi_hi]

                if self.lintable:
                    self.depletiontable_T_Z_nH =\
                       (z_hi - self.z) / (z_hi - z_lo) *\
                            10**deplg[zi_lo, :, 1:, :, self.eltind] + \
                       (self.z - z_lo) / (z_hi - z_lo) *\
                            10**deplg[zi_hi, :, 1:, :, self.eltind]
                else:
                    self.depletiontable_T_Z_nH =\
                        (z_hi - self.z) / (z_hi - z_lo) *\
                            deplg[zi_lo, :, 1:, :, self.eltind] + \
                        (self.z - z_lo) / (z_hi - z_lo) *\
                            deplg[zi_hi, :, 1:, :, self.eltind]
        
        
    def interpolate_3Dtable(self, dct_logT_logZ_lognH, table):
        '''
        retrieve the table values for the input particle density, temperature,
        and metallicity

        Parameters
        ----------
        dct_logT_logZ_lognH : dict of 1-D float arrays
            dictionary containing the following arrays describing each 
            resolution element:
                'logT': log10 temperature [K]
                'logZ': log10 metallicity [mass fraction, *not* normalized to 
                                           solar]
                'lognH': log10 hydrogen number density [cm**-3].

        Raises
        ------
        ValueError
            input arrays have different lengths
            or the requested redshift is outside the tabulated range
            or the input table shape doesn't match logTK, logZsol, lognHcm3.

        Returns
        -------
        1D float array
            the fraction interpolated table values

        '''
        if len(table.shape) != 3:
            msg = 'Interpolation is for 3 dimensional tables only, ' + \
                'not shape {}'.format(table.shape)
            raise ValueError(msg)
        
        expected_tableshape = (len(self.logTK), 
                               len(self.logZsol), 
                               len(self.lognHcm3)) 
        if table.shape != expected_tableshape: 
            msg  = 'Table shape {} did not match expected {}'
            msg = msg.format(table.shape, expected_tableshape)
            raise ValueError(msg)
            
        logT  = dct_logT_logZ_lognH['logT']
        logZ  = dct_logT_logZ_lognH['logZ']
        lognH = dct_logT_logZ_lognH['lognH']
        
        NumPart = len(lognH)
        inbalance = np.zeros(NumPart, dtype=np.float32)    
        if len(logT) != NumPart or len(logZ) != NumPart:
            raise ValueError('lognH, logZ, and logT  should have the same length')
    
        # need to compile with some extra options to get this to work: make -f make_emission_only
        print("------------------- C interpolation function output --------------------------\n")
        cfile = ol.c_interpfile
    
        acfile = ct.CDLL(cfile)
        interpfunction = acfile.interpolate_3d 
        # just a linear interpolator; works for non-emission stuff too
        # retrieved ion balance tables are T x Z x nH
    
        type_partarray = np.ctypeslib.ndpointer(dtype=ct.c_float, 
                                                          shape=(NumPart,))
        tablesize = np.prod(table.shape)
        type_table = np.ctypeslib.ndpointer(dtype=ct.c_float,
                                            shape=(tablesize,))
        
        interpfunction.argtypes = [type_partarray,
                                   type_partarray,
                                   type_partarray,
                                   ct.c_longlong, 
                                   type_table,
                                   np.ctypeslib.ndpointer(dtype=ct.c_float,
                                       shape=self.logTK.shape),
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=ct.c_float,
                                       shape=self.logZsol.shape),
                                   ct.c_int,
                                   np.ctypeslib.ndpointer(dtype=ct.c_float,
                                       shape=self.lognHcm3.shape), 
                                   ct.c_int,
                                   type_partarray]
    
        logZabs = self.logZsol + np.log10(self.solarZ)
        res = interpfunction(logT.astype(np.float32),
                             logZ.astype(np.float32),
                             lognH.astype(np.float32),
                             ct.c_longlong(NumPart),
                             np.ndarray.flatten(table.astype(np.float32)),
                             self.logTK.astype(np.float32),
                             ct.c_int(len(self.logTK)), 
                             logZabs.astype(np.float32),
                             ct.c_int(len(logZabs)), 
                             self.lognHcm3.astype(np.float32),
                             ct.c_int(len(self.lognHcm3)),
                             inbalance,
                             )
    
        print("-------------- C interpolation function output finished ----------------------\n")
    
        if res != 0:
            print('Something has gone wrong in the C function: output %s. \n',str(res))
            return None
    
        return inbalance
