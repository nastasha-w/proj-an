
'''
Retrieve the units for different quantities in FIRE simulation outputs
based on the snapshot and/or parameter files, or otherwise, the FIRE
simulation defaults. 
'''

import numpy as np
import h5py

import eagle_constants_and_units as c 

class Units:
    def __init__(self, *args, **kwargs):
        '''         
        Get the base unit values. Manual a and h values should only be used for
        testing purposes.
    
        Parameters:
        -----------
        snapfile: str (optional)
             name of the (a) snapshot file. Used to find the hubble parameter 
             and expansion factor, and units is possible. This should be an 
             hdf5 file, including  the directory path.
        parameterfile: str (optional)
             name of the parameter file used for the simulation run. Used to
             find the unit values if not found in the snapshot file, otherwise 
             ignored and not required. Note that this is typically needed for
             the magnetic field units.
        a [keyword]: float
             expansion factor to use. Overridden by anything in the files. 
             Required if using the FIRE default units.
        h [keyword]: float
             hubble parameter to use [in 100 km/s/Mpc]. Overridden by anything
             in the files.
        '''
        self.units_processed = False
        self.reqlist = ['a', 'HubbleParam', 'cosmoexp', 
                        'codevelocity_cm_per_s', 'codemageneticfield_gauss',
                        'codemass_g', 'codelength_cm']
        
        if len(args) > 0:
            self._get_kwargs_ha(required=False, **kwargs)
            snapn = args[0]
            self._read_snapshot_data(snapn)
            if self._check_baseunits_present():
                print('Unit data from snapshot')
            elif len(args) > 1:
                parfile = args[1]
                print('Getting (some) units from parameter file.')
                self._read_parameterfile_units(parfile)
            if not self._check_baseunits_present():
                print('Falling back to (some) FIRE default units')
                self._use_fire_defaults()
        else:
            self._get_kwargs_ha(required=True, **kwargs)
            print('Using FIRE default units')
            self._use_fire_defaults()
            
        self._process_code_units()
        self._get_derived_units_and_acorr()
      
    def _use_fire_defaults(self):
        if not hasattr(self, 'HubbleParam'):
            self.HubbleParam = 0.7
        if not hasattr(self, 'codemass_g'):
            self.codemass_g = 1e10 / self.HubbleParam
        if not hasattr(self, 'codelength_cm'):
            self.codelength_cm = c.cm_per_Mpc * 1e-3 / self.HubbleParam
        if not hasattr(self, 'codevelocity_cm_per_s'):
            self.codevelocity_cm_per_s = 1e5
        if not hasattr(self, 'codemageneticfield_gauss'):
            self.codemageneticfield_gauss =  1.
        if not hasattr(self, 'cosmpexp'):
            self.cosmpexp = True
    
    def _get_derived_units_and_acorr(self):
         self.codetime_s = self.codelength_cm / self.codevelocity_cm_per_s
         self.codedensity_g_per_cm3 = self.codemass_g / self.codelength_cm**3
         self.codeinternalenergy_cm2_per_s2 = self.codevelocity_cm_per_s**2
         self.codedivergencedampingfield = self.codemageneticfield_gauss * \
                                           self.codevelocity_cm_per_s
         
         self.codelength_cm *= self.a
         self.codevelocity_cm_per_s *= np.sqrt(self.a)
         self.codedensity_g_per_cm3 *= self.a**-3
        
    def _read_snapshot_data(self, snapn):
        with h5py.File(snapn) as _f:
            if 'ComovingIntegrationOn' in _f['Header'].attrs:
                self.cosmoexp = bool(_f['Header'].attrs['ComovingIntegrationOn'])
            self.HubbleParam = _f['Header'].attrs['HubbleParam']
            self.a = _f['Header'].attrs['Time']
            
            # need to get the magnetic field data from the parameter file
            if 'UnitMass_In_CGS' in _f['Header'].attrs:
                self.codemass_g = _f['Header'].attrs['UnitMass_In_CGS']
            
            if 'UnitVelocity_In_CGS' in _f['Header'].attrs:
                self.codevelocity_cm_per_s = \
                    _f['Header'].attrs['UnitVelocity_In_CGS']
            
            if 'UnitLength_In_CGS' in _f['Header'].attrs:
                self.codelength_cm = _f['Header'].attrs['UnitLength_In_CGS']
            
    def _read_parameterfile_units(self, filen):
        setl = False
        setm = False
        setv = False
        setb = False
        setc = False
        seth = False
        with open(filen, 'r') as _f:
            for line in _f:
                if line.startswith('UnitLength_in_cm'):
                    self.codelength_cm = float(line.split()[1])
                    setl = True
                elif line.startswith('UnitMass_in_g'):
                    self.codemass_g = float(line.split()[1])
                    setm = True
                elif line.startswith('UnitVelocity_in_cm_per_s'):
                    self.codevelocity_cm_per_s = float(line.split()[1])
                    setv = True  
                elif line.startswith('UnitMagneticField_in_gauss'):
                    self.codemageneticfield_gauss = float(line.split()[1])
                    setb = True  
                elif line.startswith('ComovingIntegrationOn'):
                    self.cosmoexp = bool(int(line.split()[1]))
                    setc = True
                elif line.startswith('HubbleParam'):
                    self.HubbleParam = float(line.split()[1])
                    seth = True
                if setl and setm and setv and setb and setc and seth:
                    break
        if not (setl and setm and setv and setb and setc and seth):
            print('Could not find all units in the parameterfile')
            
    def _check_baseunits_present(self):
        present = {attr: hasattr(self, attr) for attr in self.reqlist}
        alldone = np.all([present[attr] for attr in present])
        return alldone
        
    def _process_code_units(self):
        '''
        and processing depending on the header or parameterfile data.
        separated from direct read-in so that neither depends on the other
        being checked first
        '''
        if self.units_processed:
            raise RuntimeError('units already processed')
        # check presence
        alldone = self._check_baseunits_present()
        if not alldone:
            missing = {attr if not hasattr(self, attr) else None \
                       for attr in self.reqlist}
            missing -= {None}
            msg = 'Could not find required values: {}'.format(missing)
            raise RuntimeError(msg)
        self.codemass_g /= self.HubbleParam
        self.codelength_cm /= self.HubbleParam
        if not self.cosmoexp:
            self.a = 1.
        self.units_processed = True
        
    def _get_kwargs_ha(self, required=False, **kwargs):
        if 'h' in kwargs: 
            print('Using kwarg h value')
            self.HubbleParam = kwargs['h']
        #elif required:
        #    msg = 'If no snapshot is given, h must be specified as a keyword'
        #    raise ValueError(msg)
        if 'a' in kwargs:
            print('Using kwarg a value')
            self.a = kwargs['a'] 
        elif required:
            msg = 'If no snapshot is given, "a" must be specified as a keyword'
            raise ValueError(msg) 
    
    def getunits(self, field):
        '''
        get the units for a FIRE simulation output field: 
        multiply the field by that factor to get the field quantity in CGS
        '''
        if field.startswith('PartType'):
            field = '/'.join(field.split('/')[1:])
        
        if 'Metallicity' in field or 'SmoothedMetallicity' in field or\
           'ElementAbundance' in field:
            # absolute mass fractions in FIRE and EAGLE. Smoothed option is for
            # compatibility, but meaningless in non-SPH runs.
            return 1.
        
        elif field == 'ArtificialViscosity':
            return 1.
        elif field in ['BH_Mass', 'BH_Mass_AlphaDisk']:
            return self.codemass_g
        elif field == 'BH_Mdot':
            return self.codemass_g / self.codetime_s
        elif field == 'Coordinates':
            return self.codelength_cm
        elif field == 'Density':
            return self.codedensity_g_per_cm3
        elif field == 'DivergenceOfMagneticField':
            return 1. / self.codelength_cm
        elif field == 'ElectronAbundance': 
            # res. elt. mass-weighted free electrons / protons
            return 1.
        elif field == 'InternalEnergy':
            return self.codeinternalenergy_cm2_per_s2 
        elif field == 'MagneticField':
            return self.codemageneticfield_gauss
        elif field == 'Masses':
            return self.codemass_g           
        elif field == 'NeutralHydrogenAbundance': 
            # neutral hydrogen fraction (0 -- 1)
            return 1.
        elif field == 'StarFormationRate':
            return c.solar_mass / c.seconds_per_year
        elif field in ['ParticleIDs', 'ParticleChildIDsNumber',
                       'ParticleIDGenerationNumber']: 
            print('Warning: Attempting a unit conversion for Particle IDs?')
            return 1 
        elif field == 'PhotonEnergy':
            return self.codemass * self.codevelocity_cm_per_s**2
        elif field == 'SmoothingLength':
            return self.codelength_cm
        elif field == 'SoundSpeed':
            return self.codevelocity_cm_per_s
        elif field == 'StellarFormationTime':
            # cosmo: a; non-cosmological runs: time (in h**−1 Gyr)
            if self.cosmoexp:
                return 1. 
            else:
                return 1e3 * c.seconds_per_Myr / self.HubbleParam
        elif field == 'Velocities':
            return self.codevelocity_cm_per_s
        elif field in ['CosmicRayEnergy', 'DivBcleaningFunctionGradPhi',
                       'DivBcleaningFunctionPhi']:
            msg = 'Look up units for {} and add to units_fire.py'
            raise NotImplementedError(msg.format(field))
        else:
            msg = '{} is not a (known) FIRE simulation output field'
            raise ValueError(msg.format(field))