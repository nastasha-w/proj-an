
# quest modules:
# module load python/anaconda3.6

import numpy as np
import h5py

import matplotlib.pyplot as plt

import readin_fire_data as rfd

# snapshot 50: redshift 0.5; hopefully enough to make a-scaling errors clear
firedata_test = '/projects/b1026/snapshots/AGN_suite/fiducial_jet/m12i_res57000/output/snapshot_050.hdf5'
firedata_params = '/projects/b1026/snapshots/AGN_suite/fiducial_jet/m12i_res57000/params.txt-usedvalues'
ddir = '/projects/b1026/nastasha/tests/start_fire/'


def make_simple_phasediagram():
    '''
    first test: just read in the data for a histogram
    '''
    
    snap = rfd.Firesnap(firedata_test, firedata_params)

    temperature = snap.readarray_emulateEAGLE('PartType0/Temperature')
    tempconv = snap.toCGS    
    temperature = np.log10(temperature * tempconv)
    density = snap.readarray_emulateEAGLE('PartType0/Density')
    densconv = snap.toCGS
    density = np.log10(density * densconv)
    
    binsize = 0.05
    tempmin = np.min(temperature)
    tempmax = np.max(temperature)
    densmin = np.min(density)
    densmax = np.max(density)
    
    mint = np.floor(tempmin / binsize) * binsize
    maxt = np.ceil(tempmax / binsize) * binsize
    axbins_t = np.arange(mint, maxt + binsize/2., binsize)
    mind = np.floor(densmin / binsize) * binsize
    maxd = np.ceil(densmax / binsize) * binsize
    axbins_d = np.arange(mind, maxd + binsize/2., binsize)
    
    mass = snap.readarray_emulateEAGLE('PartType0/Mass')
    massconv = snap.toCGS
    # actually converting causes an overflow
    
    hist, xbins, ybins = np.histogram2d(density, temperature, 
                                        bins=[axbins_d, axbins_t], 
                                        weights=mass)
    
    fn = 'phasediagram_rhoT_firesnap.hdf5'
    with h5py.File(ddir + fn, 'w') as f:
        f.create_dataset('hist', data=hist)
        munit = '{} g per histogram bin'.format(massconv)
        f['hist'].attrs.create('units', np.string_(munit))
        f['hist'].attrs.create('toCGS', massconv) 
        f['hist'].attrs.create('dimension0', np.string_('logdensity'))                                   
        f['hist'].attrs.create('dimension1', np.string_('logtemperature'))
        f.create_dataset('logdensity', data=density)
        f['logdensity'].attrs.create('units', np.string_('g * cm**-3'))
        f.create_dataset('logtemperature', data=temperature)
        f['logtemperature'].attrs.create('units', np.string_('K'))
        
    hfrac = snap.readarray_emulateEAGLE('PartType0/ElementAbundance/Hydrogen')
    hconv = snap.toCGS # 1.0
    hfrac *= hconv # most values ~ 0.7 --0.75
    
    hdens = density + np.log10(hfrac / (rfd.uf.c.atomw_H * rfd.uf.c.u)
    hdensmin = np.min(hdens)
    hdensmax = np.max(hdens)
    minnh = np.floor(hdensmin / binsize) * binsize
    maxnh = np.ceil(hdensmax / binsize) * binsize
    axbins_nh = np.arange(minnh, maxnh + binsize/2., binsize)
    
    hist, xbins, ybins = np.histogram2d(hdens, temperature, 
                                        bins=[axbins_nh, axbins_t], 
                                        weights=mass)
    
    fn = 'phasediagram_nHT_firesnap.hdf5'                                 
    with h5py.File(ddir + fn, 'w') as f:
        f.create_dataset('hist', data=hist)
        munit = '{} g per histogram bin'.format(massconv)
        f['hist'].attrs.create('units', np.string_(munit))
        f['hist'].attrs.create('toCGS', massconv) 
        f['hist'].attrs.create('dimension0', np.string_('loghydrogendensity'))                                   
        f['hist'].attrs.create('dimension1', np.string_('logtemperature'))
        f.create_dataset('loghydrogendensity', data=hdens)
        f['loghydrogendensity'].attrs.create('units', np.string_('cm**-3'))
        f.create_dataset('logtemperature', data=temperature)
        f['logtemperature'].attrs.create('units', np.string_('K'))

def plot_simple_phasediagrams(filen):
    with h5py.File(ddir + filen, 'r') as f:
        # done for today
    
    
    