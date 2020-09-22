#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:21:50 2020

@author: wijers

Look into the gas and DM in a halo and check its morphology
"""

import numpy as np
import h5py

import eagle_constants_and_units as c
import projection_classes as pc 
import make_maps_opts_locs as ol

def get_sf(simnum, snapnum, var):
    '''
    get subhalo, particle data Simfile instances
    '''
    sf_sub = pc.Simfile(simnum, snapnum, var, file_type='sub')
    sf_particle = pc.Simfile(simnum, snapnum, var, file_type='particles')
    return sf_sub, sf_particle

def get_partpos(sf_particle, sf_sub, groupnum, region):
    
    
    gn_gas = sf_particle.readarray('PartType0/GroupNumber', region=region)
    gn_dm  = sf_particle.readarray('PartType0/GroupNumber', region=region)
    sel_gas = np.isclose(groupnum, gn_gas)
    sel_dm = np.isclose(groupnum, gn_dm)
    
    pos_gas = sf_particle.readarray('PartType0/Coordinates', region=region)[sel_gas]
    pos_dm = sf_particle.readarray('PartType0/Coordinates', region=region)[sel_dm]
    
    pos_gas *= 1. / (c.cm_per_mpc * sf_particle.a)
    pos_dm  *= 1. / (c.cm_per_mpc * sf_particle.a)
    return pos_gas, pos_dm
    

def matchhalo(sf_sub, xrange=None, yrange=None, zrange=None, M200crange=None,\
              margin=5.):
    '''
    units: cMpc, Msun
    returns: halonumber, region (cMpc/h)
    '''
    
    COP_sub = sf_sub.readarray('FOF/GroupCentreOfPotential')  
    COP_sub *= 1. / (c.cm_per_mpc * sf_sub.a)
    
    M200c_sub = sf_sub.readarray('FOF/Group_M_Crit200')
    M200c_sub *= 1. / c.solar_mass
    
    hsel = np.ones(len(COP_sub), dtype=bool)
    if xrange is not None:
        if xrange[0] is not None:
            hsel &= COP_sub[:, 0] >= xrange[0]
        if xrange[1] is not None:
            hsel &= COP_sub[:, 0] < xrange[1]
    if yrange is not None:
        if yrange[0] is not None:
            hsel &= COP_sub[:, 1] >= yrange[0]
        if yrange[1] is not None:
            hsel &= COP_sub[:, 1] < yrange[1]   
    if zrange is not None:
        if zrange[0] is not None:
            hsel &= COP_sub[:, 2] >= zrange[0]
        if zrange[1] is not None:
            hsel &= COP_sub[:, 2] < zrange[1] 
    if M200crange is not None:
        if M200crange[0] is not None:
            hsel &= M200c_sub >= M200crange[0]
        if M200crange[1] is not None:
            hsel &= M200c_sub < M200crange[1] 
    im = np.where(hsel)[0]
    if len(im) > 1:
        raise RuntimeError('Your input selection applies to multiple haloes')
    if len(im) == 0:
        raise RuntimeError('Your input selection applies to zero haloes')
    im = im[0]
    
    region = [COP_sub[im][0] - margin, COP_sub[im][0] + margin,\
              COP_sub[im][1] - margin, COP_sub[im][1] + margin,\
              COP_sub[im][2] - margin, COP_sub[im][2] + margin]
    region = [r * sf_sub.h for r in region]
    return im + 1, region

def savepartdata(sf_sub, im, region, pos_gas, pos_dm, resolution=0.01):
    '''
    save particle position histograms
    Note: does not deal with region edge overlaps! 
    '''
    
    odir = ol.pdir + 'halo_morphmaps/'
    fn = 'FoFhist_{simnum}_{snapnum}_{var}_groupnum{im}.hdf5'
    fn = fn.format(simnum=sf_sub.simnum, snapnum=sf_sub.snapnum, var=sf_sub.var,\
                   im=im)
    
    _region = [r / sf_sub.h for r in region]
    margin = np.average([_region[1] - _region[0],\
                         _region[3] - _region[2],\
                         _region[5] - _region[4],\
                         ])
    gn = 'diameter_{}_cMpc'.format(margin)
    
    with h5py.File(odir + fn, 'a') as fo:
        if 'Header' not in fo:
            hed = fo.create_group('Header')
            _gp = hed.create_group('cosmopars')
            csm = sf_sub.get_cosmopars()
            for key in csm:
                _gp.attrs.create(key, csm[key])
            hed.attrs.create('simnum', np.string_(sf_sub.simnum))
            hed.attrs.create('var', np.string_(sf_sub.var))
            hed.attrs.create('snapnum', sf_sub.snapnum)
            hed.attrs.create('info', np.string_('particle position histogram for gas and DM'))
        
        if gn in fo:
            raise ValueError('It seems this data has already been saved')
        
        p0 = np.array(_region)[::2]
        p1 = np.array(_region)[1::2]
        pmin = np.floor(p0 / resolution) * resolution
        pmax = np.ceil(p1 / resolution) * resolution
        nbins = (np.round((pmax - pmin) / resolution, 0)).astype(int)
        
        bins = [np.linspace(pmin[i], pmax[i], nbins[i] + 1) for i in range(3)]
        axpairs = [(0, 1), (1, 2), (2, 0)]
        axn = ['x', 'y', 'z']
        
        gp = fo.create_group(gn)
        for axes in axpairs:
            hist, xe, ye = np.histogram2d(pos_gas[:, axes[0]], pos_gas[:, axes[1]],\
                                          bins=[bins[axes[0]], bins[axes[1]]])
            dsname = 'gas_{}{}'.format(axn[axes[0]], axn[axes[1]])
            ds = gp.create_dataset(dsname, data=hist)
            ds.attrs.create('axis0', axes[0])
            ds.attrs.create('axis1', axes[1])
            
            hist, xe, ye = np.histogram2d(pos_dm[:, axes[0]], pos_dm[:, axes[1]],\
                                          bins=[bins[axes[0]], bins[axes[1]]])
            dsname = 'dm_{}{}'.format(axn[axes[0]], axn[axes[1]])
            ds = gp.create_dataset(dsname, data=hist)
            ds.attrs.create('axis0', axes[0])
            ds.attrs.create('axis1', axes[1])
        
        bg = gp.create_group('bins')
        for i in range(3):
            bg.create_dataset('edges{i}'.format(i=i), data=bins[i])


def check_halo8():
    '''
    check halo with a weird thing sticking out in soft X-ray emission lines
    '''
    
    sf_sub, sf_particle = get_sf('L0100N1504', 27, 'REFERENCE')
    groupnum, region = matchhalo(sf_sub, xrange=(50., 75.), yrange=(20., 60.),\
                                 zrange=(15., 27.),\
                                 M200crange=(10**14., None), margin=5.)
    print('Halo number is {}'.format(groupnum))
    pos_gas, pos_dm = get_partpos(sf_particle, sf_sub, groupnum, region)
    savepartdata(sf_sub, groupnum, region, pos_gas, pos_dm, resolution=0.01)
        
        