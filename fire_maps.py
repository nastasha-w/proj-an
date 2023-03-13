#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd
import string
import sys
import os
import uuid # generate unique file names
import glob 
import matplotlib.pyplot as plt

# Andrew Wetzel's Rockstar halo catalogue wrangler
try:
    import halo_analysis as ha
except ModuleNotFoundError:
    msg = 'Could not import module "halo_analysis";' +\
          ' Rockstar halo data read-in will fail.'
    print(msg)

import readin_fire_data as rf
import units_fire as uf
import cosmo_utils as cu
import eagle_constants_and_units as c
# paths to c functions, ion tables
import make_maps_opts_locs as ol
from make_maps_v3_master import project
from ion_utils import linetable_PS20

# setup from the internets
class NoStoredMatchError(Exception):
    def __init__(self, *args):
        if len(args) > 0:
            self.message = args[0]
        else:
            self.message = None
    def __str__(self):
        if self.message is not None:
            return 'NoStoredMatchErrorr: {0} '.format(self.message)
        else:
            return 'NoStoredMatchErrorr'


def linterpsolve(xvals, yvals, xpoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below xpoint 
    xvals, yvals: 1D arrays
    xpoint: float
    '''
    if np.all(np.diff(xvals) >= 0.):
        incr = True
    elif np.all(np.diff(xvals) <= 0.):
        incr = False
    else:
        print('linterpsolve only works for monotonic functions')
        return None
    ind1 = np.where(xvals <= xpoint)[0]
    ind2 = np.where(xvals >= xpoint)[0]
    #print(ind1)
    #print(ind2)
    if len(ind2) == 0 or len(ind1) == 0:
        print('xpoint is outside the bounds of xvals')
        return None
    if incr:
        ind1 = np.max(ind1)
        ind2 = np.min(ind2)
    else:
        ind1 = np.min(ind1)
        ind2 = np.max(ind2)
    #print('Indices x: %i, %i'%(ind1, ind2))
    #print('x values: lower %s, upper %s, searched %s'%(xvals[ind1], xvals[ind2], xpoint))
    if ind1 == ind2:
        ypoint = yvals[ind1]
    else:
        w = (xpoint - xvals[ind1]) / (xvals[ind2] - xvals[ind1]) #weight
        ypoint = yvals[ind2] * w + yvals[ind1] * (1. - w)
    #print('y values: lower %s, upper %s, solution: %s'%(yvals[ind1], yvals[ind2], ypoint))
    return ypoint

def find_intercepts(yvals, xvals, ypoint, xydct=None):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below ypoint 
    xvals, yvals: 1D arrays
    ypoint: float
    Does not distinguish between intersections separated by less than 2 xvals points
    '''
    if xvals is None:
        xvals = xydct['x']
    if yvals is None:
        yvals = xydct['y']

    if not (np.all(np.diff(xvals) <= 0.) or np.all(np.diff(xvals) >= 0.)):
        print('linterpsolve only works for monotonic x values')
        return None
    zerodiffs = yvals - ypoint
    leqzero = np.where(zerodiffs <= 0.)[0]
    if len(leqzero) == 0:
        return np.array([])
    elif len(leqzero) == 1:
        edges = [[leqzero[0], leqzero[0]]]
    else:
        segmentedges = np.where(np.diff(leqzero) > 1)[0] + 1
        if len(segmentedges) == 0: # one dip below zero -> edges are intercepts
            edges = [[leqzero[0], leqzero[-1]]]
        else:
            parts = [leqzero[: segmentedges[0]] if si == 0 else \
                     leqzero[segmentedges[si - 1] : segmentedges[si]] 
                     if si < len(segmentedges) else\
                     leqzero[segmentedges[si - 1] :] \
                     for si in range(len(segmentedges) + 1)]
            edges = [[part[0], part[-1]] for part in parts]
    intercepts = [[linterpsolve(zerodiffs[ed[0]-1: ed[0] + 1], xvals[ed[0]-1: ed[0] + 1], 0.),\
                   linterpsolve(zerodiffs[ed[1]: ed[1] + 2],   xvals[ed[1]: ed[1] + 2], 0.)]  \
                  if ed[0] != 0 and ed[1] != len(yvals) - 1 else \
                  [None,\
                   linterpsolve(zerodiffs[ed[1]: ed[1] + 2],   xvals[ed[1]: ed[1] + 2], 0.)] \
                  if ed[1] != len(yvals) - 1 else \
                  [linterpsolve(zerodiffs[ed[0]-1: ed[0] + 1], xvals[ed[0]-1: ed[0] + 1], 0.),\
                   None]  \
                  if ed[0] != 0 else \
                  [None, None]
                 for ed in edges]
    intercepts = [i for i2 in intercepts for i in i2]
    if intercepts[0] is None:
        intercepts = intercepts[1:]
    if intercepts[-1] is None:
        intercepts = intercepts[:-1]
    return np.array(intercepts)

atomw_u_dct = \
    {'Hydrogen':  c.atomw_H,
     'Helium':    c.atomw_He,
     'Carbon':    c.atomw_C,
     'Nitrogen':  c.atomw_N,
     'Oxygen':    c.atomw_O,
     'Neon':      c.atomw_Ne,
     'Magnesium': c.atomw_Mg,
     'Silicon':   c.atomw_Si,
     'Sulfur':    c.atomw_S,
     'Sulphur':   c.atomw_S,
     'Calcium':   c.atomw_Ca,
     'Iron':      c.atomw_Fe}

def elt_atomw_cgs(element):
    element = string.capwords(element)
    return atomw_u_dct[element] * c.u

class CoordinateWranger:
    def __init__(self, snapobj, center_cm, rotmatrix=None,
                 parttype=0, periodic=True, vcen_cmps=None):
        '''
        class to get position and velocity info in different coordinate
        bases

        Parameters:
        -----------
        snapobj: Firesnap or similar
            object that allows access to cosmological parameters and 
            has a method to read in simulation arrays
        center_cm: float array, shape (3,)
            center coordinates in cm (physical, not comoving)
        rotmatrix: float array, shape (3, 3) or None
            matrix by which to multiply coordinates to get the 
            coordinates in the desired basis
            None means no rotation
        parttype: int
            which particles to get coordinates for. Matches the 
            PartType<number> groups in the simulation outputs.
        periodic: bool
            Do we need to care about coordinates wrapping around the 
            simulation volume?
        vcen_cmps: float array, shape (3,)
            bulk velocity (subtracted from simulation velocities before
            any rotations, etc.), in cm/s

        Note:
        -----
        These objects store the arrays they used, so it's best to 
        delete them once you've got the data you want.
        '''
        self.snapobj = snapobj
        self.cen_cm = center_cm
        self.vcen_cmps = vcen_cmps
        self.rotmatrix = rotmatrix
        self.pt = parttype
        self.periodic = periodic
        self.coordaxis = 1 
        self.pcalcstarted = False
        self.__check_rotmatrix()
    
    def __check_rotmatrix(self):
        if self.rotmatrix is not None:
            if self.rotmatrix.shape != (3, 3):
                msg = ('Rotation matrix should have shape (3, 3) not input'
                       f'{self.rotmatrix.shape} for matrix\n{self.rotmatrix}')
                raise ValueError(msg)
            if not (np.allclose(self.rotmatrix.T, self.rotmatrix) 
                    and np.isclose(np.linalg.det(self.rotmatrix), 1.)):
                msg = ('input was not a valid rotation matrix.\n'
                       'transpose (should be same):\n'
                       f'{self.rotmatrix.T}, {self.rotmatrix}\n'
                       'determinant (should be 1.): '
                       f'{np.linalg.det(self.rotmatrix)}')
                raise ValueError(msg)
        
    def __startcalc_pos(self, subindex=None):
        h5path = f'PartType{self.pt}/Coordinates'
        if self.rotmatrix is None:
            self._subindex = subindex
        else:
            self._subindex = None
        self.coords_simxyz = self.snapobj.readarray(h5path, 
                                                    subindex=self._subindex)
        self.toCGS_coords_simxyz = self.snapobj.toCGS
        self.__center_pos()
        if self.rotmatrix is not None:
            self.__rotate_pos()
        else:
            self.coords_rotxyz = self.coords_simxyz
            self.toCGS_coords_rotxyz = self.toCGS_coords_simxyz
        if subindex is not None:
            self.coords_rotxyz = np.copy(self.coords_rotxyz[subindex])
        del self.coords_simxyz
        del self.toCGS_coords_simxyz
        self.pcalcstarted = True

    def __startcalc_vel(self, subindex=None):
        h5path = f'PartType{self.pt}/Velocities'
        if self.rotmatrix is None:
            self._subindex = subindex
        else:
            self._subindex = None
        self.vel_simxyz = self.snapobj.readarray(h5path, 
                                                 subindex=self._subindex)
        self.toCGS_vel_simxyz = self.snapobj.toCGS
        self.__center_vel()
        if self.rotmatrix is not None:
            self.__rotate_vel()
        else:
            self.vel_rotxyz = self.vel_simxyz
            self.toCGS_vel_rotxyz = self.toCGS_vel_simxyz
        del self.vel_simxyz
        del self.toCGS_vel_simxyz
        self.vcalcstarted = True

    def __rotate_pos(self):
        self.rotmatrix = np.asarray(self.rotmatrix, 
                                    dtype=self.coords_simxyz.dtype)
        self.coords_rotxyz = np.tensordot(self.rotmatrix, self.coords_simxyz,
                                          axes=([1], [self.coordaxis]))
        self.toCGS_coords_rotxyz = self.toCGS_coords_simxyz
    
    def __rotate_vel(self):
        self.rotmatrix = np.asarray(self.rotmatrix, 
                                    dtype=self.vel_simxyz.dtype)
        self.vel_rotxyz = np.tensordot(self.rotmatrix, self.vel_simxyz,
                                       axes=([1], [self.coordaxis]))
        self.toCGS_vel_rotxyz = self.toCGS_vel_simxyz
    
    def __center_pos(self):
        self.center_simu = np.astype(self.cen_cm / self.toCGS_coords_simxyz,
                                     dtype=self.coords_simxyz.dtype)
        self.coords_simxyz -= self.center_simu
        if self.periodic:
            self.boxsize_simu = self.snapobj.cosmopars.boxsize \
                                * self.snapobj.cosmopars.a \
                                / self.snapobj.cosmopars.h \
                                * c.cm_per_mpc / self.toCGS_coords_simxyz
            self.coords_simxyx += 0.5 * self.boxsize_simu
            self.coords_simxyz %= self.boxsize_simu
            self.coords_simxyx -= 0.5 * self.boxsize_simu
    
    def __center_vel(self):
        self.vcen_simu = np.astype(self.vcen_cmps / self.toCGS_vel_simxyz,
                                   dtype=self.vel_simxyz.dtype)
        self.vel_simxyz -= self.vcen_simu
        if self.periodic:
            self.cosmopars = self.snapobj.cosmopars.getdct()
            self.vboxsize_simu = self.snapobj.cosmopars.boxsize \
                                 * self.snapobj.cosmopars.a \
                                 / self.snapobj.cosmopars.h \
                                 * c.cm_per_mpc / self.toCGS_coords_simxyz \
                                 * cu.Hubble(self.cosmopars['z'], 
                                             cosmopars=self.cosmopars)
            self.vel_simxyx += 0.5 * self.vboxsize_simu
            self.vel_simxyz %= self.vboxsize_simu
            self.vel_simxyx -= 0.5 * self.vboxsize_simu
        
    def calccoords(self, coordspecs):
        '''
        calculate various coordinate values. Doing this all in one go
        should save some time from reading in large arrays multiple
        times.

        Parameters:
        -----------
        coordspecs: dict or list-like of dicts
            list: different coordinates to calculate
            dict or dicts in list: specify what to calculate
            dict keys and possible values:
                'pos': [0, 1, 2, 'allcart', 'rcen']
                    0, 1, 2: position along the axis with this index
                    'allcart': for all three of these cartesian axes
                    'rcen': distance to the center
                'vel': [0, 1, 2, 'allcart', 'vrad']
                     0, 1, 2: velocity along the axis with this index
                    'allcart': for all three of these cartesian axes
                    'vrad': radial velocity (relative to coordinate
                            center)
                    'vtot': total velocity (rms coordinate velocties)
                    note: you must specifiy vcen_cmps when initializing
                    this object to calculate this. 
                indices etc. are all for the rotated coordinates, after
                centering 
        
        Returns:
        --------
        The desired coordinates in the listed order. Always returns a 
        list of 3-tuples: (coordinate [array], CGS conversion [float], 
                           doc_dictionary [e.g., used center]) 
        note that if for some reason a coordspec is requested twice, 
        the tuples will include the same object twice
        '''
        self.coordspecs_in = [(key, coordspecs[key]) for key in coordspecs]
        ## this priority setting can get messy very fast if I try to
        ## implement too much here.
        # which (groups of) properties to calculate, and in what order
        # a group is calculated in a single function named 
        # __calc_<group key>
        self.calcorder = {'poscart': [('pos', 'allcart'), ('pos', 0),
                                      ('pos', 1), ('pos', 2)],
                          'poscen': [('pos', 'rcen')],
                          'velcart': [('vel', 'allcart'), ('vel', 0),
                                      ('vel', 1), ('vel', 2)],
                          'veltot': [('vel', 'vtot')], 
                          'velcen': [('vel', 'vrad')],
                          }
        # what to get just because it's needed later
        # note: should include dependencies of dependencies
        self.dependencies = {('pos', 'rcen'): [('pos', 'allcart')],
                             ('vel', 'vtot'): [('vel', 'allcart')],
                             ('vel', 'vrad'): [('vel', 'allcart'),
                                               ('pos', 'allcart'),
                                               ('pos', 'rcen')]
                            }
        # set up to-do list of everything that's needed (no duplicates)
        self._coords_todo = set(self.coordspecs_in.copy())
        for _coordspec in self.coordspecs_in:
            if _coordspec in self.dependencies:
                self._coords_todo |= set(self.dependencies[_coordspec])
        self.coords_todo = [[group[key] for key in group 
                             if group[key] in self._coords_todo] 
                            for group in self.calcorder]
        # holds arrays calculated for output 
        self.coords_outlist = [None] * len(self.coordspecs_in)
        # holds all calculated arrays, including those only needed as
        # dependencies. (keys are coordspecs tuples)
        self.coords_stored = {}
        
        for self.gicur, self.gcur in enumerate(self.coords_todo):
            self.gkeymatch = [key for key in self.calcorder 
                              if set(self.gcur).issubset(
                                  set(self.calcorder[key]))]
            self.gkeymatch = self.gkeymatch[0]
            print(f'calculating {self.gcur}')
            if self.gkeymatch == 'poscart':
                self.__calc_poscart(self.gcur)
            elif self.gkeymatch == 'poscen':
                self.__calc_poscen(self.gcur)
            elif self.gkeymatch == 'velcart':
                self.__calc_velcart(self.gcur)
            elif self.gkeymatch == 'veltot':
                self.__calc_veltot(self.gcur)
            elif self.gkeymatch == 'velcen':
                self.__calc_velrad(self.gcur)
            self.__update_out_todo()
            print(f'still todo: {self.still_todo}')
        del self.gcur, self.fcur, self.gkeymatch, self.coords_todo, 
        del self.coords_stored
        return self.coords_outlist
    
    def __calc_poscart(self, specs):
        if ('pos', 'allcart') in specs or len(specs) > 1:
            self.__startcalc_pos(subindex=None)
            for self.scur in specs:
                if self.spec == ('pos', 'allcart'):
                    self._todoc_cur = {'cen_cm': self.cen_cm,
                                       'rotmatrix': self.rotmatrix,
                                       'rotcoord_index': [0, 1, 2],
                                       'units': 'cm'}
                    self.coords_stored[self.scur] = (self.coords_rotxyz, 
                                                     self.toCGS_coords_rotxyz,
                                                     self._todoc_cur)
                else:
                    self._todoc_cur = {'cen_cm': self.cen_cm,
                                       'rotmatrix': self.rotmatrix,
                                       'rotcoord_index': self.scur[1],
                                       'units': 'cm'}
                    # storing a view of an array could cause unexpected
                    # side-effects
                    self._out = np.copy(self.coords_rotxyz[self.scur[1]])
                    self.coords_stored[self.scur] = (self._out, 
                                                     self.toCGS_coords_rotxyz,
                                                     self._todoc_cur)
                    del self._out
            del self.scur, self._todoc_cur
        else:
            self.__startcalc_pos(subindex=specs[0][1])
            self._todoc_cur = {'cen_cm': self.cen_cm,
                               'rotmatrix': self.rotmatrix,
                               'rotcoord_index': specs[0][1],
                               'units': 'cm'}
            self.coords_stored[specs[0]] = (self.coords_rotxyz, 
                                            self.toCGS_coords_rotxyz,
                                            self._todoc_cur)
            del self._todoc_cur

    def __calc_poscen(self, specs):
        self.scur = specs[0]
        self._in = self.coords_stored[('pos', 'allcart')]
        self._todoc_cur = self._in[2].copy()
        del self._todoc_cur['rotmatrix']
        del self._todoc_cur['rotcoord_index']
        self._out = np.sqrt(np.sum(self._in[0]**2, axis=self.coordaxis))
        self.coords_stored[self.scur] = (self._out, self._in[1], 
                                         self._todoc_cur)
        del self.scur, self._out, self._todoc_cur, self._in

    def __calc_velcart(self, specs):
        if ('vel', 'allcart') in specs or len(specs) > 1:
            self.__startcalc_vel(subindex=None)
            for self.scur in specs:
                if self.spec == ('vel', 'allcart'):
                    self._todoc_cur = {'vcen_cmps': self.vcen,
                                       'rotmatrix': self.rotmatrix,
                                       'rotcoord_index': [0, 1, 2],
                                       'units': 'cm * s**-1'}
                    self.coords_stored[self.scur] = (self.vel_rotxyz, 
                                                     self.toCGS_vel_rotxyz,
                                                     self._todoc_cur)
                else:
                    self._todoc_cur = {'vcen_cmps': self.vcen_cmps,
                                       'rotmatrix': self.rotmatrix,
                                       'rotcoord_index': self.scur[1],
                                       'units': 'cm * s**-1'}
                    # storing a view of an array could cause unexpected
                    # side-effects
                    self._out = np.copy(self.vel_rotxyz[self.scur[1]])
                    self.coords_stored[self.scur] = (self._out, 
                                                     self.toCGS_vel_rotxyz,
                                                     self._todoc_cur)
                    del self._out
            del self.scur, self._todoc_cur
        else:
            self.__startcalc_vel(subindex=specs[0][1])
            self._todoc_cur = {'vcen_cmps': self.vcen_cmps,
                               'rotmatrix': self.rotmatrix,
                               'rotcoord_index': specs[0][1],
                               'units': 'cm * s**-1'}
            self.coords_stored[specs[0]] = (self.vel_rotxyz, 
                                            self.toCGS_vel_rotxyz,
                                            self._todoc_cur)
            del self._todoc_cur

    def __calc_veltot(self, specs):
        self.scur = specs[0]
        self._in = self.coords_stored[('vel', 'allcart')]
        self._todoc_cur = self._in[2].copy()
        del self._todoc_cur['rotmatrix']
        del self._todoc_cur['rotcoord_index']
        self._out = np.sqrt(np.sum(self._in[0]**2, axis=self.coordaxis))
        self.coords_stored[self.scur] = (self._out, self._in[1], 
                                         self._todoc_cur)
        del self.scur, self._out, self._todoc_cur, self._in

    def __calc_velrad(self, specs):
        self.scur = specs[0]
        self._cendir = self.coords_stored[('pos', 'allcart')][0]
        self._cendir /= self.coords_stored[('pos', 'rcen')][0]
        self._out = np.tensordot(self.cendir, 
                                 self.coords_stored[('vel', 'allcart')][0],
                                 axes=(self.coordaxis, self.coordaxis))
        self._units = self.coords_stored[('vel', 'allcart')][1]
        self._todoc_cur = self.coords_stored[('vel', 'allcart')][2].copy()
        del self._todoc_cur['rotmatrix']
        del self._todoc_cur['rotcoord_index']
        self.pkey = ('pos', 'allcart')
        self._todoc_cur['cen_cm'] = self.coords_stored[self.pkey][2]['cen_cm']
        self.coords_stored[self.scur] = (self._out, self._units, 
                                         self._todoc_cur)
        del self.scur, self._out, self._todoc_cur, self._cendir, self._units
        del self.pkey

    def __update_out_todo(self, specs):
        # update output list
        for self.scur in specs:
            for self.i, self.si in enumerate(self.coordspecs_in):
                if self.insub == self.si:
                    self.coords_outlist[self.i] = self.coords_stored[self.si]
        del self.scur, self.i, self.si
        # clean up stored list
        self.still_todo = self.coords_todo[self.gicur + 1:]
        if len(self.still_todo) == 0:
            pass
        else:
            self.curstored = list(self.coords_stored)
            for self.kcur in self.curstored:
                if not (np.any([self.kcur in self.dependencies[_s] 
                                for _g in self.still_todo for _s in _g])):
                    del self.coords_stored[self.kcur]
            del self.kcur        

# seems to work for at least one halo 
# (m13 guinea pig at snapshot 27, comparing image to found center)
def calchalocen(coordsmassesdict, shrinkfrac=0.025, minparticles=1000, 
                initialradiusfactor=1.):
    '''
    from: https://github.com/isulta/massive-halos/blob/d2dc0dd3649f359c0cea7191bfefd11b3498eeda/scripts/halo_analysis_scripts.py#L164 
    Imran Sultan's method, citing Power et al. (2003):
    their parameter values: 
    shrinkpercent=2.5, minparticles=1000, initialradiusfactor=1

    '''
    coords = coordsmassesdict['coords']
    masses = coordsmassesdict['masses']
    totmass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / totmass
    r2 = np.sum((coords - com[np.newaxis, :])**2, axis=1)
    searchrad2 = initialradiusfactor**2 * np.max(r2)
    Npart_conv = min(minparticles, len(masses) * 0.01)
 
    it = 0
    coords_it = coords.copy()
    masses_it = masses.copy()
    comlist = [com]
    radiuslist = [np.sqrt(searchrad2)]
    while len(masses_it) > Npart_conv:
        searchrad2 *= (1. - shrinkfrac)**2
        mask = r2 <= searchrad2
        coords_it = coords_it[mask]
        masses_it = masses_it[mask]
        com = np.sum(coords_it * masses_it[:, np.newaxis], axis=0) \
               / np.sum(masses_it)
        r2 = np.sum((coords_it - com[np.newaxis, :])**2, axis=1)

        it += 1
        comlist.append(com)
        radiuslist.append(np.sqrt(searchrad2))
    return com, comlist, radiuslist

# centering seems to work for at least one halo 
# (m13 guinea pig at snapshot 27, comparing image to found center)
def calchalodata_shrinkingsphere(path, snapshot, meandef=('200c', 'BN98')):
    '''
    Using Imran Sultan's shrinking spheres method, calculate the halo 
    center, then find the halo mass and radius for a given overdensity
    citerion

    Parameters:
    -----------
    path: str
        path containing the 'output' directory or the snapshot
        files/directories for the chosen simulation
    snapshot: int
        snapshot number
    meandef: str
        overdensity definition for the halo
        'BN98': Bryan & Norman 1998 fitting formula
        '<float>c': <float> times the critical density at the snapshot 
                    redshift
        '<float>m': <float> times the mean matter density at the 
                    snapshot redshift
        tuple of values -> return a list of Mvir and Rvir, in same order
    
    Returns:
    --------
    outdct: dict
        contains 
        'Xc_cm', 'Yc_cm', 'Zc_cm': floats 
            the coordinates of the halo center 
        'Rvir_cm': float or list of floats
            the virial radius (radii) according to the halo overdensity
            criterion. float or list matches string or iterable choice
            for the overdensity definition
        'Mvir_cm': float or list of floats
            the virial mass (masses) according to the halo overdensity
            criterion. float or list matches string or iterable choice
            for the overdensity definition
     todoc: dict
        contains information on parameter values and particle types used   
    
    '''
    minparticles = 1000
    minpart_halo = 1000
    snap = rf.get_Firesnap(path, snapshot)
    todoc = {}

    # get mass and coordinate data
    # use all zoom region particle types
    parttypes = [0, 1, 4, 5]
    dct_m = {}
    dct_c = {}
    toCGS_m = None
    toCGS_c = None
    for pt in parttypes:
        cpath = 'PartType{}/Coordinates'
        mpath = 'PartType{}/Mass'
        try:
            dct_c[pt] = snap.readarray_emulateEAGLE(cpath.format(pt))
            _toCGS_c = snap.toCGS
            dct_m[pt] = snap.readarray_emulateEAGLE(mpath.format(pt))
            _toCGS_m = snap.toCGS
        except (OSError, rf.FieldNotFoundError):
            msg = 'Skipping PartType {} in center calc: not present on file'
            print(msg.format(pt))
            continue
        if toCGS_m is None:
            toCGS_m = _toCGS_m
        elif not np.isclose(toCGS_m, _toCGS_m):
                msg = 'Different particle type masses have different' + \
                      ' CGS conversions in ' + snap.firstfilen
                raise RuntimeError(msg)
        if toCGS_c is None:
            toCGS_c = _toCGS_c
        elif not np.isclose(toCGS_c, _toCGS_c):
                msg = 'Different particle type coordinates have different' + \
                      ' CGS conversions in ' + snap.firstfilen
                raise RuntimeError(msg)
    pt_used = list(dct_m.keys())
    pt_used.sort()
    totlen = sum([len(dct_m[pt]) for pt in pt_used])
    masses = np.empty((totlen,), dtype=dct_m[pt_used[0]].dtype)
    coords = np.empty((totlen, dct_c[pt_used[0]].shape[1]), 
                      dtype=dct_c[pt_used[0]].dtype)
    todoc['parttypes_used'] = tuple(pt_used)
    start = 0
    for pt in pt_used:
        partlen = len(dct_m[pt])
        masses[start: start + partlen] = dct_m[pt]
        coords[start: start + partlen] = dct_c[pt]
        start += partlen

        del dct_m[pt]
        del dct_c[pt]
    coordsmassdict = {'masses': masses, 'coords': coords}
    com_simunits, comlist, radiuslist = \
        calchalocen(coordsmassdict, shrinkfrac=0.025, 
                    minparticles=minparticles, initialradiusfactor=1.)
    print('Found center of mass [sim units]: {}'.format(com_simunits))
    todoc.update({'shrinkfrac': 0.025, 
                  'minparticles': minparticles, 
                  'initialradiusfactor': 1.,
                  'minpart_halo': minpart_halo})
    # find Rvir/Mvir
    cosmopars = snap.cosmopars.getdct()
    todoc['cosmopars'] = cosmopars
    if isinstance(meandef, type('')):
        outputsingle = True
        dens_targets_cgs = [cu.getmeandensity(meandef, cosmopars)]
    else:
        outputsingle = False
        dens_targets_cgs = [cu.getmeandensity(md, cosmopars) for md in meandef]
        
    r2 = np.sum((coords - com_simunits[np.newaxis, :])**2, axis=1)
    del coords
    rorder = np.argsort(r2)
    r2_order = r2[rorder]
    # apparent truncation error issues in cumsum for some 
    # simulations/snapshots using float32. (enclosed mass plateaus)
    masses_order = np.asarray(masses[rorder], dtype=np.float64)
    del masses, r2, rorder
    dens_targets = [target / toCGS_m * toCGS_c**3 for target in \
                    dens_targets_cgs]
    dens2_order = (cmass2 := np.cumsum(masses_order)**2) \
                  / ((4. * np.pi / 3)**2 * r2_order**3)
    # plotting sqrt dens2_order vs. sqrt r2_order, dens_targets -> 
    # intersect at ~200 ckpc/h at z=2.8 for the m13 guinea pig
    # seems reasonable...
    # tests on buggy constant halo mass issue
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.))
    outfn1 = ('/scratch1/08466/tg877653/output/'
              f'debug_rvir_{path.split("/")[-1]}_snapshot{snapshot}.pdf')
    ax.plot(np.sqrt(r2_order[::1000]), np.sqrt(dens2_order[::1000]))
    ax.set_xlabel('radius [sim units]', fontsize=12)
    ax.set_ylabel('density [sim units]', fontsize=12)
    ax.set_yscale('log')
    ax.set_xscale('log')
    for t in dens_targets:
        ax.axhline(t)
    plt.savefig(outfn1, bbox_inches='tight')

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5.5, 5.))
    outfn2 = ('/scratch1/08466/tg877653/output/'
              f'debug2_rvir_{path.split("/")[-1]}_snapshot{snapshot}.pdf')
    ax.plot(np.sqrt(r2_order[::1000]), np.sqrt(cmass2[::1000]))
    ax.set_xlabel('radius [sim units]', fontsize=12)
    ax.set_ylabel('cumul. mass [sim units]', fontsize=12)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig(outfn2, bbox_inches='tight')
    print('Last 100 values in cmass2 array (cumul. mass [sim units])**2:')
    print(cmass2[:100])


    rsols_cgs = []
    msols_cgs = []
    xydct = {'x': r2_order, 'y': dens2_order}
    for dti, dens_target in enumerate(dens_targets):
        sols = find_intercepts(None, None, dens_target**2, xydct=xydct)
        # no random low-density holes or anything
        sols = sols[sols >= r2_order[minpart_halo]]
        if len(sols) == 0:
            msg = 'No solutions found for density {}'.format(meandef[dti])
            print(msg)
        elif len(sols) == 1:
            rsol = np.sqrt(sols[0])
            msol = 4. * np.pi / 3. * rsol**3 * dens_target
            rsols_cgs.append(rsol * toCGS_c)
            msols_cgs.append(msol * toCGS_m)
        else:
            # technically a solution, but there will be some 
            # particle noise; smoothing?
            # on the other hand, those effects are proabably tiny
            sols_kpc = np.sqrt(sols) * toCGS_c / (1e-3 * c.cm_per_mpc)
            print('Found radius solution options [pkpc] {}'.format(sols_kpc))
            print('Selected first in list')
            rsol = np.sqrt(sols[0])
            msol = 4. * np.pi / 3. * rsol**3 * dens_target
            rsols_cgs.append(rsol * toCGS_c)
            msols_cgs.append(msol * toCGS_m)
    com_cgs = com_simunits * toCGS_c
    if outputsingle:
        rsols_cgs = rsols_cgs[0]
        msols_cgs = msols_cgs[0]
    outdct = {'Xc_cm': com_cgs[0], 'Yc_cm': com_cgs[1], 'Zc_cm': com_cgs[2],
              'Rvir_cm': rsols_cgs, 'Mvir_g': msols_cgs}
    return  outdct, todoc

def gethalodata_shrinkingsphere(path, snapshot, meandef=('200c', 'BN98')):
    '''
    same in/output as calchalodata_shrinkingsphere,
    but reads data from file if stored, and stores data to a temporary
    file if not.
    Run adddata_cenrvir() to add the temporary file data to the main 
    file. (Doing this during the main run could cause issues if multiple
    processes try to write to the same file at the same time.)
    '''
    # this must contain *all* todoc entries from the previous function
    # except 'parttypes_used', which is assumed to be everything but
    # PartType2 (lo-res DM) present in the NumPart_Total table
    usedvals_calchalo = {'shrinkfrac': 0.025, 
                         'minparticles': 1000., 
                         'initialradiusfactor': 1.,
                         'minpart_halo': 1000.}
    snap = rf.get_Firesnap(path, snapshot, filetype='snap')
    with h5py.File(snap.firstfilen) as f:
        pts = list(f['Header'].attrs['NumPart_Total'])
    pts = [ind for ind in range(len(pts)) if pts[ind] > 0]
    pts.remove(2)
    pts.sort()
    usedvals_calchalo['parttypes_used'] = tuple(pts)
                                                
    fdir = ol.dir_halodata
    filen_main = ol.filen_halocenrvir
    
    newcalc = False
    pparts = path.split('/')
    while '' in pparts:
        pparts.remove('')
    if pparts[-1] == 'output':
        pparts = pparts[:-1]
    simid = pparts[-1]
    
    try:
        with h5py.File(filen_main, 'r') as f:
            todoc = {}
            halodat = {}
            # check simulation run, snapshot
            if simid in f:
                smgrp = f[simid]
            else:
                raise NoStoredMatchError(f'Simulation {simid}')
            snn = f'snap_{snapshot}'
            if snn in smgrp:
                sngrp = smgrp[snn]
            else:
                raise NoStoredMatchError(f'Simulation {simid}, {snn}')
            cosmopars = {}
            for key, val in sngrp['cosmopars'].attrs.items():
                cosmopars[key] = val
            todoc['cosmopars'] = cosmopars
            # check center finding
            cengrpns = [grp for grp in sngrp.keys() if grp.startswith('cen')]
            for cengrpn in cengrpns:
                cgrp = sngrp[cengrpn]
                tomatch = usedvals_calchalo.keys()
                # using: 'parttypes_used' is a tuple, comparison to 
                # array gives boolean array, or False if different lengths
                if np.all([np.all(usedvals_calchalo[key] == cgrp.attrs[key])\
                           for key in tomatch]):
                    halodat['Xc_cm'] = cgrp.attrs['Xc_cm']
                    halodat['Yc_cm'] = cgrp.attrs['Yc_cm']
                    halodat['Zc_cm'] = cgrp.attrs['Zc_cm']
                    todoc.update(usedvals_calchalo)
                    break
            if 'Xc_cm' not in halodat:
                msg = (f'Simulation {simid}, {snn}, '
                       f'center finding parameters {usedvals_calchalo}')
                raise NoStoredMatchError(msg)
            # check Mvir/Rvir def.
            outputsingle = False
            if isinstance(meandef, type('')):
                meandef = [meandef]
                outputsingle = True
            halodat['Rvir_cm'] = [] 
            halodat['Mvir_g'] = [] 
            for md in meandef:
                subgrpn = f'Rvir_{md}'
                if subgrpn in cgrp:
                    sgrp = cgrp[subgrpn]
                    halodat['Rvir_cm'].append(sgrp.attrs['Rvir_cm'])
                    halodat['Mvir_g'].append(sgrp.attrs['Mvir_g'])
                else:
                    msg = (f'Simulation {simid}, {snn}, '
                           f'center finding parameters {usedvals_calchalo}, '
                           f'overdensity definition {md}')
                    raise NoStoredMatchError(msg)
            if outputsingle:
                halodat['Rvir_cm'] = halodat['Rvir_cm'][0]
                halodat['Mvir_g'] = halodat['Mvir_g'][0]
        print(f'Retrieved stored halo data from {filen_main}')
        return halodat, todoc    
    except NoStoredMatchError as err:
        print(err)
        print('Center, Rvir were not stored')
        newcalc = True
    
    if newcalc:
        halodat, todoc = calchalodata_shrinkingsphere(path, snapshot, 
                                                      meandef=meandef)
        filen = fdir + f'temp_cen_rvir_{uuid.uuid1()}.hdf5'
        if os.path.isfile(filen):
            msg = f'Temporary center/Rvir file {filen} already exists'
            raise RuntimeError(msg)
        with h5py.File(filen, 'w') as f:
            # sim, snap groups
            smgrp = f.create_group(simid)
            sngrp = smgrp.create_group(f'snap_{snapshot}')
            cmgrp = sngrp.create_group('cosmopars')
            for key in todoc['cosmopars']:
                cmgrp.attrs.create(key, todoc['cosmopars'][key])
            del todoc['cosmopars']
            # sim/snap subgroup for center pars.
            cengrp = sngrp.create_group('cen0')
            for cv in ['Xc_cm', 'Yc_cm', 'Zc_cm']:
                cengrp.attrs.create(cv, halodat[cv])
            for key in todoc:
                val = todoc[key]
                if isinstance(val, type('')):
                    val = np.string_(val)
                cengrp.attrs.create(key, val)
            # center pars. subgroups for mvir/rvir def.
            if isinstance(meandef, type('')):
                meandef = [meandef]
                halodat['Rvir_cm'] = [halodat['Rvir_cm']] 
                halodat['Mvir_g'] = [halodat['Mvir_g']] 
            for md, rv, mv in zip(meandef, halodat['Rvir_cm'], 
                                  halodat['Mvir_g']):
                gn = f'Rvir_{md}'
                vgrp = cengrp.create_group(gn)
                vgrp.attrs.create('Rvir_cm', rv)
                vgrp.attrs.create('Mvir_g', mv)
        return halodat, todoc

def adddata_cenrvir(rmtemp=False):
    '''
    put data in temporary cenrvir files into the main file

    Parameters:
    -----------
    rmtemp: bool
        remove temporary storage files after marking it as duplicate
        (This would be the second run of this function if a file was
        new -- this is done to ensure a file is not deleted if 
        something went wrong unexpectedly.)
    '''
    mainfilen =  ol.filen_halocenrvir 
    searchcrit = ol.dir_halodata + 'temp_cen_rvir_*.hdf5'
    tempfilens = glob.glob(searchcrit)
    if len(tempfilens) == 0:
        print('No new data to add')
        return None
    with h5py.File(mainfilen, 'a') as fo:
        #print(mainfilen)
        for tfn in tempfilens:
            with h5py.File(tfn, 'r') as fi:
                # should have one sim, snap, cen group
                # possibly multiple rvir definitions
                simid = next(iter(fi.keys()))
                #print('simid: ', simid)
                #print('main file keys: ', list(fo.keys()))
                if simid not in fo: #easy, copy whole thing
                    fi.copy(fi[simid], fo, name=simid)
                    print(f'Added file {tfn}:')
                    print(f'{simid}')
                    continue
                fo_smgrp = fo[simid]
                fi_smgrp = fi[simid]
                sngrpn = next(iter(fi_smgrp.keys()))
                #print('snap: ', sngrpn)
                #print('main file fo_smgrp keys: ', list(fo_smgrp.keys()))
                if sngrpn not in fo_smgrp: #easy, copy whole thing
                    fi.copy(fi_smgrp[sngrpn], fo_smgrp, name=sngrpn)
                    print(f'Added file {tfn}:')
                    print(f'{simid}, {sngrpn}')
                    continue
                # center matching/copy
                fo_sngrp = fo_smgrp[sngrpn]
                fi_sngrp = fi_smgrp[sngrpn]
                #print('main file cens: ', list(fo_sngrp.keys()))
                if 'cen0' not in fo_sngrp:
                    fi.copy(fi_sngrp['cen0'], fo_sngrp, name='cen0')
                    print(f'Added file {tfn}:')
                    print(f'{simid}, {sngrpn}, first center')
                    continue
                cens_fo = [grp for grp in fo_sngrp.keys() \
                           if grp.startswith('cen')]
                fi_cgrp = fi_sngrp['cen0']
                tocheck = ['Xc_cm', 'Yc_cm', 'Zc_cm']
                anymatch = False
                for cengrpn in cens_fo:
                    _fo_cgrp = fo_sngrp[cengrpn]
                    tomatch = set(fi_cgrp.attrs.keys()) - set(tocheck)
                    # using: 'parttypes_used' is a tuple, comparison to 
                    # array gives boolean array, or False if different lengths
                    if np.all([np.all(_fo_cgrp.attrs[key] \
                                      == fi_cgrp.attrs[key])\
                               for key in tomatch]):
                        fo_cgrp = _fo_cgrp
                        anymatch = True
                        if not np.all([np.all(_fo_cgrp.attrs[key] \
                                       == fi_cgrp.attrs[key])\
                                       for key in tocheck]):
                            msg = (f'{mainfilen} and {tfn} have matching'
                                   f'simulation {simid}, {sngrpn}, '
                                   f'center finding, but different centers:\n'
                                   f'{fo_cgrp.attrs.items()},\n'
                                   f'{fi_cgrp.attrs.items()}')
                            raise RuntimeError(msg)
                if not anymatch:
                    fo_cgrpn = f'cen{len(cens_fo)}'
                    fi.copy(fi_cgrp, fo_sngrp, name=fo_cgrpn)
                    print(f'Added file {tfn}:')
                    print(f'{simid}, {sngrpn}, new center')
                    continue
                #print('main file densities: ', list(fo_cgrp.keys()))
                # mvir/rvir matching/copy
                fi_mrdefs = [grp for grp in fi_cgrp.keys() \
                             if grp.startswith('Rvir_')]
                #print('new densities: ', fi_mrdefs)
                for mdn in fi_mrdefs:
                    if mdn in fo_cgrp:
                        fi_dct = dict(fi_cgrp[mdn].attrs.items())
                        fo_dct = dict(fo_cgrp[mdn].attrs.items())
                        if fi_dct == fo_dct:
                            continue
                        else:
                            msg = (f'{mainfilen} and {tfn} have matching'
                                   f'simulation {simid}, {sngrpn}, '
                                   f'centers, but different Mvir or Rvir:\n'
                                   f'{fi_dct},\n'
                                   f'{fo_dct}')
                            raise RuntimeError(msg)
                    else:
                        fi.copy(fi_cgrp[mdn], fo_cgrp, name=mdn)
                        print(f'Added file {tfn}:')
                        print(f'{simid}, {sngrpn}, {fi_mrdefs}')
                        continue
            print(f'skipped {tfn}; duplicate data')
            if rmtemp:
                print(f'deleting {tfn}')
                os.remove(tfn)

def mainhalodata_AHFsmooth(path, snapnum):
    '''
    get properties of the main halo in the snapshot from halo_00000_smooth.dat
    assume units are intrinsic simulation units
    '''
    fn = path + '/halo/ahf/halo_00000_smooth.dat'
    df = pd.read_csv(fn, sep='\t')
    i = np.where(df['snum'] == snapnum)[0][0]
    out = {}
    # units from AHF docs: http://popia.ft.uam.es/AHF/files/AHF.pdf
    props = ['Mvir', 'Rvir', 'Xc', 'Yc', 'Zc']
    outprops = {'Mvir': 'Mvir_Msunoverh',
                'Rvir': 'Rvir_ckpcoverh',
                'Xc':   'Xc_ckpcoverh',
                'Yc':   'Yc_ckpcoverh',
                'Zc':   'Zc_ckpcoverh'}
    for prop in props:
        out[outprops[prop]] = df[prop][i]
    return out

def halodata_rockstar(path, snapnum, select='maxmass', 
                      masspath='mass.vir'):
    '''
    retrieve position, mass, and radius from rockstar halo data
    uses the Bryan and Norman overdensity mass (.vir in rockstar)
    
    Parameters:
    -----------
    path: str
        path to the directory containing the output and halo directories
        and the snapshot_times.txt file
    snapnum: int
        snapshot number
    select: {'maxmass', 'mainprog', int}
        how to select the halo to use
        'maxmass': highest mass in the snapshot
        'mainprog': main progenitor of the highest-mass halo at the
                    lowest redshift available
        int: index of the halo in the snapshot catalogue (Note: not the
             tree index that's unique across snapshots)
    masspath: path in hdf5 file to mass to use
        e.g., mass.mvir, mass.200c, mass.200m
    '''
    # options: 'BN98', '####c', '######m'
    if masspath == 'mass.vir':
        meandensdef = 'BN98'
    else:
        meandensdef = masspath.split('.')[-1]
    out = {}
    if select == 'maxmass' or isinstance(select, int):
        hal = ha.io.IO.read_catalogs('snapshot', snapnum, path)
        if select == 'maxmass':
            haloind = np.argmax(hal[masspath])
            print('Using snapshot halo index: {}'.format(haloind))
        else:
            haloind = select
        out['Mvir_Msun'] = hal[masspath][haloind]
        out['Xc_ckpc'], out['Yc_ckpc'], out['Zc_ckpc'] = \
            hal['position'][haloind]
        cosmopars = {}
        cosmopars['omegalambda'] = hal.Cosmology['omega_lambda']
        cosmopars['omegam'] = hal.Cosmology['omega_matter']
        cosmopars['omegab'] = hal.Cosmology['omega_baryon']
        cosmopars['h'] = hal.Cosmology['hubble']
        cosmopars['a'] = hal.snapshot['scalefactor']
        cosmopars['z'] = hal.snapshot['redshift']
    elif select == 'mainprog':
        halt = ha.io.IO.read_tree(simulation_directory=path, 
                                  species_snapshot_indices=[snapnum])
        # high-mass stuff isn't always run to z=0
        finalsnap = np.max(halt['snapshot'])
        wherefinalsnap = np.where(halt['snapshot'] == finalsnap)[0]
        whereind_maxmfinal = np.argmax(halt[masspath][wherefinalsnap])
        treeind_maxmfinal = wherefinalsnap[whereind_maxmfinal]
        prog_main_index = treeind_maxmfinal
        while prog_main_index >= 0:
            snap_current = halt['snapshot'][prog_main_index]
            if snap_current == snapnum:
                break
            if prog_main_index < 0:
                msg = 'No main progenitor at snapshot {} was found'
                raise RuntimeError(msg.format(snap_current + 1))
            prog_main_index = halt['progenitor.main.index'][prog_main_index]
        if bool(halt['am.phantom'][prog_main_index]):
            msg = 'This halo was not found by Rockstar,'+\
                  ' but interpolated'
            raise RuntimeError(msg)
        out['Mvir_Msun'] = halt['mass'][prog_main_index]
        out['Xc_ckpc'], out['Yc_ckpc'], out['Zc_ckpc'] = \
            halt['position'][prog_main_index]
        cosmopars = {}
        cosmopars['omegalambda'] = halt.Cosmology['omega_lambda']
        cosmopars['omegam'] = halt.Cosmology['omega_matter']
        cosmopars['omegab'] = halt.Cosmology['omega_baryon']
        cosmopars['h'] = halt.Cosmology['hubble']
        # get redshift from halo catalog
        hal = ha.io.IO.read_catalogs('snapshot', snapnum, path)
        cosmopars['a'] = hal.snapshot['scalefactor']
        cosmopars['z'] = hal.snapshot['redshift']
    
    meandens = cu.getmeandensity(meandensdef, cosmopars)
    #M = r_mean * 4/3 np.pi R63
    out['Rvir_cm'] = (3. / (4. * np.pi) * out['Mvir_Msun'] \
                      * c.solar_mass / meandens)**(1./3.)
    return out, cosmopars

def test_mainhalodata_units_ahf(opt=1, dirpath=None, snapnum=None,
                            printfile=None):
    
    if opt == 1: # redshift 0 test
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_600/snapshot_600.0.hdf5'
        snapnum = 600
    elif opt == 2: # higher z test 
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_399/snapshot_399.0.hdf5'
        snapnum = 399
    elif opt == 3: # try other z
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        snapfile = dirpath + 'output/snapdir_492/snapshot_492.0.hdf5'
        snapnum = 492
    elif opt is None:
        pathopts = ['output/snapdir_{sn:03d}/snapshot_{sn:03d}.0.hdf5',
                    'output/snapshot_{sn:03d}.hdf5']
        goodpath = False
        for pathopt in pathopts:
            snapfile = dirpath + pathopt.format(sn=snapnum)
            if os.path.isfile(snapfile):
                goodpath = True
                break
        if not goodpath:
            tried = [dirpath + pathopts.format()]
            msg = 'Could not find snapshot {} in {}. Tried:'.format(snapnum, dirpath)
            msg = msg + '\n' + '\n'.join(tried)
            raise RuntimeError(msg)
    else:
        msg = 'test_mainhalodata_units parameter opt = {} is invalid'
        raise ValueError(msg.format(opt))

    halodat = mainhalodata_AHFsmooth(dirpath, snapnum)
    snap = rf.Firesnap(snapfile) 
    cen = np.array([halodat['Xc_ckpcoverh'], 
                    halodat['Yc_ckpcoverh'], 
                    halodat['Zc_ckpcoverh']])
    cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    rvir_cm = halodat['Rvir_ckpcoverh'] * snap.cosmopars.a\
              * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    print('Cosmology:')
    print(snap.cosmopars.getdct())
    print('Center [AHF units]: {}'.format(cen))
    print('Rvir [AHF units]: {}'.format(halodat['Rvir_ckpcoverh']))
    print('Center [attempted cm]: {}'.format(cen_cm))
    print('Rvir [attempted cm]: {}'.format(rvir_cm))
    
    # gas
    coords_pt0 = snap.readarray_emulateEAGLE('PartType0/Coordinates')
    coords_pt0_toCGS = snap.toCGS
    masses_pt0 = snap.readarray_emulateEAGLE('PartType0/Masses')
    masses_pt0_toCGS = snap.toCGS
    # sanity check
    med_c = np.median(coords_pt0, axis=0)
    print('Median gas coords [sim units]: {}'.format(med_c))
    print('Median gas coordinates [cm]: {}'.format(med_c * coords_pt0_toCGS))

    d2 = np.sum((coords_pt0 - cen_cm / coords_pt0_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt0_toCGS) **2
    hm_pt0 = np.sum(masses_pt0[sel])
    print('Halo gas mass (sim units): ', hm_pt0)
    print('Selected {}/{} particles'.format(np.sum(sel), len(sel)))
    del coords_pt0
    del masses_pt0
    del d2
    del sel
    # dm (high-res)
    coords_pt1 = snap.readarray_emulateEAGLE('PartType1/Coordinates')
    coords_pt1_toCGS = snap.toCGS
    masses_pt1 = snap.readarray_emulateEAGLE('PartType1/Masses')
    masses_pt1_toCGS = snap.toCGS
    med_c = np.median(coords_pt1, axis=0)
    print('Median DM coords [sim units]: {}'.format(med_c))
    print('Median DM coordinates [cm]: {}'.format(med_c * coords_pt1_toCGS))
    d2 = np.sum((coords_pt1 - cen_cm / coords_pt1_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt1_toCGS) **2
    hm_pt1 = np.sum(masses_pt1[sel])
    print('Halo dm mass (sim units): ', hm_pt1)
    print('Selected {}/{} particles'.format(np.sum(sel), len(sel)))
    del coords_pt1
    del masses_pt1
    del d2
    del sel
    # stars
    coords_pt4 = snap.readarray_emulateEAGLE('PartType4/Coordinates')
    coords_pt4_toCGS = snap.toCGS
    masses_pt4 = snap.readarray_emulateEAGLE('PartType4/Masses')
    masses_pt4_toCGS = snap.toCGS
    med_c = np.median(coords_pt4, axis=0)
    print('Median star coords [sim units]: {}'.format(med_c))
    print('Median star coordinates [cm]: {}'.format(med_c * coords_pt4_toCGS))

    d2 = np.sum((coords_pt4 - cen_cm / coords_pt4_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt4_toCGS) **2
    hm_pt4 = np.sum(masses_pt4[sel])
    print('Halo stellar mass (sim units): ', hm_pt4)
    del coords_pt4
    del masses_pt4
    del d2
    del sel
    hm = hm_pt0 + hm_pt1 + hm_pt4

    msg = 'Got halo mass {hm}, listed Mvir is {Mvir}'
    hm_list_msun = halodat['Mvir_Msunoverh'] / snap.cosmopars.h
    hm_sum_msun = hm * (masses_pt0_toCGS / cu.c.solar_mass)
    print(msg.format(hm=hm_sum_msun, Mvir=hm_list_msun))
    hm_logmsun = np.log10(hm) + np.log10(masses_pt0_toCGS / cu.c.solar_mass)
    print('sum total is 10^{logm} Msun'.format(logm=hm_logmsun))

    if printfile is not None:
        new = not os.path.isfile(printfile)
        with open(printfile, 'a') as f:
            if new:
                columns = ['snapnum', 'redshift', 'Mvir_sum_Msun', 'Mvir_AHF_Msun']
                f.write('\t'.join(columns) + '\n')
            vals = [snapnum, snap.cosmopars.z, hm_sum_msun, hm_list_msun]
            f.write('\t'.join([str(val) for val in vals]) + '\n')

def test_mainhalodata_units_rockstar(opt=1, dirpath=None, snapnum=None,
                                     printfile=None, **kwargs):
    
    if opt == 1: # redshift 1 test
        dirpath = '/projects/b1026/snapshots/MassiveFIRE/h113_A4_res33000/'
        snapfile = dirpath + 'output/snapshot_277.hdf5'
        snapnum = 277
    elif opt == 2: # higher z test 
        dirpath = '/projects/b1026/snapshots/MassiveFIRE/h113_A4_res33000/'
        snapfile = dirpath + 'output/snapshot_200.hdf5'
        snapnum = 200
    elif opt == 3: # try other z
        dirpath = '/projects/b1026/snapshots/MassiveFIRE/h113_A4_res33000/'
        snapfile = dirpath + 'output/snapshot_100.hdf5'
        snapnum = 100
    elif opt is None:
        pathopts = ['output/snapdir_{sn:03d}/snapshot_{sn:03d}.0.hdf5',
                    'output/snapshot_{sn:03d}.hdf5']
        goodpath = False
        for pathopt in pathopts:
            snapfile = dirpath + pathopt.format(sn=snapnum)
            if os.path.isfile(snapfile):
                goodpath = True
                break
        if not goodpath:
            tried = [dirpath + pathopts.format()]
            msg = 'Could not find snapshot {} in {}. Tried:'.format(snapnum, dirpath)
            msg = msg + '\n' + '\n'.join(tried)
            raise RuntimeError(msg)
    else:
        msg = 'test_mainhalodata_units parameter opt = {} is invalid'
        raise ValueError(msg.format(opt))

    halodat, halo_cosmopars = halodata_rockstar(dirpath, snapnum)
    snap = rf.get_Firesnap(dirpath, snapnum) 
    cen = np.array([halodat['Xc_ckpc'], 
                    halodat['Yc_ckpc'], 
                    halodat['Zc_ckpc']])
    cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc
    rvir_cm = halodat['Rvir_cm'] 
    print('Cosmology (snapshot):')
    print(snap.cosmopars.getdct())
    print('Cosmology (halo data):')
    print(halo_cosmopars)
    print('Center [rockstar units]: {}'.format(cen))
    print('Rvir [pkpc]: {}'.format(rvir_cm / (1e-3 * c.cm_per_mpc)))
    print('Center [attempted cm]: {}'.format(cen_cm))
    print('Rvir [attempted cm]: {}'.format(rvir_cm))
    
    # gas
    coords_pt0 = snap.readarray_emulateEAGLE('PartType0/Coordinates')
    coords_pt0_toCGS = snap.toCGS
    masses_pt0 = snap.readarray_emulateEAGLE('PartType0/Masses')
    masses_pt0_toCGS = snap.toCGS
    # sanity check
    med_c = np.median(coords_pt0, axis=0)
    print('Median gas coords [sim units]: {}'.format(med_c))
    print('Median gas coordinates [cm]: {}'.format(med_c * coords_pt0_toCGS))

    d2 = np.sum((coords_pt0 - cen_cm / coords_pt0_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt0_toCGS) **2
    hm_pt0 = np.sum(masses_pt0[sel])
    print('Halo gas mass (sim units): ', hm_pt0)
    print('Selected {}/{} particles'.format(np.sum(sel), len(sel)))
    del coords_pt0
    del masses_pt0
    del d2
    del sel
    # dm (high-res)
    coords_pt1 = snap.readarray_emulateEAGLE('PartType1/Coordinates')
    coords_pt1_toCGS = snap.toCGS
    masses_pt1 = snap.readarray_emulateEAGLE('PartType1/Masses')
    masses_pt1_toCGS = snap.toCGS
    med_c = np.median(coords_pt1, axis=0)
    print('Median DM coords [sim units]: {}'.format(med_c))
    print('Median DM coordinates [cm]: {}'.format(med_c * coords_pt1_toCGS))
    d2 = np.sum((coords_pt1 - cen_cm / coords_pt1_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt1_toCGS) **2
    hm_pt1 = np.sum(masses_pt1[sel])
    print('Halo dm mass (sim units): ', hm_pt1)
    print('Selected {}/{} particles'.format(np.sum(sel), len(sel)))
    del coords_pt1
    del masses_pt1
    del d2
    del sel
    # stars
    coords_pt4 = snap.readarray_emulateEAGLE('PartType4/Coordinates')
    coords_pt4_toCGS = snap.toCGS
    masses_pt4 = snap.readarray_emulateEAGLE('PartType4/Masses')
    masses_pt4_toCGS = snap.toCGS
    med_c = np.median(coords_pt4, axis=0)
    print('Median star coords [sim units]: {}'.format(med_c))
    print('Median star coordinates [cm]: {}'.format(med_c * coords_pt4_toCGS))

    d2 = np.sum((coords_pt4 - cen_cm / coords_pt4_toCGS)**2, axis=1)
    sel = d2 <= (rvir_cm / coords_pt4_toCGS) **2
    hm_pt4 = np.sum(masses_pt4[sel])
    print('Halo stellar mass (sim units): ', hm_pt4)
    del coords_pt4
    del masses_pt4
    del d2
    del sel
    hm = hm_pt0 + hm_pt1 + hm_pt4

    msg = 'Got halo mass {hm}, listed Mvir is {Mvir}'
    hm_list_msun = halodat['Mvir_Msun']
    hm_sum_msun = hm * (masses_pt0_toCGS / cu.c.solar_mass)
    print(msg.format(hm=hm_sum_msun, Mvir=hm_list_msun))
    hm_logmsun = np.log10(hm) + np.log10(masses_pt0_toCGS / cu.c.solar_mass)
    print('sum total is 10^{logm} Msun'.format(logm=hm_logmsun))

    if printfile is not None:
        new = not os.path.isfile(printfile)
        with open(printfile, 'a') as f:
            if new:
                columns = ['snapnum', 'redshift', 'Mvir_sum_Msun', 'Mvir_rockstar_Msun']
                f.write('\t'.join(columns) + '\n')
            vals = [snapnum, snap.cosmopars.z, hm_sum_msun, hm_list_msun]
            f.write('\t'.join([str(val) for val in vals]) + '\n')


# checkinh halo_0000_smooth.dat:
# Mvir is exactly flat over a large range of redshift values in that file
# might be an AHF issue?
def test_mainhalodata_units_multi(dirpath, printfile, version='ahf',
                                  **kwargs):
    print('running test_mainhalodata_units_multi')
    _snapdirs = os.listdir(dirpath + 'output/')
    snaps = []
    for _sd in _snapdirs:
        # looking for something like snapdir_196, extract 196
        if _sd.startswith('snapdir'):
            _snap = int(_sd.split('_')[-1])
            # special case, permissions error
            try: 
                os.listdir(dirpath + 'output/' + _sd)
                snaps.append(_snap)
            except PermissionError:
                # shows up seemingly randomly
                print('\nskipping snapshot {} due to permissions issues\n'.format(_snap))
                continue
        elif _sd.startswith('snapshot') and _sd.endswith('.hdf5'):
            # something like snapshot_164.hdf5
            print(_snap)
            _snap = int((_sd.split('_')[-1]).split('.')[0])
            try:
                f = h5py.File(dirpath + 'output/' + _sd, 'r')
                f.close()
            except Exception as err:
                print('\nSkipping snapshot {} due to h5py read issues:')
                print(err)
                print('\n')
                
    for snap in snaps:
        print('Snapshot ', snap)
        if version == 'ahf':
            test_mainhalodata_units_ahf(opt=None, dirpath=dirpath, 
                                        snapnum=snap,
                                        printfile=printfile)
        
        elif version == 'rockstar':
            test_mainhalodata_units_rockstar(opt=None, dirpath=dirpath, 
                                        snapnum=snap,
                                        printfile=printfile, **kwargs)
        else: 
            raise ValueError('invalid version option: {}'.format(version))
        print('\n')


def test_mainhalodata_units_multi_handler(opt=1):
    if opt == 1:
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        printfile = '/projects/b1026/nastasha/tests/start_fire/AHF_unit_tests/'
        printfile += 'metal_diffusion__m12i_res7100.txt'
        version = 'ahf'
    elif opt == 2:
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m11i_res7100/'
        printfile = '/projects/b1026/nastasha/tests/start_fire/AHF_unit_tests/'
        printfile += 'metal_diffusion__m11i_res7100.txt'
        version = 'ahf'
    else:
        raise ValueError('opt {} is not allowed'.format(opt))
    print('Running test_mainhalodata_units_multi(dirpath, printfile)')
    print('dirpath: ', dirpath)
    print('printfile: ', printfile)
    test_mainhalodata_units_multi(dirpath, printfile, version=version)

# tested -> seems to work
# dust on/off, redshifts 1.0, 2.8, Z=0.01, 0.0001
# compared FIRE interpolation to neighboring table values
# tested ions sum to 1: lintable=True -> yes, except for molecules,
#                       dust depletion (high nH, low T, more at higher Z)
#                       lintable=False -> no, in some regions of phase 
#                       space, without good physics reasons
def get_ionfrac(snap, ion, indct=None, table='PS20', simtype='fire',
                ps20depletion=True, lintable=True):
    '''
    Get the fraction of an element in a given ionization state in 
    a given snapshot.

    Parameters:
    -----------
    snap: snapshot reader obect
        exact class depends on the simulation
    ion: str
        ion to get the fraction of. Format e.g. 'o6', 'fe17'
    indct: dict or None
        dictionary containing any of the followign arrays
        'filter': bool, size of arrays returned by the snap reader
                  determines which resolution elements to use.
                  If not repesent, all resolution elements are used.
        If not present, the following values are obtained using snap:
        'logT': temperature in log10 K. 
        'lognH': hydrogen number density in log10 particles / cm**3
        'logZ': metal mass fraction in log10 fraction of total mass (no 
                solar scaling)
    table: {'PS20'}
        Which ionization tables to use.
    simtype: {'fire'}
        What format does the simulation reader class snap have?
    ps20depletion: bool
        Take away a fraction of the ions to account for the fraction of the
        parent element depleted onto dust.
    lintable: bool 
        interpolate the ion balance (and depletion, if applicable) in linear
        space (True), otherwise, it's done in log space (False) 

    Returns:
    --------
        the fraction of the parent element nuclei that are a part of the 
        desired ion
    '''
    if simtype == 'fire':
        readfunc = snap.readarray_emulateEAGLE
        prepath = 'PartType0/'
        redshift = snap.cosmopars.z
    else:
        raise ValueError('invalid simtype option: {}'.format(simtype))
    
    if indct is None:
        indct = {}
    if 'filter' in indct: # gas selection, e.g. a spatial region
        filter = indct['filter']
    else:
        filter = slice(None, None, None)
    if 'logT' in indct: # should be in [log10 K]
        logT = indct['logT']
    else:
        logT = np.log10(readfunc(prepath + 'Temperature')[filter])
        tocgs = snap.toCGS
        if not np.isclose(tocgs, 1.):
            logT += np.log10(tocgs)
    if 'lognH' in indct: # should be in log10 cm**-3
        lognH = indct['lognH']
    else:
        hdens = readfunc(prepath + 'Density')[filter]
        d_tocgs = snap.toCGS
        hmassfrac = readfunc(prepath + 'ElementAbundance/Hydrogen')[filter]
        hmassfrac_tocgs = snap.toCGS
        hdens *= hmassfrac 
        hdens *= d_tocgs * hmassfrac_tocgs / (c.atomw_H * c.u)
        del hmassfrac
        lognH = np.log10(hdens)
        del hdens
    if table in ['PS20']:
        if 'logZ' in indct: # no solar normalization, 
            #just straight mass fraction
            logZ = indct['logZ']
        else:
            logZ = readfunc(prepath + 'Metallicity')[filter]    
            logZ = np.log10(logZ)
            tocgs = snap.toCGS
            if not np.isclose(tocgs, 1.):
                logZ += np.log10(tocgs)
        # Inputting logZ values of -np.inf (zero metallicity, does 
        # happen) leads to NaN ion fractions in interpolation.
        # Since the closest edge of the tabulated values is used anyway
        # it's safe to substute a tiny value like -100.
        if np.any(logZ == -np.inf):
            logZ = logZ.copy()
            logZ[logZ == -np.inf] = -100.
    if table == 'PS20':
        interpdct = {'logT': logT, 'lognH': lognH, 'logZ': logZ}
        iontab = linetable_PS20(ion, redshift, emission=False, vol=True,
                 ionbalfile=ol.iontab_sylvia_ssh, 
                 emtabfile=ol.emtab_sylvia_ssh, lintable=lintable)
        ionfrac = iontab.find_ionbal(interpdct, log=False)
        if ps20depletion:
            ionfrac *= (1. - iontab.find_depletion(interpdct))
    else:
        raise ValueError('invalid table option: {}'.format(table))
    ## debug map NaN values
    #if np.any(ionfrac) < 0.:
    #    print('some ion fractions < 0 from get_ionfrac')
    #    print('min/max ion fraction values: {}, {}'.format(np.min(ionfrac),
    #                                                       np.max(ionfrac)))
    #if np.any(np.isnan(ionfrac)):
    #    msg = 'Some ion fractions were NaN: {} out of {}'
    #    print(msg.format(np.sum(np.isnan(ionfrac)), len(ionfrac)))
    # if np.any(np.isnan(iontab.iontable_T_Z_nH)):
    #     print('NaN values in the table to be interpolated')
    #     print('Parameters used in linetable_PS20:')
    #     print('ion: ', ion)
    #    print('redshift: ', redshift)
    #    print('emission: ', False)
    #    print('vol: ', True)
    #    print('ionbalfile: ', ol.iontab_sylvia_ssh)
    #    print('emtabfile: ', ol.emtab_sylvia_ssh)
    #    print('lintable: ', lintable)
    return ionfrac

# untested, including lintable option and consistency with table values
# do a test like test_ionbal_calc before using
# (note that element abundance rescaling and volume multiplication will
# make the direct table comparison harder than with the ion fractions;
# best to do some direct input tests for that)
def get_loglinelum(snap, line, indct=None, table='PS20', simtype='fire',
                   ps20depletion=True, lintable=True, ergs=False,
                   density=False):
    '''
    Get the luminosity (density) of a series of resolution elements.

    Parameters:
    -----------
    snap: snapshot reader obect
        exact class depends on the simulation
    line: str
        line to calculate the luminosity of. Should match the line
        list in the table.
    indct: dict or None
        dictionary containing any of the followign arrays
        'filter': bool, size of arrays returned by the snap reader
                  determines which resolution elements to use.
                  If not repesent, all resolution elements are used.
        If not present, the following values are obtained using snap:
        'logT': temperature in log10 K. 
        'lognH': hydrogen number density in log10 particles / cm**3
        'logZ': metal mass fraction in log10 fraction of total mass (no 
                solar scaling)
        'eltmassf': mass fraction of the line-producing 
                element (no solar scaling)
        'hmassf': log mass fraction of hydrogen (no solar scaling)
        'mass': mass in g
        'density': density in g/cm**3
                
    table: {'PS20'}
        Which ionization tables to use.
    simtype: {'fire'}
        What format does the simulation reader class snap have?
    ps20depletion: bool
        Take away a fraction of the ions to account for the fraction of the
        parent element depleted onto dust.
    lintable: bool 
        interpolate the ion balance (and depletion, if applicable) in linear
        space (True), otherwise, it's done in log space (False) 
    ergs: bool
        output luminosity in erg/s[/cm**3] (True); 
        otherwise, output in photons/s[/cm**3] (False)
    density: bool
        output luminosity density ([erg or photons]/s/cm**3) (True);
        otherwise, output (total) luminosity ([erg or photons]/s) (False)

    Returns:
    --------
        the log luminosity (erg/s or photons/s, depending on ergs value)
        or log luminosity density (erg/s/cm**3 or photons/s/cm**3, depending
        on the ergs value), depending on the density value
        
    '''
    if simtype == 'fire':
        readfunc = snap.readarray_emulateEAGLE
        prepath = 'PartType0/'
        redshift = snap.cosmopars.z
    else:
        raise ValueError('invalid simtype option: {}'.format(simtype))
    
    # read in filter, any arrays already present
    if indct is None:
        indct = {}
    if 'filter' in indct: # gas selection, e.g. a spatial region
        filter = indct['filter']
    else:
        filter = slice(None, None, None)
    if 'logT' in indct: # should be in [log10 K]
        logT = indct['logT']
    else:
        logT = np.log10(readfunc(prepath + 'Temperature')[filter])
        tocgs = snap.toCGS
        if not np.isclose(tocgs, 1.):
            logT += np.log10(tocgs)
    if 'Hmassf' in indct:
        hmassf = indct['Hmassf']
    else:
        hmassf = readfunc(prepath + 'ElementAbundance/Hydrogen')[filter]
        hmassf_tocgs = snap.toCGS
    if 'lognH' in indct: # should be in log10 cm**-3
        lognH = indct['lognH']
    else:
        hdens = readfunc(prepath + 'Density')[filter]
        d_tocgs = snap.toCGS
        hdens *= hmassf
        hdens *= d_tocgs * hmassf_tocgs / (c.atomw_H * c.u)
        del hmassfrac
        lognH = np.log10(hdens)
        del hdens
    if table in ['PS20']:
        if 'logZ' in indct: # no solar normalization, 
            #just straight mass fraction
            logZ = indct['logZ'].copy()
        else:
            logZ = readfunc(prepath + 'Metallicity')[filter]    
            logZ = np.log10(logZ)
            tocgs = snap.toCGS
            if not np.isclose(tocgs, 1.):
                logZ += np.log10(tocgs)
        # interpolation needs finite values, float32(1e-100) == 0
        logZ[logZ == -np.inf] = -100.
    if table == 'PS20':
        interpdct = {'logT': logT, 'lognH': lognH, 'logZ': logZ}
        table = linetable_PS20(line, redshift, emission=False, vol=True,
                               ionbalfile=ol.iontab_sylvia_ssh, 
                               emtabfile=ol.emtab_sylvia_ssh, 
                               lintable=lintable)
        # log10 erg / s / cm**3 
        luminosity = table.find_logemission(interpdct)
        # luminosity in table = (1 - depletion) * luminosity_if_all_elt_in_gas
        # so divide by depleted fraction to get undepleted emission
        if not ps20depletion:
            luminosity -= \
                np.log10(1. - table.find_depletion(interpdct))
        
        # table values are for solar element ratios at Z
        # rescale to actual element density / hydrogen density
        parentelt = string.capwords(table.element)
        if parentelt == 'Hydrogen':
            del logZ
            del logT
            del lognH
        else:
            linelum_erg_invs_invcm3 -= \
                table.find_assumedabundance(interpdct, log=True)
            del logT
            del lognH
            del logZ

            if 'eltmassf' in indct:
                eltmassf = indct['eltmassf']
            else:
                readpath = prepath + 'ElementAbundance/' + parentelt
                eltmassf = readfunc(readpath)[filter]
            zscale = eltmassf / hmassf
            del eltmassf
            del hmassf
            zscale *= atomw_u_dct['Hydrogen'] / table.elementmass_u
            luminosity += np.log10(zscale)
        if not density:
            # log10 erg / s / cm**3 -> erg / s
            if 'mass' in indct:
                logmass = np.log10(indct['mass'])
                m_toCGS = 1.
            else:
                logmass = np.log10(readfunc(prepath + 'Mass')[filter])
                m_toCGS = snap.toCGS
                        # log10 erg / s / cm**3 -> erg/s
            if 'density' in indct:
                logdens = np.log10(indct['density'])
                d_toCGS = 1.
            else:
                logmass = np.log10(readfunc(prepath + 'Density')[filter])
                d_toCGS = snap.toCGS
            logvol = logmass - logdens
            del logmass
            del logdens
            v_toCGS = np.log10(m_toCGS / d_toCGS)
            if not np.isclose(v_toCGS, 0.):
                logvol += v_toCGS
            luminosity += logvol
            del logvol
        if not ergs:
            # erg -> photons
            wl = table.wavelength_cm
            erg_per_photon = c.planck * c.c / wl
            luminosity -= erg_per_photon         
    else:
        raise ValueError('invalid table option: {}'.format(table))
    return luminosity
    
def get_qty(snap, parttype, maptype, maptype_args, filterdct=None):
    '''
    calculate a quantity to map

    Parameters:
    -----------
    snap: Firesnap object (readin_fire_data.py)
        used to read in what is needed
    parttype: {0, 1, 4, 5}
        particle type
    maptype: {'Mass', 'Volume', 'Metal', 'ion', 'sim-direct'}
        what sort of thing are we looking for
    maptype_args: dict or None
        additional arguments for each maptype
        for maptype value:
        'Mass': None (ignored)
        'Metal': str
            number of nuclei or nucleus density (all ions together)
            'element': str
                element name, e.g. 'oxygen'
            'density': bool
                get the metal number density instead of number of nuclei.
                The default is False.
        'ion': str
            number of ions or ion density
            'ion': str
                ion name. format e.g. 'o6', 'fe17'
            'ionfrac-method': {'PS20', 'sim'}. The default is 'PS20'.
                how to calculate the ion fractions
                'PS20': interpolate the Ploeckinger & Schaye (2020) 
                        table
                'sim': read the ion fraction in from the snapshot
            'ps20depletion': bool
                deplete a fraction of the element onto dust and include
                that factor in the ion fraction. Depletion follows the
                Ploeckinger & Schaye (2020) table values.
                The default is False.
                (ignored unless the 'ps20table' calculation is used)
            'lintable': bool
                interpolate the tables in linear space (True) or log 
                space (False). The default is True.
                (ignored unless the 'ps20table' calculation is used)
            'density': bool
                get the ion density instead of number of nuclei.
                The default is False.
        'sim-direct': str
            a quantity stored directly in the simulation, or calculated
            by the simulation snapshot class (e.g., Temperature)
            'field': str
                the name of the field (after 'PartType<#>') to read 
                in from the simulation.

    Returns:
    --------
    qty: array
        the desired quantity for all the filtered resolution elements
    toCGS:
        factor to convert the array to CGS units
    todoc:
        dictonary with useful info to store   
        always contains a 'units' entry
        for sim-direct read-in, might just be 'cgs' though
    '''
    basepath = 'PartType{}/'.format(parttype)
    filter = slice(None, None, None)
    todoc = {}
    if filterdct is not None:
        if 'filter' in filterdct:
            filter = filterdct['filter']

    if maptype == 'Mass':
        qty = snap.readarray_emulateEAGLE(basepath + 'Masses')[filter]
        toCGS = snap.toCGS
        todoc['units'] = 'g'
    elif maptype == 'Volume':
        qty = snap.readarray_emulateEAGLE(basepath + 'Masses')[filter]
        toCGS = snap.toCGS
        qty /= snap.readarray_emulateEAGLE(basepath + 'Density')[filter]
        toCGS = toCGS / snap.toCGS
        todoc['units'] = 'cm**3'
        todoc['method'] = 'Masses / Density'
    elif maptype == 'Metal':
        element = maptype_args['element']
        if element == 'total':
            eltpath = basepath + 'Metallicity'
        else:
            eltpath = basepath + 'ElementAbundance/' + string.capwords(element)
        if 'density' in maptype_args:
            output_density = maptype_args['density']
        else:
            output_density = False
        qty = snap.readarray_emulateEAGLE(eltpath)[filter]
        toCGS = snap.toCGS
        if output_density:
            qty *= snap.readarray_emulateEAGLE(basepath + 'Density')[filter]
        else:
            qty *= snap.readarray_emulateEAGLE(basepath + 'Masses')[filter]
        toCGS = toCGS * snap.toCGS
        toCGS = toCGS / elt_atomw_cgs(element)
        todoc['units'] = '(# nuclei)'
        if output_density:
            todoc['units'] += ' * cm**-3'
        todoc['density'] = output_density
    elif maptype == 'ion':
        if parttype != 0 :
            msg = 'Can only calculate ion fractions for gas (PartType0),' + \
                   ' not particle type {}'
            raise ValueError(msg.format(parttype))
        ion = maptype_args['ion']
        if 'ionfrac-method' in maptype_args:
            ionfrac_method = maptype_args['ionfrac-method']
        else:
            ionfrac_method = 'PS20'
        if 'density' in maptype_args:
            output_density = maptype_args['density']
        else:
            output_density = False
        simtype = 'fire'
        if ionfrac_method == 'PS20':
            if 'ps20depletion' in maptype_args:
                ps20depletion = maptype_args['ps20depletion']
            else:
                ps20depletion = False
            if 'lintable' in maptype_args:
                lintable = maptype_args['lintable']
            else:
                lintable = True
            # no tables read in here, just an easy way to get parent 
            # element etc.
            dummytab = linetable_PS20(ion, snap.cosmopars.z, emission=False,
                                      vol=True, lintable=lintable)
            element = dummytab.element
            eltpath = basepath + 'ElementAbundance/' + string.capwords(element)
            qty = snap.readarray_emulateEAGLE(eltpath)[filter]
            toCGS = snap.toCGS
            if output_density:
                dpath = basepath + 'Density'
                qty *= snap.readarray_emulateEAGLE(dpath)[filter]
            else:
                mpath = basepath + 'Masses'
                qty *= snap.readarray_emulateEAGLE(mpath)[filter]
            toCGS =  toCGS * snap.toCGS
            ionfrac = get_ionfrac(snap, ion, indct=filterdct, 
                                  table=ionfrac_method, 
                                  simtype=simtype, ps20depletion=ps20depletion,
                                  lintable=lintable)
            qty *= ionfrac
            toCGS = toCGS / (dummytab.elementmass_u * c.u)
            todoc['table'] = dummytab.ionbalfile
            todoc['tableformat'] = ionfrac_method
            todoc['units'] = '(# ions)'
        if ionfrac_method == 'sim':
            if simtype == 'fire' and ion == 'H1':
                eltpath = basepath + 'ElementAbundance/Hydrogen'
                qty = snap.readarray_emulateEAGLE(eltpath)[filter]
                toCGS = snap.toCGS
                if output_density:
                    dpath = basepath + 'Density'
                    qty *= snap.readarray_emulateEAGLE(dpath)[filter]
                else:
                    mpath = basepath + 'Masses'
                    qty *= snap.readarray_emulateEAGLE(mpath)[filter]
                toCGS = toCGS * snap.toCGS
                hfpath = basepath + 'NeutralHydrogenAbundance'
                qty *= snap.readarray_emulateEAGLE(hfpath)[filter]
                toCGS = toCGS * snap.toCGS
                # just for the element mass
                dummytab = linetable_PS20(ion, snap.cosmopars.z, 
                                          emission=False,
                                          vol=True, lintable=True)
                toCGS = toCGS / (dummytab.elementmass_u * c.u)
                todoc['info'] = ('neutral H fraction from simulation'
                                 ' NeutralHydrogenAbundance')
                todoc['units'] = '(# ions)'
            else:    
                msg = ('simulation read-in of ion fractions is not available'
                       'for simulation {} and ion {}')
                raise ValueError(msg.format(simtype, ion))
        if output_density:
            todoc['units'] += ' * cm**-3'
        todoc['density'] = output_density
        todoc['ionfrac-method'] = ionfrac_method
    elif maptype == 'sim-direct':
        field = maptype_args['field']
        qty = snap.readarray_emulateEAGLE(basepath + field)[filter]
        toCGS = snap.toCGS
        todoc['units'] = '(cgs {})'.format(field)
        if field == 'Pressure':
            todoc['info'] = 'thermal pressure only'
    else:
        raise ValueError('Invalid maptype: {}'.format(maptype))
    return qty, toCGS, todoc

# AHF: sorta tested (enclosed 2D mass wasn't too far above Mvir)
# Rockstar: untested draft
# shrinking spheres: sort of tested (maps look right)
# mass maps: look ok
# ion/metal maps: tested sum of ions
def massmap(dirpath, snapnum, radius_rvir=2., particle_type=0,
            pixsize_pkpc=3., axis='z', outfilen=None,
            center='shrinksph', norm='pixsize_phys',
            maptype='Mass', maptype_args=None):
    '''
    Creates a mass map projected perpendicular to a line of sight axis
    by assuming the simulation resolution elements divide their mass 
    following a C2 SPH kernel.

    Parameters:
    -----------
    snapfile: str
        file (or example file, if split) containing the snapshot data
    dirpath: str
        path to the directory containing the 'output' directory with the
        snapshots
    snapnum: int
        snapshot number
    radius_rvir: float 
        radius of the cube to project in units of Rvir. Note that in the sky 
        plane, this will be (slightly) extended to get the exact pixel size.
    particle_type: int
        particle type to project (follows FIRE format)
    pixsize_pkpc: float
        size of the map pixels in proper kpc
    axis: str, 'x', 'y', or 'z'
        axis corresponding to the line of sight 
    outfilen: str or None. 
        if a string, the name of the file to save the output data to. The
        default is None, meaning the maps are returned as output
    center: str
        how to find the halo center.
        'AHFsmooth': use halo_00000_smooth.dat from AHF 
        'rockstar-maxmass': highest mass halo at snapshot from Rockstar
        'rockstar-mainprog': main progenitor of most massive halo at
                           final snapshot from Rockstar
        'rockstar-<int>': halo with snapshot halo catalogue index <int>
                          from Rockstar 
        'shrinksph': Imran's shrinking spheres method
    norm: {'pixsize_phys'}
        how to normalize the column values 
        'pixsize_phys': [quantity] / cm**2
    maptype: {'Mass', 'Metal', 'ion'}
        what sort of thing to map
        'Mass' -> g
        'Metal' -> number of nuclei of the selected element
        'ion' -> number of ions of the selected type
    maptype_args: dict or None
        see get_qty for parameters; options depend on maptype
    Output:
    -------
    massW: 2D array of floats
        projected mass image [log g/cm^-2]
    massQ: NaN array, for future work


    '''
    if axis == 'z':
        Axis1 = 0
        Axis2 = 1
        Axis3 = 2
    elif axis == 'x':
        Axis1 = 2
        Axis2 = 0
        Axis3 = 1
    elif axis == 'y':
        Axis1 = 1
        Axis2 = 2
        Axis3 = 0
    else:
        msg = 'axis should be "x", "y", or "z", not {}'
        raise ValueError(msg.format(axis))
    
    if center == 'AHFsmooth':
        halodat = mainhalodata_AHFsmooth(dirpath, snapnum)
        snap = rf.get_Firesnap(dirpath, snapnum) 
        cen = np.array([halodat['Xc_ckpcoverh'], 
                        halodat['Yc_ckpcoverh'], 
                        halodat['Zc_ckpcoverh']])
        cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc \
                 / snap.cosmopars.h
        rvir_cm = halodat['Rvir_ckpcoverh'] * snap.cosmopars.a \
                  * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
    elif center.startswith('rockstar'):
        select = center.split('-')[-1]
        if select not in ['maxmass', 'mainprog']:
            try:
                select = int(select)
            except ValueError:
                msg = 'invalid option for center: {}'.format(center)
                raise ValueError(msg)
        halodat, _csm_halo = halodata_rockstar(dirpath, snapnum, 
                                               select=select)
        snap = rf.get_Firesnap(dirpath, snapnum) 
        cen = np.array([halodat['Xc_ckpc'], 
                        halodat['Yc_ckpc'], 
                        halodat['Zc_ckpc']])
        cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc 
        rvir_cm = halodat['Rvir_cm'] 
    elif center ==  'shrinksph':
        halodat, _ = gethalodata_shrinkingsphere(dirpath, snapnum, 
                                                 meandef='BN98')
        cen_cm = np.array([halodat['Xc_cm'], 
                           halodat['Yc_cm'], 
                           halodat['Zc_cm']])
        rvir_cm = halodat['Rvir_cm']
        snap = rf.get_Firesnap(dirpath, snapnum) 
    else:
        raise ValueError('Invalid center option {}'.format(center))

    # calculate pixel numbers and projection region based
    # on target size and extended for integer pixel number
    target_size_cm = np.array([2. * radius_rvir * rvir_cm] * 3)
    pixel_cm = pixsize_pkpc * c.cm_per_mpc * 1e-3
    npix3 = (np.ceil(target_size_cm / pixel_cm)).astype(int)
    npix_x = npix3[Axis1]
    npix_y = npix3[Axis2]
    size_touse_cm = target_size_cm
    size_touse_cm[Axis1] = npix_x * pixel_cm
    size_touse_cm[Axis2] = npix_y * pixel_cm

    if norm == 'pixsize_phys':
        multipafter = 1. / pixel_cm**2
        norm_units = ' / (physical cm)**2'
    else:
        raise ValueError('Invalid norm option {}'.format(norm))

    basepath = 'PartType{}/'.format(particle_type)
    haslsmooth = particle_type == 0
    if haslsmooth: # gas
        lsmooth = snap.readarray_emulateEAGLE(basepath + 'SmoothingLength')
        lsmooth_toCGS = snap.toCGS

    coords = snap.readarray_emulateEAGLE(basepath + 'Coordinates')
    coords_toCGS = snap.toCGS
    # needed for projection step anyway
    coords -= cen_cm / coords_toCGS
    # select box region
    # zoom regions are generally centered -> don't worry
    # about edge overlap
    box_dims_coordunit = size_touse_cm / coords_toCGS

    if haslsmooth:
        # extreme values will occur at zoom region edges -> restrict
        filter_temp = np.all(np.abs((coords)) <= 0.5 * box_dims_coordunit, 
                             axis=1)
        lmax = np.max(lsmooth[filter_temp]) 
        conv = lsmooth_toCGS / coords_toCGS
        del filter_temp
        # might be lower-density stuff outside the region, but overlapping it
        lmargin = 2. * lmax * conv
        filter = np.all(np.abs((coords)) <= 0.5 * box_dims_coordunit \
                        + lmargin, axis=1)
        lsmooth = lsmooth[filter]
        if not np.isclose(conv, 1.):
            lsmooth *= conv
    
    else:
        filter = np.all(np.abs((coords)) <= 0.5 * box_dims_coordunit, axis=1)   
    
    coords = coords[filter]
    qW, toCGS, todoc = get_qty(snap, particle_type, maptype, maptype_args, 
                               filterdct={'filter': filter})
    multipafter *= toCGS
    ## debugging: check for NaN values
    #naninds = np.where(np.isnan(qW))[0]
    #if len(naninds) > 0:
    #    print('Some qW values are NaN')
    #    print('Used {}, {}, {}'.format(particle_type, maptype, maptype_args))
    #    outfile_debug = outfilen.split('/')[:-1]
    #    outfile_debug.append('debug_qW_naninfo.hdf5')
    #    outfile_debug = '/'.join(outfile_debug)
    #
    #    numnan = len(naninds)
    #    minW = np.min(qW[np.isfinite(qW)])
    #    maxW = np.max(qW[np.isfinite(qW)])
    #    with h5py.File(outfile_debug, 'w') as f:
    #        hed = f.create_group('Header')
    #        hed.attrs.create('number of qW values', len(qW))
    #        hed.attrs.create('number of NaN values', numnan)
    #        hed.attrs.create('number of inf values', np.sum(qW == np.inf))
    #        hed.attrs.create('number of -inf values', np.sum(qW == -np.inf))
    #        hed.attrs.create('number of 0 values', np.sum(qW == 0))
    #        hed.attrs.create('number of values < 0', np.sum(qW < 0))
    #        hed.attrs.create('number of values > 0', np.sum(qW > 0))
    #        hed.attrs.create('qW_toCGS', toCGS)
    #        hed.attrs.create('multipafter', multipafter)
    #
    #        if minW > 0:
    #            bins = np.logspace(np.log10(minW), np.log10(maxW), 100)
    #        else:
    #            bins = np.linspace(minW, maxW, 100)
    #        # NaN, inf, -inf values just aren't counted
    #        hist, _ = np.histogram(qW, bins=bins)
    #
    #        f.create_dataset('qW_hist', data=hist)
    #        f.create_dataset('qW_hist_bins', data=bins)
    #
    #        # issues were for ion columns; check rho, T, Z of NaN values
    #        _filter = filter.copy()
    #        _filter[_filter] = np.isnan(qW)
    #
    #        _temp, _temp_toCGS, _temp_todoc = get_qty(snap, particle_type, 
    #                'sim-direct', {'field': 'Temperature'}, 
    #                filterdct={'filter': _filter})
    #        ds = f.create_dataset('Temperature_nanqW', data=_temp)
    #        ds.attrs.create('toCGS', _temp_toCGS)
    #        print('Temperature: ', _temp_todoc)
    #
    #        _dens, _dens_toCGS, _dens_todoc = get_qty(snap, particle_type, 
    #                'sim-direct', {'field': 'Density'}, 
    #                filterdct={'filter': _filter})
    #        ds = f.create_dataset('Density_nanqW', data=_dens)
    #        ds.attrs.create('toCGS', _dens_toCGS)
    #        print('Density: ', _dens_todoc)
    #
    #        _hden, _hden_toCGS, _emet_todoc = get_qty(snap, particle_type, 
    #                'Metal', {'element': 'Hydrogen', 'density': True}, 
    #                filterdct={'filter': _filter})
    #        ds = f.create_dataset('nH_nanqW', data=_hden)
    #        ds.attrs.create('toCGS', _hden_toCGS)
    #        print('Hydrogen number density: ', _emet_todoc)
    #
    #        if maptype == 'ion':
    #            ion = maptype_args['ion']
    #            dummytab = linetable_PS20(ion, snap.cosmopars.z, 
    #                                      emission=False,
    #                                      vol=True, lintable=True)
    #            element = dummytab.element
    #            eltpath = basepath + 'ElementAbundance/' +\
    #                      string.capwords(element)
    #            _emet, _emet_toCGS, _emet_todoc = get_qty(snap, particle_type, 
    #                    'sim-direct', {'field': eltpath}, 
    #                    filterdct={'filter': _filter})
    #            ds = f.create_dataset('massfrac_{}_nanqW', data=_emet)
    #            ds.attrs.create('toCGS', _emet_toCGS)
    #            print(f'{element} mass fraction: ', _emet_todoc)
    #else:
    #    print('No NaN values in qW')
    # stars, black holes. DM: should do neighbour finding. Won't though.
    if not haslsmooth:
        # minimum smoothing length is set in the projection
        lsmooth = np.zeros(shape=(len(qW),), dtype=coords.dtype)
        lsmooth_toCGS = 1.
    
    tree = False
    periodic = False # zoom region
    NumPart = len(qW)
    dct = {'coords': coords, 'lsmooth': lsmooth, 
           'qW': qW, 
           'qQ': np.zeros(len(qW), dtype=np.float32)}
    Ls = box_dims_coordunit
    # cosmopars uses EAGLE-style cMpc/h units for the box
    box3 = [snap.cosmopars.boxsize * c.cm_per_mpc / snap.cosmopars.h \
            / coords_toCGS] * 3
    mapW, mapQ = project(NumPart, Ls, Axis1, Axis2, Axis3, box3,
                         periodic, npix_x, npix_y,
                         'C2', dct, tree, ompproj=True, 
                         projmin=None, projmax=None)
    lmapW = np.log10(mapW)
    ## debug NaN values in maps
    #if np.any(np.isnan(mapW)):
    #    print('NaN values in mapW after projection')
    #if np.any(mapW < 0.):
    #    print('values < 0 in mapW after projection')
    #if np.any(np.isnan(lmapW)):
    #    print('NaN values in log mapW before multipafter')

    lmapW += np.log10(multipafter)

    #if np.any(np.isnan(lmapW)):
    #    print('NaN values in log mapW after multipafter')
    #if outfilen is None:
    #    return lmapW, mapQ
    
    with h5py.File(outfilen, 'w') as f:
        # map (emulate make_maps format)
        f.create_dataset('map', data=lmapW)
        f['map'].attrs.create('log', True)
        minfinite = np.min(lmapW[np.isfinite(lmapW)])
        f['map'].attrs.create('minfinite', minfinite)
        f['map'].attrs.create('max', np.max(lmapW))
        
        # cosmopars (emulate make_maps format)
        hed = f.create_group('Header')
        cgrp = hed.create_group('inputpars/cosmopars')
        csm = snap.cosmopars.getdct()
        for key in csm:
            cgrp.attrs.create(key, csm[key])
        
        # direct input parameters
        igrp = hed['inputpars']
        igrp.attrs.create('snapfiles', np.array([np.string_(fn) for fn in snap.filens]))
        igrp.attrs.create('dirpath', np.string_(dirpath))
        igrp.attrs.create('radius_rvir', radius_rvir)
        igrp.attrs.create('particle_type', particle_type)
        igrp.attrs.create('pixsize_pkpc', pixsize_pkpc)
        igrp.attrs.create('axis', np.string_(axis))
        igrp.attrs.create('norm', np.string_(norm))
        igrp.attrs.create('outfilen', np.string_(outfilen))
        # useful derived/used stuff
        igrp.attrs.create('Axis1', Axis1)
        igrp.attrs.create('Axis2', Axis2)
        igrp.attrs.create('Axis3', Axis3)
        igrp.attrs.create('diameter_used_cm', np.array(size_touse_cm))
        if haslsmooth:
            igrp.attrs.create('margin_lsmooth_cm', lmargin * coords_toCGS)
        igrp.attrs.create('center', np.string_(center))
        _grp = igrp.create_group('halodata')
        for key in halodat:
            _grp.attrs.create(key, halodat[key])
        igrp.attrs.create('maptype', np.string_(maptype))
        if maptype_args is None:
            igrp.attrs.create('maptype_args', np.string_('None'))
        else:
            igrp.attrs.create('maptype_args', np.string_('dict'))
            _grp = igrp.create_group('maptype_args_dict')
            for key in maptype_args:
                val = maptype_args[key]
                if isinstance(val, type('')):
                    val = np.string_(val)
                _grp.attrs.create(key, val)
        for key in todoc:
            if key == 'units':
                val = todoc[key]
                val = val + norm_units
            else:
                val = todoc[key]
            if isinstance(val, type('')):
                val = np.string_(val)
            igrp.attrs.create(key, val)

def massmap_wholezoom(dirpath, snapnum, pixsize_pkpc=3.,
                      outfilen_DM='map_DM_{ax}-axis.hdf5',
                      outfilen_gas='map_gas_{ax}-axis.hdf5',
                      outfilen_stars='map_stars_{ax}-axis.hdf5',
                      outfilen_BH='map_BH_{ax}-axis.hdf5'):
    '''
    for debugging: make a mass map of basically the whole zoom region
    (for centering tests)
    '''
    parttype_outfilen = {0: outfilen_gas,
                         1: outfilen_DM,
                         4: outfilen_stars,
                         5: outfilen_BH}
    snap = rf.get_Firesnap(dirpath, snapnum)
    coords = {}
    mass = {}
    lsmooth = {}
    masspath = 'PartType{}/Masses'
    coordpath = 'PartType{}/Coordinates'
    lsmoothpath = 'PartType{}/SmoothingLength'
    coords_toCGS = None
    lsmooth_toCGS = None
    mass_toCGS = None
    maxl = -np.inf
    coordsbox = np.array([[np.inf, -np.inf]] * 3) 
    for pt in parttype_outfilen:
        try:
            coords[pt] = snap.readarray(coordpath.format(pt))
            _toCGS = snap.toCGS
            if coords_toCGS is None:
                coords_toCGS = _toCGS
            elif coords_toCGS != _toCGS:
                msg = 'Different particle types have different coordinate'+\
                      ' toCGS: {}, {}'
                raise RuntimeError(msg.format(coords_toCGS, _toCGS))
        except rf.FieldNotFoundError as err:
            print('PartType {} not found'.format(pt))
            print(err)
            continue
        mass[pt] = snap.readarray(masspath.format(pt))
        _toCGS = snap.toCGS
        if mass_toCGS is None:
            mass_toCGS = _toCGS
        elif mass_toCGS != _toCGS:
            msg = 'Different particle types have different mass'+\
                    ' toCGS: {}, {}'
            raise RuntimeError(msg.format(mass_toCGS, _toCGS))
        coordsbox[:, 0] = np.min([coordsbox[:, 0], 
                                  np.min(coords[pt], axis=0)], 
                                 axis=0)
        coordsbox[:, -1] = np.max([coordsbox[:, -1], 
                                  np.max(coords[pt], axis=0)], 
                                 axis=0)
        if pt == 0:
            lsmooth[pt] = snap.readarray(lsmoothpath.format(pt))
            _toCGS = snap.toCGS
            if lsmooth_toCGS is None:
                lsmooth_toCGS = _toCGS
            elif lsmooth_toCGS != _toCGS:
                msg = 'Different particle types have different smoothing '+\
                      'length toCGS: {}, {}'
                raise RuntimeError(msg.format(lsmooth_toCGS, _toCGS))
            maxl = max(maxl, np.max(lsmooth[pt]))
    coordsbox[:, 0] -= maxl * lsmooth_toCGS / coords_toCGS
    coordsbox[:, -1] += maxl * lsmooth_toCGS / coords_toCGS
    print('coordsbox before pixel adjustments: ', coordsbox)

    target_size_cm = (coordsbox[:, -1] - coordsbox[:, 0]) * coords_toCGS
    pixel_cm = pixsize_pkpc * c.cm_per_mpc * 1e-3
    npix3 = (np.ceil(target_size_cm / pixel_cm)).astype(int)
    center = 0.5 * np.sum(coordsbox, axis=1)
    Ls = npix3 * pixel_cm / coords_toCGS
    coordsbox = center[:, np.newaxis] \
                + Ls[:, np.newaxis] * np.array([-0.5, + 0.5])[np.newaxis, :]
    print('coordsbox: ', coordsbox)

    multipafter = mass_toCGS / pixel_cm**2
    units = 'g / (physical cm)**2'
    
    for pt in coords:
        print('Running particle type ', pt)
        qW = mass[pt]
        if pt in lsmooth:
            _lsmooth = lsmooth[pt]
            _lsmooth_toCGS = lsmooth_toCGS
        else:
            # minimum smoothing length is set in the projection
            _lsmooth = np.zeros(shape=(len(qW),), dtype=(coords[pt]).dtype)
            _lsmooth_toCGS = 1.
        tree = False
        periodic = False # zoom region
        NumPart = len(qW)
        dct = {'coords': coords[pt] - center, 'lsmooth': _lsmooth, 
               'qW': qW, 
               'qQ': np.zeros(len(qW), dtype=np.float32)}
        print('Extent of coordinates: ',
              np.min(coords[pt], axis=0), 
              ', ',
              np.max(coords[pt], axis=0))
        print('Extent of centered coordinates: ',
              np.min(dct['coords'], axis=0), 
              ', ',
              np.max(dct['coords'], axis=0))
        print('Ls: ', Ls)
        print('Coordinates in box: ', 
              np.sum(np.all(np.abs(dct['coords']) < Ls, axis=1)),
              ' / ', len(dct['coords']))

        # cosmopars uses EAGLE-style cMpc/h units for the box
        box3 = [snap.cosmopars.boxsize * c.cm_per_mpc / snap.cosmopars.h \
                / coords_toCGS] * 3
        for axis in ['x', 'y', 'z']:
            if axis == 'z':
                Axis1 = 0
                Axis2 = 1
                Axis3 = 2
            elif axis == 'x':
                Axis1 = 2
                Axis2 = 0
                Axis3 = 1
            elif axis == 'y':
                Axis1 = 1
                Axis2 = 2
                Axis3 = 0
            npix_x = npix3[Axis1]
            npix_y = npix3[Axis2]

            mapW, mapQ = project(NumPart, Ls, Axis1, Axis2, Axis3, box3,
                                 periodic, npix_x, npix_y,
                                 'C2', dct, tree, ompproj=True, 
                                 projmin=None, projmax=None)
            lmapW = np.log10(mapW)
            lmapW += np.log10(multipafter)
        
            outfilen = (parttype_outfilen[pt]).format(ax=axis)
    
            with h5py.File(outfilen, 'w') as f:
                # map (emulate make_maps format)
                f.create_dataset('map', data=lmapW)
                f['map'].attrs.create('log', True)
                minfinite = np.min(lmapW[np.isfinite(lmapW)])
                f['map'].attrs.create('minfinite', minfinite)
                f['map'].attrs.create('max', np.max(lmapW))
            
                # cosmopars (emulate make_maps format)
                hed = f.create_group('Header')
                cgrp = hed.create_group('inputpars/cosmopars')
                csm = snap.cosmopars.getdct()
                for key in csm:
                    cgrp.attrs.create(key, csm[key])
        
                # direct input parameters
                igrp = hed['inputpars']
                igrp.attrs.create('snapfiles', np.array([np.string_(fn) \
                                  for fn in snap.filens]))
                igrp.attrs.create('dirpath', np.string_(dirpath))
            
                igrp.attrs.create('particle_type', pt)
                igrp.attrs.create('pixsize_pkpc', pixsize_pkpc)
                igrp.attrs.create('axis', np.string_(axis))
                igrp.attrs.create('units', np.string_(units))
                igrp.attrs.create('outfilen', np.string_(outfilen))
                # useful derived/used stuff
                igrp.attrs.create('Axis1', Axis1)
                igrp.attrs.create('Axis2', Axis2)
                igrp.attrs.create('Axis3', Axis3)
                igrp.attrs.create('maptype', np.string_('Mass'))
                igrp.attrs.create('mapped_region_cm', coordsbox)
                igrp.attrs.create('maptype_args', np.string_('None'))
                # useful for plotting centers in sim units
                igrp.attrs.create('coords_toCGS', coords_toCGS)

def getaxbins(minfinite, maxfinite, bin, extendmin=True, extendmax=True):
    if isinstance(bin, int):
        bins = np.linspace(minfinite, maxfinite, bin + 1)
    elif isinstance(bin, float):
        minbin = np.floor(minfinite / bin) * bin
        maxbin = np.ceil(maxfinite / bin) * bin
        bins = np.arange(minbin, maxbin + 0.5 * bin, bin)
    else:
        bins = np.array(bin)
        if minfinite < bins[0]:
            extendmin = True
        if maxfinite >= bins[1]:
            extendmax = True
    if extendmin:
        bins = np.append(-np.inf, bins)
    if extendmax:
        bins = np.append(bins, np.inf)
    return bins


def histogram_radprof(dirpath, snapnum,
                      weighttype, weighttype_args, axtypes, axtypes_args,
                      particle_type=0, 
                      center='shrinksph', rbins=(0., 1.), runit='Rvir',
                      logweights=True, logaxes=True, axbins=0.1,
                      outfilen=None, overwrite=True):
    '''
    make a weightype, weighttype_args weighted histogram of 
    axtypes, axtypes_args.

    Parameters:
    -----------
    dirpath: str
        path to the directory containing the 'output' directory with the
        snapshots
    snapnum: int
        snapshot number
    weightype: float
        what to weight the histogram by. Options are maptype options in 
        get_qty
    weighttype_args: dict
        additional arguments for what to weight the histogram by. Options 
        are maptype_args options in get_qty
    axtypes: list
        list of what to histogram; each entry is one histogram dimension.
        Options are maptype options in get_qty.
    axtypes_args: list of dicts
        list of additional arguments for the histogram dimensions. Options 
        are maptype_args options in get_qty. Mind the 'density' option for
        ions and metals. These are matched to axtypes by list index.
    particle_type: int
        particle type to project (follows FIRE format)
    center: str
        how to find the halo center.
        'AHFsmooth': use halo_00000_smooth.dat from AHF 
        'rockstar-maxmass': highest mass halo at snapshot from Rockstar
        'rockstar-mainprog': main progenitor of most massive halo at
                           final snapshot from Rockstar
        'rockstar-<int>': halo with snapshot halo catalogue index <int>
                          from Rockstar 
        'shrinksph': Imran's shrinking spheres method
    rbins: array-like of floats
        bin edges in 3D distance from the halo center. Ignored if center is 
        None.
    runit: {'Rvir', 'pkpc'}
        unit to use for the rbins. These bins are never log values. 
    logweights: bool
        save log of the weight sum in each bin instead of the linear sum.
    logaxes: bool or list of bools
        save and process the histogram axis quantities in log units instead
        of linear. If a list, this is applied to each dimension by matching
        list index.
    axbins: int, float, array, or list of those.
        int: use that many bins between whatever min and maxfinite values are
             present in the data
        float: use bins of that size, in (log) cgs units. The edges are chosen
             so zero is an edge if the value range includes zero.
             A useful option to allow stacking/comparison without using too 
             much storage.
        array: just the bin edges directly. Monotonically increasing, (log) 
             cgs units.
        Note that the range are always extended with -np.inf and np.inf if the
        values include non-finite ones or values outside the specified range.
        If a list is given, the options are specified per histogram axis, 
        matched by list index. Note that if giving an array option, it should 
        be enclosed in a list to ensure it is not interpreted as a per-axis
        option list.
        Units are always (log) cgs. 
    outfilen: str
        file to save to output histogram to. None means no file is saved.
        The file must include the full path.
    overwrite: bool
        If a file with name outfilen already exists, overwrite it (True) or
        raise a ValueError (False)
    Output:
    -------
    file with saved histogram data, if a file is specified
    otherwise, the histogram and bins

    '''
    if outfilen is not None:
        if os.path.isfile(outfilen) and not overwrite:
            raise ValueError('File {} already exists.'.format(outfilen))

    todoc_gen = {}
    basepath = 'PartType{}/'.format(particle_type)
    _axvals = []
    _axbins = []
    _axdoc = []
    _axbins_outunit = []
    _logaxes = []
    _axtypes = []
    _axtypes_args = []
    if not hasattr(axbins, '__len__'):
        axbins = [axbins] * len(axtypes)
    if not hasattr(logaxes, '__len__'):
        logaxes = [logaxes] * len(axtypes)

    snap = rf.get_Firesnap(dirpath, snapnum)
    
    if center is not None:
        todoc_cen = {}
        todoc_cen['center_method'] = center
        if center == 'AHFsmooth':
            halodat = mainhalodata_AHFsmooth(dirpath, snapnum)
            cen = np.array([halodat['Xc_ckpcoverh'], 
                            halodat['Yc_ckpcoverh'], 
                            halodat['Zc_ckpcoverh']])
            cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc \
                    / snap.cosmopars.h
            rvir_cm = halodat['Rvir_ckpcoverh'] * snap.cosmopars.a \
                    * 1e-3 * c.cm_per_mpc / snap.cosmopars.h
        elif center.startswith('rockstar'):
            select = center.split('-')[-1]
            if select not in ['maxmass', 'mainprog']:
                try:
                    select = int(select)
                except ValueError:
                    msg = 'invalid option for center: {}'.format(center)
                    raise ValueError(msg)
            halodat, _csm_halo = halodata_rockstar(dirpath, snapnum, 
                                                select=select)
            cen = np.array([halodat['Xc_ckpc'], 
                            halodat['Yc_ckpc'], 
                            halodat['Zc_ckpc']])
            cen_cm = cen * snap.cosmopars.a * 1e-3 * c.cm_per_mpc 
            rvir_cm = halodat['Rvir_cm'] 
        elif center ==  'shrinksph':
            halodat, _ = gethalodata_shrinkingsphere(dirpath, snapnum, 
                                                     meandef='BN98')
            cen_cm = np.array([halodat['Xc_cm'], 
                               halodat['Yc_cm'], 
                               halodat['Zc_cm']])
            rvir_cm = halodat['Rvir_cm']
            todoc_cen['Rvir_def'] = 'BN98'
        else:
            raise ValueError('Invalid center option {}'.format(center))
        todoc_cen['center_cm'] = cen_cm
        todoc_cen['Rvir_cm'] = rvir_cm
        todoc_cen['units'] = runit
        todoc_cen['log'] = False

        coords = snap.readarray_emulateEAGLE(basepath + 'Coordinates')
        coords_toCGS = snap.toCGS
        coords -= cen_cm / coords_toCGS
        
        if runit == 'Rvir':
            rbins_simu = np.array(rbins) * rvir_cm / coords_toCGS
            simu_to_runit = coords_toCGS / rvir_cm 
        elif runit == 'pkpc':
            rbins_simu = np.array(rbins) * c.cm_per_mpc * 1e-3 / coords_toCGS
            simu_to_runit = coords_toCGS / (c.cm_per_mpc * 1e-3)
        else:
            raise ValueError('Invalid runit option: {}'.format(runit))
        rbins2_simu = rbins_simu**2
        r2vals = np.sum(coords**2, axis=1)
        del coords
        filter = r2vals <= rbins2_simu[-1]
        r2vals = r2vals[filter]

        _axvals.append(r2vals)
        _axbins.append(rbins2_simu)
        _axdoc.append(todoc_cen)
        _axbins_outunit.append(np.sqrt(rbins2_simu) * simu_to_runit)
        _logaxes.append(False)
        _axtypes.append('halo_3Dradius')
        _axtypes_args.append({})
        filterdct = {'filter': filter}
    else:
        todoc_gen['info_halo'] = 'no halo particle selection applied'
        filterdct = {'filter': slice(None, None, None)}
        halodat = None
    
    for axt, axarg, logax, axb in zip(axtypes, axtypes_args, logaxes, axbins):
        qty, toCGS, todoc = get_qty(snap, particle_type, axt, axarg, 
                                    filterdct=filterdct)
        if logax:
            qty = np.log10(qty)
        qty_good = np.isfinite(qty)
        minq = np.min(qty[qty_good])
        maxq = np.max(qty[qty_good])
        needext = not np.all(qty_good)
        if hasattr(axb, '__len__'):
            if logax:
                _axb = axb - np.log10(toCGS)
            else:
                _axb = axb / toCGS
        else:
            _axb = axb
        usebins_simu = getaxbins(minq, maxq, _axb, extendmin=needext, 
                                 extendmax=needext)
        
        _axvals.append(qty)
        _axbins.append(usebins_simu)
        _axdoc.append(todoc)
        _logaxes.append(logax)
        _axtypes.append(axt)
        _axtypes_args.append(axarg)
        if logax:
            _bins_doc = usebins_simu + np.log10(toCGS)
        else:
            _bins_doc = usebins_simu * toCGS
        _axbins_outunit.append(_bins_doc)
    
    wt, wt_toCGS, wt_todoc = get_qty(snap, particle_type, weighttype, 
                                     weighttype_args, 
                                     filterdct=filterdct)
    #print(_axbins)
    maxperloop = 752**3 // 8
    if len(wt) <= maxperloop:
        hist, edges = np.histogramdd(_axvals, weights=wt, bins=_axbins)
    else:
        lentot = len(wt)
        numperloop = maxperloop
        slices = [slice(i * numperloop, 
                        min((i + 1) * numperloop, lentot), 
                        None) \
                  for i in range((lentot - 1) // numperloop + 1)]
        for slind in range(len(slices)):
            axdata_temp = [data[slices[slind]] for data in _axvals]
            hist_temp, edges_temp = np.histogramdd(axdata_temp, 
                                                   weights=wt[slices[slind]], 
                                                   bins=_axbins)
            if slind == 0 :
                hist = hist_temp
                edges = edges_temp
            else:
                hist += hist_temp
                if not np.all(np.array([np.all(edges[i] == edges_temp[i]) \
                              for i in range(len(edges))])):
                    msg = 'Error: edges mismatch in histogramming'+\
                          ' loop (slind = {})'.format(slind)
                    raise RuntimeError(msg)
    if logweights:
        hist = np.log10(hist)
        hist += np.log10(wt_toCGS)
    else:
        hist *= wt_toCGS

    if outfilen is not None:
        with h5py.File(outfilen, 'w') as f:
            # cosmopars (emulate make_maps format)
            hed = f.create_group('Header')
            cgrp = hed.create_group('cosmopars')
            csm = snap.cosmopars.getdct()
            for key in csm:
                cgrp.attrs.create(key, csm[key])

            # histogram and weight
            hgrp = f.create_group('histogram')
            hgrp.create_dataset('histogram', data=hist)
            hgrp.attrs.create('log', logweights)
            for key in wt_todoc:
                val = wt_todoc[key]
                if isinstance(val, type('')):
                    val = np.string_(val)
                if val is None:
                    val = np.string_(val)
                hgrp.attrs.create(key, val)
            hgrp.attrs.create('weight_type', np.string_(weighttype))
            wagrp = hgrp.create_group('weight_type_args')
            for key in weighttype_args:
                val = weighttype_args[key]
                if isinstance(val, type('')):
                    val = np.string_(val)
                if val is None:
                    val = np.string_(val)
                wagrp.attrs.create(key, val)
            
            # histogram axes
            for i in range(0, len(_axbins)):
                agrp = f.create_group('axis_{}'.format(i))
                _bins = _axbins_outunit[i]
                agrp.create_dataset('bins', data=_bins)
                agrp.attrs.create('log', _logaxes[i])
                if center is None:
                    agrp.attrs.create('bin_input', axbins[i])
                elif i == 0:
                    agrp.attrs.create('bin_input', rbins)
                else:
                    agrp.attrs.create('bin_input', axbins[i - 1])
                _todoc = _axdoc[i]
                for key in _todoc:
                    val = _todoc[key]
                    if isinstance(val, type('')):
                        val = np.string_(val)
                    if val is None:
                        val = np.string_(val)
                    agrp.attrs.create(key, val)
                agrp.attrs.create('qty_type', np.string_(_axtypes[i]))
                aagrp = agrp.create_group('qty_type_args')
                for key in _axtypes_args[i]:
                    val = _axtypes_args[i][key]
                    if isinstance(val, type('')):
                        val = np.string_(val)
                    if val is None:
                        val = np.string_(val)
                    aagrp.attrs.create(key, val)
            
            # direct input parameters
            igrp = hed.create_group('inputpars')
            _snf =  np.array([np.string_(fn) for fn in snap.filens])
            igrp.attrs.create('snapfiles', _snf)
            igrp.attrs.create('dirpath', np.string_(dirpath))
            igrp.attrs.create('particle_type', particle_type)
            igrp.attrs.create('outfilen', np.string_(outfilen))
            
            if halodat is not None:
                _grp = igrp.create_group('halodata')
                for key in halodat:
                    _grp.attrs.create(key, halodat[key])

            for key in todoc_gen:
                val = todoc_gen[key]
                if isinstance(val, type('')):
                    val = np.string_(val)
                if val is None:
                    val = np.string_(val)
                igrp.attrs.create(key, val)

# hard to do a true test, but check that projected masses and centering
# sort of make sense
def tryout_massmap(opt=1, center='AHFsmooth'):
    outdir = 'ls'
    _outfilen = 'mass_pt{pt}_{sc}_snap{sn}_ahf-cen_2rvir_v1.hdf5'
    if opt == 1:
        parttypes = [0, 1, 4]
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        simcode = 'metal-diffusion-m12i-res7100'
        snapnum = 600
    elif opt == 2:
        parttypes = [0, 1, 4]
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        simcode = 'metal-diffusion-m12i-res7100'
        snapnum = 399
    elif opt == 3:
        parttypes = [0, 1, 4]
        dirpath = '/projects/b1026/snapshots/metal_diffusion/m12i_res7100/'
        simcode = 'metal-diffusion-m12i-res7100'
        snapnum = 196

    for pt in parttypes:
        outfilen = outdir + _outfilen.format(pt=pt, sc=simcode, 
                                             sn=snapnum)
        massmap(dirpath, snapnum, radius_rvir=2., particle_type=pt,
                pixsize_pkpc=3., axis='z', outfilen=outfilen,
                center=center)

def tryout_ionmap(opt=1):
    outdir = '/projects/b1026/nastasha/tests/start_fire/map_tests/'
    dirpath1 = '/projects/b1026/snapshots/fire3/m13h206_m3e5/' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000/'
    simname1 = 'm13h206_m3e5__' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    # version number depends on code edits; some indices might have
    # been run with previous versions
    _outfilen = 'coldens_{qt}_{sc}_snap{sn}_shrink-sph-cen_BN98' + \
                '_2rvir{depl}_v2.hdf5'
    checkfileflag = False
    if opt == 1:
        simname = simname1
        dirpath = dirpath1
        ions = ['o6']
        maptype = 'ion'
        _maptype_args = {'ps20depletion': False}
        snapnum = 27
    
        maptype_argss = [{key: _maptype_args[key] for key in _maptype_args} \
                         for ion in ions]
        [maptype_args.update({'ion': ion}) \
         for maptype_args, ion in zip(maptype_argss, ions)]
    
    elif opt > 1 and opt <= 9:
        dirpath = '/projects/b1026/isultan/m12i_noAGNfb/'
        simname = 'm12i_noAGNfb_CR-diff-coeff-690_FIRE-2'
        _ions = ['si4', 'n5', 'o6', 'ne8']
        ions = [_ions[(opt - 2) % 4]]
        maptype = 'ion'
        _maptype_args = {'ps20depletion': False}
        snapshots = [277, 600]
        snapnum = snapshots[(opt - 2) // 4]
        
        maptype_argss = [{key: _maptype_args[key] for key in _maptype_args} \
                         for ion in ions]
        [maptype_args.update({'ion': ion}) \
         for maptype_args, ion in zip(maptype_argss, ions)]
    
    elif opt >= 10 and opt < 21:
        # check ion sum, los metallicity
        simname = simname1
        dirpath = dirpath1
        snapnum = 27
        if opt < 19:
            ions = ['O{}'.format(opt - 9)]
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
    
            maptype_argss = [{key: _maptype_args[key] for key in _maptype_args} \
                              for ion in ions]
            [maptype_args.update({'ion': ion}) \
             for maptype_args, ion in zip(maptype_argss, ions)]
        elif opt == 19:
            maptype = 'Metal'
            _maptype_args = {'element': 'Oxygen'}
            
            maptype_argss = [_maptype_args]
        elif opt == 20:
            maptype = 'Mass'
            _maptype_args = {}
            
            maptype_argss = [_maptype_args]
    elif opt >= 21 and opt < 57:
        # 36 indices; frontera paths
        outdir = '/scratch1/08466/tg877653/output/maps/set1_BH_noBH/'
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'N5', 'C2', 'Si2', 'Fe2', 'Mg2', 'Mg10']
        # z=0.4, 0.5, 0.6, redshifts match exactly at least for same ICs
        snaps = [50] # 49, 51
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # the m13 sdp1e-4 run (with BH) was only run down to z=1
        simnames = ['m12i_m6e4_MHD_fire3_fireBH_Sep052021_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm12i_m6e4_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHD_fire3_fireBH_Sep052021_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                   ]
        ind = opt - 21
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]

    elif opt >= 57 and opt < 93:
        # 36 indices; frontera paths
        outdir = '/scratch1/08466/tg877653/output/maps/set2_BH_noBH/'
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'N5', 'C2', 'Si2', 'Fe2', 'Mg2', 'Mg10']
        # z=0.4, 0.5, 0.6, redshifts match exactly at least for same ICs
        snaps = [50] # 49, 51
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # the m13s noBH run down to z=0 are all crdiff690
        # m13s with BH: m13h29 has crdiffc690, but noBH counterpart is not down to z=0
        #               same for m13h113, m13h236
        # m13h206 has m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5
        #             m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_vcr1000_Oct252021_crdiffc690_sdp1e-4_gacc31_fa0.5_fcr3e-4_vw3000
        # to z=0, with BH
        # noBH h206:  m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5
        #             m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5  
        # -> ONE m13 option down to z=0 for same physics model BH/noBH comp.             
        # also only has one m12 match, for m12i
        # checked all have 60 snaps 
        simnames = ['m12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5',
                   ]
        ind = opt - 57
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]

    elif opt >= 93 and opt < 189:
        # 96 indices; frontera paths
        ind = opt - 93
        outdir = '/scratch1/08466/tg877653/output/maps/set3_BH_noBH/'
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        # z=0.4, 0.5, 0.6, redshifts match exactly at least for same ICs
        snaps = [49, 51, 45, 60] # 49, 51
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # the m13s noBH run down to z=0 are all crdiff690
        # m13s with BH: m13h29 has crdiffc690, but noBH counterpart is not down to z=0
        #               same for m13h113, m13h236
        # m13h206 has m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5
        #             m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_vcr1000_Oct252021_crdiffc690_sdp1e-4_gacc31_fa0.5_fcr3e-4_vw3000
        # to z=0, with BH
        # noBH h206:  m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5
        #             m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5  
        # -> ONE m13 option down to z=0 for same physics model BH/noBH comp.             
        # also only has one m12 match, for m12i
        # checked all have 60 snaps 
        simnames = ['m12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm12i_m6e4_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR0_Oct142021_crdiffc690_sdp1e10_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]
    
    elif opt >= 189 and opt < 261:
        # split into m12 and m13 as two sets: different snapshot ranges
        ind = opt - 189
        outdir = '/scratch1/08466/tg877653/output/maps/set4_BH_noBH/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        # z=0.0, 0.5, 1.0
        snaps = [500, 258, 186] #z=0.0, 0.50, 1.0
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, hr.
        # m12f since m12i noBH counterpart is crossed out
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm12m_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm12m_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]
    
    elif opt >= 261 and opt < 333:
        # split into m12 and m13 as two sets: different snapshot ranges
        ind = opt - 261
        outdir = '/scratch1/08466/tg877653/output/maps/set5_BH_noBH/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        # z=0.0, 0.5, 1.0
        snaps = [60, 50, 45] #z=0.0, 0.50, 1.0
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # these m13s selected because noBH ran down to z=0
        simnames = ['m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h007_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h007_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]

    elif opt >= 333 and opt < 357:
        # split into m12 and m13 as two sets: different snapshot ranges
        ind = opt - 357
        outdir = '/scratch1/08466/tg877653/output/maps/set6_BH_noBH/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'N5', 'Mg2']
        # z=0.0, 0.5, 1.0
        snaps = [50, 45] #z=0.50, 1.0
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m13h002_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h002_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]
    elif opt == 357:
        # example of map with NaN values for debugging
        outdir = '/scratch1/08466/tg877653/output/maps/debug_mapnan/'
        checkfileflag = False
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simname = 'm13h002_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5'
        snapnum = 45
        ion = 'Mg10'
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)
        maptype = 'ion'
        _maptype_args = {'ps20depletion': False}
        _maptype_args.update({'ion': ion})
        maptype_argss = [_maptype_args.copy()]
    elif opt >= 358 and opt < 366:
        # split into m12 and m13 as two sets: different snapshot ranges
        ind = opt - 358
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10']
        # z=0.0, 0.5, 1.0
        snaps = [50] #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m12m_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        # already have the AGN with CR and no BH versions of this in set4
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]
    elif opt >= 366 and opt < 378:
        # split into m12 and m13 as two sets: different snapshot ranges
        ind = opt - 366
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10']
        # z=0.0, 0.5, 1.0
        snaps = [258] #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m13h029_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]    
    elif opt >= 378 and opt < 394:
        # split into m12 and m13 as two sets: different snapshot ranges
        ind = opt - 378
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10']
        # z=0.0, 0.5, 1.0
        snaps = [50] #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m13h029_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h029_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        # m13h206: already have AGN-no CR and no BH at z=0.5 (set 5)
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()]  
    elif opt >= 395 and opt < 398:
        # quest
        outdir = '/projects/b1026/nastasha/tests/start_fire/h1sim_tests/'
        checkfileflag = True
        # standard res M12, M13 w and w/o BH
        _dirpath = '/projects/b1026/snapshots/fire3/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simname = 'm12m_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5'
        # m13h206: already have AGN-no CR and no BH at z=0.5 (set 5)
        snapnum = 500
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)
        if opt == 395:
            maptype = 'Metal'
            _maptype_args = {'element': 'Hydrogen'}
            _outfilen = 'coldens_{qt}_{sc}_snap{sn}_shrink-sph-cen_BN98' + \
                '_2rvir{depl}_v2.hdf5'
        elif opt == 396:
            maptype = 'ion'
            _maptype_args = {'ion': 'H1', 'ionfrac-method': 'PS20',
                             'ps20depletion': False}
            _outfilen = 'coldens_{qt}-PS20_{sc}_snap{sn}_shrink-sph-cen_BN98' + \
                '_2rvir{depl}_v2.hdf5'
        elif opt == 397:
            maptype = 'ion'
            _maptype_args = {'ion': 'H1', 'ionfrac-method': 'sim',
                             'ps20depletion': False}
            _outfilen = 'coldens_{qt}-sim_{sc}_snap{sn}_shrink-sph-cen_BN98' + \
                '_2rvir{depl}_v2.hdf5'
        maptype_argss = [_maptype_args.copy()]  

    elif opt >= 398 and opt < 402:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 398
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['H1']
        # z=0.0, 0.5, 1.0
        snaps = [50] #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        # m13h206: already have AGN-no CR and no BH at z=0.5 (set 5)
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    elif opt >= 402 and opt < 404:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 402
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['H1']
        # z=0.0, 0.5, 1.0
        snaps = [258] #z=0.50,
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    elif opt >= 404 and opt < 406:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 404
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['H1']
        # z=0.0, 0.5, 1.0
        snaps = [258] #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    ]
        # m13h206: already have AGN-no CR and no BH at z=0.5 (set 5)
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    elif opt >= 406 and opt < 407:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 406
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set1/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['H1']
        # z=0.0, 0.5, 1.0
        snaps = [50] #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        # m13h206: already have AGN-no CR and no BH at z=0.5 (set 5)
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    
    elif opt >= 407 and opt < 507:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 407
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set2/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'H1']
        # z=0.0, 0.5, 1.0
        snaps = [49, 48, 47, 46, 45] #z=0.6, 0.7, 0.8, 0.9, 1.0
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    elif opt >= 507 and opt < 557:
        # add z=0.6 - 1.0, 0.1 steps, to clean sample (set2)
        ind = opt - 507
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set2/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'H1']
        snaps = [240, 224, 210, 197, 186] #z=0.6, 0.7, 0.8, 0.9, 1.0
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    elif opt >= 557 and opt < 607:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 557
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set2/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'H1']
        snaps = [240, 224, 210, 197, 186] #z=0.6, 0.7, 0.8, 0.9, 1.0
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    ]
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 
    elif opt >= 607 and opt < 632:
        # add H 1 to clean sample (excluding m12m -- has a bug)
        ind = opt - 607
        outdir = '/scratch1/08466/tg877653/output/maps/clean_set2/'
        checkfileflag = True
        # CUBS https://arxiv.org/pdf/2209.01228.pdf: 
        # At 𝑧≈1, HST/COS FUV spectra cover a wide
        # range of ions, including 
        # H i, He i, Cii, N ii to N iv, O i to O v, S ii to
        # S v, Ne iv to Ne vi, Ne viii, and Mg x
        # kinda random subset of those, H I not yet FIRE-consistent
        ions = ['Mass', 'O6', 'Ne8', 'Mg10', 'H1']
        snaps = [49, 48, 47, 46, 45] #z=0.6, 0.7, 0.8, 0.9, 1.0 #z=0.50,
        # standard res M12, M13 w and w/o BH
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        # from Lindsey's selection, sr.
        # for comparison to the to z=0.0 noBH m13s: did run to z=0.5,
        #     but second-fewest snapshots of the noBH m13s
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        # m13h206: already have AGN-no CR and no BH at z=0.5 (set 5)
        simi = ind // (len(snaps) * len(ions))
        snpi = (ind % (len(snaps) * len(ions))) // len(ions)
        ioni = ind % len(ions)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        ion = ions[ioni]
        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])
        #print(dirpath)

        if ion == 'Mass':
            maptype = 'Mass'
            maptype_argss = [{}]
        else:
            maptype = 'ion'
            if ion == 'H1':
                _maptype_args = {'ps20depletion': False, 
                                 'ionfrac-method': 'sim'}
            else:
                _maptype_args = {'ps20depletion': False}
            _maptype_args.update({'ion': ion})
            maptype_argss = [_maptype_args.copy()] 

    for maptype_args in maptype_argss:
        depl = ''
        if maptype == 'ion':
            qt = maptype_args['ion']
            if 'ionfrac-method' in maptype_args:
                if maptype_args['ionfrac-method'] == 'sim':
                    depl = '_ionfrac-fromsim'
                else:
                    _depl = maptype_args['ps20depletion']
                    if _depl:
                        depl = '_ps20-depl'
            else:
                _depl = maptype_args['ps20depletion']
                if _depl:
                    depl = '_ps20-depl'
        elif maptype == 'Metal':
            qt = maptype_args['element']
        elif maptype == 'Mass':
            qt = 'gas-mass'

        outfilen = outdir + _outfilen.format(sc=simname, sn=snapnum, 
                                             depl=depl, qt=qt)
        if checkfileflag:
            if os.path.isfile(outfilen):
                msg = 'For opt {}, output file already exists:\n{}'
                print(msg.format(opt, outfilen))
                print('Not running this map again')
                return None

        massmap(dirpath, snapnum, radius_rvir=2., particle_type=0,
                pixsize_pkpc=3., axis='z', outfilen=outfilen,
                center='shrinksph', norm='pixsize_phys',
                maptype=maptype, maptype_args=maptype_args)

def tryout_wholezoom(index):
    outdir = '/projects/b1026/nastasha/tests/start_fire/map_tests/'

    if index == 0:
        dirpath = '/projects/b1026/snapshots/fire3/m13h206_m3e5/' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000/' 
        simname = 'm13h206_m3e5__' + \
                  'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'                     
        snapnum = 27  
        outfilen_template = 'mass_pt{pt}_{sc}_snap{sn}_axis-{ax}_' + \
                            'wholezoom_v1.hdf5'
        _temp = outdir + outfilen_template 
        outfilens = {'outfilen_gas': _temp.format(pt=0, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),
                     'outfilen_DM': _temp.format(pt=1, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),
                     'outfilen_stars': _temp.format(pt=4, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),
                     'outfilen_BH': _temp.format(pt=5, sc=simname, 
                                                 sn=snapnum, ax='{ax}'),                            
                    }

    massmap_wholezoom(dirpath, snapnum, pixsize_pkpc=3.,
                      **outfilens)

def checkfields_units(dirpath, snapnum, *args, numpart=100, 
                      outfilen='fields.hdf5'):
    '''
    Read in the data from the snapshot specified by dirpath
    and snap, read in the fields in args, convert to CGS, 
    save in file outfilen.
    '''
    snap = rf.get_Firesnap(dirpath, snapnum) 
    with h5py.File(outfilen, 'w') as f:
        hed = f.create_group('Header')
        cgrp = hed.create_group('cosmopars')
        cosmopars = snap.cosmopars.getdct()
        for key in cosmopars:
            cgrp.attrs.create(key, cosmopars[key])
        hed.attrs.create('snapnum', snapnum)
        hed.attrs.create('filepath_first', np.string_(snap.firstfilen))
        _info = 'datasets from FIRE hdf5 files stored in (physical) CGS units'
        hed.attrs.create('info', np.string_(_info))
        for arg in args:
            vals = snap.readarray_emulateEAGLE(arg)[:numpart]
            toCGS = snap.toCGS
            # e.g. masses overflow float32 in CGS
            # arrays are small anyway
            vals = vals.astype(np.float64) * toCGS 
            f.create_dataset(arg, data=vals)

def tryout_hist(index):
    outdir = '/projects/b1026/nastasha/tests/start_fire/hist_tests/'
    if index == 0:
        dirpath = '/projects/b1026/snapshots/fire3/m13h206_m3e5/' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000/'
        snapnum = 27
        simname = 'm13h206_m3e5__' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
        outfilen = 'hist_Oxygen_by_Mass_0-1-2Rvir_{sc}_snap{sn}_shrink-sph-cen_BN98' + \
                   '_2rvir_v1.hdf5'

        axtypes = ['sim-direct']
        axtypes_args = [{'field': 'ElementAbundance/Oxygen'}]
        weighttype = 'Mass'
        weighttype_args = {}
        rbins = np.array([0., 1., 2.])
        runit = 'Rvir'

        outfilen = outfilen.format(sc=simname, sn=snapnum,)
    else:
        raise ValueError('invalid index: {}'.format(index))

    histogram_radprof(dirpath, snapnum,
                      weighttype, weighttype_args, axtypes, axtypes_args,
                      particle_type=0, 
                      center='shrinksph', rbins=rbins, runit=runit,
                      logweights=True, logaxes=True, axbins=0.05,
                      outfilen=outdir + outfilen)

def run_hist(opt):
    if opt >= 0 and opt < 60:
        ind = opt
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [50] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 60 and opt < 96:
        ind = opt - 60
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [50] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)

        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 96 and opt < 126:
        ind = opt - 96
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [258] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 126 and opt < 144:
        ind = opt - 126
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [258] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)

        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 144 and opt < 174:
        ind = opt - 144
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [258] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 174 and opt < 192:
        ind = opt - 174
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [258] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    
    elif opt >= 192 and opt < 207:
        ind = opt - 192
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [50] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 207 and opt < 216:
        ind = opt - 206
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [50] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    
    # add more redshifts (set 2) to previous sample
    #----------------------------------------------
    elif opt >= 216 and opt < 516:
        ind = opt - 216
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [45, 46, 47, 48, 49] # z=1.0 - 0.6
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 516 and opt < 696:
        ind = opt - 516
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [45, 46, 47, 48, 49] # z=1.0 - 0.6
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)

        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 696 and opt < 846:
        ind = opt - 696
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [186, 197, 210, 224, 240] # z=1.0 - 0.6
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 846 and opt < 936:
        ind = opt - 846
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [186, 197, 210, 224, 240] # z=1.0 - 0.6
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)

        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 936 and opt < 1086:
        ind = opt - 936
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [186, 197, 210, 224, 240] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 1086 and opt < 1176:
        ind = opt - 1086
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [186, 197, 210, 224, 240] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    
    elif opt >= 1176 and opt < 1251:
        ind = opt - 1176
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['Mass', 'Volume', 'H1']
        snaps = [45, 46, 47, 48, 49] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        axtypes_opts = [['sim-direct']] * 5
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/Oxygen'}],
                             [{'field': 'ElementAbundance/Neon'}],
                             [{'field': 'ElementAbundance/Magnesium'}],
                            ]
        axqts = ['Temperature', 'Density', 'Oxygen', 'Neon', 'Magnesium']
        axbins = [0.05, 0.05] + [0.1] * 3

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)
    elif opt >= 1251 and opt < 1296:
        ind = opt - 1251
        outdir = '/scratch1/08466/tg877653/output/hists/clean_set1_set2/'
        outname = 'hist_{axqt}_r3D_by_{wt}_{simname}_snap{snap}_bins1_v1.hdf5'
        particle_type = 0
        wts = ['O6', 'Ne8', 'Mg10']
        snaps = [45, 46, 47, 48, 49] # z=0.5
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        axtypes_opts = [['sim-direct']] * 3
        axtypes_args_opts = [[{'field': 'Temperature'}],
                             [{'field': 'Density'}],
                             [{'field': 'ElementAbundance/{elt}'}],
                            ]
        axqts = ['Temperature', 'Density', '{elt}']
        axbins = [0.05, 0.05, 0.1]

        simi = ind // (len(snaps) * len(wts) * len(axqts))
        snpi = (ind % (len(snaps) * len(wts) * len(axqts))) \
               // (len(wts) * len(axqts))
        wti = (ind % (len(wts) * len(axqts))) // len(axqts)
        axi = ind % len(axqts)
        simname = simnames[simi]
        snapnum = snaps[snpi]
        wt = wts[wti]
        axtypes = axtypes_opts[axi]
        axtypes_args = axtypes_args_opts[axi]
        axqt = axqts[axi]
        
        runit = 'pkpc'
        rbins = np.arange(40., 501., 20.) if simname.startswith('m12') else\
                np.arange(40., 1001., 20.)
        rbins = np.append(np.arange(0., 40., 5.), rbins)

        # directory is halo name + resolution 
        dp2 = '_'.join(simname.split('_')[:2])
        if dp2.startswith('m13h02_'):
            dp2 = dp2.replace('m13h02', 'm13h002')
        dirpath = '/'.join([_dirpath, dp2, simname])

        if wt in ['Mass', 'Volume']:
            weighttype = wt
            weighttype_args = dict()
        else:
            weighttype = 'ion'
            weighttype_args = {'ps20depletion': False, 'ion': wt,
                               'density': False}
            if wt == 'H1':
                weighttype_args.update({'ionfrac-method': 'sim'})
            else:
                dummytab = linetable_PS20(wt, 0.0, emission=False,
                                          vol=True, lintable=True)
                parentelt = dummytab.element
                axtypes_args = \
                    [{key: (dct[key]).format(elt=parentelt) for key in dct}\
                     for dct in axtypes_args]
                axqt = axqt.format(elt=parentelt)
        outfilen = outdir + outname.format(axqt=axqt, wt=wt, simname=simname, 
                                           snap=snapnum)

    histogram_radprof(dirpath, snapnum,
                      weighttype, weighttype_args, axtypes, axtypes_args,
                      particle_type=particle_type, 
                      center='shrinksph', rbins=rbins, runit=runit,
                      logweights=True, logaxes=True, axbins=axbins,
                      outfilen=outfilen, overwrite=False)

def run_checkfields_units(index):
    dirpath1 = '/projects/b1026/snapshots/fire3/m13h206_m3e5/' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000/'
    simname1 = 'm13h206_m3e5__' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    snaps1 = [27, 45] # earliest and latest stored, for max a factor diffs
    fields1 = ('PartType0/Density', 
               'PartType0/Masses',
               'PartType0/Pressure',
               'PartType0/Temperature',
               'PartType0/Metallicity',
               'PartType0/ElementAbundance/Hydrogen',
               'PartType0/ElementAbundance/Oxygen',
               'PartType0/Coordinates',
               'PartType0/SmoothingLength')
    outdir1 = '/projects/b1026/nastasha/tests/start_fire/'
    outtemp = 'cgs_units_test_{simname}_snap{snap:03d}.hdf5'
    if index in [0, 1]:
        snap = snaps1[index]
        outfilen = outdir1 + outtemp.format(simname=simname1, snap=snap)
        checkfields_units(dirpath1, snap, *fields1, numpart=100, 
                          outfilen=outfilen)

def test_ionbal_calc(dirpath, snapnum, ion, target_Z=0.01, delta_Z=0.001,
                     ps20depletion=False, outfilen='ionbal_test.hdf5',
                     lintable=False):
    snap = rf.get_Firesnap(dirpath, snapnum)
    cosmopars = snap.cosmopars.getdct()
    
    # filter sim. particles and calculate ion balances, rho, T, Z
    metallicity = snap.readarray_emulateEAGLE('PartType0/Metallicity')
    zfilter = metallicity >= target_Z - delta_Z
    zfilter &= metallicity <= target_Z + delta_Z
    metallicity = metallicity[zfilter]
    indct = {'filter': zfilter}
    ionbals = get_ionfrac(snap, ion, indct=indct, table='PS20', simtype='fire',
                          ps20depletion=ps20depletion, lintable=lintable)
    temperature = snap.readarray_emulateEAGLE('PartType0/Temperature')[zfilter]
    temperature *= snap.toCGS
    hdens = snap.readarray_emulateEAGLE('PartType0/Density')[zfilter]
    hconv = snap.toCGS
    hdens *= snap.readarray_emulateEAGLE('PartType0/ElementAbundance/Hydrogen')[zfilter]
    hconv *= snap.toCGS
    hconv /= (c.atomw_H * c.u)
    hdens *= hconv
    
    # get corresponding ion balance table
    # for table read-in only, lin/log shouldn't matter; problems weren't there
    iontab = linetable_PS20(ion, cosmopars['z'], emission=False, vol=True,
                            lintable=False)
    iontab.findiontable()
    tab_logT = iontab.logTK
    tab_lognH = iontab.lognHcm3
    tab_logZ = iontab.logZsol + np.log10(iontab.solarZ)
    tab_ionbal_T_Z_nH = iontab.iontable_T_Z_nH.copy()
    if ps20depletion:
        tab_ionbal_T_Z_nH = 10**tab_ionbal_T_Z_nH
        iontab.finddepletiontable()
        tab_depletion_T_Z_nH = iontab.depletiontable_T_Z_nH.copy()
        tab_ionbal_T_Z_nH *= (1. - 10**tab_depletion_T_Z_nH)
        tab_ionbal_T_Z_nH = np.log10(tab_ionbal_T_Z_nH)
        tab_ionbal_T_Z_nH[tab_ionbal_T_Z_nH < -50.] = -50.

    interpvalZ = np.log10(target_Z)
    iZhi = np.where(tab_logZ >= interpvalZ)[0][0]
    iZlo = np.where(tab_logZ <= interpvalZ)[0][-1]
    if iZlo == iZhi:
        tab_ionbal_T_nH = tab_ionbal_T_Z_nH[:, iZlo, :] 
    else:
        hiZ = tab_logZ[iZhi]
        loZ = tab_logZ[iZlo]
        tab_ionbal_T_nH = (hiZ - interpvalZ) / (hiZ - loZ) * tab_ionbal_T_Z_nH[:, iZlo, :] +\
                          (interpvalZ - loZ) / (hiZ - loZ) * tab_ionbal_T_Z_nH[:, iZhi, :]
    tab_ionbal_T_nH = 10**tab_ionbal_T_nH

    # save data
    with h5py.File(outfilen, 'w') as f:
        hed = f.create_group('Header')
        cgrp = hed.create_group('cosmopars')
        cosmopars = snap.cosmopars.getdct()
        for key in cosmopars:
            cgrp.attrs.create(key, cosmopars[key])
        hed.attrs.create('snapnum', snapnum)
        hed.attrs.create('filepath_first', np.string_(snap.firstfilen))
        _info = 'FIRE calculated ion balances and the underlying ion balance table'
        hed.attrs.create('info', np.string_(_info))
        hed.attrs.create('target_Z', target_Z)
        hed.attrs.create('delta_Z', delta_Z)
        hed.attrs.create('ion', np.string_(ion))
        hed.attrs.create('ps20depletion', ps20depletion)
        hed.attrs.create('lintable', lintable)
        
        gsim = f.create_group('simulation_data')
        gsim.create_dataset('ionbal', data=ionbals)
        gsim.create_dataset('T_K', data=temperature)
        gsim.create_dataset('nH_cm**-3', data=hdens)
        gsim.create_dataset('metallicity_abs_mass_frac', data=metallicity)
        
        gtab = f.create_group('iontab_data')
        print('About to save tab_ionbal_T_nH')
        print('{} / {} NaN'.format(np.sum(np.isnan(tab_ionbal_T_nH)), 
                                   np.prod(tab_ionbal_T_nH.shape)))
        print(tab_ionbal_T_nH)
        gtab.create_dataset('ionbal_T_nH', data=tab_ionbal_T_nH)
        gtab.create_dataset('logT_K', data=tab_logT)
        gtab.create_dataset('lognH_cm**-3', data=tab_lognH)

def run_ionbal_test(opt=0):
    dirpath1 = '/projects/b1026/snapshots/fire3/m13h206_m3e5/' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000/'
    simname1 = 'm13h206_m3e5__' + \
               'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1' + \
               '_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'
    snaps1 = [27, 45]
    ions1 = ['O{}'.format(i) for i in range(1, 10)]

    outdir =  '/projects/b1026/nastasha/tests/start_fire/ionbal_tests/'
    outtemplate1 = outdir + 'ionbal_test_PS20_{ion}_depletion-{dp}_Z-{Z}' + \
                            '_snap{snap:03d}_{sim}.hdf5'
    outtemplate2 = outdir + 'ionbal_test_PS20_{ion}_depletion-{dp}_Z-{Z}' + \
                            '_snap{snap:03d}_lintable-{lintable}_{sim}.hdf5'
    
    if opt >= 0 and opt < 6:
        dirpath = dirpath1
        simname = simname1
        ions = ions1
        ps20depletion = bool(opt % 2)
        snapnum = snaps1[opt // 4]
        target_Z = [0.01, 0.0001][(opt  // 2) % 2]
        delta_Z = 0.1 * target_Z
        outtemplate = outtemplate1
        dolintable = False
    if opt >= 6 and opt < 18:
        _opt = opt - 6
        __opt = _opt % 6
        lintable = bool(_opt // 6)
        dirpath = dirpath1
        simname = simname1
        ions = ions1
        ps20depletion = bool(__opt % 2)
        snapnum = snaps1[__opt // 4]
        target_Z = [0.01, 0.0001][(__opt  // 2) % 2]
        delta_Z = 0.1 * target_Z
        outtemplate = outtemplate2
        dolintable = True
    else:
        raise ValueError('Invalid opt {}'.format(opt))
    for ion in ions:
        if dolintable:
            outfilen = outtemplate.format(ion=ion, dp=ps20depletion, 
                                          Z=target_Z, sim=simname, 
                                          snap=snapnum, lintable=lintable)
            test_ionbal_calc(dirpath, snapnum, ion, target_Z=target_Z, 
                             delta_Z=delta_Z, ps20depletion=ps20depletion, 
                             outfilen=outfilen, lintable=lintable)
        else:
            outfilen = outtemplate.format(ion=ion, dp=ps20depletion, 
                                          Z=target_Z, sim=simname, 
                                          snap=snapnum)
            test_ionbal_calc(dirpath, snapnum, ion, target_Z=target_Z, 
                             delta_Z=delta_Z, ps20depletion=ps20depletion, 
                             outfilen=outfilen)

def run_halodata(opt):
    # test cases
    if opt >= 0 and opt < 6:
        ind = opt - 0
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = [('m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690'
                     '_sdp1e10_gacc31_fa0.5'),
                    ('m13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690'
                     '_sdp1e10_gacc31_fa0.5'),
                    ('m13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021'
                     '_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000'),
                    ]
        snaps = [45, 50]
        meandef = ('BN98', '200c', '200m', '500c')
    # clean samples 1 and 2
    elif opt >= 6 and opt < 30:
        ind = opt - 6
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h206_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h113_m3e5_MHD_fire3_fireBH_Sep182021_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm13h206_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    'm13h113_m3e5_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                   ]
        snaps = [45, 46, 47, 48, 49, 50]
        # might as well; extra overdensities are cheap
        meandef = ('BN98', '200c', '200m', '500c', '500m', 
                   '2500c', '2500m', '178c', '178m', '100c', '100m')
    elif opt >= 30 and opt < 42:
        ind = opt - 30
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m13h113_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e-4_gacc31_fa0.5',
                    'm13h206_m3e4_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp3e-4_gacc31_fa0.5',
                   ]
        snaps = [186, 197, 210, 224, 240, 258]
        # might as well; extra overdensities are cheap
        meandef = ('BN98', '200c', '200m', '500c', '500m', 
                   '2500c', '2500m', '178c', '178m', '100c', '100m')
    elif opt >= 42 and opt < 54:
        ind = opt - 42
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5',
                    'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp2e-4_gacc31_fa0.5',
                    ]
        snaps = [186, 197, 210, 224, 240, 258]
        # might as well; extra overdensities are cheap
        meandef = ('BN98', '200c', '200m', '500c', '500m', 
                   '2500c', '2500m', '178c', '178m', '100c', '100m')
    elif opt >= 54 and opt < 60:
        ind = opt - 54
        _dirpath = '/scratch3/01799/phopkins/fire3_suite_done/'
        simnames = ['m12f_m6e4_MHDCRspec1_fire3_fireBH_fireCR1_Oct252021_crdiffc1_sdp1e-4_gacc31_fa0.5_fcr1e-3_vw3000',
                    ]
        snaps = [45, 46, 47, 48, 49, 50]
        # might as well; extra overdensities are cheap
        meandef = ('BN98', '200c', '200m', '500c', '500m', 
                   '2500c', '2500m', '178c', '178m', '100c', '100m')
    simi = ind // len(snaps)
    snapi = ind % len(snaps)
    simname = simnames[simi]
    snapshot = snaps[snapi]

    dp2 = '_'.join(simname.split('_')[:2])
    if dp2.startswith('m13h02_'):
        dp2 = dp2.replace('m13h02', 'm13h002')
    dirpath = '/'.join([_dirpath, dp2, simname]) 
    
    gethalodata_shrinkingsphere(dirpath, snapshot, meandef=meandef)

def fromcommandline(index):
    '''
    This mapping is just based on the order in which I (first) ran things,
    and will not generally follow any kind of logic
    '''
    print('Running fire_maps.py process {}'.format(index))
    if index > 0 and index < 4:
        test_mainhalodata_units_ahf(opt=index)
    elif index == 4:
        # test a whole lot of snapshots in one go
        test_mainhalodata_units_multi_handler(opt=1)
    elif index == 5:
        test_mainhalodata_units_multi_handler(opt=2)
    elif index == 6:
        tryout_massmap(opt=1)
    elif index == 7:
        tryout_massmap(opt=2)
    elif index == 8:
        tryout_massmap(opt=3)
    elif index > 8 and index <= 11: # opt starts at 1
        opt = index - 8
        msg = 'Calling test_mainhalodata_units_rockstar(opt={})'
        print(msg.format(opt))
        test_mainhalodata_units_rockstar(opt=opt)
    elif index in [12, 13]:
        opt = index - 12
        run_checkfields_units(opt)
    elif index >= 14 and index < 20:
        run_ionbal_test(opt=index - 14)
    elif index == 20:
        tryout_ionmap(opt=1)
    elif index >= 21 and index < 33:
        # opt in [6, 18)
        run_ionbal_test(opt=index - 15)
    elif index >= 33 and index < 41:
        tryout_ionmap(opt=index - 31)
    elif index >= 41 and index < 52:
        tryout_ionmap(opt=index - 31)
    elif index == 52:
        tryout_hist(0)
    elif index >= 53 and index < 58:
        # launcher + script loading test
        print('Hello from index {}'.format(index))
    elif index >= 58 and index < 94:
        # set 1 maps -- 1 sim failed
        tryout_ionmap(opt=index - 58 + 21)
    elif index >= 94 and index < 130:
        # set 2 maps 
        tryout_ionmap(opt=index - 94 + 57) 
    elif index >= 130 and index < 226:
        # set 3 maps
        tryout_ionmap(opt=index - 130 + 93)
    elif index >= 226 and index < 394:
        # sets 4, 5, 6
        # set 4 (m12, high-res): 226 - 297 (72 inds)
        # sets 5,6 (m13, standard-res): 298 - 393 (96 inds)
        tryout_ionmap(opt=index - 226 + 189)
    elif index == 394:
        # NaN values in maps debug: single map example
        tryout_ionmap(opt=357)
    elif index >= 395 and index < 431:
        tryout_ionmap(opt=index - 395 + 358)
        # clean sample set 1: 
        # the parts that weren't already in sets 4-6
        # 395 - 402: m12 lower-res
        # 403 - 414: m13 higher-res
        # 415 - 430: m13 standard-res
    elif index >= 431 and index < 434:
        tryout_ionmap(opt=index - 431 + 395)
        # two H I methods and H total: sanity check for H1-sim impl.
    elif index >= 434 and index < 443:
        tryout_ionmap(opt=index - 434 + 398)
        # H I maps for clean sample z=0.5
        # 434 - 437: m13 standard-res
        # 438 - 439: m13 hi-res
        # 440 - 441: m12 hi-res
        # 442:       m12 standard-res
    elif index >= 443 and index < 668:
        tryout_ionmap(opt=index - 443 + 407)
        # 4 ions + mass for 5 redshifts 0.6 - 1.0, clean sample 
        # (no m12m)
        # 443 - 542: m13-SR (4 IC/phys)
        # 543 - 592: m13-HR (2 phys)
        # 593 - 642: m12-HR (2 phys)
        # 643 - 667: m12-SR (1 IC/phys)
        # opts [407, 632) (9 x 25 indices)
    elif index >= 668 and index < 884:
        run_hist(index - 668 + 0)
        # z=0.5 only
        # Mass, Volume, H I: (T, rho, O, Ne, Mg) profiles
        # O6, Ne8, Mg10: (T, rho, parent element) profiles
        # 668 - 727: m13-SR (4 IC/phys), Mass, Volume, HI
        # 728 - 763: m13-SR (4 IC/phys), O6, Ne8, Mg10
        # 764 - 793: m12-HR (2 IC/phys), Mass, Volume, HI
        # 794 - 811: m12-HR (2 IC/phys), O6, Ne8, Mg10
        # 812 - 841: m13-HR (2 IC/phys), Mass, Volume, HI
        # 842 - 859: m13-HR (2 IC/phys), O6, Ne8, Mg10
        # 860 - 874: m12-SR (1 IC/phys), Mass, Volume, HI
        # 875 - 883: m12-SR (1 IC/phys), O6, Ne8, Mg10
    elif index >= 884 and index < 1964:
        run_hist(index - 884 + 216)
        # z=0.6, 0.7, 0.8, 0.9, 1.0
        # Mass, Volume, H I: (T, rho, O, Ne, Mg) profiles
        # O6, Ne8, Mg10: (T, rho, parent element) profiles
        #  884 - 1183: m13-SR (4 IC/phys), Mass, Volume, HI
        # 1184 - 1363: m13-SR (4 IC/phys), O6, Ne8, Mg10
        # (480 inds)
        # 1364 - 1513: m12-HR (2 IC/phys), Mass, Volume, HI
        # 1514 - 1603: m12-HR (2 IC/phys), O6, Ne8, Mg10
        # (240 inds)
        # 1604 - 1753: m13-HR (2 IC/phys), Mass, Volume, HI
        # 1754 - 1843: m13-HR (2 IC/phys), O6, Ne8, Mg10
        # (240 inds)
        # 1844 - 1918: m12-SR (1 IC/phys), Mass, Volume, HI
        # 1919 - 1963: m12-SR (1 IC/phys), O6, Ne8, Mg10
        # (120 inds)
    elif index >= 1964 and index < 1970:
        # test halo centering script
        run_halodata(index - 1964)
    elif index >= 1970 and index < 2024:
        # clean samples 1/2 
        # 1970 - 1993: m13-SR (24 inds)
        # 1994 - 2005: m13-HR (12 inds)
        # 2006 - 2017: m12-HR (12 inds)
        # 2018 - 2023: m12-SR (6 inds)
        run_halodata(index - 1970 + 6)
    elif index >= 2024 and index < 2030:
        # debugging Mvir/Rvir finder: fp same Mvir values
        snaps = [186, 197, 210, 224, 240, 258]
        snapshot = snaps[index - 2024]
        path = ('/scratch3/01799/phopkins/fire3_suite_done/m12f_m7e3/'
                'm12f_m7e3_MHD_fire3_fireBH_Sep182021_hr'
                '_crdiffc690_sdp1e10_gacc31_fa0.5')
        res = calchalodata_shrinkingsphere(path, snapshot, 
                                           meandef=('200c', 'BN98'))
        print(path)
        print(snapshot)
        print(res)
    else:
        raise ValueError('Nothing specified for index {}'.format(index))

def launchergen(*args, logfilebase='{ind}.out'):
    '''
    not that useful; just use 
    echo -e "commands ${index} >> logfile_${index}" >> launchfile_name 
    in the batch script

    Parameters:
    -----------
    args: indexable of integers
        the indices to call fromcommandline with, one for each launched
        process
    logfilebase: string, formattable with argument 'ind'
        where to write the logfiles. {ind} is replaced by the index in each 
        line
    Returns:
    --------
    prints the launcher file lines. Direct output to a file to generate one.
    '''
    
    fillline = 'python ./fire_maps.py {ind} > ' + logfilebase + ' 2>&1'
    for arg in args:
        print(fillline.format(ind=arg))

if __name__ == '__main__':
    #print('fire_maps.py script started')
    if len(sys.argv) > 1:
        # generate launcher file for frontera
        # arguments: 
        #   --launchergen : generate a launcher file instead of 
        #                   default 'run with this index'
        #   --logfilebase=<string> : write log files for each launcher 
        #                   process to a file like this. Must contain a
        #                   '{ind}' part, since this is where the script
        #                   will fill in the index each process is called 
        #                   with
        #   integers :      the indices to call this script (fire_maps.py)
        #                   with in the launcher run
        if '--launchergen' in sys.argv:
            inds = [int(arg) if '-' not in arg else None \
                    for arg in sys.argv[1:]]
            while None in inds:
                inds.remove(None)
            kw = {}
            for arg in sys.argv[1:]:
                if '--logfilebase=' in arg:
                    kw['logfilebase'] = arg.split('=')[-1]
                    break
            launchergen(*inds, **kw)
        # just run code
        else:
            print('fire_maps.py script started')
            try:
                ind = int(sys.argv[1])
            except ValueError as msg1:
                msg2 = 'Could not interpret first command-line' + \
                       ' argument {} as int'
                msg2 = msg2.format(sys.argv[1])
                raise ValueError('/n'.join([msg1, msg2]))
            fromcommandline(ind)
    else:
        raise ValueError('Please specify an integer index > 1')
    
    








