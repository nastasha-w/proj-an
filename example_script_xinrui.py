#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:06:52 2020

@author: wijers
"""

# modules we need
import numpy as np
import read_eagle_files as eag

# plotting, including 3d plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


####### set up the python object to read in 
file_type = 'snap'  # snaphot containing all the particles in the simulation
snapnum = 28        # snapshot number (28 means redshift=0)
gadgetunits = True  # get the data in the units used in the simulation 
                    # (Eagle is based on a code called Gadget3)
suppress = False    # don't suppress printing data on what the code is doing
simulation = 'L0012N0188' # small simulation box: less data, runs faster
physics_variation = 'REFERENCE' # use simulation run with the standard Eagle 
                                # physics
region = None       # you can use this to specify the corners of a region to
                    # get the data for, if you only want to look at part of the 
                    # simulation (units: comoving Mpc / h)
                    
# string formatting: fill something in in a python string
simdir = '/disks/eagle/{sim}/{var}'.format(sim=simulation,\
                                           var=physics_variation)

# gadgetunits doens't actaully do anything here, only in read_data_array
read = eag.read_eagle_file(simdir, file_type, snapnum,\
                           gadgetunits=gadgetunits, suppress=suppress)


###### exmaple: read in mass and coordinate arrays
name = 'PartType0/Mass' # PartType0 is gas
# gasmass contains the masses of all the gas particles, in the units from the
# simulation
gasmass = read.read_data_array(name, gadgetunits=gadgetunits,\
                                     suppress=suppress, region=region)
a_sc_mass = read.a_scaling
h_sc_mass = read.h_scaling
CGS_c_mass = read.CGSconversion
toCGS_mass = read.a**a_sc_mass * read.h**h_sc_mass * CGS_c_mass

# array *= other_array or value: multiply all the values in array
# somewhat faster than array = array * other_array or value
gasmass *= toCGS_mass / 1.989e33 # convert units to solar masses 
                                 # (1 solar mass = 1.989e33 g)


name = 'PartType0/Coordinates' # PartType0 is gas
# gasmass contains the masses of all the gas particles, in the units from the
# simulation
gascoords = read.read_data_array(name, gadgetunits=gadgetunits,\
                                     suppress=suppress, region=region)
a_sc_coord = read.a_scaling
h_sc_coord = read.h_scaling
CGS_c_coord = read.CGSconversion
toCGS_coord = read.a**a_sc_coord * read.h**h_sc_coord * CGS_c_coord

# array *= other_array or value: multiply all the values in array
# somewhat faster than array = array * other_array or value
gascoords *= toCGS_coord / 3.085678e24 # convert units to Mpc (physical)
# array.shape = (number of particles, 3) 


###### make some exmaple plots

## 3d scatter plot for gas (use a subset of gas particles so rotations etc. 
# are fast)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# numpy_array[::100] will select the particles at indices 0, 100, 200, etc.
# 
subsample = 4000
# subsample the coordinate array along the first axis (which particle),
# choose a fixed index along the second axis: 0=x, 1=y, 2=z
ax.scatter(gascoords[::subsample, 0],\
           gascoords[::subsample, 1],\
           gascoords[::subsample, 2],\
           c='red', marker='o', alpha=0.2)
ax.set_xlabel('X [Mpc]')
ax.set_ylabel('Y [Mpc]')
ax.set_zlabel('Z [Mpc]')
ax.set_title('Gas particles: 1 in {num}'.format(num=subsample))

# close plot so the script will continue; check plt.savefig to save a picture
# to a file instead
plt.show()


## gas surface density plot: gas mass projected along the z-axis
# there are more sophistsicated ways of doing this that account for gas 
# particles representing an extended distribution of gas, rather than a point,
# but this is a good first step, and fine for just seeing what things look like

boxsize = read.boxsize * read.a / read.h # (boxsize in physical Mpc)
resolution = 200
edges = np.linspace(0., boxsize, num=resolution + 1) # 1 more edges than bins

# numpy histogram2d: add up masses of the particles in x/y bins
hist, xedges, yedges = np.histogram2d(gascoords[:, 0], gascoords[:, 1],\
                                    bins=[edges, edges], weights=gasmass)
# divide masses by bins sizes to get column densites
hist /= (boxsize / float(resolution))**2

# these sorts of things are useful to check; the average gas density and the 
# box volume are know, so this can be checked against those
total_gasmass = np.sum(hist) * (boxsize / float(resolution))**2
print('total mass: {tot:.2e} solar masses'.format(tot=total_gasmass))

ax = plt.subplot(111)
img = ax.imshow(np.log10(hist.T), origin='lower', interpolation='nearest',\
                extent=(0., boxsize, 0., boxsize))
ax.set_xlabel('X [Mpc]')
ax.set_ylabel('Y [Mpc]')
cbar = plt.colorbar(img, ax=ax)
# text betwen dollar signs is rendered with LaTeX math mode
# backslashes need to be escaped in python strings
cbar.set_label('Gas surface density ' + \
               '[$\\log_{10} \\, \\mathrm{M}_{\\odot} \\, \\mathrm{Mpc}^{-2}$]') 

plt.show()

