#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:39:35 2018

@author: wijers
"""

import make_maps_v3_master as m3
import numpy as np
import os
import ctypes as ct


NumPart = 10**5

### this is made for testing: just put in some dummy values with easy to check outcomes
# initial timing series
#coords = 1.5*np.ones((NumPart,3),dtype=np.float)
#lsmooth = 0.03*np.ones(NumPart, dtype=np.float)
#qW = np.ones(NumPart, dtype = np.float)
#qQ = np.ones(NumPart, dtype= np.float)
## race condition test series: all mass into one pixel -> all concurrent writes would produce race conditions
coords = 1.5005*np.ones((NumPart,3),dtype=np.float)
lsmooth = np.zeros(NumPart, dtype=np.float)
qW = np.ones(NumPart, dtype = np.float)
qQ = np.ones(NumPart, dtype= np.float)


Ls = np.array([3.,3.,3.])
Axis1 = 0
Axis2 = 1
Axis3 = 2
periodic = True
kernel = 'C2'
tree = False
dct = {'coords': coords, 'lsmooth': lsmooth, 'qW': qW, 'qQ': qQ}  
npix_x = 3000
npix_y = 3000 
box3 = np.array([3.,3.,3.])


#os.environ["OMP_NUM_THREADS"] = "2"

print('Testing if OpenMP works in general:')
test_library = ct.CDLL("/home/wijers/plot_sims/make_maps_emission_coupled/HsmlAndProject/test_omp_parallel.so")
 # set the return type
test_library.main.restype = int
print('----------')
# call the findHsmlAndProject C routine
test_library.main()
print('----------')


if False:
    print('\n\nTesting the projection function specifically:')
    ompproj = True
    resW, resQ = m3.project(NumPart,Ls,Axis1,Axis2,Axis3,box3,periodic,npix_x,npix_y,kernel,dct,tree,ompproj=ompproj)

    print("resW checksum: should be %i, got %f"%(NumPart,np.sum(resW)))
    print("resQ checkvalues: min, min > 0, max = %f, %f, %f, should be 0, 1, 1"%(np.min(resQ), np.min(resQ[resQ>0.]), np.max(resQ)) )

#m3.Simfile('L0012N0188',28,'REFERENCE')

if False:
    ompproj = True
    
    simnum = 'L0012N0188'
    snapnum = 28
    centre = [6.25,6.25,3.125]
    L_x = 12.5
    L_y = 12.5
    L_z = 6.25
    npix_x = 4000
    npix_y = 4000
    ptypeW = 'basic'
    resW, resQ = m3.make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW = None, abundsW = 'auto', quantityW = 'Mass',\
         ionQ = None, abundsQ = 'auto', quantityQ = 'Density', ptypeQ = 'basic',\
         excludeSFRW = 'T4', excludeSFRQ = 'T4', parttype = '0',\
         theta=0.0, phi=0.0, psi=0.0, \
         var='auto', axis ='z',log=True, velcut = False,\
         periodic = True, kernel = 'C2', saveres = False,\
         simulation = 'eagle', LsinMpc = None,\
         select = None, misc = None, ompproj = ompproj)

# comparison test to previous results
if True:
    ompproj = True
    
    simnum = 'L0012N0188'
    snapnum = 27
    centre = [6.25,6.25,6.25]
    L_x = 12.5
    L_y = 12.5
    L_z = 12.5
    npix_x = 800
    npix_y = 800
    ptypeW = 'basic'
    resW, resQ = m3.make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
         ptypeW,\
         ionW = None, abundsW = 'auto', quantityW = 'Mass',\
         ionQ = None, abundsQ = 'auto', quantityQ = 'Temperature', ptypeQ = 'basic',\
         excludeSFRW = 'T4', excludeSFRQ = 'T4', parttype = '0',\
         theta=0.0, phi=0.0, psi=0.0, \
         var='auto', axis ='z',log=True, velcut = False,\
         periodic = True, kernel = 'C2', saveres = True,\
         simulation = 'eagle', LsinMpc = None,\
         select = None, misc = None, ompproj = ompproj)