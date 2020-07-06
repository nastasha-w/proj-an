#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:01:11 2020

@author: wijers
"""

import numpy as np
import h5py
import make_maps_opts_locs as ol

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection



mdir = '/net/luttero/data2/imgs/test_rprof_stamps/'
teststampfile = ol.pdir + 'smallset_stamps_test.hdf5'
galids_all = [10001965, 10003249, 10005162, 35069041, 35066574]
halocat = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'

def plotstamps():
    fontsize = 12
    cmap = 'viridis'
    
    with h5py.File(teststampfile, 'r') as fi,\
        h5py.File(halocat, 'r') as hc:
        cosmopars = {key:val for key, val in hc['Header/cosmopars'].attrs.items()}
        
        xpsize = fi['Header'].attrs['pixel_size_x_cMpc']
        ypsize = fi['Header'].attrs['pixel_size_y_cMpc']
        
        galids_sfile = fi['Header/labels'][:]
        galids_halocat = hc['galaxyid'][:]
        
        
        for galid in galids_all:
            gind = np.where(galids_sfile == galid)[0][0] 
            llc = fi['Header/lower_left_corners_cMpc'][gind]
            gind = np.where(galids_halocat == galid)[0][0] 
            R200c_pkpc = hc['R200c_pkpc'][gind]
            M200c_Msun = hc['M200c_Msun'][gind]
            xcen = hc['Xcom_cMpc'][gind]
            ycen = hc['Ycom_cMpc'][gind]
            R200c_cMpc = R200c_pkpc * 1e-3 / cosmopars['a']
            stamp = fi[str(galid)][:]
            
            imgname = mdir + 'stamppplot_{gid}.pdf'.format(gid=galid)
            extent = (llc[0], llc[0] + stamp.shape[0] * xpsize,\
                      llc[1], llc[1] + stamp.shape[1] * ypsize)
            
            fig = plt.figure(figsize=(5.5, 5.))
            ax = fig.add_subplot(1, 1, 1)
            
            img = ax.imshow(stamp.T, origin='lower', interpolation='nearest',\
                          extent=extent, cmap=cmap)
            cbar = plt.colorbar(img, ax=ax)
            
            cbar.ax.tick_params(labelsize=fontsize - 1)
            cbar.ax.set_ylabel('$\\log_{10} \\, \\mathrm{ph} \\, \\mathrm{cm}^{-1} \\mathrm{s}^{-1} \\mathrm{sr}^{-1}$',\
                               fontsize=fontsize)
            ax.tick_params(labelsize=fontsize - 1)
            ax.set_xlabel('X [cMpc]', fontsize=fontsize)
            ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
            
            title = 'galaxyid: {galid}, $\\log_{{10}} \, \\mathrm{{M}}_{{\\mathrm{{200c}}}}' + \
                    ' \\,/\\, \\mathrm{{M}}_{{\\mathrm{{odot}}}} = {msun:.1f}$'
            fig.suptitle(title.format(galid=galid, msun=np.log10(M200c_Msun)))
            
            rc = mpatch.Circle((xcen, ycen), radius=R200c_cMpc,\
                               fill=False, color='red', linestyle='-')
            ax.add_artist(rc)
            
            plt.savefig(imgname, format='pdf', bbox_inches='tight')