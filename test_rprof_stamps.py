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
import matplotlib.lines as mline
import matplotlib.gridspec as gsp

import coldens_rdist as crd
import plot_utils as pu

mdir = '/net/luttero/data2/imgs/test_rprof_stamps/'
teststampfile = ol.pdir + 'smallset_stamps_test.hdf5'
galids_all = [10001965, 10003249, 10005162, 35069041, 35066574]
halocat = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'

testoutput = ol.pdir + 'radprof/' + 'radprof_smallset_stamps_test.hdf5'

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

def callprofiles():
    rbins1 = np.arange(0., 150., 10.)
    rbins2 = np.arange(0., 1000., 10.)
    rbins3 = np.arange(0., 300., 10.)
    
    rbins4 = np.arange(0., 3., 0.1)
    rbins5 = np.arange(0., 4., 0.1)
    rbins6 = np.arange(0., 6., 0.1)
    
    galids_lo = [10001965, 10003249, 10005162]
    #galids_hi = [35069041, 35066574]
    
    # should fail (stamp size too small) -> does
    #crd.getprofiles_fromstamps(teststampfile, rbins2, galids_lo,\
    #                       runit='pkpc', ytype='perc', yvals=50.,\
    #                       halocat=halocat,\
    #                       separateprofiles=False, uselogvals=True,\
    #                       outfile=None, grptag='lomass')
    print('First one')
    crd.getprofiles_fromstamps(teststampfile, rbins3, galids_lo,\
                           runit='pkpc', ytype='perc', yvals=50.,\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag='lomass')
    print('end of first one')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='perc', yvals=[10., 50., 90.],\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='perc', yvals=[10., 50., 90.],\
                           halocat=halocat,\
                           separateprofiles=True, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='mean', yvals=[10., 50., 90.],\
                           halocat=halocat,\
                           separateprofiles=True, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='mean', yvals=[10., 50., 90.],\
                           halocat=halocat,\
                           separateprofiles=True, uselogvals=False,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='mean', yvals=[10., 50., 90.],\
                           halocat=halocat,\
                           separateprofiles=True, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='mean', yvals=[10., 50., 90.],\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=False,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='fcov', yvals=-6.,\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps([teststampfile], rbins3, galids_lo,\
                           runit='pkpc', ytype='fcov', yvals=[1e-6, 1e-4, 1e-2],\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=False,\
                           outfile=None, grptag='lomass')
     
    crd.getprofiles_fromstamps(teststampfile, rbins4, galids_lo,\
                           runit='R200c', ytype='perc', yvals=50.,\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag='lomass')
    print('Should generate a warning')
    crd.getprofiles_fromstamps(teststampfile, rbins5, galids_lo,\
                           runit='R200c', ytype='perc', yvals=50.,\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag='lomass')
    #print('Should fail (unsampled bins)') #-> does fail
    #crd.getprofiles_fromstamps(teststampfile, rbins6, galids_lo,\
    #                       runit='R200c', ytype='perc', yvals=50.,\
    #                       halocat=halocat,\
    #                       separateprofiles=False, uselogvals=True,\
    #                       outfile=None, grptag='lomass')
    print('end of expected warnings')
    crd.getprofiles_fromstamps(teststampfile, rbins4, galids_lo,\
                           runit='R200c', ytype='perc', yvals=50.,\
                           halocat=halocat,\
                           separateprofiles=True, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps(teststampfile, rbins4, galids_lo,\
                           runit='R200c', ytype='mean', yvals=50.,\
                           halocat=halocat,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag='lomass')
    crd.getprofiles_fromstamps(teststampfile, rbins4, galids_lo,\
                           runit='R200c', ytype='mean', yvals=50.,\
                           halocat=halocat,\
                           separateprofiles=True, uselogvals=True,\
                           outfile=None, grptag='lomass')


def gethistdata(ytype, yval, hist, vbins):
    '''
    assumes vbins are log values, histogram dimensions are (r, v)
    '''
    if ytype == 'perc':
        return pu.percentiles_from_histogram(hist, vbins, axis=1,\
                   percentiles=np.array([yval / 100.]))[0]
    elif ytype == 'mean':
        vedges = 10**vbins
        vcens = vedges[:-1] + 0.5 * np.diff(vedges)
        mean = np.sum(hist * vcens[np.newaxis, :], axis=1) \
               / np.sum(hist, axis=1)
        return np.log10(mean)
    elif ytype == 'mean_log':
        vedges = vbins
        vcens = vedges[:-1] + 0.5 * np.diff(vedges)
        return np.sum(hist * vcens[np.newaxis, :], axis=1) \
               / np.sum(hist, axis=1)
    elif ytype == 'fcov':
        cumul_hist =  np.cumsum(hist, axis=1)
        cumul_hist = cumul_hist.astype(float) / cumul_hist[:, -1][:, np.newaxis]
        fcov = [pu.linterpsolve(vbins[:-1], cumul_hist[i, :], yval / 100.)\
                for i in range(cumul_hist.shape[0])]
        fcov = [0. if fc is None else fc for fc in fcov]
        return fcov
        
        
def plottest_radprof():
    imgname = mdir + 'testprofiles.pdf'
    kwargs = {'stamp': {'pkpc':  {'linestyle': 'solid', 'linewidth': 1.5},\
                        'R200c': {'linestyle': 'dotted', 'linewidth': 1.5}},\
              'rprof': {'pkpc':  {'linestyle': 'dashed', 'linewidth': 2.5},\
                        'R200c': {'linestyle': 'dashdot', 'linewidth': 2.5}},\
             }
    # change if other profile options are included
    colors = {'mean_log': 'C0',\
              'mean': 'C1',\
              'perc_10.0': 'C2',\
              'perc_50.0': 'C3',\
              'perc_90.0': 'C4',\
              'fcov_-6.0': 'C5',\
              'fcov_0.0001': 'C6',\
              'fcov_0.01': 'C7',\
              'fcov_1e-06': 'C8',\
              }
    fontsize = 12
    ylabel1 = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{ph} \\, \\mathrm{cm}^{-1} \\mathrm{s}^{-1} \\mathrm{sr}^{-1}]$'
    ylabel2 = 'covering fraction (legend: SB units)'
    xlabel1 = '$\\mathrm{r}_{\\perp} \\; [\\mathrm{pkpc}]$'
    xlabel2 = '$\\mathrm{r}_{\\perp} \\; [\\mathrm{R}_{\\mathrm{200c}}]$'
    
    with h5py.File(testoutput, 'r') as tf,\
        h5py.File(teststampfile, 'r') as sf,\
        h5py.File(halocat, 'r') as hc:
        
        # find the different outputs in the test output to cross-ref with the
        # stamp file
        galsets = list(tf.keys())
        galids_indiv = [name.split('_')[-1] if 'galaxy' in name else None\
                        for name in galsets]
        galids_indiv = list(set(galids_indiv) - {None})
        
        galid_sets = [name if 'galset' in name else None\
                      for name in galsets]
        galid_sets = list(set(galid_sets) - {None})
        galid_sets = [tf['{}/galaxyid'.format(galset)][:] for galset in galid_sets]
        
        galids_all = set([int(galid) for galid in galids_indiv] +\
                         [galid for _set in galid_sets for galid in _set])
        galids_all = list(galids_all)
        
        # get r-val hists from the stamp files (pkpc units -- easier)
        cosmopars = {key: val for key, val in hc['Header/cosmopars'].attrs.items()}
        rbins_s_cMpc = np.arange(0., 500., 10.) * 1e-3 / cosmopars['a']
        stamps = {gid: sf[str(gid)][:] for gid in galids_all}
        pixsize_x = sf['Header'].attrs['pixel_size_x_cMpc']       
        pixsize_y = sf['Header'].attrs['pixel_size_y_cMpc']
        
        galids_hc = hc['galaxyid'][:]
        ginds_hc = np.array([np.where(int(galid) == galids_hc)[0][0]\
                             for galid in galids_all])
        R200c_cMpc = hc['R200c_pkpc'][:] * 1e-3 / cosmopars['a'] 
        R200c_cMpc = {galid: R200c_cMpc[gind] for galid, gind in\
                      zip(galids_all, list(ginds_hc))}
        
        # assume galaxy center = stamp center
        stamps[galids_all[0]]
        distances_sfile = {gid: np.indices(stamps[gid].shape) \
                                - 0.5 * (np.array(stamps[gid].shape) - 1.)[:, np.newaxis, np.newaxis]\
                           for gid in galids_all}
        distances_sfile = {gid: np.sqrt(np.sum(distances_sfile[gid]**2 \
                                  * np.array([pixsize_x, pixsize_y])[:, np.newaxis, np.newaxis]**2,\
                                  axis=0))\
                           for gid in galids_all}
        # quick check distance calulations (since they're similar to the ones in the used function)
        #for gid in galids_all:
        #    plt.imshow(distances_sfile[gid].T, origin='lower', interpolation='nearest',\
        #               cmap='viridis', extent=(0., stamps[gid].shape[0] * pixsize_x,\
        #                                       0., stamps[gid].shape[1] * pixsize_y))
        #    plt.colorbar()
        #    plt.show()
        
        # histogram r-value pairs (matched ranges help in stacking later)
        minv = np.min([np.min(stamps[gid]) for gid in galids_all])
        maxv = np.max([np.max(stamps[gid]) for gid in galids_all])
        epsilon = 1e-7 
        vbins = np.linspace(minv - epsilon, maxv + epsilon, 200)
        
        hists = {gid: np.histogram2d(distances_sfile[gid].flatten(), stamps[gid].flatten(),\
                                     bins=[rbins_s_cMpc, vbins])[0]\
                 for gid in galids_all}
        rcens_s = rbins_s_cMpc[:-1] + 0.5 * np.diff(rbins_s_cMpc)
        
        # set up the plotting space
        numpanels = len(galsets)
        
        panelheight = 3.
        panelwidth = 3.
        legheight = 1.2
        ncols_max = 4
        wspace = 0.
        hspace = 0.3
        
        ncols = min(ncols_max, numpanels)
        nrows = (numpanels - 1) // ncols + 1
        legbelow = ncols * nrows == numpanels
        
        height = panelheight * 2 * nrows + hspace * (2 * nrows - 1) \
                 + (legheight + hspace) * legbelow
        height_ratios = [panelheight] * nrows * 2
        if legbelow:
            height_ratios += [legheight]
        width = panelwidth * ncols + wspace * (ncols - 1)
        width_ratios = [panelwidth] * ncols
        
        fig = plt.figure(figsize=(width, height))
        fig.suptitle('Comparison of various individual and combined radial profiles\n' +\
                     'check profiles use median R200c of samples to convert to R200c units',\
                     fontsize=fontsize)
        grid = gsp.GridSpec(ncols=ncols, nrows=2 * nrows + legbelow,\
                            hspace=hspace, wspace=wspace, top=0.87,\
                            height_ratios=height_ratios,\
                            width_ratios=width_ratios)
        axes = [[fig.add_subplot(grid[pi // ncols + i, pi % ncols]) \
                 for i in range(2)] for pi in range(numpanels)]
        if legbelow:
            lax = fig.add_subplot(grid[2 * nrows, :])
        else:
            leftind = ncols - (numpanels - nrows * ncols) 
            lax = fig.add_subplot(grid[2 * nrows - 2:, leftind:])
        
        galids_indiv = [int(galid) for galid in galids_indiv]
        galids_sets = set(galsets) -\
                      set(['galaxy_{}'.format(galid) for galid in galids_indiv])
        galids_sets = sorted(list(galids_sets), key=lambda x: int(x.split('_')[-1]))
        galsets_all = galids_indiv + galids_sets
        
        for axi, galset in enumerate(galsets_all):
            axv = axes[axi][0]
            axf = axes[axi][1]
            labelleft = axi % ncols == 0
            labelright = axi % ncols == ncols - 1 or axi == numpanels - 1
            pu.setticks(axf, fontsize=fontsize, labelleft=labelleft,\
                        labelright=labelright)
            pu.setticks(axv, fontsize=fontsize, labelleft=labelleft,\
                        labelright=labelright)
            if labelleft:
                axv.set_ylabel(ylabel1, fontsize=fontsize)
                axf.set_ylabel(ylabel2, fontsize=fontsize)
            axf.set_xlabel(xlabel1, fontsize=fontsize)
            
            if isinstance(galset, int):
                grp = tf['galaxy_{}'.format(galset)]
                text1 = 'galaxyid: {}'.format(galset)
                text2 = text1
                hist = hists[galset]
                bingrps = list(grp.keys())
                R200c = R200c_cMpc[galset]
            else:
                grp = tf[galset]
                text1 = galset
                galids = grp['galaxyid'][:]
                text2 = text1 + ':\n' + '\n'.join([str(galid) for galid in galids])
                
                hist = np.sum(np.array([hists[galid] for galid in galids]), axis=0)
                bingrps = list(set(list(grp.keys())) - {'galaxyid'})
                R200c = np.median([R200c_cMpc[galid] for galid in galids])
                
            axv.text(0.98, 0.98, text1, fontsize=fontsize,\
                     transform=axv.transAxes, horizontalalignment='right',\
                     verticalalignment='top')
            axf.text(0.98, 0.98, text2, fontsize=fontsize,\
                     transform=axf.transAxes, horizontalalignment='right',\
                     verticalalignment='top')
            
            # for plotting in R200c units
            axv2 = axv.twiny()
            axf2 = axf.twiny()
            axv2.set_xlabel(xlabel2, fontsize=fontsize)
            pu.setticks(axv2, fontsize=fontsize,\
                        top=True, bottom=False, left=False, right=False,\
                        labeltop=True, labelbottom=False, labelleft=False,\
                        labelright=False)
            pu.setticks(axf2, fontsize=fontsize,\
                        top=True, bottom=False, left=False, right=False,\
                        labeltop=True, labelbottom=False, labelleft=False,\
                        labelright=False)
            
            for bingrp in bingrps:
                sgrp = grp[bingrp]
                
                if bingrp == 'pkpc_bins':
                    _axv = axv
                    _axf = axf
                    rnorm = 1e3 * cosmopars['a']
                    bintype = 'pkpc'
                elif bingrp == 'R200c_bins':
                    _axv = axv2
                    _axf = axf2
                    rnorm = 1. / R200c
                    bintype = 'R200c'
                
                for binset in sgrp.keys():
                    s2grp = sgrp[binset]
                    subkeys = list(s2grp.keys())
                    edges_t = s2grp['bin_edges'][:]
                    cens_t = edges_t[:-1] + 0.5 * np.diff(edges_t)
                    subkeys.remove('bin_edges')
                    
                    for subkey in subkeys:
                        prof_t = s2grp[subkey][:]
                        log_t = bool(s2grp[subkey].attrs['logvalues'])
                        if not log_t: # plot log here
                            prof_t = np.log10(prof_t)
                        
                        ytype = subkey.split('_')[0]
                        if ytype == 'mean':
                            ytype = subkey
                            yval = None
                        else:
                            yval = float(subkey.split('_')[1])
                            if ytype == 'fcov' and not log_t:
                                yval = np.log10(yval)
                        
                        yplot_s = gethistdata(ytype, yval, hist, vbins)
                        xplot_s = rcens_s * rnorm
                        if ytype == 'fcov':
                            ax = _axf
                        else:
                            ax = _axv
                        ax.plot(xplot_s, yplot_s, **kwargs['stamp'][bintype],\
                                color=colors[subkey])
                        ax.plot(cens_t, prof_t, **kwargs['rprof'][bintype],\
                                color=colors[subkey])
            # align plot limits for the pkpc/R200c axes
            pkpc_to_R200c = 1e-3 / cosmopars['a'] / R200c
            xlim_p1 = axv.get_xlim()
            xlim_p2 = axf.get_xlim()
            xlim_R1 = axv2.get_xlim()
            xlim_R2 = axf2.get_xlim()
            print(xlim_p1)
            print(xlim_R1)
            xmin_pkpc = min([xlim_p1[0], xlim_p2[0],\
                             xlim_R1[0] / pkpc_to_R200c,\
                             xlim_R2[0] / pkpc_to_R200c])
            xmax_pkpc = max([xlim_p1[1], xlim_p2[1],\
                             xlim_R1[1] / pkpc_to_R200c,\
                             xlim_R2[1] / pkpc_to_R200c])
            axv.set_xlim(xmin_pkpc, xmax_pkpc)
            axf.set_xlim(xmin_pkpc, xmax_pkpc)
            axv2.set_xlim(xmin_pkpc * pkpc_to_R200c, xmax_pkpc * pkpc_to_R200c)            
            axf2.set_xlim(xmin_pkpc * pkpc_to_R200c, xmax_pkpc * pkpc_to_R200c) 
            
    # sync y axes for all plots
    ylimsv = [axes[i][0].get_ylim() for i in range(numpanels)]
    ymin = min([ylim[0] for ylim in ylimsv])
    ymax = max([ylim[1] for ylim in ylimsv])
    [axes[i][0].set_ylim(ymin, ymax) for i in range(numpanels)]
    
    ylimsf = [axes[i][1].get_ylim() for i in range(numpanels)]
    ymin = min([ylim[0] for ylim in ylimsf])
    ymax = max([ylim[1] for ylim in ylimsf])
    [axes[i][1].set_ylim(ymin, ymax) for i in range(numpanels)]
    
    leghandles = [mline.Line2D([], [], **kwargs['stamp']['pkpc'],\
                               color='black', label='check, pkpc'),\
                  mline.Line2D([], [], **kwargs['rprof']['pkpc'],\
                               color='black', label='test, pkpc'),\
                  mline.Line2D([], [], **kwargs['stamp']['R200c'],\
                               color='black', label='check, R200c'),\
                  mline.Line2D([], [], **kwargs['rprof']['R200c'],\
                               color='black', label='test, R200c'),\
                  ]
    leghandles += [mline.Line2D([], [], color=colors[key], label=key)\
                   for key in colors]
    if legbelow: 
        lncols = ncols
    else:
        lncols = numpanels - nrows * ncols
    lax.legend(handles=leghandles, fontsize=fontsize, ncol=lncols,\
               loc='upper center', bbox_to_anchor=(0.5, 0.99))
    lax.axis('off')
    
    plt.savefig(imgname, format='pdf', bbox_inches='tight')