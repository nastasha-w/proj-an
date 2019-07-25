#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:21:54 2018

@author: wijers
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axgrid
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import numpy as np
import make_maps_v3_master as m3


def add_colorbar(ax,img=None,vmin=None,vmax=None,cmap=None,clabel=None,newax=False,extend='neither',fontsize=12,orientation='vertical'):
    if img is None:
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,extend=extend,orientation=orientation)
    elif newax:
        div = axgrid.make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.2)
        cbar = mpl.colorbar.Colorbar(cax,img,extend=extend)
    else:
        cbar = mpl.colorbar.Colorbar(ax,img,extend=extend,orientation=orientation)
    ax.tick_params(labelsize=fontsize-2)
    if clabel is not None:
        cbar.set_label(clabel,fontsize=fontsize)
        
def plothalos(simnum, snapnum, plotqts = None):
    '''
    uses stuff from https://stackoverflow.com/questions/9081553/python-scatter-plot-size-and-style-of-the-marker/24567352#24567352
    '''
    if plotqts is None:
        plotqts = m3.get_EA_FOF_MRCOP(simnum,snapnum,mdef='200c', outdct = None)
    
    fig, axes = plt.subplots(ncols=2,nrows=1,gridspec_kw = {'width_ratios': [10.,1.]})
    
    fontsize = 12
    scmap = 'viridis'
    cmap = mpl.cm.get_cmap(scmap)
    colorqts = np.log10(plotqts['M200c_Msun'][::-1])
    vmin = np.min(colorqts[np.isfinite(colorqts)])
    vmax = np.max(colorqts[np.isfinite(colorqts)])
    
    alphamin = 0.3
    alphamax=0.8
    colors = cmap((colorqts - vmin)/(vmax-vmin))
    colors[:,3] = (colorqts - vmin)/(vmax-vmin) * (alphamax-alphamin) + alphamin #set alpha
    colors[:,3][np.logical_not(np.isfinite(colorqts))] = alphamin # set alpha for zero-mass halos
    
    xs = plotqts['COP_cMpc'][::-1, 0]
    ys = plotqts['COP_cMpc'][::-1, 1]
    sz =  plotqts['R200c_cMpc'][::-1]
    
    
    patches = [Circle((xs[ind], ys[ind]), sz[ind]) \
               for ind in range(len(xs))] # x, y axes only

    collection = PatchCollection(patches)
    collection.set(edgecolor=colors, facecolor='none',linewidth=2)
    axes[0].add_collection(collection)
    
    
    #add_colorbar(axes[1],img=None,clabel = r'$\log_{10} M_{200c} \, [M_{\odot}]$', fontsize=fontsize, cmap = scmap, vmin=vmin, vmax=vmax)
    ### add color bar by hand to account for alpha variations
    cax = axes[1]
    grid = np.arange(1001.)/1000.*(vmax-vmin) + vmin
    colors = cmap((grid - vmin)/(vmax-vmin))
    colors[:,3] = (grid - vmin)/(vmax-vmin) * (alphamax-alphamin) + alphamin #set alpha
    colors = np.array([colors]) # add outer dimension for a 2d image
    colors = np.transpose(colors,(1,0,2))
    
    cax.imshow(colors,extent=(0,1,grid[0],grid[-1]),origin='lower')
    cax.set_ylim((grid[0],grid[-1]))

    cax.tick_params(axis='x',which='both',bottom=False,top=False, labelbottom=False,labeltop=False)
    cax.tick_params(axis='y',which='both',left=False,right=True, labelleft=False,labelright=True, direction='out', labelsize=fontsize)
    cax.yaxis.set_label_position("right")
    cax.set_ylabel(r'$\log_{10} M_{200c} \, [M_{\odot}]$',fontsize=fontsize)
    
    xclim = cax.get_xlim()
    yclim = cax.get_ylim()
    cax.set_aspect(10.*(xclim[1]-xclim[0])/(yclim[1]-yclim[0]), adjustable='box-forced')
    
    ### set up axes and ticks for the main plot
    axes[0].set_xlim(0.,np.ceil(np.max(xs))) # there should be some halos close to
    axes[0].set_ylim(0.,np.ceil(np.max(ys)))
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    axes[0].set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')
    axes[0].tick_params(axis='both',which='both',bottom=True,top=True,left=True,right=True, labelbottom=True,labeltop=False, labelleft=True,labelright=False, labelsize=fontsize, direction = 'in', width = 1.5, length= 5.)
    axes[0].set_ylabel('X [cMpc]', fontsize=fontsize)
    axes[0].set_xlabel('Y [cMpc]', fontsize=fontsize)
    
    plt.show()