# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:32:05 2017

@author: wijers
"""
import numpy as np
import h5py
import pandas as pd
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.lines as mlines
import matplotlib.gridspec as gsp
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# test module: expect changes so reload modules
import make_maps_v3_master as m3
import simfileclone as sfc
if sys.version.split('.')[0] == '3':
    from importlib import reload
reload(sfc)
reload(m3)
#import make_maps_opts_locs as ol
from prof3d_galsets import combine_hists


def getaveragecolumndensity():
    element = 'Oxygen'
    ion = 'o7'
    eltweight = m3.c.atomw_O # element mass in amu (c.u is amu/g)
    iontest=m3.Simfile('L100N256',32,'REF',simulation='bahamas')
    boxlength = 100./iontest.h*m3.c.cm_per_mpc #100 Mpc/h in cm
    
    mass = iontest.readarray('PartType0/Mass',rawunits=False)  
    eltabund = iontest.readarray('PartType0/ElementAbundance/%s'%element)
    density = iontest.readarray('PartType0/Density',rawunits=False)
    temperature = iontest.readarray('PartType0/Temperature',rawunits=False)
    habund = iontest.readarray('PartType0/ElementAbundance/Hydrogen')
    
    lognH = np.log10(density*habund/(m3.c.atomw_H*m3.c.u))
    ionbal = m3.find_ionbal(0.,ion,lognH,np.log10(temperature)) # fraction of element in ionisation state ion
    
    totalnumberofions = np.sum(ionbal*eltabund*mass/(eltweight*m3.c.u))
    avcoldens = totalnumberofions/boxlength**2
    
    return np.log10(avcoldens)
    
    # o7 results matched projection map averages

def runregiontests():
    # using o7 as a test ion; test should be considered a test of the region/particle selection, since the projection itself has been tested previously    
    
    # the whole box
    m3.make_map('L100N256',32,[50.,50.,50.],100.,100.,100.,2000,2000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=True)

    # spatial subsample to compare: middle 1/2
    m3.make_map('L100N256',32,[50.,50.,50.],50.,50.,100.,1000,1000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False)
    
    # spatial subslice: over two box edges
    m3.make_map('L100N256',32,[0.,0.,50.],50.,50.,100.,1000,1000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False)
    
    # rectangular box: middle 1/4x1/8
    m3.make_map('L100N256',32,[50.,50.,50.],50.,25.,100.,1000,500,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False)
    
    # middle 1/4 in two slices, 1/2 box each
    m3.make_map('L100N256',32,[50.,50.,50.],50.,50.,50.,1000,1000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False)
    m3.make_map('L100N256',32,[50.,50.,0.],50.,50.,50.,1000,1000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False)

    # middle 1/4 in two slices, velocity space
    m3.make_map('L100N256',32,[50.,50.,50.],50.,50.,50.,1000,1000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False, velcut=True)
    m3.make_map('L100N256',32,[50.,50.,0.],50.,50.,50.,1000,1000,'coldens',ionW='o7',excludeSFRW='T4',LsinMpc=False,simulation='bahamas',periodic=False, velcut=True)





### copied from /home/wijers/plot_sims/copy_durham_2017_07_04/tester.py
dd_new = '/data1/line_em_abs/v3_master_tests/results_v3p2/'
dd_cool = '/net/luttero/data1/line_em_abs/v3_master_tests/results_coolingtests_v3p21/'
dd_sylviassh = '/data1/line_em_abs/v3_master_tests/ssh_tables_sylvia/'

def imgplot(arr1,fontsize=12,clabel = '',name = dd_new + 'test.png', title = 'test'):
    fig, ax1 = plt.subplots(nrows=1,ncols=1)
    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.imshow(arr1.T,origin='lower', cmap=cm.get_cmap('viridis'), interpolation='nearest') 
    ax1.set_title('test run',fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax1)
    cax1 = div.append_axes("right",size="5%",pad=0.1)
    cbar1 = plt.colorbar(img, cax=cax1)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)
    
    plt.savefig(name,format = 'png')
    
def compareplot(arr1,arr2,fontsize=12,clabel = '',name = dd_new + 'test.pdf',diffmax = None):
    Vmin = -10. #min(np.min(arr1[np.isfinite(arr1)]),np.min(arr2[np.isfinite(arr2)]))
    Vmax = max(np.max(arr1[np.isfinite(arr1)]),np.max(arr2[np.isfinite(arr2)]))
    diff = arr1-arr2
    if diffmax is not None:
        maxdiff = diffmax
    else:    
        maxdiff = np.max(np.abs(diff)[np.isfinite(np.abs(diff))])
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.imshow(arr1.T,origin='lower', cmap=cm.get_cmap('viridis'), vmin = Vmin, vmax=Vmax,interpolation='nearest') 
    ax1.set_title('test run',fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax1)
    cax1 = div.append_axes("right",size="5%",pad=0.1)
    cbar1 = plt.colorbar(img, cax=cax1)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2.tick_params(labelsize=fontsize)
    ax2.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax2.imshow(arr2.T,origin='lower', cmap=cm.get_cmap('viridis'), vmin = Vmin, vmax=Vmax,interpolation='nearest') 
    ax2.set_title('check run',fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax2)
    cax2 = div.append_axes("right",size="5%",pad=0.1)
    cbar2 = plt.colorbar(img, cax=cax2)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax3.tick_params(labelsize=fontsize)
    ax3.patch.set_facecolor('black')
    img = ax3.imshow((diff).T,origin='lower', cmap=cm.get_cmap('RdBu'), vmin = -maxdiff, vmax=maxdiff,interpolation='nearest') 
    ax3.set_title('test - check',fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax3)
    cax3 = div.append_axes("right",size="5%",pad=0.1)
    cbar3 = plt.colorbar(img, cax=cax3)
    cbar3.solids.set_edgecolor("face")
    cbar3.ax.set_ylabel(r'$\Delta$' + clabel, fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)
    
    ax4.set_title('test - check')
    ax4.hist(np.ndarray.flatten(diff[np.isfinite(diff)]),log=True,bins=50)
    ax4.set_ylabel('number of pixels', fontsize = fontsize)
    ax4.set_xlabel(r'$\Delta$' +clabel, fontsize=fontsize)

    fig.tight_layout()
    plt.savefig(name, format='pdf')
    
    
def comparehist(arr1,arr2,fontsize=12,clabel = '',name = dd_new + 'test.png',diffmax = None,nbins=50):
    Vmin = min(np.min(arr1[np.isfinite(arr1)]),np.min(arr2[np.isfinite(arr2)]))
    Vmax = max(np.max(arr1[np.isfinite(arr1)]),np.max(arr2[np.isfinite(arr2)]))
    diff = arr1-arr2
    if diffmax is not None:
        maxdiff = diffmax
    else:    
        maxdiff = np.max(np.abs(diff)[np.isfinite(np.abs(diff))])
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    
    arrbins = np.arange(Vmin,Vmax + 0.0001*(Vmax-Vmin)/float(nbins), (Vmax-Vmin)/float(nbins))
    diffbins = np.arange(-maxdiff,maxdiff + 0.0001*2*maxdiff/float(nbins), 2.*maxdiff/float(nbins))

    ax1.tick_params(labelsize=fontsize)
    ax1.hist(arr1[np.isfinite(arr1)],bins = arrbins, log=True )    
    ax1.set_title('test run',fontsize=fontsize)
    ax1.set_xlabel(clabel,fontsize=fontsize)
    ax1.set_ylabel('columns',fontsize=fontsize)
    
    ax2.tick_params(labelsize=fontsize)
    ax2.hist(arr2[np.isfinite(arr2)],bins = arrbins, log=True )    
    ax2.set_title('check run',fontsize=fontsize)
    ax2.set_xlabel(clabel)
    ax2.set_ylabel('columns',fontsize=fontsize)
    
    ax3.tick_params(labelsize=fontsize)
    ax3.hist(diff[np.isfinite(diff)],bins = diffbins, log=True )    
    ax3.set_title('test- check',fontsize=fontsize)
    ax3.set_xlabel(r'$\Delta$' + clabel,fontsize=fontsize)
    ax3.set_ylabel('columns',fontsize=fontsize)
    
    
    sel = np.logical_and(np.isfinite(arr2),np.isfinite(diff))
    hist, exdges, yedges = np.histogram2d(arr2[sel],diff[sel], bins = [arrbins,diffbins])
    hist = np.log10(hist/float(np.sum(hist)))
    
    ax4.tick_params(labelsize=fontsize)
    ax4.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax4.imshow((hist).T,origin='lower', cmap=cm.get_cmap('viridis'),interpolation='nearest',extent = (Vmin, Vmax,-maxdiff,maxdiff), aspect='auto') 
    ax3.set_title('test - check',fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax4)
    cax4 = div.append_axes("right",size="5%",pad=0.1)
    cbar4 = plt.colorbar(img, cax=cax4)
    cbar4.solids.set_edgecolor("face")
    cbar4.ax.set_ylabel(r'$\log_{10}$ column fraction', fontsize=fontsize)
    cbar4.ax.tick_params(labelsize=fontsize)
    
    ax4.set_title('test - check')
    ax4.set_ylabel(r'$\Delta$' +clabel, fontsize=fontsize)
    ax4.set_xlabel(clabel + ' check', fontsize=fontsize)
    
    fig.tight_layout()
    plt.savefig(name,format = 'png')
    

def compare3dhist(arr1, arr2, bins1, bins2, fontsize=12, clabel='',\
                  name=dd_new + 'test.pdf', diffmax=None):
    
    if (len(arr1.shape) != 2) or (len(arr2.shape) != 2):
        raise ValueError('compare3dhist only works for 2d arrays')
    #min(np.min(arr1[np.isfinite(arr1)]),np.min(arr2[np.isfinite(arr2)]))
    Vmax = max(np.max(arr1[np.isfinite(arr1)]), np.max(arr2[np.isfinite(arr2)]))
    Vmin = min(np.min(arr1[np.isfinite(arr1)]), np.min(arr2[np.isfinite(arr2)]))
    Vmin = max(Vmax - 10., Vmin)
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,sharex=False,sharey=False)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    ph1, ph2, eds = combine_hists(arr1, arr2, bins1, bins2, rtol=1e-5, atol=1e-8, add=False)
    diff = ph1 - ph2
    if diffmax is not None:
        maxdiff = diffmax
    else:    
        maxdiff = np.max(np.abs(diff)[np.isfinite(np.abs(diff))])
        
    ax1.tick_params(labelsize=fontsize)
    ax1.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax1.pcolormesh(eds[0], eds[1], ph1.T, cmap=cm.get_cmap('viridis'),\
                         vmin=Vmin, vmax=Vmax,\
                         rasterized=True) 
    ax1.set_title('test run', fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax1)
    cax1 = div.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(img, cax=cax1)
    cbar1.solids.set_edgecolor("face")
    cbar1.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=fontsize)

    ax2.tick_params(labelsize=fontsize)
    ax2.patch.set_facecolor(cm.get_cmap('viridis')(0.))
    img = ax2.pcolormesh(eds[0], eds[1], ph2.T, cmap=cm.get_cmap('viridis'),\
                          vmin=Vmin, vmax=Vmax,\
                          rasterized=True) 
    ax2.set_title('check run', fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax2)
    cax2 = div.append_axes("right", size="5%",pad=0.1)
    cbar2 = plt.colorbar(img, cax=cax2)
    cbar2.solids.set_edgecolor("face")
    cbar2.ax.set_ylabel(clabel, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=fontsize)

    ax3.tick_params(labelsize=fontsize)
    ax3.patch.set_facecolor('black')
    img = ax3.pcolormesh(eds[0], eds[1], (diff).T,\
                         cmap=cm.get_cmap('RdBu'),\
                         vmin=-maxdiff, vmax=maxdiff,\
                         rasterized=True) 
    ax3.set_title('test - check', fontsize=fontsize)
    div = axgrid.make_axes_locatable(ax3)
    cax3 = div.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(img, cax=cax3)
    cbar3.solids.set_edgecolor("face")
    cbar3.ax.set_ylabel(r'$\Delta$' + clabel, fontsize=fontsize)
    cbar3.ax.tick_params(labelsize=fontsize)
    
    ax4.set_title('test - check')
    ax4.hist(np.ndarray.flatten(diff[np.isfinite(diff)]), log=True, bins=50)
    ax4.set_ylabel('number of pixels', fontsize=fontsize)
    ax4.set_xlabel(r'$\Delta$' + clabel, fontsize=fontsize)

    fig.tight_layout()
    plt.savefig(name, format='pdf')
    
##############################################################################    
####################### cooling table tests ##################################
##############################################################################



def plotWSS09fig2(ax, lambda_over_nH2, logTvals, lognHvals, color=None, linestyles=None, nHinds=None):
    if linestyles is None:
        linestyles = ['dotted','dashdot','dashed','solid']
    if color is None:
        color = 'black'
    if nHinds is None:
        nHinds = range(len(lognHvals))
        
    for i in range(len(nHinds)):
        ax.plot(10**logTvals, np.abs(lambda_over_nH2[nHinds[i]]), color=color, linestyle=linestyles[i]) 
    
def testWSS09fig2():
    '''
    Reproduce figure 2 of Wiersma, Schaye, & Smith 2009
    use dct method, and both total_metals and per_element methods
    '''
    logTvals = np.arange(4.,9.1,0.01)
    lognHvals = np.array([-6.,-4.,-2.,0.])
    lognHoffvals = lognHvals + 0.02 # make sure to interpolate

    lengrid = len(lognHvals)*len(logTvals)
    dct = {}
    
    # solar mass fraction values: from the cooling tables themselves
    dct['helium']    = np.ones(lengrid)*0.28055534
    dct['calcium']   = np.ones(lengrid)*6.4355E-5
    dct['carbon']    = np.ones(lengrid)*0.0020665436
    dct['iron']      = np.ones(lengrid)*0.0011032152
    dct['magnesium'] = np.ones(lengrid)*5.907064E-4
    dct['neon']      = np.ones(lengrid)*0.0014144605
    dct['nitrogen']  = np.ones(lengrid)*8.3562563E-4
    dct['oxygen']    = np.ones(lengrid)*0.0054926244
    dct['silicon']   = np.ones(lengrid)*6.825874E-4
    dct['sulfur']    = np.ones(lengrid)*4.0898522E-4
    
    dct['metallicity'] = np.ones(lengrid)*0.0129 #same as used in table interpolation
    
    # primoridal
#    dct['helium']    = np.ones(lengrid)*0.248
#    dct['calcium']   = np.zeros(lengrid)
#    dct['carbon']    = np.zeros(lengrid)
#    dct['iron']      = np.zeros(lengrid)
#    dct['magnesium'] = np.zeros(lengrid)
#    dct['neon']      = np.zeros(lengrid)
#    dct['nitrogen']  = np.zeros(lengrid)
#    dct['oxygen']    = np.zeros(lengrid)
#    dct['silicon']   = np.zeros(lengrid)
#    dct['sulfur']    = np.zeros(lengrid)
#    
#    dct['metallicity'] = np.zeros(lengrid)
    
    
    dct['logT'] = np.array((logTvals,)*len(lognHvals)).flatten()
    dctoff = dct.copy()
    dct['lognH']    = np.array([(val,)*len(logTvals) for val in lognHvals]).flatten()
    dctoff['lognH'] = np.array([(val,)*len(logTvals) for val in lognHoffvals]).flatten()
    
    dct['Density']    = 10**(dct['lognH']    - np.log10(0.70649785/(m3.c.atomw_H*m3.c.u)) ) # using solar hydogen fraction
    dctoff['Density'] = 10**(dctoff['lognH'] - np.log10(0.70649785/(m3.c.atomw_H*m3.c.u)) ) 
   
    outshape = (len(lognHvals), len(logTvals))
    lambda_over_nH2_perelt             = (m3.find_coolingrates(3.0,  dct, method = 'per_element')).reshape(outshape)
    lambda_over_nH2_perelt_nHoff      = (m3.find_coolingrates(3.0,  dctoff, method = 'per_element')).reshape(outshape)
    lambda_over_nH2_ztot              = (m3.find_coolingrates(3.0,  dct, method = 'total_metals')).reshape(outshape)
    lambda_over_nH2_ztot_nHoff        = (m3.find_coolingrates(3.0,  dctoff, method = 'total_metals')).reshape(outshape)
    lambda_over_nH2_perelt_zoff       = (m3.find_coolingrates(3.01, dct, method = 'per_element')).reshape(outshape)
    lambda_over_nH2_perelt_nHoff_zoff = (m3.find_coolingrates(3.01, dctoff, method = 'per_element')).reshape(outshape)
    lambda_over_nH2_ztot_zoff         = (m3.find_coolingrates(3.01, dct, method = 'total_metals')).reshape(outshape)
    lambda_over_nH2_ztot_nHoff_zoff   = (m3.find_coolingrates(3.01, dctoff, method = 'total_metals')).reshape(outshape)
    
    fig, ax = plt.subplots(ncols=1,nrows=1)
    fontsize = 12
    fig.suptitle("Replication of figure 2 of Wiersma, Schaye, & Smith (2009)", fontsize=fontsize+2)
    
    linestyles = ['dotted','dashdot','dashed','solid']
    
    plotWSS09fig2(ax, lambda_over_nH2_perelt, logTvals, lognHvals, color = 'red', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_perelt_nHoff, logTvals, lognHvals, color = 'firebrick', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_ztot, logTvals, lognHvals, color = 'orange', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_ztot_nHoff, logTvals, lognHvals, color = 'chocolate', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_perelt_zoff, logTvals, lognHvals, color = 'lime', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_perelt_nHoff_zoff, logTvals, lognHvals, color = 'forestgreen', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_ztot_zoff, logTvals, lognHvals, color = 'cyan', linestyles=linestyles)
    plotWSS09fig2(ax, lambda_over_nH2_ztot_nHoff_zoff, logTvals, lognHvals, color = 'blue', linestyles=linestyles)
    
    ax.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax.set_ylabel(r'$\left|\, \Lambda\, n_H^{-2} \, \right| \, [\mathrm{erg}\, \mathrm{cm}^3 \mathrm{s}^{-1}]$',fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.e-24,1.e-21)
    ax.set_xlim(1.e4,1.e9)
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    linestyle_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label=r'$\log_{10} n_{H} \cdot \mathrm{cm^{3}} = %.1f$'%(lognHvals[i])) for i in range(len(lognHvals))]
    color_legend_handles = []
    color_legend_handles += [mlines.Line2D([], [], color='red', linestyle = 'solid', label='per element (pe)')]
    color_legend_handles += [mlines.Line2D([], [], color='firebrick', linestyle = 'solid', label='pe, nH interp. (nHi)')]
    color_legend_handles += [mlines.Line2D([], [], color='orange', linestyle = 'solid', label='total metals (Zt)')]
    color_legend_handles += [mlines.Line2D([], [], color='chocolate', linestyle = 'solid', label='Zt, nHi')]
    color_legend_handles += [mlines.Line2D([], [], color='lime', linestyle = 'solid', label='pe, Z interp. (Zi)')]
    color_legend_handles += [mlines.Line2D([], [], color='forestgreen', linestyle = 'solid', label='pe, nHi ,Zi')]
    color_legend_handles += [mlines.Line2D([], [], color='cyan', linestyle = 'solid', label='Zt, Zi')]
    color_legend_handles += [mlines.Line2D([], [], color='blue', linestyle = 'solid', label='Zt, nHi, Zi')]

    ax.legend(handles=linestyle_legend_handles+color_legend_handles, fontsize=fontsize-2, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02)) #ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor
    
    plt.savefig(dd_cool+'WSS09fig2_replica.pdf', format='pdf', bbox_inches='tight')
    
    return lambda_over_nH2_perelt

def testWSS09fig2_vardict():
    '''
    Reproduce figure 2 of Wiersma, Schaye, & Smith 2009
    use Vardict method, and both total_metals and per_element methods
    '''       
    # solar Z: use Pt abundances
    simfile = sfc.Simfileclone(sfc.Zvar_rhoT_z3)
    vardict = m3.Vardict(simfile, 0, ['lognH', 'logT', 'ElementAbundance/Sulfur', 'ElementAbundance/Helium', 'ElementAbundance/Hydrogen', 'ElementAbundance/Oxygen']) # saved to check values; earlier runs show deletion when required
    print('Initially read in: %s'%(vardict.particle.keys()))

    logTvals   = np.log10(sfc.Tvals)
    logrhovals = np.log10(sfc.rhovals * m3.c.unitdensity_in_cgs)
    lognHvals  = logrhovals + np.log10( sfc.dct_sol['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    targetnHvals = np.array([-6., -4., -2., 0.])
    nHinds = np.array([np.argmin(np.abs(lognHvals - tarval)) for tarval in targetnHvals])

    outshape = (len(lognHvals), len(logTvals))
    lambda_over_nH2_perelt = (m3.find_coolingrates(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    lambda_over_nH2_ztot   = (m3.find_coolingrates(vardict.simfile.z,  vardict, method = 'total_metals', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    
    
    fig, ax = plt.subplots(ncols=1,nrows=1)
    fontsize = 12
    fig.suptitle("Replication of figure 2 of Wiersma, Schaye, & Smith (2009)", fontsize=fontsize+2)
    
    linestyles = ['dotted','dashdot','dashed','solid']
    
    plotWSS09fig2(ax, lambda_over_nH2_perelt, logTvals, lognHvals, color = 'red', linestyles=linestyles, nHinds=nHinds)
    plotWSS09fig2(ax, lambda_over_nH2_ztot, logTvals, lognHvals, color = 'blue', linestyles=linestyles, nHinds=nHinds)
    
    ax.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax.set_ylabel(r'$\left|\, \Lambda\, n_H^{-2} \, \right| \, [\mathrm{erg}\, \mathrm{cm}^3 \mathrm{s}^{-1}]$',fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.e-24,1.e-21)
    ax.set_xlim(1.e4,1.e9)
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    linestyle_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label=r'$\log_{10} n_{H} \cdot \mathrm{cm^{3}} = %.3f$'%(lognHvals[nHinds[i]])) for i in range(len(nHinds))]
    color_legend_handles = []
    color_legend_handles += [mlines.Line2D([], [], color='red', linestyle = 'solid', label='per element')]
    color_legend_handles += [mlines.Line2D([], [], color='blue', linestyle = 'solid', label='total metals')]

    ax.legend(handles=linestyle_legend_handles+color_legend_handles, fontsize=fontsize-2, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02)) #ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor
    
    plt.savefig(dd_cool+'WSS09fig2_replica_vardict.pdf', format='pdf', bbox_inches='tight')
    
    return vardict, lambda_over_nH2_perelt, lambda_over_nH2_ztot

def testWSS09fig6():
    '''
    Reproduce figure 6 of Wiersma, Schaye, & Smith 2009
    except that this sis CIE + PIE, figure 6 has CIE only, PIE only and not total metals contribution 
    '''
   
    lambda_over_nH2_perelt            = m3.findcoolingtables(3.00, method = 'per_element')
    lambda_over_nH2_perelt_zoff       = m3.findcoolingtables(3.01, method = 'per_element')
    lambda_over_nH2_ztot            = m3.findcoolingtables(3.00, method = 'total_metals')
    lambda_over_nH2_ztot_zoff       = m3.findcoolingtables(3.01, method = 'total_metals')

    nHind = np.where(lambda_over_nH2_perelt['Metal_free']['lognHcm3'] == -4.)[0][0] # there should be an exact match
    Heind = 4 # closest match
    print("lognH: 10^%.1f"%(lambda_over_nH2_perelt['Metal_free']['lognHcm3'][nHind]))
    T = 10**(lambda_over_nH2_perelt['Metal_free']['logTK'])
    fig, ax = plt.subplots(ncols=1,nrows=1)
    fontsize = 12
    fig.suptitle("Replication of figure 6 CIE/PIE of Wiersma, Schaye, & Smith (2009): Cooling times", fontsize=fontsize+2)

    
    linestyle = 'solid'
    dct = lambda_over_nH2_perelt
    dct_tot = lambda_over_nH2_ztot
    ax.plot(T,dct['Carbon']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'red', label = 'C')
    ax.plot(T,dct['Silicon']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'indigo', label = 'Si')
    ax.plot(T,dct['Sulphur']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'cyan', label = 'S')
    ax.plot(T,dct['Oxygen']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'purple', label = 'O')
    ax.plot(T,dct['Nitrogen']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'blue', label = 'N')
    ax.plot(T,dct['Calcium']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'limegreen', label = 'Ca')
    ax.plot(T,dct['Neon']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'darkcyan', label = 'Ne')
    ax.plot(T,dct['Magnesium']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'fuchsia', label = 'Mg')
    ax.plot(T,dct['Iron']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'orange', label = 'Fe')
    ax.plot(T,dct['Metal_free']['Lambda_over_nH2'][Heind,:,nHind], linestyle = 'dashed', color= 'gray', label = 'H & He')
    ax.plot(T,dct_tot['Total_Metals']['Lambda_over_nH2'][:,nHind], linestyle = 'dashed', color= 'brown', label = 'Ztot')
    
    Zsum = dct['Carbon']['Lambda_over_nH2'][:,nHind] + dct['Silicon']['Lambda_over_nH2'][:,nHind] +\
           dct['Sulphur']['Lambda_over_nH2'][:,nHind] + dct['Oxygen']['Lambda_over_nH2'][:,nHind] +\
           dct['Nitrogen']['Lambda_over_nH2'][:,nHind] + dct['Calcium']['Lambda_over_nH2'][:,nHind] +\
           dct['Neon']['Lambda_over_nH2'][:,nHind] + dct['Magnesium']['Lambda_over_nH2'][:,nHind] +\
           dct['Iron']['Lambda_over_nH2'][:,nHind]
    ax.plot(T, Zsum, linestyle = 'dashed', color= 'tan', label = 'Zsum')
    ax.plot(T, Zsum+dct['Metal_free']['Lambda_over_nH2'][Heind,:,nHind], linestyle = 'solid', color= 'black', label = 'total')

    linestyle = 'dashdot'
    dct = lambda_over_nH2_perelt_zoff
    dct_tot = lambda_over_nH2_ztot_zoff
    ax.plot(T,dct['Carbon']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'red', label=None)
    ax.plot(T,dct['Silicon']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'indigo',  label=None)
    ax.plot(T,dct['Sulphur']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'cyan',  label=None)
    ax.plot(T,dct['Oxygen']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'purple',  label=None)
    ax.plot(T,dct['Nitrogen']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'blue',  label=None)
    ax.plot(T,dct['Calcium']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'limegreen',  label=None)
    ax.plot(T,dct['Neon']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'darkcyan',  label=None)
    ax.plot(T,dct['Magnesium']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'fuchsia',  label=None)
    ax.plot(T,dct['Iron']['Lambda_over_nH2'][:,nHind], linestyle = linestyle, color= 'orange',  label=None)
    ax.plot(T,dct['Metal_free']['Lambda_over_nH2'][Heind,:,nHind], linestyle = 'dotted', color= 'gray',  label=None)
    ax.plot(T,dct_tot['Total_Metals']['Lambda_over_nH2'][:,nHind], linestyle = 'dotted', color= 'brown',  label=None)
    
    Zsum = dct['Carbon']['Lambda_over_nH2'][:,nHind] + dct['Silicon']['Lambda_over_nH2'][:,nHind] +\
           dct['Sulphur']['Lambda_over_nH2'][:,nHind] + dct['Oxygen']['Lambda_over_nH2'][:,nHind] +\
           dct['Nitrogen']['Lambda_over_nH2'][:,nHind] + dct['Calcium']['Lambda_over_nH2'][:,nHind] +\
           dct['Neon']['Lambda_over_nH2'][:,nHind] + dct['Magnesium']['Lambda_over_nH2'][:,nHind] +\
           dct['Iron']['Lambda_over_nH2'][:,nHind]
    ax.plot(T, Zsum, linestyle = 'dotted', color= 'tan',  label=None)
    ax.plot(T, Zsum+dct['Metal_free']['Lambda_over_nH2'][Heind,:,nHind], linestyle = 'solid', color= 'black',  label=None)

    ax.legend(fontsize=fontsize)
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
     
    ax.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax.set_ylabel(r'$\left|\, \Lambda\, n_H^{-2} \, \right| \, [\mathrm{erg}\, \mathrm{cm}^3 \mathrm{s}^{-1}]$',fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.e-24,1.e-21)
    ax.set_xlim(1.e4,1.e8)
    
    plt.savefig(dd_cool+'WSS09fig6_replica.pdf', format='pdf', bbox_inches='tight')
    

def plotWSS09fig3(ax, lambda_over_nH2, logTvals, lognHvals, levels=None, colors=None, linestyle=None):
    if linestyle is None:
        linestyle = 'solid'
    if colors is None:
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']
    if levels is None:
        levels = 10**np.array([5.,6.,7.,8.,9.,10.]) # in years
    
    ax.contour(10**logTvals, 10**lognHvals, np.abs(lambda_over_nH2)/m3.c.sec_per_year, levels, colors=colors, linestyle=linestyle) 
        
def testWSS09fig3():
    '''
    Reproduce figure 3 of Wiersma, Schaye, & Smith 2009
    use dct method, and both total_metals and per_element methods
    specifically, left panels: HM01, z=3, primordial and solar abundances
    '''
    logTvals = np.arange(4.,9.1,0.02)
    lognHvals = np.arange(-6.,-0.8, 0.02)
    lognHoffvals = lognHvals + 0.02 # make sure to interpolate

    lengrid = len(lognHvals)*len(logTvals)
    dct_sol = {}
    dct_pri = {}
    
    # solar mass fraction values: from the cooling tables themselves
    dct_sol['helium']    = np.ones(lengrid)*0.28055534
    dct_sol['calcium']   = np.ones(lengrid)*6.4355E-5
    dct_sol['carbon']    = np.ones(lengrid)*0.0020665436
    dct_sol['iron']      = np.ones(lengrid)*0.0011032152
    dct_sol['magnesium'] = np.ones(lengrid)*5.907064E-4
    dct_sol['neon']      = np.ones(lengrid)*0.0014144605
    dct_sol['nitrogen']  = np.ones(lengrid)*8.3562563E-4
    dct_sol['oxygen']    = np.ones(lengrid)*0.0054926244
    dct_sol['silicon']   = np.ones(lengrid)*6.825874E-4
    dct_sol['sulfur']    = np.ones(lengrid)*4.0898522E-4
    
    dct_sol['metallicity'] = np.ones(lengrid)*0.0129 #same as used in table interpolation
    
    # primoridal
    dct_pri['helium']    = np.ones(lengrid)*0.248
    dct_pri['calcium']   = np.zeros(lengrid)
    dct_pri['carbon']    = np.zeros(lengrid)
    dct_pri['iron']      = np.zeros(lengrid)
    dct_pri['magnesium'] = np.zeros(lengrid)
    dct_pri['neon']      = np.zeros(lengrid)
    dct_pri['nitrogen']  = np.zeros(lengrid)
    dct_pri['oxygen']    = np.zeros(lengrid)
    dct_pri['silicon']   = np.zeros(lengrid)
    dct_pri['sulfur']    = np.zeros(lengrid)
    
    dct_pri['metallicity'] = np.zeros(lengrid)
    
    
    dct_sol['logT'] = np.array((logTvals,)*len(lognHvals)).flatten()
    dct_pri['logT'] = np.array((logTvals,)*len(lognHvals)).flatten()
    dct_sol_off = dct_sol.copy()
    dct_pri_off = dct_pri.copy()
    dct_sol['lognH']    = np.array([(val,)*len(logTvals) for val in lognHvals]).flatten()
    dct_pri['lognH']    = np.array([(val,)*len(logTvals) for val in lognHvals]).flatten()
    dct_sol_off['lognH'] = np.array([(val,)*len(logTvals) for val in lognHoffvals]).flatten()
    dct_pri_off['lognH'] = np.array([(val,)*len(logTvals) for val in lognHoffvals]).flatten()
    
    dct_sol['Density']    = 10**(dct_sol['lognH']    - np.log10(0.70649785/(m3.c.atomw_H*m3.c.u)) ) # using solar hydogen fraction
    dct_pri['Density']    = 10**(dct_pri['lognH']    - np.log10(0.752/(m3.c.atomw_H*m3.c.u)) ) # using primordial hydogen fraction
    dct_sol_off['Density'] = 10**(dct_sol_off['lognH'] - np.log10(0.70649785/(m3.c.atomw_H*m3.c.u)) ) 
    dct_pri_off['Density'] = 10**(dct_pri_off['lognH'] - np.log10(0.752/(m3.c.atomw_H*m3.c.u)) ) 
   
    outshape = (len(lognHvals), len(logTvals))
    tcool_perelt_sol            = (m3.find_coolingtimes(3.0,  dct_sol, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_sol      = (m3.find_coolingtimes(3.0,  dct_sol_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_sol              = (m3.find_coolingtimes(3.0,  dct_sol, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_sol        = (m3.find_coolingtimes(3.0,  dct_sol_off, method = 'total_metals')).reshape(outshape)
    tcool_perelt_zoff_sol       = (m3.find_coolingtimes(3.01, dct_sol, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_zoff_sol = (m3.find_coolingtimes(3.01, dct_sol_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_zoff_sol         = (m3.find_coolingtimes(3.01, dct_sol, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_zoff_sol   = (m3.find_coolingtimes(3.01, dct_sol_off, method = 'total_metals')).reshape(outshape)
    
    tcool_perelt_pri            = (m3.find_coolingtimes(3.0,  dct_pri, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_pri      = (m3.find_coolingtimes(3.0,  dct_pri_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_pri              = (m3.find_coolingtimes(3.0,  dct_pri, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_pri        = (m3.find_coolingtimes(3.0,  dct_pri_off, method = 'total_metals')).reshape(outshape)
    tcool_perelt_zoff_pri       = (m3.find_coolingtimes(3.01, dct_pri, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_zoff_pri = (m3.find_coolingtimes(3.01, dct_pri_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_zoff_pri         = (m3.find_coolingtimes(3.01, dct_pri, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_zoff_pri   = (m3.find_coolingtimes(3.01, dct_pri_off, method = 'total_metals')).reshape(outshape)
    
    fig, (ax1, ax2) = plt.subplots(ncols=1,nrows=2, figsize=(5.,8.))
    fontsize = 12
    fig.suptitle("Replication of figure 3 of Wiersma, Schaye, & Smith (2009)", fontsize=fontsize+2)
    
    linestyles = ['solid','dashed', 'dotted','dashdot']
    linestylelabels = ['per element (pe)', 'pe, $n_{H}$ interp. (ni)', 'total metals (Zt)', 'Zt, ni']
    colors1 = ['black', 'red', 'blue', 'green', 'orange', 'purple']
    colors2 = ['gray', 'firebrick', 'cyan', 'lime', 'salmon', 'mediumorchid']
    levels = 10**np.array([5.,6.,7.,8.,9.,10.])
    
    plotWSS09fig3(ax2, tcool_perelt_sol, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[0])
    plotWSS09fig3(ax2, tcool_perelt_nHoff_sol, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[0])
    plotWSS09fig3(ax2, tcool_ztot_sol, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[1])
    plotWSS09fig3(ax2, tcool_ztot_nHoff_sol, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[1])
    plotWSS09fig3(ax2, tcool_perelt_zoff_sol, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[2])
    plotWSS09fig3(ax2, tcool_perelt_nHoff_zoff_sol, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[2])
    plotWSS09fig3(ax2, tcool_ztot_zoff_sol, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[3])
    plotWSS09fig3(ax2, tcool_ztot_nHoff_zoff_sol, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[3])
    
    plotWSS09fig3(ax1, tcool_perelt_pri, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[0])
    plotWSS09fig3(ax1, tcool_perelt_nHoff_pri, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[0])
    plotWSS09fig3(ax1, tcool_ztot_pri, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[1])
    plotWSS09fig3(ax1, tcool_ztot_nHoff_pri, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[1])
    plotWSS09fig3(ax1, tcool_perelt_zoff_pri, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[2])
    plotWSS09fig3(ax1, tcool_perelt_nHoff_zoff_pri, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[2])
    plotWSS09fig3(ax1, tcool_ztot_zoff_pri, logTvals, lognHvals, levels=levels, colors=colors1, linestyle=linestyles[3])
    plotWSS09fig3(ax1, tcool_ztot_nHoff_zoff_pri, logTvals, lognHvals, levels=levels, colors=colors2, linestyle=linestyles[3])

    
    ax1.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax1.set_ylabel(r'$n_H \, [\mathrm{cm}^{-3}]$',fontsize=fontsize)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(1.e-6,1.e-1)
    ax1.set_xlim(1.e4,1.e8)
    ax1.set_title('$Z=0$, HM01 $z=3$')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    ax2.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax2.set_ylabel(r'$n_H \, [\mathrm{cm}^{-3}]$',fontsize=fontsize)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylim(1.e-6,1.e-1)
    ax2.set_xlim(1.e4,1.e8)
    ax2.set_title('$Z=Z_{\odot}$, HM01 $z=3$')
    
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    linestyle_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label=linestylelabels[i]) for i in range(len(linestyles))]

    color1_legend_handles = []
    color1_legend_handles += [mlines.Line2D([], [], color=colors1[0], linestyle = 'solid', label='$%.0e$ yr'%levels[0])]
    color1_legend_handles += [mlines.Line2D([], [], color=colors1[i], linestyle = 'solid', label='$%.0e$ yr'%levels[i]) for i in range(1, len(levels))]
    
    color2_legend_handles = []
    color2_legend_handles += [mlines.Line2D([], [], color=colors2[0], linestyle = 'solid', label='$%.0e$ yr, $n_H$ interp. (ni)'%levels[0])]
    color2_legend_handles += [mlines.Line2D([], [], color=colors2[i], linestyle = 'solid', label='$%.0e$ yr, ni'%levels[i]) for i in range(1, len(levels))]
    

    ax1.legend(handles=linestyle_legend_handles+color1_legend_handles, fontsize=fontsize-2, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02)) #ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor
    ax2.legend(handles=color2_legend_handles, fontsize=fontsize-2, ncol=1, loc='lower right', bbox_to_anchor=(0.98,0.02))
    
    plt.savefig(dd_cool+'WSS09fig3_replica.pdf', format='pdf', bbox_inches='tight')
    
    return tcool_perelt_sol


def testWSS09fig3_vardict():
    '''
    Reproduce figure 3 of Wiersma, Schaye, & Smith 2009
    use dct method, and both total_metals and per_element methods
    specifically, left panels: HM01, z=3, primordial and solar abundances
    '''
    simfile = sfc.Simfileclone(sfc.Zvar_rhoT_z3)
    vardict = m3.Vardict(simfile, 0, ['Mass']) # Mass should not be read in, all other entries should be cleaned up
    print('Initially read in: %s'%(vardict.particle.keys()))

    logTvals   = np.log10(sfc.Tvals)
    logrhovals = np.log10(sfc.rhovals * m3.c.unitdensity_in_cgs)
    lognHvals_sol  = logrhovals + np.log10( sfc.dct_sol['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    lognHvals_pri  = logrhovals + np.log10( sfc.dct_pri['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    
    # in simflieclone.Zvar_rhoT_z3, smoothed metallicites are primoridial and particle metallicites are solar
    outshape = (len(lognHvals_sol), len(logTvals))
    tcool_perelt_sol = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    tcool_ztot_sol   = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'total_metals', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    tcool_perelt_pri = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
    tcool_ztot_pri   = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'total_metals', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
    
    fig, (ax1, ax2) = plt.subplots(ncols=1,nrows=2, figsize=(5.,8.))
    fontsize = 12
    fig.suptitle("Replication of figure 3 of Wiersma, Schaye, & Smith (2009)", fontsize=fontsize+2)
    
    linestyles = ['solid','dashed']
    #linestylelabels = ['per element', 'total metals']
    colors1 = ['black', 'red', 'blue', 'green', 'orange', 'purple']
    colors2 = ['gray', 'firebrick', 'cyan', 'lime', 'salmon', 'mediumorchid']
    levels = 10**np.array([5.,6.,7.,8.,9.,10.])
    
    plotWSS09fig3(ax2, tcool_perelt_sol, logTvals, lognHvals_sol, levels=levels, colors=colors1, linestyle=linestyles[0])
    plotWSS09fig3(ax2, tcool_ztot_sol, logTvals, lognHvals_sol, levels=levels, colors=colors2, linestyle=linestyles[1])

    plotWSS09fig3(ax1, tcool_perelt_pri, logTvals, lognHvals_pri, levels=levels, colors=colors1, linestyle=linestyles[0])
    plotWSS09fig3(ax1, tcool_ztot_pri, logTvals, lognHvals_pri, levels=levels, colors=colors2, linestyle=linestyles[1])
    
    ax1.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax1.set_ylabel(r'$n_H \, [\mathrm{cm}^{-3}]$',fontsize=fontsize)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(1.e-6,1.e-1)
    ax1.set_xlim(1.e4,1.e8)
    ax1.set_title('$Z=0$, HM01 $z=3$')
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    ax2.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax2.set_ylabel(r'$n_H \, [\mathrm{cm}^{-3}]$',fontsize=fontsize)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylim(1.e-6,1.e-1)
    ax2.set_xlim(1.e4,1.e8)
    ax2.set_title('$Z=Z_{\odot}$, HM01 $z=3$')
    
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')
    
    #linestyle_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label=linestylelabels[i]) for i in range(len(linestyles))]
    color1_legend_handles = []
    color1_legend_handles += [mlines.Line2D([], [], color=colors1[0], linestyle = 'solid', label='$%.0e$ yr'%levels[0])]
    color1_legend_handles += [mlines.Line2D([], [], color=colors1[i], linestyle = 'solid', label='$%.0e$ yr'%levels[i]) for i in range(1, len(levels))]
    
    color2_legend_handles = []
    color2_legend_handles += [mlines.Line2D([], [], color=colors2[0], linestyle = 'solid', label='$%.0e$ yr'%levels[0])]
    color2_legend_handles += [mlines.Line2D([], [], color=colors2[i], linestyle = 'solid', label='$%.0e$ yr'%levels[i]) for i in range(1, len(levels))]

    ax1.legend(handles=color1_legend_handles, fontsize=fontsize-2, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02), title='per element') #ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor
    ax2.legend(handles=color2_legend_handles, fontsize=fontsize-2, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02), title='total metals') 
    
    plt.savefig(dd_cool+'WSS09fig3_replica_vardict.pdf', format='pdf', bbox_inches='tight')
    
    return vardict, tcool_perelt_sol, tcool_ztot_pri


def testWSS09fig4():
    '''
    Reproduce figure 4 of Wiersma, Schaye, & Smith 2009
    use dct method, and both total_metals and per_element methods
    specifically, HM01, z=0
    '''
    logTvals = np.arange(4.,8.1,0.02)
    lognHvals = np.arange(-8.,-2., 0.02)
    lognHoffvals = lognHvals + 0.02 # make sure to interpolate
    logdeltavals = lognHvals +np.log10( (m3.c.atomw_H*m3.c.u)/0.752 / ( 3./(8.*np.pi*m3.c.gravity)*m3.c.hubble**2 * m3.c.hubbleparam**2 * m3.c.omegabaryon) )#for plotting: nH -> rho (primordial) /rhob

    lengrid = len(lognHvals)*len(logTvals)
    dct_sol = {}
    dct_pri = {}
    dct_0p1 = {}
    
    # solar mass fraction values: from the cooling tables themselves
    dct_sol['helium']    = np.ones(lengrid)*0.28055534
    dct_sol['calcium']   = np.ones(lengrid)*6.4355E-5
    dct_sol['carbon']    = np.ones(lengrid)*0.0020665436
    dct_sol['iron']      = np.ones(lengrid)*0.0011032152
    dct_sol['magnesium'] = np.ones(lengrid)*5.907064E-4
    dct_sol['neon']      = np.ones(lengrid)*0.0014144605
    dct_sol['nitrogen']  = np.ones(lengrid)*8.3562563E-4
    dct_sol['oxygen']    = np.ones(lengrid)*0.0054926244
    dct_sol['silicon']   = np.ones(lengrid)*6.825874E-4
    dct_sol['sulfur']    = np.ones(lengrid)*4.0898522E-4
    
    dct_sol['metallicity'] = np.ones(lengrid)*0.0129 #same as used in table interpolation
    
    # 0.1*solar; helium: primordial + 0.1*(solar-primordial) (guess)
    dct_0p1['helium']    = np.ones(lengrid)*0.248 + 0.1*(0.28055534-0.248)
    dct_0p1['calcium']   = np.ones(lengrid)*0.1*6.4355E-5
    dct_0p1['carbon']    = np.ones(lengrid)*0.1*0.0020665436
    dct_0p1['iron']      = np.ones(lengrid)*0.1*0.0011032152
    dct_0p1['magnesium'] = np.ones(lengrid)*0.1*5.907064E-4
    dct_0p1['neon']      = np.ones(lengrid)*0.1*0.0014144605
    dct_0p1['nitrogen']  = np.ones(lengrid)*0.1*8.3562563E-4
    dct_0p1['oxygen']    = np.ones(lengrid)*0.1*0.0054926244
    dct_0p1['silicon']   = np.ones(lengrid)*0.1*6.825874E-4
    dct_0p1['sulfur']    = np.ones(lengrid)*0.1*4.0898522E-4
    
    dct_0p1['metallicity'] = np.ones(lengrid)*0.1*0.0129 
    
    # primoridal
    dct_pri['helium']    = np.ones(lengrid)*0.248
    dct_pri['calcium']   = np.zeros(lengrid)
    dct_pri['carbon']    = np.zeros(lengrid)
    dct_pri['iron']      = np.zeros(lengrid)
    dct_pri['magnesium'] = np.zeros(lengrid)
    dct_pri['neon']      = np.zeros(lengrid)
    dct_pri['nitrogen']  = np.zeros(lengrid)
    dct_pri['oxygen']    = np.zeros(lengrid)
    dct_pri['silicon']   = np.zeros(lengrid)
    dct_pri['sulfur']    = np.zeros(lengrid)
    
    dct_pri['metallicity'] = np.zeros(lengrid)
    
    
    dct_sol['logT'] = np.array((logTvals,)*len(lognHvals)).flatten()
    dct_0p1['logT'] = np.array((logTvals,)*len(lognHvals)).flatten()
    dct_pri['logT'] = np.array((logTvals,)*len(lognHvals)).flatten()
    dct_sol_off = dct_sol.copy()
    dct_0p1_off = dct_0p1.copy()
    dct_pri_off = dct_pri.copy()
    dct_sol['lognH']    = np.array([(val,)*len(logTvals) for val in lognHvals]).flatten()
    dct_0p1['lognH']    = np.array([(val,)*len(logTvals) for val in lognHvals]).flatten()
    dct_pri['lognH']    = np.array([(val,)*len(logTvals) for val in lognHvals]).flatten()
    dct_sol_off['lognH'] = np.array([(val,)*len(logTvals) for val in lognHoffvals]).flatten()
    dct_0p1_off['lognH'] = np.array([(val,)*len(logTvals) for val in lognHoffvals]).flatten()
    dct_pri_off['lognH'] = np.array([(val,)*len(logTvals) for val in lognHoffvals]).flatten()
    
    dct_sol['Density']    = 10**(dct_sol['lognH']    - np.log10(0.70649785/(m3.c.atomw_H*m3.c.u)) ) # using solar hydogen fraction
    dct_0p1['Density']    = 10**(dct_0p1['lognH']    - np.log10((0.752 + 0.1*(0.70649785-0.752))/(m3.c.atomw_H*m3.c.u)) ) # interpolating solar-primordial hydrogen fraction
    dct_pri['Density']    = 10**(dct_pri['lognH']    - np.log10(0.752/(m3.c.atomw_H*m3.c.u)) ) # using primordial hydogen fraction
    dct_sol_off['Density'] = 10**(dct_sol_off['lognH'] - np.log10(0.70649785/(m3.c.atomw_H*m3.c.u)) ) 
    dct_0p1_off['Density'] = 10**(dct_0p1_off['lognH'] - np.log10((0.752 + 0.1*(0.70649785-0.752))/(m3.c.atomw_H*m3.c.u)) ) 
    dct_pri_off['Density'] = 10**(dct_pri_off['lognH'] - np.log10(0.752/(m3.c.atomw_H*m3.c.u)) ) 
   
    outshape = (len(lognHvals), len(logTvals))
    tcool_perelt_sol            = (m3.find_coolingtimes(0.0,  dct_sol, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_sol      = (m3.find_coolingtimes(0.0,  dct_sol_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_sol              = (m3.find_coolingtimes(0.0,  dct_sol, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_sol        = (m3.find_coolingtimes(0.0,  dct_sol_off, method = 'total_metals')).reshape(outshape)
    tcool_perelt_zoff_sol       = (m3.find_coolingtimes(0.01, dct_sol, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_zoff_sol = (m3.find_coolingtimes(0.01, dct_sol_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_zoff_sol         = (m3.find_coolingtimes(0.01, dct_sol, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_zoff_sol   = (m3.find_coolingtimes(0.01, dct_sol_off, method = 'total_metals')).reshape(outshape)
    
    tcool_perelt_0p1             = (m3.find_coolingtimes(0.0,  dct_0p1, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_0p1      = (m3.find_coolingtimes(0.0,  dct_0p1_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_0p1              = (m3.find_coolingtimes(0.0,  dct_0p1, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_0p1        = (m3.find_coolingtimes(0.0,  dct_0p1_off, method = 'total_metals')).reshape(outshape)
    tcool_perelt_zoff_0p1       = (m3.find_coolingtimes(0.01, dct_0p1, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_zoff_0p1 = (m3.find_coolingtimes(0.01, dct_0p1_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_zoff_0p1         = (m3.find_coolingtimes(0.01, dct_0p1, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_zoff_0p1   = (m3.find_coolingtimes(0.01, dct_0p1_off, method = 'total_metals')).reshape(outshape)
    
    tcool_perelt_pri            = (m3.find_coolingtimes(0.0,  dct_pri, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_pri      = (m3.find_coolingtimes(0.0,  dct_pri_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_pri              = (m3.find_coolingtimes(0.0,  dct_pri, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_pri        = (m3.find_coolingtimes(0.0,  dct_pri_off, method = 'total_metals')).reshape(outshape)
    tcool_perelt_zoff_pri       = (m3.find_coolingtimes(0.01, dct_pri, method = 'per_element')).reshape(outshape)
    tcool_perelt_nHoff_zoff_pri = (m3.find_coolingtimes(0.01, dct_pri_off, method = 'per_element')).reshape(outshape)
    tcool_ztot_zoff_pri         = (m3.find_coolingtimes(0.01, dct_pri, method = 'total_metals')).reshape(outshape)
    tcool_ztot_nHoff_zoff_pri   = (m3.find_coolingtimes(0.01, dct_pri_off, method = 'total_metals')).reshape(outshape)
    
    fig, ax = plt.subplots(ncols=1,nrows=1)
    fontsize = 12
    fig.suptitle("Replication of figure 4 of Wiersma, Schaye, & Smith (2009), HM01, $z=0$, $t_{\mathrm{cool}} = t_{H}$", fontsize=fontsize+2)
    
    linestyles = ['solid','dashed', 'dotted','dashdot']
    linestylelabels = ['per element (pe)', 'total metals (Zt)', 'pe, $z$ interp. (zi)', 'Zt, zi']
    colors1 = [['black'], ['red'], ['blue']]
    colors2 = [['gray'], ['firebrick'], ['cyan']]
    colorlabels = [r'$Z=Z_{\odot}$', r'$Z=0.1Z_{\odot}$', r'$Z=0$']
    levels = [1./m3.Hubble(0.)/m3.c.sec_per_year] #default cosmology: EAGLE (1/H = 14.4 Gyr, Planck 2014 age of universe = 13.8 Gyr)
    
    plotWSS09fig3(ax, tcool_perelt_sol, logTvals, logdeltavals, levels=levels, colors=colors1[0], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_perelt_nHoff_sol, logTvals, logdeltavals, levels=levels, colors=colors2[0], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_ztot_sol, logTvals, logdeltavals, levels=levels, colors=colors1[0], linestyle=linestyles[1])
    plotWSS09fig3(ax, tcool_ztot_nHoff_sol, logTvals, logdeltavals, levels=levels, colors=colors2[0], linestyle=linestyles[1])
    plotWSS09fig3(ax, tcool_perelt_zoff_sol, logTvals, logdeltavals, levels=levels, colors=colors1[0], linestyle=linestyles[2])
    plotWSS09fig3(ax, tcool_perelt_nHoff_zoff_sol, logTvals, logdeltavals, levels=levels, colors=colors2[0], linestyle=linestyles[2])
    plotWSS09fig3(ax, tcool_ztot_zoff_sol, logTvals, logdeltavals, levels=levels, colors=colors1[0], linestyle=linestyles[3])
    plotWSS09fig3(ax, tcool_ztot_nHoff_zoff_sol, logTvals, logdeltavals, levels=levels, colors=colors2[0], linestyle=linestyles[3])
    
    plotWSS09fig3(ax, tcool_perelt_0p1, logTvals, logdeltavals, levels=levels, colors=colors1[1], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_perelt_nHoff_0p1, logTvals, logdeltavals, levels=levels, colors=colors2[1], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_ztot_0p1, logTvals, logdeltavals, levels=levels, colors=colors1[1], linestyle=linestyles[1])
    plotWSS09fig3(ax, tcool_ztot_nHoff_0p1, logTvals, logdeltavals, levels=levels, colors=colors2[1], linestyle=linestyles[1])
    plotWSS09fig3(ax, tcool_perelt_zoff_0p1, logTvals, logdeltavals, levels=levels, colors=colors1[1], linestyle=linestyles[2])
    plotWSS09fig3(ax, tcool_perelt_nHoff_zoff_0p1, logTvals, logdeltavals, levels=levels, colors=colors2[1], linestyle=linestyles[2])
    plotWSS09fig3(ax, tcool_ztot_zoff_0p1, logTvals, logdeltavals, levels=levels, colors=colors1[1], linestyle=linestyles[3])
    plotWSS09fig3(ax, tcool_ztot_nHoff_zoff_0p1, logTvals, logdeltavals, levels=levels, colors=colors2[1], linestyle=linestyles[3])
    
    plotWSS09fig3(ax, tcool_perelt_pri, logTvals, logdeltavals, levels=levels, colors=colors1[2], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_perelt_nHoff_pri, logTvals, logdeltavals, levels=levels, colors=colors2[2], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_ztot_pri, logTvals, logdeltavals, levels=levels, colors=colors1[2], linestyle=linestyles[1])
    plotWSS09fig3(ax, tcool_ztot_nHoff_pri, logTvals, logdeltavals, levels=levels, colors=colors2[2], linestyle=linestyles[1])
    plotWSS09fig3(ax, tcool_perelt_zoff_pri, logTvals, logdeltavals, levels=levels, colors=colors1[2], linestyle=linestyles[2])
    plotWSS09fig3(ax, tcool_perelt_nHoff_zoff_pri, logTvals, logdeltavals, levels=levels, colors=colors2[2], linestyle=linestyles[2])
    plotWSS09fig3(ax, tcool_ztot_zoff_pri, logTvals, logdeltavals, levels=levels, colors=colors1[2], linestyle=linestyles[3])
    plotWSS09fig3(ax, tcool_ztot_nHoff_zoff_pri, logTvals, logdeltavals, levels=levels, colors=colors2[2], linestyle=linestyles[3])

    
    ax.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax.set_ylabel(r'$ 1 + \delta$',fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.e-1,1.e4)
    ax.set_xlim(1.e4,1.e8)
    ax.set_title('$Z=0$, HM01 $z=3$')
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')

    linestyle_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label=linestylelabels[i]) for i in range(len(linestyles))]

    color1_legend_handles =  [mlines.Line2D([], [], color=colors1[i][0], linestyle = 'solid', label=colorlabels[i]) for i in range(len(colors1))]
    color2_legend_handles =  [mlines.Line2D([], [], color=colors2[0][0], linestyle = 'solid', label=colorlabels[0] + r'$, n_{H}$ interp. (ni)')]
    color2_legend_handles += [mlines.Line2D([], [], color=colors2[i][0], linestyle = 'solid', label=colorlabels[i] + r', ni') for i in range(1,len(colors2))]
    
    ax.legend(handles=linestyle_legend_handles+color1_legend_handles+color2_legend_handles, fontsize=fontsize-2, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02)) #ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor
    
    plt.savefig(dd_cool+'WSS09fig4_replica.pdf', format='pdf', bbox_inches='tight')
    
    return tcool_perelt_sol


def testWSS09fig4_vardict():
    '''
    Reproduce figure 4 of Wiersma, Schaye, & Smith 2009
    use dct method, and both total_metals and per_element methods
    specifically, HM01, z=0
    '''
    simfile = sfc.Simfileclone(sfc.Zvar_rhoT_z0)
    vardict = m3.Vardict(simfile, 0, ['Mass', 'Density']) # Mass should not be read in, all other entries should be cleaned up
    print('Initially read in: %s'%(vardict.particle.keys()))

    logTvals   = np.log10(sfc.Tvals)
    logrhovals = np.log10(sfc.rhovals * m3.c.unitdensity_in_cgs)
    #lognHvals_sol  = logrhovals + np.log10( sfc.dct_sol['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    #lognHvals_0p1  = logrhovals + np.log10( sfc.dct_0p1['Hydrogen']    / (m3.c.u*m3.c.atomw_H) )
    #lognHvals_pri  = logrhovals + np.log10( sfc.dct_pri['Hydrogen'][0] / (m3.c.u*m3.c.atomw_H) ) 
    
    logdeltavals = logrhovals - np.log10( ( 3./(8.*np.pi*m3.c.gravity)*m3.c.hubble**2 * m3.c.hubbleparam**2 * m3.c.omegabaryon) )
    
    
    abundsdct_0p1 = sfc.dct_0p1
    ## assuming calculation from lognH using 0.752 instead of actual hydrogen fraction
    #logdeltavals_mod0p1 = logdeltavals + np.log10(abundsdct_0p1['Hydrogen']/0.752)
    #logdeltavals_modsol = logdeltavals + np.log10(sfc.dct_sol['Hydrogen'][0]/0.752)
    
    # in simflieclone.Zvar_rhoT_z0, smoothed metallicites are primoridial and particle metallicites are solar
    outshape = (len(logdeltavals), len(logTvals))
    tcool_perelt_0p1 = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab=abundsdct_0p1['Hydrogen'], abunds=abundsdct_0p1, last=False)).reshape(outshape)
    tcool_ztot_0p1 = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'total_metals', T4EOS=False, hab=abundsdct_0p1['Hydrogen'], abunds=abundsdct_0p1, last=False)).reshape(outshape)
    tcool_perelt_sol = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    tcool_ztot_sol = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'total_metals', T4EOS=False, hab='ElementAbundance/Hydrogen', abunds='Pt', last=False)).reshape(outshape)
    tcool_perelt_pri = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'per_element', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
    tcool_ztot_pri = (m3.find_coolingtimes(vardict.simfile.z,  vardict, method = 'total_metals', T4EOS=False, hab='SmoothedElementAbundance/Hydrogen', abunds='Sm', last=False)).reshape(outshape)
    
    
    fig, ax = plt.subplots(ncols=1,nrows=1)
    fontsize = 12
    fig.suptitle("Replication of figure 4 of Wiersma, Schaye, & Smith (2009), HM01, $z=0$, $t_{\mathrm{cool}} = t_{H}$", fontsize=fontsize+2)
    
    linestyles = ['solid','dashed']
    linestylelabels = ['per element', 'total metals']
    colors1 = [['black'], ['red'], ['blue']]
    colors2 = [['gray'], ['firebrick'], ['cyan']]
    colorlabels = [r'$Z=Z_{\odot}$', r'$Z=0.1Z_{\odot}$', r'$Z=0$']
    levels = [1./m3.Hubble(0.)/m3.c.sec_per_year] #default cosmology: EAGLE (1/H = 14.4 Gyr, Planck 2014 age of universe = 13.8 Gyr)
    
    plotWSS09fig3(ax, tcool_perelt_sol, logTvals, logdeltavals, levels=levels, colors=colors1[0], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_ztot_sol, logTvals, logdeltavals, levels=levels, colors=colors2[0], linestyle=linestyles[1]) 
    plotWSS09fig3(ax, tcool_perelt_0p1, logTvals, logdeltavals, levels=levels, colors=colors1[1], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_ztot_0p1, logTvals, logdeltavals, levels=levels, colors=colors2[1], linestyle=linestyles[1])    
    plotWSS09fig3(ax, tcool_perelt_pri, logTvals, logdeltavals, levels=levels, colors=colors1[2], linestyle=linestyles[0])
    plotWSS09fig3(ax, tcool_ztot_pri, logTvals, logdeltavals, levels=levels, colors=colors2[2], linestyle=linestyles[1])

    
    ax.set_xlabel(r'$T \, [K]$', fontsize=fontsize)
    ax.set_ylabel(r'$ 1 + \delta$',fontsize=fontsize)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.e-1,1.e4)
    ax.set_xlim(1.e4,1.e8)
    
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both')

    linestyle_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label=linestylelabels[i]) for i in range(len(linestyles))]

    color1_legend_handles =  [mlines.Line2D([], [], color=colors1[i][0], linestyle = 'solid', label=colorlabels[i]) for i in range(len(colors1))]
    #color2_legend_handles =  [mlines.Line2D([], [], color=colors2[0][0], linestyle = 'solid', label=colorlabels[0] + r'$, n_{H}$ interp. (ni)')]
    #color2_legend_handles += [mlines.Line2D([], [], color=colors2[i][0], linestyle = 'solid', label=colorlabels[i] + r', ni') for i in range(1,len(colors2))]
    
    ax.legend(handles=linestyle_legend_handles+color1_legend_handles, fontsize=fontsize, ncol=2, loc='lower right', bbox_to_anchor=(0.98,0.02)) #ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor
    
    plt.savefig(dd_cool+'WSS09fig4_replica_vardict.pdf', format='pdf', bbox_inches='tight')
    
    return vardict


def compare_ionbal_tables_ben_serena(ion, z):
    outdir = '/net/luttero/data1/line_em_abs/v3_master_tests/iobal_tables_bengadget2/'
    outname = 'tablecomp_%s_%s.pdf'%(ion , z)
    
    Tvals = np.arange(1.5, 9.5, 0.03)
    nHvals = np.arange(-9., 4., 0.04)
    
    Tgrid = np.array([Tvals] * len(nHvals)).flatten()
    nHgrid = np.array([[val]*len(Tvals) for val in nHvals]).flatten() 
    gridshape = (len(nHvals), len(Tvals))
    extent = (1.5 * nHvals[0]  - 0.5 * nHvals[1],\
              1.5 * nHvals[-1] - 0.5 * nHvals[-2],\
              1.5 * Tvals[0]  - 0.5 * Tvals[1],\
              1.5 * Tvals[-1] - 0.5 * Tvals[-2])
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    ionbal_bo = m3.find_ionbal_bensgadget2(z, ion, {'lognH': nHgrid, 'logT': Tgrid}).reshape(gridshape)
    ionbal_sb = m3.find_ionbal(z, ion, {'lognH': nHgrid, 'logT': Tgrid}).reshape(gridshape)
    
    vmin = 0.
    vmax = max(np.max(ionbal_bo), np.max(ionbal_sb))
    
    fig = plt.figure(figsize=(5.5, 5.))
    nrows = 2
    ncols = 2
    xlabel = r'$\log_{10} \, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$'
    ylabel = r'$\log_{10} \, T \; [\mathrm{K}]$'
    fontsize = 12
    
    fig.suptitle(r'$\mathrm{%s}, z=%.3f$'%(m3.ild.getnicename(ion, mathmode=True), z), fontsize=fontsize)
    
    sax = fig.add_subplot(nrows, ncols, 1)
    img = sax.imshow(ionbal_sb.T, origin='lower', interpolation='nearest',\
                     extent=extent, vmin=vmin, vmax=vmax, cmap='viridis')
    sax.text(0.5, 1.0, "Serena Bertone's tables", fontsize=fontsize,\
             horizontalalignment='center', verticalalignment='bottom',\
             transform=sax.transAxes)
    sax.set_aspect(aspect)
    plt.colorbar(img, ax=sax)
    sax.set_ylabel(ylabel, fontsize=fontsize)
    
    bax = fig.add_subplot(nrows, ncols, 2)
    img = bax.imshow(ionbal_bo.T, origin='lower', interpolation='nearest',\
                     extent=extent, vmin=vmin, vmax=vmax, cmap='viridis')
    bax.text(0.5, 1.0, "Ben Oppenheimer's tables", fontsize=fontsize,\
             horizontalalignment='center', verticalalignment='bottom',\
             transform=bax.transAxes)
    bax.set_aspect(aspect)
    plt.colorbar(img, ax=bax)
    
    aax = fig.add_subplot(nrows, ncols, 3)
    arr = ionbal_bo - ionbal_sb
    vmax = np.max(np.abs(arr))
    vmin = -1. * vmax
    img = aax.imshow(arr.T, origin='lower', interpolation='nearest',\
                     extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu')
    aax.text(0.5, 1.0, "BO - SB", fontsize=fontsize,\
             horizontalalignment='center', verticalalignment='bottom',\
             transform=aax.transAxes)
    aax.set_aspect(aspect)
    plt.colorbar(img, ax=aax)
    aax.set_xlabel(xlabel, fontsize=fontsize)
    aax.set_ylabel(ylabel, fontsize=fontsize)
    
    rax = fig.add_subplot(nrows, ncols, 4)
    arr = np.log10(ionbal_bo) - np.log10(ionbal_sb)
    vmax = np.max(np.abs(arr[np.isfinite(arr)]))
    vmax = min(vmax, 3.)
    vmin = -1. * vmax
    img = rax.imshow(arr.T, origin='lower', interpolation='nearest',\
                     extent=extent, vmin=vmin, vmax=vmax, cmap='RdBu')
    rax.text(0.5, 1.0, "log10 BO / SB", fontsize=fontsize,\
             horizontalalignment='center', verticalalignment='bottom',\
             transform=rax.transAxes)
    rax.set_aspect(aspect)
    if vmax == 3.:
        extend = 'both'
    else:
        extend = 'neither'
    plt.colorbar(img, ax=rax, extend=extend)
    rax.set_xlabel(xlabel, fontsize=fontsize)
    
    plt.savefig(outdir + outname, format='pdf', bbox_inches='tight')
###############################################################################
#################### halo only projection tests ###############################
###############################################################################
    
# test on 25 Mpc mid-res Ref box, snap 19
mdir = '/net/luttero/data2/imgs/CGM/halo-only_projectiontest/'

def selecthalos_testhaloonly(halocat='/net/luttero/data2/proc/catalogue_RefL0025N0376_snap19_aperture30.hdf5', logMhs=[10., 11., 12.], Mtol=0.05, filename='testselection.txt'):
    with h5py.File(halocat, 'r') as hc:
        logM200c = np.log10(np.array(hc['M200c_Msun']))
        inds = [np.where(np.abs(logM200c - Mh) <= Mtol)[0] for Mh in logMhs]
        sel = np.array([np.random.choice(ind, 1) for ind in inds])
        
        galids = np.array(hc['galaxyid'])[sel]
        groups = np.array(hc['groupid'])[sel]
        logM200c_selected = logM200c[sel]
        
    with open(mdir + filename, 'w') as fo:
        fo.write('halocat: %s\n'%halocat)
        fo.write('logMhs: %s\n'%logMhs)
        fo.write('Mtol: %s\n'%Mtol)
        fo.write('galaxyid\tgroupid\tlogM200c_Msun\n')
        for i in range(len(sel)):
            fo.write('%i\t%i\t%f\n'%(galids[i], groups[i], logM200c_selected[i]))
            
def projecthalos_testhaloonly(filename_halos='testselection.txt', radius_R200c=2.):
    # read in the selected halos
    with open(mdir + filename_halos, 'r') as fs:
        ids = []
        ps = 'dummy'
        while True:
            ps = fs.readline()
            ps = ps[:-1] # strip off newline; Index Error if empty string -> end of file
            if ps == '':
                break
            parts = ps.split(' ')
            if len(parts) == 2:
                if parts[0] == 'halocat:':
                    halocat = parts[1]
                elif parts[0] == 'logMhs':
                    pass # don't need this
                elif parts[0] == 'Mtol':
                    pass # don't need this
            elif len(parts) == 1:
                parts = ps.split('\t')
                if not (parts[0][0]).isdigit(): # line defining the columns
                    gidind = np.where([part == 'galaxyid' for part in parts])[0][0]
                else:
                    ids.append(parts[gidind])
    ids = np.array([int(i) for i in ids])
    # read in the halo properties to define the projections
    with h5py.File(halocat, 'r') as hc:
        galaxyids_all = np.array(hc['galaxyid'])
        selinds = np.where(np.any(galaxyids_all[:, np.newaxis] == ids[np.newaxis, :], axis=1))[0]
        print(selinds)
        
        simprops = {key: item for key, item in hc['Header'].attrs.items()}
        cosmopars = {key: item for key, item in hc['Header/cosmopars'].attrs.items()}
        
        M200c = np.log10(np.array(hc['M200c_Msun'])[selinds])
        R200c = np.array(hc['R200c_pkpc'])[selinds] / cosmopars['a'] * 1e-3
        Xcop = np.array(hc['Xcop_cMpc'])[selinds]
        Ycop = np.array(hc['Ycop_cMpc'])[selinds]
        Zcop = np.array(hc['Zcop_cMpc'])[selinds]
        
    
    # set up the projections and record the input parameters
    argnames = ('simnum', 'snapnum', 'centre', 'L_x', 'L_y', 'L_z', 'npix_x', 'npix_y', 'ptypeW')
    args = [(simprops['simnum'], simprops['snapnum'],\
             [Xcop[i], Ycop[i], Zcop[i]],\
             4. * radius_R200c * R200c[i], 4. * radius_R200c * R200c[i], 4. * radius_R200c * R200c[i],\
             400, 400,\
             'basic') \
             for i in range(len(ids))]
    
    kwargs = {'ionW': None, 'abundsW': 'auto', 'quantityW': 'Mass',\
              'ionQ': None, 'abundsQ': 'auto', 'quantityQ': None, 'ptypeQ': None,\
              'excludeSFRW': False, 'excludeSFRQ': False, 'parttype': '0',\
              'theta': 0.0, 'phi': 0.0, 'psi': 0.0, \
              'sylviasshtables': False,\
              'var': simprops['var'], 'axis': 'z','log': True, 'velcut': False,\
              'periodic': False, 'kernel': 'C2', 'saveres': True,\
              'simulation': 'eagle', 'LsinMpc': None,\
              'select': None, 'misc': None,\
              'ompproj': False, 'numslices': None, 'hdf5': True}
    # not specified: nameonly, halosel, halosel_kwargs
    halosel_kwargs_opts = [[[{'exclsatellites': True, 'allinR200c': True, 'label': 'exclsats_allinR200c_galaxyid-%i-only'%gid} for gid in ids],\
                            [{'exclsatellites': True, 'allinR200c': True, 'label': 'exclsats_allinR200c_galaxyid-%i'%gid} for gid in ids]],\
                           [[{'exclsatellites': True, 'allinR200c': False, 'label': 'exclsats_FOF_galaxyid-%i-only'%gid} for gid in ids],\
                            [{'exclsatellites': True, 'allinR200c': False, 'label': 'exclsats_FOF_galaxyid-%i'%gid} for gid in ids]],\
                           [[{'exclsatellites': False, 'allinR200c': True, 'label': 'inclsats_allinR200c_galaxyid-%i-only'%gid} for gid in ids],\
                            [{'exclsatellites': False, 'allinR200c': True, 'label': 'inclsats_allinR200c_galaxyid-%i'%gid} for gid in ids]],\
                           [[{'exclsatellites': False, 'allinR200c': False, 'label': 'inclsats_FOF_galaxyid-%i-only'%gid} for gid in ids],\
                            [{'exclsatellites': False, 'allinR200c': False, 'label': 'inclsats_FOF_galaxyid-%i'%gid} for gid in ids]],\
                          ]
    
    # all halos or just the central one, for each id
    halosels = [[[('Mhalo_logMsun', M200c[i] - 0.01, M200c[i] + 0.01),\
                  ('X_cMpc', Xcop[i] - 0.1 * R200c[i], Xcop[i] + 0.1 * R200c[i]),\
                  ('Y_cMpc', Ycop[i] - 0.1 * R200c[i], Ycop[i] + 0.1 * R200c[i]),\
                  ('Z_cMpc', Zcop[i] - 0.1 * R200c[i], Zcop[i] + 0.1 * R200c[i]),\
                  ],\
                 [],\
                ] for i in range(len(ids))]

    
    # get names of the halos, store parameters
    basename_metadata = filename_halos.split('.')[0]
    name_metadata = basename_metadata + '_metadata.txt'
    with open(mdir + name_metadata, 'w') as fo:
        keys_kwargs = kwargs.keys()
        keys_halosel = halosel_kwargs_opts[0][0][0].keys()
        topline = '\t'.join(list(argnames) + keys_kwargs + keys_halosel + ['haloselection', 'filename'])
        fo.write(topline + '\n')
        saveline_base = '\t'.join(['%s'] * len(topline.split('\t')))
        for hi in range(len(args)):
            for hsi in range(len(halosels[0])):
                for kwi in range(len(halosel_kwargs_opts)):
                    halosel = halosels[hi][hsi]
                    if halosel is None:
                        kwargs_halosel = None
                    else:
                        kwargs_halosel = halosel_kwargs_opts[kwi][hsi][hi]
                    name = m3.make_map(*args[hi], nameonly=True, halosel=halosel, kwargs_halosel=kwargs_halosel, **kwargs)
                    saveline = saveline_base%(args[hi] + \
                                              tuple([kwargs[key] for key in keys_kwargs]) +\
                                              tuple([kwargs_halosel[key] if kwargs_halosel is not None else 'none' for key in keys_halosel]) +\
                                              ('none' if halosel is None else 'all' if halosel == [] else 'selected-only',) + (name[0],))
                    fo.write(saveline + '\n')
                    m3.make_map(*args[hi], nameonly=False, halosel=halosel, kwargs_halosel=kwargs_halosel, **kwargs) # Error: filename too long using default selection labels
            halosel = None
            kwargs_halosel = None
            name = m3.make_map(*args[hi], nameonly=True, halosel=halosel, kwargs_halosel=kwargs_halosel, **kwargs)
            saveline = saveline_base%(args[hi] + \
                                      tuple([kwargs[key] for key in keys_kwargs]) +\
                                      tuple([kwargs_halosel[key] if kwargs_halosel is not None else 'none' for key in keys_halosel]) +\
                                      ('none' if halosel is None else 'all' if halosel == [] else 'selected-only',) + (name[0],))
            fo.write(saveline + '\n')
            m3.make_map(*args[hi], nameonly=False, halosel=halosel, kwargs_halosel=kwargs_halosel, **kwargs) # Error: filename too long using default selection labels

def getmap(label, name):
    with h5py.File(name, 'r') as ft:
        ds = ft['map']
        temp = np.array(ds)
        maxt = ds.attrs['max']
        mint = ds.attrs['minfinite']
    map_full = {label: (temp, mint, maxt)} 
    return map_full

def plot_testhaloonly(filename_halos='testselection.txt'):
    
    # retieve halo catalogue and ids of used halos
    with open(mdir + filename_halos, 'r') as fs:
        ids = []
        ps = 'dummy'
        while True:
            ps = fs.readline()
            ps = ps[:-1] # strip off newline; Index Error if empty string -> end of file
            if ps == '':
                break
            parts = ps.split(' ')
            if len(parts) == 2:
                if parts[0] == 'halocat:':
                    halocat = parts[1]
                elif parts[0] == 'logMhs':
                    pass # don't need this
                elif parts[0] == 'Mtol':
                    pass # don't need this
            elif len(parts) == 1:
                parts = ps.split('\t')
                if not (parts[0][0]).isdigit(): # line defining the columns
                    gidind = np.where([part == 'galaxyid' for part in parts])[0][0]
                else:
                    ids.append(parts[gidind])
    ids = np.array([int(i) for i in ids])
    
    # read in the halo properties used to define the projections
    with h5py.File(halocat, 'r') as hc:
        galaxyids_all = np.array(hc['galaxyid'])
        selinds = np.where(np.any(galaxyids_all[:, np.newaxis] == ids[np.newaxis, :], axis=1))[0]
       # print(selinds)
        
        simprops = {key: item for key, item in hc['Header'].attrs.items()}
        cosmopars = {key: item for key, item in hc['Header/cosmopars'].attrs.items()}
        
        M200c = np.log10(np.array(hc['M200c_Msun']))
        R200c = np.array(hc['R200c_pkpc']) / cosmopars['a'] * 1e-3
        Xcop = np.array(hc['Xcop_cMpc'])
        Ycop = np.array(hc['Ycop_cMpc'])
        Zcop = np.array(hc['Zcop_cMpc'])
        
        M200c_ch = M200c[selinds]
        R200c_ch = R200c[selinds]
        Xcop_ch = Xcop[selinds]
        Ycop_ch = Ycop[selinds]
        Zcop_ch = Zcop[selinds]
    
    mdfile = filename_halos.split('.')[0] + '_metadata.txt'
    projdata = pd.read_csv(mdir + mdfile, sep='\t', header=0, usecols=['centre', 'L_x', 'L_y', 'L_z', 'allinR200c',	'exclsatellites', 'haloselection', 'label', 'filename'], index_col=False)
    # parse saved str list to floats "[<val 1>, <val 2>, <val 3>]"
    new = projdata["centre"].str[1:-1].str.split(", ", n=-1, expand=True)
    new = new.astype(float)
    projdata['cenx'] = new[0]
    projdata['ceny'] = new[1]
    projdata['cenz'] = new[2]


                    
    for gid in ids:
        centol = 1e-3
        
        idind = np.where(ids == gid)[0][0]
        Mass = M200c_ch[idind]
        radius = R200c_ch[idind]
        
        figname = mdir + 'test_galaxy-%i.pdf'%(gid)
        #print(radius)
        cen = np.array([Xcop_ch[idind], Ycop_ch[idind], Zcop_ch[idind]])
        
        projdata_gid = projdata.loc[(projdata['cenx'] - cen[0])**2 + (projdata['ceny'] - cen[1])**2 + (projdata['cenz'] - cen[2])**2 < 3 * centol**2]
        
        # extract the maps for this galaxyid 
        maps = {}
        maps.update(getmap('all gas', (projdata_gid.loc[projdata_gid['allinR200c']=='none', 'filename']).values[0]))

        maps.update(getmap('FoF', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'False',\
                                                            projdata_gid['exclsatellites']=='False',\
                                                            projdata_gid['haloselection']=='all',\
                                                            ], axis=0), 'filename']).values[0] ))
        
        maps.update(getmap('FoF selected halo', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'False',\
                                                            projdata_gid['exclsatellites']=='False',\
                                                            projdata_gid['haloselection']=='selected-only',\
                                                            ], axis=0), 'filename']).values[0] ))
                               
        maps.update(getmap('FoF + 200c', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'True',\
                                                            projdata_gid['exclsatellites']=='False',\
                                                            projdata_gid['haloselection']=='all',\
                                                            ], axis=0), 'filename']).values[0] ))
                               
        maps.update(getmap('FoF + 200c selected halo', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'True',\
                                                            projdata_gid['exclsatellites']=='False',\
                                                            projdata_gid['haloselection']=='selected-only',\
                                                            ], axis=0), 'filename']).values[0] ))
                               
        maps.update(getmap('FoF no sats', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'False',\
                                                            projdata_gid['exclsatellites']== 'True',\
                                                            projdata_gid['haloselection']== 'all',\
                                                            ], axis=0), 'filename']).values[0] ))
                               
        maps.update(getmap('FoF selected halo no sats', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'False',\
                                                            projdata_gid['exclsatellites']=='True',\
                                                            projdata_gid['haloselection']=='selected-only',\
                                                            ], axis=0), 'filename']).values[0] ))
        
        maps.update(getmap('FoF + 200c no sats', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'True',\
                                                            projdata_gid['exclsatellites']=='True',\
                                                            projdata_gid['haloselection']=='all',\
                                                            ], axis=0), 'filename']).values[0] ))
        
        maps.update(getmap('FoF + 200c selected halo no sats', (projdata_gid.loc[np.all([projdata_gid['allinR200c'] == 'True',\
                                                            projdata_gid['exclsatellites']=='True',\
                                                            projdata_gid['haloselection']=='selected-only',\
                                                            ], axis=0), 'filename']).values[0] ))
                               
        vmin = -6.5 #np.min([maps[key][1] for key in maps.keys()]) # mimum finite is set by edge effects, mainly
        vmax = np.max([maps[key][2] for key in maps.keys()])
        diffmax = vmax - vmin
        
        ## plot the maps, make comparisons
        cmap_diff = cm.get_cmap('coolwarm_r')
        cmap_diff.set_over(cmap_diff(1.))
        cmap_diff.set_under(cmap_diff(0.))
        cmap_vals = cm.get_cmap('viridis')
        cmap_vals.set_under(cmap_vals(0.))
        #ancolor = 'lightgray'
        
        clabel_vals = r'$\log_{10} \, \Sigma_{\mathrm{gas}} \; [\mathrm{g}\, \mathrm{cm}^{-2}]$'
        clabel_diff = r'$\Delta \, \log_{10}  \, \Sigma_{\mathrm{gas}}$'
        fontsize = 12
        xlabel = 'X [cMpc]'
        ylabel = 'Y [cMpc]'
        
        # all projections have same centre, Ls
        #print(projdata_gid.index)
        i_temp = projdata_gid.index[0] # centres should be the same, just pick one
        cenx = projdata_gid.at[i_temp, 'cenx']
        ceny = projdata_gid.at[i_temp, 'ceny']
        cenz = projdata_gid.at[i_temp, 'cenz']
        Lx = projdata_gid.at[i_temp, 'L_x']
        Ly = projdata_gid.at[i_temp, 'L_y']
        Lz = projdata_gid.at[i_temp, 'L_z']
        
        boxsize = cosmopars['boxsize'] / cosmopars['h']
        
        Xtemp = (Xcop - cenx + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        Ytemp = (Ycop - ceny + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        Ztemp = (Zcop - cenz + 0.5 * boxsize) % boxsize - 0.5 * boxsize
        
        sel_gid = np.all([np.abs(Xtemp) <= R200c + 0.5 * Lx, np.abs(Ytemp) <= R200c + 0.5 * Ly, np.abs(Ztemp) <= R200c + 0.5 * Lz], axis=0)
        
        Xplot = Xtemp[sel_gid] + cenx
        Yplot = Ytemp[sel_gid] + ceny
        #Zplot = Ztemp[sel_gid] + Zcop
        Rplot = R200c[sel_gid]
        
        def plotmap(ax, vals, diff=False):
            if isinstance(vals, tuple):
                vals = vals[0]
                
            if diff:
                _vmin = -1 * diffmax
                _vmax = diffmax
                _cmap = cmap_diff
                if np.any(vals) < 0.:
                    print('Warning: found differences < 0.: minimum finite value %s'%(np.min(vals[np.isinfinte(vals)])))
                vals[vals == np.inf] = _vmax + 100.
                vals[vals == -np.inf] = _vmax - 100.
            else:
                _vmin = vmin
                _vmax = vmax
                _cmap = cmap_vals
            img = ax.imshow(vals.T, origin='lower', interpolation='nearest',\
                            cmap=_cmap, vmin=_vmin, vmax=_vmax,\
                            extent=(cenx - 0.5 * Lx, cenx + 0.5 * Lx, ceny - 0.5 * Ly, ceny + 0.5 * Ly))

            patches = [Circle((Xplot[ind], Yplot[ind]), Rplot[ind]) \
               for ind in range(len(Xplot))] # x, y axes only

            collection = PatchCollection(patches)
            collection.set(edgecolor='red', facecolor='none', linewidth=1, linestyle='dotted')
            ax.add_collection(collection)
            
            circlecen = Circle(tuple(cen[:2]), radius, edgecolor='magenta', facecolor='none', linewidth=1.)
            ax.add_artist(circlecen)
            
            ax.tick_params(which='both', direction='in', right=True, top=True, labelsize=fontsize - 1)
        
            return img
        
        def add_cax(cax, img, diff=False, vertical=True):
            if diff:
                _clabel = clabel_diff
                _extend = 'both'
            else:
                _clabel = clabel_vals
                _extend = 'min'
            if vertical:
                orientation='vertical'
            else:
                orientation='horizontal'
            plt.colorbar(img, cax=cax, orientation=orientation, extend=_extend)
            cax.tick_params(which='both', labelsize=fontsize - 1)
            if vertical:
                cax.set_ylabel(_clabel, fontsize=fontsize)
                cax.set_aspect(8.)
            else:
                cax.set_xlabel(_clabel, fontsize=fontsize)
                cax.set_aspect(0.125)
                
        def annotation(ax, text, vertical=True):
            if vertical:
                posx = 0.1
                posy = 0.5
                #bbox_to_anchor = (0.0, 0.5)
                horzal = 'left'
                vertal = 'center'
                rotation = 90
            else:
                posx = 0.5
                posy = 0.1
                #bbox_to_anchor = (0.5, 0.0)
                horzal = 'center'
                vertal = 'bottom'
                rotation = 0.
            ax.text(posx, posy, text, fontsize=fontsize, horizontalalignment=horzal, verticalalignment=vertal, rotation=rotation, transform=ax.transAxes, multialignment='left')
            ax.axis('off')
        
        def addlabels(grid):
            numy, numx = grid.shape
            for i in range(numy):
                grid[i, 0].set_ylabel(ylabel, fontsize=fontsize)
                for j in range(1, numx):
                    grid[i, j].tick_params(labelleft=False)
            for j in range(numx):
                grid[numy - 1, j].set_xlabel(xlabel, fontsize=fontsize)
                for i in range(0, numy - 1):
                    grid[i, j].tick_params(labelbottom=False)
            
        plt.figure(figsize=(15., 10)) # figsize: width, height
        grid = gsp.GridSpec(3, 5, width_ratios=[1.] * 5, height_ratios=[0.7, 1., 1.], hspace=0.25, wspace=0.4, top=0.95, bottom=0.05, left=0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    
        ax_titletext = plt.subplot(grid[0, 2:])
        title = 'Halo-only projection test cube for central galaxy %i in %s, snapshot %i (z=%.4f)'%(gid, simprops['simnum'], simprops['snapnum'], cosmopars['z'])
        title = title + '\n    ' + r'galaxy halo has $\mathrm{M}_{\mathrm{200c}} = 10^{%.3f} \, \mathrm{M}_{\odot}, \mathrm{R}_{\mathrm{200c}} = %.3f \, \mathrm{pkpc}$'%(Mass, radius * 1e3 *cosmopars['a'])
        title = title + '\n' + r'abbreviations:'
        title = title + '\n    ' + r'wsats = with satellites, nsats = without satellites (Subgroupnumber == 0)'
        title = title + '\n    ' + r'FoF = all Friends of Friends group members,'
        title = title + '\n    ' + r'FoF+R = FoF particles and everythin within R200c'
        title = title + '\n    ' + r'1h = only the central halo selected (magenta circle at R200c),'
        title = title + '\n        ' + r'otherwise all halos are selected (red dotted cricles at R200c)'
        annotation(ax_titletext, title, vertical=False)
        
        grid_imgall  = gsp.GridSpecFromSubplotSpec(2, 4, subplot_spec=grid[0, 0:2], height_ratios=[1., 5.], width_ratios=[5., 2., 2., 1.], hspace=0.0, wspace=0.0) 
        
        ax = plt.subplot(grid_imgall[1, 0])
        cax = plt.subplot(grid_imgall[1, 1])
        cax_diff = plt.subplot(grid_imgall[1, 2])
        img = plotmap(ax, maps['all gas'][0])
        add_cax(cax, img, diff=False)
        anax = plt.subplot(grid_imgall[0, 0])
        annotation(anax, 'all gas in the projected region', vertical=False)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        
        #grid_subimgs  = gsp.GridSpecFromSubplotSpec(2, 2, subplot_spec=grid[1, 0:2], height_ratios=[1., 1.], width_ratios=[1., 1., 1., 1.], hspace=0.0, wspace=0.0) 
        
        #mainaxes = [[plt.subplot(grid1[yi,xi]) for yi in range(numy)] for xi in range(numx)] # in mainaxes: x = column, y = row
        #cax = plt.subplot(grid[1]) 
        
        grid_imgopts = gsp.GridSpecFromSubplotSpec(4, 5, subplot_spec=grid[1, 0:2], height_ratios=[1., 1., 5., 5.], width_ratios=[5., 5., 5., 5., 1.], hspace=0.0, wspace=0.0)
        # grid:          allinR220c, all halos    FoF only, all halos     allinR220c, 1 halo    FoF only, 1 halo
        # w/ subhalos
        # w/o subhalos
        
        toprow = plt.subplot(grid_imgopts[0, :])
        annotation(toprow, 'projections with various halo selections', vertical=False)
        topaxes = [plt.subplot(grid_imgopts[1, i]) for i in range(4)]
        annotation(topaxes[0], 'FoF+R', vertical=False)
        annotation(topaxes[1], 'FoF', vertical=False)
        annotation(topaxes[2], 'FoF+R, 1h', vertical=False)
        annotation(topaxes[3], 'FoF, 1h', vertical=False)
        rightaxes = [plt.subplot(grid_imgopts[i + 2, 4]) for i in range(2)]
        annotation(rightaxes[0], 'wsat', vertical=True)
        annotation(rightaxes[1], 'nsat', vertical=True)
        plotaxes = np.array([[plt.subplot(grid_imgopts[i + 2, j])  for j in range(4)] for i in range(2)])
        
        plotmap(plotaxes[0, 0], maps['FoF + 200c'])
        plotmap(plotaxes[0, 1], maps['FoF'])
        plotmap(plotaxes[0, 2], maps['FoF + 200c selected halo'])
        plotmap(plotaxes[0, 3], maps['FoF selected halo'])
        plotmap(plotaxes[1, 0], maps['FoF + 200c no sats'])
        plotmap(plotaxes[1, 1], maps['FoF no sats'])
        plotmap(plotaxes[1, 2], maps['FoF + 200c selected halo no sats'])
        plotmap(plotaxes[1, 3], maps['FoF selected halo no sats'])
                
        addlabels(plotaxes)

        
        grid_anyseldiffs = gsp.GridSpecFromSubplotSpec(4, 5, subplot_spec=grid[2, 0:2], height_ratios=[1., 1., 5., 5.], width_ratios=[5., 5., 5., 5., 1.], hspace=0.0, wspace=0.0)
        # grid:          allinR220c, all halos    FoF only, all halos     allinR220c, 1 halo    FoF only, 1 halo
        # w/ subhalos
        # w/o subhalos
        
        toprow = plt.subplot(grid_anyseldiffs[0, :])
        annotation(toprow, 'difference between all gas and halo selections', vertical=False)
        topaxes = [plt.subplot(grid_anyseldiffs[1, i]) for i in range(4)]
        annotation(topaxes[0], 'FoF+R', vertical=False)
        annotation(topaxes[1], 'FoF', vertical=False)
        annotation(topaxes[2], 'FoF+R, 1h', vertical=False)
        annotation(topaxes[3], 'FoF, 1h', vertical=False)
        rightaxes = [plt.subplot(grid_anyseldiffs[i + 2, 4]) for i in range(2)]
        annotation(rightaxes[0], 'wsat', vertical=True)
        annotation(rightaxes[1], 'nsat', vertical=True)
        plotaxes = np.array([[plt.subplot(grid_anyseldiffs[i + 2, j])  for j in range(4)] for i in range(2)])
            
        plotmap(plotaxes[0, 0], maps['all gas'][0] - maps['FoF + 200c'][0], diff=True)
        plotmap(plotaxes[0, 1], maps['all gas'][0] - maps['FoF'][0], diff=True)
        plotmap(plotaxes[0, 2], maps['all gas'][0] - maps['FoF + 200c selected halo'][0], diff=True)
        plotmap(plotaxes[0, 3], maps['all gas'][0] - maps['FoF selected halo'][0], diff=True)
        plotmap(plotaxes[1, 0], maps['all gas'][0] - maps['FoF + 200c no sats'][0], diff=True)
        plotmap(plotaxes[1, 1], maps['all gas'][0] - maps['FoF no sats'][0], diff=True)
        plotmap(plotaxes[1, 2], maps['all gas'][0] - maps['FoF + 200c selected halo no sats'][0], diff=True)
        img = plotmap(plotaxes[1, 3], maps['all gas'][0] - maps['FoF selected halo no sats'][0], diff=True)
        
        add_cax(cax_diff, img, diff=True) # near top left plot
        addlabels(plotaxes)
        
        
        grid_1halovsall = gsp.GridSpecFromSubplotSpec(5, 2, subplot_spec=grid[1:, 2], height_ratios=[1., 5., 5., 5., 5.], width_ratios=[5., 1.], hspace=0.0, wspace=0.0)
        
        toprow = plt.subplot(grid_1halovsall[0, 0])
        annotation(toprow, 'all halos - 1 halo', vertical=False)
        rightaxes = [plt.subplot(grid_1halovsall[i + 1, 1]) for i in range(4)]
        annotation(rightaxes[0], 'FoF+R, wsat', vertical=True)
        annotation(rightaxes[1], 'FoF, wsat', vertical=True)
        annotation(rightaxes[2], 'FoF+R, nsat', vertical=True)
        annotation(rightaxes[3], 'FoF, nsat', vertical=True)
        plotaxes = np.array([[plt.subplot(grid_1halovsall[i + 1, j])  for j in range(1)] for i in range(4)])
            
        plotmap(plotaxes[0, 0], maps['FoF + 200c'][0] - maps['FoF + 200c selected halo'][0], diff=True)
        plotmap(plotaxes[1, 0], maps['FoF'][0] - maps['FoF selected halo'][0], diff=True)
        plotmap(plotaxes[2, 0], maps['FoF + 200c no sats'][0] - maps['FoF + 200c selected halo no sats'][0], diff=True)
        plotmap(plotaxes[3, 0], maps['FoF no sats'][0] - maps['FoF selected halo no sats'][0], diff=True)
        
        addlabels(plotaxes)
        
        
        grid_R200vsFoF = gsp.GridSpecFromSubplotSpec(5, 2, subplot_spec=grid[1:, 3], height_ratios=[1., 5., 5., 5., 5.], width_ratios=[5., 1.], hspace=0.0, wspace=0.0)
        
        toprow = plt.subplot(grid_R200vsFoF[0, 0])
        annotation(toprow, 'FoF+R - FoF', vertical=False)
        rightaxes = [plt.subplot(grid_R200vsFoF[i + 1, 1]) for i in range(4)]
        annotation(rightaxes[0], 'wsat', vertical=True)
        annotation(rightaxes[1], '1h, wsat', vertical=True)
        annotation(rightaxes[2], 'nsat', vertical=True)
        annotation(rightaxes[3], '1h, nsat', vertical=True)
        plotaxes = np.array([[plt.subplot(grid_R200vsFoF[i + 1, j])  for j in range(1)] for i in range(4)])
            
        plotmap(plotaxes[0, 0], maps['FoF + 200c'][0] -  maps['FoF'][0], diff=True)
        plotmap(plotaxes[1, 0], maps['FoF + 200c selected halo'][0] - maps['FoF selected halo'][0], diff=True)
        plotmap(plotaxes[2, 0], maps['FoF + 200c no sats'][0] - maps['FoF no sats'][0], diff=True)
        plotmap(plotaxes[3, 0], maps['FoF + 200c selected halo no sats'][0] - maps['FoF selected halo no sats'][0], diff=True)
        
        addlabels(plotaxes)
        
        
        grid_wsatsvsnsats = gsp.GridSpecFromSubplotSpec(5, 2, subplot_spec=grid[1:, 4], height_ratios=[1., 5., 5., 5., 5.], width_ratios=[5., 1.], hspace=0.0, wspace=0.0)
        
        toprow = plt.subplot(grid_wsatsvsnsats[0, 0])
        annotation(toprow, 'wsats - nsats', vertical=False)
        rightaxes = [plt.subplot(grid_wsatsvsnsats[i + 1, 1]) for i in range(4)]
        
        annotation(rightaxes[0], 'FoF', vertical=True)
        annotation(rightaxes[1], 'FoF+R', vertical=True)
        annotation(rightaxes[2], 'FoF 1h', vertical=True)
        annotation(rightaxes[3], 'FoF+R 1h', vertical=True)
        plotaxes = np.array([[plt.subplot(grid_wsatsvsnsats[i + 1, j])  for j in range(1)] for i in range(4)])
            
        plotmap(plotaxes[0, 0], maps['FoF'][0] - maps['FoF no sats'][0], diff=True)
        plotmap(plotaxes[1, 0], maps['FoF + 200c'][0] - maps['FoF + 200c no sats'][0], diff=True)
        plotmap(plotaxes[2, 0], maps['FoF selected halo'][0] - maps['FoF selected halo no sats'][0], diff=True)
        plotmap(plotaxes[3, 0], maps['FoF + 200c selected halo'][0] - maps['FoF + 200c selected halo no sats'][0], diff=True)
        addlabels(plotaxes)
        
        plt.savefig(figname, format='pdf')
        

def compare_emission_v3p4_to_v3p5():
    ddir = '/net/luttero/data1/line_em_abs/v3_master_tests/test_v3p5_from_3p4/'
    v3p4 = {'fe17':   'emission_fe17_L0025N0376_27_test3.4_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            'halpha': 'emission_halpha_L0025N0376_27_test3.4_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            'o7r':    'emission_o7r_L0025N0376_27_test3.4_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            'o8':     'emission_o8_L0025N0376_27_test3.4_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            }
    v3p5 = {'fe17':   'emission_fe17_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            'halpha': 'emission_halpha_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            'o7r':    'emission_o7r_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            'o8':     'emission_o8_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5',\
            }
    
    clabel = 'SB [ph/cm2/s/sr]'
    outname_map = ddir + 'mapcomp_emission_test-3p4_check-3p5_{line}.pdf'
    outname_hist = ddir + 'histcomp_emission_test-3p4_check-3p5_{line}.pdf'
    for line in v3p4:
        f4 = h5py.File(ddir + v3p4[line], 'r')
        i4 = f4['map'][:]
        f4.close()
        f5 = h5py.File(ddir + v3p5[line], 'r')
        i5 = f5['map'][:]
        f5.close()
        
        compareplot(i5, i4, fontsize=12, clabel=clabel,\
                    name=outname_map.format(line=line), diffmax=None)
        comparehist(i5, i4, fontsize=12 ,clabel=clabel,\
                    name=outname_hist.format(line=line), diffmax=None,\
                    nbins=100)
        
#emission_fe17_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5

#emission_halpha_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5

#emission_o7r_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5


#emission_o8_L0025N0376_27_test3.5_SmAb_C2Sm_8000pix_6.25slice_zcen3.125_z-projection_noEOS.hdf5
    
    
        