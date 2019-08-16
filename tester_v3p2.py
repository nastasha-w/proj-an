# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:32:05 2017

@author: wijers
"""
import numpy as np
import h5py
#import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.lines as mlines


# test module: expect changes so reload modules
import make_maps_v3_master as m3
reload(m3)
import simfileclone as sfc
reload(sfc)



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
        R200c = np.log10(np.array(hc['R200c_pkpc'])[selinds]) / cosmopars['a'] * 1e-3
        Xcop = np.array(hc['Xcop_cMpc'])[selinds]
        Ycop = np.array(hc['Ycop_cMpc'])[selinds]
        Zcop = np.array(hc['Zcop_cMpc'])[selinds]
        
    
    # set up the projections and record the input parameters
    argnames = ('simnum', 'snapnum', 'centre', 'L_x', 'L_y', 'L_z', 'npix_x', 'npix_y', 'ptypeW')
    args = [(simprops['simnum'], simprops['snapnum'],\
             [Xcop[i], Ycop[i], Zcop[i]],\
             radius_R200c * R200c[i], radius_R200c * R200c[i], radius_R200c * R200c[i],\
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
              'ompproj': False, 'numslices': None}
    # not specified: nameonly, halosel, halosel_kwargs
    halosel_kwargs_opts = [{'exclsatellites': True, 'allinR200c': True},\
                           {'exclsatellites': True, 'allinR200c': False},\
                           {'exclsatellites': False, 'allinR200c': False},\
                           {'exclsatellites': False, 'allinR200c': True}] 
    
    # all halos or just the central one, for each id
    halosels = [[[('Mhalo_logMsun', M200c[i] - 0.01, M200c[i] + 0.01),\
                  ('X_cMpc', Xcop[i] - 0.1 * R200c[i], Xcop[i] + 0.1 * R200c[i]),\
                  ('Y_cMpc', Ycop[i] - 0.1 * R200c[i], Ycop[i] + 0.1 * R200c[i]),\
                  ('Z_cMpc', Zcop[i] - 0.1 * R200c[i], Zcop[i] + 0.1 * R200c[i]),\
                  ],
                 []\
                ] for i in range(len(ids))]

    
    # get names of the halos, store parameters
    basename_metadata = filename_halos.split('.')[0]
    name_metadata = basename_metadata + '_metadata.txt'
    with open(mdir + name_metadata, 'w') as fo:
        keys_kwargs = kwargs.keys()
        keys_halosel = halosel_kwargs_opts[0].keys()
        topline = '\t'.join(list(argnames) + keys_kwargs + keys_halosel + ['onlyselectedhalo', 'filename'])
        fo.write(topline)
        saveline_base = '\t'.join(['%s'] * len(topline.split('\t')))
        for hi in range(len(args)):
            for hsi in range(len(halosels[0])):
                for kwi in range(len(halosel_kwargs_opts)):
                    name = m3.make_map(*args[hi], nameonly=True, halosel=halosels[hi][hsi], kwargs_halosel=halosel_kwargs_opts[kwi], **kwargs)
                    saveline = saveline_base%(args + (kwargs[key] for key in keys_kwargs) + (halosel_kwargs_opts[kwi][key] for key in keys_halosel) + (halosels[hi][hsi] != [],) + (name,))
                    fo.write(saveline)
        
    # run the projections