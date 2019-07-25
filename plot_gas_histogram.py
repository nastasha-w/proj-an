#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:10:16 2019

@author: wijers
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.gridspec as gsp

import eagle_constants_and_units as c

fontsize = 12
histfile = '/data2/wijers/temp/gashistogram_L0100N1504REFERENCE_27_PtAb_T4EOS.hdf5'
pdir = '/net/luttero/data2/proc/'
mdir = '/home/wijers/Documents/papers/lynx_white_paper_ben/'
normmax_gray = 0.7 # controls how dark the color map gets
figwidth = 3.5

rho_to_nh = 0.752/(c.atomw_H * c.u)

## make the gray color map
# from stackexchange
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap
gray_m = truncate_colormap(plt.get_cmap('gist_gray_r'), maxval=normmax_gray)

def add_2dplot(ax, hist, edges, plotaxes=(0, 1), log=True, pixdens=True, shiftx=0., shifty=0., cmap=None, vmin=None, vmax=None):
    '''
    hist: 2d histogram
    edges: match hist shape
    plotaxes: 0, 1: plot hist, 1, 0: plot hist.T
    '''
    axis1, axis2 = tuple(plotaxes)
    imgtoplot = np.copy(hist)
    if axis1 > axis2: 
        imgtoplot = imgtoplot.T
        edges = [edges[1], edges[0]]
    if pixdens:
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(edges[0]), np.diff(edges[1]) ] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])
        imgtoplot /= normmatrix
        del normmatrix      
    if log:
        imgtoplot = np.log10(imgtoplot)   
    if vmin is None:
        vmin = np.min(imgtoplot[np.isfinite(imgtoplot)])
    if vmax is None:
        vmax = np.max(imgtoplot[np.isfinite(imgtoplot)])
    if cmap is None:
        cmap = mpl.cm.get_cmap('gist_gray')
    elif isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    cmap.set_under(cmap(0.))
    cmap.set_over(cmap(1.))
    ax.set_facecolor(cmap(0.))
    img = ax.pcolormesh(edges[0] + shiftx, edges[1] + shifty, imgtoplot.T, cmap=cmap, vmin=vmin, vmax=vmax)
    return img, vmin, vmax

def add_2dhist_contours(ax, hist, edges, toplotaxes=(0, 1),\
                        fraclevels=True, levels=None, legendlabel=None,\
                        shiftx=0., shifty=0., **kwargs):
    '''
    colors, linestyles: through kwargs
    toplotaxes: order determines whether axes are transposed or not
    '''
    # transpose axes if required
    binsum = np.copy(hist)
    if toplotaxes[0] > toplotaxes[1]:
        binsum = binsum.T
        edges = [edges[1], edges[0]]
         
    if levels is None:
        if fraclevels:
            levels = [1., 0.9, 0.5] # enclosed fractions for each level (approximate)
        else:
	        levels = [1e-3, 3e-2, 0.1, 0.5]

    if fraclevels: # assumes all levels are between 0 and 1
        binsum = binsum/np.sum(binsum) # redo normalisation for the smaller dataset
        #print('min/max binsum: %.4e, %.4e'%(np.min(binsum),np.max(binsum)))
        
        # for sorting, normialise bins by bin size: peak finding depends on density, should not favour larger bins
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(edges[0]), np.diff(edges[1])] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])
        
        binsumcopy = binsum.copy() # copy to rework
        bindens    = binsumcopy/normmatrix
        bindensflat= bindens.copy().reshape(np.prod(bindens.shape)) # reshape creates views; argsorting later will mess up the array we need for the plot
        binsumcopy = binsumcopy.reshape(np.prod(binsumcopy.shape))
        binsumcopy = binsumcopy[np.argsort(bindensflat)] # get all histogram values in order of histogram density (low to high)
        
        binsumcopy = np.flipud(binsumcopy) # flip to high-to-low
        cumul = np.cumsum(binsumcopy) # add values high-to-low 
        wherelist = [[(np.where(cumul<=level))[0],(np.where(cumul>=level))[0]] for level in levels] # list of max-lower and min-higher indices

        ### made for using bin counts -> binsumcopy is ordered y its own values
	    # sort out list: where arrays may be empty -> levels outside 0,1 range, probabaly
	    # set value level 0 for level == 1. -> just want everything (may have no cumulative values that large due to fp errors)
	    # if all cumulative values are too high (maxmimum bin has too high a fraction), set to first cumulative value (=max bin value)
	    # otherwise: interpolate values, or use overlap
        if np.all(normmatrix == normmatrix[0,0]): # all bins are the same size
            valslist = [cumul[0]  if  wherelist[i][0].shape == (0,) else\
	                    0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		                np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
                                               np.array([ binsumcopy[wherelist[i][0][-1]], binsumcopy[wherelist[i][1][0]] ]) )[0]\
		                for i in range(len(levels))]
            pltarr = binsum
        else: # find a reasonable interpolation of bindens in stead; need to plot the contours in binsdens as well, in this case
            bindensflat.sort() # to match cumul array indices: sort, then make high to low
            bindensflat = bindensflat[::-1]
            valslist = [bindensflat[0]  if  wherelist[i][0].shape == (0,) else\
	                    0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		                np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
		                                                 np.array([ bindensflat[wherelist[i][0][-1]], bindensflat[wherelist[i][1][0]] ]))[0]\
		                for i in range(len(levels))]
            pltarr = bindens        
        del normmatrix
        del binsumcopy
        del binsum
        del bindens
        del bindensflat
        #for i in range(len(levels)):
        #    if not (wherelist[i][0].shape == (0,) or wherelist[i][1].shape == (0,)):
	    #        print('interpolating (%f, %f) <- index %i and (%f, %f)  <- index %i to %f'\
	    #	 %(cumul[wherelist[i][0][-1]],binsumcopy[wherelist[i][0][-1]],wherelist[i][0][-1],\
	    #          cumul[wherelist[i][1][0]], binsumcopy[wherelist[i][1][0]], wherelist[i][1][0],\
    	#	   levels[i]) )
        #print(np.all(np.diff(binsumcopy)>=0.))
        uselevels = np.copy(valslist)
        # check for double values; fudge slightly if levels are the same
        anyequal = np.array([np.array(valslist) == val for val in valslist])
        if np.sum(anyequal) > len(valslist): # any levels equal to a *different* level
            eqvals = [np.where(anyequal[ind])[0] for ind in range(len(valslist))] # what levels any value is equal to
            eqgroups = set([tuple(list(eq)) for eq in eqvals]) # get the sets of unique values
            eqgroups = list(eqgroups)
            fudgeby = 1.e-8
            grouplist = [(np.where(np.array([ind in group for group in eqgroups]))[0])[0] for ind in range(len(valslist))] # which group is each uselevel index in
            groupindlist = [(np.where(ind == np.array(eqgroups[grouplist[ind]]))[0])[0] for ind in range(len(valslist))] # which group index corresponds to a goven uselevel index
            addto = [[valslist[group[0]]*fudgeby*ind for ind in range(len(group))] for group in eqgroups] #add nothing for single-element groups
                            
            valslist = [uselevels[ind] + addto[grouplist[ind]][groupindlist[ind]] for ind in range(len(valslist))]
            print('Desired cumulative fraction levels were %s; using value levels %s fudged from %s'%(levels, valslist, uselevels))
            uselevels = valslist
        else:
            print('Desired cumulative fraction levels were %s; using value levels %s'%(levels,uselevels))
    else:
        uselevels=levels
        pltarr = binsum
    
    removezerolevelprops = False
    if len(uselevels) > 1:
        if uselevels[0] == uselevels[1]:
            uselevels = uselevels[1:]
            removezerolevelprops = True
            
    #print binsum, binsum.shape
    if 'linestyles' in kwargs:        
        linestyles = kwargs['linestyles']
    else:
        linestyles = [] # to not break the legend search
    
    if removezerolevelprops: # a duplicate level was kicked out -> remove properties for that level
        if 'linestyles' in kwargs.keys():
            kwargs['linestyles'] = kwargs['linestyles'][1:]
        if 'colors' in kwargs.keys():
            kwargs['colors'] = kwargs['colors'][1:]
            
    # get pixel centres from edges
    centres0 = edges[0][:-1] + shiftx + 0.5*np.diff(edges[0]) 
    centres1 = edges[1][:-1] + shifty + 0.5*np.diff(edges[1])
    contours = ax.contour(centres0, centres1, pltarr.T, uselevels, **kwargs)
    # make a legend to avoid crowding plot region
    #for i in range(len(levels)):
    #    contours.collections[i].set_label('%.0e'%levels[i])
    # color only legend; get a solid line in the legend
    
    #ax.tick_params(labelsize=fontsize,axis='both')
    if 'solid' in linestyles:
        contours.collections[np.where(np.array(linestyles)=='solid')[0][0]].set_label(legendlabel)
    else: # just do the first one
        contours.collections[0].set_label(legendlabel)

def add_rhoavx(ax, userhob_ea, color='lightsalmon', **kwargs):
    ealabel = r'$\overline{\rho_b}$'	 
    rhob_ea = userhob_ea
    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 1
    ax.axvline(x=rhob_ea, ymin=0., ymax=1., color=color, label=ealabel, **kwargs)
     
def add_colorbar(ax, img=None, vmin=None, vmax=None, cmap=None, clabel=None,\
                 newax=False, extend='neither', fontsize=12., orientation='vertical'):
    if img is None:
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, extend=extend, orientation=orientation)
    else:
        cbar = mpl.colorbar.Colorbar(ax, img, extend=extend, orientation=orientation)
    ax.tick_params(labelsize=fontsize - 1.)
    if clabel is not None:
        cbar.set_label(clabel,fontsize=fontsize)
    return cbar

def subplot_mass_phasediagram(ax, bins_sub, logrho, logT, contourdct=None, xlim=None, ylim=None, vmin=None, vmax=None, cmap=gray_m,\
                              dobottomlabel=False, doylabel=False, subplotlabel=None, subplotind=None, logrhob=None,\
                              fraclevels=None, linestyles=None, fontsize=fontsize, subplotindheight=0.92):
    '''
    do the subplot, and sort out the axes
    rhoav is set for snapshot 28
    '''
    contourcolor = 'fuchsia'
    textcolor = 'black'
    img = ax.pcolormesh(logrho, logT, np.log10(bins_sub.T), cmap=cmap, vmin=vmin, vmax=vmax)

    if contourdct is not None:
        for key in contourdct.keys():
            toplot = contourdct[key]
            ax.contour(toplot['x'], toplot['y'], toplot['z'].T, **(toplot['kwargs']))                
    if fraclevels is not None:
        add_2dhist_contours(ax, bins_sub, [logrho, logT], toplotaxes=(0, 1),\
                            fraclevels=True, levels=fraclevels, legendlabel='mass',\
                            shiftx=0., shifty=0., colors=(contourcolor,)*len(fraclevels), linestyles=linestyles)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xlim is not None:
        ax.set_ylim(ylim)        
    # square plot
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps=[1,2,5,10], nbins=6, prune='lower'))
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize-1, direction='in', right=True, top=True, axis='both', which='both', color='black',\
                   labelleft=doylabel, labeltop=False, labelbottom=dobottomlabel, labelright=False)
    
        
    if doylabel:
        ax.set_ylabel(r'$\log_{10}\, T \; [K]$', fontsize=fontsize)
    if dobottomlabel:
        ax.set_xlabel(r'$\log_{10}\, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}]$', fontsize=fontsize)
    if subplotlabel is not None:
        ax.text(0.92,0.92 ,subplotlabel, fontsize=fontsize, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color=textcolor) # bbox=dict(facecolor='white',alpha=0.3)
    if subplotind is not None:
        ax.text(0.92, subplotindheight,subplotind, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, color=textcolor)
    return img


# phase diagram, gas distribution, temperature, density, mass, oxygen mass
def plot_mass_phase_diagram_27(fontsize=fontsize):
    '''
    formerly 'phase_diagram_by_mass_Omass_o78mass_L0100N1504_28_PtAb_T4SFR.pdf'
    
    '''

    # just hard-code loading: these histograms are small and pre-normalised
    # keys: hist, logT, lognH. hist = array: logT x lognH
    
    imgname = 'phase_diagram_by_mass_with_iondet_contours.pdf'
    
    with h5py.File(histfile, 'r') as datafile:
        #cosmopars  = {key: item for (key, item) in datafile['Header/cosmopars'].attrs.items()} 
    
        grp = datafile['histograms']
        hist = np.array(grp['Mass'])
        ax0_nH = np.array(datafile['edges/nH'])
        ax1_T  = np.array(datafile['edges/Temperature'])
        hist = np.sum(hist, axis=2) # sum over different Z contributions
        # edges are +- infinity -> set to something finite to get non-NaN pixel densities
        if ax0_nH[0] == -np.inf:
            ax0_nH[0] = 2. * ax0_nH[1] - ax0_nH[2]
        if ax1_T[0] == -np.inf:
            ax1_T[0] = 2. * ax1_T[1] - ax1_T[2]
        if ax0_nH[-1] == np.inf:
            ax0_nH[-1] = 2. * ax0_nH[-2] - ax0_nH[-3]
        if ax1_T[-1] == np.inf:
            ax1_T[-1] = 2. * ax1_T[-2] - ax1_T[-3]
        
        hist = hist / np.sum(hist) # normalize to total mass
        hist = hist / (np.diff(ax0_nH)[:, np.newaxis] * np.diff(ax1_T)[np.newaxis, :]) # convert to pixel densities
        
        if np.sum(hist[0, :]) > 0.:
            print('Some gas was at densities below the minimum finite bin value')
        if np.sum(hist[-1, :]) > 0.:
            print('Some gas was at densities above the maximum finite bin value')
        if np.sum(hist[:, 0]) > 0.:
            print('Some gas was at temperatures below the minimum finite bin value')
        if np.sum(hist[:, -1]) > 0.:
            print('Some gas was at temperatures above the maximum finite bin value')
        
    plt.figure(figsize=(5.5, 5.0)) # figsize: width, height
    numx=1
    numy=1
    grid = gsp.GridSpec(1, 2, width_ratios=[8., 1.], hspace=0.0, wspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    grid1  = gsp.GridSpecFromSubplotSpec(numx, numy, subplot_spec=grid[0], height_ratios=[1.]*numy, width_ratios=[1.]*numx, hspace=0.0, wspace=0.0)  
    mainaxes = [[plt.subplot(grid1[yi,xi]) for yi in range(numy)] for xi in range(numx)] # in mainaxes: x = column, y = row
    cax = plt.subplot(grid[1]) 

    linestyles = ['solid'] #['dashed', 'solid']
    fraclevels = None #[0.99,0.50] 
    #color_byion = 'green'
    
    xlim = (-8.5, 2.)
    ylim = (2.5, 9.)
    contour_ions = ['o6', 'o7', 'o8', 'ne9']
    # o8 (doublet): 3 mA at b=100 km/s (2.7-3.2 mA at 50-200 km/s) -> 15.4
    # o6: 50 mA -> 13.7 at b=15 km/s. at b=15 km/s, 12.4 is Nmin for 3 mA (both in the 1031 A line)
    # o7: 3 mA at 90 km/s: 15.1
    # ne9 at 160 km/s: 15.45 (-> 15.4) for 3 mA 
    
    obs_thresholds = {'o6': 13.7,\
                      'o7': 14.5,\
                      'o8': 15.0,\
                      'ne9': 15.4}
    ion_files = {'o6': pdir + 'hist_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5',\
                 'o7': pdir + 'hist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5',\
                 'o8': pdir + 'hist_coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5',\
                 'ne9': pdir + 'hist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5',\
                 }
    levels = [0.999]
    linestyles = ['solid']
    colors_ion = {'o6': 'C0',\
                  'o7': 'C1',\
                  'o8': 'C2',\
                  'ne9': 'C3'}
    names_ion = {'o6': 'O VI',\
                 'o7': 'O VII',\
                 'o8': 'O VIII',\
                 'ne9': 'Ne IX'}
    #subplotinds = [['(a)','(c)'],['(b)','(d)']]
    #subplotlabels = [['mass', 'O VII mass'],['O mass','O VIII mass']]
    #subplotkeys   = [['mass', 'o7mass'], ['Omass', 'o8mass']]
    vmax = np.max(hist)
    vmin = max(-10., vmax - 10.)
    
    for flatind in range(numx*numy):       
        xi = flatind/numx
        yi = flatind%numx
        doylabel    = (xi == 0)
        doxlabel    = (yi == numy-1)
        if doxlabel:
            subplotindheight = 0.33
        else:
            subplotindheight = 0.25
        imgsub = subplot_mass_phasediagram(mainaxes[xi][yi], hist, ax0_nH, ax1_T, contourdct=None,\
                              xlim=xlim, ylim=ylim, vmin=vmin, vmax=vmax, cmap=gray_m,\
                              dobottomlabel=doxlabel, doylabel=doylabel, subplotlabel=None, subplotind=None, logrhob=None,\
                              fraclevels=fraclevels, linestyles=linestyles, fontsize=fontsize, subplotindheight=subplotindheight)
        if flatind ==0:
            img = imgsub
    
    ## add dectable gas contours:
    for ion in contour_ions:
        with h5py.File(ion_files[ion], 'r') as fi:
            Nedges = np.array(fi['bins/axis_0'])
            Tedges = np.array(fi['bins/axis_2'])
            nHedges = np.array(fi['bins/axis_1']) + np.log10(rho_to_nh)
            hist = np.array(fi['masks_0/hist'])
            
            # edges are +- infinity -> set to something finite to get non-NaN pixel densities
            if nHedges[0] == -np.inf:
                nHedges[0] = 2. * nHedges[1] - nHedges[2]
            if Tedges[0] == -np.inf:
                Tedges[0] = 2. * Tedges[1] - Tedges[2]
            if nHedges[-1] == np.inf:
                nHedges[-1] = 2. * nHedges[-2] - nHedges[-3]
            if Tedges[-1] == np.inf:
                Tedges[-1] = 2. * Tedges[-2] - Tedges[-3]
            
            Nind = np.argmin(np.abs(Nedges - obs_thresholds[ion]))
            if np.abs(Nedges[Nind] - obs_thresholds[ion]) > 0.01:
                print('Warning: for ion %s, using threshold %s instead of desired %s'%(ion, Nedges[Nind], obs_thresholds[ion]))
            
            #hist = np.sum(hist[Nind:, :, :], axis=0) / np.sum(hist, axis=0) # fraction of absorbers observable at any rho, T
            #hist[np.isnan(hist)] = 0. # 0./0. when there is just nothing there
            #if ion == 'o8':
            #    plt.subplot(111)
            #    plt.pcolormesh(nHedges, Tedges, (np.sum(hist, axis=0)).T)
            #    plt.colorbar()
            #    plt.show()
            
            hist = np.sum(hist[Nind:, :, :], axis=0)
            #if ion == 'o8':
            #    plt.subplot(111)
            #    plt.pcolormesh(nHedges, Tedges, hist.T)
            #    plt.colorbar()
            #    plt.show()
            
            
        add_2dhist_contours(mainaxes[0][0], hist, [nHedges, Tedges], toplotaxes=(0, 1),\
                        fraclevels=True, levels=levels, legendlabel=None,\
                        shiftx=0., shifty=0., linestyles=linestyles, colors=[colors_ion[ion]] * len(levels))

    add_colorbar(cax,img=img, clabel=r'$\log_{10} \left( \partial^2 f_{\mathrm{mass}} \, / \, \partial \log_{10} T \, \partial \log_{10} n_{\mathrm{H}} \right)$', fontsize=fontsize - 1, extend = 'min')
    cax.set_aspect(15.)
    cax.tick_params(labelsize=fontsize - 1, axis='both')
    
    
    # set up legends
    #handles_tcool = [mlines.Line2D([], [], color=colors_tcool[0], linestyle='dashdot', label=r'$t_{\mathrm{C}\,/\,\mathrm{H}}^{\mathrm{net}}(%s) = %s$'%(colors_tcool_labels[0], levels_tcool_labels[0]))]
    #mainaxes[0][0].legend(handles=handles_tcool, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(0.995, 0.000), framealpha=0., edgecolor='lightgray', borderaxespad=0.)
    #handles_rhoav = [mlines.Line2D([], [], color='indianred', linestyle='solid', linewidth=1.5, label=r'$\overline{\rho_b}$')]
    #mainaxes[1][0].legend(handles=handles_rhoav, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(0.995, 0.000), framealpha=0., edgecolor='lightgray', borderaxespad=0.)
    #handles_contourtype =  [mlines.Line2D([], [], color=color_byion, linestyle='solid', linewidth=2, label=r'sightl.')]
    #handles_contourtype += [mlines.Line2D([], [], color='fuchsia', linestyle='solid', linewidth=2, label=r'mass')]
    #mainaxes[0][1].legend(handles=handles_contourtype, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(0.995, 0.01), framealpha=0., edgecolor='lightgray')  
    level_legend_handles = [] #[mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.0f%% obs. absorbers'%(100.*levels[i])) for i in range(len(levels))]
    color_legend_handles = [mlines.Line2D([], [], color=colors_ion[ion], linestyle='solid', label=r'%s: $N > %s$'%(names_ion[ion], obs_thresholds[ion])) for ion in contour_ions]
    mainaxes[0][0].legend(handles=level_legend_handles + color_legend_handles, fontsize=fontsize-1, ncol=2, loc='lower right', bbox_to_anchor=(0.995, 0.01), framealpha=0., edgecolor='lightgray')
        
    plt.savefig(mdir + imgname ,format='pdf', bbox_inches='tight') 