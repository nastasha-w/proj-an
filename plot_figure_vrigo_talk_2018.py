#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:28:54 2018

@author: wijers
"""

import numpy as np
import scipy.integrate as si
import h5py
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gsp

import makecddfs as mc # don't use getcosmopars! This requires simfile and the whole readEagle mess
import eagle_constants_and_units as c
import cosmo_utils as csu

datadir = '/net/luttero/data2/paper1/'
outputdir = '/home/wijers/Documents/talks/virgo_2018_12/'

## pltohistograms:
# plot_cor_o678_27_1sl
# plot_cor_ne878_27_1sl
# plotrho_T_by_N_o78_28_16sl
# plot_mass_phase_diagrams_28
# plotcddf_decomposed_by_rho
# plotcddf_decomposed_by_T
# plotiondiffs_Nvsdiff
# plotcddf_decomposed_by_rho
# plotcddf_decomposed_by_T

## general labels and settings
logNlabel = r'$\log_{10}\, N \; [\mathrm{cm}^{-2}]$'
logNo7label = r'$\log_{10}\, N_{\mathrm{O\,VII}} \; [\mathrm{cm}^{-2}]$'
logNo8label = r'$\log_{10}\, N_{\mathrm{O\,VIII}} \; [\mathrm{cm}^{-2}]$'
logNo6label = r'$\log_{10}\, N_{\mathrm{O\,VI}} \; [\mathrm{cm}^{-2}]$'
logNone8label = r'$\log_{10}\, N_{\mathrm{Ne\,VIII}} \; [\mathrm{cm}^{-2}]$'

cddflabel = r'$\log_{10}\, \partial^2 n \,/\, \partial N \, \partial X  $'
cddfrellabel_z = r'$\log_{10}\, f(N, z)\,/\,f(N, %s)$'
cddfrellabel = r'$\log_{10}\, f(N)\,/\,f(N, %s)$'

logEWlabel = r'$\log_{10}\, EW\; [\mathrm{m\AA}]$, rest'
logEWo7label = r'$\log_{10}\, EW_{O\,VII}\; [\mathrm{m\AA}]$, rest'
logEWo8label = r'$\log_{10}\, EW_{O\,VIII}\; [\mathrm{m\AA}]$, rest'

nHconvlabel = r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}], f_H = 0.752$'
logTlabel = r'$\log_{10}\, T \; [K]$'
logdeltalabel = r'$\log_{10}\, (1 + \delta)$'

Nrange = (12.5, 16.5)
fontsize = 13.
figwidth = 5.5
figsize_1r = (figwidth, 3.)
figsize_2r = (figwidth, 5.)
normmax_bone = 0.35
normmax_gray = 0.7

## from stackexchange
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

bone_m = truncate_colormap(plt.get_cmap('bone_r'), maxval=normmax_bone)
gray_m = truncate_colormap(plt.get_cmap('gist_gray_r'), maxval=normmax_gray)

lambda_rest = {'o7':        21.6019,\
               'o8':        18.9689,\
               'o8major':   18.9671,\
               'o8minor':   18.9725,\
               'h1':        1156.6987178884301,\
               'c3':        977.0201,\
               'c4major':   1548.2041,\
               'c5major':   40.2678,\
               'c6major':   33.7342,\
               'n6':        28.7875,\
               'n7major':   24.7792,\
               'ne8major':  770.409,\
               'ne9':       13.4471,\
               'o4':        787.711,\
               'o5':        629.730,\
               'o6major':   1031.9261,\
               'si3':       1206.500,\
               'si4major':  1393.76018,\
               'fe17major': 15.0140     }
fosc ={'o7':      0.696,\
       'o8':      0.416,\
       'o8major': 0.277,\
       'o8minor': 0.139,\
       'h1':      0.5644956 } 

multip = {'o8': ['o8major', 'o8minor']}


############################## utility functions ################################

#from Lan & Fukugita 2017, citing e.g. B.T. Draine, 2011: Physics of the interstellar and intergalactic medium
# ew in angstrom, Nion in cm^-2
# Lan and Fukugita use rest-frame wavelengths; possibly better match using redshifted wavelengths. (deltaredshift uses the same value as in coldens hist. calc., then EW useslambda_rest*deltaredshift)
# 1.13e20 comes from (np.pi*4.80320427e-10**2/9.10938215e-28/2.99792458e10**2 / 1e8)**-1 = ( np.pi*e[statcoulomb]**2/(m_e c**2) *Angstrom/cm )**-1
def lingrowthcurve(ew,ion):
    return (np.pi * c.electroncharge**2 / (c.electronmass * c.c**2) * 1e-8)**-1 * ew / (fosc[ion] * lambda_rest[ion]**2)  
def lingrowthcurve_inv(Nion,ion):
    return Nion * (fosc[ion] * lambda_rest[ion]**2) * (np.pi * c.electroncharge**2 / (c.electronmass * c.c**2) * 1e-8)

def linflatcurveofgrowth_inv(Nion,b,ion):
    '''
    equations from zuserver2.star.ucl.ac.uk/~idh/PHAS2112/Lectures/Current/Part4.pdf
    b in cm/s
    Nion in cm^-2
    out: EW in Angstrom
    '''
    # central optical depth; 1.13e20,c and pi come from comparison to the linear equation
    #print('lambda_rest[ion]: %f'%lambda_rest[ion])
    #print('fosc[ion]: %f'%fosc[ion])
    #print('Nion: ' + str(Nion))
    #print('b: ' + str(b))
    if ion in multip.keys():
        tau0s = np.array([(np.pi**0.5* c.electroncharge**2 /(c.electronmass*c.c) *1e-8) *lambda_rest[line]*fosc[line]*Nion/b for line in multip[ion]]).T
        xoffsets = (c.c/b)* (np.array([lambda_rest[line] for line in multip[ion]]) - lambda_rest[multip[ion][0]])/lambda_rest[ion] # it shouldn't matter reltive to which the offset is taken
        #print(tau0s)
        #print(xoffsets)     
        prefactor = lambda_rest[ion]/c.c*b # just use the average here
        # absorption profiles are multiplied to get total absorption
        
        integral = np.array([si.quad(lambda x: 1- np.exp(np.sum(-taus*np.exp(-1*(x-xoffsets)**2),axis=0)),-np.inf,np.inf) for taus in tau0s])

    else:
        if ion == 'o8_assingle':
            ion = 'o8'
        tau0 = (np.pi**0.5* c.electroncharge**2 /(c.electronmass*c.c) *1e-8) *lambda_rest[ion]*fosc[ion]*Nion/b
        prefactor = lambda_rest[ion]/c.c*b
        #def integrand(x):
        #    1- np.exp(-tau0*np.exp(-1*x**2))
        integral = np.array([si.quad(lambda x: 1- np.exp(-tau*np.exp(-1*x**2)),-np.inf,np.inf) for tau in tau0])

    if np.max(integral[:,1]/integral[:,0]) > 1e-5:
        print('Warning: check integration errors in linflatcurveofgrowth_inv')
    return prefactor*integral[:,0]


def getcddf(name, norm=None):
    if name[-4:] != '.npz':
        name = name + '.npz'
    with h5py.File(datadir + 'CDDFs.hdf5', 'r') as df:
        grp = df[name]
        temp = [np.array(grp['dn_absorbers_dNdX']), np.array(grp['left_edges'])]
    if norm is not None:
        temp[0] = temp[0]/norm[0]
    return temp

def readdata(filen,separator = None,headerlength=1):
    # input: file name, charachter separating columns, number of lines at the top to ignore
    # separator None means any length of whitespace is considered a separator
    # only for numerical data
    
    data = open(filen,'r')
    array = []
    # skip header:
    for i in range(headerlength):
        data.readline()
    for line in data:
        line = line.strip() # remove '\n'
        columns = line.split(separator)
        columns = [np.float(col) for col in columns]
        array.append(columns)
    data.close()
    return np.array(array)

def readin_cf06(filename,Angstrom_to_kms):
    '''
    Reads in data from Cen & Fang 2006 (WebPlotDigitize) and converts mA EWs to log10 km/s
    '''
    binsedges = readdata('/net/luttero/data2/cen_fang_2006/%s'%(filename), headerlength=1)
    binsedges = (binsedges.T)[::-1,:]
    binsedges[1] = np.log10(binsedges[1]) -3. + np.log10(Angstrom_to_kms) # mA to km/s
    return binsedges

def wrapspectrum(xvals, yvals):
    '''
    given pixel edge values, tack on two more values to make the last bin in the periodic spectrum equal to the first bin
    '''
    xsize = np.average(np.diff(xvals))
    xvals = np.append(xvals, xvals[-1] + np.arange(1, 3) * xsize)
    if len(yvals.shape) == 1:
        yvals = np.append(yvals, np.array([yvals[0], yvals[0]]))
    else:
        yvals = np.array([np.append(yvals[0, :], np.array([yvals[0, 0], yvals[0, 0]])), np.append(yvals[1, :], np.array([yvals[1, 0], yvals[1, 0]]))])
    return xvals, yvals
    
def add_ax(ax, diff, xory='x', fontsize=fontsize, label=None):

    if xory == 'x':
        ax2 = ax.twiny()
        old_ticklocs = ax.get_xticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
	
	    # use same spacing and number of ticks, but start at first integer value in the new units
        new_lim = old_xlim + diff
        tick_interval = np.average(np.diff(old_ticklocs))
        newticks = np.ceil(new_lim[0]) - np.floor((np.ceil(new_lim[0]) - new_lim[0]) / tick_interval) * tick_interval + np.arange(len(old_ticklocs))*tick_interval
        newticks = np.round(newticks,2)
        newticklabels = [str(int(tick)) if int(tick)== tick else str(tick) for tick in newticks]
   	
	    #print old_ticklocs
        print newticklabels
        #ax2.set_xticks(np.round(old_ticklocs + np.log10(rho_to_nh),2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        #ax2.set_xticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)]) 
        ax2.set_xticks(newticks - diff) # old locations, shifted just so that the round-off works out
        ax2.set_xticklabels(newticklabels)    
        if label is not None:         
            ax2.set_xlabel(label, fontsize=fontsize)

    else:
        ax2 = ax.twinx()
        old_ticklocs = ax.get_yticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
        ax2.set_yticks(np.round(old_ticklocs + diff, 2) - diff) # old locations, shifted just so that the round-off works out
        ax2.set_yticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)])        
        if label is not None:
            ax2.set_ylabel(label, fontsize=fontsize)
    ax2.set_xlim(old_xlim)
    ax2.set_ylim(old_ylim)
    ax2.tick_params(labelsize=fontsize - 1, axis='both')
    return ax2

def interp_fill_between(binsedges1, binsedges2):
    '''
    Takes in two binsedges (y,x) datasets, returns combined x values and interpolated y values for those points 
    assumes x values are sorted low-to-high
    '''
    x1 = binsedges1[1]
    x2 = binsedges2[1]
    y1 = binsedges1[0]
    y2 = binsedges2[0]
    allx = np.sort(np.array(list(x1) + list(x2[np.all(x1[:,np.newaxis] != x2[np.newaxis, :], axis = 0)]))) # all unique x values
    allx = allx[allx >= max(x1[0],x2[0])] # interpolate, don't extrapolate. For fill between, the full x range must match
    allx = allx[allx <= min(x1[-1],x2[-1])]
    y1all = np.interp(allx, x1, y1) # linear interpolation
    y2all = np.interp(allx, x2, y2) # linear interpolation   
    return allx, y1all, y2all

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

def getminmax2d(hist3d, axis=None, log=True, pixdens=False): 
    # axis = axis to sum over; None -> don't sum over any axes 
    # now works for histgrams of general dimensions
    if axis is None:
        imgtoplot = hist3d['bins']
    else:
        imgtoplot = np.sum(hist3d['bins'], axis=axis)
    if pixdens:
        if axis is None:
            naxis = range(len(hist3d['edges']))
        else:
            naxis = list(set(range(len(hist3d['edges']))) - set(axis)) # axes not to sum over
        naxis.sort() 
        numdims = len(naxis)
        binsizes = [np.diff(hist3d['edges'][axisi]) for axisi in naxis] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] + [slice(None,None,None)] + baseinds[ind+1:])] for ind in range(numdims)])
        imgtoplot /= normmatrix
        del normmatrix
    finite = np.isfinite(imgtoplot)
    if log:
        imin = np.min(imgtoplot[np.logical_and(finite, imgtoplot > 0) ])
        imax = np.max(imgtoplot[np.logical_and(finite, imgtoplot > 0) ])
        imin = np.log10(imin)
        imax = np.log10(imax)
    else:
        imin = np.min(imgtoplot[np.isfinite(imgtoplot)])
        imax = np.max(imgtoplot[np.isfinite(imgtoplot)])
    return imin, imax
    
def cddfsubplot1(ax, binsedges, subtitle=None, subfigind=None, subtitleloc=None, subfigindloc=None,\
                 xlabel=None, ylabel=None, fontsize=fontsize,\
                 colors=None, labels=None, linestyles=None,\
                 xlim=Nrange, ylim=(-21.,-11.), xticklabels=True, yticklabels=True,  ylog=False, takeylog=True, steppost=False, linemid=True,\
                 legend_loc=None, legend_ncol=1, legend_title=None, dolegend=True):
    '''
    function with many, many options because it is used for basically all the subplots in the mulitplot functions below
    input:      ax: ax object to plot cddfs in
    binedges:   list of bins,edges to plot
    '''

    if xlabel is not None:
        ax.set_xlabel(xlabel,fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel,fontsize=fontsize)
    if ylog:
        ax.set_yscale('log', nonposy='clip')
    ax.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, labelleft=yticklabels, labelbottom=xticklabels, which='both')
    ax.minorticks_on()
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if subtitle is not None:
        if subtitleloc is None:
            subtitleloc = (0.95, 0.95)
        ax.text(subtitleloc[0],subtitleloc[1], subtitle, fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'top', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    if subfigind is not None:
        if subfigindloc is None:
            ax.text(0.85,0.05,subfigind, fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)
        else: 
            ax.text(subfigindloc[0],subfigindloc[1],subfigind,fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax.transAxes)
    numh = len(binsedges)
    if colors == None:
        colors = ['blue','red','green','gray','orange','purple','gold','cyan','black']
    elif not (isinstance(colors, list) or isinstance(colors, tuple) or isinstance(colors, np.ndarray)):
         colors = (colors,)*numh
    elif numh >len(colors):
        print('Use a larger color array')
    if labels is None:
        labels = list((None,)*numh)
    if linestyles is None: 
        linestyles = list(('solid',)*numh)
    kwargs = {}

    for i in range(numh):
        if linestyles[i] == 'dashdotdot':
            linestyles[i] = 'dashed'
            kwargs = {'dashes': [4, 2, 1, 2, 1, 2]}
        elif 'dashes' in kwargs.keys():
            del kwargs['dashes']

        if steppost:
            if takeylog: # plot log10 y values
                ax.step(binsedges[i][1],np.log10(binsedges[i][0]),where = 'post',color=colors[i], label = labels[i], linewidth=2,linestyle=linestyles[i],**kwargs)
            else:
                ax.step(binsedges[i][1],binsedges[i][0],where = 'post',color=colors[i], label = labels[i], linewidth=2,linestyle=linestyles[i],**kwargs)
        elif linemid:
            if takeylog: # plot log10 y values
                ax.plot(binsedges[i][1] + 0.5*np.append(np.diff(binsedges[i][1]), np.diff(binsedges[i][1])[-1]), np.log10(binsedges[i][0]), color=colors[i], label=labels[i], linewidth=2, linestyle=linestyles[i], **kwargs)
            else:
                ax.plot(binsedges[i][1] + 0.5*np.append(np.diff(binsedges[i][1]), np.diff(binsedges[i][1])[-1]), binsedges[i][0], color=colors[i], label = labels[i], linewidth=2,linestyle=linestyles[i],**kwargs)
        else: #just plot a line: more suitable for cumulative plots
            if takeylog: # plot log10 y values
                ax.plot(binsedges[i][1],np.log10(binsedges[i][0]),color=colors[i], label = labels[i], linewidth=2,linestyle=linestyles[i],**kwargs)
            else:
                ax.plot(binsedges[i][1],binsedges[i][0],color=colors[i], label = labels[i], linewidth=2,linestyle=linestyles[i],**kwargs)
    if dolegend:
        if legend_loc is None:
            legend = ax.legend(fontsize=fontsize, title=legend_title)
        else:
            legend = ax.legend(fontsize=fontsize, loc=legend_loc, ncol=legend_ncol, title=legend_title)
        if legend_title is not None:
            legend.get_title().set_fontsize(fontsize)
        
    #ax.set_aspect('equal')


###################### 2d histogram stuff #####################################

# both in cgs, using primordial abundance from EAGLE
rho_to_nh = 0.752/(c.atomw_H*c.u)
#retrieved with mc.getcosmopars
cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 2.220446049250313e-16}
cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}
logrhob_av_ea_28 = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
logrhob_av_ea_27 = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )

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

def getsortedsubhists(group, searchkey):
    '''
    returns all the subhistograms in the hdf5 group containing the searchkey 
    string, sorted by their minvals
    specific for the hdf5 format I used for histograms and their partial sums
    '''
    # find the keys we're after (really just excludes dimension, hist_all, etc.)
    allkeys = group.keys()
    keys = list(set([ds if searchkey in ds else None for ds in allkeys]))
    keys.remove(None)
    
    vals = {key: {'hist': np.array(group[key]),\
                  'minval': group[key].attrs['minval'],\
                  'maxval': group[key].attrs['maxval'],\
                  'minincl': group[key].attrs['minincl'],\
                  'maxincl': group[key].attrs['maxincl'],\
                  } for key in keys}
    def sorthelper1(key):
        return vals[key]['minval'] 
    keys.sort(key=sorthelper1)
    
    hists    = [vals[key]['hist'] for key in keys]
    minvals  = [vals[key]['minval'] for key in keys]
    maxvals  = [vals[key]['maxval'] for key in keys]
    minincl  = [vals[key]['minincl'] for key in keys]
    maxincl  = [vals[key]['maxincl'] for key in keys]
    return hists, minvals, maxvals, minincl, maxincl

def fmtfp1(number):
    '''
    meant to catch a very specific case where %.1f print negtive zero
    '''
    rv = np.round(number, 1)
    if rv == 0.: # holds for -0.0
        rv = 0.0 # assign positive zero
    return '%.1f'%rv

#############################
## main plotting functions ##
#############################
    
################################## CDDFS ######################################

# Illustris, TNG, main, 0.1Solar, fixz
def cddfplots_overview_tngcomp(number, fontsize=fontsize):
    '''
    O7, O8 comparison to IllustrisTNG 100-1 
    + -- + -- +
    | O7 | O8 |
    + -- + -- +

    saved as cddf_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_T4EOS_z-projection_8-x-6.250000-addto-12.500000slices_offset0_and_illustrisTNG.pdf
    Poisson error band was tiny and not visible on this plot scale
    '''

    # load all the data
    o7 = getcddf('cddf_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    #o7losmatch = getcddf('cddf_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000-addto-12.500000slices_offset0_range-25.0-28.0_1060bins.npz')
    o7_fixz = getcddf('cddf_coldens_o7_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    #o7_tohist = mc.cddf_over_pixcount(0.0, 6.25, 16*32000**2, o7[1], cosmopars=None)
    #o7_counts = o7[0]*o7_tohist
    #o7_err = np.sqrt(o7_counts)
    #o7_band = [np.max(np.array([np.log10((o7_counts - o7_err) / o7_tohist), np.ones(len(o7[0]))*-100.]), axis=0), np.max(np.array([np.log10((o7_counts + o7_err) / o7_tohist), np.ones(len(o7[0]))*-100.]), axis=0)]
    
    o8 = getcddf('cddf_coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    #o8losmatch = getcddf('cddf_coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000-addto-12.500000slices_offset0_range-25.0-28.0_1060bins.npz')
    o8_fixz = getcddf('cddf_coldens_o8_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    #o8_tohist = mc.cddf_over_pixcount(0.0, 6.25, 16*32000**2, o8[1], cosmopars=None)
    #o8_counts = o8[0]*o8_tohist
    #o8_err = np.sqrt(o8_counts)
    #o8_band = [np.max(np.array([np.log10((o8_counts - o8_err) / o8_tohist), np.ones(len(o8[0]))*-100.]), axis=0), np.max(np.array([np.log10((o8_counts + o8_err) / o8_tohist), np.ones(len(o8[0]))*-100.]), axis=0)]
    
    # IllutrisTNG data
    tngfile = h5py.File('/net/luttero/data2/illustris/Box_CDDF_TNG100-1_nOVII_depth10_099.hdf5','r')
    edgesbinstng_o7 = np.log10(np.array(tngfile['Box_CDDF_nOVII_depth10'])) # file values are not log)
    tngfile = h5py.File('/net/luttero/data2/illustris/Box_CDDF_TNG100-1_nOVIII_depth10_099.hdf5','r')
    edgesbinstng_o8 = np.log10(np.array(tngfile['Box_CDDF_nOVIII_depth10'])) # file values are not log
    
    ncols = 2
    nrows = 1

    # uses gridspec and add_subplot under the hood
    fig, (ax1, ax2) = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row', figsize=figsize_1r)  
    # illustrisTNG already has y in log10, so plot separately
    #ax1.fill_between(o7[1], o7_band[0], o7_band[1], color='red', alpha=0.5)
    if number < 3:
        sel = slice(0, 1, None)
    else:
        sel = slice(0, number, None)
    cddfsubplot1(ax1, [o7, o7_fixz][sel], subtitle='O VII', subfigind=None, xlabel=logNo7label, ylabel=cddflabel,\
                 colors=['red', 'lightsalmon'], labels=[r'EAGLE', r'EA-$0.1\,Z_{\odot}$'], linestyles=['solid', 'dotted'],\
                 fontsize=fontsize, xlim=Nrange, ylim=(-19.5,-11.),\
                 xticklabels=True, yticklabels=True, legend_loc=None, takeylog=True, ylog=False, subfigindloc=(0.95, 0.72)) 
    if number > 1:
        ax1.plot(edgesbinstng_o7[0], edgesbinstng_o7[1], color='firebrick', linestyle='dashed', label = r'TNG-100-1')
    ax1.legend(fontsize=fontsize, loc='lower left')
 
    #ax2.fill_between(o8[1], o8_band[0], o8_band[1], color='blue', alpha=0.5)
    cddfsubplot1(ax2, [o8, o8_fixz][sel], subtitle='O VIII', subfigind=None, xlabel=logNo8label, ylabel=None,\
                 colors=['blue', 'dodgerblue'], labels=[r'EAGLE', r'EA-$0.1\,Z_{\odot}$'], linestyles=['solid', 'dotted'],\
                 fontsize=fontsize, xlim=Nrange, ylim=(-19.5,-11.),\
                 xticklabels=True, yticklabels=False, legend_loc=None, takeylog=True, ylog=False, subfigindloc=(0.95, 0.72)) 
    if number > 1:
        ax2.plot(edgesbinstng_o8[0], edgesbinstng_o8[1], color='navy', linestyle='dashed', label=r'TNG-100-1')
    ax2.legend(fontsize=fontsize, loc='lower left')

    fig.tight_layout()
    plt.savefig(outputdir + 'cddf_coldens_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_T4EOS_z-projection_8-x-6.250000-addto-12.500000slices_offset0_and_illustrisTNG_number%i.eps'%number, format='eps')


def cddfplots_agneffect(fontsize=fontsize, fixz=False):
    # load CDDFs
    #o7_base = getcddf('cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o7_noagn50 = getcddf('cddf_coldens_o7_L0050N0752REF_NOAGN_27_test3.1_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o7_ref50 = getcddf('cddf_coldens_o7_L0050N0752_27_test3.1_PtAb_C2Sm_16000pix_3.125slice_zcen-all_z-projection_T4EOS_8-x-3.125000-addto-6.250000slices_offset0_range-25.0-28.0_1060bins.npz')
    binsedges_o7 = [o7_ref50, o7_noagn50]

    #o7_base_fixz = getcddf('cddf_coldens_o7_L0100N1504_27_test3_0.1solarAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o7_noagn50_fixz = getcddf('cddf_coldens_o7_L0050N0752REF_NOAGN_27_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o7_ref50_fixz = getcddf('cddf_coldens_o7_L0050N0752_27_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    binsedges_o7_fixz = [o7_ref50_fixz, o7_noagn50_fixz]

    #o8_base = getcddf('cddf_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o8_noagn50 = getcddf('cddf_coldens_o8_L0050N0752REF_NOAGN_27_test3.1_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o8_ref50 = getcddf('cddf_coldens_o8_L0050N0752_27_test3.1_PtAb_C2Sm_16000pix_3.125slice_zcen-all_z-projection_T4EOS_8-x-3.125000-addto-6.250000slices_offset0_range-25.0-28.0_1060bins.npz')
    binsedges_o8 = [o8_ref50, o8_noagn50]

    #o8_base_fixz = getcddf('cddf_coldens_o8_L0100N1504_27_test3_0.1solarAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o8_noagn50_fixz = getcddf('cddf_coldens_o8_L0050N0752REF_NOAGN_27_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    o8_ref50_fixz = getcddf('cddf_coldens_o8_L0050N0752_27_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_8-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    binsedges_o8_fixz = [o8_ref50_fixz, o8_noagn50_fixz]
    
    
    ylim_abs = (-21., -11.)
    ylim_rel = (-1., 4.)
    xlim = (Nrange[0], max(Nrange[1], 17.))
    ylim_abs = (-21., -11.)
    xlim_fixz = (13.5, 16.3)
    ylabel_rel = cddfrellabel%('\mathrm{R{\endash}100}')
    colors_o7  = ['red','sienna']
    colors_o8  = ['blue','dodgerblue']
    linestyles = ['solid','dashed']
    labels = ['Ref-50', 'noAGN-50']
    elabels = [None, None, None]
    sfiloc = (0.15, 0.03)
    
    plt.figure(figsize=figsize_1r)
    grid1  = gsp.GridSpec(1, 2, width_ratios=[1., 1.], hspace=0.1, wspace=0.1)
    (ax1, ax2) = tuple([plt.subplot(grid1[yi]) for yi in range(2)])
    
    if not fixz:
        cddfsubplot1(ax1, binsedges_o7, subtitle='O VII', subfigind=None, subfigindloc=None,\
                 xlabel=logNo7label, ylabel=cddflabel, xticklabels=True, yticklabels=True,\
                 colors=colors_o7, labels=labels, linestyles=linestyles,\
                 fontsize=fontsize, xlim=xlim, ylim=ylim_abs)
        cddfsubplot1(ax2, binsedges_o8, subtitle='O VIII', subfigind=None, subfigindloc=None,\
                 xlabel=logNo8label, ylabel=None, xticklabels=True, yticklabels=False,\
                 colors=colors_o8, labels=labels, linestyles=linestyles,\
                 fontsize=fontsize, xlim=xlim, ylim=ylim_abs)    
    else:
        cddfsubplot1(ax1, binsedges_o7_fixz, subtitle=r'O VII, $0.1Z_\odot$', subfigind=None, subfigindloc=None,\
                 xlabel=logNo7label, ylabel=cddflabel, xticklabels=True, yticklabels=True,\
                 colors=colors_o7 , labels=labels, linestyles=linestyles,\
                 fontsize=fontsize, xlim=xlim_fixz, ylim=ylim_abs)
        cddfsubplot1(ax2, binsedges_o8_fixz, subtitle=r'O VIII, $0.1Z_\odot$', subfigind=None, subfigindloc=None,\
                 xlabel=logNo8label, ylabel=None, xticklabels=True, yticklabels=False,\
                 colors=colors_o8 , labels=labels, linestyles=linestyles,\
                 fontsize=fontsize, xlim=xlim_fixz, ylim=ylim_abs)

    if fixz:
        Zstr = '0.1solarAb'
    else:
        Zstr =  'PtAb'
    plt.savefig(outputdir + 'cddf_coldens_o7-o8_L0050N0752REF-NOAGN-L0100N1504_27_test3.1_%s_C2Sm_16000pix_3.125slice_zcen-all_z-projection_T4EOS_8-x-6.25slices.eps'%Zstr, format='eps', bbox_inches='tight')
    

def plotEWdists_o78_litcomp(fontsize=fontsize):
    '''
    EW distributions using various col. dens. to EW conversions, compared to Cen & Fang GSW (with feedback) models, and Branchini 2009 model B2 (with rho-Z scatter)
    currently assumes B+09, CF06 EWs are rest-frame
    saved as EWcumul_specwizard_map_match_coldens_o7-o8_L0100N1504_27_test3.1-3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_mtx-NEWconv_branchini_etal_2009_cen_fang_2006.pdf
    '''
    sfi_a = '(a)'
    sfi_b = '(b)'
    fig, ax1 = plt.subplots(ncols=1, nrows=1, sharex=True, sharey='row', figsize=(figwidth, 3.5))

    # retrieve histogram bins, edges for o7, o8:
    # lin/fit and 13/14 are extensions to lower column densities -> ignore for this plot range (> 0.1 mA)
    datafile = h5py.File(datadir + 'specwizard_misc.hdf5', 'r')
    retrievekeys_o7 = ['best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                       ]
    retrievekeys_o8 = ['best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                       ]
    binsedges_o7 = {key: [np.array(datafile['EWdists_o7_snap27/%s/bins'%key]), np.array(datafile['EWdists_o7_snap27/%s/edges'%key])] for key in retrievekeys_o7}
    binsedges_o8 = {key: [np.array(datafile['EWdists_o8_snap27/%s/bins'%key]), np.array(datafile['EWdists_o8_snap27/%s/edges'%key])] for key in retrievekeys_o8}
    cosmopars_o7 = {key: datafile['EWdists_o7_snap27/cosmopars'].attrs[key] for key in datafile['EWdists_o7_snap27/cosmopars'].attrs.keys()}
    cosmopars_o8 = {key: datafile['EWdists_o8_snap27/cosmopars'].attrs[key] for key in datafile['EWdists_o8_snap27/cosmopars'].attrs.keys()}
    simdata_o7 = {key: datafile['EWdists_o7_snap27/simdata'].attrs[key] for key in datafile['EWdists_o7_snap27/simdata'].attrs.keys()}
    simdata_o8 = {key: datafile['EWdists_o8_snap27/simdata'].attrs[key] for key in datafile['EWdists_o8_snap27/simdata'].attrs.keys()} 

    dztot_o8 = float(simdata_o8['numpix'])**2*mc.getdz(cosmopars_o8['z'], cosmopars_o8['boxsize']/cosmopars_o8['h'], cosmopars=cosmopars_o8)
    dztot_o7 = float(simdata_o7['numpix'])**2*mc.getdz(cosmopars_o7['z'], cosmopars_o7['boxsize']/cosmopars_o7['h'], cosmopars=cosmopars_o7)
    angstrom_to_kms_o7 = c.c/lambda_rest['o7']/1.e5
    angstrom_to_kms_o8 = c.c/lambda_rest['o8']/1.e5 # fosc-weighted average wavelength

    binsedges_o8_proc_cumuldz = [[np.cumsum(binsedges_o8[key][0][::-1])[::-1]/dztot_o8, np.log10(angstrom_to_kms_o8) + binsedges_o8[key][1][:-1]] for key in retrievekeys_o7] # , 'fit_13_sub' 
    binsedges_o7_proc_cumuldz = [[np.cumsum(binsedges_o7[key][0][::-1])[::-1]/dztot_o7, np.log10(angstrom_to_kms_o7) + binsedges_o7[key][1][:-1]] for key in retrievekeys_o8] # , 'fit_13_sub' 

#    ## read in Branchini et al. 2009 data, process
#    binsedges_branchini2009_o7 = readdata('/net/luttero/data2/branchini_2009/o7_chen_cumul.dat', headerlength=0)
#    binsedges_branchini2009_o7 = (binsedges_branchini2009_o7.T)[::-1,:] # read in as EW, dndz, EW, dndz, ...
#    binsedges_branchini2009_o7[1] = np.log10(binsedges_branchini2009_o7[1])
#
#    binsedges_branchini2009_o8 = readdata('/net/luttero/data2/branchini_2009/o8_chen_cumul.dat', headerlength=0)
#    binsedges_branchini2009_o8 = (binsedges_branchini2009_o8.T)[::-1,:] # read in as EW, dndz, EW, dndz, ...
#    binsedges_branchini2009_o8[1] = np.log10(binsedges_branchini2009_o8[1])
#
#    ## read in Cen & Fang 2006 data, process
#    binsedges_o7_cf06_GL_u = readin_cf06('o7EW_GSW_LTE_upper.dat',angstrom_to_kms_o7)
#    binsedges_o7_cf06_GL_l = readin_cf06('o7EW_GSW_LTE_lower.dat',angstrom_to_kms_o7)
#    binsedges_o7_cf06_GnL_u = readin_cf06('o7EW_GSW_noLTE_upper.dat',angstrom_to_kms_o7)
#    binsedges_o7_cf06_GnL_l = readin_cf06('o7EW_GSW_noLTE_lower.dat',angstrom_to_kms_o7)
#
#    binsedges_o8_cf06_GL_u = readin_cf06('o8EW_GSW_LTE_upper.dat',angstrom_to_kms_o8)
#    binsedges_o8_cf06_GL_l = readin_cf06('o8EW_GSW_LTE_lower.dat',angstrom_to_kms_o8)
#    binsedges_o8_cf06_GnL_u = readin_cf06('o8EW_GSW_noLTE_upper.dat',angstrom_to_kms_o8)
#    binsedges_o8_cf06_GnL_l = readin_cf06('o8EW_GSW_noLTE_lower.dat',angstrom_to_kms_o8)
#
#    edges_o7_cf06_GL_fb, bins_o7_cf06_GL_u_fb, bins_o7_cf06_GL_l_fb = interp_fill_between(binsedges_o7_cf06_GL_u,binsedges_o7_cf06_GL_l)
#    edges_o7_cf06_GnL_fb, bins_o7_cf06_GnL_u_fb, bins_o7_cf06_GnL_l_fb = interp_fill_between(binsedges_o7_cf06_GnL_u,binsedges_o7_cf06_GnL_l)
#    edges_o8_cf06_GL_fb, bins_o8_cf06_GL_u_fb, bins_o8_cf06_GL_l_fb = interp_fill_between(binsedges_o8_cf06_GL_u,binsedges_o8_cf06_GL_l)
#    edges_o8_cf06_GnL_fb, bins_o8_cf06_GnL_u_fb, bins_o8_cf06_GnL_l_fb = interp_fill_between(binsedges_o8_cf06_GnL_u,binsedges_o8_cf06_GnL_l)
#    bins_o7_cf06_GL_m_fb  = np.average(np.array([np.log10(bins_o7_cf06_GL_u_fb), np.log10(bins_o7_cf06_GL_l_fb)]), axis=0)
#    bins_o7_cf06_GnL_m_fb = np.average(np.array([np.log10(bins_o7_cf06_GnL_u_fb), np.log10(bins_o7_cf06_GnL_l_fb)]), axis=0)
#    bins_o8_cf06_GL_m_fb  = np.average(np.array([np.log10(bins_o8_cf06_GL_u_fb), np.log10(bins_o8_cf06_GL_l_fb)]), axis=0)
#    bins_o8_cf06_GnL_m_fb = np.average(np.array([np.log10(bins_o8_cf06_GnL_u_fb), np.log10(bins_o8_cf06_GnL_l_fb)]), axis=0)
    
    # read in Nicastro et al. 2018 data (webplotdigitize) and process
    data_nicastro_etal_2018 = readdata('/net/luttero/data2/nicastro_etal_2018/webplotdigitize_EWdistmeasurments.dat', headerlength=2, separator='\t')
    data_nicastro_etal_2018 = np.log10(data_nicastro_etal_2018)
    data_nicastro_etal_2018[:, 0] += np.log10(angstrom_to_kms_o7) - 3. # mA -> km/s
    points_n18 = (data_nicastro_etal_2018[::5, :]).T # EW rest mA array, dn(>EW)/dz array
    errends_u_n18 = (data_nicastro_etal_2018[1::5, 1])
    errends_d_n18 = (data_nicastro_etal_2018[2::5, 1])
    errends_l_n18 = (data_nicastro_etal_2018[3::5, 0])
    errends_r_n18 = (data_nicastro_etal_2018[4::5, 0])
    xerrs_n18 = [points_n18[0, :] - errends_l_n18, errends_r_n18 - points_n18[0, :]]
    yerrs_n18 = [points_n18[1, :] - errends_d_n18, errends_u_n18 - points_n18[1, :]]

    
    cddfsubplot1(ax1, binsedges_o7_proc_cumuldz, subtitle='O VII', subtitleloc=(0.95,0.962), subfigind=None, subfigindloc=(0.95,0.80),\
                 xlabel=r'$\log_{10}\, EW_{\mathrm{O VII}} \; [\mathrm{km}\,\mathrm{s}^{-1}]$', ylabel=r'$\log_{10} \, \mathrm{d}N(>EW)\,/\,\mathrm{d}z$',\
                 colors = ['blue'], labels=['EAGLE'], linestyles=['solid'], fontsize=fontsize,
                 xlim = (0.5,2.35), ylim=(-1.05 , 1.85), xticklabels=True, yticklabels=True, ylog=False, takeylog=True, steppost=False)
    #ax1.plot(edges_o7_cf06_GL_fb, bins_o7_cf06_GL_m_fb, label='CF06, G-L', color='red', linestyle = 'dashdot')
    #ax1.plot(edges_o7_cf06_GnL_fb, bins_o7_cf06_GnL_m_fb, label='CF06, G-nL', color='purple', linestyle = 'dashdot')
    #ax1.plot(binsedges_branchini2009_o7[1] - np.log10(1.25), np.log10(binsedges_branchini2009_o7[0]), label='B+09, B2', linestyle='dashed', color='green') # observed-frame EWs from redshifts 0.5 to 0 -> bracketed rest-frame EWs
    #ax1.fill_between(edges_o7_cf06_GL_fb, np.log10(bins_o7_cf06_GL_l_fb), np.log10(bins_o7_cf06_GL_u_fb), label = 'CF06, G-L', facecolors=(1.,0.,0.,0.5), edgecolor='red', linestyle = 'dashdot')
    #ax1.fill_between(edges_o7_cf06_GnL_fb, np.log10(bins_o7_cf06_GnL_l_fb), np.log10(bins_o7_cf06_GnL_u_fb), label = 'CF06, G-nL', facecolors=(0.3,0.6,0.2,0.7), edgecolor=(0.3,0.6,0.2,1.), linestyle = 'dashdot')
    #ax1.fill_betweenx(np.log10(binsedges_branchini2009_o7[0]), binsedges_branchini2009_o7[1] - np.log10(1.5), binsedges_branchini2009_o7[1], label = 'B+09, B2', linestyle = 'dashed', facecolors=(0.5,0.5,0.5,0.5), edgecolor=(0.5,0.5,0.5,1.)) # observed-frame EWs from redshifts 0.5 to 0 -> bracketed rest-frame EWs
    ax1.errorbar(points_n18[0, :], points_n18[1, :], yerr=yerrs_n18, xerr=xerrs_n18, fmt='.', color='black', label=r'Nicastro $et$ $al$. (2018), Nature', zorder=5, capsize=2)
    ax1.legend(fontsize=fontsize, loc='lower left')

    aax1 = add_ax(ax1, -1 * np.log10(angstrom_to_kms_o7) + 3, xory='x', fontsize=fontsize, label=r'$\log_{10}\, \mathrm{m\AA}$')
    aax1.tick_params(labelsize=fontsize - 1, direction='in', top=True, which='both')
    aax1.minorticks_on()

    #cddfsubplot1(ax2, binsedges_o8_proc_cumuldz, subtitle='O VIII', subtitleloc=(0.95,0.962), subfigind=None, subfigindloc=(0.95,0.80),\
    #             xlabel=r'$\log_{10}\,EW_{\mathrm{O VIII}} \; [\mathrm{km}\,\mathrm{s}^{-1}]$', ylabel=None,\
    #             colors=['blue', 'cyan'], labels=['Eagle-100', 'Eagle-6.25'], linestyles=['solid', 'solid', 'dashed'],
    #             fontsize=fontsize, xlim=(0.5,2.45), ylim=(-1.05, 1.85), xticklabels=True, yticklabels=False,\
    #             legend_loc='lower left', ylog=False, takeylog=True, legend_ncol=1, steppost=False)

    #ax2.plot(edges_o8_cf06_GL_fb, bins_o8_cf06_GL_m_fb, label=None, color='red', linestyle = 'dashdot')
    #ax2.plot(edges_o8_cf06_GnL_fb, bins_o8_cf06_GnL_m_fb, label=None, color='purple', linestyle = 'dashdot')
    #ax2.plot(binsedges_branchini2009_o8[1] - np.log10(1.25), np.log10(binsedges_branchini2009_o8[0]), label=None, linestyle='dashed', color='green') # observed-frame EWs from redshifts 0.5 to 0 -> bracketed rest-frame EWs
    #ax2.fill_between(edges_o8_cf06_GL_fb, np.log10(bins_o8_cf06_GL_l_fb), np.log10(bins_o8_cf06_GL_u_fb), label = 'CF06, G-L', facecolors=(1.,0.,0.,0.5), edgecolor='red', linestyle = 'dashdot')
    #ax2.fill_between(edges_o8_cf06_GnL_fb, np.log10(bins_o8_cf06_GnL_l_fb), np.log10(bins_o8_cf06_GnL_u_fb), label = 'CF06, G-nL', facecolors=(0.3,0.6,0.2,0.7), edgecolor=(0.3,0.6,0.2,1.), linestyle = 'dashdot')
    #ax2.fill_betweenx(np.log10(binsedges_branchini2009_o8[0]), binsedges_branchini2009_o8[1] - np.log10(1.5), binsedges_branchini2009_o8[1], label = 'B+09, B2', linestyle = 'dashed', facecolors=(0.5,0.5,0.5,0.5), edgecolor=(0.5,0.5,0.5,1.)) # observed-frame EWs from redshifts 0.5 to 0 -> bracketed rest-frame EWs
    #ax2.legend(fontsize=fontsize, loc='lower left')

    #aax2 = add_ax(ax2, -1 * np.log10(angstrom_to_kms_o8) + 3, xory='x', fontsize=fontsize, label=r'$\log_{10}\, \mathrm{m\AA}$')
    #aax2.tick_params(labelsize=fontsize - 1, direction='in', top=True, which='both')
    #aax2.minorticks_on()

    plt.savefig(outputdir + 'EWcumul_specwizard_map_match_coldens_o7-o8_L0100N1504_27_test3.1-3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_mtx-NEWconv_branchini_etal_2009_cen_fang_2006.eps', format='eps', bbox_inches='tight')


def plot_curveofgrowth_o7_o8(fontsize=fontsize, alpha=0.05):
    '''
    EW as a function of col. dens. with various b-parameter values indicated. Sample 3 data 
    '''
    # fig, axes setup and settings
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(figwidth, 4.))
    labelleft_ax1 = True
    labelbottom_ax1 = True

    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both',\
                   labelleft=labelleft_ax1, labeltop=False, labelbottom=labelbottom_ax1, labelright=False)

    # prevent tick overlap
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 10))

    if labelbottom_ax1:
        ax1.set_xlabel(logNlabel,fontsize=fontsize)  
    if labelleft_ax1:  
        ax1.set_ylabel(logEWlabel,fontsize=fontsize)        

    ylim = (-0.7, 1.8)
    xlim = (14.2, 18.15)
    ax1.set_ylim(ylim)
    ax1.set_xlim(xlim)
    
    # subtitles and subplot labels
    ax1.text(0.05,0.95,'O VII',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes, bbox=dict(facecolor='white',alpha=0.3))

    # hdf5 file with the sample3 data
    datafile = h5py.File(datadir + 'specwizard_misc.hdf5', 'r')
    coldens_o7 = np.array(datafile['specwizard_projection_match_o7/column_density_specwizard'])
    EW_o7      = np.array(datafile['specwizard_projection_match_o7/equivalent_width'])
    indsel_o7  = np.array(datafile['specwizard_projection_match_o7/indices_selected_for_o7'])
    coldens_o8 = np.array(datafile['specwizard_projection_match_o8/column_density_specwizard'])
    EW_o8      = np.array(datafile['specwizard_projection_match_o8/equivalent_width'])
    indsel_o8  = np.array(datafile['specwizard_projection_match_o8/indices_selected_for_o8'])
    EW_o7_sub      = EW_o7[indsel_o7]
    coldens_o7_sub = coldens_o7[indsel_o7]
    EW_o8_sub      = EW_o8[indsel_o8]
    coldens_o8_sub = coldens_o8[indsel_o8]
    #print np.max(coldens_o7_sub), np.max(coldens_o8_sub), np.max(EW_o7_sub), np.max(EW_o8_sub)
#    nosel_o7 = np.array(list(set(np.arange(len(coldens_o7))) - set(indsel_o7)))
#    nosel_o8 = np.array(list(set(np.arange(len(coldens_o8))) - set(indsel_o8)))
#    EW_o7_other      = EW_o7[nosel_o7]
#    coldens_o7_other = coldens_o7[nosel_o7]
#    EW_o8_other      = EW_o8[nosel_o8]
#    coldens_o8_other = coldens_o8[nosel_o8]
    #plot the subsamples
    colors = ['blue', 'red'] 
    # +3: log10 A -> log10 mA   
    ax1.scatter(coldens_o7_sub, np.log10(EW_o7_sub) + 3., s=10, color=colors[0], alpha=alpha, rasterized=True)
    ax1.scatter([],[],label = 'EAGLE', s=10, color=colors[0], alpha=1.) # adds an alpha=1 actually visible label 

    # add addtional info (o7) -- best fits for subsample fit
    bpar_logfit_o7 = 90.388292*1e5 # best fit for log EW COG
    #bpar_linfit_o7 = 9435311.30181743 # best fit for (lin) EW COG
    bpar_lower_o7 = 50*1e5             # ~ lower envelope
    bpar_vlower_o7 = 20e5
    bpar_upper_o7 = 220*1e5            # ~ upper envelope

    cgrid_o7 = np.arange(np.min(coldens_o7)*0.999,np.max(coldens_o7)+0.999*0.1,0.1)
    ax1.plot(cgrid_o7, np.log10(lingrowthcurve_inv(10**cgrid_o7,'o7')) + 3., color='gray', label='optically thin')
    ax1.plot(cgrid_o7, np.log10(linflatcurveofgrowth_inv(10**cgrid_o7,bpar_vlower_o7,'o7')) + 3., color='sienna', linestyle='dotted', label=r'$b = %i$ km/s'%int(np.round(bpar_vlower_o7/1e5,0)), linewidth=2)
    ax1.plot(cgrid_o7, np.log10(linflatcurveofgrowth_inv(10**cgrid_o7,bpar_lower_o7,'o7')) + 3., color='gold', linestyle='dotted', label=r'$b = %i$ km/s'%int(np.round(bpar_lower_o7/1e5,0)), linewidth=2)
    ax1.plot(cgrid_o7, np.log10(linflatcurveofgrowth_inv(10**cgrid_o7,bpar_logfit_o7,'o7')) + 3., color='cyan', linestyle='solid', label=r'$b = %i$ km/s'%int(np.round(bpar_logfit_o7/1e5,0)), linewidth=2)
    ax1.plot(cgrid_o7, np.log10(linflatcurveofgrowth_inv(10**cgrid_o7,bpar_upper_o7,'o7')) + 3., color='orange', linestyle='dotted', label=r'$b = %i$ km/s'%int(np.round(bpar_upper_o7/1e5,0)), linewidth=2)

    # inidcate EW of 7 mA -> N ~ 15.5 at best-fit COG
    N_ind = 15.5
    EW_ind = np.log10(linflatcurveofgrowth_inv(np.array([10**N_ind]), bpar_logfit_o7, 'o7')[0]) + 3.
    ax1.axhline(EW_ind, 0., (N_ind - xlim[0]) / (xlim[1] - xlim[0]), linestyle='dashed', color='black')
    ax1.axvline(N_ind, 0., (EW_ind - ylim[0]) / (ylim[1] - ylim[0]), linestyle='dashed', color='black')
    # legends    
    ax1.legend(fontsize=fontsize, loc='lower right')

    # stackexchange fix to get the transparencies right
    outname = outputdir + 'specwizard_curve-of-growth_o7_sample3_with_best-fit_EW-logEW_upper_lower_ests_bpar_specwizcoldens'
    plt.savefig(outname + '.pdf', format='pdf', bbox_inches='tight', dpi=400) # dpi for the raterized points
    os.system("pdftops -eps %s.pdf %s.eps"%(outname, outname))

def plot_cddfs_o78_by_delta(number, fontsize=fontsize):
    '''
    snapshot 28, ion-wieghted temperatures
    '''
    # legend looks cut-off with plt.show(), but avoids excessive lower whitespace in the saved pdf 
    fig = plt.figure(figsize=(figwidth, 3.5))
    grid = gsp.GridSpec(1, 2, width_ratios=[4., 1.],\
                        wspace=0.1, hspace=0.0, top=0.95, bottom=0.05, left=0.05, right=0.95) # grispec: nrows, ncols
    axes = np.array([fig.add_subplot(grid[yi]) for yi in [0,1]])
    
    colors_rho_o7 = ['black', 'red', 'orange', 'green', 'blue','purple']
    #colors_rho_o8 = ['black', 'red', 'orange', 'green', 'blue','purple']
    ylabel = r'$\log_{10}\, \left( \partial^2 n / \partial \log_{10} N \, \partial X \right)$'
    ylim = (-5.5, 2.5)
    xlim = (Nrange[0], max(Nrange[1], 17.))
    subfigindloc = (0.95, 0.70)
    subtitleloc = None
    
    with h5py.File(datadir + 'histograms.hdf5', 'r') as datafile:
        keys = datafile['cosmopars_eagle/snap28'].attrs.keys()
        cosmopars  = {key: datafile['cosmopars_eagle/snap28'].attrs[key] for key in keys} 
        logrhob_av = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars['h']**2 * cosmopars['omegab'] / cosmopars['a']**3 )
    
        grp_o7 = datafile['coldens_o7_snap28_in_rhobins']
        hists_o7, minvals_o7, maxvals_o7, minincl_o7, maxincl_o7 = \
            getsortedsubhists(grp_o7, 'Density')
        bins_o7_tot  = np.array(grp_o7['hist_all'])
        edges_o7     = np.array(grp_o7['edges_axis0'])
        minvals_o7   -= logrhob_av # density -> overdensity
        maxvals_o7   -= logrhob_av # density -> overdensity
        bins_o7      = [bins_o7_tot] + hists_o7 
        sep = r'\,\mathrm{\endash}\,'
        labels_o7_subs = [r'$%s %s %s$'%(fmtfp1(minvals_o7[i]), sep, fmtfp1(maxvals_o7[i])) if not minincl_o7[i] and not maxincl_o7[i] else\
                          r'$ < %s$'%(fmtfp1(maxvals_o7[i])) if minincl_o7[i] else\
                          r'$ > %s$'%(fmtfp1(minvals_o7[i]))\
                          for i in range(len(hists_o7))]
        labels_o7 = ['total'] + labels_o7_subs 
        
    dXtot = mc.getdX(cosmopars['z'], 6.25, cosmopars=cosmopars) #*numpixtot is already factored out in the stored histograms
    dlogN_o7 = np.diff(edges_o7)
    binsedges_rho_o7 = [[np.max(np.array([np.log10(bins_o7[i] / (dXtot * dlogN_o7)), -100. * np.ones(len(bins_o7[i]))]), axis=0), edges_o7[:-1]] for i in range(len(bins_o7))]
    
    cddfsubplot1(axes[0], binsedges_rho_o7[:number], subtitle='O VII', subfigind=None,\
                     xlabel=logNo7label, ylabel=ylabel,\
                     colors = colors_rho_o7, labels=None, linestyles=None, fontsize=fontsize,\
                     xlim=xlim, ylim=ylim, xticklabels=True, yticklabels=True,\
                     legend_loc=None, legend_ncol=None,\
                     ylog=False, subfigindloc=subfigindloc, takeylog=False, steppost=True, subtitleloc=subtitleloc, dolegend=False)

    legend_handles_o7 = [mlines.Line2D([], [], color=colors_rho_o7[i], linestyle='solid', label=labels_o7[i], linewidth=2.) for i in range(len(binsedges_rho_o7))]
    legend_o7 = axes[1].legend(handles=legend_handles_o7, fontsize=fontsize, ncol=1, loc='upper center', bbox_to_anchor=(0.5, 1.), columnspacing=None, handlelength=None)
    legend_o7.set_title(logdeltalabel,  prop = {'size': fontsize})
    axes[1].axis('off')
    
    plt.savefig(outputdir + 'cddfs_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_by_rho_number%i.eps'%number, bbox_inches='tight', format='eps')
    

def plot_phasediagrams_by_ion(ion='o7', fontsize=fontsize, number=0):
    '''
    two plots with one function: o7- and o8-weighted 
    density-temperature distributions 
    '''    
    if ion == 'o7':
        ionlabel = 'O\,VII'
        grpname = 'density_temperature_by_o7_snap28_in_No7bins'
        searchkey = 'NO7'
        xlim = (-32. + np.log10(rho_to_nh), -1.)
        ylim = (3., 6.7)
    elif ion == 'o8':
        ionlabel = 'O\,VIII'
        grpname = 'density_temperature_by_o8_snap28_in_No8bins'
        searchkey = 'NO8'
        xlim = (-32. + np.log10(rho_to_nh), -0.5)
        ylim = (3., 7.3)
    else:
        print('ion must be "o7" or "o8"')
        return None
    
    with h5py.File(datadir + 'histograms.hdf5', 'r') as datafile:
        keys = datafile['cosmopars_eagle/snap28'].attrs.keys()
        cosmopars  = {key: datafile['cosmopars_eagle/snap28'].attrs[key] for key in keys} 
        logrhob_av = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars['h']**2 * cosmopars['omegab'] / cosmopars['a']**3 )
        tH = 1./csu.Hubble(cosmopars['z'], cosmopars=cosmopars)
    
        grp = datafile[grpname]
        hists, minvals, maxvals, minincl, maxincl = \
            getsortedsubhists(grp, searchkey)
        #hist_tot = np.array(grp['hist_all'])
        edges_nH = np.array(grp['edges_axis0']) + np.log10(rho_to_nh)
        edges_T = np.array(grp['edges_axis1'])
        sep = r'\,\mathrm{\endash}\,'
        labels_subs = [r'$%s %s %s$'%(fmtfp1(minvals[i]), sep, fmtfp1(maxvals[i])) if not minincl[i] and not maxincl[i] else\
                       r'$ < %s$'%(fmtfp1(maxvals[i])) if minincl[i] else\
                       r'$ > %s$'%(fmtfp1(minvals[i]))\
                       for i in range(len(hists))]
        
#        grp = datafile['cooling_times_snap28']
#        grp_sol = grp['Z_solar']
#        tc_Zsol_nH = np.array(grp_sol['axis0_lognH'])
#        tc_Zsol_T  = np.array(grp_sol['axis1_logT'])
#        tc_Zsol    = np.array(grp_sol['tcool'])
#        #grp_0p1 = grp['Z_0p1solar']
#        #tc_Z0p1_nH = np.array(grp_0p1['axis0_lognH'])
#        #tc_Z0p1_T  = np.array(grp_0p1['axis1_logT'])
#        #tc_Z0p1    = np.array(grp_0p1['tcool'])
#        grp_0 = grp['Z_primordial']
#        tc_Z0_nH = np.array(grp_0['axis0_lognH'])
#        tc_Z0_T  = np.array(grp_0['axis1_logT'])
#        tc_Z0    = np.array(grp_0['tcool'])
#        
#        grp = datafile['ionbal_%s_snap28'%ion]
#        ib = np.array(grp['ionbal'])
#        nH_ib = np.array(grp['log10_nH_cm-3'])
#        T_ib = np.array(grp['log10_temperature_K'])
    
    plt.figure(figsize=(figwidth, 4.))
    grid = gsp.GridSpec(1, 2, width_ratios=[4.,1.], wspace=0.02)
    ax1 = plt.subplot(grid[0], facecolor='white') 
    ax3 = plt.subplot(grid[1])
    #ax3 = plt.subplot(grid[1,:])
    ncols_legend = 1
    legend_loc = 'upper left'
    legend_bbox_to_anchor=(0., 1.)
    cmap = bone_m

    
    # $\log_{10}\, \rho \; [\mathrm{g}\,\mathrm{cm}^{-3}]$,
    ax1.set_xlabel(r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$, $N_{\mathrm{%s}}$-weighted'%ionlabel, fontsize=fontsize)   
    ax1.set_ylabel(r'$\log_{10}\, T \; [\mathrm{K}]$, $N_{\mathrm{%s}}$-weighted'%ionlabel, fontsize=fontsize)     
    #nHlabel = r', f_H = 0.752$'
    #clabel  = r'$\log_{10} \, n_{\mathrm{%s}} \,/\, n_{\mathrm{O}}, f_H = 0.752$'%ionlabel
    
#    levels_tcool = [tH]
#    levels_tcool_labels = [r't_H']
#    linestyles_tcool = ['solid']
#    linewidths_tcool = [2.5, 2]
#    colors_tcool = ['black', 'brown'] # 'cyan'
#    colors_tcool_labels = [r't_c(Z=Z_{\odot})', r't_c(Z=0)'] #  r'$t_c(Z=0.1Z_{\odot})$'
#    
    fraclevels = [0.90] 
    linestyles_Nbins = ['solid']
    colors_Nbins = ['purple', 'red', 'gold', 'limegreen', 'blue', 'magenta'] # (1., 0.6, 0.)
   
    #set average density indicator
    add_rhoavx(ax1, logrhob_av + np.log10(rho_to_nh), color='pink')
#    
#    # plot background and colorbar: ion balance
#    nH_edges_ib = np.append(np.array([nH_ib[0] - 0.5 * (nH_ib[1] - nH_ib[0])]), nH_ib[:-1] + 0.5 * np.diff(nH_ib)) # centres -> edges
#    nH_edges_ib = np.append(nH_edges_ib, np.array([nH_ib[-1] + 0.5 * (nH_ib[-1] - nH_ib[-2])]))
#    T_edges_ib = np.append(np.array([T_ib[0] - 0.5 * (T_ib[1] - T_ib[0])]), T_ib[:-1] + 0.5 * np.diff(T_ib)) # centres -> edges
#    T_edges_ib = np.append(T_edges_ib, np.array([T_ib[-1] + 0.5 * (T_ib[-1] - T_ib[-2])]))
#    img, vmin, vmax = add_2dplot(ax1, ib, [nH_edges_ib - np.log10(rho_to_nh), T_edges_ib], plotaxes=(0, 1), log=True, pixdens=False, shiftx=0., shifty=0., cmap=cmap, vmin=-5., vmax=None) # nH -> rho for plotting
#    #img = ax1.imshow(np.log10(ib).T, interpolation='nearest', origin='lower', extent=(nH_edges_ib[0] - np.log10(rho_to_nh), nH_edges_ib[-1] - np.log10(rho_to_nh), T_edges_ib[0], T_edges_ib[-1]), vmin=-5., cmap=cmap)
#    add_colorbar(ax2, img=img, clabel=clabel, extend='min', fontsize=fontsize)
#    ax2.set_aspect(10.)
#    ax2.tick_params(labelsize=fontsize-1, axis='both')
    
    # plot contour levels for column density subsets
    for hind in range(number, len(hists)):
        add_2dhist_contours(ax1, hists[hind], [edges_nH, edges_T], toplotaxes=(0, 1),\
                            fraclevels=True, levels=fraclevels, legendlabel=None,\
                            shiftx=0., shifty=0., colors=(colors_Nbins[hind],)*len(fraclevels), linestyles=linestyles_Nbins)
	   
#	# add cooling contours    
#    contourdct = {'sol':{'x': tc_Zsol_nH, 'y': tc_Zsol_T, 'z': np.abs(tc_Zsol),\
#                         'kwargs': {'colors': colors_tcool[0], 'levels':levels_tcool, 'linestyles':linestyles_tcool, 'linewidths': linewidths_tcool[0]}},\
#                  'pri':{'x': tc_Z0_nH, 'y': tc_Z0_T, 'z': np.abs(tc_Z0),\
#                         'kwargs': {'colors': colors_tcool[1], 'levels':levels_tcool, 'linestyles':linestyles_tcool, 'linewidths': linewidths_tcool[1]}} }
#    #'0p1':{'x': tc_Z0p1_nH, 'y': tc_Z0p1_T, 'z': np.abs(tc_Z0p1),\
#    #                     'kwargs': {'colors': colors_tcool[2], 'levels':levels_tcool, 'linestyles':linestyles_tcool}} }
#    for key in contourdct.keys():
#        toplot = contourdct[key]
#        ax1.contour(toplot['x'] - np.log10(rho_to_nh), toplot['y'], toplot['z'].T, **(toplot['kwargs']))
#                
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')    
    #ax12 =  add_ax(ax1, np.log10(rho_to_nh), xory='x', label=nHlabel, fontsize=fontsize)
    #ax12.set_ylim(ylim)
    #ax12.set_xlim(xlim)
    #xlim12 = ax12.get_xlim()
    #ylim12 = ax12.get_ylim()
    #ax12.set_aspect((xlim12[1] - xlim12[0]) / (ylim12[1] - ylim12[0]), adjustable='box-forced')    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize - 1, direction = 'in', right=True, top=True, axis='both', which='both', color='black')  
    #ax12.minorticks_on()
    #ax12.tick_params(labelsize=fontsize - 1, direction = 'in', right = False, left = False, top = True, bottom = False, axis='both', which='both', color='black')
   
    # set up legend in ax below main figure
    handles_Nbins = [mlines.Line2D([], [], color=colors_Nbins[i], linestyle='solid', label=labels_subs[i]) for i in range(len(labels_subs))]
    handles_subs, labels_subs = ax1.get_legend_handles_labels() # rho_av
#    level_legend_handles = [mlines.Line2D([], [], color='gray', linestyle=linestyles_Nbins[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    #handles_tcool_linestyles = [mlines.Line2D([], [], color='gray', linestyle=linestyles_tcool[i], label=levels_tcool_labels[i]) for i in range(len(levels_tcool))]
    #handles_tcool_colors = [mlines.Line2D([], [], color=colors_tcool[i], linestyle='solid', label=colors_tcool_labels[i]) for i in range(len(colors_tcool))]
#    handles_tcool = [mlines.Line2D([], [], color=colors_tcool[ci], linewidth=linewidths_tcool[ci], linestyle=linestyles_tcool[li], label=r'$%s = %s$'%(colors_tcool_labels[ci], levels_tcool_labels[li])) for li in range(len(levels_tcool)) for ci in range(len(colors_tcool_labels))]
    legend = ax3.legend(handles=handles_Nbins + handles_subs, title=r'$\log_{10}\, N_{%s}\; [\mathrm{cm}^{-2}]$'%(ionlabel),\
              fontsize=fontsize, ncol=ncols_legend, loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    legend.get_title().set_fontsize(fontsize)
    ax3.axis('off')	
                
    plt.savefig(outputdir + 'phase_diagrams_by_N%s_EA-L0100N1504_28_PtAb_32000pix_6.25slice_z-cen-all_number%i.eps'%(ion, number) ,format='eps', bbox_inches='tight')
    

def plot_iondiffs(kind, number=0, fontsize=fontsize):
    '''
    kind: 'T', 'nH', or 'fO'
    '''    
    # length should be sufficient for the number of bins
    colors_b = ['gray', 'darkgray', 'lightgray', 'limegreen', 'blue', 'magenta'] 
    
    ## some defaults
    fraclevels = [0.90] 
    linestyles = ['solid']
    pixdens = True
    vdynrange = 7.
    xlim = None
    ylim = None
    labelyax2 = False
    labelxax1 = True
    prunex1 = True
    pruney1 = False
    # rescalef: plotted quantity (axes 2,3) is rescaled in plots (for plotting in non-stored units, e.g. solar metallicity, n_H)
    # N_vs_qty: plot quantity against column density (otherwise: quantity against quantity)
    # isdiff: qty is a difference 
    
    # snap 28 O7 vs O8
    if kind == 'T':
        nametop    = 'temperature_by_o7_o8_snap28_in_No7bins'
        namebottom = 'temperature_by_o6_o7_snap28_in_No7bins'
        searchkey_top     = 'NO7'
        searchkey_bottom  = 'NO7'
        plotaxes_top = (1, 0)
        plotaxes_bot = (0, 1)
        subtitle_top = r'$N_{\mathrm{O\, VII}}$ contours'
        subtitle_bot = r'$N_{\mathrm{O\, VII}}$ contours'
        rescalef = 0.
        xlim = (4.5, 6.9)
        ylim = (4.5, 6.9)
        pruney1 = True
        xlabel_top    = r'$\log_{10}\, T \; [\mathrm{K}]$, $\mathrm{O\,VIII}$-weighted'
        xlabel_bottom = r'$\mathrm{O\,VI}$-weighted'
        ylabel        = r'$\log_{10}\, T \; [\mathrm{K}]$, $\mathrm{O\,VII}$-weighted'
        name = 'temperature_weighted_by_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS'
    elif kind == 'rho' or kind == 'nH':
        nametop    = 'density_by_o7_o8_snap28_in_No7bins'
        namebottom = 'density_by_o6_o7_snap28_in_No7bins'
        searchkey_top     = 'NO7'
        searchkey_bottom  = 'NO7'
        plotaxes_top = (1, 0)
        plotaxes_bot = (0, 1)
        subtitle_top = r'$N_{\mathrm{O\, VII}}$ contours'
        subtitle_bot = r'$N_{\mathrm{O\, VII}}$ contours'
        rescalef = np.log10(rho_to_nh)
        xlim = (-6.8, -1.5)
        ylim = (-6.8, -1.5)
        xlabel_top    = r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$, $\mathrm{O\,VIII}$-weighted'
        xlabel_bottom = r'$\mathrm{O\,VI}$-weighted'
        ylabel        = r'$\log_{10}\, n_H \; [\mathrm{cm}^{-3}]$, $\mathrm{O\,VII}$-weighted'
        name = 'density_weighted_by_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS'
    elif kind == 'fO':
        nametop    = 'oxygenmassfraction_by_o7_o8_snap28_in_No7bins'
        namebottom = 'oxygenmassfraction_by_o6_o7_snap28_in_No7bins'
        plotaxes_top = (1, 0)
        plotaxes_bot = (0, 1)
        subtitle_top = r'$N_{\mathrm{O\, VII}}$ contours'
        subtitle_bot = r'$N_{\mathrm{O\, VII}}$ contours'
        searchkey_top     = 'NO7'
        searchkey_bottom  = 'NO7'
        rescalef = -1. * np.log10(0.005862311051135351) # solar oxygen mass fraction
        xlim = (-1.5, 0.9)
        ylim = (-1.5, 0.9)
        xlabel_top    = r'$\log_{10}\, f_{O, \mathrm{mass}} \; [\odot]$, $\mathrm{O\,VIII}$-weighted'
        xlabel_bottom = r'$\mathrm{O\,VI}$-weighted'
        ylabel        = r'$\log_{10}\, f_{O, \mathrm{mass}} \; [\odot]$, $\mathrm{O\,VII}$-weighted'
        name = 'oxygenmassfraction_weighted_by_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS'
    else:
        print('%s is not an option for kind'%kind)
    
    with h5py.File(datadir + 'histograms.hdf5', 'r') as datafile:
        sep = r'\,\mathrm{\endash}\,'
        grp = datafile[nametop]
        hists_top, minvals_top, maxvals_top, minincl_top, maxincl_top = \
            getsortedsubhists(grp, searchkey_top)
        hist_tot_top = np.array(grp['hist_all'])
        edges_ax0_top = np.array(grp['edges_axis0'])
        edges_ax1_top = np.array(grp['edges_axis1'])
        labels_subs_top = [r'$%s %s %s$'%(fmtfp1(minvals_top[i]), sep, fmtfp1(maxvals_top[i])) if not minincl_top[i] and not maxincl_top[i] else\
                       r'$ < %s$'%(fmtfp1(maxvals_top[i])) if minincl_top[i] else\
                       r'$ > %s$'%(fmtfp1(minvals_top[i]))\
                       for i in range(len(hists_top))]
        
        grp = datafile[namebottom]
        hists_bot, minvals_bot, maxvals_bot, minincl_bot, maxincl_bot = \
            getsortedsubhists(grp, searchkey_bottom)
        hist_tot_bot = np.array(grp['hist_all'])
        edges_ax0_bot = np.array(grp['edges_axis0'])
        edges_ax1_bot = np.array(grp['edges_axis1'])
        labels_subs_bot = [r'$%s %s %s$'%(fmtfp1(minvals_bot[i]), sep, fmtfp1(maxvals_bot[i])) if not minincl_bot[i] and not maxincl_bot[i] else\
                       r'$ < %s$'%(fmtfp1(maxvals_bot[i])) if minincl_bot[i] else\
                       r'$ > %s$'%(fmtfp1(minvals_bot[i]))\
                       for i in range(len(hists_top))]
    cmap = gray_m
    sfi_a = '(a)'
    sfi_b = '(b)'
    textcolor = 'black'

    plt.figure(figsize=(figwidth, figwidth))
    grid = gsp.GridSpec(2, 1, height_ratios=[5., 1.], hspace=0.17, top=0.95, bottom=0.05, left=0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    grid1  = gsp.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0], width_ratios=[1., 1.], hspace=0.0, wspace=0.0)
    ax1 = plt.subplot(grid1[0], facecolor=cmap(0.)) 
    ax2 = plt.subplot(grid1[1], facecolor=cmap(0.)) 
    #cax = plt.subplot(grid1[:, 1])
    lax = plt.subplot(grid[1])    
    ncols_legend = 2
    legendloc = 'lower center'
    legend_bbox_to_anchor = (0.5, 0.0)

    # grayscale images: total sightline distribution
    #vmin1, vmax1 = getminmax2d({'bins': hist_tot_top, 'edges': [edges_ax0_top, edges_ax1_top]}, axis=None, log=True, pixdens=pixdens) # axis is summed over; 
    #vmin2, vmax2 = getminmax2d({'bins': hist_tot_bot, 'edges': [edges_ax0_bot, edges_ax1_bot]}, axis=None, log=True, pixdens=pixdens)
    #vmax = max(vmax1, vmax2)
    #vmin = max(vmax - vdynrange, min(vmin1, vmin2))
    #img, vmin, vmax = add_2dplot(ax1, hist_tot_top, [edges_ax0_top, edges_ax1_top], plotaxes=plotaxes_top,\
    #                             log=True, vmin=vmin, vmax=vmax, cmap=cmap, shiftx=rescalef, shifty=rescalef, pixdens=pixdens)
    #img, vmin, vmax = add_2dplot(ax2, hist_tot_bot, [edges_ax0_bot, edges_ax1_bot], plotaxes=plotaxes_bot,\
    #                             log=True, vmin=vmin, vmax=vmax, cmap=cmap, shiftx=rescalef, shifty=rescalef, pixdens=pixdens)

    # add colorbar
    #if pixdens:
    #    clabel = r'$\log_{10}\, \mathrm{sightline\, fraction} \, \mathrm{dex}^{-2}$'
    #else:
    #    clabel = r'$\log_{10}$ fraction of sightlines'
    #add_colorbar(cax, img=img, clabel=clabel, extend='min', fontsize=fontsize, orientation='vertical')
    #cax.set_aspect(15.)
    #cax.tick_params(labelsize=fontsize - 1, axis='both')
   
    # plot contour levels for column density subsets
    colors = [(color,)*len(fraclevels) for color in colors_b]
    for i in range(number, len(hists_top)):
        add_2dhist_contours(ax1, hists_top[i], [edges_ax0_top, edges_ax1_top], toplotaxes=plotaxes_top,\
                        fraclevels=True, levels=fraclevels, legendlabel=None,\
                        shiftx=rescalef, shifty=rescalef, colors=colors[i], linestyles=linestyles, linewidth=2)
    for i in range(number, len(hists_bot)):
        add_2dhist_contours(ax2, hists_bot[i], [edges_ax0_bot, edges_ax1_bot], toplotaxes=plotaxes_bot,\
                        fraclevels=True, levels=fraclevels, legendlabel=None,\
                        shiftx=rescalef, shifty=rescalef, colors=colors[i], linestyles=linestyles, linewidth=2)
    
    
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize, direction='in', right=True, top=True, labelbottom=labelxax1, axis='both', which='both', color=textcolor)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize, direction='in', right=True, top=True, labelleft=labelyax2, axis='both', which='both', color=textcolor)
    
    linestyle_diffs = ['solid'] # , 'dashed', 'dotted'
    color_diffs = 'black'   
    labels_diffs = ['equal'] # , r'$\pm 0.5$ dex', r'$\pm 1$ dex'
    #pm = [0.5,1.]
    
    # set x, y limits if not specified
    if xlim is None:
        xlim1 = ax1.get_xlim()
        xlim2 = ax2.get_xlim()
        xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
    if ylim is None:
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
            
    # plot equality lines
    ax1.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], color=color_diffs, linestyle=linestyle_diffs[0])
    #ax1.plot([xlim[0], xlim[1]],[xlim[0] + pm[0], xlim[1] + pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
    #ax1.plot([xlim[0], xlim[1]],[xlim[0] - pm[0], xlim[1] - pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
    #ax1.plot([xlim[0], xlim[1]],[xlim[0] + pm[1], xlim[1] + pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
    #ax1.plot([xlim[0], xlim[1]],[xlim[0] - pm[1], xlim[1] - pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
    
    ax2.plot([xlim[0], xlim[1]],[xlim[0], xlim[1]], color=color_diffs, linestyle=linestyle_diffs[0])
    #ax2.plot([xlim[0], xlim[1]],[xlim[0] + pm[0], xlim[1] + pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
    #ax2.plot([xlim[0], xlim[1]],[xlim[0] - pm[0], xlim[1] - pm[0]], color=color_diffs, linestyle=linestyle_diffs[1])
    #ax2.plot([xlim[0], xlim[1]],[xlim[0] + pm[1], xlim[1] + pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
    #ax2.plot([xlim[0], xlim[1]],[xlim[0] - pm[1], xlim[1] - pm[1]], color=color_diffs, linestyle=linestyle_diffs[2])
            
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_xlabel(xlabel_top, fontsize=fontsize)
    ax2.set_xlabel(xlabel_bottom, fontsize=fontsize)
             
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax2.set_xlim(*xlim)
    ax2.set_ylim(*ylim)
            
    # square plot; set up axis 1 frame
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()
    ax2.set_aspect((xlim2[1]-xlim2[0])/(ylim2[1]-ylim2[0]), adjustable='box-forced')
    
#    if kind == 'T':
#        majticks = [3., 4., 5., 6., 7.]
#        minticks = list(np.round(np.arange(3., 7., 0.2), 1))
#        for i in majticks[:-1]:
#            minticks.remove(i) 
#        ax1.set_yticks(majticks)
#        ax1.set_yticks(minticks, minor=True)
#        ax2.set_yticks(majticks)
#        ax2.set_yticks(minticks, minor=True)
#        ax1.set_xticks(majticks)
#        ax1.set_xticks(minticks, minor=True)
#        ax2.set_xticks(majticks)
#        ax2.set_xticks(minticks, minor=True)
    
    # remove rightmost tick label where it's a problem (prune by hand because MaxNLocator changes all the label locations)
    # important: after limit setting, before aspect ratio setting
    if prunex1:
        # only major ticks
        old_ticklocs = ax1.get_xticks() #array	
        old_ticklocs_min = ax1.get_xticks(minor=True) #array	
        ax1.set_xticks(old_ticklocs[:-1]) 
        ax1.set_xticks(old_ticklocs_min, minor=True)
        ax1.set_xlim(*xlim)
    if pruney1:
        # only major ticks
        old_ticklocs = ax1.get_yticks() #array	
        old_ticklocs_min = ax1.get_yticks(minor=True) #array	
        ax1.set_yticks(old_ticklocs[1:]) 
        ax1.set_yticks(old_ticklocs_min, minor=True)
        ax1.set_ylim(*ylim)
        
    # subfig indices
    #ax1.text(0.95, 0.05, sfi_a, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax1.transAxes, color=textcolor)
    #ax2.text(0.95, 0.05, sfi_b, fontsize=fontsize, horizontalalignment='right', verticalalignment='bottom', transform=ax2.transAxes, color=textcolor)
    # subfig titles
    subtitleloc = (0.05, 0.95)
    ax1.text(subtitleloc[0], subtitleloc[1], subtitle_top, fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, bbox=dict(facecolor='white',alpha=0.3), color=textcolor)
    ax2.text(subtitleloc[0], subtitleloc[1], subtitle_bot, fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes, bbox=dict(facecolor='white',alpha=0.3), color=textcolor)
    # set labels manually: if edges match for O7 and O8, show one legend, and just leave out the fractions (they're on other plots already)
    legends_match = np.all(np.array(labels_subs_top) == np.array(labels_subs_bot))
    if legends_match:
        handles_subs = [mlines.Line2D([], [], color=colors_b[i], linestyle='solid', label=labels_subs_top[i]) for i in range(len(labels_subs_top))]
    else:
        print('Warning: column density edges do not match for the different ions')
        handles_subs1 = [mlines.Line2D([], [], color=colors_b[i], linestyle='solid', label=labels_subs_top[i]) for i in range(len(labels_subs_top))]   
        handles_subs2 = [mlines.Line2D([], [], color=colors_b[i], linestyle='solid', label=labels_subs_bot[i]) for i in range(len(labels_subs_bot))]
        handles_subs = handles_subs1 + handles_subs2
        #labels_subs  = labels_subs1 + labels_subs2
    # set up legend in ax below main figure
    
    #level_legend_handles = [mlines.Line2D([], [], color='gray', linestyle = linestyles[i], label='%.1f%% encl.'%(100.*fraclevels[i])) for i in range(len(fraclevels))]    
    #diffs_legend_handles = [mlines.Line2D([], [], color=color_diffs, linestyle = linestyle_diffs[i], label=labels_diffs[i]) for i in range(len(linestyle_diffs))]
    lax.legend(handles=handles_subs, fontsize=fontsize, ncol=ncols_legend, loc=legendloc, bbox_to_anchor=legend_bbox_to_anchor, borderaxespad=0.)
    lax.axis('off')

    plt.savefig(outputdir + '%s_number%i.eps'%(name, number), format='eps', bbox_inches='tight')
    

# correlation o6, ne8, O6, Ne8, O VI, Ne VIII, ion FUV
def plot_ioncor_o78_27_1sl(ion='o6', number=0, fontsize=fontsize):
    xlim = (13.0, 17.5)
    ylim = (13.5, 17.2)
    if ion == 'o6':
        searchkey = 'NO6'
        ionlab = 'O\,VI'
    elif ion == 'ne8':
        searchkey = 'NNe8'
        ionlab = 'Ne\,VIII'
    
    with h5py.File(datadir + 'histograms.hdf5', 'r') as datafile:
        grpname = 'No7_No8_byN%s_in_N%sbins'%(ion, ion)  
        grp = datafile[grpname]
        hists, minvals, maxvals, minincl, maxincl = \
            getsortedsubhists(grp, searchkey)
        #hist_tot = np.array(grp['hist_all'])
        edges_o7 = np.array(grp['edges_axis0'])
        edges_o8 = np.array(grp['edges_axis1'])
        sep = r'\,\mathrm{\endash}\,'
        labels_subs = [r'$%s %s %s$'%(fmtfp1(minvals[i]), sep, fmtfp1(maxvals[i])) if not minincl[i] and not maxincl[i] else\
                       r'$ < %s$'%(fmtfp1(maxvals[i])) if minincl[i] else\
                       r'$ > %s$'%(fmtfp1(minvals[i]))\
                       for i in range(len(hists))]
        hist_all = np.array(grp['hist_all'])
        
    # set up grid
    fig = plt.figure(figsize=(figwidth, 4.5))
    grid = gsp.GridSpec(1, 2, width_ratios=[5.,1.], wspace=0.0)
    ax1 = plt.subplot(grid[0], facecolor='white') 
    ax3 = plt.subplot(grid[1])
    ncols_legend = 1
    legendloc = 'upper left'
    legend_bbox_to_anchor=(0., 1.)
    vmin = -8.
    vmax = None
    cmap = gray_m

    # set up x-y extents from data range (max  = 0.05 + max included in hist)
    #ax1.set_xlim(8.,16.25)
    #ax1.set_ylim(9.,17.25)
    ax1.set_xlabel(logNo7label, fontsize=fontsize)
    ax1.set_ylabel(logNo8label, fontsize=fontsize)
        
    # plot background and colorbar: total distribution
    #img, vmin, vmax = add_2dplot(ax1, hist_all, [edges_o7, edges_o8], plotaxes=(0,1),\
    #                             log=True, vmin=vmin, vmax=vmax, cmap=cmap, shiftx=0., shifty=0., pixdens=True)
    # add colorbar
    #add_colorbar(ax2, img=img, clabel=r'$\log_{10}\, \mathrm{sightline\, fraction} \, \mathrm{dex}^{-2}$', extend='min', fontsize=fontsize)
    #ax2.set_aspect(10.)
    #ax2.tick_params(labelsize=fontsize - 1., axis='both')
   
    # plot contour levels for column density subsets
    fraclevels = [0.90] 
    linestyles = ['solid']
    colors_Nbins = ['purple', 'red', 'gold', 'lime', 'blue', 'magenta']
    
    for i in range(number, len(hists)):
        add_2dhist_contours(ax1, hists[i], [edges_o7, edges_o8], toplotaxes=(0,1),\
                        fraclevels=True, levels=fraclevels, legendlabel=None,\
                        shiftx=0., shifty=0., colors=colors_Nbins[i], linestyles=linestyles, linewidth=2)

    # square plot; set up axis 1 frame
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1] - xlim1[0]) / (ylim1[1] - ylim1[0]), adjustable='box-forced')   
    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize-1, direction='in', right=True, top=True, axis='both', which='both', color='black')
    ax1.text(0.05, 0.95, r'$N_{%s}$ contours'%ionlab, fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes, color='black',  bbox=dict(facecolor='white',alpha=0.3))
    # set up legend in ax below main figure
    handles_Nbins = [mlines.Line2D([], [], color=colors_Nbins[i], linestyle='solid', label=labels_subs[i]) for i in range(len(labels_subs))]
    #level_legend_handles = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_Nbins, fontsize=fontsize, ncol=ncols_legend, loc=legendloc, bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    fig.tight_layout()
    plt.savefig(outputdir + 'coldens_o7-o8_by_%s_L0100N1504_27_PtAb_C2Sm_32000pix_totalbox_T4EOS_number%i.eps'%(ion, number), format='eps', bbox_inches='tight')
    

def plot_all_tngcomp():
    cddfplots_overview_tngcomp(1, fontsize=fontsize)
    cddfplots_overview_tngcomp(2, fontsize=fontsize)
    cddfplots_overview_tngcomp(3, fontsize=fontsize)

def plot_all_cddfsplit():
    plot_cddfs_o78_by_delta(0, fontsize=fontsize)
    plot_cddfs_o78_by_delta(1, fontsize=fontsize)
    plot_cddfs_o78_by_delta(2, fontsize=fontsize)
    plot_cddfs_o78_by_delta(3, fontsize=fontsize)
    plot_cddfs_o78_by_delta(4, fontsize=fontsize)
    plot_cddfs_o78_by_delta(5, fontsize=fontsize)
    plot_cddfs_o78_by_delta(6, fontsize=fontsize)

def plot_all_iondiffs():
    plot_iondiffs('T', number=0, fontsize=15)
    plot_iondiffs('T', number=5, fontsize=15)
    plot_iondiffs('nH', number=0, fontsize=15)
    plot_iondiffs('fO', number=0, fontsize=15)    
    
def plot_all_phasediagrams():
    plot_phasediagrams_by_ion(ion='o7', fontsize=15, number=0)
    plot_phasediagrams_by_ion(ion='o7', fontsize=15, number=5)
    plot_phasediagrams_by_ion(ion='o8', fontsize=15, number=0)
    
def plot_all_ioncorr():
    plot_ioncor_o78_27_1sl(ion='o6', number=0, fontsize=15)
    plot_ioncor_o78_27_1sl(ion='o6', number=5, fontsize=15)
    plot_ioncor_o78_27_1sl(ion='ne8', number=0, fontsize=15)
    
def plot_all():
    plot_all_tngcomp()
    plot_all_cddfsplit()
    plot_all_iondiffs()
    plot_all_phasediagrams()
    plot_all_ioncorr()
    cddfplots_agneffect(fontsize=fontsize, fixz=False)
    cddfplots_agneffect(fontsize=fontsize, fixz=True)
    plot_curveofgrowth_o7_o8(fontsize=fontsize, alpha=0.05)
    plotEWdists_o78_litcomp(fontsize=fontsize)
   
    