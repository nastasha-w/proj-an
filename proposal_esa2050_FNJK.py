#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:25:57 2019

@author: wijers
"""

import numpy as np
import scipy.integrate as si
import h5py
import sys
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.gridspec as gsp

import makecddfs as mc # don't use getcosmopars! This requires simfile and the whole readEagle mess
import eagle_constants_and_units as c
import cosmo_utils as csu

import plotspecwizard as ps

datadir = None
outputdir = '/home/wijers/Documents/papers/esa2050_jelle_and_fabrizio/'

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
logNne8label = r'$\log_{10}\, N_{\mathrm{Ne\,VIII}} \; [\mathrm{cm}^{-2}]$'
logNhneutrallabel = r'$\log_{10}\, N_{\mathrm{H\, I} + \mathrm{H}_2} \; [\mathrm{cm}^{-2}]$'

cddflabel = r'$\log_{10}\, \partial^2 n/ \partial N \partial X  $'
cddfrellabel_z = r'$\log_{10}\, f(N, z)\,/\,f(N, %s)$'
cddfrellabel = r'$\log_{10}\, f(N)\,/\,f(N, %s)$'

logEWlabel = r'$\log_{10}\, EW\; [\mathrm{m\AA}]$, rest'
logEWo7label = r'$\log_{10}\, EW_{O\,VII}\; [\mathrm{m\AA}]$, rest'
logEWo8label = r'$\log_{10}\, EW_{O\,VIII}\; [\mathrm{m\AA}]$, rest'

nHconvlabel = r'$\log_{10}\, n_{\mathrm{H}} \; [\mathrm{cm}^{-3}], f_H = 0.752$'
logTlabel = r'$\log_{10}\, T \; [K]$'
logdeltalabel = r'$\log_{10}\, (1 + \delta)$'

Nrange = (12.5, 16.5)
fontsize = 12.
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

def simple_colormap(color, alpha=1.):
    rgba = mpl.colors.to_rgba(color, alpha=alpha)
    cvals = np.array([list(rgba[:3]) + [0.], rgba])
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'simple_%s'%(color), cvals)
    return new_cmap

bone_m = truncate_colormap(plt.get_cmap('bone_r'), maxval=normmax_bone)
gray_m = truncate_colormap(plt.get_cmap('gist_gray_r'), maxval=normmax_gray)
gray_light = truncate_colormap(plt.get_cmap('gist_gray_r'), maxval=0.3)

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
    if not hasattr(Nion, '__len__'):
            Nion = np.array([Nion])
            
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

def linterpsolve(xvals, yvals, xpoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below xpoint 
    xvals, yvals: 1D arrays
    xpoint: float
    '''
    if np.all(np.diff(xvals) > 0.):
        incr = True
    elif np.all(np.diff(xvals) < 0.):
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

def find_intercepts(yvals, xvals, ypoint):
    '''
    'solves' a monotonic function described by xvals and yvals by linearly 
    interpolating between the points above and below ypoint 
    xvals, yvals: 1D arrays
    ypoint: float
    Does not distinguish between intersections separated by less than 2 xvals points
    '''
    if not (np.all(np.diff(xvals) < 0.) or np.all(np.diff(xvals) > 0.)):
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
                     leqzero[segmentedges[si - 1] : segmentedges[si]] if si < len(segmentedges) else\
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


def percentiles_from_histogram(histogram, edgesaxis, axis=-1, percentiles=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
    '''
    get percentiles from the histogram along axis
    edgesaxis are the bin edges along that same axis
    histograms can be weighted by something: this function just solves 
    cumulative distribution == percentiles
    '''
    cdists = np.cumsum(histogram, axis=axis, dtype=np.float) 
    sel = list((slice(None, None, None),)*len(histogram.shape))
    sel2 = np.copy(sel)
    sel[axis] = -1
    sel2[axis] = np.newaxis
    cdists /= (cdists[tuple(sel)])[tuple(sel2)] # normalised cumulative dist: divide by total along axis
    # bin-edge corrspondence: at edge 0, cumulative value is zero
    # histogram values are counts in cells -> hist bin 0 is what is accumulated between edges 0 and 1
    # cumulative sum: counts in cells up to and including the current one: 
    # if percentile matches cumsum in cell, the percentile value is it's rigtht edges -> edge[cell index + 1]
    # effectively, if the cumsum is prepended by zeros, we get a hist bin matches edge bin matching

    oldshape1 = list(histogram.shape)[:axis] 
    oldshape2 = list(histogram.shape)[axis+1:]
    newlen1 = int(np.prod(oldshape1))
    newlen2 = int(np.prod(oldshape2))
    axlen = histogram.shape[axis]
    cdists = cdists.reshape((newlen1, axlen, newlen2))
    cdists = np.append(np.zeros((newlen1, 1, newlen2)), cdists, axis=1)
    cdists[:, -1, :] = 1. # should already be true, but avoids fp error issues

    leftarr  = cdists[np.newaxis, :, :, :] <= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    rightarr = cdists[np.newaxis, :, :, :] >= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    
    leftbininds = np.array([[[ np.max(np.where(leftarr[pind, ind1, :, ind2])[0]) \
                               for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
    # print leftarr.shape
    # print rightarr.shape
    rightbininds = np.array([[[np.min(np.where(rightarr[pind, ind1, :, ind2])[0]) \
                               for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
    # if left and right bins are the same, effictively just choose one
    # if left and right bins are separated by more than one (plateau edge), 
    #    this will give the middle of the plateau
    lweights = np.array([[[ (cdists[ind1, rightbininds[pind, ind1, ind2], ind2] - percentiles[pind]) \
                            / ( cdists[ind1, rightbininds[pind, ind1, ind2], ind2] - cdists[ind1, leftbininds[pind, ind1, ind2], ind2]) \
                            if rightbininds[pind, ind1, ind2] != leftbininds[pind, ind1, ind2] \
                            else 1.
                           for ind2 in range(newlen2)] for ind1 in range(newlen1)] for pind in range(len(percentiles))])
                
    outperc = lweights * edgesaxis[leftbininds] + (1. - lweights) * edgesaxis[rightbininds]
    outperc = outperc.reshape((len(percentiles),) + tuple(oldshape1 + oldshape2))
    return outperc

def diffpercentiles(dist, ax0, ax1, edges0, edges1, percentiles=[0.10, 0.50, 0.90]): 

    # construct diff binning from bin centers
    ax0cens = edges0[:-1] + 0.5 * np.diff(edges0)
    ax1cens = edges1[:-1] + 0.5 * np.diff(edges1)
    diffvals = ax0cens[:, np.newaxis] - ax1cens[np.newaxis, :]
    if ax0 > ax1:
        diffvals = diffvals.T
    diffvals_forbins = list(set(np.round(diffvals.flatten(), 1)))
    diffvals_forbins.sort()
    if np.min(diffvals) < np.min(diffvals_forbins):
        diffvals_forbins = [np.round(np.min(diffvals) - 0.1, 1)] + diffvals_forbins
    if np.max(diffvals) > np.max(diffvals_forbins):
        diffvals_forbins =  diffvals_forbins + [np.round(np.max(diffvals) + 0.1, 1)]
    diffbins =  [1.5 * diffvals_forbins[0] - 0.5 * diffvals_forbins[1]] +\
                list(diffvals_forbins[1:] - 0.5 * np.diff(diffvals_forbins)) + \
                [1.5 * diffvals_forbins[-1] - 0.5 * diffvals_forbins[-2]]
    diffbins = np.array(diffbins)
    #print np.diff(oxovernevals_forbins), np.diff(oxovernebins) 
    diffbinmatch = np.digitize(diffvals, diffbins) - 1 # searchsorted-type results: index 1 means between edges 0 and 1 -> bin 0
    # bin ox, ne 2d slices into 1D ratio bins
    udists = [ np.sum(dist[np.where(diffbinmatch == bi)])\
             for bi in range(len(diffbins) - 1)]
    udists = np.array(udists)
    edges = diffbins
    
    return percentiles_from_histogram(udists, edges, axis=0, percentiles=np.array(percentiles))

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
    return cbar

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
        ax.set_ylabel(ylabel, fontsize=fontsize)
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

def selectsightlines_byhalomass(specout, halocat, Mhmin=10**12, ions=['o7', 'o8'], Nvals=[14.5, 15., 15.5, 16., 16.5], Ntol=0.05, nrandom=5):
    '''
    Mhmin: Msun units
    halocat: includes full path
    '''
    specout.getcoldens(dions=ions)
    Xsl = specout.positions[:, 0] / specout.cosmopars['h'] # cMpc
    Ysl = specout.positions[:, 1] / specout.cosmopars['h']
    ionsel = np.array([np.min(np.abs(specout.coldens[ion][:, np.newaxis] - np.array(Nvals)[np.newaxis, :]), axis=1) <= Ntol for ion in ions])
    ionsel = np.any(ionsel, axis=0)

    specnums_Nsel1 = np.where(ionsel)[0]
    Xsl = Xsl[ionsel]
    Ysl = Ysl[ionsel]

    with h5py.File(halocat, 'r') as cat:
             
        cosmopars = {key: item for (key, item) in cat['Header/cosmopars'].attrs.items()}
        Xhalos = np.array(cat['Xcom_cMpc'])
        Yhalos = np.array(cat['Ycom_cMpc'])
        boxsize = cosmopars['boxsize'] / cosmopars['h']
        Mhalos = np.array(cat['M200c_Msun'])
        Rhalos = np.array(cat['R200c_pkpc']) / cosmopars['a'] * 1e-3
    
    halosel_mass = Mhalos >= Mhmin
    Xhalos = Xhalos[halosel_mass]
    Yhalos = Yhalos[halosel_mass]
    Rhalos = Rhalos[halosel_mass]
    
    xoff = np.min(np.array([np.abs(Xhalos[np.newaxis, :] - Xsl[:, np.newaxis]),\
                            np.abs(Xhalos[np.newaxis, :] + boxsize - Xsl[:, np.newaxis]),\
                            np.abs(Xhalos[np.newaxis, :] - boxsize - Xsl[:, np.newaxis])]), axis=0)
    yoff = np.min(np.array([np.abs(Yhalos[np.newaxis, :] - Ysl[:, np.newaxis]),\
                            np.abs(Yhalos[np.newaxis, :] + boxsize - Ysl[:, np.newaxis]),\
                            np.abs(Yhalos[np.newaxis, :] - boxsize - Ysl[:, np.newaxis])]), axis=0)
    
    r2halos = xoff**2 + yoff**2
    del xoff
    del yoff
    
    halos_1r = np.any(r2halos <= (Rhalos**2)[np.newaxis, :], axis=1)
    specnums_rsel = specnums_Nsel1[halos_1r]    
    
    seldct = {}
    for ion in ions:
        seldct[ion] = {}
        Nvals_sl = specout.coldens[ion][specnums_rsel]
        for Nval in Nvals:
            allN = np.where(np.abs(Nvals_sl - Nval) <= Ntol)[0]
            if len(allN) <= nrandom:
                sls = specnums_rsel[allN]
            else:
                sls = np.random.choice(specnums_rsel[allN], nrandom)
            seldct[ion][Nval] = sls
            print('Ion %s, \t N = %s, \t: %s'%(ion, Nval, sls))
    
    return seldct

def gethalodata_sightlines(specfile, sightlines, halocat, Mhmin=10**9, R200cmax=2., Rpkpcmax=200.):
    '''
    Mhmin: Msun units
    halocat: includes full path
    '''
    
    specsample = ps.SpecSample(specfile)
    # X, Y of sightlines in cMpc units    
    Xsl = specsample.fracpos[0][np.array(sightlines)] * specsample.cosmopars['boxsize'] / specsample.cosmopars['h']
    Ysl = specsample.fracpos[1][np.array(sightlines)] * specsample.cosmopars['boxsize'] / specsample.cosmopars['h']

    with h5py.File(halocat, 'r') as cat:
             
        cosmopars = {key: item for (key, item) in cat['Header/cosmopars'].attrs.items()}
        Xhalos = np.array(cat['Xcom_cMpc'])
        Yhalos = np.array(cat['Ycom_cMpc'])
        Zhalos = np.array(cat['Zcom_cMpc'])
        Vhalos = np.array(cat['VZpec_kmps'])
        Mhalos = np.array(cat['M200c_Msun'])
        Rhalos = np.array(cat['R200c_pkpc']) / cosmopars['a'] * 1e-3
        galaxyid = np.array(cat['galaxyid'])
        
    hf = csu.Hubble(cosmopars['z'], cosmopars=cosmopars)
    boxsize = cosmopars['boxsize'] / cosmopars['h']        
    boxsize_v = boxsize * c.cm_per_mpc * cosmopars['a'] * hf * 1e-5 # cMpc -> pcm -> cm/s -> km/s   
    
    halosel_mass = Mhalos >= Mhmin

    Xhalos = Xhalos[halosel_mass]
    Yhalos = Yhalos[halosel_mass]
    Rhalos = Rhalos[halosel_mass]
    Zhalos = Zhalos[halosel_mass] 
    Vhalos = Vhalos[halosel_mass]
    Mhalos = Mhalos[halosel_mass] 
    galaxyid = galaxyid[halosel_mass]

    
    Vlos   = Vhalos + Zhalos * (c.cm_per_mpc * cosmopars['a'] * hf * 1e-5)
    Vlos %= boxsize_v
    
    Rhalos_forsel = np.max(np.array([Rhalos, Rpkpcmax / cosmopars['a'] * 1e-3 * np.ones(len(Rhalos))]), axis=0)
    
    xoff = np.min(np.array([np.abs(Xhalos[np.newaxis, :] - Xsl[:, np.newaxis]),\
                            np.abs(Xhalos[np.newaxis, :] + boxsize - Xsl[:, np.newaxis]),\
                            np.abs(Xhalos[np.newaxis, :] - boxsize - Xsl[:, np.newaxis])]), axis=0)
    yoff = np.min(np.array([np.abs(Yhalos[np.newaxis, :] - Ysl[:, np.newaxis]),\
                            np.abs(Yhalos[np.newaxis, :] + boxsize - Ysl[:, np.newaxis]),\
                            np.abs(Yhalos[np.newaxis, :] - boxsize - Ysl[:, np.newaxis])]), axis=0)
    
    r2halos = xoff**2 + yoff**2
    del xoff
    del yoff
    
    halos_1r = r2halos <= (Rhalos_forsel**2)[np.newaxis, :]
    
    basename = 'halos_%i.txt'
    for sli in range(len(sightlines)):
        sl = sightlines[sli]
    
        outname = outputdir + basename%(sl)
        fo = open(outname, 'w')
        fo.write('cosmopars:\n')
        for key in cosmopars.keys():
            fo.write('%s:\t%s\n'%(key, cosmopars[key]))
        fo.write('halos: M200c >= %s Msun, impact parameter < %s R200c or %s pkpc\n'%(Mhmin, R200cmax, Rpkpcmax))
        fo.write('galaxyid: EAGLE database identifieer\n')
        fo.write('X, Y, Z, Vpec, Vlos: apply to the halo central galaxy (mass-weighted quantities)\n')
        properties = ['galaxyid', 'M200c [Msun]', 'X [cMpc]', 'Y [cMpc]', 'Z [cMpc]', 'R200c [cMpc]', 'Vpec_Z [km/s, rest]', 'Vlos [km/s, rest]', 'Impact parameter [cMpc]', 'Impact parameter [R200c]']
        basestr = '\t'.join(['%s ']*len(properties)) + '\n'
        fo.write(basestr%tuple(properties))
        
        halosel = halos_1r[sli]
        galaxyid_sub = galaxyid[halosel]
        Mhalos_sub = Mhalos[halosel]
        Xhalos_sub = Xhalos[halosel]
        Yhalos_sub = Yhalos[halosel]
        Zhalos_sub = Zhalos[halosel]
        Rhalos_sub = Rhalos[halosel]
        Vhalos_sub = Vhalos[halosel]
        Vlos_sub   = Vlos[halosel]
        bhalos_sub = np.sqrt(r2halos[sli, halosel])
        
        for hi in range(np.sum(halosel)):
            fo.write(basestr%(galaxyid_sub[hi], Mhalos_sub[hi], Xhalos_sub[hi], Yhalos_sub[hi], Zhalos_sub[hi], Rhalos_sub[hi], Vhalos_sub[hi], Vlos_sub[hi], bhalos_sub[hi], bhalos_sub[hi] / Rhalos_sub[hi]))
        fo.close()
    return None

def getsubEW_sightlines(specfile, sightlines, vranges=None):
    '''
    Mhmin: Msun units
    halocat: includes full path
    '''
    
    if vranges is None:
        # extracted by eye for ~non-blended components
        vranges = {1495:  [(1400., 1700.), (2200., 2800.), (5200., 5600.), (5600., 300. )],\
                   2146:  [(900.,  1400.), (1400., 1900.), (2200., 3000.), (5600., 6000.)],\
                   10283: [(1600., 2100.), (2300., 3100.), (6200., 6400.)],\
                   2778:  [(0.,    1000.), (2000., 2500.), (2500., 3200.)],\
                   14530: [(2500., 3100.), (4100., 4700.)],\
                   3527:  [(200.,  1200.), (3800., 4400.)],\
                   12188: [(5400., 6400.)],\
                   12118: [(1300., 1700.), (2000., 2500.), (2600., 3100.), (3100., 3800.)],\
                   11131: [(1700., 2300.), (4200., 4800.)],\
                   13217: [(2300., 3300.)],\
                   15240: [(100.,  1100.), (1100., 1600.), (1700., 2300.)],\
                  }
        
    specsample = ps.SpecSample(specfile, specnums=sightlines)
    # X, Y of sightlines in cMpc units    
    
    basename = 'EWsubs_%i.txt'
    for sli in range(len(sightlines)):
        sl = sightlines[sli]
        
        slo = specsample.sldict[sl]
        slo.getspectrum('o7','Flux')
        slo.getspectrum('o8','Flux', corr=True)
    
        vrangels = vranges[sl]
        if None in vrangels:
            vrangels.remove(None)
        vrangedct = {'%s-%s km/s, rest'%(vr[0], vr[1]): vr for vr in vrangels}
        vrangedct.update({'total': None})

        
        EWs_o7 = {key: slo.getEW('o7', corr=False, vrange=vrangedct[key], vvals=specsample.velocity) for key in vrangedct.keys()}
        EWs_o8 = {key: slo.getEW('o8', corr=False, vrange=vrangedct[key], vvals=specsample.velocity) for key in vrangedct.keys()}
    
        outname = outputdir + basename%(sl)
        fo = open(outname, 'w')
        fo.write('EWs in different rest-frame line of sight velocity ranges (ranges selected by eye from spectrum plots)\n')
        fo.write('Vmin [km/s, rest-frame]\t Vmax[km/s, rest-frame]\t O VII EW [rest-frame mA]\t O VIII doublet EW [rest-frame mA]\n')
        keys = vrangedct.keys()
        keys.sort()
        for key in keys:
            if key == 'total':
                fo.write('%s\t %s\t %s\t %s\t total\n'%(0., 2. * specsample.velocity[-1] - specsample.velocity[-2], EWs_o7[key], EWs_o8[key]))
            else:
                fo.write('%s\t %s\t %s\t %s\n'%(vrangedct[key][0], vrangedct[key][1], EWs_o7[key], EWs_o8[key]))
        fo.close()
    return None


def plot_mocks(fontsize=fontsize):
    '''
    
    mock spectrum 1   spectrum 1
    Athena mock 1     column densities
    prop. mock 1      and equivalent widths
    
    same for other spectra
    
    legend
    '''
    sightlines = [15240, 13217, 11131, 12118, 3527] #[14530, 2778, 10283, 2146, 1495]
    outname = 'mockspectra_inclhalosel.pdf' #'mockspectra.pdf'
    vranges_sl = {1495:  (1000., 2000.),\
                  2146:  (900.,  1900.),\
                  10283: (1400., 2400.),\
                  2778:  (0.,    1000.),\
                  14530: (4000., 5000.),\
                  3527:  (200.,  1200.),\
                  12188: (5400., 6400.),\
                  12118: (2900., 3900.),\
                  11131: (4200., 5200.),\
                  13217: (2300., 3300.),\
                  15240: (100.,  1100.),\
                  }
    numspecs = len(sightlines)
    filename = '/home/wijers/Documents/papers/esa2050_jelle_and_fabrizio/%s.hdf5'%'mocks_Aeff_res_o7-o8'
    halocat = '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5' # only care about central here
    
    plt.figure(figsize=(2*figwidth, 2. + numspecs * 2.))
    grid = gsp.GridSpec(numspecs + 1, 1, height_ratios=[1.] * numspecs +  [0.6], hspace=0.3, top = 0.95, bottom = 0.05, left= 0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    grids  = [gsp.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid[i], height_ratios=[1.] * 2, width_ratios=[3., 3., 1.], hspace=0.0, wspace=0.2) for i in range(numspecs)]
    lax = plt.subplot(grid[numspecs])
    laxs = [plt.subplot(grids[i][:, 2]) for i in range(numspecs)]
    axess = [np.array([[plt.subplot(grids[i][gi, yi]) for gi in range(2)] for yi in range(2)] + [laxs[i]]) for i in range(numspecs)]

    
    colors = {'o7': 'red', 'o8': 'blue'}
    linestyles = {'singlet': 'solid', 'doublet': 'dashed'}
    vlabel = r'$v\; [\mathrm{km}/\mathrm{s}]$, rest-frame'
    ylabel = r'Flux [normalized]'
    ionlabels={'o7':   r'O VII $21.60\,\mathrm{\AA}$',\
               'o8s':  r'O VIII  $18.967\,\mathrm{\AA}$',\
               'o8d':  r'O VIII  $18.967, 18.973\,\mathrm{\AA}$',\
               }
    #alpha_err = 0.1
    with h5py.File(filename, 'r') as datafile,\
         h5py.File(halocat, 'r') as cat:
             
        datagrp = datafile['mockspectra_sample3'] 

        cosmopars = {key: item for (key, item) in cat['Header/cosmopars'].attrs.items()}
        Xhalos = np.array(cat['Xcom_cMpc'])
        Yhalos = np.array(cat['Ycom_cMpc'])
        vhalos = np.array(cat['Zcom_cMpc']) * cosmopars['a'] * c.cm_per_mpc * csu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1e-5 # cMpc -> pMpc -> cm -> cm/s -> km/s
        vhalos += np.array(cat['VZpec_kmps'])
        boxsize_v = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * c.cm_per_mpc * csu.Hubble(cosmopars['z'], cosmopars=cosmopars) * 1e-5
        vhalos %= boxsize_v
        boxsize = cosmopars['boxsize'] / cosmopars['h']
        Mhalos = np.array(cat['M200c_Msun'])
        Rhalos = np.array(cat['R200c_pkpc']) / cosmopars['a'] * 1e-3

        halosel_mass = Mhalos >= 10**11.5
        
        for axes, sightline in zip(axess, sightlines):
            grp = datagrp['Spectrum%i'%sightline]
            
            # on-the-fly halo cross-ref...
            Xsl, Ysl = tuple(np.array(grp.attrs['position_cMpc']))
            
            xoff = np.min(np.array([np.abs(Xhalos - Xsl), np.abs(Xhalos + boxsize - Xsl), np.abs(Xhalos - boxsize - Xsl)]), axis=0)
            yoff = np.min(np.array([np.abs(Yhalos - Ysl), np.abs(Yhalos + boxsize - Ysl), np.abs(Yhalos - boxsize - Ysl)]), axis=0)
            
            #print('xoff, yoff stats check:')
            #print('xoff: min %s, max %s, percentiles 20, 50, 80 %s'%(np.min(xoff), np.max(xoff), np.percentile(xoff, [20., 50., 80.])))
            #print('yoff: min %s, max %s, percentiles 20, 50, 80 %s'%(np.min(yoff), np.max(yoff), np.percentile(yoff, [20., 50., 80.])))
            r2halos = xoff**2 + yoff**2
            del xoff
            del yoff
            halos_1r = r2halos <= Rhalos**2
            halos_2r = r2halos <= (2 * Rhalos)**2
            halos_rp = r2halos <= (0.2 * cosmopars['a']) **2
            del r2halos
            
            vhalos_1r = vhalos[np.logical_and(halosel_mass, halos_1r)]
            vhalos_2r = vhalos[np.logical_and(halosel_mass, halos_2r)]
            vhalos_rp = vhalos[np.logical_and(halosel_mass, halos_rp)]
            del halos_1r
            del halos_2r
            del halos_rp
            
            # ideal spectrum
            ax = axes[0][0]
            ax.minorticks_on()
            ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, labeltop=False, color='black')
            ax.text(0.01, 0.10, r'ideal spectrum', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
        
            flux_o7  = np.array(grp['flux_o7'])
            #flux_o8s = np.array(grp['flux_o8oneline'])
            flux_o8d = np.array(grp['flux_o8doublet'])
            vrest    = np.array(grp['vrest_kmps'])
            #cpos     = np.array(grp['lospos_cMpc'])
            vplot_o7,  flux_o7  = wrapspectrum(vrest, flux_o7)
            #vplot_o8s, flux_o8s = wrapspectrum(vrest, flux_o8s)
            vplot_o8d, flux_o8d = wrapspectrum(vrest, flux_o8d)
            ax.axhline(1., color='gray', linewidth=1, linestyle='solid', zorder=-2)
            ax.plot(vplot_o7, flux_o7, color=colors['o7'], linestyle=linestyles['singlet'], label=ionlabels['o7'])
            #ax.plot(vplot_o7, flux_o8s, color=colors['o8'], linestyle=linestyles['singlet'], label=ionlabels['o8s'])
            ax.plot(vplot_o7, flux_o8d, color=colors['o8'], linestyle=linestyles['doublet'], label=ionlabels['o8d'])
            
            for vh in vhalos_2r:
                ax.axvline(vh, color='gray', linestyle='dotted')
            for vh in vhalos_1r:
                ax.axvline(vh, color='black', linestyle='dotted')
            for vh in vhalos_rp:
                ax.axvline(vh, color='brown', linestyle='dotted')
                
            xlim = (vrest[0], vrest[-1] + np.average(np.diff(vrest)))
            ax.set_xlim(xlim)
            
            # Athena mock
            grp_ath = grp['Athena_mocks']
            ax = axes[0][1]
            ax.minorticks_on()
            ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=True, labelleft=True, labelright=False, labeltop=False, color='black')
            ax.text(0.01, 0.10, r'Athena X-IFU mock', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_xlabel(vlabel, fontsize=fontsize)
            
            photonnorm_o7 = grp_ath['o7'].attrs['unabsorbed_photons_per_pix']
            flux_o7       = np.array(grp_ath['o7/photoncounts_poisson'])
            vrest_o7      = np.array(grp_ath['o7/pixcenters_vrest_kmps'])
            #noiseband_o7  = np.array(grp_ath['o7/normednoise_lohi'])
            photonnorm_o8d = grp_ath['o8doublet'].attrs['unabsorbed_photons_per_pix']
            flux_o8d       = np.array(grp_ath['o8doublet/photoncounts_poisson'])
            vrest_o8d      = np.array(grp_ath['o8doublet/pixcenters_vrest_kmps'])
            #noiseband_o8d  = np.array(grp_ath['o8doublet/normednoise_lohi'])
            #cpos     = np.array(grp['lospos_cMpc'])
            vplot_o7,  flux_o7  = wrapspectrum(vrest_o7, flux_o7/float(photonnorm_o7))
            #vband_o7,  noiseband_o7  = wrapspectrum(vrest_o7, noiseband_o7)
            vplot_o8d, flux_o8d = wrapspectrum(vrest_o8d, flux_o8d/float(photonnorm_o8d))
            #vband_o8d, noiseband_o8d = wrapspectrum(vrest_o8d, noiseband_o8d)
            ax.axhline(1., color='gray', linewidth=1, linestyle='solid', zorder=-2)
            ax.step(vplot_o7, flux_o7, color=colors['o7'], linestyle=linestyles['singlet'], label=ionlabels['o7'], where='post')
            #ax.fill_between(vband_o7, noiseband_o7[0], noiseband_o7[1], step='post', color=colors['o7'], alpha=alpha_err)
            ax.step(vplot_o8d, flux_o8d, color=colors['o8'], linestyle=linestyles['doublet'], label=ionlabels['o8d'], where='post')
            #ax.fill_between(vband_o8d, noiseband_o8d[0], noiseband_o8d[1], step='post', color=colors['o8'], alpha=alpha_err)
            
            for vh in vhalos_2r:
                ax.axvline(vh, color='gray', linestyle='dotted')
            for vh in vhalos_1r:
                ax.axvline(vh, color='black', linestyle='dotted')
            for vh in vhalos_rp:
                ax.axvline(vh, color='brown', linestyle='dotted')
                
            ax.set_xlim(xlim)
            yl0, yl1 = ax.get_ylim()
            yl1 = max(yl1, 1.045)
            ax.set_ylim((yl0, yl1))
            
            # ideal spectrum
            ax = axes[1][0]
            ax.minorticks_on()
            ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=False, labelleft=True, labelright=False, labeltop=False, color='black')
            ax.text(0.01, 0.10, r'ideal spectrum', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
        
            flux_o7  = np.array(grp['flux_o7'])
            #flux_o8s = np.array(grp['flux_o8oneline'])
            flux_o8d = np.array(grp['flux_o8doublet'])
            vrest    = np.array(grp['vrest_kmps'])
            #cpos     = np.array(grp['lospos_cMpc'])
            vplot_o7,  flux_o7  = wrapspectrum(vrest, flux_o7)
            #vplot_o8s, flux_o8s = wrapspectrum(vrest, flux_o8s)
            vplot_o8d, flux_o8d = wrapspectrum(vrest, flux_o8d)
            ax.axhline(1., color='gray', linewidth=1, linestyle='solid', zorder=-2)
            ax.plot(vplot_o7, flux_o7, color=colors['o7'], linestyle=linestyles['singlet'], label=ionlabels['o7'])
            #ax.plot(vplot_o7, flux_o8s, color=colors['o8'], linestyle=linestyles['singlet'], label=ionlabels['o8s'])
            ax.plot(vplot_o7, flux_o8d, color=colors['o8'], linestyle=linestyles['doublet'], label=ionlabels['o8d'])
            
            for vh in vhalos_2r:
                ax.axvline(vh, color='gray', linestyle='dotted')
            for vh in vhalos_1r:
                ax.axvline(vh, color='black', linestyle='dotted')
            for vh in vhalos_rp:
                ax.axvline(vh, color='brown', linestyle='dotted')  
            ax.set_xlim(vranges_sl[sightline])
            
            # Hi-res mock
            grp_arc = grp['prop_mocks']
            ax = axes[1][1]
            ax.minorticks_on()
            ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=True, labelleft=True, labelright=False, labeltop=False, color='black')
            ax.text(0.01, 0.10, r'HiReX mock', fontsize=fontsize, transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
            ax.set_xlabel(vlabel, fontsize=fontsize)
            
            photonnorm_o7 = grp_arc['o7'].attrs['unabsorbed_photons_per_pix']
            flux_o7       = np.array(grp_arc['o7/photoncounts_poisson'])
            vrest_o7      = np.array(grp_arc['o7/pixcenters_vrest_kmps'])
            #noiseband_o7  = np.array(grp_arc['o7/normednoise_lohi'])
            photonnorm_o8d = grp_arc['o8doublet'].attrs['unabsorbed_photons_per_pix']
            flux_o8d       = np.array(grp_arc['o8doublet/photoncounts_poisson'])
            vrest_o8d      = np.array(grp_arc['o8doublet/pixcenters_vrest_kmps'])
            #noiseband_o8d  = np.array(grp_arc['o8doublet/normednoise_lohi'])
            #cpos     = np.array(grp['lospos_cMpc'])
            vplot_o7,  flux_o7  = wrapspectrum(vrest_o7, flux_o7/float(photonnorm_o7))
            #vband_o7,  noiseband_o7  = wrapspectrum(vrest_o7, noiseband_o7)
            vplot_o8d, flux_o8d = wrapspectrum(vrest_o8d, flux_o8d/float(photonnorm_o8d))
            #vband_o8d, noiseband_o8d = wrapspectrum(vrest_o8d, noiseband_o8d)
            ax.axhline(1., color='gray', linewidth=1, linestyle='solid', zorder=-2)
            ax.step(vplot_o7, flux_o7, color=colors['o7'], linestyle=linestyles['singlet'], label=ionlabels['o7'], where='post')
            #ax.fill_between(vband_o7, noiseband_o7[0], noiseband_o7[1], step='post', color=colors['o7'], alpha=alpha_err)
            ax.step(vplot_o8d, flux_o8d, color=colors['o8'], linestyle=linestyles['doublet'], label=ionlabels['o8d'], where='post')
            #ax.fill_between(vband_o8d, noiseband_o8d[0], noiseband_o8d[1], step='post', color=colors['o8'], alpha=alpha_err)
            
            for vh in vhalos_2r:
                ax.axvline(vh, color='gray', linestyle='dotted')
            for vh in vhalos_1r:
                ax.axvline(vh, color='black', linestyle='dotted')
            for vh in vhalos_rp:
                ax.axvline(vh, color='brown', linestyle='dotted')  
                
            ax.set_xlim(vranges_sl[sightline])
            yl0, yl1 = ax.get_ylim()
            yl0 = min(yl0, 0.02)
            ax.set_ylim((yl0, yl1))
            
            
            # tabulate line values
            ax = axes[2]
            datadescription_parts = [r'$EW_{\mathrm{O\,VII}} \,/\, \mathrm{m\AA}$' + '\n  '+ '$ = %.1f$',\
                                     r'$\log_{10}\, N_{\mathrm{O\,VII}} \cdot \mathrm{cm}^{2}$' + '\n  '+ '$= %.1f$',\
                                     r'$EW_{\mathrm{O\,VIII}} \,/\, \mathrm{m\AA}$' + '\n  '+ '$ = %.1f$',\
                                     r'$\log_{10}\, N_{\mathrm{O\,VIII}} \cdot \mathrm{cm}^{2}$' + '\n  '+ '$ = %.1f $',\
                                     ]
            datadescription_fills = (grp.attrs['EWrest_A_o7'] * 1e3, grp.attrs['log10_N_cm^-2_o7'], grp.attrs['EWrest_A_o8doublet'] * 1e3, grp.attrs['log10_N_cm^-2_o8'])
            datadescription = '\n'.join(datadescription_parts)
            datadescription = datadescription%datadescription_fills
            ax.text(0.10, 0.95, datadescription, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white',alpha=0.3))
            ax.axis('off')

        # legend
        legend_handles_lines = [mlines.Line2D([], [], color=colors[ion], linestyle=linestyles[kind], label=ionlabels[line]) for (ion, kind, line) in zip(['o7', 'o8'], ['singlet', 'doublet'], ['o7', 'o8d'])]
        #legend_handles_patches = [mpatches.Patch(color=colors['o7'], alpha=alpha_err, label=r'O VII noise-free $\pm 1 \sigma$'), mpatches.Patch(color=colors['o8'], alpha=alpha_err, label=r'O VIII doublet noise-free $\pm 1 \sigma$')]
        lax.legend(handles=legend_handles_lines, ncol=2, fontsize=fontsize, loc='lower center', bbox_to_anchor=(0.5, 0.1))
        lax.axis('off')
        
        # panel labels
        #sfiloc = (0.99, 0.10)
        #axes1[0].text(sfiloc[0], sfiloc[1], '(a)', verticalalignment='bottom', horizontalalignment='right', transform=axes1[0].transAxes, fontsize=fontsize)
        #axes1[1].text(sfiloc[0], sfiloc[1], '(b)', verticalalignment='bottom', horizontalalignment='right', transform=axes1[1].transAxes, fontsize=fontsize)
        #axes1[2].text(sfiloc[0], sfiloc[1], '(c)', verticalalignment='bottom', horizontalalignment='right', transform=axes1[2].transAxes, fontsize=fontsize)
        #axes1[3].text(sfiloc[0], sfiloc[1], '(d)', verticalalignment='bottom', horizontalalignment='right', transform=axes1[3].transAxes, fontsize=fontsize)
        #axes2[0].text(sfiloc[0], sfiloc[1], '(e)', verticalalignment='bottom', horizontalalignment='right', transform=axes2[0].transAxes, fontsize=fontsize)
        #axes2[1].text(sfiloc[0], sfiloc[1], '(f)', verticalalignment='bottom', horizontalalignment='right', transform=axes2[1].transAxes, fontsize=fontsize)
        #axes2[2].text(sfiloc[0], sfiloc[1], '(g)', verticalalignment='bottom', horizontalalignment='right', transform=axes2[2].transAxes, fontsize=fontsize)
        #axes2[3].text(sfiloc[0], sfiloc[1], '(h)', verticalalignment='bottom', horizontalalignment='right', transform=axes2[3].transAxes, fontsize=fontsize)
        #axes3[0].text(sfiloc[0], sfiloc[1], '(i)', verticalalignment='bottom', horizontalalignment='right', transform=axes3[0].transAxes, fontsize=fontsize)
        #axes3[1].text(sfiloc[0], sfiloc[1], '(j)', verticalalignment='bottom', horizontalalignment='right', transform=axes3[1].transAxes, fontsize=fontsize)
        #axes3[2].text(sfiloc[0], sfiloc[1], '(k)', verticalalignment='bottom', horizontalalignment='right', transform=axes3[2].transAxes, fontsize=fontsize)
        #axes3[3].text(sfiloc[0], sfiloc[1], '(l)', verticalalignment='bottom', horizontalalignment='right', transform=axes3[3].transAxes, fontsize=fontsize)

    plt.savefig(outputdir + outname, format='pdf', bbox_inches='tight')
    

def plot_weighted(specfile, sightline, fontsize=fontsize, prop='Temperature_K', proprange=(4., 7.5), dynrange=5., pzoom=None, vzoom=None):
    '''
    prop: 'Temperature_K', 'OverDensity', 'LOSPeculiarVelocity_KMpS'
    '''
    outname = 'mockspectra_%i_withdata_%s'%(sightline, prop) #'mockspectra.pdf'
    if pzoom is not None:
        outname = outname + '_Z-%s-%s-cMpc'%(pzoom[0], pzoom[1])
    if vzoom is not None:
        outname = outname + '_VZ-%s-%s-kmps'%(vzoom[0], vzoom[1])
    outname = outname + '.pdf'
    
    #halocat = '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5' # only care about central here
    
    fig = plt.figure(figsize=(7.0, 11.0))
    numpos = 6
    numvel = 4
    grid = gsp.GridSpec(3, 1, height_ratios=[float(numpos), float(numvel)] +  [0.6], hspace=0.3, top = 0.95, bottom = 0.05, left= 0.05, right=0.95) # total vspace, vspace zoom, pspace zoom sections: extra hspace for plot labels
    grid_pos = gsp.GridSpecFromSubplotSpec(numpos, 1, subplot_spec=grid[0], height_ratios=[1.] * numpos, hspace=0.0, wspace=0.0)
    grid_vel = gsp.GridSpecFromSubplotSpec(numvel, 1, subplot_spec=grid[1], height_ratios=[1.] * numvel, hspace=0.0, wspace=0.0)
    lax = plt.subplot(grid[2])
        
    colors = {'o7': 'red', 'o8': 'blue', 'mass': 'gray'}
    ionlabels={'o7':   r'O VII $21.60\,\mathrm{\AA}$',\
               'o8':  r'O VIII  $18.967\,\mathrm{\AA}$',\
               'mass': 'mass'}
    if prop == 'Temperature_K':
        proplabel = r'$\log_{10}$ T [K]'
        proplog = True
    elif prop == 'OverDensity':
        proplabel = r'$\log_{10} (1 + \delta)$'
        proplog = True
    elif prop == 'LOSPeculiarVelocity_KMpS':
        proplabel = r'$v_{\mathrm{pec, los}} \; [\mathrm{km}/mathrm{s}]$'
        proplog = False
    # read in surrounding halo info
    halofile = outputdir + 'halos_%i.txt'%sightline
    halodata = pd.read_csv(halofile, header=11, index_col='galaxyid', sep=' \t')
    cosmopars = {}
    targets = {'a', 'z', 'h', 'omegam', 'omegalambda', 'boxsize', 'omegab'}
    with open(halofile, 'r') as hf:
        done = False
        while not done: 
            line = hf.readline()
            if line[-1] == '\n':
                line = line[:-1]
            else:
                done = True
            line = line.split(':\t')
            if line[0] in targets:
                cosmopars[line[0]] = float(line[1])
            if set(cosmopars.keys()) == targets:
                done = True      
    # set up spectrum 
    specsample = ps.SpecSample(specfile, specnums=[sightline])
    sl = specsample.sldict[sightline]
    
    vvals = specsample.velocity
    if vzoom is None:
        vlims = (vvals[0], 2. * vvals[-1] - vvals[-2])
    else:
        vlims = tuple(vzoom)
    vlabel = r'$V \; [\mathrm{km}\, \mathrm{s}^{-1}]$'
    plims = (0., cosmopars['boxsize'] / cosmopars['h'])
    plabel = r'Z [cMpc]'
    pvals = np.linspace(plims[0], plims[1], len(vvals) + 1)[:-1]
    if pzoom is not None:
        plims = pzoom # initial plims values used to get pvals
    
    ## plot position space quantities
    paxes = []
    for i in range(numpos):
        ax = fig.add_subplot(grid_pos[i])
        paxes.append(ax)
        
        usexlabel = i == numpos -1
        ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=usexlabel, labelleft=True, labelright=False, labeltop=False, color='black')
        ax.minorticks_on()
        ax.set_xlim(*plims)
        if usexlabel:
            ax.set_xlabel(plabel, fontsize=fontsize)
            
    # mass-weighted
    ax = paxes[0]
    yvals = np.log10(sl.getspectrum('mass', 'OverDensity'))
    ax.plot(pvals, yvals, color=colors['mass'], linestyle='solid')
    maxy = np.max(yvals)
    miny = np.min(yvals[np.isfinite(yvals)])
    ax.set_ylim(max(miny, maxy - dynrange), maxy)
    ax.set_ylabel(r'$\log_{10}(1 + \delta)$')
    
    ax = paxes[1]
    yvals_prop = sl.getspectrum('mass', prop)
    if proplog:
        yvals_prop = np.log10(yvals_prop)
    yvals_prop[np.logical_not(np.isfinite(yvals))] = np.NaN
    ax.plot(pvals, yvals_prop, color=colors['mass'], linestyle='solid')
    ax.set_ylim(*proprange)
    ax.set_ylabel(proplabel, fontsize=fontsize)
    
    # o7-weighted
    ax = paxes[2]
    yvals = np.log10(sl.getspectrum('o7', 'realnionw/NIon_CM3'))
    ax.plot(pvals, yvals, color=colors['o7'], linestyle='solid')
    maxy = np.max(yvals)
    miny = np.min(yvals[np.isfinite(yvals)])
    ax.set_ylim(max(miny, maxy - dynrange), maxy)
    ax.set_ylabel(r'$\log_{10} \, n \; [\mathrm{cm}^{-3}]$')
    
    ax = paxes[3]
    yvals_prop = sl.getspectrum('o7', 'realnionw/%s'%prop)
    if proplog:
        yvals_prop = np.log10(yvals_prop)
    yvals_prop[np.logical_not(np.isfinite(yvals))] = np.NaN
    ax.plot(pvals, yvals_prop, color=colors['o7'], linestyle='solid')
    ax.set_ylim(*proprange)
    ax.set_ylabel(proplabel, fontsize=fontsize)
    
    # o8-weighted
    ax = paxes[4]
    yvals = np.log10(sl.getspectrum('o8', 'realnionw/NIon_CM3'))
    ax.plot(pvals, yvals, color=colors['o8'], linestyle='solid')
    maxy = np.max(yvals)
    miny = np.min(yvals[np.isfinite(yvals)])
    ax.set_ylim(max(miny, maxy - dynrange), maxy)
    ax.set_ylabel(r'$\log_{10} \, n \; [\mathrm{cm}^{-3}]$')
    
    ax = paxes[5]
    yvals_prop = sl.getspectrum('o8', 'realnionw/%s'%prop)
    if proplog:
        yvals_prop = np.log10(yvals_prop)
    yvals_prop[np.logical_not(np.isfinite(yvals))] = np.NaN
    ax.plot(pvals, yvals_prop, color=colors['o8'], linestyle='solid')
    ax.set_ylim(*proprange)
    ax.set_ylabel(proplabel, fontsize=fontsize)
    
    
    ## plot velocity space quantities
    vaxes = []
    for i in range(numvel):
        ax = fig.add_subplot(grid_vel[i])
        vaxes.append(ax)
        
        usexlabel = i == numvel -1
        ax.tick_params(labelsize=fontsize, direction='in', right=True, top=True, axis='both', which='both', labelbottom=usexlabel, labelleft=True, labelright=False, labeltop=False, color='black')
        ax.minorticks_on()
        ax.set_xlim(*vlims)
        if usexlabel:
            ax.set_xlabel(vlabel, fontsize=fontsize)
            
    # o7-weighted
    ax = vaxes[0]
    yvals = np.log10(sl.getspectrum('o7', 'OpticalDepth'))
    ax.plot(vvals, yvals, color=colors['o7'], linestyle='solid')
    maxy = np.max(yvals)
    miny = np.min(yvals[np.isfinite(yvals)])
    ax.set_ylim(max(miny, maxy - dynrange), maxy)
    ax.set_ylabel(r'$\log_{10} \, \tau$')
    
    ax = vaxes[1]
    yvals_prop = sl.getspectrum('o7', 'tauw/%s'%prop)
    if proplog:
        yvals_prop = np.log10(yvals_prop)
    yvals_prop[np.logical_not(np.isfinite(yvals))] = np.NaN
    ax.plot(vvals, yvals_prop, color=colors['o7'], linestyle='solid')
    ax.set_ylim(*proprange)
    ax.set_ylabel(proplabel, fontsize=fontsize)
    
    # o8-weighted
    ax = vaxes[2]
    yvals = np.log10(sl.getspectrum('o8', 'OpticalDepth'))
    ax.plot(vvals, yvals, color=colors['o8'], linestyle='solid')
    maxy = np.max(yvals)
    miny = np.min(yvals[np.isfinite(yvals)])
    ax.set_ylim(max(miny, maxy - dynrange), maxy)
    ax.set_ylabel(r'$\log_{10} \, \tau$')
    
    ax = vaxes[3]
    yvals_prop = sl.getspectrum('o8', 'tauw/%s'%prop)
    if proplog:
        yvals_prop = np.log10(yvals_prop)
    yvals_prop[np.logical_not(np.isfinite(yvals))] = np.NaN
    ax.plot(vvals, yvals_prop, color=colors['o8'], linestyle='solid')
    ax.set_ylim(*proprange)
    ax.set_ylabel(proplabel, fontsize=fontsize)
        
            
    ## legend
    legend_handles_lines = [mlines.Line2D([], [], color=colors[key], linestyle='solid', label=ionlabels[key]) for key in colors.keys()]
    #legend_handles_patches = [mpatches.Patch(color=colors['o7'], alpha=alpha_err, label=r'O VII noise-free $\pm 1 \sigma$'), mpatches.Patch(color=colors['o8'], alpha=alpha_err, label=r'O VIII doublet noise-free $\pm 1 \sigma$')]
    lax.legend(handles=legend_handles_lines, ncol=3, fontsize=fontsize, loc='lower center', bbox_to_anchor=(0.5, 0.1))
    lax.axis('off')
    
    ## annotations and additional info
    # spectrum info
    fig.suptitle(r'Spectrum %i, %s, z=$%.2f$'%(sightline, specfile.split('/')[-2], cosmopars['z']))
    # indicate halos in position space
    #return halodata
    blim = 1.1
    bpar = np.array(halodata['Impact parameter [R200c]'])
    sel = bpar < blim
    alphas = 0.3 +  (blim - bpar[sel]) / blim * 0.7 # closer to sightline -> darker
    zvals = np.array(halodata.loc[sel, 'Z [cMpc]'])
    vvals = np.array(halodata.loc[sel, 'Vlos [km/s, rest]'])
    masses = np.log10(np.array(halodata.loc[sel, 'M200c [Msun]']))
    
    for i in range(np.sum(sel)):
        ax = paxes[0]
        color = 'C%i'%(i%10)
        if zvals[i] <= plims[1] and zvals[i] >= plims[0]:
            ax.axvline(zvals[i], alpha=alphas[i], color=color, zorder=-1)
            ax.text((zvals[i] - plims[0]) / (plims[1] - plims[0]) + 0.01, 0.95, '%.1f'%masses[i], transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', color=color, fontsize=fontsize, zorder=-1)
    
        for ax in [vaxes[0]]:
            if vvals[i] <= vlims[1] and vvals[i] >= vlims[0]:
                ax.axvline(vvals[i], alpha=alphas[i], color=color, zorder=-1)
                ax.text((vvals[i] - vlims[0]) / (vlims[1] - vlims[0]) + 0.01, 0.95, '%.1f'%masses[i], transform=ax.transAxes, horizontalalignment='left', verticalalignment='top', color=color, fontsize=fontsize, zorder=-1)

    plt.savefig(outputdir + outname, format='pdf', bbox_inches='tight')
    

def save_weighted(specfile, sightline, prop='Temperature_K'):
    '''
    prop: 'Temperature_K', 'OverDensity', 'LOSPeculiarVelocity_KMpS'
    '''
    outname = 'sample3_%i_withdata_%s'%(sightline, prop) #'mockspectra.pdf'
    outname = outname + '.txt'
    
    #halocat = '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5' # only care about central here
    
    halofile = outputdir + 'halos_%i.txt'%sightline
    #halodata = pd.read_csv(halofile, header=11, index_col='galaxyid', sep=' \t')
    cosmopars = {}
    targets = {'a', 'z', 'h', 'omegam', 'omegalambda', 'boxsize', 'omegab'}
    with open(halofile, 'r') as hf:
        done = False
        while not done: 
            line = hf.readline()
            if line[-1] == '\n':
                line = line[:-1]
            else:
                done = True
            line = line.split(':\t')
            if line[0] in targets:
                cosmopars[line[0]] = float(line[1])
            if set(cosmopars.keys()) == targets:
                done = True      
    # set up spectrum 
    specsample = ps.SpecSample(specfile, specnums=[sightline])
    sl = specsample.sldict[sightline]
    
    vvals = specsample.velocity
    plims = (0., cosmopars['boxsize'] / cosmopars['h'])
    pvals = np.linspace(plims[0], plims[1], len(vvals) + 1)[:-1]
    
    columns = ['LOS velocity [km/s], rest', 'LOS position [cMpc]',\
               'Mass(position) [rho / (rho_c Omegab)]', 'Mass-weighted %s (position)'%prop,\
               'n_{O VII}(position) [cm^-3]', 'n_{O VII}-weighted %s (position)'%prop,\
               'n_{O VIII}(position) [cm^-3]', 'n_{O VIII}-weighted %s (position)'%prop,\
               'optical depth O VII (velocity)', 'optical depth O VII weighted %s (velocity)'%prop,\
               'optical depth O VIII (velocity)', 'optical depth O VIII weighted %s (velocity)'%prop
               ]
    data = {'LOS velocity [km/s], rest': vvals,\
            'LOS position [cMpc]': pvals} 
       
    data['Mass(position) [rho / (rho_c Omegab)]']  = sl.getspectrum('mass', 'OverDensity')
    data['Mass-weighted %s (position)'%prop]       = sl.getspectrum('mass', prop)
    data['n_{O VII}(position) [cm^-3]']            = sl.getspectrum('o7', 'realnionw/NIon_CM3')
    data['n_{O VII}-weighted %s (position)'%prop]  = sl.getspectrum('o7', 'realnionw/%s'%prop)
    data['n_{O VIII}(position) [cm^-3]']           = sl.getspectrum('o8', 'realnionw/NIon_CM3')
    data['n_{O VIII}-weighted %s (position)'%prop] = sl.getspectrum('o8', 'realnionw/%s'%prop)
    data['optical depth O VII (velocity)']         = sl.getspectrum('o7', 'OpticalDepth')
    data['optical depth O VII weighted %s (velocity)'%prop] = sl.getspectrum('o7', 'tauw/%s'%prop)
    data['optical depth O VIII (velocity)']        = sl.getspectrum('o8', 'OpticalDepth')
    data['optical depth O VIII weighted %s (velocity)'%prop] = sl.getspectrum('o8', 'tauw/%s'%prop)

    with open(outputdir + outname, 'w') as fo:
        metadata = 'weighted values are set to zero when weights are zero: avoid plotting those zeros\n'
        fo.write(metadata)
        header = '\t '.join(columns) + '\n'
        fo.write(header)
        template = '\t '.join(['%s'] * len(columns)) + '\n'
        #print(template)
        for i in range(len(vvals)):
            #print((data[col][i] for col in columns))
            fo.write(template%tuple([data[col][i] for col in columns]))
            
def coverimage(depth='all', dohalos=True):
    halocat = '/net/luttero/data2/proc/catalogue_RefL0025N0376_snap28_aperture30.hdf5'
    if depth == 'all':
        name = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0376_28_test3.4_C2Sm_1600pix_25.0slice_z-projection.npz'
        zmin = 0.
        zmax = 25.
    elif depth == 'half':
        name = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0376_28_test3.4_C2Sm_1600pix_12.5slice_zcen6.25_z-projection.npz'
        zmin = 0.
        zmax = 12.5
    mhmin = 2e11
    
    ndir = '/net/quasar/data2/wijers/temp/'
    img = np.load(ndir + name)['arr_0']
    cmap = 'afmhot'
    vmin = -5.
    ancolor = 'lightseagreen'
    fontsize=13.
    
    cmap = mpl.cm.get_cmap(cmap)
    cmap.set_under(cmap(0.))
    
    fig, (lax, ax) = plt.subplots(ncols=1, nrows=2, figsize=(5., 5.5), gridspec_kw={'height_ratios': [0.5, 5.], 'hspace': 0.0, 'left': 0., 'right': 1., 'top': 1., 'bottom':0.})
    ax.set_facecolor(cmap(0.))
    img[np.logical_not(np.isfinite(img))] = -100.
    
    ax.imshow(img.T, vmin=vmin, origin='lower', interpolation='nearest', extent=(0., 25., 0., 25.), cmap=cmap)
    ax.axis('off')
    
    # add text
    lax.axis('off')
    lax.set_xlim(0., 25.)
    lax.set_ylim(0., 1.)
    lax.plot([0.5, 5.5], [0.1, 0.1], linewidth=4., color='black')
    lax.text(3./25., 0.15, '5 Mpc', transform=lax.transAxes, color='black', fontsize=fontsize, horizontalalignment='center', verticalalignment='bottom')
    lax.text(0.5, 0.2, 'Gas metallicity (Eagle)', transform=lax.transAxes, color='black', fontsize=fontsize + 1, horizontalalignment='center', verticalalignment='bottom')
    
    # add halo locations
    if dohalos:
        with h5py.File(halocat, 'r') as halos:
            cosmopars = {key: item for (key, item) in halos['Header/cosmopars'].attrs.items()}
            xs = np.array(halos['Xcom_cMpc'])
            ys = np.array(halos['Ycom_cMpc'])
            zs = np.array(halos['Zcom_cMpc'])
            sz = np.array(halos['R200c_pkpc']) * 1e-3 / cosmopars['a'] # z=0 -> expansion factor = 1
            msel = np.array(halos['M200c_Msun']) >= mhmin
            zsel = np.logical_and(zs >= zmin, zs <= zmax)
            sel = np.logical_and(msel, zsel)
            del msel
            del zsel
            xs = xs[sel]
            ys = ys[sel]
            sz = sz[sel]
        
        patches = [mpl.patches.Circle((xs[ind], ys[ind]), sz[ind]) \
                   for ind in range(len(xs))] # x, y axes only
    
        collection = mpl.collections.PatchCollection(patches)
        collection.set(edgecolor=ancolor, facecolor='none', linewidth=0.7)
        ax.add_collection(collection)
    
    plt.savefig(outputdir + 'coverfig_%s_halos-%s.pdf'%(depth, dohalos), format='pdf')