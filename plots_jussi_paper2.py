#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:26:13 2019

@author: wijers
"""

import numpy as np
import string

import ion_line_data as ild
import plot_utils as pu
import tol_colors as tc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.legend_handler as mlh
import matplotlib.collections as mcol

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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap


class HandlerDashedLines(mlh.HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = ((height) / (numlines + 1)) * np.ones(xdata.shape, float)
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = mlines.Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[0] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines
    
    
def plot_coldenscorr_Tion(ionT='o6', ion1='o6', ion2='o8', Tlim=5.8):
    
    if {ion1, ion2} == {'o6', 'o8'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    if {ion1, ion2} == {'o6', 'o7'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o7_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    
    imgname = '/home/wijers/Documents/papers/jussi_paper2_Ton180/hist_coldens_%s-%s_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_splitby_T-%s-%.2f.eps'%(ion1, ion2, ionT, Tlim)
    
    fontsize = 12
    xlabel = r'$\log{10} \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion1, mathmode=True))
    ylabel = r'$\log{10} \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion2, mathmode=True))
    clabel = r'$\log_{10}$ relative fraction of absorbers'
    cmap = truncate_colormap(mpl.cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7)
    
    percentiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    color_lowerT = 'C0'
    color_upperT = 'C1'
    
    with np.load(histf) as fi:
        #print(fi.keys())
        #print(fi['dimension'])
        #print(fi['edges'])
        
        dimension = fi['dimension']
        ax_no6 = np.where(dimension == 'N%s'%(string.capwords(ion1)))[0][0]
        ax_no8 = np.where(dimension == 'N%s'%(string.capwords(ion2)))[0][0]
        ax_T = np.where(dimension == 'Temperature_w_N%s'%(string.capwords(ionT)))[0][0]
        numax = len(dimension)
        
        edges_o6 = fi['edges'][ax_no6]
        edges_o8 = fi['edges'][ax_no8]
        edges_T  = fi['edges'][ax_T]
        
        cutind_T = np.argmin(np.abs(edges_T - Tlim))
        cutval_T = edges_T[cutind_T]
        lowersel_T = slice(None, cutind_T, None)
        uppersel_T = slice(cutind_T, None, None)
        lowersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} < %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        uppersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} > %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        
        lowersel_all = [slice(None, None, None)] * numax
        lowersel_all[ax_T] = lowersel_T
        lowersel_all = tuple(lowersel_all)
        
        uppersel_all = [slice(None, None, None)] * numax
        uppersel_all[ax_T] = uppersel_T
        uppersel_all = tuple(uppersel_all)
        
        sumaxes = set(np.arange(numax))
        sumaxes -= {ax_no6, ax_no8}
        sumaxes = sorted(list(sumaxes))
        sumaxes = tuple(sumaxes)
        
        hist = fi['bins']
        hist_lowerT = np.sum(hist[lowersel_all], axis=sumaxes)
        hist_upperT = np.sum(hist[uppersel_all], axis=sumaxes)
        
    
    fig = plt.figure(figsize=(5.3, 3.))
    grid = gsp.GridSpec(1, 3, hspace=0.0, wspace=0.0, width_ratios=[5., 5., 1.])
    axes = [fig.add_subplot(grid[i]) for i in range(2)]
    cax  = fig.add_subplot(grid[2])
    
    # get percentiles from true histogram
    ax_histsum = 1 if ax_no8 > ax_no6 else 0
    percentiles_lowerT = percentiles_from_histogram(hist_lowerT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    percentiles_upperT = percentiles_from_histogram(hist_upperT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    cens_o6 = edges_o6[:-1] + 0.5 * np.diff(edges_o6)
    plotwhere_lowerT = np.sum(hist_lowerT, axis=ax_histsum) >= 20
    plotwhere_upperT = np.sum(hist_upperT, axis=ax_histsum) >= 20
    
    # switch to density histogram
    if ax_no6 < ax_no8:
        divby = np.diff(edges_o6)[:, np.newaxis] * np.diff(edges_o8)[np.newaxis, :]
    else:
        divby = np.diff(edges_o6)[np.newaxis, :] * np.diff(edges_o8)[:, np.newaxis]
    hist_lowerT /= divby
    hist_upperT /= divby
    
    vmax = np.log10(max(np.max(hist_upperT), np.max(hist_lowerT)))
    vmin = vmax - 8.
    xlim = (12., 16.5)
    ylim = (13., 17.5)
    
    if ax_no6 > ax_no8:
        hist_lowerT = hist_lowerT.T
        hist_upperT = hist_upperT.T
    
    # left plot
    ax = axes[0]
    ax.tick_params(labelsize=fontsize - 1, which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    img = ax.pcolormesh(edges_o6, edges_o8, np.log10(hist_lowerT.T), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    
    for pind in range(len(percentiles)):
        ax.plot(cens_o6[plotwhere_upperT], percentiles_upperT[pind][plotwhere_upperT], color=color_upperT, linewidth=1.5, linestyle=linestyles[pind])
        ax.plot(cens_o6[plotwhere_lowerT], percentiles_lowerT[pind][plotwhere_lowerT], color=color_lowerT, linewidth=2.5, linestyle=linestyles[pind])
    
    ax.text(0.05, 0.95, lowersel_T_label, fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    
    # right plot
    # left plot
    ax = axes[1]
    ax.tick_params(labelsize=fontsize - 1, which='both', direction='in', top=True, right=True, labelleft=False)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=fontsize)
    #ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    ax.pcolormesh(edges_o6, edges_o8, np.log10(hist_upperT.T), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    
    for pind in range(len(percentiles)):
        ax.plot(cens_o6[plotwhere_lowerT], percentiles_lowerT[pind][plotwhere_lowerT], color=color_lowerT, linewidth=1.5, linestyle=linestyles[pind])
        ax.plot(cens_o6[plotwhere_upperT], percentiles_upperT[pind][plotwhere_upperT], color=color_upperT, linewidth=2.5, linestyle=linestyles[pind])
     
    ax.text(0.05, 0.95, uppersel_T_label, fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    
    # color bar
    plt.colorbar(img, cax=cax, extend='min')
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(10.)
    cax.set_ylabel(clabel, fontsize=fontsize)
    
    # legend
    perc_toplot = percentiles[: (len(percentiles) + 1) // 2]
    lcs = []
    line = [[(0, 0)]]
    for pind in range(len(perc_toplot)):
        # set up the proxy artist
        subcols = [mpl.colors.to_rgba(color_lowerT, alpha=1.), mpl.colors.to_rgba(color_upperT, alpha=1.)]
        subcols = np.array(subcols)
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[pind], linewidth=2, colors=subcols)
        lcs.append(lc)
    perclabels = ['%.0f'%(perc * 100.) if perc != 0.5 else '%.0f'%( 100. * perc) for perc in perc_toplot]
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    lax = axes[0]
    lax.legend(lcs, perclabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.02, -0.02), frameon=False)
    #lax.axis('off')
    typelines = [mlines.Line2D([], [], color=color_lowerT, linestyle='solid', label=r'$T < %.1f$'%(cutval_T), linewidth=2.),\
                 mlines.Line2D([], [], color=color_upperT, linestyle='solid', label=r'$T < %.1f$'%(cutval_T), linewidth=2.)]
    labels = [r'$T < %.1f$'%(cutval_T), r'$T > %.1f$'%(cutval_T)]
    lax = axes[1]
    lax.legend(typelines, labels, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.02, -0.02), frameon=False, handlelength=1.5)
    
    plt.savefig(imgname, format='eps', bbox_inches='tight')
    
    
### notes on second iteration plot options
## lines/data Jussi added to the plot: 
# - FUV min (O VI): ~13
# - X-ray min (RGS): ~15.5
# - X-ray min (X-IFU): ~15.0
# - FUV O VI measurement
# - CIE O VI /O VIII fit

## Joop's feedback:
# We are not making particularly good use of the simulation. We could for
# example have asked how likely it is for an OVI absorber with log N ~
# 13.76 to have associated OVII similar to observed. And we could have
# checked how this changes if T_OVI < T(b_OVI) or T_OVI < T(b_OVI +
# Nsigma) if we want to use a more conservative upper limit on b_OVI. It
#  may actually be quite improbable to have such high NOVII. This would
# limit our confidence in the detection, or in EAGLE, but so be it.
#
#Fig. 4 is too confusing. Also, it doesn't show the distribution of
#NOVII-NOVI without restrictions. I would suggest one of the following
#two options:
#A. One panel with color coding showing the full distribution, i.e. w/o a
#constraint on TOVI. On top of that the two sets of contours for
#different TOVI. The grey scale should however show the fraction at fixed
#NOVI since our analysis is conditional on having some particular NOVI. I
#think this is what the coloured contours show, but not the grey scale
#(otherwise I don't understand the large dynamic range in the colour
#scale, > 7 decades).
#B. Two panels, both only showing contours. One panel showing the
#distribution of NOVII at fixed NOVI. The other showing the same for the
#two TOVI cuts.
#
#Also, I don't like the contour levels. Contour percentiles normally
#refer to the fraction enclosed, whereas here it seems to be 1 minus that
#fraction.
#    
#I do not understand the rational for cutting at TOVI  = 10^5.8 K
#however. I get that this gives fOVII/fOVI > 100 and that this
#corresponds to the sensitivity ratio, but I don't think that is a good
#motivation. We want to know how typical the observed OVII is given the
#constraints on OVI. I would therefore show TOVI cuts motivated by the
#measured bOVI. E.g. T < measured bOVI (assuming thermal broadening), or
#T < measured bOVI + 1/2/3 sigma (which would give log TOVI < 5.7, 5.9,
#6.0 resp.).

## Jussi's comment:
# perhaps we should make the division using the 3sigma upper limit on b_OVI 
# -- as it corresponds to 10^6 K thermal line broadening, 
# this T cut would provide more general type of division to hot/warm phases. 
# Joop also suggests two different alternatives to improve the readability 
# of Fig. 4. In my opinion, 
# option A (one panel with the full NOVI-NOVII distribution,
# and the two TOVI distribution contours plotted on top) 
# would be very suitable to the discussion section.
    
## Alexis Finoguenov
#We need to estimate the distribution of cospatial (or wrapping each 
#other in any order) warm and hot phases. For that we need to draw a 
#distribution of OVI from warm and OVII from hot.


def plot_coldenscorr_Tion_v2(ionT='o6', ion1='o6', ion2='o8', Tlim=6.0, nsig_range=1., table=None):
    xlines = {'FUV': (13., 'solid')}
    ylines = {'RGS': (15.5, 'solid'),\
              'X-IFU': (15.0, 'dashed')}
    
     # table 2 log No6, Delta log No6
    o6meas_fuv_zo6 = (13.76, 0.08)
    # table 3
    o6meas_cie_zo6 = (13.9, 0.2)
    o6meas_cie_zo7 = (13.9, 0.2)
    o7meas_cie_zo6 = (16.4, 0.2)
    o7meas_cie_zo7 = (16.4, 0.2)
    o7meas_slab_zo6 = (16.59, -0.28, 0.24)
    o7meas_slab_zo7 = (16.62, -0.28, 0.25)
    o8meas_cie_zo6 = (16.0, 0.2)
    o8meas_cie_zo7 = (16.0, 0.2)
    o8meas_slab_zo6 = (15.85, -0.65, 0.32)
    o8meas_slab_zo7 = (15.82, -0.78, 0.49)
    
    T_cie_zo6 = (1.7e6, 0.2e6)
    T_cie_zo7 = (1.7e6, 0.2e6)
    
    logTcuts = [5.7, 5.9, 6.0]
    
    measvals = {'fuv': {'zo6': {'o6': o6meas_fuv_zo6}},\
                'cie': {'zo6': {'o6': o6meas_cie_zo6,\
                                'o7': o7meas_cie_zo6,\
                                'o8': o8meas_cie_zo6,\
                                'T':  T_cie_zo6},\
                        'zo7': {'o6': o6meas_cie_zo7,\
                                'o7': o7meas_cie_zo7,\
                                'o8': o8meas_cie_zo7,\
                                'T':  T_cie_zo7},\
                        },\
                 'slab': {'zo6': {'o7': o7meas_slab_zo6,\
                                  'o8': o8meas_slab_zo6,\
                                 },\
                         'zo7':  {'o7': o7meas_slab_zo7,\
                                  'o8': o8meas_slab_zo7,\
                                 },\
                          },\
                }
    
    
    if {ion1, ion2} == {'o6', 'o8'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    if {ion1, ion2} == {'o6', 'o7'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o7_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    
    imgname = '/home/wijers/Documents/papers/jussi_paper2_Ton180/hist_coldens_%s-%s_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_splitby_T-%s-%.2f_v2.eps'%(ion1, ion2, ionT, Tlim)
    
    fontsize = 12
    xlabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion1, mathmode=True))
    ylabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion2, mathmode=True))
    clabel = r'$\log$ relative fraction of absorbers'
    cmap = truncate_colormap(mpl.cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7)
    
    percentiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    color_lowerT = 'C0'
    color_upperT = 'C1'
    
    with np.load(histf) as fi:
        #print(fi.keys())
        #print(fi['dimension'])
        #print(fi['edges'])
        
        dimension = fi['dimension']
        ax_no6 = np.where(dimension == 'N%s'%(string.capwords(ion1)))[0][0]
        ax_no8 = np.where(dimension == 'N%s'%(string.capwords(ion2)))[0][0]
        ax_T = np.where(dimension == 'Temperature_w_N%s'%(string.capwords(ionT)))[0][0]
        numax = len(dimension)
        
        edges_o6 = fi['edges'][ax_no6]
        edges_o8 = fi['edges'][ax_no8]
        edges_T  = fi['edges'][ax_T]
        
        cutind_T = np.argmin(np.abs(edges_T - Tlim))
        cutval_T = edges_T[cutind_T]
        lowersel_T = slice(None, cutind_T, None)
        uppersel_T = slice(cutind_T, None, None)
        #lowersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} < %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        #uppersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} > %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        
        lowersel_all = [slice(None, None, None)] * numax
        lowersel_all[ax_T] = lowersel_T
        lowersel_all = tuple(lowersel_all)
        
        uppersel_all = [slice(None, None, None)] * numax
        uppersel_all[ax_T] = uppersel_T
        uppersel_all = tuple(uppersel_all)
        
        sumaxes = set(np.arange(numax))
        sumaxes -= {ax_no6, ax_no8}
        sumaxes = sorted(list(sumaxes))
        sumaxes = tuple(sumaxes)
        
        hist_base = fi['bins']
        hist_lowerT = np.sum(hist_base[lowersel_all], axis=sumaxes)
        hist_upperT = np.sum(hist_base[uppersel_all], axis=sumaxes)
        hist = np.sum(hist_base, axis=sumaxes)
    
    if table == 'atmeas':
        print('model\t z_model\t T range \t N_meas \t N_meas - sigma \t N_meas + sigma \t percentile(Nmeas) \t percentile(Nmeas - sigma) \t percentile(Nmeas + sigma)\n')
        for logT in logTcuts:
            cutind_T = np.argmin(np.abs(edges_T - logT))
            cutval_T = edges_T[cutind_T]
            lowersel_T = slice(None, cutind_T, None)
            uppersel_T = slice(cutind_T, None, None)
            
            mino6 = o6meas_fuv_zo6[0] - nsig_range * o6meas_fuv_zo6[1] # left edge of lowest bin
            maxo6 = o6meas_fuv_zo6[0] + nsig_range * o6meas_fuv_zo6[1] # right edge of highest bin
            minind_o6 = np.argmin(np.abs(edges_o6 - mino6))
            maxind_o6 = np.argmin(np.abs(edges_o6 - maxo6))
            sel_o6 = slice(minind_o6, maxind_o6, None)
            
            uppersel_all = [slice(None, None, None)] * numax
            uppersel_all[ax_T] = uppersel_T
            uppersel_all[ax_no6] = sel_o6
            uppersel_all = tuple(uppersel_all)
            
            lowersel_all = [slice(None, None, None)] * numax
            lowersel_all[ax_T] = lowersel_T
            lowersel_all[ax_no6] = sel_o6
            lowersel_all = tuple(lowersel_all)
            
            sel_all = [slice(None, None, None)] * numax
            sel_all[ax_no6] = sel_o6
            sel_all = tuple(sel_all)
            
            sumaxes = set(np.arange(numax))
            sumaxes -= {ax_no8}
            sumaxes = sorted(list(sumaxes))
            sumaxes = tuple(sumaxes)
            
            hist_lowerT_t = np.sum(hist_base[lowersel_all], axis=sumaxes)
            hist_upperT_t = np.sum(hist_base[uppersel_all], axis=sumaxes)
            hist_all_t = np.sum(hist_base[sel_all], axis=sumaxes)
            
            cdist_lowerT = np.cumsum(hist_lowerT_t) / np.sum(hist_lowerT_t)
            cdist_upperT = np.cumsum(hist_upperT_t) / np.sum(hist_upperT_t)
            cdist_all    = np.cumsum(hist_all_t) / np.sum(hist_all_t)
            
            pvals_lowerT = {}
            pvals_upperT = {}
            pvals_all   = {}
            
            print('\n')
            print('T upper/lower cut: 10^%f K'%logT)
            for model in ['slab', 'cie']:
                pvals_lowerT[model] = {}
                pvals_upperT[model] = {}
                pvals_all[model]   = {}
                for zobs in measvals[model].keys():
                    val = measvals[model][zobs][ion2]
                    Nmeas = val[0]
                    Nmin  = val[0] - np.abs(val[1])
                    Nmax  = val[0] + val[2] if len(val) == 3 else val[0] + val[1]
                    
                    pvals_lowerT[model][zobs] = [pu.linterpsolve(edges_o8[1:], cdist_lowerT, Nmeas),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_lowerT, Nmin),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_lowerT, Nmax),\
                                                ]
                    pvals_upperT[model][zobs] = [pu.linterpsolve(edges_o8[1:], cdist_upperT, Nmeas),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_upperT, Nmin),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_upperT, Nmax),\
                                                ]
                    pvals_all[model][zobs]    = [pu.linterpsolve(edges_o8[1:], cdist_all, Nmeas),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_all, Nmin),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_all, Nmax),\
                                                ]
                    print('\n')
                    print('%s \t %s \t %s \t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s'%((model, zobs, 'lower', Nmeas, Nmin, Nmax) + tuple(pvals_lowerT[model][zobs])))
                    print('%s \t %s \t %s \t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s'%((model, zobs, 'upper', Nmeas, Nmin, Nmax) + tuple(pvals_upperT[model][zobs])))
                    print('%s \t %s \t %s \t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s'%((model, zobs, 'all', Nmeas, Nmin, Nmax) + tuple(pvals_all[model][zobs])))
    
    elif table == 'gemeas':
        print('model\t z_model\t T range \t N_meas \t N_meas - sigma \t N_meas + sigma \t percentile(Nmeas) \t percentile(Nmeas - sigma) \t percentile(Nmeas + sigma)\n')
        for logT in logTcuts:
            #cutind_T = np.argmin(np.abs(edges_T - logT))
            #cutval_T = edges_T[cutind_T]
            #lowersel_T = slice(None, cutind_T, None)
            #uppersel_T = slice(cutind_T, None, None)
            
            mino6 = o6meas_fuv_zo6[0] - nsig_range * o6meas_fuv_zo6[1] # left edge of lowest bin
            maxo6 = o6meas_fuv_zo6[0] + nsig_range * o6meas_fuv_zo6[1] # right edge of highest bin
            ato6 = o6meas_fuv_zo6[0]
            minind_o6 = np.argmin(np.abs(edges_o6 - mino6))
            maxind_o6 = np.argmin(np.abs(edges_o6 - maxo6))
            atind_o6 = np.argmin(np.abs(edges_o6 - ato6))
            
            sel_o6 = slice(atind_o6, None, None)
            
            sel_all = [slice(None, None, None)] * numax
            sel_all[ax_no6] = sel_o6
            sel_all = tuple(sel_all)
            
            sumaxes = set(np.arange(numax))
            sumaxes -= {ax_no8}
            sumaxes = sorted(list(sumaxes))
            sumaxes = tuple(sumaxes)
            
            hist_all_t = np.sum(hist_base[sel_all], axis=sumaxes)          
            cdist_all    = np.cumsum(hist_all_t) / np.sum(hist_all_t)
            pvals_all   = {}
            
            for model in ['slab', 'cie']:
                pvals_all[model]   = {}
                for zobs in measvals[model].keys():
                    val = measvals[model][zobs][ion2]
                    Nmeas = val[0]
                    Nmin  = val[0] - np.abs(val[1])
                    Nmax  = val[0] + val[2] if len(val) == 3 else val[0] + val[1]
                    
                    pvals_all[model][zobs]    = [pu.linterpsolve(edges_o8[1:], cdist_all, Nmeas),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_all, Nmin),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_all, Nmax),\
                                                ]
                    print('%s \t %s \t %s \t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s'%((model, zobs, 'all', Nmeas, Nmin, Nmax) + tuple(pvals_all[model][zobs])))

    elif table.startswith('gemeas_T'):
        parts = table.split('-')
        logTmin = float(parts[1])
        logTmax = float(parts[2])
        
        cutind_Tmin = np.argmin(np.abs(edges_T - logTmin))
        cutval_Tmin = edges_T[cutind_Tmin]
        cutind_Tmax = np.argmin(np.abs(edges_T - logTmax))
        cutval_Tmax = edges_T[cutind_Tmax]
        losel_T = slice(None, cutind_Tmin, None)
        misel_T = slice(cutind_Tmin, cutind_Tmax, None)
        hisel_T = slice(None, cutind_Tmax, None)
        alsel_T = slice(None, None, None)
        
        Tsels = [losel_T, misel_T, hisel_T, alsel_T]
        Tranges = ['< %s'%cutval_Tmin, '%s - %s'%(cutval_Tmin, cutval_Tmax), '> %s'%cutval_Tmax, 'all']    

        print('model\t z_model\t T range \t N_meas \t N_meas - sigma \t N_meas + sigma \t percentile(Nmeas) \t percentile(Nmeas - sigma) \t percentile(Nmeas + sigma)\n')
        for Tsel, Trange in zip(Tsels, Tranges):
            #cutind_T = np.argmin(np.abs(edges_T - logT))
            #cutval_T = edges_T[cutind_T]
            #lowersel_T = slice(None, cutind_T, None)
            #uppersel_T = slice(cutind_T, None, None)
            
            mino6 = o6meas_fuv_zo6[0] - nsig_range * o6meas_fuv_zo6[1] # left edge of lowest bin
            maxo6 = o6meas_fuv_zo6[0] + nsig_range * o6meas_fuv_zo6[1] # right edge of highest bin
            ato6 = o6meas_fuv_zo6[0]
            minind_o6 = np.argmin(np.abs(edges_o6 - mino6))
            maxind_o6 = np.argmin(np.abs(edges_o6 - maxo6))
            atind_o6 = np.argmin(np.abs(edges_o6 - ato6))
            
            sel_o6 = slice(atind_o6, None, None)
            
            sel_all = [slice(None, None, None)] * numax
            sel_all[ax_no6] = sel_o6
            sel_all[ax_T] = Tsel
            sel_all = tuple(sel_all)
            
            sumaxes = set(np.arange(numax))
            sumaxes -= {ax_no8}
            sumaxes = sorted(list(sumaxes))
            sumaxes = tuple(sumaxes)
            
            hist_all_t = np.sum(hist_base[sel_all], axis=sumaxes)          
            cdist_all    = np.cumsum(hist_all_t) / np.sum(hist_all_t)
            pvals_all   = {}
            
            for model in ['slab', 'cie']:
                pvals_all[model]   = {}
                for zobs in measvals[model].keys():
                    val = measvals[model][zobs][ion2]
                    Nmeas = val[0]
                    Nmin  = val[0] - np.abs(val[1])
                    Nmax  = val[0] + val[2] if len(val) == 3 else val[0] + val[1]
                    
                    pvals_all[model][zobs]    = [pu.linterpsolve(edges_o8[1:], cdist_all, Nmeas),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_all, Nmin),\
                                                 pu.linterpsolve(edges_o8[1:], cdist_all, Nmax),\
                                                ]
                    print('%s \t %s \t %s \t %s \t\t %s \t\t %s \t\t %s \t\t %s \t\t %s'%((model, zobs, Trange, Nmeas, Nmin, Nmax) + tuple(pvals_all[model][zobs])))
                    
                    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[5., 1.])
    ax = fig.add_subplot(grid[0])
    cax  = fig.add_subplot(grid[1])
    
    # get percentiles from true histogram
    ax_histsum = 1 if ax_no8 > ax_no6 else 0
    percentiles_lowerT = percentiles_from_histogram(hist_lowerT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    percentiles_upperT = percentiles_from_histogram(hist_upperT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    cens_o6 = edges_o6[:-1] + 0.5 * np.diff(edges_o6)
    plotwhere_lowerT = np.sum(hist_lowerT, axis=ax_histsum) >= 20
    plotwhere_upperT = np.sum(hist_upperT, axis=ax_histsum) >= 20
    
    # switch to density histogram
    if ax_no6 < ax_no8:
        divby = np.diff(edges_o6)[:, np.newaxis] * np.diff(edges_o8)[np.newaxis, :]
    else:
        divby = np.diff(edges_o6)[np.newaxis, :] * np.diff(edges_o8)[:, np.newaxis]
    #hist_lowerT /= divby
    #hist_upperT /= divby
    hist_toplot = hist / divby
    
    #vmax = np.log10(max(np.max(hist_upperT), np.max(hist_lowerT)))
    xlim = (12.1, 16.5)
    ylim = (13.1, 17.5)
    vmax = np.log10(np.max(hist_toplot[edges_o6[1:] > xlim[0], edges_o8[1:] > ylim[0]]))
    vmin = vmax - 8.
    
    if ax_no6 > ax_no8:
        hist_lowerT = hist_lowerT.T
        hist_upperT = hist_upperT.T
    
    ax.tick_params(labelsize=fontsize - 1, which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    img = ax.pcolormesh(edges_o6, edges_o8, np.log10(hist_lowerT.T), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    
    for pind in range(len(percentiles)):
        ax.plot(cens_o6[plotwhere_upperT], percentiles_upperT[pind][plotwhere_upperT], color=color_upperT, linewidth=2., linestyle=linestyles[pind])
        ax.plot(cens_o6[plotwhere_lowerT], percentiles_lowerT[pind][plotwhere_lowerT], color=color_lowerT, linewidth=2., linestyle=linestyles[pind])
    
    # add lines for detection limits and measurements
    ancolor = 'C2'
    for label in xlines:
        ax.axvline(xlines[label][0], color=ancolor, linestyle=xlines[label][1], linewidth=1.5)
        ax.text((xlines[label][0] - xlim[0]) / (xlim[1] - xlim[0]) + 0.01, 0.8,  label,\
                color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='left', verticalalignment='center', rotation=90.)
    for label in ylines:
        ax.axhline(ylines[label][0], color=ancolor, linestyle=ylines[label][1], linewidth=1.5)
        ax.text(0.99, (ylines[label][0] - ylim[0]) / (ylim[1] - ylim[0]) + 0.01,  label,\
                color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='right', verticalalignment='bottom', rotation=0.)
    # FUV
    fuvcolor='C3'
    fuvpoint = measvals['fuv']['zo6']['o6']
    ax.axvline(fuvpoint[0], 0.0, 0.7, color=fuvcolor)
    ax.errorbar(fuvpoint[0], ylim[0] + 0.35 * (ylim[1] - ylim[0]), xerr=fuvpoint[1:],\
                linewidth=1.5, color=fuvcolor, zorder=5)
    
    # X-ray
    xraycolor = 'black'
    xraypoint_x = measvals['cie']['zo7'][ion1]
    xraypoint_y = measvals['cie']['zo7'][ion2]   
    ax.errorbar(xraypoint_x[0], xraypoint_y[0], xerr=xraypoint_x[1:], yerr=xraypoint_y[1:],\
                linewidth=2., color=xraycolor, zorder=5)
    # color bar
    plt.colorbar(img, cax=cax, extend='min')
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(10.)
    cax.set_ylabel(clabel, fontsize=fontsize)
    
    # legend
    perc_toplot = percentiles[: (len(percentiles) + 1) // 2]
    lcs = []
    line = [[(0, 0)]]
    for pind in range(len(perc_toplot)):
        # set up the proxy artist
        subcols = [mpl.colors.to_rgba(color_lowerT, alpha=1.), mpl.colors.to_rgba(color_upperT, alpha=1.)]
        subcols = np.array(subcols)
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[pind], linewidth=2, colors=subcols)
        lcs.append(lc)
    perclabels = ['%.0f %%'%(2. * (0.5 - perc) * 100.) if perc != 0.5 else 'median' for perc in perc_toplot]
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    typelines = [mlines.Line2D([], [], color=color_lowerT, linestyle='solid', label=r'$\log\, \mathrm{T} < %.1f$'%(cutval_T), linewidth=2.),\
                 mlines.Line2D([], [], color=color_upperT, linestyle='solid', label=r'$\log\, \mathrm{T} < %.1f$'%(cutval_T), linewidth=2.)]
    labels = [r'$\log\, \mathrm{T}(\mathrm{%s}) < %.1f$'%(ild.getnicename(ion1, mathmode=True), cutval_T), r'$\log\, \mathrm{T}(\mathrm{%s}) > %.1f$'%(ild.getnicename(ion1, mathmode=True), cutval_T)]
    lax = ax
    leg1 = lax.legend(typelines, labels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)
    leg2 = lax.legend(lcs, perclabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.15), frameon=False)
    lax.add_artist(leg1)
    lax.add_artist(leg2)
    #lax.axis('off')
   
    #lax = axes[1]
    #lax.legend(typelines, labels, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.02, -0.02), frameon=False, handlelength=1.5)
    
    plt.savefig(imgname, format='eps', bbox_inches='tight')


### second round revised plots:
# min. O VI column vs O VII
# min. O VI column with T cuts vs O VII
# O VI min column + H I max column vs O VII
# percentiles 1 and 99 to roughly mimic initial sightline selection
    

def plot_coldenscorr_Tion_v3(plotnum=1, ionT_split=5.7, nh1_split=13.0, nsig_range=1., table=None):
    '''
    plotnum 1: O VI - O VII correlation with 1, 50, 99% markers
    plotnum 2: O VI - O VII correlation with 1, 50, 99% markers 
                                        for O VI-weighted temperature ranges
    plotnum 3: O VI - O VII correlation with percentiles markers for H I priors
    '''
    xlines = {'FUV': (13., 'solid')}
    ylines = {'RGS': (15.5, 'solid'),\
              'X-IFU': (15.0, 'dashed')}
    
     # table 2 log No6, Delta log No6
    o6meas_fuv_zo6 = (13.76, 0.08)
    # table 3
    o6meas_cie_zo6 = (13.9, 0.2)
    o6meas_cie_zo7 = (13.9, 0.2)
    o7meas_cie_zo6 = (16.4, 0.2)
    o7meas_cie_zo7 = (16.4, 0.2)
    o7meas_slab_zo6 = (16.59, -0.28, 0.24)
    o7meas_slab_zo7 = (16.62, -0.28, 0.25)
    o8meas_cie_zo6 = (16.0, 0.2)
    o8meas_cie_zo7 = (16.0, 0.2)
    o8meas_slab_zo6 = (15.85, -0.65, 0.32)
    o8meas_slab_zo7 = (15.82, -0.78, 0.49)
    
    o6meas_tot = (14.14, 0.22)
    
    T_cie_zo6 = (1.7e6, 0.2e6)
    T_cie_zo7 = (1.7e6, 0.2e6)
    
    logTcuts = [5.7, 5.9, 6.0]
    
    measvals = {'tot': {'o6': o6meas_tot},\
                'fuv': {'zo6': {'o6': o6meas_fuv_zo6}},\
                'cie': {'zo6': {'o6': o6meas_cie_zo6,\
                                'o7': o7meas_cie_zo6,\
                                'o8': o8meas_cie_zo6,\
                                'T':  T_cie_zo6},\
                        'zo7': {'o6': o6meas_cie_zo7,\
                                'o7': o7meas_cie_zo7,\
                                'o8': o8meas_cie_zo7,\
                                'T':  T_cie_zo7},\
                        },\
                 'slab': {'zo6': {'o7': o7meas_slab_zo6,\
                                  'o8': o8meas_slab_zo6,\
                                 },\
                         'zo7':  {'o7': o7meas_slab_zo7,\
                                  'o8': o8meas_slab_zo7,\
                                 },\
                          },\
                }
    
    
    histf = '/net/luttero/data2/proc/hist_coldens_o6-o7-hneutralssh_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature-NO6.npz'
    ion1 = 'o6'
    ion2 = 'o7'
    if plotnum == 1:
        imgn = 'coldens_o6_vs_o7'
    elif plotnum == 2:
        imgn = 'coldens_o6_vs_o7_Tsplit-%s'%(ionT_split)
    elif plotnum == 3:
        imgn = 'coldens_o6_vs_o7_NHIsplit-%s'%(nh1_split)
    
    imgname = '/home/wijers/Documents/papers/jussi_paper2_Ton180/histogram_L0100N1504_27_ptAb_C2Sm_T4EOS_6.25slice_zcen-all_z-projection_%s.eps'%(imgn)
    
    fontsize = 12
    xlabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion1, mathmode=True))
    ylabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion2, mathmode=True))
    clabel = r'$\log$ relative fraction of absorbers'
    cmap = truncate_colormap(mpl.cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7)
    
    percentiles = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
    linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    color_lower = 'C0'
    color_upper = 'C1'
    color_all = 'C4'
    
    with np.load(histf) as fi:
        #print(fi.keys())
        #print(fi['dimension'])
        #print(fi['edges'])
        
        dimension = fi['dimension']
        ax_no6 = np.where(dimension == 'N%s'%(string.capwords(ion1)))[0][0]
        ax_no7 = np.where(dimension == 'N%s'%(string.capwords(ion2)))[0][0]
        ax_nh1 = np.where(dimension == 'NH1')[0][0]
        ax_T = np.where(dimension == 'Temperature_w_N%s'%(string.capwords(ion1)))[0][0]
        numax = len(dimension)
        
        #sumaxes = (ax_nh1, ax_T) # autoset later
        if plotnum == 1:
            splitax = None
            splitval = None
            splitname = None
        elif plotnum == 2:
            splitax = ax_T
            splitval = ionT_split
            splitname = r'\log \, \mathrm{T}(\mathrm{%s})'%(ild.getnicename(ion1, mathmode=True))
        elif plotnum == 3:
            splitax = ax_nh1
            splitval = nh1_split
            splitname = r'\log \, \mathrm{N}(\mathrm{%s})'%(ild.getnicename('hneutralssh', mathmode=True))
        
        edges_o6 = fi['edges'][ax_no6]
        edges_o7 = fi['edges'][ax_no7]
        
        sumaxes = set(np.arange(numax))
        sumaxes -= {ax_no6, ax_no7}
        sumaxes = sorted(list(sumaxes))
        sumaxes = tuple(sumaxes)
        
        hist_base = fi['bins']
        hist = np.sum(hist_base, axis=sumaxes)
            
        if splitax is not None:
            edges_split = fi['edges'][splitax]
            
            cutind = np.argmin(np.abs(edges_split - splitval))
            cutval = edges_split[cutind]
            lowersel = slice(None, cutind, None)
            uppersel = slice(cutind, None, None)
            #lowersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} < %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
            #uppersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} > %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
            
            lowersel_all = [slice(None, None, None)] * numax
            lowersel_all[splitax] = lowersel
            lowersel_all = tuple(lowersel_all)
            
            uppersel_all = [slice(None, None, None)] * numax
            uppersel_all[splitax] = uppersel
            uppersel_all = tuple(uppersel_all)
            
            hist_lower = np.sum(hist_base[lowersel_all], axis=sumaxes)
            hist_upper = np.sum(hist_base[uppersel_all], axis=sumaxes)
            
            
    
    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[5., 1.])
    ax = fig.add_subplot(grid[0])
    cax  = fig.add_subplot(grid[1])
    
    # get percentiles from true histogram
    ax_histsum = 1 if ax_no7 > ax_no6 else 0
    plotlim = 20
    percentiles_all = percentiles_from_histogram(hist, edges_o7, axis=ax_histsum, percentiles=percentiles)
    cens_o6 = edges_o6[:-1] + 0.5 * np.diff(edges_o6)
    plotwhere = np.sum(hist, axis=ax_histsum) >= plotlim
    
    if splitax is not None:
        percentiles_lower = percentiles_from_histogram(hist_lower, edges_o7, axis=ax_histsum, percentiles=percentiles)
        percentiles_upper = percentiles_from_histogram(hist_upper, edges_o7, axis=ax_histsum, percentiles=percentiles)
        
        plotwhere_lower = np.sum(hist_lower, axis=ax_histsum) >= plotlim
        plotwhere_upper = np.sum(hist_upper, axis=ax_histsum) >= plotlim
    
    # switch to density histogram
    if ax_no6 < ax_no7:
        divby = np.diff(edges_o6)[:, np.newaxis] * np.diff(edges_o7)[np.newaxis, :]
    else:
        divby = np.diff(edges_o6)[np.newaxis, :] * np.diff(edges_o7)[:, np.newaxis]
    #hist_lowerT /= divby
    #hist_upperT /= divby
    hist_toplot = hist / divby
    
    #vmax = np.log10(max(np.max(hist_upperT), np.max(hist_lowerT)))
    xlim = (12.1, 16.5)
    ylim = (13.1, 17.5)
    vmax = np.log10(np.max(hist_toplot[edges_o6[1:] > xlim[0], edges_o7[1:] > ylim[0]]))
    vmin = vmax - 8.
    
    if ax_no6 > ax_no7:
        hist_lower = hist_lower.T
        hist_upper = hist_upper.T
    
    ax.tick_params(labelsize=fontsize - 1, which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    img = ax.pcolormesh(edges_o6, edges_o7, np.log10(hist.T), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    

    for pind in range(len(percentiles)):
        if splitax is not None:
            ax.plot(cens_o6[plotwhere_upper], percentiles_upper[pind][plotwhere_upper], color=color_upper, linewidth=2., linestyle=linestyles[pind])
            ax.plot(cens_o6[plotwhere_lower], percentiles_lower[pind][plotwhere_lower], color=color_lower, linewidth=2., linestyle=linestyles[pind])
        else:
            ax.plot(cens_o6[plotwhere], percentiles_all[pind][plotwhere], color=color_all, linewidth=2., linestyle=linestyles[pind])
    # add lines for detection limits and measurements
    if plotnum == 1:
        ancolor = 'C2'
        for label in xlines:
            ax.axvline(xlines[label][0], color=ancolor, linestyle=xlines[label][1], linewidth=1.5)
            ax.text((xlines[label][0] - xlim[0]) / (xlim[1] - xlim[0]) + 0.01, 0.8,  label,\
                    color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                    horizontalalignment='left', verticalalignment='center', rotation=90.)
        for label in ylines:
            ax.axhline(ylines[label][0], color=ancolor, linestyle=ylines[label][1], linewidth=1.5)
            ax.text(0.99, (ylines[label][0] - ylim[0]) / (ylim[1] - ylim[0]) + 0.01,  label,\
                    color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                    horizontalalignment='right', verticalalignment='bottom', rotation=0.)

    # FUV measurenment point
    fuvcolor='C3'
    fuvpoint = measvals['fuv']['zo6']['o6']
    ax.axvline(fuvpoint[0], 0.0, 0.7, color=fuvcolor)
    ax.errorbar(fuvpoint[0], ylim[0] + 0.35 * (ylim[1] - ylim[0]), xerr=fuvpoint[1:],\
                linewidth=1.5, color=fuvcolor, zorder=5)
    
    # X-ray measurement point and X-ray + UV
    xraycolor = 'black'
    xraypoint_x = measvals['tot'][ion1]
    xraypoint_y = measvals['cie']['zo7'][ion2]   
    ax.errorbar(xraypoint_x[0], xraypoint_y[0], xerr=xraypoint_x[1:], yerr=xraypoint_y[1:],\
                linewidth=2., color=xraycolor, zorder=5)
    # color bar
    plt.colorbar(img, cax=cax, extend='min')
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(10.)
    cax.set_ylabel(clabel, fontsize=fontsize)
    
    # legend
    perc_toplot = percentiles[: (len(percentiles) + 1) // 2]
    if splitax is not None:
        lcs = []
        line = [[(0, 0)]]
        for pind in range(len(perc_toplot)):    
            # set up the proxy artist
            subcols = [mpl.colors.to_rgba(color_lower, alpha=1.), mpl.colors.to_rgba(color_upper, alpha=1.)]
            subcols = np.array(subcols)
            #print(subcols)
            lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[pind], linewidth=2, colors=subcols)
            lcs.append(lc)
    else:
        lcs = [mlines.Line2D([], [], color=color_all, linestyle=linestyles[pind], label=None, linewidth=2.) for pind in range(len(perc_toplot))]
                   
    perclabels = ['%.0f %%'%(2. * (0.5 - perc) * 100.) if perc != 0.5 else 'median' for perc in perc_toplot]
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    lax = ax
    if splitax is not None:
        typelines = [mlines.Line2D([], [], color=color_lower, linestyle='solid', label=r'$ %s < %.1f$'%(splitname, cutval), linewidth=2.),\
                     mlines.Line2D([], [], color=color_upper, linestyle='solid', label=r'$ %s > %.1f$'%(splitname, cutval), linewidth=2.)]
        labels = [r'$ %s < %.1f$'%(splitname, cutval), r'$ %s > %.1f$'%(splitname, cutval)]
    
        leg1 = lax.legend(typelines, labels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)
        leg2 = lax.legend(lcs, perclabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.15), frameon=False)
        lax.add_artist(leg1)
        lax.add_artist(leg2)
    else:
        lax.legend(lcs, perclabels, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)
    #lax.axis('off')
   
    #lax = axes[1]
    #lax.legend(typelines, labels, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.02, -0.02), frameon=False, handlelength=1.5)
    
    plt.savefig(imgname, format='eps', bbox_inches='tight')
    

def plot_coldenscorr_Tion_v4(ionT='o6', ion1='o6', ion2='o7', Tlim=5.7,\
                             nsig_range=1.):
    '''
    v4: like v2, with limits, but different O VI range indicator and smaller 
        ackground histogram range
    '''
    
    xlines = {'FUV': (13., 'solid')}
    ylines = {'RGS': (15.5, 'solid'),\
              'X-IFU': (15.0, 'dotted')}
    arrowpos = {'FUV': 0.74, 'RGS': 0.87, 'X-IFU': 0.84}
    
     # table 2 log No6, Delta log No6
    o6meas_fuv_zo6 = (13.76, 0.08)
    # table 3
    o6meas_cie_zo6 = (13.8, 0.2)
    o6meas_cie_zo7 = (13.9, 0.2)
    o7meas_cie_zo6 = (16.4, 0.2)
    o7meas_cie_zo7 = (16.4, 0.2)
    o7meas_slab_zo6 = (16.52, -0.28, 0.25)
    o7meas_slab_zo7 = (16.69, -0.39, 0.37)
    o8meas_cie_zo6 = (15.9, 0.2)
    o8meas_cie_zo7 = (16.0, 0.2)
    o8meas_slab_zo6 = (15.7, -1.1, 0.4)
    o8meas_slab_zo7 = (15.7, -1.5, 0.4)
    
    T_cie_zo6 = (1.7e6, 0.2e6)
    T_cie_zo7 = (1.7e6, 0.2e6)
    
    logTcuts = [5.7, 5.9, 6.0]
    
    measvals = {'fuv': {'zo6': {'o6': o6meas_fuv_zo6}},\
                'cie': {'zo6': {'o6': o6meas_cie_zo6,\
                                'o7': o7meas_cie_zo6,\
                                'o8': o8meas_cie_zo6,\
                                'T':  T_cie_zo6},\
                        'zo7': {'o6': o6meas_cie_zo7,\
                                'o7': o7meas_cie_zo7,\
                                'o8': o8meas_cie_zo7,\
                                'T':  T_cie_zo7},\
                        },\
                 'slab': {'zo6': {'o7': o7meas_slab_zo6,\
                                  'o8': o8meas_slab_zo6,\
                                 },\
                         'zo7':  {'o7': o7meas_slab_zo7,\
                                  'o8': o8meas_slab_zo7,\
                                 },\
                          },\
                }
    
    
    if {ion1, ion2} == {'o6', 'o8'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    if {ion1, ion2} == {'o6', 'o7'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o7_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    
    imgname = '/home/wijers/Documents/papers/jussi_paper2_Ton180/hist_coldens_%s-%s_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_splitby_T-%s-%.2f_v4.eps'%(ion1, ion2, ionT, Tlim)
    
    fontsize = 12
    xlabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion1, mathmode=True))
    ylabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion2, mathmode=True))
    clabel = r'$\log$ relative fraction of absorbers'
    cmap = truncate_colormap(mpl.cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7)
    
    percentiles = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
    linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    color_lowerT = 'C0'
    color_upperT = 'C1'
    
    with np.load(histf) as fi:
        #print(fi.keys())
        #print(fi['dimension'])
        #print(fi['edges'])
        
        dimension = fi['dimension']
        ax_no6 = np.where(dimension == 'N%s'%(string.capwords(ion1)))[0][0]
        ax_no8 = np.where(dimension == 'N%s'%(string.capwords(ion2)))[0][0]
        ax_T = np.where(dimension == 'Temperature_w_N%s'%(string.capwords(ionT)))[0][0]
        numax = len(dimension)
        
        edges_o6 = fi['edges'][ax_no6]
        edges_o8 = fi['edges'][ax_no8]
        edges_T  = fi['edges'][ax_T]
        
        cutind_T = np.argmin(np.abs(edges_T - Tlim))
        cutval_T = edges_T[cutind_T]
        lowersel_T = slice(None, cutind_T, None)
        uppersel_T = slice(cutind_T, None, None)
        #lowersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} < %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        #uppersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} > %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        
        lowersel_all = [slice(None, None, None)] * numax
        lowersel_all[ax_T] = lowersel_T
        lowersel_all = tuple(lowersel_all)
        
        uppersel_all = [slice(None, None, None)] * numax
        uppersel_all[ax_T] = uppersel_T
        uppersel_all = tuple(uppersel_all)
        
        sumaxes = set(np.arange(numax))
        sumaxes -= {ax_no6, ax_no8}
        sumaxes = sorted(list(sumaxes))
        sumaxes = tuple(sumaxes)
        
        hist_base = fi['bins']
        hist_lowerT = np.sum(hist_base[lowersel_all], axis=sumaxes)
        hist_upperT = np.sum(hist_base[uppersel_all], axis=sumaxes)
        hist = np.sum(hist_base, axis=sumaxes)
                    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[5., 1.])
    ax = fig.add_subplot(grid[0])
    cax  = fig.add_subplot(grid[1])
    
    # get percentiles from true histogram
    ax_histsum = 1 if ax_no8 > ax_no6 else 0
    percentiles_lowerT = percentiles_from_histogram(hist_lowerT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    percentiles_upperT = percentiles_from_histogram(hist_upperT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    cens_o6 = edges_o6[:-1] + 0.5 * np.diff(edges_o6)
    plotwhere_lowerT = np.sum(hist_lowerT, axis=ax_histsum) >= 20
    plotwhere_upperT = np.sum(hist_upperT, axis=ax_histsum) >= 20
    
    # switch to density histogram
    if ax_no6 < ax_no8:
        divby = np.diff(edges_o6)[:, np.newaxis] * np.diff(edges_o8)[np.newaxis, :]
    else:
        divby = np.diff(edges_o6)[np.newaxis, :] * np.diff(edges_o8)[:, np.newaxis]
    #hist_lowerT /= divby
    #hist_upperT /= divby
    hist_toplot = hist / divby
    
    #vmax = np.log10(max(np.max(hist_upperT), np.max(hist_lowerT)))
    xlim = (12.1, 16.5)
    ylim = (13.1, 17.5)
    vmax = np.log10(np.max(hist_toplot[edges_o6[1:] > xlim[0], edges_o8[1:] > ylim[0]]))
    vmin = vmax - 7.
    
    if ax_no6 > ax_no8:
        hist_lowerT = hist_lowerT.T
        hist_upperT = hist_upperT.T
    
    ax.tick_params(labelsize=fontsize - 1, which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    img = ax.pcolormesh(edges_o6, edges_o8, np.log10(hist_lowerT.T), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    
    for pind in range(len(percentiles)):
        ax.plot(cens_o6[plotwhere_upperT], percentiles_upperT[pind][plotwhere_upperT], color=color_upperT, linewidth=2., linestyle=linestyles[pind])
        ax.plot(cens_o6[plotwhere_lowerT], percentiles_lowerT[pind][plotwhere_lowerT], color=color_lowerT, linewidth=2., linestyle=linestyles[pind])
    
    # add lines for detection limits and measurements
    ancolor = 'C2'
    arrowlen = 0.05
    arrowwidth = 0.002
    for label in xlines:
        ax.axvline(xlines[label][0], color=ancolor, linestyle=xlines[label][1], linewidth=1.5)
        relpos = (xlines[label][0] - xlim[0]) / (xlim[1] - xlim[0])
        ax.text(relpos + 0.01, 0.8,  label,\
                color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='left', verticalalignment='center', rotation=90.)
        ax.arrow(relpos, arrowpos[label], arrowlen, 0.,\
                 linestyle='solid', linewidth=1.5, color=ancolor,\
                 transform=ax.transAxes, width=arrowwidth)
    for label in ylines:
        ax.axhline(ylines[label][0], color=ancolor, linestyle=ylines[label][1], linewidth=1.5)
        relpos = (ylines[label][0] - ylim[0]) / (ylim[1] - ylim[0])
        ax.text(0.99, relpos + 0.01,  label,\
                color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='right', verticalalignment='bottom', rotation=0.)
        ax.arrow(arrowpos[label], relpos, 0., arrowlen,\
                 linestyle='solid', linewidth=1.5, color=ancolor,\
                 transform=ax.transAxes, width=arrowwidth)
    # FUV
    fuvcolor='navy'
    fuvpoint = measvals['fuv']['zo6']['o6']
    #ax.axvline(fuvpoint[0], 0.0, 0.7, color=fuvcolor)
    #ax.errorbar(fuvpoint[0], ylim[0] + 0.35 * (ylim[1] - ylim[0]), xerr=fuvpoint[1:],\
    #            linewidth=1.5, color=fuvcolor, zorder=5)
    ax.axvspan(fuvpoint[0] - fuvpoint[1], fuvpoint[0] + fuvpoint[1],\
               alpha=0.5, edgecolor=fuvcolor, facecolor='none', hatch='-')
    
    # X-ray
    xraycolor = 'red'
    xraypoint_x = (np.log10(10**measvals['cie']['zo7'][ion1][0] +\
                            10**measvals['fuv']['zo6'][ion1][0]),\
                   np.sqrt(measvals['cie']['zo7'][ion1][1]**2 + \
                           measvals['fuv']['zo6'][ion1][1]**2))
    xraypoint_y = measvals['cie']['zo7'][ion2]   
    ax.errorbar(xraypoint_x[0], xraypoint_y[0],\
                xerr=xraypoint_x[1:], yerr=xraypoint_y[1:],\
                linewidth=2., color=xraycolor, zorder=5)
    # color bar
    plt.colorbar(img, cax=cax, extend='min')
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(10.)
    cax.set_ylabel(clabel, fontsize=fontsize)
    
    # legend
    perc_toplot = percentiles[: (len(percentiles) + 1) // 2]
    lcs = []
    line = [[(0, 0)]]
    for pind in range(len(perc_toplot)):
        # set up the proxy artist
        subcols = [mpl.colors.to_rgba(color_lowerT, alpha=1.), mpl.colors.to_rgba(color_upperT, alpha=1.)]
        subcols = np.array(subcols)
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[pind], linewidth=2, colors=subcols)
        lcs.append(lc)
    perclabels = ['%.0f %%'%(2. * (0.5 - perc) * 100.) if perc != 0.5 else 'median' for perc in perc_toplot]
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    typelines = [mlines.Line2D([], [], color=color_lowerT, linestyle='solid', label=r'$\log\, \mathrm{T} < %.1f$'%(cutval_T), linewidth=2.),\
                 mlines.Line2D([], [], color=color_upperT, linestyle='solid', label=r'$\log\, \mathrm{T} < %.1f$'%(cutval_T), linewidth=2.)]
    labels = [r'$\log\, \mathrm{T}(\mathrm{%s}) < %.1f$'%(ild.getnicename(ion1, mathmode=True), cutval_T), r'$\log\, \mathrm{T}(\mathrm{%s}) > %.1f$'%(ild.getnicename(ion1, mathmode=True), cutval_T)]
    lax = ax
    leg1 = lax.legend(typelines, labels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.0), frameon=False)
    leg2 = lax.legend(lcs, perclabels, handler_map={type(lc): HandlerDashedLines()}, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.0, 0.15), frameon=False)
    lax.add_artist(leg1)
    lax.add_artist(leg2)
    #lax.axis('off')
   
    #lax = axes[1]
    #lax.legend(typelines, labels, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.02, -0.02), frameon=False, handlelength=1.5)
    
    plt.savefig(imgname, format='eps', bbox_inches='tight')
    
## adjusted after referee report revisions
# changed Tlim to 3 sigma upper limit on T_O VI (from b_O VI)
# modified the O VI columns and uncertainties to the sums of the two componenents
def plot_coldenscorr_Tion_v5(ionT='o6', ion1='o6', ion2='o7', Tlim=6.1,\
                             nsig_range=1.):
    '''
    '''
    
    xlines = {'FUV': (13., 'solid')}
    ylines = {'RGS': (15.5, 'solid'),\
              'X-IFU': (15.0, 'dotted')}
    arrowpos = {'FUV': 0.74, 'RGS': 0.87, 'X-IFU': 0.84}
    
    # table 2 log No6, Delta log No6
    _o6_c1 = (13.46, 0.16) 
    _o6_c2 = (13.68, 0.10)
    o6meas_fuv_zo6 = (np.log10(10**_o6_c1[0] + 10**_o6_c2[0]), 
                      1./ (10**_o6_c1[0] + 10**_o6_c2[0]) *\
                      np.sqrt(10**(2. * _o6_c1[0]) * _o6_c1[1]**2 +\
                              10**(2. * _o6_c2[0]) * _o6_c2[1]**2))
    # table 3
    o6meas_cie_zo6 = (13.8, 0.2)
    o6meas_cie_zo7 = (13.9, 0.2)
    o7meas_cie_zo6 = (16.4, 0.2)
    o7meas_cie_zo7 = (16.4, 0.2)
    o7meas_slab_zo6 = (16.52, -0.28, 0.25)
    o7meas_slab_zo7 = (16.69, -0.39, 0.37)
    o8meas_cie_zo6 = (15.9, 0.2)
    o8meas_cie_zo7 = (16.0, 0.2)
    o8meas_slab_zo6 = (15.7, -1.1, 0.4)
    o8meas_slab_zo7 = (15.7, -1.5, 0.4)
    
    T_cie_zo6 = (1.7e6, 0.2e6)
    T_cie_zo7 = (1.7e6, 0.2e6)
    
    logTcuts = [5.7, 5.9, 6.0]
    
    measvals = {'fuv': {'zo6': {'o6': o6meas_fuv_zo6}},\
                'cie': {'zo6': {'o6': o6meas_cie_zo6,\
                                'o7': o7meas_cie_zo6,\
                                'o8': o8meas_cie_zo6,\
                                'T':  T_cie_zo6},\
                        'zo7': {'o6': o6meas_cie_zo7,\
                                'o7': o7meas_cie_zo7,\
                                'o8': o8meas_cie_zo7,\
                                'T':  T_cie_zo7},\
                        },\
                 'slab': {'zo6': {'o7': o7meas_slab_zo6,\
                                  'o8': o8meas_slab_zo6,\
                                 },\
                         'zo7':  {'o7': o7meas_slab_zo7,\
                                  'o8': o8meas_slab_zo7,\
                                 },\
                          },\
                }
    cset = tc.tol_cset('vibrant')
    
    if {ion1, ion2} == {'o6', 'o8'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    if {ion1, ion2} == {'o6', 'o7'}:
        histf = '/net/luttero/data2/proc/hist_coldens_o6-o7_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    
    imgname = '/home/wijers/Documents/papers/jussi_paper2_Ton180/hist_coldens_'+\
              '{ion1}-{ion2}'+\
              '_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_'+\
              'z-projection_T4EOS_splitby_T-{ionT}-{Tlim:.2f}_v5.eps'
    imgname=imgname.format(ion1=ion1, ion2=ion2, ionT=ionT, Tlim=Tlim)
    
    fontsize = 12
    xlabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion1, 
                                                                mathmode=True))
    ylabel = r'$\log \, \mathrm{N(%s)} \; [\mathrm{cm}^{-2}]$'%(ild.getnicename(ion2, 
                                                                mathmode=True))
    clabel = r'$\log$ relative fraction of absorbers'
    cmap = truncate_colormap(mpl.cm.get_cmap('gist_yarg'), minval=0.0, maxval=0.7)
    
    percentiles = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
    linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    color_lowerT = cset.blue
    color_upperT = cset.orange
    
    with np.load(histf) as fi:
        #print(fi.keys())
        #print(fi['dimension'])
        #print(fi['edges'])
        
        dimension = np.array([_.decode() for _ in fi['dimension']])
        ax_no6 = np.where(dimension == 'N%s'%(string.capwords(ion1)))[0][0]
        ax_no8 = np.where(dimension == 'N%s'%(string.capwords(ion2)))[0][0]
        ax_T = np.where(dimension == 'Temperature_w_N%s'%(string.capwords(ionT)))[0][0]
        numax = len(dimension)
        
        edges_o6 = fi['edges'][ax_no6]
        edges_o8 = fi['edges'][ax_no8]
        edges_T  = fi['edges'][ax_T]
        
        cutind_T = np.argmin(np.abs(edges_T - Tlim))
        cutval_T = edges_T[cutind_T]
        lowersel_T = slice(None, cutind_T, None)
        uppersel_T = slice(cutind_T, None, None)
        #lowersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} < %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        #uppersel_T_label = r'$\log_{10} \, \mathrm{T(%s)} \, / \, \mathrm{K} > %.1f$'%(ild.getnicename(ionT, mathmode=True), cutval_T)
        
        lowersel_all = [slice(None, None, None)] * numax
        lowersel_all[ax_T] = lowersel_T
        lowersel_all = tuple(lowersel_all)
        
        uppersel_all = [slice(None, None, None)] * numax
        uppersel_all[ax_T] = uppersel_T
        uppersel_all = tuple(uppersel_all)
        
        sumaxes = set(np.arange(numax))
        sumaxes -= {ax_no6, ax_no8}
        sumaxes = sorted(list(sumaxes))
        sumaxes = tuple(sumaxes)
        
        hist_base = fi['bins']
        hist_lowerT = np.sum(hist_base[lowersel_all], axis=sumaxes)
        hist_upperT = np.sum(hist_base[uppersel_all], axis=sumaxes)
        hist = np.sum(hist_base, axis=sumaxes)
                    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(1, 2, hspace=0.0, wspace=0.1, width_ratios=[5., 1.])
    ax = fig.add_subplot(grid[0])
    cax  = fig.add_subplot(grid[1])
    
    # get percentiles from true histogram
    ax_histsum = 1 if ax_no8 > ax_no6 else 0
    percentiles_lowerT = percentiles_from_histogram(hist_lowerT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    percentiles_upperT = percentiles_from_histogram(hist_upperT, edges_o8, axis=ax_histsum, percentiles=percentiles)
    cens_o6 = edges_o6[:-1] + 0.5 * np.diff(edges_o6)
    plotwhere_lowerT = np.sum(hist_lowerT, axis=ax_histsum) >= 20
    plotwhere_upperT = np.sum(hist_upperT, axis=ax_histsum) >= 20
    
    # switch to density histogram
    if ax_no6 < ax_no8:
        divby = np.diff(edges_o6)[:, np.newaxis] * np.diff(edges_o8)[np.newaxis, :]
    else:
        divby = np.diff(edges_o6)[np.newaxis, :] * np.diff(edges_o8)[:, np.newaxis]
    #hist_lowerT /= divby
    #hist_upperT /= divby
    hist_toplot = hist / divby
    
    #vmax = np.log10(max(np.max(hist_upperT), np.max(hist_lowerT)))
    xlim = (12.1, 16.5)
    ylim = (13.1, 17.5)
    vmax = np.log10(np.max(hist_toplot[edges_o6[1:] > xlim[0], edges_o8[1:] > ylim[0]]))
    vmin = vmax - 7.
    
    if ax_no6 > ax_no8:
        hist_lowerT = hist_lowerT.T
        hist_upperT = hist_upperT.T
    
    ax.tick_params(labelsize=fontsize - 1, which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    img = ax.pcolormesh(edges_o6, edges_o8, np.log10(hist_lowerT.T), cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    
    for pind in range(len(percentiles)):
        ax.plot(cens_o6[plotwhere_upperT], percentiles_upperT[pind][plotwhere_upperT], color=color_upperT, linewidth=2., linestyle=linestyles[pind])
        ax.plot(cens_o6[plotwhere_lowerT], percentiles_lowerT[pind][plotwhere_lowerT], color=color_lowerT, linewidth=2., linestyle=linestyles[pind])
    
    # add lines for detection limits and measurements
    ancolor = cset.teal
    arrowlen = 0.05
    arrowwidth = 0.002
    for label in xlines:
        ax.axvline(xlines[label][0], color=ancolor, linestyle=xlines[label][1], linewidth=1.5)
        relpos = (xlines[label][0] - xlim[0]) / (xlim[1] - xlim[0])
        ax.text(relpos + 0.01, 0.8,  label,\
                color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='left', verticalalignment='center', rotation=90.)
        ax.arrow(relpos, arrowpos[label], arrowlen, 0.,\
                 linestyle='solid', linewidth=1.5, color=ancolor,\
                 transform=ax.transAxes, width=arrowwidth)
    for label in ylines:
        ax.axhline(ylines[label][0], color=ancolor, linestyle=ylines[label][1], linewidth=1.5)
        relpos = (ylines[label][0] - ylim[0]) / (ylim[1] - ylim[0])
        ax.text(0.99, relpos + 0.01,  label,\
                color=ancolor, fontsize=fontsize, transform=ax.transAxes,\
                horizontalalignment='right', verticalalignment='bottom', rotation=0.)
        ax.arrow(arrowpos[label], relpos, 0., arrowlen,\
                 linestyle='solid', linewidth=1.5, color=ancolor,\
                 transform=ax.transAxes, width=arrowwidth)
    # FUV
    fuvcolor= cset.magenta #'navy'
    fuvpoint = measvals['fuv']['zo6']['o6']
    #ax.axvline(fuvpoint[0], 0.0, 0.7, color=fuvcolor)
    #ax.errorbar(fuvpoint[0], ylim[0] + 0.35 * (ylim[1] - ylim[0]), xerr=fuvpoint[1:],\
    #            linewidth=1.5, color=fuvcolor, zorder=5)
    ax.axvspan(fuvpoint[0] - fuvpoint[1], fuvpoint[0] + fuvpoint[1],\
               alpha=0.5, edgecolor=fuvcolor, facecolor='none', hatch='-')
    
    # X-ray
    xraycolor = cset.red #'red'
    xraypoint_x = (np.log10(10**measvals['cie']['zo7'][ion1][0] +\
                            10**measvals['fuv']['zo6'][ion1][0]),\
                   np.sqrt(measvals['cie']['zo7'][ion1][1]**2 + \
                           measvals['fuv']['zo6'][ion1][1]**2))
    xraypoint_y = measvals['cie']['zo7'][ion2]   
    ax.errorbar(xraypoint_x[0], xraypoint_y[0],\
                xerr=xraypoint_x[1:], yerr=xraypoint_y[1:],\
                linewidth=2., color=xraycolor, zorder=5)
    # color bar
    plt.colorbar(img, cax=cax, extend='min')
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(10.)
    cax.set_ylabel(clabel, fontsize=fontsize)
    
    # legend
    perc_toplot = percentiles[: (len(percentiles) + 1) // 2]
    lcs = []
    line = [[(0, 0)]]
    for pind in range(len(perc_toplot)):
        # set up the proxy artist
        subcols = [mpl.colors.to_rgba(color_lowerT, alpha=1.), 
                   mpl.colors.to_rgba(color_upperT, alpha=1.)]
        subcols = np.array(subcols)
        #print(subcols)
        lc = mcol.LineCollection(line * len(subcols), linestyle=linestyles[pind], 
                                 linewidth=2, colors=subcols)
        lcs.append(lc)
    perclabels = ['%.0f %%'%(2. * (0.5 - perc) * 100.) if perc != 0.5 else \
                  'median' for perc in perc_toplot]
    # create the legend
    #lax.legend(lcs, [legendnames_techvars[var] for var in techvars], 
    #           handler_map={type(lc): HandlerDashedLines()}) #handlelength=2.5, 
    #           handleheight=3
    #handles_ax1, labels_ax1 = axes[0].get_legend_handles_labels()
    typelines = [mlines.Line2D([], [], color=color_lowerT, linestyle='solid', 
                 label=r'$\log\, \mathrm{T} < %.1f$'%(cutval_T), linewidth=2.),
                 mlines.Line2D([], [], color=color_upperT, linestyle='solid', 
                 label=r'$\log\, \mathrm{T} < %.1f$'%(cutval_T), linewidth=2.)]
    labels = [r'$\log\, \mathrm{T}(\mathrm{%s}) < %.1f$'%(ild.getnicename(ion1, 
                                                                          mathmode=True),
                                                          cutval_T), 
              r'$\log\, \mathrm{T}(\mathrm{%s}) > %.1f$'%(ild.getnicename(ion1, 
                                                                          mathmode=True), 
                                                                          cutval_T)]
    lax = ax
    leg1 = lax.legend(typelines, labels, handler_map={type(lc): HandlerDashedLines()}, 
                      fontsize=fontsize, ncol=1, loc='lower right', 
                      bbox_to_anchor=(1.0, 0.0), frameon=False)
    leg2 = lax.legend(lcs, perclabels, handler_map={type(lc): HandlerDashedLines()}, 
                      fontsize=fontsize, ncol=1, loc='lower right', 
                      bbox_to_anchor=(1.0, 0.15), frameon=False)
    lax.add_artist(leg1)
    lax.add_artist(leg2)
    #lax.axis('off')
   
    #lax = axes[1]
    #lax.legend(typelines, labels, fontsize=fontsize, ncol=1, loc='lower right', bbox_to_anchor=(1.02, -0.02), frameon=False, handlelength=1.5)
    
    plt.savefig(imgname, format='eps', bbox_inches='tight')
