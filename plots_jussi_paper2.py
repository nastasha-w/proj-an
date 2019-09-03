#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:26:13 2019

@author: wijers
"""

import numpy as np
import ion_line_data as ild

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
    
    
def plot_coldenscorr_Tion(ionT='o6'):
    
    histf = '/net/luttero/data2/proc/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz'
    Tlim = 5.8
    
    imgname = '/home/wijers/Documents/papers/jussi_paper2_Ton180/hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_splitby_T-%s.eps'%(ionT)
    
    fontsize = 12
    xlabel = r'$\log{10} \, \mathrm{N(O\,VI)} \; [\mathrm{cm}^{-2}]$'
    ylabel = r'$\log{10} \, \mathrm{N(O\,VIII)} \; [\mathrm{cm}^{-2}]$'
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
        ax_no6 = np.where(dimension == 'NO6')[0][0]
        ax_no8 = np.where(dimension == 'NO8')[0][0]
        if ionT == 'o6':
            ax_T = np.where(dimension == 'Temperature_w_NO6')[0][0]
        elif ionT == 'o8':
            ax_T = np.where(dimension == 'Temperature_w_NO8')[0][0]
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