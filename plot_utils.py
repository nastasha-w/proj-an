#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:33:56 2019

@author: wijers

Contains a number of general utility functions for making plots
"""
import numpy as np
import matplotlib as mpl
import mpl_toolkits.axes_grid1 as axgrid

# defaults
fontsize = 12

### array operations and interpolation

def getminmax2d(bins, edges, axis=None, log=True, pixdens=False): 
    # axis = axis to sum over; None -> don't sum over any axes 
    # now works for histgrams of general dimensions
    if axis is None:
        imgtoplot = bins
    else:
        imgtoplot = np.sum(bins, axis=axis)
    if pixdens:
        if axis is None:
            naxis = range(len(edges))
        else:
            if not hasattr(axis, '__len__'):
                saxis = [axis]
            else:
                saxis = axis
            naxis = list(set(range(len(edges))) - set(saxis)) # axes not to sum over
        naxis.sort() 
        numdims = len(naxis)
        binsizes = [np.diff(edges[axisi]) for axisi in naxis] # if bins are log, the log sizes are used and the enclosed log density is minimised
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

def percentiles_from_histogram(histogram, edgesaxis, axis=-1, percentiles=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
    '''
    get percentiles from the histogram along axis
    edgesaxis are the bin edges along that same axis
    histograms can be weighted by something: this function just solves 
    cumulative distribution == percentiles
    '''
    cdists = np.cumsum(histogram, axis=axis, dtype=np.float) 
    sel = list((slice(None, None, None),) * len(histogram.shape))
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
    oldshape2 = list(histogram.shape)[axis + 1:]
    newlen1 = int(np.prod(oldshape1))
    newlen2 = int(np.prod(oldshape2))
    axlen = histogram.shape[axis]
    cdists = cdists.reshape((newlen1, axlen, newlen2))
    cdists = np.append(np.zeros((newlen1, 1, newlen2)), cdists, axis=1)
    cdists[:, -1, :] = 1. # should already be true, but avoids fp error issues

    leftarr  = cdists[np.newaxis, :, :, :] <= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    rightarr = cdists[np.newaxis, :, :, :] >= percentiles[:, np.newaxis, np.newaxis, np.newaxis]
    print(leftarr)
    print(rightarr)
    
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


def handleinfedges(hist, setmin=-100., setmax=100.):
    for ei in range(len(hist['edges'])):
        if hist['edges'][ei][0] == -np.inf:
            hist['edges'][ei][0] = setmin
        if hist['edges'][ei][-1] == np.inf:
            hist['edges'][ei][-1] = setmax

def handleinfedges_dct(edges, setmin=-100., setmax=100.):
    for ei in edges.keys():
        if edges[ei][0] == -np.inf:
            edges[ei][0] = setmin
        if edges[ei][-1] == np.inf:
            edges[ei][-1] = setmax

### small plot helpers
def setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False):
    ax.minorticks_on()
    ax.tick_params(labelsize=fontsize, direction='in', right=right, top=top, axis='both', which='both', color=color,\
                   labelleft=labelleft, labeltop=labeltop, labelbottom = labelbottom, labelright=labelright)



### functions to make actual plots
def add_colorbar(ax, img=None, vmin=None, vmax=None, cmap=None, clabel=None, newax=False, extend='neither', fontsize=fontsize, orientation='vertical'):
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

def add_2dplot(ax, bins, edges, toplotaxes, log=True, usepcolor=False, pixdens=False, shiftx=0., shifty=0., **kwargs):
    # hist3d can be a histogram of any number >=2 of dimensions
    # like in plot1d, get the number of axes from the length of the edges array
    # usepcolor: if edges arrays are not equally spaced, imshow will get the ticks wrong
    summedaxes = tuple(list( set(range(len(edges)))-set(toplotaxes) )) # get axes to sum over
    toplotaxes= list(toplotaxes)
    #toplotaxes.sort()
    axis1, axis2 = tuple(toplotaxes)
    # sum over non-plotted axes
    if len(summedaxes) == 0:
        imgtoplot = bins
    else:
        imgtoplot = np.sum(bins, axis=summedaxes)
    
    
    if pixdens:
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(edges[toplotaxes[0]]), np.diff(edges[toplotaxes[1]]) ] # if bins are log, the log sizes are used and the enclosed log density is minimised
        baseinds = list((np.newaxis,)*numdims)
        normmatrix = np.prod([(binsizes[ind])[tuple(baseinds[:ind] +\
                              [slice(None,None,None)] +\
                              baseinds[ind+1:])] for ind in range(numdims)])
        if axis1 > axis2:
            imgtoplot = imgtoplot.T
        imgtoplot /= normmatrix
        if axis1 > axis2:
            imgtoplot = imgtoplot.T
        del normmatrix
        
    if log:
        imgtoplot = np.log10(imgtoplot)
    # transpose plot if axes not in standard order; normally, need to use transposed array in image
    if axis1 < axis2:
        imgtoplot = imgtoplot.T
    if usepcolor:
        _kwargs = kwargs.copy()
        if 'rasterized' not in _kwargs:
            _kwargs['rasterized'] = True           
        img = ax.pcolormesh(edges[axis1] + shiftx, edges[axis2] + shifty, imgtoplot, **_kwargs)
    else:
        img = ax.imshow(imgtoplot, origin='lower', interpolation='nearest',\
                        extent=(edges[axis1][0] + shiftx, edges[axis1][-1] + shiftx,\
                                edges[axis2][0] + shifty, edges[axis2][-1] + shifty),\
                        **kwargs)
    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
    else:
        vmin = np.min(imgtoplot[np.isfinite(imgtoplot)])
    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
    else:
        vmax = np.max(imgtoplot[np.isfinite(imgtoplot)])
    return img, vmin, vmax

def add_2dhist_contours(ax, bins, edges, toplotaxes,\
                        mins=None, maxs=None, histlegend=True, fraclevels=True,\
                        levels=None, legend=True, dimlabels=None, legendlabel=None,\
                        legendlabel_pre=None, shiftx=0., shifty=0., dimshifts=None, **kwargs):
    '''
    colors, linestyles: through kwargs
    othersmin and othersmax should be indices along the corresponding histogram axes
    assumes xlim, ylim are already set as desired
    dimlabels can be used to override (long) dimension labels from hist
    '''
    # get axes to sum over; preserve order of other axes to match limits
    
        
    summedaxes = range(len(edges))
    summedaxes.remove(toplotaxes[0])
    summedaxes.remove(toplotaxes[1])
    
    #print('min/max per edge: %s'%str([(mins[i], maxs[i], len(edges[i])) for i in [0,1,2]]))
    #print('mins: %s, maxs: %s'%(mins, maxs))
    
    if dimlabels is None:
        dimlabels = [''] * len(edges)    
    if mins is None:
        mins= (None,)*len(edges)
    if maxs is None:
        maxs = (None,)*len(edges)
    if dimshifts is None:
        dimshifts = (0.,) * len(edges)
	
    # get the selection of min/maxs and apply the selection, put axes in the desired order
    sels = [slice(mins[i], maxs[i], None) for i in range(len(edges))]
    sels = tuple(sels)
    
    if len(summedaxes) > 0:
        binsum = np.sum(bins[sels], axis=tuple(summedaxes))
    else:
        binsum = bins[sels]
    if toplotaxes[0] > toplotaxes[1]:
        binsum = binsum.transpose()
    #print('min/max binsum: %.4e, %.4e'%(np.min(binsum),np.max(binsum)))
    
    binfrac = np.sum(binsum) / np.sum(bins) # fraction of total bins selected
    # add min < dimension_quantity < max in legend label
    if legendlabel is None:
        labelparts = [r'%.1f $<$ %s $<$ %.1f, '%(edges[i][mins[i]] + dimshifts[i], dimlabels[i], edges[i][maxs[i]] + dimshifts[i]) if (mins[i] is not None and maxs[i] is not None) else\
                      r'%.1f $<$ %s, '%(edges[i][mins[i]] + dimshifts[i], dimlabels[i])                  if (mins[i] is not None and maxs[i] is None)     else\
		              r'%s $<$ %.1f, '%(dimlabels[i], edges[i][maxs[i]] + dimshifts[i])                  if (mins[i] is None and maxs[i] is not None)     else\
		              '' for i in range(len(edges))] #no label if there is no selection on that dimension
        legendlabel = ''.join(labelparts)
        # add percentage of total histogram selected
        if legendlabel[-2:] == ', ':
            legendlabel = legendlabel[:-2] + ': '
        legendlabel += '%.1f%%'%(100.*binfrac) 

    if legendlabel_pre is not None:
        legendlabel = legendlabel_pre + legendlabel
    
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    if levels is None:
        if fraclevels:
            levels = [1., 0.9, 0.5] # enclosed fractions for each level (approximate)
        else:
	        levels = [1e-3,3e-2,0.1,0.5]

    if fraclevels: # assumes all levels are between 0 and 1
        binsum = binsum/np.sum(binsum) # redo normalisation for the smaller dataset
        #print('min/max binsum: %.4e, %.4e'%(np.min(binsum),np.max(binsum)))
        
        # for sorting, normialise bins by bin size: peak finding depends on density, should not favour larger bins
        numdims = 2 # 2 axes not already summed over 
        binsizes = [np.diff(edges[toplotaxes[0]]), np.diff(edges[toplotaxes[1]]) ] # if bins are log, the log sizes are used and the enclosed log density is minimised
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
	    #print binsumcopy, binsumcopy.shape
	    #print cumul
	    #print wherelist
	    #return levels, cumul, binsumcopy, wherelist
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
        #print('min/max bindens: %.4e, %f, min/max flat: %.4e, %f'%(np.min(bindens),np.max(bindens),np.min(bindensflat),np.max(bindensflat)))
        #print('binsum shape: %s, bindens shape: %s, normmatrix shape: %s,  x: %i, y: %i'%(str(binsum.shape),str(bindens.shape), str(normmatrix.shape), len(hist['edges'][toplotaxes[0]]), len(hist['edges'][toplotaxes[1]])))
        #print('wherelist: %s'%wherelist)
        #plt.subplot(2,1,1)
        #plt.pcolor(hist['edges'][toplotaxes[0]], hist['edges'][toplotaxes[1]], np.log10(binsum.T), vmin = np.min( np.log10(binsum.T)[np.isfinite(np.log10(binsum.T))]))
        #plt.colorbar()
        #plt.subplot(2,1,2)
        #plt.pcolor(hist['edges'][toplotaxes[0]], hist['edges'][toplotaxes[1]], np.log10(bindens.T), vmin = np.min( np.log10(bindens.T)[np.isfinite(np.log10(bindens.T))]))
        #plt.colorbar()
        #plt.show()
        
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
        if uselevels[0] ==uselevels[1]:
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
    centres0 = edges[toplotaxes[0]][:-1] + shiftx + 0.5 * np.diff(edges[toplotaxes[0]]) 
    centres1 = edges[toplotaxes[1]][:-1] + shifty + 0.5 * np.diff(edges[toplotaxes[1]])
    contours = ax.contour(centres0, centres1, pltarr.T, uselevels, **kwargs)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    # make a legend to avoid crowding plot region
    #for i in range(len(levels)):
    #    contours.collections[i].set_label('%.0e'%levels[i])
    # color only legend; get a solid line in the legend
    
    #ax.tick_params(labelsize=fontsize,axis='both')
    if 'solid' in linestyles:
        contours.collections[np.where(np.array(linestyles)=='solid')[0][0]].set_label(legendlabel)
    else: # just do the first one
        contours.collections[0].set_label(legendlabel)
    if histlegend:
        ax.legend(loc='lower right',title=r'$f_{\mathrm{O VII}}, f_H=0.752$')
        
