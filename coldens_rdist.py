# to calculate the radial distribution of column densities around some point

import numpy as np
#import loadnpz_and_plot as lnp
import h5py
import ctypes as ct
import sys
import os
import fnmatch
import gc

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
from matplotlib import colors as mcolors
import matplotlib.gridspec as gsp

import make_maps_opts_locs as ol
import makecddfs as mc
import cosmo_utils as cu
import selecthalos as sh
import ion_line_data as ild

pdir = ol.pdir
rdir = pdir + 'radprof/'
mdir = '/net/luttero/data2/imgs/CGM/misc_start/'

# gets the radial distribution for one object
# note: x,y are really just placeholders; shift along with the projection
# axis 0 -> x, axis 1 -> y  

#clabel = r'$ \log_{10}\, T \; [K]$' 
#r'$ \log_{10}\, \Sigma \; [\mathrm{g}\, \mathrm{cm}^{-2}]$' 
#r'$ \log_{10}\, N_{\mathrm{O\, VI}}\; [\mathrm{cm}^{-2}]$'
#r'$ \log_{10}\, N_{\mathrm{Si\, II}}\; [\mathrm{cm}^{-2}]$'
#'Mass-weighted temperature' 
#'Mass surface density' 
#'O VI column density'

# custom excpetion to throw is a dataset in not present; useful for try/except 
# find or calculate situation
class DatasetNotFound(Exception):
    pass

def getcenfills(base, closevals=None, searchdir=None, tolerance=1e-4):
    if searchdir is None:
        searchdir = ol.ndir
    files = fnmatch.filter( next(os.walk(searchdir))[2], base%('*'))
    files_parts = [fil.split('/')[-1] for fil in files]
    files_parts = [fil[:-4].split('_') for fil in files_parts]
    files_parts = [[part if 'cen' in part else None for part in fil] for fil in files_parts]
    scens = [part for fil in files_parts for part in fil]
    scens = list(set(scens))
    if None in scens:
        scens.remove(None)
    scens = np.array([cen[4:] for cen in scens])
    
    if closevals is not None:
        print(closevals)
        print(scens)
        ismatch = [np.min(np.abs(float(cen) - np.array(closevals).astype(np.float64))) <= tolerance for cen in scens]
        scens = scens[np.array(ismatch)]
    print('Found scens:')
    print(scens)
    return scens

def rdist(quantity, L_x, npix_x, rmin, rmax, rscale, centre,
          npix_y=None, plot=None, clabel=None, title=None, label=''):
    
    if npix_y == None:
        npix_y = npix_x
    # square pixels assumed
    length_per_pixel = np.float(L_x)/npix_x

    pix_xmin = int(np.floor((centre[0] - rmax) / length_per_pixel))
    pix_ymin = int(np.floor((centre[1] - rmax) / length_per_pixel))
    pix_xmax = int(np.ceil((centre[0] + rmax) / length_per_pixel))
    pix_ymax = int(np.ceil((centre[1] + rmax) / length_per_pixel))
    
    pix_xmin_o = pix_xmin
    pix_xmax_o = pix_xmax
    pix_ymin_o = pix_ymin
    pix_ymax_o = pix_ymax
    
    c0 = centre[0]
    c1 = centre[1]
    uq = quantity
    
    # adjust image by rolling axes if the cluster overlaps a box boundry
    if pix_xmin < 0 or pix_xmax >= npix_x:
        uq = np.roll(uq,int(npix_x)/2,axis=0)
        if pix_xmin<0:
            pix_xmin += int(npix_x) / 2
            pix_xmax += int(npix_x) / 2
            c0 += int(int(npix_x) / 2) * length_per_pixel
        elif pix_xmax >=npix_x:
            pix_xmin += int(npix_x) / 2 - npix_x
            pix_xmax += int(int(npix_x)) / 2 - npix_x
            c0 += int(int(npix_x) / 2 - npix_x) * length_per_pixel
    if pix_ymin < 0 or pix_ymax >= npix_y:
        uq = np.roll(uq, int(int(npix_y) / 2), axis=1)
        if pix_ymin<0:
            pix_ymin += int(npix_y) / 2
            pix_ymax += int(npix_y) / 2
            c1 += int(int(npix_y) / 2) * length_per_pixel
        elif pix_ymax >=npix_y:
            pix_ymin += int(npix_y) / 2 - npix_y
            pix_ymax += int(npix_y) / 2 - npix_y
            c1 += int(int(npix_y) / 2 - npix_y) * length_per_pixel
    pixrange = [slice(pix_xmin, pix_xmax, None),
                slice(pix_ymin, pix_ymax, None)]
    
    slq = uq[pixrange]
    shape = slq.shape
    inds = np.indices(shape)
    # use distance to pixel centres
    dists2 = (length_per_pixel * (inds[0] + pix_xmin + 0.5) - c0)**2 + \
             (length_per_pixel * (inds[1] + pix_ymin + 0.5) - c1)**2 
    ann = np.logical_and(dists2 >= rmin**2, dists2 <= rmax**2)
    rs = np.ndarray.flatten(np.sqrt(dists2[ann]) / rscale)
    qs = np.ndarray.flatten(slq[ann])
    
    print(str(pixrange))

    if plot is not None:

        name = plot%('_zoom-x-%i-%i-y-%i-%i_%s'%(pix_xmin_o, pix_xmax_o,
                                                 pix_ymin_o, pix_ymax_o,
                                                 label))
        fontsize = 13
        colmap = 'viridis'
        xystarts = [pix_xmin_o * length_per_pixel, pix_ymin_o * length_per_pixel]
        size = (pix_xmax_o - pix_xmin_o ) * length_per_pixel
        Vmin = 10.5
        Vmax = 15.5
        
        circle = plt.Circle(centre, rscale, color='white', fill=False)   

        fig = plt.figure(figsize = (5.5, 5.))
        ax = plt.subplot(111)   
        ax.set_xlabel(r'X [cMpc]',fontsize=fontsize)
        ax.set_ylabel(r'Y [cMpc]',fontsize=fontsize)
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize)
        ax.patch.set_facecolor(cm.get_cmap(colmap)(0.)) # sets background color to lowest color map value 
        img = ax.imshow(slq.T,extent=(xystarts[0],xystarts[0]+size,xystarts[1],xystarts[1]+size),origin='lower', cmap=cm.get_cmap(colmap), vmin = Vmin, vmax=Vmax)
        #plt.title(title,fontsize=fontsize)
        ax.add_artist(circle)
        div = axgrid.make_axes_locatable(ax)
        cax = div.append_axes("right",size="5%",pad=0.1)
        cbar = plt.colorbar(img, cax=cax)
        cbar.solids.set_edgecolor("face")
        cbar.ax.set_ylabel(r'%s' % (clabel), fontsize=(fontsize+1))
        cbar.ax.tick_params(labelsize=fontsize)
        # save PDF figure
        fig.tight_layout()
        #fig.subplots_adjust(right=0.88)
        plt.savefig(name,format = 'pdf',dpi=1200)
        plt.close()
    return np.array([rs,qs])





# assumes rgarr[0] = r values, rqarr[1] = quantity values
# returns NaN for all percentiles if a bins is empty
def rqhist(rqarr, rbins, percentiles=None):
    bininds = np.digitize(rqarr[0], rbins)
    if percentiles is None: #use median and 1 sigma, 2 sigma values
        percentiles = np.array([2.275, 15.865, 50, 84.135, 97.725])
    qstats = [np.percentile((rqarr[1])[bininds==i], percentiles) if i in bininds\
              else np.ones(len(percentiles)) * np.NaN\
              for i in range(1, len(rbins))]
    return np.array(qstats).T  

def rqgeq(rqarr, rbins, values=np.array([13.,14.,15.])):
    bininds = np.digitize(rqarr[0],rbins)
    qbins = [rqarr[1][bininds==i] for i in range(1,len(rbins))]
    lens = [len(_bin) for _bin in qbins]
    qstats = [[np.sum(qbins[j]>val)/np.float(lens[j]) for val in values]
              for j in range(len(qbins))]
    return np.array(qstats).T

def rqhists(rqarrs, rbins, percentiles=None):
    return {key: rqhist(rqarrs[key], rbins, percentiles=percentiles) 
                 for key in rqarrs.keys()}
 
def rqgeqs(rqarrs, rbins, values=np.array([13.,14.,15.])):
    return {key: rqgeq(rqarrs[key],rbins,values) for key in rqarrs.keys()}
   
    
def plotrqhists(dist,bins,label,plot,ylabel=None,xlabel=None, medq=None):
    name = plot%('_Rdist-withmed_%s'%(label))
    fontsize = 16
        
    plt.plot(bins,dist[2], linewidth=2, color='blue')
    ax = plt.gca()
    ax.fill_between(bins, dist[1], dist[3], facecolor='blue', alpha=0.2)
    ax.fill_between(bins, dist[0], dist[4], facecolor='blue', alpha=0.2)
    
    if medq != None:
        percentiles = np.array([2.275,15.865,50,84.135,97.725])
        glp = np.percentile(medq,percentiles)
        ax.plot([bins[0],bins[-1]],[glp[2],glp[2]],linewidth=2,color='grey',linestyle='dashed')
        ax.fill_between([bins[0],bins[-1]],[glp[1],glp[1]],[glp[3],glp[3]],facecolor='grey',alpha=0.08)
        ax.fill_between([bins[0],bins[-1]],[glp[0],glp[0]],[glp[4],glp[4]],facecolor='grey',alpha=0.08)
            
    ax.xlim(0.,5.)
    ax.ylim(10.,15.)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    plt.savefig(name,format = 'pdf',dpi=1200)
    plt.close()
    return 0

# gets the radial distribution for a set of objects in one image
def rdists(quantity, L_x, npix_x,rmins,rmaxs,rscales,centres,labels,plot=None,clabel=None):
    return {labels[i]:  rdist(quantity,L_x,npix_x,rmins[i],rmaxs[i],rscales[i],(centres[0][i],centres[1][i]),plot=plot,clabel=clabel,label=labels[i]) for i in range(len(labels))}


def rplots(dists,rbins,labels,plot=None,ylabel=None,xlabel=None,medq=None):
    qstats  = {key: rqhists(dists[key],rbins,percentiles=None).T for key in dists.keys()}
    if plot != None:
        [plotrqhists(qstats[key], rbins[:-1] + np.diff(rbins)/2., key, plot=plot,ylabel=ylabel,xlabel=xlabel,medq=medq) for key in qstats.keys()]
    return qstats





def plotrqhistset(dists,rbins,label,plot,ylabel=None,xlabel=None, medq=None,colorby = None,colmap='jet',clabel=None,legendlabel = None):
    name = plot%('_Rdist-byhalo_%s'%(label))
    fontsize = 16
    keys = dists.keys()
    colors = ['purple','blue','cyan','green','gold','orange','red']
    bins = rbins[:-1]+ 0.5*np.diff(rbins)

    if colorby == None:
        if len(dists) > len(colors):
            print('Color list too short; please add more colors.')
            return None
    elif colorby == 'label' or 'loglabel':
        if colorby == 'label':
            colorby = [np.float(key) for key in keys]
        if colorby == 'loglabel':
            colorby = [np.log10(np.float(key)) for key in keys]
        minc = np.min(colorby)
        maxc = np.max(colorby)
        cby_normed = (np.array(colorby) - minc)/(maxc-minc)
        colors = cm.get_cmap(colmap)(cby_normed)
    else:
        minc = np.min(colorby)
        maxc = np.max(colorby)
        cby_normed = (np.array(colorby) - minc)/(maxc-minc)
        colors = cm.get_cmap(colmap)(cby_normed)

    if legendlabel == None:
        leglab = (None,)*len(dists)
    else:
        leglab = legendlabel
    
           
    plt.plot(bins,dists[keys[0]][0],linewidth=2,color=colors[0],label = leglab[0])
    ax = plt.gca()
    #ax.fill_between(bins,dists[dists.keys()[0]][1],dists[dists.keys()[0]][3],facecolor=colors[0],alpha=0.1)
    #ax.fill_between(bins,dists[dists.keys()[0]][0],dists[dists.keys()[0]][4],facecolor=colors[0],alpha=0.1)
    
    if len(dists) >1:
        for i in range(1, len(dists)):
            ax.plot(bins,dists[keys[i]][0],linewidth=2,color=colors[i],label = leglab[i])
            #ax.fill_between(bins,dists[dists.keys()[i]][1],dists[dists.keys()[i]][3],facecolor=colors[i],alpha=0.1)
            #ax.fill_between(bins,dists[dists.keys()[i]][0],dists[dists.keys()[i]][4],facecolor=colors[i],alpha=0.1)
    if medq != None:
        percentiles = np.array([2.275,15.865,50,84.135,97.725])
        glp = np.percentile(medq,percentiles)
        ax.plot([bins[0],bins[-1]],[glp[2],glp[2]],linewidth=2,color=(0.3,0.3,0.3),linestyle='dashed')
        ax.fill_between([bins[0],bins[-1]],[glp[1],glp[1]],[glp[3],glp[3]],facecolor=(0.3,0.3,0.3),alpha=0.2)
        #ax.fill_between([bins[0],bins[-1]],[glp[0],glp[0]],[glp[4],glp[4]],facecolor='grey',alpha=0.1)
    
    if colorby != None:
        cmap = mpl.cm.get_cmap(colmap)
        sm = mpl.cm.ScalarMappable(cmap=cmap,norm = mpl.colors.Normalize(vmin=minc,vmax=maxc))
        sm.set_array([minc,maxc])
        cbar = plt.colorbar(sm,ax=ax)
        #div = axgrid.make_axes_locatable(ax)
        #cax = div.append_axes("right",size="5%",pad=0.1)
        #cbar = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm = mpl.colors.Normalize(vmin=minc,vmax=maxc), orientation='vertical')
        cbar.set_label(clabel,fontsize=fontsize)
        
    plt.xlim(0.,5.)
    #plt.ylim(0.,0.00065) #plt.ylim(10.,14.5)
    ax.set_xlabel(xlabel,fontsize=fontsize)
    ax.set_ylabel(ylabel,fontsize=fontsize)
    if np.sum(legendlabel !=None) >0:
        plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(name,format = 'pdf',dpi=1200)
    plt.close()
    return 0



def rplotset(dists,rbins,label,plot=None,ylabel=None,xlabel=None,medq=None):
    qstats  = {key: rqhists(dists[key], rbins, percentiles=None) 
                    for key in dists.keys()}
    if plot != None:
        plotrqhistset(qstats, rbins[:-1] + np.diff(rbins) / 2.,
                      label, plot=plot, ylabel=ylabel, xlabel=xlabel, 
                      medq=medq)
    return qstats

def combdists(rqarr, selkeys, rbins, percentiles=None):
    # get bin indices from rqarr[0] for each cluster  
    bininds = [np.digitize(rqarr[key][0], rbins) for key in selkeys]
    # sort column densities in rqarr[1] by bin index
    totvals = [[list(rqarr[selkeys[j]][1][bininds[j]==i]) 
                for j in range(len(selkeys))] for i in range(1,len(rbins))]
    # collapse lists for each bin along cluster label dimension
    totvals = [np.array([item for sub in lis for item in sub]) for lis in totvals]
    
    if percentiles ==None: #use median and 1 sigma, 2 sigma values
        percentiles = np.array([2.275,15.865,50,84.135,97.725])
    qstats = [np.percentile(totvals[i], percentiles) 
              for i in range(len(rbins) - 1)]
    return np.array(qstats)

### low and high mass samples     
# lowmass = [key for key in np.array(o6f100_rds.keys())[ np.array(o6f100_rds.keys())< 1e14]]

### median of medians and pm 1 sigmas



def rdist_sl(base, szcens, L_x, npix_x, rmin, rmax, rscale, centre,\
             numsl=1, npix_y=None, plot=None, clabel=None, title=None,\
             label='', axis='z'):
    '''
    inputs:
    ---------------------------------------------------------------------------
    base:        (str incl. %s) string containing the file name (including 
                 directory) for the images to extract 
                 data from (data = image values)
    szcens:      (iterable of strings) values of centre coordinates along the 
                 axis of the data files; should complete the file names when 
                 substituted into base. 
                 (follow coldens/emission projection naming conventions)
    L_x:         (float) length of box along the zero coordinate (x if axis is 
                 z)
    npix_x:      (int) number of pixels along the zero coordinate (x if axis is
                 z)
    rmin:        (float, same units as L_x) minimum (2D) radius around the 
                 centre to take data from
    rmax:        (float, same units as L_x) maximum (2D) radius around the 
                 centre to take data from  
    rscale:      (float, same units as L_x) radius to scale radial coordinates 
                 to for output (i.e. units of output radii)
    centre:      (float * 3) centre around which to extract data, in EAGLE 
                 x,y,z coordinates
    axis:        ('x', 'y', or 'z') axis along which the images are projected 
                 (sets centre interpretation)
                 default: 'z'
    numsl:       (int) number of slices around the centre to use
                 default: 1
    npix_y:      (int) number of pixels along the 1 coordinate (pixels 
                 assumed to be square)
                 default: None -> set to npix_x
    plot:        (str incl. %s) name of plot to save extracted image region to,
                 including directory coordinate region saved automatically
                 labels assume lengths are in cMpc
                 default: None -> image not saved  
    label:       (str) tag to add to plot name
                 default: ''
    clabel:      (str) plot color bar label
                 default: None 
    title:       (str) plot title
                 default: None


    output:
    ---------------------------------------------------------------------------
    saved image of the selected region, if plot option set
    float array [rs,qs]: radii (rscale units) and quantity (input image units)
    '''
    
    if axis == 'z':
        c0 = centre[0]
        c1 = centre[1]
        c2 = centre[2]
    elif axis == 'x':
        c0 = centre[2]
        c1 = centre[0]
        c2 = centre[1]
    elif axis == 'y':
        c0 = centre[1]
        c1 = centre[2]
        c2 = centre[0]
    else:
        print('%s is an invalid axis option' %str(axis))
        return None

    ### find pixel range appropriate to selected area    
    if npix_y == None:
        npix_y = npix_x
    # square pixels assumed
    length_per_pixel = np.float(L_x)/npix_x

    pix_xmin = int(np.floor((centre[0] - rmax) / length_per_pixel))
    pix_ymin = int(np.floor((centre[1] - rmax) / length_per_pixel))
    pix_xmax = int(np.ceil((centre[0] + rmax) / length_per_pixel))
    pix_ymax = int(np.ceil((centre[1] + rmax) / length_per_pixel))
    
    # needed for edge adjustment cases later
    pix_xmin_o = pix_xmin
    pix_xmax_o = pix_xmax
    pix_ymin_o = pix_ymin
    pix_ymax_o = pix_ymax

    if pix_xmin < 0 or pix_xmax >= npix_x:
        if pix_xmin < 0:
            pix_xmin += int(npix_x) / 2
            pix_xmax += int(npix_x) / 2
            c0 += int(int(npix_x) / 2) * length_per_pixel
        elif pix_xmax >=npix_x:
            pix_xmin += int(npix_x) / 2 - npix_x
            pix_xmax += int(int(npix_x)) / 2 - npix_x
            c0 += int(int(npix_x)/2 - npix_x)*length_per_pixel
    if pix_ymin < 0 or pix_ymax >= npix_y:
        if pix_ymin<0:
            pix_ymin += int(npix_y) / 2
            pix_ymax += int(npix_y) / 2
            c1 += int(int(npix_y) / 2)*length_per_pixel
        elif pix_ymax >=npix_y:
            pix_ymin += int(npix_y) / 2 - npix_y
            pix_ymax += int(npix_y) / 2 - npix_y
            c1 += int(int(npix_y) / 2 - npix_y) * length_per_pixel
    pixrange = [slice(pix_xmin, pix_xmax, None),
                slice(pix_ymin, pix_ymax, None)]
    
    ### find centering appropriate to the slice and centre[2]; assumes periodicity in projected direction

    zcens = [np.float(cen) for cen in szcens]
    zcens = np.asarray(zcens)
    # for an even number of slices, the closet point will be the smaller of the two centre values
    zcens = zcens + (1 - numsl % 2) * 0.5 * (zcens[-1] - zcens[0]) \
                    / (len(zcens) - 1.)
    ceninds = np.argmin(abs(zcens - c2)) # returns the first index if multiple are the same  
    # odd number of slices: periodic boundary coincides with split point between first and last slice
    # even number of slices: point closet to index zero by argmin may be closer to the index -1 point 
    # when periodic conditions are factored in
    if ceninds == 0 and numsl % 2 == 0:
        if abs(c2 - 0.) < abs(c2 - zcens[0]):
            ceninds = len(zcens) -1
    ceninds = (ceninds - ((numsl - 1) // 2) + np.arange(numsl)) % len(zcens) # numsl/2 should be integer division 
    fills = np.array(szcens)[ceninds]

    
    ### load image and adjust by rolling axes if the cluster overlaps a box boundry
    slq = np.load(base%(fills[0]))['arr_0']
    if pix_xmin_o < 0 or pix_xmax_o >= npix_x:
        slq = np.roll(slq, int(npix_x) / 2, axis=0)
    if pix_ymin_o < 0 or pix_ymax_o >= npix_y:
        slq = np.roll(slq, int(int(npix_y) / 2), axis=1)
    slq = slq[pixrange]

    if numsl > 1:
        slq = 10**slq # the way the emission/coldens files are stored by default is as log values
        for fill in fills[1:]:
            tq = np.load(base%(fill))['arr_0']
            if pix_xmin_o < 0 or pix_xmax_o >= npix_x:
                tq = np.roll(tq, int(npix_x) / 2, axis=0)
            if pix_ymin_o < 0 or pix_ymax_o >= npix_y:
                tq = np.roll(tq, int(int(npix_y) / 2), axis=1)
            tq = tq[pixrange]
            tq = 10**tq
            slq +=tq
        slq = np.log10(slq)
            
    ### extract the data
    shape = slq.shape
    inds = np.indices(shape)
    # use distance to pixel centres
    dists2 = (length_per_pixel * (inds[0] + pix_xmin + 0.5) - c0)**2 + \
             (length_per_pixel * (inds[1] + pix_ymin + 0.5) - c1)**2 
    ann = np.logical_and(dists2 >= rmin**2, dists2 <= rmax**2)
    rs = np.ndarray.flatten(np.sqrt(dists2[ann])/rscale)
    qs = np.ndarray.flatten(slq[ann])
    
    print(str(pixrange))

    if plot !=None:

        name = plot%('_zoom-x-%i-%i-y-%i-%i_slices-%i-through-%i_%s'%(pix_xmin_o,pix_xmax_o,pix_ymin_o,pix_ymax_o,ceninds[0],ceninds[-1],label))
        fontsize=13
        colmap = 'viridis'
        xystarts = [pix_xmin_o*length_per_pixel, pix_ymin_o*length_per_pixel]
        size = (pix_xmax_o - pix_xmin_o )*length_per_pixel
        Vmin=10.5
        Vmax=15.5
        
        circle = plt.Circle(centre,rscale,color='white',fill=False)   

        fig = plt.figure(figsize = (5.5, 5.)) # large size just as a trick to get higher resolution
        ax = plt.subplot(111)   
        ax.set_xlabel(r'X [cMpc]',fontsize=fontsize)
        ax.set_ylabel(r'Y [cMpc]',fontsize=fontsize)
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize)
        ax.patch.set_facecolor(cm.get_cmap(colmap)(0.)) # sets background color to lowest color map value 
        img = ax.imshow(slq.T,extent=(xystarts[0],xystarts[0]+size,xystarts[1],xystarts[1]+size),origin='lower', cmap=cm.get_cmap(colmap), vmin = Vmin, vmax=Vmax)
        #plt.title(title,fontsize=fontsize)
        ax.add_artist(circle)
        div = axgrid.make_axes_locatable(ax)
        cax = div.append_axes("right",size="5%",pad=0.1)
        cbar = plt.colorbar(img, cax=cax)
        cbar.solids.set_edgecolor("face")
        cbar.ax.set_ylabel(r'%s' % (clabel), fontsize=(fontsize+1))
        cbar.ax.tick_params(labelsize=fontsize)
        # save PDF figure
        fig.tight_layout()
        #fig.subplots_adjust(right=0.88)
        plt.savefig(name,format = 'pdf',dpi=1200)
        plt.close()
    return np.array([rs,qs])


def rdists_sl(base, szcens, L_x, npix_x, rmins, rmaxs, rscales, centres, 
              labels, numsl=1, npix_y=None,plot=None,clabel=None,
              title=None, axis='z'):
    return {labels[i]: rdist_sl(base, szcens, L_x, npix_x, rmins[i], rmaxs[i],
                                rscales[i], 
                                (centres[0][i], centres[1][i], centres[2][i]),
                                numsl=numsl, npix_y=npix_y, plot=plot,
                                clabel=clabel, title=title, label=labels[i],
                                axis=axis)
            for i in range(len(labels))}


def rdists_sl_faster(base, szcens, L_x, npix_x,\
                     rmin, rmax, rscales, centres,\
                     numsl=1, npix_y=None, axis='z', logquantity=True,\
                     labels=None, save=None, trackprogress=False):
    '''
    inputs:
    ---------------------------------------------------------------------------
    base:        (str incl. %s) string containing the file name (including 
                 directory) for the images to extract 
                 data from (data = image values)
      * do a quick directory check
    szcens:      (iterable of strings) values of centre coordinates along the 
                 axis of the data files; should complete the file names when 
                 substituted into base. 
      * only uses length of this -> change
                 (follow coldens/emission projection naming conventions)
    L_x:         (float) length of box along the zero coordinate (x if axis is 
                 z)
    npix_x:      (int) number of pixels along the zero coordinate (x if axis is
                 z)
    rmin:        (float, same units as L_x) minimum (2D) radius around the 
                 centre to take data from
    rmax:        (float, same units as L_x) maximum (2D) radius around the 
                 centre to take data from  
    rscales:     (iterable of floats, same units as L_x) radius to scale radial
                 coordinates 
                 to for output (i.e. units of output radii)
    centres:     (iterable of float * 3) centre around which to extract data, 
                 in EAGLE x,y,z coordinates
    axis:        ('x', 'y', or 'z') axis along which the images are projected 
                 (sets centre interpretation)
                 default: 'z'
    numsl:       (int) number of slices around the centre to use
                 default: 1
    npix_y:      (int) number of pixels along the 1 coordinate (pixels 
                 assumed to be square)
                 default: None -> set to npix_x
    logquantity: whether the images to be loaded are stored as log values or 
                 not
                 default: True
    labels:      names to store rs,qs arrays with
                 default: None -> centre coordinates
    save:        name of npz file to save results to
                 default None -> save nothing
       * get some stuff from the base name?

    output:
    ---------------------------------------------------------------------------
    dct of float array [rs,qs]: radii (rscale units) and quantity (input image 
    units)
    
    warnings:
    ---------------------------------------------------------------------------
    For a large set of halos (e.g. all halos in RecalL0025N0752), the memory
    use here is large -- ~25% of the memory on quasar, from the halo selection
    alone. (It doesn't change much when loading the corresponding ~1GB column
    density maps.)
    '''
    ## loop over images, then galaxies within each image
    
    ## setup/initial check
    centres = np.array(centres)
    if axis == 'z':
        c0 = centres[:,0]
        c1 = centres[:,1]
        c2 = centres[:,2]
    elif axis == 'x':
        c0 = centres[:,2]
        c1 = centres[:,0]
        c2 = centres[:,1]
    elif axis == 'y':
        c0 = centres[:,1]
        c1 = centres[:,2]
        c2 = centres[:,0]
    else:
        raise ValueError('%s is an invalid axis option' %str(axis))
  
    if labels is None:
        labels = [str(centre) for centre in centres]

    #### find the slice ranges for each centre

    # slice centres
    if trackprogress:
        print('Retrieving slice centers (fills)')
    try:
        szcens = list(szcens)
        szcens.sort(key=lambda x: float(x))
        zcens = [np.float(cen) for cen in szcens]
        zcens = np.asarray(zcens)
        oneslice  = len(szcens) <= 1
        if len(szcens) == 0:
            szcens = ['']
            zcens = np.array([0.]) #arbitrary number to not crash stuff
    except TypeError: #szcens was None, for example
        szcens = ['']
        zcens = np.array([0.]) #arbitrary number to not crash stuff
        oneslice = True

    if trackprogress:
        print('matching galaxies to slices')
    length_per_slice = np.float(L_x) / len(zcens)
    slice_cenleft_inds = np.asarray(\
                             np.round(c2 / length_per_slice + 0.5 * (numsl % 2), 0) - 1.,\
                             dtype=int) % len(zcens)
    # odd number of slices  -> centre falls bin with index ind: ind*length_per_slice <= centre < (ind+1)*length_per_slice
    # even number of slices -> centre/left has ind+1 (bins right edge) closest to centre coordinate in length_per_slice units
    #
    # 0r    1r    2r    3r    4r    5r    6r       position (r = length_per_slice)   
    # |  0  |  1  |  2  |  3  |  4  |  5  |        slice index
    # -----------------------------------------
    #       |    *      |   * |
    #       |           +-----+  -> odd numsl case: centre index = (right edge of position bin)/r  = np.round(c2/r - 0.5, 0) 
    #       +-----------+  -> even numsl case: centre/left index = (closet position bin edge)/r -1 = np.round(c2/r, 0) -1
    #
    # at the periodic boundary: 
    #  odd case is ok, %len(zcens) takes care of -1/len(zcens) indices that may come out 
    #  even case should be as well, when using %len(zcens)

    ceninds = (slice_cenleft_inds[:, np.newaxis] \
               - ((numsl - 1) // 2) + np.arange(numsl)[np.newaxis, :])\
              % len(zcens) # numsl/2 should be integer division 
    fills = np.array(szcens)[ceninds]
    #fills_dct = {labels[i]: fills[i] for i in range(len(fills))}
    fills_toloop = list(set(list(fills.flatten()))) # eliminate doubles for the total loop
    #print(fills)
    
    ## checked selection of slices
    #print fills_toloop, slice_cenleft_inds, fills.shape
    #return fills_toloop, slice_cenleft_inds

   
    #### find the indices to select for each centre (note that this list could get pretty unwieldy if the halos selected have a large covering fraction)
    if trackprogress:
        print('Finding the pixel indices to select for different galaxies')
    if npix_y == None:
        npix_y = npix_x
    # rscales may differ, so the selection may not be the same size for each centre -> cannot use array to store all selections (unless oject array)
    selections = []

    # square pixels assumed
    length_per_pixel = np.float(L_x) / float(npix_x)
    
    # selection for a 2d array should be a list of x indices, then a list of y indices
    rminmax = np.array([rscales * rmin / length_per_pixel,\
                        rscales * rmax / length_per_pixel]).T #min/max radii in pixel units
    pixcens0 = c0 / length_per_pixel
    pixcens1 = c1 / length_per_pixel
    # get all indices with pixel centres in radius limits
    #rmaxall = np.max(rminmax)
    basegrid = [np.indices((int(2 * np.ceil(rhmax) + 3),) * 2) - np.ceil(rhmax) for (rhmin, rhmax) in rminmax] # grid centered on 0
    # account for offsets from pixel centres and find distances in pixel units
    rs      = [ ( (basegrid[i][0] + pixcens0[i] - np.round(pixcens0[i], 0))**2 + \
                  (basegrid[i][1] + pixcens1[i] - np.round(pixcens1[i], 0))**2 )**0.5 \
              for i in range(len(c0)) ] 
    rs_sel  = [np.all(np.array([rs[i] >= rminmax[i,0], rs[i] < rminmax[i,1]]), axis=0)  for i in range(len(c0))]
    # selections[i] should get all relevant pixels from an image
    selections = [ ( ((basegrid[i][0][rs_sel[i]] + np.round(pixcens0[i],0)).astype(np.int)).flatten()%npix_x,\
                     ((basegrid[i][1][rs_sel[i]] + np.round(pixcens1[i],0)).astype(np.int)).flatten()%npix_y ) \
                  for i in range(len(c0))] 
    del basegrid
    gc.collect()
    # convert distances to rscale units -> ready for output
    rs = [ (rs[i][rs_sel[i]]).flatten() * length_per_pixel/rscales[i] for i in range(len(c0)) ] 
    #rs_dct = {labels[i]: rs[i] for i in range(len(fills))}

    #### do the image load loop
    if trackprogress:
        print('Looping over the maps to extract the pixels')
    qs = [np.zeros(len(rs[i])) for i in range(len(fills))] # set up dict of the right length, initiate to zeros
    for fill in fills_toloop:
        if not oneslice:
            if '/' in base:
                if base[-4:] == '.npz':
                    fullim = np.load(base%fill)['arr_0']
                else:
                    with h5py.File(base%fill, 'r') as fi:
                        fullim = np.array(fi['map'])
            else:
                try: 
                    if base[-4:] == '.npz':
                        fullim = np.load(ol.ndir + base%fill)['arr_0']
                    else:
                        with h5py.File(ol.ndir + base%fill, 'r') as fi:
                            fullim = np.array(fi['map'])
                except IOError as exc:
                    print(exc)
                    print('Trying alt. directory')
                    if base[-4:] == '.npz':
                        fullim = np.load(ol.ndir_old + base%fill)['arr_0']
                    else:
                        with h5py.File(ol.ndir_old + base%fill, 'r') as fi:
                            fullim = np.array(fi['map'])
        else:
            if '/' in base:
                if base[-4:] == '.npz':
                    fullim = np.load(base)['arr_0']
                else:
                    with h5py.File(base, 'r') as fi:
                        fullim = np.array(fi['map'])
            else:
                try: 
                    if base[-4:] == '.npz':
                        fullim = np.load(ol.ndir + base)['arr_0']
                    else:
                        with h5py.File(ol.ndir + base, 'r') as fi:
                            fullim = np.array(fi['map'])
                except IOError as exc:
                    print(exc)
                    print('Trying alt. directory')
                    if base[-4:] == '.npz':
                        fullim = np.load(ol.ndir_old + base)['arr_0']
                    else:
                        with h5py.File(ol.ndir_old + base, 'r') as fi:
                            fullim = np.array(fi['map'])
                    
        print('Loaded %s'%fill)
        # only modify the arrays for which we want to use this slice; avoid for loops by list comprehension (arrays won't work if the selection regions have different sizes)
        if logquantity:
            qs = [qs[i] + 10**fullim[selections[i]]  if fill in fills[i]
                  else qs[i] for i in range(len(fills))]
        else:
            qs = [qs[i] +     fullim[selections[i]]  if fill in fills[i]
                  else qs[i] for i in range(len(fills))]
        
        del fullim
        num = gc.collect()
        print('garbage collector found %i unreachable objects'%num)
    
    if trackprogress:
        print('Final data operations before saving')
    if logquantity:
        qs  = [np.log10(q) for q in qs]
    
    qs  = [q.flatten() for q in qs]
    dct_out = {labels[i]: np.array([rs[i],qs[i]]) for i in range(len(selections))}
    print('Saving the r/prop data')
    if save is not None:
        try:
            if '/' not in save:
                save  = pdir + save
            if save[-5:] != '.hdf5':
                save = save + '.hdf5'
            with h5py.File(save, 'a') as fo:
                for key in dct_out.keys():
                    ds = fo.create_dataset(str(key), data=dct_out[key])
                    ds.attrs.create('units', 'N x 2: radius, value')
                hed = fo.create_group('Header')
                hed.attrs.create('filename_base', base)
                hed.attrs.create('filename_fills', np.string_(np.array(szcens)))
                hed.attrs.create('pixels_along_x', npix_x)
                hed.attrs.create('pixels_along_y', npix_y)
                hed.attrs.create('size_along_x', L_x)
                hed.attrs.create('axis', axis)
                hed.attrs.create('logvalues', logquantity)
                hed.attrs.create('radius_min_size_x_units', rmin)
                hed.attrs.create('radius_max_size_x_units', rmax)
                hed.create_dataset('rscales_size_x_units', data=rscales)
                hed.create_dataset('centres_size_x_units', data=centres)
        # wouldn't want the hdf5-saving part to cost me the data we got
        except IOError as error:
            print('Failed to save output: IOError \n%s'%error)
        except Exception as exp:
            print('Output save failed for some reason\n%s'%exp)          
    return dct_out    

def rdists_sl_from_haloids(base, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c,\
                     catname,\
                     galids='all', outname=None,\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=None,\
                     axis='z', velspace=False, offset_los=0., stamps=False,\
                     trackprogress=False):
    '''
    offset: [float, cMpc] added to all galaxy los positions 
            -> positive offset means looking at stuff redder/further away than
            the galaxy
    mindist_pkpc: None or float
             get profiles out to at least mindist_pkpc, including for smaller 
             halos. setting rmax_200c to zero and giving this means you get
             profiles out to a fixed distance in pkpc
    '''

    if '/' not in catname:
        catname = pdir + catname
    
    if '/' in base:
        searchdir = '/'.join(base.split('/')[:-1])
        base_file = base.split('/')[-1]
    else:
        searchdir = None
        base_file = base
    # for total box projections: szcens None or empty -> just leave it
    if trackprogress:
        print('Getting slice fills')
    if szcens is not None:
        if len(szcens) > 0:
            szcens = getcenfills(base_file, closevals=[float(cen) for cen in szcens],\
                                 searchdir=searchdir, tolerance=1e-4)
    #print('---------------------------------------------------------------')
    #print(szcens)
    
    if axis == 'z':
        Axis3 = 2
        axname = 'Z'
    elif axis == 'y':
        Axis3 = 1
        axname = 'Y'
    elif axis == 'x':
        Axis3 = 0
        axname = 'Z'
    else:
        raise ValueError("axis must be 'x', y', or 'z'")

    if trackprogress:
        print('reading in halo catlogue data')
    with h5py.File(catname, 'r') as fi:
        if trackprogress:
            print('Succeded in opening halo catalogue hdf5 file')
        z = fi['Header/cosmopars'].attrs['z']
        R200c_cMpc = np.array(fi['R200c_pkpc']) * 1e-3 * (1. + z)
        if mindist_pkpc is None:
            mindist_cMpc = 0.
        else:
            mindist_cMpc = mindist_pkpc * 1e-3 * (1. + z)
        centres_cMpc = np.array([np.array(fi['Xcop_cMpc']), np.array(fi['Ycop_cMpc']), np.array(fi['Zcop_cMpc'])]).T
        ids   = np.array(fi['galaxyid'])
        cosmopars = {key: item for (key, item) in fi['Header/cosmopars'].attrs.items()}
        boxsize = cosmopars['boxsize'] / cosmopars['h']
        
        if velspace:    
            vpec = np.array(fi['V%spec_kmps'%axname]) * 1e5 # cm/s
            vpec *= 1. / (cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * cu.c.cm_per_mpc * cosmopars['a']) # cm/s -> H.f. cMpc  
            centres_cMpc[:, Axis3] += vpec 
            centres_cMpc[:, Axis3] %= boxsize
        
        if offset_los != 0.:
            centres_cMpc[:, Axis3] += offset_los
            centres_cMpc[:, Axis3] %= boxsize
    
    if trackprogress:
        print('Applying galaxyid selection')
    if isinstance(galids, str):
        if galids == 'all':
            halos = ids
            R200c = R200c_cMpc
            centres = centres_cMpc
        else:
            raise ValueError('galids should be an iterable of galaxy ids or "all", not %s'%galids)
    else:
        inds = np.array([np.where(ids == galid)[0][0] for galid in galids])
        halos = ids[inds]
        R200c = R200c_cMpc[inds]
        centres = centres_cMpc[inds, :]
    
    if trackprogress:
        print('Setting extraction radii')
    adjustscale = rmax_r200c * R200c < mindist_cMpc
    if np.sum(adjustscale) > 0:
        rmax_r200c = np.ones(len(halos)) * rmax_r200c 
        if hasattr(mindist_cMpc, '__len__'):
            _mindist_cMpc = mindist_cMpc[adjustscale]
        else:
            _mindist_cMpc = mindist_cMpc            
        rmax_r200c[adjustscale] = _mindist_cMpc / R200c[adjustscale]
    if trackprogress:
        print('Calling stamp or r/prop extraction')
    if stamps:
        if base[-5:] == '.hdf5':
            return stamps_sl_hdf5(base, szcens, rmax_r200c, centres, rscales=R200c,\
                           numsl=numsl, labels=halos, save=outname)
        else:
            return stamps_sl(base, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c, R200c, centres,\
                     numsl=numsl, npix_y=npix_y, axis=axis, logquantity=logquantity,\
                     labels=halos, save=outname)
    else:          
        return rdists_sl_faster(base, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c, R200c, centres,\
                     numsl=numsl, npix_y=npix_y, axis=axis, logquantity=logquantity,\
                     labels=halos, save=outname, trackprogress=trackprogress)


def rdists_sl_from_selection(base, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c,\
                     catname,\
                     selection, maxnum, outname=None,\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=None,\
                     axis='z', velspace=False, offset_los=0., stamps=False,\
                     trackprogress=False):
    '''
    stamps: get images instead of (r, value) arrays
    '''
    print('Called rdists_sl_from_selection with inputs:')
    #print('\tselection:\t%s'%selection)
    print('\tcatname:\t%s, \tmaxnum:\t%s'%(catname, maxnum))
    print('\tbase:\t%s'%base)
    print('\tszcens:\t%s'%szcens)
    print('\tnumsl:\t%s, \taxis:\t%s, \tvelspace:\t%s, \tlogquantity:\t%s, \toffset_los:\t%s'%(numsl, axis, velspace, logquantity, offset_los))
    print('\tL_x:\t%s, \tnpix_x:\t%s, \tnpix_y:\t%s, \trmin_r200c:\t%s, \trmax_r200c:\t%s, \tmindist_pkpc:\t%s'%(L_x, npix_x, npix_y, rmin_r200c, rmax_r200c, mindist_pkpc))
    print('')
    
    if trackprogress:
        print('getting galaxy ids (selecthalos)')
    galids = sh.gethaloselections(catname, selections=[selection], names=[0])
    # selecthalos does not necessarily return things in the input order
    # if galaxyids are input, preserve their order (first instance of galaxyid)
    # neceassary for proper operation of mindist_pkpc as an array
    if np.any([sel[0] == 'galaxyid' for sel in selection]):
        si = np.where([sel[0] == 'galaxyid' for sel in selection])[0][0]
        gids_in = selection[si][1]
        subset = np.array([gid in galids[0] for gid in gids_in])
        gids_sel = gids_in[subset]
        if set(gids_sel) != set(galids[0]):
            if set(gids_sel).issubset(galids[0]):
                print('galaxy ids missing from the intended subset selection')
            else:
                print('galaxy ids selected contradict selection criteria')
            raise RuntimeError('rdists_sl_from_selection: something has gone wrong in the halo selection')
        order = np.array([np.where(gid == galids[0])[0][0] for gid in gids_sel])
        galids[0] = galids[0][order]
        
    if trackprogress:
        print('applying selection')
    if selection is not None and outname is not None:
        if '/' not in outname:
            outname = ol.pdir + outname
        if outname[-5:] != '.hdf5':
            outname = outname + '.hdf5'
        
        with h5py.File(outname, 'a') as fo:
            sel = fo.create_group('selection')
            for sl in selection:
                name = sl[0] + '_set%i'
                counter = 0 
                while name%counter in sel.keys():
                    counter += 1
                if sl[0] in ['galaxyid', 'groupid']: #int datasets
                    sel.create_dataset(name%counter, data=np.array(sl[1]))
                else:
                    sel.create_dataset(name%counter, data=np.array(sl[1:]).astype(np.float))
            if 'galaxyid' not in sel.keys():
                sel.create_dataset('galaxyid', data=galids[0])
    if trackprogress:
        print('Calling rdists_sl_from_haloids') 
    rdists_sl_from_haloids(base, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c,\
                     catname,\
                     galids=galids[0], outname=outname, mindist_pkpc=mindist_pkpc,\
                     numsl=numsl, npix_y=npix_y, logquantity=logquantity,
                     axis=axis, velspace=velspace, offset_los=offset_los,\
                     stamps=stamps, trackprogress=trackprogress)

def percentiles_from_hdf5(h5name):
    percentiles = np.array([10., 25., 50., 75., 90.])
    rbins = np.arange(0., 2., 0.1)
    
    if '/' not in h5name:
        h5name = pdir + h5name
    with h5py.File(h5name, 'r') as fi:
        keys = fi.keys()
        keys.remove('Header')
        dct_out = {key: rqhist(np.array(fi[key]), rbins, percentiles=percentiles) for key in keys}   
    return dct_out

def stamps_sl(base, szcens, L_x, npix_x,\
              rmin, rmax, rscales, centres,\
              numsl=1, npix_y=None, axis='z', logquantity=True,\
              labels=None, save=None):
    '''
    switch to stamps_sl_hdf5 is recommended: more checks and header logging 
    from hdf5 map files
    
    inputs:
    ---------------------------------------------------------------------------
    base:        (str incl. %s) string containing the file name (including 
                 directory) for the images to extract 
                 data from (data = image values)
      * do a quick directory check
    szcens:      (iterable of strings) values of centre coordinates along the 
                 axis of the data files; should complete the file names when 
                 substituted into base. 
      * only uses length of this -> change
                 (follow coldens/emission projection naming conventions)
    L_x:         (float) length of box along the zero coordinate (x if axis is 
                 z)
    npix_x:      (int) number of pixels along the zero coordinate (x if axis is
                 z)
    rmin:        (float, same units as L_x) minimum (2D) radius around the 
                 centre to take data from
    rmax:        (float, same units as L_x) maximum (2D) radius around the 
                 centre to take data from  
    rscales:     (iterable of floats, same units as L_x) radius to scale radial coordinates 
                 to for output (i.e. units of output radii)
    centres:     (iterable of float * 3) centre around which to extract data, in EAGLE 
                 x,y,z coordinates
    axis:        ('x', 'y', or 'z') axis along which the images are projected 
                 (sets centre interpretation)
                 default: 'z'
    numsl:       (int) number of slices around the centre to use
                 default: 1
    npix_y:      (int) number of pixels along the 1 coordinate (pixels 
                 assumed to be square)
                 default: None -> set to npix_x
    logquantity: whether the images to be loaded are stored as log values or not
                 default: True
    labels:      names to store rs,qs arrays with
                 default: None -> centre coordinates
    save:        name of npz file to save results to
                 default None -> save nothing
       * get some stuff from the base name?

    output:
    ---------------------------------------------------------------------------
    dct of images (2d-arrays)
    '''
    ## loop over images, then galaxies within each image
    
    ## setup/initial check
    centres = np.array(centres)
    if axis == 'z':
        c0 = centres[:,0]
        c1 = centres[:,1]
        c2 = centres[:,2]
    elif axis == 'x':
        c0 = centres[:,2]
        c1 = centres[:,0]
        c2 = centres[:,1]
    elif axis == 'y':
        c0 = centres[:,1]
        c1 = centres[:,2]
        c2 = centres[:,0]
    else:
        print('%s is an invalid axis option' %str(axis))
        return None
  
    if labels is None:
        labels = [str(centre) for centre in centres]

    #### find the slice ranges for each centre

    # slice centres
    szcens = list(szcens)
    szcens.sort(key=lambda x: float(x))
    zcens = [np.float(cen) for cen in szcens]
    zcens = np.asarray(zcens)

    length_per_slice = np.float(L_x) / len(zcens)
    slice_cenleft_inds = np.asarray(\
        np.round(c2 / length_per_slice + 0.5 * (numsl % 2), 0) - 1.,\
        dtype=int) % len(zcens)
    # odd number of slices  -> centre falls bin with index ind: ind*length_per_slice <= centre < (ind+1)*length_per_slice
    # even number of slices -> centre/left has ind+1 (bins right edge) closest to centre coordinate in length_per_slice units
    #
    # 0r    1r    2r    3r    4r    5r    6r       position (r = length_per_slice)   
    # |  0  |  1  |  2  |  3  |  4  |  5  |        slice index
    # -----------------------------------------
    #       |    *      |   * |
    #       |           +-----+  -> odd numsl case: centre index = (right edge of position bin)/r  = np.round(c2/r - 0.5, 0) 
    #       +-----------+  -> even numsl case: centre/left index = (closet position bin edge)/r -1 = np.round(c2/r, 0) -1
    #
    # at the periodic boundary: 
    #  odd case is ok, %len(zcens) takes care of -1/len(zcens) indices that may come out 
    #  even case should be as well, when using %len(zcens)

    ceninds = (slice_cenleft_inds[:, np.newaxis] -\
               ((numsl - 1) // 2) + np.arange(numsl)[np.newaxis, :]) \
              % len(zcens) # numsl/2 should be integer division 
    fills = np.array(szcens)[ceninds]
    #fills_dct = {labels[i]: fills[i] for i in range(len(fills))}
    fills_toloop = list(set(list(fills.flatten()))) # eliminate doubles for the total loop

    ## checked selection of slices
    #print fills_toloop, slice_cenleft_inds, fills.shape
    #return fills_toloop, slice_cenleft_inds

   
    #### find the indices to select for each centre (note that this list could get pretty unwieldy if the halos selected have a large covering fraction)
    if npix_y == None:
        npix_y = npix_x
    # rscales may differ, so the selection may not be the same size for each centre -> cannot use array to store all selections (unless oject array)
    selections = []

    # square pixels assumed
    length_per_pixel = np.float(L_x) / npix_x
    
    # selection for a 2d array should be a list of x indices, then a list of y indices
    rmax = np.array(rscales) * rmax / length_per_pixel 
    pixcens0 = c0 / length_per_pixel
    pixcens1 = c1 / length_per_pixel
    # get all indices with pixel centres in radius limits
    minsmaxs = np.array([[[np.floor(pixcens0[i] - rmax[i]),\
                           np.ceil(pixcens0[i] + rmax[i]) + 1],\
                          [np.floor(pixcens1[i] - rmax[i]),\
                           np.ceil(pixcens1[i] + rmax[i]) + 1]]\
                         for i in range(len(c0))]).astype(int)
    selections = [(slice(minsmaxs[i][0][0], minsmaxs[i][0][1], None),\
                   slice(minsmaxs[i][1][0], minsmaxs[i][1][1], None)) for i in range(len(c0))]
    lower_left_corners = length_per_pixel * np.array([[minsmaxs[i][0][0], minsmaxs[i][1][0]]\
                                                      for i in range(len(fills))])

    # get all indices with pixel centres in radius limits
    #rmaxall = np.max(rminmax)
    basegrid = [np.indices((int(2 * np.ceil(rhmax) + 3),) * 2)\
                - np.ceil(rhmax) for rhmax in rmax] # grid centered on 0
    # selections[i] should get all relevant pixels from an image
    selections = [ ( ((basegrid[i][0] + np.round(pixcens0[i], 0)).astype(np.int))%npix_x,\
                     ((basegrid[i][1] + np.round(pixcens1[i], 0)).astype(np.int))%npix_y )\
                  for i in range(len(c0))] 
    del basegrid
    gc.collect()

    qs = [np.zeros(selection[0].shape) for selection in selections] # set up dict of the right length, initiate to zeros
    for fill in fills_toloop:
        if len(fills) > 1:
            if '/' in base:
                fullim = np.load(base%fill)['arr_0']
            else:
                try: 
                    fullim = np.load(ol.ndir + base%fill)['arr_0']
                except:
                    fullim = np.load(ol.ndir_old + base%fill)['arr_0']
        else:
            if '/' in base:
                fullim = np.load(base)['arr_0']
            else:
                try: 
                    fullim = np.load(ol.ndir + base)['arr_0']
                except:
                    fullim = np.load(ol.ndir_old + base)['arr_0']
                    
        print('Loaded %s'%fill)
        # only modify the arrays for which we want to use this slice; avoid for loops by list comprehension (arrays won't work if the selection regions have different sizes)
        if logquantity:
            qs = [qs[i] + 10**fullim[selections[i]]  if fill in fills[i] else\
                  qs[i] for i in range(len(fills))]
        else:
            qs = [qs[i] +     fullim[selections[i]]  if fill in fills[i] else\
                  qs[i] for i in range(len(fills))]
    
    if logquantity:
        qs  = [ np.log10(q) for q in qs]

    dct_out = {labels[i]: qs[i] for i in range(len(selections))}
    if save is not None:
        simfile_set = False
        try:
            simdct = mc.get_simdata_from_outputname(base)
            simulation = simdct['simulation']
            simnum = simdct['simnum']
            var = simdct['var']
            snapnum = simdct['snapnum']
            cosmopars = mc.getcosmopars(simnum, snapnum, var, file_type='snap', simulation=simulation)
            simfile_set = True
        except Exception as exp:
            print('Retrieveing projection data failed for some reason\n%s'%exp) 
        fon = save
        if '/' not in fon:
            fon = pdir + fon
        if fon[-5:] != '.hdf5':
            fon = fon + '.hdf5'
        try:
            with h5py.File(fon , 'a') as fo:
                for key in dct_out.keys():
                    fo.create_dataset(str(key), data=dct_out[key])
                hed = fo.create_group('Header')
                hed.attrs.create('filename_base', base)
                hed.attrs.create('filename_fills', np.string_(np.array(szcens)))
                hed.attrs.create('pixels_along_x', npix_x)
                hed.attrs.create('pixels_along_y', npix_y)
                hed.attrs.create('size_along_x', L_x)
                hed.attrs.create('pixel_size_size_x_units', length_per_pixel)
                hed.attrs.create('axis', axis)
                hed.attrs.create('logvalues', logquantity)
                hed.attrs.create('radius_min_size_x_units', rmin)
                hed.attrs.create('radius_max_size_x_units', rmax)
                hed.create_dataset('rscales_size_x_units', data=rscales)
                hed.create_dataset('labels', data=np.array(labels, dtype=int))
                hed.create_dataset('centres_size_x_units', data=centres)
                hed.create_dataset('lower_left_corners_size_x_units', data=lower_left_corners)
                if simfile_set:
                    for key in simdct.keys():
                        hed.attrs.create(key, simdct[key])
                    cgrp = hed.create_group('cosmopars')
                    for key in cosmopars.keys():
                        cgrp.attrs.create(key, cosmopars[key])
        # wouldn't want the hdf5-saving part to cost me the data we got
        except IOError as error:
            print('Failed to save output: IOError \n%s'%error)
        except Exception as exp:
            print('Output save failed for some reason\n%s'%exp)          
    return dct_out    

def stamps_sl_hdf5(base, szcens, rmax, centres, rscales=1.,\
              numsl=1, labels=None, save=None):
    '''
    extract stamps from make_maps outputs in hdf5 format
    note: this is pretty memory-intensive
    
    inputs:
    ---------------------------------------------------------------------------
    base:        (str incl. %s) string containing the file name (including 
                 directory) for the images to extract 
                 data from (data = image values)
    szcens:      (iterable of strings) should complete the file names when 
                 substituted into base. 
                 (follow coldens/emission projection naming conventions)
    rmax:        (float array) (2D) radius around the centre to take data from 
                 (rmax * rscales units: cMpc) 
    rscales:     (float or float array) factor to multiply rmax with 
                 (rmax * rscales units: cMpc) 
    centres:     (iterable of float * 3) centre around which to extract data, 
                 in EAGLE x,y,z coordinates (units: cMpc)
    numsl:       (int) number of slices around the centre to use
                 default: 1
    labels:      names to store image arrays with (int; intended for galaxy 
                 ids)
                 default: None -> centre coordinates
    save:        name of hdf5 file to save results to
                 default None -> save nothing

    output:
    ---------------------------------------------------------------------------
    dct of images (2d-arrays): {label: image}
    '''
    ## loop over images, then galaxies within each image
    
    _base = base
    if '/' not in base:
        _base = ol.ndir + base
        if not os.path.isfile(_base%szcens[0]):
            _base = ol.ndir_old + base
    if not os.path.isfile(_base%szcens[0]):
        raise ValueError('Could not find file {}'.format(_base%szcens[0]))
    base = _base
    
    # first file loop: metadata
    heddata = {}
    for scen in szcens:
        _filen = base%scen
        with h5py.File(_filen, 'r') as ft:
            heddata[scen] = {}
            heddata[scen]['inputpars'] = {key: val for key, val in\
                   ft['Header/inputpars'].attrs.items()}
            heddata[scen]['cosmopars'] = {key: val for key, val in\
                   ft['Header/inputpars/cosmopars'].attrs.items()}
            heddata[scen]['halosel'] = {key: val for key, val in\
                   ft['Header/inputpars/halosel'].attrs.items()}
            heddata[scen]['misc'] = {key: val for key, val in\
                   ft['Header/inputpars/misc'].attrs.items()}
    ## sanity check on metadata: do the values we need agree between slices
    sc = szcens[0]
    # projection axis
    axis = heddata[sc]['inputpars']['axis']
    if not np.all([axis == heddata[_sc]['inputpars']['axis'] for _sc in szcens]):
        raise ValueError('stamps_sl_hdf5: Different slices had different projection axes')
    axis = axis.decode()
    if axis == 'z':
        axis1 = 0
        axis2 = 1
        axis3 = 2
    elif axis == 'x':
        axis1 = 1
        axis2 = 2
        axis3 = 0
    elif axis == 'y':
        axis1 = 2
        axis2 = 0
        axis3 = 1
    # slice dimensions    
    Ls = np.array([heddata[sc]['inputpars']['L_x'],\
                   heddata[sc]['inputpars']['L_y'],\
                   heddata[sc]['inputpars']['L_z']])
    if not np.all([np.isclose(Ls[0], heddata[_sc]['inputpars']['L_x']) and \
                   np.isclose(Ls[1], heddata[_sc]['inputpars']['L_y']) and\
                   np.isclose(Ls[2], heddata[_sc]['inputpars']['L_z']) \
                   for _sc in szcens]):
        raise ValueError('stamps_sl_hdf5: Different slices had different slice dimensions')  
    L_x = Ls[axis1]
    L_y = Ls[axis2]
    length_per_slice = Ls[axis3]
    # image pixels
    npix_x = heddata[sc]['inputpars']['npix_x']
    npix_y = heddata[sc]['inputpars']['npix_y']
    if not np.all([npix_x == heddata[_sc]['inputpars']['npix_x'] and \
                   npix_y == heddata[_sc]['inputpars']['npix_y'] \
                   for _sc in szcens]):
        raise ValueError('stamps_sl_hdf5: Different slices had different image shapes')  
    slicexycen = (heddata[sc]['inputpars']['centre'][axis1],\
                  heddata[sc]['inputpars']['centre'][axis2])
    if not np.all([np.isclose(slicexycen[0], heddata[_sc]['inputpars']['centre'][axis1]) and \
                   np.isclose(slicexycen[1], heddata[_sc]['inputpars']['centre'][axis2]) \
                   for _sc in szcens]):
        raise ValueError('stamps_sl_hdf5: Different slices had different image centres') 
    slicezcen = {_sc: heddata[_sc]['inputpars']['centre'][axis3] for _sc in szcens}
    boxsize = heddata[sc]['cosmopars']['boxsize'] / heddata[sc]['cosmopars']['h']
    if not np.all([np.isclose(boxsize,\
                   heddata[_sc]['cosmopars']['boxsize'] / heddata[_sc]['cosmopars']['h']) \
                   for _sc in szcens]):
        raise ValueError('stamps_sl_hdf5: Different slices came from different simulation box sizes ')
    
    ## setup/initial check
    centres = np.array(centres)
    c0 = centres[:, axis1]
    c1 = centres[:, axis2]
    c2 = centres[:, axis3]
  
    if labels is None:
        labels = [str(centre) for centre in centres]

    #### find the slice ranges for each centre

    # slice centres
    szcens = list(szcens)
    szcens.sort(key=slicezcen.get)
    zcens = [slicezcen[_sc] for _sc in szcens]
    zcens = np.asarray(zcens)
    if not (np.isclose(length_per_slice * len(szcens),\
                      heddata[sc]['cosmopars']['boxsize'] / heddata[sc]['cosmopars']['h']) \
            and np.allclose(np.diff(zcens), length_per_slice)):
        raise ValueError('stamps_sl_hdf5: input files are not evenly spaced slices covering the length of the simulation box')
    if not (np.isclose(L_x, boxsize) and np.isclose(L_y, boxsize)):
        raise ValueError('The slices do not span the simulation box perpendicular to the line of sight')
    slice_cenleft_inds = np.asarray(\
        np.round(c2 / length_per_slice + 0.5 * (numsl % 2), 0) - 1.,\
        dtype=int) % len(zcens)
    # odd number of slices  -> centre falls bin with index ind: ind*length_per_slice <= centre < (ind+1)*length_per_slice
    # even number of slices -> centre/left has ind+1 (bins right edge) closest to centre coordinate in length_per_slice units
    #
    # 0r    1r    2r    3r    4r    5r    6r       position (r = length_per_slice)   
    # |  0  |  1  |  2  |  3  |  4  |  5  |        slice index
    # -----------------------------------------
    #       |    *      |   * |
    #       |           +-----+  -> odd numsl case: centre index = (right edge of position bin)/r  = np.round(c2/r - 0.5, 0) 
    #       +-----------+  -> even numsl case: centre/left index = (closet position bin edge)/r -1 = np.round(c2/r, 0) -1
    #
    # at the periodic boundary: 
    #  odd case is ok, %len(zcens) takes care of -1/len(zcens) indices that may come out 
    #  even case should be as well, when using %len(zcens)

    ceninds = (slice_cenleft_inds[:, np.newaxis] -\
               ((numsl - 1) // 2) + np.arange(numsl)[np.newaxis, :]) \
              % len(zcens) # numsl/2 should be integer division 
    fills = np.array(szcens)[ceninds]
    #fills_dct = {labels[i]: fills[i] for i in range(len(fills))}
    fills_toloop = list(np.unique(fills)) # eliminate doubles for the total loop

    ## checked selection of slices
    #print fills_toloop, slice_cenleft_inds, fills.shape
    #return fills_toloop, slice_cenleft_inds

    # rscales may differ, so the selection may not be the same size for each centre -> cannot use array to store all selections (unless oject array)
    selections = []

    # square pixels assumed
    length_per_pixel_x = np.float(L_x) / npix_x
    length_per_pixel_y = np.float(L_y) / npix_y
    
    # convert everything to pixel size coordinates
    rmax_x = np.array(rscales) * rmax / length_per_pixel_x
    rmax_y = np.array(rscales) * rmax / length_per_pixel_y
    pixcens0 = c0 / length_per_pixel_x
    pixcens1 = c1 / length_per_pixel_y
    x_pix0 = (slicexycen[0] - 0.5 * L_x) / length_per_pixel_x
    y_pix0 = (slicexycen[1] - 0.5 * L_y) / length_per_pixel_y
    
    # get all indices with pixel centres in radius limits
    indx0 = (pixcens0 - rmax_x - x_pix0 - 0.5).astype(int)
    indx1 = (pixcens0 + rmax_x - x_pix0 + 0.5).astype(int) + 1
    indy0 = (pixcens1 - rmax_y - y_pix0 - 0.5).astype(int)
    indy1 = (pixcens1 + rmax_y - y_pix0 + 0.5).astype(int) + 1
    #indxc = (pixcens0 - x_pix0 + 0.5).astype(int)
    #indyc = (pixcens1 - y_pix0 + 0.5).astype(int)
    
    lower_left_corners =  np.array([(indx0  + x_pix0) * length_per_pixel_x % boxsize,\
                                    (indy0  + y_pix0) * length_per_pixel_y % boxsize]).T

    # get all indices with pixel centres in radius limits
    #rmaxall = np.max(rminmax)
    basegrid = [np.indices((indx1[i] - indx0[i],\
                            indy1[i] - indy0[i]))
                for i in range(len(c0))]
    # selections[i] should get all relevant pixels from an image
    selections = [ ( (basegrid[i][0] + indx0[i]) % npix_x,\
                     (basegrid[i][1] + indy0[i]) % npix_y )\
                  for i in range(len(c0))] 
    del basegrid
    gc.collect()

    qs = [np.zeros(selection[0].shape) for selection in selections] # set up dict of the right length, initiate to zeros
    for fill in fills_toloop:
        with h5py.File(base%fill) as _f:
            fullim = _f['map'][:]
            logquantity = bool(_f['Header/inputpars'].attrs['log']) 
        print('Loaded %s'%fill)
        # only modify the arrays for which we want to use this slice; avoid for loops by list comprehension (arrays won't work if the selection regions have different sizes)
        if logquantity:
            qs = [qs[i] + 10**fullim[selections[i]]  if fill in fills[i] else\
                  qs[i] for i in range(len(fills))]
        else:
            qs = [qs[i] +     fullim[selections[i]]  if fill in fills[i] else\
                  qs[i] for i in range(len(fills))]
    
    if logquantity:
        qs  = [ np.log10(q) for q in qs]

    dct_out = {labels[i]: qs[i] for i in range(len(selections))}
    if save is not None:
        try:
            fon = save
            if '/' not in fon:
                fon = ol.pdir + 'stamps/' + fon
            if fon[-5:] != '.hdf5':
                fon = fon + '.hdf5'
            with h5py.File(fon) as fo:
                for key in dct_out.keys():
                    fo.create_dataset(str(key), data=dct_out[key])
                hed = fo.create_group('Header')
                # map file headers
                for sc in szcens:
                    sgrp = hed.create_group(sc)
                    tmp = sgrp.create_group('inputpars')
                    for key in heddata[scen]['inputpars']:
                        tmp.attrs.create(key, heddata[scen]['inputpars'][key])
                    tmp = sgrp.create_group('inputpars/cosmopars')
                    for key in heddata[scen]['cosmopars']:
                        tmp.attrs.create(key, heddata[scen]['cosmopars'][key])
                    tmp = sgrp.create_group('inputpars/halosel')
                    for key in heddata[scen]['halosel']:
                        tmp.attrs.create(key, heddata[scen]['halosel'][key])
                    tmp = sgrp.create_group('inputpars/misc')
                    for key in heddata[scen]['misc']:
                        tmp.attrs.create(key, heddata[scen]['misc'][key])
                        
                hed.attrs.create('filename_base', np.string_(base))
                hed.attrs.create('filename_fills', np.array([np.string_(cen) for cen in szcens]))
                hed.attrs.create('pixels_along_x', npix_x)
                hed.attrs.create('pixels_along_y', npix_y)
                hed.attrs.create('size_along_x', L_x)
                hed.attrs.create('pixel_size_x_cMpc', length_per_pixel_x)
                hed.attrs.create('pixel_size_y_cMpc', length_per_pixel_y)
                hed.attrs.create('axis', np.string_(axis))
                hed.attrs.create('logvalues', logquantity)
                hed.attrs.create('rmax_rscales', rmax)
                hed.create_dataset('rscales_cMpc', data=rscales)
                hed.create_dataset('labels', data=np.array(labels, dtype=int))
                hed.create_dataset('centres_cMpc', data=centres)
                hed.create_dataset('lower_left_corners_cMpc', data=lower_left_corners)
                
        # wouldn't want the hdf5-saving part to cost me the data we got
        except IOError as error:
            print('Failed to save output: IOError \n%s'%error)
        except Exception as exp:
            print('Output save failed for some reason\n%s'%exp)          
    return dct_out    

#def stamps_sl_from_haloids(base, szcens, L_x, npix_x,\
#                     rmin_r200c, rmax_r200c,\
#                     catname, haloids, z, outname=None,\
#                     numsl=1, npix_y=None, axis='z', logquantity=True):
#    if '/' not in catname:
#        catname = pdir + catname
#    with h5py.File(catname, 'r') as fi:
#        R200c_cMpc = np.array(fi['R200c_pkpc']) / 1e3 * (1. + z)
#        centres_cMpc = np.array([np.array(fi['Xcop_cMpc']), np.array(fi['Ycop_cMpc']), np.array(fi['Zcop_cMpc'])]).T
#        ids   = np.array(fi['groupid'])
#    inds = np.array([np.where(ids == haloid)[0][0] for haloid in haloids])
#    halos = ids[inds]
#    R200c = R200c_cMpc[inds]
#    centres = centres_cMpc[inds, :]
#
#    rdists_sl_faster(base, szcens, L_x, npix_x,\
#                     rmin_r200c, rmax_r200c, R200c, centres,\
#                     numsl=numsl, npix_y=npix_y, axis=axis, logquantity=logquantity,\
#                     labels=halos, save=outname)



def gethalomask_basic(xpix, size, pixsize, indct,\
                      ypix=None, exdct=None, closest_normradius=True, axis='z',
                      periodic=True):
    '''
    size, pixsize, radii, positions: should have same units, which ones is\
                  arbitrary
    indct, exdct: entries are 'pos': 
                   positions [[X1, Y1, Z1],  ... [Xn, Yn, Zn]]
                  'rad': radii
    exdct:        None -> don't exclude anything
    closest_normradius: if excluding halos, exclude based on closest halo
                  center normalised by radius (otherwise, base on which halo is
                  closest)
    
    when excluding halos, if distances are equal by the chosen measure, the 
    other is used as a tiebreaker
    
    exactly equidistant pixels in halos of exactly the same size (to fp 
    precision) will be excluded, but if this happens a lot, chances are you've
    accidentally added the same halo to the include and exclude lists
    
    periodicity is meant for the output map (assumed to start at 0, 0: shift
    the input positions if you want the corner to be somewhere else). 
    Coordinates are always assumed to be periodic
    halo radii are assumed to be < box size (getting this wrong can lead to 
    segfaults due to negative array indices)
    '''
    if ypix is None:
        ypix = xpix
    
    if axis == 'z':
        Axis1 = 0
        Axis2 = 1
        Axis3 = 2
    elif axis == 'x':
        Axis1 = 1
        Axis2 = 2
        Axis3 = 0
    elif axis == 'y':
        Axis1 = 2
        Axis2 = 0
        Axis3 = 1
    else:
        raise ValueError('Axis should be "x", "y", or "z"')
    
    inpos = indct['pos'].flatten()
    inrad = indct['rad']
    numin = len(inrad)
    outmap = np.zeros(xpix * ypix).astype(ct.c_int)

    acfile = ct.CDLL(ol.c_halomask)    
    if exdct is None:   
        selfunction = acfile.gethalomap_2d_incl
        # ion balance tables are temperature x density x line no.
        selfunction.argtypes = [ct.c_int, ct.c_int, ct.c_float,\
                                np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(numin * 3),),\
                                np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(numin,)),\
                                ct.c_int, \
                                ct.c_int, ct.c_int, ct.c_int,\
                                ct.c_int, ct.c_float,\
                                np.ctypeslib.ndpointer(dtype=ct.c_int, shape=(xpix * ypix,))]
        selfunction.restype = None
        
        res = selfunction(ct.c_int(xpix), ct.c_int(ypix), ct.c_float(pixsize),\
                          inpos.astype(ct.c_float),\
                          inrad.astype(ct.c_float),\
                          ct.c_int(numin),\
                          ct.c_int(Axis1), ct.c_int(Axis2), ct.c_int(Axis3),\
                          ct.c_int(periodic), ct.c_float(size),\
                          outmap)
    
    else:                         
        expos = exdct['pos'].flatten()
        exrad = exdct['rad']
        numex = len(exrad)
        selfunction = acfile.gethalomap_2d_inexcl
        # ion balance tables are temperature x density x line no.
        selfunction.argtypes = [ct.c_int, ct.c_int, ct.c_float,\
                                np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(numin * 3,)),\
                                np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(numin,)),\
                                ct.c_int, \
                                np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(numex * 3,)),\
                                np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(numex,)),\
                                ct.c_int, \
                                ct.c_int, \
                                ct.c_int, ct.c_int, ct.c_int,\
                                ct.c_int, ct.c_float,\
                                np.ctypeslib.ndpointer(dtype=ct.c_int, shape=(xpix * ypix,)) ]
        selfunction.restype = None
        
        res = selfunction(ct.c_int(xpix), ct.c_int(ypix), ct.c_float(pixsize),\
                          inpos.astype(ct.c_float),\
                          inrad.astype(ct.c_float),\
                          ct.c_int(numin),\
                          expos.astype(ct.c_float),\
                          exrad.astype(ct.c_float),\
                          ct.c_int(numex),\
                          ct.c_int(closest_normradius),\
                          ct.c_int(Axis1), ct.c_int(Axis2), ct.c_int(Axis3),\
                          ct.c_int(periodic), ct.c_float(size),\
                          outmap)
    if res is not None:
        raise RuntimeError('C function returned output %s'%res)
    outmap = outmap.reshape(xpix, ypix)
    return outmap.astype(bool)



def autoname_halomask(halocat, xpix, radius_200c, radius_pkpc, closest_normradius,\
                      selection_in, selection_ex, axis):
    with h5py.File(halocat, 'r') as hc:
        simnum = hc['Header'].attrs['simnum']
        snapnum = hc['Header'].attrs['snapnum']
        var = hc['Header'].attrs['var']
        
        if var == 'REFERENCE':
            varind = 'Ref'
        elif var == 'RECALIBRATED':
            varind = 'Recal'
        else:
            varind = var
        
    basename = 'mask_%s%s_%i_%ipix_%s-projection_totalbox'%(varind, simnum, snapnum, xpix, axis)
    if 'inclsatellites' in halocat:
        basename = basename + '_inclsatellites'
    
    if closest_normradius:
        normind = '_closest-normradius'
    else:
        normind = '_closest-absradius'
    if radius_200c is not None:
        radind = '_halosize-%s-R200c'%(radius_200c)
    else:
        radind = '_halosize-%s-pkpc'%(radius_pkpc)
        
    if selection_in is not None:
        if len(selection_in) > 0:
            selind_incl = sorted(selection_in, key=lambda tup: tup[0])
            selind_incl = '_selection-in_' + '_'.join(['-'.join(str(val) for val in tup) for tup in selind_incl])
        else:
            selind_incl = ''
    else:
        selind_incl = ''
    if selection_ex is not None:
        if len(selection_ex) > 0:
            selind_excl = sorted(selection_ex, key=lambda tup: tup[0])
            selind_excl = '_selection-ex_' + '_'.join(['-'.join(str(val) for val in tup) for tup in selind_excl])
        else:
            selind_excl = ''
    else:
        selind_excl = ''
        
    return basename + radind + normind + selind_incl + selind_excl

def gethalomask_fromhalocat(halocat, xpix, ypix=None,\
                            radius_r200=1., radius_pkpc=None, closest_normradius=True,\
                            selection_in=None, selection_ex=None,\
                            axis='z', outfile=None):
    '''
    whole box _|_ projection axis
    can use position in real/velocity space to select halos along the 
    projection axis
    
    radius_r200:     radius around galaxies to extract in units of R200c
                     float, or None -> use radius_pkpc instead
                     default: 1.
    radius_pkpc:     float or None; default None
                     ignored unless radius_r200 is None
                     radius around galaxies to extract in pkpc units
    selection_in/ex: tuple of catalogue entry, min. value, max. value (either
                     can be None)
                     special cases: galaxyid / groupid - name, list/array of 
                                    selected onjects to match
                                    'X', 'Y', 'Z', : position space edges 
                                    select on position [1, 2] range, using [3] 
                                    times R200c as a margin (units cMpc)
                                    'VX', 'VY', VZ': velocity space edgesstderr.runhistograms_30033-30038-1725153.33
                                    select on position [1, 2] range
                                    (units rest-frame km/s)
    '''                             
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
    
    with h5py.File(halocat, 'r') as fo:
        cosmopars = {key: fo['Header/cosmopars'].attrs[key] for key in fo['Header/cosmopars'].attrs.keys()}
        size = cosmopars['boxsize'] / cosmopars['h']
        
        inclsel = sh.selecthalos(fo, selection_in)
        if selection_ex is not None:
            exclsel = sh.selecthalos(fo, selection_ex)
        #m200s = np.array(fo['M200c_Msun'])
        #plt.hist(np.log10(m200s[inclsel]), bins=np.arange(7., 15.1, 0.5), log=True, label='included', alpha=0.5)
        #plt.hist(np.log10(m200s[exclsel]), bins=np.arange(7., 15.1, 0.5), log=True, label='excluded', alpha=0.5)
        #plt.legend()
        #plt.xlabel('log10 M200c [Msun]')
        #plt.show()
        
        #zs = np.array(fo['Zcop_cMpc'])
        #plt.hist(zs[inclsel], bins=np.arange(0., 25., 0.5), log=True, label='included', alpha=0.5)
        #plt.hist(zs[exclsel], bins=np.arange(0., 25., 0.5), log=True, label='excluded', alpha=0.5)
        #plt.legend()
        #plt.xlabel('Z [cMpc]')
        #plt.show()
        
            if np.any(np.logical_and(inclsel, exclsel)):
                raise RuntimeError('Included and excluded halo selections overlap.')
        
        pos = np.array([np.array(fo['Xcom_cMpc']), np.array(fo['Ycom_cMpc']), np.array(fo['Zcom_cMpc'])]).T
        if radius_r200 is not None:
            rad = radius_r200 * np.array(fo['R200c_pkpc']) * 1.e-3 / cosmopars['a']
        elif radius_pkpc is not None:
            rad = radius_pkpc * 1.e-3 / cosmopars['a'] * np.ones(pos.shape[0])
        else: 
            raise ValueError('radius_r200 or radius_pkpc must be a float > 0')
        indct = {'pos': pos[inclsel], 'rad': rad[inclsel]}
        if selection_ex is not None:
            exdct = {'pos': pos[exclsel], 'rad': rad[exclsel]}
        else:
            exdct = None
    pixsize = size / float(xpix)

    mask  = gethalomask_basic(xpix, size, pixsize, indct,\
                      ypix=ypix, exdct=exdct, closest_normradius=closest_normradius, axis=axis,
                      periodic=True)

    if outfile is None:
        return mask
    else:
        if outfile == 'auto':
            outfile = autoname_halomask(halocat, xpix, radius_r200, radius_pkpc, closest_normradius,\
                                        selection_in, selection_ex, axis)
        if '/' not in outfile:
            outfile = ol.ndir + outfile
        if outfile[-5:] != '.hdf5':
            outfile = outfile + '.hdf5'
        
        with h5py.File(outfile, 'a') as out, h5py.File(halocat, 'r') as fo:
            galid = np.array(fo['galaxyid'])
            fo.copy(fo['Header'], out, name='Header')
            sel = out.create_group('selection')
            sel_in = sel.create_group('included')
            print('Check 0')
            for sl in selection_in:
                name = sl[0] + '_set%i'
                counter = 0 
                while name%counter in sel_in.keys():
                    counter += 1
                sel_in.create_dataset(name%counter, data=np.array(sl[1:]).astype(np.float))
            if 'galaxyid' not in sel_in.keys():
                sel_in.create_dataset('galaxyid', data=galid[inclsel])
            print('Check 1')
            if exdct is not None:
                sel_ex = sel.create_group('excluded')
                for sl in selection_ex:
                    name = sl[0] + '_set%i'
                    counter = 0 
                    while name%counter in sel_ex.keys():
                        counter += 1
                    sel_ex.create_dataset(name%counter, data=np.array(sl[1:]).astype(np.float))
                if 'galaxyid' not in sel_ex.keys():
                    sel_ex.create_dataset('galaxyid', data=galid[exclsel])
            print('Check 2')
            sel.attrs.create('info', 'selections give min, max (, margin) values; nan comes from None, meaning upper/lower limit')
            sel.attrs.create('velocity_units', 'rest-frame km/s')
            gen = out.create_group('Header/parameters')
            if radius_r200 is not None:
                gen.attrs.create('radius_r200c', radius_r200)
            elif radius_pkpc is not None:
                gen.attrs.create('radius_pkpc', radius_pkpc)
            gen.attrs.create('closest_normradius', closest_normradius)
            gen.attrs.create('axis', axis)
            out.create_dataset('mask', data=mask)


def saveradialprofiles_hdf5_to_hdf5(infiles, halocat, rbins, yvals,\
                                    xunit='R200c', ytype='fcov',\
                                    galids=None, separateprofiles=False,\
                                    combinedprofile=True,\
                                    outfile=None, galsettag=None):
    '''
    save radial profiles from a file to a (smaller?) hdf5 file
    output organisation:
        top-level groups:
            - galset_%i: for lists of specific galaxy ids
              - option attribute galsettag for quick set recognition
            - galaxy_%i: for single galaxy ids
        mid-level:
            for galset groups:
                - galaxyid -> list of combined galaxy ids
            - pkpc_bins
            - R200c_bins
        lower-level:
            - binset%i
        datasets:
            - bin_edges (in pkpc or R200c depending on mid-level group)
            - <'fcov' or 'perc>'_<numerical value>:
                                 covering fractions at that value or
                                 percentiles at that value 
                                 depending on the lower-level group
    entries are added to the file as needed
    note that storing every galaxy id might make the process slow and the files
    large
    
    galids: None -> all galids in radial profile file are used
            array -> only those galids are used

    '''     
    print('Input to saveradialprofiles_hdf5_to_hdf5:')
    print('\tinfiles:\t%s'%infiles)
    print('\toutfile:\t%s'%outfile)
    print('\thalocat:\t%s'%halocat)
    print('\tgalsettag:\t%s'%galsettag)
    print('\trbins:\t%s'%rbins)
    print('\tyvals:\t%s'%yvals)
    print('\txunit:\t%s, \tytype:\t%s'%(xunit, ytype))
    print('\tseparateprofiles:\t%s, combinedprofile:\t%s'%(separateprofiles, combinedprofile))
    print('\tgalids:\t%s'%galids)
    
    if isinstance(infiles, str):
        infiles = [infiles]
    
    if outfile is None:
        if len(infiles) == 1:
            _infile = infiles[0] 
        else:
            raise ValueError('For multiple input files, the output file name (outfile) must be specified')
        outfile = _infile.split('/')[-1]
        outfile = _infile[:-5] if _infile[-5:] == '.hdf5' else _infile
        outfile = outfile + '_stored_profiles.hdf5'
    if '/' not in outfile:
        outfile = rdir + outfile
    
    
    infiles = [pdir + fn if '/' not in fn else fn for fn in infiles]    
    infiles = [fn + '.hdf5' if fn[-5:] != '.hdf5' else fn for fn in infiles]
    
    if '/' not in halocat:
        halocat = pdir + halocat
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
        
    if xunit == 'R200c':
        midname = 'R200c_bins'
    elif xunit == 'pkpc':
        midname = 'pkpc_bins'
    else:
        raise ValueError('xunit options are "pkpc" and "R200c", not %s'%xunit)
    
    if ytype == 'fcov':
        lowname_base = 'fcov_%s'
    elif ytype == 'perc':
        lowname_base = 'perc_%s'
    else:
        raise ValueError('ytype options are "fcov" and "perc", not %s'%ytype)
    
    # iterate over the input files; each galaxy is extracted once at most, from
    # the first file in the list containing it
    with h5py.File(halocat, 'r') as halos,\
         h5py.File(outfile, 'a') as fo:  
        galids_cat = np.array(halos['galaxyid'])
        galids_cat_set = set(galids_cat)
        
        if galids is None:
            galids = 'all'
        else:
            galids_todo = set(galids)
            galids_todo_all = np.array(list(galids))
        rqdct = {}
            
        for _infile in infiles:
            with h5py.File(_infile, 'r') as crdcat:
            
                # find the galaxies in this _infile
                galids_thisfile = crdcat.keys()
                galids_thisfile.remove('Header')
                if 'selection' in galids_thisfile:
                    galids_thisfile.remove('selection')
                galids_thisfile = {int(galid) for galid in galids_thisfile}
                
                # get the subset also present in the halo catalogue
                galids_usable_thisfile = galids_thisfile & galids_cat_set
                if len(galids_usable_thisfile) < len(galids_thisfile):
                    print('Warning: some galaxies in %s were not present in the halo catalogue %s are are not used'%(_infile, halocat))
                    print('Galaxyids:')
                    print(galids_thisfile - galids_cat_set)
                del galids_thisfile
                
                # get the halos from this file in the overall selection
                if isinstance(galids, str):
                    if galids == 'all':
                        galids_used_thisfile = galids_usable_thisfile 
                    else:
                        raise RuntimeError('Unexpected galids value %s'%galids)
                else:
                    galids_used_thisfile = galids_usable_thisfile & galids_todo
                    galids_todo -= galids_used_thisfile
                del galids_usable_thisfile
                
                # get selector for galaxy/halo properties
                galids_used_thisfile = list(galids_used_thisfile)
                galids_used_thisfile.sort()
                sel = np.array([np.where(galid == galids_cat)[0][0] for galid in galids_used_thisfile])
                
                # get the profiles
                if xunit == 'pkpc':
                    rscales = np.array(halos['R200c_pkpc'])[sel]
                else:
                    rscales = 1.               
                # should match the galaxy order in sel
                rqarrs = [np.array(crdcat[str(galid)]) for galid in galids_used_thisfile]
                if np.any(rscales != 1.):
                    rqarrs = [np.array([rqarrs[i][0] * rscales[i], rqarrs[i][1]]) for i in range(len(rqarrs))]
            
                
                rqdct.update({galids[i]: rqarrs[i] for i in range(len(galids))})
        
        if len(galids_todo) > 0:
            raise RuntimeError('Did not retrieve profiles for %i galaxies\nmissing:\n%s'%(len(galids_todo), galids_todo))
            
        if combinedprofile:
            rqdct.update({'all': np.concatenate([rqdct[gid] for gid in rqdct.keys()], axis=1)})
        if not separateprofiles:
            rqdct = {'all': rqdct['all']}
            
        #print(rqdct)
        
        if ytype == 'perc':    
            ydct = rqhists(rqdct, rbins, percentiles=yvals)
        else:
            ydct = rqgeqs(rqdct, rbins, values=np.array(yvals))
        #print(ydct)
        # store the data
        if separateprofiles:
            for gid in galids:
                topname_temp = 'galaxy_%i'%gid
                if topname_temp in fo.keys():
                    tgrp = fo[topname_temp]
                else:
                    tgrp = fo.create_group(topname_temp)
                    tgrp.create_dataset('galaxyid', data=np.array([gid]))
                if midname in tgrp.keys():
                    mgrp = tgrp[midname]
                else:
                    mgrp = tgrp.create_group(midname)
                
                binset_keys = mgrp.keys()
                if len(binset_keys) > 0:
                    maxbinentry = max([int(name.split('_')[-1]) for name in binset_keys])
                    samebins = np.array([np.all(rbins == np.array(mgrp[key + '/bin_edges']))\
                                         if len(rbins) == len(mgrp[key + '/bin_edges'])\
                                         else False for key in binset_keys])
                    if np.sum(samebins) > 0:
                        bkey = binset_keys[np.where(samebins)[0][0]]
                        bgrp = mgrp[bkey]
                    else:
                        bkey = 'binset_%i'%(maxbinentry + 1)
                        bgrp = mgrp.create_group(bkey) 
                        bgrp.create_dataset('bin_edges', data=rbins)
                else:
                    bgrp = mgrp.create_group('binset_0') 
                    bgrp.create_dataset('bin_edges', data=rbins)
                
                for yind in range(len(yvals)):
                    yval = yvals[yind]
                    lowname_temp = lowname_base%yval
                    if lowname_temp in bgrp.keys():
                        if np.all(ydct[gid][yind, :] == np.array(bgrp[lowname_temp])):
                            pass
                        else:
                            print('Issue arose for %s/%s/%s/%s in %s'%(topname_temp, midname, bkey, lowname_temp, outfile))
                            print('Newly calculated array:')
                            print(ydct[gid][yind, :])
                            raise RuntimeError('For galaxy %i, y value %s, a stored array did not match the newly calculated version'%(gid, yval))
                    else:
                        bgrp.create_dataset(lowname_temp, data=ydct[gid][yind, :])
        
        if combinedprofile:
            print('Saving combined profile')
            topname_base = 'galset_%i'
            topkeys = fo.keys()
            topkeys = set(key if 'galset' in key else None for key in topkeys)
            if None in topkeys:
                topkeys.remove(None)
            topkeys = list(topkeys)
            if len(topkeys) > 0:
                maxtopentry = max([int(name.split('_')[-1]) for name in topkeys])
                # galaxy ids are sorted -> will be in the same order every time this is run
                samegals = np.array([np.all(galids == np.array(fo[key + '/galaxyid']))\
                                             if len(galids) == len(fo[key + '/galaxyid'])\
                                             else False for key in topkeys])
                if np.sum(samegals) > 0:
                    topname_temp = topkeys[np.where(samegals)[0][0]]
                    tgrp = fo[topname_temp]
                else:
                    galsetind = maxtopentry + 1
                    topname_temp = topname_base%(galsetind)
                    tgrp = fo.create_group(topname_temp) 
                    tgrp.create_dataset('galaxyid', data=np.array(sorted(galids_todo_all)))
            else:
                galsetind = 0
                topname_temp = topname_base%galsetind
                tgrp = fo.create_group(topname_temp) 
                tgrp.create_dataset('galaxyid', data=np.array(sorted(galids_todo_all)))
            
            if 'seltag' not in tgrp.keys():
                if galsettag is not None:
                    tgrp.attrs.create('seltag', galsettag)
            elif galsettag is not None:
                oldtag = tgrp.attrs['seltag']
                if oldtag == galsettag:
                    pass
                else:
                    print('Galaxy set %s is already saved as %s'%(galsettag, oldtag))
                
            if midname in tgrp.keys():
                mgrp = tgrp[midname]
            else:
                mgrp = tgrp.create_group(midname)
            #print('Checkpoint 1')
            binset_keys = mgrp.keys()
            if len(binset_keys) > 0:
                #print('Checkpoint case 2a')
                maxbinentry = max([int(name.split('_')[-1]) for name in binset_keys])
                samebins = np.array([np.all(rbins == np.array(mgrp[key + '/bin_edges']))\
                                             if len(rbins) == len(mgrp[key + '/bin_edges'])\
                                             else False for key in binset_keys])
                if np.sum(samebins) > 0:
                    bkey = binset_keys[np.where(samebins)[0][0]]
                    bgrp = mgrp[bkey]
                else:
                    bkey = 'binset_%i'%(maxbinentry + 1)
                    bgrp = mgrp.create_group(bkey) 
                    bgrp.create_dataset('bin_edges', data=rbins)
            else:
                #print('Checkpoint case 2b')
                bgrp = mgrp.create_group('binset_0') 
                bgrp.create_dataset('bin_edges', data=rbins)
            #print(yvals)
            for yind in range(len(yvals)):
                yval = yvals[yind]
                lowname_temp = lowname_base%yval
                #print(yval)
                if lowname_temp in bgrp.keys():
                    #print('Checkpoint case 3a (in loop)')
                    if np.all(ydct['all'][yind, :] == np.array(bgrp[lowname_temp])):
                        pass
                    else:
                        #print('Issue arose for %s/%s/%s/%s in %s'%(topname_temp, midname, bkey, lowname_temp, outfile))
                        #print('Newly calculated array:')
                        #print(ydct['all'][yind, :])
                        raise RuntimeError('For galaxy set %i, y value %s, a stored array did not match the newly calculated version'%(galsetind, yval))
                else:
                    #print('Checkpoint case 3b (in loop)')
                    #print('Creating dataset...')
                    bgrp.create_dataset(lowname_temp, data=ydct['all'][yind, :])    
    return ydct

def get_radprof(rqfilenames, halocat, rbins, yvals,\
                xunit='R200c', ytype='fcov',\
                galids=None, combinedprofile=True,\
                separateprofiles=False,\
                rpfilename=None, galsettag=None):
    '''
    retrieve profiles if they're stored, calculate and store them otherwise
    '''
    
    if isinstance(rqfilenames, str):
        rqfilenames = [rqfilenames]
    
    if rpfilename is None:
        if len(rqfilenames) == 1:
            rqfilename = rqfilenames[0]
        else:
            raise ValueError('If the input property-r distributions are stored over multiple files, the file for radial profiles (rpfilename) must be specified')
        if '/' in rqfilename:
            rdir = '/'.join(rqfilename.split('/')[:-1]) + '/'
        else:
            rdir = ol.pdir + 'radprof/'
        rqfilename = rqfilename.split('/')[-1]
        rpfilename = rqfilename[:-5] if rqfilename[-5:] == '.hdf5' else rqfilename
        rpfilename = rpfilename + '_stored_profiles.hdf5'
    if '/' not in rpfilename:
        rdir = ol.pdir + 'radprof/'
        rpfilename = rdir + rpfilename
    
    rqfilenames = [pdir + fn if '/' not in fn else fn for fn in rqfilenames]
    rqfilenames = [fn + '.hdf5' if fn[-5:] != '.hdf5' else fn for fn in rqfilenames]

    if xunit == 'R200c':
        midname = 'R200c_bins'
    elif xunit == 'pkpc':
        midname = 'pkpc_bins'
    else:
        raise ValueError('xunit options are "pkpc" and "R200c", not %s'%xunit)
    
    if ytype == 'fcov':
        lowname_base = 'fcov_%s'
    elif ytype == 'perc':
        lowname_base = 'perc_%s'
    else:
        raise ValueError('ytype options are "fcov" and "perc", not %s'%ytype)
    
    # create file
    if not os.path.isfile(rpfilename):
        ft = h5py.File(rpfilename, 'w')
        ft.close()
        
    with h5py.File(rpfilename, 'r') as fo:
        
        runcombined = False
        runseparate = False
        ydct = {}
        
        if combinedprofile:
            try: 
                topkeys = fo.keys()
                topkeys = set(key if 'galset' in key else None for key in topkeys)
                if None in topkeys:
                    topkeys.remove(None)
                topkeys = list(topkeys)
                
                if len(topkeys) == 0:
                    raise DatasetNotFound('No combined profiles for galids %s'%galids)  
                    # galaxy ids are sorted -> will be in the same order every time this is run
                samegals = np.array([np.all(galids == np.array(fo[key + '/galaxyid']))\
                                         if len(galids) == len(fo[key + '/galaxyid'])\
                                         else False for key in topkeys])
                if np.sum(samegals) == 0:
                    raise DatasetNotFound('No combined profiles for galids %s'%galids) 
                    
                topname_temp = topkeys[np.where(samegals)[0][0]]
                tgrp = fo[topname_temp]
            
                if midname not in tgrp.keys():
                    raise DatasetNotFound('No %s present for combined galids %s'%(midname, galids)) 

                mgrp = tgrp[midname]            
                binset_keys = mgrp.keys()
                if len(binset_keys) == 0:
                    raise DatasetNotFound('No bins present for combined galids %s, %s'%(galids, midname)) 
                
                samebins = np.array([np.all(rbins == np.array(mgrp[key + '/bin_edges']))\
                                             if len(rbins) == len(mgrp[key + '/bin_edges'])\
                                             else False for key in binset_keys])
                if np.sum(samebins) == 0:
                    raise DatasetNotFound('bins %s not present for combined galids %s, %s'%(rbins, galids, midname))
                    
                bkey = binset_keys[np.where(samebins)[0][0]]
                bgrp = mgrp[bkey]
                
                if np.all([lowname_base%yval in bgrp.keys() for yval in yvals]):
                    yarr = np.ones((len(yvals), len(rbins) - 1)) * np.NaN # flag missing values
                    for yind in range(len(yvals)):
                        yval = yvals[yind]
                        lowname_temp = lowname_base%yval
                        yarr[yind, :] = np.array(bgrp[lowname_temp])
                    ydct.update({'all': yarr})
                else:
                    raise DatasetNotFound('not all y values %s present for combined galids %s, %s, bins %s '%(yvals, galids, midname, rbins))
                    
            except DatasetNotFound:
                runcombined = True
            
        galids_seprun = set()
        if separateprofiles:
            topkeys = fo.keys()
            topkeys = set(key if 'galaxy' in key else None for key in topkeys)
            if None in topkeys:
                topkeys.remove(None)
            
            galids_lookup = topkeys & set(galids)
            galids_seprun = set(galids) - topkeys
            
            mgrp_present = set(gid if 'galaxy_%i/%s'%(gid, midname) in fo else None for gid in galids_lookup)
            
            galids_seprun |= (set(galids_lookup) - mgrp_present)
            galids_lookup &= mgrp_present
            galids_lookup_old = np.copy(galids_lookup)
            
            if galids_lookup_old.shape == ():
                galids_lookup_old = np.array([])
            for gid in galids_lookup_old:
                mgrp = fo['galaxy_%i/%s'%(gid, midname)]
                binset_keys = mgrp.keys()
                
                if len(binset_keys) == 0:
                    galids_lookup -= {gid}
                    galids_seprun |= {gid}
                    continue
                
                samebins = np.array([np.all(rbins == np.array(mgrp[key + '/bin_edges']))\
                                             if len(rbins) == len(mgrp[key + '/bin_edges'])\
                                             else False for key in binset_keys])
                if np.sum(samebins) == 0:
                    galids_lookup -= {gid}
                    galids_seprun |= {gid}
                    continue
                
                bkey = binset_keys[np.where(samebins)[0][0]]
                bgrp = mgrp[bkey]
            
                if np.all([lowname_base%yval in bgrp.keys() for yval in yvals]):
                    yarr = np.emtpy((len(rbins) - 1, len(yvals)))
                    for yind in range(len(yvals)):
                        yval = yvals[yind]
                        lowname_temp = lowname_base%yval
                        yarr[yind, :] = np.array(bgrp[lowname_temp])
                    ydct.update({gid: yarr})
                else:
                    galids_lookup -= {gid}
                    galids_seprun |= {gid}
    
    if len(galids_seprun) > 0:
        galids_seprun = np.array(list(galids_seprun))
        runseparate = True
        
    if runcombined or (runseparate and set(galids) == set(galids_seprun)):
        ydct_temp = saveradialprofiles_hdf5_to_hdf5(rqfilenames, halocat,\
                                    rbins, yvals,\
                                    xunit=xunit, ytype=ytype,\
                                    galids=galids, separateprofiles=runseparate,\
                                    combinedprofile=runcombined,\
                                    outfile=rpfilename, galsettag=galsettag)
        ydct.update(ydct_temp)
    elif runseparate:
        ydct_temp = saveradialprofiles_hdf5_to_hdf5(rqfilenames, halocat,\
                                    rbins, yvals,\
                                    xunit=xunit, ytype=ytype,\
                                    galids=galids_seprun, separateprofiles=runseparate,\
                                    combinedprofile=False,\
                                    outfile=rpfilename, galsettag=galsettag)
        ydct.update(ydct_temp)
        
    return ydct

def getstats(vlist, ytype='perc', yvals=[50.]):
    '''
    get ytype/yval statistics for each entry of vlist
    returns a list of vlist-length lists
    index 0: yval
    index 1: index in vlist
    index 0 is left out for ytype 'mean'
    '''
    if ytype == 'mean':
        out = [np.average(vals) for vals in vlist]
    elif ytype == 'perc':
        out = [np.percentile(vals, yvals) for vals in vlist]
        out = [[out[j][i] for j in range(len(vlist))]\
                for i in range(len(yvals))]
    elif ytype == 'fcov':
        out = [[np.sum(np.array(vals) >= yval) for vals in vlist]\
                for yval in yvals]
        out = [[float(ysub[i]) / float(len(vlist[i])) for i in range(len(vlist))]\
                for ysub in out]
    return out

def getprofiles_fromstamps(filenames, rbins, galids,\
                           runit='pkpc', ytype='perc', yvals=50.,\
                           halocat=None,\
                           separateprofiles=False, uselogvals=True,\
                           outfile=None, grptag=None):
    '''    
    input:
    ------
    filenames:   list of hdf5 files containing the stamps. Assumes the groups 
                 are named by their central galaxy ids. A single file is also 
                 ok. If a galaxyid is present in more than one file, the first
                 file in the list is used.
    rbins:       bin edges for statistics extraction
    runit:       unit the rbins are given in (string): 'pkpc' or 'R200c'
    halocat:     halo catalogue (filename; str). Used to look up R200c and halo 
                 centres
    galids:      galaxyids to get the profiles from (int or string)
    ytype:       what kind of statistic to extract (string):
                 'perc' for percentiles, 'fcov' for covering fractions, or 
                 'mean' for the average
    yvals:       if ytype is 'perc' or 'fcov': the values to get the covering
                 fractions (same units as the map) or percentiles (0-100 range)
                 for (float)
                 ignored for ytype 'mean'
    separateprofiles: (bool) get a profile for each galaxyid (True) or for all
                 of them together (False)
    outfile:     (string or None) name of the output file. constructed from the
                 input filename if None and only one input filename is given
    grptag:      tag for the autonamed output group for the profile. Useful to
                 indicate how the galaxy ids were selected
    uselogvals:  (bool) if True use log y values in the output (and interpret 
                 type='fcov' yvals as log). Otherwise, non-log y values are 
                 used in every case. (Just following what's in the input files 
                 is not an option.) 
             
    output:
    -------
    profiles stored in outfile matching the format of the hdf5 file profile 
    retrieval script. Additions to that format include the ytype='mean' option
    and storing whether the values are log10 or not.
    '''
    if '.' in filenames:
        filenames = [filenames]
    filenames = [ol.pdir + filename if '/' not in filename else filename\
                 for filename in filenames]
    
    if outfile is None:
        if len(filenames) > 1:
            raise ValueError('outfile must e specified if there is more than '+\
                             'one input file')
        else:
            outfile = filenames[0].split('/')[-1]
            outfile = rdir + 'radprof_' + outfile 
            
    files = [h5py.File(filename, 'r') for filename in filenames]
    galids = np.array(sorted([int(galid) for galid in galids]))
    
    # halo catalogue data handling
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    with h5py.File(halocat, 'r') as hc:
        cosmopars = {key: val for key, val in hc['Header/cosmopars'].attrs.items()}
    
        # check if cosmopars match (could still be different sim. boxes, but it's worth doing a simple check)
        for _file in files:
            fills = _file['Header'].attrs['filename_fills']
            if fills[0].decode() == '[':  # parse stringified list of strings, due to error in saving function...
                fills = fills.decode()
                if fills[0] == '[' and fills[-1] == ']': # it's a string-saved list -> parse as such
                    fills = fills[2:-2]
                    if "', '" in fills:
                        fills = fills.split("', '")
                    elif "' '" in fills:
                        fills = fills.split("' '")
                    elif "''" in fills:
                        fills = fills.split("''")
                    else:
                        raise RuntimeError('filenames_fills in the file Header saved in an unrecognized way:\n{}'.format(fills))
                    _fills = []
                    for fill in fills:
                        if '\n' in fill:
                            fparts = fill.split('\n')
                            fparts = [fp.strip() for fp in fparts]                     
                            fparts = [fp[1:] if fp[0] == "'" else fp for fp in fparts]
                            fparts = [fp[:-1] if fp[-1] == "'" else fp for fp in fparts]
                            fparts = [fp.strip() for fp in fparts]
                            _fills += fparts
                        elif '\t' in fill:
                            fparts = fill.split('\t')
                            fparts = [fp.strip() for fp in fparts]
                            fparts = [fp[1:] if fp[0] == "'" else fp for fp in fparts]
                            fparts = [fp[:-1] if fp[-1] == "'" else fp for fp in fparts]
                            fparts = [fp.strip() for fp in fparts]
                            _fills += fparts
                        else:
                             _fills.append(fill)
                    fills = _fills
                else:
                    raise RuntimeError('filenames_fills in the file Header saved in an unrecognized way:\n{}'.format(fills))
            else: 
                fills = [fill.decode() for fill in fills]
            #for fill in fills:
            #    st = 'Header/{fill}/inputpars/cosmopars'.format(fill=fill)
            #    print('For {fill}, cosmopars found {}'.format(st in _file, fill=fill))
            
            _cps = [{key: val for key, val in \
                    _file['Header/{fill}/inputpars/cosmopars'.format(fill=fill)].attrs.items()}\
                    for fill in fills]
            if not np.all([np.all([np.isclose(_cp[key], cosmopars[key]) for key in cosmopars])\
                           for _cp in _cps]):
                msg = 'Cosmopars recorded for a slice in file {stamps} did not match those in the halo catalcoge {hc}'
                raise RuntimeError(msg.format(stamps=_file.filename, hc=halocat))
                
        # look up required data for the galaxy ids
        gid_cat = hc['galaxyid'][:]
        galaxyids = [int(galid) for galid in galids]
        inds_gid = np.array([np.where(gid_cat == int(galid))[0][0] \
                             for galid in galaxyids])
        
        R200c_cMpc = hc['R200c_pkpc'][:] * (1e-3 / cosmopars['a'])
        cen_simx_cMpc = hc['Xcom_cMpc'][:]
        cen_simy_cMpc = hc['Ycom_cMpc'][:]
        cen_simz_cMpc = hc['Zcom_cMpc'][:]
        
        R200c_cMpc = R200c_cMpc[inds_gid]
        cen_simx_cMpc = cen_simx_cMpc[inds_gid]
        cen_simy_cMpc = cen_simy_cMpc[inds_gid]
        cen_simz_cMpc = cen_simz_cMpc[inds_gid]
        
    # get header info per file
    # get list of galids per file
    # loop over files, contained galids
    # get requested info
    # store info
    rbins2 = np.array(rbins)**2 # faster than taking sqrt of all the distances
    if not hasattr(yvals, '__len__'):
        yvals = [yvals]
    yvals = np.sort(yvals)
    
    with h5py.File(outfile, 'a') as fo:
        if separateprofiles:
            gn0 = 'galaxy_{galid}'
        else:
            # galaxy sets: named galset_<int>
            galsets = [key if 'galset' in key else None \
                       for key in fo.keys()]
            galsets = list(set(galsets) - {None})
            galsets.sort(key=lambda x: int(x.split('_')[-1])) 
            anymatch = False
            for galset in galsets:
                _galids = fo[galset + '/galaxyid'][:]
                if len(_galids) == len(galids):
                    if np.all(_galids == galids):
                        gn0 = galset
                        anymatch = True
                        break
            if not anymatch:
                i = 0
                while 'galset_{i}'.format(i=i) in fo:
                    i += 1
                gn0 = 'galset_{i}'.format(i=i)
                _g = fo.create_group(gn0)
                _g.create_dataset('galaxyid', data=galids)
                if grptag is not None:
                    _g.attrs.create('seltag', np.string_(grptag))
        
        outgroup_base = '{gal}/{runit}_bins'.format(gal=gn0, runit=runit)
        
        if not separateprofiles:
            binvlist_all = [[] for i in range(1, len(rbins2))]
            
        for gind_cat, galid in enumerate(list(galids)):
            # print(galid)
            grn = str(galid)
            fileuse = np.where([grn in file for file in files])[0]
            if len(fileuse) == 0:
                raise RuntimeError('Galaxyid {} not found in any file'.format(galid))
            fileuse = fileuse[0]
            _file = files[fileuse]
            gind = np.where(int(galid) == _file['Header/labels'][:])[0][0]
            
            stamp = _file[grn][:]
            # not taken modulo anything, just centre - size, so can be used to 
            # calculate distance to centre without modulo math
            llc_cMpc = _file['Header/lower_left_corners_cMpc'][gind]
            pixsize_cMpc0 = _file['Header'].attrs['pixel_size_x_cMpc']
            pixsize_cMpc1 = _file['Header'].attrs['pixel_size_y_cMpc']
            axis = _file['Header'].attrs['axis'].decode()
            logval = bool(_file['Header'].attrs['logvalues'])
            period = cosmopars['boxsize'] / cosmopars['h']
            #print(logval)
            if axis == 'z':
                cen0 = cen_simx_cMpc[gind_cat]
                cen1 = cen_simy_cMpc[gind_cat]
            elif axis == 'x':
                cen0 = cen_simy_cMpc[gind_cat]
                cen1 = cen_simz_cMpc[gind_cat]
            elif axis == 'y':
                cen0 = cen_simz_cMpc[gind_cat]
                cen1 = cen_simx_cMpc[gind_cat]
            
            pos = np.indices(stamp.shape)
            pos0 = pos[0]
            pos1 = pos[1]
            delta0 = (llc_cMpc[0] + (pos0 + 0.5) * pixsize_cMpc0 - cen0 \
                      + 0.5 * period) % period - 0.5 * period
            delta1 = (llc_cMpc[1] + (pos1 + 0.5) * pixsize_cMpc1 - cen1 \
                      + 0.5 * period) % period - 0.5 * period
            dist2 = delta0**2 + delta1**2
            
            if runit == 'R200c':
                R200c = R200c_cMpc[gind_cat]
                dist2 *= (1. / R200c**2)
            elif runit == 'pkpc':
                dist2 *= (1e3 * cosmopars['a'])**2
            # check distance coverage of the stamp (assumes centres match approximately)
            hi0 = stamp.shape[0] // 2
            hi1 = stamp.shape[1] // 2
            #print(dist2[-1, -1], dist2[-1, hi1], dist2[hi0, -1])
            #print(rbins2[-2])
            if rbins2[-1] <= dist2[-1, hi1] and rbins2[-1] <= dist2[hi0, -1]:
                pass
            elif rbins2[-2] < dist2[-1, -1]:
                print('Large radial bins will not include full azimuthal sampling for galaxy {}'.format(galid))
            else:
                raise RuntimeError('Large radial bins are entirely unsampled for galaxy {}'.format(galid))
                
            dist2 = dist2.flatten()
            vals = stamp.flatten()
            if (not uselogvals) and logval:
                vals = 10**vals
            elif uselogvals and (not logval):
                vals = np.log10(vals)
            inds = np.digitize(dist2, rbins2)
            binvlist = [list(vals[inds==i]) for i in range(1, len(rbins2))]
            #numpix = sum([len(sl) for sl in binvlist])
            
            if separateprofiles:
                #print('Saving data for {}'.format(galid))
                profiles = getstats(binvlist, ytype=ytype, yvals=yvals)
                
                outgroup = outgroup_base.format(galid=galid)
                if outgroup in fo:
                    g0 = fo[outgroup]
                else:
                    g0 = fo.create_group(outgroup)
                if grptag is not None:
                    galgrp = outgroup.split('/')[0]
                    if galgrp not in fo: 
                        _g = fo.create_group(galgrp)
                        _g.attrs.create('seltag', np.string_(grptag))
                # naming: binset_<index>
                binsets = list(g0.keys())
                binsets.sort(key=lambda x: int(x.split('_')[-1])) 
                bmatch = False
                for binset in binsets:
                    bin_edges = g0[binset + '/bin_edges']
                    if len(bin_edges) == len(rbins):
                        if np.allclose(bin_edges, rbins):
                            bmatch = True
                            bgrp = g0[binset]
                            break
                if not bmatch:
                    i = 0
                    while 'binset_{i}'.format(i=i) in g0:
                        i += 1
                    bgrp = g0.create_group('binset_{i}'.format(i=i))
                    bgrp.create_dataset('bin_edges', data=rbins)
                    
                if ytype == 'mean':
                    if uselogvals:
                        dsname = 'mean_log'
                    else:
                        dsname = 'mean'
                    if dsname in bgrp:
                        pass # already saved
                    else:
                        ds = bgrp.create_dataset(dsname, data=np.array(profiles))
                        ds.attrs.create('logvalues', uselogvals)
                else:
                    for ind, yval in enumerate(yvals):
                        dsname = '{ytype}_{yval}'.format(ytype=ytype, yval=yval)
                        if dsname in bgrp:
                            continue
                        else:
                             ds = bgrp.create_dataset(dsname, data=np.array(profiles[ind]))
                             ds.attrs.create('logvalues', uselogvals)
            else: # not separateprofiles                
                binvlist_all = [binvlist_all[i] + binvlist[i] for i in range(len(binvlist_all))]
        
        if not separateprofiles:
            binvlist_all = [np.array(vlist) for vlist in binvlist_all]
            profiles = getstats(binvlist_all, ytype=ytype, yvals=yvals)
            
            outgroup = outgroup_base.format(galid=galid)
            if grptag is not None:
                galgrp = outgroup.split('/')[0]
                if galgrp not in fo: 
                    print('Adding seltag')
                    _g = fo.create_group(galgrp)
                    _g.attrs.create('seltag', np.string_(grptag))
            if outgroup in fo:
                g0 = fo[outgroup]
            else:
                g0 = fo.create_group(outgroup)
            # naming: binset_<index>
            binsets = list(g0.keys())
            binsets.sort(key=lambda x: int(x.split('_')[-1])) 
            bmatch = False
            for binset in binsets:
                bin_edges = g0[binset + '/bin_edges']
                if len(bin_edges) == len(rbins):
                    if np.allclose(bin_edges, rbins):
                        bmatch = True
                        bgrp = g0[binset]
                        break
            if not bmatch:
                i = 0
                while 'binset_{i}'.format(i=i) in g0:
                    i += 1
                bgrp = g0.create_group('binset_{i}'.format(i=i))
                bgrp.create_dataset('bin_edges', data=rbins)
                
            if ytype == 'mean':
                if uselogvals:
                    dsname = 'mean_log'
                else:
                    dsname = 'mean'
                if dsname in bgrp:
                    pass # already saved
                else:
                    ds = bgrp.create_dataset(dsname, data=np.array(profiles))
                    ds.attrs.create('logvalues', uselogvals)
            else:
                for ind, yval in enumerate(yvals):
                    dsname = '{ytype}_{yval}'.format(ytype=ytype, yval=yval)
                    if dsname in bgrp:
                        continue
                    else:
                         ds = bgrp.create_dataset(dsname, data=np.array(profiles[ind]))
                         ds.attrs.create('logvalues', uselogvals)
    
    [file.close() for file in files]
    return None


def combineprofiles(filenames, rbins, galids,
                    runit='pkpc', ytype_in='mean', yvals_in=50.,
                    ytype_out='perc', yvals_out=50.,
                    uselogvals=True,
                    outfile=None, grptag=None):
    '''    
    starting from individual galaxy profiles, get ensemble statistics
    
    input:
    ------
    filenames:   list of hdf5 files containing the profiles. Assumes the 
                 groups are named by their central galaxy ids ('galaxy_<id>').
                 A single file is also ok. If a galaxyid is present in more 
                 than one file, the first file in the list is used.
    rbins:       bin edges for statistics extraction (checked against stored
                 values)
    runit:       unit the rbins are given in (string): 'pkpc' or 'R200c'
    galids:      galaxyids to combine the profiles from (int or string 
                 list-like)
    ytype_in/out: what kind of statistic to extract (string)
                 in are the galaxy profiles, out is how the values in each bin
                 are combined:
                 'perc' for percentiles, 'fcov' for covering fractions, or 
                 'mean' for the average
    yvals_in/out: if ytype is 'perc' or 'fcov': the values to get the covering
                 fractions (same units as the map) or percentiles (0-100 range)
                 for (float)
                 ignored for ytype 'mean'
                 for in, only a single value is accepted
    outfile:     (string or None) name of the output file. same as the
                 input filename if None and only one input filename is given
    grptag:      tag for the autonamed output group for the profile. Useful to
                 indicate how the galaxy ids were selected
    uselogvals:  (bool) if True use log y values in the output (and interpret 
                 type='fcov' yvals as log). Otherwise, non-log y values are 
                 used in every case. (Just following what's in the input files 
                 is not an option.) 
                 if 'logvalues' aren't recorded, it's assumed they match 
                 uselogvalues
             
    output:
    -------
    profiles stored in outfile matching the format of the hdf5 file profile 
    retrieval script. Additions to that format include the ytype='mean' option
    and storing whether the values are log10 or not.
    
    profiles are stored as <ytype_out>[_<yval_out>]_from_<ytype_in>[_<yval_in>]
    '''
    
    if '.' in filenames:
        filenames = [filenames]
    filenames = [ol.pdir + filename if '/' not in filename else filename\
                 for filename in filenames]
    
    if outfile is None:
        if len(filenames) > 1:
            raise ValueError('outfile must be specified if there is more than '+\
                             'one input file')
        else:
            outfile = filenames[0]
            
    files = [h5py.File(filename, 'r') for filename in filenames]
    galids = np.array(sorted([int(galid) for galid in galids]))
    
    # groundwork for input profiles
    binpart = runit + '_bins'
    pathtemplate = 'galaxy_{galaxyid}/' + binpart 
    if ytype_in == 'mean':
        profn = ytype_in
    elif ytype_in == 'perc':
        if hasattr(yvals_in, '__len__'):
            raise ValueError('Only a single value is allowed for yvals_in')
        profn = 'perc_{val}'.format(val=yvals_in)
    elif ytype_in == 'fcov':
        profn = None # check log value before comparing name
    
    # read in input profiles
    proflist = []
    for galid in galids:
        found = False
        searchpath = pathtemplate.format(galaxyid=galid)
        for file in files:
            if searchpath not in file:
                continue
            sgrp = file[searchpath]
            bingrns = sgrp.keys()
            for bingrn in bingrns:
                edges = sgrp[bingrn]['bin_edges'][:]
                if len(edges) != len(rbins):
                    continue
                if not np.allclose(edges, rbins):
                    continue
                
                if profn is None: # matching yvals_in depends on uselogvalues
                    dsnames = list(sgrp[bingrn].keys())
                    for name in dsnames:
                        if dsname.startswith(ytype_in):
                            _yval = float(dsname.split('_'))
                            if 'logvalues' in sgrp[bingrn][dsname].attrs:
                                _logv = bool(sgrp[bingrn][dsname].attrs['logvalues'])
                            else: # assume same as asked here
                                _logv = uselogvals
                            if _logv and not uselogvals:
                                _yval = 10**_yval
                            elif uselogvals and not _logv:
                                _yval = np.log10(_yval)
                            
                            if np.isclose(_yval, yvals_in):
                                found = True
                                profile = sgrp[bingrn][dsname][:]
                                if _logv and not uselogvals:
                                    profile = 10**profile
                                elif uselogvals and not _logv:
                                    profile = np.log10(profile)
                                
                                proflist.append(profile)
                                break
                else:
                    if profn in sgrp[bingrn]:
                        profile = sgrp[bingrn][profn][:]
                        if 'logvalues' in sgrp[bingrn][profn].attrs:
                            _logv = bool(sgrp[bingrn][profn].attrs['logvalues'])
                        else: # assume same as asked here
                            _logv = uselogvals
                        if _logv and not uselogvals:
                            profile = 10**profile
                        elif uselogvals and not _logv:
                            profile = np.log10(profile)
                        proflist.append(profile) 
                        found = True
                        
                if found: # binset loop
                    break
            if found: # file loop
                break
        if not found:
            msg = 'No profile for these files, bins and profile type was found'
            msg += 'for galaxy {}'.format(galid)
            msg += '\nbins: {rbins} {runit}'.format(rbins=rbins, runit=runit)
            msg += '\nprofile: {ytype_in} {yvals_in}'.format(ytype_in=ytype_in,
                                                             yvals_in=yvals_in)
            raise RuntimeError(msg)
            
    [file.close() for file in files]
    
    if len(galids) != len(proflist):
        raise RuntimeError('recovered {} profiles for {} galaxies'.format(
            len(proflist), len(galids)))
    proflist = np.array(proflist)
    
    # input is list of values to get stats from
    if not hasattr(yvals_out, '__len__'):
        yvals = [yvals_out]
    yvals_out = np.sort(yvals_out)
    outprofs = getstats(proflist.T, ytype=ytype_out, yvals=yvals_out)

    
    with h5py.File(outfile, 'a') as fo:
        # galaxy sets: named galset_<int>
        galsets = [key if 'galset' in key else None \
                   for key in fo.keys()]
        galsets = list(set(galsets) - {None})
        galsets.sort(key=lambda x: int(x.split('_')[-1])) 
        anymatch = False
        for galset in galsets:
            _galids = fo[galset + '/galaxyid'][:]
            if len(_galids) == len(galids):
                if np.all(_galids == galids):
                    gn0 = galset
                    anymatch = True
                    break
        if not anymatch:
            i = 0
            while 'galset_{i}'.format(i=i) in fo:
                i += 1
            gn0 = 'galset_{i}'.format(i=i)
            _g = fo.create_group(gn0)
            _g.create_dataset('galaxyid', data=galids)
            if grptag is not None:
                _g.attrs.create('seltag', np.string_(grptag))
        
        outgroup_base = '{gal}/{runit}_bins'.format(gal=gn0, runit=runit)
        outgroup = outgroup_base.format(galid=galid)
        
        if grptag is not None:
            galgrp = outgroup.split('/')[0]
            if galgrp not in fo: 
                print('Adding seltag')
                _g = fo.create_group(galgrp)
                _g.attrs.create('seltag', np.string_(grptag))
        if outgroup in fo:
            g0 = fo[outgroup]
        else:
            g0 = fo.create_group(outgroup)
        # naming: binset_<index>
        binsets = list(g0.keys())
        binsets.sort(key=lambda x: int(x.split('_')[-1])) 
        bmatch = False
        for binset in binsets:
            bin_edges = g0[binset + '/bin_edges']
            if len(bin_edges) == len(rbins):
                if np.allclose(bin_edges, rbins):
                    bmatch = True
                    bgrp = g0[binset]
                    break
        if not bmatch:
            i = 0
            while 'binset_{i}'.format(i=i) in g0:
                i += 1
            bgrp = g0.create_group('binset_{i}'.format(i=i))
            bgrp.create_dataset('bin_edges', data=rbins)
        
        
        if ytype_in == 'mean':
            if uselogvals:
                _ynin = 'mean_log'
            else:
                _ynin = 'mean'
        else:
            _ynin = '{ytype}_{yval}'.format(ytype=ytype_in, yval=yvals_in)
        
        if ytype_out == 'mean':
            if uselogvals:
                _ynout = 'mean_log'
            else:
                _ynout = 'mean'
                
            dsname = _ynout + '_from_' + _ynin
            
            if dsname in bgrp:
                pass # already saved
            else:
                ds = bgrp.create_dataset(dsname, data=np.array(profiles))
                ds.attrs.create('logvalues', uselogvals)
        else:
            for ind, yval in enumerate(yvals_out):
                dsname = '{ytype}_{yval}'.format(ytype=ytype_out, yval=yval)
                dsname += + '_from_' + _ynin
                if dsname in bgrp:
                    continue
                else:
                     ds = bgrp.create_dataset(dsname, data=np.array(outprofs[ind]))
                     ds.attrs.create('logvalues', uselogvals)

    return None




#################################################################
# plots investigating the pet halos and convergence of profiles #  
#################################################################
petcolors = {'med': 'black',\
             'lo_mbh': 'pink',\
	         'hi_mbh': 'firebrick',\
	         'lo_ssfr': 'lime',\
	         'hi_ssfr': 'forestgreen',\
	         'lo_mst': 'cyan',\
	         'hi_mst': 'blue'}

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

gray_dark = truncate_colormap(plt.get_cmap('gist_gray'), maxval=0.6)
pink_dark = truncate_colormap(plt.get_cmap('pink'), minval=0.1, maxval=0.6)

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

def suplabel(fig, axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None: 
        label_prop = dict()
    fig.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)
    
def subplot_pethalos(perc, bins, label, ax, color='black', ylabel=None, xlabel=None, title=None, fontsize=12):

    ax.plot(bins, perc[2], linewidth=2, color=color, label=label)
    ax.fill_between(bins, perc[1], perc[3], facecolor=color, alpha=0.2)
    ax.fill_between(bins, perc[0], perc[4], facecolor=color, alpha=0.2)
            
    ax.set_xlim(0.,2.)
    #ax.set_ylim(10.,15.)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    return None

def plot_pethalos1_rpofiles(dct_rp, ionname, ion):
    pets = np.load(pdir + '/pethalos1_L0100N1504_27.npz')
    masses = pets['M200c_Msun']
    bins = np.arange(0.05, 1.9, 0.1)
    fig, axes = plt.subplots(3, 2)
    axes = np.array(axes).flatten()
    lax = axes[5]
    titles = [r'$\log_{10}\, M\; [M_{\odot}]= %.1f$'%mass for mass in masses]
    xlabel = r'$R \, /\, R_{200c}$'
    ylabel = r'$\log_{10}\, N_{\mathrm{%s}} \; [\mathrm{cm}^{-2}]$'%ionname
    labels = pets.keys()
    labels.remove('M200c_Msun')
    fontsize = 12
    fig.suptitle(r'2d radial profiles for $\mathrm{%s}$'%ionname)
    for label in labels:
        for mind in range(len(masses)):
    	    if pets[label][mind] is None:
    	        continue
    	    subplot_pethalos(dct_rp[str(pets[label][mind])],\
    	      bins, label, axes[mind], color=petcolors[label],\
    	      ylabel=ylabel, xlabel=xlabel, title=titles[mind],\
    	      fontsize=fontsize)
    handles, labels = axes[0].get_legend_handles_labels()
    lax.legend(handles, labels, fontsize=fontsize, ncol=2)
    lax.axis('off')
    plt.savefig('/net/luttero/data2/imgs/CGM/misc_start/profiles_%s_pets1_1slice_L0100N1504_27.pdf'%ion, format='pdf', bbox_inches='tight')
     

def plot_halo_environments(haloid, halocat, halocolor='black', boundaryslices=16,\
                           projaxis=2, ax=None, mapaxis=0, logMmin=-100.,\
                           vcen=False, labelprojaxis=True, addtitle=True):
    '''
    indicates halo positions in position and velocity space, along with
    surrounding halos
    '''
    if '/' not in halocat:
        halocat = pdir + halocat
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
    with h5py.File(halocat, 'r') as fi:
        ids = np.array(fi['groupid'])
        cosmopars = {key: fi['Header/cosmopars'].attrs[key] for key in fi['Header/cosmopars'].attrs.keys()}
        M200s = R200s = np.array(fi['M200c_Msun'])
        R200s = np.array(fi['R200c_pkpc']) / 1.e3 / (1. + cosmopars['z'])  
        centres = np.array([np.array(fi['Xcop_cMpc']), np.array(fi['Ycop_cMpc']), np.array(fi['Zcop_cMpc'])]).T
        vproj = np.array(fi['V%spec_kmps'%(['X', 'Y', 'Z'][projaxis])])
    boxsize_cMpc = cosmopars['boxsize'] / cosmopars['h']
    ind = np.where(ids == haloid)[0][0]
    R200 = R200s[ind]
    centre = centres[ind]
    
    
    mapaxes = [0, 1, 2]
    mapaxes.remove(projaxis)
    if mapaxis not in mapaxes:
        mapaxis += 1
        mapaxis %= 3
    # select galaxies in a cilinder around the main halo,
    maxnormdist_proj = 2. * boxsize_cMpc / float(boundaryslices) / R200
    maxnormdist_map = 4. 
    
    Hz =  mc.csu.Hubble(cosmopars['z'],cosmopars=cosmopars)
    losposdiffs = vproj / Hz * cosmopars['a'] # peculiar velocity -> los comoving position offset. Max halo vpec are ~1500 km/s
    centresv = np.copy(centres)
    centresv[:, projaxis] += losposdiffs
    centresv[:, projaxis] %= boxsize_cMpc
    centrev = centresv[ind]    
    
     # centre the coordinates on the halo of choice -> avoid issues with edge overlaps
    if vcen:
        cencoord = centrev[projaxis]
        centres[:, projaxis] -= (cencoord - boxsize_cMpc/2.)
        centres[:, projaxis] %= boxsize_cMpc
        centres[:, projaxis] += (cencoord- boxsize_cMpc/2.)
        centresv[:, projaxis] -= (cencoord - boxsize_cMpc/2.)
        centresv[:, projaxis] %= boxsize_cMpc
        centresv[:, projaxis] += (cencoord - boxsize_cMpc/2.)
    else:
        cencoord = centre[projaxis]
        centres[:, projaxis] -= (cencoord - boxsize_cMpc/2.)
        centres[:, projaxis] %= boxsize_cMpc
        centres[:, projaxis] += (cencoord- boxsize_cMpc/2.)
        centresv[:, projaxis] -= (cencoord - boxsize_cMpc/2.)
        centresv[:, projaxis] %= boxsize_cMpc
        centresv[:, projaxis] += (cencoord - boxsize_cMpc/2.)
    
    selection1 = np.where(np.all(np.array([ (centres[:, projaxis] - centre[projaxis]) / (R200s + R200) <= maxnormdist_proj,\
                                           ((centres[:, mapaxes[0]] - centre[mapaxes[0]])**2 +\
                                            (centres[:, mapaxes[1]] - centre[mapaxes[1]])**2) / (R200s + R200)**2 <= (2 * maxnormdist_map)**2\
                                         ]), axis=0))[0]
    selection1 = np.array(list(set(selection1) & set(np.where(M200s >= 10**logMmin)[0])))
    R200s1 = R200s[selection1]
    centres1 = centres[selection1]
    normdist1 = np.sum((centres[:, :] - centre[np.newaxis, :])**2, axis=1) / (2 * maxnormdist_map * (R200s + R200))**2 
    
    # velocity space (position coordinates); still uses position space radius, not e.g. virial velocty along V axis
    selection2 = np.where(np.all(np.array([ (centresv[:, projaxis] - centrev[projaxis]) / (R200s + R200) <= maxnormdist_proj,\
                                           ((centresv[:, mapaxes[0]] - centrev[mapaxes[0]])**2 +\
                                            (centresv[:, mapaxes[1]] - centrev[mapaxes[1]])**2) / (R200s + R200)**2 <= (2 * maxnormdist_map)**2\
                                         ]), axis=0))[0]
    selection2 = np.array(list(set(selection2) & set(np.where(M200s >= 10**logMmin)[0])))
    R200s2 = R200s[selection2]
    centres2 = centresv[selection2]
    normdist2 = np.sum((centres2[:, :] - centrev[np.newaxis, :])**2, axis=1) / (2 * maxnormdist_map * (R200s2 + R200))**2 
    
    fontsize = 12
    cmap1 = gray_dark
    cmap2 = pink_dark
    #cmap = mpl.cm.get_cmap(scmap)
    vmin = 0.
    vmax = 1.
    if isinstance(halocolor, str):
        halocolor = mcolors.to_rgba(halocolor)
    
    selind1 = np.where(selection1 == ind)[0][0]
    selind2 = np.where(selection2 == ind)[0][0]
    #alphamin = 0.3
    #alphamax = 0.8
    colors1 = cmap1((normdist1 - vmin)/(vmax-vmin)) # np.array((halocolor,)* len(normdist))
    colors1[selind1] = halocolor
    colors2 = cmap2((normdist2 - vmin)/(vmax-vmin)) # np.array((halocolor,)* len(normdist))
    colors2[selind2] = halocolor
    #colors[:,3] = (normdist - vmin)/(vmax-vmin) * (alphamax-alphamin) + alphamin #set alpha
    #colors[:,3][np.logical_not(np.isfinite(colorqts))] = alphamin # set alpha for zero-mass halos
    
    xs = centres1[:, mapaxis]
    ys = centres1[:, projaxis]
    sz = R200s1
    xs2 = centres2[:, mapaxis]
    ys2 = centres2[:, projaxis]
    sz2 = R200s2
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    # handle ranges 
    if vcen:
        ucentre = centrev
    else:
        ucentre = centre
    if boundaryslices is not None:
        slicepos =  (np.arange(boundaryslices)/float(boundaryslices) +  0.5/float(boundaryslices)) * boxsize_cMpc
        slicepos -= (ucentre[projaxis] - boxsize_cMpc/2.)
        slicepos %= boxsize_cMpc
        slicepos += (ucentre[projaxis] - boxsize_cMpc/2.)
        for sl in slicepos:
            ax.axhline(sl, color='gray', alpha=0.3)
        ymax = np.min(slicepos[slicepos > ucentre[projaxis] + 2 * R200])
        ymin = np.max(slicepos[slicepos < ucentre[projaxis] - 2 * R200])
        ymax = max(ymax, ucentre[projaxis] + maxnormdist_map * R200)
        ymin = min(ymin, ucentre[projaxis] - maxnormdist_map * R200)
        ylim = (ymin, ymax)
        labelbottom = False
    else:
        ylim = (centre[projaxis] - maxnormdist_map * R200, centre[projaxis] + maxnormdist_map * R200)
        labelbottom = True


    connect = np.array(list(set(selection1) & set(selection2)))
    matchinds = [[np.where(selection1 == connect[i])[0][0], np.where(selection2 == connect[i])[0][0]] for i in range(len(connect))]
    for pair in matchinds:
        if (ys[pair[0]] >= ylim[0] - 2 * R200s[pair[0]] and ys[pair[0]] <= ylim[0] + 2 * R200s[pair[0]]) or\
           (ys2[pair[1]] >= ylim[0] - 2 * R200s2[pair[1]] and ys2[pair[1]] <= ylim[0] + 2 * R200s2[pair[1]]):
            ax.plot([xs[pair[0]], xs2[pair[1]]], [ys[pair[0]], ys2[pair[1]]], color='gray', linewidth=1)
        
    patches2 = [Circle((xs[i], ys[i]), 2. * sz[i]) \
               for i in range(len(xs))] # x, y axes only
    collection2 = PatchCollection(patches2)
    collection2.set(edgecolor=colors1, facecolor='none', linewidth=1)
    ax.add_collection(collection2)
    ax.scatter([xs[selind1]], [ys[selind1]], color=halocolor, s=30, marker='*')
    
    patches22 = [Circle((xs2[i], ys2[i]), 2. * sz2[i]) \
               for i in range(len(xs2))] # x, y axes only
    collection22 = PatchCollection(patches22)
    collection22.set(edgecolor=colors2, facecolor='none', linewidth=1)
    ax.add_collection(collection22)
    ax.scatter([xs2[selind2]], [ys2[selind2]], color=halocolor, s=30, marker='x')
    
   
    
    ax.set_xlim(centre[mapaxis] - maxnormdist_map * R200, centre[mapaxis] + maxnormdist_map * R200)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=fontsize - 1, direction='out', top=False, right=False, bottom=labelbottom, labelleft=labelprojaxis, labelbottom=labelbottom, which='both')
    #xlim = ax.get_xlim()
    #ylim = ax.get_ylim()
    ax.set_aspect('equal')
    xlabel = '%s [cMpc]'%(['X', 'Y', 'Z'][mapaxis])
    ylabel = '%s [cMpc]'%(['X', 'Y', 'Z'][projaxis])
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if labelprojaxis:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if addtitle:
        ax.set_title('FOF halo %s'%haloid, fontsize=fontsize)

def subplot_profile(ax, x, y, kind='percentile', linewidth=1., **kwargs):
    '''
    '''
    if kind == 'percentile':
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
            del kwargs['alpha']
        elif len(y) in [2, 3]:
            alpha = 0.3
        elif len(y) in [4, 5]:
            alpha = 0.5
        
        if 'linestyles' in kwargs:
            linestyle = kwargs['linestyles'][0]
            del kwargs['linestyles']
        else:
            linestyle = 'solid'
            
        if len(y) == 1 or len(y.shape) == 1:
            ax.plot(x, y[0], linestyle=linestyle, linewidth=linewidth, **kwargs)
        elif len(y) == 2:
            ax.fill_between(x, y[0], y[1], alpha=alpha, **kwargs)
        elif len(y) == 3:
            ax.fill_between(x, y[0], y[2], alpha=alpha, **kwargs)
            ax.plot(x, y[1], linestyle=linestyle, linewidth=linewidth, **kwargs)
        elif len(y) == 4:
            ax.fill_between(x, y[0], y[3], alpha=alpha, **kwargs)   
            ax.fill_between(x, y[1], y[2], alpha=alpha, **kwargs)  
        elif len(y) == 5:
            ax.fill_between(x, y[0], y[4], alpha=alpha, **kwargs)   
            ax.fill_between(x, y[1], y[3], alpha=alpha, **kwargs)  
            ax.plot(x, y[2], linestyle=linestyle, linewidth=linewidth, **kwargs)

    elif kind == 'covfrac':
        if 'linestyles' in kwargs.keys():
            linestyles = kwargs['linestyles']
        elif 'linestyle' in kwargs.keys():
            linestyles = [kwargs['linestyle']] *len(y)
        else:
            linestyles = ['solid'] *len(y)   
        
        if 'colors' in kwargs.keys():
            colors = kwargs['colors']
        elif 'color' in kwargs.keys():
            colors = [kwargs['color']] *len(y)
        else:
            colors = ['c%i'%(i%10) for i in range(len(y))]
        
        if 'labels' in kwargs.keys():
            labels = kwargs['labels']
        elif 'label' in kwargs.keys():
            labels = [kwargs['label']] + [None] * (len(y) - 1)
        else:
            labels = [None] * len(y)
            
        for yi in range(len(y)):
            ax.plot(x, y[yi], linewidth=linewidth, color=colors[yi], label=labels[yi], linestyle=linestyles[yi])
        
def plot_rdistsubset_from_hdf5(crcat, halocat, xunit='R200c', yaxis='perc',\
                               subsets=None, galids=None,\
                               xlim=(0., 2.), yvals=None,\
                               yqty=r'$\log_{10}\, N_{\mathrm{O\, VII}} \; [\mathrm{cm}^{-2}]$',\
                               ylim=None, colorby=None, plotsinglehalos='full',\
                               savename=None):
    '''
    xunit: R200c or pkpc
    yaxis: perc (percentiles) or fcov (covering fractions)
    yvals: pecentile values if Nperc, 
           threshold log column densities (log cm^-2) if fcov
    subsets: list of (halocat key, min/None, max/None values) 
    galids: list of galaxy ids to use (any subsets are applied to these)
    plotsinglehalos: 'full' - plot all the same things as for the ensemble
                     'min'  - plot only a thin line for each halo
                     False  - don't plot the single halos at all
    '''
    fontsize = 12
    
    crdcat = h5py.File(crcat, 'r')
    if galids is None:
        galids = crdcat.keys()
        galids.remove('Header')
        galids = [int(galid) for galid in galids]
        galids.sort() # to get consistency between plots
        galids = np.array(galids)

    with h5py.File(halocat, 'r') as halos:
        cosmopars = {key: halos['Header/cosmopars'].attrs[key] for key in halos['Header/cosmopars'].attrs.keys()}
        allids = np.array(halos['galaxyid'])
        sel = np.array([np.where(galid == allids)[0][0] for galid in galids])
        if subsets is not None:
            subsel = np.ones(len(galids)).astype(bool)
            for subset in subsets:
                prop = np.array(halos[subset[0]])[sel]
                if subset[1] is not None:
                    subsel &= (prop >= subset[1])
                if subset[2] is not None:
                    subsel &= (prop <  subset[2])
            sel = sel[subsel]
            galids = galids[subsel]
                        
        if xunit == 'pkpc':
            rscales = np.array(halos['R200c_pkpc'])[sel]
        else:
            rscales = 1.
        if colorby == 'Mass':
            cvals = np.log10(np.array(halos['M200c_Msun'])[sel])
            clabel = r'$\log_{10}\, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}]$'
        elif colorby == 'Tvir':
            cvals = np.array(halos['M200c_Msun'])[sel]
            cvals = np.log10(cu.Tvir(cvals, cosmopars=cosmopars, mu=0.59))
            clabel = r'$\log_{10}\, \mathrm{T}_{\mathrm{vir}} \; [\mathrm{K}]$'
        
    rqarrs = [np.array(crdcat[str(galid)]) for galid in galids]
    if np.any(rscales != 1.):
        rqarrs = [np.array([rqarrs[i][0] * rscales[i], rqarrs[i][1]]) for i in range(len(rqarrs))]
    
    if plotsinglehalos != False:
        rqdct = {galids[i]: rqarrs[i] for i in range(len(galids))}
    else:
        rqdct = {}
    rqdct.update({'all': np.concatenate(rqarrs, axis=1)})

    
    binsize = (xlim[1] - xlim[0]) / 20.
    bins = np.arange(xlim[0], xlim[1] + 0.5 * binsize, binsize)    
    binsc = bins[:-1] + 0.5 * np.diff(bins)
    
    if yaxis == 'perc':    
        ydct = rqhists(rqdct, bins, percentiles=yvals)
    else:
        ydct = rqgeqs(rqdct, bins, values=np.array(yvals))
    
    if colorby is None:    
        fig, ax = plt.subplots(1, 1)
    else:
        plt.figure()
        grid = gsp.GridSpec(1, 2, width_ratios=[6., 1.])
        (ax, cax) = (plt.subplot(grid[i]) for i in range(2))
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', labelsize=fontsize - 1, top=True, left=True, right=True)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    if xunit == 'R200c':
        ax.set_xlabel(r'$\mathrm{R}_{\perp} \; [\mathrm{R}_{200c}]$', fontsize=fontsize)
    else:
        ax.set_xlabel(r'$\mathrm{R}_{\perp} \; [\mathrm{pkpc}]$', fontsize=fontsize)
    if yaxis == 'perc':
        ax.set_ylabel(yqty, fontsize=fontsize)
    else:
        if yqty[0] == r'$' and yqty[-1] == r'$':
            syqty = yqty[1:-1]
        else:
            syqty = yqty
        ax.set_ylabel('$\\mathrm{f}(%s > \mathrm{min.\, val})$'%(syqty), fontsize=fontsize)
        
    if colorby is None:
        colors =  ['C%i'%(gali % 10) for gali in range(len(galids))]
    else:
        colormap = cm.get_cmap('jet')
        colors = colormap((cvals - np.min(cvals)) / (np.max(cvals) - np.min(cvals)))
        
    if yaxis == 'perc':
        if plotsinglehalos == 'full':
            for gali in range(len(galids)):
                galid = galids[gali]
                color = colors[gali]
                
                subplot_profile(ax, binsc, ydct[galid], kind='percentile', linewidth=1., color=color)    
                if len(galids) <= 10 and colorby is None:
                    ax.text(0.01 + 0.2 * (gali % 5), 0.05 + 0.1 * (gali // 5) , str(galid), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
            if len(galids) > 10:
                ax.text(0.01, 0.05, '%i galaxies'%len(galids), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        elif plotsinglehalos == 'min':
            for gali in range(len(galids)):
                galid = galids[gali]
                color = colors[gali]
                
                num = len(ydct[galid])
                if num % 2: # odd
                    mini = num // 2 
                    maxi = mini + 1
                else:
                    maxi = num // 2 + 1
                    mini = num // 2 - 1
                    
                subplot_profile(ax, binsc, ydct[galid][mini:maxi], kind='percentile', linewidth=0.5, color=color)   
                
                if len(galids) <= 10 and colorby is None:
                    ax.text(0.01 + 0.2 * (gali % 5), 0.05 + 0.1 * (gali // 5) , str(galid), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
            if len(galids) > 10:
                ax.text(0.01, 0.05, '%i galaxies'%len(galids), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        else:
            ax.text(0.01, 0.05, '%i galaxies'%len(galids), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        subplot_profile(ax, binsc, ydct['all'], kind='percentile', linewidth=2., color='black')   
        
        legendtext = ['percentiles:']
        legendtext = legendtext + ['%.0f%%'%val for val in yvals]
        legendtext = '\n'.join(legendtext)
        ax.text(0.99, 0.99, legendtext, fontsize=fontsize-1, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color='black') #multialigmnent='right',
    
    else:
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        if plotsinglehalos == 'full':
            for gali in range(len(galids)):
                galid = galids[gali]
                color = colors[gali]
                
                subplot_profile(ax, binsc, ydct[galid], kind='covfrac', linewidth=1., color=color, linestyles=linestyles)    
                if len(galids) <= 10 and colorby is None:
                    ax.text(0.01 + 0.2 * (gali % 5), 0.05 + 0.1 * (gali // 5) , str(galid), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
            if len(galids) > 10:
                ax.text(0.01, 0.05, '%i galaxies'%len(galids), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        elif plotsinglehalos == 'min':
            for gali in range(len(galids)):
                galid = galids[gali]
                color = colors[gali]
                
                subplot_profile(ax, binsc, ydct[galid], kind='covfrac', linewidth=0.5, color=color, linestyles=linestyles)   
                
                if len(galids) <= 10 and colorby is None:
                    ax.text(0.01 + 0.2 * (gali % 5), 0.05 + 0.1 * (gali // 5) , str(galid), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
            if len(galids) > 10:
                ax.text(0.01, 0.05, '%i galaxies'%len(galids), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        else:
            ax.text(0.01, 0.05, '%i galaxies'%len(galids), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        subplot_profile(ax, binsc, ydct['all'], kind='covfrac', linewidth=2., color='black', linestyles=linestyles)   
        
        th_legend_handles = [mlines.Line2D([], [], color='black', linestyle=linestyles[i], label='%.1f'%(yvals[i])) for i in range(len(yvals))]
        legend = ax.legend(handles=th_legend_handles, fontsize=fontsize - 1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
        legend.set_title('min. val',  prop = {'size': fontsize - 1})
        
    if colorby is not None:
        add_colorbar(cax, img=None, vmin=np.min(cvals), vmax=np.max(cvals), cmap=colormap, clabel=clabel,\
                 newax=False, extend='neither', fontsize=fontsize, orientation='vertical')
    
    if savename is not None:
        if '/' not in savename:
            savename = mdir + savename
        if savename[-4:] != '.pdf':
            savename = savename + '.pdf'
        plt.savefig(savename, format='pdf', bbox_inches='tight')
        

def plot_rdistsubsets_from_hdf5_comparecrfiles\
    (crcats, halocat, xunit='R200c', yaxis='perc',\
     subsets=None, galids=None,\
     xlim=(0., 2.), yvals=None,\
     yqty=r'$\log_{10}\, N_{\mathrm{O\, VII}} \; [\mathrm{cm}^{-2}]$',\
     ylim=None, colorby=None, colors=None, plotsinglehalos=False,\
     savename=None, crcatnames=None, crcatlabels=None,\
     logx=False, logy=False):
    '''
    xunit: R200c or pkpc
    yaxis: perc (percentiles) or fcov (covering fractions)
    yvals: pecentile values if Nperc, 
           threshold log column densities (log cm^-2) if fcov
    subsets: list of (halocat key, min/None, max/None values) 
    galids: list of galaxy ids to use (any subsets are applied to these)
    plotsinglehalos: 'full' - plot all the same things as for the ensemble
                     'min'  - plot only a thin line for each halo
                     False  - don't plot the single halos at all
    '''
    fontsize = 12
    if crcatnames is None:
        crcatnames = range(len(crcats))
    if crcatlabels is None:
        crcatlabels = [None] * range(len(crcats))
    
    crdcats = {crcatnames[i]: h5py.File(crcats[i], 'r') for i in range(len(crcats))}
    crdlabels = {crcatnames[i]: crcatlabels[i] for i in range(len(crcats))}
    
    catnames = crcatnames
    catnames.sort()
    
    if galids is None:
        galids = {}
        for catname in crdcats.keys():
            galids_temp = crdcats[catname].keys()
            galids_temp.remove('Header')
            if 'selection' in galids_temp:
                galids_temp.remove('selection')
            galids_temp = [int(galid) for galid in galids_temp]
            galids_temp.sort() # to get consistency between plots
            galids_temp = np.array(galids_temp)
            galids.update({catname: galids_temp})

    with h5py.File(halocat, 'r') as halos:
        cosmopars = {key: halos['Header/cosmopars'].attrs[key] for key in halos['Header/cosmopars'].attrs.keys()}
        allids = np.array(halos['galaxyid'])
        sels = {catname: np.array([np.where(galid == allids)[0][0] for galid in galids[catname]]) for catname in galids.keys()}
        if subsets is not None:
            subsels = {catname: np.ones(len(galids[catname])).astype(bool) for catname in galids.keys()}
            for subset in subsets:
                props = {catname: np.array(halos[subset[0]])[sels[catname]] for catname in galids.keys()}
                if subset[1] is not None:
                    subsels  = {catname: np.logical_and(subsels[catname], props[catname] >= subset[1]) for catname in galids.keys()} 
                if subset[2] is not None:
                    subsels  = {catname: np.logical_and(subsels[catname], props[catname] < subset[2]) for catname in galids.keys()} 
            sels = {catname: sels[catname][subsels[catname]] for catname in galids.keys()}
            galids = {catname: galids[catname][subsels[catname]] for catname in galids.keys()}
                        
        if xunit == 'pkpc':
            rscales = {catname: np.array(halos['R200c_pkpc'])[sels[catname]] for catname in galids.keys()}
        else:
            rscales = {catname: 1. for catname in galids.keys()}
        if colorby == 'Mass':
            cvals = {catname: np.log10(np.array(halos['M200c_Msun'])[sels[catname]]) for catname in galids.keys()} 
            clabel = r'$\log_{10}\, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}]$'
        elif colorby == 'Tvir':
            cvals = {catname: np.array(halos['M200c_Msun'])[sels[catname]] for catname in galids.keys()}
            cvals = {catname: np.log10(cu.Tvir(cvals[catname], cosmopars=cosmopars, mu=0.59)) for catname in galids.keys()}
            clabel = r'$\log_{10}\, \mathrm{T}_{\mathrm{vir}} \; [\mathrm{K}]$'
        
    rqarrs = {catname: [np.array(crdcats[catname][str(galid)]) for galid in galids[catname]] for catname in galids.keys()}
    rqarrs = {catname: [np.array([rqarrs[catname][i][0] * rscales[catname][i], rqarrs[catname][i][1]])\
                        for i in range(len(rqarrs[catname]))] \
                       if np.any(rscales[catname] != 1.) else \
                       rqarrs[catname]\
                       for catname in catnames}
   
    if plotsinglehalos != False:
        rqdcts = {catname: {galids[catname][i]: rqarrs[catname][i] for i in range(len(galids[catname]))} for catname in galids.keys()}
    else:
        rqdcts = {}
    rqdcts.update({catname: {'all': np.concatenate(rqarrs[catname], axis=1)} for catname in galids.keys()})

    if logx:
        bins = np.array([0.] + list(xlim[0] * 10**((np.arange(20)+ 1.)/20. * np.log10(xlim[1]/xlim[0]))))
        binsc = 10**(np.log10(bins)[:-1] + 0.5*np.diff(np.log10(bins)))
        binsc[0] = xlim[0]
    else:
        binsize = (xlim[1] - xlim[0]) / 20.
        bins = np.arange(xlim[0], xlim[1] + 0.5 * binsize, binsize)    
        binsc = bins[:-1] + 0.5 * np.diff(bins)
    
    if yaxis == 'perc':    
        ydcts = {catname: rqhists(rqdcts[catname], bins, percentiles=yvals) for catname in galids.keys()}
    else:
        ydcts = {catname: rqgeqs(rqdcts[catname], bins, values=np.array(yvals)) for catname in galids.keys()}
    
    if colorby is None:    
        fig, ax = plt.subplots(1, 1)
    else:
        plt.figure()
        grid = gsp.GridSpec(1, 2, width_ratios=[6., 1.])
        (ax, cax) = (plt.subplot(grid[i]) for i in range(2))
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', labelsize=fontsize - 1, top=True, left=True, right=True)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    if xunit == 'R200c':
        ax.set_xlabel(r'$\mathrm{R}_{\perp} \; [\mathrm{R}_{200c}]$', fontsize=fontsize)
    else:
        ax.set_xlabel(r'$\mathrm{R}_{\perp} \; [\mathrm{pkpc}]$', fontsize=fontsize)
    if yaxis == 'perc':
        ax.set_ylabel(yqty, fontsize=fontsize)
    else:
        if yqty[0] == r'$' and yqty[-1] == r'$':
            syqty = yqty[1:-1]
        else:
            syqty = yqty
        ax.set_ylabel('$\\mathrm{f}(%s > \mathrm{min.\, val})$'%(syqty), fontsize=fontsize)
        
    if colorby is None:
        if colors is None:
            colors =  {catnames[i]: 'C%i'%(i % 10) for i in range(len(catnames))}
        else:
            colors =  {catnames[i]: colors[i] for i in range(len(catnames))}
    else:
        colormap = cm.get_cmap('jet')
        minval = min([np.min(cvals[catname]) for catname in galids.keys()])
        maxval = max([np.max(cvals[catname]) for catname in galids.keys()])
        colors = {catname: colormap((cvals[catname] - minval) / (maxval - minval)) for catname in galids.keys()}
    
    samegals = np.all([np.all(galids[catname] == galids[catnames[0]]) for catname in catnames])
    if yaxis == 'perc':
        for cati in range(len(catnames)):
            catname = catnames[cati]
            color = colors[catname]
            ydct = ydcts[catname]
            galids_sub = galids[catname]
            
            if plotsinglehalos == 'full':
                for gali in range(len(galids)):
                    galid = galids_sub[gali]    
                    if colorby:
                        subplot_profile(ax, binsc, ydct[galid], kind='percentile', linewidth=1., color=color[gali])  
                    else:
                        subplot_profile(ax, binsc, ydct[galid], kind='percentile', linewidth=1., color=color)  
                if not samegals:
                    ax.text(0.01 + 0.2 * (cati % 5), 0.05 + 0.1 * (cati // 5), '%i galaxies'%len(galids_sub), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
            elif plotsinglehalos == 'min':
                for gali in range(len(galids)):
                    galid = galids[gali]
                    
                    num = len(ydct[galid])
                    if num % 2: # odd
                        mini = num // 2 
                        maxi = mini + 1
                    else:
                        maxi = num // 2 + 1
                        mini = num // 2 - 1
                    if colorby:
                        subplot_profile(ax, binsc, ydct[galid][mini:maxi], kind='percentile', linewidth=0.5, color=color[gali])   
                    else:
                        subplot_profile(ax, binsc, ydct[galid][mini:maxi], kind='percentile', linewidth=0.5, color=color)   
                    
            
                if not samegals:
                    ax.text(0.01 + 0.2 * (cati % 5), 0.05 + 0.1 * (cati // 5), '%i galaxies'%len(galids_sub), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
            elif not samegals:
                ax.text(0.01 + 0.2 * (cati % 5), 0.05 + 0.1 * (cati // 5), '%i galaxies'%len(galids_sub), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
        
        for catname in catnames:
            subplot_profile(ax, binsc, ydcts[catname]['all'], kind='percentile', linewidth=2., color=colors[catname])   
        if samegals:
            ax.text(0.01, 0.05, '%i galaxies for all'%len(galids[catnames[0]]), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
        
        legendtext = ['percentiles:']
        legendtext = legendtext + ['%.0f%%'%val for val in yvals]
        legendtext = '\n'.join(legendtext)
        if logx: 
            rightv = 0.99
        else:
            rightv = 0.60
        ax.text(0.99, rightv, legendtext, fontsize=fontsize-1, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color='black') #, multialigmnent='right'
        
        if colorby is None:
            if not logx:
                loc = 'upper right'
                anchor = (0.99, 0.99)
            else:
                loc = 'lower left'
                anchor = (0.01, 0.09)
            legend_handles = [mlines.Line2D([], [], color=colors[catnames[i]], linestyle='solid', label=crdlabels[catnames[i]]) for i in range(len(catnames))]
            legend = ax.legend(handles=legend_handles, fontsize=fontsize - 1, loc=loc, bbox_to_anchor=anchor, frameon=False, ncol=2)
        
    
    else:
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        numgals = []
        for cati in range(len(catnames)):
            catname = catnames[cati]
            color = colors[catname]
            ydct = ydcts[catname]
            galids_sub = galids[catname]
            if plotsinglehalos == 'full':
                for gali in range(len(galids)):
                    galid = galids[gali]
                    if colorby:                    
                        subplot_profile(ax, binsc, ydct[galid], kind='covfrac', linewidth=1., color=color[gali], linestyles=linestyles)    
                    else:
                        subplot_profile(ax, binsc, ydct[galid], kind='covfrac', linewidth=1., color=color, linestyles=linestyles)    
                    
                #if not samegals:
                #    ax.text(0.01 + 0.2 * (cati % 5), 0.05 + 0.1 * (cati // 5), '%i galaxies'%len(galids_sub), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
                numgals = numgals + [len(galids_sub)]
            elif plotsinglehalos == 'min':
                for gali in range(len(galids)):
                    galid = galids[gali]
                    
                    if colorby:                    
                        subplot_profile(ax, binsc, ydct[galid], kind='covfrac', linewidth=0.5, color=color[gali], linestyles=linestyles)  
                    else:
                        subplot_profile(ax, binsc, ydct[galid], kind='covfrac', linewidth=0.5, color=color, linestyles=linestyles)  
                    
                #if not samegals:
                #    ax.text(0.01 + 0.2 * (cati % 5), 0.05 + 0.1 * (cati // 5), '%i galaxies'%len(galids_sub), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
                numgals = numgals + [len(galids_sub)]
            elif not samegals:
                numgals = numgals + [len(galids_sub)]
                #ax.text(0.01 + 0.2 * (cati % 5), 0.05 + 0.1 * (cati // 5), '%i galaxies'%len(galids_sub), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color=color)
        for catname in catnames:
            subplot_profile(ax, binsc, ydcts[catname]['all'], kind='covfrac', linewidth=2., color=colors[catname], linestyles=linestyles)   

        if samegals:
            ax.text(0.01, 0.05, '%i galaxies for all'%len(galids[catnames[0]]), fontsize=fontsize-1, horizontalalignment='left', verticalalignment = 'top', transform=ax.transAxes, color='black')
            
        th_legend_handles = [mlines.Line2D([], [], color='black', linestyle=linestyles[i], label='%.1f'%(yvals[i])) for i in range(len(yvals))]
        if len(numgals) > 0:
            #print(crdlabels)
            #print(numgals)
            legend_handles = [mlines.Line2D([], [], color=colors[catnames[i]], linestyle='solid', label=crdlabels[catnames[i]] + ': %i'%numgals[i]) for i in range(len(catnames))]
        else:
            legend_handles = [mlines.Line2D([], [], color=colors[catnames[i]], linestyle='solid', label=crdlabels[catnames[i]]) for i in range(len(catnames))]
        if not logy:
            loc1 = 'upper right'
            anchor1 = (0.99, 0.99)
            loc2 = 'upper right'
            anchor2 = (0.99, 0.87)
            legend1 = plt.legend(handles=th_legend_handles, fontsize=fontsize - 2, loc=loc1, bbox_to_anchor=anchor1, ncol=min(3, len(th_legend_handles)), frameon=False)
            legend1.set_title('min. val',  prop = {'size': fontsize - 1})
            legend2 = plt.legend(legend=legend_handles, fontsize=fontsize - 2, loc=loc2,  bbox_to_anchor=anchor2, ncol=2, frameon=False)
            ax.add_artist(legend1)
            ax.add_artist(legend2)
        else:
            loc1 = 'lower left'
            anchor1 = (0.01, 0.4)
            loc2 = 'lower left'
            anchor2 = (0.01, 0.01)
            legend1 = plt.legend(handles=th_legend_handles, fontsize=fontsize - 2, loc=loc1, bbox_to_anchor=anchor1, ncol=1, frameon=False)
            legend1.set_title('min. val',  prop = {'size': fontsize - 1})
            legend2 = plt.legend(handles=legend_handles, fontsize=fontsize - 2, loc=loc2,  bbox_to_anchor=anchor2, ncol=2, frameon=False)
            ax.add_artist(legend1)
            ax.add_artist(legend2)
            
    if colorby is not None:
        add_colorbar(cax, img=None, vmin=np.min(cvals), vmax=np.max(cvals), cmap=colormap, clabel=clabel,\
                 newax=False, extend='neither', fontsize=fontsize, orientation='vertical')
    
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
        
    if savename is not None:
        if '/' not in savename:
            savename = mdir + savename
        if savename[-4:] != '.pdf':
            savename = savename + '.pdf'
        plt.savefig(savename, format='pdf', bbox_inches='tight')


def ploth1profiles(yaxis='fcov', subsets=None, crcset='posspace_1slice'):
    halocat = ol.pdir + 'catalogue_RecalL0025N0752_snap11_aperture30_inclsatellites.hdf5'
    
    ylim = None
    colors = None
    if crcset == 'velspace_4-8slice_Mst-7.0-7.5':
        crcats = {\
                  'w168_o0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_4slice_offset-0.000000_velspace.hdf5',\
                  'w168_op100': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_4slice_offset-0.520833_velspace.hdf5',\
                  'w168_op200': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_4slice_offset-1.041667_velspace.hdf5',\
                  'w168_om100': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_4slice_offset--0.520833_velspace.hdf5',\
                  'w168_om200': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_4slice_offset--1.041667_velspace.hdf5',\
                  'w337_op100': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_8slice_offset-0.520833_velspace.hdf5',\
                  'w337_op200': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_8slice_offset-1.041667_velspace.hdf5',\
                  'w337_om100': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_8slice_offset--0.520833_velspace.hdf5',\
                  'w337_om200': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_8slice_offset--1.041667_velspace.hdf5',\
                  'w337_o0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_8slice_offset-0.000000_velspace.hdf5',\
                  }
        
        crcatlabels = {\
                      'w168_op100': 'v 168 +100',\
                      'w168_op200': 'v 168 +200',\
                      'w168_om100': 'v 168 -100',\
                      'w168_om200': 'v 168 -200',\
                      'w168_o0':    'v 168',\
                      'w337_op100': 'v 337 +100',\
                      'w337_op200': 'v 337 +200',\
                      'w337_om100': 'v 337 -100',\
                      'w337_om200': 'v 337 -200',\
                      'w337_o0':    'v 337',\
                      }
        savename = 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_comp-slices-4-8_compoffsets_velspace_inclsats'
    
    elif crcset == 'velspace_4-8slice_Mstvar':
        crcats = {\
                  'w168_7.0-7.5':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_4slice_offset-0.000000_velspace.hdf5',\
                  'w337_7.0-7.5':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_8slice_offset-0.000000_velspace.hdf5',\
                  'w168_7.5-8.0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.5-8.0_4slice_offset-0.000000_velspace.hdf5',\
                  'w337_7.5-8.0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.5-8.0_8slice_offset-0.000000_velspace.hdf5',\
                  'w168_8.0-8.5':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-8.0-8.5_4slice_offset-0.000000_velspace.hdf5',\
                  'w337_8.0-8.5':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-8.0-8.5_8slice_offset-0.000000_velspace.hdf5',\
                  'w168_8.5-9.0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-8.5-9.0_4slice_offset-0.000000_velspace.hdf5',\
                  'w337_8.5-9.0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-8.5-9.0_8slice_offset-0.000000_velspace.hdf5',\
                  'w168_9.0-9.5':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-9.0-9.5_4slice_offset-0.000000_velspace.hdf5',\
                  'w337_9.0-9.5':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-9.0-9.5_8slice_offset-0.000000_velspace.hdf5',\
                  'w168_9.5-10.0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-9.5-10.0_4slice_offset-0.000000_velspace.hdf5',\
                  'w337_9.5-10.0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-9.5-10.0_8slice_offset-0.000000_velspace.hdf5',\
                  #'w168_10.0-inf':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-10.0-inf_4slice_offset-0.000000_velspace.hdf5',\
                  #'w337_10.0-inf':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-10.0-inf_8slice_offset-0.000000_velspace.hdf5',\
                  }
        
        crcatlabels = {\
                  'w168_7.0-7.5':    'M* 7-7.5, w 168',\
                  'w337_7.0-7.5':    'M* 7-7.5, w 337',\
                  'w168_7.5-8.0':    'M* 7.5-8, w 168',\
                  'w337_7.5-8.0':    'M* 7.5-8, w 337',\
                  'w168_8.0-8.5':    'M* 8-8.5, w 168',\
                  'w337_8.0-8.5':    'M* 8-8.5, w 337',\
                  'w168_8.5-9.0':    'M* 8.5-9, w 168',\
                  'w337_8.5-9.0':    'M* 8.5-9, w 337',\
                  'w168_9.0-9.5':    'M* 9-9.5, w 168',\
                  'w337_9.0-9.5':    'M* 9-9.5, w 337',\
                  'w168_9.5-10.0':   'M* 9.5-10, w 168',\
                  'w337_9.5-10.0':   'M* 9.5-10, w 337',\
                  #'w168_10.0-inf':   'M* >10, w 168',\
                  #'w337_10.0-inf':   'M* >10, w 337',\
                      }
        savename = 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mstar0p5dex-200rnd-7.0-7.5_comp-slices-4-8_Mstarbins_velspace_inclsats'
    
    elif crcset == 'velspace_4-8slice_Mhalovar_nossh':
        crcats = {\
                  'w168_9.0-9.5':     ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-9.0-9.5_4slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w337_9.0-9.5':     ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-9.0-9.5_8slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w168_9.5-10.0':    ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-9.5-10.0_4slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w337_9.5-10.0':    ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-9.5-10.0_8slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w168_10.0-10.5':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-10.0-10.5_4slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w337_10.0-10.5':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-10.0-10.5_8slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w168_10.5-11.0':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-10.5-11.0_4slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w337_10.5-11.0':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-10.5-11.0_8slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w168_11.0-11.5':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-11.0-11.5_4slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w337_11.0-11.5':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-11.0-11.5_8slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w168_11.5-12.0':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-11.5-12.0_4slice_offset-0.000000_velspace_centrals.hdf5',\
                  'w337_11.5-12.0':   ol.pdir + 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-11.5-12.0_8slice_offset-0.000000_velspace_centrals.hdf5',\
                  }
        
        crcatlabels = {\
                  'w168_9.0-9.5':    'Mh 9.0-9.5, w 168',\
                  'w337_9.0-9.5':    'Mh 9.0-9.5, w 337',\
                  'w168_9.5-10.0':   'Mh 9.5-10.0, w 168',\
                  'w337_9.5-10.0':   'Mh 9.5-10.0, w 337',\
                  'w168_10.0-10.5':  'Mh 10.0-10.5, w 168',\
                  'w337_10.0-10.5':  'Mh 10.0-10.5, w 337',\
                  'w168_10.5-11.0':  'Mh 10.5-11.0, w 168',\
                  'w337_10.5-11.0':  'Mh 10.5-11.0, w 337',\
                  'w168_11.0-11.5':  'Mh 11.0-11.5, w 168',\
                  'w337_11.0-11.5':  'Mh 11.0-11.5, w 337',\
                  'w168_11.5-12.0':  'Mh 11.5-12.0, w 168',\
                  'w337_11.5-12.0':  'Mh 11.5-12.0, w 337',\
                  #'w168_10.0-inf':   'M* >10, w 168',\
                  #'w337_10.0-inf':   'M* >10, w 337',\
                      }
        savename = 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-9.0-12.0_4-8slice_offset-0.000000_velspace_centrals'
        ylim = (1e-3, 1.1)
        colors = ['C%i'%i for i in range(6)]
        colors = colors + [mcolors.colorConverter.to_rgb(col) + (0.3,) for col in colors]
        
    elif crcset == 'posspace_1slice':
        crcats = {\
                  'w0.5_op0.5': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset-0.520833_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_op1.0': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset-1.041667_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_op1.6': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset-1.562500_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_op2.1': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset-2.083333_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_om0.5': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset--0.520833_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_om1.0': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset--1.041667_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_om1.6': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset--1.562500_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_om2.1': ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset--2.083333_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  'w0.5_o0':    ol.pdir + 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset-0.000000_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  }
        
        crcatlabels = {\
                      'w0.5_op0.5': '+ 0.52 cMpc (1 sl)',\
                      'w0.5_op1.0': '+ 1.04 cMpc',\
                      'w0.5_op1.6': '+ 1.56 cMpc',\
                      'w0.5_op2.1': '+ 2.08 cMpc',\
                      'w0.5_om0.5': '- 0.52 cMpc',\
                      'w0.5_om1.0': '- 1.04 cMpc',\
                      'w0.5_om1.6': '- 1.56 cMpc',\
                      'w0.5_om2.1': '- 2.08 cMpc',\
                      'w0.5_o0':    'centered',\
                      }
        savename = 'coldens_rdist_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_compoffsets_inclsats_Mh-ge-9.5_random1000_first100'
    
    elif crcset == 'mocks':
        halocat = ol.pdir + 'catalogue_mockhalos.hdf5'
        crcats = {\
                  'w1_op1': ol.pdir + 'coldens_rdist_mockhalos_offset-1.250000.hdf5',\
                  'w1_op2': ol.pdir + 'coldens_rdist_mockhalos_offset-2.500000.hdf5',\
                  'w1_op3': ol.pdir + 'coldens_rdist_mockhalos_offset-3.750000.hdf5',\
                  'w1_op4': ol.pdir + 'coldens_rdist_mockhalos_offset-5.000000.hdf5',\
                  'w1_om1': ol.pdir + 'coldens_rdist_mockhalos_offset--1.250000.hdf5',\
                  'w1_om2': ol.pdir + 'coldens_rdist_mockhalos_offset--2.500000.hdf5',\
                  'w1_om3': ol.pdir + 'coldens_rdist_mockhalos_offset--3.750000.hdf5',\
                  'w1_om4': ol.pdir + 'coldens_rdist_mockhalos_offset--5.000000.hdf5',\
                  'w1_o0':  ol.pdir + 'coldens_rdist_mockhalos_offset-0.000000.hdf5',\
                  }
        crcatlabels = {\
                  'w1_op1': '+1',\
                  'w1_op2': '+2',\
                  'w1_op3': '+3',\
                  'w1_op4': '+4',\
                  'w1_om1': '-1',\
                  'w1_om2': '-2',\
                  'w1_om3': '-3',\
                  'w1_om4': '-4',\
                  'w1_o0':  'cen',\
                  }
        savename = 'stamps_mockhalos_1slices_compoffsets'
        
    names = crcats.keys()
    labels  = [crcatlabels[name] for name in names]
    cats    = [crcats[name] for name in names]
    
    savename = savename + '_%s'%yaxis 
    if subsets is not None:
        savename  = savename + '_' + '_'.join(['-'.join(str(val) for val in tup) for tup in subsets])
    
    if yaxis == 'fcov':
        yvals = [17.2, 17.5, 17.9]
        logy=True
    elif yaxis == 'perc':
        yvals = [25., 50., 75.]
        logy=False
    
    plot_rdistsubsets_from_hdf5_comparecrfiles\
    (cats, halocat, xunit='pkpc', yaxis=yaxis,\
     subsets=subsets, galids=None,\
     xlim=(1., 50.), yvals=yvals,\
     yqty=r'$\log_{10}\, N_{\mathrm{H\, I}} \; [\mathrm{cm}^{-2}]$',\
     ylim=ylim, colorby=None, colors=colors, plotsinglehalos=False,\
     savename=savename, crcatnames=names, crcatlabels=labels,
     logx=True, logy=logy)

def plotenv_slices(galid, catset='posspace'):
    halocat = ol.pdir + 'catalogue_RecalL0025N0752_snap11_aperture30_inclsatellites.hdf5'
    if catset == 'velspace':
        crcats = {\
                   0.       : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset-0.000000_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   0.520833 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset-0.520833_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   1.041667 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset-1.041667_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   1.562500 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset-1.562500_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   2.083333 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset-2.083333_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -0.520833 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset--0.520833_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -1.041667 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset--1.041667_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -1.562500 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset--1.562500_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -2.083333 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_1slice_offset--2.083333_velspace_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  }
    elif catset == 'posspace':
        crcats = {\
                   0.       : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset-0.000000_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   0.520833 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset-0.520833_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   1.041667 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset-1.041667_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   1.562500 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset-1.562500_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                   2.083333 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset-2.083333_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -0.520833 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset--0.520833_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -1.041667 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset--1.041667_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -1.562500 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_1slice_offset--1.562500_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  -2.083333 : ol.pdir + 'stamps_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EO_1slice_offset--2.083333_inclsats_Mh-ge-9.5_random1000_first100.hdf5',\
                  }
    elif catset == 'mocks':
        halocat = ol.pdir + 'catalogue_mockhalos.hdf5'
        crcats = {\
                  1.25: ol.pdir + 'stamps_mockhalos_offset-1.250000.hdf5',\
                  2.50: ol.pdir + 'stamps_mockhalos_offset-2.500000.hdf5',\
                  3.75: ol.pdir + 'stamps_mockhalos_offset-3.750000.hdf5',\
                  5.00: ol.pdir + 'stamps_mockhalos_offset-5.000000.hdf5',\
                 -1.25: ol.pdir + 'stamps_mockhalos_offset--1.250000.hdf5',\
                 -2.50: ol.pdir + 'stamps_mockhalos_offset--2.500000.hdf5',\
                 -3.75: ol.pdir + 'stamps_mockhalos_offset--3.750000.hdf5',\
                 -5.00: ol.pdir + 'stamps_mockhalos_offset--5.000000.hdf5',\
                  0.00:  ol.pdir + 'stamps_mockhalos_offset-0.000000.hdf5',\
                  }
    cmap = 'viridis'
    fontsize = 12
    
    with h5py.File(halocat, 'r') as hc:
        galids = np.array(hc['galaxyid'])
        ind = np.where(galids == galid)[0][0]
        M200c = hc['M200c_Msun'][ind]
        R200c = hc['R200c_pkpc'][ind]
        Xpos  = hc['Xcop_cMpc'][ind]
        Ypos  = hc['Ycop_cMpc'][ind]
        Mstar = hc['Mstar_Msun'][ind]
        Vcmax = hc['Vmax_circ_kmps'][ind]
        iscen = hc['SubGroupNumber'][ind] == 0
        SFR   = hc['SFR_MsunPerYr'][ind]
        cosmopars = {key: item for (key, item) in hc['Header/cosmopars'].attrs.items()}
        
        pos_cMpc_to_vrf_kmps = cu.Hubble(cosmopars['z'], cosmopars=cosmopars) * cu.c.cm_per_mpc * cosmopars['a'] * 1e-5
        print(cosmopars)
        cMpc_to_pkpc = 1e3 * cosmopars['a']
        
    nsub = len(crcats)
    keys = crcats.keys()
    keys.sort()
    ncol = nsub // 3
    nrow = (nsub - 1) // ncol +  1
    
    fig = plt.figure(figsize=(2.5*ncol + 1., 2.5*nrow))
    grid = gsp.GridSpec(nrow, ncol + 1, hspace=0.10, wspace=0.1, top=0.90, bottom=0.05, left=0.05, right=0.95, width_ratios=[2.5]*ncol + [1.])
    axes = np.array([[fig.add_subplot(grid[y, x]) for x in range(ncol)] for y in range(nrow)])
    cax = fig.add_subplot(grid[:, ncol])
    
    extents = {}
    imgs = {}
    titles = {}
    for i in range(len(keys)):
        key = keys[i]
        print(key)
        titles[key] = 'slice offset: %.0f km/s'%(key * pos_cMpc_to_vrf_kmps)
        with h5py.File(crcats[key], 'r') as fi:
            img = np.array(fi[str(galid)])
            imgs[key] = img 
            galids_fi = np.array(fi['Header/labels'])
            ind = np.where(galids_fi == galid)[0][0]
            llc = fi['Header/lower_left_corners_size_x_units'][ind]
            pixsize = fi['Header'].attrs['pixel_size_size_x_units']
            
            pkpc_xunits = cosmopars['boxsize'] / cosmopars['h'] * cosmopars['a'] * 1e3 / fi['Header'].attrs['size_along_x']
            extents[key] = (llc[0] * pkpc_xunits, (llc[0] + pixsize * img.shape[0]) * pkpc_xunits, llc[1] * pkpc_xunits, (llc[1] + pixsize * img.shape[1]) * pkpc_xunits)
            
    vmax = max([np.max(imgs[key_]) for key_ in keys])
    minval = min([np.min(imgs[key_][np.isfinite(imgs[key_])]) for key_ in keys])
    vmin = min(max(11., minval), vmax - 2.)
    
    for i in range(len(keys)):
        col = i %  ncol
        row = i // ncol
        key = keys[i]
        ax = axes[row, col]         
        im = ax.imshow(imgs[key].T, interpolation='nearest', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=extents[key])
        
        ax.tick_params(labelsize=fontsize-1, left=True, right=True, top=True, bottom=True, direction='in', labelbottom=(row==nrow-1), labelleft=(col==0))
        
        patches = [Circle((Xpos * cMpc_to_pkpc, Ypos * cMpc_to_pkpc), R200c)] # x, y axes only
        collection = PatchCollection(patches)
        collection.set(edgecolor='red', facecolor='none', linewidth=1)
        ax.add_collection(collection)
        
        ax.text(0.05, 0.95, titles[key], fontsize=fontsize-1, horizontalalignment='left', verticalalignment='top', color='black', transform=ax.transAxes, bbox=dict(facecolor='white',alpha=0.5))
        
    plt.colorbar(im, cax=cax, orientation='vertical', extend='min')
    cax.tick_params(labelsize=fontsize-1)
    cax.set_ylabel(r'$\log_{10}\, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$', fontsize=fontsize)
    cax.set_aspect(12.)
    
    suplabel(fig, 'x', 'X [pkpc]',label_prop={'fontsize': fontsize},
             labelpad=5,
             ha='center',va='center')
    suplabel(fig, 'y', 'Y [pkpc]',label_prop={'fontsize': fontsize},
             labelpad=5,
             ha='center',va='center')
    
    fig.suptitle('Image slices for galaxy %i:\n'%(galid) +\
                 r'$\log_{10} M_{200c}/M_{\odot} =$' + ' %.1f'%(np.log10(M200c))\
                 + r'$, \; \log_{10} M_{*} / M_{\odot} =$' + ' %.1f'%(np.log10(Mstar))\
                 +'$, \; \log_{10}\; \mathrm{SFR} / M_{\odot}\mathrm{yr}^{-1} =$' ' %.1f'%(np.log10(SFR))\
                 + r'$, v_{c, \max} / \mathrm{km}\,\mathrm{s}^{-1} =$' + ' %.0f'%Vcmax\
                 + (', central' if iscen else ', satellite'),
                 fontsize=fontsize)
    
    savename = mdir + crcats[0.].split('/')[-1]
    if savename[-4:] == '.npz':
        savename = savename[:-4]
    savename = savename + '_%islices_galaxy%i.pdf'%(len(crcats), galid)
    
    plt.savefig(savename, format='pdf', bbox_inches='tight')



def plot_rdistsbyprops_from_hdf5(crcat, halocat, xunit='R200c', yaxis='perc',\
                               subsets='Mh0p5dex', numexamples=10,\
                               xlim=(0., 2.), yvals=None,\
                               ylim=None,\
                               savename=None):
    '''
    xunit: R200c or pkpc
    yaxis: perc (percentiles) or fcov (covering fractions)
    yvals: pecentile values if Nperc, 
           threshold log column densities (log cm^-2) if fcov

    '''
    fontsize = 12
    
    ion = crcat.split('/')[-1]
    ion = ion.split('_')[2]
    yqty=r'$\log_{10}\, N_{\mathrm{' + ild.getnicename(ion, mathmode=True) + r'}} \; [\mathrm{cm}^{-2}]$'
    
    crdcat = h5py.File(crcat, 'r')
    #if galids is None:
    #    galids_crc = crdcat.keys()
    #    galids_crc.remove('Header')
    #    galids_crc = [int(galid) for galid in galids_crc]
    #    galids_crc.sort() # to get consistency between plots
    #    galids_crc = np.array(galids_crc)
        
    if subsets == 'Mh0p5dex':
        selector = sh.L0100N1504_27_Mh0p5dex
        galids = selector.galids()
        selections = {name: sorted(selection) for (name, selection) in zip(selector.names, selector.selections)}
        clabel = r'$\log_{10} \, \mathrm{M}_{200c} \; [\mathrm{M}_{\odot}]$'
        
    with h5py.File(halocat, 'r') as halos:
        cosmopars = {key: halos['Header/cosmopars'].attrs[key] for key in halos['Header/cosmopars'].attrs.keys()}
        allids = np.array(halos['galaxyid'])
        sels = {key: np.array([np.where(galid == allids)[0][0] for galid in galids[key]]) for key in galids.keys()}

        rscales = {}
        if xunit == 'pkpc':
            r200s = np.array(halos['R200c_pkpc'])
            rscales = {key: r200s[sels[key]] for key in galids.keys()}
        else:
            rscales = {key: 1. for key in galids.keys()}

        
    rqarrs = {key: [np.array(crdcat[str(galid)]) for galid in galids[key]] for key in galids.keys()}
    if np.any([np.any(rscales[key] != 1.) for key in galids.keys()]):
        rqarrs = {key: [np.array([rqarrs[key][i][0] * rscales[key][i], rqarrs[key][i][1]]) for i in range(len(rqarrs[key]))] for key in galids.keys()}
    
    
    rqdct_samples = {key: np.concatenate(rqarrs[key], axis=1) for key in galids.keys()}
    
    if numexamples > 0:
        rqdct_examples = {key: {galids[key][i]: rqarrs[key][i] for i in range(min(numexamples, len(galids[key])))}  for key in galids.keys()}
    else:
        rqdct_examples = {}
    
    binsize = (xlim[1] - xlim[0]) / 20.
    bins = np.arange(xlim[0], xlim[1] + 0.5 * binsize, binsize)    
    binsc = bins[:-1] + 0.5 * np.diff(bins)
    
    if yaxis == 'perc':    
        ydct = rqhists(rqdct_samples, bins, percentiles=yvals)
        if numexamples > 0:
            ydct_examples = {key: rqhists(rqdct_examples[key], bins, percentiles=yvals) for key in galids.keys()}
    else:
        ydct = rqgeqs(rqdct_samples, bins, values=np.array(yvals))
        if numexamples > 0:
            ydct_examples = {key: rqgeqs(rqdct_examples[key], bins, values=np.array(yvals)) for key in galids.keys()}
    
    samples = {key: (np.log10(selections[key][0][1]) if selections[key][0][1] is not None else None,\
                     np.log10(selections[key][0][2]) if selections[key][0][2] is not None else None ) for key in galids.keys()} # assuming only only selection criterion per bin, and log values
    samplebins = [val for key in galids.keys() for val in samples[key]]
    samplebins = list(set(samplebins))
    if None in samplebins:
        samplebins.remove(None)
    samplebins.sort()
    avdiff = np.average(np.diff(samplebins))
    # sample center coloring only makes sense if bins are non-overlapping
    samplecens = {key: 0.5 * (samples[key][0] + samples[key][1]) if None not in samples[key] else\
                       samples[key][1] - 0.5 * avdiff if samples[key][0] is None else \
                       samples[key][0] + 0.5 * avdiff if samples[key][1] is None else \
                       np.NaN for key in galids.keys()}
    extlower = np.any([samples[key][0] is None for key in galids.keys()])
    extupper = np.any([samples[key][1] is None for key in galids.keys()])
    
    cbar_extend = 'neither'
    if extlower:
        samplebins = [samplebins[0] - 0.5 * avdiff] + samplebins
        cbar_extend = 'min'
    if extupper:
        samplebins = samplebins + [samplebins[-1] + 0.5 * avdiff]
        cbar_extend = 'max'
    if extlower and extupper:
        cbar_extend = 'both'

    plt.figure()
    grid = gsp.GridSpec(1, 2, width_ratios=[6., 1.])
    (ax, cax) = (plt.subplot(grid[i]) for i in range(2))
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', labelsize=fontsize - 1, top=True, left=True, right=True)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    
    if xunit == 'R200c':
        ax.set_xlabel(r'$\mathrm{R}_{\perp} \; [\mathrm{R}_{200c}]$', fontsize=fontsize)
    else:
        ax.set_xlabel(r'$\mathrm{R}_{\perp} \; [\mathrm{pkpc}]$', fontsize=fontsize)
    if yaxis == 'perc':
        ax.set_ylabel(yqty, fontsize=fontsize)
    else:
        if yqty[0] == r'$' and yqty[-1] == r'$':
            syqty = yqty[1:-1]
        else:
            syqty = yqty
        ax.set_ylabel('$\\mathrm{f}(%s > \mathrm{min.\, val})$'%(syqty), fontsize=fontsize)
    
    
    
    cmap = cm.get_cmap('rainbow')
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'segmap', cmaplist, cmap.N)
    cmap.set_over(cmap(1.))
    cmap.set_under(cmap(0.))
    
    # define the bins and normalize
    bounds = samplebins
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # create a second axes for the colorbar
    
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
        spacing='proportional', ticks=bounds, boundaries=bounds, format='%.1f',\
        extend=cbar_extend, orientation='vertical')
    
    sample_cnormed = {key: (samplecens[key] - bounds[0]) / (bounds[-1] - bounds[0]) for key in samplecens.keys()}
    print(sample_cnormed)
    
    cax.tick_params(labelsize=fontsize-1)
    cax.set_ylabel(clabel, fontsize=fontsize)
    cax.set_aspect(12.)    
    
    if yaxis == 'perc':
        kind = 'percentile'
        
        num = len(yvals)
        if num % 2: # odd
            mini = num // 2 
            maxi = mini + 1
        else:
            maxi = num // 2 + 1
            mini = num // 2 - 1
        
        legendtext = ['percentiles:']
        legendtext = legendtext + ['%.0f%%'%val for val in yvals]
        legendtext = '\n'.join(legendtext)
        ax.text(0.99, 0.99, legendtext, fontsize=fontsize-1, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color='black') #multialigmnent='right',
        
        linestyles = ['solid'] * num
    else:
        kind = 'convfrac'

        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
        
        mini = 0
        maxi = len(yvals)
        
        th_legend_handles = [mlines.Line2D([], [], color='black', linestyle=linestyles[i], label='%.1f'%(yvals[i])) for i in range(len(yvals))]
        legend = ax.legend(handles=th_legend_handles, fontsize=fontsize - 1, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=False)
        legend.set_title('min. val',  prop = {'size': fontsize - 1})
    
    keys = ydct.keys()
    if numexamples > 0:
        for key in keys:
            color = cmap(sample_cnormed[key])
            ys    = ydct_examples[key]
    
            for yk in ys.keys():
                subplot_profile(ax, binsc, ys[yk][mini:maxi], kind=kind, linewidth=0.5, color=color, linestyles=linestyles, alpha=0.1)    
    
    for key in keys:
        if kind == 'percentile' and len(ydct[key]) % 2: 
            mid = len(ydct[key]) // 2
            ax.plot(binsc, ydct[key][mid], color='black', linewidth=3., linestyle=linestyles[0])
        elif kind == 'covfrac':
            for i in range(len(ydct[key])):
                ax.plot(binsc, ydct[key][i], color='black', linewidth=3., linestyle=linestyles[i])
        subplot_profile(ax, binsc, ydct[key], kind=kind, linewidth=2., color=cmap(sample_cnormed[key]), linestyles=linestyles)   
        
    if savename is not None:
        if '/' not in savename:
            savename = mdir + savename
        if savename[-4:] != '.pdf':
            savename = savename + '.pdf'
        plt.savefig(savename, format='pdf', bbox_inches='tight')
        
        
###############################
########### tests #############
###############################
        
def testhalomask_basic(argset=0):
    '''
    included halos: blue circles
    excluded halos: red cricles
    black: True
    white: False
    
    argset: choose a small test case for the C selection routine
    '''
    # default args (modify based on argset)
    xpix = 100
    ypix = None
    periodic = True
    pixsize = 2.
    size = 200.
    closest_normradius = False
    axis = 'z'

    inpos = np.array([[20., 20., 20], [0., 0., 0.], [190., 190., 190.], [100., 50., 90.]])
    inrad = np.array([40., 20., 15., 30.])
    indct = {'pos': inpos, 'rad': inrad}
    
    expos = np.array([[20., 20., 20.], [10., 10., 10.], [150., 150., 150.], [60., 45., 100.]])    
    exrad = np.array([25., 15., 10., 20.])
    exdct = {'pos': expos, 'rad': exrad}
    
    ### includes only
    if argset == 0: # includes only, periodic
        exdct = None
        indct = {'pos': np.array([[]]), 'rad': np.array([])}
    elif argset == 1: # includes only, periodic
        exdct = None
    elif argset == 2: # includes only, non-periodic
        exdct = None
        periodic = False
    elif argset == 3: # includes only, non-periodic sub-volume
        exdct = None
        periodic = False
        xpix = 50
        ypix = 75
        size = 200.
    elif argset == 4:
        exdct = None
        axis = 'x'
    elif argset == 5:
        exdct = None
        axis = 'y'
    elif argset == 6:
        exdct = None
        axis = 'x'
        periodic=False
    elif argset == 7:
        exdct = None
        axis = 'y'   
        periodic=False
        xpix = 90
        ypix = 95
    elif argset == 8:
        pass
    elif argset == 9:
        closest_normradius = True
    elif argset == 10:
        exdct = {'pos': np.array([[50., 100., 20.]]), 'rad': np.array([5.])}
    elif argset == 11:
        closest_normradius = True
        periodic = False
    elif argset == 12:
        periodic = False
        xpix = 50
        ypix = 75
    elif argset == 13:
        axis = 'x'
        closest_normradius = True
    elif argset == 14:
        axis = 'y'
        periodic = False
        xpix = 75
        ypix = 65
        
    parstring = ['Input parameters:',\
                 'xpix:       %i'%xpix,\
                 'ypix:       %s'%ypix,\
                 'pixel size: %.2f'%pixsize,\
                 'box size:   %.2f'%size,\
                 'periodic:   %s'%periodic,\
                 'projection axis:  %s'%(axis),\
                 'exclude anything: %s'%(exdct is not None),\
                 r'excl. on $r/r_{\mathrm{halo}}$: %s'%closest_normradius]
    parstring = '\n'.join(parstring)
    tfmap = gethalomask_basic(xpix, size, pixsize, indct,\
                      ypix=ypix, exdct=exdct, closest_normradius=closest_normradius, axis=axis,
                      periodic=periodic)
    # retrieve some parameters like gethalomask should 
    if ypix is None:
        ypix = xpix
    if axis == 'z':
        axis1 = 0
        axis2 = 1
    elif axis == 'x':
        axis1 = 1
        axis2 = 2
    elif axis == 'y':
        axis1 = 2
        axis2 = 0
    
    if not np.all(np.array(tfmap.shape) == np.array([xpix, ypix])):
        raise RuntimeError('Ouput shape %s does not match desired %i, %i'%(tfmap.shape, xpix, ypix))
    
    fig = plt.figure()
    grid = gsp.GridSpec(2, 2, height_ratios=[5., 1.], width_ratios=[3., 2.], hspace=0.35, wspace=0.1)
    ax = fig.add_subplot(grid[0, 0])
    lax = fig.add_subplot(grid[1, 0])
    pax = fig.add_subplot(grid[:, 1])

    cmap = 'gist_yarg'
    fcmap = cm.get_cmap(cmap)
    color_true = fcmap(1.)
    color_false = fcmap(0.)    
    fontsize = 12.
    
    ax.imshow((tfmap.T).astype(float), interpolation='nearest', origin='lower', extent=(0., xpix*pixsize, 0., ypix*pixsize), cmap=cmap)
    
    circles_in = [mpatch.Circle(centre[[axis1, axis2]], rscale) for (centre, rscale) in zip(indct['pos'], indct['rad'])]
    offsets = np.array([[1., 1.], [1., 0.], [1., -1.], [0., 1.], [0., -1.], [-1., 1.], [-1., 0.], [-1., -1.]]) * size
    if periodic:
        circles_in = circles_in + [mpatch.Circle(centre[[axis1, axis2]] + offset, rscale) for (centre, rscale) in zip(indct['pos'], indct['rad']) for offset in offsets]
    collection_in = PatchCollection(circles_in, edgecolors='blue', facecolors='none')
    ax.add_collection(collection_in)
    
    if exdct is not None:
        circles_ex = [mpatch.Circle(centre[[axis1, axis2]], rscale) for (centre, rscale) in zip(exdct['pos'], exdct['rad'])]
        offsets = np.array([[1., 1.], [1., 0.], [1., -1.], [0., 1.], [0., -1.], [-1., 1.], [-1., 0.], [-1., -1.]]) * size
        if periodic:
            circles_ex = circles_ex + [mpatch.Circle(centre[[axis1, axis2]] + offset, rscale) for (centre, rscale) in zip(exdct['pos'], exdct['rad']) for offset in offsets]
        collection_ex = PatchCollection(circles_ex, edgecolors='red', facecolor='none')
        ax.add_collection(collection_ex)
        
    ax.set_xlim(0., xpix*pixsize)
    ax.set_ylim(0., ypix*pixsize)
    ax.set_xlabel(['X', 'Y', 'Z'][axis1], fontsize=fontsize)
    ax.set_ylabel(['X', 'Y', 'Z'][axis2], fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-1, which='both', direction='in', right=True, top=True)
    
    legend_handles = [mlines.Line2D([], [], color='blue', label='incl. halos'),\
                      mlines.Line2D([], [], color='red',  label='excl. halos'),\
                      mpatch.Patch(facecolor=color_true, edgecolor='black',label='True'),\
                      mpatch.Patch(facecolor=color_false, edgecolor='black',label='False'),\
                      ]
    lax.legend(handles=legend_handles, ncol=2, fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    lax.axis('off')
    
    pax.text(0.05, 0.95, parstring, horizontalalignment='left', verticalalignment='top', transform=pax.transAxes, fontsize=fontsize)
    pax.axis('off')

    plt.savefig('/net/luttero/data2/imgs/CGM/halomask_test/test_%i.pdf'%argset, format='pdf')    


def testhalomask_selection(maskfile, halocat):
    '''
    included halos: blue circles
    excluded halos: red cricles
    black: True
    white: False
    
    warning: do not use on very large files: plots are mostly useful for 
    zooming (interactive mode), but will likely fail if using too many pixels
    (e.g. matplotlib does not like 32000^2 images)
    '''
    # default args (modify based on argset)
    if '/' not in maskfile:
        maskfile = ol.pdir + maskfile
    if maskfile[-5:] != '.hdf5':
        maskfile = maskfile + '.hdf5'
    
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
    
    with h5py.File(maskfile, 'r') as ms:
         
        tfmap = np.array(ms['mask'])
        xpix, ypix = tfmap.shape
        axis = ms['Header/parameters'].attrs['axis']
        closest_normradius = ms['Header/parameters'].attrs['closest_normradius']
        radius_r200c =   ms['Header/parameters'].attrs['radius_r200c']
        cosmopars = {key: ms['Header/cosmopars'].attrs[key] for key in ms['Header/cosmopars'].attrs.keys()}
        periodic = True
        size = cosmopars['boxsize'] / cosmopars['h']
        pixsize = size / float(xpix)
        anyexcl = 'excluded' in ms['selection'].keys()
        
        galid_incl = np.array(ms['selection/included/galaxyid'])
        if anyexcl:
            galid_excl = np.array(ms['selection/excluded/galaxyid'])
        
    parstring = ['Input parameters:',\
                 'xpix:       %i'%xpix,\
                 'ypix:       %s'%ypix,\
                 'pixel size: %.2f'%pixsize,\
                 'box size:   %.2f'%size,\
                 'periodic:   %s'%periodic,\
                 'projection axis:  %s'%(axis),\
                 'exclude anything: %s'%(anyexcl),\
                 r'excl. on $r/r_{\mathrm{halo}}$: %s'%closest_normradius]
    parstring = '\n'.join(parstring)


    # retrieve some parameters like gethalomask should 
    if ypix is None:
        ypix = xpix
    if axis == 'z':
        axis1 = 0
        axis2 = 1
    elif axis == 'x':
        axis1 = 1
        axis2 = 2
    elif axis == 'y':
        axis1 = 2
        axis2 = 0
    
    with h5py.File(halocat, 'r') as hc:
        galid_all = np.array(hc['galaxyid'])
        pos = np.array([np.array(hc['Xcop_cMpc']), np.array(hc['Ycop_cMpc']), np.array(hc['Zcop_cMpc'])]).T
        rad = np.array(hc['R200c_pkpc'])
        rad *= radius_r200c / cosmopars['a'] * 1.e-3
        
        inclsel = np.array([gali in galid_incl for gali in galid_all])
        indct = {'pos': pos[inclsel], 'rad': rad[inclsel]}
        if anyexcl:
            exclsel = np.array([gali in galid_excl for gali in galid_all])
            exdct = {'pos': pos[exclsel], 'rad': rad[exclsel]}
        else:
            exdct = None
             
    fig = plt.figure()
    grid = gsp.GridSpec(2, 2, height_ratios=[5., 1.], width_ratios=[3., 2.], hspace=0.35, wspace=0.1)
    ax = fig.add_subplot(grid[0, 0])
    lax = fig.add_subplot(grid[1, 0])
    pax = fig.add_subplot(grid[:, 1])

    cmap = 'gist_yarg'
    fcmap = cm.get_cmap(cmap)
    color_true = fcmap(1.)
    color_false = fcmap(0.)    
    fontsize = 12.
    
    ax.imshow((tfmap.T).astype(float), interpolation='nearest', origin='lower', extent=(0., xpix*pixsize, 0., ypix*pixsize), cmap=cmap)
    
    circles_in = [mpatch.Circle(centre[[axis1, axis2]], rscale) for (centre, rscale) in zip(indct['pos'], indct['rad'])]
    offsets = np.array([[1., 1.], [1., 0.], [1., -1.], [0., 1.], [0., -1.], [-1., 1.], [-1., 0.], [-1., -1.]]) * size
    if periodic:
        circles_in = circles_in + [mpatch.Circle(centre[[axis1, axis2]] + offset, rscale) for (centre, rscale) in zip(indct['pos'], indct['rad']) for offset in offsets]
    collection_in = PatchCollection(circles_in, edgecolors='blue', facecolors='none')
    ax.add_collection(collection_in)
    
    if exdct is not None:
        circles_ex = [mpatch.Circle(centre[[axis1, axis2]], rscale) for (centre, rscale) in zip(exdct['pos'], exdct['rad'])]
        offsets = np.array([[1., 1.], [1., 0.], [1., -1.], [0., 1.], [0., -1.], [-1., 1.], [-1., 0.], [-1., -1.]]) * size
        if periodic:
            circles_ex = circles_ex + [mpatch.Circle(centre[[axis1, axis2]] + offset, rscale) for (centre, rscale) in zip(exdct['pos'], exdct['rad']) for offset in offsets]
        collection_ex = PatchCollection(circles_ex, edgecolors='red', facecolor='none')
        ax.add_collection(collection_ex)
        
    ax.set_xlim(0., xpix*pixsize)
    ax.set_ylim(0., ypix*pixsize)
    ax.set_xlabel(['X', 'Y', 'Z'][axis1] + ' [cMpc]', fontsize=fontsize)
    ax.set_ylabel(['X', 'Y', 'Z'][axis2] + ' [cMpc]', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-1, which='both', direction='in', right=True, top=True)
    
    legend_handles = [mlines.Line2D([], [], color='blue', label='incl. halos'),\
                      mlines.Line2D([], [], color='red',  label='excl. halos'),\
                      mpatch.Patch(facecolor=color_true, edgecolor='black',label='True'),\
                      mpatch.Patch(facecolor=color_false, edgecolor='black',label='False'),\
                      ]
    lax.legend(handles=legend_handles, ncol=2, fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    lax.axis('off')
    
    pax.text(0.05, 0.95, parstring, horizontalalignment='left', verticalalignment='top', transform=pax.transAxes, fontsize=fontsize)
    pax.axis('off')

    plt.savefig('/net/luttero/data2/imgs/CGM/halomask_test/test_%s_%s.pdf'%(maskfile.split('/')[-1][:-5], halocat.split('/')[-1][:-5]), format='pdf')  
    
def make_mock_halocat(z=0.1):
    '''
    meant to test extraction in the rdist pipeline; galaxies are just centrally
    peaked power law profiles, with values in different slices determined by 
    evalutating the PL at the slice center
    '''
    boxsize = 25.
    
    #### generate the halo catalogue
    hpar = cu.c.hubbleparam
    cosmopars = {'h': hpar,\
                 'z': z,\
                 'a': 1./ (1. + z),\
                 'omegab': cu.c.omegabaryon,\
                 'omegam': cu.c.omega0,\
                 'omegalambda': cu.c.omegalambda,\
                 'boxsize': boxsize * hpar}
    
    ### galaxies:
    xyzcop = np.array([[ 0.,     0.,    0.  ],\
                       [ 6.25,  12.5,  18.75],\
                       [ 3.125,  5.,    7.  ],\
                       [12.6,   24.9,  14.6 ],\
                       [12.65,  24.85, 17.8 ],\
                       [18.2,   18.45, 15.5 ],\
                      ])
    vxyz  = np.array([[0.]*3,\
                      [-100.]*3,\
                      [200.]*3,\
                      [200.]*3,\
                      [-200.]*3,\
                      [-500., 700., 0.]])
    
    galaxyid = np.arange(6)
    R200cpkpc = np.array([100., 200., 50., 25., 300., 150.])
    M200cMsun = R200cpkpc**3 * 1e4 # values don't really matter, but get the scaling about right
    MstarMsun = np.ones(6)
    Vmax      = np.ones(6)
    sfr       = np.ones(6)
    subnums   = np.zeros(6)
    
    with h5py.File(ol.pdir + 'catalogue_mockhalos.hdf5', 'a') as fo:
        cp = fo.create_group('Header/cosmopars')
        for key in cosmopars.keys():
            cp.attrs.create(key, cosmopars[key])
        fo.create_dataset('Xcop_cMpc', data=xyzcop[:, 0])
        fo.create_dataset('Ycop_cMpc', data=xyzcop[:, 1])
        fo.create_dataset('Zcop_cMpc', data=xyzcop[:, 2])
        
        fo.create_dataset('VXpec_kmps', data=vxyz[:, 0])
        fo.create_dataset('VYpec_kmps', data=vxyz[:, 1])
        fo.create_dataset('VZpec_kmps', data=vxyz[:, 2])
        
        fo.create_dataset('galaxyid',  data=galaxyid)
        fo.create_dataset('M200c_Msun', data=M200cMsun)
        fo.create_dataset('R200c_pkpc', data=R200cpkpc)
        fo.create_dataset('Mstar_Msun', data=MstarMsun)
        fo.create_dataset('Vmax_circ_kmps', data=Vmax)
        fo.create_dataset('SubGroupNumber', data=subnums)
        fo.create_dataset('SFR_MsunPerYr', data=sfr)
        

def make_mock_projections(halocat, npix=500, nslices=20, velspace=False):
    
    with h5py.File(halocat, 'r') as fi:
        cosmopars = {key: item for (key, item) in fi['Header/cosmopars'].attrs.items()}
        xyzcen  = np.array([np.array(fi['Xcop_cMpc']), np.array(fi['Ycop_cMpc']), np.array(fi['Zcop_cMpc'])]).T
        rvir    = np.array(fi['R200c_pkpc']) / cosmopars['a'] * 1e-3
        mass    = np.array(fi['M200c_Msun'])

    zcens_sl = (np.arange(nslices) + 0.5) * cosmopars['boxsize'] / cosmopars['h'] / float(nslices) 
    name_sl  = ol.ndir + 'mockprojections_%s_zcen%s'%(halocat.split('/')[-1][:-5], '%s')

    box     = cosmopars['boxsize'] / cosmopars['h']
    out     = np.zeros((nslices, npix, npix))
    xygrid  = (np.indices((npix, npix)) + 0.5) / float(npix) * box
    
    for hi in range(len(mass)):
        xdists = np.min([np.abs((xygrid[0] - xyzcen[hi][0]) % box), np.abs((xyzcen[hi][0] - xygrid[0]) % box)], axis=0)
        ydists = np.min([np.abs((xygrid[1] - xyzcen[hi][1]) % box), np.abs((xyzcen[hi][1] - xygrid[1]) % box)], axis=0)
        zdists = np.min([np.abs((zcens_sl - xyzcen[hi][2]) % box), np.abs((xyzcen[hi][2] - zcens_sl) % box)], axis=0)
        totdists2 = (xdists**2)[np.newaxis, :, :]  + (ydists**2)[np.newaxis, :, :]  + (zdists**2)[:, np.newaxis, np.newaxis]
                
        rv = rvir[hi]
        mv = mass[hi]
        totdists2 *= 1. / rv**2
        sel = totdists2 <= 50.**2 # go out this far to make sure to get something; halos can be pretty far from slice centers in Rvir units

        out[sel] += mv * (totdists2[sel])**-2
    
    out = np.log10(out)
    
    for si in range(nslices):
        zcen_sl = zcens_sl[si]
        name    = name_sl%(zcen_sl)
        np.savez(name, out[si])
        
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        argset = int(sys.argv[1])
    else:
        argset = 0
    testhalomask_basic(argset=argset)
