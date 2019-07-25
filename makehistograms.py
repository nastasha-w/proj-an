'''
for making histograms (in n dimensions) from images 
'''
import numpy as np
import make_maps_opts_locs as ol

ndir = ol.ndir
mdir = '/net/luttero/data2/imgs/eagle_vs_bahamas_illustris/' # luttero location
pdir = ol.pdir

def makehist(arrdict,**kwargs):
    '''
    pretty thin wrapper for histogramdd; just flattens images and applies isfinite mask
    arrdict is a dictionary with a list of arrays for each key 
    (loops over keys and adds up the results if there are multiple keys)
    '''
    

    if len(arrdict.keys()) > 1:
        setrange = False
        if 'bins' not in kwargs.keys() and 'range' not in kwargs.keys():
            # need to synchronise bins between arrays to be histogrammed 
            # if we want to add them -> get min and max
            kwargs.update({'bins': 50}) # set number of bins to a default (histogramdd default is 10)
            setrange = True
        elif isinstance(kwargs['bins'],int): # only number of bins is set
            setrange = True
        else: # bins must be a sequence
            setrange = [isinstance(binset,int) for binset in kwargs['bins']]

        if np.any(np.array(setrange)):
            print('For multiple arrays to histogram together, a range or bin edges must be specified')
            return None
           
        first = True
        for key in arrdict.keys():
            kwargssub = kwargs
            if 'weights' in kwargs.keys(): # weights can be set for each array separately
                if isinstance(kwargs['weights'],dict):
                    kwargssub.update({'weights': kwargs['weights'][key]})

            if first: # need to set up  
                #print('Calling makehist:')
                #print('arrdict: %s'%str(arrdict))
                #print('kwargs: %s'%str(kwargssub))
                outhist, edges = makehist({key: arrdict[key]}, **kwargssub)
                first = False
            else:
                #print('Calling makehist:')
                #print('arrdict: %s'%str(arrdict))
                #print('kwargs: %s'%str(kwargssub))
                outhistsub, edgessub = makehist({key: arrdict[key]}, **kwargssub)
                #print edgessub
                if not np.all([np.all(np.array(edgessub[ind])==np.array(edges[ind]))] for ind in range(len(edgessub))): # if edges do not match, the histograms should not be added; need the per index check because the array objects are different even though they contain the same values
                    print('Error: edge mismatch when processing array %s'%key)
                    return None
                outhist += outhistsub

    else: # len(arrdict) <= 1:
        inarrs = arrdict[arrdict.keys()[0]] # extract arrays
        #print(inarrs)
        allfinite = np.all(np.array([np.isfinite(inarrs[ind]) for ind in range(len(inarrs))]),axis=0) # check where all arrays are finite 
        print('For key %s, %i values were excluded for being non-finite.'%(arrdict.keys()[0],np.prod(allfinite.shape)-np.sum(allfinite)))
        inarrs = [(inarr[allfinite]).flatten() for inarr in inarrs] # exclude non-finite values and flatten arrays
        outhist, edges = np.histogramdd(inarrs,**kwargs)
    
    return outhist, edges        


def makehist_fromnpz(*filenames,**kwargs): 
    # number of filenames is unknown beforehand (corresponds to number of dimensions in final histogram)
    '''
    kwargs are passed on to makehist -> histogramdd, 
    except that weight arrays can be specified by file 
    (use fills as dict keys to match main dict)
    and except fills=None,save=None,dimlabels = None defaults
    '''
    filenames = list(filenames)
    for ind in range(len(filenames)):
        if '/' not in filenames[ind]:
            filenames[ind] = ndir + filenames[ind] # reasonable guess for the right directory
	    if filenames[ind][-4:] != '.npz':
	        filenames[ind] = filenames[ind] + '.npz'

    # extract and setset defaults for kwargs not meant for makehistogram, then remove before passing to makehistogram
    if 'fills' not in kwargs.keys():
        fills = None
    else:
        fills = kwargs['fills']
        del kwargs['fills']
    if 'save' not in kwargs.keys():
        save = None
    else:
        save = kwargs['save']
        del kwargs['save']
    if 'dimlabels' not in kwargs.keys():
        dimlabels = None
    else:
        dimlabels = kwargs['dimlabels']    
        del kwargs['dimlabels']
    if 'sel' not in kwargs.keys():
        sel = slice(None, None, None)
    else:
        sel = kwargs['sel']
        del kwargs['sel']
    if fills is None:
        arrdict = {'arr': [np.load(filename)['arr_0'][sel] for filename in filenames]}
    else:
        arrdict = {fill: [np.load(filename%fill)['arr_0'][sel] for filename in filenames] for fill in fills}
    #print('Calling makehist:')
    #print('arrdict: %s'%str(arrdict))
    #print('kwargs: %s'%str(kwargs))
    hist, edges = makehist(arrdict,**kwargs)

    if save is not None:
        tosave = {'bins':hist, 'edges':edges}
        if dimlabels is not None: # store array of what is histogrammed over each dimension
            tosave.update({'dimension':np.array(dimlabels)}) 
        np.savez(save,**tosave)     
    
    return hist, edges
    
    
## example call: (using 0.1 bins and 0.1-rounded values covering the full data range)
# import makehistograms as mh
# import numpy as np
# mh.makehist_fromnpz('Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.2_C2Sm_32000pix_100.0slice_z-projection.npz',\
#                     'Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.2_C2Sm_32000pix_100.0slice_z-projection.npz',\
#                     'coldens_o7_L0100N1504_28_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_100.0slice_z-projection_T4EOS.npz',\
#                     save = '/net/luttero/data2/proc/hist_coldens_o7_L0100N1504_28_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_100.0slice_z-projection_T4EOS_and_weighted_Temperature_Density',\
#                      bins = [],\ 
#                     dimlabels = np.array(['Density_w_NO7', 'Temperature_w_NO7', 'NO7'])   )
#
# Error found in temperature projection: all temperatures in projected map came out as -inf
# Density: same, NO7: same
# Diff shows that in the run files, the only differences between EA100 and BA400 are the expected ones: simulation, simnum, snapnum, Ls, centre, LsinMpc 
# log files (temperature): pre-projection diagnostics look reasonable (lognH and logT min, max, median)
#                          post-projection min/max and sums are zero
# one clue is that NumPart is somehow negative (this is not true inn the BA400 1of1 logfile)
# the zero padding for < 100 particles in the projection routine would then lead to replacing the whole array with 100 zeros
# the NumPart output comes from the C routine, though
# looking into ctypes, it looks like 1504^3 overflows the c_int type. 1504^3/2 does not do this, so it seems that is why this has not been an issue before
# 1024^3 is ok, so BAHAMAS shouldn't hit this snag
# but it is set by NumPart = vardict_WQ.particle['coords_cMpc-vel'].shape[0], which really shouldn't output negative numbers
# make_maps: added in extra min/max/median checks for diagnostics
# 
#
# mh.makehist_fromnpz('Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
#                     'Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
#                     'coldens_o7_BA-L400N1024_32_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz',\
#                     fills = ['33.3333333333', '100.0','166.666666667','233.333333333','300.0','366.666666667'],\
#                     save = '/net/luttero/data2/proc/coldens_o7_BA-L400N1024_32_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_Temperature_Density',\
#                     bins = [np.arange(-31.1,-24.39,0.1), np.arange(3.2,6.71,0.1), + np.arange(13.4,16.91,0.1)],\
#                     dimlabels = np.array(['Density_w_NO7', 'Temperature_w_NO7', 'NO7'])   )
# For key 366.666666667, 0 values were excluded for being non-finite.
# For key 33.3333333333, 0 values were excluded for being non-finite.
# For key 300.0, 0 values were excluded for being non-finite.
# For key 166.666666667, 0 values were excluded for being non-finite.
# For key 233.333333333, 0 values were excluded for being non-finite.
# For key 100.0, 0 values were excluded for being non-finite.


def getminmax_fromnpz(filename,fills=None):
    if fills is None:
        arr = np.load(filename)
        if 'minfinite' in arr.keys() and 'max' in arr.keys():
            minval = float(arr['minfinite'])
            maxval = float(arr['max'])
        else:
            arr = arr['arr_0']
            arr = arr[np.isfinite(arr)]
            maxval = np.max(arr)
            minval = np.min(arr)
    else: 
        minval = np.inf
        maxval = -1*np.inf
        for fill in fills:
            submin, submax = getminmax_fromnpz(filename%fill)
            print('Min, max for %s: %f, %f'%(fill, submin, submax))
            minval = min(submin,minval)
            maxval = max(submax,maxval)
    return minval,maxval




####### plotting these histograms in various ways 
# loading will only work if all files are present

# load 3D histograms, divide out number of pixels
def dictfromnpz(filename):
    npz = np.load(filename)
    dct = {key: npz[key] for key in npz.keys()}
    return dct

# T, rho, NO7 dicts
h3ba = dictfromnpz(pdir + 'hist_coldens_o7_BA-L400N1024_32_test3.2_PtAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_Temperature_Density.npz')
h3ba['bins'] /= (6*32000**2)

h3eahi = dictfromnpz(pdir + 'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density.npz')
h3eahi['bins'] /= (32000**2)

h3eami = dictfromnpz(pdir + 'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_5600pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density.npz')
h3eami['bins'] /= (5600**2)

maxminfrac = 1./5600**2

# NO7, NH dicts
h3bah = dictfromnpz(pdir + 'hist_coldens_o7-and-hydrogen_BA-L400N1024_32_test3.2_PtAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz')

h3bah['bins'] /= (6*32000**2)

h3eahih = dictfromnpz(pdir + 'hist_coldens_o7-and-hydrogen_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_cens-sum_xyz-projection-average_T4EOS_totalbox.npz')

h3eahih['bins'] /= (3*32000**2)

h3eamih = dictfromnpz(pdir + 'hist_coldens_o7-and-hydrogen_L0100N1504_28_test3.1_PtAb_C2Sm_5600pix_6.25slice_cens-sum_xyz-projection-average_T4EOS_totalbox.npz')

h3eamih['bins'] /= (3*5600**2)

maxminfrac = 1./(3*5600**2)

# load ion balance table (z=0. matches snapshots)
import make_maps_v3_master as m3
o7_ib, logTK_ib, lognHcm3_ib = m3.findiontables('o7',0.0)


# imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines

import makecddfs as mc
import eagle_constants_and_units as c #only use for physical constants and unit conversion!

# conversions and cosmology
cosmopars_ea = mc.getcosmopars('L0100N1504',28,'REFERENCE',file_type = 'snap',simulation = 'eagle')
cosmopars_ba = mc.getcosmopars('L400N1024',32,'REFERENCE',file_type = 'snap',simulation = 'bahamas')

# both in cgs, using primordial abundance from EAGLE
rho_to_nh = 0.752/(c.atomw_H*c.u)

logrhocgs_ib = lognHcm3_ib - np.log10(rho_to_nh)

logrhob_av_ea = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ea['h']**2 * cosmopars_ea['omegab'] ) 
logrhob_av_ba = np.log10( 3./(8.*np.pi*c.gravity)*c.hubble**2 * cosmopars_ba['h']**2 * cosmopars_ba['omegab'] )

## settings/choices
fontsize=12


def getlabel(hist,axis):
    if 'Temperature' in hist['dimension'][axis]:
        return r'$\log_{10} T \, [K], N_{\mathrm{O VII}} \mathrm{-weighted}$'
    elif 'Density' in hist['dimension'][axis]:
        return r'$\log_{10} \rho \, [\mathrm{g}\,\mathrm{cm}^{-3}], N_{\mathrm{O VII}} \mathrm{-weighted}$'    
    elif 'NO7' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{O VII}} \, [\mathrm{cm}^{-2}]$'
    elif 'NH' in hist['dimension'][axis]:
        return r'$\log_{10} N_{\mathrm{H}} \, [\mathrm{cm}^{-2}]$'
    else:
        print('No label found for axis %i: %s'%(axis, hist['dimension'][axis]))
	return None

        
def add_2dplot(ax,hist3d,toplotaxes,log=True,**kwargs):
    # hist3d can be a histogram of any number >=2 of dimensions
    # like in plot1d, get the number of axes from the length of the edges array
    summedaxes = tuple(list( set(range(len(hist3d['edges'])))-set(toplotaxes) )) # get axes to sum over
    toplotaxes= list(toplotaxes)
    #toplotaxes.sort()
    axis1, axis2 = tuple(toplotaxes)
    # sum over non-plotted axes
    if len(summedaxes) == 0:
        imgtoplot = hist3d['bins']
    else:
        imgtoplot = np.sum(hist3d['bins'],axis=summedaxes)

    if log:
        imgtoplot = np.log10(imgtoplot)
    # transpose plot if axes not in standard order; normally, need to use transposed array in image
    if axis1 < axis2:
        imgtoplot = imgtoplot.T
    img = ax.imshow(imgtoplot,origin='lower',interpolation='nearest',extent=(hist3d['edges'][axis1][0],hist3d['edges'][axis1][-1],hist3d['edges'][axis2][0],hist3d['edges'][axis2][-1]),**kwargs)
    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
    else:
        vmin = np.min(imgtoplot[np.isfinite(imgtoplot)])
    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
    else:
        vmax = np.max(imgtoplot[np.isfinite(imgtoplot)])
    return img, vmin, vmax

def getminmax2d(hist3d, axis=None, log=True): 
    # axis = axis to sum over; None -> don't sum over any axes 
    # now works for histgrams of general dimensions
    if axis is None:
        imgtoplot = hist3d['bins']
    else:
        imgtoplot = np.sum(hist3d['bins'],axis=axis)
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

def add_1dplot(ax,hist3d,axis1,log=True,**kwargs):
    # edges is an object array, contains array for each dimension -> number of dimesions is length of edges
    # modified so this works for any dimension of histogram
    chosenaxes = list(set(range(len(hist3d['edges'])))-set([axis1])) # get axes to sum over: all except axis1
    if len(chosenaxes) == 0:
        bins = hist3d['bins']
    else: # if histogram has more than 1 dimension, sum over the others
        bins = np.sum(hist3d['bins'],axis=tuple(chosenaxes))
    # plot the histogram on ax 	
    ax.step(hist3d['edges'][axis1][:-1],bins,where = 'post',**kwargs)
    if log:
        ax.set_yscale('log')

def add_colorbar(ax,img=None,vmin=None,vmax=None,cmap=None,clabel=None,newax=False,extend='neither',fontsize=fontsize):
    if img is None:
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap,norm=norm,orientation='vertical',extend=extend)
    elif newax:
        div = axgrid.make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.2)
        cbar = mpl.colorbar.Colorbar(cax,img,extend=extend)
    else:
        cbar = mpl.colorbar.Colorbar(ax,img,extend=extend)
    ax.tick_params(labelsize=fontsize-2)
    if clabel is not None:
        cbar.set_label(clabel,fontsize=fontsize)

def add_ax2rho(ax,xory = 'x',fontsize=fontsize):

    if xory == 'x':
        ax2 = ax.twiny()
        old_ticklocs = ax.get_xticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
	
	# use same spacing and number of ticks, but start at first integer value in the new units
	new_lim = old_xlim + np.log10(rho_to_nh)
	numticks = len(old_ticklocs)
	newticks = np.ceil(new_lim[0]) + np.array(old_ticklocs) - old_ticklocs[0]
	newticks = np.round(newticks,2)
        newticklabels = [str(int(tick)) if int(tick)== tick else str(tick) for tick in newticks]
   	
	#print old_ticklocs
	print newticklabels
        #ax2.set_xticks(np.round(old_ticklocs + np.log10(rho_to_nh),2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        #ax2.set_xticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)]) 
	ax2.set_xticks(newticks - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        ax2.set_xticklabels(newticklabels)             
        ax2.set_xlabel(r'$\log_{10} n_H \, [\mathrm{cm}^{-3}], f_H = 0.752$',fontsize=fontsize)
        ax2.set_xlim(old_xlim)
        ax2.set_ylim(old_ylim)
    else:
        ax2 = ax.twinx()
        old_ticklocs = ax.get_yticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
        ax2.set_yticks(np.round(old_ticklocs + np.log10(rho_to_nh),2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        ax2.set_yticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)])        
        ax2.set_ylabel(r'$\log_{10} n_H \, [\mathrm{cm}^{-3}], f_H = 0.752$',fontsize=fontsize)
        ax2.set_xlim(old_xlim)
        ax2.set_ylim(old_ylim)
    ax2.tick_params(labelsize=fontsize,axis='both')
    return ax2

def add_rhoavx(ax,onlyeagle=False,eacolor='lightgray',bacolor='gray'):
    if onlyeagle:
        ealabel = r'$\overline{\rho_b}$'
    else:
        ealabel = r'EAGLE $\overline{\rho_b}$'
	 
    ax.axvline(x=logrhob_av_ea,ymin=0.,ymax=1.,color=eacolor,linewidth=1,label = ealabel)
    if not onlyeagle:
        ax.axvline(x=logrhob_av_ba,ymin=0.,ymax=1.,color=bacolor,linewidth=1,label = r'BAHAMAS $\overline{\rho_b}$')

def add_collsional_ionbal(ax,**kwargs):
    ax.plot(logTK_ib,o7_ib[-1,:],**kwargs)
    ax.set_ylabel('fraction of O atoms in O VII state',fontsize=fontsize)

def add_ionbal_contours(ax,legend=True,**kwargs):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    logrho, logT = np.meshgrid(logrhocgs_ib, logTK_ib)
    levels = [1e-3,3e-2, 0.3, 0.9]        
    colors = ['mediumvioletred','magenta','orchid','palevioletred']
    contours = ax.contour(logrho, logT, o7_ib.T, levels, colors = colors, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # make a legend to avoid crowding plot region
    for i in range(len(levels)):
        contours.collections[i].set_label('%.0e'%levels[i])
    if legend:
        ax.legend(loc='lower right',title=r'$f_{\mathrm{O VII}}, f_H=0.752$')
    #ax.clabel(contours, inline=True, fontsize=fontsize, inline_spacing = 0,manual = [(-28.,6.7),(-26.,6.5),(-24.,6.3),(-27.,5.5),(-28.5,4.5),(-25.,5.2)])

def add_ionbal_img(ax,**kwargs):
    # this is intended as a background: keep the limits of the data plot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # easy to change to options if I want o8 or something later
    ionbal = o7_ib
    logT = logTK_ib
    logrho = logrhocgs_ib

    # color bar handling (defaults etc. are handled here -> remove from kwargs before passing to imshow)
    if 'cmap' in kwargs:
        cmap = mpl.cm.get_cmap(kwargs['cmap'])
        del kwargs['cmap'] 
    else: # default color map
        cmap = mpl.cm.get_cmap('gist_gray')
    cmap.set_under(cmap(0))
    ax.set_facecolor(cmap(0))
    
    # to get extents, we need bin edges rather than centres; tables have evenly spaced logrho, logT
    logrho_diff = np.average(np.diff(logrho))
    logrho_ib_edges = np.array(list(logrho - logrho_diff/2.) \
                                  + [logrho[-1] + logrho_diff/2.])
    logT_diff = np.average(np.diff(logT))				  
    logT_ib_edges = np.array(list(logT - logT_diff/2.) \
                                  + [logT[-1] + logT_diff/2.])
				  
    img = ax.imshow(np.log10(ionbal.T),origin='lower',interpolation='nearest',extent=(logrho_ib_edges[0],logrho_ib_edges[-1],logT_ib_edges[0],logT_ib_edges[-1]),cmap=cmap,**kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]), adjustable='box-forced')
    
    if 'vmin' in kwargs.keys():
        vmin = kwargs['vmin']
    else:
        vmin = np.min(ionbal[np.isfinite(ionbal)])
    if 'vmax' in kwargs.keys():
        vmax = kwargs['vmax']
    else:
        vmax = np.max(ionbal[np.isfinite(ionbal)])
    return img, vmin, vmax

    
def add_2dhist_contours(ax,hist,toplotaxes,mins= None, maxs=None, histlegend=True, fraclevels=True, levels=None, legend = True, dimlabels=None,legendlabel = None, legendlabel_pre = None, **kwargs):
    '''
    colors, linestyles: through kwargs
    othersmin and othersmax should be indices along the corresponding histogram axes
    assumes xlim, ylim are already set as desired
    dimlabels can be used to override (long) dimension labels from hist
    '''
    # get axes to sum over; preserve order of other axes to match limits
    summedaxes = range(len(hist['edges']))
    summedaxes.remove(toplotaxes[0])
    summedaxes.remove(toplotaxes[1])
    
    # handle some defaults
    bins = hist['bins']
    edges = hist['edges']
    if dimlabels is None:
        dimlabels = [getlabel(hist,axis) for axis in range(len(edges))]    
    if mins is None:
        mins= (None,)*len(edges)
    if maxs is None:
        maxs = (None,)*len(edges)
	
    # get the selection of min/maxs and applay the selection
    sels = [slice(mins[i],maxs[i],None) for i in range(len(edges))]
    sels = tuple(sels)
    
    if len(summedaxes) >0:
        binsum = np.sum(bins[sels],axis=tuple(summedaxes))
    else:
        binsum = bins[sels]

    binfrac = np.sum(binsum)/np.sum(bins) # fraction of total bins selected
    # add min < dimension_quantity < max in legend label
    if legendlabel is None:
        labelparts = [r'%.1f $<$ %s $<$ %.1f, '%(edges[i][mins[i]],dimlabels[i],edges[i][maxs[i]]) if (mins[i] is not None and maxs[i] is not None) else\
                      r'%.1f $<$ %s, '%(edges[i][mins[i]],dimlabels[i])                  if (mins[i] is not None and maxs[i] is None)     else\
		      r'%s $<$ %.1f, '%(dimlabels[i],edges[i][maxs[i]])                  if (mins[i] is None and maxs[i] is not None)     else\
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
	binsumcopy = binsum.copy() # copy to rework
	binsumcopy = binsumcopy.reshape(np.prod(binsumcopy.shape))
	binsumcopy.sort() # get all histogram values in order (low to high)
	binsumcopy = np.flipud(binsumcopy) # flip to high-to-low
	cumul = np.cumsum(binsumcopy) # add values high-to-low 
        wherelist = [[(np.where(cumul<=level))[0],(np.where(cumul>=level))[0]] for level in levels] # list of max-lower and min-higher indices

	# sort out list: where arrays may be empty -> levels outside 0,1 range, probabaly
	# set value level 0 for level == 1. -> just want everything (may have no cumulative values that large due to fp errors)
	# if all cumulative values are too high (maxmimum bin has too high a fraction), set to first cumulative value (=max bin value)
	# otherwise: interpolate values, or use overlap
	#print binsumcopy, binsumcopy.shape
	#print cumul
	#print wherelist
	#return levels, cumul, binsumcopy, wherelist
	valslist = [cumul[0]  if  wherelist[i][0].shape == (0,) else\
	            0.        if (wherelist[i][1].shape == (0,) or levels[i] == 1) else\
		    np.interp([levels[i]], np.array([      cumul[wherelist[i][0][-1]],      cumul[wherelist[i][1][0]] ]),\
		                           np.array([ binsumcopy[wherelist[i][0][-1]], binsumcopy[wherelist[i][1][0]] ]))[0]\
		    for i in range(len(levels))]
        #for i in range(len(levels)):
        #    if not (wherelist[i][0].shape == (0,) or wherelist[i][1].shape == (0,)):
	#        print('interpolating (%f, %f) <- index %i and (%f, %f)  <- index %i to %f'\
	#	 %(cumul[wherelist[i][0][-1]],binsumcopy[wherelist[i][0][-1]],wherelist[i][0][-1],\
	#          cumul[wherelist[i][1][0]], binsumcopy[wherelist[i][1][0]], wherelist[i][1][0],\
	#	   levels[i]) )
        #print(np.all(np.diff(binsumcopy)>=0.))
	uselevels = valslist
	print('Desired cumulative fraction levels were %s; using value levels %s'%(levels,uselevels))
    else:
        uselevels=levels
            
    #print binsum, binsum.shape
    if 'linestyles' in kwargs:        
        linestyles = kwargs['linestyles']
    else:
        linestyles = [] # to not break the legend search
    # get pixel centres from edges
    centres0 = edges[toplotaxes[0]][:-1] +0.5*np.diff(edges[toplotaxes[0]]) 
    centres1 = edges[toplotaxes[1]][:-1] +0.5*np.diff(edges[toplotaxes[1]])
    contours = ax.contour(centres0, centres1, binsum.T, uselevels, **kwargs)
    #ax.set_xlim(xlim)
    #ax.set_ylim(ylim)
    # make a legend to avoid crowding plot region
    #for i in range(len(levels)):
    #    contours.collections[i].set_label('%.0e'%levels[i])
    # color only legend; get a solid line in the legend
    
    ax.tick_params(labelsize=fontsize,axis='both')
    if 'solid' in linestyles:
        contours.collections[np.where(np.array(linestyles)=='solid')[0][0]].set_label(legendlabel)
    else: # just do the first one
        contours.collections[0].set_label(legendlabel)
    if histlegend:
        ax.legend(loc='lower right',title=r'$f_{\mathrm{O VII}}, f_H=0.752$')
    



def plotrhohists():
    fig, ax = plt.subplots(nrows=1,ncols=1)
    add_1dplot(ax,h3ba,0,color='blue',label='BAHAMAS')
    add_1dplot(ax,h3eahi,0,color='red',label='EAGLE, 32k pix')
    add_1dplot(ax,h3eami,0,color='orange',label='EAGLE, 5600 pix',linestyle = 'dashed')
    ax.set_xlabel(getlabel(h3ba,0),fontsize=fontsize)
    ax.set_ylabel('fraction of pixels',fontsize=fontsize)
    add_rhoavx(ax)
    ax.legend(fontsize=fontsize)
    add_ax2rho(ax)
    plt.savefig(mdir + 'rho_histograms.png',format = 'png',bbox_inches='tight')

def plotThists():
    fig, ax = plt.subplots(nrows=1,ncols=1)
    add_1dplot(ax,h3ba,1,color='blue',label='BAHAMAS')
    add_1dplot(ax,h3eahi,1,color='red',label='EAGLE, 32k pix')
    add_1dplot(ax,h3eami,1,color='orange',label='EAGLE, 5600 pix',linestyle = 'dashed')
    ax.set_xlabel(getlabel(h3ba,1),fontsize=fontsize)
    ax.set_ylabel('fraction of pixels',fontsize=fontsize)

    ax2 = ax.twinx()
    add_collsional_ionbal(ax2,color='gray',linestyle='dotted',label='coll. ion. eq.')
    ax2.set_ylim(1e-9,None)
    ax2.set_yscale('log')

    ax.set_xlim(3.,8.)
    ax2.set_xlim(ax.get_xlim())

    ax.legend(fontsize=fontsize)
    ax2.legend(fontsize=fontsize)
    plt.savefig(mdir + 'T_histograms.png',format = 'png',bbox_inches='tight')

def plotNO7hists(usehhists=False):
    fig, ax = plt.subplots(nrows=1,ncols=1)
    if usehhists:
        add_1dplot(ax,h3bah,0,color='blue',label='BAHAMAS, z')
        add_1dplot(ax,h3eahih,0,color='red',label='EAGLE, xyz, 32k pix')
        add_1dplot(ax,h3eamih,0,color='orange',label='EAGLE, xyz, 5600 pix',linestyle = 'dashed')
        ax.set_xlabel(getlabel(h3bah,0),fontsize=fontsize)
        ax.set_ylabel('fraction of pixels',fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_xlim(7.,None)
        plt.savefig(mdir + 'NO7_histograms_from_o7-hydrogen.png',format = 'png',bbox_inches='tight')
        
    else:
        add_1dplot(ax,h3ba,2,color='blue',label='BAHAMAS')
        add_1dplot(ax,h3eahi,2,color='red',label='EAGLE, 32k pix')
        add_1dplot(ax,h3eami,2,color='orange',label='EAGLE, 5600 pix',linestyle = 'dashed')
        ax.set_xlabel(getlabel(h3ba,2),fontsize=fontsize)
        ax.set_ylabel('fraction of pixels',fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        ax.set_xlim(7.,None)
        plt.savefig(mdir + 'NO7_histograms.png',format = 'png',bbox_inches='tight')
    
def plotNHhists():
    fig, ax = plt.subplots(nrows=1,ncols=1)
    add_1dplot(ax,h3bah,1,color='blue',label='BAHAMAS, z')
    add_1dplot(ax,h3eahih,1,color='red',label='EAGLE, xyz, 32k pix')
    add_1dplot(ax,h3eamih,1,color='orange',label='EAGLE, xyz, 5600 pix',linestyle = 'dashed')
    ax.set_xlabel(getlabel(h3bah,1),fontsize=fontsize)
    ax.set_ylabel('fraction of pixels',fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    #ax.set_xlim(7.,None)
    plt.savefig(mdir + 'NH_histograms.png',format = 'png',bbox_inches='tight')
        


def subplotrhoT(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ionballegend=True,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(0,1),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,1),fontsize=fontsize)
    ax2 = add_ax2rho(ax,xory = 'x')
    add_rhoavx(ax)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.15))
    # set the plot to be square
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim()  
    add_ionbal_contours(ax2,label='O7 fractions',legend=ionballegend) # will generally reset ax limits -> reset
    ax.set_xlim(xlim1)
    ax.set_ylim(ylim1)  
    ax2.set_ylim(ylim1)
    ax2.set_xlim(xlim1)
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    ax2.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    if mainlegend:
        ax.legend(loc= 'lower right')
    return img, vmin, vmax

def plotrhoT():
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.])
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(h3ba, 2)
    imin2,imax2 = getminmax2d(h3eahi, 2)
    imin3,imax3 = getminmax2d(h3eami, 2)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    cmap = 'nipy_spectral'
    img, vmin, vmax = subplotrhoT(ax1, h3ba ,title= 'BAHAMAS',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ionballegend=False)
    subplotrhoT(ax2, h3eahi ,title= 'EAGLE, 32k pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ionballegend=True)
    subplotrhoT(ax3, h3eami ,title= 'EAGLE, 5600 pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ionballegend=False)
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'phase_diagrams.png',format = 'png',bbox_inches='tight')



def subplotrhoNO7(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ionballegend=True,ymin=None,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(0,2),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,0),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    ax2 = add_ax2rho(ax,xory = 'x')
    add_rhoavx(ax)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.15))
    # set the plot to be square
    if ymin is not None:
        ax.set_ylim(ymin,ax.get_ylim()[1])
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim() 
    #print ylim1 
    #ax.set_xlim(xlim1)
    #ax.set_ylim(ylim1)  
    ax2.set_ylim(ylim1)
    ax2.set_xlim(xlim1)
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    ax2.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    if mainlegend:
        ax.legend(loc= 'lower right')
    return img, vmin, vmax

def plotrhoNO7():
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.])
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(h3ba, 1)
    imin2,imax2 = getminmax2d(h3eahi, 1)
    imin3,imax3 = getminmax2d(h3eami, 1)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    ymin=7.
    cmap = 'nipy_spectral'
 
    img, vmin, vmax = subplotrhoNO7(ax1, h3ba ,title= 'BAHAMAS',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ymin=ymin)
    subplotrhoNO7(ax2, h3eahi ,title= 'EAGLE, 32k pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    subplotrhoNO7(ax3, h3eami ,title= 'EAGLE, 5600 pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'rho_NO7.png',format = 'png',bbox_inches='tight')


def subplotTNO7(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ionballegend=True,ymin=None,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(1,2),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,2),fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.15))
    # set the plot to be square
    if ymin is not None:
        ax.set_ylim(ymin,ax.get_ylim()[1])
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim() 
    #print ylim1 
    #ax.set_xlim(xlim1)
    #ax.set_ylim(ylim1)  
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    if mainlegend:
        ax.legend(loc= 'lower right')
    return img, vmin, vmax

def plotTNO7():
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.])
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(h3ba, 0)
    imin2,imax2 = getminmax2d(h3eahi, 0)
    imin3,imax3 = getminmax2d(h3eami, 0)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    #ymin=None
    ymin=7.
    cmap = 'nipy_spectral'
 
    img, vmin, vmax = subplotTNO7(ax1, h3ba ,title= 'BAHAMAS',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ymin=ymin)
    subplotTNO7(ax2, h3eahi ,title= 'EAGLE, 32k pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    subplotTNO7(ax3, h3eami ,title= 'EAGLE, 5600 pix',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'T_NO7.png',format = 'png',bbox_inches='tight')


def subplotNHNO7(ax, hist, title=None,vmin=None,vmax=None,mainlegend=True,ymin=None,**kwargs):
    img,vmin,vmax = add_2dplot(ax,hist,(1,0),vmin=vmin,vmax=vmax,**kwargs)
    ax.set_xlabel(getlabel(hist,1),fontsize=fontsize)
    ax.set_ylabel(getlabel(hist,0),fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize+2, position = (0.5,1.02))
    # set the plot to be square
    if ymin is not None:
        ax.set_ylim(ymin,ax.get_ylim()[1])
    xlim1 = ax.get_xlim()
    ylim1 = ax.get_ylim() 
    #print ylim1 
    #ax.set_xlim(xlim1)
    #ax.set_ylim(ylim1)  
    ax.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    if mainlegend:
        ax.legend(loc= 'lower right')
    return img, vmin, vmax

def plotNHNO7():
    fig = plt.figure(figsize=(16.,5.))
    grid = gsp.GridSpec(1,4,width_ratios=[5.,5.,5.,1.],wspace=0.5,hspace=0.5)
    ax1, ax2, ax3, ax4 = tuple(plt.subplot(grid[i]) for i in range(4)) 
    
    # set all vmin, vmax to global min/max values to synchonise (plots are log, so min/max should match)
    imin1,imax1 = getminmax2d(h3bah)
    imin2,imax2 = getminmax2d(h3eahih)
    imin3,imax3 = getminmax2d(h3eamih)
    vmin = min([imin1,imin2,imin3])
    vmax = max([imax1,imax2,imax3])
    
    # set minimum column density
    #ymin=None
    ymin=7.
    cmap = 'nipy_spectral'
 
    img, vmin, vmax = subplotNHNO7(ax1, h3bah ,title= 'BAHAMAS, z-projection',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=True,ymin=ymin)
    subplotNHNO7(ax2, h3eahih ,title= 'EAGLE, 32k pix, xyz-average',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    subplotNHNO7(ax3, h3eamih ,title= 'EAGLE, 5600 pix, xyz-average',vmin=vmin,vmax=vmax,cmap = cmap,mainlegend=False,ymin=ymin)
    
    add_colorbar(ax4,img=img,clabel=r'$\log_{10}$ fraction of pixels')
    ax4.set_aspect(10., adjustable='box-forced')
    #ax4.set_aspect(10)
    plt.savefig(mdir + 'NH_NO7.png',format = 'png',bbox_inches='tight')


def plotrhoT_byNO7(slidemode = False):
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(2)) 
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3, ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(3))
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(h3eahi['edges'][0][0],h3eahi['edges'][0][-1])
    ax1.set_ylim(h3eahi['edges'][1][0],h3eahi['edges'][1][-1])
    ax1.set_ylabel(getlabel(h3eahi,1),fontsize=fontsize)
    ax1.set_xlabel(getlabel(h3eahi,0),fontsize=fontsize)
    
    # square plot
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (None,None,r'$N_{O VII}$')
    
    img, vmin, vmax = add_ionbal_img(ax1,cmap='gist_gray',vmin=-7.)
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10} \, n_{O VII} \,/ n_{O},\, f_H = 0.752$',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
    
    #add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,None), histlegend=False, fraclevels=True,\
    #                        levels=[0.99999], linestyles = ['dashdot'],colors = ['saddlebrown'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels)
			    
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels)
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'],dimlabels = dimlabels)	  
    #set average density indicator
    ax12 = add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    
    if not slidemode:	    
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_by_NO7.png',format = 'png',bbox_inches='tight') 
    else:
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_by_NO7_slide.png',format = 'png',bbox_inches='tight')


def plotrhoT_eaba_byNO7(slidemode = False,fontsize=14):
    # set up grid
    if not slidemode: # ax3 for legend under plot and colorbar
        fig = plt.figure(figsize=(8.,8.))
        grid = gsp.GridSpec(2,2,height_ratios=[6.,2.],width_ratios=[7.,1.],wspace=0.0)
        ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(2)) 
        ax3 = plt.subplot(grid[1,:])
	ncols_legend = 2
	legendloc=9
	legend_bbox_to_anchor=(0.5,1.)
    if slidemode: # ax3 for legend right of plot
        fontsize=14
        fig = plt.figure(figsize=(12.,6.))
        grid = gsp.GridSpec(1,3,width_ratios=[4.,7.,1.],wspace=0.0)
        ax3, ax1, ax2 = tuple(plt.subplot(grid[0,i]) for i in range(3))
	ncols_legend = 1 
	legendloc= 1
	legend_bbox_to_anchor=(0.95,1.)

    # set up x-y extents from data range
    ax1.set_xlim(h3eahi['edges'][0][0],h3eahi['edges'][0][-1])
    ax1.set_ylim(h3eahi['edges'][1][0],h3eahi['edges'][1][-1])
    ax1.set_ylabel(getlabel(h3eahi,1),fontsize=fontsize)
    ax1.set_xlabel(getlabel(h3eahi,0),fontsize=fontsize)
    
    # square plot
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()
    ax1.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')

    
    fraclevels = [0.99,0.50,0.10] 
    linestyles = ['dotted','dashed','solid']
    dimlabels = (None,None,r'$N_{O VII}$')
    
    img, vmin, vmax = add_ionbal_img(ax1,cmap='gist_gray',vmin=-7.)
    # add colorbar
    add_colorbar(ax2,img=img,clabel=r'$\log_{10} \, n_{O VII} \,/ n_{O},\, f_H = 0.752$',extend = 'min',fontsize=fontsize)
    ax2.set_aspect(10.)
    ax2.tick_params(labelsize=fontsize,axis='both')
    
    #add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,None), histlegend=False, fraclevels=True,\
    #                        levels=[0.99999], linestyles = ['dashdot'],colors = ['saddlebrown'],dimlabels = dimlabels)
    # EAGLE stnd-res
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['red','red','red'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')			    
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orange','orange','orange'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['gold','gold','gold'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['green','green','green'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['blue','blue','blue'],dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
    add_2dhist_contours(ax1,h3eahi,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['darkviolet','darkviolet','darkviolet'], dimlabels = dimlabels,\
			    legendlabel_pre = 'EA-hi, ')
	
	
    # EAGLE ba-res			    
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['firebrick','firebrick','firebrick'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')			    
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
    #                            levels=fraclevels, linestyles = linestyles,colors = ['chocolate','chocolate','chocolate'],dimlabels = dimlabels,\
    #		    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['goldenrod','goldenrod','goldenrod'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['darkgreen','darkgreen','darkgreen'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['darkblue','darkblue','darkblue'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
    #add_2dhist_contours(ax1,h3eami,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
    #                        levels=fraclevels, linestyles = linestyles,colors = ['purple','purple','purple'],dimlabels = dimlabels,\
    #			    legendlabel_pre = 'EA-mi, ')
			    
    # BAHAMAS			    
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,None), maxs=(None,None,250), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['lightcoral','lightcoral','lightcoral'],dimlabels = dimlabels,\
         		     legendlabel_pre = 'BA, ')			    
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,250), maxs=(None,None,270), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['sandybrown','sandybrown','sandybrown'],dimlabels = dimlabels,\
    		    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,270), maxs=(None,None,280), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['yellow','yellow','yellow'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,280), maxs=(None,None,290), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['lime','lime','lime'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,290), maxs=(None,None,300), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['dodgerblue','dodgerblue','dodgerblue'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')
    add_2dhist_contours(ax1,h3ba,(0,1),mins= (None,None,300), maxs=(None,None,None), histlegend=False, fraclevels=True,\
                            levels=fraclevels, linestyles = linestyles,colors = ['orchid','orchid','orchid'],dimlabels = dimlabels,\
    			    legendlabel_pre = 'BA, ')		    
			    	  
    #set average density indicator
    ax12 = add_rhoavx(ax1,onlyeagle=True,eacolor='darksalmon')
    ax12 = add_ax2rho(ax1,fontsize=fontsize) 
    ax12.set_ylim(ylim1)
    ax12.set_xlim(xlim1)
    ax12.set_aspect((xlim1[1]-xlim1[0])/(ylim1[1]-ylim1[0]), adjustable='box-forced')
    
    
    # set up legend in ax below main figure
    handles_subs, labels_subs = ax1.get_legend_handles_labels()
    level_legend_handles = [mlines.Line2D([], [], color='tan', linestyle = linestyles[i], label='%.1f%% enclosed'%(100.*fraclevels[i])) for i in range(len(fraclevels))]
    ax3.legend(handles=handles_subs + level_legend_handles,fontsize=fontsize,ncol=ncols_legend,loc=legendloc,bbox_to_anchor=legend_bbox_to_anchor)
    ax3.axis('off')	
    
    
    if not slidemode:	    
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_ea-hi-ba_by_NO7.png',format = 'png',bbox_inches='tight') 
    else:
        plt.savefig('/net/luttero/data2/imgs/gas_state/phase_diagram_ea-hi-ba_by_NO7_slide.png',format = 'png',bbox_inches='tight')
