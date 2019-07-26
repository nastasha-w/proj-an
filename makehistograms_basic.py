# -*- coding: utf-8 -*-
'''
Spyder Editor

for making histograms (in n dimensions) from images 

also has some specific histogram implementations 
(basic means: does not need make_maps as an import, and all the installation 
that comes with)
'''

import numpy as np
import h5py
import make_maps_opts_locs as ol

ndir = ol.ndir
mdir = '/net/luttero/data2/imgs/eagle_vs_bahamas_illustris/' # luttero location
pdir = ol.pdir

def makehist(arrdict,**kwargs):
    '''
    pretty thin wrapper for histogramdd; just flattens images and applies isfinite mask
    arrdict is a dictionary with a list of array npz files for each key; load only as needed! 
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
        kwargss = kwargs.copy()
        if 'sel' not in kwargss.keys():
            sel = slice(None, None, None)
        else:
            sel = kwargss['sel']
            del kwargss['sel']
        
        inarrs = [arr['arr_0'][sel] for arr in arrdict[arrdict.keys()[0]]] # extract arrays, load
        #print(inarrs)
        if kwargss['includeinf']:
            allfinite = slice(None, None, None)
            print('Doing key %s'%(arrdict.keys()[0]))
        else:
            allfinite = np.all(np.array([np.isfinite(inarrs[ind]) for ind in range(len(inarrs))]),axis=0) # check where all arrays are finite 
            print('For key %s, %i values were excluded for being non-finite.'%(arrdict.keys()[0],np.prod(allfinite.shape)-np.sum(allfinite)))
        del kwargss['includeinf']
        inarrs = [(inarr[allfinite]).flatten() for inarr in inarrs] # exclude non-finite values and flatten arrays  

        if len(inarrs) * len(inarrs[0]) > (32000**2 *4):
            sublen =  32000**2 * 4 // len(inarrs) 
            numsub =  (len(inarrs[0]) - 1) // sublen + 1
            if numsub * sublen < len(inarrs[0]):
                raise RuntimeError('A bug in the array looping for histograms would cuase only part of the input arrays to be used')
            for subi in range(numsub):
                subsel = slice(sublen * subi, sublen * (subi + 1))
                if 'weights' in kwargss.keys():
                    kwargsss = kwargss.copy()
                    kwargsss['weights'] = kwargss['weights'][subsel]
                else:
                    kwargsss = kwargss.copy()
                if subi == 0:
                    outhist, edges = np.histogramdd([inarr[subsel] for inarr in inarrs], **kwargsss)
                else:
                    outhistsub, edgessub = np.histogramdd([inarr[subsel] for inarr in inarrs], **kwargsss)
                    outhist += outhistsub
                    del outhistsub
                    if not np.all([np.all(edgessub[j] == edges[j]) for j in range(len(edges))]):
                        raise RuntimeError('Edges for subhistogram %i are different from previous ones'%subi)
        else:
            outhist, edges = np.histogramdd(inarrs,**kwargss)
        del inarrs # make sure memory is not used up by arrays we're done with
    
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
    
    if fills is None:
        arrdict_files = {'arr': [np.load(filename) for filename in filenames]}
    else:
        arrdict_files = {fill: [np.load(filename%fill) for filename in filenames] for fill in fills}
    if 'includeinf' not in kwargs.keys():
        kwargs['includeinf'] = False
    elif kwargs['includeinf']:
        if 'bins' not in kwargs.keys():
            raise ValueError('kw argument bins must be specified to include infinite values')
        else: # ensure bin lists start and end at infinite values
            kwargs['bins'] = [np.array([-np.inf] + list(edges)) if edges[0] != -np.inf else\
                              np.array(edges) for edges in kwargs['bins']]
            kwargs['bins'] = [np.array(list(edges) + [np.inf]) if edges[-1] != np.inf else\
                              np.array(edges) for edges in kwargs['bins']]
    #print('Calling makehist:')
    #print('arrdict: %s'%str(arrdict))
    #print('kwargs: %s'%str(kwargs))
    hist, edges = makehist(arrdict_files,**kwargs)

    if save is not None:
        tosave = {'bins':hist, 'edges':edges}
        if dimlabels is not None: # store array of what is histogrammed over each dimension
            tosave.update({'dimension':np.array(dimlabels)}) 
        np.savez(save,**tosave)     
    
    return hist, edges

def getminmax_fromnpz(filename, fills=None):
    if fills is None:
        arr = np.load(filename)
        if 'max' in arr.keys() and 'minfinite' in arr.keys():
            minval = arr['minfinite']
            maxval = arr['max']
        else: 
            arr = arr['arr_0']
            arr = arr[np.isfinite(arr)]
            maxval = np.max(arr)
            minval = np.min(arr)
    else: 
        minval = np.inf
        maxval = -1*np.inf
        for fill in fills:
            print('Doing fill %s'%fill)
            submin, submax = getminmax_fromnpz(filename%fill)
            minval = min(submin[()], minval)
            maxval = max(submax[()], maxval)
    return minval,maxval


def makehist_1slice_masked(*filenames, **kwargs):
    '''
    filenames:     names of corresponding files to histogram 
                   (number = histogram dimension)
    maskfilenames: (kwarg) names of files containing masks to apply to 
                   filenames. Each mask produces a different histogram.
                   the mask should be contained in a dataset called 'mask'
                   if not included, no mask is applied
    kwargs:        passed to histogramdd. Notably, bins should be set if
                   combining with other histograms       
    for multiple slices: use a wrapper giving different sets of filenames 
    '''

    kwargss = kwargs.copy()
    if 'bins' not in kwargs.keys():
        raise ValueError('Histogram bins must be specified')
   
    filenames = [ol.ndir + filen if '/' not in filen else filen for filen in filenames]
    filenames = [filen + '.npz' if filen[-4:] != '.npz' else filen for filen in filenames]
    files = [np.load(filen) for filen in filenames]
 
    inarrs = [arr['arr_0'] for arr in files] # extract arrays, load

    if kwargss['includeinf']:
        allfinite = slice(None, None, None)
    else:
        allfinite = np.all(np.array([np.isfinite(inarrs[ind]) for ind in range(len(inarrs))]), axis=0) # check where all arrays are finite 
        print('%i values were excluded for being non-finite.'%(np.prod(allfinite.shape) - np.sum(allfinite)))
    del kwargss['includeinf']
    inarrs = [(inarr[allfinite]).flatten() for inarr in inarrs] # exclude non-finite values and flatten arrays  
    
    if 'maskfilenames' in kwargss.keys():
        maskfilenames = kwargss['maskfilenames']
        maskfilenames = [filen if filen is None else ol.ndir + filen if '/' not in filen else filen for filen in maskfilenames]
        maskfilenames = [filen if filen is None else filen + '.hdf5' if filen[-5:] != '.hdf5' else filen for filen in maskfilenames]
        del kwargss['maskfilenames']
    else:
        maskfilenames = [None]
    
    histsdct = {}
    edgesdct = {}
    covfracs = {}
    for maskfile in maskfilenames:
        if maskfile is not None:
            with h5py.File(maskfile, 'r')as mf:
                mask = np.array(mf['mask']) # same treatment as in input arrays to match selection
                key = maskfile.split('/')[-1][:-5]
                covfracs[key] = float(np.sum(mask)) / float(np.prod(mask.shape))
                mask = (mask[allfinite]).flatten()
                inarrs_sub = [arr[mask] for arr in inarrs]
        else:
            inarrs_sub = inarrs
            key = 'nomask'
            covfracs[key] = 1.
            
        if len(inarrs_sub) * len(inarrs_sub[0]) > (32000**2 * 4):
            sublen =  32000**2 * 4 // len(inarrs_sub) 
            numsub =  (len(inarrs_sub[0]) - 1) // sublen + 1
            if numsub * sublen < len(inarrs[0]):
                raise RuntimeError('A bug in the array looping for histograms would cause only part of the input arrays to be used')
            for subi in range(numsub):
                subsel = slice(sublen * subi, sublen * (subi + 1))
                if subi == 0:
                    outhist, edges = np.histogramdd([inarr[subsel] for inarr in inarrs_sub], **kwargss)
                else:
                    outhistsub, edgessub = np.histogramdd([inarr[subsel] for inarr in inarrs_sub], **kwargss)
                    outhist += outhistsub
                    del outhistsub
                    if not np.all([np.all(edgessub[j] == edges[j]) for j in range(len(edges))]):
                        raise RuntimeError('Edges for subhistogram %i are different from previous ones'%subi)
        else:
            outhist, edges = np.histogramdd(inarrs_sub, **kwargss)
        histsdct[key] = outhist
        edgesdct[key] = edges
        del inarrs_sub
    del inarrs # make sure memory is not used up by arrays we're done with
    
    return histsdct, edgesdct, covfracs        

def makehist_masked_toh5py(*filenames, **kwargs):
    '''
    filenames: different quantities to make into an n-dimensional array
    fills:     different things to fill into the filenames for e.g. different
               slices (must match across filenames)
    maskfiles: list of files containing masks to apply to the files before 
               histogramming, or dict of {fill: such a list} if fills is not 
               None
               (if 'None' is included in the list, no mask is applied)
               or multiple fills, mask files are combined by list index
    masknames: name for each mask -- matched by list index to makefiles at each
               fill
               used in the hdf5 file for naming if fills is not None
    includeinf:include all values in histogram; extend bin edges to ensure that
    '''

    if 'outfilename' not in kwargs.keys():
        raise ValueError("The output file name 'outfilename' must be specified")
    else:
        outfilename = kwargs['outfilename']
        del kwargs['outfilename']
    if '/' not in outfilename:
        outfilename = ol.pdir + outfilename
    if outfilename[-5:] != '.hdf5':
        outfilename = outfilename + '.hdf5'
        
    if 'fills' not in kwargs.keys():
        fills = None
    else:
        fills = kwargs['fills']
        del kwargs['fills'] 
        
    if 'maskfiles' not in kwargs.keys():
        if fills is None:
            maskfiles = [None]
        else:
            maskfiles = {fill: [None] for fill in fills}               
    else:
        maskfiles = kwargs['maskfiles']
        if fills is None:
            maskfiles = [filen if filen is None else ol.ndir + filen if '/' not in filen else filen for filen in maskfiles]
            maskfiles = [filen if filen is None else filen + '.hdf5' if filen[-5:] != '.hdf5' else filen for filen in maskfiles]
        else:
            maskfiles = {fill: [filen if filen is None else ol.ndir + filen if '/' not in filen else filen for filen in maskfiles[fill]] for fill in maskfiles.keys()}
            maskfiles = {fill: [filen if filen is None else filen + '.hdf5' if filen[-5:] != '.hdf5' else filen for filen in maskfiles[fill]] for fill in maskfiles.keys()}
            k0 = maskfiles.keys()[0]
            if not np.all([len(maskfiles[key]) == len(maskfiles[k0]) for key in maskfiles.keys()]):
                raise ValueError('The number of masks for different fills does not match')
        del kwargs['maskfiles']
    
    if 'masknames' not in kwargs.keys():
        if maskfiles == [None]:
            masknames = [None]
        else:    
            masknames = ['masks_%i'%ind for ind in range(len(maskfiles[maskfiles.keys()[0]]))]
    else:
        masknames = kwargs['masknames']
        del kwargs['masknames']
    
    if 'includeinf' not in kwargs.keys():
        kwargs['includeinf'] = True
    if 'bins' not in kwargs.keys():
        raise ValueError("kw argument 'bins' must be specified")
    if kwargs['includeinf']:
        kwargs['bins'] = [np.array([-np.inf] + list(edges)) if edges[0] != -np.inf else\
                          np.array(edges) for edges in kwargs['bins']]
        kwargs['bins'] = [np.array(list(edges) + [np.inf]) if edges[-1] != np.inf else\
                          np.array(edges) for edges in kwargs['bins']]
    # change bins to float32 to avoid errors down the line, when comparing input and output bins
    kwargs['bins'] = [arr.astype(np.float32) for arr in kwargs['bins']]

    # record the more compact mask file data and the filenames
    with h5py.File(outfilename, 'a') as fo:
        fo.create_dataset('input_filenames', data=np.array([filen.split('/')[-1] for filen in filenames]))
        bgrp = fo.create_group('bins')
        [bgrp.create_dataset('axis_%i'%i, data=np.array(kwargs['bins'][i])) for i in range(len(kwargs['bins']))]
        
        if fills is not None:
            fo.create_dataset('fills', data=np.array([str(fill) for fill in fills]))
        if fills is None:
            for mfind in maskfiles:
                filen = maskfiles[mfind]
                if filen is not None:
                    with h5py.File(filen, 'r') as mf:
                        maskstr = filen.split('/')[-1]
                        if maskstr[-5:] == '.hdf5':
                            maskstr = maskstr[:-5]
                        subgrp = fo.create_group('masks/%s'%maskstr)
                        mf.copy(mf['Header'], subgrp, name='Header')
                        try:
                            mfsel = mf['selection']
                            
                            if 'included' in mfsel.keys():
                                ssubgrp = subgrp.create_group('selection/included')
                                keys = mfsel['included'].keys()
                                keys.remove('galaxyid')
                                [ssubgrp.create_dataset(key, data=np.array(mfsel['included/%s'%key])) for key in keys]
                                
                            if 'excluded' in mfsel.keys():
                                ssubgrp = subgrp.create_group('selection/excluded')
                                keys = mfsel['excluded'].keys()
                                keys.remove('galaxyid')
                                [ssubgrp.create_dataset(key, data=np.array(mfsel['excluded/%s'%key])) for key in keys]
                        except KeyError:
                            pass
        else:
            for fill in fills:
                maskfiles_ = maskfiles[fill]
                for filen in maskfiles_:
                    if filen is not None:
                        with h5py.File(filen, 'r') as mf:
                            maskstr = filen.split('/')[-1]
                            subgrp = fo.create_group('masks/%s/%s'%(fill, maskstr))
                            mf.copy(mf['Header'], subgrp, name='Header')
                            try:
                                mfsel = mf['selection']
                                
                                if 'included' in mfsel.keys():
                                    ssubgrp = subgrp.create_group('selection/included')
                                    keys = mfsel['included'].keys()
                                    keys.remove('galaxyid')
                                    [ssubgrp.create_dataset(key, data=np.array(mfsel['included/%s'%key])) for key in keys]
                                    
                                if 'excluded' in mfsel.keys():
                                    ssubgrp = subgrp.create_group('selection/excluded')
                                    keys = mfsel['excluded'].keys()
                                    keys.remove('galaxyid')
                                    [ssubgrp.create_dataset(key, data=np.array(mfsel['excluded/%s'%key])) for key in keys]
                            except KeyError:
                                pass
                                
    # run the actual histograms
    if fills is None:
        hists, edges, covfracs = makehist_1slice_masked(*tuple(filenames), maskfilenames=maskfiles, **kwargs)
        if not np.all(np.array([np.all(np.array([np.all(kwargs['bins'][i] == edges[key][i]) for i in range(len(kwargs['bins'])) ])) for key in edges.keys()])):
            errmes = "Input bins do not match returned edges"
            errmes = errmes + '\n no fills call'
            errmes = errmes + '\ninput edges were:\n' + str(kwargs['bins'])
            errmes = errmes + '\noutput edges were:\n' + str(edges)
            raise RuntimeError(errmes)
    else:
        # map mask keys for each fill to mask index
        hists = None
        covfracs = None
        for fill in fills:
            maskkeys = [maskfile.split('/')[-1][:-5] if maskfile is not None else 'nomask' for maskfile in maskfiles[fill]]
            maskkeytoname = {maskkeys[i]: masknames[i] for i in range(len(masknames))}
            #print(maskkeytoname)
            #masknametokey = {masknames[i]: maskkeys[i] for i in range(len(masknames))}
            subhists, edges, subcovfracs = makehist_1slice_masked(*(filename%fill for filename in filenames), maskfilenames=maskfiles[fill], **kwargs)
            if not np.all(np.array([np.all(np.array([np.all(kwargs['bins'][i] == edges[key][i]) for i in range(len(kwargs['bins'])) ])) for key in edges.keys()])):
                errmes = "Input bins do not match returned edges"
                errmes = errmes + '\n error occurred for fill %s'%(fill)
                errmes = errmes + '\ninput edges were:\n' + str(kwargs['bins'])
                errmes = errmes + '\noutput edges were:\n' + str(edges)
                raise RuntimeError(errmes)
            if hists is None:
                #print('maskkeys: %s'%maskkeys)
                #print('maskkeytoname: %s'%maskkeytoname)
                #print('subhists keys: %s'%subhists.keys())
                hists = {maskkeytoname[key]: subhists[key] for key in subhists.keys()}
                covfracs = {maskkeytoname[key]: subcovfracs[key] for key in subcovfracs.keys()}
            else:
                #if not (set(subhists.keys()) == set(hists.keys())):
                #    raise RuntimeError('hists, subhists keys do not match:\n hists:\t%s\nsubshists:\t%s\n'%(hists.keys(), subhists.keys()))
                #if not (set(subcovfracs.keys()) == set(covfracs.keys())):
                #    raise RuntimeError('covfracs, subcovfracs keys do not match:\n covfracss:\t%s\nsubcovfracs:\t%s\n'%(covfracs.keys(), subcovfracs.keys()))               
                hists = {maskkeytoname[key]: hists[maskkeytoname[key]] + subhists[key] for key in subhists.keys()}
                covfracs = {maskkeytoname[key]: covfracs[maskkeytoname[key]] + subcovfracs[key] for key in subcovfracs.keys()}
        covfracs = {key: covfracs[key] / float(len(fills)) for key in covfracs.keys()}
    
    with h5py.File(outfilename, 'a') as fo:
        for key in hists.keys():
            sgrp = fo.create_group(key)
            sgrp.create_dataset('hist', data=hists[key])
            sgrp.attrs.create('covfrac', covfracs[key])
            if fills is not None:
                for fill in fills:
                    maskfiles_ = maskfiles[fill]
                    maskname = key
                    maskind = np.where([maskname == mn for mn in masknames])[0][0]
                    maskfile = maskfiles_[maskind]
                    if maskfile is not None:
                        sgrp.attrs.create('mask_%s'%fill, maskfile)
                    else:
                        sgrp.attrs.create('mask_%s'%fill, 'None')
    return None
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


def makehistograms_ionbals(kind = 'diff'):
    '''
    kind = ...
           diff: NO7, NO8, o7diff, o8diff
           o7:   NO7, NO8, o7_fromo7, o7_fromo8
           o8:   NO7, NO8, o7_fromo8, o8_fromo8
    Will take a lot of memory.
    '''
    import make_maps_v3_master as m3
    
    # max ib o7: 0.994, max ib o8: 0.516
    edges_ibo8 = np.arange(0.,0.516001, 0.002)
    edges_ibo7 = np.arange(0.,0.996001, 0.004)
    edges_ibdiffo8 = np.arange(-0.516,0.516001, 0.004)
    edges_ibdiffo7 = np.arange(-0.996,0.996001, 0.008)
    edges_iblogdiffo8 = [-100., -80., -60., -40., -30., -20.,] + list(np.arange(-15., -3.95, 1.)) + list(np.arange(-39, 39.5, 1,)/10.)  + list(np.arange(-20, 20.5, 1.)/100.) + list(np.arange(4., 15.05, 1.)) + [20., 30., 40., 60., 80., 100.] 
    edges_iblogdiffo8 = list(set(edges_iblogdiffo8))
    edges_iblogdiffo8.sort()
    edges_iblogdiffo8 = np.array(edges_iblogdiffo8)
    edges_iblogdiffo7 = edges_iblogdiffo8
    edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
    edgesNO8 = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0))
    
    fillsea = [str(float(i)) for i in (np.arange(16)/16.+1/32.)*100.]
    
    rho_NO7_name_16_28 = 'Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    T_NO7_name_16_28 = 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    NO7_name_16_28 = 'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    
    rho_NO8_name_16_28 = 'Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    T_NO8_name_16_28 = 'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    NO8_name_16_28 = 'coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

    outname_diff = 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_ibdiff-o7-o8_from_o7-o8-weighted-rho-T'
    dimension_diff = ['NO7', 'NO8', 'ionbalO7_from_O7-weighted_minus_O8-weighted', 'ionbalO8_from_O7-weighted_minus_O8-weighted']
    edges_diff = [edgesNO7, edgesNO8, edges_ibdiffo7, edges_ibdiffo8]
    
    outname_o7 = 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_ibo7_from_o7-o8-weighted-rho-T'
    dimension_o7 = ['NO7', 'NO8', 'ionbalO7_from_O7-weighted-Temperature-Density', 'ionbalO7_from_O8-weighted-Temperature-Density']
    edges_o7 = [edgesNO7, edgesNO8, edges_ibo7, edges_ibo7]
    
    outname_o8 = 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_ibo8_from_o7-o8-weighted-rho-T'
    dimension_o8 = ['NO7', 'NO8', 'ionbalO8_from_O7-weighted-Temperature-Density', 'ionbalO8_from_O8-weighted-Temperature-Density']
    edges_o8 = [edgesNO7, edgesNO8, edges_ibo8, edges_ibo8]
    
    outname_logdiff = 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_iblogreldiff-o7-o8_from_o7-o8-weighted-rho-T'
    dimension_logdiff = ['NO7', 'NO8', 'ionbalO7_from_O7-weighted_over_O8-weighted', 'ionbalO8_from_O7-weighted_over_O8-weighted']
    edges_logdiff = [edgesNO7, edgesNO8, edges_iblogdiffo7, edges_iblogdiffo8]
    
    
    first = True
    for fill in fillsea:
        # will take a lot of memory, but boundary condition impementation for interpolation won't allow check after ion balance retrieval
        
        # load arrays
        To7 = np.load(ndir + T_NO7_name_16_28%(fill))['arr_0']
        nHo7 = np.load(ndir + rho_NO7_name_16_28%(fill))['arr_0']
        No7 = np.load(ndir + NO7_name_16_28%(fill))['arr_0']
        
        To8 = np.load(ndir + T_NO8_name_16_28%(fill))['arr_0']
        nHo8 = np.load(ndir + rho_NO8_name_16_28%(fill))['arr_0']
        No8 = np.load(ndir + NO8_name_16_28%(fill))['arr_0']
        
        # check where all arrays are finite
        allfinite = np.isfinite(To7) 
        allfinite &= np.isfinite(nHo7)
        allfinite &= np.isfinite(No7)
        allfinite &= np.isfinite(To8)
        allfinite &= np.isfinite(nHo8)
        allfinite &= np.isfinite(No8)
        
        print('For key %s, %i values were excluded for being non-finite.'%(fill,np.prod(allfinite.shape)-np.sum(allfinite)))
        
        # select only finite values
        To7 = (To7[allfinite]).flatten()
        To8 = (To8[allfinite]).flatten()
        nHo7 = (nHo7[allfinite]).flatten()
        nHo8 = (nHo8[allfinite]).flatten()
        No7 = (No7[allfinite]).flatten()
        No8 = (No8[allfinite]).flatten()
        
        # use rho -> nH from primordial hydrogen mass fraction
        nHo7 += np.log10(0.752/(m3.c.atomw_H*m3.c.u))
        nHo8 += np.log10(0.752/(m3.c.atomw_H*m3.c.u))
        
        
        if kind == 'diff':
            # get ion balances
            o7diff = m3.find_ionbal(0.0,'o7', nHo7, To7)
            o8diff = m3.find_ionbal(0.0,'o8', nHo7, To7)
            del To7
            del nHo7
            o7_by_o8 = m3.find_ionbal(0.0,'o7', nHo8, To8)
            o8_by_o8 = m3.find_ionbal(0.0,'o8', nHo8, To8)
            del To8
            del nHo8
            
            o7diff -= o7_by_o8
            del o7_by_o8
            o8diff -= o8_by_o8
            del o8_by_o8
            
            if first:
                hist, edges = np.histogramdd([No7, No8, o7diff, o8diff], bins=edges_diff)
                first = False
            else:
                histsub, edgessub = np.histogramdd([No7, No8, o7diff, o8diff], bins=edges_diff)
                if not np.all([np.all(np.array(edgessub[ind])==np.array(edges[ind]))] for ind in range(len(edgessub))): # if edges do not match, the histograms should not be added; need the per index check because the array objects are different even though they contain the same values
                    print('Error: edge mismatch when processing array %s'%fill)
                    return edgessub
                hist += histsub
                del histsub
            del o7diff
            del o8diff
            del No7
            del No8
            
        
        elif kind == 'o8':
            # get ion balances
            o8_by_o7 = m3.find_ionbal(0.0,'o8', nHo7, To7)
            del To7
            del nHo7
            o8_by_o8 = m3.find_ionbal(0.0,'o8', nHo8, To8)
            del To8
            del nHo8
            
            if first:
                hist, edges = np.histogramdd([No7, No8, o8_by_o7, o8_by_o8], bins=edges_o8)
                first = False
            else:
                histsub, edgessub = np.histogramdd([No7, No8, o8_by_o7, o8_by_o8], bins=edges_o8)
                if not np.all([np.all(np.array(edgessub[ind])==np.array(edges[ind]))] for ind in range(len(edgessub))): # if edges do not match, the histograms should not be added; need the per index check because the array objects are different even though they contain the same values
                    print('Error: edge mismatch when processing array %s'%fill)
                    return edgessub
                hist += histsub
                del histsub
            del o8_by_o7
            del o8_by_o8
            del No7
            del No8
            
            
        elif kind == 'o7':
            # get ion balances
            o7_by_o7 = m3.find_ionbal(0.0,'o7', nHo7, To7)
            del To7
            del nHo7
            o7_by_o8 = m3.find_ionbal(0.0,'o7', nHo8, To8)
            del To8
            del nHo8
            
            if first:
                hist, edges = np.histogramdd([No7, No8, o7_by_o7, o7_by_o8], bins=edges_o7)
                first = False
            else:
                histsub, edgessub = np.histogramdd([No7, No8, o7_by_o7, o7_by_o8], bins=edges_o7)
                if not np.all([np.all(np.array(edgessub[ind])==np.array(edges[ind]))] for ind in range(len(edgessub))): # if edges do not match, the histograms should not be added; need the per index check because the array objects are different even though they contain the same values
                    print('Error: edge mismatch when processing array %s'%fill)
                    return edgessub
                hist += histsub
                del histsub                
            del o7_by_o7
            del o7_by_o8
            del No7
            del No8
        
        elif kind == 'logdiff':
            # get ion balances
            o7diff = m3.find_ionbal(0.0,'o7', nHo7, To7)
            o8diff = m3.find_ionbal(0.0,'o8', nHo7, To7)
            del To7
            del nHo7
            o7_by_o8 = m3.find_ionbal(0.0,'o7', nHo8, To8)
            o8_by_o8 = m3.find_ionbal(0.0,'o8', nHo8, To8)
            del To8
            del nHo8
            
            o7diff /= o7_by_o8
            del o7_by_o8
            o8diff /= o8_by_o8
            del o8_by_o8
            
            o8diff = np.log10(o8diff)
            o7diff = np.log10(o7diff)
            
            if first:
                hist, edges = np.histogramdd([No7, No8, o7diff, o8diff], bins=edges_logdiff)
                first = False
            else:
                histsub, edgessub = np.histogramdd([No7, No8, o7diff, o8diff], bins=edges_logdiff)
                if not np.all([np.all(np.array(edgessub[ind])==np.array(edges[ind]))] for ind in range(len(edgessub))): # if edges do not match, the histograms should not be added; need the per index check because the array objects are different even though they contain the same values
                    print('Error: edge mismatch when processing array %s'%fill)
                    return edgessub
                hist += histsub
                del histsub
            del o7diff
            del o8diff
            del No7
            del No8
        else:
            print('argument "kind" did not match options')
        
    if kind == 'diff':
        np.savez(pdir + outname_diff, bins=hist, edges=edges, dimension=dimension_diff)
    elif kind == 'o7':
        np.savez(pdir + outname_o7, bins=hist, edges=edges, dimension=dimension_o7)
    elif kind == 'o8':
        np.savez(pdir + outname_o8, bins=hist, edges=edges, dimension=dimension_o8)
    elif kind == 'logdiff':
        np.savez(pdir + outname_logdiff, bins=hist, edges=edges, dimension=dimension_logdiff)
        
    return None


def makehistograms_Zdiff(kind='h1', simset='L0025N0376_19'):
    '''
    kind = ...
           SFR:  SFR,  Z_SFR,  Zmass / ZSFR, Zh1 / ZSFR
           mass: mass, Z_mass, Zh1 / Zmass, ZSFR / Zmass
           h1  : h1,   Z_h1  , Zmass / Zh1, ZSFR / Zh1
    Will take a lot of memory.
    '''
    
    if simset == 'L0025N0376_19':
        fills = [str(i) for i in np.arange(4)/4. * 25. + 25./8.]
        name_mass   = 'Mass_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_mass = 'Metallicity_T4EOS_Mass_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        name_h1     = 'coldens_h1ssh_L0025N0376_19_test3.31_PtAb_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_h1   = 'Metallicity_T4EOS_coldens_h1ssh_PtAb_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        name_sfr    = 'StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
        name_Z_sfr  = 'Metallicity_T4EOS_StarFormationRate_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        
        edges_mass   = np.arange(-5.8, 0.15, 0.1)
        edges_sfr    = np.array([-42., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
        edges_h1     = np.arange(11.1, 22.35, 0.1)
        edges_Z      = np.array([-40., -10., -8., -6.] + list(np.arange(-5., -0.95, 0.1)))
        edges_Zdiff  = np.array([-40., -10., -8., -6., -5., -4., -3.] + list(np.arange(-2., -0.05, 0.1)))
        edges_Zdiff  = np.array(list(edges_Zdiff) + [0.] + list(-1*edges_Zdiff[::-1]))
        inclinf = False
        
        outnamebase = 'L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection'
        sZname = 'Z'
        smassname = 'Mass'
        ssfrname  = 'StarFormationRate'
        sh1name   = 'coldens_h1ssh'
        dZname = 'Metallicity'
        dmassname = 'Mass'
        dsfrname = 'SFR'
        dh1name = 'NHI'
    
    elif simset == 'L0025N0376_19_SmZ':
        fills = [str(i) for i in np.arange(4)/4. * 25. + 25./8.]
        name_mass   = 'Mass_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_mass = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        name_h1     = 'coldens_h1ssh_L0025N0376_19_test3.31_PtAb_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_h1   = 'SmoothedMetallicity_T4EOS_coldens_h1ssh_PtAb_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        name_sfr    = 'StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
        name_Z_sfr  = 'SmoothedMetallicity_T4EOS_StarFormationRate_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        
        edges_mass   = np.arange(-5.8, 0.15, 0.1)
        edges_sfr    = np.array([-42., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
        edges_h1     = np.arange(11.1, 22.35, 0.1)
        edges_Z      = np.array([-40., -10., -8., -6.] + list(np.arange(-5., -0.95, 0.1)))
        edges_Zdiff  = np.array([-40., -10., -8., -6., -5., -4., -3.] + list(np.arange(-2., -0.05, 0.1)))
        edges_Zdiff  = np.array(list(edges_Zdiff) + [0.] + list(-1*edges_Zdiff[::-1]))
        inclinf = False
        
        outnamebase = 'L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection'
        sZname = 'SmZ'
        smassname = 'Mass'
        ssfrname  = 'StarFormationRate'
        sh1name   = 'coldens_h1ssh'
        dZname = 'SmoothedMetallicity'
        dmassname = 'Mass'
        dsfrname = 'SFR'
        dh1name = 'NHI'
    
    elif simset == 'L0025N0376_19_SmZ_hn':
        fills = [str(i) for i in np.arange(4)/4. * 25. + 25./8.]
        name_mass   = 'Mass_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_mass = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        name_h1     = 'coldens_hneutralssh_L0025N0376_19_test3.31_PtAb_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_h1   = 'SmoothedMetallicity_T4EOS_coldens_hneutralssh_PtAb_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        name_sfr    = 'StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
        name_Z_sfr  = 'SmoothedMetallicity_T4EOS_StarFormationRate_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
        
        edges_mass   = np.arange(-5.8, 0.15, 0.1)
        edges_sfr    = np.array([-42., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
        edges_h1     = np.arange(11.1, 22.35, 0.1)
        edges_Z      = np.array([-40., -10., -8., -6.] + list(np.arange(-5., -0.95, 0.1)))
        edges_Zdiff  = np.array([-40., -10., -8., -6., -5., -4., -3.] + list(np.arange(-2., -0.05, 0.1)))
        edges_Zdiff  = np.array(list(edges_Zdiff) + [0.] + list(-1*edges_Zdiff[::-1]))
        inclinf = False
        
        outnamebase = 'L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection'
        sZname = 'SmZ'
        smassname = 'Mass'
        ssfrname  = 'StarFormationRate'
        sh1name   = 'coldens_hneutralssh'
        dZname = 'SmoothedMetallicity'
        dmassname = 'Mass'
        dsfrname = 'SFR'
        dh1name = 'NHneutral'
    
    elif simset == 'L0025N0752Recal_19_SmZ':
        fills = [str(i) for i in np.arange(8)/8. * 25. + 25./16.]
        
        name_mass   = 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_mass = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
        name_h1     = 'coldens_h1ssh_L0025N0752RECALIBRATED_19_test3.31_PtAb_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_h1   = 'SmoothedMetallicity_T4EOS_coldens_h1ssh_PtAb_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
        name_sfr    = 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
        name_Z_sfr  = 'SmoothedMetallicity_T4EOS_StarFormationRate_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
        
        edges_mass   = np.array([-np.inf] + list(np.arange(-6.3, 0.15, 0.1)))
        edges_sfr    = np.array([-np.inf, -46., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
        edges_h1     = np.array([-np.inf] + list(np.arange(10.3, 23.35, 0.1)))
        edges_Z      = np.array([-np.inf, -52., -40., -10., -8., -6.] + list(np.arange(-5., -0.75, 0.1)))
        edges_Zdiff  = np.array([-52., -40., -10., -8., -6., -5., -4., -3.] + list(np.arange(-2., -0.05, 0.1)))
        edges_Zdiff  = np.array([-np.inf] + list(edges_Zdiff) + [0.] + list(-1*edges_Zdiff[::-1]) + [np.inf])
        inclinf = True
        
        outnamebase = 'L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection'
        sZname = 'SmZ'
        smassname = 'Mass'
        ssfrname  = 'StarFormationRate'
        sh1name   = 'coldens_h1ssh'
        dZname = 'SmoothedMetallicity'
        dmassname = 'Mass'
        dsfrname = 'SFR'
        dh1name = 'NHI'
    
    elif simset == 'L0025N0752Recal_19_SmZ_hn':
        fills = [str(i) for i in np.arange(8)/8. * 25. + 25./16.]
        
        name_mass   = 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_mass = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
        name_h1     = 'coldens_hneutralssh_L0025N0752RECALIBRATED_19_test3.31_PtAb_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
        name_Z_h1   = 'SmoothedMetallicity_T4EOS_coldens_hneutralssh_PtAb_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
        name_sfr    = 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
        name_Z_sfr  = 'SmoothedMetallicity_T4EOS_StarFormationRate_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
        
        edges_mass   = np.array([-np.inf] + list(np.arange(-6.3, 0.15, 0.1)))
        edges_sfr    = np.array([-np.inf, -46., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
        edges_h1     = np.array([-np.inf] + list(np.arange(10.3, 23.35, 0.1)))
        edges_Z      = np.array([-np.inf, -52., -40., -10., -8., -6.] + list(np.arange(-5., -0.75, 0.1)))
        edges_Zdiff  = np.array([-52., -40., -10., -8., -6., -5., -4., -3.] + list(np.arange(-2., -0.05, 0.1)))
        edges_Zdiff  = np.array([-np.inf] + list(edges_Zdiff) + [0.] + list(-1*edges_Zdiff[::-1]) + [np.inf])
        inclinf = True
        
        outnamebase = 'L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection'
        sZname = 'SmZ'
        smassname = 'Mass'
        ssfrname  = 'StarFormationRate'
        sh1name   = 'coldens_hneutralssh'
        dZname = 'SmoothedMetallicity'
        dmassname = 'Mass'
        dsfrname = 'SFR'
        dh1name = 'NHneutral'
        
    if kind == 'SFR':
        name_baseq = name_sfr
        name_zbase = name_Z_sfr
        name_d1    = name_Z_mass
        name_d2    = name_Z_h1
        
        edges_baseq = edges_sfr
        
        dimension = [ssfrname, '%s_w_%s'%(dZname, dsfrname), '%s_w_%s_minus_%s_w_%s'%(dZname, dmassname, dZname, dsfrname), '%s_w_%s_minus_%s_w_%s'%(dZname, dh1name, dZname, dsfrname)]      
        outname = '%s_and_weighted_%s_diff_with_%s_%s_'%(ssfrname, sZname, smassname, sh1name)  + outnamebase
        
    
    elif kind == 'h1':
        name_baseq = name_h1
        name_zbase = name_Z_h1
        name_d1    = name_Z_mass
        name_d2    = name_Z_sfr
        
        edges_baseq = edges_h1
        
        dimension = [dh1name, '%s_w_%s'%(dZname, dh1name), '%s_w_%s_minus_%s_w_%s'%(dZname, dmassname, dZname, dh1name), '%s_w_%s_minus_%s_w_%s'%(dZname, dsfrname, dZname, dh1name)]
        outname = '%s_and_weighted_%s_diff_with_%s_%s_'%(sh1name, sZname, smassname, ssfrname)  + outnamebase
        
    elif kind == 'mass':
        name_baseq = name_mass
        name_zbase = name_Z_mass
        name_d1    = name_Z_h1
        name_d2    = name_Z_sfr
        
        edges_baseq = edges_mass
        
        dimension = [dmassname, '%s_w_%s'%(dZname, dmassname), '%s_w_%s_minus_%s_w_%s'%(dZname, dh1name, dZname, dmassname), '%s_w_%s_minus_%s_w_%s'%(dZname, dsfrname, dZname, dmassname)]
        outname = '%s_and_weighted_%s_diff_with_%s_%s_'%(smassname, sZname, sh1name, ssfrname)  + outnamebase
    # 'StarFormationRate', 'Metallicity_w_Mass', 'Metallicity_w_NH1', 'Metallicity_w_SFR'
        
    first = True
    edges = [edges_baseq, edges_Z, edges_Zdiff, edges_Zdiff]
    
    for fill in fills:
        # will take a lot of memory, but boundary condition impementation for interpolation won't allow check after ion balance retrieval
        
        # load arrays
        try:
            baseq = np.load(ol.ndir + name_baseq%(fill))['arr_0']
            Zbase = np.load(ol.ndir + name_zbase%(fill))['arr_0']
            d1 = np.load(ol.ndir + name_d1%(fill))['arr_0']
            d2 = np.load(ol.ndir + name_d2%(fill))['arr_0']
        except IOError:
            baseq = np.load(ol.ndir_old + name_baseq%(fill))['arr_0']
            Zbase = np.load(ol.ndir_old + name_zbase%(fill))['arr_0']
            d1 = np.load(ol.ndir_old + name_d1%(fill))['arr_0']
            d2 = np.load(ol.ndir_old + name_d2%(fill))['arr_0']
        
        
        # check where all arrays are finite
        if inclinf:
            allfinite = (slice(None, None, None),)*2
        else:
            allfinite = np.isfinite(baseq) 
            allfinite &= np.isfinite(Zbase)
            allfinite &= np.isfinite(d1)
            allfinite &= np.isfinite(d2)
            
            print('For key %s, %i values were excluded for being non-finite.'%(fill, np.prod(allfinite.shape)-np.sum(allfinite)))
        
        # select only finite values
        baseq = (baseq[allfinite]).flatten()
        Zbase = (Zbase[allfinite]).flatten()
        d1 = (d1[allfinite]).flatten()
        d2 = (d2[allfinite]).flatten()
        
        # take the differences
        d1 -= Zbase
        d2 -= Zbase
        
        # where both have -inf, difference is NaN
        if inclinf:
            d1[np.isnan(d1)] = 0. 
            d2[np.isnan(d2)] = 0. 
        
        if first:
            hist, edges = np.histogramdd([baseq, Zbase, d1, d2], bins=edges)
            first = False
        else:
            histsub, edgessub = np.histogramdd([baseq, Zbase, d1, d2], bins=edges)
            if not np.all([np.all(np.array(edgessub[ind])==np.array(edges[ind]))] for ind in range(len(edgessub))): # if edges do not match, the histograms should not be added; need the per index check because the array objects are different even though they contain the same values
                raise RuntimeError('Error: edge mismatch when processing array %s'%fill)
                return edgessub
            hist += histsub
            del histsub
        del baseq
        del Zbase
        del d1
        del d2
            
    np.savez(pdir + outname, bins=hist, edges=edges, dimension=dimension)
    return None
