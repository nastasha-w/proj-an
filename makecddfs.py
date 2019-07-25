import numpy as np
import h5py  # the exception should be a temporary problem; no need to load stuff for that in advance

import eagle_constants_and_units as c
import cosmo_utils as csu
import make_maps_opts_locs as ol
import projection_classes as pc

# for on cosma
pdir = '/net/luttero/data2/proc/'
ndir = '/net/luttero/data2/temp/'
mdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/img/'
pdir = ol.pdir
ndir = ol.ndir
mdir = ol.mdir

def get_simdata_from_outputname(outputname):
    '''
    get simnum, snapnum, var, simulation
    '''

    outputname = outputname.split('/')[-1] # just the file, no need to include all the directories
    parts = outputname.split('_')
    simnumparts = [string for string in parts \
                   if (len(string) >= 10 and string[0] == 'L' and string[5] == 'N')\
                   or string[:4] == 'BA-L'\
                   or string[:4] == 'EA-L' \
                   or string[:10] == 'EA-ioneq-L'] # EAGLE L####N####VAR or BAHAMAS BA-L*N*VAR simunums  
    outdict = {}
    
    if len(simnumparts) != 1:
        print('No clear simnum found for %s; options are %s'%(outputname,simnumparts))
    else: 
        simnumvar = simnumparts[0]
         
        # some var names unfortunately include underscores, so just get everything up to _#?_test
        # start at _test, assume form <simname><L#N#><var>_<snapnum>_test...
        simnumvarend = outputname.find(simnumvar) + len(simnumvar)
        testind = outputname.find('test')
        # entry before test is one or two numbers for the snapshot: walk backwards until that is past
        snapnum = ''
        i = testind
        if outputname[i] == 't':  # 't' from test -> step back          
            i -= 1
        if outputname[i] == '_':  # '_' from _test -> step back         
            i -= 1
        while(True): # numbers: snapshot number 
           snapnum = outputname[i] + snapnum # walking backwards, so add new digit to the front
           i -= 1 
           if not outputname[i].isdigit():
                break                  
        if outputname[i] == '_':
            i -= 1  # i is now the index of the last character belonging ot the simnum-variation entry       
        simnumvar += outputname[simnumvarend:i+1]        
        outdict['snapnum'] = int(snapnum)
        

        if simnumvar[:3] == 'BA-':
            outdict['simulation'] = 'bahamas'
            simnumvar = simnumvar[3:]
        elif simnumvar[:9] == 'EA-ioneq-': #Ben's non-equilibrium zooms
            outdict['simulation'] = 'eagle-ioneq'
            simnumvar = simnumvar[9:]
        elif simnumvar[:3] == 'EA-':
            outdict['simulation'] = 'eagle'
            simnumvar = simnumvar[3:]
        else: # default / no further specification means it's an EAGLE box 
            outdict['simulation'] = 'eagle'
        
        if outdict['simulation'] == 'eagle':
            var = simnumvar[10:] #EAGLE simnums always have length 10
            if var == '':
                var = 'REFERENCE'
            outdict['var'] = var
            outdict['simnum'] = simnumvar[:10]
        
        if outdict['simulation'] == 'eagle-ioneq':
            var = simnumvar[10:] #EAGLE simnums always have length 10
            if var == '':
                var = 'REFERENCE'
            outdict['var'] = var
            outdict['simnum'] = simnumvar[:10]
            

        elif outdict['simulation'] == 'bahamas':
            # find first non-number after L<numbers>N<numbers>
            simnum = ''
            var = ''
            
            i=0
            if simnumvar[i] == 'L':
                i += 1
                simnum += 'L'
            while(True): # numbers
                simnum += simnumvar[i]
                i += 1 
                if not simnumvar[i].isdigit():
                    break                        
            if simnumvar[i] == 'N':
                i += 1
                simnum += 'N'
            while(True): # numbers
                simnum += simnumvar[i]
                i += 1 
                if i == len(simnumvar): # for var None, numbers end where the string ends
                    break
                if not simnumvar[i].isdigit():
                    break
            
            var = simnumvar[i:] #the stuff after 'L<numbers>N<numbers>'
            if var == '':
                var = 'REFERENCE'

            outdict['simnum'] = simnum
            outdict['var'] = var
    
    pixpart = [part for part in parts if 'pix' in part][0]
    outdict['numpix'] = int(pixpart[:-3])
    
    axisparts = [part for part in parts if 'projection' in part]
    if len(axisparts) == 0: # old default not included in filename
        axis = 'z'
    else:
        axis = axisparts[0][0]
    outdict['axis'] = axis
    

    return outdict



def getcosmopars(simnum,snapnum,var,file_type = 'snap',simulation = 'eagle'):
    # uses Simfile class from make_maps
    simfile = pc.Simfile(simnum,snapnum,var,file_type = 'snap',simulation = simulation)
    # older (v3.1) simfile versions do not retieve omegam and omegalambda from
    # output files -> use some of the newer code directly for older versions
    try:
        cosmopars = {'h':           simfile.h,\
                     'z':           simfile.z,\
                     'a':           simfile.a,\
                     'omegam':      simfile.omegam,\
                     'omegalambda': simfile.omegalambda,\
                     'omegab':      simfile.omegab,\
                     'boxsize':     simfile.boxsize}
    except AttributeError: # simfile has no omegam or omegalambda attributes
        cosmopars = {'h':           simfile.h,\
                     'z':           simfile.z,\
                     'a':           simfile.a,\
                     'boxsize':     simfile.boxsize}
        if simulation == 'eagle': 
            filenamebase = simfile.readfile.fname
            try:
                hdf5file = h5py.File( filenamebase+"0.hdf5", 'r' )
            except:
                hdf5file = h5py.File( filenamebase+"hdf5", 'r' )
            omegam      = hdf5file['Header'].attrs['Omega0']
            omegalambda = hdf5file['Header'].attrs['OmegaLambda']
            omegab      = hdf5file['Header'].attrs['OmegaBaryon']
            hdf5file.close()
            cosmopars.update({'omegam':      omegam,\
                              'omegalambda': omegalambda,\
                              'omegab':      omegab})
        else:
            print('How did a v < 3.1 Simfile work on something other than eagle??')
            return None
    return cosmopars


def getdX(redshift,L_z,cosmopars=None):
    # assuming L_z is smaller than the distance over which H varies significantly; 
    # assumed in single-snapshot projection anyway 
    if cosmopars is not None:
        redshift = cosmopars['z']
        hpar = cosmopars['h']
    else:     
        hpar = c.hubbleparam
    dz = csu.Hubble(redshift,cosmopars=cosmopars)/c.c * L_z * c.cm_per_mpc
    dX = dz * (1+redshift)**2*c.hubble*hpar/csu.Hubble(redshift,cosmopars=cosmopars) 
    return dX


def getdz(redshift,L_z,cosmopars=None):
    # assuming L_z is smaller than the distance over which H varies significantly; 
    # assumed in single-snapshot projection anyway 
    if cosmopars is not None:
        redshift = cosmopars['z']
    dz = csu.Hubble(redshift,cosmopars=cosmopars)/c.c * L_z * c.cm_per_mpc
    return dz


def dXcddf_to_dzcddf_factor(redshift,cosmopars=None):
    if cosmopars is not None:
        redshift = cosmopars['z']
    return (1+redshift)**2*c.hubble*c.hubbleparam/csu.Hubble(redshift,cosmopars=cosmopars)


def cddf_over_pixcount(redshift,L_z,numpix,edges,cosmopars=None):
    '''
    Here, edges is the output edges, so the left-hand sides only
    L_z is the total length in cMpc (all slices added up)
    '''
    if cosmopars is not None:
        redshift = cosmopars['z']
    dXtot = getdX(redshift,L_z,cosmopars=cosmopars)*numpix
    binsize_logN = (edges[-1] - edges[0])/float(len(edges)-1)
    logcoldens_midbins = edges + 0.5*binsize_logN

    dN = np.log(10.)*10**(logcoldens_midbins)*binsize_logN
    return dXtot*dN


def geq_hist_from_binsedges(binsedges):
    
    # bins = d#sightlines/dNdX or d#sightlines/dNdz
    # edges = left edges of evenly spaced log N bins 
    if isinstance(binsedges,list) or isinstance(binsedges,np.ndarray):
        bins = binsedges[0]
        edges = binsedges[1] 
    else: #assume npz file or dict
        bins = binsedges['bins']
        if 'logedges' in binsedges.keys():
            edges = binsedges['logedges']
        else:
            edges = binsedges['edges']
    # get bin sizes divided by in cddf calculation (should match dN in histogram calculations)
    binsize_logN = (edges[-1] - edges[0])/float(len(edges)-1)
    logcoldens_midbins = edges + 0.5*binsize_logN
    dN = np.log(10.)*10**(logcoldens_midbins)*binsize_logN

    dnumdXz = bins*dN 
    dnumgeqdXz = np.array([np.sum(dnumdXz[i:]) for i in range(len(dnumdXz))]) # count all bin contributions, starting with the bin itself
    return dnumgeqdXz, edges


def make_coldens_hist(coldens, L_z, redshift, bins = 50, colmin = None,colmax = None,save = None, cosmopars=None,verbose=False):
    ''' 
    returns d n_sightnlines / dN dX (not /d log N)
    L_z is sightline length in cMpc through the slice
    should also give d #sightlines/ dEW dX if EWs are input in place of coldens
    coldens is allowed to be a dictionary with a single entry
    '''
    if verbose:
        print('make_coldens_hist called')
    if cosmopars is not None:
        redshift = cosmopars['z']
    if isinstance(coldens,dict): # allow non-copy array passing to function
        coldens = coldens[coldens.keys()[0]] 
        if verbose: 
            print('array retrieved from dict')
        
    numpix = np.prod(coldens.shape)
     # we don't need the shape information anymore, and zero entries won't contribute anyway
    if verbose:
        print('selecting finite values...')
    isf = np.isfinite(coldens)
    if verbose:
        print('applying selection...')
    pcoldens = coldens[isf]
    del isf
    if verbose:
        print('flattening array...')
    if len(pcoldens.shape) != 1:
        pcoldens = np.ndarray.flatten(pcoldens)
    if verbose:
        print('done')             

    if colmin == None:
        colmin = np.min(pcoldens)

    if colmax == None:
        colmax = np.max(pcoldens)
         
    dX = getdX(redshift,L_z,cosmopars=cosmopars)

    # histogram; input values are already log column densities 
    if verbose:
        print('Making histogram...')
    hist, bin_edges = np.histogram(pcoldens,bins = bins, \
        range = (colmin,colmax)) 
    if verbose:
        print('done')    
    logcoldens_midbins = 0.5*(bin_edges[1:]+bin_edges[:-1])
    hist = hist/(np.log(10.)*10**(logcoldens_midbins)*(bin_edges[1:]-bin_edges[:-1]))/(numpix*dX) 
    
    if verbose:
        print('make_coldens_hist returning')
    return hist, bin_edges[:-1]


def make_coldens_hist_dz(coldens, L_z, redshift, bins = 50, colmin = None,colmax = None,cosmopars=None):
    ''' 
    returns d n_sightnlines / dN dX (not /d log N)
    L_z is sightline length in cMpc through the slice
    should also give d #sightlines/ dEW dX if EWs are input in place of coldens
    coldens is allowed to be a dictionary with a single entry
    '''
    if cosmopars is not None:
        redshift = cosmopars['z']
    if isinstance(coldens,dict): # allow non-copy array passing to function
        coldens = coldens[coldens.keys()[0]] 

    numpix = np.prod(coldens.shape)
    pcoldens = np.ndarray.flatten(coldens[np.isfinite(coldens)]) # we don't need the shape information anymore, and zero entries won't contribute anyway
             
    if colmin == None:
        colmin = np.min(pcoldens)

    if colmax == None:
        colmax = np.max(pcoldens)
         
    dz = getdz(redshift,L_z, cosmopars=cosmopars)

    # histogram; input values are already log column densities 
    hist, bin_edges = np.histogram(pcoldens,bins = bins, \
        range = (colmin,colmax))     
    logcoldens_midbins = 0.5*(bin_edges[1:]+bin_edges[:-1])
    hist = hist/(np.log(10.)*10**(logcoldens_midbins)*(bin_edges[1:]-bin_edges[:-1]))/(numpix*dz) 
    
    return hist, bin_edges[:-1]


def imreduce(img, factor, log=True, method = 'average', wimg=None, wlog='auto'):
    """
    img: 2D image array
    factor: factor by which to reduce the number of array elements along each axis
    log: whether or not the array contains log data values
    """
    if method not in ['average','waverage','sum']:
        print("Method must be one of 'average','waverage','sum'")
        return None
    if log:
        inimg = 10**img
    else:
        inimg = img
    inshape = np.array(img.shape)
    
    if np.sum(inshape%factor) != 0:
        print('Output grid must have a integer number of cells: cannot reduce image pixels by a factor %i'%factor)
        return None
        
    if method == 'average' or method == 'sum':
        inimg = np.array(np.split(inimg,inshape[0]/factor,axis=0))
        inimg = np.array(np.split(inimg,inshape[1]/factor,axis=-1))
        inimg = np.sum(inimg,axis=-1)
        inimg = np.sum(inimg,axis=-1)
        if method == 'average':
            inimg = inimg/np.float(factor**2)        
        #outimg = np.average(inimg[])

    if method == 'waverage':
        if wimg.shape != img.shape:
            print('Weight array must have the same shape as image array. Was %s, %s'%(wimg.shape, img.shape))
            return None
        if wimg == None:
            print('A weighting array wimg must be provided for the weighted average method.\n')
            return None
        if wlog == 'auto':
            wlog = log
        if wlog:
            inwimg = 10**wimg
        else:
            inwimg = wimg
        
        inimg *= inwimg
        inimg = np.array(np.split(inimg,inshape[0] // factor,axis=0))
        inimg = np.array(np.split(inimg,inshape[1] // factor,axis=-1))
        inimg = np.sum(inimg,axis=-1)
        inimg = np.sum(inimg,axis=-1)
        inwimg = np.array(np.split(inwimg,inshape[0] // factor,axis=0))
        inwimg = np.array(np.split(inwimg,inshape[1] // factor,axis=-1))
        inwimg = np.sum(inwimg,axis=-1)
        inwimg = np.sum(inwimg,axis=-1)
        inimg /= inwimg
        
    if log:
        inimg = np.log10(inimg)
    
    return inimg.T



def make_coldens_hist_slices(namebase,fills, L_z, redshift, numbins = 50, colmin = None,colmax = None, add=1 ,offset=0, red=1,save=None,sel = (slice(None,None,None),slice(None,None,None)),dz=False, cosmopars = None):
    '''
     returns d n_sightnlines / dN dX (not /d log N)
     combines all input slices if add =1
     if add =/= 1: adds <add> slices (must be integer factor of len(fills)), 
       using <fills> in order (mod len(fills)) starting at index <offset>
     add option used for slice thickness convergence tests
     red: factor by which to reduce pixel size (factor in imreduce) -- for pixel size convergence tests
     save: file name to save result to (in proc directory); None -> not saved
     !!! L_z should be for the individual slices (add=1) or slabs: L_z = add * boxlength/(number of slices), not the whole box !!!
    '''   

    if cosmopars is not None:
        redshift = cosmopars['z']
 
    # determine overall maximum and minimum, so that the bins all match up
    if add ==1: 
        if colmin == None or colmax == None:  
            fmin = False
            fmax = False
            if colmin == None:
                colmin = 1e60
                fmin = True
            if colmax == None:
                colmax = -1e60
                fmax = True  
        
            for fill in fills: 
                if red ==1: 
                    cols = np.load(namebase%fill)['arr_0'][sel]
                else:
                    cols = imreduce(np.load(namebase%fill)['arr_0'][sel],red)
            
                if fmin:
                    cmin = np.min(cols[np.isfinite(cols)])
                    colmin = min(cmin,colmin)
                if fmax:
                    cmax = np.max(cols[np.isfinite(cols)]) 
                    colmax = max(cmax,colmax)
        print('Minimum log10 column density: %s, maximum log10 column density: %s\n'%(str(colmin), str(colmax)))

        bins = np.zeros(numbins)
        edges = np.zeros(numbins)
        for fill in fills:
            if red ==1:
                if dz:
                    nbins, nedges = make_coldens_hist_dz({'arr': np.load(namebase%fill)['arr_0'][sel]}, L_z, redshift, bins = numbins, colmin = colmin,colmax = colmax, cosmopars=cosmopars)
                else: # dX
                    nbins, nedges = make_coldens_hist({'arr': np.load(namebase%fill)['arr_0'][sel]}, L_z, redshift, bins = numbins, colmin = colmin,colmax = colmax, cosmopars=cosmopars)
            else: #reduce image resolution
                if dz:
                    nbins, nedges = make_coldens_hist_dz({'arr': imreduce(np.load(namebase%fill)['arr_0'][sel],red)}, L_z, redshift, bins = numbins, colmin = colmin,colmax = colmax, cosmopars=cosmopars)
                else: #dX
                    nbins, nedges = make_coldens_hist({'arr': imreduce(np.load(namebase%fill)['arr_0'][sel],red)}, L_z, redshift, bins = numbins, colmin = colmin,colmax = colmax, cosmopars=cosmopars)
            bins += nbins/np.float(len(fills))
            print('All except first should be equal to %s: %s'%(str(numbins),str(sum(edges == nedges))))        
            edges = nedges
    # divide bins by number of slices considered at the end: total numer of pixels considered  = number of pixels per slice x number of slices  



    else: # add =/= 1
        fills = np.array(fills)
        fills = np.roll(fills,-1*offset) # first slice starts at slice #<offset>
        if len(fills) % add != 0:
            print('Add option must divide the number of slices.\n')
            return None
        nslabs = len(fills) // add

        if colmin == None or colmax == None:  
            fmin = False
            fmax = False
            if colmin == None:
                colmin = 1e60
                fmin = True
            if colmax == None:
                colmax = -1e60
                fmax = True  

            for slind in range(nslabs):
                
                fill = fills[slind*add]
                if red ==1: 
                    cols = 10**np.load(namebase%fill)['arr_0'][sel]
                else:
                    cols = 10**imreduce(np.load(namebase%fill)['arr_0'][sel],red)
                
                for i in 1+np.array(range(add-1)):
                    fill = fills[slind*add + i]
                    if red ==1: 
                        cols += 10**np.load(namebase%fill)['arr_0'][sel]
                    else:
                        cols += 10**imreduce(np.load(namebase%fill)['arr_0'][sel],red)
                cols = np.log10(cols)

                if fmin:
                    cmin = np.min(cols[np.isfinite(cols)])
                    colmin = min(cmin,colmin)
                if fmax:
                    cmax = np.max(cols[np.isfinite(cols)]) 
                    colmax = max(cmax,colmax)
        print('Minimum log10 column density: %s, maximum log10 column density: %s\n'%(str(colmin), str(colmax)))

        bins = np.zeros(numbins)
        edges = np.zeros(numbins)
        for slind in range(nslabs):
                
            fill = fills[slind*add]
            if red ==1: 
                cols = 10**np.load(namebase%fill)['arr_0'][sel]
            else:
                cols = 10**imreduce(np.load(namebase%fill)['arr_0'][sel],red)
            
            for i in 1+np.array(range(add-1)):
                fill = fills[slind*add + i]
                if red ==1: 
                    cols += 10**np.load(namebase%fill)['arr_0'][sel]
                else:
                    cols += 10**imreduce(np.load(namebase%fill)['arr_0'][sel],red)
            cols = np.log10(cols)
            if dz:
                nbins, nedges = make_coldens_hist_dz({'arr': cols}, L_z, redshift, bins = numbins, colmin = colmin,colmax = colmax, cosmopars=cosmopars)
            else: #dX
                nbins, nedges = make_coldens_hist({'arr': cols}, L_z, redshift, bins = numbins, colmin = colmin,colmax = colmax, cosmopars=cosmopars)
            
            bins += nbins // nslabs # number of slabs replace numer of slices in add == 1 case
            print('All except first should be equal to %s: %s'%(str(numbins),str(sum(edges == nedges))))        
            edges = nedges
    
    print('loop finished')    



    if save is not None:
        print('saving %s'%(pdir+save))
        np.savez(pdir+save,bins=bins,logedges=edges)

    return bins, edges







def namecddf(namebase,fills, numbins, colmin, colmax, add, offset, red, L_z, sel, dz):

    if dz == False:
        start = 'cddf_'
    else:
        start = 'cddf-dz_'

    # split file name base into npz-free file name and directory
    dirname = namebase.split('/')
    infilename = dirname[-1]

    if len(fills) == 1: # if it's one slice or a totalbox, we don't need the -all substitution
        infilename = infilename%fills[0] 
    else: #(assuming we're getting the cddf for the whole set, with only one fill)
        infilename = infilename%'-all'

    if infilename[-4:] == '.npz':
        infilename = infilename[:-4]

    dirname = '/'.join(dirname[:-1])
    
    outfilename = start + infilename 
    
    # slicing: thickness, adding, and offsets
    if add == 1:
        sslices = '_%i-x-%fslices'%(len(fills),L_z)
    else:
        sslices = '_%i-x-%f-addto-%fslices_offset%i'%(len(fills) // add, L_z / float(add), L_z, offset)
    # selection
    if sel == (slice(None,None,None),slice(None,None,None)):
        ssel = ''
    else: # default str(slice(...)) has spaces in it -> redo without spaces
        sel0 = list(sel[0].indices(10**16))
        sel1 = list(sel[1].indices(10**16))
        if sel0[1] == 10**16:
            sel0[1] = None
        if sel1[1] == 10**16:
            sel1[1] = None
        ssel = '_slice(%s,%s,%s)-slice(%s,%s,%s)'%(sel0[0],sel0[1],sel0[2],sel1[0],sel1[1],sel1[2])
    # binning (rangeNone-None is acceptable)
    scolrange = '_range%s-%s_%ibins'%(colmin,colmax,numbins)
    
    outfilename += sslices + ssel + scolrange
    
    # pixels/reduction:
    if red != 1:
        # split name after 'pix' and insert '-redto-###'
        pixind = outfilename.find('pix')
        outfilename1 = outfilename[:pixind+3]
        outfilename2 = outfilename[pixind+3:]
        numpixold = ''
        i = pixind-1 # start at last digit
        while(True): # numbers
           numpixold = outfilename1[i] + numpixold # walking backwards, so add new digit to the front
           i -= 1 
           if not outfilename1[i].isdigit():
               break
        numpixold = int(numpixold)
        outfilename = outfilename1 + '-redto-%i'%(numpixold // red) + outfilename2 
    return dirname, outfilename


def getcddf_npztonpz(namebase, fills, numbins = 530, colmin = -25.,colmax = 28., add=1 ,offset=0, red=1,sel = (slice(None,None,None),slice(None,None,None)),dz=False,save=True):
    '''
    !! Only use if the fills include all slices along the line of sight !!
    Uses different default bins from called cddf makers; a more universal standard should make for easiler comparisons, and the range should be sufficiently large
    gets L_z from box size, length of fills (assumes everything is used)
    
    mainly just automatic naming and L_z-finding, but this saves a lot of misery checking these things
    '''

    simdata = get_simdata_from_outputname(namebase) # simnum, snapnum, var
    cosmopars = getcosmopars(simdata['simnum'],simdata['snapnum'],simdata['var'],file_type = 'snap',simulation = simdata['simulation']) # h , omegam, omegalambda, boxsize, a, z
    
    L_z = cosmopars['boxsize']/cosmopars['h'] * float(add)/float(len(fills)) # boxsize in cMpc/h -> cMpc, add/len(fills) gives fraction of whole box in one slice/slab
     
    dirname, outfilename = namecddf(namebase,fills, numbins, colmin, colmax, add, offset, red, L_z, sel, dz)

    if dirname == '': # no /'s in inputted namebase
        namebase = ndir + namebase
    
    if save:
        insave = outfilename
    else:
        insave = None
    
    print('namebase: %s'%namebase)
    print('derived simulation data: %s,\nderived cosmological parameters: %s\nL_z: %f, add: %s, offset: %s, red: %s, sel: %s\noutput name: %s'%(simdata,cosmopars,L_z,add,offset,red,sel,outfilename))
    
    return make_coldens_hist_slices(namebase, fills, L_z, cosmopars['z'], numbins = numbins, colmin = colmin, colmax = colmax, add=add ,offset=offset, red=red, save=insave, sel=sel, dz=dz, cosmopars=cosmopars)


def getcddfav(cddfs,outname): #naming would be tricker to automate here
    # cddfs should be a list or some iterable
    # elements can be strings (npz file names), dicts or loaded npz files ('bins', 'logedges' keys), or [bins, edges] lists or arrays

    if isinstance(cddfs[0],str): #npz file names: load files, and pass on to case handling that
        loadedfiles = []
        for cddf in cddfs:
            if '/' in cddf: # assume all needed directories are included
                pass
            else:
                cddf = pdir + cddf
            if cddf[-4:] != '.npz':
                cddf += '.npz'
            loadedfiles += [np.load(cddf)]
        cddfs = loadedfiles 
        del cddf
 
    if hasattr(cddfs[0],'keys'): # we are dealing with a dict or loaded npz file
        # check whether the average is sensible: using the same column density bins
        edges = np.array([cddf['logedges'] for cddf in cddfs])
        bins = np.array([cddf['bins'] for cddf in cddfs])
    else: # assume list or 2d array bins, edges
        edges = np.array([cddf[1] for cddf in cddfs])
        bins = np.array([cddf[0] for cddf in cddfs])
    
    edgesmatch = not np.any(edges[0,:] - edges[:,:]) # True if bins arrays for all cddfs match exactly
    if not edgesmatch:
        print('logedges arrays do not match; returning bins arrays')
        return edges
    # return makes this effectively an else clause
    binsav = np.average(bins,axis=0) # zero axis varies which cddf is examined
    
    if not '/' in outname:
        outname = pdir + outname 
    if outname[-4:] == '.npz':
        outname = outname[:-4]
    np.savez(outname, bins = binsav, logedges = edges[0])
    return None
