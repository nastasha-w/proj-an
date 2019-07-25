'''
Combines specwizard post-processing with makecddfs to make and analyse cddfs 
using make_maps for a baseline column density distribution
and specwizard for higher-resolution corrections to column densities 
and to get equivalent widths

Due to updates:
!!! only use this stuff for O7 resonance and O8 doublet lines !!!
Or manually set sp.multip below
'''


import numpy as np
import h5py
import scipy

import specwiz_proc as sp
import makecddfs as mc
import make_maps_opts_locs as ol
import eagle_constants_and_units as c
import cosmo_utils as csu
import loadnpz_and_plot as lnp

ndir = ol.ndir
mdir = '/net/luttero/data2/imgs/specwizard/'
pdir = ol.pdir
sdir = '/cosma5/data/dp004/dc-wije1/specwiz/'


### reset specwiz_proc multiplet list to an earlier version with only close lines
# basically, only use this stuff for O7 resonance and O8 doublet lines
sp.multip = {'o8': ['o8major', 'o8minor']}




# need to read in the spectra and column density map (in slices?), 
# set up bins (use best-fit -parameter conversion + margin for EWs by default), 
# get a matrix from map bins to spectra bins (apply COG conversion for EWs below lower limit)
# and get the converted CDDF or EW distribution     

### Possible b paramater issue: ~90 km/s for o7, ~150 km/s for o8 on test10, seems reasonable
# on sample1, it looks a bit different: ~110 km/s for o7 and ~ 13000 km/s for o8, which is especially weird since sample1 does not have o8.
# sp.bparfit([sample1],ions= 'all',EWlog = True, coldenslog = True)
# {u'o7': (array([ 10812363.19632609]), array([[  1.29193751e+09]]))}
# EWlog=False: 11972953.76545441 for o7

# on sample 3, using the o8 average wavelength, summed f_osc in the b parameter fits, with and without restriction to subsamples selected by that ion 
# in cm/s
# using the specwizard-only fitted b parameters
#		fit to EW							fit to log EW
# o7 all:	array([ 9840259.42170963]), array([[  9.59790006e+08]])	 	array([ 9394476.54761442]), array([[  1.11612478e+09]])
# o7 sub:	array([ 9435311.30181743]), array([[  1.95659449e+09]])		array([ 9079723.13153631]), array([[  2.02420167e+09]])
#
# o8 all:	array([ 16821065.58485449]), array([[  1.72628514e+09]])	array([ 16898102.090489]), array([[  1.92059913e+09]])
# o8 sub:	array([ 16952799.79765258]), array([[  5.25352573e+09]])	array([ 17208520.3464764]), array([[  6.11472779e+09]])
# using the o8 doublet calculation (updated bparfit, sum of two doublet absorption lines is integrated)
# o8 all:	array([ 15817274.11244309]), array([[  1.94887757e+09]]))	array([ 15896143.97338231]), array([[  2.17032391e+09]])
# o8 sub:       array([ 15957908.19474436]), array([[  5.91213221e+09]])        array([ 16226916.07507808]), array([[  6.86502354e+09]])


# b parameters redone on correct T4EOS sample3 run, using gettable_bparfits and its print function
#  = snapshot 27 (z=0.1) L0100N1504 sample, uniform N selection in O6, O7, O8 100 Mpc sightlines
# sample of 16384 sightlines total 
#Best fit b parameters [km/s], and sqrt scipy square error est. [km/s]
#
#o7
#	    all				    sub			
#specN
#lin:	98.690890, 0.308938	93.945635, 0.446645
#log:	94.140913, 0.333367	90.388292, 0.451441
#projN
#lin:	96.857264, 0.124567	93.123428, 0.345458
#log:	91.016967, 0.149213	94.989047, 0.247280
#
#o8
#	    all				    sub			
#specN
#lin:	164.591782, 0.407497	 165.442213, 0.707790
#log:	165.215261, 0.431307	 167.670031, 0.764814
#projN
#lin:	158.698163, 0.093699 162.553913, 0.244158
#log:	107.121154, 0.022424	 125.184603, 0.053818
#
#o8d
#	all				sub			
#specN
#lin:	154.316278, 0.434182 155.233485, 0.753059
#log:	154.947876, 0.459858	 157.577218, 0.812915
#projN
#lin:	140.475124, 0.086091	 127.477183, 0.221842
#log:	104.381034, 0.018046	 117.187084, 0.046446
#
#
# b parameters for the sample5 run, using gettable_bparfits and its print function
#  = snapshot 26 (z=0.2) L0100N1504 sample, uniform N selection in O7, O8 100 Mpc sightlines
# sample of 16384 sightlines total 
#Best fit b parameters [km/s], and sqrt scipy square error est. [km/s]
#
#o7
#	    all				     sub			
#specN
#lin:	103.927124, 0.305812	 96.581453, 0.375190
#log:	102.153892, 0.325177	 92.268874, 0.373823
#projN
#lin:	102.624513, 0.220827	 95.316743, 0.290489
#log:	99.999696, 0.135771	 92.520120, 0.186455
#
#o8
#	    all				     sub			
#specN
#lin:	171.134892, 0.454178 173.264218, 0.648272
#log:	170.094022, 0.470184 175.748523, 0.699383
#projN
#lin:	146.578444, 0.105977	 135.534993, 0.201449
#log:	131.156334, 0.027130	 138.202724, 0.046783
#
#o8d
#	    all				     sub			
#specN
#lin:	161.276541, 0.481738 163.538918, 0.686278
#log:	160.149677, 0.499498	 166.146775, 0.739536
#projN
#lin:	122.894515, 0.092938	 157.286214, 0.190804
#log:	140.922845, 0.027891	 106.635944, 0.036115
#
#
# b parameters for the sample4 run, using gettable_bparfits and its print function
#  = snapshot 19 (z=1) L0100N1504 sample, uniform N selection in O7, O8 100 Mpc sightlines
# sample of 16384 sightlines total 
#Best fit b parameters [km/s], and sqrt scipy square error est. [km/s]
#
#o7
#	    all				     sub			
#specN
#lin:	126.107959, 0.342102	 118.916932, 0.453680
#log:	121.288991, 0.345957	 114.125431, 0.455177
#projN
#lin:	123.575501, 0.269640	 116.282855, 0.376939
#log:	111.921316, 0.158157	 105.771215, 0.230097
#
#o8
#	    all				     sub			
#specN
#lin:	191.764232, 0.497520	 186.009072, 0.678074
#log:	188.366319, 0.497246	 182.583633, 0.670824
#projN
#lin:	172.066789, 0.114722	 181.926422, 0.448327
#log:	178.863493, 0.161357	 150.736934, 0.066244
#
#o8d
#	    all				     sub			
#specN
#lin:	183.097842, 0.520495	 177.087861, 0.711084
#log:	179.508836, 0.521390 173.457073, 0.705089
#projN
#lin:	178.059089, 0.112132 172.767887, 0.454904
#log:	148.314645, 0.044856	 109.753965, 0.052232




# go from maps and spectra to matched EW and 2D- and sightline-projected column density arrays

def find_duplicate_positions(pos):
    '''
    pos = Nx3 array (any type)
    returns: set of sorted tuples of indices that all represent the same position
    '''
    equal = np.all(pos[np.newaxis, :, :] == pos[:, np.newaxis, :], axis=2)
    whichequal = [np.where(eqsub)[0] for eqsub in equal]
    whichduplicate = set([tuple(np.sort(which)) if len(which) > 1 else None for which in whichequal]) # need tuples since set needs hashable objects; sort guarantees match for equal sets (should not be a problem, just in case)
    whichduplicate.remove(None)
    return whichduplicate

def find_close_positions(pos, diff):
    '''
    pos = Nx3 array (any type)
    returns: set of sorted tuples of indices that all represent the same position
    '''
    equal = np.all(np.abs(pos[np.newaxis, :, :] - pos[:, np.newaxis, :]) < diff, axis=2)
    whichequal = [np.where(eqsub)[0] for eqsub in equal]
    whichduplicate = set([tuple(np.sort(which)) if len(which) > 1 else None for which in whichequal]) # need tuples since set needs hashable objects; sort guarantees match for equal sets (should not be a problem, just in case)
    whichduplicate.remove(None)
    return whichduplicate

def get_coldens_EW_info(specfile, mapfile, ion, slicecen, sidelength, red=1, periodic=False, slices=1, offset=0., fills=None, slicerange=slice(None,None,None), makenpz=True):
    '''
    Creates dict and npz file containing map column densities, positions, specwizard column densities, and specwizard EWs
    deals with slices internally, since specfile methods get all slices in one go
    fills must be in the same order as slices are put into in Specout: start at first full slice 
    slicerange not currently implemented
    '''
    specout = sp.Specout(specfile,getall=False)
    specout.getcoldens(dions=[ion], slices=slices, offset=offset, realspace=False) # we want velocity space column densities to match the EW measurements
    specout.getEW(dions=[ion], slices=slices, offset=offset)
    # access via specout.EW/coldens[ion] -> array of values by spectrum number (slices==1) 
    # or specout.EW/coldens[ion][(slices,offset)] -> array of values by (spectrum number,slice number), slices start at first full slice along los
    if slices == 1:
        coldensmap = sp.Coldensmap(mapfile, slicecen, sidelength, red=red, periodic=periodic)
        coldensmap.getpixindsvals(specout)
        # acces at coldensmap.coldens_match (coldensmap.coldens_nearmatch for neighbours)
        positions  = specout.positions
        coldens_sp = specout.coldens[ion]
        coldens_mp = coldensmap.coldens_match
        EWs        = specout.EW[ion] 
    else:
        if len(fills) != slices:
            print('Number of slices in Specout and Coldensmap must match')
            return None
        coldensmap = sp.Coldensmapslices(mapfile, fills, slicecen, sidelength, red=red, periodic=periodic)
        coldensmap.getpixindsvals(specout)
        # save into 2d-array: (spectrum number,slice number)
        positions  = specout.positions
        coldens_sp = specout.coldens[ion]
        EWs        = specout.EW[ion] 
        #coldens_mp = np.empty( coldens_sp.shape,dtype=(coldensmap.mapdict[fills[0]]).coldens_match.dtype )
        #for i in range(len(fills)):
        coldens_mp = np.array([ (coldensmap.mapdict[fills[i]]).coldens_match for i in range(len(fills)) ])
        coldens_mp = coldens_mp.T # specnum, slice index order to match EW array 

    savedict = {'position':   positions,\
                'coldens_sp': coldens_sp,\
                'coldens_mp': coldens_mp,\
                'EWs':        EWs,\
                'slices':     np.array([slices]),\
                'offset':     np.array([offset]),\
                'specfile':   np.array([specfile]),\
                'mapfile':    np.array([mapfile]),\
                'fills':      np.array([fills]),\
                'ion':        ion}

    # name output and save
    if makenpz:
        mapfilename = mapfile
        if fills is not None:
            if isinstance(fills[0],str): # just one fill value
                fillin = ('-all',)
            else: # multiple fills
                fillin = ('-all',)*len(fills[0])
            mapfilename = mapfilename%fillin
        if mapfilename[-4:] == '.npz':   # np.savez adds on the .npz extension
            mapfilename = mapfilename[:-4]
        
        specfile_parts = specfile.split('/')
        if 'specwiz' in specfile_parts:  # remove directory
            sdir_ind = np.where(np.array(specfile_parts) == 'specwiz')[0][0]
            specfile_parts = specfile_parts[sdir_ind+1:]
        if specfile_parts[-1][-5:] == '.hdf5': # remove hdf5 extension
            specfile_parts[-1] = specfile_parts[-1][:-5]
        if specfile_parts[-1][:5] == 'spec.': # remove 'spec.' from specwizard auto-naming
            specfile_parts[-1] = specfile_parts[-1][5:]
        #print specfile_parts              
        specfilename = '_'.join(specfile_parts)

        mapfilename = mapfilename.split('/')[-1]
        if mapfilename[-4:] == '.npz':
            mapfilename = mapfilename[:-4]

        filename =  pdir + 'specwizard_map_match_' + mapfilename + '_' + specfilename
        np.savez(filename, **savedict)
    return savedict 

def save_spcm_basics_to_hdf5(openhdf5group, spcmfilename):
    grp = openhdf5group
    spcm = Spcm(spcmfilename)
    
    grp.attrs.create('spcm_filename', spcmfilename)
    grp.create_dataset('column_density_specwizard', data=spcm.cd_sp)
    grp['column_density_specwizard'].attrs.create('units', 'log10 cm^-2')
    grp.create_dataset('column_density_projection', data=spcm.cd_mp)
    grp['column_density_projection'].attrs.create('units', 'log10 cm^-2')
    grp.create_dataset('equivalent_width', data=spcm.EW)
    grp['equivalent_width'].attrs.create('units', 'rest-frame A')
    grp.create_dataset('position', data=spcm.pos)
    grp['position'].attrs.create('units', 'cMpc/h')
    return None

def save_main_spcm_data_to_hdf5():
    specwizfile = h5py.File('/net/luttero/data2/paper1/specwizard_misc.hdf5', 'a')
    
    grp_o7match = specwizfile.create_group('specwizard_projection_match_o7')
    save_spcm_basics_to_hdf5(grp_o7match, '/net/luttero/data2/proc/specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz')
    grp_o8match = specwizfile.create_group('specwizard_projection_match_o8')
    save_spcm_basics_to_hdf5(grp_o8match, '/net/luttero/data2/proc/specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz')
    spcm_o7 = Spcm('/net/luttero/data2/proc/specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz')
    spcm_o8 = Spcm('/net/luttero/data2/proc/specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz')
    selectionf = h5py.File('/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5')
    spcm_o7.subsample_match(np.array(selectionf['Selection/file0/selected_pixels_thision']),norm='pix')
    spcm_o8.subsample_match(np.array(selectionf['Selection/file1/selected_pixels_thision']),norm='pix')
    specwizfile['specwizard_projection_match_o7'].create_dataset('indices_selected_for_o7', data=spcm_o7.indsclosest)
    specwizfile['specwizard_projection_match_o8'].create_dataset('indices_selected_for_o8', data=spcm_o8.indsclosest)
    selectionf.close()
    
    grp_o7 = specwizfile.create_group('EWdists_o7_snap27')
    grp_o8 = specwizfile.create_group('EWdists_o8_snap27')
    comp_EW_dists_o7(hdf5grouptosaveto=grp_o7)
    comp_EW_dists_o8(hdf5grouptosaveto=grp_o8)
    
    comp_EW_dists_o8_snap19_26(specwizfile, snapshot=19)
    comp_EW_dists_o8_snap19_26(specwizfile, snapshot=26)
    comp_EW_dists_o7_snap19_26(specwizfile, snapshot=19)
    comp_EW_dists_o7_snap19_26(specwizfile, snapshot=26)
    
    comp_EW_dists_fullspcms(specwizfile, 'o7', 19)
    comp_EW_dists_fullspcms(specwizfile, 'o8', 19)
    comp_EW_dists_fullspcms(specwizfile, 'o7', 26)
    comp_EW_dists_fullspcms(specwizfile, 'o8', 26)
    
    specwizfile.close()
    
class Spcm:
    def __init__(self,*args,**kwargs):
        '''
        use dict output of get_coldens_EW_info or read in file, otherwise get data from this function
        '''
        if isinstance(args[0],dict): # dictionary from get_coldens_EW_info
            self.dct = args[0]
        elif len(args) == 1: # one file name -> saved dct
            self.dctfile = args[0]
            if '/' not in self.dctfile:
                self.dctfile = pdir + self.dctfile
            self.dct = np.load(self.dctfile)
        else: # process like get_coldens_info call
            self.dct = get_coldens_EW_info(*args,**kwargs)

        self.cd_sp = self.dct['coldens_sp']
        self.cd_mp = self.dct['coldens_mp']
        self.pos   = self.dct['position'] # gadget units
        self.EW    = self.dct['EWs']
        self.specfile = self.dct['specfile'][0]
        self.mapfile  = self.dct['mapfile'][0]
        self.fills    = self.dct['fills']
        if 'ion' in self.dct.keys():
            self.ion  = self.dct['ion']
            if hasattr(self.ion,'item'): # extract string from 0D array
                self.ion = self.ion.item()
        else:
            print('Add ion to Spcm manually (<Spcm instance>.ion = <ion name>)')

    def subsample_match(self,positions, norm=None, cosmopars=None):
        '''
        select a subsample of EW, coldens_sp, coldens_mp based on a list of positions (e.g. from the sightline selection output for a single ion)
        norm = factor to normalise the positions to cMpc/h units
               or 'pix': use simulation data and assume whole slice perpendicular to the projection direction is used to convert pixel index -> position
        '''
        allpositions = self.pos # gagdet units (cMpc/h)
        if norm == 'pix':
            import makecddfs as mc #requires read_eagle, so only do if necessary
            simdatadict = mc.get_simdata_from_outputname(self.mapfile)
            numpix = simdatadict['numpix']
            if cosmopars is None:
                cosmopars = mc.getcosmopars(simdatadict['simnum'],simdatadict['snapnum'],simdatadict['var'],file_type = 'snap',simulation = simdatadict['simulation'])
            possel = (positions + 0.5)/float(numpix) * cosmopars['boxsize'] #boxsize is in cMpc/h units, positions in pixel units are indices, offset from the centres  
        elif norm is not None:
            possel = positions*norm

        self.indsclosest = ( (possel[:, np.newaxis, 0] - allpositions[np.newaxis, :, 0])**2 +\
                             (possel[:, np.newaxis, 1] - allpositions[np.newaxis, :, 1])**2 ).argmin(axis=1)
        
        self.cd_sp = self.cd_sp[self.indsclosest]
        self.cd_mp = self.cd_mp[self.indsclosest]
        self.pos   = self.pos[self.indsclosest,:]
        self.EW    = self.EW[self.indsclosest]

 
    def calccddf(self,**kwargs):
        if self.fills == np.array(None): # fills needs to be a list for the cddf calculator function 
            self.fills = ['']
            self.mapfile += '%s'
        self.cddf_bins, self.cddf_edges = mc.getcddf_npztonpz(self.mapfile, self.fills, **kwargs)
        self.cddfname = self.mapfile
        # kwargs and defaults:
        # numbins = 530, colmin = -25.,colmax = 28., add=1 ,offset=0, red=1,
        # sel = (slice(None,None,None),slice(None,None,None)),dz=False,save=True

    def readcddf(self,filename):
        '''
        Only gets bins and edges; combined with the SpCm N-EW relation
        sets cddfname for corrct extractions of CDDF -> histogram parameters
        '''
        if '/' not in filename:
            self.cddfname = pdir + filename
        else:
            self.cddfname = filename
        self.cddf = np.load(self.cddfname)
        self.cddf_bins  = self.cddf['bins']
        self.cddf_edges = self.cddf['logedges'] 
        del filename

    def get_coldensconv(self,ratio=True, edges_mp = None, edges_target = None):
        '''
        if no edges are given, they will be matched to the cddf; this assumes the cddf has already been calculated or read in
        generates a histogram of make_maps col. dens. (x) vs. specwizard col. dens., or specwizard/make_maps col. dens. (y) 
        '''
        if edges_mp is None: # default is to match cddf
            self.conv_edges_mp = self.cddf_edges
            self.conv_edges_mp = np.array(list(self.conv_edges_mp) + [self.conv_edges_mp[-1] + np.average(np.diff(self.conv_edges_mp))])   #adding missing rightmost edge 
        else:
            self.conv_edges_mp = edges_mp
        
        if ratio: # need fine binning to avoid generating spurious scatter in cddf conversion
            self.cd_ratio = self.cd_sp - self.cd_mp # these are log values
            self.minratio = np.min(self.cd_ratio)
            self.maxratio = np.max(self.cd_ratio)

            # set target (y) edges
            if edges_target is None:
                self.numsteps = np.round( (self.maxratio - self.minratio)/0.005,0) # get step size of ~ 0.005
                self.conv_edges_ratio = np.arange(self.minratio*0.9999, self.maxratio*1.0002, (self.maxratio*1.0001 - self.minratio*0.9999)/self.numsteps)
            elif isinstance(edges_target,int): # number of steps meant
                self.conv_edges_ratio = np.arange(self.minratio*0.9999, self.maxratio*1.0002, (self.maxratio*1.0001 - self.minratio*0.9999)/float(edges_target))
            elif isinstance(edges_target,float): # step size meant
                self.conv_edges_ratio = np.arange(self.minratio*0.9999, self.maxratio*(1. + edges_target*0.999), edges_target)
            else:
                if self.minratio < edges_target[0]:
                    print('Warning: ratios below indicated minimum are present')
                if self.maxratio > edges_target[-1]:
                    print('Warning: ratios above indicated maximum are present')
                self.conv_edges_ratio = edges_target


            self.conv_ratio, checkx, checky = np.histogram2d(self.cd_mp,self.cd_ratio,bins= [self.conv_edges_mp, self.conv_edges_ratio])
            if not ( np.all(checkx==self.conv_edges_mp) and np.all(checky = self.conv_edges_ratio) ):
                print('Error in histogram2d: input and output edges do not match') 
                return None
        else:
            self.minsp = np.min(self.cd_sp)
            self.maxsp = np.max(self.cd_sp)

            # set target (y) edges
            if edges_target is None:
                self.numsteps = np.round( (self.maxsp - self.minsp)/np.average(np.diff(self.conv_edges_mp )),0) # get step size equal to that in input array
                self.conv_edges_sp = np.arange(self.minsp*0.9999, self.maxsp*1.0002, (self.maxsp*1.0001 - self.minsp*0.9999)/self.numsteps)
            elif isinstance(edges_target,int): # number of steps meant
                self.conv_edges_sp = np.arange(self.minsp*0.9999, self.maxsp*1.0002, (self.maxsp*1.0001 - self.minsp*0.9999)/float(edges_target))
            elif isinstance(edges_target,float): # step size meant
                self.conv_edges_sp = np.arange(self.minsp*0.9999, self.maxsp*(1. + edges_target*0.999), edges_target)
            else:
                if self.minsp < edges_target[0]:
                    print('Warning: specwizard column densities below indicated minimum are present')
                if self.maxsp > edges_target[-1]:
                    print('Warning: specwizard column densities above indicated maximum are present')
                self.conv_edges_sp = edges_target


            self.conv_mpsp, checkx, checky = np.histogram2d(self.cd_mp,self.cd_sp,bins= [self.conv_edges_mp, self.conv_edges_sp])
            if not ( np.all(checkx==self.conv_edges_mp) and np.all(checky = self.conv_edges_sp) ):
                print('Error in histogram2d: input and output edges do not match') 
                return None

    def get_COG(self, edges_cd = None, edges_ew = None, usesp = False):
        '''
        usesp = use specwizard column densities (in stead of projected)
        '''
        if usesp:
            self.cd = self.cd_sp
        else:
            self.cd = self.cd_mp

        if edges_cd is None: # default is to match cddf
            self.cog_edges_cd = self.cddf_edges
            self.cog_edges_cd = np.array(list(self.cddf_edges) + [self.cddf_edges[-1] + np.average(np.diff(self.cddf_edges))])   #adding missing rightmost edge 
        else:
            self.cog_edges_cd = edges_cd
        
        if edges_ew is None: # use linear COG to get evenly spaced edges to match coldens bins (with margins)
            try:
                self.cog_edges_ew = np.log10(sp.lingrowthcurve_inv(10**self.cog_edges_cd, self.ion))
            except KeyError: # multiple lines -> major gives the one used in specwizard
                self.cog_edges_ew = np.log10(sp.lingrowthcurve_inv(10**self.cog_edges_cd, self.ion + 'major'))
            self.cog_edges_ew = np.arange(np.log10(np.min(self.EW))*0.9999,np.log10(np.max(self.EW))*1.0001,np.average(np.diff(self.cog_edges_ew)))
        else:
            self.cog_edges_ew = edges_ew
            if np.min(self.EW) < self.cog_edges_ew[0]:
                print('Warning: specwizard EWs below indicated minimum are present')
            if np.max(self.EW) > self.cog_edges_ew[-1]:
                print('Warning: specwizard EWs above indicated maximum are present')
        
        self.cog, checkx, checky = np.histogram2d(self.cd,np.log10(self.EW),bins= [self.cog_edges_cd, self.cog_edges_ew])
        if not ( np.all(checkx==self.cog_edges_cd) and np.all(checky == self.cog_edges_ew) ):
            print('Error in histogram2d: input and output edges do not match') 
            return None



    def gethist_fromcddf(self, cosmopars=None):
        '''
        Assumes any slice covers the full depth of the box, and EAGLE cosmology
        should probably work for other numbers of slices as well, as long as 
        the slices add up to the whole box
        
        uses the name of the read-in cddf to extract simulation data if
        readccdf was used, or the mapfile name otherwise (cddfname, set by both
        cddf generation methods)
        
        cosmopars option is meant for cases where the snapshot is not present 
        on the system
        '''       
        self.simdata = mc.get_simdata_from_outputname(self.cddfname)
        if cosmopars is None:
            self.cosmopars = mc.getcosmopars(self.simdata['simnum'],self.simdata['snapnum'],self.simdata['var'],file_type = 'snap',simulation = self.simdata['simulation'])
        else:
            self.cosmopars = cosmopars 
        print('Using simdata, cosmopars:')
        print(self.simdata)
        print(self.cosmopars)
        self.hist = mc.cddf_over_pixcount(self.cosmopars['z'],self.cosmopars['boxsize']/self.cosmopars['h'],self.simdata['numpix']**2, self.cddf_edges, cosmopars=self.cosmopars)*self.cddf_bins

        self.hist = np.round(self.hist,0).astype(int)
        print('Retrieved %i pixels -> %f slices'%(np.sum(self.hist),np.sum(self.hist)/float(self.simdata['numpix']**2) ))
        

#### 1-d interpolation methods that explicitly conserve bin count
def interp0(data_bins,data_edges,target_edges,margin_upper=0,margin_lower=0):
    '''
    margin: number of data bins on each end provided only for interpolation purposes (unused in this 0-order scheme)

    tested: no margin, block functions, evenly spaced bins and irregular bins in data and target
    '''
    # uses newaxis explicitly to prevent broadcasting errors in the case of square matrices    

    #  ----+---------+---- data bins
    #  ---- _________ '''  assumed values throught each bin (flat)
    #  +------+------+---- 
    zeromatrix = np.zeros((len(data_edges)-margin_upper-margin_lower-1,len(target_edges)-1),dtype=np.float)
    # calculate fraction of each data cell overlapping each target cell
    # for 6 cases: 
    #    - if there is any overlap, it is always the smallest right edge - largest left edge value
    #    - if there is no overlap, this value is negative
    # this is done for each data (x/0 axis) and target (y/1 axis) pair
    #  +---+     |     +---+  |    +-+     |  +-----+  | +--+      |      +-+
    #    +---+   |  +---+     | +-------+  |    +-+    |      +-+  | +--+     

    # 0-order interpolation means no extra data bins are used for interpolation -> just ignore any margin bins
    data_bins = data_bins[margin_lower:len(data_bins)-margin_upper]
    data_edges = data_edges[margin_lower:len(data_edges)-margin_upper]
    
    if len(data_bins) == 0:
        print('No data to interpolate')
        return np.array([],dtype=np.float)
    elif len(data_bins) != len(data_edges) -1:
        print('Data bin and edge array lengths are not compatible. (Edges should contain the leftmost and rightmost edge.)')
        return None
 
    zeromatrix = np.zeros((len(data_edges)-1,len(target_edges)-1),dtype=np.float)

    
    data_bin_sizes   = np.diff(data_edges) # for normalisation
    conv_matrix      = np.max(np.array([ zeromatrix,\
                                         np.min( np.array([np.repeat(data_edges[1:,np.newaxis],len(target_edges)-1,axis=1),\
                                                           np.repeat(target_edges[np.newaxis,1:],len(data_edges)-1,axis=0)\
                                                          ]),axis=0)\
                                         - \
                                         np.max( np.array([np.repeat(data_edges[:-1,np.newaxis],len(target_edges)-1,axis=1),\
                                                           np.repeat(target_edges[np.newaxis,:-1],len(data_edges)-1,axis=0)\
                                                          ]),axis=0)\
                                       ])\
                              ,axis=0)

    #print conv_matrix 
    # normalise over y axis to conserve counts (divide by data bin size -- sum over y axis fails if data bin is not (entirely) in target range)
    conv_matrix /= data_bin_sizes[:,np.newaxis] 
    #print conv_matrix
    target_bins = np.sum(conv_matrix*data_bins[:,np.newaxis],axis=0) # matrix * vector
    return target_bins

    
def convert_hist_to_hist(x_bins,x_edges,matrix, matrix_x, matrix_y, extend_lower = None, save=None, ratio = False, overlap_interpolation_method = 0):
    '''
    takes input histogram with x_bins, x_edges data
    and converts it to an output histogram
    using a transformation matrix and some extension for lower values
    which directly converts the bin edges (i.e. assumes no scatter) 
    the converted lower bins are assumed not to 'stick out' above the matrix y values, and have a minimum value below matrix y

    Has some options for conversion of values smaller than the matrix
    includes, but not for larger values. (That is where the tricky parts 
    are in col. dens. corrections and EW modelling.)
    Matrix x bins must match the largest data x_bins used;  
    self.cog from  get_COG and
    self.conv_mpsp, self.conv_ratio from get_coldensconv will produce such matrices
     
    tested: all in matrix range: diagonal, off-diagonal, scatter
            extend_lower: 'equal', lambda x: x+1 on id, +1, also extended with +0.5, -0.5. +1.5, +2.5 (different overlap cases), diagonal, off-diagonal, scatter 
            see hist_conversion_tests.py for the matrices used
    input:
    -----------------------------------
    x_bins, x_edges: binned data - edges in include right edges
                     bins should be counts, not e.g. a cddf with some 
                     normalisation that differs from bin to bin 
    matrix, _x, _y : conversion matrix for x data -> some y quantity
                     values and the x and y edges for each matrix bin
    extend_lower:    how to extrapolate the conversion to x data below 
                     the matrix_x range
                     'equal' -> convert to the same value 
                    +-           (e.g. make_maps to specwizard col. dens.)
      assumes x in  |'lincog_<ion>'     -> use the linear curve of growth 
      log_10 cm^-2,-+                      for <ion> 
      b in cm/s     |'fitcog_<ion>_<b>' -> use the b parameter <b> COG for
                    |                      <ion> 
                    +- add _log to end if EW bins are in log Angstroms                    

                     <function> -> convert using that function 
                                   (must work on arrays)
    overlap_interpol how to interpolate bins onto a new grid if there is
    ation_method:    significant overlap between the matrix and lower 
                     extension region outcomes 
                     (small overlaps just lead to one larger bin for
                     the overlap)
                     options are 0, 1, 2 for order of polynomial 
                     interpolation
    '''         
    ### check input data ordering
    # check if x_bins and matrix_x, matrix_y are sorted low-to-high, as is assumed
    # if ordered high-to-low, just flip
    if np.all(np.diff(x_edges) > 0 ): # zero size bins will give normalisation issues
        pass # all is fine
    elif np.all(np.diff(x_edges) < 0 ): # reverse ordering
         x_bins = x_bins[::-1]
         x_edges = x_edges[::-1]
    else:
        print('Input x data should be ordered')
        return None

    if np.all(np.diff(matrix_x) > 0 ): # zero size bins will give normalisation issues
        pass # all is fine
    elif np.all(np.diff(matrix_x) < 0 ): # reverse ordering
         matrix_x = matrix_x[::-1]
         matrix   = matrix[::-1,:]
    else:
        print('x matrix edge values should be ordered')
        return None
    
    if np.all(np.diff(matrix_y) > 0 ): # zero size bins will give normalisation issues
        pass # all is fine
    elif np.all(np.diff(matrix_y) < 0 ): # reverse ordering
         matrix_x = matrix_y[::-1]
         matrix   = matrix[:,::-1]
    else:
        print('y matrix edge values should be ordered')
        return None
    ### check that bins and edges match up in number
    if len(matrix_x) != matrix.shape[0] + 1:
        print('matrix_x does not match matrix x dimension')
    if len(matrix_y) != matrix.shape[1] + 1:
        print('matrix_y does not match matrix y dimension')
    if len(x_edges) != len(x_bins) +1:
        print('Length of x_bins and x_edges do not match')

    ### ensure correct normalisation of conversion matrix to conserve number of bin counts
    #matrix should conserve number of input bins: for each x bin, get a y PDF
    ynorms =  np.sum(matrix,axis=1)
    ynorms[ynorms==0] = 1. # if a y pdf is empty, should map to nothing; set norm to 1. to avoid 0./0. erros
    matrix = matrix / ynorms[:,np.newaxis]

    ### check what will be needed in the conversion: extrapolation, interpolation
    # if the x bins in the matrix and data match, the job will be easy
    xmatch = False
    do_extend_lower = False
    domorecheck = True

    # max data larger than max value included in the matrix
    if x_edges[-1] > matrix_x[-1]:
        print('Extrapolation of data binning above matrix x range is not implemented')
        return None
    
    # x data extends over the full matrix range, possibly to lower values
    elif len(x_edges) == len(matrix_x):
        if np.all(x_edges==matrix_x): # all bins match
            xmatch = True
            domorecheck = False

    elif len(matrix_x) > len(x_edges):
        if np.all(x_edges[-len(matrix_x):] == matrix_x): # highest bins match; need to extend lower values.
            do_extend_lower = True
            domorecheck = False

    # x data does not extend to max matrix x value, possily to lower values
    # check if the bins in the data range do match the matrix bins
    if domorecheck:
        if x_edges[0] < matrix_x[0]: # lower range 'sticks out', previous check have not approved yet
            if np.all( x_edges[x_edges >= matrix_x[0]] == matrix_x[:np.sum(x_edges >= matrix_x[0])]): # upper part of x_bins matches lower bins of matrix_x 
                lenover = np.sum(x_edges >= matrix_x[0]) # number of edges that match 
                # throw out unneeded matrix parts
                matrix = matrix[:lenover-1,:]
                matrix_x = matrix_x[:lenover]
                do_extend_lower=True       
 
                if lenover == 0:
                    print('Data is fully outside matrix range; just use extend_lower to do your conversion')
                    return None     
            else:
                print('highest data bins must match matrix bins')
                return None
        # check if x_bins is some contiguous subarray of matrix_x
        elif np.sum(x_edges[0]==matrix_x) != 0: # there is a match to the first edge    
            startind = np.where(x_edges[0]==matrix_x)[0][0]
            if np.all(x_edges == matrix_x[startind:startind+len(x_edges)]): # the subarray matches
                # throw out unneeded matrix parts
                matrix = matrix[startind:startind+len(x_edges)-1]
                matrix_x = matrix_x[startind:startind+len(x_edges)]
                xmatch = True
            else: # no subarray match
                print('Data x edges do not match matrix edges')
                return 0
        else:
            print('highest data bins must match matrix bins')
            return None

    
    ### actually do the conversion for different cases

    if xmatch: # just do a matrix multiplication
        yout_bins  = np.sum(matrix*x_bins[:,np.newaxis],axis=0)
        yout_edges = matrix_y

    elif do_extend_lower:
        # for the larger x values, same as in the matched case 
        yout_bins_upper = np.sum(matrix*x_bins[-len(matrix_x)+1:,np.newaxis],axis=0)
        yout_edges_upper = matrix_y

        # for the lower values, convert edges using the chosen method (scatter modelling not implemented)
        # define conversion function depending on extend_lower method
        if hasattr(extend_lower, '__call__'): #use some defined function
            def convfunc(x):
                return extend_lower(x)        
        elif extend_lower == 'equal':
            def convfunc(x):
                return x
        elif 'lincog' in extend_lower: # assumes N_ion is in log_10 cm^-2
            extend_lower = extend_lower.split('_')
            ion = extend_lower[1]   
            if 'log' in extend_lower:
                def convfunc(x):
                    return np.log10(sp.lingrowthcurve_inv(10**x,ion)) 
            else: 
                def convfunc(x):
                    return sp. lingrowthcurve_inv(10**x,ion) 
        elif 'fitcog' in extend_lower: # assumes N_ion is in log_10 cm^-2, b in cm/s
            extend_lower = extend_lower.split('_')
            ion = extend_lower[1]
            b   = float(extend_lower[2]) 
            if 'log' in extend_lower:
                def convfunc(x):
                    return np.log10(sp.linflatcurveofgrowth_inv(10**x,b,ion))
            else:   
                def convfunc(x):
                    return sp.linflatcurveofgrowth_inv(10**x,b,ion)
        else:
            print('An extension of the conversion to lower x values is required. Please specify a (valid) extension method.')
            return None
        
        yout_edges_lower = np.array(convfunc(x_edges[:-len(matrix_x)+1])) # include the upper/lower border edge here
        yout_bins_lower  = np.array(x_bins[:-len(matrix_x)+1])            # no scatter: bins translate directly

        #print('y edges upper: %s'%yout_edges_upper)
        #print('y bins upper: %s'%yout_bins_upper)
        #print('y edges lower: %s'%yout_edges_lower)
        #print('y bins lower: %s'%yout_bins_lower)
        #guarantee arrays are 1D (may be 0D after selections)
        yout_edges_lower = yout_edges_lower.flatten() 
        yout_bins_lower = yout_bins_lower.flatten()
        yout_edges_upper = yout_edges_upper.flatten() 
        yout_bins_upper = yout_bins_upper.flatten()
      
        # combine upper and lower y values obtained into a single distribution   
        if yout_edges_upper[0] == yout_edges_lower[-1]: # lower and upper bins translate seamlessly -> just glue the bins and edges arrays together
           yout_edges = np.array(list(yout_edges_lower[:-1]) + list(yout_edges_upper))
           yout_bins  = np.array(list(yout_bins_lower) + list(yout_bins_upper))

        elif yout_edges_upper[0] > yout_edges_lower[-1]: # there is a gap in the y values; this should not generallly occur in e.g. cddfs. 
                                                         # fill in zero counts in the gap
           print('There is a gap between the matrix and lower extrapolated y values: %f to %f'%(yout_edges_lower[-1],yout_edges_upper[0]))
           yout_edges = np.array(list(yout_edges_lower) + list(yout_edges_upper))
           yout_bins  = np.array(list(yout_bins_lower) + [0.] + list(yout_bins_upper))

        else: # expected case if there is scatter at the lower end of the matrix conversion range or if the bins do not match exactly   
           # separate in affected and unaffected bins in the lower and upper regions
           where_edges_lower_nooverlap = yout_edges_lower <= yout_edges_upper[0]
           where_edges_upper_nooverlap = yout_edges_upper >= yout_edges_lower[-1]
           
           edges_lower_nooverlap = yout_edges_lower[where_edges_lower_nooverlap]
           bins_lower_nooverlap = yout_bins_lower[where_edges_lower_nooverlap[1:]] # right edges not in overlap region
           edges_upper_nooverlap = yout_edges_upper[where_edges_upper_nooverlap]
           bins_upper_nooverlap = yout_bins_upper[where_edges_upper_nooverlap[:-1]] # left edges not in overlap region

           
           # extreme edges are outside the overlap region itself
           if len(edges_lower_nooverlap) > 0:
               edges_lower_overlap = np.array([edges_lower_nooverlap[-1]] + list(yout_edges_lower[np.logical_not(where_edges_lower_nooverlap)]))
               bins_lower_overlap = yout_bins_lower[np.logical_not(where_edges_lower_nooverlap[1:])] # right edges in overlap region
           else:
               edges_lower_overlap = yout_edges_lower[np.logical_not(where_edges_lower_nooverlap)]
               bins_lower_overlap = yout_bins_lower
           if len(edges_upper_nooverlap) > 0:
               edges_upper_overlap = np.array(list(yout_edges_upper[np.logical_not(where_edges_upper_nooverlap)]) + [edges_upper_nooverlap[0]])
               bins_upper_overlap = yout_bins_upper[np.logical_not(where_edges_upper_nooverlap[:-1])] # left edges in overlap region
           else:
               yout_edges_upper[np.logical_not(where_edges_upper_nooverlap)] 
               bins_upper_overlap = yout_bins_upper # left edges in overlap region
          
           if yout_edges_upper[-1] < yout_edges_lower[-1] or yout_edges_upper[0] < yout_edges_lower[0]:
               print('extend_lower region produces too large output values: some above matrix y values or all above lower matrix y edge')
               return None
           
           minbinsize = min([np.min(np.diff(yout_edges_upper)),np.min(np.diff(yout_edges_lower))])
           if yout_edges_lower[-1] - yout_edges_upper[0] < 2.*minbinsize: 
               # overlap is small -> just merge the bins that overlap 
               # (the bin can become large, but this region will be a bit messy whatever I do)
               # equally spaced bins -> you get a ~2.5 * normal size bin
               # max. 2 * max bin size + 2 * min bin size - epsilon bin size
               # +---x-----+--+
               # |   |  +---+----x----+---+
               # |   |           |    |   |
               # |   |           |    |   |
               # +---+-----------+----+---+

               yout_bins_overlap = np.sum(bins_lower_overlap) + np.sum(bins_upper_overlap)

               yout_edges = np.array(list(edges_lower_nooverlap) + list(edges_upper_nooverlap))
               yout_bins  = np.array(list(bins_lower_nooverlap) + [yout_bins_overlap] + list(bins_upper_nooverlap))

           else: 
                # the overlap region is relatively large -> make new bins in the overlap region and interpolate upper and lower bins onto them          
                # +---x-----+--+----+
                # |   |  +---+----+----x---+
                # |   |                |   |
                # |   |                |   |
                # +---+---+---+---+----+---+
  
                # find resonale overlap region bins (upper and lower bins are assumed to interally have similar sizes, 
                # but the sizes over the two may differ more)
                aimsize = max([ np.average(np.diff(edges_lower_overlap)), np.average(np.diff(edges_upper_overlap)) ])
                overlap_leftedge  = edges_lower_overlap[0]
                overlap_rightedge = edges_upper_overlap[-1]
                numbins = np.round((overlap_rightedge - overlap_leftedge)/aimsize,0)
                # put the left and right edges in explicitly to guarantee a mtach to the no-overlap regions
                yout_edges_overlap = np.array( [overlap_leftedge] +\
                                                list(overlap_leftedge + np.arange(1,int(numbins)-1)*(overlap_rightedge-overlap_leftedge)/numbins ) +\
                                                [overlap_rightedge] )
                # interpolate bins onto new grid
                if overlap_interpolation_method == 0:
                    yout_overlap_upper_bincontr = interp0(bins_upper_overlap,edges_upper_overlap,yout_edges_overlap,margin_upper=0,margin_lower=0)
                    yout_overlap_lower_bincontr = interp0(bins_lower_overlap,edges_lower_overlap,yout_edges_overlap,margin_upper=0,margin_lower=0)
                    yout_bins_overlap = yout_overlap_upper_bincontr + yout_overlap_lower_bincontr
                else:
                    print('Only 0-order interpolation is currently implemented')
                    return None
                
                yout_edges = np.array(list(edges_lower_nooverlap) + list(yout_edges_overlap[1:-1]) + list(edges_upper_nooverlap))
                yout_bins = np.array(list(bins_lower_nooverlap) + list(yout_bins_overlap) + list(bins_upper_nooverlap))

    else: # need to rebin the bins or matrix
        if x_bins[0] < matrix_x[0] or x_bins[-1] > matrix_x[-1]:
            print('Warning: matrix will have to be extrapolated for conversion')
        print('Conversion for non-matching bins is not yet implemented')        
    
    if save is not None:
        if '/' not in save:
            save = pdir + save
        np.savez(save,bins=yout_bins,edges=yout_edges)
    return yout_bins, yout_edges



def bparfit(spcms, EWlog=True, usespecwizcoldens=False, cbounds=(None,None), singletfit=False, **kwargs):
    '''
    more or less a copy of specwiz_proc's bparfit, just for a different input class 
    b in cm/s throughout
    fit the earlier linflatcurveofgrowth to specouts for the best-fit  parameter
    '''
    # kwargs: e.g. bounds for scipy.optimize.curve_fit

    try:
        ion = spcms.ion
    except AttributeError: # it was a list of specouts
        try: 
            ion = spcms[0].ion 
        except AttributeError: # it has not been set yet in Spcm; input manually
            ion = kwargs['ion']

    try: # we've only got one
        if usespecwizcoldens:
            xs = spcms.cd_sp
        else:
            xs = spcms.cd_mp
        ys = spcms.EW
    except AttributeError: # spcms was a list 
        # find how many spetra we're dealing with, in each specout
        numspecs = [len(spcm.EW) for spcm in spcms]
        numspecstot = sum(numspecs)
        # get start/stop indices for each spectrum in final total array
        specstarts = [sum(numspecs[:i]) for i in range(len(numspecs))]
        specstarts += [numspecstot]
        # set up NaN arrays -> should be clear if something has gone wrong filling them
        xs = np.ones(numspecstot)*np.NaN
        ys = np.ones(numspecstot)*np.NaN
        # fill total x/y arrays with values form each specout
        for i in range(len(spcms)):
            if usespecwizcoldens:
                xs[specstarts[i]:specstarts[i+1]] = spcms[i].cd_sp
            else:
                xs[specstarts[i]:specstarts[i+1]] = spcms[i].cd_mp
            ys[specstarts[i]:specstarts[i+1]] = spcms[i].EW

    # depending on whether or not to fit in log space, the fit function and input values need to be modified  
    xs = 10**xs # column densities from log column densities
    if EWlog:
        ys = np.log10(ys)
    
    if ion == 'o8' and singletfit:
        ion = 'o8_assingle'
     
    if not EWlog:
        def fitfunc(x,b):
           return sp.linflatcurveofgrowth_inv(x,b,ion)    
    else:
        def fitfunc(x,b):
            return np.log10(sp.linflatcurveofgrowth_inv(x,b,ion))
        

        # inital guess from visual fit:  ~ 100 km/s. 0 < b < speed of light is physically required  
        # outputs optimal, covariance
    if 'ion' in kwargs.keys(): # not for curve_fit
        del kwargs['ion']
    out = scipy.optimize.curve_fit(fitfunc, xs, ys, p0 = 100*1e5,bounds=(np.array([0]),np.array([3e5*1e5])),**kwargs)        

    return out

outbins = np.arange(-25.,25,0.05)
inbins = outbins


def gettable_bparfits(snap=27):
    if snap == 27:
        spcm_o7_name = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
        spcm_o8_name = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'
        sampledataiondir_o7 = 'file0'
        sampledataiondir_o8 = 'file1'
    elif snap == 26:
        spcm_o7_name = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample5_snap_026_z000p183.0.npz'
        spcm_o8_name = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample5_snap_026_z000p183.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/sample5/los_sample5_o7-o8_L0100N1504_snap26_data.hdf5'
        sampledataiondir_o7 = 'file0'
        sampledataiondir_o8 = 'file1'
    elif snap == 19:
        spcm_o7_name = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample4_snap_019_z001p004.0.npz'
        spcm_o8_name = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample4_snap_019_z001p004.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/sample4/los_sample4_o7-o8_L0100N1504_snap19_data.hdf5'
        sampledataiondir_o7 = 'file0'
        sampledataiondir_o8 = 'file1' 
    else:
        raise ValueError('Snapshot %i is not an option'%snap)
    
    simdata = mc.get_simdata_from_outputname(spcm_o7_name)
    try:
        cosmopars = mc.getcosmopars(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
        L_z = cosmopars['boxsize'] / cosmopars['h']
    except: # something failed; just copy parameters from the EAGLE wiki and constants & units
        # typically, due to snapshot not being on the system being used
        cosmopars = {}
        cosmopars['h'] = c.hubbleparam
        cosmopars['omegam'] = c.omega0
        cosmopars['omegab'] = c.omegabaryon
        cosmopars['omegalambda'] = c.omegalambda
        L_z = 100.
        cosmopars['boxsize'] = L_z * cosmopars['h']
        if snap == 19: 
            cosmopars['a'] = 0.498972
        elif snap == 26:
            cosmopars['a'] = 0.845516 
        cosmopars['z'] = 1. / cosmopars['a'] - 1.
        
    spcm_o7 = Spcm(spcm_o7_name)
    spcm_o8 = Spcm(spcm_o8_name)
    
    spcm_o7_sub = Spcm(spcm_o7_name)
    spcm_o8_sub = Spcm(spcm_o8_name)
    samplefile = h5py.File(sampledataname,'r')
        
    positions_o7 = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir_o7)])
    spcm_o7_sub.subsample_match(positions_o7, norm='pix', cosmopars=cosmopars)

    positions_o8 = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir_o8)])
    spcm_o8_sub.subsample_match(positions_o8 ,norm='pix', cosmopars=cosmopars)
    
    dct_out = {}
    # grid: ion (o7, o8, o8 singlet fit) x specwizard, projected col. dens. x EWlog (True, False) x subset/all values 
    for gind in range(3*2*2*2):
        ionind = gind / (2*2*2)
        cvarind = gind / (2*2) - 2*ionind
        EWlogind = gind / 2 - 2*cvarind - 2*2*ionind
        setind = gind % 2
        
        if ionind == 0:
            ion = 'o7'
            singletfit = True
        elif ionind == 1:
            ion = 'o8'
            singletfit = False
        elif ionind == 2:
            ion = 'o8'
            singletfit = True
        else:
            print('Wrong value %i for ionind'%ionind)
        
        if setind == 0 and ion == 'o7':
            spcm = spcm_o7
        elif setind == 1 and ion == 'o7':
            spcm = spcm_o7_sub
        elif setind == 0 and ion == 'o8':
            spcm = spcm_o8
        elif setind == 1 and ion == 'o8':
            spcm = spcm_o8_sub
        else:
            print('Wrong setind, ion combination %i, %s'%(setind, ion))
        usespecwizcoldens = bool(cvarind)
        EWlog = bool(EWlogind)
        
        if EWlog:
            sEWlog = 'EWlog'
        else:
            sEWlog = 'EWlin'
        if ion == 'o8' and not singletfit:
            iontag = 'd'
        else:
            iontag = ''
        if setind == 0:
            sset = 'all'
        else:
            sset = 'sub'
        if usespecwizcoldens:
            sN = 'specN'
        else:
            sN = 'projN'
        
        res = bparfit(spcm, EWlog=EWlog, usespecwizcoldens=usespecwizcoldens, cbounds=(None,None), singletfit=singletfit)
        name = '%s%s_%s_%s_%s'%(ion, iontag, sN, sEWlog, sset)
        dct_out[name] = res
    
    return dct_out

def print_bparfittable_out_to_txt(dct_out):
    #all_keys = dct_out.keys()
    ions = ['o7', 'o8', 'o8d']
    Nvar = ['specN', 'projN']
    lvar = ['lin', 'log']
    sets = ['all', 'sub']
    
    print('Best fit b parameters [km/s], and sqrt scipy square error est. [km/s]')
    for ion in ions:
        print('\n%s'%ion)
        print(('\t%s\t\t\t'*len(sets))%tuple(sets))
        for typevar in Nvar:
            print(typevar)
            for fitvar in lvar:
                keys = ['%s_%s_EW%s_%s'%(ion, typevar, fitvar, set_) for set_ in sets]
                str_base = '%s:'+ '\t%f, %f' * len(keys) 
                str_fill = (fitvar,) + tuple([dct_out[key][ind][0]/1e5 if ind==0 else np.sqrt(dct_out[key][ind][0])/1e5 for key in keys for ind in range(2)])
                str_out = str_base % str_fill
                print(str_out)
            

### how I got the sample1 EW distribution
# import combine_specwiz_makemaps as csm
# specdata = csm.Spcm('specwizard_map_match_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample1_snap_027_z000p101.0.npz')
# specdata.readcddf('cddf_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_1-x-100.000000slices_range-25.0-28.0_1060bins.npz')
# specdata.ion='o7' #was not recorded in the earlier npz generation method-> put in manually
# specdata.get_COG()
# specdata.gethist_fromcddf()
# edgesok = specdata.cog_edges_cd >=13.
## feature at ~10^-4.5 \AA, presumably due to switch to direct conversion of from N to EW use a linear COG (it's at the right EW for that) 
## looks almost extactly the same using b=108 km/s for extend_lower (confirming that this is quite firmly in the linear regime)
# EWbins, EWedges = csm.convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok[:-1],:], specdata.cog_edges_cd[edgesok], specdata.cog_edges_ew, extend_lower = 'lincog_o7_log', save=None, ratio = False, overlap_interpolation_method = 0)
# EWbins_f, EWedges_f = csm.convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok[:-1],:], specdata.cog_edges_cd[edgesok], specdata.cog_edges_ew, extend_lower = 'fitcog_o7_108e5_log', save=None, ratio = False, overlap_interpolation_method = 0)
## trying larger switchover value
# edgesok = specdata.cog_edges_cd >=14.
# EWbins_hs, EWedges_hs = csm.convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok[:-1],:], specdata.cog_edges_cd[edgesok], specdata.cog_edges_ew, extend_lower = 'lincog_o7_log', save=None, ratio = False, overlap_interpolation_method = 0)
# EWbins_f_hs, EWedges_f_hs = csm.convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok[:-1],:], specdata.cog_edges_cd[edgesok], specdata.cog_edges_ew, extend_lower = 'fitcog_o7_108e5_log', save=None, ratio = False, overlap_interpolation_method = 0)

# should be able to add a 16-slice cddf in readcddf and just get the EW distriution assuming the 100cMpc-slice N-EW relation

    #Best fit b parameters [km/s], and sqrt scipy square error est. [km/s]
#
#o7
#	all				sub			
#specN
#lin:	98.692515, 0.308945	94.355857, 0.442312
#log:	94.142950, 0.333377	90.797881, 0.449924
#projN
#lin:	97.252969, 0.217881	93.312712, 0.340881
#log:	91.020095, 0.149217	89.882789, 0.228616
#
#o8
#	all				sub			
#specN
#lin:	164.591782, 0.407497	165.909238, 0.710193
#log:	165.215261, 0.431307	168.222513, 0.767175
#projN
#lin:	158.698163, 0.093699	134.868214, 0.245098
#log:	107.121154, 0.022424	136.649043, 0.054702
#
#o8d
#	all				sub			
#specN
#lin:	154.316278, 0.434182	155.727566, 0.755392
#log:	154.947876, 0.459858	158.163355, 0.815135
#projN
#lin:	140.475124, 0.086091	142.077984, 0.229120
#log:	104.381034, 0.018046	140.249975, 0.054715

#spcm_o7_name = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
#spcm_o8_name = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
# sampledataname = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'   
def comp_EW_dists_o7(hdf5grouptosaveto=None):
    spcmname = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
    sampledataname = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'  
    sampledataiondir = 'file0'
    # assumes full box size = slice length 
    specdata = Spcm(spcmname)
    ion = specdata.ion
    
    simdata = mc.get_simdata_from_outputname(spcmname)
    cosmopars = mc.getcosmopars(simdata['simnum'],simdata['snapnum'],simdata['var'],file_type = 'snap',simulation = simdata['simulation'])
    L_z = cosmopars['boxsize']/cosmopars['h']
    
    # operations on specdata:
    specdata.readcddf('cddf_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_1-x-100.000000slices_range-25.0-28.0_1060bins.npz')
    specdata.get_COG()
    specdata.gethist_fromcddf()

    edgesok_13 = specdata.cog_edges_cd >=13.
    edgesok_14 = specdata.cog_edges_cd >=14.

    logEWfit_all =      94.140913*1e5
    logEWfit_o7sample = 90.388292*1e5
    logEWfit_sub = logEWfit_o7sample
    
    binsedges_dct = {}
    binsedges_dct['lin_13_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_all done')
    binsedges_dct['lin_14_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_all done')
    binsedges_dct['fit_13_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_all done')
    binsedges_dct['fit_14_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0) 
    print('fit_14_all done')

    # get subsample
    samplefile = h5py.File(sampledataname,'r')
    positions = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir)])
    specdata.subsample_match(positions,norm='pix') 
    specdata.get_COG()

    binsedges_dct['lin_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_sub done')
    binsedges_dct['lin_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_sub done')
    binsedges_dct['fit_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_sub done')
    binsedges_dct['fit_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_14_sub done')

    # linear and fitted cog curve edge conversion; use cog edges to get the right edge of the last bin
    tohist = mc.cddf_over_pixcount(cosmopars['z'],L_z,simdata['numpix']**2,specdata.cddf_edges,cosmopars=cosmopars)
    binsedges_dct['lin_cog'] = (specdata.cddf_bins*tohist, np.log10(sp.lingrowthcurve_inv(10**specdata.cog_edges_cd,ion)))
    binsedges_dct['fit_cog_all'] = (specdata.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd,logEWfit_all,ion)))
    binsedges_dct['fit_cog_sub'] = (specdata.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd,logEWfit_sub,ion)))

    ## combine with 6.25Mpc cddf
    specdata_6p25 = Spcm(spcmname)
    simdata = mc.get_simdata_from_outputname(spcmname)
    cosmopars = mc.getcosmopars(simdata['simnum'],simdata['snapnum'],simdata['var'],file_type = 'snap',simulation = simdata['simulation'])
    L_z = cosmopars['boxsize']/cosmopars['h'] # total dX is independent of slice thickness (smaller L_z per pixel, but proportionately more pixels)
    
    specdata_6p25.readcddf('cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    specdata_6p25.get_COG()
    specdata_6p25.gethist_fromcddf()

    edgesok_13 = specdata_6p25.cog_edges_cd >=13.
    edgesok_14 = specdata_6p25.cog_edges_cd >=14.
    
    binsedges_dct['lin_13_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_13[:-1],:], specdata_6p25.cog_edges_cd[edgesok_13], specdata_6p25.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_all done')
    binsedges_dct['lin_14_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_14[:-1],:], specdata_6p25.cog_edges_cd[edgesok_14], specdata_6p25.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_all done')
    binsedges_dct['fit_13_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_13[:-1],:], specdata_6p25.cog_edges_cd[edgesok_13], specdata_6p25.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_all done')
    binsedges_dct['fit_14_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_14[:-1],:], specdata_6p25.cog_edges_cd[edgesok_14], specdata_6p25.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0) 
    print('fit_14_all done')
    
    # linear and fitted cog curve edge conversion; use cog edges to get the right edge of the last bin
    tohist = mc.cddf_over_pixcount(cosmopars['z'], L_z, simdata['numpix']**2, specdata_6p25.cddf_edges, cosmopars=cosmopars)
    binsedges_dct['lin_cog_6p25'] = (specdata_6p25.cddf_bins*tohist, np.log10(sp.lingrowthcurve_inv(10**specdata_6p25.cog_edges_cd, ion)))
    binsedges_dct['fit_cog_all_6p25'] = (specdata_6p25.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata_6p25.cog_edges_cd, logEWfit_all, ion)))
    binsedges_dct['fit_cog_sub_6p25'] = (specdata_6p25.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata_6p25.cog_edges_cd, logEWfit_sub, ion)))
    
    if hdf5grouptosaveto is not None:
        out = hdf5grouptosaveto
        out.attrs.create('file_spcm_used', spcmname)
        out.attrs.create('file_ionselections', sampledataname)
        out.attrs.create('best-fit_b_to_logEW_all_sightlines', logEWfit_all)
        out.attrs.create('best-fit_b_to_logEW_o7-selected_sightlines', logEWfit_sub)
        out.attrs.create('CDDF', 'cddf_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_1-x-100.000000slices_range-25.0-28.0_1060bins.npz')
        out.attrs.create('CDDF_6.25cMpc', 'cddf_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
        out.attrs.create('info', 'EW histograms: stored as bins [number of sightlines, total = number of pixels in col. dens. map with non-zero column density], edges [rest-frame A]')
        group_temp = out.create_group('cosmopars')
        for key in cosmopars:
            group_temp.attrs.create(key, cosmopars[key])
        group_temp = out.create_group('simdata')
        for key in simdata:
            group_temp.attrs.create(key, simdata[key])
        storekeys_dct = {'lin_13_all': 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'lin_14_all': 'linear_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'fit_13_all': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'fit_14_all': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'lin_13_sub': 'linear_cog_extrapolation_below_10^13cm^-2_o7-selected_sightlines',\
                         'lin_14_sub': 'linear_cog_extrapolation_below_10^14cm^-2_o7-selected_sightlines',\
                         'fit_13_sub': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_o7-selected_sightlines',\
                         'fit_14_sub': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_o7-selected_sightlines',\
                         'lin_cog':    'linear_cog_at_all_column_densities',\
                         'fit_cog_all': 'best-fit-b_all_sightlines_at_all_column_densities',\
                         'fit_cog_sub': 'best-fit-b_o7-selected_sightlines_at_all_column_densities',\
                         'lin_13_all_6p25': 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'lin_14_all_6p25': 'linear_cog_extrapolation_below_10^14cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'fit_13_all_6p25': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'fit_14_all_6p25': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'lin_cog_6p25':    'linear_cog_at_all_column_densities_with_6.25cMpc_CDDF',\
                         'fit_cog_all_6p25': 'best-fit-b_all_sightlines_at_all_column_densities_with_6.25cMpc_CDDF',\
                         'fit_cog_sub_6p25': 'best-fit-b_o7-selected_sightlines_at_all_column_densities_with_6.25cMpc_CDDF',\
                         }
        for key in binsedges_dct.keys():
            group_temp = out.create_group(storekeys_dct[key])
            group_temp.create_dataset('bins', data=binsedges_dct[key][0])
            group_temp['bins'].attrs.create('units', 'histogram counts in bin')
            group_temp.create_dataset('edges', data=binsedges_dct[key][1])
            group_temp['edges'].attrs.create('units', 'log10 rest-frame A')
    return binsedges_dct, cosmopars, simdata

def comp_EW_dists_o7_snap19_26(openhdf5file, snapshot=19):
    
    spcmname = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
    sampledataname = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'  
    sampledataiondir = 'file0'
    if snapshot == 19:
        cddfname = 'cddf_coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
    elif snapshot == 26:
        cddfname = 'cddf_coldens_o7_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'    

    # assumes full box size = slice length 
    specdata = Spcm(spcmname)
    ion = specdata.ion
    
    simdata = mc.get_simdata_from_outputname(cddfname)
    try:
        cosmopars = mc.getcosmopars(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
        L_z = cosmopars['boxsize'] / cosmopars['h']
    except: # something failed; just copy parameters from the EAGLE wiki and constants & units
        L_z = 100.
        cosmopars = {}
        if snapshot == 19:
            cosmopars['a'] = 0.498972
        elif snapshot == 26:
            cosmopars['a'] = 0.845516 
        cosmopars['z'] = 1. / cosmopars['a'] - 1.
        cosmopars['h'] = c.hubbleparam
        cosmopars['omegam'] = c.omega0
        cosmopars['omegab'] = c.omegabaryon
        cosmopars['omegalambda'] = c.omegalambda
        cosmopars['boxsize'] = L_z * cosmopars['h']
        
    # operations on specdata:
    specdata.readcddf(cddfname)
    specdata.get_COG()
    specdata.gethist_fromcddf(cosmopars=cosmopars)

    edgesok_13 = specdata.cog_edges_cd >=13.
    edgesok_14 = specdata.cog_edges_cd >=14.

    logEWfit_all =      94.140913*1e5
    logEWfit_o7sample = 90.388292*1e5
    logEWfit_sub = logEWfit_o7sample
    
    binsedges_dct = {}
    binsedges_dct['lin_13_all'] = convert_hist_to_hist(specdata.hist, specdata.cog_edges_cd, specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_all done')
    binsedges_dct['lin_14_all'] = convert_hist_to_hist(specdata.hist, specdata.cog_edges_cd, specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_all done')
    binsedges_dct['fit_13_all'] = convert_hist_to_hist(specdata.hist, specdata.cog_edges_cd, specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_all done')
    binsedges_dct['fit_14_all'] = convert_hist_to_hist(specdata.hist, specdata.cog_edges_cd, specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0) 
    print('fit_14_all done')

    # get subsample
    samplefile = h5py.File(sampledataname, 'r')
    positions = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir)])
    specdata.subsample_match(positions, norm='pix') 
    specdata.get_COG()

    binsedges_dct['lin_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_sub done')
    binsedges_dct['lin_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_sub done')
    binsedges_dct['fit_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_sub done')
    binsedges_dct['fit_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_14_sub done')

    # linear and fitted cog curve edge conversion; use cog edges to get the right edge of the last bin
    tohist = mc.cddf_over_pixcount(cosmopars['z'], L_z,simdata['numpix']**2, specdata.cddf_edges, cosmopars=cosmopars)
    binsedges_dct['lin_cog'] = (specdata.cddf_bins * tohist, np.log10(sp.lingrowthcurve_inv(10**specdata.cog_edges_cd, ion)))
    binsedges_dct['fit_cog_all'] = (specdata.cddf_bins * tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd, logEWfit_all, ion)))
    binsedges_dct['fit_cog_sub'] = (specdata.cddf_bins * tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd, logEWfit_sub, ion)))

    if openhdf5file is not None:
        out = openhdf5file.create_group('EWdists_o7_snap%s_snap27-N-EWconv'%snapshot)
        out.attrs.create('file_spcm_used', spcmname)
        out.attrs.create('file_ionselections', sampledataname)
        out.attrs.create('best-fit_b_to_logEW_all_sightlines', logEWfit_all)
        out.attrs.create('best-fit_b_to_logEW_o7-selected_sightlines', logEWfit_sub)
        out.attrs.create('CDDF', cddfname)
        out.attrs.create('info', 'EW histograms: stored as bins [number of sightlines, total = number of pixels in col. dens. map with non-zero column density], edges [rest-frame A]')
        group_temp = out.create_group('cosmopars')
        for key in cosmopars:
            group_temp.attrs.create(key, cosmopars[key])
        group_temp = out.create_group('simdata')
        for key in simdata:
            group_temp.attrs.create(key, simdata[key])
        storekeys_dct = {'lin_13_all': 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'lin_14_all': 'linear_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'fit_13_all': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'fit_14_all': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'lin_13_sub': 'linear_cog_extrapolation_below_10^13cm^-2_o7-selected_sightlines',\
                         'lin_14_sub': 'linear_cog_extrapolation_below_10^14cm^-2_o7-selected_sightlines',\
                         'fit_13_sub': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_o7-selected_sightlines',\
                         'fit_14_sub': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_o7-selected_sightlines',\
                         'lin_cog':    'linear_cog_at_all_column_densities',\
                         'fit_cog_all': 'best-fit-b_all_sightlines_at_all_column_densities',\
                         'fit_cog_sub': 'best-fit-b_o7-selected_sightlines_at_all_column_densities',\
                         }
        for key in binsedges_dct.keys():
            group_temp = out.create_group(storekeys_dct[key])
            group_temp.create_dataset('bins', data=binsedges_dct[key][0])
            group_temp['bins'].attrs.create('units', 'histogram counts in bin')
            group_temp.create_dataset('edges', data=binsedges_dct[key][1])
            group_temp['edges'].attrs.create('units', 'log10 rest-frame A')
    return binsedges_dct, cosmopars, simdata

def comp_EW_dists_o8(hdf5grouptosaveto=None):
    spcmname = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
    sampledataname = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'  
    sampledataiondir = 'file1'
    # assumes full box size = slice length 
    specdata = Spcm(spcmname)
    ion = specdata.ion
    
    simdata = mc.get_simdata_from_outputname(spcmname)
    cosmopars = mc.getcosmopars(simdata['simnum'],simdata['snapnum'],simdata['var'],file_type = 'snap',simulation = simdata['simulation'])
    L_z = cosmopars['boxsize']/cosmopars['h']
    
    # operations on specdata: 
    specdata.readcddf('cddf_coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_1-x-100.000000slices_range-25.0-28.0_1060bins.npz')
    specdata.get_COG()
    specdata.gethist_fromcddf()

    edgesok_13 = specdata.cog_edges_cd >=13.
    edgesok_14 = specdata.cog_edges_cd >=14.

    ## doublet o8 values
    logEWfit_all =      154.947876*1e5
    logEWfit_o8sample = 157.577218*1e5
    logEWfit_sub = logEWfit_o8sample
    
    binsedges_dct = {}
    binsedges_dct['lin_13_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_all done')
    binsedges_dct['lin_14_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_all done')
    binsedges_dct['fit_13_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_all done')
    binsedges_dct['fit_14_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0) 
    print('fit_14_all done')

    # get subsample
    samplefile = h5py.File(sampledataname,'r')
    positions = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir)])
    specdata.subsample_match(positions,norm='pix')
    specdata.get_COG() 

    binsedges_dct['lin_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_sub done')
    binsedges_dct['lin_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_sub done')
    binsedges_dct['fit_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_sub done')
    binsedges_dct['fit_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_14_sub done')

    # linear and fitted cog curve edge conversion
    tohist = mc.cddf_over_pixcount(cosmopars['z'],L_z,simdata['numpix']**2,specdata.cddf_edges,cosmopars=cosmopars)
    binsedges_dct['lin_cog'] = (specdata.cddf_bins*tohist, np.log10(sp.lingrowthcurve_inv(10**specdata.cog_edges_cd,ion)))
    binsedges_dct['fit_cog_all'] = (specdata.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd,logEWfit_all,ion)))
    binsedges_dct['fit_cog_sub'] = (specdata.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd,logEWfit_sub,ion)))

    ## combine with 6.25Mpc cddf
    specdata_6p25 = Spcm(spcmname)
    simdata = mc.get_simdata_from_outputname(spcmname)
    cosmopars = mc.getcosmopars(simdata['simnum'],simdata['snapnum'],simdata['var'],file_type = 'snap',simulation = simdata['simulation'])
    L_z = cosmopars['boxsize']/cosmopars['h'] # total dX is independent of slice thickness (smaller L_z per pixel, but proportionately more pixels)
    
    specdata_6p25.readcddf('cddf_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
    specdata_6p25.get_COG()
    specdata_6p25.gethist_fromcddf()

    edgesok_13 = specdata_6p25.cog_edges_cd >=13.
    edgesok_14 = specdata_6p25.cog_edges_cd >=14.
    
    binsedges_dct['lin_13_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_13[:-1],:], specdata_6p25.cog_edges_cd[edgesok_13], specdata_6p25.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_all done')
    binsedges_dct['lin_14_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_14[:-1],:], specdata_6p25.cog_edges_cd[edgesok_14], specdata_6p25.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_all done')
    binsedges_dct['fit_13_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_13[:-1],:], specdata_6p25.cog_edges_cd[edgesok_13], specdata_6p25.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_all done')
    binsedges_dct['fit_14_all_6p25'] = convert_hist_to_hist(specdata_6p25.hist,specdata_6p25.cog_edges_cd,specdata_6p25.cog[edgesok_14[:-1],:], specdata_6p25.cog_edges_cd[edgesok_14], specdata_6p25.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0) 
    print('fit_14_all done')
    
    # linear and fitted cog curve edge conversion; use cog edges to get the right edge of the last bin
    tohist = mc.cddf_over_pixcount(cosmopars['z'], L_z, simdata['numpix']**2, specdata_6p25.cddf_edges, cosmopars=cosmopars)
    binsedges_dct['lin_cog_6p25'] = (specdata_6p25.cddf_bins*tohist, np.log10(sp.lingrowthcurve_inv(10**specdata_6p25.cog_edges_cd, ion)))
    binsedges_dct['fit_cog_all_6p25'] = (specdata_6p25.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata_6p25.cog_edges_cd, logEWfit_all, ion)))
    binsedges_dct['fit_cog_sub_6p25'] = (specdata_6p25.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata_6p25.cog_edges_cd, logEWfit_sub, ion)))
    
    if hdf5grouptosaveto is not None:
        out = hdf5grouptosaveto
        out.attrs.create('file_spcm_used', spcmname)
        out.attrs.create('file_ionselections', sampledataname)
        out.attrs.create('best-fit_b_to_logEW_all_sightlines', logEWfit_all)
        out.attrs.create('best-fit_b_to_logEW_o8-selected_sightlines', logEWfit_sub)
        out.attrs.create('CDDF', 'cddf_coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_1-x-100.000000slices_range-25.0-28.0_1060bins.npz')
        out.attrs.create('CDDF_6.25cMpc', 'cddf_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz')
        out.attrs.create('info', 'EW histograms: stored as bins [number of sightlines, total = number of pixels in col. dens. map with non-zero column density], edges [rest-frame A]')
        group_temp = out.create_group('cosmopars')
        for key in cosmopars:
            group_temp.attrs.create(key, cosmopars[key])
        group_temp = out.create_group('simdata')
        for key in simdata:
            group_temp.attrs.create(key, simdata[key])
        storekeys_dct = {'lin_13_all': 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'lin_14_all': 'linear_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'fit_13_all': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'fit_14_all': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'lin_13_sub': 'linear_cog_extrapolation_below_10^13cm^-2_o8-selected_sightlines',\
                         'lin_14_sub': 'linear_cog_extrapolation_below_10^14cm^-2_o8-selected_sightlines',\
                         'fit_13_sub': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_o8-selected_sightlines',\
                         'fit_14_sub': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_o8-selected_sightlines',\
                         'lin_cog':    'linear_cog_at_all_column_densities',\
                         'fit_cog_all': 'best-fit-b_all_sightlines_at_all_column_densities',\
                         'fit_cog_sub': 'best-fit-b_o8-selected_sightlines_at_all_column_densities',\
                         'lin_13_all_6p25': 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'lin_14_all_6p25': 'linear_cog_extrapolation_below_10^14cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'fit_13_all_6p25': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'fit_14_all_6p25': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_all_sightlines_with_6.25cMpc_CDDF',\
                         'lin_cog_6p25':    'linear_cog_at_all_column_densities_with_6.25cMpc_CDDF',\
                         'fit_cog_all_6p25': 'best-fit-b_all_sightlines_at_all_column_densities_with_6.25cMpc_CDDF',\
                         'fit_cog_sub_6p25': 'best-fit-b_o8-selected_sightlines_at_all_column_densities_with_6.25cMpc_CDDF',\
                         }
        for key in binsedges_dct.keys():
            group_temp = out.create_group(storekeys_dct[key])
            group_temp.create_dataset('bins', data=binsedges_dct[key][0])
            group_temp['bins'].attrs.create('units', 'histogram counts in bin')
            group_temp.create_dataset('edges', data=binsedges_dct[key][1])
            group_temp['edges'].attrs.create('units', 'log10 rest-frame A')
    
    return binsedges_dct, cosmopars, simdata

def comp_EW_dists_o8_snap19_26(openhdf5file, snapshot=19):
    spcmname = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
    sampledataname = '/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5'  
    sampledataiondir = 'file1'
    if snapshot == 19:
        cddfname = 'cddf_coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
    elif snapshot == 26:
        cddfname = 'cddf_coldens_o8_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'    

    # assumes full box size = slice length 
    specdata = Spcm(spcmname)
    ion = specdata.ion
    
    simdata = mc.get_simdata_from_outputname(cddfname)
    try:
        cosmopars = mc.getcosmopars(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
        L_z = cosmopars['boxsize'] / cosmopars['h']
    except: # something failed; just copy parameters from the EAGLE wiki and constants & units
        L_z = 100.
        cosmopars = {}
        if snapshot == 19:
            cosmopars['a'] = 0.498972
        elif snapshot == 26:
            cosmopars['a'] = 0.845516 
        cosmopars['z'] = 1. / cosmopars['a'] - 1.
        cosmopars['h'] = c.hubbleparam
        cosmopars['omegam'] = c.omega0
        cosmopars['omegab'] = c.omegabaryon
        cosmopars['omegalambda'] = c.omegalambda
        cosmopars['boxsize'] = L_z * cosmopars['h']
    
    # operations on specdata:
    specdata.readcddf(cddfname)
    specdata.get_COG()
    specdata.gethist_fromcddf(cosmopars=cosmopars)

    edgesok_13 = specdata.cog_edges_cd >=13.
    edgesok_14 = specdata.cog_edges_cd >=14.

    ## doublet o8 values
    logEWfit_all =      154.947876*1e5
    logEWfit_o8sample = 157.577218*1e5
    logEWfit_sub = logEWfit_o8sample
    
    binsedges_dct = {}
    binsedges_dct['lin_13_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_all done')
    binsedges_dct['lin_14_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_all done')
    binsedges_dct['fit_13_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_all done')
    binsedges_dct['fit_14_all'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_all), save=None, ratio = False, overlap_interpolation_method = 0) 
    print('fit_14_all done')

    # get subsample
    samplefile = h5py.File(sampledataname,'r')
    positions = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir)])
    specdata.subsample_match(positions,norm='pix')
    specdata.get_COG() 

    binsedges_dct['lin_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_13_sub done')
    binsedges_dct['lin_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'lincog_%s_log'%(ion), save=None, ratio = False, overlap_interpolation_method = 0)
    print('lin_14_sub done')
    binsedges_dct['fit_13_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_13[:-1],:], specdata.cog_edges_cd[edgesok_13], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_13_sub done')
    binsedges_dct['fit_14_sub'] = convert_hist_to_hist(specdata.hist,specdata.cog_edges_cd,specdata.cog[edgesok_14[:-1],:], specdata.cog_edges_cd[edgesok_14], specdata.cog_edges_ew, extend_lower = 'fitcog_%s_%f_log'%(ion,logEWfit_sub), save=None, ratio = False, overlap_interpolation_method = 0)
    print('fit_14_sub done')

    # linear and fitted cog curve edge conversion
    tohist = mc.cddf_over_pixcount(cosmopars['z'],L_z,simdata['numpix']**2,specdata.cddf_edges,cosmopars=cosmopars)
    binsedges_dct['lin_cog'] = (specdata.cddf_bins*tohist, np.log10(sp.lingrowthcurve_inv(10**specdata.cog_edges_cd,ion)))
    binsedges_dct['fit_cog_all'] = (specdata.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd,logEWfit_all,ion)))
    binsedges_dct['fit_cog_sub'] = (specdata.cddf_bins*tohist, np.log10(sp.linflatcurveofgrowth_inv(10**specdata.cog_edges_cd,logEWfit_sub,ion)))

    
    if openhdf5file is not None:
        out = openhdf5file.create_group('EWdists_o8_snap%s_snap27-N-EWconv'%snapshot)
        out.attrs.create('file_spcm_used', spcmname)
        out.attrs.create('file_ionselections', sampledataname)
        out.attrs.create('best-fit_b_to_logEW_all_sightlines', logEWfit_all)
        out.attrs.create('best-fit_b_to_logEW_o8-selected_sightlines', logEWfit_sub)
        out.attrs.create('CDDF', cddfname)
        out.attrs.create('info', 'EW histograms: stored as bins [number of sightlines, total = number of pixels in col. dens. map with non-zero column density], edges [rest-frame A]')
        group_temp = out.create_group('cosmopars')
        for key in cosmopars:
            group_temp.attrs.create(key, cosmopars[key])
        group_temp = out.create_group('simdata')
        for key in simdata:
            group_temp.attrs.create(key, simdata[key])
        storekeys_dct = {'lin_13_all': 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'lin_14_all': 'linear_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'fit_13_all': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_all_sightlines',\
                         'fit_14_all': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_all_sightlines',\
                         'lin_13_sub': 'linear_cog_extrapolation_below_10^13cm^-2_o8-selected_sightlines',\
                         'lin_14_sub': 'linear_cog_extrapolation_below_10^14cm^-2_o8-selected_sightlines',\
                         'fit_13_sub': 'best-fit-b_cog_extrapolation_below_10^13cm^-2_o8-selected_sightlines',\
                         'fit_14_sub': 'best-fit-b_cog_extrapolation_below_10^14cm^-2_o8-selected_sightlines',\
                         'lin_cog':    'linear_cog_at_all_column_densities',\
                         'fit_cog_all': 'best-fit-b_all_sightlines_at_all_column_densities',\
                         'fit_cog_sub': 'best-fit-b_o8-selected_sightlines_at_all_column_densities',\
                         }
        for key in binsedges_dct.keys():
            group_temp = out.create_group(storekeys_dct[key])
            group_temp.create_dataset('bins', data=binsedges_dct[key][0])
            group_temp['bins'].attrs.create('units', 'histogram counts in bin')
            group_temp.create_dataset('edges', data=binsedges_dct[key][1])
            group_temp['edges'].attrs.create('units', 'log10 rest-frame A')
    
    return binsedges_dct, cosmopars, simdata



def comp_EW_dists_snap19spcm(openhdf5file, ion):
    
    if ion == 'o7':
        spcmname = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3-verf_snap19_snap_019_z001p004.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/los_sample3-verf_o7-o8_L0100N1504_snap19_data.hdf5'  
        sampledataiondir = 'file0'

        cddfname = 'cddf_coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
   
    elif ion == 'o8':
        spcmname = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3-verf_snap19_snap_019_z001p004.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/los_sample3-verf_o7-o8_L0100N1504_snap19_data.hdf5'  
        sampledataiondir = 'file1'

        cddfname = 'cddf_coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'

    # assumes full box size = slice length 
    specdata = Spcm(spcmname)
    ion = specdata.ion
    
    simdata = mc.get_simdata_from_outputname(cddfname)
    try:
        cosmopars = mc.getcosmopars(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
        L_z = cosmopars['boxsize'] / cosmopars['h']
    except: # something failed; just copy parameters from the EAGLE wiki and constants & units
        L_z = 100.
        cosmopars = {}
        #snapshot 19
        cosmopars['a'] = 0.498972
        cosmopars['z'] = 1. / cosmopars['a'] - 1.
        cosmopars['h'] = c.hubbleparam
        cosmopars['omegam'] = c.omega0
        cosmopars['omegab'] = c.omegabaryon
        cosmopars['omegalambda'] = c.omegalambda
        cosmopars['boxsize'] = L_z * cosmopars['h']
        
    # operations on specdata:
    specdata.readcddf(cddfname)
    specdata.get_COG()
    specdata.gethist_fromcddf(cosmopars=cosmopars)

    if ion == 'o7':
        minN = 15.0 
    elif ion == 'o8':
        minN = 15.5
    
    # rebin histograms from 0.05 to 0.1 dex bins (hopefully reduces binning noise)
    Nhist = np.copy(specdata.hist)
    Nedges = np.copy(specdata.cog_edges_cd)
    edgesok = Nedges >= minN
    cog = np.copy(specdata.cog)

    binsedges_dct = {}
    errests_dct = {}
    binsedges_dct['lin_%s_all'%minN] = convert_hist_to_hist(Nhist, Nedges,\
                                                            cog[edgesok[:-1],:], Nedges[edgesok], specdata.cog_edges_ew,\
                                                            extend_lower='lincog_%s_log'%(ion), save=None, ratio=False, overlap_interpolation_method=0)
    print('lin_%s_all done'%minN)
    
    # estimated errors from conversion matrix noise =   sqrt sum_{N bins} ((conversion matrix noise) * (N counts))**2
    # matrix noise = sqrt(counts) / normalisation for N (ignoring noi)
    errest = np.sqrt( np.sum( cog[edgesok[:-1], :] / np.sum(cog[edgesok[:-1],:], axis=0)[np.newaxis, :]**2 * Nhist[edgesok[:-1], np.newaxis]**2, axis=0))
    errests_dct['lin_%s_all'%minN] = errest

    # get subsample
    samplefile = h5py.File(sampledataname, 'r')
    positions = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir)])
    specdata.subsample_match(positions, norm='pix') 
    specdata.get_COG()
    
    Nhist = np.copy(specdata.hist)
    Nedges = np.copy(specdata.cog_edges_cd)
    #rebin_factor = 2
    edgesok = Nedges >= minN
    cog = np.copy(specdata.cog)

    binsedges_dct['lin_%s_sub'%minN] = convert_hist_to_hist(Nhist, Nedges,\
                                                            specdata.cog[edgesok[:-1],:], Nedges[edgesok], specdata.cog_edges_ew,\
                                                            extend_lower='lincog_%s_log'%(ion), save=None, ratio=False, overlap_interpolation_method=0)
    
    errest = np.sqrt( np.sum( cog[edgesok[:-1], :] /np.sum(cog[edgesok[:-1],:], axis=0)[np.newaxis, :]**2 * Nhist[edgesok[:-1], np.newaxis]**2, axis=0))
    errests_dct['lin_%s_sub'%minN] = errest

    # linear and fitted cog curve edge conversion; use cog edges to get the right edge of the last bin
    tohist = mc.cddf_over_pixcount(cosmopars['z'], L_z,simdata['numpix']**2, specdata.cddf_edges, cosmopars=cosmopars)
    binsedges_dct['lin_cog'] = (specdata.cddf_bins * tohist, np.log10(sp.lingrowthcurve_inv(10**specdata.cog_edges_cd, ion)))


    if openhdf5file is not None:
        out = openhdf5file.create_group('EWdists_%s_snap19_snap19-256sl-N-EWconv'%ion)
        out.attrs.create('file_spcm_used', spcmname)
        out.attrs.create('file_ionselections', sampledataname)
        out.attrs.create('CDDF', cddfname)
        out.attrs.create('info', 'EW histograms: stored as bins [number of sightlines, total = number of pixels in col. dens. map with non-zero column density], edges [rest-frame A]')
        group_temp = out.create_group('cosmopars')
        for key in cosmopars:
            group_temp.attrs.create(key, cosmopars[key])
        group_temp = out.create_group('simdata')
        for key in simdata:
            group_temp.attrs.create(key, simdata[key])
        storekeys_dct = {'lin_%s_all'%minN: 'linear_cog_extrapolation_below_10^%scm^-2_all_sightlines'%minN,\
                         'lin_%s_sub'%minN: 'linear_cog_extrapolation_below_10^%scm^-2_%s-selected_sightlines'%(minN, ion),\
                         'lin_cog':    'linear_cog_at_all_column_densities',\
                         }
        for key in binsedges_dct.keys():
            group_temp = out.create_group(storekeys_dct[key])
            group_temp.create_dataset('bins', data=binsedges_dct[key][0])
            group_temp['bins'].attrs.create('units', 'histogram counts in bin')
            group_temp.create_dataset('edges', data=binsedges_dct[key][1])
            group_temp['edges'].attrs.create('units', 'log10 rest-frame A')
            if key != 'lin_cog':
                group_temp.create_dataset('matrix_err', data=errests_dct[key])
    return binsedges_dct, errests_dct, cosmopars, simdata


def comp_EW_dists_fullspcms(openhdf5file, ion, snap):
    '''
    implemented for snapshots 19, 26
    '''
    
    if ion == 'o7' and snap == 19:
        spcmname = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample4_snap_019_z001p004.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/sample4/los_sample4_o7-o8_L0100N1504_snap19_data.hdf5'  
        sampledataiondir = 'file0'

        cddfname = 'cddf_coldens_o7_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
   
    elif ion == 'o8' and snap == 19:
        spcmname = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample4_snap_019_z001p004.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/sample4/los_sample4_o7-o8_L0100N1504_snap19_data.hdf5'  
        sampledataiondir = 'file1'

        cddfname = 'cddf_coldens_o8_L0100N1504_19_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'

    elif ion == 'o7' and snap == 26:
        spcmname = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample5_snap_026_z000p183.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/sample5/los_sample5_o7-o8_L0100N1504_snap26_data.hdf5'  
        sampledataiondir = 'file0'

        cddfname = 'cddf_coldens_o7_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
   
    elif ion == 'o8' and snap == 26:
        spcmname = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample5_snap_026_z000p183.0.npz'
        sampledataname = '/net/luttero/data2/specwizard_data/sample5/los_sample5_o7-o8_L0100N1504_snap26_data.hdf5'  
        sampledataiondir = 'file1'

        cddfname = 'cddf_coldens_o8_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_16-x-6.250000slices_range-25.0-28.0_1060bins.npz'
    
    else:
        raise ValueError('conversion for ion %s, snapshot %i is not implemented, posssibly because the data is unavailable'%(ion, snap)) 

    # assumes full box size = slice length 
    specdata = Spcm(spcmname)
    if ion != specdata.ion:
        raise RuntimeError('SpCm ion %s did not match input ion %s'%(ion, specdata.ion))
    ion = specdata.ion
    
    simdata = mc.get_simdata_from_outputname(cddfname)
    try:
        cosmopars = mc.getcosmopars(simdata['simnum'], simdata['snapnum'], simdata['var'], file_type='snap', simulation=simdata['simulation'])
        L_z = cosmopars['boxsize'] / cosmopars['h']
    except: # something failed; just copy parameters from the EAGLE wiki and constants & units
        # typically, due to snapshot not being on the system being used
        cosmopars = {}
        cosmopars['h'] = c.hubbleparam
        cosmopars['omegam'] = c.omega0
        cosmopars['omegab'] = c.omegabaryon
        cosmopars['omegalambda'] = c.omegalambda
        L_z = 100.
        cosmopars['boxsize'] = L_z * cosmopars['h']
        if snap == 19: 
            cosmopars['a'] = 0.498972
        elif snap == 26:
            cosmopars['a'] = 0.845516 
        cosmopars['z'] = 1. / cosmopars['a'] - 1.
        
        
    # operations on specdata:
    specdata.readcddf(cddfname)
    specdata.get_COG()
    specdata.gethist_fromcddf(cosmopars=cosmopars)

    if ion == 'o7':
        minN = 13.0 
    elif ion == 'o8':
        minN = 13.0
    
    # rebin histograms from 0.05 to 0.1 dex bins (hopefully reduces binning noise)
    Nhist = np.copy(specdata.hist)
    Nedges = np.copy(specdata.cog_edges_cd)
    edgesok = Nedges >= minN
    cog = np.copy(specdata.cog)

    binsedges_dct = {}
    #errests_dct = {}
    binsedges_dct['lin_%s_all'%minN] = convert_hist_to_hist(Nhist, Nedges,\
                                                            cog[edgesok[:-1],:], Nedges[edgesok], specdata.cog_edges_ew,\
                                                            extend_lower='lincog_%s_log'%(ion), save=None, ratio=False, overlap_interpolation_method=0)
    print('lin_%s_all done'%minN)
    
    # estimated errors from conversion matrix noise =   sqrt sum_{N bins} ((conversion matrix noise) * (N counts))**2
    # matrix noise = sqrt(counts) / normalisation for N (ignoring noi)
    #errest = np.sqrt( np.sum( cog[edgesok[:-1], :] / np.sum(cog[edgesok[:-1],:], axis=0)[np.newaxis, :]**2 * Nhist[edgesok[:-1], np.newaxis]**2, axis=0))
    #errests_dct['lin_%s_all'%minN] = errest

    # get subsample
    samplefile = h5py.File(sampledataname, 'r')
    positions = np.array(samplefile['Selection/%s/selected_pixels_thision'%(sampledataiondir)])
    specdata.subsample_match(positions, norm='pix', cosmopars=cosmopars) 
    specdata.get_COG()
    
    Nhist = np.copy(specdata.hist)
    Nedges = np.copy(specdata.cog_edges_cd)
    #rebin_factor = 2
    edgesok = Nedges >= minN
    cog = np.copy(specdata.cog)

    binsedges_dct['lin_%s_sub'%minN] = convert_hist_to_hist(Nhist, Nedges,\
                                                            specdata.cog[edgesok[:-1],:], Nedges[edgesok], specdata.cog_edges_ew,\
                                                            extend_lower='lincog_%s_log'%(ion), save=None, ratio=False, overlap_interpolation_method=0)
    
    #errest = np.sqrt( np.sum( cog[edgesok[:-1], :] /np.sum(cog[edgesok[:-1],:], axis=0)[np.newaxis, :]**2 * Nhist[edgesok[:-1], np.newaxis]**2, axis=0))
    #errests_dct['lin_%s_sub'%minN] = errest

    # linear and fitted cog curve edge conversion; use cog edges to get the right edge of the last bin
    tohist = mc.cddf_over_pixcount(cosmopars['z'], L_z,simdata['numpix']**2, specdata.cddf_edges, cosmopars=cosmopars)
    binsedges_dct['lin_cog'] = (specdata.cddf_bins * tohist, np.log10(sp.lingrowthcurve_inv(10**specdata.cog_edges_cd, ion)))


    if openhdf5file is not None:
        out = openhdf5file.create_group('EWdists_%s_snap%i'%(ion, snap))
        out.attrs.create('file_spcm_used', spcmname)
        out.attrs.create('file_ionselections', sampledataname)
        out.attrs.create('CDDF', cddfname)
        out.attrs.create('info', 'EW histograms: stored as bins [number of sightlines, total = number of pixels in col. dens. map with non-zero column density], edges [rest-frame A]')
        group_temp = out.create_group('cosmopars')
        for key in cosmopars:
            group_temp.attrs.create(key, cosmopars[key])
        group_temp = out.create_group('simdata')
        for key in simdata:
            group_temp.attrs.create(key, simdata[key])
        storekeys_dct = {'lin_%s_all'%minN: 'linear_cog_extrapolation_below_10^%scm^-2_all_sightlines'%minN,\
                         'lin_%s_sub'%minN: 'linear_cog_extrapolation_below_10^%scm^-2_%s-selected_sightlines'%(minN, ion),\
                         'lin_cog':    'linear_cog_at_all_column_densities',\
                         }
        for key in binsedges_dct.keys():
            group_temp = out.create_group(storekeys_dct[key])
            group_temp.create_dataset('bins', data=binsedges_dct[key][0])
            group_temp['bins'].attrs.create('units', 'histogram counts in bin')
            group_temp.create_dataset('edges', data=binsedges_dct[key][1])
            group_temp['edges'].attrs.create('units', 'log10 rest-frame A')
            #if key != 'lin_cog':
            #    group_temp.create_dataset('matrix_err', data=errests_dct[key])
    return binsedges_dct, cosmopars, simdata

def binsedges_to_dXcumul(dct, cosmopars,simdata):
    return {(np.cumsum(bins[::-1])[::-1]/mc.getdX(cosmopars['z'],cosmopars['boxsize']/cosmopars['h'],cosmopars=cosmopars), edges)/(simdata['numpix']**2) for (bins, edges) in dct}

def strformat_latextable(val):
    if np.round(val, 1) >= 10.:
        out = '%.1f'%val
    elif np.round(val, 2) >= 1.:
        out = '%.2f'%val
    elif np.round(val, 3) >= 0.1:
        out = '%.3f'%val
    else:
        out = '%.4f'%val 
    return out

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

def getEWdistvals_o78(): # just to store these lines, really
    '''
    prints LaTeX table of these values
    '''
    binsedges_o8, cosmopars_o8, simdata_o8 = comp_EW_dists_o8()
    binsedges_o7, cosmopars_o7, simdata_o7 = comp_EW_dists_o7() 

    dXtot_o8 = float(simdata_o8['numpix'])**2*mc.getdX(cosmopars_o8['z'],cosmopars_o8['boxsize']/cosmopars_o8['h'],cosmopars=cosmopars_o8)
    dXtot_o7 = float(simdata_o7['numpix'])**2*mc.getdX(cosmopars_o7['z'],cosmopars_o7['boxsize']/cosmopars_o7['h'],cosmopars=cosmopars_o7)

    #binsedges_o8_proc = [[binsedges_o8[key][0]/np.diff(binsedges_o8[key][1])/dXtot_o8, binsedges_o8[key][1][:-1]] for key in ['lin_13_all_6p25']]  #'lin_13_all', 'lin_13_sub', 
    #binsedges_o7_proc = [[binsedges_o7[key][0]/np.diff(binsedges_o7[key][1])/dXtot_o7, binsedges_o7[key][1][:-1]]for key in ['lin_13_all_6p25']]
    
    binsedges_o8_proc_cumuldX = [[np.cumsum(binsedges_o8[key][0][::-1])[::-1]/dXtot_o8, binsedges_o8[key][1][:-1]] for key in ['lin_13_all_6p25']] 
    binsedges_o7_proc_cumuldX = [[np.cumsum(binsedges_o7[key][0][::-1])[::-1]/dXtot_o7, binsedges_o7[key][1][:-1]] for key in ['lin_13_all_6p25']] 

    #bec_o8_dX_all = binsedges_o8_proc_cumuldX[0]
    #bec_o8_dX_sub = binsedges_o8_proc_cumuldX[1]
    bec_o8_dX_all6 = binsedges_o8_proc_cumuldX[-1]
    #bec_o7_dX_all = binsedges_o7_proc_cumuldX[0]
    #bec_o7_dX_sub = binsedges_o7_proc_cumuldX[1]
    bec_o7_dX_all6 = binsedges_o7_proc_cumuldX[-1]
    
    plt.plot(bec_o7_dX_all6[1] + 3., bec_o7_dX_all6[0], color='black', linestyle='dashed', label='old version O7 z=0.1 6.25 cMpc')
    plt.plot(bec_o8_dX_all6[1] + 3., bec_o8_dX_all6[0], color='gray',  linestyle='dashed', label='old version O8 z=0.1 6.25 cMpc')
    
    # remove leading EW=0 entries, and a strange zero in O7 and O8 bin diffs that might indicate a bug, but occurs at low EWs, so I'm just ignoring it for now
    #bec_o8_dX_all = [bec_o8_dX_all[0][np.isfinite(bec_o8_dX_all[1])][9:], bec_o8_dX_all[1][np.isfinite(bec_o8_dX_all[1])][9:]]
    #bec_o8_dX_sub = [bec_o8_dX_sub[0][np.isfinite(bec_o8_dX_sub[1])][9:], bec_o8_dX_sub[1][np.isfinite(bec_o8_dX_sub[1])][9:]]
    #bec_o8_dX_all6 = [bec_o8_dX_all6[0][np.isfinite(bec_o8_dX_all6[1])][9:], bec_o8_dX_all6[1][np.isfinite(bec_o8_dX_all6[1])][9:]]
    #bec_o7_dX_all = [bec_o7_dX_all[0][np.isfinite(bec_o7_dX_all[1])][7:], bec_o7_dX_all[1][np.isfinite(bec_o7_dX_all[1])][7:]]
    #bec_o7_dX_sub = [bec_o7_dX_sub[0][np.isfinite(bec_o7_dX_sub[1])][7:], bec_o7_dX_sub[1][np.isfinite(bec_o7_dX_sub[1])][7:]]
    #bec_o7_dX_all6 = [bec_o7_dX_all6[0][np.isfinite(bec_o7_dX_all6[1])][7:], bec_o7_dX_all6[1][np.isfinite(bec_o7_dX_all6[1])][7:]]
    
    dXoverdz_0p0 = 1.
    dXoverdz_0p1 = (1+0.1)**2*c.hubble*cosmopars_o8['h']/csu.Hubble(0.1)
    dXoverdz_0p2 = (1+0.2)**2*c.hubble*cosmopars_o8['h']/csu.Hubble(0.2)
    dXoverdz_1p0 = (1+1.)**2*c.hubble*cosmopars_o8['h']/csu.Hubble(1.)
    
    # example (EW in list: rest-frame EW in log10 A)
    #dNdz_o7_EWr_4mA_all_z0p0 = linterpsolve(bec_o7_dX_all[1], bec_o7_dX_all[0], np.log10(4.e-3/(1.+0.0)))*dXoverdz_0p0
    
    ## loop and print
    zvals = [0.0, 0.1, 0.2, 1.0]
    dXoverdz = {0.0: dXoverdz_0p0, 0.1: dXoverdz_0p1, 0.2: dXoverdz_0p2, 1.0: dXoverdz_1p0}
    print(dXoverdz)
    print(zvals)
    ions = ['o7', 'o8']
    sets = ['all6']
    #becs_dX = {('o7', 'all'): bec_o7_dX_all, ('o7', 'sub'): bec_o7_dX_sub, ('o7', 'all6'): bec_o7_dX_all6, ('o8', 'all'): bec_o8_dX_all, ('o8', 'sub'): bec_o8_dX_sub, ('o8', 'all6'): bec_o8_dX_all6}
    becs_dX = {('o7', 'all6'): bec_o7_dX_all6, ('o8', 'all6'): bec_o8_dX_all6}
    EWs = [6.8, 5.2, 4., 3., 1.]
    #return becs_dX
    
    for EW in EWs:
        printEW = True
        for set_ in sets:
            if printEW:
                EWpart = '$%.1f$\t& '%EW
            else:
                EWpart = '\t\t& '
            if set_ == 'all':
                lenvarpart = '$100$\t& all\t'
            elif set_ == 'sub':
                lenvarpart = '$100$\t& ion\t'
            if set_ == 'all6':
                lenvarpart = '$6.25$\t& all\t'
            lineheader = EWpart + lenvarpart
            printEW = False
            
            o7vals = [linterpsolve(becs_dX[('o7', set_)][1], becs_dX[('o7', set_)][0], np.log10(EW * 1e-3/(1.+ z))) * dXoverdz[z] for z in zvals]
            o8vals = [linterpsolve(becs_dX[('o8', set_)][1], becs_dX[('o8', set_)][0], np.log10(EW * 1e-3/(1.+ z))) * dXoverdz[z] for z in zvals]
            
            o7vals_dX = [linterpsolve(becs_dX[('o7', set_)][1], becs_dX[('o7', set_)][0], np.log10(EW * 1e-3/(1.+ z))) for z in zvals]
            o8vals_dX = [linterpsolve(becs_dX[('o8', set_)][1], becs_dX[('o8', set_)][0], np.log10(EW * 1e-3/(1.+ z))) for z in zvals]
            print(o7vals_dX)
            print(o8vals_dX)
            #o8vals = list(np.zeros(4))
            strvals = [strformat_latextable(val) for val in o7vals + o8vals]
            strvals_split = [vals.split('.')[i] for vals in strvals for i in range(2)]
            
            linevalues_base = '& %s&%s\t'*len(strvals)
            linevalues = linevalues_base%tuple(strvals_split) + r'\\'
            line = lineheader + linevalues
            print(line)
    #return becs_dX[('o7', 'all6')]

def getEWdistvals_o78_cddfsatz(plot=False): # just to store these lines, really
    '''
    prints LaTeX table of these values
    '''
    ions = ['o7', 'o8']
    snapshots = [27, 26, 19]
    #snapkeys = {19: 'snap19_snap27-N-EWconv', 26: 'snap26_snap27-N-EWconv', 27: 'snap27'} # old format using snap 27 CoG for all
    snapkeys = {19: 'snap19', 26: 'snap26', 27: 'snap27'}
    # format: ion : snapshot : [bins, edges]
    dct_out = {ion: {} for ion in ions}
    dct_dXoverdz = {ion: {} for ion in ions}
    dct_zvals = {ion: {} for ion in ions}
    grps = {ion: {snap: 'EWdists_%s_%s'%(ion, snapkeys[snap])  for snap in snapshots} for ion in ions}
    keyprefs = ['linear_cog_extrapolation_below_10^13cm^-2_all_sightlines_with_6.25cMpc_CDDF', 'linear_cog_extrapolation_below_10^13cm^-2_all_sightlines', 'linear_cog_extrapolation_below_10^13.0cm^-2_all_sightlines']
    
    with h5py.File('/net/luttero/data2/paper1/specwizard_misc.hdf5', 'r') as fi:
        for ion in ions:
            for snap in snapshots:
                grpname = grps[ion][snap]
                grp = fi[grpname]
                key = keyprefs[0]
                ki = 0
                while key not in grp.keys():
                    try:
                        ki += 1
                        key = keyprefs[ki]
                    except IndexError:
                        raise ValueError('None of the listed keys %s were datasets in group %s'%(keyprefs, grpname))
                bins = np.array(grp['%s/bins'%key])     
                edges = np.array(grp['%s/edges'%key])
                cosmopars = {key: item for (key, item) in grp['cosmopars'].attrs.items()}
                simdata = {key: item for (key, item) in grp['simdata'].attrs.items()}
                print('For group %s:'%grpname)
                print('\tusing key: \t%s'%key)
                print('\tcosmopars: \t%s'%cosmopars)
                print('\tsimdata: \t%s'%simdata)
                print('')
                
                dXtot = float(simdata['numpix'])**2 * mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars)
                bins_cumul = np.cumsum(bins[::-1])[::-1] / dXtot
                cumulpoints = edges[:-1] # left edges of bins are the values at which >= edge cumulative values are defined
                
                
                if plot:
                    plt.plot(cumulpoints + 3., bins_cumul, label=grpname)
                dct_out[ion][snap] = [bins_cumul, cumulpoints]
                dct_dXoverdz[ion][snap] = mc.getdX(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars) / mc.getdz(cosmopars['z'], cosmopars['boxsize'] / cosmopars['h'], cosmopars=cosmopars)
                dct_zvals[ion][snap] = cosmopars['z']
                
    if not (np.all([np.abs(dct_zvals['o7'][snap] - dct_zvals['o8'][snap]) <= 1e-5 for snap in snapshots]) and \
            np.all([np.abs(dct_dXoverdz['o7'][snap] - dct_dXoverdz['o8'][snap]) <= 1e-6 for snap in snapshots]) ):
        raise RuntimeError('Extracted z, dX / dz for same snapshots did not match')
    
    if plot:
        plt.xlabel(r'$\log_{10} \, \mathrm{EW} \; [\mathrm{m\AA}]$', fontsize=12)
        plt.ylabel(r'$\partial n(> EW) \,/\, \partial X$', fontsize=12)
        plt.yscale('log')
        plt.legend(fontsize=12)
        plt.xlim(-0.65, 1.3)
        plt.ylim(10**-2.1, 10**1.85)
        plt.savefig('/net/luttero/data2/paper1/plot_EWdist_evolution_used_for_table.pdf', format='pdf')
    
    # example (EW in list: rest-frame EW in log10 A)
    #dNdz_o7_EWr_4mA_all_z0p0 = linterpsolve(bec_o7_dX_all[1], bec_o7_dX_all[0], np.log10(4.e-3/(1.+0.0)))*dXoverdz_0p0
    
    ## loop and print
    EWs = [6.8, 5.2, 4., 3., 1.]
    print(dct_dXoverdz)
    print(dct_zvals)
    dct_zvals['o7'][27] = 0.11
    
    for EW in EWs:
        printEW = True
        if printEW:
            EWpart = '$%.1f$\t'%EW
        else:
            EWpart = '\t\t'
        lineheader = EWpart
        
        o7vals = [linterpsolve(dct_out['o7'][snap][1], dct_out['o7'][snap][0], np.log10(EW * 1e-3/(1.+ dct_zvals['o7'][snap]))) * dct_dXoverdz['o7'][snap] for snap in snapshots]
        o8vals = [linterpsolve(dct_out['o8'][snap][1], dct_out['o8'][snap][0], np.log10(EW * 1e-3/(1.+ dct_zvals['o8'][snap]))) * dct_dXoverdz['o8'][snap] for snap in snapshots]
        
        o7vals_dX = [linterpsolve(dct_out['o7'][snap][1], dct_out['o7'][snap][0], np.log10(EW * 1e-3/(1.+ dct_zvals['o7'][snap]))) for snap in snapshots]
        o8vals_dX = [linterpsolve(dct_out['o8'][snap][1], dct_out['o8'][snap][0], np.log10(EW * 1e-3/(1.+ dct_zvals['o8'][snap]))) for snap in snapshots]
        #print(o7vals_dX)
        #print(o8vals_dX)
        #o8vals = list(np.zeros(4))
        strvals = [strformat_latextable(val) for val in o7vals + o8vals]
        strvals_split = [vals.split('.')[i] for vals in strvals for i in range(2)]
        
        linevalues_base = '& %s&%s\t' * len(strvals)
        linevalues = linevalues_base%tuple(strvals_split) + r'\\'
        line = lineheader + linevalues
        print(line)

    #return dct_out['o7'][27]
##### associated plots #####
## some are more or less copied from specwiz_proc, but more for talks and papers than just quick looks

def dictfromnpz(filename):
    npz = np.load(filename)
    dct = {key: npz[key] for key in npz.keys()}
    return dct

import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_mp_to_sp_corr(fontsize=14, ion='o7'):
     # percentile stuff from crdists
     spcm_o7_name = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
     spcm_o8_name = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
     if ion == 'o7':
         data = Spcm(spcm_o7_name)
         ionlab = 'O\,VII'
     elif ion == 'o8':
         data = Spcm(spcm_o8_name)
         ionlab = 'O\,VIII'
     ys = data.cd_sp - data.cd_mp
     xs = data.cd_mp
     
     # bin and get percentiles (5,25,50,75,95)
     xbins = np.arange(0.999*np.min(xs), np.max(xs)*1.001+0.1*0.999,0.1)
     bininds = np.digitize(xs,xbins)
     percentiles = [5,25,50,75,95]
     ypercents = np.array([np.percentile(ys[bininds==i],percentiles) for i in range(1,len(xbins))])
     ypercents = ypercents.T
     xbins_mid = xbins[:-1] + np.diff(xbins)/2.
     
     # plot the points: outside central 5-95% range, in bins with < nminbin points
     # select bins with too few points, and the points they contain
     nminbin = 20
     xbins_count =  np.array([ len(np.where(bininds==i)[0]) for i in range(1,len(xbins)) ])
     bininds_lowcount = xbins_count <  nminbin
     bininds_ok = xbins_count > nminbin
     outliers_frombin = np.array([ bininds == i for i in np.arange(1,len(xbins))[bininds_lowcount] ] ) # check for each data point index if it is in a low count bin 
     outliers_frombin = np.any(outliers_frombin,axis=0) # select data point if it is in any low count bin    

     # get outliers from percentiles (just overplotting would also show points 'sticking out' of the end bins)
     isoutlier = np.zeros(len(xs),dtype=bool)
     for i in range(1,len(xbins)): # do percentile selection
         isoutlier[bininds == i] = np.any( np.array([ys[bininds == i] < ypercents[0][i-1],ys[bininds == i] > ypercents[-1][i-1]]) ,axis=0)
     isoutlier = np.any(np.array([isoutlier,outliers_frombin]),axis=0) # add bin selection        

     #plot in this order 
     # note that ok bins selection will not create gaps is low bins numbers occur away from the ends
     fig, ax = plt.subplots(nrows=1,ncols=1)
     # use isoutlier as selection in scatter to get all points outside percentiles, outliers_frombin to get only low bin count points
     ax.scatter(xs[outliers_frombin],ys[outliers_frombin],color='blue',s=5)
     ax.fill_between(xbins_mid[bininds_ok],ypercents[0,bininds_ok],ypercents[-1,bininds_ok],facecolor='skyblue',alpha=1.,label = '5%-95%')
     ax.fill_between(xbins_mid[bininds_ok],ypercents[1,bininds_ok],ypercents[-2,bininds_ok],facecolor='steelblue',alpha=1., label = '25%-75%')
     ax.plot(xbins_mid,ypercents[2],color='navy',label='median')
     ax.plot([xbins[0],xbins[-1]],[0,0],linestyle='dashed',color='gray',label='equal')

     ax.set_xlabel(r'$\log_{10} N_{%s} \, [\mathrm{cm}^{-2}],$ 2D-grid'%ionlab,fontsize=fontsize)
     ax.set_ylabel(r'$\log_{10} (N_{%s}, \mathrm{2D-grid} \,/\, N_{O VII}, \mathrm{specwizard})$'%ionlab,fontsize=fontsize)
     ax.legend(fontsize=fontsize)
     ax.tick_params(labelsize=fontsize,axis='both')

     plt.savefig(mdir + 'specwiz_vs_projection_coldens_%s_with_sample3_vs_test3default_L0100N1504_27_32000pix_nminbin-20.pdf'%(ion),format = 'pdf',bbox_inches='tight')



def curveofgrowth_o7(usespecwizcoldens=False,fontsize=14,alpha=0.1):
    fig, ax = plt.subplots(nrows=1,ncols=1)

    data = Spcm('specwizard_map_match_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample1_snap_027_z000p101.0.npz')
    ion='o7'

    if usespecwizcoldens:
        coldens = data.cd_sp
    else:
        coldens = data.cd_mp
    EW = data.EW
    scattercolor='teal' 
   
    ax.scatter(coldens,np.log10(EW), s=10,color=scattercolor,alpha=alpha)
    ax.scatter([],[],label = 'specwizard',s=10,color=scattercolor,alpha=1.) # adds an alpha=1 actually visible label

    bpar_logfit = 10812363.19632609 # best fit for log EW COG
    bpar_linfit = 11972953.76545441 # best fit for (lin) EW COG
    bpar_lower = 50*1e5             # ~ lower envelope
    bpar_upper = 250*1e5            # ~ upper envelope

    
    cgrid = np.arange(np.min(coldens)*0.999,np.max(coldens)+0.999*0.1,0.1)
    ax.plot(cgrid,np.log10(sp.lingrowthcurve_inv(10**cgrid,ion)),color='gray', label = 'linear')
    ax.plot(cgrid,np.log10(sp.linflatcurveofgrowth_inv(10**cgrid,bpar_lower,ion)), color='red', linestyle = 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_lower/1e5,0)))
    ax.plot(cgrid,np.log10(sp.linflatcurveofgrowth_inv(10**cgrid,bpar_logfit,ion)), color='cyan', label =  r'$b = %i$ km/s'%int(np.round(bpar_logfit/1e5,0)))
    ax.plot(cgrid,np.log10(sp.linflatcurveofgrowth_inv(10**cgrid,bpar_linfit,ion)), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_linfit/1e5,0)))
    ax.plot(cgrid,np.log10(sp.linflatcurveofgrowth_inv(10**cgrid,bpar_upper,ion)), color='orange', linestyle= 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_upper/1e5,0)))

    ax.set_ylim(ax.get_ylim()[0],-0.7)


    ax.set_ylabel(r'$\log_{10} EW\, [\mathrm{\AA}]$, rest frame',fontsize=fontsize)        
    ax.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.tick_params(labelsize=fontsize,axis='both')
    ax.set_title(r'O VII curve of growth',fontsize=fontsize)

    plt.savefig(mdir + 'specwizard_curve-of-growth_o7_sample1_with_best-fit_EW-logEW_upper_lower_ests_bpar.png',format = 'png',bbox_inches='tight')








def plotEWdists_o7(binsedges):
    '''
    binsedges, cosmopars, simdata = csm.comp_EW_dists_o7()
    '''
    import loadnpz_and_plot as lnp
    reload(lnp)
    fig, ax = plt.subplots(ncols=1,nrows =1)
    
    binsedges_keys = ['lin_13_all', 'fit_13_all', 'lin_14_all', 'fit_14_all', 'lin_13_sub', 'fit_13_sub', 'lin_14_sub', 'fit_14_sub', 'lin_cog', 'fit_cog_all', 'fit_cog_sub']
    binsedges_list = [binsedges[key] for key in binsedges_keys]
    binsedges_normed = [ (binsedges[0]/np.array(list(np.diff(binsedges[1])) + [1.]), binsedges[1]) for binsedges in binsedges_list] # last bin is zero -> just add something to match array size

    lnp.cddfsubplot1(ax, binsedges_normed, subtitle = 'EW histogram, normalised',subfigind = None,xlabel=r'$\log_{10} EW_{\mathrm{O VII}} \, [\mathrm{\AA}]$',ylabel ='# sightlines, normalised',colors = ['black', 'gray', 'blue', 'cyan', 'saddlebrown', 'chocolate', 'red', 'orange', 'olive', 'darkgreen', 'lawngreen'], labels = binsedges_keys,linestyles=['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed','dotted', 'dotted', 'dotted'], fontsize =14,xlim = (-5.,-1.7), ylim=(10**7.3,10**8.9),xticklabels=True,yticklabels=True,legendloc = 'lower left',ylog = True,subfigindloc=None,takeylog=False,legend_ncol = 2)

    plt.savefig(mdir + 'EWhist_logbinsizenormed_specwizard_map_match_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_NEWconv_variations_diffszoom.png',format = 'png',bbox_inches='tight')


def plotEWdists_o8(binsedges):
    '''
    binsedges, cosmopars, simdata = csm.comp_EW_dists_o8()
    '''
    import loadnpz_and_plot as lnp
    reload(lnp)
    fig, ax = plt.subplots(ncols=1,nrows =1)
    
    binsedges_keys = ['lin_13_all', 'fit_13_all', 'lin_14_all', 'fit_14_all', 'lin_13_sub', 'fit_13_sub', 'lin_14_sub', 'fit_14_sub', 'lin_cog', 'fit_cog_all', 'fit_cog_sub']
    binsedges_list = [binsedges[key] for key in binsedges_keys]
    binsedges_normed = [ (binsedges[0]/np.array(list(np.diff(binsedges[1])) + [1.]), binsedges[1]) for binsedges in binsedges_list] # last bin is zero -> just add something to match array size

    lnp.cddfsubplot1(ax, binsedges_normed, subtitle = 'EW histogram, normalised',subfigind = None,xlabel=r'$\log_{10} EW_{\mathrm{O VIII}} \, [\mathrm{\AA}]$',ylabel ='# sightlines, normalised',colors = ['black', 'gray', 'blue', 'cyan', 'saddlebrown', 'chocolate', 'red', 'orange', 'olive', 'darkgreen', 'lawngreen'], labels = binsedges_keys,linestyles=['solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed','dotted', 'dotted', 'dotted'], fontsize =14,xlim = (-5.5,-3.5), ylim=(10**8.5,10**9.8),xticklabels=True,yticklabels=True,legendloc = 'lower left',ylog = True,subfigindloc=None,takeylog=False,legend_ncol = 2)

    plt.savefig(mdir + 'EWhist_logbinsizenormed_specwizard_map_match_coldens_o8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_NEWconv_variations_diffszoom2_16SLICECDDF.png',format = 'png',bbox_inches='tight')










#################################################
###           Plots for the paper             ###
#################################################


def plotEWdists_o78():
    '''
    EW distributions using various col. dens. to EW conversions
    '''
    fontsize = 12
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(ncols=2,nrows=2, sharex=True,sharey='row', gridspec_kw = {'wspace':0.1, 'hspace': 0.1})
    import makecddfs as mc

    # retrieve histogram bins, edges for o7, o8:
    # keys: ['lin_13_all', 'fit_13_all', 'lin_14_all', 'fit_14_all', 'lin_13_sub', 'fit_13_sub', 'lin_14_sub', 'fit_14_sub', 'lin_cog', 'fit_cog_all', 'fit_cog_sub']
    # lin/fit and 13/14 are extensions to lower column densities -> ignore for this plot range (> 0.1 mA)
    binsedges_o8, cosmopars_o8, simdata_o8 = comp_EW_dists_o8()
    binsedges_o7, cosmopars_o7, simdata_o7 = comp_EW_dists_o7() 

    dXtot_o8 = float(simdata_o8['numpix'])**2*mc.getdX(cosmopars_o8['z'],cosmopars_o8['boxsize']/cosmopars_o8['h'],cosmopars=cosmopars_o8)
    dXtot_o7 = float(simdata_o7['numpix'])**2*mc.getdX(cosmopars_o7['z'],cosmopars_o7['boxsize']/cosmopars_o7['h'],cosmopars=cosmopars_o7)

    binsedges_o8_proc = [[binsedges_o8[key][0]/np.diff(binsedges_o8[key][1])/dXtot_o8, binsedges_o8[key][1][:-1]] for key in ['fit_13_all', 'fit_13_sub', 'fit_cog_all', 'fit_cog_sub', 'fit_13_all_6p25', 'lin_cog']] 
    binsedges_o7_proc = [[binsedges_o7[key][0]/np.diff(binsedges_o7[key][1])/dXtot_o7, binsedges_o7[key][1][:-1]] for key in ['fit_13_all', 'fit_13_sub', 'fit_cog_all', 'fit_cog_sub', 'fit_13_all_6p25', 'lin_cog']]

    
    binsedges_o8_proc_cumuldX = [[np.cumsum(binsedges_o8[key][0][::-1])[::-1]/dXtot_o8, binsedges_o8[key][1][:-1]] for key in ['fit_13_all', 'fit_13_sub', 'fit_cog_all', 'fit_cog_sub', 'fit_13_all_6p25', 'lin_cog']] 
    binsedges_o7_proc_cumuldX = [[np.cumsum(binsedges_o7[key][0][::-1])[::-1]/dXtot_o7, binsedges_o7[key][1][:-1]] for key in ['fit_13_all', 'fit_13_sub', 'fit_cog_all', 'fit_cog_sub', 'fit_13_all_6p25', 'lin_cog']] 

    xlim = (-4.,-1.3)
    ylim_hist = (-5., 1.9)
    ylim_dX   = (-6., 1.8)
    lnp.cddfsubplot1(ax1, binsedges_o7_proc, subtitle = 'O VII',subfigind ='(a)', subfigindloc=(0.88, 0.05), xlabel=None ,ylabel =r'$\log_{10} \, \mathrm{d}N \, /\, \mathrm{d} \log_{10} EW \, \mathrm{d}X$',colors = ['green', 'blue', 'darkslategray', 'cyan', 'olive', 'gray'], labels = ['mtx, all', 'mtx, o7', None, None , 'mtx, all, 6.25', None], linestyles=['solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted'], fontsize =fontsize, xlim = xlim, ylim=ylim_hist ,xticklabels=True,yticklabels=True,legendloc = 'lower left', ylog=False, takeylog=True, legend_ncol = 1)

    lnp.cddfsubplot1(ax2, binsedges_o8_proc, subtitle = 'O VIII',subfigind ='(b)', subfigindloc=(0.88, 0.05), xlabel=None,ylabel =None,colors = ['purple', 'red', 'saddlebrown', 'cyan', 'orange', 'gray'], labels = ['mtx, all', 'mtx, o8', None, None, 'mtx, all, 6.25', None], linestyles=['solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted'], fontsize =fontsize, xlim = xlim, ylim=ylim_hist ,xticklabels=False,yticklabels=False,legendloc = 'lower left',ylog=False, takeylog=True, legend_ncol=1)

    lnp.cddfsubplot1(ax3, binsedges_o7_proc_cumuldX, subtitle = 'O VII',subfigind ='(c)', subfigindloc=(0.88, 0.05), xlabel=r'$\log_{10} EW_{\mathrm{O VII}} \, [\mathrm{\AA}]$',ylabel =r'$\log_{10} \, \mathrm{d}N(>EW)\,/\,\mathrm{d}X$',colors = ['green', 'blue', 'darkslategray', 'cyan', 'olive', 'gray'], labels =  [None, None, 'COG, all', 'COG, o7', None, 'lin. COG'], linestyles=['solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted'], fontsize =fontsize, xlim = xlim, ylim=ylim_dX,xticklabels=True,yticklabels=True,legendloc = 'lower left', ylog=False, takeylog=True,legend_ncol = 1,steppost=False)

    lnp.cddfsubplot1(ax4, binsedges_o8_proc_cumuldX, subtitle = 'O VIII',subfigind ='(d)', subfigindloc=(0.88, 0.05), xlabel=r'$\log_{10} EW_{\mathrm{O VIII}} \, [\mathrm{\AA}]$',ylabel =None, colors = ['purple', 'red', 'saddlebrown', 'cyan', 'orange', 'gray'], labels = [None, None,  'COG, all', 'COG, o8', None, 'lin. COG'], linestyles=['solid', 'dashed', 'solid', 'dashed', 'dashdot', 'dotted'], fontsize =fontsize, xlim = xlim, ylim=ylim_dX,xticklabels=True,yticklabels=False,legendloc = 'lower left', ylog=False, takeylog=True,legend_ncol = 1,steppost=False)

    plt.savefig(mdir + 'EWhist_and_cumuldX_specwizard_map_match_coldens_o7-o8_L0100N1504_27_test3.1-3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_NEWconv_variations.pdf', format='pdf', bbox_inches='tight')






def plot_curveofgrowth_o7_o8(usespecwizcoldens=True,fontsize=12,alpha=0.05, slidemode = False):
    '''
    EW as a function of col. dens. with various b-parameter values indicated, and completeness limits. Sample 3 data 
    '''

    # fig, axes setup and settings
    if slidemode:
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize = (10.5,5.),gridspec_kw = {'wspace': 0.0})
        labelleft_ax1 = True
        labelleft_ax2 = False
        labelbottom_ax1 = True
        labelbottom_ax2 = True
        
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,figsize = (5.5,10.),gridspec_kw = {'hspace': 0.0})
        labelleft_ax1 = True
        labelleft_ax2 = True
        labelbottom_ax1 = False
        labelbottom_ax2 = True

    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',\
                   labelleft = labelleft_ax1, labeltop = False, labelbottom = labelbottom_ax1, labelright = False)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',\
                   labelleft = labelleft_ax2, labeltop = False, labelbottom = labelbottom_ax2, labelright = False)
    # prevent tick overlap
    if slidemode:
        pass
    else:
        ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 10, prune='lower'))
        ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 10))

    if labelbottom_ax1:
        ax1.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    if labelbottom_ax2:
        ax2.set_xlabel(r'$\log_{10} N \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)
    if labelleft_ax1:  
        ax1.set_ylabel(r'$\log_{10} EW\, [\mathrm{\AA}]$, rest frame',fontsize=fontsize)        
    if labelleft_ax2:
        ax2.set_ylabel(r'$\log_{10} EW\, [\mathrm{\AA}]$, rest frame',fontsize=fontsize) 

    ylim = (-5.,-1.)
    ax1.set_ylim(ylim)
    ax1.set_xlim((12.9,17.9))
    ax2.set_ylim(ylim)
    ax2.set_xlim((12.9,17.9))
    
    # subtitles and subplot labels
    ax1.text(0.05,0.95,'O VII',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    ax2.text(0.05,0.95,'O VIII',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax2.transAxes, bbox=dict(facecolor='white',alpha=0.3))

    ax1.text(0.95,0.55,'(a)',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax1.transAxes)
    ax2.text(0.95,0.47,'(b)',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax2.transAxes)


    # hdf5 file with sample3 selection data: index/file 0 = o7 (test3, not 3.1, but should be the same), 1 = o8
    selectionf = h5py.File('/net/luttero/data2/specwizard_data/los_sample3_o6-o7-o8_L0100N1504_data.hdf5')

    # get completeness limits for o7:
    samplehist_o7 = np.array(selectionf['Selection/file0/sample_pixels_per_bin'])
    samplebins_o7 = np.array(selectionf['Header/bins'][0,:])
    selectedhist_o7, temp = np.histogram(np.array(selectionf['Selection/file0/selected_values_allions']), bins = samplebins_o7)
    #completeness_lim_o7 = samplebins_o7[np.max(np.where(samplehist_o7 > selectedhist_o7)[0]) +1] # right edge of largest col. dens. bin where total > selected

    samplehist_o8 = np.array(selectionf['Selection/file1/sample_pixels_per_bin'])
    samplebins_o8 = np.array(selectionf['Header/bins'][1,:])
    selectedhist_o8, temp = np.histogram(np.array(selectionf['Selection/file1/selected_values_allions']), bins = samplebins_o8)
    #completeness_lim_o8 = samplebins_o8[np.max(np.where(samplehist_o8 > selectedhist_o8)[0]) +1] # right edge of largest col. dens. bin where total > selected  

    # get coldens, EW from Spcm for o7, split into o7-selected and other samples
    spcm_o7_name = ol.pdir + 'specwizard_map_match_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
    spcm_o8_name = ol.pdir + 'specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox__net_luttero_data2_specwizard_data_sample3_snap_027_z000p101.0.npz'
    data_o7 = Spcm(spcm_o7_name)
    if usespecwizcoldens:
        coldens_o7 = data_o7.cd_sp
    else:
        coldens_o7 = data_o7.cd_mp
    EW_o7 = data_o7.EW

    data_o7.subsample_match(np.array(selectionf['Selection/file0/selected_pixels_thision']),norm='pix')
    indsmatch_o7 = data_o7.indsclosest
    if usespecwizcoldens:
        coldens_o7_sub = data_o7.cd_sp
    else:
        coldens_o7_sub = data_o7.cd_mp
    EW_o7_sub = data_o7.EW

    nosub_o7 = np.array(list( set(np.arange(len(coldens_o7))) - set(indsmatch_o7))) 
    EW_o7_nosub = EW_o7[nosub_o7]
    coldens_o7_nosub = coldens_o7[nosub_o7]


    # get coldens, EW from Spcm for o8
    data_o8 = Spcm(spcm_o8_name)
    if usespecwizcoldens:
        coldens_o8 = data_o8.cd_sp
    else:
        coldens_o8 = data_o8.cd_mp
    EW_o8 = data_o8.EW

    data_o8.subsample_match(np.array(selectionf['Selection/file1/selected_pixels_thision']),norm='pix')
    indsmatch_o8 = data_o8.indsclosest
    if usespecwizcoldens:
        coldens_o8_sub = data_o8.cd_sp
    else:
        coldens_o8_sub = data_o8.cd_mp
    EW_o8_sub = data_o8.EW

    nosub_o8 = np.array(list( set(np.arange(len(coldens_o8))) - set(indsmatch_o8))) 
    EW_o8_nosub = EW_o8[nosub_o8]
    coldens_o8_nosub = coldens_o8[nosub_o8]


    # plot the subsamples
    colors = [['blue','green'],['red','purple']] 
   
    ax1.scatter(coldens_o7_sub,np.log10(EW_o7_sub), s=10,color=colors[0][0],alpha=alpha)
    ax1.scatter([],[],label = 'o7-selected',s=10,color=colors[0][0],alpha=1.) # adds an alpha=1 actually visible label
    ax1.scatter(coldens_o7_nosub,np.log10(EW_o7_nosub), s=10,color=colors[0][1],alpha=alpha)
    ax1.scatter([],[],label = 'other o7',s=10,color=colors[0][1],alpha=1.) # adds an alpha=1 actually visible label

    ax2.scatter(coldens_o8_sub,np.log10(EW_o8_sub), s=10,color=colors[1][0],alpha=alpha)
    ax2.scatter([],[],label = 'o8-selected',s=10,color=colors[1][0],alpha=1.) # adds an alpha=1 actually visible label
    ax2.scatter(coldens_o8_nosub,np.log10(EW_o8_nosub), s=10,color=colors[1][1],alpha=alpha)
    ax2.scatter([],[],label = 'other o8',s=10,color=colors[1][1],alpha=1.) # adds an alpha=1 actually visible label
    
    # add addtional info (o7) -- best fits for subsample fit
    bpar_logfit_o7 = 90.797881*1e5 # best fit for log EW COG
    #bpar_linfit_o7 = 9435311.30181743 # best fit for (lin) EW COG
    bpar_lower_o7 = 50*1e5             # ~ lower envelope
    bpar_vlower_o7 = 20e5
    bpar_upper_o7 = 220*1e5            # ~ upper envelope

    
    cgrid_o7 = np.arange(np.min(coldens_o7)*0.999,np.max(coldens_o7)+0.999*0.1,0.1)
    ax1.plot(cgrid_o7, np.log10(sp.lingrowthcurve_inv(10**cgrid_o7,'o7')),color='gray', label = 'linear')
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_vlower_o7,'o7')), color='sienna', linestyle = 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_vlower_o7/1e5,0)), linewidth = 2)
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_lower_o7,'o7')), color='gold', linestyle = 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_lower_o7/1e5,0)), linewidth = 2)
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_logfit_o7,'o7')), color='cyan', label =  r'$b = %i$ km/s'%int(np.round(bpar_logfit_o7/1e5,0)), linewidth = 2)
    #ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_linfit_o7,'o7')), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_linfit_o7/1e5,0)), linewidth = 2)
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_upper_o7,'o7')), color='orange', linestyle= 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_upper_o7/1e5,0)), linewidth = 2)
    #ax1.axvline(x=completeness_lim_o7, ymin = 0.7, color = 'brown', label = 'complete', linewidth = 2, linestyle = 'dashed')

   # add addtional info (o8) -- best fits for subsample fit
    bpar_logfit_o8 = 158.163355*1e5 # best fit for log EW COG
    bpar_logfit_o8_proj = 140.249975*1e5 # best fit for (log) EW COG using projected column densities
    bpar_weirdfit_o8_proj = 104.381034*1e5    # best fit for log EW COG, full sample, using projected column densities
    bpar_lower_o8 = 70*1e5             # ~ lower envelope
    bpar_upper_o8 = 300*1e5            # ~ upper envelope

    
    cgrid_o8 = np.arange(np.min(coldens_o8)*0.999,np.max(coldens_o8)+0.999*0.1,0.1)
    ax2.plot(cgrid_o8, np.log10(sp.lingrowthcurve_inv(10**cgrid_o8,'o8')),color='gray', label = 'linear')
    ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_lower_o8,'o8')), color='gold', linestyle = 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_lower_o8/1e5,0)), linewidth = 2)
    ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_logfit_o8,'o8')), color='cyan', label =  r'$b = %i$ km/s'%int(np.round(bpar_logfit_o8/1e5,0)), linewidth = 2)
    #ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_logfit_o8_proj,'o8')), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_logfit_o8_proj/1e5,0)), linewidth = 2)
    #ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_weirdfit_o8_proj,'o8')), color='olive', label =  r'$b = %i$ km/s'%int(np.round(bpar_weirdfit_o8_proj/1e5,0)), linewidth = 2)
    ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_upper_o8,'o8')), color='orange', linestyle= 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_upper_o8/1e5,0)), linewidth = 2)
    #ax2.axvline(x=completeness_lim_o8, ymin = 0.7, color = 'brown', label = 'complete', linewidth = 2, linestyle = 'dashed')


    # legends    
    ax1.legend(fontsize=fontsize, loc='lower right')
    ax2.legend(fontsize=fontsize, loc='lower right')

    if slidemode:
        plt.savefig(mdir + 'specwizard_curve-of-growth_o7-o8_sample3_with_best-fit_EW-logEW_upper_lower_ests_bpar_specwizcoldens_slide.png',format = 'png',bbox_inches='tight',dpi=300)
    else:
        plt.savefig(mdir + 'specwizard_curve-of-growth_o7-o8_sample3_with_best-fit_EW-logEW_upper_lower_ests_bpar_specwizcoldens.pdf',format = 'pdf',bbox_inches='tight')




def plot_curveofgrowth_o7_o8_simplified(usespecwizcoldens=False,fontsize=20,alpha=0.05, slidemode = True):
    '''
    !!!! uses old sample3 data
    EW as a function of col. dens. with various b-parameter values indicated, and completeness limits. Sample 3 data 
    simplified version for talks
    '''

    # fig, axes setup and settings
    if slidemode:
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize = (10.5,5.),gridspec_kw = {'wspace': 0.0})
        labelleft_ax1 = True
        labelleft_ax2 = False
        labelbottom_ax1 = True
        labelbottom_ax2 = True
        
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,figsize = (5.5,10.),gridspec_kw = {'hspace': 0.0})
        labelleft_ax1 = True
        labelleft_ax2 = True
        labelbottom_ax1 = False
        labelbottom_ax2 = True

    ax1.minorticks_on()
    ax1.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',\
                   labelleft = labelleft_ax1, labeltop = False, labelbottom = labelbottom_ax1, labelright = False)
    ax2.minorticks_on()
    ax2.tick_params(labelsize=fontsize,direction = 'in', right = True, top = True, axis='both', which = 'both',\
                   labelleft = labelleft_ax2, labeltop = False, labelbottom = labelbottom_ax2, labelright = False)
    # prevent tick overlap
    if slidemode:
        pass
    else:
        ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 10, prune='lower'))
        ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(steps = [1,2,5,10], nbins = 10))

    if labelbottom_ax1:
        ax1.set_xlabel(r'$\log_{10} N_{\mathrm{O\, VII}} \, [\mathrm{cm}^{-2}]$',fontsize=fontsize)  
    if labelbottom_ax2:
        if slidemode:
            ionlab = 'O\, VIII'
        else:
            ionlab = 'ion'
        ax2.set_xlabel(r'$\log_{10} N_{\mathrm{%s}} \, [\mathrm{cm}^{-2}]$'%ionlab,fontsize=fontsize)
    if labelleft_ax1:  
        ax1.set_ylabel(r'$\log_{10} EW\, [\mathrm{\AA}]$, rest frame',fontsize=fontsize)        
    if labelleft_ax2:
        ax2.set_ylabel(r'$\log_{10} EW\, [\mathrm{\AA}]$, rest frame',fontsize=fontsize) 

    ylim = (-5.,-1.)
    ax1.set_ylim(ylim)
    ax1.set_xlim((12.9,17.9))
    ax2.set_ylim(ylim)
    ax2.set_xlim((12.9,17.9))
    
    # subtitles and subplot labels
    ax1.text(0.05,0.95,'O VII',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax1.transAxes, bbox=dict(facecolor='white',alpha=0.3))
    ax2.text(0.05,0.95,'O VIII',fontsize=fontsize, horizontalalignment = 'left', verticalalignment = 'top', transform=ax2.transAxes, bbox=dict(facecolor='white',alpha=0.3))

    #ax1.text(0.95,0.67,'(a)',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax1.transAxes)
    #ax2.text(0.95,0.67,'(b)',fontsize=fontsize, horizontalalignment = 'right', verticalalignment = 'bottom', transform=ax2.transAxes)


    # hdf5 file with sample3 selection data: index/file 0 = o7 (test3, not 3.1, but should be the same), 1 = o8
    selectionf = h5py.File('/cosma/home/dp004/dc-wije1/specwizard/Ali_Spec_src/los/los_sample3_o6-o7-o8_L0100N1504_data.hdf5')

    # get completeness limits for o7:
    samplehist_o7 = np.array(selectionf['Selection/file0/sample_pixels_per_bin'])
    samplebins_o7 = np.array(selectionf['Header/bins'][0,:])
    selectedhist_o7, temp = np.histogram(np.array(selectionf['Selection/file0/selected_values_allions']), bins = samplebins_o7)
    completeness_lim_o7 = samplebins_o7[np.max(np.where(samplehist_o7 > selectedhist_o7)[0]) +1] # right edge of largest col. dens. bin where total > selected

    samplehist_o8 = np.array(selectionf['Selection/file1/sample_pixels_per_bin'])
    samplebins_o8 = np.array(selectionf['Header/bins'][1,:])
    selectedhist_o8, temp = np.histogram(np.array(selectionf['Selection/file1/selected_values_allions']), bins = samplebins_o8)
    completeness_lim_o8 = samplebins_o8[np.max(np.where(samplehist_o8 > selectedhist_o8)[0]) +1] # right edge of largest col. dens. bin where total > selected  

    # get coldens, EW from Spcm for o7, split into o7-selected and other samples
    data_o7 = Spcm('specwizard_map_match_coldens_o7_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101.0.npz')
    if usespecwizcoldens:
        coldens_o7 = data_o7.cd_sp
    else:
        coldens_o7 = data_o7.cd_mp
    EW_o7 = data_o7.EW

    #data_o7.subsample_match(np.array(selectionf['Selection/file0/selected_pixels_thision']),norm='pix')
    #indsmatch_o7 = data_o7.indsclosest
    #if usespecwizcoldens:
    #    coldens_o7_sub = data_o7.cd_sp
    #else:
    #    coldens_o7_sub = data_o7.cd_mp
    #EW_o7_sub = data_o7.EW

    #nosub_o7 = np.array(list( set(np.arange(len(coldens_o7))) - set(indsmatch_o7))) 
    #EW_o7_nosub = EW_o7[nosub_o7]
    #coldens_o7_nosub = coldens_o7[nosub_o7]


    # get coldens, EW from Spcm for o8
    data_o8 = Spcm('specwizard_map_match_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_sample3_snap_027_z000p101.0.npz')
    if usespecwizcoldens:
        coldens_o8 = data_o8.cd_sp
    else:
        coldens_o8 = data_o8.cd_mp
    EW_o8 = data_o8.EW

    #data_o8.subsample_match(np.array(selectionf['Selection/file1/selected_pixels_thision']),norm='pix')
    #indsmatch_o8 = data_o8.indsclosest
    #if usespecwizcoldens:
    #    coldens_o8_sub = data_o8.cd_sp
    #else:
    #    coldens_o8_sub = data_o8.cd_mp
    #EW_o8_sub = data_o8.EW

    #nosub_o8 = np.array(list( set(np.arange(len(coldens_o8))) - set(indsmatch_o8))) 
    #EW_o8_nosub = EW_o8[nosub_o8]
    #coldens_o8_nosub = coldens_o8[nosub_o8]


    # plot the subsamples
    colors = [['blue'],['red']] 
   
    ax1.scatter(coldens_o7,np.log10(EW_o7), s=10,color=colors[0][0],alpha=alpha)
    ax1.scatter([],[],label = 'EAGLE',s=10,color=colors[0][0],alpha=1.) # adds an alpha=1 actually visible label
    #ax1.scatter(coldens_o7_sub,np.log10(EW_o7_sub), s=10,color=colors[0][1],alpha=alpha)
    #ax1.scatter([],[],label = 'other o7',s=10,color=colors[0][1],alpha=1.) # adds an alpha=1 actually visible label

    ax2.scatter(coldens_o8,np.log10(EW_o8), s=10,color=colors[1][0],alpha=alpha)
    ax2.scatter([],[],label = 'EAGLE',s=10,color=colors[1][0],alpha=1.) # adds an alpha=1 actually visible label
    #ax2.scatter(coldens_o8_sub,np.log10(EW_o8_sub), s=10,color=colors[1][1],alpha=alpha)
    #ax2.scatter([],[],label = 'other o8',s=10,color=colors[1][1],alpha=1.) # adds an alpha=1 actually visible label


    # add addtional info (o7) -- best fits for subsample fit
    bpar_logfit_o7 = 9079723.13153631 # best fit for log EW COG
    bpar_linfit_o7 = 9435311.30181743 # best fit for (lin) EW COG
    bpar_lower_o7 = 50*1e5             # ~ lower envelope
    bpar_upper_o7 = 220*1e5            # ~ upper envelope

    
    cgrid_o7 = np.arange(np.min(coldens_o7)*0.999,np.max(coldens_o7)+0.999*0.1,0.1)
    ax1.plot(cgrid_o7, np.log10(sp.lingrowthcurve_inv(10**cgrid_o7,'o7')),color='black', label = 'linear')
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_lower_o7,'o7')), color='purple', linestyle = 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_lower_o7/1e5,0)), linewidth = 2)
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_logfit_o7,'o7')), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_logfit_o7/1e5,0)), linewidth = 2)
    #ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_linfit_o7,'o7')), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_linfit_o7/1e5,0)), linewidth = 2)
    ax1.plot(cgrid_o7, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o7,bpar_upper_o7,'o7')), color='dodgerblue', linestyle= 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_upper_o7/1e5,0)), linewidth = 2)
    #ax1.axvline(x=completeness_lim_o7, ymin = 0.7, color = 'brown', label = 'complete', linewidth = 2, linestyle = 'dashed')

   # add addtional info (o7) -- best fits for subsample fit
    bpar_logfit_o8 = 16226916.07507808 # best fit for log EW COG
    bpar_linfit_o8 = 15957908.19474436 # best fit for (lin) EW COG
    bpar_lower_o8 = 70*1e5             # ~ lower envelope
    bpar_upper_o8 = 300*1e5            # ~ upper envelope

    
    cgrid_o8 = np.arange(np.min(coldens_o8)*0.999,np.max(coldens_o8)+0.999*0.1,0.1)
    ax2.plot(cgrid_o8, np.log10(sp.lingrowthcurve_inv(10**cgrid_o8,'o8')),color='black', label = 'linear')
    ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_lower_o8,'o8')), color='purple', linestyle = 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_lower_o8/1e5,0)), linewidth = 2)
    ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_logfit_o8,'o8')), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_logfit_o8/1e5,0)), linewidth = 2)
    #ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_linfit_o8,'o8')), color='lime', label =  r'$b = %i$ km/s'%int(np.round(bpar_linfit_o8/1e5,0)), linewidth = 2)
    ax2.plot(cgrid_o8, np.log10(sp.linflatcurveofgrowth_inv(10**cgrid_o8,bpar_upper_o8,'o8')), color='dodgerblue', linestyle= 'dotted', label =  r'$b = %i$ km/s'%int(np.round(bpar_upper_o8/1e5,0)), linewidth = 2)
    #ax2.axvline(x=completeness_lim_o8, ymin = 0.7, color = 'brown', label = 'complete', linewidth = 2, linestyle = 'dashed')


    # legends    
    ax1.legend(fontsize=fontsize)
    ax2.legend(fontsize=fontsize)

    if slidemode:
        plt.savefig(mdir + 'specwizard_curve-of-growth_o7-o8_sample3_with_best-fit_EW-logEW_upper_lower_ests_bpar_specwizcoldens_simplified_slide.png',format = 'png',bbox_inches='tight',dpi=300)
    else:
        plt.savefig(mdir + 'specwizard_curve-of-growth_o7-o8_sample3_with_best-fit_EW-logEW_upper_lower_ests_bpar_specwizcoldens_simplified.pdf',format = 'pdf',bbox_inches='tight')



def readin_cf06(filename,Angstrom_to_kms):
    '''
    Reads in data from Cen & Fang 2006 (WebPlotDigitize) and converts mA EWs to log10 km/s
    '''
    import loadnpz_and_plot as lnp
    reload(lnp)

    binsedges = lnp.readdata('/net/luttero/data2/cen_fang_2006/%s'%(filename), headerlength=1)
    binsedges = (binsedges.T)[::-1,:]
    binsedges[1] = np.log10(binsedges[1]) -3. + np.log10(Angstrom_to_kms) # mA to km/s
    return binsedges

def add_ax(ax, diff, xory = 'x',fontsize=None, label = None):

    if xory == 'x':
        ax2 = ax.twiny()
        old_ticklocs = ax.get_xticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
	
	# use same spacing and number of ticks, but start at first integer value in the new units
	new_lim = (old_xlim[0] + diff, old_xlim[1] + diff)
	newticks = np.array(list(set(list(np.ceil(new_lim[0]) + np.array(old_ticklocs) - old_ticklocs[0])) | set(list(np.floor(new_lim[0]) + np.array(old_ticklocs) - old_ticklocs[0]))))
	newticks = np.round(newticks,2)
        newticklabels = [str(int(tick)) if int(tick)== tick else str(tick) for tick in newticks]
   	
	#print old_ticklocs
	print newticklabels
        #ax2.set_xticks(np.round(old_ticklocs + np.log10(rho_to_nh),2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        #ax2.set_xticklabels(['%.2f' %number for number in np.round(old_ticklocs + np.log10(rho_to_nh),2)]) 
	ax2.set_xticks(newticks - diff) # old locations, shifted just so that the round-off works out
        ax2.set_xticklabels(newticklabels)    
        if label is not None:         
            ax2.set_xlabel(label,fontsize=fontsize)
        ax2.set_xlim(old_xlim)
        ax2.set_ylim(old_ylim)
    else:
        ax2 = ax.twinx()
        old_ticklocs = ax.get_yticks() #array
        old_xlim = ax.get_xlim()
        old_ylim = ax.get_ylim()
        ax2.set_yticks(np.round(old_ticklocs +diff,2) - np.log10(rho_to_nh)) # old locations, shifted just so that the round-off works out
        ax2.set_yticklabels(['%.2f' %number for number in np.round(old_ticklocs +diff,2)])        
        if label is not None:
            ax2.set_ylabel(label,fontsize=fontsize)
        ax2.set_xlim(old_xlim)
        ax2.set_ylim(old_ylim)
    ax2.minorticks_on() 
    ax2.tick_params(labelsize=fontsize,axis='both', direction = 'in')
    return ax2

def interp_fill_between(binsedges1,binsedges2):
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

def plotEWdists_o78_litcomp(slidemode=False):
    '''
    EW distributions using various col. dens. to EW conversions, compared to Cen & Fang GSW (with feedback) models, and Branchini 2009 model B2 (with rho-Z scatter)
    currently assumes B+09, CF06 EWs are rest-frame
    saved as EWcumul_specwizard_map_match_coldens_o7-o8_L0100N1504_27_test3.1-3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_mtx-NEWconv_branchini_etal_2009_cen_fang_2006.pdf
    '''
    if slidemode: 
        fontsize = 14
        sfi_a = ''
        sfi_b = ''
    else:
        fontsize = 12
        sfi_a = '(a)'
        sfi_b = '(b)'
    fig, (ax1, ax2) = plt.subplots(ncols=2,nrows=1, sharex=True,sharey='row', gridspec_kw = {'wspace':0.1, 'hspace': 0.1})
    import loadnpz_and_plot as lnp
    reload(lnp)
    import makecddfs as mc

    # retrieve histogram bins, edges for o7, o8:
    # keys: ['lin_13_all', 'fit_13_all', 'lin_14_all', 'fit_14_all', 'lin_13_sub', 'fit_13_sub', 'lin_14_sub', 'fit_14_sub', 'lin_cog', 'fit_cog_all', 'fit_cog_sub']
    # lin/fit and 13/14 are extensions to lower column densities -> ignore for this plot range (> 0.1 mA)
    binsedges_o8, cosmopars_o8, simdata_o8 = comp_EW_dists_o8()
    binsedges_o7, cosmopars_o7, simdata_o7 = comp_EW_dists_o7() 

    dztot_o8 = float(simdata_o8['numpix'])**2*mc.getdz(cosmopars_o8['z'],cosmopars_o8['boxsize']/cosmopars_o8['h'],cosmopars=cosmopars_o8)
    dztot_o7 = float(simdata_o7['numpix'])**2*mc.getdz(cosmopars_o7['z'],cosmopars_o7['boxsize']/cosmopars_o7['h'],cosmopars=cosmopars_o7)

    #binsedges_o8_proc = [[binsedges_o8[key][0]/np.diff(binsedges_o8[key][1])/dXtot_o8, binsedges_o8[key][1][:-1]] for key in ['fit_13_all', 'fit_13_sub', 'fit_cog_all', 'fit_cog_sub', 'lin_cog']] 
    #binsedges_o7_proc = [[binsedges_o7[key][0]/np.diff(binsedges_o7[key][1])/dXtot_o7, binsedges_o7[key][1][:-1]]for key in ['fit_13_all', 'fit_13_sub', 'fit_cog_all', 'fit_cog_sub', 'lin_cog']]

    angstrom_to_kms_o7 = c.c/sp.lambda_rest['o7']/1.e5
    angstrom_to_kms_o8 = c.c/sp.lambda_rest['o8']/1.e5 # fosc-weighted average wavelength

    binsedges_o8_proc_cumuldz = [[np.cumsum(binsedges_o8[key][0][::-1])[::-1]/dztot_o8, np.log10(angstrom_to_kms_o8) + binsedges_o8[key][1][:-1]] for key in ['fit_13_all', 'fit_13_all_6p25']] # , 'fit_13_sub' 
    binsedges_o7_proc_cumuldz = [[np.cumsum(binsedges_o7[key][0][::-1])[::-1]/dztot_o7, np.log10(angstrom_to_kms_o7) + binsedges_o7[key][1][:-1]] for key in ['fit_13_all', 'fit_13_all_6p25']] # , 'fit_13_sub' 

    #xlim = (-4.,-1.3)
    #ylim_hist = (1e-5, 10**1.5)
    #ylim_dX   = (1.e-6, 10**1.8)

    ## read in Branchini et al. 2009 data, process
    binsedges_branchini2009_o7 = lnp.readdata('/net/luttero/data2/branchini_2009/o7_chen_cumul.dat', headerlength=0)
    binsedges_branchini2009_o7 = (binsedges_branchini2009_o7.T)[::-1,:] # read in as EW, dndz, EW, dndz, ...
    binsedges_branchini2009_o7[1] = np.log10(binsedges_branchini2009_o7[1])

    binsedges_branchini2009_o8 = lnp.readdata('/net/luttero/data2/branchini_2009/o8_chen_cumul.dat', headerlength=0)
    binsedges_branchini2009_o8 = (binsedges_branchini2009_o8.T)[::-1,:] # read in as EW, dndz, EW, dndz, ...
    binsedges_branchini2009_o8[1] = np.log10(binsedges_branchini2009_o8[1])

    ## read in Cen & Fang 2006 data, process
    binsedges_o7_cf06_GL_u = readin_cf06('o7EW_GSW_LTE_upper.dat',angstrom_to_kms_o7)
    binsedges_o7_cf06_GL_l = readin_cf06('o7EW_GSW_LTE_lower.dat',angstrom_to_kms_o7)
    binsedges_o7_cf06_GnL_u = readin_cf06('o7EW_GSW_noLTE_upper.dat',angstrom_to_kms_o7)
    binsedges_o7_cf06_GnL_l = readin_cf06('o7EW_GSW_noLTE_lower.dat',angstrom_to_kms_o7)

    binsedges_o8_cf06_GL_u = readin_cf06('o8EW_GSW_LTE_upper.dat',angstrom_to_kms_o8)
    binsedges_o8_cf06_GL_l = readin_cf06('o8EW_GSW_LTE_lower.dat',angstrom_to_kms_o8)
    binsedges_o8_cf06_GnL_u = readin_cf06('o8EW_GSW_noLTE_upper.dat',angstrom_to_kms_o8)
    binsedges_o8_cf06_GnL_l = readin_cf06('o8EW_GSW_noLTE_lower.dat',angstrom_to_kms_o8)

    edges_o7_cf06_GL_fb, bins_o7_cf06_GL_u_fb, bins_o7_cf06_GL_l_fb = interp_fill_between(binsedges_o7_cf06_GL_u,binsedges_o7_cf06_GL_l)
    edges_o7_cf06_GnL_fb, bins_o7_cf06_GnL_u_fb, bins_o7_cf06_GnL_l_fb = interp_fill_between(binsedges_o7_cf06_GnL_u,binsedges_o7_cf06_GnL_l)
    edges_o8_cf06_GL_fb, bins_o8_cf06_GL_u_fb, bins_o8_cf06_GL_l_fb = interp_fill_between(binsedges_o8_cf06_GL_u,binsedges_o8_cf06_GL_l)
    edges_o8_cf06_GnL_fb, bins_o8_cf06_GnL_u_fb, bins_o8_cf06_GnL_l_fb = interp_fill_between(binsedges_o8_cf06_GnL_u,binsedges_o8_cf06_GnL_l)

    lnp.cddfsubplot1(ax1, binsedges_o7_proc_cumuldz, subtitle = 'O VII',subfigind =sfi_a,xlabel=r'$\log_{10} EW_{\mathrm{O VII}} \, [\mathrm{km}\,\mathrm{s}^{-1}]$',ylabel =r'$\log_{10} \, \mathrm{d}N(>EW)\,/\,\mathrm{d}z$',colors = ['blue', 'cyan'], labels=[None, None], linestyles=['solid', 'solid'], fontsize =fontsize, xlim = (0.5,2.45), ylim=(-1.05 , 1.85), xticklabels=True, yticklabels=True, ylog=False, subfigindloc=(0.95,0.84), takeylog=True, steppost=False, subtitleloc=(0.95,0.962))

    ax1.fill_between(edges_o7_cf06_GL_fb, np.log10(bins_o7_cf06_GL_l_fb), np.log10(bins_o7_cf06_GL_u_fb), label = 'CF06, G-L', facecolors=(1.,0.,0.,0.5), edgecolor='red', linestyle = 'dashdot')
    ax1.fill_between(edges_o7_cf06_GnL_fb, np.log10(bins_o7_cf06_GnL_l_fb), np.log10(bins_o7_cf06_GnL_u_fb), label = 'CF06, G-nL', facecolors=(0.3,0.6,0.2,0.7), edgecolor=(0.3,0.6,0.2,1.), linestyle = 'dashdot')
    ax1.fill_betweenx(np.log10(binsedges_branchini2009_o7[0]), binsedges_branchini2009_o7[1] - np.log10(1.5), binsedges_branchini2009_o7[1], label = 'B+09, B2', linestyle = 'dashed', facecolors=(0.5,0.5,0.5,0.5), edgecolor=(0.5,0.5,0.5,1.)) # observed-frame EWs from redshifts 0.5 to 0 -> bracketed rest-frame EWs

    ax1.legend(fontsize=fontsize, loc = 'lower left')

    aax1 = add_ax(ax1, -1*np.log10(angstrom_to_kms_o7) +3, xory = 'x',fontsize=fontsize, label = r'$\log_{10} \mathrm{m\AA}$')
    aax1.tick_params(labelsize=fontsize-1,direction='in',top=True, which = 'both')
    aax1.minorticks_on()

    lnp.cddfsubplot1(ax2, binsedges_o8_proc_cumuldz, subtitle = 'O VIII',subfigind =sfi_b,xlabel=r'$\log_{10} EW_{\mathrm{O VIII}} \, [\mathrm{km}\,\mathrm{s}^{-1}]$',ylabel =None,colors = ['blue', 'cyan'], labels = ['EAGLE-100', 'EAGLE-6.25'], linestyles=['solid', 'solid', 'dashed'], fontsize =fontsize, xlim = (0.5,2.45), ylim=(-1.05, 1.85),xticklabels=True,yticklabels=False, legendloc = 'lower left', ylog=False, subfigindloc=(0.95,0.84), takeylog=True, legend_ncol = 1, steppost=False, subtitleloc=(0.95,0.962))

    ax2.fill_between(edges_o8_cf06_GL_fb, np.log10(bins_o8_cf06_GL_l_fb), np.log10(bins_o8_cf06_GL_u_fb), label = 'CF06, G-L', facecolors=(1.,0.,0.,0.5), edgecolor='red', linestyle = 'dashdot')
    ax2.fill_between(edges_o8_cf06_GnL_fb, np.log10(bins_o8_cf06_GnL_l_fb), np.log10(bins_o8_cf06_GnL_u_fb), label = 'CF06, G-nL', facecolors=(0.3,0.6,0.2,0.7), edgecolor=(0.3,0.6,0.2,1.), linestyle = 'dashdot')
    ax2.fill_betweenx(np.log10(binsedges_branchini2009_o8[0]), binsedges_branchini2009_o8[1] - np.log10(1.5), binsedges_branchini2009_o8[1], label = 'B+09, B2', linestyle = 'dashed', facecolors=(0.5,0.5,0.5,0.5), edgecolor=(0.5,0.5,0.5,1.)) # observed-frame EWs from redshifts 0.5 to 0 -> bracketed rest-frame EWs

    #ax2.legend(fontsize=fontsize, loc = 'lower left')

    aax2 = add_ax(ax2, -1*np.log10(angstrom_to_kms_o8) + 3, xory = 'x',fontsize=fontsize, label = r'$\log_{10} \mathrm{m\AA}$')
    aax2.tick_params(labelsize=fontsize-1,direction='in',top=True, which = 'both')
    aax2.minorticks_on()

    if slidemode:
        plt.savefig(mdir + 'EWcumul_specwizard_map_match_coldens_o7-o8_L0100N1504_27_test3.1-3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_mtx-NEWconv_branchini_etal_2009_cen_fang_2006_slide.png',format = 'png',bbox_inches='tight',dpi=300)
    else:
        plt.savefig(mdir + 'EWcumul_specwizard_map_match_coldens_o7-o8_L0100N1504_27_test3.1-3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox_sample3_snap_027_z000p101_mtx-NEWconv_branchini_etal_2009_cen_fang_2006.pdf',format = 'pdf',bbox_inches='tight')
