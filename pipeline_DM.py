
# -*- coding: utf-8 -*-
"""
pipeline for dispersion measure maps
"""

import numpy as np
import h5py
import sys

import make_maps_v3_master as m3
import make_maps_opts_locs as ol

datadir_head = '/fred/oz071/abatten/EAGLE/'
anlsdir_head = '/fred/oz071/abatten/ADMIRE_ANALYSIS/'


def parse_parameterfile(filename, head):
    
    with open(filename, 'r') as pf:
        
    
    #DataDir : (str) The directory containing the *.npz files from the projections code
    #OutputDir (or outputdir) : (str) The path to the directory to place all non-plot outputs. I.e interpolation files.    
    #SimName : (str) The name of the simulation    
    #EOS : (str) The ISM equation of state used (I normally use T4EOS, but T4 is also fine)    
    #ProjectionAxis : (str) The axis that we are projecting    
    #MapzVals : (list) A list containing the redshifts of the projections. This was used because the .npz files didn’t contain any redshift data other than snapshot number. So I added this in manually.    
    #Boxsize : (float/str) The length of the box (in every section this is a float except [Conversion] where it is a string)    
    #NumPixels (or numpixels): (int) The number of pixels per side of the projection.
    #MapDir : (str) The path to the directory containing the EAGLE column density maps.    
    #masterdir : (str) The path to the directory containing the master interpolated hdf5 file. Usually the same as outputdir.    
    #Header (or header) : (str) The name of the attribute that contains the header information.    
    #Dataset (or dataset): (str) The name of the dataset that will contain the DM maps.




def make_map(simnum, snapnum, var, pqty, numsl, sliceind, numpix,\
             sfgas='T4', outputdir='./', axis='z', nameonly=False):
    '''
    wrapper for make_maps_v3_master -> make_map with a smaller option set
    
    pqty is the thing to project; options are: 
        'hydrogen', 'helium', 'hneutralssh', 'he1', 'he2'
        (these are the ones you'll need, anyway)
    sfgas: how to handle star-forming gas; option are:
        'T4', True, False ('only' and 'from' aren't relevant here)
    nameonly: just get the name of the output file, including the full path
    
    numsl, sliceind: slice the whole box into numsl slices, project slice 
        sliceind now (ints)
    '''
    
    #/fred/oz071/abatten/EAGLE/RefL0012N0188/RefL0012N0188/snapshot_027_z000p101
    m3.ol.ndir = outputdir # override the default in opts_locs to get a per-case directory (won't work that easily with the data directory since that one gets mangled around by three layers of scripts)
    
    if var in ['Ref', 'ref', 'reference', 'Reference', 'REFERENCE']:
        var = 'REFERENCE'
        varlabel = 'Ref'
    elif var in ['Recal', 'recal', 'Rec', 'rec', 'Recalibrated', 'recalibrated', 'RECALIBRATED']:
        var = 'RECALIBRATED'
        varlabel = 'Recal'
    else:
        raise ValueError('Invalid option %s for var'%var)
    
    simpart = (varlabel + simnum + '/') * 2
    datapath = datadir_head + simpart
    
    if simnum[:5] == 'L0100':
        boxsize = 100.
    elif simnum[:-5] == 'L0050':
        boxsize = 50.
    elif simnum[:-5] == 'L0025':
        boxsize = 25.
    elif simnum[:-5] == 'L0012':
        boxsize = 12.5
     else:
        raise ValueError('Invalid option %s for simnum'%simnum)
        
    centre = np.array((0.5 * boxsize,) * 3)
    L_x, L_y, L_z = (boxsize,) * 3
    
    if (not isinstance(numsl, int)) or (not isinstance(sliceind, int)):
        raise ValueError('numsl and sliceind must be integers')
        
    if axis == 'z':
        L_z = L_z / np.float(numslices) # split into numslices slices along projection axis
        centre[2] = centre[2] - (numslices + 1.) * L_z / 2. + sliceind * L_z  
    if axis == 'x':
        L_x = L_x / np.float(numslices) # split into numslices slices along projection axis
        centre[0] = centre[0] - (numslices + 1.) * L_x / 2. + sliceind * L_x  
    if axis == 'y':
        L_y = L_y / np.float(numslices) # split into numslices slices along projection axis
        centre[1] = centre[1] - (numslices + 1.) * L_y / 2. + sliceind * L_y
        
    # will need to modify read_eagle_files to not add 'data/' to the input directort path...
    kernel = 'C2'
    abunds = 'Pt'
    
    args = (simnum, snapnum, centre, L_x, L_y, L_z, numpix, numpix, 'coldens')
    kwargs = {'ionW': pqty, 'abundsW': abunds, 'quantityW': None, 'excludeSFRW': sfgas,\
              'ionQ': None, 'abundsQ': abunds, 'quantityQ': None, 'excludeSFRQ': None,\
              'ptypeQ': None, 'parttype'='0',\
              'theta': 0.0, 'phi': 0.0, 'psi': 0.0, \
              'sylviasshtables': False,\
              'var': var, 'axis': axis,'log': True, 'velcut'False,\
              'periodic': True, 'kernel': kernel, 'saveres': True,\
              'simulation':'eagle', 'LsinMpc': True,\
              'select': None, 'misc': None, 'halosel': None, 'kwargs_halosel': None,\
              'ompproj': True, 'numslices': None, 'hdf5': True}
    name = m3.make_map(*args, nameonly=True, **kwargs)
    if nameonly:
        return name
    retval = m3.make_map(*args, nameonly=False, **kwargs)
    
    with h5py.File(name, 'r+') as fo:
        
        # set attributes
        hed = fo['Header']
        hed.attrs.create('SimName', simnum)
        hed.attrs.create('Snapshot', snapnum)
        hed.attrs.create('Redshift', hed['cosmopars'].attrs['z'])
        hed.attrs.create('EOS', sfgas)
        hed.attrs.create('ProjectionAxis', axis)
        hed.attrs.create('Boxsize', '%.1f Mpc'%boxsize)
        hed.attrs.create('NumPixels', numpix)
        if axis == 'z':
            slice_length = L_z
        elif axis == 'x':
            slice_length = L_x
        elif axis == 'y':
            slice_length = L_y
        hed.attrs.create('SliceLength', slice_length)
        hed.attrs.create('CodeVersion', m3.version)
        hed.attrs.create('MetalAbundancesType', abunds)
        hed.attrs.create('KernelShape', kernel)
        
        fo['map'].attrs.create("Units", "cm**-2")
    return retval

def run_projection(parameterfile, index, checknum=False):
    
    
    
    make_map(simnum, snapnum, var, pqty, numsl, sliceind, numpix,\
             sfgas='T4', outputdir='', axis='z', nameonly=False)

def add_files(simnum, snapnum, var, numsl, numpix,\
              outputdir='', axis='z', ismopt='T4EOS'):
    '''
    ismopt: 'T4EOS', 'Neutral_ISM', 'Ionised_ISM'
    assumes the 'T4' option was not used for the total element columns
    '''
    if ismopt == 'T4EOS':
        sfgas_dct = {'hydrogen': True,\
                     'helium': True,\
                     'hneutralssh': 'T4',\
                     'he1': 'T4',\
                     'he2': 'T4'}
    elif ismopt == 'Neutral_ISM':
        sfgas_dct = {'hydrogen': False,\
                     'helium': False,\
                     'hneutralssh': False,\
                     'he1': False,\
                     'he2': False}
    elif ismopt == 'Ionised_ISM':
        sfgas_dct = {'hydrogen': True,\
                     'helium': True,\
                     'hneutralssh': False,\
                     'he1': False,\
                     'he2': False}
    
    files_ion = {key: [make_map(simnum, snapnum, var, key, numsl, sliceind, numpix,\
                       sfgas=sfgas_dct[key], outputdir=outputdir, axis=axis, nameonly=True)\
                       for sliceind in range(numsl)]\
                 for key in sfgas_dct.keys()}
    filenames_sum = {key: 'dummy' for key in sfgas_dct.keys()} # TODO: add real sum names
    filename_DM = 'dummy' # TODO: add real DM name
    
    # TODO: add Adam's Header Attributes to the sum and electron files
    total = np.zeros((numpix, ) * 2, dtype=np.float32)
    mainheadercopied = False
    with h5py.File(filename_DM, 'w') as fo:
        hed_main = fo.create_group('Header')
        
        for ion, factor in [('hydrogen', 1.), ('helium', 2.), ('hneutralssh', -1.), ('he1', -2.), ('he2', -1.)]:
            try:
                h5py.File(filenames_sum[ion], 'r')
            except IOError: # sum file does not exist yet -> add up subfiles
                total2 = np.zeros((numpix, ) * 2, dtype=np.float32)
                hed2 = fo.create_group('Header')
                with h5py.File(filenames_sum[ion], 'w') as ft:
                    for subfileind in range(len(files_ion[key])):
                        subfile = files_ion[key][subfileind]
                        h5py.File(filenames_sum[ion], 'r') as fts:
                            ft3.copy('Header', hed2, name='slice_%i'%subfileind)
                            ft['Header/slice_%i'%subfileind].attrs.create('original_filename', subfile)
                            sub2 = 10**np.array(ft3['map'])
                            total2 += sub2
                            del sub2
                    total2 = np.log10(total2)
                    ft.create_dataset('map', data=total2)
                    ft['map'].attrs.create('max', np.max(total2))
                    ft['map'].attrs.create('minfinite', np.min(total2[np.isfinite(total2)]))
                del total2
                
            with h5py.File(filenames_sum[ion], 'r') as ft:
                 ft.copy('Header', hed_main, name=ion)
                 fo['Header/%s'%ion].attrs.create('original_filename', filenames_sum[ion])
                 fo['Header/%s'%ion].attrs.create('multip_factor', factor)
                 sub = 10**np.array(ft['map'])
                 total += factor * sub
                 del sub
                 
                 if not mainheadercopied:
                     ft.copy('Header', fo, name='Header')
                     # overwrite what needs changing
                     ft['Header'].attrs.create('EOS', ismopt)
                     sl_orig = ft['Header'].attrs['SliceLength']
                     ft['Header'].attrs.create('SliceLength', sl_orig * len(files_ion[ion]))
                     
        ft.create_dataset('map', data=total)
        ft['map'].attrs.create('max', np.max(total))
        ft['map'].attrs.create('minfinite', np.min(total[np.isfinite(total)]))
        ft['map'].attrs.create('Units', "cm**-2")
        ft['map'].attrs.create('log', False)
    del total
    
def main():
    '''
    what to run if the script is called from the command line. 
    First arg is the parameterfile,
    second is the step, 
    third arg for the 'project' step can be 'checknum', which just returns 
      the number of new projections needed (checks how many files already 
      exist); this is useful for checking how to ste up a batch script
      an integer third arg inidcates which file for the total should be run in
      this instance: typically input a slurm array number or something
    '''
    parfile = sys.argv[1]
    step = sys.argv[2]

    step_opts = ['project', 'add']
    if step not in step_opts:
        raise ValueError('step (argument 2) should be one of %s'%step_opts)
        
    checknum = False
    if step == 'project' and len(sys.argv) > 3:
        if sys.argv[3] == 'checknum':
            checknum = True
        else:
            raise ValueError('third argument option %s unknown'%(sys.argv[3]))
    
    ## project step: files needed for the different electron ISM options:
    # hydrogen, helium: sfgas=True (ionized and 10^4 K), sfgas=False (neutral ISM)
    #   for the total elements, a 'T4' option can be given, but this is the same as sfgas=True
    #   since total element column don't depend on gas temperature
    # hneutralssh, he1, he2: sfgas='T4' (10^4 K), sfgas=False (neutral and ionized ISM)

## if the script is called: arguments are parameterfile, step
 if __name__ == '__main__':   
    main() 