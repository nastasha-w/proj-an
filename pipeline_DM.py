
# -*- coding: utf-8 -*-
"""
pipeline for dispersion measure maps
"""

import numpy as np
import h5py
import sys
import os

import make_maps_v3_master as m3
import make_maps_opts_locs as ol

datadir_head = '/fred/oz071/abatten/EAGLE/'
anlsdir_head = '/fred/oz071/abatten/ADMIRE_ANALYSIS/'

# parameters to read in from a parameter file, and the data types to convert 
# them to

def parse_parameterfile(filename, head='Projection'):
    '''
    filename: name of the parameter file
    head:     section heading of the parameter file to read in
              (Note that this function is written for the projection and 
              addition steps specifically, so non-default values are basically
              just if you want to call the section in the parameter file 
              something else.)
    
    parameter names and header names or not case or white-space sensitive. The
    last instance in the file is what is used. The values of most of the 
    parameters are not checked here beyond whether non-string types are 
    correct, since those checks are done by make_map when generating the list 
    of projections to run 
    
    expected values and parameters:
    SimName:        e.g. L0100N1504. If longer than 10 characters, the first 
                    part is used for the SimVar value
    SimVar:         Ref or Recal. Can also be specified as the first part of 
                    SimName. (Note that specifiying both ways may have 
                    unpredictable results.)
    SnapNum:        snapshot number (integer)
    
    EOS:            what to do with star-forming gas. This is specified at the 
                    level of the electrons (meaning different options for ions
                    and elements). Options are  
                    T4EOS, Neutral_ISM, Ionised_ISM
                    default: T4EOS
    ProjectionAxis: x, y, or z 
                    default: z
    NumPixels:      integer; default 32000 * box size / 100
    NumSl:          number of slices along the line of sight to use in the 
                    intial projection. Won't matter for the final result beyond
                    fp addition errors, but can be important for the larger 
                    boxes so all needed data fits on a node. Basically a 
                    parallelisation strategy at the cost of more memory use for
                    intermediate steps.
    OutputDirIons:  output directory for the ion column density maps. Uses a 
                    default based on SimName, SnapNum, SimVar, EOS if not 
                    specified
    OutputDirElectrons:   output directory for the electron column density maps. 
                    Uses a default based on SimName, SnapNum, SimVar, EOS if 
                    not specified
    Header:         folder in the electron hdf5 file with the header info
                    default: Header
                    (the header is always called Header in the ion files)
    Dataset:        dataset in the electron hdf5 file with the map
                    default: map
                    (the dataset is always called map in the ion files)
    
                    
                    
    '''
    
    # head -> all lowercase part between []
    reqparams = {'simnum', 'snapnum', 'var', 'numsl', 'numpix', 'ismopt',\
                 'axis', 'outputdir_ions', 'outputdir_electrons',\
                 'hedname', 'mapname'}
    params_nodefault = {'simnum', 'snapnum',  'numsl'}
    
    head = head.strip()
    if head[0] == '[':
        head  = head[1:]
    if head[-1] == ']':
        head = head[:-1]
    head = head.strip()
    head = head.lower()
    
    with open(filename, 'r') as pf:
        rightsection = False
        paramdct = {}
        
        for line in pf:
            line = line.split('#')[0] # strip off/ignore comments
            line = line.strip() # remove leading and trailing spaces and tabs
            if line == '': # ignore blank lines
                continue
            
            # start of a section for a new projcessing step?
            if line[0] ==  '[' and line[-1] == ']':
                ishead = True
            else:
                ishead = False
            
            # skip lines in other section
            if (not ishead) and (not rightsection):
                continue
            
            # set whether to read the parameters from this section 
            if ishead:
                # line -> all lowercase part between []
                line = line[1:-1]
                line = line.strip()
                line = line.lower()
                
                if line == head:
                    rightsection = True
                else:
                    rightsection = False
                continue
            
            # we're in the right section, the line isn't blank, let's read in parameters
            parts = tuple(line.split(':'))
            if len(parts) != 2:
                raise ValueError('In parameter file %s, the following line did not match the expected name : value format\n\t%s'%(filename, line))
            name, val = tuple(parts)
            name = name.strip()
            val = val.strip()
            name = name.lower()
            
            if name == 'OutputDirIons'.lower():
                paramdct['outputdir_ions'] = val
            elif name == 'OutputDirElectrons'.lower():
                paramdct['outputdir_electrons'] = val
            elif name == 'SimName'.lower():
                if len(val) > 10:
                    paramdct['simnum'] = val[-10:]
                    paramdct['var'] = val[:-10]
                else:
                    paramdct['simnum'] = val
            elif name == 'SimVar'.lower():
                paramdct['var'] = val
            elif name == 'EOS'.lower():
                paramdct['ismopt'] = val
            elif name == 'ProjectionAxis'.lower():
                paramdct['axis'] = val.lower() 
            elif name == 'NumPixels'.lower():
                try: 
                    val = int(val)
                except ValueError:
                    raise ValueError('Falied to convert NumPixels value %s to an integer'%(val))
                paramdct['numpix'] = val 
            elif name == 'Header'.lower():
                paramdct['hedname'] = val
            elif name == 'Dataset'.lower():
                paramdct['mapname'] = val
            elif name == 'SnapNum'.lower():
                try: 
                    val = int(val)
                except ValueError:
                    raise ValueError('Falied to convert SnapNum value %s to an integer'%(val))
                paramdct['snapnum'] = val
            elif name == 'NumSl'.lower():
                try: 
                    val = int(val)
                except ValueError:
                    raise ValueError('Falied to convert NumSl value %s to an integer'%(val))
                paramdct['numsl'] = val
            else:
                print('Ignoring parameter %s; no use specified'%name)
                
    # set defaults for stuff not in the parameter file
    if not params_nodefault.issubset(set(paramdct.keys())):
        raise ValueError('Some parameters without defaults are not set in the parameter file.')
    
    if paramdct['var'] in ['REFERENCE', 'reference', 'Reference', 'Ref', 'REF', 'ref']:
        paramdct['var'] = 'REFERENCE'
    elif paramdct['var'] in ['RECALIBRATED', 'recalibrated', 'Recalibrated', 'Recal', 'recal', 'RECAL', 'REC', 'Rec', 'rec']:
        paramdct['var'] = 'RECALIBRATED'
        
    if 'axis' not in paramdct.keys():
        paramdct['axis'] = 'z'
    if 'var' not in paramdct.keys():
        paramdct['var'] = 'REFERENCE'
    if 'ismopt' not in  paramdct.keys():
        paramdct['ismopt'] = 'T4EOS'
    if 'outputdir_ions' not in paramdct.keys():
        base = '/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_%s/all_snapshot_data/maps/%s/ColDens/'
        if paramdct['var'] == 'REFERENCE':
            psimnum = 'Ref'
        elif  paramdct['var'] == 'RECALIBRATED':
            psimnum = 'Rec'
        else:
            raise NotImplementedError('Automatic output directory setting is not enabled for simulation variaion %s'%(paramdct['var']))
        simname = psimnum + paramdct['simnum']
        paramdct['outputdir_ions'] = base%(psimnum, paramdct['ismopt'])
    if 'outputdir_electrons' not in paramdct.keys():
        base = '/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_%s/all_snapshot_data/maps/%s/DM/'
        if paramdct['var'] == 'REFERENCE':
            psimnum = 'Ref'
        elif  paramdct['var'] == 'RECALIBRATED':
            psimnum = 'Rec'
        else:
            raise NotImplementedError('Automatic output directory setting is not enabled for simulation variaion %s'%(paramdct['var']))
        simname = psimnum + paramdct['simnum']
        paramdct['outputdir_electrons'] = base%(simname, paramdct['ismopt']) 
    if 'hedname' not in paramdct.keys():
        paramdct['hedname'] = 'Header'
    if 'mapname' not in paramdct.keys():
        paramdct['hedname'] = 'map'
    if 'numpixels' not in paramdct.keys():
        if paramdct['simnum'][1:5] == '0100':
            paramdct['numpix'] = 32000
        elif paramdct['simnum'][1:5] == '0050':
            paramdct['numpix'] = 16000
        elif paramdct['simnum'][1:5] == '0025':
            paramdct['numpix'] = 8000
        elif paramdct['simnum'][1:5] == '0012':
            paramdct['numpix'] = 4000
            
    #if not os.path.isdir(paramdct['outputdir_electrons']):
    #    raise ValueError('Output directory for electron column density maps (OutputDirElectrons) does not exist:\n\t%s'%(paramdct['outputdir_electrons']))
    #if not os.path.isdir(paramdct['outputdir_ions']):
    #    raise ValueError('Output directory for ion column density maps (OutputDirIons)does not exist:\n\t%s'%(paramdct['outputdir_ions']))
    
    if not set(paramdct.keys()) == reqparams:
        raise ValueError('Falied to read in all expected parameters from parameter file %s'%(filename))
    
    return paramdct

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
    elif simnum[:5] == 'L0050':
        boxsize = 50.
    elif simnum[:5] == 'L0025':
        boxsize = 25.
    elif simnum[:5] == 'L0012':
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
              'ptypeQ': None, 'parttype': '0',\
              'theta': 0.0, 'phi': 0.0, 'psi': 0.0, \
              'sylviasshtables': False,\
              'var': var, 'axis': axis,'log': True, 'velcut': False,\
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
              outputdir='', axis='z', ismopt='T4EOS', hedname='Header', mapname='map'):
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
    
    # retrive file names 
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
        hed_main = fo.create_group(hedname)
        
        for ion, factor in [('hydrogen', 1.), ('helium', 2.), ('hneutralssh', -1.), ('he1', -2.), ('he2', -1.)]:
            try:
                h5py.File(filenames_sum[ion], 'r')
            except IOError: # sum file does not exist yet -> add up subfiles
                total2 = np.zeros((numpix, ) * 2, dtype=np.float32)
                hed2 = fo.create_group('Header')
                with h5py.File(filenames_sum[ion], 'w') as ft:
                    for subfileind in range(len(files_ion[key])):
                        subfile = files_ion[key][subfileind]
                        with h5py.File(filenames_sum[ion], 'r') as fts:
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
                     ft.copy('Header', fo, name=hedname)
                     # overwrite what needs changing
                     ft[hedname].attrs.create('EOS', ismopt)
                     sl_orig = ft['Header'].attrs['SliceLength']
                     ft[hedname].attrs.create('SliceLength', sl_orig * len(files_ion[ion]))
                     
        ft.create_dataset(mapname, data=total)
        ft[mapname].attrs.create('max', np.max(total))
        ft[mapname].attrs.create('minfinite', np.min(total[np.isfinite(total)]))
        ft[mapname].attrs.create('Units', "cm**-2")
        ft[mapname].attrs.create('log', False)
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

    step_opts = ['project', 'add', 'checknum']
    if step not in step_opts:
        raise ValueError('step (argument 2) should be one of %s'%step_opts)
        

    
    ## project step: files needed for the different electron ISM options:
    # hydrogen, helium: sfgas=True (ionized and 10^4 K), sfgas=False (neutral ISM)
    #   for the total elements, a 'T4' option can be given, but this is the same as sfgas=True
    #   since total element column don't depend on gas temperature
    # hneutralssh, he1, he2: sfgas='T4' (10^4 K), sfgas=False (neutral and ionized ISM)

## if the script is called: arguments are parameterfile, step
if __name__ == '__main__':   
    main() 