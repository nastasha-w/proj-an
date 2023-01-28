# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:46:28 2016

@author: wijers

Helper file for make_maps: containts lists, directories, etc. used in emission calculations
Note: does not actually work on my laptop! Just to keep spyder from complaining
"""


import string as string
#import h5py
#import numpy as np

import ion_header as ion
import eagle_constants_and_units as c

# example call:
# mmap.emission_calc([6.25,6.25,6.25],12.5,12.5,2.5,800,800,'carbon',49,log=True,Wfig='test_CV_Sb_L0012N0188_snap28_zslice_0p2_box_center_GadSm_SmAb', Wtitle='Surface brightness', Wmin=-10, Wmax=None, Wlabel=r'$S_B$ [photons $s^{-1}$ $cm^{-2}$ $sr^{-1}$]', Wcol='gray',periodic = True, kernel = 'gadget')

#######################
#      choices        #
#######################
  
file_type = '' #'snap'
simdir_eagle= '' #'/Users/Nastasha/phd/eagle_testdata/%s/' # contains e.g. L0100N1504 directories
simdir_bahamas = '' #'/disks/galform11/BAHAMAS/AGN_TUNED_nu0_%s_WMAP9/' 
simdir_eagle_noneq = '' #'/net/galaxy.strw.leidenuniv.nl/data2/oppenheimer/L025box/data_L025N0752/'
simdir_ceaglehydrangea = '' #'/net/quasar/data3/Hydrangea/CE-%i/HYDRO/' #'/disks/eagle/C-EAGLE/barnesRuns/PhaseI/halo_%02i/'
npzdir = '' #'/Users/Nastasha/phd/sim_maps/'
imgdir = '' #'/Users/Nastasha/phd/imgs/'


ndir_old = npzdir
ndir = '' #'/Users/Nastasha/phd/sim_maps/'
mdir = imgdir
pdir = '' #'/Users/Nastasha/phd/sim_proc/'
pdir2 = '' #'/Users/Nastasha/phd/sim_proc/'
sdir = '' #'/net/luttero/data2/specwizard_data/'


############################
#    setup and functions   #
############################
frontera_work = '/work2/08466/tg877653/frontera/'
frontera_scratch = '/scratch1/08466/tg877653/'
pre = frontera_scratch

c_interpfile  = pre + 'code/proj-an-c/interp2d/interp.so'
c_gridcoarser = '' #'/home/wijers/gridcoarser/gridcoarser.so'
c_halomask    = '' #'/home/wijers/halomaps/gethalomap.so'
hsml_dir = pre + 'code/proj-an-c/HsmlAndProject_OMP/'
# redhift table monotonically increasing, starting at 0
#dir_iontab = '/disks/strw17/serena/IonizationTables/HM01G+C/%s'
dir_iontab = '' #'/Users/Nastasha/phd/tables/ionbal/HM01G+C/%s'
dir_emtab = '' #'/Users/Nastasha/phd/tables/lineem/lines/%s/'
dir_coolingtab = '' #'/disks/eagle/BG_Tables/CoolingTables/'
emtab_sylvia_ssh = pre + 'iontab/PS20/UVB_dust1_CR1_G1_shield1_lines.hdf5' 
#'/net/luttero/data2/iontables_ssh_sylvia_2018_10_11/UV_dust1_CR0_G0_shield1.hdf5'
iontab_sylvia_ssh = pre + 'iontab/PS20/UVB_dust1_CR1_G1_shield1.hdf5'
dir_iontab_ben_gadget2 = '' #'/net/virgo/data2/oppenheimer/ionfiles/Eagle/'
simdir_fire = '/scratch3/01799/phopkins/fire3_suite_done/'

kernel_list = ['C2','gadget']
# desngb = 58 read out from sample hdf5 file (RunTimePars)
desngb = 58

# must be monotonic
zopts = \
['0.0000','0.1006','0.2709','0.3988','0.4675','0.7778','0.9567','1.2590',\
'1.4870','1.7370','2.0130','2.3160','2.4790','2.8290','3.0170','3.2140',\
'3.4210','3.6380','4.1050','4.6190','4.8950','5.4880']
elements = ['sulfur','silicon','oxygen','nitrogen','neon','magnesium','iron',\
'hydrogen','helium','carbon','calcium']
ions = ['al1', 'al2', 'al3',\
        'c1', 'c2', 'c3', 'c4', 'c5', 'c6',\
        'fe2', 'fe3', 'fe17',\
        'h1',\
        'he1', 'he2',\
        'mg1', 'mg2',
        'n2', 'n3', 'n4', 'n5', 'n6', 'n7',\
        'ne8', 'ne9', 'ne10',\
        'o1', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8',\
        's5',\
        'si2', 'si3', 'si4', 'si13']
        
zpoints = [float(z) for z in zopts] 

# copied from Parameters/ChemicalElements in simulation hdf5 files. 
# Seems to be the same accross simulations (and it should be, if the same cooling tabels are used) 
# matches Wiersema, Schaye, Theuns et al. 2009 table 1 values
solar_abunds_ea = {'calcium':  6.435500108636916E-5,\
                'carbon':   0.002066543558612466,\
                'helium':   0.2805553376674652,\
                'hydrogen': 0.706497848033905,\
                'iron':     0.0011032151523977518,\
                'magnesium':5.907064187340438E-4,\
                'neon':     0.0014144604792818427,\
                'nitrogen': 8.356256294064224E-4,\
                'oxygen':   0.00549262436106801,\
                'silicon':  6.825873861089349E-4,\
                'sulfur':  4.0898521547205746E-4}
# Serena Bertone's element tables use different values (copied from calcium file): 
# 'element solar abundances in Cloudy. Number density relative to H.'
solar_abunds_sb = {'calcium':  2.290866E-6,\
                'carbon':   2.4547108E-4,\
                'helium':   0.077624775,\
                'hydrogen': 1.0,\
                'iron':     2.8183817E-5,\
                'magnesium':3.467368E-5,\
                'neon':     1.0E-4,\
                'nitrogen': 8.511386E-5,\
                'oxygen':   4.8977835E-4,\
                'silicon':  3.467368E-5,\
                'sulfur':  1.8197019E-5}
# ALA01 = Allende Prieto, Lambert & Asplund (2001)
# ALA02 = Allende Prieto, Lambert & Asplund (2002) 
# H01 = Holweger (2001)
# see Hazy 7.02 for other line's sources
sources_abunds_sb = {\
                'carbon':   'AP02',
                'iron':     'H01',
                'magnesium':'H01',
                'neon':     'H01',
                'nitrogen': 'H01',
                'oxygen':   'AP01',
                'silicon':  'H01',
                }
# use bertone abundances for comparison with table, converted from number density relative to hydrogen to mass fraction
solar_elts = solar_abunds_sb.keys()
totdens = sum(ion.atomw[string.capwords(elt)]*solar_abunds_sb[elt] for elt in solar_elts)
def abundconv(elt):
    return ion.atomw[string.capwords(elt)]*solar_abunds_sb[elt]/totdens
solar_abunds = {elt: abundconv(elt) for elt in solar_elts}

Zsun_sylviastables = 0.013371374 # Sylvia's 2018 tables, from Grevesse, Asplund, Sauval, and Scott (2010, APandSS, 328, 179)
Zsun_ea = 0.0127 # Wiersma, Schaye,Theuns et al. (2009), consistent with Serena's tables

elements_ion = {'c1': 'carbon', 'c2': 'carbon', 'c3': 'carbon',\
                'c4': 'carbon', 'c5': 'carbon', 'c6': 'carbon',\
                'c5r': 'carbon',\
                'fe2': 'iron', 'fe3': 'iron',\
                'fe15': 'iron', 'fe16': 'iron',\
                'fe17': 'iron', \
                'fe18': 'iron', 'fe17-other1': 'iron', 'fe19': 'iron',\
                'h1': 'hydrogen', 'h2': 'hydrogen', 'lyalpha': 'hydrogen',\
                'halpha': 'hydrogen', 'h1ssh': 'hydrogen',\
                'hmolssh': 'hydrogen', 'hneutralssh': 'hydrogen',\
                'he1': 'helium', 'he2': 'helium',\
                'mg1': 'magnesium', 'mg2': 'magnesium', 'mg10': 'magnesium',\
                'mg12': 'magnesium',\
                'mg11r': 'magnesium',\
                'n2': 'nitrogen', 'n3': 'nitrogen', 'n4': 'nitrogen',\
                'n5': 'nitrogen', 'n6': 'nitrogen', 'n7': 'nitrogen',\
                'n6r': 'nitrogen', 'n6-actualr': 'nitrogen',\
                'ne4': 'neon', 'ne8': 'neon', 'ne9': 'neon', 'ne10': 'neon',\
                'ne9r': 'neon',\
                'o1': 'oxygen', 'o2': 'oxygen', 'o3': 'oxygen', 'o4': 'oxygen',\
                'o5': 'oxygen', 'o6': 'oxygen', 'o7': 'oxygen', 'o8': 'oxygen',\
                'o7r': 'oxygen', 'o7ix': 'oxygen', 'o7iy': 'oxygen',\
                'o7f': 'oxygen', 'o6major': 'oxygen', 'o6minor': 'oxygen',\
                's5': 'sulfur', 's15r': 'sulfur',\
                'si2': 'silicon', 'si3': 'silicon', 'si4': 'silicon',\
                'si9': 'silicon', 'si10': 'silicon', 'si11': 'silicon',\
                'si12': 'silicon', 'si13': 'silicon',\
                'si13r': 'silicon', 'si14': 'silicon',\
                }

ion_list_bensgadget2tables = ['h1', 'he1', 'he2',\
                              'c2', 'c3', 'c4', 'c5', 'c6',\
                              'n4', 'n5', 'n6',\
                              'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8',\
                              'ne4', 'ne8', 'ne9', 'mg2', 'mg10',\
                              'al2', 'al3',\
                              'si2', 'si3', 'si4', 'si12',\
                              'fe2',\
                              ]
# HydrogenI 
# HeliumI   HeliumII  
# CarbonII  CarbonIII CarbonIV  CarbonV CarbonVI  
# NitrogeIV NitrogenV NitrogeVI 
#OxygenI   OxygenII  OxygenIII OxygenIV  OxygenV   OxygenVI  OxygenVII OxygeVIII 
#NeonIV    NeonVIII  NeonIX
#MagnesiII MagnesiuX 
#AluminuII AluminIII 
# SiliconII SilicoIII SiliconIV SilicoXII  
# IronII


# because of course different files use different spellings of sulphur/sulfur
# 'standard' names to cooling table names
eltdct_to_ct ={\
          'hydrogen': 'Hydrogen',\
          'helium':   'Helium',\
          'carbon':   'Carbon',\
          'iron':     'Iron',\
          'magnesium':'Magnesium',\
          'nitrogen': 'Nitrogen',\
          'neon':     'Neon',\
          'oxygen':   'Oxygen',\
          'sulfur':   'Sulphur',\
          'sulphur':  'Sulphur',\
          'silicon':  'Silicon',\
          'calcium':  'Calcium'}

eltdct_from_ct ={\
          'Hydrogen': 'Hydrogen',\
          'Helium':   'Helium',\
          'Carbon':   'Carbon',\
          'Iron':     'Iron',\
          'Magnesium':'Magnesium',\
          'Nitrogen': 'Nitrogen',\
          'Neon':     'Neon',\
          'Oxygen':   'Oxygen',\
          'Sulphur':  'Sulfur',\
          'Silicon':  'Silicon',\
          'Calcium':  'Calcium'}


# for emission lines; not strictly ions (Serena Bertone and Freeke van de Voort line choices)   
# O VII triplet names: https://www.aanda.org/articles/aa/pdf/2015/07/aa26324-15.pdf 
# and "Metal-line emission from the WHIM: I. Soft X-rays" (Bertone et al. 2010) 
# intercombination x line from Bertone et al. energy and Kaastra et al. wavelength ratio         
# o6minor same 
# n6r: actually the forbidden line, not resonance (oops). n6-actualr is the resonance line  
line_nos_ion = {'c5r': 49,'c6': 56,
                'o6major': 119, 'o6minor': 120,
                'o7r': 136, 'o7ix': 137, 'o7iy': 138, 'o7f': 139,
                'o8':149,
                'n6r': 65, 'n6-actualr': 63, 'n7': 69,
                'ne9r': 169, 'ne10': 174,
                'mg11r': 176, 'mg12': 183 ,
                'si13r': 233,
                'halpha':9, 'lyalpha':1,
                'fe17': 594, 'fe18': 620, 'fe17-other1': 590, 'fe19': 662,
                }
line_eng_ion = {'c5r': 307.88*c.ev_to_erg,
                'c6': 367.47*c.ev_to_erg,
                'n6r': 419.86*c.ev_to_erg,
                'n6-actualr': c.planck * (c.c / (28.79 * 1e-8)),
                'n7': 500.24*c.ev_to_erg,
                'o6major': 12.01*c.ev_to_erg,
                'o6minor': 12.01 * c.ev_to_erg * (1031.9261 / 1037.6167),
                'o7r': 573.95*c.ev_to_erg,
                'o7ix': 568.81*c.ev_to_erg,
                'o7iy': 568.74*c.ev_to_erg,
                'o7f':  560.98*c.ev_to_erg,
                'o8': 653.55*c.ev_to_erg,
                'ne9r': 921.95 * c.ev_to_erg,
                'ne10': 1022.0 * c.ev_to_erg,
                'mg11r':  (c.c / (9.169 * 1e-8)) * c.planck,
                'mg12': 1471.8 * c.ev_to_erg,
                'si13r': 1864.9 * c.ev_to_erg,
                'halpha':1.89*c.ev_to_erg,
                'lyalpha':10.19*c.ev_to_erg,
                'fe17': 726.97*c.ev_to_erg,
                'fe18': c.planck * (c.c / (14.31 * 1e-8)),
                'fe17-other1': c.planck * (c.c / (15.10 * 1e-8)),
                'fe19': c.planck * (c.c / (13.46 * 1e-8)),
                } # eV
line_eng_ion_wrong = {'ne10': 921.95 * c.ev_to_erg}                                                                                                                    


snapzstr = {'024': '000p366',
            '025': '000p271',
            '026': '000p183',
            '027': '000p101',
            '028': '000p000'}

### C-EAGLE/Hydrangea
halos_hydrangea = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   18, 21, 22, 24, 25, 28, 29]
halos_ceagle = [17, 19, 20, 23, 26, 27]

#def get_halodata_ceh_z0(halonum, names):
#    filename = './c-eagle-hydrangea_cluster_data.txt'
#    separator = '\t'
#    halokey = 'CE-%s'%(str(halonum))
#    if isinstance(names, str):
#        names = [names]
#    with open(filename, 'r') as f:
#        for line in f:
#            if line[0] == '#': # first 2 lines: comments
#                continue
#            if line[:4] == 'Halo': # legend
#                line = line.strip
#                keys = np.array(line.split(separator))
#                inds = [np.where(name==keys)[0][0] for name in names] # which columns contain the data we want
#            elif line[:len(halokey)] == halokey: # line containing data for a halo
#                line = line.strip() # remove '\n'
#                columns = line.split(separator)
#                ret = [float(columns[ind]) if ind !=0 else str(columns[ind]) for ind in inds]
#                if len(ret) == 1:
#                    ret = ret[0]
#    return ret 
