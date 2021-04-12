#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:23:37 2021

@author: Nastasha

make the plots, and some of the tables, from paper 3: 
CGM soft X-ray emission in EAGLE
"""

import numpy as np
import pandas as pd
import h5py
import string

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.patches as mpatch
import matplotlib.ticker as ticker

import tol_colors as tc

import eagle_constants_and_units as c
import cosmo_utils as cu
import plot_utils as pu
import make_maps_opts_locs as ol
import ion_line_data as ild

# directories
mdir = '/net/luttero/data2/imgs/paper3/img_paper/' 
tmdir = '/net/luttero/data2/imgs/paper3/img_talks/'
ddir = '/net/luttero/data2/imgs/paper3/datasets/'
# add after line name for PS20 lines
siontab = '_iontab-PS20-UVB-dust1-CR1-G1-shield1_depletion-F' 

fontsize = 12

### lines: categories and plotting info
# for determining line categories
all_lines_SB = ['c5r', 'n6r', 'n6-actualr', 'ne9r', 'ne10', 'mg11r', 'mg12',
                'si13r', 'fe18', 'fe17-other1', 'fe19', 'o7r', 'o7ix', 'o7iy',
                'o7f', 'o8', 'fe17', 'c6', 'n7']
all_lines_PS20 = ['C  5      40.2678A', 'C  6      33.7372A',
                  'N  6      29.5343A', 'N  6      28.7870A',
                  'N  7      24.7807A', 'O  7      21.6020A',
                  'O  7      21.8044A', 'O  7      21.8070A',
                  'O  7      22.1012A', 'O  8      18.9709A',
                  'Ne 9      13.4471A', 'Ne10      12.1375A',
                  'Mg11      9.16875A', 'Mg12      8.42141A',
                  'Si13      6.64803A', 'Fe17      17.0510A',
                  'Fe17      15.2620A', 'Fe17      16.7760A',
                  'Fe17      17.0960A', 'Fe18      16.0720A',
                  ]

# names, colors, etc.
nicenames_lines_SB = {'c5r': 'C V',
                      'n6r': 'N VI (f)',
                      'n6-actualr': 'N VI',
                      'ne9r': 'Ne IX',
                      'ne10': 'Ne X',
                      'mg11r': 'Mg XI',
                      'mg12': 'Mg XII',
                      'si13r': 'Si XIII',
                      'fe18': 'Fe XVIII',
                      'fe17-other1': 'Fe XVII (15.10 A)',
                      'fe19': 'Fe XIX',
                      'o7r': 'O VII (r)',
                      'o7ix': 'O VII (ix)',
                      'o7iy': 'O VII (i)',
                      'o7f': 'O VII (f)',
                      'o8': 'O VIII',
                      'fe17': 'Fe XVII (17.05 A)',
                      'c6': 'C VI',
                      'n7': 'N VII',
                     }
nicenames_lines_PS20 = {'C  5      40.2678A': 'C V',
                        'C  6      33.7372A': 'C VI',
                        'N  6      29.5343A': 'N VI (f)',
                        'N  6      28.7870A': 'N VI',
                        'N  7      24.7807A': 'N VII',
                        'O  7      21.6020A': 'O VII (f)',
                        'O  7      21.8044A': 'O VII (i)',
                        'O  7      21.8070A': 'O VII (x or y)',
                        'O  7      22.1012A': 'O VII (r)',
                        'O  8      18.9709A': 'O VIII',
                        'Ne 9      13.4471A': 'Ne IX',
                        'Ne10      12.1375A': 'Ne X',
                        'Mg11      9.16875A': 'Mg XI',
                        'Mg12      8.42141A': 'Mg XII',
                        'Si13      6.64803A': 'Si XIII',
                        'Fe17      17.0510A': 'Fe XVII (17.05 A)' ,
                        'Fe17      15.2620A': 'Fe XVII (15.26 A)',
                        'Fe17      16.7760A': 'Fe XVII (16.78 A)',
                        'Fe17      17.0960A': 'Fe XVII (17.10 A)',
                        'Fe18      16.0720A': 'Fe XVIII',
                        } 
nicenames_lines = nicenames_lines_SB.copy()
nicenames_lines.update(nicenames_lines_PS20)

linematch_PS20 = {'C  5      40.2678A': 'c5r',
                  'C  6      33.7372A': 'c6',
                  'N  6      29.5343A': 'n6r',
                  'N  6      28.7870A': 'n6-actualr',
                  'N  7      24.7807A': 'n7',
                  'O  7      21.6020A': 'o7f',
                  'O  7      21.8044A': None,
                  'O  7      21.8070A': 'o7iy',
                  'O  7      22.1012A': 'o7r',
                  'O  8      18.9709A': 'o8',
                  'Ne 9      13.4471A': 'ne9r',
                  'Ne10      12.1375A': 'ne10',
                  'Mg11      9.16875A': 'mg11r',
                  'Mg12      8.42141A': 'mg12',
                  'Si13      6.64803A': 'si13r',
                  'Fe17      17.0510A': 'fe17',
                  'Fe17      15.2620A': None,
                  'Fe17      16.7760A': None,
                  'Fe17      17.0960A': None,
                  'Fe18      16.0720A': None,
                  }  
_matchedlines = {None if linematch_PS20[line] is None else line 
                 for line in linematch_PS20}
_matchedlines -= {None}
linematch_SB = {linematch_PS20[line]: line for line in _matchedlines}

def line_energy_ev_PS20(line):
    epart = float(line[4:-1].strip()) # wavelength (A for these lines)
    epart = c.planck * c.c / (epart * 1e-8) / c.ev_to_erg
    return epart

def line_energy_ev(line):
    if line in all_lines_PS20:
        return line_energy_ev_PS20(line)
    elif line in all_lines_SB:
        return ol.line_eng_ion[line] / c.ev_to_erg

atomnums = {'hydrogen': 1,
            'helium': 2,
            'carbon': 6,
            'nitrogen': 7,
            'oxygen': 8,
            'neon': 10,
            'magnesium': 12,
            'silicon': 14,
            'iron': 26,
            }
abbr_to_elt = {'H':  'hydrogen',
               'He': 'helium',
               'C':  'carbon',
               'N':  'nitrogen',
               'O':  'oxygen',
               'Ne': 'neon',
               'Mg': 'magnesium',
               'Si': 'silicon',
               'Fe': 'iron'
               }
elt_to_abbr = {abbr_to_elt[key]: key for key in abbr_to_elt}

parentelts = {line: abbr_to_elt[line[:2].strip()] for line in all_lines_PS20}
parentelts.update({line: ol.elements_ion[line] for line in all_lines_SB})

# for setting plotting defaults
plot_lines_SB = ['c5r', 'c6', 'n6-actualr', 'n7', 'o7r', 'o7iy', 'o7f', 'o8',
                 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r']
plot_lines_PS20 = ['Fe17      15.2620A', 'Fe17      16.7760A',
                   'Fe17      17.0510A', 'Fe17      17.0960A', 
                   'Fe18      16.0720A']
plot_lines_default = plot_lines_SB + plot_lines_PS20
plot_lines_default = sorted(plot_lines_default, key=line_energy_ev)

# colors
_c1 = tc.tol_cset('bright') # tip: don't use 'wine' and 'green' in the same plot (muted scheme)

linesets = [['c5r', 'n6-actualr', 'o7r', 'ne9r', 'mg11r', 'si13r'],
            ['c6', 'n7', 'o8', 'ne10', 'mg12'],
            ['o7r', 'o7iy', 'o7f'],
            ['Fe17      15.2620A', 'Fe17      16.7760A', 'Fe17      17.0510A',
             'Fe17      17.0960A', 'Fe18      16.0720A'],
            ]
lineset_names = ['He $\\alpha$ (r)',
                 'K $\\alpha$',
                 'O VII He $\\alpha$',
                 'Fe L-shell',
                 ]
lineargs_sets =\
        {'c5r':  {'linestyle': 'solid',   'color': _c1.blue},
         'c6':   {'linestyle': 'dashed',  'color': _c1.blue},
         'n6-actualr':  {'linestyle': 'dotted',  'color': _c1.cyan},
         'n7':   {'linestyle': 'dashed',  'color': _c1.cyan},
         'o7r':  {'linestyle': 'solid',   'color': _c1.green},
         'o7iy': {'dashes': [6, 2],       'color': _c1.yellow},
         'o7f':  {'linestyle': 'dotted',  'color': _c1.red},
         'o8':   {'linestyle': 'dashed',  'color': _c1.green},
         'ne9r':  {'linestyle': 'solid',  'color': _c1.yellow},
         'ne10':  {'linestyle': 'dashed', 'color': _c1.yellow},
         'mg11r': {'linestyle': 'solid',  'color': _c1.red},
         'mg12':  {'linestyle': 'dashed', 'color': _c1.red},
         'si13r': {'linestyle': 'solid',  'color': _c1.purple},
         'Fe17      15.2620A': {'dashes': [6, 2], 'color': _c1.green},
         'Fe17      16.7760A': {'dashes': [2, 2, 2, 2], 'color': _c1.blue},
         'Fe17      17.0510A': {'linestyle': 'dotted',  'color': _c1.yellow},
         'Fe17      17.0960A': {'linestyle': 'dashdot', 'color': _c1.red},
         'Fe18      16.0720A': {'linestyle': 'solid', 'color': _c1.purple}, 
         }

# reference points: table data
# table point with max T value (using EAGLE solar Z and element abundance)
line_Tmax = {'c5r':    5.95,
              'c6':    6.15,
              'n6r':   6.15,
              'n6-actualr': 6.15,
              'n7':    6.3,
              'o7r':   6.3,
              'o7ix':  6.35,
              'o7iy':  6.35,
              'o7f':   6.35,
              'o8':    6.5,
              'ne9r':  6.6,
              'ne10':  6.8,
              'mg11r': 6.8,
              'mg12':  7.0,
              'si13r': 7.0,
              'fe17':  6.75,
              'fe17-other1': 6.8,
              'fe18':  6.9,
              'fe19':  7.0,
              'C  5      40.2678A': 6.0,
              'C  6      33.7372A': 6.1,
              'N  6      29.5343A': 6.1,
              'N  6      28.7870A': 6.2,
              'N  7      24.7807A': 6.3,
              'O  7      21.6020A': 6.3,
              'O  7      21.8044A': 6.3,
              'O  7      21.8070A': 6.3,
              'O  7      22.1012A': 6.3,
              'O  8      18.9709A': 6.5,
              'Ne 9      13.4471A': 6.6,
              'Ne10      12.1375A': 6.8,
              'Mg11      9.16875A': 6.8,
              'Mg12      8.42141A': 7.0,
              'Si13      6.64803A': 7.0,
              'Fe17      17.0510A': 6.7,
              'Fe17      15.2620A': 6.8,
              'Fe17      16.7760A': 6.7,
              'Fe17      17.0960A': 6.7,
              'Fe18      16.0720A': 6.8,
              }
# Trange: rounded to 0.1 dex, copied by hand from table
line_Trange = {'c5r':   (5.7, 6.3),
               'c6':    (5.9, 6.8),
               'n6r':   (5.9, 6.5),
               'n6-actualr': (5.9, 6.5),
               'n7':    (6.1, 7.0),
               'o7r':   (6.0, 6.7),
               'o7ix':  (6.0, 6.7),
               'o7iy':  (6.0, 6.7),
               'o7f':   (6.0, 6.7),
               'o8':    (6.2, 7.2),
               'ne9r':  (6.3, 7.0),
               'ne10':  (6.5, 7.5),
               'mg11r': (6.4, 7.2),
               'mg12':  (6.7, 7.8),
               'si13r': (6.6, 7.4),
               'fe17':  (6.4, 7.1),
               'fe17-other1': (6.4, 7.1),
               'fe18':  (6.6, 7.1),
               'fe19':  (6.7, 7.2),
               'C  5      40.2678A': (5.7, 6.3),
               'C  6      33.7372A': (5.9, 6.8),
               'N  6      29.5343A': (5.8, 6.5),
               'N  6      28.7870A': (5.9, 6.5),
               'N  7      24.7807A': (6.1, 7.0),
               'O  7      21.6020A': (6.0, 6.6),
               'O  7      21.8044A': (6.0, 6.6),
               'O  7      21.8070A': (6.0, 6.6),
               'O  7      22.1012A': (6.0, 6.7),
               'O  8      18.9709A': (6.2, 7.2),
               'Ne 9      13.4471A': (6.2, 7.0),
               'Ne10      12.1375A': (6.5, 7.5),
               'Mg11      9.16875A': (6.4, 7.2),
               'Mg12      8.42141A': (6.7, 7.7),
               'Si13      6.64803A': (6.6, 7.4),
               'Fe17      17.0510A': (6.3, 7.0),
               'Fe17      15.2620A': (6.4, 7.0),
               'Fe17      16.7760A': (6.4, 7.0),
               'Fe17      17.0960A': (6.3, 7.0),
               'Fe18      16.0720A': (6.5, 7.1),
                }

### galaxies 
mass_edges_standard = (11., 11.5, 12.0, 12.5, 13.0, 13.5, 14.0)


def getsamplemedianmass():
    mass_edges = mass_edges_standard
    galdataf = mdir + '3dprof/' + 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    galdata_all = pd.read_csv(galdataf, header=2, sep='\t', index_col='galaxyid')
    
    masses = np.log10(np.array(galdata_all['M200c_Msun']))
    minds = np.digitize(masses, mass_edges)
    meddct = {mass_edges[i]: np.median(masses[minds == i + 1])\
              for i in range(len((mass_edges)))}
    return meddct
# from getsamplemedianmass
medianmasses = {11.0: 11.197684653627299,\
                11.5: 11.693016261428347,\
                12.0: 12.203018243950218,\
                12.5: 12.69846038894407,\
                13.0: 13.176895227999415,\
                13.5: 13.666537415167888,\
                14.0: 14.235991474257528,\
                }
    
### some default parameters, not always stored in the data files
# units and cosmology
rho_to_nh = 0.752 / (c.atomw_H * c.u)
arcmin_to_rad = np.pi / (180. * 60.)

cosmopars_eagle = {'omegab': c.omegabaryon,
                   'omegam': c.omega0,
                   'omegalambda': c.omegalambda,
                   'h': c.hubbleparam,
                  }
cosmopars_27 = {'a': 0.9085634947881763,
                'boxsize': 67.77,
                'h': 0.6777,
                'omegab': 0.0482519,
                'omegalambda': 0.693,
                'omegam':  0.307,
                'z': 0.10063854175996956,
                }

# instruments (x limits of the SB - impact parameter plot)
res_arcsec = {'Athena X-IFU': 5.,
              'Athena WFI':  3.,
              'Lynx PSF':    1.,
              'Lynx HDXI pixel':  0.3,
              }
fov_arcmin = {'Athena X-IFU': 5.,
              'Athena WFI':  40.,
              'Lynx HDXI':  22.,
              'Lynx X-ray microcalorimeter':  5.,
              }

def pkpc_to_arcmin(r_pkpc, z, cosmopars=None):
    da = cu.ang_diam_distance_cm(z, cosmopars=cosmopars) 
    angle = r_pkpc * c.cm_per_mpc * 1e-3 / da
    return angle / arcmin_to_rad

def arcmin_to_pkpc(angle_arcmin, z, cosmopars=None):
    da = cu.ang_diam_distance_cm(z, cosmopars=cosmopars) 
    r_pkpc = angle_arcmin * arcmin_to_rad  * da / (1e-3 * c.cm_per_mpc)
    return r_pkpc


### misc utils
def getoutline(linewidth):
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"),\
               mppe.Stroke(linewidth=linewidth + 0.5, foreground="white"),\
               mppe.Normal()]
    return patheff

def add_cbar_mass(cax, massedges=mass_edges_standard,\
             orientation='vertical', clabel=None, fontsize=fontsize, aspect=10.):
    '''
    returns color bar object, color dictionary (keys: lower mass edges)
    '''
    massedges = np.array(massedges)
    clist = tc.tol_cmap('rainbow_discrete', 
                        lut=len(massedges))(np.linspace(0.,  1., len(massedges)))
    keys = sorted(massedges)
    colors = {keys[i]: clist[i] for i in range(len(keys))}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist[:-1])
    cmap.set_over(clist[-1])
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(massedges, np.array(massedges[-1] + 1.)),\
                                ticks=massedges,\
                                spacing='proportional', extend='max',\
                                orientation=orientation)
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    if clabel is not None:
        cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(aspect)
    
    return cbar, colors


### overview image plot
def plotstampzooms_overview():
    '''
    overview of the different emission lines on large scales
    '''
    lines = plot_lines_default
    # all -> recusive calls for all lines
    rsc_slice = 2.
    rsc_zsmall = 1.
    rsc_zbig = 1.
    line_focus = 'o8'
    lines_med = ['o7r', 'c5r', 'Fe17      17.0510A']
    sliceshift_y = 15. # zoom region overlaps edge -> shift y coordinates up
    
    filebase = ddir + 'stamps/' + \
               'emission_{line}_L0100N1504_27_test3.5_SmAb_C2Sm_32000pix' + \
               '_6.25slice_zcen21.875_z-projection_noEOS_stamps.hdf5'
    
    grn_slice = 'slice'
    grn_zsmall = 'zoom1_small'
    grn_zbig   = 'zoom1_big'
    groups_all = [grn_slice, grn_zsmall, grn_zbig]
    
    groups = {line: [grn_zsmall] for line in lines}
    groups[line_focus] = groups_all
        
    outname = mdir + 'emission_overview_SB-PS20_L0100N1504_27_test3.x_SmAb' +\
              '_C2Sm_32000pix_6.25slice_zcen21.875_z-projection_noEOS_stamps' 
    outname = outname.replace('.', 'p')
    outname = outname + '.pdf'
    
    minhalomass = 11.
    halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    
    marklength_slice = 10. #cMpc
    marklength_z = 2. # cMpc
    
    vmin = -12. # log10 photons / cm2 / s / sr 
    vtrans = -1.5
    vmax = 1.0
    scaler200 = 2. # show radii at this times R200c
    cmap = pu.paste_cmaps(['gist_yarg', 'plasma'], [vmin, vtrans, vmax],\
                          trunclist=[[0.0, 0.5], [0.35, 1.0]])

    maps = {}
    extents = {}
    depths = {}
    paxes = {}
    resolutions = {}
    cosmoparss = {}
    snapshots = {}
    for line in lines:
        maps[line] = {}
        extents[line] = {}
        depths[line] = {}
        paxes[line] = {}
        resolutions[line] = {}
        cosmoparss[line] = {}
        snapshots[line] = {}
                    
        filen = filebase.format(line=line)
        if line in ['ne10', 'n6-actualr']:
            filen = filen.replace('test3.5', 'test3.6')
        elif line in all_lines_PS20:
            filen = filen.replace('test3.5', 'test3.7')          
            filen = filen.replace(line, line.replace(' ', '-') + siontab)
        try:
            with h5py.File(filen, 'r') as ft:
                for grn in groups[line]:
                    if grn not in ft:
                        print('Could not find the group {grp} for {line}: {filen}.'.format(\
                              line=line, filen=filen, grp=grn))
                        continue
                    grp = ft[grn] 
                    maps[line][grn] = grp['map'][:]
                    cosmopars = {key: val for key, val in \
                        grp['Header_in/inputpars/cosmopars'].attrs.items()}
                    cosmoparss[line][grn] = cosmopars
                    L_x = grp['Header_in/inputpars'].attrs['L_x'] 
                    L_y = grp['Header_in/inputpars'].attrs['L_y']
                    L_z = grp['Header_in/inputpars'].attrs['L_z']
                    centre = np.array(grp['Header_in/inputpars'].attrs['centre'])
                    LsinMpc = bool(grp['Header_in/inputpars'].attrs['LsinMpc'])
                    Ls = np.array([L_x, L_y, L_z])
                    if not LsinMpc:
                        Ls /= cosmopars['h']
                        centre /= cosmopars['h']
                    axis = grp['Header_in/inputpars'].attrs['axis'].decode()
                    if axis == 'z':
                        axis1 = 0
                        axis2 = 1
                        axis3 = 2
                    elif axis == 'y':
                        axis1 = 2
                        axis2 = 0
                        axis3 = 1
                    elif axis == 'x':
                        axis1 = 1
                        axis2 = 2
                        axis3 = 0
                    paxes[line][grn] = (axis1, axis2, axis3)
                    extent_x = Ls[axis1]
                    extent_y = Ls[axis2]
                    snapshots[line][grn] = grp['Header_in/inputpars'].attrs['snapnum']
                    
                    if bool(grp['map'].attrs['subregion']):
                        extents[line][grn] = np.array([np.array(grp['map'].attrs['edges_axis0_cMpc']),\
                                                       np.array(grp['map'].attrs['edges_axis1_cMpc'])])
                    else:
                        extents[line][grn] = np.array([[0., extent_x],\
                                                       [0., extent_y]])
                    depths[line][grn] = (centre[axis3] - 0.5 * Ls[axis3], centre[axis3] + 0.5 * Ls[axis3]) 
                    resolutions[line][grn] = ((extents[line][grn][0][1] - extents[line][grn][0][0])\
                                               * 1e3 * cosmopars['a'] / maps[line][grn].shape[0],\
                                              (extents[line][grn][1][1] - extents[line][grn][1][0])\
                                               * 1e3 * cosmopars['a'] / maps[line][grn].shape[1])
                
        except IOError:
            print('Could not find the file for {line}: {filen}.'.format(\
                  line=line, filen=filen))
    
    print('Using map resolutions:\n{res}'.format(res=resolutions))
    #_lines = sorted(maps.keys(), key=ol.line_eng_ion.get)
    _lines = sorted(maps.keys(), key=line_energy_ev)
    
    # get halo catalogue data for overplotting
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    with h5py.File(halocat, 'r') as hc:
        snapnum = hc['Header'].attrs['snapnum']
        cosmopars = {key: val for key, val in hc['Header/cosmopars'].attrs.items()}
        if not np.all(snapnum == np.array([snapshots[line][grn] \
                                           for line in _lines for grn in groups[line]])):
            raise RuntimeError('Stamp snapshots do not match halo catalogue snapshot')
        masses = np.log10(hc['M200c_Msun'][:])
        radii = hc['R200c_pkpc'] / cosmopars['a'] * 1e-3
        pos = np.array([hc['Xcom_cMpc'][:],\
                        hc['Ycom_cMpc'][:],\
                        hc['Zcom_cMpc'][:]])
        msel = masses >= minhalomass
        masses = masses[msel]
        radii = radii[msel]
        pos = pos[:, msel]
    
    # this is all very fine-tuned by hand
    patheff_text = [mppe.Stroke(linewidth=2.0, foreground="white"),\
                            mppe.Stroke(linewidth=0.4, foreground="black"),\
                            mppe.Normal()]  
    
    panelsize_large = 3.
    panelsize_small = panelsize_large * 0.5
    panelsize_med = (panelsize_large + 3. * panelsize_small) / 4.
    margin = 0.2
    ncol_small = 4
    nrows_small = (len(lines) - len(lines_med) - 2) // ncol_small + 1
    
    figwidth =  2. * panelsize_large + 1. * panelsize_med + 2. * margin
    figheight = 1. * panelsize_large + nrows_small * panelsize_small + 2. * margin
    
    fig = plt.figure(figsize=(figwidth, figheight))
    
    axes = {line: {} for line in _lines}
    _ps0_l = panelsize_large / figwidth
    _ps1_l = panelsize_large / figheight
    _ps0_s = panelsize_small / figwidth
    _ps1_s = panelsize_small / figheight
    _ps0_m = panelsize_med / figwidth
    _ps1_m = panelsize_med / figheight
    _x0 = margin / figwidth
    _y0 = margin / figheight
    _x1 = 1. - _x0
    _y1 = 1. - _y0
    
    #[left, bottom, width, height]
    # focus lines: top row
    axes[line_focus][grn_zbig]  = fig.add_axes([_x0, _y1 - _ps1_l, _ps0_l, _ps1_l])
    axes[line_focus][grn_slice] = fig.add_axes([_x0 + _ps0_l, _y1 - _ps1_l, _ps0_l, _ps1_l])
    #axes[line_focus][grn_zsmall] = fig.add_axes([_x0 + 2. * _ps0_l, _y1 - _ps1_m, _ps0_m, _ps1_m])
    # right column: medium-panel lines
    for li, line in enumerate([line_focus] + lines_med):
        bottom = _y1 - (li + 1.) * _ps1_m
        axes[line][grn_zsmall] = fig.add_axes([_x0 + 2. * _ps0_l, bottom, _ps0_m, _ps1_m])
    # block: small panel lines
    slines = list(np.copy(_lines)) 
    for line in lines_med + [line_focus]:
        slines.remove(line)
    for li, line in enumerate(slines):
        col = li % ncol_small
        row = li // ncol_small
        left = _x0 + col * _ps0_s
        bottom = _y1 - _ps1_l - (row + 1.) * _ps1_s
        axes[line][grn_zsmall] = fig.add_axes([left, bottom, _ps0_s, _ps1_s])
    # lower right: color bars
    #_ht = _ps1_s
    #bottom = _y0
    #left = _x0 + (len(slines) % ncol_small) * _ps0_s
    #width =  0.5 * (_x1 - 3. * _x0 - left)
    #cax1  = fig.add_axes([left + 0.5 * _x0, bottom, width, _ht])
    #cax2  = fig.add_axes([left + width + _x0, bottom, width, _ht])
    # lower right: color bars
    
    
    clabel_img = '$\\log_{10} \\, \\mathrm{SB} \\; [\\mathrm{ph.} \\, \\mathrm{cm}^{-2} \\mathrm{s}^{-1} \\mathrm{sr}^{-1}]$'
    clabel_hmass = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    
    if ncol_small * nrows_small <= len(slines) + 2:
        clabel_over_bar = True 
        texth  = _y0
        _ht = (_ps1_s - _y0) * 0.5 - texth
        bottom = _y0
        left = _x0 + (len(slines) % ncol_small) * _ps0_s
        _width = _x1 - left - 2. * _x0
        width = min(_width, 2. * _ps0_s)
        left = left + 0.5 * (_width - width)
        cax1  = fig.add_axes([left + _x0, bottom + texth, width, _ht])
        cax2  = fig.add_axes([left + _x0, bottom + 2. * texth + _ht, width, _ht])
    else:
        clabel_over_bar = False
        _ht = _ps1_s
        bottom = _y0
        left = _x0 + (len(slines) % ncol_small) * _ps0_s
        width =  0.5 * (_x1 - 3. * _x0 - left)
        cax1  = fig.add_axes([left + 0.5 * _x0, bottom, width, _ht])
        cax2  = fig.add_axes([left + width + _x0, bottom, width, _ht])    
    
    c_aspect = 1./10.
        
    if clabel_over_bar:
        _cl = None
    else:
        _cl = clabel_hmass
    cbar, colordct = add_cbar_mass(cax2, massedges=mass_edges_standard,\
             orientation='horizontal', clabel=_cl, fontsize=fontsize,\
             aspect=c_aspect)
    if clabel_over_bar:
        cax2.text(0.5, 0.5, clabel_hmass, fontsize=fontsize,\
                  path_effects=patheff_text, transform=cax2.transAxes,\
                  verticalalignment='center', horizontalalignment='center')
    print('Max value in maps: {}'.format(max([np.max(maps[line][grn])\
          for line in _lines for grn in groups[line]])))
    
    cmap_img = cmap
    cmap_img.set_under(cmap_img(0.))
    cmap_img.set_bad(cmap_img(0.))
    
    for line in _lines:
        _groups = groups[line]
        for grn in _groups:
            ax = axes[line][grn]
                
            if grn == grn_slice:
                direction_big = 'left'
                direction_small = 'right'
                marklength = marklength_slice
                scaler200 = rsc_slice
            elif grn == grn_zsmall:
                marklength = marklength_z 
                scaler200 = rsc_zsmall
            elif grn == grn_zbig:
                marklength = marklength_z 
                scaler200 = rsc_zbig
            #labelbottom = li > len(_lines) - ncols - 1
            #labeltop = li < ncols 
            #labelleft = li % ncols == 0
            #labelright = li % ncols == ncols - 1
            #ax.tick_params(labelsize=fontsize - 1,  direction='in',\
            #               labelbottom=labelbottom, labeltop=labeltop,\
            #               labelleft=labelleft, labelright=labelright,\
            #               top=labeltop, left=labelleft, bottom=labelbottom,\
            #               right=labelright)
            #lbase = '{ax} [cMpc]'
            #axis1 = paxes[line][0]
            #axis2 = paxes[line][1]
            #axis3 = paxes[line][2]
            #if labelbottom:
            #    xl = lbase.format(ax=['X', 'Y', 'Z'][axis1])
            #    ax.set_xlabel(xl, fontsize=fontsize)
            #if labelleft:
            #    yl = lbase.format(ax=['X', 'Y', 'Z'][axis2])
            #    ax.set_ylabel(yl, fontsize=fontsize)
            ax.tick_params(top=False, bottom=False, left=False, right=False,
                          labeltop=False, labelbottom=False, labelleft=False,
                          labelright=False)
            
            posx = pos[axis1]
            posy = pos[axis2]
            posz = pos[axis3]
            
            if grn == grn_slice:
                _deltay = extents[line][grn][1][1] - extents[line][grn][1][0]
                _npixy = maps[line][grn].shape[1]
                nshift = int(sliceshift_y / _deltay * _npixy + 0.5)
                shift_y = nshift * (_deltay / _npixy)
                print('Shifting slice map by {shifty}'.format(shifty=shift_y))
                maps[line][grn] = np.roll(maps[line][grn], nshift)
                #posy = np.copy(posy)
                #posy -= sliceshift_y
                extents[line][grn][1][0] -= sliceshift_y
                extents[line][grn][1][1] -= sliceshift_y
            
            margin = np.max(radii)
            zrange = depths[line][grn]
            xrange = [extents[line][grn][0][0] - margin,\
                      extents[line][grn][0][1] + margin]
            yrange = [extents[line][grn][1][0] - margin,\
                      extents[line][grn][1][1] + margin]
                
            ax.set_facecolor(cmap_img(0.))    
            _img = maps[line][grn]
            _img[_img < vmin] = vmin # avoid weird outlines at img ~ vmin in image
            img = ax.imshow(_img.T, origin='lower', interpolation='nearest',\
                      extent=(extents[line][grn][0][0], extents[line][grn][0][1],\
                              extents[line][grn][1][0], extents[line][grn][1][1]),\
                      cmap=cmap_img, vmin=vmin, vmax=vmax) 
                      
            hsel = np.ones(len(posx), dtype=bool)
            cosmopars = cosmoparss[line][grn]
            boxsize = cosmopars['boxsize'] / cosmopars['h'] 
            hsel &= cu.periodic_sel(posz, zrange, boxsize)
            if xrange[1] - xrange[0] < boxsize:
                hsel &= cu.periodic_sel(posx, xrange, boxsize)
            if yrange[1] - yrange[0] < boxsize:
                hsel &= cu.periodic_sel(posy, yrange, boxsize)
            
            posx = posx[hsel]
            posy = posy[hsel]
            ms = masses[hsel]
            rd = radii[hsel]
            
            # add periodic repetitions if the plotted edges are periodic
            if xrange[1] - xrange[0] > boxsize - 2. * margin or\
               yrange[1] - yrange[0] > boxsize - 2. * margin:
               _p = cu.pad_periodic([posx, posy], margin, boxsize, additional=[ms, rd])
               
               posx = _p[0][0]
               posy = _p[0][1]
               ms = _p[1][0]
               rd = _p[1][1]
            elif xrange[0] < 0. or yrange[0] < 0. or\
                 xrange[1] > boxsize or yrange[1] > boxsize:
               _m = []
               if xrange[0] < 0.:
                   _m.append(np.abs(xrange[0]))
               if yrange[0] < 0.:
                   _m.append(np.abs(yrange[0]))    
               if xrange[1] > boxsize:
                   _m.append(xrange[1] - boxsize)
               if yrange[1] > boxsize:
                   _m.append(yrange[1] - boxsize)
               _margin = margin + max(_m)
               print(_margin)
               _p = cu.pad_periodic([posx, posy], _margin, boxsize, additional=[ms, rd])
               
               posx = _p[0][0]
               posy = _p[0][1]
               ms = _p[1][0]
               rd = _p[1][1]
            
            me = np.array(sorted(list(colordct.keys())) + [17.])
            mi = np.max(np.array([np.searchsorted(me, ms) - 1,\
                                  np.zeros(len(ms), dtype=np.int)]),\
                        axis=0)
            colors = np.array([colordct[me[i]] for i in mi])
            
            patches = [mpatch.Circle((posx[ind], posy[ind]), scaler200 * rd[ind]) \
                       for ind in range(len(posx))] # x, y axes only
        
            patheff = [mppe.Stroke(linewidth=1.2, foreground="black"),\
                           mppe.Stroke(linewidth=0.7, foreground="white"),\
                           mppe.Normal()] 
            collection = mcol.PatchCollection(patches)
            collection.set(edgecolor=colors, facecolor='none', linewidth=0.7,\
                           path_effects=patheff)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.add_collection(collection)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            
            ltext = nicenames_lines[line]
            ax.text(0.95, 0.95, ltext, fontsize=fontsize, path_effects=patheff_text,\
                    horizontalalignment='right', verticalalignment='top',\
                    transform=ax.transAxes, color='black')
    
            mtext = str(marklength)
            if mtext[-2:] == '.0':
                mtext = mtext[:-2]
            mtext = mtext + ' cMpc'
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xr = xlim[1] - xlim[0]
            yr = ylim[1] - ylim[0]
            if marklength > 2.5 * xr:
                print('Marklength {} is too large for the plotted range'.format(marklength))
                continue
            
            if line == line_focus and grn != grn_zsmall:
                y_this = _ps1_l
                x_this = _ps0_l
            elif line in lines_med + [line_focus]:
                y_this = _ps1_m
                x_this = _ps0_m
            else:
                y_this = _ps1_s
                x_this = _ps0_s
            xs = xlim[0] + 0.1 * xr * _ps0_l / x_this
            ypos = ylim[0] + 0.07 * yr # * _ps1_l / y_this
            xcen = xs + 0.5 * marklength
            
            patheff = [mppe.Stroke(linewidth=3.0, foreground="white"),\
                       mppe.Stroke(linewidth=2.0, foreground="black"),\
                       mppe.Normal()] 
            ax.plot([xs, xs + marklength], [ypos, ypos], color='black',\
                    path_effects=patheff, linewidth=2)
            ax.text(xcen, ypos + 0.01 * yr * _ps1_l / y_this,\
                    mtext, fontsize=fontsize,\
                    path_effects=patheff_text, horizontalalignment='center',\
                    verticalalignment='bottom', color='black')
            
            if grn in [grn_zsmall, grn_zbig]: # add dotted circles for haloes just over the slice edges
                posx = pos[axis1]
                posy = pos[axis2]
                posz = pos[axis3]
                margin = np.max(radii) # cMpc
                
                zrange1 = [depths[line][grn][1],  depths[line][grn][1] + margin]
                zrange2 = [depths[line][grn][0] - margin, depths[line][grn][0]]
                xrange = [extents[line][grn][0][0] - margin,\
                          extents[line][grn][0][1] + margin]
                yrange = [extents[line][grn][1][0] - margin,\
                          extents[line][grn][1][1] + margin]
                          
                hsel = np.ones(len(posx), dtype=bool)
                _hsel = cu.periodic_sel(posz, zrange1, boxsize)
                _hsel |=  cu.periodic_sel(posz, zrange2, boxsize)
                hsel &= _hsel
                if xrange[1] - xrange[0] < boxsize:
                    hsel &= cu.periodic_sel(posx, xrange, boxsize)
                if yrange[1] - yrange[0] < boxsize:
                    hsel &= cu.periodic_sel(posy, yrange, boxsize)
                    
                posx = posx[hsel]
                posy = posy[hsel]
                posz = posz[hsel]
                ms = masses[hsel]
                rd = radii[hsel]
                
                # haloes are not just generally close to the edges, but within r200c of them
                hsel =  np.abs((posz + 0.5 * boxsize - depths[line][grn][1]) \
                               % boxsize - 0.5 * boxsize) < rd
                hsel &= np.abs((posz + 0.5 * boxsize - depths[line][grn][0]) \
                               % boxsize - 0.5 * boxsize) < rd
                
                posx = posx[hsel]
                posy = posy[hsel]
                posz = posz[hsel]
                ms = ms[hsel]
                rd = rd[hsel]
                
                # add periodic repetitions if the plotted edges are periodic
                if xrange[1] - xrange[0] > boxsize - 2. * margin or\
                   yrange[1] - yrange[0] > boxsize - 2. * margin:
                   _p = cu.pad_periodic([posx, posy], margin, boxsize, additional=[ms, rd])
                   
                   posx = _p[0][0]
                   posy = _p[0][1]
                   ms = _p[1][0]
                   rd = _p[1][1]
                elif xrange[0] < 0. or yrange[0] < 0. or\
                     xrange[1] > boxsize or yrange[1] > boxsize:
                   _m = []
                   if xrange[0] < 0.:
                       _m.append(np.abs(xrange[0]))
                   if yrange[0] < 0.:
                       _m.append(np.abs(yrange[0]))    
                   if xrange[1] > boxsize:
                       _m.append(xrange[1] - boxsize)
                   if yrange[1] > boxsize:
                       _m.append(yrange[1] - boxsize)
                   _margin = margin + max(_m)
                   _p = cu.pad_periodic([posx, posy], _margin, boxsize, additional=[ms, rd])
                   
                   posx = _p[0][0]
                   posy = _p[0][1]
                   ms = _p[1][0]
                   rd = _p[1][1]
                
                me = np.array(sorted(list(colordct.keys())) + [17.])
                mi = np.max(np.array([np.searchsorted(me, ms) - 1,\
                                      np.zeros(len(ms), dtype=np.int)]),\
                            axis=0)
                colors = np.array([colordct[me[i]] for i in mi])
                
                patches = [mpatch.Circle((posx[ind], posy[ind]), scaler200 * rd[ind]) \
                           for ind in range(len(posx))] # x, y axes only
            
                patheff = [mppe.Stroke(linewidth=1.2, foreground="black"),\
                               mppe.Stroke(linewidth=0.7, foreground="white"),\
                               mppe.Normal()] 
                collection = mcol.PatchCollection(patches)
                collection.set(edgecolor=colors, facecolor='none', linewidth=0.7,\
                               path_effects=patheff, linestyle='dotted')
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                ax.add_collection(collection)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                    
            
            if grn == grn_slice:
                square_big = extents[line][grn_zbig]
                square_small = extents[line][grn_zsmall]
                lw_square = 1.2
                _patheff = [mppe.Stroke(linewidth=1.5, foreground="white"),\
                           mppe.Stroke(linewidth=lw_square, foreground="black"),\
                           mppe.Normal()] 
                
                _lx = [square_big[0][0], square_big[0][0], square_big[0][1],\
                       square_big[0][1], square_big[0][0]]
                _ly = [square_big[1][0], square_big[1][1], square_big[1][1],\
                       square_big[1][0], square_big[1][0]]
                ax.plot(_lx, _ly, color='black', path_effects=_patheff,\
                        linewidth=lw_square)
                _lx = [square_small[0][0], square_small[0][0], square_small[0][1],\
                       square_small[0][1], square_small[0][0]]
                _ly = [square_small[1][0], square_small[1][1], square_small[1][1],\
                       square_small[1][0], square_small[1][0]]
                ax.plot(_lx, _ly, color='black', path_effects=_patheff,\
                        linewidth=lw_square)
                
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                if direction_big == 'right':
                    xi = 1
                else:
                    xi = 0
                ax.plot([square_big[0][xi], xlim[xi]], [square_big[1][0], ylim[0]],\
                        color='black', path_effects=_patheff,\
                        linewidth=lw_square)
                ax.plot([square_big[0][xi], xlim[xi]], [square_big[1][1], ylim[1]],\
                        color='black', path_effects=_patheff,\
                        linewidth=lw_square)
                
                if direction_small == 'right':
                    xi = 1
                else:
                    xi = 0
                ylow = ylim[1] - (ylim[1] - ylim[0]) * (_ps1_m / _ps1_l)
                ax.plot([square_small[0][xi], xlim[xi]], [square_small[1][0], ylow],\
                        color='black', path_effects=_patheff,\
                        linewidth=lw_square)
                ax.plot([square_small[0][xi], xlim[xi]], [square_small[1][1], ylim[1]],\
                        color='black', path_effects=_patheff,\
                        linewidth=lw_square)
            elif line == line_focus:
                lw_square = 1.2
                _patheff = [mppe.Stroke(linewidth=1.5, foreground="white"),\
                           mppe.Stroke(linewidth=lw_square, foreground="black"),\
                           mppe.Normal()] 
                _lx = [xlim[0], xlim[0], xlim[1], xlim[1], xlim[0]]
                _ly = [ylim[0], ylim[1], ylim[1], ylim[0], ylim[0]]
                ax.plot(_lx, _ly, color='black', path_effects=_patheff,\
                        linewidth=lw_square)
                
    plt.colorbar(img, cax=cax1, orientation='horizontal', extend='both')
    if clabel_over_bar:
        cax1.text(0.5, 0.5, clabel_img, fontsize=fontsize,\
                  path_effects=patheff_text, transform=cax1.transAxes,\
                  verticalalignment='center', horizontalalignment='center')
    else:
        cax1.set_xlabel(clabel_img, fontsize=fontsize)
    cax1.tick_params(labelsize=fontsize - 1, which='both')
    cax1.set_aspect(c_aspect)   
    
    print('Halos indicated at {rs} x R200c'.format(rs=scaler200))

    if outname is not None:
        if '/' not in outname:
            outname = mdir + outname
        if outname[-4:] != '.pdf':
            outname = outname + '.pdf'
        plt.savefig(outname, format='pdf', bbox_inches='tight')
        
### emissivity curves
def plot_emcurves():
    '''
    emissivity as a function of temperature in CIE (nH = 10 cm**-3)
    '''
    
    z=0.1
    outname = mdir + 'emcurves_z{}_HM01_ionizedmu.pdf'.format(str(z).replace('.', 'p'))
    
    # eagle cosmology
    cosmopars = cosmopars_eagle.copy()
    cosmopars['z'] = z
    cosmopars['a'] = 1. / (1. + z)
        
    lineargs = lineargs_sets.copy()
    lineargs.update({'fe17': {'linestyle': 'solid',  'color': _c1.yellow}})
    lw = 2
    pe = getoutline(lw)
    
    _linesets = linesets.copy()
    _linesets[3] = ['fe17'] + _linesets[3]
    
    linelabels = nicenames_lines.copy()
    
    lines = [line for _l in _linesets for line in _l]
    lines_SB = {line if line in all_lines_SB else None for line in lines}
    lines_PS20 = {line if line in all_lines_PS20 else None for line in lines}
    lines_SB -= {None}
    lines_PS20 -= {None}
    lines_SB = list(lines_SB)
    lines_PS20 = list(lines_PS20)
    
    for line in lines:
        lkw = lineargs[line].copy()
        if 'dashes' in lkw:
            del lkw['dashes']
        if 'linestyle' in lkw:
            del lkw['linestyle']
        lineargs[line] = lkw
        
    fname_SB = 'emissivitycurves_z-{z}_nH-{nH}.txt'.format(z=z, nH='CIE')
    fname_PS20 = 'emissivitycurves_PS20_z-{z}_nH-{nH}.txt'.format(z=z, nH=1.0)
    #fname_PS20_2 = 'emissivitycurves_PS20_z-{z}_nH-{nH}.txt'.format(z=z, nH=6.0)
    
    fformat = {'sep':'\t', 'index_col':'T', 'comment':'#'}
    cdata_SB = pd.read_csv(ddir + fname_SB, **fformat)
    cdata_PS20 = pd.read_csv(ddir + fname_PS20, **fformat)
    #cdata_PS20_2 = pd.read_csv(ddir + fname_PS20_2, **fformat)
    
    ncols = 2
    nrows = (len(linesets) - 1) // ncols + 1
    fig = plt.figure(figsize=(11., 7.))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols,  hspace=0.45, wspace=0.0)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(len(linesets))]
    
    xlim = (5.3, 8.5)
    ylim = (-28., -23.)
    
    ylabel = '$\log_{10} \\, \\Lambda \,/\, \\mathrm{n}_{\\mathrm{H}}^{2} \\,/\\, \\mathrm{V} \\; [\\mathrm{erg} \\, \\mathrm{cm}^{3} \\mathrm{s}^{-1}]$'
    xlabel = r'$\log_{10} \, \mathrm{T} \; [\mathrm{K}]$'
    xlabel2 = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}}(\\mathrm{T}_{\\mathrm{200c}}) \\; [\\mathrm{M}_{\\odot}]$'
    
    #labelax = fig.add_subplot(grid[:, :], frameon=False)
    #labelax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #labelax.set_xlabel(xlabel, fontsize=fontsize)
    #labelax.set_ylabel(ylabel, fontsize=fontsize)

    for axi, ax in enumerate(axes):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        
        labely = axi % ncols == 0
        if labely:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        pu.setticks(ax, fontsize=fontsize, top=False, labeltop=False, 
                    labelleft=labely)
        _lines = _linesets[axi]
        ax.grid(b=True)
        
        for line in _lines:
            if line in lines_SB:            
                kwargs = lineargs[line].copy()
                if line == 'fe17':
                    kwargs.update({'linestyle': 'dashed', 'linewidth': 2})
                else:
                    kwargs.update({'linestyle': 'solid', 'linewidth': 2}) 
                pe = getoutline(kwargs['linewidth'])      
                xv = cdata_SB.index
                yv = np.array(cdata_SB[line])
                ax.plot(xv, yv, path_effects=pe, **kwargs)
                
            elif line in lines_PS20:
                kwargs = lineargs[line].copy()
                kwargs.update({'linestyle': 'solid', 'linewidth': 2})
                pe = getoutline(kwargs['linewidth'])
                xv = cdata_PS20.index
                yv = np.array(cdata_PS20[line])
                ax.plot(xv,yv, path_effects=pe, **kwargs)
                
                #kwargs.update({'linestyle': 'dotted', 'linewidth': 2})                
                #pe = getoutline(kwargs['linewidth'])
                #xv = cdata_PS20_2.index
                #yv = np.array(cdata_PS20_2[line])
                #ax.plot(xv, yv, path_effects=pe, **kwargs)
                
            if line != 'fe17':
                Tmax = xv[np.argmax(yv)]
                ax.axvline(Tmax, 0.92, 1., linewidth=3., **lineargs[line])
            if len(linelabels[line]) > 10:
                label = linelabels[line]
                splitpoints = np.where([char == ' ' for char in label])[0]
                split = splitpoints[np.argmin(np.abs(splitpoints - len(label) // 2))]
                label = label[:split] + '\n' + label[split + 1:]
                linelabels[line] = label
                
        #xlim = ax.get_xlim()
        axy2 = ax.twiny()
        axy2.set_xlim(*xlim)
        mhalos = np.arange(11.5, 15.1, 0.5)
        Tvals = np.log10(cu.Tvir_hot(10**mhalos * c.solar_mass,\
                                     cosmopars=cosmopars))
        limsel = Tvals >= xlim[0]
        Tvals = Tvals[limsel]
        mhalos = mhalos[limsel]
        Tlabels = ['%.1f'%mh for mh in mhalos]
        axy2.set_xticks(Tvals)
        axy2.set_xticklabels(Tlabels)
        pu.setticks(axy2, fontsize=fontsize, left=False, right=False,
                    top=True, bottom=False,
                    labelleft=False, labelright=False,
                    labeltop=True, labelbottom=False)
        axy2.minorticks_off()
        axy2.set_xlabel(xlabel2,
                        fontsize=fontsize)
        pe = getoutline(2.)
        if 'fe17' in _lines:
            _lines.remove('fe17')
        handles = [mlines.Line2D([], [], label=linelabels[line],
                                 linewidth=2., path_effects=pe, 
                                 **lineargs[line])\
                   for line in _lines]
        #handles2 = [mlines.Line2D([], [], label='$\\mathrm{{n}}_{{\\mathrm{{H}}}} = {:.0f}$'.format(nH),\
        #                         color='black', **lsargs2[iv])\
        #           for iv, nH in enumerate(indvals)]
        #handles3 = [mlines.Line2D([], [], label='CIE',\
        #                         color='black', **lsargs2[-1])]
        ax.legend(handles=handles, fontsize=fontsize, ncol=1,\
                  bbox_to_anchor=(1.0, 1.0), loc='upper right', frameon=True)
        
        setname = lineset_names[axi]
        ax.text(0.05, 0.95, setname, fontsize=fontsize,\
                horizontalalignment='left', verticalalignment='top',\
                transform=ax.transAxes,\
                bbox={'alpha': 0.3, 'facecolor':'white'})
            
    plt.savefig(outname, format='pdf', bbox_inches='tight')

### total luminosity splits
def plot_barchart_Ls(simple=False):
    '''
    simple: total luminosity fractions for different halo masses (True) or 
            fractions broken down by SF/nSF and subhalo membership category
    '''
    outname = mdir + 'luminosity_total_fractions_z0p1{}.pdf'
    outname = outname.format('_simple' if simple else '')
    _ddir = ddir + 'lumfracs/'
    print('Numbers in annotations: log10 L density [erg/s/cMpc**3] rest-frame')
    
    # change order because the two two-line names overlap
    # 'n6r',  'o7ix',
    if simple:
        lines = plot_lines_SB + plot_lines_PS20
        #lines.sort(key=line_Tmax.get)
        labels = nicenames_lines.copy()
        for line in labels:
            _label = labels[line]
            if '(' in _label and len(_label) > 10:
                splitpoint = np.where([char == '(' for char in _label])
                splitpoint = splitpoint[0][0] - 1
                _label = _label[:splitpoint] + '\n' + _label[splitpoint + 1:]
                labels[line] = _label
    else:
        lines = all_lines_PS20 + all_lines_SB
        lines.sort(key=line_energy_ev)
        labels = nicenames_lines.copy()
        labels.update({line: line.replace(' ' * 6, '\n') for line in lines}) # easy PS20/SB comp
    labels.update({'Mass': 'Mass'})
    # avoid large whitespace just to fit lines names
    #labels['fe17'] = 'Fe XVII\n(17.05 A)'
    #labels['fe17-other1'] = 'Fe XVII\n(15.10 A)'
    keys = ['Mass'] + lines 
    
    # hdf5 group and histogram axis names
    grn_tot = 'StarFormationRate_T4EOS'
    grn_halo = 'M200c_halo_allinR200c_subhalo_category_StarFormationRate_T4EOS'
    axn_sf = 'StarFormationRate_T4EOS'
    axn_hm = 'M200c_halo_allinR200c'
    axn_sh = 'subhalo_category'
    fn_base_SB = 'particlehist_Luminosity_{line}_L0100N1504_27_test3.6' + \
                 '_SmAb_T4EOS.hdf5'
    fn_base_PS20 = 'particlehist_Luminosity_{line}_iontab-PS20-UVB-' + \
                   'dust1-CR1-G1-shield1_depletion-F_L0100N1504_27' + \
                   '_test3.7_SmAb_T4EOS.hdf5'
    filename_M = 'particlehist_Mass_L0100N1504_27_test3.6_T4EOS.hdf5'
    
    sflabels = {0: 'nSF', 1: 'SF'}
    shlabels = {0: 'central', 1: 'subhalo', 2: 'unbound'}
    
    edges_target = np.arange(11., 15.1, 0.5)
    mmax_igm = c.solar_mass
    
    filenames = {line: _ddir + fn_base_SB.format(line=line) \
                 if line in all_lines_SB else \
                 _ddir + fn_base_PS20.format(line=line.replace(' ', '-'))\
                for line in lines}
    filenames.update({'Mass': _ddir + filename_M})
    
    data = {}
    haxes = {}
    hmedges = {}
    for key in keys:
        filen = filenames[key]
        data[key] = {}
        haxes[key] = {}
        with h5py.File(filen, 'r') as fi:
            data[key]['tot'] = fi[grn_tot]['histogram'][:]
            data[key]['halo'] = fi[grn_halo]['histogram'][:]
            haxes[key]['sf'] = fi[grn_halo][axn_sf].attrs['histogram axis']
            haxes[key]['hm'] = fi[grn_halo][axn_hm].attrs['histogram axis']
            haxes[key]['sh'] = fi[grn_halo][axn_sh].attrs['histogram axis']
            hmedges[key] = fi[grn_halo][axn_hm]['bins'][:]
            
            if key == 'Mass':
                cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
            
    if simple:
        figsize = (5.5, 10.)
        height_ratios = [8., 1.]
        ncols = 1
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=2, ncols=1, hspace=0.1, wspace=0.0,\
                            height_ratios=height_ratios)
        ax =  fig.add_subplot(grid[0, 0]) 
        cax = fig.add_subplot(grid[1, 0])
    else:
        figsize = (11., 20.)
        height_ratios = [15., 0.5]
        ncols = 3
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=2, ncols=ncols, hspace=0.2, wspace=0.4,\
                            height_ratios=height_ratios)
        axes =  [fig.add_subplot(grid[0, i]) for i in range(ncols)] 
        cax = fig.add_subplot(grid[1, :])
        
    clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    massedges = np.array(edges_target)
    
    clist = tc.tol_cmap('rainbow_discrete', lut=len(massedges))(np.linspace(0.,  1., len(massedges)))
    c_under = clist[-1]
    clist = clist[:-1]
    c_igm = 'gray'
    _keys = sorted(massedges)
    colors = {_keys[i]: clist[i] for i in range(len(_keys) - 1)}
    #del _masks
    
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist)
    #cmap.set_over(clist[-1])
    cmap.set_under(c_under)
    norm = mpl.colors.BoundaryNorm(massedges, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=np.append(np.array(massedges[0] - 1.), massedges),\
                                ticks=massedges,\
                                spacing='proportional', extend='min',\
                                orientation='horizontal')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    cax.set_aspect(0.1)
    
    colors.update({'lom': c_under, 'igm': c_igm})
    #print(hmedges.keys())
    
    yc = np.arange((len(keys)  - 1) // ncols + 1)  # the label locations
    width = 0.9  # the width of the bars
    incrspace = 0.3
    morder = ['igm', 'lom'] + list(edges_target[:-1]) + ['over']
    for ki, key in enumerate(keys):
        if not simple:
            axi = ki // len(yc)
            ax = axes[axi]
         
        # match halo mass edges:
        edges_in = hmedges[key]
        edges_t = 10**edges_target * c.solar_mass
        
        e_igm = np.where(np.isclose(edges_in, mmax_igm))[0][0]
        s_igm = slice(0, e_igm, None)
        
        e_dct = {edges_target[i]: np.where(np.isclose(edges_in, edges_t[i]))[0][0]\
                 for i in range(len(edges_target))}
        s_dct = {edges_target[i]: slice(e_dct[edges_target[i]],\
                                        e_dct[edges_target[i + 1]], None) \
                 for i in range(len(edges_target) - 1)}
        s_dct['igm'] = s_igm
        s_dct['lom'] = slice(e_igm, e_dct[edges_target[0]], None)
        s_dct['over'] = slice(e_dct[edges_target[-1]], None, None)
        
        # total mass fractions
        if simple: 
            _width = width
            if 'Fe17' in key: # longer labels -> need more vertical space
                yc[ki + 1:] += 2. * incrspace
                yc[ki]  += incrspace
            zeropt = yc[ki]
        else:
            _width = width / 7.
            zeropt = yc[ki % len(yc)] - 3.5 * _width
        
        baseslice = [slice(None, None, None)] * len(data[key]['halo'].shape)
        
        total = np.sum(data[key]['tot'])
        
        # total IGM/ halo mass split
        cumul = 0.
        for mk in morder:
            if mk == 'igm':
                slices = list(np.copy(baseslice))
                slices[haxes[key]['hm']] = slice(e_igm, None, None)
                current = total - np.sum(data[key]['halo'][tuple(slices)])
                current /= total
            else:
                slices = list(np.copy(baseslice))
                slices[haxes[key]['hm']] = s_dct[mk]
                current = np.sum(data[key]['halo'][tuple(slices)])
                current /= total
            if mk == 'over':
                if current == 0.:
                    continue
                else:
                    print('Warning: for {line}, a fraction {} is in masses above max'.format(current, line=key))
            ax.barh(zeropt, current, _width, color=colors[mk], left=cumul)
            cumul += current
        # annotate
        if key != 'Mass':
            dens = total / (cosmopars['boxsize'] / cosmopars['h'])**3 
            text = '{:.1f}'.format(np.log10(dens))
            ax.text(0.99, zeropt, text, fontsize=fontsize,\
                    horizontalalignment='right',\
                    verticalalignment='center')
        if not simple:
            for sfi in range(2):
                for shi in range(3):
                    zeropt += _width
                    cumul = 0.
                    for mk in morder:
                        if mk in ['over', 'igm']:
                            continue
                        else:
                            slices = list(np.copy(baseslice))
                            slices[haxes[key]['hm']] = s_dct[mk]
                            slices[haxes[key]['sh']] = shi
                            slices[haxes[key]['sf']] = sfi
                            current = np.sum(data[key]['halo'][tuple(slices)])
                            current /= total
                        
                        ax.barh(zeropt, current, _width, color=colors[mk],\
                                left=cumul)
                        cumul += current
                        
                    slices = list(np.copy(baseslice))
                    slices[haxes[key]['hm']] = slice(e_igm, None, None)
                    slices[haxes[key]['sh']] = shi
                    slices[haxes[key]['sf']] = sfi
                    subtot = np.sum(data[key]['halo'][tuple(slices)])
                    text = '{:.1f}'.format(np.log10(subtot / 
                                                    (cosmopars['boxsize'] / \
                                                    cosmopars['h'])**3 ))
                    text = ', '.join([shlabels[shi], sflabels[sfi]]) +\
                           ': ' + text
                    ax.text(0.99, zeropt, text, fontsize=fontsize,\
                            horizontalalignment='right',\
                            verticalalignment='center')
    
    if simple:                
        pu.setticks(ax, fontsize)
        #ax.tick_params(axis='y', labelsize=fontsize - 3.)
        ax.set_xlim(0., 1.)
        ax.minorticks_off()
        ax.set_yticks(yc)
        ax.set_yticklabels([labels[key] for key in keys])
        ax.set_xlabel('fraction of total', fontsize=fontsize)
    else:
        for axi, ax in enumerate(axes):
            pu.setticks(ax, fontsize + 1)
            ax.set_xlim(0., 1.)
            ax.minorticks_off()
            nkeys = len(yc)
            while len(yc) * axi + nkeys > len(keys):
                nkeys -= 1 # last column 
            ax.set_yticks(np.arange(nkeys))
            ax.set_yticklabels([labels[keys[axi * len(yc) + i]] \
                                for i in range(nkeys)])
            ax.set_xlabel('fraction of total', fontsize=fontsize)
        
        ylims = [ax.get_ylim() for ax in axes]
        y0 = np.min([ylim[0] for ylim in ylims])
        y1 = np.max([ylim[1] for ylim in ylims])
        [ax.set_ylim(y0, y1) for ax in axes]
        
    plt.savefig(outname, format='pdf', bbox_inches='tight')





### instruments and minimum SB
def plot_Aeff_galabs():
    datname = ddir + 'Aeff_cm2_{ins}.dat'
    datname_wabs = ddir + 'wabs_absfrac.dat'
    Ekey = 'energy [keV]' 
    Akey = 'effective area [cm**2]'
    fkey = 'transmitted fraction'
    
    names = ['athena-xifu', 'lynx-lxm-main', 'xrism-resolve'] # , 'lynx-lxm-uhr'
    labels = {'athena-xifu': 'X-IFU',
              'lynx-lxm-main': 'LXM',
              'lynx-lxm-uhr': 'LXM-UHR',
              'xrism-resolve': 'XRISM-R',
              }
    cset = tc.tol_cset('vibrant')
    colors = {'athena-xifu': cset.blue,
              'lynx-lxm-main': cset.orange,
              'lynx-lxm-uhr': cset.red,
              'xrism-resolve': cset.teal}
    
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.gca()
    fontsize = 12
    
    for isn in names:
        _fn = datname.format(ins=isn)
        data = pd.read_csv(_fn, header=1, sep='\t')
        
        label = labels[isn] 
        ax.plot(data[Ekey], data[Akey], label=label, color=colors[isn], 
                linewidth=2)
        
    data = pd.read_csv(datname_wabs, header=3, sep='\t')
    ax.plot(data[Ekey], data[fkey] * 1e4, color='black',
            label='wabs * 1e4 $\\mathrm{cm}^{2}$')
                
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E [keV]', fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{A}_{\\mathrm{eff}} \\; [\\mathrm{cm}^{2}]$',\
                  fontsize=fontsize)
    ax.tick_params(labelsize=fontsize-1, direction='in', which='both',\
                   top=True, right=True)
    ax.grid(b=True)
    ax.grid(b=True, which='minor', axis='x')
    #xlim = ax.get_xlim()
    ax.set_xlim(0.1, 3.)
    ax.set_ylim(0.15, 3e4)
    ax.legend(fontsize=fontsize, loc='lower right', framealpha=1.)
    plt.savefig(mdir + 'Aeff_galabs_instruments.pdf', bbox_inches='tight')

def plot_minSB():
    names = ['athena-xifu', 'lynx-lxm-main', 'lynx-lxm-uhr', 'xrism-resolve']
    labels = {'athena-xifu': 'X-IFU',
              'lynx-lxm-main': 'LXM-main',
              'lynx-lxm-uhr': 'LXM-UHR',
              'xrism-resolve': 'XRISM-R',
              }
    cset = tc.tol_cset('vibrant')
    colors = {'athena-xifu': cset.blue,
              'lynx-lxm-main': cset.orange,
              'lynx-lxm-uhr': cset.red,
              'xrism-resolve': cset.teal
              }
    
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.gca()
    fontsize = 12
    
    exptimes = [1e6, 1e7]
    linestyles = ['solid', 'dotted']
    
    alphas_g = [1.0, 0.3]
    galabs = [True, False]
    
    addl = ' {omegat:.0e} am2 s, S/N over {deltae:.1f} eV'
    Ekey = 'input line energy [keV]'
    mkey = 'minimum detectable SB [photons * cm**-2 * s**-1 * sr**-1]'
    gkey = 'including effect of galactic absorption'
    tkey = 'stacked area * exposure time [arcmin**2 * s]'
    wkey = 'input line width [km * s**-1]'
    skey = 'required detection significance [sigma]'
    rkey = 'signal/noise extraction region in the spectrum [full width, eV]'
    
    erngs = {}
    for isn in names:
        fn = ddir + 'minSB_curves_{ins}.dat'.format(ins=isn)
        data = pd.read_csv(fn, sep='\t', header=4)
        
        for et, ls in zip(exptimes, linestyles):
            for ga, ag in zip(galabs, alphas_g):
                sel = np.isclose(data[tkey], et)
                sel &= data[gkey] == ga
                _data = data[sel]
                
                xv = np.array(_data[Ekey])
                yv = np.array(_data[mkey])
                erng = _data[rkey]
                if not np.allclose(erng, erng.at[erng.keys()[0]]):
                    raise RuntimeError('Multiple extraction ranges for one plot line')
                erng = erng[erng.keys()[0]]
                if isn in erngs:
                    if not np.isclose(erngs[isn], erng):
                        raise RuntimeError('Multiple extraction ranges for one instrument')
                else:
                    erngs[isn] = erng
                lw_in = _data[wkey]
                if not np.allclose(lw_in, lw_in.at[lw_in.keys()[0]]):
                    raise RuntimeError('Multiple line widths for one plot line')
                nsigma = _data[skey]
                if not np.allclose(nsigma, nsigma.at[nsigma.keys()[0]]):
                    raise RuntimeError('Multiple line widths for one plot line')
                
                label = labels[isn] + addl.format(omegat=et, deltae=erng)
                ax.plot(xv, yv, label=label, color=colors[isn],\
                        linestyle=ls, alpha=ag)
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E [keV]', fontsize=fontsize)
    ax.set_ylabel('$\\min \\mathrm{SB} \\; [\\mathrm{ph} \\; \\mathrm{s}^{-1} \\mathrm{cm}^{-2} \\mathrm{sr}^{-1}]$',\
                  fontsize=fontsize)
    ax.set_xticks([0.2, 0.3, 0.5, 1., 2., 3.])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.grid(b=True)
    ax.tick_params(direction='in', labelsize=fontsize - 1., which='both',
                   top=True, right=True)
    ax.set_xlim(0.18, 3.15)
    ax.set_ylim(2e-3, 8e3)
    
    handles1 = [mlines.Line2D([], [],
                         label=labels[isn] + \
                         ', S/N over {deltae:.1f} eV'.format(deltae=erngs[isn]),
                         color=colors[isn])\
                for isn in names]
    handles2 = [mlines.Line2D([], [],
                         label='{:.0e} $\\mathrm{{arcmin}}^{{2}}$ s'.format(omt),
                         linestyle=ls, color='gray')\
                for omt, ls in zip(exptimes, linestyles)]
    handles3 = [mlines.Line2D([], [],
                         label='with MW abs.' if ga else 'without MW abs.',
                         linestyle='solid', color='black', alpha=ag)\
                for ga, ag in zip(galabs, alphas_g)]
        
    ax.legend(handles=handles1 + handles2 + handles3, fontsize=fontsize-2,
              loc='upper right', bbox_to_anchor=(1., 1.), framealpha=1.)
    plt.savefig(mdir + 'minSB_instruments_varying_omegatexp.pdf', 
                bbox_inches='tight')

def printabundancetable(elts):
    '''
    print the solar abundances of elts in LaTeX table form
    '''
    numfmt = 'r@{$\\times$}l'
    tstart = '\\begin{{tabular}}{{l {numfmt} {numfmt} l }}'.format(numfmt=numfmt)
    head1 = 'element & \\multicolumn{4}{c}{metallicity} ' + \
            '& source \\\\'
    head2 = ' & \\multicolumn{2}{c}{$n_{\\mathrm{elt}} \\,/\\, n_{\\mathrm{H}}$}' +\
            ' & \\multicolumn{2}{c}{$\\rho_{\\mathrm{elt}} \\,/\\, \\rho_{\\mathrm{tot}}$} \\\\'
    hline = '\\hline'
    fillstr = '{elt} & ${num_sb:.2f}$&$10^{{{exp_sb}}}$ & ' + \
              '${num_ea:.2f}$&$10^{{{exp_ea}}}$ & {src_sb} \\\\'
    tend = '\\end{tabular}'
    
    print(tstart)
    print(hline)
    print(head1)
    print(head2)
    print(hline)
    for elt in elts:
        en = elt_to_abbr[elt]
        
        val_sb = ol.solar_abunds_sb[elt]
        src_sb = ol.sources_abunds_sb[elt]
        exp_sb = int(np.floor(np.log10(val_sb)))
        num_sb = val_sb / (10** exp_sb)
        
        val_ea = ol.solar_abunds_ea[elt]
        exp_ea = int(np.floor(np.log10(val_ea)))
        num_ea = val_ea / (10** exp_ea)
        
        line = fillstr.format(elt=en, num_sb=num_sb, exp_sb=exp_sb,\
                              num_ea=num_ea, exp_ea=exp_ea,\
                              src_sb=src_sb)
        print(line)
    print(hline)
    print(tend)


def printlatex_minsb(lineset='SB'):
    '''
    print a LaTeX table with the minimum SB for the different lines in a set.
    Combine different sets into one table by hand,

    Parameters
    ----------
    lineset : string, optional
        which lines to print. Options are 'SB' and 'PS20'. The default is 'SB'.

    Returns
    -------
    None.

    '''
    if lineset == 'SB':
        lines = plot_lines_SB
        filen='minSBtable.dat'
    elif lineset == 'PS20':
        lines = plot_lines_PS20
        filen = 'minSBtable_PS20_Fe-L-shell.dat'
        
    df = pd.read_csv(ddir + filen, sep='\t')     
    groupby = ['line name', 'linewidth [km/s]',
               'sky area * exposure time [arcmin**2 s]', 
               'full measured spectral range [eV]',
               'detection significance [sigma]', 
               'galaxy absorption included in limit',
               'instrument']
    df2 = df.groupby(groupby)['minimum detectable SB [phot/s/cm**2/sr]'].mean().reset_index()
    zopts = df['redshift'].unique()
    print('Using redshifts: {}'.format(zopts))
    print('\n\n')
    
    # get difference between absorbed/unabsorbed minimum (only depends on line energy)
    groupby = ['galaxy absorption included in limit',
               'line name', 'linewidth [km/s]',
               'sky area * exposure time [arcmin**2 s]', 
               'full measured spectral range [eV]',
               'detection significance [sigma]', 
               'instrument']
    df3 = df.set_index(groupby) 
    df3 = df3.loc[True].divide(df3.loc[False])
    df3.reset_index()
    df_diff = df3.groupby('line name')['minimum detectable SB [phot/s/cm**2/sr]'].mean().reset_index()
    df_diff = df_diff.set_index('line name')
    df_diff = np.log10(df_diff)
    
    instruments = df2['instrument'].unique()
    #_lines =  df2['line name'].unique()
    omegat = df2['sky area * exposure time [arcmin**2 s]'].unique()
    #galabs = df2['galaxy absorption included in limit'].unique()
    
    omegat_galabs = [(1e7, True), (1e6, True), (1e5, True)]
    omegat_coln = ['1e7', '1e6', '1e5']
    nsc = len(omegat_galabs)
    #subfmt = ' & '.join(['{}'] * nsc)
    instruments = ['xrism-resolve', 'athena-xifu', 'lynx-lxm-uhr', 'lynx-lxm-main']
    insnames = ['XRISM Resolve', 'Athena X-IFU', 'LXM UHR', 'LXM main']
    
    insfmt = '\\multicolumn{{{nsc}}}{{c}}{{{insn}}}'
    head1 = 'instrument & ' + ' & '.join([insfmt.format(nsc=nsc, insn=insn)\
                                                  for insn in insnames]) +\
            ' & \\multicolumn{1}{c}{$\\Delta_{\\mathrm{wabs}}$} \\\\'
            
    head2 = '$\\Delta \\Omega \\, \\Delta \\mathrm{t} \\; [\\mathrm{arcmin}^2 \\mathrm{s}]$'
    head2 = head2 + ' & ' +  ' & '.join([' & '.join(omegat_coln)] *\
                                         len(instruments)) +\
            ' &  \\multicolumn{1}{c}{$[\\log_{10} \\mathrm{SB}]$} \\\\'
    start = '\\begin{{tabular}}{{{cols}}}'.format(\
                    cols='l' + 'r' * (nsc * len(instruments) + 1))
    end = '\\end{tabular}'
    fmtl = '{line} & ' + ' & '.join(['{}'] * nsc * len(instruments)) + \
           ' & {delta_wabs:.2f} \\\\'
    hline = '\\hline'
    
    print(start)
    print(hline)
    print(head1)
    print(head2)
    print(hline)
    for line in lines:
        # order of loops is important for consistent results with column names
        vals = []
        for ins in instruments:
            for omegat_target, galabs_target in omegat_galabs:
                
                otk = omegat[np.where(np.isclose(omegat, omegat_target))[0][0]]
                sel = np.logical_and(df2['instrument'] == ins,\
                                     df2['galaxy absorption included in limit'] == galabs_target)
                sel = np.logical_and(sel, df2['line name'] == line)
                sel = np.logical_and(sel, df2['sky area * exposure time [arcmin**2 s]'] == otk)
                
                if np.sum(sel) != 1:
                    print('for line {}, galabs {}, omegat {}, instrument {}'.format(\
                          line, galabs_target, otk, ins))
                    print(df2[sel])
                ind = df2.index[sel][0]
                val = df2.at[ind, 'minimum detectable SB [phot/s/cm**2/sr]']
                
                pval = '-' if val == np.inf else '{:.1f}'.format(np.log10(val))
                if pval == '-0.0':
                    pval = '0.0'
                vals.append(pval)
        pl = fmtl.format(*tuple(vals), line=nicenames_lines[line], 
                         delta_wabs=df_diff.at[line, 'minimum detectable SB [phot/s/cm**2/sr]'])
        print(pl)
    print(hline)
    print(end)    
    # columns: 'line name', 'E rest [keV]', 'linewidth [km/s]', 'redshift',\
    # 'sky area * exposure time [arcmin**2 s]', 
    # 'full measured spectral range [eV]', 'detection significance [sigma]',\
    # 'galaxy absorption included in limit',\
    # 'minimum detectable SB [phot/s/cm**2/sr]', 'instrument'

def sigdig_fmt(val, numdig):
    '''
    format for significant digits (no x 10^n notation)
    '''
    minus = '-' if val < 0 else ''
    val = np.abs(val)
    lead = int(np.floor(np.log10(val)))
    numtrail = numdig - lead
    out = str(np.round(val, numtrail - 1))
    if '.' in out:
        if out[-2:] == '.0' and len(out) > numdig + 1: 
            # trailing fp '.0' adds a sig. digit
            out = out[:-2]
        elif '.' in out and len(out) < numdig + 1:
            # trailing zeros removed 
            out = out + '0' * (numdig + 1 - len(out))
        if out[:2] == '0.' and len(out) < numtrail + 1:
            # add more zeros (leading zeros are counted above)
            out = out + '0' * (numtrail + 1 - len(out))
    if float(out) == 0.:
        minus = ''
    return minus + out

def printlatex_linedata(emcurve_file):
    '''
    print a latex table with the line emission curve data. Transitions need to
    be added afterwards by hand.

    Parameters
    ----------
    emcurve_file : string
        file containing the CIE emission curves.

    Returns
    -------
    None.

    '''
    hline = '\\hline'
    
    if 'PS20' in emcurve_file:
        cloudyversion = '17.01'
        ps20 = True
    else:
        cloudyversion = '7.02'
        ps20 = False
    emfrac = 0.1
    columns = ['ion', 'wl', 'E', 'Lmax', 'Tmax', 'Trng', 'ul', 'll', 'name']
    fillstr = ' \t& '.join(['{{{}}}'.format(col) for col in columns]) \
              + ' \\\\'
    hed1_dct = {'ion':  'ion',
                'wl':   '$\\lambda$',
                'E':    'E',
                'Lmax': '$\\max \, \\Lambda \\,\\mathrm{n}_' + \
                        '\\mathrm{H}^{-2} \\mathrm{V}^{-1}$',
                'Tmax': '$\\mathrm{T}_{\\mathrm{peak}}$',
                'Trng': '$\\mathrm{{T}}_{{{f} '.format(f=emfrac) + \
                        '\\times \\mathrm{peak}}$',
                'ul':   'upper level',
                'll':   'lower level',
                'name': 'name',
                }
    hed2_dct = {'ion':  '',
                'wl':   '$\\textnormal{\\AA}$',
                'E':    'keV',
                'Lmax': '$\\log_{10} \\, \\mathrm{erg} \\, ' + \
                        '\\mathrm{cm}^{3} \\mathrm{s}^{-1}$',
                'Tmax': ' $\\log_{10}$~K',
                'Trng': ' $\\log_{10}$~K',
                'ul':   '',
                'll':   '',
                'name': '{\\textsc CLOUDY}~v' + cloudyversion,
                }
    print(hline)
    print(fillstr.format(**hed1_dct))
    print(fillstr.format(**hed2_dct))
    print(hline)
    
    fformat = {'sep':'\t', 'index_col':'T', 'comment':'#'}
    cdata = pd.read_csv(ddir + emcurve_file, **fformat)
    lines = list(cdata.columns)
    lines.sort(key=line_energy_ev)
    
    for line in lines:
        filldct = {}
        if ps20:
            _line = line
        else:
            ind = ol.line_nos_ion[line]
            parentelt = ol.elements_ion[line]
            tablefilename = ol.dir_emtab%(ol.zopts[0]) + parentelt + '.hdf5'
            tablefile = h5py.File(tablefilename, 'r')
            _line = tablefile['lambda'][ind].decode()
            
        ion = _line[:4]
        wl = _line[4:-1]
        if _line[-1] != 'A':
            raise NotImplementedError('Only A wavelength interpretation implemented')
        elt = ion[:-2].strip()
        stage = ild.arabic_to_roman[int(ion[2:])]
        ion = elt + ' ' + stage
        filldct['wl'] = wl
        filldct['ion'] = ion
        E = c.planck * c.c / (float(wl) * 1e-8) / c.ev_to_erg * 1e-3
        filldct['E'] = sigdig_fmt(E, 4)
        filldct['name'] = _line.replace(' ', '\\_')
        filldct['ul'] = ''
        filldct['ll'] = ''
        
        T = cdata.index
        L = np.array(cdata[line])
        mi = np.argmax(L)
        filldct['Lmax'] = sigdig_fmt(L[mi], 3)
        filldct['Tmax'] = '{:.2f}'.format(T[mi])
        Trng = pu.find_intercepts(L, T, np.log10(emfrac) + L[mi])
        if len(Trng) != 2:
            raise RuntimeError('Found T range {} for line {}'.format(Trng, 
                                                                     line))
        filldct['Trng'] = '{:.1f}--{:.1f}'.format(*tuple(Trng))
        print(fillstr.format(**filldct))
        
    print(hline)
    
    