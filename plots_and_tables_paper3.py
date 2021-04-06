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


mdir = '/net/luttero/data2/imgs/paper3/img_paper/' 
tmdir = '/net/luttero/data2/imgs/paper3/img_talks/'
ddir = '/net/luttero/data2/imgs/paper3/datasets/'

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

parentelts = {line: abbr_to_elt[line[:2].strip()] for line in all_lines_PS20}
parentelts.update({line: ol.elements_ion[line] for line in all_lines_SB})

# for setting plotting defaults
plot_lines_SB = ['c5r', 'c6', 'n6-actualr', 'n7', 'o7r', 'o7iy', 'o7f', 'o8',
                 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r']
plot_lines_PS20 = ['Fe17      17.0510A',
                   'Fe17      15.2620A', 'Fe17      16.7760A',
                   'Fe17      17.0960A', 'Fe18      16.0720A']
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
         'n6r':  {'linestyle': 'dotted',  'color': _c1.cyan},
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




