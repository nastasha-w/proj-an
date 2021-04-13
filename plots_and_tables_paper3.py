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
import os
import sys

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

if sys.version.split('.')[0] == '3':
    def isstr(x):
        return isinstance(x, str)
elif sys.version.split('.')[0] == '2':
    def isstr(x):
        return isinstance(x, basestring)
    
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

def readin_radprof(filename, seltags, ys, runit='pkpc', separate=False,\
                   binset='binset_0', retlog=True, ofmean=False):
    '''
    from the coldens_rdist radial profile hdf5 file format: 
    extract the profiles matching the input criteria
    
    input:
    ------
    filename: name of the hdf5 file containing the profiles
    seltag:   list (iterable) of seltag attributes for the galaxy/halo set used
    ys:       list (iterable) of tuples (iterable) containing ytype 
              ('mean', 'mean_log', 'perc', or 'fcov') and 
              yval (threashold for 'fcov', percentile (0-100) for 'perc',\
              ignored for 'mean' and 'mean_log')
    runits:   units for the bins: 'pkpc' or 'R200c'
    separate: get individual profiles for the galaxies in this group (subset
              contained in this file) if True, otherwise, use the combined 
              profile 
    binset:   the name of the binset group ('binset_<number>')
    retlog:   return log unit values (ignored if log/non-log isn't recorded)
    ofmean:   return statistics of the individual mean profiles instead of the
              whole sample. Only works if separate is False. (assumed stored 
              as mean_log)
    
    output:
    -------
    two dictionaries:
    y dictionary, bin edges dictionary
    
    nested dictionaries containing the retrieved values
    keys are nested, using the input values as keys:
    seltag: ys: [galaxyid: (if separate)] array of y values or bin edges
    
    '''
    
    if '/' not in filename:
        filename = ol.pdir + 'radprof/' + filename

    spath = '{{gal}}/{runit}_bins/{bin}/{{{{ds}}}}'.format(runit=runit, 
                                                           bin=binset)
        
    with h5py.File(filename, 'r') as fi:
        # match groups to seltags
        gkeys = list(fi.keys())
        gkeys.remove('Header')
        setkeys = [key if 'galset' in key else None for key in gkeys]
        setkeys = list(set(setkeys) - {None})
        
        seltags_f = {key: fi[key].attrs['seltag'].decode()\
                     if 'seltag' in fi[key].attrs.keys() else None \
                     for key in setkeys}
        galsets_seltag = {tag: list(set([key if seltags_f[key] == tag else\
                                         None for key in seltags_f])\
                                    - {None})\
                          for tag in seltags}
        
        indkeys = list(set(gkeys) - set(setkeys))
        indgals = [int(key.split('_')[-1]) for key in indkeys]
        
        spaths = {}
        galid_smatch = {}
        ys_out = {}
        bins_out = {}
        for seltag in seltags:
            keys_tocheck = galsets_seltag[seltag]
            if len(keys_tocheck) == 0:
                msg = 'No matches found for seltag {}'
                raise RuntimeError(msg.format(seltag))
            
            if separate: # just pick one; should have the same galaxyids
                extag = keys_tocheck[0]
                galids = fi['{gs}/galaxyid'.format(gs=extag)][:]
                # cross-reference stored galids (should be a smallish number)
                # and listed set galids
                galids_use = [galid if galid in indgals else None \
                              for galid in galids]
                galids_use = list(set(galids_use) - {None})
                gb = 'galaxy_{}'
                spaths.update({galid: spath.format(gal=gb.format(galid))\
                               for galid in galids_use})
                galid_smatch.update({galid: seltag for galid in galids_use})
            else:
                spaths.update({key: spath.format(gal=key) \
                               for key in keys_tocheck})
                galid_smatch.update({key: seltag for key in keys_tocheck})
        for ytv in ys:
            ykey = ytv
            temppaths = spaths.copy()
            if ytv[0] in ['fcov', 'perc']:
                ypart = '{}_{:.1f}'.format(*tuple(ytv))
            else:
                if ytv[0] == 'm':
                    ypart = ytv
                else:
                    ypart = ytv[0]
            if ofmean:
                ypart = ypart + '_of_mean_log'
            ypaths = {key: temppaths[key].format(ds=ypart) for key in temppaths}
            bpaths = {key: temppaths[key].format(ds='bin_edges') \
                      for key in temppaths}

            for key in temppaths:
                ypath = ypaths[key]
                bpath = bpaths[key]
                if ypath not in fi:
                    if '_of_mean_log' in ypath:
                        ypath = ypath.replace('of_mean_log', 'of_mean')
                    if ypath not in fi:
                        print('For {seltag}, {ykey}, {key}'.format(
                            seltag=seltag, ykey=ykey, key=key))
                        print('did not find dataset: {}'.format(ypath))
                        
                if ypath in fi:
                    vals = fi[ypath][:]
                    if 'logvalues' in fi[ypath].attrs.keys():
                        logv_s = bool(fi[ypath].attrs['logvalues'])
                        if 'fcov' in ypath: # logv_s is for 
                            if retlog:
                                vals = np.log10(vals)
                        else:
                            if retlog and (not logv_s):
                                vals = np.log10(vals)
                            elif (not retlog) and logv_s:
                                vals = 10**vals
                    bins = fi[bpath][:]
                
                    seltag = galid_smatch[key]
                    if seltag not in ys_out:
                        ys_out[seltag] = {}
                        bins_out[seltag] = {}
                    if isstr(key): # string instance? (hard to test between pytohn 2 and 3 with isinstance)
                        if ykey not in ys_out[seltag]:
                            ys_out[seltag][ykey] = {}
                            bins_out[seltag][ykey] = {}
                        ys_out[seltag][ykey][key] = vals
                        bins_out[seltag][ykey][key] = bins
                    else:
                        ys_out[seltag][ykey] = vals
                        bins_out[seltag][ykey] = bins
    return ys_out, bins_out

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
    
    yc = np.arange((len(keys)  - 1) // ncols + 1, dtype=np.float)  # the label locations
    width = 0.9  # the width of the bars
    incrspace = 0.09
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
                #_width += incrspace
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
        ax.tick_params(axis='y', labelsize=fontsize - 1.)
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

# halo luminosities L200c with scatter 
def plot_luminosities_nice(addedges=(0., 1.), talkversion=False, slidenum=0):
    '''
    '''
    
    outname = 'luminosities_nice_{mi}-{ma}-R200c'.format(\
                            mi=addedges[0], ma=addedges[1])
    if talkversion:
        outdir = tmdir
        outname = outname + 'talkversion_{}'.format(slidenum)
        fontsize = 14
    else:
        outdir = mdir
        fontsize = 12
    outname = outname.replace('.', 'p')
    outname = outdir + outname + '.pdf'
    
    lines = plot_lines_SB + plot_lines_PS20
    cosmopars = cosmopars_27
    lsargs = lineargs_sets.copy()
    linewidth = 2.
    patheff = getoutline(linewidth)

    linelabels = nicenames_lines.copy()
    _linesets = linesets.copy()
    #_linesets[3] = [_linesets[3][2], _linesets[3][3], linesets[3][0], linesets[3][1]]
    
    _lines = lines.copy() # new lines added in loop
    for line in _lines:
        _label = linelabels[line]
        if len(_label) > 10 and '(' in _label:
            split = np.where([char == '(' for char in _label])
            split = split[0][0] - 1
            _label = _label[:split] + '\n' + _label[split + 1:]
            linelabels[line] = _label
        # get an initial comparison of the SB and PS20 tables
        lkw = lsargs[line]
        if 'linestyle' in lkw:
            del lkw['linestyle']
        if 'dashes' in lkw:
            del lkw['dashes']
        lkw['linestyle'] = 'solid'
        lsargs[line] = lkw
           # if line in linematch_PS20:
           #     lkw['linestyle'] = 'dashed'
           #     sbline = linematch_PS20[line]
           #     if sbline is None:
           #         continue
           #     lines.append(sbline)
           #     linelabels[sbline] = None
           #     lsargs[sbline] = lkw.copy()
           #     lsargs[sbline]['linestyle'] = 'solid'
           #     
           #     _lsi = np.where([line in _ls for _ls in _linesets])[0][0]
           #     _linesets[_lsi].append(sbline)
           # elif line in linematch_SB:
           #     lkw['linestyle'] = 'solid'
           #     psline = linematch_SB[line]
           #     if psline is None:
           #         continue
           #     lines.append(psline)
           #     linelabels[psline] = None
           #     lsargs[psline] = lkw.copy()
           #     lsargs[psline]['linestyle'] = 'dashed'
           #     
           #     _lsi = np.where([line in _ls for _ls in _linesets])[0][0]
           #     _linesets[_lsi].append(psline)

            
            
    ylabel = '$\\log_{10} \\, \\mathrm{L} \\; [\\mathrm{photons} \\,/\\, 100\\,\\mathrm{ks} \\,/\\, \\mathrm{m}^{2}]$'
    time = 1e5 #s
    Aeff = 1e4 # cm^2 
    ylim = (-3., 4.5)
             
    xlabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\odot}]$' 
    
    base_SB = 'luminosities_halos_L0100N1504_27_Mh0p5dex_1000_{}-{}-R200c' + \
              '_SmAb.hdf5'
    base_PS20 = 'luminosities_PS20_depletion-F_halos_L0100N1504_27_Mh0p5dex' +\
                '_1000_{}-{}-R200c_SmAb.hdf5'
    filename_SB = ddir + base_SB.format(str(addedges[0]), str(addedges[1]))
    filename_PS20 = ddir + base_PS20.format(str(addedges[0]), str(addedges[1]))
    with h5py.File(filename_SB, 'r') as fi:
        galids_SB_l = fi['galaxyids'][:]
        read_lines_SB = [line.decode() for line in fi.attrs['lines']]
        lums_SB = fi['luminosities'][:]
        cosmopars = {key: val for key, val in \
                     fi['Header/cosmopars'].attrs.items()}
        
        ldist = cu.lum_distance_cm(cosmopars['z'], cosmopars=cosmopars)
        print(ldist)
        l_to_flux = 1. / (4 * np.pi * ldist**2) * (1. + cosmopars['z']) # photon flux -> compensate for flux decrease due to redshifting in ldist
        Erest_SB = np.array([line_energy_ev(line) * c.ev_to_erg \
                             for line in read_lines_SB])
        lums_SB *= 1./ (Erest_SB[np.newaxis, :, np.newaxis]) \
                   * l_to_flux * Aeff * time
    with h5py.File(filename_PS20, 'r') as fi:
        galids_PS20_l = fi['galaxyids'][:]
        read_lines_PS20 = [line.decode().replace('-', ' ')\
                           for line in fi.attrs['lines']]
        lums_PS20 = fi['luminosities'][:]
        cosmopars = {key: val for key, val in fi['Header/cosmopars'].attrs.items()}
        
        ldist = cu.lum_distance_cm(cosmopars['z'], cosmopars=cosmopars)
        print(ldist)
        l_to_flux = 1. / (4 * np.pi * ldist**2) * (1. + cosmopars['z']) # photon flux -> compensate for flux decrease due to redshifting in ldist
        Erest_PS20 = np.array([line_energy_ev(line) * c.ev_to_erg \
                               for line in read_lines_PS20])
        lums_PS20 *= 1./ (Erest_PS20[np.newaxis, :, np.newaxis]) \
                     * l_to_flux * Aeff * time
        
    file_galdata = ddir + 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    galdata_all = pd.read_csv(file_galdata, header=2, sep='\t', 
                              index_col='galaxyid')
    masses_SB = np.array(galdata_all['M200c_Msun'][galids_SB_l])
    masses_PS20 = np.array(galdata_all['M200c_Msun'][galids_PS20_l])
    
    mbins = np.array(list(np.arange(11., 13.05, 0.1)) +\
                     [13.25, 13.5, 13.75, 14.0, 14.6])
    bincen = mbins[:-1] + 0.5 * np.diff(mbins)
    
    lums_SB = np.log10(np.sum(lums_SB, axis=2))
    lums_PS20 = np.log10(np.sum(lums_PS20, axis=2))
    
    bininds_SB = np.digitize(np.log10(masses_SB), mbins)
    bininds_PS20 = np.digitize(np.log10(masses_PS20), mbins) 
    
    if talkversion:
        ncols = 2
        figsize = (11., 7.)
        nrows = (len(linesets) - 1) // ncols + 1
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0.0)
        axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(len(linesets))]
        
    else:
        ncols = 1
        figsize = (5.5, 12.)
        nrows = len(linesets)
        fig = plt.figure(figsize=figsize)
        grid = gsp.GridSpec(nrows=nrows, ncols=ncols, hspace=0.0, wspace=0.0)
        axes = [fig.add_subplot(grid[i, 0]) for i in range(nrows)]
    
    labelax = fig.add_subplot(grid[:nrows, :ncols], frameon=False)
    labelax.tick_params(labelcolor='none', top=False, bottom=False,
                        left=False, right=False)
    labelax.set_xlabel(xlabel, fontsize=fontsize)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    
    for axi, ax in enumerate(axes):
        lineset = _linesets[axi]
        linesetlabel = lineset_names[axi]
        
        labelx = axi >= len(linesets) - ncols
        labely = axi % ncols == 0
        #if labelx:
        #    ax.set_xlabel(xlabel, fontsize=fontsize)
        #if labely:
        #    ax.set_ylabel(ylabel, fontsize=fontsize)
        pu.setticks(ax, fontsize, labelleft=labely, labelbottom=labelx)
        ax.grid(b=True
                )
        _labels = {}
        for _li, line in enumerate(lineset):
            label = linelabels[line]
            if axi == 0 and label == 'O VII (r)':
                label = 'O VII'
            _labels[line] = label
            if talkversion:
                ncomp = 1 + _li + np.sum([len(_linesets[_axi]) for _axi in range(axi)])
                if ncomp > slidenum:
                    continue
            
            if line in all_lines_PS20:
                li = np.where([line == _l for _l in read_lines_PS20])[0][0]
                lums = lums_PS20[:, li]
                bininds = bininds_PS20
            elif line in all_lines_SB:
                li = np.where([line == _l for _l in read_lines_SB])[0][0]
                lums = lums_SB[:, li]
                bininds = bininds_SB
                
            med = [np.median(lums[bininds == i]) \
                   for i in range(1, len(mbins))]
            ax.plot(bincen, med, label=label, linewidth=linewidth,\
                    path_effects=patheff, **lsargs[line])
            
            ud = [np.percentile(lums[bininds == i], [10., 90.]) \
                  for i in range(1, len(mbins))]
            ud = np.array(ud).T
            ud[0, :] = med - ud[0, :]
            ud[1, :] = ud[1, :] - med
            lsi = np.where([l == line for l in lineset])[0][0]
            cycle = len(lineset)
            #print(lsi)
            #print(cycle)
            sl = slice(lsi, None, cycle) # avoid overlapping error bars
            _lsargs = lsargs[line].copy()
            _lsargs['linestyle'] = 'none'
            ax.errorbar(bincen[sl], med[sl], yerr=ud[:, sl],\
                        linewidth=linewidth,\
                        path_effects=patheff,\
                        **_lsargs)
            
        #handles, labels = ax.get_legend_handles_labels()
        handles = [mlines.Line2D([], [], linewidth=linewidth,\
                    path_effects=patheff, label=_labels[line],\
                    **lsargs[line])\
                   for line in lineset]
        isplit = len(handles) // 2
        h1 = handles[:isplit]
        h2 = handles[isplit:]
        fc = (1., 1., 1., 0.)
           
        if axi == 3:
            l2 = ax.legend(handles=h2, fontsize=fontsize, 
                           bbox_to_anchor=(1.0, 0.), loc='lower right', 
                           ncol=1, title=linesetlabel, facecolor=fc)
            l1 = ax.legend(handles=h1, fontsize=fontsize, 
                           bbox_to_anchor=(0.0, 1.0), loc='upper left',
                           ncol=1, facecolor=fc)
            l2.get_title().set_fontsize(fontsize)
            ax.add_artist(l2)
        else:
            l1 = ax.legend(handles=h2, fontsize=fontsize, 
                           bbox_to_anchor=(1.0, 0.), loc='lower right', 
                           ncol=1, facecolor=fc)
            l2 = ax.legend(handles=h1, fontsize=fontsize, 
                           bbox_to_anchor=(0.0, 1.0), loc='upper left', 
                           ncol=1, title=linesetlabel, facecolor=fc)
            l2.get_title().set_fontsize(fontsize)
            ax.add_artist(l1)
        
        #ax.add_artist(l2) # otherwise, it's plotted twice -> less transparent
        #ax.text(0.02, 0.98, linesetlabel, fontsize=fontsize,\
        #        verticalalignment='top', horizontalalignment='left',\
        #        transform=ax.transAxes)
    # sync lims
    xlims = [ax.get_xlim() for ax in axes]
    x0 = np.min([xl[0] for xl in xlims])
    x1 = np.max([xl[1] for xl in xlims])
    if talkversion:
        x0 = 10.8
        x1 = 14.45
    [ax.set_xlim(x0, x1) for ax in axes]
    #handles = [mlines.Line2D([], [], linestyle='solid', color='black',
    #                         label='SB tables'),
    #           mlines.Line2D([], [], linestyle='dashed', color='black',
    #                         label='PS20 tables')]
    #l3 = axes[0].legend(handles=handles, fontsize=fontsize, 
    #                    bbox_to_anchor=(0.5, 1.),
    #                    loc='lower center', ncol=2, facecolor=fc) 
    #axes[0].add_artist(l3)
    #ylims = [ax.get_ylim() for ax in axes]
    #y0 = np.min([yl[0] for yl in ylims])
    #y1 = np.max([yl[1] for yl in ylims])
    [ax.set_ylim(*ylim) for ax in axes]
        
    plt.savefig(outname, format='pdf', bbox_inches='tight')

### main plot, radial profiles (mean and median of annular means), 
# observablility, detectability
def plot_radprof_main(talkversion=False, slidenum=0, talkvnum=0):
    '''
    plot mean and median profiles for the different lines in different halo 
    mass bins
    
    talkversion: fewer emission lines, add curves one at a time
    '''
    
    print('Values are calculated from 3.125^2 ckpc^2 pixels')
    print('for means: in annuli of 0-10 pkpc, then 0.25 dex bins up to ~3.5 R200c')
    #print('for medians: in annuli of 10 pkpc up to 100 pkpc, then 0.1 dex bins up to ~3.5 R200c')
    print('for median of means: annuli of 0.1 dex starting from 10 pkpc')
    print('z=0.1, Ref-L100N1504, 6.25 cMpc slice Z-projection, SmSb, C2 kernel')
    print('Using max. 1000 (random) galaxies in each mass bin, centrals only')
    
    # get minimum SB for the different instruments
    omegat_use = [1e6, 1e7]
    if talkversion:
        legendtitle_minsb = 'min. SB ($5\\sigma$) for $\\Delta' +\
            ' \\Omega \\, \\Delta t =$ \n ${:.0e}, {:.0e}'.format(*tuple(omegat_use))+\
            '\\, \\mathrm{arcmin}^{2} \\, \\mathrm{s}$'
    else:
        legendtitle_minsb = 'min. SB ($5\\sigma$ detection) for $\\Delta' +\
            ' \\Omega \\, \\Delta t = {:.0e}, {:.0e}'.format(*tuple(omegat_use))+\
            '\\, \\mathrm{arcmin}^{2} \\, \\mathrm{s}$'
    
    filen_SB = 'minSBtable.dat'
    filen_PS20 = 'minSBtable_PS20_Fe-L-shell.dat'
    df = pd.read_csv(ddir + filen_SB, sep='\t')   
    _df = pd.read_csv(ddir + filen_PS20, sep='\t')
    df = pd.concat([df, _df], ignore_index=True)
    del _df
    groupby = ['line name', 'linewidth [km/s]',
               'sky area * exposure time [arcmin**2 s]', 
               'full measured spectral range [eV]',
               'detection significance [sigma]', 
               'galaxy absorption included in limit',
               'instrument']
    df2 = df.groupby(groupby)['minimum detectable SB [phot/s/cm**2/sr]'].mean().reset_index()
    zopts = df['redshift'].unique()
    print('Using redshifts for min SB: {}'.format(zopts))
    
    expectedsame = ['linewidth [km/s]', 'detection significance [sigma]']
    for colname in expectedsame:
        if np.allclose(df2[colname], df2.at[0, colname]):
            print('Using {}: {}'.format(colname, df2.at[0, colname]))
            del df2[colname]
            groupby.remove(colname)
        else:
            raise RuntimeError('Multiple values for {}; choose one'.format(colname))
            
    instruments = ['athena-xifu', 'lynx-lxm-main', 'lynx-lxm-uhr', 'xrism-resolve']        
    inslabels = {'athena-xifu': 'Athena X-IFU',
                 'lynx-lxm-main': 'Lynx Main',
                 'lynx-lxm-uhr': 'Lynx UHR',
                 'xrism-resolve': 'XRISM-R'
                 }        
    _kwargs = {'facecolor': 'none', 'edgecolor': 'gray'}
    kwargs_ins = {'athena-xifu':   {'hatch': '||'},
                  'lynx-lxm-main': {'hatch': '\\\\'},
                  'lynx-lxm-uhr':  {'hatch': '//'},
                  'xrism-resolve': {'hatch': '---'},
                  }        
    for key in kwargs_ins:
        kwargs_ins[key].update(_kwargs)
        
    # taken from the table in the paper (half of PSF and FOV)
    xmin_ins = {'athena-xifu':   0.5 * arcmin_to_pkpc(5. / 60., z=0.1),
                'lynx-lxm-main': 0.5 * arcmin_to_pkpc(0.5 / 60., z=0.1),
                'lynx-lxm-uhr':  0.5 * arcmin_to_pkpc(0.5 / 60., z=0.1),
                'xrism-resolve': 0.5 * arcmin_to_pkpc(72. / 60., z=0.1),
                }   
    xmax_ins = {'athena-xifu':   0.5 * arcmin_to_pkpc(5., z=0.1),
                'lynx-lxm-main': 0.5 * arcmin_to_pkpc(5., z=0.1),
                'lynx-lxm-uhr':  0.5 * arcmin_to_pkpc(1., z=0.1),
                'xrism-resolve': 0.5 * arcmin_to_pkpc(2.9, z=0.1),
                }  
                              
    if talkversion:
        fontsize = 14
    else:
        fontsize = 12
    linewidth = 1.5
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="b"),\
               mppe.Stroke(linewidth=linewidth, foreground="w"),\
               mppe.Normal()]
    xlabel = '$\\mathrm{r}_{\perp} \\; [\\mathrm{pkpc}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{SB} \\; ' + \
             '[\\mathrm{photons}\\,\\mathrm{cm}^{-2}' + \
             '\\mathrm{s}^{-1}\\mathrm{sr}^{-1}]$'
    y2label = '$\\log_{10} \\, \\mathrm{SB} \\; ' + \
              '[\\mathrm{photons}\\,\\mathrm{m}^{-2} ' + \
              '(100 \\,\\mathrm{ks})^{-1}(10\\,\\mathrm{arcmin}^{2})^{-1}]$'
    right_over_left = 1e4 * 1e5 * ((10. * np.pi**2 / 60.**2 / 180**2))
    # right_over_left = (ph / cm**2 / s / sr)  /  (ph / m**2 / 100 ks / 10 arcmin^2)
    # right_over_left = m**2 / cm**2 * 100 ks / s * 10 arcmin**2 / rad**2 
    # value ratio is inverse of unit ratio    
    ys = [('mean',), ('perc', 50.)]
    ykey_mean = ('mean',)
    ykey_median = ('perc', 50.)
    ls_mean = 'dotted'
    ls_median = 'solid'
    
    outname = 'radprof2d_0.1-0.25dex-annuli_L0100N1504_27_test3.x_SmAb_C2Sm_6.25slice_noEOS_to-2R200c_1000_centrals_' +\
              'halomasscomp_mean-median'
    outname = outname.replace('.', 'p')
    if talkversion:
        if talkvnum == 0:
            outname = tmdir + outname + '_talkversion_{}'.format(slidenum)
        else:
            outname = tmdir + outname +\
                      '_talkversion-{}_{}'.format(talkvnum, slidenum)
    else:
        outname = mdir + outname
    outname = outname + '.pdf'
    
    if talkversion:
        if talkvnum == 0:
            mmin = 11.
        elif talkvnum in [1, 2, 3]:
            mmin = 11.5
    else:
        mmin = 11.

    medges = np.arange(mmin, 14.1, 0.5)
    seltag_keys = {medges[i]: 'geq{:.1f}_le{:.1f}'.format(medges[i], medges[i + 1])\
                               if i < len(medges) - 1 else\
                               'geq{:.1f}'.format(medges[i])\
                    for i in range(len(medges))}
    seltags = [seltag_keys[key] for key in seltag_keys]
    
    if talkversion:
        _lines = ['c6', 'o7r', 'o8', 'mg12']
        if talkvnum == 2:
            _lines = ['o7r', 'o8', 'ne10', 'mg12']
        elif talkvnum == 3:
            _lines = ['c6', 'n7', 'o8', 'ne10', 'mg12']
        
        numlines = len(_lines)
        fontsize = 14
        
        ncols = 3
        nrows = (numlines - 1) // ncols + 1
        figwidth = 11. 
        caxwidth = figwidth / float(ncols + 1)
        
    else:
        _lines = plot_lines_SB + plot_lines_PS20
        _lines.sort(key=line_energy_ev)
        numlines = len(_lines)
        ncols = 4
        nrows = (numlines - 1) // ncols + 1
        figwidth = 11. 
        caxwidth = 1.
    
    if ncols * nrows - numlines >= 2:
        cax_right = False
        _ncols = ncols
        panelwidth = figwidth / ncols
        width_ratios = [panelwidth] * ncols
        c_orientation = 'horizontal'
        c_aspect = 0.08
    else:
        cax_right = True
        _ncols = ncols + 1
        panelwidth = (figwidth - caxwidth) / ncols
        width_ratios = [panelwidth] * ncols + [caxwidth]
        c_orientation = 'vertical'
        c_aspect = 10.
    
    rfilebase = 'radprof_stamps_emission_{line}{it}_L0100N1504_27_' + \
                'test3.{tv}_' + \
                'SmAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_noEOS' + \
                '_1slice_to-min3p5R200c_Mh0p5dex_1000' +\
                '_centrals_M-ge-10p5.hdf5'
    siontab = {line: '_iontab-PS20-UVB-dust1-CR1-G1-shield1_depletion-F' \
               if line in all_lines_PS20 else '' for line in _lines}
    testversion = {line: '7' if line in all_lines_PS20 else '6' \
                   for line in _lines}
    binset_mean = 'binset_0'
    binset_medianofmeans = 'binset_1'
    cosmopars = cosmopars_27 # for virial radius indicators
    filens = {}
    for line in _lines:
        _filen = ddir + 'radprof/' +\
                 rfilebase.format(line=line.replace(' ', '-'), 
                                  it=siontab[line], tv=testversion[line])
        if not os.path.isfile(_filen):
            _filen = _filen.replace('zcen-all', 'zcen3.125') # file naming bug
            if not os.path.isfile(_filen):
                if line in ['n6-actualr', 'ne10']: # other file naming thing
                    #_filen = _filen.replace('zcen3.125', 'zcen-all')
                    _filen = _filen.replace('Mh0p5dex', 
                                            'L0100N1504_27_Mh0p5dex')
        if not os.path.isfile(_filen):
            msg = 'Could not find file for {}:\n{}'
            raise RuntimeError(msg.format(line, _filen))
            
        filens[line] = _filen
        #print(filens[line])
        
    panelheight = panelwidth    
    figheight = panelheight * nrows
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(ncols=_ncols, nrows=nrows, hspace=0.0, wspace=0.0,\
                        width_ratios=width_ratios)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) for i in range(numlines)]
    if cax_right:
        ncols_insleg = 1 
        leg_kw = {'loc': 'upper left', 'bbox_to_anchor': (0.15, 1.)}
        insleg_kw = leg_kw.copy()
        if nrows > 5: 
            csl = slice(nrows // 2 - 1, nrows // 2 + 2, None)
            lsl = slice(0, 1, None)
            l2sl = slice(1, 2, None)
        elif nrows > 2:
            csl = slice(2, None, None)
            lsl = slice(0, 1, None)
            l2sl = slice(1, 2, None)
        elif nrows == 2:
            csl = slice(1, None, None)
            lsl = slice(0, 1, None)
            l2sl = slice(0, 1, None)
            insleg_kw = {'loc': 'lower left',
                     'bbox_to_anchor': (0.15, 0.0),
                     'handlelength': 1.8,
                     'columnspacing': 0.8,
                     }
        else:
            raise RuntimeError('Could not find a place for the legend and color bar at the right of the plot (1 row)')
        cax = fig.add_subplot(grid[csl, ncols])
        lax = fig.add_subplot(grid[lsl, ncols])
        lax.axis('off')
        lax2 = fig.add_subplot(grid[l2sl, ncols])
        lax2.axis('off')
        
    else:
        ind_min = ncols - (nrows * ncols - numlines)
        _cax = fig.add_subplot(grid[nrows - 1, ind_min:])
        _cax.axis('off')
        _l, _b, _w, _h = (_cax.get_position()).bounds
        vert = nrows * ncols - numlines <= 2
        if vert:
            wmargin_c = panelwidth * 0.13 / figwidth
            wmargin_l = panelwidth * 0.05 / figwidth
            hmargin_b = panelheight * 0.07 / figheight
            hmargin_t = panelheight * 0.07 / figheight
            lspace = 0.3 * panelheight / figheight
            cspace = _h - hmargin_b - hmargin_t - lspace
            cax = fig.add_axes([_l + wmargin_c, _b + hmargin_b,\
                                _w - wmargin_c, cspace])
            w1 = 0.35 * (_w - 1. * wmargin_l)
            w2 = 0.65 * (_w - 1. * wmargin_l)
            lax = fig.add_axes([_l + wmargin_l, _b  + hmargin_b + cspace,\
                                w1, lspace])
            lax2 = fig.add_axes([_l + _w - w2 * 0.975, _b  + hmargin_b + cspace,\
                                w2, lspace])
                
            ncols_insleg = (len(instruments) + 1) // 2 
            
            leg_kw = {'loc': 'upper left',
                  'bbox_to_anchor': (0.0, 1.),
                  'handlelength': 2.,
                  'columnspacing': 1.,
                  }
            insleg_kw = {'loc': 'upper right',
                  'bbox_to_anchor': (1.0, 1.),
                  'handlelength': 1.8,
                  'columnspacing': 0.8,
                  }
            
        else:
            wmargin = panelwidth * 0.1 / figwidth
            hmargin = panelheight * 0.05 / figheight
            
            vspace = 0.25 * panelheight / figheight
            hspace_c = 0.7 * (_w - 3. * wmargin)
            hspace_l = _w - 3. * wmargin - hspace_c
            
            
            cax = fig.add_axes([_l + 2. * wmargin + hspace_l, _b + hmargin,\
                                hspace_c, vspace])
            lax = fig.add_axes([_l + wmargin, _b,\
                                hspace_l, vspace])
            lax2 = fig.add_axes([_l + wmargin, _b  + vspace + hmargin,\
                                _w - 2. * wmargin, vspace])    
            ncols_insleg = 4
             
            leg_kw = {'loc': 'center left',
                      'bbox_to_anchor':(0., 0.5),
                      'handlelength': 2.,
                      'columnspacing': 1.,
                      }
            insleg_kw = {'loc': 'center left',
                      'bbox_to_anchor':(0., 0.5),
                      'handlelength': 2.,
                      'columnspacing': 1.,
                      }
        lax.axis('off')
        lax2.axis('off')
        
        
    labelax = fig.add_subplot(grid[:nrows, :ncols], frameon=False)
    labelax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    labelax.set_ylabel(ylabel, fontsize=fontsize)
    l2ax = labelax.twinx()
    l2ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    l2ax.spines['right'].set_visible(False)
    l2ax.spines['top'].set_visible(False)
    l2ax.spines['bottom'].set_visible(False)
    l2ax.spines['left'].set_visible(False)
    l2ax.set_ylabel(y2label, fontsize=fontsize)
    
    ind_min = ncols - (nrows * ncols - numlines)
    if nrows * ncols - numlines <= 2:
        labelax.set_xlabel(xlabel, fontsize=fontsize)    
    else:
        labelax1 = fig.add_subplot(grid[:nrows, :ind_min], frameon=False)
        labelax1.tick_params(labelcolor='none', top=False, bottom=False,
                             left=False, right=False)
        labelax1.set_xlabel(xlabel, fontsize=fontsize) 
        
        labelax2 = fig.add_subplot(grid[:nrows - 1, ind_min:], frameon=False)
        labelax2.tick_params(labelcolor='none', top=False, bottom=False,
                             left=False, right=False)
        labelax2.set_xlabel(xlabel, fontsize=fontsize) 
    #l2ax.yaxis.set_label_position('right')
    #l2ax.axis('off')
    
    
    clabel = '$\\log_{10} \\, \\mathrm{M}_{\\mathrm{200c}} \\; [\\mathrm{M}_{\\odot}]$'
    cbar, colordct = add_cbar_mass(cax, massedges=medges,\
             orientation=c_orientation, clabel=clabel, fontsize=fontsize,\
             aspect=c_aspect)
    
    axes2 =[]
    for li, line in enumerate(_lines):
        ax = axes[li]
        labely = li % ncols == 0
        labelx = numlines -1 - li < ncols
        labelright = (li % ncols == ncols - 1) or (li == len(axes) - 1)
        pu.setticks(ax, fontsize=fontsize, labelleft=labely, labelbottom=labelx,\
                    right=False)
        ax.set_xscale('log')
        ax.grid(b=True)
        ax2 = ax.twinx()
        pu.setticks(ax2, fontsize=fontsize, left=False, right=True, bottom=False,\
                    top=False, labelright=labelright, labelleft=False, labeltop=False,\
                    labelbottom=False)
        axes2.append(ax2)
        
        filename = filens[line]
        #print(line)
        #print(filename)
        yvals, bins = readin_radprof(filename, seltags, [ykey_mean],
                                     runit='pkpc', separate=False,
                                     binset=binset_mean, retlog=True,
                                     ofmean=True)
        _yvals, _bins = readin_radprof(filename, seltags, [ykey_median],
                                       runit='pkpc', separate=False,
                                     binset=binset_medianofmeans, retlog=True,
                                     ofmean=True)
        #print(bins.keys())
        #print(_bins.keys())
        for tag in yvals:
            #print(tag)
            bins[tag].update(_bins[tag])
            yvals[tag].update(_yvals[tag])
        for mi, me in enumerate(medges):
            tag = seltag_keys[me]
            
            # plot profiles
            for ykey, ls, zo in zip([ykey_mean, ykey_median], 
                                    [ls_mean, ls_median], 
                                    [5, 6]):
                if talkversion:
                    if talkvnum == 2:
                        if ykey == ykey_mean:
                            continue
                    if ykey == ykey_mean:
                        yi = 1
                    elif ykey == ykey_median:
                        yi = 0
                    else:
                        yi = 2 
                    ncomp = 1 + li * len(medges) * len(ys) \
                            + yi * len(medges) + mi
                    if ncomp > slidenum:
                        continue
                ed = bins[tag][ykey]
                vals = yvals[tag][ykey]
                cens = ed[:-1] + 0.5 * np.diff(ed)
                ax.plot(cens, vals, color=colordct[me], linewidth=2.,\
                        path_effects=patheff, linestyle=ls, zorder=zo)
            
            # indicate R200c
            mmin = 10**me
            if mi < len(medges) - 1:
                mmax = 10**medges[mi + 1]
            else:
                mmax = 10**14.53 # max mass in the box at z=0.1
            rs = cu.R200c_pkpc(np.array([mmin, mmax]), cosmopars)
            ax.axvspan(rs[0], rs[1], ymin=0, ymax=1, alpha=0.1, \
                       color=colordct[me])
        
        # might not work for generic lines
        ion = line.split('-')[0]
        while not ion[-1].isdigit():
            ion = ion[:-1]            
        linelabel = ild.getnicename(ion)
        if ion == 'o7': # get (i), (r), (f) label:
            linelabel = nicenames_lines[line]
            
        ev = ol.line_eng_ion[line] / c.ev_to_erg
        numdig = 4
        lead = int(np.ceil(np.log10(ev)))
        appr = str(np.round(ev, numdig - lead))
        if '.' in appr:
            if len(appr) < numdig + 1: # trailing zeros after decimal point cut off
                appr = appr + '0' * (numdig + 1 - len(appr))
            elif len(appr) >= numdig + 2 and appr[-2:] == '.0': # .0 added to floats: remove
                appr = appr[:-2]
        eng = '{} eV'.format(appr)
        linelabel =  eng + '\n' + linelabel 
        ax.text(0.98, 0.97, linelabel, fontsize=fontsize,\
                transform=ax.transAxes, horizontalalignment='right',\
                verticalalignment='top')
        if li == 0 and not (talkversion and talkvnum == 2):
            handles = [mlines.Line2D([],[], label=label, color='black', ls=ls,\
                                     linewidth=2.) \
                       for ls, label in zip([ls_mean, ls_median], ['mean', 'med. mean'])]
            lax.legend(handles=handles, fontsize=fontsize, **leg_kw)
        
        # add SB mins
        _sel = df2['galaxy absorption included in limit']
        _xlim = ax.get_xlim()
        _ylim = ax.get_ylim()
        for ins in instruments: 
            if talkversion and talkvnum == 2:
                if 'lynx' in ins:
                    continue
                
            sel = np.logical_and(_sel, df2['instrument'] == ins)
            sel &= df2['line name'] == line
            
            minsel = sel & np.isclose(df2['sky area * exposure time [arcmin**2 s]'],
                                      np.min(omegat_use))
            maxsel = sel & np.isclose(df2['sky area * exposure time [arcmin**2 s]'],
                                      np.max(omegat_use))
            if np.sum(maxsel) != 1 or np.sum(minsel) != 1:
                msg = 'Something went wrong finding the SB limits:'
                msg = 'selected {}, {} values'.format(len(minsel), len(maxsel))
                raise RuntimeError(msg)
            imin = df2.index[minsel][0]
            imax = df2.index[maxsel][0]
            miny = np.log10(df2.at[imin, 'minimum detectable SB [phot/s/cm**2/sr]'])
            maxy = np.log10(df2.at[imax, 'minimum detectable SB [phot/s/cm**2/sr]'])
            
            minx = xmin_ins[ins]
            maxx = xmax_ins[ins]
            #print(minx, maxx, miny, maxy)
            
            patch = mpatch.Rectangle([minx, miny], maxx - minx, maxy - miny,
                                     **kwargs_ins[ins])
            ax.add_artist(patch)
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)
        
    # sync plot ranges
    xlims = [ax.get_xlim() for ax in axes]
    xmin = min([xlim[0] for xlim in xlims])
    xmax = max([xlim[1] for xlim in xlims])
    if talkversion:
        xmin = 4.
        xmax = 4.5e3
    [ax.set_xlim(xmin, xmax) for ax in axes]

    # three most energetic ions have very low mean SB -> impose limits
    #ylims = [ax.get_ylim() for ax in axes]
    if talkversion:
        ymin = -2.95 #min([ylim[0] for ylim in ylims])
    else:
        ymin = -3.95
    ymax = 2. #max([ylim[1] for ylim in ylims])
    [ax.set_ylim(ymin, ymax) for ax in axes]
    [ax.set_ylim(ymin + np.log10(right_over_left), 
                 ymax + np.log10(right_over_left))\
     for ax in axes2]
    
    _inss = list(np.copy(instruments))
    if talkversion and talkvnum == 2:
        for ins in instruments:
            if 'lynx' in ins:
                _inss.remove(ins)
            
    handles_ins = [mpatch.Patch(label=inslabels[ins], **kwargs_ins[ins]) \
                   for ins in _inss]    
    leg_ins = lax2.legend(handles=handles_ins, fontsize=fontsize, 
                          ncol=ncols_insleg, **insleg_kw)
    leg_ins.set_title(legendtitle_minsb)
    leg_ins.get_title().set_fontsize(fontsize)
    
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
    
    