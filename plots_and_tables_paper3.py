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
    
    filebase = ddir 'stamps/' + \
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
    _lines = sorted(maps.keys(), key=ol.line_eng_ion.get)
    
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












