#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:23:46 2021

@author: Nastasha

test the SP20 tables implementation and compare to Serena Bertone's tables
"""

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.cm as cm

import make_maps_v3_master as m3
import make_maps_opts_locs as ol
import cosmo_utils as cu
import plot_utils as pu

from importlib import reload
reload(m3) # testing script -> make sure I'm using the latest version

mdir  = '/net/luttero/data1/line_em_abs/v3_master_tests/ssh_tables_SP20/'
m3.ol.ndir = mdir

ions = ['h1ssh', 'hmolssh', 'mg2', 'si4', 'o6', 'o7', 'o8', 'ne8',\
        'ne9', 'ne10', 'fe17']


# check line data for these
lines_SP20 = ['C  5      40.2678A',
              'C  6      33.7372A',
              'N  6      29.5343A',
              'N  6      28.7870A',
              'N  7      24.7807A',
              'O  7      21.6020A',
              'O  7      21.8044A',
              'O  7      21.8070A',
              'O  7      22.1012A',
              'O  8      18.9709A',
              'Ne 9      13.4471A',
              'Ne10      12.1375A',
              'Mg11      9.16875A',
              'Mg12      8.42141A',
              'Si13      6.64803A',
              'Fe17      17.0510A',
              'Fe17      15.2620A',
              'Fe17      16.7760A',
              'Fe17      17.0960A',
              'Fe18      16.0720A',
              ]

lines_SB = ['c5r', 'n6-actualr', 'ne9r', 'ne10', 'mg11r', 'mg12', 'si13r',
            'fe18', 'fe17-other1', 'fe19', 'o7r', 'o7iy', 'o7f', 'o8', 'fe17',
            'c6', 'n7']

nicenames_lines =  {'c5r': 'C V',\
                    'n6r': 'N VI (f)',\
                    'n6-actualr': 'N VI',\
                    'ne9r': 'Ne IX',\
                    'ne10': 'Ne X',\
                    'mg11r': 'Mg XI',\
                    'mg12': 'Mg XII',\
                    'si13r': 'Si XIII',\
                    'fe18': 'Fe XVIII',\
                    'fe17-other1': 'Fe XVII (15.10 A)',\
                    'fe19': 'Fe XIX',\
                    'o7r': 'O VII (r)',\
                    'o7ix': 'O VII (ix)',\
                    'o7iy': 'O VII (i)',\
                    'o7f': 'O VII (f)',\
                    'o8': 'O VIII',\
                    'fe17': 'Fe XVII (17.05 A)',\
                    'c6': 'C VI',\
                    'n7': 'N VII',\
                    }

linematch_SP20 = {'C  5      40.2678A': 'c5r',
                  'C  6      33.7372A': 'c6',
                  'N  6      29.5343A': 'n6r',
                  'N  6      28.7870A': 'n6-actualr',
                  'N  7      24.7807A': 'n7',
                  'O  7      21.6020A': 'o7f',
                  'O  7      21.8044A': 'o7iy',
                  'O  7      21.8070A': 'o7i',
                  'O  7      22.1012A': 'o7r',
                  'O  8      18.9709A': 'o8',
                  'Ne 9      13.4471A': 'ne9r',
                  'Ne10      12.1375A': 'ne10',
                  'Mg11      9.16875A': 'mg11r',
                  'Mg12      8.42141A': 'mg12',
                  'Si13      6.64803A': 'si13r',
                  'Fe17      17.0510A': 'fe17r',
                  'Fe17      15.2620A': None,
                  'Fe17      16.7760A': None,
                  'Fe17      17.0960A': None,
                  'Fe18      16.0720A': None,
                  }

zeroval_PS20 = -50.
zeroval_SB = -100.

def plottables_PS20(line, z, table='emission'):
    
    fontsize = 12
    xlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{3}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
    
    zeroval = zeroval_PS20
    
    if table == 'emission':
        title = 'emissivity of the {line} line at $z = {z:.2f}$'
        clabel = '$\\log_{{10}} \\, \\mathrm{{\\Lambda}} \\,\\mathrm{{n}}_' +\
         '{{\\mathrm{{H}}}}^{{-2}} \\, \\mathrm{{V}}^{{-1}}  \\;' +\
         ' [\\mathrm{{erg}} \\, \\mathrm{{cm}}^{{3}} \\mathrm{{s}}^{{-1}}]$'
         
        tab = m3.linetable_PS20(line, z, emission=True)
        tab.findemtable()
        table_T_Z_nH = np.copy(tab.emtable_T_Z_nH) 
        
        tozero = table_T_Z_nH == zeroval
        table_T_Z_nH -= 2.* tab.lognHcm3[np.newaxis, np.newaxis, :]
        table_T_Z_nH[tozero] = zeroval

    elif table == 'dust':
        title = 'dust depletion fraction of {elt} at $z = {z:.2f}$'
        clabel = '$\\log_{{10}}$ fraction of {elt} in dust'
        
        tab = m3.linetable_PS20(line, z, emission=False)
        tab.finddepletiontable()
        table_T_Z_nH = np.copy(tab.depletiontable_T_Z_nH)
        
    elif table == 'ionbal':
        title = 'fraction {ion} / {elt} at $z = {z:.2f}$'
        clabel = '$\\log_{{10}} \\; \\mathrm{{n}}(\\mathrm{{{ion}}}) \\, /' + \
            ' \\, \\mathrm{{n}}(\\mathrm{{{elt}}})$'
            
        tab = m3.linetable_PS20(line, z, emission=False)
        tab.findiontable()
        table_T_Z_nH = np.copy(tab.iontable_T_Z_nH)
    
    kws = {'ion': line, 'line': line, 'elt': tab.elementshort, 'z': tab.z}
    title = title.format(**kws)
    clabel = clabel.format(**kws)

    deltaT = np.average(tab.logTK)
    deltanH = np.average(tab.lognHcm3)
    extent = (tab.lognHcm3[0] - 0.5 * deltanH, tab.lognHcm3[-1] + 0.5 * deltanH,
              tab.logTK[0] - 0.5 * deltaT, tab.logTK[-1] + 0.5 * deltaT)
      
    if np.all(table_T_Z_nH == zeroval):
        msg = 'skipping line {line}: no emission!'
        msg = msg.format(line=line)
        print(msg)
        return
        
    vmin = np.min(table_T_Z_nH[table_T_Z_nH > zeroval])
    vmax = np.max(table_T_Z_nH)
    #table = np.log10(table)
    
    ncols = 4
    numZ = len(tab.logZsol)
    nrows = (numZ - 1) // ncols + 1
    figwidth = 11.
    cwidth = 1.
    panelwidth = (figwidth - cwidth) / float(ncols)
    panelheight = panelwidth
    figheight = panelheight * nrows
    width_ratios = [panelwidth] * ncols + [cwidth]
    
    fig = plt.figure(figsize=(figwidth, figheight))
    grid = gsp.GridSpec(ncols=ncols + 1, nrows=nrows,
                        hspace=0.0, wspace=0.0,
                        width_ratios=width_ratios)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) \
            for i in range(numZ)]
    cax = fig.add_subplot(grid[:, ncols]) 
    
    cmap = cm.get_cmap('viridis')
    cmap.set_under('white')
    
    for mi, logZ in enumerate(tab.logZsol):
        ax = axes[mi]
        
        left = mi % ncols == 0 
        bottom = numZ - mi <= ncols 
        pu.setticks(ax, fontsize, labelleft=left,
                    labelbottom=bottom)
        if bottom:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if left:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        if np.isclose(logZ, -50.):
            labelZ = 'primordial'
        else:
            labelZ = '$\\log_{{10}} \\, \\mathrm{{Z}}\\, /' +\
                ' \\, \\mathrm{{Z}}_{{\\odot}} = {logZ}$'
            labelZ = labelZ.format(logZ=logZ)
        img = ax.imshow(table_T_Z_nH[:, mi, :], interpolation='nearest',
                        origin='lower', extent=extent, cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.set_aspect((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))
        
        ax.text(0.05, 0.95, labelZ, fontsize=fontsize,
                transform=ax.transAxes, horizontalalignment='left',
                verticalalignment='top')

    # color bar 
    pu.add_colorbar(cax, img=img, cmap=cmap, vmin=vmin,
                    clabel=clabel, fontsize=fontsize, 
                    orientation='vertical', extend='min')
    cax.set_aspect(10.)

    fig.suptitle(title.format(line=line, z=z), fontsize=fontsize)
    
    _line = line.replace(' ', '_')
    outname = mdir+ 'PS20_table_{line}_z{z:.2f}.pdf'
    outname = outname.format(line=_line, z=z)
    plt.savefig(outname, format='pdf', bbox_inches='tight')        
     

def plottables_SB(line, z, table='emission'):
    
    fontsize = 12
    xlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{3}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
    
    if table == 'emission':
        title = 'emissivity of the {line} line at $z = {z:.2f}$'
        clabel = '$\\log_{{10}} \\, \\mathrm{{\\Lambda}} \\,\\mathrm{{n}}_' +\
         '{{\\mathrm{{H}}}}^{{-2}} \\, \\mathrm{{V}}^{{-1}}  \\;' +\
         ' [\\mathrm{{erg}} \\, \\mathrm{{cm}}^{{3}} \\mathrm{{s}}^{{-1}}]$'
         
        eltlines, logTK, lognHcm3 = cu.findemtables(ol.elements_ion[line], z)
        table_T_nH = np.copy(eltlines[:, :, ol.line_nos_ion[line]])

    elif table == 'dust':
        raise ValueError("Dust tables are not available in Serena's set")
        title = 'dust depletion fraction of {elt} at $z = {z:.2f}$'
        clabel = '$\\log_{{10}}$ fraction of {elt} in dust'
        
    elif table == 'ionbal':
        title = 'fraction {ion} / {elt} at $z = {z:.2f}$'
        clabel = '$\\log_{{10}} \\; \\mathrm{{n}}(\\mathrm{{{ion}}}) \\, /' + \
            ' \\, \\mathrm{{n}}(\\mathrm{{{elt}}})$'
            
        balance, logTK, lognHcm3 = cu.findiontables(line, z)
        table_T_nH = balance.T
    
    kws = {'ion': line, 'line': nicenames_lines[line], 
           'elt': ol.elements_ion[line], 'z': z}
    title = title.format(**kws)
    clabel = clabel.format(**kws)
    
    deltaT = np.average(np.diff(logTK))
    deltanH = np.average(np.diff(lognHcm3))
    extent = (lognHcm3[0] - 0.5 * deltanH, lognHcm3[-1] + 0.5 * deltanH,
              logTK[0] - 0.5 * deltaT, logTK[-1] + 0.5 * deltaT)

    fig, (ax, cax) = plt.subplots(ncols=2, nrows=1,
                                  gridspec_kw={'width_ratios': [5., 1.],
                                               'wspace': 0.3})     
    cmap = cm.get_cmap('viridis')
    cmap.set_under('white')
    
    zeroval = zeroval_SB
    table_T_nH[table_T_nH == zeroval] = -np.inf
    vmin = np.min(table_T_nH[table_T_nH > zeroval])
    
    img = ax.imshow(table_T_nH, interpolation='nearest', origin='lower', 
                    extent=extent, cmap=cmap, vmin=vmin)
    
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    txt = '$\\mathrm{{Z}}_{{\\odot}}$: {:.2e} '+\
          '$[\\mathrm{{n}}_{{\\mathrm{{H}}}}]$'
    txt = txt.format(ol.solar_abunds_sb[ol.elements_ion[line]]) 
    ax.text(0.05, 0.95, txt, fontsize=fontsize,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='top')
    
    # color bar 
    pu.add_colorbar(cax, img=img, cmap=cmap, vmin=vmin,
                    clabel=clabel, fontsize=fontsize, 
                    orientation='vertical', extend='min')
    cax.set_aspect(10.)
    
    fig.suptitle(title.format(line=nicenames_lines[line], z=z),
                 fontsize=fontsize)
    
    outname = mdir + 'SB_{table}_table_{line}_z{z:.2f}.pdf'
    outname = outname.format(line=line, z=z, table=table)
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
# test basic table retrieval and sensitbility
def plot_tables(zs):
    for z in zs:
        for line in lines_SP20:
            plottables_PS20(line, z, table='emission')
        for line in lines_SB:
            plottables_SB(line, z, table='emission')
        for ion in ions:
            plottables_PS20(line, z, table='ionbal')
            plottables_PS20(line, z, table='dust')
            plottables_SB(line, z, table='ionbal')

# compare sets of tables: second reasonability check and assesment of 
# differences
def compare_tables(line_PS20, line_SB, z, table='emission'):
    pass

# test interpolation of the tables graphically
def test_interp(lines_PS20, table='emission'):
    pass

# compare maps with the two table sets
# emission maps for a few ions, with and without abundance adjustments
# absorption maps
# no-dust maps with both table sets
# with and without dust depletion

def compare_maps(args_map1, args_map2, kwargs_map1, kwargs_map2,\
                 imgname=None):
    pass

if __name__ == '__main__':
    zs_test = [0.0, 0.1, 1., 3.]  
    plot_tables(zs_test)
      
            
