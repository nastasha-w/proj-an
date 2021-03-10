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
import matplotlib.patheffects as mppe
import matplotlib.lines as mlines

import tol_colors as tc

import make_maps_v3_master as m3
import make_maps_opts_locs as ol
import calcfmassh as cfh
import cosmo_utils as cu
import plot_utils as pu
from ion_line_data import element_to_abbr

from importlib import reload
reload(m3) # testing script -> make sure I'm using the latest version

mdir  = '/net/luttero/data1/line_em_abs/v3_master_tests/ssh_tables_SP20/'
m3.ol.ndir = mdir

ions = ['h1ssh', 'hmolssh', 'mg2', 'o1', 'si4', 'o6', 'o7', 'o8', 'ne8',\
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

zeroval_PS20 = -50.
zeroval_SB = -100.

def gethssh_R13(ion, z, logTK=None, lognHcm3=None):
    
    if logTK is None:
        logTK = np.arange(2., 8.1, 0.2)
    if lognHcm3 is None:
        lognHcm3 = np.arange(-8., 4.1, 0.2)
    
    Tgrid = np.array([[x] * len(lognHcm3) for x in logTK]).flatten()
    ngrid = np.array([lognHcm3] * len(logTK)).flatten()
    dct = {'Temperature': 10**Tgrid, 'nH': 10**ngrid, 
           'eos': np.zeros(len(Tgrid), dtype=bool)}
    
    h1hmolfrac = cfh.nHIHmol_over_nH(dct, z, UVB='HM01', useLSR=False)
    if ion == 'h1ssh':
        h1hmolfrac *= (1. - cfh.rhoHmol_over_rhoH(dct))
    elif ion == 'hmolssh':
        h1hmolfrac *= cfh.rhoHmol_over_rhoH(dct)
    else: # 'hneutralssh' -> want just the total
        raise ValueError('{} is not an ion option'.format(ion))
    
    table = h1hmolfrac.reshape((len(logTK), len(lognHcm3)))
    return table, logTK, lognHcm3 
    

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
        clabel = '$\\log_{{10}} \\; \\mathrm{{m}}(\\mathrm{{{ion}}}) \\, /' + \
            ' \\, \\mathrm{{m}}(\\mathrm{{{elt}}})$'
        
        if line in ['h1ssh', 'hmolssh']:
            table_T_nH, logTK, lognHcm3 = gethssh_R13(line, z)
            table_T_nH = np.log10(table_T_nH)
            zeroval = -50. # whatever
        else:
            balance, logTK, lognHcm3 = cu.findiontables(line, z)
            table_T_nH = np.log10(balance.T)
            zeroval = np.min(table_T_nH[np.isfinite(table_T_nH)])
    
    if line in nicenames_lines:
        linen = nicenames_lines[line]
    else:
        linen = line
    kws = {'ion': line, 'line': linen, 
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
    
    txt = '$\\mathrm{{Z}}_{{\\odot}}$: '+\
          '$\\mathrm{{n}}_{{\\mathrm{{{elt}}}}} \\, / ' + \
              '\\, \\mathrm{{n}}_{{\\mathrm{{H}}}} =$ {abund:.2e}'
    txt = txt.format(abund=ol.solar_abunds_sb[ol.elements_ion[line]],\
                     elt=element_to_abbr[ol.elements_ion[line]]) 
    ax.text(0.05, 0.95, txt, fontsize=fontsize,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='top')
    
    # color bar 
    pu.add_colorbar(cax, img=img, cmap=cmap, vmin=vmin,
                    clabel=clabel, fontsize=fontsize, 
                    orientation='vertical', extend='min')
    cax.set_aspect(10.)
    
    fig.suptitle(title, fontsize=fontsize)
    
    outname = mdir + 'SB_{table}_table_{line}_z{z:.2f}.pdf'
    outname = outname.format(line=line, z=z, table=table)
    plt.savefig(outname, format='pdf', bbox_inches='tight')


# compare sets of tables: second reasonability check and assesment of 
# differences
def compare_tables(line_PS20, line_SB, z, table='emission'):
        
    fontsize = 12
    xlabel = '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}} \\; [\\mathrm{cm}^{3}]$'
    ylabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
    cierellabel = '$\\Delta \\log_{10}$'
    
    title_SB = "Serena Bertone's tables"
    title_PS20 = 'Ploeckinger & Schaye (2020)'
        
    if table == 'emission':
        title = 'emissivity of the {line_PS20} / {line_SB} line at $z = {z:.2f}$'
        clabel = '$\\log_{{10}} \\, \\mathrm{{\\Lambda}} \\,\\mathrm{{n}}_' +\
         '{{\\mathrm{{H}}}}^{{-2}} \\, \\mathrm{{V}}^{{-1}}  \\;' +\
         ' [\\mathrm{{erg}} \\, \\mathrm{{cm}}^{{3}} \\mathrm{{s}}^{{-1}}]$'
        
        eltlines, logTK, lognHcm3 = cu.findemtables(ol.elements_ion[line_SB],
                                                    z)
        table_T_nH_SB = np.copy(eltlines[:, :, ol.line_nos_ion[line_SB]])
        
    elif table == 'dust':
        raise ValueError("Dust tables are not available in Serena's set")
        title = 'dust depletion fraction of {elt} at $z = {z:.2f}$'
        clabel = '$\\log_{{10}}$ fraction of {elt} in dust'
        
    elif table == 'ionbal':
        title = 'fraction {line_PS20} / {elt} at $z = {z:.2f}$'
        clabel = '$\\log_{{10}} \\; \\mathrm{{m}}(\\mathrm{{{ion}}}) \\, /' + \
            ' \\, \\mathrm{{m}}(\\mathrm{{{elt}}})$'
        
        if line_SB in ['h1ssh', 'hmolssh']:
            table_T_nH_SB, logTK, lognHcm3 = gethssh_R13(line_SB, z)
            table_T_nH_SB = np.log10(table_T_nH_SB)
        else:
            balance, logTK, lognHcm3 = cu.findiontables(line_SB, z)
            table_T_nH_SB = np.log10(balance.T)
    else:
        raise ValueError('{} is not a valid table option'.format(table))
    
    Tgrid = np.array([[x] * len(lognHcm3) for x in logTK]).flatten()
    ngrid = np.array([lognHcm3] * len(logTK)).flatten()
    dct = {'logT': Tgrid, 'lognH': ngrid, 
           'logZ': np.ones(len(Tgrid), dtype=np.float32) *\
               np.log10(ol.Zsun_ea)}
    zeroval = max(zeroval_SB, zeroval_PS20)
    
    if table == 'emission':        
        tab = m3.linetable_PS20(line_PS20, z, emission=True)
        table_T_nH_PS20 = tab.find_logemission(dct)
        table_T_nH_PS20 = table_T_nH_PS20.reshape((len(logTK), 
                                                   len(lognHcm3)))

        tozero = table_T_nH_PS20 <= zeroval_PS20
        table_T_nH_PS20 -= 2.* lognHcm3[np.newaxis, :]
        table_T_nH_PS20[tozero] = zeroval_PS20
        
        assumed_abunds_SB = ol.solar_abunds_sb[tab.element.lower()]
        dct_Z = {'logZ': np.array([dct['logZ'][0]])}
        assumed_abunds_PS20 = tab.find_assumedabundance(dct_Z)[0]
        
        table_T_nH_PS20_res = table_T_nH_PS20 +\
            np.log10(assumed_abunds_SB / assumed_abunds_PS20)
        
        table_T_nH_PS20_res[tozero] = zeroval
        
        info_SB = '$\\mathrm{{Z}} =$ {Z:.3e}, ' +\
                  '$\\mathrm{{n}}(\\mathrm{{{elt}}}) \\, /' + \
                  ' \\, \\mathrm{{n}}(\\mathrm{{H}}) =$ {abund:.3e}'
        info_PS20 = info_SB
        info_PS20_res = info_SB
        
        info_SB = info_SB.format(Z=10**dct_Z['logZ'][0],
                                 elt=element_to_abbr[ol.elements_ion[line_SB]],
                                 abund=ol.solar_abunds_sb[ol.elements_ion[line_SB]])
        info_PS20 = info_PS20.format(Z=10**dct_Z['logZ'][0],
                                 elt=tab.elementshort,
                                 abund=assumed_abunds_PS20)
        info_PS20_res = info_PS20_res.format(Z=10**dct_Z['logZ'][0],
                                 elt=tab.elementshort,
                                 abund=assumed_abunds_SB)
        
    elif table == 'ionbal':
        tab = m3.linetable_PS20(line_PS20, z, emission=False)
        table_T_nH_PS20 = np.log10(tab.find_ionbal(dct))    
        table_T_nH_PS20 = table_T_nH_PS20.reshape((len(logTK), 
                                                   len(lognHcm3)))
        tozero = table_T_nH_PS20 <= zeroval_PS20
        
        gasphase = 1. - tab.find_depletion(dct).reshape((len(logTK), 
                                                         len(lognHcm3)))
        table_T_nH_PS20_res = table_T_nH_PS20 + np.log10(gasphase)
        table_T_nH_PS20_res[tozero] = zeroval
        
        info_SB = ''
        info_PS20 = ''
        info_PS20_res = 'incl. dust depletion'
    
    if line_SB in nicenames_lines:
        linen = nicenames_lines[line_SB]
    else:
        linen = line_SB
    kws = {'ion': line_PS20, 'line_SB': linen, 'line_PS20': line_PS20, 
           'elt': tab.elementshort, 'z': z}
    title = title.format(**kws)
    clabel = clabel.format(**kws)
    
    deltaT = np.average(np.diff(logTK))
    deltanH = np.average(np.diff(lognHcm3))
    extent = (lognHcm3[0] - 0.5 * deltanH, lognHcm3[-1] + 0.5 * deltanH,
              logTK[0] - 0.5 * deltaT, logTK[-1] + 0.5 * deltaT)

    cmap_img = cm.get_cmap('viridis')
    cmap_img.set_under('white')
        
    vmin = np.min(table_T_nH_SB[table_T_nH_SB > zeroval])
    vmin = max(vmin, np.min(table_T_nH_PS20[table_T_nH_PS20 > zeroval]))
    vmin = max(vmin, np.min(table_T_nH_PS20_res[table_T_nH_PS20_res > zeroval]))
    
    vmax = np.max(table_T_nH_SB)
    vmax = max(vmax, np.max(table_T_nH_PS20))
    vmax = max(vmax, np.max(table_T_nH_PS20_res))
    #vmax = 0.
    
    vmax_cie = np.max(table_T_nH_PS20[:, -1])
    
    clevels = list(np.linspace(vmax_cie - 3., vmax_cie - 0.2, 5))[:-1] +\
              list(np.linspace(vmax_cie - 0.2, vmax, 3)[:-1]) 
    if vmax_cie - 3. > vmin + 2.:
        clevels = list(np.linspace(vmin, vmax_cie - 3., 4))[1:-1] + clevels
        
    cmap_contours = cm.get_cmap('plasma_r') 
    colors_contours = cmap_contours(np.linspace(0., 1., len(clevels)))
    
    fig = plt.figure(figsize=(11., 5.))
    grid = gsp.GridSpec(ncols=4, nrows=2,
                        hspace=0.3, wspace=0.3,
                        width_ratios=[4., 4., 4., 1.])
    
    cax = fig.add_subplot(grid[0, 3]) 
    lax = fig.add_subplot(grid[1, 2:])
    compax = fig.add_subplot(grid[1, 0])
    
    _cieax = fig.add_subplot(grid[1, 1])
    _l, _b, _w, _h = (_cieax.get_position()).bounds
    cieax = fig.add_axes([_l, _b + 0.3 * _h, _w, 0.7 * _h])
    cierelax = fig.add_axes([_l, _b, _w, 0.3 * _h])
    _cieax.axis('off')
    
    
    lax.axis('off')
    
    table_T_nH_SB[table_T_nH_SB <= zeroval] = -np.inf
    table_T_nH_PS20[table_T_nH_PS20 <= zeroval] = -np.inf
    table_T_nH_PS20_res[table_T_nH_PS20_res <= zeroval] = -np.inf
    
    linewidth = 2
    patheff = [mppe.Stroke(linewidth=linewidth + 0.5, foreground="black"),\
               mppe.Stroke(linewidth=linewidth + 0.5, foreground="white"),\
               mppe.Normal()]
    
    cierelmin = 0.
    cierelmax = 0.
    for ind, (_table, linestyle, label, line, info, title) \
        in enumerate(zip([table_T_nH_SB, table_T_nH_PS20, table_T_nH_PS20_res],
                         ['solid', 'dashed', 'dotted'],
                         ['(a)', '(b)', '(c)'],
                         [line_SB, line_PS20, line_PS20],
                         [info_SB, info_PS20, info_PS20_res],
                         [title_SB, title_PS20, title_PS20])):
        ax = fig.add_subplot(grid[0, ind])
        img = ax.imshow(_table, interpolation='nearest', origin='lower', 
                        extent=extent, cmap=cmap_img, vmin=vmin, vmax=vmax)
        cs = ax.contour(lognHcm3, logTK, _table, levels=clevels,
                         colors=colors_contours, origin='lower',
                         linestyles=linestyle, linewidths=linewidth)
        cs2 = compax.contour(lognHcm3, logTK, _table, levels=clevels,
                             colors=colors_contours, origin='lower', 
                             linestyles=linestyle, linewidths=linewidth,
                             path_effects=patheff)
        plt.setp(cs2.collections, path_effects=patheff)
        
        if ind == 0:
            cie_yvals_base = np.copy(_table[:, -1])
        
        cieax.plot(logTK, _table[:, -1], linestyle=linestyle,
                   linewidth=linewidth, color='black', alpha=0.3)
        reldiff = _table[:, -1] - cie_yvals_base
        cierelax.plot(logTK, reldiff,
                      linestyle=linestyle, color='black', alpha=0.3)
        
        # for difference axis limits
        regionofinterest = _table[:, -1] >= np.max(_table[:, -1]) - 2.5
        regionofinterest |= cie_yvals_base >= np.max(cie_yvals_base) - 2.5  
        cierelmin = min(cierelmin, np.min(reldiff[regionofinterest]))
        cierelmax = max(cierelmax, np.max(reldiff[regionofinterest]))
        
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        pu.setticks(ax, fontsize=fontsize)
        
        ax.text(0.5, 1.01, label + ' ' + title, fontsize=fontsize - 1,
                transform=ax.transAxes, horizontalalignment='center',
                verticalalignment='bottom')
        
        if line in nicenames_lines:
            linen = nicenames_lines[line]
        else:
            linen = line
        if ind == 0:
            ltext = '{label}: {line}\n   '.format(label=label, line=linen) +\
                     info
        else:
            ltext = ltext + '\n{label}: {line}\n   '.format(label=label,
                                                            line=linen) +\
                     info
        
    pu.setticks(compax, fontsize=fontsize)
    compax.set_xlabel(xlabel, fontsize=fontsize)
    compax.set_ylabel(ylabel, fontsize=fontsize)
    compax.grid(True)
    
    pu.setticks(cieax, fontsize=fontsize, labelleft=False, labelright=True,
                labelbottom=False)
    #cieax.set_xlabel(ylabel, fontsize=fontsize)
    cieax.set_ylabel(clabel, fontsize=fontsize)
    cieax.yaxis.set_label_position('right')
    cieax.grid(True)
    cieax.set_ylim(vmax_cie - 6., vmax_cie + 0.3)
    txt = 'CIE: $\\log_{{10}} \\, \\mathrm{{n}}_{{\\mathrm{{H}}}} \\, /' + \
        '\\, \\mathrm{{cm}}^{{-3}} =$ {lognH:.1f}'.format(lognH=lognHcm3[-1])
    cieax.text(0.05, 0.05, txt, fontsize=fontsize - 1,
               horizontalalignment='left',
               verticalalignment='bottom', transform=cieax.transAxes,
               bbox={'facecolor': 'white', 'alpha': 0.3})   
    pu.setticks(cieax, fontsize=fontsize, labelleft=False, labelright=True)
    
    cierelax.set_xlabel(ylabel, fontsize=fontsize)
    cierelax.grid(True)
    pu.setticks(cierelax, fontsize=fontsize, labelleft=False, labelright=True)
    #cierelax.text(0.0, 1.0, cierellabel, fontsize=fontsize,
    #              transform=cierelax.transAxes,
    #              horizontalalignment='left', verticalalignment='top',
    #              bbox={'facecolor': 'white', 'alpha': 0.3})
    cierelax.set_ylabel(cierellabel, fontsize=fontsize)
    
    #cierelax.yaxis.set_label_position('right')
    #print(cierelmin, cierelmax)
    ylim = (1.1 * cierelmin, 1.1 * cierelmax)
    y0 = min(ylim[0], -0.05)
    y0 = max(y0, -1.5)
    y1 = max(ylim[1], 0.05)
    y1 = min(y1, 1.5)
    y0_old = y0
    y0 = min(y0, -0.15 * y1)
    y1 = max(y1, -0.15 * y0_old)
    cierelax.set_ylim(y0, y1)
        
    lax.text(0.15, 0.95, ltext, fontsize=fontsize, horizontalalignment='left',
               verticalalignment='top', transform=lax.transAxes)
    
    # color bar 
    cbar = pu.add_colorbar(cax, img=img, cmap=cmap_img, vmin=vmin,
                           clabel=clabel, fontsize=fontsize, 
                           orientation='vertical', extend='min')
    cax.set_aspect(8.)
    cax.tick_params(labelsize=fontsize - 1.)
    cbar.add_lines(cs)
        
    outname = mdir + 'comp_SB_PS20_{table}_table_{line_SB}_{line_PS20}_z{z:.2f}.pdf'
    outname = outname.format(line_SB=line_SB, line_PS20=line_PS20,
                             z=z, table=table)
    plt.savefig(outname, format='pdf', bbox_inches='tight')

    
# test interpolation of the tables graphically

def test_interp(line, table='emission'):
    
    fontsize = 12    
    cset =  tc.tol_cset('bright') 
    
    title = '{table} interpolation test\n' + \
             'interpolating in one dimension,' + \
             ' random grid points for the others'
    
    axes = ['T', 'n', 'Z', 'z']
    xlabels = {'T': '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$',
               'n': '$\\log_{10} \\, \\mathrm{n}_{\\mathrm{H}}' + \
                    ' \\; [\\mathrm{cm}^{-3}]$',
               'Z': '$\\log_{10} \\, \\mathrm{Z} \\; [\\mathrm{Z}_{\\odot}]$',
               'z': 'redshift'}
    
    # not used for the interpolation, just to get filenames etc.
    dummytab = m3.linetable_PS20(line, 0.0)
    filen    = dummytab.ionbalfile
    filen_em = dummytab.emtabfile
    
    with h5py.File(filen, 'r') as f:
        gridvalues = {'T': f['TableBins/TemperatureBins'][:], 
                      'n': f['TableBins/DensityBins'][:], 
                      'Z': f['TableBins/MetallicityBins'][:],
                      'z': f['TableBins/RedshiftBins'][:],
                      }
        if table == 'emission':
            with h5py.File(filen_em, 'r') as fe:
                lineid = fe['IdentifierLines'][:]
            lineid = np.array([_line.decode() for _line in lineid])
            match = [line == _line for _line in lineid]
            lineind = np.where(match)[0][0]
        elif table == 'ionbal':
            lineind = dummytab.ionstage - 1
        elif table == 'dust':
            lineind = dummytab.eltind
        
    # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Line/ion stage
    tabledims = {'z': 0,
                 'T': 1,
                 'Z': 2,
                 'n': 3}
    
    if table == 'emission':
        ylabel = '$\\log_{{10}} \\, \\mathrm{{\\Lambda}}' + \
        '\\, \\mathrm{{V}}^{{-1}}  \\;' +\
        ' [\\mathrm{{erg}} \\, \\mathrm{{cm}}^{{3}} \\mathrm{{s}}^{{-1}}]$'
    elif table == 'dust':
        ylabel = '$\\log_{{10}}$ fraction of {elt} in dust'
    elif table == 'ionbal':
        ylabel = '$\\log_{{10}} \\; \\mathrm{{m}}(\\mathrm{{{ion}}}) \\, /' + \
            ' \\, \\mathrm{{m}}(\\mathrm{{{elt}}})$'
    elif table == 'assumed_abundance':
                ylabel = '$\\log_{{10}} \\, \\mathrm{{n}}_{{\\mathrm{{{elt}}}}}' +\
                 '\\, / \\, \\mathrm{{n}}_{{\\mathrm{{H}}}}$'
    
    if table == 'assumed_abundance':
        axes = ['Z']
    elif table in ['emission', 'dust', 'ionbal']:
        pass
    else:
        raise ValueError('{} is not a valid table option'.format(table))
    
    if len(axes) == 1:
        fig, ax = plt.subplots(ncols=1, nrows=1)
        axs = [ax]
    elif len(axes) == 4:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(7.5, 7.),
                                gridspec_kw={'hspace': 0.3, 'wspace': 0.4})
        axs = [axs[i //2, i %2] for i in range(4)]
    
    np.random.seed(seed=0)
    numsample = 2 #len(cset)
    samplesize = 20
    
    for ind, (ax, tabax) in enumerate(zip(axs, axes)):
        grid_x = gridvalues[tabax]
        
        if len(axes) == 1: #abundances
            test_range = [grid_x[0] - 5., grid_x[-1] + 5.]
            samplex = np.random.uniform(low=test_range[0], high=test_range[1],
                                        size=samplesize)
            if table == 'assumed_abundance':
                ylabel = '$\\log_{{10}} \\, \\mathrm{{n}}_{{\\mathrm{{{elt}}}}}' +\
                 '\\, / \\, \\mathrm{{n}}_{{\\mathrm{{H}}}}$'
                # logZ, element
                tablepath = 'TotalAbundances'
                
                with h5py.File(filen, 'r') as f:
                    grid_y = f[tablepath][:, dummytab.eltind]
                dct_Z = {'logZ': samplex + np.log10(dummytab.solarZ)}
                sampley = np.log10(dummytab.find_assumedabundance(dct_Z))
            
            title = title.format(table=table)
            ylabel = ylabel.format(elt=dummytab.elementshort,
                                    ion='{}{}'.format(dummytab.elementshort,
                                                      dummytab.ionstage)) 
            
            pu.setticks(ax, fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.set_xlabel(xlabels[tabax], fontsize=fontsize)
            ax.plot(grid_x, grid_y, color=cset[0], linewidth=2)
            ax.scatter(grid_x, grid_y, color=cset[0], marker='o', s=30,
                        label='table', alpha=0.3)
            ax.scatter(samplex, sampley,
                       color=cset[0], alpha=0.7,
                       marker='x', s=10, label='interp')
            ax.legend(fontsize=fontsize)
            
        else:
            otheraxes = list(np.copy(axes))
            otheraxes.remove(tabax)
            gridinds = {d: np.random.randint(0, high=len(gridvalues[d]),
                                             size=numsample)
                        for d in otheraxes}
            #label = ', '.join(['$\\mathrm{{{{{d}}}}} = {{{d}:.1f}}$'.format(d=d) \
            #                   for d in otheraxes])
            label = ', '.join(['{d}={{{d}:.1f}}'.format(d=d) \
                               for d in otheraxes])
            seltuples = np.array([[slice(None, None, None)] * 5] * numsample)
            for d in otheraxes:
                seltuples[:, tabledims[d]] = gridinds[d]
            
            labelfill = {}
            if tabax == 'z':
                _samplesize = 1
                test_range = [grid_x[0], grid_x[-1]]
            else:
                _samplesize = samplesize
                test_range = [grid_x[0] - 5., grid_x[-1] + 5.]
            for i in range(numsample):
                dct_T_Z_nH = {}
                if 'T' in otheraxes:
                    _a = [gridvalues['T'][gridinds['T'][i]]] * _samplesize
                    dct = {'logT': np.array(_a)}
                    dct_T_Z_nH.update(dct)
                    labelfill['T'] = _a[0]
                else:
                    samplex = np.random.uniform(low=test_range[0], 
                                                high=test_range[1],
                                                size=samplesize)
                    dct = {'logT': samplex}
                    dct_T_Z_nH.update(dct)
                if 'Z' in otheraxes:
                    _a = [gridvalues['Z'][gridinds['Z'][i]]] * _samplesize
                    dct = {'logZ': np.array(_a) + np.log10(dummytab.solarZ)}
                    dct_T_Z_nH.update(dct)
                    labelfill['Z'] = _a[0]
                else:
                    samplex = np.random.uniform(low=test_range[0], 
                                                high=test_range[1],
                                                size=samplesize)
                    dct = {'logZ': samplex + np.log10(dummytab.solarZ)}
                    dct_T_Z_nH.update(dct)
                if 'n' in otheraxes:
                    _a = [gridvalues['n'][gridinds['n'][i]]] * _samplesize
                    dct = {'lognH': np.array(_a)}
                    dct_T_Z_nH.update(dct)
                    labelfill['n'] = _a[0]
                else:
                    samplex = np.random.uniform(low=test_range[0], 
                                                high=test_range[1],
                                                size=samplesize)
                    dct = {'lognH': samplex}
                    dct_T_Z_nH.update(dct)
                if 'z' in otheraxes:
                    z = gridvalues['z'][gridinds['z'][i]]
                    labelfill['z'] = z
                else:
                    samplex = np.random.uniform(low=test_range[0], 
                                                high=test_range[1],
                                                size=samplesize)
                    z = samplex
                
                _label = label.format(**labelfill)
                                
                emission = False
                if table == 'emission':
                    tablepath = 'Tdep/EmissivitiesVol' 
                    emission = True
                    if tabax == 'z':
                        _tables = [m3.linetable_PS20(line, z, emission=emission)\
                                   for z in z]
                        sampley = [_table.find_logemission(dct_T_Z_nH) \
                                   for _table in _tables]
                    else:
                        _table = m3.linetable_PS20(line, z, emission=emission)
                        sampley = _table.find_logemission(dct_T_Z_nH)
                    
                elif table == 'dust':
                    tablepath = 'Tdep/Depletion' 
                    _table = m3.linetable_PS20(line, z, emission=emission)
                    if tabax == 'z':
                        _tables = [m3.linetable_PS20(line, z, emission=emission)\
                                   for z in z]
                        sampley = [_table.find_depletion(dct_T_Z_nH) \
                                   for _table in _tables]
                        sampley = np.log10(sampley)
                        # table zero value
                        sampley[sampley == -np.inf] = -50.
                    else:
                        _table = m3.linetable_PS20(line, z, emission=emission)
                        sampley = np.log10(_table.find_depletion(dct_T_Z_nH))
                        # table zero value
                        sampley[sampley == -np.inf] = -50.
                    
                elif table == 'ionbal':
                    tablepath = 'Tdep/IonFractions/{eltnum:02d}{eltname}'
                    tablepath = tablepath.format(eltnum=dummytab.eltind, 
                                                 eltname=dummytab.element.lower())
                    if tabax == 'z':
                        _tables = [m3.linetable_PS20(line, z, emission=emission)\
                                   for z in z]
                        sampley = [_table.find_ionbal(dct_T_Z_nH, log=True) \
                                   for _table in _tables]
                    else:
                        _table = m3.linetable_PS20(line, z, emission=emission)
                        sampley = _table.find_ionbal(dct_T_Z_nH, log=True)        
                sampley = np.array(sampley)    
                
                tableslice = seltuples[i]
                tableslice[4] = lineind
                tableslice = tuple(tableslice)
                #print(tabax)
                #for d in otheraxes:
                #    msg = '{d}: {val:.2f}'
                #    val = gridvalues[d][tableslice[tabledims[d]]]
                #    print(msg.format(d=d, val=val))
                #print(filen_em)
                #print(tableslice)
                with h5py.File(filen_em if emission else filen, 'r') as f:
                    #print(tablepath)
                    #print(tablepath in f)
                    #print(f[tablepath].shape)
                    grid_y = f[tablepath][tableslice]
                                    
                ax.plot(grid_x, grid_y, color=cset[i], linewidth=1.)
                ax.scatter(grid_x, grid_y, color=cset[i], marker='o', s=30,
                            label=None, alpha=0.2)
                ax.scatter(samplex, sampley,
                           color=cset[i], alpha=0.8,
                           marker='x', s=10, label=_label)
            
            title = title.format(table=table)
            _ylabel = ylabel.format(elt=dummytab.elementshort,
                                    ion='{}{}'.format(dummytab.elementshort,
                                                      dummytab.ionstage)) 
            pu.setticks(ax, fontsize)
            ax.set_ylabel(_ylabel, fontsize=fontsize)
            ax.set_xlabel(xlabels[tabax], fontsize=fontsize)
            handles1, labels = ax.get_legend_handles_labels()
            handles2 = [mlines.Line2D([], [], label='interp', color='black',
                                      alpha=0.8, marker='x', linewidth=0.0),
                        mlines.Line2D([], [], label='table', color='black',
                                      alpha=0.2, marker='o', linewidth=1.5),
                        ]
            ax.legend(handles=handles2 + handles1, fontsize=fontsize - 3.)
            #ax.legend(fontsize=fontsize)       
         
    fig.suptitle(title, fontsize=fontsize)    
    
    
    # 0: Redshift, 1: Temperature, 2: Metallicity, 3: Density, 4: Line/ion stage
    
    outname = mdir + 'interptest_PS20_{table}_table_{line}.pdf'
    outname = outname.format(line=line,table=table)
    plt.savefig(outname, format='pdf', bbox_inches='tight')

# compare maps with the two table sets
# emission maps for a few ions, with and without abundance adjustments
# absorption maps
# no-dust maps with both table sets
# with and without dust depletion

def compare_maps(args_map1, args_map2, kwargs_map1, kwargs_map2,\
                 imgname=None):
    pass



# test basic table retrieval and sensitbility
def plot_tablesets(zs):
    for z in zs:
        for line in lines_SP20:
            plottables_PS20(line, z, table='emission')
        for line in lines_SB:
            plottables_SB(line, z, table='emission')
        for ion in ions:
            plottables_PS20(ion, z, table='ionbal')
            plottables_PS20(ion, z, table='dust')
            plottables_SB(ion, z, table='ionbal')

def compare_tablesets(z):
    for ion in ions:
        compare_tables(ion, ion, z, table='ionbal')
    for line in linematch_SP20:
        if linematch_SP20[line] is None:
            continue
        compare_tables(line, linematch_SP20[line], z, table='emission')

    
if __name__ == '__main__':
    zs_test = [0.0, 0.1, 1., 3.]  
    plot_tablesets(zs_test)
    
    z = 0.1
    compare_tablesets(z)
            
