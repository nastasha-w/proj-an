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
    cierellabel = '$\\Delta$'
    
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
        cierelax.plot(logTK, _table[:, -1] - cie_yvals_base,
                      linestyle=linestyle, color='black', alpha=0.3)
        
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
    cierelax.set_ylabel(cierellabel, fontsize=fontsize)
    cierelax.yaxis.set_label_position('right')
    ylim = cierelax.get_ylim()
    y0 = min(ylim[0], -0.05)
    y0 = max(y0, -1.5)
    y1 = max(ylim[1], 0.05)
    y1 = min(y1, 1.5)
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
    plot_tablesets(zs_test)
    
    z = 0.1
    compare_tablesets(z)
            
