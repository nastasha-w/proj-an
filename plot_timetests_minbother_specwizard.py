#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:22:47 2020

@author: Nastasha
"""

# no further post-processing here; just the raw stored spectra
import h5py 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines

import eagle_constants_and_units as c
import ion_line_data as ild


# a few useful defaults
imgdir = '/cosma5/data/dp004/dc-wije1/specwiz/imgs/timetests/'

logfiles = ['/cosma5/data/dp004/dc-wije1/specwiz/timetest_shacc_r32/stdout.L0100N1504-shacc-32-2009555',\
            '/cosma5/data/dp004/dc-wije1/specwiz/timetest_hiacc_r32/stdout.L0100N1504-hiacc-32-2009548',\
            '/cosma5/data/dp004/dc-wije1/specwiz/timetest_miacc_r32/stdout.L0100N1504-miacc-32-2009549',\
            '/cosma5/data/dp004/dc-wije1/specwiz/timetest_loacc_r32/stdout.L0100N1504-loacc-32-2009550',\
            ]
specfiles = ['/cosma5/data/dp004/dc-wije1/specwiz/timetest_shacc_r32/spec.snap_027_z000p101.0.hdf5',\
             '/cosma5/data/dp004/dc-wije1/specwiz/timetest_hiacc_r32/spec.snap_027_z000p101.0.hdf5',\
             '/cosma5/data/dp004/dc-wije1/specwiz/timetest_miacc_r32/spec.snap_027_z000p101.0.hdf5',\
             '/cosma5/data/dp004/dc-wije1/specwiz/timetest_loacc_r32/spec.snap_027_z000p101.0.hdf5',\
             ]
labels = ['shacc', 'hiacc', 'miacc', 'loacc']
# not recorded in the hdf5 files
minbother_vals = {'shacc': {'minbother_red': 1e-14,\
                            'minbother_blue': 1e-13},\
                  'hiacc': {'minbother_red': 1e-14,\
                            'minbother_blue': 1e-13},\
                  'miacc': {'minbother_red': 1e-4,\
                            'minbother_blue': 1e-3},\
                  'loacc': {'minbother_red': 1e-3,\
                            'minbother_blue': 1e-2},\
                  }



def plot_timing(*args, **kwargs):
    '''
    plot the timing of various parts of specwizard based on the log files
    specwizard must have been run with verbose = T for this to work, and 
    compiled with MPI (even if it is not run in that mode)
    the timings are compared at the same spectrum number, which is only 
    meaningful if those spectra are the same. Since in MPI mode, data is only
    logged for process 0, in this mode, the number of MPI processes used must 
    also match
    
    args:
    -----
    stdout files to get timing from (strings)
    
    kwargs: 
    -------
    'function': the name of the timed program to match (string)
                options: 
                'makespectra' (default; for each short spectrum)
                'projectdata' (for each short spectrum)
                'insertspectra' (long spectra only; for each short spectrum)
                'long spectrum' (long spectra only; total time for spectrum)
    'labels':   labels for the different log files (list-like of strings)
                default: log file names, with directory path removed
    'savename': if set and not None, save the plot to this file (string)
    'savedir':  (default: '/cosma5/data/dp004/dc-wije1/specwiz/imgs/timetests/')
                directory to save the file in (string)
    
    '''
    
    filens = args
    
    if 'labels' not in kwargs:
        labels = [filen.split('/')[-1] for filen in filens]
    else:
        labels = kwargs['labels']
    if 'function' in kwargs:
        mfunc = kwargs['function']
    else:
        mfunc = 'makespectra'
        
    times = []
    for filen in filens:
        smatch = mfunc + ' time'
        with open(filen, 'r') as fi:
            # looking for lines like
            # projectdata time =   0.2600E+00 s, makespectra time =   0.4090E+00 s
            lines = fi.readlines()
            lines = [line.decode() for line in lines]
            matchlines = [_ln.split(',') if smatch in _ln else None \
                          for _ln in lines]
            linesel = np.where([_ln is not None for _ln in matchlines])[0]
            matchlines = [matchlines[ind] for ind in linesel]
            timests = [_ln[np.where([smatch in _pt for _pt in _ln])[0][0]]\
                       for _ln in matchlines]
            timests = [st.split('=')[-1].strip() for st in timests]
            _times = [float(st.split(' ')[0]) for st in timests]
        
        times.append(_times)
    
    times = np.array(times)
    # stored times are cumulative over the sightlines
    times = np.append(times[:, 0][:, np.newaxis], np.diff(times, axis=1), axis=1)
    
    fontsize = 12
    fig = plt.figure(figsize=(5.5, 5.))
    ax = fig.add_subplot(1, 1, 1)
    
    for i in range(times.shape[0]):
        ydata = times[i]
        xdata = np.arange(len(ydata))
        ax.scatter(xdata, ydata, label=labels[i], marker='o', alpha=0.5)
    ax.set_xlabel('spectrum number', fontsize=fontsize)
    ax.set_ylabel('{fname} time [s]'.format(fname=mfunc), fontsize=fontsize)
    ax.tick_params(which='both', direction='in', labelsize=fontsize - 1,\
                   labelleft=True, labelbottom=True,\
                   top=True, left=True, bottom=True, right=True)
    ax.set_yscale('log')
    ax.legend(fontsize=fontsize, ncol=2)
    
    if 'savename' in kwargs:
        foname = kwargs['savename']
        if foname is None:
            return
        if 'savedir' in kwargs:
            savedir = kwargs['savedir']
        else:
            savedir = '/cosma5/data/dp004/dc-wije1/specwiz/imgs/timetests/'
        if savedir[-1] != '/':
            savedir = savedir + '/'
        foname = savedir + foname
        
        plt.savefig(foname, bbox_inches='tight')
                

def comparedct(dct1, dct2):
    '''
    compares two dictionaries
    input:
    ------
    dct1, dct2: the two dictionaries
    
    returns:
    --------
    keysonlyin1 (set), keysonlyin2 (set), keysdifferent (set)
    '''
    keys1 = set(dct1.keys())
    keys2 = set(dct2.keys())
    keysonlyin1 = keys1 - keys2
    keysonlyin2 = keys2 - keys1
    keysboth = keys1 & keys2
    dct_eq = {key: np.all(dct1[key] == dct2[key]) for key in keysboth}
    keysdiff = {key if not dct_eq[key] else None for key in dct_eq}
    keysdiff -= {None}
    return keysonlyin1, keysonlyin2, keysdiff

def plotdiffs_spectra(file_test, file_check,\
                      label_test='test', label_check='check',\
                      maxionsperplot=10, plotdir=imgdir):
    '''
    plot the differences between spectra in two files. test is compared to the
    baseline file check. The labels are used in the plots, and to get the 
    minbother values.
    This assumes the spectra in the two files are along the same sightlines, 
    etc.
    
    '''
    # arrays and attributes to compare for run difference overview 
    gpaths_attrs = ['Constants',\
                    'Header',\
                    'Header/ModifyMetallicityParameters',\
                    'Parameters/ChemicalElements',\
                    'Parameters/SpecWizardRuntimeParameters',\
                    'Projection',\
                    ]
    gpaths_arrays = ['Projection/ncontr',\
                     'Projection/x_fraction_array',\
                     'Projection/y_fraction_array',\
                     'VHubble_KMpS',\
                     ]
    attrs_ignore = ['SpectrumFile', 'outputdir']
    plotpath = 'Spectrum{specnum}/{ion}/Flux'
    lognpath = 'Spectrum{specnum}/{ion}/LogTotalIonColumnDensity'
    
    with h5py.File(file_test, 'r') as ft,\
        h5py.File(file_check, 'r') as fc:
        
        attrs_diff = {}
        for path in gpaths_attrs:
            dct_test = {key: val for key, val in ft[path].attrs.items()} 
            dct_check = {key: val for key, val in fc[path].attrs.items()} 
            test_missing, check_missing, diffkeys = \
               comparedct(dct_test, dct_check)
            if len(test_missing) > 0 or len(check_missing) > 0:
                erm = 'Test file missing keys {t}\nCheck file missing keys {c}'
                erm.format(t=test_missing, c=check_missing)
                raise RuntimeError(erm)
            attrs_diff.update({key: (dct_test[key], dct_check[key]) \
                               for key in diffkeys})
            if path == 'Header':
                ions = np.array([ion.decode() for ion in dct_test['Ions']])
                fosc = dct_test['Transitions_Oscillator_Strength']
                lang = dct_test['Transitions_Rest_Wavelength']
                #boxsize = dct_test['BoxSize']
                redshift = dct_test['Redshift']
                
                ions, ioninds = np.unique(ions, return_index=True)
                fosc = {ions[i]: fosc[ioninds[i]] for i in range(len(ions))}
                lang = {ions[i]: lang[ioninds[i]] for i in range(len(ions))}

        for path in gpaths_arrays:
            ta = np.array(ft[path])
            tc = np.array(fc[path])
            if not np.all(ta ==tc):
                erm = 'The sightlines for the input files are different'
                raise ValueError(erm)
            if path == 'VHubble_KMpS':
                vvals_test = ta  
                vvals_check = tc
                
        outofspectra = False
        specnum = 0
        spectra_test = {}
        spectra_check = {}
        nvals_test = {}
        nvals_check = {}
        while not outofspectra:
            spectra_test[specnum] = {}
            spectra_check[specnum] = {}
            nvals_test[specnum] = {}
            nvals_check[specnum] = {}
            for ion in ions:
                try:
                    spectra_test[specnum][ion] = np.array(ft[plotpath.format(\
                                specnum=specnum, ion=ion)])
                    spectra_check[specnum][ion] = np.array(fc[plotpath.format(\
                                specnum=specnum, ion=ion)])
                    nvals_test[specnum][ion] = np.array(ft[lognpath.format(\
                                specnum=specnum, ion=ion)])[()]
                    nvals_check[specnum][ion] = np.array(fc[lognpath.format(\
                                specnum=specnum, ion=ion)])[()]
                except KeyError:
                    outofspectra = True
            if outofspectra:
                del spectra_test[specnum]
                del spectra_check[specnum]
                del nvals_test[specnum]
                del nvals_check[specnum]
            specnum += 1   
               
    minbother_red = minbother_vals[label_test]['minbother_red']\
                    if label_test in minbother_vals else 0.
    minbother_blue = minbother_vals[label_test]['minbother_blue']\
                    if label_test in minbother_vals else 0.
    lyalpha = 1215.6701
    
    title = 'Spectrum {specnum}, $z = {z:.2f}$, comparing\n' + \
            '{test}: {tlist}\n' +\
            '{check}: {clist}'
    dctfmt = '{key}={val}'
    attrs_note = set(attrs_diff.keys()) - set(attrs_ignore)
    dct_note = {key: attrs_diff[key] for key in attrs_note}
    dct_note.update({'minbother_red': (minbother_red,\
                            minbother_vals[label_check]['minbother_red']),\
                     'minbother_blue': (minbother_blue,\
                            minbother_vals[label_check]['minbother_blue']),\
                     })
    keys_note = sorted(list(dct_note.keys()))
    tlist = ', '.join([dctfmt.format(key=key, val=dct_note[key][0]) \
                       for key in keys_note])
    clist = ', '.join([dctfmt.format(key=key, val=dct_note[key][1]) \
                       for key in keys_note])
                      
    numions = len(ions) 
    numplots_persl = (numions  - 1) // maxionsperplot + 1
    ions = sorted(ions, key=lang.get) # sort by wavelength
    ions = ions[::-1]
    
    name = plotdir + 'Spectrum{specnum}_ionset{pi}_{tlabel}-vs-{clabel}.pdf'
    kwargs_test = {'color': 'C0', 'linestyle': 'dashed'}
    kwargs_check = {'color': 'C1', 'linestyle': 'solid'}
    fontsize = 10
    
    margin = 0.6
    panelheight = 0.8
    panelwidth = 2.5
    titleheight = 0.7
    wspace = 0.35
    
    t1 = 'flux difference: {test} - {check}'.format(test=label_test,\
                              check=label_check)
    t2 = '$\\log_{{10}}$ flux ratio: {test} / {check}'.format(\
                         test=label_test, check=label_check)
    leghandles = [mlines.Line2D([], [], label=label_test, **kwargs_test),\
                  mlines.Line2D([], [], label=label_check, **kwargs_check),\
                  ]
    leghandles_diff =\
                  [mlines.Line2D([], [], label='$\\Delta($minbother$)$',\
                                linestyle='dashed', color='black'),\
                  mlines.Line2D([], [], label='$\\Delta$',\
                                linestyle='solid', color='gray'),\
                  ]
    ylab0 = 'normalized flux'
    ylab1 = '$\\Delta$ flux'
    ylab2 = '$\\Delta \\, \\log_{{10}}$ flux'
    xlab = '$\\Delta \\, v \\; [\\mathrm{{km}} \\, \\mathrm{{s}}^{{-1}}]$'
    ptbase = '{ion}, ${wl:.2f} \\, \\mathrm{{\AA}}$' # \\lambda = 
    nlabelbase = '$\\mathrm{{N}} = {:.1f}$' # \\log_{{10}} \\mathrm{{N}} / \\mathrm{{cm}}^{{-2}}
    tickparams = {'which': 'both', 'direction': 'in',\
                  'labelsize': fontsize - 1,\
                  'left': True, 'right': True, 'top': True, 'bottom': True,\
                  'labelleft': True, 'labelbottom': False}
    bbox = {'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'none'}
    
    maxdiff_rec = 0.
    maxreldiff = 0.
    
    for specnum in spectra_test:
        for pi in range(numplots_persl):
            iimin = pi * maxionsperplot
            iimax = min((pi + 1) * maxionsperplot, numions)
            _ions = ions[iimin : iimax]
            _nions = len(_ions)
            
            fname = name.format(specnum=specnum, pi=pi,\
                               tlabel=label_test, clabel=label_check)
            ptitle = title.format(specnum=specnum, z=redshift,\
                                  test=label_test, tlist=tlist,\
                                  check=label_check, clist=clist)
            
            totalheight = margin * 2 + panelheight * len(_ions) + titleheight
            totalwidth = margin * 2 + panelwidth * 3 + wspace * 2 
            bottom = margin / totalheight
            top = 1. - bottom
            left = margin / totalwidth
            right = 1. - left
            hrs = [titleheight] + [panelheight] * _nions
            
            fig = plt.figure(figsize=(totalwidth, totalheight))
            grid = gsp.GridSpec(nrows=len(_ions) + 1, ncols=3,\
                                hspace=0.0, wspace=wspace, height_ratios=hrs,\
                                top=top, bottom=bottom, left=left, right=right)
            ## add column titles
            fig.suptitle(ptitle, fontsize=fontsize)
            taxs = [fig.add_subplot(grid[0, i]) for i in range(3)]
            axs0 = [fig.add_subplot(grid[i + 1, 0]) for i in range(_nions)]
            axs1 = [fig.add_subplot(grid[i + 1, 1]) for i in range(_nions)]
            axs2 = [fig.add_subplot(grid[i + 1, 2]) for i in range(_nions)]
            
            [tax.axis('off') for tax in taxs]
            #taxs[0].text(0.5, 0.8, 'spectra', fontsize=fontsize,\
            #   verticalalignment='top', horizontalalignment='center',\
            #   transform=taxs[0].transAxes)
            
            leg = taxs[0].legend(handles=leghandles, fontsize=fontsize, ncol=2,\
                loc='lower center', bbox_to_anchor=(0.5, 0.1), frameon=False,\
                ) #title='spectra', title_fontsize=fontsize
            leg.set_title(r'spectra', prop={'size': fontsize})            
            
            taxs[1].text(0.5, 0.3, t1, fontsize=fontsize,\
                verticalalignment='bottom', horizontalalignment='center',\
                transform=taxs[1].transAxes)
            
            leg2 = taxs[2].legend(handles=leghandles_diff,\
                fontsize=fontsize, ncol=2,\
                loc='lower center', bbox_to_anchor=(0.5, 0.1), frameon=False,\
                ) #title='spectra', title_fontsize=fontsize
            leg2.set_title(t2, prop={'size': fontsize})
            #taxs[2].text(0.5, 0.3, t2, fontsize=fontsize,\
            #    verticalalignment='bottom', horizontalalignment='center',\
            #    transform=taxs[2].transAxes)
            
            for ii in range(_nions):
                ion = _ions[ii]
                ax0 = axs0[ii]
                ax1 = axs1[ii]
                ax2 = axs2[ii]
                
                tps = tickparams.copy()
                tps.update({'labelbottom': ii == _nions - 1})
                ax0.tick_params(**tps)
                ax1.tick_params(**tps)
                ax2.tick_params(**tps)
                
                if ii == _nions // 2: # somewhere in the middle
                    ax0.set_ylabel(ylab0, fontsize=fontsize, labelpad=6.)
                    ax1.set_ylabel(ylab1, fontsize=fontsize, labelpad=6.)
                    ax2.set_ylabel(ylab2, fontsize=fontsize, labelpad=6.)
                if ii == _nions - 1:
                    ax0.set_xlabel(xlab, fontsize=fontsize)
                    ax1.set_xlabel(xlab, fontsize=fontsize)
                    ax2.set_xlabel(xlab, fontsize=fontsize)
                    
                ax0.plot(vvals_check, spectra_check[specnum][ion],\
                         **kwargs_check)
                ax0.plot(vvals_test, spectra_test[specnum][ion],\
                         **kwargs_test)
                pt = ptbase.format(ion=ild.getnicename(ion), wl=lang[ion])              
                ax0.text(0.98, 0.05, pt, fontsize=fontsize - 1,\
                         verticalalignment='bottom', \
                         horizontalalignment='right', transform=ax0.transAxes,\
                         bbox=bbox)
                nt = nlabelbase.format(nvals_check[specnum][ion])
                if nvals_check[specnum][ion] != nvals_test[specnum][ion]:
                    print('Warning: different column densities for' +\
                          ' specnum {sn}, {ion}'.format(sn=specnum, ion=ion) +\
                          ': {t} and {c}'.format(t=nvals_test[specnum][ion],\
                             c=nvals_check[specnum][ion]))
                ax1.text(0.98, 0.05, nt, fontsize=fontsize - 1,\
                         verticalalignment='bottom', \
                         horizontalalignment='right', transform=ax1.transAxes,\
                         bbox=bbox)
                
                maxdiff = minbother_red if lang[ion] > 1.001 * lyalpha else\
                          minbother_blue
                          
                ax1.plot(vvals_check, spectra_test[specnum][ion] -\
                         spectra_check[specnum][ion], color='gray')
                ax1.axhline(maxdiff, linestyle='dashed', color='black')
                _md = np.max(np.abs(spectra_test[specnum][ion] -\
                         spectra_check[specnum][ion]))
                maxdiff_rec = max(maxdiff_rec, _md)
                
                lograt = np.log10(spectra_test[specnum][ion] \
                                  / spectra_test[specnum][ion])
                # zero difference if both fluxes are zero
                lograt[spectra_test[specnum][ion] == \
                       spectra_test[specnum][ion]] = 0. 
                ax2.plot(vvals_check, lograt, color='gray')
                ax2.axhline(np.log10(1. + maxdiff), linestyle='dashed',\
                            color='black')
                maxreldiff = max(maxreldiff, np.max(np.abs(lograt)))
                
            plt.savefig(fname)
            plt.close() # large number of plots -> limit memory use
    print('Max. difference: {md}, Max. log10 rel. diff: {mlrd}'.format(
            md=maxdiff_rec, mlrd=maxreldiff))                   
                
def plotdiffs_EW(file_test, file_check,\
                 label_test='test', label_check='check',\
                 plotdir=imgdir, logdiff=True):       
    '''
    plot the total EW differences in two files, as a function of column 
    density, for the different ions. test is compared to the
    baseline file check. The labels are used in the plots, and to get the 
    minbother values.
    This assumes the spectra in the two files are along the same sightlines, 
    etc.
    '''
    # arrays and attributes to compare for run difference overview 
    gpaths_attrs = ['Constants',\
                    'Header',\
                    'Header/ModifyMetallicityParameters',\
                    'Parameters/ChemicalElements',\
                    'Parameters/SpecWizardRuntimeParameters',\
                    'Projection',\
                    ]
    gpaths_arrays = ['Projection/ncontr',\
                     'Projection/x_fraction_array',\
                     'Projection/y_fraction_array',\
                     'VHubble_KMpS',\
                     ]
    attrs_ignore = ['SpectrumFile', 'outputdir']
    plotpath = 'Spectrum{specnum}/{ion}/Flux'
    lognpath = 'Spectrum{specnum}/{ion}/LogTotalIonColumnDensity'
    
    with h5py.File(file_test, 'r') as ft,\
        h5py.File(file_check, 'r') as fc:
        
        attrs_diff = {}
        for path in gpaths_attrs:
            dct_test = {key: val for key, val in ft[path].attrs.items()} 
            dct_check = {key: val for key, val in fc[path].attrs.items()} 
            test_missing, check_missing, diffkeys = \
               comparedct(dct_test, dct_check)
            if len(test_missing) > 0 or len(check_missing) > 0:
                erm = 'Test file missing keys {t}\nCheck file missing keys {c}'
                erm.format(t=test_missing, c=check_missing)
                raise RuntimeError(erm)
            attrs_diff.update({key: (dct_test[key], dct_check[key]) \
                               for key in diffkeys})
            if path == 'Header':
                ions = np.array([ion.decode() for ion in dct_test['Ions']])
                fosc = dct_test['Transitions_Oscillator_Strength']
                lang = dct_test['Transitions_Rest_Wavelength']
                #boxsize = dct_test['BoxSize']
                redshift = dct_test['Redshift']
                
                ions, ioninds = np.unique(ions, return_index=True)
                fosc = {ions[i]: fosc[ioninds[i]] for i in range(len(ions))}
                lang = {ions[i]: lang[ioninds[i]] for i in range(len(ions))}

        for path in gpaths_arrays:
            ta = np.array(ft[path])
            tc = np.array(fc[path])
            if not np.all(ta ==tc):
                erm = 'The sightlines for the input files are different'
                raise ValueError(erm)
            if path == 'VHubble_KMpS':
                vvals_test = ta  
                vvals_check = tc
                
        outofspectra = False
        specnum = 0
        # rest-frame EWs [Angstrom]
        EW_test = {}
        EW_check = {}
        nvals_test = {}
        nvals_check = {}
        while not outofspectra:
            EW_test[specnum] = {}
            EW_check[specnum] = {}
            nvals_test[specnum] = {}
            nvals_check[specnum] = {}
            for ion in ions:
                try:
                    tt = np.array(ft[plotpath.format(\
                                     specnum=specnum, ion=ion)])
                    tc = np.array(fc[plotpath.format(\
                                specnum=specnum, ion=ion)])
                    EW_test[specnum][ion] = np.sum(1. - tt) / len(tt) \
                              * np.average(np.diff(vvals_test)) * 1e5 / c.c \
                              * lang[ion]
                    EW_check[specnum][ion] = np.sum(1. - tc) / len(tc) \
                              * np.average(np.diff(vvals_check)) * 1e5 / c.c \
                              * lang[ion]
                    nvals_test[specnum][ion] = np.array(ft[lognpath.format(\
                                specnum=specnum, ion=ion)])[()]
                    nvals_check[specnum][ion] = np.array(fc[lognpath.format(\
                                specnum=specnum, ion=ion)])[()]
                except KeyError:
                    outofspectra = True
            if outofspectra:
                del EW_test[specnum]
                del EW_check[specnum]
                del nvals_test[specnum]
                del nvals_check[specnum]
            specnum += 1   
    efac = np.average(np.diff(vvals_check)) * 1e5 / c.c * len(vvals_check)
    
    minbother_red = minbother_vals[label_test]['minbother_red']\
                    if label_test in minbother_vals else 0.
    minbother_blue = minbother_vals[label_test]['minbother_blue']\
                    if label_test in minbother_vals else 0.
    lyalpha = 1215.6701
    
    title = '$z = {z:.2f}$, comparing\n' + \
            '{test}: {tlist}\n' +\
            '{check}: {clist}'
    dctfmt = '{key}={val}'
    attrs_note = set(attrs_diff.keys()) - set(attrs_ignore)
    dct_note = {key: attrs_diff[key] for key in attrs_note}
    dct_note.update({'minbother_red': (minbother_red,\
                            minbother_vals[label_check]['minbother_red']),\
                     'minbother_blue': (minbother_blue,\
                            minbother_vals[label_check]['minbother_blue']),\
                     })
    keys_note = sorted(list(dct_note.keys()))
    tlist = ', '.join([dctfmt.format(key=key, val=dct_note[key][0]) \
                       for key in keys_note])
    clist = ', '.join([dctfmt.format(key=key, val=dct_note[key][1]) \
                       for key in keys_note])
    ptitle = title.format(z=redshift,\
                          test=label_test, tlist=tlist,\
                          check=label_check, clist=clist)
    
    numions = len(ions) 
    ions = sorted(ions, key=lang.get) # sort by wavelength
    ions = ions[::-1]
    
    name = plotdir + 'EWdiff_log-{logdiff}_{tlabel}-vs-{clabel}.pdf'
    name = name.format(logdiff=logdiff, tlabel=label_test, clabel=label_check)
    fontsize = 10
    
    margin = 1.
    panelheight = 1.5
    panelwidth = 2.
    #titleheight = 0.5
    wspace = 0.25
    hspace = 0.25
    
    
    ylab = '$\\Delta \\, \\log_{{10}} \\, \\mathrm{{EW}}$' if logdiff else\
           '$\\Delta \\, \\mathrm{{EW}} \\; [\\mathrm{{\\AA}}]$'
    xlab = '$\\log_{{10}} \\, \\mathrm{{N}} \\; [\\mathrm{{cm}}^{{-2}}]$'
    ptbase = '{ion}, ${wl:.2f} \\, \\mathrm{{\AA}}$' # \\lambda = 
    tickparams = {'which': 'both', 'direction': 'in',\
                  'labelsize': fontsize - 1,\
                  'left': True, 'right': True, 'top': True, 'bottom': True,\
                  'labelleft': True, 'labelbottom': True}
    bbox = {'facecolor': 'white', 'alpha': 0.5, 'edgecolor': 'none'}
    
    maxdiff_rec = 0.
    
    ncols = 4
    nrows = (numions - 1) // ncols + 1
    
    totalheight = margin * 2 + panelheight * nrows + hspace * (nrows - 1) 
    totalwidth = margin * 2 + panelwidth * ncols + wspace * (ncols - 1) 
    
    bottom = margin / totalheight
    top = 1. - bottom
    left = margin / totalwidth
    right = 1. - left
            
    fig = plt.figure(figsize=(totalwidth, totalheight))
    grid = gsp.GridSpec(nrows=nrows, ncols=ncols,\
                        hspace=hspace, wspace=wspace,\
                        top=top, bottom=bottom, left=left, right=right)
    fig.suptitle(ptitle, fontsize=fontsize)
    axes = [fig.add_subplot(grid[i // ncols, i % ncols]) \
            for i in range(numions)]
            
    for ii in range(numions):
        ion = ions[ii]
        ax = axes[ii]
        
        tps = tickparams.copy()
        ax.tick_params(**tps)
        
        if numions - ii <= ncols: 
            ax.set_xlabel(xlab, fontsize=fontsize)
        if ii % ncols == 0:
            ax.set_ylabel(ylab, fontsize=fontsize)
            
        pt = ptbase.format(ion=ild.getnicename(ion), wl=lang[ion]) 
        if logdiff:        
            ax.text(0.95, 0.05, pt, fontsize=fontsize - 1,\
                    verticalalignment='bottom', \
                    horizontalalignment='right', transform=ax.transAxes,\
                    bbox=bbox)
        else:
            ax.text(0.95, 0.95, pt, fontsize=fontsize - 1,\
                    verticalalignment='top', \
                    horizontalalignment='right', transform=ax.transAxes,\
                    bbox=bbox)
            ax.set_yscale('log')
        specnums = sorted(nvals_test.keys())
        nt = np.array([nvals_test[specnum][ion] for specnum in specnums])
        nc = np.array([nvals_check[specnum][ion] for specnum in specnums])
        et = np.array([EW_test[specnum][ion] for specnum in specnums])
        ec = np.array([EW_check[specnum][ion] for specnum in specnums])
        
        if not np.all(nt == nc):
            print('Warning: different column densities for' +\
                  ' {ion}'.format(sn=specnum, ion=ion) +\
                  ':\n{t}\n{c}'.format(t=nt, c=nc))
        
        maxdiff = minbother_red if lang[ion] > 1.001 * lyalpha else\
                  minbother_blue
        if logdiff:
            ax.axhline(np.log10(1. + maxdiff), linestyle='dashed',\
                    color='black')
            yvals = np.log10(et) - np.log10(ec)
            yvals[et == ec] = 0.
        else:
            ax.axhline(maxdiff * efac * lang[ion])
            yvals = et - ec
        ax.scatter(nc, yvals, marker='o', color='gray', alpha=0.7)
        
        maxdiff_rec = max(maxdiff_rec, np.max(np.abs(yvals)))
                        
    plt.savefig(name)
    print('Max. difference: {}'.format(maxdiff_rec))              
    
                
    
    
