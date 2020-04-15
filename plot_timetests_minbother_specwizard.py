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

# a few useful defaults
imgdir = '/cosma5/data/dp004/dc-wije1/specwiz/imgs/timetests/'

logfiles = ['/cosma5/data/dp004/dc-wije1/specwiz/timetest_shacc_r32/stdout.L0100N1504-shacc-32-2009555',\
            '/cosma5/data/dp004/dc-wije1/specwiz/timetest_hiacc_r32/stdout.L0100N1504-hiacc-32-2009548',\
            '/cosma5/data/dp004/dc-wije1/specwiz/timetest_miacc_r32/stdout.L0100N1504-miacc-32-2009549',\
            '/cosma5/data/dp004/dc-wije1/specwiz/timetest_loacc_r32/stdout.L0100N1504-loacc-32-2009550',\
            ]
labels = ['shacc', 'hiacc', 'miacc', 'loacc']



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
    times = np.append(times[0][:, np.newaxis], -np.diff(times, axis=1), axis=1)
    
    fontsize = 12
    fig = plt.figure(figsize=(5.5, 3.))
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
    ax.legend(fontsize=fontsize)
    
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
                



