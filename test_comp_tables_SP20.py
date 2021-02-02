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

import make_maps_v3_master as m3

from importlib import reload
reload(m3) # testing script -> make sure I'm using the latest version

mdir  = '/net/luttero/data1/line_em_abs/v3_master_tests/ssh_tables_SP20/'
m3.ol.ndir = mdir

ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'ne10', 'fe17']


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

def plottables_SP20(line, z):
    
    
