
# quest modules:
# module load python/anaconda3.6

import numpy as np
import h5py

import matplotlib.pyplot as plt

import readin_fire_data as rfd

# snapshot 50: redshift 0.5; hopefully enough to make a-scaling errors clear
firedata_test = '/projects/b1026/snapshots/AGN_suite/fiducial_jet/m12i_res57000/output/snapshot_050.hdf5'
firedata_params = '/projects/b1026/snapshots/AGN_suite/fiducial_jet/m12i_res57000/params.txt-usedvalues'
ddir = '/projects/b1026/nastasha/tests/start_fire/'


def make_simple_phasediagram():
    '''
    first test: just read in the data for a histogram
    '''