#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd

wdir = '/Users/nastasha/ciera/projects_lead/tdist_groups/data/'

# run on laptop, copy from quasar
# xargs -a list.txt scp -t new_folder
def make_copylist_groups():
    dir_laptop = '/Users/nastasha/phd/data/paper3/3dprof/'
    fn_halo = 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    fn_mass = 'filenames_L0100N1504_27_Mh0p5dex_1000_Mass_Trprof.txt'
    fn_vol = 'filenames_L0100N1504_27_Mh0p5dex_1000_Volume_Trprof.txt'
     
    min_mass_msun = 1e13
    copyfiles = []
    copylistn = wdir + 'copylist_trdist_groups_quasar.txt'

    halodat = pd.read_csv(dir_laptop + fn_halo, header=2, index_col='galaxyid', sep='\t')
    gids = halodat.index[halodat['M200c_Msun'] >= min_mass_msun]
    gids = np.array(gids)
    massfn = pd.read_csv(dir_laptop + fn_mass, sep='\t', index_col='galaxyid')
    copyfiles += list(massfn['filename'][gids])
    volfn = pd.read_csv(dir_laptop + fn_vol, sep='\t', index_col='galaxyid')
    copyfiles += list(volfn['filename'][gids])

    print('directory: ', '/'.join(copyfiles[0].split('/')[:-1]))
    copyfiles = [filen.split('/')[-1] for filen in copyfiles]
    outdat = '\n'.join(copyfiles)
    text_file = open(copylistn, "w")
    text_file.write(outdat)
    text_file.close()

    

