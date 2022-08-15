#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import pandas as pd

# run on laptop, copy from quasar
# xargs -a list.txt scp -t new_folder
def make_copylist_groups():
    dir_laptop = '/Users/nastasha/phd/data/paper3/3dprof/'
    fn_halo = 'halodata_L0100N1504_27_Mh0p5dex_1000.txt'
    fn_mass = 'filenames_L0100N1504_27_Mh0p5dex_1000_Mass_Trprof.txt'
    fn_vol = 'filenames_RecalL0025N0752_27_Mh0p5dex_1000_Volume_Trprof.txt'

    min_mass_msun = 1e13
    

