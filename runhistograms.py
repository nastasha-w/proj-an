#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:25:03 2018

@author: wijers


Note that this is sort-of synced with the cosma version; some jobinds will not 
work, because the files are not here

"""
import numpy as np
import sys
import fnmatch 
import os
import string
import h5py

import eagle_constants_and_units as c
import cosmo_utils as cu

import makehistograms_basic as mh
import make_maps_opts_locs as ol
import makecddfs as mc
import coldens_rdist as crd
from loadnpz_and_plot import imreduce
import selecthalos as sh
import prof3d_galsets as p3g

if __name__ == '__main__':
    jobind = int(sys.argv[1])
else:
    jobind = None

def getcenfills(base, closevals=None, searchdir=None, tolerance=1e-4):
    if searchdir is None:
        searchdir = ol.ndir
    files = fnmatch.filter( next(os.walk(searchdir))[2], base%('*'))
    files_parts = [fil.split('/')[-1] for fil in files]
    files_parts = [fil.split('_') for fil in files_parts]
    files_parts = [[part if 'cen' in part else None for part in fil] for fil in files_parts]
    scens = [part for fil in files_parts for part in fil]
    scens = list(set(scens))
    if None in scens:
        scens.remove(None)
    scens = np.array([cen[4:] for cen in scens])
    
    if closevals is not None:
        #print(np.array(closevals).astype(np.float64))
        #print(float(scens[0]))
        ismatch = [np.min(np.abs(float(cen) - np.array(closevals).astype(np.float64))) <= tolerance for cen in scens]
        #print(ismatch)
        scens = scens[np.array(ismatch)]
    return scens
    
    
fillsba = [str(float(i)) for i in (np.arange(6)/6.+1/12.)*400.]
fillsea = [str(float(i)) for i in (np.arange(16)/16.+1/32.)*100.]
fillsea25 = [str(float(i)) for i in (np.arange(4)/4. + 1./8.)*25. ]   

### z=0.00, 32000pix 
# 100Mpc fixz EA/BA o7
edgesT = np.arange(36)/35.*(6.7-3.2) + 3.2
edgesrho = np.arange(74)/73.*(-23.7 + 31.0) -31.0
edgesN = np.arange(36)/35.*(16.9-13.4) + 13.4

if jobind == 1:
    print('Doing BA400, 6 slices, (NO7, rho, T), fixz')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
                    'Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
                    'coldens_o7_BA-L400N1024_32_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz',\
                    bins = [edgesrho,edgesT,edgesN],\
                    dimlabels = ['Density_w_NO7','Temperature_w_NO7','NO7'],\
                    save=mh.pdir +'hist_coldens_o7_BA-L400N1024_32_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_Temperature_Density',\
                    fills = fillsba)
elif jobind == 2:
    print('Doing EA100, 1 slice, (NO7, rho, T), fixz')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.2_C2Sm_32000pix_100.0slice_z-projection.npz',\
                    'coldens_o7_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    bins = [edgesrho,edgesT,edgesN],\
                    dimlabels = ['Density_w_NO7','Temperature_w_NO7','NO7'],\
                    save=mh.pdir +'hist_coldens_o7_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density')

# 100Mpc PtAb EA/BA o7
edgesT = np.arange(47)/46.*(7.8-3.2) + 3.2
edgesrho = np.arange(86)/85.*(-23.3 + 31.8) -31.8
edgesN = np.array(list((np.arange(25)/24.*(11.0+13.0) -13.0)[:-1]) + list(np.arange(78)/77.*(18.7-11.0) + 11.0)) 
edgesfO = np.arange(54)/53.*(-1.2 + 6.5) -6.5

if jobind == 3:
    print('Doing BA400, 6 slices, (NO7, rho, T, [O])')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o7_PtAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
                    'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
                    'coldens_o7_BA-L400N1024_32_test3.2_PtAb_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_BA-L400N1024_32_test3.21_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
                    bins = [edgesrho,edgesT,edgesN,edgesfO],\
                    dimlabels = ['Density_w_NO7','Temperature_w_NO7','NO7','OxygenMassFrac_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o7_BA-L400N1024_32_test3.2_PtAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_Temperature_Density_ElementAbundance-Oxygen',\
                    fills = fillsba)
elif jobind == 4:
    print('Doing EA100, 1 slice, (NO7, rho, T, [O])')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    bins = [edgesrho,edgesT,edgesN,edgesfO],\
                    dimlabels = ['Density_w_NO7','Temperature_w_NO7','NO7','OxygenMassFrac_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density_ElementAbundance-Oxygen')

# 100Mpc fixz EA o8
edgesT = np.arange(38)/37.*(7.3-3.6) + 3.6
edgesrho = np.arange(73)/72.*(-23.5 + 30.7) -30.7
edgesN = np.arange(24)/23.*(16.5-14.2) + 14.2

if jobind == 5:
    print('Doing EA100, 1 slice, (NO8, rho, T), fixz')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o8_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'Temperature_T4EOS_coldens_o8_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'coldens_o8_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    bins = [edgesrho,edgesT,edgesN],\
                    dimlabels = ['Density_w_NO8','Temperature_w_NO8','NO8'],\
                    save=mh.pdir +'hist_coldens_o8_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density')

# 100Mpc PtAb EA/BA o8
edgesT = np.arange(41)/40.*(7.6-3.6) + 3.6
edgesrho = np.arange(86)/85.*(-22.9 + 31.4) -31.4
edgesN = np.array(list((np.arange(24)/23.*(11.0+12.0) -12.0)[:-1]) + list(np.arange(71)/70.*(18.0-11.0) + 11.0)) 
edgesfO = np.arange(54)/53.*(-1.2 + 6.5) -6.5

if jobind == 6:
    print('Doing EA100, 1 slice, (NO8, rho, T, [O])')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    bins = [edgesrho,edgesT,edgesN,edgesfO],\
                    dimlabels = ['Density_w_NO8','Temperature_w_NO8','NO8','OxygenMassFrac_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density_ElementAbundance-Oxygen')

elif jobind == 7:
    print('Doing BA400, 6 slices, (NO8, [O])')
    mh.makehist_fromnpz('coldens_o8_BA-L400N1024_32_test3.21_PtAb_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_BA-L400N1024_32_test3.21_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz',\
                    bins = [edgesN,edgesfO],\
                    dimlabels = ['NO8','OxygenMassFrac_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o8_BA-L400N1024_32_test3.21_PtAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen',\
                    fills = fillsba)

# 6.25Mpc PtAb EA o7
edgesT = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(60)/59.*(8.4-2.5) +2.5))
edgesrho = np.array(list((np.arange(10)/9.*(-40.+76.) -76.0)[:-1]) + list((np.arange(9)/8.*(-32.0+40.) -40.0)[:-1]) + list(np.arange(88)/87.*(-23.3+32.0) - 32.0))
edgesN = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(69)/68.*(17.8-11.0) + 11.0)) 
edgesfO = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(59)/58.*(-1.2+ 7.0) - 7.0))

if jobind == 8:
    print('Doing EA100, 16 slices, (NO7, rho, T, [O])')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    bins = [edgesrho,edgesT,edgesN,edgesfO],\
                    dimlabels = ['Density_w_NO7','Temperature_w_NO7','NO7','OxygenMassFrac_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature_Density_ElementAbundance-Oxygen',\
                    fills = fillsea)

# 6.25Mpc PtAb EA o8
edgesT = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho = np.arange(97)/96.*(-22.9+32.5) - 32.5
edgesN = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0)) 
edgesfO = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(58)/57.*(-1.3+ 7.0) - 7.0))

if jobind == 9:
    print('Doing EA100, 16 slices, (NO8, rho, T, [O])')
    mh.makehist_fromnpz('Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    'coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    bins = [edgesrho,edgesT,edgesN,edgesfO],\
                    dimlabels = ['Density_w_NO8','Temperature_w_NO8','NO8','OxygenMassFrac_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature_Density_ElementAbundance-Oxygen',\
                    fills = fillsea)

# EA-100 PtAb 16 slices O6/7/8 correlations
edgesNO7 = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(69)/68.*(17.8-11.0) + 11.0)) 
edgesNO8 = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0)) 
edgesNO6 = np.array(list((np.arange(51)/50.*(11.0+39.0) -39.0)[:-1]) + list(np.arange(57)/56.*(16.6-11.0) + 11.0)) 

if jobind == 10:
    print('Doing EA100, 16 slices, (NO6, NO7, NO8)')
    mh.makehist_fromnpz('coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                        'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                        'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    bins = [edgesNO6, edgesNO7, edgesNO8],\
                    dimlabels = ['NO6', 'NO7', 'NO8'],\
                    save=mh.pdir +'hist_coldens_o6-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS',\
                    fills = fillsea)

# larger hi-res region
edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
edgesNO8 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(97)/96.*(17.6-8.0) + 8.0)) 
edgesNO6 = np.array(list((np.arange(48)/47.*(8.0+39.0) -39.0)[:-1]) + list(np.arange(87)/86.*(16.6-8.0) + 8.0)) 
if jobind == 11:
    print('Doing EA100, 16 slices, (NO6, NO7, NO8)')
    mh.makehist_fromnpz('coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                        'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                        'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    bins = [edgesNO6, edgesNO7, edgesNO8],\
                    dimlabels = ['NO6', 'NO7', 'NO8'],\
                    save=mh.pdir +'hist_coldens_o6-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_hires-8',\
                    fills = fillsea)

# EA-100 PtAb 16 slices O6/7/8 correlations
edgesNO6 = np.arange(88)/87.*(16.6-7.9) + 7.9
edgesNO7 = np.arange(93)/92.*(17.8-8.6) + 8.6
edgesNO8 = np.arange(73)/72.*(17.3-10.1) + 10.1 
if jobind == 12:
    print('Doing EA100, 1 slice, (NO6, NO7, NO8)')
    mh.makehist_fromnpz('coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                        'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalboxW.npz',\
                        'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    bins = [edgesNO6, edgesNO7, edgesNO8],\
                    dimlabels = ['NO6', 'NO7', 'NO8'],\
                    save=mh.pdir +'hist_coldens_o6-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox')


# EA-100 PtAb 1 slice Ne8 O7/8 correlations
edgesNNe8 = np.arange(78)/77.*(15.7-8.0) + 8.0
edgesNO7 = np.arange(93)/92.*(17.8-8.6) + 8.6
edgesNO8 = np.arange(73)/72.*(17.3-10.1) + 10.1 
if jobind == 13:
    print('Doing EA100, 1 slice, (NNe8, NO7, NO8)')
    mh.makehist_fromnpz('coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-sum_T4SFR_totalbox.npz',\
                        'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalboxW.npz',\
                        'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    bins = [edgesNNe8, edgesNO7, edgesNO8],\
                    dimlabels = ['NNe8', 'NO7', 'NO8'],\
                    save=mh.pdir +'hist_coldens_ne8-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox')

# EA-100 PtAb 16 slices Ne8 O7/8 correlations
edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
edgesNO8 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(97)/96.*(17.6-8.0) + 8.0)) 
edgesNNe8 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(88)/77.*(15.7-8.0) + 8.0))
if jobind == 14:
    print('Doing EA100, 16 slices, (NNe8, NO7, NO8)')
    mh.makehist_fromnpz('coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen%s_T4SFR.npz',\
                        'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                        'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    bins = [edgesNNe8, edgesNO7, edgesNO8],\
                    dimlabels = ['NNe8', 'NO7', 'NO8'],\
                    save=mh.pdir +'hist_coldens_ne8-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS',\
                    fills = fillsea)


# EA-100 PtAb 16 slices fO-mass, fO-NO7, NO7 correlations
edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
edgesfO_NO7 = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(59)/58.*(-1.2+ 7.0) - 7.0))
edgesfO_mass = np.array(list((np.arange(7)/6.*(-10.+40.) -40.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(57)/56.*(-1.4+ 7.0) - 7.0)) 
if jobind == 15:
    print('Doing EA100, 16 slices, (NO7, [O]-mass, [O]-NO7)')
    mh.makehist_fromnpz('coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    'ElementAbundance-Oxygen_T4EOS_Mass_T4EOS_L0100N1504_28_test3.11_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    bins = [edgesNO7,edgesfO_NO7,edgesfO_mass],\
                    dimlabels = ['NO7','OxygenMassFrac_w_NO7','OxygenMassFrac_w_Mass'],\
                    save=mh.pdir +'hist_coldens_o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen',\
                    fills = fillsea)

# EA-100 PtAb 16 slices fO-mass, fO-NO8, NO8 correlations
edgesNO8 = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0))
edgesfO_NO8 = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(58)/57.*(-1.3+ 7.0) - 7.0))
edgesfO_mass = np.array(list((np.arange(7)/6.*(-10.+40.) -40.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(57)/56.*(-1.4+ 7.0) - 7.0)) 
if jobind == 16:
    print('Doing EA100, 16 slices, (NO7, [O]-mass, [O]-NO7)')
    mh.makehist_fromnpz('coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    'ElementAbundance-Oxygen_T4EOS_Mass_T4EOS_L0100N1504_28_test3.11_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                    bins = [edgesNO8,edgesfO_NO8,edgesfO_mass],\
                    dimlabels = ['NO8','OxygenMassFrac_w_NO8','OxygenMassFrac_w_Mass'],\
                    save=mh.pdir +'hist_coldens_o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen',\
                    fills = fillsea)

# EA-100 PtAb 1 slice fO-mass, fO-NO7, NO7 correlations
edgesNO7 = np.arange(93)/92.*(17.8-8.6) + 8.6
edgesfO_NO7 = np.arange(37)/36.*(-1.3 + 4.9) -4.9
edgesfO_mass = np.arange(69)/68.*(-1.4+ 8.2) - 8.2  
if jobind == 17:
    print('Doing EA100, 1 slice, (NO7, [O]-mass, [O]-NO7)')
    mh.makehist_fromnpz('coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'ElementAbundance-Oxygen_T4EOS_Mass_T4EOS_L0100N1504_28_test3.11_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    bins = [edgesNO7,edgesfO_NO7,edgesfO_mass],\
                    dimlabels = ['NO7','OxygenMassFrac_w_NO7','OxygenMassFrac_w_Mass'],\
                    save=mh.pdir +'hist_coldens_o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen')

# EA-100 PtAb 1 slice fO-mass, fO-NO8, NO8 correlations
edgesNO8 = np.arange(83)/82.*(17.6-9.4) + 9.4
edgesfO_NO8 = np.arange(31)/30.*(-1.3 + 4.3) - 4.3
edgesfO_mass = np.arange(69)/68.*(-1.4+ 8.2) - 8.2 
if jobind == 18:
    print('Doing EA100, 1 slice, (NO7, [O]-mass, [O]-NO7)')
    mh.makehist_fromnpz('coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz',\
                    'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    'ElementAbundance-Oxygen_T4EOS_Mass_T4EOS_L0100N1504_28_test3.11_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz',\
                    bins = [edgesNO8,edgesfO_NO8,edgesfO_mass],\
                    dimlabels = ['NO8','OxygenMassFrac_w_NO8','OxygenMassFrac_w_Mass'],\
                    save=mh.pdir +'hist_coldens_o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen')
    
# EA-100 PtAb 16 slices ion-weighted rho, T, fO mismatch
edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
edgesfO_NO7 = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(59)/58.*(-1.2+ 7.0) - 7.0))
edgesT_NO7 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(60)/59.*(8.5-2.5) +2.5))
edgesrho_NO7 = np.array(list((np.arange(10)/9.*(-40.+76.) -76.0)[:-1]) + list((np.arange(9)/8.*(-32.0+40.) -40.0)[:-1]) + list(np.arange(88)/87.*(-23.3+32.0) - 32.0))

rho_NO7_name_16_28 = 'Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO7_name_16_28 = 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO7_name_16_28 = 'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
fO_NO7_name_16_28 = 'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
                    
edgesNO8 = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0))
edgesfO_NO8 = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(58)/57.*(-1.3+ 7.0) - 7.0))
edgesT_NO8 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho_NO8 = np.arange(97)/96.*(-22.9+32.5) - 32.5

rho_NO8_name_16_28 = 'Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO8_name_16_28 = 'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO8_name_16_28 = 'coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
fO_NO8_name_16_28 = 'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'

if jobind == 19:
    print('Doing EA100, 16 slices, (NO7, NO8, rho_NO7, rho_NO8)')
    mh.makehist_fromnpz(NO7_name_16_28, NO8_name_16_28, rho_NO7_name_16_28, rho_NO8_name_16_28,\
                    bins = [edgesNO7,edgesNO8, edgesrho_NO7, edgesrho_NO8],\
                    dimlabels = ['NO7','NO8','Density_w_NO7', 'Density_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density',\
                    fills = fillsea)    
    
if jobind == 20:
    print('Doing EA100, 16 slices, (NO7, NO8, T_NO7, T_NO8)')
    mh.makehist_fromnpz(NO7_name_16_28, NO8_name_16_28, T_NO7_name_16_28, T_NO8_name_16_28,\
                    bins = [edgesNO7,edgesNO8, edgesT_NO7, edgesT_NO8],\
                    dimlabels = ['NO7','NO8','Temperature_w_NO7', 'Temperature_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)   

if jobind == 21:
    print('Doing EA100, 16 slices, (NO7, NO8, fO_NO7, fO_NO8)')
    mh.makehist_fromnpz(NO7_name_16_28, NO8_name_16_28, fO_NO7_name_16_28, fO_NO8_name_16_28,\
                    bins = [edgesNO7,edgesNO8, edgesfO_NO7, edgesfO_NO8],\
                    dimlabels = ['NO7','NO8','OxygenMassFrac_w_NO7', 'OxygenMassFrac_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen',\
                    fills = fillsea)   

if jobind == 22:
    print('Doing EA100, 16 slices, (rho_NO7, rho_NO8, T_NO7, T_NO8)')
    mh.makehist_fromnpz(rho_NO7_name_16_28, rho_NO8_name_16_28, T_NO7_name_16_28, T_NO8_name_16_28,\
                    bins = [edgesrho_NO7, edgesrho_NO8,  edgesT_NO7, edgesT_NO8],\
                    dimlabels = ['Density_w_NO7', 'Density_w_NO8', 'Temperature_w_NO7', 'Temperature_w_NO8'],\
                    save=mh.pdir +'hist_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_coldens_o7-o8_weighted_Density_Temperature',\
                    fills = fillsea)   

# get ion balance histograms (not direct from 2d maps -> call separate function)
if jobind == 23:
    mh.makehistograms_ionbals(kind = 'diff')
elif jobind == 24:
    mh.makehistograms_ionbals(kind = 'o7')
elif jobind == 25:
    mh.makehistograms_ionbals(kind = 'o8')
elif jobind == 26:
    mh.makehistograms_ionbals(kind = 'logdiff')


rho_NO6_name_16_27 = 'Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO6_name_16_27 = 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO6_name_16_27 = 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO6 = np.array(list((np.arange(5)/4.*(0.0+40.) -40.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(71)/70.*(17.-10.) +10.))
edgesT_NO6 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesrho_NO6 = np.arange(111)/110.*(-22.0+33.0) - 33.0

rho_NO8_name_16_27 = 'Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.11_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO8_name_16_27 = 'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO8_name_16_27 = 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO8 = np.array(list((np.arange(4)/3.*(0.0+30.) -30.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(81)/80.*(18.-10.) +10.))
edgesT_NO8 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesrho_NO8 = np.array(list((np.arange(5)/4.*(-40.+80.) -80.0)[:-1]) + list((np.arange(8)/7.*(-33.0+40.) -40.0)[:-1]) + list(np.arange(111)/110.*(-22.0+33.0) - 33.0))

if jobind == 27:
    print('Doing EA100, 16 slices, (NO6, NO8, T_NO6, T_NO8)')
    mh.makehist_fromnpz(NO6_name_16_27, NO8_name_16_27, T_NO6_name_16_27, T_NO8_name_16_27,\
                    bins = [edgesNO6, edgesNO8, edgesT_NO6, edgesT_NO8],\
                    dimlabels = ['NO6','NO8','Temperature_w_NO6', 'Temperature_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)   
if jobind == 28:
    print('Doing EA100, 16 slices, (NO6, NO8, rho_NO6, rho_NO8)')
    mh.makehist_fromnpz(NO6_name_16_27, NO8_name_16_27, rho_NO6_name_16_27, rho_NO8_name_16_27,\
                    bins = [edgesNO6, edgesNO8, edgesrho_NO6, edgesrho_NO8],\
                    dimlabels = ['NO6','NO8','Density_w_NO6', 'Density_w_NO8'],\
                    save=mh.pdir +'hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density',\
                    fills = fillsea) 

if jobind == 29:
    print('Doing EA100, 16 slices, (rho_NO6, rho_NO8, T_NO6, T_NO8)')
    mh.makehist_fromnpz(rho_NO6_name_16_27, rho_NO8_name_16_27, T_NO6_name_16_27, T_NO8_name_16_27,\
                    bins = [edgesrho_NO6, edgesrho_NO8,  edgesT_NO6, edgesT_NO8],\
                    dimlabels = ['Density_w_NO6', 'Density_w_NO8', 'Temperature_w_NO6', 'Temperature_w_NO8'],\
                    save=mh.pdir +'hist_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_coldens_o6-o8_weighted_Density_Temperature',\
                    fills = fillsea)   

# snap 28 o6-o7 
edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
edgesfO_NO7 = np.array(list((np.arange(15)/14.*(-10.+80.) -80.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(59)/58.*(-1.2+ 7.0) - 7.0))
edgesT_NO7 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho_NO7 = np.array(list((np.arange(5)/4.*(-36.+76.) -76.0)[:-1]) + list((np.arange(5)/4.*(-32.0+36.) -36.0)[:-1]) + list(np.arange(88)/87.*(-23.3+32.0) - 32.0))

rho_NO7_name_16_28 = 'Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO7_name_16_28 = 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO7_name_16_28 = 'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
fO_NO7_name_16_28 = 'ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'

edgesNO6 = np.array(list((np.arange(10)/9.*(9.0+36.0) -36.0)[:-1]) + list((np.arange(3)/2.*(11.0-9.0) +9.0)[:-1]) + list(np.arange(57)/56.*(16.6-11.0) + 11.0))
edgesfO_NO6 = np.array(list((np.arange(9)/8.*(-10.+90.) -90.0)[:-1]) + list((np.arange(4)/3.*(-7.+10.) -10.)[:-1]) + list(np.arange(60)/59.*(-1.1+ 7.0) - 7.0))
edgesT_NO6 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho_NO6 = np.array(list((np.arange(5)/4.*(-36.+76.) -76.0)[:-1]) + list((np.arange(5)/4.*(-32.0+36.) -36.0)[:-1]) + list(np.arange(90)/89.*(-23.1+32.0) - 32.0))

rho_NO6_name_16_28 =   'Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO6_name_16_28 = 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO6_name_16_28   = 'coldens_o6_L0100N1504_28_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
fO_NO6_name_16_28 ='ElementAbundance-Oxygen_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'

if jobind == 30:
    print('Doing EA100, 16 slices, (NO6, NO7, rho_NO6, rho_NO7)')
    mh.makehist_fromnpz(NO6_name_16_28, NO7_name_16_28, rho_NO6_name_16_28, rho_NO7_name_16_28,\
                    bins = [edgesNO6, edgesNO7, edgesrho_NO6, edgesrho_NO7],\
                    dimlabels = ['NO6','NO7','Density_w_NO6', 'Density_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o6-o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density',\
                    fills = fillsea)    
    
if jobind == 31:
    print('Doing EA100, 16 slices, (NO6, NO7, T_NO6, T_NO7)')
    mh.makehist_fromnpz(NO6_name_16_28, NO7_name_16_28, T_NO6_name_16_28, T_NO7_name_16_28,\
                    bins = [edgesNO6, edgesNO7, edgesT_NO6, edgesT_NO7],\
                    dimlabels = ['NO6','NO7','Temperature_w_NO6', 'Temperature_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o6-o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)   

if jobind == 32:
    print('Doing EA100, 16 slices, (NO6, NO7, fO_NO6, fO_NO7)')
    mh.makehist_fromnpz(NO6_name_16_28, NO7_name_16_28, fO_NO6_name_16_28, fO_NO7_name_16_28,\
                    bins = [edgesNO6, edgesNO7, edgesfO_NO6, edgesfO_NO7],\
                    dimlabels = ['NO6','NO7','OxygenMassFrac_w_NO6', 'OxygenMassFrac_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o6-o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen',\
                    fills = fillsea)   
NO6_name_16_27 = 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' 
NO7_name_16_27 = 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NO8_name_16_27 = 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NNe9_name_16_27 = 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO6 = np.array(list((np.arange(10)/9.*(9.0+36.0) -36.0)[:-1]) + list((np.arange(3)/2.*(11.0-9.0) +9.0)[:-1]) + list(np.arange(57)/56.*(16.6-11.0) + 11.0))
edgesNO7 = np.array(list((np.arange(9)/8.*(9.0+23.0) -23.0)[:-1])+ list((np.arange(3)/2.*(11.0-9.0) +9.0)[:-1]) + list(np.arange(69)/68.*(17.8-11.0) + 11.0)) 
edgesNO8 = np.array(list((np.arange(9)/8.*(9.0+23.0) -23.0)[:-1]) + list((np.arange(3)/2.*(11.-9.0) +9.0)[:-1]) + list(np.arange(71)/70.*(18.-11.) +11.))
edgesNNe9 = np.array(list((np.arange(9)/8.*(9.0+23.0) -23.0)[:-1]) + list((np.arange(3)/2.*(11.-9.0) +9.0)[:-1]) + list(np.arange(61)/60.*(17.-11.) +11.))

if jobind == 33:
    print('Doing EA100, 16 slices, (NO6, NO7, NO8, NNe9)')
    mh.makehist_fromnpz(NO6_name_16_27, NO7_name_16_27, NO8_name_16_27, NNe9_name_16_27,\
                    bins = [edgesNO6, edgesNO7, edgesNO8, edgesNNe9],\
                    dimlabels = ['NO6','NO7','NO8', 'NNe9'],\
                    save=mh.pdir +'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS',\
                    fills = fillsea)   

fO_NO8_name_16_27 = 'ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
fNe_NNe9_name_16_27 = 'ElementAbundance-Neon_T4EOS_coldens_ne9_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'

mu_logNo8 = 15.5 
sigma_logNo8 = 0.3 
mu_logNo6 = 13.263 
sigma_logNo6 = 0.11
mu_logNne9 = 15.4
sigma_logNne9 = 0.3
ul_logNo7 = 14.9
    
edgesNO6 = np.array([-40., 12.] + list(np.arange(16)/15.*(14.0-12.5) + 12.5) + [14.5, 15.0, 17.]) # focus on region of detection: 13.263 +- 0.11
edgesNO7 = np.array([-23.0,  14.4, 14.9, 15.4, 18.]) # only need to include/exclude upper limit (14.9); leave a bit of room for variation
edgesNO8 = np.array(list((np.arange(3)/2.*(9.0+23.0) -23.0)[:-1]) + list((np.arange(4)/3.*(12.- 9.0) +9.0)[:-1]) + list(np.arange(54)/53.*(17.3-12.) +12.))
edgesNNe9 = np.array(list((np.arange(3)/2.*(9.0+23.0) -23.0)[:-1]) + list((np.arange(4)/3.*(12.- 9.0) +9.0)[:-1]) + list(np.arange(51)/50.*(17.-12.) +12.))
edgesfO_NO8 = np.array([-83., -10., -9., -8., -7., -6.] + list(np.arange(41)/40.*(-1.2 + 5.2) - 5.2) )
edgesfNe_NNe9 = np.array([-83., -10., -9., -8., -7., -6.] + list(np.arange(40)/39.*(-1.9 + 5.8) - 5.8) )

if jobind == 34:
    print('Doing EA100, 16 slices, (NO6, NO7, NO8, NNe9, fO_NO8, fNe_Ne9)')
    mh.makehist_fromnpz(NO6_name_16_27, NO7_name_16_27, NO8_name_16_27, NNe9_name_16_27, fO_NO8_name_16_27, fNe_NNe9_name_16_27,\
                    bins = [edgesNO6, edgesNO7, edgesNO8, edgesNNe9, edgesfO_NO8, edgesfNe_NNe9],\
                    dimlabels = ['NO6','NO7','NO8', 'NNe9', 'OxygenMassFrac_w_NO8', 'NeonMassFrac_w_NNe9'],\
                    save=mh.pdir + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and weighted_ElementAbundance-Oxygen_ElementAbundance-Neon',\
                    fills = fillsea)  



### emission - emission-weighted overdensity correlations, to gauge effects of 
## smoothing and ISM emission
o8base = 'emission_o8_L0050N0752_27_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_z-projection_noEOS*.npz'
o8density = 'Density_noEOS_emission_o8_SmAb_noEOS_L0050N0752_27_test3.31_C2Sm_16000pix_6.25slice_zcen3.125_z-projection.npz'
# based on individual min/max values:
edges_o8em = np.array([-np.inf] + list(np.arange(-22., -10.5, 1.)) + list(np.arange(-10.9, 4.25, 0.1)) +[np.inf])
edges_o8dens = np.arange(-32., -23.65, 0.1)
if jobind in range(35, 35 + 16):
    o8files = fnmatch.filter( next(os.walk(ol.ndir))[2], o8base)
    o8files.sort()
    o8file = o8files[jobind - 35]
    print('Doing EA50, 1 of 8 slices, (Sbo8, rho_sbo8)')
    mh.makehist_fromnpz(o8file, o8density,\
                    bins = [edges_o8em, edges_o8dens],\
                    dimlabels = ['SBO8', 'Density_w_SBO8'],\
                    save=mh.pdir + o8file[:-4] + '_and_weighted_Density_noEOS_nokernel',\
                    fills=None) 

o7base = 'emission_o7r_L0050N0752_27_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_z-projection_noEOS*.npz'
o7density = 'Density_noEOS_emission_o7r_SmAb_noEOS_L0050N0752_27_test3.31_C2Sm_16000pix_6.25slice_zcen3.125_z-projection.npz'

edges_o7rdens = np.arange(-32.2, -23.65, 0.1)
edges_o7rem = np.array([-np.inf] + list(np.arange(-23., -10.5, 1.)) + list(np.arange(-10.9, 3.75, 0.1)) +[np.inf])
if jobind in range(51, 51 + 16):
    o7files = fnmatch.filter( next(os.walk(ol.ndir))[2], o7base)
    o7files.sort()
    o7file = o7files[jobind - 51]
    print('Doing EA50, 1 of 8 slices, (Sbo7r, rho_sbo7r)')
    mh.makehist_fromnpz(o7file, o7density,\
                    bins = [edges_o7rem, edges_o7rdens],\
                    dimlabels = ['SBO7r', 'Density_w_SBO7r'],\
                    save=mh.pdir + o7file[:-4] + '_and_weighted_Density_noEOS_nokernel',\
                    fills=None)

o8base = 'emission_o8_L0100N1504_19_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_x25.0-pm50.0_y25.0-pm50.0_z-projection_noEOS*.npz'
o8density = 'Density_noEOS_emission_o8_SmAb_noEOS_L0100N1504_19_test3.31_C2Sm_16000pix_6.25slice_zcen3.125_x25.0-pm50.0_y25.0-pm50.0_z-projection.npz'
# based on individual min/max values:
edges_o8em = np.array([-np.inf] + list(np.arange(-23., -10.5, 1.)) + list(np.arange(-10.9, 4.85, 0.1)) +[np.inf])
edges_o8dens = np.arange(-31., -22.85, 0.1)
if jobind in range(67, 67 + 16):
    import psfconv_map as pm
    
    o8files = fnmatch.filter( next(os.walk(ol.ndir))[2], o8base)
    o8files.sort()
    o8file = o8files[jobind - 67]
    outname = o8file[:-4] + '_and_wtd_Density_noEOS_nokernel'
    outname1 = outname.replace('SFRtoSNetoEmOverLambda', 'SFRemissest')
    outname = outname1.replace('kernelconv_', '')
    
#    if 'fwhm' in outname:
#        #parts = outname.split('_')
#        #kpart = [part if 'fwhm' in part else None for part in parts]
#        #kpart = list(set(kpart))
#        #kpart.remove(None)
#        #kpart = kpart[0]
#        #kernel = float(kpart[5:-6]) #fwhm-<value>arcsec
#        
#        
#    else:
#        sel = (slice(None, None, None), )*2
    convk = pm.get_kernel(60., 1. / 0.498972 - 1., 50., 16000, numsig=10) # exclude region contaminated by edges effect with maximum kernel size
    excl = convk.shape[0] // 2 + 1
    sel = (slice(excl, -1 * excl, None), )*2
        
    print('Doing EA100-part, 1 of 16 slices, (Sbo8, rho_sbo8)')
    mh.makehist_fromnpz(o8file, o8density,\
                    bins = [edges_o8em, edges_o8dens],\
                    dimlabels = ['SBO8', 'Density_w_SBO8'],\
                    save=mh.pdir + outname,\
                    sel=sel,\
                    fills=None) 

o7base = 'emission_o7r_L0100N1504_19_test3.31_SmAb_C2Sm_16000pix_6.25slice_zcen3.125_x25.0-pm50.0_y25.0-pm50.0_z-projection_noEOS*.npz'
o7density = 'Density_noEOS_emission_o7r_SmAb_noEOS_L0100N1504_19_test3.31_C2Sm_16000pix_6.25slice_zcen3.125_x25.0-pm50.0_y25.0-pm50.0_z-projection.npz'

edges_o7rdens = np.arange(-31.1, -22.75, 0.1)
edges_o7rem = np.array([-np.inf] + list(np.arange(-23., -10.5, 1.)) + list(np.arange(-10.9, 4.35, 0.1)) +[np.inf])
if jobind in range(83, 83 + 16):
    import psfconv_map as pm
    
    o7files = fnmatch.filter( next(os.walk(ol.ndir))[2], o7base)
    o7files.sort()
    o7file = o7files[jobind - 83]
    outname = o7file[:-4] + '_and_wtd_Density_noEOS_nokernel'
    outname1 = outname.replace('SFRtoSNetoEmOverLambda', 'SFRemissest')
    outname = outname1.replace('kernelconv_', '')
    
#    if 'fwhm' in outname:
#        #parts = outname.split('_')
#        #kpart = [part if 'fwhm' in part else None for part in parts]
#        #kpart = list(set(kpart))
#        #kpart.remove(None)
#        #kpart = kpart[0]
#        #kernel = float(kpart[5:-6]) #fwhm-<value>arcsec
#        
#
#    else:
#        sel = (slice(None, None, None), )*2
    convk = pm.get_kernel(60., 1. / 0.498972 - 1., 50., 16000, numsig=10) 
    excl = convk.shape[0] // 2 + 1
    sel = (slice(excl, -1 * excl, None), )*2
        
    print('Doing EA100-part, 1 of 16 slices, (Sbo7r, rho_sbo7r)')
    mh.makehist_fromnpz(o7file, o7density,\
                    bins = [edges_o7rem, edges_o7rdens],\
                    dimlabels = ['SBO7r', 'Density_w_SBO7r'],\
                    save=mh.pdir + outname,\
                    sel=sel,\
                    fills=None)

## O6 phase diagram low z
rho_NO6_name_16_27 = 'Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO6_name_16_27 = 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO6_name_16_27 = 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO6 = np.array(list((np.arange(5)/4.*(0.0+40.) -40.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(71)/70.*(17.-10.) +10.))
edgesT_NO6 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesrho_NO6 = np.arange(111)/110.*(-22.0+33.0) - 33.0

if jobind == 99:
    print('Doing EA100, 16 slices, (NO6, rho_NO6, T_NO6)')
    mh.makehist_fromnpz(NO6_name_16_27, rho_NO6_name_16_27, T_NO6_name_16_27,\
                    bins = [edgesNO6, edgesrho_NO6, edgesT_NO6],\
                    dimlabels = ['NO6', 'Density_w_NO6','Temperature_w_NO6'],\
                    save=mh.pdir +'hist_coldens_o6_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature',\
                    fills = fillsea)   

fO_Sm_NO7_name = 'SmoothedElementAbundance-Oxygen_wiEOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
fO_Sm_NO8_name = 'SmoothedElementAbundance-Oxygen_wiEOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO7_name_16_28 = 'coldens_o7_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NO8_name_16_28 = 'coldens_o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO8 = np.array(list((np.arange(3)/2.*(9.0+23.0) -23.0)[:-1]) + list((np.arange(4)/3.*(12.- 9.0) +9.0)[:-1]) + list(np.arange(57)/56.*(17.6-12.) +12.))
edgesNO7 = np.array(list((np.arange(3)/2.*(9.0+23.0) -23.0)[:-1]) + list((np.arange(4)/3.*(12.- 9.0) +9.0)[:-1]) + list(np.arange(59)/58.*(17.8-12.) +12.))
logfOsol = np.log10(ol.solar_abunds['oxygen'])
edgesfO_NO7 = np.array(list((np.arange(3)/2.*(-8.0 + 68.0) -68.0)[:-1]) + [-8., -7.][:-1] + list(np.arange(57)/56.*(-1.4 + 7.) - 7.)) # some margin around edges, then get values that are nice (to 0.1 decimal) in solar units
edgesfO_NO7 = np.round(edgesfO_NO7 - logfOsol, 1) + logfOsol
edgesfO_NO8 =  edgesfO_NO7 # very similar min/max 

if jobind == 100:
    print('Doing EA100, 16 slices, (NO7, NO8, fO-Sm_NO7, fO-Sm_NO8)')
    mh.makehist_fromnpz(NO7_name_16_28, NO8_name_16_28, fO_Sm_NO7_name, fO_Sm_NO8_name,\
                    bins = [edgesNO7, edgesNO8, edgesfO_NO7, edgesfO_NO8],\
                    dimlabels = ['NO7', 'NO8','fO-Sm_NO7', 'fO-Sm_NO8'],\
                    save=mh.pdir +'coldens_o7-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Sm-Oxygen',\
                    fills = fillsea)  

## O6 phase diagram low z
rho_NNe8_name_16_27 =   'Density_T4EOS_coldens_ne8_PtAb_T4EOS_L0100N1504_28_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NNe8_name_16_27 = 'Temperature_T4EOS_coldens_ne8_PtAb_T4EOS_L0100N1504_28_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NNe8_name_16_27 = 'coldens_ne8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNNe8 = np.array(list((np.arange(5)/4.*(0.0+40.) -40.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(71)/70.*(17.-10.) +10.))
edgesT_NNe8 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesrho_NNe8 = np.arange(101)/100.*(-23.0+33.0) - 33.0

if jobind == 100:
    print('Doing EA100, 16 slices, (NNe8, rho_NNe8, T_NNe8)')
    mh.makehist_fromnpz(NNe8_name_16_27, rho_NNe8_name_16_27, T_NNe8_name_16_27,\
                    bins = [edgesNNe8, edgesrho_NNe8, edgesT_NNe8],\
                    dimlabels = ['NNe8', 'Density_w_NNe8','Temperature_w_NNe8'],\
                    save=mh.pdir +'hist_coldens_ne8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature',\
                    fills = fillsea)  

## Ne8 comparison to O6, O7, O8
rho_NNe8_name_16_28 =   'Density_T4EOS_coldens_ne8_PtAb_T4EOS_L0100N1504_28_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NNe8_name_16_28 = 'Temperature_T4EOS_coldens_ne8_PtAb_T4EOS_L0100N1504_28_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NNe8_name_16_28 = 'coldens_ne8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNNe8 = np.array(list((np.arange(5)/4.*(0.0+40.) -40.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(71)/70.*(17.-10.) +10.))
edgesT_NNe8 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesrho_NNe8 = np.arange(101)/100.*(-23.0+33.0) - 33.0

edgesNO7 = np.array(list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
edgesT_NO7 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho_NO7 = np.array(list((np.arange(5)/4.*(-36.+76.) -76.0)[:-1]) + list((np.arange(5)/4.*(-32.0+36.) -36.0)[:-1]) + list(np.arange(88)/87.*(-23.3+32.0) - 32.0))

rho_NO7_name_16_28 =   'Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO7_name_16_28 = 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO7_name_16_28 = 'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO6 = np.array(list((np.arange(10)/9.*(9.0+36.0) -36.0)[:-1]) + list((np.arange(3)/2.*(11.0-9.0) +9.0)[:-1]) + list(np.arange(57)/56.*(16.6-11.0) + 11.0))
edgesT_NO6 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho_NO6 = np.array(list((np.arange(5)/4.*(-36.+76.) -76.0)[:-1]) + list((np.arange(5)/4.*(-32.0+36.) -36.0)[:-1]) + list(np.arange(90)/89.*(-23.1+32.0) - 32.0))

rho_NO6_name_16_28 =   'Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO6_name_16_28 = 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO6_name_16_28   = 'coldens_o6_L0100N1504_28_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

edgesNO8 = np.array(list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0))
edgesT_NO8 = np.array(list((np.arange(3)/2.*(1.0+45.) -45.)[:-1]) + list((np.arange(4)/3.*(2.5-1.0) +1.0)[:-1]) + list(np.arange(61)/60.*(8.5-2.5) +2.5))
edgesrho_NO8 = np.arange(97)/96.*(-22.9+32.5) - 32.5

rho_NO8_name_16_28 =   'Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO8_name_16_28 = 'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
NO8_name_16_28 = 'coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

if jobind == 101:
    print('Doing EA100, 16 slices, (NNe8, NO7, T_NNe8, T_NO7)')
    mh.makehist_fromnpz(NNe8_name_16_28, NO7_name_16_28, T_NNe8_name_16_28, T_NO7_name_16_28,\
                    bins = [edgesNNe8, edgesNO7, edgesT_NNe8, edgesT_NO7],\
                    dimlabels = ['NNe8', 'NO7','Temperature_w_NNe8', 'Temperature_w_NO7'],\
                    save=mh.pdir +'hist_coldens_ne8-o7_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)
if jobind == 102:
    print('Doing EA100, 16 slices, (NNe8, NO7, rho_NNe8, rho_NO7)')
    mh.makehist_fromnpz(NNe8_name_16_28, NO7_name_16_28, rho_NNe8_name_16_28, rho_NO7_name_16_28,\
                    bins = [edgesNNe8, edgesNO7, edgesrho_NNe8, edgesrho_NO7],\
                    dimlabels = ['NNe8', 'NO7','Density_w_NNe8', 'Density_w_NO7'],\
                    save=mh.pdir +'hist_coldens_ne8-o7_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density',\
                    fills = fillsea)
if jobind == 103:
    print('Doing EA100, 16 slices, (NNe8, NO8, T_NNe8, T_NO8)')
    mh.makehist_fromnpz(NNe8_name_16_28, NO8_name_16_28, T_NNe8_name_16_28, T_NO8_name_16_28,\
                    bins = [edgesNNe8, edgesNO8, edgesT_NNe8, edgesT_NO8],\
                    dimlabels = ['NNe8', 'NO8','Temperature_w_NNe8', 'Temperature_w_NO8'],\
                    save=mh.pdir +'hist_coldens_ne8-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)
if jobind == 104:
    print('Doing EA100, 16 slices, (NNe8, NO8, rho_NNe8, rho_NO8)')
    mh.makehist_fromnpz(NNe8_name_16_28, NO8_name_16_28, rho_NNe8_name_16_28, rho_NO8_name_16_28,\
                    bins = [edgesNNe8, edgesNO8, edgesrho_NNe8, edgesrho_NO8],\
                    dimlabels = ['NNe8', 'NO8','Density_w_NNe8', 'Density_w_NO8'],\
                    save=mh.pdir +'hist_coldens_ne8-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density',\
                    fills = fillsea)
if jobind == 105:
    print('Doing EA100, 16 slices, (NNe8, NO6, T_NNe8, T_NO6)')
    mh.makehist_fromnpz(NNe8_name_16_28, NO6_name_16_28, T_NNe8_name_16_28, T_NO6_name_16_28,\
                    bins = [edgesNNe8, edgesNO6, edgesT_NNe8, edgesT_NO6],\
                    dimlabels = ['NNe8', 'NO6','Temperature_w_NNe8', 'Temperature_w_NO6'],\
                    save=mh.pdir +'hist_coldens_ne8-o6_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)
if jobind == 106:
    print('Doing EA100, 16 slices, (NNe8, NO6, rho_NNe8, rho_NO6)')
    mh.makehist_fromnpz(NNe8_name_16_28, NO6_name_16_28, rho_NNe8_name_16_28, rho_NO6_name_16_28,\
                    bins = [edgesNNe8, edgesNO6, edgesrho_NNe8, edgesrho_NO6],\
                    dimlabels = ['NNe8', 'NO6','Density_w_NNe8', 'Density_w_NO6'],\
                    save=mh.pdir +'hist_coldens_ne8-o6_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density',\
                    fills = fillsea)

edgesNO7 = np.array([-np.inf] + list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
NO7_name_16_28 = 'coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
edgesNO8 = np.array([-np.inf] + list((np.arange(35)/34.*(11.0+23.0) -23.0)[:-1]) + list(np.arange(67)/66.*(17.6-11.0) + 11.0))
NO8_name_16_28 = 'coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
edges_Hneutral = np.array(list(np.arange(11)/10.*(14. - 9.) + 9.)[:-1] + list(np.arange(91)/90. * (23. - 14.) + 14.))
Hneutral_name_16_28 = 'coldens_hneutralssh_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

if jobind == 107:
    print('Doing Hneutral, O7, O8')
    mh.makehist_fromnpz(Hneutral_name_16_28, NO7_name_16_28, NO8_name_16_28,\
                    bins = [edges_Hneutral, edgesNO7, edgesNO8],\
                    dimlabels = ['NHneutralssh', 'NO7','NO8'],\
                    save=mh.pdir +'hist_coldens_hneutralssh-o7-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_inclzeros',\
                    fills = fillsea, includeinf=True)

edgesNO7 = np.array([-np.inf] + list((np.arange(32)/31.*(8.0+23.0) -23.0)[:-1]) + list(np.arange(99)/98.*(17.8-8.0) + 8.0)) 
NO7_name_16_27 = 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
edgesNO8 = np.array([-np.inf] + list((np.arange(34)/33.*(10.0+23.0) -23.0)[:-1]) + list(np.arange(77)/76.*(17.6-10.0) + 10.0))
NO8_name_16_27 = 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
edges_Hneutral = np.array(list(np.arange(4)/3.*(11. - 8.) + 8.)[:-1] + list(np.arange(121)/120. * (23. - 11.) + 11.))
Hneutral_name_16_27 = 'coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

if jobind == 108:
    print('Doing Hneutral, O7, O8')
    mh.makehist_fromnpz(Hneutral_name_16_27, NO7_name_16_27, NO8_name_16_27,\
                    bins = [edges_Hneutral, edgesNO7, edgesNO8],\
                    dimlabels = ['NHneutralssh', 'NO7','NO8'],\
                    save=mh.pdir +'hist_coldens_hneutralssh-o7-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_inclzeros',\
                    fills = fillsea, includeinf=True)
    

NO6_name_16_27 = 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' 
NO7_name_16_27 = 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NO8_name_16_27 = 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NNe9_name_16_27 = 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
T_NO6_name_16_27 = 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
T_NO8_name_16_27 = 'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'

mu_logNo6 = 13.263 
sigma_logNo6 = 0.11

mu_logNo8_slab = 15.5 
sigma_logNo8_slab = 0.2 
mu_logNne9_slab = 15.4
sigma_logNne9_slab_up = 0.1
sigma_logNne9_slab_down = 0.2
ul_logNo7_slab = 15.5

mu_logNo8_cie = 15.4
sigma_logNo8_cie = 0.2
mu_logNne9_cie = 14.9
sigma_logNne9_cie = 0.2
mu_logNo7_cie = 14.8
sigma_logNo7_cie = 0.2
    
edgesNO6  = np.array([-40.0, 12.] + list(np.arange(16) / 15. * (14.0 - 12.5) + 12.5) + [14.5, 15.0, 17.]) # focus on region of detection: 13.263 +- 0.11
edgesNO7  = np.array([-23.0, 12.] + list(np.arange(16) / 15. * (16.0 - 14.5) + 14.5) + [18.])
edgesNO8  = np.array([-23.0, 12.] + list(np.arange(22) / 21. * (16.5 - 14.4) + 14.4) + [17.3])
edgesNNe9 = np.array([-23.0, 12.] + list(np.arange(25) / 24. * (16.3 - 13.9) + 13.9) + [16.9])
edgesT_NO6 = np.array([-50.] + list((np.arange(6)/5.*(2.5-0.0) + 0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesT_NO8 = np.array([-50.] + list((np.arange(6)/5.*(2.5-0.0) + 0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))

if jobind == 109:
    print('Doing EA100, 16 slices, (NO6, NO7, NO8, NNe9, T_NO6, T_NO8)')
    mh.makehist_fromnpz(NO6_name_16_27, NO7_name_16_27, NO8_name_16_27, NNe9_name_16_27, T_NO6_name_16_27, T_NO8_name_16_27,\
                    bins = [edgesNO6, edgesNO7, edgesNO8, edgesNNe9, edgesT_NO6, edgesT_NO8],\
                    dimlabels = ['NO6','NO7','NO8', 'NNe9', 'Temperature_w_NO6', 'Temperature_w_NO8'],\
                    save=mh.pdir + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_o6-o8_weighted_Temperatures',\
                    fills = fillsea, includeinf=True)      

# O6 phase diagram ased on N, Z, and X-ray counterpart column densities
NO6_name_16_27 = 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' 
NO7_name_16_27 = 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NO8_name_16_27 = 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
NNe9_name_16_27 = 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
T_NO6_name_16_27 = 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
rho_NO6_name_16_27 = 'Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
fOSm_NO6_name_16_27 = 'SmoothedElementAbundance-Oxygen_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'

edgesNO6  = np.array([-40.0, 12., 12.5, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 14.0, 14.5, 15.0, 15.5, 16., 16.5, 17.]) # focus on region of detection: 13.263 +- 0.11
edgesNO7  = np.array([14.4, 14.6, 14.8, 15.0, 15.2, 15.5])
edgesNO8  = np.array([15.0, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9])
edgesNNe9 = np.array([14.5, 14.7, 14.9, 15.1, 15.2, 15.3, 15.4, 15.6, 15.8])
edgesT_NO6 = np.array(list((np.arange(6)/5.*(2.5-0.0) + 0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
edgesrho_NO6 = np.arange(102)/101.*(-22.2 + 32.3) - 32.3 #-32.281895, -22.323387
logfOsol = np.log10(ol.solar_abunds['oxygen'])
edgesfO_NO6 = np.array([-8.2, -6.2, -5.2][:-1] + list(np.arange(20)/19.*(-1.4 + 5.2) - 5.2)) # (-76.52521, -1.4523125), some margin around edges, then get values that are nice (to 0.1 decimal) in solar units
edgesfO_NO6 = np.round(edgesfO_NO6 - logfOsol, 1) + logfOsol

if jobind == 110:
    print('Doing EA100, 16 slices, (NO6, NO7, NO8, NNe9, rho_NO6, T_NO6, fOSm_NO6)')
    mh.makehist_fromnpz(NO6_name_16_27, NO7_name_16_27, NO8_name_16_27, NNe9_name_16_27, rho_NO6_name_16_27, T_NO6_name_16_27, fOSm_NO6_name_16_27,\
                    bins = [edgesNO6, edgesNO7, edgesNO8, edgesNNe9, edgesrho_NO6, edgesT_NO6, edgesfO_NO8],\
                    dimlabels = ['NO6','NO7','NO8', 'NNe9', 'Density_w_NO6', 'Temperature_w_NO6', 'SmoothedElementAbundance-Oxygen_w_NO6'],\
                    save=mh.pdir + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_o6_weighted_Temperature_Density_fOSm',\
                    fills = fillsea, includeinf=True)
    
# Z weighted by different stuff comparisons
fills_25_4 = [str(i) for i in np.arange(4)/4. * 25. + 25./8.]
name_mass   = ol.ndir_old + 'Mass_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
name_Z_mass = ol.ndir_old + 'Metallicity_T4EOS_Mass_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
name_h1     = ol.ndir_old + 'coldens_h1ssh_L0025N0376_19_test3.31_PtAb_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
name_Z_h1   = ol.ndir_old + 'Metallicity_T4EOS_coldens_h1ssh_PtAb_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'
name_sfr    = ol.ndir_old + 'StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
name_Z_sfr  = ol.ndir_old + 'Metallicity_T4EOS_StarFormationRate_T4EOS_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen%s_z-projection.npz'

edges_mass   = np.arange(-5.8, 0.15, 0.1)
edges_sfr    = np.array([-42., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
edges_h1     = np.arange(11.1, 22.35, 0.1)
edges_Z      = np.array([-40., -10., -8., -6.] + list(np.arange(-5., -0.95, 0.1)))

if jobind == 111:
    print('Doing EA25, 4 slices, (Mass, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_mass, name_Z_mass, name_Z_h1, name_Z_sfr,\
                    bins = [edges_mass, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['Mass', 'Metallicity_w_Mass', 'Metallicity_w_NH1', 'Metallicity_w_SFR'],\
                    save=mh.pdir + 'Mass_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass_coldens-h1ssh_StarFormationRate',\
                    fills = fills_25_4, includeinf=True)
    
elif jobind == 112:
    print('Doing EA25, 4 slices, (Nh1, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_h1, name_Z_mass, name_Z_h1, name_Z_sfr,\
                    bins = [edges_h1, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['NH1', 'Metallicity_w_Mass', 'Metallicity_w_NH1', 'Metallicity_w_SFR'],\
                    save=mh.pdir + 'coldens_h1ssh_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass_coldens-h1ssh_StarFormationRate',\
                    fills = fills_25_4, includeinf=True)

elif jobind == 113:
    print('Doing EA25, 4 slices, (SFR, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_sfr, name_Z_mass, name_Z_h1, name_Z_sfr,\
                    bins = [edges_sfr, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['StarFormationRate', 'Metallicity_w_Mass', 'Metallicity_w_NH1', 'Metallicity_w_SFR'],\
                    save=mh.pdir + 'StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass_coldens-h1ssh_StarFormationRate',\
                    fills = fills_25_4, includeinf=True)

elif jobind == 114:
    mh.makehistograms_Zdiff(kind='h1')
elif jobind == 115:
    mh.makehistograms_Zdiff(kind='SFR')
elif jobind == 116:
    mh.makehistograms_Zdiff(kind='mass')

elif jobind == 117:
    print('Doing EA25, 4 slices, (Mass, Nh1, SFR, Z-Mass)')
    mh.makehist_fromnpz(name_mass, name_h1, name_sfr, name_Z_mass,\
                    bins = [edges_mass, edges_h1, edges_sfr, edges_Z],\
                    dimlabels = ['Mass', 'NH1', 'StarFormationRate', 'Metallicity_w_Mass'],\
                    save=mh.pdir + 'Mass_coldens_h1ssh_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass',\
                    fills = fills_25_4, includeinf=True)

#### comparing Ref and noAGN o7-o8
Nbase = 'coldens_%s_L0050N0752%s_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
Tbase = 'Temperature_T4EOS_coldens_%s_PtAb_T4EOS_L0050N0752%s_28_test3.31_C2Sm_16000pix_6.25slice_zcen%s_z-projection.npz'
rhobase = 'Density_wiEOS_coldens_%s_PtAb_T4EOS_L0050N0752%s_28_test3.31_C2Sm_16000pix_6.25slice_zcen%s_z-projection.npz'
zbase = 'SmoothedElementAbundance-Oxygen_wiEOS_coldens_%s_PtAb_T4EOS_L0050N0752%s_28_test3.31_C2Sm_16000pix_6.25slice_zcen%s_z-projection.npz'

edgesN = np.array([-23., 10., 11., 12.] + list(np.arange(12.1, 17.95, 0.1)))
edgesT = np.array(list((np.arange(6)/5.*(2.5-0.0) + 0.0)[:-1]) + list(np.arange(56)/55.*(8.0-2.5) + 2.5))
edgesrho = np.arange(107)/106.*(-21.8 + 32.4) - 32.4 #-32.281895, -22.323387
logfOsol = np.log10(ol.solar_abunds['oxygen'])
edgesfO = np.array([-8.2, -6.2, -5.2][:-1] + list(np.arange(42)/41.*(-1.1 + 5.2) - 5.2)) # (-76.52521, -1.4523125), some margin around edges, then get values that are nice (to 0.1 decimal) in solar units
edgesfO = np.round(edgesfO - logfOsol, 1) + logfOsol

fills50 = [str(i) for i in np.arange(8)/8.*50. + 50./16.]
ions   = ['o7', 'o8']
eavars = ['', 'EagleVariation_NoAGN']

if jobind == 118:
    for ion in ions:
        for var in eavars:
            print('Doing EA50, 8 slices, (N, rho, T, fOSm) for %s, %s'%(ion, var))
            names = tuple([name%(ion, var, '%s') for name in [Nbase, rhobase, Tbase, zbase]])
            mh.makehist_fromnpz(*names,\
                    bins = [edgesN, edgesrho, edgesT, edgesfO],\
                    dimlabels = ['N%s'%(string.capwords(ion)), 'Density_w_N%s'%(string.capwords(ion)),'Temperature_w_N%s'%(string.capwords(ion)), 'fO-Sm_w_N%s'%(string.capwords(ion))],\
                    save=mh.pdir + 'coldens_%s_L0050N0752%s_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen%s_z-projection_T4EOS_and_weighted_Density_Temperature_SmoothedElementAbundance-Oxygen'%(ion, var, '-all'),\
                    fills = fills50, includeinf=True)

if jobind == 119:
    mh.makehistograms_Zdiff(kind='h1', simset='L0025N0376_19_SmZ')
elif jobind == 120:
    mh.makehistograms_Zdiff(kind='SFR', simset='L0025N0376_19_SmZ')
elif jobind == 121:
    mh.makehistograms_Zdiff(kind='mass', simset='L0025N0376_19_SmZ')
elif jobind == 122:
    mh.makehistograms_Zdiff(kind='h1', simset='L0025N0376_19_SmZ_hn')
elif jobind == 123:
    mh.makehistograms_Zdiff(kind='SFR', simset='L0025N0376_19_SmZ_hn')
elif jobind == 124:
    mh.makehistograms_Zdiff(kind='mass', simset='L0025N0376_19_SmZ_hn')
elif jobind == 125:
    mh.makehistograms_Zdiff(kind='h1', simset='L0025N0752Recal_19_SmZ')
elif jobind == 126:
    mh.makehistograms_Zdiff(kind='SFR', simset='L0025N0752Recal_19_SmZ')
elif jobind == 127:
    mh.makehistograms_Zdiff(kind='mass', simset='L0025N0752Recal_19_SmZ')
elif jobind == 128:
    mh.makehistograms_Zdiff(kind='h1', simset='L0025N0752Recal_19_SmZ_hn')
elif jobind == 129:
    mh.makehistograms_Zdiff(kind='SFR', simset='L0025N0752Recal_19_SmZ_hn')
elif jobind == 130:
    mh.makehistograms_Zdiff(kind='mass', simset='L0025N0752Recal_19_SmZ_hn')

fills = [str(i) for i in np.arange(8)/8. * 25. + 25./16.]
name_mass   = 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
name_Z_mass = 'SmoothedMetallicity_T4EOS_Mass_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
name_h1     = 'coldens_h1ssh_L0025N0752RECALIBRATED_19_test3.31_PtAb_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
name_Z_h1   = 'SmoothedMetallicity_T4EOS_coldens_h1ssh_PtAb_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'
name_sfr    = 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
name_Z_sfr  = 'SmoothedMetallicity_T4EOS_StarFormationRate_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'

edges_mass   = np.array([-np.inf] + list(np.arange(-6.3, 0.15, 0.1)))
edges_sfr    = np.array([-np.inf, -46., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
edges_h1     = np.array([-np.inf] + list(np.arange(10.3, 23.35, 0.1)))
edges_Z      = np.array([-np.inf, -52., -40., -10., -8., -6.] + list(np.arange(-5., -0.75, 0.1)))

if jobind == 131:
    print('Doing EA25-Recal, 8 slices, (Mass, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_mass, name_Z_mass, name_Z_h1, name_Z_sfr,\
                    bins = [edges_mass, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['Mass', 'SmoothedMetallicity_w_Mass', 'SmoothedMetallicity_w_NH1', 'SmoothedMetallicity_w_SFR'],\
                    save=mh.pdir + 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-h1ssh_StarFormationRate',\
                    fills = fills, includeinf=True)
    
elif jobind == 132:
    print('Doing EA25-Recal, 8 slices, (Nh1, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_h1, name_Z_mass, name_Z_h1, name_Z_sfr,\
                    bins = [edges_h1, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['NH1', 'SmoothedMetallicity_w_Mass', 'SmoothedMetallicity_w_NH1', 'SmoothedMetallicity_w_SFR'],\
                    save=mh.pdir + 'coldens_h1ssh_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-h1ssh_StarFormationRate',\
                    fills = fills, includeinf=True)

elif jobind == 133:
    print('Doing EA25-Recal, 8 slices, (SFR, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_sfr, name_Z_mass, name_Z_h1, name_Z_sfr,\
                    bins = [edges_sfr, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['StarFormationRate', 'SmoothedMetallicity_w_Mass', 'SmoothedMetallicity_w_NH1', 'SmoothedMetallicity_w_SFR'],\
                    save=mh.pdir + 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-h1ssh_StarFormationRate',\
                    fills = fills, includeinf=True)

name_hn     = 'coldens_hneutralssh_L0025N0752RECALIBRATED_19_test3.31_PtAb_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
name_Z_hn   = 'SmoothedMetallicity_T4EOS_coldens_hneutralssh_PtAb_T4EOS_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection.npz'

if jobind == 134:
    print('Doing EA25-Recal, 8 slices, (Mass, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_mass, name_Z_mass, name_Z_hn, name_Z_sfr,\
                    bins = [edges_mass, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['Mass', 'SmoothedMetallicity_w_Mass', 'SmoothedMetallicity_w_NH1', 'SmoothedMetallicity_w_SFR'],\
                    save=mh.pdir + 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-hneutralssh_StarFormationRate',\
                    fills = fills, includeinf=True)
    
elif jobind == 135:
    print('Doing EA25-Recal, 8 slices, (Nh1, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_hn, name_Z_mass, name_Z_hn, name_Z_sfr,\
                    bins = [edges_h1, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['NH1', 'SmoothedMetallicity_w_Mass', 'SmoothedMetallicity_w_NH1', 'SmoothedMetallicity_w_SFR'],\
                    save=mh.pdir + 'coldens_hneutralssh_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-hneutralssh_StarFormationRate',\
                    fills = fills, includeinf=True)

elif jobind == 136:
    print('Doing EA25-Recal, 8 slices, (SFR, Z-Mass, Z-h1, Z-sfr)')
    mh.makehist_fromnpz(name_sfr, name_Z_mass, name_Z_hn, name_Z_sfr,\
                    bins = [edges_sfr, edges_Z, edges_Z, edges_Z],\
                    dimlabels = ['StarFormationRate', 'SmoothedMetallicity_w_Mass', 'SmoothedMetallicity_w_NH1', 'SmoothedMetallicity_w_SFR'],\
                    save=mh.pdir + 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-hneutralssh_StarFormationRate',\
                    fills = fills, includeinf=True)

### for CDDFs in different regions around groups (w/ Sean Johnson)
fills = ['12.5', '37.5', '62.5', '87.5']
filename_23 = 'coldens_o7_L0100N1504_23_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen%s_z-projection_T4EOS.npz'
filename_24 = 'coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen%s_z-projection_T4EOS.npz'
vsels_23 = {'12.5': 'VZ-0.0-1484.9420864572726',\
            '37.5': 'VZ-1484.9420864572726-2969.884172914545',\
            '62.5': 'VZ-2969.884172914545-4454.826259371817',\
            '87.5': 'VZ-4454.826259371817-5939.76834582909'}
vsels_24 = {'12.5': 'VZ-0.0-1506.6750627451243',\
            '37.5': 'VZ-1506.6750627451243-3013.3501254902485',\
            '62.5': 'VZ-3013.3501254902485-4520.025188235373',\
            '87.5': 'VZ-4520.025188235373-6026.700250980497'}
psels = {'12.5': 'Z-0.0-25.0-0.0',\
         '37.5': 'Z-25.0-50.0-0.0',\
         '62.5': 'Z-50.0-75.0-0.0',\
         '87.5': 'Z-75.0-100.0-0.0'}
sizes = [100., 200., 500., 1000., 10000.]

base_23 = 'mask_RefL0100N1504_23_32000pix_z-projection_totalbox_halosize-%s-pkpc_closest-normradius_selection-in_M200c_Msun-5e+13-None_%s.hdf5'
base_24 = 'mask_RefL0100N1504_24_32000pix_z-projection_totalbox_halosize-%s-pkpc_closest-normradius_selection-in_M200c_Msun-5e+13-None_%s.hdf5'

masks_23 = {key: [None] + [base_23%(size, psels[key]) for size in sizes] + [base_23%(size, vsels_23[key]) for size in sizes] for key in vsels_23.keys()}
masks_24 = {key: [None] + [base_24%(size, psels[key]) for size in sizes] + [base_24%(size, vsels_24[key]) for size in sizes] for key in vsels_24.keys()}

masknames = ['nomask',\
             '100pkpc_pos_M200c-geq-5e13-Msun',\
             '200pkpc_pos_M200c-geq-5e13-Msun',\
             '500pkpc_pos_M200c-geq-5e13-Msun',\
             '1000pkpc_pos_M200c-geq-5e13-Msun',\
             '10000pkpc_pos_M200c-geq-5e13-Msun',\
             '100pkpc_vel_M200c-geq-5e13-Msun',\
             '200pkpc_vel_M200c-geq-5e13-Msun',\
             '500pkpc_vel_M200c-geq-5e13-Msun',\
             '1000pkpc_vel_M200c-geq-5e13-Msun',\
             '10000pkpc_vel_M200c-geq-5e13-Msun',\
             ]

if jobind == 137:
    mh.makehist_masked_toh5py(filename_23, fills=fills, maskfiles=masks_23, masknames=masknames, includeinf=True, bins=[np.arange(-25., 28.05, 0.05)], outfilename='cddf_' + (filename_23%('-all'))[:-4] + '_masks_M200c-geq-5e13-Msun.hdf5')
elif jobind == 138:
    mh.makehist_masked_toh5py(filename_24, fills=fills, maskfiles=masks_24, masknames=masknames, includeinf=True, bins=[np.arange(-25., 28.05, 0.05)], outfilename='cddf_' + (filename_24%('-all'))[:-4] + '_masks_M200c-geq-5e13-Msun.hdf5')

fills = [str(i) for i in np.arange(16)/16.*100. + 100./32.]
includeinf = True
rho_to_nh = 0.752/(c.atomw_H * c.u)

filenames_o6 = [ol.ndir_old + 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                ol.ndir_old + 'Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ol.ndir_old + 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ]
outfilename_o6 = 'hist_coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5'
filenames_o7 = [ol.ndir_old + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                ol.ndir_old + 'Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_27_test3.11_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ol.ndir_old + 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_27_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ]
outfilename_o7 = 'hist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5'
filenames_o8 = [ol.ndir + 'coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                ol.ndir + 'Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.4_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ol.ndir_old + 'Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ]
outfilename_o8 = 'coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5'
filenames_ne9 = [ol.ndir_old + 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                 ol.ndir_old + 'Density_T4EOS_coldens_ne9_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                 ol.ndir_old + 'Temperature_T4EOS_coldens_ne9_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz',\
                ]
outfilename_ne9 = 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_temperature_density.hdf5'
# extreme edges same as those in snap 27 gas histograms
Tbins = np.array([-np.inf] + list(np.arange(2.5, 9.55, 0.05)) + [np.inf])
rhobins = np.array([-np.inf, -9.5] + list(np.arange(-9.0, 2.70, 0.05)) + [np.inf]) - np.log10(rho_to_nh)
Nbins_o6 = np.array([-np.inf, 11.0, 11.5, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14.0, 14.5, 15.0, 15.5, np.inf])
Nbins_o7 = np.array([-np.inf, 12.0, 12.5, 13.0, 13.5, 14.0, 14.3, 14.5, 14.6, 14.7, 14.8, 14.9, 15., 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16., 16.5, np.inf])
Nbins_o8 = np.array([-np.inf, 12.0, 12.5, 13.0, 13.5, 14.0, 14.3, 14.5, 14.6, 14.7, 14.8, 14.9, 15., 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16., 16.5, np.inf])
Nbins_ne9 = np.array([-np.inf, 12.0, 12.5, 13.0, 13.5, 14.0, 14.1, 14.2, 14.3, 14.5, 14.6, 14.7, 14.8, 14.9, 15., 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16., 16.1, 16.2, 16.3, 16.4, 16.5, 17.0, np.inf])
if jobind == 139:
    mh.makehist_masked_toh5py(*tuple(filenames_o6), bins=[Nbins_o6, rhobins, Tbins], includeinf=includeinf, fills=fills, outfilename=outfilename_o6)
elif jobind == 140:
    mh.makehist_masked_toh5py(*tuple(filenames_o7), bins=[Nbins_o7, rhobins, Tbins], includeinf=includeinf, fills=fills, outfilename=outfilename_o7)
elif jobind == 141:
    mh.makehist_masked_toh5py(*tuple(filenames_o8), bins=[Nbins_o8, rhobins, Tbins], includeinf=includeinf, fills=fills, outfilename=outfilename_o8)
elif jobind == 142:
    mh.makehist_masked_toh5py(*tuple(filenames_ne9), bins=[Nbins_ne9, rhobins, Tbins], includeinf=includeinf, fills=fills, outfilename=outfilename_ne9)   
    
    
### for CDDFs in different isolation regions (w/ Sean Johnson)
fills = ['12.5', '37.5', '62.5', '87.5']
filename_24 = 'coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen%s_z-projection_T4EOS.npz'
vsels_24 = {'12.5': 'VZ-0.0-1506.6750627451243',\
            '37.5': 'VZ-1506.6750627451243-3013.3501254902485',\
            '62.5': 'VZ-3013.3501254902485-4520.025188235373',\
            '87.5': 'VZ-4520.025188235373-6026.700250980497'}
psels = {'12.5': 'Z-0.0-25.0-0.0',\
         '37.5': 'Z-25.0-50.0-0.0',\
         '62.5': 'Z-50.0-75.0-0.0',\
         '87.5': 'Z-75.0-100.0-0.0'}
size_mass = [('630', 'Mstar_Msun-5011872336.27-None'), ('2273', 'Mstar_Msun-1.58489319246e+11-None')]

base_24 = 'mask_RefL0100N1504_24_32000pix_z-projection_totalbox_inclsatellites_halosize-%s-pkpc_closest-normradius_selection-in_%s_%s.hdf5'

masks_24 = {key: [None] + [base_24%(szms + (psels[key],)) for szms in size_mass] + [base_24%(szms + (vsels_24[key],)) for szms in size_mass] for key in vsels_24.keys()}

masknames = ['nomask',\
             '630pkpc_pos_logMstar-geq-9.7-Msun',\
             '2273pkpc_pos_logMstar-geq-11.2-Msun',\
             '630pkpc_vel_logMstar-geq-9.7-Msun',\
             '2273pkpc_vel_logMstar-geq-11.2-Msun',\
             ]

if jobind == 143:
    mh.makehist_masked_toh5py(filename_24, fills=fills, maskfiles=masks_24, masknames=masknames, includeinf=True, bins=[np.arange(-25., 28.05, 0.05)], outfilename='cddf_' + (filename_24%('-all'))[:-4] + '_masks_Mstar-9.7-11.2_isolation.hdf5')

halocat =  '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5'
if jobind == 144:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**9.7, None)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-9.7_posspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=1500.,\
                     axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 145:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**11.2, None)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-11.2_posspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=3500.,\
                     axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 146:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**9.7, None)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-9.7_velspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=1500.,\
                     axis='z', velspace=True, offset_los=0., stamps=False)
elif jobind == 147:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**11.2, None)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-11.2_velspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=3500.,\
                     axis='z', velspace=True, offset_los=0., stamps=False)

elif jobind == 148:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**9.6, 10**9.8)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-9.6-9.8_posspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=3500.,\
                     axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 149:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**11.1, 10**11.3)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-11.1-11.3_posspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=3500.,\
                     axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 150:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**9.6, 10**9.8)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-9.6-9.8_velspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=3500.,\
                     axis='z', velspace=True, offset_los=0., stamps=False)
elif jobind == 151:
    crd.rdists_sl_from_selection(filename_24, fills, 100., 32000,\
                     0., 1.,\
                     halocat,\
                     [('Mstar_Msun', 10**11.1, 10**11.3)], None, outname='coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-11.1-11.3_velspace',\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=3500.,\
                     axis='z', velspace=True, offset_los=0., stamps=False)

fills = ['12.5', '37.5', '62.5', '87.5']
filename_24 = 'coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen%s_z-projection_T4EOS.npz'
vsels_24 = {'12.5': 'VZ-0.0-1506.6750627451243',\
            '37.5': 'VZ-1506.6750627451243-3013.3501254902485',\
            '62.5': 'VZ-3013.3501254902485-4520.025188235373',\
            '87.5': 'VZ-4520.025188235373-6026.700250980497'}
psels = {'12.5': 'Z-0.0-25.0-0.0',\
         '37.5': 'Z-25.0-50.0-0.0',\
         '62.5': 'Z-50.0-75.0-0.0',\
         '87.5': 'Z-75.0-100.0-0.0'}
size_mass = [('630', 'Mstar_Msun-1995262314.97-None'), ('2273', 'Mstar_Msun-1e+11-None')]

base_24 = 'mask_RefL0100N1504_24_32000pix_z-projection_totalbox_inclsatellites_halosize-%s-pkpc_closest-normradius_selection-in_%s_%s.hdf5'

masks_24 = {key: [None] + [base_24%(szms + (psels[key],)) for szms in size_mass] + [base_24%(szms + (vsels_24[key],)) for szms in size_mass] for key in vsels_24.keys()}

masknames = ['nomask',\
             '630pkpc_pos_logMstar-geq-9.3-Msun',\
             '2273pkpc_pos_logMstar-geq-11.0-Msun',\
             '630pkpc_vel_logMstar-geq-9.3-Msun',\
             '2273pkpc_vel_logMstar-geq-11.0-Msun',\
             ]

if jobind == 152:
    mh.makehist_masked_toh5py(filename_24, fills=fills, maskfiles=masks_24, masknames=masknames, includeinf=True, bins=[np.arange(-25., 28.05, 0.05)], outfilename='cddf_' + (filename_24%('-all'))[:-4] + '_masks_Mstar-9.3-11.0_isolation.hdf5')

halocat =  '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5'
yvals_fcov = [14., 14.5, 15., 15.1, 15.2, 15.3, 15.4, 15.5, 16., 16.5]
rbins = [0., 10., 20., 30., 40., 50., 70., 100., 150.] + list(np.arange(200., 3550., 100.))
rbins2 = [0., 10., 20., 30., 40., 50., 70., 100., 150.] + list(np.arange(200., 1550., 100.))
xunit = 'pkpc'
galsettag = 'all'

if jobind == 153:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-9.6-9.8_posspace'
    crd.get_radprof(rqfilename, halocat, rbins, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
elif jobind == 154:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-9.6-9.8_velspace'
    crd.get_radprof(rqfilename, halocat, rbins, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)

elif jobind == 155:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-11.1-11.3_posspace'
    crd.get_radprof(rqfilename, halocat, rbins, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
elif jobind == 156:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-11.1-11.3_velspace'
    crd.get_radprof(rqfilename, halocat, rbins, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
elif jobind == 157:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-9.7_posspace'
    crd.get_radprof(rqfilename, halocat, rbins2, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
elif jobind == 158:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-9.7_velspace'
    crd.get_radprof(rqfilename, halocat, rbins2, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
elif jobind == 159:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-11.2_posspace'
    crd.get_radprof(rqfilename, halocat, rbins, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
elif jobind == 160:
    rqfilename = 'coldens_rdist_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_25.0slice_zcen-all_z-projection_T4EOS_1sl_logMstar-Msun-ge-11.2_velspace'
    crd.get_radprof(rqfilename, halocat, rbins, yvals_fcov,\
                    xunit=xunit, ytype='fcov',\
                    galids=None, combinedprofile=True,\
                    separateprofiles=False,\
                    rpfilename=None, galsettag=galsettag)
 
    
fills = ['16.6666666667', '50.0', '83.3333333333']
filename_23 = 'coldens_o7_L0100N1504_23_test3.4_PtAb_C2Sm_32000pix_33.3333333333slice_zcen%s_z-projection_T4EOS.npz'
filename_24 = 'coldens_o7_L0100N1504_24_test3.31_PtAb_C2Sm_32000pix_33.3333333333slice_zcen%s_z-projection_T4EOS.npz'

vsels_23 = {'16.6666666667': 'VZ-0.0-1979.92278194303',\
            '50.0':          'VZ-1979.92278194303-3959.84556388606',\
            '83.3333333333': 'VZ-3959.84556388606-5939.76834582909',\
            }
vsels_24 = {'16.6666666667': 'VZ-0.0-2008.9000836601656',\
            '50.0':          'VZ-2008.9000836601656-4017.800167320331',\
            '83.3333333333': 'VZ-4017.800167320331-6026.700250980497',\
            }
psels = {'16.6666666667': 'Z-0.0-33.333333333333336-0.0',\
         '50.0':          'Z-33.333333333333336-66.66666666666667-0.0',\
         '83.3333333333': 'Z-66.66666666666667-100.0-0.0',\
         }

size_mass_isl = [('630', 'Mstar_Msun-1995262314.97-None'), ('630', 'Mstar_Msun-5011872336.27-None'), ('2273', 'Mstar_Msun-1e+11-None'), ('2273', 'Mstar_Msun-1.58489319246e+11-None')]
sizes_grp = ['100.0', '200.0', '500.0', '1000.0', '10000.0']
masses_grp = ['M200c_Msun-2.5e+13-None', 'M200c_Msun-5e+13-None']


mask_base_24_grp = 'mask_RefL0100N1504_24_32000pix_z-projection_totalbox_halosize-%s-pkpc_closest-normradius_selection-in_%s_%s.hdf5'
mask_base_24_isl = 'mask_RefL0100N1504_24_32000pix_z-projection_totalbox_inclsatellites_halosize-%s-pkpc_closest-normradius_selection-in_%s_%s.hdf5'
mask_base_23_grp = 'mask_RefL0100N1504_23_32000pix_z-projection_totalbox_halosize-%s-pkpc_closest-normradius_selection-in_%s_%s.hdf5'

masks_24_isl = {key: [None] + [mask_base_24_isl%(szms + (psels[key],)) for szms in size_mass_isl] + [mask_base_24_isl%(szms + (vsels_24[key],)) for szms in size_mass_isl] for key in vsels_24.keys()}
masks_24_grp = {key: [None] + [mask_base_24_grp%((sz, ms, psels[key])) for sz in sizes_grp for ms in masses_grp] + [mask_base_24_grp%((sz, ms, vsels_24[key])) for sz in sizes_grp for ms in masses_grp] for key in vsels_24.keys()}
masks_23_grp = {key: [None] + [mask_base_23_grp%((sz, ms, psels[key])) for sz in sizes_grp for ms in masses_grp] + [mask_base_23_grp%((sz, ms, vsels_23[key])) for sz in sizes_grp for ms in masses_grp] for key in vsels_23.keys()}
             
masknames_isl = ['nomask',\
                 '630pkpc_pos_logMstar-geq-9.3-Msun',\
                 '630pkpc_pos_logMstar-geq-9.7-Msun',\
                 '2273pkpc_pos_logMstar-geq-11.0-Msun',\
                 '2273pkpc_pos_logMstar-geq-11.2-Msun',\
                 '630pkpc_vel_logMstar-geq-9.3-Msun',\
                 '630pkpc_vel_logMstar-geq-9.7-Msun',\
                 '2273pkpc_vel_logMstar-geq-11.0-Msun',\
                 '2273pkpc_vel_logMstar-geq-11.2-Msun',\
                ]
masknames_grp = ['nomask',\
                 '100pkpc_pos_M200c-geq-2.5e13-Msun',\
                 '100pkpc_pos_M200c-geq-5.0e13-Msun',\
                 '200pkpc_pos_M200c-geq-2.5e13-Msun',\
                 '200pkpc_pos_M200c-geq-5.0e13-Msun',\
                 '500pkpc_pos_M200c-geq-2.5e13-Msun',\
                 '500pkpc_pos_M200c-geq-5.0e13-Msun',\
                 '1000pkpc_pos_M200c-geq-2.5e13-Msun',\
                 '1000pkpc_pos_M200c-geq-5.0e13-Msun',\
                 '10000pkpc_pos_M200c-geq-2.5e13-Msun',\
                 '10000pkpc_pos_M200c-geq-5.0e13-Msun',\
                 '100pkpc_vel_M200c-geq-2.5e13-Msun',\
                 '100pkpc_vel_M200c-geq-5.0e13-Msun',\
                 '200pkpc_vel_M200c-geq-2.5e13-Msun',\
                 '200pkpc_vel_M200c-geq-5.0e13-Msun',\
                 '500pkpc_vel_M200c-geq-2.5e13-Msun',\
                 '500pkpc_vel_M200c-geq-5.0e13-Msun',\
                 '1000pkpc_vel_M200c-geq-2.5e13-Msun',\
                 '1000pkpc_vel_M200c-geq-5.0e13-Msun',\
                 '10000pkpc_vel_M200c-geq-2.5e13-Msun',\
                 '10000pkpc_vel_M200c-geq-5.0e13-Msun',\
                 ]

if jobind == 161:
    mh.makehist_masked_toh5py(filename_24, fills=fills, maskfiles=masks_24_isl, masknames=masknames_isl,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + (filename_24%('-all'))[:-4] + '_masks_Mstar-9.2-9.3-11.0-11.2_isolation.hdf5')
elif jobind == 162:
    mh.makehist_masked_toh5py(filename_24, fills=fills, maskfiles=masks_24_grp, masknames=masknames_grp,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + (filename_24%('-all'))[:-4] + '_masks_M200c-2.5e13-5e13_environment.hdf5')
elif jobind == 163:
    mh.makehist_masked_toh5py(filename_23, fills=fills, maskfiles=masks_23_grp, masknames=masknames_grp,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + (filename_23%('-all'))[:-4] + '_masks_M200c-2.5e13-5e13_environment.hdf5')
    
##### 16 slice CDDF splits, halos assigned based on slice center membership 
fills = [str(i) for i in np.arange(16) / 16. * 100. + 100. / 32.] # slice centers
psels = {fill: 'Z-%s-%s-0.0'%(float(fill) - 100. / 32., float(fill) + 100. / 32.) for fill in fills}

mass_fills = [('logM200c_Msun-9.0-9.5',   'logM200c_Msun-9.5-None'),\
              ('logM200c_Msun-9.5-10.0',  'logM200c_Msun-None-9.5_logM200c_Msun-10.0-None'),\
              ('logM200c_Msun-10.0-10.5', 'logM200c_Msun-10.5-None_logM200c_Msun-None-10.0'),\
              ('logM200c_Msun-10.5-11.0', 'logM200c_Msun-None-10.5_logM200c_Msun-11.0-None'),\
              ('logM200c_Msun-11.0-11.5', 'logM200c_Msun-11.5-None_logM200c_Msun-None-11.0'),\
              ('logM200c_Msun-11.5-12.0', 'logM200c_Msun-None-11.5_logM200c_Msun-12.0-None'),\
              ('logM200c_Msun-12.0-12.5', 'logM200c_Msun-12.5-None_logM200c_Msun-None-12.0'),\
              ('logM200c_Msun-12.5-13.0', 'logM200c_Msun-None-12.5_logM200c_Msun-13.0-None'),\
              ('logM200c_Msun-13.0-13.5', 'logM200c_Msun-13.5-None_logM200c_Msun-None-13.0'),\
              ('logM200c_Msun-13.5-14.0', 'logM200c_Msun-None-13.5_logM200c_Msun-14.0-None'),\
              ('logM200c_Msun-14.0-inf',  'logM200c_Msun-None-14.0'),\
              ]

mask_names = ['nomask'] + [fl[0] for fl in mass_fills]

mask_base = 'mask_RefL0100N1504_27_32000pix_z-projection_totalbox_halosize-1.0-R200c_closest-normradius_%s.hdf5'
mask_selpart = 'selection-in_%s_%s_selection-ex_%s_%s' # Z_in, M_in, Z_ex, M_ex

masks_all = {fill: [None] + [mask_base%(mask_selpart%(psels[fill], mass_fills[i][0], psels[fill], mass_fills[i][1])) for i in range(len(mass_fills))] for fill in fills}

filename_fe17 = ol.ndir + 'coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' 
filename_ne8  = ol.ndir + 'coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen%s_T4SFR.npz'
filename_o8   = ol.ndir + 'coldens_o8_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_o7   = ol.ndir_old + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_o6   = ol.ndir + 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_ne9  = ol.ndir_old + 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_hn   = ol.ndir_old + 'coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

if jobind == 164:
    filename = filename_fe17
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')
elif jobind == 165:
    filename = filename_ne8
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')
elif jobind == 166:
    filename = filename_o8
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')
elif jobind == 167:
    filename = filename_o7
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')
elif jobind == 168:
    filename = filename_o6
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')
elif jobind == 169:
    filename = filename_ne9
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')    
elif jobind == 170:
    filename = filename_hn
    mh.makehist_masked_toh5py(filename, fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename='cddf_' + ((filename.split('/')[-1])%('-all'))[:-4] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5')


### SFR vs HI vs Sigma_gas
fills = [str(i) for i in np.arange(8)/8. * 25. + 25./16.]
name_mass   = 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
name_h1     = 'coldens_h1ssh_L0025N0752RECALIBRATED_19_test3.31_PtAb_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
name_hn     = 'coldens_hneutralssh_L0025N0752RECALIBRATED_19_test3.31_PtAb_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz'
name_sfr    = 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen%s_z-projection_T4EOS.npz' # log10( Msun / yr / kpc^2 )= -17.179065788706197
        
edges_mass   = np.array([-np.inf] + list(np.arange(-6.3, 0.15, 0.1)))
edges_sfr    = np.array([-np.inf, -46., -30., -25., -23.,] + list(np.arange(-22., -15.85, 0.1)))
edges_h1     = np.array([-np.inf] + list(np.arange(10.3, 23.35, 0.1)))
inclinf = True
        
outname_h1 = 'Mass_h1ssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection'
outname_hn = 'Mass_hneutralssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection'    


if jobind == 171: 
    mh.makehist_fromnpz(name_mass, name_h1, name_sfr,\
                    bins = [edges_mass, edges_h1, edges_sfr],\
                    dimlabels = ['Mass', 'NHI', 'StarFormationRate'],\
                    save=mh.pdir + outname_h1,\
                    fills=fills, includeinf=inclinf)
elif jobind == 172: 
    mh.makehist_fromnpz(name_mass, name_hn, name_sfr,\
                    bins = [edges_mass, edges_h1, edges_sfr],\
                    dimlabels = ['Mass', 'NHI', 'StarFormationRate'],\
                    save=mh.pdir + outname_hn,\
                    fills=fills, includeinf=inclinf)   

## o8 emission origin
if jobind == 173:
    emname = 'emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
    Tname = 'Temperature_T4EOS_emission_o8_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection.npz'
    rhoname = 'Density_T4EOS_emission_o8_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection.npz'
    closefills = np.arange(7)/7. * 100. + 100./14.
    fills = getcenfills(emname, closevals=closefills, searchdir=None, tolerance=1e-4) 
    
    embins = np.array([-np.inf, -50., -40., -30., -20.] + list(np.arange(-15., -9.5, 1.)) + list(np.arange(-9.9, 5.15, 0.1)) + [np.inf]) # (-49.38524, 5.066906)
    Tbins = np.array([-np.inf] + list(np.arange(2.5, 8.55, 0.1)) + [np.inf]) # (2.7191238, 8.466448)
    rhobins = np.array([-np.inf] + list(np.arange(-32.4, -22.35, 0.1)) + [np.inf]) # (-32.248672, -22.516676)

    mh.makehist_masked_toh5py(emname, rhoname, Tname, fills=fills,\
                              includeinf=True, bins=[embins, rhobins, Tbins],\
                              outfilename='hist_emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_and_weighted_rho_T.hdf5')


## masks to apply to halo-only projections (quasar)
## update mode: should check which input and output files exist, and generate 
## what missing outputs it can
fills = [str(i) for i in np.arange(16) / 16. * 100. + 100. / 32.] # slice centers
psels = {fill: 'Z-%s-%s-0.0'%(float(fill) - 100. / 32., float(fill) + 100. / 32.) for fill in fills}

mass_fills = [('logM200c_Msun-9.0-9.5',   'logM200c_Msun-9.5-None'),\
              ('logM200c_Msun-9.5-10.0',  'logM200c_Msun-None-9.5_logM200c_Msun-10.0-None'),\
              ('logM200c_Msun-10.0-10.5', 'logM200c_Msun-10.5-None_logM200c_Msun-None-10.0'),\
              ('logM200c_Msun-10.5-11.0', 'logM200c_Msun-None-10.5_logM200c_Msun-11.0-None'),\
              ('logM200c_Msun-11.0-11.5', 'logM200c_Msun-11.5-None_logM200c_Msun-None-11.0'),\
              ('logM200c_Msun-11.5-12.0', 'logM200c_Msun-None-11.5_logM200c_Msun-12.0-None'),\
              ('logM200c_Msun-12.0-12.5', 'logM200c_Msun-12.5-None_logM200c_Msun-None-12.0'),\
              ('logM200c_Msun-12.5-13.0', 'logM200c_Msun-None-12.5_logM200c_Msun-13.0-None'),\
              ('logM200c_Msun-13.0-13.5', 'logM200c_Msun-13.5-None_logM200c_Msun-None-13.0'),\
              ('logM200c_Msun-13.5-14.0', 'logM200c_Msun-None-13.5_logM200c_Msun-14.0-None'),\
              ('logM200c_Msun-14.0-inf',  'logM200c_Msun-None-14.0'),\
              ]

mask_names = ['nomask'] + [fl[0] for fl in mass_fills]

mask_base = 'mask_RefL0100N1504_27_32000pix_z-projection_totalbox_halosize-1.0-R200c_closest-normradius_%s.hdf5'
mask_selpart = 'selection-in_%s_%s_selection-ex_%s_%s' # Z_in, M_in, Z_ex, M_ex

masks_all = {fill: [None] + [mask_base%(mask_selpart%(psels[fill], mass_fills[i][0], psels[fill], mass_fills[i][1])) for i in range(len(mass_fills))] for fill in fills}

keys = ['o7', 'o8', 'o6', 'ne8', 'fe17', 'ne9', 'hneutralssh']
medges = np.arange(9., 14.1, 0.5)
halofills = [''] +\
            ['Mhalo_%s<=log200c<%s'%(medges[i], medges[i + 1]) if i < len(medges) - 1 else \
             'Mhalo_%s<=log200c'%medges[i] for i in range(len(medges))]
filenames_all = {key: [ol.ndir + 'coldens_%s_L0100N1504_27_test3.4_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS_halosel_%s_allinR200c_endhalosel.hdf5'%(key, '%s', halofill) for halofill in halofills]
                 for key in keys}

if jobind == 174:
    for key in filenames_all.keys():
        filenames = filenames_all[key]
        print('Checking %s'%key)
        outfilenames_all = ['cddf_' + ((fn.split('/')[-1])%('-all'))[:-5] + '_masks_M200c-0p5dex_mass-excl-ge-9_halosize-1.0-R200c_closest-normradius_halocen-margin-0.hdf5' for fn in filenames]
        outdir = ol.pdir
        for ind in range(len(filenames)):
            if np.all([os.path.isfile((filenames[ind])%fill) for fill in fills]) and not os.path.isfile(outdir + outfilenames_all[ind]):
                print('Running %s'%(outfilenames_all[ind]))
                mh.makehist_masked_toh5py(filenames[ind], fills=fills, maskfiles=masks_all, masknames=mask_names,\
                              includeinf=True, bins=[np.arange(-25., 28.05, 0.05)],\
                              outfilename=outfilenames_all[ind])                
            else:
                print('Skipping %s'%(outfilenames_all[ind]))


if jobind == 175:   
    fillsea = [str(float(i)) for i in (np.arange(16)/16.+1/32.)*100.]

    T_NO6_name_16_27 = ol.ndir_old + 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    NO6_name_16_27 = ol.ndir + 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    
    edgesNO6 = np.array(list((np.arange(5)/4.*(0.0+40.) -40.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(71)/70.*(17.-10.) +10.))
    edgesT_NO6 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
    
    T_NO7_name_16_27 = ol.ndir_old + 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_27_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    NO7_name_16_27 = ol.ndir + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    
    edgesNO7 = np.array(list((np.arange(4)/3.*(0.0+30.) -30.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(81)/80.*(18.-10.) +10.))
    edgesT_NO7 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))

    print('Doing EA100, 16 slices, (NO6, NO7, T_NO6, T_NO7)')
    mh.makehist_fromnpz(NO6_name_16_27, NO7_name_16_27, T_NO6_name_16_27, T_NO7_name_16_27,\
                    bins = [edgesNO6, edgesNO7, edgesT_NO6, edgesT_NO7],\
                    dimlabels = ['NO6','NO7','Temperature_w_NO6', 'Temperature_w_NO7'],\
                    save=mh.pdir +'hist_coldens_o6-o7_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature',\
                    fills = fillsea)   

# Ion histograms

if jobind >= 176 and jobind <= 183:   
    import make_maps_v3_master as m3
    simnum = 'L0100N1504'
    snapnum = 27
    var = 'REFERENCE'
    axesdct = [{'ptype': 'basic', 'quantity': 'Temperature', 'excludeSFR': 'T4'},\
               {'ptype': 'basic', 'quantity': 'Density'},\
               {'ptype': 'halo', 'quantity': 'Mass'},\
               {'ptype': 'halo', 'quantity': 'subcat'}]
    axbins = [0.1, 0.1, np.array([-np.inf, 0.0, 9.0, 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., np.inf]) + np.log10(c.solar_mass), None]
    if jobind == 176:
        ptype = 'basic'
        quantity = 'Mass'
        ion = None
    elif jobind == 177:
        ptype = 'Nion'
        quantity = None
        ion = 'o6'
    elif jobind == 178:
        ptype = 'Nion'
        quantity = None
        ion = 'o7'
    elif jobind == 179:
        ptype = 'Nion'
        quantity = None
        ion = 'o8'
    elif jobind == 180:
        ptype = 'Nion'
        quantity = None
        ion = 'ne8'
    elif jobind == 181:
        ptype = 'Nion'
        quantity = None
        ion = 'ne9'
    elif jobind == 182:
        ptype = 'Nion'
        quantity = None
        ion = 'fe17'
    elif jobind == 183:
        ptype = 'Nion'
        quantity = None
        ion = 'hneutralssh'
        
    m3.makehistograms_perparticle(ptype, simnum, snapnum, var, axesdct,
                               simulation='eagle',\
                               excludeSFR='T4', abunds='Pt', ion=ion, parttype='0', quantity=quantity,\
                               axbins=axbins,\
                               sylviasshtables=False, allinR200c=True, mdef='200c',\
                               L_x=None, L_y=None, L_z=None, centre=None, Ls_in_Mpc=True,\
                               misc=None,\
                               name_append=None, logax=True, loghist=False,
                               nameonly=False)

if jobind == 184:   
    fillsea = [str(float(i)) for i in (np.arange(16) / 16. + 1 / 32.) * 100.]

    T_NO6_name_16_27 = ol.ndir_old + 'Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_27_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    NO6_name_16_27 = ol.ndir + 'coldens_o6_L0100N1504_27_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    
    edgesNO6 = np.array([-np.inf] + list((np.arange(5)/4.*(0.0+40.) -40.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(71)/70.*(17.-10.) +10.))
    edgesT_NO6 = np.array([-np.inf] + list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))
    
    #T_NO7_name_16_27 = ol.ndir_old + 'Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_27_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    NO7_name_16_27 = ol.ndir + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    
    edgesNO7 = np.array([-np.inf] + list((np.arange(4)/3.*(0.0 + 30.) - 30.)[:-1]) + list((np.arange(11)/10.*(10.-0.0) +0.0)[:-1]) + list(np.arange(81)/80.*(18.-10.) +10.))
    #edgesT_NO7 = np.array(list((np.arange(6)/5.*(0.0+50.) -50.)[:-1]) + list((np.arange(6)/5.*(2.5-0.0) +0.0)[:-1]) + list(np.arange(66)/65.*(9.0-2.5) +2.5))

    # min/max: (8.870864, 22.710325)
    NH1_name_16_27 = ol.ndir + 'coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    edgesNH1 = edgesNH1 = np.array([-np.inf] + [8.0, 9.0, 10.0] + list(np.arange(11.0, 22.85, 0.1)))
    
    print('Doing EA100, 16 slices, (NO6, NO7, NH1, T_NO6)')
    mh.makehist_fromnpz(NO6_name_16_27, NO7_name_16_27, NH1_name_16_27, T_NO6_name_16_27,\
                    bins = [edgesNO6, edgesNO7, edgesNH1, edgesT_NO6],\
                    dimlabels = ['NO6','NO7','NH1', 'Temperature_w_NO6'],\
                    save=mh.pdir +'hist_coldens_o6-o7-hneutralssh_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature-NO6',\
                    fills = fillsea)   
################################################
################# CDDFS ########################
################################################

if jobind == 10001:
    mc.getcddf_npztonpz('coldens_o7_EA-ioneq-L0025N0752_26_test3.21_PtAb_C2Sm_16000pix_6.25slice_zcen%s_z-projection_T4EOS.npz', fillsea25, numbins=1060)
elif jobind == 10002:
    mc.getcddf_npztonpz('coldens_o7_EA-ioneq-L0025N0752_26_test3.21_PtAb_C2Sm_16000pix_6.25slice_zcen%s_z-projection_T4EOS_BenOpp1-chemtables.npz', fillsea25, numbins=1060)
elif jobind == 10003:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=1)
elif jobind == 10004:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=2)
elif jobind == 10005:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=4)
elif jobind == 10006:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=8)
elif jobind == 10007:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10008:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=1)
elif jobind == 10009:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=2)
elif jobind == 10010:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=4)
elif jobind == 10011:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=8)
elif jobind == 10012:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10013:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=1)
elif jobind == 10014:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=2)
elif jobind == 10015:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=4)
elif jobind == 10016:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=8)
elif jobind == 10017:
    mc.getcddf_npztonpz('coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10018:
    mc.getcddf_npztonpz('coldens_o6_L0100N1504_28_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz', fillsea, numbins=1060)
elif jobind == 10019:
    mc.getcddf_npztonpz('coldens_ne8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz', fillsea, numbins=1060)
    
# Sofia Gallego HI CDDFs at z=3.5
fills_ea25_48 =  [str(float(i)) for i in (np.arange(48)/48. + 1./96.)*25. ]   
if jobind == 10020:
    base = 'coldens_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_8000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
    fills = getcenfills(base, closevals=fills_ea25_48, searchdir=None, tolerance=1e-4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=1)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=2)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=3)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=6)
elif jobind == 10021:
    base = 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_8000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
    fills = getcenfills(base, closevals=fills_ea25_48, searchdir=None, tolerance=1e-4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=1)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=2)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=3)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=6)
elif jobind == 10022:
    base = 'coldens_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_4000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
    fills = getcenfills(base, closevals=fills_ea25_48, searchdir=None, tolerance=1e-4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=1)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=2)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=3)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=6)
elif jobind == 10023:
    base = 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_4000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
    fills = getcenfills(base, closevals=fills_ea25_48, searchdir=None, tolerance=1e-4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=1)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=2)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=3)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=6)
elif jobind == 10024:
    base = 'coldens_h1ssh_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
    fills = getcenfills(base, closevals=fills_ea25_48, searchdir=None, tolerance=1e-4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=1)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=2)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=3)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=6)
elif jobind == 10025:
    base = 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
    fills = getcenfills(base, closevals=fills_ea25_48, searchdir=None, tolerance=1e-4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=1)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=2)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=3)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=4)
    mc.getcddf_npztonpz(base, fills, numbins=1060, add=6)
    
elif jobind == 10026:
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=1)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=2)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=4)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=8)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10027:
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=1)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=2)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=4)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=8)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10028:
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=1)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=2)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=4)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=8)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_fully-ionized-EOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10029:
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=1)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=2)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=4)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=8)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_12_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10030:
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=1)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=2)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=4)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=8)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_19_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=16)
elif jobind == 10031:
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=1)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=2)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=4)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=8)
    mc.getcddf_npztonpz(ol.ndir_old + 'coldens_electrons_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_noEOS_totalbox%s.npz', [''], numbins=1060, red=16)   
    
    
fills100 = [str(i) for i in (np.arange(16) + 0.5) * 100./16.]
if jobind == 10032:
    mc.getcddf_npztonpz('coldens_o7_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz', fills100, numbins=1060)
elif jobind == 10033:
    mc.getcddf_npztonpz('coldens_o8_L0100N1504_26_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz', fills100, numbins=1060)
    

############################################
############ radial profiles ###############
############################################

# radial profiles for Sofia's stuff
name_h1 = 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen%s_z-projection_T4EOS_velocity-sliced.npz'
fills_ea25_48 =  [str(float(i)) for i in (np.arange(48)/48. + 1./96.)*25. ]   
outname_base = 'coldens_h1_L0025N0752RECALIBRATED_11_test3.31_PtAb_C2Sm_16000pix_0.520833333333slice_zcen-all_z-projection_T4EOS_velocity-sliced_Mhalo0p5dex-%.1f-%.1f_%islice_offset-0.000000_velspace_centrals.hdf5'
catname = 'catalogue_RecalL0025N0752_snap11_aperture30_inclsatellites.hdf5'
selections = [[('M200c_Msun', 10**minval, 10**(minval + 0.5)),  ('SubGroupNumber', -0.5, 0.5)] for minval in np.arange(9., 11.6, 0.5)] #
numsls = [4, 8]

if jobind in range(20000, 20000 + 12):
    fills = getcenfills(name_h1, closevals=fills_ea25_48, searchdir=ol.ndir, tolerance=1e-4)
    ari = jobind - 20000
    si = ari // 2
    ni = ari % 2
    selection = selections[si]
    numsl = numsls[ni]
    outname = outname_base%(np.log10(selection[0][1]), np.log10(selection[0][2]) , numsl)
    print(outname)
    crd.rdists_sl_from_selection(name_h1, fills, 25., 16000,\
                             0., 2., catname, selection, 200, outname=outname,\
                             numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=100.,\
                             axis='z', velspace=True, offset_los=0., stamps=False)

base_o7trip = 'emission_o7trip_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
base_o8 =     'emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
base_fe17 =   'emission_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
szcens = [str(i) for i in np.arange(7)/7. * 100. + 100./14.]
L_x = 100.
npix_x = 32000
rmin_r200c = 0.
rmax_r200c = 2.5
mindist_pkpc = 1200.
catname = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30_inclsatellites.hdf5'
selection = [('M200c_Msun', 10**11.95, 10**13.05), ('SubGroupNumber', -0.05, 0.05)]

if jobind == 20013:
    outname = 'rdist_emission_o7trip_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0'
    crd.rdists_sl_from_selection(base_o7trip, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c,\
                     catname,\
                     selection, np.inf, outname=outname,\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                     axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20014:
    outname = 'rdist_emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0'
    crd.rdists_sl_from_selection(base_o8, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c,\
                     catname,\
                     selection, np.inf, outname=outname,\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                     axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20015:
    outname = 'rdist_emission_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0'
    crd.rdists_sl_from_selection(base_fe17, szcens, L_x, npix_x,\
                     rmin_r200c, rmax_r200c,\
                     catname,\
                     selection, np.inf, outname=outname,\
                     numsl=1, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                     axis='z', velspace=False, offset_los=0., stamps=False)


### radial profiles from 16-slice snap 27 EAGLE

filename_fe17 = ol.ndir + 'coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' 
filename_ne8  = ol.ndir + 'coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen%s_T4SFR.npz'
filename_o8   = ol.ndir + 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_o7   = ol.ndir + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_o6   = ol.ndir + 'coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
filename_ne9  = ol.ndir + 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz' 
filename_hn   = ol.ndir + 'coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'

if jobind >= 20016 and jobind <= 20029: 
    szcens = [str(i) for i in np.arange(16)/16. * 100. + 100./32.]
    L_x = 100.
    npix_x = 32000
    rmin_r200c = 0.
    rmax_r200c = 3.
    mindist_pkpc = 100.
    catname = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    # select 1000 halos randomly in  0.5 dex M200c bins (trying to do everything just gives memory errors)
    galids_dct = sh.L0100N1504_27_Mh0p5dex_7000.galids() 
    #print('for debug: galids_dct:\n')
    #print(galids_dct)
    #print('\n')
    allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
    #print('for debug: allids')
    #print(allids)
    #print('\n')
    selection = [('galaxyid', np.array(allids))]
else:
    selection = 'string: trigger error'

if jobind == 20016:
    numsl = 1
    filename = filename_fe17
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20017:
    numsl = 2
    filename = filename_fe17
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
    
elif jobind == 20018:
    numsl = 1
    filename = filename_ne8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20019:
    numsl = 2
    filename = filename_ne8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20020:
    numsl = 1
    filename = filename_o8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20021:
    numsl = 2
    filename = filename_o8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20022:
    numsl = 1
    filename = filename_o7
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20023:
    numsl = 2
    filename = filename_o7
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20024:
    numsl = 1
    filename = filename_o6
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20025:
    numsl = 2
    filename = filename_o6
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20026:
    numsl = 1
    filename = filename_ne9
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20027:
    numsl = 2
    filename = filename_ne9
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20028:
    numsl = 1
    filename = filename_hn
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20029:
    numsl = 2
    filename = filename_hn
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)


if jobind >= 20030 and jobind <= 20043: 
    szcens = [str(i) for i in np.arange(16)/16. * 100. + 100./32.]
    L_x = 100.
    npix_x = 32000
    rmin_r200c = 0.
    rmax_r200c = 3.
    mindist_pkpc = 100.
    catname = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    # select 1000 halos randomly in  0.5 dex M200c bins (trying to do everything just gives memory errors)
    galids_dct = sh.L0100N1504_27_Mh0p5dex_100.galids() 
    #print('for debug: galids_dct:\n')
    #print(galids_dct)
    #print('\n')
    allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
    #print('for debug: allids')
    #print(allids)
    #print('\n')
    selection = [('galaxyid', np.array(allids))]
else:
    selection = 'string: trigger error'

if jobind == 20030:
    numsl = 1
    filename = filename_fe17
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20031:
    numsl = 2
    filename = filename_fe17
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
    
elif jobind == 20032:
    numsl = 1
    filename = filename_ne8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20033:
    numsl = 2
    filename = filename_ne8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20034:
    numsl = 1
    filename = filename_o8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20035:
    numsl = 2
    filename = filename_o8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20036:
    numsl = 1
    filename = filename_o7
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20037:
    numsl = 2
    filename = filename_o7
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20038:
    numsl = 1
    filename = filename_o6
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20039:
    numsl = 2
    filename = filename_o6
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20040:
    numsl = 1
    filename = filename_ne9
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20041:
    numsl = 2
    filename = filename_ne9
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20042:
    numsl = 1
    filename = filename_hn
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20043:
    numsl = 2
    filename = filename_hn
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
    

if jobind >= 20044 and jobind <= 20057: 
    szcens = [str(i) for i in np.arange(16)/16. * 100. + 100./32.]
    L_x = 100.
    npix_x = 32000
    rmin_r200c = 0.
    rmax_r200c = 3.
    mindist_pkpc = 100.
    catname = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    # select 1000 halos randomly in  0.5 dex M200c bins (trying to do everything just gives memory errors)
    galids_dct = sh.L0100N1504_27_Mh0p5dex_1000.galids() 
    #print('for debug: galids_dct:\n')
    #print(galids_dct)
    #print('\n')
    allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
    #print('for debug: allids')
    #print(allids)
    #print('\n')
    selection = [('galaxyid', np.array(allids))]
else:
    selection = 'string: trigger error'

if jobind == 20044:
    numsl = 1
    filename = filename_fe17
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20045:
    numsl = 2
    filename = filename_fe17
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
    
elif jobind == 20046:
    numsl = 1
    filename = filename_ne8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20047:
    numsl = 2
    filename = filename_ne8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20048:
    numsl = 1
    filename = filename_o8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20049:
    numsl = 2
    filename = filename_o8
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20050:
    numsl = 1
    filename = filename_o7
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20051:
    numsl = 2
    filename = filename_o7
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20052:
    numsl = 1
    filename = filename_o6
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20053:
    numsl = 2
    filename = filename_o6
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20054:
    numsl = 1
    filename = filename_ne9
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20055:
    numsl = 2
    filename = filename_ne9
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20056:
    numsl = 1
    filename = filename_hn
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
elif jobind == 20057:
    numsl = 2
    filename = filename_hn
    outname = ol.pdir + 'rdist_%s_%islice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
    

### get 3d profiles
elif jobind in range(20058, 20066):
    weighttype = ['Mass', 'Volume', 'o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17'][jobind - 20058]
    p3g.genhists(samplename=None, rbinu='R200c', idsel=None, weighttype=weighttype, logM200min=11.0)    


### get 2d profiles by stellar mass sample
if jobind in range(20066, 20072): 
    szcens = [str(i) for i in np.arange(16)/16. * 100. + 100./32.]
    L_x = 100.
    npix_x = 32000
    rmin_r200c = 0.
    rmax_r200c = 3.
    
    catname = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    with h5py.File(catname, 'r') as cat:
        r200cvals = np.array(cat['R200c_pkpc'])
        galids = np.array(cat['galaxyid'])
        cosmopars = {key: item for key, item in cat['Header/cosmopars'].attrs.items()}
        
    # select 1000 halos randomly in  0.5 dex Mstar bins (trying to do everything just gives memory errors)
    galids_dct = sh.L0100N1504_27_Mstar_Mhbinmatch_1000.galids() 
     # set minimum distance based on virial radius of halo mass bin
    radii_mstarbins = {key: [r200cvals[galids == galid] for galid in galids_dct[key]] for key in galids_dct}
    p99_radii_mstarbins = {key: np.percentile(radii_mstarbins[key], 99.) for key in radii_mstarbins} # don't use the maxima since those are determined by outliers
    #print('for debug: galids_dct:\n')
    #print(galids_dct)
    #print('\n')
    allids = [gid for key in galids_dct.keys() for gid in galids_dct[key]]
    gkeys = list(galids_dct.keys())
    keymatch = [gkeys[np.where([gid in galids_dct[key] for key in gkeys])[0][0]] for gid in allids]
    mindist_pkpc = rmax_r200c * np.array([p99_radii_mstarbins[gkey] for gkey in keymatch])
    #print('for debug: allids')
    #print(allids)
    #print('\n')
    selection = [('galaxyid', np.array(allids))]
    
    ions = ['o6', 'o7', 'o8', 'ne8', 'ne9', 'fe17']
    ionind = jobind - 20066
    ion = ions[ionind]
    numsl = 1
    
    filenames = {'o6': ol.ndir + 'coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                 'o7': ol.ndir + 'coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                 'o8': ol.ndir + 'coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                 'fe17': ol.ndir + 'coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                 'ne8': ol.ndir + 'coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen%s_T4SFR.npz',\
                 'ne9': ol.ndir + 'coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz',\
                 }
    filename = filenames[ion]
    outname = ol.pdir + 'rdist_%s_%islice_to-99p-3R200c_Mstar-M200c-0p5dex-match_centrals.hdf5'%((filename.split('/')[-1][:-4])%('-all'), numsl)
    
    crd.rdists_sl_from_selection(filename, szcens, L_x, npix_x,\
                         rmin_r200c, rmax_r200c,\
                         catname,\
                         selection, np.inf, outname=outname,\
                         numsl=numsl, npix_y=None, logquantity=True, mindist_pkpc=mindist_pkpc,\
                         axis='z', velspace=False, offset_los=0., stamps=False)
    
###############################################################################
####### mask generation: fast enough for ipython, but good to have documented #
###############################################################################

# group environment: centrals only
halocat_24 = 'catalogue_RefL0100N1504_snap24_aperture30.hdf5'
halocat_23 = 'catalogue_RefL0100N1504_snap23_aperture30.hdf5'
# closest galaxies: centrals and satellites
halocat_23_sat = 'catalogue_RefL0100N1504_snap23_aperture30_inclsatellites.hdf5'
halocat_24_sat = 'catalogue_RefL0100N1504_snap24_aperture30_inclsatellites.hdf5'

if jobind == 30000: # Leiden
    # group criterion:
    #   one mass, (with a lower one for comparison). This is about total mass    
    #   different radii (pkpc) around the group
    #   snapshots 23, 24: z=0.37, z=0.50 pretty much bracket the z=0.43 of the absorber found
    #   V ans Z selections matching slices, and total
    
    # isolation criterion:
    #   two masses, with two variations: stellar mass
    #   radii go with the specific masses, since these are the masses galaxies were found at
    #   snapshot 24: z=0.37 is pretty close to the absorber's z=0.35
    #   V and Z selections matching slices, and total
    
    # get z, Vz selections (only need cosmological parameters from catalogues, cen/sat doesn't matter)
    with h5py.File(crd.ol.pdir + halocat_24_sat, 'r') as hc24:
        cosmopars_24 = {key: item for (key, item) in hc24['Header/cosmopars'].attrs.items()}
    with h5py.File(crd.ol.pdir + halocat_23_sat, 'r') as hc23:
        cosmopars_23 = {key: item for (key, item) in hc23['Header/cosmopars'].attrs.items()}
    vvals_24 = cu.Hubble(cosmopars_24['z'], cosmopars=cosmopars_24) * 100. * cosmopars_24['a'] * c.cm_per_mpc * 1e-5 * np.arange(0., 1.1, 1./3.)
    vzsels_24 = [('VZ', vvals_24[i], vvals_24[i+1]) for i in range(3)]
    vvals_23 = cu.Hubble(cosmopars_23['z'], cosmopars=cosmopars_23) * 100. * cosmopars_23['a'] * c.cm_per_mpc * 1e-5 * np.arange(0., 1.1, 1./3.)
    vzsels_23 = [('VZ', vvals_23[i], vvals_23[i+1]) for i in range(3)]
    
    zvals = np.arange(0., 100.1, 100./3.)
    zsels = [('Z', zvals[i], zvals[i+1], 0.) for i in range(3)]

    msels_group = [('M200c_Msun', 5e+13, None), ('M200c_Msun', 2.5e13, None)]   
    radii_group = [100., 200., 500., 1000., 10000.]
    
    msels_isolation = [('Mstar_Msun', 10**9.3, None), ('Mstar_Msun', 10**9.7, None), ('Mstar_Msun', 10**11.0, None), ('Mstar_Msun', 10**11.2, None)]
    radii_isolation = [630, 630, 2273, 2273] # pkpc
    
    # group loop
    for rad in radii_group:
        for snap in [23, 24]:
            if snap == 23:
                halocat = halocat_23
                vzsels = vzsels_23
            elif snap == 24:
                halocat = halocat_24
                vzsels = vzsels_24
            psels = zsels + vzsels 
            for psel in psels:
                for msel in msels_group:
                    sel = [msel, psel]
                    crd.gethalomask_fromhalocat(halocat, 32000,\
                            radius_r200=None, radius_pkpc=rad, closest_normradius=True,\
                            selection_in=sel, selection_ex=None,\
                            axis='z', outfile='auto')
                    
    # isolated loop:
    for rmi in range(len(radii_isolation)):
        msel = msels_isolation[rmi]
        rad  = radii_isolation[rmi]

        halocat = halocat_24_sat
        vzsels = vzsels_24
        psels = zsels + vzsels 
        for psel in psels:
            sel = [msel, psel]
            crd.gethalomask_fromhalocat(halocat, 32000,\
                    radius_r200=None, radius_pkpc=rad, closest_normradius=True,\
                    selection_in=sel, selection_ex=None,\
                    axis='z', outfile='auto')

#rqfile_o7 = 'rdist_emission_o7trip_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-1p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0.hdf5'
#rqfile_o8 = 'rdist_emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-1p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0.hdf5'
#rqfile_fe17 = 'rdist_emission_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-1p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0.hdf5'
rqfile_o7 = 'rdist_emission_o7trip_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0.hdf5'
rqfile_o8 = 'rdist_emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0.hdf5'
rqfile_fe17 = 'rdist_emission_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0.hdf5'

halocat = 'catalogue_RefL0100N1504_snap27_aperture30_inclsatellites.hdf5'        
if jobind == 30001: # cosma
    yvals = [1., 5., 10., 25., 50., 75., 90., 95., 99.]
    rbins_R200c = np.arange(0., 2.55, 0.1)
    rbins_pkpc = np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 160., 180., 200., 225., 250., 275.] + list(np.arange(300., 1205., 50.)))
    
    # extract galaxy ids for different mass ranges, centrals only
    medges = 10**np.arange(11.95, 13.1, 0.1)
    msels  = {'logM200c_Msun_%.1f'%(0.5*(np.log10(medges[i]) + np.log10(medges[i+1]))):('M200c_Msun', medges[i], medges[i+1]) for i in range(len(medges) - 1)}
    
    with h5py.File(ol.pdir + halocat, 'r') as hc:
        galids = np.array(hc['galaxyid'])
        mhalos = np.array(hc['M200c_Msun'])
        iscens = np.array(hc['SubGroupNumber']) == 0

        galids = galids[iscens]
        mhalos = mhalos[iscens]
        del iscens

        galsels = {key: galids[np.logical_and(mhalos >= msels[key][1], mhalos < msels[key][2])] for key in msels.keys()}
        del galids
        del mhalos
    
    for galselkey in galsels.keys():
        galids = galsels[galselkey]
        for rqfile in [rqfile_o7, rqfile_o8, rqfile_fe17]:
            for xunit in ['pkpc', 'R200c']:
                if xunit == 'pkpc':
                    rbins = rbins_pkpc
                elif xunit == 'R200c':
                    rbins = rbins_R200c

                crd.get_radprof(rqfile, halocat, rbins, yvals,\
                            xunit=xunit, ytype='perc',\
                            galids=galids, combinedprofile=True,\
                            separateprofiles=True,\
                            rpfilename=None, galsettag=galselkey)

elif jobind == 30002: # save reduced-resolution emission maps
    basename_o7 = 'emission_o7trip_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
    basename_o8 = 'emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
    basename_fe17 = 'emission_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen%s_z-projection_T4EOS.npz'
    
    fills = ['7.14285714286', '21.4285714286', '35.7142857143', '50.0', '64.2857142857', '78.5714285714', '92.8571428571']
    
    basenames = [basename_o7, basename_o8, basename_fe17]
    
    for basename in basenames:
        outname = ol.pdir + (basename[:-4])%'-all' + '_reduced_res.hdf5'
        with h5py.File(outname, 'w') as fo:
            for fill in fills:
                grp = fo.create_group('zcen-%s'%fill)
                grp.attrs.create('zcen', float(fill))
                
                # 32000 pixels -> store at 1600, 800, 400
                img = np.load(ol.ndir + basename%fill)['arr_0']
                img1600 = imreduce(img, 20, log=True, method='average')
                img800  = imreduce(img, 40)
                img400  = imreduce(img, 80)
                del img
                
                grp.create_dataset('1600pix', data=img1600)
                grp.create_dataset('800pix', data=img800)
                grp.create_dataset('400pix', data=img400)

if jobind == 30003: # cosma
    yvals = [1., 5., 10., 25., 50., 75., 90., 95., 99.]
    #rbins_R200c = np.arange(0., 2.55, 0.1)
    rbins_pkpc = np.array([0., 3., 5., 7., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 160., 180., 200., 225., 250., 275.] + list(np.arange(300., 1205., 50.)))
    
    # extract galaxy ids for different mass ranges, centrals only
    medges = 10**np.arange(11.95, 13.1, 0.1)
    msels  = {'logM200c_Msun_%.1f'%(0.5*(np.log10(medges[i]) + np.log10(medges[i+1]))):('M200c_Msun', medges[i], medges[i+1]) for i in range(len(medges) - 1)}
    
    with h5py.File(ol.pdir + halocat, 'r') as hc:
        galids = np.array(hc['galaxyid'])
        mhalos = np.array(hc['M200c_Msun'])
        iscens = np.array(hc['SubGroupNumber']) == 0

        galids = galids[iscens]
        mhalos = mhalos[iscens]
        del iscens

        galsels = {key: galids[np.logical_and(mhalos >= msels[key][1], mhalos < msels[key][2])] for key in msels.keys()}
        del galids
        del mhalos
    
    for galselkey in galsels.keys():
        galids = galsels[galselkey]
        for rqfile in [rqfile_o7, rqfile_o8, rqfile_fe17]:
            xunit = 'pkpc'
            rbins = rbins_pkpc
               
            crd.get_radprof(rqfile, halocat, rbins, yvals,\
                        xunit=xunit, ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=True,\
                        rpfilename=None, galsettag=galselkey)


if jobind == 30004: # Leiden
    # group criterion:
    #   one mass, (with a lower one for comparison). This is about total mass    
    #   different radii (pkpc) around the group
    #   snapshots 23, 24: z=0.37, z=0.50 pretty much bracket the z=0.43 of the absorber found
    #   V ans Z selections matching slices, and total
    
    # isolation criterion:
    #   two masses, with two variations: stellar mass
    #   radii go with the specific masses, since these are the masses galaxies were found at
    #   snapshot 24: z=0.37 is pretty close to the absorber's z=0.35
    #   V and Z selections matching slices, and total
    
    # get z, Vz selections (only need cosmological parameters from catalogues, cen/sat doesn't matter)
    # snapshot 27 CDDF splits by halo mass
    halocat = ol.pdir + 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    
    zvals = np.arange(0., 100.1, 6.25)
    zsels = [('Z', zvals[i], zvals[i+1], 0.) for i in range(len(zvals) - 1)]
    
    mvals = list(np.arange(9.0, 14.1, 0.5)) + [np.inf]    
    msels_incl = [('logM200c_Msun', mvals[i], mvals[i+1]) for i in range(len(mvals) - 1)]   
    
    msels_excl = [list(({('logM200c_Msun', None, msels_incl[i][1]) if msels_incl[i][1] > mvals[0] else None} |\
                        {('logM200c_Msun', msels_incl[i][2], None) if msels_incl[i][2] < mvals[-1] else None})\
                       - {None})\
                  for i in range(len(msels_incl))]
    
    for zsel in zsels:
        for mind in range(len(msels_incl)):
            mincl = msels_incl[mind]
            mexcl = msels_excl[mind]
            sel_incl = [zsel, mincl]
            sel_excl = [zsel] + mexcl
            crd.gethalomask_fromhalocat(halocat, 32000,\
                    radius_r200=1., radius_pkpc=None, closest_normradius=True,\
                    selection_in=sel_incl, selection_ex=sel_excl,\
                    axis='z', outfile='auto')
                    

if jobind >= 30005 and jobind <= 30008: # cosma
    if jobind == 30005:
        filenames = [\
                 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 ]
    elif jobind == 30006:
         filenames = ['rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                      ]
    elif jobind == 30007:
         filenames = ['rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                      ]
    elif jobind == 30008:
         filenames = ['rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-7000_centrals.hdf5',\
                ]
    halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    yvals = [1., 5., 10., 25., 50., 75., 90., 95., 99.]
    #rbins_R200c = np.arange(0., 2.55, 0.1)
    rbins_pkpc = np.array([0., 3., 5., 7., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 160., 180., 200., 225., 250., 275., 300., 325., 350., 375., 400., 425., 450., 475., 500.])
    rbins_r200c = np.arange(0., 2.51, 0.05)
    # negative margins -> margins take away from selection regions
    zedges = np.arange(0., 100.1, 6.25)
    zsels  = {'Z_off-edge-by-R200c': [('Z', zedges[i], zedges[i+1], 1.) for i in range(len(zedges) - 1)]}

    # galids selection from galaxyselector (halo masses in 0.5 dex)
    galids_mass = sh.L0100N1504_27_Mh0p5dex_7000.galids()
    selcombs = {'logM200c_Msun_%s'%key: galids_mass[key] for key in galids_mass.keys()}
    keys = galids_mass.keys() # set up beforehand to match orders in lists
    sels_temp = sh.gethaloselections(halocat, selections=[zsels['Z_off-edge-by-R200c'] + [('galaxyid', galids_mass[key])] for key in keys], names=['logM200c_Msun_%s_Z_off-edge-by-R200c'%key for key in keys])
    selcombs.update(sels_temp)
    
    for skey in selcombs.keys():
        galids = selcombs[skey]
        for rqfile in filenames:     
            print('Starting %s, pkpc'%rqfile)
            crd.get_radprof(rqfile, halocat, rbins_pkpc, yvals,\
                        xunit='pkpc', ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=False,\
                        rpfilename=None, galsettag=skey)
            print('Finished %s, pkpc\n'%rqfile)
            print('Starting %s, R200c'%rqfile)
            crd.get_radprof(rqfile, halocat, rbins_r200c, yvals,\
                        xunit='R200c', ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=False,\
                        rpfilename=None, galsettag=skey)
            print('Finished %s, R200c\n'%rqfile)


if jobind >= 30009 and jobind <= 30012: # cosma
    if jobind == 30009:
        filenames = [\
                 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 ]
    elif jobind == 30010:
         filenames = ['rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                      ]
    elif jobind == 30011:
         filenames = ['rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                      ]
    elif jobind == 30012:
         filenames = ['rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                 #'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-100_centrals.hdf5',\
                ]
    halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    yvals = [1., 5., 10., 25., 50., 75., 90., 95., 99.]
    #rbins_R200c = np.arange(0., 2.55, 0.1)
    rbins_pkpc = np.array([0., 3., 5., 7., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 160., 180., 200., 225., 250., 275., 300., 325., 350., 375., 400., 425., 450., 475., 500.])
    rbins_r200c = np.arange(0., 2.51, 0.05)
    # negative margins -> margins take away from selection regions
    zedges = np.arange(0., 100.1, 6.25)
    #zsels  = {'Z_off-edge-by-R200c': [('Z', zedges[i], zedges[i+1], 1.) for i in range(len(zedges) - 1)]}

    # galids selection from galaxyselector (halo masses in 0.5 dex)
    galids_mass = sh.L0100N1504_27_Mh0p5dex_100.galids()
    selcombs = {'logM200c_Msun_%s'%key: galids_mass[key] for key in galids_mass.keys()}
    keys = galids_mass.keys() # set up beforehand to match orders in lists
    #sels_temp = sh.gethaloselections(halocat, selections=[zsels['Z_off-edge-by-R200c'] + [('galaxyid', galids_mass[key])] for key in keys], names=['logM200c_Msun_%s_Z_off-edge-by-R200c'%key for key in keys])
    #selcombs.update(sels_temp)
    
    for skey in selcombs.keys():
        galids = selcombs[skey]
        for rqfile in filenames:     
            print('Starting %s, pkpc'%rqfile)
            crd.get_radprof(rqfile, halocat, rbins_pkpc, yvals,\
                        xunit='pkpc', ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=False,\
                        rpfilename=None, galsettag=skey)
            print('Finished %s, pkpc\n'%rqfile)
            print('Starting %s, R200c'%rqfile)
            crd.get_radprof(rqfile, halocat, rbins_r200c, yvals,\
                        xunit='R200c', ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=False,\
                        rpfilename=None, galsettag=skey)
            print('Finished %s, R200c\n'%rqfile)

if jobind >= 30013 and jobind <= 30016: # cosma
    if jobind == 30013:
        filenames = [\
                 'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_fe17_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_hneutralssh_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 ]
    elif jobind == 30014:
         filenames = ['rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_ne8_L0100N1504_27_test3_PtAb_C2Sm_32000pix_6.250000slice_zcen-all_T4SFR_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                      ]
    elif jobind == 30015:
         filenames = ['rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_o6_L0100N1504_27_test3.11_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                      ]
    elif jobind == 30016:
         filenames = ['rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_o7_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_1slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                 #'rdist_coldens_o8_L0100N1504_27_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_2slice_to-500-pkpc-or-2p5-R200c_M200c-0p5dex-1000_centrals.hdf5',\
                ]
    halocat = 'catalogue_RefL0100N1504_snap27_aperture30.hdf5'
    yvals = [1., 5., 10., 25., 50., 75., 90., 95., 99.]
    #rbins_R200c = np.arange(0., 2.55, 0.1)
    rbins_pkpc = np.array([0., 3., 5., 7., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 120., 140., 160., 180., 200., 225., 250., 275., 300., 325., 350., 375., 400., 425., 450., 475., 500.])
    rbins_r200c = np.arange(0., 2.51, 0.05)
    # negative margins -> margins take away from selection regions
    zedges = np.arange(0., 100.1, 6.25)
    #zsels  = {'Z_off-edge-by-R200c': [('Z', zedges[i], zedges[i+1], 1.) for i in range(len(zedges) - 1)]}

    # galids selection from galaxyselector (halo masses in 0.5 dex)
    galids_mass = sh.L0100N1504_27_Mh0p5dex_1000.galids()
    selcombs = {'logM200c_Msun_%s'%key: galids_mass[key] for key in galids_mass.keys()}
    keys = galids_mass.keys() # set up beforehand to match orders in lists
    #sels_temp = sh.gethaloselections(halocat, selections=[zsels['Z_off-edge-by-R200c'] + [('galaxyid', galids_mass[key])] for key in keys], names=['logM200c_Msun_%s_Z_off-edge-by-R200c'%key for key in keys])
    #selcombs.update(sels_temp)
    
    for skey in selcombs.keys():
        galids = selcombs[skey]
        for rqfile in filenames:     
            print('Starting %s, pkpc'%rqfile)
            crd.get_radprof(rqfile, halocat, rbins_pkpc, yvals,\
                        xunit='pkpc', ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=False,\
                        rpfilename=None, galsettag=skey)
            print('Finished %s, pkpc\n'%rqfile)
            print('Starting %s, R200c'%rqfile)
            crd.get_radprof(rqfile, halocat, rbins_r200c, yvals,\
                        xunit='R200c', ytype='perc',\
                        galids=galids, combinedprofile=True,\
                        separateprofiles=False,\
                        rpfilename=None, galsettag=skey)
            print('Finished %s, R200c\n'%rqfile)