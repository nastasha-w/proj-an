#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 18:23:49 2018

@author: wijers
"""

import numpy as np
import makehistograms_basic as mh
import sys

jobind = int(sys.argv[1])
print('Running set %i\n'%jobind)

if jobind ==1: # O7/O8 totals from EAGLE, in sofar as was incomplete; 11 read-ins
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))
      
    
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.2_C2Sm_32000pix_100.0slice_z-projection.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))

    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o8_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))


    ## all values -inf
    #name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.2_C2Sm_32000pix_100.0slice_z-projection.npz'
    #fills = None
    #minmax = mh.getminmax_fromnpz(name, fills=None)
    #print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))

elif jobind ==19:
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))

    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o8_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen-all_z-projection_totalboxQ.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/coldens_o7_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))

elif jobind == 21:    
    name = '/net/luttero/data2/temp/coldens_o8_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz'
    fills = None
    minmax = mh.getminmax_fromnpz(name, fills=None)
    print('%s:\n %f, %f\n'%(name, minmax[0],minmax[1]))

 # O7/O8 slices from BAHAMAS, in sofar as was incomplete; 12 read-ins/jobind
fills_BA = ['33.3333333333', '100.0', '166.666666667', '233.333333333', '300.0', '366.666666667']
if jobind ==2:
    name = '/net/luttero/data2/temp/coldens_o7_BA-L400N1024_32_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz'
    fills = fills_BA
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz'
    fills = fills_BA
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==3:
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_BA-L400N1024_32_test3.2_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz'
    fills = fills_BA
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==20:
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_BA-L400N1024_32_test3.21_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz'
    fills = fills_BA
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==4:
    name = '/net/luttero/data2/temp/coldens_o8_BA-L400N1024_32_test3.21_PtAb_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz'
    fills = fills_BA
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_BA-L400N1024_32_test3.21_C2Sm_32000pix_66.6666666667slice_zcen%s-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection.npz'
    fills = fills_BA
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    

# EAGLE O7/O8 16 slices: 1 slice set per job
fills_EA = np.arange(16)/16.*100. +100./32.
fills_EA_s = [str(fill) for fill in fills_EA]

if jobind ==5:
    name = '/net/luttero/data2/temp/coldens_o7_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==6:
    name = '/net/luttero/data2/temp/coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==7:
    name = '/net/luttero/data2/temp/coldens_o8_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==8:
    name = '/net/luttero/data2/temp/coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==9:
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==10:
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==11:
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o8_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==12:
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==13:
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==14:
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==15:
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o7_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==16:
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o7_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind ==17:
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o8_0.000549262436107massfracAb-0.752massfracHAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind ==18:
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_28_test3.1_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
elif jobind == 19:
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind == 20:
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind == 21:
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_o6_PtAb_T4EOS_L0100N1504_28_test3.3_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind == 22:
    name = '/net/luttero/data2/temp/coldens_o6_L0100N1504_28_test3.3_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1])) 
    
# ne9  N, rho, T:
elif jobind == 23:
    name = '/net/luttero/data2/temp/coldens_ne9_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen%s_z-projection_T4EOS.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print(name)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind == 24:    
    name = '/net/luttero/data2/temp/Temperature_T4EOS_coldens_ne9_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print(name)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))
    
    name = '/net/luttero/data2/temp/Density_T4EOS_coldens_ne9_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print(name)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind == 25:
    name = '/net/luttero/data2/temp/ElementAbundance-Oxygen_T4EOS_coldens_o8_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print(name)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))

elif jobind == 26:
    name = '/net/luttero/data2/temp/ElementAbundance-Neon_T4EOS_coldens_ne9_PtAb_T4EOS_L0100N1504_27_test3.31_C2Sm_32000pix_6.25slice_zcen%s_z-projection.npz'
    fills = fills_EA_s
    minmax = mh.getminmax_fromnpz(name, fills=fills)
    print(name)
    print('%s\n\t%s: \n %f, %f\n'%(name, str(fills), minmax[0],minmax[1]))