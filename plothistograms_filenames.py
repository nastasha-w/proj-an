#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:59:07 2019

@author: wijers

prevent each reload of plothistograms from reloading all the histogram files
"""

# T, rho, NO7 dicts

import numpy as np
import h5py
import make_maps_opts_locs as ol

pdir = ol.pdir

def dictfromnpz(filename):
    npz = np.load(filename, encoding='bytes') # encoding: python 2 -> 3 fix
    dct = {key: npz[key] for key in npz.keys()}
    return dct
# stick loaded histograms in dict-like object to get backwards compatibilty, but only load if used to preventunnecessary  up-front load time
class npzldif:
     def __init__(self,filename, numpixtot):
         self.filename = filename
         self.npt = numpixtot
         self.isloaded = False
     def load(self):
         self.dct = dictfromnpz(self.filename)
         self.dct['bins'] /= float(self.npt)
         self.isloaded = True
     def keys(self):
         if not self.isloaded:
              self.load()              
         return self.dct.keys()
     def get(self,key):
         if not self.isloaded:
              self.load()   
         return self.dct.get(key)
     def __getitem__(self,key):
         if not self.isloaded:
              self.load()   
         return self.dct[key]

class h5pyldif:
     def __init__(self,filename, numpixtot):
         self.filename = filename
         self.npt = numpixtot
         self.isloaded = False
     def load(self):
         self.dct = h5py.File(self.filename, 'r')
         self.isloaded = True
     def keys(self):
         if not self.isloaded:
              self.load()              
         return self.dct.keys()
     def get(self,key):
         if not self.isloaded:
              self.load()   
         return np.array(self.dct.get(key))
     def __getitem__(self, key):
         if not self.isloaded:
              self.load()   
         return np.array(self.dct[key])

### BAHAMAS, EAGLE, EAGLE at BAHAMAS res: o7, weighted rho, T
# PtAb         
h3ba = npzldif(pdir + 'hist_coldens_o7_BA-L400N1024_32_test3.2_PtAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_Temperature_Density.npz', 6*32000**2)
h3eahi = npzldif(pdir + 'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density.npz', 32000**2)
h3eami = npzldif(pdir + 'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_5600pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density.npz', 5600**2)
# 0.1solar
h3ba_fz = npzldif(pdir + 'hist_coldens_o7_BA-L400N1024_32_test3.2_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS_and_weighted_Temperature_Density.npz', 6*32000**2)
h3eahi_fz = npzldif(pdir + 'hist_coldens_o7_L0100N1504_28_test3.1_0.000549262436107massfracAb-0.752massfracHAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_Temperature_Density.npz', 32000**2)

maxminfrac = 1./5600**2

# NO7, NH dicts
h3bah = npzldif(pdir + 'hist_coldens_o7-and-hydrogen_BA-L400N1024_32_test3.2_PtAb_C2Sm_32000pix_66.6666666667slice_zcen-all-hm1_x200.0-pm400.0-hm1_y200.0-pm400.0-hm1_z-projection_T4EOS.npz', 6*32000**2)

h3eahih = npzldif(pdir + 'hist_coldens_o7-and-hydrogen_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_cens-sum_xyz-projection-average_T4EOS_totalbox.npz', 3*32000**2)

h3eamih = npzldif(pdir + 'hist_coldens_o7-and-hydrogen_L0100N1504_28_test3.1_PtAb_C2Sm_5600pix_6.25slice_cens-sum_xyz-projection-average_T4EOS_totalbox.npz', 3*5600**2)

maxminfrac = 1./(3*5600**2)

# N, T, rho, fO for o7, o8 z=0.0
ea3o6 = npzldif(pdir + 'hist_coldens_o6_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature.npz', 16*32000**2)

ea4o7 = npzldif(pdir + 'hist_coldens_o7_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature_Density_ElementAbundance-Oxygen.npz', 16*32000**2)

ea4o8 = npzldif(pdir + 'hist_coldens_o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature_Density_ElementAbundance-Oxygen.npz', 16*32000**2)

ea3ne8 = npzldif(pdir + 'hist_coldens_ne8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature.npz', 16*32000**2)
eafOo7 = npzldif(pdir + 'hist_coldens_o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen.npz', 16*32000**2) 
eafOo8 = npzldif(pdir + 'hist_coldens_o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen.npz', 16*32000**2)

eafOo7_100 = npzldif(pdir + 'hist_coldens_o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen.npz', 32000**2) 
eafOo8_100 = npzldif(pdir + 'hist_coldens_o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox_and_weighted_ElementAbundance-Oxygen_and_Mass-weighted-ElementAbundance-Oxygen.npz', 32000**2)

earhoo78 = npzldif(pdir + 'hist_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density.npz', 16*32000**2)
eaTo78 = npzldif(pdir + 'hist_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz', 16*32000**2)
eafOo78 = npzldif(pdir + 'hist_coldens_o7-o8_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen.npz', 16*32000**2)
eafOSmo78 = npzldif(pdir + 'coldens_o7-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Sm-Oxygen.npz', 16 * 32000**2) 

earhoTbyo78 = npzldif(pdir + 'hist_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_coldens_o7-o8_weighted_Density_Temperature.npz', 16*32000**2)
eaibo7byo78 = npzldif(pdir + 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_ibo7_from_o7-o8-weighted-rho-T.npz',16*32000**2) 
eaibo8byo78 = npzldif(pdir + 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_ibo8_from_o7-o8-weighted-rho-T.npz',16*32000**2)
eaibdiffo78byo78 = npzldif(pdir + 'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_ibdiff-o7-o8_from_o7-o8-weighted-rho-T.npz',16*32000**2)
eaibreldiffo78byo78 = npzldif(pdir +'hist_o7-o8_L0100N1504_28_test3.1_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_iblogreldiff-o7-o8_from_o7-o8-weighted-rho-T.npz', 16*32000)

earhoo68_27 = npzldif(pdir + 'hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density.npz', 16*32000**2)
eaTo68_27   = npzldif(pdir + 'hist_coldens_o6-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz', 16*32000**2)

earhoo67_28 = npzldif(pdir + 'hist_coldens_o6-o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density.npz', 16*32000**2)
eaTo67_28   = npzldif(pdir + 'hist_coldens_o6-o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz', 16*32000**2)
eafOo67_28  = npzldif(pdir + 'hist_coldens_o6-o7_L0100N1504_28_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_ElementAbundance-Oxygen.npz', 16*32000**2)

earhoo6ne8 = npzldif(pdir + 'hist_coldens_ne8-o6_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density.npz', 16*32000**2)
eaTo6ne8   = npzldif(pdir + 'hist_coldens_ne8-o6_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz', 16*32000**2)
earhoo7ne8 = npzldif(pdir + 'hist_coldens_ne8-o7_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density.npz', 16*32000**2)
eaTo7ne8   = npzldif(pdir + 'hist_coldens_ne8-o7_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz', 16*32000**2)
earhoo8ne8 = npzldif(pdir + 'hist_coldens_ne8-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density.npz', 16*32000**2)
eaTo8ne8   = npzldif(pdir + 'hist_coldens_ne8-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Temperature.npz', 16*32000**2)


# o6/7/8 at z=0.1

ea3o678_16 = npzldif(pdir + 'hist_coldens_o6-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_hires-8.npz', 16*32000**2)
ea3o678 = npzldif(pdir + 'hist_coldens_o6-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz', 32000**2)

ea3ne8o78_16 = npzldif(pdir + 'hist_coldens_ne8-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.npz', 16*32000**2)
ea3ne8o78 = npzldif(pdir + 'hist_coldens_ne8-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-sum_z-projection_T4EOS_totalbox.npz', 32000**2)

eao678ne9_16 = npzldif(pdir + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS.npz', 16 * 32000**2)
eao678ne9wfONe_16 = npzldif(pdir + 'hist_coldens_hneutralssh-o7-o8_L0100N1504_28_test3.31_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_inclzeros.npz', 16*32000**2)
eao678ne9wTo68_16 = npzldif(pdir + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_o6-o8_weighted_Temperatures.npz', 16*32000**2)
eao678ne9_o6wrhoTfOSm_16 = npzldif(pdir + 'hist_coldens_o6-o7-o8-ne9_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_and_o6_weighted_Temperature_Density_fOSm.npz', 16*32000**2)
                                  
# o7/8 AGN effect at z=0
ea4o7_ref50 = npzldif(pdir + 'coldens_o7_L0050N0752_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature_SmoothedElementAbundance-Oxygen.npz', 8 * 16000**2)
ea4o8_ref50 = npzldif(pdir + 'coldens_o8_L0050N0752_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature_SmoothedElementAbundance-Oxygen.npz', 8 * 16000**2)
ea4o7_noagn50 = npzldif(pdir + 'coldens_o7_L0050N0752EagleVariation_NoAGN_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature_SmoothedElementAbundance-Oxygen.npz', 8 * 16000**2)
ea4o8_noagn50 = npzldif(pdir + 'coldens_o8_L0050N0752EagleVariation_NoAGN_28_test3.31_PtAb_C2Sm_16000pix_6.25slice_zcen-all_z-projection_T4EOS_and_weighted_Density_Temperature_SmoothedElementAbundance-Oxygen.npz', 8 * 16000**2)

ea3_heutralo78 = npzldif(pdir + 'hist_coldens_hneutralssh-o7-o8_L0100N1504_27_test3.x_PtAb_C2Sm_32000pix_6.25slice_zcen-all_z-projection_T4EOS_inclzeros.npz', 16 * 32000**2)

# metallicity measurements
ea25Zmeas_h1   = npzldif(pdir + 'coldens_h1ssh_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass_coldens-h1ssh_StarFormationRate.npz', 4 * 8000**2)
ea25Zmeas_mass = npzldif(pdir + 'Mass_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass_coldens-h1ssh_StarFormationRate.npz', 4 * 8000**2)
ea25Zmeas_sfr  = npzldif(pdir + 'StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass_coldens-h1ssh_StarFormationRate.npz', 4 * 8000**2)

ea25Zdiff_h1   = npzldif(pdir + 'coldens_h1ssh_and_weighted_Z_diff_with_Mass_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)
ea25Zdiff_mass = npzldif(pdir + 'Mass_and_weighted_Z_diff_with_NHI_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)
ea25Zdiff_sfr  = npzldif(pdir + 'StarFormationRate_and_weighted_Z_diff_with_Mass_coldens_h1ssh_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)

ea25SmZdiff_h1   = npzldif(pdir + 'coldens_h1ssh_and_weighted_SmZ_diff_with_Mass_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)
ea25SmZdiff_mass = npzldif(pdir + 'Mass_and_weighted_SmZ_diff_with_coldens_h1ssh_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)
ea25SmZdiff_sfr  = npzldif(pdir + 'StarFormationRate_and_weighted_SmZ_diff_with_Mass_coldens_h1ssh_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)

ea25SmZdiff_h1_hn   = npzldif(pdir + 'coldens_hneutralssh_and_weighted_SmZ_diff_with_Mass_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)
ea25SmZdiff_mass_hn = npzldif(pdir + 'Mass_and_weighted_SmZ_diff_with_coldens_hneutralssh_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)
ea25SmZdiff_sfr_hn  = npzldif(pdir + 'StarFormationRate_and_weighted_SmZ_diff_with_Mass_coldens_hneutralssh_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection.npz', 4 * 8000**2)

ea25RecSmZdiff_h1   = npzldif(pdir + 'coldens_h1ssh_and_weighted_SmZ_diff_with_Mass_StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)
ea25RecSmZdiff_mass = npzldif(pdir + 'Mass_and_weighted_SmZ_diff_with_coldens_h1ssh_StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)
ea25RecSmZdiff_sfr  = npzldif(pdir + 'StarFormationRate_and_weighted_SmZ_diff_with_Mass_coldens_h1ssh_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)

ea25RecSmZdiff_h1_hn   = npzldif(pdir + 'coldens_hneutralssh_and_weighted_SmZ_diff_with_Mass_StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)
ea25RecSmZdiff_mass_hn = npzldif(pdir + 'Mass_and_weighted_SmZ_diff_with_coldens_hneutralssh_StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)
ea25RecSmZdiff_sfr_hn  = npzldif(pdir + 'StarFormationRate_and_weighted_SmZ_diff_with_Mass_coldens_hneutralssh_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)

ea25Zmass_meascomp = npzldif(pdir + 'Mass_coldens_h1ssh_StarFormationRate_L0025N0376_19_test3.31_C2Sm_8000pix_6.25slice_zcen-all_z-projection_T4EOS_and_Metallicity_w_Mass.npz', 4 * 8000**2)

ea25RecZmeas_h1   = npzldif(pdir + 'coldens_h1ssh_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-h1ssh_StarFormationRate.npz', 8 * 10000**2)
ea25RecZmeas_mass = npzldif(pdir + 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-h1ssh_StarFormationRate.npz', 8 * 10000**2)
ea25RecZmeas_sfr  = npzldif(pdir + 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-h1ssh_StarFormationRate.npz', 8 * 10000**2)

ea25RecZmeas_hn_h1   = npzldif(pdir + 'coldens_hneutralssh_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-hneutralssh_StarFormationRate.npz', 8 * 10000**2)
ea25RecZmeas_hn_mass = npzldif(pdir + 'Mass_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-hneutralssh_StarFormationRate.npz', 8 * 10000**2)
ea25RecZmeas_hn_sfr  = npzldif(pdir + 'StarFormationRate_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection_T4EOS_and_SmoothedMetallicity_w_Mass_coldens-hneutralssh_StarFormationRate.npz', 8 * 10000**2)

ea25RecZmeas_basecomp = npzldif(pdir + 'Mass_h1ssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)
ea25RecZmeas_basecomp_hn = npzldif(pdir + 'Mass_hneutralssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection.npz', 8 * 10000**2)