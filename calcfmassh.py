# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:59:23 2017

@author: wijers

Calculate h1 fractions from Rahmati, Pawlik, Raicevic, Schaye 2013;
hmol fractions from Rahmati, Schaye, Pawlik, Raicevic 2013, 
citing Altay et al. 2011, Blitz and Rosolowsky 2006
using input arrays from EAGLE



Note decrease in goodness of fit to z=3 HI CDDF in 
Rahmati, Schaye, Pawlik, Raicevic 2013, where local stellar radiation is 
considered as well as the UVB and recombination radiation from the earlier 
paper 

"""

#### imports
import numpy as np


#### tables: DO NOT MESS WITH THIS

# Rahmati, Pawlik, Raicevic, Schaye 2013, table A1
fitparamtable_highz =\
{\
0. : {'logn0_cmm3': -2.94, 'alpha1': -3.98, 'alpha2': -1.09, 'beta': 1.29, 'onemf': 0.99},\
1. : {'logn0_cmm3': -2.29, 'alpha1': -2.94, 'alpha2': -0.90, 'beta': 1.21, 'onemf': 0.97},\
2. : {'logn0_cmm3': -2.06, 'alpha1': -2.22, 'alpha2': -1.09, 'beta': 1.75, 'onemf': 0.97},\
3. : {'logn0_cmm3': -2.13, 'alpha1': -1.99, 'alpha2': -0.88, 'beta': 1.72, 'onemf': 0.96},\
4. : {'logn0_cmm3': -2.23, 'alpha1': -2.05, 'alpha2': -0.75, 'beta': 1.93, 'onemf': 0.98},\
5. : {'logn0_cmm3': -2.35, 'alpha1': -2.63, 'alpha2': -0.57, 'beta': 1.77, 'onemf': 0.99},\
}

# Rahmati, Pawlik, Raicevic, Schaye 2013, table A2
fitparamtable_z0 =\
{'logn0_cmm3': -2.56, 'alpha1': -1.86, 'alpha2': -0.51, 'beta': 2.83, 'onemf': 0.99}

# Rahmati, Pawlik, Raicevic, Schaye 2013, table 2
# HM01 is what is used in the paper (and EAGLE), so use this one for self-consistency
phototables =\
{
'HM01':\
{
0.: {'UVB_s-1': 8.34e-14, 'sigmabar_nu_h1_cm2': 3.27e-18, 'Eeq_eV': 16.9, 'nH_ssh_cm-3': 1.1e-3},\
1.: {'UVB_s-1': 7.39e-13, 'sigmabar_nu_h1_cm2': 2.76e-18, 'Eeq_eV': 17.9, 'nH_ssh_cm-3': 5.1e-3},\
2.: {'UVB_s-1': 1.50e-12, 'sigmabar_nu_h1_cm2': 2.55e-18, 'Eeq_eV': 18.3, 'nH_ssh_cm-3': 8.7e-3},\
3.: {'UVB_s-1': 1.16e-12, 'sigmabar_nu_h1_cm2': 2.49e-18, 'Eeq_eV': 18.5, 'nH_ssh_cm-3': 7.4e-3},\
4.: {'UVB_s-1': 7.92e-13, 'sigmabar_nu_h1_cm2': 2.45e-18, 'Eeq_eV': 18.6, 'nH_ssh_cm-3': 5.8e-3},\
5.: {'UVB_s-1': 5.43e-13, 'sigmabar_nu_h1_cm2': 2.45e-18, 'Eeq_eV': 18.6, 'nH_ssh_cm-3': 4.5e-3},\
},\
'HM12':\
{
0.: {'UVB_s-1': 2.27e-14, 'sigmabar_nu_h1_cm2': 2.68e-18, 'Eeq_eV': 18.1, 'nH_ssh_cm-3': 5.1e-4},\
1.: {'UVB_s-1': 3.42e-13, 'sigmabar_nu_h1_cm2': 2.62e-18, 'Eeq_eV': 18.2, 'nH_ssh_cm-3': 3.3e-3},\
2.: {'UVB_s-1': 8.98e-13, 'sigmabar_nu_h1_cm2': 2.61e-18, 'Eeq_eV': 18.2, 'nH_ssh_cm-3': 6.1e-3},\
3.: {'UVB_s-1': 8.74e-13, 'sigmabar_nu_h1_cm2': 2.61e-18, 'Eeq_eV': 18.2, 'nH_ssh_cm-3': 6.0e-3},\
4.: {'UVB_s-1': 6.14e-13, 'sigmabar_nu_h1_cm2': 2.60e-18, 'Eeq_eV': 18.3, 'nH_ssh_cm-3': 4.7e-3},\
5.: {'UVB_s-1': 4.57e-13, 'sigmabar_nu_h1_cm2': 2.58e-18, 'Eeq_eV': 18.3, 'nH_ssh_cm-3': 3.9e-3},\
},\
'FG09':\
{
0.: {'UVB_s-1': 3.99e-14, 'sigmabar_nu_h1_cm2': 2.59e-18, 'Eeq_eV': 18.3, 'nH_ssh_cm-3': 7.7e-4},\
1.: {'UVB_s-1': 3.03e-13, 'sigmabar_nu_h1_cm2': 2.37e-18, 'Eeq_eV': 18.8, 'nH_ssh_cm-3': 3.1e-3},\
2.: {'UVB_s-1': 6.00e-13, 'sigmabar_nu_h1_cm2': 2.27e-18, 'Eeq_eV': 19.1, 'nH_ssh_cm-3': 5.1e-3},\
3.: {'UVB_s-1': 5.53e-13, 'sigmabar_nu_h1_cm2': 2.15e-18, 'Eeq_eV': 19.5, 'nH_ssh_cm-3': 5.0e-3},\
4.: {'UVB_s-1': 4.31e-13, 'sigmabar_nu_h1_cm2': 2.02e-18, 'Eeq_eV': 19.9, 'nH_ssh_cm-3': 4.4e-3},\
5.: {'UVB_s-1': 3.52e-13, 'sigmabar_nu_h1_cm2': 1.94e-18, 'Eeq_eV': 20.1, 'nH_ssh_cm-3': 4.0e-3},\
},\
}


#### functions

def linterp(x1, x2, y1, y2, xint):
    return ((xint - x1) * y2 + (x2 - xint) * y1) / (x2 - x1)
    
def linterptab(z1, z2, tab1, tab2, zint):
    '''
    will use tab1 to determine which keys to use
    '''
    return {key: linterp(z1, z2, tab1[key], tab2[key], zint) for key in tab1.keys()}
    
def ABCquadeq(dct_ABC):
    '''
    Rahmati, Pawlik, Raicevic, & Schaye 2013, eq. A8
    only one solution, dct_ABC entries will be changed
    '''
    A = dct_ABC['A']
    B = dct_ABC['B']
    C = dct_ABC['C']

    # C -> 4AC/B**2
    C *= 4.
    C *= A
    C /= B
    C /= B

    totaylor = 3e-4 
    # x for minimum relative difference of taylor and nominal calculations of 1 - sqrt(1 -x), at difference of ~6e-5 
    # done for float32 since that is what the EAGLE arrays used here are stored in
    out = np.empty(A.shape, dtype=np.float32)
    tempsel = np.abs(C) >= totaylor                                                   # 1 array 
    out[tempsel] = B[tempsel] / (2. * A[tempsel]) * (1. - np.sqrt(1. - C[tempsel]))# 1 array, 1 temp
    np.logical_not(tempsel, out=tempsel)
    out[tempsel] = B[tempsel] * C[tempsel] / (4. * A[tempsel])                   # 1 array, 1 temp # taylor: 1 - sqrt(1-x) ~ 1 - (1 - 1/2 x) = 1/2 x
    del tempsel                                                             # -1 array
    del A
    del B
    del C
    del dct_ABC
    return out
    
def getfitparams(z): 
    '''
    How to get fit parameters from the tables
    not optimised for speed; designed for use for a single redshift and many particles
    Uses larger-volume (table A2) parameters for z=0
    '''
    maxz = np.max(fitparamtable_highz.keys())
    
    if z < 1.:
        tab1 = fitparamtable_z0
        tab2 = fitparamtable_highz[1.]
        return linterptab(0., 1., tab1, tab2,z)
    elif z < maxz:
        zs = np.array(fitparamtable_highz.keys())
        z1 = np.max(zs[zs <= z])
        z2 = np.min(zs[zs > z])
        tab1 = fitparamtable_highz[z1]
        tab2 = fitparamtable_highz[z2]
        #print z1, z2, tab1, tab2
        return linterptab(z1,z2,tab1,tab2,z)
    else:
        if z != maxz:
            print('Warning: redshift %.2f outside interpolation range; using redshift %.2f values'%(z,maxz))
        return fitparamtable_highz[maxz]
        
def getUVBparams(z, UVB='HM01'):
    '''
    How to get UVB parameters from the tables
    not optimised for speed; designed for use for a single redshift and many particles
    '''
    
    phototable = phototables[UVB]
    zs = np.array(phototable.keys())
    maxz = np.max(zs)
    if z >= maxz:
        if z != maxz:
            print('Warning: redshift %.2f outside interpolation range; using redshift %.2f values'%(z,maxz))
        return phototable[maxz]
    else:
        z1 = np.max(zs[zs<=z])
        z2 = np.min(zs[zs>z])
        tab1 = phototable[z1]
        tab2 = phototable[z2]
        #print z1, z2, tab1, tab2
        return linterptab(z1, z2, tab1, tab2,z)
    
def Gammaphot_over_GammaUVB(dct_nH, z):
    '''
    Rahmati, Pawlik, Raicevic, Schaye 2013, equation A1:
    Gammaphot/GammaUVB = (1. -f)*(1 + (nH/n0)**beta)**alpha1 + f*(1 + nH/n0)**alpha2
    '''
    params = getfitparams(z)
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    beta   = params['beta']
    onemf  = params['onemf']
    n0     = 10**params['logn0_cmm3']
    #print alpha1, alpha2, beta, onemf, n0
    return onemf*(1. + (dct_nH['nH'] / n0)**beta)**alpha1 + (1. - onemf) * (1 + dct_nH['nH'] / n0)**alpha2

def GammaLSR(dct_nH_T):
    '''
    Rahmati, Schaye, Pawlik, Raicevic 2013, equation 7
    nH in particles/cm^3
    output units: s^-1
    Assumes an H mass fraction of 0.75 (equations 4,5), and fg/fth = 1
    following Ali's IDL implementation
    '''
    return 1.3e-13 * (dct_nH_T['nH'] * dct_nH_T['Temperature'] * 1e-4)**0.2      
    
def LambdaT(dct_T):
    '''
    Rahmati, Pawlik, Raicevic, Schaye 2013, equation A6, citing Theuns et al. 1998
    output units : cm3/s 
    input units: K?
    '''
    return 1.17e-10 * dct_T['Temperature']**0.5 * np.exp(-157809. / dct_T['Temperature']) / (1. + np.sqrt(dct_T['Temperature'] * 1.0e-5))

def alphaA(dct_T):
    '''
    Rahmati, Pawlik, Raicevic, Schaye 2013, equation A3, citing Hui en Gnedin 1997
    output units : cm3/s 
    input units: K?
    '''
    lambdapar = 315614. / dct_T['Temperature']
    return 1.269e-13 * lambdapar**1.503 / (1. + (lambdapar / 0.522)**0.47)**1.923 
    
def nHIHmol_over_nH(dct_nH_T, z, UVB='HM01', useLSR=False):
    '''
    HM01: used in EAGLE
    LSR: local stellar radiation (analytical model, not based on stars in simulation)
    Rahmati, Pawlik, Schaye, Raicevic equation A8, eta in text above A4
    does not account for molecular hydrogen, so I will assume this also reflects 
    the mass density ratios (i.e., that some of the HI combining into H2 does
    not cause more HI to form; seems reasonable in that this fraction would be 
    close to one before H2 forms in large amounts anyway)
    '''
    # a little bit messy, but saves memory to do things this way
    C = alphaA(dct_nH_T)                                    # sets 1 array, 1 temp
    A = LambdaT(dct_nH_T)                                   # sets 1 array, 2 temp
    UVB = getUVBparams(z, UVB = UVB)['UVB_s-1']
    if useLSR:
        B = (Gammaphot_over_GammaUVB(dct_nH_T,z) * UVB + GammaLSR(dct_nH_T)) / dct_nH_T['nH']         # sets 1 array, 1 temp
    else:
        B = Gammaphot_over_GammaUVB(dct_nH_T,z) * UVB / dct_nH_T['nH']
    B += 2*C + A
    A += C
    h1hmolfrac = ABCquadeq({'A': A, 'B': B, 'C': C}) # 1 temp in function
    del A
    del B
    del C
    h1hmolfrac[h1hmolfrac > 1.] = 1. # handle numerical nHI > nH situations # 1 temp 
    h1hmolfrac[h1hmolfrac < 0.] = 0. # handle numerical nHI < 0 situations
    return h1hmolfrac

def rhoHmol_over_rhoH(dct_nH_T_eos, EOS='eagle'):
    '''
    Rahmati, Schaye, Pawlik, Raicevic 2013, equation A4
    citing Altay et al. 2011 good obs. fit, Blitz and Rosolowsky 2006 scaling
    and OWLS EOS gammaeff (equal to EAGLE value)
    nH:   hydrogen number density in particles/cm^3
    eos:  selection criterion for applying the equation of state 
          (bool array or slice)
    T:    temperature in K; dictionary key is 'Temperature'
    '''
    # same EOS gamma_eff checked in EAGLE output hdf5 file
    # EAGLE EOS: 8000 K at 0.1 cm^-3, then T = 8000.*( nH / 0.1 cm^-3)**(1./3.) at higher nH
    if EOS == 'owls': # assumes gas is on the EOS
        return 1. / (1. + 24.54 * (dct_nH_T_eos['nH'] / 0.1)**-1.23) # 1 temp
    elif EOS == 'eagle':
        nstar = 0.1 # cm^-3
        Pfloor_nH = 0.1 * 8.0e3 # K cm^-3
        Pfloor_ntot = (1 + 0.248 * (0.752**-1 -1.)) * Pfloor_nH # assuming only H/He, f_H_mass = 0.752, K cm^-3
        gamma_eos = 4. / 3.
        # Blitz and Rosolowsky 2006
        alpha = 0.92
        P0    = 3.5e4 # K cm^-3
        C1 = (Pfloor_ntot / P0)**(-1 * alpha) # (Pfloor_nH / P0)**(-1 * alpha) # (Pfloor_nH / P0)**(alpha) # 
        
        # calculate the pressure: n * T for most gas, EAGLE EOS for star-forming gas
        Pgas_nH = dct_nH_T_eos['Temperature'] * dct_nH_T_eos['nH']
        Pmin_nH = Pfloor_nH * (dct_nH_T_eos['nH'][dct_nH_T_eos['eos']] / nstar)**gamma_eos
        #Pmin_nH = Pfloor_nH * (dct_nH_T_eos['nH'][dct_nH_T_eos['nH'] > nstar] / nstar)**gamma_eos
        #Pgas_nH[dct_nH_T_eos['eos']] = np.maximum(Pmin_nH, Pgas_nH[dct_nH_T_eos['eos']]) # impose EAGLE pressure floor: important if using 10^4 K for EOS gas
        Pgas_nH[dct_nH_T_eos['eos']] = Pmin_nH
        #Pgas_nH[dct_nH_T_eos['nH'] > nstar] = Pmin_nH
        #del impose_eos
        del Pmin_nH
        # assuming fH is never too far from the primordial value, the ntot/nH values in the P/Pfloor ratio cancel out
        # otherwise, rho / mu_primordial is used in the simulation itself so pure n_H isn't the way to go anyway
        return (1. + (Pgas_nH / Pfloor_nH)**(-1 * alpha) * C1)**-1
        #return (Pgas_nH / Pfloor_nH)**(alpha) * C1
    else:
        raise ValueError('rhoHmol_over_rhoH: EOS must be "eagle" or "owls"')
    


#### tests

def testtotaylorparameters():
    import matplotlib.pyplot as plt
    
    xs = 10**np.arange(-10.,0.,0.01,dtype=np.float32)
    nominal = 1. - np.sqrt(1. - xs)
    taylor = 0.5*xs
    
    plt.plot(xs,np.abs(nominal / taylor -1.))
    plt.xscale('log')    
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('nominal/taylor - 1')
    plt.title('Comparison between 1 - sqrt(1-x) and its Taylor approximation at x=1: 0.5x')    
    
    plt.show()

def plotHIfracs(T=1e4,z=3.):
    '''
    using 10^4 K is an assumption in the plot comparison
    
    compare left panel to e.g. left panel of fig. 3, Rahmati, Pawlik,
    Raicevic, & Schaye 2013 (z=3., no LSR)
    
    compare right panel to e.g. fig. 6 upper left panel in that work
    '''
    import matplotlib.pyplot as plt
    fontsize = 12
    
    nHs = 10**np.arange(-10.,4.,0.01)
    Ts = T*np.ones(nHs.shape)
    dct_nH_T = {'nH': nHs, 'Temperature': Ts}
    
    plt.subplot(121)
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, z, UVB='HM01', useLSR=False)), label='HM01', color='blue')
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, z, UVB='HM12', useLSR=False)), label='HM12', color='green')    
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, z, UVB='FG09', useLSR=False)), label='FG09', color='red')    
    plt.xlim((-6., 2.))
    plt.ylim(-6.1, 0.2)
    plt.xlabel(r'$\log_{10} n_H \, [\mathrm{cm^{-3}}]$', fontsize=fontsize)
    plt.ylabel(r'$\log_{10} (n_{\mathrm{HI}} + n_{\mathrm{H}_{2}})\, /\, n_H $', fontsize=fontsize)
    plt.title('HI and Hmol fraction of hydrogen for different UVB models')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, 5., UVB='HM01', useLSR=False)), label=r'$z=5$', color='maroon', linestyle='solid')
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, 4., UVB='HM01', useLSR=False)), label=r'$z=4$', color='red', linestyle='dashed') 
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, 3., UVB='HM01', useLSR=False)), label=r'$z=3$', color='brown', linestyle='dashed')  
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, 2., UVB='HM01', useLSR=False)), label=r'$z=2$', color='teal', linestyle='dashdot')
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, 1., UVB='HM01', useLSR=False)), label=r'$z=1$', color='darkblue', linestyle='dashdot')
    plt.plot(np.log10(nHs), np.log10(nHIHmol_over_nH(dct_nH_T, 0., UVB='HM01', useLSR=False)), label=r'$z=0$', color='lightcoral', linestyle='dotted') 
    plt.xlim((-5.1, 0.2))
    plt.ylim((-8.1, 0.25))
    plt.xlabel(r'$\log_{10} n_H \, [\mathrm{cm^{-3}}]$', fontsize=fontsize)
    plt.ylabel(r'$\log_{10} (n_{\mathrm{HI}} + n_{\mathrm{H}_{2}})\, /\, n_H $', fontsize=fontsize)
    plt.title('HI and Hmol fraction of hydrogen at different z for HM01 UVB')
    plt.legend()
    
    plt.show()