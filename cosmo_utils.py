# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:43:18 2017

@author: wijers

General cosmological utility functions; initially copied from make_maps to be
loaded without the entire read_Eagle machinery
"""

import numpy as np
import ctypes as ct
import h5py
import numbers as num # for instance checking

import make_maps_opts_locs as ol # needed for some ion data
import eagle_constants_and_units as c

def findemtables(element,zcalc):
    
    #### checks and setup
    
    if not element in ol.elements:
        print("There will be an error somewhere: %s is not included or misspelled. \n" % element)
    
    if zcalc < 0. and zcalc > 1e-4:
        zcalc = 0.0
        zname = ol.zopts[0]
        interp = False

    elif zcalc in ol.zpoints:
        # only need one table
        zname = ol.zopts[ol.zpoints.index(zcalc)]
        interp = False
    
    elif zcalc <= ol.zpoints[-1]:
        # linear interpolation between two tables
        zarray = np.asarray(ol.zpoints)
        zname1 = ol.zopts[len(zarray[zarray<zcalc])-1]
        zname2 = ol.zopts[-len(zarray[zarray>zcalc])]
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n") 
        
    
    #### read in the tables; interpolate tables in z if needed and possible
    
    if not interp:
        tablefilename = ol.dir_emtab + zname + '/Tables/' + element + '.hdf5'
        tablefile = h5py.File(tablefilename, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK =   np.array(tablefile.get('logt'),dtype=np.float32)  
        logrhocm3 =   np.array(tablefile.get('logd'),dtype=np.float32)
        lines =   np.array(tablefile.get('lines'),dtype=np.float32)  
        
        
        tablefile.close()
    
    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        tablefilename1 = ol.dir_emtab + zname1 + '/Tables/' + element + '.hdf5'
        tablefile1 = h5py.File(tablefilename1, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK1 =   np.array(tablefile1.get('logt'),dtype=np.float32)  
        logrhocm31 =   np.array(tablefile1.get('logd'),dtype=np.float32)
        lines1 =   np.array(tablefile1.get('lines'),dtype=np.float32) 
        
        tablefile1.close()
        
        tablefilename2 = ol.dir_emtab + zname2 + '/Tables/' + element + '.hdf5'
        tablefile2 = h5py.File(tablefilename2, "r")
        #energies = np.array(tablefile.get('header/spectrum/logenergy_ryd'))
        #fluxes = np.array(tablefile.get('header/spectrum/logflux'))
        logTK2 =   np.array(tablefile2.get('logt'),dtype=np.float32)  
        logrhocm32 =   np.array(tablefile2.get('logd'),dtype=np.float32)
        lines2 =   np.array(tablefile2.get('lines'),dtype=np.float32) 
        
        tablefile2.close()
        
        if (np.all(logTK1 == logTK2) and np.all(logrhocm31 == logrhocm32)):
            print("interpolating 2 emission tables")
            lines = 1./(float(zname2)-float(zname1)) * ( (float(zname2)-zcalc)*lines1 + (zcalc-float(zname1))*lines2 )
            logTK = logTK1
            logrhocm3 = logrhocm31
        else: 
            print("Temperature and density ranges of the two interpolation z tables don't match. \n")
            print("Using nearest z table in stead.")
            if abs(zcalc - float(zname1)) < abs(zcalc - float(zname2)):
                logTK = logTK1
                logrhocm3 = logrhocm31
                lines = lines1
            else:
                logTK = logTK2
                logrhocm3 = logrhocm32
                lines = lines2
    
    return lines, logTK, logrhocm3

           
# calculate emission using C function (interpolator)
def find_emdenssq(z,elt,lognH,logT,lineind):

    p_emtable, logTK, lognHcm3 = findemtables(elt,z)
    emtable = p_emtable[:,:,lineind]
    NumPart = len(lognH)
    inlogemission = np.zeros(NumPart,dtype=np.float32)
    
    if len(logT) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = ol.c_interpfile

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_emdenssq

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

    # argument conversion

    res = interpfunction(lognH.astype(np.float32),\
               logT.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(emtable.astype(np.float32)),\
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               inlogemission \
              )
    
    print("-------------- C interpolation function output finished ----------------------\n")
    
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None
        
    return inlogemission



def findiontables(ion,z):
    # README in dir_iontab:
    # files are hdf5, contain ionisation fraction of a species for rho, T, z
    
    
    #### checks and setup
    
    if not ion in ol.ions:
        print("There will be an error somewhere: %s is not included or misspelled. \n" % ion)
    
    tablefilename = ol.dir_iontab + ion + '.hdf5'      
    tablefile = h5py.File(tablefilename, "r")
    logTK =   np.array(tablefile.get('logt'),dtype=np.float32)  
    lognHcm3 =   np.array(tablefile.get('logd'),dtype=np.float32)
    ziopts = np.array(tablefile.get('redshift'),dtype=np.float32) # monotonically incresing, first entry is zero
    balance_d_t_z = np.array(tablefile.get('ionbal'),dtype=np.float32)
    tablefile.close()

    if z < 0. and z > 1e-4:
        z = 0.0
        zind = 0 
        interp = False
        
    elif z in ziopts:
        # only need one table
        zind = np.argwhere(z == ziopts)
        interp = False
    
    elif z <= ziopts[-1]:
        # linear interpolation between two tables
        zind1 = np.sum(ziopts<z)-1
        zind2 = -sum(ziopts>z)
        interp = True
    else:
        print("Chosen z value requires extrapolation. This has not been implemented. \n") 
        
    
    #### read in the tables; interpolate tables in z if needed and possible
    
    if not interp:
       balance = np.squeeze(balance_d_t_z[:,:,zind]) # for some reason, extra dimensions are tacked on 
    
    if interp: #linear interpolation: 1./(a1-a0) * ( (a1-a)*f0 + (a-a0)*f1 )
        balance1 = balance_d_t_z[:,:,zind1]
        balance2 = balance_d_t_z[:,:,zind2]
        
        print("interpolating 2 emission tables")
        balance = 1./( ziopts[zind2] - ziopts[zind1]) * ( (ziopts[zind2]-z)*balance1 + (z-ziopts[zind1])*balance2 )

    
    return balance, logTK, lognHcm3

def find_ionbal(z,ion,lognH,logT):
    
    # compared to the line emission files, the order of the nH, T indices in the balance tables is switched
    balance, logTK, lognHcm3 = findiontables(ion,z) #(np.array([[0.,0.],[0.,1.],[0.,2.]]), np.array([0.,1.,2.]), np.array([0.,1.]) ) 
    NumPart = len(lognH)
    inbalance = np.zeros(NumPart,dtype=np.float32)
    
    if len(logT) != NumPart:
        print('logrho and logT should have the same length')
        return None

    # need to compile with some extra options to get this to work: make -f make_emission_only
    print("------------------- C interpolation function output --------------------------\n")
    cfile = '/home/wijers/plot_sims/make_maps_emission_coupled/HsmlAndProject/emission.so'

    acfile = ct.CDLL(cfile)
    interpfunction = acfile.interpolate_emdenssq # just a linear interpolator; works for non-emission stuff too

    interpfunction.argtypes = [np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,)),\
                           ct.c_longlong , \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK)*len(lognHcm3),)), \
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(logTK),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(len(lognHcm3),)), \
                           ct.c_int,\
                           np.ctypeslib.ndpointer(dtype=ct.c_float, shape=(NumPart,))]

 
    res = interpfunction(logT.astype(np.float32),\
               lognH.astype(np.float32),\
               ct.c_longlong(NumPart),\
               np.ndarray.flatten(balance.astype(np.float32)),\
               logTK.astype(np.float32),\
               ct.c_int(len(logTK)), \
               lognHcm3.astype(np.float32),\
               ct.c_int(len(lognHcm3)),\
               inbalance \
              )

    print("-------------- C interpolation function output finished ----------------------\n")
    
    if res != 0:
        print('Something has gone wrong in the C function: output %s. \n',str(res))
        return None
        
    return inbalance


# cosmological basics


def comoving_distance_cm(z, cosmopars=None): # assumes Omega_k = 0
    if z < 1e-8:
        print('Using 0 comoving distance from z. \n')
        return 0.
    if cosmopars is None:
        # use cosmological parameters for EAGLE from its constants and units file
        hpar = c.hubbleparam 
        omega0 = c.omega0
        omegalambda = c.omegalambda
    else:
        # use cosmological parameters for the simulation file (read in in Simfile.__init__ from hdf5 output files)
        hpar = cosmopars['h'] # use hubble parameter from the simulation; c.hubble is 100 km/s/Mpc 
        z = cosmopars['z']    # override input z by the value for the used snapshot
        omega0 = cosmopars['omegam']
        omegalambda = cosmopars['omegalambda']
    
    def integrand(zi):
        return (omega0*(1.+zi)**3 + omegalambda)**0.5
    zi_arr = np.arange(0,z+z/512.,z/512.)
    com = np.trapz(1./integrand(zi_arr),x=zi_arr)
    return com * c.c/(c.hubble*hpar)

def ang_diam_distance_cm(z):
    return comoving_distance_cm(z)/(1.+z)

def lum_distance_cm(z):
    return comoving_distance_cm(z)*(1.+z)

def Hubble(z, cosmopars=None):
    if cosmopars is None:
        # use cosmological parameters for EAGLE from its constants and units file
        hpar = c.hubbleparam 
        omega0 = c.omega0
        omegalambda = c.omegalambda
    else:
        # use cosmological parameters for the simulation file (read in in Simfile.__init__ from hdf5 output files)
        hpar = cosmopars['h'] # use hubble parameter from the simulation; c.hubble is 100 km/s/Mpc 
        z = cosmopars['z']    # override input z by the value for the used snapshot
        omega0 = cosmopars['omegam']
        omegalambda = cosmopars['omegalambda']
        
    return (c.hubble * hpar) * (omega0 * (1. + z)**3 + omegalambda)**0.5

def rhocrit(z, cosmopars=None):
    '''
    critical density at z; units: g / cm^-3
    cosmopars z overrides input z if cosmopars are given
    '''
    rhoc = 3. / (8. * np.pi * c.gravity) * (Hubble(z, cosmopars=cosmopars))**2
    return rhoc

def rhom(z, cosmopars=None):
    '''
    mean matter (DM + baryons) density at z; units: g / cm^-3
    cosmopars z overrides input z if cosmopars are given 
    '''
    if cosmopars is None:
        rhoc0 = rhocrit(0., None)
        omegam0 = c.omega0
        _z = z
    else:
        cp = cosmopars.copy()
        cp['z'] = 0.
        cp['a'] = 1.
        rhoc0 = rhocrit(0., cosmopars=cp)
        omegam0 = cosmopars['omegam']
        _z = cosmopars['z']
    rhom = rhoc0 * omegam0 * (1. + _z)**3
    return rhom

def conc_mass_MS15(Mh, cosmopars=None):
    '''
    Schaller et al. 2015 Eagle concentration-mass relation: 
    DM in full hydro Eagle fit
    Note: fit is for z=0, so avoid at high z!!
    '''
    if cosmopars is None:
        hpar = c.hubbleparam
    else:
        hpar = cosmopars['h']
    
    return 5.699 * (Mh / (1e14 * hpar * c.solar_mass))**-0.074

def rho_NFW(r, Mh, delta=200, ref='rhocrit', z=0., cosmopars=None, c='Schaller15'):
    '''
    returns: density (g /cm^3)
    Mh: halo mass (g)
    r: cm, physical
    delta: overdensity threshold
    c: concentration - number or 'Schaller15' for that relation (z=0 fit)
    ref: reference density for delta ('rhocrit' or 'rhom')
    '''
    if c == 'Schaller15':
        c = conc_mass_MS15(Mh, cosmopars=cosmopars)
    elif not isinstance(c, num.Number):
        raise ValueError('Value %s for c is not a valid option'%(c))
    if ref == 'rhocrit':
        rho_ref = rhocrit(z, cosmopars=cosmopars)
    elif ref == 'rhom':
        rho_ref = rhom(z, cosmopars=cosmopars)
    else:
        raise ValueError('Value %s for ref is not a valid option'%(ref))
    Redge = (Mh * 3. / (4. * np.pi * delta * rho_ref)) ** (1. / 3.)
    rnorm = c * r / Redge
    rhoval = (delta * rho_ref * c**3 ) / \
             (3. * (np.log(1. + c) - c / (1. + c)) * rnorm * (1. + rnorm)**2)
    return rhoval

def Rhalo(Mh, delta=200, ref='rhocrit', z=0., cosmopars=None):
    if ref == 'rhocrit':
        rho_ref = rhocrit(z, cosmopars=cosmopars)
    elif ref == 'rhom':
        rho_ref = rhom(z, cosmopars=cosmopars)
    else:
        raise ValueError('Value %s for ref is not a valid option'%(ref))
    Redge = (Mh * 3. / (4. * np.pi * delta * rho_ref)) ** (1. / 3.)
    return Redge

def Tvir_hot(Mh, delta=200, ref='rhocrit', z=0., cosmopars=None):
    mu = 0.59 # about right for ionised (hot) gas, primordial
    Rh = Rhalo(Mh, delta=delta, ref=ref, z=z, cosmopars=cosmopars)
    return (mu * c.protonmass) / (3. * c.boltzmann) * c.gravity * Mh / Rh

def solidangle(alpha,beta): # alpha = 0.5 * pix_length_1/D_A, beta = 0.5 * pix_length_2/D_A
    #from www.mpia.de/~mathar/public/mathar20051002.pdf
    # citing  A. Khadjavi, J. Opt. Soc. Am. 58, 1417 (1968).
    # stored in home/papers
    # using the exact formula, with alpha = beta, 
    # the python exact formula gives zero for alpha = beta < 10^-3--10^-4
    # assuming the pixel sizes are not so odd that the exact formula is needed in one direction and gives zero for the other,
    # use the Taylor expansion to order 4 
    # testing the difference between the Taylor and python exact calculations shows that 
    # for square pixels, 10**-2.5 is a reasonable cut-off
    # for rectangular pixels, the cut-off seems to be needed in both values
    if alpha < 10**-2.5 or beta <10**-2.5:
        return 4*alpha*beta - 2*alpha*beta*(alpha**2+beta**2)
    else: 
        return 4*np.arccos(((1+alpha**2 +beta**2)/((1+alpha**2)*(1+beta**2)))**0.5)

def Tvir(m200c, cosmopars='eagle', mu=0.59, z=None):
    '''
    input: 
    m200c in solar masses
    z directly or via cosmopars (cosmopars 'wins')
    output:
    Tvir in K       
    '''
    # formula from LSS + Gal. Form. notes, page 35
    # mu = means molecular weight / hydrogen mass. 0.59 is for primordial, fully ionised, gas
    if cosmopars == 'eagle':
        h = c.hubbleparam
        if z is None:
            raise ValueError('Some value of z is needed if cosmopars are not supplied.')
    else:
        h = cosmopars['h']
        z = cosmopars['z']
    return 4.1e5 * (mu/0.59) * (m200c/(1e12/h))**(2./3.) * (1.+z)


def R200c_pkpc(M200c, cosmopars):
    '''
    M200c: solar masses
    '''
    M200c *= c.solar_mass # to cgs
    rhoc = (3. / (8. * np.pi * c.gravity) * Hubble(cosmopars['z'], cosmopars=cosmopars)**2) # Hubble(z) will assume an EAGLE cosmology
    R200c = (M200c / (200. * rhoc))**(1./3.)
    return R200c / c.cm_per_mpc * 1e3 

def Teos_eagle(rho):
    '''
    rho = density (g / cm^-3)
    '''
    X = 0.752 # (primodial also used in code, but for T <-> entropy)

    nstar = 0.1 # cm^-3
    Tstar = 8.0e3 # K
    gamma_eos = 4. / 3.
    rhostar = nstar * c.atomw_H * c.u / X
    
    Teos = Tstar * (rho / rhostar)**(gamma_eos - 1.)
    return Teos
        
def hasSFR_eagle(rho, T, Z, cosmopars):
    '''
    rho = density (g / cm^-3)
    Z = metallcitity (mass fraction)
    T = temperature (K) !! temperature including EOS, not imposed 10^4 K !!
    
    Eagle paper (Schaye et al. 2015), section 4.3
    
    seems to mostly match recorded SFR values when using particle Z values
    '''
    X = 0.752
    nH = X * rho / (c.atomw_H * c.u)

    nHmin = 0.1 * (Z / 0.002)**-0.64
    nHmin = np.minimum(10., nHmin)
    nHmin = np.maximum(nHmin, 57.7 * rhocrit(cosmopars['z'], cosmopars=cosmopars) * cosmopars['omegab']) 
    
    Tmax = 10**0.5 * Teos_eagle(rho) # 0.5 dex above EOS
    
    hassfr = np.logical_and(nH >= nHmin, T <= Tmax)
    return hassfr 
    
def SFR_eagle(pm, T, rho, Z, cosmopars):
    '''
    !! Doesn't work very well. Possibly has bugs/errors !!
    pm = particle mass array (g)
    T = temperature (K) !! temperature including EOS, not imposed 10^4 K !!
    rho = density (g / cm^-3)
    
    Eagle paper (Schaye et al. 2015), section 4.3
    
    values seem systematically somewhat below recorded SFR, but differences
    exist both ways
    '''
    X = 0.752
    #mu_hot = 0.59 # about right for ionised (hot) gas, primordial
    mu_neutral = (X + (1. - X) * c.atomw_H / c.atomw_He)**-1 # probaly appropriate for cool gas; used for EOS pressure floor entropy conversion
    gamma = 5. / 3.
    f_g = 1.
    A = 1.515e-4 * c.solar_mass / c.sec_per_year / (c.cm_per_mpc * 1e-3)**2
    nlo = 1.4
    nhi = 2.
    nHpiv = 1e3
    
    nH = X * rho / (c.atomw_H * c.u)
    # pressure would probably be the pressure-entropy SPH pressure durin the run, but that is not easily available here
    # P = n kb T (ideal gas law), n = rho / (mu * mH)
    P = c.boltzmann * T * rho / (mu_neutral * c.atomw_H * c.u)
    
    # SFR = m_g * A * (M_sun / pc^2)^n * (gamma / G * f_g * P) * (n - 1) / 2
            # gamma = 5/3, G = newton constant, f_g = 1 (gas fraction), P = total pressure
            # A = 1.515 × 10−4 M⊙ yr−1 kpc−2, n = 1.4 (n = 2 at nH > 10^3 cm^-3)
    SFR = pm * A * (c.solar_mass / (c.cm_per_mpc * 1e-6)**2)**(-1. * nlo) * (gamma / c.gravity * f_g * P)**(0.5 * nlo - 0.5)
    SFR = np.array(SFR)
    hisel = nH > nHpiv
    SFR_hi = pm * A * (c.solar_mass / (c.cm_per_mpc * 1e-6)**2)**(-1. * nhi) * (gamma / c.gravity * f_g * P)**(0.5 * nhi - 0.5)
    SFR[hisel] = SFR_hi[hisel]
    
    hassfr = hasSFR_eagle(rho, T, Z, cosmopars)
    SFR[np.logical_not(hassfr)] = 0.
    return SFR[()] # return a single value is floats were input, otherwise the array



# John Helly's routine, via Peter
def match(arr1, arr2, arr2_sorted=False, arr2_index=None):
    """
    For each element in arr1 return the index of the element with the
    same value in arr2, or -1 if there is no element with the same value.
    Setting arr2_sorted=True will save some time if arr2 is already sorted
    into ascending order.

    A precomputed sorting index for arr2 can be supplied using the
    arr2_index parameter. This can save time if the routine is called
    repeatedly with the same arr2 but arr2 is not already sorted.

    It is assumed that each element in arr1 only occurs once in arr2.
    """

    # Workaround for a numpy bug (<=1.4): ensure arrays are native endian
    # because searchsorted ignores endian flag
    if not(arr1.dtype.isnative):
        arr1_n = np.asarray(arr1, dtype=arr1.dtype.newbyteorder("="))
    else:
        arr1_n = arr1
    if not(arr2.dtype.isnative):
        arr2_n = np.asarray(arr2, dtype=arr2.dtype.newbyteorder("="))
    else:
        arr2_n = arr2

    # Sort arr2 into ascending order if necessary
    tmp1 = arr1_n
    if arr2_sorted:
        tmp2 = arr2_n
        idx = slice(0,len(arr2_n))
    else:
        if arr2_index is None:
            idx = np.argsort(arr2_n)
            tmp2 = arr2_n[idx]
        else:
            # Use supplied sorting index
            idx = arr2_index
            tmp2 = arr2_n[arr2_index]

    # Find where elements of arr1 are in arr2
    ptr  = np.searchsorted(tmp2, tmp1)

    # Make sure all elements in ptr are valid indexes into tmp2
    # (any out of range entries won't match so they'll get set to -1
    # in the next bit)
    ptr[ptr>=len(tmp2)] = 0
    ptr[ptr<0]          = 0

    # Return -1 where no match is found
    ind  = tmp2[ptr] != tmp1
    ptr[ind] = -1

    # Put ptr back into original order
    ind = np.arange(len(arr2_n))[idx]
    ptr = np.where(ptr>= 0, ind[ptr], -1)
    
    return ptr

def getdX(redshift,L_z,cosmopars=None):
    # assuming L_z is smaller than the distance over which H varies significantly; 
    # assumed in single-snapshot projection anyway 
    if cosmopars is not None:
        redshift = cosmopars['z']
        hpar = cosmopars['h']
    else:     
        hpar = c.hubbleparam
    dz = Hubble(redshift, cosmopars=cosmopars) / c.c * L_z * c.cm_per_mpc
    dX = dz * (1+redshift)**2 * c.hubble * hpar / Hubble(redshift, cosmopars=cosmopars) 
    return dX


def getdz(redshift,L_z,cosmopars=None):
    # assuming L_z is smaller than the distance over which H varies significantly; 
    # assumed in single-snapshot projection anyway 
    if cosmopars is not None:
        redshift = cosmopars['z']
    dz = Hubble(redshift, cosmopars=cosmopars) / c.c * L_z * c.cm_per_mpc
    return dz