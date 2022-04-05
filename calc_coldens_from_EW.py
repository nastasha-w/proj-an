#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# for the second paper with Sarah Walsh 
# (full simulation through Athena observation modelling)

import ion_line_data as ild
import eagle_constants_and_units as c
import numpy as np
import scipy.optimize as spo

# wijers_etal_2020 best-fit values (Table 4, Fig. 7), 
# bracketing values at EW ~ 0.1, 0.2 eV from Fig. 7
ions = [ild.o7major, ild.o8doublet, ild.ne9major, ild.fe17major]
EWs_eV = [0.1, 0.2]
z = 0.1

bpars_bestfit = {'o7': 83e5,
                 'o8': 112e5,
                 'ne9': 82e5,
                 'fe17': 92e5,
                }
bpars_bracket = [50e5, 200e5]

headers = ['EW_eV_obs', 'ion', 'wavelength_A_rest', 'EW_mA_rest', 'b_kmps', 'N_logcm**-2']
fillstr = '{EW_eV:.2f}\t{ion:4}\t{wavelength_A:.2f}\t{EW_mA:.3f}\t' + \
          '{b_kmps:.0f}\t{N_logcmm2:.2f}'
print('redshift: {z:.2f}'.format(z=z))
print('\t'.join(headers))
for ion in ions:
    if hasattr(ion, 'speclines'):
        comps = ion.speclines
        ionnames = [comps[key].ion for key in comps]
        if np.all([ion == ionnames[0] for ion in ionnames]):
            ionname = ionnames[0]
        else:
            msg = 'line set {} contained different ions: {}'
            raise RuntimeError(msg.format(ion, ionnames))
        # need a single wavelength for the EW conversion 
        # using oscillator strength weighted values
        wls = np.array([comps[key].lambda_angstrom for key in comps])
        fos = np.array([comps[key].fosc for key in comps])
        wavelength_A = np.sum(wls * fos) / np.sum(fos)
    else:
        ionname = ion.ion
        wavelength_A = ion.lambda_angstrom
    for _EW in EWs_eV:
        _EW_A = _EW / (1. + z) * c.ev_to_erg * \
                (wavelength_A * 1e-8)**2 \
                / (c.planck * c.c) * 1e8
        bpars = [bpars_bestfit[ionname]] + bpars_bracket
        for bpar_cmps in bpars:
            logNion_init = 15.5
            def lossfunc(logNion):
                _EW_ = ild.linflatdampedcurveofgrowth_inv(10**logNion, 
                                                          bpar_cmps, ion)
                return (_EW_ - _EW_A)**2
            optres = spo.minimize(lossfunc, x0=logNion_init, method='COBYLA', 
                                  tol=1e-2, options={'rhobeg': 0.5})
            if optres.success:
                Nfit = optres.x
                #print('Best fit for {ion}: {fit}'.format(ion=ion, fit=Nfit))
            else:
                print('N_ion fitting failed:')
                print(optres.message)
                print(optres)
                Nfit = np.NaN
            fillfmt = {'EW_eV': _EW,
                       'ion': ionname,
                       'wavelength_A': wavelength_A,
                       'EW_mA': _EW_A * 1e3,
                       'b_kmps': bpar_cmps * 1e-5,
                       'N_logcmm2': Nfit,
                       }
            print(fillstr.format(**fillfmt))
        print('')

