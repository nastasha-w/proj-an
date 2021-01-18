# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:30:46 2017

@author: wijers

To read in files from the online Eagle halo catalogue
"""
import eagleSqlTools as sql
import numpy as np
import h5py
import matplotlib.pyplot as plt

import make_maps_opts_locs as ol
import projection_classes as pc
import eagle_constants_and_units as c

#import cosmo_utils as csu

pdir = ol.pdir

con = sql.connect('wdf201', password='LZJ687pn')
# old; don't work anymore 'nwijers', password='G3Wuve94'
entries = ['groupid','m200c_solar','r200c_pkpc','xcop_cmpc','ycop_cmpc','zcop_cmpc']

def getvar(var):
    if var in ['REFERENCE','REF','Ref','ref','',None,'Reference','reference']:
        varind = 'Ref'
    elif var in ['Recalibrated','RECALIBRATED','recalibrated','Recal','RECAL','recal','Rec','rec','REC']:
        varind = 'Recal'
    elif var in ['S15_AGNdT9','AGNdT9']: # the default in the queries in Joop's fully calibrated AGNdT9 model
        varind = 'AGNdT9'
    else:
        varind = var
    return varind

def getvar_re(var):
    if var == 'Ref':
        var = 'REFERENCE'
    elif var == 'Recal':
        var = 'RECALIBRATED'
    elif var == 'AGNdT9':
        var = 'S15_AGNdT9'
    return var

def generatequery(simnum,snapnum,Mmin=0.,var = 'REFERENCE'):
    # indicators are sometimes different from the var in make_maps and Cosma directory names
    # set a few so the more familiar var options get the right simulations
    varind = getvar(var)
    return  "SELECT\
        FOF.GroupID as groupid,\
        FOF.Group_M_Crit200 as m200c_solar,\
        FOF.Group_R_Crit200 as r200c_pkpc,\
        FOF.GroupCentreOfPotential_x as xcop_cmpc,\
        FOF.GroupCentreOfPotential_y as ycop_cmpc,\
        FOF.GroupCentreOfPotential_z as zcop_cmpc\
        FROM\
        %s%s_FOF as FOF\
        WHERE\
        FOF.SNAPNUM=%i and \
        FOF.Group_M_Crit200 >%f\
        ORDER BY\
        m200c_solar"%(varind,simnum,snapnum,Mmin)
        
def generatequery_centralproperties(simnum, snapnum, Mhmin=0., var='REFERENCE', apsize=30):
    # indicators are sometimes different from the var in make_maps and Cosma directory names
    # set a few so the more familiar var options get the right simulations
    varind = getvar(var)
    return  "SELECT\
        FOF.GroupID as groupid,\
        FOF.Group_M_Crit200 as M200c_Msun,\
        FOF.Group_R_Crit200 as R200c_pkpc,\
        FOF.GroupCentreOfPotential_x as Xgroupcop_cMpc,\
        FOF.GroupCentreOfPotential_y as Ygroupcop_cMpc,\
        FOF.GroupCentreOfPotential_z as Zgroupcop_cMpc,\
        SH.GalaxyID as galaxyid,\
        SH.CentreOfPotential_x as Xcop_cMpc,\
        SH.CentreOfPotential_y as Ycop_cMpc,\
        SH.CentreOfPotential_z as Zcop_cMpc,\
        SH.CentreOfMass_x as Xcom_cMpc,\
        SH.CentreOfMass_y as Ycom_cMpc,\
        SH.CentreOfMass_z as Zcom_cMpc,\
        SH.Velocity_x as VXpec_kmps,\
        SH.Velocity_y as VYpec_kmps,\
        SH.Velocity_z as VZpec_kmps,\
        SH.Vmax as Vmax_circ_kmps,\
        SH.VmaxRadius as RVmax_pkpc,\
        AP.SFR as SFR_MsunPerYr,\
        AP.Mass_BH as MBH_Msun_aperture,\
        AP.Mass_Star as Mstar_Msun,\
        AP.VelDisp as Vdisp_1d_stars_aperture_kmps\
        FROM\
        %s%s_FOF as FOF, \
        %s%s_Subhalo as SH, \
        %s%s_Aperture as AP \
        WHERE\
        FOF.SnapNum = %i and \
        FOF.SnapNum = SH.SnapNum and \
        AP.ApertureSize = %i and \
        FOF.Group_M_Crit200 > %f and \
        SH.SubGroupNumber = 0 and \
        AP.GalaxyID = SH.GalaxyID and \
        FOF.GroupID = SH.GroupID \
        ORDER BY\
        M200c_Msun"%(varind, simnum, varind, simnum, varind, simnum, snapnum, apsize, Mhmin)

def generatequery_censatproperties(simnum, snapnum, Mhmin=0., var='REFERENCE', apsize=30):
    # indicators are sometimes different from the var in make_maps and Cosma directory names
    # set a few so the more familiar var options get the right simulations
    varind = getvar(var)
    return  "SELECT\
        FOF.GroupID as groupid,\
        FOF.Group_M_Crit200 as M200c_Msun,\
        FOF.Group_R_Crit200 as R200c_pkpc,\
        FOF.GroupCentreOfPotential_x as Xgroupcop_cMpc,\
        FOF.GroupCentreOfPotential_y as Ygroupcop_cMpc,\
        FOF.GroupCentreOfPotential_z as Zgroupcop_cMpc,\
        SH.GalaxyID as galaxyid,\
        SH.CentreOfMass_x as Xcom_cMpc,\
        SH.CentreOfMass_y as Ycom_cMpc,\
        SH.CentreOfMass_z as Zcom_cMpc,\
        SH.Velocity_x as VXpec_kmps,\
        SH.Velocity_y as VYpec_kmps,\
        SH.Velocity_z as VZpec_kmps,\
        SH.Masstype_DM as DMMass_Msun,\
        SH.Masstype_Gas as GasMass_Msun,\
        SH.SubGroupNumber as SubGroupNumber,\
        SH.Vmax as Vmax_circ_kmps,\
        SH.VmaxRadius as RVmax_pkpc,\
        SH.CentreOfPotential_x as Xcop_cMpc,\
        SH.CentreOfPotential_y as Ycop_cMpc,\
        SH.CentreOfPotential_z as Zcop_cMpc,\
        SH.HalfMassProjRad_Gas as HMRgas_proj_pkpc,\
        SH.HalfMassProjRad_Star as HMRstar_proj_pkpc,\
        SH.HalfMassRad_Gas as HMRgas_pkpc,\
        SH.HalfMassRad_Star as HMRstar_pkpc,\
        AP.SFR as SFR_MsunPerYr,\
        AP.Mass_BH as MBH_Msun_aperture,\
        AP.Mass_Star as Mstar_Msun,\
        AP.VelDisp as Vdisp_1d_stars_aperture_kmps\
        FROM\
        %s%s_FOF as FOF, \
        %s%s_Subhalo as SH, \
        %s%s_Aperture as AP \
        WHERE\
        FOF.SnapNum = %i and \
        FOF.SnapNum = SH.SnapNum and \
        SH.Spurious = 0 and\
        AP.ApertureSize = %i and \
        FOF.Group_M_Crit200 > %f and \
        AP.GalaxyID = SH.GalaxyID and \
        FOF.GroupID = SH.GroupID \
        ORDER BY\
        M200c_Msun"%(varind, simnum, varind, simnum, varind, simnum, snapnum, apsize, Mhmin)
        
def generatenpz(simnum,snapnum,Mmin=None,var=None):
    query = generatequery(simnum,snapnum,Mmin,var=var)    
    data = sql.execute_query(con,query)
    tosave = {entry: data[entry] for entry in entries}
    if var is None:
        varstr = ''
    else:
        varstr = var
    if Mmin != None:    
        np.savez('halocatalogue_M-R-COP_%s%s_%i_M-200c_ge_%f'%(varstr,simnum,snapnum,Mmin),**tosave)
    else:
        np.savez('halocatalogue_M-R-COP_%s%s_%i'%(varstr,simnum,snapnum),**tosave)
    return tosave

def generatehdf5_centrals(simnum, snapnum, Mhmin=0., var='REFERENCE', apsize=30):
    query = generatequery_centralproperties(simnum, snapnum, Mhmin=Mhmin, var=var, apsize=apsize)    
    data = sql.execute_query(con, query) #eturns a structured array
    dct = {key: data[key] for key in data.dtype.names}
    varind = getvar(var)
    h5name = pdir + 'catalogue_%s%s_snap%i_aperture%i.hdf5'%(varind, simnum, snapnum, apsize)
    print('saving data to %s'%h5name)
    try:
        simvar = getvar_re(varind)
        sf = pc.Simfile(simnum, snapnum, simvar, file_type=ol.file_type, simulation='eagle')
        with h5py.File(h5name, 'w') as fo:
            hed = fo.create_group('Header')
            hed.attrs.create('snapnum', snapnum)
            hed.attrs.create('simnum', np.string_(simnum))
            hed.attrs.create('var', np.string_(var))
            hed.attrs.create('subhalo_aperture_size_Mstar_Mbh_SFR_pkpc', apsize)
            hed.attrs.create('Mhalo_min_Msun', Mhmin)
            cgrp = hed.create_group('cosmopars')
            cgrp.attrs.create('boxsize', sf.boxsize)
            cgrp.attrs.create('a', sf.a)
            cgrp.attrs.create('z', sf.z)
            cgrp.attrs.create('h', sf.h)
            cgrp.attrs.create('omegam', sf.omegam)
            cgrp.attrs.create('omegab', sf.omegab)
            cgrp.attrs.create('omegalambda', sf.omegalambda)
            for key in dct.keys():
                fo.create_dataset(key, data=dct[key])
    except IOError:
        print('creating hdf5 file failed')
    return dct

def generatehdf5_censat(simnum, snapnum, Mhmin=0., var='REFERENCE', apsize=30, nameonly=False):
    query = generatequery_censatproperties(simnum, snapnum, Mhmin=Mhmin, var=var, apsize=apsize)    
    data = sql.execute_query(con, query) #eturns a structured array
    dct = {key: data[key] for key in data.dtype.names}
    varind = getvar(var)
    name = 'catalogue_%s%s_snap%i_aperture%i_inclsatellites.hdf5'%(varind, simnum, snapnum, apsize)
    if nameonly:
        return name
    h5name = pdir + name
    print('saving data to %s'%h5name)
    try:
        simvar = getvar_re(varind)
        sf = pc.Simfile(simnum, snapnum, simvar, file_type=ol.file_type, simulation='eagle')
        with h5py.File(h5name, 'w') as fo:
            hed = fo.create_group('Header')
            hed.attrs.create('snapnum', snapnum)
            hed.attrs.create('simnum', np.string_(simnum))
            hed.attrs.create('var', np.string_(var))
            hed.attrs.create('subhalo_aperture_size_Mstar_Mbh_SFR_pkpc', apsize)
            hed.attrs.create('Mhalo_min_Msun', Mhmin)
            cgrp = hed.create_group('cosmopars')
            cgrp.attrs.create('boxsize', sf.boxsize)
            cgrp.attrs.create('a', sf.a)
            cgrp.attrs.create('z', sf.z)
            cgrp.attrs.create('h', sf.h)
            cgrp.attrs.create('omegam', sf.omegam)
            cgrp.attrs.create('omegab', sf.omegab)
            cgrp.attrs.create('omegalambda', sf.omegalambda)
            for key in dct.keys():
                fo.create_dataset(key, data=dct[key])
    except IOError:
        print('creating hdf5 file failed')
    return dct


#########################################
##### halo selection from FOF files #####
#########################################


def get_EA_FOF_MRCOP(simnum, snapnum, var=None, mdef='200c', sfap='30kpc', outdct=None):
    msun = c.solar_mass
    #mpc  = c.cm_per_mpc
    #yr = c.sec_per_year
    startype = 4
    bhtype   = 5

    if var is None:
        var = 'REFERENCE'
    fo = pc.eag.read_eagle_file(ol.simdir_eagle%simnum + var + '/', 'sub', snapnum,  gadgetunits=True, suppress=False) # read_eagle_files object
    if mdef == '200c':
        massstring = 'Group_M_Crit200'
        sizestring = 'Group_R_Crit200'
        masslabel = 'M200c_Msun'
        sizelabel = 'R200c_cMpc'
    if sfap == '30kpc':
        Mststring = 'ApertureMeasurements/Mass/030kpc' # mass x parttype array
        SFRstring = 'ApertureMeasurements/SFR/030kpc'
        Mstlabel = 'Mstar_30pkpc_Msun'
        Mbhlabel = 'Mbh_30pkpc_Msun'
        SFRlabel = 'SFR_30pkpc_MsunPerYr'
    copstring = 'GroupCentreOfPotential'
    coplabel = 'COP_cMpc'

    # output in cMpc, Msun, Msun/yr
    mass = fo.read_data_array('FOF/%s'%massstring, gadgetunits=True)
    a_temp =   fo.a_scaling
    h_temp =   fo.h_scaling
    cgs_temp = fo.CGSconversion
    mconv = fo.a**a_temp * fo.h ** h_temp * cgs_temp  / msun
    mass *= mconv
    size = fo.read_data_array('FOF/%s'%sizestring, gadgetunits=True)
    size /= fo.h # gadget units -> cMpc
    cop = fo.read_data_array('FOF/%s'%copstring, gadgetunits=True)
    cop /= fo.h # gadget units -> cMpc
    subinds = fo.read_data_array('FOF/FirstSubhaloID', gadgetunits=True).astype(int)
    submass = fo.read_data_array('Subhalo/%s'%Mststring, gadgetunits=True)[subinds, :]
    mstar = submass[:, startype]
    mbh   = submass[:, bhtype]
    del submass
    a_temp =   fo.a_scaling
    h_temp =   fo.h_scaling
    cgs_temp = fo.CGSconversion
    mconv = fo.a**a_temp * fo.h ** h_temp * cgs_temp  / msun
    mstar *= mconv
    mbh *= mconv
    sfr = fo.read_data_array('Subhalo/%s'%SFRstring, gadgetunits=True)[subinds] # eagle wiki: already in Msun/yr
    #a_temp =   fo.a_scaling
    #h_temp =   fo.h_scaling
    #cgs_temp = fo.CGSconversion
    #sconv = fo.a**a_temp * fo.h ** h_temp * cgs_temp  / (msun/yr)
    #mstar *= sconv

    if outdct is None:
        outdct = {}
    outdct[masslabel] = mass
    outdct[sizelabel] = size
    outdct[coplabel]  = cop
    outdct[Mstlabel]  = mstar
    outdct[Mbhlabel]  = mbh
    outdct[SFRlabel]  = sfr
    return outdct



def selecthalos(hdfname, logMhs):
    if '/' not in hdfname:
        hdfname = pdir + hdfname
    with h5py.File(hdfname, 'r') as fi:
        mstar = np.array(fi['Mstar_Msun'])
        SFR   = np.array(fi['SFR_MsunPerYr'])
        mbh   = np.array(fi['MBH_Msun_aperture'])
        mhalo = np.array(fi['M200c_Msun'])
        ids   = np.array(fi['groupid'])
        #print np.min(np.log10(mhalo)), np.max(np.log10(mhalo)) 
    # set of halos with log halo mass agreeing with selection to 1 decimal place   
    mhsels = [np.where(np.abs(np.log10(mhalo) - mh) < 0.05)[0] for mh in logMhs]
    percentiles = np.array([5., 15., 25., 75., 85., 95.])
    prc_mbh   = [np.percentile(mbh[mhsel], percentiles) for mhsel in mhsels]
    prc_mstar = [np.percentile(mstar[mhsel], percentiles) for mhsel in mhsels]
    prc_ssfr  = [np.percentile(SFR[mhsel]/mstar[mhsel], percentiles) for mhsel in mhsels]
    bh_med = [mhsels[i][np.all(np.array([mbh[mhsels[i]] >= prc_mbh[i][2], mbh[mhsels[i]] <= prc_mbh[i][3] ]), axis=0)] for i in range(len(mhsels))]
    bh_lo  = [mhsels[i][np.all(np.array([mbh[mhsels[i]] >= prc_mbh[i][0], mbh[mhsels[i]] <= prc_mbh[i][1] ]), axis=0)] for i in range(len(mhsels))]
    bh_hi  = [mhsels[i][np.all(np.array([mbh[mhsels[i]] >= prc_mbh[i][4], mbh[mhsels[i]] <= prc_mbh[i][5] ]), axis=0)] for i in range(len(mhsels))]
    st_med = [mhsels[i][np.all(np.array([mstar[mhsels[i]] >= prc_mstar[i][2], mstar[mhsels[i]] <= prc_mstar[i][3] ]), axis=0)] for i in range(len(mhsels))]
    st_lo  = [mhsels[i][np.all(np.array([mstar[mhsels[i]] >= prc_mstar[i][0], mstar[mhsels[i]] <= prc_mstar[i][1] ]), axis=0)] for i in range(len(mhsels))]
    st_hi  = [mhsels[i][np.all(np.array([mstar[mhsels[i]] >= prc_mstar[i][4], mstar[mhsels[i]] <= prc_mstar[i][5] ]), axis=0)] for i in range(len(mhsels))]
    sf_med = [mhsels[i][np.all(np.array([(SFR/mstar)[mhsels[i]] >= prc_ssfr[i][2], (SFR/mstar)[mhsels[i]] <= prc_ssfr[i][3] ]), axis=0)] for i in range(len(mhsels))]
    sf_lo  = [mhsels[i][np.all(np.array([(SFR/mstar)[mhsels[i]] >= prc_ssfr[i][0], (SFR/mstar)[mhsels[i]] <= prc_ssfr[i][1] ]), axis=0)] for i in range(len(mhsels))]
    sf_hi  = [mhsels[i][np.all(np.array([(SFR/mstar)[mhsels[i]] >= prc_ssfr[i][4], (SFR/mstar)[mhsels[i]] <= prc_ssfr[i][5] ]), axis=0)] for i in range(len(mhsels))]
    meds  = [np.array(list(set(bh_med[i]) & set(st_med[i]) & set(sf_med[i]))) for i in range(len(mhsels))]
    lo_bh = [np.array(list(set(bh_lo[i]) & set(st_med[i]) & set(sf_med[i]))) for i in range(len(mhsels))]
    hi_bh = [np.array(list(set(bh_hi[i]) & set(st_med[i]) & set(sf_med[i]))) for i in range(len(mhsels))]
    lo_st = [np.array(list(set(bh_med[i]) & set(st_lo[i]) & set(sf_med[i]))) for i in range(len(mhsels))]
    hi_st = [np.array(list(set(bh_med[i]) & set(st_hi[i]) & set(sf_med[i]))) for i in range(len(mhsels))]
    lo_sf = [np.array(list(set(bh_med[i]) & set(st_med[i]) & set(sf_lo[i]))) for i in range(len(mhsels))]
    hi_sf = [np.array(list(set(bh_med[i]) & set(st_med[i]) & set(sf_hi[i]))) for i in range(len(mhsels))]
    dct = {'med': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in meds],\
           'lo_mbh': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in lo_bh],\
           'hi_mbh': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in hi_bh],\
           'lo_mst': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in lo_st],\
           'hi_mst': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in hi_st],\
           'lo_ssfr': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in lo_sf],\
           'hi_ssfr': [np.random.choice(ids[sel], 1)[0] if len(sel) > 0 else None for sel in hi_sf],\
           'M200c_Msun': logMhs}
    return dct
    

def plotstamp(halocat, dct_loadedslice, galaxyid, nvir=2, slicename=None, clabel=None):
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
    halos = h5py.File(ol.pdir + halocat, 'r')
    loadedslice = dct_loadedslice[dct_loadedslice.keys()[0]]
    
    ind   = np.where(np.array(halos['galaxyid']) == galaxyid)[0][0]
    Mstar = np.array(halos['Mstar_Msun'])[ind]
    M200c = np.array(halos['M200c_Msun'])[ind]
    aexp  = halos['Header/cosmopars'].attrs['a']
    boxsz = halos['Header/cosmopars'].attrs['boxsize'] / halos['Header/cosmopars'].attrs['h']
    Rvir  = np.array(halos['R200c_pkpc'])[ind] / 1.e3  / aexp
    cen   = np.array([np.array(halos['Xcop_cMpc'])[ind], np.array(halos['Ycop_cMpc'])[ind], np.array(halos['Zcop_cMpc'])[ind]])

    xyminmax_pix = np.array([cen[:2] - 2. * Rvir, cen[:2] + 2. * Rvir]).T
    xyminmax_pix = np.round(xyminmax_pix / boxsz * float(loadedslice.shape[0]) + 0.5 , 0).astype(int)
    
    ax = plt.subplot(111)
    img = ax.imshow(loadedslice[xyminmax_pix[0, 0] : xyminmax_pix[0, 1] + 1, xyminmax_pix[1, 0] : xyminmax_pix[1, 1] + 1].T,\
                    origin='lower', interpolation='nearest',\
                    extent=(cen[0] - 2*Rvir, cen[0] + nvir*Rvir, cen[1] - nvir*Rvir, cen[1] + nvir*Rvir))
    # extent: not extact, but should be good enough
    plt.colorbar(img, ax=ax, label=clabel)
    c1 = plt.Circle(cen[:2], Rvir, edgecolor='red', facecolor=None, fill=False)
    c2 = plt.Circle(cen[:2], 0.1 * Rvir, edgecolor='red', facecolor=None, fill=False)
    ax.add_artist(c1)
    ax.add_artist(c2)
    
    ax.tick_params(labelsize=11, direction='out', right=True, top=True, axis='both', which='both', color='black',\
                   labelleft=True, labeltop=False, labelbottom=True, labelright=False)
    ax.set_xlabel('X [cMpc]', fontsize=12)
    ax.set_ylabel('Y [cMpc]', fontsize=12)
    ax.text(0.05, 0.95, 'galaxy %i'%galaxyid, fontsize=12, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='red')
    ax.text(0.05, 0.89, r'$M_{200c} =$ %.2e $\mathrm{M}_{\odot}$'%M200c, fontsize=12, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='red')
    ax.text(0.05, 0.83, r'$M_{*} =$ %.2e $\mathrm{M}_{\odot}$'%Mstar, fontsize=12, verticalalignment='top', horizontalalignment='left', transform=ax.transAxes, color='red')
    
    plt.savefig('/net/luttero/data2/imgs/CGM/misc_start/' + slicename + '_galaxy-%s.pdf'%galaxyid)
## generatenpz seems to have some issues bewteen L0025N0752 REF and RECAL (returns exact same M200c, R200c, COP, haloID for the object with M200c > 1e13 Msun, which is unexpected and disagrees with the closer match to L0025N0376-REF for the hi-res RECAL that the group files show)
## this one is slower, though (since halos seem to be mass-sorted, reading in every halo as is done here is probably not strictly necessary)
#def generatenpz_fromgroup(simnum,snapnum,Mmin=None,var=None):
#
#    import make_maps_v3_master as m3
#    sim = m3.Simfile(simnum,snapnum,var,file_type='sub')
#    # read in arrays and convert to desired units (rawunits = True -> CGS)
#    m200c_solar = sim.readarray('FOF/Group_M_Crit200',rawunits=False)/m3.c.solar_mass
#    cop_cmpc = sim.readarray('FOF/GroupCentreOfPotential',rawunits=True)/sim.h # raw units are cMpc/h
#    r200c_pkpc = sim.readarray('FOF/Group_R_Crit200',rawunits=False)/m3.c.cm_per_mpc*1e3
#    # group ids are not included here
#    if var is None:
#        varstr = ''
#    else:
#        varstr = var
#    if Mmin != None:    
#        np.savez('halocatalogue_M-R-COP_%s%s_%i_M-200c_ge_%f'%(varstr,simnum,snapnum,Mmin),**tosave)
#    else:
#        np.savez('halocatalogue_M-R-COP_%s%s_%i'%(varstr,simnum,snapnum),**tosave)
#    return tosave

def namequantity(ptype, ion=None, abunds='auto', quantity=None,\
                 excludeSFR=False, parttype='0',
                 sylviasshtables=False,\
                 select=None, misc=None): 

    # EOS particle handling
    if excludeSFR == True:
        SFRind = '_noEOS'
    elif excludeSFR == False:
        SFRind = '_wiEOS'
    elif excludeSFR == 'T4':
        SFRind = '_T4EOS'
    elif excludeSFR == 'from':
        SFRind = '_fromSFR'
    elif excludeSFR == 'only':
        SFRind = '_onlyEOS'

    # abundances 
    if ptype in ['coldens', 'emission']:
        if abunds[0] not in ['Sm','Pt', 'auto']:
            sabunds = '%smassfracAb'%str(abunds[0])  
        else: 
            sabunds = abunds[0] + 'Ab'
        if type(abunds[1]) == float:
            sabunds = sabunds + '-%smassfracHAb'%str(abunds[1])
        elif abunds[1] != abunds[0]:
            sabunds = sabunds + '-%smassfracHAb'%abunds[1]
                        
    if parttype != '0':
        sparttype = '_PartType%s'%parttype
    else:
        sparttype = ''
    
    if sylviasshtables and ptype == 'coldens':
        siontab = '_iontab-sylviasHM12shh'    
    else:
        siontab = '' 

    #avoid / in file names
    if ptype == 'basic':
        squantity = quantity
        squantity = squantity.replace('/','-')

    if ptype == 'coldens' or ptype == 'emission':
        res = '%s_%s_%s' %(ptype, ion, sabunds) + SFRind + siontab
    elif ptype == 'basic':
        res = ol.ndir + '%s%s' %(squantity, sparttype) + SFRind

    if misc is not None:
        miscind = '_'+'_'.join(['%s-%s'%(key, misc[key]) for key in misc.keys()])
        res = res + miscind
          
    return res


#def projecthalos(ids, halocat, ptypeW,\
#                 depth=2., depthunits='R200c', radius=1., radiusunits='R200c',\
#                 axis='z', hdf5out=None, cMpc_per_pix=3.125e-3,
#                 ionW=None, abundsW='auto', quantityW=None,\
#                 ionQ=None, abundsQ='auto', quantityQ=None, ptypeQ=None,\
#                 excludeSFRW=False, excludeSFRQ=False, parttype='0',\
#                 theta=0.0, phi=0.0, psi=0.0, \
#                 sylviasshtables=False,\
#                 log=True, kernel='C2',\
#                 select=None, misc=None, ompproj=True):
#    '''
#    calls make_maps_v3_master.make_map to make projections around a set of 
#    halos from a catalogue; the list of haloids selects the halos
#    
#    if hdfout is not none, the output is all saved to one hdf5 file, and not to 
#    the individual npz files generated by make_map
#    
#    radiusunits: 'R200c', 'pkpc' or 'cMpc' (str)
#    depthunits:  'R200c', 'pkpc', 'cMpc', 'Vpec_kmps', 'Vobs_kmps', 'Vmax',
#                 'Vdisp' (str)
#    radius:      half size of the projection perpendicular to the projection 
#                 axis
#    depth:       full size of the projected region along the projection axis
#    axis:        'x', 'y', or 'z' (str)
#    hdf5out:     output file name (str)
#                 None -> just save the npz outputs of make_map
#                 'auto' -> auto name output file (implementation is iffy)
#    ids:         group ids (iterable of long long ints)
#    halocat:     name of hdf5 file with ids and properties (str);
#                            
#    '''
#    if axis == 'x':
#        Axis1 = 1
#        Axis2 = 2
#        Axis3 = 0
#        axname = 'X'
#    elif axis == 'y':
#        Axis1 = 2
#        Axis2 = 0
#        Axis3 = 1
#        axname = 'Y'
#    elif axis == 'z':
#        Axis1 = 0
#        Axis2 = 1
#        Axis3 = 2
#        axname = 'Z'
#    else:
#        raise ValueError('Invalid axis value %s; should be "x", "y", or "z"'%axis)
#
#    if hdf5out is not None:
#        saveres = False
#        if hdf5out == 'auto':
#            basename = halocat.split('/')[-1]
#            if basename[-5:] == '.hdf5':
#                basename = basename[:-5]
#            basename = basename + '_depth-%s%s_radius-%s%s_%scMpcPerPix_%sSm_%s-projection_'%(depth, depthunits, radius, radiusunits, cMpc_per_pix, kernel, axis)
#            wname = namequantity(ptypeW, ionW, abunds=abundsW, quantity=quantityW,\
#                 excludeSFR=excludeSFRW, parttype=parttype, sylviasshtables=sylviasshtables,\
#                 select=select, misc=misc)
#            name = basename + wname
#            if ptypeQ is not None:
#                qname = namequantity(ptypeQ, ionQ, abunds=abundsQ, quantity=quantityQ,\
#                                     excludeSFR=excludeSFRQ, parttype=parttype,\
#                                     sylviasshtables=sylviasshtables,\
#                                     select=select, misc=misc)
#                name = name +  '_and_weighted_' + qname
#            hdf5out = name
#        if '.hdf5' not in hdf5out:
#            hdf5out = hdf5out + '.hdf5'
#        if '/' not in hdf5out:
#            hdf5out = pdir + hdf5out
#    else:
#        saveres = True
#
#    if '/' not in halocat:
#        halocat = pdir + halocat
#    if '.hdf5' not in halocat:
#        halocat = halocat + 'hdf5'
#    with h5py.File(halocat, 'r') as fi:
#        cosmopars = {key: fi['Header/cosmopars'].attrs[key] for key in  fi['Header/cosmopars'].attrs.keys()}
#        hed = fi['Header']
#        simnum = hed.attrs['simnum']
#        snapnum = hed.attrs['snapnum']
#        var     = hed.attrs['var']
#        idlist  = np.array(fi['groupid'])
#        centres = np.array([np.array(fi['Xcop_cMpc']), np.array(fi['Ycop_cMpc']), np.array(fi['Zcop_cMpc'])]).T
#        if depthunits == 'R200c' or 'V' in depthunits or radiusunits == 'R200c':
#            R200s = np.array(fi['R002c_pkpc']) / 1.e3 / cosmopars['a']  # -> cMpc
#        if 'V' in depthunits:
#            vcenlos = np.array(fi['V%spec_kmps'%axname]) * 1.e5 # -> cm/s
#        if depthunits == 'Vmax':
#            vlenlos = np.array(fi['Vmax_circ_kmps']) * 1e5
#        elif depthunits == 'Vdisp': 
#            vlenlos = np.array(fi['Vdisp_1d_stars_aperture_kmps']) * 1e5
#        
#    idinds = np.array([np.where(idlist == i)[0][0] for i in ids])
#    
#    if radiusunits == 'R200c':
#        radii = R200s[idinds] * radius
#    elif radiusunits == 'pkpc':
#        radii = np.ones(len(ids)) * radius / 1.e3 / cosmopars['a']
#    elif radiusunits == 'cMpc':
#        radii = np.ones(len(ids)) * radius
#    else:
#        raise ValueError('Invalid choice %s for radiusunits')  
#        
#    if 'V' in depthunits:
#        Hz = csu.Hubble(cosmopars['z'], cosmopars=cosmopars) #  1/s
#        centres[:, Axis3] += (vcenlos / Hz / cosmopars['a'] / csu.c.cm_per_mpc) # proper cm -> comoving Mpc
#        centres[:, Axis3] %= (cosmopars['boxsize'] / cosmopars['h']) # handle periodic boundaries
#        velcut = True # convert offsets and depths to cMpc before projection
#    else:
#        velcut = False
#    if depthunits == 'R200c':
#        depths = R200s[idinds] * depth
#    elif depthunits == 'pkpc':
#        depths = np.ones(len(ids)) * depth / 1.e3 / cosmopars['a']
#    elif depthunits == 'cMpc':
#        depths = np.ones(len(ids)) * depth
#    elif depthunits in ['Vmax', 'Vdisp']:
#        depths = depth * vlenlos / Hz / cosmopars['a'] / csu.c.cm_per_mpc
#    elif depthunits == 'Vpec_kmps':
#        depths = np.ones(len(ids)) * depth * 1.e5 / Hz / cosmopars['a'] / csu.c.cm_per_mpc
#    elif depthunits == 'Vobs_kmps':
#        depths = np.ones(len(ids)) * depth * 1.e5 / Hz / csu.c.cm_per_mpc # Vobs = Vpec * (1 + z) = Vpec / a
#    else:
#        raise ValueError('Invalid choice %s for depthunits')  
#    
#    checkprevious = False
#    if hdf5out is not None: # save metadata
#        heddct = {'depth': depth, 'depthunits': depthunits,\
#                  'radius': radius, 'radiusunits': radiusunits,\
#                  'axis': axis, 'cMpc_per_pixel': cMpc_per_pix,\
#                  'kernel': kernel, 'log': log, 'parttype': parttype,
#                  'simnum': simnum, 'snapnum': snapnum, 'var': var}
#        with h5py.File(hdf5out, 'a') as fo:
#            if 'Header' in fo.keys():
#                oldheddct = {key: fo['Header'].attrs[key] for key in fo['Header'].keys()}
#                if not heddct == oldheddct:
#                    raise IOError('File %s already exists, and has incompatible projection settings'%hdf5out)
#                checkprevious = True
#            else:
#                hed = fo.create_group('Header')
#                for key in heddct.keys():
#                    hed.attrs.create(key, heddct[key])
#            if 'cosmopars' not in fo['Header'].keys():
#                cgrp = fo['Header'].create_group('cosmopars')
#                for key in cosmopars.keys():
#                    cgrp.attrs.create(key, cosmopars[key])
#                
#            
#    for i in range(len(idinds)):
#        rad = radii[idinds[i]]
#        if axis == 'x':
#            L_x = depths[i]
#            L_y = 2. * rad
#            L_z = 2. * rad
#        elif axis == 'y':
#            L_y = depths[i]
#            L_z = 2. * rad
#            L_x = 2. * rad
#        elif axis == 'z':
#            L_z = depths[i]
#            L_x = 2. * rad
#            L_y = 2. * rad
#        
#        
#        centre = centres[idinds[i]]
#        npix_x = int(np.round(2. * rad / cMpc_per_pix, 0))
#        npix_y = npix_x
#        res = m3.make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
#                          ptypeW,\
#                          ionW=ionW, abundsW=abundsW, quantityW=quantityW,\
#                          ionQ=ionQ, abundsQ=abundsQ, quantityQ=quantityQ, ptypeQ=ptypeQ,\
#                          excludeSFRW=excludeSFRW, excludeSFRQ=excludeSFRQ, parttype=parttype,\
#                          theta=theta, phi=phi, psi=psi, \
#                          sylviasshtables=sylviasshtables,\
#                          var=var, axis=axis, log=log, velcut=velcut,\
#                          periodic=False, kernel=kernel, saveres=saveres,\
#                          simulation='eagle', LsinMpc=True,\
#                          select=select, misc=misc, ompproj=ompproj, nameonly=False) 
#        if not isinstance(res, tuple):
#            res = (res, None)
#        if not saveres:
#            names = m3.make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
#                          ptypeW,\
#                          ionW=ionW, abundsW=abundsW, quantityW=quantityW,\
#                          ionQ=ionQ, abundsQ=abundsQ, quantityQ=quantityQ, ptypeQ=ptypeQ,\
#                          excludeSFRW=excludeSFRW, excludeSFRQ=excludeSFRQ, parttype=parttype,\
#                          theta=theta, phi=phi, psi=psi, \
#                          sylviasshtables=sylviasshtables,\
#                          var=var, axis=axis, log=log, velcut=velcut,\
#                          periodic=False, kernel=kernel, saveres=saveres,\
#                          simulation='eagle', LsinMpc=True,\
#                          select=select, misc=misc, ompproj=ompproj, nameonly=True)
#            grpname = str(idinds[i])
#            dosave = True
#            if checkprevious: # compare to previous results if they were already stored; no overwrites
#                if grpname in fo.keys():
#                    dosave = False
#                    if names[0] == fo['%s/proj'%grpname].attrs['npzname']:
#                        print('Projection for haloid %s was already done'%grpname)
#                        maxdiff = np.max(np.abs(np.array(fo['%s/proj'%grpname]) - res[0]))
#                        print('Maximum difference between the projections is %e'%maxdiff)
#                        if names[1] is not None:
#                            if 'av' in fo[grpname].keys():
#                                if names[1] == fo['%s/av'%grpname].attrs['npzname']:
#                                    print('Average for haloid %s was already done'%grpname)
#                                    maxdiff = np.max(np.abs(np.array(fo['%s/av'%grpname]) - res[0]))
#                                    print('Maximum difference between the averages is %e'%maxdiff)
#                                else:
#                                    print('Average for haloid %s already exists for file %s, but contains a different projection')
#                            else:
#                                print('The averaged projection did not previously exist')    
#                    else:
#                        print('Group for haloid %s already exists for file %s, but contains a different projection')
#            if dosave:
#                grp = fo[grpname]
#                grp.attrs.create('depth_cMpc', depths[i])
#                grp.attrs.create('sidelength_cMpc', 2 * rad)
#                grp.attrs.create('centre_cMpc', centre)
#                grp.attrs.create('lower_left_corner_cMpc', np.array([centre[Axis1] - rad, centre[Axis2] - rad]))
#                proj = grp.create_dataset('proj', data=res[0])
#                proj.attrs.create('npzname', names[0])
#                proj.attrs.create('ion', ionW)
#                proj.attrs.create('abunds', abundsW)
#                proj.attrs.create('quantity', quantityW)
#                proj.attrs.create('excludeSFR', excludeSFRW)
#                proj.attrs.create('sylviasshtables', sylviasshtables) 
#                proj.attrs.create('select', select)
#                if misc is not None:
#                    grp.attrs.create('misc', 'see npz names')
#                if res[1] is not None:
#                    av = grp.create_dataset('av', data=res[1])
#                    av.attrs.create('npzname', names[1])
#                    av.attrs.create('ion', ionQ)
#                    av.attrs.create('abunds', abundsQ)
#                    av.attrs.create('quantity', quantityQ)
#                    av.attrs.create('excludeSFR', excludeSFRQ)
#                    av.attrs.create('sylviasshtables', sylviasshtables) 
#                    av.attrs.create('select', select)
