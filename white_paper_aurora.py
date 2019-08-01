#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:54:37 2019

@author: wijers
"""


mdir = '/home/wijers/Documents/papers/aurora_white_paper_wide-field-xray/'

import numpy as np
import h5py
import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gsp
import matplotlib.cm as cm

import make_maps_opts_locs as ol
import makecddfs as mc # don't use getcosmopars! This requires simfile and the whole readEagle mess
import eagle_constants_and_units as c
import cosmo_utils as cu

import selecthalos as sh
import make_maps_v3_master as m3
import projection_classes as pc
import eagleSqlTools as sql

#cosmopars_ea_28 = {'a': 0.9999999999999998, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 2.220446049250313e-16}
cosmopars_ea_27 = {'a': 0.9085634947881763, 'boxsize': 67.77, 'h': 0.6777, 'omegab': 0.0482519, 'omegalambda': 0.693, 'omegam': 0.307, 'z': 0.10063854175996956}
#logrhob_av_ea_28 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_28['h']**2 * cosmopars_ea_28['omegab'] ) 
logrhob_av_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * c.hubble**2 * cosmopars_ea_27['h']**2 * cosmopars_ea_27['omegab'] / cosmopars_ea_27['a']**3 )
logrhoc_ea_27 = np.log10( 3. / (8. * np.pi * c.gravity) * cu.Hubble(cosmopars_ea_27['z'], cosmopars=cosmopars_ea_27)**2)

rho_to_nh = 0.752 / (c.atomw_H * c.u)
deg2 = (np.pi / 180.)**2
arcsec2 = deg2 / 60.**4
arcmin2 = deg2 / 60.**2

def add_colorbar(ax, img=None, vmin=None, vmax=None, cmap=None, clabel=None,\
                 newax=False, extend='neither', fontsize=12., orientation='vertical'):
    if img is None:
        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, extend=extend, orientation=orientation)
    else:
        cbar = mpl.colorbar.Colorbar(ax, img, extend=extend, orientation=orientation)
    ax.tick_params(labelsize=fontsize - 1.)
    if clabel is not None:
        cbar.set_label(clabel,fontsize=fontsize)
    return cbar

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def selecthalosample1():
    sample=1
    if sample == 1:
        logmasses=np.arange(12., 13.05, 0.1)
        logMtol=0.01
        halocat = '/net/luttero/data2/proc/catalogue_RefL0100N1504_snap27_aperture30.hdf5' # CGM -> centrals only
        samplefile = mdir + 'sample_1.txt'
    
    with h5py.File(halocat, 'r') as cat:

        cosmopars = {key: item for (key, item) in cat['Header/cosmopars'].attrs.items()}
        Xhalos = np.array(cat['Xcom_cMpc'])
        Yhalos = np.array(cat['Ycom_cMpc'])
        Zhalos = np.array(cat['Zcom_cMpc'])
        #boxsize = cosmopars['boxsize'] / cosmopars['h']
        Mhalos = np.array(cat['M200c_Msun'])
        Rhalos = np.array(cat['R200c_pkpc']) / cosmopars['a'] * 1e-3
        galids = np.array(cat['galaxyid'])
        
        halosel_mass = [np.where(np.abs(np.log10(Mhalos) - logm) < logMtol)[0] for logm in logmasses]
        inds_selected = np.array([np.random.choice(halos, 1)[0] for halos in halosel_mass])
        
        galids = galids[inds_selected]
        Mhalos = Mhalos[inds_selected]
        Rhalos = Rhalos[inds_selected]
        Xhalos = Xhalos[inds_selected]
        Yhalos = Yhalos[inds_selected]
        Zhalos = Zhalos[inds_selected]
        
        with open(samplefile, 'w+') as fo:
            fo.write('galaxyid\tXcom_cMpc\tYcom_cMpc\tZcom_cMpc\tM200c_Msun\tR200c_cMpc\n')
            fmtbase = '\t'.join(['%s'] * 6) + '\n'
            for i in range(len(inds_selected)):
                wstr = fmtbase%(galids[i], Xhalos[i], Yhalos[i], Zhalos[i], Mhalos[i], Rhalos[i])
                fo.write(wstr)

def getparentsample2(sample=2):
    if sample == 2:    
        # generate query
        var = 'REFERENCE'
        sqlvar = 'Ref'
        sim = 'L0100N1504'
        snap = 27
        Mhmin = 10**11.45
        Mhmax = 10**12.55
        apsize = 30
    elif sample == 3:
         # generate query
        var = 'REFERENCE'
        sqlvar = 'Ref'
        sim = 'L0100N1504'
        snap = 27
        Mhmin = 10**11.95
        Mhmax = 10**13.05
        apsize = 30
    # set a few so the more familiar var options get the right simulations
    query =\
        "SELECT \
        FOF.GroupID as groupid,\
        FOF.Group_M_Crit200 as M200c_Msun,\
        FOF.Group_R_Crit200 as R200c_pkpc,\
        SH.GalaxyID as galaxyid,\
        SH.CentreOfMass_x as Xcom_cMpc,\
        SH.CentreOfMass_y as Ycom_cMpc,\
        SH.CentreOfMass_z as Zcom_cMpc,\
        SH.Velocity_x as VXpec_kmps,\
        SH.Velocity_y as VYpec_kmps,\
        SH.Velocity_z as VZpec_kmps,\
        SH.Masstype_DM as DMMass_Msun,\
        SH.Masstype_Gas as GasMass_Msun,\
        AP.SFR as SFR_MsunPerYr,\
        AP.Mass_BH as MBH_Msun_aperture,\
        AP.Mass_Star as Mstar_Msun,\
        MK.KappaCoRot as KappaCoRot,\
        MK.DiscToTotal as DiscToTotal \
        FROM \
        %s%s_FOF as FOF, \
        %s%s_Subhalo as SH, \
        %s%s_Aperture as AP, \
        %s%s_MorphoKinem as MK \
        WHERE \
        FOF.SnapNum = %i and \
        FOF.SnapNum = SH.SnapNum and \
        SH.Spurious = 0 and \
        Sh.SubGroupNumber = 0 and \
        AP.ApertureSize = %i and \
        FOF.Group_M_Crit200 > %f and \
        FOF.Group_M_Crit200 < %f and \
        AP.GalaxyID = SH.GalaxyID and \
        AP.GalaxyID = MK.GalaxyID and \
        FOF.GroupID = SH.GroupID \
        ORDER BY\
        M200c_Msun"%(sqlvar, sim, sqlvar, sim, sqlvar, sim, sqlvar, sim, snap, apsize, Mhmin, Mhmax)
    con = sql.connect('nwijers', password='G3Wuve94') 
    data = sql.execute_query(con, query) #eturns a structured array
    dct = {key: data[key] for key in data.dtype.names}
    h5name = mdir + 'parentsample%i_Lstarish-centrals_%s%s_snap%i_aperture%i.hdf5'%(sample, var, sim, snap, apsize)
    print('saving data to %s'%h5name)
    try:
        sf = pc.Simfile(sim, snap, var, file_type='snap', simulation='eagle')
        with h5py.File(h5name, 'w') as fo:
            hed = fo.create_group('Header')
            hed.attrs.create('snapnum', snap)
            hed.attrs.create('simnum', sim)
            hed.attrs.create('var', var)
            hed.attrs.create('subhalo_aperture_size_Mstar_Mbh_SFR_pkpc', apsize)
            hed.attrs.create('Mhalo_min_Msun', Mhmin)
            hed.attrs.create('Mhalo_max_Msun', Mhmax)
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

def plotparentsample(sample=2):
    if sample == 2:
        filename = mdir + 'parentsample2_Lstarish-centrals_REFERENCEL0100N1504_snap27_aperture30.hdf5'
    elif sample == 3:
        filename = mdir + 'parentsample3_Lstarish-centrals_REFERENCEL0100N1504_snap27_aperture30.hdf5'

    with h5py.File(filename, 'r') as fi:
        Mhalo = np.log10(np.array(fi['M200c_Msun']))
        sSFR  = np.log10(np.array(fi['SFR_MsunPerYr']) / np.array(fi['Mstar_Msun']))
        kappa = np.array(fi['KappaCoRot'])
        dt    = np.array(fi['DiscToTotal'])
    
    fig, axes = plt.subplots(nrows=2, ncols=3, gridspec_kw={'hspace': 0.3, 'wspace': 0.45})
    fontsize = 12
    alpha=0.05
    Mhalolabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    sSFRlabel = r'$\log_{10} \, \mathrm{sSFR} \; [\mathrm{yr}^{-2}]$'
    kappalabel = r'$\kappa_{\mathrm{corot}}$'
    dtlabel    = 'D / T'
    
    for ax in axes.flatten():
        ax.tick_params(labelsize=fontsize - 1, direction='in', which='both', top=True, right=True)
    sSFR = np.max([sSFR, -14. * np.ones(len(sSFR))], axis=0)
    
    ax = axes[0, 0]
    ax.hist2d(Mhalo, sSFR, bins=50, cmap='gist_yarg')
    ax.set_xlabel(Mhalolabel, fontsize=fontsize)
    ax.set_ylabel(sSFRlabel, fontsize=fontsize)
        
    ax = axes[0, 1]
    ax.hist2d(Mhalo, kappa, bins=50, cmap='gist_yarg')
    ax.set_xlabel(Mhalolabel, fontsize=fontsize)
    ax.set_ylabel(kappalabel, fontsize=fontsize)
    
    ax = axes[0, 2]
    ax.hist2d(Mhalo, dt, bins=50, cmap='gist_yarg')
    ax.set_xlabel(Mhalolabel, fontsize=fontsize)
    ax.set_ylabel(dtlabel, fontsize=fontsize)
    
    ax = axes[1, 0]
    ax.hist2d(sSFR, kappa, bins=50, cmap='gist_yarg')
    ax.set_xlabel(sSFRlabel, fontsize=fontsize)
    ax.set_ylabel(kappalabel, fontsize=fontsize)
    
    ax = axes[1, 1]
    ax.hist2d(sSFR, dt, bins=50, cmap='gist_yarg')
    ax.set_xlabel(sSFRlabel, fontsize=fontsize)
    ax.set_ylabel(dtlabel, fontsize=fontsize)
    
    ax = axes[1, 2]
    ax.hist2d(dt, kappa, bins=50, cmap='gist_yarg')
    ax.set_xlabel(dtlabel, fontsize=fontsize)
    ax.set_ylabel(kappalabel, fontsize=fontsize)
    
    plt.savefig(mdir + 'parentsample%i.pdf'%sample, format='pdf', bbox_inches='tight')

def selectsample2(parent=2, sub='a'):
    samplename = mdir + 'sample_%s%s.txt'%(parent, sub)
    halocat = mdir + 'parentsample%i_Lstarish-centrals_REFERENCEL0100N1504_snap27_aperture30.hdf5'%parent
    if parent== 2 and sub == 'a':
        logmasses = [11.5, 12.0, 12.5]
        logMtol = 0.05
    if parent == 3 and sub == 'a':
        logmasses = [12.0, 12.5, 13.0]
        logMtol = 0.05
        
    with h5py.File(halocat, 'r') as cat:

        cosmopars = {key: item for (key, item) in cat['Header/cosmopars'].attrs.items()}
        Xhalos = np.array(cat['Xcom_cMpc'])
        Yhalos = np.array(cat['Ycom_cMpc'])
        Zhalos = np.array(cat['Zcom_cMpc'])
        #boxsize = cosmopars['boxsize'] / cosmopars['h']
        Mhalos = np.array(cat['M200c_Msun'])
        Rhalos = np.array(cat['R200c_pkpc']) / cosmopars['a'] * 1e-3
        galids = np.array(cat['galaxyid'])
        sSFR  = np.array(cat['SFR_MsunPerYr']) / np.array(cat['Mstar_Msun'])
        kappa = np.array(cat['KappaCoRot'])
        dt    = np.array(cat['DiscToTotal'])
        
        halosel_mass = [np.where(np.abs(np.log10(Mhalos) - logm) < logMtol)[0] for logm in logmasses]
        inds_selected = np.array([np.random.choice(halos, 1)[0] for halos in halosel_mass])
        
        galids = galids[inds_selected]
        Mhalos = Mhalos[inds_selected]
        Rhalos = Rhalos[inds_selected]
        Xhalos = Xhalos[inds_selected]
        Yhalos = Yhalos[inds_selected]
        Zhalos = Zhalos[inds_selected]
        
        with open(samplename, 'w+') as fo:
            fo.write('galaxyid\tXcom_cMpc\tYcom_cMpc\tZcom_cMpc\tM200c_Msun\tR200c_cMpc\tsSFR_PerYr\tKappaCoRot\tDisktoTotal\n')
            fmtbase = '\t'.join(['%s'] * 9) + '\n'
            for i in range(len(inds_selected)):
                wstr = fmtbase%(galids[i], Xhalos[i], Yhalos[i], Zhalos[i], Mhalos[i], Rhalos[i], sSFR[i], kappa[i], dt[i])
                fo.write(wstr)
                
def projectsample(sample=1, radiusmargin=1.2):
    if sample == 1:
        samplename = mdir + 'sample_%i.txt'%sample
        metadataname = mdir + 'sample_%i_projdata.txt'%sample
    elif sample == '2a':
        samplename = mdir + 'sample_%s.txt'%sample
        metadataname = mdir + 'sample_%s_projdata.txt'%sample
    elif sample == '2c':
        samplename = mdir + 'sample_%s.txt'%sample
        metadataname = mdir + 'sample_%s_projdata.txt'%sample
        
    snapnum = 27
    lines = ['o6major', 'o6minor'] # ['o8', 'o7r', 'o7ix', 'o7iy', 'o7f', 'fe17'] 
    simnum = 'L0100N1504'
    var='auto'
    pix = 800
    
    abundsW = 'Pt'
    quantityW = None
    excludeSFRW = 'T4'
    
    ptypeQ = None
    ionQ = None
    abundsQ = 'auto'
    quantityQ = None
    excludeSFRQ = None
        
    saveres = True
    log = True
    misc = None
    ompproj = True
    
    if ompproj: # check number of threads is set
       numthreads  = int(os.environ['OMP_NUM_THREADS'])
    
    df = pd.read_csv(samplename, sep='\t', header=0)
    #galids = df['galaxyid']
    inds = np.array(df.index)
    names = {}
    
    for ind in inds:
        npix_x=pix
        npix_y=pix
        
        radius = df.loc[ind, 'R200c_cMpc']
        L_x = 2. * radius * radiusmargin
        L_y = 2. * radius * radiusmargin
        L_z = 2. * radius * radiusmargin
        centre = np.array([df.loc[ind, 'Xcom_cMpc'], df.loc[ind, 'Ycom_cMpc'], df.loc[ind, 'Zcom_cMpc']], dtype=np.float)
        #print type(centre[0]) 
        #print type(centre[1]) 
        #print type(centre[2]) 
        axis = 'z'
        
        periodic = False
        velcut = False
        kernel = 'C2'
        
        parttype = '0'
        
        ptypeW = 'emission'

        galid = df.loc[ind, 'galaxyid']
        names[galid] = {}
        
        for line in lines:
    
            ionW = line
            
            
            # document input parameters:(processed input is also printed by make_map)
            print('\n')
            
            print('Overview of function input parameters: [cMpc] where applicable \n')
            print('simnum: \t %s' %simnum)
            print('snapnum: \t %i' % snapnum)
            print('centre: \t %s' %str(centre))
            print('L_x, L_y, L_z: \t %f, %f, %f \n' %(L_x, L_y, L_z))
            
            print('kernel: \t %s' %kernel)
            print('axis: \t %s' %axis)
            print('periodic: \t %s' %str(periodic))
            print('npix_x,npix_y: \t %i, %i \n' %(npix_x, npix_y))
            
            print('projection --')
            print(' ptype: %s\n ion: %s\n abunds: %s\n quantity: %s\n SFR: %s'\
                  %(ptypeW, ionW, abundsW, quantityW, excludeSFRW))
            print('weighted --')
            print(' ptype: %s\n ion: %s\n abunds: %s\n quantity: %s\n SFR: %s'\
                  %(ptypeQ, ionQ, abundsQ, quantityQ, excludeSFRQ))
            
            print('saveres: \t %s' %str(saveres))
            print('log: \t %s' %str(True))
            
            print('\n')
            
            # function call
            print centre 
            name = m3.make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
                     ptypeW,\
                     ionW=ionW, abundsW=abundsW, quantityW=quantityW,\
                     ionQ=ionQ, abundsQ=abundsQ, quantityQ=quantityQ, ptypeQ=ptypeQ,\
                     excludeSFRW=excludeSFRW, excludeSFRQ=excludeSFRQ, parttype=parttype,\
                     theta=0.0, phi=0.0, psi=0.0, \
                     var=var, axis=axis,log=log, velcut=velcut,\
                     periodic=periodic, kernel=kernel, saveres=saveres,\
                     misc=misc, ompproj=ompproj, nameonly=True)
            
            names[galid][line] = name
            
            m3.make_map(simnum, snapnum, centre, L_x, L_y, L_z, npix_x, npix_y, \
                     ptypeW,\
                     ionW=ionW, abundsW=abundsW, quantityW=quantityW,\
                     ionQ=ionQ, abundsQ=abundsQ, quantityQ=quantityQ, ptypeQ=ptypeQ,\
                     excludeSFRW=excludeSFRW, excludeSFRQ=excludeSFRQ, parttype=parttype,\
                     theta=0.0, phi=0.0, psi=0.0, \
                     var=var, axis=axis,log=log, velcut=velcut,\
                     periodic=periodic, kernel=kernel, saveres=saveres,\
                     misc=misc, ompproj=ompproj, nameonly=False)
            
    with open(metadataname, 'a+') as mdf:
        mdf.write('galaxyid\tline\tfilename\n')
        keys = names.keys()
        lines = names[keys[0]].keys()
        
        for key in keys:
            for line in lines:
                mdf.write('%s\t%s\t%s\n'%(key, line, names[key][line][0]))
                

def plotsample(sample=1):

    samplename = mdir + 'sample_%i.txt'%sample
    metadataname = mdir + 'sample_%i_projdata.txt'%sample
 
    ds = pd.read_csv(samplename, sep='\t', header=0)
    dm = pd.read_csv(metadataname, sep='\t', header=0)
    
    linelabels = {'o7': 'O VII triplet',\
                  'o8': 'O VIII',\
                  'fe17': 'Fe XVII'}
    
    
    for ind in np.array(ds.index):
        galid = ds.loc[ind, 'galaxyid']
        mhalo = ds.loc[ind, 'M200c_Msun']
        rhalo = ds.loc[ind, 'R200c_cMpc']
        
        plotname = mdir + 'emission_galaxy_%i_1.2R200c.pdf'%(galid)
        
        filenames = dm[dm['galaxyid'] == galid]
        lines = np.array(dm['line'])
        indsel = {line: np.array(filenames.index[filenames['line'] == line])[0] for line in lines}
        names = {line: filenames.loc[indsel[line], 'filename'] for line in lines}
        files = {line: np.load(names[line] + '.npz')['arr_0'] for line in names.keys()}
        files['o7'] = np.log10(np.sum(10**np.array([files[line] for line in ['o7r', 'o7ix', 'o7iy', 'o7f']]), axis=0))
        
        # assuming  z-projection
        sfn = names[lines[0]]
        sfn = sfn.split('/')[-1]
        sfn = sfn.split('_')

        lxpart = set(part if len(part) > 0  else None for part in sfn)
        lxpart = lxpart - {None}
        lxpart = set(part if  (part[0] == 'x' and '-pm' in part) > 0  else None for part in sfn)
        lxpart = lxpart - {None}
        lxpart = list(lxpart)
        if len(lxpart) != 1:
            raise RuntimeError('Failed to retrieve projection size from file %s'(names[lines[0]]))
        lxpart = lxpart[0].split('-')        
        lx = float(lxpart[-1][2:])
        cx = float(lxpart[0][1:])
        
        lypart = set(part if len(part) > 0  else None for part in sfn)
        lypart = lypart - {None}
        lypart = set(part if (part[0] == 'y' and '-pm' in part) else None for part in sfn)
        lypart = lypart - {None}
        lypart = list(lypart)
        if len(lypart) != 1:
            raise RuntimeError('Failed to retrieve projection size from file %s'(names[lines[0]]))
        lypart = lypart[0].split('-')        
        ly = float(lypart[-1][2:])
        cy = float(lypart[0][1:])
        
        extent = (cx - 0.5 * lx, cx + 0.5 * lx, cy - 0.5 * ly, cy + 0.5 * ly)
        
        fig = plt.figure(figsize=(8.5, 3.))
        grid = gsp.GridSpec(1, 4, hspace=0.25, wspace=0.0, width_ratios=[3., 3., 3., 1.])
        axes = [fig.add_subplot(grid[i]) for i in range(3)]
        cax = fig.add_subplot(grid[3])
        cmap = 'plasma'
        ancolor = 'forestgreen'
        
        fontsize = 12
        vmax = max([np.max(files[line]) for line in ['o7', 'o8', 'fe17']])
        vmin = min([np.min(files[line][np.isfinite(files[line])] if np.any(np.isfinite(files[line])) else -np.inf)  for line in ['o7', 'o8', 'fe17']])
        vmin = max(vmin, vmax - 6.)
        cmapf = cm.get_cmap(cmap)
        
        # o7 triplet
        line = 'o7'
        ax = axes[0]
        ax.tick_params(labelsize=fontsize - 1)
        ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        
        ax.set_facecolor(cmapf(0.))
        img = ax.imshow(files[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        circle = plt.Circle((cx, cy), rhalo, color=ancolor, fill=False)
        ax.add_artist(circle)
        
        add_colorbar(cax, img=img, vmin=None, vmax=None, cmap=None, clabel=r'$\log_{10} \, \mathrm{SB} \; [\mathrm{photons} \, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{sr}^{-1}]$',\
                 newax=False, extend='min', fontsize=fontsize, orientation='vertical')
        ax.text(0.05, 0.95, linelabels[line], fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color=ancolor) #bbox=dict(facecolor='white',alpha=0.3),
        ax.text(0.05, 0.05, r'$\log_{10} \, \mathrm{M}_{200\mathrm{c}} / \mathrm{M}_{\odot} = %.2f$'%(np.log10(mhalo)), fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color=ancolor) 
        
        line = 'o8'
        ax = axes[1]
        ax.tick_params(labelsize=fontsize - 1, labelleft=False)
        ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        #ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        
        ax.set_facecolor(cmapf(0.))
        ax.imshow(files[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        circle = plt.Circle((cx, cy), rhalo, color=ancolor, fill=False)
        ax.add_artist(circle)
        
        
        ax.text(0.05, 0.95, linelabels[line], fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color=ancolor) #bbox=dict(facecolor='white',alpha=0.3),
        ax.text(0.05, 0.05, 'galaxy %i'%(galid), fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color=ancolor) 

        line = 'fe17'
        ax = axes[2]
        ax.tick_params(labelsize=fontsize - 1, labelleft=False)
        ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        #ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        
        ax.set_facecolor(cmapf(0.))
        ax.imshow(files[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        circle = plt.Circle((cx, cy), rhalo, color=ancolor, fill=False)
        ax.add_artist(circle)
        
        ax.text(0.05, 0.95, linelabels[line], fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color=ancolor) #bbox=dict(facecolor='white',alpha=0.3),
        ax.text(0.2, 0.8, r'$\mathrm{R}_{200\mathrm{c}}$', fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color=ancolor) 
        
        cax.set_aspect(12.)
        cax.tick_params(labelsize=fontsize - 1)
        
        plt.savefig(plotname)
        

def plotsample2(parent=2, sub='a'):

    if parent == 2 and sub == 'a':
        samplename = mdir + 'sample_2%s.txt'%sub
        metadataname = mdir + 'sample_2%s_projdata.txt'%sub
        afactor = 0.908563 # from eagle wiki
    elif parent == 2 and sub == 'c': # larger two of the sample 2a halos
        samplename = mdir + 'sample_%s%s.txt'%(parent, sub)
        metadataname = mdir + 'sample_%i%s_projdata.txt'%(parent, sub)
        afactor = 0.908563 # from eagle wiki
        
    plotname = mdir + 'emission_sample%i%s_1.5R200c_arcmin.pdf'%(parent, sub)
    ds = pd.read_csv(samplename, sep='\t', header=0)
    dm = pd.read_csv(metadataname, sep='\t', header=0)
    
    linelabels = {'o7': 'O VII triplet',\
                  'o8': 'O VIII',\
                  'o6': 'O VI doublet',\
                  'fe17': 'Fe XVII'}
    
    imgs = {}
    extents = {}
    rhalos = {}
    mhalos = {}
    centers = {}
    for ind in np.array(ds.index):
        galid = ds.loc[ind, 'galaxyid']
        mhalo = ds.loc[ind, 'M200c_Msun']
        rhalo = ds.loc[ind, 'R200c_cMpc']
        
        
        filenames = dm[dm['galaxyid'] == galid]
        lines = np.array(dm['line'])
        indsel = {line: np.array(filenames.index[filenames['line'] == line])[0] for line in lines}
        names = {line: filenames.loc[indsel[line], 'filename'] for line in lines}
        files = {line: np.load(names[line] + '.npz')['arr_0'] + np.log10(arcmin2) for line in names.keys()}
        files['o7'] = np.log10(np.sum(10**np.array([files[line] for line in ['o7r', 'o7ix', 'o7iy', 'o7f']]), axis=0))
        files['o6'] = np.log10(np.sum(10**np.array([files[line] for line in ['o6major', 'o6minor']]), axis=0)) #  
        # assuming  z-projection
        sfn = names[lines[0]]
        sfn = sfn.split('/')[-1]
        sfn = sfn.split('_')

        lxpart = set(part if len(part) > 0  else None for part in sfn)
        lxpart = lxpart - {None}
        lxpart = set(part if  (part[0] == 'x' and '-pm' in part) > 0  else None for part in sfn)
        lxpart = lxpart - {None}
        lxpart = list(lxpart)
        if len(lxpart) != 1:
            raise RuntimeError('Failed to retrieve projection size from file %s'(names[lines[0]]))
        lxpart = lxpart[0].split('-')        
        lx = float(lxpart[-1][2:])
        cx = float(lxpart[0][1:])
        
        lypart = set(part if len(part) > 0  else None for part in sfn)
        lypart = lypart - {None}
        lypart = set(part if (part[0] == 'y' and '-pm' in part) else None for part in sfn)
        lypart = lypart - {None}
        lypart = list(lypart)
        if len(lypart) != 1:
            raise RuntimeError('Failed to retrieve projection size from file %s'(names[lines[0]]))
        lypart = lypart[0].split('-')        
        ly = float(lypart[-1][2:])
        cy = float(lypart[0][1:])
        
        extent = (cx - 0.5 * lx, cx + 0.5 * lx, cy - 0.5 * ly, cy + 0.5 * ly)
        
        imgs[galid] = {'o7': files['o7'], 'o8': files['o8'], 'o6': files['o6']} # 'fe17': files['fe17']
        rhalos[galid] = rhalo
        mhalos[galid] = mhalo
        extents[galid] = extent
        centers[galid] = (cx, cy)
        
        
    fontsize = 11
    
    vmax_xray = max([np.max(imgs[galid][line]) for line in ['o7', 'o8'] for galid in imgs.keys()]) #, 'fe17'
    vmin_xray = min([np.min(imgs[galid][line][np.isfinite(imgs[galid][line])] if np.any(np.isfinite(imgs[galid][line])) else np.inf)  for line in ['o7', 'o8'] for galid in imgs.keys()]) #, 'fe17'
    vmin_xray = max(vmin_xray, vmax_xray - 5.)
    
    vmax_uv = max([np.max(imgs[galid][line]) for line in ['o6'] for galid in imgs.keys()]) #, 'fe17'
    vmin_uv = min([np.min(imgs[galid][line][np.isfinite(imgs[galid][line])] if np.any(np.isfinite(imgs[galid][line])) else np.inf)  for line in ['o6'] for galid in imgs.keys()]) #, 'fe17'
    vmin_uv = max(vmin_uv, vmax_uv - 5.)
    
    numgals = len(imgs.keys())
    panelsize = 2.0
    cbarsize  = 0.3
    labelsize = 0.3
    fig = plt.figure(figsize=(panelsize * 4 + labelsize, panelsize * numgals + cbarsize))
    grid = gsp.GridSpec(nrows=numgals + 1, ncols=5, hspace=0.0, wspace=0.0, width_ratios=[panelsize] * 4 + [labelsize], height_ratios=[panelsize] * numgals + [cbarsize])
    axes = [[fig.add_subplot(grid[j, i]) for i in range(5)] for j in range(numgals)]
    cgrid = gsp.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=grid[numgals, :], wspace=0.0, hspace=0.0, width_ratios=[0.32, 0.36, 0.32])
    cax1 = fig.add_subplot(cgrid[0]) #
    cax2 = fig.add_subplot(cgrid[2])
    caxl = fig.add_subplot(cgrid[1])
    
    cmap_xray = 'inferno'
    cmap_uv   = 'viridis'
    ancolor = 'lightgray'
    cmapf_xray = cm.get_cmap(cmap_xray)
    cmapf_uv = cm.get_cmap(cmap_uv)
    cmapf_xray = truncate_colormap(cmapf_xray, minval=0.15, maxval=1.0, n=-1)
    cmapf_uv   = truncate_colormap(cmapf_uv, minval=0.0, maxval=0.93, n=-1)
    cmapf_xray.set_under(cmapf_xray(0.))
    cmapf_uv.set_under(cmapf_uv(0.))
    
    imgs = {key: {line: np.max([imgs[key][line], -100. * np.ones(imgs[key][line].shape)], axis=0) for line in imgs[key].keys()} for key in imgs.keys()}
    
    galids = sorted(imgs.keys(), key=mhalos.__getitem__)
    for galind in range(len(galids)):
        galid = galids[galind]
        img_sub = imgs[galid]
        mhalo = mhalos[galid]
        rhalo = rhalos[galid]
        extent = extents[galid]
        _axes = axes[galind]
        center = centers[galid]
        
        # o7 triplet
        line = 'o6'
        ax = _axes[1]
        ax.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
        #ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        #ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        
        ax.set_facecolor(cmapf_xray(0.))
        img_uv = ax.imshow(img_sub[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin_uv, vmax=vmax_uv, cmap=cmapf_uv)
        circle = plt.Circle(center, rhalo, color=ancolor, fill=False)
        ax.add_artist(circle)
        
        if galind == 0:
            ax.text(0.5, 1.05, linelabels[line], fontsize=fontsize + 1, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes) #bbox=dict(facecolor='white',alpha=0.3),
        
        
        line = 'o7'
        ax = _axes[2]
        ax.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
        #ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        #ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        
        ax.set_facecolor(cmapf_xray(0.))
        ax.imshow(img_sub[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin_xray, vmax=vmax_xray, cmap=cmapf_xray)
        circle = plt.Circle(center, rhalo, color=ancolor, fill=False)
        ax.add_artist(circle)
        
        if galind == 0:
            ax.text(0.5, 1.05, linelabels[line], fontsize=fontsize + 1, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes) #bbox=dict(facecolor='white',alpha=0.3),

        ypos = center[1] - 1.4 * rhalo
        xstart = center[0] - 1.4 * rhalo
        xlen = 200e-3 / afactor # 200 pkpc in cMpc units
        ax.plot([xstart, xstart + xlen], [ypos] * 2, color=ancolor, linewidth=2.)
        ax.text((xstart - extent[0]) / (extent[1] - extent[0]),\
                (ypos - extent[2]) / (extent[3] - extent[2]) + 0.02,\
                r'$200 \, \mathrm{pkpc}$', fontsize=fontsize, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color=ancolor)
        
        
#        line = 'fe17'
#        ax = _axes[3]
#        ax.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
#        #ax.set_xlabel('X [cMpc]', fontsize=fontsize)
#        #ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
#        
#        ax.set_facecolor(cmapf(0.))
#        ax.imshow(img_sub[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
        line = 'o8'
        ax = _axes[3]
        ax.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
        #ax.set_xlabel('X [cMpc]', fontsize=fontsize)
        #ax.set_ylabel('Y [cMpc]', fontsize=fontsize)
        
        ax.set_facecolor(cmapf_xray(0.))
        img_xray = ax.imshow(img_sub[line].T, origin='lower', interpolation='nearest', extent=extent, vmin=vmin_xray, vmax=vmax_xray, cmap=cmapf_xray)
        circle = plt.Circle(center, rhalo, color=ancolor, fill=False)
        ax.add_artist(circle)
        
        if galind == 0:
             ax.text(0.5, 1.05, linelabels[line], fontsize=fontsize + 1, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes) #bbox=dict(facecolor='white',alpha=0.3),
        ax.text(0.7, 0.2, r'$\mathrm{R}_{200\mathrm{c}}$', fontsize=fontsize, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, color=ancolor) 
        
        
        ax = _axes[0]
        ax.tick_params(left=False, right=False, top=False, bottom=False, labelleft=False, labelbottom=False)
        if galind == 0:
             ax.text(0.5, 1.05, '$gri$ image', fontsize=fontsize + 1, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes) #bbox=dict(facecolor='white',alpha=0.3),
        try:
            gri = mpl.image.imread(mdir + 'galrand_%i.png'%galid)
        except IOError:
            gri = None
        if gri is None:
            with h5py.File('/home/wijers/Documents/papers/aurora_white_paper_wide-field-xray/parentsample2_Lstarish-centrals_REFERENCEL0100N1504_snap27_aperture30.hdf5', 'r') as cat:
                _galids = np.array(cat['galaxyid'])
                _ind = np.where(_galids == galid)[0][0]
                mstar = np.log10(np.array(cat['Mstar_Msun'])[_ind])
                del _ind
                del _galids
            ax.text(0.5, 0.5, 'Image not found\n' + r'$\log_{10} \, M_{*} = %.1f \mathrm{M}_{\odot}$'%(mstar), fontsize=fontsize, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            ax.imshow(gri, interpolation='nearest', extent=(-30., 30, -30., 30.))
        
        # add zoom box and outline
        for spine in _axes[0].spines.values():
            spine.set_edgecolor(ancolor)
            spine.set_linewidth(2.)
        size_sub = 60e-3 / afactor
        ax = _axes[1]
        ax.plot([center[0] - 0.5 * size_sub, center[0] + 0.5 * size_sub, center[0] + 0.5 * size_sub, center[0] - 0.5 * size_sub, center[0] - 0.5 * size_sub],\
                [center[1] - 0.5 * size_sub, center[1] - 0.5 * size_sub, center[1] + 0.5 * size_sub, center[1] + 0.5 * size_sub, center[1] - 0.5 * size_sub],\
                color=ancolor, linestyle='dotted')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot([center[0] - 0.5 * size_sub, xlim[0]], [center[1] + 0.5 * size_sub, ylim[1]], color=ancolor, linestyle='dotted')
        ax.plot([center[0] - 0.5 * size_sub, xlim[0]], [center[1] - 0.5 * size_sub, ylim[0]], color=ancolor, linestyle='dotted')
    
        # add galaxy info
        ax = _axes[4]
        text = r'$ \mathrm{M}_{200\mathrm{c}} = %.1f$'%(np.log10(mhalo)) # '%i\n'%galid + 
        ax.text(0.0, 0.5, text, fontsize=fontsize + 1, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, rotation=-90.) 
        ax.axis('off')
         
    clabel = r'$\log_{10} \, \mathrm{SB} \; [\mathrm{ph.} \, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{arcmin}^{-2}]$'

    add_colorbar(cax1, img=img_uv, vmin=None, vmax=None, cmap=None, clabel=None,\
                 newax=False, extend='min', fontsize=fontsize, orientation='horizontal')
    cax1.set_aspect(1./10.)
    cax1.tick_params(labelsize=fontsize - 1, direction='in', which='both', pad=-13, top=False)
    ticklabels = cax1.get_xticklabels()
    xaxis = cax1.get_xaxis()
    ticks = xaxis.get_majorticklines()
    #colors = np.array([(i,) * 3 for i in np.linspace(0.9, 0., len(ticklabels))])
    colors = np.array([(0.,) * 3] * len(ticklabels))
    colors[:len(colors) // 2, :] = np.array(mpl.colors.to_rgb(ancolor))
    #print(len(ticklabels))
    #print(len(ticks))
    # twice as many tick instances as tick locaitons and labels: top/bottom?
    for ticklabel, tickcolor in zip(ticklabels, list(colors)):
        ticklabel.set_color(tickcolor)
    for tick, tickcolor in zip(ticks[::2], list(colors)):
        tick.set_color(tickcolor)
        
    add_colorbar(cax2, img=img_xray, vmin=None, vmax=None, cmap=None, clabel=None,\
                 newax=False, extend='min', fontsize=fontsize, orientation='horizontal')
    cax2.set_aspect(1./10.)
    cax2.tick_params(labelsize=fontsize - 1, direction='in', which='both', pad=-13, top=False)
    ticklabels = cax2.get_xticklabels()
    xaxis = cax2.get_xaxis()
    ticks = xaxis.get_majorticklines()
    #colors = np.array([(i,) * 3 for i in np.linspace(0.9, 0., len(ticklabels))])
    colors = np.array([(0.,) * 3] * len(ticklabels))
    colors[:len(colors) // 2, :] = np.array(mpl.colors.to_rgb(ancolor))
    #print(len(ticklabels))
    #print(len(ticks))
    # twice as many tick instances as tick locaitons and labels: top/bottom?
    for ticklabel, tickcolor in zip(ticklabels, list(colors)):
        ticklabel.set_color(tickcolor)
    for tick, tickcolor in zip(ticks[::2], list(colors)):
        tick.set_color(tickcolor)
        
        
    caxl.text(0.50, 0.00, clabel, fontsize=fontsize, horizontalalignment='center', verticalalignment='bottom', transform=caxl.transAxes)
    caxl.axis('off')
    plt.savefig(plotname)    
    

def plotSBprofiles(lines='o7trip', fontsize=12, xunit='pkpc', percentiles=[10., 90.], inclmasses='all', indicatenumgals=True):
    rdir = '/net/luttero/data2/proc/radprof/'
    
    if isinstance(lines, str):
        lines = [lines]
    lineorder = {'o7trip': 0, 'o8': 1, 'fe17': 2}
    lines.sort(key=lineorder.__getitem__)    
    
    outname = mdir + 'SB_profile_%s_%s_z0p1_DeltaZ-14p29'%('-'.join(lines), xunit)
    if inclmasses != 'all':
        outname = outname + '_masses-%s'%('-'.join([str(mass) for mass in inclmasses]))
    if indicatenumgals:
        outname = outname + '_wng'
    outname = outname + '.pdf'
    
    cmapname = 'rainbow'
    dynrange = 8.0
    
    filebase = rdir + 'rdist_emission_%s_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_1slice_to-1200-pkpc-or-2p5-R200c_M200c-logMsun-11p95-13p05_SubGroupNumber-0_stored_profiles.hdf5'
        
    linenames = {'o7trip': r'O VII He $\alpha$ triplet',\
                 'o8':     'O VIII 653.55 eV',\
                 'fe17':   'Fe XVII 726.97 eV'}
    
    ylabel = r'$\log_{10} \, \mathrm{SB} \; [\mathrm{photons}\,\mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{sr}^{-1}]$'
    if xunit == 'pkpc':
        xlabel = r'$\log_{10} \, r_{\perp} \; [\mathrm{pkpc}]$'
        datapath1 = '/%s_bins/binset_1/'%xunit
    elif xunit == 'R200c':
        xlabel = r'$\log_{10} \, r_{\perp} \,/\, R_\mathrm{200c}$'
        datapath1 = '/%s_bins/binset_0/'%xunit
    clabel = r'$\log_{10} \, \mathrm{M}_{\mathrm{200c}} \; [\mathrm{M}_{\odot}]$'
    
    med = 'perc_50.0'
    lop = 'perc_%.1f'%percentiles[0]
    hip = 'perc_%.1f'%percentiles[1]  
    
    specialpars = {12.0: {'linewidth': 3.5, 'linestyle': 'dashed'},\
                   12.5: {'linewidth': 3.5, 'linestyle': 'dotted'},\
                  }
    
    profiles = {}
    numgals = {}
    for line in lines:
        profiles[line] = {}    
        numgals[line] = {}
        filename = filebase%line
        with h5py.File(filename, 'r') as fi:
            keys = fi.keys()
            setkeys = set(key if 'galset' in key else None for key in keys)
            setkeys -= {None}
            for key in setkeys:
                grp = fi[key + datapath1]
                tag = fi[key].attrs['seltag']
                profiles[line][tag] = {}
                bins = np.log10(np.array(grp['bin_edges']))
                if bins[0] == -np.inf:
                    bins[0] = bins[1] - bins[2]
                profiles[line][tag]['bins'] = bins
                profiles[line][tag]['med'] = np.array(grp[med])
                profiles[line][tag]['lop'] = np.array(grp[lop])
                profiles[line][tag]['hip'] = np.array(grp[hip])
                numgals[line][tag] = len(fi[key + '/galaxyid'])
                del tag
                del bins
    if not np.all([set(profiles[lines[0]].keys()) == set(profiles[line].keys()) for line in lines]):
        raise RuntimeError('Tags for the different lines did not match')
    masstags = profiles[lines[0]].keys()
    if not np.all([np.all([numgals[line][tag] == numgals[lines[0]][tag] for tag in masstags]) for line in lines]):
        raise RuntimeError('Number of galaxies in the same mass bin for the different lines did not match')
    
    masses = {tag: float(tag.split('_')[-1]) for tag in masstags}
    #print(masses)
    masstags.sort(key=masses.__getitem__)
    
    if inclmasses != 'all':
        seltags = set(tag if masses[tag] in inclmasses else None for tag in masstags)
        seltags -= {None}
    else:
        seltags = set(masstags)
        
    ymax = max([max([np.max(profiles[line][tag]['med']) for tag in seltags]) for line in lines])
    ymax += 0.4 # margin
    ymin = min([min([np.min(profiles[line][tag]['lop'][np.isfinite(profiles[line][tag]['lop'])]) if np.any(np.isfinite(profiles[line][tag]['lop'])) else np.inf for tag in seltags]) for line in lines])
    ymin = max(ymin, ymax - dynrange)
    
    numlines = len(lines)
    fig = plt.figure(figsize=(4.0 * numlines + 0.5, 4.0))
    grid = gsp.GridSpec(1, numlines + 1, width_ratios=[8.] * numlines +  [1.], wspace=0.0)
    cax = fig.add_subplot(grid[len(lines)])
    
    # set up the color bar and color map
    massbounds = np.array([masses[key] for key in masstags]) # masstags are sorted
    massbounds = np.array(list(massbounds - 0.5 * np.average(np.diff(massbounds))) + [massbounds[-1] + 0.5 * np.average(np.diff(massbounds))])
    clist = cm.get_cmap(cmapname, len(massbounds) - 1)(np.linspace(0., 1.,len(massbounds) - 1))
    for ti in range(len(masstags)):
        if masstags[ti] not in seltags:
            clist[ti] = np.array([1., 1., 1., 1.])
         
    #print(clist)
    cmap = mpl.colors.ListedColormap(clist)
    norm = mpl.colors.BoundaryNorm(massbounds, cmap.N)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,\
                                norm=norm,\
                                boundaries=massbounds,\
                                ticks=[masses[tag] for tag in masstags[::2]],\
                                spacing='proportional',\
                                orientation='vertical')
    # to use 'extend', you must
    # specify two extra boundaries:
    # boundaries=[0] + bounds + [13],
    # extend='both',
    # ticks=bounds,  # optional
    cbar.set_label(clabel, fontsize=fontsize)
    cax.tick_params(labelsize=fontsize - 1)
    
    # annotate color bar with sample size per bin
    if indicatenumgals:
        ancolor = 'black'
        for tag in masstags:
            if tag not in seltags:
                continue
            ypos = masses[tag]
            xpos = 0.5
            cax.text(xpos, (ypos - massbounds[0]) / (massbounds[-1] - massbounds[0]), str(numgals[lines[0]][tag]), fontsize=fontsize, color=ancolor, verticalalignment='center', horizontalalignment='center')
    
    # plot the profiles
    axes = []
    for li in range(numlines):
        line = lines[li]
        ax = fig.add_subplot(grid[li])
        axes.append(ax)
        
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_title(linenames[line], fontsize=fontsize + 1.)
        ax.set_ylim(ymin, ymax)
        
        if li == 0:
            ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.minorticks_on()
        ax.tick_params(labelsize=fontsize - 1, direction='in', top=True, right=True, which='both', labelleft=li == 0)
        
        if xunit == 'pkpc':
            ax.set_xlim(0.5, 3.)
        else:
            ax.set_xlim(-0.8, 0.35)
        
        for ti in range(len(masstags)):
            tag = masstags[ti]
            if tag not in seltags:
                continue
            mass = masses[tag]
            color = clist[ti]
            bins = profiles[line][tag]['bins']
            bins = bins[:-1] + 0.5 * np.diff(bins)
            if mass in specialpars.keys():
                kwargs = specialpars[mass]
                plotspread = True
            else:
                kwargs = {'linewidth': 1.5, 'linestyle': 'solid'}
                plotspread = False
            ax.plot(bins, profiles[line][tag]['med'], color='black', linestyle='solid', linewidth=kwargs['linewidth'] + 0.5)
            ax.plot(bins, profiles[line][tag]['med'], color=color, label=mass, **kwargs) # label for initial testing
            if plotspread:
                ax.fill_between(bins, profiles[line][tag]['lop'], profiles[line][tag]['hip'], color=color, alpha=0.5)
        
        if li != 0:
            old_ticklocs = ax.get_xticks()
            old_ticklocs_min = ax.get_xticks(minor = True)
            ax.set_xticks(old_ticklocs[1:])
            ax.set_xticks(old_ticklocs_min, minor=True)
            
    legend_handles_lines = [mlines.Line2D([], [], color='black', linestyle='solid', label='median')]
    #print(tuple(percentiles))
    legend_handles_patches = [mpatches.Patch(color='black', alpha=0.5, label=r'$%.0f \endash %.0f$%%'%tuple(percentiles))]
    #handles, labels = ax.get_legend_handles_labels()
    axes[0].legend(handles=legend_handles_lines + legend_handles_patches, ncol=1, fontsize=fontsize, loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=False)

    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
def plotSBmaps(line, res=800, fill=0):
    pdir = '/net/luttero/data2/proc/'
    filenamebase = 'emission_%s_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_reduced_res.hdf5'
    size = 100. # cMpc
    
    filename = pdir + filenamebase%line
    fontsize = 12
    
    linenames = {'o7trip': r'O VII He $\alpha$ triplet',\
                 'o8':     'O VIII 653.55 eV',\
                 'fe17':   'Fe XVII 726.97 eV'}
    
    clabel = r' SB $[\log_{10} \, \mathrm{erg}\, \mathrm{cm}^{-2}\mathrm{s}^{-1}\mathrm{arcmin}^{-2}]$'
    unitconv = ol.line_eng_ion[line] * arcmin2
    
    with h5py.File(filename, 'r') as fi:
        keys = fi.keys()
        zcens = {key: float(key.split('-')[-1]) for key in keys}
        keys.sort(key=zcens.__getitem__)
        path = '%s/%ipix'%(keys[fill], res)
        
        image = np.array(fi[path])
    
    #deg2 = (np.pi / 180.)**2
    #arcsec2 = deg2 / 60.**4
    image = image + np.log10(unitconv)
    
    fig = plt.figure(figsize=(5.0, 5.0))
    ax = fig.add_axes([0., 0., 1., 1.])
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.07])
    
    cax.tick_params(left=False, right=False, bottom=False, top=False, labelbottom=False)
    ax.tick_params(left=False, right=False, bottom=False, top=False)
    cax.spines['bottom'].set_color('white') 
    cax.spines['top'].set_color('white')
    cax.spines['left'].set_color('white')
    cax.spines['right'].set_color('white')
    
    cmap1 = 'gist_gray'
    cmap2 = 'inferno'
    vmin = -25.
    vmax = -15.
    vpiv = -18.
    # cobble together a color map
    nsample = 256
    cmap1 = mpl.cm.get_cmap(cmap1)
    cmap2 = mpl.cm.get_cmap(cmap2)
    # the parts of the 0., 1. range to map each color bar to
    range1_mapto = np.linspace(0., (vpiv - vmin)/ (vmax - vmin), nsample)
    range2_mapto = np.linspace((vpiv - vmin)/ (vmax - vmin), 1., nsample)
    # the parts of each color bar to use
    range1_mapfrom = np.linspace(0., 0.65, nsample) 
    range2_mapfrom = np.linspace(0.2, 1., nsample)
    maplist1 = [(range1_mapto[i], cmap1(range1_mapfrom[i])) for i in range(nsample)]
    maplist2 = [(range2_mapto[i], cmap2(range2_mapfrom[i])) for i in range(nsample)]
    
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'bw_to_color', maplist1 + maplist2)
    cmap.set_under(cmap(0.))
    cmap.set_over(cmap(1.))
    
    extent = (0., size, 0., size)
    
    ax.set_facecolor(cmap(0.))
    img = ax.imshow(image.T, origin='lower', interpolation='nearest', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    
    fig.colorbar(img, cax=cax, orientation='horizontal')
    
    cax.text(0., 0.5, str(vmin), color='white', horizontalalignment='left', verticalalignment='center', fontsize=fontsize, transform=cax.transAxes)
    cax.text(1., 0.5, str(vmax), color='black', horizontalalignment='right', verticalalignment='center', fontsize=fontsize, transform=cax.transAxes)
    cax.text(0.5, 0.5, clabel, color='white', horizontalalignment='center', verticalalignment='center', fontsize=fontsize, transform=cax.transAxes)
    cax.tick_params(labelbottom=False, bottom=False)
    
    ax.text(0.5, 0.95, linenames[line], color='white', horizontalalignment='center', verticalalignment='top', fontsize=fontsize + 1, transform=ax.transAxes)

    ax.plot([4., 14.], [0.98 * size] * 2, color='white', linewidth=2)
    ax.text(8. / size, 0.97, '10 cMpc', color='white', horizontalalignment='center', verticalalignment='top', fontsize=fontsize, transform=ax.transAxes)
    
    plt.savefig(mdir + 'SBmap_%s_slice-%-of-7_detlim-%s.pdf'%(line, fill + 1, vpiv), format='pdf')
    

def plotionwpd(line, minsb, fraction=False):
    pdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/proc/'
    mdir = '/cosma5/data/dp004/dc-wije1/line_em_abs/img/'
    if line == 'o8':
        filename = 'hist_emission_o8_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_and_weighted_rho_T.hdf5'
    
    fontsize = 12
    
    linenames = {'o7trip': r'O VII He $\alpha$ triplet',\
                 'o8':     'O VIII 653.55 eV',\
                 'fe17':   'Fe XVII 726.97 eV'}
    
    logrhob = logrhob_av_ea_27 + np.log10(rho_to_nh)
    
    xlabel = r'$\log_{10} \, n_{\mathrm{H}} \; \mathrm{cm}^{-3}$ ' + linenames[line] + ' weighted'
    ylabel = r'$\log_{10} \, T [\mathrm{K}]$ ' + linenames[line] + ' weighted'
    
    unitconv = ol.line_eng_ion[line] * arcmin2
    
    with h5py.File(pdir + filename, 'r') as fi:
        hist = np.array(fi['masks_0/hist'])
        ax0_em = np.array(fi['bins/axis_0']) + np.log10(unitconv)
        ax1_nh = np.array(fi['bins/axis_1']) + np.log10(rho_to_nh)
        ax2_tk = np.array(fi['bins/axis_2'])
        totpix = 7 * 32000**2
        
        if ax1_nh[0] == -np.inf:
            ax1_nh[0] = 2. * ax1_nh[1] - ax1_nh[2]
        if ax2_tk[0] == -np.inf:
            ax2_tk[0] = 2. * ax2_tk[1] - ax2_tk[2]
        if ax1_nh[-1] == np.inf:
            ax1_nh[-1] = 2. * ax1_nh[-2] - ax1_nh[-3] 
        if ax2_tk[-1] == np.inf:
            ax2_tk[-1] = 2. * ax2_tk[-2] - ax2_tk[-3]
        
    fig, (ax, cax) = plt.subplots(ncols=2, nrows=1, figsize=(5.5, 5.0), gridspec_kw={'wspace': 0.0, 'width_ratios': [5., 0.5]})

    minsb_ind = np.argmin(np.abs(ax0_em - minsb))
    minsb_real = ax0_em[minsb_ind]
    maximg = np.sum(hist[:, :, :], axis=0) / float(totpix) / np.diff(ax1_nh)[:, np.newaxis] / np.diff(ax2_tk)[np.newaxis, :]
    img = np.sum(hist[minsb_ind:, :, :], axis=0) / float(totpix) / np.diff(ax1_nh)[:, np.newaxis] / np.diff(ax2_tk)[np.newaxis, :]

    
    if fraction:
        clabel = r'observable fraction'
        cmap = 'viridis'
        img = img.astype(np.float)
        img /= maximg
        vmax=1.
        vmin=0.
        imgob = ax.pcolormesh(ax1_nh, ax2_tk, img.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    else:
        clabel = r'$ \log_{10} \, \mathrm{sky \; fraction} \,/\, \Delta \log_{10} n_{\mathrm{H}} \,  \Delta \log_{10} T$'
        cmap = 'gist_yarg'
        img = np.log10(img)
        maximg = np.log10(maximg)    
        vmax = np.max(maximg)
        vmin = vmax - 10.
        imgob = ax.pcolormesh(ax1_nh, ax2_tk, img.T, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    
    ax.axvline(logrhob, color='blue', linestyle='dotted')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    fig.colorbar(imgob, cax=cax, orientation='vertical', extend='neither')
    cax.set_ylabel(clabel, fontsize=fontsize)
    
    fig.suptitle(linenames[line] + r'$ > 10^{%.2f} \; \mathrm{erg}\, \mathrm{cm}^{-2} \mathrm{s}^{-1} \mathrm{arcmin}^{-2}$'%minsb_real, fontsize=fontsize)
    
    outname = 'phase_diagram_hist_emission_%s_L0100N1504_27_test3.31_PtAb_C2Sm_32000pix_14.2857142857slice_zcen-all_z-projection_T4EOS_and_weighted_rho_T_minSB-%s_fraction-%s.pdf'%(line, minsb_real, fraction)
    plt.savefig(mdir + outname, format='pdf', dpi=400)
