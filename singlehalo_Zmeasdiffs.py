#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:38:11 2019

@author: wijers
"""

import numpy as np
import h5py
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1 as axgrid
import matplotlib.gridspec as gsp
import matplotlib.lines as mlines
import matplotlib.legend_handler as mlh
import matplotlib.collections as mcol
import matplotlib.patheffects as mppe
import matplotlib.patches as mpatch

import eagle_constants_and_units as c
import make_maps_opts_locs as ol
import make_maps_v3_master as m3
import plothistograms_filenames as pfn
import plot_utils as pu

# where to put text files and plots (projections go in ndir)
mdir = '/net/luttero/data2/imgs/Zmeascomps/singlegals/'


# generate a galaxy sample (centrals only)
def gensample(setname, halocat, targets_Mstar=(9.0, 9.5, 10.0, 10.5), tol_Mstar=0.05, numeach=4, onlycentrals=True):
    '''
    generate a galaxy sample, save the resulting data in mdir/setname.txt
    select by stellar mass ()
    '''
    if '/' not in halocat:
        halocat = ol.pdir + halocat
    if halocat[-5:] != '.hdf5':
        halocat = halocat + '.hdf5'
        
    with h5py.File(halocat, 'r') as hc:
        mstar = np.log10(hc['Mstar_Msun'])
        subn  = np.array(hc['SubGroupNumber']) 
        if onlycentrals:
            mstar[subn > 0] = np.NaN # exlcude from mass selection
        # select the halos
        subsets = [np.where(np.abs(tarm - mstar) <= tol_Mstar)[0] for tarm in targets_Mstar]
        selinds = [np.random.choice(sub, size=numeach, replace=False) if len(sub) > numeach else sub for sub in subsets]
        selinds = np.array([ind for sel in selinds for ind in sel])
        
        # get other useful data on them
        galids = np.array(hc['galaxyid'])[selinds]
        groupids = np.array(hc['groupid'])[selinds]
        mstar = mstar[selinds]
        subn = subn[selinds]
        xpos = np.array(hc['Xcom_cMpc'])[selinds]
        ypos = np.array(hc['Ycom_cMpc'])[selinds]
        zpos = np.array(hc['Zcom_cMpc'])[selinds]
        R200c_group = np.array(hc['R200c_pkpc'])[selinds] * 1e-3 / hc['Header/cosmopars'].attrs['a']
        HMR_gal = np.max([np.array(hc['HMRgas_proj_pkpc'])], axis=0) * 1e-3 / hc['Header/cosmopars'].attrs['a']
        # = np.array(hc['R200c_pkpc'])[selinds] * 1e-3 / hc['Header/cosmopars'].attrs['a']
    
    with open(mdir + setname + '.txt', 'w') as fo:
        # metadata
        fo.write('halocat:\t%s\n'%halocat)
        fo.write('targets_Mstar:\t%s\n'%(str(targets_Mstar)))
        fo.write('tol_Mstar:\t%s\n'%(tol_Mstar))
        fo.write('numeach:\t%i\n'%(numeach))
        fo.write('onlycentrals:\t%s\n'%(onlycentrals))
        # column heads
        fo.write('galaxyid\tgroupid\tMstar_logMsun\tSubGroupNumber\tXcom_cMpc\tYcom_cMpc\tZcom_cMpc\tR200c_cMpc\tprojHMR_gas_cMpc\n')
        for i in range(len(galids)):
            fo.write('%i\t%i\t%f\t%i\t%f\t%f\t%f\t%f\t%f\n'%(galids[i], groupids[i], mstar[i], subn[i], xpos[i], ypos[i], zpos[i], R200c_group[i], HMR_gal[i]))

                
# run all the projections for one galaxy
def run_projections(centre, projbox,\
                    pixres=3.125, simnum='L0025N0752', snapnum=19,\
                    var='RECALIBRATED', excludeSFR='T4', axis='z',\
                    halosel=None, kwargs_halosel=None):
    '''
    In the directions _|_ to the axis, the projection region is slightly 
    extended to get the target resolution
    '''
    
    # set number of pixels, adjust box size
    pixres *= 1e-3
    npix = int(np.ceil(projbox / pixres))
    L_p = npix * pixres
    if axis == 'z':
        L_x, L_y = (L_p,) * 2
        L_z = projbox
    elif axis == 'y':
        L_x, L_z = (L_p,) * 2
        L_y = projbox
    elif axis == 'x':
        L_y, L_z = (L_p,) * 2
        L_x = projbox
    
    kwargs_base = {'var': var, 'excludeSFRW': excludeSFR, 'excludeSFRQ': excludeSFR,\
              'axis': axis, 'periodic': False, 'saveres': True, 
              'halosel': halosel, 'kwargs_halosel': kwargs_halosel,\
              'simulation': 'eagle', 'LsinMpc': True, 'ompproj': True,\
              'hdf5': True,\
              'ptypeQ': 'basic', 'quantityQ': 'SmoothedMetallicity'}
    
    args_base = [simnum, snapnum, centre, L_x, L_y, L_z, npix, npix]
    projdct = {'h1ssh': ('coldens', 'h1ssh'),\
               'hneutralssh': ('coldens', 'hneutralssh'),\
               'Gasmass': ('basic', 'Mass'),\
               'SFR': ('basic', 'StarFormationRate')}

    outdct = {}
    for tupk in projdct.keys():
        tup = projdct[tupk] 
        args = tuple(args_base + [tup[0]])
        kwargs = kwargs_base.copy()
        # which is used depends on ptype, but the other one is ignored anyway
        kwargs['ionW'] = tup[1]
        kwargs['quantityW'] = tup[1]
        
        outnames = m3.make_map(*args, nameonly=True, **kwargs)
        outdct[tupk] = outnames
        
        m3.make_map(*args, nameonly=False, **kwargs)
        
    return outdct


def project_sample(setname, pixres=3.125, excludeSFR='T4', axis='z', projrad=(50., 'pkpc')):
    '''
    projrad: (size, units). units are one of pkpc, ckpc, pMpc, cMpc, R200c, or
             HMR (projected gas half-mass radius)
             note that R200c applies to the current parent halo, even if the 
             galaxy is a satellite
    pixres:  size of a pixel in ckpc
    setname: name of the text file containing the info for this set of galaxies
    '''
    textname = mdir + setname + '.txt'
    mdname = mdir + setname + '_projfiles.txt'
    
    with open(textname, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%setname)
                else:
                    break
            elif line.startswith('halocat'):
                halocat = line.split(':')[1]
                halocat = halocat.strip()
                headlen += 1
            elif ':' in line or line == '\n':
                headlen += 1
                
    # infer box, snapnum, var from used catalogue 
    with h5py.File(halocat, 'r') as hc:
        hed = hc['Header']
        cosmopars = {key: item for key, item in hed['cosmopars'].attrs.items()}
        simnum = hed.attrs['simnum']
        snapnum = hed.attrs['snapnum']
        var = hed.attrs['var']
        ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
        
    galdata_all = pd.read_csv(textname, header=headlen, sep='\t')   
    ginds = np.array(galdata_all.index)
    
    with open(mdname, 'w') as to:
        to.write('galaxyid\tweight\tpfile\tqfile\n')
        
    for gind in ginds:
        gdata = galdata_all.loc[gind]
        galid = gdata['galaxyid']
        groupid = gdata['groupid']
        mstar = gdata['Mstar_logMsun']
        centre = [gdata['Xcom_cMpc'], gdata['Ycom_cMpc'], gdata['Zcom_cMpc']] 
        R200c = gdata['R200c_cMpc']
        HMR = gdata['projHMR_gas_cMpc']
        
        projbox = 2. * projrad[0] # radius -> diameter
        if 'kpc' in projrad[1]:
            projbox *= 1e-3
        if projrad[1][0] == 'p':
            projbox /= cosmopars['a']
        if projrad[1] == 'R200c':
            projbox *= R200c
        if projrad[1] == 'HMR':
            projbox *= HMR
        
        output_names = run_projections(centre, projbox,\
                        pixres=pixres, simnum=simnum, snapnum=snapnum,\
                        var=var, excludeSFR=excludeSFR, axis=axis,\
                        halosel=None, kwargs_halosel=None)
        
        for namek in output_names:
            names = output_names[namek]
            for name in names:
                with h5py.File(name, 'a') as fo:
                    mgrp = fo.create_group('Header/target_galaxy')
                    mgrp.attrs.create('galaxyid', galid)
                    mgrp.attrs.create('Mstar_%ipkpc_logMsun'%(ap), mstar)
                    mgrp.attrs.create('groupid', groupid)
        
        with open(mdname, 'a') as to:
            for namek in output_names:
                to.write('%i\t%s\t%s\t%s\n'%(galid, namek, output_names[namek][0], output_names[namek][1]))
     
        
###############################################################################
#                              plot utils                                     #            
###############################################################################

units = {'SFR': c.solar_mass / c.sec_per_year / (1e-3 * c.cm_per_mpc)**2, \
         'Gasmass': c.solar_mass / (1e-3 * c.cm_per_mpc)**2, \
         'hneutralssh': 1., \
         'h1ssh': 1., \
         'Z': ol.Zsun_sylviastables}    

cosmopars_default = {}
cosmopars_default[19] = {'a': 0.4989716956678924,\
                         'z': 1.0041216940401045,\
                         'h': 0.6777,\
                         'boxsize': 16.9425,\
                         'omegam': 0.307,\
                         'omegalambda': 0.693,\
                         'omegab': 0.0482519,\
                         }

def readoutmap_forimgplot(filename, unit, cosmopars):
    with h5py.File(filename, 'r') as fp:
        img = np.array(fp['map']) - np.log10(unit)
        grp = fp['Header/inputpars']
        cen = grp.attrs['centre']
        L_x = grp.attrs['L_x']
        L_y = grp.attrs['L_y']
        L_z = grp.attrs['L_z']
        Ls = np.array([L_x, L_y, L_z])
        if not bool(grp.attrs['LsinMpc']):
            Ls *= cosmopars['h']
        axis = grp.attrs['axis'].decode() # bytestring to str
        if axis == 'z':
            Axis3 = 2
            Axis2 = 1
            Axis1 = 0
        elif axis == 'y':
            Axis3 = 1
            Axis2 = 0
            Axis1 = 2
        elif axis == 'x':
            Axis3 = 0
            Axis2 = 2
            Axis1 = 1
        extent = np.array([cen[Axis1] -  0.5 * Ls[Axis1],\
                           cen[Axis1] +  0.5 * Ls[Axis1],\
                           cen[Axis2] -  0.5 * Ls[Axis2],\
                           cen[Axis2] +  0.5 * Ls[Axis2],\
                           ]) \
                 * 1e3 * cosmopars['a'] # convert to pkpc
        depth = Ls[Axis3] * 1e3 * cosmopars['a']
        ret = {'map': img, 'extent': tuple(extent), 'depth': depth}
        return ret
     
def plotsample_imgs(setname, galids_toplot='all'):
    textname = mdir + setname + '.txt'
    mdname = mdir + setname + '_projfiles.txt'
    
    if galids_toplot not in ['all', None]:
        if not hasattr(galids_toplot, '__len__'):
            galids_toplot = [galids_toplot]
    
    with open(textname, 'r') as fi:
        # scan for halo catalogue (only metadata needed for this)
        headlen = 0
        halocat = None
        while True:
            line = fi.readline()
            if line == '':
                if halocat is None:
                    raise RuntimeError('Reached the end of %s without finding the halo catalogue name'%setname)
                else:
                    break
            elif line.startswith('halocat'):
                halocat = line.split(':')[1]
                halocat = halocat.strip()
                headlen += 1
            elif ':' in line or line == '\n':
                headlen += 1
    
    with h5py.File(halocat, 'r') as hc:
        hed = hc['Header']
        cosmopars = {key: item for key, item in hed['cosmopars'].attrs.items()}
        simnum = hed.attrs['simnum']
        snapnum = hed.attrs['snapnum']
        var = hed.attrs['var']
        ap = hed.attrs['subhalo_aperture_size_Mstar_Mbh_SFR_pkpc']
    
    galdata_all = pd.read_csv(textname, header=headlen, sep='\t')
    filenames_all = pd.read_csv(mdname, sep='\t')
    
    ginds = np.array(galdata_all.index)
    labels = {'SFR': r'$\log_{10} \, \Sigma_{\mathrm{SFR}} \; [\mathrm{M}_{\odot} \, \mathrm{yr}^{-1} \mathrm{pkpc}^{-2}]$',\
               'h1ssh': r'$\log_{10} \, \mathrm{N}_{\mathrm{H\,I}}} \; [\mathrm{cm}}^{-2}}]$',\
               'hneutralssh': r'$\log_{10} \, \mathrm{N}_{\mathrm{H\,I} + \mathrm{H}_{2}}} \; [\mathrm{cm}}^{-2}}]$',\
               'Gasmass': r'$\log_{10} \, \Sigma_{\mathrm{gas}} \; [\mathrm{M}_{\odot} \mathrm{pkpc}^{-2}]$',\
               'Z': r'$\log_{10} \, \mathrm{Z} \; [\mathrm{Z}_{\odot}]$'}
    Zlabels = {'SFR': 'SF',\
               'h1ssh': 'H I',\
               'hneutralssh': r'$\mathrm{H\,I} + \mathrm{H}_{2}$',\
               'Gasmass': 'gas'}
    
    fontsize = 12
    xlabel = 'pkpc'
    ylabel = 'pkpc'
    Zdifflabel = r'$\Delta \, \log_{10} \, \mathrm{Z}$'
    cmaps = {'Gasmass': 'viridis',\
             'h1ssh': 'magma',\
             'hneutralssh': 'magma',\
             'SFR': 'inferno',\
             'Z': 'plasma',\
             'diff': 'RdBu'}
    vminmax = {'Gasmass': (3., 8.0),\
               'h1ssh': (14., 22.),\
               'hneutralssh': (14., 22.),\
               'SFR': (-7., -1.),\
               'Z':  (-2.5, 0.5),\
               'diff': (-1., 1.)}
    
    
    for gind in ginds:
        gdata = galdata_all.loc[gind]
        galid = gdata['galaxyid']
        if galids_toplot not in ['all', None]:
            if galid not in galids_toplot:
                continue
        groupid = gdata['groupid']
        mstar = gdata['Mstar_logMsun']
        subn = gdata['SubGroupNumber']
        R200c = gdata['R200c_cMpc']
        
        titletext = 'galaxy id: %i\n'%galid + \
                    'group id: %i\n'%groupid + \
                    r'$\log_{10} \, \mathrm{M}_{\star} \, / \, M_{\odot}$' + ': %.3f (%i pkpc)\n'%(mstar, ap) + \
                    ('central\n' if subn == 0 else 'satellite\n') + \
                    r'parent halo $\mathrm{R}_{\mathrm{200c}} \, /\, \mathrm{pkpc}$' + ': %.3f\n'%(R200c * 1e3 * cosmopars['a']) + \
                    'simulation: %s-%s\n'%(simnum, var) + \
                    'snapshot: %i '%(snapnum) + r'($z=%.2f$)'%(cosmopars['z']) + '\n' 
        
        filenames = filenames_all.loc[filenames_all['galaxyid'] == galid]    
        #print(galid)
        #print(filenames)        
        weights = np.array(filenames['weight'])
        
        plotdct = {}
        for weight in weights:
            loc = np.where(np.logical_and(filenames_all['weight'] == weight, filenames_all['galaxyid'] == galid))[0][0]
            pfile = filenames_all.at[loc, 'pfile']
            qfile = filenames_all.at[loc, 'qfile']
            
            plotdct[weight] = readoutmap_forimgplot(pfile, units[weight], cosmopars)
            plotdct[weight + 'SmZ'] = readoutmap_forimgplot(qfile, units['Z'], cosmopars)
            
        # plot layout:
        # Gas mass         h1      hneutral    SFR
        #                weighted smZs
        # Zh1 - Zmass  diff cbar      |text
        # Zhn - Zmass  Zhn - Zh1      |annotations
        # ZSF - Zmass  ZSF - Zh1  ZSF - Zhn  
        
        panelwidth = 2
        panelheight = panelwidth
        caxw = 0.5
        padh = 0.2
        numw = len(weights)
        fig = plt.figure(figsize=(numw * panelwidth + caxw, (2 + numw - 1) * panelheight + caxw + 2 * padh))
        maingrid = gsp.GridSpec(ncols=numw + 1, nrows=numw - 1 + 5, hspace=0.0, wspace=0.0,\
                                height_ratios= [caxw] + [padh] + [panelheight] * 2 + [padh] + [panelheight] * (numw - 1),\
                                width_ratios= [panelwidth] * numw + [caxw])
        
        caxes_w = [fig.add_subplot(maingrid[0, i]) for i in range(numw)]
        axes_w = [fig.add_subplot(maingrid[2, i]) for i in range(numw)]
        axes_q = [fig.add_subplot(maingrid[3, i]) for i in range(numw)]
        cax_q  = fig.add_subplot(maingrid[3, numw])
        
        weights = ['Gasmass', 'h1ssh', 'hneutralssh', 'SFR']
        zmin = min([ np.min(plotdct[weight + 'SmZ']['map'][ np.isfinite(plotdct[weight + 'SmZ']['map']) ]) for weight in weights])
        zmax = max([ np.max(plotdct[weight + 'SmZ']['map'][ np.isfinite(plotdct[weight + 'SmZ']['map']) ]) for weight in weights])
        
        for wi in range(len(weights)):
            weight = weights[wi] 
            labely = wi==0
            axw = axes_w[wi]
            axq = axes_q[wi]
            cax = caxes_w[wi]
            
            extent = plotdct[weight]['extent']
            img = axw.imshow(plotdct[weight]['map'].T, extent=extent, origin='lower', interpolation='nearest', cmap=cmaps[weight], vmin=vminmax[weight][0], vmax=vminmax[weight][1])
            
            #axw.tick_params(which='both', direction='in', top=True, right=True, labelleft=labely, labelbottom=False, labelsize=fontsize - 1.)
            axw.tick_params(which='both', direction='in', left=False, bottom=False, labelleft=False, labelbottom=False, right=False, top=False)
            axq.tick_params(which='both', direction='in', left=False, bottom=False, labelleft=False, labelbottom=False, right=False, top=False)
            
            if labely:
                #axw.set_ylabel(ylabel, fontsize=fontsize)
                pe=[mppe.Stroke(linewidth=5, foreground="black"), mppe.Normal()]
                pet = [mppe.Stroke(linewidth=1., foreground="black"), mppe.Normal()]
                axw.plot([extent[0] + 8., extent[0] + 18.], [extent[2] + 0.05 * (extent[3] - extent[2])] * 2, linewidth=4.5, color='gray', path_effects=pe)
                axw.text(13. / (extent[1] - extent[0]), 0.07, '10 pkpc', fontsize=fontsize, color='gray', horizontalalignment='center', verticalalignment='bottom', transform=axw.transAxes, path_effects=pet)
            
            plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
            cax.set_xlabel(labels[weight], fontsize=fontsize)
            cax.tick_params(labelsize=fontsize - 1., direction='in')
            cax.set_aspect(1. / 8.)
            cax.xaxis.set_label_position('top')
            
            qimg = axq.imshow(plotdct[weight + 'SmZ']['map'].T, extent=plotdct[weight + 'SmZ']['extent'],\
                              origin='lower', interpolation='nearest', cmap=cmaps['Z'], vmin=vminmax['Z'][0], vmax=vminmax['Z'][1])
        
        plt.colorbar(qimg, cax=cax_q, orientation='vertical', extend='both')
        cax_q.set_ylabel(labels['Z'], fontsize=fontsize)
        cax_q.tick_params(labelsize=fontsize - 1.)
        cax_q.set_aspect(8.)
        #cbar_q = plt.colorbar(...)
        #cbar_q.set_ticks(ticks) 
        #cax_q.xaxis.set_label_position('top')
        #cax_q.set_yticklabels(<list>)
        
        for ri in range(numw - 1):
            for ci in range(numw - 1):
                if ci > ri:
                    continue
                w1 = weights[ri + 1] + 'SmZ'
                w2 = weights[ci] + 'SmZ'
                ax = fig.add_subplot(maingrid[5 + ri, ci])
                map1 = plotdct[w1]['map']
                ext1 = plotdct[w1]['extent']
                map2 = plotdct[w2]['map']
                ext2 = plotdct[w2]['extent']
                
                if np.max(np.abs(np.array(ext1) - np.array(ext2))) > 1e-6:
                    raise RuntimeError('Extents of different weighted Zs do not match: %s, %s'%(str(ext1), str(ext2)))
                    
                img = ax.imshow(map2.T - map1.T, extent=ext1, origin='lower', interpolation='nearest', cmap=cmaps['diff'], vmin=vminmax['diff'][0], vmax=vminmax['diff'][1])
                ax.text(0.02, 0.98, '%s - %s'%(Zlabels[w2[:-3]], Zlabels[w1[:-3]]), fontsize=fontsize, transform=ax.transAxes, horizontalalignment='left', verticalalignment='top')
                ax.tick_params(which='both', direction='in', left=False, bottom=False, labelleft=False, labelbottom=False, right=False, top=False)
        
        ticks = np.linspace(vminmax['diff'][0], vminmax['diff'][1], 5)
        cax_d = fig.add_subplot(maingrid[5, 1])
        plt.colorbar(img, cax=cax_d, orientation='horizontal', extend='both', ticks=ticks)
        cax_d.set_xlabel(Zdifflabel, fontsize=fontsize)
        cax_d.tick_params(labelsize=fontsize - 1., direction='in')
        cax_d.set_aspect(1. / 8.)
        cax_d.xaxis.set_label_position('top')
        
        tax = fig.add_subplot(maingrid[5:7, 2:numw])
        tax.text(0.05, 1.0, titletext, fontsize=fontsize, transform=tax.transAxes, horizontalalignment='left', verticalalignment='top')
        tax.axis('off')
        
        figname = mdir + 'imgplot_%s_%i.pdf'%(setname, galid)
        plt.savefig(figname, format='pdf', bbox_inches='tight')


def plot_Zmeas_basecomp(hn=False, add_sample=None, galids=None):
    fontsize = 12.
    if add_sample is not None and galids is not None:
        s_tag = add_sample
        if galids == 'all':
            s_tag = '_' + s_tag + '-' + galids
        else:
            s_tag = '_' + s_tag + '-' + '-'.join([str(galid) for galid in galids])
        legheight = 1.
        mdname = mdir + add_sample + '_projfiles.txt'
    else:
        s_tag = ''
        legheight = 0.
        
    if hn:
        hist = pfn.ea25RecZmeas_basecomp_hn
        outname = 'Mass_hneutralssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection'
    else:
        hist = pfn.ea25RecZmeas_basecomp
        outname = 'Mass_h1ssh_SFR_L0025N0752RECALIBRATED_19_test3.31_C2Sm_10000pix_3.125slice_zcen-all_z-projection'
    outname = outname + s_tag + '.pdf'
    
    msax = 0
    h1ax = 1
    sfax = 2

    labels = {msax: r'$\log_{10} \, \Sigma_{\mathrm{gas}}\; [\mathrm{M}_{\odot} \, \mathrm{pkpc}^{-2}]$',\
              h1ax: r'$\log_{10} \, N_{\mathrm{H\,I}} \; [\mathrm{cm}^{-2}]$',\
              sfax: r'$\log_{10} \, \mathrm{SFR} \; [\mathrm{M}_{\odot}\, \mathrm{pkpc}^{-2} \mathrm{yr}^{-1}]$',\
              }
    
    convs = {msax: c.solar_mass / (c.cm_per_mpc / 1.e3)**2,\
             h1ax: 1.,\
             sfax: c.solar_mass / (c.cm_per_mpc / 1.e3)**2 / c.sec_per_year,\
            }
    
    lims = {msax: (4., 9.0), h1ax: (11., 23.5), sfax:(-8., 0.5)}
    
    fig = plt.figure(figsize=(10.5, 3.5))
    grid = gsp.GridSpec(nrows=2, ncols=4, width_ratios=[1., 1., 1., 0.2], height_ratios=[2., legheight], wspace=0.45, hspace=0.0, top=0.90, bottom=0.05, left=0.05) # grispec: nrows, ncols
    mainaxes = np.array([fig.add_subplot(grid[0, xi]) for xi in range(3)]) # in mainaxes: x = column, y = row
    cax = fig.add_subplot(grid[0, 3])
    
    sumaxes =  [(sfax,), (h1ax,), (msax,)]
    plotaxes = [(msax, h1ax), (msax, sfax), (h1ax, sfax)]
    vmaxs = [pu.getminmax2d(hist['bins'], hist['edges'], axis=i, log=True, pixdens=True)[1] for i in sumaxes]
    vmax = max(vmaxs)
    vmin = vmax - 8.
    
    cmap = 'gist_yarg'
    #colors = ['saddlebrown', 'maroon', 'red', 'orange', 'gold', 'forestgreen', 'lime', 'cyan', 'blue', 'purple', 'magenta']
    #dimlabels = (labels[0], labels[1], labels[2], labels[3])
    #dimshifts = (-1.*np.log10(convs[i]) for i in range(4))
    #edgeinds = [np.argmin(np.abs(contourbins[i] - hist['edges'][0] + np.log10(measconv))) for i in range(len(contourbins))]
    #mins = [(edge,) + (None,) * 3 for edge in [None] + edgeinds]
    #maxs = [(edge,) + (None,) * 3 for edge in edgeinds + [None]]

    percarr = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    linestyles_perc = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
    colors_perc = ['magenta'] *len(percarr)

    for i in range(3):
        ax = mainaxes[i]
        sumaxis = sumaxes[i]
        plotaxis = plotaxes[i]
        plotaxis = tuple(plotaxis)
        imgminmax = pu.add_2dplot(ax, hist['bins'], hist['edges'], plotaxis,\
                                  log=True, usepcolor=True, pixdens=True,\
                                  shiftx=-1.*np.log10(convs[plotaxis[0]]),\
                                  shifty=-1.*np.log10(convs[plotaxis[1]]),\
                                  cmap=cmap, vmin=vmin, vmax=vmax)
        
        #for j in range(len(mins)):
        #    add_2dhist_contours(ax, hist, plotaxes, mins=mins[j], maxs=maxs[j],\
        #                        histlegend=False, fraclevels=True, levels=levels, legend=False,\
        #                        dimlabels=dimlabels, legendlabel=None, legendlabel_pre=None,\
        #                        shiftx=-1.*np.log10(convs[plotaxes[0]]), shifty=-1.*np.log10(convs[plotaxes[1]]), dimshifts=dimshifts,\
        #                        colors=[colors[j]] * len(levels), linestyles=linestyles)
        #
        #
        
        subhist = np.sum(hist['bins'], axis=sumaxis)
        if plotaxis[0] > plotaxis[1]:
            subhist=subhist.T
        percentiles = pu.percentiles_from_histogram(subhist, hist['edges'][plotaxis[1]], axis=1, percentiles=percarr)
        cens_ax0 = hist['edges'][plotaxis[0]]
        cens_ax0 = cens_ax0[:-1] + 0.5 * np.diff(cens_ax0)
        plotcrit = np.sum(subhist, axis=1) * hist.npt >= 10.
        plotsl   = slice(np.where(plotcrit)[0][0], np.where(plotcrit)[0][-1] + 1, None)
        for pi in range(len(percentiles)):
            ax.plot(cens_ax0[plotsl] - np.log10(convs[plotaxis[0]]), percentiles[pi, plotsl]- np.log10(convs[plotaxis[1]]), linestyle=linestyles_perc[pi], color=colors_perc[pi], label='%.0f %%'%(percarr[pi] * 100.))
        
        handles_subs, labels_subs = ax.get_legend_handles_labels()
        
        ax.set_xlabel(labels[plotaxis[0]], fontsize=fontsize)
        ax.set_ylabel(labels[plotaxis[1]], fontsize=fontsize)
        pu.setticks(ax, fontsize, color='black', labelbottom=True, top=True, labelleft=True, labelright=False, right=True, labeltop=False)
        
        
        ax.set_xlim(*lims[plotaxis[0]])
        ax.set_ylim(*lims[plotaxis[1]])
        
    pu.add_colorbar(cax, img=imgminmax[0], vmin=vmin, vmax=vmax, cmap=cmap, \
                    clabel=r'$\log_{10}\, \mathrm{absorber\, fraction} / \mathrm{pix. size} \; [\mathrm{dex}^{-2}] $',
                    newax=False, extend='min', fontsize=fontsize, orientation='vertical')
    cax.set_aspect(10., adjustable='box-forced')
    cax.tick_params(labelsize=fontsize)
    
    #handles_encl = [mlines.Line2D([], [], color='gray', linestyle=linestyles[i], label='%.0f %%'%(levels[i]*100.)) for i in range(len(levels))]
    handles_encl = []
    mainaxes[2].legend(handles=handles_subs + handles_encl, fontsize=fontsize - 1, ncol=1, loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False)
    
    leghandlelist = []
    # alpha depends on sumaxes: much fewer points when SF = 0 not included
    alphas = {sfax: 0.05,\
              msax: 0.15,\
              h1ax: 0.15,\
              }
    if add_sample is not None and galids is not None:
        wtmap = {msax: 'Gasmass',\
                 h1ax: 'hneutralssh' if hn else 'h1ssh',\
                 sfax: 'SFR'}
        
        filenames_all = pd.read_csv(mdname, sep='\t')
        if galids == 'all':
            galids = list(set(np.array(filenames_all['galaxyid'])))
            galids.sort()
        
        for gind in range(len(galids)):
            galid = galids[gind]
            color = 'C%i'%(gind%10)
            
            #filenames = filenames_all.loc[filenames_all['galaxyid'] == galid]    
            weights = [wtmap[key] for key in wtmap]
            plotdct = {}
            for weight in weights:
                loc = np.where(np.logical_and(filenames_all['weight'] == weight, filenames_all['galaxyid'] == galid))[0][0]
                pfile = filenames_all.at[loc, 'pfile']
                #qfile = filenames_all.at[loc, 'qfile']
                
                plotdct[weight] = readoutmap_forimgplot(pfile, units[weight], cosmopars_default[19])
                #plotdct[weight + 'SmZ'] = readoutmap_forimgplot(qfile, units['Z'], cosmopars_default[19])
            
            for axi in range(3):
                xvals = (plotdct[wtmap[plotaxes[axi][0]]]['map']).flatten()
                yvals = (plotdct[wtmap[plotaxes[axi][1]]]['map']).flatten()
                depth = plotdct[wtmap[plotaxes[axi][1]]]['depth']
                width = plotdct[wtmap[plotaxes[axi][1]]]['extent']
                width = width[1] - width[0]
                mainaxes[axi].scatter(xvals, yvals, color=color, alpha=alphas[sumaxes[axi][0]])
            label = r'%i: ${%.2f}^2 \times %.2f$ pkpc'%(galid, width, depth)
            leghandlelist.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', alpha=0.5, label=label))
        lax = fig.add_subplot(grid[1, :])
        lax.legend(handles=leghandlelist, fontsize=fontsize - 1, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.5), frameon=False)
        lax.axis('off')
        
    plt.savefig(mdir + outname, format='pdf', bbox_inches='tight')  


    