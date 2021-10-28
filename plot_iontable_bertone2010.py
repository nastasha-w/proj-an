
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from matplotlib import cm

import make_maps_v3_master as m3
import ion_line_data as ild
import plot_utils as pu
import cosmo_utils as cu
import eagle_constants_and_units as c
import make_maps_opts_locs as ol

def plot_iontab(ion, z):
    balance, logTK, lognHcm3 = m3.findiontables(ion, z)
    balance = np.log10(balance)
    
    dT = np.average(np.diff(logTK))
    eT = logTK - 0.5 * dT
    eT = np.append(eT, [logTK[-1] + 0.5 * dT])
    
    dn = np.average(np.diff(lognHcm3))
    en = lognHcm3 - 0.5 * dn
    en = np.append(en, [lognHcm3[-1] + 0.5 * dn])
    
    Tlabel = '$\\log_{10} \\, \\mathrm{T} \\; [\\mathrm{K}]$'
    nlabel = '$\\log_{10}\\,\\mathrm{n}_{\\mathrm{H}}\\;[\\mathrm{cm}^{-3}]$'
    clabel = '$\\log_{{10}}\\, \\mathrm{{f}}_{{\\mathrm{{{ion}}}}}$'
    clabel = clabel.format(ion=ild.getnicename(ion, mathmode=True))
    fontsize = 12
    
    fig = plt.figure(figsize=(5.5, 5.))
    grid = gsp.GridSpec(ncols=2, nrows=1, hspace=0.0, wspace=0.1,
                        width_ratios=[10, 1])
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
            
    
    ax.tick_params(which='both', direction='in', labelsize=fontsize - 1.,
                   top=True, right=True)
    #ax.grid(b=True)
    ax.set_xlabel(nlabel, fontsize=fontsize)
    ax.set_ylabel(Tlabel, fontsize=fontsize)
    
    ax.text(-1.5, 5.8, 'CIE', fontsize=fontsize, color='white',
            horizontalalignment='center', verticalalignment='center')
    ax.text(-6, 4., 'PIE', fontsize=fontsize, color='white',
            rotation=80, 
            horizontalalignment='center', verticalalignment='center')
    nhav = cu.rhocrit(z, cosmopars=None) * c.omegabaryon * 0.752 / (c.atomw_H * c.u)
    nhav = np.log10(nhav)
    
    ax.axvline(nhav, linestyle='dashed', linewidth=1., color='black')
    ax.text(nhav + 0.2, 8.1, '$\\overline{\\mathrm{n}_{\\mathrm{H}}}$', fontsize=fontsize,
            color='black', horizontalalignment='left', verticalalignment='top')
    
    _cmap = cm.get_cmap('gist_yarg')
    cmap = pu.truncate_colormap(_cmap, minval=0.0, maxval=0.7)
    img = ax.pcolormesh(en, eT, balance.T, cmap=cmap, vmin=-5., rasterized=True)
    pu.add_colorbar(cax, img=img, clabel=clabel, newax=False, 
                    extend='min', fontsize=fontsize, orientation='vertical')
                    
    outname = ol.mdir + 'iontable_Bertone-etal-2010_z{z}_{ion}'
    outname = outname.format(z=z, ion=ion)
    outname = outname.replace('.', 'p') + '.pdf'
    
    plt.savefig(outname, format='pdf', bbox_inches='tight')
    
if __name__ == '__main__':
    plot_iontab('o7', 0.1006)