#import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import aplpy as apl
from astropy.io import fits
import numpy as np
import config_figs
from sauron_colormap import register_sauron_colormap
try:
    register_sauron_colormap()
except:
    pass
from matplotlib.colors import ListedColormap
from glob import glob
import os


def moment_zero(mom0, galaxy, path, spec_res=10, savename=None, units='Jy/beam km/s', alpha_co=4.35, peak=False):

    fig = plt.figure(figsize=(11, 8))

    # show the image in colour
    f = apl.FITSFigure(mom0, figure=fig)
    #f.set_theme('publication')

    # add the galaxy name in the upper right corner
    f.add_label(0.8, 0.9, galaxy, relative=True, size=30)

    f.show_contour(mom0, cmap='magma_r', 
                   levels=np.linspace(np.nanmax(mom0.data)*1e-9, np.nanmax(mom0.data), 20),
                   filled=True, overlap=True)

    # axes and ticks
    f.ticks.set_color('black')
    f.ticks.set_length(10)
    f.ticks.set_linewidth(2)
    f.tick_labels.set_xformat('hh:mm:ss')
    f.tick_labels.set_yformat('dd:mm:ss')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f.ticks.set_minor_frequency(5)

    # add a colourbar
    colors = plt.contourf([[0, 0], [0, 0]],
                          levels=np.linspace(0, np.nanmax(mom0.data) + np.nanmax(mom0.data) * 0.05, 20),
                          cmap='magma_r')

    if np.nanmax(mom0.data) < 0.1:
        ticks = np.arange(0, np.nanmax(mom0.data) + 0.015, 0.005)
    if np.nanmax(mom0.data) < 0.5:
        ticks = np.arange(0, np.nanmax(mom0.data) + 0, 0.05)
    elif np.nanmax(mom0.data) < 1:
        ticks = np.arange(0, np.nanmax(mom0.data) + 0.1, 0.1)
    elif np.nanmax(mom0.data) < 2:
        ticks = np.arange(0, np.nanmax(mom0.data) + 0.2, 0.2)
    elif np.nanmax(mom0.data) < 5:
        ticks = np.arange(0, np.nanmax(mom0.data) + 0.5, 0.5)
    elif np.nanmax(mom0.data) < 10:
        ticks = np.arange(0, np.nanmax(mom0.data) + 1, 1)
    elif np.nanmax(mom0.data) < 20:
        ticks = np.arange(0, np.nanmax(mom0.data) + 1, 2)
    elif np.nanmax(mom0.data) < 100:
        ticks = np.arange(0, np.nanmax(mom0.data) + 5, 10)
    elif np.nanmax(mom0.data) < 200:
        ticks = np.arange(0, np.nanmax(mom0.data) + 10, 20)
    elif np.nanmax(mom0.data) < 1000:
        ticks = np.arange(0, np.nanmax(mom0.data) + 20, 40)
    else:
        pass
        #ticks = np.arange(0, np.nanmax(mom0.data) + 100, 200)

    try:
        cbar = fig.colorbar(colors, ticks=ticks)
    except:
        cbar = fig.colorbar(colors)

    if peak:
        cbar.set_label('Peak temperature [K]')
    elif units == 'Jy/beam km/s':
        cbar.set_label(r'Integrated intensity [Jy b$^{-1}$ km s$^{-1}$]')
    elif units == 'K km/s pc^2':
        if 'err' in savename:
            cbar.set_label(r'CO luminosity error [K km s$^{-1}$ pc$^2$]')
        else:
            cbar.set_label(r'CO luminosity [K km s$^{-1}$ pc$^2$]')
    elif units == 'K km/s':
        if 'err' in savename:
            cbar.set_label(r'CO surface brightness error [K km s$^{-1}$]')
        else:
            cbar.set_label(r'CO surface brightness [K km s$^{-1}$]')
    elif units == 'Msol pc-2':
        #if 'err' in savename:
        #    cbar.set_label(r'mol. mass surface density error [dex]')
        #else:
        cbar.set_label(r'mol. mass surface density [M$_\odot$ pc$^{-2}$]')
    elif units == 'Msol/pix':
        #if 'err' in savename:
        #    cbar.set_label(r'mol. mass error [dex]')
        #else:
        cbar.set_label(r'mol. mass [M$_\odot$]')
    else:
        raise AttributeError('Please choose between "K km/s" and "M_Sun/pc^2"')

    # show the beam of the observations
    f.add_beam(frame=False, linewidth=5)  # automatically imports BMAJ, BMIN, and BPA
    f.beam.set_edgecolor('k')
    f.beam.set_facecolor('None')
    f.beam.set_borderpad(1)

    # show a scalebar
    #if self.galaxy.distance:
    #    length = np.degrees(1e-3 / self.galaxy.distance)  # length of the scalebar in degrees, corresponding to 1 kpc
    #    fig.add_scalebar(length=length, label='1 kpc', frame=False)
    #    fig.scalebar.set_linewidth(5)

    #plt.tight_layout()

    if savename:
        if spec_res == 10:
            plt.savefig(path + savename + '.png', bbox_inches='tight')
            plt.savefig(path.split('by_galaxy')[0] + 'by_product/moment_maps/10kms/' + savename + '.png', bbox_inches='tight') 
            plt.savefig(path + savename + '.pdf', bbox_inches='tight')
            plt.savefig(path.split('by_galaxy')[0] + 'by_product/moment_maps/10kms/' + savename + '.pdf', bbox_inches='tight') 
        elif spec_res == 30:          
            plt.savefig(path + savename + '.png', bbox_inches='tight')
            plt.savefig(path.split('by_galaxy')[0] + 'by_product/moment_maps/30kms/' + savename + '.png', bbox_inches='tight') 
            plt.savefig(path + savename + '.pdf', bbox_inches='tight')
            plt.savefig(path.split('by_galaxy')[0] + 'by_product/moment_maps/30kms/' + savename + '.pdf', bbox_inches='tight') 


def moment_1_2(mom, galaxy, moment, path, spec_res=10, savename=None):

    vel_array = np.load(path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/' + savename.split('_mom1')[0].split('_mom2')[0] + '_vel_array.npy')
    
    #sysvel = (sysvel + 5) // 10 * 10

    fig = plt.figure(figsize=(11, 8))

    # show the image in colour
    f = apl.FITSFigure(mom, figure=fig)

    # axes and ticks
    f.ticks.set_color('black')
    f.ticks.set_length(10)
    f.ticks.set_linewidth(2)
    f.tick_labels.set_xformat('hh:mm:ss')
    f.tick_labels.set_yformat('dd:mm:ss')
    f.ticks.show()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f.ticks.set_minor_frequency(5)

    #add a colourbar
    if moment == 2:
        #vrange2 = 5 * np.nanmedian(mom.data)
        vrange2 = np.nanmax(mom.data[np.isfinite(mom.data)])

        f.show_contour(mom, cmap='sauron', levels=np.linspace(0, vrange2, len(vel_array)), vmin=0,
                         vmax=vrange2, extend='both', filled=True, overlap=True)
        
        colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(0, vrange2, len(vel_array)),
                              cmap='sauron')

        if vrange2 < 11:
            ticks = np.arange(0, vrange2 + 1, 1)
        elif vrange2 < 100:
            ticks = np.arange(0, vrange2 + 10, 10)
        elif vrange2 < 500:
            ticks = np.arange(0, vrange2 + 20, 20)
        else:
            ticks = []
            
        cbar = fig.colorbar(colors, ticks=ticks)
        if 'err' in savename:
            cbar.set_label(r'Observed $\sigma_v$ error [km s$^{-1}$]')
        else:
            cbar.set_label(r'Observed $\sigma_v$ [km s$^{-1}$]')

    elif moment == 1:
        #vrange = int(vel_array[0])

        #f.show_contour(mom, cmap='sauron', levels=np.linspace(-vrange, vrange,
        #    len(vel_array)), vmin=-vrange, vmax=vrange, extend='both', filled=True,
        #                 overlap=True)
        
        #colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(-vrange, vrange,
        #                                                           len(vel_array)), cmap='sauron')
        
        vmin = np.nanmin(mom.data)
        vmax = np.nanmax(mom.data)
        
        f.show_contour(mom, cmap='sauron', levels=np.linspace(vmin, vmax,
            len(vel_array)), vmin=vmin, vmax=vmax, extend='both', filled=True,
                         overlap=True)
        
        colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(vmin, vmax,
                                                                   len(vel_array)), cmap='sauron')
        
        vrange = vel_array[-1] - vel_array[0]
        
        if vrange < 16:
            tickarr = np.arange(round(vmin), round(vmax), 3)
        elif vrange < 60:
            tickarr = np.arange(round(vmin/5) * 5 - 5, round(vmax/5) * 5 + 5, 10)
        elif vrange < 130:
            tickarr = np.arange(round(vmin/10) * 10 - 10, round(vmax/10) * 10 + 10, 20)
        elif vrange < 1000:
            ticks = np.arange(round(vmin/10) * 10 - 10, round(vmax/10) * 10 + 10, 50)
        else:
            ticks = []

        try:
            cbar = fig.colorbar(colors, ticks=ticks)
        except:
            ticks = np.concatenate((tickarr, [0], abs(tickarr)))
            cbar = fig.colorbar(colors, ticks=ticks)
        if 'err' in savename:
            cbar.set_label(r'Velocity error [km s$^{-1}$]')
        else:
            cbar.set_label(r'Velocity [km s$^{-1}$]')

    # show the beam of the observations
    f.add_beam(frame=False, linewidth=5)  # automatically imports BMAJ, BMIN, and BPA
    f.beam.set_edgecolor('k')
    f.beam.set_facecolor('None')
    f.beam.set_borderpad(1)

    # show a scalebar
    #if self.galaxy.distance:
    #    length = np.degrees(1.e-3 / self.galaxy.distance)  # length of the scalebar in degrees, corresponding to 1 kpc
    #    fig.add_scalebar(length=length, label='1 kpc', frame=False)
    #    fig.scalebar.set_linewidth(5)

    #Make sure the axis labels don't fall off the figure
    #plt.tight_layout()

    if spec_res == 10:        
        plt.savefig(path + 'by_galaxy/' + galaxy + '/10kms/' + savename + '.png', bbox_inches='tight')
        plt.savefig(path + 'by_product/moment_maps/10kms/' + savename + '.png', bbox_inches='tight') 
        plt.savefig(path + 'by_galaxy/' + galaxy + '/10kms/' + savename + '.pdf', bbox_inches='tight')
        plt.savefig(path + 'by_product/moment_maps/10kms/' + savename + '.pdf', bbox_inches='tight') 
    elif spec_res == 30:       
        plt.savefig(path + 'by_galaxy/' + galaxy + '/10kms/' + savename + '.png', bbox_inches='tight')
        plt.savefig(path + 'by_product/moment_maps/30kms/' + savename + '.png', bbox_inches='tight') 
        plt.savefig(path + 'by_galaxy/' + galaxy + '/10kms/' + savename + '.pdf', bbox_inches='tight')
        plt.savefig(path + 'by_product/moment_maps/30kms/' + savename + '.pdf', bbox_inches='tight') 
        

def perform_moment_imaging(glob_path, targets, spec_res=10):
    
    files = glob(glob_path + 'by_galaxy/' + '**/')
    galaxies = list(set([f.split('/')[8].split('_')[0] for f in files]))
    
    for galaxy in galaxies:
        
        if not galaxy in targets:
            continue
        else:
            print(galaxy)

        if spec_res == 10:
            path = glob_path + 'by_galaxy/' + galaxy + '/10kms/'
        elif spec_res == 30:
            path = glob_path + 'by_galaxy/' + galaxy + '/30kms/'
        
        if os.path.exists(path + galaxy + '_Ico_K_kms-1.png'):
            continue
        
        mom0_K_kmss = glob(path + '*Ico*.fits')
        mom0_K_kms_pc2s = glob(path + '*Lco*.fits')
        mom0_Msol_pc2 = glob(path + '*mmol_pc-2*.fits')
        mom0_Msol_pix = glob(path + '*mmol_pix-1*.fits')
        
        peakTs = glob(path + '*peak_temp_k*.fits')
        mom1s = glob(path + '*mom1*.fits')
        mom2s = glob(path + '*mom2*.fits')
        
        #try:
        for mom0 in mom0_K_kmss:
            moment_zero(fits.open(mom0)[0], galaxy=galaxy, path=path, 
                        savename=mom0.split('/')[-1].split('.fits')[0], 
                        spec_res=spec_res, units='K km/s', alpha_co=4.35, peak=False)
        for mom0 in mom0_K_kms_pc2s:
            moment_zero(fits.open(mom0)[0], galaxy=galaxy, path=path, 
                        savename=mom0.split('/')[-1].split('.fits')[0], 
                        spec_res=spec_res, units='K km/s pc^2', alpha_co=4.35, peak=False)
        for mom0 in mom0_Msol_pc2:
            moment_zero(fits.open(mom0)[0], galaxy=galaxy, path=path, 
                        savename=mom0.split('/')[-1].split('.fits')[0], 
                        spec_res=spec_res, units='Msol pc-2', alpha_co=4.35, peak=False)
        for mom0 in mom0_Msol_pix:
            moment_zero(fits.open(mom0)[0], galaxy=galaxy, path=path, 
                        savename=mom0.split('/')[-1].split('.fits')[0], 
                        spec_res=spec_res, units='Msol/pix', alpha_co=4.35, peak=False)
        for peakT in peakTs:
            moment_zero(fits.open(peakT)[0], galaxy=galaxy, path=path, 
                        savename=peakT.split('/')[-1].split('.fits')[0], 
                        spec_res=spec_res, peak=True)
        for mom1 in mom1s:
            moment_1_2(fits.open(mom1)[0], savename=mom1.split('/')[-1].split('.fits')[0], galaxy=galaxy, moment=1, path=glob_path, spec_res=spec_res)
        for mom2 in mom2s:
            moment_1_2(fits.open(mom2)[0], savename=mom2.split('/')[-1].split('.fits')[0], galaxy=galaxy, moment=2, path=glob_path, spec_res=spec_res)
                
        #except:
        #    print(galaxy)

if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    perform_moment_imaging(path)










