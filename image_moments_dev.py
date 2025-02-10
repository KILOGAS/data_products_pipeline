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


def moment_zero(galaxy, units='Jy/beam km/s', alpha_co=5.4):
    
    if ifumatched:
        mom0 = fits.open(path + galaxy + '_smoothed_ifumatched_mom0_Jyb-1_kms-1.fits')[0]
    elif smoothed:
        mom0 = fits.open(path + galaxy + '_smoothed_mom0_Jyb-1_kms-1.fits')[0]
    else:
        mom0 = fits.open(path + galaxy + '_mom0_Jyb-1_kms-1.fits')[0]
        
    #print(mom0.header)

    fig = plt.figure(figsize=(11, 8))

    # show the image in colour
    f = apl.FITSFigure(mom0, figure=fig)
    f.set_theme('publication')

    # add the galaxy name in the upper right corner
    f.add_label(0.8, 0.9, galaxy, relative=True, size=30)
    
    plt.figure()
    plt.imshow(mom0.data)
    
    print(np.linspace(np.nanmax(mom0.data)*1e-9, np.nanmax(mom0.data), 20))

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
        ticks = np.arange(0, np.nanmax(mom0.data) + 100, 200)

    cbar = fig.colorbar(colors, ticks=ticks)

    if units == 'Jy/beam km/s':
        cbar.set_label(r'Integrated intensity [Jy b$^{-1}$ km s$^{-1}$]')
    elif units == 'M_Sun/pc^2':
        cbar.set_label(r'Surface density [M$_\odot$ pc$^{-2}$]')
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

    plt.tight_layout()

    if units == 'Jy/beam km/s':
        if smoothed:
            if ifumatched:
                plt.savefig(path + galaxy + '_smoothed_ifumatched_mom0_Jyb-1_kms-1.pdf', bbox_inches='tight')
                plt.savefig(path + galaxy + '_smoothed_ifumatched_mom0_Jyb-1_kms-1.png', bbox_inches='tight')
            else:
                plt.savefig(path + galaxy + '_smoothed_mom0_Jyb-1_kms-1.pdf', bbox_inches='tight')
                plt.savefig(path + galaxy + '_smoothed_mom0_Jyb-1_kms-1.png', bbox_inches='tight')
        else:
            plt.savefig(path + galaxy + '_mom0_Jyb-1_kms-1.pdf', bbox_inches='tight')
            plt.savefig(path + galaxy + '_mom0_Jyb-1_kms-1.png', bbox_inches='tight')
    #elif units == 'M_Sun/pc^2':
    #    plt.savefig(self.savepath + 'mom0_Msolpc-2.pdf', bbox_inches='tight')


def moment_1_2(galaxy, moment):

    if moment == 1:
        if ifumatched:
            mom = fits.open(path + galaxy + '_smoothed_ifumatched_mom1.fits')[0]
        elif smoothed:
            mom = fits.open(path + galaxy + '_smoothed_mom1.fits')[0]
        else:
            mom = fits.open(path + galaxy + '_mom1.fits')[0]
    elif moment == 2:
        if ifumatched:
            mom = fits.open(path + galaxy + '_smoothed_ifumatched_mom2.fits')[0]
        elif smoothed:
            mom = fits.open(path + galaxy + '_smoothed_mom2.fits')[0]
        else:
            mom = fits.open(path + galaxy + '_mom2.fits')[0]
    
    vel_array = np.load(path + galaxy + '_vel_array.npy')

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
        vrange2 = np.nanmax(mom.data)

        f.show_contour(mom, cmap='sauron', levels=np.linspace(0, vrange2, len(vel_array)), vmin=0,
                         vmax=vrange2, extend='both', filled=True, overlap=True)
        
        colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(0, vrange2, len(vel_array)),
                              cmap='sauron')

        if vrange2 < 11:
            ticks = np.arange(0, vrange2 + 1, 1)
        elif vrange2 < 100:
            ticks = np.arange(0, vrange2 + 10, 10)
        else:
            ticks = np.arange(0, vrange2 + 20, 20)
            
        cbar = fig.colorbar(colors, ticks=ticks)
        cbar.set_label(r'Observed $\sigma_v$ [km s$^{-1}$]')

    elif moment == 1:
        #vrange = int(vel_array[0])

        #f.show_contour(mom, cmap='sauron', levels=np.linspace(-vrange, vrange,
        #    len(vel_array)), vmin=-vrange, vmax=vrange, extend='both', filled=True,
        #                 overlap=True)
        
        #colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(-vrange, vrange,
        #                                                           len(vel_array)), cmap='sauron')
        
        #vmin = 13450
        #vmax = 13850
        
        vmin = np.nanmin(mom.data)
        vmax = np.nanmax(mom.data)
        
        f.show_contour(mom, cmap='sauron', levels=np.linspace(vmin, vmax,
            len(vel_array)), vmin=vmin, vmax=vmax, extend='both', filled=True,
                         overlap=True)
        
        colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(vmin, vmax,
                                                                   len(vel_array)), cmap='sauron')
        
        #if vrange < 16:
        #    tickarr = np.arange(-vrange, 0, 3)
        #elif vrange < 60:
        #    tickarr = np.arange(-vrange, 0, 10)
        #elif vrange < 130:
        #    tickarr = np.arange(-vrange, 0, 20)
        #else:
        ticks = np.arange(round(vmin/10) * 10 - 10, round(vmax/10) * 10 + 10, 20)

        #ticks = np.concatenate((tickarr, [0], abs(tickarr)))
        cbar = fig.colorbar(colors, ticks=ticks)
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
    plt.tight_layout()

    if moment == 1:
        if smoothed:
            if ifumatched:
                plt.savefig(path + galaxy + '_smoothed_ifumatched_mom1.pdf', bbox_inches='tight')
                plt.savefig(path + galaxy + '_smoothed_ifumatched_mom1.png', bbox_inches='tight')
            else:
                plt.savefig(path + galaxy + '_smoothed_mom1.pdf', bbox_inches='tight')
                plt.savefig(path + galaxy + '_smoothed_mom1.png', bbox_inches='tight')
        else:
            plt.savefig(path + galaxy + '_mom1.pdf', bbox_inches='tight')
            plt.savefig(path + galaxy + '_mom1.png', bbox_inches='tight')
            
    elif moment == 2:
        if smoothed:
            if ifumatched:
                plt.savefig(path + galaxy + '_smoothed_ifumatched_mom2.pdf', bbox_inches='tight')
                plt.savefig(path + galaxy + '_smoothed_ifumatched_mom2.png', bbox_inches='tight')
            else:
                plt.savefig(path + galaxy + '_smoothed_mom2.pdf', bbox_inches='tight')
                plt.savefig(path + galaxy + '_smoothed_mom2.png', bbox_inches='tight')
        else:
            plt.savefig(path + galaxy + '_mom2.pdf', bbox_inches='tight')
            plt.savefig(path + galaxy + '_mom2.png', bbox_inches='tight')


#path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/IFU_matched_cubes/moment_maps/'
glob_path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'

files = glob(glob_path + '**/')
galaxies = list(set([f.split('/')[6].split('_')[0] for f in files]))

ifumatched = False
smoothed = False

for galaxy in galaxies:
    
    path = glob_path + '/' + galaxy + '/moment_maps/'

    moment_zero(galaxy=galaxy, units='Jy/beam km/s', alpha_co=5.4)
    moment_1_2(galaxy=galaxy, moment=1)
    moment_1_2(galaxy=galaxy, moment=2)













