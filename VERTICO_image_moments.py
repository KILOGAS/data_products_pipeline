import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import aplpy as apl
from astropy.io import fits
import numpy as np
import config_figs
from sauron_colormap import register_sauron_colormap; register_sauron_colormap()
from targets import galaxies
from create_moments import MomentMaps
from clip_cube import ClipCube
from matplotlib.colors import ListedColormap


class CreateImages:

    def __init__(self, galaxy, path_pbcorr, path_uncorr, savepath=None, tosave=False):
        self.galaxy = galaxy
        self.path_pbcorr = path_pbcorr
        self.path_uncorr = path_uncorr
        self.savepath = savepath or './'
        self.tosave = tosave


    def moment_zero(self, units='M_Sun/pc^2', path='', alpha_co=5.4, peak=False):

        if peak:
            if self.refresh:
                if self.overwrite:
                    image = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                                    savepath=self.savepath, tosave=True, sample=self.sample,
                                       redo_clip=self.redo_clip).peak_temperature()
                else:
                    image = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                                    tosave=False, sample=self.sample, redo_clip=self.redo_clip).\
                        peak_temperature()
            else:
                image = fits.open(path + 'peakT.fits')[0]

        elif self.refresh:
            if units == 'M_Sun/pc^2':
                if self.overwrite:
                    _, image, _, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                            savepath=self.savepath, tosave=True, sample=self.sample,
                                            redo_clip=self.redo_clip).calc_moms(units='M_Sun/pc^2', alpha_co=alpha_co)
                else:
                    _, image, _, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                                    tosave=False, sample=self.sample, redo_clip=self.redo_clip).\
                        calc_moms(units='M_Sun/pc^2')
            elif units == 'K km/s':
                if self.overwrite:
                    _, image, _, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                                    savepath=self.savepath, tosave=True, sample=self.sample,
                                                   redo_clip=self.redo_clip).calc_moms(units='K km/s')
                else:
                    _, image, _, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                                    tosave=False, sample=self.sample, redo_clip=self.redo_clip).\
                        calc_moms(units='K km/s')
            else:
                raise AttributeError('Please choose between "K km/s" and "M_Sun/pc^2"')
        elif units == 'M_Sun/pc^2':
            image = fits.open(self.savepath + '_mom0_Msolpc-2.fits')[0]
        else:
            image = fits.open(self.savepath + '_mom0_Kkms-1.fits')[0]

        f = plt.figure(figsize=self.galaxy.figsize)

        # show the image in colour
        fig = apl.FITSFigure(image, figure=f)
        fig.set_theme('publication')

        # add the galaxy name in the upper right corner
        fig.add_label(0.8, 0.9, self.galaxy.name, relative=True, size=20)

        fig.show_contour(image, cmap='magma_r', levels=np.linspace(np.amax(image.data)*1e-9, np.amax(image.data), 20),
                         filled=True, overlap=True)

        # axes and ticks
        fig.ticks.set_color('black')
        fig.ticks.set_length(10)
        fig.ticks.set_linewidth(2)
        fig.tick_labels.set_xformat('hh:mm:ss')
        fig.tick_labels.set_yformat('dd:mm:ss')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig.ticks.set_minor_frequency(5)

        # add a colourbar
        colors = plt.contourf([[0, 0], [0, 0]],
                              levels=np.linspace(0, np.amax(image.data) + np.amax(image.data) * 0.05, 20),
                              cmap='magma_r')

        if np.amax(image.data) < 0.1:
            ticks = np.arange(0, np.amax(image.data) + 0.015, 0.005)
        if np.amax(image.data) < 0.5:
            ticks = np.arange(0, np.amax(image.data) + 0, 0.05)
        elif np.amax(image.data) < 1:
            ticks = np.arange(0, np.amax(image.data) + 0.1, 0.1)
        elif np.amax(image.data) < 2:
            ticks = np.arange(0, np.amax(image.data) + 0.2, 0.2)
        elif np.amax(image.data) < 5:
            ticks = np.arange(0, np.amax(image.data) + 0.5, 0.5)
        elif np.amax(image.data) < 10:
            ticks = np.arange(0, np.amax(image.data) + 1, 1)
        elif np.amax(image.data) < 20:
            ticks = np.arange(0, np.amax(image.data) + 1, 2)
        elif np.amax(image.data) < 100:
            ticks = np.arange(0, np.amax(image.data) + 5, 10)
        elif np.amax(image.data) < 200:
            ticks = np.arange(0, np.amax(image.data) + 10, 20)
        elif np.amax(image.data) < 1000:
            ticks = np.arange(0, np.amax(image.data) + 20, 40)
        else:
            ticks = np.arange(0, np.amax(image.data) + 100, 200)

        cbar = f.colorbar(colors, ticks=ticks)
        if peak:
            if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                cbar.set_label('Peak temperature [Jy b$^{-1}$]')
            else:
                cbar.set_label('Peak temperature [K]')
        elif units == 'K km/s':
            if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                cbar.set_label(r'Integrated intensity [Jy b$^{-1}$ km s$^{-1}$]')
            else:
                cbar.set_label(r'Integrated intensity [K km s$^{-1}$]')
        elif units == 'M_Sun/pc^2':
            cbar.set_label(r'Surface density [M$_\odot$ pc$^{-2}$]')
        else:
            raise AttributeError('Please choose between "K km/s" and "M_Sun/pc^2"')

        # show the beam of the observations
        fig.add_beam(frame=False, linewidth=5)  # automatically imports BMAJ, BMIN, and BPA
        fig.beam.set_edgecolor('k')
        fig.beam.set_facecolor('None')
        fig.beam.set_borderpad(1)

        # show a scalebar
        if self.galaxy.distance:
            length = np.degrees(1e-3 / self.galaxy.distance)  # length of the scalebar in degrees, corresponding to 1 kpc
            fig.add_scalebar(length=length, label='1 kpc', frame=False)
            fig.scalebar.set_linewidth(5)

        plt.tight_layout()

        if self.tosave:
            if peak:
                plt.savefig(self.savepath + 'peakT.pdf', bbox_inches='tight')
            elif units == 'K km/s':
                if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                    plt.savefig(self.savepath + 'mom0_Jyb-1kms-1.pdf', bbox_inches='tight')
                else:
                    plt.savefig(self.savepath + 'mom0_Kkms-1.pdf', bbox_inches='tight')
            elif units == 'M_Sun/pc^2':
                plt.savefig(self.savepath + 'mom0_Msolpc-2.pdf', bbox_inches='tight')
        return


    def moment_1_2(self, moment=1):

        if moment == 1:
            if self.refresh:
                if self.overwrite:
                    _, _, image, _, sysvel = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                                         savepath=self.savepath, sun=self.sun, tosave=True,
                                                        sample=self.sample, redo_clip=self.redo_clip).calc_moms()
                else:
                    _, _, image, _, sysvel = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                                         sun=self.sun, tosave=False, sample=self.sample,
                                                        redo_clip=self.redo_clip).calc_moms()
            else:
                image = fits.open(self.path + 'mom1.fits')[0]

        elif moment == 2:
            if self.refresh:
                if self.overwrite:
                    _, _, _, image, sysvel = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                                         savepath=self.savepath, sun=self.sun, tosave=True,
                                                        sample=self.sample, redo_clip=self.redo_clip).calc_moms()
                else:
                    _, _, _, image, sysvel = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                                         sun=self.sun, tosave=False, sample=self.sample,
                                                        redo_clip=self.redo_clip).calc_moms()
            else:
                image = fits.open(self.path + 'mom2.fits')[0]

        cube_pbcorr, cube_uncorr = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                               sun=self.sun, tosave=False, sample=self.sample).readfits()
        emiscube, noisecube = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                            savepath=self.savepath,
                                            tosave=self.tosave, sample=self.sample).split_cube(cube_uncorr)
        vel_array, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                      tosave=False, sample=self.sample, redo_clip=self.redo_clip).\
            create_vel_array(emiscube)

        #sysvel = (sysvel + 5) // 10 * 10

        f = plt.figure(figsize=self.galaxy.figsize)

        # show the image in colour
        fig = apl.FITSFigure(image, figure=f)

        # axes and ticks
        fig.ticks.set_color('black')
        fig.ticks.set_length(10)
        fig.ticks.set_linewidth(2)
        fig.tick_labels.set_xformat('hh:mm:ss')
        fig.tick_labels.set_yformat('dd:mm:ss')
        fig.ticks.show()
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig.ticks.set_minor_frequency(5)

        #add a colourbar
        if moment == 2:
            if self.galaxy.vrange2:
                vrange2 = self.galaxy.vrange2
            else:
                vrange2 = 5 * np.nanmedian(image.data)

            fig.show_contour(image, cmap='sauron', levels=np.linspace(0, vrange2, len(vel_array)), vmin=0,
                             vmax=vrange2, extend='both', filled=True, overlap=True)
            colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(0, vrange2, len(vel_array)),
                                  cmap='sauron')

            if vrange2 < 11:
                ticks = np.arange(0, vrange2 + 1, 1)
            elif vrange2 < 100:
                ticks = np.arange(0, vrange2 + 10, 10)
            else:
                ticks = np.arange(0, vrange2 + 20, 20)
            cbar = f.colorbar(colors, ticks=ticks)
            cbar.set_label(r'Observed $\sigma_v$ [km s$^{-1}$]')

        else:
            if self.galaxy.vrange:
                vrange = self.galaxy.vrange
            else:
                vrange = int(vel_array[0] - sysvel)

            fig.show_contour(image, cmap='sauron', levels=np.linspace(-vrange, vrange,
                len(vel_array)), vmin=-vrange, vmax=vrange, extend='both', filled=True,
                             overlap=True)
            colors = plt.contourf([[0, 0], [0, 0]], levels=np.linspace(-vrange, vrange,
                                                                       len(vel_array)), cmap='sauron')
            if vrange < 16:
                tickarr = np.arange(-vrange, 0, 3)
            elif vrange < 60:
                tickarr = np.arange(-vrange, 0, 10)
            elif vrange < 130:
                tickarr = np.arange(-vrange, 0, 20)
            else:
                tickarr = np.arange(-vrange, 0, 40)

            ticks = np.concatenate((tickarr, [0], abs(tickarr)))
            cbar = f.colorbar(colors, ticks=ticks)
            cbar.set_label(r'Velocity [km s$^{-1}$]')

        # show the beam of the observations
        fig.add_beam(frame=False, linewidth=5)  # automatically imports BMAJ, BMIN, and BPA
        fig.beam.set_edgecolor('k')
        fig.beam.set_facecolor('None')
        fig.beam.set_borderpad(1)

        # show a scalebar
        if self.galaxy.distance:
            length = np.degrees(1.e-3 / self.galaxy.distance)  # length of the scalebar in degrees, corresponding to 1 kpc
            fig.add_scalebar(length=length, label='1 kpc', frame=False)
            fig.scalebar.set_linewidth(5)

        #Make sure the axis labels don't fall off the figure
        plt.tight_layout()

        if self.tosave:
            if moment == 2:
                plt.savefig(self.savepath+'mom2.pdf', bbox_inches='tight',)
            else:
                plt.savefig(self.savepath+'mom1.pdf', bbox_inches='tight')

        return