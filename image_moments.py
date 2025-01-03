import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot as plt
import aplpy as apl
from astropy import wcs
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import numpy as np
import config_figs
from sauron_colormap import register_sauron_colormap; register_sauron_colormap()
from targets import galaxies
from create_moments import MomentMaps
from clip_cube import ClipCube
from matplotlib.colors import ListedColormap


class CreateImages:

    def __init__(self, galname, path_pbcorr, path_uncorr, savepath=None, refresh=False, overwrite=True,
                 sun=True, tosave=False, sample=None, redo_clip=False):
        self.galaxy = galaxies(galname, sample)
        self.path_pbcorr = path_pbcorr
        self.path_uncorr = path_uncorr
        self.refresh = refresh
        self.overwrite = overwrite
        self.savepath = savepath or './'
        self.sun = sun
        self.tosave = tosave
        self.sample = sample
        self.redo_clip = redo_clip

    @staticmethod
    def custom_cmap(cmap=plt.cm.afmhot_r):
        """
        Cut off the dark colours in afmhot_r (or something else) to make sure the black lines in the PVD are visible on it.
        :return:
        """
        cmap = cmap
        # my_cmap = cmap(np.arange(cmap.N))

        # Cut off the colors at purple so they don't go to black
        my_cmap = cmap(np.arange(0, cmap.N - 50, 1))

        my_cmap = ListedColormap(my_cmap)

        return my_cmap

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

    def PVD(self, axis='major', find_angle=False, check_slit=False):
        """
        Plot a position-velocity diagram of the galaxy
        :param mom0: moment 0 image of the galaxy
        :param mom1: moment 1 image of the galaxy
        :param angle: angle by which the images should be rotated to obtain a horizontal projection of the galaxy
        :param centre: the vertical offset around which the emission is centred in the rotated image
        """

        #matplotlib.rcParams['axes.linewidth'] = 0.5

        if self.refresh:
            if self.overwrite:
                PV = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                 savepath=self.savepath, tosave=True, sample=self.sample, redo_clip=self.redo_clip).\
                    PVD(axis=axis, find_angle=find_angle, check_slit=check_slit)
            else:
                PV = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun, tosave=False,
                                sample=self.sample, redo_clip=self.redo_clip).\
                    PVD(axis=axis, find_angle=find_angle, check_slit=check_slit)
        else:
            PV = fits.open(self.path + 'pvd.fits')[0]

        clipped_cube, _, _, _, sysvel = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                               sun=self.sun, tosave=False, savepath=self.savepath, sample=self.sample,
                                                   redo_clip=self.redo_clip).calc_moms()


        # Create a physical position axis using the middle found and the spatial resolution, in arc seconds
        res = clipped_cube.header['CDELT2']
        vres = clipped_cube.header['CDELT3'] / 1000  # velocity resolution
        position = np.arange(0, PV.shape[1], 1)
        offset = (position - len(position) / 2) * res * 3600

        # Plot the PVD
        fig, ax = plt.subplots(figsize=(15, 8))

        # Add mock axis to get a nice colourbar
        #ax1 = fig.add_axes([0.5, 0.5, 0.00000000000001, 0.0000000000001])
        #mock = ax1.contourf(PV.data, cmap=self.custom_cmap(), levels=20)
        #ax1.axis('off')

       # ax = fig.add_axes([0, 0, 1, 1])
        ax_kpc = ax.twiny()  # add second x-axis at the top of the figure (position in kpc)
        ax_rel = ax.twinx()  # add second y-axis to the right of the figure (relative velocity)

        # Create the extra y axis in absolute velocities
        def absolute_yax(ax):
            """
            Convert velocities to relative velocities for the second x axis
            """

            def absolute_vel(v):
                """
                Convert the relative velocities to absolute velocities
                """
                return v + sysvel

            y1, y2 = ax.get_ylim()
            ax_rel.set_ylim(absolute_vel(y1), absolute_vel(y2))
            ax_rel.figure.canvas.draw()

        # Create the extra x axis in kpc
        def x_kpc(ax):
            """
            Convert velocities to relative velocities for the second x axis.
            arcseconds to degrees to radians, and Mpc to kpc
            """

            def deg_to_kpc(arcsecs):
                return self.galaxy.distance * arcsecs / 3600. * (np.pi / 180.) * 1000.

            x1, x2 = ax.get_xlim()
            ax_kpc.set_xlim(deg_to_kpc(x1), deg_to_kpc(x2))
            ax_kpc.figure.canvas.draw()

        # Automatically update ylim of the relative velocity axis when ylim of ax changes.
        ax.callbacks.connect("ylim_changed", absolute_yax)
        ax.callbacks.connect("xlim_changed", x_kpc)

        velocity, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                    sun=self.sun, savepath=self.savepath, tosave=False, sample=self.sample,
                                    redo_clip=self.redo_clip).create_vel_array(clipped_cube)

        velocity = np.flip(velocity) - sysvel

        levels = list(np.linspace(np.amax(PV.data) * 0.2, np.amax(PV.data), 10))

        # Contours in black
        C2 = ax.contour(PV.data,
                   extent=[np.amax(offset), np.amin(offset), np.amax(velocity), np.amin(velocity)],
                   colors='k', levels=levels, linewidths=0.05)

        # Filling with coloured contours
        C1 = ax.contourf(PV.data,
                    extent=[np.amax(offset), np.amin(offset), np.amax(velocity), np.amin(velocity)],
                    cmap=self.custom_cmap(), levels=levels)

        # Add a colourbar
        cbar = fig.colorbar(C1, pad=0.12, format='%.2f')
        cbar.add_lines(C2)
        if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
            cbar.set_label('Brightness temperature [Jy b${-1}$]')
        else:
            cbar.set_label('Brightness temperature [K]')

        # Make the plot pretty
        ax.set_xlim(np.amin(offset) - 0.2 * np.amax(offset), np.amax(offset) + 0.2 * np.amax(offset))
        if self.galaxy.vrange:
            ax.set_ylim(-self.galaxy.vrange * 1.5, self.galaxy.vrange * 1.5)
        else:
            ax.set_ylim(-np.amax(velocity) * 1.1, np.amax(velocity) * 1.1)
        ax.set_xlabel('Offset [arcsec]')
        ax_kpc.set_xlabel('Offset [kpc]')
        ax.set_ylabel(r'Relative velocity [km s$^{-1}$]')
        ax_rel.set_ylabel(r'Velocity [km s$^{-1}$]', labelpad=0)
        x1, x2 = ax.get_xlim()
        y1, y2 = ax.get_ylim()
        #plt.gca().tick_params(pad=8)
        ax.errorbar(0.8 * x2, 0.7 * y2, xerr=clipped_cube.header['BMAJ'] * 3600 / 2., yerr=vres / 2., ecolor='k', capsize=2.5)
        ax.annotate('PA = ' + str(self.galaxy.angle - 180) + '$^o$', xy=(-0.8 * x2, -0.7 * y2), fontsize=25)

        plt.tight_layout()

        if self.tosave:
            if axis == 'major':
                plt.savefig(self.savepath + 'pvd_major.pdf', bbox_inches='tight')
            if axis == 'minor':
                plt.savefig(self.savepath + 'pvd_minor.pdf', bbox_inches='tight')

    def spectrum(self, x_axis='velocity', useclipped=False):

        if self.refresh:
            if self.overwrite:
                spectrum, velocity, v_off, frequency = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                                    sun=self.sun, savepath=self.savepath, tosave=True,
                                                                  sample=self.sample, redo_clip=self.redo_clip).spectrum(useclipped=useclipped)

            else:
                spectrum, velocity, v_off, frequency = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                                    sun=self.sun, savepath=self.savepath, tosave=False,
                                                                  sample=self.sample, redo_clip=self.redo_clip).spectrum(useclipped=useclipped)

        else:
            temp = np.loadtxt(self.savepath + 'spectrum.csv', delimiter=',')
            spectrum = temp[:, 0]
            velocity = temp[:, 1]
            v_off = temp[:, 2]
            frequency = temp[:, 3]

        clipped_cube, _, _, _, sysvel = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr,
                                               sun=self.sun, tosave=False, savepath=self.savepath, sample=self.sample,
                                                   redo_clip=self.redo_clip).calc_moms()

        fig, ax = plt.subplots(figsize=(7, 7))

        extra_chans = 5

        if x_axis == 'velocity':
            ax.plot(velocity, spectrum, color='k', drawstyle='steps')
            x = np.arange(np.amin(velocity) - 100, np.amax(velocity) + 100, 1)
            ax.set_xlim(velocity[len(velocity) - 1] + extra_chans, velocity[0] - extra_chans)
            ax.set_xlabel(r'Velocity [km s$^{-1}$]')

        elif x_axis == 'vel_offset':
            ax.plot(v_off, spectrum, color='k', drawstyle='steps')
            x = np.arange(np.amin(v_off) - 100, np.amax(v_off) + 100, 1)
            ax.set_xlim(v_off[len(v_off) - 1] + extra_chans, v_off[0] - extra_chans)
            ax.set_xlabel(r'Velocity offset [km s$^{-1}$]')

        elif x_axis == 'frequency':
            ax.plot(frequency, spectrum, color='k', drawstyle='steps')
            x = np.arange(np.amin(frequency) - extra_chans, np.amax(frequency) + extra_chans, 1)
            ax.set_xlim(frequency[len(frequency) - 1], frequency[0])
            ax.set_xlabel(r'Frequency [GHz]')
        else:
            raise AttributeError('Please choose between "velocity" , "vel_offset", and "frequency" for "x-axis"')

        # Line through zero
        zeroline = np.zeros(len(x))
        plt.plot(x, zeroline, linestyle=':', c='r', linewidth=1)

        if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
            ax.set_ylabel('Brightness temperature [Jy b${-1}$]')
        else:
            ax.set_ylabel('Brightness temperature [K]')

        plt.tight_layout()

        if self.tosave:
            if x_axis == 'frequency':
                plt.savefig(self.savepath + 'spectrum_freq.pdf', bbox_inches='tight')
            if x_axis == 'velocity':
                plt.savefig(self.savepath + 'spectrum_vel.pdf', bbox_inches='tight')
            if x_axis == 'vel_offset':
                plt.savefig(self.savepath + 'spectrum_vel_offset.pdf', bbox_inches='tight')

    def radial_profile(self, y_units='K kms', x_units='kpc', alpha_co=5.4, table_path=None, check_aperture=False):

        if not ((y_units == 'K km/s') or (y_units == 'M_Sun pc^-2')):
            raise AttributeError('Please choose between "K kms" and "M_Sun pc^-2" for the keyword "y_units".')
        if not ((x_units == 'kpc') or (x_units == 'arcsec')):
            raise AttributeError('Please choose between "kpc" and "arcsec" for the keyword "x_units".')

        if self.refresh:
            if self.overwrite:
                rad_prof_K, rad_prof_K_err, rad_prof_Msun, rad_prof_Msun_err, radii_arcsec, radii_kpc = MomentMaps(self.galaxy.name, self.path_pbcorr,
                        self.path_uncorr, sun=self.sun, savepath=self.savepath, tosave=True, sample=self.sample, redo_clip=self.redo_clip).\
                    radial_profile(alpha_co=alpha_co, table_path=table_path, check_aperture=check_aperture)
            else:
                rad_prof_K, rad_prof_K_err, rad_prof_Msun, rad_prof_Msun_err, radii_arcsec, radii_kpc = MomentMaps(self.galaxy.name, self.path_pbcorr,
                        self.path_uncorr, sun=self.sun, savepath=self.savepath, tosave=False, sample=self.sample, redo_clip=self.redo_clip).\
                    radial_profile(alpha_co=alpha_co, table_path=table_path, check_aperture=check_aperture, hires=False)
        else:
            temp = np.loadtxt(self.savepath + 'rad_prof.csv', delimiter=',')
            rad_prof_K = temp[:, 0]
            rad_prof_K_err = temp[:, 1]
            rad_prof_Msun = temp[:, 2]
            rad_prof_Msun_err = temp[:, 3]
            radii_arcsec = temp[:, 4]
            radii_kpc = temp[:, 5]

        plt.figure(figsize=(10, 7))

        if x_units == 'kpc':
            plt.xlabel('Radius [kpc]')
            if y_units == 'K km/s':
                plt.errorbar(radii_kpc, np.log10(rad_prof_K), yerr=rad_prof_K_err/rad_prof_K * 0.434, c='k', linestyle='None',
                             marker='o', ms=10)
                if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                    plt.ylabel(r'log(Integrated intensity [Jy b$^{-1}$ km s$^{-1}$])')
                else:
                    plt.ylabel(r'log(Integrated intensity [K km s$^{-1}$])')
            else:
                plt.errorbar(radii_kpc, np.log10(rad_prof_Msun), yerr=rad_prof_Msun_err/rad_prof_Msun * 0.434, c='k', linestyle='None',
                             marker='o', ms=10)
                plt.ylabel(r'log(Surface density [$M_\odot$ pc$^{-2}$])')
        elif x_units == 'arcsec':
            plt.xlabel(r'Radius [$^{\prime\prime}$]')
            if y_units == 'K km/s':
                plt.errorbar(radii_arcsec, np.log10(rad_prof_K), yerr=rad_prof_K_err/rad_prof_K * 0.434, c='k', linestyle='None',
                             marker='o', ms=10)
                if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                    plt.ylabel(r'log(Integrated intensity [Jy b${-1}$ km s$^{-1}$])')
                else:
                    plt.ylabel(r'log(Integrated intensity [K km s$^{-1}$])')
            else:
                plt.errorbar(radii_arcsec, np.log10(rad_prof_Msun), yerr=rad_prof_Msun_err/rad_prof_Msun * 0.434, c='k', linestyle='None',
                             marker='o', ms=10)
                plt.ylabel(r'log(Surface density [$M_\odot$ pc$^{-2}$])')

        plt.xlim(-0.01)
        if y_units == 'K km/s':
            if not self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                plt.ylim(-1)
        else:
            plt.ylim(0)

        plt.tight_layout()

        if self.tosave:
            if x_units == 'kpc':
                if y_units == 'K km/s':
                    if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                        plt.savefig(self.savepath + 'rad_prof_kpc_Jyb-1kms-1.pdf', bbox_inches='tight')
                    else:
                        plt.savefig(self.savepath + 'rad_prof_kpc_Kkms-1.pdf', bbox_inches='tight')
                else:
                    plt.savefig(self.savepath + 'radi_prof_kpc_Msolpc-2.pdf', bbox_inches='tight')
            if x_units == 'arcsec':
                if y_units == 'K km/s':
                    if self.sample == 'viva' or self.sample == 'things' or self.sample == None:
                        plt.savefig(self.savepath + 'rad_prof_arcsec_Jyb-1kms-1.pdf', bbox_inches='tight')
                    else:
                        plt.savefig(self.savepath + 'rad_prof_arcsec_Kkms-1.pdf', bbox_inches='tight')
                else:
                    plt.savefig(self.savepath + 'rad_prof_arcsec_Msolpc-2.pdf', bbox_inches='tight')

    def mom0_noise_maps(self, path=''):

        if self.refresh:
            if self.overwrite:
                mom0_unc, mom0_SN, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                    tosave=True, savepath=self.savepath, sample=self.sample,
                                                     redo_clip=self.redo_clip).uncertainty_maps()
            else:
                mom0_unc, mom0_SN, _, _ = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                       tosave=False, sample=self.sample, redo_clip=self.redo_clip).uncertainty_maps()
        else:
            mom0_unc = fits.open(path + 'mom0_unc.fits')[0]
            mom0_SN = fits.open(path + 'mom0_SN.fits')[0]

        # Image the uncertainty map
        f = plt.figure(figsize=self.galaxy.figsize)
        fig = apl.FITSFigure(mom0_unc, figure=f)
        fig.set_theme('publication')
        fig.show_colorscale(cmap='magma_r')

        # Add a colourbar
        fig.add_colorbar()
        fig.colorbar.set_axis_label_text(r'Uncertainty (K km s$^{-1}$)')
        fig.colorbar.set_axis_label_font(size=30)
        fig.colorbar.set_axis_label_pad(15)

        # axes and ticks
        fig.ticks.set_color('black')
        fig.ticks.set_length(10)
        fig.ticks.set_linewidth(2)
        fig.tick_labels.set_xformat('hh:mm:ss')
        fig.tick_labels.set_yformat('dd:mm:ss')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig.ticks.set_minor_frequency(5)

        plt.tight_layout()

        if self.tosave:
            plt.savefig(self.savepath + 'mom0_unc.pdf', bbox_inches='tight')

        # Image the S/N map
        f = plt.figure(figsize=self.galaxy.figsize)
        fig = apl.FITSFigure(mom0_SN, figure=f)
        fig.set_theme('publication')
        fig.show_colorscale(cmap='magma_r')

        # Add a colourbar
        fig.add_colorbar()
        fig.colorbar.set_axis_label_text('S/N')
        fig.colorbar.set_axis_label_font(size=30)
        fig.colorbar.set_axis_label_pad(15)

        # axes and ticks
        fig.ticks.set_color('black')
        fig.ticks.set_length(10)
        fig.ticks.set_linewidth(2)
        fig.tick_labels.set_xformat('hh:mm:ss')
        fig.tick_labels.set_yformat('dd:mm:ss')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig.ticks.set_minor_frequency(5)

        plt.tight_layout()

        if self.tosave:
            plt.savefig(self.savepath + 'mom0_SN.pdf', bbox_inches='tight')

    def mom1_2_noise_maps(self, path=''):

        if self.refresh:
            if self.overwrite:
                mom0_unc, SN_hdu, mom1_unc, mom2_unc = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                               tosave=True, savepath=self.savepath, sample=self.sample, redo_clip=self.redo_clip).uncertainty_maps()
            else:
                mom0_unc, SN_hdu, mom1_unc, mom2_unc = MomentMaps(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                               tosave=False, sample=self.sample, redo_clip=self.redo_clip).uncertainty_maps()
        else:
            mom1_unc = fits.open(path + 'mom1_unc.fits')[0]
            mom2_unc = fits.open(path + 'mom2_unc.fits')[0]

        # Image the uncertainty maps
        mom1_unc.data = np.log10(mom1_unc.data)
        f = plt.figure(figsize=self.galaxy.figsize)
        fig = apl.FITSFigure(mom1_unc, figure=f)
        fig.set_theme('publication')
        fig.show_colorscale(cmap='sauron')

        # Add a colourbar
        fig.add_colorbar()
        fig.colorbar.set_axis_label_text(r'log uncertainty (km s$^{-1}$)')
        fig.colorbar.set_axis_label_font(size=30)
        fig.colorbar.set_axis_label_pad(15)

        # axes and ticks
        fig.ticks.set_color('black')
        fig.ticks.set_length(10)
        fig.ticks.set_linewidth(2)
        fig.tick_labels.set_xformat('hh:mm:ss')
        fig.tick_labels.set_yformat('dd:mm:ss')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig.ticks.set_minor_frequency(5)

        plt.tight_layout()

        if self.tosave:
            plt.savefig(self.savepath + 'mom1_unc.pdf', bbox_inches='tight')

        # Moment 2
        mom2_unc.data = np.log10(mom2_unc.data)
        f = plt.figure(figsize=self.galaxy.figsize)
        fig = apl.FITSFigure(mom2_unc, figure=f)
        fig.set_theme('publication')
        fig.show_colorscale(cmap='sauron')

        # Add a colourbar
        fig.add_colorbar()
        fig.colorbar.set_axis_label_text(r'log uncertainty (km s$^{-1}$)')
        fig.colorbar.set_axis_label_font(size=30)
        fig.colorbar.set_axis_label_pad(15)

        # axes and ticks
        fig.ticks.set_color('black')
        fig.ticks.set_length(10)
        fig.ticks.set_linewidth(2)
        fig.tick_labels.set_xformat('hh:mm:ss')
        fig.tick_labels.set_yformat('dd:mm:ss')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig.ticks.set_minor_frequency(5)

        plt.tight_layout()

        if self.tosave:
            plt.savefig(self.savepath + 'mom2_unc.pdf', bbox_inches='tight')
