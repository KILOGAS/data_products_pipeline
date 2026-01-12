from astropy.io import fits
import numpy as np
import scipy.ndimage as ndimage
from targets import galaxies
from clip_cube import ClipCube
from photutils import EllipticalAnnulus
from photutils import aperture_photometry
from astropy import wcs
from astropy.stats import mad_std
import os


class MomentMaps:

    def __init__(self, galname, path_pbcorr, path_uncorr, savepath=None, sun=True, tosave=False, sample=None, redo_clip=False):
        self.galaxy = galaxies(galname, sample)
        self.path_pbcorr = path_pbcorr
        self.path_uncorr = path_uncorr
        self.savepath = savepath or './'
        self.tosave = tosave
   
    
    def beam_area(cube, cellsize):
        return (np.pi * (cube.header['BMAJ'] / cellsize) * 
                (cube.header['BMIN'] / cellsize)) / (4 * np.log(2))


    def new_header(self, header):
        """
        Change the 3D header to the corresponding 2D one.

        Parameters
        ----------
        header : FITS header
            Corresponding to the data cube.

        Returns
        -------
        header : FITS header
            Corresponding to the 2D image created from the 3D data cube.

        """

        header = header.copy()

        try:
            header.pop('PC3_1')
            header.pop('PC3_2')
            header.pop('PC1_3')
            header.pop('PC2_3')
            header.pop('PC3_3')
        except:
            try:
                header.pop('PC03_01')
                header.pop('PC03_03')
                header.pop('PC03_02')
                header.pop('PC01_03')
                header.pop('PC02_03')
            except:
                pass

        header.pop('CTYPE3')
        header.pop('CRVAL3')
        header.pop('CDELT3')
        header.pop('CRPIX3')
        try:
            header.pop('CUNIT3')
        except:
         pass
        header.pop('NAXIS3')
        try:
            header.pop('OBSGEO-Z')
        except:
            pass

        header['NAXIS'] = 2
        try:
            if header['WCSAXES'] == 3:
                header['WCSAXES'] = 2
        except:
            pass
        try:
            header.pop('CROTA3')
        except:
            pass

        return header


    def create_vel_array(self, cube):
        """
        Creates the velocity array corresponding to the spectral axis
        of the cube in km/s.

        Parameters
        ----------
        cube : FITS file
            The spectral cube from which we will make the velocity array.

        Raises
        ------
        KeyError
            Raised if the units are something other than velocity or frequency.

        Returns
        -------
        vel_array : 1D numpy array
            Contains the velocities corresponding to the cube spectral axis.
            Only contains values from the velocity corresponding to the start of the 
        spectral line data onwards
        vel_narray : 3D numpy array
            Same as vel_array but in the shape of the data cube (i.e. tiled in
                                                    the spatial dimensions).
        vel_array_full : 1D numpy array
            Same as vel_array but for the entire spectral axis.

        """

        cube_orig, _ = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun, savepath=self.savepath,
                        tosave=self.tosave, sample=self.sample).readfits()

        v_ref = cube_orig.header['CRPIX3']  # Location of the reference channel
        if cube_orig.header['CTYPE3'] == 'VRAD' or cube_orig.header['CTYPE3'] == 'VELOCITY':
            v_val = cube_orig.header['CRVAL3'] / 1000  # Velocity in the reference channel, m/s to km/s
            v_step = cube_orig.header['CDELT3'] / 1000  # Velocity step in each channel, m/s to km/s
        elif cube_orig.header['CTYPE3'] == 'FREQ':
            if self.sample == 'viva' or self.sample == 'things':
                v_val = 299792.458 * (1 - (cube_orig.header['CRVAL3'] / 1e9) / 1.420405752)
                v_shift = 299792.458 * (1 - ((cube_orig.header['CRVAL3'] + cube_orig.header['CDELT3']) / 1e9) / 1.420405752)
                v_step = - (v_val - v_shift)
            else:
                v_val = 299792.458 * (1 - (cube_orig.header['CRVAL3'] / 1e9) / 230.538000)
                v_shift = 299792.458 * (1 - ((cube_orig.header['CRVAL3'] + cube_orig.header['CDELT3']) / 1e9) / 230.538000)
                v_step = - (v_val - v_shift)
        else:
            raise KeyError('Pipeline cannot deal with these units yet.')

        # Construct the velocity arrays (keep in mind that fits-files are 1 indexed)
        vel_array = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1 + self.galaxy.start) * v_step + v_val
        vel_narray = np.tile(vel_array, (len(cube.data[0, 0, :]), len(cube.data[0, :, 0]), 1)).transpose()
        vel_array_full = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1) * v_step + v_val

        return vel_array, vel_narray, vel_array_full


    def add_clipping_keywords(self, header):
        """
        Add information to the header specifying details about the clipping.

        Parameters
        ----------
        header : FITS header
            Header of the cube that was clipped.

        Returns
        -------
        header : FITS header
            Header of the cube with clipping-related keywords added.

        """
        if self.sun:
            try:
                header.add_comment('Cube was clipped using the Sun+18 masking method', before='BUNIT')
            except:
                header.add_comment('Cube was clipped using the Sun+18 masking method', after='NAXIS2')
            header['CLIPL_L'] = self.galaxy.cliplevel_low
            header.comments['CLIPL_L'] = 'S/N threshold specified for the "wing mask"'
            header['CLIPL_H'] = self.galaxy.cliplevel_high
            header.comments['CLIPL_H'] = 'S/N threshold specified for the "core mask"'
            header['NCHAN_L'] = self.galaxy.nchan_low
            header.comments['NCHAN_L'] = '# of consecutive channels specified for the "core mask"'
            header['NCHAN_H'] = self.galaxy.nchan_high
            header.comments['NCHAN_H'] = '# of consecutive channels specified for the "wing mask"'
        else:
            try:
                header.add_comment('Cube was clipped using the Dame11 masking method', before='BUNIT')
            except:
                header.add_comment('Cube was clipped using the Dame11 masking method', after='NAXIS2')
            header['CLIPL'] = self.galaxy.cliplevel
            header.comments['CLIPL'] = 'SNR used for clip (Dame11)'

        header['CLIP_RMS'] = self.uncertainty_maps(calc_rms=True)
        header.comments['CLIP_RMS'] = 'rms noise level used in masking (K km/s)'

        return header


    def calc_moms(self, units='M_Sun/pc^2', alpha_co=5.4):
        """
        Clip the spectral cube according to the desired method (either the 
        method Pythonified by Jiayi Sun or the more basic smooth + clip 
        strategy from Dame+11), and create moment 0, 1, and 2 maps. Saves maps 
        as fits files if so desired. Also calculate the systemic velocity from 
        the moment 1 map.

        Parameters
        ----------
        units : str, optional
            Preferred units (either 'K km/s' or 'M_Sun/pc^2'). The default is 
                             'M_Sun/pc^2'.
        alpha_co : float, optional
            In case units == 'M_Sun/pc^2', multiply the moment 0 map by this 
            factor to obtain these units. The default is 5.4, which is the 
            value for CO(2-1) quoted in https://arxiv.org/pdf/1805.00937.pdf.

        Raises
        ------
        AttributeError
            Raised if units is set to anything other than 'K km/s' or 
            'M_Sun/pc^2'.

        Returns
        -------
        cube : FITS file
            3D spectral line cube for which the moment maps will be calculated.
        mom0_hdu : FITS file
            Contains the moment 0 map + corresponding header.
        mom1_hdu : FITS file
            Contains the moment 1 map + corresponding header.
        mom2_hdu : FITS file
            Contains the moment 2 map + corresponding header.
        sysvel : float
            Systemic velocity of the gas in the system in km/s.

        """

        if self.redo_clip:
            cube, _ = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun, savepath=self.savepath,
                            tosave=self.tosave, sample=self.sample).do_clip()
        elif os.path.exists(self.savepath + 'subcube_slab.fits'):
            cube = fits.open(self.savepath + 'subcube_slab.fits')[0]
        else:
            cube, _ = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                               savepath=self.savepath, tosave=self.tosave, sample=self.sample).do_clip()

        vel_array, vel_narray, vel_fullarray = self.create_vel_array(cube)

        if cube.header['CTYPE3'] == 'VRAD' or cube.header['CTYPE3'] == 'VELOCITY':
            mom0 = np.sum((cube.data * abs(cube.header['CDELT3']) / 1000), axis=0)
        elif cube.header['CTYPE3'] == 'FREQ':
            if self.sample == 'viva' or self.sample == 'things':
                v_val = 299792.458 * (1 - (cube.header['CRVAL3'] / 1e9) / 1.420405752)
                v_shift = 299792.458 * (1 - ((cube.header['CRVAL3'] + cube.header['CDELT3']) / 1e9) / 1.420405752)
                v_step = - (v_val - v_shift)
            else:
                v_val = 299792.458 * (1 - (cube.header['CRVAL3'] / 1e9) / 230.538000)
                v_shift = 299792.458 * (1 - ((cube.header['CRVAL3'] + cube.header['CDELT3']) / 1e9) / 230.538000)
                v_step = - (v_val - v_shift)
            mom0 = np.sum(cube.data * abs(v_step), axis=0)
        else:
            raise AttributeError("Can't deal with these units yet.")

        if units == 'M_Sun/pc^2':
            #mom0 = mom0 / cube.header['JTOK'] * 91.7 * alpha_co * (cube.header['BMAJ'] * 3600 * cube.header[
            #    'BMIN'] * 3600) ** (-1) / 4

            if self.sample == 'viva' or self.sample == 'things':
                coldens_atom_cm = mom0 * 1.10e24 / (cube.header['BMAJ'] * 3600 * cube.header['BMIN'] * 3600)
                Msol_to_matom = 1.187883838e57
                pc_to_cm = 9.521e36
                coldens_Msol_pc = coldens_atom_cm / Msol_to_matom * pc_to_cm
                mom0 = coldens_Msol_pc
            elif self.sample == None:
                xco = 1.23361968e+20
                mom0 *= 91.9 * xco / (cube.header['BMAJ'] * 3600 * cube.header['BMIN'] * 3600) * 1.6014457E-20
            else:
                mom0 *= alpha_co
        elif units == 'K km/s':
            #if self.sample == 'viva':
            #    mom0 = mom0 * 1.222e3 / (cube.header['BMAJ'] * 3600 * cube.header['BMIN'] * 3600) / (1.420405751) ** 2
            pass
        else:
            raise AttributeError('Please choose between "K km/s" and "M_Sun/pc^2"')
        mom1 = np.sum(cube.data * vel_narray, axis=0) / np.sum(cube.data, axis=0)
        mom2 = np.sqrt(np.sum(abs(cube.data) * (vel_narray - mom1) ** 2, axis=0) / np.sum(abs(cube.data), axis=0))

        # Calculate the systemic velocity from the spatial inner part of the cube (to avoid PB effects)
        inner_cube = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, savepath=self.savepath,
                              tosave=self.tosave, sample=self.sample).innersquare(mom1)

        from matplotlib import pyplot as plt
        plt.imshow(inner_cube)

        sysvel = np.nanmean(inner_cube)# + self.galaxy.sysvel_offset
        mom1 -= sysvel

        mom0_hdu = fits.PrimaryHDU(mom0, self.new_header(cube.header))
        mom1_hdu = fits.PrimaryHDU(mom1, self.new_header(cube.header))
        mom2_hdu = fits.PrimaryHDU(mom2, self.new_header(cube.header))

        #self.pixel_size_check(header=mom0_hdu.header)

        # Change or add any (additional) keywords to the headers
        if units == 'M_Sun/pc^2':
            mom0_hdu.header['BTYPE'] = 'Column density'
            if not self.sample == 'viva' or self.sample == 'things':
                mom0_hdu.header.comments['BTYPE'] = 'Total molecular gas (H_2 + He)'
        else:
            mom0_hdu.header['BTYPE'] = 'Integrated intensity'

        mom1_hdu.header['BTYPE'] = 'Velocity'
        mom2_hdu.header['BTYPE'] = 'Linewidth'
        mom0_hdu.header['BUNIT'] = units; mom0_hdu.header.comments['BUNIT'] = ''
        mom1_hdu.header['BUNIT'] = 'km/s'; mom1_hdu.header.comments['BUNIT'] = ''
        mom2_hdu.header['BUNIT'] = 'km/s'; mom2_hdu.header.comments['BUNIT'] = ''
        if not self.sample == 'viva' or self.sample == 'things':
            mom0_hdu.header['ALPHA_CO'] = alpha_co; #mom0_hdu.header.comments['ALPHA_CO'] = 'Assuming a line ratio of 0.8'
        mom1_hdu.header['SYSVEL'] = sysvel; mom1_hdu.header.comments['SYSVEL'] = 'km/s'

        self.add_clipping_keywords(mom0_hdu.header)
        self.add_clipping_keywords(mom1_hdu.header)
        self.add_clipping_keywords(mom2_hdu.header)

        if self.tosave:
            if units == 'M_Sun/pc^2':
                mom0_hdu.writeto(self.savepath + 'mom0_Msolpc-2.fits', overwrite=True)
            if units == 'K km/s':
                if self.sample == 'viva' or self.sample == 'things':
                    mom0_hdu.writeto(self.savepath + 'mom0_Jyb-1kms-1.fits', overwrite=True)
                else:
                    mom0_hdu.writeto(self.savepath + 'mom0_Kkms-1.fits', overwrite=True)
            mom1_hdu.writeto(self.savepath + 'mom1.fits', overwrite=True)
            mom2_hdu.writeto(self.savepath + 'mom2.fits', overwrite=True)

        return cube, mom0_hdu, mom1_hdu, mom2_hdu, sysvel