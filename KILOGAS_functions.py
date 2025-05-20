# -*- coding: utf-8 -*-
"""
@author: Blake Ledger, created Feb 7 2025

This is a compilation of functions from Nikki Zabel's KILOGAS
GitHub page and what was used for VERTICO. I kept what I felt
was needed for the KILOGAS smooth and clip, and changed the
code and functions to remove steps which were not necessary.
"""

from astropy.io import fits
import numpy as np
from scipy.ndimage import binary_dilation, label

class KILOGAS_clip:

    def __init__(self, galname, directory, path_pbcorr, path_uncorr, start, stop, sun_method_params):
        self.galaxy = galname
        self.directory = directory
        self.path_pbcorr = path_pbcorr
        self.path_uncorr = path_uncorr
        self.start = start
        self.stop = stop
        self.nchan_low = sun_method_params[0]
        self.cliplevel_low = sun_method_params[1]
        self.nchan_high = sun_method_params[2]
        self.cliplevel_high = sun_method_params[3]
        self.prune_by_npix = sun_method_params[4]
        self.prune_by_fracbeam = sun_method_params[5]
        self.expand_by_npix = sun_method_params[6]
        self.expand_by_fracbeam = sun_method_params[7]
        self.expand_by_nchan = sun_method_params[8]

    def readfits(self):
        """
        Read in the fits files containing the primary beam corrected and uncorrected specral cubes.
        
        Parameters
        ----------
        None

        Returns
        -------
        cube_pbcorr : FITS file
            FITS file containing the spectral cube with primary beam correction applied and the header.
        cube_uncorr : FITS file
            FITS file containing the spectral cube without primary beam correction applied and the header.
        """
        
        print("READ .FITS")
        cube_pbcorr = fits.open(self.path_pbcorr)[0]
        cube_uncorr = fits.open(self.path_uncorr)[0]

        # If the first and last channels consist of nans only, remove them
        spectrum_pbcorr = np.nansum(cube_pbcorr.data, axis=(1, 2))
        cube_pbcorr.data = cube_pbcorr.data[spectrum_pbcorr != 0, :, :]
        
        spectrum_uncorr = np.nansum(cube_uncorr.data, axis=(1, 2))
        cube_uncorr.data = cube_uncorr.data[spectrum_uncorr != 0, :, :]

        # Count the number of empty channels at the start of the cube, so the header can be corrected
        firstzero = np.nonzero(spectrum_pbcorr)[0][0]
        cube_pbcorr.header['CRVAL3'] += cube_pbcorr.header['CDELT3'] * firstzero
        cube_uncorr.header['CRVAL3'] += cube_uncorr.header['CDELT3'] * firstzero

        # Get rid of nans
        cube_pbcorr.data[~np.isfinite(cube_pbcorr.data)] = 0
        cube_uncorr.data[~np.isfinite(cube_uncorr.data)] = 0

        return cube_pbcorr, cube_uncorr


    def split_cube(self, cube):
        """
        Split a cube into a cube containing the emission line and a cube 
        containing the line-free channels.

        Parameters
        ----------
        cube : FITS file
            FITS file containing the ALMA cube.

        Returns
        -------
        emiscube_hdu : FITS file
            FITS file containing the emission line cube.
        noisecube_hdu : FITS file
            FITS file containing a cube of the line-free channels.

        """
        
        ## This is where we need the channel inputs from Tim. I can use my example
        ## start and stop parameters with CO line channels from CARTA for KGAS15.
        
        print("SPLIT CUBE")
        emiscube = cube.data[self.start:self.stop, :, :]
        noisecube = np.concatenate((cube.data[:self.start, :, :], cube.data[self.stop:, :, :]), axis=0)

        emiscube_hdu = fits.PrimaryHDU(emiscube, cube.header)
        emiscube_hdu.header['NAXIS3'] = emiscube.shape[0]

        noisecube_hdu = fits.PrimaryHDU(noisecube, cube.header)
        noisecube_hdu.header['NAXIS3'] = noisecube.shape[0]

        return emiscube_hdu, noisecube_hdu
    

    def do_clip(self):
        """
        Perform the clipping of the data cube, using the method adopted
        and optimised by Jiayi Sun.

        Parameters
        ----------
        None

        Returns
        -------
        FITS file
            The clipped and trimmed version of the input data cube.
        FITS file
            The corresponding noise cube, of the same dimensions as the output
            data cube.

        """
        
        print("DO CLIP")
        cube_pbcorr, cube_uncorr = self.readfits()    
        cube_uncorr_copy = cube_uncorr.copy()
        print("Number of channels in initial cube:", len(cube_pbcorr.data))


        #The following steps are not necessary if we already have the line identified and the channels
        #where the line starts and stops.
        
        # Get a rough estimate of the noise in order to do the clipping
        #noisecube_temp = np.concatenate((cube_uncorr_copy.data[:10, :, :], cube_uncorr_copy.data[-10:, :, :]),
        #                                axis=0)
        #noisecube_temp_hdu = fits.PrimaryHDU(noisecube_temp, cube_uncorr_copy.header)

        # Create an initial mask and identify first and last channel containing emission
        #mask_full = self.sun_method(cube_uncorr_copy, noisecube_temp_hdu)
        #mask_idx = np.where(mask_full == 1)[0]
        #start = mask_idx[0]
        #stop = mask_idx[-1]
        
        # Create an updated noise cube
        #noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:start, :, :],
        #                                   cube_uncorr_copy.data[stop:, :, :]), axis=0)
        #noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)

        # Make a more accurate mask based on the new noise cube
        #mask_full = self.sun_method(cube_uncorr_copy, noisecube_uncorr_hdu)
        #mask_hdu = fits.PrimaryHDU(mask_full.astype(int), cube_pbcorr.header)
        #mask_idx = np.where(mask_full == 1)[0]
        
        #start = mask_idx[0]
        #stop = mask_idx[-1]
        
        print("The first channel before the CO line:", self.start)
        print("The last channel after the CO line:", self.stop)

        # Spit the cube in an emission and noise part
        emiscube_pbcorr = cube_pbcorr.data[self.start:self.stop, :, :]
        emiscube_uncorr = cube_uncorr.data[self.start:self.stop, :, :]

        noisecube_pbcorr = np.concatenate((cube_pbcorr.data[:self.start, :, :],
                                           cube_pbcorr.data[self.stop:, :, :]), axis=0)
        noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:self.start, :, :],
                                           cube_uncorr_copy.data[self.stop:, :, :]), axis=0)

        emiscube_uncorr_hdu = fits.PrimaryHDU(emiscube_uncorr, cube_uncorr_copy.header)
        noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)
        noisecube_pbcorr_hdu = fits.PrimaryHDU(noisecube_pbcorr, cube_pbcorr.header)

        mask = self.sun_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu)
        mask_hdu = fits.PrimaryHDU(mask.astype(int), cube_pbcorr.header)
        mask_hdu.writeto(self.directory + '/' + self.galaxy + '/' + 'clip_mask.fits', 
                         overwrite=True)

        print("MASK MADE")
        
        #At the moment I am not saving the mask.
        if False:#:save_mask:
            mask_hdu.header.add_comment('Cube was clipped using the Sun+18 masking method', before='BUNIT')
            try:
                mask_hdu.header.pop('BTYPE')
                mask_hdu.header.pop('BUNIT')
            except:
                pass
            try:
                mask_hdu.header.pop('DATAMAX')
                mask_hdu.header.pop('DATAMIN')
                mask_hdu.header.pop('JTOK')
                mask_hdu.header.pop('RESTFRQ')
            except:
                pass
            mask_hdu.header['CLIP_RMS'] = self.sun_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu, calc_rms=True)
            mask_hdu.header.comments['CLIP_RMS'] = 'rms [K km/s] for clipping'

        emiscube_pbcorr[mask == 0] = 0
        clipped_hdu = fits.PrimaryHDU(emiscube_pbcorr, cube_pbcorr.header)

        # Adjust the header to match the velocity range used
        clipped_hdu.header['CRVAL3'] += self.start * clipped_hdu.header['CDELT3']
        
        self.add_clipping_keywords(emiscube_uncorr_hdu, noisecube_uncorr_hdu, clipped_hdu.header)
        print("CLIP APPLIED")

        return clipped_hdu, noisecube_pbcorr_hdu


    def sun_method(self, emiscube, noisecube, calc_rms=False):
        """
        Apply Jiayi Sun's clipping method, including options to prune 
        detections with small areas on the sky, or expand the mask in the mask
        along the spatial axes or spectral axis.

        Parameters
        ----------
        emiscube : FITS file
            3D cube containing the spectral line.
        noisecube : FITS file
            3D cube containing the line-free channels.
        calc_rms : bool, optional
            If set to "True" this function will only return the RMS estimate 
            for the data cube. The default is False.

        Raises
        ------
        AttributeError
            Will raise an AttributeError if some of the required parameters
            are not set (nchan_low, cliplevel_low, nchan_high, and 
                         cliplevel_high).

        Returns
        -------
        mask : FITS file
            The final mask, which will be used for clipping the original cube.

        """
        
        # Check if the necessary parameters are provided
        if not (
                self.nchan_low and self.cliplevel_low and self.nchan_high and
                self.cliplevel_high):
            raise AttributeError('If you want to use Sun\'s method, please provide "nchan_low", "cliplevel_low", '
                                 '"nchan_high", and "cliplevel_high" as sun_method_params.')

        # Estimate the rms from the spatial inner part of the cube
        inner = self.innersquare(noisecube.data)
        rms = np.nanstd(inner)
        
        if calc_rms:
            return rms

        snr = emiscube.data / rms

        # Generate core mask
        mask_core = (snr > self.cliplevel_high).astype(bool)
        for i in range(self.nchan_high - 1):
            mask_core &= np.roll(mask_core, shift=1, axis=0)
        mask_core[:self.nchan_high - 1] = False
        for i in range(self.nchan_high - 1):
            mask_core |= np.roll(mask_core, shift=-1, axis=0)

        # Generate wing mask
        mask_wing = (snr > self.cliplevel_low).astype(bool)
        for i in range(self.nchan_low - 1):
            mask_wing &= np.roll(mask_wing, shift=1, axis=0)
        mask_wing[:self.nchan_low - 1] = False
        for i in range(self.nchan_low - 1):
            mask_wing |= np.roll(mask_wing, shift=-1, axis=0)

        # Dilate core mask inside wing mask
        mask = binary_dilation(mask_core, iterations=0, mask=mask_wing)

        # Prune detections with small projected areas on the sky
        if self.prune_by_fracbeam or self.prune_by_npix:
            mask = self.prune_small_detections(emiscube, mask)

        # Expand along spatial dimensions by a fraction of the beam FWHM
        if self.expand_by_fracbeam or self.expand_by_npix:
            mask = self.expand_along_spatial(emiscube, mask)

        # Expand along spectral dimension by a number of channels
        if self.expand_by_nchan:
            mask = self.expand_along_spectral(mask)

        return mask
    
    def prune_small_detections(self, cube, mask):
        """
        Mask structures in the spectral cube that are smaller than the desired 
        size specified by "prune_by_npix" or "prune_by_fracbeam" in the galaxy 
        parameters. Based on the function designed by Jiayi Sun.

        Parameters
        ----------
        cube : FITS file
            The ALMA cube, used to extract the relevant beam information from 
            the header.
        mask : 3D numpy array
            The mask that we previously created from the smoothed data cube.

        Returns
        -------
        mask : 3D numpy array
            Updated mask with the small detections set to 0.

        """

        print("PRUNING MASK")
        if self.prune_by_npix:
            prune_by_npix = self.prune_by_npix
        else:
            res = cube.header['CDELT2']  # deg. / pix.
            bmaj_pix = cube.header['BMAJ'] / res  # deg. / (deg. / pix.)
            bmin_pix = cube.header['BMIN'] / res  # deg. / (deg. / pix.)
            beam_area_pix = np.pi * bmaj_pix * bmin_pix
            prune_by_npix = beam_area_pix * self.prune_by_fracbeam

        labels, count = label(mask)
        for idx in np.arange(count) + 1:
            if (labels == idx).any(axis=0).sum() < prune_by_npix:
                mask[labels == idx] = False

        return mask
    
    def expand_along_spatial(self, cube, mask):
        """
        Expand the mask along spatial dimensions by an amount provided by 
        either "expand_by_npix" or "expand_by_fracbeam" in the galaxy 
        parameters.

        Parameters
        ----------
        cube : FITS file
            The ALMA cube, used to extract the relevant beam information from 
            the header.
        mask : 3D numpy array
            The mask that we previously created from the smoothed data cube.

        Returns
        -------
        mask : 3D numpy array
            Updated, expanded mask with the additional pixels set to 1.

        """

        print("SPATIAL MASK EXPANSION")

        if self.expand_by_npix:
            expand_by_npix = int(self.expand_by_npix)
        else:
            res = cube.header['CDELT2']  # deg. / pix.
            bmaj = cube.header['BMAJ']  # deg.
            bmin = cube.header['BMIN']  # deg.
            beam_hwhm_pix = np.average([bmaj, bmin]) / res / 2  # deg. / (deg. / pix.)
            expand_by_npix = int(beam_hwhm_pix * self.expand_by_fracbeam)

        structure = np.zeros([3, expand_by_npix * 2 + 1, expand_by_npix * 2 + 1])
        Y, X = np.ogrid[:expand_by_npix * 2 + 1, :expand_by_npix * 2 + 1]
        R = np.sqrt((X - expand_by_npix) ** 2 + (Y - expand_by_npix) ** 2)
        structure[1, :] = R <= expand_by_npix
        mask = binary_dilation(mask, iterations=1, structure=structure)

        return mask
    
    def expand_along_spectral(self, mask):
        """
        Expand the mask along the velocity direction as provided by 
        "expand_by_nchan" in the galaxy parameters.

        Parameters
        mask : 3D numpy array
            The mask that we previously created from the smoothed data cube.

        Returns
        -------
        mask : 3D numpy array
            Updated, expanded mask with the additional pixels set to 1.

        """
        
        print("SPECTRAL MASK EXPANSION")

        for i in range(self.expand_by_nchan):
            tempmask = np.roll(mask, shift=1, axis=0)
            tempmask[0, :] = False
            mask |= tempmask
            tempmask = np.roll(mask, shift=-1, axis=0)
            tempmask[-1, :] = False
            mask |= tempmask

        return mask

    def add_clipping_keywords(self, emiscube, noisecube, header):
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
        try:
            header.add_comment('Cube was clipped using the Sun+18 masking method', before='BUNIT')
        except:
            header.add_comment('Cube was clipped using the Sun+18 masking method', after='NAXIS2')
        header['CLIPL_L'] = self.cliplevel_low
        header.comments['CLIPL_L'] = 'S/N threshold specified for "wing mask"'
        header['CLIPL_H'] = self.cliplevel_high
        header.comments['CLIPL_H'] = 'S/N threshold specified for "core mask"'
        header['NCHAN_L'] = self.nchan_low
        header.comments['NCHAN_L'] = '# of consecutive channels for "core mask"'
        header['NCHAN_H'] = self.nchan_high
        header.comments['NCHAN_H'] = '# of consecutive channels for "wing mask"'
   

        header['CLIP_RMS'] = self.sun_method(emiscube, noisecube, calc_rms=True)
        header.comments['CLIP_RMS'] = 'rms [K km/s] for clipping'

        return header

    def innersquare(self, cube):
        """
        Get the central square (in spatial directions) of the spectral cube (useful for calculating the rms in a PB
        corrected spectral cube). Can be used for 2 and 3 dimensions, in the latter case the velocity axis is left
        unchanged.
        
        Parameters
        ----------
        cube : 2D or 3D array
            3D array input cube or 2D image
            
        Returns
        -------
        cube : 2D or 3D array
            2D or 3D array of the inner 1/8 of the cube in the spatial directions
        """

        if len(cube.shape) == 3:
            start_x = int(cube.shape[1] / 2 - cube.shape[1] / 8)
            stop_x = int(cube.shape[1] / 2 + cube.shape[1] / 8)
            start_y = int(cube.shape[2] / 2 - cube.shape[1] / 8)
            stop_y = int(cube.shape[2] / 2 + cube.shape[1] / 8)
            inner_square = cube[:, start_x:stop_x, start_y:stop_y]
            if (inner_square == inner_square).any():
                return inner_square
            else:
                return cube

        elif len(cube.shape) == 2:
            start_x = int(cube.shape[0] / 2 - 20)
            stop_x = int(cube.shape[0] / 2 + 20)
            start_y = int(cube.shape[1] / 2 - 20)
            stop_y = int(cube.shape[1] / 2 + 20)
            inner_square = cube[start_x:stop_x, start_y:stop_y]
            if (inner_square == inner_square).any():
                return inner_square
            else:
                return cube
            
        else:
            raise AttributeError('Please provide a 2D or 3D array.')