import aplpy as apl
from astropy.io import fits
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, label
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.ned import Ned
from matplotlib import pyplot as plt


class ClipCube:

    def __init__(self, galaxy, path_pbcorr, path_uncorr, savepath=None, tosave=True):
        self.galaxy = galaxy
        self.path_pbcorr = path_pbcorr
        self.path_uncorr = path_uncorr
        self.savepath = savepath or './'
        self.tosave = tosave


    @staticmethod
    def remove_stokes(cube):
        """
        If the fits file containing the spectral cube has a Stokes axis, remove it and remove the corresponding
        header keywords.
        :param cube (HDU file): HDU file containing the spectral cube and its header
        :return (HDU file): the input HDU file with the Stokes axis and corresponding header keywords removed.
        """

        cube.data = np.squeeze(cube.data)

        try:
            cube.header.pop('PC01_04')
            cube.header.pop('PC02_04')
            cube.header.pop('PC03_04')
            cube.header.pop('PC04_04')
            cube.header.pop('PC04_01')
            cube.header.pop('PC04_02')
            cube.header.pop('PC04_03')
        except:
            pass
        try:
            cube.header.pop('PC1_4')
            cube.header.pop('PC2_4')
            cube.header.pop('PC3_4')
            cube.header.pop('PC4_4')
            cube.header.pop('PC4_1')
            cube.header.pop('PC4_2')
            cube.header.pop('PC4_3')
        except:
            pass
        cube.header.pop('CTYPE4')
        cube.header.pop('CRVAL4')
        cube.header.pop('CRPIX4')
        try:
            cube.header.pop('CUNIT4')
        except:
            pass
        cube.header.pop('CDELT4')
        try:
            cube.header.pop('CROTA4')
        except:
            pass

        return cube


    def innersquare(self, cube):
        """
        Get the central square (in spatial directions) of the spectral cube (useful for calculating the rms in a PB
        corrected spectral cube). Can be used for 2 and 3 dimensions, in the latter case the velocity axis is left
        unchanged.
        :param cube (2D or 3D array): input cube or 2D image
        :return: 2D or 3D array of the inner 1/8 of the cube in the spatial directions
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
            
            
    def calc_noise_in_cube(self,
            masking_scheme='simple', mask=None,
            spatial_average_npix=None, spatial_average_nbeam=5.0,
            spectral_average_nchan=5, verbose=False):
        """
        From Jiayi Sun's script: 
            https://github.com/astrojysun/Sun_Astro_Tools/blob/master/sun_astro_tools/spectralcube.py

        Estimate rms noise in a (continuum-subtracted) spectral cube.
        Parameters
        ----------
        masking_scheme : {'simple', 'user'}, optional
            Scheme for flagging signal in the cube. 'simple' means to flag
            all values above 3*rms or below -3*rms (default scheme);
            'user' means to use the user-specified mask (i.e., `mask`).
        mask : `np.ndarray` object, optional
            User-specified signal mask (this parameter is ignored if
            `masking_scheme` is not 'user')
        spatial_average_npix : int, optional
            Size of the spatial averaging box, in terms of pixel number
            If not None, `spatial_average_nbeam` will be ingored.
            (Default: None)
        spatial_average_nbeam : float, optional
            Size of the spatial averaging box, in the unit of beam FWHM
            (Default: 5.0)
        spectral_average_nchan : int, optional
            Size of the spectral averaging box, in terms of channel number
            (Default: 5)
        verbose : bool, optional
            Whether to print the detailed processing information in terminal
            Default is to not print.

        Returns
        -------
        rmscube : SpectralCube object
            Spectral cube containing the rms noise at each ppv location
        """

        if masking_scheme not in ['simple', 'user']:
            raise ValueError("'masking_scheme' should be specified as"
                             "either 'simple' or 'user'")
        elif masking_scheme == 'user' and mask is None:
            raise ValueError("'masking_scheme' set to 'user', yet "
                             "no user-specified mask found")

        # Calculate the noise in the line-free part of the PB UN/CORRECTED cube
        cube_pbcorr, cube_uncorr = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                    savepath=self.savepath, tosave=self.tosave, sample=self.sample).readfits()

        # Centre and clip empty rows and columns to get it in the same shape as the other products
        if not self.redo_clip:
            if os.path.exists(self.savepath + 'noise_subcube_slab.fits'):
                noisecube = fits.read(self.savepath + 'noise_subcube_slab.fits')[0]
            else:
                _, noisecube = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                        savepath=self.savepath, tosave=self.tosave, sample=self.sample).do_clip(clip_also_nat='noise')
        else:
            _, noisecube = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, sun=self.sun,
                                    savepath=self.savepath, tosave=self.tosave, sample=self.sample).do_clip(clip_also_nat='noise')

        # extract negative values (only needed if masking_scheme='simple')
        if masking_scheme == 'simple':
            if verbose:
                print("Extracting negative values...")
            negdata = np.where(noisecube.data < 0, noisecube.data, np.nan)
            negdata = np.stack([negdata, -1 * negdata], axis=-1)
        else:
            negdata = None

        # find rms noise as a function of channel
        if verbose:
            print("Estimating rms noise as a function of channel...")
        if masking_scheme == 'user':
            mask_v = mask
        elif masking_scheme == 'simple':
            rms_v = mad_std(negdata, axis=(1, 2, 3), ignore_nan=True)
            uplim_v = (3 * rms_v).reshape(-1, 1, 1)
            lolim_v = (-3 * rms_v).reshape(-1, 1, 1)
            mask_v = (((noisecube.data - uplim_v) < 0) &
                      ((noisecube.data - lolim_v) > 0))

        rms_v = mad_std(np.where(mask_v == 1, noisecube.data, np.nan), axis=(1, 2), ignore_nan=True)
        rms_v = ndimage.generic_filter(rms_v, np.nanmedian,
                               mode='constant', cval=np.nan,
                               size=spectral_average_nchan)

        # find rms noise as a function of sightline
        if verbose:
            print("Estimating rms noise as a function of sightline...")
        if masking_scheme == 'user':
            mask_s = mask
        elif masking_scheme == 'simple':
            rms_s = mad_std(negdata, axis=(0, 3), ignore_nan=True)
            uplim_s = 3 * rms_s
            lolim_s = -3 * rms_s
            mask_s = (((noisecube.data - uplim_s) < 0) &
                      ((noisecube.data - lolim_s) > 0))
        rms_s = mad_std(np.where(mask_s == 1, noisecube.data, np.nan), axis=0, ignore_nan=True)
        if spatial_average_npix is None:
            beamFWHM_pix = cube_pbcorr.header['BMAJ'] / cube_pbcorr.header['CDELT2']
            beamFWHM_pix = np.max([beamFWHM_pix, 3])
            spatial_average_npix = int(spatial_average_nbeam * beamFWHM_pix)
        rms_s = ndimage.generic_filter(rms_s, np.nanmedian, mode='constant', cval=np.nan, size=spatial_average_npix)

        # create rms noise cube from the tensor product of rms_v and rms_s
        if verbose:
            print("Creating rms noise cube (direct tensor product)...")
        rmscube = np.einsum('i,jk', rms_v, rms_s)

        # correct the normalization of the rms cube
        if masking_scheme == 'user':
            mask_n = mask
        elif masking_scheme == 'simple':
            rms_n = mad_std(negdata, ignore_nan=True)
            uplim_n = 3 * rms_n
            lolim_n = -3 * rms_n
            mask_n = (((noisecube.data - uplim_n) < 0) &
                      ((noisecube.data - lolim_n) > 0))
        rms_n = mad_std(noisecube.data[mask_n])
        rmscube /= rms_n

        # Write as FITS file
        rmscube_hdu = fits.PrimaryHDU(rmscube, noisecube.header)
        rmscube_hdu.writeto(self.savepath + 'rms_cube.fits', overwrite=True)

        return rmscube_hdu


    def readfits(self):
        """
        Read in the fits files containing the primary beam corrected and uncorrected specral cubes.
        :return: Two HDU files containing the spectral cube with and without primary beam correction applied,
        respectively, and their headers.
        """

        cube_pbcorr = fits.open(self.path_pbcorr)[0]
        cube_uncorr = fits.open(self.path_uncorr)[0]

        if self.galaxy.stokes:
            cube_pbcorr = self.remove_stokes(cube_pbcorr)
            cube_uncorr = self.remove_stokes(cube_uncorr)

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


    def cut_empty_rows(self, cube, noisecube=None):
        """
        Finds rows with no data and trims them from the cube. Leaves an empty
        border of one beam size around the edges.

        Parameters
        ----------
        cube : FITS file.
            3D cube containing the spectral line data.
            
        noisecube : FITS file, optional
            3D cube containing the line-free channels. The default is None.

        Returns
        -------
        cube : FITS file
            3D input cube with empty rows trimmed.
        noisecube : TYPE
            3D line-free input cube with empty rows trimmed.

        """

        w = wcs.WCS(cube.header, naxis=2)
        #try:
        #    centre_sky = SkyCoord(cube.header['OBSRA'], cube.header['OBSDEC'], unit=(u.deg, u.deg))
        #except:
        #    ra = Ned.query_object(self.galaxy.name)['RA'][0]
        #    dec = Ned.query_object(self.galaxy.name)['DEC'][0]
        #    centre_sky = SkyCoord(ra, dec, unit=(u.deg, u.deg))

        #centre_pix = wcs.utils.skycoord_to_pixel(centre_sky, w)

        beam = cube.header['BMAJ']  # deg
        res = cube.header['CDELT2']  # deg/pixel
        beam_pix = beam / res

        # Find empty rows
        empty_x = np.all(cube.data == 0, axis=(0, 2))

        # If the number of empty rows is larger than the beamsize, cut away rows and leave as many as the beam size
        if np.sum(empty_x) > beam_pix:
            beam_pix = int(np.round(beam_pix))
            idx_false = [i for i, x in enumerate(empty_x) if not x]
            first_false = idx_false[0]
            last_false = idx_false[-1]
            empty_x[first_false:last_false] = False  # Make sure empty rows in the middle of the data are not affected
            last_false += 1
            if first_false - beam_pix > 0:
                empty_x[first_false - beam_pix:first_false] = False
            else:
                empty_x[0:first_false] = False
            empty_x[last_false:last_false + beam_pix] = False

            cube.data = cube.data[:, ~empty_x, :]
            if noisecube:
                noisecube.data = noisecube.data[:, ~empty_x, :]

            # Adjust the central pixel in the image header correspondingly
            pix_shift = [i for i, x in enumerate(empty_x) if not x][1] - 1
            if empty_x[0]:
                cube.header['CRPIX2'] -= pix_shift
            cube.header['NAXIS2'] = cube.shape[1]
            if noisecube:
                noisecube.header['CRPIX2'] -= pix_shift
                noisecube.header['NAXIS2'] = noisecube.shape[1]

        if not noisecube:
            noisecube = np.array([0])

        return cube, noisecube


    def cut_empty_columns(self, cube, noisecube=None):
        """
        Finds columns with no data and trims them from the cube. Leaves an empty
        border of one beam size around the edges.

        Parameters
        ----------
        cube : FITS file.
            3D cube containing the spectral line data.
            
        noisecube : FITS file, optional
            3D cube containing the line-free channels. The default is None.

        Returns
        -------
        cube : FITS file
            3D input cube with empty columns trimmed.
        noisecube : TYPE
            3D line-free input cube with empty columns trimmed.

        """

        if len(noisecube.shape) == 1:
            noisecube = None

        beam = cube.header['BMAJ']  # deg
        res = cube.header['CDELT2']  # deg/pixel
        beam_pix = beam / res

        # Find empty columns
        empty_y = np.all(cube.data == 0, axis=(0, 1))

        # If the number of empty columns is larger than the beamsize, cut away rows and leave as many as the beam size
        if np.sum(empty_y) > beam_pix:
            beam_pix = int(np.round(beam_pix))
            idx_false = [i for i, x in enumerate(empty_y) if not x]
            first_false = idx_false[0]
            last_false = idx_false[-1]
            empty_y[first_false:last_false] = False  # Make sure empty rows in the middle of the data are not affected
            last_false += 1
            if first_false - beam_pix > 0:
                empty_y[first_false - beam_pix:first_false] = False
            else:
                empty_y[0:first_false] = False
            empty_y[last_false:last_false + beam_pix] = False

            cube.data = cube.data[:, :, ~empty_y]
            if noisecube:
                noisecube.data = noisecube.data[:, :, ~empty_y]

            # Adjust the header
            pix_shift = [i for i, x in enumerate(empty_y) if not x][1] - 1
            if empty_y[0]:
                cube.header['CRPIX1'] -= pix_shift
            cube.header['NAXIS1'] = cube.shape[2]
            if noisecube:
                noisecube.header['CRPIX1'] -= pix_shift
                cube.header['NAXIS1'] = noisecube.shape[2]

        if not noisecube:
            noisecube = np.array([0])

        return cube, noisecube


    def centre_data(self, cube, noisecube=None):
        """
        Pad the image with zeros so that the centre of the galaxy overlaps with the centre of the
        cube
        :return:
        """

        # Read in central coordinates from NED
        ra = Ned.query_object(self.galaxy.name)['RA'][0]
        dec = Ned.query_object(self.galaxy.name)['DEC'][0]

        # Find the central pixel based on these coordinates and WCS info
        w = wcs.WCS(cube.header, naxis=2)
        centre_sky = SkyCoord(ra, dec, unit=(u.deg, u.deg))
        centre_pix = wcs.utils.skycoord_to_pixel(centre_sky, w)

        # The amount the centre needs to shift to overlap with the central coordinates
        shift_x = int(np.round(centre_pix[1] - cube.shape[1] / 2))
        shift_y = int(np.round(centre_pix[0] - cube.shape[2] / 2))

        # Pad the image with twice the amount it has to shift, so that the new centre overlaps with the coordinates
        if shift_x > 0:
            temp = np.zeros((cube.shape[0], cube.shape[1] + shift_x * 2, cube.shape[2]))
            temp[:, 0:cube.shape[1], :] = cube.data
            if noisecube:
                temp_noise = np.zeros((noisecube.shape[0], noisecube.shape[1] + shift_x * 2, noisecube.shape[2]))
                temp_noise[:, 0:noisecube.shape[1], :] = noisecube.data
        elif shift_x < 0:
            temp = np.zeros((cube.shape[0], cube.shape[1] + abs(shift_x) * 2, cube.shape[2]))
            temp[:, temp.shape[1] - cube.shape[1]:temp.shape[1], :] = cube.data
            if noisecube:
                temp_noise = np.zeros((noisecube.shape[0], noisecube.shape[1] + abs(shift_x) * 2, noisecube.shape[2]))
                temp_noise[:, temp_noise.shape[1] - noisecube.shape[1]:temp_noise.shape[1], :] = noisecube.data
        else:
            temp = cube.data
            if noisecube:
                temp_noise = noisecube.data

        # Same in the y-direction
        if shift_y > 0:
            cube_new = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2] + shift_y * 2))
            cube_new[:, :, 0:temp.shape[2]] = temp
            if noisecube:
                noisecube_new = np.zeros((temp_noise.shape[0], temp_noise.shape[1], temp_noise.shape[2] + shift_y * 2))
                noisecube_new[:, :, 0:temp_noise.shape[2]] = temp_noise
        elif shift_y < 0:
            cube_new = np.zeros((temp.shape[0], temp.shape[1], temp.shape[2] + abs(shift_y) * 2))
            cube_new[:, :, cube_new.shape[2] - temp.shape[2]:cube_new.shape[2]] = temp
            if noisecube:
                noisecube_new = np.zeros((temp_noise.shape[0], temp_noise.shape[1], temp_noise.shape[2] + abs(shift_y) * 2))
                noisecube_new[:, :, noisecube_new.shape[2] - temp_noise.shape[2]:noisecube_new.shape[2]] = temp_noise
        else:
            cube_new = temp
            if noisecube:
                noisecube_new = temp_noise

        new_header = cube.header.copy()
        # Only change the CRPIX if the padding happens BEFORE the current CRPIX
        if shift_y < 0:
            new_header['CRPIX1'] = cube.header['CRPIX1'] - 2 * shift_y
        if shift_x < 0:
            new_header['CRPIX2'] = cube.header['CRPIX2'] - 2 * shift_x
        new_header['NAXIS1'] = cube_new.shape[2]
        new_header['NAXIS2'] = cube_new.shape[1]

        cube_hdu = fits.PrimaryHDU(cube_new, new_header)

        if not noisecube:
            noisecube_new = np.array([0])

        noisecube_hdu = fits.PrimaryHDU(noisecube_new, new_header)

        if noisecube:
            noisecube_hdu.header['NAXIS3'] = noisecube.header['NAXIS3']

        return cube_hdu, noisecube_hdu


    def make_square(self, cube, noisecube=None):
        """
        Pad the cube in such a way that its spatial dimensions become square.

        Parameters
        ----------
        cube : FITS file
            The 3D spectral line cube.
        noisecube : FITS file, optional
            The 3D line-free cube. The default is None.

        Returns
        -------
        square_cube_hdu : FITS file
            Input spectral line cube with extra rows/columns so that 
            NAXIS1 = NAXIS2.
        noisecube_hdu : TYPE
            Input line-free cube with extra rows/columns so that 
            NAXIS1 = NAXIS2.

        """

        img = np.sum(cube.data, axis=0)

        shape_diff = np.max(img.shape) - np.min(img.shape)
        square_cube = np.zeros((cube.shape[0], np.max(img.shape), np.max(img.shape)))
        square_noisecube = np.zeros((noisecube.shape[0], np.max(img.shape), np.max(img.shape)))

        new_header = cube.header.copy()

        if img.shape[0] > img.shape[1]:
            square_cube[:, :, int(shape_diff / 2):int(shape_diff / 2 + img.shape[1])] = cube.data
            square_noisecube[:, :, int(shape_diff / 2):int(shape_diff / 2 + img.shape[1])] = noisecube.data
            new_header['CRPIX1'] = cube.header['CRPIX1'] + int(shape_diff / 2)
        else:
            square_cube[:, int(shape_diff / 2):int(shape_diff / 2 + img.shape[0]), :] = cube.data
            square_noisecube[:, int(shape_diff / 2):int(shape_diff / 2 + img.shape[0]), :] = noisecube.data
            new_header['CRPIX2'] = cube.header['CRPIX2'] + int(shape_diff / 2)

        new_header['NAXIS1'] = cube.shape[2]
        new_header['NAXIS2'] = cube.shape[1]

        square_cube_hdu = fits.PrimaryHDU(square_cube, new_header)

        if not noisecube:
            square_noisecube = np.array([0])

        noisecube_hdu = fits.PrimaryHDU(square_noisecube, new_header)

        if noisecube:
            try:
                noisecube_hdu.header['NAXIS3'] = noisecube.header['NAXIS3']
            except:
                pass

        return square_cube_hdu, noisecube_hdu


    def preprocess(self, cube, noisecube=None):
        """
        Perform all the steps to create the (spatially) smallest, square 
        version of the cube, with padding around the edges corresponding to 
        one beam size.

        Parameters
        ----------
        cube : TYPE
            DESCRIPTION.
        noisecube : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        cube : TYPE
            DESCRIPTION.
        noisecube : TYPE
            DESCRIPTION.

        """
        cube, noisecube = self.cut_empty_rows(cube, noisecube)
        cube, noisecube = self.cut_empty_columns(cube, noisecube)
        cube, noisecube = self.centre_data(cube, noisecube)
        cube, noisecube = self.make_square(cube, noisecube)
        return cube, noisecube


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
        start, stop = self.do_clip(get_chans=True)

        emiscube = cube.data[start:stop, :, :]
        noisecube = np.concatenate((cube.data[:start, :, :], cube.data[stop:, :, :]), axis=0)

        emiscube_hdu = fits.PrimaryHDU(emiscube, cube.header)
        emiscube_hdu.header['NAXIS3'] = emiscube.shape[0]

        noisecube_hdu = fits.PrimaryHDU(noisecube, cube.header)
        noisecube_hdu.header['NAXIS3'] = noisecube.shape[0]

        return emiscube_hdu, noisecube_hdu


    def create_smooth_mask(self, emiscube_smooth, noisecube, return_rms=False):
        """
        Creates a mask of the input cube where spaxels above the desired SNR 
        are set to 1 and spaxels below the desired SNR are set to 0.

        Parameters
        ----------
        emiscube_smooth : FITS file
            3D cube containing the spectral line and smoothed in both spatial
            directions and the spectral direction, which forms the base of the
            mask.
        noisecube : FITS file
            3D cube containing the line-free channels, which will be used to
            measure the RMS.
        return_rms : bool, optional
            If set to 'True' the function will only return the RMS without 
            carrying on to making the mask. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        #cube_pbcorr, cube_uncorr = self.readfits()

        #emiscube, noisecube = self.split_cube(cube_uncorr)
        inner_noisecube = self.innersquare(noisecube.data)
        rms = np.nanstd(inner_noisecube)

        if return_rms:
            return rms

        #emiscube_smooth, noisecube_smooth = self.split_cube(smooth_cube)

        emiscube_smooth.data[emiscube_smooth.data < self.galaxy.cliplevel * rms] = 0
        emiscube_smooth.data[emiscube_smooth.data > self.galaxy.cliplevel * rms] = 1

        return emiscube_smooth.data.astype(bool)


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

        if self.galaxy.prune_by_npix:
            prune_by_npix = self.galaxy.prune_by_npix
        else:
            res = cube.header['CDELT2']  # deg. / pix.
            bmaj_pix = cube.header['BMAJ'] / res  # deg. / (deg. / pix.)
            bmin_pix = cube.header['BMIN'] / res  # deg. / (deg. / pix.)
            beam_area_pix = np.pi * bmaj_pix * bmin_pix
            prune_by_npix = beam_area_pix * self.galaxy.prune_by_fracbeam

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

        if self.galaxy.expand_by_npix:
            expand_by_npix = int(self.galaxy.expand_by_npix)
        else:
            res = cube.header['CDELT2']  # deg. / pix.
            bmaj = cube.header['BMAJ']  # deg.
            bmin = cube.header['BMIN']  # deg.
            beam_hwhm_pix = np.average([bmaj, bmin]) / res / 2  # deg. / (deg. / pix.)
            expand_by_npix = int(beam_hwhm_pix * self.galaxy.expand_by_fracbeam)

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
        for i in range(self.galaxy.expand_by_nchan):
            tempmask = np.roll(mask, shift=1, axis=0)
            tempmask[0, :] = False
            mask |= tempmask
            tempmask = np.roll(mask, shift=-1, axis=0)
            tempmask[-1, :] = False
            mask |= tempmask

        return mask


    def sun_method(self, emiscube, noisecube, calc_rms=False):
        """
        Apply Jiayi Sun's clipping method', including options to prune 
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
                self.galaxy.nchan_low and self.galaxy.cliplevel_low and self.galaxy.nchan_high and
                self.galaxy.cliplevel_high):
            raise AttributeError('If you want to use Sun\'s method, please provide "nchan_low", "cliplevel_low", '
                                 '"nchan_high", and "cliplevel_high".')

        # Estimate the rms from the spatial inner part of the cube
        inner = self.innersquare(noisecube.data)
        rms = np.nanstd(inner)

        if calc_rms:
            return rms

        snr = emiscube.data / rms

        # Generate core mask
        mask_core = (snr > self.galaxy.cliplevel_high).astype(bool)
        for i in range(self.galaxy.nchan_high - 1):
            mask_core &= np.roll(mask_core, shift=1, axis=0)
        mask_core[:self.galaxy.nchan_high - 1] = False
        for i in range(self.galaxy.nchan_high - 1):
            mask_core |= np.roll(mask_core, shift=-1, axis=0)

        # Generate wing mask
        mask_wing = (snr > self.galaxy.cliplevel_low).astype(bool)
        for i in range(self.galaxy.nchan_low - 1):
            mask_wing &= np.roll(mask_wing, shift=1, axis=0)
        mask_wing[:self.galaxy.nchan_low - 1] = False
        for i in range(self.galaxy.nchan_low - 1):
            mask_wing |= np.roll(mask_wing, shift=-1, axis=0)

        # Dilate core mask inside wing mask
        mask = binary_dilation(mask_core, iterations=0, mask=mask_wing)

        # Prune detections with small projected areas on the sky
        if self.galaxy.prune_by_fracbeam or self.galaxy.prune_by_npix:
            mask = self.prune_small_detections(emiscube, mask)

        # Expand along spatial dimensions by a fraction of the beam FWHM
        if self.galaxy.expand_by_fracbeam or self.galaxy.expand_by_npix:
            mask = self.expand_along_spatial(emiscube, mask)

        # Expand along spectral dimension by a number of channels
        if self.galaxy.expand_by_nchan:
            mask = self.expand_along_spectral(mask)

        #mask = np.ones(emiscube.shape)

        return mask


    def smooth_mask(self, cube, noisecube, return_rms=False):
        """
        Apply a Gaussian blur, using sigma = 4 in the velocity direction 
        (from past experience this value works best), to the non-pb-corrected 
        cube, for use of the simple smooth + clip strategy. 
        The mode 'nearest' seems to give the best results.

        Parameters
        ----------
        cube : FITS file
            The non-pb-corrected part of the ALMA cube containing the spectral
            line.
        noisecube : FITS file
            The non-pb-corrected part of the ALMA cube containing the line-free
            channels.
        return_rms : bool, optional
            If set to "True", the function will return the RMS estimate and not
            continue with the smoothing. The default is False.

        Returns
        -------
        mask : FITS file
            A mask with the dimensions of the input cube, based on the smoothed
            version of this cube.

        """

        beam = np.array([cube.header['BMAJ'], cube.header['BMIN']]) / cube.header['CDELT2']
        sigma = 1.5 * cube.header['BMAJ'] / cube.header['CDELT2']  # / np.sqrt(8. * np.log(2.))

        '''
        xpixels = cube.shape[1]
        ypixels = cube.shape[2]
        cent = [xpixels / 2, ypixels / 2]
        rot = 0
        dirfac = 1

        # Convolve with 1.5 the beam in the spatial direction
        # Calculate the psf
        x, y = np.indices((int(xpixels), int(ypixels)), dtype='float')
        x -= cent[0]
        y -= cent[1]

        a = (np.cos(np.radians(rot)) ** 2) / (2.0 * (sigma[0] ** 2)) + (np.sin(np.radians(rot)) ** 2) / (
        2.0 * (sigma[1] ** 2))
        b = ((dirfac) * (np.sin(2.0 * np.radians(rot)) ** 2) / (4.0 * (sigma[0] ** 2))) + (
        (-1 * dirfac) * (np.sin(2.0 * np.radians(rot)) ** 2) / (4.0 * (sigma[1] ** 2)))
        c = (np.sin(np.radians(rot)) ** 2) / (2.0 * (sigma[0] ** 2)) + (np.cos(np.radians(rot)) ** 2) / (
        2.0 * (sigma[1] ** 2))

        psf = np.exp(-1 * (a * (x ** 2) - 2.0 * b * (x * y) + c * (y ** 2)))
        '''
        #cube_spatial_smooth = np.empty(cube.shape)
        #for i in range(cube.shape[0]):
        #    cube_spatial_smooth[i, :, :] = convolve_fft(cube.data[i, :, :], psf)

        #smooth_cube = ndimage.filters.gaussian_filter1d(cube_spatial_smooth, 4, axis=0, order=0, mode='nearest')
        smooth_cube = ndimage.uniform_filter(cube.data, size=[4, sigma, sigma], mode='constant')  # mode='nearest'

        smooth_hdu = fits.PrimaryHDU(smooth_cube, cube.header)
        mask = self.create_smooth_mask(smooth_hdu, noisecube, return_rms=return_rms)

        return mask


    def do_clip(self, clip_also=None, clip_also_nat='noise', get_chans=False):
        """
        Perform the clipping of the data cube, either using the method adopted
        and optimised by Jiayi Sun, or the simpler method from Dame+11.

        Parameters
        ----------
        clip_also : FITS file, optional
            A cube with the same dimensions as the input ALMA cube, which will 
            be trimmed in the same way as the data cube, so it will end up 
            having the same dimensions. The default is None.
        clip_also_nat : str, optional
            The nature of "clip_also", 'noise', 'mask', or 'pb'. 
            The default is 'noise'.
        get_chans : bool, optional
            If set to true, this function will only return the first and last 
            channels containing the emission line. The default is False.

        Returns
        -------
        FITS file
            The clipped and trimmed version of the input data cube.
        FITS file
            The corresponding noise cube, of the same dimensions as the output
            data cube.

        """
        cube_pbcorr, cube_uncorr = self.readfits()
        cube_uncorr_copy = cube_uncorr.copy()

        if self.sun:

            # Get a rough estimate of the noise in order to do the clipping
            noisecube_temp = np.concatenate((cube_uncorr_copy.data[:10, :, :], cube_uncorr_copy.data[-10:, :, :]),
                                            axis=0)
            noisecube_temp_hdu = fits.PrimaryHDU(noisecube_temp, cube_uncorr_copy.header)

            # Create an initial mask and identify first and last channel containing emission
            mask_full = self.sun_method(cube_uncorr_copy, noisecube_temp_hdu)
            mask_idx = np.where(mask_full == 1)[0]
            start = mask_idx[0]
            stop = mask_idx[-1]

            # Create an updated noise cube
            noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:start, :, :],
                                               cube_uncorr_copy.data[stop:, :, :]), axis=0)
            noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)

            # Make a more accurate mask based on the new noise cube
            mask_full = self.sun_method(cube_uncorr_copy, noisecube_uncorr_hdu)
            mask_hdu = fits.PrimaryHDU(mask_full.astype(int), cube_pbcorr.header)
            mask_idx = np.where(mask_full == 1)[0]
            start = mask_idx[0]
            stop = mask_idx[-1]

            if get_chans:
                return start, stop

            # Spit the cube in an emission and noise part
            emiscube_pbcorr = cube_pbcorr.data[start:stop, :, :]
            emiscube_uncorr = cube_uncorr.data[start:stop, :, :]

            noisecube_pbcorr = np.concatenate((cube_pbcorr.data[:start, :, :],
                                               cube_pbcorr.data[stop:, :, :]), axis=0)
            noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:start, :, :],
                                               cube_uncorr_copy.data[stop:, :, :]), axis=0)

            emiscube_uncorr_hdu = fits.PrimaryHDU(emiscube_uncorr, cube_uncorr_copy.header)
            noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)
            noisecube_pbcorr_hdu = fits.PrimaryHDU(noisecube_pbcorr, cube_uncorr_copy.header)

            mask = self.sun_method(emiscube_uncorr_hdu, noisecube_uncorr_hdu)

            if self.tosave:
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
                mask_hdu.header.comments['CLIP_RMS'] = 'rms value used for clipping in K km/s'
                mask_hdu.writeto(self.savepath + 'mask_cube.fits', overwrite=True)

        else:
            # Get a rough estimate of the noise in order to do the clipping
            noisecube_temp = np.concatenate((cube_uncorr_copy.data[:10, :, :], cube_uncorr_copy.data[-10:, :, :]),
                                            axis=0)
            noisecube_temp_hdu = fits.PrimaryHDU(noisecube_temp, cube_uncorr_copy.header)

            # Create an initial mask and identify first and last channel containing emission
            mask_full = self.smooth_mask(cube_uncorr_copy, noisecube_temp_hdu)
            mask_idx = np.where(mask_full == 1)[0]
            start = mask_idx[0]
            stop = mask_idx[-1]

            # Create an updated noise cube
            noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:start, :, :],
                                               cube_uncorr_copy.data[stop:, :, :]), axis=0)
            noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)

            # Make a more accurate mask based on the new noise cube
            mask_full = self.smooth_mask(cube_uncorr_copy, noisecube_uncorr_hdu)
            mask_hdu = fits.PrimaryHDU(mask_full.astype(int), cube_pbcorr.header)
            mask_idx = np.where(mask_full == 1)[0]
            start = mask_idx[0]
            stop = mask_idx[-1]

            if get_chans:
                return start, stop

            # Spit the cube in an emission and noise part
            emiscube_pbcorr = cube_pbcorr.data[start:stop, :, :]
            emiscube_uncorr = cube_uncorr.data[start:stop, :, :]

            noisecube_pbcorr = np.concatenate((cube_pbcorr.data[:start, :, :],
                                               cube_pbcorr.data[stop:, :, :]), axis=0)
            noisecube_uncorr = np.concatenate((cube_uncorr_copy.data[:start, :, :],
                                               cube_uncorr_copy.data[stop:, :, :]), axis=0)

            emiscube_uncorr_hdu = fits.PrimaryHDU(emiscube_uncorr, cube_uncorr_copy.header)
            noisecube_uncorr_hdu = fits.PrimaryHDU(noisecube_uncorr, cube_uncorr_copy.header)

            mask = self.smooth_mask(emiscube_uncorr_hdu, noisecube_uncorr_hdu)

            mask_idx = np.where(mask == 1)[0]
            self.galaxy.start = mask_idx[0]
            self.galaxy.stop = mask_idx[-1]

            mask_hdu = fits.PrimaryHDU(mask.astype(int), cube_pbcorr.header)

            if self.tosave:
                mask_hdu.header.add_comment('Cube was clipped using the Dame11 masking method', before='BUNIT')
                mask_hdu.header.pop('BTYPE')
                mask_hdu.header.pop('BUNIT')
                try:
                    mask_hdu.header.pop('DATAMAX')
                    mask_hdu.header.pop('DATAMIN')
                except:
                    pass
                try:
                    mask_hdu.header.pop('JTOK')
                except:
                    pass
                mask_hdu.header.pop('RESTFRQ')
                mask_hdu.header['CLIP_RMS'] = self.smooth_mask(emiscube_uncorr_hdu, noisecube_uncorr_hdu, return_rms=True)
                mask_hdu.header.comments['CLIP_RMS'] = 'rms value used for clipping in K km/s'
                mask_hdu.writeto(self.savepath + 'mask_cube.fits', overwrite=True)

        emiscube_pbcorr[mask == 0] = 0
        clipped_hdu = fits.PrimaryHDU(emiscube_pbcorr, cube_pbcorr.header)

        # Adjust the header to match the velocity range used
        clipped_hdu.header['CRVAL3'] += start * clipped_hdu.header['CDELT3']

        if self.tosave:
            clipped_hdu.writeto(self.savepath + 'subcube.fits', overwrite=True)

        # Do some pre-processing to make the creation of the moments easier
        if not clip_also:
            clipped_hdu_temp, noisecube_hdu = self.preprocess(clipped_hdu, noisecube=noisecube_pbcorr_hdu)
        else:
            clipped_hdu_temp, noisecube_hdu = self.preprocess(clipped_hdu, noisecube=clip_also)
        temp1, unclipped_trimmed_hdu = self.cut_empty_columns(mask_hdu, noisecube=cube_pbcorr)
        temp2, unclipped_trimmed_hdu = self.cut_empty_rows(temp1, noisecube=unclipped_trimmed_hdu)
        unclipped_trimmed_hdu.writeto(self.savepath + 'unclipped_subcube.fits', overwrite=True)
        clipped_hdu = clipped_hdu_temp

        if self.tosave:
            clipped_hdu.writeto(self.savepath + 'subcube_slab.fits', overwrite=True)
            if clip_also_nat == 'noise':
                noisecube_hdu.writeto(self.savepath + 'noise_subcube_slab.fits', overwrite=True)
            elif clip_also_nat == 'mask':
                noisecube_hdu.writeto(self.savepath + 'mask_subcube_slab.fits', overwrite=True)
            elif clip_also_nat == 'pb':
                noisecube_hdu.writeto(self.savepath + 'pb_subcube_slab.fits', overwrite=True)

        return clipped_hdu, noisecube_hdu