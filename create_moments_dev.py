from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os


def beam_area(cube, cellsize):
    return (np.pi * (cube.header['BMAJ'] / cellsize) * 
            (cube.header['BMIN'] / cellsize)) / (4 * np.log(2))


def new_header(header):
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

    header.pop('CTYPE3')
    header.pop('CRVAL3')
    header.pop('CDELT3')
    header.pop('CRPIX3')
    header.pop('CUNIT3')
    header.pop('PC3_1')
    header.pop('PC3_2')
    header.pop('PC1_3')
    header.pop('PC2_3')
    header.pop('PC3_3')
    header.pop('NAXIS3')
    
    header['NAXIS'] = 3

    return header


def create_vel_array(cube, savepath):
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


    v_ref = cube.header['CRPIX3']  # Location of the reference channel
    
    if cube.header['CTYPE3'] == 'VRAD' or cube.header['CTYPE3'] == 'VELOCITY' or cube.header['CTYPE3'] == 'VOPT':
        v_val = cube.header['CRVAL3'] / 1000  # Velocity in the reference channel, m/s to km/s
        v_step = cube.header['CDELT3'] / 1000  # Velocity step in each channel, m/s to km/s
        
    elif cube.header['CTYPE3'] == 'FREQ':
        v_val = 299792.458 * (1 - (cube.header['CRVAL3'] / 1e9) / 230.538000)
        v_shift = 299792.458 * (1 - ((cube.header['CRVAL3'] + cube.header['CDELT3']) / 1e9) / 230.538000)
        v_step = - (v_val - v_shift)
    #else:
    #    raise KeyError('Pipeline cannot deal with these units yet.')

    # Construct the velocity arrays (keep in mind that fits-files are 1 indexed)
    #vel_array = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1 + self.galaxy.start) * v_step + v_val
    vel_array = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1) * v_step + v_val
    vel_narray = np.tile(vel_array, (len(cube.data[0, 0, :]), len(cube.data[0, :, 0]), 1)).transpose()
    vel_array_full = (np.arange(0, len(cube.data[:, 0, 0])) - v_ref + 1) * v_step + v_val
    
    np.save(savepath + 'vel_array.npy', vel_array)

    return vel_array, vel_narray, vel_array_full, v_step


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


def calc_moms(cube, savepath, units='Jy/beam km/s', alpha_co=5.4):
    """
    Clip the spectral cube according to the desired method (either the 
    method Pythonified by Jiayi Sun or the more basic smooth + clip 
    strategy from Dame+11), and create moment 0, 1, and 2 maps. Saves maps 
    as fits files if so desired. Also calculate the systemic velocity from 
    the moment 1 map.

    Parameters
    ----------
    units : str, optional
        Preferred units (either 'Jy/beam km/s' or 'M_Sun/pc^2'). The default is 
                         'Jy/beam km/s'.
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

    vel_array, vel_narray, vel_fullarray, dv = create_vel_array(cube, savepath)

    if units == 'M_Sun/pc^2':
        pass
    elif units == 'Jy/beam km/s':
        mom0 = np.nansum((cube.data * dv), axis=0)
    elif units == 'K':
        pass
    elif units == 'K/pc^2':
        glob_tab = fits.open('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KILOGAS_global_catalog_FWHM.fits')[1]
        
    mom1 = np.nansum(cube.data * vel_narray, axis=0) / np.nansum(cube.data, axis=0)
    mom2 = np.sqrt(np.nansum(abs(cube.data) * (vel_narray - mom1) ** 2, axis=0) / np.nansum(abs(cube.data), axis=0))

    # Calculate the systemic velocity from the spatial inner part of the cube (to avoid PB effects)
    #inner_cube = ClipCube(self.galaxy.name, self.path_pbcorr, self.path_uncorr, savepath=self.savepath,
    #                      tosave=self.tosave, sample=self.sample).innersquare(mom1)

    #from matplotlib import pyplot as plt
    #plt.imshow(inner_cube)

    #vsys = np.nanmean(inner_cube)# + self.galaxy.sysvel_offset
    #mom1 -= vsys
    
    #plt.plot(vel_array, np.nansum(cube.data, axis=(1,2)))
    #plt.imshow(mom1, cmap='jet', vmin=13390, vmax=13880)
    #plt.imshow(mom2, cmap='jet')

    moment_header = new_header(cube.header)

    mom0_hdu = fits.PrimaryHDU(mom0, moment_header)
    mom1_hdu = fits.PrimaryHDU(mom1, moment_header)
    mom2_hdu = fits.PrimaryHDU(mom2, moment_header)

    #self.pixel_size_check(header=mom0_hdu.header)

    # Change or add any (additional) keywords to the headers
    if units == 'M_Sun/pc^2':
        mom0_hdu.header['BTYPE'] = 'Column density'
        mom0_hdu.header.comments['BTYPE'] = 'Total molecular gas (H_2 + He)'
    else:
        mom0_hdu.header['BTYPE'] = 'Integrated intensity'

    mom1_hdu.header['BTYPE'] = 'Velocity'
    mom2_hdu.header['BTYPE'] = 'Linewidth'
    mom0_hdu.header['BUNIT'] = units; mom0_hdu.header.comments['BUNIT'] = ''
    mom1_hdu.header['BUNIT'] = 'km/s'; mom1_hdu.header.comments['BUNIT'] = ''
    mom2_hdu.header['BUNIT'] = 'km/s'; mom2_hdu.header.comments['BUNIT'] = ''
    #if not self.sample == 'viva' or self.sample == 'things':
    #    mom0_hdu.header['ALPHA_CO'] = alpha_co; #mom0_hdu.header.comments['ALPHA_CO'] = 'Assuming a line ratio of 0.8'
    #mom1_hdu.header['SYSVEL'] = sysvel; mom1_hdu.header.comments['SYSVEL'] = 'km/s'

    if units == 'M_Sun/pc^2':
        mom0_hdu.writeto(savepath + 'mom0_Msolpc-2.fits', overwrite=True)
    elif units == 'Jy/beam km/s':
        mom0_hdu.writeto(savepath + 'mom0_Jyb-1_kms-1.fits', overwrite=True)
    mom1_hdu.writeto(savepath + 'mom1.fits', overwrite=True)
    mom2_hdu.writeto(savepath + 'mom2.fits', overwrite=True)
    
    
def calc_uncs(cube):
    
    # Calculate the number of channels by converting the cube into a boolean
    cube_bool = cube.data.copy()
    
    # Set any nans to 0, or they will be converted to True
    cube_bool[cube_bool != cube_bool] = 0
    
    N_map = np.sum(cube_bool, axis=0)
    
    # The noise map is the rms divided by the PB map
    #pb_map = 
    #noise_map = cube.header['CLIP_RMS'] / pb_map
    
    
def calc_peak_t(cube, savepath):
    
    peak_temp = np.nanmax(cube.data, axis=0)
    
    peak_temp_hdu = fits.PrimaryHDU(peak_temp, new_header(cube.header))
    peak_temp_hdu.header['BTYPE'] = 'Peak temperature'
    peak_temp_hdu.header['BUNIT'] = 'K'; peak_temp_hdu.header.comments['BUNIT'] = ''

    peak_temp_hdu.writeto(savepath + 'peak_temp_k.fits', overwrite=True)


def perform_moment_creation(path):
    
    #files = glob(path + '**/*subcube.fits')
    files = glob(path + '**/*test.fits')
    galaxies = list(set([f.split('/')[6].split('_')[0] for f in files]))
    
    #galaxies = ['KGAS58']
    
    for galaxy in galaxies:
             
        if not os.path.exists(path + galaxy + '/moment_maps'):
            os.mkdir(path + galaxy + '/moment_maps')
        
        cubes = glob(path + galaxy + '**/*test.fits')
        
        for cube in cubes:
            savepath = path + galaxy + '/moment_maps/' + cube.split('/')[-1].split('.fits')[0] + '_'
    
            cube_fits = fits.open(cube)[0]
    
            calc_moms(cube_fits, savepath)
            calc_peak_t(cube_fits, savepath)


if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    perform_moment_creation(path)

    




