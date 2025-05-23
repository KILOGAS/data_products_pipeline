from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os
from astropy.cosmology import FlatLambdaCDM


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


def create_vel_array(cube, savepath=None):
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
    
    if savepath:
        np.save(savepath + 'vel_array.npy', vel_array)

    return vel_array, vel_narray, vel_array_full, v_step


def calc_moms(cube, galaxy, glob_cat, savepath=None, units='K km/s', alpha_co=4.35, R21=0.7):
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

    mom0 = np.nansum((cube.data * dv), axis=0)
    
    mom0[mom0 < 0] = np.nan
    
    # Set redshift parameters needed for physical unit calculations
    glob_tab = fits.open(glob_cat)[1]        
    z = glob_tab.data['Z'][glob_tab.data['KGAS_ID'] == int(galaxy.split('KGAS')[1])][0]
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    pc_to_pix = (cube.header['CDELT2'] * cosmo.kpc_proper_per_arcmin(z).value * 60 * 1000) ** 2
    
    if units == 'Msol pc-2':
        mom0 *= alpha_co
        mom0 /= R21
        mom0 *= (1 + z)
        
    elif units == 'Msol/pix':
        mom0 *= alpha_co
        mom0 /= R21
        mom0 *= (1 + z)
        mom0 *= pc_to_pix
    
    elif units == 'K km/s pc^2':
        mom0 *= pc_to_pix
        mom0 *= (1 + z)
        
    mom0[np.isinf(mom0)] = np.nan
    mom0[mom0 == 0] = np.nan    
        
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
    if units == 'K km/s pc^2':
        mom0_hdu.header['BTYPE'] = 'Lco'
        mom0_hdu.header.comments['BTYPE'] = 'CO luminosity'
        mom0_hdu.header['BUNIT'] = units
        mom0_hdu.header.comments['BUNIT'] = ''
    elif units == 'Msol pc-2':
        mom0_hdu.header['BTYPE'] = 'mmol pc^-2'
        mom0_hdu.header.comments['BTYPE'] = 'Molecular gas mass surface density'
        mom0_hdu.header['BUNIT'] = 'Msol pc^-2'
        mom0_hdu.header.comments['BUNIT'] = ''
    elif units == 'Msol/pix':
        mom0_hdu.header['BTYPE'] = 'mmol_pix'
        mom0_hdu.header.comments['BTYPE'] = 'Molecular gas mass in pixel'
        mom0_hdu.header['BUNIT'] = 'Msol'
        mom0_hdu.header.comments['BUNIT'] = ''
    else:
        mom0_hdu.header['BTYPE'] = 'Ico'
        mom0_hdu.header.comments['BTYPE'] = 'CO surface brightness'
        mom0_hdu.header['BUNIT'] = units
        mom0_hdu.header.comments['BUNIT'] = ''

    mom1_hdu.header['BTYPE'] = 'co_vel'
    mom2_hdu.header['BTYPE'] = 'co_obs_lw'
    
    mom1_hdu.header.comments['BTYPE'] = 'Absolute CO velocity'
    mom2_hdu.header.comments['BTYPE'] = 'Observed line of sight CO line width'
     
    mom1_hdu.header['BUNIT'] = 'km/s'; mom1_hdu.header.comments['BUNIT'] = ''
    mom2_hdu.header['BUNIT'] = 'km/s'; mom2_hdu.header.comments['BUNIT'] = ''
    #if not self.sample == 'viva' or self.sample == 'things':
    #    mom0_hdu.header['ALPHA_CO'] = alpha_co; #mom0_hdu.header.comments['ALPHA_CO'] = 'Assuming a line ratio of 0.8'
    #mom1_hdu.header['SYSVEL'] = sysvel; mom1_hdu.header.comments['SYSVEL'] = 'km/s'

    if savepath:
        if units == 'K km/s':
            mom0_hdu.writeto(savepath + 'Ico_K_kms-1.fits', overwrite=True)
        elif units == 'K km/s pc^2':
            mom0_hdu.writeto(savepath + 'Lco_K_kms-1_pc2.fits', overwrite=True)
        elif units == 'Msol pc-2':
            mom0_hdu.writeto(savepath + 'mmol_pc-2.fits', overwrite=True)
        elif units == 'Msol/pix':
            mom0_hdu.writeto(savepath + 'mmol_pix-1.fits', overwrite=True)
        mom1_hdu.writeto(savepath + 'mom1.fits', overwrite=True)
        mom2_hdu.writeto(savepath + 'mom2.fits', overwrite=True)
        
        # Create a dummy alpha_co map
        #alpha_co_map = np.ones_like(mom0)
        #alpha_co_map[mom0 != mom0] = 0
        #alpha_co_map[mom0 == 0] = 0
        #alpha_co_map *= alpha_co
        #alpha_co_hdu = fits.PrimaryHDU(mom0, moment_header)
        #alpha_co_hdu.writeto(savepath + 'alpha_co.fits', overwrite=True)
        
    return mom0_hdu, mom1_hdu, mom2_hdu
    
def calc_uncs(cube, path, galaxy, glob_cat, savepath, units='K km/s', alpha_co=4.35, R21=0.7):
    
    # Calculate the number of channels by converting the cube into a boolean
    cube_bool = cube.data.copy()
    
    # Set any nans to 0, or they will be converted to True
    cube_bool[cube_bool != cube_bool] = 0
    cube_bool = cube_bool.astype('bool')
    
    N_map = np.sum(cube_bool, axis=0)
    
    # The noise map is the rms divided by the PB map. The PB map should have 
    # values only where the clipped data cube has values.
    try:
        pb_file = glob(path + '/' + galaxy + '/' + galaxy + '*.pb.fits')[0]
        pb_cube = fits.open(pb_file)[0]
    except:
        try:
            path_pbcorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.image.pbcor.ifumatched.fits"
            path_uncorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.image.ifumatched.fits"
            cube_pb_corr = fits.open(path_pbcorr)[0]
        except:
            try:
                path_pbcorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.image.pbcor.ifumatched.fits"
                path_uncorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.image.ifumatched.fits"
                cube_pb_corr = fits.open(path_pbcorr)[0]
            except:
                try:
                    path_pbcorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.contsub.image.pbcor.ifumatched.fits"
                    path_uncorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.contsub.image.ifumatched.fits"
                    cube_pb_corr = fits.open(path_pbcorr)[0]
                except:
                    path_pbcorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.contsub.image.pbcor.ifumatched.fits"
                    path_uncorr = path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.contsub.image.ifumatched.fits"
                    cube_pb_corr = fits.open(path_pbcorr)[0]
            
        cube_uncorr = fits.open(path_uncorr)[0]
        pb_cube = cube_pb_corr.copy()
        pb_cube.data = cube_uncorr.data / cube_pb_corr.data
    
    pb_cube.data[cube_bool.data != cube_bool.data] = np.nan
    noise_cube = cube.header['CLIP_RMS'] / pb_cube.data
    
    # Use the median value of the PB cube along the spectral axis to create a 
    # representative 2D map.
    noise_map = np.nanmedian(noise_cube, axis=0)
    
    # Set redshift parameters needed for physical unit calculations
    glob_tab = fits.open(glob_cat)[1]        
    z = glob_tab.data['Z'][glob_tab.data['KGAS_ID'] == int(galaxy.split('KGAS')[1])][0]
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    pc_to_pix = (cube.header['CDELT2'] * cosmo.kpc_proper_per_arcmin(z).value * 60 * 1000) ** 2
    
    mom0_uncertainty = noise_map * np.sqrt(N_map) * abs(cube.header['CDELT3'] / 1000)

    mom0_uncertainty[np.isinf(mom0_uncertainty)] = np.nan
    mom0_uncertainty[mom0_uncertainty <= 0] = np.nan    

    if units == 'Msol pc-2':
        mom0_hdu, _, _ = calc_moms(cube, galaxy, glob_cat, savepath=None, units='Msol pc-2')
        mom0_uncertainty *= alpha_co
        mom0_uncertainty /= R21
        mom0_uncertainty *= (1 + z)
        #mom0_uncertainty *= 0.434 * (mom0_uncertainty / 10 ** mom0_hdu.data)
        
        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header['BTYPE'] = 'mmol pc^-2 error'
        mom0_uncertainty_hdu.header.comments['BTYPE'] = 'Error mol. gas mass surf. dens.'
        mom0_hdu.header['BUNIT'] = 'Msol pc-2'
        mom0_hdu.header.comments['BUNIT'] = ''
   
        mom0_uncertainty_hdu.writeto(savepath + 'mmol_pc-2_err.fits', overwrite=True)
        
    elif units == 'Msol/pix':
        mom0_hdu, _, _ = calc_moms(cube, galaxy, glob_cat, savepath=None, units='Msol/pix')
        
        mom0_uncertainty *= alpha_co
        mom0_uncertainty /= R21
        mom0_uncertainty *= (1 + z)
        mom0_uncertainty *= pc_to_pix
        #mom0_uncertainty  *= 0.434 * (mom0_uncertainty / 10 ** mom0_hdu.data)
        
        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header['BTYPE'] = 'mmol error'
        mom0_uncertainty_hdu.header.comments['BTYPE'] = 'Error mol. gas mass in pixel'
        mom0_hdu.header['BUNIT'] = 'Msol pix^-1'
        mom0_hdu.header.comments['BUNIT'] = ''
        
        mom0_uncertainty_hdu.writeto(savepath + 'mmol_pix-1_err.fits', overwrite=True)
    
    elif units == 'K km/s pc^2':
        mom0_hdu, _, _, = calc_moms(cube, galaxy, glob_cat, units='K km/s pc^2')
        
        mom0_uncertainty *= pc_to_pix
        mom0_uncertainty *= (1 + z)
        
        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header['BTYPE'] = 'Lco error'
        mom0_uncertainty_hdu.header.comments['BTYPE'] = 'Error in CO luminosity'
        mom0_hdu.header['BUNIT'] = 'K km s^-1 pc^2'
        mom0_hdu.header.comments['BUNIT'] = ''
        
        mom0_uncertainty_hdu.writeto(savepath + 'Lco_K_kms-1_pc2_err.fits', overwrite=True)
        
    else:        
        mom0_hdu, mom1_hdu, mom2_hdu = calc_moms(cube, galaxy, glob_cat)
        
        mom0_uncertainty_hdu = fits.PrimaryHDU(mom0_uncertainty, mom0_hdu.header)
        mom0_uncertainty_hdu.header['BTYPE'] = 'Ico error'
        mom0_uncertainty_hdu.header.comments['BTYPE'] = 'Error in CO SB'
        mom0_hdu.header['BUNIT'] = 'K km s^-1'
        mom0_hdu.header.comments['BUNIT'] = ''
        
        mom0_uncertainty_hdu.writeto(savepath + 'Ico_K_kms-1_err.fits', overwrite=True)
        
        SN_map =  mom0_hdu.data / mom0_uncertainty
        SN_hdu = fits.PrimaryHDU(SN_map, mom0_hdu.header)
        SN_hdu.header.pop('BUNIT')
        SN_hdu.writeto(savepath + 'mom0_SN.fits', overwrite=True)
        
        mom1_uncertainty = (N_map * abs(cube.header['CDELT3'] / 1000) / (2 * np.sqrt(3))) * \
                           (mom0_uncertainty / mom0_hdu.data)  # Eqn 15 doc. Chris
        mom1_uncertainty_hdu = fits.PrimaryHDU(mom1_uncertainty, mom1_hdu.header)
               
        mom2_uncertainty = ((N_map * abs(cube.header['CDELT3'] / 1000)) ** 2 / (8 * np.sqrt(5))) * \
                           (mom0_uncertainty / mom0_hdu.data) * (mom2_hdu.data) ** -1  # Eqn 30 doc. Chris           
        mom2_uncertainty_hdu = fits.PrimaryHDU(mom2_uncertainty, mom2_hdu.header)
        
        mom1_uncertainty_hdu.writeto(savepath + 'mom1_err.fits', overwrite=True)
        mom2_uncertainty_hdu.writeto(savepath + 'mom2_err.fits', overwrite=True)
    

def calc_peak_t(cube, savepath):
    
    peak_temp = np.nanmax(cube.data, axis=0)
    
    peak_temp_hdu = fits.PrimaryHDU(peak_temp, new_header(cube.header))
    peak_temp_hdu.header['BTYPE'] = 'Peak temperature'
    peak_temp_hdu.header['BUNIT'] = 'K'; peak_temp_hdu.header.comments['BUNIT'] = ''

    peak_temp_hdu.writeto(savepath + 'peak_temp_k.fits', overwrite=True)


def perform_moment_creation(path, targets, glob_cat):
    
    files = glob(path + '**/*subcube.fits')
    
    galaxies = list(set([f.split('/')[8].split('_')[0] for f in files]))
    
    for galaxy in galaxies:
        
        if galaxy not in targets:
            continue
        
        print(galaxy)
        
        cubes = glob(path + galaxy + '/*subcube.fits')
        
        for cube in cubes:
            
            savepath = path + galaxy + '/' + cube.split('/')[-1].split('.fits')[0].split('expanded')[0]
    
            cube_fits = fits.open(cube)[0]
    
            calc_moms(cube_fits, galaxy, glob_cat=glob_cat, savepath=savepath, units='K km/s', alpha_co=4.35, R21=0.7)
            calc_moms(cube_fits, galaxy, glob_cat=glob_cat, savepath=savepath, units='K km/s pc^2', alpha_co=4.35, R21=0.7)
            calc_moms(cube_fits, galaxy, glob_cat=glob_cat, savepath=savepath, units='Msol pc-2', alpha_co=4.35, R21=0.7)
            calc_moms(cube_fits, galaxy, glob_cat=glob_cat, savepath=savepath, units='Msol/pix', alpha_co=4.35, R21=0.7)
            calc_peak_t(cube_fits, savepath=savepath)
            
            calc_uncs(cube_fits, path, galaxy, glob_cat=glob_cat, savepath=savepath, 
                      units='K km/s', alpha_co=4.35, R21=0.7)
            calc_uncs(cube_fits, path, galaxy, glob_cat=glob_cat, savepath=savepath, 
                      units='K km/s pc^2', alpha_co=4.35, R21=0.7)
            calc_uncs(cube_fits, path, galaxy, glob_cat=glob_cat, savepath=savepath, 
                      units='Msol pc-2', alpha_co=4.35, R21=0.7)
            calc_uncs(cube_fits, path, galaxy, glob_cat=glob_cat, savepath=savepath, 
                      units='Msol/pix', alpha_co=4.35, R21=0.7)


if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    perform_moment_creation(path)

    




