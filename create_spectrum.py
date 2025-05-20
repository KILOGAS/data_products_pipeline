from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from photutils.aperture import CircularAperture
from glob import glob
import os
from create_moments_dev import create_vel_array


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def spectrum(cube, galaxy, start, stop, path, savepath, extra_chans=10, non_det=False):

    _, _, vel_array_full, _ = create_vel_array(cube)
    
    if non_det:
        
        table = fits.open('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KILOGAS_global_catalog_FWHM.fits')[1]
        R50 = table.data['R50_ARCSEC'][table.data['KGAS_ID'] == int(galaxy.split('KGAS')[1])][0]
        rad_pix = R50 / (cube.header['CDELT2'] * 3600)
        
        try:
            aper = CircularAperture([int(cube.shape[1] / 2), int(cube.shape[2] / 2)], rad_pix)
        except:
            print(galaxy)
            print(R50)
            return
        
        plt.figure()
        plt.imshow(np.sum(cube.data, axis=0))
        aper.plot(color='red', lw=2)        
        
        aper_mask = aper.to_mask(method='center')
        mask = aper_mask.to_image(shape=cube.shape[1:])
        
        mask[mask==0] = np.nan
        mask3d = np.tile(mask, (cube.shape[0], 1, 1))
        masked_data = mask3d * cube.data
        
        spectrum = np.nansum(masked_data, axis=(1, 2))
        
        #rms = np.nanstd(spectrum)
        #fwhm = 100  # km/s
        #sigma = fwhm / np.sqrt(8 * np.log(2))
        #x0 = vel_array_full[int(len(vel_array_full) / 2)]
        #gaussian = gauss(vel_array_full, 3*rms, x0, sigma)
        
    else:
        clip_mask = fits.open(path + galaxy + '/' + galaxy + '_mask_cube.fits')[0]
        mask = np.sum(clip_mask.data, axis=0)
        mask = mask.astype(float)
        mask[mask > 0] = 1
        
        mask3d = np.tile(mask, (cube.shape[0], 1, 1))
        masked_data = mask3d * cube.data
        masked_data[masked_data == 0] = np.nan
        
        spectrum = np.nansum(masked_data, axis=(1, 2))
        
    if start - extra_chans < 0:
        spectrum_velocities = vel_array_full[0:stop + extra_chans]
        spectrum = spectrum[0:stop + extra_chans]
        #spectrum_vel_offset = spectrum_velocities - sysvel + self.galaxy.sysvel_offset
    elif stop + extra_chans > len(vel_array_full):
        spectrum_velocities = vel_array_full[start:]
        spectrum = spectrum[start:]
    else:
        spectrum_velocities = vel_array_full[start - extra_chans:stop + extra_chans]
        spectrum = spectrum[start - extra_chans:stop + extra_chans]
        #spectrum_vel_offset = spectrum_velocities - sysvel + self.galaxy.sysvel_offset

    #rest_freq = 230538000000
    #spectrum_frequencies = rest_frequency * (1 - spectrum_velocities / 299792.458) / 1e9

    csv_header = 'Spectrum (K), Velocity (km/s)'

    np.savetxt(savepath + '_spectrum.csv',
               np.column_stack((spectrum, spectrum_velocities)),
               delimiter=',', header=csv_header)

    return spectrum, spectrum_velocities


def plot_spectrum(spectrum, velocity, extra_chans=0, x_axis='velocity', 
             useclipped=False, savepath=None):


    fig, ax = plt.subplots(figsize=(7, 7))

    if x_axis == 'velocity':
        ax.plot(velocity, spectrum, color='k', drawstyle='steps')
        x = np.arange(np.amin(velocity) - 100, np.amax(velocity) + 100, 1)
        ax.set_xlim(velocity[len(velocity) - 1] + extra_chans, velocity[0] - extra_chans)
        ax.set_xlabel(r'Velocity [km s$^{-1}$]')
    '''
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
    '''

    # Line through zero
    #zeroline = np.zeros(len(x))
    #plt.plot(x, zeroline, linestyle=':', c='r', linewidth=1)

    ax.set_ylabel('Brightness temperature [K]')

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath + '_spectrum.png', bbox_inches='tight')
        plt.savefig(savepath + '_spectrum.pdf', bbox_inches='tight')


def get_all_spectra(path, targets):
    
    files = glob(path + '**/*co2-1*image.pbcor*.fits')
    
    galaxies = list(set([f.split('/')[7].split('_')[0] for f in files]))
    
    clipping_table = fits.open('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do.fits')[1]
    KGAS_ID = clipping_table.data['KGAS_ID']
    minchan = clipping_table.data['minchan']
    maxchan = clipping_table.data['maxchan']
    clipping_chans = {'KGAS' + id.astype(str): [min, max] for id, min, max in zip(KGAS_ID, minchan, maxchan)}
    
    detected = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[6], dtype=bool)
    detected_id = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[0], dtype=int)
    
    for galaxy in galaxies:
        
        if not galaxy in targets:
            continue
        
        non_det = ~detected[detected_id == int(galaxy.split('KGAS')[1])]
        
        start = clipping_chans[galaxy][0]
        stop = clipping_chans[galaxy][1]
             
        if not os.path.exists(path + galaxy + '/moment_maps'):
            os.mkdir(path + galaxy + '/moment_maps')
        
        cubes = glob(path + galaxy + '/*co2-1*image.pbcor*.fits')
        
        for cube in cubes:
            
            savepath = path + galaxy + '/moment_maps/' + cube.split('/')[-1].split('.fits')[0]
    
            cube_fits = fits.open(cube)[0]
    
            try:
                spec, vel = spectrum(cube_fits, galaxy, start, stop, path, savepath, extra_chans=10, non_det=non_det)            
                plot_spectrum(spec, vel, extra_chans=0, savepath=savepath)
            except:
                pass
        

if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    get_all_spectra(path)















