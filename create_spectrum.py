from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from photutils.aperture import CircularAperture
from glob import glob
import os
from create_moments_dev import create_vel_array
import pandas as pd


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def calc_beam_area(bmaj,bmin,cellsize=1):
    return (np.pi*(bmaj/cellsize)*(bmin/cellsize))/(4*np.log(2))

def brightness_temp_to_flux_dens(T, bmaj, bmin, nu=230.538):
    return T * nu ** 2 * bmaj * bmin / 1.222e3

def make_spectrum(cube, galaxy, start, stop, path, glob_cat, extra_chans=10, non_det=False, spec_res=10):

    _, _, vel_array_full, _ = create_vel_array(galaxy, cube, spec_res=spec_res)
    
    bmaj = cube.header['BMAJ'] * 3600
    bmin = cube.header['BMIN'] * 3600

    if spec_res == 10:
        beam_area = calc_beam_area(bmaj, bmin, cellsize=0.1)
    elif spec_res == 30:
        beam_area = calc_beam_area(bmaj, bmin, cellsize=0.5)
    
    if non_det:
        
        table = fits.open(glob_cat)[1]
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
        
        spectrum = np.nanmean(masked_data, axis=(1, 2))

        # Add the spectrum in mJy
        masked_data_mJyb = brightness_temp_to_flux_dens(masked_data, bmaj, bmin)
        spectrum_mJy = np.nansum(masked_data_mJyb, axis=(1, 2)) / beam_area
        
        #rms = np.nanstd(spectrum)
        #fwhm = 100  # km/s
        #sigma = fwhm / np.sqrt(8 * np.log(2))
        #x0 = vel_array_full[int(len(vel_array_full) / 2)]
        #gaussian = gauss(vel_array_full, 3*rms, x0, sigma)
        
    else:
        clip_mask = fits.open(path + 'by_galaxy/' + galaxy + '/' + str(spec_res) + 'kms/' + galaxy + '_mask_cube.fits')[0]
        mask = np.sum(clip_mask.data, axis=0)
        mask = mask.astype(float)
        mask[mask > 0] = 1
        
        mask3d = np.tile(mask, (cube.shape[0], 1, 1))
        masked_data = mask3d * cube.data
        masked_data[masked_data == 0] = np.nan
        
        spectrum = np.nanmean(masked_data, axis=(1, 2))
        
        # Add the spectrum in mJy
        masked_data_mJyb = brightness_temp_to_flux_dens(masked_data, bmaj, bmin)
        spectrum_mJy = np.nansum(masked_data_mJyb, axis=(1, 2)) / beam_area
        
    if start - extra_chans < 0:
        spectrum_velocities = vel_array_full[0:stop + extra_chans]
        spectrum = spectrum[0:stop + extra_chans]
        spectrum_mJy = spectrum_mJy[0:stop + extra_chans]
        #spectrum_vel_offset = spectrum_velocities - sysvel + self.galaxy.sysvel_offset
    elif stop + extra_chans > len(vel_array_full):
        spectrum_velocities = vel_array_full[start:]
        spectrum = spectrum[start:]
        spectrum_mJy = spectrum_mJy[start:]
    else:
        spectrum_velocities = vel_array_full[start - extra_chans:stop + extra_chans]
        spectrum = spectrum[start - extra_chans:stop + extra_chans]
        spectrum_mJy = spectrum_mJy[start - extra_chans:stop + extra_chans]
        #spectrum_vel_offset = spectrum_velocities - sysvel + self.galaxy.sysvel_offset

    #rest_freq = 230538000000
    #spectrum_frequencies = rest_frequency * (1 - spectrum_velocities / 299792.458) / 1e9
    
    csv_header = 'Spectrum (K), Spectrum (mJy), Velocity (km/s)'

    if spec_res == 10:
        np.savetxt(path + 'by_galaxy/' + galaxy + '/10kms/' + galaxy + '_spectrum.csv',
                   np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
                   delimiter=',', header=csv_header)
        np.savetxt(path + 'by_product/spectrum/10kms/' + galaxy + '_spectrum.csv',
           np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
           delimiter=',', header=csv_header)
    
    elif spec_res == 30:
        np.savetxt(path + 'by_galaxy/' + galaxy + '/30kms/' + galaxy + '_spectrum.csv',
                   np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
                   delimiter=',', header=csv_header)
        np.savetxt(path + 'by_product/spectrum/30kms/' + galaxy + '_spectrum.csv',
           np.column_stack((spectrum, spectrum_mJy, spectrum_velocities)),
           delimiter=',', header=csv_header)
        
    return spectrum, spectrum_mJy, spectrum_velocities


def plot_spectrum(galaxy, spectrum, spectrum_mJy, velocity, extra_chans=0, x_axis='velocity', 
             useclipped=False, savepath=None, spec_res=10):

    fig, ax = plt.subplots(figsize=(7, 7))

    if x_axis == 'velocity':
        ax.plot(velocity, spectrum, color='k', drawstyle='steps')
        #x = np.arange(np.amin(velocity) - 100, np.amax(velocity) + 100, 1)
        ax.set_xlim(velocity[0] - extra_chans, velocity[len(velocity) - 1] + extra_chans)
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

    ax2 = ax.twinx()
    ax2.plot(velocity, spectrum_mJy, color='k', drawstyle='steps')
    ax2.set_ylabel('Flux Density (mJy)', color='k')

    # Line through zero
    #zeroline = np.zeros(len(x))
    #plt.plot(x, zeroline, linestyle=':', c='r', linewidth=1)

    ax.set_ylabel('Brightness temperature [K]')

    plt.tight_layout()

    if savepath:
        if spec_res == 10:
            plt.savefig(savepath + 'by_galaxy/' + galaxy + '/10kms/' + galaxy + '_spectrum.png', bbox_inches='tight')
            plt.savefig(savepath + 'by_galaxy/' + galaxy + '/10kms/' + galaxy + '_spectrum.pdf', bbox_inches='tight')
            plt.savefig(savepath + 'by_product/spectrum/10kms/' + galaxy + '_spectrum.png', bbox_inches='tight')
            plt.savefig(savepath + 'by_product/spectrum/10kms/' + galaxy + '_spectrum.pdf', bbox_inches='tight')
        elif spec_res == 30:
            plt.savefig(savepath + 'by_galaxy/' + galaxy + '/30kms/' + galaxy + '_spectrum.png', bbox_inches='tight')
            plt.savefig(savepath + 'by_galaxy/' + galaxy + '/30kms/' + galaxy + '_spectrum.pdf', bbox_inches='tight')
            plt.savefig(savepath + 'by_product/spectrum/30kms/' + galaxy + '_spectrum.png', bbox_inches='tight')
            plt.savefig(savepath + 'by_product/spectrum/30kms/' + galaxy + '_spectrum.pdf', bbox_inches='tight')


def get_all_spectra(read_path, save_path, targets, target_id, detected, chans2do, glob_cat, spec_res=10):
    
    files = glob(read_path + '**/*co2-1*image.pbcor*.fits')
    
    galaxies = list(set([f.split('/')[8].split('_')[0] for f in files]))
    
    #clipping_table = fits.open(chans2do)[1]
    clipping_table = pd.read_csv(chans2do)
    KGAS_ID = np.array(clipping_table['KGAS_ID'])
    #minchan = clipping_table.data['minchan']
    #maxchan = clipping_table.data['maxchan']
    minchan_v = np.array(clipping_table['minchan_v'])
    maxchan_v = np.array(clipping_table['maxchan_v'])
    
    #clipping_chans = {'KGAS' + id.astype(str): [min, max] for id, min, max in zip(KGAS_ID, minchan, maxchan)}
    clipping_vels = {'KGAS' + id.astype(str): [min, max] for id, min, max in zip(KGAS_ID, minchan_v, maxchan_v)}
    
    detected_id = target_id
    
    for galaxy in galaxies:
        
        if not galaxy in targets:
            continue

        print("Creating spectrum for " + galaxy + ". \n")
        
        non_det = ~detected[detected_id == int(galaxy.split('KGAS')[1])]

        if spec_res == 10:
            if not os.path.exists(save_path + 'by_galaxy/' + galaxy):
                os.mkdir(save_path + 'by_galaxy/' + galaxy)
            if not os.path.exists(save_path + 'by_product/spectrum'):
                os.mkdir(save_path + 'by_product/spectrum')            
            if not os.path.exists(save_path + 'by_galaxy/' + galaxy + '/10kms'):
                os.mkdir(save_path + 'by_galaxy/' + galaxy + '/10kms')
            if not os.path.exists(save_path + 'by_product/spectrum/10kms'):
                os.mkdir(save_path + 'by_product/spectrum/10kms')
        elif spec_res == 30:  
            if not os.path.exists(save_path + 'by_galaxy/' + galaxy):
                os.mkdir(save_path + 'by_galaxy/' + galaxy)
            if not os.path.exists(save_path + 'by_product/spectrum'):
                os.mkdir(save_path + 'by_product/spectrum')
            if not os.path.exists(save_path + 'by_galaxy/' + galaxy + '/30kms'):
                os.mkdir(save_path + 'by_galaxy/' + galaxy + '/30kms')
            if not os.path.exists(save_path + 'by_product/spectrum/30kms'):
                os.mkdir(save_path + 'by_product/spectrum/30kms')

        if spec_res == 10:
            cube = glob(read_path + galaxy + '/*co2-1_10.0kmps*image.pbcor*.fits')[0]
        elif spec_res == 30:
            cube = glob(read_path + galaxy + '/*co2-1_30.0kmps*image.pbcor*.fits')[0]
            
        cube_fits = fits.open(cube)[0]

        #if spec_res == 10:
        #    start = clipping_chans[galaxy][0]
        #    stop = clipping_chans[galaxy][1]
        #else:
        start_v = clipping_vels[galaxy][0]
        stop_v = clipping_vels[galaxy][1]
        
        _, _, vel_array, _ = create_vel_array(galaxy, cube_fits, spec_res=spec_res)
        start = np.argmin(abs(vel_array - start_v))
        stop = np.argmin(abs(vel_array - stop_v))

        try:
            spec, spec_mJy, vel = make_spectrum(cube_fits, galaxy, start, stop, save_path, glob_cat=glob_cat, 
                                  extra_chans=10, non_det=non_det, spec_res=spec_res)   
            plot_spectrum(galaxy, spec, spec_mJy, vel, extra_chans=0, savepath=save_path, spec_res=spec_res)
        except:
            pass

    print("Done.")
        

if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    get_all_spectra(path)
















