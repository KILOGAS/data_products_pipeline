## Main pipeline updates from Blake Ledger, June 3 2025

# - removed unused smooth_and_clip.py libraries from import list to clean up code and avoid version confusion; also removed calls to the unused smooth_and_clip functions
# - renamed library "KILOGAS_smooth_clip_script_IFU_matched_Dame.py" to "smooth_and_clip.py" for better naming convention now that we are towards a finalized version of the pipeline; also renamed the call to this library in the script
# - renamed library "KILOGAS_functions_smooth_clip_IFU_matched_Dame.py" to "smooth_and_clip_functions.py" for better naming convention now that we are towards a finalized version of the pipeline; also renamed the call to this library in the "smooth_and_clip.py" script


## Smoothing and clipping script and functions updates

# - updated the Dame parameters in the "smooth_and_clip.py" script based on testing by Blake done on May 30, 2025, to dame_method_params=[3,2,4,(2/np.pi)]
# - cleaned up some of the Sun method code / function calls, as they are less necessary
# - in the "smooth_and_clip.py" script, updated the filenames to include "kms" as a variable input
# - added an elif for the "dame" method to be explicit along with the "sun" method in the smooth_and_clip_functions.py
# - when saving the binary mask cube, changed so that the correct velocity range is saved in the header
# - adjusted order of functions based on priority (e.g. moved all Sun method functions to end of script)
# - updated saving keywords in header to include the Dame method as the main option

## Notes / questions for Nikki

# - check saving path on clipped_hdu.writeto portion of the code


import matplotlib
matplotlib.use('Agg')
import KILOGAS_smooth_clip_script_IFU_matched_Dame #rename to "smooth_and_clip"
import create_moments_dev
import image_moments_dev
import create_spectrum
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from glob import glob
import os


if __name__ == '__main__':
    
    #ifu_match = False
    local = False

    #targets = [d.split('/')[-1] for d in glob(main_directory + '*') if os.path.isdir(d) and os.listdir(d)]

    if local:
        main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Full_sample_22_April/matched/'
        save_path = main_directory
        detected = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[6], dtype=bool)
        target_id = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[0], dtype=int)
        chans2do = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do.fits'
        glob_cat = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/KILOGAS_global_catalog_FWHM.fits'
    
    else:
        main_directory = '/arc/projects/KILOGAS/cubes/v1.0/matched/'
        save_path = '/arc/projects/KILOGAS/products/v0.1/matched/by_galaxy/'
        chans2do = 'KGAS_chans2do.fits'
        detected = np.genfromtxt('KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[6], dtype=bool)
        target_id = np.genfromtxt('KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[0], dtype=int)
        glob_cat = 'KILOGAS_global_catalog_FWHM.fits'

    detections = ['KGAS' + str(target) for target, flag in zip(target_id, detected) if flag]
    targets = ['KGAS' + str(target) for target in target_id]

    targets = [targets[20]]
    detections = [detections[20]]


    
    #rename to smooth_and_clip
    #KILOGAS_smooth_clip_script_IFU_matched_Dame.perform_smooth_and_clip(read_path=main_directory, 
    #                                                                    save_path=save_path, targets=detections, chans2do=chans2do)
    #create_moments_dev.perform_moment_creation(path=save_path, targets=detections, glob_cat=glob_cat)
    #image_moments_dev.perform_moment_imaging(glob_path=save_path, targets=detections)
    create_spectrum.get_all_spectra(read_path=main_directory, save_path=save_path, targets=targets, 
                                    target_id=target_id, detected=detected, chans2do=chans2do, glob_cat=glob_cat)















