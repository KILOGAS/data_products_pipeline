import matplotlib
matplotlib.use('Agg')
import smooth_and_clip
#import KILOGAS_smooth_clip_script_IFU_matched_Dame
import create_moments_dev
import image_moments_dev
import create_spectrum
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from glob import glob
import os
import shutil


if __name__ == '__main__':
    
    #ifu_match = False
    local = False
    clear_save_directory = False
    version = 0.3
    spec_res = 10
    pb_thresh = 40
    prune_by_npix = 100

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
        save_path = '/arc/projects/KILOGAS/products/v' + str(version) + '/matched/'
        chans2do = 'KGAS_chans2do.fits'
        detected = np.genfromtxt('KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[6], dtype=bool)
        target_id = np.genfromtxt('KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[0], dtype=int)
        glob_cat = 'KILOGAS_global_catalog_FWHM.fits'

    detections = ['KGAS' + str(target) for target, flag in zip(target_id, detected) if flag]
    non_detections = ['KGAS' + str(target) for target, flag in zip(target_id, detected) if not flag]
    targets = ['KGAS' + str(target) for target in target_id]

    targets = ['KGAS55', 'KGAS61', 'KGAS84', 'KGAS107', 'KGAS108', 'KGAS112', 'KGAS146', 'KGAS325']
    detections = ['KGAS55', 'KGAS61', 'KGAS84', 'KGAS107', 'KGAS108', 'KGAS112', 'KGAS146', 'KGAS325']

    if clear_save_directory:
        for galaxy in non_detections:
            directory = os.path.join(save_path, galaxy)    
            for root, dirs, files in os.walk(directory, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    os.unlink(file_path)
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    shutil.rmtree(dir_path)
    
    smooth_and_clip.perform_smooth_and_clip(read_path=main_directory, save_path=save_path, 
                                            targets=detections, chans2do=chans2do, kms=spec_res, pb_thresh=pb_thresh, prune_by_npix=prune_by_npix)
    create_moments_dev.perform_moment_creation(path=save_path, data_path=main_directory, targets=detections, glob_cat=glob_cat, spec_res=spec_res)
    image_moments_dev.perform_moment_imaging(glob_path=save_path, targets=detections, spec_res=spec_res)
    #create_spectrum.get_all_spectra(read_path=main_directory, save_path=save_path, targets=targets, 
    #                                target_id=target_id, detected=detected, chans2do=chans2do, glob_cat=glob_cat, spec_res=spec_res)

















