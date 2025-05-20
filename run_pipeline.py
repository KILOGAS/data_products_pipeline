import matplotlib
matplotlib.use('Agg')
#import smooth_and_clip
import KILOGAS_smooth_clip_script_IFU_matched_Dame
import create_moments_dev
import image_moments_dev
import create_spectrum
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from glob import glob
import os


if __name__ == '__main__':
    
    #ifu_match = False
    
    main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Full_sample_22_April/matched/'
    
    #targets = [d.split('/')[-1] for d in glob(main_directory + '*') if os.path.isdir(d) and os.listdir(d)]

    detected = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[6], dtype=bool)
    target_id = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                          delimiter=',', skip_header=1, usecols=[0], dtype=int)

    detections = ['KGAS' + str(target) for target, flag in zip(target_id, detected) if flag]
    targets = ['KGAS' + str(target) for target in target_id]
    
    #smooth_and_clip.perform_smooth_and_clip(main_directory, ifu_match)
    
    #KILOGAS_smooth_clip_script_IFU_matched_Dame.perform_smooth_and_clip(main_directory, detections)
    #create_moments_dev.perform_moment_creation(main_directory, detections)
    #image_moments_dev.perform_moment_imaging(main_directory, detections)
    create_spectrum.get_all_spectra(main_directory, targets)