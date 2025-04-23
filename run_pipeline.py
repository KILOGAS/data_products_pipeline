import matplotlib
matplotlib.use('Agg')
#import smooth_and_clip
import KILOGAS_smooth_clip_script_IFU_matched_Dame
import create_moments_dev
import image_moments_dev
import create_spectrum
import numpy as np
import warnings; warnings.filterwarnings("ignore")


main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/DB_test/matched/'

detected = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                      delimiter=',', skip_header=1, usecols=[6], dtype=bool)
detected_id = np.genfromtxt('/mnt/ExtraSSD/ScienceProjects/KILOGAS/KGAS_chans2do_v_detected.csv', 
                      delimiter=',', skip_header=1, usecols=[0], dtype=int)

ifu_match = False

targets = ['KGAS74', 'KGAS75', 'KGAS76', 'KGAS77', 'KGAS78', 'KGAS79', 'KGAS435']

if __name__ == '__main__':
    
    for target in targets:
        
        print(target)
        
        detection = detected[detected_id == int(target.split('KGAS')[1])][0]

        if detection:
            #smooth_and_clip.perform_smooth_and_clip(main_directory, ifu_match)
            KILOGAS_smooth_clip_script_IFU_matched_Dame.perform_smooth_and_clip(main_directory)
            create_moments_dev.perform_moment_creation(main_directory, targets)
            image_moments_dev.perform_moment_imaging(main_directory, [target])
            
        create_spectrum.get_all_spectra(main_directory, targets)