import smooth_and_clip
import create_moments_dev
import image_moments_dev
import create_spectrum


#main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Test_cubes/'
main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/DB_test/original/'

ifu_match = False

targets = ['KGAS7', 'KGAS74', 'KGAS86', 'KGAS215', 'KGAS291', 'KGAS300', 'KGAS435']

if __name__ == '__main__':
    #smooth_and_clip.perform_smooth_and_clip(main_directory, ifu_match)
    create_moments_dev.perform_moment_creation(main_directory, targets)
    image_moments_dev.perform_moment_imaging(main_directory, targets)
    create_spectrum.get_all_spectra(main_directory, targets)