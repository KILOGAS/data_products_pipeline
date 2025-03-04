import smooth_and_clip
import create_moments_dev
import image_moments_dev


#main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Test_cubes/'
main_directory = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/IFU_matched_cubes/'

ifu_match = True

if __name__ == '__main__':
    smooth_and_clip.perform_smooth_and_clip(main_directory, ifu_match)
    create_moments_dev.perform_moment_creation(main_directory)
    image_moments_dev.perform_moment_imaging(main_directory)