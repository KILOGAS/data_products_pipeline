from image_moments import CreateImages
import os
import glob
import warnings; warnings.filterwarnings("ignore")

# Settings
tosave = True
path = 'some_path'
resolution = 'native'
version = '1_2'

galaxies = glob.glob(path)

for galaxy in galaxies:

    print(galaxy)

    if resolution == 'native':
        if not os.path.exists(path + 'something_native/' + galaxy + '/'):
            os.mkdir(path + 'something_native/' + galaxy + '/')

    CreateImages(galaxy, some_settings).moment_zero()