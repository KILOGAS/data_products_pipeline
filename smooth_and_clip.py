# -*- coding: utf-8 -*-
"""
@author: Blake Ledger, created Feb 7 2025

Code written by Blake Ledger (UVic) to perform the Smooth+Clip analysis for the
sample of KILOGAS galaxies. This is part of the Data Products Working Group
workflow, where Tim Davis (Cardiff) is working on line identification, Nikki
Zabel (U Cape Town) is working on moment map creation, and Hsi-An Pan (Tamkang U.)
will do the convolution and grid matching to the IFU data products.

Smoothing expands the mask a little bit, so that the clipping is not too harsh and
some of the lower S/N emission around the edges isn't missed. In this code, there is
the slightly more advanced way, the "Sun" way, which refers to an algorithm that was
translated to Python code by Jiayi Sun a couple of years ago.

Clipping is making a mask out of the smoothed cube (setting everything below a
certain S/N threshold to 0) and applying it to the original (unsmoothed) cube, so
that we get rid of all the noise and get cleaner data products

This is the main smooth_and_clip.py Python file which should be run to perform
the relative steps. The script requires a path pointing to the location of the
image cubes, a list of targets, and an input file from Tim/the line identification
script which has the "start" and "stop" parameters indicating the first and last
channels around the CO line.
"""

def perform_smooth_and_clip(path, ifu_match=False):
    from KILOGAS_functions import KILOGAS_clip
    import os

    targets = ['KGAS15', 'KGAS16', 'KGAS58']
    targets = ['KGAS26', 'KGAS28', 'KGAS55', 'KGAS58']

    clipping_chans = {'KGAS15': [168,188], 'KGAS16': [60,90], 'KGAS58': [95,150], 
                      'KGAS26': [15,50], 'KGAS28': [155,205], 'KGAS55': [40,75]}
    save_files = True

    if ifu_match:
        nchan_hi = 3
        snr_hi = 3
        nchan_lo = 2
        snr_lo = 2.5
        prune_by_npix = None
        prune_by_fracbeam = None
        expand_by_npix = None
        expand_by_fracbeam = None
        expand_by_nchan = None        
    else:
        nchan_hi = 3
        snr_hi = 3
        nchan_lo = 2
        snr_lo = 1.5
        prune_by_npix = None
        prune_by_fracbeam = 0.15
        expand_by_npix = None
        expand_by_fracbeam = None
        expand_by_nchan = None

    #Use Jiayi's method to get the signal in the cube.
    #Default nchan_hi=3, snr_hi=3.5, nchan_lo=2, snr_lo=2
    #Sun method expansions to remove more noisy features smaller than the beam.
    #Default prune_by_npix=None, prune_by_fracbeam=1., expand_by_npix=None, expand_by_fracbeam=0., expand_by_nchan=2

    sun_method_params = [nchan_hi, snr_hi, nchan_lo, snr_lo, prune_by_npix, 
                         prune_by_fracbeam, expand_by_npix, expand_by_fracbeam,
                         expand_by_nchan]

    #path = "path_where_image_cubes_are_stored"
    #targets = ["list of target names"]
    #clipping_chans = TBD on format, but first/last channels of CO line


    for galaxy in targets:
        print("Current target:", galaxy)
            
        if ifu_match:
            path_pbcorr = path+galaxy+"_co2-1_7m+12m.image.pbcor.smoothed.ifumatched.fits"
            path_uncorr = path+galaxy+"_co2-1_7m+12m.image.smoothed.ifumatched.fits"
            savepath = path+galaxy+"/"+galaxy+'_ifumatch_test.fits'
        else:
            path_pbcorr = path+galaxy+"_co2-1_7m+12m.image.pbcor.fits"
            path_uncorr = path+galaxy+"_co2-1_7m+12m.image.fits"
            savepath = path+galaxy+"/"+galaxy+'_test.fits'
        
        start = clipping_chans[galaxy][0]
        stop = clipping_chans[galaxy][1]
        
        #print(sun_method_params)
        
        clip_emiscube, clipped_noisecube = KILOGAS_clip(galaxy, path_pbcorr, path_uncorr, start, stop, sun_method_params).do_clip()

        if save_files:
            if not os.path.exists(path + galaxy):
                os.mkdir(path + galaxy)
            clip_emiscube.writeto(savepath, overwrite=True)
            #clip_emiscube.writeto(path+galaxy+"/"+galaxy+'_co2-1_7m+12m.image.pbcor.expanded_pruned_subcube.fits', overwrite=True)
        print("CLIP CUBE SAVED FOR", galaxy)
        print()
        

if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    perform_smooth_and_clip(path)

