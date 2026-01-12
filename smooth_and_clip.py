# -*- coding: utf-8 -*-
"""
@author: Blake Ledger, updated June 3, 2025

This is the main script which performs the smooth and clip on
the ALMA data cubes, IFU-matched and native_resolution.

Some additional updates and changes, including removing most of the infrastructure for
Sun method testing as we are focused on the Dame+ 2011 smooth+clip method, which was implemented in the code by Tim Davis on April 17, 2025 and updated by Blake + Scott June 3, 2025.
"""


def perform_smooth_and_clip(
    read_path,
    save_path,
    targets,
    chans2do,
    kms=10,
    pb_thresh=40,
    prune_by_npix=None,
    ifu_match=True,
):
    ## Libraries to import.
    from astropy.io import fits
    import numpy as np
    from create_moments import create_vel_array
    import pandas as pd

    ## Main functions which will be used for the smoothing and clipping.
    from smooth_and_clip_functions import KILOGAS_clip

    ## important fits file from Tim Davis that describes the min/max channel of CO line for each galaxy
    ## clipping_channels; columns are ['KGAS_ID', 'RMS', 'minchan', 'maxchan', 'minchan_v', 'maxchan_v']
    df = pd.read_csv(chans2do)
    # print(df.head())

    # NOTE: We have abandoned the Sun method but leaving the below lines in
    # just in case.
    # sun_method_params = [nchan_hi, snr_hi, nchan_lo, snr_lo, prune_by_npix, prune_by_fracbeam,
    #                     expand_by_npix, expand_by_fracbeam, expand_by_nchan]
    # sun_method_params = [3, 3, 2, 2, None, 0.1, None, None, None]

    # dame_method_params = [S/N clip, beam expand factor, channel expand factor, prune_by_fracbeam]
    # Initial parameters implemented by Tim, dame_method_params=[4, 1.5, 4, (2/np.pi)]
    dame_method_params = [
        3,
        2,
        4,
        (2 / np.pi),
    ]  # New parameters tested by Blake for detecting more faint emission

    # These galaxies should not be trimmed to 40% of the primary beam response
    pb_exceptions = [
        28,
        37,
        61,
        66,
        70,
        75,
        82,
        84,
        154,
        182,
        188,
        219,
        223,
        238,
        239,
        241,
        334,
        340,
        426,
    ]

    verbose, save = False, True

    # Method is either "dame" or "sun". We have abandoned the latter but keeping
    # this parameter just in case.
    method = "dame"

    for i, galaxy in enumerate(targets):
        ## get min/max channels of current target
        kgasid = int(galaxy[4:])
        idx = np.where(df["KGAS_ID"] == kgasid)[0]

        if verbose:
            print("Current target:", galaxy)
            # print('KGASID:', df['KGAS_ID'][idx][0], ', minchan:', minchan, ', maxchan:', maxchan)

            verbose = False

        # We don't know for eacht galaxy if it includes 7m observations or what the
        # filename is, so we try importing things until we find the cube.
        if ifu_match:
            try:
                path_pbcorr = (
                    read_path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(kms)
                    + ".0kmps_12m.image.pbcor.ifumatched.fits"
                )
                path_uncorr = (
                    read_path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(kms)
                    + ".0kmps_12m.image.ifumatched.fits"
                )
                cube = fits.open(path_pbcorr)[0]
            except:
                try:
                    path_pbcorr = (
                        read_path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(kms)
                        + ".0kmps_7m+12m.image.pbcor.ifumatched.fits"
                    )
                    path_uncorr = (
                        read_path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(kms)
                        + ".0kmps_7m+12m.image.ifumatched.fits"
                    )
                    cube = fits.open(path_pbcorr)[0]
                except:
                    try:
                        path_pbcorr = (
                            read_path
                            + galaxy
                            + "/"
                            + galaxy
                            + "_co2-1_"
                            + str(kms)
                            + ".0kmps_12m.contsub.image.pbcor.ifumatched.fits"
                        )
                        path_uncorr = (
                            read_path
                            + galaxy
                            + "/"
                            + galaxy
                            + "_co2-1_"
                            + str(kms)
                            + ".0kmps_12m.contsub.image.ifumatched.fits"
                        )
                        cube = fits.open(path_pbcorr)[0]
                    except:
                        try:
                            path_pbcorr = (
                                read_path
                                + galaxy
                                + "/"
                                + galaxy
                                + "_co2-1_"
                                + str(kms)
                                + ".0kmps_7m+12m.contsub.image.pbcor.ifumatched.fits"
                            )
                            path_uncorr = (
                                read_path
                                + galaxy
                                + "/"
                                + galaxy
                                + "_co2-1_"
                                + str(kms)
                                + ".0kmps_7m+12m.contsub.image.ifumatched.fits"
                            )
                            cube = fits.open(path_pbcorr)[0]
                        except:
                            print("Cube not available for " + galaxy + ". \n")
                            continue
        else:
            try:
                path_pbcorr = (
                    read_path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(kms)
                    + ".0kmps_12m.image.pbcor.fits"
                )
                path_uncorr = (
                    read_path
                    + galaxy
                    + "/"
                    + galaxy
                    + "_co2-1_"
                    + str(kms)
                    + ".0kmps_12m.image.fits"
                )
                cube = fits.open(path_pbcorr)[0]
            except:
                try:
                    path_pbcorr = (
                        read_path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(kms)
                        + ".0kmps_7m+12m.image.pbcor.fits"
                    )
                    path_uncorr = (
                        read_path
                        + galaxy
                        + "/"
                        + galaxy
                        + "_co2-1_"
                        + str(kms)
                        + ".0kmps_7m+12m.image.fits"
                    )
                    cube = fits.open(path_pbcorr)[0]
                except:
                    try:
                        path_pbcorr = (
                            read_path
                            + galaxy
                            + "/"
                            + galaxy
                            + "_co2-1_"
                            + str(kms)
                            + ".0kmps_12m.contsub.image.pbcor.fits"
                        )
                        path_uncorr = (
                            read_path
                            + galaxy
                            + "/"
                            + galaxy
                            + "_co2-1_"
                            + str(kms)
                            + ".0kmps_12m.contsub.image.fits"
                        )
                        cube = fits.open(path_pbcorr)[0]
                    except:
                        try:
                            path_pbcorr = (
                                read_path
                                + galaxy
                                + "/"
                                + galaxy
                                + "_co2-1_"
                                + str(kms)
                                + ".0kmps_7m+12m.contsub.image.pbcor.fits"
                            )
                            path_uncorr = (
                                read_path
                                + galaxy
                                + "/"
                                + galaxy
                                + "_co2-1_"
                                + str(kms)
                                + ".0kmps_7m+12m.contsub.image.fits"
                            )
                            cube = fits.open(path_pbcorr)[0]
                        except:
                            print("Cube not available for " + galaxy + ". \n")
                            continue

        # We need the non-PB-corrected cube for clipping, but it is not available
        # for each galaxy. Try to open it and skip the galaxy if it doesn't exist.
        try:
            fits.open(path_uncorr)
        except:
            print("Uncorrected cube not available for " + galaxy + ". \n")
            continue

        # Cubes are set to be masked outside of 40% of the PB response, but exceptions
        # are made for some galaxies, for which the threshold is set to 0% here.
        if kgasid in pb_exceptions:
            pb_clip_thresh = 0
        else:
            pb_clip_thresh = pb_thresh

        # Calculate the start/stop channels corresponding to
        # the start/stop velocities in the "chans2do" table.
        vminchan = df["minchan_v"].iloc[idx].iloc[0]
        vmaxchan = df["maxchan_v"].iloc[idx].iloc[0]

        # Create the velocity array corresponding to the cube and use it to find
        # the channel that is closest to the start and stop velocities.
        vel_array, _, _ = create_vel_array(galaxy, cube)
        minchan = np.argmin(abs(vel_array - vminchan))
        maxchan = np.argmin(abs(vel_array - vmaxchan))

        ## do the smooth and clip
        kgasclip = KILOGAS_clip(
            galaxy,
            path_pbcorr,
            path_uncorr,
            minchan,
            maxchan,
            verbose,
            save,
            read_path,
            save_path,
            dame_method_params=dame_method_params,
            spec_res=kms,
            pb_thresh=pb_clip_thresh,
            prune_by_npix=prune_by_npix,
        )
        clipped_emiscube, clipped_noisecube = kgasclip.do_clip(method=method)


if __name__ == "__main__":
    # NOTE: all hardcoded for now as it isn't really used, but can improve
    # to take user input.

    version = 1.1
    ifu_matched = True
    spec_res = 10
    pb_thresh = 40

    targets = ["KGAS107"]

    if ifu_matched:
        read_path = "/arc/projects/KILOGAS/cubes/v1.0/matched/"
        save_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/matched/"
    else:
        read_path = "/arc/projects/KILOGAS/cubes/v1.0/nyquist/"
        save_path = "/arc/projects/KILOGAS/products/v" + str(version) + "/original/"

    chans2do = "KGAS_chans2do_v_optical_Sept25.csv"

    perform_smooth_and_clip(
        read_path,
        save_path,
        targets,
        chans2do,
        kms=spec_res,
        pb_thresh=pb_thresh,
        prune_by_npix=None,
        ifu_match=ifu_matched,
    )
