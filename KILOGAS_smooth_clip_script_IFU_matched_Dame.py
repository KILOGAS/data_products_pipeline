# -*- coding: utf-8 -*-
"""
@author: Blake Ledger, updated April 16, 2025

This is the main script which performs the smooth and clip on
the ALMA data cubes, IFU-matched and native_resolution.

Some additional updates and changes, including implementing
the Dame+ 2011 smooth+clip method, by Tim Davis on April 17, 2025.
"""

def perform_smooth_and_clip(read_path, save_path, targets, chans2do):

    ## Libraries to import.
    from spectral_cube import SpectralCube
    import astropy.units as u
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np
    
    ## Main functions which will be used for the smoothing and clipping.
    from KILOGAS_functions_smooth_clip_Dame import KILOGAS_clip
    
    
    ## read_path points to where image cubes are stored; "path_pointing_to_data"
    read_path = read_path
    
    ## save_path points to where you want to save the smooth and clipped cubes; "path_to_save"
    save_path = save_path
    
    ## target list of test galaxies; ["list of target names"]
    targets = targets
    
    ## important fits file from Tim Davis that describes the min/max channel of CO line for each galaxy
    ## clipping_channels; columns are ['KGAS_ID', 'RMS', 'minchan', 'maxchan', 'minchan_v', 'maxchan_v']
    clipping_channels = fits.open(chans2do)
    cols = clipping_channels[1].columns ## pull out column names
    tbdata = clipping_channels[1].data  ## pull out data
    
    ## create a dataframe with the table data
    df = Table(tbdata)
    
    #sun_method_params = [nchan_hi,snr_hi,nchan_lo,snr_lo,prune_by_npix,prune_by_fracbeam,
    #                     expand_by_npix,expand_by_fracbeam,expand_by_nchan]
    sun_method_params = [3,3,2,2, None, 0.1, None, None, None]
    #dame_method_params = [S/N clip, beam expand factor, channel expand factor, prune_by_fracbeam]
    dame_method_params=[4,1.5,4,(2/np.pi)]
    
    verbose, save = True, True

    method='dame'
    kms='10'
    
    for i,galaxy in enumerate(targets):
        
        verbose = True
        
        ## get min/max channels of current target
        kgasid = int(galaxy[4:])
        idx = np.where(df['KGAS_ID']==kgasid)
        
        minchan, maxchan  = df['minchan'][idx][0],df['maxchan'][idx][0]
        vminchan,vmaxchan=df['minchan_v'][idx][0],df['maxchan_v'][idx][0]
        
        if verbose:
            print("Current target:", galaxy)
            print('KGASID:', df['KGAS_ID'][idx][0], ', minchan:', minchan, ', maxchan:', maxchan)
            
            verbose = False
    
        try:
            path_pbcorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.image.pbcor.ifumatched.fits"
            path_uncorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.image.ifumatched.fits"
            fits.open(path_pbcorr)
        except:
            try:
                path_pbcorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.image.pbcor.ifumatched.fits"
                path_uncorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.image.ifumatched.fits"
                fits.open(path_pbcorr)
            except:
                try:
                    path_pbcorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.contsub.image.pbcor.ifumatched.fits"
                    path_uncorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_12m.contsub.image.ifumatched.fits"
                    fits.open(path_pbcorr)
                except:
                    path_pbcorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.contsub.image.pbcor.ifumatched.fits"
                    path_uncorr = read_path+galaxy+"/"+galaxy+"_co2-1_10.0kmps_7m+12m.contsub.image.ifumatched.fits"
                
    
        ## do the smooth and clip
        kgasclip = KILOGAS_clip(galaxy, path_pbcorr, path_uncorr, minchan, maxchan,
                            verbose, save, read_path, save_path,
                              sun_method_params=sun_method_params,dame_method_params=dame_method_params)
        clipped_emiscube, clipped_noisecube = kgasclip.do_clip(method=method)
        
        
        
if __name__ == '__main__':
    path = '/mnt/ExtraSSD/ScienceProjects/KILOGAS/Code_Blake/'
    perform_smooth_and_clip(path)
        
        
        
        
        
        
        
        
        
        
        
        
    