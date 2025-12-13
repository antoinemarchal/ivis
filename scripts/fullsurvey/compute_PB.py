# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
from astropy.constants import c
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
import glob
from reproject import reproject_interp

from ivis.io import DataProcessor
from ivis.logger import logger
from ivis.models import ClassicIViS3D
from ivis.imager import Imager3D
from ivis.types import VisIData
from ivis.readers import CasacoreReader

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/totoro/anmarchal/data/gaskap/fullsurvey/untar/merge/"
    path_beams = "/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/merge/"
    path_sd = "./" #path single-dish data - dummy here
    pathout = "/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/" 
    
    #create data processor
    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)
    
    #Define target header
    # cfield = SkyCoord(ra="5h36m41s", dec="-67d58m44s", frame="icrs")
    cfield = SkyCoord(ra="5h05m43.8s", dec="-70d05m48.5s", frame="icrs")
    # target_header, w = data_processor.make_imaging_header(cfield, fov_deg=12, pix_arcsec=4)
    target_header, w = data_processor.make_imaging_header(cfield, fov_deg=15)
    shape_out = (target_header["NAXIS2"], target_header["NAXIS1"])
    
    #mean pb                                                                                             
    filenames = sorted(glob.glob(path_beams + "*.fits"))
    n_beams = len(filenames)
    
    # --- full-resolution target WCS/grid ---
    w_full = w               # assuming you defined this already from target_header
    shape_full = shape_out   # (NAXIS2, NAXIS1) at full resolution
    hdr_full = w_full.to_header()
    
    # --- define coarse grid (downsampled by factor) ---
    factor = 15
    ny_full, nx_full = shape_full
    ny_c = ny_full // factor
    nx_c = nx_full // factor
    shape_coarse = (ny_c, nx_c)
    
    hdr_coarse = hdr_full.copy()
    hdr_coarse['NAXIS1'] = nx_c
    hdr_coarse['NAXIS2'] = ny_c
    
    # scale pixel size
    if 'CDELT1' in hdr_coarse:
        hdr_coarse['CDELT1'] *= factor
    if 'CDELT2' in hdr_coarse:
        hdr_coarse['CDELT2'] *= factor
        
    # scale reference pixel so the reference world coord stays roughly at same sky position
    if 'CRPIX1' in hdr_coarse:
        hdr_coarse['CRPIX1'] /= factor
    if 'CRPIX2' in hdr_coarse:
        hdr_coarse['CRPIX2'] /= factor

    # --- running mean on coarse grid ---
    pb_mean_coarse = None
    
    for i, fname in enumerate(tqdm(filenames)):
        # open beam
        hdu_pb = fits.open(fname)
        hdr_pb = hdu_pb[0].header
        pb2 = hdu_pb[0].data
        hdu_pb.close()

        # clean NaNs
        pb2 = np.nan_to_num(pb2, nan=0.0)
        
        # beam WCS
        w_pb = wcs.WCS(hdr_pb).celestial
        
        # reproject to coarse target grid
        pb2_coarse, footprint = reproject_interp(
            (pb2, w_pb.to_header()),
            hdr_coarse,
            shape_coarse
        )
        
        pb2_coarse = np.nan_to_num(pb2_coarse, nan=0.0)
        
        # init running mean
        if pb_mean_coarse is None:
            pb_mean_coarse = np.zeros_like(pb2_coarse, dtype=np.float32)
            
        # running mean on coarse grid
        pb_mean_coarse += (pb2_coarse - pb_mean_coarse) / (i + 1)
        
    # --- reproject coarse mean back to full resolution ---
    pb_mean_full, footprint = reproject_interp(
        (pb_mean_coarse, hdr_coarse),
        hdr_full,
        shape_full
    )

    # normalise & mask on full-res grid
    pb_mean_full /= np.nanmax(pb_mean_full)
    mask = np.where(pb_mean_full > 0.05, 1, np.nan)

    #Write output array on disk
    fits.writeto(pathout + "mask_7arcsec.fits", 
                 pb_mean_full, target_header, overwrite=True)

