# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
from astropy.constants import k_B, c
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
from reproject import reproject_interp
import time

from ivis.io import DataProcessor
from ivis.logger import logger
from ivis.models import ClassicIViS3D
from ivis.imager import Imager3D
from ivis.types import VisIData
from ivis.readers import CasacoreReader
from spectral_cube import SpectralCube

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w

def K_to_jy_arcsec2(T_K, nu_hz):
    """
    Convert brightness temperature [K] -> Jy/arcsec^2
    using Rayleigh–Jeans law (surface brightness).
    """
    T = np.asarray(T_K) * u.K
    nu = np.asarray(nu_hz) * u.Hz

    # Broadcast frequency if needed (for cubes)
    if nu.ndim == 1 and T.ndim == 3:
        nu = nu[:, None, None]

    # Specific intensity per steradian
    I_nu = (2 * k_B * nu**2 / c**2 * T) / u.sr

    # Convert sr -> arcsec^2
    I_jy_arcsec2 = I_nu.to(
        u.Jy / u.arcsec**2,
        equivalencies=u.dimensionless_angles()
    )

    return I_jy_arcsec2.value


if __name__ == '__main__':    
    #path data
    path_ms = "/totoro/anmarchal/data/gaskap/fullsurvey/untar/merge/merge1/"
    
    path_beams = "/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/merge/" #directory of primary beams
    path_sd = "./" #path single-dish data - dummy here
    pathout = "/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/" #path where data will be packaged and stored

    #create data processor
    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

    #Define target header
    cfield = SkyCoord(ra="5h05m43.8s", dec="-70d05m48.5s", frame="icrs")
    target_header, w = data_processor.make_imaging_header(cfield, fov_deg=15)
    shape = (target_header["NAXIS2"], target_header["NAXIS1"])

    #Open Parkes data
    path="/totoro/anmarchal/data/parkes/"
    fitsname = "GASS_HI_LMC_cube.fits"

    # Read cube
    hdu_sd = fits.open(path+fitsname)
    hdr_sd = hdu_sd[0].header
    w_sd = wcs2D(hdr_sd)
    cube_sd = SpectralCube.read(path+fitsname)
    
    # Make sure the spectral axis is in km/s (radio convention is usually right for HI)
    cube_sd = cube_sd.with_spectral_unit(u.km/u.s, velocity_convention="radio")
    
    v0 = 231.23153293 * u.km/u.s
    dv = 5.0 * u.km/u.s
    vmin, vmax = v0 - dv, v0 + dv 
    
    # Cut cube
    cube_cut = cube_sd.spectral_slab(vmin, vmax)
    
    print("Original shape:", cube_sd.shape)
    print("Cut shape     :", cube_cut.shape)
    print("v range cut   :", cube_cut.spectral_axis.min(), "→", cube_cut.spectral_axis.max())
    
    # # Optional: write to a new FITS
    # cube_cut.write(path+"GASS_HI_LMC_cube_{}_pm5kms.fits".format(v0.value), overwrite=True)

    target_velocity = v0#253.2170684#* u.km/u.s 231.23153293
    hi_slice = cube_sd.spectral_interpolate(np.array([target_velocity.value])*u.km/u.s)
    hi_slice_array = hi_slice.hdu.data[0]  # Convert the SpectralCube slice to a NumPy array
    
    #reproject on target_header
    w = wcs2D(target_header)
    target_hdr = w.to_header()
    sd_K, footprint = reproject_interp((hi_slice_array,w_sd.to_header()), target_hdr, shape_out=(shape[0],shape[1]))
    sd_K[sd_K != sd_K] = 0.        
    
    nu_Hz = 1.42040575177e9*u.Hz #HI 21cm
    sd = K_to_jy_arcsec2(sd_K,nu_Hz)

    #Write output array on disk
    fits.writeto(path + "gass_chan_765.fits", sd, target_hdr, overwrite=True)

    
