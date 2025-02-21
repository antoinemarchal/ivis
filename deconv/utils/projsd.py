# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from reproject import reproject_from_healpix, reproject_interp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from scipy import interpolate
from tqdm import tqdm
from spectral_cube import SpectralCube

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w

def get_sd(fitsname):
    hdu_GASS = fits.open(fitsname)
    hdr_GASS = hdu_GASS[0].header
    w_GASS = wcs2D(hdr_GASS)

    # Load the GASS data cube (downloaded earlier)
    cube = SpectralCube.read(fitsname)

    # Define the velocity you want
    target_velocity = np.array([161.36638687]) * u.km / u.s  # Correct way to create a Quantity

    # Interpolate the HI intensity at the exact velocity
    hi_slice = cube.spectral_interpolate(target_velocity)
        
    #REF WCS INPUT USER
    cfield = SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs')
    filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
    target_header = fits.open(filename)[0].header
    target_header["CRVAL1"] = cfield.ra.value
    target_header["CRVAL2"] = cfield.dec.value
    shape = (target_header["NAXIS2"],target_header["NAXIS1"])

    w = wcs2D(target_header)
    target_header = w.to_header()
    target_header["BMIN"] = 0.26666667
    target_header["BMAJ"] = 0.26666667

    # Reproject the HI slice to the new header using reproject package
    hi_slice_array = hi_slice.hdu.data[0]  # Convert the SpectralCube slice to a NumPy array        
    reproj, footprint = reproject_interp((hi_slice_array,w_GASS.to_header()), target_header, shape_out=(shape[0],shape[1]))
    reproj[reproj != reproj] = 0.        

    #write on disk
    pathout="/priv/avatar/amarchal/GASS/data/"
    hdu0 = fits.PrimaryHDU(reproj, header=target_header)
    hdulist = fits.HDUList([hdu0])
    hdulist.writeto(pathout + "reproj_GASS_v.fits", overwrite=True)

    
if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/chan950/"
    filename = "scienceData.M344-11B.SB30584_SB30625_SB30665.beam18_SL.ms.contsub_chan950.ms"    
    
    #Open single dish data
    path="/priv/avatar/amarchal/GASS/data/"
    fitsname="GASS_HI_slab_cube_MCs.fit"

    get_sd(path+fitsname)
