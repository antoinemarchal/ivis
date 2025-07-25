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

if __name__ == '__main__':    
    #Open single dish data
    path="/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/" #path where data will be packaged and stored
    fitsname="lambda_chipass_mollweide.fits"
    hdu_CHIPASS = fits.open(path+fitsname)
    hdr_CHIPASS = hdu_CHIPASS[1].header
    w_CHIPASS = wcs2D(hdr_CHIPASS)
    field = hdu_CHIPASS[1].data
        
    #REF WCS INPUT USER
    cfield = SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs')
    filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
    target_header = fits.open(filename)[0].header
    target_header["CRVAL1"] = cfield.ra.value
    target_header["CRVAL2"] = cfield.dec.value
    shape = (target_header["NAXIS2"],target_header["NAXIS1"])

    w = wcs2D(target_header)
    target_header = w.to_header()

    # Reproject the HI slice to the new header using reproject package
    reproj, footprint = reproject_interp((field,w_CHIPASS.to_header()), target_header, shape_out=(shape[0],shape[1]))
    reproj[reproj != reproj] = 0.        
    
    #write on disk
    pathout="/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/" #path where data will be packaged and stored
    hdu0 = fits.PrimaryHDU(reproj, header=target_header)
    hdulist = fits.HDUList([hdu0])
    hdulist.writeto(pathout + "reproj_CHIPASS.fits", overwrite=True)
