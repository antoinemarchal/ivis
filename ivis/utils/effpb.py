# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
from astropy import units as u
from astropy.coordinates import SkyCoord
from tqdm import tqdm

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w

if __name__ == '__main__':    
    #path data
    path_beams = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/BEAMS/" #directory of primary beams
    mfits = sorted(glob.glob(path_beams+"*.fits"))
    
    #REF WCS INPUT USER
    cfield = SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs')
    filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
    target_header = fits.open(filename)[0].header
    target_header["CRVAL1"] = cfield.ra.value
    target_header["CRVAL2"] = cfield.dec.value
    shape = (target_header["NAXIS2"],target_header["NAXIS1"])

    w = wcs2D(target_header)
    target_header = w.to_header()

    effpb = np.zeros(shape)
    for filename in mfits:
        #Open single dish data
        print("Open: ", filename)
        hdu_BEAM = fits.open(filename)
        hdr_BEAM = hdu_BEAM[0].header
        w_BEAM = wcs2D(hdr_BEAM)
        field = hdu_BEAM[0].data
        
        reproj, footprint = reproject_interp((field,w_BEAM.to_header()), target_header, shape_out=(shape[0],shape[1]))
        reproj[reproj != reproj] = 0.
        print("Reprojection completed.")
                
        effpb += reproj/len(mfits)
        print("Append to effective primary beam.")
    
    #write on disk
    pathout = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/" #path where data will be packaged and stored
    hdu0 = fits.PrimaryHDU(effpb, header=target_header)
    hdulist = fits.HDUList([hdu0])
    hdulist.writeto(pathout + "effpb.fits", overwrite=True)
