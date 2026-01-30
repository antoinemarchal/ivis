# -*- coding: utf-8 -*-
import os
import numpy as np
from astropy import units as u
from astropy.constants import c as c_light, k_B
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
from reproject import reproject_interp
import time
from numpy.fft import fft2, ifft2, fftshift
from joblib import Parallel, delayed
from spectral_cube import SpectralCube
from concurrent.futures import ProcessPoolExecutor, as_completed

from ivis.io import DataProcessor
from ivis.logger import logger
from ivis.models import ClassicIViS3D
from ivis.imager import Imager3D
from ivis.types import VisIData
from ivis.readers import CasacoreReader

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w


def jy_arcsec2_to_K(I_jy_arcsec2, nu):
    # Jy/arcsec^2 -> W m^-2 Hz^-1 sr^-1
    I_si = (u.Quantity(I_jy_arcsec2, u.Jy/u.arcsec**2)).to(
        u.W/u.m**2/u.Hz/u.sr,
        equivalencies=u.dimensionless_angles()
    )

    nu = u.Quantity(nu, u.Hz)  # works for float or Quantity

    # RJ
    T = (c_light**2 / (2.0 * k_B * nu**2)) * I_si

    # IMPORTANT: allow sr to behave as dimensionless here
    return T.to_value(u.K, equivalencies=u.dimensionless_angles())


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

    #PB
    fitsname="/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/mask_7arcsec.fits"
    hdu = fits.open(fitsname)
    pb_mean_full = hdu[0].data
    #compute mask
    mask = np.where(pb_mean_full > 0.05, 1, np.nan)

    #Open data cube
    path="/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/"
    fitsname="ASKAP_+Parkes.fits"
    hdu = fits.open(path+fitsname)
    hdr = hdu[0].header
    cube = hdu[0].data

    nu_Hz = 1.42040575177e9 * u.Hz
    vmin = jy_arcsec2_to_K(-8.e-5,nu_Hz)
    vmax = jy_arcsec2_to_K(1.5e-4,nu_Hz)

    #Make figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    
    data0 = (jy_arcsec2_to_K(cube[0], nu_Hz).astype(np.float32) * mask.astype(np.float32))
    img = ax.imshow(data0, vmin=-40, vmax=100, origin="lower", cmap="inferno")
    
    # static contours? if pb_mean_full doesn’t change, draw once:
    ax.contour(pb_mean_full, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    
    cax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label(r"$T_b$ (K)", fontsize=18.)
    
    for i in tqdm(range(cube.shape[0])):
        data = (jy_arcsec2_to_K(cube[i], nu_Hz).astype(np.float32) * mask.astype(np.float32))
        img.set_data(data)
        
        fig.savefig(f"./plots/new/ASKAP_+Parkes_{i:03d}.png", dpi=400, pad_inches=0.02, bbox_inches='tight')

    plt.close(fig)

    
    # for i in tqdm(np.arange(cube.shape[0])):
    #     #write plot on disk ATCA
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    #     ax.set_xlabel(r"RA (deg)", fontsize=18.)
    #     ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    #     img = ax.imshow(jy_arcsec2_to_K(cube[i],nu_Hz)*mask, vmin=-40, vmax=100, 
    #                     origin="lower", cmap="inferno")
    #     ax.contour(pb_mean_full, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    #     colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    #     cbar = fig.colorbar(img, cax=colorbar_ax)
    #     cbar.ax.tick_params(labelsize=14.)
    #     cbar.set_label(r"$T_b$ (K)", fontsize=18.)
    #     plt.savefig("./plots/new/ASKAP_+Parkes_{:03d}.png".format(i), format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
    #     plt.close(fig)
