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
from numpy.fft import fft2, ifft2, fftshift
from joblib import Parallel, delayed

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

def gauss_beam(sigma, shape, cx, cy, FWHM=False):
    ny, nx = shape
    X=np.arange(nx)
    Y=np.arange(ny)
    ymap,xmap=np.meshgrid(X,Y)

    if (nx % 2) == 0:
        xmap = xmap - (nx)/2.
    else:
        xmap = xmap - (nx-1.)/2.

    if (ny % 2) == 0:
        ymap = ymap - (ny)/2.
    else:
        ymap = ymap - (ny-1.)/2.

    map = np.sqrt((xmap-cx)**2.+(ymap-cy)**2.)

    if FWHM == True:
        sigma = sigma / (2.*np.sqrt(2.*np.log(2.)))

    gauss = np.exp(-0.5*(map)**2./sigma**2.)

    return gauss

def process_one(i):
    target_velocity = v[i]
    
    hi_slice = cube_sd.spectral_interpolate(np.array([target_velocity]) * u.km/u.s)
    hi_slice_array = hi_slice.hdu.data[0]
    
    sd_K, footprint = reproject_interp(
        (hi_slice_array, w_sd.to_header()),
        target_hdr,
        shape_out=(shape[0], shape[1])
    )
    sd_K = np.nan_to_num(sd_K, nan=0.0)
    
    nu_Hz = 1.42040575177e9 * u.Hz
    sd = K_to_jy_arcsec2(sd_K, nu_Hz)
    
    fftfield_low  = fft2(sd)
    fftfield_high = fft2(np.asarray(cube_askap[i]))
    
    out = ifft2(fftfield_low * fftbeam + fftfield_high * fftpsf_inv).real.astype(np.float32)
    
    #write plot on disk ATCA
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    img = ax.imshow(out*mask, vmin=-8.e-5, vmax=1.5e-4, origin="lower", cmap="inferno")
    ax.contour(pb_mean_full, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
    plt.savefig("./plots/new/ASKAP_+Parkes_{:03d}.png".format(i), format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
        
    return i, out

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

    #Open cube
    path="/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/"
    fitsname="output_merged_all.fits"
    hdu = fits.open(path+fitsname)
    hdr = hdu[0].header
    cube = SpectralCube.read(path+fitsname)
    cube_askap = hdu[0].data

    v = cube.spectral_axis.value
    
    v0 = np.mean(cube.spectral_axis.value) * u.km/u.s
    dv = 1.1 * abs(np.max(cube.spectral_axis.value) - np.min(cube.spectral_axis.value) ) * u.km/u.s
    vmin, vmax = v0 - dv, v0 + dv 
    
    # Cut cube
    cube_cut = cube_sd.spectral_slab(vmin, vmax)
    
    print("Original shape:", cube_sd.shape)
    print("Cut shape     :", cube_cut.shape)
    print("v range cut   :", cube_cut.spectral_axis.min(), "→", cube_cut.spectral_axis.max())
    
    # # Optional: write to a new FITS
    # cube_cut.write(path+"GASS_HI_LMC_cube_{}_pm5kms.fits".format(v0.value), overwrite=True)
    
    #reproject on target_header
    w = wcs2D(target_header)
    target_hdr = w.to_header()

    #Beam sd    
    bmaj = 0.5     #ATTENTION
    cdelt2 = np.abs(target_header["CDELT2"])
    bmaj_pix = bmaj / cdelt2 
    beam = gauss_beam(bmaj_pix, shape, 0, 0, FWHM=True)
    beam /= np.sum(beam)
    fftbeam = abs((fft2(beam)))    
    fftpsf_inv = 1-np.abs(fftbeam)

    #PB
    fitsname="/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/mask_7arcsec.fits"
    hdu = fits.open(fitsname)
    pb_mean_full = hdu[0].data
    #compute mask
    mask = np.where(pb_mean_full > 0.05, 1, np.nan)

    #Parallel version
    n_jobs = 12
        
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(process_one)(i) for i in range(len(v))
    )
    
    corrected = np.zeros((len(v), cube.shape[1], cube.shape[2]), dtype=np.float32)
    for i, out in results:
        corrected[i] = out

    #Write output array on disk
    path="/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/"
    fits.writeto(path + "ASKAP_+Parkes.fits", corrected, hdr, overwrite=True)

    stop


    
    # #Output array
    # corrected = np.zeros((len(v),cube.shape[1],cube.shape[2]))    

    # for i in np.arange(len(v)):
    #     target_velocity = v[i]
    #     hi_slice = cube_sd.spectral_interpolate(np.array([target_velocity])*u.km/u.s)
    #     hi_slice_array = hi_slice.hdu.data[0]  # Convert the SpectralCube slice to a NumPy array

    #     sd_K, footprint = reproject_interp((hi_slice_array,w_sd.to_header()), target_hdr, shape_out=(shape[0],shape[1]))
    #     sd_K[sd_K != sd_K] = 0.        
        
    #     nu_Hz = 1.42040575177e9*u.Hz #HI 21cm
    #     sd = K_to_jy_arcsec2(sd_K,nu_Hz)

    #     fftfield_low = fft2(sd)
    #     fftfield_high = fft2(np.array(cube_askap[i]))
    #     corrected[i] = ifft2(fftfield_low * fftbeam + fftfield_high * fftpsf_inv).real
