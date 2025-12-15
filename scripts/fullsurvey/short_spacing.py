# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
from reproject import reproject_interp
from astropy.constants import k_B, c
from spectral_cube import SpectralCube
from numpy.fft import fft2, ifft2, fftshift

from ivis.io import DataProcessor
from ivis.imager import Imager
from ivis.models import ClassicIViS
from ivis.logger import logger
from ivis.utils import dunits, dutils

plt.ion()

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

def interpolate_velocity_plane(cube, vel):
    """
    Linearly interpolate a SpectralCube at a single velocity vel.
    Returns a 2D numpy array in cube.unit.
    """
    vel = vel.to(u.km/u.s)
    v_axis = cube.spectral_axis.to(u.km/u.s)
    
    # Ensure axis is increasing
    if v_axis[0] > v_axis[-1]:
        cube = cube[::-1]
        v_axis = v_axis[::-1]
        
    v_vals = v_axis.value
    v_target = vel.value
    
    # Find bracketing indices
    i2 = np.searchsorted(v_vals, v_target)
    if i2 == 0 or i2 == len(v_vals):
        raise ValueError("Requested velocity is outside the cube spectral range.")
    i1 = i2 - 1
    
    v1, v2 = v_vals[i1], v_vals[i2]
    w2 = (v_target - v1) / (v2 - v1)
    w1 = 1.0 - w2
    
    # Read only the two 2D planes
    # filled_data[...] returns a Quantity; .value gives numpy array
    d1 = cube.filled_data[i1, :, :].value
    d2 = cube.filled_data[i2, :, :].value
    
    plane = w1 * d1 + w2 * d2  # numpy 2D array
    return plane * cube.unit   # if you want a Quantity back


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

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w

if __name__ == '__main__':
    #PB
    fitsname="/Users/antoine/Desktop/fullsurvey/mask_7arcsec.fits"
    hdu = fits.open(fitsname)
    pb_mean_full = hdu[0].data
    #compute mask
    mask = np.where(pb_mean_full > 0.05, 1, np.nan)

    #Open IViS result
    # fitsname="/Users/antoine/Desktop/fullsurvey/output_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_new_PB_Nw_0.fits"
    fitsname="/Users/antoine/Desktop/fullsurvey/output_chan_765_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_new_PB_Nw_0.fits"

    # fitsname="/Users/antoine/Desktop/fullsurvey/output_1blocks_7arcsec_lambda_r_1_positivity_true.fits"
    hdu = fits.open(fitsname)
    target_header = hdu[0].header
    w = wcs2D(target_header)
    result = hdu[0].data #* 2 #ASKAP convention
    shape = result[0].shape
    
    #Open SD data
    fitsname="/Users/antoine/Desktop/fullsurvey/GASS_HI_LMC_cube_253pm5kms.fits"
    hdu_sd = fits.open(fitsname)
    hdr_sd = hdu_sd[0].header
    w_sd = wcs2D(hdr_sd)
    # Load the sd data cube (downloaded earlier)
    cube_sd = SpectralCube.read(fitsname)

    # #Open ATCA data
    # fitsname="/Users/antoine/Desktop/fullsurvey/lmc.hi.K.LSR_231pm5kms.fits"
    # hdu_atca = fits.open(fitsname)
    # hdr_atca = hdu_atca[0].header
    # w_atca = wcs2D(hdr_atca)
    # # Load the sd data cube (downloaded earlier)
    # cube_atca = SpectralCube.read(fitsname)
    # beam_atca = Beam(1.66666656733E-02*u.deg,  1.66666656733E-02*u.deg, 1.e-12*u.deg)
    
    #Beam sd
    beam_sd = Beam(0.26666*u.deg,  0.26666*u.deg, 1.e-12*u.deg)

    #Restored beam
    rbmaj = 21
    bmaj = rbmaj*u.arcsec.to(u.deg)
    cdelt2 = np.abs(target_header["CDELT2"])
    bmaj_pix = bmaj / cdelt2
    rbeam = gauss_beam(bmaj_pix, shape, 0, 0, FWHM=True)
    rbeam /= np.sum(rbeam)
    fftrbeam = abs(fft2(rbeam))

    #Beam sd    
    #ATTENTION
    bmaj = 1
    cdelt2 = np.abs(target_header["CDELT2"])
    bmaj_pix = bmaj / cdelt2 
    beam = gauss_beam(bmaj_pix, shape, 0, 0, FWHM=True)
    beam /= np.sum(beam)
    fftbeam = abs((fft2(beam)))
    
    fftpsf_inv = 1-np.abs(fftbeam)

    # Interpolate the HI intensity at the exact velocity

    # Usage
    target_velocity = 253.2170684#* u.km/u.s 231.23153293
    # hi_slice = interpolate_velocity_plane(cube_sd, target_velocity)
    # hi_slice_array = hi_slice.value
    # target_velocity = 231.3
    hi_slice = cube_sd.spectral_interpolate(np.array([target_velocity])*u.km/u.s)
    hi_slice_array = hi_slice.hdu.data[0]  # Convert the SpectralCube slice to a NumPy array
    
    #reproject on target_header
    w = wcs2D(target_header)
    target_hdr = w.to_header()
    sd_K, footprint = reproject_interp((hi_slice_array,w_sd.to_header()), target_hdr, shape_out=(shape[0],shape[1]))
    sd_K[sd_K != sd_K] = 0.        
    
    sd = sd_K / (beam_sd.sr).to(u.arcsec**2).value #convert Jy/beam to Jy/arcsec^2
    
    # nu_Hz = 1.42040575177e9*u.Hz FIXME
    # output = dunits.jy_per_arcsec2_to_K(result, nu_Hz)
    # result = output[0]
    
    fftfield_low = fft2(sd * 1)
    fftfield_high = fft2(np.array(result[0]))
    corrected = ifft2(fftfield_low * fftbeam + fftfield_high * fftpsf_inv).real

    #write plot on disk
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    vmin, vmax = np.nanpercentile(result[0], (0.01, 99.99))
    img = ax.imshow(corrected*mask, vmin=-8.e-5, vmax=1.5e-4, origin="lower", cmap="inferno")
    ax.contour(pb_mean_full, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
    plt.savefig("/Users/antoine/Desktop/fullsurvey/output_chan_765_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_new_PB_Nw_0.png", format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)

    stop

    #ATCA
    hi_slice_atca = cube_atca.spectral_interpolate(np.array([target_velocity])*u.km/u.s)
    hi_slice_array_atca = hi_slice_atca.hdu.data[0]  # Convert the SpectralCube slice to a NumPy array
    atca_K, footprint = reproject_interp((hi_slice_array_atca,w_atca.to_header()), target_hdr, shape_out=(shape[0],shape[1]))
    atca_K[atca_K != atca_K] = 0.        
    # atca = atca_K / (beam_atca.sr).to(u.arcsec**2).value #convert Jy/beam to Jy/arcsec^2
    I_atca = K_to_jy_arcsec2(atca_K, 1.42040575177e9)

    #write plot on disk ATCA
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    img = ax.imshow(I_atca*mask, vmin=-8.e-5, vmax=1.5e-4, origin="lower", cmap="gray_r")
    ax.contour(pb_mean_full, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
    plt.savefig("./ATCA_+Parkesgray_r.png", format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
