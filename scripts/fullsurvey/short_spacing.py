import os

import numpy as np
from astropy import units as u
from astropy.constants import c, k_B
from astropy.io import fits
from astropy import wcs
from numpy.fft import fft2, ifft2
from reproject import reproject_interp
from spectral_cube import SpectralCube

INPUT_PATHS = [
    "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_1_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits",
    "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR.fits",
]
SD_CUBE_PATH = "/Users/antoine/Desktop/fullsurvey/GASS_HI_LMC_cube.fits"
NU_HZ = 1.42040575177e9
TARGET_VELOCITY = 238.6 * u.km / u.s
INT_FWHM_DEG = (21 * u.arcsec).to(u.deg).value
SD_FWHM_DEG = (16 * u.arcmin).to(u.deg).value


def gaussian_transfer_function(shape, fwhm_pix):
    ky = np.fft.fftfreq(shape[0])
    kx = np.fft.fftfreq(shape[1])
    kx, ky = np.meshgrid(kx, ky)
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-(2 * np.pi**2) * sigma**2 * (kx**2 + ky**2))


def feather_casa_like(sd, itf, fwhm_sd_pix, fwhm_int_pix, eps=1e-12):
    f_sd = fft2(np.nan_to_num(sd, nan=0.0))
    f_itf = fft2(np.nan_to_num(itf, nan=0.0))
    b_sd = gaussian_transfer_function(sd.shape, fwhm_sd_pix)
    b_itf = gaussian_transfer_function(sd.shape, fwhm_int_pix)
    taper = (b_sd**2) / (b_sd**2 + b_itf**2 + eps)
    taper[0, 0] = 1.0
    return ifft2(f_itf + taper * (f_sd - f_itf)).real


def k_to_jy_arcsec2(data_k, nu_hz):
    intensity = (2 * k_B * (nu_hz * u.Hz) ** 2 / c**2) * (np.asarray(data_k) * u.K) / u.sr
    return intensity.to(u.Jy / u.arcsec**2, equivalencies=u.dimensionless_angles()).value


def wcs2d(header):
    out = wcs.WCS(naxis=2)
    out.wcs.crpix = [header["CRPIX1"], header["CRPIX2"]]
    out.wcs.cdelt = [header["CDELT1"], header["CDELT2"]]
    out.wcs.crval = [header["CRVAL1"], header["CRVAL2"]]
    out.wcs.ctype = [header["CTYPE1"], header["CTYPE2"]]
    return out


def interpolate_velocity_plane(cube, velocity):
    spectral_axis = cube.spectral_axis.to(u.km / u.s)
    if spectral_axis[0] > spectral_axis[-1]:
        cube = cube[::-1]
        spectral_axis = spectral_axis[::-1]
    values = spectral_axis.value
    target = velocity.to_value(u.km / u.s)
    i2 = np.searchsorted(values, target)
    if i2 == 0 or i2 == len(values):
        raise ValueError("Requested velocity is outside the cube spectral range.")
    i1 = i2 - 1
    w2 = (target - values[i1]) / (values[i2] - values[i1])
    w1 = 1.0 - w2
    d1 = cube.filled_data[i1].value
    d2 = cube.filled_data[i2].value
    return w1 * d1 + w2 * d2


def first_plane(data):
    return np.asarray(data[0] if data.ndim == 3 else data, dtype=float)


def image_header(header):
    out = wcs2d(header).to_header()
    out["BUNIT"] = "Jy / arcsec2"
    return out


def write_like(path, data, header):
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)


if __name__ == "__main__":
    with fits.open(INPUT_PATHS[0]) as hdul:
        ref_header = hdul[0].header.copy()
        ref_shape = first_plane(hdul[0].data).shape

    sd_cube = SpectralCube.read(SD_CUBE_PATH)
    sd_plane = interpolate_velocity_plane(sd_cube, TARGET_VELOCITY)
    sd_header = wcs2d(fits.getheader(SD_CUBE_PATH)).to_header()
    sd_regrid_k, _ = reproject_interp(
        (sd_plane, sd_header),
        wcs2d(ref_header).to_header(),
        shape_out=ref_shape,
    )
    sd_regrid = k_to_jy_arcsec2(np.nan_to_num(sd_regrid_k, nan=0.0), NU_HZ)

    for input_path in INPUT_PATHS:
        with fits.open(input_path) as hdul:
            target_header = hdul[0].header.copy()
            image = first_plane(hdul[0].data)
            if "LINEAR" in os.path.basename(input_path):
                image = k_to_jy_arcsec2(image, NU_HZ)

        pixel_scale_deg = abs(target_header["CDELT2"])
        feathered = feather_casa_like(
            sd_regrid,
            image,
            SD_FWHM_DEG / pixel_scale_deg,
            INT_FWHM_DEG / pixel_scale_deg,
        )

        output_root = os.path.splitext(input_path)[0]
        output_header = image_header(target_header)
        write_like(f"{output_root}_sd_regrid.fits", sd_regrid, output_header)
        write_like(f"{output_root}_short_spacing.fits", feathered, output_header)
