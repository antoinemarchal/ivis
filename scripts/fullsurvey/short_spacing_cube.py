import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
from astropy import units as u
from astropy.constants import c, k_B
from astropy.io import fits
from astropy import wcs
from numpy.fft import fft2, ifft2
from reproject import reproject_interp
from spectral_cube import SpectralCube
from tqdm import tqdm

INPUT_CUBE_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_792_6_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits"
SD_CUBE_PATH = "/Users/antoine/Desktop/fullsurvey/GASS_HI_LMC_cube.fits"
NU_HZ = 1.42040575177e9
INT_FWHM_DEG = (21 * u.arcsec).to(u.deg).value
SD_FWHM_DEG = (16 * u.arcmin).to(u.deg).value
SHORT_SPACING_METHOD = "legacy_fft"
LEGACY_SD_BEAM_FWHM_DEG = 0.5
N_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def k_to_jy_arcsec2_factor(nu_hz):
    intensity_per_kelvin = (2 * k_B * (nu_hz * u.Hz) ** 2 / c**2) * (1.0 * u.K) / u.sr
    return intensity_per_kelvin.to_value(
        u.Jy / u.arcsec**2,
        equivalencies=u.dimensionless_angles(),
    )


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


def gauss_beam(sigma, shape, cx=0.0, cy=0.0, fwhm=False):
    ny, nx = shape
    x = np.arange(nx)
    y = np.arange(ny)
    ymap, xmap = np.meshgrid(x, y)

    if (nx % 2) == 0:
        xmap = xmap - nx / 2.0
    else:
        xmap = xmap - (nx - 1.0) / 2.0

    if (ny % 2) == 0:
        ymap = ymap - ny / 2.0
    else:
        ymap = ymap - (ny - 1.0) / 2.0

    radius = np.sqrt((xmap - cx) ** 2.0 + (ymap - cy) ** 2.0)

    if fwhm:
        sigma = sigma / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    return np.exp(-0.5 * radius**2.0 / sigma**2.0)


def legacy_fft_short_spacing(sd, itf, target_header, sd_beam_fwhm_deg=LEGACY_SD_BEAM_FWHM_DEG):
    shape = sd.shape
    cdelt2 = abs(target_header["CDELT2"])

    restored_fwhm_pix = INT_FWHM_DEG / cdelt2
    restored_beam = gauss_beam(restored_fwhm_pix, shape, fwhm=True)
    restored_beam /= np.sum(restored_beam)
    _ = np.abs(fft2(restored_beam))

    sd_fwhm_pix = sd_beam_fwhm_deg / cdelt2
    beam = gauss_beam(sd_fwhm_pix, shape, fwhm=True)
    beam /= np.sum(beam)
    fftbeam = np.abs(fft2(beam))
    fftpsf_inv = 1.0 - fftbeam

    fftfield_low = fft2(np.nan_to_num(sd, nan=0.0))
    fftfield_high = fft2(np.nan_to_num(itf, nan=0.0))
    return ifft2(fftfield_low * fftbeam + fftfield_high * fftpsf_inv).real


def k_to_jy_arcsec2(data_k, nu_hz):
    return np.asarray(data_k) * k_to_jy_arcsec2_factor(nu_hz)


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


def prepare_spectral_interpolator(cube, velocities):
    if velocities[0] > velocities[-1]:
        velocities = velocities[::-1]
        cube_data = cube.filled_data[:].value[::-1]
    else:
        cube_data = cube.filled_data[:].value
    return velocities, np.asarray(cube_data, dtype=np.float32)


def interpolate_velocity_plane_cached(cube_data, spectral_values, target_value):
    i2 = np.searchsorted(spectral_values, target_value)
    if i2 == 0 or i2 == len(spectral_values):
        raise ValueError("Requested velocity is outside the cube spectral range.")
    i1 = i2 - 1
    w2 = (target_value - spectral_values[i1]) / (spectral_values[i2] - spectral_values[i1])
    w1 = 1.0 - w2
    return w1 * cube_data[i1] + w2 * cube_data[i2]


def spectral_axis_radio_velocity(cube, rest_frequency_hz):
    rest_frequency = rest_frequency_hz * u.Hz
    axis = cube.spectral_axis
    try:
        return axis.to(u.km / u.s)
    except u.UnitConversionError:
        return axis.to(u.km / u.s, equivalencies=u.doppler_radio(rest_frequency))


def image_cube_header(header):
    out = header.copy()
    out["BUNIT"] = "Jy / arcsec2"
    return out


def write_cube(path, data, header):
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(path, overwrite=True)


def build_legacy_fft_weights(shape, pixel_scale_deg, sd_beam_fwhm_deg=LEGACY_SD_BEAM_FWHM_DEG):
    sd_fwhm_pix = sd_beam_fwhm_deg / pixel_scale_deg
    beam = gauss_beam(sd_fwhm_pix, shape, fwhm=True)
    beam /= np.sum(beam)
    fftbeam = np.abs(fft2(beam))
    return fftbeam, 1.0 - fftbeam


def build_casa_taper(shape, pixel_scale_deg, eps=1e-12):
    b_sd = gaussian_transfer_function(shape, SD_FWHM_DEG / pixel_scale_deg)
    b_itf = gaussian_transfer_function(shape, INT_FWHM_DEG / pixel_scale_deg)
    taper = (b_sd**2) / (b_sd**2 + b_itf**2 + eps)
    taper[0, 0] = 1.0
    return taper


def feather_casa_with_taper(sd, itf, taper):
    f_sd = fft2(np.nan_to_num(sd, nan=0.0))
    f_itf = fft2(np.nan_to_num(itf, nan=0.0))
    return ifft2(f_itf + taper * (f_sd - f_itf)).real


def legacy_fft_short_spacing_cached(sd, itf, fftbeam, fftpsf_inv):
    fftfield_low = fft2(np.nan_to_num(sd, nan=0.0))
    fftfield_high = fft2(np.nan_to_num(itf, nan=0.0))
    return ifft2(fftfield_low * fftbeam + fftfield_high * fftpsf_inv).real


def process_channel(
    chan_idx,
    velocity_values,
    input_data,
    sd_cube_data,
    sd_velocity_values,
    sd_header,
    target_wcs_header,
    spatial_shape,
    k_to_jy_factor,
    fftbeam,
    fftpsf_inv,
    taper,
):
    image = input_data[chan_idx]
    sd_plane = interpolate_velocity_plane_cached(sd_cube_data, sd_velocity_values, velocity_values[chan_idx])
    sd_regrid_k, _ = reproject_interp(
        (sd_plane, sd_header),
        target_wcs_header,
        shape_out=spatial_shape,
    )
    sd_regrid = np.nan_to_num(sd_regrid_k, nan=0.0) * k_to_jy_factor

    if fftbeam is not None:
        feathered = legacy_fft_short_spacing_cached(
            sd_regrid,
            image,
            fftbeam,
            fftpsf_inv,
        )
    else:
        feathered = feather_casa_with_taper(
            sd_regrid,
            image,
            taper,
        )

    return chan_idx, sd_regrid.astype(np.float32), feathered.astype(np.float32)


if __name__ == "__main__":
    input_cube = SpectralCube.read(INPUT_CUBE_PATH)
    sd_cube = SpectralCube.read(SD_CUBE_PATH)
    sd_header = wcs2d(fits.getheader(SD_CUBE_PATH)).to_header()

    with fits.open(INPUT_CUBE_PATH) as hdul:
        target_header = hdul[0].header.copy()
        input_data = np.asarray(hdul[0].data, dtype=float)

    velocities = spectral_axis_radio_velocity(input_cube, NU_HZ)
    velocity_values = velocities.to_value(u.km / u.s)
    sd_velocity_values = sd_cube.spectral_axis.to_value(u.km / u.s)
    sd_velocity_values, sd_cube_data = prepare_spectral_interpolator(sd_cube, sd_velocity_values)
    print("Channel velocities [km/s] (radio convention):")
    print(velocity_values)
    print(
        f"Velocity summary: nchan={len(velocity_values)}, "
        f"first={velocity_values[0]:.6f}, last={velocity_values[-1]:.6f}, "
        f"step={velocity_values[1] - velocity_values[0]:.6f}"
    )
    spatial_shape = input_data.shape[-2:]
    nchan = input_data.shape[0]

    sd_regrid_cube = np.empty((nchan, spatial_shape[0], spatial_shape[1]), dtype=np.float32)
    feathered_cube = np.empty_like(sd_regrid_cube)

    pixel_scale_deg = abs(target_header["CDELT2"])
    target_wcs_header = wcs2d(target_header).to_header()
    k_to_jy_factor = k_to_jy_arcsec2_factor(NU_HZ)

    if SHORT_SPACING_METHOD == "legacy_fft":
        fftbeam, fftpsf_inv = build_legacy_fft_weights(
            spatial_shape,
            pixel_scale_deg,
            sd_beam_fwhm_deg=LEGACY_SD_BEAM_FWHM_DEG,
        )
        taper = None
    else:
        fftbeam = None
        fftpsf_inv = None
        taper = build_casa_taper(spatial_shape, pixel_scale_deg)

    worker_count = min(N_WORKERS, nchan)
    print(f"Processing {nchan} channels with {worker_count} worker threads")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                process_channel,
                chan_idx,
                velocity_values,
                input_data,
                sd_cube_data,
                sd_velocity_values,
                sd_header,
                target_wcs_header,
                spatial_shape,
                k_to_jy_factor,
                fftbeam,
                fftpsf_inv,
                taper,
            )
            for chan_idx in range(nchan)
        ]

        for future in tqdm(as_completed(futures), total=nchan, desc="Short-spacing cube"):
            chan_idx, sd_regrid, feathered = future.result()
            sd_regrid_cube[chan_idx] = sd_regrid
            feathered_cube[chan_idx] = feathered

    output_root = os.path.splitext(INPUT_CUBE_PATH)[0]
    output_header = image_cube_header(target_header)
    write_cube(f"{output_root}_sd_regrid_cube.fits", sd_regrid_cube, output_header)
    write_cube(f"{output_root}_short_spacing_cube.fits", feathered_cube, output_header)
    total_elapsed = time.perf_counter() - t0
    print(f"Completed {nchan} channels in {total_elapsed:.1f}s ({nchan / total_elapsed:.2f} ch/s)")
