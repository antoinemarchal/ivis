import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.constants import c, k_B
from astropy.io import fits
from reproject import reproject_interp
from spectral_cube import SpectralCube


PB_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR_PB_eff.fits"
FEATHERED_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_1_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_short_spacing.fits"
ATCA_CUBE_PATH = "/Users/antoine/Desktop/fullsurvey/lmc.hi.K.LSR.fits"
TARGET_VELOCITY = 238.6 * u.km / u.s
NU_HZ = 1.42040575177e9
OUTPUT_DIR = "/Users/antoine/Desktop/fullsurvey/inferno"
ATCA_REGRID_FITS = os.path.join(OUTPUT_DIR, "ATCA_regrid_on_ASKAP.fits")
ASKAP_ON_ATCA_FITS = os.path.join(OUTPUT_DIR, "ASKAP_+Parkes_on_ATCA.fits")
VMIN = -8.0e-5
VMAX = 1.5e-4
PLOT_SCALE = 1.0e3


def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr["CRPIX1"], hdr["CRPIX2"]]
    w.wcs.cdelt = np.array([hdr["CDELT1"], hdr["CDELT2"]])
    w.wcs.crval = [hdr["CRVAL1"], hdr["CRVAL2"]]
    w.wcs.ctype = [hdr["CTYPE1"], hdr["CTYPE2"]]
    return w


def K_to_jy_arcsec2(T_K, nu_hz):
    T = np.asarray(T_K) * u.K
    nu = np.asarray(nu_hz) * u.Hz
    I_nu = (2 * k_B * nu**2 / c**2 * T) / u.sr
    I_jy_arcsec2 = I_nu.to(
        u.Jy / u.arcsec**2,
        equivalencies=u.dimensionless_angles(),
    )
    return I_jy_arcsec2.value


def interpolate_velocity_plane(cube, vel):
    vel = vel.to(u.km / u.s)
    v_axis = cube.spectral_axis.to(u.km / u.s)

    if v_axis[0] > v_axis[-1]:
        cube = cube[::-1]
        v_axis = v_axis[::-1]

    v_vals = v_axis.value
    v_target = vel.value

    i2 = np.searchsorted(v_vals, v_target)
    if i2 == 0 or i2 == len(v_vals):
        raise ValueError("Requested velocity is outside the cube spectral range.")
    i1 = i2 - 1

    v1, v2 = v_vals[i1], v_vals[i2]
    w2 = (v_target - v1) / (v2 - v1)
    w1 = 1.0 - w2

    d1 = cube.filled_data[i1, :, :].value
    d2 = cube.filled_data[i2, :, :].value

    plane = w1 * d1 + w2 * d2
    return plane * cube.unit


def first_plane(data):
    return np.asarray(data[0] if data.ndim == 3 else data, dtype=float)


def image_header(header):
    out = wcs2D(header).to_header()
    out["BUNIT"] = "Jy / arcsec2"
    return out


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    hdu = fits.open(PB_PATH)
    pb_mean_full = hdu[0].data
    pb_mean_full /= np.max(pb_mean_full)
    mask = np.where(pb_mean_full > 0.05, 1, np.nan)

    hdu_feathered = fits.open(FEATHERED_PATH)
    target_header = hdu_feathered[0].header
    feathered = first_plane(hdu_feathered[0].data)
    w = wcs2D(target_header)
    shape = feathered.shape
    target_hdr = w.to_header()

    fitsname = ATCA_CUBE_PATH
    hdu_atca = fits.open(fitsname)
    hdr_atca = hdu_atca[0].header
    w_atca = wcs2D(hdr_atca)
    cube_atca = SpectralCube.read(fitsname)

    hi_slice_array_atca = interpolate_velocity_plane(cube_atca, TARGET_VELOCITY)
    atca_K, footprint = reproject_interp(
        (hi_slice_array_atca, w_atca.to_header()),
        target_hdr,
        shape_out=(shape[0], shape[1]),
    )
    atca_K[atca_K != atca_K] = 0.0
    I_atca = K_to_jy_arcsec2(atca_K, NU_HZ)
    ASKAP, footprint = reproject_interp(
        (feathered, target_hdr), w_atca.to_header(), shape_out=hi_slice_array_atca.shape
    )

    fits.PrimaryHDU(
        data=np.asarray(I_atca, dtype=np.float32),
        header=image_header(target_header),
    ).writeto(
        ATCA_REGRID_FITS, overwrite=True
    )
    fits.PrimaryHDU(
        data=np.asarray(ASKAP, dtype=np.float32),
        header=image_header(hdr_atca),
    ).writeto(
        ASKAP_ON_ATCA_FITS, overwrite=True
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=w_atca)
    ax.set_xlabel(r"RA", fontsize=18.0)
    ax.set_ylabel(r"DEC", fontsize=18.0)
    img = ax.imshow(
        K_to_jy_arcsec2(hi_slice_array_atca, NU_HZ) * PLOT_SCALE,
        vmin=VMIN * PLOT_SCALE,
        vmax=VMAX * PLOT_SCALE,
        origin="lower",
        cmap="inferno",
    )
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.0)
    cbar.set_label(r"$T_b\ (\mathrm{mJy}\,\mathrm{arcsec}^{-2})$", fontsize=18.0)
    plt.savefig(
        "/Users/antoine/Desktop/fullsurvey/inferno/ATCA_+Parkes_inferno_w_atca.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=400,
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=w_atca)
    ax.set_xlabel(r"RA", fontsize=18.0)
    ax.set_ylabel(r"DEC", fontsize=18.0)
    img = ax.imshow(
        ASKAP * PLOT_SCALE,
        vmin=VMIN * PLOT_SCALE,
        vmax=VMAX * PLOT_SCALE,
        origin="lower",
        cmap="inferno",
    )
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.0)
    cbar.set_label(r"$T_b\ (\mathrm{mJy}\,\mathrm{arcsec}^{-2})$", fontsize=18.0)
    plt.savefig(
        "/Users/antoine/Desktop/fullsurvey/inferno/ASKAP_+Parkes_inferno_w_atca.png",
        format="png",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=400,
    )
