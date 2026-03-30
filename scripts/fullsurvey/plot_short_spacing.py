import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from astropy import units as u
from astropy.constants import c, k_B
from astropy import wcs
from astropy.io import fits


PB_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR_PB_eff.fits"
JOINT_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_1_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits"
LINEAR_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR.fits"
INPUT_PATHS = [JOINT_PATH, LINEAR_PATH]
OUTPUT_DIR = "/Users/antoine/Desktop/IVIS_paper/ASKAP/plot_short_spacing_exports"

NU_HZ = 1.42040575177e9
VMIN = -8.0e-5
VMAX = 1.5e-4
CMAP = "inferno"
SUBIMAGE_FRACTION = 0.6


def wcs2d(header):
    out = wcs.WCS(naxis=2)
    out.wcs.crpix = [header["CRPIX1"], header["CRPIX2"]]
    out.wcs.cdelt = [header["CDELT1"], header["CDELT2"]]
    out.wcs.crval = [header["CRVAL1"], header["CRVAL2"]]
    out.wcs.ctype = [header["CTYPE1"], header["CTYPE2"]]
    return out


def k_to_jy_arcsec2(data_k, nu_hz):
    intensity = (2 * k_B * (nu_hz * u.Hz) ** 2 / c**2) * (np.asarray(data_k) * u.K) / u.sr
    return intensity.to(u.Jy / u.arcsec**2, equivalencies=u.dimensionless_angles()).value


def first_plane(data):
    return np.asarray(data[0] if data.ndim == 3 else data, dtype=float)


def centered_subimage_bounds(shape, fraction):
    ny, nx = shape
    sub_ny = max(1, int(ny * fraction))
    sub_nx = max(1, int(nx * fraction))
    y0 = (ny - sub_ny) // 2
    x0 = (nx - sub_nx) // 2
    return x0, y0, sub_nx, sub_ny


def load_original_image(path):
    with fits.open(path) as hdul:
        data = first_plane(hdul[0].data)
        header = hdul[0].header
    if "LINEAR" in os.path.basename(path):
        data = k_to_jy_arcsec2(data, NU_HZ)
    return data, header


def output_png_path(input_path, suffix):
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(OUTPUT_DIR, f"{base}{suffix}.png")


def plot_image(data, header, pb, output_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=wcs2d(header))
    ax.set_xlabel(r"RA (deg)", fontsize=18.0)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.0)
    img = ax.imshow(data, vmin=VMIN, vmax=VMAX, origin="lower", cmap=CMAP)
    ax.contour(pb, linestyles="--", levels=[0.05, 0.1], colors=["w", "w"])
    x0, y0, width, height = centered_subimage_bounds(data.shape, SUBIMAGE_FRACTION)
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=2.0,
            edgecolor="white",
            facecolor="none",
        )
    )
    cax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=14.0)
    cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.0)
    fig.savefig(output_path, dpi=400, pad_inches=0.02, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with fits.open(PB_PATH) as hdul:
        pb = hdul[0].data.astype(float)
    pb /= np.nanmax(pb)
    mask = np.where(pb > 0.05, 1.0, np.nan)

    joint, joint_header = load_original_image(JOINT_PATH)
    linear, _ = load_original_image(LINEAR_PATH)

    plot_image(
        (joint - np.nanmean(joint)) * mask,
        joint_header,
        pb,
        output_png_path(JOINT_PATH, "_ASKAP_only"),
    )
    plot_image(
        ((joint - np.nanmean(joint)) - (linear - np.nanmean(linear))) * mask,
        joint_header,
        pb,
        output_png_path(JOINT_PATH, "_RESIDUAL_JOINT_VS_LINEAR_ASKAP_only"),
    )

    for input_path in INPUT_PATHS:
        root = os.path.splitext(input_path)[0]
        for suffix in ("_short_spacing.fits", "_sd_regrid.fits"):
            with fits.open(f"{root}{suffix}") as hdul:
                data = hdul[0].data.astype(float)
                header = hdul[0].header
            plot_image(
                data * mask,
                header,
                pb,
                output_png_path(input_path, suffix[:-5]),
            )
