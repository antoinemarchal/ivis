import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
from astropy.wcs import WCS


PB_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR_PB_eff.fits"
INPUT_CUBE_PATH = (
    "/Users/antoine/Desktop/IVIS_paper/ASKAP/"
    "output_chan_1267_6_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_short_spacing_cube.fits" #792
)
input_cube_name = os.path.basename(INPUT_CUBE_PATH)
channel_match = re.search(r"chan_(\d+)", input_cube_name)
channel_suffix = channel_match.group(1) if channel_match else "unknown"
OUTPUT_DIR = f"/Users/antoine/Desktop/IVIS_paper/ASKAP/central_rgb_exports_{channel_suffix}"
OUTPUT_SUFFIX = f"_{channel_suffix}"
OUTPUT_DIRECT_PNG_PATH = f"{OUTPUT_DIR}/central_rgb_direct{OUTPUT_SUFFIX}.png"
OUTPUT_SMOOTHED_PNG_PATH = f"{OUTPUT_DIR}/central_rgb_smoothed{OUTPUT_SUFFIX}.png"
OUTPUT_COMPARISON_PNG_PATH = f"{OUTPUT_DIR}/central_rgb_comparison{OUTPUT_SUFFIX}.png"
GAMMA = 0.5  # previous: 0.8
LOW_PERCENTILE = 5.0  # previous: 14.0
HIGH_PERCENTILE = 99.7  # previous: 99.85
SATURATION_BOOST = 1.55  # previous: 1.3
CONTRAST_BOOST = 1.35  # previous: 0.84


def normalize(img, vmin, vmax, gamma):
    norm = np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = norm**gamma
    return np.clip(0.5 + CONTRAST_BOOST * (norm - 0.5), 0.0, 1.0)


def spectral_values_from_header(header, nchan):
    if "CRVAL3" not in header or "CRPIX3" not in header or "CDELT3" not in header:
        return None
    indices = np.arange(nchan, dtype=float)
    return header["CRVAL3"] + (indices + 1.0 - header["CRPIX3"]) * header["CDELT3"]


def finalize_rgb(rgb):
    out = np.asarray(rgb, dtype=float).copy()
    luminance = np.mean(out, axis=-1, keepdims=True)
    out = np.clip(luminance + SATURATION_BOOST * (out - luminance), 0.0, 1.0)
    nan_mask = np.isnan(out).any(axis=-1)
    out[nan_mask] = [1.0, 1.0, 1.0]
    return out


def rgb_legend_image(height=300, width=24):
    y = np.linspace(0.0, 1.0, height)[:, None]
    legend = np.zeros((height, width, 3), dtype=float)
    legend[..., 0] = y
    legend[..., 1] = 1.0 - np.abs(2.0 * y - 1.0)
    legend[..., 2] = 1.0 - y
    return legend


def draw_rgb_legend(fig):
    cax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cax.set_facecolor("white")
    mappable = plt.cm.ScalarMappable(
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        cmap=colors.ListedColormap(["white", "white"]),
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=14.0, colors="white", length=3)
    cbar.set_label(
        r"$T_b\ (\mathrm{mJy}\,\mathrm{arcsec}^{-2})$",
        fontsize=18.0,
        color="white",
    )
    cbar.outline.set_edgecolor("white")
    for spine in cax.spines.values():
        spine.set_color("white")


def save_rgb_figure(path, rgb, wcs_2d, pb):
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=wcs_2d)
    ax.set_facecolor("white")
    ax.imshow(rgb, origin="lower")
    ax.contour(pb, linestyles="--", levels=[0.05, 0.1], colors=["w", "w"])
    ax.set_xlabel("RA", fontsize=18.0)
    ax.set_ylabel("DEC", fontsize=18.0)
    draw_rgb_legend(fig)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_comparison_figure(path, rgb_direct, rgb_smoothed, wcs_2d, pb):
    fig = plt.figure(figsize=(22, 10), facecolor="white")
    axes = [
        fig.add_axes([0.05, 0.1, 0.33, 0.8], projection=wcs_2d),
        fig.add_axes([0.54, 0.1, 0.33, 0.8], projection=wcs_2d),
    ]
    for ax, rgb in zip(axes, [rgb_direct, rgb_smoothed]):
        ax.set_facecolor("white")
        ax.imshow(rgb, origin="lower")
        ax.contour(pb, linestyles="--", levels=[0.05, 0.1], colors=["w", "w"])
        ax.set_xlabel("RA", fontsize=18.0)
        ax.set_ylabel("DEC", fontsize=18.0)
    legend_axes = [
        fig.add_axes([0.40, 0.11, 0.02, 0.78]),
        fig.add_axes([0.89, 0.11, 0.02, 0.78]),
    ]
    for cax in legend_axes:
        cax.set_facecolor("white")
        mappable = plt.cm.ScalarMappable(
            norm=colors.Normalize(vmin=0.0, vmax=1.0),
            cmap=colors.ListedColormap(["white", "white"]),
        )
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=14.0, colors="white", length=3)
        cbar.set_label(
            r"$T_b\ (\mathrm{mJy}\,\mathrm{arcsec}^{-2})$",
            fontsize=18.0,
            color="white",
        )
        cbar.outline.set_edgecolor("white")
        for spine in cax.spines.values():
            spine.set_color("white")
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def robust_limits(images, low_percentile, high_percentile):
    stacked = np.concatenate([img[np.isfinite(img)] for img in images if np.isfinite(img).any()])
    if stacked.size == 0:
        raise ValueError("No finite pixel values found to determine display limits.")
    vmin = np.percentile(stacked, low_percentile)
    vmax = np.percentile(stacked, high_percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Invalid display limits derived from the selected channels.")
    return float(vmin), float(vmax)


def read_cube_metadata(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
        shape = data.shape
    return header, shape


def load_cube_channels(path, channel_indices):
    unique_indices = sorted(set(channel_indices))
    planes = {}
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        for idx in unique_indices:
            planes[idx] = np.asarray(data[idx], dtype=float)
    return planes


def make_rgb_image(red, green, blue):
    vmin, vmax = robust_limits([blue, green, red], LOW_PERCENTILE, HIGH_PERCENTILE)
    rgb = finalize_rgb(
        np.stack(
            [
                normalize(red, vmin, vmax, GAMMA),
                normalize(green, vmin, vmax, GAMMA),
                normalize(blue, vmin, vmax, GAMMA),
            ],
            axis=-1,
        )
    )
    return rgb, vmin, vmax


def make_rgb_labels(values=None, unit="", channels=None):
    if values is None:
        return {
            "B": f"B: ch {channels[0]}",
            "G": f"G: ch {channels[1]}",
            "R": f"R: ch {channels[2]}",
        }
    return {
        "B": f"B: {values[0]:.1f}",
        "G": f"G: {values[1]:.1f}",
        "R": f"R: {values[2]:.1f}",
    }


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    header, cube_shape = read_cube_metadata(INPUT_CUBE_PATH)
    with fits.open(PB_PATH) as hdul:
        pb = np.asarray(hdul[0].data, dtype=float)
    pb /= np.nanmax(pb)

    if len(cube_shape) != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube_shape}.")

    nchan = cube_shape[0]
    if nchan < 4:
        raise ValueError("Need at least 4 channels to build the smoothed RGB image.")

    center = nchan // 2
    if center == 0 or center == nchan - 1:
        raise ValueError("Central channel does not have both adjacent channels available.")

    i = max(0, min((nchan - 4) // 2, nchan - 4))
    channels_to_load = [center - 1, center, center + 1, i, i + 1, i + 2, i + 3]
    planes = load_cube_channels(INPUT_CUBE_PATH, channels_to_load)

    blue_direct = planes[center - 1]
    green_direct = planes[center]
    red_direct = planes[center + 1]
    rgb_direct, vmin_direct, vmax_direct = make_rgb_image(red_direct, green_direct, blue_direct)

    blue = 0.5 * (planes[i] + planes[i + 1])
    green = 0.5 * (planes[i + 1] + planes[i + 2])
    red = 0.5 * (planes[i + 2] + planes[i + 3])
    rgb_smoothed, vmin_smoothed, vmax_smoothed = make_rgb_image(red, green, blue)

    wcs_2d = WCS(header).celestial
    spectral_values = spectral_values_from_header(header, nchan)
    if spectral_values is None:
        direct_labels = make_rgb_labels(channels=(center - 1, center, center + 1))
        smoothed_labels = make_rgb_labels(channels=(f"{i}-{i+1}", f"{i+1}-{i+2}", f"{i+2}-{i+3}"))
        save_rgb_figure(OUTPUT_DIRECT_PNG_PATH, rgb_direct, wcs_2d, pb)
        save_rgb_figure(OUTPUT_SMOOTHED_PNG_PATH, rgb_smoothed, wcs_2d, pb)
        save_comparison_figure(
            OUTPUT_COMPARISON_PNG_PATH,
            rgb_direct,
            rgb_smoothed,
            wcs_2d,
            pb,
        )
        print(f"Using direct channels: B={center-1}, G={center}, R={center+1}")
        print(f"Direct display limits: vmin={vmin_direct:.6e}, vmax={vmax_direct:.6e}")
        print(f"Using channels: B=({i},{i+1}), G=({i+1},{i+2}), R=({i+2},{i+3})")
        print(f"Smoothed display limits: vmin={vmin_smoothed:.6e}, vmax={vmax_smoothed:.6e}")
    else:
        unit = header.get("CUNIT3", "").strip()
        direct_labels = make_rgb_labels(
            values=(spectral_values[center - 1], spectral_values[center], spectral_values[center + 1]),
            unit=unit,
        )
        blue_val = 0.5 * (spectral_values[i] + spectral_values[i + 1])
        green_val = 0.5 * (spectral_values[i + 1] + spectral_values[i + 2])
        red_val = 0.5 * (spectral_values[i + 2] + spectral_values[i + 3])
        smoothed_labels = make_rgb_labels(
            values=(blue_val, green_val, red_val),
            unit=unit,
        )
        save_rgb_figure(OUTPUT_DIRECT_PNG_PATH, rgb_direct, wcs_2d, pb)
        save_rgb_figure(OUTPUT_SMOOTHED_PNG_PATH, rgb_smoothed, wcs_2d, pb)
        save_comparison_figure(
            OUTPUT_COMPARISON_PNG_PATH,
            rgb_direct,
            rgb_smoothed,
            wcs_2d,
            pb,
        )
        print(
            "Using direct channels and spectral values: "
            f"B={center-1} ({spectral_values[center-1]:.6f} {unit}), "
            f"G={center} ({spectral_values[center]:.6f} {unit}), "
            f"R={center+1} ({spectral_values[center+1]:.6f} {unit})"
        )
        print(f"Direct display limits: vmin={vmin_direct:.6e}, vmax={vmax_direct:.6e}")
        print(
            "Using smoothed channels and spectral values: "
            f"B=({i},{i+1}) -> {blue_val:.6f} {unit}, "
            f"G=({i+1},{i+2}) -> {green_val:.6f} {unit}, "
            f"R=({i+2},{i+3}) -> {red_val:.6f} {unit}"
        )
        print(f"Smoothed display limits: vmin={vmin_smoothed:.6e}, vmax={vmax_smoothed:.6e}")
