import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS


OUTPUT_DIR = "/Users/antoine/Desktop/IVIS_paper/ASKAP/central_rgb_exports"
PB_PATH = "/Users/antoine/Desktop/IVIS_paper/ASKAP/output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR_PB_eff.fits"
INPUT_CUBE_PATH = (
    "/Users/antoine/Desktop/IVIS_paper/ASKAP/"
    "output_chan_1267_6_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_short_spacing_cube.fits"
)
OUTPUT_DIRECT_PNG_PATH = f"{OUTPUT_DIR}/central_rgb_direct.png"
OUTPUT_SMOOTHED_PNG_PATH = f"{OUTPUT_DIR}/central_rgb_smoothed.png"
OUTPUT_COMPARISON_PNG_PATH = f"{OUTPUT_DIR}/central_rgb_comparison.png"
GAMMA = 0.5
LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 99.7
SATURATION_BOOST = 1.55
CONTRAST_BOOST = 1.35


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


def draw_rgb_legend(fig, labels):
    cax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cax.imshow(rgb_legend_image(), origin="lower", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 150, 299])
    cax.set_yticklabels([labels["B"], labels["G"], labels["R"]], fontsize=14.0)
    cax.yaxis.tick_right()
    cax.set_ylabel(r"$v_{\mathrm{LSR}}\ (\mathrm{km}\,\mathrm{s}^{-1})$", fontsize=18.0, rotation=270, labelpad=22)
    cax.yaxis.set_label_position("right")
    for spine in cax.spines.values():
        spine.set_visible(True)


def save_rgb_figure(path, rgb, wcs_2d, labels, pb):
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=wcs_2d)
    ax.set_facecolor("white")
    ax.imshow(rgb, origin="lower")
    ax.contour(pb, linestyles="--", levels=[0.05, 0.1], colors=["w", "w"])
    ax.set_xlabel("RA", fontsize=18.0)
    ax.set_ylabel("DEC", fontsize=18.0)
    draw_rgb_legend(fig, labels)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_comparison_figure(path, rgb_direct, rgb_smoothed, wcs_2d, labels_left, labels_right, pb):
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
    for cax, labels in zip(legend_axes, [labels_left, labels_right]):
        cax.imshow(rgb_legend_image(), origin="lower", aspect="auto")
        cax.set_xticks([])
        cax.set_yticks([0, 150, 299])
        cax.set_yticklabels([labels["B"], labels["G"], labels["R"]], fontsize=14.0)
        cax.yaxis.tick_right()
        cax.set_ylabel(r"$v_{\mathrm{LSR}}\ (\mathrm{km}\,\mathrm{s}^{-1})$", fontsize=18.0, rotation=270, labelpad=22)
        cax.yaxis.set_label_position("right")
        for spine in cax.spines.values():
            spine.set_visible(True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
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

    with fits.open(INPUT_CUBE_PATH) as hdul:
        cube = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    with fits.open(PB_PATH) as hdul:
        pb = np.asarray(hdul[0].data, dtype=float)
    pb /= np.nanmax(pb)

    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got shape {cube.shape}.")

    nchan = cube.shape[0]
    if nchan < 4:
        raise ValueError("Need at least 4 channels to build the smoothed RGB image.")

    center = nchan // 2
    if center == 0 or center == nchan - 1:
        raise ValueError("Central channel does not have both adjacent channels available.")

    i = max(0, min((nchan - 4) // 2, nchan - 4))

    blue_direct = cube[center - 1]
    green_direct = cube[center]
    red_direct = cube[center + 1]
    vmin_direct, vmax_direct = robust_limits(
        [blue_direct, green_direct, red_direct], LOW_PERCENTILE, HIGH_PERCENTILE
    )
    rgb_direct = finalize_rgb(
        np.stack(
            [
                normalize(red_direct, vmin_direct, vmax_direct, GAMMA),
                normalize(green_direct, vmin_direct, vmax_direct, GAMMA),
                normalize(blue_direct, vmin_direct, vmax_direct, GAMMA),
            ],
            axis=-1,
        )
    )

    blue = 0.5 * (cube[i] + cube[i + 1])
    green = 0.5 * (cube[i + 1] + cube[i + 2])
    red = 0.5 * (cube[i + 2] + cube[i + 3])
    vmin_smoothed, vmax_smoothed = robust_limits(
        [blue, green, red], LOW_PERCENTILE, HIGH_PERCENTILE
    )

    rgb_smoothed = finalize_rgb(
        np.stack(
            [
                normalize(red, vmin_smoothed, vmax_smoothed, GAMMA),
                normalize(green, vmin_smoothed, vmax_smoothed, GAMMA),
                normalize(blue, vmin_smoothed, vmax_smoothed, GAMMA),
            ],
            axis=-1,
        )
    )

    wcs_2d = WCS(header).celestial
    spectral_values = spectral_values_from_header(header, nchan)
    if spectral_values is None:
        direct_labels = make_rgb_labels(channels=(center - 1, center, center + 1))
        smoothed_labels = make_rgb_labels(channels=(f"{i}-{i+1}", f"{i+1}-{i+2}", f"{i+2}-{i+3}"))
        save_rgb_figure(OUTPUT_DIRECT_PNG_PATH, rgb_direct, wcs_2d, direct_labels, pb)
        save_rgb_figure(OUTPUT_SMOOTHED_PNG_PATH, rgb_smoothed, wcs_2d, smoothed_labels, pb)
        save_comparison_figure(
            OUTPUT_COMPARISON_PNG_PATH,
            rgb_direct,
            rgb_smoothed,
            wcs_2d,
            direct_labels,
            smoothed_labels,
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
        save_rgb_figure(OUTPUT_DIRECT_PNG_PATH, rgb_direct, wcs_2d, direct_labels, pb)
        save_rgb_figure(OUTPUT_SMOOTHED_PNG_PATH, rgb_smoothed, wcs_2d, smoothed_labels, pb)
        save_comparison_figure(
            OUTPUT_COMPARISON_PNG_PATH,
            rgb_direct,
            rgb_smoothed,
            wcs_2d,
            direct_labels,
            smoothed_labels,
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
