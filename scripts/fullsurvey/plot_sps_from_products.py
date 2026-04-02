import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ASKAP_DIR = "/Users/antoine/Desktop/IVIS_paper/ASKAP"
PRODUCTS_DIR = os.path.join(ASKAP_DIR, "products")
OUTPUT_PATH = os.path.join(ASKAP_DIR, "SPS_ASKAP.png")
OUTPUT_PATH_LINEAR = os.path.join(ASKAP_DIR, "SPS_ASKAP_linear.png")
OUTPUT_PATH_LOW_VEL = os.path.join(ASKAP_DIR, "SPS_ASKAP_low_vel.png")

LAMBDA_M = 0.211
ARCMIN_PER_RAD = 3437.75
FIT_TARGET_D_M = 1000.0
FIT_D_RANGE_M = (600.0, 1600.0)


matplotlib.rc("xtick", labelsize=16)
matplotlib.rc("ytick", labelsize=16)


def k_to_D(k):
    return LAMBDA_M * ARCMIN_PER_RAD * k


def D_to_k(D):
    return D / (LAMBDA_M * ARCMIN_PER_RAD)


def load_products(filename):
    return np.load(os.path.join(PRODUCTS_DIR, filename))


def gaussian_beam_power(ks, fwhm_arcmin, amplitude):
    sigma_arcmin = fwhm_arcmin / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amplitude * np.exp(-4.0 * (np.pi**2) * (sigma_arcmin**2) * (ks**2))


def fit_gaussian_to_series(ks, sps):
    ks = np.asarray(ks, dtype=float)
    sps = np.asarray(sps, dtype=float)
    Ds = k_to_D(ks)
    fit_mask = (
        np.isfinite(ks)
        & np.isfinite(sps)
        & (ks > 0.0)
        & (sps > 0.0)
        & (Ds >= FIT_D_RANGE_M[0])
        & (Ds <= FIT_D_RANGE_M[1])
    )
    if not np.any(fit_mask):
        raise ValueError("No valid points available to fit the Gaussian beam.")

    ks_fit = ks[fit_mask]
    sps_fit = sps[fit_mask]
    Ds_fit = Ds[fit_mask]
    anchor_idx = np.argmin(np.abs(Ds_fit - FIT_TARGET_D_M))
    anchor_k = ks_fit[anchor_idx]
    anchor_sps = sps_fit[anchor_idx]

    fwhm_grid_arcsec = np.linspace(5.0, 120.0, 2000)
    best = None
    best_error = np.inf
    for fwhm_arcsec in fwhm_grid_arcsec:
        fwhm_arcmin = fwhm_arcsec / 60.0
        anchor_model = gaussian_beam_power(np.array([anchor_k]), fwhm_arcmin, 1.0)[0]
        amplitude = anchor_sps / anchor_model
        model = gaussian_beam_power(ks_fit, fwhm_arcmin, amplitude)
        error = np.mean((np.log10(model) - np.log10(sps_fit)) ** 2)
        if error < best_error:
            best_error = error
            best = (fwhm_arcmin, amplitude)
    return best


def plot_sps(output_path, series, ylim):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for item in series:
        ax.plot(
            item["ks"],
            item["sps"],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=item.get("linewidth"),
            marker=item.get("marker"),
            markersize=item.get("markersize"),
            label=item["label"],
        )
    ks_all = np.concatenate([np.asarray(item["ks"], dtype=float) for item in series])
    ks_ref = np.logspace(np.log10(np.nanmin(ks_all)), np.log10(np.nanmax(ks_all)), 512)
    red_series = next((item for item in series if "short spacing" in item["label"].lower()), None)
    if red_series is None:
        raise ValueError("Could not identify the short-spacing series for Gaussian fitting.")
    fitted_fwhm_arcmin, fitted_amplitude = fit_gaussian_to_series(red_series["ks"], red_series["sps"])
    ax.plot(
        ks_ref,
        gaussian_beam_power(ks_ref, fitted_fwhm_arcmin, fitted_amplitude),
        color="blue",
        linestyle="--",
        linewidth=2.0,
        label=rf"Gaussian {fitted_fwhm_arcmin * 60.0:.1f}$''$",
    )
    ax.set_xlabel(r"$k$ (arcmin$^{-1}$)", fontsize=16)
    ax.set_ylabel(r"$P(k)$ [(Jy arcsec$^{-2}$)$^2$]", fontsize=16)
    ax.set_ylim(ylim)
    secax = ax.secondary_xaxis("top", functions=(k_to_D, D_to_k))
    secax.set_xlabel(r"$D$ (m)", fontsize=16)
    ax.legend()
    plt.savefig(output_path, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"{os.path.basename(output_path)} effective beam FWHM: {fitted_fwhm_arcmin * 60.0:.2f} arcsec")


if __name__ == "__main__":
    joint = load_products("sps_askap_joint.npz")
    linear = load_products("sps_askap_linear.npz")
    low_vel = load_products("sps_askap_low_vel.npz")

    plot_sps(
        OUTPUT_PATH,
        [
            {
                "ks": joint["ks_sd"][joint["sd_mask"]],
                "sps": joint["sps1d_sd"][joint["sd_mask"]],
                "color": "green",
                "linestyle": "-",
                "linewidth": 2,
                "label": "SD regrid",
            },
            {
                "ks": joint["ks_joint"],
                "sps": joint["sps1d_joint"],
                "color": "black",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint deconvolution",
            },
            {
                "ks": joint["ks_joint_ss"],
                "sps": joint["sps1d_joint_ss"],
                "color": "red",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint + short spacing",
            },
        ],
        ylim=[1.0e-14, 1.0e-2],
    )

    plot_sps(
        OUTPUT_PATH_LINEAR,
        [
            {
                "ks": linear["ks_sd"][linear["sd_mask"]],
                "sps": linear["sps1d_sd"][linear["sd_mask"]],
                "color": "green",
                "linestyle": "-",
                "linewidth": 2,
                "label": "SD regrid",
            },
            {
                "ks": linear["ks_linear"],
                "sps": linear["sps1d_linear"],
                "color": "k",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Linear mosaicking",
            },
            {
                "ks": linear["ks_linear_ss"],
                "sps": linear["sps1d_linear_ss"],
                "color": "r",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Linear + short spacing",
            },
        ],
        ylim=[1.0e-14, 1.0e-2],
    )

    plot_sps(
        OUTPUT_PATH_LOW_VEL,
        [
            {
                "ks": low_vel["ks_low_vel_sd"][low_vel["low_vel_sd_mask"]],
                "sps": low_vel["sps1d_low_vel_sd"][low_vel["low_vel_sd_mask"]],
                "color": "green",
                "linestyle": "-",
                "linewidth": 2,
                "label": "SD regrid",
            },
            {
                "ks": low_vel["ks_low_vel_joint"],
                "sps": low_vel["sps1d_low_vel_joint"],
                "color": "black",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint deconvolution",
            },
            {
                "ks": low_vel["ks_low_vel_joint_ss"],
                "sps": low_vel["sps1d_low_vel_joint_ss"],
                "color": "red",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint + short spacing",
            },
        ],
        ylim=[1.0e-14, 1.0e-2],
    )
