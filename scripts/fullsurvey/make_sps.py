import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c, k_B
from astropy.io import fits


ASKAP_DIR = "/Users/antoine/Desktop/IVIS_paper/ASKAP"
PB_PATH = os.path.join(ASKAP_DIR, "output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR_PB_eff.fits")
JOINT_PATH = os.path.join(ASKAP_DIR, "output_chan_795_1_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits")
JOINT_SS_PATH = os.path.join(ASKAP_DIR, "output_chan_795_1_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_short_spacing.fits")
JOINT_SD_PATH = os.path.join(ASKAP_DIR, "output_chan_795_1_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_sd_regrid.fits")
LINEAR_PATH = os.path.join(ASKAP_DIR, "output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR.fits")
LINEAR_SS_PATH = os.path.join(ASKAP_DIR, "output_chan_795_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_LINEAR_short_spacing.fits")
LOW_VEL_JOINT_PATH = os.path.join(ASKAP_DIR, "output_chan_1270_vel_6.4905_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0.fits")
LOW_VEL_JOINT_SS_PATH = os.path.join(ASKAP_DIR, "output_chan_1270_vel_6.4905_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_short_spacing.fits")
LOW_VEL_JOINT_SD_PATH = os.path.join(ASKAP_DIR, "output_chan_1270_vel_6.4905_2blocks_7arcsec_lambda_r_1_positivity_true_iter_20_Nw_0_sd_regrid.fits")
OUTPUT_PATH = os.path.join(ASKAP_DIR, "SPS_ASKAP.png")
OUTPUT_PATH_LINEAR = os.path.join(ASKAP_DIR, "SPS_ASKAP_linear.png")
OUTPUT_PATH_LOW_VEL = os.path.join(ASKAP_DIR, "SPS_ASKAP_low_vel.png")
PRODUCTS_DIR = os.path.join(ASKAP_DIR, "products")

NU_HZ = 1.42040575177e9
LAMBDA_M = 0.211
ARCMIN_PER_RAD = 3437.75
SD_MAX_D_M = 100.0
SUBIMAGE_FRACTION = 0.6


sys.path.insert(0, os.path.join(ASKAP_DIR, "SMC_power_spec"))

import marchalib as ml  # noqa: E402


matplotlib.rc("xtick", labelsize=16)
matplotlib.rc("ytick", labelsize=16)


def first_plane(data):
    return np.asarray(data[0] if data.ndim == 3 else data, dtype=float)


def k_to_jy_arcsec2(data_k, nu_hz):
    intensity = (2 * k_B * (nu_hz * u.Hz) ** 2 / c**2) * (np.asarray(data_k) * u.K) / u.sr
    return intensity.to(u.Jy / u.arcsec**2, equivalencies=u.dimensionless_angles()).value


def load_image(path):
    with fits.open(path) as hdul:
        data = first_plane(hdul[0].data)
        header = hdul[0].header.copy()
    if "LINEAR" in os.path.basename(path) and header.get("BUNIT") != "Jy / arcsec2":
        data = k_to_jy_arcsec2(data, NU_HZ)
    return data, header


def centered_subimage(data, fraction):
    ny, nx = data.shape
    sub_ny = max(1, int(ny * fraction))
    sub_nx = max(1, int(nx * fraction))
    y0 = (ny - sub_ny) // 2
    x0 = (nx - sub_nx) // 2
    return data[y0:y0 + sub_ny, x0:x0 + sub_nx]


def compute_sps(data, pb_eff, header):
    shape = data.shape
    tapper = ml.edges.apodize(0.95, shape)
    field = data * pb_eff
    field_zm = field - np.mean(field)
    field_apod = field_zm * tapper
    return ml.powspec(field_apod, reso=(header["CDELT2"] * u.deg).to(u.arcmin).value)


def k_to_D(k):
    return LAMBDA_M * ARCMIN_PER_RAD * k


def D_to_k(D):
    return D / (LAMBDA_M * ARCMIN_PER_RAD)


def save_products(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


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
    ax.set_xlabel(r"$k$ (arcmin$^{-1}$)", fontsize=16)
    ax.set_ylabel(r"$P(k)$ [(Jy arcsec$^{-2}$)$^2$]", fontsize=16)
    ax.set_ylim(ylim)
    secax = ax.secondary_xaxis("top", functions=(k_to_D, D_to_k))
    secax.set_xlabel(r"$D$ (m)", fontsize=16)
    ax.legend()
    plt.savefig(output_path, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PRODUCTS_DIR, exist_ok=True)

    joint, header = load_image(JOINT_PATH)
    joint_ss, _ = load_image(JOINT_SS_PATH)
    joint_sd, _ = load_image(JOINT_SD_PATH)
    linear, _ = load_image(LINEAR_PATH)
    linear_ss, _ = load_image(LINEAR_SS_PATH)
    low_vel_joint, _ = load_image(LOW_VEL_JOINT_PATH)
    low_vel_joint_ss, _ = load_image(LOW_VEL_JOINT_SS_PATH)
    low_vel_joint_sd, _ = load_image(LOW_VEL_JOINT_SD_PATH)

    with fits.open(PB_PATH) as hdul:
        pb_eff = np.asarray(hdul[0].data, dtype=float)
    pb_eff /= np.nanmax(pb_eff)
    pb_eff = centered_subimage(pb_eff, SUBIMAGE_FRACTION)

    joint = centered_subimage(joint, SUBIMAGE_FRACTION)
    joint_ss = centered_subimage(joint_ss, SUBIMAGE_FRACTION)
    joint_sd = centered_subimage(joint_sd, SUBIMAGE_FRACTION)
    linear = centered_subimage(linear, SUBIMAGE_FRACTION)
    linear_ss = centered_subimage(linear_ss, SUBIMAGE_FRACTION)
    low_vel_joint = centered_subimage(low_vel_joint, SUBIMAGE_FRACTION)
    low_vel_joint_ss = centered_subimage(low_vel_joint_ss, SUBIMAGE_FRACTION)
    low_vel_joint_sd = centered_subimage(low_vel_joint_sd, SUBIMAGE_FRACTION)

    ks_sd, sps1d_sd = compute_sps(joint_sd, pb_eff, header)
    ks_joint, sps1d_joint = compute_sps(joint, pb_eff, header)
    sd_mask = k_to_D(ks_sd) <= SD_MAX_D_M
    ks_linear, sps1d_linear = compute_sps(linear, pb_eff, header)
    ks_joint_ss, sps1d_joint_ss = compute_sps(joint_ss, pb_eff, header)
    ks_linear_ss, sps1d_linear_ss = compute_sps(linear_ss, pb_eff, header)
    ks_low_vel_sd, sps1d_low_vel_sd = compute_sps(low_vel_joint_sd, pb_eff, header)
    ks_low_vel_joint, sps1d_low_vel_joint = compute_sps(low_vel_joint, pb_eff, header)
    ks_low_vel_joint_ss, sps1d_low_vel_joint_ss = compute_sps(low_vel_joint_ss, pb_eff, header)
    low_vel_sd_mask = k_to_D(ks_low_vel_sd) <= SD_MAX_D_M

    save_products(
        os.path.join(PRODUCTS_DIR, "sps_askap_joint.npz"),
        ks_sd=ks_sd,
        sps1d_sd=sps1d_sd,
        sd_mask=sd_mask,
        ks_joint=ks_joint,
        sps1d_joint=sps1d_joint,
        ks_joint_ss=ks_joint_ss,
        sps1d_joint_ss=sps1d_joint_ss,
    )
    save_products(
        os.path.join(PRODUCTS_DIR, "sps_askap_linear.npz"),
        ks_sd=ks_sd,
        sps1d_sd=sps1d_sd,
        sd_mask=sd_mask,
        ks_linear=ks_linear,
        sps1d_linear=sps1d_linear,
        ks_linear_ss=ks_linear_ss,
        sps1d_linear_ss=sps1d_linear_ss,
    )
    save_products(
        os.path.join(PRODUCTS_DIR, "sps_askap_low_vel.npz"),
        ks_low_vel_sd=ks_low_vel_sd,
        sps1d_low_vel_sd=sps1d_low_vel_sd,
        low_vel_sd_mask=low_vel_sd_mask,
        ks_low_vel_joint=ks_low_vel_joint,
        sps1d_low_vel_joint=sps1d_low_vel_joint,
        ks_low_vel_joint_ss=ks_low_vel_joint_ss,
        sps1d_low_vel_joint_ss=sps1d_low_vel_joint_ss,
    )

    plot_sps(
        OUTPUT_PATH,
        [
            {
                "ks": ks_sd[sd_mask],
                "sps": sps1d_sd[sd_mask],
                "color": "green",
                "linestyle": "-",
                "linewidth": 2,
                "label": "SD regrid",
            },
            {
                "ks": ks_joint,
                "sps": sps1d_joint,
                "color": "black",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint deconvolution",
            },
            {
                "ks": ks_joint_ss,
                "sps": sps1d_joint_ss,
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
                "ks": ks_sd[sd_mask],
                "sps": sps1d_sd[sd_mask],
                "color": "green",
                "linestyle": "-",
                "linewidth": 2,
                "label": "SD regrid",
            },
            {
                "ks": ks_linear,
                "sps": sps1d_linear,
                "color": "k",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Linear mosaicking",
            },
            {
                "ks": ks_linear_ss,
                "sps": sps1d_linear_ss,
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
                "ks": ks_low_vel_sd[low_vel_sd_mask],
                "sps": sps1d_low_vel_sd[low_vel_sd_mask],
                "color": "green",
                "linestyle": "-",
                "linewidth": 2,
                "label": "SD regrid",
            },
            {
                "ks": ks_low_vel_joint,
                "sps": sps1d_low_vel_joint,
                "color": "black",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint deconvolution",
            },
            {
                "ks": ks_low_vel_joint_ss,
                "sps": sps1d_low_vel_joint_ss,
                "color": "red",
                "linestyle": "None",
                "marker": ".",
                "markersize": 8.0,
                "label": "Joint + short spacing",
            },
        ],
        ylim=[1.0e-14, 1.0e-2],
    )
