# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
from astropy.constants import c
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
import time

plt.ion()

#PLOT RESULT
pathout = "/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/" 

fname = "output_1blocks_7arcsec_lambda_r_1_positivity_false_iter_50.fits"
hdul = fits.open(pathout+fname)
hdr = hdul[0].header
result = hdul[0].data
hdul.close()

fname = "mask_7arcsec.fits"
hdul = fits.open(pathout+fname)
hdr = hdul[0].header
pb_mean_full = hdul[0].data
hdul.close()

#compute mask
mask = np.where(pb_mean_full > 0.05, 1, np.nan)

# 2D celestial WCS from header
w = wcs.WCS(hdr).celestial    # or WCS(hdr, naxis=2)

#write plot on disk
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
ax.set_xlabel(r"RA (deg)", fontsize=18.)
ax.set_ylabel(r"DEC (deg)", fontsize=18.)
vmin, vmax = np.nanpercentile(result[0], (0.01, 99.99))
img = ax.imshow(result[0]*mask, vmin=-0.00010, vmax=0.00015, origin="lower", cmap="inferno")
ax.contour(pb_mean_full, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
cbar = fig.colorbar(img, cax=colorbar_ax)
cbar.ax.tick_params(labelsize=14.)
cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
plt.savefig("./plots/output_1blocks_7arcsec_lambda_r_1_positivity_false_iter_50.png", format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)

