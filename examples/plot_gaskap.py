import glob
import os
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
from reproject import reproject_interp
from tqdm import tqdm as tqdm
import imageio
from io import BytesIO
from PIL import Image  # For JPEG conversion
from scipy.ndimage import gaussian_filter

import marchalib as ml #remove

plt.ion()

#Open data
fitsname = "/home/amarchal/Projects/deconv/examples/data/ASKAP/ASKAP_shared/result_chan_0874_to_1079_03_pbmask.fits"
hdu = fits.open(fitsname)
hdr = hdu[0].header
cube = hdu[0].data

#REF WCS INPUT USER
cfield = SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs')
filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
target_header = fits.open(filename)[0].header
target_header["CRVAL1"] = cfield.ra.value
target_header["CRVAL2"] = cfield.dec.value

#Open PB file per antenna
path = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/"
hdu_pb = fits.open(path+"effpb.fits")
hdr_pb = hdu_pb[0].header
effpb = hdu_pb[0].data    
effpb /= np.max(effpb)    
mask = np.where(effpb > 0.05, 1, np.nan)

w_img = ml.wcs2D(target_header)

# Path output
pathout="/home/amarchal/Projects/deconv/examples/data/ASKAP/ASKAP_shared/"

# Create a video writer
writer = imageio.get_writer(pathout + 'movie_askap.mp4', fps=10)

# Setup for plotting (only done once)
fig = plt.figure(figsize=(10, 10), dpi=200)
ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=w_img)
ax.set_xlabel(r"RA (deg)", fontsize=18.)
ax.set_ylabel(r"DEC (deg)", fontsize=18.)

# Precompute the contour once before the loop (as it doesn't change)
contours = ax.contour(effpb, linestyles="--", levels=[0.2, 0.3], colors=["w", "w"])

# Create the colorbar once before the loop
img = ax.imshow(cube[0], vmin=-25, vmax=35, origin="lower", cmap="inferno")
colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
cbar = fig.colorbar(img, cax=colorbar_ax)
cbar.ax.tick_params(labelsize=14.)
cbar.set_label(r"$T_b$ (K)", fontsize=18.)

# Write to the video file
for i in tqdm(np.arange(len(cube))):
    # Update only the image data, no need to recreate or redraw the colorbar or contours
    img.set_data(cube[i])
    
    # Save the figure to a BytesIO buffer (avoid plt.savefig overhead)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    buf.seek(0)

    # Convert image to an array and write to video
    image = np.array(plt.imread(buf))
    writer.append_data(image)

    buf.close()  # Free memory

# Close the writer to finalize the video
writer.close()

stop

# Setup for plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=w_img)
ax.set_xlabel(r"RA (deg)", fontsize=18.)
ax.set_ylabel(r"DEC (deg)", fontsize=18.)

# Loop through the cube to generate each frame and write it to the video
for i in tqdm(np.arange(len(cube))):
    img = ax.imshow(cube[i]*mask, vmin=-2, vmax=7, origin="lower", cmap="viridis")
    ax.contour(pb_mean, linestyles="--", levels=[0.2, 0.3], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (K)", fontsize=18.)
    
    # ax.imshow(cube[i] * mask, vmin=-2, vmax=7, origin="lower", cmap="viridis")
    # ax.contour(pb_mean, linestyales="--", levels=[0.2, 0.3], colors=["w", "w"])

    # colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    # cbar = fig.colorbar(ax.imshow(cube[i] * mask, vmin=-2, vmax=7, origin="lower", cmap="viridis"), cax=colorbar_ax)
    # cbar.ax.tick_params(labelsize=14.)
    # cbar.set_label(r"$T_b$ (K)", fontsize=18.)
    
    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, dpi=100)
    buf.seek(0)

    # Convert the image to an array and write to the video
    image = np.array(plt.imread(buf))
    writer.append_data(image)
    # ax.clear()  # Clear axes for the next frame

# Close the writer to finalize the video
writer.close()

    

# #PLOT RESULT
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w_img)
# ax.set_xlabel(r"RA (deg)", fontsize=18.)
# ax.set_ylabel(r"DEC (deg)", fontsize=18.)
# img = ax.imshow(cube[140]*mask, vmin=-4, vmax=8, origin="lower", cmap="viridis")
# ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
# colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
# cbar = fig.colorbar(img, cax=colorbar_ax)
# cbar.ax.tick_params(labelsize=14.)
# cbar.set_label(r"$T_b$ (K)", fontsize=18.)
# plt.savefig(pathout + 'deconv_result_cloud_MeerKAT_GBT.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)    
