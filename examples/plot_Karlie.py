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

import marchalib as ml #remove

plt.ion()

#Open data
fitsname = "/home/amarchal/Projects/deconv/examples/data/MeerKAT/result_chan_0800_to_1100.fits"
hdu = fits.open(fitsname)
hdr = hdu[0].header
cube = hdu[0].data[140:]

#REF WCS INPUT USER
filename = "/priv/avatar/amarchal/Projects/deconv/examples/data/MeerKAT/MW-C10_mom0th_NHI.fits"
target_header = fits.open(filename)[0].header

#mean pb
path_beams = "/priv/avatar/amarchal/Projects/deconv/examples/data/MeerKAT/BEAMS/" #directory of 
filenames = sorted(glob.glob(path_beams+"*.fits"))
n_beams = len(filenames)
pb_all = np.zeros((n_beams,cube.shape[1],cube.shape[2]))
w = ml.wcs2D(target_header)
shape_out = (target_header["NAXIS2"],target_header["NAXIS1"])
for i in tqdm(np.arange(n_beams)):
    #open beam cube
    hdu_pb = fits.open(filenames[i])
    hdr_pb = hdu_pb[0].header
    pb2 = hdu_pb[0].data
    pb2[pb2 != pb2] = 0.
    shape = (hdr_pb["NAXIS2"],hdr_pb["NAXIS1"])
    w_pb = ml.wcs2D(hdr_pb)
    pb2, footprint = reproject_interp((pb2,w_pb.to_header()), w.to_header(), shape_out)
    pb2[pb2 != pb2] = 0.
    pb_all += pb2
    pb_mean = np.nanmean(pb_all,0)
    pb_mean /= np.nanmax(pb_mean)    
    mask = np.where(pb_mean > 0.05, 1, np.nan)
    
w_img = ml.wcs2D(target_header)

# Path output
pathout="/home/amarchal/Projects/deconv/examples/data/MeerKAT/MeerKAT_shared/"

# Create a video writer
writer = imageio.get_writer(pathout + 'movie.mp4', fps=10)

# Setup for plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0.1, 0.1, 0.78, 0.8], projection=w_img)
ax.set_xlabel(r"RA (deg)", fontsize=18.)
ax.set_ylabel(r"DEC (deg)", fontsize=18.)

# Loop through the cube to generate each frame and write it to the video
for i in range(len(cube)):
    ax.imshow(cube[i] * mask, vmin=-2, vmax=6, origin="lower", cmap="viridis")
    ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w", "w"])

    # Add colorbar if needed (optional, you could set it once if static)
    if i == 0:
        colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
        cbar = fig.colorbar(ax.imshow(cube[i] * mask, vmin=-4, vmax=8, origin="lower", cmap="viridis"), cax=colorbar_ax)
        cbar.ax.tick_params(labelsize=14.)
        cbar.set_label(r"$T_b$ (K)", fontsize=18.)

    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, dpi=200)
    buf.seek(0)

    # Convert the image to an array and write to the video
    image = np.array(plt.imread(buf))
    writer.append_data(image)
    ax.clear()  # Clear axes for the next frame

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
