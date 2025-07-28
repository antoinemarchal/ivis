import glob
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
from reproject import reproject_interp
from astropy.modeling import models, fitting

from ivis.io import DataProcessor
from ivis.imager import Imager
from ivis.logger import logger
from ivis.utils import dutils, mod_loss

plt.ion()

path_ms = "/Users/antoine/Desktop/ivis_data_1beam/msl_mw/" #directory of measurement sets    
path_beams = "/Users/antoine/Desktop/ivis_data_1beam/BEAMS/" #directory of primary beams
path_sd = "/Users/antoine/Desktop/ivis_data_1beam/" #path single-dish data
pathout = "/Users/antoine/Desktop/ivis_data_1beam/" #path where data will be packaged and stored

#REF WCS INPUT USER
filename = "/Users/antoine/Desktop/ivis_data/MW-C10_mom0th_NHI.fits"
target_header = fits.open(filename)[0].header
shape = (target_header["NAXIS2"],target_header["NAXIS1"])
    
#create data processor
data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

# pre-compute pb and interpolation grids — this can be commented after first compute
logger.disabled = True
data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits") 
logger.disabled = False

pb, grid = data_processor.read_pb_and_grid(fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits")

#Dummy sd array
sd = np.zeros(shape)
#Dummy Beam sd
beam_sd = Beam(1*u.deg, 1*u.deg, 1.e-12*u.deg)

#Read data
vis_data = data_processor.read_vis_from_scratch(uvmin=0, uvmax=np.inf,
                                                target_frequency=None,
                                                target_channel=0,
                                                extension=".ms",
                                                blocks='single',
                                                max_workers=1)

#Inject 1 Jy source in one pixel
cell_size = (np.abs(target_header["CDELT2"]) *u.deg).to(u.arcsec).value
sky_model = np.zeros(shape, dtype=np.float32)
sky_model[shape[0] // 2, shape[1] // 2] = 1.0 / (cell_size ** 2)  # Jy / arcsec^2

#Model visibilities with IVis forward single frequency model
image_processor = Imager(vis_data,      # visibilities
                         pb,            # array of primary beams
                         grid,          # array of interpolation grids
                         None,            # single dish data in unit of Jy/arcsec^2
                         None,       # beam of single-dish data in radio_beam format
                         target_header, # header on which to image the data
                         sky_model,   # init array of parameters
                         0,       # maximum number of iterations
                         0,     # hyper-parameter single-dish
                         0,      # hyper-parameter regularization
                         False,    # impose a positivity constaint
                         0,        # device: 0 is GPU; "cpu" is CPU
                         beam_workers=1)

model_vis =  image_processor.forward_model()

#Add noise
# model_vis: complex64 array of model visibilities
# vis_data.sigma: float32 array of noise std dev per visibility (same shape as model_vis)
noise_real = np.random.normal(loc=0.0, scale=vis_data.sigma)
noise_imag = np.random.normal(loc=0.0, scale=vis_data.sigma)
noise = noise_real + 1j * noise_imag

# Add realistic thermal noise to the model visibilities
vis_data.data = model_vis + noise

init_params = np.zeros(shape, dtype=np.float32)

#user parameters
max_its = 40
lambda_sd = 0
lambda_r = 0.
device = 0#"cpu" #0 is GPU and "cpu" is CPU
positivity = True

#create image processor
image_processor = Imager(vis_data,      # visibilities
                         pb,            # array of primary beams
                         grid,          # array of interpolation grids
                         sd,            # single dish data in unit of Jy/arcsec^2
                         beam_sd,       # beam of single-dish data in radio_beam format
                         target_header, # header on which to image the data
                         init_params,   # init array of parameters
                         max_its,       # maximum number of iterations
                         lambda_sd,     # hyper-parameter single-dish
                         lambda_r,      # hyper-parameter regularization
                         positivity,    # impose a positivity constaint
                         device,        # device: 0 is GPU; "cpu" is CPU
                         beam_workers=1)
#get image
result = image_processor.process(units="Jy/arcsec^2") #"Jy/arcsec^2" or "K"

#mean pb
filenames = sorted(glob.glob(path_beams+"*.fits"))
n_beams = len(filenames)
pb_all = np.zeros((n_beams,result.shape[0],result.shape[1]))
w = dutils.wcs2D(target_header)
for i in tqdm(np.arange(n_beams)):
    #open beam cube
    hdu_pb = fits.open(filenames[i])
    hdr_pb = hdu_pb[0].header
    pb2 = hdu_pb[0].data
    pb2[pb2 != pb2] = 0.
    w_pb = dutils.wcs2D(hdr_pb)
    pb2, footprint = reproject_interp((pb2,w_pb.to_header()), w.to_header(), shape)
    pb2[pb2 != pb2] = 0.
    pb_all += pb2
    pb_mean = np.nanmean(pb_all,0)
    pb_mean /= np.nanmax(pb_mean)    
    mask = np.where(pb_mean > 0.2, 1, np.nan)

#PLOT RESULT
fig = plt.figure(figsize=(5, 5))
ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
ax.set_xlabel(r"RA (deg)", fontsize=18.)
ax.set_ylabel(r"DEC (deg)", fontsize=18.)
img = ax.imshow(result*mask, vmin=0, vmax=1.e-3, origin="lower")
ax.contour(pb_mean, linestyles="--", levels=[0.2, 0.3], colors=["w","w"])
colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
cbar = fig.colorbar(img, cax=colorbar_ax)
cbar.ax.tick_params(labelsize=14.)
cbar.set_label(r"$I$ (Jy/arcsec$^{2})$", fontsize=18.)
#    plt.savefig(pathout + 'ivis_result_cloud_MeerKAT.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)

#Zoom in and calculate restoring beam
ny, nx = shape
cy, cx = ny // 2, nx // 2  # center coordinates

zoom = 20  # pixels around center
cutout = result[cy - zoom:cy + zoom + 1, cx - zoom:cx + zoom + 1]

plt.imshow(cutout, origin="lower", vmin=0, vmax=1.e-2)

flux, Bmaj, Bmin, theta = dutils.fit_elliptical_gaussian(cutout, pixel_scale_arcsec=cell_size)

