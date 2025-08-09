import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
from reproject import reproject_interp
import pywph as pw
import torch
import pytorch_finufft

from ivis.io import DataProcessor
from ivis.logger import logger
from ivis.utils import dutils, mod_loss, fourier
from ivis.models import ClassicIViS3D, TWiSTModel
from ivis.imager import Imager3D 

plt.ion()

# -------------------
# Paths and WCS
# -------------------
path_ms = "../docs/tutorials/data_tutorials/msdir2/"
path_beams = "../docs/tutorials/data_tutorials/ivis_data/BEAMS/"
path_sd = None
pathout = "../docs/tutorials/data_tutorials/ivis_data/"

filename = "../docs/tutorials/data_tutorials/ivis_data/MW-C10_mom0th_NHI.fits"
target_header = fits.open(filename)[0].header
shape = (target_header["NAXIS2"], target_header["NAXIS1"])

# -------------------
# Prepare DataProcessor
# -------------------
data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

# Optional: pre-compute PB and grid
# logger.disabled = True
# data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits")
# logger.disabled = False

pb, grid = data_processor.read_pb_and_grid(
    fitsname_pb="reproj_pb.fits",
    fitsname_grid="grid_interp.fits"
)

# Dummy single-dish array and beam
sd = np.zeros(shape, dtype=np.float32)
beam_sd = Beam(1 * u.deg, 1 * u.deg, 1.e-12 * u.deg)

# -------------------
# Read visibilities into VisIData dataclass
# -------------------
vis_data = data_processor.read_vis_visidata(
    uvmin=0.0,
    uvmax=np.inf,
    # target_channel=0,
    chan_sel=slice(0, 3),
    keep_autocorr=False,
    prefer_weight_spectrum=False,
)

# -------------------
# User parameters
# -------------------
max_its = 20
lambda_sd = 0
lambda_r = 1
device = 0        # 0 for GPU, "cpu" for CPU
positivity = False
init_params = np.zeros((3, shape[0], shape[1]), dtype=np.float32)

# -------------------
# Create Imager3D
# -------------------
image_processor = Imager3D(
    vis_data=vis_data,
    pb=pb,
    grid=grid,
    sd=sd,
    beam_sd=beam_sd,
    hdr=target_header,
    init_params=init_params,
    max_its=max_its,
    lambda_sd=lambda_sd,
    positivity=positivity,
    device=device,
    beam_workers=1
)

# -------------------
# Choose model
# -------------------
model = ClassicIViS3D(lambda_r=lambda_r, Nw=0)

# -------------------
# Run optimization
# -------------------
base = image_processor.process(model=model, units="Jy/arcsec^2")

# # Save or inspect result
# fits.writeto(pathout + "ivis_reconstruction.fits", base, target_header, overwrite=True)
