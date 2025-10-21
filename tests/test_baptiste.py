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
from astropy.coordinates import SkyCoord

from ivis.io import DataProcessor
from ivis.readers import CasacoreReader
from ivis.models import ClassicIViS3D
from ivis.imager import Imager3D
from ivis.types import VisIData
from ivis.logger import logger

plt.ion()

# -------------------
# Paths and WCS
# -------------------
path_ms = "../docs/tutorials/data_tutorials/baptiste/"
path_beams = "../docs/tutorials/data_tutorials/baptiste/m101_fits/" #FIXME
path_sd = None
pathout = "../docs/tutorials/data_tutorials/baptiste/"

filename = "../docs/tutorials/data_tutorials/baptiste/m101_fits/m101_optical.fits"
target_header = fits.open(filename)[0].header
shape = (target_header["NAXIS2"], target_header["NAXIS1"])

# -------------------
# Prepare DataProcessor
# -------------------
data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

# # Optional: pre-compute PB and grid
# data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits")

pb, grid = data_processor.read_pb_and_grid(
    fitsname_pb="reproj_pb.fits",
    fitsname_grid="grid_interp.fits"
)
pb = np.ones(pb.shape)

# Dummy single-dish array and beam
sd = np.zeros(shape, dtype=np.float32)
beam_sd = Beam(1 * u.deg, 1 * u.deg, 1.e-12 * u.deg)

# -------------------
# Read visibilities into VisIData dataclass
# -------------------
reader = CasacoreReader(
    prefer_weight_spectrum=False,
    keep_autocorr=False,
    n_workers=4)

I: VisIData = reader.read_blocks_I(
    path_ms,
    uvmin=0.0,
    uvmax=np.inf,
    chan_sel=slice(0, 1),
    rest_freq=1.42040575177e9, #HI rest frequency in Hz
)

# -------------------
# User parameters
# -------------------
max_its = 25
lambda_sd = 0
lambda_r = 1
cost_device = 0        # 0 for GPU, "cpu" for CPU
optim_device = 0        # 0 for GPU, "cpu" for CPU
positivity = True
init_params = np.zeros((1, shape[0], shape[1]), dtype=np.float32)

# -------------------
# Create Imager3D
# -------------------
image_processor = Imager3D(
    vis_data=I,
    pb=pb,
    grid=grid,
    sd=sd,
    beam_sd=beam_sd,
    hdr=target_header,
    init_params=init_params,
    max_its=max_its,
    lambda_sd=lambda_sd,
    positivity=positivity,
    cost_device=cost_device,
    optim_device=optim_device,
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
