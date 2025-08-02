import glob
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
from reproject import reproject_interp
import pywph as pw
import torch

from ivis.io import DataProcessor
from ivis.imager import Imager
from ivis.logger import logger
from ivis.utils import dutils, mod_loss, fourier

from ivis.models import ClassicIViS, TWiSTModel

path_ms = "../docs/tutorials/data_tutorials/ivis_data/msl_mw/" #directory of measurement sets    
path_beams = "../docs/tutorials/data_tutorials/ivis_data/BEAMS/" #directory of primary beams
path_sd = None #path single-dish data
pathout = "../docs/tutorials/data_tutorials/ivis_data/" #path where data will be packaged and stored

#REF WCS INPUT USER
filename = "../docs/tutorials/data_tutorials/ivis_data/MW-C10_mom0th_NHI.fits"
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

## WPH noise stat (start from pre-computed noise cube here for testing)
device = 0

#params WPH
logger.info("Get WPH operator and load moments model.")
M, N = shape # map size
J = int(np.log2(min(M, N)))-2 # number of scales
L = 4 # number of angles
pbc = False # periodic boundary conditions
dn = 5 # number of translations
# wph_model = ["S11","S00","S01","Cphase","C01","C00","L"] # list of WPH coefficients
wph_model = ["S11"] # list of WPH coefficients
logger.warning("Only using S11.")
# get operator
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
wph_op.load_model(wph_model)

# Open noise data cube
with fits.open(pathout + "noise_cube.fits", memmap=True) as hdul:
    data = hdul[0].data
    if not data.dtype.isnative:
        data = data.byteswap().view(data.dtype.newbyteorder('='))

noise_cube = data

#rescale
noise_cube /= 1e-5
logger.warning("normalized noise cube")

# Compute coeffs
n_noise = noise_cube.shape[0]
coeffs_list=[]
for i in tqdm(np.arange(n_noise)):
    coeffs = wph_op.apply(torch.from_numpy(noise_cube[i]).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32), norm=None, pbc=pbc)
    coeffs_list.append(coeffs)
    
coeffs_list_cpu = [c.detach().cpu() for c in coeffs_list]
mu = torch.stack(coeffs_list_cpu).mean(dim=0)
std = torch.stack(coeffs_list_cpu).std(dim=0)

print(mu.shape, std.shape)


