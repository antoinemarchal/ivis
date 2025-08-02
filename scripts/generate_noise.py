import glob
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
from reproject import reproject_interp

from ivis.io import DataProcessor
from ivis.imager import Imager
from ivis.logger import logger
from ivis.utils import dutils, mod_loss, fourier

from ivis.models import ClassicIViS

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

#put pb at 1 for pre-PB response
pb = np.full(pb.shape, 1)

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

sky_model = np.zeros(shape, dtype=np.float32)

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

model_vis =  image_processor.forward_model(model=ClassicIViS())

#user parameters
max_its = 20
lambda_sd = 0 #not relevant here
lambda_r = 1 #Control the strength of the Laplacian filtering
device = 0#"cpu" #0 is GPU and "cpu" is CPU
positivity = False #Set to False because noise fluctuates around 0

logger.info("This might take a few minutes if using a CPU...")

n_noise = 20 #number of noise realisations 
noise_cube = np.zeros((n_noise,shape[0],shape[1]))
for i in tqdm(np.arange(n_noise)):
    #Add noise
    fact=1 #Scale the noise.ipynbe with this if needed
    # np.random.seed(42) #FIXME
    noise_real = np.random.normal(loc=0.0, scale=vis_data.sigma*fact)
    noise_imag = np.random.normal(loc=0.0, scale=vis_data.sigma*fact)
    noise = noise_real + 1j * noise_imag

    # Add realistic thermal noise to the model visibilities
    vis_data.data = model_vis + noise

    #Initial parameters (zero array)
    init_params = np.zeros(shape, dtype=np.float32)

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

    # Get model
    model = ClassicIViS()
    
    #get image
    noise_cube[i] = image_processor.process(model=model, units="Jy/arcsec^2") 
    
#Save on disk
hdu = fits.PrimaryHDU(noise_cube)
hdu.writeto(pathout+"noise_cube.fits", overwrite=True)
