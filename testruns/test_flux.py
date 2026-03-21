import glob
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
from reproject import reproject_interp

from ivis.io import DataProcessor
from ivis.imager import Imager3D
from ivis.logger import logger
from ivis.utils import dutils
from ivis.readers import CasacoreReader
from ivis.types import VisIData

from ivis.models import ClassicIViS3D

path_ms = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data_1beam/msl_mw/" #directory of measurement sets    
path_beams = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data_1beam/BEAMS/" #directory of primary beams
path_sd = None #path single-dish data
pathout = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data_1beam/" #path where data will be packaged and stored

#REF WCS INPUT USER
filename = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data_1beam/MW-C10_mom0th_NHI.fits"
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
)

cell_size = (np.abs(target_header["CDELT2"]) *u.deg).to(u.arcsec).value
sky_model = np.zeros((1,shape[0],shape[1]), dtype=np.float32)
sky_model[0][shape[0] // 2, shape[1] // 2] = 1.0 / (cell_size ** 2)  # Jy / arcsec^2


#Model visibilities with IVis forward single frequency model
image_processor = Imager3D(I,      # visibilities
                         pb,            # array of primary beams
                         grid,          # array of interpolation grids
                         None,            # single dish data in unit of Jy/arcsec^2
                         None,       # beam of single-dish data in radio_beam format
                         target_header, # header on which to image the data
                         sky_model,   # init array of parameters
                         0,       # maximum number of iterations
                         0,     # hyper-parameter single-dish
                         False,    # impose a positivity constaint
                         0,        # device: 0 is GPU; "cpu" is CPU
                         0,
                         beam_workers=1)

model = ClassicIViS3D()
model_vis =  image_processor.forward_model(model=model)


# Loop over multiple regularization values for flux recovery
lambda_r_array = [0., 0.01, 0.1, 1, 10, 100]
Nr = len(lambda_r_array)
for i in np.arange(Nr):
    #Add noise
    fact=1 #Scale the noise with this if needed
    noise_real = np.random.normal(loc=0.0, scale=I.sigma_I*fact)
    noise_imag = np.random.normal(loc=0.0, scale=I.sigma_I*fact)
    noise = noise_real + 1j * noise_imag
    
    # Add realistic thermal noise to the model visibilities
    I.data_I = model_vis + noise
    
    #user parameters
    max_its = 40
    lambda_sd = 0 #not relevant here
    lambda_r = lambda_r_array[i] #Control the strength of the Laplacian filtering
    cost_device = 0#"cpu" #0 is GPU and "cpu" is CPU
    optim_device = 0
    positivity = True #Set to true because we know here the sky is positive for a point source
    
    #Initial parameters (zero array)
    init_params = np.zeros((1,shape[0],shape[1]), dtype=np.float32)
    
    #create image processor
    image_processor = Imager3D(I,      # visibilities
                               pb,            # array of primary beams
                               grid,          # array of interpolation grids
                               sd,            # single dish data in unit of Jy/arcsec^2
                               beam_sd,       # beam of single-dish data in radio_beam format
                               target_header, # header on which to image the data
                               init_params,   # init array of parameters
                               max_its,       # maximum number of iterations
                               lambda_sd,     # hyper-parameter single-dish
                               positivity,    # impose a positivity constaint
                               cost_device,        # device: 0 is GPU; "cpu" is CPU
                               optim_device,
                               beam_workers=1)
    #get image
    model = ClassicIViS3D(lambda_r=lambda_r, Nw=0)
    result = image_processor.process(model=model, units="Jy/arcsec^2") #"Jy/arcsec^2" or "K"
    
    #Cutout
    ny, nx = shape
    cy, cx = ny // 2, nx // 2  # center coordinates

    zoom = 20  # pixels around center
    cutout = result[0][cy - zoom:cy + zoom + 1, cx - zoom:cx + zoom + 1]
    
    
    flux, Bmaj, Bmin, theta = dutils.fit_elliptical_gaussian(cutout, pixel_scale_arcsec=cell_size)
    logger.info(flux)

    # stop
