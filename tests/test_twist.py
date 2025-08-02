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


def main():
    path_ms = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data/msl_mw/" #directory of measurement sets    
    path_beams = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data/BEAMS/" #directory of primary beams
    path_sd = None #path single-dish data
    pathout = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data/" #path where data will be packaged and stored
    
    #REF WCS INPUT USER
    filename = "/Users/antoine/Desktop/Synthesis/ivis/docs/tutorials/data_tutorials/ivis_data/MW-C10_mom0th_NHI.fits"
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
    wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device="cpu")
    wph_op.load_model(wph_model)

    # Open noise data cube
    with fits.open(pathout+"noise_cube.fits", memmap=True) as hdul:
        data = hdul[0].data
        if not data.dtype.isnative:
            data = data.byteswap().newbyteorder()
            
    noise_cube = data  # shape (nfreq, ny, nx)
    
    # Compute coeffs
    n_noise = noise_cube.shape[0]
    coeffs_list=[]
    for i in tqdm(np.arange(n_noise)):
        coeffs = wph_op.apply(torch.from_numpy(noise_cube[i]).unsqueeze(0).unsqueeze(0).to("cpu"), norm=None, pbc=pbc)
        coeffs_list.append(coeffs)
    
    mu = np.mean(np.array(coeffs_list),(0,1,2))
    std = np.std(np.array(coeffs_list),(0,1,2))

    print(mu.shape, sig.shape)


if __name__ == "__main__":
    main()

