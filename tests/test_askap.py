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

from ivis.io import DataProcessor
from ivis.logger import logger
from ivis.models import ClassicIViS3D
from ivis.imager import Imager3D
from ivis.types import VisIData
from ivis.readers import CasacoreReader

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/gaskap/fullsurvey/"#sb69152/"
    
    path_beams = "/priv/avatar/amarchal/Projects/ivis/examples/data/ASKAP/BEAMS/" #directory of primary beams
    path_sd = "/priv/avatar/amarchal/GASS/data/" #path single-dish data - dummy here
    pathout = "/priv/avatar/amarchal/Projects/ivis/examples/data/ASKAP/" #path where data will be packaged and stored

    #REF WCS INPUT USER
    cfield = SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs')
    filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
    target_header = fits.open(filename)[0].header
    target_header["CRVAL1"] = cfield.ra.value
    target_header["CRVAL2"] = cfield.dec.value
    shape = (target_header["NAXIS2"], target_header["NAXIS1"])

    #SD data -- NOT USE HERE
    fitsname = "reproj_GASS_v.fits"
    hdu_sd = fits.open(path_sd+fitsname)
    hdr_sd = hdu_sd[0].header
    sd = hdu_sd[0].data; sd[sd != sd] = 0. #NaN to 0
    beam_sd = Beam((16*u.arcmin).to(u.deg),(16*u.arcmin).to(u.deg), 1.e-12*u.deg) #must be all in deg

    #create data processor
    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)
    pb, grid = data_processor.read_pb_and_grid(fitsname_pb="reproj_pb_Dave.fits", fitsname_grid="grid_interp_Dave.fits")

    # -------------------
    # Read visibilities into VisIData dataclass
    # -------------------
    reader = CasacoreReader(
        prefer_weight_spectrum=False,
        keep_autocorr=False,
        n_workers=4)
    
    I: VisIData = reader.read_blocks_I(
        ms_root=path_ms,
        uvmin=0, uvmax=12000,
        chan_sel=slice(950,951),
        mode="concat",
    )
    
    # -------------------
    # User parameters
    # -------------------
    max_its = 20
    lambda_sd = 0
    lambda_r = 1
    cost_device = 0        # 0 for GPU, "cpu" for CPU
    optim_device = "cpu"        # 0 for GPU, "cpu" for CPU
    positivity = False
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
    result = image_processor.process(model=model, units="Jy/arcsec^2")

    
