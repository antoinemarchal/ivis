# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from radio_beam import Beam

from deconv.pipeline import Pipeline
from deconv import logger

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/gaskap/fullsurvey/sb67521/"#sb68329/"
    
    path_beams = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/BEAMS/" #directory of primary beams
    path_sd = "/priv/avatar/amarchal/GASS/data/" #path single-dish data - dummy here
    pathout = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/" #path where data will be packaged and stored

    #REF WCS INPUT USER
    cfield = SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs')
    filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
    target_header = fits.open(filename)[0].header
    target_header["CRVAL1"] = cfield.ra.value
    target_header["CRVAL2"] = cfield.dec.value
    shape = (target_header["NAXIS2"], target_header["NAXIS1"])

    # Single-dish data and Beam
    sd = np.zeros(shape); #Dummy array for single dish
    beam_sd = Beam((16*u.arcmin).to(u.deg),(16*u.arcmin).to(u.deg), 1.e-12*u.deg) #must be all in deg
    
    #____________________________________________________________________________
    # Define separate worker counts
    data_processor_workers = 12  # Workers for DataProcessor
    imager_workers = 2           # Workers for the Imager
    queue_maxsize = 3            # Queue size to balance memory and speed
    blocks = 'single'            # Single or multiple blocks in path_ms
    extension = ".ms"
    fixms = False

    # User parameters Imager
    max_its = 2
    lambda_sd = 0#1
    lambda_r = 20
    device = 0#"cpu" #0 is GPU and "cpu" is CPU
    positivity = False
    units = "Jy/arcsec^2"
    uvmin = 0                    
    uvmax = 7000

    # Cube parameters
    start, end, step = 900, 903, 1
    filename = f"result_chan_{start:04d}_to_{end-1:04d}_{step:02d}_Jy_arcsec2.fits"

    pipeline = Pipeline(
        path_ms, path_beams, path_sd, pathout,
        target_header, units, sd, beam_sd, max_its, lambda_sd, lambda_r, positivity, device,
        start=start, end=end, step=step, data_processor_workers=data_processor_workers,
        imager_workers=imager_workers, queue_maxsize=queue_maxsize, uvmin=uvmin, uvmax=uvmax,
        extension=extension, blocks=blocks, fixms=fixms
    )
    
    pipeline.run()
    pipeline.write(filename)
