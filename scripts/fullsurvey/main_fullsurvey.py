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
    path_ms = "/totoro/anmarchal/data/gaskap/fullsurvey/untar/merge/"
    
    path_beams = "/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/merge/" #directory of primary beams
    path_sd = "./" #path single-dish data - dummy here
    pathout = "/totoro/anmarchal/data/gaskap/fullsurvey/products/merge/" #path where data will be packaged and stored

    #create data processor
    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

    #Define target header
    # cfield = SkyCoord(ra="5h36m41s", dec="-67d58m44s", frame="icrs")
    cfield = SkyCoord(ra="5h05m43.8s", dec="-70d05m48.5s", frame="icrs")
    # target_header, w = data_processor.make_imaging_header(cfield, fov_deg=12, pix_arcsec=4)
    target_header, w = data_processor.make_imaging_header(cfield, fov_deg=15)
    shape = (target_header["NAXIS2"], target_header["NAXIS1"])

    # # Optional: pre-compute PB and grid
    # data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb2.fits", fitsname_grid="grid_interp2.fits")

    # pb, grid = data_processor.read_pb_and_grid(
    #     fitsname_pb="reproj_pb_high.fits",
    #     fitsname_grid="grid_interp_high.fits"
    # )
    pb, grid = data_processor.read_pb_and_grid(
        fitsname_pb="reproj_pb2.fits",
        fitsname_grid="grid_interp2.fits"
    )

    # Dummy single-dish array and beam
    sd = np.zeros(shape, dtype=np.float32)
    beam_sd = Beam(1 * u.deg, 1 * u.deg, 1.e-12 * u.deg)

    # -------------------
    # Read visibilities into VisIData dataclass
    # -------------------
    reader = CasacoreReader(
        prefer_weight_spectrum=False,
        keep_autocorr=False,
        n_workers=24)
    
    I: VisIData = reader.read_blocks_I(
        ms_root=path_ms,
        uvmin=0, uvmax=np.inf,
        chan_sel=slice(810,811),
        # chan_sel=slice(765,766),
        rest_freq=1.42040575177e9, #HI rest frequency in Hz
        mode="merge",
        target_center=cfield,
        # target_radius=1*u.deg
    )

    # -------------------
    # User parameters
    # -------------------
    max_its = 20
    lambda_sd = 0
    lambda_r = 1
    cost_device = 0        # 0 for GPU, "cpu" for CPU
    optim_device = "cpu"        # 0 for GPU, "cpu" for CPU
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
    result = image_processor.process(model=model, units="Jy/arcsec^2")

    #Write output array on disk
    fits.writeto(pathout + "output_1blocks_7arcsec_lambda_r_2_positivity_true_iter_20_new_PB_Nw_0.fits", result, target_header, overwrite=True)

    stop
    
    #PLOT RESULT
    pathout="./"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    # vmin, vmax = np.nanpercentile(result[0], (0.008, 99.992))
    img = ax.imshow(result[0], vmin=vmin, vmax=vmax, origin="lower", cmap="gray_r")
    # ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
    plt.savefig(pathout + 'output_merge_full_765.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
