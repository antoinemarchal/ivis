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
from ivis.models import Classic3D
from ivis.imager import Imager3D
from ivis.types import VisIData
from ivis.readers import CasacoreReaderSPW

plt.ion()

def promote_header_2d_to_3d_velocity(target_header_2d,
                                     v0_kms,
                                     nchan,
                                     dv_kms,
                                     specsys="LSRK",
                                     restfrq_hz=1.42040575177e9):
    """
    Promote a 2D FITS header to 3D by adding a VRAD (radio velocity) axis.

    Parameters
    ----------
    v0_kms : float
        Velocity at channel 0 (FITS pixel CRPIX3=1), in km/s.
    nchan : int
        Number of spectral planes in the data (use 1 for (1,ny,nx)).
    dv_kms : float
        Channel spacing in km/s (can be negative if velocity decreases with channel index).
    """
    hdr = target_header_2d.copy()

    nchan = int(nchan)
    if nchan < 1:
        raise ValueError("nchan must be >= 1")
    dv_kms = float(dv_kms)

    hdr["NAXIS"]  = 3
    hdr["NAXIS1"] = int(hdr["NAXIS1"])
    hdr["NAXIS2"] = int(hdr["NAXIS2"])
    hdr["NAXIS3"] = nchan

    hdr["CTYPE3"] = "VRAD"
    hdr["CUNIT3"] = "km/s"
    hdr["CRPIX3"] = 1.0
    hdr["CRVAL3"] = float(v0_kms)
    hdr["CDELT3"] = dv_kms

    hdr["SPECSYS"] = specsys
    hdr["RESTFRQ"] = float(restfrq_hz)

    return hdr


if __name__ == '__main__':    
    #path data
    path_ms = "/totoro/anmarchal/data/lglbs/msets/"    
    path_beams = "/totoro/anmarchal/data/lglbs/BEAMS/" #directory of primary beams
    path_sd = "/totoro/anmarchal/data/parkes/" #path single-dish data - dummy here
    pathout = "/totoro/anmarchal/data/lglbs/products/" #path where data will be packaged and stored

    #create data processor
    data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

    #Define target header
    cfield = SkyCoord(ra="1h34m33.2s", dec="+30d47m6s", frame="icrs") 
    target_header, w = data_processor.make_imaging_header(cfield, fov_deg=1.5, pix_arcsec=3)
    shape = (target_header["NAXIS2"], target_header["NAXIS1"])

    # # Optional: pre-compute PB and grid
    # data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits")

    pb, grid = data_processor.read_pb_and_grid(
        fitsname_pb="reproj_pb.fits",
        fitsname_grid="grid_interp.fits"
    )

    # Dummy single-dish array and beam
    # hdu_sd = fits.open(path_sd+"gass_chan_765.fits")
    # sd = hdu_sd[0].data
    sd = np.zeros(shape, dtype=np.float32)
    beam_sd = Beam(1 * u.deg, 1 * u.deg, 1.e-12 * u.deg)

    # -------------------
    # Read visibilities into VisIData dataclass
    # -------------------
    reader = CasacoreReaderSPW(
        prefer_weight_spectrum=False,
        keep_autocorr=False,
        n_workers=12)
    
    I: VisIData = reader.read_blocks_I(
        ms_root=path_ms,
        uvmin=0, uvmax=4000,
        chan_sel=slice(1900,2150),
        rest_freq=1.42040575177e9, #HI rest frequency in Hz
        mode="merge",
        spw_sel=[5, 5, 5, 8, 8, 8, 4, 4, 4, 4, 4, 4],
    )
    
    # -------------------
    # User parameters
    # -------------------
    max_its = 20
    lambda_sd = 0
    lambda_r = 10
    cost_device = 1 #"cpu"        # 0 for GPU, "cpu" for CPU
    optim_device = 1#"cpu"        # 0 for GPU, "cpu" for CPU
    positivity = False #ATTENTION
    init_params = np.zeros((1, shape[0], shape[1]), dtype=np.float32)

    # -------------------
    # Choose model
    # -------------------
    model = Classic3D(lambda_r=lambda_r)

    nchan = len(I.velocity)
    cube = np.zeros((nchan,shape[0],shape[1]))
    
    for i in np.arange(nchan):
        Ic = I.single_channel(i, copy=False)
        
        # -------------------
        # Create Imager3D
        # -------------------
        image_processor = Imager3D(
            vis_data=Ic, #iter over I
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
        # Run optimization
        # -------------------
        result = image_processor.process(model=model, units="Jy/arcsec^2")

        #In Cube
        cube[i] = result

    v0 = float(I.velocity[0])
    dv = np.diff(I.velocity)[0]
    hdr3 = promote_header_2d_to_3d_velocity(target_header, v0_kms=v0, nchan=nchan, dv_kms=dv)    
    
    #Write output array on disk
    # fits.writeto(pathout + "M33.fits", result, target_header, overwrite=True)
    fits.writeto(pathout + "M33.fits", cube, hdr3, overwrite=True)
        
    # #PLOT RESULT
    # pathout="./"
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    # ax.set_xlabel(r"RA (deg)", fontsize=18.)
    # ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    # # vmin, vmax = np.nanpercentile(result[0], (0.008, 99.992))
    # img = ax.imshow(result[0], vmin=vmin, vmax=vmax, origin="lower", cmap="gray_r")
    # # ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    # colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    # cbar = fig.colorbar(img, cax=colorbar_ax)
    # cbar.ax.tick_params(labelsize=14.)
    # cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
    # plt.savefig(pathout + 'output_merge_full_765.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
