# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
from astropy.wcs import WCS

from ivis.io import DataProcessor
from ivis.imager import Imager3D
from ivis import logger
from ivis.models import ClassicIViS3D
from ivis.types import VisIData
from ivis.readers import CasacoreReader

import marchalib as ml #remove

plt.ion()

if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/chan950/"
    base_beams = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/BEAMS"
    path_sd = "/priv/avatar/amarchal/IMAGING/data/" #path single-dish data
    pathout = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/" #path where data will be packaged and stored

    for idx in range(0, 108):
        path_beams = f"{base_beams}/BEAM_{idx:03d}.fits"

        #REF WCS INPUT USER
        filename = "/priv/avatar/amarchal/MPol-dev/examples/workflow/img.fits"
        target_header = fits.open(filename)[0].header
        shape = (target_header["NAXIS2"], target_header["NAXIS1"])
        
        #create data processor
        data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)
        
        #Define target header
        # cfield = SkyCoord(ra="5h05m43.8s", dec="-70d05m48.5s", frame="icrs")
        target_header = fits.open(path_beams)[0].header
        shape = (target_header["NAXIS2"], target_header["NAXIS1"])
        w = WCS(target_header)
        cfield = SkyCoord(
            ra=w.wcs.crval[0] * u.deg,
            dec=w.wcs.crval[1] * u.deg,
            frame="icrs",
        )
        target_header, w = data_processor.make_imaging_header(cfield, fov_deg=3)
        shape = (target_header["NAXIS2"], target_header["NAXIS1"])
        
        # # Optional: pre-compute PB and grid
        data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb_small.fits", fitsname_grid="grid_interp_small.fits")
        
        pb, grid = data_processor.read_pb_and_grid(
            fitsname_pb="reproj_pb_small.fits",
            fitsname_grid="grid_interp_small.fits"
        )
        
        # -------------------
        # Read visibilities into VisIData dataclass
        # -------------------
        reader = CasacoreReader(
            prefer_weight_spectrum=False,
            keep_autocorr=False,
            n_workers=4)
        
        I: VisIData = reader.read_blocks_I(
            ms_root=path_ms,
            uvmin=0, uvmax=np.inf,
            chan_sel=slice(0,1),
            mode="merge",
            beam_sel=[idx]
        )
        
        # #read single-dish data from "pathout" directory
        # sd, beam_sd = data_processor.read_sd()
        #single-dish data and beam
        fitsname = "GASS_SMC_Jy_regrid.fits"
        hdu_sd = fits.open(path_sd+fitsname)
        hdr_sd = hdu_sd[0].header
        sd = hdu_sd[0].data[950-532]; sd[sd != sd] = 0. #NaN to 0
        #Beam sd
        beam_sd = Beam(hdr_sd["BMIN"]*u.deg, hdr_sd["BMAJ"]*u.deg, 1.e-12*u.deg)
        sd /= (beam_sd.sr).to(u.arcsec**2).value #convert Jy/beam to Jy/arcsec^2
        
        #____________________________________________________________________________
        #user parameters
        max_its = 20
        lambda_sd = 0#10
        lambda_r = 20
        device = 0 #0 is GPU and "cpu" is CPU
        positivity = True
        beam_workers = 1
        
        #create image processor
        init_params = np.zeros((1,shape[0],shape[1]))
        
        image_processor = Imager3D(I,      # visibilities
                                   pb,            # array of primary beams
                                   grid,          # array of interpolation grids
                                   sd,            # single dish data in unit of Jy/arcsec^2
                                   beam_sd,       # beam of single-dish data in radio_beam format
                                   target_header, # header on which to image the data
                                   init_params,   # array to start this optimization with 
                                   max_its,       # maximum number of iterations
                                   lambda_sd,     # hyper-parameter single-dish
                                   positivity,    # impose a positivity constaint
                                   device,        # device: 0 is GPU; "cpu" is CPU
                                   beam_workers
                                   )
        # #get image
        model = ClassicIViS3D(lambda_r=lambda_r, Nw=0)
        
        result = image_processor.process(model=model, units="Jy/arcsec^2") #"Jy/arcsec^2" or "K"
        
        #write on disk
        hdu0 = fits.PrimaryHDU(result, header=target_header)
        hdulist = fits.HDUList([hdu0])
        hdulist.writeto(pathout + "linear/result_deconv_single_{:03d}.fits".format(idx), overwrite=True)
    #_____________________________________________________________________________

    stop

    #Open PB file per antenna
    fitsname  = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/reproj_pb_all_rotated.fits"
    hdu_pb = fits.open(fitsname)
    hdr_pb = hdu_pb[0].header
    pb_all = hdu_pb[0].data    
    pb_mean = np.mean(pb_all,0)
    pb_mean /= np.max(pb_mean)    
    mask = np.where(pb_mean > 0.05, 1, np.nan)

    w_img = ml.wcs2D(target_header)

    #Primary beam + pointings position
    pathout="/priv/avatar/amarchal/ASKAP/IMAGING/plot/"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w_img)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    img = ax.imshow(np.zeros(mask.shape), vmin=0, vmax=1, origin="lower", cmap="inferno")
    ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    ax.scatter(
        [c.ra.deg for c in I.centers],
        [c.dec.deg for c in I.centers],
        transform=ax.get_transform("icrs"),
        s=100,
        facecolors="none",
        edgecolors="white",
        linewidths=2.0,
        zorder=10,
    )
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$A_{\nu}(r)$", fontsize=18.)
    plt.savefig(pathout + 'PB_ASKAP_SB30584_SB30625_SB30665.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)

    stop

    #PLOT RESULT
    pathout="/priv/avatar/amarchal/ASKAP/IMAGING/plot/"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w_img)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    img = ax.imshow(result*mask, vmin=-30, vmax=40, origin="lower", cmap="inferno")
    ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (K)", fontsize=18.)
    plt.savefig(pathout + 'deconv_SMC_nufftt_ASKAP.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
    
    #PLOT RESULT
    pathout="/priv/avatar/amarchal/ASKAP/IMAGING/plot/"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w_img)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    img = ax.imshow(result*mask, vmin=-15, vmax=90, origin="lower", cmap="inferno")
    ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (K)", fontsize=18.)
    plt.savefig(pathout + 'deconv_SMC_nufftt_ASKAP_Parkes.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)

    #PLOT RESULT    
    pathout="/priv/avatar/amarchal/ASKAP/IMAGING/plot/"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w_img)
    ax.set_xlabel(r"RA (deg)", fontsize=18.)
    ax.set_ylabel(r"DEC (deg)", fontsize=18.)
    img = ax.imshow(sd*mask, vmin=0, vmax=5.e-4, origin="lower", cmap="inferno")
    ax.contour(pb_mean, linestyles="--", levels=[0.05, 0.1], colors=["w","w"])
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.)
    cbar.set_label(r"$T_b$ (Jy/arcsec^2)", fontsize=18.)
    plt.savefig(pathout + 'deconv_SMC_nufftt_Parkes.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
