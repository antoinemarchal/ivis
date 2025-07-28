# -*- coding: utf-8 -*-
"""
Imager module for joint deconvolution using GPU-accelerated optimization.

This module provides the `Imager` class which performs non-linear
optimization combining interferometric and single-dish data.

Author: Antoine Marchal
"""

import os
import glob
import sys
import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy import optimize
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
from numpy.fft import fft2
from torch.fft import fft2 as tfft2
from dataclasses import dataclass
from reproject import reproject_interp
import subprocess
import tarfile
import concurrent.futures
from pathlib import Path

from ivis.logger import logger
from ivis.utils import dunits, dutils, mod_loss

# Imager class    
class Imager:
    """
    A GPU-accelerated imager for joint deconvolution of interferometric and single-dish data.

    Parameters
    ----------
    vis_data : object
        Visibility data structure containing uvw coordinates, visibilities, and beam info.
    pb : ndarray
        Primary beam model array.
    grid : ndarray
        Grid array for SIN projection evaluation.
    sd : ndarray
        Single-dish map used for zero-spacing constraint.
    beam_sd : radio_beam.Beam
        Beam object for the single-dish map.
    hdr : dict
        FITS header containing WCS and shape information.
    init_params : ndarray
        Initial parameters (not flattened).
    max_its : int
        Maximum number of iterations for the optimizer.
    lambda_sd : float
        Regularization strength for the single-dish constraint.
    lambda_r : float
        Regularization strength for the spatial prior (e.g., Laplacian).
    positivity : bool
        Whether to enforce a positivity constraint during optimization.
    device : int or str
        Device to use: 0 for GPU, 'cpu' for CPU.
    beam_workers : int
        Number of workers for parallel beam convolution.
    """
        
    def __init__(self, vis_data, pb, grid, sd, beam_sd, hdr, init_params, max_its, lambda_sd, lambda_r, positivity, device, beam_workers):
        super(Imager, self).__init__()
        self.vis_data = vis_data
        self.pb = pb
        self.grid = grid
        self.sd = sd
        self.beam_sd = beam_sd
        self.hdr = hdr
        self.init_params = init_params
        self.max_its = max_its
        self.lambda_sd = lambda_sd
        self.lambda_r = lambda_r
        self.positivity = positivity
        self.beam_workers = beam_workers
        logger.info("[Initialize Imager        ]")
        logger.info(f"Number of iterations to be performed by the optimizer: {self.max_its}")

        # Logger for hyper-parameters
        if self.lambda_r == 0: logger.warning("lambda_r = 0 - No spatial regularization.")
        if self.lambda_sd == 0: logger.warning("lambda_sd = 0 - No short spacing correction (ignoring single dish data).")

        # Check if CUDA is found on the machine and fall back on CPU otherwise
        self.device = self.get_device(device)
        if self.device == 0: logger.info(f"Using GPU device: {torch.cuda.get_device_name(device)}")
        if self.device == "cpu": logger.info(f"Using {self.beam_workers} workers for beam parallelisation.")


    @staticmethod
    def get_device(user_device):
        """
        Selects the appropriate compute device (CPU or GPU) based on availability and user request.

        Parameters
        ----------
        user_device : int or str
            0 to request GPU, otherwise uses CPU.

        Returns
        -------
        torch.device
            The selected torch device.
        """
        if user_device == 0:  # User requested GPU
            try:
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
                else:
                    raise RuntimeError("CUDA not available.")
            except RuntimeError as e:
                logger.warning(f"{e} Falling back on CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU.")

        return device


    def process_beam_positions(self):
        """
        Determines the first and last indices for each beam in the visibility dataset.

        Returns
        -------
        idmin : ndarray of int
            First occurrence index for each beam.
        idmax : ndarray of int
            Last occurrence index (exclusive upper bound) for each beam.
        """
        # nb = len(self.vis_data.coords)
        # idmin = np.zeros(nb); idmax = np.zeros(nb)
        # for i in np.arange(nb):
        #     idmin[i] = np.where(self.vis_data.beam == i)[0][0];
        #     idmax[i] = len(np.where(self.vis_data.beam == i)[0])#-1
        nb = len(self.vis_data.coords)
        
        # Find unique beam indices and their first occurrence
        unique_beams, first_idx = np.unique(self.vis_data.beam, return_index=True)
        
        # Get counts of occurrences per beam
        beam_counts = np.bincount(self.vis_data.beam, minlength=nb)
        
        # Initialize arrays
        idmin = np.zeros(nb, dtype=int)
        idmax = np.zeros(nb, dtype=int)
        
        # Assign first index
        idmin[unique_beams] = first_idx
        
        # Assign (count - 1) for each beam
        idmax[unique_beams] = beam_counts[unique_beams] #- 1
        
        return idmin, idmax


    def forward_model(self):
        # Image parameters
        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        
        # Convert λ to radians per pixel
        uu_radpix = dunits._lambda_to_radpix(self.vis_data.uu, cell_size)
        vv_radpix = dunits._lambda_to_radpix(self.vis_data.vv, cell_size)
        ww_radpix = dunits._lambda_to_radpix(self.vis_data.ww, cell_size)
        
        # Get beam slice indices
        idmin, idmax = self.process_beam_positions()
        
        # Ensure pb and grid have native byte order for PyTorch compatibility
        pb_native = np.asarray(self.pb, dtype=np.float32)
        grid_native = np.asarray(self.grid, dtype=np.float32)
        
        # Compute model visibilities
        model_vis = mod_loss.single_frequency_model(
            x=self.init_params,
            data=self.vis_data.data,
            uu=uu_radpix,
            vv=vv_radpix,
            pb=pb_native,
            idmina=idmin,
            idmaxa=idmax,
            device=self.device,
            cell_size=cell_size.value,
            grid_array=grid_native,
        )
        
        return model_vis
    

    def process(self, units, disk=False):
        """
        Runs the imaging optimization pipeline and returns a restored image in the requested unit.

        Parameters
        ----------
        units : str
            Output unit. Must be one of: 'Jy/arcsec^2', 'Jy/beam', or 'K'.
        disk : bool, optional
            If True, writes intermediate results to disk (currently unused).

        Returns
        -------
        result : ndarray
            Restored image in the requested unit.
        """
        #Image parameters
        cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        #tapper for apodization
        tapper = dutils.apodize(0.98, shape)
        
        #Convert lambda to radian per pixel
        uu_radpix = dunits._lambda_to_radpix(self.vis_data.uu, cell_size)
        vv_radpix = dunits._lambda_to_radpix(self.vis_data.vv, cell_size)
        ww_radpix = dunits._lambda_to_radpix(self.vis_data.ww, cell_size)

        #Build kernel for regularization
        kernel_map = dutils.laplacian(shape)
        fftkernel = abs(fft2(kernel_map))

        #generate fftbeam
        bmaj = self.beam_sd.major.value
        cdelt2 = cell_size.to(u.deg).value
        bmaj_pix = bmaj / cdelt2
        beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
        fftbeam = abs((fft2(beam)))

        #fft single-dish map
        fftsd = cell_size.value**2 * tfft2(torch.from_numpy(np.float32(self.sd))).numpy()

        #Get idx beams in array
        # logger.info("Processing beams position..")
        idmin, idmax = self.process_beam_positions()

        #define bounds for optimisation
        if self.positivity == False:
            bounds = dutils.ROHSA_bounds(data_shape=shape, lb_amp=-np.inf, ub_amp=np.inf)
        else:
            bounds = dutils.ROHSA_bounds(data_shape=shape, lb_amp=0, ub_amp=np.inf)
            
        # Use gradient-descent to minimise cost
        logger.info('Starting optimisation (using LBFGS-B)')
        if self.positivity == True:
            logger.info('Optimizer bounded - Positivity == True')
            logger.warning('Optimizer bounded - Because there is noise in the data, it is generally not recommanded to add a positivity constaint.')
        else:
            logger.info('Optimizer not bounded - Positivity == False')
                
        # Precompute type conversions (done once)
        params_f32 = self.init_params.ravel().astype(np.float32)
        beam_f32 = np.asarray(self.vis_data.beam, dtype=np.float32)
        fftbeam_f32 = np.asarray(fftbeam, dtype=np.float32)
        data_c64 = np.asarray(self.vis_data.data, dtype=np.complex64)
        uu_f32 = np.asarray(uu_radpix, dtype=np.float32)
        vv_f32 = np.asarray(vv_radpix, dtype=np.float32)
        ww_f32 = np.asarray(ww_radpix, dtype=np.float32)
        pb_f32 = np.asarray(self.pb, dtype=np.float32)
        idmin_i32 = np.asarray(idmin, dtype=np.int32)
        idmax_i32 = np.asarray(idmax, dtype=np.int32)
        sigma_f32 = np.asarray(self.vis_data.sigma, dtype=np.float32)
        fftsd_c64 = np.asarray(fftsd, dtype=np.complex64)
        tapper_f32 = np.asarray(tapper, dtype=np.float32)
        fftkernel_f32 = np.asarray(fftkernel, dtype=np.float32)
        grid_f32 = np.asarray(self.grid, dtype=np.float32)
        
        opt_args = (
            beam_f32, fftbeam_f32, data_c64, uu_f32, vv_f32, ww_f32,
            pb_f32, idmin_i32, idmax_i32, self.device, sigma_f32, fftsd_c64,
            tapper_f32, self.lambda_sd, self.lambda_r, fftkernel_f32, shape,
            cell_size.value, grid_f32, self.beam_workers
        )
        
        options = {
            'maxiter': self.max_its,
            'maxfun': int(1e6),
            'iprint': 25,
        }
        
        # Run optimization
        opt_output = optimize.minimize(
            mod_loss.objective, params_f32,
            args=opt_args,
            jac=True,
            tol=1.e-8,
            bounds=bounds,
            method='L-BFGS-B',
            options=options
        )

        # logger.info(opt_output)        
        result = np.reshape(opt_output.x, shape) #* 2
        logger.warning("multiply by 2 for ASKAP.")

        #unit conversion
        if units == "Jy/arcsec^2":
            return result

        if units == "Jy/beam":
            logger.info("assuming a synthesized beam of 4.2857 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            beam_r = Beam(4.2857*cell_size, 4.2857*cell_size, 1.e-12*u.deg) 
            return result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    

        elif units == "K":
            logger.info("assuming a synthesized beam of 3 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            nu = self.vis_data.frequency[0] *u.Hz
            beam_r = Beam(3*cell_size, 3*cell_size, 1.e-12*u.deg)
            result_Jy = result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    
            return (result_Jy*u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
            
        else: logger.info("unit must be 'Jy/arcsec^2' or 'K'")            
    

