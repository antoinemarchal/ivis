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
from scipy.optimize import fmin_l_bfgs_b
import time

from ivis.logger import logger
from ivis.utils import dunits, dutils
from ivis.optim.solvers import (
    optimize_scipy_lbfgsb,
    optimize_torch_cg,
    optimize_torch_fista,
    optimize_torch_lbfgs,
)

#------------------------------------
#------------ Imager3D --------------
#------------------------------------
class Imager3D:
    """
    GPU-accelerated imager for joint deconvolution of interferometric
    and single-dish data, using the new VisIData dataclass.
    """

    def __init__(self, vis_data, pb, grid, sd, beam_sd, hdr,
                 init_params, max_its, lambda_sd, positivity,
                 cost_device="auto", optim_device="auto", beam_workers=0):

        self.vis_data = vis_data
        self.pb = pb
        self.grid = grid
        self.sd = sd
        self.beam_sd = beam_sd
        self.hdr = hdr
        self.init_params = init_params
        self.max_its = max_its
        self.lambda_sd = lambda_sd
        self.positivity = positivity
        self.beam_workers = beam_workers

        # moved to dutils
        self.cost_device  = dutils.get_device(cost_device)
        self.optim_device = dutils.get_device(optim_device)

        logger.info("[Initialize Imager3D       ]")
        logger.info(f"Number of iterations to be performed by the optimizer: {self.max_its}")

        if self.lambda_sd == 0:
            logger.warning("lambda_sd = 0 — No short-spacing correction.")

        if self.positivity:
            logger.info("Optimizer bounded - Positivity == True")
        else:
            logger.info("Optimizer not bounded - Positivity == False")

    def forward_model(self, model, x=None):
        """
        Compute model visibilities from the current image parameters
        using the given model's forward operator.
        """
        if model is None:
            raise ValueError("Must pass a model instance to `forward_model()`.")

        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)

        pb_native = np.asarray(self.pb, dtype=np.float32)
        grid_native = np.asarray(self.grid, dtype=np.float32)
        x_model = self.init_params if x is None else x

        return model.forward(
            x=x_model,
            vis_data=self.vis_data,
            pb=pb_native,
            device=self.cost_device,
            cell_size=cell_size.value,
            grid_array=grid_native
        )

    def adjoint_model(self, model, vis=None, return_real=False):
        """
        Apply the backward/adjoint of the model forward operator to a visibility cube or
        flat visibility vector. If vis is None, uses self.vis_data.data_I.
        """
        if model is None:
            raise ValueError("Must pass a model instance to `adjoint_model()`.")

        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)

        pb_native = np.asarray(self.pb, dtype=np.float32)
        grid_native = np.asarray(self.grid, dtype=np.float32)

        if hasattr(model, "backward"):
            return model.backward(
                vis=vis,
                vis_data=self.vis_data,
                pb=pb_native,
                device=self.cost_device,
                cell_size=cell_size.value,
                grid_array=grid_native,
                x_shape=self.init_params.shape,
                return_real=return_real,
            )

        return model.adjoint(
            vis=vis,
            vis_data=self.vis_data,
            pb=pb_native,
            device=self.cost_device,
            cell_size=cell_size.value,
            grid_array=grid_native,
            x_shape=self.init_params.shape,
            return_real=return_real,
        )

    # ------------------------------------------------------------------
    # process(): SAME logic + SAME log strings as your original version
    # ------------------------------------------------------------------
    def process(self, model=None, solver="LBFGS-B", units="Jy/arcsec^2",
                history_size=10, dtype=torch.float32, initial_step=None,
                initial_update=1.0e-5, backtracking_factor=0.5,
                grow_factor=1.25, loss_only_line_search=False):
        """
        Devices
        -------
        - optim_device: where PyTorch LBFGS params/optimizer live
        - cost_device : where model.objective() runs

        Solver compatibility
        --------------------
        - ``"FISTA"`` supports either positivity setting.
        - ``"L-BFGS-B"`` requires ``positivity=True``.
        - ``"LBFGS"`` and ``"CG"`` require ``positivity=False``.

        FISTA options
        -------------
        ``initial_step``, ``initial_update``, ``backtracking_factor``,
        ``grow_factor``, and ``loss_only_line_search`` are passed to FISTA
        when ``solver="FISTA"``.  The latter is an opt-in optimization for
        models that support loss-only candidate evaluations.

        Notes
        -----
        objective() is expected to call backward() internally (unchanged).
        """
        if model is None:
            raise ValueError("Must pass a model instance to `process()`.")

        solver_name = str(solver).upper().replace("_", "-")
        solver_name = {
            "LBFGSB": "L-BFGS-B",
            "LBFGS-B": "L-BFGS-B",
        }.get(solver_name, solver_name)
        if solver_name not in {"L-BFGS-B", "LBFGS", "FISTA", "CG"}:
            raise ValueError(
                f"Unknown solver '{solver}'. Choose 'L-BFGS-B', 'LBFGS', 'FISTA', or 'CG'."
            )
        if self.positivity and solver_name in {"LBFGS", "CG"}:
            raise ValueError(f"solver='{solver_name}' is incompatible with positivity=True.")
        if not self.positivity and solver_name == "L-BFGS-B":
            raise ValueError("solver='L-BFGS-B' requires positivity=True.")
        if solver_name == "CG" and not (
            hasattr(model, "apply_normal_operator") and hasattr(model, "quadratic_rhs")
        ):
            raise ValueError(
                "solver='CG' requires model.apply_normal_operator() and model.quadratic_rhs()."
            )

        cost_dev  = self.cost_device
        optim_dev = self.optim_device

        # --- Units logger (unchanged strings) ---
        if units == "Jy/arcsec^2":
            logger.info("Units of output: Jy/arcsec^2.")

        elif units == "K":
            logger.info("Units of output: K (using frequency in Hz).")

        elif units == "Jy/beam":
            logger.warning(
                "Units of output: Jy/beam. "
                "Note: this is not the preferred unit in IViS, as the effective beam "
                "depends on regularization. Unlike CLEAN, the model is not reconvolved "
                "with a Gaussian restoring beam. "
                "We recommend using 'K' units for diffuse extended emission."
            )

        else:
            logger.error(
                f"Unknown unit '{units}'. Units must be 'Jy/arcsec^2', 'K', or (not recommended) 'Jy/beam'. "
                "Defaulting to Jy/arcsec^2."
            )

        # --- Image/grid params (unchanged) ---
        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        tapper = dutils.apodize(0.98, shape)

        # --- FFT beam for reg (unchanged) ---
        kernel_map = dutils.laplacian(shape)
        fftkernel = abs(fft2(kernel_map))
        bmaj_pix = self.beam_sd.major.to(u.deg).value / cell_size.to(u.deg).value
        beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
        fftbeam = abs(fft2(beam))

        # --- FFT single-dish (unchanged) ---
        fftsd = cell_size.value**2 * tfft2(torch.from_numpy(np.float32(self.sd))).cpu().numpy()

        # --- Common params (unchanged keys/values) ---
        params = dict(
            vis_data=self.vis_data,
            pb=np.asarray(self.pb, dtype=np.float32),
            fftbeam=np.asarray(fftbeam, dtype=np.float32),
            fftsd=np.asarray(fftsd, dtype=np.complex64),
            tapper=np.asarray(tapper, dtype=np.float32),
            lambda_sd=self.lambda_sd,
            fftkernel=np.asarray(fftkernel, dtype=np.float32),
            cell_size=cell_size.value,
            grid_array=np.asarray(self.grid, dtype=np.float32),
            beam_workers=self.beam_workers,
        )

        param_shape = self.init_params.shape

        if solver_name == "FISTA":
            flat = optimize_torch_fista(
                model=model,
                x_init=self.init_params,
                dtype=dtype,
                max_its=self.max_its,
                cost_dev=cost_dev,
                optim_dev=optim_dev,
                params=params,
                positivity=self.positivity,
                initial_step=initial_step,
                initial_update=initial_update,
                backtracking_factor=backtracking_factor,
                grow_factor=grow_factor,
                loss_only_line_search=loss_only_line_search,
            )
            result = flat.reshape(param_shape)
        elif solver_name == "L-BFGS-B":
            x0 = self.init_params.ravel().astype(np.float64)
            raw_bounds = dutils.ROHSA_bounds(param_shape, lb_amp=0, ub_amp=np.inf)
            bounds64 = [(float(lo), float(hi)) for (lo, hi) in raw_bounds]
            result = optimize_scipy_lbfgsb(
                model=model, x0=x0, bounds64=bounds64,
                param_shape=param_shape, max_its=self.max_its,
                cost_dev=cost_dev, optim_dev=optim_dev, params=params,
            )
        elif solver_name == "CG":
            flat = optimize_torch_cg(
                model=model, x_init=self.init_params, dtype=dtype,
                max_its=self.max_its, cost_dev=cost_dev, optim_dev=optim_dev,
                params=params,
            )
            result = flat.reshape(param_shape)
        else:  # LBFGS
            flat = optimize_torch_lbfgs(
                model=model, x_init=self.init_params, dtype=dtype,
                history_size=history_size, max_its=self.max_its,
                cost_dev=cost_dev, optim_dev=optim_dev, params=params,
            )
            result = flat.reshape(param_shape)

        # --- Unit conversion (unchanged) ---
        if units == "Jy/arcsec^2":
            output = result

        elif units == "Jy/beam":
            assumed_fwhm_pix = 3  # hard-coded value
            logger.warning(
                f"Converting to Jy/beam assuming a restoring beam of "
                f"{assumed_fwhm_pix} × cell_size = "
                f"{assumed_fwhm_pix * cell_size:.3f} FWHM."
            )
            beam_r = Beam(assumed_fwhm_pix * cell_size,
                          assumed_fwhm_pix * cell_size, 1.e-12 * u.deg)
            output = result * beam_r.sr.to(u.arcsec**2).value

        elif units == "K":
            nu_Hz = self.vis_data.frequency
            output = dunits.jy_per_arcsec2_to_K(result, nu_Hz)

        else:
            logger.warning("Unknown unit type. Returning result in Jy/arcsec^2.")
            output = result

        logger.info("Successful run. Please clap.")
        return output


#------------------------------------
#------------ Imager ----------------
#------------------------------------
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
        
    def __init__(self, vis_data, pb, grid, sd, beam_sd, hdr, init_params, max_its, lambda_sd, positivity, device, beam_workers):
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
        self.positivity = positivity
        self.beam_workers = beam_workers
        logger.info("[Initialize Imager        ]")
        logger.info(f"Number of iterations to be performed by the optimizer: {self.max_its}")

        # Logger for hyper-parameters
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


    def forward_model(self, model, x=None):
        """
        Compute model visibilities from an input image using the provided model's forward operator.

        Parameters
        ----------
        model : object
            A model instance (e.g., ClassicIViS) that implements a `.forward(...)` method
            to simulate visibilities from image-domain parameters.
        x : np.ndarray or None, optional
            Image parameters to forward-project. If None, uses ``self.init_params``.

        Returns
        -------
        model_vis : np.ndarray
            Complex model visibilities, one per (u,v) coordinate in the data.

        Raises
        ------
        ValueError
            If no model is provided.

        Notes
        -----
        - Converts spatial frequencies to units of radians per pixel based on image header.
        - Uses internal primary beam and interpolation grid arrays.
        - Forwards all necessary inputs to the model's `forward` method.
        """
        if model is None:
            raise ValueError("You must pass a model instance to `forward_model()`.")

        # Image parameters
        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        
        # Convert λ to radians per pixel
        uu_radpix = dunits._lambda_to_radpix(self.vis_data.uu, cell_size)
        vv_radpix = dunits._lambda_to_radpix(self.vis_data.vv, cell_size)
        ww_radpix = dunits._lambda_to_radpix(self.vis_data.ww, cell_size)  # if needed
        
        # Get beam slice indices
        idmin, idmax = self.process_beam_positions()
        
        # Native arrays
        pb_native = np.asarray(self.pb, dtype=np.float32)
        grid_native = np.asarray(self.grid, dtype=np.float32)
        x_model = self.init_params if x is None else x
        
        return model.forward(
            x=x_model,
            data=self.vis_data.data,
            uu=uu_radpix,
            vv=vv_radpix,
            ww=self.vis_data.ww,
            pb=pb_native,
            idmina=idmin,
            idmaxa=idmax,
            device=self.device,
            cell_size=cell_size.value,
            grid_array=grid_native
        )
    

    def process(self, model=None, units="Jy/arcsec^2", disk=False):
        """
        Runs the imaging optimization pipeline and returns a restored image in the requested unit.

        Parameters
        ----------
        model : object
            An imaging model instance implementing a `.loss(x, ...)` method compatible with scipy.optimize.minimize.
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
        param_shape = self.init_params.shape  # (H, W) or (2, H, W)
        if self.positivity == False:
            bounds = dutils.ROHSA_bounds(data_shape=param_shape, lb_amp=-np.inf, ub_amp=np.inf)
        else:
            bounds = dutils.ROHSA_bounds(data_shape=param_shape, lb_amp=0, ub_amp=np.inf)
            
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
        ww_f32 = np.asarray(self.vis_data.ww, dtype=np.float32) #Not radpix - original
        pb_f32 = np.asarray(self.pb, dtype=np.float32)
        idmin_i32 = np.asarray(idmin, dtype=np.int32)
        idmax_i32 = np.asarray(idmax, dtype=np.int32)
        sigma_f32 = np.asarray(self.vis_data.sigma, dtype=np.float32)
        fftsd_c64 = np.asarray(fftsd, dtype=np.complex64)
        tapper_f32 = np.asarray(tapper, dtype=np.float32)
        fftkernel_f32 = np.asarray(fftkernel, dtype=np.float32)
        grid_f32 = np.asarray(self.grid, dtype=np.float32)

        # ---- Precompute params ----
        params = dict(
            beam=beam_f32,
            fftbeam=fftbeam_f32,
            data=data_c64,
            uu=uu_f32,
            vv=vv_f32,
            ww=ww_f32,
            pb=pb_f32,
            idmina=idmin_i32,
            idmaxa=idmax_i32,
            sigma=sigma_f32,
            fftsd=fftsd_c64,
            tapper=tapper_f32,
            lambda_sd=self.lambda_sd,
            fftkernel=fftkernel_f32,
            cell_size=cell_size.value,
            grid_array=grid_f32,
            beam_workers=self.beam_workers
        )

        # ---- Define closure for optimization ----
        def objective_flat(x):
            return model.loss(x, shape=shape, device=device, **params)
        
        shape = self.init_params.shape
        device = self.device
        
        options = {
            'maxiter': self.max_its,
            'maxfun': int(1e6),
            'iprint': 25,
        }
        
        if model is None:
            logger.error("You must pass a model instance (e.g., ClassicIViS) to `process()`.")
            raise ValueError("Missing model input.")

        if not hasattr(model, 'loss'):
            logger.error("Provided model does not implement a `.loss(x, ...)` method compatible with scipy.optimize.minimize.")
            raise TypeError("Invalid model type.")

        # ---- Run optimizer ----
        opt_output = optimize.minimize(
            objective_flat,
            params_f32,
            jac=True,
            tol=1.e-8,
            bounds=bounds,
            method='L-BFGS-B',
            options=options
        )

        # logger.info(opt_output)        
        result = np.reshape(opt_output.x, self.init_params.shape) #* 2
        logger.warning("multiply by 2 for ASKAP.")

        #unit conversion
        if units == "Jy/arcsec^2":
            return result

        # if units == "Jy/beam":
        #     logger.info("assuming a synthesized beam of 4.2857 x cell_size")
        #     cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
        #     beam_r = Beam(4.2857*cell_size, 4.2857*cell_size, 1.e-12*u.deg) 
        #     return result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    

        elif units == "K":
            logger.info("assuming a synthesized beam of 3 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            nu = self.vis_data.frequency[0] *u.Hz
            beam_r = Beam(3*cell_size, 3*cell_size, 1.e-12*u.deg)
            result_Jy = result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    
            return (result_Jy*u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
            
        else: logger.info("unit must be 'Jy/arcsec^2' or 'K'")
