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
from scipy.optimize import fmin_l_bfgs_b

from ivis.logger import logger
from ivis.utils import dunits, dutils

class Imager3D:
    """
    GPU-accelerated imager for joint deconvolution of interferometric
    and single-dish data, using the new VisIData dataclass.
    """

    def __init__(self, vis_data, pb, grid, sd, beam_sd, hdr,
                 init_params, max_its, lambda_sd, positivity, cost_device="auto",
                 optim_device="auto", beam_workers=0):
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
        
        self.cost_device  = self.get_device(cost_device)
        self.optim_device = self.get_device(optim_device)

        logger.info("[Initialize Imager3D       ]")
        logger.info(f"Number of iterations to be performed by the optimizer: {self.max_its}")

        if self.lambda_sd == 0:
            logger.warning("lambda_sd = 0 — No short-spacing correction.")
            
        if self.positivity == True:
            logger.info('Optimizer bounded - Positivity == True')
            logger.warning('Optimizer bounded - Because there is noise in the data, it is generally not recommanded to add a positivity constaint.')
        else:
            logger.info('Optimizer not bounded - Positivity == False')


    @staticmethod
    def get_device(spec="auto") -> torch.device:
        """
        Resolve a compute device from a flexible spec:
          - "auto"      -> cuda:0 if available; else mps (Apple Silicon); else cpu
          - "cpu"       -> cpu
          - "cuda"      -> cuda:0 (if available)
          - "cuda:i"    -> that specific GPU index (if available)
          - "mps"       -> Apple Metal Performance Shaders (if available)
          - int i       -> cuda:i if CUDA available; else cpu
          - torch.device -> returned as-is
        """
        # passthrough
        if isinstance(spec, torch.device):
            return spec

        # int -> cuda:i if possible
        if isinstance(spec, int):
            if spec >= 0 and torch.cuda.is_available():
                idx = int(spec)
                if idx < torch.cuda.device_count():
                    logger.info(f"Using GPU cuda:{idx} ({torch.cuda.get_device_name(idx)})")
                    return torch.device(f"cuda:{idx}")
                else:
                    logger.warning(f"Requested cuda:{idx} but only {torch.cuda.device_count()} device(s); using cuda:0")
                    return torch.device("cuda:0")
            logger.info("CUDA unavailable or invalid index; using CPU.")
            return torch.device("cpu")

        # string path
        if isinstance(spec, str):
            s = spec.strip().lower()

            if s in ("auto", ""):
                if torch.cuda.is_available():
                    logger.info(f"Using GPU (auto) cuda:0 ({torch.cuda.get_device_name(0)})")
                    return torch.device("cuda:0")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    logger.info("Using Apple MPS (auto).")
                    return torch.device("mps")
                logger.info("Using CPU (auto).")
                return torch.device("cpu")

            if s == "cpu":
                logger.info("Using CPU (user-specified).")
                return torch.device("cpu")

            if s == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    logger.info("Using Apple MPS (user-specified).")
                    return torch.device("mps")
                logger.warning("MPS requested but not available; falling back to CPU.")
                return torch.device("cpu")

            if s.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available; falling back to CPU.")
                    return torch.device("cpu")
                # allow "cuda" or "cuda:i"
                idx = 0
                if ":" in s:
                    try:
                        idx = int(s.split(":", 1)[1])
                    except Exception:
                        logger.warning(f"Could not parse device index from '{spec}', defaulting to cuda:0.")
                        idx = 0
                if idx < torch.cuda.device_count():
                    logger.info(f"Using GPU cuda:{idx} ({torch.cuda.get_device_name(idx)})")
                    return torch.device(f"cuda:{idx}")
                else:
                    logger.warning(f"Requested cuda:{idx} but only {torch.cuda.device_count()} device(s); using cuda:0")
                    return torch.device("cuda:0")

        logger.warning(f"Unrecognized device spec '{spec}'; defaulting to CPU.")
        return torch.device("cpu")


    def forward_model(self, model):
        """
        Compute model visibilities from the current image parameters
        using the given model's forward operator.
        """
        if model is None:
            raise ValueError("Must pass a model instance to `forward_model()`.")

        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)

        pb_native = np.asarray(self.pb, dtype=np.float32)
        grid_native = np.asarray(self.grid, dtype=np.float32)

        return model.forward(
            x=self.init_params,
            vis_data=self.vis_data,
            pb=pb_native,
            device=self.device,
            cell_size=cell_size.value,
            grid_array=grid_native
        )


    def process(self, model=None, units="Jy/arcsec^2",
                history_size=10, dtype=torch.float32):
        """
        Devices:
          - optim_device: where PyTorch LBFGS params/optimizer live
          - cost_device : where model.objective() runs
        Rules:
          - positivity==True (bounded): SciPy L-BFGS-B (CPU-only).
          - positivity==False (unbounded): PyTorch LBFGS on optim_device; cost on cost_device.
        NOTE: objective() is expected to call backward() internally (unchanged).
        """
        if model is None:
            raise ValueError("Must pass a model instance to `process()`.")

        cost_dev  = self.cost_device
        optim_dev = self.optim_device

        # ---- local helper: format GPU mem for any CUDA device ----
        def _gpu_mem_str(dev: torch.device) -> str:
            if dev.type != "cuda":
                return ""
            idx = dev.index if (dev.index is not None) else torch.cuda.current_device()
            alloc    = torch.cuda.memory_allocated(idx) / 1024**2
            reserved = torch.cuda.memory_reserved(idx)  / 1024**2
            peak     = torch.cuda.max_memory_allocated(idx) / 1024**2
            total    = torch.cuda.get_device_properties(idx).total_memory / 1024**2
            return f"GPU[{idx}]: {alloc:.2f} MB alloc, {reserved:.2f} MB reserved, {peak:.2f} MB peak, {total:.2f} MB total"

        # --- Image/grid params ---
        cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        tapper = dutils.apodize(0.98, shape)

        # --- FFT beam for reg ---
        kernel_map = dutils.laplacian(shape)
        fftkernel = abs(fft2(kernel_map))
        bmaj_pix = self.beam_sd.major.to(u.deg).value / cell_size.to(u.deg).value
        beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
        fftbeam = abs(fft2(beam))

        # --- FFT single-dish ---
        fftsd = cell_size.value**2 * tfft2(torch.from_numpy(np.float32(self.sd))).cpu().numpy()

        # --- Common params ---
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

        # --- SciPy path (positivity -> bounded -> CPU only)
        if getattr(self, "positivity", False):
            from scipy.optimize import fmin_l_bfgs_b

            if optim_dev.type == "cuda":
                logger.info("positivity=True with optim_device on CUDA → falling back to CPU for SciPy L-BFGS-B.")

            x0 = self.init_params.ravel().astype(np.float64)
            raw_bounds = dutils.ROHSA_bounds(param_shape, lb_amp=0, ub_amp=np.inf)
            bounds64 = [(float(lo), float(hi)) for (lo, hi) in raw_bounds]

            def fun_and_grad(x):
                # cost on cost_dev, but SciPy itself is CPU
                f, g = model.loss(
                    x, shape=param_shape, device=cost_dev,
                    jac=True, **params
                )
                return float(f), np.ascontiguousarray(g, dtype=np.float64)

            logger.info(f"Starting optimisation: SciPy L-BFGS-B (CPU optimizer), cost on {cost_dev}")
            x_opt, f_opt, info = fmin_l_bfgs_b(
                fun_and_grad, x0, bounds=bounds64,
                m=7, pgtol=1e-8, factr=1e7, maxls=20,
                maxiter=int(self.max_its), iprint=25,
            )
            result = x_opt.reshape(param_shape)

        # --- PyTorch path (unbounded LBFGS on optim_device; cost on cost_device)
        else:
            # Reset peak stats on any CUDA devices we might use
            for dev in [cost_dev, optim_dev]:
                if dev.type == "cuda":
                    idx = dev.index if dev.index is not None else torch.cuda.current_device()
                    torch.cuda.reset_peak_memory_stats(idx)

            logger.info(
                f"Starting optimisation: PyTorch LBFGS on {optim_dev} (unconstrained); "
                f"cost on {cost_dev}"
            )
            x_param = torch.tensor(self.init_params, dtype=dtype, device=optim_dev, requires_grad=True)

            opt = torch.optim.LBFGS(
                [x_param],
                lr=1.0,
                max_iter=int(self.max_its),
                history_size=history_size,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-8,
                tolerance_change=0.0,
            )

            def closure():
                opt.zero_grad(set_to_none=True)

                if cost_dev == optim_dev:
                    # Evaluate directly on x_param; objective() will call backward()
                    loss = model.objective(
                        x_param, device=cost_dev, **params
                    )
                    if x_param.grad is None:
                        raise RuntimeError("objective() did not produce gradients on x_param.")
                else:
                    # Cross-device: leaf copy on cost_dev; copy its grad back
                    x_for_cost = x_param.detach().to(cost_dev).requires_grad_(True)
                    loss = model.objective(
                        x_for_cost, device=cost_dev, **params
                    )
                    if x_for_cost.grad is None:
                        raise RuntimeError("objective() did not produce gradients on x_for_cost.")
                    x_param.grad = x_for_cost.grad.to(optim_dev)
                    del x_for_cost  # release ASAP

                # --- logging & sync (log any CUDA device in play) ---
                mem_bits = []
                for dev in [cost_dev, optim_dev]:
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                        mem_bits.append(_gpu_mem_str(dev))

                mem_info = " | ".join(mem_bits)
                logger.info(
                    f"[PID {os.getpid()}] Iter cost: {float(loss.detach().cpu()):.6e} "
                    f"(optim_dev={optim_dev}, cost_dev={cost_dev})"
                    + (f" | {mem_info}" if mem_info else "")
                )
                return loss  # objective already did backward()

            import time
            if cost_dev.type == "cuda":
                torch.cuda.synchronize(cost_dev)
            t0 = time.perf_counter()
            final_loss = opt.step(closure)
            if cost_dev.type == "cuda":
                torch.cuda.synchronize(cost_dev)
            elapsed = time.perf_counter() - t0

            end_mem_bits = []
            for dev in [cost_dev, optim_dev]:
                if dev.type == "cuda":
                    end_mem_bits.append(_gpu_mem_str(dev))
            end_mem_info = " | ".join(end_mem_bits)

            logger.info(
                f"[Timing] LBFGS (optim_dev={optim_dev}, cost_dev={cost_dev}) "
                f"took {elapsed:.2f} s; final loss={float(final_loss):.6g}"
                + (f" | {end_mem_info}" if end_mem_info else "")
            )

            result = x_param.detach().cpu().numpy().reshape(param_shape)

        logger.warning("Multiply by 2 for ASKAP if needed.")

        # --- Unit conversion ---
        if units == "Jy/arcsec^2":
            return result
        elif units == "Jy/beam":
            beam_r = Beam(4.2857 * cell_size, 4.2857 * cell_size, 1.e-12 * u.deg)
            return result * beam_r.sr.to(u.arcsec**2).value
        elif units == "K":
            nu = self.vis_data.frequency[0] * u.Hz
            beam_r = Beam(3 * cell_size, 3 * cell_size, 1.e-12 * u.deg)
            result_Jy = result * beam_r.sr.to(u.arcsec**2).value
            return (result_Jy * u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
        else:
            logger.error("Unknown unit type.")
            return result

        
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


    def forward_model(self, model):
        """
        Compute model visibilities from an input image using the provided model's forward operator.

        Parameters
        ----------
        model : object
            A model instance (e.g., ClassicIViS) that implements a `.forward(...)` method
            to simulate visibilities from image-domain parameters.

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
        
        return model.forward(
            x=self.init_params,
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
    





            # def process(self, model=None, units="Jy/arcsec^2", disk=False):
    #     """
    #     Run imaging optimization.
    #     """
    #     if model is None:
    #         raise ValueError("Must pass a model instance to `process()`.")
    #     if not hasattr(model, "loss"):
    #         raise TypeError("Model must implement `.loss()`.")

    #     # --- Image/grid params ---
    #     cell_size = (self.hdr["CDELT2"] * u.deg).to(u.arcsec)
    #     shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
    #     tapper = dutils.apodize(0.98, shape)

    #     # --- FFT beam for reg ---
    #     kernel_map = dutils.laplacian(shape)
    #     fftkernel = abs(fft2(kernel_map))

    #     bmaj_pix = self.beam_sd.major.to(u.deg).value / cell_size.to(u.deg).value
    #     beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
    #     fftbeam = abs(fft2(beam))

    #     # --- FFT single-dish ---
    #     fftsd = cell_size.value**2 * tfft2(torch.from_numpy(np.float32(self.sd))).numpy()

    #     # --- Bounds ---
    #     param_shape = self.init_params.shape
    #     if self.positivity:
    #         bounds = dutils.ROHSA_bounds(param_shape, lb_amp=0, ub_amp=np.inf)
    #     else:
    #         bounds = dutils.ROHSA_bounds(param_shape, lb_amp=-np.inf, ub_amp=np.inf)

    #     # --- Precompute params dict ---
    #     params = dict(
    #         vis_data=self.vis_data,
    #         pb=np.asarray(self.pb, dtype=np.float32),
    #         fftbeam=np.asarray(fftbeam, dtype=np.float32),
    #         fftsd=np.asarray(fftsd, dtype=np.complex64),
    #         tapper=np.asarray(tapper, dtype=np.float32),
    #         lambda_sd=self.lambda_sd,
    #         fftkernel=np.asarray(fftkernel, dtype=np.float32),
    #         cell_size=cell_size.value,
    #         grid_array=np.asarray(self.grid, dtype=np.float32),
    #         beam_workers=self.beam_workers
    #     )

    #     device = self.device

    #     # --- Closure for optimizer ---
    #     def objective_flat(x):
    #         # Pass through **params so loss() sees vis_data, pb, fftbeam, etc.
    #         return model.loss(x, shape=param_shape, device=device, **params)

    #     # --- Optimize ---
    #     options = dict(maxiter=self.max_its, maxfun=int(1e6), iprint=25)

    #     logger.info("Starting optimisation (using LBFGS-B)")
    #     opt_output = optimize.minimize(
    #         objective_flat,
    #         self.init_params.ravel().astype(np.float32),
    #         jac=True,
    #         tol=1e-8,
    #         bounds=bounds,
    #         method="L-BFGS-B",
    #         options=options
    #     )

    #     result = np.reshape(opt_output.x, self.init_params.shape)
    #     logger.warning("Multiply by 2 for ASKAP if needed.")

    #     # --- Unit conversion ---
    #     if units == "Jy/arcsec^2":
    #         return result
    #     elif units == "Jy/beam":
    #         beam_r = Beam(4.2857 * cell_size, 4.2857 * cell_size, 1.e-12 * u.deg)
    #         return result * beam_r.sr.to(u.arcsec**2).value
    #     elif units == "K":
    #         nu = self.vis_data.frequency[0] * u.Hz
    #         beam_r = Beam(3 * cell_size, 3 * cell_size, 1.e-12 * u.deg)
    #         result_Jy = result * beam_r.sr.to(u.arcsec**2).value
    #         return (result_Jy * u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
    #     else:
    #         logger.error("Unknown unit type.")
    #         return result

