# -*- coding: utf-8 -*-
"""
===================================
Utility Functions for IViS Imaging
===================================

This module provides a collection of utility functions used in IViS
for radio interferometric reconstruction, model fitting, and image-domain operations.

It includes tools for:

- WCS construction and reprojection using `astropy.wcs`
- Edge apodization (cosine taper) for windowing
- Construction of Laplacian kernels for regularization
- Synthetic Gaussian beam generation and injection
- Coordinate grid generation for PyTorch-based warping
- Elliptical Gaussian fitting for beam or source characterization

Functions
---------
- wcs2D: Build a 2D WCS object from a FITS header.
- apodize: Create a cosine taper for edge apodization.
- ROHSA_kernel: Return a discrete Laplacian kernel used in ROHSA.
- laplacian: Create a Laplacian kernel padded into a full-size map.
- ROHSA_bounds: Generate lower and upper parameter bounds.
- gauss_beam: Generate a normalized 2D Gaussian kernel.
- get_grid: Generate a sampling grid for torch.nn.functional.grid_sample.
- format_input_tensor: Reshape tensors for compatibility with PyTorch ops.
- insert_elliptical_gaussian_source: Inject an elliptical Gaussian model into an image.
- fit_elliptical_gaussian: Fit an elliptical Gaussian and visualize the result.

Dependencies
------------
- numpy
- torch
- matplotlib
- astropy

Author
------
Antoine Marchal, 2024–2025
"""

import numpy as np
import torch
import torch.nn.functional as F
from astropy.wcs.utils import pixel_to_pixel
from astropy import wcs
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

from ivis.logger import logger

def gpu_mem_str(dev: torch.device) -> str:
    """
    Return a formatted string describing CUDA memory usage for a device.
    Returns empty string if dev is not a CUDA device.
    """
    if dev.type != "cuda":
        return ""

    idx = dev.index if (dev.index is not None) else torch.cuda.current_device()

    alloc = torch.cuda.memory_allocated(idx) / 1024**2
    reserved = torch.cuda.memory_reserved(idx) / 1024**2
    peak = torch.cuda.max_memory_allocated(idx) / 1024**2
    total = torch.cuda.get_device_properties(idx).total_memory / 1024**2

    return (
        f"GPU[{idx}]: "
        f"{alloc:.2f} MB alloc, "
        f"{reserved:.2f} MB reserved, "
        f"{peak:.2f} MB peak, "
        f"{total:.2f} MB total"
    )


def get_device(spec="auto") -> torch.device:
    """
    Resolve a compute device from a flexible spec:
      - "auto"       -> cuda:0 if available; else mps; else cpu
      - "cpu"        -> cpu
      - "cuda"       -> cuda:0 (if available)
      - "cuda:i"     -> cuda:i if available
      - "mps"        -> Apple MPS if available
      - int i        -> cuda:i if available; else cpu
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
                logger.warning(
                    f"Requested cuda:{idx} but only {torch.cuda.device_count()} device(s); using cuda:0"
                )
                return torch.device("cuda:0")
        logger.info("CUDA unavailable or invalid index; using CPU.")
        return torch.device("cpu")

    # string spec
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

            idx = 0
            if ":" in s:
                try:
                    idx = int(s.split(":", 1)[1])
                except Exception:
                    logger.warning(
                        f"Could not parse device index from '{spec}', defaulting to cuda:0."
                    )
                    idx = 0

            if idx < torch.cuda.device_count():
                logger.info(f"Using GPU cuda:{idx} ({torch.cuda.get_device_name(idx)})")
                return torch.device(f"cuda:{idx}")

            logger.warning(
                f"Requested cuda:{idx} but only {torch.cuda.device_count()} device(s); using cuda:0"
            )
            return torch.device("cuda:0")

    logger.warning(f"Unrecognized device spec '{spec}'; defaulting to CPU.")
    return torch.device("cpu")


def _to_nchw(arr: np.ndarray, expect_complex: bool = False) -> torch.Tensor:
    """
    Convert numpy array into 4D NCHW tensor for interpolation.

    - If expect_complex=False: arr is (H,W), (B,H,W), or (N,B,H,W)
      returns (N,1,H,W).
    - If expect_complex=True: arr has trailing axis 2 for (real,imag)
      e.g. (H,W,2), (B,H,W,2), or (N,1,H,W,2).
      returns (N,2,H,W).
    """
    arr = np.array(arr, dtype=np.float32, order="C", copy=False)
    t = torch.from_numpy(arr)

    if expect_complex:
        if t.ndim == 3 and t.shape[-1] == 2:
            # (H,W,2) → (1,2,H,W)
            t = t.permute(2, 0, 1).unsqueeze(0)
        elif t.ndim == 4 and t.shape[-1] == 2:
            # (B,H,W,2) → (B,2,H,W)
            t = t.permute(0, 3, 1, 2)
        elif t.ndim == 5 and t.shape[1] == 1 and t.shape[-1] == 2:
            # (N,1,H,W,2) → (N,2,H,W)
            t = t.squeeze(1).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported complex shape {arr.shape}")
    else:
        if t.ndim == 2:
            # (H,W) → (1,1,H,W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            # (B,H,W) → (B,1,H,W)
            t = t.unsqueeze(1)
        elif t.ndim == 4 and t.shape[1] == 1:
            # (N,1,H,W) already fine
            pass
        else:
            raise ValueError(f"Unsupported real shape {arr.shape}")

    return t.float()


def downsample_pb(pb: np.ndarray, factor: int) -> np.ndarray:
    """Downsample primary beam array (real, 2D/3D) with bilinear interpolation."""
    t = _to_nchw(pb, expect_complex=False)
    small = F.interpolate(t, scale_factor=1.0/factor,
                          mode="bilinear", align_corners=False)
    return small.squeeze(0).squeeze(0).cpu().numpy()


def downsample_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """Downsample uv-grid array with trailing complex axis (...,2)."""
    t = _to_nchw(grid, expect_complex=True)
    small = F.interpolate(t, scale_factor=1.0/factor,
                          mode="bilinear", align_corners=False)
    # Back to (N,H,W,2)
    return small.permute(0, 2, 3, 1).cpu().numpy()



def downsample_hdr(hdr, factor: int):
    """Return a copy of FITS-like header with updated NAXIS and CDELT."""
    hdr = hdr.copy()
    hdr["NAXIS1"] //= factor
    hdr["NAXIS2"] //= factor
    hdr["CDELT1"] *= factor
    hdr["CDELT2"] *= factor
    return hdr


def wcs2D(hdr):
    """
    Construct a 2D WCS object from a FITS header.

    Parameters
    ----------
    hdr : dict-like
        FITS header containing WCS keywords.

    Returns
    -------
    w : astropy.wcs.WCS
        2D WCS object.
    """
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w


def apodize(radius, shape):
    """
    from JF Robitaille package.
    Create edges apodization tapper
    
    Parameters
    ----------
    nx, ny : integers
    size of the tapper
    radius : float
    radius must be lower than 1 and greater than 0.
    
    Returns
    -------
    
    tapper : numpy array ready to multiply on your image
    to apodize edges
    """
    ny = shape[0]
    nx = shape[1]

    if (radius >= 1) or (radius <= 0.):
        print('Error: radius must be lower than 1 and greater than 0.')
        return
        
    ni = np.fix(radius*nx)
    dni = int(nx-ni)
    nj = np.fix(radius*ny)
    dnj = int(ny-nj)
    
    tap1d_x = np.ones(nx)
    tap1d_y = np.ones(ny)
    
    tap1d_x[0:dni] = (np.cos(3. * np.pi/2. + np.pi/2.* (1.* np.arange(dni)/(dni-1)) ))
    tap1d_x[nx-dni:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dni)/(dni-1)) ))
    tap1d_y[0:dnj] = (np.cos(3. * np.pi/2. + np.pi/2. * (1.* np.arange( dnj )/(dnj-1)) ))
    tap1d_y[ny-dnj:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dnj)/(dnj-1)) ))
    
    tapper = np.zeros((ny, nx))
    
    for i in range(nx):
        tapper[:,i] = tap1d_y
                        
    for i in range(ny):
        tapper[i,:] = tapper[i,:] * tap1d_x
        
    return tapper


def ROHSA_kernel():
    """
    Return a Laplacian-like kernel used in ROHSA.

    Returns
    -------
    kernel : ndarray
        3x3 Laplacian kernel normalized by 1/4.
    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4.


def laplacian(shape):
    """
    Construct a 2D Laplacian kernel map for convolution in Fourier space.

    Parameters
    ----------
    shape : tuple
        Shape of the output Laplacian map (ny, nx).

    Returns
    -------
    kernel_map : ndarray
        Laplacian kernel zero-padded in a map of the given shape.
    """
    ny, nx = shape
    X=np.arange(nx)
    Y=np.arange(ny)
    ymap,xmap=np.meshgrid(X,Y)

    kernel = ROHSA_kernel()
    kernel_map = np.zeros(shape)

    center_x = nx // 2
    center_y = ny // 2

    # if (nx % 2) == 0:
    
    kernel_map[center_y-1:center_y+2,center_x-1:center_x+2] = kernel
    # else:    
    #     kernel_map[center_y-1:center_y+2,center_x-1:center_x+2] = kernel
    
    return kernel_map


def ROHSA_bounds(data_shape, lb_amp, ub_amp):
    """
    Create lower and upper bounds arrays for ROHSA optimization.

    Parameters
    ----------
    data_shape : tuple
        Shape of the model parameter array.
    lb_amp : float
        Lower bound for amplitude.
    ub_amp : float
        Upper bound for amplitude.

    Returns
    -------
    bounds : ndarray
        Array of shape (N, 2), with N = np.prod(data_shape), where each row is (lower, upper).
    """
    bounds_inf = np.full(data_shape, lb_amp)
    bounds_sup = np.full(data_shape, ub_amp)
    
    return np.column_stack((bounds_inf.ravel(), bounds_sup.ravel()))


def gauss_beam(sigma, shape, FWHM=False):
    """
    Generate a circular symmetric 2D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian in pixels (or FWHM if `FWHM=True`).
    shape : tuple
        Shape of the output map (ny, nx).
    FWHM : bool, optional
        If True, `sigma` is interpreted as FWHM. Default is False.

    Returns
    -------
    gauss : ndarray
        2D normalized Gaussian array.
    """
    ny, nx = shape
    X=np.arange(nx)
    Y=np.arange(ny)
    ymap,xmap=np.meshgrid(X,Y)

    if (nx % 2) == 0:
        xmap = xmap - (nx)/2.
    else:
        xmap = xmap - (nx-1.)/2.

    if (ny % 2) == 0:
        ymap = ymap - (ny)/2.
    else:
        ymap = ymap - (ny-1.)/2.

    map = np.sqrt(xmap**2.+ymap**2.)

    if FWHM == True:
        sigma = sigma / (2.*np.sqrt(2.*np.log(2.)))

    gauss = np.exp(-0.5*(map)**2./sigma**2.)
    gauss /= np.sum(gauss)

    return gauss


def get_grid(shape_input_tensor, wcs_in, wcs_out, shape_out):
    """
    Create a normalized sampling grid for spatial reprojecting in PyTorch.

    Parameters
    ----------
    shape_input_tensor : tuple
        Shape of the input tensor: (B, C, H_in, W_in).
    wcs_in : astropy.wcs.WCS
        WCS of the input image.
    wcs_out : astropy.wcs.WCS
        Target WCS for output grid.
    shape_out : tuple
        Shape of the output image (H_out, W_out).

    Returns
    -------
    grid : torch.Tensor
        Grid of shape (1, H_out, W_out, 2) normalized to [-1, 1].
    """
    # Generate the output grid coordinates
    x_out, y_out = torch.meshgrid(
        torch.arange(shape_out[1], dtype=torch.float32), #FIXME
        torch.arange(shape_out[0], dtype=torch.float32), #FIXME
        indexing='xy'
    )

    # Convert output pixel coordinates to input pixel coordinates
    # world_coords = wcs_out.pixel_to_world(x_out.numpy(), y_out.numpy())
    # y_in, x_in = wcs_in.world_to_pixel(world_coords)
    y_in, x_in = pixel_to_pixel(wcs_out, wcs_in, x_out.numpy(), y_out.numpy())

    # Convert to PyTorch tensors
    x_in_tensor = torch.tensor(x_in, dtype=torch.float32)
    y_in_tensor = torch.tensor(y_in, dtype=torch.float32)

    # Stack coordinates for grid sampling
    grid = torch.stack((y_in_tensor, x_in_tensor), dim=-1)  # Shape: [H, W, 2]

    # Add batch dimension to the grid
    grid = grid.unsqueeze(0)  # Shape: [1, H, W, 2]

    # Normalize grid coordinates to [-1, 1]
    grid[..., 0] = (grid[..., 0] / (shape_input_tensor[2] - 1)) * 2 - 1  # Normalize y
    grid[..., 1] = (grid[..., 1] / (shape_input_tensor[3] - 1)) * 2 - 1  # Normalize x

    return grid


def format_input_tensor(input_tensor):
    """
    Format a 2D or 3D input tensor into a 4D tensor for grid sampling.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor of shape (H, W) or (C, H, W).

    Returns
    -------
    input_tensor_reshape : torch.Tensor
        Tensor of shape (1, C, H, W).
    """
    # Ensure the input tensor has 4 dimensions
    if input_tensor.dim() == 2:  # If shape is [H_in, W_in]
        input_tensor_reshape = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif input_tensor.dim() == 3:  # If shape is [C, H_in, W_in]
        input_tensor_reshape = input_tensor.unsqueeze(0)  # Add batch dim
        
    return input_tensor_reshape


def insert_elliptical_gaussian_source(shape, cell_size, flux_jy=1.0,
                                      fwhm_maj_arcsec=15.0, fwhm_min_arcsec=7.5,
                                      pa_deg=0.0, center=None):
    """
    Generate a sky model with an elliptical Gaussian source.

    Parameters
    ----------
    shape : tuple
        Output image shape (ny, nx).
    cell_size : float
        Pixel size in arcsec.
    flux_jy : float, optional
        Total integrated flux of the source in Jy. Default is 1.0.
    fwhm_maj_arcsec : float, optional
        FWHM of the major axis in arcsec. Default is 15.0.
    fwhm_min_arcsec : float, optional
        FWHM of the minor axis in arcsec. Default is 7.5.
    pa_deg : float, optional
        Position angle in degrees (counter-clockwise from x-axis). Default is 0.0.
    center : tuple, optional
        Pixel coordinates of the source center (y, x). Default is center of image.

    Returns
    -------
    sky_model : ndarray
        2D image array (float32) in units of Jy/pixel.
    """
    ny, nx = shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')

    # Set center
    if center is None:
        cy, cx = ny // 2, nx // 2
    else:
        cy, cx = center

    # Convert FWHM to sigma in pixels
    sigma_maj_pix = (fwhm_maj_arcsec / 2.3548) / cell_size
    sigma_min_pix = (fwhm_min_arcsec / 2.3548) / cell_size
    theta_rad = np.deg2rad(pa_deg)

    # Rotate coordinate grid
    dx = x - cx
    dy = y - cy
    dx_rot = dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
    dy_rot = -dx * np.sin(theta_rad) + dy * np.cos(theta_rad)

    # Elliptical 2D Gaussian (unit integral)
    gaussian = np.exp(-0.5 * ((dx_rot / sigma_maj_pix)**2 + (dy_rot / sigma_min_pix)**2))
    gaussian /= np.sum(gaussian) * cell_size**2  # now in Jy/arcsec²

    # Scale to desired flux
    sky_model = gaussian * flux_jy
    return sky_model.astype(np.float32)


def fit_elliptical_gaussian(cutout, pixel_scale_arcsec=1.0):
    """
    Fit elliptical Gaussian to image cutout and return flux, Bmaj, Bmin in arcsec.

    Parameters
    ----------
    cutout : 2D ndarray
        Image array containing a single source, in units of Jy/arcsec^2.
    pixel_scale_arcsec : float
        Pixel size in arcsec/pixel.

    Returns
    -------
    flux : float
        Integrated flux in Jy.
    Bmaj : float
        FWHM of major axis in arcsec.
    Bmin : float
        FWHM of minor axis in arcsec.
    theta : float
        Position angle in degrees (CCW from +x).
    """
    y, x = np.mgrid[:cutout.shape[0], :cutout.shape[1]]

    # Initial guess: symmetric Gaussian at center
    amp_guess = np.max(cutout)
    x0 = cutout.shape[1] / 2
    y0 = cutout.shape[0] / 2
    sigma_guess = 1.5

    gauss_init = models.Gaussian2D(
        amplitude=amp_guess,
        x_mean=x0,
        y_mean=y0,
        x_stddev=sigma_guess,
        y_stddev=sigma_guess,
        theta=0.0
    )

    fitter = fitting.LevMarLSQFitter()
    fitted = fitter(gauss_init, x, y, cutout)

    # Extract fit parameters
    sigma_x = fitted.x_stddev.value  # in pixels
    sigma_y = fitted.y_stddev.value  # in pixels
    theta = np.rad2deg(fitted.theta.value)  # radians → degrees

    # Convert sigma to arcsec
    sigma_x_arcsec = sigma_x * pixel_scale_arcsec
    sigma_y_arcsec = sigma_y * pixel_scale_arcsec

    # Convert to FWHM (FWHM = 2.3548 * sigma)
    FWHM_x = 2.3548 * sigma_x_arcsec
    FWHM_y = 2.3548 * sigma_y_arcsec

    # Integrated flux in Jy = amp (Jy/arcsec²) × 2πσxσy [in arcsec²]
    flux = fitted.amplitude.value * 2 * np.pi * sigma_x_arcsec * sigma_y_arcsec

    # Ensure Bmaj ≥ Bmin
    if FWHM_x >= FWHM_y:
        Bmaj, Bmin = FWHM_x, FWHM_y
    else:
        Bmaj, Bmin = FWHM_y, FWHM_x
        theta += 90.0  # adjust PA if axes swapped

    # Plot input and fitted model
    model_data = fitted(x, y)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cutout, origin='lower', cmap='inferno')
    plt.colorbar(label='Jy/arcsec$^{2}$')
    plt.title('Input Cutout')

    plt.subplot(1, 2, 2)
    plt.imshow(cutout, origin='lower', cmap='inferno')
    plt.contour(model_data, levels=5, colors='white', linewidths=1)
    plt.colorbar(label='Jy/arcsec²')
    plt.title('Fitted Gaussian Contours')

    plt.suptitle(f"Flux = {flux:.3f} Jy   Bmaj = {Bmaj:.2f}\"   Bmin = {Bmin:.2f}\"   PA = {theta:.1f}°")
    plt.tight_layout()
    plt.show()

    return flux, Bmaj, Bmin, theta
