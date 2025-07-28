import numpy as np
import torch
from astropy.wcs.utils import pixel_to_pixel
from astropy import wcs
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt

def wcs2D(hdr):
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
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4.


def laplacian(shape):
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
    bounds_inf = np.full(data_shape, lb_amp)
    bounds_sup = np.full(data_shape, ub_amp)
    
    return np.column_stack((bounds_inf.ravel(), bounds_sup.ravel()))


def gauss_beam(sigma, shape, FWHM=False):
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
    # Ensure the input tensor has 4 dimensions
    if input_tensor.dim() == 2:  # If shape is [H_in, W_in]
        input_tensor_reshape = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    elif input_tensor.dim() == 3:  # If shape is [C, H_in, W_in]
        input_tensor_reshape = input_tensor.unsqueeze(0)  # Add batch dim
        
    return input_tensor_reshape


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
