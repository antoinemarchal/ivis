import numpy as np
import torch
from astropy.wcs.utils import pixel_to_pixel
from astropy.wcs import WCS

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

# def ROHSA_bounds(data_shape, lb_amp, ub_amp):
#     bounds_inf = np.zeros(data_shape)
#     bounds_sup = np.zeros(data_shape)

#     bounds_a = [lb_amp, ub_amp]

#     bounds_inf[:,:] = bounds_a[0]
#     bounds_sup[:,:] = bounds_a[1]

#     bounds = [(bounds_inf.ravel()[i], bounds_sup.ravel()[i]) for i in np.arange(len(bounds_sup.ravel()))]
                 
#     return np.array(bounds)


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

