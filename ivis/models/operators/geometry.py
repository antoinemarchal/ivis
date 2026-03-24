import numpy as np
import torch


def resolve_pb_grid_lists(vis_data, pb_list=None, grid_list=None, pb=None, grid_array=None):
    nbeam = len(vis_data.centers)

    if pb_list is None:
        if pb is None:
            raise ValueError("Need pb_list or pb array")
        pb_list = [pb[b] for b in range(nbeam)]

    if grid_list is None:
        if grid_array is None:
            raise ValueError("Need grid_list or grid_array")
        grid_list = [grid_array[b] for b in range(nbeam)]

    return pb_list, grid_list


def to_image2d_tensor(x2d, device, name="x2d"):
    xt = torch.as_tensor(x2d, device=device)
    if xt.ndim == 1:
        side = int(np.sqrt(xt.numel()))
        xt = xt.view(side, side)
    if xt.ndim != 2:
        raise ValueError(f"{name} must be 2D (H,W), got shape {tuple(xt.shape)}")
    return xt


def uvw_to_radpix(uu, vv, cell_size, device):
    cell_rad = cell_size * np.pi / (180.0 * 3600.0)
    u_radpix = torch.from_numpy(uu).to(device).float() * (2.0 * np.pi * cell_rad)
    v_radpix = torch.from_numpy(vv).to(device).float() * (2.0 * np.pi * cell_rad)
    return cell_rad, u_radpix, v_radpix
