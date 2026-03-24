import numpy as np
import torch

from ivis.models.operators.geometry import uvw_to_radpix
from ivis.models.operators.nufft import backward_nufft, forward_nufft
from ivis.models.operators.reprojection import (
    backward_reprojection_manual,
    forward_reprojection_with_primary_beam,
    get_reprojection_interp_cache,
)


def forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device):
    x_primary_beam = forward_reprojection_with_primary_beam(
        x2d=x2d, primary_beam=primary_beam, grid=grid, device=device
    )
    _, u_radpix, v_radpix = uvw_to_radpix(uu=uu, vv=vv, cell_size=cell_size, device=device)
    return forward_nufft(
        x_pb=x_primary_beam,
        u_radpix=u_radpix,
        v_radpix=v_radpix,
        cell_size=cell_size,
    )


def backward_beam(y, primary_beam, grid, uu, vv, ww, cell_size, image_shape, device, cache_store):
    yt = torch.as_tensor(y, device=device).to(torch.complex64).reshape(-1)
    primary_beam_arr = np.asarray(primary_beam)
    primary_beam_shape = tuple(primary_beam_arr.shape)
    _, u_radpix, v_radpix = uvw_to_radpix(uu=uu, vv=vv, cell_size=cell_size, device=device)
    dirty_pb = backward_nufft(
        y=yt,
        pb_shape=primary_beam_shape,
        u_radpix=u_radpix,
        v_radpix=v_radpix,
        cell_size=cell_size,
    )

    primary_beam_t = torch.from_numpy(primary_beam_arr).to(device).float()
    grid_cache = get_reprojection_interp_cache(grid, image_shape, device, cache_store)
    return backward_reprojection_manual(
        z2d=dirty_pb * primary_beam_t,
        grid=grid_cache,
        image_shape=image_shape,
        device=device,
        cache_store=cache_store,
    )
