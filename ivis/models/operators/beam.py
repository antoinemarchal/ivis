import numpy as np
import torch
import torch.nn.functional as F

from ivis.models.operators.geometry import to_image2d_tensor, uvw_to_radpix
from ivis.models.operators.nufft import backward_nufft, forward_nufft
from ivis.models.operators.reprojection import (
    backward_reprojection_manual,
    get_reprojection_interp_cache,
)


def _get_uv_cache(uu, vv, cell_size, device, cache_store):
    if cache_store is None:
        return uvw_to_radpix(uu=uu, vv=vv, cell_size=cell_size, device=device)

    cache_key = (
        "uv_radpix",
        id(uu),
        id(vv),
        float(cell_size),
        str(torch.device(device)),
    )
    cached = cache_store.get(cache_key)
    if cached is not None:
        return cached

    cached = uvw_to_radpix(uu=uu, vv=vv, cell_size=cell_size, device=device)
    cache_store[cache_key] = cached
    return cached


def _get_primary_beam_tensor(primary_beam, device, cache_store):
    if cache_store is None:
        return torch.from_numpy(np.asarray(primary_beam)).to(device).float()

    primary_beam_arr = np.asarray(primary_beam)
    cache_key = (
        "primary_beam_t",
        id(primary_beam),
        str(torch.device(device)),
        tuple(primary_beam_arr.shape),
    )
    cached = cache_store.get(cache_key)
    if cached is not None:
        return cached

    cached = torch.from_numpy(primary_beam_arr).to(device).float()
    cache_store[cache_key] = cached
    return cached


def forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device, cache_store=None):
    xt = to_image2d_tensor(x2d, device, name="x2d").unsqueeze(0).unsqueeze(0).float()
    if cache_store is None:
        grid_t = torch.from_numpy(np.asarray(grid)).to(device).float()
    else:
        grid_t = get_reprojection_interp_cache(
            grid, tuple(xt.shape[-2:]), device, cache_store
        )["grid_t"]
    repro = F.grid_sample(
        xt, grid_t, mode="bilinear", align_corners=True
    ).squeeze(0).squeeze(0)
    primary_beam_t = _get_primary_beam_tensor(primary_beam, device, cache_store)
    x_primary_beam = repro * primary_beam_t
    _, u_radpix, v_radpix = _get_uv_cache(
        uu=uu, vv=vv, cell_size=cell_size, device=device, cache_store=cache_store
    )
    return forward_nufft(
        x_pb=x_primary_beam,
        u_radpix=u_radpix,
        v_radpix=v_radpix,
        cell_size=cell_size,
    )


def backward_beam(y, primary_beam, grid, uu, vv, ww, cell_size, image_shape, device, cache_store):
    yt = torch.as_tensor(y, device=device).to(torch.complex64).reshape(-1)
    primary_beam_t = _get_primary_beam_tensor(primary_beam, device, cache_store)
    primary_beam_shape = tuple(primary_beam_t.shape)
    _, u_radpix, v_radpix = _get_uv_cache(
        uu=uu, vv=vv, cell_size=cell_size, device=device, cache_store=cache_store
    )
    dirty_pb = backward_nufft(
        y=yt,
        pb_shape=primary_beam_shape,
        u_radpix=u_radpix,
        v_radpix=v_radpix,
        cell_size=cell_size,
    )

    grid_cache = get_reprojection_interp_cache(grid, image_shape, device, cache_store)
    return backward_reprojection_manual(
        z2d=dirty_pb * primary_beam_t,
        grid=grid_cache,
        image_shape=image_shape,
        device=device,
        cache_store=cache_store,
    )
