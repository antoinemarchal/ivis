import numpy as np
import torch

from ivis.models.operators.geometry import to_image2d_tensor


def forward_reprojection_with_primary_beam(x2d, primary_beam, grid, device):
    xt = to_image2d_tensor(x2d, device, name="x2d")
    xt = xt.unsqueeze(0).unsqueeze(0).float().to(device)
    grid_t = torch.from_numpy(np.asarray(grid)).to(device).float()
    repro = torch.nn.functional.grid_sample(
        xt, grid_t, mode="bilinear", align_corners=True
    ).squeeze(0).squeeze(0)
    primary_beam_t = torch.from_numpy(np.asarray(primary_beam)).to(device).float()
    return repro * primary_beam_t


def backward_reprojection_autodiff(z2d, grid, image_shape, device):
    grid_t = torch.from_numpy(np.asarray(grid)).to(device).float()

    zt = to_image2d_tensor(z2d, device, name="z2d").to(torch.complex64)
    template = torch.zeros((1, 1, *image_shape), dtype=torch.float32, device=device, requires_grad=True)
    repro = torch.nn.functional.grid_sample(
        template, grid_t, mode="bilinear", align_corners=True
    ).squeeze(0).squeeze(0)

    grad_real = torch.autograd.grad(
        torch.sum(repro * zt.real), template, retain_graph=True
    )[0].squeeze(0).squeeze(0)
    grad_imag = torch.autograd.grad(
        torch.sum(repro * zt.imag), template
    )[0].squeeze(0).squeeze(0)

    return torch.complex(grad_real, grad_imag)


def get_reprojection_interp_cache(grid, image_shape, device, cache_store):
    grid_arr = np.asarray(grid)
    grid_key = (
        id(grid),
        tuple(image_shape),
        str(torch.device(device)),
        tuple(grid_arr.shape),
    )
    cached = cache_store.get(grid_key)
    if cached is not None:
        return cached

    grid_t = torch.from_numpy(grid_arr).to(device).float()
    if grid_t.ndim != 4 or grid_t.shape[0] != 1 or grid_t.shape[-1] != 2:
        raise ValueError(f"grid must have shape (1,H,W,2), got {tuple(grid_t.shape)}")

    hin, win = image_shape
    gout_h, gout_w = grid_t.shape[1], grid_t.shape[2]
    gx = grid_t[0, :, :, 0]
    gy = grid_t[0, :, :, 1]

    x = torch.zeros_like(gx) if win == 1 else 0.5 * (gx + 1.0) * (win - 1)
    y = torch.zeros_like(gy) if hin == 1 else 0.5 * (gy + 1.0) * (hin - 1)

    x0 = torch.floor(x).to(torch.int64)
    y0 = torch.floor(y).to(torch.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx1 = x - x0.to(x.dtype)
    wy1 = y - y0.to(y.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    def corner(ix, iy, w):
        valid = (ix >= 0) & (ix < win) & (iy >= 0) & (iy < hin)
        idx = torch.zeros_like(ix, dtype=torch.int64)
        idx[valid] = iy[valid] * win + ix[valid]
        weight = torch.zeros_like(w, dtype=torch.float32)
        weight[valid] = w[valid].float()
        return idx.reshape(-1), weight.reshape(-1)

    idx00, w00 = corner(x0, y0, wx0 * wy0)
    idx10, w10 = corner(x1, y0, wx1 * wy0)
    idx01, w01 = corner(x0, y1, wx0 * wy1)
    idx11, w11 = corner(x1, y1, wx1 * wy1)

    cached = {
        "grid_t": grid_t,
        "out_shape": (gout_h, gout_w),
        "idx00": idx00,
        "idx10": idx10,
        "idx01": idx01,
        "idx11": idx11,
        "w00": w00,
        "w10": w10,
        "w01": w01,
        "w11": w11,
    }
    cache_store[grid_key] = cached
    return cached


def backward_reprojection_manual(z2d, grid, image_shape, device, cache_store):
    zt = to_image2d_tensor(z2d, device, name="z2d").to(torch.complex64)

    cache = grid if isinstance(grid, dict) else get_reprojection_interp_cache(
        grid, image_shape, device, cache_store
    )
    hin, win = image_shape
    gout_h, gout_w = cache["out_shape"]
    if zt.shape != (gout_h, gout_w):
        raise ValueError(f"z2d shape {tuple(zt.shape)} does not match grid output shape {(gout_h, gout_w)}")

    acc_real = torch.zeros(hin * win, dtype=torch.float32, device=device)
    acc_imag = torch.zeros(hin * win, dtype=torch.float32, device=device)
    flat_real = zt.real.reshape(-1)
    flat_imag = zt.imag.reshape(-1)

    for idx_name, w_name in (("idx00", "w00"), ("idx10", "w10"), ("idx01", "w01"), ("idx11", "w11")):
        idx = cache[idx_name]
        weight = cache[w_name]
        acc_real.scatter_add_(0, idx, flat_real * weight)
        acc_imag.scatter_add_(0, idx, flat_imag * weight)

    return torch.complex(acc_real.view(hin, win), acc_imag.view(hin, win))
