import os
import numpy as np
import torch
from torch.fft import fft2 as tfft2

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.operators import (
    backward_beam,
    forward_beam,
    resolve_pb_grid_lists,
)
from ivis.models.utils.gpu import print_gpu_memory

class Classic3D(BaseModel):
    def __init__(self, lambda_r=1, Nw=None, use_2pi=True, conj_data=True):
        self.lambda_r = lambda_r
        self.Nw = None
        self.use_2pi = use_2pi
        self.conj_data = conj_data
        self._interp_cache = {}

    def loss(self, x, shape, device, vis_data, **kwargs):
        u = x.reshape(shape)
        u = torch.from_numpy(u).to(device).requires_grad_(True)

        L = self.objective(x=u, vis_data=vis_data, device=device, **kwargs)
        grad = u.grad.cpu().numpy().astype(x.dtype)

        if torch.device(device).type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            total = torch.cuda.get_device_properties(device).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                f"GPU: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total"
            )
        else:
            logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

        return L.item(), grad.ravel()

    @torch.no_grad()
    def forward(
        self,
        x,
        vis_data,
        device,
        primary_beam_list=None,
        primary_beam=None,
        pb_list=None,
        grid_list=None,
        pb=None,
        grid_array=None,
        cell_size=None,
        fill_flagged="zero",
    ):
        dev = torch.device(device)
        x = torch.as_tensor(x, device=dev).float()
        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        if not hasattr(vis_data, "data_I"):
            raise ValueError("vis_data must have data_I to infer the output cube shape.")
        nchan, nbeam, nvis = vis_data.data_I.shape
        out = np.zeros((nchan, nbeam, nvis), dtype=np.complex64)
        has_flags = hasattr(vis_data, "flag_I")

        for c, b, Icb, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            model_vis = forward_beam(
                x2d=x[c],
                primary_beam=primary_beam_list[b],
                grid=grid_list[b],
                uu=uu,
                vv=vv,
                ww=ww,
                cell_size=cell_size,
                device=dev,
            ).detach().cpu().numpy().astype(np.complex64)
            nv = int(vis_data.nvis[b])
            if nv > nvis:
                raise ValueError(f"vis_data.nvis[{b}]={nv} exceeds cube width {nvis}.")
            good = ~np.asarray(vis_data.flag_I[c, b, :nv], dtype=bool)
            if model_vis.size != int(np.count_nonzero(good)):
                raise ValueError(
                    f"Model visibility count {model_vis.size} does not match "
                    f"unflagged slot count {int(np.count_nonzero(good))} for channel {c}, beam {b}."
                )
            out[c, b, :nv][good] = model_vis
            if has_flags and fill_flagged == "zero":
                fl = np.asarray(vis_data.flag_I[c, b], dtype=bool)
                if fl.shape == out[c, b].shape:
                    out[c, b][fl] = 0.0

        return out

    @torch.no_grad()
    def backward(
        self,
        vis,
        vis_data,
        device,
        x_shape=None,
        primary_beam_list=None,
        primary_beam=None,
        pb_list=None,
        grid_list=None,
        pb=None,
        grid_array=None,
        cell_size=None,
        return_real=False,
    ):
        dev = torch.device(device)
        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        if x_shape is None:
            if vis_data is None or not hasattr(vis_data, "data_I"):
                raise ValueError("Need x_shape or vis_data.data_I to infer image cube shape.")
            x_shape = (vis_data.data_I.shape[0],) + tuple(np.asarray(primary_beam_list[0]).shape)

        if len(x_shape) != 3:
            raise ValueError(f"x_shape must be (nchan, H, W), got {x_shape}")
        nchan, height, width = x_shape

        vis_in = vis_data.data_I if vis is None else vis
        use_cube = hasattr(vis_in, "shape") and tuple(vis_in.shape) == tuple(vis_data.data_I.shape)
        use_flat = not use_cube

        if use_flat:
            flat_vis = np.asarray(vis_in, dtype=np.complex64).reshape(-1)
            offset = 0

        result = torch.zeros((nchan, height, width), dtype=torch.complex64, device=dev)

        for c, b, Icb, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            if use_cube:
                nv = int(vis_data.nvis[b])
                flg = np.asarray(vis_data.flag_I[c, b, :nv], dtype=bool)
                y = np.asarray(vis_in[c, b, :nv], dtype=np.complex64)[~flg]
            else:
                block_size = Icb.size
                y = flat_vis[offset:offset + block_size]
                if y.size != block_size:
                    raise ValueError("Flat visibility vector is shorter than expected from vis_data.")
                offset += block_size

            result[c] = result[c] + backward_beam(
                y=y,
                primary_beam=primary_beam_list[b],
                grid=grid_list[b],
                uu=uu,
                vv=vv,
                ww=ww,
                cell_size=cell_size,
                image_shape=(height, width),
                device=dev,
                cache_store=self._interp_cache,
            )

        if use_flat and offset != flat_vis.size:
            raise ValueError("Flat visibility vector is longer than expected from vis_data.")

        result = result.real if return_real else result
        return result.detach().cpu().numpy().astype(np.float32 if return_real else np.complex64)

    def objective(
        self,
        x,
        vis_data,
        device,
        primary_beam_list=None,
        primary_beam=None,
        pb_list=None,
        grid_list=None,
        pb=None,
        grid_array=None,
        cell_size=None,
        fftsd=None,
        fftbeam=None,
        tapper=None,
        lambda_sd=0.0,
        lambda_pos=0.0,
        fftkernel=None,
        beam_workers=4,
        verbose=False,
        **_,
    ):
        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        loss_scalar = 0.0

        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            model_vis = forward_beam(
                x2d=x[c],
                primary_beam=primary_beam_list[b],
                grid=grid_list[b],
                uu=uu,
                vv=vv,
                ww=ww,
                cell_size=cell_size,
                device=device,
            )

            I_use = I.conj() if self.conj_data else I
            vis_real = torch.from_numpy(I_use.real).to(device)
            vis_imag = torch.from_numpy(I_use.imag).to(device)
            sig = torch.from_numpy(sI).to(device)

            residual_real = (model_vis.real - vis_real) / sig
            residual_imag = (model_vis.imag - vis_imag) / sig
            J = torch.sum(residual_real**2 + residual_imag**2)
            L = 0.5 * J
            L.backward(retain_graph=True)
            loss_scalar += L.item()

            if verbose:
                print_gpu_memory(device)

        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t = torch.from_numpy(fftsd).to(device)
            fftbeam_t = torch.from_numpy(fftbeam).to(device)
            tapper_t = torch.from_numpy(tapper).to(device)
            xfft2 = tfft2(x * tapper_t)
            model_sd = (cell_size**2) * xfft2 * fftbeam_t
            Lsd = 0.5 * (torch.nansum((model_sd.real - fftsd_t.real) ** 2) +
                         torch.nansum((model_sd.imag - fftsd_t.imag) ** 2)) * lambda_sd
            Lsd.backward(retain_graph=True)
            loss_scalar += Lsd.item()

        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t = torch.from_numpy(tapper).to(device)
            fftkernel_t = torch.from_numpy(fftkernel).to(device)
            xfft2 = tfft2(x * tapper_t)
            conv = (cell_size**2) * xfft2 * fftkernel_t
            Lr = 0.5 * torch.nansum(torch.abs(conv) ** 2) * self.lambda_r
            Lr.backward()
            loss_scalar += Lr.item()

        return torch.tensor(loss_scalar)
