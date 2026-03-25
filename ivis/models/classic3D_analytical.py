import os

import numpy as np
import torch
from torch.fft import fft2 as tfft2, ifft2 as tifft2

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.operators import (
    backward_beam,
    forward_beam,
    resolve_pb_grid_lists,
)


class Classic3DAnalytical(BaseModel):
    """
    Fully analytical quadratic model for:
        - visibility data term
        - optional single-dish Fourier term
        - optional quadratic Fourier regularization

    No autograd is used for the main loss/gradient.
    """

    def __init__(self, lambda_r=1.0, Nw=None, use_2pi=True, conj_data=True):
        self.lambda_r = float(lambda_r)
        self.Nw = None
        self.use_2pi = use_2pi
        self.conj_data = conj_data
        self._interp_cache = {}

    def loss(self, x, shape, device, vis_data, **kwargs):
        dev = torch.device(device)
        x_t = torch.as_tensor(x.reshape(shape), device=dev, dtype=torch.float32)

        L, grad = self.objective_and_grad(
            x=x_t,
            vis_data=vis_data,
            device=dev,
            **kwargs,
        )

        grad_np = grad.detach().cpu().numpy().astype(np.asarray(x).dtype)
        loss_value = float(L.detach().cpu())

        if dev.type == "cuda":
            allocated = torch.cuda.memory_allocated(dev) / 1024**2
            reserved = torch.cuda.memory_reserved(dev) / 1024**2
            total = torch.cuda.get_device_properties(dev).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Total cost: "
                f"{np.format_float_scientific(loss_value, precision=5)} | "
                f"GPU: {allocated:.2f} MB allocated, "
                f"{reserved:.2f} MB reserved, {total:.2f} MB total"
            )
        else:
            logger.info(
                f"[PID {os.getpid()}] Total cost: "
                f"{np.format_float_scientific(loss_value, precision=5)}"
            )

        return loss_value, grad_np.ravel()

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
        x = torch.as_tensor(x, device=dev, dtype=torch.float32)

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
                    f"unflagged slot count {int(np.count_nonzero(good))} "
                    f"for channel {c}, beam {b}."
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

    def objective(self, x, vis_data, device, **kwargs):
        L, grad = self.objective_and_grad(
            x=x,
            vis_data=vis_data,
            device=device,
            **kwargs,
        )
        if isinstance(x, torch.Tensor) and x.requires_grad:
            x.grad = grad
        return L

    def quadratic_rhs(
        self,
        x_shape,
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
        fftkernel=None,
        **_,
    ):
        dev = torch.device(device)
        if len(x_shape) != 3:
            raise ValueError(f"x_shape must be (nchan, H, W), got {x_shape}")

        nchan, height, width = x_shape
        rhs = torch.zeros((nchan, height, width), dtype=torch.float32, device=dev)

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            data_np = I.conj() if self.conj_data else I
            data_t = torch.as_tensor(data_np, device=dev, dtype=torch.complex64)
            sig_inv2_t = 1.0 / torch.as_tensor(sI, device=dev, dtype=torch.float32).square()
            rhs_cb = backward_beam(
                y=data_t * sig_inv2_t,
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
            rhs[c] += rhs_cb.real.to(torch.float32)

        if lambda_sd > 0.0:
            if fftsd is None or fftbeam is None or tapper is None:
                raise ValueError("lambda_sd > 0 requires fftsd, fftbeam, and tapper.")

            tapper_t = torch.as_tensor(tapper, device=dev, dtype=torch.float32)
            fftsd_t = torch.as_tensor(fftsd, device=dev, dtype=torch.complex64)
            fftbeam_t = torch.as_tensor(fftbeam, device=dev, dtype=torch.complex64)
            npix = height * width
            rhs_sd = (
                lambda_sd
                * tapper_t
                * (npix * tifft2((cell_size**2) * torch.conj(fftbeam_t) * fftsd_t))
            )
            rhs += rhs_sd.real.to(torch.float32)

        return rhs

    def apply_normal_operator(
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
        fftkernel=None,
        **_,
    ):
        dev = torch.device(device)
        x = torch.as_tensor(x, device=dev, dtype=torch.float32)
        if x.ndim != 3:
            raise ValueError(f"x must have shape (nchan, H, W), got {tuple(x.shape)}")

        nchan, height, width = x.shape
        out = torch.zeros_like(x, dtype=torch.float32)

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            sig_inv2_t = 1.0 / torch.as_tensor(sI, device=dev, dtype=torch.float32).square()
            hx = forward_beam(
                x2d=x[c],
                primary_beam=primary_beam_list[b],
                grid=grid_list[b],
                uu=uu,
                vv=vv,
                ww=ww,
                cell_size=cell_size,
                device=dev,
            )
            ahwhx = backward_beam(
                y=hx * sig_inv2_t,
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
            out[c] += ahwhx.real.to(torch.float32)

        if lambda_sd > 0.0:
            if fftbeam is None or tapper is None:
                raise ValueError("lambda_sd > 0 requires fftbeam and tapper.")

            tapper_t = torch.as_tensor(tapper, device=dev, dtype=torch.float32)
            fftbeam_t = torch.as_tensor(fftbeam, device=dev, dtype=torch.complex64)
            model_sd = (cell_size**2) * tfft2(x * tapper_t) * fftbeam_t
            valid = torch.isfinite(model_sd.real) & torch.isfinite(model_sd.imag)
            model_sd = torch.where(valid, model_sd, torch.zeros_like(model_sd))
            npix = height * width
            ahasx = (
                lambda_sd
                * tapper_t
                * (npix * tifft2((cell_size**2) * torch.conj(fftbeam_t) * model_sd))
            )
            out += ahasx.real.to(torch.float32)

        if self.lambda_r > 0.0 and fftkernel is not None:
            if tapper is None:
                raise ValueError("lambda_r > 0 with fftkernel requires tapper.")

            tapper_t = torch.as_tensor(tapper, device=dev, dtype=torch.float32)
            fftkernel_t = torch.as_tensor(fftkernel, device=dev, dtype=torch.complex64)
            conv = (cell_size**2) * tfft2(x * tapper_t) * fftkernel_t
            valid = torch.isfinite(conv.real) & torch.isfinite(conv.imag)
            conv = torch.where(valid, conv, torch.zeros_like(conv))
            npix = height * width
            ahakx = (
                self.lambda_r
                * tapper_t
                * (npix * tifft2((cell_size**2) * torch.conj(fftkernel_t) * conv))
            )
            out += ahakx.real.to(torch.float32)

        return out

    def objective_and_grad(
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
        dev = torch.device(device)
        x = torch.as_tensor(x, device=dev, dtype=torch.float32)

        if x.ndim != 3:
            raise ValueError(f"x must have shape (nchan, H, W), got {tuple(x.shape)}")

        nchan, height, width = x.shape
        grad = torch.zeros_like(x, dtype=torch.float32)
        loss_scalar = torch.zeros((), device=dev, dtype=torch.float32)

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            model_vis = forward_beam(
                x2d=x[c],
                primary_beam=primary_beam_list[b],
                grid=grid_list[b],
                uu=uu,
                vv=vv,
                ww=ww,
                cell_size=cell_size,
                device=dev,
            )

            data_np = I.conj() if self.conj_data else I
            data_t = torch.as_tensor(data_np, device=dev, dtype=torch.complex64)
            sig_t = torch.as_tensor(sI, device=dev, dtype=torch.float32)

            residual = model_vis - data_t
            weighted_sq = (residual.real**2 + residual.imag**2) / (sig_t**2)
            loss_scalar = loss_scalar + 0.5 * torch.sum(weighted_sq)

            adjoint_vis = residual / (sig_t**2)

            grad_cb = backward_beam(
                y=adjoint_vis,
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
            grad[c] += grad_cb.real.to(torch.float32)

        if lambda_sd > 0.0:
            if fftsd is None or fftbeam is None or tapper is None:
                raise ValueError("lambda_sd > 0 requires fftsd, fftbeam, and tapper.")

            tapper_t = torch.as_tensor(tapper, device=dev, dtype=torch.float32)
            fftsd_t = torch.as_tensor(fftsd, device=dev, dtype=torch.complex64)
            fftbeam_t = torch.as_tensor(fftbeam, device=dev, dtype=torch.complex64)

            model_sd = (cell_size**2) * tfft2(x * tapper_t) * fftbeam_t
            resid_sd = model_sd - fftsd_t

            valid = torch.isfinite(resid_sd.real) & torch.isfinite(resid_sd.imag)
            resid_sd_masked = torch.where(valid, resid_sd, torch.zeros_like(resid_sd))

            loss_scalar = loss_scalar + 0.5 * lambda_sd * torch.sum(torch.abs(resid_sd_masked) ** 2)

            npix = height * width
            grad_sd_complex = (
                lambda_sd
                * tapper_t
                * (npix * tifft2((cell_size**2) * torch.conj(fftbeam_t) * resid_sd_masked))
            )
            grad += grad_sd_complex.real.to(torch.float32)

        if self.lambda_r > 0.0 and fftkernel is not None:
            if tapper is None:
                raise ValueError("lambda_r > 0 with fftkernel requires tapper.")

            tapper_t = torch.as_tensor(tapper, device=dev, dtype=torch.float32)
            fftkernel_t = torch.as_tensor(fftkernel, device=dev, dtype=torch.complex64)

            conv = (cell_size**2) * tfft2(x * tapper_t) * fftkernel_t
            valid = torch.isfinite(conv.real) & torch.isfinite(conv.imag)
            conv_masked = torch.where(valid, conv, torch.zeros_like(conv))

            loss_scalar = loss_scalar + 0.5 * self.lambda_r * torch.sum(torch.abs(conv_masked) ** 2)

            npix = height * width
            grad_reg_complex = (
                self.lambda_r
                * tapper_t
                * (npix * tifft2((cell_size**2) * torch.conj(fftkernel_t) * conv_masked))
            )
            grad += grad_reg_complex.real.to(torch.float32)

        if lambda_pos > 0.0:
            raise NotImplementedError("lambda_pos is not implemented in Classic3DAnalytical.")

        return loss_scalar, grad
