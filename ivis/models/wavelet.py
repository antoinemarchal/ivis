import os

import numpy as np
import torch
from scipy.linalg import qr
from torch.fft import fft2 as tfft2

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.operators import forward_beam, resolve_pb_grid_lists
from ivis.models.utils.gpu import print_gpu_memory


def _is_power_of_two(n):
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def _haar_basis_power_of_two(n):
    """Return an orthonormal Haar basis matrix of shape (n, n)."""
    n = int(n)
    if not _is_power_of_two(n):
        raise ValueError("n must be a power of two for the exact Haar basis.")

    basis = [np.ones(n, dtype=np.float32) / np.sqrt(n)]
    block = n
    while block >= 2:
        half = block // 2
        scale_norm = np.sqrt(block)
        for start in range(0, n, block):
            vec = np.zeros(n, dtype=np.float32)
            vec[start : start + half] = 1.0 / scale_norm
            vec[start + half : start + block] = -1.0 / scale_norm
            basis.append(vec)
        block //= 2
    return np.stack(basis, axis=0)


def _haar_basis_any_length(n):
    """
    Build a full orthonormal multiscale basis for arbitrary channel counts.

    For power-of-two lengths this is the exact Haar basis. Otherwise, construct
    the Haar basis on the next power-of-two grid, crop it to n channels, then
    orthonormalize the cropped atoms so the model still has exactly n basis
    functions for n channels.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    if _is_power_of_two(n):
        return _haar_basis_power_of_two(n)

    padded_n = 1 << int(np.ceil(np.log2(n)))
    padded_basis = _haar_basis_power_of_two(padded_n)
    cropped = padded_basis[:, :n]
    q, _ = qr(cropped.T, mode="economic")
    return q.T.astype(np.float32, copy=False)


class WaveletSpectralModel(BaseModel):
    """
    Full spectral wavelet model with one coefficient map per channel.

    The spectral axis is represented in a fixed orthonormal 1D Haar basis.
    For ``nchan`` channels the optimized tensor must have shape
    ``(nchan, ny, nx)``.
    """

    def __init__(
        self,
        lambda_r=1,
        lambda_wavelet=0.0,
        conj_data=True,
    ):
        self.lambda_r = lambda_r
        self.lambda_wavelet = float(lambda_wavelet)
        self.conj_data = conj_data
        self._basis_cache = {}

    def _get_basis(self, nchan, device, dtype):
        cache_key = (int(nchan), str(torch.device(device)), str(dtype))
        cached = self._basis_cache.get(cache_key)
        if cached is not None:
            return cached

        basis_np = _haar_basis_any_length(nchan)
        basis = torch.from_numpy(basis_np).to(device=device, dtype=dtype)
        self._basis_cache[cache_key] = basis
        return basis

    def _channel_image(self, c, coeffs, basis):
        return torch.sum(coeffs * basis[:, c][:, None, None], dim=0)

    def reconstruct_cube(self, x, device=None):
        """
        Reconstruct the spectral cube from wavelet coefficient maps.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Coefficient maps with shape (nchan, ny, nx).
        device : str or torch.device, optional
            Target device. If omitted, keep the current tensor device.

        Returns
        -------
        torch.Tensor
            Reconstructed cube with shape (nchan, ny, nx).
        """
        if torch.is_tensor(x):
            coeffs = x
            if device is not None:
                coeffs = coeffs.to(device)
        else:
            target_device = torch.device(device) if device is not None else torch.device("cpu")
            coeffs = torch.as_tensor(x, device=target_device).float()

        basis = self._get_basis(
            nchan=coeffs.shape[0],
            device=coeffs.device,
            dtype=coeffs.dtype,
        )
        return torch.einsum("kc,khw->chw", basis, coeffs)

    def loss(self, x, shape, device, vis_data, **kwargs):
        dev = torch.device(device)
        u = x.reshape(shape)
        u = torch.from_numpy(u).to(dev).requires_grad_(True)

        L = self.objective(
            x=u,
            vis_data=vis_data,
            device=device,
            **kwargs,
        )

        grad = u.grad.detach().cpu().numpy().astype(x.dtype)

        if dev.type == "cuda":
            allocated = torch.cuda.memory_allocated(dev) / 1024**2
            reserved = torch.cuda.memory_reserved(dev) / 1024**2
            total = torch.cuda.get_device_properties(dev).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                f"GPU: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total"
            )
        else:
            logger.info(
                f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}"
            )

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
        if x.shape[0] != nchan:
            raise ValueError(f"Expected x.shape[0] == nchan={nchan}, got {x.shape[0]}")

        out = np.zeros((nchan, nbeam, nvis), dtype=np.complex64)
        has_flags = hasattr(vis_data, "flag_I")
        basis = self._get_basis(nchan=nchan, device=dev, dtype=x.dtype)

        for c in range(nchan):
            x2d = self._channel_image(c, x, basis)

            for b in range(nbeam):
                Icb, sI, uu, vv, ww = vis_data.slice_chan_beam_I(c, b)
                if Icb.size == 0:
                    continue

                model_vis = forward_beam(
                    x2d=x2d,
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
        dev = torch.device(device)
        dtype = x.dtype

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

        nchan = vis_data.frequency.shape[0]
        nbeam = vis_data.uu.shape[0]
        if x.shape[0] != nchan:
            raise ValueError(f"Expected x.shape[0] == nchan={nchan}, got {x.shape[0]}")

        loss = torch.tensor(0.0, device=dev, dtype=dtype)
        tapper_t = torch.from_numpy(tapper).to(dev, dtype=dtype) if tapper is not None else None
        fftbeam_t = torch.from_numpy(fftbeam).to(dev) if fftbeam is not None else None
        fftkernel_t = torch.from_numpy(fftkernel).to(dev) if fftkernel is not None else None
        basis = self._get_basis(nchan=nchan, device=dev, dtype=dtype)

        for c in range(nchan):
            x2d = self._channel_image(c, x, basis)

            if lambda_pos > 0.0:
                loss = loss + lambda_pos * torch.sum(torch.clamp(-x2d, min=0.0) ** 2)

            xfft2 = None
            need_fft = (
                (lambda_sd > 0.0 and fftsd is not None and fftbeam is not None)
                or (self.lambda_r > 0.0 and fftkernel is not None)
            )
            if need_fft:
                xfft2 = tfft2(x2d * tapper_t)

            for b in range(nbeam):
                I, sI, uu, vv, ww = vis_data.slice_chan_beam_I(c, b)
                if I.size == 0:
                    continue

                model_vis = forward_beam(
                    x2d=x2d,
                    primary_beam=primary_beam_list[b],
                    grid=grid_list[b],
                    uu=uu,
                    vv=vv,
                    ww=ww,
                    cell_size=cell_size,
                    device=dev,
                )

                I_use = I.conj() if self.conj_data else I
                vis_real = torch.from_numpy(I_use.real).to(dev)
                vis_imag = torch.from_numpy(I_use.imag).to(dev)
                sig = torch.from_numpy(sI).to(dev)
                sig = torch.clamp(sig, min=1e-6)

                residual_real = (model_vis.real - vis_real) / sig
                residual_imag = (model_vis.imag - vis_imag) / sig
                J = torch.sum(residual_real**2 + residual_imag**2)
                loss = loss + 0.5 * J

                if verbose:
                    print_gpu_memory(device)

            if lambda_sd > 0.0 and fftsd is not None:
                fftsd_c = fftsd if fftsd.ndim == 2 else fftsd[c]
                fftsd_t = torch.from_numpy(fftsd_c).to(dev)

                model_sd = (cell_size**2) * xfft2 * fftbeam_t
                Lsd = 0.5 * (
                    torch.nansum((model_sd.real - fftsd_t.real) ** 2)
                    + torch.nansum((model_sd.imag - fftsd_t.imag) ** 2)
                ) * lambda_sd
                loss = loss + Lsd

            if self.lambda_r > 0.0 and fftkernel is not None:
                conv = (cell_size**2) * xfft2 * fftkernel_t
                Lr = 0.5 * torch.nansum(torch.abs(conv) ** 2) * self.lambda_r
                loss = loss + Lr

        if self.lambda_wavelet > 0.0:
            loss = loss + 0.5 * self.lambda_wavelet * torch.sum(x**2)

        loss.backward()
        return loss
