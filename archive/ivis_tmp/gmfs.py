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

import os
import numpy as np
import torch
from torch.fft import fft2 as tfft2

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.operators import (
    forward_beam,
    resolve_pb_grid_lists,
)
from ivis.models.utils.gpu import print_gpu_memory


class GMFS(BaseModel):
    """
    Discrete Gaussian-mixture spectral model.

    The optimized parameters are coefficient maps over a fixed dictionary
    of Gaussian spectral atoms defined on a grid of (mu, sigma).

    Example
    -------
    mu_grid  = [5, 15, 25, 35]
    sig_grid = [3, 6, 10]

    Then x.shape[0] must be 12 = len(mu_grid) * len(sig_grid)

    x[k, y, x] is the coefficient map for one fixed Gaussian atom.
    """

    def __init__(
        self,
        mu_grid=(5, 15, 25, 35),
        sig_grid=(3, 6, 10),
        lambda_r=1,
        use_2pi=True,
        conj_data=True,
    ):
        self.mu_grid = np.asarray(mu_grid, dtype=np.float32)
        self.sig_grid = np.asarray(sig_grid, dtype=np.float32)

        if self.mu_grid.ndim != 1 or self.sig_grid.ndim != 1:
            raise ValueError("mu_grid and sig_grid must be 1D sequences.")
        if np.any(self.sig_grid <= 0):
            raise ValueError("All sigma values in sig_grid must be > 0.")

        self.nmu = len(self.mu_grid)
        self.nsig = len(self.sig_grid)
        self.nbasis = self.nmu * self.nsig

        self.lambda_r = lambda_r
        self.use_2pi = use_2pi
        self.conj_data = conj_data
        self._interp_cache = {}

    # ------------------------------------------------------------------
    # Basis helpers
    # ------------------------------------------------------------------
    def _make_basis(self, nchan, device, dtype):
        """
        Returns
        -------
        basis : torch.Tensor
            Shape (nbasis, nchan)
        """
        c = torch.arange(nchan, device=device, dtype=dtype)

        mu_t = torch.as_tensor(self.mu_grid, device=device, dtype=dtype)
        sig_t = torch.as_tensor(self.sig_grid, device=device, dtype=dtype)

        basis_list = []
        for mu in mu_t:
            for sig in sig_t:
                phi = torch.exp(-0.5 * ((c - mu) / sig) ** 2)
                basis_list.append(phi)

        basis = torch.stack(basis_list, dim=0)  # (nbasis, nchan)
        return basis

    def _channel_image(self, c, coeffs, basis):
        """
        coeffs shape: (nbasis, ny, nx)
        basis shape : (nbasis, nchan)
        returns     : (ny, nx)
        """
        return torch.sum(coeffs * basis[:, c][:, None, None], dim=0)

    # ------------------------------------------------------------------
    # SciPy-style loss wrapper
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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

        if x.shape[0] != self.nbasis:
            raise ValueError(
                f"Expected x.shape[0] == nbasis={self.nbasis}, got {x.shape[0]}"
            )

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

        basis = self._make_basis(nchan=nchan, device=dev, dtype=x.dtype)

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

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
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

        if x.shape[0] != self.nbasis:
            raise ValueError(
                f"Expected x.shape[0] == nbasis={self.nbasis}, got {x.shape[0]}"
            )

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        loss = torch.tensor(0.0, device=dev, dtype=dtype)

        nchan = vis_data.frequency.shape[0]
        nbeam = vis_data.uu.shape[0]

        tapper_t = torch.from_numpy(tapper).to(dev, dtype=dtype) if tapper is not None else None
        fftbeam_t = torch.from_numpy(fftbeam).to(dev) if fftbeam is not None else None
        fftkernel_t = torch.from_numpy(fftkernel).to(dev) if fftkernel is not None else None

        basis = self._make_basis(nchan=nchan, device=dev, dtype=dtype)

        for c in range(nchan):
            x2d = self._channel_image(c, x, basis)

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

        if lambda_pos > 0.0:
            loss = loss + lambda_pos * torch.nansum(torch.clamp(-x, min=0.0) ** 2)

        loss.backward()
        return loss

class GMFS2(BaseModel):
    def __init__(self, ncomp, lambda_r=1, use_2pi=True, conj_data=True):
        self.ncomp = int(ncomp)
        self.lambda_r = lambda_r
        self.use_2pi = use_2pi
        self.conj_data = conj_data
        self._interp_cache = {}


    def _unpack_params(self, x):
        """
        x shape: (3*ncomp, ny, nx)
        ordered as [a1, mu1, sig1, a2, mu2, sig2, ...]
        """
        p = x.view(self.ncomp, 3, *x.shape[-2:])
        A = p[:, 0]
        MU = p[:, 1]
        SIG = p[:, 2]
        return A, MU, SIG


    def _channel_image(self, c, A, MU, SIG):
        """
        Build one 2D image from Gaussian parameter maps for channel index c.
        MU and SIG are in channel-index units.
        """
        c_t = torch.as_tensor(c, device=A.device, dtype=A.dtype)
        SIG_safe = torch.clamp(SIG, min=1e-3)
        return torch.sum(A * torch.exp(-0.5 * ((c_t - MU) / SIG_safe) ** 2), dim=0)
    

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
        out = np.zeros((nchan, nbeam, nvis), dtype=np.complex64)
        has_flags = hasattr(vis_data, "flag_I")

        A, MU, SIG = self._unpack_params(x)

        for c in range(nchan):
            x2d = self._channel_image(c, A, MU, SIG)
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

        loss = torch.tensor(0.0, device=dev, dtype=dtype)

        A, MU, SIG = self._unpack_params(x)

        nchan = vis_data.frequency.shape[0]
        nbeam = vis_data.uu.shape[0]

        tapper_t = torch.from_numpy(tapper).to(dev, dtype=dtype)
        fftbeam_t = torch.from_numpy(fftbeam).to(dev)
        fftkernel_t = torch.from_numpy(fftkernel).to(dev)
        
        for c in range(nchan):
            x2d = self._channel_image(c, A, MU, SIG)

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
                    device=device,
                )

                I_use = I.conj() if self.conj_data else I
                vis_real = torch.from_numpy(I_use.real).to(dev)
                vis_imag = torch.from_numpy(I_use.imag).to(dev)
                sig = torch.from_numpy(sI).to(dev)
                
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

        loss.backward()

        g = x.grad
        
        logger.info(f"A   grad mean/max: {g[0].abs().mean():.3e} / {g[0].abs().max():.3e}")
        logger.info(f"MU  grad mean/max: {g[1].abs().mean():.3e} / {g[1].abs().max():.3e}")
        logger.info(f"SIG grad mean/max: {g[2].abs().mean():.3e} / {g[2].abs().max():.3e}")
        
        return loss
