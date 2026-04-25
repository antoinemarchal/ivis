import os

import numpy as np
import torch
from torch.fft import fft2 as tfft2
from astropy.constants import c as c_light

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.operators import forward_beam, resolve_pb_grid_lists
from ivis.models.utils.gpu import print_gpu_memory


# LRSB: Low-Rank Spectral Basis
class LRSB(BaseModel):
    """
    Low-rank spectral basis model driven by a user-supplied basis matrix.

    The basis must have shape (nbasis, nchan).
    """

    def __init__(
        self,
        basis,
        lambda_r=1.0,
        lambda_pos=0.0,
        conj_data=True,
        assume_channel_invariant_operator=False,
        reference_channel=0,
    ):
        basis_arr = np.asarray(basis, dtype=np.float32)
        if basis_arr.ndim != 2:
            raise ValueError("basis must have shape (nbasis, nchan).")

        self.lambda_r = float(lambda_r)
        self.lambda_pos = float(lambda_pos)
        self.conj_data = conj_data
        self.assume_channel_invariant_operator = bool(assume_channel_invariant_operator)
        self.reference_channel = int(reference_channel)
        self._basis_np = basis_arr.astype(np.float32, copy=False)
        self._basis_cache = {}

    @property
    def nbasis(self):
        return int(self._basis_np.shape[0])

    @property
    def nchan(self):
        return int(self._basis_np.shape[1])

    def _get_basis(self, device, dtype):
        cache_key = (str(torch.device(device)), str(dtype))
        cached = self._basis_cache.get(cache_key)
        if cached is not None:
            return cached

        basis = torch.from_numpy(self._basis_np).to(device=device, dtype=dtype)
        self._basis_cache[cache_key] = basis
        return basis

    def reconstruct_cube(self, x, device=None, return_numpy=False):
        if torch.is_tensor(x):
            coeffs = x
            if device is not None:
                coeffs = coeffs.to(device)
        else:
            target_device = torch.device(device) if device is not None else torch.device("cpu")
            coeffs = torch.as_tensor(x, device=target_device).float()

        if coeffs.shape[0] != self.nbasis:
            raise ValueError(
                f"Expected coeffs.shape[0] == nbasis={self.nbasis}, got {coeffs.shape[0]}"
            )

        basis = self._get_basis(device=coeffs.device, dtype=coeffs.dtype)
        cube = torch.einsum("kc,khw->chw", basis, coeffs)
        if return_numpy:
            return cube.detach().cpu().numpy()
        return cube

    def reconstruct_cube_from_coeffs(self, coeffs, device=None, return_numpy=True):
        return self.reconstruct_cube(coeffs, device=device, return_numpy=return_numpy)

    def _lambda_r_for_basis(self, basis_index):
        return float(self.lambda_r)

    def _weights_for_channel(self, basis, channel_index):
        return basis[:, channel_index]

    def _reference_operator_inputs(self, vis_data, beam_index):
        if self.reference_channel < 0 or self.reference_channel >= self.nchan:
            raise ValueError(
                f"reference_channel={self.reference_channel} is outside [0, {self.nchan - 1}]"
            )
        nv = int(vis_data.nvis[beam_index])
        scale = vis_data.frequency[self.reference_channel] / c_light.value
        uu_ref = vis_data.uu[beam_index, :nv] * scale
        vv_ref = vis_data.vv[beam_index, :nv] * scale
        ww_ref = vis_data.ww[beam_index, :nv] * scale
        return uu_ref, vv_ref, ww_ref, nv

    def _build_invariant_beam_cache(
        self,
        x,
        vis_data,
        beam_index,
        device,
        primary_beam_list,
        grid_list,
        cell_size,
    ):
        dev = torch.device(device)
        uu_ref, vv_ref, ww_ref, nref = self._reference_operator_inputs(vis_data, beam_index)
        hk_list = []
        for k in range(self.nbasis):
            hk_full = forward_beam(
                x2d=x[k],
                primary_beam=primary_beam_list[beam_index],
                grid=grid_list[beam_index],
                uu=uu_ref,
                vv=vv_ref,
                ww=ww_ref,
                cell_size=cell_size,
                device=dev,
            )
            if hk_full.numel() != nref:
                raise ValueError(
                    f"Reference forward model size {hk_full.numel()} does not match beam visibility count {nref} for beam {beam_index}."
                )
            hk_list.append(hk_full)
        return torch.stack(hk_list, dim=0)

    def _prepare_channel_beam_blocks(self, vis_data, device):
        dev = torch.device(device)
        nchan, nbeam, _ = vis_data.data_I.shape
        blocks = [[None for _ in range(nbeam)] for _ in range(nchan)]

        for c in range(nchan):
            for b in range(nbeam):
                nv = int(vis_data.nvis[b])
                good_np = ~np.asarray(vis_data.flag_I[c, b, :nv], dtype=bool)
                if not np.any(good_np):
                    continue

                I, sI, uu, vv, ww = vis_data.slice_chan_beam_I(c, b)
                blocks[c][b] = {
                    "I": I,
                    "sI": sI,
                    "uu": uu,
                    "vv": vv,
                    "ww": ww,
                    "good_np": good_np,
                    "good_t": torch.as_tensor(good_np, device=dev, dtype=torch.bool),
                }

        return blocks

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
        if nchan != self.nchan:
            raise ValueError(f"Expected nchan={self.nchan} from basis, got {nchan}.")
        if x.shape[0] != self.nbasis:
            raise ValueError(f"Expected x.shape[0] == nbasis={self.nbasis}, got {x.shape[0]}")

        basis = self._get_basis(device=dev, dtype=x.dtype)
        out = np.zeros((nchan, nbeam, nvis), dtype=np.complex64)
        has_flags = hasattr(vis_data, "flag_I")
        blocks = self._prepare_channel_beam_blocks(vis_data=vis_data, device=dev)

        if self.assume_channel_invariant_operator:
            for b in range(nbeam):
                hk_stack = self._build_invariant_beam_cache(
                    x=x,
                    vis_data=vis_data,
                    beam_index=b,
                    device=dev,
                    primary_beam_list=primary_beam_list,
                    grid_list=grid_list,
                    cell_size=cell_size,
                )
                for c in range(nchan):
                    block = blocks[c][b]
                    if block is None:
                        continue
                    weights = basis[:, c]
                    model_vis = torch.einsum(
                        "k,kn->n",
                        weights.to(hk_stack.dtype),
                        hk_stack[:, block["good_t"]],
                    )
                    good = block["good_np"]
                    model_vis = model_vis.detach().cpu().numpy().astype(np.complex64)

                    nv = int(vis_data.nvis[b])
                    if nv > nvis:
                        raise ValueError(f"vis_data.nvis[{b}]={nv} exceeds cube width {nvis}.")

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
                del hk_stack
        else:
            for c in range(nchan):
                weights = basis[:, c]
                x2d = torch.sum(x * weights[:, None, None], dim=0)
                for b in range(nbeam):
                    block = blocks[c][b]
                    if block is None:
                        continue
                    model_vis = forward_beam(
                        x2d=x2d,
                        primary_beam=primary_beam_list[b],
                        grid=grid_list[b],
                        uu=block["uu"],
                        vv=block["vv"],
                        ww=block["ww"],
                        cell_size=cell_size,
                        device=dev,
                    )
                    good = block["good_np"]

                    model_vis = model_vis.detach().cpu().numpy().astype(np.complex64)

                    nv = int(vis_data.nvis[b])
                    if nv > nvis:
                        raise ValueError(f"vis_data.nvis[{b}]={nv} exceeds cube width {nvis}.")

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
        lambda_pos=None,
        fftkernel=None,
        beam_workers=4,
        verbose=False,
        **_,
    ):
        dev = torch.device(device)
        if x.ndim != 3:
            raise ValueError(f"x must have shape (nbasis, H, W), got {tuple(x.shape)}")
        if x.shape[0] != self.nbasis:
            raise ValueError(f"Expected x.shape[0] == nbasis={self.nbasis}, got {x.shape[0]}")
        dtype = x.dtype

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        nchan = vis_data.frequency.shape[0]
        nbeam = vis_data.uu.shape[0]
        if nchan != self.nchan:
            raise ValueError(f"Expected nchan={self.nchan} from basis, got {nchan}.")

        loss = torch.tensor(0.0, device=dev, dtype=dtype)
        lambda_pos = self.lambda_pos if lambda_pos is None else float(lambda_pos)
        tapper_t = torch.from_numpy(tapper).to(dev, dtype=dtype) if tapper is not None else None
        fftbeam_t = torch.from_numpy(fftbeam).to(dev) if fftbeam is not None else None
        fftkernel_t = torch.from_numpy(fftkernel).to(dev) if fftkernel is not None else None
        basis = self._get_basis(device=dev, dtype=dtype)
        blocks = self._prepare_channel_beam_blocks(vis_data=vis_data, device=dev)

        if self.assume_channel_invariant_operator:
            need_sd_fft = lambda_sd > 0.0 and fftsd is not None and fftbeam is not None
            channel_cache = []
            for c in range(nchan):
                weights = self._weights_for_channel(basis, c)
                need_x2d = (lambda_pos > 0.0) or need_sd_fft
                x2d = torch.sum(x * weights[:, None, None], dim=0) if need_x2d else None
                xfft2 = tfft2(x2d * tapper_t) if need_sd_fft else None
                channel_cache.append((weights, x2d, xfft2))

                if lambda_pos > 0.0:
                    loss = loss + lambda_pos * torch.sum(torch.clamp(-x2d, min=0.0) ** 2)

                if need_sd_fft:
                    fftsd_c = fftsd if fftsd.ndim == 2 else fftsd[c]
                    fftsd_t = torch.from_numpy(fftsd_c).to(dev)
                    model_sd = (cell_size**2) * xfft2 * fftbeam_t
                    Lsd = 0.5 * (
                        torch.nansum((model_sd.real - fftsd_t.real) ** 2)
                        + torch.nansum((model_sd.imag - fftsd_t.imag) ** 2)
                    ) * lambda_sd
                    loss = loss + Lsd

            for b in range(nbeam):
                hk_stack = self._build_invariant_beam_cache(
                    x=x,
                    vis_data=vis_data,
                    beam_index=b,
                    device=dev,
                    primary_beam_list=primary_beam_list,
                    grid_list=grid_list,
                    cell_size=cell_size,
                )
                for c in range(nchan):
                    block = blocks[c][b]
                    if block is None:
                        continue
                    weights, _, _ = channel_cache[c]
                    model_vis = torch.einsum(
                        "k,kn->n",
                        weights.to(hk_stack.dtype),
                        hk_stack[:, block["good_t"]],
                    )

                    I_use = block["I"].conj() if self.conj_data else block["I"]
                    vis_real = torch.from_numpy(I_use.real).to(dev)
                    vis_imag = torch.from_numpy(I_use.imag).to(dev)
                    sig = torch.from_numpy(block["sI"]).to(dev)
                    sig = torch.clamp(sig, min=1e-6)

                    residual_real = (model_vis.real - vis_real) / sig
                    residual_imag = (model_vis.imag - vis_imag) / sig
                    J = torch.sum(residual_real**2 + residual_imag**2)
                    loss = loss + 0.5 * J

                    if verbose:
                        print_gpu_memory(device)
                del hk_stack
        else:
            for c in range(nchan):
                weights = self._weights_for_channel(basis, c)
                need_sd_fft = lambda_sd > 0.0 and fftsd is not None and fftbeam is not None
                x2d = torch.sum(x * weights[:, None, None], dim=0)

                if lambda_pos > 0.0:
                    loss = loss + lambda_pos * torch.sum(torch.clamp(-x2d, min=0.0) ** 2)

                xfft2 = None
                if need_sd_fft:
                    xfft2 = tfft2(x2d * tapper_t)

                for b in range(nbeam):
                    block = blocks[c][b]
                    if block is None:
                        continue
                    model_vis = forward_beam(
                        x2d=x2d,
                        primary_beam=primary_beam_list[b],
                        grid=grid_list[b],
                        uu=block["uu"],
                        vv=block["vv"],
                        ww=block["ww"],
                        cell_size=cell_size,
                        device=dev,
                    )

                    I_use = block["I"].conj() if self.conj_data else block["I"]
                    vis_real = torch.from_numpy(I_use.real).to(dev)
                    vis_imag = torch.from_numpy(I_use.imag).to(dev)
                    sig = torch.from_numpy(block["sI"]).to(dev)
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
            for k in range(self.nbasis):
                lambda_r_k = self._lambda_r_for_basis(k)
                if lambda_r_k <= 0.0:
                    continue
                coeff_fft2 = tfft2(x[k] * tapper_t)
                conv = (cell_size**2) * coeff_fft2 * fftkernel_t
                Lr = 0.5 * torch.nansum(torch.abs(conv) ** 2) * lambda_r_k
                loss = loss + Lr

        loss.backward()
        return loss


class LRSB_C(LRSB):
    """
    LRSB variant with explicit continuum basis functions.

    This augments the learned line basis with fixed smooth spectral modes.
    By default, it adds a single flat continuum mode psi_0(nu) = 1.
    """

    def __init__(
        self,
        basis,
        continuum_basis=None,
        continuum_order=0,
        frequency=None,
        reference_frequency=None,
        continuum_only_channels=None,
        lambda_r_line_factor=1.0,
        lambda_r_cont_factor=1.0,
        **kwargs,
    ):
        line_basis = np.asarray(basis, dtype=np.float32)
        if line_basis.ndim != 2:
            raise ValueError("basis must have shape (nbasis, nchan).")

        continuum_arr = self._prepare_continuum_basis(
            nchan=line_basis.shape[1],
            continuum_basis=continuum_basis,
            continuum_order=continuum_order,
            frequency=frequency,
            reference_frequency=reference_frequency,
            line_basis=line_basis,
        )

        self._line_basis_np = line_basis.astype(np.float32, copy=False)
        self._continuum_basis_np = continuum_arr.astype(np.float32, copy=False)
        self._continuum_order = int(continuum_arr.shape[0] - 1)
        self._reference_frequency = (
            None if reference_frequency is None else float(reference_frequency)
        )
        self._continuum_only_channels = self._prepare_channel_mask(
            nchan=line_basis.shape[1],
            channels=continuum_only_channels,
        )
        self.lambda_r_line_factor = float(lambda_r_line_factor)
        self.lambda_r_cont_factor = float(lambda_r_cont_factor)
        hybrid_basis = np.concatenate((self._line_basis_np, self._continuum_basis_np), axis=0)
        super().__init__(basis=hybrid_basis, **kwargs)

    @staticmethod
    def _prepare_continuum_basis(
        nchan,
        continuum_basis,
        continuum_order,
        frequency,
        reference_frequency,
        line_basis,
    ):
        if continuum_basis is None:
            order = int(continuum_order)
            if order < 0:
                raise ValueError("continuum_order must be >= 0.")

            if order == 0:
                continuum_arr = np.ones((1, nchan), dtype=np.float32)
            else:
                if frequency is None:
                    raise ValueError("frequency is required when continuum_order > 0.")

                freq = np.asarray(frequency, dtype=np.float32)
                if freq.ndim != 1 or freq.shape[0] != nchan:
                    raise ValueError(f"frequency must have shape ({nchan},), got {freq.shape}.")

                nu_ref = float(np.mean(freq) if reference_frequency is None else reference_frequency)
                if nu_ref == 0.0:
                    raise ValueError("reference_frequency must be non-zero.")

                xnu = (freq - nu_ref) / nu_ref
                continuum_arr = np.stack([xnu**m for m in range(order + 1)], axis=0).astype(
                    np.float32,
                    copy=False,
                )
        else:
            continuum_arr = np.asarray(continuum_basis, dtype=np.float32)
            if continuum_arr.ndim == 1:
                continuum_arr = continuum_arr[None, :]

        if continuum_arr.ndim != 2:
            raise ValueError("continuum_basis must have shape (ncont, nchan).")
        if continuum_arr.shape[1] != nchan:
            raise ValueError(
                f"continuum_basis must have nchan={nchan}, got {continuum_arr.shape[1]}."
            )

        return continuum_arr

    @staticmethod
    def _prepare_channel_mask(nchan, channels):
        mask = np.zeros(nchan, dtype=bool)
        if channels is None:
            return mask

        idx = np.asarray(channels)
        if idx.dtype == bool:
            if idx.shape != (nchan,):
                raise ValueError(
                    f"Boolean continuum_only_channels mask must have shape ({nchan},), got {idx.shape}."
                )
            return idx.astype(bool, copy=True)

        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size == 0:
            return mask
        if np.any(idx < 0) or np.any(idx >= nchan):
            raise ValueError(f"continuum_only_channels must be within [0, {nchan - 1}].")
        mask[idx] = True
        return mask

    @property
    def line_nbasis(self):
        return int(self._line_basis_np.shape[0])

    @property
    def continuum_nbasis(self):
        return int(self._continuum_basis_np.shape[0])

    @property
    def continuum_basis(self):
        return self._continuum_basis_np

    @property
    def continuum_order(self):
        return self._continuum_order

    @property
    def reference_frequency(self):
        return self._reference_frequency

    @property
    def continuum_only_channels(self):
        return self._continuum_only_channels.copy()

    def _lambda_r_for_basis(self, basis_index):
        if basis_index < self.line_nbasis:
            return float(self.lambda_r) * self.lambda_r_line_factor
        return float(self.lambda_r) * self.lambda_r_cont_factor

    def split_coeffs(self, x):
        if torch.is_tensor(x):
            return x[: self.line_nbasis], x[self.line_nbasis :]
        coeffs = np.asarray(x)
        return coeffs[: self.line_nbasis], coeffs[self.line_nbasis :]

    def reconstruct_line_cube(self, x, device=None, return_numpy=False):
        line_coeffs, _ = self.split_coeffs(x)
        line_model = LRSB(basis=self._line_basis_np)
        return line_model.reconstruct_cube(line_coeffs, device=device, return_numpy=return_numpy)

    def reconstruct_continuum_cube(self, x, device=None, return_numpy=False):
        _, continuum_coeffs = self.split_coeffs(x)
        continuum_model = LRSB(basis=self._continuum_basis_np)
        return continuum_model.reconstruct_cube(
            continuum_coeffs, device=device, return_numpy=return_numpy
        )

    def _weights_for_channel(self, basis, channel_index):
        weights = basis[:, channel_index]
        if not self._continuum_only_channels[channel_index]:
            return weights

        line_zeros = torch.zeros(
            self.line_nbasis,
            device=weights.device,
            dtype=weights.dtype,
        )
        return torch.cat((line_zeros, weights[self.line_nbasis :]), dim=0)


class LRSBMemory(LRSB):
    """
    Memory-streaming LRSB variant.

    LRSB stores a smaller coefficient cube than Classic3D, but its objective
    still accumulates one large autograd graph by default. This variant
    backpropagates independent loss blocks as soon as they are computed.
    """

    def _backward_loss(self, loss, loss_value):
        loss.backward()
        return loss_value + loss.detach()

    def _channel_image(self, x, weights):
        return torch.sum(x * weights[:, None, None], dim=0)

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
        lambda_pos=None,
        fftkernel=None,
        beam_workers=4,
        verbose=False,
        **_,
    ):
        dev = torch.device(device)
        if x.ndim != 3:
            raise ValueError(f"x must have shape (nbasis, H, W), got {tuple(x.shape)}")
        if x.shape[0] != self.nbasis:
            raise ValueError(f"Expected x.shape[0] == nbasis={self.nbasis}, got {x.shape[0]}")

        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        dtype = x.dtype
        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        nchan = vis_data.frequency.shape[0]
        nbeam = vis_data.uu.shape[0]
        if nchan != self.nchan:
            raise ValueError(f"Expected nchan={self.nchan} from basis, got {nchan}.")

        loss_value = torch.zeros((), device=dev, dtype=dtype)
        lambda_pos = self.lambda_pos if lambda_pos is None else float(lambda_pos)
        tapper_t = torch.from_numpy(tapper).to(dev, dtype=dtype) if tapper is not None else None
        fftbeam_t = torch.from_numpy(fftbeam).to(dev) if fftbeam is not None else None
        fftkernel_t = torch.from_numpy(fftkernel).to(dev) if fftkernel is not None else None
        basis = self._get_basis(device=dev, dtype=dtype)
        blocks = self._prepare_channel_beam_blocks(vis_data=vis_data, device=dev)
        need_sd_fft = lambda_sd > 0.0 and fftsd is not None and fftbeam is not None

        for c in range(nchan):
            weights = self._weights_for_channel(basis, c)

            if lambda_pos > 0.0:
                x2d = self._channel_image(x, weights)
                Lpos = lambda_pos * torch.sum(torch.clamp(-x2d, min=0.0) ** 2)
                loss_value = self._backward_loss(Lpos, loss_value)
                del x2d, Lpos

            if need_sd_fft:
                fftsd_c = fftsd if fftsd.ndim == 2 else fftsd[c]
                fftsd_t = torch.from_numpy(fftsd_c).to(dev)
                x2d = self._channel_image(x, weights)
                xfft2 = tfft2(x2d * tapper_t)
                model_sd = (cell_size**2) * xfft2 * fftbeam_t
                Lsd = 0.5 * (
                    torch.nansum((model_sd.real - fftsd_t.real) ** 2)
                    + torch.nansum((model_sd.imag - fftsd_t.imag) ** 2)
                ) * lambda_sd
                loss_value = self._backward_loss(Lsd, loss_value)
                del fftsd_t, x2d, xfft2, model_sd, Lsd

        if self.assume_channel_invariant_operator:
            for b in range(nbeam):
                hk_stack = self._build_invariant_beam_cache(
                    x=x,
                    vis_data=vis_data,
                    beam_index=b,
                    device=dev,
                    primary_beam_list=primary_beam_list,
                    grid_list=grid_list,
                    cell_size=cell_size,
                )
                beam_loss = torch.zeros((), device=dev, dtype=dtype)

                for c in range(nchan):
                    block = blocks[c][b]
                    if block is None:
                        continue

                    weights = self._weights_for_channel(basis, c)
                    model_vis = torch.einsum(
                        "k,kn->n",
                        weights.to(hk_stack.dtype),
                        hk_stack[:, block["good_t"]],
                    )
                    I_use = block["I"].conj() if self.conj_data else block["I"]
                    vis_real = torch.from_numpy(I_use.real).to(dev)
                    vis_imag = torch.from_numpy(I_use.imag).to(dev)
                    sig = torch.from_numpy(block["sI"]).to(dev)
                    sig = torch.clamp(sig, min=1e-6)

                    residual_real = (model_vis.real - vis_real) / sig
                    residual_imag = (model_vis.imag - vis_imag) / sig
                    J = torch.sum(residual_real**2 + residual_imag**2)
                    beam_loss = beam_loss + 0.5 * J

                    if verbose:
                        print_gpu_memory(device)

                    del model_vis, vis_real, vis_imag, sig, residual_real, residual_imag, J

                if beam_loss.requires_grad:
                    loss_value = self._backward_loss(beam_loss, loss_value)
                else:
                    loss_value = loss_value + beam_loss.detach()
                del hk_stack, beam_loss
        else:
            for c in range(nchan):
                weights = self._weights_for_channel(basis, c)

                for b in range(nbeam):
                    block = blocks[c][b]
                    if block is None:
                        continue

                    x2d = self._channel_image(x, weights)
                    model_vis = forward_beam(
                        x2d=x2d,
                        primary_beam=primary_beam_list[b],
                        grid=grid_list[b],
                        uu=block["uu"],
                        vv=block["vv"],
                        ww=block["ww"],
                        cell_size=cell_size,
                        device=dev,
                    )

                    I_use = block["I"].conj() if self.conj_data else block["I"]
                    vis_real = torch.from_numpy(I_use.real).to(dev)
                    vis_imag = torch.from_numpy(I_use.imag).to(dev)
                    sig = torch.from_numpy(block["sI"]).to(dev)
                    sig = torch.clamp(sig, min=1e-6)

                    residual_real = (model_vis.real - vis_real) / sig
                    residual_imag = (model_vis.imag - vis_imag) / sig
                    J = torch.sum(residual_real**2 + residual_imag**2)
                    block_loss = 0.5 * J
                    loss_value = self._backward_loss(block_loss, loss_value)

                    if verbose:
                        print_gpu_memory(device)

                    del (
                        x2d,
                        model_vis,
                        vis_real,
                        vis_imag,
                        sig,
                        residual_real,
                        residual_imag,
                        J,
                        block_loss,
                    )

        if self.lambda_r > 0.0 and fftkernel is not None:
            for k in range(self.nbasis):
                lambda_r_k = self._lambda_r_for_basis(k)
                if lambda_r_k <= 0.0:
                    continue
                coeff_fft2 = tfft2(x[k] * tapper_t)
                conv = (cell_size**2) * coeff_fft2 * fftkernel_t
                Lr = 0.5 * torch.nansum(torch.abs(conv) ** 2) * lambda_r_k
                loss_value = self._backward_loss(Lr, loss_value)
                del coeff_fft2, conv, Lr

        return loss_value


class LRSB_CMemory(LRSBMemory, LRSB_C):
    """
    Memory-streaming LRSB_C variant.

    This combines the hybrid line+continuum basis construction from LRSB_C
    with the blockwise backward pass from LRSBMemory.
    """

    def __init__(
        self,
        basis,
        continuum_basis=None,
        continuum_order=0,
        frequency=None,
        reference_frequency=None,
        continuum_only_channels=None,
        lambda_r_line_factor=1.0,
        lambda_r_cont_factor=1.0,
        **kwargs,
    ):
        LRSB_C.__init__(
            self,
            basis=basis,
            continuum_basis=continuum_basis,
            continuum_order=continuum_order,
            frequency=frequency,
            reference_frequency=reference_frequency,
            continuum_only_channels=continuum_only_channels,
            lambda_r_line_factor=lambda_r_line_factor,
            lambda_r_cont_factor=lambda_r_cont_factor,
            **kwargs,
        )
