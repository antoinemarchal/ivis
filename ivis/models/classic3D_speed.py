"""Cached and batched implementation of the Classic3D objective.

``Classic3DSpeed`` is deliberately opt-in.  It keeps the Classic3D data term
and regularizers, but caches static observing geometry, batches compatible
reprojections, and reuses FINUFFT/cuFINUFFT plans between evaluations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_finufft
from torch.fft import fft2 as tfft2

from ivis.models.classic3D import Classic3D
from ivis.models.operators.geometry import resolve_pb_grid_lists, uvw_to_radpix
from ivis.models.operators.nufft import forward_nufft

try:  # FINUFFT is an optional IVIS CPU dependency.
    import finufft
except ImportError:  # pragma: no cover - exercised by the functional fallback
    finufft = None

try:  # cuFINUFFT is optional; pytorch_finufft remains the CUDA fallback.
    import cufinufft
except ImportError:  # pragma: no cover - depends on the CUDA installation
    cufinufft = None


def _array_identity(value: Any) -> tuple[Any, ...]:
    """Stable identity for an ndarray or a repeatable ndarray view."""
    array = np.asarray(value)
    return (
        int(array.__array_interface__["data"][0]),
        tuple(array.shape),
        tuple(array.strides),
        array.dtype.str,
    )


def _float_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Make a native-endian float32 tensor once, including FITS arrays."""
    return torch.from_numpy(np.array(value, dtype=np.float32, copy=True)).to(device)


class _PlanType2(torch.autograd.Function):
    """Type-2 FINUFFT whose adjoint reuses a cached type-1 FINUFFT plan."""

    @staticmethod
    def forward(ctx, image: torch.Tensor, plan_pair: "_PlanPair") -> torch.Tensor:
        ctx.plan_pair = plan_pair
        result = plan_pair.forward.execute(image.detach().contiguous().numpy())
        return torch.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        result = ctx.plan_pair.adjoint.execute(
            grad_output.detach().contiguous().numpy()
        )
        return torch.from_numpy(result), None


class _CudaPlanType2(torch.autograd.Function):
    """Type-2 cuFINUFFT whose adjoint reuses a cached type-1 plan."""

    @staticmethod
    def forward(ctx, image: torch.Tensor, plan_pair: "_PlanPair") -> torch.Tensor:
        ctx.plan_pair = plan_pair
        return plan_pair.forward.execute(image.detach().contiguous())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.plan_pair.adjoint.execute(grad_output.detach().contiguous()), None


@dataclass
class _PlanPair:
    forward: Any
    adjoint: Any


@dataclass
class _Block:
    c: int
    data_real: torch.Tensor
    data_imag: torch.Tensor
    sigma: torch.Tensor
    u_radpix: torch.Tensor | None
    v_radpix: torch.Tensor | None
    # CUDA's functional interface takes a stacked (2, nvis) tensor.  It is
    # static for a visibility block, so constructing it in every closure is
    # needless device work and allocation churn.
    points: torch.Tensor | None
    plan_pair: _PlanPair | None
    cuda_plan_pair: _PlanPair | None


@dataclass
class _Batch:
    c: int
    grids: torch.Tensor
    primary_beams: torch.Tensor
    blocks: list[_Block]
    cuda_plan_pair: _PlanPair | None
    data_real: torch.Tensor | None
    data_imag: torch.Tensor | None
    sigma: torch.Tensor | None


class Classic3DSpeed(Classic3D):
    """Classic3D with cached FINUFFT plans and batched reprojections.

    Parameters
    ----------
    nufft_eps
        Requested CPU FINUFFT relative tolerance.  ``1e-4`` was benchmarked
        as a fast mode; use ``1e-6`` to match Classic3D's numerical setting.
    reprojection_batch_size
        Number of same-channel, same-output-shape pointings in one
        ``grid_sample`` call.  The final batch may be smaller.
    use_plan_cache
        Reuse CPU FINUFFT or CUDA cuFINUFFT plans when the corresponding
        optional package is installed. Other installations retain the
        functional NUFFT backend.
    common_uv_sampling
        Keep only exact UV samples shared by every non-empty pointing in a
        channel. This excludes unique samples, but enables multi-transform
        CUDA plans. Disabled by default.
    """

    def __init__(
        self,
        lambda_r: float = 1,
        use_2pi: bool = True,
        conj_data: bool = True,
        *,
        nufft_eps: float = 1e-4,
        reprojection_batch_size: int = 4,
        use_plan_cache: bool = True,
        common_uv_sampling: bool = False,
    ):
        super().__init__(lambda_r=lambda_r, use_2pi=use_2pi, conj_data=conj_data)
        if nufft_eps <= 0:
            raise ValueError("nufft_eps must be positive.")
        if reprojection_batch_size < 1:
            raise ValueError("reprojection_batch_size must be at least one.")
        self.nufft_eps = float(nufft_eps)
        self.reprojection_batch_size = int(reprojection_batch_size)
        self.use_plan_cache = bool(use_plan_cache)
        self.common_uv_sampling = bool(common_uv_sampling)
        self._speed_caches: dict[tuple[Any, ...], list[_Batch]] = {}
        self._static_tensor_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def clear_speed_cache(self) -> None:
        """Release cached static tensors and FINUFFT/cuFINUFFT plans."""
        self._speed_caches.clear()
        self._static_tensor_cache.clear()

    def _static_tensor(self, value: Any, device: torch.device) -> torch.Tensor:
        """Cache an immutable numpy input after its one-time device transfer."""
        array = np.asarray(value)
        key = (_array_identity(array), str(device))
        tensor = self._static_tensor_cache.get(key)
        if tensor is None:
            # ``ascontiguousarray`` also handles non-contiguous FITS views
            # accepted by the original objective.
            tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
            self._static_tensor_cache[key] = tensor
        return tensor

    def _cache_key(self, vis_data, primary_beam_list, grid_list, device, cell_size, shape):
        return (
            _array_identity(vis_data.data_I),
            _array_identity(vis_data.sigma_I),
            _array_identity(vis_data.flag_I),
            _array_identity(vis_data.uu),
            _array_identity(vis_data.vv),
            tuple(_array_identity(item) for item in primary_beam_list),
            tuple(_array_identity(item) for item in grid_list),
            str(device),
            float(cell_size),
            tuple(shape),
            self.conj_data,
            self.nufft_eps,
            self.reprojection_batch_size,
            self.use_plan_cache,
            self.common_uv_sampling,
        )

    def _make_plan_pair(self, u_radpix, v_radpix, image_shape):
        if not self.use_plan_cache or finufft is None:
            return None
        x = (-v_radpix).detach().contiguous().numpy()
        y = u_radpix.detach().contiguous().numpy()
        forward = finufft.Plan(
            2, image_shape, eps=self.nufft_eps, isign=1,
            dtype="complex64", modeord=0,
        )
        adjoint = finufft.Plan(
            1, image_shape, eps=self.nufft_eps, isign=-1,
            dtype="complex64", modeord=0,
        )
        forward.setpts(x, y)
        adjoint.setpts(x, y)
        return _PlanPair(forward=forward, adjoint=adjoint)

    def _make_cuda_plan_pair(self, u_radpix, v_radpix, image_shape, device, n_trans=1):
        """Create reusable cuFINUFFT plans for a static CUDA visibility block."""
        if not self.use_plan_cache or cufinufft is None:
            return None
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        # pytorch_finufft also uses cuFINUFFT's native mode order for this
        # model.  Do not pass ``modeord``: older cuFINUFFT Python bindings do
        # not accept it, and native order is the desired modeord=0 behavior.
        options = dict(dtype="complex64", gpu_device_id=device_id)
        forward = cufinufft.Plan(
            2, image_shape, n_trans=n_trans, eps=self.nufft_eps, isign=1, **options
        )
        adjoint = cufinufft.Plan(
            1, image_shape, n_trans=n_trans, eps=self.nufft_eps, isign=-1, **options
        )
        forward.setpts(-v_radpix, u_radpix)
        adjoint.setpts(-v_radpix, u_radpix)
        return _PlanPair(forward=forward, adjoint=adjoint)

    @staticmethod
    def _uv_occurrence_keys(uu, vv):
        """Exact UV keys that retain the multiplicity of repeated samples."""
        points = np.ascontiguousarray(np.column_stack((uu, vv)))
        counts: dict[bytes, int] = {}
        keys = []
        for row in points:
            coordinate = row.tobytes()
            occurrence = counts.get(coordinate, 0)
            counts[coordinate] = occurrence + 1
            keys.append((coordinate, occurrence))
        return keys

    def _common_uv_indices(self, vis_data):
        """Map blocks to a canonical per-channel sequence of shared UV rows."""
        by_channel: dict[int, list[tuple[tuple[int, int], list[tuple[bytes, int]]]]] = defaultdict(list)
        for c, b, _data, _sigma, uu, vv, _ww in vis_data.iter_chan_beam_I():
            by_channel[c].append(((c, b), self._uv_occurrence_keys(uu, vv)))

        selections: dict[tuple[int, int], np.ndarray] = {}
        for c, entries in by_channel.items():
            common = set(entries[0][1])
            for _block_key, keys in entries[1:]:
                common.intersection_update(keys)
            if not common:
                raise ValueError(
                    f"common_uv_sampling found no UV samples shared by all "
                    f"non-empty pointings in channel {c}."
                )

            canonical = [key for key in entries[0][1] if key in common]
            for block_key, keys in entries:
                positions = {key: index for index, key in enumerate(keys)}
                selections[block_key] = np.asarray(
                    [positions[key] for key in canonical], dtype=np.intp
                )
        return selections

    def _prepare_cache(self, vis_data, primary_beam_list, grid_list, device, cell_size, image_shape):
        cache_key = self._cache_key(
            vis_data, primary_beam_list, grid_list, device, cell_size, image_shape
        )
        cached = self._speed_caches.get(cache_key)
        if cached is not None:
            return cached

        common_indices = self._common_uv_indices(vis_data) if self.common_uv_sampling else None

        # Store geometry only until it has been concatenated into each batch.
        # Persisting both the individual tensors and their concatenated batch
        # tensors roughly doubled the grid/PB cache footprint.
        by_channel_and_shape: dict[
            tuple[int, tuple[int, int]], list[tuple[_Block, torch.Tensor, torch.Tensor]]
        ] = defaultdict(list)
        for c, b, data, sigma, uu, vv, _ww in vis_data.iter_chan_beam_I():
            if common_indices is not None:
                indices = common_indices[(c, b)]
                data, sigma = data[indices], sigma[indices]
                uu, vv = uu[indices], vv[indices]
            grid = _float_tensor(grid_list[b], device)
            primary_beam = _float_tensor(primary_beam_list[b], device)
            if grid.ndim != 4 or grid.shape[0] != 1 or grid.shape[-1] != 2:
                raise ValueError(f"grid must have shape (1,H,W,2), got {tuple(grid.shape)}")
            if tuple(grid.shape[1:3]) != tuple(primary_beam.shape):
                raise ValueError("grid and primary beam output shapes must match.")

            _cell_rad, u_radpix, v_radpix = uvw_to_radpix(uu, vv, cell_size, device)
            plan_pair = (
                self._make_plan_pair(u_radpix, v_radpix, tuple(primary_beam.shape))
                if device.type == "cpu"
                else None
            )
            cuda_plan_pair = (
                self._make_cuda_plan_pair(
                    u_radpix, v_radpix, tuple(primary_beam.shape), device
                )
                if device.type == "cuda" and not self.common_uv_sampling
                else None
            )
            data_use = data.conj() if self.conj_data else data
            block = _Block(
                c=c,
                data_real=_float_tensor(data_use.real, device),
                data_imag=_float_tensor(data_use.imag, device),
                sigma=_float_tensor(sigma, device),
                # CPU plans retain their points internally.  Keeping a second
                # pair of UV tensors for every block is unnecessary there.
                u_radpix=None if plan_pair is not None else u_radpix,
                v_radpix=None if plan_pair is not None else v_radpix,
                points=(torch.stack([-v_radpix, u_radpix], dim=0)
                        if device.type == "cuda" and cuda_plan_pair is None else None),
                plan_pair=plan_pair,
                cuda_plan_pair=cuda_plan_pair,
            )
            by_channel_and_shape[(c, tuple(primary_beam.shape))].append(
                (block, grid, primary_beam)
            )

        batches: list[_Batch] = []
        for (c, _shape), entries in by_channel_and_shape.items():
            for start in range(0, len(entries), self.reprojection_batch_size):
                batch_entries = entries[start:start + self.reprojection_batch_size]
                blocks = [entry[0] for entry in batch_entries]
                batch_cuda_plan_pair = None
                if device.type == "cuda" and self.common_uv_sampling:
                    batch_cuda_plan_pair = self._make_cuda_plan_pair(
                        blocks[0].u_radpix,
                        blocks[0].v_radpix,
                        tuple(batch_entries[0][2].shape),
                        device,
                        n_trans=len(blocks),
                    )
                batches.append(_Batch(
                    c=c,
                    grids=torch.cat([entry[1] for entry in batch_entries], dim=0),
                    primary_beams=torch.stack([entry[2] for entry in batch_entries]),
                    blocks=blocks,
                    cuda_plan_pair=batch_cuda_plan_pair,
                    data_real=(torch.stack([block.data_real for block in blocks])
                               if batch_cuda_plan_pair is not None else None),
                    data_imag=(torch.stack([block.data_imag for block in blocks])
                               if batch_cuda_plan_pair is not None else None),
                    sigma=(torch.stack([block.sigma for block in blocks])
                           if batch_cuda_plan_pair is not None else None),
                ))
        self._speed_caches[cache_key] = batches
        return batches

    def _forward_nufft(self, image, block, cell_size):
        if block.plan_pair is not None:
            return (cell_size ** 2) * _PlanType2.apply(
                image.to(torch.complex64), block.plan_pair
            )
        if block.cuda_plan_pair is not None:
            return (cell_size ** 2) * _CudaPlanType2.apply(
                image.to(torch.complex64), block.cuda_plan_pair
            )
        # The functional CUDA backend supports the same requested tolerance,
        # even though it does not yet have the CPU plan-cache implementation.
        if image.device.type == "cuda":
            if block.points is None:
                raise RuntimeError("CUDA NUFFT requires cached UV points.")
            return (cell_size ** 2) * pytorch_finufft.functional.finufft_type2(
                block.points, image.to(torch.complex64),
                isign=1, modeord=0, eps=self.nufft_eps,
            )
        if block.u_radpix is None or block.v_radpix is None:
            raise RuntimeError("NUFFT fallback requires cached UV coordinates.")
        return forward_nufft(
            x_pb=image,
            u_radpix=block.u_radpix,
            v_radpix=block.v_radpix,
            cell_size=cell_size,
        )

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
        **_,
    ):
        dev = torch.device(device)
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
        speed_batches = self._prepare_cache(
            vis_data, primary_beam_list, grid_list, dev, cell_size, tuple(x.shape[-2:])
        )

        loss_value = torch.zeros((), dtype=x.dtype, device=dev)
        for batch in speed_batches:
            batch_size = len(batch.blocks)
            images = x[batch.c].unsqueeze(0).unsqueeze(0).expand(
                batch_size, -1, -1, -1
            )
            projected = F.grid_sample(
                images, batch.grids, mode="bilinear", align_corners=True
            ).squeeze(1)
            beamed = projected * batch.primary_beams

            if batch.cuda_plan_pair is not None:
                model_vis = (cell_size ** 2) * _CudaPlanType2.apply(
                    beamed.to(torch.complex64), batch.cuda_plan_pair
                )
                residual_real = (model_vis.real - batch.data_real) / batch.sigma
                residual_imag = (model_vis.imag - batch.data_imag) / batch.sigma
                batch_loss = 0.5 * torch.sum(
                    residual_real.square() + residual_imag.square()
                )
            else:
                batch_loss = torch.zeros((), dtype=x.dtype, device=dev)
                for index, block in enumerate(batch.blocks):
                    model_vis = self._forward_nufft(beamed[index], block, cell_size)
                    residual_real = (model_vis.real - block.data_real) / block.sigma
                    residual_imag = (model_vis.imag - block.data_imag) / block.sigma
                    batch_loss = batch_loss + 0.5 * torch.sum(
                        residual_real.square() + residual_imag.square()
                    )
            batch_loss.backward()
            loss_value = loss_value + batch_loss.detach()

        # Keep Classic3D's regularizers numerically and structurally unchanged.
        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t = self._static_tensor(fftsd, dev)
            fftbeam_t = self._static_tensor(fftbeam, dev)
            tapper_t = self._static_tensor(tapper, dev)
            for c in range(x.shape[0]):
                fftsd_c = fftsd_t[c] if fftsd_t.ndim == x.ndim else fftsd_t
                fftbeam_c = fftbeam_t[c] if fftbeam_t.ndim == x.ndim else fftbeam_t
                tapper_c = tapper_t[c] if tapper_t.ndim == x.ndim else tapper_t
                xfft2 = tfft2(x[c] * tapper_c)
                model_sd = (cell_size**2) * xfft2 * fftbeam_c
                loss = 0.5 * (
                    torch.nansum((model_sd.real - fftsd_c.real) ** 2)
                    + torch.nansum((model_sd.imag - fftsd_c.imag) ** 2)
                ) * lambda_sd
                loss.backward()
                loss_value = loss_value + loss.detach()

        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t = self._static_tensor(tapper, dev)
            fftkernel_t = self._static_tensor(fftkernel, dev)
            for c in range(x.shape[0]):
                fftkernel_c = fftkernel_t[c] if fftkernel_t.ndim == x.ndim else fftkernel_t
                tapper_c = tapper_t[c] if tapper_t.ndim == x.ndim else tapper_t
                xfft2 = tfft2(x[c] * tapper_c)
                conv = (cell_size**2) * xfft2 * fftkernel_c
                loss = 0.5 * torch.nansum(torch.abs(conv) ** 2) * self.lambda_r
                loss.backward()
                loss_value = loss_value + loss.detach()

        return loss_value


# Compatibility-friendly spelling for users who prefer the requested file name.
Classic3D_speed = Classic3DSpeed
