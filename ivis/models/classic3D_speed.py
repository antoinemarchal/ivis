"""Cached and batched implementation of the Classic3D objective.

``Classic3DSpeed`` is deliberately opt-in.  It keeps the Classic3D data term
and regularizers, but caches static observing geometry, batches compatible
reprojections, and reuses FINUFFT/cuFINUFFT plans between evaluations.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_finufft
from torch.fft import fft2 as tfft2

from ivis.logger import logger
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
    """Type-2 cuFINUFFT using a plan shared by compatible visibility blocks.

    cuFINUFFT plans own a large oversampled Fourier workspace.  Points are
    deliberately an input to this function rather than being fixed when the
    plan is made: one plan pair can therefore serve all blocks of an image
    shape, without retaining one workspace per pointing.
    """

    @staticmethod
    def forward(
        ctx, image: torch.Tensor, points: torch.Tensor, plan_pair: "_PlanPair"
    ) -> torch.Tensor:
        ctx.plan_pair = plan_pair
        ctx.save_for_backward(points)
        plan_pair.forward.setpts(points[0], points[1])
        return plan_pair.forward.execute(image.detach().contiguous())

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (points,) = ctx.saved_tensors
        ctx.plan_pair.adjoint.setpts(points[0], points[1])
        return (
            ctx.plan_pair.adjoint.execute(grad_output.detach().contiguous()),
            None,
            None,
        )


@dataclass
class _PlanPair:
    forward: Any
    adjoint: Any


@dataclass
class _Block:
    c: int
    b: int
    data_real: torch.Tensor | None
    data_imag: torch.Tensor | None
    sigma: torch.Tensor | None
    u_radpix: torch.Tensor | None
    v_radpix: torch.Tensor | None
    # CUDA requires a stacked (2, nvis) tensor.  It is static for a visibility
    # block and is also used to configure a shared cuFINUFFT plan pair.
    points: torch.Tensor | None
    plan_pair: _PlanPair | None
    cuda_plan_pair: _PlanPair | None


@dataclass
class _Batch:
    c: int
    # Keep one source geometry tensor per pointing.  The concatenated/stacked
    # tensors are created only while this batch is evaluated; retaining one
    # such pair for every channel batch exhausts GPU memory on large mosaics.
    grids: tuple[torch.Tensor, ...]
    primary_beams: tuple[torch.Tensor, ...]
    blocks: list[_Block]


@dataclass
class _CudaVisibility:
    """A stable GPU copy of a visibility block eligible for plan reuse."""

    data_real: torch.Tensor
    data_imag: torch.Tensor
    sigma: torch.Tensor
    points: torch.Tensor

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.data_real, self.data_imag, self.sigma, self.points)
        )


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
        Reuse CPU FINUFFT plans and, on CUDA, one cuFINUFFT plan pair per
        image shape/device when the corresponding optional package is
        installed. Other installations retain the functional NUFFT backend.
    cuda_visibility_cache_bytes
        Maximum GPU memory for stable cached CUDA visibility blocks.  ``None``
        selects ``min(8 GiB, 25% of device memory)``.  Other blocks are
        streamed and use the functional CUDA NUFFT backend.
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
        cuda_visibility_cache_bytes: int | None = None,
    ):
        super().__init__(lambda_r=lambda_r, use_2pi=use_2pi, conj_data=conj_data)
        if nufft_eps <= 0:
            raise ValueError("nufft_eps must be positive.")
        if reprojection_batch_size < 1:
            raise ValueError("reprojection_batch_size must be at least one.")
        if cuda_visibility_cache_bytes is not None and cuda_visibility_cache_bytes < 0:
            raise ValueError("cuda_visibility_cache_bytes must be non-negative.")
        self.nufft_eps = float(nufft_eps)
        self.reprojection_batch_size = int(reprojection_batch_size)
        self.use_plan_cache = bool(use_plan_cache)
        self.cuda_visibility_cache_bytes = cuda_visibility_cache_bytes
        self._speed_caches: dict[tuple[Any, ...], list[_Batch]] = {}
        self._static_tensor_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._cuda_plan_pairs: dict[tuple[Any, ...], _PlanPair] = {}
        self._cuda_visibility_cache: OrderedDict[tuple[int, int], _CudaVisibility] = OrderedDict()
        self._cuda_visibility_cache_bytes_used = 0
        self._cuda_visibility_cache_source_key: tuple[Any, ...] | None = None

    def clear_speed_cache(self) -> None:
        """Release cached static tensors and FINUFFT/cuFINUFFT plans."""
        self._speed_caches.clear()
        self._static_tensor_cache.clear()
        self._cuda_plan_pairs.clear()
        self._cuda_visibility_cache.clear()
        self._cuda_visibility_cache_bytes_used = 0
        self._cuda_visibility_cache_source_key = None

    def projected_fista(
        self,
        x_init,
        *,
        device,
        max_its: int,
        initial_step: float | None = None,
        initial_update: float = 1.0e-5,
        backtracking_factor: float = 0.5,
        grow_factor: float = 1.25,
        **params,
    ) -> np.ndarray:
        """Minimize this model's objective with exact non-negative projection.

        This model-local solver deliberately uses ``objective`` directly rather
        than the shared FISTA implementation: Classic3DSpeed streams its
        gradient and does not expose the normal-operator interface required by
        that generic solver.  It is a monotone, backtracking FISTA variant;
        every accepted physical image satisfies ``x >= 0`` exactly.
        """
        if max_its < 1:
            raise ValueError("max_its must be at least one.")
        if not 0.0 < backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must lie between zero and one.")
        if grow_factor < 1.0:
            raise ValueError("grow_factor must be at least one.")
        if initial_update <= 0.0:
            raise ValueError("initial_update must be positive.")

        dev = torch.device(device)
        x = torch.as_tensor(x_init, dtype=torch.float32, device=dev).clone().clamp_min_(0)
        y = x.clone()
        t_k = 1.0

        def evaluate(candidate):
            leaf = candidate.detach().requires_grad_(True)
            loss = self.objective(leaf, device=dev, **params)
            if leaf.grad is None:
                raise RuntimeError("Classic3DSpeed objective did not produce a gradient.")
            return loss.detach(), leaf.grad.detach()

        loss_x, grad_x = evaluate(x)
        if initial_step is None:
            # Choose a scale-aware first trial step, then let majorization
            # backtracking make it safe. ``initial_update`` is the requested
            # maximum first-step excursion in the image's physical units.
            initial_step = initial_update / max(float(grad_x.abs().max()), 1.0e-20)
        step = float(initial_step)
        logger.info(
            f"Starting projected FISTA on {dev}; exact positivity projection; "
            f"initial step={step:.6e}; initial update={initial_update:.6e}"
        )

        for iteration in range(1, int(max_its) + 1):
            loss_y, grad_y = evaluate(y)
            reference_y = y
            local_t = t_k

            # A monotone restart prevents Nesterov extrapolation from moving
            # away from the last accepted constrained image.
            if loss_y > loss_x:
                reference_y = x
                loss_y, grad_y = loss_x, grad_x
                local_t = 1.0

            accepted = False
            for _ in range(30):
                candidate = (reference_y - step * grad_y).clamp_min(0)
                loss_candidate, grad_candidate = evaluate(candidate)
                delta = candidate - reference_y
                majorizer = loss_y + torch.sum(grad_y * delta) + torch.sum(delta * delta) / (2.0 * step)
                if torch.isfinite(loss_candidate) and loss_candidate <= majorizer:
                    accepted = True
                    break
                step *= backtracking_factor

            if not accepted:
                raise RuntimeError("Projected FISTA failed to find a finite descent step.")
            if loss_candidate > loss_x:
                # A finite projected step may pass the local majorization test
                # after an extrapolated restart but still not improve x. Keep
                # the previous feasible point and restart on the next pass.
                y = x.clone()
                t_k = 1.0
                step *= backtracking_factor
                logger.info(
                    f"[Projected FISTA Iter {iteration}/{max_its}] restart; "
                    f"loss={float(loss_x):.6e}; step={step:.6e}"
                )
                continue

            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * local_t * local_t))
            y = candidate + ((local_t - 1.0) / t_next) * (candidate - x)
            relative_change = torch.linalg.vector_norm(candidate - x) / torch.clamp_min(
                torch.linalg.vector_norm(candidate), 1.0e-20
            )
            x, loss_x, grad_x, t_k = candidate, loss_candidate, grad_candidate, t_next
            step *= grow_factor
            logger.info(
                f"[Projected FISTA Iter {iteration}/{max_its}] "
                f"loss={float(loss_x):.6e}; rel_change={float(relative_change):.6e}; "
                f"step={step:.6e}"
            )

        return x.detach().cpu().numpy()

    def process_projected_fista(
        self,
        image_processor,
        *,
        units: str = "Jy/arcsec^2",
        initial_step: float | None = None,
        initial_update: float = 1.0e-5,
        backtracking_factor: float = 0.5,
        grow_factor: float = 1.25,
    ) -> np.ndarray:
        """Run projected FISTA using an existing :class:`Imager3D` setup.

        This is a Speed-model convenience entry point, so callers can retain
        their usual ``Imager3D`` configuration without modifying the shared
        imager or solver dispatch.  It supports the same output units as
        ``Imager3D.process``.
        """
        from astropy import units as u
        from radio_beam import Beam

        from ivis.utils import dunits, dutils

        if units not in {"Jy/arcsec^2", "Jy/beam", "K"}:
            logger.warning("Unknown unit type. Returning result in Jy/arcsec^2.")
            units = "Jy/arcsec^2"

        hdr = image_processor.hdr
        shape = (hdr["NAXIS2"], hdr["NAXIS1"])
        cell_size = (hdr["CDELT2"] * u.deg).to(u.arcsec)
        tapper = dutils.apodize(0.98, shape)
        fftkernel = np.abs(np.fft.fft2(dutils.laplacian(shape)))
        bmaj_pix = image_processor.beam_sd.major.to(u.deg).value / cell_size.to(u.deg).value
        beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
        fftbeam = np.abs(np.fft.fft2(beam))
        fftsd = cell_size.value**2 * tfft2(
            torch.from_numpy(np.float32(image_processor.sd))
        ).cpu().numpy()
        params = dict(
            vis_data=image_processor.vis_data,
            pb=np.asarray(image_processor.pb, dtype=np.float32),
            fftbeam=np.asarray(fftbeam, dtype=np.float32),
            fftsd=np.asarray(fftsd, dtype=np.complex64),
            tapper=np.asarray(tapper, dtype=np.float32),
            lambda_sd=image_processor.lambda_sd,
            fftkernel=np.asarray(fftkernel, dtype=np.float32),
            cell_size=cell_size.value,
            grid_array=np.asarray(image_processor.grid, dtype=np.float32),
            beam_workers=image_processor.beam_workers,
        )
        result = self.projected_fista(
            image_processor.init_params,
            device=image_processor.cost_device,
            max_its=image_processor.max_its,
            initial_step=initial_step,
            initial_update=initial_update,
            backtracking_factor=backtracking_factor,
            grow_factor=grow_factor,
            **params,
        )

        if units == "Jy/arcsec^2":
            output = result
        elif units == "Jy/beam":
            assumed_fwhm_pix = 3
            logger.warning(
                "Converting to Jy/beam assuming a restoring beam of "
                f"{assumed_fwhm_pix} × cell_size = "
                f"{assumed_fwhm_pix * cell_size:.3f} FWHM."
            )
            beam_r = Beam(
                assumed_fwhm_pix * cell_size,
                assumed_fwhm_pix * cell_size,
                1.e-12 * u.deg,
            )
            output = result * beam_r.sr.to(u.arcsec**2).value
        else:  # K
            output = dunits.jy_per_arcsec2_to_K(result, image_processor.vis_data.frequency)

        logger.info("Successful run. Please clap.")
        return output

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

    def _cuda_plan_key(self, image_shape, device):
        """Key a cuFINUFFT workspace by the properties that determine it."""
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        return (tuple(image_shape), str(device.type), device_id, self.nufft_eps)

    def _make_cuda_plan_pair(self, image_shape, device):
        """Create a cuFINUFFT workspace shared by compatible CUDA blocks."""
        if not self.use_plan_cache or cufinufft is None:
            return None
        device_id = device.index if device.index is not None else torch.cuda.current_device()
        # pytorch_finufft also uses cuFINUFFT's native mode order for this
        # model.  Do not pass ``modeord``: older cuFINUFFT Python bindings do
        # not accept it, and native order is the desired modeord=0 behavior.
        options = dict(dtype="complex64", gpu_device_id=device_id)
        forward = cufinufft.Plan(
            2, image_shape, eps=self.nufft_eps, isign=1, **options
        )
        adjoint = cufinufft.Plan(
            1, image_shape, eps=self.nufft_eps, isign=-1, **options
        )
        return _PlanPair(forward=forward, adjoint=adjoint)

    def _get_cuda_plan_pair(self, image_shape, device):
        """Return the single cached CUDA workspace for this image shape."""
        if not self.use_plan_cache or cufinufft is None:
            return None
        key = self._cuda_plan_key(image_shape, device)
        plan_pair = self._cuda_plan_pairs.get(key)
        if plan_pair is None:
            plan_pair = self._make_cuda_plan_pair(image_shape, device)
            self._cuda_plan_pairs[key] = plan_pair
        return plan_pair

    def _cuda_visibility_cache_limit(self, device: torch.device) -> int:
        if self.cuda_visibility_cache_bytes is not None:
            return int(self.cuda_visibility_cache_bytes)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        return min(8 * 1024**3, total_memory // 4)

    def _cuda_visibility(self, vis_data, block: _Block, cell_size, device):
        """Get a stable cached CUDA block, or stream one for this evaluation.

        The fixed cache deliberately stops filling at its budget instead of
        using an LRU.  FISTA scans all blocks in the same order every time, so
        an undersized LRU would evict every entry before the next evaluation
        reached it and would provide no reuse.
        """
        key = (block.c, block.b)
        cached = self._cuda_visibility_cache.get(key)
        if cached is not None:
            return cached, True

        data, sigma, uu, vv, _ww = vis_data.slice_chan_beam_I(block.c, block.b)
        data_use = data.conj() if self.conj_data else data
        data_real = _float_tensor(data_use.real, device)
        data_imag = _float_tensor(data_use.imag, device)
        sigma_t = _float_tensor(sigma, device)
        _cell_rad, u_radpix, v_radpix = uvw_to_radpix(uu, vv, cell_size, device)
        points = torch.stack([-v_radpix, u_radpix], dim=0)
        entry = _CudaVisibility(data_real, data_imag, sigma_t, points)

        if self._cuda_visibility_cache_bytes_used + entry.nbytes <= self._cuda_visibility_cache_limit(device):
            self._cuda_visibility_cache[key] = entry
            self._cuda_visibility_cache_bytes_used += entry.nbytes
            return entry, True
        return entry, False

    def _prepare_cache(self, vis_data, primary_beam_list, grid_list, device, cell_size, image_shape):
        cache_key = self._cache_key(
            vis_data, primary_beam_list, grid_list, device, cell_size, image_shape
        )
        if device.type == "cuda" and self._cuda_visibility_cache_source_key != cache_key:
            self._cuda_visibility_cache.clear()
            self._cuda_visibility_cache_bytes_used = 0
            self._cuda_visibility_cache_source_key = cache_key
        cached = self._speed_caches.get(cache_key)
        if cached is not None:
            return cached

        # Geometry is identical for each channel of a pointing.  Cache only
        # one GPU copy per pointing, and keep batch concatenations transient
        # in ``objective``.  Retaining a concatenation for every channel batch
        # otherwise makes the cache scale as channels * pointings * image area.
        beam_geometry: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        by_channel_and_shape: dict[
            tuple[int, tuple[int, int]], list[tuple[_Block, torch.Tensor, torch.Tensor]]
        ] = defaultdict(list)
        for c, b, data, sigma, uu, vv, _ww in vis_data.iter_chan_beam_I():
            geometry = beam_geometry.get(b)
            if geometry is None:
                grid = _float_tensor(grid_list[b], device)
                primary_beam = _float_tensor(primary_beam_list[b], device)
                if grid.ndim != 4 or grid.shape[0] != 1 or grid.shape[-1] != 2:
                    raise ValueError(
                        f"grid must have shape (1,H,W,2), got {tuple(grid.shape)}"
                    )
                if tuple(grid.shape[1:3]) != tuple(primary_beam.shape):
                    raise ValueError("grid and primary beam output shapes must match.")
                geometry = (grid, primary_beam)
                beam_geometry[b] = geometry
            grid, primary_beam = geometry

            cuda_plan_pair = (
                self._get_cuda_plan_pair(tuple(primary_beam.shape), device)
                if device.type == "cuda"
                else None
            )
            if device.type == "cuda":
                block = _Block(
                    c=c, b=b, data_real=None, data_imag=None, sigma=None,
                    u_radpix=None, v_radpix=None, points=None,
                    plan_pair=None, cuda_plan_pair=cuda_plan_pair,
                )
            else:
                _cell_rad, u_radpix, v_radpix = uvw_to_radpix(
                    uu, vv, cell_size, device
                )
                plan_pair = self._make_plan_pair(
                    u_radpix, v_radpix, tuple(primary_beam.shape)
                )
                data_use = data.conj() if self.conj_data else data
                block = _Block(
                    c=c, b=b,
                    data_real=_float_tensor(data_use.real, device),
                    data_imag=_float_tensor(data_use.imag, device),
                    sigma=_float_tensor(sigma, device),
                    u_radpix=None if plan_pair is not None else u_radpix,
                    v_radpix=None if plan_pair is not None else v_radpix,
                    points=None, plan_pair=plan_pair, cuda_plan_pair=None,
                )
            by_channel_and_shape[(c, tuple(primary_beam.shape))].append(
                (block, grid, primary_beam)
            )

        batches: list[_Batch] = []
        for (c, _shape), entries in by_channel_and_shape.items():
            for start in range(0, len(entries), self.reprojection_batch_size):
                batch_entries = entries[start:start + self.reprojection_batch_size]
                batches.append(_Batch(
                    c=c,
                    grids=tuple(entry[1] for entry in batch_entries),
                    primary_beams=tuple(entry[2] for entry in batch_entries),
                    blocks=[entry[0] for entry in batch_entries],
                ))
        self._speed_caches[cache_key] = batches
        return batches

    def _forward_nufft(self, image, block, cell_size, *, points=None, use_cuda_plan=True):
        if block.plan_pair is not None:
            return (cell_size ** 2) * _PlanType2.apply(
                image.to(torch.complex64), block.plan_pair
            )
        if block.cuda_plan_pair is not None and use_cuda_plan:
            points = block.points if points is None else points
            if points is None:
                raise RuntimeError("Cached CUDA NUFFT plan requires UV points.")
            return (cell_size ** 2) * _CudaPlanType2.apply(
                image.to(torch.complex64), points, block.cuda_plan_pair
            )
        # The functional CUDA backend supports the same requested tolerance,
        # even though it does not yet have the CPU plan-cache implementation.
        if image.device.type == "cuda":
            points = block.points if points is None else points
            if points is None:
                raise RuntimeError("CUDA NUFFT requires cached UV points.")
            return (cell_size ** 2) * pytorch_finufft.functional.finufft_type2(
                points, image.to(torch.complex64),
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
            grids = torch.cat(batch.grids, dim=0)
            primary_beams = torch.stack(batch.primary_beams)
            # SciPy L-BFGS-B (positivity=True) supplies float64 parameters,
            # whereas cached grids are deliberately float32. grid_sample
            # requires matching types; the cast is differentiable, so its
            # float32 gradient is accumulated correctly on the float64 leaf.
            images = x[batch.c].to(dtype=grids.dtype).unsqueeze(0).unsqueeze(0)
            images = images.expand(batch_size, -1, -1, -1)
            projected = F.grid_sample(
                images, grids, mode="bilinear", align_corners=True
            ).squeeze(1)
            beamed = projected * primary_beams

            batch_loss = torch.zeros((), dtype=x.dtype, device=dev)
            for index, block in enumerate(batch.blocks):
                if dev.type == "cuda":
                    visibility, use_cuda_plan = self._cuda_visibility(
                        vis_data, block, cell_size, dev
                    )
                    data_real = visibility.data_real
                    data_imag = visibility.data_imag
                    sigma_t = visibility.sigma
                    points = visibility.points
                else:
                    if block.data_real is None or block.data_imag is None or block.sigma is None:
                        raise RuntimeError("CPU NUFFT block is missing cached visibility data.")
                    data_real = block.data_real
                    data_imag = block.data_imag
                    sigma_t = block.sigma
                    points = None
                    use_cuda_plan = True
                model_vis = self._forward_nufft(
                    beamed[index], block, cell_size, points=points,
                    use_cuda_plan=use_cuda_plan,
                )
                residual_real = (model_vis.real - data_real) / sigma_t
                residual_imag = (model_vis.imag - data_imag) / sigma_t
                batch_loss = batch_loss + 0.5 * torch.sum(
                    residual_real.square() + residual_imag.square()
                )
            batch_loss.backward()
            loss_value = loss_value + batch_loss.detach()
            # ``backward`` has consumed this batch's autograd graph.  Release
            # its large, transient reprojection tensors before the next batch.
            del (
                grids, primary_beams, images, projected, beamed, batch_loss,
                data_real, data_imag, sigma_t, points, model_vis,
                residual_real, residual_imag,
            )
            if dev.type == "cuda":
                del visibility

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
