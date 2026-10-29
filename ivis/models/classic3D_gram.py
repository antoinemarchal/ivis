"""Opt-in Classic3D objective with a cached visibility Gram operator.

Import this module explicitly; no existing model or solver is replaced::

    from ivis.models.classic3D_gram import Classic3D_gram
    model = Classic3D_gram(lambda_r=1, nufft_eps=1e-6)
    result = imager.process(model=model, solver="FISTA")

FISTA reuses gradients across iterations by default for this fixed quadratic
model. Set ``quadratic_gradient_reuse=False`` in ``imager.process`` to compare
with the original FISTA path; ``loss_only_line_search=True`` then avoids
candidate gradients during backtracking.

``local_gradients=True`` restricts reprojection gradients and dirty-image
accumulation to each pointing's image footprint. Set it to False for the
original full-image path. Bilinear support is retained, including image-edge
neighbours; changing coordinate frames can introduce float32 roundoff.

For each channel/pointing, two type-1 NUFFTs prepare a weighted Toeplitz
kernel and dirty image. Subsequent objective calls use padded real FFTs,
primary beams, and bilinear reprojection. With ``padding_fraction=2.0``
(default), this represents the ideal weighted Fourier Gram operator;
agreement with a NUFFT forward/adjoint pair is limited by the requested
NUFFT tolerance and floating-point precision.

``padding_fraction`` accepts values from 1.0 (no padding) to 2.0 (full
Toeplitz embedding). Smaller embeddings periodically extend the central
kernel lags, averaging the two representatives on even-size Nyquist edges.
This keeps the quadratic and its gradient consistent, but approximates the
visibility fidelity and does not guarantee a positive semidefinite Hessian.
Validate reduced-padding reconstructions with the original visibility loss.

Inputs must stay fixed during a solve. Call ``clear_speed_cache()`` after
editing array contents in place. Only the most recently prepared observation
is cached. On CUDA, the pointing cache is bounded automatically to leave
room for setup, FISTA and FFT workspaces. Overflow tensors are kept in pinned
host memory and transferred only for the current pointing batch. Override
the pointing budget with ``gpu_cache_bytes`` (0 streams all pointings).
``max_cache_bytes`` separately limits total persistent Gram tensors on both
devices, excluding FFT workspaces and regularizer inputs.

``gram_dtype=torch.float64`` improves the accuracy of the cached quadratic
at additional memory and computation cost. Scalar loss products always use
float64. The expanded visibility loss can still lose accuracy near an almost
perfect fit: validate the final reconstruction with the original visibility
objective. ``visibility_loss()`` provides that independent check.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytorch_finufft
import torch
import torch.nn.functional as F
from scipy.fft import next_fast_len

from ivis.logger import logger
from ivis.models.classic3D import Classic3D
from ivis.models.operators.geometry import resolve_pb_grid_lists, uvw_to_radpix

try:
    import finufft
except ImportError:
    finufft = None

try:
    import cufinufft
except ImportError:
    cufinufft = None


def _identity(array: np.ndarray) -> tuple:
    return (array.__array_interface__["data"][0], array.shape,
            array.strides, array.dtype.str)


def _dot64(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    # Keep double-precision products without three full double image cubes
    # alive at once. Flattening may copy a cropped convolution, but in its
    # original precision and only for the current pointing batch.
    result = torch.zeros((), dtype=torch.float64, device=left.device)
    for a, b in zip(left.reshape(-1).split(1 << 20),
                    right.reshape(-1).split(1 << 20)):
        result.add_(torch.dot(a.double(), b.double()))
    return result


def _norm_squared64(values: torch.Tensor) -> torch.Tensor:
    # In particular, avoid nansum(..., dtype=float64) casting an entire
    # full-survey image to double before reduction.
    result = torch.zeros((), dtype=torch.float64, device=values.device)
    for chunk in values.reshape(-1).split(1 << 22):
        result.add_(torch.nansum(chunk.abs().double().square()))
    return result


@dataclass
class _Geometry:
    grid: torch.Tensor
    beam: torch.Tensor
    # Bounds in the global image: y0, y1, x0, x1 (exclusive upper limits).
    bounds: tuple[int, int, int, int]

    def to(self, device):
        return _Geometry(self.grid.to(device, non_blocking=True),
                         self.beam.to(device, non_blocking=True), self.bounds)


def _full_bounds(image_shape):
    return (0, image_shape[-2], 0, image_shape[-1])


def _footprint_bounds(grid, image_shape):
    """Conservative bilinear support, computed once from the stored grid dtype.

    Extrema can overestimate a rotated footprint, which is safe. One extra
    pixel at either end protects against unnormalization roundoff. A crop
    axis has at least two pixels unless the original image axis has size one:
    align_corners=True cannot represent off-axis positions on a size-one crop.
    """
    # Keep the large reductions on the grid's device. Only four extrema are
    # needed on the host, transferred together to avoid four CUDA syncs.
    extrema = torch.stack((grid.amin(dim=(0, 1, 2)),
                           grid.amax(dim=(0, 1, 2)))).cpu().tolist()
    bounds = []
    for component, size in ((1, image_shape[-2]), (0, image_shape[-1])):
        if size == 1:
            bounds.extend((0, 1))
            continue
        lo = (extrema[0][component] + 1) * 0.5 * (size - 1)
        hi = (extrema[1][component] + 1) * 0.5 * (size - 1)
        start = max(0, min(size, int(np.floor(lo)) - 1))
        stop = max(0, min(size, int(np.ceil(hi)) + 2))
        if stop <= start:
            return (0, 0, 0, 0)
        if stop - start == 1:
            start = max(0, start - 1)
            stop = min(size, start + 2)
        bounds.extend((start, stop))
    return tuple(bounds)


def _remap_grid_(grid, old_bounds, new_bounds):
    """Change normalized coordinate frames in place, without retaining indices."""
    for component, axis in ((0, 2), (1, 0)):
        old_start, old_stop = old_bounds[axis:axis + 2]
        new_start, new_stop = new_bounds[axis:axis + 2]
        if (old_start, old_stop) == (new_start, new_stop):
            continue
        coords = grid[..., component]
        coords.add_(1).mul_(0.5).mul_(old_stop - old_start - 1)
        coords.add_(old_start - new_start)
        coords.div_(new_stop - new_start - 1).mul_(2).sub_(1)
    return grid


def _crop(image, bounds):
    y0, y1, x0, x1 = bounds
    return image[y0:y1, x0:x1]


def _batch_bounds(batch):
    return (min(b.geometry.bounds[0] for b in batch),
            max(b.geometry.bounds[1] for b in batch),
            min(b.geometry.bounds[2] for b in batch),
            max(b.geometry.bounds[3] for b in batch))


@dataclass
class _GramBlock:
    channel: int
    geometry: _Geometry
    spectrum: torch.Tensor
    padded_shape: tuple[int, int]


@dataclass
class _GramCache:
    key: tuple
    blocks: list[_GramBlock]
    batches: list[list[_GramBlock]]
    rhs: torch.Tensor
    constant: torch.Tensor
    nbytes: int
    gpu_nbytes: int
    # Keep identity-keyed inputs alive; otherwise recycled memory addresses
    # could make a different observation appear to be the cached one.
    owners: tuple[np.ndarray, ...]


def _project(image: torch.Tensor, geometry: _Geometry) -> torch.Tensor:
    return F.grid_sample(
        image[None, None], geometry.grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0, 0] * geometry.beam


def _project_adjoint(values, geometry, image_shape):
    # Use the exact transpose of the same grid_sample operation. A local 2-D
    # leaf avoids full-cube backward temporaries and cached interpolation
    # indices for every pointing.
    with torch.enable_grad():
        # An expanded zero leaf needs no full-size forward image allocation.
        template = torch.zeros((), device=values.device, dtype=geometry.grid.dtype)
        template = template.expand(image_shape).detach().requires_grad_(True)
        projected = _project(template, geometry)
        return torch.autograd.grad(projected, template, values)[0].detach()


def _project_global(image, geometry):
    local = _crop(image, geometry.bounds)
    if not local.numel():
        return torch.zeros_like(geometry.beam)
    return _project(local, geometry)


def _padded_shape(image_shape, padding_fraction):
    # Exactly 1 means no padding, even for awkward FFT lengths. Preserve the
    # original full embedding at 2; intermediate sizes round up for FFTs.
    if padding_fraction == 1.0:
        return tuple(image_shape)
    return tuple(next_fast_len(min(2 * n - 1, int(np.ceil(padding_fraction * n))))
                 for n in image_shape)


def _lag_shape(image_shape, padded_shape):
    # An odd lag grid provides both +/- P/2 for even P, so the ambiguous
    # Nyquist edge can be averaged rather than silently overwritten.
    return tuple(min(2 * n - 1, 2 * (p // 2) + 1)
                 for n, p in zip(image_shape, padded_shape))


def _embed_kernel(kernel, image_shape, padded_shape=None):
    """Embed central lags, with a symmetric periodic extension if reduced."""
    if padded_shape is None:
        padded_shape = _padded_shape(image_shape, 2.0)
    if tuple(kernel.shape) != _lag_shape(image_shape, padded_shape):
        raise ValueError("Kernel must contain the centered lag grid for this padding.")
    embedded = kernel.new_zeros(padded_shape)
    radius_y, radius_x = (n // 2 for n in kernel.shape)
    iy = torch.arange(-radius_y, radius_y + 1, device=kernel.device) % padded_shape[0]
    ix = torch.arange(-radius_x, radius_x + 1, device=kernel.device) % padded_shape[1]
    # Real images need Re(N* W N). Enforce its centrosymmetry before the
    # FFT; its circulant spectrum is real, but need not be nonnegative.
    kernel = 0.5 * (kernel + kernel.flip((0, 1)))
    if kernel.shape[0] > padded_shape[0]:
        kernel[0].add_(kernel[-1]).mul_(0.5)
        kernel, iy = kernel[:-1], iy[:-1]
    if kernel.shape[1] > padded_shape[1]:
        kernel[:, 0].add_(kernel[:, -1]).mul_(0.5)
        kernel, ix = kernel[:, :-1], ix[:-1]
    embedded[iy[:, None], ix[None, :]] = kernel
    spectrum = torch.fft.rfft2(embedded).real.contiguous()
    return spectrum, padded_shape


def _apply_kernel(image, block, spectrum=None):
    spectrum = block.spectrum if spectrum is None else spectrum
    transformed = torch.fft.rfft2(image, s=block.padded_shape)
    transformed.mul_(spectrum)
    convolved = torch.fft.irfft2(transformed, s=block.padded_shape)
    height, width = image.shape[-2:]
    return convolved[..., :height, :width]


class Classic3D_gram(Classic3D):
    """Classic3D with a per-pointing Toeplitz visibility normal operator.

    Parameters match Classic3D, with additional keyword-only options:

    nufft_eps : float
        Accuracy requested when constructing kernels and dirty images.
    gram_dtype : torch.dtype
        float32 (default) or float64 for geometry, kernels and convolutions.
        UV coordinates retain Classic3D's float32 conversion convention.
    padding_fraction : float
        FFT extent relative to each pointing axis, between 1 and 2 inclusive.
        1 means no padding. 2 (default) preserves the full Toeplitz operator
        with next_fast_len(2*N-1) samples. Intermediate extents round upward
        to FFT-friendly sizes, capped at the full embedding. Smaller grids
        use a symmetric periodic approximation and may change convexity.
        Dirty images and inverse-variance weighting remain unchanged.
    reprojection_batch_size : int
        Same-channel pointings of the same shape to reproject and convolve
        together (default 1, as in Classic3D_optimized).
    use_plan_cache : bool
        Reuse FINUFFT/cuFINUFFT type-1 plans by shape during preparation.
        Setup plans are released once all kernels have been constructed;
        repeated evaluations use only the cached kernels and regular FFTs.
    local_gradients : bool
        Restrict reprojection and its adjoint to cached image footprints.
        Batches use the union of their footprints. Default False until GPU
        validation; False selects the original full-image accumulation.
    max_cache_bytes : int or None
        Optional limit for persistent Gram tensors, checked before building
        each block. A failed preparation leaves the cache empty.
    gpu_cache_bytes : int or None
        CUDA budget for pointing geometry and spectra. None chooses a budget
        from free memory, reserving at least 40% of the GPU for working arrays
        and allocator fragmentation between differently sized FFT buffers.
        Overflow stays in pinned CPU memory at the same precision. The global
        dirty image and regularizer tensors remain on the compute device.
        A value of 0 streams all pointings. This is not a total VRAM limit;
        the image, solver and one pointing batch still need to fit on CUDA.

    psf_positions : sequence of (row, column) or None
        Measures the response of the whole estimator to unit point sources.
        When given, the normal-equation right-hand side (the dirty image) is
        replaced by H*delta, where delta holds a unit pixel at each position
        in every channel. The objective, regularizers and solvers are
        unchanged, so an unconstrained solve returns (H + lambda*L)^-1 H delta:
        the point-source response of the same estimator, in the same units,
        with the same lambda scaling and iteration count as a normal run.
        Positions must be far enough apart that their responses do not
        overlap. The data values are ignored. Default None: normal imaging.

    objective() populates x.grad explicitly, as expected by IVIS solvers.
    FISTA uses quadratic gradient reuse by default. The original loss-only
    FISTA path and the existing unconstrained CG interface also work.
    """

    supports_loss_only_objective = True
    supports_quadratic_gradient_reuse = True

    def __init__(self, lambda_r=1, use_2pi=True, conj_data=True, *,
                 nufft_eps=1e-6, gram_dtype=torch.float32,
                 reprojection_batch_size=1, use_plan_cache=True,
                 max_cache_bytes=None, gpu_cache_bytes=None,
                 padding_fraction=2.0, local_gradients=False, psf_positions=None):
        super().__init__(lambda_r=lambda_r, use_2pi=use_2pi, conj_data=conj_data)
        if not np.isfinite(nufft_eps) or nufft_eps <= 0:
            raise ValueError("nufft_eps must be finite and positive.")
        if gram_dtype not in (torch.float32, torch.float64):
            raise ValueError("gram_dtype must be torch.float32 or torch.float64.")
        if (not isinstance(reprojection_batch_size, int)
                or reprojection_batch_size < 1):
            raise ValueError("reprojection_batch_size must be a positive integer.")
        if max_cache_bytes is not None and max_cache_bytes < 1:
            raise ValueError("max_cache_bytes must be positive or None.")
        if gpu_cache_bytes is not None and gpu_cache_bytes < 0:
            raise ValueError("gpu_cache_bytes must be nonnegative or None.")
        if not np.isfinite(padding_fraction) or not 1.0 <= padding_fraction <= 2.0:
            raise ValueError("padding_fraction must be finite and between 1 and 2.")
        self.nufft_eps = float(nufft_eps)
        self.gram_dtype = gram_dtype
        self.reprojection_batch_size = reprojection_batch_size
        self.use_plan_cache = bool(use_plan_cache)
        self.max_cache_bytes = max_cache_bytes
        self.gpu_cache_bytes = gpu_cache_bytes
        self.padding_fraction = float(padding_fraction)
        self.local_gradients = bool(local_gradients)
        if psf_positions is not None:
            try:
                psf_positions = tuple((int(row), int(col)) for row, col in psf_positions)
            except (TypeError, ValueError):
                raise ValueError("psf_positions must be a sequence of (row, column) integer pairs.") from None
            if not psf_positions:
                raise ValueError("psf_positions must not be empty; use None for normal imaging.")
        self.psf_positions = psf_positions
        self._gram_cache: _GramCache | None = None
        self._static_tensors: dict[tuple, tuple[np.ndarray, torch.Tensor]] = {}

    @property
    def cache_nbytes(self):
        """Bytes in persistent Gram tensors, excluding regularizer inputs."""
        return 0 if self._gram_cache is None else self._gram_cache.nbytes

    @property
    def cache_gpu_nbytes(self):
        """Bytes in persistent CUDA Gram tensors, including the global RHS."""
        return 0 if self._gram_cache is None else self._gram_cache.gpu_nbytes

    def _gpu_budget(self, device, x_shape, beams):
        if device.type != "cuda":
            return 0
        if self.gpu_cache_bytes is not None:
            return self.gpu_cache_bytes
        free, total = torch.cuda.mem_get_info(device)
        # Unused PyTorch allocator blocks can also be reused by this cache.
        available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        itemsize = torch.empty((), dtype=self.gram_dtype).element_size()
        image_bytes = int(np.prod(x_shape)) * itemsize
        padded_bytes = max((int(np.prod(_padded_shape(b.shape, self.padding_fraction))) * itemsize
                            for b in beams), default=0)
        workspace = 40 * image_bytes + 8 * self.reprojection_batch_size * padded_bytes
        reserve = max(2 * total // 5, workspace)
        return max(0, int(available - reserve - image_bytes - 8))

    def clear_speed_cache(self):
        self._gram_cache = None
        self._static_tensors.clear()
        self._interp_cache.clear()

    def _check_budget(self, nbytes):
        if self.max_cache_bytes is not None and nbytes > self.max_cache_bytes:
            raise MemoryError(
                f"Gram cache needs at least {nbytes} bytes; "
                f"max_cache_bytes={self.max_cache_bytes}. Use fewer channels "
                "or increase the cache budget."
            )

    def _tensor(self, value, device, dtype=None):
        array = np.asarray(value)
        # NPY sidecars are native-endian and can be transferred directly from
        # a memory map. FITS arrays and other non-native inputs still need a
        # native-endian copy before torch can consume them.
        if (not array.dtype.isnative or not array.flags.c_contiguous
                or not array.flags.writeable):
            array = np.array(array, dtype=array.dtype.newbyteorder("="),
                             order="C", copy=True)
        tensor = torch.from_numpy(array)
        return tensor.to(device=device, dtype=dtype)

    def _static(self, value, device):
        array = np.asarray(value)
        key = (_identity(array), str(device))
        cached = self._static_tensors.get(key)
        if cached is None:
            cached = (array, self._tensor(array, device))
            self._static_tensors[key] = cached
        return cached[1]

    @staticmethod
    def _device(device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def _type1(self, points, values, shape, plans):
        backend = finufft if points.device.type == "cpu" else cufinufft
        if self.use_plan_cache and backend is not None:
            key = (tuple(shape), str(points.device), values.dtype)
            plan = plans.get(key)
            if plan is None:
                options: dict[str, Any] = {
                    "dtype": "complex64" if values.dtype == torch.complex64 else "complex128",
                }
                if points.device.type == "cpu":
                    options["modeord"] = 0
                else:
                    # Native cuFINUFFT order is modeord=0; older versions
                    # do not accept the modeord keyword.
                    options["gpu_device_id"] = points.device.index
                plan = backend.Plan(1, shape, eps=self.nufft_eps, isign=-1, **options)
                plans[key] = plan
            if points.device.type == "cpu":
                plan.setpts(points[0].contiguous().numpy(), points[1].contiguous().numpy())
                return torch.from_numpy(plan.execute(values.contiguous().numpy()))
            plan.setpts(points[0], points[1])
            return plan.execute(values.contiguous())
        return pytorch_finufft.functional.finufft_type1(
            points, values, shape, isign=-1, modeord=0, eps=self.nufft_eps,
        )

    @torch.no_grad()
    def _prepare(self, x_shape, device, vis_data, cell_size=None,
                 primary_beam_list=None, primary_beam=None, pb_list=None,
                 grid_list=None, pb=None, grid_array=None, **_):
        device = self._device(device)
        x_shape = tuple(x_shape)
        if len(x_shape) != 3 or x_shape[0] != vis_data.data_I.shape[0]:
            raise ValueError("x must have shape (visibility channels, height, width).")
        if min(x_shape) < 1:
            raise ValueError("Image dimensions must be positive.")
        if cell_size is None or not np.isfinite(cell_size) or cell_size == 0:
            raise ValueError("cell_size must be finite and nonzero.")
        beams, grids = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )
        if len(beams) != len(vis_data.centers) or len(grids) != len(beams):
            raise ValueError("One primary beam and grid are required per pointing.")
        owners = tuple(np.asarray(a) for a in (
            vis_data.data_I, vis_data.sigma_I, vis_data.flag_I,
            vis_data.uu, vis_data.vv, vis_data.frequency, vis_data.nvis,
            *beams, *grids,
        ))
        key = (tuple(_identity(a) for a in owners),
               np.asarray(vis_data.frequency).tobytes(),
               np.asarray(vis_data.nvis).tobytes(),
               x_shape, str(device), float(cell_size), self.conj_data,
               self.nufft_eps, self.gram_dtype, self.reprojection_batch_size,
               self.use_plan_cache, self.gpu_cache_bytes, self.padding_fraction,
               self.local_gradients)
        if self._gram_cache is not None and self._gram_cache.key == key:
            self._check_budget(self._gram_cache.nbytes)
            return self._gram_cache
        self.clear_speed_cache()

        itemsize = torch.empty((), dtype=self.gram_dtype).element_size()
        gpu_budget = self._gpu_budget(device, x_shape, beams)
        pointing_gpu_bytes = 0
        nbytes = int(np.prod(x_shape)) * itemsize + 8
        self._check_budget(nbytes)
        rhs = torch.zeros(x_shape, dtype=self.gram_dtype, device=device)
        constant = torch.zeros((), dtype=torch.float64, device=device)

        def store(tensor):
            nonlocal pointing_gpu_bytes
            size = tensor.numel() * tensor.element_size()
            if device.type == "cuda":
                if pointing_gpu_bytes + size <= gpu_budget:
                    pointing_gpu_bytes += size
                    return tensor.to(device)
                # Blocking D2H completes before the GPU temporary is freed.
                # Pinned storage allows nonblocking H2D during evaluations.
                return tensor.to("cpu").pin_memory()
            return tensor
        geometry_by_beam: dict[int, _Geometry] = {}
        blocks = []
        setup_plans = {}
        complex_dtype = torch.complex64 if self.gram_dtype == torch.float32 else torch.complex128
        scale = float(cell_size) ** 2

        for c, b, data, sigma, uu, vv, _ww in vis_data.iter_chan_beam_I():
            if not (np.isfinite(data).all() and np.isfinite(sigma).all()
                    and (sigma > 0).all() and np.isfinite(uu).all()
                    and np.isfinite(vv).all()):
                raise ValueError("Unflagged visibilities and UVs must be finite; sigma must be positive.")
            geometry = geometry_by_beam.get(b)
            if geometry is None:
                beam_array, grid_array_b = np.asarray(beams[b]), np.asarray(grids[b])
                if (beam_array.ndim != 2 or min(beam_array.shape) < 1
                        or grid_array_b.shape != (1, *beam_array.shape, 2)):
                    raise ValueError("A beam must be (H,W) and its grid (1,H,W,2).")
                nbytes += (beam_array.size + grid_array_b.size) * itemsize
                self._check_budget(nbytes)
                # Localize on the compute device before applying the cache
                # budget. CUDA avoids repeatedly scanning/remapping billions
                # of grid entries on the CPU. Overflow is still pinned on the
                # host by store(); only one grid is transiently on CUDA here.
                grid_device = device if self.local_gradients else "cpu"
                grid_tensor = self._tensor(grid_array_b, grid_device, self.gram_dtype)
                # CPU execution remaps grids in place. Keep user arrays and
                # file-backed inputs unchanged when no device transfer made
                # an independent tensor for us.
                if self.local_gradients and grid_tensor.device.type == "cpu":
                    grid_tensor = grid_tensor.clone()
                bounds = (_footprint_bounds(grid_tensor, x_shape[-2:])
                          if self.local_gradients else _full_bounds(x_shape))
                if bounds[1] > bounds[0]:
                    _remap_grid_(grid_tensor, _full_bounds(x_shape), bounds)
                geometry = _Geometry(store(grid_tensor),
                                     store(self._tensor(beam_array, "cpu", self.gram_dtype)),
                                     bounds)
                del grid_tensor
                geometry_by_beam[b] = geometry
            height, width = geometry.beam.shape
            padded = _padded_shape((height, width), self.padding_fraction)
            nbytes += padded[0] * (padded[1] // 2 + 1) * itemsize
            self._check_budget(nbytes)

            # Match Classic3D's UV rounding before optionally promoting the
            # arithmetic precision of the Gram construction.
            _, u, v = uvw_to_radpix(uu, vv, cell_size, device)
            points = torch.stack((-v, u)).to(self.gram_dtype)
            weight = self._tensor(sigma, device, self.gram_dtype).square().reciprocal()
            values = self._tensor(data.conj() if self.conj_data else data,
                                  device, complex_dtype)
            kernel = self._type1(points, weight.to(complex_dtype),
                                 _lag_shape((height, width), padded), setup_plans).real * scale**2
            spectrum, padded = _embed_kernel(kernel, (height, width), padded)
            del kernel
            spectrum = store(spectrum)
            dirty = self._type1(points, weight * values, (height, width), setup_plans).real * scale
            geometry_device = geometry.to(device)
            rhs_tile = _crop(rhs[c], geometry.bounds)
            if rhs_tile.numel():
                rhs_tile.add_(_project_adjoint(dirty, geometry_device, rhs_tile.shape))
            del rhs_tile
            constant += 0.5 * torch.sum(
                weight.double() * (values.real.double().square() + values.imag.double().square())
            )
            blocks.append(_GramBlock(c, geometry, spectrum, padded))
            del dirty, values, weight, points, u, v, geometry_device

        setup_plans.clear()
        if device.type == "cuda":
            # Setup NUFFT workspaces are gone; release unused allocator
            # blocks once before the differently sized full-image FFTs.
            torch.cuda.empty_cache()
        groups = defaultdict(list)
        for block in blocks:
            if block.geometry.bounds[1] > block.geometry.bounds[0]:
                groups[(block.channel, tuple(block.geometry.beam.shape))].append(block)
        batches = [group[start:start + self.reprojection_batch_size]
                   for group in groups.values()
                   for start in range(0, len(group), self.reprojection_batch_size)]
        gpu_nbytes = (pointing_gpu_bytes + rhs.numel() * itemsize + 8
                      if device.type == "cuda" else 0)
        self._gram_cache = _GramCache(key, blocks, batches, rhs, constant,
                                      nbytes, gpu_nbytes, owners)
        if self.psf_positions is not None:
            self._use_point_sources(self._gram_cache, x_shape)
        if device.type == "cuda":
            logger.info(f"Gram cache: {gpu_nbytes / 2**30:.2f} GiB GPU, "
                        f"{(nbytes - gpu_nbytes) / 2**30:.2f} GiB host; "
                        f"{len(blocks)} channel/pointing blocks; "
                        f"local_gradients={self.local_gradients}; "
                        f"padding_fraction={self.padding_fraction:g}, "
                        f"FFT shapes={sorted({b.padded_shape for b in blocks})}")
        return self._gram_cache

    @torch.no_grad()
    def _use_point_sources(self, cache, x_shape):
        """Replace the dirty-image right-hand side by H*delta (see ``psf_positions``)."""
        _, height, width = x_shape
        delta = torch.zeros(x_shape, dtype=self.gram_dtype, device=cache.rhs.device)
        for row, col in self.psf_positions:
            if not (0 <= row < height and 0 <= col < width):
                raise ValueError(f"psf_positions entry {(row, col)} is outside the {height}x{width} image.")
            delta[:, row, col] = 1
        _, h_delta = self._visibility_terms(delta, cache, True, normal_only=True)
        cache.rhs = h_delta.to(cache.rhs.dtype)
        cache.constant = torch.zeros_like(cache.constant)
        logger.info(f"PSF mode: right-hand side replaced by H*delta at {len(self.psf_positions)} position(s); data values ignored.")

    def _visibility_terms(self, x, cache, compute_grad, *, normal_only=False):
        loss = cache.constant.new_zeros(()) if normal_only else (
            cache.constant - _dot64(x.to(self.gram_dtype), cache.rhs)
        )
        grad = torch.zeros_like(cache.rhs) if compute_grad else None
        source = x.detach().to(self.gram_dtype)
        for batch in cache.batches:
            block = batch[0]
            bounds = _batch_bounds(batch)
            with torch.set_grad_enabled(compute_grad):
                # The crop itself must be the leaf. A view of a global leaf
                # would still allocate a full-image gradient during backward.
                local = _crop(source[block.channel], bounds).detach().requires_grad_(compute_grad)
                if len(batch) == 1:
                    geometry = block.geometry.to(source.device)
                    projected = _project(local, geometry)
                    spectrum = block.spectrum.to(source.device, non_blocking=True)
                else:
                    grids = torch.cat([item.geometry.grid.to(source.device, non_blocking=True)
                                       for item in batch])
                    for index, item in enumerate(batch):
                        _remap_grid_(grids[index], item.geometry.bounds, bounds)
                    beams = torch.stack([item.geometry.beam.to(source.device, non_blocking=True)
                                         for item in batch])
                    images = local[None, None].expand(len(batch), -1, -1, -1)
                    projected = F.grid_sample(images, grids, mode="bilinear",
                                              padding_mode="zeros", align_corners=True)[:, 0] * beams
                    spectrum = torch.stack([item.spectrum.to(source.device, non_blocking=True)
                                            for item in batch])
            with torch.no_grad():
                applied = _apply_kernel(projected.detach(), block, spectrum)
                if not normal_only:
                    loss += 0.5 * _dot64(projected.detach(), applied)
            if compute_grad:
                contribution = torch.autograd.grad(projected, local, applied)[0]
                _crop(grad[block.channel], bounds).add_(contribution)
                del contribution
            if len(batch) > 1:
                del grids, beams, images
            else:
                del geometry
            del projected, applied, spectrum, local
        if compute_grad and not normal_only:
            grad.sub_(cache.rhs)
        return loss.detach(), grad

    @torch.no_grad()
    def _regularizer_terms(self, x, *, compute_grad, cell_size,
                           fftsd=None, fftbeam=None, tapper=None, lambda_sd=0.0,
                           fftkernel=None, normal_only=False, **_):
        use_sd = lambda_sd > 0 and fftsd is not None
        use_reg = self.lambda_r > 0 and fftkernel is not None
        loss = torch.zeros((), dtype=torch.float64, device=x.device)
        grad = torch.zeros_like(x) if compute_grad else None
        if not (use_sd or use_reg):
            return loss, grad
        if tapper is None or (use_sd and fftbeam is None):
            raise ValueError("Active regularizers require tapper and, for single dish, fftbeam.")
        taper_t = self._static(tapper, x.device)
        sd_t = self._static(fftsd, x.device) if use_sd else None
        beam_t = self._static(fftbeam, x.device) if use_sd else None
        kernel_t = self._static(fftkernel, x.device) if use_reg else None

        def channel(tensor, c):
            return tensor[c] if tensor.ndim == x.ndim else tensor

        scale = float(cell_size) ** 2
        for c in range(x.shape[0]):
            taper = channel(taper_t, c)
            transformed = torch.fft.fft2(x[c] * taper)
            frequency_grad = None
            if use_sd:
                beam = channel(beam_t, c)
                target = channel(sd_t, c)
                residual = scale * transformed * beam
                if not normal_only:
                    # Subtract components separately: complex subtraction
                    # can spread a NaN into the other, valid component.
                    residual.real.sub_(target.real)
                    residual.imag.sub_(target.imag)
                # Missing real/imaginary SD samples have independent masks,
                # also when applying the Hessian without the target offset.
                residual.real.masked_fill_(torch.isnan(target.real), 0.0)
                residual.imag.masked_fill_(torch.isnan(target.imag), 0.0)
                loss += 0.5 * lambda_sd * _norm_squared64(residual)
                if compute_grad:
                    frequency_grad = residual.mul_(beam.conj()).mul_(scale * lambda_sd)
                del residual
            if use_reg:
                kernel = channel(kernel_t, c)
                residual = scale * transformed * kernel
                loss += 0.5 * self.lambda_r * _norm_squared64(residual)
                if compute_grad:
                    residual.mul_(kernel.conj()).mul_(scale * self.lambda_r)
                    if frequency_grad is None:
                        frequency_grad = residual
                    else:
                        frequency_grad = frequency_grad + residual
                del residual
            del transformed
            if compute_grad:
                # The adjoint of an unnormalised forward FFT is the inverse
                # with norm="forward" (i.e. N * the default inverse FFT).
                # Apply the two quadratic gradients without an autograd graph.
                backward = torch.fft.ifft2(frequency_grad, norm="forward")
                backward.mul_(taper.conj())
                grad[c].copy_(backward.real)
                del backward, frequency_grad
        return loss, grad

    def objective(self, x, vis_data, device, compute_grad=True, **params):
        device = self._device(device)
        if x.device != device or x.dtype not in (torch.float32, torch.float64):
            raise ValueError("x must be float32/float64 on the requested cost device.")
        if compute_grad and not x.is_leaf:
            raise ValueError("objective() requires a leaf tensor when computing x.grad.")
        cache = self._prepare(tuple(x.shape), device, vis_data, **params)
        # Only the local reprojection uses autograd. Quadratic convolution
        # and regularization gradients are applied analytically.
        with torch.no_grad():
            loss, grad = self._visibility_terms(x, cache, compute_grad)
            reg_loss, reg_grad = self._regularizer_terms(x, compute_grad=compute_grad, **params)
            if compute_grad:
                x.grad = grad.to(x.dtype).add_(reg_grad)
            return loss + reg_loss

    def loss(self, x, shape, device, vis_data, **params):
        # Evaluate the full cube once: the parent channel loop would rebuild
        # a single-observation Gram cache for every channel on every call.
        tensor = torch.as_tensor(x.reshape(shape), device=device)
        value = self.objective(tensor, vis_data, device, **params)
        return float(value), tensor.grad.detach().cpu().numpy().astype(x.dtype).ravel()

    @torch.no_grad()
    def apply_normal_operator(self, x, vis_data, device, **params):
        """Apply the full real Hessian, including SD and spatial penalties."""
        cache = self._prepare(tuple(x.shape), device, vis_data, **params)
        _, result = self._visibility_terms(x, cache, True, normal_only=True)
        _, regularizer = self._regularizer_terms(x, compute_grad=True,
                                                normal_only=True, **params)
        return result.to(x.dtype) + regularizer

    @torch.no_grad()
    def quadratic_rhs(self, x_shape, vis_data, device, **params):
        """Return the full normal-equation RHS for the existing CG solver."""
        cache = self._prepare(tuple(x_shape), device, vis_data, **params)
        zero = torch.zeros(x_shape, dtype=self.gram_dtype, device=cache.rhs.device)
        _, regularizer_gradient = self._regularizer_terms(zero, compute_grad=True, **params)
        return cache.rhs - regularizer_gradient

    @torch.no_grad()
    def visibility_loss(self, x, vis_data, device, **params):
        """Direct visibility loss for validation, excluding regularizers.

        This deliberately executes type-2 NUFFTs and is not used by FISTA.
        It follows this model's Gram precision and NUFFT tolerance.
        """
        cache = self._prepare(tuple(x.shape), device, vis_data, **params)
        blocks = iter(cache.blocks)
        result = torch.zeros((), dtype=torch.float64, device=cache.rhs.device)
        scale = float(params["cell_size"]) ** 2
        for c, _b, data, sigma, uu, vv, _ww in vis_data.iter_chan_beam_I():
            block = next(blocks)
            geometry = block.geometry.to(cache.rhs.device)
            projected = _project_global(x[c].to(self.gram_dtype), geometry)
            _, u, v = uvw_to_radpix(uu, vv, params["cell_size"], cache.rhs.device)
            points = torch.stack((-v, u)).to(self.gram_dtype)
            complex_dtype = torch.complex64 if self.gram_dtype == torch.float32 else torch.complex128
            predicted = scale * pytorch_finufft.functional.finufft_type2(
                points, projected.to(complex_dtype), isign=1, modeord=0, eps=self.nufft_eps,
            )
            observed = self._tensor(data.conj() if self.conj_data else data,
                                    cache.rhs.device, complex_dtype)
            residual = (predicted - observed) / self._tensor(sigma, cache.rhs.device, self.gram_dtype)
            result += 0.5 * (_dot64(residual.real, residual.real)
                            + _dot64(residual.imag, residual.imag))
            del geometry, projected, points, u, v, predicted, observed, residual
        return result
