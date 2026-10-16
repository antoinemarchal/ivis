"""Local footprints must preserve the forward/adjoint pair and the solver."""
import numpy as np
import pytest
import torch

from ivis.models.classic3D_gram import (
    Classic3D_gram, _Geometry, _crop, _footprint_bounds, _full_bounds,
    _project, _project_adjoint, _project_global, _remap_grid_,
)
from ivis.optim.solvers import optimize_torch_fista
from .test_classic3d_gram import case


DEVICES = ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA'))]


def pixel_grid(shape, beam_shape, x_start, y_start):
    yy, xx = np.meshgrid(np.arange(beam_shape[0]), np.arange(beam_shape[1]), indexing='ij')
    xx = x_start + 0.8 * xx + 0.07 * yy
    yy = y_start + 0.9 * yy - 0.03 * xx
    # Degenerate original axes retain their original normalized coordinates.
    gx = 2 * xx / (shape[1] - 1) - 1 if shape[1] > 1 else xx
    gy = 2 * yy / (shape[0] - 1) - 1 if shape[0] > 1 else yy
    return np.stack((gx, gy), axis=-1)[None].astype('float32')


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape,origin', [
    ((61, 73), (12.3, 14.7)), ((61, 73), (-2.4, -1.8)),
    ((61, 73), (71.7, 59.4)), ((61, 73), (100, 100)),
    ((61, 73), (-0.9, -0.9)), ((1, 73), (10.2, 12.3)),
    ((61, 1), (12.3, 10.2)), ((1, 7714), (6200.2, 0)),
    ((7714, 1), (0, 6300.7)),
])
def test_local_projection_and_adjoint(shape, origin, dtype, device):
    grid = torch.tensor(pixel_grid(shape, (5, 6), *origin), dtype=dtype, device=device)
    bounds = _footprint_bounds(grid, shape)
    assert bounds != _full_bounds(shape)
    remapped = grid.clone()
    if bounds[1] > bounds[0]:
        _remap_grid_(remapped, _full_bounds(shape), bounds)
    torch.manual_seed(20)
    beam = torch.rand((5, 6), dtype=dtype, device=device)
    original = _Geometry(grid.to(device), beam, _full_bounds(shape))
    cropped = _Geometry(remapped.to(device), beam, bounds)
    image = torch.randn(shape, dtype=dtype, device=device)
    values = torch.randn((5, 6), dtype=dtype, device=device)
    tol = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(_project_global(image, cropped), _project(image, original),
                               rtol=tol, atol=tol)
    expected = _project_adjoint(values, original, shape)
    actual = torch.zeros_like(image)
    tile = _crop(actual, bounds)
    if tile.numel():
        tile.add_(_project_adjoint(values, cropped, tile.shape))
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    # Adjoint identity within the cropped operator, including beam weighting.
    torch.testing.assert_close((_project_global(image, cropped) * values).sum(),
                               (image * actual).sum(), rtol=tol, atol=tol)


def local_case():
    x, params = case(shape=(48, 64), beam_shapes=((5, 6), (5, 6), (4, 5)))
    params['grid_list'] = [pixel_grid(x.shape[-2:], b.shape, *origin)
                           for b, origin in zip(params['pb_list'],
                                                [(9.2, 6.3), (11.7, 8.4), (-1.7, 25.2)])]
    return x, params


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('padding', [1.2, 2.0])
@pytest.mark.parametrize('batch', [1, 2])
def test_local_objective_rhs_normal_and_streaming(device, dtype, padding, batch):
    x, params = local_case()
    opts = dict(lambda_r=0.02, gram_dtype=dtype, padding_fraction=padding,
                nufft_eps=1e-6 if dtype == torch.float32 else 1e-12,
                reprojection_batch_size=batch, gpu_cache_bytes=0)
    full = Classic3D_gram(**opts, local_gradients=False)
    local = Classic3D_gram(**opts, local_gradients=True)
    tol = 3e-5 if dtype == torch.float32 else 2e-11

    def evaluate(model, array=x, grad=True):
        tensor = torch.tensor(array, device=device, dtype=dtype, requires_grad=grad)
        loss = model.objective(tensor, device=device, compute_grad=grad, **params)
        return loss, tensor.grad

    f0, g0 = evaluate(full)
    f1, g1 = evaluate(local)
    torch.testing.assert_close(f1, f0, rtol=tol, atol=tol)
    torch.testing.assert_close(g1, g0, rtol=tol, atol=tol)
    torch.testing.assert_close(local._gram_cache.rhs, full._gram_cache.rhs, rtol=tol, atol=tol)
    assert local.cache_nbytes == full.cache_nbytes
    assert local.cache_gpu_nbytes == full.cache_gpu_nbytes
    assert all(b.geometry.bounds != _full_bounds(x.shape) for b in local._gram_cache.blocks)
    tensor = torch.tensor(x, dtype=dtype, device=device)
    rhs = local.quadratic_rhs(x.shape, device=device, **params)
    normal = local.apply_normal_operator(tensor, device=device, **params)
    torch.testing.assert_close(normal - rhs, g1, rtol=tol, atol=tol)
    torch.testing.assert_close(evaluate(local, grad=False)[0], f1, rtol=tol, atol=tol)
    torch.testing.assert_close(local.visibility_loss(tensor, device=device, **params),
                               full.visibility_loss(tensor, device=device, **params),
                               rtol=tol, atol=tol)
    if dtype == torch.float64:
        direction = torch.flip(tensor, dims=(1,))
        hy = local.apply_normal_operator(direction, device=device, **params)
        torch.testing.assert_close((tensor * hy).sum(), (direction * normal).sum(),
                                   rtol=tol, atol=tol)
        h = 1e-4
        fp = evaluate(local, (tensor + h * direction).cpu().numpy(), False)[0]
        fm = evaluate(local, (tensor - h * direction).cpu().numpy(), False)[0]
        torch.testing.assert_close((fp - fm) / (2 * h), (g1 * direction).sum(),
                                   rtol=1e-7, atol=1e-7)


def test_all_footprints_outside_and_option_invalidation():
    x, params = local_case()
    params['grid_list'] = [np.full_like(grid, 3) for grid in params['grid_list']]
    params['lambda_sd'] = 0
    model = Classic3D_gram(lambda_r=0, local_gradients=True)
    tensor = torch.tensor(x, requires_grad=True)
    f0 = model.objective(tensor, device='cpu', **params)
    assert not tensor.grad.any()
    original = model._gram_cache
    assert not original.batches
    model.local_gradients = False
    f1 = model.objective(tensor, device='cpu', **params)
    assert model._gram_cache is not original
    assert not tensor.grad.any()
    torch.testing.assert_close(f1, f0)


def test_backward_uses_crop_leaves(monkeypatch):
    # A correct numeric answer alone would miss slicing a global autograd
    # leaf, which recreates the full-image temporary we intend to eliminate.
    x, params = local_case()
    original_grad = torch.autograd.grad
    shapes = []

    def record_grad(outputs, inputs, *args, **kwargs):
        shapes.append(tuple(inputs.shape))
        assert inputs.is_leaf
        return original_grad(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, 'grad', record_grad)
    model = Classic3D_gram(local_gradients=True, reprojection_batch_size=2)
    model.objective(torch.tensor(x, requires_grad=True), device='cpu', **params)
    assert shapes
    assert all(np.prod(shape) < np.prod(x.shape[-2:]) // 4 for shape in shapes)


@pytest.mark.parametrize('device', DEVICES)
def test_grid_preparation_uses_compute_device_even_when_streamed(device, monkeypatch):
    import ivis.models.classic3D_gram as module

    x, params = local_case()
    original_bounds, original_remap = module._footprint_bounds, module._remap_grid_
    calls = []

    def bounds(grid, *args):
        assert grid.device == torch.device(device)
        calls.append('bounds')
        return original_bounds(grid, *args)

    def remap(grid, *args):
        assert grid.device == torch.device(device)
        calls.append('remap')
        return original_remap(grid, *args)

    monkeypatch.setattr(module, '_footprint_bounds', bounds)
    monkeypatch.setattr(module, '_remap_grid_', remap)
    model = Classic3D_gram(local_gradients=True, gpu_cache_bytes=0)
    model._prepare(x.shape, device=device, **params)
    assert calls.count('bounds') == len(params['grid_list'])
    assert calls.count('remap') == len(params['grid_list'])
    assert all(b.geometry.grid.device.type == 'cpu' for b in model._gram_cache.blocks)
    if device.startswith('cuda'):
        assert all(b.geometry.grid.is_pinned() for b in model._gram_cache.blocks)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_local_fista_matches_full(device, dtype):
    x, params = local_case()
    results = []
    for local in (False, True):
        model = Classic3D_gram(lambda_r=0.02, gram_dtype=dtype, local_gradients=local,
                               padding_fraction=1.2, reprojection_batch_size=2,
                               nufft_eps=1e-6 if dtype == torch.float32 else 1e-12)
        results.append(optimize_torch_fista(
            model=model, x_init=np.zeros_like(x), dtype=dtype, max_its=20,
            cost_dev=torch.device(device), optim_dev=torch.device(device), params=params,
            positivity=True, initial_step=1e-3, loss_only_line_search=True))
    tolerance = 2e-4 if dtype == torch.float32 else 1e-9
    assert np.linalg.norm(results[0] - results[1]) / np.linalg.norm(results[0]) < tolerance
