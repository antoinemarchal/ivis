import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ivis.types import VisIData
from ivis.models.classic3D_gram import Classic3D_gram
from ivis.models.classic3D_optimized import Classic3D_optimized
from ivis.models.operators.geometry import uvw_to_radpix
from ivis.optim.solvers import optimize_torch_fista, optimize_torch_cg

torch.set_num_threads(1)


def case(shape=(6, 7), channels=2, beam_shapes=((4, 5), (4, 5), (5, 4))):
    rng = np.random.default_rng(42)
    nb, nv = len(beam_shapes), 11
    counts = np.arange(nv, nv-nb, -1)
    flags = rng.uniform(size=(channels, nb, nv)) < 0.2
    if channels > 1 and nb > 1:
        flags[1, 1] = True
    vis = VisIData(
        frequency=np.linspace(1.4e9, 1.6e9, channels),
        velocity=np.arange(channels), centers=np.array([None]*nb, dtype=object),
        nvis=counts,
        uu=rng.uniform(-12000, 12000, (nb, nv)),
        vv=rng.uniform(-12000, 12000, (nb, nv)), ww=np.zeros((nb, nv)),
        data_I=(rng.normal(size=(channels, nb, nv)) + 1j*rng.normal(size=(channels, nb, nv))).astype('complex64'),
        sigma_I=rng.uniform(0.5, 2.0, size=(channels, nb, nv)).astype('float32'), flag_I=flags,
    )
    beams, grids = [], []
    for h, w in beam_shapes:
        yy, xx = np.meshgrid(np.linspace(-1.15, 1.1, h), np.linspace(-1.1, 1.2, w), indexing='ij')
        grids.append(np.stack((xx + 0.08*yy, yy - 0.06*xx), axis=-1).astype('float32')[None])
        beams.append(rng.uniform(0.2, 1.1, (h, w)).astype('float32'))
    x = rng.normal(scale=0.1, size=(channels, *shape)).astype('float32')
    params = dict(vis_data=vis, pb_list=beams, grid_list=grids, cell_size=1.9,
                  tapper=rng.uniform(0.2, 1, shape).astype('float32'),
                  fftsd=(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape)).astype('complex64'),
                  fftbeam=rng.uniform(0.1, 1, x.shape).astype('float32'),
                  fftkernel=rng.uniform(0.1, 1, shape).astype('float32'), lambda_sd=0.03)
    return x, params


def evaluate(model, x, params, dtype=torch.float32):
    t = torch.tensor(x, dtype=dtype, requires_grad=True)
    value = model.objective(t, device='cpu', **params)
    return value.detach(), t.grad.detach()


@pytest.mark.parametrize('conj', [False, True])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('regularizers', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_matches_optimized(conj, batch, regularizers, dtype):
    x, params = case()
    if not regularizers:
        params['lambda_sd'] = 0
    kwargs = dict(lambda_r=0.02 if regularizers else 0, conj_data=conj,
                  reprojection_batch_size=batch, nufft_eps=1e-6)
    old = Classic3D_optimized(**kwargs)
    new = Classic3D_gram(**kwargs)
    f0, g0 = evaluate(old, x, params, dtype)
    f1, g1 = evaluate(new, x, params, dtype)
    torch.testing.assert_close(f1, f0.double(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(g1, g0, rtol=3e-5, atol=3e-5)
    leaf = torch.tensor(x, dtype=dtype, requires_grad=True)
    sentinel = torch.full_like(leaf, 123)
    leaf.grad = sentinel
    with torch.no_grad():
        loss_only = new.objective(leaf, device='cpu', compute_grad=False, **params)
    torch.testing.assert_close(loss_only, f1)
    assert leaf.grad is sentinel
    assert not loss_only.requires_grad


@pytest.mark.parametrize('shape', [(4, 5), (5, 4), (5, 5), (4, 4), (1, 3)])
@pytest.mark.parametrize('use_plan_cache', [False, True])
def test_exact_dft_and_boundary_impulses(shape, use_plan_cache):
    x, params = case(shape=shape, channels=1, beam_shapes=(shape,))
    params['lambda_sd'] = 0
    model = Classic3D_gram(lambda_r=0, nufft_eps=1e-12,
                           gram_dtype=torch.float64, use_plan_cache=use_plan_cache)
    for image in [x, np.zeros_like(x)]:
        if not image.any():
            image[0, -1, -1] = 1
        tensor = torch.tensor(image, dtype=torch.float64, requires_grad=True)
        c, b, data, sigma, uu, vv, _ = next(params['vis_data'].iter_chan_beam_I())
        _, u, v = uvw_to_radpix(uu, vv, params['cell_size'], 'cpu')
        points = torch.stack((-v, u)).double()
        yy, xx = torch.meshgrid(torch.arange(shape[0])-shape[0]//2,
                                torch.arange(shape[1])-shape[1]//2, indexing='ij')
        modes = torch.stack((yy.reshape(-1), xx.reshape(-1))).double()
        operator = params['cell_size']**2 * torch.exp(1j*(points.T @ modes))
        projected = F.grid_sample(tensor[:, None], torch.tensor(params['grid_list'][0]).double(), align_corners=True)[0,0]
        projected = projected * torch.tensor(params['pb_list'][0]).double()
        residual = (operator @ projected.flatten().to(torch.complex128) - torch.tensor(data.conj().copy()).to(torch.complex128)) / torch.tensor(sigma).double()
        exact = 0.5*(residual.abs().square()).sum()
        exact.backward()
        f, g = evaluate(model, image, params, torch.float64)
        torch.testing.assert_close(f, exact.detach(), rtol=2e-11, atol=2e-11)
        torch.testing.assert_close(g, tensor.grad, rtol=2e-10, atol=2e-10)
        direct = model.visibility_loss(torch.tensor(image).double(), device='cpu', **params)
        torch.testing.assert_close(f, direct, rtol=2e-11, atol=2e-11)


def test_normal_rhs_symmetry_and_gradient():
    x, params = case()
    model = Classic3D_gram(lambda_r=0.02, nufft_eps=1e-12, gram_dtype=torch.float64, reprojection_batch_size=2)
    tensor = torch.tensor(x).double()
    f, gradient = evaluate(model, x, params, torch.float64)
    rhs = model.quadratic_rhs(x_shape=x.shape, device='cpu', **params)
    Hx = model.apply_normal_operator(tensor, device='cpu', **params)
    torch.testing.assert_close(Hx-rhs, gradient, rtol=1e-12, atol=1e-12)
    other = torch.flip(tensor, dims=(1,))
    Hy = model.apply_normal_operator(other, device='cpu', **params)
    torch.testing.assert_close((tensor*Hy).sum(), (other*Hx).sum(), rtol=1e-12, atol=1e-12)
    assert (tensor*Hx).sum() >= 0
    h = 1e-5
    fp, _ = evaluate(model, x.astype('float64')+h*other.numpy(), params, torch.float64)
    fm, _ = evaluate(model, x.astype('float64')-h*other.numpy(), params, torch.float64)
    torch.testing.assert_close((fp-fm)/(2*h), (gradient*other).sum(), rtol=1e-8, atol=1e-8)


def test_cache_reuse_and_budget(monkeypatch):
    x, params = case()
    model = Classic3D_gram(lambda_r=0.02)
    evaluate(model, x, params)
    original = model._gram_cache
    size = model.cache_nbytes
    assert size > 0
    def fail(*args, **kwargs):
        raise AssertionError('NUFFT repeated during cached evaluation')
    monkeypatch.setattr(model, '_type1', fail)
    evaluate(model, x*0.9, params)
    assert model._gram_cache is original
    model.clear_speed_cache()
    assert model.cache_nbytes == 0
    model = Classic3D_gram(max_cache_bytes=1)
    with pytest.raises(MemoryError):
        evaluate(model, x, params)
    assert model.cache_nbytes == 0
    model = Classic3D_gram(max_cache_bytes=size)
    evaluate(model, x, params)
    assert model.cache_nbytes == size


def test_geometry_metadata_invalidation():
    x, params = case()
    model = Classic3D_gram()
    evaluate(model, x, params)
    first = model._gram_cache
    params['vis_data'].frequency[0] *= 1.01
    evaluate(model, x, params)
    assert model._gram_cache is not first
    second = model._gram_cache
    params['vis_data'].nvis[0] -= 1
    evaluate(model, x, params)
    assert model._gram_cache is not second


def test_empty_vis_and_masked_sd():
    x, params = case()
    params['vis_data'].flag_I[:] = True
    params['fftsd'][0, 0, 0] = complex(float('nan'), 1)
    model = Classic3D_gram(lambda_r=0.02)
    f, g = evaluate(model, x, params)
    assert torch.isfinite(f) and torch.isfinite(g).all()
    normal = model.apply_normal_operator(torch.tensor(x), device='cpu', **params)
    rhs = model.quadratic_rhs(x_shape=x.shape, device='cpu', **params)
    torch.testing.assert_close(normal-rhs, g, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('positivity', [False, True])
@pytest.mark.parametrize('loss_only', [False, True])
@pytest.mark.parametrize('quadratic_reuse', [None, False])
def test_fista(positivity, loss_only, quadratic_reuse):
    x, params = case(shape=(4, 5), channels=1, beam_shapes=((4,5), (4,5)))
    init = np.zeros_like(x)
    model = Classic3D_gram(lambda_r=0.02, gram_dtype=torch.float64, nufft_eps=1e-10)
    initial_f, _ = evaluate(model, init, params, torch.float64)
    result = optimize_torch_fista(model=model, x_init=init, dtype=torch.float64,
        max_its=30, cost_dev=torch.device('cpu'), optim_dev=torch.device('cpu'),
        params=params, positivity=positivity, initial_step=1e-3,
        loss_only_line_search=loss_only, quadratic_gradient_reuse=quadratic_reuse)
    final_f, _ = evaluate(model, result, params, torch.float64)
    assert final_f < initial_f
    assert np.isfinite(result).all()
    if positivity:
        assert (result >= 0).all()


@pytest.mark.parametrize('padding_fraction', [1.2, 2.0])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('positivity', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires a CUDA GPU'))])
def test_fista_quadratic_reuse_matches_original(padding_fraction, dtype, positivity, device):
    x, params = case(shape=(4, 5), channels=1, beam_shapes=((4, 5), (4, 5)))
    # Exercise the fixed single-dish masks as well as spatial regularization.
    params['fftsd'][0, 0, 0] = complex(float('nan'), 1)
    results = []
    model = Classic3D_gram(lambda_r=0.02, gram_dtype=dtype,
                           nufft_eps=1e-6 if dtype == torch.float32 else 1e-12,
                           padding_fraction=padding_fraction, gpu_cache_bytes=0)
    for reuse in (False, None):
        # Compare the two solver paths with exactly the same prepared operator.
        results.append(optimize_torch_fista(
            model=model, x_init=np.zeros_like(x), dtype=dtype, max_its=20,
            cost_dev=torch.device(device), optim_dev=torch.device(device),
            params=params, positivity=positivity, initial_step=1e-3,
            loss_only_line_search=True, quadratic_gradient_reuse=reuse))
    if dtype == torch.float64:
        np.testing.assert_allclose(results[1], results[0], rtol=1e-10, atol=1e-10)
    else:
        # Float32 roundoff changes near-convergence backtracking decisions, so
        # trajectories need not be bit-identical. Require <0.2% image error
        # together with agreement in the same high-precision objective.
        relative_difference = np.linalg.norm(results[1] - results[0]) / np.linalg.norm(results[0])
        assert relative_difference < 2e-3
        reference = Classic3D_gram(lambda_r=0.02, gram_dtype=torch.float64,
                                   nufft_eps=1e-12, padding_fraction=padding_fraction)
        losses = [evaluate(reference, result, params, torch.float64)[0] for result in results]
        torch.testing.assert_close(losses[1], losses[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('reuse', [None, False])
def test_imager_fista_quadratic_reuse_option(reuse, monkeypatch):
    from astropy import units as u
    from radio_beam import Beam
    from ivis.imager import Imager3D

    # The imager's 2% apodization needs at least two pixels at each edge.
    x, params = case(shape=(64, 64), channels=1, beam_shapes=((4, 5), (4, 5)))
    imager = Imager3D(
        params['vis_data'], np.asarray(params['pb_list']), np.asarray(params['grid_list']),
        np.zeros(x.shape[-2:], dtype=np.float32), Beam(1 * u.arcsec),
        dict(NAXIS1=64, NAXIS2=64, CDELT2=params['cell_size'] / 3600),
        init_params=np.zeros_like(x), max_its=2, lambda_sd=0.0, positivity=True,
        cost_device='cpu', optim_device='cpu')
    model = Classic3D_gram(lambda_r=0.02)
    original_objective, calls = model.objective, []

    def objective(*args, **kwargs):
        calls.append(kwargs.get('compute_grad', True))
        return original_objective(*args, **kwargs)

    monkeypatch.setattr(model, 'objective', objective)
    options = {} if reuse is None else dict(quadratic_gradient_reuse=reuse)
    result = imager.process(model=model, solver='FISTA', initial_step=1e-5,
                            loss_only_line_search=True, **options)
    assert result.shape == x.shape and np.isfinite(result).all()
    assert len(calls) == (3 if reuse is None else 5)


def test_cg_against_dense_solution():
    x, params = case(shape=(3,4), channels=1, beam_shapes=((3,4), (3,4)))
    model = Classic3D_gram(lambda_r=0.1, gram_dtype=torch.float64, nufft_eps=1e-12)
    rhs = model.quadratic_rhs(x_shape=x.shape, device='cpu', **params).flatten()
    basis = torch.eye(x.size, dtype=torch.float64)
    H = torch.stack([model.apply_normal_operator(v.reshape(x.shape), device='cpu', **params).flatten() for v in basis], dim=1)
    expected = torch.linalg.solve(H, rhs).numpy().reshape(x.shape)
    result = optimize_torch_cg(model=model, x_init=np.zeros_like(x), dtype=torch.float64,
        max_its=100, cost_dev=torch.device('cpu'), optim_dev=torch.device('cpu'),
        params=params, tol=1e-10)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_numpy_loss_wrapper_and_validation():
    x, params = case()
    model = Classic3D_gram(lambda_r=0.02)
    direct_f, direct_g = evaluate(model, x, params, torch.float64)
    f, g = model.loss(x.astype('float64').ravel(), shape=x.shape, device='cpu', **params)
    np.testing.assert_allclose(f, direct_f.numpy())
    np.testing.assert_allclose(g.reshape(x.shape), direct_g.numpy())
    params['vis_data'].sigma_I[:] = 0
    model.clear_speed_cache()
    with pytest.raises(ValueError, match='sigma'):
        evaluate(model, x, params)


def test_setup_plan_reuse(monkeypatch):
    import ivis.models.classic3D_gram as module
    x, params = case()
    real_plan = module.finufft.Plan
    calls = []
    def plan(kind, shape, **kwargs):
        calls.append((kind, tuple(shape)))
        return real_plan(kind, shape, **kwargs)
    monkeypatch.setattr(module.finufft, 'Plan', plan)
    model = Classic3D_gram()
    evaluate(model, x, params)
    assert len(calls) == 4  # two distinct image shapes and their lag grids
    assert len(set(calls)) == 4
    assert all(kind == 1 for kind, shape in calls)
    evaluate(model, x*1.01, params)
    assert len(calls) == 4


@pytest.mark.parametrize('positivity', [False, True])
def test_float32_fista_against_optimized(positivity):
    x, params = case(shape=(4,5), channels=1, beam_shapes=((4,5), (4,5)))
    init = np.zeros_like(x)
    kwargs = dict(lambda_r=0.02, nufft_eps=1e-6, reprojection_batch_size=2)
    old, new = Classic3D_optimized(**kwargs), Classic3D_gram(**kwargs)
    results = []
    for model in (old, new):
        results.append(optimize_torch_fista(model=model, x_init=init, dtype=torch.float32,
            max_its=50, cost_dev=torch.device('cpu'), optim_dev=torch.device('cpu'),
            params=params, positivity=positivity, initial_step=1e-3,
            loss_only_line_search=True))
    f0, _ = evaluate(old, results[0], params)
    f1, _ = evaluate(old, results[1], params)
    torch.testing.assert_close(f1, f0, rtol=2e-3, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize('budget', [0, 600])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('padding_fraction', [1.5, 2.0])
def test_streamed_cuda_matches_resident_and_visibility(budget, batch, padding_fraction, monkeypatch):
    # Nonuniform sigma, flagged data, multiple channels and nontrivial grids
    # exercise the weighted objective and its transpose through both caches.
    x, params = case()
    opts = dict(lambda_r=0.02, nufft_eps=1e-6, reprojection_batch_size=batch)
    resident = Classic3D_gram(**opts, gpu_cache_bytes=1 << 20, padding_fraction=padding_fraction)
    streamed = Classic3D_gram(**opts, gpu_cache_bytes=budget, padding_fraction=padding_fraction)
    direct = Classic3D_optimized(**opts)

    def cuda_eval(model, grad=True):
        tensor = torch.tensor(x, device='cuda:0', requires_grad=grad)
        with torch.set_grad_enabled(grad):
            loss = model.objective(tensor, device='cuda:0', compute_grad=grad, **params)
        return loss.detach(), tensor.grad

    f0, g0 = cuda_eval(resident)
    f1, g1 = cuda_eval(streamed)
    f2, g2 = cuda_eval(direct)
    torch.testing.assert_close(f1, f0, rtol=2e-6, atol=1e-5)
    torch.testing.assert_close(g1, g0, rtol=3e-5, atol=3e-5)
    if padding_fraction == 2.0:
        torch.testing.assert_close(f1, f2.double(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(g1, g2, rtol=3e-5, atol=3e-5)
    assert streamed.cache_nbytes == resident.cache_nbytes
    fixed_bytes = x.size * 4 + 8
    assert streamed.cache_gpu_nbytes <= fixed_bytes + budget
    tensors = [t for b in streamed._gram_cache.blocks
               for t in (b.geometry.grid, b.geometry.beam, b.spectrum)]
    host = [t for t in tensors if t.device.type == 'cpu']
    assert host and all(t.is_pinned() for t in host)
    assert any(t.device.type == 'cuda' for t in tensors) == (budget > 0)

    def fail(*args, **kwargs):
        raise AssertionError('NUFFT repeated after preparation')

    monkeypatch.setattr(streamed, '_type1', fail)
    torch.testing.assert_close(cuda_eval(streamed, False)[0], f1)
    tensor = torch.tensor(x, device='cuda:0')
    normal = streamed.apply_normal_operator(tensor, device='cuda:0', **params)
    rhs = streamed.quadratic_rhs(x.shape, device='cuda:0', **params)
    torch.testing.assert_close(normal - rhs, g1, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(
        streamed.visibility_loss(tensor, device='cuda:0', **params),
        resident.visibility_loss(tensor, device='cuda:0', **params),
    )
    streamed.clear_speed_cache()
    assert streamed.cache_gpu_nbytes == streamed.cache_nbytes == 0


def test_gpu_cache_budget_validation():
    with pytest.raises(ValueError, match='gpu_cache_bytes'):
        Classic3D_gram(gpu_cache_bytes=-1)


@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sd_weight,reg_weight', [(0.0, 0.02), (0.03, 0.0), (0.03, 0.02)])
def test_analytic_regularizer_matches_autograd(device, dtype, sd_weight, reg_weight):
    x, params = case()
    rng = np.random.default_rng(35)
    # Complex, channel-dependent coefficients also check conjugation in the
    # adjoint; an identity FFT kernel alone would miss that error.
    params['fftkernel'] = (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)).astype('complex64')
    params['fftbeam'] = params['fftbeam'] * (0.7 + 0.2j)
    params['fftsd'][0, 1, 1] = complex(float('nan'), 1.0)
    params['fftsd'][1, 2, 1] = complex(1.0, float('nan'))
    params['lambda_sd'] = sd_weight
    params['tapper'] = np.broadcast_to(params['tapper'], x.shape).copy()
    params['tapper'][1] *= 0.5
    tensor = torch.tensor(x, device=device, dtype=dtype, requires_grad=True)
    taper = torch.as_tensor(params['tapper'], device=device)
    transformed = torch.fft.fft2(tensor * taper)
    exact = tensor.new_zeros(())
    if sd_weight:
        target = torch.as_tensor(params['fftsd'], device=device)
        predicted = params['cell_size']**2 * transformed * torch.as_tensor(params['fftbeam'], device=device)
        real = (predicted.real - target.real).masked_fill(torch.isnan(target.real), 0.0)
        imag = (predicted.imag - target.imag).masked_fill(torch.isnan(target.imag), 0.0)
        exact = exact + 0.5 * sd_weight * (real.square().sum() + imag.square().sum())
    if reg_weight:
        conv = params['cell_size']**2 * transformed * torch.as_tensor(params['fftkernel'], device=device)
        exact = exact + 0.5 * reg_weight * conv.abs().square().sum()
    expected_grad = torch.autograd.grad(exact, tensor)[0]
    model = Classic3D_gram(lambda_r=reg_weight)
    actual, gradient = model._regularizer_terms(tensor, compute_grad=True, **params)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, exact.double(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(gradient, expected_grad, rtol=tolerance, atol=tolerance)
    loss_only, no_grad = model._regularizer_terms(tensor, compute_grad=False, **params)
    torch.testing.assert_close(loss_only, actual)
    assert no_grad is None and not actual.requires_grad
    normal = model._regularizer_terms(tensor, compute_grad=True, normal_only=True, **params)[1]
    offset = model._regularizer_terms(torch.zeros_like(tensor), compute_grad=True, **params)[1]
    torch.testing.assert_close(normal + offset, gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('padding_fraction', [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize('shape', [(4, 5), (5, 4), (7, 7), (1, 5)])
def test_padding_matches_dense_periodic_weighted_operator(padding_fraction, shape):
    # Independent dense construction: evaluate the weighted cosine kernel
    # at wrapped pixel separations, averaging ambiguous +/- P/2 edges.
    x, params = case(shape=shape, channels=1, beam_shapes=(shape,))
    params['lambda_sd'] = 0
    model = Classic3D_gram(lambda_r=0, gram_dtype=torch.float64, nufft_eps=1e-12,
                           padding_fraction=padding_fraction)
    actual, gradient = evaluate(model, x, params, torch.float64)
    block = model._gram_cache.blocks[0]
    py, px = block.padded_shape
    if padding_fraction == 1:
        assert block.padded_shape == shape
    _, _, data, sigma, uu, vv, _ = next(params['vis_data'].iter_chan_beam_I())
    _, u, v = uvw_to_radpix(uu, vv, params['cell_size'], 'cpu')
    points = torch.stack((-v, u)).double()
    yy, xx = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
    yy, xx = yy.flatten(), xx.flatten()
    dy = (yy[:, None] - yy[None, :] + py // 2) % py - py // 2
    dx = (xx[:, None] - xx[None, :] + px // 2) % px - px // 2
    alt_y = torch.where((py % 2 == 0) & (dy == -py // 2), -dy, dy)
    alt_x = torch.where((px % 2 == 0) & (dx == -px // 2), -dx, dx)
    weight = torch.tensor(sigma).double().square().reciprocal()
    dense = torch.zeros((x.size, x.size), dtype=torch.float64)
    for dy_use in (dy, alt_y):
        for dx_use in (dx, alt_x):
            phase = dy_use[..., None] * points[0] + dx_use[..., None] * points[1]
            dense += (phase.cos() * weight).sum(-1) * (params['cell_size']**4 / 4)
    torch.testing.assert_close(dense, dense.T, rtol=1e-12, atol=1e-12)

    modes = torch.stack((yy - shape[0] // 2, xx - shape[1] // 2)).double()
    forward = params['cell_size']**2 * torch.exp(1j * (points.T @ modes))
    data_t = torch.tensor(data.conj().copy()).to(torch.complex128)
    dirty = (forward.conj().T @ (weight * data_t)).real
    tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    projected = F.grid_sample(tensor[:, None], torch.tensor(params['grid_list'][0]).double(),
                              align_corners=True)[0, 0] * torch.tensor(params['pb_list'][0]).double()
    local = projected.flatten()
    expected = 0.5 * local @ dense @ local - local @ dirty + 0.5 * (weight * data_t.abs().square()).sum()
    expected.backward()
    torch.testing.assert_close(actual, expected.detach(), rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(gradient, tensor.grad, rtol=2e-10, atol=2e-10)
    normal = model.apply_normal_operator(tensor.detach(), device='cpu', **params)
    rhs = model.quadratic_rhs(x.shape, device='cpu', **params)
    torch.testing.assert_close(normal - rhs, gradient, rtol=2e-12, atol=2e-12)


def test_padding_invalidates_cache_preserves_dirty_and_reduces_memory():
    x, params = case(shape=(20, 24), channels=1, beam_shapes=((20, 24),))
    model = Classic3D_gram()
    evaluate(model, x, params)
    full = model._gram_cache
    rhs, constant = full.rhs.clone(), full.constant.clone()
    full_direct = model.visibility_loss(torch.tensor(x), device='cpu', **params)
    model.padding_fraction = 1.5
    evaluate(model, x, params)
    assert model._gram_cache is not full
    assert model.cache_nbytes < full.nbytes
    assert model._gram_cache.blocks[0].padded_shape == (30, 36)
    torch.testing.assert_close(model._gram_cache.rhs, rhs)
    torch.testing.assert_close(model._gram_cache.constant, constant)
    torch.testing.assert_close(model.visibility_loss(torch.tensor(x), device='cpu', **params), full_direct)


@pytest.mark.parametrize('value', [0.9, 2.1, float('nan'), float('inf'), -float('inf')])
def test_padding_fraction_validation(value):
    with pytest.raises(ValueError, match='padding_fraction'):
        Classic3D_gram(padding_fraction=value)
