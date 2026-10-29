"""positivity_floor: FISTA with a negative lower bound (soft positivity) reaches the bound-constrained minimiser."""
import numpy as np
import pytest
import torch

from ivis.imager import Imager3D
from ivis.optim.solvers import optimize_torch_fista


class Separable:
    """0.5 * sum(d * (x - t)^2): the minimiser under x >= floor is max(t, floor), elementwise."""
    supports_loss_only_objective = True
    supports_quadratic_gradient_reuse = True

    def __init__(self, target, curvature=1.0):
        self.target = torch.as_tensor(target, dtype=torch.float64)
        self.curvature = curvature

    def objective(self, x, device, compute_grad=True):
        diff = x.detach().double() - self.target
        if compute_grad:
            x.grad = (self.curvature * diff).to(x.dtype)
        return 0.5 * self.curvature * torch.sum(diff ** 2)


def solve(model, initial, **kwargs):
    options = dict(model=model, x_init=initial, dtype=torch.float64, max_its=300,
                   cost_dev=torch.device('cpu'), optim_dev=torch.device('cpu'), params={},
                   initial_step=0.5, grow_factor=1.0, positivity=True)
    options.update(kwargs)
    return optimize_torch_fista(**options)


TARGET = np.array([[-2.0, -0.4, 0.0, 0.7], [1.5, -0.05, -1.0, 0.2]])


def test_default_floor_is_plain_positivity():
    result = solve(Separable(TARGET), np.zeros(TARGET.shape))
    np.testing.assert_allclose(result, np.maximum(TARGET, 0), atol=1e-9)


def test_scalar_zero_floor_matches_the_default_exactly():
    base = solve(Separable(TARGET), np.zeros(TARGET.shape))
    zero = solve(Separable(TARGET), np.zeros(TARGET.shape), positivity_floor=0.0)
    np.testing.assert_array_equal(base, zero)


@pytest.mark.parametrize('floor', [-0.3, np.float32(-0.3), torch.tensor(-0.3)])
def test_scalar_floor_reaches_the_bounded_minimiser(floor):
    result = solve(Separable(TARGET), np.zeros(TARGET.shape), positivity_floor=floor)
    np.testing.assert_allclose(result, np.maximum(TARGET, -0.3), atol=1e-6)
    assert result.min() >= float(floor) - 1e-12                                   # the floor's own value (float32 -0.3 is not float64 -0.3)


@pytest.mark.parametrize('shape', [(2, 4), (1, 4), (2, 1), (4,)])
def test_array_floor_is_broadcast_per_pixel(shape):
    rng = np.random.default_rng(3)
    floor = -np.abs(rng.normal(0.5, 0.3, size=shape))
    result = solve(Separable(TARGET), np.zeros(TARGET.shape), positivity_floor=floor)
    np.testing.assert_allclose(result, np.maximum(TARGET, floor), atol=1e-6)
    assert (result >= np.broadcast_to(floor, TARGET.shape) - 1e-12).all()


def test_floor_on_a_channel_image_stack():
    target = np.random.default_rng(1).normal(size=(2, 5, 6))
    floor = -0.5 * np.abs(np.random.default_rng(2).normal(size=(5, 6)))          # one map for every channel
    result = solve(Separable(target), np.zeros(target.shape), positivity_floor=floor)
    np.testing.assert_allclose(result, np.maximum(target, floor[None]), atol=1e-6)


def test_noise_like_target_keeps_both_signs_with_a_floor_but_not_without():
    target = np.random.default_rng(5).normal(size=4000)
    plain = solve(Separable(target), np.zeros(4000))
    soft = solve(Separable(target), np.zeros(4000), positivity_floor=-4.0)
    assert plain.min() == 0.0 and (plain == 0).mean() > 0.4
    assert abs(soft.mean() - target.mean()) < 1e-6
    np.testing.assert_allclose(soft, np.maximum(target, -4.0), atol=1e-6)


@pytest.mark.parametrize('bad', [0.1, np.array([[-1.0, 0.2, -1.0, -1.0], [-1.0] * 4]), np.nan, -np.inf])
def test_invalid_floor_values_are_refused(bad):
    with pytest.raises(ValueError):
        solve(Separable(TARGET), np.zeros(TARGET.shape), positivity_floor=bad)


def test_floor_that_does_not_broadcast_is_refused():
    with pytest.raises(ValueError, match='broadcast'):
        solve(Separable(TARGET), np.zeros(TARGET.shape), positivity_floor=-np.ones((3, 4)))
    with pytest.raises(ValueError, match='broadcast'):
        solve(Separable(TARGET), np.zeros(TARGET.shape), positivity_floor=-np.ones((2, 2, 4)))


def test_floor_without_positivity_is_refused():
    with pytest.raises(ValueError, match='positivity=True'):
        solve(Separable(TARGET), np.zeros(TARGET.shape), positivity=False, positivity_floor=-1.0)


def test_imager_requires_positivity_for_a_floor():
    common = dict(vis_data=None, pb=None, grid=None, sd=None, beam_sd=None, hdr=None,
                  init_params=np.zeros((1, 2, 2)), max_its=1, lambda_sd=0, cost_device='cpu', optim_device='cpu')
    with pytest.raises(ValueError, match='positivity=True'):
        Imager3D(positivity=False, positivity_floor=-1.0, **common)
    imager = Imager3D(positivity=True, positivity_floor=-1.0, **common)
    assert imager.positivity_floor == -1.0
    assert Imager3D(positivity=True, **common).positivity_floor is None
