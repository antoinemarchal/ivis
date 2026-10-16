import numpy as np
import pytest
import torch

from ivis.optim.solvers import optimize_torch_fista


class DenseQuadratic:
    supports_loss_only_objective = True
    supports_quadratic_gradient_reuse = True

    def __init__(self, matrix, rhs):
        self.matrix = torch.as_tensor(matrix, dtype=torch.float64)
        self.rhs = torch.as_tensor(rhs, dtype=torch.float64)
        self.calls = []

    def objective(self, x, device, compute_grad=True):
        self.calls.append((compute_grad, x.detach().numpy().copy()))
        value = x.detach().double()
        normal = self.matrix @ value
        loss = 0.5 * torch.dot(value, normal) - torch.dot(self.rhs, value) + 20.0
        if compute_grad:
            x.grad = (normal - self.rhs).to(x.dtype)
        return loss


def solve(model, initial, **kwargs):
    options = dict(model=model, x_init=initial, dtype=torch.float64,
                   max_its=5, cost_dev=torch.device('cpu'),
                   optim_dev=torch.device('cpu'), params={}, initial_step=0.02,
                   grow_factor=1.0, loss_only_line_search=True)
    options.update(kwargs)
    return optimize_torch_fista(**options)


@pytest.mark.parametrize('positivity', [False, True])
@pytest.mark.parametrize('reuse', [None, True])
def test_quadratic_default_reuses_gradients_and_halves_iteration_evaluations(positivity, reuse):
    args = (np.diag([1.0, 4.0]), np.array([-0.5, 0.8]))
    initial = np.array([1.0, -1.0])
    baseline, model = DenseQuadratic(*args), DenseQuadratic(*args)
    expected = solve(baseline, initial, positivity=positivity, quadratic_gradient_reuse=False)
    actual = solve(model, initial, positivity=positivity, quadratic_gradient_reuse=reuse)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert len(baseline.calls) == 11  # Initial evaluation, then two per iteration.
    assert len(model.calls) == 6     # Initial evaluation, then one per iteration.
    assert all(compute_grad for compute_grad, _ in model.calls)
    expected_trials = [x for compute_grad, x in baseline.calls if not compute_grad]
    np.testing.assert_allclose([x for _, x in model.calls[1:]], expected_trials,
                               rtol=1e-13, atol=1e-13)
    if positivity:
        assert (actual >= 0).all()


@pytest.mark.parametrize('positivity', [False, True])
@pytest.mark.parametrize('seed', [17, 25])
def test_quadratic_reuse_preserves_backtracking_and_restarts(positivity, seed, caplog):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(8, 8))
    matrix = factor.T @ factor + 0.01 * np.eye(8)
    rhs, initial = rng.normal(size=(2, 8))
    baseline, model = DenseQuadratic(matrix, rhs), DenseQuadratic(matrix, rhs)
    options = dict(positivity=positivity, max_its=15, initial_step=1.0, grow_factor=1.25)
    expected = solve(baseline, initial, quadratic_gradient_reuse=False, **options)
    expected_restarts = [record.message for record in caplog.records if 'restart;' in record.message]
    caplog.clear()
    actual = solve(model, initial, **options)
    actual_restarts = [record.message for record in caplog.records if 'restart;' in record.message]
    assert actual_restarts == expected_restarts
    if positivity and seed == 17:
        # Restart at y: legacy loss-only FISTA must recompute a gradient at x.
        assert sum(grad for grad, _ in baseline.calls) > options['max_its'] + 1
    if positivity and seed == 25:
        # A projected candidate can pass majorization but exceed the last loss.
        assert expected_restarts
    expected_trials = [x for compute_grad, x in baseline.calls if not compute_grad]
    assert len(expected_trials) > options['max_its']  # Includes rejected trial steps.
    assert len(model.calls) == len(expected_trials) + 1
    np.testing.assert_allclose([x for _, x in model.calls[1:]], expected_trials,
                               rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_quadratic_reuse_refreshes_borderline_comparisons(caplog):
    rng = np.random.default_rng(17)
    factor = rng.normal(size=(8, 8))
    matrix = factor.T @ factor + 0.01 * np.eye(8)
    rhs, initial = rng.normal(size=(2, 8))
    options = dict(positivity=True, max_its=30, initial_step=1.0, grow_factor=1.25)
    expected = solve(DenseQuadratic(matrix, rhs), initial,
                     quadratic_gradient_reuse=False, **options)
    caplog.clear()
    actual = solve(DenseQuadratic(matrix, rhs), initial, **options)
    timing = next(record.message for record in caplog.records if '[Timing]' in record.message)
    assert int(timing.split('quadratic_refreshes=')[1].split()[0]) > 0
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)


def test_quadratic_reuse_requires_explicit_model_capability():
    model = DenseQuadratic(np.eye(2), np.ones(2))
    model.supports_quadratic_gradient_reuse = False
    initial = np.zeros(2)
    solve(model, initial)
    assert len(model.calls) == 11
    model.calls.clear()
    with pytest.raises(ValueError, match='fixed quadratic objective'):
        solve(model, initial, quadratic_gradient_reuse=True)
    assert not model.calls


def test_quadratic_reuse_does_not_require_loss_only_support():
    model = DenseQuadratic(np.eye(2), np.ones(2))
    model.supports_loss_only_objective = False
    result = solve(model, np.zeros(2))
    assert np.isfinite(result).all()
    assert all(compute_grad for compute_grad, _ in model.calls)
    with pytest.raises(ValueError, match='loss_only_line_search'):
        solve(model, np.zeros(2), quadratic_gradient_reuse=False)
