"""psf_positions swaps only the normal-equation right-hand side for H*delta; the objective itself is unchanged."""
import dataclasses

import numpy as np
import pytest
import torch

from ivis.models.classic3D_gram import Classic3D_gram
from .test_classic3d_gram import case

POSITIONS = [(2, 3), (4, 1)]


def gradient(model, x, params):
    leaf = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    model.objective(leaf, device='cpu', **params)
    return leaf.grad.detach()


def options(**extra):
    return dict(gram_dtype=torch.float64, nufft_eps=1e-10, **extra)


def test_gradient_at_zero_is_minus_h_delta():
    x, params = case(shape=(6, 7), channels=2)
    params['lambda_sd'] = 0
    plain = Classic3D_gram(lambda_r=0, **options())
    delta = np.zeros_like(x)
    for row, col in POSITIONS:
        delta[:, row, col] = 1
    h_delta = plain.apply_normal_operator(torch.tensor(delta, dtype=torch.float64), device='cpu', **params)
    psf = Classic3D_gram(lambda_r=0, psf_positions=POSITIONS, **options())
    torch.testing.assert_close(gradient(psf, np.zeros_like(x), params), -h_delta, rtol=1e-9, atol=1e-9)
    assert float(h_delta.abs().max()) > 0


@pytest.mark.parametrize('lambda_r', [0, 0.05])
def test_curvature_is_unchanged(lambda_r):
    """grad(x) - grad(0) is (H + regularizer) x for both models, so only the right-hand side moved."""
    x, params = case(shape=(6, 7), channels=2)
    params['lambda_sd'] = 0
    plain = Classic3D_gram(lambda_r=lambda_r, **options())
    psf = Classic3D_gram(lambda_r=lambda_r, psf_positions=POSITIONS, **options())
    zero = np.zeros_like(x)
    torch.testing.assert_close(gradient(psf, x, params) - gradient(psf, zero, params),
                               gradient(plain, x, params) - gradient(plain, zero, params), rtol=1e-9, atol=1e-9)


def test_data_values_are_ignored():
    """Only the weights and uv coverage matter: different data values give the same gradient."""
    x, params = case(shape=(6, 7), channels=2)
    params['lambda_sd'] = 0
    vis = params['vis_data']
    changed = dataclasses.replace(vis, data_I=(vis.data_I * 3 + 1).astype('complex64'))
    first = Classic3D_gram(lambda_r=0.05, psf_positions=POSITIONS, **options())
    second = Classic3D_gram(lambda_r=0.05, psf_positions=POSITIONS, **options())
    torch.testing.assert_close(gradient(first, x, params), gradient(second, x, {**params, 'vis_data': changed}),
                               rtol=1e-9, atol=1e-9)
    plain_first = Classic3D_gram(lambda_r=0.05, **options())
    plain_second = Classic3D_gram(lambda_r=0.05, **options())
    assert not torch.allclose(gradient(plain_first, x, params), gradient(plain_second, x, {**params, 'vis_data': changed}))


def test_default_is_unchanged():
    x, params = case(shape=(6, 7), channels=2)
    a = Classic3D_gram(lambda_r=0.05, **options())
    b = Classic3D_gram(lambda_r=0.05, psf_positions=None, **options())
    assert b.psf_positions is None
    # Separate instances differ by float64 round-off (FFT ordering), with or without the option.
    torch.testing.assert_close(gradient(a, x, params), gradient(b, x, params), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('bad', [[], [(1,)], [(1, 2, 3)], 'ab', [('a', 'b')]])
def test_invalid_positions_are_refused(bad):
    with pytest.raises(ValueError):
        Classic3D_gram(psf_positions=bad)


@pytest.mark.parametrize('position', [(6, 0), (0, 7), (-1, 2), (2, -1)])
def test_positions_outside_the_image_are_refused(position):
    x, params = case(shape=(6, 7), channels=2)
    model = Classic3D_gram(lambda_r=0, psf_positions=[position], **options())
    with pytest.raises(ValueError, match='outside'):
        gradient(model, x, params)
