"""Optional preparation must preserve model objectives and reconstructions."""
import numpy as np
import pytest
import torch
from astropy import units as u
from radio_beam import Beam

import ivis.imager as imager_module
from ivis.imager import Imager3D
from ivis.models.classic3D import Classic3D
from ivis.models.classic3D_optimized import Classic3D_optimized
from ivis.models.classic3D_gram import Classic3D_gram
from .test_classic3d_gram import case, evaluate


def make_imager(shape=(63, 64), lambda_sd=0.03):
    x, params = case(shape=shape, beam_shapes=((4, 5), (4, 5)))
    imager = Imager3D(
        params['vis_data'], np.stack(params['pb_list']), np.stack(params['grid_list']),
        x.copy(), Beam(5 * u.arcsec),
        dict(NAXIS2=shape[0], NAXIS1=shape[1], CDELT2=1.9 / 3600),
        np.zeros_like(x), max_its=3, lambda_sd=lambda_sd, positivity=True,
        cost_device='cpu', optim_device='cpu',
    )
    params['lambda_sd'] = lambda_sd
    return imager, x, params


@pytest.mark.parametrize('shape', [(63, 64), (64, 66), (65, 67)])
def test_terms_match_original(shape):
    imager, _, _ = make_imager(shape)
    old = imager._prepare_image_terms(1.9 * u.arcsec, shape)
    new = imager._prepare_image_terms(1.9 * u.arcsec, shape, True)
    for key in old:
        assert old[key].shape == new[key].shape
        assert old[key].dtype == new[key].dtype
        np.testing.assert_allclose(new[key], old[key], rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(new['tapper'], old['tapper'])


@pytest.mark.parametrize('model_class', [Classic3D, Classic3D_optimized, Classic3D_gram])
@pytest.mark.parametrize('lambda_sd', [0, 0.03])
def test_objective_gradient_and_fista_match(model_class, lambda_sd):
    imager, x, params = make_imager(lambda_sd=lambda_sd)
    results, images = [], []
    for fast in (False, True):
        terms = imager._prepare_image_terms(1.9 * u.arcsec, x.shape[-2:], fast)
        results.append(evaluate(model_class(lambda_r=0.02), x, dict(params, **terms)))
        images.append(imager.process(model=model_class(lambda_r=0.02), solver='FISTA',
                                     initial_step=1e-6, fast_setup=fast))
    for new, old in zip(results[1], results[0]):
        torch.testing.assert_close(new, old, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(images[1], images[0], rtol=2e-6, atol=1e-8)


def test_cached_geometry_and_disabled_single_dish(monkeypatch):
    imager, _, _ = make_imager(lambda_sd=0)
    shape = (63, 64)
    first = imager._prepare_image_terms(1.9 * u.arcsec, shape, True)

    def fail(*args, **kwargs):
        raise AssertionError('unnecessary image preparation')

    with monkeypatch.context() as patch:
        patch.setattr(imager_module.dutils, 'apodize', fail)
        patch.setattr(imager_module.dutils, 'gauss_beam', fail)
        patch.setattr(imager_module.dutils, 'laplacian', fail)
        patch.setattr(imager_module, 'fft2', fail)
        patch.setattr(imager_module, 'tfft2', fail)
        cached = imager._prepare_image_terms(1.9 * u.arcsec, shape, True)
    assert cached['tapper'] is first['tapper']
    assert cached['fftkernel'] is first['fftkernel']
    assert cached['fftsd'] is cached['fftbeam'] is None
    changed = imager._prepare_image_terms(1.9 * u.arcsec, (65, 67), True)
    assert changed['tapper'].shape == changed['fftkernel'].shape == (65, 67)
    assert changed['tapper'] is not first['tapper']


def test_active_single_dish_inputs_are_refreshed():
    imager, _, _ = make_imager()
    shape = (63, 64)
    first = imager._prepare_image_terms(1.9 * u.arcsec, shape, True)
    imager.sd += 0.5
    imager.beam_sd = Beam(9 * u.arcsec)
    updated = imager._prepare_image_terms(2.4 * u.arcsec, shape, True)
    expected = imager._prepare_image_terms(2.4 * u.arcsec, shape)
    assert updated['tapper'] is first['tapper']
    assert not np.array_equal(updated['fftsd'], first['fftsd'])
    assert not np.array_equal(updated['fftbeam'], first['fftbeam'])
    np.testing.assert_array_equal(updated['fftsd'], expected['fftsd'])
    np.testing.assert_array_equal(updated['fftbeam'], expected['fftbeam'])
