import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord

from ivis.models import Classic3D, Classic3DSpeed
from ivis.types import VisIData


def test_classic3d_speed_matches_classic3d_at_default_accuracy():
    """Cached plans and batched grids retain the Classic3D data objective."""
    nbeam, nvis, height, width = 2, 4, 4, 4
    vis_data = VisIData(
        frequency=np.array([1.4e9]),
        velocity=np.array([0.0]),
        centers=np.array([
            SkyCoord(0 * u.deg, 0 * u.deg),
            SkyCoord(1 * u.deg, 0 * u.deg),
        ]),
        nvis=np.full(nbeam, nvis),
        uu=np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]], dtype=np.float32),
        vv=np.array([[0.5, 1.5, 2.5, 3.5], [0.2, 1.2, 2.2, 3.2]], dtype=np.float32),
        ww=np.zeros((nbeam, nvis), dtype=np.float32),
        data_I=np.array([[[1 + 0.5j, 0.5 + 1j, 1.5 - 0.5j, 0.25 + 0.75j],
                          [0.3 + 0.2j, 0.7 - 0.4j, 1.1 + 0.6j, 0.2 - 0.8j]]], dtype=np.complex64),
        sigma_I=np.ones((1, nbeam, nvis), dtype=np.float32),
        flag_I=np.zeros((1, nbeam, nvis), dtype=bool),
    )
    grid = np.zeros((nbeam, 1, height, width, 2), dtype=np.float32)
    pb = np.ones((nbeam, height, width), dtype=np.float32)
    params = dict(
        vis_data=vis_data,
        device="cpu",
        pb=pb,
        grid_array=grid,
        cell_size=1.0,
        lambda_sd=0.0,
        fftkernel=None,
    )
    x0 = np.linspace(0.1, 1.0, height * width, dtype=np.float32).reshape(1, height, width)

    x_ref = torch.tensor(x0, requires_grad=True)
    loss_ref = Classic3D(lambda_r=0.0).objective(x_ref, **params)

    x_speed = torch.tensor(x0, requires_grad=True)
    speed = Classic3DSpeed(
        lambda_r=0.0, nufft_eps=1e-6, reprojection_batch_size=2
    )
    loss_speed = speed.objective(x_speed, **params)

    assert torch.allclose(loss_ref, loss_speed, rtol=3e-5, atol=3e-5)
    assert torch.allclose(x_ref.grad, x_speed.grad, rtol=3e-5, atol=3e-5)

    # The second call exercises the persistent tensor and FINUFFT-plan cache.
    x_warm = torch.tensor(x0, requires_grad=True)
    loss_warm = speed.objective(x_warm, **params)
    assert torch.allclose(loss_speed, loss_warm, rtol=1e-6, atol=1e-6)
    assert torch.allclose(x_speed.grad, x_warm.grad, rtol=1e-6, atol=1e-6)
