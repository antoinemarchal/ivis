import ivis
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from radio_beam import Beam

from ivis.imager import Imager3D
from ivis.models import Classic3D
from ivis.types import VisIData


def test_package_exports_version():
    assert ivis.__version__ == "0.1.0"


def test_package_exports_logger():
    assert ivis.logger.name == "IViS"


def _identity_grid(height, width):
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1)[None, ...]


def test_get_started_forward_model_smoke():
    height = width = 8
    nvis = 4

    vis_data = VisIData(
        frequency=np.array([1.4e9]),
        velocity=np.array([0.0]),
        centers=np.array([SkyCoord(0 * u.deg, 0 * u.deg)]),
        nvis=np.array([nvis]),
        uu=np.array([[0.0, 5.0, -5.0, 10.0]], dtype=np.float32),
        vv=np.array([[0.0, -3.0, 3.0, 1.0]], dtype=np.float32),
        ww=np.zeros((1, nvis), dtype=np.float32),
        data_I=np.zeros((1, 1, nvis), dtype=np.complex64),
        sigma_I=np.ones((1, 1, nvis), dtype=np.float32),
        flag_I=np.zeros((1, 1, nvis), dtype=bool),
    )

    pb = np.ones((1, height, width), dtype=np.float32)
    grid = np.stack([_identity_grid(height, width)], axis=0)
    sd = np.zeros((height, width), dtype=np.float32)
    init_params = np.zeros((1, height, width), dtype=np.float32)
    init_params[0, height // 2, width // 2] = 1.0

    hdr = {
        "NAXIS1": width,
        "NAXIS2": height,
        "CDELT2": 1.0 / 3600.0,
    }
    beam_sd = Beam(1.0 * u.arcsec, 1.0 * u.arcsec, 0.0 * u.deg)

    imager = Imager3D(
        vis_data,
        pb,
        grid,
        sd,
        beam_sd,
        hdr,
        init_params=init_params,
        max_its=1,
        lambda_sd=0,
        positivity=False,
        cost_device="cpu",
        optim_device="cpu",
        beam_workers=0,
    )

    model_vis = imager.forward_model(Classic3D(lambda_r=1))

    assert model_vis.shape == (1, 1, nvis)
    assert np.isfinite(model_vis.real).all()
    assert np.isfinite(model_vis.imag).all()
