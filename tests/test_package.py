import ivis
import importlib
import numpy as np
import sys
import astropy.units as u
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import types

from ivis.imager import Imager3D
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


def _install_fake_pytorch_finufft(monkeypatch):
    def finufft_type2(points, c, isign=1, modeord=0):
        return c.new_zeros(points.shape[1], dtype=c.dtype)

    def finufft_type1(points, y, out_shape, isign=-1, modeord=0):
        return y.new_zeros(out_shape, dtype=y.dtype)

    fake_module = types.SimpleNamespace(
        functional=types.SimpleNamespace(
            finufft_type1=finufft_type1,
            finufft_type2=finufft_type2,
        )
    )
    monkeypatch.setitem(sys.modules, "pytorch_finufft", fake_module)


def test_get_started_forward_model_smoke(monkeypatch):
    height = width = 8
    nvis = 4

    _install_fake_pytorch_finufft(monkeypatch)
    Classic3D = importlib.import_module("ivis.models").Classic3D

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


def test_classic3d_memory_matches_classic3d_objective_and_gradient(monkeypatch):
    torch = importlib.import_module("torch")
    classic3d_mod = importlib.import_module("ivis.models.classic3D")
    classic3d_memory_mod = importlib.import_module("ivis.models.classic3D_memory")
    Classic3D = classic3d_mod.Classic3D
    Classic3DMemory = classic3d_memory_mod.Classic3DMemory

    def fake_forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device):
        flat = x2d.reshape(-1)
        idx = torch.arange(len(uu), device=device) % flat.numel()
        return flat[idx].to(torch.complex64) * torch.tensor(1.25 - 0.5j, device=device)

    monkeypatch.setattr(classic3d_mod, "forward_beam", fake_forward_beam)
    monkeypatch.setattr(classic3d_memory_mod, "forward_beam", fake_forward_beam)

    nchan, nbeam, nvis = 2, 2, 3
    height = width = 3
    vis_data = VisIData(
        frequency=np.array([1.4e9, 1.41e9]),
        velocity=np.array([0.0, 1.0]),
        centers=np.array([SkyCoord(0 * u.deg, 0 * u.deg), SkyCoord(1 * u.deg, 1 * u.deg)]),
        nvis=np.array([nvis, nvis]),
        uu=np.ones((nbeam, nvis), dtype=np.float32),
        vv=np.ones((nbeam, nvis), dtype=np.float32),
        ww=np.zeros((nbeam, nvis), dtype=np.float32),
        data_I=(
            np.arange(nchan * nbeam * nvis, dtype=np.float32).reshape(nchan, nbeam, nvis)
            + 1j * np.ones((nchan, nbeam, nvis), dtype=np.float32)
        ).astype(np.complex64),
        sigma_I=np.full((nchan, nbeam, nvis), 2.0, dtype=np.float32),
        flag_I=np.zeros((nchan, nbeam, nvis), dtype=bool),
    )

    x0 = np.linspace(0.1, 1.0, nchan * height * width, dtype=np.float32).reshape(nchan, height, width)
    common_params = dict(
        vis_data=vis_data,
        device="cpu",
        pb=np.ones((nbeam, height, width), dtype=np.float32),
        grid_array=np.zeros((nbeam, 1, height, width, 2), dtype=np.float32),
        cell_size=1.0,
        lambda_sd=0.0,
        fftkernel=None,
    )

    x_classic = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    loss_classic = Classic3D(lambda_r=0.0).objective(x_classic, **common_params)
    grad_classic = x_classic.grad.detach().clone()

    x_memory = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    loss_memory = Classic3DMemory(lambda_r=0.0).objective(x_memory, **common_params)
    grad_memory = x_memory.grad.detach().clone()

    assert torch.allclose(loss_memory, loss_classic.detach(), rtol=1e-6, atol=1e-5)
    assert torch.allclose(grad_memory, grad_classic, rtol=1e-6, atol=1e-5)
