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


def test_lrsb_memory_matches_lrsb_objective_and_gradient(monkeypatch):
    torch = importlib.import_module("torch")
    lrsb_mod = importlib.import_module("ivis.models.lrsb")
    LRSB = lrsb_mod.LRSB
    LRSBMemory = lrsb_mod.LRSBMemory

    def fake_forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device):
        flat = x2d.reshape(-1)
        idx = torch.arange(len(uu), device=device) % flat.numel()
        return flat[idx].to(torch.complex64) * torch.tensor(0.75 + 0.25j, device=device)

    monkeypatch.setattr(lrsb_mod, "forward_beam", fake_forward_beam)

    nchan, nbasis, nbeam, nvis = 3, 2, 2, 4
    height = width = 3
    vis_data = VisIData(
        frequency=np.array([1.4e9, 1.41e9, 1.42e9]),
        velocity=np.array([0.0, 1.0, 2.0]),
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

    basis = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.0, 0.75, 1.25],
        ],
        dtype=np.float32,
    )
    x0 = np.linspace(-0.5, 1.0, nbasis * height * width, dtype=np.float32).reshape(
        nbasis, height, width
    )
    common_params = dict(
        vis_data=vis_data,
        device="cpu",
        pb=np.ones((nbeam, height, width), dtype=np.float32),
        grid_array=np.zeros((nbeam, 1, height, width, 2), dtype=np.float32),
        cell_size=1.0,
        lambda_sd=0.0,
        fftkernel=None,
    )

    for invariant in (False, True):
        model_params = dict(
            basis=basis,
            lambda_r=0.0,
            lambda_pos=0.3,
            assume_channel_invariant_operator=invariant,
        )

        x_lrsb = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        loss_lrsb = LRSB(**model_params).objective(x_lrsb, **common_params)
        grad_lrsb = x_lrsb.grad.detach().clone()

        x_memory = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
        loss_memory = LRSBMemory(**model_params).objective(x_memory, **common_params)
        grad_memory = x_memory.grad.detach().clone()

        assert torch.allclose(loss_memory, loss_lrsb.detach(), rtol=1e-6, atol=1e-5)
        assert torch.allclose(grad_memory, grad_lrsb, rtol=1e-6, atol=1e-5)


def test_lrsb_c_matches_lrsb_with_explicit_continuum_basis(monkeypatch):
    torch = importlib.import_module("torch")
    lrsb_mod = importlib.import_module("ivis.models.lrsb")
    LRSB = lrsb_mod.LRSB
    LRSB_C = lrsb_mod.LRSB_C

    def fake_forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device):
        flat = x2d.reshape(-1)
        idx = torch.arange(len(uu), device=device) % flat.numel()
        return flat[idx].to(torch.complex64) * torch.tensor(0.5 - 0.125j, device=device)

    monkeypatch.setattr(lrsb_mod, "forward_beam", fake_forward_beam)

    nchan, nbeam, nvis = 3, 2, 4
    nline, height, width = 2, 3, 3
    vis_data = VisIData(
        frequency=np.array([1.4e9, 1.41e9, 1.42e9]),
        velocity=np.array([0.0, 1.0, 2.0]),
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

    line_basis = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.0, 0.75, 1.25],
        ],
        dtype=np.float32,
    )
    continuum_basis = np.ones((1, nchan), dtype=np.float32)
    hybrid_basis = np.concatenate((line_basis, continuum_basis), axis=0)

    x0 = np.linspace(-0.5, 1.0, hybrid_basis.shape[0] * height * width, dtype=np.float32).reshape(
        hybrid_basis.shape[0], height, width
    )
    common_params = dict(
        vis_data=vis_data,
        device="cpu",
        pb=np.ones((nbeam, height, width), dtype=np.float32),
        grid_array=np.zeros((nbeam, 1, height, width, 2), dtype=np.float32),
        cell_size=1.0,
        lambda_sd=0.0,
        fftkernel=None,
    )

    x_ref = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    ref_model = LRSB(basis=hybrid_basis, lambda_r=0.0, lambda_pos=0.2)
    ref_loss = ref_model.objective(x_ref, **common_params)
    ref_grad = x_ref.grad.detach().clone()

    x_hybrid = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    hybrid_model = LRSB_C(basis=line_basis, continuum_basis=continuum_basis, lambda_r=0.0, lambda_pos=0.2)
    hybrid_loss = hybrid_model.objective(x_hybrid, **common_params)
    hybrid_grad = x_hybrid.grad.detach().clone()

    assert hybrid_model.line_nbasis == nline
    assert hybrid_model.continuum_nbasis == 1
    assert torch.allclose(hybrid_loss, ref_loss.detach(), rtol=1e-6, atol=1e-5)
    assert torch.allclose(hybrid_grad, ref_grad, rtol=1e-6, atol=1e-5)


def test_lrsb_cmemory_matches_lrsb_c_objective_and_gradient(monkeypatch):
    torch = importlib.import_module("torch")
    lrsb_mod = importlib.import_module("ivis.models.lrsb")
    LRSB_C = lrsb_mod.LRSB_C
    LRSB_CMemory = lrsb_mod.LRSB_CMemory

    def fake_forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device):
        flat = x2d.reshape(-1)
        idx = torch.arange(len(uu), device=device) % flat.numel()
        return flat[idx].to(torch.complex64) * torch.tensor(0.75 + 0.25j, device=device)

    monkeypatch.setattr(lrsb_mod, "forward_beam", fake_forward_beam)

    nchan, nbeam, nvis = 3, 2, 4
    height = width = 3
    vis_data = VisIData(
        frequency=np.array([1.4e9, 1.41e9, 1.42e9]),
        velocity=np.array([0.0, 1.0, 2.0]),
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

    basis = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.0, 0.75, 1.25],
        ],
        dtype=np.float32,
    )
    x0 = np.linspace(-0.5, 1.0, 3 * height * width, dtype=np.float32).reshape(3, height, width)
    common_params = dict(
        vis_data=vis_data,
        device="cpu",
        pb=np.ones((nbeam, height, width), dtype=np.float32),
        grid_array=np.zeros((nbeam, 1, height, width, 2), dtype=np.float32),
        cell_size=1.0,
        lambda_sd=0.0,
        fftkernel=None,
    )

    x_ref = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    ref_loss = LRSB_C(basis=basis, lambda_r=0.0, lambda_pos=0.3).objective(x_ref, **common_params)
    ref_grad = x_ref.grad.detach().clone()

    x_memory = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    memory_loss = LRSB_CMemory(basis=basis, lambda_r=0.0, lambda_pos=0.3).objective(
        x_memory, **common_params
    )
    memory_grad = x_memory.grad.detach().clone()

    assert torch.allclose(memory_loss, ref_loss.detach(), rtol=1e-6, atol=1e-5)
    assert torch.allclose(memory_grad, ref_grad, rtol=1e-6, atol=1e-5)


def test_lrsb_c_taylor_basis_matches_explicit_continuum_basis(monkeypatch):
    torch = importlib.import_module("torch")
    lrsb_mod = importlib.import_module("ivis.models.lrsb")
    LRSB_C = lrsb_mod.LRSB_C

    def fake_forward_beam(x2d, primary_beam, grid, uu, vv, ww, cell_size, device):
        flat = x2d.reshape(-1)
        idx = torch.arange(len(uu), device=device) % flat.numel()
        return flat[idx].to(torch.complex64) * torch.tensor(0.25 + 0.5j, device=device)

    monkeypatch.setattr(lrsb_mod, "forward_beam", fake_forward_beam)

    nchan, nbeam, nvis = 3, 2, 4
    height = width = 3
    frequency = np.array([1.40e9, 1.41e9, 1.42e9], dtype=np.float32)
    nu_ref = 1.41e9
    vis_data = VisIData(
        frequency=frequency,
        velocity=np.array([0.0, 1.0, 2.0]),
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

    basis = np.array(
        [
            [1.0, 0.5, -0.25],
            [0.0, 0.75, 1.25],
        ],
        dtype=np.float32,
    )
    xnu = (frequency - nu_ref) / nu_ref
    explicit_continuum_basis = np.stack([np.ones_like(xnu), xnu], axis=0).astype(np.float32)
    x0 = np.linspace(-0.5, 1.0, 4 * height * width, dtype=np.float32).reshape(4, height, width)
    common_params = dict(
        vis_data=vis_data,
        device="cpu",
        pb=np.ones((nbeam, height, width), dtype=np.float32),
        grid_array=np.zeros((nbeam, 1, height, width, 2), dtype=np.float32),
        cell_size=1.0,
        lambda_sd=0.0,
        fftkernel=None,
    )

    x_explicit = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    explicit_model = LRSB_C(
        basis=basis,
        continuum_basis=explicit_continuum_basis,
        lambda_r=0.0,
        lambda_pos=0.2,
    )
    explicit_loss = explicit_model.objective(x_explicit, **common_params)
    explicit_grad = x_explicit.grad.detach().clone()

    x_taylor = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    taylor_model = LRSB_C(
        basis=basis,
        continuum_order=1,
        frequency=frequency,
        reference_frequency=nu_ref,
        lambda_r=0.0,
        lambda_pos=0.2,
    )
    taylor_loss = taylor_model.objective(x_taylor, **common_params)
    taylor_grad = x_taylor.grad.detach().clone()

    assert taylor_model.continuum_order == 1
    assert taylor_model.reference_frequency == nu_ref
    assert np.allclose(taylor_model.continuum_basis, explicit_continuum_basis)
    assert torch.allclose(taylor_loss, explicit_loss.detach(), rtol=1e-6, atol=1e-5)
    assert torch.allclose(taylor_grad, explicit_grad, rtol=1e-6, atol=1e-5)
