import os
import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.utils.tensor_ops import format_input_tensor
from ivis.models.utils.gpu import print_gpu_memory


# ------------------------------------------
# -------- ClassicIViS3DStaged (new) -------
# ------------------------------------------
class ClassicIViS3DStaged:
    """
    A NUFFT-based interferometric forward model with optional GPU staging
    of static inputs (PBs, grids, uvw, visibilities, sigmas).

    Key features:
    - stage_static: 'auto' | 'cpu' | 'cuda'
        'auto'  -> use CUDA if available, else CPU
        'cpu'   -> keep static tensors on CPU
        'cuda'  -> push static tensors to current CUDA device
    - Nw: None or odd integer for w-stacking bins (auto-odd if even)
    - conj_data: match legacy pipeline where data are conjugated
    - use_2pi: whether to scale uu,vv by 2π·cell_rad (keep True unless you
               have a specific alt. convention)
    - Caches everything static per (device, vis_data id). Call clear_cache()
      if vis_data contents change.

    Notes:
    - pytorch_finufft works on CPU or CUDA tensors; we ensure device consistency.
    - Normalization factor uses (cell_size**2) to match your previous code.
    """

    def __init__(self,
                 lambda_r=1.0,
                 Nw=None,
                 use_2pi=True,
                 conj_data=True,
                 stage_static='auto',   # 'auto' | 'cpu' | 'cuda'
                 pin_h2d=True):         # use pinned memory for one-time copies
        self.lambda_r  = float(lambda_r)
        self.Nw        = None if (Nw is None or Nw <= 1) else (Nw if (Nw % 2 == 1) else Nw + 1)
        self.use_2pi   = bool(use_2pi)
        self.conj_data = bool(conj_data)

        if stage_static not in ('auto', 'cpu', 'cuda'):
            raise ValueError("stage_static must be 'auto', 'cpu', or 'cuda'")
        self.stage_static = stage_static
        self.pin_h2d = bool(pin_h2d)

        self._cache = {}      # {(static_dev, id(vis_data)): pack}
        self._r2_cache = {}   # {(H,W,cell_rad,static_dev): r2_tensor}

    # ---------- Utilities ----------
    @staticmethod
    def _cellsize_arcsec_to_rad(cell_size_arcsec: float) -> float:
        return cell_size_arcsec * np.pi / (180.0 * 3600.0)

    @staticmethod
    def _tfft2(x):
        # centered 2D FFT (unitary scaling left to caller, matches your prior usage)
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))

    @staticmethod
    def _log_gpu(device, msg):
        if torch.device(device).type == 'cuda':
            alloc = torch.cuda.memory_allocated(device) / 1024**2
            res   = torch.cuda.memory_reserved(device)  / 1024**2
            tot   = torch.cuda.get_device_properties(device).total_memory / 1024**2
            print(f"{msg} | GPU: {alloc:.1f} MB allocated, {res:.1f} MB reserved, {tot:.0f} MB total")
        else:
            print(msg)

    def clear_cache(self):
        self._cache.clear()
        self._r2_cache.clear()

    # Decide static tensors device
    def _choose_static_device(self, compute_device: torch.device) -> torch.device:
        if self.stage_static == 'cpu':
            return torch.device('cpu')
        if self.stage_static == 'cuda':
            if torch.cuda.is_available():
                return compute_device if compute_device.type == 'cuda' else torch.device('cuda:0')
            return torch.device('cpu')
        # 'auto'
        if torch.cuda.is_available():
            return compute_device if compute_device.type == 'cuda' else torch.device('cuda:0')
        return torch.device('cpu')

    def _to_device_once(self, arr, device, dtype=None, pin=False):
        """
        Convert numpy/torch -> torch on target device (no grad).
        """
        if isinstance(arr, torch.Tensor):
            t = arr
            if dtype is not None and t.dtype != dtype:
                t = t.to(dtype)
            return t.to(device, non_blocking=pin).requires_grad_(False)

        # numpy path
        np_arr = np.asarray(arr)
        if pin and device.type == 'cuda':
            t = torch.from_numpy(np_arr).pin_memory()
            return t.to(device, non_blocking=True, dtype=dtype).requires_grad_(False)
        else:
            return torch.as_tensor(np_arr, device=device, dtype=dtype).requires_grad_(False)

    def _get_r2(self, H, W, cell_rad, device):
        key = (H, W, float(cell_rad), str(device))
        r2 = self._r2_cache.get(key)
        if r2 is not None and r2.device == device:
            return r2
        with torch.no_grad():
            lx = torch.linspace(-W/2, W/2 - 1, W, device=device) * cell_rad
            ly = torch.linspace(-H/2, H/2 - 1, H, device=device) * cell_rad
            l, m = torch.meshgrid(lx, ly, indexing='xy')
            r2 = (l**2 + m**2).contiguous()
        self._r2_cache[key] = r2
        return r2

    # ---------- Static preparation ----------
    def _prepare_static(self, vis_data, pb_list, grid_list, cell_size_arcsec, compute_device):
        """
        Stage & cache all static inputs on the chosen static device.
        """
        static_dev = self._choose_static_device(torch.device(compute_device))
        key = (str(static_dev), id(vis_data))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        cell_rad = self._cellsize_arcsec_to_rad(cell_size_arcsec)
        scale_uv = (2.0 * np.pi * cell_rad) if self.use_2pi else cell_rad

        # Per-beam static tensors
        pb_t  = [self._to_device_once(pb,   static_dev, torch.float32, pin=self.pin_h2d) for pb in pb_list]
        grd_t = [self._to_device_once(grid, static_dev, torch.float32, pin=self.pin_h2d) for grid in grid_list]

        # Per (chan, beam) packs
        chan_beam_packs = []
        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            I_use = I.conj() if self.conj_data else I
            I_re  = self._to_device_once(I_use.real, static_dev, torch.float32, pin=self.pin_h2d)
            I_im  = self._to_device_once(I_use.imag, static_dev, torch.float32, pin=self.pin_h2d)
            sig   = self._to_device_once(sI,         static_dev, torch.float32, pin=self.pin_h2d)

            u_rad = self._to_device_once(uu, static_dev, torch.float32, pin=self.pin_h2d) * scale_uv
            v_rad = self._to_device_once(vv, static_dev, torch.float32, pin=self.pin_h2d)

            if self.use_2pi:
                v_rad = v_rad * scale_uv  # keep same scaling as u

            if self.Nw is not None:
                ww_t = self._to_device_once(ww, static_dev, torch.float32, pin=self.pin_h2d)
                with torch.no_grad():
                    w_edges   = torch.linspace(ww_t.min(), ww_t.max(), self.Nw + 1, device=static_dev)
                    w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
                    bin_ids   = torch.bucketize(ww_t, w_edges) - 1
                    bin_ids.clamp_(0, self.Nw - 1)
            else:
                ww_t = w_edges = w_centers = bin_ids = None

            chan_beam_packs.append({
                "c": c, "b": b,
                "I_re": I_re, "I_im": I_im, "sig": sig,
                "u_rad": u_rad, "v_rad": v_rad,
                "ww": ww_t, "w_edges": w_edges, "w_centers": w_centers, "bin_ids": bin_ids
            })

        pack = {
            "static_device": static_dev,
            "cell_rad": cell_rad,
            "scale_uv": scale_uv,
            "pb": pb_t,
            "grid": grd_t,
            "chan_beam": chan_beam_packs
        }
        self._cache[key] = pack
        return pack

    # ---------- Forward ----------
    def _forward_beam_cached(self, x2d, pb, grid, u_rad, v_rad,
                             ww, w_centers, bin_ids, cell_rad, static_dev, compute_device):
        """
        Forward model using already-staged statics.
        - x2d is on compute_device
        - statics are on static_dev
        We upcast/move lightweight things as needed to match compute device for NUFFT.
        """
        # Reproject & PB multiply on compute device (keep x2d where the compute happens)
        if x2d.ndim == 1:
            side = int(np.sqrt(x2d.numel()))
            x2d = x2d.view(side, side)
        if x2d.ndim != 2:
            raise ValueError(f"x2d must be (H,W); got {tuple(x2d.shape)}")

        # Move pb/grid to compute device (they are small compared to uv / data)
        pb_c   = pb.to(compute_device, non_blocking=True)
        grid_c = grid.to(compute_device, non_blocking=True)

        xt = x2d.unsqueeze(0).unsqueeze(0).float()                 # [1,1,H,W] on compute_device
        repro = torch.nn.functional.grid_sample(xt, grid_c, mode='bilinear', align_corners=True)
        x_pb = repro.squeeze(0).squeeze(0) * pb_c                  # [H,W] on compute_device

        # NUFFT expects all tensors on same device; move u,v (and w path) to compute device once per call
        u_c = u_rad.to(compute_device, non_blocking=True)
        v_c = v_rad.to(compute_device, non_blocking=True)

        cell_size_sq = (cell_rad * (180.0 * 3600.0) / np.pi)**2  # invert cellsize_rad -> arcsec, then square
        # NOTE: keep normalization identical to your previous code:
        # earlier you had (cell_size**2) outside; here we reconstruct via cell_rad.

        if self.Nw is None:
            points = torch.stack([-v_c, u_c], dim=0)               # [2,N]
            cimg = x_pb.to(torch.complex64)
            vis = pytorch_finufft.functional.finufft_type2(points, cimg, isign=1, modeord=0)
            return vis * (cell_size_sq)

        # W-stacking path
        H, W = x_pb.shape
        r2 = self._get_r2(H, W, cell_rad, compute_device)
        model_vis = torch.empty_like(u_c, dtype=torch.complex64)

        # move bin info to compute device (lightweight)
        w_centers_c = w_centers.to(compute_device, non_blocking=True)
        bin_ids_c   = bin_ids.to(compute_device, non_blocking=True)

        for j in range(self.Nw):
            idx = (bin_ids_c == j).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            phase = torch.exp(1j * np.pi * w_centers_c[j] * r2)
            x_mod = x_pb.to(torch.complex64) * phase
            points = torch.stack([-v_c[idx], u_c[idx]], dim=0)
            vis_bin = pytorch_finufft.functional.finufft_type2(points, x_mod, isign=1, modeord=0)
            model_vis[idx] = vis_bin * (cell_size_sq)

        return model_vis

    # ---------- Objective & Loss ----------
    def objective(self, x, vis_data, device,
                  pb_list=None, grid_list=None, pb=None, grid_array=None,
                  cell_size=None, fftsd=None, fftbeam=None, tapper=None,
                  lambda_sd=0.0, fftkernel=None, beam_workers=4, verbose=False, **_):
        """
        All compute happens on `device`. Statics are staged per `stage_static`.
        """
        compute_device = torch.device(device)

        # Fan-out pb/grid if provided as stacks
        nbeam = len(vis_data.centers)
        if pb_list is None:
            if pb is None: raise ValueError("Need pb_list or pb array")
            pb_list = [pb[b] for b in range(nbeam)]
        if grid_list is None:
            if grid_array is None: raise ValueError("Need grid_list or grid_array")
            grid_list = [grid_array[b] for b in range(nbeam)]

        # Prepare statics (possibly on a different device)
        S = self._prepare_static(vis_data, pb_list, grid_list, cell_size, compute_device)
        loss_scalar = 0.0

        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        # χ² over all (chan, beam)
        for pack in S["chan_beam"]:
            c = pack["c"]; b = pack["b"]
            x_chan = x[c].to(compute_device)

            model_vis = self._forward_beam_cached(
                x_chan,
                pb=S["pb"][b], grid=S["grid"][b],
                u_rad=pack["u_rad"], v_rad=pack["v_rad"],
                ww=pack["ww"], w_centers=pack["w_centers"], bin_ids=pack["bin_ids"],
                cell_rad=S["cell_rad"],
                static_dev=S["static_device"],
                compute_device=compute_device
            )

            # residuals (move light arrays to compute device)
            vis_real = pack["I_re"].to(compute_device, non_blocking=True)
            vis_imag = pack["I_im"].to(compute_device, non_blocking=True)
            sig      = pack["sig"].to(compute_device,  non_blocking=True)

            residual_real = (model_vis.real - vis_real) / sig
            residual_imag = (model_vis.imag - vis_imag) / sig
            J = torch.sum(residual_real**2 + residual_imag**2)
            L = 0.5 * J
            L.backward(retain_graph=True)
            loss_scalar += float(L.item())

            if verbose:
                self._log_gpu(compute_device, f"[χ²] c={c} b={b} L={L.item():.3e}")

        # Optional SD term
        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t   = fftsd   if isinstance(fftsd,   torch.Tensor) else torch.from_numpy(fftsd)
            fftbeam_t = fftbeam if isinstance(fftbeam, torch.Tensor) else torch.from_numpy(fftbeam)
            tapper_t  = tapper  if isinstance(tapper,  torch.Tensor) else torch.from_numpy(tapper)

            fftsd_t   = fftsd_t.to(compute_device)
            fftbeam_t = fftbeam_t.to(compute_device)
            tapper_t  = tapper_t.to(compute_device)

            xfft2 = self._tfft2(x * tapper_t)
            model_sd = (cell_size**2) * xfft2 * fftbeam_t
            Lsd = 0.5 * (torch.nansum((model_sd.real - fftsd_t.real)**2) +
                         torch.nansum((model_sd.imag - fftsd_t.imag)**2)) * lambda_sd
            Lsd.backward(retain_graph=True)
            loss_scalar += float(Lsd.item())
            if verbose:
                self._log_gpu(compute_device, f"[SD] Lsd={Lsd.item():.3e}")

        # Optional Reg term
        if self.lambda_r > 0.0 and fftkernel is not None:
            fftkernel_t = fftkernel if isinstance(fftkernel, torch.Tensor) else torch.from_numpy(fftkernel)
            tapper_t    = tapper    if isinstance(tapper,   torch.Tensor) else torch.from_numpy(tapper)

            fftkernel_t = fftkernel_t.to(compute_device)
            tapper_t    = tapper_t.to(compute_device)

            xfft2 = self._tfft2(x * tapper_t)
            conv = (cell_size**2) * xfft2 * fftkernel_t
            Lr = 0.5 * torch.nansum(torch.abs(conv)**2) * self.lambda_r
            Lr.backward()
            loss_scalar += float(Lr.item())
            if verbose:
                self._log_gpu(compute_device, f"[Reg] Lr={Lr.item():.3e}")

        return torch.tensor(loss_scalar, device=compute_device)

    def loss(self, x_np, shape, device, vis_data, **kwargs):
        """
        Optimizer-friendly wrapper: returns (loss_scalar, grad_flat_numpy).
        """
        compute_device = torch.device(device)
        u = torch.from_numpy(x_np.reshape(shape)).to(compute_device).requires_grad_(True)

        L = self.objective(x=u, vis_data=vis_data, device=compute_device, **kwargs)
        grad = u.grad.detach().cpu().numpy().astype(x_np.dtype)

        if compute_device.type == 'cuda':
            alloc = torch.cuda.memory_allocated(compute_device) / 1024**2
            res   = torch.cuda.memory_reserved(compute_device)  / 1024**2
            tot   = torch.cuda.get_device_properties(compute_device).total_memory / 1024**2
            logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                  f"GPU: {alloc:.1f} MB allocated, {res:.1f} MB reserved, {tot:.0f} MB total")
        else:
            logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

        return float(L.item()), grad.ravel()


#------------------------------------------
#------------ ClassiCIViS3D ---------------
#------------------------------------------
class ClassicIViS3D(BaseModel):
    def __init__(self, lambda_r=1, Nw=None, use_2pi=True, conj_data=True):
        self.lambda_r = lambda_r
        self.Nw = None if (Nw is None or Nw <= 1) else (Nw if Nw % 2 == 1 else Nw+1)
        self.conj_data = conj_data  # match old pipeline that did np.conj(data)
        ...

    # --- helper to match old _lambda_to_radpix ---
    @staticmethod
    def _cellsize_arcsec_to_rad(cell_size_arcsec: float) -> float:
        return cell_size_arcsec * np.pi / (180.0 * 3600.0)

    
    def loss(self, x, shape, device, vis_data, **kwargs):
        """
        Optimizer-friendly loss wrapper (returns loss and grad as numpy).
        """
        u = x.reshape(shape)
        u = torch.from_numpy(u).to(device).requires_grad_(True)

        L = self.objective(x=u, vis_data=vis_data, device=device, **kwargs)
        grad = u.grad.cpu().numpy().astype(x.dtype)

        if torch.device(device).type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved  = torch.cuda.memory_reserved(device)  / 1024**2
            total     = torch.cuda.get_device_properties(device).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                f"GPU: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total"
            )
        else:
            logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

        return L.item(), grad.ravel()


    def forward_beam(self, x2d, pb, grid, uu, vv, ww, cell_size, device):
        # x2d -> 2D tensor [H,W]
        xt = torch.as_tensor(x2d, device=device)
        if xt.ndim == 1:
            side = int(np.sqrt(xt.numel()))
            xt = xt.view(side, side)
        if xt.ndim != 2:
            raise ValueError(f"x2d must be 2D (H,W), got shape {tuple(xt.shape)}")
        
        # 1) PB * reprojection
        xt = xt.unsqueeze(0).unsqueeze(0).float().to(device)              # [1,1,H,W]
        grid_t = torch.from_numpy(np.asarray(grid)).to(device).float()    # [1,H,W,2]
        repro = torch.nn.functional.grid_sample(xt, grid_t, mode='bilinear',
                                                align_corners=True).squeeze(0).squeeze(0)  # [H,W]
        x_pb = repro * torch.from_numpy(np.asarray(pb)).to(device).float()                 # [H,W]
        
        # 2) Units: arcsec -> radians; wavelengths -> radians/pixel with 2π
        cell_rad = cell_size * np.pi / (180.0 * 3600.0)
        u_radpix = torch.from_numpy(uu).to(device).float() * (2.0*np.pi*cell_rad)
        v_radpix = torch.from_numpy(vv).to(device).float() * (2.0*np.pi*cell_rad)
        
        # 3) Flat-sky NUFFT
        if self.Nw is None:
            points = torch.stack([-v_radpix, u_radpix], dim=0)            # [2, N]
            c = x_pb.to(torch.complex64)
            return (cell_size**2) * pytorch_finufft.functional.finufft_type2(
                points, c, isign=1, modeord=0
            )
        
        # 4) W-stacking path
        uua = u_radpix
        vva = v_radpix
        wwa = torch.from_numpy(ww).to(device).float()

        H, W = x_pb.shape
        lx = torch.linspace(-W/2, W/2 - 1, W, device=device) * cell_rad
        ly = torch.linspace(-H/2, H/2 - 1, H, device=device) * cell_rad
        l, m = torch.meshgrid(lx, ly, indexing='xy')
        r2 = l**2 + m**2
        
        model_vis = torch.empty_like(uua, dtype=torch.complex64)
        w_edges = torch.linspace(wwa.min(), wwa.max(), self.Nw + 1, device=device)
        w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
        bin_ids = torch.bucketize(wwa, w_edges) - 1
        bin_ids = torch.clamp(bin_ids, 0, self.Nw - 1)
        
        for j in range(self.Nw):
            idx = (bin_ids == j).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            
            points = torch.stack([-vva[idx], uua[idx]], dim=0)
            phase = torch.exp(1j * np.pi * w_centers[j] * r2) # r^2 uses radians
            x_mod = x_pb.to(torch.complex64) * phase
            
            vis_bin = (cell_size**2) * pytorch_finufft.functional.finufft_type2(
                points, x_mod, isign=1, modeord=0
            )
            model_vis[idx] = vis_bin
            
        return model_vis


    def objective(self, x, vis_data, device,
                  pb_list=None, grid_list=None, pb=None, grid_array=None,
                  cell_size=None, fftsd=None, fftbeam=None, tapper=None,
                  lambda_sd=0.0, fftkernel=None, beam_workers=4, verbose=False, **_):
        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        # fan-out pb/grid if given as full stacks
        nbeam = len(vis_data.centers)
        if pb_list is None:
            if pb is None: raise ValueError("Need pb_list or pb array")
            pb_list = [pb[b] for b in range(nbeam)]
        if grid_list is None:
            if grid_array is None: raise ValueError("Need grid_list or grid_array")
            grid_list = [grid_array[b] for b in range(nbeam)]

        loss_scalar = 0.0

        # --- χ² over all (chan, beam) ---
        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            x_chan = x[c] # (H,W)
            model_vis = self.forward_beam(x_chan, pb_list[b], grid_list[b], uu, vv, ww, cell_size, device)

            # match old pipeline: data were conjugated
            I_use = I.conj() if self.conj_data else I

            vis_real = torch.from_numpy(I_use.real).to(device)
            vis_imag = torch.from_numpy(I_use.imag).to(device)
            sig      = torch.from_numpy(sI).to(device)

            residual_real = (model_vis.real - vis_real) / sig
            residual_imag = (model_vis.imag - vis_imag) / sig
            J = torch.sum(residual_real**2 + residual_imag**2)
            L = 0.5 * J
            L.backward(retain_graph=True)
            loss_scalar += L.item()

            if verbose:
                print_gpu_memory(device)

        # --- SD term (unchanged) ---
        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t = torch.from_numpy(fftsd).to(device)
            fftbeam_t = torch.from_numpy(fftbeam).to(device)
            tapper_t = torch.from_numpy(tapper).to(device)
            xfft2 = tfft2(x * tapper_t)
            model_sd = (cell_size**2) * xfft2 * fftbeam_t
            Lsd = 0.5 * (torch.nansum((model_sd.real - fftsd_t.real)**2) +
                         torch.nansum((model_sd.imag - fftsd_t.imag)**2)) * lambda_sd
            Lsd.backward(retain_graph=True)
            loss_scalar += Lsd.item()

        # --- Reg (unchanged) ---
        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t = torch.from_numpy(tapper).to(device)
            fftkernel_t = torch.from_numpy(fftkernel).to(device)
            xfft2 = tfft2(x * tapper_t)
            conv = (cell_size**2) * xfft2 * fftkernel_t
            Lr = 0.5 * torch.nansum(torch.abs(conv)**2) * self.lambda_r
            Lr.backward()
            loss_scalar += Lr.item()

        return torch.tensor(loss_scalar)


#------------------------------------------
#------------ ClassiCIViS -----------------
#------------------------------------------
class ClassicIViS(BaseModel):
    """
    Classic IViS imaging model using visibility-domain loss and regularization.

    This class implements a forward operator and a loss function suitable for
    optimization via L-BFGS-B. The model supports CPU and GPU backends.
    """
            
    def __init__(self, lambda_r=1, Nw=None):
        self.lambda_r = lambda_r
        if self.lambda_r == 0:
            logger.warning("lambda_r = 0 - No spatial regularization.")
            
        if Nw is None or Nw <= 1:
            logger.info("Nw <= 1 or None, using flat-sky NUFFT (no w-stacking).")
            self.Nw = None
        else:
        # Force odd Nw
            if Nw % 2 == 0:
                Nw += 1
                logger.info(f"Adjusted Nw to closest odd number: {Nw}")
            self.Nw = Nw
            logger.info(f"Using w-stacking with Nw = {self.Nw}")
                    
            
    def loss(self, x, shape, device, **kwargs):
        """
        Compute total loss and gradient for optimizer (flattened interface).
        
        Parameters
        ----------
        x : np.ndarray
            Flattened image parameters.
        shape : tuple
            Target shape of the image (e.g. H, W).
        device : torch.device or str
            "cuda" or "cpu".
        **kwargs : dict
            All other named arguments needed for objective().
        
        Returns
        -------
        loss : float
            Scalar total loss.
        grad : np.ndarray
           Flattened gradient of the loss.
        """
        u = x.reshape(shape)
        u = torch.from_numpy(u).to(device).requires_grad_(True)
        
        L = self.objective(x=u, device=device, **kwargs)
        grad = u.grad.cpu().numpy().astype(x.dtype)
        
        if torch.device(device).type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved  = torch.cuda.memory_reserved(device)  / 1024**2
            total     = torch.cuda.get_device_properties(device).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                f"GPU: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total"
            )
        else:
            logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")
            
        return L.item(), grad.ravel()

    
    def forward(
            self, x, data, uu, vv, ww, pb, idmina, idmaxa, device, cell_size, grid_array
    ):
        """
        Simulate model visibilities (low-memory version using forward_beam).
        
        Returns
        -------
        model_vis : np.ndarray
        Modeled complex visibilities (NumPy, detached).
        """
        nvis = len(uu)
        model_vis = torch.zeros(nvis, dtype=torch.complex64, device=device)
        
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device)
        else:
            x = x.to(device)
            
        n_beams = len(idmina)
        for i in range(n_beams):
            idmin = idmina[i]
            idmax = idmin + idmaxa[i]
            
            vis = self.forward_beam(
                x, i, data, uu, vv, ww, pb, idmina, idmaxa, device, cell_size, grid_array
            )
            model_vis[idmin:idmax] = vis
            
        return model_vis.detach().cpu().numpy()


    def forward_beam(self, x, i, data, uu, vv, ww, pb, idmina, idmaxa, device, cell_size, grid_array):
        """
        Simulate model visibilities for a single beam.
        Supports w-stacking if self.Nw is enabled.
        """
        idmin = idmina[i]
        idmax = idmaxa[i]
        uua = torch.from_numpy(uu[idmin:idmin+idmax]).to(device)
        vva = torch.from_numpy(vv[idmin:idmin+idmax]).to(device)
        wwa = torch.from_numpy(ww[idmin:idmin+idmax]).to(device)
        pba = torch.from_numpy(pb[i]).to(device)
        grid = torch.from_numpy(grid_array[i]).to(device)
        
        # Prepare 2D (l, m) coordinates in radians
        H, W = pba.shape
        delta_rad = cell_size * np.pi / (180 * 3600)
        lx = torch.linspace(-W/2, W/2 - 1, W, device=device) * delta_rad
        ly = torch.linspace(-H/2, H/2 - 1, H, device=device) * delta_rad
        l, m = torch.meshgrid(lx, ly, indexing='xy')
        r2 = l**2 + m**2
        
        # Reproject input image
        input_tensor = format_input_tensor(x).float().to(device)
        reprojected_tensor = torch.nn.functional.grid_sample(
            input_tensor, grid, mode='bilinear', align_corners=True
        ).squeeze()
        x_pb = reprojected_tensor * pba  # real-valued

        # === Flat-sky version ===
        if self.Nw is None:
            points = torch.stack([-vva, uua], dim=0)
            c = x_pb.to(torch.complex64)
            model_vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(
                points, c, isign=1, modeord=0
            )
            return model_vis
        
        # === W-stacked version ===
        model_vis = torch.empty_like(uua, dtype=torch.complex64)
        
        # Bin visibilities by w
        w_edges = torch.linspace(wwa.min(), wwa.max(), self.Nw + 1, device=device)
        w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
        bin_ids = torch.bucketize(wwa, w_edges) - 1
        bin_ids = torch.clamp(bin_ids, 0, self.Nw - 1)
        
        for j in range(self.Nw):
            idx = (bin_ids == j).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            
            u_bin = uua[idx]
            v_bin = vva[idx]
            points = torch.stack([-v_bin, u_bin], dim=0)
            
            phase = torch.exp(1j * np.pi * w_centers[j] * r2)
            x_mod = x_pb.to(torch.complex64) * phase
            
            vis_bin = cell_size**2 * pytorch_finufft.functional.finufft_type2(
                points, x_mod, isign=1, modeord=0
            )
            
            model_vis[idx] = vis_bin
            
            del vis_bin, x_mod, phase, points, u_bin, v_bin, idx
            torch.cuda.empty_cache()
            
        return model_vis
        

    def objective(self, x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, sigma, fftsd, tapper,
                  lambda_sd, fftkernel, cell_size, grid_array, device, beam_workers=4, verbose=False):
        
        """
        Compute full imaging loss: χ² visibility + single-dish + regularization.

        Parameters
        ----------
        x : torch.Tensor
            Sky image tensor with gradients.
        beam : np.ndarray
            Gaussian restoring beam (not used directly).
        fftbeam : np.ndarray
            Beam FFT for SD regularization.
        data : np.ndarray
            Complex visibilities.
        uu, vv, ww : np.ndarray
            UVW coordinates.
        pb, grid_array : list of np.ndarray
            Beam and grid per pointing.
        idmina, idmaxa : list of int
            Visibility slices per beam.
        sigma : np.ndarray
            Per-visibility standard deviations.
        fftsd : np.ndarray
            FFT of SD image.
        tapper : np.ndarray
            Image tapering window.
        lambda_sd : float
            Weight of SD consistency term.
        fftkernel : np.ndarray
            FFT kernel for image regularization.
        cell_size : float
            Pixel size in arcsec.
        device : str
            "cpu" or "cuda".
        beam_workers : int
            Number of parallel CPU threads.

        Returns
        -------
        loss_scalar : torch.Tensor
            Scalar loss with gradients.
        """
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.zero_()

        beam = torch.from_numpy(beam).to(device)
        tapper = torch.from_numpy(tapper).to(device)

        loss_scalar = 0.0

        # -------------------------------
        # ---- Beam χ² --------- --------
        # -------------------------------
        n_beams = len(idmina)
        for i in range(n_beams):
            idmin = idmina[i]
            idmax = idmaxa[i]
            vis_real = torch.from_numpy(data.real[idmin:idmin+idmax]).to(device)
            vis_imag = torch.from_numpy(data.imag[idmin:idmin+idmax]).to(device)
            sig = torch.from_numpy(sigma[idmin:idmin+idmax]).to(device)

            model_vis = self.forward_beam(x, i, data, uu, vv, ww, pb, idmina, idmaxa, device, cell_size, grid_array)
                            
            residual_real = (model_vis.real - vis_real) / sig
            residual_imag = (model_vis.imag - vis_imag) / sig
            J = torch.sum(residual_real**2 + residual_imag**2)
            
            L = 0.5 * J
            L.backward(retain_graph=True)
            loss_scalar += L.item()

            del model_vis, residual_real, residual_imag, L, J, vis_real, vis_imag, sig
            torch.cuda.empty_cache()
            
            if verbose:
                print_gpu_memory(device)
            
        # ----------------------------
        # ---- Single Dish loss ------
        # ----------------------------
        fftsd_torch = torch.from_numpy(fftsd).to(device)
        fftbeam_torch = torch.from_numpy(fftbeam).to(device)
        xfft2 = tfft2(x * tapper)
        model_sd = cell_size**2 * xfft2 * fftbeam_torch
        J2 = torch.nansum((model_sd.real - fftsd_torch.real) ** 2)
        J22 = torch.nansum((model_sd.imag - fftsd_torch.imag) ** 2)
        Lsd = 0.5 * (J2 + J22) * lambda_sd
        Lsd.backward(retain_graph=True)
        loss_scalar += Lsd.item()
        
        # ----------------------------
        # ---- Regularization --------
        # ----------------------------
        fftkernel_torch = torch.from_numpy(fftkernel).to(device)
        xfft2 = tfft2(x * tapper)
        conv = cell_size**2 * xfft2 * fftkernel_torch
        R = torch.nansum(abs(conv) ** 2)
        Lr = 0.5 * R * self.lambda_r
        Lr.backward()
        loss_scalar += Lr.item()
        
        if verbose:
            print_gpu_memory(device)

        return torch.tensor(loss_scalar)




