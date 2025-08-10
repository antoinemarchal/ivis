import os
import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2

from ivis.logger import logger
from ivis.models.base import BaseModel
from ivis.models.utils.tensor_ops import format_input_tensor
from ivis.models.utils.gpu import print_gpu_memory


import os
import numpy as np
import torch
import pytorch_finufft


class ClassicIViS3DStagedFast:
    """
    Faster staged IViS:
      - Precomputes per-(chan,beam): scaled u/v, (w_edges, w_centers, bin_ids), and per-bin FINUFFT points.
      - Caches PB and grid on staging device.
      - χ² forward path runs in float32/complex64 for speed (cast from x on-the-fly).
      - SD/Reg paths are IDENTICAL to your original code (dtype follows x, e.g., float64/complex128).
      - Keeps original knobs: lambda_r, Nw, use_2pi, conj_data, stage_static, pin_h2d.
    """

    def __init__(self, lambda_r=1, Nw=None, use_2pi=True, conj_data=True,
                 stage_static='auto', pin_h2d=True):
        self.lambda_r  = float(lambda_r)
        self.Nw        = None if (Nw is None or Nw <= 1) else (Nw if (Nw % 2 == 1) else Nw + 1)
        self.use_2pi   = bool(use_2pi)
        self.conj_data = bool(conj_data)

        if stage_static not in ('auto','cpu','cuda'):
            raise ValueError("stage_static must be 'auto' | 'cpu' | 'cuda'")
        self.stage_static = stage_static
        self.pin_h2d = bool(pin_h2d)

        self._cache = {}     # key -> staged dict
        self._r2_cache = {}  # (H,W,cell_rad,device_str,real_dtype) -> r2 tensor

    # ---------- helpers ----------
    @staticmethod
    def _cellsize_arcsec_to_rad(cell_size_arcsec: float) -> float:
        return cell_size_arcsec * np.pi / (180.0 * 3600.0)

    def _choose_static_device(self, compute_device: torch.device) -> torch.device:
        if self.stage_static == 'cpu':  return torch.device('cpu')
        if self.stage_static == 'cuda': return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        return compute_device  # auto

    def clear_cache(self):
        self._cache.clear()
        self._r2_cache.clear()

    def _to_dev(self, arr, device, dtype=None, pin=False, contig=True):
        # tensor or numpy -> torch on device (optionally pinned)
        if isinstance(arr, torch.Tensor):
            t = arr if (dtype is None or arr.dtype == dtype) else arr.to(dtype)
            t = t.to(device, non_blocking=pin)
        else:
            a = np.asarray(arr)
            if pin and device.type == 'cuda':
                t = torch.from_numpy(a).pin_memory().to(device, non_blocking=True, dtype=dtype)
            else:
                t = torch.as_tensor(a, device=device, dtype=dtype)
        return t.contiguous() if contig else t

    def _get_r2(self, H, W, cell_rad, device, real_dtype):
        key = (H, W, float(cell_rad), str(device), str(real_dtype))
        r2 = self._r2_cache.get(key)
        if r2 is not None and r2.device == device:
            return r2
        with torch.no_grad():
            lx = torch.linspace(-W/2, W/2 - 1, W, device=device, dtype=real_dtype) * cell_rad
            ly = torch.linspace(-H/2, H/2 - 1, H, device=device, dtype=real_dtype) * cell_rad
            l, m = torch.meshgrid(lx, ly, indexing='xy')
            r2 = (l**2 + m**2).contiguous()
        self._r2_cache[key] = r2
        return r2

    # ---------- staging ----------
    def _prepare_static(self, vis_data, pb_list, grid_list, cell_size_arcsec, compute_device):
        static_dev = self._choose_static_device(torch.device(compute_device))
        key = (str(static_dev), id(vis_data), id(pb_list), id(grid_list),
               float(cell_size_arcsec), bool(self.use_2pi), self.Nw)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        cell_rad = self._cellsize_arcsec_to_rad(cell_size_arcsec)
        uv_scale = (2.0 * np.pi * cell_rad) if self.use_2pi else 1.0
        norm_area = (cell_size_arcsec ** 2)

        # Stage PB/grid as float32 (used by grid_sample)
        pb_t  = [self._to_dev(pb,   static_dev, torch.float32, pin=self.pin_h2d) for pb in pb_list]
        grd_t = [self._to_dev(grid, static_dev, torch.float32, pin=self.pin_h2d) for grid in grid_list]

        # Precompute per-(chan,beam) packs in *original iteration order*
        chan_beam = []
        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            # data/sigma (keep their original dtype, move on-demand later)
            I_use = I.conj() if self.conj_data else I
            I_re = self._to_dev(I_use.real, static_dev, None, pin=self.pin_h2d)
            I_im = self._to_dev(I_use.imag, static_dev, None, pin=self.pin_h2d)
            sig  = self._to_dev(sI,         static_dev, None, pin=self.pin_h2d)

            # u,v scaled (float32) for fast NUFFT
            u32 = (self._to_dev(uu, static_dev, torch.float32, pin=self.pin_h2d) * uv_scale).contiguous()
            v32 = (self._to_dev(vv, static_dev, torch.float32, pin=self.pin_h2d) * uv_scale).contiguous()

            # Optional W-stacking precompute: per-bin indices and FINUFFT points
            if self.Nw is not None:
                ww32 = self._to_dev(ww, static_dev, torch.float32, pin=self.pin_h2d)
                w_edges   = torch.linspace(ww32.min(), ww32.max(), self.Nw + 1, device=static_dev, dtype=torch.float32)
                w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
                bin_ids   = torch.bucketize(ww32, w_edges) - 1
                bin_ids.clamp_(0, self.Nw - 1)
                idx_list     = [(bin_ids == j).nonzero(as_tuple=True)[0] for j in range(self.Nw)]
                points_list  = [torch.stack([-v32[idx], u32[idx]], dim=0).contiguous()
                                if idx.numel() > 0 else None for idx in idx_list]
            else:
                w_centers = bin_ids = idx_list = points_list = None

            chan_beam.append({
                "c": c, "b": b,
                "I_re": I_re, "I_im": I_im, "sig": sig,
                "u32": u32, "v32": v32,
                "w_centers": w_centers, "bin_ids": bin_ids,
                "idx_list": idx_list, "points_list": points_list
            })

        out = {
            "static_device": static_dev,
            "cell_rad": cell_rad,
            "uv_scale": uv_scale,
            "norm_area": norm_area,
            "pb": pb_t,
            "grid": grd_t,
            "chan_beam": chan_beam
        }
        self._cache[key] = out
        return out

    # ---------- fast forward for one (chan,beam) using staged statics ----------
    def _forward_beam_fast(self, x2d, pb32, grid32, pack, cell_rad, norm_area, compute_device):
        # x2d can be float64; cast a WORKING COPY to float32 for speed (autograd will handle the cast)
        xt = (x2d if isinstance(x2d, torch.Tensor) else torch.as_tensor(x2d, device=compute_device))
        if xt.ndim == 1:
            side = int(np.sqrt(xt.numel())); xt = xt.view(side, side)
        if xt.ndim != 2:
            raise ValueError(f"x2d must be 2D (H,W), got {tuple(xt.shape)}")

        xt32 = xt.to(device=compute_device, dtype=torch.float32)

        # PB * reprojection (float32)
        grid32 = grid32.to(compute_device, non_blocking=True).contiguous()
        pb32   = pb32.to(compute_device,   non_blocking=True).contiguous()
        xt4    = xt32.unsqueeze(0).unsqueeze(0).contiguous()
        repro  = torch.nn.functional.grid_sample(xt4, grid32, mode='bilinear', align_corners=True)\
                    .squeeze(0).squeeze(0).contiguous()
        x_pb32 = (repro * pb32).contiguous()

        norm_c = torch.as_tensor(norm_area, device=compute_device, dtype=torch.float32).to(torch.complex64)

        # Flat-sky
        if self.Nw is None:
            points = torch.stack([-pack["v32"].to(compute_device, non_blocking=True),
                                   pack["u32"].to(compute_device, non_blocking=True)], dim=0).contiguous()
            vis = pytorch_finufft.functional.finufft_type2(points, x_pb32.to(torch.complex64),
                                                           isign=1, modeord=0)
            return (vis * norm_c).contiguous()

        # W-stacking: reuse precomputed per-bin points; phase per bin
        H, W = x_pb32.shape
        r2 = self._get_r2(H, W, cell_rad, compute_device, real_dtype=torch.float32)
        model_vis = torch.empty_like(pack["u32"], dtype=torch.complex64, device=compute_device)

        w_centers = pack["w_centers"].to(compute_device, non_blocking=True)
        points_list = pack["points_list"]
        idx_list    = pack["idx_list"]

        x_c = x_pb32.to(torch.complex64).contiguous()

        for j in range(self.Nw):
            idx = idx_list[j]
            if idx is None or idx.numel() == 0:
                continue
            pts = points_list[j].to(compute_device, non_blocking=True)
            phase = torch.exp(1j * np.pi * w_centers[j] * r2).to(torch.complex64)  # r in radians
            x_mod = (x_c * phase).contiguous()
            vis_j = pytorch_finufft.functional.finufft_type2(pts, x_mod, isign=1, modeord=0)
            model_vis[idx] = vis_j * norm_c

        return model_vis

    # ---------- objective ----------
    def objective(self, x, vis_data, device,
                  pb_list=None, grid_list=None, pb=None, grid_array=None,
                  cell_size=None, fftsd=None, fftbeam=None, tapper=None,
                  lambda_sd=0.0, fftkernel=None, beam_workers=1, verbose=False, **_):
        compute_device = torch.device(device)
        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        # expand PB/grid like Classic
        nbeam = len(vis_data.centers)
        if pb_list is None:
            if pb is None: raise ValueError("Need pb_list or pb array")
            pb_list = [pb[b] for b in range(nbeam)]
        if grid_list is None:
            if grid_array is None: raise ValueError("Need grid_list or grid_array")
            grid_list = [grid_array[b] for b in range(nbeam)]

        S = self._prepare_static(vis_data, pb_list, grid_list, cell_size, compute_device)

        loss_scalar = 0.0

        # --- χ² over staged (chan,beam) in captured order ---
        for pack in S["chan_beam"]:
            c = pack["c"]; b = pack["b"]
            x_cb = x[c].to(compute_device)

            model_vis = self._forward_beam_fast(
                x2d=x_cb,
                pb32=S["pb"][b], grid32=S["grid"][b],
                pack=pack,
                cell_rad=S["cell_rad"], norm_area=S["norm_area"],
                compute_device=compute_device
            )

            # Data/sigma to compute_device, keep their original dtype
            vr = pack["I_re"].to(compute_device, non_blocking=True)
            vi = pack["I_im"].to(compute_device, non_blocking=True)
            sg = pack["sig"].to(compute_device,  non_blocking=True)

            # Residuals (let torch promote to highest real dtype, typically float64)
            res_r = (model_vis.real - vr) / sg
            res_i = (model_vis.imag - vi) / sg
            Lchi = 0.5 * torch.sum(res_r**2 + res_i**2)
            Lchi.backward(retain_graph=True)
            loss_scalar += float(Lchi.item())

            if verbose and compute_device.type == 'cuda':
                alloc = torch.cuda.memory_allocated(compute_device) / 1024**2
                resv  = torch.cuda.memory_reserved(compute_device)  / 1024**2
                tot   = torch.cuda.get_device_properties(compute_device).total_memory / 1024**2
                print(f"[χ²] c={c} b={b} L={Lchi.item():.3e} | GPU: {alloc:.1f} MB ({resv:.1f} MB reserved / {tot:.0f} MB)")

        # --- SD term (UNCHANGED from your working code) ---
        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t   = torch.from_numpy(fftsd).to(compute_device)
            fftbeam_t = torch.from_numpy(fftbeam).to(compute_device)
            tapper_t  = torch.from_numpy(tapper).to(compute_device)
            # NOTE: use your original tfft2 helper (must be in scope)
            xfft2     = tfft2(x * tapper_t)
            model_sd  = (cell_size**2) * xfft2 * fftbeam_t
            Lsd = 0.5 * (torch.nansum((model_sd.real - fftsd_t.real)**2) +
                         torch.nansum((model_sd.imag - fftsd_t.imag)**2)) * lambda_sd
            Lsd.backward(retain_graph=True)
            loss_scalar += float(Lsd.item())

            if verbose:
                print(f"[SD]  Lsd={float(Lsd.item()):.3e}")

        # --- Regularization term (UNCHANGED from your working code) ---
        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t    = torch.from_numpy(tapper).to(compute_device)
            fftkernel_t = torch.from_numpy(fftkernel).to(compute_device)
            xfft2       = tfft2(x * tapper_t)
            conv        = (cell_size**2) * xfft2 * fftkernel_t
            Lr = 0.5 * torch.nansum(torch.abs(conv)**2) * self.lambda_r
            Lr.backward()
            loss_scalar += float(Lr.item())

            if verbose:
                print(f"[Reg] lambda_r={self.lambda_r}  Lr={float(Lr.item()):.3e}")

        return torch.tensor(loss_scalar, device=compute_device)

    # ---------- optimizer wrapper ----------
    def loss(self, x, shape, device, vis_data, **kwargs):
        """
        Keep x's dtype (e.g., float64) so SD/Reg stay in the original precision.
        Forward/χ² will cast a working copy to float32 internally for speed.
        """
        compute_device = torch.device(device)
        u_np = x.reshape(shape)
        u = torch.from_numpy(u_np).to(device=compute_device).requires_grad_(True)

        L = self.objective(x=u, vis_data=vis_data, device=compute_device, **kwargs)
        grad = u.grad.detach().cpu().numpy().astype(x.dtype)

        if compute_device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(compute_device) / 1024**2
            reserved  = torch.cuda.memory_reserved(compute_device)  / 1024**2
            total     = torch.cuda.get_device_properties(compute_device).total_memory / 1024**2
            print(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                  f"GPU: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total")
        else:
            print(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

        return float(L.item()), grad.ravel()


class ClassicIViS3DStaged:
    """
    Minimal staged IViS with χ² + optional SD + regularization:
      - Stages only PB/grids + constants (cell_rad, uv_scale, norm).
      - Preserves Classic loop order & numerics (float32 forward, complex64 NUFFT).
      - Keeps original keywords: lambda_r, Nw, use_2pi, conj_data, stage_static, pin_h2d.
    """

    def __init__(self, lambda_r=1, Nw=None, use_2pi=True, conj_data=True,
                 stage_static='auto', pin_h2d=True):
        self.lambda_r  = float(lambda_r)
        self.Nw        = None if (Nw is None or Nw <= 1) else (Nw if Nw % 2 == 1 else Nw + 1)
        self.use_2pi   = bool(use_2pi)
        self.conj_data = bool(conj_data)

        if stage_static not in ('auto','cpu','cuda'):
            raise ValueError("stage_static must be 'auto' | 'cpu' | 'cuda'")
        self.stage_static = stage_static
        self.pin_h2d = bool(pin_h2d)

        self._statics = None
        self._r2_cache = {}

    # ---------- helpers ----------
    @staticmethod
    def _cellsize_arcsec_to_rad(cell_size_arcsec: float) -> float:
        return cell_size_arcsec * np.pi / (180.0 * 3600.0)

    @staticmethod
    def _tfft2(x):
        # fftshift(fft2(ifftshift(x)))
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))

    def _choose_static_device(self, compute_device: torch.device) -> torch.device:
        if self.stage_static == 'cpu':
            return torch.device('cpu')
        if self.stage_static == 'cuda':
            return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        return compute_device

    def clear_cache(self):
        self._statics = None
        self._r2_cache.clear()

    def _to_device_once(self, arr, device, dtype=None, pin=False):
        if isinstance(arr, torch.Tensor):
            t = arr if (dtype is None or arr.dtype == dtype) else arr.to(dtype)
            return t.to(device, non_blocking=pin).requires_grad_(False)
        np_arr = np.asarray(arr)
        if pin and device.type == 'cuda':
            return torch.from_numpy(np_arr).pin_memory().to(device, non_blocking=True, dtype=dtype).requires_grad_(False)
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

    # ---------- staging (PB/grids/constants only) ----------
    def _stage(self, pb_list, grid_list, cell_size_arcsec, compute_device):
        static_dev = self._choose_static_device(torch.device(compute_device))
        key = (str(static_dev), id(pb_list), id(grid_list), float(cell_size_arcsec), bool(self.use_2pi), self.Nw)
        if self._statics is not None and self._statics["key"] == key:
            return self._statics

        cell_rad  = self._cellsize_arcsec_to_rad(cell_size_arcsec)
        uv_scale  = (2.0*np.pi*cell_rad) if self.use_2pi else 1.0
        norm_area = (cell_size_arcsec**2)  # arcsec^2 like Classic

        pbs   = [self._to_device_once(pb,   static_dev, torch.float32, pin=self.pin_h2d).contiguous() for pb in pb_list]
        grids = [self._to_device_once(grid, static_dev, torch.float32, pin=self.pin_h2d).contiguous() for grid in grid_list]

        self._statics = {"key": key, "static_device": static_dev, "cell_rad": cell_rad,
                         "uv_scale": uv_scale, "norm": norm_area, "pb": pbs, "grid": grids}
        return self._statics

    # ---------- forward per (chan, beam) ----------
    def _forward_beam(self, x2d, pb32, grid32, uu, vv, ww,
                      cell_rad, uv_scale, norm_area, compute_device):
        # x -> float32 [H,W]
        xt = x2d if isinstance(x2d, torch.Tensor) else torch.as_tensor(x2d)
        if xt.ndim == 1:
            side = int(np.sqrt(xt.numel())); xt = xt.view(side, side)
        if xt.ndim != 2:
            raise ValueError(f"x2d must be 2D (H,W), got {tuple(xt.shape)}")
        xt = xt.to(compute_device, dtype=torch.float32, non_blocking=True)

        # PB * reprojection
        grid32 = grid32.to(compute_device, dtype=torch.float32, non_blocking=True).contiguous()
        pb32   = pb32.to(compute_device,   dtype=torch.float32, non_blocking=True).contiguous()
        xt4    = xt.unsqueeze(0).unsqueeze(0).contiguous()
        repro  = torch.nn.functional.grid_sample(xt4, grid32, mode='bilinear', align_corners=True)\
                    .squeeze(0).squeeze(0).contiguous()
        x_pb32 = (repro * pb32).contiguous()

        # u,v scaled
        u32 = (self._to_device_once(uu, compute_device, torch.float32) * uv_scale).contiguous()
        v32 = (self._to_device_once(vv, compute_device, torch.float32) * uv_scale).contiguous()
        norm_c64 = torch.as_tensor(norm_area, device=compute_device, dtype=torch.float32).to(torch.complex64)

        if self.Nw is None:
            points = torch.stack([-v32, u32], dim=0).contiguous()
            vis    = pytorch_finufft.functional.finufft_type2(points, x_pb32.to(torch.complex64), isign=1, modeord=0)
            return (vis * norm_c64).contiguous()

        # W-stacking
        ww32 = self._to_device_once(ww, compute_device, torch.float32)
        H, W = x_pb32.shape
        r2   = self._get_r2(H, W, cell_rad, compute_device).to(torch.float32)

        w_edges   = torch.linspace(ww32.min(), ww32.max(), self.Nw + 1, device=compute_device, dtype=torch.float32)
        w_centers = 0.5*(w_edges[:-1] + w_edges[1:])
        bin_ids   = torch.bucketize(ww32, w_edges) - 1
        bin_ids.clamp_(0, self.Nw - 1)

        out   = torch.empty_like(u32, dtype=torch.complex64)
        x_c64 = x_pb32.to(torch.complex64).contiguous()

        for j in range(self.Nw):
            idx = (bin_ids == j).nonzero(as_tuple=True)[0]
            if idx.numel() == 0: continue
            pts   = torch.stack([-v32[idx], u32[idx]], dim=0).contiguous()
            phase = torch.exp(1j * np.pi * w_centers[j] * r2).to(torch.complex64)  # use np.pi for parity
            x_mod = (x_c64 * phase).contiguous()
            vis_j = pytorch_finufft.functional.finufft_type2(pts, x_mod, isign=1, modeord=0)
            out[idx] = vis_j * norm_c64

        return out

    # ---------- objective (χ² + optional SD + Reg) ----------
    def objective(self, x, vis_data, device,
                  pb_list=None, grid_list=None, pb=None, grid_array=None,
                  cell_size=None, fftsd=None, fftbeam=None, tapper=None,
                  lambda_sd=0.0, fftkernel=None, beam_workers=1, verbose=False, **_):
        compute_device = torch.device(device)
        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        # expand PB/grid
        nbeam = len(vis_data.centers)
        if pb_list is None:
            if pb is None: raise ValueError("Need pb_list or pb array")
            pb_list = [pb[b] for b in range(nbeam)]
        if grid_list is None:
            if grid_array is None: raise ValueError("Need grid_list or grid_array")
            grid_list = [grid_array[b] for b in range(nbeam)]

        S = self._stage(pb_list, grid_list, cell_size, compute_device)

        loss_scalar = 0.0

        # --- χ² ---
        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            x_cb = x[c]

            I_use = I.conj() if self.conj_data else I
            vis_r = torch.as_tensor(np.asarray(I_use.real), device=compute_device)  # likely float64
            vis_i = torch.as_tensor(np.asarray(I_use.imag), device=compute_device)
            sig   = torch.as_tensor(np.asarray(sI),        device=compute_device)

            model_vis = self._forward_beam(
                x_cb, S["pb"][b], S["grid"][b], uu, vv, ww,
                S["cell_rad"], S["uv_scale"], S["norm"], compute_device
            )

            res_r = (model_vis.real - vis_r) / sig
            res_i = (model_vis.imag - vis_i) / sig
            Lchi = 0.5 * torch.sum(res_r**2 + res_i**2)
            Lchi.backward(retain_graph=True)
            loss_scalar += float(Lchi.item())

            # --- SD term (unchanged) ---
            if lambda_sd > 0.0 and fftsd is not None:
                fftsd_t   = torch.from_numpy(fftsd).to(device)
                fftbeam_t = torch.from_numpy(fftbeam).to(device)
                tapper_t  = torch.from_numpy(tapper).to(device)
                xfft2     = tfft2(x * tapper_t)              # <- use the SAME helper as Classic
                model_sd  = (cell_size**2) * xfft2 * fftbeam_t
                Lsd = 0.5 * (torch.nansum((model_sd.real - fftsd_t.real)**2) +
                             torch.nansum((model_sd.imag - fftsd_t.imag)**2)) * lambda_sd
                Lsd.backward(retain_graph=True)
                loss_scalar += Lsd.item()
                
            # --- Reg (unchanged) ---
            if self.lambda_r > 0.0 and fftkernel is not None:
                tapper_t    = torch.from_numpy(tapper).to(device)
                fftkernel_t = torch.from_numpy(fftkernel).to(device)
                xfft2       = tfft2(x * tapper_t)            # <- same helper
                conv        = (cell_size**2) * xfft2 * fftkernel_t
                Lr = 0.5 * torch.nansum(torch.abs(conv)**2) * self.lambda_r
                Lr.backward()
                loss_scalar += Lr.item()
                
        return torch.tensor(loss_scalar, device=compute_device)

    # ---------- optimizer wrapper ----------
    def loss(self, x, shape, device, vis_data, **kwargs):
        compute_device = torch.device(device)
        # Classic forward uses float32; keep that for the image param tensor too
        u = torch.from_numpy(x.reshape(shape)).to(device=compute_device).requires_grad_(True)

        L = self.objective(x=u, vis_data=vis_data, device=compute_device, **kwargs)
        grad = u.grad.detach().cpu().numpy().astype(x.dtype)

        if compute_device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(compute_device) / 1024**2
            reserved  = torch.cuda.memory_reserved(compute_device)  / 1024**2
            total     = torch.cuda.get_device_properties(compute_device).total_memory / 1024**2
            print(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)} | "
                  f"GPU: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved, {total:.2f} MB total")
        else:
            print(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

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




