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

# ------------------------------------------
#------------ ClassiCIViS3D ----------------
#-------------------------------------------
from torch.utils.checkpoint import checkpoint as _checkpoint

class ClassicIViS3D_old(BaseModel):
    """
    Classic IViS 3D (flat-sky optimized, memory-safe):
      - Stages PB & grids to device once.
      - Stages per-(chan,beam): points (scaled), Ir, Ii, sig.
      - Reprojects beams in micro-batches (beam_batch) to limit VRAM.
      - Optional per-chunk backward and gradient checkpointing.
      - One-line knobs: beam_batch, chunk_backward, gradient_checkpoint, nufft_eps.
    Numerics preserved: 2π uv scaling, (cell_size**2), align_corners=True, conj_data.
    """

    def __init__(
        self,
        lambda_r=1,
        Nw=None,
        use_2pi=True,
        conj_data=True,
        beam_batch=2,
        chunk_backward=True,
        gradient_checkpoint=False,
        nufft_eps=None,
    ):
        self.lambda_r  = lambda_r
        self.Nw        = None if (Nw is None or Nw <= 1) else (Nw if (Nw % 2 == 1) else Nw + 1)
        self.use_2pi   = bool(use_2pi)
        self.conj_data = bool(conj_data)

        # Performance/memory knobs
        self.beam_batch = int(max(1, beam_batch))
        self.chunk_backward = bool(chunk_backward)
        self.gradient_checkpoint = bool(gradient_checkpoint)
        self.nufft_eps = nufft_eps  # None -> FINUFFT default (≈1e-6 for single)

        # Caches
        self._static = {"device": None, "pb": None, "grid": None}  # staged per-beam statics
        self._cb_pack = None                                       # staged per-(chan,beam) tensors (flat-sky)
        self._r2_cache = {}                                        # used only if Nw is not None

    # --- helper (matches old _lambda_to_radpix) ---
    @staticmethod
    def _cellsize_arcsec_to_rad(cell_size_arcsec: float) -> float:
        return cell_size_arcsec * np.pi / (180.0 * 3600.0)

    # ---------- staging: PB & grids ----------
    def stage_static(self, device, pb, grid_array):
        """
        Copy PB and grid to `device` once. Call again only if inputs/device change.
        pb: [nbeam, H, W]
        grid_array: [nbeam, H, W, 2] (values in [-1,1], align_corners=True)
        """
        nbeam = len(pb)
        pb_t, grid_t = [], []
        for b in range(nbeam):
            pb_t.append(torch.from_numpy(np.asarray(pb[b], dtype=np.float32)).to(device))
            grid_t.append(torch.from_numpy(np.asarray(grid_array[b], dtype=np.float32)).to(device))
        self._static = {"device": str(device), "pb": pb_t, "grid": grid_t}

    def _ensure_staged_static(self, device, pb=None, grid_array=None):
        need = (
            self._static["pb"] is None or
            self._static["grid"] is None or
            self._static["device"] != str(device)
        )
        if need:
            if pb is None or grid_array is None:
                raise ValueError("Pass `pb` and `grid_array` on first call to stage them.")
            self.stage_static(device, pb, grid_array)

    # ---------- staging: per-(chan,beam) tensors for FLAT-SKY ----------
    def stage_dynamic_flat(self, vis_data, device, cell_size):
        """
        Stage per-(chan,beam) data once for flat-sky mode:
        - points = stack(-vv, uu) * scale  (scale = 2π * cell_rad)
        - Ir, Ii, sig
        """
        cell_rad = self._cellsize_arcsec_to_rad(cell_size)
        scale = (2.0 * np.pi * cell_rad) if self.use_2pi else cell_rad

        cb_pack = {}
        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            uu_t = torch.from_numpy(uu).to(device, dtype=torch.float32)
            vv_t = torch.from_numpy(vv).to(device, dtype=torch.float32)

            I_use = I.conj() if self.conj_data else I
            Ir_t = torch.from_numpy(I_use.real).to(device, dtype=torch.float32)
            Ii_t = torch.from_numpy(I_use.imag).to(device, dtype=torch.float32)
            sig_t = torch.from_numpy(sI).to(device, dtype=torch.float32)

            points_scaled = torch.stack([-vv_t, uu_t], dim=0) * scale  # [2, N]
            cb_pack[(c, b)] = {"points": points_scaled, "Ir": Ir_t, "Ii": Ii_t, "sig": sig_t}

        self._cb_pack = cb_pack

    # ---------- (optional) r^2 cache if you ever enable w-term ----------
    def _r2_for(self, H, W, cell_size_arcsec, device):
        key = (int(H), int(W), float(cell_size_arcsec))
        r2 = self._r2_cache.get(key)
        if r2 is not None and r2.device == torch.device(device):
            return r2
        cell_rad = self._cellsize_arcsec_to_rad(cell_size_arcsec)
        lx = torch.linspace(-W/2, W/2 - 1, W, device=device) * cell_rad
        ly = torch.linspace(-H/2, H/2 - 1, H, device=device) * cell_rad
        l, m = torch.meshgrid(lx, ly, indexing='xy')
        r2 = l*l + m*m
        self._r2_cache[key] = r2
        return r2

    # ---------- micro-batched reprojection ----------
    def _reproject_batch(self, x_chan, device, beam_indices):
        """
        Reproject a single channel image to multiple beams in one call.
        beam_indices: list[int], returns x_pb tensors in the same order.
        """
        x = torch.as_tensor(x_chan, device=device).float()
        if x.ndim == 1:
            side = int(np.sqrt(x.numel()))
            x = x.view(side, side)
        x_bchw = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        grids = [self._static["grid"][b] for b in beam_indices]  # each [1,H,W,2]
        pbs   = [self._static["pb"][b]   for b in beam_indices]  # each [H,W]

        grid_batch = torch.stack(grids, dim=0).squeeze(1)        # [B,H,W,2]
        x_batch    = x_bchw.repeat(len(grids), 1, 1, 1)          # [B,1,H,W]

        repro_batch = torch.nn.functional.grid_sample(
            x_batch, grid_batch, mode='bilinear', align_corners=True
        ).squeeze(1)                                             # [B,H,W]

        xpb_batch = repro_batch * torch.stack(pbs, dim=0)        # [B,H,W]
        return xpb_batch

    # ---------- flat-sky NUFFT from precomputed x_pb and points ----------
    def _nufft_flat_from_xpb(self, x_pb, points_scaled):
        img_c = x_pb.to(torch.complex64)  # keep as complex64 for speed (default eps~1e-6)
        if self.nufft_eps is None:
            return pytorch_finufft.functional.finufft_type2(
                points_scaled, img_c, isign=1, modeord=0
            )
        else:
            return pytorch_finufft.functional.finufft_type2(
                points_scaled, img_c, isign=1, modeord=0, eps=self.nufft_eps
            )

    # ---------- tiny beam block (optionally checkpointed) ----------
    def _beam_block(self, x_pb, points, Ir, Ii, sig, cell_size):
        mv = (cell_size**2) * self._nufft_flat_from_xpb(x_pb, points)
        rr = (mv.real - Ir) / sig
        ri = (mv.imag - Ii) / sig
        return 0.5 * torch.sum(rr*rr + ri*ri)

    # ---------- objective ----------
    def objective(self, x, vis_data, device,
                  pb_list=None, grid_list=None, pb=None, grid_array=None,
                  cell_size=None, fftsd=None, fftbeam=None, tapper=None,
                  lambda_sd=0.0, fftkernel=None, beam_workers=4, verbose=False, **kwargs):

        if cell_size is None:
            raise ValueError("`cell_size` (arcsec) is required.")

        # Stage statics
        if pb is None or grid_array is None:
            if pb_list is not None and grid_list is not None:
                pb, grid_array = pb_list, grid_list
            else:
                raise ValueError("Pass `pb` and `grid_array` on first call.")
        self._ensure_staged_static(device, pb=pb, grid_array=grid_array)

        # Stage dynamics (flat-sky)
        if self.Nw is None and self._cb_pack is None:
            self.stage_dynamic_flat(vis_data, device, cell_size)

        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        # Group (c,b) by channel so we can micro-batch beams
        from collections import defaultdict
        by_chan = defaultdict(list)
        for (c, b), pack in self._cb_pack.items() if self.Nw is None else []:
            by_chan[c].append((b, pack))

        Ltot = torch.zeros((), device=device, dtype=torch.float32)

        # -------- χ² term (flat-sky, micro-batched beams) --------
        if self.Nw is None:
            bb = int(max(1, kwargs.get("beam_batch", self.beam_batch)))
            do_chunk_back = bool(kwargs.get("chunk_backward", self.chunk_backward))
            use_ckpt = bool(kwargs.get("gradient_checkpoint", self.gradient_checkpoint))

            for c, items in by_chan.items():
                # Sort by beam id; micro-batch consecutive beams for efficient stacking
                items.sort(key=lambda z: z[0])

                for i0 in range(0, len(items), bb):
                    chunk = items[i0:i0+bb]
                    b_idx = [b for (b, _) in chunk]
                    packs = [p for (_, p) in chunk]

                    # Reproject this channel for these beams
                    xpb_batch = self._reproject_batch(x[c], device, b_idx)  # [bb, H, W]

                    # Accumulate loss for this micro-batch
                    L_chunk = torch.zeros((), device=device, dtype=torch.float32)
                    for k, pack in enumerate(packs):
                        x_pb = xpb_batch[k]

                        if use_ckpt:
                            # Wrap args as tensors only
                            cs_t = torch.tensor(cell_size, device=device, dtype=torch.float32)
                            Lk = _checkpoint(self._beam_block, x_pb, pack["points"],
                                             pack["Ir"], pack["Ii"], pack["sig"], cs_t)
                        else:
                            Lk = self._beam_block(x_pb, pack["points"], pack["Ir"], pack["Ii"], pack["sig"], cell_size)
                        L_chunk = L_chunk + Lk

                    # Backprop now or later?
                    if do_chunk_back:
                        retain = not (i0 + bb >= len(items))  # keep graph if more chunks remain
                        L_chunk.backward(retain_graph=retain)
                        Ltot = Ltot + L_chunk.detach()
                        # drop big tensors early
                        del xpb_batch, L_chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        # Accumulate; backprop once after all chunks of all channels
                        Ltot = Ltot + L_chunk
                        del xpb_batch, L_chunk

        else:
            # W-stack fallback (not optimized here)
            for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
                xt = torch.as_tensor(x[c], device=device).float()
                if xt.ndim == 1:
                    side = int(np.sqrt(xt.numel()))
                    xt = xt.view(side, side)
                xt = xt.unsqueeze(0).unsqueeze(0)
                grid_t = self._static["grid"][b]
                pb_t   = self._static["pb"][b]
                repro = torch.nn.functional.grid_sample(xt, grid_t, mode='bilinear',
                                                        align_corners=True).squeeze(0).squeeze(0)
                x_pb = repro * pb_t

                cell_rad = self._cellsize_arcsec_to_rad(cell_size)
                scale = (2.0 * np.pi * cell_rad) if self.use_2pi else cell_rad
                uu_t = torch.from_numpy(uu).to(device, dtype=torch.float32) * scale
                vv_t = torch.from_numpy(vv).to(device, dtype=torch.float32)
                points = torch.stack([-vv_t * scale, uu_t], dim=0)
                I_use = I.conj() if self.conj_data else I
                Ir = torch.from_numpy(I_use.real).to(device, dtype=torch.float32)
                Ii = torch.from_numpy(I_use.imag).to(device, dtype=torch.float32)
                sig = torch.from_numpy(sI).to(device, dtype=torch.float32)
                mv = (cell_size**2) * self._nufft_flat_from_xpb(x_pb, points)
                rr = (mv.real - Ir) / sig
                ri = (mv.imag - Ii) / sig
                Ltot = Ltot + 0.5 * torch.sum(rr*rr + ri*ri)

        # -------- SD term --------
        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t   = torch.from_numpy(fftsd).to(device)
            fftbeam_t = torch.from_numpy(fftbeam).to(device)
            tapper_t  = torch.from_numpy(tapper).to(device)
            xfft2 = tfft2(x * tapper_t)
            model_sd = (cell_size**2) * xfft2 * fftbeam_t
            Lsd = 0.5 * (torch.nansum((model_sd.real - fftsd_t.real)**2) +
                         torch.nansum((model_sd.imag - fftsd_t.imag)**2)) * lambda_sd
            Ltot = Ltot + Lsd

        # -------- Reg term --------
        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t = torch.from_numpy(tapper).to(device)
            fftkernel_t = torch.from_numpy(fftkernel).to(device)
            xfft2 = tfft2(x * tapper_t)
            conv = (cell_size**2) * xfft2 * fftkernel_t
            Lr = 0.5 * torch.nansum(torch.abs(conv)**2) * self.lambda_r
            Ltot = Ltot + Lr

        # Backprop if we accumulated without per-chunk backward
        if not self.chunk_backward:
            Ltot.backward()

        return Ltot

    # ---------- optimizer-friendly wrapper ----------
    def loss(self, x, shape, device, vis_data, **kwargs):
        """
        Returns (loss_value, grad_flat) as numpy (LBFGS-friendly).
        """
        u = x.reshape(shape)
        u = torch.from_numpy(u).to(device).requires_grad_(True)

        # Stage statics first call; dynamic staging happens in objective
        if self._static["pb"] is None:
            self._ensure_staged_static(
                device=device,
                pb=kwargs.get('pb', None) or kwargs.get('pb_list', None),
                grid_array=kwargs.get('grid_array', None) or kwargs.get('grid_list', None),
            )

        L = self.objective(x=u, vis_data=vis_data, device=device, **kwargs)

        if u.grad is None:
            L.backward()

        grad = u.grad.detach().cpu().numpy().astype(x.dtype)

        if torch.device(device).type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved  = torch.cuda.memory_reserved(device)  / 1024**2
            total     = torch.cuda.get_device_properties(device).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Iter cost: {np.format_float_scientific(L.item(), precision=6)} | "
                f"GPU: {allocated:.2f} MB alloc, {reserved:.2f} MB res, {total:.2f} MB total"
            )
        else:
            logger.info(f"[PID {os.getpid()}] Iter cost: {np.format_float_scientific(L.item(), precision=6)} (device: cpu)")

        return float(L.item()), grad.ravel()


# ------------------------------------------
#------------ classicivis3d ----------------
#-------------------------------------------
class ClassicIViS3D_old1(BaseModel):
    """
    Classic IViS 3D with flat-sky optimizations:
      - Stages PB & grids to device once.
      - Stages per-(chan,beam) arrays once (Ir, Ii, sig, and points).
      - Batches reprojection across beams: one grid_sample per channel.
      - Single backward() per objective call.
    Numerics preserved: 2π uv scaling, (cell_size**2), align_corners=True, conj_data.
    """

    def __init__(self, lambda_r=1, Nw=None, use_2pi=True, conj_data=True):
        self.lambda_r  = lambda_r
        self.Nw        = None if (Nw is None or Nw <= 1) else (Nw if (Nw % 2 == 1) else Nw + 1)
        self.use_2pi   = bool(use_2pi)
        self.conj_data = bool(conj_data)

        # Caches
        self._static = {"device": None, "pb": None, "grid": None}  # staged per-beam statics
        self._cb_pack = None                                       # staged per-(chan,beam) tensors (flat-sky)
        self._r2_cache = {}                                        # used only if Nw is not None
        self._chan_cache = {"c": None, "xpb_list": None}           # batched reprojection cache per-channel

    # --- helper (matches old _lambda_to_radpix) ---
    @staticmethod
    def _cellsize_arcsec_to_rad(cell_size_arcsec: float) -> float:
        return cell_size_arcsec * np.pi / (180.0 * 3600.0)

    # ---------- staging: PB & grids ----------
    def stage_static(self, device, pb, grid_array):
        """
        Copy PB and grid to `device` once. Call again only if inputs/device change.
        pb: [nbeam, H, W]
        grid_array: [nbeam, H, W, 2] (values in [-1,1], align_corners=True)
        """
        nbeam = len(pb)
        pb_t, grid_t = [], []
        for b in range(nbeam):
            pb_t.append(torch.from_numpy(np.asarray(pb[b], dtype=np.float32)).to(device))
            grid_t.append(torch.from_numpy(np.asarray(grid_array[b], dtype=np.float32)).to(device))
        self._static = {"device": str(device), "pb": pb_t, "grid": grid_t}

    def _ensure_staged_static(self, device, pb=None, grid_array=None):
        need = (
            self._static["pb"] is None or
            self._static["grid"] is None or
            self._static["device"] != str(device)
        )
        if need:
            if pb is None or grid_array is None:
                raise ValueError("Pass `pb` and `grid_array` on first call to stage them.")
            self.stage_static(device, pb, grid_array)

    # ---------- staging: per-(chan,beam) tensors for FLAT-SKY ----------
    def stage_dynamic_flat(self, vis_data, device, cell_size):
        """
        Stage per-(chan,beam) data once for flat-sky mode:
        - points = stack(-vv, uu) * scale  (scale = 2π * cell_rad)
        - Ir, Ii, sig
        """
        cell_rad = self._cellsize_arcsec_to_rad(cell_size)
        scale = (2.0 * np.pi * cell_rad) if self.use_2pi else cell_rad

        cb_pack = {}
        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            uu_t = torch.from_numpy(uu).to(device, dtype=torch.float32)
            vv_t = torch.from_numpy(vv).to(device, dtype=torch.float32)

            I_use = I.conj() if self.conj_data else I
            Ir_t = torch.from_numpy(I_use.real).to(device, dtype=torch.float32)
            Ii_t = torch.from_numpy(I_use.imag).to(device, dtype=torch.float32)
            sig_t = torch.from_numpy(sI).to(device, dtype=torch.float32)

            points_scaled = torch.stack([-vv_t, uu_t], dim=0) * scale  # [2, N]
            cb_pack[(c, b)] = {"points": points_scaled, "Ir": Ir_t, "Ii": Ii_t, "sig": sig_t}

        self._cb_pack = cb_pack

    # ---------- (optional) r^2 cache if you ever enable w-term ----------
    def _r2_for(self, H, W, cell_size_arcsec, device):
        key = (int(H), int(W), float(cell_size_arcsec))
        r2 = self._r2_cache.get(key)
        if r2 is not None and r2.device == torch.device(device):
            return r2
        cell_rad = self._cellsize_arcsec_to_rad(cell_size_arcsec)
        lx = torch.linspace(-W/2, W/2 - 1, W, device=device) * cell_rad
        ly = torch.linspace(-H/2, H/2 - 1, H, device=device) * cell_rad
        l, m = torch.meshgrid(lx, ly, indexing='xy')
        r2 = l*l + m*m
        self._r2_cache[key] = r2
        return r2

    # ---------- batched reprojection across beams (1 grid_sample per channel) ----------
    def _batched_reproject_pb(self, x_chan, device):
        """
        Returns list of x_pb per beam for the given channel using one grid_sample pass.
        Caches per-channel to avoid recomputation within the same channel loop.
        """
        # quick cache check
        if self._chan_cache["c"] is x_chan and self._chan_cache["xpb_list"] is not None:
            return self._chan_cache["xpb_list"]

        x = torch.as_tensor(x_chan, device=device).float()
        if x.ndim == 1:
            side = int(np.sqrt(x.numel()))
            x = x.view(side, side)
        x_bchw = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        grids = self._static["grid"]  # list of [1,H,W,2] float32
        pbs   = self._static["pb"]    # list of [H,W]     float32
        B = len(grids)

        # Stack grids into [B,H,W,2] and repeat image across batch
        grid_batch = torch.stack(grids, dim=0).squeeze(1)  # [B,H,W,2]
        x_batch    = x_bchw.repeat(B, 1, 1, 1)             # [B,1,H,W]

        repro_batch = torch.nn.functional.grid_sample(
            x_batch, grid_batch, mode='bilinear', align_corners=True
        ).squeeze(1)  # [B,H,W]

        xpb_batch = repro_batch * torch.stack(pbs, dim=0)  # [B,H,W]
        xpb_list = [xpb_batch[i] for i in range(B)]

        # cache for this channel tensor
        self._chan_cache["c"] = x_chan
        self._chan_cache["xpb_list"] = xpb_list
        return xpb_list

    # ---------- flat-sky NUFFT from precomputed x_pb and points ----------
    @staticmethod
    def _nufft_flat_from_xpb(x_pb, points_scaled, cell_size):
        img_c = x_pb.to(torch.complex64)  # keep as complex64 for speed (default eps~1e-6)
        return (cell_size**2) * pytorch_finufft.functional.finufft_type2(
            points_scaled, img_c, isign=1, modeord=0
        )

    # ---------- (optional) w-stack path (kept for completeness; not optimized here) ----------
    def forward_beam_wstack(self, x2d, pb_t, grid_t, uu, vv, ww, cell_size, device):
        xt = torch.as_tensor(x2d, device=device).float()
        if xt.ndim == 1:
            side = int(np.sqrt(xt.numel()))
            xt = xt.view(side, side)
        xt = xt.unsqueeze(0).unsqueeze(0)
        repro = torch.nn.functional.grid_sample(xt, grid_t, mode='bilinear',
                                                align_corners=True).squeeze(0).squeeze(0)
        x_pb = repro * pb_t

        cell_rad = self._cellsize_arcsec_to_rad(cell_size)
        scale = (2.0 * np.pi * cell_rad) if self.use_2pi else cell_rad
        uu_t = torch.from_numpy(uu).to(device, dtype=torch.float32) * scale
        vv_t = torch.from_numpy(vv).to(device, dtype=torch.float32) * scale
        ww_t = torch.from_numpy(ww).to(device, dtype=torch.float32)

        H, W = x_pb.shape
        r2 = self._r2_for(H, W, cell_size, device)
        model_vis = torch.empty_like(uu_t, dtype=torch.complex64)
        w_edges = torch.linspace(ww_t.min(), ww_t.max(), self.Nw + 1, device=device)
        w_centers = 0.5 * (w_edges[:-1] + w_edges[1:])
        bin_ids = torch.bucketize(ww_t, w_edges) - 1
        bin_ids = torch.clamp(bin_ids, 0, self.Nw - 1)

        for j in range(self.Nw):
            idx = (bin_ids == j).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            points = torch.stack([-vv_t[idx], uu_t[idx]], dim=0)
            phase = torch.exp(1j * np.pi * w_centers[j] * r2)
            x_mod = x_pb.to(torch.complex64) * phase
            vis_bin = (cell_size**2) * pytorch_finufft.functional.finufft_type2(
                points, x_mod, isign=1, modeord=0
            )
            model_vis[idx] = vis_bin
        return model_vis

    # ---------- objective ----------
    def objective(self, x, vis_data, device,
                  pb_list=None, grid_list=None, pb=None, grid_array=None,
                  cell_size=None, fftsd=None, fftbeam=None, tapper=None,
                  lambda_sd=0.0, fftkernel=None, beam_workers=4, verbose=False, **_):

        if cell_size is None:
            raise ValueError("`cell_size` (arcsec) is required.")

        # Stage statics and dynamics
        if pb is None or grid_array is None:
            if pb_list is not None and grid_list is not None:
                pb, grid_array = pb_list, grid_list
            else:
                raise ValueError("Pass `pb` and `grid_array` on first call.")
        self._ensure_staged_static(device, pb=pb, grid_array=grid_array)

        # Clear per-channel repro cache
        self._chan_cache = {"c": None, "xpb_list": None}

        # Flat-sky staging (points & data) once
        if self.Nw is None and self._cb_pack is None:
            self.stage_dynamic_flat(vis_data, device, cell_size)

        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        Ltot = torch.zeros((), device=device, dtype=torch.float32)

        # ----- χ² term -----
        if self.Nw is None:
            # One reprojection per channel; reuse for each beam
            last_c = None
            xpb_list = None
            for (c, b), pack in self._cb_pack.items():
                if c is not last_c:
                    xpb_list = self._batched_reproject_pb(x[c], device)  # [B,H,W]
                    last_c = c
                x_pb = xpb_list[b]

                model_vis = self._nufft_flat_from_xpb(x_pb, pack["points"], cell_size)
                rr = (model_vis.real - pack["Ir"]) / pack["sig"]
                ri = (model_vis.imag - pack["Ii"]) / pack["sig"]
                Ltot = Ltot + 0.5 * torch.sum(rr*rr + ri*ri)
        else:
            # W-stack fallback (not your current use)
            # Re-iterate vis_data (no dynamic pack used here)
            for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
                x_chan = x[c]
                pb_t   = self._static["pb"][b]
                grid_t = self._static["grid"][b]
                model_vis = self.forward_beam_wstack(x_chan, pb_t, grid_t, uu, vv, ww, cell_size, device)

                I_use = I.conj() if self.conj_data else I
                Ir = torch.from_numpy(I_use.real).to(device, dtype=torch.float32)
                Ii = torch.from_numpy(I_use.imag).to(device, dtype=torch.float32)
                sig = torch.from_numpy(sI).to(device, dtype=torch.float32)
                rr = (model_vis.real - Ir) / sig
                ri = (model_vis.imag - Ii) / sig
                Ltot = Ltot + 0.5 * torch.sum(rr*rr + ri*ri)

        # ----- SD term -----
        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t   = torch.from_numpy(fftsd).to(device)
            fftbeam_t = torch.from_numpy(fftbeam).to(device)
            tapper_t  = torch.from_numpy(tapper).to(device)
            xfft2 = tfft2(x * tapper_t)
            model_sd = (cell_size**2) * xfft2 * fftbeam_t
            Ltot = Ltot + 0.5 * (torch.nansum((model_sd.real - fftsd_t.real)**2) +
                                 torch.nansum((model_sd.imag - fftsd_t.imag)**2)) * lambda_sd

        # ----- Reg term -----
        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t = torch.from_numpy(tapper).to(device)
            fftkernel_t = torch.from_numpy(fftkernel).to(device)
            xfft2 = tfft2(x * tapper_t)
            conv = (cell_size**2) * xfft2 * fftkernel_t
            Ltot = Ltot + 0.5 * torch.nansum(torch.abs(conv)**2) * self.lambda_r

        # One backward (cheaper than per-beam backprops)
        Ltot.backward()

        return Ltot

    # ---------- optimizer-friendly wrapper ----------
    def loss(self, x, shape, device, vis_data, **kwargs):
        """
        Returns (loss_value, grad_flat) as numpy (LBFGS-friendly).
        """
        u = x.reshape(shape)
        u = torch.from_numpy(u).to(device).requires_grad_(True)

        # Stage statics first call; dynamic staging happens in objective
        if self._static["pb"] is None:
            self._ensure_staged_static(
                device=device,
                pb=kwargs.get('pb', None) or kwargs.get('pb_list', None),
                grid_array=kwargs.get('grid_array', None) or kwargs.get('grid_list', None),
            )

        L = self.objective(x=u, vis_data=vis_data, device=device, **kwargs)

        if u.grad is None:
            L.backward()

        grad = u.grad.detach().cpu().numpy().astype(x.dtype)

        if torch.device(device).type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved  = torch.cuda.memory_reserved(device)  / 1024**2
            total     = torch.cuda.get_device_properties(device).total_memory / 1024**2
            logger.info(
                f"[PID {os.getpid()}] Iter cost: {np.format_float_scientific(L.item(), precision=6)} | "
                f"GPU: {allocated:.2f} MB alloc, {reserved:.2f} MB res, {total:.2f} MB total"
            )
        else:
            logger.info(f"[PID {os.getpid()}] Iter cost: {np.format_float_scientific(L.item(), precision=6)} (device: cpu)")

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




