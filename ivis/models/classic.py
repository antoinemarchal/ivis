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
                points, c, isign=1, modeord=0, eps = 1e-3
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
                points, x_mod, isign=1, modeord=0, eps = 1e-3
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




