#work in progress
import os
import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2
from joblib import Parallel, delayed
from scipy.signal import find_peaks

from ivis.logger import logger
from ivis.models.base import BaseModel

from ivis.models.utils.tensor_ops import format_input_tensor
from ivis.models.utils.gpu import print_gpu_memory
from ivis.utils.fourier import powspec_torch

class TWiSTModel(BaseModel):
    """
    fixme
    """
        
    def __init__(self, wph_op, lambda_wph, pbc, mu, sig, pk, lambda_pk, lambda_ortho):
        self.wph_op = wph_op
        # self.noise_cube = noise_cube.astype(np.float32)  # (R, H, W), stay on CPU as NumPy
        # self.coeff_ref = coeff_ref.cpu().to(dtype=torch.complex64)  # shape (1, 1, X)
        self.lambda_wph = lambda_wph
        self.pbc = pbc
        self.mu = mu
        self.sig = sig
        self.pk = pk
        self.lambda_pk = lambda_pk
        self.lambda_ortho = lambda_ortho

    def loss(self, x, *args):
        """
        Compute total loss and gradient for optimization.

        Parameters
        ----------
        x : np.ndarray
            Flattened image parameter array.
        *args : tuple
            Contains all data and hyperparameters needed for loss computation.

        Returns
        -------
        loss : float
            Scalar value of the total loss.
        grad : np.ndarray
            Flattened gradient of the loss.
        """
        (
            beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device,
            sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, shape,
            cell_size, grid_array, beam_workers
        ) = args

        u = x.reshape((2,shape[0],shape[1]))
        u = torch.from_numpy(u).to(device).requires_grad_(True)

        L = self.compute_loss(
            u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa,
            device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel,
            cell_size, grid_array, beam_workers, verbose=False
        )
        grad = u.grad.cpu().numpy().astype(x.dtype)

        logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")
        return L.item(), grad.ravel()

    
    def forward(self, x, data, uu, vv, pb, idmina, idmaxa, device, cell_size, grid_array):
        """
        Simulate model visibilities from image-domain parameters.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Input image parameters.
        data : np.ndarray
            Observed visibilities (not used here but kept for API symmetry).
        uu, vv : np.ndarray
            Spatial frequencies in rad/pix.
        pb : np.ndarray
            Primary beams per pointing.
        idmina, idmaxa : list
            Slice indices for visibilities per beam.
        device : str or torch.device
            Device to run on.
        cell_size : float
            Pixel size in arcsec.
        grid_array : list of np.ndarray
            Interpolation grids per beam.

        Returns
        -------
        model_vis : np.ndarray
            Simulated visibilities.
        """
        model_vis = torch.zeros(len(uu), dtype=torch.complex64, device=device)

        if not torch.is_tensor(x):
            x = torch.from_numpy(x).to(device)
        else:
            x = x.to(device)

        n_beams = len(idmina)
        for i in range(n_beams):
            idmin = idmina[i]
            idmax = idmaxa[i]
            uua = torch.from_numpy(uu[idmin:idmin+idmax]).to(device)
            vva = torch.from_numpy(vv[idmin:idmin+idmax]).to(device)
            pba = torch.from_numpy(pb[i]).to(device)
            grid = torch.from_numpy(grid_array[i]).to(device)

            points = torch.stack([-vva, uua], dim=0)
            input_tensor = format_input_tensor(x).float().to(device)
            reprojected_tensor = torch.nn.functional.grid_sample(
                input_tensor, grid, mode='bilinear', align_corners=True
            ).squeeze()
            reproj = reprojected_tensor * pba
            c = reproj.to(torch.complex64)
            vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)
            model_vis[idmin:idmin+idmax] = vis

        return model_vis.detach().cpu().numpy()
    

    def compute_loss(
            self, x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device,
            sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size,
            grid_array, beam_workers=4, verbose=False
    ):
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.zero_()
            
        # Split maps
        x1 = x[0]
        x2 = x[1]
        
        # Move to device
        beam = torch.from_numpy(beam).to(device)
        tapper = torch.from_numpy(tapper).to(device)
        
        loss_scalar = torch.tensor(0.0, device=device)

        # ------------------------
        # ---- P(k) Loss ---------
        # ------------------------

        # Compute power spectrum of x2 (noise map or image component)
        ks, ps_x2 = powspec_torch(
            x2,  # assumed shape (H, W)
            pixel_size_arcsec=7,  # convert deg to arcsec
            nbins=self.pk.shape[0]  # match number of bins
        )

        # Compute chi^2 loss against target P(k)
        residual_pk = (ps_x2 - self.pk) / (self.pk + 1e-10)
        Lpk = 0.5 * torch.sum(residual_pk ** 2) * self.lambda_pk

        # Backpropagate
        Lpk.backward(retain_graph=True)
        loss_scalar += Lpk.item()

        # ---------------------------------
        # ---- WPH Orthogonality Loss ----
        # ---------------------------------
        x1_detached = x1.detach() / 1.e-5  # must match scaling!
        x1_formatted = x1_detached.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        x2_formatted = x2.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        
        u1, nb_chunks = self.wph_op.preconfigure(x1_formatted, requires_grad=False, pbc=self.pbc)
        u2, _ = self.wph_op.preconfigure(x2_formatted, requires_grad=True, pbc=self.pbc)
        
        Lortho = 0.0
        
        for chunk_id in range(nb_chunks):
            c1 = self.wph_op.apply(u1, chunk_id, norm=None, pbc=self.pbc)
            c2 = self.wph_op.apply(u2, chunk_id, norm=None, pbc=self.pbc)
            
            inner = torch.sum(torch.real(c1 * torch.conj(c2)))  # ⟨Φ(x1), Φ(x2)⟩
            L = self.lambda_ortho * inner
            L.backward(retain_graph=True)
            Lortho += L.item()
            
            del c1, c2
            torch.cuda.empty_cache()
            
        loss_scalar += Lortho
        
        # # ------------------------
        # # ---- WPH Loss ----------
        # # ------------------------
        
        # x_in = x2 / 1e-5
        # x_formatted = x_in.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)  # [1, 1, H, W]
        
        # # Flatten and move to device
        # mu_flat = self.mu.to(device).reshape(-1)         # [N], complex64
        # sig_flat = self.sig.to(device).reshape(-1)       # [N], float32
        # mask = self.wph_mask.to(device)                  # [2, 1, 1, N]
        
        # # Squeeze to [2, N]
        # while mask.ndim > 2:
        #     mask = mask.squeeze(1)
            
        # u, nb_chunks = self.wph_op.preconfigure(x_formatted, requires_grad=True, pbc=self.pbc)
        # Lwph = 0.0
        
        # for chunk_id in range(nb_chunks):
        #     coeffs_chunk, indices = self.wph_op.apply(u, chunk_id, norm=None, ret_indices=True, pbc=self.pbc)
            
        #     # Flatten chunk to match mu/sig/mask shapes
        #     coeffs_chunk_flat = coeffs_chunk.view(-1)        # [N_chunk]
        #     mu_chunk = mu_flat[indices]                      # [N_chunk]
        #     sig_chunk = sig_flat[indices]                    # [N_chunk]
        #     mask_real = mask[0][indices]                     # [N_chunk]
        #     mask_imag = mask[1][indices]                     # [N_chunk]
            
        #     # Residuals
        #     res_real = ((coeffs_chunk_flat.real - mu_chunk.real) / sig_chunk)[mask_real]
        #     res_imag = ((coeffs_chunk_flat.imag - mu_chunk.imag) / sig_chunk)[mask_imag]
            
        #     residual = torch.cat([res_real, res_imag], dim=0)
            
        #     # Loss and backward
        #     L = 0.5 * torch.sum(residual ** 2) * self.lambda_wph
        #     L.backward(retain_graph=True)
        #     Lwph += L.item()
            
        #     del coeffs_chunk, coeffs_chunk_flat, res_real, res_imag, residual, indices
        #     torch.cuda.empty_cache()
            
        # loss_scalar += Lwph

        # ------------------------
        # ---- WPH Loss ---------
        # ------------------------
        x_in = x2 / 1.e-5 #FIXME ATTENTION
        x_formatted = x_in.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        
        mu_flat = self.mu.squeeze().to(device, dtype=torch.complex64)
        sig_flat = self.sig.squeeze().to(device, dtype=torch.complex64)
        
        u, nb_chunks = self.wph_op.preconfigure(x_formatted, requires_grad=True, pbc=self.pbc)
        Lwph = 0.0
        
        for chunk_id in range(nb_chunks):
            coeffs_chunk, indices = self.wph_op.apply(u, chunk_id, norm=None, ret_indices=True, pbc=self.pbc)

            mu_real = mu_flat[indices].real
            mu_imag = mu_flat[indices].imag
            sig_real = sig_flat[indices].real
            sig_imag = sig_flat[indices].imag

            eps = 1e-6
            residual_real = (torch.real(coeffs_chunk) - mu_real) #/ (sig_real + eps)
            residual_imag = (torch.imag(coeffs_chunk) - mu_imag) #/ (sig_imag + eps)
            
            residual = torch.cat([residual_real, residual_imag])
            
            L = 0.5 * torch.sum(residual ** 2) * self.lambda_wph
            L.backward(retain_graph=True)
            Lwph += L.item()
            
            del coeffs_chunk, residual_real, residual_imag, residual, indices
            torch.cuda.empty_cache()
            
        loss_scalar += Lwph
        
        # -------------------------------
        # ---- Beam χ² vis loss --------
        # -------------------------------
        for i in range(len(idmina)):
            idmin, idmax = idmina[i], idmaxa[i]
            uua = uu[idmin:idmin+idmax]
            vva = vv[idmin:idmin+idmax]
            wwa = ww[idmin:idmin+idmax]
            vis_real = data.real[idmin:idmin+idmax]
            vis_imag = data.imag[idmin:idmin+idmax]
            sig_i = sigma[idmin:idmin+idmax]
            
            J = self.compute_vis_cuda(x1, x2, uua, vva, wwa, vis_real, vis_imag, sig_i,
                                      pb[i], cell_size, device, grid_array[i])
            L = 0.5 * J
            L.backward(retain_graph=True)  # <---- explicitly keeping retain_graph=True
            loss_scalar += L.item()
            
            torch.cuda.empty_cache()
            if verbose:
                print_gpu_memory(device)
                
        # ----------------------------
        # ---- Single Dish loss ------
        # ----------------------------
        fftsd_torch = torch.from_numpy(fftsd).to(device)
        fftbeam_torch = torch.from_numpy(fftbeam).to(device)
        xfft2 = tfft2(x1 * tapper)
        model_sd = cell_size**2 * xfft2 * fftbeam_torch
        
        diff_real = model_sd.real - fftsd_torch.real
        diff_imag = model_sd.imag - fftsd_torch.imag
        Lsd = 0.5 * (torch.sum(diff_real**2) + torch.sum(diff_imag**2)) * lambda_sd
        Lsd.backward(retain_graph=True)  # Optional — still kept if needed
        loss_scalar += Lsd.item()
        
        del xfft2, model_sd, fftsd_torch, fftbeam_torch, diff_real, diff_imag
        torch.cuda.empty_cache()
        
        # ----------------------------
        # ---- Regularization --------
        # ----------------------------
        fftkernel_torch = torch.from_numpy(fftkernel).to(device)
        for xreg in [x1, x2]:
            xfft2 = tfft2(xreg * tapper)
            conv = cell_size**2 * xfft2 * fftkernel_torch
            Lr = 0.5 * torch.sum(torch.abs(conv) ** 2) * lambda_r
            Lr.backward(retain_graph=False)  # Final term: no retain
            loss_scalar += Lr.item()
            
            del xfft2, conv
            torch.cuda.empty_cache()
            
        del fftkernel_torch
        if verbose:
            print_gpu_memory(device)
            
        return loss_scalar
    
    # def compute_loss(
    #         self, x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device,
    #         sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size,
    #         grid_array, beam_workers=4, verbose=False
    # ):
    #     x.requires_grad_(True)
    #     if x.grad is not None:
    #         x.grad.zero_()
            
    #     # Split maps
    #     x1 = x[0]
    #     x2 = x[1]
        
    #     # Move to device
    #     beam = torch.from_numpy(beam).to(device)
    #     tapper = torch.from_numpy(tapper).to(device)
        
    #     loss_scalar = torch.tensor(0.0, device=device)
        
    #     # ------------------------
    #     # ---- WPH Loss ---------
    #     # ------------------------
        
    #     x_in = x1*1.e5  # We want gradients w.r.t. x1
    #     idx = torch.randint(0, self.noise_cube.shape[0], (1,)).item()
    #     N_j = torch.from_numpy(self.noise_cube[idx]).to(device)
    #     x_noisy = (x_in + N_j).unsqueeze(0).unsqueeze(0)  # No detach!
        
    #     # WPH call — pass in x_noisy directly
    #     u, nb_chunks = self.wph_op.preconfigure(x_noisy, requires_grad=True, pbc=False)
    #     Lwph_total = torch.tensor(0.0, device=device)
        
    #     for chunk_id in range(nb_chunks):
    #         coeffs_chunk, indices = self.wph_op.apply(u, chunk_id, norm=None, ret_indices=True, pbc=False)
    #         coeffs_ref_chunk = self.coeff_ref[0, 0, indices].to(device)

    #         # Explicitly convert to complex
    #         coeffs_chunk = coeffs_chunk.to(torch.complex64)
    #         coeffs_ref_chunk = coeffs_ref_chunk.to(torch.complex64)

    #         residual_real = coeffs_chunk.real - coeffs_ref_chunk.real
    #         residual_imag = coeffs_chunk.imag - coeffs_ref_chunk.imag
    #         residual = torch.cat([residual_real, residual_imag])
            
    #         L = 0.5 * residual.pow(2).sum() * self.lambda_wph
    #         L.backward(retain_graph=True)
    #         Lwph_total += L.item()
            
    #         del coeffs_chunk, residual, indices, coeffs_ref_chunk
    #         torch.cuda.empty_cache()
        
    #     loss_scalar += Lwph_total
        
    #     if verbose:
    #         print_gpu_memory(device)

        
    #     # -------------------------------
    #     # ---- Beam χ² vis loss --------
    #     # -------------------------------
    #     for i in range(len(idmina)):
    #         idmin, idmax = idmina[i], idmaxa[i]
    #         uua = uu[idmin:idmin+idmax]
    #         vva = vv[idmin:idmin+idmax]
    #         wwa = ww[idmin:idmin+idmax]
    #         vis_real = data.real[idmin:idmin+idmax]
    #         vis_imag = data.imag[idmin:idmin+idmax]
    #         sig_i = sigma[idmin:idmin+idmax]
            
    #         J = self.compute_vis_cuda(x1, uua, vva, wwa, vis_real, vis_imag, sig_i,
    #                                   pb[i], cell_size, device, grid_array[i])
    #         L = 0.5 * J
    #         L.backward(retain_graph=True)  # <---- explicitly keeping retain_graph=True
    #         loss_scalar += L.item()
            
    #         torch.cuda.empty_cache()
    #         if verbose:
    #             print_gpu_memory(device)
                
    #     # ----------------------------
    #     # ---- Single Dish loss ------
    #     # ----------------------------
    #     fftsd_torch = torch.from_numpy(fftsd).to(device)
    #     fftbeam_torch = torch.from_numpy(fftbeam).to(device)
    #     xfft2 = tfft2(x1 * tapper)
    #     model_sd = cell_size**2 * xfft2 * fftbeam_torch
        
    #     diff_real = model_sd.real - fftsd_torch.real
    #     diff_imag = model_sd.imag - fftsd_torch.imag
    #     Lsd = 0.5 * (torch.sum(diff_real**2) + torch.sum(diff_imag**2)) * lambda_sd
    #     Lsd.backward(retain_graph=True)  # Optional — still kept if needed
    #     loss_scalar += Lsd.item()
        
    #     del xfft2, model_sd, fftsd_torch, fftbeam_torch, diff_real, diff_imag
    #     torch.cuda.empty_cache()
        
    #     # ----------------------------
    #     # ---- Regularization --------
    #     # ----------------------------
    #     fftkernel_torch = torch.from_numpy(fftkernel).to(device)
    #     for xreg in [x1, x2]:
    #         xfft2 = tfft2(xreg * tapper)
    #         conv = cell_size**2 * xfft2 * fftkernel_torch
    #         Lr = 0.5 * torch.sum(torch.abs(conv) ** 2) * lambda_r
    #         Lr.backward(retain_graph=False)  # Final term: no retain
    #         loss_scalar += Lr.item()
            
    #         del xfft2, conv
    #         torch.cuda.empty_cache()
            
    #     del fftkernel_torch
    #     if verbose:
    #         print_gpu_memory(device)
            
    #     return loss_scalar

    def compute_vis_cuda(self, x1, x2, uua, vva, wwa, vis_real, vis_imag, sig, pb, cell_size, device, grid):
        """
        Compute visibility-domain chi-squared loss for a single beam using NUFFT on GPU.
        
        Parameters
        ----------
        x : torch.Tensor
            2D sky image.
        uua, vva : np.ndarray
            UV coordinates in rad/pixel.
        wwa : np.ndarray
            W coordinate (unused here, passed for API compatibility).
        vis_real, vis_imag : np.ndarray
            Real and imaginary parts of observed visibilities.
        sigma : np.ndarray
            Visibility-domain standard deviation for weighting.
        pb : np.ndarray
            Primary beam image.
        cell_size : float
            Pixel size in arcsec.
        device : str or torch.device
            Target device ("cuda", "cpu", etc.).
        grid : np.ndarray
            Grid for image re-projection.
        
        Returns
        -------
        loss : torch.Tensor
            Weighted chi-squared loss for this beam.
        """
        uua = torch.from_numpy(uua).to(device)
        vva = torch.from_numpy(vva).to(device)
        pb = torch.from_numpy(pb).to(device)
        vis_real = torch.from_numpy(vis_real).to(device)
        vis_imag = torch.from_numpy(vis_imag).to(device)
        sig = torch.from_numpy(sig).to(device)
        grid = torch.from_numpy(grid).to(device)

        points = torch.stack([-vva, uua], dim=0).to(device)
        input_tensor_1 = format_input_tensor(x1).float()
        reprojected_tensor_1 = torch.nn.functional.grid_sample(input_tensor_1, grid, mode='bilinear', align_corners=True).squeeze()
        input_tensor_2 = format_input_tensor(x2).float()
        reprojected_tensor_2 = torch.nn.functional.grid_sample(input_tensor_2, grid, mode='bilinear', align_corners=True).squeeze()

        reproj = reprojected_tensor_1 * pb 
        reproj2 = reprojected_tensor_2 * pb

        c = reproj.to(torch.complex64)
        model_vis1 = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)

        c2 = reproj2.to(torch.complex64)
        model_vis2 = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c2, isign=1, modeord=0)

        model_vis = model_vis1 + model_vis2

        J1 = torch.nansum((model_vis.real - vis_real)**2 / sig**2)
        J11 = torch.nansum((model_vis.imag - vis_imag)**2 / sig**2)

        return J1 + J11
