import os
import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2
from joblib import Parallel, delayed

from ivis.logger import logger


def format_input_tensor(input_tensor):
    """
    Format an input tensor for grid_sample.

    Ensures the input has shape (N=1, C=1, H, W) as required by PyTorch's grid_sample.
    """
    if input_tensor.dim() == 2:
        return input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        return input_tensor.unsqueeze(0)
    return input_tensor


class ClassicIViS:
    """
    Classic IViS imaging model using visibility-domain loss and regularization.

    This class implements a forward operator and a loss function suitable for
    optimization via L-BFGS-B. The model supports CPU and GPU backends.
    """
        
    def __init__(self):
        pass

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

        u = x.reshape(shape)
        u = torch.from_numpy(u).to(device).requires_grad_(True)

        L = self.compute_loss_Pool(
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


    def compute_loss_Pool(
        self, x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device,
        sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size,
        grid_array, beam_workers=4, verbose=False
    ):
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
        lambda_r : float
            Weight of regularization term.
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
        n_beams = len(idmina)

        if device == "cpu":
            beam_indices = np.array_split(np.arange(n_beams), beam_workers)
            results = Parallel(n_jobs=beam_workers)(
                delayed(self.batch_worker)(
                    batch, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size, device, grid_array
                ) for batch in beam_indices
            )
            for partial_loss in results:
                loss_scalar += partial_loss.item()
        else:
            for i in range(n_beams):
                idmin = idmina[i]
                idmax = idmaxa[i]
                uua = uu[idmin:idmin+idmax]
                vva = vv[idmin:idmin+idmax]
                wwa = ww[idmin:idmin+idmax]
                vis_real = data.real[idmin:idmin+idmax]
                vis_imag = data.imag[idmin:idmin+idmax]
                sig = sigma[idmin:idmin+idmax]

                J = self.compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig,
                                          pb[i], cell_size, device, grid_array[i])
                L = 0.5 * J
                L.backward(retain_graph=True)
                loss_scalar += L.item()

                torch.cuda.empty_cache()
                if verbose:
                    self.print_gpu_memory(device)

        fftsd_torch = torch.from_numpy(fftsd).to(device)
        fftbeam_torch = torch.from_numpy(fftbeam).to(device)
        xfft2 = tfft2(x * tapper)
        model_sd = cell_size**2 * xfft2 * fftbeam_torch
        J2 = torch.nansum((model_sd.real - fftsd_torch.real) ** 2)
        J22 = torch.nansum((model_sd.imag - fftsd_torch.imag) ** 2)
        Lsd = 0.5 * (J2 + J22) * lambda_sd
        Lsd.backward(retain_graph=True)
        loss_scalar += Lsd.item()

        fftkernel_torch = torch.from_numpy(fftkernel).to(device)
        xfft2 = tfft2(x * tapper)
        conv = cell_size**2 * xfft2 * fftkernel_torch
        R = torch.nansum(abs(conv) ** 2)
        Lr = 0.5 * R * lambda_r
        Lr.backward()
        loss_scalar += Lr.item()

        if verbose:
            self.print_gpu_memory(device)

        return torch.tensor(loss_scalar)

    def compute_vis_cuda(self, x, uua, vva, wwa, vis_real, vis_imag, sig, pb, cell_size, device, grid):
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
        input_tensor = format_input_tensor(x).float()
        reprojected_tensor = torch.nn.functional.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True).squeeze()
        reproj = reprojected_tensor * pb
        c = reproj.to(torch.complex64)
        model_vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)

        J1 = torch.nansum((model_vis.real - vis_real)**2 / sig**2)
        J11 = torch.nansum((model_vis.imag - vis_imag)**2 / sig**2)

        return J1 + J11

    def batch_worker(self, batch_indices, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size,
                     device, grid_array):
        """
        Evaluate the total loss for a batch of beams (CPU-parallelized variant).
        
        Parameters
        ----------
        batch_indices : list[int]
            Indices of beams to evaluate.
        x : torch.Tensor
            Current image tensor.
        uu, vv, ww : np.ndarray
            UVW coordinates.
        data : np.ndarray
            Complex visibilities.
        sigma : np.ndarray
            Visibility weights.
        pb : list[np.ndarray]
            Primary beam per pointing.
        idmina, idmaxa : list[int]
            Start and size for each beam's visibilities.
        cell_size : float
            Pixel size in arcsec.
        device : str
            Device string ("cpu" expected).
        grid_array : list[np.ndarray]
            Interpolation grid per beam.
        
        Returns
        -------
        total_loss : torch.Tensor
            Sum of chi-squared losses across all selected beams.
        """
        total_loss = torch.tensor(0.0)
        for i in batch_indices:
            J = self.compute_vis_cuda(
                x,
                uu[idmina[i]:idmina[i]+idmaxa[i]],
                vv[idmina[i]:idmina[i]+idmaxa[i]],
                ww[idmina[i]:idmina[i]+idmaxa[i]],
                data.real[idmina[i]:idmina[i]+idmaxa[i]],
                data.imag[idmina[i]:idmina[i]+idmaxa[i]],
                sigma[idmina[i]:idmina[i]+idmaxa[i]],
                pb[i],
                cell_size,
                device,
                grid_array[i]
            )
            total_loss += 0.5 * J
        return total_loss

    def print_gpu_memory(self, device="cuda:0"):
        """
        Print GPU memory usage (allocated, reserved, total) for the specified device.
        
        Parameters
        ----------
        device : str
            CUDA device string, e.g., "cuda:0".
        """
        device = torch.device(device)
        allocated = torch.cuda.memory_allocated(device) / 1e6
        reserved = torch.cuda.memory_reserved(device) / 1e6
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1e6
        print(f"[{device}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Total: {total_mem:.2f} MB")
