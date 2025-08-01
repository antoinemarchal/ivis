import os
import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2
from tqdm import tqdm as tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed
import gc

from ivis.logger import logger  # Import the logger

def print_gpu_memory(device="cuda:0"):
    device = torch.device(device)
    allocated = torch.cuda.memory_allocated(device) / 1e6  # in MB
    reserved  = torch.cuda.memory_reserved(device) / 1e6   # in MB
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e6  # in MB

    print(f"[{device}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Total: {total_mem:.2f} MB")
    

def format_input_tensor(input_tensor):
    #ensure the input tensor has 4 dimensions
    if input_tensor.dim() == 2:  # If shape is [H_in, W_in]
        #add batch and channel dims
        input_tensor_reshape = input_tensor.unsqueeze(0).unsqueeze(0) 
    elif input_tensor.dim() == 3: # If shape is [C, H_in, W_in]
        #add batch dim
        input_tensor_reshape = input_tensor.unsqueeze(0)  
        
    return input_tensor_reshape


def objective(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, shape, cell_size, grid_array, beam_workers):
    #reshape x into u grid
    u = x.reshape(shape)
    #track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True) 
    # L = compute_loss_low_memory(u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array)
    L = compute_loss_Pool(u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array, beam_workers, False)
    #compute the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)

    logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

    return L.item(), u_grad.ravel()


def single_frequency_model(x, data, uu, vv, pb, idmina, idmaxa, device, cell_size, grid_array):
    # model_vis has same length as uu, and is complex64
    model_vis = torch.zeros(len(uu), dtype=torch.complex64, device=device)

    if not torch.is_tensor(x):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)

    # INTERFEROMETER - JOINT loop over beams
    n_beams = len(idmina)
    for i in range(n_beams):
        idmin = idmina[i]
        idmax = idmaxa[i]

        uua = torch.from_numpy(uu[idmin:idmin+idmax]).to(device)
        vva = torch.from_numpy(vv[idmin:idmin+idmax]).to(device)

        pba = torch.from_numpy(pb[i]).to(device)
        grid = torch.from_numpy(grid_array[i]).to(device)

        # move points to device
        points = torch.stack([-vva, uua], dim=0)

        # reproject field at beam position
        input_tensor = format_input_tensor(x).float().to(device)
        reprojected_tensor = torch.nn.functional.grid_sample(
            input_tensor, grid, mode='bilinear', align_corners=True
        ).squeeze()

        # apply primary beam
        reproj = reprojected_tensor * pba

        # pack into complex number
        c = reproj.to(torch.complex64)

        # compute visibilities with NuFFT
        vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)
        model_vis[idmin:idmin+idmax] = vis

    return model_vis.detach().cpu().numpy()


def compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig, pb, cell_size, device, grid):
    #move data to device
    uua = torch.from_numpy(uua).to(device)
    vva = torch.from_numpy(vva).to(device)
    wwa = torch.from_numpy(wwa).to(device)
    pb = torch.from_numpy(pb).to(device)
    vis_real = torch.from_numpy(vis_real).to(device)
    vis_imag = torch.from_numpy(vis_imag).to(device)
    sig = torch.from_numpy(sig).to(device)
    grid = torch.from_numpy(grid).to(device)
    
    #move points to device
    points = torch.zeros((2,len(uua)))
    points[0] = -vva
    points[1] = uua
    points = points.to(device)
    
    #reproject field at beam position
    input_tensor = format_input_tensor(x).float()
    # Perform interpolation
    reprojected_tensor = torch.nn.functional.grid_sample(input_tensor, grid, mode='bilinear', align_corners=True)
    # Remove batch and channel dimensions
    reprojected_tensor = reprojected_tensor.squeeze()

    #apply primary beam
    reproj = reprojected_tensor * pb
    
    #pack into complex number
    c = (reproj).to(torch.float32) + 1J * 0
    
    #compute visibilities with NuFFT
    model_vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)

    #cost real and imag
    J1 = torch.nansum((model_vis.real - vis_real)**2 / sig**2)
    J11 = torch.nansum((model_vis.imag - vis_imag)**2 / sig**2)
    
    return J1 + J11


def compute_loss_Pool(
    x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma,
    fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array,
    beam_workers=4, verbose=False
):
    """
    Compute total loss (interferometric + single-dish + regularization),
    memory-safe on GPU and parallel on CPU if needed.
    Uses per-beam backward passes to avoid memory accumulation.
    """
    x.requires_grad_(True)
    if x.grad is not None:
        x.grad.zero_()

    beam = torch.from_numpy(beam).to(device)
    tapper = torch.from_numpy(tapper).to(device)

    loss_scalar = 0.0  # running total of loss (detached)

    n_beams = len(idmina)

    # --------------------
    # INTERFEROMETER TERM
    # --------------------
    if device == "cpu":
        # Parallel CPU path (unchanged)
        beam_indices = np.array_split(np.arange(n_beams), beam_workers)
        results = Parallel(n_jobs=beam_workers)(
            delayed(batch_worker)(
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

            J = compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig,
                                 pb[i], cell_size, device, grid_array[i])
            L = 0.5 * J
            L.backward(retain_graph=True)
            loss_scalar += L.item()

            del uua, vva, wwa, vis_real, vis_imag, sig, J, L
            torch.cuda.empty_cache()

            # Optional memory tracking
            if verbose and device != "cpu":
                print_gpu_memory(device)

    # ----------------------
    # SINGLE DISH TERM
    # ----------------------
    fftsd_torch = torch.from_numpy(fftsd).to(device)
    fftbeam_torch = torch.from_numpy(fftbeam).to(device)

    xfft2 = tfft2(x * tapper)
    model_sd = cell_size**2 * xfft2 * fftbeam_torch

    J2 = torch.nansum((model_sd.real - fftsd_torch.real) ** 2)
    J22 = torch.nansum((model_sd.imag - fftsd_torch.imag) ** 2)
    Lsd = 0.5 * (J2 + J22) * lambda_sd
    Lsd.backward(retain_graph=True)
    loss_scalar += Lsd.item()

    del fftsd_torch, fftbeam_torch, model_sd, J2, J22, Lsd
    torch.cuda.empty_cache()

    # --------------------
    # REGULARIZATION TERM
    # --------------------
    fftkernel_torch = torch.from_numpy(fftkernel).to(device)
    xfft2 = tfft2(x * tapper)
    conv = cell_size**2 * xfft2 * fftkernel_torch
    R = torch.nansum(abs(conv) ** 2)
    Lr = 0.5 * R * lambda_r
    Lr.backward()
    loss_scalar += Lr.item()

    del conv, fftkernel_torch, xfft2, R, Lr
    torch.cuda.empty_cache()

    # Optional memory tracking
    if verbose and device != "cpu":
        print_gpu_memory(device)

    return torch.tensor(loss_scalar)


def beam_worker(i, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size, device, grid_array):
    idmin = idmina[i]
    idmax = idmaxa[i]
    uua = uu[idmin:idmin+idmax]
    vva = vv[idmin:idmin+idmax]
    wwa = ww[idmin:idmin+idmax]
    vis_real = data.real[idmin:idmin+idmax]
    vis_imag = data.imag[idmin:idmin+idmax]
    sig = sigma[idmin:idmin+idmax]

    return compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig, pb[i], cell_size, device, grid_array[i])


def batch_worker(batch_indices, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size, device, grid_array):
    total_loss = torch.tensor(0.0)
    for i in batch_indices:
        J = beam_worker(i, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size, device, grid_array)
        total_loss += 0.5 * J
    return total_loss


########################################################
def compute_loss(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array): #would have to fixed with safe-memory usage
    #move beam to device
    beam = torch.from_numpy(beam).to(device)

    #init total loss
    loss_tot = torch.zeros(1).to(device)

    #INTERFEROMETER - JOINT loop over beams
    n_beams = len(idmina)
    for i in np.arange(n_beams):
        idmin = idmina[i]; idmax = idmaxa[i]
        uua = uu[idmin:idmin+idmax]
        vva = vv[idmin:idmin+idmax]
        wwa = ww[idmin:idmin+idmax]
        vis_real = data.real[idmin:idmin+idmax]
        vis_imag = data.imag[idmin:idmin+idmax]
        sig = sigma[idmin:idmin+idmax] 

        #get cost for each beam
        J = compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig, pb[i], cell_size, device, grid_array[i])
        #add to total loss
        loss_tot += 0.5 * J
        
    #SINGLE DISH
    fftsd = torch.from_numpy(fftsd).to(device)
    tapper = torch.from_numpy(tapper).to(device)
    #fft2 of tapper sky image
    xfft2 = tfft2(x*tapper)
    #convolution in fourier space to apply single dish beam
    model_sd = cell_size**2 * xfft2 * torch.from_numpy(fftbeam).to(device)
    #residual real and imag
    J2 = torch.nansum((model_sd.real - fftsd.real)**2)
    J22 = torch.nansum((model_sd.imag - fftsd.imag)**2)
    #add to total loss
    loss_tot += 0.5 * (J2 + J22) * lambda_sd

    #REGULARIZATION
    conv = cell_size**2 * xfft2 * torch.from_numpy(fftkernel).to(device)
    R = torch.nansum(abs(conv)**2)
    loss_tot += 0.5 * R * lambda_r
    
    loss_tot.backward(retain_graph=True)    

    return loss_tot.detach().cpu()


# #WORK IN PROGRESS
# def compute_vis_cuda_batched(x, uu_batch, vv_batch, vis_real_batch, vis_imag_batch,
#                               sigma_batch, pb_batch, grid_batch, cell_size, device,
#                               lengths):
#     """
#     Batched GPU loss computation using FINUFFT.
#     Projects the sky model onto each beam and computes interferometric loss.
#     """
#     B = len(lengths)
#     H, W = pb_batch.shape[1:]

#     # Format x and expand for batch reprojection
#     input_tensor = format_input_tensor(x).float().to(device)
#     input_tensor = input_tensor.expand(B, -1, -1, -1)  # [B, 1, H, W]

#     # Move batch data to GPU
#     uu_batch = torch.from_numpy(uu_batch).to(device)
#     vv_batch = torch.from_numpy(vv_batch).to(device)
#     vis_real_batch = torch.from_numpy(vis_real_batch).to(device)
#     vis_imag_batch = torch.from_numpy(vis_imag_batch).to(device)
#     sigma_batch = torch.from_numpy(sigma_batch).to(device)
#     pb_batch = torch.from_numpy(pb_batch).to(device)
#     grid_batch = torch.from_numpy(grid_batch).to(device)

#     # Handle potential extra grid dimension
#     if grid_batch.ndim == 5 and grid_batch.shape[1] == 1:
#         grid_batch = grid_batch.squeeze(1)

#     # Reproject x to each beam's coordinate system
#     reproj = torch.nn.functional.grid_sample(
#         input_tensor, grid_batch, mode='bilinear', align_corners=True
#     )  # [B, 1, H, W]
#     reproj = reproj.squeeze(1) * pb_batch  # [B, H, W]

#     losses = []

#     for i in range(B):
#         length = lengths[i]
#         if length == 0:
#             continue

#         uua = uu_batch[i, :length].reshape(-1)
#         vva = vv_batch[i, :length].reshape(-1)
#         vis_real = vis_real_batch[i, :length]
#         vis_imag = vis_imag_batch[i, :length]
#         sig = sigma_batch[i, :length]

#         # Create (2, N) target points
#         points = torch.zeros((2, length), device=device)
#         points[0] = -vva
#         points[1] = uua

#         # No flatten: keep c as 2D complex image
#         c = reproj[i].float() + 1j * 0  # shape [H, W], complex

#         model_vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(
#             points, c, isign=1, modeord=0
#         )

#         res_real = model_vis.real - vis_real
#         res_imag = model_vis.imag - vis_imag
#         J = torch.nansum((res_real**2 + res_imag**2) / sig**2)
#         losses.append(0.5 * J)

#     return torch.stack(losses).sum()



# def compute_loss_Pool(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma,
#                       fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array,
#                       beam_workers=4, verbose=False):
    
#     beam = torch.from_numpy(beam).to(device)
#     loss_tot = torch.zeros(1, device=device)
#     n_beams = len(idmina)
    
#     # --- PARALLEL ON CPU WITH BATCHED BEAMS ---
#     if device == "cpu":
#         # Split beam indices into batches
#         beam_indices = np.array_split(np.arange(n_beams), beam_workers)
        
#         # Parallel execution
#         results = Parallel(n_jobs=beam_workers)(
#             delayed(batch_worker)(
#                 batch, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size, device, grid_array
#             ) for batch in beam_indices
#         )
        
#         for partial_loss in results:
#             loss_tot += partial_loss.to(device)

#     # --- SEQUENTIAL ON GPU ---
#     else:
#         for i in range(n_beams):
#             idmin = idmina[i]
#             idmax = idmaxa[i]
#             uua = uu[idmin:idmin+idmax]
#             vva = vv[idmin:idmin+idmax]
#             wwa = ww[idmin:idmin+idmax]
#             vis_real = data.real[idmin:idmin+idmax]
#             vis_imag = data.imag[idmin:idmin+idmax]
#             sig = sigma[idmin:idmin+idmax]

#             J = compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig,
#                                  pb[i], cell_size, device, grid_array[i])
#             loss_tot += 0.5 * J
            
#     # # --- MINI-BATCHED ON GPU --- No improvement because nufft is not batched, only the interpolation
#     # else:
#     #     max_len = max(idmaxa[i] for i in range(n_beams))
#     #     batch_size = 32

#     #     for i in range(0, n_beams, batch_size):
#     #         batch_ids = range(i, min(i + batch_size, n_beams))
#     #         b = len(batch_ids)
#     #         H, W = pb[batch_ids[0]].shape

#     #         uu_batch = np.zeros((b, max_len), dtype=np.float32)
#     #         vv_batch = np.zeros((b, max_len), dtype=np.float32)
#     #         vis_real_batch = np.zeros((b, max_len), dtype=np.float32)
#     #         vis_imag_batch = np.zeros((b, max_len), dtype=np.float32)
#     #         sigma_batch = np.zeros((b, max_len), dtype=np.float32)
#     #         pb_batch = np.zeros((b, H, W), dtype=np.float32)
#     #         grid_batch = np.zeros((b, H, W, 2), dtype=np.float32)
#     #         lengths = []

#     #         for j, beam_idx in enumerate(batch_ids):
#     #             idmin = idmina[beam_idx]
#     #             idmax = idmaxa[beam_idx]
#     #             length = idmax
#     #             lengths.append(length)

#     #             uu_batch[j, :length] = uu[idmin:idmin+idmax]
#     #             vv_batch[j, :length] = vv[idmin:idmin+idmax]
#     #             vis_real_batch[j, :length] = data.real[idmin:idmin+idmax]
#     #             vis_imag_batch[j, :length] = data.imag[idmin:idmin+idmax]
#     #             sigma_batch[j, :length] = sigma[idmin:idmin+idmax]

#     #             pb_batch[j] = pb[beam_idx]
#     #             grid_j = grid_array[beam_idx]
#     #             if grid_j.ndim == 3:
#     #                 grid_j = grid_j[np.newaxis, ...]
#     #             elif grid_j.ndim == 5 and grid_j.shape[0] == 1:
#     #                 grid_j = grid_j.squeeze(0)

#     #             assert grid_j.shape[:3] == (1, H, W), f"grid[{beam_idx}].shape[:3] != (1, {H}, {W})"
#     #             grid_batch[j] = grid_j

#     #         batch_loss = compute_vis_cuda_batched(
#     #             x, uu_batch, vv_batch, vis_real_batch, vis_imag_batch,
#     #             sigma_batch, pb_batch, grid_batch, cell_size, device, lengths
#     #         )

#     #         loss_tot += batch_loss
#     #         torch.cuda.empty_cache()

#     # --- SINGLE DISH COMPONENT ---
#     fftsd = torch.from_numpy(fftsd).to(device)
#     tapper = torch.from_numpy(tapper).to(device)
#     xfft2 = tfft2(x * tapper)
#     model_sd = cell_size**2 * xfft2 * torch.from_numpy(fftbeam).to(device)

#     J2 = torch.nansum((model_sd.real - fftsd.real) ** 2)
#     J22 = torch.nansum((model_sd.imag - fftsd.imag) ** 2)
#     loss_tot += 0.5 * (J2 + J22) * lambda_sd

#     # --- REGULARIZATION ---
#     conv = cell_size**2 * xfft2 * torch.from_numpy(fftkernel).to(device)
#     R = torch.nansum(abs(conv) ** 2)
#     loss_tot += 0.5 * R * lambda_r

#     loss_tot.backward(retain_graph=True)

#     # Optional memory tracking
#     if verbose and device.type != "cpu":
#         alloc = torch.cuda.memory_allocated(device) / 1e6
#         reserved = torch.cuda.memory_reserved(device) / 1e6
#         logger.info(f"[GPU {device}] Allocated: {alloc:.2f} MB, Reserved: {reserved:.2f} MB")

#     return loss_tot.detach().cpu()


# #FIXME
# def compute_loss_low_memory(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array):
#     # Move beam to device (only once)
#     beam = torch.from_numpy(beam).to(device)

#     # Initialize total loss
#     loss_tot = torch.zeros(1).to(device)

#     # INTERFEROMETER - Joint loop over beams
#     n_beams = len(idmina)
#     for i in np.arange(n_beams):
#         idmin = idmina[i]
#         idmax = idmaxa[i]

#         # Slice and move only the necessary data to the device for NuFFT computation
#         uua = torch.from_numpy(uu[idmin:idmin+idmax]).to(device)
#         vva = torch.from_numpy(vv[idmin:idmin+idmax]).to(device)
#         wwa = torch.from_numpy(ww[idmin:idmin+idmax]).to(device)
#         pb_i = torch.from_numpy(pb[i]).to(device)
#         grid_i = torch.from_numpy(grid_array[i]).to(device)

#         # Create points tensor for NuFFT
#         points = torch.zeros((2, len(uua)), device=device)
#         points[0] = -vva
#         points[1] = uua

#         # Reproject field at beam position
#         input_tensor = format_input_tensor(x).float()
#         # Perform interpolation
#         reprojected_tensor = torch.nn.functional.grid_sample(input_tensor, grid_i, mode='bilinear', align_corners=True)
#         # Remove batch and channel dimensions
#         reprojected_tensor = reprojected_tensor.squeeze()

#         # Apply primary beam
#         reproj = reprojected_tensor * pb_i

#         # Pack into complex number for NuFFT
#         c = reproj.to(torch.complex64)

#         # Compute visibilities with NuFFT (only transfer the necessary data for this operation)
#         model_vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)

#         # Free memory after each iteration
#         del uua, vva, wwa, pb_i, grid_i, points, reprojected_tensor, reproj, c
#         torch.cuda.empty_cache()

#         # Slice the data for the current beam for the cost function calculation
#         vis_real = torch.from_numpy(data.real[idmin:idmin+idmax]).to(device)
#         vis_imag = torch.from_numpy(data.imag[idmin:idmin+idmax]).to(device)
#         sig = torch.from_numpy(sigma[idmin:idmin+idmax]).to(device)

#         # Compute cost (real and imag)
#         J1 = torch.nansum((model_vis.real - vis_real)**2 / sig**2)
#         J11 = torch.nansum((model_vis.imag - vis_imag)**2 / sig**2)

#         # Free memory after each iteration
#         del vis_real, vis_imag, sig
#         torch.cuda.empty_cache()

#         # Add to total loss
#         loss_tot += 0.5 * (J1 + J11)

#     if lambda_sd != 0: 
#         # SINGLE DISH
#         fftsd = torch.from_numpy(fftsd).to(device)
#         tapper = torch.from_numpy(tapper).to(device)
#         # FFT2 of tapper sky image
#         xfft2 = tfft2(x * tapper)
#         # Convolution in Fourier space to apply single dish beam
#         model_sd = cell_size**2 * xfft2 * torch.from_numpy(fftbeam).to(device)
#         # Residual real and imag
#         J2 = torch.nansum((model_sd.real - fftsd.real)**2)
#         J22 = torch.nansum((model_sd.imag - fftsd.imag)**2)
#         # Add to total loss
#         loss_tot += 0.5 * (J2 + J22) * lambda_sd

#     if lambda_r != 0:
#         # REGULARIZATION
#         conv = cell_size**2 * xfft2 * torch.from_numpy(fftkernel).to(device)
#         R = torch.nansum(abs(conv)**2)
#         loss_tot += 0.5 * R * lambda_r

#     loss_tot.backward(retain_graph=True)

#     return loss_tot.detach().cpu()






#JNK

# dl = cell_size.to(u.rad)
# dl = torch.from_numpy(dl).to(device)

# loss_tot += loss.detach(3).cpu()

# for i in tqdm(np.arange(1)):
#     i = 68 #FIXME TEST ONE BEAM ONLY

# conv = torchaudio.functional.fftconvolve(x*tapper, torch.from_numpy(kernel).to(device), mode="same")    

# model_vis *= g
# # w-proj
# model_vis_real = (g.real*model_vis.real - g.imag*model_vis.imag)
# model_vis_imag = (g.real*model_vis.imag + g.imag*model_vis.real)        
# J1 = torch.nansum((model_vis_real - vis_real)**2 / w)
# J11 = torch.nansum((model_vis_imag - vis_imag)**2 / w)

# J = compute_vis_cuda_shift(x, uua, vva, wwa, vis_real, vis_imag, w, g, pb[i], cell_size, device, wmap, yy_shift[i], xx_shift[i])

# def compute_vis_cuda_shift(x, uua, vva, wwa, vis_real, vis_imag, w, g, ppb, cell_size, device, wmap, yy_shift, xx_shift):
#         uua = torch.from_numpy(uua).to(device)
#         vva = torch.from_numpy(vva).to(device)
#         wwa = torch.from_numpy(wwa).to(device)
#         pbb = torch.from_numpy(ppb).to(device)
#         vis_real = torch.from_numpy(vis_real).to(device)
#         vis_imag = torch.from_numpy(vis_imag).to(device)
#         w = torch.from_numpy(w).to(device)
#         g = torch.from_numpy(g).to(device)
        
#         #shift positions
#         xshift = torch.roll(x, shifts=(yy_shift,xx_shift), dims=(0,1))
#         xshift = torch.unsqueeze(xshift,0)

#         #move points to device
#         points = torch.zeros((2,len(uua)))
#         points[0] = -vva
#         points[1] = uua
#         points = points.to(device)

#         #apply pb
#         c = (xshift*pbb).to(torch.float32) + 1J * 0

#         #compute visibilities
#         model_vis = cell_size.value**2 * pytorch_finufft.functional.finufft_type2(points, torch.fft.fftshift(c), isign=1, modeord=1)
        
#         J1 = torch.nansum((model_vis.real - vis_real)**2 / w)
#         J11 = torch.nansum((model_vis.imag - vis_imag)**2 / w)
    
#         return J1 + J11

# def wproj(x, uua2, vva2, wwa2, ppb, device, cell_size):
#     #move points to device
#     points = torch.zeros((2,len(uua2)))
#     points[0] = -vva2
#     points[1] = uua2
#     points = points.to(device)
    
#     #apply pb
#     c = (x*ppb).to(torch.float32) + 1J * 0
    
#     #compute visibilities
#     model_vis = cell_size.value**2 * pytorch_finufft.functional.finufft_type2(points, torch.fft.fftshift(c), isign=1, modeord=1)
#     return model_vis

# def compute_vis_cuda_wproj(x, uua, vva, wwa, vis_real, vis_imag, w, g, ppb, cell_size, device, wmap):
#         uua = torch.from_numpy(uua).to(device)
#         vva = torch.from_numpy(vva).to(device)
#         wwa = torch.from_numpy(wwa).to(device)
#         ppb = torch.from_numpy(ppb).to(device)
#         vis_real = torch.from_numpy(vis_real).to(device)
#         vis_imag = torch.from_numpy(vis_imag).to(device)
#         w = torch.from_numpy(w).to(device)
#         g = torch.from_numpy(g).to(device)

#         J1 = 0.; J11 = 0.
#         #start loop of w-proj
#         for k in np.arange(10):
#             #find w range
#             widx = torch.where((wwa > -1000) & (wwa < 0))
            
#             uua2 = uua[widx]
#             vva2 = vva[widx]
#             wwa2 = wwa[widx]
#             w2 = w[widx]
#             vis_real2 = vis_real[widx]
#             vis_imag2 = vis_imag[widx]
            
#             model_vis = wproj(x, uua2, vva2, wwa2, ppb, device, cell_size)
#             #w-proj
#             # model_vis_real = (g.real*model_vis.real - g.imag*model_vis.imag)
#             # model_vis_imag = (g.real*model_vis.imag + g.imag*model_vis.real)        
#             # J1 = torch.nansum((model_vis_real - vis_real)**2 / w)# / lvis)
#             # J11 = torch.nansum((model_vis_imag - vis_imag)**2 / w)# / lvis)
            
#             J1 += torch.nansum((model_vis.real - vis_real2)**2 / w2)# / lvis)
#             J11 += torch.nansum((model_vis.imag - vis_imag2)**2 / w2)# / lvis)
            
#         return J1 + J11

# def compute_vis_cuda_rot(x, uua, vva, wwa, vis_real, vis_imag, w, g, ppb, cell_size, device, wmap, yy_shift, xx_shift):
#     uua = torch.from_numpy(uua).to(device)
#     vva = torch.from_numpy(vva).to(device)
#     wwa = torch.from_numpy(wwa).to(device)
#     pbb = torch.from_numpy(ppb).to(device)
#     vis_real = torch.from_numpy(vis_real).to(device)
#     vis_imag = torch.from_numpy(vis_imag).to(device)
#     w = torch.from_numpy(w).to(device)
#     g = torch.from_numpy(g).to(device)
    
#     #shift positions
#     xshift = torch.roll(x, shifts=(yy_shift,xx_shift), dims=(0,1))
#     xshift = torch.unsqueeze(xshift,0)
    
#     #move points to device
#     points = torch.zeros((2,len(uua)))
#     points[0] = -vva
#     points[1] = uua
#     points = points.to(device)

#     angle = torch.tensor([-45]).to(device)
#     xrot = kornia.geometry.transform.rotate(x*pbb, angle)[0]
#     # if delta[i] > 0:
#     #     rotater = v2.RandomRotation(degrees=(45,0))
#     # else:
#     rotater = v2.RandomRotation(degrees=(0,-45))
#     xxrot = torch.unsqueeze(xrot, 0)
#     xxxrot = rotater(xxrot)[0]
        
#     #apply pb
#     c = (xxxrot).to(torch.float32) + 1J * 0
    
#     #compute visibilities
#     model_vis = cell_size.value**2 * pytorch_finufft.functional.finufft_type2(points, torch.fft.fftshift(c), isign=1, modeord=1)
    
#     J1 = torch.nansum((model_vis.real - vis_real)**2 / w)
#     J11 = torch.nansum((model_vis.imag - vis_imag)**2 / w)
    
#     return J1 + J11

    
