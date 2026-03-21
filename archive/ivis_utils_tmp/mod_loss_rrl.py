import os
import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2
from tqdm import tqdm as tqdm
from multiprocessing import Pool
from joblib import Parallel, delayed
import gc

from ivis import logger

def format_input_tensor(input_tensor):
    #ensure the input tensor has 4 dimensions
    if input_tensor.dim() == 2:  # If shape is [H_in, W_in]
        #add batch and channel dims
        input_tensor_reshape = input_tensor.unsqueeze(0).unsqueeze(0) 
    elif input_tensor.dim() == 3: # If shape is [C, H_in, W_in]
        #add batch dim
        input_tensor_reshape = input_tensor.unsqueeze(0)  
        
    return input_tensor_reshape


def objective(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, shape, cell_size, grid_array, beam_workers, frequency):
    #reshape x into u grid
    u = x.reshape(shape)
    #track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True) 
    # L = compute_loss_low_memory(u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array)
    L = compute_loss(u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array, frequency)
    #compute the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)

    logger.info(f"[PID {os.getpid()}] Total cost: {np.format_float_scientific(L.item(), precision=5)}")

    return L.item(), u_grad.ravel()


def compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig, pb, cell_size, device, grid, frequency, f0):
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

    amp = x[0]; alpha = x[1]

    frequency = torch.tensor(frequency, device=device, dtype=torch.float32)
    f0 = torch.tensor(f0, device=device, dtype=torch.float32)

    model = amp #* (frequency/f0)**-alpha
    
    #reproject field at beam position
    input_tensor = format_input_tensor(model).float()
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


def compute_loss(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array, frequency):
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
        J = compute_vis_cuda(x, uua, vva, wwa, vis_real, vis_imag, sig, pb[i], cell_size, device, grid_array[i], frequency[i], frequency[0])
        #add to total loss
        loss_tot += 0.5 * J
        
    #REGULARIZATION
    for i in np.arange(x.shape[0]):
        xfft2 = tfft2(x[i])
        conv = cell_size**2 * xfft2 * torch.from_numpy(fftkernel).to(device)
        R = torch.nansum(abs(conv)**2)
        loss_tot += 0.5 * R * lambda_r
    
    loss_tot.backward(retain_graph=True)

    return loss_tot.detach().cpu()


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


def compute_loss_Pool(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma,
                      fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array,
                      beam_workers=4):
    
    beam = torch.from_numpy(beam).to(device)
    loss_tot = torch.zeros(1, device=device)
    n_beams = len(idmina)
    
    # --- PARALLEL ON CPU WITH BATCHED BEAMS ---
    if device == "cpu":
        # Split beam indices into batches
        beam_indices = np.array_split(np.arange(n_beams), beam_workers)
        
        # Parallel execution
        results = Parallel(n_jobs=beam_workers)(
            delayed(batch_worker)(
                batch, x, uu, vv, ww, data, sigma, pb, idmina, idmaxa, cell_size, device, grid_array
            ) for batch in beam_indices
        )
        
        for partial_loss in results:
            loss_tot += partial_loss.to(device)

    # --- SEQUENTIAL ON GPU ---
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
            loss_tot += 0.5 * J
            

    # --- SINGLE DISH COMPONENT ---
    fftsd = torch.from_numpy(fftsd).to(device)
    tapper = torch.from_numpy(tapper).to(device)
    xfft2 = tfft2(x * tapper)
    model_sd = cell_size**2 * xfft2 * torch.from_numpy(fftbeam).to(device)

    J2 = torch.nansum((model_sd.real - fftsd.real) ** 2)
    J22 = torch.nansum((model_sd.imag - fftsd.imag) ** 2)
    loss_tot += 0.5 * (J2 + J22) * lambda_sd

    # --- REGULARIZATION ---
    conv = cell_size**2 * xfft2 * torch.from_numpy(fftkernel).to(device)
    R = torch.nansum(abs(conv) ** 2)
    loss_tot += 0.5 * R * lambda_r

    loss_tot.backward(retain_graph=True)

    return loss_tot.detach().cpu()
