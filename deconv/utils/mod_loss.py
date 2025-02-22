import numpy as np
import torch
import pytorch_finufft
from torch.fft import fft2 as tfft2
from tqdm import tqdm as tqdm

from deconv import logger  # Import the logger

def format_input_tensor(input_tensor):
    #ensure the input tensor has 4 dimensions
    if input_tensor.dim() == 2:  # If shape is [H_in, W_in]
        #add batch and channel dims
        input_tensor_reshape = input_tensor.unsqueeze(0).unsqueeze(0) 
    elif input_tensor.dim() == 3: # If shape is [C, H_in, W_in]
        #add batch dim
        input_tensor_reshape = input_tensor.unsqueeze(0)  
        
    return input_tensor_reshape


def objective(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, shape, cell_size, grid_array):
    #reshape x into u grid
    u = x.reshape(shape)
    #track operations on u
    u = torch.from_numpy(u).to(device).requires_grad_(True) 
    # L = compute_loss_low_memory(u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array)
    L = compute_loss(u, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array)
    #compute the gradient
    u_grad = u.grad.cpu().numpy().astype(x.dtype)

    logger.info(f"Total cost: {np.format_float_scientific(L.item(), precision=5)}")

    return L.item(), u_grad.ravel()


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


def compute_loss(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array):
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

#FIXME
def compute_loss_low_memory(x, beam, fftbeam, data, uu, vv, ww, pb, idmina, idmaxa, device, sigma, fftsd, tapper, lambda_sd, lambda_r, fftkernel, cell_size, grid_array):
    # Move beam to device (only once)
    beam = torch.from_numpy(beam).to(device)

    # Initialize total loss
    loss_tot = torch.zeros(1).to(device)

    # INTERFEROMETER - Joint loop over beams
    n_beams = len(idmina)
    for i in np.arange(n_beams):
        idmin = idmina[i]
        idmax = idmaxa[i]

        # Slice and move only the necessary data to the device for NuFFT computation
        uua = torch.from_numpy(uu[idmin:idmin+idmax]).to(device)
        vva = torch.from_numpy(vv[idmin:idmin+idmax]).to(device)
        wwa = torch.from_numpy(ww[idmin:idmin+idmax]).to(device)
        pb_i = torch.from_numpy(pb[i]).to(device)
        grid_i = torch.from_numpy(grid_array[i]).to(device)

        # Create points tensor for NuFFT
        points = torch.zeros((2, len(uua)), device=device)
        points[0] = -vva
        points[1] = uua

        # Reproject field at beam position
        input_tensor = format_input_tensor(x).float()
        # Perform interpolation
        reprojected_tensor = torch.nn.functional.grid_sample(input_tensor, grid_i, mode='bilinear', align_corners=True)
        # Remove batch and channel dimensions
        reprojected_tensor = reprojected_tensor.squeeze()

        # Apply primary beam
        reproj = reprojected_tensor * pb_i

        # Pack into complex number for NuFFT
        c = reproj.to(torch.complex64)

        # Compute visibilities with NuFFT (only transfer the necessary data for this operation)
        model_vis = cell_size**2 * pytorch_finufft.functional.finufft_type2(points, c, isign=1, modeord=0)

        # Free memory after each iteration
        del uua, vva, wwa, pb_i, grid_i, points, reprojected_tensor, reproj, c
        torch.cuda.empty_cache()

        # Slice the data for the current beam for the cost function calculation
        vis_real = torch.from_numpy(data.real[idmin:idmin+idmax]).to(device)
        vis_imag = torch.from_numpy(data.imag[idmin:idmin+idmax]).to(device)
        sig = torch.from_numpy(sigma[idmin:idmin+idmax]).to(device)

        # Compute cost (real and imag)
        J1 = torch.nansum((model_vis.real - vis_real)**2 / sig**2)
        J11 = torch.nansum((model_vis.imag - vis_imag)**2 / sig**2)

        # Free memory after each iteration
        del vis_real, vis_imag, sig
        torch.cuda.empty_cache()

        # Add to total loss
        loss_tot += 0.5 * (J1 + J11)

    # SINGLE DISH
    fftsd = torch.from_numpy(fftsd).to(device)
    tapper = torch.from_numpy(tapper).to(device)
    # FFT2 of tapper sky image
    xfft2 = tfft2(x * tapper)
    # Convolution in Fourier space to apply single dish beam
    model_sd = cell_size**2 * xfft2 * torch.from_numpy(fftbeam).to(device)
    # Residual real and imag
    J2 = torch.nansum((model_sd.real - fftsd.real)**2)
    J22 = torch.nansum((model_sd.imag - fftsd.imag)**2)
    # Add to total loss
    loss_tot += 0.5 * (J2 + J22) * lambda_sd

    # REGULARIZATION
    conv = cell_size**2 * xfft2 * torch.from_numpy(fftkernel).to(device)
    R = torch.nansum(abs(conv)**2)
    loss_tot += 0.5 * R * lambda_r

    loss_tot.backward(retain_graph=True)

    return loss_tot.detach().cpu()






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

    
