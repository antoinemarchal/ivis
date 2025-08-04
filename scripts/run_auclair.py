import time
import numpy as np
import torch
import scipy
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pywph as pw
import os
import sys
from astropy.io import fits

from comp_sep_functions import create_batch, compute_bias_std, compute_mask, compute_loss_BR, compute_loss_JMD

plt.ion()

print("GPU: ", torch.cuda.get_device_name(0))
print("NumPy",np.__version__)
print("Torch",torch.__version__)
print("SciPy",scipy.__version__)
print("PyWPH",pw.__version__)

def objective(x):
    """
    Computes the loss and the corresponding gradient.

    Parameters
    ----------
    x : torch 1D tensor
        Flattened running map.

    Returns
    -------
    float
        Loss value.
    torch 1D tensor
        Gradient of the loss.

    """
    global eval_cnt
    global loss_list
    start_time = time.time()
    u = x.reshape((N, N)) # Reshape x
    u = torch.from_numpy(u).to(device).requires_grad_(True) # Track operations on u
    if style == 'BR':
        # Compute the loss 'à la Bruno'
        L = compute_loss_BR(u, coeffs_target, std, mask, device, Mn, wph_op, noise, pbc) 
    if style == 'JMD':
        # Compute the loss 'à la Jean-Marc'
        L = compute_loss_JMD(u, coeffs_target, std, mask, device, wph_op, pbc) 
    u_grad = u.grad.cpu().numpy().astype(x.dtype) # Compute the gradient
    if eval_cnt % 5 == 0:
        print(f"Evaluation: {eval_cnt}")
        print("L = "+str(round(L.item(),5)))
        print("(computed in "+str(round(time.time() - start_time,3))+"s)")
        print("")
    eval_cnt += 1
    loss_list.append(L.item())
    return L.item(), u_grad.ravel()

#______________________________________________
Mn = 20

path = "/home/amarchal/Projects/ivis/docs/tutorials/data_tutorials/ivis_data/"

d = fits.open(path+"image.fits")[0].data
noise = fits.open(path+"noise_cube.fits")[0].data * np.sqrt(0.0936) #Scaling the noise

center_y, center_x = d.shape[0] // 2, d.shape[1] // 2

# Compute half-size
half_size = 512 // 2

# Slice indices
start_y = center_y - half_size
end_y = center_y + half_size
start_x = center_x - half_size
end_x = center_x + half_size

# Cut array
d = d[start_y:end_y, start_x:end_x].astype(np.float64)
noise = noise[:,start_y:end_y, start_x:end_x].astype(np.float64)

M, N = np.shape(d) # map size
J = int(np.log2(min(M,N)))-2 # number of scales
L = 4 # number of angles
pbc = True # periodic boundary conditions
dn = 5 # number of translations
wph_model = ["S11","S00","S01","Cphase","C01","C00","L"] # list of WPH coefficients

style = 'BR'#'JMD'
method = 'L-BFGS-B'
n_epoch_init = 3
n_epoch = 5
n_iter = 50
device = 0#"cpu"
batch_size = 5

batch_number = int(Mn/batch_size)
n_batch = create_batch(noise, device, batch_number, batch_size, M)

wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)

# Initialization of evaluation count.
eval_cnt = 0
# Initialization of the running map s_tilde0.
s_tilde0 = d
# Creation of the loss list.
loss_list = []
# WPH model loading (only the power-spectrum-like coefficients in the first step).
wph_op.load_model(["S11"])
# Loop of the epochs.
for i in range(n_epoch_init):
    print("Starting epoch "+str(i+1)+"...")
    # Bring s_tilde0 from array to tensor.
    s_tilde0 = torch.from_numpy(s_tilde0).to(device)
    print('Computing loss arguments...')
    # Computation of the noise-induced bias and std on the s_tilde0 map.
    # The bias is only used for style='JMD', but is computed
    # in both cases (no significant additional calculations).
    bias, std = compute_bias_std(s_tilde0, n_batch, wph_op, pbc, Mn, batch_number, batch_size, device)
    # Computation of the WPH statistics of "d".
    coeffs = wph_op.apply(torch.from_numpy(d).to(device), norm=None, pbc=pbc)
    if style == 'BR':
        # In BR's formalism, the target WPH coefficients are the ones of "d".
        # They are split into real and imaginary parts.
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs),dim=0),
                                   torch.unsqueeze(torch.imag(coeffs),dim=0)))
    if style == 'JMD':
        # In JMD's formalism, the target WPH coefficients are computed as
        # the ones of "d" corrected from the bias estimated before.
        # They are here also split into real and imaginary parts.
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs)-bias[0],dim=0),
                                   torch.unsqueeze(torch.imag(coeffs)-bias[1],dim=0)))
    # Computation of the mask for the WPH statistics threshold.
    mask = compute_mask(1, s_tilde0, std, wph_op, wph_model, pbc, device)
    print('Loss arguments computed !')
    print('Beginning optimization...')
    # Beginning of the optimization.
    result = opt.minimize(objective, s_tilde0.cpu().ravel(), method=method, jac=True, tol=None,
                          options={"maxiter": n_iter, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
    final_loss, s_tilde0, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    # Reshaping of the running map s_tilde0.
    s_tilde0 = s_tilde0.reshape((N, N)).astype(np.float64)
    print("Epoch "+str(i+1)+" done !")

# Initialization of evaluation count.
eval_cnt = 0
# Initialization of the running map s_tilde.
s_tilde = s_tilde0
# Creation of the loss list.
loss_list = []
# WPH model loading (all the WPH coefficients in the second step).
wph_op.load_model(wph_model)
# Loop of the epochs.
for i in range(n_epoch):
    print("Starting epoch "+str(i+1)+"...")
    # Bring s_tilde from array to tensor.
    s_tilde = torch.from_numpy(s_tilde).to(device)
    print('Computing loss arguments...')
    # Computation of the noise-induced bias and std on the s_tilde map.
    # The bias is only used for style='JMD', but is computed
    # in both cases (no significant additional calculations).
    bias, std = compute_bias_std(s_tilde, n_batch, wph_op, pbc, Mn, batch_number, batch_size, device)
    # Computation of the WPH statistics of "d".
    coeffs = wph_op.apply(torch.from_numpy(d).to(device), norm=None, pbc=pbc)
    if style == 'BR':
        # In BR's formalism, the target WPH coefficients are the ones of "d".
        # They are split into real and imaginary parts.
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs),dim=0),
                                   torch.unsqueeze(torch.imag(coeffs),dim=0)))
    if style == 'JMD':
        # In JMD's formalism, the target WPH coefficients are computed as
        # the ones of "d" corrected from the bias estimated before.
        # They are here also split into real and imaginary parts.
        coeffs_target = torch.cat((torch.unsqueeze(torch.real(coeffs)-bias[0],dim=0),
                                   torch.unsqueeze(torch.imag(coeffs)-bias[1],dim=0)))
    # Computation of the mask for the WPH statistics threshold.
    mask = compute_mask(2, s_tilde, std, wph_op, wph_model, pbc, device)
    print('Loss arguments computed !')
    print('Beginning optimization...')
    # Beginning of the optimization.
    result = opt.minimize(objective, s_tilde.cpu().ravel(), method=method, jac=True, tol=None,
                          options={"maxiter": n_iter, "gtol": 1e-14, "ftol": 1e-14, "maxcor": 20})
    final_loss, s_tilde, niter, msg = result['fun'], result['x'], result['nit'], result['message']
    # Reshaping of the running map s_tilde.
    s_tilde = s_tilde.reshape((N, N)).astype(np.float64)
    print("Epoch "+str(i+1)+" done !")
    # Plot of the running map.
    plt.imshow(s_tilde, vmax=8)

stop
    
#Write outpout
pathout = ""
fitsout = ""
print("Write output " + fitsout + " file on disk")
hdu0 = fits.PrimaryHDU(s_tilde)#, header=target_header)
hdulist = fits.HDUList([hdu0])
hdulist.writeto(pathout + fitsout, overwrite=True)

