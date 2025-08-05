import glob
from tqdm import tqdm as tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from astropy.io import fits
from radio_beam import Beam
from astropy import units as u
from reproject import reproject_interp
import pywph as pw
import torch
import pytorch_finufft
 
from ivis.io import DataProcessor
from ivis.imager import Imager
from ivis.logger import logger
from ivis.utils import dutils, mod_loss, fourier

from ivis.models import ClassicIViS, TWiSTModel

plt.ion()

def powspec_torch(image, pixel_size_arcsec=1.0, nbins=None, autocorr=False, im2=None):
    """
    Compute 1D power spectrum or autocorrelation using PyTorch.

    Parameters
    ----------
    image : torch.Tensor
        Input 2D tensor.
    pixel_size_arcsec : float
        Pixel size in arcseconds.
    nbins : int
        Number of radial bins.
    autocorr : bool
        If True, return autocorrelation instead of power spectrum.
    im2 : torch.Tensor, optional
        Optional second image for cross-spectrum.

    Returns
    -------
    k_bin_centers : torch.Tensor
        Radial frequency bins [arcmin^{-1}].
    power_1d : torch.Tensor
        Azimuthally averaged power spectrum.
    """
    device = image.device
    H, W = image.shape
    if nbins is None:
        nbins = max(H, W) // 2

    dx_arcmin = pixel_size_arcsec / 60.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=dx_arcmin)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=dx_arcmin)).to(device)
    kx, ky = torch.meshgrid(fx, fy, indexing='xy')
    k_radius = torch.sqrt(kx**2 + ky**2)

    # FFTs
    ft1 = torch.fft.fft2(image)
    if im2 is not None:
        ft2 = torch.fft.fft2(im2)
        ps2d = torch.real(ft1 * torch.conj(ft2)) / (H * W)
    else:
        ps2d = torch.real(ft1 * torch.conj(ft1)) / (H * W)

    if autocorr:
        ps2d = torch.fft.ifft2(ps2d).real

    ps_flat = torch.fft.fftshift(ps2d).flatten()
    k_flat = k_radius.flatten()

    kmin = k_flat[k_flat > 0].min()
    kmax = k_flat.max()
    bins = torch.linspace(kmin, kmax, nbins + 1, device=device)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    inds = torch.bucketize(k_flat, bins) - 1

    power_1d = torch.zeros(nbins, device=device)
    for i in range(nbins):
        mask = inds == i
        if torch.any(mask):
            power_1d[i] = ps_flat[mask].mean()

    return bin_centers, power_1d


def generate_noise_map_beamwise(
    uu, vv, idmina, idmaxa, sigma, shape, device
):
    """
    Generate a dirty noise map by beam using inverse NUFFT (adjoint of Type 2),
    consistent with your visibility modeling setup.

    Parameters
    ----------
    uu, vv : np.ndarray
        UV coordinates (radians/pixel).
    idmina, idmaxa : list of int
        Indices of start and length per beam.
    sigma : np.ndarray
        Visibility-domain standard deviation per visibility.
    shape : tuple
        (H, W) image shape.
    device : str
        Torch device, e.g., "cuda" or "cpu".

    Returns
    -------
    noise_map : torch.Tensor
        2D torch tensor [H, W] containing the summed dirty noise image over beams.
    """
    H, W = shape
    noise_map = torch.zeros(H, W, dtype=torch.float32, device=device)

    n_beams = len(idmina)
    for i in range(n_beams):
        idmin = idmina[i]
        idmax = idmaxa[i]

        # Grab UV coordinates and sigma for this beam
        uua = torch.from_numpy(uu[idmin:idmin+idmax]).to(device)
        vva = torch.from_numpy(vv[idmin:idmin+idmax]).to(device)
        siga = torch.from_numpy(sigma[idmin:idmin+idmax]).to(device)

        # Generate complex Gaussian visibility noise
        noise_real = torch.randn_like(siga)
        noise_imag = torch.randn_like(siga)
        noise_vis = torch.complex(noise_real, noise_imag) * siga

        # NUFFT points
        points = torch.stack([-vva, uua], dim=0)  # FINUFFT ordering

        # Inverse NUFFT (adjoint)
        noise_dirty = pytorch_finufft.functional.finufft_type1(
            points, noise_vis, shape, isign=-1, modeord=0
        )

        # Add real part to noise map (assumes real-valued imaging)
        noise_map += noise_dirty.real

    return noise_map


path_ms = "../docs/tutorials/data_tutorials/ivis_data/msl_mw/" #directory of measurement sets    
path_beams = "../docs/tutorials/data_tutorials/ivis_data/BEAMS/" #directory of primary beams
path_sd = None #path single-dish data
pathout = "../docs/tutorials/data_tutorials/ivis_data/" #path where data will be packaged and stored

#REF WCS INPUT USER
filename = "../docs/tutorials/data_tutorials/ivis_data/MW-C10_mom0th_NHI.fits"
target_header = fits.open(filename)[0].header
shape = (target_header["NAXIS2"],target_header["NAXIS1"])

#create data processor
data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)

# pre-compute pb and interpolation grids — this can be commented after first compute
logger.disabled = True
data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits") 
logger.disabled = False

pb, grid = data_processor.read_pb_and_grid(fitsname_pb="reproj_pb.fits", fitsname_grid="grid_interp.fits")

#Dummy sd array
sd = np.zeros(shape)
#Dummy Beam sd
beam_sd = Beam(1*u.deg, 1*u.deg, 1.e-12*u.deg)

#Read data
vis_data = data_processor.read_vis_from_scratch(uvmin=0, uvmax=np.inf,
                                                target_frequency=None,
                                                target_channel=0,
                                                extension=".ms",
                                                blocks='single',
                                                max_workers=1)


#user parameters
max_its = 1
lambda_sd = 0
lambda_r = 1
device = 0#"cpu" #0 is GPU and "cpu" is CPU
positivity = False

#init parameters
init_params = np.zeros((2,shape[0],shape[1]))

# create image processor
image_processor = Imager(vis_data,      # visibilities
                         pb,            # array of primary beams
                         grid,          # array of interpolation grids
                         sd,            # single dish data in unit of Jy/arcsec^2
                         beam_sd,       # beam of single-dish data in radio_beam format
                         target_header, # header on which to image the data
                         init_params[0],# init array of parameters
                         max_its,       # maximum number of iterations
                         lambda_sd,     # hyper-parameter single-dish
                         lambda_r,      # hyper-parameter regularization
                         positivity,    # impose a positivity constaint
                         device,        # device: 0 is GPU; "cpu" is CPU
                         beam_workers=1)
# choose model
model = ClassicIViS()
# get image
base = image_processor.process(model=model, units="Jy/arcsec^2") #"Jy/arcsec^2" or "K"
ks, ps_base = powspec_torch(torch.from_numpy(base).to(device) , pixel_size_arcsec=7)

stop

#generate noise per beam
idmina, idmaxa = image_processor.process_beam_positions()

noise_cube = np.zeros((n_beams, 20, pb.shape[1], pb.shape[2]), dtype=np.float32)

NN = 5
n_beams = vis_data.beam.max()+1
for beam_id in range(n_beams):
    idmin = idmina[beam_id]
    idmax = idmaxa[beam_id]  # number of visibilities (not end index)

    u_b = vis_data.uu[idmin:idmin+idmax]
    v_b = vis_data.vv[idmin:idmin+idmax]
    sig_b = vis_data.sigma[idmin:idmin+idmax]

    u_b = torch.tensor(u_b, dtype=torch.float32, device="cuda")
    v_b = torch.tensor(v_b, dtype=torch.float32, device="cuda")
    sig_b = torch.tensor(sig_b, dtype=torch.float32, device="cuda")

    points = torch.stack([-v_b, u_b], dim=0)

    for k in range(NN):
        noise_real = torch.randn(len(sig_b), device="cuda") * sig_b
        noise_imag = torch.randn(len(sig_b), device="cuda") * sig_b
        noise_vis = torch.complex(noise_real, noise_imag)

        dirty = pytorch_finufft.functional.finufft_type1(
            points, noise_vis, (pb.shape[1],pb.shape[2]), isign=-1, modeord=0
        ).real

        noise_cube[beam_id, k] = dirty.detach().cpu().numpy()

## WPH noise stat (start from pre-computed noise cube here for testing)
#params WPH
logger.info("Get WPH operator and load moments model.")
M, N = shape # map size
J = int(np.log2(min(M, N)))-2 # number of scales
L = 4 # number of angles
pbc = False # periodic boundary conditions
dn = 5 # number of translations
# wph_model = ["S11","S00","S01","Cphase","C01","C00","L"] # list of WPH coefficients
# wph_model = ["S11","S00","S01","Cphase"] # list of WPH coefficients
wph_model = ["S11"] # list of WPH coefficients
# logger.warning("Only using S11.")
# get operator
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
wph_op.load_model(wph_model)

# Open noise data cube
with fits.open(pathout + "noise_cube.fits", memmap=True) as hdul:
    data = hdul[0].data[:5]
    if not data.dtype.isnative:
        data = data.byteswap().view(data.dtype.newbyteorder('='))

noise_cube = data * np.sqrt(0.0936)

sigma_n = np.std(noise_cube)

#P(k)
ps_list = []
for i in range(noise_cube.shape[0]):
    noise_map = torch.from_numpy(noise_cube[i]).to(device)
    ks, ps = powspec_torch(noise_map, pixel_size_arcsec=7)
    ps_list.append(ps)

mean_noise_ps = torch.stack(ps_list).mean(dim=0)

#rescale
noise_cube /= 1e-5
logger.warning("normalized noise cube")

# Compute coeffs
n_noise = noise_cube.shape[0]
coeffs_list=[]
for i in tqdm(np.arange(n_noise)):
    coeffs = wph_op.apply(torch.from_numpy(noise_cube[i]).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32), norm=None, pbc=pbc)
    coeffs_list.append(coeffs)
    
coeffs_list_cpu = [c.detach().cpu() for c in coeffs_list]
mu = torch.stack(coeffs_list_cpu).mean(dim=0)
std = torch.stack(coeffs_list_cpu).std(dim=0)

stop

# hdu0 = fits.PrimaryHDU(base)#, header=target_header)
# hdulist = fits.HDUList([hdu0])
# hdulist.writeto(pathout + "image.fits", overwrite=True)

# coeffs = wph_op.apply(torch.from_numpy(base*1.e5).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32), norm=None, pbc=pbc)

# init_params[0] = base
# init_params[1] = noise_cube[0] * 1.e-5

# create image processor
image_processor = Imager(vis_data,      # visibilities
                         pb,            # array of primary beams
                         grid,          # array of interpolation grids
                         sd,            # single dish data in unit of Jy/arcsec^2
                         beam_sd,       # beam of single-dish data in radio_beam format
                         target_header, # header on which to image the data
                         init_params,   # init array of parameters
                         max_its,       # maximum number of iterations
                         lambda_sd,     # hyper-parameter single-dish
                         lambda_r,      # hyper-parameter regularization
                         positivity,    # impose a positivity constaint
                         device,        # device: 0 is GPU; "cpu" is CPU
                         beam_workers=1)
# choose model
model = TWiSTModel(wph_op, 0.e8, True, mu, std, mean_noise_ps, 10, 10)
# get image
result = image_processor.process(model=model, units="Jy/arcsec^2") #"Jy/arcsec^2" or "K"
