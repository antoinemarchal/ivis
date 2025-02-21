# -*- coding: utf-8 -*-
import glob
import numpy as np
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy import optimize
from radio_beam import Beam
import torch
from tqdm import tqdm as tqdm
from numpy.fft import fft2
from torch.fft import fft2 as tfft2
from dataclasses import dataclass
from reproject import reproject_interp
import subprocess

import marchalib as ml

from deconv.utils import dunits, dutils, dformat, dms2npz, mod_loss, dcasacore, plotms

@dataclass #modified from MPol
class VisData:
    uu: np.ndarray
    vv: np.ndarray
    ww: np.ndarray
    sigma: np.ndarray
    data: np.ndarray
    beam: np.ndarray
    coords: np.ndarray
    frequency: np.ndarray


class DataVisualizer:
    def __init__(self, path_ms, path_beams, path_sd, pathout):
        super(DataVisualizer, self).__init__()
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout


    def msplot(self, idfile):
        #get msl from path
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        plotms.plotms(msl[idfile])


class DataProcessor:
    def __init__(self, path_ms, path_beams, path_sd, pathout):
        super(DataProcessor, self).__init__()
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout


    def fixms(self):
        #get msl from path
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        # Apply fix_ms_dir to each MS file
        for ms in msl:
            print(f"Processing {ms}...")
            subprocess.run(["fix_ms_dir", ms])  # Run the command for each MS file    

        print("All MS files processed.")
    

    def package_ms(self, filename, select_fraction=1, uvmin=0, uvmax=7000, nchan=1, start=0, width=1, inc=1):
        #get filenames of all ms from mspath
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        print("number of ms files = {}".format(len(msl)))        

        # get data
        frequency, uu, vv, ww, weight, sigma, data, flag, beam, ra_hms, dec_dms = dms2npz.get_baselines(msl, select_fraction=select_fraction, sigma_rescale=1.0, incl_model_data=False, datacolumn="data", nchan=nchan, start=start, width=width, inc=inc, uvmin=np.float32(uvmin), uvmax=np.float32(uvmax))

        print(data.shape, uu.shape, weight.shape, sigma.shape)
        
        print("write " + filename + " on disk")
        np.savez(
            self.pathout + filename,
            frequency=frequency, # [GHz]
            uu=uu, # [lambda]
            vv=vv, # [lambda]
            ww=ww, # [lambda]
            weight=weight, # [1/Jy^2]
            sigma=sigma, # assumed to be [Jy]
            data=data, # [Jy]       
            flag=flag, # [Bool]
            beam=beam, #beam position in the list
            ra_hms=ra_hms, #phase center Right ascension [h:m:s]
            dec_dms=dec_dms, #phase center Declination [h:m:s]
        )
        
        return None


    def compute_pb_and_grid(self, hdr, fitsname_pb=None, fitsname_grid=None):
        #shape image
        shape_img = (hdr["NAXIS1"],hdr["NAXIS2"])
        input_shape = (1,1,shape_img[0],shape_img[1])        

        #get beam files 
        filenames = sorted(glob.glob(self.path_beams+"*.fits"))
        n_beams = len(filenames)
        print("number of beams:", n_beams)
        #compute shape of scaled primary beam
        print("using {} to rescale the PB with cell_size of target hdr".format(filenames[0]))
        hdr_pb = fits.open(filenames[0])[0].header
        shape = (hdr_pb["NAXIS2"],hdr_pb["NAXIS1"])

        #ratio cell_size
        ratio = hdr_pb["CDELT2"] / hdr["CDELT2"]
        print("ratio pixel size PB and target: ", ratio)
        shape_out = (int(hdr_pb["NAXIS1"]*ratio),int(hdr_pb["NAXIS2"]*ratio))

        #init interpolation grid
        if round(ratio) != 1:
            reproj_pb = np.zeros((n_beams,shape_out[0],shape_out[1]))
            grid_array = np.zeros((n_beams,1,shape_out[0],shape_out[1],2))
        else:
            reproj_pb = np.zeros((n_beams,shape[0],shape[1]))
            grid_array = np.zeros((n_beams,1,shape[0],shape[1],2))
        for i in tqdm(np.arange(n_beams)):
            #open beam cube
            hdu_pb = fits.open(filenames[i])
            hdr_pb = hdu_pb[0].header
            pb = hdu_pb[0].data
            shape = (hdr_pb["NAXIS2"],hdr_pb["NAXIS1"])
            w_pb = ml.wcs2D(hdr_pb)
            input_header =  w_pb.to_header()

            if round(ratio) != 1:
                #update hdr to rescale
                hdr_pb["CDELT1"] = hdr["CDELT1"]
                hdr_pb["CDELT2"] = hdr["CDELT2"]
                hdr_pb["NAXIS1"] = shape_out[0]
                hdr_pb["NAXIS2"] = shape_out[1]
                hdr_pb["CRPIX1"] = int(hdr_pb["NAXIS1"] / 2)
                hdr_pb["CRPIX2"] = int(hdr_pb["NAXIS2"] / 2)
            
                #Reproj
                w_pb = ml.wcs2D(hdr_pb)
                target_header = w_pb.to_header()
                reproj, footprint = reproject_interp((pb,input_header), target_header, shape_out)
                reproj[reproj != reproj] = 0. #make sure NaN to 0 
                reproj_pb[i] = reproj
                
                wcs_in = ml.wcs2D(hdr)
                wcs_out = ml.wcs2D(hdr_pb)
                
                #Reshape tensor and get grid
                grid = dutils.get_grid(input_shape, wcs_in, wcs_out, shape_out)
                grid_array[i] = grid.detach().cpu().numpy()
                
            else:
                target_header = hdr
                reproj_pb[i] = pb

                wcs_in = ml.wcs2D(hdr)
                wcs_out = ml.wcs2D(hdr_pb)
                
                #Reshape tensor and get grid
                grid = dutils.get_grid(input_shape, wcs_in, wcs_out, shape)
                grid_array[i] = grid.detach().cpu().numpy()

        #NaN to 0
        reproj_pb[reproj_pb != reproj_pb] = 0.

        #Write on disk
        hdu0 = fits.PrimaryHDU(reproj_pb, header=target_header)
        hdulist = fits.HDUList([hdu0])
        hdulist.writeto(self.pathout + fitsname_pb, overwrite=True)
        
        hdu0 = fits.PrimaryHDU(grid_array, header=None)
        hdulist = fits.HDUList([hdu0])
        hdulist.writeto(self.pathout + fitsname_grid, overwrite=True)

        return None#reproj_pb, grid_array


    def read_vis(self, _npz, select_fraction=1):
        #read packaged data
        print("read {}".format(_npz))
        archive = np.load(self.pathout + _npz, allow_pickle=True)

        #Select subset of visibilities
        uu_lam, vv_lam, ww_lam, sigma, beam, data, coords, frequency = dformat.format_data(select_fraction, archive)

        # store everything as 1D
        vis_data = VisData(uu_lam, vv_lam, ww_lam, sigma, data, beam, coords, frequency)
        
        return vis_data

    
    def read_vis_from_scratch(self, uvmin=0, uvmax=7000, chunks=1.e7, target_frequency=1421104492.034763):
        #get filenames of all ms from mspath
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        print("number of ms files = {}".format(len(msl)))        
        
        vis_data = dcasacore.readmsl(msl, uvmin, uvmax, chunks, target_frequency)
                
        return vis_data #REMOVE ra and dec / should be useless from here


    def read_pb_and_grid(self, fitsname_pb, fitsname_grid):
        #read pre-computed pb
        hdu_grid = fits.open(self.pathout + fitsname_grid)
        #read pre-computed grid                
        hdu_pb = fits.open(self.pathout + fitsname_pb)

        return hdu_pb[0].data, hdu_grid[0].data


    def read_sd(self):
        sd = 0; beam_sd = 0
        
        return sd, beam_sd

    
class Imager:
    def __init__(self, vis_data, pb, grid, sd, beam_sd, hdr, max_its, lambda_sd, lambda_r, positivity, device):
        super(Imager, self).__init__()
        self.vis_data = vis_data
        self.pb = pb
        self.grid = grid
        self.sd = sd
        self.beam_sd = beam_sd
        self.hdr = hdr
        self.max_its = max_its
        self.lambda_sd = lambda_sd
        self.lambda_r = lambda_r
        self.positivity = positivity
        self.device = device
        

    def process(self, units, disk=False):
        #Image parameters
        cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        #tapper for apodization
        tapper = ml.edges.apodize(0.98, shape)
        
        #Convert lambda to radian per pixel
        uu_radpix = dunits._lambda_to_radpix(self.vis_data.uu, cell_size)
        vv_radpix = dunits._lambda_to_radpix(self.vis_data.vv, cell_size)
        ww_radpix = dunits._lambda_to_radpix(self.vis_data.ww, cell_size)

        #Build kernel for regularization
        kernel_map = dutils.laplacian(shape)
        fftkernel = abs(fft2(kernel_map))

        #generate fftbeam
        bmaj = self.beam_sd.major.value
        cdelt2 = cell_size.to(u.deg).value
        bmaj_pix = bmaj / cdelt2
        beam = dutils.gauss_beam(bmaj_pix, shape, FWHM=True)
        fftbeam = abs((fft2(beam)))

        #fft single-dish map
        fftsd = cell_size.value**2 * tfft2(torch.from_numpy(np.float32(self.sd))).numpy()

        #Get idx beams in array
        nb = len(self.vis_data.coords)
        idmin = np.zeros(nb); idmax = np.zeros(nb)
        for i in tqdm(np.arange(nb)):
            idmin[i] = np.where(self.vis_data.beam == i)[0][0];
            idmax[i] = len(np.where(self.vis_data.beam == i)[0])-1

        #init array
        init_params = np.zeros(shape).ravel() #Start with null map

        #define bounds for optimisation
        if self.positivity == False:
            bounds = dutils.ROHSA_bounds(data_shape=shape, lb_amp=-np.inf, ub_amp=np.inf)
        else:
            bounds = dutils.ROHSA_bounds(data_shape=shape, lb_amp=0, ub_amp=np.inf)
            
        # Use gradient-descent to minimise cost
        print('Starting optimisation')
        opt_output = optimize.minimize(mod_loss.objective, init_params.ravel().astype(np.float32),
                                       args=(
                                           beam.astype(np.float32),
                                           fftbeam.astype(np.float32),
                                           self.vis_data.data.astype(np.complex64),
                                           uu_radpix.astype(np.float32),
                                           vv_radpix.astype(np.float32),
                                           ww_radpix.astype(np.float32),
                                           self.pb.astype(np.float32),
                                           idmin.astype(np.int32),
                                           idmax.astype(np.int32),
                                           self.device,
                                           self.vis_data.sigma.astype(np.float32),
                                           fftsd.astype(np.complex64),
                                           tapper.astype(np.float32),
                                           self.lambda_sd,
                                           self.lambda_r,
                                           fftkernel.astype(np.float32),
                                           shape,
                                           cell_size.value, #in arcsec
                                           self.grid.astype(np.float32)
                                       ),
                                       jac=True,
                                       tol=1.e-8,
                                       bounds=bounds, method='L-BFGS-B',
                                       options={'maxiter': self.max_its, 'maxfun': 1e100, 'iprint': 1, 'disp': 2})

        print(opt_output)
        
        result = np.reshape(opt_output.x, shape)

        #unit conversion
        if units == "Jy/arcsec^2":
            return result

        if units == "Jy/beam":
            print("assuming a synthesized beam of 3 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            nu = self.vis_data.frequency *1.e9 *u.Hz
            beam_r = Beam(3*cell_size, 3*cell_size, 1.e-12*u.deg)
            return result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    

        elif units == "K":
            print("assuming a synthesized beam of 3 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            nu = self.vis_data.frequency *1.e9 *u.Hz
            beam_r = Beam(3*cell_size, 3*cell_size, 1.e-12*u.deg)
            result_Jy = result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    
            return (result_Jy*u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
            
        else: print("unit must be 'Jy/arcsec^2' or 'K'")            
    
