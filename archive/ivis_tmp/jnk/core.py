# -*- coding: utf-8 -*-
import os
import glob
import sys
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
import tarfile
import concurrent.futures
from pathlib import Path

from ivis.logger import logger
from ivis.utils import dutils
# from ivis.utils import dunits, dutils, dformat, dms2npz, mod_loss, dcasacore, plotms

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

# DataVisualizer class    
class DataVisualizer:
    def __init__(self, path_ms, path_beams, path_sd, pathout):
        super(DataVisualizer, self).__init__()
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout
        logger.info("[Initialize DataVisualizer]")


    def msplot(self, idfile):
        #get msl from path
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        logger.info("Plotting XX for ant2&3 vs channel and velocity.")
        plotms.plotms(msl[idfile])


# DataProcessor class    
class DataProcessor:
    def __init__(self, path_ms, path_beams, path_sd, pathout):
        super(DataProcessor, self).__init__()
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout
        logger.info("[Initialize DataProcessor ]")
        

    def fixms(self): #fixme ran in parallel 
        #get msl from path
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        # Apply fix_ms_dir to each MS file
        for ms in msl:
            logger.info(f"Processing {ms}...")
            subprocess.run(["fix_ms_dir", ms])  # Run the command for each MS file    

        logger.info("All MS files processed.")


    def extract_tar(self, tar_file, clear=False):
        """Extracts a .tar file and optionally deletes the .tar file after extraction"""
        output_dir = os.path.dirname(tar_file)  # Extract in the same directory
        logger.info(f"Starting extraction: {tar_file}")

        try:
            with tarfile.open(tar_file) as tar:
                tar.extractall(path=output_dir)
            logger.info(f"Finished extracting: {tar_file}")
        except Exception as e:
            logger.warning(f"Error extracting {tar_file}: {e}")

        if clear:
            try:
                os.remove(tar_file)  # Delete the .tar file
                logger.info(f"Deleted .tar file: {tar_file}")
            except Exception as e:
                logger.warning(f"Error deleting {tar_file}: {e}")

        return tar_file  # Return tar_file for deletion after processing


    def untardir(self, max_workers=4, clear=False):
        """Recursively extracts all .tar files in the given directory and subdirectories in sorted order"""
        if not os.path.isdir(self.path_ms):
            logger.warning(f"Invalid directory: {self.path_ms}")
            return

        # # Ask user to confirm before starting extraction process
        # confirm = input(f"Do you want to start extracting and possibly delete all .tar files in {self.path_ms}? (y/n): ")
        # if confirm.lower() != 'y':
        #     logger.info("Extraction process canceled.")
        #     return

        # Get all .tar files, convert Path objects to strings, and sort them
        tar_files = sorted(str(p) for p in Path(self.path_ms).rglob("*.tar"))

        if not tar_files:
            logger.info("No .tar files found.")
            return

        logger.info(f"Found {len(tar_files)} .tar files. Starting extraction...")

        # Extract files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.extract_tar, tar_files, [clear] * len(tar_files))

        logger.info("All .tar files have been processed.")
            

    def read_vis_from_scratch(self, uvmin=0, uvmax=7000, chunks=1.e7, target_frequency=None, target_channel=0, extension=".ms", blocks="single", max_workers=1):
        if blocks == 'single':
            logger.info("Processing single scheduling block.")

            # Get filenames of all ms files
            msl = sorted(glob.glob(os.path.join(self.path_ms, f"*{extension}")))
            logger.info("Number of MS files = {}".format(len(msl)))

            if max_workers > 1:
                logger.info("Processing read ms files in parallel.")
                vis_data = dcasacore.readmsl(msl, uvmin, uvmax, target_frequency, target_channel,
                                             max_workers)
            else:
                logger.info("Processing read ms files with single thread.")
                vis_data = dcasacore.readmsl_no_parallel(msl, uvmin, uvmax, target_frequency,
                                                         target_channel)
            return vis_data

        elif blocks == 'multiple':
            logger.info("Processing multiple scheduling blocks.")

            subdirs = [d for d in sorted(os.listdir(self.path_ms)) if os.path.isdir(os.path.join(self.path_ms, d))]
            if not subdirs:
                logger.error("No subdirectories found in the path.")
                sys.exit()

            all_vis_data = []

            for subdir in subdirs:
                subdir_path = os.path.join(self.path_ms, subdir)
                msl = sorted(glob.glob(os.path.join(subdir_path, f"*{extension}")))

                if not msl:
                    logger.warning(f"No MS files found in {subdir_path}, skipping...")
                    continue

                if max_workers > 1:
                    logger.info("Processing read ms files in parallel.")
                    vis_data = dcasacore.readmsl(msl, uvmin, uvmax, target_frequency, target_channel,
                                                 max_workers)
                else:
                    logger.info("Processing read ms files with single thread.")
                    vis_data = dcasacore.readmsl_no_parallel(msl, uvmin, uvmax, target_frequency,
                                                 target_channel)
                    
                all_vis_data.append(vis_data)

            if not all_vis_data:
                logger.error("No valid data found across subdirectories.")
                sys.exit()

            # Concatenate the data
            concatenated_data = self.concatenate_vis_data(all_vis_data)
            return concatenated_data

        else:
            logger.error("Provide 'single' or 'multiple' for blocks.")
            sys.exit()


    @staticmethod
    def concatenate_vis_data(vis_data_list):
        """Concatenates a list of VisData objects along the first axis and sorts all arrays based on the beam array, keeping coords and frequency as they are."""
        
        # Concatenate all arrays (except for coords and frequency)
        uu = np.concatenate([v.uu for v in vis_data_list])
        vv = np.concatenate([v.vv for v in vis_data_list])
        ww = np.concatenate([v.ww for v in vis_data_list])
        sigma = np.concatenate([v.sigma for v in vis_data_list])
        data = np.concatenate([v.data for v in vis_data_list])
        beam = np.concatenate([v.beam for v in vis_data_list])
        
        # Get the indices that would sort the beam array
        sort_indices = np.argsort(beam)
        
        # Apply the sorting indices to all arrays to keep them aligned
        uu = uu[sort_indices]
        vv = vv[sort_indices]
        ww = ww[sort_indices]
        sigma = sigma[sort_indices]
        data = data[sort_indices]
        beam = beam[sort_indices]  # Reorder the beam array using sort_indices
        
        # Keep the first value of coords and frequency (without concatenation)
        coords = vis_data_list[0].coords
        frequency = vis_data_list[0].frequency
        
        return VisData(
            uu=uu, 
            vv=vv, 
            ww=ww, 
            sigma=sigma, 
            data=data, 
            beam=beam, 
            coords=coords, 
            frequency=frequency
        )


    def compute_pb_and_grid(self, hdr, fitsname_pb=None, fitsname_grid=None):
        #shape image
        shape_img = (hdr["NAXIS1"],hdr["NAXIS2"])
        input_shape = (1,1,shape_img[0],shape_img[1])        

        #get beam files 
        filenames = sorted(glob.glob(self.path_beams+"*.fits"))
        n_beams = len(filenames)
        logger.info("number of beams:", n_beams)
        #compute shape of scaled primary beam
        logger.info("using {} to rescale the PB with cell_size of target hdr".format(filenames[0]))
        hdr_pb = fits.open(filenames[0])[0].header
        shape = (hdr_pb["NAXIS2"],hdr_pb["NAXIS1"])

        #ratio cell_size
        ratio = hdr_pb["CDELT2"] / hdr["CDELT2"]
        logger.info("ratio pixel size PB and target: ", ratio)
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
            w_pb = dutils.wcs2D(hdr_pb)
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
                w_pb = dutils.wcs2D(hdr_pb)
                target_header = w_pb.to_header()
                reproj, footprint = reproject_interp((pb,input_header), target_header, shape_out)
                reproj[reproj != reproj] = 0. #make sure NaN to 0 
                reproj_pb[i] = reproj
                
                wcs_in = dutils.wcs2D(hdr)
                wcs_out = dutils.wcs2D(hdr_pb)
                
                #Reshape tensor and get grid
                grid = dutils.get_grid(input_shape, wcs_in, wcs_out, shape_out)
                grid_array[i] = grid.detach().cpu().numpy()
                
            else:
                target_header = hdr
                reproj_pb[i] = pb

                wcs_in = dutils.wcs2D(hdr)
                wcs_out = dutils.wcs2D(hdr_pb)
                
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


    def read_pb_and_grid(self, fitsname_pb, fitsname_grid):
        #read pre-computed pb
        hdu_grid = fits.open(self.pathout + fitsname_grid)
        #read pre-computed grid                
        hdu_pb = fits.open(self.pathout + fitsname_pb)

        return hdu_pb[0].data, hdu_grid[0].data


    def read_sd(self):
        sd = 0; beam_sd = 0
        
        return sd, beam_sd

# Imager class    
class Imager:
    def __init__(self, vis_data, pb, grid, sd, beam_sd, hdr, init_params, max_its, lambda_sd, lambda_r, positivity, device):
        super(Imager, self).__init__()
        self.vis_data = vis_data
        self.pb = pb
        self.grid = grid
        self.sd = sd
        self.beam_sd = beam_sd
        self.hdr = hdr
        self.init_params = init_params
        self.max_its = max_its
        self.lambda_sd = lambda_sd
        self.lambda_r = lambda_r
        self.positivity = positivity
        logger.info("[Initialize Imager        ]")
        logger.info(f"Number of iterations to be performed by the optimizer: {self.max_its}")

        # Logger for hyper-parameters
        if self.lambda_r == 0: logger.warning("lambda_r = 0 - No spatial regularization.")
        if self.lambda_sd == 0: logger.warning("lambda_sd = 0 - No short spacing correction (ignoring single dish data).")

        # Check if CUDA is found on the machine and fall back on CPU otherwise
        self.device = self.get_device(device)
        if self.device == 0: logger.info(f"Using GPU device: {torch.cuda.get_device_name(device)}")


    @staticmethod
    def get_device(user_device):
        if user_device == 0:  # User requested GPU
            try:
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
                else:
                    raise RuntimeError("CUDA not available.")
            except RuntimeError as e:
                logger.warning(f"{e} Falling back on CPU.")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU.")

        return device


    def process_beam_positions(self):
        """
        Computes the first and last occurrence indices of each beam in self.vis_data.beam.
        
        Returns:
        idmin (np.ndarray): Array of first occurrence indices for each beam.
        idmax (np.ndarray): Array of last occurrence indices for each beam.
        """
        # nb = len(self.vis_data.coords)
        # idmin = np.zeros(nb); idmax = np.zeros(nb)
        # for i in np.arange(nb):
        #     idmin[i] = np.where(self.vis_data.beam == i)[0][0];
        #     idmax[i] = len(np.where(self.vis_data.beam == i)[0])#-1
        nb = len(self.vis_data.coords)
        
        # Find unique beam indices and their first occurrence
        unique_beams, first_idx = np.unique(self.vis_data.beam, return_index=True)
        
        # Get counts of occurrences per beam
        beam_counts = np.bincount(self.vis_data.beam, minlength=nb)
        
        # Initialize arrays
        idmin = np.zeros(nb, dtype=int)
        idmax = np.zeros(nb, dtype=int)
        
        # Assign first index
        idmin[unique_beams] = first_idx
        
        # Assign (count - 1) for each beam
        idmax[unique_beams] = beam_counts[unique_beams] #- 1
        
        return idmin, idmax


    def process(self, units, disk=False):
        #Image parameters
        cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
        shape = (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
        #tapper for apodization
        tapper = dutils.apodize(0.98, shape)
        
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
        # logger.info("Processing beams position..")
        idmin, idmax = self.process_beam_positions()

        #define bounds for optimisation
        if self.positivity == False:
            bounds = dutils.ROHSA_bounds(data_shape=shape, lb_amp=-np.inf, ub_amp=np.inf)
        else:
            bounds = dutils.ROHSA_bounds(data_shape=shape, lb_amp=0, ub_amp=np.inf)
            
        # Use gradient-descent to minimise cost
        logger.info('Starting optimisation (using LBFGS-B)')
        if self.positivity == True:
            logger.info('Optimizer bounded - Positivity == True')
            logger.warning('Optimizer bounded - Because there is noise in the data, it is generally not recommanded to add a positivity constaint.')
        else:
            logger.info('Optimizer not bounded - Positivity == False')
                
        # Precompute type conversions (done once)
        params_f32 = self.init_params.ravel().astype(np.float32)
        beam_f32 = np.asarray(self.vis_data.beam, dtype=np.float32)
        fftbeam_f32 = np.asarray(fftbeam, dtype=np.float32)
        data_c64 = np.asarray(self.vis_data.data, dtype=np.complex64)
        uu_f32 = np.asarray(uu_radpix, dtype=np.float32)
        vv_f32 = np.asarray(vv_radpix, dtype=np.float32)
        ww_f32 = np.asarray(ww_radpix, dtype=np.float32)
        pb_f32 = np.asarray(self.pb, dtype=np.float32)
        idmin_i32 = np.asarray(idmin, dtype=np.int32)
        idmax_i32 = np.asarray(idmax, dtype=np.int32)
        sigma_f32 = np.asarray(self.vis_data.sigma, dtype=np.float32)
        fftsd_c64 = np.asarray(fftsd, dtype=np.complex64)
        tapper_f32 = np.asarray(tapper, dtype=np.float32)
        fftkernel_f32 = np.asarray(fftkernel, dtype=np.float32)
        grid_f32 = np.asarray(self.grid, dtype=np.float32)
        
        opt_args = (
            beam_f32, fftbeam_f32, data_c64, uu_f32, vv_f32, ww_f32,
            pb_f32, idmin_i32, idmax_i32, self.device, sigma_f32, fftsd_c64,
            tapper_f32, self.lambda_sd, self.lambda_r, fftkernel_f32, shape,
            cell_size.value, grid_f32
        )
        
        options = {
            'maxiter': self.max_its,
            'maxfun': int(1e6),
            'iprint': 1,
        }
        
        # Run optimization
        opt_output = optimize.minimize(
            mod_loss.objective, params_f32,
            args=opt_args,
            jac=True,
            tol=1.e-8,
            bounds=bounds,
            method='L-BFGS-B',
            options=options
        )


        # logger.info(opt_output)        
        result = np.reshape(opt_output.x, shape)

        #unit conversion
        if units == "Jy/arcsec^2":
            return result

        if units == "Jy/beam":
            logger.info("assuming a synthesized beam of 3 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            nu = self.vis_data.frequency *1.e9 *u.Hz
            beam_r = Beam(3*cell_size, 3*cell_size, 1.e-12*u.deg)
            return result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    

        elif units == "K":
            logger.info("assuming a synthesized beam of 3 x cell_size")
            cell_size = (self.hdr["CDELT2"] *u.deg).to(u.arcsec)
            nu = self.vis_data.frequency *1.e9 *u.Hz
            beam_r = Beam(3*cell_size, 3*cell_size, 1.e-12*u.deg)
            result_Jy = result * (beam_r.sr).to(u.arcsec**2).value #Jy/arcsec^2 to Jy/beam    
            return (result_Jy*u.Jy).to(u.K, u.brightness_temperature(nu, beam_r)).value
            
        else: logger.info("unit must be 'Jy/arcsec^2' or 'K'")            
    


# opt_output = optimize.minimize(mod_loss.objective, self.init_params.ravel().astype(np.float32),
#                                        args=(
#                                            beam.astype(np.float32),
#                                            fftbeam.astype(np.float32),
#                                            self.vis_data.data.astype(np.complex64),
#                                            uu_radpix.astype(np.float32),
#                                            vv_radpix.astype(np.float32),
#                                            ww_radpix.astype(np.float32),
#                                            self.pb.astype(np.float32),
#                                            idmin.astype(np.int32),
#                                            idmax.astype(np.int32),
#                                            self.device,
#                                            self.vis_data.sigma.astype(np.float32),
#                                            fftsd.astype(np.complex64),
#                                            tapper.astype(np.float32),
#                                            self.lambda_sd,
#                                            self.lambda_r,
#                                            fftkernel.astype(np.float32),
#                                            shape,
#                                            cell_size.value, #in arcsec
#                                            self.grid.astype(np.float32)
#                                        ),
#                                        jac=True,
#                                        tol=1.e-8,
#                                        bounds=bounds, method='L-BFGS-B',
#                                        options={'maxiter': self.max_its, 'maxfun': 1e6, 'iprint': 1, 'disp': 2})

# def read_vis_from_scratch(self, uvmin=0, uvmax=7000, chunks=1.e7, target_frequency=None, target_channel=0, extension=".ms", blocks="single"):
    #     if blocks == 'single':
    #         logger.info("processing single scheduling block.")
            
    #         #get filenames of all ms from mspath
    #         msl = sorted(glob.glob(self.path_ms+"*"+extension))
    #         logger.info("number of ms files = {}".format(len(msl)))
            
    #         vis_data = dcasacore.readmsl(msl, uvmin, uvmax, target_frequency, target_channel)
            
    #         return vis_data
        
    #     elif blocks == 'multiple':
    #         logger.info("Processing multiple scheduling blocks.")
    #         logger.warning("work in progress")
    #         sys.exit()  # Exits the program

    #     else: 
    #         logger.error("Provide 'single' or 'multiple' blocks.")
    #         sys.exit()  # Exits the program

# def read_vis(self, _npz, select_fraction=1):
#     #read packaged data
#     logger.info("read {}".format(_npz))
#     archive = np.load(self.pathout + _npz, allow_pickle=True)

#     #Select subset of visibilities
#     uu_lam, vv_lam, ww_lam, sigma, beam, data, coords, frequency = dformat.format_data(select_fraction, archive)
    
#     # store everything as 1D
#     vis_data = VisData(uu_lam, vv_lam, ww_lam, sigma, data, beam, coords, frequency)
    
#     return vis_data

# def package_ms(self, filename, select_fraction=1, uvmin=0, uvmax=7000, nchan=1, start=0, width=1, inc=1):
#     #get filenames of all ms from mspath
#     msl = sorted(glob.glob(self.path_ms+"*.ms"))
#     logger.info("number of ms files = {}".format(len(msl)))        

#     # get data
#     frequency, uu, vv, ww, weight, sigma, data, flag, beam, ra_hms, dec_dms = dms2npz.get_baselines(msl, select_fraction=select_fraction, sigma_rescale=1.0, incl_model_data=False, datacolumn="data", nchan=nchan, start=start, width=width, inc=inc, uvmin=np.float32(uvmin), uvmax=np.float32(uvmax))
    
#     logger.info(data.shape, uu.shape, weight.shape, sigma.shape)
    
#     logger.info("write " + filename + " on disk")
#     np.savez(
#         self.pathout + filename,
#         frequency=frequency, # [GHz]
#         uu=uu, # [lambda]
#         vv=vv, # [lambda]
#         ww=ww, # [lambda]
#         weight=weight, # [1/Jy^2]
#         sigma=sigma, # assumed to be [Jy]
#         data=data, # [Jy]       
#         flag=flag, # [Bool]
#         beam=beam, #beam position in the list
#         ra_hms=ra_hms, #phase center Right ascension [h:m:s]
#         dec_dms=dec_dms, #phase center Declination [h:m:s]
#     )
    
#     return None


