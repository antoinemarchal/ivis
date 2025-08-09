# -*- coding: utf-8 -*-
"""
Data I/O and preprocessing utilities for visibility and beam data.

This module defines the VisData dataclass and the DataProcessor class for
handling Measurement Sets, extracting tar archives, and preparing input
for imaging in the IViS pipeline.
"""

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
from reproject import reproject_interp
from dataclasses import dataclass
import subprocess
import tarfile
import concurrent.futures
from tqdm import tqdm as tqdm

from ivis.logger import logger
from ivis.utils import dutils, dcasacore
from ivis.utils.readers import read_ms_block_I


@dataclass #modified from MPol
class VisData:
    """Container for visibility data used in the imaging pipeline."""
    uu: np.ndarray
    vv: np.ndarray
    ww: np.ndarray
    sigma: np.ndarray
    data: np.ndarray
    beam: np.ndarray
    coords: np.ndarray
    frequency: np.ndarray

# DataProcessor class    
class DataProcessor:
    """
    Handles loading, extraction, and preprocessing of measurement set data,
    as well as beam grid reprojection.

    Parameters
    ----------
    path_ms : str
        Path to measurement sets (.ms or tar archives).
    path_beams : str
        Path to beam FITS files.
    path_sd : str
        Path to single-dish FITS file (optional, can be empty).
    pathout : str
        Output directory for processed primary beams and grids.
    """
        
    def __init__(self, path_ms, path_beams, path_sd, pathout):
        super(DataProcessor, self).__init__()
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout
        logger.info("[Initialize DataProcessor ]")
        

    def fixms(self): #fixme ran in parallel
        """
        Applies `fix_ms_dir` to all .ms files in the input path.
        """
        #get msl from path
        msl = sorted(glob.glob(self.path_ms+"*.ms"))
        # Apply fix_ms_dir to each MS file
        for ms in msl:
            logger.info(f"Processing {ms}...")
            subprocess.run(["fix_ms_dir", ms])  # Run the command for each MS file    

        logger.info("All MS files processed.")


    def extract_tar(self, tar_file, clear=False):
        """
        Extracts a .tar file to its directory and optionally deletes it.

        Parameters
        ----------
        tar_file : str
            Path to .tar archive.
        clear : bool, optional
            Whether to delete the archive after extraction.

        Returns
        -------
        str
            Path to the processed .tar file.
        """
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
        """
        Recursively extracts all .tar files in `path_ms` using multithreading.

        Parameters
        ----------
        max_workers : int
            Number of threads for parallel extraction.
        clear : bool
            Whether to delete the archives after extraction.
        """
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


    def read_vis_visidata(
            self,
            uvmin: float = 0.0,
            uvmax: float = float("inf"),
            *,
            chan_sel=None,                   # None | slice | list[int] | np.ndarray[int]
            target_channel: int | None = None,
            target_frequency: float | None = None,  # Hz
            keep_autocorr: bool = False,
            prefer_weight_spectrum: bool = True,
            n_workers: int = 0,
    ):
        """
        Load visibilities as a channel-major I-only VisIData using read_ms_block_I.
        
        Channel selection priority:
        1) If `chan_sel` is given, use it as-is.
        2) Else if `target_channel` is given, use that single channel (slice(tc, tc+1)).
        3) Else if `target_frequency` is given (Hz), pick the nearest channel.
        4) Else load all channels.
        
        Parameters
        ----------
        uvmin, uvmax : float
            Baseline length limits in wavelengths (approx at median selected λ).
        chan_sel : slice | list[int] | np.ndarray[int] | None
            Channel subset to load. If None, use target_* logic above.
        target_channel : int | None
            Single channel index to load (ignored if chan_sel is provided).
        target_frequency : float | None
            Frequency in Hz; nearest channel will be selected (ignored if chan_sel is provided).
        keep_autocorr : bool
            If False, drop ANTENNA1==ANTENNA2 baselines.
        prefer_weight_spectrum : bool
            If True and WEIGHT_SPECTRUM is present, use it for per-channel σ.

        Returns
        -------
        VisIData
            Container with shapes: data_I/sigma_I/flag_I -> (nchan, nbeam, nvis_max)
        """
        ms_dir = self.path_ms
        
        # Resolve channel selection
        if chan_sel is not None:
            pass  # use as-is
        elif target_channel is not None:
            tc = int(target_channel)
            chan_sel = slice(tc, tc + 1)
        elif target_frequency is not None:
            # Nearest-channel lookup from first MS in directory
            import glob, os
            from casacore.tables import table  # local import to avoid hard dep at module import
            
            ms_list = sorted(glob.glob(os.path.join(ms_dir, "*.ms")))
            if not ms_list:
                raise FileNotFoundError(f"No .ms found in: {ms_dir}")
            first_ms = ms_list[0]
            # read CHAN_FREQ (Hz)
            with table(f"{first_ms}/SPECTRAL_WINDOW", readonly=True) as t:
                freqs = np.atleast_1d(np.squeeze(t.getcol("CHAN_FREQ"))).astype(float)
            idx = int(np.argmin(np.abs(freqs - float(target_frequency))))
            chan_sel = slice(idx, idx + 1)
        else:
            chan_sel = None  # all channels

        # Delegate to the optimized reader (reads only selected channels via getcolslice)
        visI = read_ms_block_I(
            ms_dir,
            uvmin=float(uvmin),
            uvmax=float(uvmax),
            chan_sel=chan_sel,
            keep_autocorr=keep_autocorr,
            prefer_weight_spectrum=prefer_weight_spectrum,
            n_workers=n_workers,
        )
        return visI


    def read_vis_from_scratch(
            self,
            uvmin=0,
            uvmax=7000,
            chunks=1.e7,
            target_frequency=None,
            target_channel=0,
            extension=".ms",
            blocks="single",
            max_workers=1,
            max_blocks=None  # <-- new parameter
    ):
        """
        Reads visibilities from Measurement Sets in 'single' or 'multiple' block mode.

        Parameters
        ----------
        uvmin, uvmax : float
            Minimum and maximum baseline lengths in lambda.
        chunks : float
            Unused placeholder for chunk size.
        target_frequency : float or None
            Target frequency for selecting channels.
        target_channel : int
            Index of the desired frequency channel.
        extension : str
            File extension to filter MS files.
        blocks : {'single', 'multiple'}
            Whether to process a flat list or nested directory structure.
        max_workers : int
            Number of workers for parallel loading.
        max_blocks : int or None
            Optional limit on number of blocks when using 'multiple' mode.

        Returns
        -------
        VisData
            Concatenated visibility dataset.
        """
        if blocks == 'single':
            logger.info("Processing single scheduling block.")
            
            msl = sorted(glob.glob(os.path.join(self.path_ms, f"*{extension}")))
            logger.info(f"Number of MS files = {len(msl)}")
            
            if max_workers > 1:
                logger.info("Reading MS files in parallel.")
                vis_data = dcasacore.readmsl(msl, uvmin, uvmax, target_frequency, target_channel, max_workers)
            else:
                logger.info("Reading MS files with single thread.")
                vis_data = dcasacore.readmsl_no_parallel(msl, uvmin, uvmax, target_frequency, target_channel)
            return vis_data
        
        elif blocks == 'multiple':
            logger.info("Processing multiple scheduling blocks.")
            
            subdirs = [d for d in sorted(os.listdir(self.path_ms)) if os.path.isdir(os.path.join(self.path_ms, d))]
            if not subdirs:
                logger.error("No subdirectories found in the path.")
                sys.exit()
                
            if max_blocks is not None:
                subdirs = subdirs[:max_blocks]
                logger.info(f"Limiting to first {max_blocks} blocks.")
                
            all_vis_data = []
            
            for subdir in subdirs:
                subdir_path = os.path.join(self.path_ms, subdir)
                msl = sorted(glob.glob(os.path.join(subdir_path, f"*{extension}")))
                
                if not msl:
                    logger.warning(f"No MS files found in {subdir_path}, skipping...")
                    continue

                if max_workers > 1:
                    logger.info(f"Reading MS files in parallel from {subdir}.")
                    vis_data = dcasacore.readmsl(msl, uvmin, uvmax, target_frequency, target_channel, max_workers)
                else:
                    logger.info(f"Reading MS files in single-thread from {subdir}.")
                    vis_data = dcasacore.readmsl_no_parallel(msl, uvmin, uvmax, target_frequency, target_channel)

                all_vis_data.append(vis_data)

            if not all_vis_data:
                logger.error("No valid data found across subdirectories.")
                sys.exit()
                
            logger.info(f"Concatenating visibility data from {len(all_vis_data)} blocks.")
            return self.concatenate_vis_data(all_vis_data)

        else:
            logger.error("Provide 'single' or 'multiple' for blocks.")
            sys.exit()


    @staticmethod
    def concatenate_vis_data(vis_data_list):
        """
        Concatenates a list of VisData objects into one sorted by beam index.

        Parameters
        ----------
        vis_data_list : list of VisData
            List of visibility datasets to merge.

        Returns
        -------
        VisData
            Combined and beam-sorted visibility data.
        """    
        # Concatenate all arrays (except for coords and frequency)
        uu = np.concatenate([v.uu for v in vis_data_list])
        vv = np.concatenate([v.vv for v in vis_data_list])
        ww = np.concatenate([v.ww for v in vis_data_list])
        sigma = np.concatenate([v.sigma for v in vis_data_list])
        data = np.concatenate([v.data for v in vis_data_list])
        beam = np.concatenate([v.beam for v in vis_data_list])
        frequency = np.concatenate([v.frequency for v in vis_data_list])
        
        # Get the indices that would sort the beam array
        sort_indices = np.argsort(beam)
        
        # Apply the sorting indices to all arrays to keep them aligned
        uu = uu[sort_indices]
        vv = vv[sort_indices]
        ww = ww[sort_indices]
        sigma = sigma[sort_indices]
        data = data[sort_indices]
        beam = beam[sort_indices]  # Reorder the beam array using sort_indices
        frequency = frequency[sort_indices]  # Reorder the beam array using sort_indices
        
        # Keep the first value of coords and frequency (without concatenation)
        coords = vis_data_list[0].coords
        # frequency = vis_data_list[0].frequency
        
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
        """
        Reprojects and interpolates primary beams onto a common WCS grid.

        Parameters
        ----------
        hdr : dict
            FITS header of target WCS.
        fitsname_pb : str
            Filename to store the reprojected primary beams.
        fitsname_grid : str
            Filename to store the interpolation grids.
        """
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
            input_header = w_pb.to_header()

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
        """
        Loads precomputed primary beams and interpolation grids from disk.

        Parameters
        ----------
        fitsname_pb : str
            Filename of the primary beam FITS.
        fitsname_grid : str
            Filename of the grid FITS.

        Returns
        -------
        tuple of np.ndarray
            (primary beam array, interpolation grid array)
        """
        #read pre-computed pb
        hdu_grid = fits.open(self.pathout + fitsname_grid)
        #read pre-computed grid                
        hdu_pb = fits.open(self.pathout + fitsname_pb)

        return hdu_pb[0].data, hdu_grid[0].data


    def read_sd(self):
        """
        Stub for reading single-dish data.

        Returns
        -------
        tuple
            (sd = 0, beam_sd = 0)
        """
        sd = 0; beam_sd = 0
        
        return sd, beam_sd
