# -*- coding: utf-8 -*-
import os
import glob
import sys
import time
import psutil
import queue
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from radio_beam import Beam
import subprocess
import tarfile
import concurrent.futures
import multiprocessing

from ivis.logger import logger
from ivis.io import DataProcessor#, DataVisualizer
from ivis.imager import Imager

# Class Pipeline
class Pipeline:
    def __init__(self, path_ms, path_beams, path_sd, pathout, filename, target_header, sd, beam_sd,
                 units="Jy/arcsec^2", max_its=20, lambda_sd=0, lambda_r=1, positivity=False,
                 device="cpu", start=0, end=4, step=1, data_processor_workers=12,
                 imager_workers=8, beam_workers=4, queue_maxsize=4, uvmin=0, uvmax=7000,
                 extension=".ms", blocks="single", max_blocks=None, fixms=False, precompute=False,
                 write_mode="final"):
        # Save paths and parameters.
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout
        self.filename = filename

        # Instantiate I/O components.
        self.data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)
        # self.data_visualizer = DataVisualizer(path_ms, path_beams, path_sd, pathout)

        # Get imaging auxiliary data.
        # untardir and fixms
        # self.data_processor.untardir(max_workers=6, clear=False) #warning clean=True will clear the .tar files
        # fixms ASKAP data only using Alec's package
        if fixms == True: self.data_processor.fixms()
        # Continuum subtraction using casatools
        # XXX fixme
        # Compute effective primary beam - not used in imaging
        # XXX fixme put effpb.py in core.py

        # pre-compute pb and interpolation grids
        if precompute == True:
            logger.info("Precompute primary beam and interpolation grid.")
            self.data_processor.compute_pb_and_grid(target_header,
                                                    fitsname_pb="reproj_pb_full.fits",
                                                    fitsname_grid="grid_interp_full.fits")
                    
        
        # Get imaging auxiliary data.
        logger.info("Read primary beam and interpolation grid.")
        self.pb, self.grid = self.data_processor.read_pb_and_grid("reproj_pb_full.fits",
                                                                  "grid_interp_full.fits")
        # Other user parameters
        self.target_header = target_header
        self.max_its = max_its
        self.lambda_sd = lambda_sd
        self.lambda_r = lambda_r
        self.positivity = positivity
        self.device = device
        self.units = units
        self.uvmin = uvmin
        self.uvmax = uvmax
        self.extension = extension
        self.blocks = blocks
        self.max_blocks = max_blocks
        self.write_mode = write_mode  # "live" or "final"
        
        # Cube parameters.
        self.start = start
        self.end = end
        self.step = step
        self.idlist = np.arange(start, end, step)
        self.shape = (target_header["NAXIS2"], target_header["NAXIS1"])
        self.num_channels = len(self.idlist)
        self.num_elements_per_channel = int(np.prod(self.shape))
        self.cube_shape = (self.num_channels, *self.shape)
        
        # Get sd data from path_sd - fixme
        # sd, beam_sd = data_processor.read_sd()
        # Single-dish data and Beam
        self.sd = sd#np.zeros(self.shape); #Dummy array for single dish
        self.beam_sd = beam_sd#Beam((16*u.arcmin).to(u.deg),(16*u.arcmin).to(u.deg), 1.e-12*u.deg) #must be all in deg

        # Queue and worker settings.
        self.data_processor_workers = data_processor_workers
        self.imager_workers = imager_workers
        self.beam_workers = beam_workers
        self.queue_maxsize = queue_maxsize

        # shared cube of write mode is final
        if self.write_mode == "final":
            self.cube_total_size = self.num_channels * self.num_elements_per_channel
            self.shared_cube = multiprocessing.Array('d', self.cube_total_size, lock=True)
        # else live mode 
        else:
            self.shared_cube = None

    def preload_visibilities(self, vis_queue):
        """Preloads visibility data into a multiprocessing queue with backpressure handling."""
        logger.info("Starting visibility preloading...")
        
        for i in self.idlist:
            logger.info(f"Waiting for space in vis_queue to load channel {i}...")
            
            # Wait until there's room in the queue
            while vis_queue.full():
                time.sleep(0.1)

            logger.info(f"Loading visibilities for channel {i}...")
            try:
                vis_data = self.data_processor.read_vis_from_scratch(
                    uvmin=self.uvmin,
                    uvmax=self.uvmax,
                    target_frequency=None,
                    target_channel=i,
                    extension=self.extension,
                    blocks=self.blocks,
                    max_blocks=self.max_blocks,
                    max_workers=self.data_processor_workers
                )
                logger.info(f"Successfully loaded and queuing channel {i}")
                vis_queue.put((i, vis_data))
            except Exception as e:
                logger.error(f"Error loading channel {i}: {e}", exc_info=True)
                vis_queue.put((i, None))  # Insert a marker that this channel failed
                
        logger.info("All visibilities enqueued. Sending stop signals to workers...")

        # Send sentinel values to signal worker termination
        for _ in range(self.imager_workers):
            while vis_queue.full():
                time.sleep(0.1)
            vis_queue.put(None)

        logger.info("Finished visibility preloading.")


    # def preload_visibilities(self, vis_queue):
    #     """Loads visibility data while ensuring strict queue space waiting."""
    #     logger.info("Starting visibility preloading...")

    #     for i in self.idlist:
    #         logger.info(f"Waiting for space in queue to load channel {i}...")

    #         # Strictly wait until there is space in the queue
    #         while vis_queue.full():
    #             time.sleep(0.1) # Small wait to prevent busy looping

    #         logger.info(f"Loading visibilities for channel {i}...")

    #         try:
    #             vis_data = self.data_processor.read_vis_from_scratch(
    #                 uvmin=self.uvmin, uvmax=self.uvmax, target_frequency=None,
    #                 target_channel=i, extension=self.extension, blocks=self.blocks,
    #                 max_blocks=self.max_blocks, max_workers=self.data_processor_workers
    #             )
    #             logger.info(f"Queueing channel {i} to vis_queue.")
    #             vis_queue.put((i, vis_data)) # Add visibility data to queue
    #         except Exception as e:
    #             logger.error(f"Error loading channel {i}: {e}")
    #             vis_queue.put((i, None)) # Send error signal

    #     logger.info("Finished preloading all visibilities.")
                
    #     # # Send one sentinel value for each worker, so they know when to stop
    #     # for _ in range(self.imager_workers):
    #     #     vis_queue.put(None)
    #     # After enqueuing all work items
    #     for _ in range(self.imager_workers):
    #         while vis_queue.full():
    #             time.sleep(0.1)  # Wait for a worker to consume
    #         vis_queue.put(None)


    def run(self):
        # Init FITS file to write output
        # self.init_output_fits_noalloc()
        self.init_output_fits()
        
        # Create a queue with a strict max size
        vis_queue = multiprocessing.Queue(maxsize=self.queue_maxsize)
        
        # Live writing: use a writer queue and process
        if self.write_mode == "live":
            # write_queue size set by the number of imager workers
            write_queue = multiprocessing.Queue(maxsize=2*self.imager_workers) #times 2 to give more space
            writer_process = multiprocessing.Process(target=self.write_worker, args=(write_queue,))
            writer_process.start()
        else:
            write_queue = None  # Not used

        # Start the preloading process
        preload_process = multiprocessing.Process(target=self.preload_visibilities, args=(vis_queue,))
        preload_process.start()
        
        # Wait for a few channels to preload before launching workers
        preload_threshold = min(2, self.queue_maxsize)
        logger.info(f"Waiting for at least {preload_threshold} channels to preload...")
        
        while vis_queue.qsize() < preload_threshold:
            time.sleep(0.1)
            
        logger.info("Preload threshold reached. Starting processing workers.")
        
        # Start parallel processing workers
        processing_workers = []
        for _ in range(self.imager_workers):
            worker = multiprocessing.Process(
                target=self.process_visibilities, args=(vis_queue, write_queue)
            )
            worker.start()
            processing_workers.append(worker)
        
        # Ensure preloading completes
        preload_process.join()

        # Ensure all processing workers complete
        for worker in processing_workers:
            worker.join()
            
        if self.write_mode == "live":
            write_queue.put(None)
            writer_process.join()
            
        logger.info("Processing completed successfully.")
        
        if self.write_mode == "final":
            self.write_final_cube()

        
    def write_worker(self, write_queue):
        output_path = os.path.join(self.pathout, self.filename)
        with fits.open(output_path, mode='update', memmap=True) as hdul:
            cube = hdul[0].data
            while True:
                item = write_queue.get()
                if item is None:
                    logger.info("Writer received stop signal.")
                    break

                i, result = item
                logger.info(f"Writer received result for channel {i}")
                channel_index = i - self.start
                cube[channel_index, :, :] = result.astype(np.float32)
                hdul.flush()
                logger.info(f"Channel {i} written to disk.")


    def process_visibilities(self, vis_queue, write_queue):
        # Set a unique seed per process (only once, at worker startup)
        np.random.seed(os.getpid())
        while True:
            try:
                item = vis_queue.get()
                if item is None:
                    logger.info(f"[PID {os.getpid()}] Received stop signal.")
                    break
                
                i, vis_data = item
                if vis_data is None:
                    logger.error(f"[PID {os.getpid()}] Skipping channel {i} due to failed visibility loading.")
                    continue

                logger.info(f"[PID {os.getpid()}] Processing visibilities for channel {i}...")
                
                init_params = np.zeros(self.shape).ravel()
                image_processor = Imager(
                    vis_data=vis_data,
                    pb=self.pb,
                    grid=self.grid,
                    sd=self.sd,
                    beam_sd=self.beam_sd,
                    hdr=self.target_header,
                    init_params=init_params,
                    max_its=self.max_its,
                    lambda_sd=self.lambda_sd,
                    lambda_r=self.lambda_r,
                    positivity=self.positivity,
                    device=self.device,
                    beam_workers=self.beam_workers
                )
                
                try:
                    result = image_processor.process(units=self.units)
                    logger.info(f"[PID {os.getpid()}] Channel {i} processed successfully.")
                    
                    if self.write_mode == "live":
                        write_queue.put((i, result))
                        logger.info(f"[PID {os.getpid()}] Channel {i} queued to writer.")
                    else:
                        channel_index = i - self.start
                        offset = channel_index * self.num_elements_per_channel
                        flat_result = result.ravel()

                        with self.shared_cube.get_lock():
                            for j in range(self.num_elements_per_channel):
                                self.shared_cube[offset + j] = flat_result[j]

                except Exception as e:
                    logger.error(f"[PID {os.getpid()}] Error processing channel {i}: {e}", exc_info=True)
                    
            except Exception as outer_e:
                logger.error(f"[PID {os.getpid()}] Worker crashed outside main loop: {outer_e}", exc_info=True)
                break


    def write_final_cube(self):
        logger.info("Writing final FITS cube to disk.")
        full_path = os.path.join(self.pathout, self.filename)
        final_cube = np.frombuffer(self.shared_cube.get_obj()).reshape(self.cube_shape)
        logger.info(f"Final cube shape: {final_cube.shape}")
        for i in range(self.num_channels):
            stats = final_cube[i]
            logger.info(f"Channel {i}: min={stats.min()}, max={stats.max()}, mean={stats.mean()}")
        hdu0 = fits.PrimaryHDU(final_cube.astype(np.float32), header=self.target_header)
        hdu0.writeto(full_path, overwrite=True)
                

    def init_output_fits(self):
        full_path = os.path.join(self.pathout, self.filename)
        
        # Estimate required memory in bytes
        required_bytes = np.prod(self.cube_shape) * 4  # float32 = 4 bytes
        available_bytes = psutil.virtual_memory().available
        
        if required_bytes > available_bytes:
            logger.error(f"Not enough RAM to allocate output cube "
                         f"({required_bytes/1e9:.2f} GB required, "
                         f"{available_bytes/1e9:.2f} GB available). Aborting.")
            raise MemoryError("Insufficient RAM for cube allocation.")
        
        # Allocate and write FITS
        data = np.zeros(self.cube_shape, dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data, header=self.target_header)
        hdu.writeto(full_path, overwrite=True)
        logger.info(f"Initialized output FITS cube at {full_path}")





###########################################################
        # def init_output_fits_noalloc(self):
    #     """
    #     Initializes an empty FITS file on disk with the correct header and data section,
    #     without allocating the full data cube in memory. Ensures 2880-byte compliance.
    #     """
    #     full_path = os.path.join(self.pathout, self.filename)
        
    #     # Create a valid FITS header
    #     header = self.target_header.copy()
    #     header['BITPIX'] = -32  # IEEE float32
    #     header['NAXIS'] = 3
    #     header['NAXIS1'] = self.shape[1]
    #     header['NAXIS2'] = self.shape[0]
    #     header['NAXIS3'] = self.num_channels
        
    #     # Build header block
    #     hdu = fits.PrimaryHDU(data=None, header=header)
    #     hdul = fits.HDUList([hdu])
    #     hdul.writeto(full_path, overwrite=True)
        
    #     # Re-open to compute actual header size on disk
    #     with open(full_path, 'r+b') as f:
    #         f.seek(0)
    #         raw_header = f.read()
    #         header_end = raw_header.find(b'END') + 80  # 'END' plus one full card
    #         header_padded = 2880 * ((header_end + 2879) // 2880)
            
    #         # Compute data size in bytes
    #         num_pixels = self.num_channels * self.shape[0] * self.shape[1]
    #         data_size = num_pixels * 4  # float32 = 4 bytes
            
    #         # FITS standard: total file size must be multiple of 2880 bytes
    #         total_bytes = header_padded + data_size
    #         padded_total = 2880 * ((total_bytes + 2879) // 2880)
    #         padding_needed = padded_total - total_bytes
            
    #         # Seek to end of header and write data + final padding
    #         f.seek(header_padded)
    #         f.write(b'\x00' * data_size)
    #         if padding_needed > 0:
    #             f.write(b'\x00' * padding_needed)
                
    #     logger.info(f"Initialized FITS file with preallocated data section: {full_path}")
            
##############################################################
    # This code was used to run and write the data on disk separately
    # def process_visibilities_old(self, vis_queue):
    #     while True:
    #         item = vis_queue.get()
    #         if item is None:
    #             logger.info("Received stop signal. Exiting worker.")
    #             break # Exit loop when sentinel value is received
    #         i, vis_data = item
    #         if vis_data is None:
    #             logger.error(f"Skipping channel {i} due to failed visibility loading.")
    #             continue # Skip failed visibility data

    #         logger.info(f"Processing visibilities for channel {i}...")

    #         init_params = np.zeros(self.shape).ravel()
    #         image_processor = Imager(
    #             vis_data=vis_data,
    #             pb=self.pb,
    #             grid=self.grid,
    #             sd=self.sd,
    #             beam_sd=self.beam_sd,
    #             hdr=self.target_header,
    #             init_params=init_params,
    #             max_its=self.max_its,
    #             lambda_sd=self.lambda_sd,
    #             lambda_r=self.lambda_r,
    #             positivity=self.positivity,
    #             device=self.device,
    #             beam_workers=self.beam_workers
    #         )
    #         try:
    #             result = image_processor.process(units=self.units)
    #             logger.info(f"Successfully processed channel {i}. Storing result in shared_cube.")
                
    #             # Compute the correct offset into the flat array:
    #             channel_index = i - self.start
    #             offset = channel_index * self.num_elements_per_channel
    #             flat_result = result.ravel()

    #             # Write the flat_result into the shared_cube (with locking)
    #             with self.shared_cube.get_lock():
    #                 for j in range(self.num_elements_per_channel):
    #                     self.shared_cube[offset + j] = flat_result[j]
    #         except Exception as e:
    #             logger.error(f"Error processing channel {i}: {e}")

        # def run(self, output_filename):
    #     # Create a queue with a strict max size
    #     vis_queue = multiprocessing.Queue(maxsize=self.queue_maxsize)

    #     # Start the preloading process
    #     preload_process = multiprocessing.Process(target=self.preload_visibilities, args=(vis_queue,))
    #     preload_process.start()

    #     # Start parallel processing workers
    #     processing_workers = []
    #     for _ in range(self.imager_workers):
    #         worker = multiprocessing.Process(target=self.process_visibilities, args=(vis_queue,))
    #         worker.start()
    #         processing_workers.append(worker)

    #     # Ensure preloading completes
    #     preload_process.join()

    #     # Ensure all processing workers complete
    #     for worker in processing_workers:
    #         worker.join()

    #     logger.info("Processing completed successfully.")

    # def write(self, output_filename):
    #     final_cube = np.frombuffer(self.shared_cube.get_obj()).reshape(self.cube_shape)
    #     hdu0 = fits.PrimaryHDU(final_cube, header=self.target_header)
    #     hdu0.writeto(self.pathout+output_filename, overwrite=True)
    #     logger.info("Writing FITS file on disk.")
###########################################
