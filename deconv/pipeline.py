# -*- coding: utf-8 -*-
import os
import glob
import sys
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import subprocess
import tarfile
import concurrent.futures
import multiprocessing

from deconv.io import DataProcessor#, DataVisualizer
from deconv.imager import Imager
from deconv import logger

# Class Pipeline
class Pipeline:
    def __init__(self, path_ms, path_beams, path_sd, pathout, target_header, units,
                 sd, beam_sd, max_its, lambda_sd, lambda_r, positivity, device,
                 start=0, end=4, step=1, data_processor_workers=12, imager_workers=8,
                 queue_maxsize=4, uvmin=0, uvmax=7000, extension=".ms", blocks="single", fixms=False):
        # Save paths and parameters.
        self.path_ms = path_ms
        self.path_beams = path_beams
        self.path_sd = path_sd
        self.pathout = pathout

        # Instantiate I/O components.
        self.data_processor = DataProcessor(path_ms, path_beams, path_sd, pathout)
        # self.data_visualizer = DataVisualizer(path_ms, path_beams, path_sd, pathout)

        # Get imaging auxiliary data.
        # untardir and fixms
        # self.data_processor.untardir(max_workers=6, clear=False) #warning clean=True will clear the .tar files
        # fixms ASKAP data only using Alec's package
        if fixms == True: self.data_processor.fixms()
        # Continuum subtractin using casatools
        # XXX fixme
        # Compute effective primary beam - not used in imaging
        # XXX fixme put effpb.py in core.py

        # pre-compute pb and interpolation grids
        # self.data_processor.compute_pb_and_grid(target_header, fitsname_pb="reproj_pb_Dave.fits",
        #                                         fitsname_grid="grid_interp_Dave.fits") 
        
        
        # Get imaging auxiliary data.
        self.pb, self.grid = self.data_processor.read_pb_and_grid("reproj_pb_Dave.fits",
                                                                  "grid_interp_Dave.fits")
        # Get sd data from path_sd - fixme
        # sd, beam_sd = data_processor.read_sd()

        self.target_header = target_header
        self.sd = sd
        self.beam_sd = beam_sd
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

        # Cube parameters.
        self.start = start
        self.end = end
        self.step = step
        self.idlist = np.arange(start, end, step)
        self.shape = (target_header["NAXIS2"], target_header["NAXIS1"])
        self.num_channels = len(self.idlist)
        self.num_elements_per_channel = int(np.prod(self.shape))
        self.cube_shape = (self.num_channels, *self.shape)

        # Queue and worker settings.
        self.data_processor_workers = data_processor_workers
        self.imager_workers = imager_workers
        self.queue_maxsize = queue_maxsize

        # Shared memory for the output cube.
        self.cube_total_size = self.num_channels * self.num_elements_per_channel
        self.shared_cube = multiprocessing.Array('d', self.cube_total_size, lock=True)

    def preload_visibilities(self, vis_queue):
        """Loads visibility data while ensuring strict queue space waiting."""
        logger.info("Starting visibility preloading...")

        for i in self.idlist:
            logger.info(f"Waiting for space in queue to load channel {i}...")

            # Strictly wait until there is space in the queue
            while vis_queue.full():
                time.sleep(0.1) # Small wait to prevent busy looping

            logger.info(f"Loading visibilities for channel {i}...")

            try:
                vis_data = self.data_processor.read_vis_from_scratch(
                    uvmin=self.uvmin, uvmax=self.uvmax, target_frequency=None,
                    target_channel=i, extension=self.extension, blocks=self.blocks,
                    max_workers=self.data_processor_workers
                )
                vis_queue.put((i, vis_data)) # Add visibility data to queue
            except Exception as e:
                logger.error(f"Error loading channel {i}: {e}")
                vis_queue.put((i, None)) # Send error signal

        logger.info("Finished preloading all visibilities.")
                
        # Send one sentinel value for each worker, so they know when to stop
        for _ in range(self.imager_workers):
            vis_queue.put(None)

    def process_visibilities(self, vis_queue):
        while True:
            item = vis_queue.get()
            if item is None:
                logger.info("Received stop signal. Exiting worker.")
                break # Exit loop when sentinel value is received
            i, vis_data = item
            if vis_data is None:
                logger.error(f"Skipping channel {i} due to failed visibility loading.")
                continue # Skip failed visibility data

            logger.info(f"Processing visibilities for channel {i}...")

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
                device=self.device
            )
            try:
                result = image_processor.process(units=self.units)
                logger.info(f"Successfully processed channel {i}. Storing result in shared_cube.")
                
                # Compute the correct offset into the flat array:
                channel_index = i - self.start
                offset = channel_index * self.num_elements_per_channel
                flat_result = result.ravel()

                # Write the flat_result into the shared_cube (with locking)
                with self.shared_cube.get_lock():
                    for j in range(self.num_elements_per_channel):
                        self.shared_cube[offset + j] = flat_result[j]
            except Exception as e:
                logger.error(f"Error processing channel {i}: {e}")

    def run(self):
        # Create a queue with a strict max size
        vis_queue = multiprocessing.Queue(maxsize=self.queue_maxsize)

        # Start the preloading process
        preload_process = multiprocessing.Process(target=self.preload_visibilities, args=(vis_queue,))
        preload_process.start()

        # Start parallel processing workers
        processing_workers = []
        for _ in range(self.imager_workers):
            worker = multiprocessing.Process(target=self.process_visibilities, args=(vis_queue,))
            worker.start()
            processing_workers.append(worker)

        # Ensure preloading completes
        preload_process.join()

        # Ensure all processing workers complete
        for worker in processing_workers:
            worker.join()

        logger.info("Processing completed successfully.")

    def write(self, output_filename):
        final_cube = np.frombuffer(self.shared_cube.get_obj()).reshape(self.cube_shape)
        hdu0 = fits.PrimaryHDU(final_cube, header=self.target_header)
        hdu0.writeto(self.pathout+output_filename, overwrite=True)
        logger.info("Writing FITS file on disk.")

