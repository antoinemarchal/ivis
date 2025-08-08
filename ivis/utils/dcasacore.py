# -*- coding: utf-8 -*-
"""
Measurement Set I/O Utilities
-----------------------------

This module provides tools for extracting and filtering visibility data from
radio interferometric Measurement Sets (MS), either in serial or parallel mode.
It supports both `casacore.tables` and optional `dask-ms` backends.

Functions
---------
- ``phasecenter_casacore(ms)``
- ``read_channel_casacore(...)``
- ``readmsl(...)`` / ``readmsl_no_parallel(...)``

Returned data are stored in a `VisData` dataclass containing visibility coordinates,
data, noise estimates, beam indices, and sky positions.

Author: Antoine Marchal
"""

import os
import glob
import time
import sys
import contextlib
import numpy as np
from tqdm import tqdm as tqdm
from astropy.constants import c
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from casacore.tables import table, taql
from pathlib import Path
# from daskms import xds_from_table

from ivis.logger import logger  # Import the logger
# from ivis.utils import vlsrk_from_ms

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
    velocity: np.ndarray

def phasecenter_casacore(ms):
    """Extracts the phase center (RA/Dec) from the FIELD table of a Measurement Set."""
    # Suppress casacore output when opening tables
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            # Open the FIELD subtable
            field_tab = table(f"{ms}/FIELD")

    # Read the PHASE_DIR column — shape is (1, 1, 2)
    phase_dir = field_tab.getcol("PHASE_DIR")
    field_tab.close()

    # Extract RA and Dec in radians
    ra_rad, dec_rad = phase_dir[0, 0, :]

    # Convert to HMS / DMS strings
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=":")
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=":")

    return ra_hms, dec_dms


def phasecenter_dask(ms): #FIXME
    """(Dask version) Extracts the phase center from FIELD table of MS using dask-ms."""
    # Load the PHASE_DIR column from the FIELD table
    field_xds = xds_from_table(f"{ms}::FIELD", columns=["PHASE_DIR"])
    
    # Extract phase center (shape: [Nfields, 1, 2])
    phase_center = field_xds[0].PHASE_DIR.compute()  # Convert Dask array to NumPy
    
    # Select the first field (adjust if needed)
    ra_rad, dec_rad = phase_center[0, 0]  # RA and Dec in radians
    
    # Convert to h:m:s and d:m:s
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=":")
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=":")

    # logger.info(f"RA: {ra_hms}, Dec: {dec_dms}")

    return ra_hms, dec_dms


def readms_dask(ms_path, uvmin, uvmax, chunks, target_frequency):
    """(Obsolete) Reads a single channel from an MS using dask-ms and filters baselines."""
    ms_path = Path(ms_path)

    #get phase center
    ra_hms, dec_dms = phasecenter_dask(ms_path)

    spw_table = table((ms_path / "SPECTRAL_WINDOW").as_posix())
    
    msds = xds_from_table(
        ms_path.as_posix(),
        index_cols=["SCAN_NUMBER", "TIME", "ANTENNA1", "ANTENNA2"],
        group_cols=["DATA_DESC_ID"],
        chunks={"row": chunks}
    )[0]
    
    ddids = xds_from_table(f"{ms_path.as_posix()}::DATA_DESCRIPTION")[0].compute()
    spws = xds_from_table(f"{ms_path.as_posix()}::SPECTRAL_WINDOW", group_cols="__row__")[0]
    pols = xds_from_table(f"{ms_path.as_posix()}::POLARIZATION", group_cols="__row__")[0]

    # Assuming `spws.CHAN_FREQ.compute().data` gives you the frequencies for the channels
    frequencies = np.squeeze(spws.CHAN_FREQ.compute().data)
    wavelengths = c.value / frequencies  # m
    rest_freq_u = 1.42040575177e9 * u.Hz  # More precise 21 cm HI line rest frequency
    velocities = ((rest_freq_u - frequencies * u.Hz) / rest_freq_u * c).to(u.km/u.s)

    # create coordinates
    # if len(np.array([frequencies])) == 1:
    #     msds_with_coords = msds.assign_coords(
    #         {
    #             "freq": ("chan", np.array([frequencies])),
    #             "pol": ("corr", np.squeeze(pols.CORR_TYPE.compute().data)),
    #     }
    #     )
    # else:
    msds_with_coords = msds.assign_coords(
        {
            "freq": ("chan", np.squeeze(frequencies * u.Hz)),
            "vel": ("chan", np.squeeze(velocities)),
            "pol": ("corr", np.squeeze(pols.CORR_TYPE.compute().data)),
        }
    )

    # Print traget frequency
    logger.info(f"target freq: {target_frequency.value}")

    # Check if the target frequency is out of the range (min/max)
    if target_frequency.value < frequencies.min() or target_frequency.value > frequencies.max():
        logger.error("target frequency out of range")
        sys.exit()  # Exits the program
    else:
        logger.info(f"target frequency {target_frequency.value} is within range.")
        # Continue with selecting the channel as usual
        channel_index = np.abs(frequencies - target_frequency.value).argmin()
        msds_with_coords = msds_with_coords.isel(chan=channel_index)
            
    logger.info(msds_with_coords)
        
    DATA = msds_with_coords.DATA
    SIGMA = msds_with_coords.SIGMA
    FLAG = msds_with_coords.FLAG
    
    STOKES_I = (DATA[..., 0] + DATA[..., -1]) * 0.5 #FIXME ASKAP conventions
    SIGMA_I = np.sqrt(SIGMA[..., 0]**2 + SIGMA[..., -1]**2) #correct propagation of uncertainties

    I = STOKES_I.compute().data#[:,0]
    SIGMA = SIGMA_I.compute().data
    UVW = msds_with_coords.UVW

    UVW_lambda = UVW / c.value * msds_with_coords.freq.compute().data.value
    baseline_lengths = np.sqrt((UVW_lambda ** 2).sum(axis=1))  # Euclidean norm
        
    UVW_lambda = UVW_lambda.compute().data

    # Keep cross correlation and filter flagged baselines
    ANTENNA1 = msds_with_coords.ANTENNA1
    ANTENNA2 = msds_with_coords.ANTENNA2
    xc = np.where((ANTENNA1 != ANTENNA2) & (FLAG[...,0] == False) & (FLAG[...,-1] == False) & (baseline_lengths >= uvmin) & (baseline_lengths <= uvmax))[0]
    UVW_lambda = UVW_lambda[xc]
    I = I[xc]
    SIGMA = SIGMA[xc]    

    logger.info(f"shape UVW_lambda: {UVW_lambda.shape}")
    logger.info(f"shape I: {I.shape}")
    logger.info(f"shape SIGMA: {SIGMA.shape}")
    
    logger.info(f"returned frequency: {msds_with_coords.freq.compute().data}")
    logger.info(f"returned velocity: {msds_with_coords.vel.compute().data}")

    return msds_with_coords.freq.compute().data.value, msds_with_coords.vel.compute().data, UVW_lambda, SIGMA, I, ra_hms, dec_dms
    # return msds_with_coords.freq.compute().data, UVW_lambda[:,:,0], SIGMA, I[:,0], ra_hms, dec_dms


def read_channel_casacore(ms_path, uvmin, uvmax, target_frequency, target_channel):
    """
    Read a single frequency channel from a Measurement Set using casacore.tables.
    
    Parameters
    ----------
    ms_path : str
        Path to the Measurement Set directory.
    uvmin : float
        Minimum baseline length in wavelengths.
    uvmax : float
        Maximum baseline length in wavelengths.
    target_frequency : Quantity
        Target frequency (in Hz).
    target_channel : int
        Index of the desired frequency channel.

    Returns
    -------
    freq : float
        Extracted channel frequency (Hz).
    vel : Quantity
        Computed velocity in the LSRK frame.
    uvw_lambda : ndarray
        (N, 3) array of baseline vectors in units of lambda.
    sigma_i : ndarray
        Estimated noise.
    stokes_i : ndarray
        Stokes I visibilities.
    ra_hms, dec_dms : str
        RA/Dec of phase center in HMS/DMS string format.
    """
    #get phase center
    # ra_hms, dec_dms = phasecenter_dask(ms_path)
    ra_hms, dec_dms = phasecenter_casacore(ms_path)

    # Suppress casacore output when opening tables
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            # Open the Measurement Set
            ms_table = table(ms_path, readonly=True)
            
            # Open the spectral window table to get frequency information
            spw_table = table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True)
            # frequencies = np.squeeze(spw_table.getcol("CHAN_FREQ"))
            frequencies = np.atleast_1d(np.squeeze(spw_table.getcol("CHAN_FREQ")))
            
    if target_frequency is not None and target_channel is not None:
        logger.error("Error: Only one of target_frequency or target_channel should be provided.")
        sys.exit(1)  # Exit with an error code
        
    if target_frequency is not None:
        # Do something with target_frequency
        # logger.info(f"Using target_frequency: {target_frequency}")
        
        # Find the closest frequency index
        channel_index = np.abs(frequencies - target_frequency.value).argmin()
        
    elif target_channel is not None:
        # Do something with target_channel
        # logger.info(f"Using target_channel: {target_channel}")        
        channel_index = target_channel        
    else:
        logger.error("Error: At least one of target_frequency or target_channel must be provided.")
        sys.exit(1)  # Exit with an error code

    # Compute the velocity for the selected channel
    rest_freq_u = 1.42040575177e9 * u.Hz  # Rest frequency of HI line (21 cm)
    #FIXME
    # velocity = vlsrk_from_ms.calculate_velocity_for_single_freq(ms_path,
    #                                                             frequencies[channel_index] * u.Hz,
    #                                                             rest_freq_u, field_id=0)    
    velocity = ((rest_freq_u - frequencies[channel_index] * u.Hz) / rest_freq_u * c).to(u.km / u.s)       

    logger.info(f"Selected channel: {channel_index} | Frequency: {frequencies[channel_index]} Hz | Velocity (LSRK): {velocity.value} km/s")
    
    # Read UVW coordinates
    uvw = ms_table.getcol("UVW")
    
    # Read full SIGMA array (per-channel)
    sigma = ms_table.getcol("SIGMA")  # Full SIGMA array, per-channel
    
    # Read only the selected channel (all rows, specific channel, all polarizations)
    data = ms_table.getcolslice("DATA",
                                blc=[channel_index, 0],  # All rows, target channel, first pol
                                trc=[channel_index, -1])  # All rows, target channel, last pol
    
    # Read the FLAG column for the selected channel
    flag = ms_table.getcolslice("FLAG",
                                blc=[channel_index, 0],
                                trc=[channel_index, -1])

    # # Open the POLARIZATION table to check polarization types
    # pol_table = table(f"{ms_path}/POLARIZATION", readonly=True)
    # corr_types = pol_table.getcol("CORR_TYPE")[0]  # Assume one row; typical for most MS files
    # pol_table.close()
    
    # logger.info(f"Polarization types (CORR_TYPE): {corr_types}")
    

    # Compute Stokes I (XX + YY) / 2
    stokes_i = (data[..., 0] + data[..., -1]) * 0.5  
    # if list(corr_types) == [5, 6, 7, 8]:  # XX, XY, YX, YY
    #     stokes_i = 0.5 * (data[..., 0] + data[..., 3])
    # elif list(corr_types) == [9, 10, 11, 12]:  # RR, RL, LR, LL
    #     stokes_i = 0.5 * (data[..., 0] + data[..., 3])
    # elif list(corr_types) == [1, 2, 3, 4]:  # I, Q, U, V
    #     stokes_i = data[..., 0]  # Already I
    # else:
    #     raise ValueError(f"Unsupported polarization combination: {corr_types}")    

    # Use full sigma values (per channel)
    # Temporarily suppress overflow warnings
    np.seterr(over='ignore')    
    # Your operation that causes the warning
    sigma_i = 0.5 * np.sqrt(sigma[..., 0]**2 + sigma[..., -1]**2)
    # Restore the default error handling (optional, but good practice)
    np.seterr(over='warn')  # 'warn' is the default setting
    
    # Compute baseline lengths
    wavelengths = c.value / frequencies[channel_index]
    uvw_lambda = uvw / wavelengths
    baseline_lengths = np.sqrt((uvw_lambda ** 2).sum(axis=1))  
    
    # Apply UV filtering and flagging (per channel)
    # Get antennas
    ant1 = ms_table.getcol("ANTENNA1")
    ant2 = ms_table.getcol("ANTENNA2")

    # Look if autocorr is flagged
    # # Find indices of auto-correlations
    # auto_indices = np.where(ant1 != ant2)[0]    
    # # Loop over and print FLAG status for each auto-correlation
    # for idx in auto_indices:
    #     print(f"Auto-correlation baseline {idx}: ANTENNA {ant1[idx]}-{ant2[idx]}, FLAG = {flag[idx]}")
        
    # Remove autocorrelations (i.e., antenna1 == antenna2)
    is_not_autocorr = (ant1 != ant2)
    unflagged = (flag[..., 0] == False) & (flag[..., -1] == False)  # Keep only unflagged
    valid_baselines = (baseline_lengths >= uvmin) & (baseline_lengths <= uvmax)
    mask = unflagged & valid_baselines[:, np.newaxis] & is_not_autocorr[:, np.newaxis] # Match shape
    
    # Apply mask
    uvw_lambda = uvw_lambda[mask[:, 0]]
    stokes_i = stokes_i[mask[:, 0]][:,0]
    sigma_i = sigma_i[mask[:, 0]]

    # Close the ms_table after processing
    ms_table.close()

    # logger.info(f"shape UVW_lambda: {uvw_lambda.shape}")
    # logger.info(f"shape I: {stokes_i.shape}")
    # logger.info(f"shape SIGMA: {sigma_i.shape}")

    logger.info(f"Extracted {uvw_lambda.shape[0]} valid baselines.")
    
    return frequencies[channel_index], velocity, uvw_lambda, sigma_i, stokes_i, ra_hms, dec_dms
    

def process_ms(ms, uvmin, uvmax, target_frequency, target_channel, k):
    """ Process a single MS file and return the extracted data. """
    iteration_start_time = time.time()
    
    try:
        logger.info(f"Processing file {k}: {ms}")
        
        freq, vel, UVW, SIGMA, DATA, ra, dec = read_channel_casacore(ms, uvmin, uvmax, target_frequency, target_channel)
        
        UU, VV, WW = UVW[:,0], UVW[:,1], UVW[:,2]
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')

        beam_index = np.full(len(UU), k)  # Beam index should start at 0
        
        iteration_time = time.time() - iteration_start_time
        logger.info(f"Finished processing {ms} in {iteration_time:.2f}s")
        
        return UU, VV, WW, SIGMA, DATA, beam_index, c, freq, vel
    
    except Exception as e:
        logger.error(f"Error processing {ms}: {e}")
        return None  # Returning None allows us to filter failed tasks later


def readmsl(msl, uvmin, uvmax, target_frequency, target_channel, n_workers):
    """ Parallelized version of readmsl using ProcessPoolExecutor. """

    uu, vv, ww, sigma, data, beam, centers = [], [], [], [], [], [], []
    total_files = len(msl)
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_ms = {executor.submit(process_ms, ms, uvmin, uvmax, target_frequency, target_channel, k): ms
                        for k, ms in enumerate(msl)}

        for future in as_completed(future_to_ms):
            result = future.result()
            if result is not None:
                UU, VV, WW, SIGMA, DATA, beam_index, c, freq, vel = result
                uu.append(UU); vv.append(VV); ww.append(WW)
                sigma.append(SIGMA); data.append(DATA)
                beam.append(beam_index); centers.append(c)
    
    # Concatenate results
    uu = np.concatenate(uu); vv = np.concatenate(vv); ww = np.concatenate(ww)
    sigma = np.concatenate(sigma); data = np.concatenate(data); beam = np.concatenate(beam)

    # Sort by beam
    sort = np.argsort(beam)
    uu_lam = uu[sort]; vv_lam = vv[sort]; ww_lam = ww[sort]
    sigma = sigma[sort]; beam = beam[sort]; data = data[sort]

    # Convert to appropriate types
    uu_lam = np.float32(uu_lam); vv_lam = np.float32(vv_lam); ww_lam = np.float32(ww_lam)
    sigma = np.float32(sigma); beam = np.int32(beam); data = np.complex64(data)

    # Take the complex conjugate
    data = np.conj(data)

    # Create VisData object
    vis_data = VisData(uu_lam, vv_lam, ww_lam, sigma, data, beam, centers, freq/1.e9, vel)

    logger.info(f"Total processing time: {(time.time() - start_time) / 60:.2f} minutes")

    return vis_data


def readmsl_no_parallel(msl, uvmin, uvmax, target_frequency, target_channel):
    uu=[]
    vv=[]
    ww=[]
    sigma=[]
    data=[]
    nvis=[]
    beam=[]
    centers=[]
    frequency=[]

    # Log the start of processing
    total_files = len(msl)
    start_time = time.time()  # Record the start time of the entire loop

    for k, ms in enumerate(msl, start=1):
        # for ms in msl:
        iteration_start_time = time.time()  # Record start time for this iteration
        logger.info(f"Processing file {k}/{total_files}: {ms}")
        #Faster than dask_ms to extract one channel
        freq, vel, UVW, SIGMA, DATA, ra, dec = read_channel_casacore(ms, uvmin, uvmax,
                                                                     target_frequency, target_channel)
        # chunks=1.e8
        # freq, vel, UVW, SIGMA, DATA, ra, dec = readms_dask(ms, uvmin, uvmax, chunks, target_frequency)
        UU = UVW[:,0]; VV = UVW[:,1]; WW = UVW[:,2]
        c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
        
        #append ms to list
        uu.append(UU); vv.append(VV); ww.append(WW)
        sigma.append(SIGMA)
        data.append(DATA)
        nvis.append(len(UU))
        beam.append(np.full(len(UU),k-1)) #beam index should start at 0
        centers.append(c)
        frequency.append(freq)

        # Time taken for this iteration
        iteration_time = time.time() - iteration_start_time
        
        # Estimate remaining time (ETA)
        elapsed_time = time.time() - start_time  # Time elapsed since the start of the loop
        avg_iteration_time = elapsed_time / k  # Average time per iteration so far
        remaining_time = avg_iteration_time * (total_files - k)  # Estimated remaining time
        
        # Log time taken for the iteration and ETA
        logger.info(f"Time for iteration {k}: {iteration_time:.2f}s, ETA: {remaining_time / 60:.2f} minutes")
        
    #concatenate all files at the end
    uu = np.concatenate(uu); vv = np.concatenate(vv); ww = np.concatenate(ww)
    sigma = np.concatenate(sigma); data = np.concatenate(data); beam = np.concatenate(beam)
    
    #sort by beam
    sort = np.argsort(beam)
    uu_lam = uu[sort]; vv_lam = vv[sort]; ww_lam = ww[sort]
    sigma = sigma[sort];
    beam = beam[sort];
    data = data[sort];

    #convert to float32
    uu_lam = np.float32(uu_lam); vv_lam = np.float32(vv_lam); ww_lam = np.float32(ww_lam)
    sigma = np.float32(sigma);
    beam = np.int32(beam);
    data = np.complex64(data)
    frequency = np.float64(frequency);
    
    # take the complex conjugate
    data = np.conj(data)

    vis_data = VisData(uu_lam, vv_lam, ww_lam, sigma, data, beam, centers, frequency, vel) #Freq in Hz

    return vis_data

if __name__ == '__main__':    
    #path data
    path_ms = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/chan950/"
    filename = "scienceData.M344-11B.SB30584_SB30625_SB30665.beam18_SL.ms.contsub_chan950.ms"    
    
    #get filenames of all ms from mspath
    msl = sorted(glob.glob(path_ms+"*.ms"))
    logger.info(f"number of ms files = {len(msl)}")
    
    uvmin = 0; uvmax=7000
    vis_data = readmsl(msl, uvmin, uvmax, chunks=1.e7)
    
        
# # Define chunk sizes to test
# chunk_sizes = [1.e4, 1.e5, 1.e6, 1.e7, 1.e8, 1.e9]  # Test multiple values

# results = {}

# for chunk in chunk_sizes:
#     logger.info(f"\n🔹 Testing chunk size: {int(chunk)}")
#     start_time = time.time()

#     try:
#         msds = xds_from_table(
#             ms_path.as_posix(),
#             index_cols=["SCAN_NUMBER", "TIME", "ANTENNA1", "ANTENNA2"],
#             group_cols=["DATA_DESC_ID"],
#             chunks={"row": int(chunk)},  # Apply chunk size
#         )[0]

#         # Force computation to measure time accurately
#         msds.DATA.data[:10].compute()

#         elapsed_time = time.time() - start_time
#         results[int(chunk)] = elapsed_time
#         logger.info(f"✅ Chunk {int(chunk)} rows -> {elapsed_time:.2f} sec")
#     except Exception as e:
#         logger.info(f"❌ Chunk {int(chunk)} rows -> Failed: {str(e)}")

# # Print summary of results
# logger.info("\n🚀 Best Chunk Size Based on Speed:")
# for chunk, time_taken in sorted(results.items(), key=lambda x: x[1]):
#     logger.info(f"🔸 Chunk {chunk}: {time_taken:.2f} sec")
