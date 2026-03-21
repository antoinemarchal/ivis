import numpy as np
from pathlib import Path
from casacore.tables import table
from astropy import units as u
from astropy.constants import c

def read_ms_one_channel(ms_path, target_frequency, uvmin, uvmax):
    """
    Reads a specific channel from the MS file, selected by frequency.
    
    Args:
        ms_path (str): Path to the measurement set.
        target_frequency (float): The target frequency to select.
        uvmin (float): Minimum baseline length.
        uvmax (float): Maximum baseline length.
        
    Returns:
        tuple: Contains the selected frequency, UVW data, I, SIGMA, and other info.
    """
    # Open the measurement set
    tab = table(ms_path, readonly=True)

    # Get the spectral window table
    spw_table = table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True)
    
    # Get the frequencies from the spectral window table
    frequencies = np.squeeze(spw_table.getcol("CHAN_FREQ"))

    # Check if the target frequency is within range
    if target_frequency < frequencies.min() or target_frequency > frequencies.max():
        print("Target frequency is out of range.")
        return None

    # Find the closest frequency index
    channel_index = np.abs(frequencies - target_frequency).argmin()
    print(f"Selected channel: {channel_index}, Frequency: {frequencies[channel_index]}")

    # Get total rows (number of visibilities)
    nrows = tab.nrows()
        
    # Extract only the selected channel (all rows, specific channel, all polarizations)
    data = tab.getcolslice("DATA",
                           blc=[channel_index, 0],   # Start index: target channel, first pol
                           trc=[channel_index, -1])  # End index: target channel, last pol

    print(data.shape)
    

def read_channel_casacore(ms_path, uvmin, uvmax, chunck, target_frequency):
    """
    Reads a specific channel from the MS file using casacore.tables,
    selected by frequency, without loading the entire dataset.

    Args:
        ms_path (str): Path to the Measurement Set.
        uvmin (float): Minimum baseline length.
        uvmax (float): Maximum baseline length.
        target_frequency (float): The target frequency in Hz.

    Returns:
        tuple: (Selected frequency, UVW data, Stokes I, SIGMA, Velocity)
    """
    # Open the Measurement Set
    ms_table = table(ms_path, readonly=True)
    
    # Open the spectral window table to get frequency information
    spw_table = table(f"{ms_path}/SPECTRAL_WINDOW", readonly=True)
    frequencies = np.squeeze(spw_table.getcol("CHAN_FREQ"))

    # Check if the target frequency is within range
    if target_frequency < frequencies.min() or target_frequency > frequencies.max():
        print("Target frequency is out of range.")
        return None

    # Find the closest frequency index
    channel_index = np.abs(frequencies - target_frequency).argmin()

    # Compute the velocity for the selected channel
    rest_freq_u = 1.42040575177e9 * u.Hz  # Rest frequency of HI line (21 cm)
    velocity = ((rest_freq_u - frequencies[channel_index] * u.Hz) / rest_freq_u * c).to(u.km / u.s)

    print(f"Selected channel: {channel_index} | Frequency: {frequencies[channel_index]} Hz | Frequency: {velocity.value} km/s")

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

    # Compute Stokes I (XX + YY) / 2
    stokes_i = (data[..., 0] + data[..., -1]) * 0.5  
    # Use full sigma values (per channel)
    sigma_i = np.sqrt(sigma[..., 0]**2 + sigma[..., -1]**2)  

    # Compute baseline lengths
    wavelengths = c.value / frequencies[channel_index]  # λ = c / ν
    uvw_lambda = uvw / wavelengths
    baseline_lengths = np.sqrt((uvw_lambda ** 2).sum(axis=1))  

    # Apply UV filtering and flagging (per channel)
    cross_corr = (flag[..., 0] == False) & (flag[..., -1] == False)  # Keep only unflagged
    valid_baselines = (baseline_lengths >= uvmin) & (baseline_lengths <= uvmax)
    mask = cross_corr & valid_baselines[:, np.newaxis]  # Match shape

    # Apply mask
    uvw_lambda = uvw_lambda[mask[:, 0]]
    stokes_i = stokes_i[mask[:, 0]]
    sigma_i = sigma_i[mask[:, 0]]

    print(f"Extracted {uvw_lambda.shape[0]} valid baselines.")

    return frequencies[channel_index], uvw_lambda, stokes_i, sigma_i, velocity



if __name__ == '__main__':    
    #path data
    ms_path = "/priv/myrtle1/gaskap/karlie/meerkat2024/data/fields/original_ms/contsub/split/" #directory of measurement sets
    filename="MW-C10_1.contsub.split.ms"
    
    velocity = -155*u.km/u.s
    rest_freq_u = 1.42040575177e9 * u.Hz  # Must be in Hz
    chan_freq = rest_freq_u - (velocity * rest_freq_u) / c
    
    uvmin = 0  # Example UV minimum baseline length in meters
    uvmax = 7000  # Example UV maximum baseline length in meters
    
    # result = read_ms_one_channel(ms_path+filename, chan_freq.value, uvmin, uvmax)

    # read_ms_one_channel(ms_path+filename, chan_freq.value, uvmin, uvmax)
    result = read_channel_casacore(ms_path+filename, uvmin, uvmax, chan_freq.value)
