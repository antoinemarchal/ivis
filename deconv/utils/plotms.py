import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from daskms import xds_from_ms
import dask.array as da

plt.ion()

from deconv.utils.vlsrk_from_ms import print_spectral_window_frame, convert_freq_to_velocity

def plotms(msfile):
    # Print window frame
    print_spectral_window_frame(msfile)

    # Compute velocity array
    rest_freq = 1.42040575177e9 * u.Hz  # More precise 21 cm HI line rest frequency
    velocity = convert_freq_to_velocity(msfile, rest_freq).to(u.km/u.s)
    
    # Load only necessary columns with chunking
    ds = xds_from_ms(
        msfile,
        columns=["DATA", "ANTENNA1", "ANTENNA2"],  # Load antenna columns for filtering
        group_cols=[],  
        chunks={"row": 1000}  # Adjust based on dataset size
    )[0]
    
    # Compute ANTENNA1 and ANTENNA2 masks efficiently
    antenna1 = ds.ANTENNA1.compute()  # Avoids Dask boolean indexing issue
    antenna2 = ds.ANTENNA2.compute()
    selected_rows = np.where((antenna1 == 2) & (antenna2 == 3))[0]  # Get row indices
    
    # Apply filtering using isel()
    ds_filtered = ds.isel(row=selected_rows)

    # Extract XX (corr=0) and YY (corr=1) without computing yet
    data_xx = ds_filtered.DATA[..., 0]  # XX
    data_yy = ds_filtered.DATA[..., 1]  # YY
    
    # Compute mean over rows (baselines) using Dask (keeps laziness)
    mean_xx = data_xx.mean(axis=0)
    mean_yy = data_yy.mean(axis=0)
    
    # Compute Stokes I using Dask (still lazy)
    stokes_I = (mean_xx + mean_yy) / 2
    
    # Compute only when needed (final result)
    stokes_I_computed = stokes_I.compute()

    # Plot 1: XX Amplitude vs Channel
    plt.figure(figsize=(10, 5))
    plt.plot(np.abs(stokes_I_computed), label="Stokes I (Intensity)", color="black")
    plt.xlabel("Channel")
    plt.ylabel("Mean Intensity (Jy)")
    plt.title("Stokes I Intensity vs. Channel")
    plt.grid()
    plt.legend()
    plt.show()
    
    # Plot 2: XX Amplitude vs Velocity
    plt.figure(figsize=(10, 5))
    plt.plot(velocity, np.abs(stokes_I_computed), label="Stokes I (Intensity)", color="black")
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Mean Intensity (Jy)")
    plt.title("Stokes I Intensity vs. Velocity")
    plt.grid()
    plt.legend()
    plt.show()
    


if __name__ == '__main__':
    # Path to Measurement Set
    pathms = "/home/amarchal/Projects/deconv/examples/data/MeerKAT/original_contsub/"
    msfile = "MW-C10_5.ms.contsub"

    plotms(pathms+msfile)
