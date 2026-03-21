import numpy as np
import shutil
import os
from casatools import componentlist
from casatasks import simobserve, listobs
import casatasks
import matplotlib.pyplot as plt

def create_component_list(component_list_path, BASE_DIR):
    """Deletes the component list file if it exists."""
    filename = os.path.join(BASE_DIR, "sources.cl")
    if os.path.exists(filename):
        try:
            shutil.rmtree(filename)
            print(f"✅ Deleted component list: {component_list_path}")
        except Exception as e:
            print(f"❌ Error removing component list: {e}")
            
    """Creates a CASA component list with known sources."""
    cl = componentlist()  # Initialize the component list  

    # # Add an extended Gaussian source  
    # cl.addcomponent(
    #     flux=1.0, fluxunit='Jy', shape='gaussian', freq='1.4GHz',
    #     dir='J2000 12h32m00.0s -30d05m00.0s',
    #     majoraxis='10arcsec', minoraxis='5arcsec', positionangle='45deg'
    # )

    # HI line parameters
    nu0 = 1.420405751e9  # HI rest frequency in Hz
    velocity_width = 10.0  # Total velocity range in km/s
    velocity_res = 1.0  # Channel resolution in km/s
    num_channels = 10  # 10 spectral channels
    c = 3.0e5  # Speed of light in km/s

    # Compute frequency grid (±5 km/s around HI rest frequency)
    velocities = np.linspace(-5.0, 5.0, num_channels)  
    frequencies = nu0 * (1 + velocities / c)  

    # Define Gaussian spectral flux profile
    sigma = velocity_width / 2.355  # Convert FWHM to Gaussian sigma
    fluxes = np.exp(-0.5 * (velocities / sigma) ** 2)  # Normalized Gaussian

    # Normalize peak flux
    fluxes *= 1.0  # Peak flux = 1 Jy

    # Add spectral channels as individual components
    for freq, flux in zip(frequencies, fluxes):
        cl.addcomponent(
            flux=flux, fluxunit="Jy", shape="gaussian",
            freq=f"{freq / 1e9}GHz",  
            dir="J2000 12h30m00.0s -30d00m00.0s",
            majoraxis="1arcmin", minoraxis="40arcsec", positionangle="45deg"
        )

    # Save and close component list  
    cl.rename(component_list_path)  
    cl.close()  

    return component_list_path  

def run_simulation(component_list_path, project_name):
    """Runs simobserve to create a synthetic Measurement Set (MS)."""

    simobserve(
        complist=component_list_path,
        project=project_name,
        incenter="1.420405751 GHz",  # Must be a valid string
        inwidth="4.735 kHz",  # 1.0 km/s per channel
        compwidth="47.35kHz",  # Defines total bandwidth (ensures 10 channels)
        totaltime="3600s",
        mapsize="2arcmin",
        # obsmode="int",
        refdate="2023/01/01",
        indirection="J2000 12h30m00.0s -30d00m00.0s",  # Forces CASA to treat it as a spectral cube
        hourangle="0.0h",
        integration="10s",
        thermalnoise="",
        antennalist="/pkg/linux/casa-release-5.4.1-32.el7/data/alma/simmos/vla.d.cfg",  # Pass absolute path for antennalist
    )

def inspect_ms(project_name, BASE_DIR):
    """Lists the observation summary of the generated MS."""
    ms_path = os.path.join(BASE_DIR, project_name, project_name + ".vla.d.ms")  # Ensure correct MS path

    try:
        listobs(ms_path)  
    except Exception as e:
        print(f"Error: Measurement Set not found. {e}")


if __name__ == '__main__':
    # Define base directory for all output
    BASE_DIR = "/home/amarchal/Projects/deconv/examples/data/simu"
    os.makedirs(BASE_DIR, exist_ok=True)  # Ensure the base directory exists
    
    # Define paths inside BASE_DIR
    component_list_path = "sources.cl"
    project_path = "simu_HI"  # Ensure project is inside BASE_DIR
    
    # Run simulation steps  
    create_component_list(component_list_path, BASE_DIR)
    run_simulation(component_list_path, project_path)
    inspect_ms(project_path, BASE_DIR)

    # Look at result
    ms_file = BASE_DIR+"/"+project_path+"/simu_HI.vla.d.ms"
    casatasks.listobs(ms_file)
    casatasks.listobs(ms_file, listfile="simu_HI/simu_HI_listobs.txt", overwrite=True)
