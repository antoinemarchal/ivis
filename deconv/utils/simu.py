import shutil
import os
from casatools import componentlist
from casatasks import simobserve, listobs

def clear_log_directory():
    """Clears the logtable directory if it exists to avoid directory conflicts."""
    log_dir = 'logtable'  # Default log directory
    if os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)  # Remove the entire logtable directory
            print(f"Removed existing logtable directory: {log_dir}")
        except Exception as e:
            print(f"Error removing logtable directory: {e}")
    
def create_component_list():
    """Creates a CASA component list with known sources."""
    cl = componentlist()  # Initialize the component list
    
    # Add two sources with different flux and positions
    cl.addcomponent(
        flux=1.0, fluxunit='Jy', shape='point', freq='1.4GHz',
        dir='J2000 12h30m00.0s -30d00m00.0s'  # Source 1
    )
    cl.addcomponent(
        flux=0.5, fluxunit='Jy', shape='point', freq='1.4GHz',
        dir='J2000 12h31m00.0s -30d10m00.0s'  # Source 2
    )
    
    # Save the component list in the default location (current working directory)
    component_list_path = 'mysources.cl'
    cl.rename(component_list_path)  # Save the component list file
    cl.close()  # Close the component list tool
    
    return component_list_path  # Return the path to the component list file
    
def run_simulation(component_list_path):
    """Runs simobserve to create a synthetic Measurement Set (MS)."""
    # Ensure the logtable directory is cleared before running the simulation
    clear_log_directory()
    
    # Run the simulation with proper frequency formatting
    simobserve(
        project='mysim',  # Default project name, will create a 'mysim.ms'
        complist=component_list_path,  # Use the component list file for the sky model
        incenter='1.4GHz',  # Central frequency as '1.4e9' for 1.4 GHz
        inwidth='10MHz',  # Bandwidth as '10e6' for 10 MHz
        totaltime='3600s',  # Total observation time (1 hour)
        compwidth='8GHz',
        antennalist='/home/amarchal/Projects/deconv/examples/data/simu/ngvla-revD.spiral.cfg'  # Antenna list
    )
    
def inspect_ms():
    """Lists the observation summary of the generated MS."""
    ms_path = 'mysim.ms'  # Default Measurement Set name
    try:
        listobs(ms_path)  # List the MS content and summary
    except Exception as e:
        print(f"Error: Measurement Set not found. {e}")
        
if __name__ == '__main__':
    # Step 1: Create the component list
    component_list_path = create_component_list()
    # Step 2: Run the simulation with the correct component list path
    run_simulation(component_list_path)
    
    # Step 3: Inspect the Measurement Set to verify the simulation
    inspect_ms()
    
