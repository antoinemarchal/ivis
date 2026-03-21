import glob
import os
import time
import shutil
from casatools import table
from casatasks import uvcontsub, uvcontsub_old
from concurrent.futures import ProcessPoolExecutor

def remove_existing_ms(output_ms):
    """ Removes the existing Measurement Set (MS) if it exists. """
    if os.path.exists(output_ms):
        print(f"Output MS {output_ms} exists. Deleting it.")
        shutil.rmtree(output_ms)

        
def remove_weights_column(ms_path):
    """ Removes the WEIGHT_SPECTRUM column manually from the MS. """
    ms_table = table(ms_path, readonly=False)

    # Check if 'WEIGHT_SPECTRUM' exists
    if 'WEIGHT_SPECTRUM' in ms_table.colnames():
        # Ask the user if they want to proceed with removing the column
        user_input = input("WEIGHT_SPECTRUM column found. Do you want to remove it? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Removing WEIGHT_SPECTRUM column.")
            ms_table.removecols(['WEIGHT_SPECTRUM'])  # Remove WEIGHT_SPECTRUM column
        else:
            print("Skipping removal of WEIGHT_SPECTRUM column.")
    else:
        print("No WEIGHT_SPECTRUM column found. Skipping removal.")


def subtract_continuum(ms_path, output_ms, spw='0:1~550;1150~1450', fitorder=1, datacolumn='data', old=True):
    """ Performs continuum subtraction on a Measurement Set (MS) using the uvcontsub task. """
    # Remove the existing output MS if it exists
    remove_existing_ms(output_ms)

    # # Ask the user if they want to remove the WEIGHT_SPECTRUM column before continuing
    # remove_weights_column(ms_path)

    # Call the uvcontsub task on the MS copy
    if old == False:
        print("Using uvcontsub.")

        uvcontsub(vis=ms_path,
                  outputvis=output_ms,
                  spw=spw,
                  fitmethod='gsl',  # GSL fitting method
                  fitorder=fitorder,
                  datacolumn=datacolumn,
                  writemodel=False)

        print(f"Continuum subtraction completed for {ms_path}. Output saved in {output_ms}.")

    else:
        print("Using uvcontsub_old.")
        print("Writing file in same location with a .consub extension.")

        uvcontsub_old(vis=ms_path, fitspw=spw, fitorder=fitorder)


if __name__ == '__main__':
    #Get all ms from path
    ms_path = "/home/amarchal/Projects/deconv/examples/data/MeerKAT/original/"
    msl = sorted(glob.glob(ms_path+"*.ms"))
    
    n_cores = 5  # Change this to the number of cores you want to use
    
    def process_ms(ms):
        subtract_continuum(ms, "", spw='0:1~550;1150~1450', fitorder=1, datacolumn='data', old=True)

    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        executor.map(process_ms, msl)
        
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
