import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from casatasks import cvel2
from concurrent.futures import ProcessPoolExecutor
from deconv import logger

plt.ion()

def run_cvel2(ms, extout, nchan, start, width, restfreq):
    try:
        # Extract filename without the full path
        filename = os.path.basename(ms) 
        outputvis = ms + "_" + str(nchan) + "_" + str(start[:-4]) + "_" + str(width[:-4]) + extout  # Create the output filename with extension

        # Run cvel2
        cvel2(vis=ms, outputvis=outputvis,
              outframe='LSRK', mode='velocity', veltype='radio',
              restfreq=restfreq, nchan=nchan, start=start,
              width=width)

        logger.info(f"Processed {ms} -> {outputvis} (Rest frequency: {restfreq})")
    except Exception as e:
        logger.error(f"Error processing {ms}: {e}")

def dcvel2(path_ms, extension=".ms", extout=".vlsrk", nchan=1, start="0km/s", width="1km/s", restfreq="1.42040575177GHz", max_workers=12):
    # Get list of MS files
    msl = sorted(glob.glob(path_ms + "*" + extension))
    
    if not msl:
        logger.warning("No files found matching the extension.")
        return

    logger.info(f"Found {len(msl)} files to process.")

    # Run cvel2 in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_cvel2, ms, extout, nchan, start, width, restfreq) for ms in msl]
        
        # Wait for all tasks to finish
        for future in futures:
            future.result()  # This will raise an exception if something goes wrong

    logger.info("All MS files processed.")

if __name__ == '__main__':
    # Path to Measurement Set
    path_ms = "/home/amarchal/Projects/deconv/examples/data/MeerKAT/original/"

    dcvel2(path_ms, extension=".contsub", extout=".vlsrk", nchan=300, start='-250km/s',
           width='0.7km/s', restfreq="1.42040575177GHz", max_workers=12)
