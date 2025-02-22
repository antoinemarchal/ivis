from importlib import reload
from astropy import units as u
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from astropy import wcs
import numpy as np
from jax_finufft import nufft2
from pathlib import Path
from scipy import optimize
import pandas as pd
from astropy.coordinates import SkyCoord
from radio_beam import Beam
import glob
from reproject import reproject_from_healpix, reproject_interp
from tqdm import tqdm as tqdm
import os

import marchalib as ml

# Define central positions using Astropy SkyCoord
src_positions = {
    "src1": SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs'),
    "src2": SkyCoord(ra="1h15m56s", dec="-72d03m29s", frame='icrs'),
    "src3": SkyCoord(ra="1h21m47s", dec="-71d48m15s", frame='icrs')
}

# Define position angles (PA) for each source in degrees
position_angles = {
    "src1": -164.9592 * u.deg,
    "src2": -163.5689 * u.deg,
    "src3": -164.9603 * u.deg
}

# Given offsets in degrees
offsets = np.array([
    [-2.75, -2.16506], [-1.75, -2.16506], [-0.75, -2.16506], [0.25, -2.16506], [1.25, -2.16506], [2.25, -2.16506], 
    [-2.25, -1.29904], [-1.25, -1.29904], [-0.25, -1.29904], [0.75, -1.29904], [1.75, -1.29904], [2.75, -1.29904], 
    [-2.75, -0.433013], [-1.75, -0.433013], [-0.75, -0.433013], [0.25, -0.433013], [1.25, -0.433013], [2.25, -0.433013], 
    [-2.25, 0.433013], [-1.25, 0.433013], [-0.25, 0.433013], [0.75, 0.433013], [1.75, 0.433013], [2.75, 0.433013], 
    [-2.75, 1.29904], [-1.75, 1.29904], [-0.75, 1.29904], [0.25, 1.29904], [1.25, 1.29904], [2.25, 1.29904], 
    [-2.25, 2.16506], [-1.25, 2.16506], [-0.25, 2.16506], [0.75, 2.16506], [1.75, 2.16506], [2.75, 2.16506]
]) * u.deg

# Compute new beam positions with rotation
beam_positions = []
for _ in range(4):  # Repeat 4 times
    for src, central_pos in src_positions.items():
        pa = position_angles[src]
        
        # Apply rotation matrix to offsets
        cos_pa = np.cos(pa)
        sin_pa = np.sin(pa)
        rotated_offsets = np.array([
            [cos_pa * x - sin_pa * y, sin_pa * x + cos_pa * y] 
            for x, y in offsets.to_value(u.deg)
        ]) * u.deg
        
        # Compute final positions
        beam_positions.extend([central_pos.spherical_offsets_by(ra_offset, dec_offset) 
                               for ra_offset, dec_offset in rotated_offsets])

# Convert to an array
coords = SkyCoord(beam_positions)

# # Print results
# print(coords)

# Extract RA (HMS) and Dec (DMS) with full precision
ra_h, ra_m, ra_s = coords.ra.hms
dec_d, dec_m, dec_s = coords.dec.dms

# Separate RA and Dec into lists
ra_list = [f"{int(ra_h[i]):02d}h{int(ra_m[i]):02d}m{ra_s[i]:.8f}s" for i in range(len(ra_h))]
dec_list = [f"{int(dec_d[i]):+03d}d{int(dec_m[i]):02d}m{abs(dec_s[i]):.8f}s" for i in range(len(dec_d))]

pathms = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/msl/"
msl = sorted(glob.glob(pathms+"*.ms"))

os.system("export LD_LIBRARY_PATH=/pkg/linux/casa-release-5.4.1-32.el7/lib/:${LD_LIBRARY_PATH}")

#Beams
beams = np.arange(len(msl))
for i in tqdm(beams):
    os.system("chgcentre " + msl[i] + " " + ra_list[i] + " " + dec_list[i])
