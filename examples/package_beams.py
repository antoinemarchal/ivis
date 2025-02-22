from importlib import reload
from astropy import units as u
import matplotlib.pyplot as plt
import argparse
from astropy.io import fits
from astropy import wcs
import numpy as np
from pathlib import Path
from scipy import optimize
from scipy import ndimage
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from radio_beam import Beam
import glob
from reproject import reproject_from_healpix, reproject_interp
import casatools
from casatools import table
from tqdm import tqdm as tqdm

from dataclasses import dataclass

import marchalib as ml

msmd = casatools.msmetadata()
ms = casatools.ms()
tb = casatools.table()

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    return w

def phasecenter(ms, leave):
    #open ms metadata
    msmd.open(ms)
    field_id = leave  # Adjust if needed
    phase_center = msmd.phasecenter(field_id)

    # Extract RA and Dec in radians
    ra_rad = phase_center['m0']['value']
    dec_rad = phase_center['m1']['value']
    
    # Convert to hms (RA) and dms (Dec) using Astropy
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=':')
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=':')

    return ra_hms, dec_dms

def get_phasecenter_from_field(ms, field_id=0):
    tb = table()  # Create a table object
    tb.open(f"{ms}/FIELD")  # Open the FIELD table in the MS

    # Extract the phase center for the given field (usually field_id = 0 for the first field)
    phase_center = tb.getcell('PHASE_DIR', field_id)

    # RA and Dec are stored in radians
    ra_rad = phase_center[0]  # RA in radians
    dec_rad = phase_center[1]  # Dec in radians

    # Convert RA (radians) to h:m:s format
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=':')
    
    # Convert Dec (radians) to d:m:s format
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=':')

    print(f"RA: {ra_hms}, Dec: {dec_dms}")

    # Close the table
    tb.close()

    return ra_hms, dec_dms

#path ms data
pathms = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/msl_fixms_concat/"
msl = sorted(glob.glob(pathms+"*.ms"))
#path beams
path="/priv/avatar/nipingel/ASKAP/SMC/data/pilot_obs/ms_data/10941_10944/wsclean-test/beam_maps/holography_beams/"
BEAMS = sorted(glob.glob(path+"*.fits"))

pathout = "/priv/avatar/amarchal/Projects/deconv/examples/data/ASKAP/BEAMS/"

for i in tqdm(np.arange(len(msl))):
    #get beam number
    beam_number_str = msl[i][:-16][-2:]
    leave_str = msl[i][:-25][-1:]
    #open beam
    fitsname="/priv/avatar/nipingel/ASKAP/SMC/data/pilot_obs/ms_data/10941_10944/wsclean-test/beam_maps/holography_beams/ASKAP_BEAM" + beam_number_str + ".fits"
    print(fitsname)
    hdu_pb = fits.open(fitsname)
    hdr = hdu_pb[0].header
    pb = hdu_pb[0].data[268] #guess by Cameron of f 21cm #FIXME
    # pb /= np.max(pb)

    #tapper
    tapper = ml.edges.apodize(0.7, pb.shape) #warning FIXME

    #get phase center coord
    print(leave_str)
    if leave_str == "A":
        ra_hms, dec_dms = phasecenter(msl[i], 0)
    elif leave_str == "B":
        ra_hms, dec_dms = phasecenter(msl[i], 1)
    else:
        ra_hms, dec_dms = phasecenter(msl[i], 2)

    # ra_hms, dec_dms = get_phasecenter_from_field(msl[i])
    c = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg), frame='icrs')
    # c = beam_positions_icrs[i]
    print(c)
    
    #Update center header pb antenna
    hdr["CRVAL1"] = c.ra.value
    hdr["CRVAL2"] = c.dec.value

    #apply rotation beam
    pb_rot = ndimage.rotate(pb*tapper, -164.9592, reshape=False) * tapper #FIXME

    #Write on disk
    hdu0 = fits.PrimaryHDU(pb_rot, header=hdr)
    hdulist = fits.HDUList([hdu0])
    hdulist.writeto(pathout + "BEAM_{:03d}.fits".format(i), overwrite=True)


# # Define central positions using Astropy SkyCoord
# src_positions = {
#     "src1": SkyCoord(ra="1h21m46s", dec="-72d19m26s", frame='icrs'),
#     "src2": SkyCoord(ra="1h15m56s", dec="-72d03m29s", frame='icrs'),
#     "src3": SkyCoord(ra="1h21m47s", dec="-71d48m15s", frame='icrs')
# }

# # Define position angles (PA) for each source in degrees
# position_angles = {
#     "src1": -164.9592 * u.deg,
#     "src2": -163.5689 * u.deg,
#     "src3": -164.9603 * u.deg
# }

# # Given offsets in degrees
# offsets = np.array([
#     [-2.75, -2.16506], [-1.75, -2.16506], [-0.75, -2.16506], [0.25, -2.16506], [1.25, -2.16506], [2.25, -2.16506], 
#     [-2.25, -1.29904], [-1.25, -1.29904], [-0.25, -1.29904], [0.75, -1.29904], [1.75, -1.29904], [2.75, -1.29904], 
#     [-2.75, -0.433013], [-1.75, -0.433013], [-0.75, -0.433013], [0.25, -0.433013], [1.25, -0.433013], [2.25, -0.433013], 
#     [-2.25, 0.433013], [-1.25, 0.433013], [-0.25, 0.433013], [0.75, 0.433013], [1.75, 0.433013], [2.75, 0.433013], 
#     [-2.75, 1.29904], [-1.75, 1.29904], [-0.75, 1.29904], [0.25, 1.29904], [1.25, 1.29904], [2.25, 1.29904], 
#     [-2.25, 2.16506], [-1.25, 2.16506], [-0.25, 2.16506], [0.75, 2.16506], [1.75, 2.16506], [2.75, 2.16506]
# ]) * u.deg

# # Compute new beam positions with rotation
# beam_positions = []
# for _ in range(4):  # Repeat 4 times
#     for src, central_pos in src_positions.items():
#         pa = position_angles[src]
        
#         # Apply rotation matrix to offsets
#         cos_pa = np.cos(pa)
#         sin_pa = np.sin(pa)
#         rotated_offsets = np.array([
#             [cos_pa * x - sin_pa * y, sin_pa * x + cos_pa * y] 
#             for x, y in offsets.to_value(u.deg)
#         ]) * u.deg
        
#         # Compute final positions
#         beam_positions.extend([central_pos.spherical_offsets_by(ra_offset, dec_offset) 
#                                for ra_offset, dec_offset in rotated_offsets])

# # Convert to an array
# beam_positions_icrs = SkyCoord(beam_positions)

# # Print results
# print(beam_positions_icrs)
