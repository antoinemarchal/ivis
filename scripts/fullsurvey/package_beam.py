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
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from radio_beam import Beam
import glob
from reproject import reproject_from_healpix, reproject_interp
import casatools
from casatools import table
from tqdm import tqdm as tqdm
import re

from dataclasses import dataclass

msmd = casatools.msmetadata()
ms = casatools.ms()
tb = casatools.table()

def apodize(radius, shape):
    """
    Create edges apodization tapper
    
    Parameters
    ----------
    nx, ny : integers
    size of the tapper
    radius : float
    radius must be lower than 1 and greater than 0.
    
    Returns
    -------
    
    tapper : numpy array ready to multiply on your image
    to apodize edges
    """
    ny = shape[0]
    nx = shape[1]

    if (radius >= 1) or (radius <= 0.):
        print('Error: radius must be lower than 1 and greater than 0.')
        return
        
    ni = np.fix(radius*nx)
    dni = int(nx-ni)
    nj = np.fix(radius*ny)
    dnj = int(ny-nj)
    
    tap1d_x = np.ones(nx)
    tap1d_y = np.ones(ny)
    
    tap1d_x[0:dni] = (np.cos(3. * np.pi/2. + np.pi/2.* (1.* np.arange(dni)/(dni-1)) ))
    tap1d_x[nx-dni:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dni)/(dni-1)) ))
    tap1d_y[0:dnj] = (np.cos(3. * np.pi/2. + np.pi/2. * (1.* np.arange( dnj )/(dnj-1)) ))
    tap1d_y[ny-dnj:] = (np.cos(0. + np.pi/2. * (1.* np.arange(dnj)/(dnj-1)) ))
    
    tapper = np.zeros((ny, nx))
    
    for i in range(nx):
        tapper[:,i] = tap1d_y
                        
    for i in range(ny):
        tapper[i,:] = tapper[i,:] * tap1d_x
        
    return tapper


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
pathms = "/totoro/anmarchal/data/gaskap/fullsurvey/untar/merge/"
msl = sorted(glob.glob(pathms+"*.ms"))
#path beams
path="/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/"
BEAMS = sorted(glob.glob(path+"*.fits"))

pathout = "/totoro/anmarchal/data/gaskap/fullsurvey/holography_beams/merge/"

for i in tqdm(range(len(msl))):
    name = msl[i]

    # --- beam number ---
    beam_match = re.search(r'beam(\d+)', name)
    if not beam_match:
        raise ValueError(f"Could not find beam number in filename: {name}")
    beam_number_str = beam_match.group(1).zfill(2)

    # --- leave_str: A/B/C from second MS_M...X_ block ---
    leave_matches = re.findall(r'MS_M[0-9+\-]+([ABC])_', name)
    if not leave_matches:
        raise ValueError(f"Could not find leave letter (A/B/C) in filename: {name}")
    leave_str = leave_matches[-1]

    # --- print info ---
    print(f"[{i}] name={name} | leave={leave_str} | beam={beam_number_str}")

    # --- open beam file ---
    fitsname = path + "ASKAP_BEAM" + beam_number_str + ".fits"
    print(f"   Opening beam: {fitsname}")

    hdu_pb = fits.open(fitsname)
    hdr = hdu_pb[0].header
    pb = hdu_pb[0].data[268]  # FIXME: hard-coded freq plane

    #tapper
    tapper = apodize(0.7, pb.shape) #warning FIXME

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
