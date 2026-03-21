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
from scipy import ndimage
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from radio_beam import Beam
import glob
from reproject import reproject_from_healpix, reproject_interp
import casatools
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

def phasecenter(ms):
    #open ms metadata
    msmd.open(ms)
    field_id = 0  # Adjust if needed
    phase_center = msmd.phasecenter(field_id)

    # Extract RA and Dec in radians
    ra_rad = phase_center['m0']['value']
    dec_rad = phase_center['m1']['value']
    
    # Convert to hms (RA) and dms (Dec) using Astropy
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=':')
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=':')

    return ra_hms, dec_dms

def get_pointing(msfile):
    # Open POINTING table
    tb.open(msfile + "/POINTING")
    directions = tb.getcol("DIRECTION")  # Shape: (2, N)
    tb.close()

    print(directions.shape)
    
    # Extract the most recent pointing direction (last entry)
    ra_rad = directions[0][0][-1]  # Last RA in radians
    dec_rad = directions[1][0][-1]  # Last Dec in radians

    # Convert RA and Dec to degrees
    ra_deg = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    # Convert RA and Dec to hms and dms using Astropy Angle
    ra_hms = Angle(ra_deg, unit=u.deg).to_string(unit=u.hour, precision=2, sep=':')
    dec_dms = Angle(dec_deg, unit=u.deg).to_string(unit=u.deg, precision=2, sep=':')
    
    return ra_hms, dec_dms


#path ms data
pathms = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/chan950/"
msl = sorted(glob.glob(pathms+"*.ms"))
#path beams
path="/priv/avatar/nipingel/ASKAP/SMC/data/pilot_obs/ms_data/10941_10944/wsclean-test/beam_maps/holography_beams/"
BEAMS = sorted(glob.glob(path+"*.fits"))

pathout = "/priv/avatar/amarchal/MPol-dev/examples/workflow/data/BEAMS/"

for i in tqdm(np.arange(len(msl))):
    #get beam number
    beam_number_str = msl[i][:-25][-2:]
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
    ra_hms, dec_dms = phasecenter(msl[i])
    c = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg), frame='icrs')
    print(c)

    ra_hms_pos, dec_dms_pos = get_pointing(msl[i])
    c_pos = SkyCoord(ra_hms_pos, dec_dms_pos, unit=(u.hourangle, u.deg), frame='icrs')
    print(c_pos)
    
    #Update center header pb antenna
    hdr["CRVAL1"] = c.ra.value
    hdr["CRVAL2"] = c.dec.value

    #apply rotation beam
    pb_rot = ndimage.rotate(pb*tapper, 67.531, reshape=False) * tapper #FIXME

    #Write on disk
    hdu0 = fits.PrimaryHDU(pb_rot, header=hdr)
    hdulist = fits.HDUList([hdu0])
    hdulist.writeto(pathout + "BEAM_{:03d}.fits".format(i), overwrite=True)
