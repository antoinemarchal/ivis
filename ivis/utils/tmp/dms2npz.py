#adapted from Ian C. MPol-dev
#amarchal 10/24
import glob
import time
from importlib import reload
import numpy as np
import casatools
from casatools import msmetadata
import matplotlib.pyplot as plt
import argparse
from astropy.constants import c
from astropy.coordinates import Angle
import astropy.units as u
from tqdm import tqdm as tqdm
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import wcs

from deconv.utils import process_casa

msmd = casatools.msmetadata()
ms = casatools.ms()
tb = casatools.table()

def phasecenter(ms):
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


def get_baselines(msl, select_fraction=0.1, sigma_rescale=1.0, incl_model_data=False,
                  datacolumn="data", nchan=1, start=0, width=1, inc=1, uvmin=0, uvmax=2000):

    #get metadata
    spw_id = 0
    msmd.open(msl[0])
    chan_freq = msmd.chanfreqs(spw_id)
    chanres = msmd.chanres(spw_id, asvel=True)
    # chaneffbws = msmd.chaneffbws(spw_id, asvel=True)
    msmd.done()
    nchan_tot = len(chan_freq)
    print("number of channels = ", nchan_tot)
    print("channel freqs in GHz = ", chan_freq*1.e-9)
    print("channel resolution in km/s = ", chanres[0]) #same spacing
    HIrestfreq=1.420405752*u.GHz
    radio_HI_equiv = u.doppler_radio(HIrestfreq)
    chan_freq = chan_freq*1.e-9 * u.GHz
    radio_velocity = chan_freq.to(u.km/u.s, equivalencies=radio_HI_equiv)
    # print(radio_velocity)

    # calculate wavelengths in meters
    wavelengths = c.value / chan_freq[start].value*1.e-9  # m
    frequency = chan_freq[start] # [GHz]
    print("Processed velocity {}".format(radio_velocity[start]))

    # read data
    uu = []
    vv = []
    ww = []
    weight = []
    sigma = []
    data = []
    flag = []
    nvis = []
    beam = []
    ra_hms = []
    dec_dms = []
    k=0
    for ms in tqdm(msl):
        print("process file: ", ms)
        # get processed visibilities
        # including complex conjugation
        d = process_casa.get_processed_visibilities(filename=ms, datadescid=0,
                                                    sigma_rescale=sigma_rescale,
                                                    incl_model_data=incl_model_data,
                                                    datacolumn=datacolumn, nchan=nchan,
                                                    start=start, width=width, inc=inc,
                                                    uvmin=uvmin, uvmax=uvmax)

        #append and destroy polarizatino axis
        uu.append(d["uu"])
        vv.append(d["vv"])
        ww.append(d["ww"])
        weight.append(d["weight"])
        sigma.append(d["sigma"])
        data.append(d["data"])
        flag.append(d["flag"])
        nvis.append(len(d["uu"]))
        beam.append(np.full(len(d["uu"]),k))
        ra_hms.append(d["ra_hms"])
        dec_dms.append(d["dec_dms"])
        k+=1
        
    # concatenate all files at the end
    uu = np.concatenate(uu)
    vv = np.concatenate(vv)
    ww = np.concatenate(ww)
    weight = np.concatenate(weight)
    sigma = np.concatenate(sigma)
    data = np.concatenate(data, axis=1)
    flag = np.concatenate(flag, axis=1)
    beam = np.concatenate(beam)

    index = np.arange(len(uu))
    rng = np.random.default_rng(seed=42)
    ind = rng.choice(index, size=int(select_fraction * len(uu)))

    print("Keep random {} % of the data".format(select_fraction*100))
    print("Number of visibilities kept = {}".format(len(ind)))

    # calculate baselines in lambda
    uu = uu / wavelengths  # [lambda]
    vv = vv / wavelengths  # [lambda]
    ww = ww / wavelengths  # [lambda]
    
    return frequency, uu[ind], vv[ind], ww[ind], weight[ind], sigma[ind], data[:,ind], flag[:,ind], beam[ind], ra_hms, dec_dms
