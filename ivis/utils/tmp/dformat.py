import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from tqdm import tqdm as tqdm

def remove_flagged(archive):
    #Remove flagged data
    mask = (archive["flag"] == False)[0]
    # print("number of unflagged baselines = {}".format(len(mask)))

    data = archive["data"][0][mask]
    uu_lam = archive["uu"][mask]
    vv_lam = archive["vv"][mask]
    ww_lam = archive["ww"][mask]
    sigma = archive["sigma"][mask]
    beam = archive["beam"][mask]
    
    print("{} visibilities".format(len(data)))

    frequency = archive["frequency"]
    ra_hms = archive["ra_hms"]
    dec_dms = archive["dec_dms"]
    c = SkyCoord(ra_hms, dec_dms, unit=(u.hourangle, u.deg), frame='icrs')
    
    return uu_lam, vv_lam, ww_lam, sigma, beam, data, c, frequency


def format_data(select_fraction, archive):
    #remove flagged data and format
    uu_lam, vv_lam, ww_lam, sigma, beam, data, centers, frequency = remove_flagged(archive)

    #random selection of a subsample
    index = np.arange(len(uu_lam))
    rng = np.random.default_rng(seed=42)
    ind = rng.choice(index, size=int(select_fraction * len(uu_lam)))
    uu_lam = uu_lam[ind]
    vv_lam = vv_lam[ind]
    ww_lam = ww_lam[ind]
    sigma = sigma[ind]
    beam = beam[ind]
    data = data[ind]
    print("Keep random {} % of the data".format(select_fraction*100))
    print("Number of visibilities kept = {}".format(len(ind)))

    #Sort by beam
    sort = np.argsort(beam)
    uu_lam = uu_lam[sort]
    vv_lam = vv_lam[sort]
    ww_lam = ww_lam[sort]
    sigma = sigma[sort]
    beam = beam[sort]
    data = data[sort]

    # convert to float32
    uu_lam = np.float32(uu_lam)
    vv_lam = np.float32(vv_lam)
    ww_lam = np.float32(ww_lam)
    sigma = np.float32(sigma)
    beam = np.int32(beam)
    data = np.complex64(data)

    return uu_lam, vv_lam, ww_lam, sigma, beam, data, centers, frequency

