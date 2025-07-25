#adapted from Ian C. MPol-dev
#amarchal 10/24
import time as timeos
from importlib import reload
import numpy as np
from astropy.coordinates import Angle
import astropy.units as u

from deconv.utils import process

try:
    import casatools

    # initialize the relevant CASA tools
    msmd = casatools.msmetadata()
    ms = casatools.ms()
except ModuleNotFoundError as e:
    print(
        "casatools module not found on system. If your system configuration is compatible, you can try installing these optional dependencies with `pip install 'visread[casa]'`. More information on Modular CASA can be found https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages "
    )
    raise e

def get_processed_visibilities(
        filename, datadescid, sigma_rescale=1.0, incl_model_data=None,
        datacolumn="corrected_data", nchan=1, start=0, width=1, inc=1,
        uvmin=0., uvmax=2000.
):
    r"""
    Process all of the visibilities from a specific datadescid. This means

    * (If necessary) reversing the channel dimension such that channel frequency decreases with increasing array index (blueshifted to redshifted)
    * averaging the polarizations together
    * rescaling weights
    * scanning and removing any auto-correlation visibilities

    Args:
        filename (str): path to measurementset to process
        datadescid (int): a specific datadescid to process
        sigma_rescale (float): by what factor should the sigmas be rescaled (applied to weights via ``rescale_weights``)
        incl_model_data (bool): include the model_data column?

    Returns:
        dictionary with keys "frequencies", "uu", "data", "flag", "weight"
    """

    # # get sorted channels, data, and flags
    # chan_freq, data, flag, model_data = get_data(
    #     filename, datadescid, incl_model_data, datacolumn=datacolumn,
    #     nchan=nchan, start=start, width=width, inc=inc
    # )
    model_data = None

    print("process channel number {}".format(start))

    # get the channel frequencies
    msmd.open(filename)
    chan_freq = msmd.chanfreqs(datadescid)
    field_id = 0  # Adjust if needed
    phase_center = msmd.phasecenter(field_id)

    # Extract RA and Dec in radians
    ra_rad = phase_center['m0']['value']
    dec_rad = phase_center['m1']['value']
    
    # Convert to hms (RA) and dms (Dec) using Astropy
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=':')
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=':')
    
    # Print the results
    print(f"Phase center for field ID {field_id}:")
    print(f"Right Ascension (RA): {ra_hms} (h:m:s)")
    print(f"Declination (Dec): {dec_dms} (d:m:s)")
    
    msmd.done()

    # get the data and flags
    ms.open(filename)
    ms.selectinit(datadescid=datadescid)
    ms.selectchannel(nchan=nchan,start=start,width=width,inc=inc)
    ms.select({'uvdist':[uvmin,uvmax]})
    # ms.selectpolarization(["V"])
    ms.selectpolarization(["XX","YY"])
    
    # ms.selectpolarization(["I"])
    # ms.selecttaql('ANTENNA1!=ANTENNA2') #unsure if it works

    # #change to velocity reference frame FIXME
    # ms.cvel(mode='velocity', outframe='LSRK', veltype='radio', restfreq='1.420405752GHz')
    # stop

    #column to getdata
    keys = ["flag", datacolumn, "uvw", "weight", "antenna1", "antenna2", "sigma"]

    #init FAST
    t0 = timeos.time()
    data = []
    flag = []
    uvw = []
    weight = []
    sigma = []
    ant1 = []
    ant2 = []
    ms.iterinit(interval=1000) #1000 seems to be optimal
    ms.iterorigin()
    moretodo = True
    while moretodo:
        # print(moretodo)
        if incl_model_data:
            keys += ["model_data"]
            q = ms.getdata(keys)
            model_data = q["model_data"]
        else:
            q = ms.getdata(keys)
            data.append(q[datacolumn])
            flag.append(q["flag"])
            uvw.append(q["uvw"])
            weight.append(q["weight"])
            sigma.append(q["sigma"])
            ant1.append(q["antenna1"])
            ant2.append(q["antenna2"])

            # print("data shape = ", q[datacolumn].shape)
            # print("uvw shape = ", q["uvw"].shape)
            # print("sigma shape = ", q["sigma"].shape)
            moretodo = ms.iternext()
    ms.selectinit(reset=True)
    ms.close()

    #concatenate along visibility axis
    data = np.concatenate(data,2)
    flag = np.concatenate(flag,2)
    uvw = np.concatenate(uvw,1)
    weight = np.concatenate(weight,1)
    sigma = np.concatenate(sigma,1)
    ant1 = np.concatenate(ant1,0)
    ant2 = np.concatenate(ant2,0)
    uu, vv, ww = uvw  # [m]
    t1 = timeos.time()
    print("time in s = {}".format(t1-t0))

    # # get baselines, weights, and antennas
    # ms.open(filename)
    # ms.selectinit(datadescid=datadescid)
    # # ms.selectpolarization(["XX","YY"])
    # q = ms.getdata(["uvw", "weight", "antenna1", "antenna2", "time"])
    # ms.selectinit(reset=True)
    # ms.close()
    # uvw = q["uvw"]
    # weight = q["weight"]
    # ant1 = q["antenna1"]
    # ant2 = q["antenna2"]
    # time = q["time"]
    # uu, vv, ww = q["uvw"]  # [m]

    # rescale weights FIXME
    if data.shape[0] == 2:
        weight = process.rescale_weights(weight, sigma_rescale)
        sigma = process.rescale_weights(sigma, sigma_rescale)

    # calculate the cross correlation mask
    xc = np.where(ant1 != ant2)[0]
    # apply the xc mask across channels, drop autocorrelation channels
    uu = uu[xc]
    vv = vv[xc]
    ww = ww[xc]
    data = data[:,:, xc]
    flag = flag[:,:, xc]
    # uvw = uvw[:, xc]
    weight = weight[:, xc]
    sigma = sigma[:, xc]

    # print("shape data [npol, nchan, nvis] = ", data.shape)
    # print("shape flag [npol, nchan, nvis] = ", flag.shape)
    print("shape sigma [npol, nvis] = ", sigma.shape)    

    if data.shape[0] == 2:
        # average the data across polarization
        data = process.average_data_polarization(data, weight)
        flag = process.average_flag_polarization(flag)
        
        if incl_model_data:
            model_data = process.average_data_polarization(model_data, weigh)

        # finally average weights across polarization
        weight = process.average_weight_polarization(weight)
        sigma = process.average_weight_polarization(sigma)

    # take the complex conjugate
    data = np.conj(data)
    if incl_model_data:
        model_data = np.conj(model_data)

    # print("average pol___")    
    # print("shape data [nchan, nvis] = ", data.shape)
    # print("shape flag [nchan, nvis] = ", flag.shape)
    # print("shape weight [nvis] = ", weight.shape)    

    return {
        # "frequencies": chan_freq,
        "uu": uu,
        "vv": vv,
        "ww": ww,
        "antenna1": ant1,
        "antenna2": ant2,
        # "time": time,
        "data": data,
        "model_data": model_data,
        "flag": flag,
        "weight": weight,
        "sigma": sigma,
        "ra_hms": ra_hms,
        "dec_dms": dec_dms
    }
