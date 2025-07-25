import numpy as np
from astropy import units as u

def _lambda_to_radpix(lam, cell_size):
    #this function was taken from MPol/fourier
    #convert cell_szie from arcsec to rad
    cell_size_rad = cell_size.to(u.rad)
    
    # lambda is equivalent to cycles per sky radian
    # convert from 'cycles per sky radian' to 'radians per sky radian'
    u_rad_per_rad = lam * 2 * np.pi  # [radians / sky radian]
    
    # convert from 'radians per sky radian' to 'radians per sky pixel'
    # assumes pixels are square and dl and dm are interchangeable
    u_rad_per_pix = u_rad_per_rad * cell_size_rad # [radians / pixel]
    
    return u_rad_per_pix
