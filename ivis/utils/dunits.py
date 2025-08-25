import numpy as np
from astropy import units as u
from astropy.constants import k_B
from astropy.constants import c as c_light

def jy_per_arcsec2_to_K(I_jy_arcsec2, nu_hz_array):
    """
    Convert an image/cube from Jy/arcsec^2 to brightness temperature [K],
    handling per-channel frequencies. No beam needed.
    """
    # Convert Jy/arcsec^2 -> W m^-2 Hz^-1 sr^-1 (treat angles as dimensionless => no leftover sr)
    I_si = (I_jy_arcsec2 * u.Jy / u.arcsec**2).to(
        u.W / u.m**2 / u.Hz / u.sr,
        equivalencies=u.dimensionless_angles()
    )
    
    # Broadcast frequency over (H,W) if needed
    nu = (np.asarray(nu_hz_array) * u.Hz)
    if nu.ndim == 1:
        nu = nu[:, None, None]
        
        # Rayleigh–Jeans: T = c^2 I_nu / (2 k_B nu^2)
        T = (c_light**2 / (2.0 * k_B * nu**2) * I_si).to(
            u.K, equivalencies=u.dimensionless_angles()
        )
    return T.value


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
