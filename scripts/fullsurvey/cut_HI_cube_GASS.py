import numpy as np
import astropy.units as u
from astropy.io import fits
from spectral_cube import SpectralCube

fitsname = "/Users/antoine/Desktop/fullsurvey/GASS_HI_LMC_cube.fits"

# Read cube
cube_sd = SpectralCube.read(fitsname)

# Make sure the spectral axis is in km/s (radio convention is usually right for HI)
cube_sd = cube_sd.with_spectral_unit(u.km/u.s, velocity_convention="radio")

v0 = 253.0 * u.km/u.s
dv = 5.0 * u.km/u.s
vmin, vmax = v0 - dv, v0 + dv   # 211..251 km/s

# Cut cube
cube_cut = cube_sd.spectral_slab(vmin, vmax)

print("Original shape:", cube_sd.shape)
print("Cut shape     :", cube_cut.shape)
print("v range cut   :", cube_cut.spectral_axis.min(), "→", cube_cut.spectral_axis.max())

# Optional: write to a new FITS
cube_cut.write("/Users/antoine/Desktop/fullsurvey/GASS_HI_LMC_cube_253pm5kms.fits",
               overwrite=True)
