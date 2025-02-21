import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation, Angle
from astropy import constants as const
from astropy.time import Time
import astropy.units as u
from casatools import table
import casatools

msmd = casatools.msmetadata()

def get_vlsrk_for_fov(msfile, fov_size_deg=6.0, grid_size=10, plot=True):
    tb = table()

    # 🔹 Extract Observation Time (MJD)
    tb.open(msfile)
    time_mjd = tb.getcol("TIME")[0] / 86400.0  # Convert from seconds to days
    tb.close()
    obs_time = Time(time_mjd, format='mjd')

    # 🔹 Extract ASKAP Position from ANTENNA Table
    tb.open(msfile + "/ANTENNA")
    antenna_positions = tb.getcol("POSITION")  # X, Y, Z in meters
    tb.close()
    x, y, z = antenna_positions[:, 0]
    askap_location = EarthLocation.from_geocentric(x, y, z, unit=u.m)

    # 🔹 Extract Phase Center from FIELD Table
    tb.open(msfile + "/FIELD")
    phase_dir = tb.getcol("PHASE_DIR")  # Shape: (2, 1, num_fields)
    tb.close()
    ra_center, dec_center = np.rad2deg(phase_dir[:, 0, 0])  # Convert to degrees

    # 🔹 Generate a Grid of Source Positions Across the FoV
    ra_offsets = np.linspace(-fov_size_deg / 2, fov_size_deg / 2, grid_size)
    dec_offsets = np.linspace(-fov_size_deg / 2, fov_size_deg / 2, grid_size)
    ra_grid, dec_grid = np.meshgrid(ra_offsets, dec_offsets)

    vlsrk_map = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            ra = ra_center + ra_grid[i, j]
            dec = dec_center + dec_grid[i, j]
            source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

            # 🔹 Compute Barycentric Velocity Correction
            barycorr = source.radial_velocity_correction(kind="barycentric", 
                                                         obstime=obs_time, 
                                                         location=askap_location)

            # 🔹 Compute LSRK Correction
            v_sun_magnitude = 20.0 * u.km / u.s
            l_sun, b_sun = np.deg2rad(57.0), np.deg2rad(25.0)

            l, b = source.galactic.l.radian, source.galactic.b.radian
            lsrk_corr = v_sun_magnitude * (np.cos(b) * np.cos(b_sun) * np.cos(l - l_sun) + np.sin(b) * np.sin(b_sun))

            vlsrk_map[i, j] = lsrk_corr.to(u.km/u.s).value

    # 🔹 Plot VLSRK Variations Across the Field of View
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(vlsrk_map, extent=[-fov_size_deg/2, fov_size_deg/2, -fov_size_deg/2, fov_size_deg/2],
                   origin='lower', cmap='coolwarm')
        plt.colorbar(label="LSRK Velocity Correction (km/s)")
        plt.xlabel("RA Offset (deg)")
        plt.ylabel("Dec Offset (deg)")
        plt.title("VLSRK Variations Across ASKAP's Field of View")
        plt.show()

    return vlsrk_map

if __name__ == '__main__':    
    # # Example usage
    path="/home/amarchal/Projects/deconv/examples/data/ASKAP/msl_fixms/"
    msfile = "scienceData.MS_M345-09A_4.SB68827.MS_M345-09C_4.beam20_SL.dop.1chan.ms"  # Replace with your actual MS file

    get_vlsrk_for_fov(path+msfile, fov_size_deg=6.0, grid_size=10)
