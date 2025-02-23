import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, Angle
from astropy import constants as const
from astropy.time import Time
import astropy.units as u
from casatools import table, msmetadata
import casatools
from astropy.constants import c

# Create an msmetadata object
msmd = msmetadata()

def convert_freq_to_velocity(ms_file, rest_frequency=1420405751.77):
    """Converts frequency to velocity using the correct radio convention."""
    tb = table()
    
    try:
        # Open the SPECTRAL_WINDOW table within the MS file
        tb.open(ms_file + '/SPECTRAL_WINDOW')
    except Exception as e:
        print(f"Error opening SPECTRAL_WINDOW table: {e}")
        return
    
    # Get observed frequencies (CHAN_FREQ)
    if 'CHAN_FREQ' in tb.colnames():
        chan_freq = tb.getcol('CHAN_FREQ')  # Observed frequencies (Hz)
        
        # Convert frequencies to velocities using the radio convention
        rest_freq_u = rest_frequency  # Convert rest frequency to astropy units
        velocities = (rest_freq_u - chan_freq * u.Hz) / rest_freq_u * c  # Using the correct formula

        # Print observed frequencies, rest frequency, and velocity results
        print(f"Observed Frequency (CHAN_FREQ) | Rest Frequency (HI Line) | Velocity (km/s)")
        
        # Print only the first and last values
        print(f"First channel: {chan_freq[0][0]*u.Hz} | {rest_frequency} | {velocities[0].to(u.km/u.s).value}")
        print(f"Last channel: {chan_freq[-1][0]*u.Hz} | {rest_frequency} | {velocities[-1].to(u.km/u.s).value}")
        
    else:
        print("Required column 'CHAN_FREQ' not found in SPECTRAL_WINDOW table.")
    
    tb.close()

    return velocities


def get_spectral_window_info(ms_file):
    """Extracts and prints all information from the SPECTRAL_WINDOW table in the MS file."""
    tb = table()

    try:
        # Open the SPECTRAL_WINDOW table within the MS file
        tb.open(ms_file + '/SPECTRAL_WINDOW')
    except Exception as e:
        print(f"Error opening SPECTRAL_WINDOW table: {e}")
        return

    # Get all column names in the table
    column_names = tb.colnames()

    # Print information for each column
    print("Spectral Window Table Information:")
    for col_name in column_names:
        try:
            # Get the data for each column
            column_data = tb.getcol(col_name)
            
            print(f"\nColumn: {col_name}")
            print(f"Data: {column_data}")

        except Exception as e:
            print(f"Error reading column {col_name}: {e}")

    tb.close()


def print_spectral_window_frame(ms_file):
    tb = table()

    try:
        # Open the SPECTRAL_WINDOW table within the MS file
        tb.open(ms_file + '/SPECTRAL_WINDOW')
    except Exception as e:
        print(f"Error opening SPECTRAL_WINDOW table: {e}")
        return

    # Check the MEAS_FREQ_REF column for the frame type (reference frame)
    if 'MEAS_FREQ_REF' in tb.colnames():
        meas_freq_ref = tb.getcol('MEAS_FREQ_REF')  # Get data from MEAS_FREQ_REF
        print("MEAS_FREQ_REF (Frame Type) information:")

        # Map frame code to frame type
        frame_codes = {
            0: "TOPO (Topocentric)",
            1: "LSRK (Local Standard Rest Kinematic)",
            2: "LSRD (Local Standard Rest Dynamical)",
            3: "BARY (Barycentric)",
            4: "GEO (Geocentric)"
        }

        # Handle unknown frame by assuming LSRK
        frame_type = frame_codes.get(meas_freq_ref[0], "Unknown frame")
        if frame_type == "Unknown frame":
            print(f"Reference frame type: {frame_type} (assuming LSRK)")
            frame_type = "LSRK (Local Standard Rest Kinematic)"  # Default to LSRK
        else:
            print(f"Reference frame type: {frame_type}")
    else:
        print("MEAS_FREQ_REF column not found in SPECTRAL_WINDOW table.")

    tb.close()

    
def get_observation_times(msfile):
    """
    Compute the observation start time, end time, duration, and average time from a Measurement Set (MS).
    
    Args:
        msfile (str): Path to the Measurement Set file.
    
    Returns:
        tuple: (start_time_utc, end_time_utc, avg_time_utc, duration_seconds)
    """
    # Open MS MAIN table
    tb = table(msfile, readonly=True)
    time_data = tb.getcol("TIME")  # Read TIME column (in MJD seconds)
    tb.close()  # Close the table

    # Compute start, end, and average times (convert seconds to MJD)
    start_time_mjd = np.min(time_data) / 86400  
    end_time_mjd = np.max(time_data) / 86400  
    avg_time_mjd = np.mean(time_data) / 86400  

    # Convert MJD to UTC
    start_time_utc = Time(start_time_mjd, format='mjd').iso
    end_time_utc = Time(end_time_mjd, format='mjd').iso
    avg_time_utc = Time(avg_time_mjd, format='mjd').iso
    
    # Compute duration
    duration_seconds = np.max(time_data) - np.min(time_data)

    return start_time_utc, end_time_utc, avg_time_utc, duration_seconds

def get_phase_center_from_ms(ms, field_id):
    #open ms metadata
    msmd.open(ms)
    phase_center = msmd.phasecenter(field_id)

    # Extract RA and Dec in radians
    ra_rad = phase_center['m0']['value']
    dec_rad = phase_center['m1']['value']
    
    # Convert to hms (RA) and dms (Dec) using Astropy
    ra_hms = Angle(ra_rad, unit=u.rad).to_string(unit=u.hourangle, sep=':')
    dec_dms = Angle(dec_rad, unit=u.rad).to_string(unit=u.deg, sep=':')

    print("phase center: ", ra_hms, dec_dms)

    return ra_rad*u.rad, dec_rad*u.rad


def get_askap_location(): #not used here
    """Returns ASKAP location using Astropy's predefined site."""
    askap_location = EarthLocation.of_site('askap')  # Using 'askap' directly
    return askap_location


def get_observation_metadata(msfile):
    """
    Extracts average observation time and ASKAP location from the Measurement Set.
    
    Args:
        msfile (str): Path to the Measurement Set file.
    
    Returns:
        tuple: (avg_obs_time, location)
            - avg_obs_time (Time): Average observation time in UTC.
            - location (EarthLocation): Geocentric coordinates of ASKAP.
    """
    # Open MS MAIN table
    tb = table(msfile, readonly=True)
    time_data = tb.getcol("TIME")  # Read TIME column (in MJD seconds)
    tb.close()  # Close the table

    avg_time_mjd = np.mean(time_data) / 86400
    avg_obs_time = Time(avg_time_mjd, format='mjd')

    print("Average Observation Time (UTC):", avg_obs_time.iso)  # Print in ISO format

    # 🔹 Extract ASKAP Location from ANTENNA Table
    tb.open(msfile + "/ANTENNA")
    antenna_positions = tb.getcol("POSITION")  # X, Y, Z in meters
    tb.close()
    
    # Use the first antenna as the reference
    x, y, z = antenna_positions[:, 0]  
    location = EarthLocation.from_geocentric(x, y, z, unit=u.m)

    return avg_obs_time, location


def get_frequency_from_ms(msfile):
    """Extracts the frequency from the Measurement Set (MS)."""
    tb = table()
    
    # Extract the spectral window ID (assuming it's stored in the SPECTRAL_WINDOW table)
    tb.open(msfile + "/SPECTRAL_WINDOW")
    freq_array = tb.getcol("CHAN_FREQ")#[0]  # Frequency in Hz for the first channel
    tb.close()
    
    # Return the frequency of the first channel as an astropy Quantity
    return freq_array * u.Hz


def get_heliocentric_velocity(msfile, phase_center_coord, obs_time, location, rest_freq):
    """Computes the heliocentric velocity using data from the MS."""
    
    # Compute the heliocentric velocity correction
    heliocentric_corr = phase_center_coord.radial_velocity_correction(kind="heliocentric", 
                                                                    obstime=obs_time, 
                                                                    location=location)
    
    # Get Frequency from MS
    freq_array = get_frequency_from_ms(msfile)

    # Compute Doppler Velocity using astropy.units.doppler_radio (radio convention)
    radio_velocity_equiv = u.doppler_radio(rest_freq)  # Create equivalency
    radio_velocity = freq_array.to(u.km/u.s, equivalencies=radio_velocity_equiv)

    # Compute Heliocentric velocity
    heliocentric_velocity = radio_velocity - heliocentric_corr

    return heliocentric_velocity

# Function to compute VLSRK
def get_vlsrk_velocity(msfile, phase_center_coord, obs_time, location, rest_freq):
    """
    Compute the VLSRK velocity from the frequency of the MS file.
    
    Args:
        msfile (str): The Measurement Set (MS) file path.
        phase_center_coord (SkyCoord): The phase center coordinates of the observation (in celestial coordinates).
        obs_time (str): The observation time (in ISO format, e.g., "2025-02-16T00:00:00").
        location (EarthLocation): The location of the telescope (e.g., ASKAP).
    
    Returns:
        float: The VLSRK velocity in km/s.
    """
    
    # Open the MS file using CASA or relevant package
    # For simplicity, assuming msfile is in a format that you can extract frequency from
    # Example: Using the CASA package (make sure you import necessary functions)
    
    # For now, let's assume the MS frequency is extracted from the MS header directly
    # Replace this with code to read the MS file's frequency, e.g., using CASA's ms tool
    
    # Example frequency from MS file (just an assumption)
    ms_freq = get_frequency_from_ms(msfile)
        
    # Convert observation time to Astropy Time object
    observation_time = Time(obs_time)

    # Get the position of the telescope (e.g., ASKAP location)
    # If a custom location is provided, use that
    if location is None:
        location = get_askap_location()
    
    # Calculate the observer's velocity using the location and time
    observer = location.get_gcrs(obstime=observation_time)
    
    # Extract velocity components using the correct attributes (d_x, d_y, d_z)
    velocity_x = observer.velocity.d_x.to(u.km / u.s)  # Convert to km/s
    velocity_y = observer.velocity.d_y.to(u.km / u.s)  # Convert to km/s
    velocity_z = observer.velocity.d_z.to(u.km / u.s)  # Convert to km/s
    
    # Compute the total velocity magnitude
    observer_velocity_kms = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
    
    # Using the Doppler shift formula (radio convention)
    # Radio velocity (Doppler shift) using the rest frequency
    radio_equiv = u.doppler_radio(rest_freq)
    radio_velocity = ms_freq.to(u.km / u.s, equivalencies=radio_equiv)
    
    # Now calculate the VLSRK based on the observer's velocity (ASKAP) and the Doppler velocity
    vlsrk_velocity = radio_velocity - observer_velocity_kms

    print("observer_velocity_kms: ",  observer_velocity_kms)

    return vlsrk_velocity


def get_velocities(msfile, field_id=0, rest_freq=1.42040575177e9*u.Hz):
    """Computes both VLSRK and heliocentric velocities at the phase center."""
    
    # Get Phase Center RA, Dec
    phase_center_ra, phase_center_dec = get_phase_center_from_ms(msfile, field_id)
    phase_center_coord = SkyCoord(ra=phase_center_ra, dec=phase_center_dec, frame="icrs")

    # Get Observation Time & ASKAP Location from MS
    obs_time, location = get_observation_metadata(msfile)

    # Compute the heliocentric velocity
    heliocentric_velocity = get_heliocentric_velocity(msfile, phase_center_coord, obs_time, location, rest_freq)

    # Compute the VLSRK velocity
    vlsrk_velocity = get_vlsrk_velocity(msfile, phase_center_coord, obs_time, location, rest_freq)

    return heliocentric_velocity, vlsrk_velocity


if __name__ == '__main__':    
    # # Example usage
    path="/home/amarchal/Projects/deconv/examples/data/ASKAP/msl_fixms/scienceData.MS_M345-09A_4/"
    msfile = "scienceData.MS_M345-09A_4.SB68827.MS_M345-09C_4.beam20_SL.dop.1chan.ms"

    # path="/home/amarchal/Projects/deconv/examples/data/MeerKAT/msl_cloud/"
    # msfile="MW-C10_1_chan_-155kms.ms"
    # path="/home/amarchal/Projects/deconv/examples/data/MeerKAT/msl_mw/"
    # msfile="MW-C10_1_MW_chan_-32kms.ms"
    # path="/home/amarchal/Projects/deconv/examples/data/ATCA/rg17/msl/"
    # msfile="rg17.2100_2023-10-06.ms"

    # path="/home/amarchal/Projects/deconv/examples/data/MeerKAT/original/"
    # msfile="MW-C10_1.ms.contsub"

    # Print info header
    print_spectral_window_frame(path+msfile)
    
    #Info time
    start_utc, end_utc, avg_utc, duration = get_observation_times(path+msfile)
    
    print(f"Observation Start Time (UTC): {start_utc}")
    print(f"Observation End Time (UTC): {end_utc}")
    print(f"Average Observation Time (UTC): {avg_utc}")
    print(f"Observation Duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    print("")
    
    # Example Usage
    rest_freq = 1.42040575177e9 * u.Hz  # More precise 21 cm HI line rest frequency
    velcocity = convert_freq_to_velocity(path+msfile, rest_freq) #OK
    heliocentric_velocity, vlsrk_velocity = get_velocities(path+msfile, field_id=0, rest_freq=rest_freq) # field_id is 2 here for interleave C. Don't use

    print("")
    print(f"Heliocentric Velocity: {heliocentric_velocity}")
    print(f"VLSRK Velocity: {vlsrk_velocity}")

    
