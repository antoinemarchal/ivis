import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, Angle
from astropy import constants as const
from astropy.time import Time
import astropy.units as u
from casatools import table, msmetadata
from casatools import msmetadata, measures
import casatools
from astropy.constants import c

from deconv import logger

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

    return velocities.to(u.km/u.s)


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


def get_spectral_window_frame(ms_file):
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
 
        # Map frame code to frame type
        frame_codes = {
            0: "TOPO",# (Topocentric)",
            1: "LSRK",# (Local Standard Rest Kinematic)",
            2: "LSRD",# (Local Standard Rest Dynamical)",
            3: "BARY",# (Barycentric)",
            4: "GEO"# (Geocentric)"
        }

        # Handle unknown frame by assuming LSRK
        frame_type = frame_codes.get(meas_freq_ref[0], "Unknown frame")
        if frame_type == "Unknown frame":
            logger.info(f"Reference frame type: {frame_type} (assuming 0:TOPO)")
            frame_type = "TOPO"
        else:
            logger.info(f"Reference frame type: {frame_type}")
    else:
        logger.warning("MEAS_FREQ_REF column not found in SPECTRAL_WINDOW table.")

    tb.close()

    return frame_type

    
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

    # print("phase center: ", ra_hms, dec_dms)

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


def calculate_velocity(ms_file, rest_freq, field_id=0):
    """
    Calls get_spectral_window_frame to determine the frame type and computes the corresponding velocity.
    Returns the LSRK velocity if frame type is "LSRK", or computes the velocity assuming "TOPO" frame.

    Parameters:
    ms_file (str): Path to the Measurement Set (MS) file.
    rest_freq (Quantity): Rest frequency of the spectral line as an astropy Quantity (e.g., `rest_freq = 1.4e9 * u.Hz`).
    field_id (int, optional): Field ID to extract phase center. Default is 0.

    Returns:
    np.ndarray: A 1D array of velocities in km/s.
    """
    frame_type = get_spectral_window_frame(ms_file)  # Get frame type from SPECTRAL_WINDOW table

    if frame_type == "LSRK":
        # If frame type is LSRK, use the convert_freq_to_velocity function
        return convert_freq_to_velocity(ms_file, rest_freq)
    
    elif frame_type == "TOPO":
        # If frame type is TOPO, use the compute_vlsrk function
        msmd = msmetadata()
        me = measures()
        msmd.open(ms_file)

        # Extract telescope name
        telescope = msmd.observatorynames()[0]  # Get first observatory name

        # Use existing function to get phase center
        ra, dec = get_phase_center_from_ms(ms_file, field_id)

        # Use existing function to get observation time
        _, _, avg_time_utc, _ = get_observation_times(ms_file)
        obs_time = Time(avg_time_utc, format='iso', scale='utc')

        # Get telescope location using measures
        obs_pos = me.observatory(telescope)
        observatory = EarthLocation.from_geocentric(obs_pos['m0']['value'] * u.m,
                                                    obs_pos['m1']['value'] * u.m,
                                                    obs_pos['m2']['value'] * u.m)

        target = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')

        # Compute velocity correction
        v_corr = target.radial_velocity_correction(kind='barycentric', obstime=obs_time, location=observatory).to(u.m / u.s)

        # Ensure rest_freq is a Quantity (if not already)
        if not isinstance(rest_freq, u.Quantity):
            rest_freq = rest_freq * u.Hz  # Convert to Quantity if needed

        vlsrk_list = []  # List to hold velocities for all channels

        # Iterate over spectral windows
        for spw in msmd.spwsforfield(msmd.fieldsforname(msmd.fieldnames()[0])[0]):
            chan_freqs = msmd.chanfreqs(spw) * u.Hz  # Extract channel frequencies

            # Make sure chan_freqs are Quantity objects
            if not isinstance(chan_freqs, u.Quantity):
                chan_freqs = chan_freqs * u.Hz

            # Compute velocity using radio convention (LSRK)
            v_lsrk = c * (rest_freq - chan_freqs) / rest_freq + v_corr  # Correct velocity formula

            print(v_corr.to(u.km/u.s))

            # Convert to km/s by dividing by 1000 (and keeping the astropy Quantity format)
            v_lsrk_kms = v_lsrk.to(u.km / u.s)  # Convert velocity to km/s

            # Append the velocities to the list
            vlsrk_list.extend(v_lsrk_kms.value)  # .value extracts the underlying numpy array

        msmd.close()

        # Return the velocities as a numpy array with astropy Quantity in km/s
        return np.array(vlsrk_list) * u.km / u.s  # Return as Quantity in km/s

    else:
        print(f"Unsupported frame type: {frame_type}")
        return None


def calculate_velocity_for_single_freq(ms_file, freq, rest_freq, field_id=0):
    """
    Computes the velocity for a single frequency using the radio convention and considering the frame type.

    Parameters:
    ms_file (str): Path to the Measurement Set (MS) file.
    freq (Quantity): The observed frequency in Hz as an astropy Quantity (e.g., `freq = 1.4e9 * u.Hz`).
    rest_freq (Quantity): Rest frequency of the spectral line as an astropy Quantity (e.g., `rest_freq = 1.4e9 * u.Hz`).
    field_id (int, optional): Field ID to extract phase center. Default is 0.

    Returns:
    Quantity: The velocity in km/s (astropy Quantity).
    """
    frame_type = get_spectral_window_frame(ms_file)  # Get frame type from SPECTRAL_WINDOW table

    if frame_type == "LSRK":
        # If frame type is LSRK, use the convert_freq_to_velocity function
        return convert_freq_to_velocity(ms_file, rest_freq)
    
    elif frame_type == "TOPO":
        # If frame type is TOPO, calculate velocity manually using observed frequency and radio convention
        msmd = msmetadata()
        me = measures()
        msmd.open(ms_file)

        # Extract telescope name
        telescope = msmd.observatorynames()[0]  # Get first observatory name

        # Use existing function to get phase center
        ra, dec = get_phase_center_from_ms(ms_file, field_id)

        # Use existing function to get observation time
        _, _, avg_time_utc, _ = get_observation_times(ms_file)
        obs_time = Time(avg_time_utc, format='iso', scale='utc')

        # Get telescope location using measures
        obs_pos = me.observatory(telescope)
        observatory = EarthLocation.from_geocentric(obs_pos['m0']['value'] * u.m,
                                                    obs_pos['m1']['value'] * u.m,
                                                    obs_pos['m2']['value'] * u.m)

        target = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')

        # Compute velocity correction
        v_corr = target.radial_velocity_correction(kind='barycentric', obstime=obs_time, location=observatory).to(u.m / u.s)

        # Ensure rest_freq is a Quantity (if not already)
        if not isinstance(rest_freq, u.Quantity):
            rest_freq = rest_freq * u.Hz  # Convert to Quantity if needed

        # Calculate the velocity for the given observed frequency using the radio convention
        velocity = c * (rest_freq - freq) / rest_freq + v_corr  # Correct velocity formula

        # Convert to km/s
        velocity_kms = velocity.to(u.km / u.s)

        msmd.close()

        # Return the velocity as an astropy Quantity in km/s
        return velocity_kms

    else:
        print(f"Unsupported frame type: {frame_type}")
        return None


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

    path="/home/amarchal/Projects/deconv/examples/data/MeerKAT/original/"
    msfile="MW-C10_2.ms.contsub"

    # Print info header
    frame_type = get_spectral_window_frame(path+msfile)

    #Info time
    start_utc, end_utc, avg_utc, duration = get_observation_times(path+msfile)
    
    print(f"Observation Start Time (UTC): {start_utc}")
    print(f"Observation End Time (UTC): {end_utc}")
    print(f"Average Observation Time (UTC): {avg_utc}")
    print(f"Observation Duration: {duration:.2f} seconds ({duration/3600:.2f} hours)")
    print("")
    
    # Example Usage
    rest_freq = 1.42040575177e9 * u.Hz  # More precise 21 cm HI line rest frequency    
    vlsrk = calculate_velocity(path+msfile, rest_freq)

    
