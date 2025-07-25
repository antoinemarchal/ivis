import glob
import os
import shutil
from casacore.tables import table, taql
import casatools

def concatenate_ms(ms_list, output_ms, overwrite=False):
    """Concatenates multiple Measurement Sets (MS) into a single MS using TAQL.
    
    Args:
        ms_list (list): List of MS files to concatenate.
        output_ms (str): The name of the output MS file.
        overwrite (bool): Whether to overwrite the existing output MS file if it exists.
    """
    
    # Check if the output file exists and handle overwriting or skipping
    if os.path.exists(output_ms):
        if overwrite:
            print(f"Output file {output_ms} already exists. Overwriting...")
            # If it's a directory, delete it forcefully
            if os.path.isdir(output_ms):
                print(f"Output path {output_ms} is a directory. Removing directory and its contents...")
                shutil.rmtree(output_ms)  # Forcefully remove the directory and its contents
            else:
                # If it's a file, we don't need to do anything special for deletion
                print(f"Output path {output_ms} is a file. Removing file...")
                os.remove(output_ms)  # This deletes the existing MS file
        else:
            print(f"Output file {output_ms} already exists, and overwrite=False. Skipping concatenation.")
            return  # Skip concatenation if overwrite is False and the file exists
    
    # Concatenate the MS files
    if len(ms_list) < 2:
        raise ValueError("You need at least two MS files to concatenate.")

    # Copy the first MS as the base for the output
    print(f"Creating {output_ms} from {ms_list[0]}")
    table(ms_list[0]).copy(output_ms, deep=True)

    # Use TAQL to append the remaining MS files
    for ms in ms_list[1:]:
        print(f"Appending {ms} to {output_ms} using TAQL")
        taql(f"INSERT INTO {output_ms} SELECT * FROM {ms}")

    print(f"Concatenation completed: {output_ms}")


def normalize_ms_if_needed(ms_list):
    """Normalize MS files only if they have different column structures."""
    tb = casatools.table()

    # Extract column sets for each MS file
    column_sets = {}
    for ms in ms_list:
        tb.open(ms)
        column_sets[ms] = set(tb.colnames())
        tb.close()

    # Check if all MS files have the same columns
    common_columns = set.intersection(*column_sets.values())
    all_same = all(cols == common_columns for cols in column_sets.values())

    if all_same:
        print("All MS files have the same columns. No normalization needed.")
        return ms_list  # Return original MS files

    print("Column mismatch detected. Normalizing MS files...")

    normalized_ms_list = []
    for ms in ms_list:
        output_ms = ms.replace(".ms", "_normalized.ms")  # Name for normalized file
        print(f"Creating normalized MS: {output_ms}")

        tb.open(ms)
        tb.copy(output_ms, deep=True, valuecopy=True)
        tb.close()

        # Drop extra columns
        tb.open(output_ms, nomodify=False)
        for col in tb.colnames():
            if col not in common_columns:
                print(f"Dropping column {col} from {output_ms}")
                tb.removecols(col)
        tb.close()

        normalized_ms_list.append(output_ms)

    return normalized_ms_list  # Return the normalized file list

    
def normalize_ms_in_memory(ms_list):
    """Normalize MS files in memory without writing them to disk."""
    tb = casatools.table()

    # Extract column sets for each MS file
    column_sets = {}
    for ms in ms_list:
        tb.open(ms)
        column_sets[ms] = set(tb.colnames())
        tb.close()

    # Find the common columns
    common_columns = set.intersection(*column_sets.values())

    # Check if all MS files have the same columns
    all_same = all(cols == common_columns for cols in column_sets.values())

    if all_same:
        print("All MS files have the same columns. No normalization needed.")
        return ms_list  # Return original MS files if they are already normalized

    print("Column mismatch detected. Normalizing MS files in memory...")

    normalized_ms_list = []
    # Remove extra columns in memory (without saving to disk)
    for ms in ms_list:
        tb.open(ms, nomodify=False)
        columns_to_remove = [col for col in tb.colnames() if col not in common_columns]
        for col in columns_to_remove:
            print(f"Dropping column {col} from {ms}")
            tb.removecols(col)
        normalized_ms_list.append(ms)  # Keep the original MS in memory (no new file written)
        tb.close()

    # Return the modified list in memory
    return normalized_ms_list


if __name__ == '__main__':    
    # Example usage
    mspath="/home/amarchal/Projects/deconv/examples/data/ASKAP/msl_fixms/"
    msl = sorted(glob.glob(mspath+"*.ms"))

    subdirs = sorted([d for d in os.listdir(mspath) if os.path.isdir(os.path.join(mspath, d))])
    
    # Create a 2D list where each row corresponds to one subdirectory
    msl_2d = [sorted(glob.glob(os.path.join(mspath, subdir, "*.ms"))) for subdir in subdirs]
    
    # Check the structure
    print("number of Subdir: ", len(msl_2d))  # Should print 4
    for i, sublist in enumerate(msl_2d):
        print(f"Subdir {i}: {len(sublist)} files")  # Should print 108 for each if correct

    # Transpose the list to group nth elements from each subdirectory
    msl_transposed = list(zip(*msl_2d))
    
    # Concatanate
    output_dir = "/home/amarchal/Projects/deconv/examples/data/ASKAP/msl_fixms_concat/"  # Change this to your desired output directory
    
    for ms_list in msl_transposed:
        normalized_ms_list = normalize_ms_in_memory(ms_list)  # Normalize MS files if needed, in memory
        # ms_list = normalize_ms_if_needed(ms_list)  # Normalize only if needed
        output_ms = os.path.join(output_dir, os.path.basename(ms_list[0]))  # Keep original name in output dir
        concatenate_ms(ms_list, output_ms, overwrite=True)
    
