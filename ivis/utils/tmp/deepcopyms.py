import shutil

def deepcopy_ms(pathin, pathout):
    """
    Copies the Measurement Set (MS) from pathin to pathout.
    
    Parameters:
        pathin (str): Path to the source MS directory or file.
        pathout (str): Path where the MS should be copied.
    """
    try:
        shutil.copytree(pathin, pathout)
        print(f"MS file successfully copied from {pathin} to {pathout}.")
    except Exception as e:
        print(f"Error copying MS file: {e}")

if __name__ == '__main__':
    # Path to Measurement Set
    ms_path = "/home/amarchal/Projects/deconv/examples/data/MeerKAT/original/MW-C10_5.ms"
    output_ms= "/home/amarchal/Projects/deconv/examples/data/MeerKAT/original_contsub/MW-C10_5.ms"

    deepcopy_ms(ms_path, output_ms)

    
