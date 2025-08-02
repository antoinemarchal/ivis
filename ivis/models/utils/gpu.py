import torch

def print_gpu_memory(device="cuda:0"):
    """
    Print GPU memory usage (allocated, reserved, total) for the specified device.

    Parameters
    ----------
    device : str or torch.device
        CUDA device string, e.g., "cuda:0".
    """
    device = torch.device(device)
    allocated = torch.cuda.memory_allocated(device) / 1e6
    reserved = torch.cuda.memory_reserved(device) / 1e6
    total_mem = torch.cuda.get_device_properties(device).total_memory / 1e6
    print(f"[{device}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Total: {total_mem:.2f} MB")
