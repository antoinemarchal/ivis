import torch

def format_input_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Format an input tensor for PyTorch's grid_sample.

    Ensures shape is (N=1, C=1, H, W), as required by grid_sample.

    Parameters
    ----------
    input_tensor : torch.Tensor
        A 2D, 3D, or already 4D tensor.

    Returns
    -------
    formatted_tensor : torch.Tensor
        Tensor reshaped for use with grid_sample.
    """
    if input_tensor.dim() == 2:
        return input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        return input_tensor.unsqueeze(0)
    return input_tensor
