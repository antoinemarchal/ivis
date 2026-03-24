import torch
import pytorch_finufft


def forward_nufft(x_pb, u_radpix, v_radpix, cell_size):
    points = torch.stack([-v_radpix, u_radpix], dim=0)
    c = x_pb.to(torch.complex64)
    return (cell_size**2) * pytorch_finufft.functional.finufft_type2(
        points, c, isign=1, modeord=0
    )


def backward_nufft(y, pb_shape, u_radpix, v_radpix, cell_size):
    points = torch.stack([-v_radpix, u_radpix], dim=0)
    return (cell_size**2) * pytorch_finufft.functional.finufft_type1(
        points, y, pb_shape, isign=-1, modeord=0
    )
