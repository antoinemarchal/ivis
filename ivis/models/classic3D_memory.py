import torch
from torch.fft import fft2 as tfft2

from ivis.models.classic3D import Classic3D
from ivis.models.operators import forward_beam, resolve_pb_grid_lists
from ivis.models.utils.gpu import print_gpu_memory


class Classic3DMemory(Classic3D):
    """
    Memory-streaming Classic3D variant.

    The standard Classic3D objective accumulates a full autograd graph before
    one backward pass. This variant backpropagates each channel/beam loss block
    immediately, reducing peak memory for many-channel PyTorch optimizations.
    """

    def objective(
        self,
        x,
        vis_data,
        device,
        primary_beam_list=None,
        primary_beam=None,
        pb_list=None,
        grid_list=None,
        pb=None,
        grid_array=None,
        cell_size=None,
        fftsd=None,
        fftbeam=None,
        tapper=None,
        lambda_sd=0.0,
        lambda_pos=0.0,
        fftkernel=None,
        beam_workers=4,
        verbose=False,
        **_,
    ):
        x.requires_grad_(True)
        if x.is_leaf and x.grad is not None:
            x.grad.zero_()

        primary_beam_list, grid_list = resolve_pb_grid_lists(
            vis_data,
            pb_list=primary_beam_list if primary_beam_list is not None else pb_list,
            grid_list=grid_list,
            pb=primary_beam if primary_beam is not None else pb,
            grid_array=grid_array,
        )

        loss_value = torch.zeros((), dtype=x.dtype, device=device)

        for c, b, I, sI, uu, vv, ww in vis_data.iter_chan_beam_I():
            model_vis = forward_beam(
                x2d=x[c],
                primary_beam=primary_beam_list[b],
                grid=grid_list[b],
                uu=uu,
                vv=vv,
                ww=ww,
                cell_size=cell_size,
                device=device,
            )

            I_use = I.conj() if self.conj_data else I
            vis_real = torch.from_numpy(I_use.real).to(device)
            vis_imag = torch.from_numpy(I_use.imag).to(device)
            sig = torch.from_numpy(sI).to(device)

            residual_real = (model_vis.real - vis_real) / sig
            residual_imag = (model_vis.imag - vis_imag) / sig
            J = torch.sum(residual_real**2 + residual_imag**2)
            block_loss = 0.5 * J
            block_loss.backward()
            loss_value = loss_value + block_loss.detach()

            if verbose:
                print_gpu_memory(device)

            del model_vis, vis_real, vis_imag, sig, residual_real, residual_imag, J, block_loss

        if lambda_sd > 0.0 and fftsd is not None:
            fftsd_t = torch.from_numpy(fftsd).to(device)
            fftbeam_t = torch.from_numpy(fftbeam).to(device)
            tapper_t = torch.from_numpy(tapper).to(device)
            for c in range(x.shape[0]):
                fftsd_c = fftsd_t[c] if fftsd_t.ndim == x.ndim else fftsd_t
                fftbeam_c = fftbeam_t[c] if fftbeam_t.ndim == x.ndim else fftbeam_t
                tapper_c = tapper_t[c] if tapper_t.ndim == x.ndim else tapper_t
                xfft2 = tfft2(x[c] * tapper_c)
                model_sd = (cell_size**2) * xfft2 * fftbeam_c
                Lsd = 0.5 * (
                    torch.nansum((model_sd.real - fftsd_c.real) ** 2)
                    + torch.nansum((model_sd.imag - fftsd_c.imag) ** 2)
                ) * lambda_sd
                Lsd.backward()
                loss_value = loss_value + Lsd.detach()
                del fftsd_c, fftbeam_c, tapper_c, xfft2, model_sd, Lsd
            del fftsd_t, fftbeam_t, tapper_t

        if self.lambda_r > 0.0 and fftkernel is not None:
            tapper_t = torch.from_numpy(tapper).to(device)
            fftkernel_t = torch.from_numpy(fftkernel).to(device)
            for c in range(x.shape[0]):
                fftkernel_c = fftkernel_t[c] if fftkernel_t.ndim == x.ndim else fftkernel_t
                tapper_c = tapper_t[c] if tapper_t.ndim == x.ndim else tapper_t
                xfft2 = tfft2(x[c] * tapper_c)
                conv = (cell_size**2) * xfft2 * fftkernel_c
                Lr = 0.5 * torch.nansum(torch.abs(conv) ** 2) * self.lambda_r
                Lr.backward()
                loss_value = loss_value + Lr.detach()
                del fftkernel_c, tapper_c, xfft2, conv, Lr
            del tapper_t, fftkernel_t

        return loss_value
