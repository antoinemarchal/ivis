import os
import time
import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b
import torch.nn.functional as F

from ivis.logger import logger
from ivis.utils import dutils

def optimize_scipy_lbfgsb(*, model, x0, bounds64, param_shape, max_its,
                          cost_dev, optim_dev, params):
    if optim_dev.type == "cuda":
        logger.info("positivity=True with optim_device on CUDA --> falling back to CPU for SciPy L-BFGS-B.")

    def fun_and_grad(x):
        f, g = model.loss(x, shape=param_shape, device=cost_dev, jac=True, **params)
        return float(f), np.ascontiguousarray(g, dtype=np.float64)

    logger.info(f"Starting optimisation: SciPy L-BFGS-B (CPU optimizer), cost on {cost_dev}")
    x_opt, f_opt, info = fmin_l_bfgs_b(
        fun_and_grad, x0, bounds=bounds64,
        m=7, pgtol=1e-8, factr=1e7, maxls=20,
        maxiter=int(max_its), iprint=25,
    )
    return x_opt.reshape(param_shape)


def optimize_torch_lbfgs(*, model, x_init, dtype, history_size, max_its,
                           cost_dev, optim_dev, params,
                           lr=1.0,
                           line_search_fn="strong_wolfe",
                           positivity=False,
):
    """
    PyTorch LBFGS optimizer with optional cross-device cost evaluation.

    Assumptions:
    - model.objective(x, device=..., **params) calls backward() internally
      and populates x.grad (side-effect gradients).
    - We drive iterations explicitly (outer loop), so you get exactly max_its steps.

    Returns
    -------
    x_opt : np.ndarray
    """

    # Reset CUDA peak stats (unchanged behavior)
    for dev in (cost_dev, optim_dev):
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(idx)

    logger.info(
        f"Starting optimisation: PyTorch LBFGS on {optim_dev}; cost on {cost_dev} | "
        f"positivity(project)={bool(positivity)}"
    )

    x_param = torch.tensor(x_init, dtype=dtype, device=optim_dev, requires_grad=True)

    # One inner LBFGS step per outer iteration (predictable iteration count)
    opt = torch.optim.LBFGS(
        [x_param],
        lr=float(lr),
        max_iter=1,
        history_size=int(history_size),
        line_search_fn=line_search_fn,
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )

    def mem_info_str() -> str:
        bits = []
        if cost_dev.type == "cuda":
            torch.cuda.synchronize(cost_dev)
            bits.append(dutils.gpu_mem_str(cost_dev))
        if optim_dev.type == "cuda" and (optim_dev.index != getattr(cost_dev, "index", None)):
            torch.cuda.synchronize(optim_dev)
            bits.append(dutils.gpu_mem_str(optim_dev))
        return " | ".join(bits)

    def eval_loss_and_grad():
        """
        Returns (loss_tensor, mem_info_string).
        Ensures x_param.grad is populated on optim_dev.
        """
        if cost_dev == optim_dev:
            loss = model.objective(x_param, device=cost_dev, **params)
            if x_param.grad is None:
                raise RuntimeError("objective() did not produce gradients on x_param.")
            return loss, mem_info_str()

        # Cross-device: compute loss+grad on cost_dev then copy grad back
        x_for_cost = x_param.detach().to(cost_dev).requires_grad_(True)
        loss = model.objective(x_for_cost, device=cost_dev, **params)
        if x_for_cost.grad is None:
            raise RuntimeError("objective() did not produce gradients on x_for_cost.")

        mem = mem_info_str()  # log BEFORE freeing/copying
        x_param.grad = x_for_cost.grad.to(optim_dev)
        del x_for_cost
        return loss, mem

    def closure():
        opt.zero_grad(set_to_none=True)
        loss, _ = eval_loss_and_grad()
        return loss  # objective() already called backward()

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    t0 = time.perf_counter()

    final_loss = None
    for it in range(int(max_its)):
        loss = opt.step(closure)
        final_loss = loss

        # Optional projection step (positivity)
        if positivity:
            with torch.no_grad():
                x_param.clamp_(min=0.0)

        # Logging (once per outer iter)
        mem = mem_info_str()
        logger.info(
            f"[PID {os.getpid()}] Iter {it+1}/{max_its} | "
            f"cost: {float(loss.detach().cpu()):.6e} "
            f"(optim_dev={optim_dev}, cost_dev={cost_dev})"
            + (f" | {mem}" if mem else "")
        )

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    elapsed = time.perf_counter() - t0

    # End-of-run mem info
    if optim_dev == cost_dev:
        end_mem_info = dutils.gpu_mem_str(cost_dev) if cost_dev.type == "cuda" else ""
    else:
        end_mem_info = " | ".join(
            dutils.gpu_mem_str(d) for d in (cost_dev, optim_dev) if d.type == "cuda"
        )

    logger.info(
        f"[Timing] LBFGS (optim_dev={optim_dev}, cost_dev={cost_dev}) "
        f"took {elapsed:.2f} s; final loss={float(final_loss):.6g}"
        + (f" | {end_mem_info}" if end_mem_info else "")
    )

    return x_param.detach().cpu().numpy()


def optimize_torch_lbfgs_2(*, model, x_init, dtype, history_size, max_its,
                         cost_dev, optim_dev, params):
    """
    PyTorch LBFGS optimizer with optional cross-device cost evaluation.

    Notes:
    - Prints "Iter i/max_its" using PyTorch LBFGS internal iteration counter
      opt.state[x_param]["n_iter"] (counts true LBFGS iterations, not closure calls).
    - Keeps your original log style (includes optim_dev/cost_dev + optional GPU mem info).
    """
    # Reset CUDA peak stats (unchanged behavior)
    for dev in (cost_dev, optim_dev):
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(idx)

    logger.info(
        f"Starting optimisation: PyTorch LBFGS on {optim_dev} (unconstrained); "
        f"cost on {cost_dev}"
    )

    x_param = torch.tensor(x_init, dtype=dtype, device=optim_dev, requires_grad=True)

    opt = torch.optim.LBFGS(
        [x_param],
        lr=1.0,
        max_iter=int(max_its),
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-8,
        tolerance_change=0.0,
    )

    # Grab the optimizer state dict for this parameter (LBFGS stores n_iter here)
    state = opt.state[x_param]

    def mem_info_str() -> str:
        bits = []
        if cost_dev.type == "cuda":
            torch.cuda.synchronize(cost_dev)
            bits.append(dutils.gpu_mem_str(cost_dev))
        if optim_dev.type == "cuda" and (optim_dev.index != getattr(cost_dev, "index", None)):
            torch.cuda.synchronize(optim_dev)
            bits.append(dutils.gpu_mem_str(optim_dev))
        return " | ".join(bits)

    def eval_loss_and_grad():
        """
        Returns (loss_tensor, mem_info_string).
        Ensures x_param.grad is populated on optim_dev.
        """
        if cost_dev == optim_dev:
            loss = model.objective(x_param, device=cost_dev, **params)
            if x_param.grad is None:
                raise RuntimeError("objective() did not produce gradients on x_param.")
            return loss, mem_info_str()

        # Cross-device: compute loss+grad on cost_dev then copy grad back
        x_for_cost = x_param.detach().to(cost_dev).requires_grad_(True)
        loss = model.objective(x_for_cost, device=cost_dev, **params)
        if x_for_cost.grad is None:
            raise RuntimeError("objective() did not produce gradients on x_for_cost.")

        mem = mem_info_str()  # log BEFORE freeing/copying
        x_param.grad = x_for_cost.grad.to(optim_dev)
        del x_for_cost
        return loss, mem

    # Optional: only log once per true LBFGS iteration (closure can be called many times)
    last_logged = {"iter": -1}

    def closure():
        opt.zero_grad(set_to_none=True)
        loss, mem_info = eval_loss_and_grad()

        # True LBFGS iteration counter (increments once per step, not per closure call)
        n_iter = state.get("n_iter", 0)

        if n_iter != last_logged["iter"]:
            last_logged["iter"] = n_iter
            logger.info(
                f"[PID {os.getpid()}] Iter {n_iter}/{max_its} | "
                f"cost: {float(loss.detach().cpu()):.6e} "
                f"(optim_dev={optim_dev}, cost_dev={cost_dev})"
                + (f" | {mem_info}" if mem_info else "")
            )

        return loss  # objective() already called backward()

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    t0 = time.perf_counter()
    final_loss = opt.step(closure)
    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    elapsed = time.perf_counter() - t0

    # End-of-run mem info
    if optim_dev == cost_dev:
        end_mem_info = dutils.gpu_mem_str(cost_dev) if cost_dev.type == "cuda" else ""
    else:
        end_mem_info = " | ".join(
            dutils.gpu_mem_str(d) for d in (cost_dev, optim_dev) if d.type == "cuda"
        )

    logger.info(
        f"[Timing] LBFGS (optim_dev={optim_dev}, cost_dev={cost_dev}) "
        f"took {elapsed:.2f} s; final loss={float(final_loss):.6g}"
        + (f" | {end_mem_info}" if end_mem_info else "")
    )

    return x_param.detach().cpu().numpy()

# def optimize_torch_lbfgs(*, model, x_init, dtype, history_size, max_its,
#                          cost_dev, optim_dev, params):
#     for dev in [cost_dev, optim_dev]:
#         if dev.type == "cuda":
#             idx = dev.index if dev.index is not None else torch.cuda.current_device()
#             torch.cuda.reset_peak_memory_stats(idx)

#     logger.info(
#         f"Starting optimisation: PyTorch LBFGS on {optim_dev} (unconstrained); "
#         f"cost on {cost_dev}"
#     )

#     x_param = torch.tensor(x_init, dtype=dtype, device=optim_dev, requires_grad=True)

#     opt = torch.optim.LBFGS(
#         [x_param],
#         lr=1.0,
#         max_iter=int(max_its),
#         history_size=history_size,
#         line_search_fn="strong_wolfe",
#         tolerance_grad=1e-8,
#         tolerance_change=0.0,
#     )

#     def _mem_info() -> str:
#         mem_bits = []
#         if cost_dev.type == "cuda":
#             torch.cuda.synchronize(cost_dev)
#             mem_bits.append(dutils.gpu_mem_str(cost_dev))
#         if optim_dev.type == "cuda" and (optim_dev.index != cost_dev.index):
#             torch.cuda.synchronize(optim_dev)
#             mem_bits.append(dutils.gpu_mem_str(optim_dev))
#         return " | ".join(mem_bits)

#     def closure():
#         opt.zero_grad(set_to_none=True)

#         if cost_dev == optim_dev:
#             loss = model.objective(x_param, device=cost_dev, **params)
#             if x_param.grad is None:
#                 raise RuntimeError("objective() did not produce gradients on x_param.")
#             mem_info = _mem_info()
#         else:
#             x_for_cost = x_param.detach().to(cost_dev).requires_grad_(True)
#             loss = model.objective(x_for_cost, device=cost_dev, **params)
#             if x_for_cost.grad is None:
#                 raise RuntimeError("objective() did not produce gradients on x_for_cost.")
#             mem_info = _mem_info()
#             x_param.grad = x_for_cost.grad.to(optim_dev)
#             del x_for_cost

#         logger.info(
#             f"[PID {os.getpid()}] Iter cost: {float(loss.detach().cpu()):.6e} "
#             f"(optim_dev={optim_dev}, cost_dev={cost_dev})"
#             + (f" | {mem_info}" if mem_info else "")
#         )
#         return loss

#     if cost_dev.type == "cuda":
#         torch.cuda.synchronize(cost_dev)
#     t0 = time.perf_counter()
#     final_loss = opt.step(closure)
#     if cost_dev.type == "cuda":
#         torch.cuda.synchronize(cost_dev)
#     elapsed = time.perf_counter() - t0

#     if optim_dev == cost_dev:
#         end_mem_info = dutils.gpu_mem_str(cost_dev) if cost_dev.type == "cuda" else ""
#     else:
#         end_mem_info = " | ".join(
#             [dutils.gpu_mem_str(d) for d in (cost_dev, optim_dev) if d.type == "cuda"]
#         )

#     logger.info(
#         f"[Timing] LBFGS (optim_dev={optim_dev}, cost_dev={cost_dev}) "
#         f"took {elapsed:.2f} s; final loss={float(final_loss):.6g}"
#         + (f" | {end_mem_info}" if end_mem_info else "")
#     )

#     return x_param.detach().cpu().numpy()
