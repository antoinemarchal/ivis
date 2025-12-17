import os
import time
import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b

from ivis.logger import logger
from ivis.utils import dutils


def optimize_scipy_lbfgsb(*, model, x0, bounds64, param_shape, max_its,
                          cost_dev, optim_dev, params):
    if optim_dev.type == "cuda":
        logger.info("positivity=True with optim_device on CUDA → falling back to CPU for SciPy L-BFGS-B.")

    def fun_and_grad(x):
        f, g = model.loss(x, shape=param_shape, device=cost_dev, jac=True, **params)
        return float(f), np.ascontiguousarray(g, dtype=np.float64)

    logger.info(f"Starting optimisation: SciPy L-BFGS-B (CPU optimizer), cost on {cost_dev}")
    x_opt, f_opt, info = fmin_l_bfgs_b(
        fun_and_grad, x0, bounds=bounds64,
        m=7, pgtol=1e-8, factr=1e7, maxls=20,
        maxiter=int(max_its), iprint=-1,
    )
    return x_opt.reshape(param_shape)


def optimize_torch_lbfgs(*, model, x_init, dtype, history_size, max_its,
                         cost_dev, optim_dev, params):
    for dev in [cost_dev, optim_dev]:
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

    def _mem_info() -> str:
        mem_bits = []
        if cost_dev.type == "cuda":
            torch.cuda.synchronize(cost_dev)
            mem_bits.append(dutils.gpu_mem_str(cost_dev))
        if optim_dev.type == "cuda" and (optim_dev.index != cost_dev.index):
            torch.cuda.synchronize(optim_dev)
            mem_bits.append(dutils.gpu_mem_str(optim_dev))
        return " | ".join(mem_bits)

    iter_k = {"k": 0}  # counts closure calls (true expensive evals)

    def closure():
        iter_k["k"] += 1
        opt.zero_grad(set_to_none=True)

        if cost_dev == optim_dev:
            loss = model.objective(x_param, device=cost_dev, **params)
            if x_param.grad is None:
                raise RuntimeError("objective() did not produce gradients on x_param.")
            mem_info = _mem_info()
        else:
            x_for_cost = x_param.detach().to(cost_dev).requires_grad_(True)
            loss = model.objective(x_for_cost, device=cost_dev, **params)
            if x_for_cost.grad is None:
                raise RuntimeError("objective() did not produce gradients on x_for_cost.")
            mem_info = _mem_info()
            x_param.grad = x_for_cost.grad.to(optim_dev)
            del x_for_cost

        logger.info(
            f"[PID {os.getpid()}] [Iter {iter_k['k']}/{max_its}] "
            f"Iter cost: {float(loss.detach().cpu()):.6e} "
            f"(optim_dev={optim_dev}, cost_dev={cost_dev})"
            + (f" | {mem_info}" if mem_info else "")
        )
        return loss

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    t0 = time.perf_counter()
    final_loss = opt.step(closure)
    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    elapsed = time.perf_counter() - t0

    if optim_dev == cost_dev:
        end_mem_info = dutils.gpu_mem_str(cost_dev) if cost_dev.type == "cuda" else ""
    else:
        end_mem_info = " | ".join(
            [dutils.gpu_mem_str(d) for d in (cost_dev, optim_dev) if d.type == "cuda"]
        )

    logger.info(
        f"[Timing] LBFGS (optim_dev={optim_dev}, cost_dev={cost_dev}) "
        f"took {elapsed:.2f} s; final loss={float(final_loss):.6g}; "
        f"closure_calls={iter_k['k']}"
        + (f" | {end_mem_info}" if end_mem_info else "")
    )

    return x_param.detach().cpu().numpy()

