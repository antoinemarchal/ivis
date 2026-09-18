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


def optimize_torch_cg(*, model, x_init, dtype, max_its, cost_dev, optim_dev, params,
                      tol=1e-6, atol=0.0):
    if not hasattr(model, "apply_normal_operator") or not hasattr(model, "quadratic_rhs"):
        raise TypeError("CG solver requires model.apply_normal_operator() and model.quadratic_rhs().")

    if optim_dev != cost_dev:
        logger.info(f"CG uses the cost device for operator applications; solving on {cost_dev}.")

    logger.info(f"Starting optimisation: Conjugate Gradient on {cost_dev}")

    x = torch.tensor(x_init, dtype=dtype, device=cost_dev)
    rhs = model.quadratic_rhs(x_shape=tuple(x.shape), device=cost_dev, **params).to(dtype)

    def apply_a(v):
        return model.apply_normal_operator(v, device=cost_dev, **params).to(dtype)

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    t0 = time.perf_counter()

    r = rhs - apply_a(x)
    p = r.clone()
    rr = torch.sum(r * r)
    rhs_norm = torch.linalg.vector_norm(rhs).item()
    threshold = max(float(atol), float(tol) * rhs_norm)

    iter_count = 0
    for k in range(1, int(max_its) + 1):
        Ap = apply_a(p)
        denom = torch.sum(p * Ap)
        denom_value = float(denom.detach().cpu())
        if abs(denom_value) < 1e-30:
            logger.warning(f"[CG] Breakdown at iter {k}: p^T A p is too small ({denom_value:.3e}).")
            iter_count = k - 1
            break

        alpha = rr / denom
        x = x + alpha * p
        r = r - alpha * Ap

        res_norm = torch.linalg.vector_norm(r).item()
        logger.info(
            f"[PID {os.getpid()}] [CG Iter {k}/{max_its}] residual={res_norm:.6e}"
        )
        iter_count = k
        if res_norm <= threshold:
            break

        rr_new = torch.sum(r * r)
        beta = rr_new / rr
        p = r + beta * p
        rr = rr_new

    final_loss = float(model.objective(x, device=cost_dev, **params).detach().cpu())
    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    elapsed = time.perf_counter() - t0

    end_mem_info = dutils.gpu_mem_str(cost_dev) if cost_dev.type == "cuda" else ""
    logger.info(
        f"[Timing] CG (cost_dev={cost_dev}) took {elapsed:.2f} s; "
        f"final loss={final_loss:.6g}; iterations={iter_count}"
        + (f" | {end_mem_info}" if end_mem_info else "")
    )

    return x.detach().cpu().numpy()


def optimize_torch_fista(
    *, model, x_init, dtype, max_its, cost_dev, optim_dev, params,
    positivity=False, initial_step=None, initial_update=1.0e-5,
    backtracking_factor=0.5, grow_factor=1.25,
):
    """Monotone backtracking FISTA using gradients from ``model.objective``."""
    if max_its < 1:
        raise ValueError("max_its must be at least one.")
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must lie between zero and one.")
    if grow_factor < 1.0:
        raise ValueError("grow_factor must be at least one.")
    if initial_update <= 0.0:
        raise ValueError("initial_update must be positive.")

    if optim_dev != cost_dev:
        logger.info(f"FISTA uses the cost device for objective evaluations; solving on {cost_dev}.")

    x = torch.as_tensor(x_init, dtype=dtype, device=cost_dev).clone()
    if positivity:
        x.clamp_(min=0)
    y = x.clone()
    t_k = 1.0

    def evaluate(candidate):
        leaf = candidate.detach().requires_grad_(True)
        loss = model.objective(leaf, device=cost_dev, **params)
        if leaf.grad is None:
            raise RuntimeError("objective() did not produce a gradient.")
        return loss.detach(), leaf.grad.detach()

    loss_x, grad_x = evaluate(x)
    if initial_step is None:
        initial_step = initial_update / max(float(grad_x.abs().max()), 1.0e-20)
    step = float(initial_step)
    logger.info(
        f"Starting optimisation: FISTA on {cost_dev}"
        + (" with positivity projection" if positivity else "")
        + f"; initial step={step:.6e}"
    )

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    t0 = time.perf_counter()

    for iteration in range(1, int(max_its) + 1):
        loss_y, grad_y = evaluate(y)
        reference_y, local_t = y, t_k
        if loss_y > loss_x:
            reference_y, loss_y, grad_y, local_t = x, loss_x, grad_x, 1.0

        accepted = False
        for _ in range(30):
            candidate = reference_y - step * grad_y
            if positivity:
                candidate = candidate.clamp_min(0)
            loss_candidate, grad_candidate = evaluate(candidate)
            delta = candidate - reference_y
            majorizer = loss_y + torch.sum(grad_y * delta) + torch.sum(delta * delta) / (2.0 * step)
            if torch.isfinite(loss_candidate) and loss_candidate <= majorizer:
                accepted = True
                break
            step *= backtracking_factor

        if not accepted:
            raise RuntimeError("FISTA failed to find a finite descent step.")
        if loss_candidate > loss_x:
            y, t_k, step = x.clone(), 1.0, step * backtracking_factor
            logger.info(f"[FISTA Iter {iteration}/{max_its}] restart; loss={float(loss_x):.6e}; step={step:.6e}")
            continue

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * local_t * local_t))
        y = candidate + ((local_t - 1.0) / t_next) * (candidate - x)
        rel_change = torch.linalg.vector_norm(candidate - x) / torch.clamp_min(
            torch.linalg.vector_norm(candidate), 1.0e-20
        )
        x, loss_x, grad_x, t_k = candidate, loss_candidate, grad_candidate, t_next
        step *= grow_factor
        logger.info(
            f"[PID {os.getpid()}] [FISTA Iter {iteration}/{max_its}] "
            f"loss={float(loss_x):.6e}; rel_change={float(rel_change):.6e}; step={step:.6e}"
        )

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    elapsed = time.perf_counter() - t0
    mem_info = dutils.gpu_mem_str(cost_dev) if cost_dev.type == "cuda" else ""
    logger.info(
        f"[Timing] FISTA (cost_dev={cost_dev}, positivity={positivity}) took {elapsed:.2f} s; "
        f"final loss={float(loss_x):.6g}"
        + (f" | {mem_info}" if mem_info else "")
    )
    return x.detach().cpu().numpy()
