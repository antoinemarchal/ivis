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


def _quadratic_extrapolation(loss, gradient, previous_gradient, displacement, momentum):
    """Cost and gradient at x + momentum * displacement for a fixed quadratic.

    ``displacement`` is x - previous_x. Accumulate scalar products in double
    precision in chunks to avoid full-size double image temporaries on CUDA.
    The returned gradient has its own storage whenever momentum is nonzero.
    """
    if momentum == 0.0:
        return loss, gradient
    gradient_delta = gradient - previous_gradient
    slope = torch.zeros((), dtype=torch.float64, device=gradient.device)
    curvature = torch.zeros_like(slope)
    chunks = zip(displacement.reshape(-1).split(1 << 20),
                 gradient.reshape(-1).split(1 << 20),
                 gradient_delta.reshape(-1).split(1 << 20))
    for delta, grad, grad_delta in chunks:
        delta64 = delta.double()
        slope.add_(torch.dot(delta64, grad.double()))
        curvature.add_(torch.dot(delta64, grad_delta.double()))
    extrapolated_loss = loss + momentum * slope + 0.5 * momentum**2 * curvature
    extrapolated_gradient = gradient_delta.mul_(momentum).add_(gradient)
    return extrapolated_loss, extrapolated_gradient


def _lower_bound_projection(positivity, positivity_floor, shape, dtype, device):
    """Return the projection applied to FISTA iterates, or None when there is no bound.

    positivity_floor=None keeps the original ``clamp(min=0)``. Otherwise the lower bound is
    the (scalar or broadcastable array) floor, which must be finite and <= 0: a soft
    positivity that lets the solution go as low as -c*sigma, so that noise stays two-sided
    while strongly negative sidelobes are still removed.
    """
    if not positivity:
        if positivity_floor is not None:
            raise ValueError("positivity_floor requires positivity=True.")
        return None
    if positivity_floor is None:
        return lambda v: v.clamp_min(0)
    floor = torch.as_tensor(np.asarray(positivity_floor), dtype=dtype, device=device)
    if not bool(torch.isfinite(floor).all()):
        raise ValueError("positivity_floor must be finite.")
    if bool((floor > 0).any()):
        raise ValueError("positivity_floor must be <= 0 (a negative lower bound, e.g. -4 sigma).")
    try:
        broadcast = torch.broadcast_shapes(tuple(floor.shape), tuple(shape))
    except RuntimeError:
        broadcast = None
    if broadcast != tuple(shape):
        raise ValueError(f"positivity_floor of shape {tuple(floor.shape)} does not broadcast to the image shape {tuple(shape)}.")
    if floor.ndim == 0:
        value = float(floor)
        return lambda v: v.clamp_min(value)
    return lambda v: torch.maximum(v, floor)


def optimize_torch_fista(
    *, model, x_init, dtype, max_its, cost_dev, optim_dev, params,
    positivity=False, initial_step=None, initial_update=1.0e-5,
    backtracking_factor=0.5, grow_factor=1.25, loss_only_line_search=False,
    quadratic_gradient_reuse=None, positivity_floor=None,
):
    """Monotone backtracking FISTA using gradients from ``model.objective``.

    ``loss_only_line_search`` avoids computing a gradient for each trial
    point.  It is available only to models that opt in with
    ``supports_loss_only_objective = True`` and honour
    ``compute_grad=False`` in ``objective``.

    ``quadratic_gradient_reuse=None`` (default) enables reuse for models
    declaring ``supports_quadratic_gradient_reuse = True``. Such models must
    have a fixed quadratic objective throughout the solve. Costs and gradients
    at extrapolated points are obtained from accepted iterates; every candidate
    is evaluated with its gradient for use by the next iteration. This takes
    precedence over ``loss_only_line_search``. Set reuse to False for the
    original evaluation sequence, or True to require a compatible model.
    Borderline comparisons refresh the extrapolated point directly to avoid
    roundoff in the quadratic identities stalling a nearly converged solve.

    ``positivity_floor`` (default None: iterates are clamped at 0) replaces the lower
    bound of the positivity projection by a negative scalar or per-pixel array, e.g.
    -4 sigma(x, y). It requires ``positivity=True``.
    """
    if max_its < 1:
        raise ValueError("max_its must be at least one.")
    if not 0.0 < backtracking_factor < 1.0:
        raise ValueError("backtracking_factor must lie between zero and one.")
    if grow_factor < 1.0:
        raise ValueError("grow_factor must be at least one.")
    if initial_update <= 0.0:
        raise ValueError("initial_update must be positive.")
    supports_reuse = getattr(model, "supports_quadratic_gradient_reuse", False)
    if quadratic_gradient_reuse is None:
        quadratic_gradient_reuse = supports_reuse
    elif quadratic_gradient_reuse and not supports_reuse:
        raise ValueError(
            "quadratic_gradient_reuse=True requires a model declaring "
            "supports_quadratic_gradient_reuse=True (a fixed quadratic objective)."
        )
    if (loss_only_line_search and not quadratic_gradient_reuse
            and not getattr(model, "supports_loss_only_objective", False)):
        raise ValueError(
            "loss_only_line_search=True requires a model that supports "
            "objective(..., compute_grad=False)."
        )

    if optim_dev != cost_dev:
        logger.info(f"FISTA uses the cost device for objective evaluations; solving on {cost_dev}.")

    x = torch.as_tensor(x_init, dtype=dtype, device=cost_dev).clone()
    project = _lower_bound_projection(positivity, positivity_floor, x.shape, dtype, cost_dev)
    if project is not None:
        x.copy_(project(x))
    y = x.clone()
    t_k = 1.0
    objective_evaluations = 0
    quadratic_refreshes = 0
    extrapolated_evaluation = False
    # Gram arithmetic can have lower precision than the solver image.
    reuse_eps = max(torch.finfo(dtype).eps,
                    torch.finfo(getattr(model, "gram_dtype", dtype)).eps)

    def evaluate(candidate, *, compute_grad=True):
        nonlocal objective_evaluations
        objective_evaluations += 1
        leaf = candidate.detach().requires_grad_(compute_grad)
        if compute_grad:
            loss = model.objective(leaf, device=cost_dev, **params)
            if leaf.grad is None:
                raise RuntimeError("objective() did not produce a gradient.")
            return loss.detach(), leaf.grad.detach()

        # The model opts in above.  ``no_grad`` is essential: simply skipping
        # ``backward`` would still retain a full candidate graph.
        with torch.no_grad():
            loss = model.objective(
                leaf, device=cost_dev, compute_grad=False, **params
            )
        return loss.detach(), None

    def near_roundoff(left, right):
        scale = torch.maximum(left.abs(), right.abs()).clamp_min(torch.finfo(dtype).tiny)
        return torch.abs(left - right) <= 8 * reuse_eps * scale

    def evaluate_trial(reference, reference_loss, reference_gradient, step):
        candidate = reference - step * reference_gradient
        if project is not None:
            candidate = project(candidate)
        loss, gradient = evaluate(
            candidate, compute_grad=quadratic_gradient_reuse or not loss_only_line_search
        )
        delta = candidate - reference
        majorizer = (reference_loss + torch.sum(reference_gradient * delta)
                     + torch.sum(delta * delta) / (2.0 * step))
        return candidate, loss, gradient, majorizer

    loss_x, grad_x = evaluate(x)
    if quadratic_gradient_reuse:
        loss_y, grad_y = loss_x, grad_x
    if initial_step is None:
        initial_step = initial_update / max(float(grad_x.abs().max()), 1.0e-20)
    step = float(initial_step)
    logger.info(
        f"Starting optimisation: FISTA on {cost_dev}"
        + (" with positivity projection" if positivity else "")
        + f"; initial step={step:.6e}; quadratic gradient reuse={quadratic_gradient_reuse}"
    )

    if cost_dev.type == "cuda":
        torch.cuda.synchronize(cost_dev)
    t0 = time.perf_counter()

    for iteration in range(1, int(max_its) + 1):
        if not quadratic_gradient_reuse:
            loss_y, grad_y = evaluate(y)
        elif extrapolated_evaluation and near_roundoff(loss_y, loss_x):
            loss_y, grad_y = evaluate(y)
            extrapolated_evaluation = False
            quadratic_refreshes += 1
        reference_y, local_t = y, t_k
        if loss_y > loss_x:
            # In loss-only mode an accepted candidate deliberately has no
            # cached gradient.  A monotone restart is rare, but when it
            # occurs obtain the gradient at x before using it as the new
            # reference point.
            if grad_x is None:
                _loss_x_check, grad_x = evaluate(x)
            reference_y, loss_y, grad_y, local_t = x, loss_x, grad_x, 1.0
            extrapolated_evaluation = False

        accepted = False
        for _ in range(30):
            candidate, loss_candidate, grad_candidate, majorizer = evaluate_trial(
                reference_y, loss_y, grad_y, step
            )
            if extrapolated_evaluation and near_roundoff(loss_candidate, majorizer):
                loss_y, grad_y = evaluate(reference_y)
                extrapolated_evaluation = False
                quadratic_refreshes += 1
                if loss_y > loss_x:
                    reference_y, loss_y, grad_y, local_t = x, loss_x, grad_x, 1.0
                # Rebuild the trial with the refreshed gradient at the same
                # step size; a refresh does not consume a backtracking step.
                candidate, loss_candidate, grad_candidate, majorizer = evaluate_trial(
                    reference_y, loss_y, grad_y, step
                )
            if torch.isfinite(loss_candidate) and loss_candidate <= majorizer:
                accepted = True
                break
            step *= backtracking_factor

        if not accepted:
            raise RuntimeError("FISTA failed to find a finite descent step.")
        if loss_candidate > loss_x:
            y, t_k, step = x.clone(), 1.0, step * backtracking_factor
            if quadratic_gradient_reuse:
                loss_y, grad_y = loss_x, grad_x
                extrapolated_evaluation = False
            logger.info(f"[FISTA Iter {iteration}/{max_its}] restart; loss={float(loss_x):.6e}; step={step:.6e}")
            continue

        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * local_t * local_t))
        momentum = (local_t - 1.0) / t_next
        displacement = candidate - x
        y = candidate + momentum * displacement
        if quadratic_gradient_reuse:
            loss_y, grad_y = _quadratic_extrapolation(
                loss_candidate, grad_candidate, grad_x, displacement, momentum
            )
            extrapolated_evaluation = momentum != 0.0
        del displacement
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
        f"; objective_evaluations={objective_evaluations}"
        f"; quadratic_refreshes={quadratic_refreshes}"
        + (f" | {mem_info}" if mem_info else "")
    )
    return x.detach().cpu().numpy()
