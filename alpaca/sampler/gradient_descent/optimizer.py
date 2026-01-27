"""Gradient-based optimization routines for MAP estimation.

Implements Adam pre-conditioning and L-BFGS refinement with multi-start
parallel execution, automatic GPU memory management, and retry logic.
"""

import gc
import json
import os

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
from herculens.Inference.loss import Loss
from numpyro.infer.util import unconstrain_fn

from alpaca.sampler.utils import now


def make_safe_loss(loss_obj):
    """Wrap a loss function with numerical safeguards.

    Replaces NaN/Inf values with large penalty (1e30) to handle singular
    regions in lens modeling where the lens equation solver fails.

    Args:
        loss_obj: Loss function mapping unconstrained parameters to scalar.

    Returns:
        Numerically stabilized loss function.
    """
    def _safe(uvec):
        val = loss_obj(uvec)
        return jnp.where(jnp.isfinite(val), val, jnp.array(1e30))

    return _safe


def adam_preopt(
    loss_fn,
    u0,
    n_steps: int = 500,
    lr: float = 5e-3,
    warmup_fraction: float = 0.1,
    grad_clip: float = 10.0,
    use_cosine_decay: bool = True,
    min_lr_fraction: float = 0.01,
):
    """Adam pre-optimization with warmup, cosine decay, and gradient clipping.

    Args:
        loss_fn: Objective function to minimize.
        u0: Initial parameters in unconstrained space.
        n_steps: Number of gradient steps.
        lr: Peak learning rate (reached after warmup).
        warmup_fraction: Fraction of n_steps for linear warmup.
        grad_clip: Maximum gradient norm for clipping.
        use_cosine_decay: Whether to use cosine annealing after warmup.
        min_lr_fraction: Minimum lr as fraction of peak at end of decay.

    Returns:
        Parameters corresponding to minimum loss during optimization.
    """
    warmup_steps = int(n_steps * warmup_fraction)
    decay_steps = n_steps - warmup_steps

    if use_cosine_decay and decay_steps > 0:
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=lr * 0.01,
                    end_value=lr,
                    transition_steps=max(warmup_steps, 1),
                ),
                optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=decay_steps,
                    alpha=min_lr_fraction,
                ),
            ],
            boundaries=[warmup_steps],
        )
    else:
        if warmup_steps > 0:
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=lr * 0.01,
                        end_value=lr,
                        transition_steps=warmup_steps,
                    ),
                    optax.constant_schedule(lr),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.constant_schedule(lr)

    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=schedule),
    )
    state = opt.init(u0)

    @jax.jit
    def step(u, state):
        loss_val, g = jax.value_and_grad(loss_fn)(u)
        updates, state = opt.update(g, state, u)
        u2 = optax.apply_updates(u, updates)
        return u2, state, loss_val

    u = u0
    best = u0
    best_val = float(loss_fn(u0))

    for _ in range(n_steps):
        u, state, loss_val = step(u, state)
        lv = float(loss_val)
        if np.isfinite(lv) and lv < best_val:
            best = u
            best_val = lv
    return best


def adam_preopt_legacy(loss_fn, u0, n_steps: int = 200, lr: float = 5e-3):
    """
    Legacy Adam pre-optimization (constant learning rate, no clipping).

    Kept for backwards compatibility. Use adam_preopt() for better results.
    """
    opt = optax.adam(lr)
    state = opt.init(u0)

    @jax.jit
    def step(u, state):
        loss_val, g = jax.value_and_grad(loss_fn)(u)
        updates, state = opt.update(g, state)
        u2 = optax.apply_updates(u, updates)
        return u2, state, loss_val

    u = u0
    best = u0
    best_val = float(loss_fn(u0))

    for _ in range(n_steps):
        u, state, loss_val = step(u, state)
        lv = float(loss_val)
        if np.isfinite(lv) and lv < best_val:
            best = u
            best_val = lv
    return best


def _adam_preopt_jax(
    loss_fn,
    u0,
    n_steps: int = 500,
    lr: float = 5e-3,
    warmup_fraction: float = 0.1,
    grad_clip: float = 10.0,
    use_cosine_decay: bool = True,
    min_lr_fraction: float = 0.01,
):
    """Pure JAX Adam pre-optimization compatible with vmap/pmap.

    Args:
        loss_fn: JAX-differentiable objective function.
        u0: Initial parameters in unconstrained space.
        n_steps: Number of gradient steps.
        lr: Peak learning rate (after warmup).
        warmup_fraction: Fraction of n_steps for linear warmup.
        grad_clip: Maximum gradient norm for clipping.
        use_cosine_decay: Use cosine annealing after warmup.
        min_lr_fraction: Minimum lr as fraction of peak.

    Returns:
        Tuple of (best_params, best_loss).
    """
    warmup_steps = int(n_steps * warmup_fraction)
    decay_steps = n_steps - warmup_steps

    if use_cosine_decay and decay_steps > 0:
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=lr * 0.01,
                    end_value=lr,
                    transition_steps=max(warmup_steps, 1),
                ),
                optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=decay_steps,
                    alpha=min_lr_fraction,
                ),
            ],
            boundaries=[warmup_steps],
        )
    else:
        if warmup_steps > 0:
            schedule = optax.join_schedules(
                schedules=[
                    optax.linear_schedule(
                        init_value=lr * 0.01,
                        end_value=lr,
                        transition_steps=warmup_steps,
                    ),
                    optax.constant_schedule(lr),
                ],
                boundaries=[warmup_steps],
            )
        else:
            schedule = optax.constant_schedule(lr)

    # Build optimizer with gradient clipping
    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=schedule),
    )
    state = opt.init(u0)

    def step_fn(carry, _):
        u, state, best_u, best_val = carry
        loss_val, g = jax.value_and_grad(loss_fn)(u)
        updates, new_state = opt.update(g, state, u)
        new_u = optax.apply_updates(u, updates)

        # Track best parameters
        is_better = jnp.isfinite(loss_val) & (loss_val < best_val)
        new_best_u = jax.tree_util.tree_map(
            lambda b, n: jnp.where(is_better, n, b), best_u, new_u
        )
        new_best_val = jnp.where(is_better, loss_val, best_val)

        return (new_u, new_state, new_best_u, new_best_val), loss_val

    init_loss = loss_fn(u0)
    init_carry = (u0, state, u0, init_loss)

    (_, _, best_u, best_val), _ = jax.lax.scan(step_fn, init_carry, None, length=n_steps)

    return best_u, best_val


def _single_start_optimize(
    loss_fn,
    u0,
    do_preopt: bool,
    adam_steps: int,
    adam_lr: float,
    adam_warmup_fraction: float,
    adam_grad_clip: float,
    adam_use_cosine_decay: bool,
    lbfgs_maxiter: int,
    lbfgs_tol: float,
):
    """Single optimization trajectory: Adam pre-conditioning + L-BFGS refinement.

    Designed to be vmapped/pmapped across multiple starting points for
    parallel multi-start optimization.

    Args:
        loss_fn: Loss function to minimize.
        u0: Initial unconstrained parameters (PyTree).
        do_preopt: Whether to apply Adam pre-conditioning.
        adam_steps: Number of Adam iterations.
        adam_lr: Adam peak learning rate.
        adam_warmup_fraction: Fraction of steps for warmup.
        adam_grad_clip: Maximum gradient norm for clipping.
        adam_use_cosine_decay: Whether to use cosine annealing.
        lbfgs_maxiter: Maximum L-BFGS iterations.
        lbfgs_tol: L-BFGS convergence tolerance.

    Returns:
        Tuple of (optimized_params, final_loss).
    """
    # Adam pre-optimization
    if do_preopt:
        u_start, _ = _adam_preopt_jax(
            loss_fn, u0,
            n_steps=adam_steps,
            lr=adam_lr,
            warmup_fraction=adam_warmup_fraction,
            grad_clip=adam_grad_clip,
            use_cosine_decay=adam_use_cosine_decay,
        )
    else:
        u_start = u0

    # L-BFGS refinement using pure JAX implementation
    lbfgs = jaxopt.LBFGS(
        fun=loss_fn,
        maxiter=lbfgs_maxiter,
        tol=lbfgs_tol,
        history_size=10,
    )
    u_opt, lbfgs_state = lbfgs.run(u_start)

    final_loss = loss_fn(u_opt)

    return u_opt, final_loss


def run_gradient_descent(
    prob_model,
    img: np.ndarray,
    noise_map: np.ndarray,
    outdir: str,
    n_starts_initial: int = 50,
    n_top_for_refinement: int = 5,
    n_refinement_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    random_seed: int = 73,
    adam_steps_initial: int = 500,
    adam_steps_refinement: int = 750,
    adam_lr: float = 5e-3,
    adam_warmup_fraction: float = 0.1,
    adam_grad_clip: float = 10.0,
    adam_use_cosine_decay: bool = True,
    lbfgs_maxiter_initial: int = 600,
    lbfgs_maxiter_refinement: int = 1000,
    lbfgs_tol: float = 1e-5,
    verbose: bool = True,
    measured_delays=None,
    delay_errors=None,
    use_rayshoot_consistency: bool = False,
    rayshoot_consistency_sigma: float = 0.0002,
    use_rayshoot_systematic_error: bool = False,
    max_retry_iterations: int = 1,
    chi2_red_threshold: float = 2.0,
) -> dict:
    """Two-phase gradient descent optimization for MAP estimation.

    Primary optimization routine using a two-phase approach: (1) initial
    exploration with many parallel starts, (2) refinement of top results
    with perturbations. Each phase uses Adam + L-BFGS with automatic
    fallback to sequential processing if GPU memory is exhausted.

    Args:
        prob_model: Herculens probabilistic lens model.
        img: Observed image array.
        noise_map: Per-pixel noise standard deviation.
        outdir: Output directory for results.
        n_starts_initial: Number of starts in exploration phase.
        n_top_for_refinement: Number of top results to refine.
        n_refinement_perturbations: Perturbations per top result.
        perturbation_scale: Scale of Gaussian perturbations.
        random_seed: Random seed for reproducibility.
        adam_steps_initial: Adam steps for initial phase.
        adam_steps_refinement: Adam steps for refinement phase.
        adam_lr: Peak Adam learning rate.
        adam_warmup_fraction: Fraction of steps for warmup.
        adam_grad_clip: Maximum gradient norm.
        adam_use_cosine_decay: Use cosine annealing.
        lbfgs_maxiter_initial: L-BFGS iterations (initial).
        lbfgs_maxiter_refinement: L-BFGS iterations (refinement).
        lbfgs_tol: L-BFGS convergence tolerance.
        verbose: Print progress information.
        measured_delays: Observed time delays for likelihood.
        delay_errors: Time delay uncertainties.
        use_rayshoot_consistency: Enable ray-shooting consistency loss.
        rayshoot_consistency_sigma: Ray-shooting consistency tolerance.
        use_rayshoot_systematic_error: Include systematic error model.
        max_retry_iterations: Maximum optimization attempts. If chi2_red exceeds
            threshold, restart with new random seeds. Set to 1 for no retry.
        chi2_red_threshold: Reduced chi^2 threshold for retry. If best fit has
            chi2_red > threshold, restart optimization (up to max_retry_iterations).

    Returns:
        Dictionary with best-fit parameters, loss values, and timing info.
    """
    # Import loss-building helpers that remain in the main inference module
    from alpaca.sampler.losses import _build_rayshoot_consistency_loss, _build_time_delay_loss

    t_total_start = now()
    n_devices = jax.device_count()

    if verbose:
        print("=" * 60)
        print("TWO-PHASE GRADIENT DESCENT OPTIMIZATION")
        print("=" * 60)
        print(f"Devices available: {n_devices}")
        print(f"Phase 1: {n_starts_initial} initial starts (parallel)")
        print(f"Phase 2: Top {n_top_for_refinement} x {n_refinement_perturbations} = "
              f"{n_top_for_refinement * n_refinement_perturbations} refinements (parallel)")
        if max_retry_iterations > 1:
            print(f"Retry: up to {max_retry_iterations} iterations if \u03c7\u00b2_red > {chi2_red_threshold:.2f}")
        print("=" * 60)

    # Build loss function
    base_loss = Loss(prob_model)
    td_loss = _build_time_delay_loss(
        prob_model, prob_model.lens_image, measured_delays, delay_errors
    )
    rayshoot_loss = _build_rayshoot_consistency_loss(
        prob_model, prob_model.lens_image, use_rayshoot_consistency, rayshoot_consistency_sigma,
        use_rayshoot_systematic_error=use_rayshoot_systematic_error
    )

    loss_terms = [base_loss]
    if td_loss is not None:
        loss_terms.append(td_loss)
    if rayshoot_loss is not None:
        loss_terms.append(rayshoot_loss)

    if len(loss_terms) == 1:
        loss_fn = base_loss
    else:
        def loss_fn(u):
            return sum(term(u) for term in loss_terms)
    safe_loss = make_safe_loss(loss_fn)

    # Helper to stack pytrees
    def stack_pytrees(pytrees):
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pytrees)

    # Helper function for chunked optimization with OOM fallback
    def run_optimizations_with_chunked_fallback(
        all_u0_list, optimize_fn, phase_name, n_devices, verbose
    ):
        """
        Run optimizations with progressive fallback: full parallel -> chunked -> sequential.

        Fallback chain: full batch -> chunks of 25 -> chunks of 10 -> chunks of 5 -> sequential
        """
        n_total = len(all_u0_list)
        chunk_sizes_to_try = [n_total, 25, 10, 5, 1]  # Full, then progressively smaller chunks

        for chunk_size in chunk_sizes_to_try:
            if chunk_size > n_total:
                continue

            is_sequential = (chunk_size == 1)
            is_full_batch = (chunk_size == n_total)

            try:
                # Clear caches before each attempt
                jax.clear_caches()
                gc.collect()

                if is_sequential:
                    if verbose:
                        print(f"[{phase_name}] Running sequential (chunk_size=1)...")
                    results_u, results_loss = [], []
                    for i, u0 in enumerate(all_u0_list):
                        u_opt, loss = optimize_fn(u0)
                        results_u.append(u_opt)
                        results_loss.append(float(loss))
                        if verbose and (i + 1) % 10 == 0:
                            print(f"[{phase_name}] Completed {i + 1}/{n_total}")
                    u_opt_all = stack_pytrees(results_u)
                    losses_all = jnp.array(results_loss)
                    return u_opt_all, losses_all

                if is_full_batch:
                    if verbose:
                        print(f"[{phase_name}] Running {n_total} parallel optimizations...")
                else:
                    if verbose:
                        print(f"[{phase_name}] Running in chunks of {chunk_size}...")

                # Process in chunks
                all_results_u = []
                all_results_loss = []
                n_chunks = (n_total + chunk_size - 1) // chunk_size

                for chunk_idx in range(n_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, n_total)
                    chunk_u0_list = all_u0_list[start_idx:end_idx]
                    actual_chunk_size = len(chunk_u0_list)

                    u0_stacked_chunk = stack_pytrees(chunk_u0_list)

                    # Try multi-device pmap if available
                    if n_devices > 1 and actual_chunk_size >= n_devices:
                        starts_per_device = (actual_chunk_size + n_devices - 1) // n_devices
                        n_padded = starts_per_device * n_devices

                        # Pad if needed
                        if n_padded > actual_chunk_size:
                            padded_list = chunk_u0_list + [chunk_u0_list[0]] * (n_padded - actual_chunk_size)
                            u0_stacked_chunk = stack_pytrees(padded_list)

                        # Reshape for pmap
                        u0_pmap = jax.tree_util.tree_map(
                            lambda x, _nd=n_devices, _spd=starts_per_device: x.reshape((_nd, _spd) + x.shape[1:]),
                            u0_stacked_chunk
                        )
                        pmap_opt = jax.pmap(lambda batch: jax.vmap(optimize_fn)(batch))
                        u_opt_pmap, losses_pmap = pmap_opt(u0_pmap)

                        # Flatten and trim padding
                        u_opt_chunk = jax.tree_util.tree_map(
                            lambda x, _np=n_padded, _acs=actual_chunk_size: x.reshape((_np,) + x.shape[2:])[:_acs],
                            u_opt_pmap
                        )
                        losses_chunk = losses_pmap.reshape(-1)[:actual_chunk_size]
                    else:
                        # Single device: use vmap
                        u_opt_chunk, losses_chunk = jax.vmap(optimize_fn)(u0_stacked_chunk)

                    # Collect results from this chunk
                    for i in range(actual_chunk_size):
                        u_i = jax.tree_util.tree_map(lambda x, _i=i: x[_i], u_opt_chunk)
                        all_results_u.append(u_i)
                        all_results_loss.append(float(losses_chunk[i]))

                    if verbose and not is_full_batch:
                        print(f"[{phase_name}] Chunk {chunk_idx + 1}/{n_chunks} done ({end_idx}/{n_total} total)")

                u_opt_all = stack_pytrees(all_results_u)
                losses_all = jnp.array(all_results_loss)
                return u_opt_all, losses_all

            except Exception as e:
                error_str = str(e)
                is_oom = ("RESOURCE_EXHAUSTED" in error_str or
                          "out of memory" in error_str.lower() or
                          "oom" in error_str.lower())

                if is_oom and chunk_size > 1:
                    if verbose:
                        print(f"[{phase_name}] OOM with chunk_size={chunk_size}, trying smaller chunks...")
                    jax.clear_caches()
                    gc.collect()
                    continue
                else:
                    raise

        # Should never reach here, but just in case
        raise RuntimeError(f"[{phase_name}] All fallback strategies failed")

    # Track best result across all retry iterations
    best_result_overall = None
    best_chi2_red_overall = float('inf')
    all_iteration_results = []

    # =================================================================
    # RETRY LOOP: Repeat optimization if chi^2_red > threshold
    # =================================================================
    for iteration in range(max_retry_iterations):
        # Clear caches at the start of each iteration to prevent memory accumulation
        jax.clear_caches()
        gc.collect()

        # Use different random seed for each iteration
        iter_seed = random_seed + iteration * 10000

        if iteration > 0:
            if verbose:
                print("\n" + "=" * 60)
                print(f"RETRY ITERATION {iteration + 1}/{max_retry_iterations}")
                print(f"Previous best \u03c7\u00b2_red: {best_chi2_red_overall:.4f} > threshold {chi2_red_threshold:.2f}")
                print(f"Starting {n_starts_initial} new initial starts with seed {iter_seed}...")
                print("=" * 60)

        # =================================================================
        # PHASE 1: Initial exploration (PARALLELIZED)
        # =================================================================
        if verbose:
            print("\n[Phase 1] Sampling initial parameters...")

        t_phase1_start = now()

        # Sample from prior for each start (use iter_seed for retry variation)
        all_u0_initial = []
        for i in range(n_starts_initial):
            key = jax.random.PRNGKey(int(iter_seed + 101 * i))
            init_params = prob_model.get_sample(prng_key=key)
            u0 = unconstrain_fn(prob_model.model, (), {}, init_params)
            all_u0_initial.append(u0)

        # Define optimization function for Phase 1
        def optimize_single_phase1(u0):
            return _single_start_optimize(
                safe_loss, u0,
                do_preopt=True,
                adam_steps=adam_steps_initial,
                adam_lr=adam_lr,
                adam_warmup_fraction=adam_warmup_fraction,
                adam_grad_clip=adam_grad_clip,
                adam_use_cosine_decay=adam_use_cosine_decay,
                lbfgs_maxiter=lbfgs_maxiter_initial,
                lbfgs_tol=lbfgs_tol,
            )

        # Run Phase 1 with chunked fallback on OOM
        u_opt_initial, losses_initial = run_optimizations_with_chunked_fallback(
            all_u0_initial, optimize_single_phase1, "Phase 1", n_devices, verbose
        )

        # Convert to numpy and find top results
        losses_initial_np = np.asarray(losses_initial)
        top_indices = np.argsort(losses_initial_np)[:n_top_for_refinement]

        # Extract top results
        top_u_opts = [
            jax.tree_util.tree_map(lambda x, _idx=idx: x[_idx], u_opt_initial)
            for idx in top_indices
        ]
        top_losses = losses_initial_np[top_indices]

        t_phase1 = now() - t_phase1_start

        if verbose:
            print(f"\n[Phase 1] Completed in {t_phase1:.1f}s")
            print(f"[Phase 1] Best {n_top_for_refinement} losses: {top_losses}")

        # Clear caches between phases to free up memory
        jax.clear_caches()
        gc.collect()

        # =================================================================
        # PHASE 2: Refinement with perturbations (PARALLELIZED)
        # =================================================================
        if verbose:
            print(f"\n[Phase 2] Generating {n_top_for_refinement * n_refinement_perturbations} perturbations...")

        t_phase2_start = now()

        # Generate all perturbations (use iter_seed for retry variation)
        all_u0_refinement = []
        refinement_metadata = []  # Track which top result each perturbation came from

        for top_idx, u_base in enumerate(top_u_opts):
            for perturb_idx in range(n_refinement_perturbations):
                key = jax.random.PRNGKey(iter_seed + 10000 + top_idx * 1000 + perturb_idx)
                if perturb_idx == 0:
                    # First perturbation is no perturbation (keep original)
                    u_perturbed = u_base
                else:
                    # Add Gaussian noise
                    noise = jax.tree_util.tree_map(
                        lambda x, _key=key: jax.random.normal(_key, shape=x.shape) * perturbation_scale,
                        u_base
                    )
                    u_perturbed = jax.tree_util.tree_map(lambda x, n: x + n, u_base, noise)
                all_u0_refinement.append(u_perturbed)
                refinement_metadata.append({"top_idx": top_idx, "perturb_idx": perturb_idx})

        n_refinements = len(all_u0_refinement)

        # Define optimization function for Phase 2 (more iterations, lower lr)
        def optimize_single_phase2(u0):
            return _single_start_optimize(
                safe_loss, u0,
                do_preopt=True,
                adam_steps=adam_steps_refinement,
                adam_lr=adam_lr * 0.5,  # Lower lr for refinement
                adam_warmup_fraction=adam_warmup_fraction,
                adam_grad_clip=adam_grad_clip,
                adam_use_cosine_decay=adam_use_cosine_decay,
                lbfgs_maxiter=lbfgs_maxiter_refinement,
                lbfgs_tol=lbfgs_tol,
            )

        # Run Phase 2 with chunked fallback on OOM
        u_opt_refinement, losses_refinement = run_optimizations_with_chunked_fallback(
            all_u0_refinement, optimize_single_phase2, "Phase 2", n_devices, verbose
        )

        losses_refinement_np = np.asarray(losses_refinement)

        t_phase2 = now() - t_phase2_start

        if verbose:
            print(f"\n[Phase 2] Completed in {t_phase2:.1f}s")
            print(f"[Phase 2] Best 5 refinement losses: {np.sort(losses_refinement_np)[:5]}")

        # =================================================================
        # Find best result from this iteration and compute chi2
        # =================================================================
        # Combine all losses
        all_losses_np = np.concatenate([losses_initial_np, losses_refinement_np])
        best_global_idx = np.argmin(all_losses_np)

        if best_global_idx < n_starts_initial:
            # Best is from Phase 1
            best_u = jax.tree_util.tree_map(lambda x, _idx=best_global_idx: x[_idx], u_opt_initial)
            best_phase = "initial"
        else:
            # Best is from Phase 2
            ref_idx = best_global_idx - n_starts_initial
            best_u = jax.tree_util.tree_map(lambda x, _idx=ref_idx: x[_idx], u_opt_refinement)
            best_phase = "refinement"

        best_loss = float(all_losses_np[best_global_idx])
        best_params = prob_model.constrain(best_u)
        best_chi2_red = float(prob_model.reduced_chi2(best_params, n_params=None))

        t_iter = now() - t_phase1_start

        # Compute chi2 for all results (for summary)
        initial_chi2_reds = []
        for i in range(n_starts_initial):
            u_i = jax.tree_util.tree_map(lambda x, _i=i: x[_i], u_opt_initial)
            params_i = prob_model.constrain(u_i)
            chi2_i = float(prob_model.reduced_chi2(params_i, n_params=None))
            initial_chi2_reds.append(chi2_i)

        refinement_chi2_reds = []
        for i in range(n_refinements):
            u_i = jax.tree_util.tree_map(lambda x, _i=i: x[_i], u_opt_refinement)
            params_i = prob_model.constrain(u_i)
            chi2_i = float(prob_model.reduced_chi2(params_i, n_params=None))
            refinement_chi2_reds.append(chi2_i)

        # Store iteration result
        iter_result = {
            "iteration": iteration + 1,
            "best_u": best_u,
            "best_params": best_params,
            "best_loss": best_loss,
            "best_chi2_red": best_chi2_red,
            "best_phase": best_phase,
            "timing": {"phase1": float(t_phase1), "phase2": float(t_phase2), "total": float(t_iter)},
            "initial_losses": losses_initial_np.tolist(),
            "initial_chi2_reds": initial_chi2_reds,
            "refinement_losses": losses_refinement_np.tolist(),
            "refinement_chi2_reds": refinement_chi2_reds,
        }
        all_iteration_results.append(iter_result)

        # Update best overall result
        if best_chi2_red < best_chi2_red_overall:
            best_chi2_red_overall = best_chi2_red
            best_result_overall = iter_result

        if verbose:
            print(f"\n[Iteration {iteration + 1}] Best \u03c7\u00b2_red: {best_chi2_red:.4f}")

        # Check if fit is good enough
        if best_chi2_red <= chi2_red_threshold:
            if verbose:
                print(f"[Iteration {iteration + 1}] Good fit found (\u03c7\u00b2_red = {best_chi2_red:.4f} \u2264 {chi2_red_threshold:.2f})")
            break
        elif iteration < max_retry_iterations - 1:
            if verbose:
                print(f"[Iteration {iteration + 1}] Fit not good enough (\u03c7\u00b2_red = {best_chi2_red:.4f} > {chi2_red_threshold:.2f})")
                print(f"[Iteration {iteration + 1}] Will retry with new random starts...")
        else:
            if verbose:
                print(f"[Iteration {iteration + 1}] Max iterations reached. Using best result from all iterations.")

    # =================================================================
    # Final result: use best across all iterations
    # =================================================================
    t_total = now() - t_total_start

    # Get best result
    best_params = best_result_overall["best_params"]
    best_chi2_red = best_result_overall["best_chi2_red"]
    best_loss = best_result_overall["best_loss"]
    best_phase = best_result_overall["best_phase"]
    best_iteration = best_result_overall["iteration"]

    # Save best result
    best_params_json = {
        k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
        for k, v in best_params.items()
    }

    os.makedirs(outdir, exist_ok=True)
    best_path = os.path.join(outdir, "best_fit_params.json")
    with open(best_path, "w") as f:
        json.dump(best_params_json, f, indent=2)

    # Build summary
    summary = {
        "n_starts_initial": n_starts_initial,
        "n_top_for_refinement": n_top_for_refinement,
        "n_refinement_perturbations": n_refinement_perturbations,
        "total_optimizations": n_starts_initial + n_refinements,
        "best_loss": best_loss,
        "best_chi2_red": best_chi2_red,
        "best_from_phase": best_phase,
        "best_from_iteration": best_iteration,
        "total_iterations": len(all_iteration_results),
        "chi2_red_threshold": chi2_red_threshold,
        "best_params_json": best_params_json,
        "timing": {
            "total": float(t_total),
        },
        "initial_losses": best_result_overall["initial_losses"],
        "initial_chi2_reds": best_result_overall["initial_chi2_reds"],
        "refinement_losses": best_result_overall["refinement_losses"],
        "refinement_chi2_reds": best_result_overall["refinement_chi2_reds"],
        "all_iterations": [
            {
                "iteration": r["iteration"],
                "best_chi2_red": r["best_chi2_red"],
                "best_loss": r["best_loss"],
                "timing": r["timing"],
            }
            for r in all_iteration_results
        ],
    }

    summary_path = os.path.join(outdir, "multi_start_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print("=" * 60)
        print("GRADIENT DESCENT COMPLETE")
        print(f"  Total time: {t_total:.1f}s")
        print(f"  Total iterations: {len(all_iteration_results)}/{max_retry_iterations}")
        print(f"  Best from: iteration {best_iteration}, {best_phase} phase")
        print(f"  Best \u03c7\u00b2_red: {best_chi2_red:.4f}")
        print(f"  Best loss: {best_loss:.4g}")
        if len(all_iteration_results) > 1:
            chi2_list = [f"{r['best_chi2_red']:.4f}" for r in all_iteration_results]
            print(f"  \u03c7\u00b2_red per iteration: {chi2_list}")
        print("=" * 60)

    return summary


def load_multistart_summary(outdir: str, verbose: bool = True):
    """Restore multi-start optimization results from disk.

    Loads the optimization summary JSON and corresponding best-fit parameter
    file, enabling continuation of analysis without recomputation.

    Args:
        outdir: Directory containing multi_start_summary.json.
        verbose: Print loading status messages.

    Returns:
        Complete optimization summary with best-fit parameters.
    """
    import json as _json
    import os as _os

    summary_path = _os.path.join(outdir, "multi_start_summary.json")
    if not _os.path.exists(summary_path):
        raise FileNotFoundError(f"No multi-start summary found at {summary_path}")

    with open(summary_path) as f:
        summary = _json.load(f)

    best_run = summary.get("best_run", 0)
    best_params_path = _os.path.join(outdir, f"best_fit_params_run{best_run:02d}.json")

    if not _os.path.exists(best_params_path):
        raise FileNotFoundError(f"No best-fit file found at {best_params_path}")

    with open(best_params_path) as f:
        best_params = _json.load(f)

    summary["best_params_json"] = best_params
    if verbose:
        print(f"Loaded summary from: {summary_path}")
        print(f"Loaded best-fit parameters from: {best_params_path}")
    return summary
