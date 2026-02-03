"""Gradient-based optimization routines for MAP estimation.

Implements Adam pre-conditioning and L-BFGS refinement with multi-start
parallel execution, automatic GPU memory management, and retry logic.

author: hkrizic
"""

import gc
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
from numpyro.infer.util import unconstrain_fn

from alpaca.sampler.utils import now


def make_safe_loss(loss_obj):
    """Wrap a loss function with numerical safeguards.

    Replaces NaN/Inf values with large penalty (1e30) to handle singular
    regions in lens modeling where the lens equation solver fails.

    Parameters
    ----------
    loss_obj : callable
        Loss function mapping unconstrained parameters to scalar.

    Returns
    -------
    callable
        Numerically stabilized loss function.
    """
    def _safe(uvec):
        """Evaluate loss with NaN/Inf replacement.

        Parameters
        ----------
        uvec : jax.Array
            Unconstrained parameter vector.

        Returns
        -------
        jax.Array
            Loss value, or 1e30 if original value was non-finite.
        """
        val = loss_obj(uvec)
        return jnp.where(jnp.isfinite(val), val, jnp.array(1e30))

    return _safe


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

    Parameters
    ----------
    loss_fn : callable
        JAX-differentiable objective function.
    u0 : jax.Array
        Initial parameters in unconstrained space.
    n_steps : int
        Number of gradient steps.
    lr : float
        Peak learning rate (after warmup).
    warmup_fraction : float
        Fraction of n_steps for linear warmup.
    grad_clip : float
        Maximum gradient norm for clipping.
    use_cosine_decay : bool
        Use cosine annealing after warmup.
    min_lr_fraction : float
        Minimum lr as fraction of peak.

    Returns
    -------
    tuple of (jax.Array, float)
        Best parameters and best loss value.
    """
    warmup_steps = int(n_steps * warmup_fraction)
    decay_steps = n_steps - warmup_steps

    # Build learning rate schedule
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

    # Optimization step
    def step_fn(carry, _):
        """Perform one Adam gradient step and track best parameters.

        Parameters
        ----------
        carry : tuple
            Tuple of (current params, optimizer state, best params, best loss).
        _ : None
            Unused scan input.

        Returns
        -------
        tuple
            Updated carry and current loss value for scan output.
        """
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

    # The carry contains: current params, opt state, best params, best loss
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
    """
    Single optimization trajectory: Adam pre-conditioning + L-BFGS refinement.

    Designed to be vmapped/pmapped across multiple starting points for
    parallel multi-start optimization.

    Parameters
    ----------
    loss_fn : callable
        Loss function to minimize.
    u0 : PyTree
        Initial unconstrained parameters.
    do_preopt : bool
        Whether to apply Adam pre-conditioning.
    adam_steps : int
        Number of Adam iterations.
    adam_lr : float
        Adam peak learning rate.
    adam_warmup_fraction : float
        Fraction of steps for warmup.
    adam_grad_clip : float
        Maximum gradient norm for clipping.
    adam_use_cosine_decay : bool
        Whether to use cosine annealing.
    lbfgs_maxiter : int
        Maximum L-BFGS iterations.
    lbfgs_tol : float
        L-BFGS convergence tolerance.

    Returns
    -------
    tuple of (PyTree, float)
        Optimized parameters and final loss value.
    """
    # Adam pre-optimization
    if do_preopt:
        u_start, _ = _adam_preopt_jax(
            loss_fn, u0,
            n_steps=adam_steps,
            lr=adam_lr, # peak learning rate
            warmup_fraction=adam_warmup_fraction, # fraction of steps for warmup
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


def _stack_pytrees(pytrees):
    """Stack a list of pytrees along a new leading axis.

    Transforms N individual pytrees into a single pytree where each leaf
    has an additional leading dimension of size N.

    Parameters
    ----------
    pytrees : list of PyTree
        List of pytrees with identical structure.

    Returns
    -------
    PyTree
        Single pytree with leaves stacked along axis 0.
    """
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *pytrees)


def _run_optimizations_chunked(
    all_u0_list, optimize_fn, phase_name, n_devices, verbose
):
    """Run optimizations with progressive fallback on OOM.

    Fallback chain: full batch -> chunks of 25 -> chunks of 10 ->
    chunks of 5 -> sequential (chunk_size=1).

    Parameters
    ----------
    all_u0_list : list of PyTree
        Initial parameter pytrees.
    optimize_fn : callable
        Function mapping u0 -> (u_opt, loss).
    phase_name : str
        Label for log messages.
    n_devices : int
        Number of JAX devices available.
    verbose : bool
        Print progress information.

    Returns
    -------
    tuple of (PyTree, jax.Array)
        Stacked optimized parameters and 1-D loss array.
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
                u_opt_all = _stack_pytrees(results_u)
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

                u0_stacked_chunk = _stack_pytrees(chunk_u0_list)

                # Try multi-device pmap if available
                if n_devices > 1 and actual_chunk_size >= n_devices:
                    starts_per_device = (actual_chunk_size + n_devices - 1) // n_devices
                    n_padded = starts_per_device * n_devices


                    # Pad if needed
                    if n_padded > actual_chunk_size:
                        padded_list = chunk_u0_list + [chunk_u0_list[0]] * (n_padded - actual_chunk_size)
                        u0_stacked_chunk = _stack_pytrees(padded_list)

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

            u_opt_all = _stack_pytrees(all_results_u)
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
    use_image_pos_offset: bool = False,
    max_retry_iterations: int = 1,
    chi2_red_threshold: float = 2.0,
) -> dict:
    """Two-phase gradient descent optimization for MAP estimation.

    Primary optimization routine using a two-phase approach: (1) initial
    exploration with many parallel starts, (2) refinement of top results
    with perturbations. Each phase uses Adam + L-BFGS with automatic
    fallback to sequential processing if GPU memory is exhausted.

    Parameters
    ----------
    prob_model : herculens ProbModel
        Herculens probabilistic lens model.
    img : ndarray
        Observed image array.
    noise_map : ndarray
        Per-pixel noise standard deviation.
    outdir : str
        Output directory for results.
    n_starts_initial : int
        Number of starts in exploration phase.
    n_top_for_refinement : int
        Number of top results to refine.
    n_refinement_perturbations : int
        Perturbations per top result.
    perturbation_scale : float
        Scale of Gaussian perturbations.
    random_seed : int
        Random seed for reproducibility.
    adam_steps_initial : int
        Adam steps for initial phase.
    adam_steps_refinement : int
        Adam steps for refinement phase.
    adam_lr : float
        Peak Adam learning rate.
    adam_warmup_fraction : float
        Fraction of steps for warmup.
    adam_grad_clip : float
        Maximum gradient norm.
    adam_use_cosine_decay : bool
        Use cosine annealing.
    lbfgs_maxiter_initial : int
        L-BFGS iterations (initial).
    lbfgs_maxiter_refinement : int
        L-BFGS iterations (refinement).
    lbfgs_tol : float
        L-BFGS convergence tolerance.
    verbose : bool
        Print progress information.
    measured_delays : array-like or None
        Observed time delays for likelihood.
    delay_errors : array-like or None
        Time delay uncertainties.
    use_rayshoot_consistency : bool
        Enable ray-shooting consistency loss.
    rayshoot_consistency_sigma : float
        Ray-shooting consistency tolerance.
    use_rayshoot_systematic_error : bool
        Include systematic error model.
    use_image_pos_offset : bool
        Apply image position offsets for cosmography.
    max_retry_iterations : int
        Maximum optimization attempts. If chi2_red exceeds threshold,
        restart with new random seeds. Set to 1 for no retry.
    chi2_red_threshold : float
        Reduced chi-squared threshold for retry. If best fit has
        chi2_red > threshold, restart optimization.

    Returns
    -------
    dict
        Summary of optimization results and best-fit parameters:
        {
            "n_starts_initial": ...,
            "n_top_for_refinement": ...,
            "n_refinement_perturbations": ...,
            "total_optimizations": ...,
            "best_loss": ...,
            "best_chi2_red": ...,
            "best_from_phase": ...,
            "best_from_iteration": ...,
            "total_iterations": ...,
            "chi2_red_threshold": ...,
            "best_params_json": {...},
            "timing": {"total": ...},
            "initial_losses": [...],
            "initial_chi2_reds": [...],
            "refinement_losses": [...],
            "refinement_chi2_reds": [...],
            "all_iterations": [
                {
                    "iteration": ...,
                    "best_chi2_red": ...,
                    "best_loss": ...,
                    "timing": {...},
                },
                ...
            ],
        }
    """
    from alpaca.sampler.gradient_descent.likelihood import build_likelihood

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
    loss_fn = build_likelihood(
        prob_model,
        measured_delays=measured_delays,
        delay_errors=delay_errors,
        use_rayshoot_consistency=use_rayshoot_consistency,
        rayshoot_consistency_sigma=rayshoot_consistency_sigma,
        use_rayshoot_systematic_error=use_rayshoot_systematic_error,
        use_image_pos_offset=use_image_pos_offset,
    )
    safe_loss = make_safe_loss(loss_fn)

    # Build phase-specific optimize functions via partial
    optimize_phase1 = partial(
        _single_start_optimize,
        safe_loss,
        do_preopt=True,
        adam_steps=adam_steps_initial,
        adam_lr=adam_lr,
        adam_warmup_fraction=adam_warmup_fraction,
        adam_grad_clip=adam_grad_clip,
        adam_use_cosine_decay=adam_use_cosine_decay,
        lbfgs_maxiter=lbfgs_maxiter_initial,
        lbfgs_tol=lbfgs_tol,
    )

    optimize_phase2 = partial(
        _single_start_optimize,
        safe_loss,
        do_preopt=True,
        adam_steps=adam_steps_refinement,
        adam_lr=adam_lr * 0.5,  # Lower lr for refinement
        adam_warmup_fraction=adam_warmup_fraction,
        adam_grad_clip=adam_grad_clip,
        adam_use_cosine_decay=adam_use_cosine_decay,
        lbfgs_maxiter=lbfgs_maxiter_refinement,
        lbfgs_tol=lbfgs_tol,
    )

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

        # Run Phase 1 with chunked fallback on OOM
        u_opt_initial, losses_initial = _run_optimizations_chunked(
            all_u0_initial, optimize_phase1, "Phase 1", n_devices, verbose
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

        # Run Phase 2 with chunked fallback on OOM
        u_opt_refinement, losses_refinement = _run_optimizations_chunked(
            all_u0_refinement, optimize_phase2, "Phase 2", n_devices, verbose
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

    # Build flattened arrays (Phase 1 + Phase 2 from best iteration)
    all_chi2_reds = (best_result_overall["initial_chi2_reds"]
                     + best_result_overall["refinement_chi2_reds"])
    all_losses_flat = (best_result_overall["initial_losses"]
                       + best_result_overall["refinement_losses"])
    best_run = int(np.argmin(all_losses_flat))

    # Build summary
    summary = {
        "n_starts_initial": n_starts_initial,
        "n_top_for_refinement": n_top_for_refinement,
        "n_refinement_perturbations": n_refinement_perturbations,
        "total_optimizations": n_starts_initial + n_refinements,
        "best_loss": best_loss,
        "best_chi2_red": best_chi2_red,
        "best_run": best_run,
        "best_from_phase": best_phase,
        "best_from_iteration": best_iteration,
        "total_iterations": len(all_iteration_results),
        "chi2_red_threshold": chi2_red_threshold,
        "best_params_json": best_params_json,
        "timing": {
            "total": float(t_total),
        },
        "chi2_reds": all_chi2_reds,
        "all_losses": all_losses_flat,
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

    Parameters
    ----------
    outdir : str
        Directory containing multi_start_summary.json.
    verbose : bool
        Print loading status messages.

    Returns
    -------
    dict
        Complete optimization summary with best-fit parameters.
    """
    import json as _json
    import os as _os

    summary_path = _os.path.join(outdir, "multi_start_summary.json")
    if not _os.path.exists(summary_path):
        raise FileNotFoundError(f"No multi-start summary found at {summary_path}")

    with open(summary_path) as f:
        summary = _json.load(f)

    best_params_path = _os.path.join(outdir, "best_fit_params.json")

    if not _os.path.exists(best_params_path):
        raise FileNotFoundError(f"No best-fit file found at {best_params_path}")

    with open(best_params_path) as f:
        best_params = _json.load(f)

    summary["best_params_json"] = best_params
    if verbose:
        print(f"Loaded summary from: {summary_path}")
        print(f"Loaded best-fit parameters from: {best_params_path}")
    return summary
