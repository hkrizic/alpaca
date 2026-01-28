"""NUTS sampler execution and sample I/O.

Contains the core NUTS sampling logic using NumPyro, including multi-chain
parallel execution, diagnostics, and sample persistence.
"""

import json
import os
from functools import partial

import jax
import numpy as np

from alpaca.sampler.utils import now


def _print_filtered_mcmc_summary(mcmc, exclude_patterns=None):
    """Print MCMC summary excluding parameters matching certain patterns.

    This is useful for Correlated Fields models where source_pixels_field_xi
    has hundreds of parameters that make the output unreadable.

    Args:
        mcmc: NumPyro MCMC object with samples.
        exclude_patterns: List of string patterns to exclude from summary.
            Parameters containing any of these patterns will be skipped.
    """
    import numpyro.diagnostics as diagnostics

    if exclude_patterns is None:
        exclude_patterns = []

    samples = mcmc.get_samples(group_by_chain=True)

    # Filter out excluded parameters and convert to numpy
    filtered_samples = {}
    excluded_count = 0
    for key, value in samples.items():
        should_exclude = any(pattern in key for pattern in exclude_patterns)
        if should_exclude:
            excluded_count += 1
        else:
            # Convert to numpy array - diagnostics.summary() may not work with JAX arrays
            filtered_samples[key] = np.asarray(value)

    if not filtered_samples:
        print("[NUTS/NumPyro] No parameters to display after filtering.")
        if excluded_count > 0:
            print(f"[NUTS/NumPyro] ({excluded_count} parameters excluded by filter)")
        return

    # Compute diagnostics for filtered samples
    # NumPyro 0.19+ returns: {param_name: {stat_name: value, ...}, ...}
    summary_dict = diagnostics.summary(filtered_samples)

    # Check which parameters are in summary vs samples
    summary_params = set(summary_dict.keys())
    sample_params = set(filtered_samples.keys())
    missing_from_summary = sample_params - summary_params
    if missing_from_summary:
        print(f"[NUTS/NumPyro] WARNING: Parameters in samples but not in diagnostics: {missing_from_summary}")
        for param in missing_from_summary:
            arr = filtered_samples[param]
            print(f"  {param}: shape={arr.shape}, dtype={arr.dtype}, "
                  f"min={float(arr.min()):.4g}, max={float(arr.max()):.4g}, "
                  f"has_nan={bool(np.any(np.isnan(arr)))}, has_inf={bool(np.any(np.isinf(arr)))}")

    # Print header
    print()
    header = f"{'':>40} {'mean':>10} {'std':>10} {'median':>10} {'5.0%':>10} {'95.0%':>10} {'n_eff':>10} {'r_hat':>10}"
    print(header)

    # Print each parameter
    for key in sorted(filtered_samples.keys()):
        try:
            # NumPyro 0.19+ format: summary_dict[param_name] = {stat_name: value}
            stats = summary_dict[key]

            # Skip vector-valued parameters whose stats are arrays (e.g.
            # x_image, y_image, ps_amp, shapelets_amp_S).  float() would
            # raise TypeError on these.
            mean_val = stats['mean']
            if np.ndim(mean_val) > 0:
                n_elem = np.size(mean_val)
                print(f"{key:>40} {'(vector param, ' + str(n_elem) + ' elements — skipped)':>70}")
                continue

            mean = float(mean_val)
            std = float(stats['std'])
            median = float(stats['median'])
            q5 = float(stats['5.0%'])
            q95 = float(stats['95.0%'])
            n_eff = float(stats['n_eff'])
            r_hat = float(stats['r_hat'])

            print(f"{key:>40} {mean:>10.2f} {std:>10.2f} {median:>10.2f} {q5:>10.2f} {q95:>10.2f} {n_eff:>10.2f} {r_hat:>10.2f}")
        except KeyError:
            # Parameter might have unusual shape that diagnostics.summary() doesn't handle
            print(f"{key:>40} {'(diagnostics unavailable)':>70}")

    print()
    if excluded_count > 0:
        print(f"[NUTS/NumPyro] ({excluded_count} source_pixels_field_xi parameters excluded from summary)")
    print()


def _numpyro_model(logdensity_fn, init_params_structure):
    """NumPyro model that uses an external log-density function.

    Samples unconstrained parameters from ImproperUniform priors and
    applies the log-density as a potential factor.

    Args:
        logdensity_fn: Log-density function mapping parameter dict to scalar.
        init_params_structure: Dict of initial parameter values (single chain)
            used to infer parameter names and shapes.
    """
    import numpyro
    import numpyro.distributions as dist

    params = {}
    for key, init_val in init_params_structure.items():
        # Get the shape for this parameter (excluding chain dimension)
        if init_val.ndim > 1:
            param_shape = init_val.shape[1:]
        elif init_val.ndim == 1:
            param_shape = ()
        else:
            param_shape = ()

        # Use ImproperUniform prior (flat over all reals)
        params[key] = numpyro.sample(
            key,
            dist.ImproperUniform(
                dist.constraints.real,
                batch_shape=(),
                event_shape=param_shape,
            )
        )

    # Add the log-density as a factor
    with numpyro.handlers.block():
        log_prob = logdensity_fn(params)

    numpyro.factor("log_density", log_prob)


def _compute_log_densities_batched(compute_fn, samples, n_total, batch_size):
    """Compute log-densities in batches of a given size.

    Args:
        compute_fn: Vmapped + jitted log-density function.
        samples: Dict of raw sample arrays (total samples along axis 0).
        n_total: Total number of samples.
        batch_size: Number of samples per batch.

    Returns:
        1-D numpy array of log-density values.
    """
    log_density_batches = []
    for i in range(0, n_total, batch_size):
        batch_samples = jax.tree_util.tree_map(
            lambda x, _i=i: x[_i : _i + batch_size],
            samples
        )
        batch_log_prob = compute_fn(batch_samples)
        log_density_batches.append(np.asarray(batch_log_prob))
    return np.concatenate(log_density_batches)


def run_nuts_numpyro(
    logdensity_fn,
    initial_positions: dict,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = None,
    seed: int = 42,
    outdir: str = None,
    verbose: bool = True,
    target_accept_prob: float = 0.8,
    max_tree_depth: int = 10,
    chain_method: str = "parallel",
    progress_bar: bool = True,
):
    """Execute NUTS sampling with NumPyro and multi-device parallelization.

    Performs Hamiltonian Monte Carlo using the No-U-Turn Sampler (Hoffman &
    Gelman 2014) via NumPyro. Chains run in parallel using jax.pmap() when
    multiple devices are available.

    Args:
        logdensity_fn: Log-density function mapping parameter dict to scalar.
            Must be JAX-compatible (jit, grad).
        initial_positions: Initial parameter values with shape (num_chains,)
            or (num_chains, param_dim) per leaf.
        num_warmup: Warmup iterations for step size and mass matrix adaptation.
        num_samples: Posterior samples per chain after warmup.
        num_chains: Number of parallel chains (default: all available devices).
        seed: Random seed for reproducibility.
        outdir: Output directory for sample persistence.
        verbose: Print progress and diagnostics.
        target_accept_prob: Target Metropolis acceptance rate (default 0.8).
        max_tree_depth: Maximum tree depth for NUTS (default 10).
        chain_method: "parallel" (pmap), "vectorized" (vmap), or "sequential".
        progress_bar: Show progress bar during sampling.

    Returns:
        Results dict containing samples, samples_by_chain, log_density,
        divergences, acceptance_rate, runtime, and config.

    Note:
        When chain_method="parallel", chains are distributed across available
        accelerators using jax.pmap for efficient parallel execution.
    """
    import numpyro
    from numpyro.infer import MCMC, NUTS

    n_devices = jax.device_count()

    if num_chains is None:
        num_chains = n_devices

    if verbose:
        print(f"[NUTS/NumPyro] Starting with {num_chains} chains on {n_devices} devices")
        print(f"[NUTS/NumPyro] Warmup: {num_warmup}, Samples: {num_samples}")
        print(f"[NUTS/NumPyro] Chain method: {chain_method}")

    # Extract initial position for chain 0 (NumPyro will handle multi-chain init)
    first_key = list(initial_positions.keys())[0]
    first_val = initial_positions[first_key]
    provided_chains = first_val.shape[0] if hasattr(first_val, 'shape') and first_val.ndim > 0 else 1

    if provided_chains != num_chains:
        raise ValueError(
            f"initial_positions has {provided_chains} chains but num_chains={num_chains}. "
            f"Ensure each parameter leaf has shape (num_chains,) or (num_chains, dim)."
        )

    # Get structure for the model (use first chain's values)
    init_structure = jax.tree_util.tree_map(lambda x: x[0], initial_positions)

    # Create NUTS kernel
    nuts_kernel = NUTS(
        partial(_numpyro_model, logdensity_fn, init_structure),
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        init_strategy=numpyro.infer.init_to_value(values=init_structure),
        dense_mass=True,
    )

    # Prepare initial parameters for all chains
    # NumPyro expects init_params to be a dict where each value has shape (num_chains,) or (num_chains, dim)
    init_params = initial_positions

    rng_key = jax.random.PRNGKey(seed)

    # Chain methods to try in order of preference (fastest to slowest)
    chain_methods_to_try = [chain_method]
    if chain_method != "sequential":
        chain_methods_to_try.append("sequential")

    mcmc = None
    runtime = None

    for try_chain_method in chain_methods_to_try:
        try:
            if verbose and try_chain_method != chain_method:
                print(f"[NUTS/NumPyro] Retrying with chain_method='{try_chain_method}'...")

            # Create MCMC sampler with current chain method
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method=try_chain_method,
                progress_bar=progress_bar if verbose else False,
            )

            t0 = now()

            # Run MCMC
            mcmc.run(rng_key, init_params=init_params)

            runtime = now() - t0
            break  # Success, exit retry loop

        except Exception as e:
            error_str = str(e).lower()
            is_oom = ("resource_exhausted" in error_str or
                      "out of memory" in error_str or
                      "oom" in error_str)
            if is_oom and try_chain_method != "sequential":
                if verbose:
                    print(f"[NUTS/NumPyro] OOM with chain_method='{try_chain_method}', will retry...")
                # Clear JAX caches to free memory before retry
                jax.clear_caches()
                continue
            else:
                raise  # Re-raise if not OOM or already at sequential

    if mcmc is None or runtime is None:
        raise RuntimeError("MCMC sampling failed after all retry attempts")

    if verbose:
        print(f"[NUTS/NumPyro] Sampling completed in {runtime:.2f}s")

    # Get samples - returns dict with shape (num_chains, num_samples, ...)
    samples_raw = mcmc.get_samples(group_by_chain=True)

    # Also get flattened samples
    samples_flat_raw = mcmc.get_samples(group_by_chain=False)

    # Convert to numpy arrays
    samples_by_chain = jax.tree_util.tree_map(
        lambda x: np.asarray(x),
        samples_raw
    )

    samples_flat = jax.tree_util.tree_map(
        lambda x: np.asarray(x),
        samples_flat_raw
    )

    # Get extra fields (divergences, etc.)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    # Extract divergences
    divergences = None
    if "diverging" in extra_fields:
        divergences = np.asarray(extra_fields["diverging"]).reshape(-1)
        n_divergent = np.sum(divergences)
        div_rate = n_divergent / divergences.size
        if verbose:
            print(f"[NUTS/NumPyro] Divergences: {n_divergent} ({div_rate*100:.2f}%)")
            if div_rate > 0.05:
                print("[NUTS/NumPyro] WARNING: High divergence rate. Consider reparameterization.")

    # Compute log-density for each sample
    if verbose:
        print("[NUTS/NumPyro] Computing log-densities for samples...")

    compute_log_density_vmap = jax.jit(jax.vmap(logdensity_fn))

    n_total_samples = len(next(iter(samples_flat_raw.values())))

    # Try progressively smaller batch sizes on OOM
    batch_sizes_to_try = [100, 50, 25, 10, 5, 1]
    log_density_flat = None

    for batch_size in batch_sizes_to_try:
        try:
            if verbose:
                print(f"[NUTS/NumPyro] Computing log-densities in batches of {batch_size}...")
            log_density_flat = _compute_log_densities_batched(
                compute_log_density_vmap, samples_flat_raw, n_total_samples, batch_size
            )
            break  # Success, exit the retry loop
        except Exception as e:
            error_str = str(e).lower()
            is_oom = ("resource_exhausted" in error_str or
                      "out of memory" in error_str or
                      "oom" in error_str)
            if is_oom and batch_size > 1:
                if verbose:
                    print(f"[NUTS/NumPyro] OOM with batch_size={batch_size}, retrying with smaller batch...")
                # Clear JAX caches to free memory
                jax.clear_caches()
                continue
            else:
                raise  # Re-raise if not OOM or already at minimum batch size

    if log_density_flat is None:
        raise RuntimeError("Failed to compute log-densities even with batch_size=1")

    # Also compute by chain
    log_density_by_chain = log_density_flat.reshape((num_chains, num_samples))

    # Get acceptance rate from MCMC stats
    acceptance_rate = None
    if hasattr(mcmc, 'last_state') and mcmc.last_state is not None:
        try:
            # NumPyro stores accept_prob in extra fields
            if "accept_prob" in extra_fields:
                acceptance_rate = np.mean(np.asarray(extra_fields["accept_prob"]))
                if verbose:
                    print(f"[NUTS/NumPyro] Mean acceptance rate: {acceptance_rate:.3f}")
        except Exception:
            pass

    # Print MCMC summary if verbose (filter out source_pixels_field_xi for readability)
    # Note: numpyro.diagnostics.summary() may fail with some NumPyro versions
    # The samples are still valid even if diagnostics don't print
    if verbose:
        try:
            print("\n[NUTS/NumPyro] Diagnostics summary (unconstrained parameter space):")
            print("[NUTS/NumPyro] Note: values below are in the internal unconstrained")
            print("[NUTS/NumPyro] parameterisation used by NUTS, NOT physical units.")
            _print_filtered_mcmc_summary(mcmc, exclude_patterns=["source_pixels_field_xi"])
        except Exception as e:
            print(f"[NUTS/NumPyro] Could not print diagnostics summary: {e}")
            print("[NUTS/NumPyro] (This doesn't affect the sampling results)")

    result_dict = {
        "samples": samples_flat,
        "samples_by_chain": samples_by_chain,
        "log_density": log_density_flat,
        "log_density_by_chain": log_density_by_chain,
        "divergences": divergences,
        "acceptance_rate": acceptance_rate,
        "step_sizes": None,  # NumPyro doesn't expose adapted step sizes easily
        "runtime": runtime,
        "config": {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": num_chains,
            "seed": seed,
            "target_accept_prob": target_accept_prob,
            "max_tree_depth": max_tree_depth,
            "chain_method": chain_method,
            "n_devices": n_devices,
        },
        "mcmc": mcmc,  # Store MCMC object for additional diagnostics
    }

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "nuts_samples.npz")

        save_dict = {
            "log_density": log_density_flat,
            "runtime": runtime,
            "config_json": np.array(json.dumps(result_dict["config"])),
        }
        for k, v in samples_flat.items():
            save_dict[f"s_{k}"] = v

        if divergences is not None:
            save_dict["divergences"] = divergences

        np.savez_compressed(out_path, **save_dict)
        if verbose:
            print(f"[NUTS/NumPyro] Samples saved to {out_path}")

    return result_dict


def load_nuts_samples(outdir: str):
    """Restore NUTS samples from compressed archive.

    Args:
        outdir: Directory containing nuts_samples.npz.

    Returns:
        Reconstructed samples dictionary with log-density and diagnostics.
    """
    path = os.path.join(outdir, "nuts_samples.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No NUTS samples found at {path}")

    data = np.load(path, allow_pickle=True)
    samples = {}
    for k in data.files:
        if k.startswith("s_"):
            samples[k[2:]] = data[k]

    result = {
        "samples": samples,
        "log_density": data["log_density"],
        "runtime": float(data["runtime"]) if "runtime" in data else 0.0,
    }

    if "divergences" in data.files:
        result["divergences"] = data["divergences"]
    if "step_sizes" in data.files:
        result["step_sizes"] = data["step_sizes"]
    if "config_json" in data.files:
        result["config"] = json.loads(str(data["config_json"]))

    return result
