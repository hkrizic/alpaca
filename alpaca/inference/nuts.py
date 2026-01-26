"""
NUTS (No-U-Turn Sampler) for posterior sampling.

Uses the model's log-density (including all likelihood terms) directly,
ensuring consistency with gradient descent optimization.
"""

from typing import Dict
from time import time as now
import os
import json

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC
from numpyro.infer.util import constrain_fn
from numpyro.handlers import seed as numpyro_seed

from herculens.Inference.loss import Loss


def run_nuts(
    prob_model,
    initial_positions: Dict,
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

    The loss function is obtained from the ProbModel's Loss class, ensuring
    consistency with gradient descent optimization.

    Args:
        prob_model: Probabilistic lens model with model() method defining
            the likelihood via numpyro.factor statements.
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
    n_devices = jax.device_count()

    if num_chains is None:
        num_chains = n_devices

    if verbose:
        print(f"[NUTS/NumPyro] Starting with {num_chains} chains on {n_devices} devices")
        print(f"[NUTS/NumPyro] Warmup: {num_warmup}, Samples: {num_samples}")
        print(f"[NUTS/NumPyro] Chain method: {chain_method}")

    # Build log-density function from ProbModel
    loss_obj = Loss(prob_model)

    def logdensity_fn(params):
        """Negative loss = log-density (Loss returns -log_density)."""
        return -loss_obj(params)

    # Extract initial position for chain 0 (NumPyro will handle multi-chain init)
    first_key = list(initial_positions.keys())[0]
    first_val = initial_positions[first_key]
    provided_chains = first_val.shape[0] if hasattr(first_val, 'shape') and first_val.ndim > 0 else 1

    if provided_chains != num_chains:
        raise ValueError(
            f"initial_positions has {provided_chains} chains but num_chains={num_chains}. "
            f"Ensure each parameter leaf has shape (num_chains,) or (num_chains, dim)."
        )

    def numpyro_model(init_params_structure):
        """
        NumPyro model that uses the external log-density function.
        We sample unconstrained parameters and apply the log-density as potential.
        """
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

    # Get structure for the model (use first chain's values)
    init_structure = jax.tree_util.tree_map(lambda x: x[0], initial_positions)

    # Create NUTS kernel
    nuts_kernel = NUTS(
        lambda: numpyro_model(init_structure),
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
                print(f"[NUTS/NumPyro] WARNING: High divergence rate. Consider reparameterization.")

    # Compute log-density for each sample
    # We need to evaluate the logdensity_fn on each sample
    if verbose:
        print("[NUTS/NumPyro] Computing log-densities for samples...")

    def compute_log_density_single(params):
        return logdensity_fn(params)

    compute_log_density_vmap = jax.jit(jax.vmap(compute_log_density_single))

    # Batch the log-density computation to avoid OOM
    # Compute for flattened samples in chunks with automatic fallback on OOM
    n_total_samples = len(next(iter(samples_flat_raw.values())))

    def compute_log_densities_with_batch_size(batch_size):
        """Compute log densities with a given batch size."""
        log_density_batches = []
        for i in range(0, n_total_samples, batch_size):
            batch_samples = jax.tree_util.tree_map(
                lambda x: x[i : i + batch_size],
                samples_flat_raw
            )
            batch_log_prob = compute_log_density_vmap(batch_samples)
            log_density_batches.append(np.asarray(batch_log_prob))
        return np.concatenate(log_density_batches)

    # Try progressively smaller batch sizes on OOM
    batch_sizes_to_try = [100, 50, 25, 10, 5, 1]
    log_density_flat = None

    for batch_size in batch_sizes_to_try:
        try:
            if verbose:
                print(f"[NUTS/NumPyro] Computing log-densities in batches of {batch_size}...")
            log_density_flat = compute_log_densities_with_batch_size(batch_size)
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

    # Print MCMC summary if verbose
    if verbose:
        try:
            mcmc.print_summary()
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


def get_nuts_posterior(
    nuts_results: Dict,
    prob_model=None,
    n_samples: int = None,
    random_seed: int = None,
):
    """Convert NUTS samples to standardized posterior container.

    Transforms NumPyro MCMC output into the standard posterior format,
    flattening multi-dimensional parameters for visualization and analysis.

    NUTS samples are in unconstrained space. If prob_model is provided,
    samples are transformed back to constrained (physical) space.

    Args:
        nuts_results: Output from run_nuts or load_nuts_samples
            containing 'samples' and 'log_density'.
        prob_model: The probabilistic model used for sampling (optional).
            Required to transform samples to constrained space.
        n_samples: Optional subsampling to fixed sample count.
        random_seed: Seed for reproducible subsampling.

    Returns:
        Dictionary with flattened samples and parameter names.
    """
    samples_dict = nuts_results["samples"]
    log_density = np.asarray(nuts_results["log_density"])

    # Transform from unconstrained to constrained (physical) space
    if prob_model is not None:
        # Convert numpy arrays to jax arrays for constrain_fn
        samples_jax = {k: jnp.array(v) for k, v in samples_dict.items()}

        # Transform each sample from unconstrained to constrained space
        # Wrap model with seed handler to provide PRNG key for tracing
        n_total = len(log_density)
        seeded_model = numpyro_seed(prob_model.model, rng_seed=0)

        # First pass: determine which keys are actually in constrained space
        sample_0 = {k: samples_jax[k][0] for k in samples_jax.keys()}
        constrained_0 = constrain_fn(seeded_model, (), {}, sample_0)
        constrained_keys = list(constrained_0.keys())

        # Reinitialize with correct keys from constrained space
        constrained_samples = {k: [] for k in constrained_keys}

        for i in range(n_total):
            # Extract single sample
            sample_i = {k: samples_jax[k][i] for k in samples_jax.keys()}
            # Transform to constrained space
            constrained_i = constrain_fn(seeded_model, (), {}, sample_i)
            for k in constrained_keys:
                constrained_samples[k].append(np.asarray(constrained_i[k]))

        # Stack back into arrays
        samples_dict = {k: np.array(v) for k, v in constrained_samples.items()}

    param_names = sorted(samples_dict.keys())
    if "D_dt" in param_names:
        param_names = [p for p in param_names if p != "D_dt"] + ["D_dt"]

    arrays = []
    flat_names = []

    n_total = len(log_density)

    for key in param_names:
        arr = np.asarray(samples_dict[key])
        if arr.shape[0] != n_total:
            raise ValueError(f"Shape mismatch for {key}: expected {n_total}, got {arr.shape[0]}")

        if arr.ndim == 1:
            arrays.append(arr.reshape(-1, 1))
            flat_names.append(key)
        else:
            flat = arr.reshape(n_total, -1)
            arrays.append(flat)
            for k in range(flat.shape[1]):
                flat_names.append(f"{key}_{k}")

    flat_samples = np.hstack(arrays)

    if n_samples is not None and n_samples < n_total:
        rng = np.random.default_rng(random_seed)
        idxs = rng.choice(n_total, size=n_samples, replace=False)
        flat_samples = flat_samples[idxs]
        log_density = log_density[idxs]

    return {
        "engine": "nuts_numpyro",
        "samples": flat_samples,
        "log_likelihood": log_density,
        "log_weights": None,
        "param_names": flat_names,
    }


def get_posterior_summary(samples: Dict, key: str) -> Dict:
    """
    Get summary statistics for a single parameter.

    Parameters
    ----------
    samples : Dict
        Samples dictionary from run_nuts.
    key : str
        Parameter name.

    Returns
    -------
    Dict
        Summary with mean, std, median, quantiles.
    """
    vals = samples[key]
    if vals.ndim == 1:
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q16": float(np.percentile(vals, 16)),
            "q84": float(np.percentile(vals, 84)),
            "q2.5": float(np.percentile(vals, 2.5)),
            "q97.5": float(np.percentile(vals, 97.5)),
        }
    else:
        return {
            "mean": np.mean(vals, axis=0),
            "std": np.std(vals, axis=0),
        }
