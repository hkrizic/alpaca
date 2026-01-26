"""
Nautilus nested sampling for posterior inference.

Uses the model's likelihood terms for nested sampling, providing
evidence estimation and posterior samples.
"""

from typing import Dict
from time import time as now
import time
import os
import json
import platform
import socket
import getpass

import numpy as np
import jax

from nautilus import Prior, Sampler

from herculens.Inference.loss import Loss
import jax.numpy as jnp


def get_environment_info() -> Dict:
    """Record computational environment metadata for reproducibility."""
    info = {}
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()
    info["hostname"] = socket.gethostname()
    try:
        info["user"] = getpass.getuser()
    except Exception:
        pass
    info["pid"] = os.getpid()
    info["cpu_count"] = os.cpu_count()
    try:
        info["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        info["jax_version"] = jax.__version__
    except Exception:
        pass
    return info


class TimedLoglike:
    """Instrumented wrapper for likelihood functions.

    Records call count and cumulative evaluation time for performance
    profiling of nested sampling runs.
    """

    def __init__(self, loglike):
        self.loglike = loglike
        self.n_calls = 0
        self.total_time = 0.0
        self.max_time = 0.0

    def __call__(self, sample_dict: Dict):
        t0 = now()
        val = self.loglike(sample_dict)
        dt = now() - t0
        self.n_calls += 1
        self.total_time += dt
        if dt > self.max_time:
            self.max_time = dt
        return val


def _save_nautilus_timing_json(timing_dict: Dict, filepath: str) -> str:
    """Persist timing diagnostics to JSON alongside checkpoint."""
    base_dir = os.path.dirname(os.path.abspath(filepath))
    log_dir = os.path.join(base_dir, "logs", "benchmark_timing")
    os.makedirs(log_dir, exist_ok=True)

    ckpt_name = os.path.splitext(os.path.basename(filepath))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"timing_{ckpt_name}_{timestamp}.json"
    out_path = os.path.join(log_dir, out_name)

    payload = {
        "checkpoint": os.path.abspath(filepath),
        "timestamp": timestamp,
        "timing": timing_dict,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


def _nautilus_points_to_array(points):
    """Convert Nautilus posterior samples to a homogeneous array.

    Args:
        points: Posterior samples from Sampler.posterior(), either structured
            array or standard ndarray.

    Returns:
        Tuple of (samples, names) where samples is shape (N, D) and names
        is a list of parameter names or None.
    """
    if hasattr(points, "dtype") and points.dtype.names is not None:
        names = list(points.dtype.names)
        samples = np.vstack([np.asarray(points[name], float) for name in names]).T
        return samples, names
    else:
        arr = np.asarray(points, float)
        return arr, None


def run_nautilus(
    prior: Prior,
    loglike,
    n_live: int,
    filepath: str,
    resume: bool = False,
    verbose: bool = True,
    run_kwargs: Dict = None,
    vectorized: bool = False,
    n_batch: int = None,
    pool: int = None,
):
    """Execute importance nested sampling with Nautilus.

    Performs nested sampling (Skilling 2004) using the neural network-
    accelerated Nautilus algorithm (Lange 2023). Computes posterior samples
    and the Bayesian evidence Z for model comparison.

    Args:
        prior: Prior distribution over model parameters (nautilus.Prior).
        loglike: Log-likelihood function.
        n_live: Number of live points controlling sampling resolution.
        filepath: HDF5 checkpoint path for persistence and resumption.
        resume: Continue from existing checkpoint if available.
        verbose: Print progress messages.
        run_kwargs: Additional keyword arguments for sampler.run().
        vectorized: Enable vectorized likelihood evaluation for GPU.
        n_batch: Batch size for vectorized evaluation.
        pool: Worker pool size for parallelization.

    Returns:
        Tuple of (sampler, points, log_w, log_l) where sampler is the
        nautilus.Sampler instance, points is a structured array of posterior
        samples, log_w is the array of log importance weights, and log_l
        is the array of log-likelihood values.
    """
    # HDF5 locking handling
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    timed_loglike = TimedLoglike(loglike)

    if run_kwargs is None:
        run_kwargs = {}
    else:
        run_kwargs = dict(run_kwargs)

    run_kwargs.setdefault("verbose", verbose)

    # Arguments for the Nautilus Sampler constructor
    sampler_kwargs = dict(
        n_live=n_live,
        filepath=filepath,
        resume=resume,
    )

    # Parallelization / vectorization configuration
    if vectorized:
        sampler_kwargs["vectorized"] = True
        sampler_kwargs["pass_dict"] = True
        if pool is not None:
            sampler_kwargs["pool"] = (None, pool)
        if n_batch is not None:
            sampler_kwargs["n_batch"] = n_batch
    else:
        if pool is not None:
            sampler_kwargs["pool"] = pool

    sampler = Sampler(
        prior,
        timed_loglike,
        **sampler_kwargs,
    )

    t0 = now()
    try:
        success = sampler.run(**run_kwargs)
        t1 = now()
        interrupted = False
    except KeyboardInterrupt:
        t1 = now()
        interrupted = True
        success = False
        if verbose:
            print("[KeyboardInterrupt] NAUTILUS sampling interrupted by user.")
            print(f"Checkpoint is available at: {filepath}")

    total_runtime = t1 - t0
    overhead = total_runtime - timed_loglike.total_time

    if timed_loglike.n_calls > 0:
        avg_call = timed_loglike.total_time / timed_loglike.n_calls
    else:
        avg_call = 0.0

    # Attach timing info to sampler for later inspection
    sampler.timing = dict(
        total_runtime=float(total_runtime),
        n_loglike_calls=int(timed_loglike.n_calls),
        loglike_total=float(timed_loglike.total_time),
        loglike_max=float(timed_loglike.max_time),
        avg_loglike=float(avg_call),
        sampler_overhead=float(overhead),
        calls_per_second=(
            float(timed_loglike.n_calls / total_runtime)
            if total_runtime > 0 else 0.0
        ),
        success=bool(success),
    )

    # Include sampler configuration if available
    try:
        sampler.timing["n_live"] = int(getattr(sampler, "n_live"))
    except Exception:
        sampler.timing.setdefault("n_live", int(n_live))
    try:
        sampler.timing["n_dim"] = int(getattr(sampler, "n_dim"))
    except Exception:
        pass

    # Attach environment snapshot
    sampler.timing.update(get_environment_info())

    # Persist timing information to disk next to the checkpoint
    log_path = _save_nautilus_timing_json(sampler.timing, filepath)

    # Get posterior from whatever is in the checkpoint / sampler state
    points, log_w, log_l = sampler.posterior()

    if verbose:
        if interrupted:
            print(
                f"NAUTILUS Sampling interrupted. "
                f"Elapsed time: {total_runtime / 60:.2f} minutes "
                f"({total_runtime:.2f} seconds)"
            )
        else:
            print(
                f"NAUTILUS Sampling complete! "
                f"Runtime: {total_runtime / 60:.2f} minutes "
                f"({total_runtime:.2f} seconds)"
            )

        print(
            f"  loglike: {timed_loglike.n_calls} calls, "
            f"total {timed_loglike.total_time:.2f} s, "
            f"avg {avg_call:.4f} s/call, "
            f"max {timed_loglike.max_time:.4f} s"
        )
        print(f"  sampler overhead (proposals + I/O etc.): {overhead:.2f} s")
        print(f"  timing benchmarks saved to: {log_path}")

    return sampler, points, log_w, log_l


def load_posterior_from_checkpoint(
    prior: Prior,
    loglike,
    n_live: int,
    filepath: str,
):
    """Restore Nautilus posterior from checkpoint without additional sampling.

    Args:
        prior: Prior distribution (must match checkpoint).
        loglike: Log-likelihood function (required for Sampler initialization).
        n_live: Number of live points (must match checkpoint).
        filepath: Path to HDF5 checkpoint file.

    Returns:
        Tuple of (sampler, points, log_w, log_l).
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    sampler = Sampler(prior, loglike, n_live=n_live, filepath=filepath, resume=True)
    points, log_w, log_l = sampler.posterior()
    return sampler, points, log_w, log_l


def get_nautilus_posterior(
    sampler,
    points,
    log_w,
    log_l,
    n_samples=None,
    random_seed=None,
    use_weights=True,
):
    """Convert Nautilus output to standardized posterior container.

    Transforms weighted nested sampling output into the standard posterior
    format, with optional importance resampling to generate equally-weighted
    samples suitable for corner plots and summary statistics.

    Args:
        sampler: Sampler instance (for evidence extraction).
        points: Raw posterior samples from sampler.posterior().
        log_w: Log importance weights.
        log_l: Log-likelihood values.
        n_samples: Number of samples after resampling. If None, returns weighted.
        random_seed: Seed for reproducible resampling.
        use_weights: Use importance weights for resampling.

    Returns:
        Standardized posterior container dictionary.
    """
    samples, param_names = _nautilus_points_to_array(points)
    log_w = np.asarray(log_w, float).ravel()
    log_l = np.asarray(log_l, float).ravel()

    n_total = samples.shape[0]
    if log_w.shape[0] != n_total or log_l.shape[0] != n_total:
        raise ValueError("Inconsistent shapes between points, log_w, and log_l.")

    if random_seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(int(random_seed))

    # Resample if requested
    if n_samples is not None:
        n_samples = int(n_samples)
        if use_weights:
            # Importance resampling using the NAUTILUS weights
            w = np.exp(log_w - np.max(log_w))
            w_sum = w.sum()
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                p = None
            else:
                p = w / w_sum
            idx = rng.choice(n_total, size=n_samples, replace=True, p=p)
        else:
            replace = n_samples > n_total
            idx = rng.choice(n_total, size=n_samples, replace=replace)

        samples = samples[idx]
        log_l = log_l[idx]
        log_w_out = np.zeros(samples.shape[0], dtype=float)
    else:
        log_w_out = log_w

    # Try to get log-evidence if available
    log_evidence = None
    if hasattr(sampler, "evidence"):
        try:
            log_evidence = float(sampler.evidence())
        except Exception:
            pass

    return {
        "engine": "nautilus",
        "samples": samples,
        "log_likelihood": log_l,
        "log_weights": log_w_out,
        "param_names": param_names,
        "log_evidence": log_evidence,
        "n_raw_samples": int(n_total),
        "timing": getattr(sampler, "timing", None),
    }


def build_prior_and_loglike(prob_model, use_vmap: bool = True, use_pmap: bool = True):
    """Build nautilus Prior and log-likelihood from ProbModel.

    The log-likelihood is computed using the Loss class from herculens,
    ensuring consistency with gradient descent optimization.

    When use_vmap=True (default), the likelihood is vectorized using JAX vmap
    for efficient batched evaluation. When use_pmap=True and multiple devices
    are available, evaluation is parallelized across devices using JAX pmap.

    Args:
        prob_model: Probabilistic model with prior_config defining parameter bounds.
        use_vmap: Use JAX vmap for vectorized likelihood evaluation.
        use_pmap: Use JAX pmap for multi-device parallelization (requires use_vmap=True).

    Returns:
        Tuple of (prior, loglike) where prior is a nautilus.Prior and
        loglike is a callable accepting a parameter dict (scalar or batched).

    Example:
        prior, loglike = build_prior_and_loglike(prob_model)

        # When using vectorized loglike, set vectorized=True in run_nautilus
        sampler, points, log_w, log_l = run_nautilus(
            prior, loglike, n_live=1000, filepath="checkpoint.hdf5",
            vectorized=True,  # Required for batched evaluation
        )
    """
    # Build prior from prob_model's prior_config
    prior = Prior()
    for name, bounds in prob_model.prior_config.items():
        lo, hi = bounds
        prior.add_parameter(name, dist=(lo, hi))

    # Build log-likelihood from Loss
    loss_obj = Loss(prob_model)

    def loglike_single(params: Dict) -> float:
        """Scalar log-likelihood function (negative loss)."""
        return -loss_obj(params)

    if not use_vmap:
        # Return simple scalar likelihood
        def loglike(params: Dict) -> float:
            try:
                return float(loglike_single(params))
            except Exception:
                return -1e30
        return prior, loglike

    # Build vectorized likelihood
    loglike_single_jit = jax.jit(loglike_single)
    loglike_vmap = jax.jit(jax.vmap(loglike_single_jit))

    n_devices = jax.device_count()

    # Multi-device sharding (pmap)
    if use_pmap and n_devices > 1:
        def _loglike_chunk(chunk_params_dict: Dict):
            return jax.vmap(loglike_single_jit)(chunk_params_dict)
        loglike_chunk_pmap = jax.pmap(_loglike_chunk)
    else:
        loglike_chunk_pmap = None

    def _prepare_batched_params(sample_dict: Dict):
        """Convert dict-of-numpy to dict-of-JAX with leading batch axis."""
        jax_params = {k: jnp.asarray(v) for k, v in sample_dict.items()}

        # Detect batch size B
        B = None
        for v in jax_params.values():
            if v.ndim > 0:
                B = v.shape[0]
                break

        if B is None:
            return None, jax_params

        def to_batched(x):
            x = jnp.asarray(x)
            if x.ndim == 0:
                return jnp.broadcast_to(x, (B,))
            elif x.ndim == 1:
                if x.shape[0] != B:
                    raise ValueError("Inconsistent batch sizes in loglike.")
                return x
            else:
                # For array parameters, ensure first dim is batch
                return x

        batched = {k: to_batched(v) for k, v in jax_params.items()}
        return B, batched

    def loglike(sample_dict: Dict):
        """
        Accepts scalar or batched dicts (numpy / Python scalars),
        returns float or 1D numpy array of log-likelihoods.
        """
        try:
            B, batched_params = _prepare_batched_params(sample_dict)

            # Scalar call: no batch dimension
            if B is None:
                ll = loglike_single_jit(batched_params)
                return float(ll)

            # Batched call
            if loglike_chunk_pmap is not None and B >= n_devices:
                # Multi-device path: shard batch across devices
                B_per = (B + n_devices - 1) // n_devices
                B_pad = B_per * n_devices
                pad = B_pad - B

                def pad_leaf(x):
                    if pad == 0:
                        return x
                    # Repeat last element to pad
                    if x.ndim == 1:
                        return jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)])
                    else:
                        return jnp.concatenate([x, jnp.repeat(x[-1:], pad, axis=0)])

                padded = {k: pad_leaf(v) for k, v in batched_params.items()}

                def reshape_leaf(x):
                    if x.ndim == 1:
                        return x.reshape((n_devices, B_per))
                    else:
                        return x.reshape((n_devices, B_per) + x.shape[1:])

                sharded = {k: reshape_leaf(v) for k, v in padded.items()}

                # pmap over devices, vmap inside each
                ll_sharded = loglike_chunk_pmap(sharded)
                ll_flat = ll_sharded.reshape((B_pad,))
                ll_final = ll_flat[:B]
            else:
                # Single-device vmap
                ll_final = loglike_vmap(batched_params)

            return np.asarray(ll_final, dtype=np.float64)

        except Exception:
            # Return very low likelihood for failures
            B, _ = _prepare_batched_params(sample_dict)
            if B is None:
                return -1e30
            return np.full(B, -1e30, dtype=np.float64)

    return prior, loglike
