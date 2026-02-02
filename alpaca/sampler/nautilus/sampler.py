"""Nautilus sampler execution and checkpoint management.

Provides functions to run importance nested sampling, manage timing
diagnostics, and restore posteriors from checkpoint files.

author: hkrizic
"""

import json
import os
import time

from nautilus import Prior, Sampler

from alpaca.sampler.utils import get_environment_info, now


class TimedLoglike:
    """Instrumented wrapper for likelihood functions.

    Records call count and cumulative evaluation time for performance
    profiling of nested sampling runs.
    """

    def __init__(self, loglike):
        """Initialize the timed likelihood wrapper.

        Parameters
        ----------
        loglike : callable
            Log-likelihood function to wrap. Must accept a parameter
            dictionary and return a scalar or array.
        """
        self.loglike = loglike
        self.n_calls = 0
        self.total_time = 0.0
        self.max_time = 0.0

    def __call__(self, sample_dict: dict):
        """Evaluate the wrapped log-likelihood with timing instrumentation.

        Parameters
        ----------
        sample_dict : dict
            Parameter dictionary passed through to the wrapped function.

        Returns
        -------
        float or ndarray
            Log-likelihood value(s) from the wrapped function.
        """
        t0 = now()
        val = self.loglike(sample_dict)
        dt = now() - t0
        self.n_calls += 1
        self.total_time += dt
        if dt > self.max_time:
            self.max_time = dt
        return val


def _save_nautilus_timing_json(timing_dict: dict, filepath: str) -> str:
    """Persist timing diagnostics to JSON alongside checkpoint.

    Parameters
    ----------
    timing_dict : dict
        Timing diagnostics to save.
    filepath : str
        Checkpoint filepath used to determine output location.

    Returns
    -------
    str
        Path to the saved timing JSON file.
    """
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


def run_nautilus(
    prior: Prior,
    loglike,
    n_live: int,
    filepath: str,
    resume: bool = False,
    verbose: bool = True,
    run_kwargs: dict = None,
    PARALLELIZE_OPTION: bool = False,
    NUMBER_OF_POOLS: int | None = None,
    n_batch: int | None = None,
):
    """Execute importance nested sampling with Nautilus.

    Performs nested sampling (Skilling 2004) using the neural network-
    accelerated Nautilus algorithm (Lange 2023). Computes posterior samples
    and the Bayesian evidence Z for model comparison.

    Parameters
    ----------
    prior : nautilus.Prior
        Prior distribution over model parameters.
    loglike : callable
        Log-likelihood function.
    n_live : int
        Number of live points controlling sampling resolution.
    filepath : str
        HDF5 checkpoint path for persistence and resumption.
    resume : bool
        Continue from existing checkpoint if available.
    verbose : bool
        Print progress messages.
    run_kwargs : dict or None
        Additional keyword arguments for sampler.run().
    PARALLELIZE_OPTION : bool
        Enable vectorized likelihood evaluation for GPU.
    NUMBER_OF_POOLS : int or None
        Worker pool size for parallelization.
    n_batch : int or None
        Batch size for vectorized evaluation.

    Returns
    -------
    tuple of (Sampler, ndarray, ndarray, ndarray)
        Sampler instance, structured array of posterior samples,
        log importance weights, and log-likelihood values.
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

    # Arguments for the NAUTILUS Sampler constructor
    sampler_kwargs = dict(
        n_live=n_live,
        filepath=filepath,
        resume=resume,
    )

    # --- Parallelisation / vectorisation configuration ---
    if PARALLELIZE_OPTION:
        # Vectorized mode: loglike accepts a dict of arrays
        sampler_kwargs["vectorized"] = True
        sampler_kwargs["pass_dict"] = True

        # For GPU-backed likelihoods it's usually best to keep *likelihood*
        # calls single-process (to avoid spawning multiple JAX contexts),
        # and only parallelise sampler internals.
        if NUMBER_OF_POOLS is not None:
            # (likelihood pool, sampler pool)
            sampler_kwargs["pool"] = (None, NUMBER_OF_POOLS)

        if n_batch is not None:
            sampler_kwargs["n_batch"] = n_batch

    else:
        # Classic CPU parallelisation: Multiprocessing pool calls scalar likelihood.
        if NUMBER_OF_POOLS is not None:
            sampler_kwargs["pool"] = NUMBER_OF_POOLS

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
            print(f"Checkpoint should be available at: {filepath}")

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
        sampler.timing["n_live"] = int(sampler.n_live)
    except Exception:
        sampler.timing.setdefault("n_live", int(n_live))
    try:
        sampler.timing["n_dim"] = int(sampler.n_dim)
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

    Parameters
    ----------
    prior : nautilus.Prior
        Prior distribution (must match checkpoint).
    loglike : callable
        Log-likelihood function (required for Sampler initialization).
    n_live : int
        Number of live points (must match checkpoint).
    filepath : str
        Path to HDF5 checkpoint file.

    Returns
    -------
    tuple of (Sampler, ndarray, ndarray, ndarray)
        Sampler instance, posterior samples, log weights, and log-likelihoods.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    sampler = Sampler(prior, loglike, n_live=n_live, filepath=filepath, resume=True)
    points, log_w, log_l = sampler.posterior()
    return sampler, points, log_w, log_l
