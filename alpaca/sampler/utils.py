"""Runtime environment, timing, and posterior utilities shared across samplers.

author: hkrizic
"""

import getpass
import os
import platform
import socket
import time
from datetime import datetime

import jax
import numpy as np


def get_environment_info() -> dict[str, object]:
    """Record computational environment metadata for reproducibility.

    Returns
    -------
    dict
        JAX device config, library versions, CPU info,
        threading environment variables, and ISO 8601 timestamp.
    """
    info: dict[str, object] = {}

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

    try:
        devices = jax.devices()
        info["jax_device_count"] = len(devices)
        info["jax_platforms"] = sorted({d.platform for d in devices})
    except Exception:
        pass

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "JAX_NUM_THREADS",
    ):
        if var in os.environ:
            info[f"env_{var.lower()}"] = os.environ[var]

    info["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return info


def now() -> float:
    """Return high-resolution wall-clock time in seconds."""
    return time.perf_counter()


def _nautilus_points_to_array(points):
    """Convert Nautilus posterior samples to a homogeneous array.

    Parameters
    ----------
    points : ndarray
        Posterior samples from Sampler.posterior(), either structured
        array or standard ndarray.

    Returns
    -------
    tuple of (ndarray, list or None)
        Samples array of shape (N, D) and list of parameter names or None.
    """
    if hasattr(points, "dtype") and points.dtype.names is not None:
        names = list(points.dtype.names)
        samples = np.vstack([np.asarray(points[name], float) for name in names]).T
        return samples, names
    else:
        arr = np.asarray(points, float)
        return arr, None


def _get_rng(random_seed):
    """Initialize a NumPy random number generator.

    Parameters
    ----------
    random_seed : int or None
        Random seed for reproducibility. If None, uses global state.

    Returns
    -------
    numpy.random.RandomState or module
        Seeded RandomState instance or numpy.random module.
    """
    if random_seed is None:
        return np.random
    return np.random.RandomState(int(random_seed))


def _standard_posterior_dict(engine, samples, log_likelihood, log_weights,
                             param_names=None, extra=None):
    """Construct a standardized posterior sample container.

    Parameters
    ----------
    engine : str
        Inference algorithm identifier (e.g., "nautilus", "emcee").
    samples : array-like
        Parameter samples, shape (N, D).
    log_likelihood : array-like or None
        Log-likelihood at each sample, or None.
    log_weights : array-like or None
        Importance weights (nested sampling) or None for MCMC.
    param_names : list of str or None
        Parameter identifiers for labeling.
    extra : dict or None
        Additional metadata (evidence, diagnostics, timing).

    Returns
    -------
    dict
        Dictionary with keys: engine, samples, log_likelihood, log_weights,
        param_names, meta.
    """
    samples = np.asarray(samples, float)
    if log_likelihood is not None:
        log_likelihood = np.asarray(log_likelihood, float)
    if log_weights is not None:
        log_weights = np.asarray(log_weights, float)
    if param_names is not None:
        param_names = list(param_names)

    return dict(
        engine=str(engine),
        samples=samples,
        log_likelihood=log_likelihood,
        log_weights=log_weights,
        param_names=param_names,
        meta=extra or {},
    )


def _prepare_time_delay_inputs(measured_delays, delay_errors):
    """Validate and normalize time-delay measurements and errors.

    Parameters
    ----------
    measured_delays : array-like or None
        Observed time delays relative to image 0, or None.
    delay_errors : array-like or None
        1-sigma uncertainties on the delays, or None.

    Returns
    -------
    tuple of (ndarray, ndarray) or None
        Delays and errors as 1D numpy arrays, or None if inputs are None.
    """
    if measured_delays is None or delay_errors is None:
        return None
    delays = np.atleast_1d(np.asarray(measured_delays, float))
    errors = np.atleast_1d(np.asarray(delay_errors, float))
    if delays.ndim != 1 or errors.ndim != 1:
        raise ValueError("measured_delays and delay_errors must be 1D arrays.")
    if errors.size == 1 and delays.size > 1:
        errors = np.full_like(delays, float(errors.item()))
    if delays.shape != errors.shape:
        raise ValueError("measured_delays and delay_errors must have the same shape.")
    if np.any(errors <= 0):
        raise ValueError("delay_errors must be strictly positive.")
    return delays, errors
