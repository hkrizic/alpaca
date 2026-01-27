"""Runtime environment, timing, and posterior utilities shared across samplers."""

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

    Args:
        None

    Returns:
        Dictionary with JAX device config, library versions, CPU info,
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


def _get_rng(random_seed):
    """Initialize a NumPy random number generator.

    Args:
        random_seed: Random seed for reproducibility. If None, uses global state.

    Returns:
        Seeded RandomState instance or numpy.random module.
    """
    if random_seed is None:
        return np.random
    return np.random.RandomState(int(random_seed))


def _standard_posterior_dict(engine, samples, log_likelihood, log_weights,
                             param_names=None, extra=None):
    """Construct a standardized posterior sample container.

    Args:
        engine: Inference algorithm identifier (e.g., "nautilus", "emcee").
        samples: Parameter samples, shape (N, D).
        log_likelihood: Log-likelihood at each sample, or None.
        log_weights: Importance weights (nested sampling) or None for MCMC.
        param_names: Parameter identifiers for labeling.
        extra: Additional metadata (evidence, diagnostics, timing).

    Returns:
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


def _vector_to_paramdict(theta: np.ndarray, param_names):
    """Map parameter vector to named dictionary."""
    return {name: float(val) for name, val in zip(param_names, theta)}
