"""Bayesian Information Criterion (BIC) computation utilities.

Provides functions to compute BIC from posterior samples or pipeline results,
used for model comparison in gravitational lens modeling.
"""


import numpy as np


def compute_bic(posterior: dict, n_pixels: int, n_params: int = None) -> float:
    """Compute the Bayesian Information Criterion from posterior samples.

    BIC = k * ln(N) - 2 * ln(L_max), where k is number of parameters,
    N is number of data points, L_max is maximum likelihood.

    Args:
        posterior: Standardized posterior dict with 'log_likelihood' and
            optionally 'param_names'.
        n_pixels: Number of pixels (data points).
        n_params: Number of free parameters. If None, inferred from param_names.

    Returns:
        BIC value (lower is better).

    Raises:
        ValueError: If posterior lacks log_likelihood field.
    """
    log_likelihood = posterior.get("log_likelihood")
    if log_likelihood is None:
        raise ValueError("Posterior must contain 'log_likelihood' field for BIC computation.")

    log_likelihood = np.asarray(log_likelihood)
    log_L_max = np.max(log_likelihood)

    if n_params is None:
        param_names = posterior.get("param_names", [])
        n_params = len(param_names)

    bic = n_params * np.log(n_pixels) - 2.0 * log_L_max
    return float(bic)


def compute_bic_from_results(results: dict) -> dict:
    """Compute BIC from pipeline results dictionary.

    Args:
        results: Results dict from run_pipeline with 'posterior' and 'setup'.

    Returns:
        Dictionary with bic, n_params, n_pixels, log_L_max, shapelets_n_max.

    Raises:
        ValueError: If results lacks required fields.
    """
    posterior = results.get("posterior")
    if posterior is None:
        raise ValueError("Results must contain 'posterior' from sampling.")

    setup = results.get("setup", {})
    img = setup.get("img")
    noise_map = setup.get("noise_map")

    if img is not None:
        n_pixels = int(np.sum(np.isfinite(img)))
    elif noise_map is not None:
        n_pixels = int(np.sum(np.isfinite(noise_map)))
    else:
        raise ValueError("Cannot determine n_pixels: 'img' or 'noise_map' not found in setup.")

    log_likelihood = np.asarray(posterior["log_likelihood"])
    log_L_max = float(np.max(log_likelihood))
    n_params = len(posterior.get("param_names", []))

    bic = compute_bic(posterior, n_pixels, n_params)

    config = results.get("config")
    shapelets_n_max = None
    if config is not None:
        shapelets_n_max = getattr(config, "shapelets_n_max", None)

    return {
        "bic": bic,
        "n_params": n_params,
        "n_pixels": n_pixels,
        "log_L_max": log_L_max,
        "shapelets_n_max": shapelets_n_max,
    }
