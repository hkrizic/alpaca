"""Posterior processing for Nautilus nested sampling output.

Converts weighted nested sampling results into standardized posterior
containers with optional importance resampling.
"""

import numpy as np

from alpaca.sampler.utils import _get_rng, _nautilus_points_to_array, _standard_posterior_dict


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

    rng = _get_rng(random_seed)

    # Resample if requested
    if n_samples is not None:
        n_samples = int(n_samples)
        if use_weights:
            # Importance resampling using the NAUTILUS weights
            w = np.exp(log_w - np.max(log_w))
            w_sum = w.sum()
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                # Fall back to uniform if something is off
                p = None
            else:
                p = w / w_sum
            idx = rng.choice(n_total, size=n_samples, replace=True, p=p)
        else:
            # Uniform subsampling from the available points
            replace = n_samples > n_total
            idx = rng.choice(n_total, size=n_samples, replace=replace)

        samples = samples[idx]
        log_l = log_l[idx]
        # After resampling, treat samples as unweighted
        log_w_out = np.zeros(samples.shape[0], dtype=float)
    else:
        # Keep the original weighted set
        log_w_out = log_w

    # Try to get log-evidence if available
    log_evidence = None
    if hasattr(sampler, "logz"):
        try:
            log_evidence = float(np.array(sampler.logz).ravel()[-1])
        except Exception:
            log_evidence = None

    extra = dict(
        n_raw_samples=int(n_total),
        n_samples=int(samples.shape[0]),
        log_evidence=log_evidence,
        timing=getattr(sampler, "timing", None),
        raw_log_weights_shape=log_w.shape,
    )

    # Keep the original log-weights also in the metadata, for advanced use
    extra["raw_log_weights"] = log_w

    return _standard_posterior_dict(
        engine="nautilus",
        samples=samples,
        log_likelihood=log_l,
        log_weights=log_w_out,
        param_names=param_names,
        extra=extra,
    )
