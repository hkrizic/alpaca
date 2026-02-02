"""NUTS posterior processing.

Converts raw NUTS samples into the standardized posterior format,
including transformation from unconstrained to constrained space.

author: hkrizic
"""


import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed as numpyro_seed
from numpyro.infer.util import constrain_fn

from alpaca.sampler.utils import _get_rng, _standard_posterior_dict


def get_nuts_posterior(nuts_results: dict, prob_model=None, n_samples=None, random_seed=None):
    """Convert NUTS samples to standardized posterior container.

    Transforms NumPyro MCMC output into the standard posterior format,
    flattening multi-dimensional parameters for visualization and analysis.

    NUTS samples are in unconstrained space. If prob_model is provided,
    samples are transformed back to constrained (physical) space.

    Parameters
    ----------
    nuts_results : dict
        Output from run_nuts_numpyro or load_nuts_samples containing
        'samples' and 'log_density'.
    prob_model : herculens ProbModel or None
        The probabilistic model used for sampling. Required to transform
        samples to constrained space.
    n_samples : int or None
        Optional subsampling to fixed sample count.
    random_seed : int or None
        Seed for reproducible subsampling.

    Returns
    -------
    dict
        Standardized posterior container with equally-weighted samples
        in constrained (physical) space.
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
        constrained_samples = {k: [] for k in samples_dict.keys()}
        seeded_model = numpyro_seed(prob_model.model, rng_seed=0)

        # Vectorized transformation - process all samples at once
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
        rng = _get_rng(random_seed)
        idxs = rng.choice(n_total, size=n_samples, replace=False)
        flat_samples = flat_samples[idxs]
        log_density = log_density[idxs]

    return _standard_posterior_dict(
        engine="nuts_numpyro",
        samples=flat_samples,
        log_likelihood=log_density,
        log_weights=None,
        param_names=flat_names
    )
