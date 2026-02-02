"""
alpaca.pipeline.stages.sampling

Pipeline wrappers that set up and run NUTS or Nautilus sampling.

``_run_nuts_sampling()`` delegates log-density construction to
``alpaca.sampler.nuts.likelihood`` (``build_nuts_logdensity``), which combines
the Herculens Loss object with time-delay, ray-shooting, and lens_gamma
prior terms.

``_run_nautilus_sampling()`` delegates to ``alpaca/sampler/nautilus/`` which has
its own likelihood builder (``nautilus/likelihood.py``) and prior construction
(``nautilus/prior.py``).

author: hkrizic
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from alpaca.pipeline.io import _save_json
from alpaca.plotting.diagnostics import plot_nuts_diagnostics
from alpaca.sampler.nautilus import (
    build_nautilus_prior_and_loglike,
    get_nautilus_posterior,
    run_nautilus,
)
from alpaca.sampler.nuts import (
    build_nuts_logdensity,
    get_nuts_posterior,
    run_nuts_numpyro,
)


def _run_nuts_sampling(
    prob_model,
    best_params: dict,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    config,
    dirs: dict[str, str],
    verbose: bool,
    *,
    measured_delays: np.ndarray | None = None,
    delay_errors: np.ndarray | None = None,
) -> dict:
    """
    Run NUTS (NumPyro) sampling with parallel or vectorized chains.

    This function builds the log-density from the probabilistic model,
    constructs initial positions in unconstrained space by perturbing the
    MAP estimate, runs NumPyro NUTS, generates diagnostic plots, and
    converts the raw chains to a standardised posterior dictionary which
    is saved to disk.

    Parameters
    ----------
    prob_model : ProbModel or ProbModelCorrField
        Probabilistic model providing parameter transformations and priors.
    best_params : dict
        Best-fit parameter dictionary from multi-start optimisation.
    lens_image : LensImage
        Herculens ``LensImage`` used for forward modelling.
    img : np.ndarray
        2-D observed lens image.
    noise_map : np.ndarray
        2-D noise map (same shape as *img*).
    config : SamplerConfig
        Sampler configuration dataclass with NUTS-specific settings
        (number of warmup steps, samples, chains, target acceptance, etc.).
    dirs : dict of str
        Output directory mapping produced by ``_make_output_structure``.
    verbose : bool
        If ``True``, print progress information.
    measured_delays : np.ndarray, optional
        Measured time delays (relative to the first image).
    delay_errors : np.ndarray, optional
        1-sigma errors on *measured_delays*.

    Returns
    -------
    dict
        Standardised posterior dictionary with keys ``"samples"``,
        ``"param_names"``, ``"engine"``, ``"log_likelihood"``, etc.
    """
    from numpyro.infer.util import unconstrain_fn

    # Build log-density (handles seeding, Loss, all likelihood terms)
    logdensity_fn, seeded_model, original_model = build_nuts_logdensity(
        prob_model, best_params, lens_image, config,
        measured_delays=measured_delays, delay_errors=delay_errors,
    )

    num_chains = config.nuts_num_chains
    if num_chains is None:
        num_chains = jax.device_count()

    if verbose:
        print(f"Running NumPyro NUTS with {num_chains} chains on {jax.device_count()} devices")

    # Convert best_params to unconstrained space
    best_params_jax = {k: jnp.array(v) for k, v in best_params.items()}
    u_best = unconstrain_fn(seeded_model, (), {}, best_params_jax)

    # Create initial positions with small perturbation for each chain
    rng_init = jax.random.PRNGKey(config.random_seed)

    def expand_for_chains(key, u):
        """
        Replicate a single unconstrained parameter for all chains with small noise.

        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key for generating perturbations.
        u : jax.Array
            Unconstrained parameter value (scalar or 1-D).

        Returns
        -------
        jax.Array
            Array of shape ``(num_chains,) + u.shape`` with independent
            Gaussian perturbations of scale 0.01 added.
        """
        noise_keys = jax.random.split(key, num_chains)
        if u.ndim == 0:
            noise = jax.vmap(lambda k: jax.random.normal(k) * 0.01)(noise_keys)
            return u + noise
        else:
            noise = jax.vmap(lambda k: jax.random.normal(k, shape=u.shape) * 0.01)(noise_keys)
            return u + noise

    init_keys = jax.random.split(rng_init, len(u_best))
    initial_positions = {}
    for i, (name, val) in enumerate(u_best.items()):
        initial_positions[name] = expand_for_chains(init_keys[i], val)

    # Determine chain method based on device count
    n_devices = jax.device_count()
    if n_devices > 1 and num_chains >= n_devices:
        chain_method = "parallel"
    else:
        chain_method = "vectorized"

    if verbose:
        print(f"Using chain_method='{chain_method}' for NUTS sampling")

    # Run NUTS with NumPyro
    nuts_result = run_nuts_numpyro(
        logdensity_fn,
        initial_positions,
        num_warmup=config.nuts_num_warmup,
        num_samples=config.nuts_num_samples,
        num_chains=num_chains,
        seed=config.random_seed,
        outdir=dirs["sampling_chains"],
        verbose=verbose,
        target_accept_prob=config.nuts_target_accept,
        chain_method=chain_method,
        progress_bar=True,
    )

    # Generate and save NUTS diagnostic plots
    if verbose:
        print("Generating NUTS diagnostic plots...")
    try:
        plot_nuts_diagnostics(
            nuts_result,
            prob_model=prob_model,
            param_names=["lens_theta_E", "lens_gamma", "lens_e1", "lens_e2", "light_Re_L", "light_n_L"],
            max_params=6,
            rolling_window=50,
            outdir=dirs["sampling_chains"],
            dpi=150,
        )
        if verbose:
            print(f"NUTS diagnostics saved to {dirs['sampling_chains']}")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate NUTS diagnostics: {e}")

    # Convert to standard posterior format (transforms to constrained/physical space)
    posterior = get_nuts_posterior(nuts_result, prob_model=prob_model,
                                   n_samples=config.n_posterior_samples,
                                   random_seed=config.random_seed)

    # Save posterior
    _save_json(os.path.join(dirs["posterior"], "posterior_summary.json"), {
        "engine": posterior["engine"],
        "n_samples": posterior["samples"].shape[0],
        "n_params": posterior["samples"].shape[1],
        "param_names": posterior["param_names"],
    })
    np.savez_compressed(
        os.path.join(dirs["posterior"], "posterior_samples.npz"),
        samples=posterior["samples"],
        param_names=posterior["param_names"],
    )

    # Restore original model to avoid side effects
    prob_model.model = original_model

    return posterior


def _run_nautilus_sampling(
    best_params: dict,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    config,
    dirs: dict[str, str],
    verbose: bool,
    *,
    measured_delays: np.ndarray | None = None,
    delay_errors: np.ndarray | None = None,
    use_corr_fields: bool = False,
) -> dict:
    """
    Run Nautilus nested sampling.

    This function builds the prior and log-likelihood from the MAP estimate,
    runs the Nautilus sampler, converts the weighted samples to a
    standardised posterior dictionary, and saves the results to disk.

    Parameters
    ----------
    best_params : dict
        Best-fit parameter dictionary from multi-start optimisation.
    lens_image : LensImage
        Herculens ``LensImage`` used for forward modelling.
    img : np.ndarray
        2-D observed lens image.
    noise_map : np.ndarray
        2-D noise map (same shape as *img*).
    config : SamplerConfig
        Sampler configuration dataclass with Nautilus-specific settings
        (number of live points, batch size, JAX options, etc.).
    dirs : dict of str
        Output directory mapping produced by ``_make_output_structure``.
    verbose : bool
        If ``True``, print progress information.
    measured_delays : np.ndarray, optional
        Measured time delays (relative to the first image).
    delay_errors : np.ndarray, optional
        1-sigma errors on *measured_delays*.
    use_corr_fields : bool, optional
        Whether the source model uses Correlated Fields.

    Returns
    -------
    dict
        Standardised posterior dictionary with keys ``"samples"``,
        ``"param_names"``, ``"engine"``, ``"log_likelihood"``,
        ``"meta"`` (including ``"log_evidence"``), etc.
    """
    # Build prior and likelihood
    prior, paramdict_to_kwargs, loglike = build_nautilus_prior_and_loglike(
        best_params,
        lens_image,
        img,
        noise_map,
        use_uniform_for_bounded=config.nautilus_use_uniform_priors,
        use_jax=config.nautilus_use_jax,
        use_multi_device=config.nautilus_use_multi_device,
        measured_delays=measured_delays,
        delay_errors=delay_errors,
        use_rayshoot_consistency=config.use_rayshoot_consistency,
        rayshoot_consistency_sigma=config.rayshoot_consistency_sigma,
        use_source_position_rayshoot=config.use_source_position_rayshoot,
        use_rayshoot_systematic_error=config.use_rayshoot_systematic_error,
        rayshoot_sys_error_min=config.rayshoot_sys_error_min,
        rayshoot_sys_error_max=config.rayshoot_sys_error_max,
        lens_gamma_prior_type=config.lens_gamma_prior_type,
        lens_gamma_prior_low=config.lens_gamma_prior_low,
        lens_gamma_prior_high=config.lens_gamma_prior_high,
        lens_gamma_prior_sigma=config.lens_gamma_prior_sigma,
        use_corr_fields=use_corr_fields,
        use_image_pos_offset=config.use_image_pos_offset,
        image_pos_offset_sigma=config.image_pos_offset_sigma,
    )

    # Checkpoint path
    ckpt_path = os.path.join(dirs["sampling_chains"], "nautilus_checkpoint.hdf5")

    # Determine batch size
    if config.nautilus_use_jax:
        n_dev = max(1, jax.device_count())
        n_batch = config.nautilus_batch_per_device * n_dev
        if verbose:
            print(f"Using JAX with {n_dev} device(s); n_batch = {n_batch}")
    else:
        n_batch = None

    # Run Nautilus
    sampler, points, log_w, log_l = run_nautilus(
        prior=prior,
        loglike=loglike,
        n_live=config.nautilus_n_live,
        filepath=ckpt_path,
        resume=False,
        verbose=verbose,
        PARALLELIZE_OPTION=config.nautilus_use_jax,
        NUMBER_OF_POOLS=None,
        n_batch=n_batch,
    )

    # Convert to standard posterior format
    posterior = get_nautilus_posterior(
        sampler, points, log_w, log_l,
        n_samples=config.n_posterior_samples,
        random_seed=config.random_seed,
        use_weights=True,
    )

    # Save posterior
    _save_json(os.path.join(dirs["posterior"], "posterior_summary.json"), {
        "engine": posterior["engine"],
        "n_samples": posterior["samples"].shape[0],
        "n_params": posterior["samples"].shape[1],
        "param_names": posterior["param_names"],
        "log_evidence": posterior["meta"].get("log_evidence"),
    })
    np.savez_compressed(
        os.path.join(dirs["posterior"], "posterior_samples.npz"),
        samples=posterior["samples"],
        param_names=posterior["param_names"],
    )

    return posterior
