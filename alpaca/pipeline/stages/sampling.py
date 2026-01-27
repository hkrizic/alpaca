"""
alpaca.pipeline.stages.sampling

NUTS and Nautilus sampling stage implementations.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from herculens.Inference.loss import Loss

from alpaca.pipeline.io import _save_json
from alpaca.plotting.diagnostics import plot_nuts_diagnostics
from alpaca.sampler.nautilus import (
    build_nautilus_prior_and_loglike,
    get_nautilus_posterior,
    run_nautilus,
)
from alpaca.sampler.nuts import (
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
    """Run NUTS (NumPyro) sampling with parallel chains."""
    from numpyro.handlers import seed
    from numpyro.infer.util import constrain_fn, unconstrain_fn

    # Wrap model with seed handler to provide PRNG key for tracing
    # This is needed because NumPyro's log_density requires a PRNG key
    original_model = prob_model.model
    prob_model.model = seed(original_model, rng_seed=0)

    loss_obj = Loss(prob_model)

    if "D_dt" not in best_params:
        best_params = dict(best_params)
        best_params["D_dt"] = 0.5 * (500.0 + 10000.0)

    seeded_model = seed(original_model, rng_seed=0)
    use_time_delays = measured_delays is not None and delay_errors is not None
    if use_time_delays:
        measured_td_j = jnp.asarray(measured_delays)
        sigma2_td = jnp.asarray(delay_errors) ** 2
        const_td = jnp.log(2.0 * jnp.pi * sigma2_td)
        c_km_s = jnp.asarray(299792.458)

        kwargs_ref = prob_model.params2kwargs(best_params)
        kwargs_point = kwargs_ref.get("kwargs_point_source")
        if kwargs_point is None:
            raise ValueError("Time-delay data provided but no point sources found.")
        nps = int(np.asarray(kwargs_point[0]["ra"]).size)
        if measured_td_j.shape[0] != (nps - 1):
            raise ValueError(
                "measured_delays must have length n_images-1 (relative to image 0)."
            )

        mass_model = lens_image.MassModel

        def _potential_jax(x, y, kwargs_lens):
            potential = jnp.zeros_like(x)
            for i, func in enumerate(mass_model.func_list):
                potential = potential + func.function(x, y, **kwargs_lens[i])
            return potential

        def _fermat_potential_jax(x_image, y_image, kwargs_lens):
            potential = _potential_jax(x_image, y_image, kwargs_lens)
            x_source, y_source = mass_model.ray_shooting(
                x_image, y_image, kwargs_lens
            )
            geometry = 0.5 * ((x_image - x_source) ** 2 + (y_image - y_source) ** 2)
            return geometry - potential

    # Ray shooting consistency setup
    use_rayshoot = config.use_rayshoot_consistency
    use_source_pos_rayshoot = config.use_source_position_rayshoot
    # For Correlated Fields models, force use of mean (no src_center_x/y params)
    is_corr_field_model = hasattr(prob_model, "source_field")
    if is_corr_field_model and use_source_pos_rayshoot:
        use_source_pos_rayshoot = False
    # Use the model's setting, not config, since model determines what params exist
    use_rayshoot_sys = getattr(prob_model, "use_rayshoot_systematic_error", False)
    if use_rayshoot:
        sigma2_rayshoot_fixed = jnp.asarray(config.rayshoot_consistency_sigma ** 2)
        mass_model_rayshoot = lens_image.MassModel

    # Lens gamma prior setup
    # For uniform prior, no explicit term needed (flat)
    # For normal prior, add truncated normal log-density term
    use_lens_gamma_normal_prior = config.lens_gamma_prior_type == "normal"
    if use_lens_gamma_normal_prior:
        lens_gamma_mu = jnp.asarray(best_params["lens_gamma"])
        lens_gamma_sigma = jnp.asarray(config.lens_gamma_prior_sigma)
        lens_gamma_sigma2 = lens_gamma_sigma ** 2
        # Bounds are enforced by constrain_fn (no need to store separately)

    def logdensity_fn(params_unconstrained):
        ll = -loss_obj(params_unconstrained)
        if use_time_delays:
            params_constrained = constrain_fn(seeded_model, (), {}, params_unconstrained)
            D_dt = params_constrained["D_dt"]
            kwargs = prob_model.params2kwargs(params_constrained)
            ra = kwargs["kwargs_point_source"][0]["ra"]
            dec = kwargs["kwargs_point_source"][0]["dec"]
            phi = _fermat_potential_jax(ra, dec, kwargs["kwargs_lens"])
            delta_phi = phi[1:] - phi[0]
            dt_pred = (c_km_s / D_dt) * delta_phi
            resid_td = dt_pred - measured_td_j
            ll_td = -0.5 * jnp.sum(resid_td * resid_td / sigma2_td + const_td)
            ll = ll + ll_td
        if use_rayshoot:
            if not use_time_delays:
                params_constrained = constrain_fn(seeded_model, (), {}, params_unconstrained)
                kwargs = prob_model.params2kwargs(params_constrained)
            ra = kwargs["kwargs_point_source"][0]["ra"]
            dec = kwargs["kwargs_point_source"][0]["dec"]
            x_src, y_src = mass_model_rayshoot.ray_shooting(ra, dec, kwargs["kwargs_lens"])
            if use_source_pos_rayshoot:
                # Use sampled source position as reference
                x_src_ref = params_constrained["src_center_x"]
                y_src_ref = params_constrained["src_center_y"]
            else:
                # Use mean of ray-traced positions as reference (original behavior)
                x_src_ref = jnp.mean(x_src)
                y_src_ref = jnp.mean(y_src)
            scatter = (x_src - x_src_ref) ** 2 + (y_src - y_src_ref) ** 2
            # Compute effective sigma^2 (fixed + optional systematic)
            if use_rayshoot_sys:
                # Transform from log-space (the sampled parameter) to linear space
                log_sigma_sys = params_unconstrained["log_sigma_rayshoot_sys"]
                sigma_sys = jnp.exp(log_sigma_sys)
                sigma2_total = sigma2_rayshoot_fixed + sigma_sys ** 2
            else:
                sigma2_total = sigma2_rayshoot_fixed
            ll = ll - 0.5 * jnp.sum(scatter) / sigma2_total
        # Add lens_gamma normal prior term (if configured)
        # For uniform prior, nothing is added (flat prior in bounds)
        if use_lens_gamma_normal_prior:
            if not (use_time_delays or use_rayshoot):
                params_constrained = constrain_fn(seeded_model, (), {}, params_unconstrained)
            lens_gamma_val = params_constrained["lens_gamma"]
            # Truncated normal log-prior: Gaussian penalty term
            # (bounds already enforced by constrain_fn, so we just add Gaussian)
            ll_prior = -0.5 * (lens_gamma_val - lens_gamma_mu) ** 2 / lens_gamma_sigma2
            ll = ll + ll_prior
        return ll

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
    """Run Nautilus nested sampling."""
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
