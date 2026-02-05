"""NUTS log-density construction for Hamiltonian Monte Carlo sampling.

Builds the NUTS log-density function combining imaging likelihood (via
Herculens Loss), time-delay cosmography, and ray-shooting consistency
terms. Seeds the probabilistic model so that NumPyro's constrain_fn /
unconstrain_fn can trace through it.

author: hkrizic
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _build_fermat_potential_fn(mass_model):
    """Return a Fermat-potential callable for the given mass model.

    Parameters
    ----------
    mass_model : herculens MassModel
        ``lens_image.MassModel`` — provides ``func_list`` and
        ``ray_shooting``.

    Returns
    -------
    callable
        ``fermat_potential(x_image, y_image, kwargs_lens) -> jnp.ndarray``
    """

    def _potential_jax(x, y, kwargs_lens):
        """Compute the lensing potential at image positions.

        Parameters
        ----------
        x : jax.Array
            x-coordinates of image positions.
        y : jax.Array
            y-coordinates of image positions.
        kwargs_lens : list of dict
            Keyword arguments for each lens mass component.

        Returns
        -------
        jax.Array
            Lensing potential evaluated at the given positions.
        """
        potential = jnp.zeros_like(x)
        for i, func in enumerate(mass_model.func_list):
            potential = potential + func.function(x, y, **kwargs_lens[i])
        return potential

    def _fermat_potential_jax(x_image, y_image, kwargs_lens):
        """Compute the Fermat potential at image positions.

        Parameters
        ----------
        x_image : jax.Array
            x-coordinates of image positions.
        y_image : jax.Array
            y-coordinates of image positions.
        kwargs_lens : list of dict
            Keyword arguments for each lens mass component.

        Returns
        -------
        jax.Array
            Fermat potential at the given image positions.
        """
        potential = _potential_jax(x_image, y_image, kwargs_lens)
        x_source, y_source = mass_model.ray_shooting(
            x_image, y_image, kwargs_lens
        )
        geometry = 0.5 * ((x_image - x_source) ** 2 + (y_image - y_source) ** 2)
        return geometry - potential

    return _fermat_potential_jax


def build_nuts_logdensity(
    prob_model,
    best_params: dict,
    lens_image,
    config,
    *,
    measured_delays: np.ndarray | None = None,
    delay_errors: np.ndarray | None = None,
) -> tuple[callable, object, object]:
    """Build the log-density closure used by NUTS sampling.

    This function seeds ``prob_model.model``, constructs the Herculens
    ``Loss`` object, and assembles all additional likelihood terms
    (time-delay, ray-shooting consistency).

    Parameters
    ----------
    prob_model : ProbModel | ProbModelCorrField
        The probabilistic lens model whose ``.model`` will be wrapped
        with a NumPyro seed handler.
    best_params : dict
        MAP parameter dictionary from gradient descent.
    lens_image : herculens LensImage
        Provides the mass model for Fermat potential / ray-shooting.
    config : SamplerConfig
        Pipeline sampler configuration.
    measured_delays : array, optional
        Measured time delays (relative to image 0).
    delay_errors : array, optional
        1-sigma uncertainties on the measured delays.

    Returns
    -------
    logdensity_fn : callable
        ``logdensity_fn(params_unconstrained) -> float`` — the closure
        to pass to ``run_nuts_numpyro``.
    seeded_model : numpyro model
        The seeded model needed by the caller for ``unconstrain_fn``
        (initial position computation).
    original_model : callable
        The original ``prob_model.model`` so the caller can restore it
        after sampling.
    """
    from herculens.Inference.loss import Loss
    from numpyro.handlers import seed
    from numpyro.infer.util import constrain_fn

    # ------------------------------------------------------------------
    # Seed the model and build the base imaging loss
    # ------------------------------------------------------------------
    original_model = prob_model.model
    prob_model.model = seed(original_model, rng_seed=0)

    loss_obj = Loss(prob_model)

    if "D_dt" not in best_params:
        best_params = dict(best_params)
        best_params["D_dt"] = 0.5 * (500.0 + 10000.0)

    seeded_model = seed(original_model, rng_seed=0)

    # ------------------------------------------------------------------
    # Time-delay setup
    # ------------------------------------------------------------------
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
        _fermat_potential_jax = _build_fermat_potential_fn(mass_model)

    # ------------------------------------------------------------------
    # Ray-shooting consistency setup
    # ------------------------------------------------------------------
    use_rayshoot = config.use_rayshoot_consistency
    use_source_pos_rayshoot = config.use_source_position_rayshoot
    is_corr_field_model = hasattr(prob_model, "source_field")
    if is_corr_field_model and use_source_pos_rayshoot:
        use_source_pos_rayshoot = False
    use_rayshoot_sys = getattr(prob_model, "use_rayshoot_systematic_error", False)
    use_image_pos_offset = getattr(prob_model, "use_image_pos_offset", False)
    if use_rayshoot:
        sigma2_rayshoot_fixed = jnp.asarray(config.rayshoot_consistency_sigma ** 2)
        mass_model_rayshoot = lens_image.MassModel

    # ------------------------------------------------------------------
    # Log-density closure
    # ------------------------------------------------------------------
    def logdensity_fn(params_unconstrained):
        """Compute the total log-density for NUTS sampling.

        Combines imaging likelihood, time-delay, and ray-shooting
        consistency terms into a single scalar log-density.

        Parameters
        ----------
        params_unconstrained : dict
            Unconstrained parameter dictionary from NumPyro.

        Returns
        -------
        jax.Array
            Scalar log-density value.
        """
        ll = -loss_obj(params_unconstrained)
        if use_time_delays:
            params_constrained = constrain_fn(seeded_model, (), {}, params_unconstrained)
            D_dt = params_constrained["D_dt"]
            kwargs = prob_model.params2kwargs(params_constrained)
            ra = kwargs["kwargs_point_source"][0]["ra"]
            dec = kwargs["kwargs_point_source"][0]["dec"]
            if use_image_pos_offset:
                offset_x = params_constrained["offset_x_image"]
                offset_y = params_constrained["offset_y_image"]
                ra = ra + offset_x
                dec = dec + offset_y
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
            if use_image_pos_offset:
                offset_x = params_constrained["offset_x_image"]
                offset_y = params_constrained["offset_y_image"]
                ra = ra + offset_x
                dec = dec + offset_y
            x_src, y_src = mass_model_rayshoot.ray_shooting(ra, dec, kwargs["kwargs_lens"])
            if use_source_pos_rayshoot:
                x_src_ref = params_constrained["src_center_x"]
                y_src_ref = params_constrained["src_center_y"]
            else:
                x_src_ref = jnp.mean(x_src)
                y_src_ref = jnp.mean(y_src)
            scatter = (x_src - x_src_ref) ** 2 + (y_src - y_src_ref) ** 2
            if use_rayshoot_sys:
                log_sigma_sys = params_constrained["log_sigma_rayshoot_sys"]
                sigma_sys = jnp.exp(log_sigma_sys)
                sigma2_total = sigma2_rayshoot_fixed + sigma_sys ** 2
            else:
                sigma2_total = sigma2_rayshoot_fixed
            ll = ll - 0.5 * jnp.sum(scatter) / sigma2_total
        return ll

    return logdensity_fn, seeded_model, original_model
