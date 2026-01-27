"""Likelihood loss term builders for time delays and ray shooting consistency."""

import jax.numpy as jnp
import numpy as np

from alpaca.sampler.constants import C_KM_S


def _prepare_time_delay_inputs(measured_delays, delay_errors):
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


def _build_time_delay_loss(
    prob_model,
    lens_image,
    measured_delays,
    delay_errors,
):
    td_data = _prepare_time_delay_inputs(measured_delays, delay_errors)
    if td_data is None:
        return None

    measured_td, errors_td = td_data
    measured_td_j = jnp.asarray(measured_td)
    sigma2_td = jnp.asarray(errors_td) ** 2
    const_td = jnp.log(2.0 * jnp.pi * sigma2_td)
    c_km_s = jnp.asarray(C_KM_S)

    mass_model = lens_image.MassModel

    def _potential_jax(x, y, kwargs_lens):
        potential = jnp.zeros_like(x)
        for i, func in enumerate(mass_model.func_list):
            potential = potential + func.function(x, y, **kwargs_lens[i])
        return potential

    def _fermat_potential_jax(x_image, y_image, kwargs_lens):
        potential = _potential_jax(x_image, y_image, kwargs_lens)
        x_source, y_source = mass_model.ray_shooting(x_image, y_image, kwargs_lens)
        geometry = 0.5 * ((x_image - x_source) ** 2 + (y_image - y_source) ** 2)
        return geometry - potential

    def _td_nll(u_params):
        params = prob_model.constrain(u_params)
        D_dt = params["D_dt"]
        kwargs = prob_model.params2kwargs(params)
        ra = kwargs["kwargs_point_source"][0]["ra"]
        dec = kwargs["kwargs_point_source"][0]["dec"]
        phi = _fermat_potential_jax(ra, dec, kwargs["kwargs_lens"])
        delta_phi = phi[1:] - phi[0]
        dt_pred = (c_km_s / D_dt) * delta_phi
        resid_td = dt_pred - measured_td_j
        nll = 0.5 * jnp.sum(resid_td * resid_td / sigma2_td + const_td)
        return jnp.where(jnp.isfinite(D_dt) & (D_dt > 0), nll, jnp.inf)

    return _td_nll


def _build_rayshoot_consistency_loss(
    prob_model,
    lens_image,
    use_rayshoot_consistency: bool = False,
    rayshoot_consistency_sigma: float = 0.0002,
    use_source_position_rayshoot: bool = True,
    use_rayshoot_systematic_error: bool = False,
):
    """Build ray shooting consistency loss for multistart optimization.

    Penalizes scatter in ray-traced source positions - all multiple images
    should map back to the same source position in a valid lens model.

    Args:
        prob_model: Herculens probabilistic lens model.
        lens_image: Herculens forward model instance.
        use_rayshoot_consistency: Whether to enable this term.
        rayshoot_consistency_sigma: Astrometric uncertainty floor in arcsec.
        use_source_position_rayshoot: If True, compare ray-traced positions to
            sampled source position. If False, compare to mean of ray-traced
            positions. For Correlated Fields models, forced to False.
        use_rayshoot_systematic_error: If True, read sigma_rayshoot_sys from
            params to compute total sigma.

    Returns:
        Loss function taking unconstrained parameters, or None if disabled.
    """
    if not use_rayshoot_consistency:
        return None

    # For Correlated Fields models, force use of mean (no src_center_x/y params)
    is_corr_field_model = hasattr(prob_model, "source_field")
    if is_corr_field_model and use_source_position_rayshoot:
        use_source_position_rayshoot = False

    sigma2_rayshoot_fixed = jnp.asarray(rayshoot_consistency_sigma) ** 2
    mass_model = lens_image.MassModel
    # Read from prob_model to ensure consistency with what parameters actually exist
    use_sys_error = getattr(prob_model, "use_rayshoot_systematic_error", False)

    def _rayshoot_nll(u_params):
        params = prob_model.constrain(u_params)
        kwargs = prob_model.params2kwargs(params)
        ra = kwargs["kwargs_point_source"][0]["ra"]
        dec = kwargs["kwargs_point_source"][0]["dec"]
        x_src, y_src = mass_model.ray_shooting(ra, dec, kwargs["kwargs_lens"])
        if use_source_position_rayshoot:
            # Use sampled source position as reference
            x_src_ref = params["src_center_x"]
            y_src_ref = params["src_center_y"]
        else:
            # Use mean of ray-traced positions as reference
            x_src_ref = jnp.mean(x_src)
            y_src_ref = jnp.mean(y_src)
        scatter = (x_src - x_src_ref) ** 2 + (y_src - y_src_ref) ** 2
        # Compute effective sigma^2 (fixed + optional systematic)
        if use_sys_error:
            # Parameter is sampled in log-space as log_sigma_rayshoot_sys
            log_sigma_sys = params["log_sigma_rayshoot_sys"]
            sigma_sys = jnp.exp(log_sigma_sys)
            sigma2_total = sigma2_rayshoot_fixed + sigma_sys ** 2
        else:
            sigma2_total = sigma2_rayshoot_fixed
        return 0.5 * jnp.sum(scatter) / sigma2_total

    return _rayshoot_nll
