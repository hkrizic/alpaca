"""Likelihood construction for gradient descent optimization.

Builds the combined loss function from imaging likelihood (Herculens),
time-delay terms, and ray-shooting consistency terms.

Note: NUTS builds its own time-delay and ray-shooting terms inline in
alpaca/pipeline/stages/sampling.py because it uses NumPyro's constrain_fn
rather than prob_model.constrain for parameter transformation.

author: hkrizic
"""

import jax.numpy as jnp
from herculens.Inference.loss import Loss

from alpaca.sampler.utils import _prepare_time_delay_inputs

C_KM_S = 299792.458  # speed of light in km/s


def build_likelihood(
    prob_model,
    measured_delays=None,
    delay_errors=None,
    use_rayshoot_consistency: bool = False,
    rayshoot_consistency_sigma: float = 0.0002,
    use_rayshoot_systematic_error: bool = False,
    use_image_pos_offset: bool = False,
):
    """Build the combined loss function for gradient descent optimization.

    Combines the base imaging likelihood from Herculens with optional
    time-delay and ray-shooting consistency terms into a single callable.

    Parameters
    ----------
    prob_model : herculens ProbModel
        Herculens probabilistic lens model (must have .lens_image attribute).
    measured_delays : array-like or None
        Observed time delays relative to image 0.
    delay_errors : array-like or None
        1-sigma uncertainties on the delays.
    use_rayshoot_consistency : bool
        Enable ray-shooting consistency loss term.
    rayshoot_consistency_sigma : float
        Astrometric uncertainty floor in arcsec for ray-shooting.
    use_rayshoot_systematic_error : bool
        If True, read sigma_rayshoot_sys from params for total sigma.
    use_image_pos_offset : bool
        If True, apply offset_x/y_image to positions.

    Returns
    -------
    callable
        Loss function mapping unconstrained parameters to scalar.
    """
    lens_image = prob_model.lens_image

    base_loss = Loss(prob_model)
    td_loss = _build_time_delay_loss(
        prob_model, lens_image, measured_delays, delay_errors,
        use_image_pos_offset=use_image_pos_offset,
    )
    rayshoot_loss = _build_rayshoot_consistency_loss(
        prob_model, lens_image, use_rayshoot_consistency,
        rayshoot_consistency_sigma,
        use_rayshoot_systematic_error=use_rayshoot_systematic_error,
        use_image_pos_offset=use_image_pos_offset,
    )

    loss_terms = [base_loss]
    if td_loss is not None:
        loss_terms.append(td_loss)
    if rayshoot_loss is not None:
        loss_terms.append(rayshoot_loss)

    if len(loss_terms) == 1:
        return base_loss
    return lambda u: sum(term(u) for term in loss_terms)


def _build_time_delay_loss(
    prob_model,
    lens_image,
    measured_delays,
    delay_errors,
    use_image_pos_offset: bool = False,
):
    """Build time-delay negative log-likelihood for gradient descent.

    Parameters
    ----------
    prob_model : herculens ProbModel
        Herculens probabilistic lens model.
    lens_image : herculens LensImage
        Herculens forward model instance.
    measured_delays : array-like or None
        Observed time delays relative to image 0.
    delay_errors : array-like or None
        1-sigma uncertainties on the delays.
    use_image_pos_offset : bool
        If True, apply offset_x/y_image to positions before computing
        Fermat potentials.

    Returns
    -------
    callable or None
        Loss function taking unconstrained parameters, or None if no
        delay data provided.
    """
    td_data = _prepare_time_delay_inputs(measured_delays, delay_errors)
    if td_data is None:
        return None

    measured_td, errors_td = td_data
    measured_td_j = jnp.asarray(measured_td)
    sigma2_td = jnp.asarray(errors_td) ** 2
    const_td = jnp.log(2.0 * jnp.pi * sigma2_td)
    c_km_s = jnp.asarray(C_KM_S)

    mass_model = lens_image.MassModel
    # Read from prob_model to ensure consistency with what parameters actually exist
    use_offset = getattr(prob_model, "use_image_pos_offset", False) and use_image_pos_offset

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
        x_source, y_source = mass_model.ray_shooting(x_image, y_image, kwargs_lens)
        geometry = 0.5 * ((x_image - x_source) ** 2 + (y_image - y_source) ** 2)
        return geometry - potential

    def _td_nll(u_params):
        """Compute time-delay negative log-likelihood for one parameter set.

        Parameters
        ----------
        u_params : jax.Array
            Unconstrained parameter vector.

        Returns
        -------
        jax.Array
            Scalar negative log-likelihood contribution from time delays.
        """
        params = prob_model.constrain(u_params)
        D_dt = params["D_dt"]
        kwargs = prob_model.params2kwargs(params)
        ra = kwargs["kwargs_point_source"][0]["ra"]
        dec = kwargs["kwargs_point_source"][0]["dec"]
        # Apply image position offsets for cosmography (not imaging)
        if use_offset:
            offset_x = params["offset_x_image"]
            offset_y = params["offset_y_image"]
            ra = ra + offset_x
            dec = dec + offset_y
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
    use_image_pos_offset: bool = False,
):
    """Build ray shooting consistency loss for multistart optimization.

    Penalizes scatter in ray-traced source positions - all multiple images
    should map back to the same source position in a valid lens model.

    Parameters
    ----------
    prob_model : herculens ProbModel
        Herculens probabilistic lens model.
    lens_image : herculens LensImage
        Herculens forward model instance.
    use_rayshoot_consistency : bool
        Whether to enable this term.
    rayshoot_consistency_sigma : float
        Astrometric uncertainty floor in arcsec.
    use_source_position_rayshoot : bool
        If True, compare ray-traced positions to sampled source position.
        If False, compare to mean of ray-traced positions. For Correlated
        Fields models, forced to False.
    use_rayshoot_systematic_error : bool
        If True, read sigma_rayshoot_sys from params to compute total sigma.
    use_image_pos_offset : bool
        If True, apply offset_x/y_image to positions before ray shooting
        (accounts for astrometric errors).

    Returns
    -------
    callable or None
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
    use_offset = getattr(prob_model, "use_image_pos_offset", False) and use_image_pos_offset

    def _rayshoot_nll(u_params):
        """Compute ray-shooting consistency negative log-likelihood.

        Parameters
        ----------
        u_params : jax.Array
            Unconstrained parameter vector.

        Returns
        -------
        jax.Array
            Scalar penalty for scatter in ray-traced source positions.
        """
        params = prob_model.constrain(u_params)
        kwargs = prob_model.params2kwargs(params)
        ra = kwargs["kwargs_point_source"][0]["ra"]
        dec = kwargs["kwargs_point_source"][0]["dec"]
        # Apply image position offsets for cosmography (not imaging)
        if use_offset:
            offset_x = params["offset_x_image"]
            offset_y = params["offset_y_image"]
            ra = ra + offset_x
            dec = dec + offset_y
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
