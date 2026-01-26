"""
Time delay likelihood: p(measured_delays | D_dt, image_positions, lens_model)

Constrains the time-delay distance D_dt using measured time delays between
multiple images of a lensed quasar.
"""

import jax.numpy as jnp

# Speed of light in km/s
C_KM_S = 299792.458


def gravitational_potential(
    x: jnp.ndarray,
    y: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
) -> jnp.ndarray:
    """
    Compute gravitational lensing potential at given positions.

    Parameters
    ----------
    x, y : jnp.ndarray
        Image plane coordinates (arcsec).
    kwargs_lens : list
        List of kwargs dicts for each lens component.
    mass_model : MassModel
        Herculens MassModel instance.

    Returns
    -------
    jnp.ndarray
        Potential values at each position.
    """
    potential = jnp.zeros_like(x)
    for i, func in enumerate(mass_model.func_list):
        potential = potential + func.function(x, y, **kwargs_lens[i])
    return potential


def fermat_potential(
    x_image: jnp.ndarray,
    y_image: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
) -> jnp.ndarray:
    """
    Compute Fermat potential at image positions.

    The Fermat potential is: phi_F = 0.5 * |x - x_s|^2 - psi(x)

    where x_s is the source position obtained by ray-tracing.

    Parameters
    ----------
    x_image, y_image : jnp.ndarray
        Image positions (arcsec).
    kwargs_lens : list
        List of kwargs dicts for each lens component.
    mass_model : MassModel
        Herculens MassModel instance.

    Returns
    -------
    jnp.ndarray
        Fermat potential at each image position.
    """
    # Gravitational potential
    psi = gravitational_potential(x_image, y_image, kwargs_lens, mass_model)

    # Ray-trace to source plane
    x_source, y_source = mass_model.ray_shooting(x_image, y_image, kwargs_lens)

    # Geometric term: 0.5 * |x - x_s|^2
    geometry = 0.5 * ((x_image - x_source) ** 2 + (y_image - y_source) ** 2)

    return geometry - psi


def predict_time_delays(
    D_dt: jnp.ndarray,
    x_image: jnp.ndarray,
    y_image: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
    reference_image: int = 0,
) -> jnp.ndarray:
    """
    Predict time delays relative to a reference image.

    dt_ij = (D_dt / c) * (phi_F,i - phi_F,j)

    Parameters
    ----------
    D_dt : jnp.ndarray
        Time-delay distance in Mpc.
    x_image, y_image : jnp.ndarray
        Image positions (arcsec).
    kwargs_lens : list
        List of kwargs dicts for each lens component.
    mass_model : MassModel
        Herculens MassModel instance.
    reference_image : int
        Index of reference image (default: 0).

    Returns
    -------
    jnp.ndarray
        Predicted time delays relative to reference image (length n_images - 1).
    """
    phi = fermat_potential(x_image, y_image, kwargs_lens, mass_model)
    delta_phi = phi - phi[reference_image]

    # Remove reference image from output
    # delta_phi[reference_image] = 0 by construction
    delta_phi_others = jnp.concatenate([
        delta_phi[:reference_image],
        delta_phi[reference_image + 1:]
    ])

    # Time delay: dt = (c / D_dt) * delta_phi
    # Note: delta_phi is in arcsec^2, need conversion factor
    # The formula is: dt [days] = D_dt [Mpc] / c * delta_phi [arcsec^2] * conversion
    # But herculens uses: dt = (c_km_s / D_dt) * delta_phi (in some internal units)
    # Let's follow the existing convention:
    dt_pred = (C_KM_S / D_dt) * delta_phi_others

    return dt_pred


def time_delay_loglike(
    D_dt: jnp.ndarray,
    x_image: jnp.ndarray,
    y_image: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
    measured_delays: jnp.ndarray,
    delay_errors: jnp.ndarray,
    reference_image: int = 0,
    include_constant: bool = True,
) -> jnp.ndarray:
    """
    Compute time delay log-likelihood.

    Assumes Gaussian errors on measured time delays.

    Parameters
    ----------
    D_dt : jnp.ndarray
        Time-delay distance in Mpc.
    x_image, y_image : jnp.ndarray
        Image positions (arcsec).
    kwargs_lens : list
        List of kwargs dicts for each lens component.
    mass_model : MassModel
        Herculens MassModel instance.
    measured_delays : jnp.ndarray
        Observed time delays relative to reference image.
    delay_errors : jnp.ndarray
        1-sigma uncertainties on measured delays.
    reference_image : int
        Index of reference image (default: 0).
    include_constant : bool
        Whether to include normalization constant.

    Returns
    -------
    jnp.ndarray
        Scalar log-likelihood value.
    """
    dt_pred = predict_time_delays(
        D_dt, x_image, y_image, kwargs_lens, mass_model, reference_image
    )

    residual = dt_pred - measured_delays
    sigma2 = delay_errors ** 2

    chi2 = jnp.sum(residual ** 2 / sigma2)

    if include_constant:
        const = jnp.sum(jnp.log(2.0 * jnp.pi * sigma2))
        ll = -0.5 * (chi2 + const)
    else:
        ll = -0.5 * chi2

    # Guard against invalid D_dt values
    ll = jnp.where(jnp.isfinite(D_dt) & (D_dt > 0), ll, -jnp.inf)

    return ll
