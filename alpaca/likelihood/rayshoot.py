"""
Ray shooting consistency likelihood.

Penalizes scatter in ray-traced source positions - all multiple images
should map back to the same source position in a valid lens model.
"""

import jax.numpy as jnp


def compute_source_scatter(
    x_image: jnp.ndarray,
    y_image: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
    reference_position: str = "mean",
    src_center_x: float = None,
    src_center_y: float = None,
) -> jnp.ndarray:
    """
    Compute scatter in ray-traced source positions.

    Parameters
    ----------
    x_image, y_image : jnp.ndarray
        Image positions (arcsec).
    kwargs_lens : list
        List of kwargs dicts for each lens component.
    mass_model : MassModel
        Herculens MassModel instance.
    reference_position : str
        How to compute reference position: "mean" or "sampled".
    src_center_x, src_center_y : float
        Sampled source center (required if reference_position="sampled").

    Returns
    -------
    jnp.ndarray
        Sum of squared distances from reference position (arcsec^2).
    """
    # Ray-trace to source plane
    x_src, y_src = mass_model.ray_shooting(x_image, y_image, kwargs_lens)

    # Compute reference position
    if reference_position == "sampled":
        if src_center_x is None or src_center_y is None:
            raise ValueError(
                "src_center_x and src_center_y required for reference_position='sampled'"
            )
        x_ref = src_center_x
        y_ref = src_center_y
    else:
        # Use mean of ray-traced positions
        x_ref = jnp.mean(x_src)
        y_ref = jnp.mean(y_src)

    # Compute scatter
    scatter = jnp.sum((x_src - x_ref) ** 2 + (y_src - y_ref) ** 2)

    return scatter


def rayshoot_consistency_loglike(
    x_image: jnp.ndarray,
    y_image: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
    sigma_fixed: float,
    sigma_systematic: float = None,
    reference_position: str = "mean",
    src_center_x: float = None,
    src_center_y: float = None,
) -> jnp.ndarray:
    """
    Compute ray shooting consistency log-likelihood.

    This term encourages all multiple images to map back to the same
    source position, acting as a soft constraint on lens model validity.

    Parameters
    ----------
    x_image, y_image : jnp.ndarray
        Image positions (arcsec).
    kwargs_lens : list
        List of kwargs dicts for each lens component.
    mass_model : MassModel
        Herculens MassModel instance.
    sigma_fixed : float
        Fixed astrometric uncertainty (arcsec).
    sigma_systematic : float, optional
        Additional systematic error term (arcsec). If provided, total
        variance is sigma_fixed^2 + sigma_systematic^2.
    reference_position : str
        How to compute reference position: "mean" or "sampled".
    src_center_x, src_center_y : float
        Sampled source center (required if reference_position="sampled").

    Returns
    -------
    jnp.ndarray
        Scalar log-likelihood value.
    """
    scatter = compute_source_scatter(
        x_image, y_image, kwargs_lens, mass_model,
        reference_position, src_center_x, src_center_y
    )

    # Compute total variance
    sigma2_total = sigma_fixed ** 2
    if sigma_systematic is not None:
        sigma2_total = sigma2_total + sigma_systematic ** 2

    # Log-likelihood (Gaussian penalty on scatter)
    ll = -0.5 * scatter / sigma2_total

    return ll


def source_plane_spread(
    x_image: jnp.ndarray,
    y_image: jnp.ndarray,
    kwargs_lens: list,
    mass_model,
) -> float:
    """
    Compute spread of ray-traced source positions in mas.

    Useful diagnostic for assessing lens model quality.

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
    float
        RMS spread in milliarcseconds.
    """
    x_src, y_src = mass_model.ray_shooting(x_image, y_image, kwargs_lens)

    x_mean = jnp.mean(x_src)
    y_mean = jnp.mean(y_src)

    rms = jnp.sqrt(jnp.mean((x_src - x_mean) ** 2 + (y_src - y_mean) ** 2))

    # Convert arcsec to mas
    return float(rms * 1000)
