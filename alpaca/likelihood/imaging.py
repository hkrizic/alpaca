"""
Imaging likelihood: p(data | model_image, noise_map)

Standard Gaussian pixel likelihood assuming independent noise per pixel.
"""

import jax.numpy as jnp
import numpy as np


def imaging_loglike(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
    include_constant: bool = True,
) -> jnp.ndarray:
    """
    Compute Gaussian imaging log-likelihood.

    ln L = -0.5 * sum_i [(d_i - m_i)^2 / sigma_i^2 + ln(2*pi*sigma_i^2)]

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data.
    model : jnp.ndarray
        Model image prediction.
    noise_map : jnp.ndarray
        Per-pixel noise standard deviation.
    include_constant : bool
        Whether to include the ln(2*pi*sigma^2) normalization term.

    Returns
    -------
    jnp.ndarray
        Scalar log-likelihood value.
    """
    residual = data - model
    sigma2 = noise_map ** 2

    chi2 = jnp.sum(residual ** 2 / sigma2)

    if include_constant:
        const = jnp.sum(jnp.log(2.0 * jnp.pi * sigma2))
        return -0.5 * (chi2 + const)
    else:
        return -0.5 * chi2


def imaging_chi2(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute chi-squared statistic.

    chi^2 = sum_i [(d_i - m_i)^2 / sigma_i^2]

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data.
    model : jnp.ndarray
        Model image prediction.
    noise_map : jnp.ndarray
        Per-pixel noise standard deviation.

    Returns
    -------
    jnp.ndarray
        Scalar chi-squared value.
    """
    residual = data - model
    return jnp.sum(residual ** 2 / noise_map ** 2)


def reduced_chi2(
    data: jnp.ndarray,
    model: jnp.ndarray,
    noise_map: jnp.ndarray,
    n_params: int,
) -> float:
    """
    Compute reduced chi-squared statistic.

    chi^2_red = chi^2 / (n_pixels - n_params)

    Parameters
    ----------
    data : jnp.ndarray
        Observed image data.
    model : jnp.ndarray
        Model image prediction.
    noise_map : jnp.ndarray
        Per-pixel noise standard deviation.
    n_params : int
        Number of free parameters in the model.

    Returns
    -------
    float
        Reduced chi-squared value.
    """
    chi2 = imaging_chi2(data, model, noise_map)
    n_pix = data.size
    dof = max(n_pix - n_params, 1)
    return float(chi2 / dof)
