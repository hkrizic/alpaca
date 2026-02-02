"""Prior construction for Nautilus nested sampling.

Builds multivariate prior distributions centered on MAP estimates
for strong gravitational lens model parameters.

author: hkrizic
"""


import numpy as np
from nautilus import Prior
from scipy.stats import lognorm, loguniform, norm
from scipy.stats import uniform as uniform_dist

from alpaca.sampler.nautilus.likelihood import (
    build_gaussian_loglike,
    build_gaussian_loglike_jax,
    make_paramdict_to_kwargs,
    make_paramdict_to_kwargs_jax,
)
from alpaca.sampler.nautilus.prior_utils import _bounded_prior, tnorm


def build_nautilus_prior(
    best_params: dict,
    use_uniform_for_bounded: bool = False,
    uniform_widen_factor: float = 1.1,
    lens_gamma_prior_type: str = "normal",
    lens_gamma_prior_low: float = 1.2,
    lens_gamma_prior_high: float = 2.8,
    lens_gamma_prior_sigma: float = 0.25,
    use_rayshoot_systematic_error: bool = False,
    rayshoot_sys_error_min: float = 0.00005,
    rayshoot_sys_error_max: float = 0.005,
    use_image_pos_offset: bool = False,
    image_pos_offset_sigma: float = 0.01,
) -> tuple[Prior, dict, int]:
    """Construct a Nautilus prior centered on MAP estimate.

    Builds a multivariate prior distribution for nested sampling, with
    individual parameter priors centered on optimization results. Supports
    the full lens model including mass profile, external shear,
    lens/source light, point sources, and optional shapelet components.

    Prior choices follow standard conventions in strong lensing:
    - Gaussian priors for positions (well-constrained by data)
    - Truncated normal for bounded physical parameters
    - Log-normal for strictly positive amplitudes

    Parameters
    ----------
    best_params : dict
        MAP parameter estimates from optimization.
    use_uniform_for_bounded : bool
        Replace truncated normals with uniform distributions.
    uniform_widen_factor : float
        Expansion factor for uniform prior bounds.
    lens_gamma_prior_type : str
        Prior type for lens_gamma ("uniform" or "normal").
    lens_gamma_prior_low : float
        Lower bound for lens_gamma prior.
    lens_gamma_prior_high : float
        Upper bound for lens_gamma prior.
    lens_gamma_prior_sigma : float
        Standard deviation for normal prior.
    use_rayshoot_systematic_error : bool
        Include systematic error parameter.
    rayshoot_sys_error_min : float
        Minimum systematic error value.
    rayshoot_sys_error_max : float
        Maximum systematic error value.
    use_image_pos_offset : bool
        Include image position offset parameters.
    image_pos_offset_sigma : float
        Prior sigma for image position offsets (arcsec).

    Returns
    -------
    tuple of (nautilus.Prior, dict, int)
        Prior object, flattened parameter dict, and number of point
        source images.
    """
    best_flat = dict(best_params)
    if "D_dt" not in best_flat:
        best_flat["D_dt"] = 0.5 * (500.0 + 10000.0)  # Default if missing

    x_im = np.atleast_1d(best_params.get("x_image", []))
    y_im = np.atleast_1d(best_params.get("y_image", []))
    amp_im = np.atleast_1d(best_params.get("ps_amp", []))
    nps = len(x_im)

    for i in range(nps):
        best_flat[f"x_image_{i}"] = float(x_im[i])
        best_flat[f"y_image_{i}"] = float(y_im[i])
        best_flat[f"ps_amp_{i}"] = float(amp_im[i])

    shapelets_amp = np.atleast_1d(best_params.get("shapelets_amp_S", []))
    n_shapelets = len(shapelets_amp)
    use_shapelets = ("shapelets_beta_S" in best_params) and (n_shapelets > 0)

    if use_shapelets:
        best_flat["shapelets_beta_S"] = float(best_params["shapelets_beta_S"])
        for i in range(n_shapelets):
            best_flat[f"shapelets_amp_S_{i}"] = float(shapelets_amp[i])

    prior = Prior()

    # Mass model: lens center, Einstein radius, ellipticity, power-law slope
    prior.add_parameter("lens_center_x", dist=norm(loc=best_flat["lens_center_x"], scale=0.3))
    prior.add_parameter("lens_center_y", dist=norm(loc=best_flat["lens_center_y"], scale=0.3))

    prior.add_parameter(
        "lens_theta_E",
        dist=_bounded_prior(
            best_flat["lens_theta_E"], 0.3, 0.3, 2.2,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "lens_e1",
        dist=_bounded_prior(
            best_flat["lens_e1"], 0.2, -0.4, 0.4,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "lens_e2",
        dist=_bounded_prior(
            best_flat["lens_e2"], 0.2, -0.4, 0.4,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    # Power-law slope (gamma) - configurable prior type and bounds
    if lens_gamma_prior_type == "uniform":
        lens_gamma_dist = uniform_dist(
            loc=lens_gamma_prior_low,
            scale=lens_gamma_prior_high - lens_gamma_prior_low
        )
    else:
        # Normal (truncated normal) prior centered on MAP value
        lens_gamma_dist = tnorm(
            best_flat["lens_gamma"],
            lens_gamma_prior_sigma,
            lens_gamma_prior_low,
            lens_gamma_prior_high
        )
    prior.add_parameter("lens_gamma", dist=lens_gamma_dist)
    # External Shear (gamma1, gamma2)
    prior.add_parameter(
        "lens_gamma1",
        dist=_bounded_prior(
            best_flat["lens_gamma1"], 0.15, -0.3, 0.3,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "lens_gamma2",
        dist=_bounded_prior(
            best_flat["lens_gamma2"], 0.15, -0.3, 0.3,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "D_dt",
        dist=uniform_dist(loc=500, scale=10000 - 500),
    )

    # --- Lens light ---
    # Amplitudes strictly positive -> Lognormal
    mu_Lamp = np.log(max(best_flat["light_amp_L"], 1e-8))
    prior.add_parameter("light_amp_L", dist=lognorm(s=1.0, scale=np.exp(mu_Lamp)))

    prior.add_parameter(
        "light_Re_L",
        dist=_bounded_prior(
            best_flat["light_Re_L"], 0.25, 0.05, 2.5,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "light_n_L",
        dist=_bounded_prior(
            best_flat["light_n_L"], 0.5, 0.7, 5.5,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "light_e1_L",
        dist=_bounded_prior(
            best_flat["light_e1_L"], 0.2, -0.6, 0.6,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "light_e2_L",
        dist=_bounded_prior(
            best_flat["light_e2_L"], 0.2, -0.6, 0.6,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )

    # --- Source light (Sersic) ---
    mu_Samp = np.log(max(best_flat["light_amp_S"], 1e-8))
    prior.add_parameter("light_amp_S", dist=lognorm(s=1.2, scale=np.exp(mu_Samp)))
    prior.add_parameter(
        "light_Re_S",
        dist=_bounded_prior(
            best_flat["light_Re_S"], 0.2, 0.03, 1.2,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "light_n_S",
        dist=_bounded_prior(
            best_flat["light_n_S"], 0.5, 0.5, 4.5,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter("src_center_x", dist=norm(loc=best_flat["src_center_x"], scale=0.6))
    prior.add_parameter("src_center_y", dist=norm(loc=best_flat["src_center_y"], scale=0.6))
    prior.add_parameter(
        "light_e1_S",
        dist=_bounded_prior(
            best_flat.get("light_e1_S", 0.0), 0.35, -0.8, 0.8,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )
    prior.add_parameter(
        "light_e2_S",
        dist=_bounded_prior(
            best_flat.get("light_e2_S", 0.0), 0.35, -0.8, 0.8,
            use_uniform_for_bounded=use_uniform_for_bounded,
            uniform_widen_factor=uniform_widen_factor,
        ),
    )

    # --- Source shapelets (optional) ---
    if use_shapelets:
        # These bounds are chosen to roughly match the generative Uniform prior
        beta_lo, beta_hi = 0.02, 0.6
        sigma_beta = 0.25 * (beta_hi - beta_lo)
        prior.add_parameter(
            "shapelets_beta_S",
            dist=_bounded_prior(
                best_flat["shapelets_beta_S"],
                sigma_beta,
                beta_lo,
                beta_hi,
                use_uniform_for_bounded=use_uniform_for_bounded,
                uniform_widen_factor=uniform_widen_factor,
            ),
        )

        for i in range(n_shapelets):
            amp_i = best_flat[f"shapelets_amp_S_{i}"]
            sigma_amp = max(abs(amp_i), 1e-6)
            prior.add_parameter(
                f"shapelets_amp_S_{i}",
                dist=norm(loc=amp_i, scale=sigma_amp),
            )

    # --- Point sources ---
    for i in range(nps):
        prior.add_parameter(
            f"x_image_{i}",
            dist=norm(loc=best_flat[f"x_image_{i}"], scale=0.2),
        )
        prior.add_parameter(
            f"y_image_{i}",
            dist=norm(loc=best_flat[f"y_image_{i}"], scale=0.2),
        )
        mu_A = np.log(max(best_flat[f"ps_amp_{i}"], 1e-10))
        prior.add_parameter(
            f"ps_amp_{i}",
            dist=lognorm(s=0.6, scale=np.exp(mu_A)),
        )

    # --- Image position offsets (optional) ---
    # These offsets are ONLY used for TD/rayshoot likelihood, not imaging
    if use_image_pos_offset:
        for i in range(nps):
            prior.add_parameter(
                f"offset_x_image_{i}",
                dist=norm(loc=0.0, scale=image_pos_offset_sigma),
            )
            prior.add_parameter(
                f"offset_y_image_{i}",
                dist=norm(loc=0.0, scale=image_pos_offset_sigma),
            )
            # Initialize best_flat with zero offsets
            best_flat[f"offset_x_image_{i}"] = best_params.get(f"offset_x_image_{i}", 0.0)
            best_flat[f"offset_y_image_{i}"] = best_params.get(f"offset_y_image_{i}", 0.0)

    # --- Ray shooting systematic error (optional) ---
    if use_rayshoot_systematic_error:
        prior.add_parameter(
            "sigma_rayshoot_sys",
            dist=loguniform(rayshoot_sys_error_min, rayshoot_sys_error_max),
        )
        # Initialize best_flat with geometric mean of bounds
        best_flat["sigma_rayshoot_sys"] = best_params.get(
            "sigma_rayshoot_sys",
            np.sqrt(rayshoot_sys_error_min * rayshoot_sys_error_max)
        )

    return prior, best_flat, nps


def build_nautilus_prior_and_loglike(
    best_params: dict,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    use_uniform_for_bounded: bool = False,
    use_jax: bool = False,
    use_multi_device: bool = True,
    measured_delays=None,
    delay_errors=None,
    use_rayshoot_consistency: bool = False,
    rayshoot_consistency_sigma: float = 0.0002,
    use_source_position_rayshoot: bool = True,
    use_rayshoot_systematic_error: bool = False,
    rayshoot_sys_error_min: float = 0.00005,
    rayshoot_sys_error_max: float = 0.005,
    lens_gamma_prior_type: str = "normal",
    lens_gamma_prior_low: float = 1.2,
    lens_gamma_prior_high: float = 2.8,
    lens_gamma_prior_sigma: float = 0.25,
    use_corr_fields: bool = False,
    use_image_pos_offset: bool = False,
    image_pos_offset_sigma: float = 0.01,
):
    """Configure complete Nautilus inference setup.

    Factory function combining prior and likelihood construction for
    nested sampling. Optionally enables JAX acceleration for the
    forward model evaluation.

    Parameters
    ----------
    best_params : dict
        MAP parameter estimates for prior centering.
    lens_image : herculens LensImage
        Herculens forward model.
    img : ndarray
        Observed image data.
    noise_map : ndarray
        Per-pixel noise standard deviation.
    use_uniform_for_bounded : bool
        Use uniform instead of truncated normal priors.
    use_jax : bool
        Enable XLA-compiled likelihood evaluation.
    use_multi_device : bool
        Enable multi-GPU parallelism (requires use_jax=True).
    measured_delays : array-like or None
        Time-delay measurements relative to image 0.
    delay_errors : array-like or None
        1-sigma uncertainties for time delays.
    use_rayshoot_consistency : bool
        Add ray shooting consistency term.
    rayshoot_consistency_sigma : float
        Standard deviation (arcsec) for rayshoot term.
    use_source_position_rayshoot : bool
        Compare to sampled source position.
    use_rayshoot_systematic_error : bool
        Include systematic error parameter.
    rayshoot_sys_error_min : float
        Minimum systematic error value.
    rayshoot_sys_error_max : float
        Maximum systematic error value.
    lens_gamma_prior_type : str
        Prior type for lens_gamma ("uniform" or "normal").
    lens_gamma_prior_low : float
        Lower bound for lens_gamma prior.
    lens_gamma_prior_high : float
        Upper bound for lens_gamma prior.
    lens_gamma_prior_sigma : float
        Standard deviation for normal prior.
    use_corr_fields : bool
        Model uses Correlated Fields.
    use_image_pos_offset : bool
        Include image position offset parameters.
    image_pos_offset_sigma : float
        Prior sigma for image position offsets (arcsec).

    Returns
    -------
    tuple of (nautilus.Prior, callable, callable)
        Prior object, parameter transformer, and log-likelihood function.
    """
    prior, best_flat, nps = build_nautilus_prior(
        best_params,
        use_uniform_for_bounded=use_uniform_for_bounded,
        lens_gamma_prior_type=lens_gamma_prior_type,
        lens_gamma_prior_low=lens_gamma_prior_low,
        lens_gamma_prior_high=lens_gamma_prior_high,
        lens_gamma_prior_sigma=lens_gamma_prior_sigma,
        use_rayshoot_systematic_error=use_rayshoot_systematic_error,
        rayshoot_sys_error_min=rayshoot_sys_error_min,
        rayshoot_sys_error_max=rayshoot_sys_error_max,
        use_image_pos_offset=use_image_pos_offset,
        image_pos_offset_sigma=image_pos_offset_sigma,
    )

    if use_jax:
        paramdict_to_kwargs = make_paramdict_to_kwargs_jax(best_flat, nps)
        loglike = build_gaussian_loglike_jax(
            lens_image,
            img,
            noise_map,
            paramdict_to_kwargs,
            use_multi_device=use_multi_device,
            measured_delays=measured_delays,
            delay_errors=delay_errors,
            use_rayshoot_consistency=use_rayshoot_consistency,
            rayshoot_consistency_sigma=rayshoot_consistency_sigma,
            use_source_position_rayshoot=use_source_position_rayshoot,
            use_rayshoot_systematic_error=use_rayshoot_systematic_error,
            use_corr_fields=use_corr_fields,
            use_image_pos_offset=use_image_pos_offset,
        )
    else:
        paramdict_to_kwargs = make_paramdict_to_kwargs(best_flat, nps)
        loglike = build_gaussian_loglike(
            lens_image,
            img,
            noise_map,
            paramdict_to_kwargs,
            measured_delays=measured_delays,
            delay_errors=delay_errors,
            use_rayshoot_consistency=use_rayshoot_consistency,
            rayshoot_consistency_sigma=rayshoot_consistency_sigma,
            use_source_position_rayshoot=use_source_position_rayshoot,
            use_rayshoot_systematic_error=use_rayshoot_systematic_error,
            use_corr_fields=use_corr_fields,
            use_image_pos_offset=use_image_pos_offset,
        )

    return prior, paramdict_to_kwargs, loglike
