"""Likelihood construction for Bayesian lens model inference.

Provides factory functions that build log-likelihood callables for use with
nested sampling (Nautilus), MCMC (emcee), and JAX-accelerated inference.

Extracted from tdlmc_inference.py during package restructuring.
"""


import jax
import jax.numpy as jnp
import numpy as np

from alpaca.sampler.constants import C_KM_S
from alpaca.sampler.losses import (
    _prepare_time_delay_inputs,
)

# ======================================================================
# PARAMETER DICTIONARY TRANSFORMERS
# ======================================================================

def make_paramdict_to_kwargs(best_flat: dict, nps: int):
    """Create parameter dictionary transformer for Herculens forward model.

    Maps flat parameter dictionaries from samplers to the nested kwargs
    structure required by LensImage.model(). Missing parameters are filled
    from the reference best_flat dictionary.

    Args:
        best_flat: Reference parameter dictionary with default values.
        nps: Number of point source images.

    Returns:
        Function mapping {param_name: value} to Herculens kwargs. If
        return_extras=True, returns (kwargs, extras) with D_dt.
    """

    def paramdict_to_kwargs(d: dict, return_extras: bool = False):
        P = dict(best_flat)
        P.update({k: d[k] for k in d.keys() if k in P})

        # ---- Lens mass + shear ----
        kwargs_lens = [
            dict(
                theta_E=P["lens_theta_E"],
                e1=P["lens_e1"],
                e2=P["lens_e2"],
                center_x=P["lens_center_x"],
                center_y=P["lens_center_y"],
                gamma=P["lens_gamma"],
            ),
            dict(
                gamma1=P["lens_gamma1"],
                gamma2=P["lens_gamma2"],
                ra_0=0.0,
                dec_0=0.0,
            ),
        ]

        # ---- Lens light ----
        kwargs_lens_light = [
            dict(
                amp=P["light_amp_L"],
                R_sersic=P["light_Re_L"],
                n_sersic=P["light_n_L"],
                e1=P["light_e1_L"],
                e2=P["light_e2_L"],
                center_x=P["lens_center_x"],
                center_y=P["lens_center_y"],
            )
        ]

        # ---- Source light: Sersic (+ optional shapelets) ----
        kwargs_source = [
            dict(
                amp=P["light_amp_S"],
                R_sersic=P["light_Re_S"],
                n_sersic=P["light_n_S"],
                e1=P["light_e1_S"],
                e2=P["light_e2_S"],
                center_x=P["src_center_x"],
                center_y=P["src_center_y"],
            )
        ]

        # Optional SHAPELETS component (if present in the parameter dict)
        shapelet_keys = sorted(
            [k for k in P.keys() if k.startswith("shapelets_amp_S_")]
        )
        if "shapelets_beta_S" in P and len(shapelet_keys) > 0:
            amps = np.array([P[k] for k in shapelet_keys], dtype=float)
            kwargs_source.append(
                dict(
                    amps=amps,
                    beta=P["shapelets_beta_S"],
                    center_x=P["src_center_x"],
                    center_y=P["src_center_y"],
                )
            )

        # ---- Point sources ----
        if nps:
            ra = np.array([P[f"x_image_{i}"] for i in range(nps)])
            dec = np.array([P[f"y_image_{i}"] for i in range(nps)])
            amp = np.array([P[f"ps_amp_{i}"] for i in range(nps)])
            kwargs_point = [dict(ra=ra, dec=dec, amp=amp)]
        else:
            kwargs_point = None

        kwargs = dict(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )
        if return_extras:
            extras = {}
            if "D_dt" in P:
                extras["D_dt"] = P["D_dt"]
            return kwargs, extras
        return kwargs

    return paramdict_to_kwargs


def make_paramdict_to_kwargs_jax(best_flat: dict, nps: int):
    """JAX-compatible parameter transformer for accelerated likelihood evaluation.

    Variant of make_paramdict_to_kwargs that maintains JAX array types
    throughout, enabling use within jit-compiled, vmapped, or pmapped
    likelihood functions without host-device synchronization.

    Args:
        best_flat: Reference parameter dictionary with default values.
        nps: Number of point source images.

    Returns:
        JAX-traceable function mapping parameters to Herculens kwargs. If
        return_extras=True, returns (kwargs, extras) with D_dt.
    """
    best_flat_jax = {k: jnp.asarray(v) for k, v in best_flat.items()}

    def paramdict_to_kwargs_jax(params: dict, return_extras: bool = False):
        # Merge sampled params with defaults from best_flat
        P = {k: params.get(k, best_flat_jax[k]) for k in best_flat_jax.keys()}

        # ---- Lens mass + shear ----
        kwargs_lens = [
            dict(
                theta_E=P["lens_theta_E"],
                e1=P["lens_e1"],
                e2=P["lens_e2"],
                center_x=P["lens_center_x"],
                center_y=P["lens_center_y"],
                gamma=P["lens_gamma"],
            ),
            dict(
                gamma1=P["lens_gamma1"],
                gamma2=P["lens_gamma2"],
                ra_0=jnp.array(0.0),
                dec_0=jnp.array(0.0),
            ),
        ]

        # ---- Lens light ----
        kwargs_lens_light = [
            dict(
                amp=P["light_amp_L"],
                R_sersic=P["light_Re_L"],
                n_sersic=P["light_n_L"],
                e1=P["light_e1_L"],
                e2=P["light_e2_L"],
                center_x=P["lens_center_x"],
                center_y=P["lens_center_y"],
            )
        ]

        # ---- Source light: Sersic (+ optional shapelets) ----
        kwargs_source = [
            dict(
                amp=P["light_amp_S"],
                R_sersic=P["light_Re_S"],
                n_sersic=P["light_n_S"],
                e1=P["light_e1_S"],
                e2=P["light_e2_S"],
                center_x=P["src_center_x"],
                center_y=P["src_center_y"],
            )
        ]

        # Optional SHAPELETS component (if present in the parameter dict)
        shapelet_keys = sorted(
            [k for k in P.keys() if k.startswith("shapelets_amp_S_")]
        )
        if "shapelets_beta_S" in P and len(shapelet_keys) > 0:
            amp_list = [P[k] for k in shapelet_keys]
            amps = jnp.stack(amp_list)
            kwargs_source.append(
                dict(
                    amps=amps,
                    beta=P["shapelets_beta_S"],
                    center_x=P["src_center_x"],
                    center_y=P["src_center_y"],
                )
            )

        # ---- Point sources (JAX stack needed) ----
        if nps:
            ra_list = [P[f"x_image_{i}"] for i in range(nps)]
            dec_list = [P[f"y_image_{i}"] for i in range(nps)]
            amp_list_ps = [P[f"ps_amp_{i}"] for i in range(nps)]
            ra = jnp.stack(ra_list)
            dec = jnp.stack(dec_list)
            amp_ps = jnp.stack(amp_list_ps)
            kwargs_point = [dict(ra=ra, dec=dec, amp=amp_ps)]
        else:
            kwargs_point = None

        kwargs = dict(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )
        if return_extras:
            extras = {}
            if "D_dt" in P:
                extras["D_dt"] = P["D_dt"]
            return kwargs, extras
        return kwargs

    return paramdict_to_kwargs_jax


# ======================================================================
# LIKELIHOOD FUNCTIONS
# ======================================================================

def build_gaussian_loglike(
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    paramdict_to_kwargs,
    measured_delays=None,
    delay_errors=None,
    use_rayshoot_consistency: bool = False,
    rayshoot_consistency_sigma: float = 0.0002,
    use_source_position_rayshoot: bool = True,
    use_rayshoot_systematic_error: bool = False,
    use_corr_fields: bool = False,
):
    """Construct Gaussian pixel likelihood for lens model inference.

    Implements the standard imaging likelihood assuming independent Gaussian
    noise per pixel:

        ln L(theta) = -0.5 Sum_i [(d_i - m(theta)_i)^2/sigma_i^2 + ln(2*pi*sigma_i^2)]

    where d is the observed data, m(theta) the forward model, and sigma the noise map.
    Supports both scalar evaluation and batched calls for nested sampling.
    Optionally adds Gaussian time-delay and ray shooting consistency terms.

    Args:
        lens_image: Herculens forward model instance.
        img: Observed image data d.
        noise_map: Per-pixel noise standard deviation sigma.
        paramdict_to_kwargs: Parameter transformer for forward model.
        measured_delays: Time-delay measurements relative to image 0.
        delay_errors: 1sigma uncertainties for time delays.
        use_rayshoot_consistency: Add ray shooting consistency term.
        rayshoot_consistency_sigma: Standard deviation (arcsec) for rayshoot term.
        use_source_position_rayshoot: Compare to sampled source position (True)
            or mean of ray-traced positions (False). Forced False for CorrFields.
        use_rayshoot_systematic_error: Include systematic error parameter.
        use_corr_fields: Model uses Correlated Fields (forces mean-based rayshoot).

    Returns:
        Log-likelihood function accepting parameter dictionaries.
    """
    # For Correlated Fields models, force use of mean (no src_center_x/y params)
    if use_corr_fields and use_source_position_rayshoot:
        use_source_position_rayshoot = False
    good = np.isfinite(noise_map) & (noise_map > 0)
    sigma2 = (noise_map[good] ** 2)
    const_term = np.log(2.0 * np.pi * sigma2)
    td_data = _prepare_time_delay_inputs(measured_delays, delay_errors)
    use_time_delays = td_data is not None
    if use_time_delays:
        measured_td, errors_td = td_data
        kwargs_ref, extras_ref = paramdict_to_kwargs({}, return_extras=True)
        if "D_dt" not in extras_ref:
            raise ValueError("Time-delay likelihood requires D_dt in parameters.")
        kwargs_point = kwargs_ref.get("kwargs_point_source")
        if kwargs_point is None:
            raise ValueError("Time-delay data provided but no point sources found.")
        nps = int(np.asarray(kwargs_point[0]["ra"]).size)
        if nps < 2:
            raise ValueError("Need at least two point-source images for time delays.")
        if measured_td.size != (nps - 1):
            raise ValueError(
                "measured_delays must have length n_images-1 (relative to image 0)."
            )
        sigma2_td = errors_td ** 2
        const_td = np.log(2.0 * np.pi * sigma2_td)

    # Ray shooting consistency term setup
    sigma2_rayshoot_fixed = rayshoot_consistency_sigma ** 2
    mass_model = lens_image.MassModel

    def _single_loglike(sample_dict: dict) -> float:
        """Standard scalar log-likelihood for one parameter point."""
        kw = paramdict_to_kwargs(sample_dict)
        model = lens_image.model(**kw)
        r = (img - model)[good]
        ll = -0.5 * np.sum(r * r / sigma2 + const_term)
        if use_time_delays:
            D_dt = float(sample_dict["D_dt"])
            if not np.isfinite(D_dt) or D_dt <= 0:
                return -np.inf
            ra = kw["kwargs_point_source"][0]["ra"]
            dec = kw["kwargs_point_source"][0]["dec"]
            phi = lens_image.MassModel.fermat_potential(ra, dec, kw["kwargs_lens"])
            delta_phi = phi[1:] - phi[0]
            dt_pred = (C_KM_S / D_dt) * delta_phi
            resid_td = dt_pred - measured_td
            ll += -0.5 * np.sum(resid_td * resid_td / sigma2_td + const_td)
        if use_rayshoot_consistency:
            # Ray shoot image positions to source plane
            ra = kw["kwargs_point_source"][0]["ra"]
            dec = kw["kwargs_point_source"][0]["dec"]
            x_src, y_src = mass_model.ray_shooting(ra, dec, kw["kwargs_lens"])
            if use_source_position_rayshoot:
                # Use sampled source position as reference
                x_src_ref = float(sample_dict["src_center_x"])
                y_src_ref = float(sample_dict["src_center_y"])
            else:
                # Use mean of ray-traced positions as reference
                x_src_ref = np.mean(x_src)
                y_src_ref = np.mean(y_src)
            scatter = (x_src - x_src_ref) ** 2 + (y_src - y_src_ref) ** 2
            # Compute effective sigma^2 (fixed + optional systematic)
            if use_rayshoot_systematic_error:
                sigma_sys = float(sample_dict["sigma_rayshoot_sys"])
                sigma2_eff = sigma2_rayshoot_fixed + sigma_sys ** 2
            else:
                sigma2_eff = sigma2_rayshoot_fixed
            ll += -0.5 * np.sum(scatter) / sigma2_eff
        return float(ll)

    def loglike(sample_dict: dict):
        """
        Nautilus entry point. Detects if input is batched or scalar.
        """
        # Detect batched input: any parameter given as a 1D+ array
        n_batch = None
        is_batched = False
        for v in sample_dict.values():
            arr = np.asarray(v)
            if arr.ndim > 0:
                is_batched = True
                if n_batch is None:
                    n_batch = arr.shape[0]
                elif arr.shape[0] != n_batch:
                    raise ValueError(
                        "Inconsistent batch sizes in vectorized likelihood input."
                    )

        # Scalar / non-batched case
        if not is_batched or n_batch is None:
            return _single_loglike(sample_dict)

        # Degenerate n_batch=1 case: unwrap to scalar for numerical stability
        if n_batch == 1:
            single_dict = {
                k: (np.asarray(v)[0] if np.asarray(v).ndim > 0 else v)
                for k, v in sample_dict.items()
            }
            return _single_loglike(single_dict)

        # Proper batched case (Python loop over batch)
        # Note: For massive batching, use the JAX version below.
        out = np.empty(n_batch, dtype=float)
        for i in range(n_batch):
            d_i = {}
            for k, v in sample_dict.items():
                arr = np.asarray(v)
                if arr.ndim == 0:
                    d_i[k] = float(arr)
                else:
                    d_i[k] = arr[i]
            out[i] = _single_loglike(d_i)
        return out

    return loglike


def build_gaussian_loglike_jax(
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    paramdict_to_kwargs_jax,
    use_multi_device: bool = True,
    measured_delays=None,
    delay_errors=None,
    use_rayshoot_consistency: bool = False,
    rayshoot_consistency_sigma: float = 0.0002,
    use_source_position_rayshoot: bool = True,
    use_rayshoot_systematic_error: bool = False,
    use_corr_fields: bool = False,
):
    """GPU-accelerated Gaussian likelihood with multi-device support.

    XLA-compiled implementation of the Gaussian pixel likelihood with
    automatic vectorization (vmap) for batched evaluation and optional
    data parallelism across multiple accelerators (pmap). Optionally adds
    Gaussian time-delay and ray shooting consistency terms.

    Args:
        lens_image: Herculens forward model instance.
        img: Observed image data.
        noise_map: Per-pixel noise standard deviation.
        paramdict_to_kwargs_jax: JAX-compatible parameter transformer.
        use_multi_device: Enable pmap distribution across available devices.
        measured_delays: Time-delay measurements relative to image 0.
        delay_errors: 1sigma uncertainties for time delays.
        use_rayshoot_consistency: Add ray shooting consistency term.
        rayshoot_consistency_sigma: Standard deviation (arcsec) for rayshoot term.
        use_source_position_rayshoot: Compare to sampled source position (True)
            or mean of ray-traced positions (False). Forced False for CorrFields.
        use_rayshoot_systematic_error: Include systematic error parameter.
        use_corr_fields: Model uses Correlated Fields (forces mean-based rayshoot).

    Returns:
        Accelerated log-likelihood function.
    """
    # For Correlated Fields models, force use of mean (no src_center_x/y params)
    if use_corr_fields and use_source_position_rayshoot:
        use_source_position_rayshoot = False
    # Move static data to JAX device
    img_j = jnp.asarray(img)
    noise_j = jnp.asarray(noise_map)
    good = jnp.isfinite(noise_j) & (noise_j > 0.0)
    sigma2 = noise_j[good] ** 2
    const_term = jnp.log(2.0 * jnp.pi * sigma2)
    td_data = _prepare_time_delay_inputs(measured_delays, delay_errors)
    use_time_delays = td_data is not None
    if use_time_delays:
        measured_td, errors_td = td_data
        kwargs_ref, extras_ref = paramdict_to_kwargs_jax({}, return_extras=True)
        if "D_dt" not in extras_ref:
            raise ValueError("Time-delay likelihood requires D_dt in parameters.")
        kwargs_point = kwargs_ref.get("kwargs_point_source")
        if kwargs_point is None:
            raise ValueError("Time-delay data provided but no point sources found.")
        nps = int(np.asarray(kwargs_point[0]["ra"]).shape[0])
        if nps < 2:
            raise ValueError("Need at least two point-source images for time delays.")
        if measured_td.size != (nps - 1):
            raise ValueError(
                "measured_delays must have length n_images-1 (relative to image 0)."
            )
        measured_td_j = jnp.asarray(measured_td)
        sigma2_td = jnp.asarray(errors_td) ** 2
        const_td = jnp.log(2.0 * jnp.pi * sigma2_td)
        c_km_s = jnp.asarray(C_KM_S)

    mass_model = lens_image.MassModel

    # Ray shooting consistency term setup
    sigma2_rayshoot_fixed = jnp.asarray(rayshoot_consistency_sigma ** 2)

    def _potential_jax(x, y, kwargs_lens):
        potential = jnp.zeros_like(x)
        for i, func in enumerate(mass_model.func_list):
            potential = potential + func.function(x, y, **kwargs_lens[i])
        return potential

    def _fermat_potential_jax(x_image, y_image, kwargs_lens):
        # JAX-friendly equivalent of MassModel.fermat_potential.
        potential = _potential_jax(x_image, y_image, kwargs_lens)
        x_source, y_source = mass_model.ray_shooting(x_image, y_image, kwargs_lens)
        geometry = 0.5 * ((x_image - x_source) ** 2 + (y_image - y_source) ** 2)
        return geometry - potential

    # ----- 1. Scalar log-likelihood (one parameter set) -----
    def _loglike_single(params_dict: dict):
        if use_time_delays:
            kw, extras = paramdict_to_kwargs_jax(params_dict, return_extras=True)
            D_dt = extras["D_dt"]
        else:
            kw = paramdict_to_kwargs_jax(params_dict)
        model = lens_image.model(**kw)
        model_j = jnp.asarray(model)
        r = (img_j - model_j)[good]
        ll = -0.5 * jnp.sum(r * r / sigma2 + const_term)
        if use_time_delays:
            ra = kw["kwargs_point_source"][0]["ra"]
            dec = kw["kwargs_point_source"][0]["dec"]
            phi = _fermat_potential_jax(ra, dec, kw["kwargs_lens"])
            delta_phi = phi[1:] - phi[0]
            dt_pred = (c_km_s / D_dt) * delta_phi
            resid_td = dt_pred - measured_td_j
            ll_td = -0.5 * jnp.sum(resid_td * resid_td / sigma2_td + const_td)
            ll = ll + ll_td
        if use_rayshoot_consistency:
            # Ray shoot image positions to source plane
            ra = kw["kwargs_point_source"][0]["ra"]
            dec = kw["kwargs_point_source"][0]["dec"]
            x_src, y_src = mass_model.ray_shooting(ra, dec, kw["kwargs_lens"])
            if use_source_position_rayshoot:
                # Use sampled source position as reference
                x_src_ref = params_dict["src_center_x"]
                y_src_ref = params_dict["src_center_y"]
            else:
                # Use mean of ray-traced positions as reference
                x_src_ref = jnp.mean(x_src)
                y_src_ref = jnp.mean(y_src)
            scatter = (x_src - x_src_ref) ** 2 + (y_src - y_src_ref) ** 2
            # Compute effective sigma^2 (fixed + optional systematic)
            if use_rayshoot_systematic_error:
                sigma_sys = params_dict["sigma_rayshoot_sys"]
                sigma2_eff = sigma2_rayshoot_fixed + sigma_sys ** 2
            else:
                sigma2_eff = sigma2_rayshoot_fixed
            ll = ll - 0.5 * jnp.sum(scatter) / sigma2_eff
        return ll

    loglike_single_jit = jax.jit(_loglike_single)
    loglike_vmap = jax.jit(jax.vmap(loglike_single_jit))

    n_devices = jax.device_count()

    # ----- 2. Multi-device sharding (pmap) -----
    if use_multi_device and n_devices > 1:
        # Vectorised over a chunk of the batch on each device
        def _loglike_chunk(chunk_params_dict: dict):
            # Each leaf has shape (chunk_size,)
            return jax.vmap(loglike_single_jit)(chunk_params_dict)

        loglike_chunk_pmap = jax.pmap(_loglike_chunk)
    else:
        loglike_chunk_pmap = None

    # ----- 3. Helpers for batched inputs from NAUTILUS -----
    def _prepare_batched_params(sample_dict: dict):
        """Convert NAUTILUS dict-of-numpy -> dict-of-JAX with leading batch axis."""
        jax_params = {k: jnp.asarray(v) for k, v in sample_dict.items()}

        # Detect batch size B
        B = None
        for v in jax_params.values():
            if v.ndim > 0:
                B = v.shape[0]
                break

        if B is None:
            # No batched parameters -> scalar call
            return None, jax_params

        def to_batched(x):
            x = jnp.asarray(x)
            if x.ndim == 0:
                return jnp.broadcast_to(x, (B,))
            elif x.ndim == 1:
                if x.shape[0] != B:
                    raise ValueError("Inconsistent batch sizes in JAX loglike.")
                return x
            else:
                raise ValueError(
                    "Parameters with ndim>1 are not supported in this JAX loglike."
                )

        batched = {k: to_batched(v) for k, v in jax_params.items()}
        return B, batched

    # ----- 4. NAUTILUS-facing wrapper -----
    def loglike(sample_dict: dict):
        """
        Accepts scalar or batched dicts (numpy / Python scalars),
        returns float or 1D numpy array of log-likelihoods.
        """
        B, batched_params = _prepare_batched_params(sample_dict)

        # Scalar call: no batch dimension
        if B is None:
            ll = loglike_single_jit(batched_params)
            return float(ll)

        # Batched call
        if loglike_chunk_pmap is not None and B >= n_devices:
            # Multi-device path: shard batch across devices
            # We must pad the batch so it splits evenly across devices.
            B_per = (B + n_devices - 1) // n_devices
            B_pad = B_per * n_devices
            pad = B_pad - B

            def pad_leaf(x):
                if pad == 0:
                    return x
                # Repeat last element to pad; this only affects discarded tail
                return jnp.concatenate([x, jnp.repeat(x[-1], pad)])

            padded = {k: pad_leaf(v) for k, v in batched_params.items()}

            def reshape_leaf(x):
                # (n_devices, B_per)
                return x.reshape((n_devices, B_per))

            sharded = {k: reshape_leaf(v) for k, v in padded.items()}

            # pmap over devices, vmap inside each
            ll_sharded = loglike_chunk_pmap(sharded)   # (n_devices, B_per)
            ll_flat = ll_sharded.reshape((B_pad,))
            ll_final = ll_flat[:B] # Slice off padding
        else:
            # Single-device vmap (still fully JAX)
            ll_final = loglike_vmap(batched_params)

        return np.asarray(ll_final, dtype=np.float64)

    return loglike
