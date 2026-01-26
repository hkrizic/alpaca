"""
Probabilistic model for lens modeling with Correlated Fields source.

This model uses a pixelated source with Gaussian Process prior (Correlated Fields)
for source reconstruction. All likelihood terms are defined in the model() method.
"""

from typing import Dict, Optional
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from herculens.Inference.ProbModel.numpyro import NumpyroModel

from ..likelihood import (
    imaging_loglike,
    time_delay_loglike,
    rayshoot_consistency_loglike,
)


class ProbModelCorrField(NumpyroModel):
    """
    NumPyro probabilistic model with Correlated Fields source.

    Features:
    - EPL + Shear mass model
    - Sersic lens light
    - Correlated Field (pixelated GP) source light
    - Point sources at image positions
    - Optional time delay likelihood (constrains D_dt)
    - Optional ray shooting consistency

    Note: Source pixels are ALWAYS sampled, including during NUTS.
    """

    def __init__(
        self,
        lens_image,
        img: np.ndarray,
        noise_map: np.ndarray,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        x0s: np.ndarray,
        y0s: np.ndarray,
        peak_vals: np.ndarray,
        source_field,
        num_pixels: int = 80,
        # Time delay settings
        measured_delays: Optional[np.ndarray] = None,
        delay_errors: Optional[np.ndarray] = None,
        # Rayshoot settings
        use_rayshoot_consistency: bool = False,
        rayshoot_consistency_sigma: float = 0.0002,
        use_rayshoot_systematic_error: bool = False,
        rayshoot_sys_error_min: float = 0.00005,
        rayshoot_sys_error_max: float = 0.005,
        # Prior bounds
        D_dt_min: float = 500.0,
        D_dt_max: float = 10000.0,
    ):
        """Initialize the probabilistic model with Correlated Fields source."""
        super().__init__()
        self.lens_image = lens_image
        self.data = jnp.asarray(img)
        self.noise_map = jnp.asarray(noise_map)
        self.xgrid = jnp.asarray(xgrid)
        self.ygrid = jnp.asarray(ygrid)
        self.x0s = jnp.asarray(x0s)
        self.y0s = jnp.asarray(y0s)
        self.peak_vals = jnp.asarray(peak_vals)
        self.source_field = source_field
        self.num_pixels = int(num_pixels)

        # Time delay settings
        self.use_time_delays = (
            measured_delays is not None and delay_errors is not None
        )
        if self.use_time_delays:
            self.measured_delays = jnp.asarray(measured_delays)
            self.delay_errors = jnp.asarray(delay_errors)
        else:
            self.measured_delays = None
            self.delay_errors = None

        # Rayshoot settings
        # Note: For CorrFields, we always use mean reference (no src_center_x/y)
        self.use_rayshoot_consistency = bool(use_rayshoot_consistency)
        self.rayshoot_consistency_sigma = float(rayshoot_consistency_sigma)
        self.use_rayshoot_systematic_error = bool(use_rayshoot_systematic_error)
        self.rayshoot_sys_error_min = float(rayshoot_sys_error_min)
        self.rayshoot_sys_error_max = float(rayshoot_sys_error_max)

        # Prior bounds
        self.D_dt_min = float(D_dt_min)
        self.D_dt_max = float(D_dt_max)

        # Flags
        self.use_corr_fields = True
        self.use_source_shapelets = False

        # Number of physical parameters (for chi2_red)
        self._num_physical_params = 26

    def model(self):
        """NumPyro model specification with Correlated Field source."""
        img = np.asarray(self.data)
        xgrid = np.asarray(self.xgrid)
        ygrid = np.asarray(self.ygrid)

        # Estimate center from light distribution
        p5, p995 = np.percentile(img, [5.0, 99.5])
        clip = np.clip(img, p5, p995)
        w = np.maximum(clip - clip.min(), 0.0)
        W = w.sum() + 1e-12
        cx = float((w * xgrid).sum() / W)
        cy = float((w * ygrid).sum() / W)

        # =====================================================================
        # Lens mass parameters (EPL + Shear)
        # =====================================================================
        lens_center_x = numpyro.sample("lens_center_x", dist.Normal(cx, 0.3))
        lens_center_y = numpyro.sample("lens_center_y", dist.Normal(cy, 0.3))
        theta_E = numpyro.sample("lens_theta_E", dist.Uniform(0.3, 2.2))
        e1 = numpyro.sample("lens_e1", dist.Uniform(-0.4, 0.4))
        e2 = numpyro.sample("lens_e2", dist.Uniform(-0.4, 0.4))
        gamma = numpyro.sample("lens_gamma", dist.Uniform(1.5, 2.5))
        gamma1 = numpyro.sample("lens_gamma1", dist.Uniform(-0.3, 0.3))
        gamma2 = numpyro.sample("lens_gamma2", dist.Uniform(-0.3, 0.3))

        # Time-delay distance
        D_dt = numpyro.sample("D_dt", dist.Uniform(self.D_dt_min, self.D_dt_max))

        # Ray shooting systematic error (log-uniform prior)
        if self.use_rayshoot_systematic_error:
            log_sigma_sys = numpyro.sample(
                "log_sigma_rayshoot_sys",
                dist.Uniform(
                    jnp.log(self.rayshoot_sys_error_min),
                    jnp.log(self.rayshoot_sys_error_max),
                ),
            )
            sigma_rayshoot_sys = jnp.exp(log_sigma_sys)
        else:
            sigma_rayshoot_sys = None

        # =====================================================================
        # Lens light parameters (Sersic)
        # =====================================================================
        amp90 = float(np.percentile(img, 90.0))
        light_amp_L = numpyro.sample(
            "light_amp_L", dist.LogNormal(np.log(max(amp90, 1e-6)), 1.0)
        )
        light_Re_L = numpyro.sample("light_Re_L", dist.Uniform(0.05, 2.5))
        light_n_L = numpyro.sample("light_n_L", dist.Uniform(0.7, 5.5))
        light_e1_L = numpyro.sample("light_e1_L", dist.Uniform(-0.6, 0.6))
        light_e2_L = numpyro.sample("light_e2_L", dist.Uniform(-0.6, 0.6))

        # =====================================================================
        # Source light (Correlated Field)
        # =====================================================================

        source_pixels = self.source_field.numpyro_sample_pixels()

        # =====================================================================
        # Point source parameters
        # =====================================================================
        x_image = numpyro.sample(
            "x_image", dist.Independent(dist.Normal(self.x0s, 0.2), 1)
        )
        y_image = numpyro.sample(
            "y_image", dist.Independent(dist.Normal(self.y0s, 0.2), 1)
        )
        ps_amp = numpyro.sample(
            "ps_amp",
            dist.Independent(
                dist.LogNormal(jnp.log(jnp.maximum(self.peak_vals, 1e-6)), 0.6), 1
            ),
        )

        # =====================================================================
        # Build kwargs for lens_image.model()
        # =====================================================================
        kwargs_lens = [
            dict(
                theta_E=theta_E, e1=e1, e2=e2,
                center_x=lens_center_x, center_y=lens_center_y, gamma=gamma,
            ),
            dict(gamma1=gamma1, gamma2=gamma2, ra_0=0.0, dec_0=0.0),
        ]

        kwargs_lens_light = [
            dict(
                amp=light_amp_L, R_sersic=light_Re_L, n_sersic=light_n_L,
                e1=light_e1_L, e2=light_e2_L,
                center_x=lens_center_x, center_y=lens_center_y,
            )
        ]

        kwargs_source = [{'pixels': source_pixels}]

        kwargs_point = [dict(ra=x_image, dec=y_image, amp=ps_amp)]

        # =====================================================================
        # Compute model image
        # =====================================================================
        model_img = self.lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

        # =====================================================================
        # Likelihood terms (using unified likelihood module)
        # =====================================================================

        # 1. Imaging likelihood
        ll_imaging = imaging_loglike(self.data, model_img, self.noise_map)
        numpyro.factor("imaging_ll", ll_imaging)

        # 2. Time delay likelihood (if enabled)
        if self.use_time_delays:
            ll_td = time_delay_loglike(
                D_dt=D_dt,
                x_image=x_image,
                y_image=y_image,
                kwargs_lens=kwargs_lens,
                mass_model=self.lens_image.MassModel,
                measured_delays=self.measured_delays,
                delay_errors=self.delay_errors,
            )
            numpyro.factor("time_delay_ll", ll_td)

        # 3. Ray shooting consistency (if enabled)
        # Note: For CorrFields, always use mean reference (no src_center params)
        if self.use_rayshoot_consistency:
            ll_rayshoot = rayshoot_consistency_loglike(
                x_image=x_image,
                y_image=y_image,
                kwargs_lens=kwargs_lens,
                mass_model=self.lens_image.MassModel,
                sigma_fixed=self.rayshoot_consistency_sigma,
                sigma_systematic=sigma_rayshoot_sys,
                reference_position="mean",
            )
            numpyro.factor("rayshoot_ll", ll_rayshoot)

    def params2kwargs(self, params: Dict) -> Dict:
        """Convert flat parameter dict to kwargs for lens_image.model()."""
        kwargs_lens = [
            dict(
                theta_E=params["lens_theta_E"],
                e1=params["lens_e1"],
                e2=params["lens_e2"],
                center_x=params["lens_center_x"],
                center_y=params["lens_center_y"],
                gamma=params["lens_gamma"],
            ),
            dict(
                gamma1=params["lens_gamma1"],
                gamma2=params["lens_gamma2"],
                ra_0=0.0,
                dec_0=0.0,
            ),
        ]

        kwargs_lens_light = [
            dict(
                amp=params["light_amp_L"],
                R_sersic=params["light_Re_L"],
                n_sersic=params["light_n_L"],
                e1=params["light_e1_L"],
                e2=params["light_e2_L"],
                center_x=params["lens_center_x"],
                center_y=params["lens_center_y"],
            )
        ]

        # Source pixels from correlated field
        params_jax = {
            k: jnp.asarray(v) if isinstance(v, (list, np.ndarray)) else v
            for k, v in params.items()
        }
        source_pixels = self.source_field.model(params_jax)
        if source_pixels.ndim == 1:
            source_pixels = source_pixels.reshape((self.num_pixels, self.num_pixels))
        kwargs_source = [{'pixels': source_pixels}]

        kwargs_point = [
            dict(
                ra=params["x_image"],
                dec=params["y_image"],
                amp=params["ps_amp"],
            )
        ]

        return dict(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

    def model_image_from_params(self, params: Dict) -> jnp.ndarray:
        """Generate model image from parameter dict."""
        kwargs = self.params2kwargs(params)
        return jnp.asarray(self.lens_image.model(**kwargs))

    def reduced_chi2(self, params: Dict, n_params: int = None) -> float:
        """Compute reduced chi-squared for the model."""
        from ..likelihood.imaging import reduced_chi2
        model_img = self.model_image_from_params(params)
        if n_params is None:
            n_params = self._num_physical_params
        return reduced_chi2(self.data, model_img, self.noise_map, n_params)

    def get_source_pixels_from_params(self, params: Dict) -> jnp.ndarray:
        """Extract source pixel values from parameter dict."""
        params_jax = {
            k: jnp.asarray(v) if isinstance(v, (list, np.ndarray)) else v
            for k, v in params.items()
        }
        source_pixels = self.source_field.model(params_jax)
        if source_pixels.ndim == 1:
            source_pixels = source_pixels.reshape((self.num_pixels, self.num_pixels))
        return source_pixels
