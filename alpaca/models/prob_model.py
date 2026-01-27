"""
Probabilistic lens models and lens image construction for TDLMC.
Includes STARRED PSF reconstruction support and Correlated Field source models.
"""


import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Inference.ProbModel.numpyro import NumpyroModel
from herculens.Instrument.noise import Noise
from herculens.Instrument.psf import PSF
from herculens.LensImage.lens_image import LensImage
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source_model import PointSourceModel

# --- Optional Correlated Field imports ----------------------------------------
try:
    import herculens as hcl
    CorrelatedField = hcl.CorrelatedField
    _HAS_CORR_FIELD = True
except (ImportError, AttributeError):
    CorrelatedField = None
    _HAS_CORR_FIELD = False


def make_lens_image(
    img: np.ndarray,
    psf_kernel: np.ndarray,
    noise_map: np.ndarray,
    pixel_grid: PixelGrid,
    ps_grid: PixelGrid,
    supersampling_factor: int = 5,
    convolution_type: str = "jax_scipy_fft",
    use_source_shapelets: bool = False,
    shapelets_n_max: int = 6,
    # Correlated Fields options
    use_corr_fields: bool = False,
    corr_field_num_pixels: int = 80,
    corr_field_interpolation: str = 'fast_bilinear',
    source_arc_mask: np.ndarray | None = None,
):
    """
    Build PSF, Noise, Mass/Light models and LensImage.

    Parameters
    ----------
    img : np.ndarray
        Observed image data.
    psf_kernel : np.ndarray
        PSF kernel array.
    noise_map : np.ndarray
        Per-pixel noise standard deviation.
    pixel_grid : PixelGrid
        Main pixel grid for the image.
    ps_grid : PixelGrid
        Supersampled grid for point sources.
    supersampling_factor : int
        Supersampling factor for source light.
    convolution_type : str
        Convolution method for PSF.
    use_source_shapelets : bool
        Use Shapelets basis for source (mutually exclusive with use_corr_fields).
    shapelets_n_max : int
        Maximum shapelet order.
    use_corr_fields : bool
        Use Correlated Fields (pixelated) for source.
    corr_field_num_pixels : int
        Number of pixels per side for the source grid.
    corr_field_interpolation : str
        Interpolation type for pixelated source ('fast_bilinear', 'bilinear', etc.).
    source_arc_mask : np.ndarray, optional
        Boolean mask for adaptive source grid (required for corr fields).

    Returns
    -------
    tuple
        (lens_image, noise, psf, mass_model, lens_light_model,
         source_light_model, point_source_model)
    """
    npix_y, npix_x = img.shape

    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel) # herculens PSF function
    noise = Noise(npix_x, npix_y, noise_map=noise_map) # herculens Noise function

    mass_model = MassModel(["EPL", "SHEAR"]) # herculens MassModel function
    lens_light_model = LightModel(["SERSIC_ELLIPSE"]) # herculens LightModel function

    # Source light model selection
    if use_corr_fields:
        # Correlated Fields: pixelated source with GP prior
        if not _HAS_CORR_FIELD:
            raise ImportError(
                "Correlated Fields require herculens with CorrelatedField support. "
                "Please update herculens or set use_corr_fields=False."
            )
        # Create PixelatedLight source model
        source_light_model = LightModel(
            hcl.PixelatedLight(
                interpolation_type=corr_field_interpolation,
                adaptive_grid=True,
                allow_extrapolation=False,
            ),
            kwargs_pixelated=dict(num_pixels=corr_field_num_pixels),
        )
    elif use_source_shapelets:
        # Shapelets: analytic basis functions
        source_light_model = LightModel(
            ["SERSIC_ELLIPSE", "SHAPELETS"],
            shapelets_n_max=shapelets_n_max,
        )
    else:
        # Simple Sersic source
        source_light_model = LightModel(["SERSIC_ELLIPSE"])

    point_source_model = PointSourceModel(["IMAGE_POSITIONS"], mass_model, ps_grid)

    kwargs_numerics = dict(
        supersampling_factor=supersampling_factor,
        convolution_type=convolution_type,
        supersampling_convolution=False,
    )

    # Build LensImage with optional arc mask for correlated fields
    lens_image = LensImage(
        pixel_grid,
        psf,
        noise_class=noise,
        lens_mass_model_class=mass_model,
        lens_light_model_class=lens_light_model,
        source_model_class=source_light_model,
        point_source_model_class=point_source_model,
        source_arc_mask=source_arc_mask if use_corr_fields else None,
        kwargs_numerics=kwargs_numerics,
    )

    return (
        lens_image,
        noise,
        psf,
        mass_model,
        lens_light_model,
        source_light_model,
        point_source_model,
    )


def create_corr_field(
    source_model: LightModel,
    img: np.ndarray,
    num_pixels: int = 80,
    mean_intensity: float | None = None,
    offset_std: tuple[float, float] = (0.5, 1e-3),
    loglogavgslope: tuple[float, float] = (-6., 0.5),
    fluctuations: tuple[float, float] = (1., 0.5),
    flexibility: tuple[float, float] | None = None,
    asperity: tuple[float, float] | None = None,
    cropped_border_size: int = 20,
    exponentiate: bool = True,
):
    """
    Create a CorrelatedField instance for source modeling.

    The CorrelatedField uses a Gaussian Process prior on a pixelated grid
    to model complex source morphologies with controlled smoothness.

    Parameters
    ----------
    source_model : LightModel
        The PixelatedLight source model.
    img : np.ndarray
        Observed image (used to estimate mean intensity if not specified).
    num_pixels : int
        Number of pixels per side for the source grid.
    mean_intensity : float, optional
        Mean source intensity. If None, estimated from data.
    offset_std : tuple
        Prior (mean, std) for the offset standard deviation.
    loglogavgslope : tuple
        Prior (mean, std) for log-log average slope (controls smoothness).
        More negative values produce smoother fields.
    fluctuations : tuple
        Prior (mean, std) for fluctuation amplitude.
    flexibility : tuple, optional
        Prior for flexibility parameter (if supported).
    asperity : tuple, optional
        Prior for asperity parameter (if supported).
    cropped_border_size : int
        Border cropping size for FFT operations.
    exponentiate : bool
        If True, exponentiate the field to ensure positivity.

    Returns
    -------
    CorrelatedField
        Configured correlated field for numpyro sampling.
    """
    if not _HAS_CORR_FIELD:
        raise ImportError(
            "CorrelatedField requires herculens with correlated_field support. "
            "Please update herculens."
        )

    # Estimate mean intensity from data if not specified
    if mean_intensity is None:
        mean_intensity = float(np.percentile(img, 70))
        mean_intensity = max(mean_intensity, 1e-4)

    # Create the CorrelatedField
    source_field = CorrelatedField(
        "source_pixels",
        source_model,
        offset_mean=np.log(mean_intensity) if exponentiate else mean_intensity,
        prior_offset_std=offset_std,
        prior_loglogavgslope=loglogavgslope,
        prior_fluctuations=fluctuations,
        prior_flexibility=flexibility,
        prior_asperity=asperity,
        cropped_border_size=cropped_border_size,
        exponentiate=exponentiate,
    )

    return source_field


class ProbModel(NumpyroModel):
    """
    NumPyro probabilistic model for the TDLMC lens:
    EPL+Shear, Sersic lens/source, optional Shapelets, Point Sources.
    """

    def __init__(
        self,
        lens_image: LensImage,
        img: np.ndarray,
        noise_map: np.ndarray,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        x0s: np.ndarray,
        y0s: np.ndarray,
        peak_vals: np.ndarray,
        use_source_shapelets: bool = False,
        shapelets_n_max: int = 6,
        shapelets_beta_min: float = 0.02,
        shapelets_beta_max: float = 0.6,
        shapelets_amp_sigma: float = 1.0,
        use_rayshoot_systematic_error: bool = False,
        rayshoot_sys_error_min: float = 0.00005,
        rayshoot_sys_error_max: float = 0.005,
    ):
        super().__init__()
        self.lens_image = lens_image
        self.data = jnp.asarray(img)
        self.noise_map = jnp.asarray(noise_map)
        self.xgrid = jnp.asarray(xgrid)
        self.ygrid = jnp.asarray(ygrid)
        self.x0s = jnp.asarray(x0s)
        self.y0s = jnp.asarray(y0s)
        self.peak_vals = jnp.asarray(peak_vals)

        self.use_source_shapelets = bool(use_source_shapelets)
        self.shapelets_n_max = int(shapelets_n_max)
        self.shapelets_beta_min = float(shapelets_beta_min)
        self.shapelets_beta_max = float(shapelets_beta_max)
        self.shapelets_amp_sigma = float(shapelets_amp_sigma)
        self.use_rayshoot_systematic_error = bool(use_rayshoot_systematic_error)
        self.rayshoot_sys_error_min = float(rayshoot_sys_error_min)
        self.rayshoot_sys_error_max = float(rayshoot_sys_error_max)

    def model(self):
        img = np.asarray(self.data)
        xgrid = np.asarray(self.xgrid)
        ygrid = np.asarray(self.ygrid)
        x0s = self.x0s
        y0s = self.y0s
        peak_vals = self.peak_vals

        p5, p995 = np.percentile(img, [5.0, 99.5])
        clip = np.clip(img, p5, p995)
        w = np.maximum(clip - clip.min(), 0.0)
        W = w.sum() + 1e-12
        cx = float((w * xgrid).sum() / W)
        cy = float((w * ygrid).sum() / W)

        lens_center_x = numpyro.sample("lens_center_x", dist.Normal(cx, 0.3))
        lens_center_y = numpyro.sample("lens_center_y", dist.Normal(cy, 0.3))
        theta_E = numpyro.sample("lens_theta_E", dist.Uniform(0.3, 2.2))
        e1 = numpyro.sample("lens_e1", dist.Uniform(-0.4, 0.4))
        e2 = numpyro.sample("lens_e2", dist.Uniform(-0.4, 0.4))
        gamma = numpyro.sample("lens_gamma", dist.Uniform(1.5, 2.5))
        gamma1 = numpyro.sample("lens_gamma1", dist.Uniform(-0.3, 0.3))
        gamma2 = numpyro.sample("lens_gamma2", dist.Uniform(-0.3, 0.3))
        numpyro.sample("D_dt", dist.Uniform(500.0, 10000.0))

        # Ray shooting systematic error as free parameter (log-uniform prior)
        # Note: We only sample log_sigma_rayshoot_sys here. The transformation
        # to sigma_rayshoot_sys = exp(log_sigma_rayshoot_sys) is done in post-processing.
        # We avoid numpyro.deterministic because herculens' unconstrain_reparam
        # doesn't handle deterministic sites properly.
        if self.use_rayshoot_systematic_error:
            numpyro.sample(
                "log_sigma_rayshoot_sys",
                dist.Uniform(
                    jnp.log(self.rayshoot_sys_error_min),
                    jnp.log(self.rayshoot_sys_error_max),
                ),
            )

        amp90 = float(np.percentile(img, 90.0))
        light_amp_L = numpyro.sample(
            "light_amp_L", dist.LogNormal(np.log(max(amp90, 1e-6)), 1.0)
        )
        light_Re_L = numpyro.sample("light_Re_L", dist.Uniform(0.05, 2.5))
        light_n_L = numpyro.sample("light_n_L", dist.Uniform(0.7, 5.5))
        light_e1_L = numpyro.sample("light_e1_L", dist.Uniform(-0.6, 0.6))
        light_e2_L = numpyro.sample("light_e2_L", dist.Uniform(-0.6, 0.6))

        amp70 = float(np.percentile(img, 70.0))
        light_amp_S = numpyro.sample(
            "light_amp_S", dist.LogNormal(np.log(max(amp70, 3e-6)), 1.2)
        )
        light_Re_S = numpyro.sample("light_Re_S", dist.Uniform(0.03, 1.2))
        light_n_S = numpyro.sample("light_n_S", dist.Uniform(0.5, 4.5))
        light_e1_S = numpyro.sample("light_e1_S", dist.Uniform(-0.8, 0.8))
        light_e2_S = numpyro.sample("light_e2_S", dist.Uniform(-0.8, 0.8))
        src_center_x = numpyro.sample("src_center_x", dist.Normal(0.0, 0.6))
        src_center_y = numpyro.sample("src_center_y", dist.Normal(0.0, 0.6))

        if self.use_source_shapelets:
            n_max = max(int(self.shapelets_n_max), 0)
            n_basis = (n_max + 1) * (n_max + 2) // 2
            shapelets_beta_S = numpyro.sample(
                "shapelets_beta_S",
                dist.Uniform(self.shapelets_beta_min, self.shapelets_beta_max),
            )
            shapelet_amp_scale = float(max(amp70, 3e-6) * self.shapelets_amp_sigma)
            shapelets_amp_S = numpyro.sample(
                "shapelets_amp_S",
                dist.Independent(
                    dist.Normal(
                        jnp.zeros(n_basis),
                        shapelet_amp_scale * jnp.ones(n_basis),
                    ),
                    1,
                ),
            )

        x_image = numpyro.sample(
            "x_image",
            dist.Independent(dist.Normal(x0s, 0.2), 1),
        )
        y_image = numpyro.sample(
            "y_image",
            dist.Independent(dist.Normal(y0s, 0.2), 1),
        )
        ps_amp = numpyro.sample(
            "ps_amp",
            dist.Independent(
                dist.LogNormal(jnp.log(jnp.maximum(peak_vals, 1e-6)), 0.6),
                1,
            ),
        )

        kwargs_lens = [
            dict(
                theta_E=theta_E,
                e1=e1,
                e2=e2,
                center_x=lens_center_x,
                center_y=lens_center_y,
                gamma=gamma,
            ),
            dict(gamma1=gamma1, gamma2=gamma2, ra_0=0.0, dec_0=0.0),
        ]
        kwargs_lens_light = [
            dict(
                amp=light_amp_L,
                R_sersic=light_Re_L,
                n_sersic=light_n_L,
                e1=light_e1_L,
                e2=light_e2_L,
                center_x=lens_center_x,
                center_y=lens_center_y,
            )
        ]
        kwargs_source = [
            dict(
                amp=light_amp_S,
                R_sersic=light_Re_S,
                n_sersic=light_n_S,
                e1=light_e1_S,
                e2=light_e2_S,
                center_x=src_center_x,
                center_y=src_center_y,
            )
        ]

        if self.use_source_shapelets:
            kwargs_source.append(
                dict(
                    amps=shapelets_amp_S,
                    beta=shapelets_beta_S,
                    center_x=src_center_x,
                    center_y=src_center_y,
                )
            )

        kwargs_point = [dict(ra=x_image, dec=y_image, amp=ps_amp)]

        model_img = self.lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_img, self.noise_map), 2),
            obs=self.data,
        )

    def params2kwargs(self, params: dict):
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
        kwargs_source = [
            dict(
                amp=params["light_amp_S"],
                R_sersic=params["light_Re_S"],
                n_sersic=params["light_n_S"],
                e1=params["light_e1_S"],
                e2=params["light_e2_S"],
                center_x=params["src_center_x"],
                center_y=params["src_center_y"],
            )
        ]

        if getattr(self, "use_source_shapelets", False):
            kwargs_source.append(
                dict(
                    amps=params["shapelets_amp_S"],
                    beta=params["shapelets_beta_S"],
                    center_x=params["src_center_x"],
                    center_y=params["src_center_y"],
                )
            )

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

    def model_image_from_params(self, params: dict):
        kwargs = self.params2kwargs(params)
        model_img = self.lens_image.model(**kwargs)
        return jnp.asarray(model_img)

    def reduced_chi2(self, params: dict, n_params: int | None = None) -> float:
        model_img = self.model_image_from_params(params)
        resid = (self.data - model_img) / self.noise_map
        chi2 = jnp.sum(resid**2)

        n_pix = self.data.size
        if n_params is None:
            n_params = len(params)
        dof = max(int(n_pix - n_params), 1)
        chi2_red = chi2 / dof
        return float(chi2_red)


class ProbModelCorrField(NumpyroModel):
    """
    NumPyro probabilistic model using Correlated Fields for the source.

    This model uses a pixelated source representation with a Gaussian Process
    prior that enforces smoothness. Unlike Shapelets, this approach can capture
    arbitrary source morphologies while maintaining physical regularization.

    The model includes:
    - EPL + Shear mass model
    - Sersic lens light
    - Correlated Field (pixelated) source light
    - Point sources at image positions
    """

    def __init__(
        self,
        lens_image: LensImage,
        img: np.ndarray,
        noise_map: np.ndarray,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        x0s: np.ndarray,
        y0s: np.ndarray,
        peak_vals: np.ndarray,
        source_field,
        num_pixels: int = 80,
        use_rayshoot_systematic_error: bool = False,
        rayshoot_sys_error_min: float = 0.00005,
        rayshoot_sys_error_max: float = 0.005,
    ):
        """
        Initialize the Correlated Field probabilistic model.

        Parameters
        ----------
        lens_image : LensImage
            Herculens LensImage object with PixelatedLight source.
        img : np.ndarray
            Observed image data.
        noise_map : np.ndarray
            Per-pixel noise standard deviation.
        xgrid, ygrid : np.ndarray
            Coordinate grids.
        x0s, y0s : np.ndarray
            Initial point source positions (detected or truth).
        peak_vals : np.ndarray
            Initial point source amplitudes.
        source_field : CorrelatedField
            The CorrelatedField instance for source sampling.
        use_rayshoot_systematic_error : bool
            Include ray shooting systematic error as free parameter.
        rayshoot_sys_error_min, rayshoot_sys_error_max : float
            Bounds for ray shooting systematic error (arcsec).
        """
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
        self.use_rayshoot_systematic_error = bool(use_rayshoot_systematic_error)
        self.rayshoot_sys_error_min = float(rayshoot_sys_error_min)
        self.rayshoot_sys_error_max = float(rayshoot_sys_error_max)

        # Flag for identification
        self.use_corr_fields = True
        self.use_source_shapelets = False

        # Number of "physical" parameters for chi2_red computation
        # This excludes the thousands of source pixel parameters
        # Approximate count: lens (10) + lens_light (5) + point_source (8) + D_dt (1) + src_center (2) + optional (2)
        self._num_physical_params = 28

    @property
    def num_parameters_for_chi2(self) -> int:
        """Number of parameters to use for reduced chi2 calculation.

        For Correlated Fields, this returns the number of 'physical' model
        parameters (lens, light, point source), excluding the source pixel
        parameters which would make chi2_red meaningless.
        """
        return self._num_physical_params

    def model(self):
        """NumPyro model specification with Correlated Field source."""
        img = np.asarray(self.data)
        xgrid = np.asarray(self.xgrid)
        ygrid = np.asarray(self.ygrid)
        x0s = self.x0s
        y0s = self.y0s
        peak_vals = self.peak_vals

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
        numpyro.sample("D_dt", dist.Uniform(500.0, 10000.0))

        # Ray shooting systematic error as free parameter (log-uniform prior)
        # Note: We only sample log_sigma_rayshoot_sys here. The transformation
        # to sigma_rayshoot_sys = exp(log_sigma_rayshoot_sys) is done in post-processing.
        if self.use_rayshoot_systematic_error:
            numpyro.sample(
                "log_sigma_rayshoot_sys",
                dist.Uniform(
                    jnp.log(self.rayshoot_sys_error_min),
                    jnp.log(self.rayshoot_sys_error_max),
                ),
            )

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
        # Source light parameters (Correlated Field)
        # =====================================================================
        source_pixels = self.source_field.numpyro_sample_pixels()

        # Note: No src_center_x/y for Correlated Fields - the rayshoot
        # consistency check will use mean of ray-traced positions instead

        # =====================================================================
        # Point source parameters
        # =====================================================================
        x_image = numpyro.sample(
            "x_image",
            dist.Independent(dist.Normal(x0s, 0.2), 1),
        )
        y_image = numpyro.sample(
            "y_image",
            dist.Independent(dist.Normal(y0s, 0.2), 1),
        )
        ps_amp = numpyro.sample(
            "ps_amp",
            dist.Independent(
                dist.LogNormal(jnp.log(jnp.maximum(peak_vals, 1e-6)), 0.6),
                1,
            ),
        )

        # =====================================================================
        # Build kwargs for lens_image.model()
        # =====================================================================
        kwargs_lens = [
            dict(
                theta_E=theta_E,
                e1=e1,
                e2=e2,
                center_x=lens_center_x,
                center_y=lens_center_y,
                gamma=gamma,
            ),
            dict(gamma1=gamma1, gamma2=gamma2, ra_0=0.0, dec_0=0.0),
        ]

        kwargs_lens_light = [
            dict(
                amp=light_amp_L,
                R_sersic=light_Re_L,
                n_sersic=light_n_L,
                e1=light_e1_L,
                e2=light_e2_L,
                center_x=lens_center_x,
                center_y=lens_center_y,
            )
        ]

        # Source is just the pixelated field
        kwargs_source = [{'pixels': source_pixels}]

        kwargs_point = [dict(ra=x_image, dec=y_image, amp=ps_amp)]

        # =====================================================================
        # Generate model image and likelihood
        # =====================================================================
        model_img = self.lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_img, self.noise_map), 2),
            obs=self.data,
        )

    def params2kwargs(self, params: dict):
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
        # The source_field.model() expects JAX arrays, so convert all params
        # (multistart optimization may return Python lists)
        params_jax = {
            k: jnp.asarray(v) if isinstance(v, (list, np.ndarray)) else v
            for k, v in params.items()
        }
        source_pixels = self.source_field.model(params_jax)
        # Reshape to 2D if needed (plotting expects 2D array)
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

    def model_image_from_params(self, params: dict):
        """Generate model image from parameter dict."""
        kwargs = self.params2kwargs(params)
        model_img = self.lens_image.model(**kwargs)
        return jnp.asarray(model_img)

    def reduced_chi2(self, params: dict, n_params: int | None = None) -> float:
        """Compute reduced chi-squared for the model.

        For Correlated Fields, uses only the physical model parameters
        (lens, light, point source) for DOF calculation, not the thousands
        of source pixel parameters.
        """
        model_img = self.model_image_from_params(params)
        resid = (self.data - model_img) / self.noise_map
        chi2 = jnp.sum(resid**2)

        n_pix = self.data.size
        if n_params is None:
            # Use physical params only for CorrFields
            n_params = self._num_physical_params
        dof = max(int(n_pix - n_params), 1)
        chi2_red = chi2 / dof
        return float(chi2_red)

    def get_source_pixels_from_params(self, params: dict) -> jnp.ndarray:
        """
        Extract source pixel values from parameter dict.

        Parameters
        ----------
        params : Dict
            Parameter dictionary (e.g., from multistart optimization).

        Returns
        -------
        jnp.ndarray
            Source pixel values, shape (num_pixels, num_pixels).
        """
        params_jax = {
            k: jnp.asarray(v) if isinstance(v, (list, np.ndarray)) else v
            for k, v in params.items()
        }
        source_pixels = self.source_field.model(params_jax)
        if source_pixels.ndim == 1:
            source_pixels = source_pixels.reshape((self.num_pixels, self.num_pixels))
        return source_pixels
