"""
LensImage construction utilities.
"""

from typing import Optional, Tuple
import numpy as np

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.LensImage.lens_image import LensImage

# Optional Correlated Field imports
try:
    import herculens as hcl
    CorrelatedField = hcl.CorrelatedField
    HAS_CORR_FIELD = True
except (ImportError, AttributeError):
    CorrelatedField = None
    HAS_CORR_FIELD = False


def make_lens_image(
    img: np.ndarray,
    psf_kernel: np.ndarray,
    noise_map: np.ndarray,
    pixel_grid: PixelGrid,
    ps_grid: PixelGrid,
    supersampling_factor: int = 5,
    convolution_type: str = "jax_scipy_fft",
    # Source model selection
    use_source_shapelets: bool = True,
    shapelets_n_max: int = 6,
    use_corr_fields: bool = False,
    corr_field_num_pixels: int = 80,
    corr_field_interpolation: str = "fast_bilinear",
    source_arc_mask: Optional[np.ndarray] = None,
) -> Tuple:
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
        Use Shapelets basis for source.
    shapelets_n_max : int
        Maximum shapelet order.
    use_corr_fields : bool
        Use Correlated Fields for source.
    corr_field_num_pixels : int
        Number of pixels per side for source grid.
    corr_field_interpolation : str
        Interpolation type for pixelated source.
    source_arc_mask : np.ndarray, optional
        Boolean mask for adaptive source grid.

    Returns
    -------
    tuple
        (lens_image, noise, psf, mass_model, lens_light_model,
         source_light_model, point_source_model)
    """
    npix_y, npix_x = img.shape

    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel)
    noise = Noise(npix_x, npix_y, noise_map=noise_map)

    mass_model = MassModel(["EPL", "SHEAR"])
    lens_light_model = LightModel(["SERSIC_ELLIPSE"])

    # Source light model selection
    if use_corr_fields:
        if not HAS_CORR_FIELD:
            raise ImportError(
                "Correlated Fields require herculens with CorrelatedField support."
            )
        source_light_model = LightModel(
            hcl.PixelatedLight(
                interpolation_type=corr_field_interpolation,
                adaptive_grid=True,
                allow_extrapolation=False,
            ),
            kwargs_pixelated=dict(num_pixels=corr_field_num_pixels),
        )
    elif use_source_shapelets:
        source_light_model = LightModel(
            ["SERSIC_ELLIPSE", "SHAPELETS"],
            shapelets_n_max=shapelets_n_max,
        )
    else:
        source_light_model = LightModel(["SERSIC_ELLIPSE"])

    point_source_model = PointSourceModel(["IMAGE_POSITIONS"], mass_model, ps_grid)

    kwargs_numerics = dict(
        supersampling_factor=supersampling_factor,
        convolution_type=convolution_type,
        supersampling_convolution=False,
    )

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
    mean_intensity: Optional[float] = None,
    offset_std: Tuple[float, float] = (0.5, 1e-3),
    loglogavgslope: Tuple[float, float] = (-6.0, 0.5),
    fluctuations: Tuple[float, float] = (1.0, 0.5),
    flexibility: Optional[Tuple[float, float]] = None,
    asperity: Optional[Tuple[float, float]] = None,
    cropped_border_size: int = 20,
    exponentiate: bool = True,
):
    """
    Create a CorrelatedField instance for source modeling.

    Parameters
    ----------
    source_model : LightModel
        The PixelatedLight source model.
    img : np.ndarray
        Observed image (for estimating mean intensity).
    num_pixels : int
        Number of pixels per side for source grid.
    mean_intensity : float, optional
        Mean source intensity. If None, estimated from data.
    offset_std : tuple
        Prior (mean, std) for offset standard deviation.
    loglogavgslope : tuple
        Prior (mean, std) for log-log average slope.
    fluctuations : tuple
        Prior (mean, std) for fluctuation amplitude.
    flexibility, asperity : tuple, optional
        Additional GP priors.
    cropped_border_size : int
        Border cropping for FFT.
    exponentiate : bool
        Exponentiate field for positivity.

    Returns
    -------
    CorrelatedField
        Configured correlated field for numpyro sampling.
    """
    if not HAS_CORR_FIELD:
        raise ImportError("CorrelatedField requires herculens with support.")

    if mean_intensity is None:
        mean_intensity = float(np.percentile(img, 70))
        mean_intensity = max(mean_intensity, 1e-4)

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
