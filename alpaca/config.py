"""
Configuration dataclass for the TDLMC pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import json


@dataclass
class PipelineConfig:
    """Configuration for the lens modeling pipeline."""

    # ==========================================================================
    # Data and paths
    # ==========================================================================
    base_path: str = ""
    rung: int = 2
    code_id: int = 1
    seed: int = 119

    # ==========================================================================
    # Image and grid settings
    # ==========================================================================
    pix_scl: float = 0.08  # arcsec/pixel
    ps_oversample: int = 2
    supersampling_factor: int = 5
    convolution_type: str = "jax_scipy_fft"

    # ==========================================================================
    # Point source detection
    # ==========================================================================
    n_ps_detect: int = 4
    lens_mask_radius: float = 0.5  # arcsec
    local_win: int = 3
    min_peak_frac: float = 0.15
    min_sep: float = 0.18  # arcsec

    # ==========================================================================
    # Noise handling
    # ==========================================================================
    boost_noise_around_ps: bool = False
    boost_noise_f_max: float = 5.0
    boost_noise_frac_min_sep: float = 0.4
    boost_noise_min_npix: float = 2.5
    boost_noise_max_npix: float = 6.0

    # ==========================================================================
    # Source model selection (mutually exclusive)
    # ==========================================================================
    use_source_shapelets: bool = True
    use_corr_fields: bool = False

    # Shapelets settings
    shapelets_n_max: int = 6
    shapelets_beta_min: float = 0.02
    shapelets_beta_max: float = 0.6
    shapelets_amp_sigma: float = 1.0

    # Correlated Fields settings
    corr_field_num_pixels: int = 80
    corr_field_mean_intensity: Optional[float] = None
    corr_field_offset_std: Tuple[float, float] = (0.5, 1e-3)
    corr_field_loglogavgslope: Tuple[float, float] = (-6.0, 0.5)
    corr_field_fluctuations: Tuple[float, float] = (1.0, 0.5)
    corr_field_flexibility: Optional[Tuple[float, float]] = None
    corr_field_asperity: Optional[Tuple[float, float]] = None
    corr_field_cropped_border_size: int = 20
    corr_field_exponentiate: bool = True
    corr_field_interpolation: str = "fast_bilinear"

    # Arc mask for correlated fields
    arc_mask_inner_radius: float = 0.3
    arc_mask_outer_radius: float = 2.5
    custom_arc_mask_path: Optional[str] = None

    # ==========================================================================
    # Likelihood terms
    # ==========================================================================
    # Time delays
    use_time_delays: bool = True
    measured_delays: Optional[Tuple[float, ...]] = None
    delay_errors: Optional[Tuple[float, ...]] = None

    # Ray shooting consistency
    use_rayshoot_consistency: bool = False
    rayshoot_consistency_sigma: float = 0.0002  # arcsec
    use_source_position_rayshoot: bool = True

    # Systematic error model
    use_rayshoot_systematic_error: bool = False
    rayshoot_sys_error_min: float = 0.00005  # arcsec
    rayshoot_sys_error_max: float = 0.005    # arcsec

    # ==========================================================================
    # Prior settings
    # ==========================================================================
    D_dt_min: float = 500.0   # Mpc
    D_dt_max: float = 10000.0  # Mpc

    lens_gamma_prior_type: Literal["uniform", "normal"] = "normal"
    lens_gamma_prior_low: float = 1.2
    lens_gamma_prior_high: float = 2.8
    lens_gamma_prior_sigma: float = 0.25

    # ==========================================================================
    # Gradient descent settings
    # ==========================================================================
    random_seed: int = 42
    n_starts_initial: int = 50
    n_top_for_refinement: int = 5
    n_refinement_perturbations: int = 10
    perturbation_scale: float = 0.1

    adam_steps_initial: int = 500
    adam_steps_refinement: int = 750
    adam_lr: float = 5e-3
    adam_warmup_fraction: float = 0.1
    adam_grad_clip: float = 10.0
    adam_use_cosine_decay: bool = True

    lbfgs_maxiter_initial: int = 600
    lbfgs_maxiter_refinement: int = 1000
    lbfgs_tol: float = 1e-5

    max_retry_iterations: int = 1
    chi2_red_threshold: float = 2.0

    # ==========================================================================
    # NUTS settings
    # ==========================================================================
    nuts_num_warmup: int = 1000
    nuts_num_samples: int = 2000
    nuts_num_chains: Optional[int] = None  # None = use all devices
    nuts_target_accept: float = 0.8
    nuts_max_tree_depth: int = 10
    nuts_chain_method: Literal["parallel", "vectorized", "sequential"] = "parallel"

    # ==========================================================================
    # Nautilus settings
    # ==========================================================================
    nautilus_n_live: int = 1000
    nautilus_n_batch: Optional[int] = None  # Batch size for vectorized evaluation
    nautilus_pool: Optional[int] = None  # Worker pool size for parallelization
    nautilus_n_posterior_samples: Optional[int] = None  # Resample to this many samples

    # ==========================================================================
    # PSF reconstruction settings
    # ==========================================================================
    use_psf_reconstruction: bool = False  # Enable STARRED PSF reconstruction
    psf_reconstruction_iterations: int = 1  # Number of PSF reconstruction iterations
    psf_cutout_size: int = 99  # Cutout size for STARRED
    psf_supersampling_factor: int = 3  # STARRED upsampling factor
    psf_mask_other_peaks: bool = True  # Mask other PS positions in cutouts
    psf_mask_radius: int = 8  # Radius for masking other peaks
    psf_rotation_mode: Optional[str] = "none"  # Rotation augmentation: "none", "180", "90"
    psf_negative_sigma_threshold: Optional[float] = None  # Threshold for negative pixel masking

    def validate(self):
        """Validate configuration settings."""
        if self.use_source_shapelets and self.use_corr_fields:
            raise ValueError(
                "Cannot use both Shapelets and Correlated Fields. "
                "Set only one of use_source_shapelets or use_corr_fields to True."
            )
        if not self.use_source_shapelets and not self.use_corr_fields:
            raise ValueError(
                "Must enable either use_source_shapelets or use_corr_fields."
            )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
