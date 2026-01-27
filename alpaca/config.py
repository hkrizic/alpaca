"""
Configuration dataclasses for the Alpaca lens modeling pipeline.

Defines all configuration options as structured, validated dataclasses with
sensible default values. Configuration is organized hierarchically:

- PSFReconstructionConfig: PSF reconstruction phase
- GradientDescentConfig: Two-phase MAP optimization
- SamplerConfig: Posterior sampling (NUTS, Nautilus)
- PlottingConfig: Plot generation
- CorrFieldConfig: Correlated Field source model
- PipelineConfig: Master configuration combining all phases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PSFReconstructionConfig:
    """Configuration for PSF reconstruction phase."""
    n_iterations: int = 3
    multistart_starts_per_iteration: int = 10
    starred_cutout_size: int = 99
    starred_supersampling_factor: int = 3
    starred_mask_other_peaks: bool = True
    starred_mask_radius: int = 8
    starred_rotation_mode: str = "none"
    starred_negative_sigma_threshold: float | None = None
    run_multistart: bool = True
    parallelized: bool = True
    verbose: bool = True


@dataclass
class GradientDescentConfig:
    """Configuration for two-phase gradient descent optimization.

    The optimization proceeds in two phases:
    1. Initial exploration: Many random starts optimized in parallel
    2. Refinement: Top results are perturbed and re-optimized with more iterations
    """
    # Phase 1: Initial exploration
    n_starts_initial: int = 50           # Number of initial random starts
    adam_steps_initial: int = 500        # Adam steps per start in Phase 1
    lbfgs_maxiter_initial: int = 600     # L-BFGS iterations in Phase 1

    # Phase 2: Refinement
    n_top_for_refinement: int = 5        # How many top results to refine
    n_refinement_perturbations: int = 10 # Perturbations per top result
    perturbation_scale: float = 0.1      # Scale of Gaussian perturbations
    adam_steps_refinement: int = 750     # Adam steps in Phase 2
    lbfgs_maxiter_refinement: int = 1000 # L-BFGS iterations in Phase 2

    # Adam optimizer settings (shared between phases)
    adam_lr: float = 5e-3
    adam_warmup_fraction: float = 0.1    # 10% warmup
    adam_grad_clip: float = 10.0         # gradient clipping norm
    adam_use_cosine_decay: bool = True   # cosine annealing after warmup

    # L-BFGS settings
    lbfgs_tol: float = 1e-5

    # General settings
    random_seed: int = 73
    verbose: bool = True

    # Time delay likelihood
    use_time_delays: bool = False

    # Ray shooting consistency
    use_rayshoot_consistency: bool = False
    rayshoot_consistency_sigma: float = 0.0002  # arcsec, astrometric uncertainty floor

    # Ray shooting systematic error as free parameter
    use_rayshoot_systematic_error: bool = False
    rayshoot_sys_error_min: float = 0.00005  # arcsec (0.05 mas)
    rayshoot_sys_error_max: float = 0.005    # arcsec (5 mas)

    # Retry logic: restart if reduced chi^2 exceeds threshold
    max_retry_iterations: int = 1            # Max total iterations (1 = no retry)
    chi2_red_threshold: float = 2.0          # Retry if chi^2_red > threshold


@dataclass
class SamplerConfig:
    """Configuration for posterior sampling."""
    sampler: Literal["nuts", "nautilus", "default"] = "default"

    # Time-delay likelihood
    use_time_delays: bool = True

    # Ray shooting consistency term
    # Adds penalty if ray-traced image positions don't converge to same source
    use_rayshoot_consistency: bool = False
    rayshoot_consistency_sigma: float = 0.0002  # arcsec, astrometric uncertainty floor
    # If True, compare ray-traced positions to sampled source position (src_center_x/y)
    # If False, compare to mean of ray-traced positions (original behavior)
    use_source_position_rayshoot: bool = True
    # Ray shooting systematic error as free parameter
    use_rayshoot_systematic_error: bool = False
    rayshoot_sys_error_min: float = 0.00005  # arcsec (0.05 mas)
    rayshoot_sys_error_max: float = 0.005    # arcsec (5 mas)

    # NUTS (NumPyro) settings
    nuts_num_warmup: int = 1000
    nuts_num_samples: int = 2000
    nuts_num_chains: int | None = None  # None = use all devices
    nuts_target_accept: float = 0.8

    # Nautilus settings
    nautilus_n_live: int = 1000
    nautilus_use_jax: bool = True
    nautilus_use_multi_device: bool = True
    nautilus_batch_per_device: int = 64
    nautilus_use_uniform_priors: bool = True

    # Prior configuration for lens_gamma (power-law slope)
    # Allows switching between uniform and normal (truncated) priors
    lens_gamma_prior_type: Literal["uniform", "normal"] = "normal"
    lens_gamma_prior_low: float = 1.2    # Lower bound
    lens_gamma_prior_high: float = 2.8   # Upper bound
    lens_gamma_prior_sigma: float = 0.25 # Std dev for normal prior (centered on MAP)

    # Common
    random_seed: int = 42
    n_posterior_samples: int = 5000
    verbose: bool = True


@dataclass
class PlottingConfig:
    """Configuration for plot generation."""
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300

    # PSF plots
    plot_psf_comparison: bool = True
    plot_psf_residuals: bool = True

    # Model plots
    plot_model_summary: bool = True
    plot_residuals: bool = True

    # Sampling plots
    plot_corner: bool = True
    plot_chains: bool = True
    plot_posterior_draws: int = 3  # Number of random posterior draws to plot

    # Corner plot parameters
    corner_params: list[str] | None = None  # None = auto-select
    max_corner_params: int = 10


@dataclass
class CorrFieldConfig:
    """Configuration for Correlated Field source model.

    Correlated Fields use a pixelated source representation with a Gaussian
    Process prior that enforces smoothness. This is an alternative to Shapelets
    for modeling complex source morphologies.

    The key hyperparameters control the smoothness and amplitude of the source:
    - loglogavgslope: Controls smoothness (more negative = smoother)
    - fluctuations: Controls amplitude of deviations from mean
    - offset_std: Controls variation in overall brightness level
    """
    # Grid settings
    num_pixels: int = 80  # Number of pixels per side for source grid

    # Mean intensity (in log-space since exponentiate=True)
    mean_intensity: float | None = None  # If None, estimated from data

    # GP prior hyperparameters (mean, std) for the power spectrum
    offset_std: tuple[float, float] = (0.5, 1e-3)  # Prior on offset std
    loglogavgslope: tuple[float, float] = (-6., 0.5)  # Smoothness (smaller = smoother)
    fluctuations: tuple[float, float] = (1., 0.5)  # Amplitude fluctuations

    # Optional additional priors (set to None to disable)
    flexibility: tuple[float, float] | None = None
    asperity: tuple[float, float] | None = None

    # Technical settings
    cropped_border_size: int = 20  # Border cropping for FFT
    exponentiate: bool = True  # Exponentiate field to ensure positivity
    interpolation_type: str = 'fast_bilinear'  # Interpolation method
    adaptive_grid: bool = True  # Adapt grid to arc mask

    # Arc mask settings (for adaptive grid)
    arc_mask_inner_radius: float = 0.3  # Inner radius to mask lens center (arcsec)
    arc_mask_outer_radius: float = 2.5  # Outer radius for source region (arcsec)
    custom_arc_mask_path: str | None = None  # Path to custom mask (FITS or .npy)


@dataclass
class PipelineConfig:
    """Master configuration for the full pipeline."""
    # Data identification
    base_dir: str = "."
    rung: int = 2
    code_id: int = 1
    seed: int = 120

    # Model settings - Source light representation
    # Note: use_source_shapelets and use_corr_fields are mutually exclusive
    use_source_shapelets: bool = True
    shapelets_n_max: int = 6
    use_corr_fields: bool = False  # Use Correlated Fields instead of Shapelets
    corr_field_config: CorrFieldConfig = field(default_factory=CorrFieldConfig)

    # Noise boosting around point sources
    boost_noise_around_ps: bool = True
    boost_noise_kwargs: dict = field(default_factory=lambda: {
        "f_max": 5.0,
        "radius": None,
        "frac_min_sep": 0.4,
        "min_npix": 2.5,
        "max_npix": 6.0,
    })

    # Point source detection settings
    ps_min_sep: float = 0.18  # Minimum separation between detected images (arcsec)
    ps_fallback_to_truth: bool = True  # Use truth positions if detection fails/mismatches

    # Likelihood settings
    # Ray shooting consistency: penalizes scatter in ray-traced source positions
    use_rayshoot_consistency: bool = False
    rayshoot_consistency_sigma: float = 0.0002  # arcsec, astrometric uncertainty floor
    # Ray shooting systematic error as free parameter
    use_rayshoot_systematic_error: bool = False
    rayshoot_sys_error_min: float = 0.00005  # arcsec (0.05 mas)
    rayshoot_sys_error_max: float = 0.005    # arcsec (5 mas)

    # Phase toggles
    run_psf_reconstruction: bool = True
    run_multistart: bool = True
    run_sampling: bool = True

    # Phase configurations
    psf_config: PSFReconstructionConfig = field(default_factory=PSFReconstructionConfig)
    gradient_descent_config: GradientDescentConfig = field(default_factory=GradientDescentConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    plotting_config: PlottingConfig = field(default_factory=PlottingConfig)

    # Output
    output_subdir: str = "pipeline_output"
