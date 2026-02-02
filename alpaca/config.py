"""
Configuration dataclasses for the Alpaca lens modeling pipeline.

author: hkrizic

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
    """Configuration for the PSF reconstruction phase.

    Controls the iterative PSF refinement using STARRED. The reconstruction
    isolates point sources, builds cutouts, and reconstructs the PSF kernel
    across multiple iterations with optional multi-start optimization.

    Attributes
    ----------
    n_iterations : int
        Number of PSF reconstruction iterations to perform.
    multistart_starts_per_iteration : int
        Number of random starts for the multi-start optimization at each
        iteration.
    starred_cutout_size : int
        Size (in pixels) of the cutout around each point source used for
        STARRED reconstruction.
    starred_supersampling_factor : int
        Supersampling factor for the STARRED PSF model.
    starred_mask_other_peaks : bool
        Whether to mask other detected peaks when reconstructing a single
        point source.
    starred_mask_radius : int
        Radius (in pixels) of the mask applied around other peaks.
    starred_rotation_mode : str
        Rotation augmentation mode for STARRED. One of ``"none"``,
        ``"90"``, or ``"all"``.
    starred_negative_sigma_threshold : float or None
        If set, clip negative pixels in the reconstructed PSF below this
        many standard deviations. ``None`` disables clipping.
    run_multistart : bool
        Whether to use multi-start optimization during PSF reconstruction.
    parallelized : bool
        Whether to parallelize multi-start evaluations across devices.
    verbose : bool
        Whether to print progress information during reconstruction.
    """
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

    1. **Initial exploration**: Many random starts optimized in parallel
       using Adam followed by L-BFGS.
    2. **Refinement**: The top results from Phase 1 are perturbed and
       re-optimized with more iterations.

    Includes automatic OOM fallback (full parallel -> chunked ->
    sequential) and optional retry logic when the reduced chi-squared
    exceeds a threshold.

    Attributes
    ----------
    n_starts_initial : int
        Number of initial random starts for Phase 1.
    adam_steps_initial : int
        Number of Adam optimizer steps per start in Phase 1.
    lbfgs_maxiter_initial : int
        Maximum L-BFGS iterations per start in Phase 1.
    n_top_for_refinement : int
        Number of top results from Phase 1 to carry into Phase 2.
    n_refinement_perturbations : int
        Number of Gaussian perturbations applied to each top result in
        Phase 2.
    perturbation_scale : float
        Standard deviation of the Gaussian perturbations applied in
        Phase 2.
    adam_steps_refinement : int
        Number of Adam optimizer steps in Phase 2.
    lbfgs_maxiter_refinement : int
        Maximum L-BFGS iterations in Phase 2.
    adam_lr : float
        Learning rate for the Adam optimizer (shared between phases).
    adam_warmup_fraction : float
        Fraction of Adam steps devoted to linear learning-rate warmup.
    adam_grad_clip : float
        Maximum gradient norm for gradient clipping in Adam.
    adam_use_cosine_decay : bool
        Whether to use cosine annealing of the learning rate after warmup.
    lbfgs_tol : float
        Convergence tolerance for L-BFGS.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress information.
    use_time_delays : bool
        Whether to include time-delay likelihood terms in the loss.
    use_rayshoot_consistency : bool
        Whether to include a ray-shooting consistency penalty in the loss.
    rayshoot_consistency_sigma : float
        Astrometric uncertainty floor (arcsec) for the ray-shooting
        consistency term.
    use_rayshoot_systematic_error : bool
        Whether to include a free systematic-error parameter for
        ray-shooting.
    rayshoot_sys_error_min : float
        Minimum allowed ray-shooting systematic error (arcsec).
    rayshoot_sys_error_max : float
        Maximum allowed ray-shooting systematic error (arcsec).
    use_image_pos_offset : bool
        Whether to add per-image position offsets for time-delay and
        ray-shooting likelihood terms.
    image_pos_offset_sigma : float
        Prior sigma (arcsec) on per-image position offsets.
    max_retry_iterations : int
        Maximum number of full optimization attempts. A value of 1 means
        no retry.
    chi2_red_threshold : float
        Reduced chi-squared threshold above which the optimization is
        retried.
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

    # Image position offset: separate offsets for TD/rayshoot likelihood terms
    # When enabled, offset_x_i and offset_y_i parameters are added for each image.
    # These offsets are ONLY applied to time-delay and ray-shooting terms, not imaging.
    use_image_pos_offset: bool = False
    image_pos_offset_sigma: float = 0.01     # arcsec, prior sigma for offsets

    # Retry logic: restart if reduced chi^2 exceeds threshold
    max_retry_iterations: int = 1            # Max total iterations (1 = no retry)
    chi2_red_threshold: float = 2.0          # Retry if chi^2_red > threshold


@dataclass
class SamplerConfig:
    """Configuration for posterior sampling.

    Supports two sampling backends: NUTS (NumPyro HMC) and Nautilus
    (nested sampling). When ``sampler`` is ``"default"``, the backend is
    chosen automatically based on the number of model parameters
    (Nautilus for <= 50 parameters, NUTS otherwise).

    Attributes
    ----------
    sampler : {"nuts", "nautilus", "default"}
        Which sampling backend to use. ``"default"`` selects
        automatically based on parameter count.
    use_time_delays : bool
        Whether to include the time-delay likelihood term.
    use_rayshoot_consistency : bool
        Whether to include a ray-shooting consistency penalty.
    rayshoot_consistency_sigma : float
        Astrometric uncertainty floor (arcsec) for the ray-shooting
        consistency term.
    use_source_position_rayshoot : bool
        If ``True``, compare ray-traced positions to the sampled source
        position (``src_center_x/y``). If ``False``, compare to the mean
        of ray-traced positions.
    use_rayshoot_systematic_error : bool
        Whether to include a free systematic-error parameter for
        ray-shooting.
    rayshoot_sys_error_min : float
        Minimum allowed ray-shooting systematic error (arcsec).
    rayshoot_sys_error_max : float
        Maximum allowed ray-shooting systematic error (arcsec).
    use_image_pos_offset : bool
        Whether to add per-image position offsets for time-delay and
        ray-shooting likelihood terms.
    image_pos_offset_sigma : float
        Prior sigma (arcsec) on per-image position offsets.
    nuts_num_warmup : int
        Number of warmup (burn-in) steps for NUTS.
    nuts_num_samples : int
        Number of posterior samples to draw per chain with NUTS.
    nuts_num_chains : int or None
        Number of MCMC chains for NUTS. ``None`` uses all available
        devices.
    nuts_target_accept : float
        Target acceptance probability for the NUTS step-size adaptation.
    nautilus_n_live : int
        Number of live points for Nautilus nested sampling.
    nautilus_use_jax : bool
        Whether Nautilus should use the JAX-accelerated likelihood.
    nautilus_use_multi_device : bool
        Whether to distribute Nautilus likelihood evaluations across
        multiple devices.
    nautilus_batch_per_device : int
        Batch size per device for Nautilus likelihood evaluations.
    nautilus_use_uniform_priors : bool
        Whether to use uniform priors (instead of truncated-normal) for
        Nautilus.
    lens_gamma_prior_type : {"uniform", "normal"}
        Prior distribution type for the power-law slope ``lens_gamma``.
    lens_gamma_prior_low : float
        Lower bound for the ``lens_gamma`` prior.
    lens_gamma_prior_high : float
        Upper bound for the ``lens_gamma`` prior.
    lens_gamma_prior_sigma : float
        Standard deviation for the normal ``lens_gamma`` prior (centered
        on the MAP estimate).
    random_seed : int
        Random seed for reproducibility.
    n_posterior_samples : int
        Number of posterior samples to retain after sampling.
    verbose : bool
        Whether to print progress information.
    """
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

    # Image position offset: separate offsets for TD/rayshoot likelihood terms
    # When enabled, offset_x_i and offset_y_i parameters are added for each image.
    # These offsets are ONLY applied to time-delay and ray-shooting terms, not imaging.
    use_image_pos_offset: bool = False
    image_pos_offset_sigma: float = 0.01     # arcsec, prior sigma for offsets

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
    """Configuration for plot generation.

    Controls which diagnostic and summary plots are produced by the
    pipeline, along with output format and resolution settings.

    Attributes
    ----------
    save_plots : bool
        Global toggle for saving any plots to disk.
    plot_format : str
        Output format for saved plots (e.g., ``"png"``, ``"pdf"``).
    dpi : int
        Resolution (dots per inch) for saved plots.
    plot_psf_comparison : bool
        Whether to generate PSF comparison plots.
    plot_psf_residuals : bool
        Whether to generate PSF residual plots.
    plot_model_summary : bool
        Whether to generate model summary plots showing data, model,
        and residuals.
    plot_residuals : bool
        Whether to generate standalone residual plots.
    plot_corner : bool
        Whether to generate corner (pair) plots of posterior samples.
    plot_chains : bool
        Whether to generate MCMC chain trace plots.
    plot_posterior_draws : int
        Number of random posterior draws to visualize.
    corner_params : list of str or None
        Explicit list of parameter names to include in the corner plot.
        ``None`` selects parameters automatically.
    max_corner_params : int
        Maximum number of parameters to include in the corner plot when
        using automatic selection.
    """
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
    """Configuration for the Correlated Field source model.

    Correlated Fields use a pixelated source representation with a Gaussian
    Process prior that enforces smoothness. This is an alternative to
    Shapelets for modeling complex source morphologies and requires the
    ``nifty8`` package.

    The key hyperparameters control the smoothness and amplitude of the
    source:

    - ``loglogavgslope``: Controls smoothness (more negative = smoother).
    - ``fluctuations``: Controls amplitude of deviations from the mean.
    - ``offset_std``: Controls variation in the overall brightness level.

    Attributes
    ----------
    num_pixels : int
        Number of pixels per side for the source grid.
    mean_intensity : float or None
        Mean intensity of the field in log-space (since
        ``exponentiate=True``). If ``None``, estimated from data.
    offset_std : tuple of float
        Prior ``(mean, std)`` on the offset standard deviation.
    loglogavgslope : tuple of float
        Prior ``(mean, std)`` on the log-log average slope of the power
        spectrum. More negative values produce smoother fields.
    fluctuations : tuple of float
        Prior ``(mean, std)`` on the amplitude of field fluctuations.
    flexibility : tuple of float or None
        Prior ``(mean, std)`` on spectral flexibility. ``None`` disables
        this prior.
    asperity : tuple of float or None
        Prior ``(mean, std)`` on spectral asperity. ``None`` disables
        this prior.
    cropped_border_size : int
        Number of border pixels cropped to mitigate FFT edge effects.
    exponentiate : bool
        Whether to exponentiate the field to ensure positivity.
    interpolation_type : str
        Interpolation method used when mapping source pixels to the
        image plane (e.g., ``'fast_bilinear'``).
    adaptive_grid : bool
        Whether to adapt the source grid extent to the arc mask.
    arc_mask_inner_radius : float
        Inner radius (arcsec) for masking the lens center in the arc
        mask.
    arc_mask_outer_radius : float
        Outer radius (arcsec) defining the source region in the arc
        mask.
    custom_arc_mask_path : str or None
        Path to a custom arc mask file (FITS or ``.npy``). ``None``
        uses the default circular mask.
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
    """Master configuration for the full ALPACA pipeline.

    Aggregates all phase-specific configurations and exposes top-level
    settings that affect multiple phases (e.g., source model choice,
    noise boosting, likelihood terms, and phase toggles). This is the
    single object passed to ``run_pipeline()``.

    Attributes
    ----------
    use_source_shapelets : bool
        Whether to use a Shapelet basis for the source light model.
        Mutually exclusive with ``use_corr_fields``.
    shapelets_n_max : int
        Maximum Shapelet order when ``use_source_shapelets`` is
        ``True``.
    use_corr_fields : bool
        Whether to use a Correlated Field pixelated source model.
        Mutually exclusive with ``use_source_shapelets``.
    corr_field_config : CorrFieldConfig
        Configuration for the Correlated Field source model, used only
        when ``use_corr_fields`` is ``True``.
    boost_noise_around_ps : bool
        Whether to boost the noise map around detected point sources to
        reduce their influence on the extended-source fit.
    boost_noise_kwargs : dict
        Keyword arguments passed to
        ``boost_noise_around_point_sources()``, controlling the boost
        amplitude, radius, and scaling.
    ps_min_sep : float
        Minimum separation (arcsec) between detected point-source
        images during detection.
    ps_fallback_to_truth : bool
        Whether to fall back to truth positions if point-source
        detection fails or yields the wrong number of images.
    use_rayshoot_consistency : bool
        Whether to include a ray-shooting consistency penalty across
        pipeline phases.
    rayshoot_consistency_sigma : float
        Astrometric uncertainty floor (arcsec) for the ray-shooting
        consistency term.
    use_rayshoot_systematic_error : bool
        Whether to include a free systematic-error parameter for
        ray-shooting.
    rayshoot_sys_error_min : float
        Minimum allowed ray-shooting systematic error (arcsec).
    rayshoot_sys_error_max : float
        Maximum allowed ray-shooting systematic error (arcsec).
    use_image_pos_offset : bool
        Whether to add per-image position offsets for time-delay and
        ray-shooting likelihood terms.
    image_pos_offset_sigma : float
        Prior sigma (arcsec) on per-image position offsets.
    run_psf_reconstruction : bool
        Whether to execute the PSF reconstruction phase.
    run_multistart : bool
        Whether to execute the gradient-descent multi-start optimization
        phase.
    run_sampling : bool
        Whether to execute the posterior sampling phase.
    psf_config : PSFReconstructionConfig
        Configuration for the PSF reconstruction phase.
    gradient_descent_config : GradientDescentConfig
        Configuration for the gradient-descent optimization phase.
    sampler_config : SamplerConfig
        Configuration for the posterior sampling phase.
    plotting_config : PlottingConfig
        Configuration for plot generation.
    output_dir : str
        Directory path where pipeline outputs are written.
    """
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

    # Image position offset: separate offsets for TD/rayshoot likelihood terms
    # When enabled, offset_x_i and offset_y_i parameters are added for each image.
    # These offsets are ONLY applied to time-delay and ray-shooting terms, not imaging.
    use_image_pos_offset: bool = False
    image_pos_offset_sigma: float = 0.01     # arcsec, prior sigma for offsets

    # Phase toggles
    run_psf_reconstruction: bool = True
    run_multistart: bool = True
    run_sampling: bool = True

    # Phase configurations
    psf_config: PSFReconstructionConfig = field(default_factory=PSFReconstructionConfig)
    gradient_descent_config: GradientDescentConfig = field(default_factory=GradientDescentConfig)
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    plotting_config: PlottingConfig = field(default_factory=PlottingConfig)

    # Output directory
    output_dir: str = "pipeline_output"
