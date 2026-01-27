"""
Run Configuration for Alpaca Pipeline
======================================

Edit the settings below, then call ``load_config()`` to get a ready-to-use
``PipelineConfig`` object.
"""

import datetime

from alpaca.config import (
    PipelineConfig,
    PSFReconstructionConfig,
    GradientDescentConfig,
    SamplerConfig,
    PlottingConfig,
    CorrFieldConfig,
)

# =============================================================================
# DATA IDENTIFICATION
# =============================================================================
BASE_DIR = "."
RUNG = 2
CODE_ID = 1       # or read from env: int(os.environ.get("LENS_CODE", 1))
SEED = 120        # or read from env: int(os.environ.get("LENS_SEED", 120))

# =============================================================================
# SOURCE LIGHT MODEL  (choose ONE)
# =============================================================================
# Option 1: Shapelets (analytic basis functions)
USE_SHAPELETS = True
SHAPELETS_N_MAX = 6

# Option 2: Correlated Fields (pixelated source with GP prior)
USE_CORR_FIELDS = False  # set True to use instead of Shapelets

# =============================================================================
# CORRELATED FIELD HYPERPARAMETERS  (only used if USE_CORR_FIELDS = True)
# =============================================================================
CORR_FIELD_NUM_PIXELS = 80
CORR_FIELD_LOGLOGAVGSLOPE = (-6.0, 0.5)   # smoothness prior (smaller = smoother)
CORR_FIELD_FLUCTUATIONS = (1.0, 0.5)       # fluctuation amplitude prior
CORR_FIELD_MEAN_INTENSITY = None            # None = auto-estimate from data
CORR_FIELD_BORDER_SIZE = 20                 # border cropping for FFT

# Arc mask (defines where source light can appear)
CORR_FIELD_ARC_MASK_INNER = 0.3   # inner radius in arcsec (mask lens center)
CORR_FIELD_ARC_MASK_OUTER = 2.5   # outer radius in arcsec
CORR_FIELD_CUSTOM_MASK_PATH = None # e.g. "./my_custom_mask.npy"

# =============================================================================
# SAMPLER
# =============================================================================
SAMPLER = "nuts"   # "nuts", "nautilus", or "default"

# =============================================================================
# PIPELINE PHASE TOGGLES
# =============================================================================
RUN_PSF_RECONSTRUCTION = False
RUN_MULTISTART = True
RUN_SAMPLING = True

# =============================================================================
# LIKELIHOOD SETTINGS
# =============================================================================
USE_TIME_DELAYS = True

# Ray shooting consistency (penalises scatter in ray-traced source positions)
USE_RAYSHOOT_CONSISTENCY = True
RAYSHOOT_CONSISTENCY_SIGMA = 0.0002  # arcsec, astrometric uncertainty floor
USE_SOURCEPOSITION_RAYSHOOT = True   # compare to sampled source position

# Ray shooting systematic error as free parameter
USE_RAYSHOOT_SYSTEMATIC_ERROR = True
RAYSHOOT_SYS_ERROR_MIN = 0.00005  # arcsec (0.05 mas)
RAYSHOOT_SYS_ERROR_MAX = 0.005    # arcsec (5 mas)

# =============================================================================
# GRADIENT DESCENT / MULTI-START
# =============================================================================
N_MULTISTART = 50

# Retry settings: if chi2_red > threshold after optimization, restart
MAX_RETRY_ITERATIONS = 1    # 1 = no retry, 2-3 for automatic retries
CHI2_RED_THRESHOLD = 2.0

# =============================================================================
# PSF RECONSTRUCTION
# =============================================================================
N_PSF_ITERATIONS = 4
PSF_MULTISTART_STARTS = 20
PSF_ROTATION_MODE = "90"             # "none", "180", or "90"
PSF_NEGATIVE_SIGMA_THRESHOLD = 2.0   # mask pixels with residual < -N*sigma

# =============================================================================
# SAMPLER-SPECIFIC
# =============================================================================
NUTS_NUM_WARMUP = 3000
NUTS_NUM_SAMPLES = 5000
NAUTILUS_N_LIVE = 1000

# =============================================================================
# PRIOR: LENS GAMMA (power-law slope)
# =============================================================================
LENS_GAMMA_PRIOR_TYPE = "normal"  # "uniform" or "normal" (truncated)
LENS_GAMMA_PRIOR_LOW = 1.5
LENS_GAMMA_PRIOR_HIGH = 2.5
LENS_GAMMA_PRIOR_SIGMA = 0.2     # only used when type is "normal"

# =============================================================================
# POINT SOURCE DETECTION
# =============================================================================
PS_MIN_SEP = 0.08          # minimum separation between images (arcsec)
PS_FALLBACK_TO_TRUTH = True  # use truth positions if detection fails


# =============================================================================
# BUILD PIPELINE CONFIG  -- nothing below here needs editing
# =============================================================================

def load_config() -> PipelineConfig:
    """Build a ``PipelineConfig`` from the settings defined above."""

    use_shapelets = USE_SHAPELETS
    use_corr_fields = USE_CORR_FIELDS

    # Enforce mutual exclusivity
    if use_shapelets and use_corr_fields:
        print("Both USE_SHAPELETS and USE_CORR_FIELDS are True; "
              "setting USE_SHAPELETS to False.")
        use_shapelets = False

    # Output directory with source model suffix
    if use_corr_fields:
        suffix = "CorrField"
    elif use_shapelets:
        suffix = "Shapelets"
    else:
        suffix = "Sersic"
    dateandtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = f"./{suffix}/run_{dateandtime}/"

    config = PipelineConfig(
        base_dir=BASE_DIR,
        rung=RUNG,
        code_id=CODE_ID,
        seed=SEED,

        # Source model
        use_source_shapelets=use_shapelets,
        shapelets_n_max=SHAPELETS_N_MAX,
        use_corr_fields=use_corr_fields,
        corr_field_config=CorrFieldConfig(
            num_pixels=CORR_FIELD_NUM_PIXELS,
            mean_intensity=CORR_FIELD_MEAN_INTENSITY,
            loglogavgslope=CORR_FIELD_LOGLOGAVGSLOPE,
            fluctuations=CORR_FIELD_FLUCTUATIONS,
            cropped_border_size=CORR_FIELD_BORDER_SIZE,
            arc_mask_inner_radius=CORR_FIELD_ARC_MASK_INNER,
            arc_mask_outer_radius=CORR_FIELD_ARC_MASK_OUTER,
            custom_arc_mask_path=CORR_FIELD_CUSTOM_MASK_PATH,
        ),

        boost_noise_around_ps=True,

        # Point source detection
        ps_min_sep=PS_MIN_SEP,
        ps_fallback_to_truth=PS_FALLBACK_TO_TRUTH,

        # Likelihood
        use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
        rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
        use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
        rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
        rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,

        # Phase toggles
        run_psf_reconstruction=RUN_PSF_RECONSTRUCTION,
        run_multistart=RUN_MULTISTART,
        run_sampling=RUN_SAMPLING,

        psf_config=PSFReconstructionConfig(
            n_iterations=N_PSF_ITERATIONS,
            multistart_starts_per_iteration=PSF_MULTISTART_STARTS,
            starred_cutout_size=99,
            starred_supersampling_factor=3,
            starred_mask_other_peaks=True,
            starred_mask_radius=8,
            starred_rotation_mode=PSF_ROTATION_MODE,
            starred_negative_sigma_threshold=PSF_NEGATIVE_SIGMA_THRESHOLD,
            parallelized=True,
        ),

        gradient_descent_config=GradientDescentConfig(
            n_starts_initial=N_MULTISTART,
            adam_steps_initial=500,
            lbfgs_maxiter_initial=600,
            adam_lr=5e-3,
            adam_warmup_fraction=0.1,
            adam_grad_clip=10.0,
            adam_use_cosine_decay=True,
            use_time_delays=USE_TIME_DELAYS,
            use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
            rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
            use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
            rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
            rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,
            max_retry_iterations=MAX_RETRY_ITERATIONS,
            chi2_red_threshold=CHI2_RED_THRESHOLD,
        ),

        sampler_config=SamplerConfig(
            sampler=SAMPLER,
            use_time_delays=USE_TIME_DELAYS,
            use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
            rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
            use_source_position_rayshoot=USE_SOURCEPOSITION_RAYSHOOT,
            use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
            rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
            rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,
            nuts_num_warmup=NUTS_NUM_WARMUP,
            nuts_num_samples=NUTS_NUM_SAMPLES,
            nuts_num_chains=None,
            nuts_target_accept=0.8,
            nautilus_n_live=NAUTILUS_N_LIVE,
            nautilus_use_jax=True,
            nautilus_use_multi_device=True,
            n_posterior_samples=5000,
            lens_gamma_prior_type=LENS_GAMMA_PRIOR_TYPE,
            lens_gamma_prior_low=LENS_GAMMA_PRIOR_LOW,
            lens_gamma_prior_high=LENS_GAMMA_PRIOR_HIGH,
            lens_gamma_prior_sigma=LENS_GAMMA_PRIOR_SIGMA,
        ),

        plotting_config=PlottingConfig(
            save_plots=True,
            plot_format="png",
            dpi=300,
            plot_corner=True,
            plot_chains=True,
            plot_posterior_draws=3,
            corner_params=[
                "lens_theta_E", "lens_gamma", "D_dt",
            ],
        ),
        output_subdir=output_dir,
    )

    # Print summary
    print(f"Configuration:")
    print(f"  Data: rung={RUNG}, code={CODE_ID}, seed={SEED}")
    if use_corr_fields:
        print(f"  Source model: Correlated Fields (pixels={CORR_FIELD_NUM_PIXELS})")
        print(f"    - loglogavgslope: {CORR_FIELD_LOGLOGAVGSLOPE}")
        print(f"    - fluctuations: {CORR_FIELD_FLUCTUATIONS}")
    elif use_shapelets:
        print(f"  Source model: Shapelets (n_max={SHAPELETS_N_MAX})")
    else:
        print(f"  Source model: Sersic only")
    print(f"  Sampler: {SAMPLER}")
    print(f"  Output directory: {output_dir}")
    print(f"  Time delays in likelihood: {USE_TIME_DELAYS}")
    print(f"  Ray shooting consistency: {USE_RAYSHOOT_CONSISTENCY} "
          f"(sigma={RAYSHOOT_CONSISTENCY_SIGMA}, "
          f"use_source_pos={USE_SOURCEPOSITION_RAYSHOOT})")
    if USE_RAYSHOOT_SYSTEMATIC_ERROR:
        print(f"  Ray shooting systematic error: free parameter "
              f"[{RAYSHOOT_SYS_ERROR_MIN}, {RAYSHOOT_SYS_ERROR_MAX}] arcsec")
    if MAX_RETRY_ITERATIONS > 1:
        print(f"  Gradient descent retry: up to {MAX_RETRY_ITERATIONS} "
              f"iterations if chi2_red > {CHI2_RED_THRESHOLD:.2f}")
    print(f"  Lens gamma prior: {LENS_GAMMA_PRIOR_TYPE} "
          f"[{LENS_GAMMA_PRIOR_LOW}, {LENS_GAMMA_PRIOR_HIGH}]", end="")
    if LENS_GAMMA_PRIOR_TYPE == "normal":
        print(f" (sigma={LENS_GAMMA_PRIOR_SIGMA})")
    else:
        print()
    print(f"  Point source detection: min_sep={PS_MIN_SEP}\", "
          f"fallback_to_truth={PS_FALLBACK_TO_TRUTH}")

    return config
