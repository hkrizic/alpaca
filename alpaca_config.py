#!/usr/bin/env python
"""
ALPACA Pipeline Configuration

All configuration settings for the lens modeling pipeline.
Edit this file to customize your run, then execute with run_alpaca.py
"""

import os
import datetime

# =============================================================================
# DATA IDENTIFICATION
# =============================================================================
BASE_DIR = "."  # Directory containing TDC and TDC_results folders
RUNG = 2
# Read CODE_ID and SEED from environment variables (set by SLURM array job)
CODE_ID = 1
SEED = 119

# =============================================================================
# SOURCE MODEL SETTINGS
# =============================================================================
# Choose one: Shapelets OR Correlated Fields
USE_SHAPELETS = True
USE_CORR_FIELDS = False
SHAPELETS_N_MAX = 6

# Shapelets hyperparameters
SHAPELETS_BETA_MIN = 0.02
SHAPELETS_BETA_MAX = 0.6
SHAPELETS_AMP_SIGMA = 1.0

# Correlated Fields settings (if USE_CORR_FIELDS = True)
CORR_FIELD_NUM_PIXELS = 80
CORR_FIELD_OFFSET_STD = (0.5, 1e-3)
CORR_FIELD_LOGLOGAVGSLOPE = (-6.0, 0.5)
CORR_FIELD_FLUCTUATIONS = (1.0, 0.5)
ARC_MASK_INNER_RADIUS = 0.3
ARC_MASK_OUTER_RADIUS = 2.5

# =============================================================================
# SAMPLER SELECTION
# =============================================================================
# Options: "nuts" or "nautilus"
SAMPLER = "nuts"

# Phase toggles
USE_PSF_RECONSTRUCTION = False
RUN_MULTISTART = True
RUN_SAMPLING = True

# =============================================================================
# TIME-DELAY LIKELIHOOD
# =============================================================================
USE_TIME_DELAYS = True
# Set these if you have measured time delays:
MEASURED_DELAYS = None  # e.g., (10.5, 25.3, 35.1) for 3 delays relative to image 0
DELAY_ERRORS = None     # e.g., (1.0, 1.2, 1.1) corresponding errors

# =============================================================================
# RAY SHOOTING CONSISTENCY
# =============================================================================
USE_RAYSHOOT_CONSISTENCY = True
RAYSHOOT_CONSISTENCY_SIGMA = 0.0002  # arcsec, astrometric uncertainty floor

# If True, compare ray-traced positions to sampled source position (src_center_x/y)
# If False, compare to mean of ray-traced positions
USE_SOURCEPOSITION_RAYSHOOT = True

# Systematic error model for ray shooting
USE_RAYSHOOT_SYSTEMATIC_ERROR = True
RAYSHOOT_SYS_ERROR_MIN = 5e-5   # arcsec
RAYSHOOT_SYS_ERROR_MAX = 0.005  # arcsec

# =============================================================================
# PRIOR SETTINGS
# =============================================================================
# D_dt prior bounds (time-delay distance in Mpc)
D_DT_MIN = 500.0
D_DT_MAX = 10000.0

# Lens gamma (power-law slope) prior
# Options: "uniform" or "normal" (truncated normal centered on MAP value)
LENS_GAMMA_PRIOR_TYPE = "normal"
LENS_GAMMA_PRIOR_LOW = 1.5
LENS_GAMMA_PRIOR_HIGH = 2.5
LENS_GAMMA_PRIOR_SIGMA = 0.2

# =============================================================================
# POINT SOURCE DETECTION
# =============================================================================
PS_MIN_SEP = 0.08  # Minimum separation between images (arcsec)

# Noise boosting around point sources
BOOST_NOISE_AROUND_PS = True
BOOST_NOISE_F_MAX = 5.0
BOOST_NOISE_FRAC_MIN_SEP = 0.4
BOOST_NOISE_MIN_NPIX = 2.5
BOOST_NOISE_MAX_NPIX = 6.0

# =============================================================================
# GRADIENT DESCENT OPTIMIZATION
# =============================================================================
RANDOM_SEED = 42

# Multi-start settings
N_MULTISTART_INITIAL = 50
N_TOP_FOR_REFINEMENT = 5
N_REFINEMENT_PERTURBATIONS = 10
PERTURBATION_SCALE = 0.1

# Adam optimizer settings
ADAM_STEPS_INITIAL = 500
ADAM_STEPS_REFINEMENT = 750
ADAM_LR = 5e-3
ADAM_WARMUP_FRACTION = 0.1
ADAM_GRAD_CLIP = 10.0
ADAM_USE_COSINE_DECAY = True

# L-BFGS settings
LBFGS_MAXITER_INITIAL = 600
LBFGS_MAXITER_REFINEMENT = 1000
LBFGS_TOL = 1e-5

# =============================================================================
# NUTS SAMPLER SETTINGS
# =============================================================================
NUTS_NUM_WARMUP = 3000
NUTS_NUM_SAMPLES = 5000
NUTS_NUM_CHAINS = None  # None = use all available devices
NUTS_TARGET_ACCEPT = 0.8
NUTS_MAX_TREE_DEPTH = 10
NUTS_CHAIN_METHOD = "parallel"  # "parallel", "vectorized", or "sequential"

# =============================================================================
# NAUTILUS SAMPLER SETTINGS
# =============================================================================
NAUTILUS_N_LIVE = 1000
NAUTILUS_N_BATCH = None  # Batch size for vectorized evaluation
NAUTILUS_POOL = None  # Worker pool size for parallelization
NAUTILUS_N_POSTERIOR_SAMPLES = None  # Resample to this many samples (None = keep all)

# =============================================================================
# PSF RECONSTRUCTION SETTINGS
# =============================================================================
# STARRED PSF reconstruction (requires starred-astro package)
PSF_RECONSTRUCTION_ITERATIONS = 1  # Number of PSF reconstruction iterations
PSF_CUTOUT_SIZE = 99  # Cutout size for STARRED
PSF_SUPERSAMPLING_FACTOR = 3  # STARRED upsampling factor
PSF_MASK_OTHER_PEAKS = True  # Mask other PS positions in cutouts
PSF_MASK_RADIUS = 8  # Radius for masking other peaks
PSF_ROTATION_MODE = "none"  # Rotation augmentation: "none", "180", "90"
PSF_NEGATIVE_SIGMA_THRESHOLD = None  # Threshold for negative pixel masking

# =============================================================================
# OUTPUT SETTINGS
# =============================================================================
# Output directory (auto-generated if None)
OUTPUT_DIR = None  # Will be set automatically based on source type and timestamp


def get_output_dir():
    """Generate output directory path."""
    if OUTPUT_DIR is not None:
        return OUTPUT_DIR
    source_type = "CorrField" if USE_CORR_FIELDS else "Shapelets"
    dateandtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"./{source_type}/run_{dateandtime}/"


def build_config():
    """Build PipelineConfig from settings."""
    from alpaca import PipelineConfig

    return PipelineConfig(
        # Data paths
        base_path=BASE_DIR,
        rung=RUNG,
        code_id=CODE_ID,
        seed=SEED,

        # Source model
        use_source_shapelets=USE_SHAPELETS,
        use_corr_fields=USE_CORR_FIELDS,
        shapelets_n_max=SHAPELETS_N_MAX,
        shapelets_beta_min=SHAPELETS_BETA_MIN,
        shapelets_beta_max=SHAPELETS_BETA_MAX,
        shapelets_amp_sigma=SHAPELETS_AMP_SIGMA,

        # Correlated Fields settings
        corr_field_num_pixels=CORR_FIELD_NUM_PIXELS,
        corr_field_offset_std=CORR_FIELD_OFFSET_STD,
        corr_field_loglogavgslope=CORR_FIELD_LOGLOGAVGSLOPE,
        corr_field_fluctuations=CORR_FIELD_FLUCTUATIONS,
        arc_mask_inner_radius=ARC_MASK_INNER_RADIUS,
        arc_mask_outer_radius=ARC_MASK_OUTER_RADIUS,

        # Point source detection
        min_sep=PS_MIN_SEP,

        # Noise boosting
        boost_noise_around_ps=BOOST_NOISE_AROUND_PS,
        boost_noise_f_max=BOOST_NOISE_F_MAX,
        boost_noise_frac_min_sep=BOOST_NOISE_FRAC_MIN_SEP,
        boost_noise_min_npix=BOOST_NOISE_MIN_NPIX,
        boost_noise_max_npix=BOOST_NOISE_MAX_NPIX,

        # Time delays
        use_time_delays=USE_TIME_DELAYS,
        measured_delays=MEASURED_DELAYS,
        delay_errors=DELAY_ERRORS,

        # Ray shooting consistency
        use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
        rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
        use_source_position_rayshoot=USE_SOURCEPOSITION_RAYSHOOT,
        use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
        rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
        rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,

        # Prior bounds
        D_dt_min=D_DT_MIN,
        D_dt_max=D_DT_MAX,
        lens_gamma_prior_type=LENS_GAMMA_PRIOR_TYPE,
        lens_gamma_prior_low=LENS_GAMMA_PRIOR_LOW,
        lens_gamma_prior_high=LENS_GAMMA_PRIOR_HIGH,
        lens_gamma_prior_sigma=LENS_GAMMA_PRIOR_SIGMA,

        # Optimization settings
        random_seed=RANDOM_SEED,
        n_starts_initial=N_MULTISTART_INITIAL,
        n_top_for_refinement=N_TOP_FOR_REFINEMENT,
        n_refinement_perturbations=N_REFINEMENT_PERTURBATIONS,
        perturbation_scale=PERTURBATION_SCALE,

        # Adam settings
        adam_steps_initial=ADAM_STEPS_INITIAL,
        adam_steps_refinement=ADAM_STEPS_REFINEMENT,
        adam_lr=ADAM_LR,
        adam_warmup_fraction=ADAM_WARMUP_FRACTION,
        adam_grad_clip=ADAM_GRAD_CLIP,
        adam_use_cosine_decay=ADAM_USE_COSINE_DECAY,

        # L-BFGS settings
        lbfgs_maxiter_initial=LBFGS_MAXITER_INITIAL,
        lbfgs_maxiter_refinement=LBFGS_MAXITER_REFINEMENT,
        lbfgs_tol=LBFGS_TOL,

        # NUTS settings
        nuts_num_warmup=NUTS_NUM_WARMUP,
        nuts_num_samples=NUTS_NUM_SAMPLES,
        nuts_num_chains=NUTS_NUM_CHAINS,
        nuts_target_accept=NUTS_TARGET_ACCEPT,
        nuts_max_tree_depth=NUTS_MAX_TREE_DEPTH,
        nuts_chain_method=NUTS_CHAIN_METHOD,

        # Nautilus settings
        nautilus_n_live=NAUTILUS_N_LIVE,
        nautilus_n_batch=NAUTILUS_N_BATCH,
        nautilus_pool=NAUTILUS_POOL,
        nautilus_n_posterior_samples=NAUTILUS_N_POSTERIOR_SAMPLES,

        # PSF reconstruction settings
        use_psf_reconstruction=USE_PSF_RECONSTRUCTION,
        psf_reconstruction_iterations=PSF_RECONSTRUCTION_ITERATIONS,
        psf_cutout_size=PSF_CUTOUT_SIZE,
        psf_supersampling_factor=PSF_SUPERSAMPLING_FACTOR,
        psf_mask_other_peaks=PSF_MASK_OTHER_PEAKS,
        psf_mask_radius=PSF_MASK_RADIUS,
        psf_rotation_mode=PSF_ROTATION_MODE,
        psf_negative_sigma_threshold=PSF_NEGATIVE_SIGMA_THRESHOLD,
    )


def print_config_summary():
    """Print configuration summary."""
    source_type = "Correlated Fields" if USE_CORR_FIELDS else "Shapelets"

    print("=" * 70)
    print("ALPACA: Automated Lens-modelling Pipeline for Accelerated TD Cosmography Analysis")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Data: rung={RUNG}, code={CODE_ID}, seed={SEED}")
    print(f"  Source model: {source_type}", end="")
    if USE_SHAPELETS:
        print(f" (n_max={SHAPELETS_N_MAX})")
    else:
        print()
    print(f"  Sampler: {SAMPLER}")
    print(f"  Time delays in likelihood: {USE_TIME_DELAYS}")
    print(f"  Ray shooting consistency: {USE_RAYSHOOT_CONSISTENCY}", end="")
    if USE_RAYSHOOT_CONSISTENCY:
        print(f" (sigma={RAYSHOOT_CONSISTENCY_SIGMA}, use_source_pos={USE_SOURCEPOSITION_RAYSHOOT})")
    else:
        print()
    if USE_RAYSHOOT_SYSTEMATIC_ERROR:
        print(f"  Ray shooting systematic error: [{RAYSHOOT_SYS_ERROR_MIN}, {RAYSHOOT_SYS_ERROR_MAX}] arcsec")
    print(f"  Lens gamma prior: {LENS_GAMMA_PRIOR_TYPE} [{LENS_GAMMA_PRIOR_LOW}, {LENS_GAMMA_PRIOR_HIGH}]", end="")
    if LENS_GAMMA_PRIOR_TYPE == "normal":
        print(f" (sigma={LENS_GAMMA_PRIOR_SIGMA})")
    else:
        print()
    print(f"  D_dt prior: [{D_DT_MIN}, {D_DT_MAX}] Mpc")
    print(f"  PSF reconstruction: {USE_PSF_RECONSTRUCTION}", end="")
    if USE_PSF_RECONSTRUCTION:
        print(f" ({PSF_RECONSTRUCTION_ITERATIONS} iterations)")
    else:
        print()
    print("=" * 70)
