"""
Run Configuration for Alpaca Pipeline (Generic)
================================================

Pipeline settings for the generic entry point ``run_alpaca.py``.
Edit the settings below, then call ``load_config()`` to get a ready-to-use
``PipelineConfig`` object.

For TDLMC-specific configuration, see ``run_config_TDC.py`` instead.

author: hkrizic
"""

import datetime
import os

import numpy as np
from astropy.io import fits

from alpaca.config import (
    CorrFieldConfig,
    GradientDescentConfig,
    PipelineConfig,
    PlottingConfig,
    PSFReconstructionConfig,
    SamplerConfig,
)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR = "./results/run_{date}/"  # {date} is replaced at runtime

# =============================================================================
# SOURCE LIGHT MODEL  (choose ONE)
# =============================================================================
# Option 1: Shapelets (analytic basis functions)
USE_SHAPELETS = True
SHAPELETS_N_MAX = 8

# Option 2: Correlated Fields (pixelated source with GP prior)
USE_CORR_FIELDS = False  # set True to use instead of Shapelets

# =============================================================================
# CORRELATED FIELD HYPERPARAMETERS  (only used if USE_CORR_FIELDS = True)
# =============================================================================
CORR_FIELD_NUM_PIXELS = 80
CORR_FIELD_LOGLOGAVGSLOPE = (-4.0, 1.5)   # smoothness prior (smaller = smoother)
CORR_FIELD_FLUCTUATIONS = (1.5, 1.0)       # fluctuation amplitude prior
CORR_FIELD_MEAN_INTENSITY = None            # None = auto-estimate from data
CORR_FIELD_BORDER_SIZE = 25                 # border cropping for FFT

# Arc mask (defines where source light can appear)
CORR_FIELD_ARC_MASK_INNER = 0.3   # inner radius in arcsec (mask lens center)
CORR_FIELD_ARC_MASK_OUTER = 3.5   # outer radius in arcsec
CORR_FIELD_CUSTOM_MASK_PATH = None # e.g. "./my_custom_mask.npy"

# =============================================================================
# SAMPLER
# =============================================================================
SAMPLER = "nuts"   # "nuts", "nautilus", or "default"

# =============================================================================
# PIPELINE PHASE TOGGLES
# =============================================================================
RUN_PSF_RECONSTRUCTION = False # set False if PSF is known
RUN_MULTISTART = True
RUN_SAMPLING = True

# =============================================================================
# LIKELIHOOD SETTINGS
# =============================================================================
# Ray shooting consistency (penalises scatter in ray-traced source positions)
USE_RAYSHOOT_CONSISTENCY = True
RAYSHOOT_CONSISTENCY_SIGMA = 0.0002  # arcsec, astrometric uncertainty floor
USE_SOURCEPOSITION_RAYSHOOT = True   # compare to sampled source position

# Ray shooting systematic error as free parameter
USE_RAYSHOOT_SYSTEMATIC_ERROR = True
RAYSHOOT_SYS_ERROR_MIN = 0.00005  # arcsec (0.05 mas)
RAYSHOOT_SYS_ERROR_MAX = 0.005    # arcsec (5 mas)

# Image position offset for TD/rayshoot terms (accounts for astrometric errors)
USE_IMAGE_POS_OFFSET = True
IMAGE_POS_OFFSET_SIGMA = 0.08    # arcsec, prior sigma for offsets

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
N_PSF_ITERATIONS = 3
PSF_MULTISTART_STARTS = 50
PSF_ROTATION_MODE = "none"             # "none", "180", or "90"
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


# =============================================================================
# FUNCTIONS  -- nothing below here needs editing
# =============================================================================

def load_config(use_time_delays: bool = True) -> PipelineConfig:
    """
    Build a ``PipelineConfig`` from the settings defined above.

    Reads all module-level constants (source model, sampler, likelihood,
    gradient descent, PSF, and prior settings) and assembles them into a
    single ``PipelineConfig`` dataclass.  The output directory is
    timestamped with the current date and time.

    Parameters
    ----------
    use_time_delays : bool
        Whether the pipeline should include time delays in the likelihood.
        Typically derived from whether time-delay data is provided.

    Returns
    -------
    PipelineConfig
        Fully populated pipeline configuration object ready for
        ``run_pipeline()``.
    """
    use_shapelets = USE_SHAPELETS
    use_corr_fields = USE_CORR_FIELDS

    # Enforce mutual exclusivity
    if use_shapelets and use_corr_fields:
        print("Both USE_SHAPELETS and USE_CORR_FIELDS are True; "
              "setting USE_SHAPELETS to False.")
        use_shapelets = False

    # Output directory
    dateandtime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = OUTPUT_DIR.replace("{date}", dateandtime)
    if not output_dir.endswith("/"):
        output_dir += "/"

    config = PipelineConfig(
        output_dir=output_dir,

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
        ps_fallback_to_truth=False,

        # Likelihood
        use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
        rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
        use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
        rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
        rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,
        use_image_pos_offset=USE_IMAGE_POS_OFFSET,
        image_pos_offset_sigma=IMAGE_POS_OFFSET_SIGMA,

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
            use_time_delays=use_time_delays,
            use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
            rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
            use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
            rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
            rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,
            use_image_pos_offset=USE_IMAGE_POS_OFFSET,
            max_retry_iterations=MAX_RETRY_ITERATIONS,
            chi2_red_threshold=CHI2_RED_THRESHOLD,
        ),

        sampler_config=SamplerConfig(
            sampler=SAMPLER,
            use_time_delays=use_time_delays,
            use_rayshoot_consistency=USE_RAYSHOOT_CONSISTENCY,
            rayshoot_consistency_sigma=RAYSHOOT_CONSISTENCY_SIGMA,
            use_source_position_rayshoot=USE_SOURCEPOSITION_RAYSHOOT,
            use_rayshoot_systematic_error=USE_RAYSHOOT_SYSTEMATIC_ERROR,
            rayshoot_sys_error_min=RAYSHOOT_SYS_ERROR_MIN,
            rayshoot_sys_error_max=RAYSHOOT_SYS_ERROR_MAX,
            use_image_pos_offset=USE_IMAGE_POS_OFFSET,
            image_pos_offset_sigma=IMAGE_POS_OFFSET_SIGMA,
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
    )

    # Print summary
    print("Configuration:")
    print(f"  Output directory: {output_dir}")
    if use_corr_fields:
        print(f"  Source model: Correlated Fields (pixels={CORR_FIELD_NUM_PIXELS})")
        print(f"    - loglogavgslope: {CORR_FIELD_LOGLOGAVGSLOPE}")
        print(f"    - fluctuations: {CORR_FIELD_FLUCTUATIONS}")
    elif use_shapelets:
        print(f"  Source model: Shapelets (n_max={SHAPELETS_N_MAX})")
    else:
        print("  Source model: Sersic only")
    print(f"  Sampler: {SAMPLER}")
    print(f"  Time delays in likelihood: {use_time_delays}")
    print(f"  Ray shooting consistency: {USE_RAYSHOOT_CONSISTENCY} "
          f"(sigma={RAYSHOOT_CONSISTENCY_SIGMA}, "
          f"use_source_pos={USE_SOURCEPOSITION_RAYSHOOT})")
    if USE_RAYSHOOT_SYSTEMATIC_ERROR:
        print(f"  Ray shooting systematic error: free parameter "
              f"[{RAYSHOOT_SYS_ERROR_MIN}, {RAYSHOOT_SYS_ERROR_MAX}] arcsec")
    if USE_IMAGE_POS_OFFSET:
        print(f"  Image position offset: enabled (sigma={IMAGE_POS_OFFSET_SIGMA} arcsec)")
    if MAX_RETRY_ITERATIONS > 1:
        print(f"  Gradient descent retry: up to {MAX_RETRY_ITERATIONS} "
              f"iterations if chi2_red > {CHI2_RED_THRESHOLD:.2f}")
    print(f"  Lens gamma prior: {LENS_GAMMA_PRIOR_TYPE} "
          f"[{LENS_GAMMA_PRIOR_LOW}, {LENS_GAMMA_PRIOR_HIGH}]", end="")
    if LENS_GAMMA_PRIOR_TYPE == "normal":
        print(f" (sigma={LENS_GAMMA_PRIOR_SIGMA})")
    else:
        print()
    print(f"  Point source detection: min_sep={PS_MIN_SEP}\"")

    return config


def load_data(
    image_path: str,
    psf_path: str,
    noise_map_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load image, PSF, and noise map from FITS or .npy files.

    Each path is dispatched to ``_load_array`` which supports ``.fits``
    and ``.npy`` formats.

    Parameters
    ----------
    image_path : str
        Path to the 2-D lens image (``.fits`` or ``.npy``).
    psf_path : str
        Path to the 2-D PSF kernel (``.fits`` or ``.npy``).
    noise_map_path : str
        Path to the 2-D noise map (``.fits`` or ``.npy``).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        ``(img, psf_kernel, noise_map)`` -- the loaded 2-D arrays cast
        to ``float``.
    """
    img = _load_array(image_path)
    psf_kernel = _load_array(psf_path)
    noise_map = _load_array(noise_map_path)
    return img, psf_kernel, noise_map


def parse_positions_and_delays(
    image_positions: dict[str, tuple[float, float]],
    time_delays: dict[str, tuple[float, float]] | None,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    np.ndarray | None,
    np.ndarray | None,
    list[str],
]:
    """
    Convert user-facing dicts into arrays expected by ``run_pipeline``.

    Extracts ordered ``(x, y)`` arrays from *image_positions* and, when
    time delays are provided, builds the corresponding delay and error
    arrays relative to the first (reference) image.

    Parameters
    ----------
    image_positions : dict[str, tuple[float, float]]
        Mapping of image labels to approximate ``(x, y)`` positions in arcsec.
        These are used to label the auto-detected images so that time delays
        are matched to the correct pairs.  The first entry is the reference
        image for time delays.
    time_delays : dict[str, tuple[float, float]] or None
        Mapping of non-reference image labels to ``(delay, error)`` in days.
        Set to *None* to run without time-delay likelihood.

    Returns
    -------
    positions : tuple of (np.ndarray, np.ndarray)
        Approximate ``(x_array, y_array)`` with the reference image first.
    measured_delays : np.ndarray or None
        Delays relative to the reference image (length ``n_images - 1``).
    delay_errors : np.ndarray or None
        1-sigma errors on the delays.
    labels : list of str
        Ordered image labels (e.g. ``["A", "B", "C", "D"]``).

    Raises
    ------
    ValueError
        If a non-reference label in *image_positions* is missing from
        *time_delays*.
    """
    labels = list(image_positions.keys())
    xs = np.array([image_positions[lab][0] for lab in labels])
    ys = np.array([image_positions[lab][1] for lab in labels])

    if time_delays is not None:
        ref_label = labels[0]
        delays = []
        errors = []
        for label in labels[1:]:
            if label not in time_delays:
                raise ValueError(
                    f"Time delay for image '{label}' not found in time_delays. "
                    f"Expected keys: {labels[1:]} (all non-reference images)."
                )
            dt, err = time_delays[label]
            delays.append(dt)
            errors.append(err)
        measured_delays = np.array(delays)
        delay_errors = np.array(errors)
        print(f"  Reference image: {ref_label}")
        for label in labels[1:]:
            dt, err = time_delays[label]
            print(f"  dt({label}-{ref_label}) = {dt:.2f} +/- {err:.2f} days")
    else:
        measured_delays = None
        delay_errors = None

    return (xs, ys), measured_delays, delay_errors, labels


def _load_array(path: str) -> np.ndarray:
    """
    Load a 2-D array from a FITS or .npy file.

    Parameters
    ----------
    path : str
        Filesystem path to the data file.  Supported extensions are
        ``.fits`` / ``.fit`` and ``.npy``.

    Returns
    -------
    np.ndarray
        The loaded 2-D array cast to ``float``.

    Raises
    ------
    ValueError
        If the file extension is not ``.fits``, ``.fit``, or ``.npy``.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit"):
        return np.asarray(fits.getdata(path), dtype=float)
    elif ext == ".npy":
        return np.load(path).astype(float)
    else:
        raise ValueError(f"Unsupported file format '{ext}' for {path}. "
                         "Use .fits or .npy.")
