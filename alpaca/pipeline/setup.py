"""
alpaca.pipeline.setup

Pipeline setup helpers: rebuilding ProbModel with a new PSF,
matching detected point sources to truth positions, and loading
time-delay data.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
from herculens.LensImage.lens_image import LensImage

from alpaca.models.prob_model import (
    ProbModel,
    ProbModelCorrField,
    create_corr_field,
    make_lens_image,
)
from alpaca.utils.cosmology import parse_good_team_lens_info_file, parse_lens_info_file


def _build_prob_model_with_psf_and_lens_image(
    setup: dict,
    psf_kernel: np.ndarray
) -> tuple[ProbModel, LensImage]:
    """
    Rebuild ProbModel and LensImage using a new PSF kernel.

    This function creates a new LensImage with the reconstructed PSF and
    returns both the updated ProbModel and LensImage for use in the pipeline.
    Supports both standard ProbModel (with/without Shapelets) and ProbModelCorrField.

    Parameters
    ----------
    setup : Dict
        Setup dictionary from setup_tdlmc_lens containing all model components.
    psf_kernel : np.ndarray
        New PSF kernel (e.g., from STARRED reconstruction).

    Returns
    -------
    prob_model : ProbModel or ProbModelCorrField
        New ProbModel using the reconstructed PSF.
    lens_image : LensImage
        New LensImage using the reconstructed PSF.
    """
    # Get settings from existing prob_model
    base_prob_model = setup.get("prob_model")
    use_source_shapelets = bool(getattr(base_prob_model, "use_source_shapelets", False))
    shapelets_n_max = int(getattr(base_prob_model, "shapelets_n_max", 6))
    use_rayshoot_systematic_error = bool(getattr(base_prob_model, "use_rayshoot_systematic_error", False))
    rayshoot_sys_error_min = float(getattr(base_prob_model, "rayshoot_sys_error_min", 0.00005))
    rayshoot_sys_error_max = float(getattr(base_prob_model, "rayshoot_sys_error_max", 0.005))

    # Check if using Correlated Fields
    use_corr_fields = bool(getattr(base_prob_model, "use_corr_fields", False))
    source_field = setup.get("source_field")
    source_arc_mask = setup.get("source_arc_mask")

    # Infer numerics settings from the existing lens_image
    base_lens_image = setup.get("lens_image")
    if hasattr(base_lens_image, "_kwargs_numerics"):
        numerics = base_lens_image._kwargs_numerics
        supersampling_factor = numerics.get("supersampling_factor", 5)
        convolution_type = numerics.get("convolution_type", "jax_scipy_fft")
    else:
        supersampling_factor = 5
        convolution_type = "jax_scipy_fft"

    # Get correlated field settings if applicable
    corr_field_num_pixels = 80
    corr_field_interpolation = 'fast_bilinear'
    if use_corr_fields and source_field is not None:
        # Try to get settings from the existing source field
        if hasattr(source_field, 'num_pixels'):
            corr_field_num_pixels = source_field.num_pixels
        if hasattr(source_field, 'interpolation_type'):
            corr_field_interpolation = source_field.interpolation_type

    # Build new LensImage with the new PSF
    lens_image_new, _, _, _, _, source_light_model, _ = make_lens_image(
        img=np.asarray(setup["img"]),
        psf_kernel=np.asarray(psf_kernel),
        noise_map=np.asarray(setup["noise_map"]),
        pixel_grid=setup["pixel_grid"],
        ps_grid=setup["ps_grid"],
        supersampling_factor=int(supersampling_factor),
        convolution_type=str(convolution_type),
        use_source_shapelets=bool(use_source_shapelets),
        shapelets_n_max=int(shapelets_n_max),
        use_corr_fields=bool(use_corr_fields),
        corr_field_num_pixels=int(corr_field_num_pixels),
        corr_field_interpolation=str(corr_field_interpolation),
        source_arc_mask=source_arc_mask,
    )

    # Build new ProbModel with the new LensImage
    if use_corr_fields and source_field is not None:
        # Rebuild source field with new source model
        # Note: We need to recreate the CorrelatedField since the source model changed
        new_source_field = create_corr_field(
            source_model=source_light_model,
            img=np.asarray(setup["img"]),
            num_pixels=corr_field_num_pixels,
            # Use existing source field settings
            mean_intensity=getattr(source_field, '_mean_intensity', None),
            offset_std=getattr(source_field, '_offset_std', (0.5, 1e-3)),
            loglogavgslope=getattr(source_field, '_loglogavgslope', (-6., 0.5)),
            fluctuations=getattr(source_field, '_fluctuations', (1., 0.5)),
            flexibility=getattr(source_field, '_flexibility', None),
            asperity=getattr(source_field, '_asperity', None),
            cropped_border_size=getattr(source_field, 'cropped_border_size', 20),
            exponentiate=getattr(source_field, 'exponentiate', True),
        )
        prob_model = ProbModelCorrField(
            lens_image=lens_image_new,
            img=np.asarray(setup["img"]),
            noise_map=np.asarray(setup["noise_map"]),
            xgrid=np.asarray(setup["xgrid"]),
            ygrid=np.asarray(setup["ygrid"]),
            x0s=np.asarray(setup["x0s"]),
            y0s=np.asarray(setup["y0s"]),
            peak_vals=np.asarray(setup["peak_vals"]),
            source_field=new_source_field,
            num_pixels=corr_field_num_pixels,
            use_rayshoot_systematic_error=use_rayshoot_systematic_error,
            rayshoot_sys_error_min=rayshoot_sys_error_min,
            rayshoot_sys_error_max=rayshoot_sys_error_max,
        )
        # Update setup with new source field
        setup["source_field"] = new_source_field
    else:
        # Standard ProbModel (with or without Shapelets)
        prob_model = ProbModel(
            lens_image=lens_image_new,
            img=np.asarray(setup["img"]),
            noise_map=np.asarray(setup["noise_map"]),
            xgrid=np.asarray(setup["xgrid"]),
            ygrid=np.asarray(setup["ygrid"]),
            x0s=np.asarray(setup["x0s"]),
            y0s=np.asarray(setup["y0s"]),
            peak_vals=np.asarray(setup["peak_vals"]),
            use_source_shapelets=bool(use_source_shapelets),
            shapelets_n_max=int(shapelets_n_max),
            use_rayshoot_systematic_error=use_rayshoot_systematic_error,
            rayshoot_sys_error_min=rayshoot_sys_error_min,
            rayshoot_sys_error_max=rayshoot_sys_error_max,
        )

    return prob_model, lens_image_new


def _match_point_sources(x_det, y_det, x_truth, y_truth):
    """
    Match detected point-source positions to truth positions using
    a minimal total squared distance assignment.
    """
    x_det = np.asarray(x_det, float)
    y_det = np.asarray(y_det, float)
    x_truth = np.asarray(x_truth, float)
    y_truth = np.asarray(y_truth, float)

    if x_det.shape != y_det.shape:
        raise ValueError("Detected x/y arrays must have the same shape.")
    if x_truth.shape != y_truth.shape:
        raise ValueError("Truth x/y arrays must have the same shape.")
    if x_det.size != x_truth.size:
        raise ValueError("Detected and truth arrays must have the same length.")

    n = x_det.size
    indices = range(n)
    best_perm = None
    best_cost = None
    for perm in itertools.permutations(indices):
        cost = 0.0
        for i, j in enumerate(perm):
            dx = x_det[i] - x_truth[j]
            dy = y_det[i] - y_truth[j]
            cost += dx * dx + dy * dy
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    return list(best_perm)


def _load_time_delay_data(base_dir, rung, code_id, seed, x0s, y0s, verbose=False,
                          fallback_to_truth=False):
    """
    Load measured time delays and align them to the detected image ordering.

    If fallback_to_truth=True and detection count doesn't match truth count,
    uses truth positions instead of detected positions.

    Returns
    -------
    measured_delays : np.ndarray
    delay_errors : np.ndarray
    det_labels : list
    used_fallback : bool
        True if truth positions were used instead of detected positions
    truth_positions : tuple or None
        (x_truth, y_truth) if fallback was used, None otherwise
    """
    good_team_path = os.path.join(
        base_dir,
        f"TDC/rung{rung}/code{code_id}/f160w-seed{seed}/lens_info_for_Good_team.txt",
    )
    open_box_path = os.path.join(
        base_dir,
        f"TDC/rung{rung}_open_box/code{code_id}/f160w-seed{seed}/lens_all_info.txt",
    )

    if not os.path.exists(good_team_path):
        raise FileNotFoundError(f"Missing measured delays file: {good_team_path}")
    if not os.path.exists(open_box_path):
        raise FileNotFoundError(
            f"Missing open-box file for image matching: {open_box_path}"
        )

    good_info = parse_good_team_lens_info_file(good_team_path)
    delays = good_info["time_delays_BCD_minus_A"]["values"]
    errors = good_info["time_delays_BCD_minus_A"]["errors"]
    if delays is None or errors is None:
        raise ValueError(f"Time delays not found in {good_team_path}")
    delays = np.asarray(delays, float)
    errors = np.asarray(errors, float)
    if delays.size != errors.size:
        raise ValueError("Time-delay values and errors must have same length.")

    truth_info = parse_lens_info_file(open_box_path)
    x_truth = np.asarray(truth_info["AGN_light"]["image_plane"]["x"], float)
    y_truth = np.asarray(truth_info["AGN_light"]["image_plane"]["y"], float)

    nps_detected = len(np.atleast_1d(x0s))
    nps_truth = x_truth.size
    used_fallback = False
    truth_positions = None

    # Check for mismatch
    if nps_detected != nps_truth:
        if fallback_to_truth:
            if verbose:
                print(f"WARNING: Detected {nps_detected} images but truth has {nps_truth}.")
                print("         Using truth positions as fallback.")
            # Use truth positions instead
            x0s = x_truth
            y0s = y_truth
            used_fallback = True
            truth_positions = (x_truth, y_truth)
        else:
            raise ValueError(
                f"Detected {nps_detected} images but truth has {nps_truth} positions. "
                f"Set ps_fallback_to_truth=True to use truth positions."
            )

    nps = len(np.atleast_1d(x0s))
    if nps > 4:
        raise ValueError("Time-delay mapping supports up to 4 images (A-D).")
    if nps < 2:
        raise ValueError("Need at least two point-source images for time delays.")

    labels = ["A", "B", "C", "D"][:nps]
    perm = _match_point_sources(x0s, y0s, x_truth, y_truth)
    det_labels = [labels[j] for j in perm]

    delay_map = {"A": 0.0}
    err_map = {"A": 0.0}
    for idx, label in enumerate(labels[1:]):
        if idx >= delays.size:
            break
        delay_map[label] = float(delays[idx])
        err_map[label] = float(errors[idx])

    t0 = delay_map.get(det_labels[0])
    s0 = err_map.get(det_labels[0])
    if t0 is None or s0 is None:
        raise ValueError("Could not map reference image to delay data.")

    measured = []
    measured_err = []
    for i in range(1, nps):
        label = det_labels[i]
        if label not in delay_map:
            raise ValueError(f"No time-delay data for image {label}.")
        ti = delay_map[label]
        si = err_map[label]
        measured.append(ti - t0)
        measured_err.append(np.sqrt(si * si + s0 * s0))

    if verbose:
        mapping = ", ".join(f"img{i}={lab}" for i, lab in enumerate(det_labels))
        print(f"Time-delay mapping: {mapping}")

    return (np.asarray(measured, float), np.asarray(measured_err, float),
            det_labels, used_fallback, truth_positions)
