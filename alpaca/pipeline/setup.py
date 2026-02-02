"""
alpaca.pipeline.setup

Pipeline setup helpers: rebuilding ProbModel with a new PSF and
matching detected point sources to truth positions.

author: hkrizic
"""

from __future__ import annotations

import itertools

import numpy as np
from herculens.LensImage.lens_image import LensImage

from alpaca.models.prob_model import (
    ProbModel,
    ProbModelCorrField,
    create_corr_field,
    make_lens_image,
)


def _build_prob_model_with_psf_and_lens_image(
    setup: dict,
    psf_kernel: np.ndarray
) -> tuple[ProbModel, LensImage]:
    """
    Rebuild ProbModel and LensImage using a new PSF kernel.

    This function creates a new LensImage with the reconstructed PSF and
    returns both the updated ProbModel and LensImage for use in the pipeline.
    Supports both standard ProbModel (with/without Shapelets) and
    ProbModelCorrField.

    Parameters
    ----------
    setup : dict
        Setup dictionary returned by ``setup_lens`` containing all model
        components (grids, detected positions, noise map, etc.).
    psf_kernel : np.ndarray
        New 2-D PSF kernel (e.g. from STARRED reconstruction).

    Returns
    -------
    tuple of (ProbModel or ProbModelCorrField, LensImage)
        A two-element tuple where the first element is the newly constructed
        probabilistic model and the second is the corresponding
        ``LensImage``, both built with the supplied *psf_kernel*.
    """
    # Get settings from existing prob_model
    base_prob_model = setup.get("prob_model")
    use_source_shapelets = bool(getattr(base_prob_model, "use_source_shapelets", False))
    shapelets_n_max = int(getattr(base_prob_model, "shapelets_n_max", 6))
    use_rayshoot_systematic_error = bool(getattr(base_prob_model, "use_rayshoot_systematic_error", False))
    rayshoot_sys_error_min = float(getattr(base_prob_model, "rayshoot_sys_error_min", 0.00005))
    rayshoot_sys_error_max = float(getattr(base_prob_model, "rayshoot_sys_error_max", 0.005))
    use_image_pos_offset = bool(getattr(base_prob_model, "use_image_pos_offset", False))
    image_pos_offset_sigma = float(getattr(base_prob_model, "image_pos_offset_sigma", 0.01))

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

    # Get correlated field settings from the setup dict (stored by setup_lens)
    corr_field_num_pixels = setup.get("corr_field_num_pixels", 80)
    corr_field_interpolation = 'fast_bilinear'
    if use_corr_fields and source_field is not None:
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
            use_image_pos_offset=use_image_pos_offset,
            image_pos_offset_sigma=image_pos_offset_sigma,
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
            use_image_pos_offset=use_image_pos_offset,
            image_pos_offset_sigma=image_pos_offset_sigma,
        )

    return prob_model, lens_image_new


def _match_point_sources(x_det, y_det, x_truth, y_truth):
    """
    Match detected point-source positions to reference positions.

    The matching is determined by finding the permutation that minimises the
    total squared Euclidean distance between the two position sets (brute-force
    over all permutations).

    Parameters
    ----------
    x_det : array_like
        x-coordinates of detected point sources.
    y_det : array_like
        y-coordinates of detected point sources.
    x_truth : array_like
        x-coordinates of reference (truth) positions.
    y_truth : array_like
        y-coordinates of reference (truth) positions.

    Returns
    -------
    list of int
        Permutation array ``perm`` such that detected source *i* corresponds
        to reference source ``perm[i]``.

    Raises
    ------
    ValueError
        If the x/y arrays within a set have different shapes, or the detected
        and reference sets have different lengths.
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
