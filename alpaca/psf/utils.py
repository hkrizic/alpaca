"""
Utility and helper functions for PSF reconstruction.

This module provides small helper functions used across the PSF reconstruction
pipeline, including directory and FITS I/O, rotation augmentation, model image
construction, masking, cropping/padding, and diagnostics printing.

author: hkrizic
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

# Local project modules
from alpaca.models.prob_model import ProbModel, make_lens_image


# ----------------------------------------------------------------------------- #
# Small utilities
# ----------------------------------------------------------------------------- #
def _ensure_dir(path: str) -> None:
    """
    Create a directory (and all parents) if it does not already exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def _save_fits(path: str, array: np.ndarray, overwrite: bool = True) -> None:
    """
    Save a NumPy array as a FITS file.

    Parameters
    ----------
    path : str
        Output file path (including ``.fits`` extension).
    array : np.ndarray
        Array to write. Will be cast to ``float64``.
    overwrite : bool, optional
        Whether to overwrite an existing file. Default is True.
    """
    arr = np.asarray(array)
    fits.writeto(path, arr.astype(np.float64), overwrite=overwrite)


def _rotation_k_list(rotation_mode: str | None) -> list[int]:
    """
    Convert a rotation mode string into a list of ``np.rot90`` *k* values.

    Parameters
    ----------
    rotation_mode : str or None
        Rotation mode. Supported values:

        - ``"none"`` / ``None`` / ``"0"`` / ``"false"`` / ``"off"`` --
          no rotation (returns ``[0]``).
        - ``"180"`` -- 0 and 180 degree rotations (returns ``[0, 2]``).
        - ``"90"`` / ``"270"`` / ``"all"`` / ``"full"`` -- four 90-degree
          rotations (returns ``[0, 1, 2, 3]``).

    Returns
    -------
    list[int]
        List of ``k`` values to pass to ``np.rot90``.

    Raises
    ------
    ValueError
        If ``rotation_mode`` is not a recognised value.
    """
    mode = "none" if rotation_mode is None else str(rotation_mode).lower()
    if mode in ("none", "0", "false", "off"):
        return [0]
    if mode in ("180",):
        return [0, 2]
    if mode in ("90", "270", "all", "full"):
        return [0, 1, 2, 3]
    raise ValueError(f"Unknown rotation_mode='{rotation_mode}'. Use 'none', '180', or '90'.")


def _augment_rotations(
    stamps: np.ndarray,
    masks: np.ndarray | None = None,
    sigma2: np.ndarray | None = None,
    *,
    rotation_mode: str | None = "none",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Augment point source stamps with rotated copies.

    For each input stamp, generates rotated copies according to
    ``rotation_mode`` and concatenates them along the first axis. Masks
    and variance maps are rotated consistently.

    Parameters
    ----------
    stamps : np.ndarray
        Array of shape ``(N, H, W)`` containing point source cutouts.
    masks : np.ndarray or None, optional
        Mask array of shape ``(N, H, W)``. Default is None.
    sigma2 : np.ndarray or None, optional
        Variance array of shape ``(N, H, W)``. Default is None.
    rotation_mode : str or None, optional
        Rotation mode string (see ``_rotation_k_list``). Default is
        ``"none"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, np.ndarray | None]
        Tuple ``(stamps_aug, masks_aug, sigma2_aug)`` with the augmented
        arrays. If the original ``masks`` or ``sigma2`` was None, the
        corresponding output is also None.
    """
    stamps = np.asarray(stamps)
    k_list = _rotation_k_list(rotation_mode)
    if len(k_list) == 1:
        return stamps, masks, sigma2

    out_stamps: list[np.ndarray] = []
    out_masks: list[np.ndarray] = []
    out_sigma2: list[np.ndarray] = []

    for i in range(stamps.shape[0]):
        for k in k_list:
            out_stamps.append(np.rot90(stamps[i], k=k))
            if masks is not None:
                out_masks.append(np.rot90(masks[i], k=k))
            if sigma2 is not None:
                out_sigma2.append(np.rot90(sigma2[i], k=k))

    stamps_aug = np.asarray(out_stamps, dtype=stamps.dtype)
    masks_aug = np.asarray(out_masks, dtype=masks.dtype) if masks is not None else None
    sigma2_aug = np.asarray(out_sigma2, dtype=sigma2.dtype) if sigma2 is not None else None
    return stamps_aug, masks_aug, sigma2_aug


def _build_model_images_from_best(
    prob_model,
    best_params: dict,
    zero_point_sources: bool = False,
) -> np.ndarray:
    """
    Build a model image from best-fit parameters.

    Constructs a forward-model image using the probabilistic model and
    the given parameter dictionary. Optionally zeros all point source
    amplitudes before computing the image.

    Parameters
    ----------
    prob_model : ProbModel
        Probabilistic lens model.
    best_params : dict
        Best-fit parameter dictionary (flat key-value pairs).
    zero_point_sources : bool, optional
        If True, set all point source amplitudes to zero before building
        the model image. Default is False.

    Returns
    -------
    np.ndarray
        2D model image array.
    """
    kwargs = prob_model.params2kwargs(best_params)

    if zero_point_sources:
        kps = kwargs.get("kwargs_point_source", [{}])
        new_kps = []
        for d in kps:
            d = dict(d)
            amp = d.get("amp", 0.0)
            d["amp"] = np.zeros_like(np.asarray(amp))
            new_kps.append(d)
        kwargs["kwargs_point_source"] = new_kps

    model_img = prob_model.lens_image.model(**kwargs)
    return np.asarray(model_img)


def _build_model_with_single_ps_zeroed(
    prob_model,
    best_params: dict,
    ps_index: int,
) -> np.ndarray:
    """
    Build a model image with a single point source amplitude set to zero.

    Constructs a forward-model image identical to the full model except
    that the amplitude of point source ``ps_index`` is zeroed out. This
    is used to isolate individual point sources by subtraction.

    Parameters
    ----------
    prob_model : ProbModel
        Probabilistic lens model.
    best_params : dict
        Best-fit parameter dictionary (flat key-value pairs).
    ps_index : int
        Index of the point source whose amplitude should be zeroed.

    Returns
    -------
    np.ndarray
        2D model image array with the specified point source removed.
    """
    kwargs = prob_model.params2kwargs(best_params)

    kps = kwargs.get("kwargs_point_source", [{}])
    new_kps = []

    for d in kps:
        d = dict(d)
        amp = np.asarray(d.get("amp", 0.0))
        new_amp = amp.copy()
        if ps_index < len(new_amp):
            new_amp[ps_index] = 0.0
        d["amp"] = new_amp
        new_kps.append(d)

    kwargs["kwargs_point_source"] = new_kps
    model_img = prob_model.lens_image.model(**kwargs)
    return np.asarray(model_img)


def _circular_mask_centered(size: int, cy: float, cx: float, radius: int) -> np.ndarray:
    """
    Return a cutout-sized mask with a zeroed circle at the given position.

    Creates a square mask of ones and sets pixels within a circular region
    of the specified radius centred at ``(cy, cx)`` to zero.

    Parameters
    ----------
    size : int
        Side length of the square mask.
    cy : float
        Y-coordinate of the circle centre (in cutout pixel coordinates).
    cx : float
        X-coordinate of the circle centre (in cutout pixel coordinates).
    radius : int
        Radius of the circular mask region in pixels.

    Returns
    -------
    np.ndarray
        2D mask of shape ``(size, size)`` with values 1.0 (keep) and
        0.0 (masked inside the circle).
    """
    mask = np.ones((size, size), dtype=float)
    cy = float(cy)
    cx = float(cx)
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(size, int(np.ceil(cy + radius + 1)))
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(size, int(np.ceil(cx + radius + 1)))

    if y1 > y0 and x1 > x0:
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
        mask[y0:y1, x0:x1][circle] = 0.0
    return mask


def _find_nonfinite_fields(kwargs_dict: dict) -> list[str]:
    """
    Return a flat list of field names containing NaN or Inf values.

    Iterates over a nested dictionary of the form
    ``{group: {name: array, ...}, ...}`` and identifies any arrays that
    contain non-finite values.

    Parameters
    ----------
    kwargs_dict : dict
        Nested dictionary mapping group names to dictionaries of named
        arrays (e.g. STARRED parameter dictionaries).

    Returns
    -------
    list[str]
        List of ``"group.name"`` strings for fields containing NaN or Inf.
    """
    bad: list[str] = []
    for group, values in kwargs_dict.items():
        if not isinstance(values, dict):
            continue
        for name, arr in values.items():
            arr_np = np.asarray(arr)
            if not np.all(np.isfinite(arr_np)):
                bad.append(f"{group}.{name}")
    return bad


def _center_crop_or_pad(kernel: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Centre-crop or zero-pad a 2D kernel to the desired shape.

    If the kernel is larger than ``target_shape``, it is centre-cropped.
    If it is smaller, it is zero-padded symmetrically.

    Parameters
    ----------
    kernel : np.ndarray
        Input 2D kernel array.
    target_shape : tuple[int, int]
        Desired output shape ``(height, width)``.

    Returns
    -------
    np.ndarray
        2D array of shape ``target_shape``.
    """
    kernel = np.asarray(kernel, float)
    ky, kx = kernel.shape
    ty, tx = target_shape

    out = np.zeros((ty, tx), dtype=float)

    y_start = max((ty - ky) // 2, 0)
    x_start = max((tx - kx) // 2, 0)
    y_end = y_start + min(ky, ty)
    x_end = x_start + min(kx, tx)

    ky_start = max((ky - ty) // 2, 0)
    kx_start = max((kx - tx) // 2, 0)
    ky_end = ky_start + (y_end - y_start)
    kx_end = kx_start + (x_end - x_start)

    out[y_start:y_end, x_start:x_end] = kernel[ky_start:ky_end, kx_start:kx_end]
    return out


def _build_lens_center_mask_map(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    center_x: float,
    center_y: float,
    radius_pix: float,
) -> np.ndarray:
    """
    Return a full-frame mask that zeros out a circular region around the lens centre.

    Creates a mask of ones with the same shape as the pixel grids, then sets
    pixels within ``radius_pix`` of the lens centre to zero. Useful for
    excluding the lens galaxy from PSF reconstruction.

    Parameters
    ----------
    xgrid : np.ndarray
        2D array of x-coordinates in arcseconds (pixel grid).
    ygrid : np.ndarray
        2D array of y-coordinates in arcseconds (pixel grid).
    center_x : float
        X-coordinate of the lens centre in arcseconds.
    center_y : float
        Y-coordinate of the lens centre in arcseconds.
    radius_pix : float
        Radius of the masked region in pixel units. Converted to arcseconds
        internally using the pixel scale inferred from ``xgrid``.

    Returns
    -------
    np.ndarray
        2D mask array (1=keep, 0=masked) with the same shape as ``xgrid``.
    """
    xgrid = np.asarray(xgrid, float)
    ygrid = np.asarray(ygrid, float)
    if radius_pix <= 0:
        return np.ones_like(xgrid, dtype=float)
    pix_scl = float(np.median(np.diff(xgrid[0])))
    radius = float(radius_pix) * pix_scl
    r = np.hypot(xgrid - float(center_x), ygrid - float(center_y))
    mask = np.ones_like(xgrid, dtype=float)
    mask[r <= radius] = 0.0
    return mask


def _safe_normalized_residual(
    data: np.ndarray, model: np.ndarray, noise_map: np.ndarray, *, noise_map_is_sigma: bool
) -> np.ndarray:
    """
    Compute the normalised residual ``(data - model) / sigma``.

    Safely handles zero, negative, and non-finite sigma values by setting
    the corresponding output pixels to zero.

    Parameters
    ----------
    data : np.ndarray
        Observed data array.
    model : np.ndarray
        Model array with the same shape as ``data``.
    noise_map : np.ndarray
        Noise map with the same shape as ``data``.
    noise_map_is_sigma : bool
        If True, ``noise_map`` contains standard deviation values. If
        False, ``noise_map`` contains variance and the square root is
        taken internally.

    Returns
    -------
    np.ndarray
        Normalised residual array ``(data - model) / sigma``.
    """
    data = np.asarray(data, float)
    model = np.asarray(model, float)
    nm = np.asarray(noise_map, float)
    sigma = nm if noise_map_is_sigma else np.sqrt(np.clip(nm, 0.0, None))
    out = np.zeros_like(data, dtype=float)
    np.divide(data - model, sigma, out=out, where=np.isfinite(sigma) & (sigma > 0))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _infer_numerics_from_prob_model(prob_model) -> tuple[int, str]:
    """
    Extract numerics settings from an existing ProbModel on a best-effort basis.

    Attempts to read the ``kwargs_numerics`` dictionary from the ProbModel's
    internal ``LensImage`` object to retrieve the super-sampling factor and
    convolution type. Falls back to sensible defaults if the attributes are
    not found.

    Parameters
    ----------
    prob_model : ProbModel
        Existing probabilistic lens model.

    Returns
    -------
    tuple[int, str]
        A tuple ``(supersampling_factor, convolution_type)`` with defaults
        of ``(5, "jax_scipy_fft")`` if extraction fails.
    """
    kwargs_numerics = getattr(getattr(prob_model, "lens_image", None), "kwargs_numerics", None)
    if kwargs_numerics is None:
        kwargs_numerics = getattr(getattr(prob_model, "lens_image", None), "_kwargs_numerics", None)
    if not isinstance(kwargs_numerics, dict):
        kwargs_numerics = {}
    supersampling_factor = int(kwargs_numerics.get("supersampling_factor", 5))
    convolution_type = str(kwargs_numerics.get("convolution_type", "jax_scipy_fft"))
    return supersampling_factor, convolution_type


def _build_prob_model_with_psf(setup: dict, psf_kernel: np.ndarray):
    """
    Rebuild a ProbModel using the same grids and noise but a different PSF kernel.

    Extracts configuration (super-sampling, convolution type, source model
    settings) from the existing ProbModel stored in ``setup``, constructs a
    new ``LensImage`` with the provided PSF kernel, and wraps it in a fresh
    ``ProbModel`` instance.

    Parameters
    ----------
    setup : dict
        Pre-built setup dictionary from ``setup_lens``, containing keys
        ``"prob_model"``, ``"img"``, ``"noise_map"``, ``"pixel_grid"``,
        ``"ps_grid"``, ``"xgrid"``, ``"ygrid"``, ``"x0s"``, ``"y0s"``,
        and ``"peak_vals"``.
    psf_kernel : np.ndarray
        New PSF kernel to use (2D array, should be normalised).

    Returns
    -------
    ProbModel
        New probabilistic lens model configured with the given PSF kernel.
    """
    base_prob_model = setup.get("prob_model")
    supersampling_factor, convolution_type = _infer_numerics_from_prob_model(base_prob_model)
    use_source_shapelets = bool(getattr(base_prob_model, "use_source_shapelets", False))
    shapelets_n_max = int(getattr(base_prob_model, "shapelets_n_max", 6))

    lens_image, *_ = make_lens_image(
        img=np.asarray(setup["img"]),
        psf_kernel=np.asarray(psf_kernel),
        noise_map=np.asarray(setup["noise_map"]),
        pixel_grid=setup["pixel_grid"],
        ps_grid=setup["ps_grid"],
        supersampling_factor=int(supersampling_factor),
        convolution_type=str(convolution_type),
        use_source_shapelets=bool(use_source_shapelets),
        shapelets_n_max=int(shapelets_n_max),
    )

    prob_model = ProbModel(
        lens_image=lens_image,
        img=np.asarray(setup["img"]),
        noise_map=np.asarray(setup["noise_map"]),
        xgrid=np.asarray(setup["xgrid"]),
        ygrid=np.asarray(setup["ygrid"]),
        x0s=np.asarray(setup["x0s"]),
        y0s=np.asarray(setup["y0s"]),
        peak_vals=np.asarray(setup["peak_vals"]),
        use_source_shapelets=bool(use_source_shapelets),
        shapelets_n_max=int(shapelets_n_max),
    )
    return prob_model


def _make_iteration_output_dirs(outdir: str, iteration_index: int) -> dict[str, str]:
    """
    Create and return a set of organised output folders for a given iteration.

    Creates a directory tree under ``outdir/Iteration_{iteration_index}/``
    with sub-directories for PSF products, models, isolated images, STARRED
    inputs/debug, residuals, and multi-start results.

    Parameters
    ----------
    outdir : str
        Root output directory.
    iteration_index : int
        1-based iteration number.

    Returns
    -------
    dict[str, str]
        Dictionary mapping descriptive keys (``"iteration_dir"``,
        ``"psf_dir"``, ``"models_dir"``, ``"isolated_dir"``,
        ``"starred_inputs_dir"``, ``"starred_debug_dir"``,
        ``"residuals_dir"``, ``"multistart_dir"``) to their absolute
        directory paths.
    """
    iter_dir = os.path.join(outdir, f"Iteration_{int(iteration_index)}")
    dirs = {
        "iteration_dir": iter_dir,
        "psf_dir": os.path.join(iter_dir, "psf"),
        "models_dir": os.path.join(iter_dir, "models"),
        "isolated_dir": os.path.join(iter_dir, "isolated_ps"),
        "starred_inputs_dir": os.path.join(iter_dir, "starred_inputs"),
        "starred_debug_dir": os.path.join(iter_dir, "starred_debug_plots"),
        "residuals_dir": os.path.join(iter_dir, "residuals"),
        "multistart_dir": os.path.join(iter_dir, "multistart"),
    }
    for p in dirs.values():
        _ensure_dir(p)
    return dirs


def _print_multistart_chi2(summary: dict, prefix: str = "[Multi-start]") -> None:
    """
    Print the best and minimum reduced chi-squared from a multi-start summary.

    Reads the ``"best_chi2_red"`` and ``"chi2_reds"`` fields from the
    summary dictionary and prints a formatted diagnostic line. Does
    nothing if the required fields are missing or empty.

    Parameters
    ----------
    summary : dict
        Multi-start summary dictionary as returned by
        ``run_gradient_descent`` or ``load_multistart_summary``.
    prefix : str, optional
        Prefix string for the printed line. Default is ``"[Multi-start]"``.
    """
    # Try direct best_chi2_red field first (new two-phase format)
    best_chi2_direct = summary.get("best_chi2_red")
    chi2_reds = summary.get("chi2_reds", None)

    if best_chi2_direct is not None:
        print(f"{prefix} best chi2_red={best_chi2_direct:.3g}")
        return

    if chi2_reds is None:
        return
    try:
        chi2_arr = np.asarray(chi2_reds, dtype=float)
        if chi2_arr.size == 0:
            return
        min_chi2 = float(np.nanmin(chi2_arr))
        print(f"{prefix} best chi2_red={min_chi2:.3g}")
    except Exception:
        return
