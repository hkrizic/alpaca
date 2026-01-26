from __future__ import annotations
"""
PSF Reconstruction Module for ALPACA

STARRED PSF reconstruction from lensed point sources.

Key approach:
- Extract CENTERED cutouts from isolated point source images
- Each PS must be centered in its stamp (required by STARRED's noise propagation)
- Use fixed_background=False to fit the pixelated narrow PSF (diffraction spikes)
- Masks are passed to STARRED via masks= parameter (not by zeroing pixels)
"""

import os
from typing import Dict, Tuple, List

import numpy as np
from astropy.io import fits
from scipy.ndimage import shift as ndimage_shift

# Local project modules - using new alpaca imports
from ..models.setup import setup_lens_system
from ..models.lens_image import make_lens_image
from ..models.prob_model import ProbModel
from ..inference.gradient_descent import run_gradient_descent
from ..benchmarking import load_multistart_summary

# --- Optional STARRED imports -------------------------------------------------
try:
    from starred.procedures.psf_routines import run_multi_steps_PSF_reconstruction
    from starred.psf.psf import PSF as StarredPSF
    from starred.psf.parameters import ParametersPSF
    from starred.plots import plot_function as pltf

    _HAS_STARRED = True
except Exception:
    run_multi_steps_PSF_reconstruction = None
    StarredPSF = None
    ParametersPSF = None
    pltf = None
    _HAS_STARRED = False


# ----------------------------------------------------------------------------- #
# Small utilities
# ----------------------------------------------------------------------------- #
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_fits(path: str, array: np.ndarray, overwrite: bool = True) -> None:
    arr = np.asarray(array)
    fits.writeto(path, arr.astype(np.float64), overwrite=overwrite)


def _rotation_k_list(rotation_mode: str | None) -> List[int]:
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
    stamps = np.asarray(stamps)
    k_list = _rotation_k_list(rotation_mode)
    if len(k_list) == 1:
        return stamps, masks, sigma2

    out_stamps: List[np.ndarray] = []
    out_masks: List[np.ndarray] = []
    out_sigma2: List[np.ndarray] = []

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
    best_params: Dict,
    zero_point_sources: bool = False,
) -> np.ndarray:
    """Build a model image from best-fit parameters."""
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
    best_params: Dict,
    ps_index: int,
) -> np.ndarray:
    """Build a model image with a single point source amplitude set to zero."""
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


def isolate_point_sources(model_with_ps: np.ndarray, model_without_ps: np.ndarray) -> np.ndarray:
    return np.asarray(model_with_ps) - np.asarray(model_without_ps)


def generate_isolated_ps_images(
    data_image: np.ndarray,
    prob_model,
    best_params: Dict,
    n_point_sources: int = 4,
) -> List[np.ndarray]:
    """
    Generate isolated point source images:
        isolated_i = data - model(with PS_i amp = 0)

    Each returned image is the FULL frame (e.g. 99x99) and is NOT shifted/recentered.
    """
    data = np.asarray(data_image)
    isolated_images: List[np.ndarray] = []
    for i in range(int(n_point_sources)):
        model_without_ps_i = _build_model_with_single_ps_zeroed(prob_model, best_params, ps_index=i)
        isolated_images.append(data - np.asarray(model_without_ps_i))
    return isolated_images


# ----------------------------------------------------------------------------- #
# Centered cutouts from isolated images (STARRED-compatible)
# ----------------------------------------------------------------------------- #
def _circular_mask_centered(size: int, cy: float, cx: float, radius: int) -> np.ndarray:
    """Return a cutout-sized mask with a zeroed circle at (cy, cx) relative to cutout."""
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


def build_centered_cutouts_from_isolated(
    isolated_images: List[np.ndarray],
    peaks_px: np.ndarray,
    cutout_size: int,
    *,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    global_mask_map: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Extract CENTERED cutouts from isolated images for STARRED.

    Each cutout is centered on its corresponding PS position, so that
    the PS appears at the center of each stamp. This is required by STARRED's
    internal noise propagation which shifts maps by -x0, -y0.

    Parameters
    ----------
    isolated_images : List[np.ndarray]
        List of isolated images, one per point source.
    peaks_px : np.ndarray
        Array of (y, x) pixel coordinates for each point source.
    cutout_size : int
        Size of square cutouts (must be odd).
    mask_other_peaks : bool
        If True, mask positions of other point sources in each cutout.
    mask_radius : int
        Radius for masking other point sources.
    global_mask_map : np.ndarray | None
        Optional global mask in full-frame coordinates.

    Returns
    -------
    cutouts : (N, cutout_size, cutout_size) float
        Centered cutouts, one per PS.
    masks : (N, cutout_size, cutout_size) float or None
        1=keep, 0=mask. None if no masking requested.
    """
    peaks_px = np.asarray(peaks_px, float)
    N = len(isolated_images)

    if len(peaks_px) != N:
        raise ValueError(f"Number of isolated images ({N}) must match number of peaks ({len(peaks_px)})")

    # Ensure odd cutout size
    if cutout_size % 2 == 0:
        cutout_size += 1

    r = cutout_size // 2

    # Get image shape from first isolated image
    img_shape = np.asarray(isolated_images[0]).shape
    ny, nx = img_shape

    cutouts: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    for j, (iso_img, (py, px)) in enumerate(zip(isolated_images, peaks_px)):
        iso_img = np.asarray(iso_img, float)
        py = float(py)
        px = float(px)
        y_int, x_int = int(py), int(px)
        shift_y = -(py - y_int)
        shift_x = -(px - x_int)

        # Extract cutout centered on PS position
        y_min = max(0, y_int - r)
        y_max = min(ny, y_int + r + 1)
        x_min = max(0, x_int - r)
        x_max = min(nx, x_int + r + 1)

        cutout = iso_img[y_min:y_max, x_min:x_max].copy()

        # Pad if near edge to ensure constant size
        if cutout.shape != (cutout_size, cutout_size):
            pad_y = (r - (y_int - y_min), r + 1 - (y_max - y_int))
            pad_x = (r - (x_int - x_min), r + 1 - (x_max - x_int))
            cutout = np.pad(cutout, (pad_y, pad_x), mode="constant", constant_values=0)

        if shift_y != 0.0 or shift_x != 0.0:
            cutout = ndimage_shift(
                cutout,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=0.0,
            )

        cutouts.append(cutout)

        # Build mask for this cutout
        mask = np.ones((cutout_size, cutout_size), dtype=float)

        # Apply global mask if provided (transform to cutout coordinates)
        if global_mask_map is not None:
            gm = np.asarray(global_mask_map)
            gm_cutout = gm[y_min:y_max, x_min:x_max].copy()
            if gm_cutout.shape != (cutout_size, cutout_size):
                gm_cutout = np.pad(gm_cutout, (pad_y, pad_x), mode="constant", constant_values=0)
            if shift_y != 0.0 or shift_x != 0.0:
                gm_cutout = ndimage_shift(
                    gm_cutout,
                    shift=(shift_y, shift_x),
                    order=0,
                    mode="constant",
                    cval=0.0,
                )
            mask *= (gm_cutout > 0.5).astype(float)

        # Mask other PS positions (in cutout coordinates)
        if mask_other_peaks:
            for i, (qy, qx) in enumerate(peaks_px):
                if i == j:
                    continue
                # Position of other PS relative to this cutout's center
                cy_in_cutout = r + (float(qy) - py)
                cx_in_cutout = r + (float(qx) - px)

                # Only mask if within cutout bounds
                if 0 <= cy_in_cutout < cutout_size and 0 <= cx_in_cutout < cutout_size:
                    mask *= _circular_mask_centered(cutout_size, cy_in_cutout, cx_in_cutout, mask_radius)

        masks_list.append(mask)

    cutouts_arr = np.asarray(cutouts, dtype=float)

    need_masks = mask_other_peaks or (global_mask_map is not None)
    if need_masks:
        masks_arr = np.asarray(masks_list, dtype=float)
        return cutouts_arr, masks_arr

    return cutouts_arr, None


def build_centered_noise_cutouts(
    noise_map: np.ndarray,
    peaks_px: np.ndarray,
    cutout_size: int,
    *,
    noise_map_is_sigma: bool = True,
) -> np.ndarray:
    """
    Extract centered noise cutouts (as sigma^2) matching the PS cutouts.

    Parameters
    ----------
    noise_map : np.ndarray
        Full-frame noise map.
    peaks_px : np.ndarray
        Array of (y, x) pixel coordinates for each point source.
    cutout_size : int
        Size of square cutouts (must be odd).
    noise_map_is_sigma : bool
        If True, noise_map contains sigma (std dev). If False, contains variance.

    Returns
    -------
    sigma2_cutouts : (N, cutout_size, cutout_size) float
        Variance (sigma^2) cutouts.
    """
    peaks_px = np.asarray(peaks_px, float)
    noise_map = np.asarray(noise_map, float)
    ny, nx = noise_map.shape

    if cutout_size % 2 == 0:
        cutout_size += 1

    r = cutout_size // 2

    cutouts: List[np.ndarray] = []

    for (py, px) in peaks_px:
        py = float(py)
        px = float(px)
        y_int, x_int = int(py), int(px)
        shift_y = -(py - y_int)
        shift_x = -(px - x_int)

        y_min = max(0, y_int - r)
        y_max = min(ny, y_int + r + 1)
        x_min = max(0, x_int - r)
        x_max = min(nx, x_int + r + 1)

        cutout = noise_map[y_min:y_max, x_min:x_max].copy()
        pad_val = np.median(cutout[cutout > 0]) if np.any(cutout > 0) else 1.0

        if cutout.shape != (cutout_size, cutout_size):
            pad_y = (r - (y_int - y_min), r + 1 - (y_max - y_int))
            pad_x = (r - (x_int - x_min), r + 1 - (x_max - x_int))
            # Pad with median noise value
            cutout = np.pad(cutout, (pad_y, pad_x), mode="constant", constant_values=pad_val)

        if shift_y != 0.0 or shift_x != 0.0:
            cutout = ndimage_shift(
                cutout,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=pad_val,
            )

        cutouts.append(cutout)

    cutouts_arr = np.asarray(cutouts, dtype=float)

    # Convert to variance if input is sigma
    if noise_map_is_sigma:
        cutouts_arr = cutouts_arr ** 2

    return cutouts_arr


def _find_nonfinite_fields(kwargs_dict: Dict) -> List[str]:
    """Return a flat list of field names containing NaN/inf values."""
    bad: List[str] = []
    for group, values in kwargs_dict.items():
        if not isinstance(values, dict):
            continue
        for name, arr in values.items():
            arr_np = np.asarray(arr)
            if not np.all(np.isfinite(arr_np)):
                bad.append(f"{group}.{name}")
    return bad


def _center_crop_or_pad(kernel: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop or zero-pad a 2D kernel to the desired shape."""
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
    """Return a full-frame mask (1=keep, 0=mask) around lens center."""
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
    """Return (data-model)/sigma, using noise_map as sigma or variance."""
    data = np.asarray(data, float)
    model = np.asarray(model, float)
    nm = np.asarray(noise_map, float)
    sigma = nm if noise_map_is_sigma else np.sqrt(np.clip(nm, 0.0, None))
    out = np.zeros_like(data, dtype=float)
    np.divide(data - model, sigma, out=out, where=np.isfinite(sigma) & (sigma > 0))
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _infer_numerics_from_prob_model(prob_model) -> tuple[int, str]:
    """Best-effort extraction of numerics settings from an existing ProbModel."""
    kwargs_numerics = getattr(getattr(prob_model, "lens_image", None), "kwargs_numerics", None)
    if kwargs_numerics is None:
        kwargs_numerics = getattr(getattr(prob_model, "lens_image", None), "_kwargs_numerics", None)
    if not isinstance(kwargs_numerics, dict):
        kwargs_numerics = {}
    supersampling_factor = int(kwargs_numerics.get("supersampling_factor", 5))
    convolution_type = str(kwargs_numerics.get("convolution_type", "jax_scipy_fft"))
    return supersampling_factor, convolution_type


def _build_prob_model_with_psf(setup: Dict, psf_kernel: np.ndarray):
    """Rebuild ProbModel using the same grids/noise but a different PSF kernel."""
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


def _make_iteration_output_dirs(outdir: str, iteration_index: int) -> Dict[str, str]:
    """Create and return a set of organized output folders for a given iteration."""
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


def _print_multistart_chi2(summary: Dict, prefix: str = "[Multi-start]") -> None:
    """Print best/min reduced chi^2 from a multistart summary (if available)."""
    try:
        best_run = int(summary.get("best_run"))
    except Exception:
        best_run = None
    chi2_reds = summary.get("chi2_reds", None)
    if best_run is None or chi2_reds is None:
        return
    try:
        chi2_arr = np.asarray(chi2_reds, dtype=float)
        if chi2_arr.size == 0 or best_run < 0 or best_run >= chi2_arr.size:
            return
        best_chi2 = float(chi2_arr[best_run])
        min_chi2 = float(np.nanmin(chi2_arr))
        stop_reason = summary.get("stop_reason", None)
        if stop_reason:
            print(f"{prefix} best chi2_red={best_chi2:.3g} (min={min_chi2:.3g}, n={chi2_arr.size}, {stop_reason})")
        else:
            print(f"{prefix} best chi2_red={best_chi2:.3g} (min={min_chi2:.3g}, n={chi2_arr.size})")
    except Exception:
        return


# ----------------------------------------------------------------------------- #
# STARRED-backed PSF update
# ----------------------------------------------------------------------------- #
def _reconstruct_psf_starred(
    peaks_px: np.ndarray,
    noise_map: np.ndarray,
    isolated_images: List[np.ndarray],
    cutout_size: int = 99,
    supersampling_factor: int = 3,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    rotation_mode: str | None = "none",
    negative_sigma_threshold: float | None = None,
    verbose: bool = True,
    debug_save_dir: str | None = None,
    export_dir: str | None = None,
    global_mask_map: np.ndarray | None = None,
    noise_map_is_sigma: bool = True,
) -> np.ndarray:
    """
    Reconstruct PSF using STARRED from isolated point source images.

    STARRED requires all stars to be centered in their stamps because
    its internal noise propagation shifts noise maps by -x0, -y0.
    """
    if not _HAS_STARRED:
        raise ImportError("STARRED not available. Install 'starred-astro' to use reconstruct_PSF.")

    peaks_px = np.asarray(peaks_px, float)
    if peaks_px.size == 0:
        raise ValueError("No point-source peaks provided to STARRED.")

    if cutout_size % 2 == 0:
        cutout_size += 1

    # Extract centered cutouts from isolated images (PS at center of each stamp)
    psf_kernel_list, masks_for_starred = build_centered_cutouts_from_isolated(
        isolated_images=isolated_images,
        peaks_px=peaks_px,
        cutout_size=int(cutout_size),
        mask_other_peaks=mask_other_peaks,
        mask_radius=int(mask_radius),
        global_mask_map=global_mask_map,
    )

    sigma2_maps_list = build_centered_noise_cutouts(
        noise_map=np.asarray(noise_map, float),
        peaks_px=peaks_px,
        cutout_size=int(cutout_size),
        noise_map_is_sigma=bool(noise_map_is_sigma),
    )

    psf_kernel_list, masks_for_starred, sigma2_maps_list = _augment_rotations(
        psf_kernel_list,
        masks=masks_for_starred,
        sigma2=sigma2_maps_list,
        rotation_mode=rotation_mode,
    )

    if negative_sigma_threshold is not None:
        sigma = np.sqrt(np.clip(sigma2_maps_list, 0.0, None))
        mask_neg = psf_kernel_list >= (-float(negative_sigma_threshold) * sigma)
        if masks_for_starred is None:
            masks_for_starred = mask_neg.astype(float)
        else:
            masks_for_starred = masks_for_starred * mask_neg.astype(float)

    psf_kernel_list = np.nan_to_num(psf_kernel_list, nan=0.0, posinf=0.0, neginf=0.0)
    if masks_for_starred is not None:
        masks_for_starred = np.nan_to_num(masks_for_starred, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2_maps_list = np.nan_to_num(sigma2_maps_list, nan=0.0, posinf=0.0, neginf=0.0)

    # Debug save (inputs to STARRED)
    if debug_save_dir is not None:
        try:
            _ensure_dir(debug_save_dir)
            _save_fits(os.path.join(debug_save_dir, "starred_psf_stamps.fits"), psf_kernel_list)
            _save_fits(os.path.join(debug_save_dir, "starred_psf_sigma2.fits"), sigma2_maps_list)
            if masks_for_starred is not None:
                _save_fits(os.path.join(debug_save_dir, "starred_psf_masks.fits"), masks_for_starred)
        except Exception as e:
            print(f"[reconstruct_PSF][DEBUG] Failed to save debug products: {e}")

    if not np.isfinite(psf_kernel_list).any():
        raise ValueError("PSF stamps contain no finite values after preprocessing.")

    # Normalise for numerical stability (do NOT touch masks)
    norm = max(float(np.nanmax(np.abs(psf_kernel_list))), 1e-6) / 100.0
    psf_kernel_list = psf_kernel_list / norm
    sigma2_maps_list = sigma2_maps_list / (norm * norm + 1e-12)

    indneg = np.where(sigma2_maps_list <= 0)
    if len(indneg[0]) > 0:
        pos = sigma2_maps_list[sigma2_maps_list > 0]
        sigma2_maps_list[indneg] = float(np.median(pos)) if pos.size else 1.0

    N, image_size, _ = np.shape(psf_kernel_list)

    if verbose:
        print(f"[reconstruct_PSF] N={N}, stamp_size={image_size}, supersampling={supersampling_factor}")

    model = StarredPSF(
        image_size=int(image_size),
        number_of_sources=int(N),
        upsampling_factor=int(supersampling_factor),
        elliptical_moffat=True,
    )

    # fixed_background=False: allows fitting the pixelated narrow PSF (diffraction spikes)
    kwargs_init, kwargs_fixed, kwargs_up, kwargs_down = model.smart_guess(
        psf_kernel_list,
        fixed_background=False,
        adjust_sky=True,
        guess_method='max'
    )

    parameters = ParametersPSF(
        kwargs_init,
        kwargs_fixed,
        kwargs_up=kwargs_up,
        kwargs_down=kwargs_down,
    )

    # --- STARRED CONFIG ---
    # Full PSF reconstruction strategy (similar to H0pe notebook):
    # 1. First fit analytical Moffat model (fix background/narrow PSF)
    # 2. Then fit the pixelated narrow PSF (fix Moffat) - this captures diffraction spikes
    # 3. Final joint refinement

    kwargs_lbfgs = {"maxiter": 1000, "restart_from_init": True}

    kwargs_optax1 = {
        "max_iterations": 5000,
        "min_iterations": None,
        "init_learning_rate": 1e-2,
        "schedule_learning_rate": True,
        "restart_from_init": False,
        "stop_at_loss_increase": False,
        "progress_bar": verbose,
        "return_param_history": True,
    }

    kwargs_optax2 = {
        "max_iterations": 5000,
        "min_iterations": None,
        "init_learning_rate": 5e-3,
        "schedule_learning_rate": True,
        "restart_from_init": False,
        "stop_at_loss_increase": False,
        "progress_bar": verbose,
        "return_param_history": True,
    }


    fitting_sequence = [["background"], ["moffat"], ["moffat"]]
    optim_list = ["l-bfgs-b", "adabelief", "adabelief"]
    kwargs_optim_list = [kwargs_lbfgs, kwargs_optax1, kwargs_optax2]

    (
        model,
        parameters,
        loss,
        kwargs_partial_list,
        LogL_list,
        loss_history_list,
    ) = run_multi_steps_PSF_reconstruction(
        psf_kernel_list,
        model,
        parameters,
        sigma2_maps_list,
        masks=masks_for_starred,
        lambda_scales=3,
        lambda_hf=3,
        fitting_sequence=fitting_sequence,
        optim_list=optim_list,
        kwargs_optim_list=kwargs_optim_list,
        regularize_full_psf=False,
        verbose=verbose,
        adjust_sky=True,
    )

    bad_params = _find_nonfinite_fields(kwargs_partial_list[-1]) if kwargs_partial_list else []
    if bad_params:
        raise ValueError("Non-finite STARRED parameters after optimisation: " + ", ".join(bad_params))

    kwargs_final = kwargs_partial_list[-1]

    # Optional: export STARRED debug products (best-effort; API differs across versions)
    if export_dir is not None:
        try:
            _ensure_dir(export_dir)
            outpath = os.path.join(export_dir, "starred_export")
            _ensure_dir(outpath)
            try:
                model.export(outpath, kwargs_final, psf_kernel_list, sigma2_maps_list, format="fits")
            except TypeError:
                model.export(outpath, kwargs_final, psf_kernel_list, sigma2_maps_list, masks=masks_for_starred, format="fits")
        except Exception as e:
            print(f"[reconstruct_PSF][DEBUG] STARRED export failed: {e}")

    # Return the low-res PSF (native grid)
    kernel_low = model.get_full_psf(**kwargs_final, norm=True, high_res=False)
    kernel_low = np.asarray(kernel_low, float)
    kernel_low = np.nan_to_num(kernel_low, nan=0.0, posinf=0.0, neginf=0.0)

    total = float(kernel_low.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("STARRED returned an invalid PSF kernel (non-finite or zero sum).")
    kernel_low /= (total + 1e-12)

    return kernel_low


def reconstruct_psf_from_star_rotations(
    star_image: np.ndarray | List[np.ndarray],
    noise_map: np.ndarray | List[np.ndarray],
    *,
    cutout_size: int | None = None,
    supersampling_factor: int = 3,
    mask_other_peaks: bool = False,
    mask_radius: int = 8,
    negative_sigma_threshold: float | None = None,
    verbose: bool = True,
    debug_save_dir: str | None = None,
    export_dir: str | None = None,
    noise_map_is_sigma: bool = True,
    rotation_mode: str = "0",
) -> np.ndarray:
    """
    Reconstruct a PSF from 90-degree rotations of one or more centered stars.

    Each star is assumed to be centered in its stamp (as produced by the
    cutout step in noise_map_creation.ipynb). For N stars, STARRED receives
    4*N images via 0/90/180/270 degree rotations.
    """
    if not _HAS_STARRED:
        raise ImportError("STARRED not available. Install 'starred-astro' to use reconstruct_PSF.")

    if isinstance(star_image, (list, tuple)):
        images = [np.asarray(img, float) for img in star_image]
    else:
        arr = np.asarray(star_image, float)
        images = [arr[i] for i in range(arr.shape[0])] if arr.ndim == 3 else [arr]

    if isinstance(noise_map, (list, tuple)):
        noise_maps = [np.asarray(nm, float) for nm in noise_map]
    else:
        arr = np.asarray(noise_map, float)
        noise_maps = [arr[i] for i in range(arr.shape[0])] if arr.ndim == 3 else [arr]

    if len(images) == 0:
        raise ValueError("No star images provided.")
    if len(noise_maps) not in (1, len(images)):
        raise ValueError("noise_map must be a single 2D map or one per star image.")

    first_shape = images[0].shape
    if any(img.ndim != 2 for img in images):
        raise ValueError("All star images must be 2D arrays.")
    if any(img.shape != first_shape for img in images):
        raise ValueError("All star images must have the same shape.")
    if any(nm.ndim != 2 for nm in noise_maps):
        raise ValueError("All noise maps must be 2D arrays.")
    if len(noise_maps) == 1:
        noise_maps = noise_maps * len(images)
    if any(nm.shape != first_shape for nm in noise_maps):
        raise ValueError("All noise maps must match the star image shape.")

    ny, nx = first_shape
    if cutout_size is None:
        cutout_size = int(min(ny, nx))

    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    peaks_px = np.asarray([[center_y, center_x]] * len(images), float)

    if len(images) > 1 and mask_other_peaks:
        if verbose:
            print("[reconstruct_PSF] mask_other_peaks disabled for multi-star cutouts.")
        mask_other_peaks = False

    noise_map_full = np.nanmedian(np.stack(noise_maps, axis=0), axis=0) if len(noise_maps) > 1 else noise_maps[0]

    return _reconstruct_psf_starred(
        peaks_px=peaks_px,
        noise_map=noise_map_full,
        isolated_images=images,
        cutout_size=int(cutout_size),
        supersampling_factor=int(supersampling_factor),
        mask_other_peaks=bool(mask_other_peaks),
        mask_radius=int(mask_radius),
        rotation_mode=rotation_mode,
        negative_sigma_threshold=negative_sigma_threshold,
        verbose=bool(verbose),
        debug_save_dir=debug_save_dir,
        export_dir=export_dir,
        global_mask_map=None,
        noise_map_is_sigma=bool(noise_map_is_sigma),
    )


def _reconstruct_psf_starred_both(
    peaks_px: np.ndarray,
    noise_map: np.ndarray,
    isolated_images: List[np.ndarray],
    cutout_size: int = 99,
    supersampling_factor: int = 3,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    rotation_mode: str | None = "none",
    negative_sigma_threshold: float | None = None,
    verbose: bool = True,
    debug_save_dir: str | None = None,
    export_dir: str | None = None,
    global_mask_map: np.ndarray | None = None,
    noise_map_is_sigma: bool = True,
) -> Dict[str, np.ndarray]:
    """Run STARRED once and return both low- and high-res PSF kernels."""
    if not _HAS_STARRED:
        raise ImportError("STARRED not available. Install 'starred-astro' to use reconstruct_PSF.")

    peaks_px = np.asarray(peaks_px, float)
    if peaks_px.size == 0:
        raise ValueError("No point-source peaks provided to STARRED.")

    if cutout_size % 2 == 0:
        cutout_size += 1

    psf_kernel_list, masks_for_starred = build_centered_cutouts_from_isolated(
        isolated_images=isolated_images,
        peaks_px=peaks_px,
        cutout_size=int(cutout_size),
        mask_other_peaks=mask_other_peaks,
        mask_radius=int(mask_radius),
        global_mask_map=global_mask_map,
    )

    sigma2_maps_list = build_centered_noise_cutouts(
        noise_map=np.asarray(noise_map, float),
        peaks_px=peaks_px,
        cutout_size=int(cutout_size),
        noise_map_is_sigma=bool(noise_map_is_sigma),
    )

    psf_kernel_list, masks_for_starred, sigma2_maps_list = _augment_rotations(
        psf_kernel_list,
        masks=masks_for_starred,
        sigma2=sigma2_maps_list,
        rotation_mode=rotation_mode,
    )

    if negative_sigma_threshold is not None:
        sigma = np.sqrt(np.clip(sigma2_maps_list, 0.0, None))
        mask_neg = psf_kernel_list >= (-float(negative_sigma_threshold) * sigma)
        if masks_for_starred is None:
            masks_for_starred = mask_neg.astype(float)
        else:
            masks_for_starred = masks_for_starred * mask_neg.astype(float)

    psf_kernel_list = np.nan_to_num(psf_kernel_list, nan=0.0, posinf=0.0, neginf=0.0)
    if masks_for_starred is not None:
        masks_for_starred = np.nan_to_num(masks_for_starred, nan=0.0, posinf=0.0, neginf=0.0)
    sigma2_maps_list = np.nan_to_num(sigma2_maps_list, nan=0.0, posinf=0.0, neginf=0.0)

    if debug_save_dir is not None:
        try:
            _ensure_dir(debug_save_dir)
            _save_fits(os.path.join(debug_save_dir, "starred_psf_stamps.fits"), psf_kernel_list)
            _save_fits(os.path.join(debug_save_dir, "starred_psf_sigma2.fits"), sigma2_maps_list)
            if masks_for_starred is not None:
                _save_fits(os.path.join(debug_save_dir, "starred_psf_masks.fits"), masks_for_starred)
        except Exception as e:
            print(f"[reconstruct_PSF][DEBUG] Failed to save debug products: {e}")

    if not np.isfinite(psf_kernel_list).any():
        raise ValueError("PSF stamps contain no finite values after preprocessing.")

    norm = max(float(np.nanmax(np.abs(psf_kernel_list))), 1e-6) / 100.0
    psf_kernel_list = psf_kernel_list / norm
    sigma2_maps_list = sigma2_maps_list / (norm * norm + 1e-12)

    indneg = np.where(sigma2_maps_list <= 0)
    if len(indneg[0]) > 0:
        pos = sigma2_maps_list[sigma2_maps_list > 0]
        sigma2_maps_list[indneg] = float(np.median(pos)) if pos.size else 1.0

    N, image_size, _ = np.shape(psf_kernel_list)
    if verbose:
        print(f"[reconstruct_PSF] N={N}, stamp_size={image_size}, supersampling={supersampling_factor}")

    model = StarredPSF(
        image_size=int(image_size),
        number_of_sources=int(N),
        upsampling_factor=int(supersampling_factor),
        elliptical_moffat=True,
    )

    kwargs_init, kwargs_fixed, kwargs_up, kwargs_down = model.smart_guess(
        psf_kernel_list,
        fixed_background=False,
        adjust_sky=True,
    )

    parameters = ParametersPSF(
        kwargs_init,
        kwargs_fixed,
        kwargs_up=kwargs_up,
        kwargs_down=kwargs_down,
    )

    kwargs_lbfgs = {"maxiter": 1000, "restart_from_init": True}
    kwargs_optax1 = {
        "max_iterations": 5000,
        "min_iterations": None,
        "init_learning_rate": 1e-2,
        "schedule_learning_rate": True,
        "restart_from_init": False,
        "stop_at_loss_increase": False,
        "progress_bar": verbose,
        "return_param_history": True,
    }
    kwargs_optax2 = {
        "max_iterations": 5000,
        "min_iterations": None,
        "init_learning_rate": 5e-3,
        "schedule_learning_rate": True,
        "restart_from_init": False,
        "stop_at_loss_increase": False,
        "progress_bar": verbose,
        "return_param_history": True,
    }

    fitting_sequence = [["background"], ["moffat"], ["moffat"]]
    optim_list = ["l-bfgs-b", "adabelief", "adabelief"]
    kwargs_optim_list = [kwargs_lbfgs, kwargs_optax1, kwargs_optax2]

    (
        model,
        parameters,
        loss,
        kwargs_partial_list,
        LogL_list,
        loss_history_list,
    ) = run_multi_steps_PSF_reconstruction(
        psf_kernel_list,
        model,
        parameters,
        sigma2_maps_list,
        masks=masks_for_starred,
        lambda_scales=3,
        lambda_hf=3,
        fitting_sequence=fitting_sequence,
        optim_list=optim_list,
        kwargs_optim_list=kwargs_optim_list,
        regularize_full_psf=False,
        verbose=verbose,
        adjust_sky=True,
    )

    bad_params = _find_nonfinite_fields(kwargs_partial_list[-1]) if kwargs_partial_list else []
    if bad_params:
        raise ValueError("Non-finite STARRED parameters after optimisation: " + ", ".join(bad_params))

    kwargs_final = kwargs_partial_list[-1]

    if export_dir is not None:
        try:
            _ensure_dir(export_dir)
            outpath = os.path.join(export_dir, "starred_export")
            _ensure_dir(outpath)
            try:
                model.export(outpath, kwargs_final, psf_kernel_list, sigma2_maps_list, format="fits")
            except TypeError:
                model.export(outpath, kwargs_final, psf_kernel_list, sigma2_maps_list, masks=masks_for_starred, format="fits")
        except Exception as e:
            print(f"[reconstruct_PSF][DEBUG] STARRED export failed: {e}")

    kernel_low = model.get_full_psf(**kwargs_final, norm=True, high_res=False)
    kernel_high = model.get_full_psf(**kwargs_final, norm=True, high_res=True)

    kernel_low = np.asarray(kernel_low, float)
    kernel_high = np.asarray(kernel_high, float)
    kernel_low = np.nan_to_num(kernel_low, nan=0.0, posinf=0.0, neginf=0.0)
    kernel_high = np.nan_to_num(kernel_high, nan=0.0, posinf=0.0, neginf=0.0)

    sum_low = float(kernel_low.sum())
    sum_high = float(kernel_high.sum())
    if not np.isfinite(sum_low) or sum_low <= 0 or not np.isfinite(sum_high) or sum_high <= 0:
        raise ValueError("STARRED returned an invalid PSF kernel (non-finite or zero sum).")

    kernel_low /= (sum_low + 1e-12)
    kernel_high /= (sum_high + 1e-12)

    return {"psf_low_res": kernel_low, "psf_high_res": kernel_high}


def reconstruct_PSF(
    current_psf: np.ndarray,
    peaks_px: np.ndarray,
    noise_map: np.ndarray,
    isolated_images: List[np.ndarray],
    *,
    cutout_size: int = 99,
    supersampling_factor: int = 3,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    rotation_mode: str | None = "none",
    negative_sigma_threshold: float | None = None,
    verbose: bool = True,
    debug_save_dir: str | None = None,
    export_dir: str | None = None,
    global_mask_map: np.ndarray | None = None,
    noise_map_is_sigma: bool = True,
) -> np.ndarray:
    """
    Reconstruct PSF using STARRED from isolated point source images.

    Parameters
    ----------
    current_psf : np.ndarray
        Current PSF kernel (used for target shape and fallback).
    peaks_px : np.ndarray
        Array of (y, x) pixel positions for each point source.
    noise_map : np.ndarray
        Noise map (sigma or variance).
    isolated_images : List[np.ndarray]
        List of isolated images, one per point source.
    cutout_size : int
        Size of cutouts to extract (must be odd).
    supersampling_factor : int
        STARRED upsampling factor.
    mask_other_peaks : bool
        Whether to mask other PS positions in each cutout.
    mask_radius : int
        Radius for masking other peaks.
    verbose : bool
        Print progress info.
    debug_save_dir : str | None
        Directory to save debug FITS files.
    global_mask_map : np.ndarray | None
        Optional global mask in full-frame coordinates.
    noise_map_is_sigma : bool
        If True, noise_map is sigma. If False, variance.

    Returns
    -------
    np.ndarray
        Reconstructed PSF kernel.
    """
    if not _HAS_STARRED:
        print("[reconstruct_PSF] STARRED not available. Returning current PSF.")
        return np.asarray(current_psf)

    target_shape = tuple(np.asarray(current_psf).shape[:2])

    try:
        kernel_new = _reconstruct_psf_starred(
            peaks_px=np.asarray(peaks_px, float),
            noise_map=noise_map,
            isolated_images=isolated_images,
            cutout_size=int(cutout_size),
            supersampling_factor=int(supersampling_factor),
            mask_other_peaks=bool(mask_other_peaks),
            mask_radius=int(mask_radius),
            rotation_mode=rotation_mode,
            negative_sigma_threshold=negative_sigma_threshold,
            verbose=bool(verbose),
            debug_save_dir=debug_save_dir,
            export_dir=export_dir,
            global_mask_map=global_mask_map,
            noise_map_is_sigma=bool(noise_map_is_sigma),
        )
        kernel_new = _center_crop_or_pad(kernel_new, target_shape=target_shape)
        kernel_new /= (kernel_new.sum() + 1e-12)
        return kernel_new
    except Exception as e:
        print(f"[reconstruct_PSF] STARRED failed: {e}")
        return np.asarray(current_psf)


# ----------------------------------------------------------------------------- #
# Pipeline integration functions
# ----------------------------------------------------------------------------- #
def run_psf_reconstruction_step(
    setup: Dict,
    best_params: Dict,
    noise_map_original: np.ndarray,
    *,
    cutout_size: int = 99,
    supersampling_factor: int = 3,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    rotation_mode: str | None = "none",
    negative_sigma_threshold: float | None = None,
    verbose: bool = True,
    save_dir: str | None = None,
) -> Tuple[np.ndarray, "ProbModel"]:
    """
    Run a single PSF reconstruction step using STARRED.

    This function is designed to integrate with the pipeline:
    1. Uses the best_params from gradient descent
    2. Generates isolated point source images
    3. Reconstructs the PSF using STARRED
    4. Rebuilds the ProbModel with the new PSF

    Parameters
    ----------
    setup : Dict
        Setup dictionary from setup_lens_system.
    best_params : Dict
        Best-fit parameters from gradient descent.
    noise_map_original : np.ndarray
        Original noise map WITHOUT noise boosting (for PSF reconstruction).
    cutout_size : int
        Size of cutouts for STARRED.
    supersampling_factor : int
        STARRED upsampling factor.
    mask_other_peaks : bool
        Mask other PS positions in cutouts.
    mask_radius : int
        Radius for masking.
    rotation_mode : str | None
        Rotation augmentation mode.
    negative_sigma_threshold : float | None
        Threshold for negative pixel masking.
    verbose : bool
        Print progress.
    save_dir : str | None
        Directory to save PSF products.

    Returns
    -------
    psf_new : np.ndarray
        Reconstructed PSF kernel.
    prob_model_new : ProbModel
        New ProbModel with the reconstructed PSF.
    """
    if not _HAS_STARRED:
        if verbose:
            print("[PSF Reconstruction] STARRED not available, skipping PSF reconstruction")
        return setup["psf_kernel"], setup["prob_model"]

    prob_model = setup["prob_model"]
    data_img = np.asarray(setup["img"])
    psf_current = np.asarray(setup["psf_kernel"])

    # Get point source positions in pixel coordinates
    x0s = np.asarray(setup["x0s"])
    y0s = np.asarray(setup["y0s"])
    pix_scl = setup.get("pix_scl", 0.08)

    # Convert to pixel coordinates (y, x format for STARRED)
    # The x0s, y0s are in arcsec, need to convert to pixel
    xgrid = np.asarray(setup["xgrid"])
    ygrid = np.asarray(setup["ygrid"])
    ny, nx = xgrid.shape

    # Find pixel positions
    peaks_px = []
    for x_arcsec, y_arcsec in zip(x0s, y0s):
        # Convert arcsec to pixel
        px = (x_arcsec - xgrid[0, 0]) / pix_scl
        py = (y_arcsec - ygrid[0, 0]) / pix_scl
        peaks_px.append([py, px])
    peaks_px = np.asarray(peaks_px, float)

    n_point_sources = len(peaks_px)

    if verbose:
        print(f"[PSF Reconstruction] Generating isolated images for {n_point_sources} point sources")

    # Generate isolated point source images
    isolated_images = generate_isolated_ps_images(
        data_image=data_img,
        prob_model=prob_model,
        best_params=best_params,
        n_point_sources=n_point_sources,
    )

    # Build lens center mask if mask_radius > 0
    global_mask_map = None
    if mask_radius > 0:
        center_x = float(best_params.get("lens_center_x", 0.0))
        center_y = float(best_params.get("lens_center_y", 0.0))
        global_mask_map = _build_lens_center_mask_map(
            xgrid, ygrid,
            center_x=center_x,
            center_y=center_y,
            radius_pix=float(mask_radius),
        )

    # Setup save directories
    debug_save_dir = None
    export_dir = None
    if save_dir is not None:
        _ensure_dir(save_dir)
        debug_save_dir = os.path.join(save_dir, "starred_inputs")
        export_dir = os.path.join(save_dir, "starred_debug")
        _ensure_dir(debug_save_dir)
        _ensure_dir(export_dir)

    # Reconstruct PSF
    if verbose:
        print("[PSF Reconstruction] Running STARRED reconstruction...")

    psf_new = reconstruct_PSF(
        current_psf=psf_current,
        peaks_px=peaks_px,
        noise_map=noise_map_original,
        isolated_images=isolated_images,
        cutout_size=cutout_size,
        supersampling_factor=supersampling_factor,
        mask_other_peaks=mask_other_peaks,
        mask_radius=mask_radius,
        rotation_mode=rotation_mode,
        negative_sigma_threshold=negative_sigma_threshold,
        verbose=verbose,
        debug_save_dir=debug_save_dir,
        export_dir=export_dir,
        global_mask_map=global_mask_map,
        noise_map_is_sigma=True,
    )

    # Save PSF products
    if save_dir is not None:
        psf_dir = os.path.join(save_dir, "psf")
        _ensure_dir(psf_dir)
        _save_fits(os.path.join(psf_dir, "psf_initial.fits"), psf_current)
        _save_fits(os.path.join(psf_dir, "psf_reconstructed.fits"), psf_new)
        _save_fits(os.path.join(psf_dir, "psf_residual.fits"), psf_new - psf_current)

        # Save isolated images
        isolated_dir = os.path.join(save_dir, "isolated_ps")
        _ensure_dir(isolated_dir)
        for i, iso_img in enumerate(isolated_images):
            _save_fits(os.path.join(isolated_dir, f"isolated_ps_{i}.fits"), iso_img)

    # Rebuild ProbModel with new PSF
    if verbose:
        print("[PSF Reconstruction] Rebuilding model with new PSF...")

    prob_model_new = _build_prob_model_with_psf(setup, psf_new)

    if verbose:
        print("[PSF Reconstruction] Done")

    return psf_new, prob_model_new


def run_psf_reconstruction_iterations(
    setup: Dict,
    noise_map_original: np.ndarray,
    *,
    n_iterations: int = 1,
    cutout_size: int = 99,
    supersampling_factor: int = 3,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    rotation_mode: str | None = "none",
    negative_sigma_threshold: float | None = None,
    # Gradient descent settings
    n_starts_initial: int = 50,
    n_top_for_refinement: int = 5,
    n_refinement_perturbations: int = 10,
    perturbation_scale: float = 0.1,
    random_seed: int = 42,
    adam_steps_initial: int = 500,
    adam_steps_refinement: int = 750,
    adam_lr: float = 5e-3,
    lbfgs_maxiter_initial: int = 600,
    lbfgs_maxiter_refinement: int = 1000,
    lbfgs_tol: float = 1e-5,
    verbose: bool = True,
    save_dir: str | None = None,
) -> Tuple[np.ndarray, "ProbModel", Dict]:
    """
    Run iterative PSF reconstruction with gradient descent between iterations.

    Each iteration:
    1. Run gradient descent with current PSF
    2. Generate isolated PS images from best-fit model
    3. Reconstruct PSF with STARRED

    Parameters
    ----------
    setup : Dict
        Setup dictionary from setup_lens_system.
    noise_map_original : np.ndarray
        Original noise map WITHOUT noise boosting.
    n_iterations : int
        Number of PSF reconstruction iterations.
    [other parameters as in run_psf_reconstruction_step]
    [gradient descent parameters]
    verbose : bool
        Print progress.
    save_dir : str | None
        Directory to save products.

    Returns
    -------
    psf_final : np.ndarray
        Final reconstructed PSF kernel.
    prob_model_final : ProbModel
        Final ProbModel with reconstructed PSF.
    results : Dict
        Results from each iteration.
    """
    if not _HAS_STARRED:
        if verbose:
            print("[PSF Reconstruction] STARRED not available, skipping PSF reconstruction")
        return setup["psf_kernel"], setup["prob_model"], {}

    psf_current = np.asarray(setup["psf_kernel"])
    prob_model_current = setup["prob_model"]

    iteration_results = []

    for it in range(1, n_iterations + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"PSF Reconstruction Iteration {it}/{n_iterations}")
            print("=" * 70)

        # Create iteration directory
        iter_save_dir = None
        if save_dir is not None:
            iter_save_dir = os.path.join(save_dir, f"psf_iteration_{it}")
            _ensure_dir(iter_save_dir)

        # Run gradient descent with current PSF
        if verbose:
            print(f"[Iteration {it}] Running gradient descent...")

        gd_outdir = iter_save_dir if iter_save_dir else "."
        gd_results = run_gradient_descent(
            prob_model=prob_model_current,
            output_dir=gd_outdir,
            n_starts_initial=n_starts_initial,
            n_top_for_refinement=n_top_for_refinement,
            n_refinement_perturbations=n_refinement_perturbations,
            perturbation_scale=perturbation_scale,
            random_seed=random_seed,
            adam_steps_initial=adam_steps_initial,
            adam_steps_refinement=adam_steps_refinement,
            adam_lr=adam_lr,
            lbfgs_maxiter_initial=lbfgs_maxiter_initial,
            lbfgs_maxiter_refinement=lbfgs_maxiter_refinement,
            lbfgs_tol=lbfgs_tol,
            verbose=verbose,
        )

        best_params = gd_results["best_params"]

        # Update setup with current prob_model for PSF reconstruction
        setup_current = dict(setup)
        setup_current["prob_model"] = prob_model_current
        setup_current["psf_kernel"] = psf_current

        # Run PSF reconstruction
        psf_new, prob_model_new = run_psf_reconstruction_step(
            setup=setup_current,
            best_params=best_params,
            noise_map_original=noise_map_original,
            cutout_size=cutout_size,
            supersampling_factor=supersampling_factor,
            mask_other_peaks=mask_other_peaks,
            mask_radius=mask_radius,
            rotation_mode=rotation_mode,
            negative_sigma_threshold=negative_sigma_threshold,
            verbose=verbose,
            save_dir=iter_save_dir,
        )

        iteration_results.append({
            "iteration": it,
            "gd_results": gd_results,
            "psf_input": psf_current.copy(),
            "psf_output": psf_new.copy(),
            "best_params": best_params,
        })

        psf_current = psf_new
        prob_model_current = prob_model_new

    return psf_current, prob_model_current, {"iterations": iteration_results}


__all__ = [
    "reconstruct_PSF",
    "reconstruct_psf_from_star_rotations",
    "generate_isolated_ps_images",
    "build_centered_cutouts_from_isolated",
    "build_centered_noise_cutouts",
    "run_psf_reconstruction_step",
    "run_psf_reconstruction_iterations",
    "_HAS_STARRED",
]
