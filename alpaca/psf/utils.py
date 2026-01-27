"""
Utility and helper functions for PSF reconstruction.
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
    os.makedirs(path, exist_ok=True)


def _save_fits(path: str, array: np.ndarray, overwrite: bool = True) -> None:
    arr = np.asarray(array)
    fits.writeto(path, arr.astype(np.float64), overwrite=overwrite)


def _rotation_k_list(rotation_mode: str | None) -> list[int]:
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
    best_params: dict,
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


def _find_nonfinite_fields(kwargs_dict: dict) -> list[str]:
    """Return a flat list of field names containing NaN/inf values."""
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


def _build_prob_model_with_psf(setup: dict, psf_kernel: np.ndarray):
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


def _make_iteration_output_dirs(outdir: str, iteration_index: int) -> dict[str, str]:
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


def _print_multistart_chi2(summary: dict, prefix: str = "[Multi-start]") -> None:
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
