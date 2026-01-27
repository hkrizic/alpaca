"""
PSF reconstruction functions using STARRED.
"""

from __future__ import annotations

import os

import numpy as np

from alpaca.psf.isolation import (
    build_centered_cutouts_from_isolated,
    build_centered_noise_cutouts,
)
from alpaca.psf.utils import (
    _augment_rotations,
    _center_crop_or_pad,
    _ensure_dir,
    _find_nonfinite_fields,
    _save_fits,
)

# --- Optional STARRED imports -------------------------------------------------
try:
    from starred.plots import plot_function as pltf
    from starred.procedures.psf_routines import run_multi_steps_PSF_reconstruction
    from starred.psf.parameters import ParametersPSF
    from starred.psf.psf import PSF as StarredPSF

    _HAS_STARRED = True
except Exception:
    run_multi_steps_PSF_reconstruction = None
    StarredPSF = None
    ParametersPSF = None
    pltf = None
    _HAS_STARRED = False


# ----------------------------------------------------------------------------- #
# STARRED-backed PSF update
# ----------------------------------------------------------------------------- #
def _reconstruct_psf_starred(
    peaks_px: np.ndarray,
    noise_map: np.ndarray,
    isolated_images: list[np.ndarray],
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
    # model.export(outpath, kwargs_final, new_data, new_sigma2, format='fits') TODO
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
    star_image: np.ndarray | list[np.ndarray],
    noise_map: np.ndarray | list[np.ndarray],
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
    isolated_images: list[np.ndarray],
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
) -> dict[str, np.ndarray]:
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
    isolated_images: list[np.ndarray],
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
