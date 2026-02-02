"""
PSF reconstruction functions using STARRED.

This module wraps the STARRED PSF reconstruction library to produce both
low-resolution and high-resolution PSF kernels from isolated point source
cutouts.

author: hkrizic
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
    """
    Run STARRED PSF reconstruction and return both low- and high-resolution kernels.

    Builds centered cutouts and noise variance maps from the isolated point
    source images, optionally augments them with rotations, then runs
    STARRED's multi-step PSF reconstruction procedure (L-BFGS + AdaBelief).
    The resulting PSF kernels are normalised to unit sum.

    Parameters
    ----------
    peaks_px : np.ndarray
        Array of shape ``(N, 2)`` with ``(y, x)`` pixel coordinates for
        each point source.
    noise_map : np.ndarray
        Full-frame noise map (2D array).
    isolated_images : list[np.ndarray]
        List of isolated point source images (full frame), one per source.
    cutout_size : int, optional
        Size of square cutouts in pixels. Default is 99.
    supersampling_factor : int, optional
        STARRED super-sampling factor for the high-resolution PSF.
        Default is 3.
    mask_other_peaks : bool, optional
        Whether to mask other point source positions within each cutout.
        Default is True.
    mask_radius : int, optional
        Radius in pixels for circular masks applied to other point source
        positions. Default is 8.
    rotation_mode : str or None, optional
        Rotation augmentation mode. One of ``"none"``, ``"180"``, or
        ``"90"``/``"all"``. Default is ``"none"``.
    negative_sigma_threshold : float or None, optional
        If set, mask pixels with values more negative than this many sigma
        below zero. Default is None.
    verbose : bool, optional
        Whether to print progress information. Default is True.
    debug_save_dir : str or None, optional
        If provided, save STARRED input stamps, sigma2, and masks as FITS
        files to this directory. Default is None.
    export_dir : str or None, optional
        If provided, export STARRED reconstruction products to this
        directory. Default is None.
    global_mask_map : np.ndarray or None, optional
        Optional global mask in full-frame coordinates (1=keep, 0=mask).
        Default is None.
    noise_map_is_sigma : bool, optional
        If True, ``noise_map`` contains standard deviation values; if
        False, variance. Default is True.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:

        - ``"psf_low_res"`` : np.ndarray -- Low-resolution PSF kernel,
          normalised to unit sum.
        - ``"psf_high_res"`` : np.ndarray -- High-resolution
          (super-sampled) PSF kernel, normalised to unit sum.

    Raises
    ------
    ImportError
        If the ``starred-astro`` package is not installed.
    ValueError
        If no point source peaks are provided, PSF stamps contain no
        finite values, or the resulting kernel has a non-finite or zero
        sum.
    """
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
