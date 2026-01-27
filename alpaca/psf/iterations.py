"""
Iterative PSF reconstruction pipeline.
"""

from __future__ import annotations

import os

import numpy as np

# Local project modules
from alpaca.psf.isolation import (
    build_centered_noise_cutouts,
    generate_isolated_ps_images,
    isolate_point_sources,
)
from alpaca.psf.reconstruction import _reconstruct_psf_starred_both
from alpaca.psf.utils import (
    _build_lens_center_mask_map,
    _build_model_images_from_best,
    _build_prob_model_with_psf,
    _center_crop_or_pad,
    _ensure_dir,
    _make_iteration_output_dirs,
    _print_multistart_chi2,
    _safe_normalized_residual,
    _save_fits,
)
from alpaca.sampler.gradient_descent import load_multistart_summary, run_gradient_descent


# ----------------------------------------------------------------------------- #
# Main one-iteration pipeline
# ----------------------------------------------------------------------------- #
def run_psf_reconstruction_iteration(
    setup: dict,
    output_dir: str,
    *,
    n_starts: int = 20,
    multistart_min_starts: int = 3,
    multistart_max_starts: int | None = None,
    multistart_chi2_red_threshold: float | None = 4.0,
    random_seed: int = 73,
    do_preopt: bool = True,
    adam_steps: int = 250,
    adam_lr: float = 3e-3,
    maxiter: int = 600,
    rel_eps: float = 0.01,
    starred_cutout_size: int = 99,
    starred_supersampling_factor: int = 3,
    starred_mask_other_peaks: bool = True,
    starred_mask_radius: int = 8,
    starred_rotation_mode: str | None = "none",
    starred_negative_sigma_threshold: float | None = None,
    save_products: bool = True,
    verbose_starred: bool = True,
    starred_global_mask_map: np.ndarray | None = None,
    noise_map_is_sigma: bool = True,
    run_multistart_bool: bool = True,
) -> dict:
    """Run a single PSF reconstruction iteration using STARRED.

    Parameters
    ----------
    setup : dict
        Pre-built setup dictionary (e.g. from ``setup_lens``).
    output_dir : str
        Root directory for all output products.
    """
    outdir = output_dir
    img = setup["img"]
    noise_map = setup["noise_map"]
    psf_initial = setup["psf_kernel"]
    prob_model = setup["prob_model"]
    peaks_px = np.asarray(setup["peaks_px"], float)
    n_point_sources = len(peaks_px)

    _ensure_dir(outdir)

    # Organize outputs for this iteration
    iter_dir = os.path.join(outdir, "Iteration_1")
    psf_dir = os.path.join(iter_dir, "psf")
    models_dir = os.path.join(iter_dir, "models")
    isolated_dir = os.path.join(iter_dir, "isolated_ps")
    starred_inputs_dir = os.path.join(iter_dir, "starred_inputs")
    starred_debug_dir = os.path.join(iter_dir, "starred_debug_plots")
    residuals_dir = os.path.join(iter_dir, "residuals")

    for p in (iter_dir, psf_dir, models_dir, isolated_dir, starred_inputs_dir, starred_debug_dir, residuals_dir):
        _ensure_dir(p)

    if run_multistart_bool:
        max_starts_eff = int(n_starts) if multistart_max_starts is None else int(multistart_max_starts)
        summary = run_gradient_descent(
            prob_model=prob_model,
            img=img,
            noise_map=noise_map,
            outdir=outdir,
            n_starts_initial=int(max_starts_eff),
            n_top_for_refinement=5,
            n_refinement_perturbations=5,
            perturbation_scale=0.1,
            random_seed=int(random_seed),
            adam_steps_initial=int(adam_steps),
            adam_steps_refinement=int(adam_steps),
            adam_lr=float(adam_lr),
            lbfgs_maxiter_initial=int(maxiter),
            lbfgs_maxiter_refinement=int(maxiter),
            lbfgs_tol=float(rel_eps),
            verbose=False,
        )
    else:
        summary = load_multistart_summary(outdir, verbose=True)

    _print_multistart_chi2(summary)

    best_params = summary.get("best_params_json", {})
    lens_mask_map = None
    if starred_mask_radius > 0:
        center_x = float(best_params.get("lens_center_x", 0.0))
        center_y = float(best_params.get("lens_center_y", 0.0))
        lens_mask_map = _build_lens_center_mask_map(
            setup["xgrid"],
            setup["ygrid"],
            center_x=center_x,
            center_y=center_y,
            radius_pix=float(starred_mask_radius),
        )
    global_mask_map = starred_global_mask_map
    if lens_mask_map is not None:
        global_mask_map = lens_mask_map if global_mask_map is None else (global_mask_map * lens_mask_map)

    model_with_ps = _build_model_images_from_best(prob_model, best_params, zero_point_sources=False)
    model_without_ps = _build_model_images_from_best(prob_model, best_params, zero_point_sources=True)
    ps_only_model = isolate_point_sources(model_with_ps, model_without_ps)
    starred_image_obs = np.asarray(img) - np.asarray(model_without_ps)

    if verbose_starred:
        print(f"[PSF Reconstruction] Generating isolated images (n_ps={n_point_sources})")

    isolated_ps_images = generate_isolated_ps_images(
        data_image=img,
        prob_model=prob_model,
        best_params=best_params,
        n_point_sources=n_point_sources,
    )

    # Reconstruct PSF and keep both STARRED outputs (low-res and high-res)
    target_shape = tuple(np.asarray(psf_initial).shape[:2])
    psf_updated_high_res: np.ndarray | None = None
    try:
        debug_inputs_dir = starred_inputs_dir if save_products else None
        debug_export_dir = starred_debug_dir if save_products else None
        starred_out = _reconstruct_psf_starred_both(
            peaks_px=peaks_px,
            noise_map=np.asarray(noise_map),
            isolated_images=isolated_ps_images,
            cutout_size=int(starred_cutout_size),
            supersampling_factor=int(starred_supersampling_factor),
            mask_other_peaks=bool(starred_mask_other_peaks),
            mask_radius=int(starred_mask_radius),
            rotation_mode=starred_rotation_mode,
            negative_sigma_threshold=starred_negative_sigma_threshold,
            verbose=bool(verbose_starred),
            debug_save_dir=debug_inputs_dir,
            export_dir=debug_export_dir,
            global_mask_map=global_mask_map,
            noise_map_is_sigma=bool(noise_map_is_sigma),
        )
        psf_updated_low_res = np.asarray(starred_out["psf_low_res"], float)
        psf_updated_high_res = np.asarray(starred_out["psf_high_res"], float)
        psf_updated_low_res = _center_crop_or_pad(psf_updated_low_res, target_shape=target_shape)
        psf_updated_low_res /= (psf_updated_low_res.sum() + 1e-12)
    except Exception as e:
        print(f"[run_psf_reconstruction_iteration] STARRED failed: {e}")
        psf_updated_low_res = np.asarray(psf_initial)

    # PSF residuals: compare updated PSF vs initial PSF on the same (native) grid
    psf_initial_native = _center_crop_or_pad(np.asarray(psf_initial), target_shape=target_shape)
    psf_updated_native = _center_crop_or_pad(np.asarray(psf_updated_low_res), target_shape=target_shape)
    psf_residual = psf_updated_native - psf_initial_native

    psf_residual_sigma: np.ndarray | None = None
    psf_residual_over_sigma: np.ndarray | None = None
    try:
        sigma2_maps_list = build_centered_noise_cutouts(
            noise_map=np.asarray(noise_map, float),
            peaks_px=np.asarray(peaks_px, float),
            cutout_size=int(starred_cutout_size),
            noise_map_is_sigma=bool(noise_map_is_sigma),
        )
        sigma2_mean = np.nanmean(np.asarray(sigma2_maps_list, float), axis=0)
        sigma_mean = np.sqrt(np.clip(sigma2_mean, 0.0, None))
        psf_residual_sigma = _center_crop_or_pad(sigma_mean, target_shape=target_shape)
        psf_residual_over_sigma = _safe_normalized_residual(
            data=psf_updated_native,
            model=psf_initial_native,
            noise_map=psf_residual_sigma,
            noise_map_is_sigma=True,
        )
    except Exception as e:
        print(f"[run_psf_reconstruction_iteration][DEBUG] Failed to compute PSF residual sigma map: {e}")

    if save_products:
        _save_fits(os.path.join(models_dir, "model_with_point_sources.fits"), model_with_ps)
        _save_fits(os.path.join(models_dir, "model_without_point_sources.fits"), model_without_ps)
        _save_fits(os.path.join(models_dir, "starred_modelwithps_minus_noPS.fits"), ps_only_model)
        _save_fits(os.path.join(models_dir, "starred_observedimage_minus_noPS.fits"), starred_image_obs)

        _save_fits(os.path.join(psf_dir, "psf_initial.fits"), np.asarray(psf_initial))
        _save_fits(os.path.join(psf_dir, "psf_updated_low_res.fits"), np.asarray(psf_updated_low_res))
        if psf_updated_high_res is not None:
            _save_fits(os.path.join(psf_dir, "psf_updated_high_res.fits"), np.asarray(psf_updated_high_res))

        _save_fits(os.path.join(residuals_dir, "psf_residual_updated_minus_initial.fits"), psf_residual)
        if psf_residual_sigma is not None:
            _save_fits(os.path.join(residuals_dir, "psf_residual_sigma.fits"), psf_residual_sigma)
        if psf_residual_over_sigma is not None:
            _save_fits(os.path.join(residuals_dir, "psf_residual_over_sigma.fits"), psf_residual_over_sigma)

        for i, iso_img in enumerate(isolated_ps_images):
            _save_fits(os.path.join(isolated_dir, f"isolated_ps_{i}.fits"), iso_img)

    return {
        "outdir": outdir,
        "iteration_dir": iter_dir,
        "summary": summary,
        "best_params": best_params,
        "model_with_ps": model_with_ps,
        "model_without_ps": model_without_ps,
        "ps_only_model": ps_only_model,
        "starred_image_obs": starred_image_obs,
        "psf_initial": psf_initial,
        "psf_updated": psf_updated_low_res,
        "psf_updated_high_res": psf_updated_high_res,
        "psf_residual": psf_residual,
        "psf_residual_sigma": psf_residual_sigma,
        "psf_residual_over_sigma": psf_residual_over_sigma,
        "isolated_ps_images": isolated_ps_images,
    }


def run_psf_reconstruction_iterations(
    setup: dict,
    output_dir: str,
    *,
    n_iterations: int = 3,
    n_starts: int = 20,
    multistart_min_starts: int = 3,
    multistart_max_starts: int | None = None,
    multistart_chi2_red_threshold: float | None = 4.0,
    random_seed: int = 73,
    do_preopt: bool = True,
    adam_steps: int = 250,
    adam_lr: float = 3e-3,
    maxiter: int = 600,
    rel_eps: float = 0.01,
    starred_cutout_size: int = 99,
    starred_supersampling_factor: int = 3,
    starred_mask_other_peaks: bool = True,
    starred_mask_radius: int = 8,
    starred_rotation_mode: str | None = "none",
    starred_negative_sigma_threshold: float | None = None,
    save_products: bool = True,
    verbose_starred: bool = True,
    starred_global_mask_map: np.ndarray | None = None,
    noise_map_is_sigma: bool = True,
    run_multistart_bool: bool = True,
    parallelized_bool: bool = True,
) -> dict:
    """
    Run an iterative PSF reconstruction loop.

    Parameters
    ----------
    setup : dict
        Pre-built setup dictionary (e.g. from ``setup_lens``).
    output_dir : str
        Root directory for all output products.

    Iteration k:
      1) Run multistart lens modelling using PSF_{k-1}
      2) Build isolated point-source images from the best-fit model
      3) Reconstruct PSF_k with STARRED from those isolated images

    The updated PSF from each iteration is fed into the next iteration's multistart.
    """
    n_iterations = int(n_iterations)
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")

    outdir = output_dir
    img = np.asarray(setup["img"])
    noise_map = np.asarray(setup["noise_map"])
    peaks_px = np.asarray(setup["peaks_px"], float)
    n_point_sources = len(peaks_px)

    psf_initial = np.asarray(setup["psf_kernel"], float)
    psf_initial = np.nan_to_num(psf_initial, nan=0.0, posinf=0.0, neginf=0.0)
    psf_initial /= (psf_initial.sum() + 1e-12)
    psf_current = psf_initial.copy()

    _ensure_dir(outdir)

    iteration_results: list[dict] = []

    for it in range(1, n_iterations + 1):
        dirs = _make_iteration_output_dirs(outdir, it)
        iter_dir = dirs["iteration_dir"]
        psf_dir = dirs["psf_dir"]
        models_dir = dirs["models_dir"]
        isolated_dir = dirs["isolated_dir"]
        starred_inputs_dir = dirs["starred_inputs_dir"]
        starred_debug_dir = dirs["starred_debug_dir"]
        residuals_dir = dirs["residuals_dir"]
        multistart_dir = dirs["multistart_dir"]

        prob_model = _build_prob_model_with_psf(setup, psf_current)

        if run_multistart_bool:
            max_starts_eff = int(n_starts) if multistart_max_starts is None else int(multistart_max_starts)
            summary = run_gradient_descent(
                prob_model=prob_model,
                img=img,
                noise_map=noise_map,
                outdir=multistart_dir,
                n_starts_initial=int(max_starts_eff),
                n_top_for_refinement=5,
                n_refinement_perturbations=5,
                perturbation_scale=0.1,
                random_seed=int(random_seed),
                adam_steps_initial=int(adam_steps),
                adam_steps_refinement=int(adam_steps),
                adam_lr=float(adam_lr),
                lbfgs_maxiter_initial=int(maxiter),
                lbfgs_maxiter_refinement=int(maxiter),
                lbfgs_tol=float(rel_eps),
                verbose=True,
            )
        else:
            summary = load_multistart_summary(multistart_dir, verbose=True)

        _print_multistart_chi2(summary, prefix=f"[Multi-start][Iteration {it}]")

        best_params = summary.get("best_params_json", {})
        lens_mask_map = None
        if starred_mask_radius > 0:
            center_x = float(best_params.get("lens_center_x", 0.0))
            center_y = float(best_params.get("lens_center_y", 0.0))
            lens_mask_map = _build_lens_center_mask_map(
                setup["xgrid"],
                setup["ygrid"],
                center_x=center_x,
                center_y=center_y,
                radius_pix=float(starred_mask_radius),
            )
        global_mask_map = starred_global_mask_map
        if lens_mask_map is not None:
            global_mask_map = lens_mask_map if global_mask_map is None else (global_mask_map * lens_mask_map)

        model_with_ps = _build_model_images_from_best(prob_model, best_params, zero_point_sources=False)
        model_without_ps = _build_model_images_from_best(prob_model, best_params, zero_point_sources=True)
        ps_only_model = isolate_point_sources(model_with_ps, model_without_ps)
        starred_image_obs = np.asarray(img) - np.asarray(model_without_ps)

        if verbose_starred:
            print(f"[PSF Reconstruction][Iteration {it}] Generating isolated images (n_ps={n_point_sources})")

        isolated_ps_images = generate_isolated_ps_images(
            data_image=img,
            prob_model=prob_model,
            best_params=best_params,
            n_point_sources=n_point_sources,
        )

        # STARRED PSF update
        target_shape = tuple(np.asarray(psf_current).shape[:2])
        psf_updated_high_res: np.ndarray | None = None
        try:
            debug_inputs_dir = starred_inputs_dir if save_products else None
            debug_export_dir = starred_debug_dir if save_products else None
            starred_out = _reconstruct_psf_starred_both(
                peaks_px=peaks_px,
                noise_map=np.asarray(noise_map),
                isolated_images=isolated_ps_images,
                cutout_size=int(starred_cutout_size),
                supersampling_factor=int(starred_supersampling_factor),
                mask_other_peaks=bool(starred_mask_other_peaks),
                mask_radius=int(starred_mask_radius),
                rotation_mode=starred_rotation_mode,
                negative_sigma_threshold=starred_negative_sigma_threshold,
                verbose=bool(verbose_starred),
                debug_save_dir=debug_inputs_dir,
                export_dir=debug_export_dir,
                global_mask_map=global_mask_map,
                noise_map_is_sigma=bool(noise_map_is_sigma),
            )
            psf_updated_low_res = np.asarray(starred_out["psf_low_res"], float)
            psf_updated_high_res = np.asarray(starred_out["psf_high_res"], float)
            psf_updated_low_res = _center_crop_or_pad(psf_updated_low_res, target_shape=target_shape)
            psf_updated_low_res = np.nan_to_num(psf_updated_low_res, nan=0.0, posinf=0.0, neginf=0.0)
            psf_updated_low_res /= (psf_updated_low_res.sum() + 1e-12)
        except Exception as e:
            print(f"[run_psf_reconstruction_iterations][Iteration {it}] STARRED failed: {e}")
            psf_updated_low_res = np.asarray(psf_current)

        # PSF residuals (updated - previous), optionally normalized by sigma
        psf_prev_native = _center_crop_or_pad(np.asarray(psf_current), target_shape=target_shape)
        psf_updated_native = _center_crop_or_pad(np.asarray(psf_updated_low_res), target_shape=target_shape)
        psf_residual = psf_updated_native - psf_prev_native

        psf_residual_sigma: np.ndarray | None = None
        psf_residual_over_sigma: np.ndarray | None = None
        try:
            sigma2_maps_list = build_centered_noise_cutouts(
                noise_map=np.asarray(noise_map, float),
                peaks_px=np.asarray(peaks_px, float),
                cutout_size=int(starred_cutout_size),
                noise_map_is_sigma=bool(noise_map_is_sigma),
            )
            sigma2_mean = np.nanmean(np.asarray(sigma2_maps_list, float), axis=0)
            sigma_mean = np.sqrt(np.clip(sigma2_mean, 0.0, None))
            psf_residual_sigma = _center_crop_or_pad(sigma_mean, target_shape=target_shape)
            psf_residual_over_sigma = _safe_normalized_residual(
                data=psf_updated_native,
                model=psf_prev_native,
                noise_map=psf_residual_sigma,
                noise_map_is_sigma=True,
            )
        except Exception as e:
            print(
                f"[run_psf_reconstruction_iterations][Iteration {it}][DEBUG] "
                f"Failed to compute PSF residual sigma map: {e}"
            )

        if save_products:
            _save_fits(os.path.join(models_dir, "model_with_point_sources.fits"), model_with_ps)
            _save_fits(os.path.join(models_dir, "model_without_point_sources.fits"), model_without_ps)
            _save_fits(os.path.join(models_dir, "starred_modelwithps_minus_noPS.fits"), ps_only_model)
            _save_fits(os.path.join(models_dir, "starred_observedimage_minus_noPS.fits"), starred_image_obs)

            _save_fits(os.path.join(psf_dir, "psf_input.fits"), np.asarray(psf_current))
            _save_fits(os.path.join(psf_dir, "psf_updated_low_res.fits"), np.asarray(psf_updated_low_res))
            if psf_updated_high_res is not None:
                _save_fits(os.path.join(psf_dir, "psf_updated_high_res.fits"), np.asarray(psf_updated_high_res))

            _save_fits(os.path.join(residuals_dir, "psf_residual_updated_minus_previous.fits"), psf_residual)
            if psf_residual_sigma is not None:
                _save_fits(os.path.join(residuals_dir, "psf_residual_sigma.fits"), psf_residual_sigma)
            if psf_residual_over_sigma is not None:
                _save_fits(os.path.join(residuals_dir, "psf_residual_over_sigma.fits"), psf_residual_over_sigma)

            for i, iso_img in enumerate(isolated_ps_images):
                _save_fits(os.path.join(isolated_dir, f"isolated_ps_{i}.fits"), iso_img)

        iteration_results.append(
            {
                "iteration_index": int(it),
                "iteration_dir": iter_dir,
                "multistart_dir": multistart_dir,
                "summary": summary,
                "best_params": best_params,
                "psf_input": np.asarray(psf_current),
                "psf_updated": np.asarray(psf_updated_low_res),
                "psf_updated_high_res": psf_updated_high_res,
                "psf_residual": psf_residual,
                "psf_residual_sigma": psf_residual_sigma,
                "psf_residual_over_sigma": psf_residual_over_sigma,
            }
        )

        psf_current = np.asarray(psf_updated_low_res)

    return {
        "outdir": outdir,
        "n_iterations": int(n_iterations),
        "psf_initial": psf_initial,
        "psf_final": psf_current,
        "iterations": iteration_results,
    }


# Alias for notebooks
run_iteration = run_psf_reconstruction_iteration
run_iterations = run_psf_reconstruction_iterations
