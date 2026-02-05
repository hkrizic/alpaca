"""
alpaca.pipeline.runner

Main pipeline orchestration: run_pipeline, load_pipeline_results.

The pipeline accepts generic image data (img, psf_kernel, noise_map) and
optional time-delay information.  It does NOT depend on any specific
external data layout.

author: hkrizic
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Configuration imports
from alpaca.config import (
    PipelineConfig,
)

# Data imports
from alpaca.data.setup import setup_lens
from alpaca.plotting.diagnostics import plot_psf_comparison

# Plotting imports
from alpaca.plotting.model_plots import (
    plot_model_summary_custom,
    plot_ray_tracing_check,
)

# PSF imports
from alpaca.psf.iterations import run_psf_reconstruction_iterations as run_psf_iterations

# Sampler imports
from alpaca.sampler.gradient_descent import (
    load_multistart_summary,
    run_gradient_descent,
)
from alpaca.utils.bic import compute_bic_from_results

# Optional imports for plotting
try:
    from herculens.Analysis.plot import Plotter
    _HAS_PLOTTER = True
except ImportError:
    _HAS_PLOTTER = False
    Plotter = None

# Sibling module imports
from alpaca.pipeline.io import _make_output_structure, _save_fits, _save_json
from alpaca.pipeline.setup import (
    _build_prob_model_with_psf_and_lens_image,
    _match_point_sources,
)
from alpaca.pipeline.stages.plotting import _generate_posterior_plots
from alpaca.pipeline.stages.sampling import _run_nautilus_sampling, _run_nuts_sampling


def run_pipeline(
    config: PipelineConfig | None = None,
    *,
    img: np.ndarray,
    psf_kernel: np.ndarray,
    noise_map: np.ndarray,
    image_positions: tuple[np.ndarray, np.ndarray] | None = None,
    measured_delays: np.ndarray | None = None,
    delay_errors: np.ndarray | None = None,
    time_delay_labels: list[str] | None = None,
    sampler: Literal["nuts", "nautilus", "default"] | None = None,
    n_psf_iterations: int | None = None,
    n_multistart: int | None = None,
    run_psf_reconstruction: bool | None = None,
    run_multistart_opt: bool | None = None,
    run_sampling: bool | None = None,
    use_source_shapelets: bool | None = None,
    shapelets_n_max: int | None = None,
    use_corr_fields: bool | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the complete lens modeling pipeline.

    This function orchestrates three phases:

    1. PSF reconstruction using STARRED (iterative)
    2. Multi-start MAP optimization
    3. Posterior sampling with the chosen sampler

    All results are saved in organized directories with comprehensive plots.

    Parameters
    ----------
    config : PipelineConfig, optional
        Full configuration object.  If *None*, a default is created.
    img : np.ndarray
        2-D observed lens image.
    psf_kernel : np.ndarray
        2-D initial PSF kernel.
    noise_map : np.ndarray
        2-D noise map (same shape as *img*).
    image_positions : tuple of (np.ndarray, np.ndarray), optional
        Approximate reference positions ``(x_array, y_array)`` in arcsec,
        used to label auto-detected images (A, B, C, D).  The pipeline
        always auto-detects the actual positions from *img*; these
        approximate positions only control the ordering so that time
        delays are assigned to the correct image pairs.
        If *None*, auto-detected order (brightest first) is used.
    measured_delays : np.ndarray, optional
        Measured time delays (relative to the first image).
    delay_errors : np.ndarray, optional
        1-sigma errors on *measured_delays*.
    time_delay_labels : list of str, optional
        Labels for each image (e.g. ``["A", "B", "C", "D"]``).
    sampler : {"nuts", "nautilus", "default"}, optional
        Override sampler choice.
    n_psf_iterations : int, optional
        Override number of PSF reconstruction iterations.
    n_multistart : int, optional
        Override number of multi-start runs.
    run_psf_reconstruction, run_multistart_opt, run_sampling : bool, optional
        Override phase toggles.
    use_source_shapelets : bool, optional
        Enable/disable source shapelets.
    shapelets_n_max : int, optional
        Maximum shapelet order.
    use_corr_fields : bool, optional
        Enable/disable Correlated Fields for source
        (mutually exclusive with shapelets).
    verbose : bool
        Print progress information.

    Returns
    -------
    dict
        Pipeline results containing:

        - config: Final configuration used
        - setup: Lens model setup dictionary
        - psf_result: PSF reconstruction results (if run)
        - multistart_summary: Multi-start optimization summary
        - posterior: Standardized posterior container
        - output_dirs: Dictionary of output directory paths
    """
    # Initialize config
    if config is None:
        config = PipelineConfig()

    # Apply overrides
    if sampler is not None:
        config.sampler_config.sampler = sampler
    if n_psf_iterations is not None:
        config.psf_config.n_iterations = n_psf_iterations
    if n_multistart is not None:
        config.gradient_descent_config.n_starts_initial = n_multistart
    if run_psf_reconstruction is not None:
        config.run_psf_reconstruction = run_psf_reconstruction
    if run_multistart_opt is not None:
        config.run_multistart = run_multistart_opt
    if run_sampling is not None:
        config.run_sampling = run_sampling
    if use_source_shapelets is not None:
        config.use_source_shapelets = use_source_shapelets
    if shapelets_n_max is not None:
        config.shapelets_n_max = shapelets_n_max
    if use_corr_fields is not None:
        config.use_corr_fields = use_corr_fields

    # Validate mutual exclusivity of source models
    if config.use_source_shapelets and config.use_corr_fields:
        raise ValueError(
            "Cannot use both Shapelets and Correlated Fields. "
            "Set only one of use_source_shapelets or use_corr_fields to True."
        )

    # Setup timing
    t_start = time.perf_counter()

    # Output directory
    output_dir = config.output_dir
    dirs = _make_output_structure(output_dir)

    if verbose:
        print("=" * 70)
        print("LENS MODELING PIPELINE")
        print("=" * 70)
        print(f"Output: {output_dir}")
        if config.use_corr_fields:
            print(f"Source model: Correlated Fields (pixels={config.corr_field_config.num_pixels})")
        elif config.use_source_shapelets:
            print(f"Source model: Shapelets (n_max={config.shapelets_n_max})")
        else:
            print("Source model: Sersic only")
        print(f"Sampler: {config.sampler_config.sampler}")
        print("=" * 70)

    # Save configuration
    config_dict = asdict(config)
    _save_json(os.path.join(dirs["root"], "pipeline_config.json"), config_dict)

    results = {
        "config": config,
        "output_dirs": dirs,
    }

    # ========================================================================
    # Build initial setup (before PSF reconstruction)
    # ========================================================================
    if verbose:
        print("\nSetting up lens model...")

    setup = setup_lens(
        img=img,
        psf_kernel=psf_kernel,
        noise_map=noise_map,
        use_source_shapelets=config.use_source_shapelets,
        shapelets_n_max=config.shapelets_n_max,
        boost_noise_around_ps=config.boost_noise_around_ps,
        boost_kwargs=config.boost_noise_kwargs,
        min_sep=config.ps_min_sep,
        use_rayshoot_systematic_error=config.use_rayshoot_systematic_error,
        rayshoot_sys_error_min=config.rayshoot_sys_error_min,
        rayshoot_sys_error_max=config.rayshoot_sys_error_max,
        use_image_pos_offset=config.use_image_pos_offset,
        image_pos_offset_sigma=config.image_pos_offset_sigma,
        use_corr_fields=config.use_corr_fields,
        corr_field_num_pixels=config.corr_field_config.num_pixels,
        corr_field_mean_intensity=config.corr_field_config.mean_intensity,
        corr_field_loglogavgslope=config.corr_field_config.loglogavgslope,
        corr_field_fluctuations=config.corr_field_config.fluctuations,
        corr_field_flexibility=config.corr_field_config.flexibility,
        corr_field_asperity=config.corr_field_config.asperity,
        corr_field_cropped_border_size=config.corr_field_config.cropped_border_size,
        corr_field_exponentiate=config.corr_field_config.exponentiate,
        corr_field_interpolation=config.corr_field_config.interpolation_type,
        arc_mask_inner_radius=config.corr_field_config.arc_mask_inner_radius,
        arc_mask_outer_radius=config.corr_field_config.arc_mask_outer_radius,
        custom_arc_mask_path=config.corr_field_config.custom_arc_mask_path,
        output_dir=dirs["root"],
    )

    # If the caller supplied approximate image positions, use them to
    # reorder the auto-detected positions so that labels (A, B, C, D)
    # match the correct detected images.  This ensures time delays are
    # assigned to the right image pairs.
    if image_positions is not None:
        # _match_point_sources returns perm where perm[i]=j means
        # detected image i matches reference image j.  We need the inverse:
        # for each reference label j, which detected image goes in slot j.
        fwd_perm = _match_point_sources(
            setup["x0s"], setup["y0s"],
            np.asarray(image_positions[0]), np.asarray(image_positions[1]),
        )
        reorder = np.argsort(fwd_perm)
        setup["x0s"] = setup["x0s"][reorder]
        setup["y0s"] = setup["y0s"][reorder]
        setup["peak_vals"] = setup["peak_vals"][reorder]
        setup["peaks_px"] = np.asarray(setup["peaks_px"])[reorder]

        # Rebuild the probabilistic model with the reordered positions
        prob_model_new, lens_image_new = _build_prob_model_with_psf_and_lens_image(
            setup, psf_kernel
        )
        setup["prob_model"] = prob_model_new
        setup["lens_image"] = lens_image_new

        if verbose:
            print("Matched auto-detected positions to reference labels:")
            ref_xs = np.asarray(image_positions[0])
            ref_ys = np.asarray(image_positions[1])
            labels = time_delay_labels or [str(i) for i in range(len(ref_xs))]
            for i, lab in enumerate(labels):
                dx = setup["x0s"][i] - ref_xs[i]
                dy = setup["y0s"][i] - ref_ys[i]
                sep = np.hypot(dx, dy)
                print(f"  {lab}: detected ({setup['x0s'][i]:+.4f}, {setup['y0s'][i]:+.4f})"
                      f"  ref ({ref_xs[i]:+.4f}, {ref_ys[i]:+.4f})"
                      f"  offset {sep:.4f}\"")

    # ========================================================================
    # Phase 1: PSF Reconstruction
    # ========================================================================
    psf_result = None
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: PSF RECONSTRUCTION")
        print("=" * 70)
    if config.run_psf_reconstruction:
        if verbose:
            print("Starting PSF reconstruction...")
        t_psf_start = time.perf_counter()

        psf_result = run_psf_iterations(
            setup=setup,
            output_dir=dirs["psf_fits"],
            n_iterations=config.psf_config.n_iterations,
            n_starts=config.psf_config.multistart_starts_per_iteration,
            starred_cutout_size=config.psf_config.starred_cutout_size,
            starred_supersampling_factor=config.psf_config.starred_supersampling_factor,
            starred_mask_other_peaks=config.psf_config.starred_mask_other_peaks,
            starred_mask_radius=config.psf_config.starred_mask_radius,
            starred_rotation_mode=config.psf_config.starred_rotation_mode,
            starred_negative_sigma_threshold=config.psf_config.starred_negative_sigma_threshold,
            run_multistart_bool=config.psf_config.run_multistart,
            parallelized_bool=config.psf_config.parallelized,
            verbose_starred=config.psf_config.verbose,
        )

        psf_final = psf_result["psf_final"]

        # Save PSF results
        _save_fits(os.path.join(dirs["psf_fits"], "psf_initial.fits"), psf_result["psf_initial"])
        _save_fits(os.path.join(dirs["psf_fits"], "psf_final.fits"), psf_final)

        for i, iter_data in enumerate(psf_result["iterations"]):
            _save_fits(os.path.join(dirs["psf_fits"], f"psf_iteration_{i+1}.fits"),
                      iter_data["psf_updated"])

        # Plot PSF comparison
        if config.plotting_config.save_plots and config.plotting_config.plot_psf_comparison:
            psf_iterations = [it["psf_updated"] for it in psf_result["iterations"]]
            sigma_map = psf_result["iterations"][-1].get("psf_residual_sigma")

            plot_psf_comparison(
                psf_initial=psf_result["psf_initial"],
                psf_final=psf_final,
                psf_iterations=psf_iterations,
                sigma_map=sigma_map,
                save_path=os.path.join(dirs["psf_plots"], f"psf_comparison.{config.plotting_config.plot_format}"),
                dpi=config.plotting_config.dpi,
            )

        t_psf = time.perf_counter() - t_psf_start
        if verbose:
            print(f"PSF reconstruction completed in {t_psf:.2f}s")

        # Rebuild model with the reconstructed PSF
        if verbose:
            print("Rebuilding model with reconstructed PSF...")
        prob_model, lens_image_new = _build_prob_model_with_psf_and_lens_image(
            setup, psf_final
        )
        setup["prob_model"] = prob_model
        setup["lens_image"] = lens_image_new
        setup["psf_kernel"] = psf_final
        if verbose:
            print(f"Model rebuilt with new PSF (shape: {psf_final.shape})")

        results["psf_result"] = psf_result
    else:
        if verbose:
            print("Skipping PSF reconstruction phase.")

    # ========================================================================
    # Finalize setup references
    # ========================================================================
    prob_model = setup["prob_model"]
    results["setup"] = setup

    lens_image = setup["lens_image"]
    number_of_params = prob_model.num_parameters
    if verbose:
        print(f"Model has {number_of_params} parameters")

    # ========================================================================
    # Phase 2: Multi-start Optimization
    # ========================================================================
    best_params = None
    multistart_summary = None

    if config.run_multistart:
        if verbose:
            print("\n" + "=" * 70)
            print("PHASE 2: MULTI-START OPTIMIZATION")
            print("=" * 70)

        t_ms_start = time.perf_counter()

        ms_config = config.gradient_descent_config

        # Validate time-delay data if the config requires it
        if ms_config.use_time_delays:
            if measured_delays is None or delay_errors is None:
                raise ValueError(
                    "use_time_delays is True but measured_delays or delay_errors "
                    "were not provided. Either pass them to run_pipeline() or set "
                    "gradient_descent_config.use_time_delays=False."
                )
            results["time_delays"] = {
                "measured_delays": measured_delays.tolist(),
                "delay_errors": delay_errors.tolist(),
                "image_labels": time_delay_labels,
            }

        # Run two-phase gradient descent optimization
        multistart_summary = run_gradient_descent(
            prob_model=prob_model,
            img=img,
            noise_map=noise_map,
            outdir=dirs["multistart"],
            n_starts_initial=ms_config.n_starts_initial,
            n_top_for_refinement=ms_config.n_top_for_refinement,
            n_refinement_perturbations=ms_config.n_refinement_perturbations,
            perturbation_scale=ms_config.perturbation_scale,
            random_seed=ms_config.random_seed,
            adam_steps_initial=ms_config.adam_steps_initial,
            adam_steps_refinement=ms_config.adam_steps_refinement,
            adam_lr=ms_config.adam_lr,
            adam_warmup_fraction=ms_config.adam_warmup_fraction,
            adam_grad_clip=ms_config.adam_grad_clip,
            adam_use_cosine_decay=ms_config.adam_use_cosine_decay,
            lbfgs_maxiter_initial=ms_config.lbfgs_maxiter_initial,
            lbfgs_maxiter_refinement=ms_config.lbfgs_maxiter_refinement,
            lbfgs_tol=ms_config.lbfgs_tol,
            verbose=ms_config.verbose,
            measured_delays=measured_delays,
            delay_errors=delay_errors,
            use_rayshoot_consistency=ms_config.use_rayshoot_consistency,
            rayshoot_consistency_sigma=ms_config.rayshoot_consistency_sigma,
            use_rayshoot_systematic_error=ms_config.use_rayshoot_systematic_error,
            use_image_pos_offset=ms_config.use_image_pos_offset,
            max_retry_iterations=ms_config.max_retry_iterations,
            chi2_red_threshold=ms_config.chi2_red_threshold,
        )

        best_params = multistart_summary["best_params_json"]

        t_ms = time.perf_counter() - t_ms_start
        if verbose:
            best_chi2 = multistart_summary["best_chi2_red"]
            best_phase = multistart_summary.get("best_from_phase", "N/A")
            print(f"Multi-start completed in {t_ms:.2f}s")
            print(f"Best chi2_red = {best_chi2:.4f} ({best_phase} phase)")

        # Save source pixels for correlated fields (required for later analysis
        # since the CF basis cannot be recreated)
        if config.use_corr_fields:
            source_pixels = prob_model.get_source_pixels_from_params(best_params)
            source_pixels_path = os.path.join(dirs["multistart"], "best_source_pixels.fits")
            _save_fits(source_pixels_path, np.asarray(source_pixels))
            if verbose:
                print(f"Saved source pixels to {source_pixels_path}")

        # Plot model at best-fit
        if config.plotting_config.save_plots and config.plotting_config.plot_model_summary:
            kwargs = prob_model.params2kwargs(best_params)
            model_img = np.asarray(lens_image.model(**kwargs))

            # Use herculens plotter if available
            if _HAS_PLOTTER:
                plotter = Plotter(flux_vmin=1e-3, flux_vmax=10, res_vmax=4)
                plotter.set_data(img)
                # For correlated fields, don't pass kwargs_grid_source so herculens
                # uses the adaptive source grid with the configured num_pixels.
                # For other models, use pixel_scale_factor=1 to match image resolution.
                kwargs_grid_src = None if config.use_corr_fields else dict(pixel_scale_factor=1)
                fig = plotter.model_summary(
                    lens_image, kwargs,
                    show_source=True,
                    kwargs_grid_source=kwargs_grid_src,
                )
                fig.suptitle("Best-fit Model (Multi-start)", fontsize=14)
                fig.savefig(
                    os.path.join(dirs["multistart_plots"],
                                f"model_summary_herculens.{config.plotting_config.plot_format}"),
                    dpi=config.plotting_config.dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)
            else:
                plot_model_summary_custom(
                    img=img,
                    model_img=model_img,
                    noise_map=noise_map,
                    save_path=os.path.join(dirs["multistart_plots"],
                                        f"model_summary.{config.plotting_config.plot_format}"),
                    title="Best-fit Model (Multi-start)",
                    dpi=config.plotting_config.dpi,
                )

        # Ray tracing check: verify 4 images converge to same source position
        if config.plotting_config.save_plots:
            try:
                ray_trace_result = plot_ray_tracing_check(
                    best_params=best_params,
                    mass_model=lens_image.MassModel,
                    save_path=os.path.join(dirs["multistart_plots"],
                                          f"ray_tracing_check.{config.plotting_config.plot_format}"),
                    dpi=config.plotting_config.dpi,
                    title="Ray Tracing Check (Best-fit)",
                )
                results["ray_tracing"] = ray_trace_result
                if verbose:
                    spread_mas = ray_trace_result["spread_mas"]
                    print(f"Ray tracing check: source plane spread = {spread_mas:.2f} mas")
                    if spread_mas < 1:
                        print("  -> Excellent convergence!")
                    elif spread_mas < 10:
                        print("  -> Acceptable convergence")
                    elif spread_mas > 50:
                        print("  -> Warning: Large spread, model may have issues")
            except Exception as e:
                if verbose:
                    print(f"Warning: Ray tracing check failed: {e}")

        results["multistart_summary"] = multistart_summary
        results["best_params"] = best_params

    else:
        # Try to load existing multistart results
        try:
            multistart_summary = load_multistart_summary(dirs["multistart"], verbose=verbose)
            best_params = multistart_summary["best_params_json"]
            results["multistart_summary"] = multistart_summary
            results["best_params"] = best_params
        except FileNotFoundError as err:
            if config.run_sampling:
                raise ValueError("Cannot run sampling without multi-start results. "
                                "Set run_multistart=True or provide existing results.") from err

    # ========================================================================
    # Phase 3: Posterior Sampling
    # ========================================================================
    posterior = None

    if config.run_sampling and best_params is not None:
        if verbose:
            print("\n" + "=" * 70)
            print(f"PHASE 3: POSTERIOR SAMPLING ({config.sampler_config.sampler.upper()})")
            print("=" * 70)

        t_samp_start = time.perf_counter()
        sampler_cfg = config.sampler_config
        # Propagate top-level likelihood settings to sampler config
        sampler_cfg.use_rayshoot_consistency = config.use_rayshoot_consistency
        sampler_cfg.rayshoot_consistency_sigma = config.rayshoot_consistency_sigma
        if sampler_cfg.sampler == "default":
            if number_of_params <= 50:
                print("Using NAUTILUS sampler based on parameter count")
                sampler_cfg.sampler = "nautilus"  # Default to Nautilus
            else:
                print("Using NUTS sampler based on parameter count")
                sampler_cfg.sampler = "nuts"  # Default to NUTS

        if sampler_cfg.use_time_delays:
            if measured_delays is None or delay_errors is None:
                raise ValueError(
                    "use_time_delays is True but measured_delays or delay_errors "
                    "were not provided. Either pass them to run_pipeline() or set "
                    "sampler_config.use_time_delays=False."
                )
            results["time_delays"] = {
                "measured_delays": measured_delays.tolist(),
                "delay_errors": delay_errors.tolist(),
                "image_labels": time_delay_labels,
            }

        if sampler_cfg.sampler == "nuts":
            posterior = _run_nuts_sampling(
                prob_model, best_params, lens_image, img, noise_map,
                sampler_cfg, dirs, verbose,
                measured_delays=measured_delays,
                delay_errors=delay_errors,
            )

        elif sampler_cfg.sampler == "nautilus":
            posterior = _run_nautilus_sampling(
                best_params, lens_image, img, noise_map,
                sampler_cfg, dirs, verbose,
                measured_delays=measured_delays,
                delay_errors=delay_errors,
                use_corr_fields=config.use_corr_fields,
            )

        else:
            raise ValueError(f"Unknown sampler: {sampler_cfg.sampler}")

        t_samp = time.perf_counter() - t_samp_start
        if verbose:
            print(f"Sampling completed in {t_samp:.2f}s ({t_samp/60:.2f} min)")

        results["posterior"] = posterior

        # Compute and save BIC
        if posterior is not None and "log_likelihood" in posterior:
            try:
                bic_info = compute_bic_from_results(results)
                results["bic"] = bic_info

                # Save BIC to file
                _save_json(os.path.join(dirs["posterior"], "bic.json"), bic_info)

                if verbose:
                    print(f"\nBIC = {bic_info['bic']:.2f}")
                    print(f"  n_params = {bic_info['n_params']}")
                    print(f"  n_pixels = {bic_info['n_pixels']}")
                    print(f"  log(L_max) = {bic_info['log_L_max']:.2f}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not compute BIC: {e}")

        # Generate posterior plots
        if config.plotting_config.save_plots and posterior is not None:
            _generate_posterior_plots(
                posterior=posterior,
                prob_model=prob_model,
                lens_image=lens_image,
                img=img,
                noise_map=noise_map,
                dirs=dirs,
                config=config.plotting_config,
            )

    # Save final timing
    t_total = time.perf_counter() - t_start
    timing = {
        "total_runtime_s": t_total,
        "total_runtime_min": t_total / 60,
    }
    _save_json(os.path.join(dirs["root"], "timing.json"), timing)

    if verbose:
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Total runtime: {t_total:.2f}s ({t_total/60:.2f} min)")
        if "bic" in results:
            print(f"BIC: {results['bic']['bic']:.2f} (n_params={results['bic']['n_params']})")
        if "ray_tracing" in results:
            print(f"Ray tracing spread: {results['ray_tracing']['spread_mas']:.2f} mas")
        print(f"Results saved to: {output_dir}")

    return results


def load_pipeline_results(output_dir: str) -> dict:
    """
    Load results from a previous pipeline run.

    Parameters
    ----------
    output_dir : str
        Path to pipeline output directory.

    Returns
    -------
    dict
        Loaded results including config, posterior samples, etc.
    """
    results = {}

    # Load config
    config_path = os.path.join(output_dir, "pipeline_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            results["config"] = json.load(f)

    # Load timing
    timing_path = os.path.join(output_dir, "timing.json")
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            results["timing"] = json.load(f)

    # Load multistart summary
    ms_path = os.path.join(output_dir, "multistart", "multi_start_summary.json")
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            results["multistart_summary"] = json.load(f)

    # Load posterior
    post_samples_path = os.path.join(output_dir, "posterior", "posterior_samples.npz")
    if os.path.exists(post_samples_path):
        data = np.load(post_samples_path, allow_pickle=True)
        results["posterior"] = {
            "samples": data["samples"],
            "param_names": list(data["param_names"]) if "param_names" in data.files else None,
        }

    post_summary_path = os.path.join(output_dir, "posterior", "posterior_summary.json")
    if os.path.exists(post_summary_path):
        with open(post_summary_path) as f:
            post_summary = json.load(f)
            if "posterior" not in results:
                results["posterior"] = {}
            results["posterior"]["summary"] = post_summary

    # Load final PSF
    psf_path = os.path.join(output_dir, "psf_reconstruction", "fits", "psf_final.fits")
    if os.path.exists(psf_path):
        results["psf_final"] = fits.getdata(psf_path)

    return results
