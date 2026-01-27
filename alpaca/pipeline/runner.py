"""
alpaca.pipeline.runner

Main pipeline orchestration: run_pipeline, quick_pipeline, load_pipeline_results.
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
from alpaca.data.loader import tdlmc_paths
from alpaca.data.setup import setup_tdlmc_lens
from alpaca.plotting.diagnostics import (
    plot_multistart_summary,
    plot_psf_comparison,
)

# Plotting imports
from alpaca.plotting.model_plots import plot_model_summary_custom, plot_ray_tracing_check

# PSF imports
from alpaca.psf.iterations import run_psf_reconstruction_iterations as run_psf_iterations

# Sampler imports
from alpaca.sampler.gradient_descent import (
    compute_bic_from_results,
    load_multistart_summary,
    run_gradient_descent,
)

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
    _load_time_delay_data,
)
from alpaca.pipeline.stages.plotting import _generate_posterior_plots
from alpaca.pipeline.stages.sampling import _run_nautilus_sampling, _run_nuts_sampling


def run_pipeline(
    config: PipelineConfig | None = None,
    # Quick access to common parameters (override config if provided)
    base_dir: str | None = None,
    rung: int | None = None,
    code_id: int | None = None,
    seed: int | None = None,
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

    This function orchestrates:
    1. PSF reconstruction using STARRED (iterative)
    2. Multi-start MAP optimization
    3. Posterior sampling with the chosen sampler

    All results are saved in organized directories with comprehensive plots.

    Parameters
    ----------
    config : PipelineConfig, optional
        Full configuration object. If None, uses defaults.
    base_dir : str, optional
        Override base directory containing TDC data.
    rung, code_id, seed : int, optional
        Override TDLMC system identification.
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
        Enable/disable Correlated Fields for source (mutually exclusive with shapelets).
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
    if base_dir is not None:
        config.base_dir = base_dir
    if rung is not None:
        config.rung = rung
    if code_id is not None:
        config.code_id = code_id
    if seed is not None:
        config.seed = seed
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

    # Get paths
    data_dir, results_dir = tdlmc_paths(config.base_dir, config.rung, config.code_id, config.seed)
    output_dir = os.path.join(results_dir, config.output_subdir)
    dirs = _make_output_structure(output_dir)

    if verbose:
        print("=" * 70)
        print("LENS MODELING PIPELINE")
        print("=" * 70)
        print(f"Data: rung={config.rung}, code={config.code_id}, seed={config.seed}")
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
    # Phase 1: PSF Reconstruction
    # ========================================================================
    psf_kernel = None
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
            base=config.base_dir,
            rung=config.rung,
            code_id=config.code_id,
            seed=config.seed,
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

        psf_kernel = psf_result["psf_final"]

        # Save PSF results
        _save_fits(os.path.join(dirs["psf_fits"], "psf_initial.fits"), psf_result["psf_initial"])
        _save_fits(os.path.join(dirs["psf_fits"], "psf_final.fits"), psf_kernel)

        for i, iter_data in enumerate(psf_result["iterations"]):
            _save_fits(os.path.join(dirs["psf_fits"], f"psf_iteration_{i+1}.fits"),
                      iter_data["psf_updated"])

        # Plot PSF comparison
        if config.plotting_config.save_plots and config.plotting_config.plot_psf_comparison:
            psf_iterations = [it["psf_updated"] for it in psf_result["iterations"]]
            sigma_map = psf_result["iterations"][-1].get("psf_residual_sigma")

            plot_psf_comparison(
                psf_initial=psf_result["psf_initial"],
                psf_final=psf_kernel,
                psf_iterations=psf_iterations,
                sigma_map=sigma_map,
                save_path=os.path.join(dirs["psf_plots"], f"psf_comparison.{config.plotting_config.plot_format}"),
                dpi=config.plotting_config.dpi,
            )

        t_psf = time.perf_counter() - t_psf_start
        if verbose:
            print(f"PSF reconstruction completed in {t_psf:.2f}s")

        results["psf_result"] = psf_result
    else:
        if verbose:
            print("Skipping PSF reconstruction phase.")
    # ========================================================================
    # Setup lens model (with final PSF if available)
    # ========================================================================
    if verbose:
        print("\nSetting up lens model...")

    setup = setup_tdlmc_lens(
        base=config.base_dir,
        rung=config.rung,
        code_id=config.code_id,
        seed=config.seed,
        min_sep=config.ps_min_sep,
        use_source_shapelets=config.use_source_shapelets,
        shapelets_n_max=config.shapelets_n_max,
        boost_noise_around_ps=config.boost_noise_around_ps,
        boost_kwargs=config.boost_noise_kwargs,
        use_rayshoot_systematic_error=config.use_rayshoot_systematic_error,
        rayshoot_sys_error_min=config.rayshoot_sys_error_min,
        rayshoot_sys_error_max=config.rayshoot_sys_error_max,
        # Correlated Fields settings
        use_corr_fields=config.use_corr_fields,
        corr_field_num_pixels=config.corr_field_config.num_pixels,
        corr_field_mean_intensity=config.corr_field_config.mean_intensity,
        corr_field_offset_std=config.corr_field_config.offset_std,
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

    # If we have a reconstructed PSF, rebuild the model with it
    if psf_kernel is not None:
        if verbose:
            print("Rebuilding model with reconstructed PSF...")
        prob_model, lens_image_new = _build_prob_model_with_psf_and_lens_image(
            setup, psf_kernel
        )
        setup["prob_model"] = prob_model
        setup["lens_image"] = lens_image_new
        setup["psf_kernel"] = psf_kernel
        if verbose:
            print(f"Model rebuilt with new PSF (shape: {psf_kernel.shape})")
    else:
        prob_model = setup["prob_model"]

    results["setup"] = setup

    img = setup["img"]
    noise_map = setup["noise_map"]
    lens_image = setup["lens_image"]
    number_of_params = prob_model.num_parameters
    if verbose:
        print(f"Model has {number_of_params} parameters")

    measured_delays = None
    delay_errors = None
    time_delay_labels = None

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
        measured_delays = None
        delay_errors = None
        if ms_config.use_time_delays:
            try:
                (measured_delays, delay_errors, time_delay_labels,
                 used_fallback, truth_positions) = _load_time_delay_data(
                    base_dir=config.base_dir,
                    rung=config.rung,
                    code_id=config.code_id,
                    seed=config.seed,
                    x0s=setup["x0s"],
                    y0s=setup["y0s"],
                    verbose=verbose,
                    fallback_to_truth=config.ps_fallback_to_truth,
                )
                # If fallback was used, rebuild setup with truth positions
                if used_fallback and truth_positions is not None:
                    if verbose:
                        print("Rebuilding model with truth positions...")
                    x_truth, y_truth = truth_positions
                    setup = setup_tdlmc_lens(
                        base=config.base_dir,
                        rung=config.rung,
                        code_id=config.code_id,
                        seed=config.seed,
                        n_ps_detect=len(x_truth),
                        min_sep=0.01,  # Very small to accept close images
                        use_source_shapelets=config.use_source_shapelets,
                        shapelets_n_max=config.shapelets_n_max,
                        boost_noise_around_ps=config.boost_noise_around_ps,
                        boost_kwargs=config.boost_noise_kwargs,
                        # Correlated Fields settings
                        use_corr_fields=config.use_corr_fields,
                        corr_field_num_pixels=config.corr_field_config.num_pixels,
                        corr_field_mean_intensity=config.corr_field_config.mean_intensity,
                        corr_field_offset_std=config.corr_field_config.offset_std,
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
                        output_dir=None,  # Mask already saved in first setup call
                    )
                    # Override with truth positions
                    setup["x0s"] = x_truth
                    setup["y0s"] = y_truth
                    prob_model = setup["prob_model"]
                    prob_model.x0s = x_truth
                    prob_model.y0s = y_truth
                    results["used_truth_positions"] = True
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"{exc}. Set gradient_descent_config.use_time_delays=False to skip."
                ) from exc
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
            max_retry_iterations=ms_config.max_retry_iterations,
            chi2_red_threshold=ms_config.chi2_red_threshold,
        )

        best_params = multistart_summary["best_params_json"]

        t_ms = time.perf_counter() - t_ms_start
        if verbose:
            best_chi2 = multistart_summary.get("best_chi2_red",
                        multistart_summary.get("chi2_reds", [0])[multistart_summary.get("best_run", 0)])
            print(f"Multi-start completed in {t_ms:.2f}s")
            print(f"Best chi2_red = {best_chi2:.4f}")

        # Plot multi-start summary
        if config.plotting_config.save_plots:
            plot_multistart_summary(
                multistart_summary,
                save_path=os.path.join(dirs["multistart_plots"],
                                       f"multistart_summary.{config.plotting_config.plot_format}"),
                dpi=config.plotting_config.dpi,
            )

        # Plot model at best-fit
        if config.plotting_config.save_plots and config.plotting_config.plot_model_summary:
            kwargs = prob_model.params2kwargs(best_params)
            model_img = np.asarray(lens_image.model(**kwargs))

            plot_model_summary_custom(
                img=img,
                model_img=model_img,
                noise_map=noise_map,
                save_path=os.path.join(dirs["multistart_plots"],
                                       f"model_summary.{config.plotting_config.plot_format}"),
                title="Best-fit Model (Multi-start)",
                dpi=config.plotting_config.dpi,
            )

            # Use herculens plotter if available
            if _HAS_PLOTTER:
                plotter = Plotter(flux_vmin=1e-3, flux_vmax=10, res_vmax=4)
                plotter.set_data(img)
                fig = plotter.model_summary(
                    lens_image, kwargs,
                    show_source=True,
                    kwargs_grid_source=dict(pixel_scale_factor=1),
                )
                fig.suptitle("Best-fit Model (Multi-start)", fontsize=14)
                fig.savefig(
                    os.path.join(dirs["multistart_plots"],
                                f"model_summary_herculens.{config.plotting_config.plot_format}"),
                    dpi=config.plotting_config.dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)

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
        except FileNotFoundError:
            # Try standard results dir
            try:
                multistart_summary = load_multistart_summary(results_dir, verbose=verbose)
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
            if measured_delays is None or delay_errors is None or time_delay_labels is None:
                try:
                    (measured_delays, delay_errors, time_delay_labels,
                     used_fallback, truth_positions) = _load_time_delay_data(
                        base_dir=config.base_dir,
                        rung=config.rung,
                        code_id=config.code_id,
                        seed=config.seed,
                        x0s=setup["x0s"],
                        y0s=setup["y0s"],
                        verbose=verbose,
                        fallback_to_truth=config.ps_fallback_to_truth,
                    )
                    # If fallback was used, rebuild setup with truth positions
                    if used_fallback and truth_positions is not None:
                        if verbose:
                            print("Rebuilding model with truth positions...")
                        x_truth, y_truth = truth_positions
                        setup = setup_tdlmc_lens(
                            base=config.base_dir,
                            rung=config.rung,
                            code_id=config.code_id,
                            seed=config.seed,
                            n_ps_detect=len(x_truth),
                            min_sep=0.01,
                            use_source_shapelets=config.use_source_shapelets,
                            shapelets_n_max=config.shapelets_n_max,
                            boost_noise_around_ps=config.boost_noise_around_ps,
                            boost_kwargs=config.boost_noise_kwargs,
                            # Correlated Fields settings
                            use_corr_fields=config.use_corr_fields,
                            corr_field_num_pixels=config.corr_field_config.num_pixels,
                            corr_field_mean_intensity=config.corr_field_config.mean_intensity,
                            corr_field_offset_std=config.corr_field_config.offset_std,
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
                            output_dir=None,  # Mask already saved in first setup call
                        )
                        setup["x0s"] = x_truth
                        setup["y0s"] = y_truth
                        prob_model = setup["prob_model"]
                        prob_model.x0s = x_truth
                        prob_model.y0s = y_truth
                        lens_image = setup["lens_image"]
                        results["used_truth_positions"] = True
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"{exc}. Set sampler_config.use_time_delays=False to skip."
                    ) from exc
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


def quick_pipeline(
    rung: int,
    code_id: int,
    seed: int,
    base_dir: str = ".",
    sampler: Literal["nuts", "nautilus", "default"] = "default",
    n_psf_iterations: int = 3,
    n_multistart: int = 20,
    use_shapelets: bool = True,
    shapelets_n_max: int = 6,
    verbose: bool = True,
) -> dict:
    """
    Quick pipeline execution with sensible defaults.

    Parameters
    ----------
    rung, code_id, seed : int
        TDLMC system identification.
    base_dir : str
        Base directory containing TDC data.
    sampler : {"nuts", "nautilus"}
        Sampler choice.
    n_psf_iterations : int
        Number of PSF reconstruction iterations.
    n_multistart : int
        Number of multi-start optimization runs.
    use_shapelets : bool
        Enable source shapelets.
    shapelets_n_max : int
        Maximum shapelet order.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Pipeline results.
    """
    return run_pipeline(
        base_dir=base_dir,
        rung=rung,
        code_id=code_id,
        seed=seed,
        sampler=sampler,
        n_psf_iterations=n_psf_iterations,
        n_multistart=n_multistart,
        use_source_shapelets=use_shapelets,
        shapelets_n_max=shapelets_n_max,
        verbose=verbose,
    )


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
