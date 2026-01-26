"""
Main pipeline for lens modeling.

This module provides the high-level interface for running the complete
lens modeling pipeline: setup -> PSF reconstruction (optional) -> gradient descent -> posterior sampling.

Output folder structure:
    output_dir/
    ├── timing.json
    ├── pipeline_config.json
    ├── PSF/
    │   ├── starred_exports/
    │   └── psf_comparison.png
    ├── gradient_descent/
    │   ├── best_fit_params.json
    │   └── bestfit_residuals.png
    ├── sampling/
    │   ├── chains_*.png (NUTS only)
    │   └── nuts_samples.npz or nautilus_samples.npz
    └── final_outputs/
        ├── output.json (bic, posterior_summary, ray_tracing_summary)
        ├── posterior_samples.npz
        ├── bestloglikelihood_residuals.png
        ├── diagnostics.png
        └── plots/
            ├── corner_plot.png
            ├── ray_tracing_scatter.png
            └── marginalized_posteriors.png

Supported samplers:
- NUTS: No-U-Turn Sampler (Hamiltonian Monte Carlo)
- Nautilus: Nested sampling with evidence estimation
"""

from typing import Dict, Literal
import os
from time import time as now

import numpy as np
import jax
import jax.numpy as jnp

from .config import PipelineConfig
from .models import setup_lens_system
from .inference import (
    run_gradient_descent,
    run_nuts,
    run_nautilus,
    build_prior_and_loglike,
    get_nautilus_posterior,
)
from .psf import (
    run_psf_reconstruction_iterations,
    _HAS_STARRED,
)
from .output import (
    save_params_json,
    save_timing_json,
    save_output_json,
    samples_dict_to_array,
    array_row_to_params,
    compute_bic_from_samples,
    compute_posterior_summary,
    compute_ray_tracing_summary,
    find_best_loglike_sample,
    print_pipeline_header,
    print_pipeline_complete,
)
from .plotting import (
    plot_psf_comparison,
    plot_herculens_residual,
    plot_diagnostics,
    plot_corner,
    plot_ray_tracing_scatter,
    plot_marginalized_posteriors,
)


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _create_output_dirs(run_dir: str, use_psf: bool = False) -> Dict[str, str]:
    """Create the standard output folder structure.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping folder names to paths.
    """
    dirs = {
        "run_dir": run_dir,
        "psf": os.path.join(run_dir, "PSF"),
        "psf_starred_exports": os.path.join(run_dir, "PSF", "starred_exports"),
        "gradient_descent": os.path.join(run_dir, "gradient_descent"),
        "sampling": os.path.join(run_dir, "sampling"),
        "final_outputs": os.path.join(run_dir, "final_outputs"),
        "final_plots": os.path.join(run_dir, "final_outputs", "plots"),
    }

    # Create directories
    for name, path in dirs.items():
        if name == "psf" or name == "psf_starred_exports":
            if use_psf:
                _ensure_dir(path)
        else:
            _ensure_dir(path)

    return dirs


def _create_initial_positions(best_params: Dict, num_chains: int) -> Dict:
    """Create initial positions for NUTS from MAP parameters."""
    initial_positions = {}
    rng = np.random.default_rng(42)

    for name, value in best_params.items():
        value = np.asarray(value)
        if value.ndim == 0:
            scale = max(abs(float(value)) * 0.01, 0.01)
            perturbed = float(value) + rng.normal(0, scale, size=num_chains)
            initial_positions[name] = jnp.asarray(perturbed)
        else:
            scale = np.maximum(np.abs(value) * 0.01, 0.01)
            perturbed = value + rng.normal(0, scale, size=(num_chains,) + value.shape)
            initial_positions[name] = jnp.asarray(perturbed)

    return initial_positions


def run_pipeline(
    config: PipelineConfig,
    run_sampling: bool = True,
    sampler: Literal["nuts", "nautilus"] = "nuts",
    verbose: bool = True,
) -> Dict:
    """
    Run the complete lens modeling pipeline.

    Steps:
    1. Set up lens system (data loading, model construction)
    2. Optionally run PSF reconstruction (using original noise map)
    3. Run gradient descent for MAP estimation
    4. Optionally run posterior sampling (NUTS or Nautilus)

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.
    run_sampling : bool
        Whether to run posterior sampling after gradient descent.
    sampler : str
        Sampler to use: "nuts" or "nautilus".
    verbose : bool
        Print progress information.

    Returns
    -------
    Dict
        Results dictionary with setup, MAP results, and samples.
    """
    timing = {"start": now()}

    if verbose:
        print_pipeline_header(config, run_sampling, sampler)

    # =========================================================================
    # Step 1: Setup
    # =========================================================================
    if verbose:
        print("\n[1/4] Setting up lens system...")

    timing["setup_start"] = now()

    setup = setup_lens_system(config)
    output_dir = setup["output_dir"]

    # Create run directory
    from datetime import datetime
    source_type = "CorrField" if config.use_corr_fields else "Shapelets"
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(output_dir, source_type, f"run_{run_name}")

    # Create folder structure
    use_psf_reconstruction = getattr(config, 'use_psf_reconstruction', False)
    dirs = _create_output_dirs(run_dir, use_psf=use_psf_reconstruction)

    # Save config to parent directory
    config.save(os.path.join(run_dir, "pipeline_config.json"))

    if verbose:
        print(f"  Output directory: {run_dir}")
        print(f"  Detected {len(setup['x0s'])} point sources")

    # Store original noise map (without boosting) for PSF reconstruction
    noise_map_original = np.asarray(setup.get("noise_map_original", setup["noise_map"]))

    # Track PSF reconstruction results
    psf_reconstruction_results = None
    timing["setup_end"] = now()

    # =========================================================================
    # Step 2: PSF Reconstruction (optional)
    # =========================================================================
    if use_psf_reconstruction:
        timing["psf_start"] = now()
        if not _HAS_STARRED:
            if verbose:
                print("\n[2/4] PSF reconstruction requested but STARRED not available, skipping...")
        else:
            if verbose:
                n_psf_iter = getattr(config, 'psf_reconstruction_iterations', 1)
                print(f"\n[2/4] Running PSF reconstruction ({n_psf_iter} iteration(s))...")

            psf_initial = np.asarray(setup["psf_kernel"])

            psf_new, prob_model_new, psf_results = run_psf_reconstruction_iterations(
                setup=setup,
                noise_map_original=noise_map_original,
                n_iterations=getattr(config, 'psf_reconstruction_iterations', 1),
                cutout_size=getattr(config, 'psf_cutout_size', 99),
                supersampling_factor=getattr(config, 'psf_supersampling_factor', 3),
                mask_other_peaks=getattr(config, 'psf_mask_other_peaks', True),
                mask_radius=getattr(config, 'psf_mask_radius', 8),
                rotation_mode=getattr(config, 'psf_rotation_mode', 'none'),
                negative_sigma_threshold=getattr(config, 'psf_negative_sigma_threshold', None),
                n_starts_initial=config.n_starts_initial,
                n_top_for_refinement=config.n_top_for_refinement,
                n_refinement_perturbations=config.n_refinement_perturbations,
                perturbation_scale=config.perturbation_scale,
                random_seed=config.random_seed,
                adam_steps_initial=config.adam_steps_initial,
                adam_steps_refinement=config.adam_steps_refinement,
                adam_lr=config.adam_lr,
                lbfgs_maxiter_initial=config.lbfgs_maxiter_initial,
                lbfgs_maxiter_refinement=config.lbfgs_maxiter_refinement,
                lbfgs_tol=config.lbfgs_tol,
                verbose=verbose,
                save_dir=dirs["psf_starred_exports"],
            )

            # Update setup with new PSF and prob_model
            setup["psf_kernel"] = psf_new
            setup["prob_model"] = prob_model_new
            psf_reconstruction_results = psf_results

            # Create PSF comparison plot
            plot_psf_comparison(
                psf_initial=psf_initial,
                psf_reconstructed=psf_new,
                save_path=os.path.join(dirs["psf"], "psf_comparison.png"),
            )

            if verbose:
                print("  PSF reconstruction complete")
        timing["psf_end"] = now()
    else:
        if verbose:
            print("\n[2/4] PSF reconstruction: Skipped (disabled in config)")

    # =========================================================================
    # Step 3: Gradient Descent
    # =========================================================================
    if verbose:
        print("\n[3/4] Running gradient descent optimization...")

    timing["gd_start"] = now()

    gd_results = run_gradient_descent(
        prob_model=setup["prob_model"],
        output_dir=dirs["gradient_descent"],
        n_starts_initial=config.n_starts_initial,
        n_top_for_refinement=config.n_top_for_refinement,
        n_refinement_perturbations=config.n_refinement_perturbations,
        perturbation_scale=config.perturbation_scale,
        random_seed=config.random_seed,
        adam_steps_initial=config.adam_steps_initial,
        adam_steps_refinement=config.adam_steps_refinement,
        adam_lr=config.adam_lr,
        adam_warmup_fraction=config.adam_warmup_fraction,
        adam_grad_clip=config.adam_grad_clip,
        adam_use_cosine_decay=config.adam_use_cosine_decay,
        lbfgs_maxiter_initial=config.lbfgs_maxiter_initial,
        lbfgs_maxiter_refinement=config.lbfgs_maxiter_refinement,
        lbfgs_tol=config.lbfgs_tol,
        verbose=verbose,
        max_retry_iterations=getattr(config, 'max_retry_iterations', 1),
        chi2_red_threshold=getattr(config, 'chi2_red_threshold', 2.0),
    )

    # Save MAP results to gradient_descent folder
    best_params = gd_results["best_params"]
    save_params_json(best_params, os.path.join(dirs["gradient_descent"], "best_fit_params.json"))

    # Create Herculens residual plot for best fit
    plot_herculens_residual(
        prob_model=setup["prob_model"],
        params=best_params,
        save_path=os.path.join(dirs["gradient_descent"], "bestfit_residuals.png"),
    )

    timing["gd_end"] = now()

    # =========================================================================
    # Step 4: Posterior Sampling (optional)
    # =========================================================================
    nuts_results = None
    nautilus_results = None
    samples_array = None
    param_names = None

    if run_sampling:
        timing["sampling_start"] = now()

        if verbose:
            sampler_name = "NUTS" if sampler == "nuts" else "Nautilus"
            print(f"\n[4/4] Running {sampler_name} posterior sampling...")

        if sampler == "nuts":
            # Prepare initial positions for NUTS
            num_chains = config.nuts_num_chains or jax.device_count()
            initial_positions = _create_initial_positions(best_params, num_chains)

            nuts_results = run_nuts(
                prob_model=setup["prob_model"],
                initial_positions=initial_positions,
                num_warmup=config.nuts_num_warmup,
                num_samples=config.nuts_num_samples,
                num_chains=num_chains,
                seed=config.random_seed,
                outdir=dirs["sampling"],
                verbose=verbose,
                target_accept_prob=config.nuts_target_accept,
                max_tree_depth=getattr(config, 'nuts_max_tree_depth', 10),
                chain_method=getattr(config, 'nuts_chain_method', 'parallel'),
                progress_bar=verbose,
            )

            # Save samples to sampling folder
            np.savez(
                os.path.join(dirs["sampling"], "nuts_samples.npz"),
                **nuts_results["samples"],
            )

            # Convert to array format
            samples_array, param_names = samples_dict_to_array(nuts_results["samples"])

        elif sampler == "nautilus":
            # Build prior and likelihood from prob_model
            prior, loglike = build_prior_and_loglike(
                setup["prob_model"],
                use_vmap=True,
                use_pmap=True,
            )

            # Checkpoint path
            checkpoint_path = os.path.join(dirs["sampling"], "nautilus_checkpoint.hdf5")

            # Run Nautilus
            sampler_obj, points, log_w, log_l = run_nautilus(
                prior=prior,
                loglike=loglike,
                n_live=config.nautilus_n_live,
                filepath=checkpoint_path,
                resume=False,
                verbose=verbose,
                vectorized=True,
                n_batch=getattr(config, 'nautilus_n_batch', None),
                pool=getattr(config, 'nautilus_pool', None),
            )

            # Convert to posterior dict
            nautilus_results = get_nautilus_posterior(
                sampler_obj, points, log_w, log_l,
                n_samples=getattr(config, 'nautilus_n_posterior_samples', None),
                random_seed=config.random_seed,
            )

            # Save samples to sampling folder
            np.savez(
                os.path.join(dirs["sampling"], "nautilus_samples.npz"),
                samples=nautilus_results["samples"],
                log_weights=nautilus_results["log_weights"],
                log_likelihood=nautilus_results["log_likelihood"],
            )

            samples_array = nautilus_results["samples"]
            param_names = nautilus_results["param_names"]

        else:
            raise ValueError(f"Unknown sampler: {sampler}. Use 'nuts' or 'nautilus'.")

        timing["sampling_end"] = now()

        # =====================================================================
        # Generate final outputs
        # =====================================================================
        if verbose:
            print("\nGenerating final outputs...")

        # Save posterior_samples.npz to final_outputs
        np.savez(
            os.path.join(dirs["final_outputs"], "posterior_samples.npz"),
            samples=samples_array,
            param_names=param_names,
        )

        # Compute and save combined output.json
        bic_data = compute_bic_from_samples(
            samples_array, param_names, setup["prob_model"],
            setup["img"], setup["noise_map"],
        )

        posterior_summary = compute_posterior_summary(samples_array, param_names)

        ray_tracing_summary = compute_ray_tracing_summary(
            samples_array, param_names, setup["prob_model"].lens_image.MassModel,
        )

        save_output_json(
            bic_data, posterior_summary, ray_tracing_summary,
            os.path.join(dirs["final_outputs"], "output.json"),
        )

        # Generate best log-likelihood residual plot
        best_ll_idx = find_best_loglike_sample(samples_array, param_names, setup["prob_model"])
        best_ll_params = array_row_to_params(samples_array[best_ll_idx], param_names)
        plot_herculens_residual(
            prob_model=setup["prob_model"],
            params=best_ll_params,
            save_path=os.path.join(dirs["final_outputs"], "bestloglikelihood_residuals.png"),
        )

        # Generate diagnostics plot
        plot_diagnostics(
            samples_array, param_names,
            sampler=sampler,
            nuts_results=nuts_results,
            nautilus_results=nautilus_results,
            save_path=os.path.join(dirs["final_outputs"], "diagnostics.png"),
        )

        # Generate plots in final_outputs/plots/
        # Corner plot
        plot_corner(
            samples_array, param_names,
            save_path=os.path.join(dirs["final_plots"], "corner_plot.png"),
        )

        # Ray tracing scatter plot
        plot_ray_tracing_scatter(
            samples_array, param_names,
            mass_model=setup["prob_model"].lens_image.MassModel,
            save_path=os.path.join(dirs["final_plots"], "ray_tracing_scatter.png"),
        )

        # Marginalized posteriors
        plot_marginalized_posteriors(
            samples_array, param_names,
            save_path=os.path.join(dirs["final_plots"], "marginalized_posteriors.png"),
        )

    # Save timing to parent directory
    timing["end"] = now()
    timing["total_seconds"] = timing["end"] - timing["start"]
    timing["total_minutes"] = timing["total_seconds"] / 60

    save_timing_json(timing, os.path.join(run_dir, "timing.json"))

    if verbose:
        print_pipeline_complete(run_dir, timing)

    return dict(
        config=config,
        setup=setup,
        dirs=dirs,
        psf_reconstruction_results=psf_reconstruction_results,
        gd_results=gd_results,
        nuts_results=nuts_results,
        nautilus_results=nautilus_results,
        run_dir=run_dir,
        timing=timing,
    )
