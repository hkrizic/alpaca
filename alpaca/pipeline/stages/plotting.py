"""
alpaca.pipeline.stages.plotting

Posterior plot generation stage.

author: hkrizic
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from alpaca.plotting.diagnostics import plot_chain_diagnostics
from alpaca.plotting.posterior_plots import plot_corner_posterior, plot_posterior_ray_tracing

# Optional imports for plotting
try:
    from herculens.Analysis.plot import Plotter
    _HAS_PLOTTER = True
except ImportError:
    _HAS_PLOTTER = False
    Plotter = None


def _generate_posterior_plots(
    posterior: dict,
    prob_model,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    dirs: dict[str, str],
    config,
) -> None:
    """
    Generate all posterior-related diagnostic and summary plots.

    Depending on the plotting configuration, this function produces:

    - A corner plot of selected posterior parameters.
    - Chain trace plots (when MCMC chains are available).
    - Model visualisations for random posterior draws and the
      best log-likelihood sample (requires ``herculens``).
    - A posterior ray-tracing consistency check if image-position
      parameters are present.

    Parameters
    ----------
    posterior : dict
        Standardised posterior container with keys ``"samples"``,
        ``"param_names"``, ``"log_likelihood"``, ``"engine"``, etc.
    prob_model : ProbModel or ProbModelCorrField
        Probabilistic model used for parameter transformations.
    lens_image : LensImage
        Herculens ``LensImage`` object for forward modelling.
    img : np.ndarray
        2-D observed lens image.
    noise_map : np.ndarray
        2-D noise map (same shape as *img*).
    dirs : dict of str
        Output directory mapping produced by ``_make_output_structure``.
    config : PlottingConfig
        Plotting configuration dataclass controlling which plots are
        generated, image format, DPI, etc.

    Returns
    -------
    None
    """
    # Corner plot
    if config.plot_corner:
        plot_corner_posterior(
            posterior=posterior,
            param_names=config.corner_params,
            max_params=config.max_corner_params,
            save_path=os.path.join(dirs["posterior_plots"], f"corner.{config.plot_format}"),
            title=f"Posterior ({posterior['engine']})",
            dpi=config.dpi,
        )

    # Chain diagnostics (if available)
    if config.plot_chains and posterior.get("_chain") is not None:
        plot_chain_diagnostics(
            chain=posterior["_chain"],
            param_names=posterior.get("_param_names_ordered", posterior.get("param_names", [])),
            save_path=os.path.join(dirs["posterior_plots"], f"chain_traces.{config.plot_format}"),
            dpi=config.dpi,
        )

    # Posterior draws
    if config.plot_posterior_draws > 0 and _HAS_PLOTTER:
        samples = posterior.get("samples")
        param_names = posterior.get("param_names", [])
        log_likelihood = posterior.get("log_likelihood")

        if samples is not None and len(param_names) > 0:
            plotter = Plotter(flux_vmin=1e-3, flux_vmax=10, res_vmax=4)
            plotter.set_data(img)

            def _reconstruct_params(sample_idx):
                """
                Convert a flat posterior sample to a reconstructed parameter dict.

                Flattened array keys (e.g. ``shapelets_amp_S_0``,
                ``shapelets_amp_S_1``) are reassembled into a single array
                entry ``shapelets_amp_S``.

                Parameters
                ----------
                sample_idx : int
                    Row index into the ``samples`` array.

                Returns
                -------
                dict
                    Mapping from parameter names to scalar or array values.
                """
                sample_dict = {name: samples[sample_idx, j] for j, name in enumerate(param_names)}

                # Reconstruct array parameters from flattened keys
                # e.g., shapelets_amp_S_0, shapelets_amp_S_1 -> shapelets_amp_S = [v0, v1]
                reconstructed = {}
                array_params = {}
                for key, val in sample_dict.items():
                    # Check if this is a flattened array parameter (ends with _N)
                    parts = key.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_name = parts[0]
                        idx_arr = int(parts[1])
                        if base_name not in array_params:
                            array_params[base_name] = {}
                        array_params[base_name][idx_arr] = val
                    else:
                        reconstructed[key] = val

                # Convert array_params dicts to arrays
                for base_name, idx_vals in array_params.items():
                    max_idx = max(idx_vals.keys())
                    arr = np.array([idx_vals[j] for j in range(max_idx + 1)])
                    reconstructed[base_name] = arr

                return reconstructed

            def _plot_sample(sample_idx, filename, title):
                """
                Plot a single posterior sample using the Herculens plotter.

                Parameters
                ----------
                sample_idx : int
                    Row index into the ``samples`` array.
                filename : str
                    Output filename (written inside ``dirs["posterior_draws"]``).
                title : str
                    Title string for the figure.

                Returns
                -------
                None
                """
                reconstructed = _reconstruct_params(sample_idx)
                kwargs = prob_model.params2kwargs(reconstructed)

                # For correlated fields, don't pass kwargs_grid_source so herculens
                # uses the adaptive source grid with the configured num_pixels.
                # For other models, use pixel_scale_factor=1 to match image resolution.
                use_corr = getattr(prob_model, 'use_corr_fields', False)
                kwargs_grid_src = None if use_corr else dict(pixel_scale_factor=1)
                fig = plotter.model_summary(
                    lens_image, kwargs,
                    show_source=True,
                    kwargs_grid_source=kwargs_grid_src,
                )
                fig.suptitle(title, fontsize=14)
                fig.savefig(
                    os.path.join(dirs["posterior_draws"], filename),
                    dpi=config.dpi,
                    bbox_inches="tight",
                )
                plt.close(fig)

            # 1. Plot best log-likelihood sample first
            if log_likelihood is not None:
                try:
                    best_ll_idx = int(np.argmax(log_likelihood))
                    best_ll_value = float(log_likelihood[best_ll_idx])
                    _plot_sample(
                        best_ll_idx,
                        f"posterior_best_logL.{config.plot_format}",
                        f"Best Log-Likelihood Sample (idx={best_ll_idx}, logL={best_ll_value:.2f})"
                    )
                except Exception as e:
                    warnings.warn(f"Failed to plot best log-likelihood sample: {e}", stacklevel=2)

            # 2. Plot random posterior draws
            rng = np.random.default_rng(42)
            n_draws = min(config.plot_posterior_draws, samples.shape[0])
            draw_indices = rng.choice(samples.shape[0], size=n_draws, replace=False)

            for i, idx in enumerate(draw_indices):
                try:
                    _plot_sample(
                        idx,
                        f"posterior_draw_{i}.{config.plot_format}",
                        f"Posterior Draw #{idx}"
                    )
                except Exception as e:
                    warnings.warn(f"Failed to plot posterior draw {i}: {e}", stacklevel=2)

    # Ray tracing check for posterior samples
    samples = posterior.get("samples")
    param_names = posterior.get("param_names", [])
    has_image_pos = any(p.startswith('x_image_') for p in param_names)

    if has_image_pos and samples is not None:
        try:
            ray_trace_result = plot_posterior_ray_tracing(
                samples=samples,
                param_names=param_names,
                mass_model=lens_image.MassModel,
                n_samples=100,
                save_path=os.path.join(dirs["posterior_plots"], f"ray_tracing_check.{config.plot_format}"),
                dpi=config.dpi,
            )
            # Save ray tracing summary
            with open(os.path.join(dirs["posterior"], "ray_tracing_summary.json"), "w") as f:
                json.dump({
                    "mean_spread_mas": float(ray_trace_result["mean_spread_mas"]),
                    "median_spread_mas": float(ray_trace_result["median_spread_mas"]),
                    "quality": ray_trace_result["quality"],
                }, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to generate posterior ray tracing plot: {e}", stacklevel=2)
