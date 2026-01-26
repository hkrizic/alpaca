"""
Output and diagnostics module for ALPACA pipeline.

Provides functions for:
- Printing results summaries to terminal
- Computing diagnostics (BIC, chi2, ray tracing summary)
- Saving JSON files (params, output.json, timing, etc.)
- Converting between sample formats
"""

import os
import json
from typing import Dict, Optional, List

import numpy as np


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

def json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_params_json(params: Dict, path: str):
    """Save parameters to JSON file.

    Args:
        params: Parameter dictionary.
        path: Output file path.
    """
    serializable = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = float(v) if np.isscalar(v) else v
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved parameters to: {path}")


def save_timing_json(timing: Dict, path: str):
    """Save timing information to JSON file.

    Args:
        timing: Timing dictionary with start/end timestamps.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(timing, f, indent=2, default=json_serializer)
    print(f"Saved timing to: {path}")


def save_output_json(
    bic_data: Dict,
    posterior_summary: Dict,
    ray_tracing_summary: Dict,
    path: str,
):
    """Save combined output.json file.

    Args:
        bic_data: BIC computation results.
        posterior_summary: Posterior summary statistics.
        ray_tracing_summary: Ray tracing summary.
        path: Output file path.
    """
    output = {
        "bic": bic_data,
        "posterior_summary": posterior_summary,
        "ray_tracing_summary": ray_tracing_summary,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=json_serializer)
    print(f"Saved output.json to: {path}")


# =============================================================================
# Sample Format Conversion
# =============================================================================

def samples_dict_to_array(samples_dict: Dict) -> tuple:
    """Convert samples dictionary to (samples_array, param_names).

    Args:
        samples_dict: Dictionary mapping parameter names to sample arrays.

    Returns:
        Tuple of (samples_array, param_names) where samples_array has
        shape (n_samples, n_params).
    """
    param_names = []
    columns = []

    for name, vals in samples_dict.items():
        vals = np.asarray(vals)
        if vals.ndim == 1:
            param_names.append(name)
            columns.append(vals)
        else:
            # Flatten multi-dimensional parameters
            for i in range(vals.shape[1]):
                param_names.append(f"{name}_{i}")
                columns.append(vals[:, i])

    samples_array = np.column_stack(columns)
    return samples_array, param_names


def array_row_to_params(row: np.ndarray, param_names: List[str]) -> Dict:
    """Convert a single row of samples array to a params dict.

    Args:
        row: Single sample row of shape (n_params,).
        param_names: List of parameter names.

    Returns:
        Parameter dictionary with arrays reconstructed from flattened names.
    """
    params = {}
    for i, name in enumerate(param_names):
        if "_" in name and name.split("_")[-1].isdigit():
            # Part of array parameter
            base = "_".join(name.split("_")[:-1])
            idx = int(name.split("_")[-1])
            if base not in params:
                params[base] = []
            while len(params[base]) <= idx:
                params[base].append(None)
            params[base][idx] = float(row[i])
        else:
            params[name] = float(row[i])

    # Convert lists to arrays
    for key in params:
        if isinstance(params[key], list):
            params[key] = np.array(params[key])

    return params


# =============================================================================
# BIC and Summary Computation
# =============================================================================

def compute_bic(
    chi2: float,
    n_params: int,
    n_pixels: int,
) -> float:
    """
    Compute Bayesian Information Criterion.

    BIC = chi2 + n_params * ln(n_pixels)

    Parameters
    ----------
    chi2 : float
        Chi-squared value.
    n_params : int
        Number of free parameters.
    n_pixels : int
        Number of data points (pixels).

    Returns
    -------
    float
        BIC value. Lower is better.
    """
    return chi2 + n_params * np.log(n_pixels)


def compute_bic_from_samples(
    samples: np.ndarray,
    param_names: List[str],
    prob_model,
    img: np.ndarray,
    noise_map: np.ndarray,
) -> Dict:
    """Compute BIC from posterior samples.

    Args:
        samples: Posterior samples array of shape (n_samples, n_params).
        param_names: List of parameter names.
        prob_model: ProbModel instance.
        img: Observed image array.
        noise_map: Per-pixel noise standard deviation.

    Returns:
        Dictionary with bic, aic, chi2, chi2_reduced, n_data, n_params.
    """
    # Find best log-likelihood sample
    best_idx = find_best_loglike_sample(samples, param_names, prob_model)
    best_params = array_row_to_params(samples[best_idx], param_names)

    # Compute chi2 for best sample
    kwargs = prob_model.params2kwargs(best_params)
    model_img = prob_model.lens_image.model(**kwargs)
    resid = (img - model_img) / (noise_map + 1e-12)
    chi2 = float(np.sum(resid**2))

    n_data = img.size
    n_params_count = len(param_names)

    # BIC = chi2 + k * ln(n)
    bic = chi2 + n_params_count * np.log(n_data)

    # AIC = chi2 + 2k
    aic = chi2 + 2 * n_params_count

    chi2_red = chi2 / max(1, n_data - n_params_count)

    return {
        "bic": float(bic),
        "aic": float(aic),
        "chi2": float(chi2),
        "chi2_reduced": float(chi2_red),
        "n_data": int(n_data),
        "n_params": int(n_params_count),
    }


def compute_posterior_summary(samples: np.ndarray, param_names: List[str]) -> Dict:
    """Compute summary statistics for all parameters.

    Args:
        samples: Posterior samples array of shape (n_samples, n_params).
        param_names: List of parameter names.

    Returns:
        Dictionary mapping parameter names to summary statistics.
    """
    summary = {}
    for i, name in enumerate(param_names):
        vals = samples[:, i]
        summary[name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q16": float(np.percentile(vals, 16)),
            "q84": float(np.percentile(vals, 84)),
            "q2.5": float(np.percentile(vals, 2.5)),
            "q97.5": float(np.percentile(vals, 97.5)),
        }
    return summary


def compute_ray_tracing_summary(
    samples: np.ndarray,
    param_names: List[str],
    mass_model,
    n_samples: int = 100,
    random_seed: int = 42,
) -> Dict:
    """Compute ray tracing convergence summary.

    Args:
        samples: Posterior samples array of shape (n_total, n_params).
        param_names: List of parameter names.
        mass_model: Herculens MassModel instance.
        n_samples: Number of samples to evaluate.
        random_seed: Random seed for sample selection.

    Returns:
        Dictionary with ray tracing summary statistics.
    """
    # Find image position parameters
    n_images = sum(1 for p in param_names if p.startswith('x_image_'))
    if n_images == 0:
        return {"error": "No x_image_* parameters found"}

    def get_param(name, sample_idx):
        try:
            idx = param_names.index(name)
            return samples[sample_idx, idx]
        except ValueError:
            return None

    def get_kwargs_lens(sample_idx):
        theta_E = get_param('lens_theta_E', sample_idx)
        e1 = get_param('lens_e1', sample_idx)
        e2 = get_param('lens_e2', sample_idx)
        center_x = get_param('lens_center_x', sample_idx)
        center_y = get_param('lens_center_y', sample_idx)
        gamma = get_param('lens_gamma', sample_idx)
        gamma1 = get_param('lens_gamma1', sample_idx)
        gamma2 = get_param('lens_gamma2', sample_idx)

        if any(v is None for v in [theta_E, e1, e2, center_x, center_y, gamma, gamma1, gamma2]):
            return None

        return [
            dict(theta_E=float(theta_E), e1=float(e1), e2=float(e2),
                 center_x=float(center_x), center_y=float(center_y), gamma=float(gamma)),
            dict(gamma1=float(gamma1), gamma2=float(gamma2), ra_0=0.0, dec_0=0.0),
        ]

    # Sample subset for speed
    n_total = samples.shape[0]
    n_samples = min(n_samples, n_total)
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(n_total, n_samples, replace=False)

    spreads = []
    for idx in sample_indices:
        kwargs_lens = get_kwargs_lens(idx)
        if kwargs_lens is None:
            continue

        x_imgs = np.array([get_param(f'x_image_{i}', idx) for i in range(n_images)])
        y_imgs = np.array([get_param(f'y_image_{i}', idx) for i in range(n_images)])

        x_src, y_src = mass_model.ray_shooting(x_imgs, y_imgs, kwargs_lens)
        spread = np.sqrt(np.std(x_src)**2 + np.std(y_src)**2)
        spreads.append(spread)

    if len(spreads) == 0:
        return {"error": "Could not compute ray tracing"}

    spreads = np.array(spreads)
    mean_spread_mas = float(np.mean(spreads) * 1000)
    median_spread_mas = float(np.median(spreads) * 1000)
    max_spread_mas = float(np.max(spreads) * 1000)

    if median_spread_mas < 1:
        quality = "Excellent (< 1 mas)"
    elif median_spread_mas < 10:
        quality = "Good (< 10 mas)"
    elif median_spread_mas < 50:
        quality = "Acceptable (< 50 mas)"
    else:
        quality = "Warning: Large spread - model may have issues"

    return {
        "mean_spread_mas": mean_spread_mas,
        "median_spread_mas": median_spread_mas,
        "max_spread_mas": max_spread_mas,
        "n_samples_evaluated": len(spreads),
        "n_images": n_images,
        "quality": quality,
    }


def find_best_loglike_sample(
    samples: np.ndarray,
    param_names: List[str],
    prob_model,
    n_eval: int = 500,
    random_seed: int = 42,
) -> int:
    """Find the sample index with the best (highest) log-likelihood.

    Args:
        samples: Posterior samples array of shape (n_total, n_params).
        param_names: List of parameter names.
        prob_model: ProbModel instance.
        n_eval: Number of samples to evaluate.
        random_seed: Random seed for sample selection.

    Returns:
        Index of the best log-likelihood sample.
    """
    best_idx = 0
    best_ll = -np.inf

    # Evaluate a subset for efficiency
    n_total = samples.shape[0]
    n_eval = min(n_eval, n_total)
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(n_total, n_eval, replace=False)

    for idx in indices:
        params = array_row_to_params(samples[idx], param_names)
        try:
            kwargs = prob_model.params2kwargs(params)
            model_img = prob_model.lens_image.model(**kwargs)
            resid = (prob_model.img - model_img) / (prob_model.noise_map + 1e-12)
            ll = -0.5 * float(np.sum(resid**2))
            if ll > best_ll:
                best_ll = ll
                best_idx = idx
        except Exception:
            continue

    return best_idx


# =============================================================================
# Terminal Printing Functions
# =============================================================================

def print_gradient_descent_results(gd_results: Dict, verbose: bool = True):
    """
    Print gradient descent optimization results.

    Parameters
    ----------
    gd_results : Dict
        Results from run_gradient_descent.
    verbose : bool
        If True, print detailed output.
    """
    if not verbose or gd_results is None:
        return

    print("\n" + "=" * 60)
    print("GRADIENT DESCENT RESULTS")
    print("=" * 60)
    print(f"  Best loss: {gd_results.get('best_loss', 'N/A'):.4f}")
    print(f"  Reduced chi^2: {gd_results.get('chi2_red', 'N/A'):.4f}")

    best_params = gd_results.get("best_params", {})
    print("\n  Best-fit parameters (MAP):")

    key_params = [
        "lens_theta_E", "lens_gamma", "lens_e1", "lens_e2",
        "lens_gamma1", "lens_gamma2",
        "light_Re_L", "light_n_L",
        "light_Re_S", "light_n_S",
        "D_dt",
    ]

    for param in key_params:
        if param in best_params:
            val = best_params[param]
            if np.isscalar(val) or (hasattr(val, 'shape') and val.shape == ()):
                print(f"    {param}: {float(val):.4f}")


def print_sampling_results(
    sampling_results: Dict,
    sampler: str,
    config=None,
    verbose: bool = True,
):
    """
    Print posterior sampling results.

    Parameters
    ----------
    sampling_results : Dict
        Results from run_nuts or run_nautilus.
    sampler : str
        Sampler name ("nuts" or "nautilus").
    config : PipelineConfig, optional
        Configuration for additional context.
    verbose : bool
        If True, print detailed output.
    """
    if not verbose or sampling_results is None:
        return

    samples = sampling_results.get("samples", {})
    summary = sampling_results.get("summary", {})

    print("\n" + "=" * 60)
    print("POSTERIOR SAMPLING RESULTS")
    print("=" * 60)
    print(f"  Sampler: {sampler.upper()}")

    if sampler == "nautilus" and "log_evidence" in sampling_results:
        print(f"  log(Z) = {sampling_results['log_evidence']:.2f}")

    # Get number of samples
    first_key = next(iter(samples.keys()), None)
    if first_key is not None:
        n_samples = len(samples[first_key])
        print(f"  Number of samples: {n_samples}")

    print("\n  Parameter summaries (mean +/- std):")
    print("  " + "-" * 50)

    key_params_summary = [
        "lens_theta_E", "lens_gamma", "lens_e1", "lens_e2",
        "lens_gamma1", "lens_gamma2",
        "light_Re_L", "light_n_L", "light_e1_L", "light_e2_L",
        "light_Re_S", "light_n_S",
        "src_center_x", "src_center_y",
        "D_dt",
    ]

    if config is not None and getattr(config, 'use_rayshoot_systematic_error', False):
        key_params_summary.append("log_sigma_rayshoot_sys")

    for param in key_params_summary:
        if param in summary:
            s = summary[param]
            if isinstance(s.get("mean"), (int, float)):
                mean = s["mean"]
                std = s["std"]
                print(f"    {param:25s}: {mean:10.4f} +/- {std:.4f}")

    # Print D_dt with confidence interval
    if "D_dt" in summary:
        s = summary["D_dt"]
        print("\n  Time-delay distance:")
        print(f"    D_dt = {s['mean']:.1f} +/- {s['std']:.1f} Mpc")
        if "q16" in s and "q84" in s:
            print(f"    68% CI: [{s['q16']:.1f}, {s['q84']:.1f}] Mpc")


def print_bic(gd_results: Dict, setup: Dict, config=None, verbose: bool = True):
    """
    Print BIC computation results.

    Parameters
    ----------
    gd_results : Dict
        Results from gradient descent.
    setup : Dict
        Setup dictionary containing data.
    config : PipelineConfig, optional
        Configuration for context.
    verbose : bool
        If True, print output.
    """
    if not verbose or gd_results is None:
        return

    chi2 = gd_results.get("chi2", None)
    n_params = gd_results.get("n_params", 30)
    data = setup.get("img") if setup else None

    if chi2 is None or data is None:
        return

    n_pixels = data.size
    bic = compute_bic(chi2, n_params, n_pixels)

    print("\n" + "=" * 60)
    print("BIC (Bayesian Information Criterion)")
    print("=" * 60)
    print(f"  BIC = {bic:.2f}")
    print(f"  n_params = {n_params}")
    print(f"  n_pixels = {n_pixels}")

    if config is not None and getattr(config, 'use_source_shapelets', False):
        print(f"  shapelets_n_max = {config.shapelets_n_max}")

    print("\nNote: Lower BIC is better. To compare different n_max values,")
    print("run: python bic_selection.py")


def print_full_summary(
    results: Dict,
    sampler: str = "nuts",
    config=None,
    verbose: bool = True,
):
    """
    Print complete pipeline results summary.

    Parameters
    ----------
    results : Dict
        Full results dictionary from run_pipeline.
    sampler : str
        Sampler name used.
    config : PipelineConfig, optional
        Configuration for context.
    verbose : bool
        If True, print output.
    """
    if not verbose:
        return

    # Gradient descent results
    print_gradient_descent_results(results.get("gd_results"), verbose=True)

    # Sampling results
    sampling_results = results.get("nuts_results") or results.get("nautilus_results")
    print_sampling_results(sampling_results, sampler, config, verbose=True)

    # BIC
    print_bic(results.get("gd_results"), results.get("setup"), config, verbose=True)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    timing = results.get("timing", {})
    total_minutes = timing.get("total_minutes", results.get("time_total", 0) / 60)
    print(f"Total time: {total_minutes:.1f} minutes")
    print(f"Results saved to: {results.get('run_dir', 'N/A')}")
    print("=" * 70)


def print_pipeline_header(config, run_sampling: bool, sampler: str):
    """Print pipeline header with configuration summary.

    Args:
        config: PipelineConfig instance.
        run_sampling: Whether sampling will be run.
        sampler: Sampler name ("nuts" or "nautilus").
    """
    print("=" * 70)
    print("ALPACA: Automated Lens-modelling Pipeline for Accelerated TD Cosmography Analysis")
    print("=" * 70)
    print(f"Rung: {config.rung}, Code: {config.code_id}, Seed: {config.seed}")
    source_type = "Correlated Fields" if config.use_corr_fields else "Shapelets"
    print(f"Source model: {source_type}")
    if config.use_source_shapelets:
        print(f"  n_max = {config.shapelets_n_max}")
    if getattr(config, 'use_psf_reconstruction', False):
        print(f"PSF reconstruction: Enabled ({getattr(config, 'psf_reconstruction_iterations', 1)} iterations)")
    if run_sampling:
        print(f"Sampler: {sampler.upper()}")
    print("=" * 70)


def print_pipeline_complete(run_dir: str, timing: Dict):
    """Print pipeline completion message.

    Args:
        run_dir: Output directory path.
        timing: Timing dictionary.
    """
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {timing.get('total_minutes', 0):.1f} minutes")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)


__all__ = [
    # JSON helpers
    "json_serializer",
    "save_params_json",
    "save_timing_json",
    "save_output_json",
    # Sample format conversion
    "samples_dict_to_array",
    "array_row_to_params",
    # BIC and summary computation
    "compute_bic",
    "compute_bic_from_samples",
    "compute_posterior_summary",
    "compute_ray_tracing_summary",
    "find_best_loglike_sample",
    # Terminal printing
    "print_gradient_descent_results",
    "print_sampling_results",
    "print_bic",
    "print_full_summary",
    "print_pipeline_header",
    "print_pipeline_complete",
]
