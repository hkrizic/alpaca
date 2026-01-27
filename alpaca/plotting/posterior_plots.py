"""
Posterior distribution plots for lens modeling.

Contains corner plots for Nautilus and PSO posteriors, ray-tracing
convergence analysis from posterior samples, and generic corner plots.

Functions:
    - nautilus_corner_plot: GetDist corner plot from Nautilus posteriors.
    - pso_corner_plot: GetDist corner plot from PSO swarm samples.
    - plot_posterior_ray_tracing: Source plane convergence analysis.
    - plot_corner_posterior: Generic corner plot from posterior samples.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from getdist import MCSamples, plots

# Optional imports for corner plotting backends
try:
    from getdist import MCSamples as _MCSamples
    from getdist import plots as gd_plots
    _HAS_GETDIST = True
except ImportError:
    _HAS_GETDIST = False
    _MCSamples = None
    gd_plots = None

try:
    import corner
    _HAS_CORNER = True
except ImportError:
    _HAS_CORNER = False


def nautilus_corner_plot(
    prior,
    loglike,
    checkpoint_path: str,
    number_live: int,
    out_dir: str,
    params_to_corner: list[str] | None = None,
    truth_values: dict[str, float] | None = None,
    tag: str = "corner_nautilus",
    rel_error: bool = False,
):
    """Create GetDist corner plot from Nautilus posterior with optional truth markers.

    Args:
        prior: Nautilus prior object from build_nautilus_prior_and_loglike.
        loglike: Log-likelihood function.
        checkpoint_path: Path to the Nautilus checkpoint file.
        number_live: Number of live points used in Nautilus run.
        out_dir: Output directory (required).
        params_to_corner: Parameter names to include. If None, uses defaults.
        truth_values: Optional dict mapping parameter names to true values
            for plotting truth markers and error annotations.
        tag: Filename prefix for the saved plot.
        rel_error: If True, show relative error instead of absolute in annotations.
    """
    from alpaca.sampler.nautilus import load_posterior_from_checkpoint

    if params_to_corner is None:
        params_to_corner = [
            "lens_theta_E",
            "lens_gamma",
            "lens_e1",
            "lens_e2",
            "light_Re_L",
            "light_n_L",
            "light_e1_L",
            "light_e2_L",
        ]

    os.makedirs(out_dir, exist_ok=True)

    sampler, points, log_w, log_l = load_posterior_from_checkpoint(
        prior, loglike, n_live=number_live, filepath=checkpoint_path
    )
    weights = np.exp(log_w)
    df = pd.DataFrame(points, columns=sampler.prior.keys)

    available = [p for p in params_to_corner if p in df.columns]
    samples_list = [df[p].to_numpy() for p in available]
    names = available

    if truth_values is not None:
        markers = [truth_values.get(k, np.nan) for k in names]
    else:
        markers = None

    settings_mcsamples = {
        "smooth_scale_1D": 0.5,
        "smooth_scale_2D": 0.5,
    }
    mcsamples_nautilus = MCSamples(
        samples=samples_list, names=names, settings=settings_mcsamples, weights=weights
    )

    g = plots.get_subplot_plotter(subplot_size=2)
    g.settings.legend_fontsize = 18
    g.settings.axes_labelsize = 14

    mcsamples_list = [mcsamples_nautilus]
    colors = ["tab:blue"]
    contour_lws = [2]
    legend_labels = ["Posterior (NAUTILUS)"]

    g.triangle_plot(
        mcsamples_list,
        params=names,
        legend_labels=legend_labels,
        filled=True,
        colors=colors,
        contour_colors=colors,
        markers=markers,
        contour_lws=contour_lws,
    )

    scale_factor = 1.1
    n = len(names)
    for i in range(n):
        ax_diag = g.subplots[i][i]
        x_lo, x_hi = ax_diag.get_xlim()
        x_mid = 0.5 * (x_hi + x_lo)
        x_half = 0.5 * (x_hi - x_lo) * scale_factor
        ax_diag.set_xlim(x_mid - x_half, x_mid + x_half)
        for j in range(i):
            ax = g.subplots[i][j]
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_mid = 0.5 * (x_hi + x_lo)
            y_mid = 0.5 * (y_hi + y_lo)
            ax.set_xlim(
                x_mid - 0.5 * (x_hi - x_lo) * scale_factor,
                x_mid + 0.5 * (x_hi - x_lo) * scale_factor,
            )
            ax.set_ylim(
                y_mid - 0.5 * (y_hi - y_lo) * scale_factor,
                y_mid + 0.5 * (y_hi - y_lo) * scale_factor,
            )

    if truth_values is not None:
        def _weighted_mean(x, w):
            w = np.asarray(w, float)
            w = w / (w.sum() + 1e-300)
            return float(np.sum(w * np.asarray(x, float)))

        for i, name in enumerate(names):
            mu = _weighted_mean(samples_list[i], weights)
            tv = truth_values.get(name, np.nan)
            ax = g.subplots[i][i]

            if rel_error and np.isfinite(tv) and abs(tv) > 1e-12:
                err_pct = 100.0 * (mu - tv) / tv
                label = f"$\\Delta\\%= {err_pct:+.2f}\\%$"
            else:
                err_abs = mu - tv
                label = (
                    f"$\\mathrm{{Truth}} = {tv:.3f}$\n"
                    f"$\\Delta = {err_abs:+.2e}$"
                )

            ax.text(
                0.98,
                0.96,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85
                ),
            )

    out_path = os.path.join(out_dir, f"{tag}.png")
    g.export(out_path, dpi=300)
    plt.show()
    print("Saved NAUTILUS corner plot to:", out_path)


def pso_corner_plot(
    out_dir: str,
    chain_path: str | None = None,
    params_to_corner: list[str] | None = None,
    truth_values: dict[str, float] | None = None,
    tag: str = "corner_pso",
    rel_error: bool = False,
):
    """Create GetDist corner plot from PSO swarm samples with optional truth markers.

    Analogous to nautilus_corner_plot but uses PSO swarm samples instead of
    nested sampling posterior.

    Args:
        out_dir: Directory containing pso_chain.npz and for saving output.
        chain_path: Path to PSO chain file. If None, uses out_dir/pso_chain.npz.
        params_to_corner: Parameter names to include. If None, uses defaults.
        truth_values: Optional dict mapping parameter names to true values
            for plotting truth markers and error annotations.
        tag: Filename prefix for the saved plot.
        rel_error: If True, show relative error instead of absolute.

    Raises:
        FileNotFoundError: If PSO chain file not found.
        ValueError: If no requested parameters found in chain.
    """
    if params_to_corner is None:
        params_to_corner = [
            "lens_theta_E",
            "lens_gamma",
            "lens_e1",
            "lens_e2",
            "light_Re_L",
            "light_n_L",
            "light_e1_L",
            "light_e2_L",
        ]

    if chain_path is None:
        chain_path = os.path.join(out_dir, "pso_chain.npz")

    if not os.path.exists(chain_path):
        raise FileNotFoundError(
            f"PSO chain file not found at {chain_path}.\n"
            "Re-run run_pso(..., save_chain=True) to create it."
        )

    data = np.load(chain_path, allow_pickle=True)
    samples_all = np.asarray(data["samples"], float)
    names_all = [str(n) for n in data["names"]]

    name_to_idx = {n: i for i, n in enumerate(names_all)}

    available = [p for p in params_to_corner if p in name_to_idx]
    if not available:
        raise ValueError(
            "None of the requested params_to_corner are in the PSO chain file."
        )
    names = available
    samples_list = [samples_all[:, name_to_idx[p]] for p in names]

    if truth_values is not None:
        markers = [truth_values.get(k, np.nan) for k in names]
    else:
        markers = None

    settings_mcsamples = {
        "smooth_scale_1D": 0.5,
        "smooth_scale_2D": 0.5,
    }
    weights = np.ones(samples_all.shape[0], dtype=float)

    mcsamples_pso = MCSamples(
        samples=samples_list, names=names, settings=settings_mcsamples, weights=weights
    )

    g = plots.get_subplot_plotter(subplot_size=2)
    g.settings.legend_fontsize = 18
    g.settings.axes_labelsize = 14

    g.triangle_plot(
        [mcsamples_pso],
        params=names,
        legend_labels=["PSO swarm"],
        filled=True,
        colors=["tab:orange"],
        contour_colors=["tab:orange"],
        contour_lws=[2],
        markers=markers,
    )

    scale_factor = 1.1
    n = len(names)
    for i in range(n):
        ax_diag = g.subplots[i][i]
        x_lo, x_hi = ax_diag.get_xlim()
        x_mid = 0.5 * (x_hi + x_lo)
        x_half = 0.5 * (x_hi - x_lo) * scale_factor
        ax_diag.set_xlim(x_mid - x_half, x_mid + x_half)
        for j in range(i):
            ax = g.subplots[i][j]
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_mid = 0.5 * (x_hi + x_lo)
            y_mid = 0.5 * (y_hi + y_lo)
            ax.set_xlim(
                x_mid - 0.5 * (x_hi - x_lo) * scale_factor,
                x_mid + 0.5 * (x_hi - x_lo) * scale_factor,
            )
            ax.set_ylim(
                y_mid - 0.5 * (y_hi - y_lo) * scale_factor,
                y_mid + 0.5 * (y_hi - y_lo) * scale_factor,
            )

    if truth_values is not None:
        def _weighted_mean(x, w):
            w = np.asarray(w, float)
            w = w / (w.sum() + 1e-300)
            return float(np.sum(w * np.asarray(x, float)))

        for i, name in enumerate(names):
            mu = _weighted_mean(samples_list[i], weights)
            tv = truth_values.get(name, np.nan)
            ax = g.subplots[i][i]

            if rel_error and np.isfinite(tv) and abs(tv) > 1e-12:
                err_pct = 100.0 * (mu - tv) / tv
                label = f"$\\Delta\\%= {err_pct:+.2f}\\%$"
            else:
                err_abs = mu - tv
                label = (
                    f"$\\mathrm{{Truth}} = {tv:.3f}$\n"
                    f"$\\Delta = {err_abs:+.2e}$"
                )

            ax.text(
                0.98,
                0.96,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85
                ),
            )

    out_path = os.path.join(out_dir, f"{tag}.png")
    g.export(out_path, dpi=300)
    plt.show()
    print("Saved PSO corner plot to:", out_path)


def plot_posterior_ray_tracing(
    samples: np.ndarray,
    param_names: list[str],
    mass_model,
    n_samples: int | None = 100,
    save_path: str | None = None,
    dpi: int = 300,
    random_seed: int = 42,
) -> dict:
    """Ray-trace image positions to source plane for posterior samples.

    Produces scatter plot of back-traced source positions and histogram of
    source position spread as a convergence quality metric.

    Args:
        samples: Posterior samples array of shape (n_total, n_params).
        param_names: Parameter names corresponding to sample columns.
        mass_model: Herculens MassModel instance (EPL + SHEAR).
        n_samples: Number of samples to use. If None, use all.
        save_path: Base path for saving plots (adds _scatter.png, _histogram.png).
        dpi: Plot resolution.
        random_seed: Random seed for sample selection.

    Returns:
        Dictionary containing:
            - all_x_src, all_y_src: Back-traced positions (n_samples, n_images)
            - spreads: Spread values for each sample
            - mean_spread_mas, median_spread_mas: Statistics in milliarcseconds
            - quality: Quality assessment string

    Raises:
        ValueError: If no x_image_* parameters found.
        KeyError: If required lens parameters missing.
    """
    def get_param(name, sample_idx=None):
        try:
            idx = param_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Parameter '{name}' not found in param_names") from exc
        if sample_idx is None:
            return samples[:, idx]
        return samples[sample_idx, idx]

    n_images = sum(1 for p in param_names if p.startswith('x_image_'))
    if n_images == 0:
        raise ValueError("No x_image_* parameters found in param_names")

    def get_kwargs_lens(sample_idx):
        return [
            dict(
                theta_E=float(get_param('lens_theta_E', sample_idx)),
                e1=float(get_param('lens_e1', sample_idx)),
                e2=float(get_param('lens_e2', sample_idx)),
                center_x=float(get_param('lens_center_x', sample_idx)),
                center_y=float(get_param('lens_center_y', sample_idx)),
                gamma=float(get_param('lens_gamma', sample_idx)),
            ),
            dict(
                gamma1=float(get_param('lens_gamma1', sample_idx)),
                gamma2=float(get_param('lens_gamma2', sample_idx)),
                ra_0=0.0,
                dec_0=0.0,
            ),
        ]

    def get_image_positions(sample_idx):
        x_imgs = np.array([get_param(f'x_image_{i}', sample_idx) for i in range(n_images)])
        y_imgs = np.array([get_param(f'y_image_{i}', sample_idx) for i in range(n_images)])
        return x_imgs, y_imgs

    n_total = samples.shape[0]
    if n_samples is None or n_samples > n_total:
        sample_indices = np.arange(n_total)
    else:
        rng = np.random.default_rng(random_seed)
        sample_indices = rng.choice(n_total, n_samples, replace=False)

    print(f'Ray-tracing {len(sample_indices)} posterior samples...')

    source_positions = []
    for idx in sample_indices:
        kwargs_lens = get_kwargs_lens(idx)
        x_img, y_img = get_image_positions(idx)
        x_src, y_src = mass_model.ray_shooting(x_img, y_img, kwargs_lens)
        source_positions.append((np.array(x_src), np.array(y_src)))

    all_x_src = np.array([sp[0] for sp in source_positions])
    all_y_src = np.array([sp[1] for sp in source_positions])

    spreads = []
    for i in range(len(source_positions)):
        x_src, y_src = source_positions[i]
        spread = np.sqrt(np.std(x_src)**2 + np.std(y_src)**2)
        spreads.append(spread)
    spreads = np.array(spreads)

    mean_spread_mas = np.mean(spreads) * 1000
    median_spread_mas = np.median(spreads) * 1000

    print('Source position spread:')
    print(f'  Mean: {mean_spread_mas:.2f} mas')
    print(f'  Median: {median_spread_mas:.2f} mas')
    print(f'  Max: {np.max(spreads)*1000:.2f} mas')

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'][:n_images]
    image_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:n_images]
    labels = [f'Image {letter}' for letter in image_letters]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for img_idx in range(n_images):
        x_src_all = all_x_src[:, img_idx]
        y_src_all = all_y_src[:, img_idx]
        ax.scatter(x_src_all, y_src_all, s=5, alpha=0.3, color=colors[img_idx], label=labels[img_idx])

    mean_x = np.mean(all_x_src)
    mean_y = np.mean(all_y_src)
    ax.scatter(mean_x, mean_y, s=200, marker='*', color='red', edgecolor='black',
               zorder=10, label='Mean source')

    ax.set_xlabel('x [arcsec]', fontsize=12)
    ax.set_ylabel('y [arcsec]', fontsize=12)
    ax.set_title(f'Back-traced source positions\n({n_images} images × {len(sample_indices)} posterior samples)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for img_idx in range(n_images):
        x_src_all = all_x_src[:, img_idx]
        y_src_all = all_y_src[:, img_idx]
        ax.scatter(x_src_all, y_src_all, s=20, alpha=0.5, color=colors[img_idx], label=labels[img_idx])

    ax.scatter(mean_x, mean_y, s=200, marker='*', color='red', edgecolor='black',
               zorder=10, label='Mean source')

    std_x = np.std(all_x_src)
    std_y = np.std(all_y_src)
    margin = max(std_x, std_y) * 5 + 0.01
    ax.set_xlim(mean_x - margin, mean_x + margin)
    ax.set_ylim(mean_y - margin, mean_y + margin)

    ax.set_xlabel('x [arcsec]', fontsize=12)
    ax.set_ylabel('y [arcsec]', fontsize=12)
    ax.set_title(f'Zoomed view\nSpread: {median_spread_mas:.2f} mas (median)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        scatter_path = save_path.replace('.png', '_scatter.png') if '.png' in save_path else f"{save_path}_scatter.png"
        plt.savefig(scatter_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved ray tracing scatter plot to: {scatter_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(spreads * 1000, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax.axvline(median_spread_mas, color='red', linestyle='--', linewidth=2,
               label=f'Median = {median_spread_mas:.2f} mas')

    ax.set_xlabel('Source position spread [mas]', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Distribution of {n_images}-image source plane convergence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        hist_path = save_path.replace('.png', '_histogram.png') if '.png' in save_path else f"{save_path}_histogram.png"
        plt.savefig(hist_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved ray tracing histogram to: {hist_path}")
    plt.close(fig)

    if median_spread_mas < 1:
        quality = "Excellent (< 1 mas)"
    elif median_spread_mas < 10:
        quality = "Good (< 10 mas)"
    elif median_spread_mas < 50:
        quality = "Acceptable (< 50 mas)"
    else:
        quality = "Warning: Large spread - model may have issues"
    print(f'Quality assessment: {quality}')

    return {
        "all_x_src": all_x_src,
        "all_y_src": all_y_src,
        "spreads": spreads,
        "mean_spread_mas": mean_spread_mas,
        "median_spread_mas": median_spread_mas,
        "quality": quality,
    }


def plot_corner_posterior(
    posterior: dict,
    param_names: list[str] | None = None,
    max_params: int = 10,
    save_path: str = None,
    title: str = "Posterior",
    dpi: int = 300,
    truths: dict[str, float] | None = None,
) -> None:
    """Create corner plot from posterior samples."""
    if not _HAS_CORNER and not _HAS_GETDIST:
        warnings.warn("Neither corner nor getdist available for corner plots", stacklevel=2)
        return

    samples = posterior.get("samples")
    all_names = posterior.get("param_names", [])

    if samples is None or len(all_names) == 0:
        return

    # Select parameters to plot (force D_dt to the end when available)
    has_d_dt = "D_dt" in all_names
    if param_names is None:
        if has_d_dt:
            keep = max(0, max_params - 1)
            base = [p for p in all_names if p != "D_dt"][:keep]
            param_names = base + ["D_dt"]
        else:
            param_names = all_names[:max_params]
    else:
        param_names = [p for p in param_names if p in all_names and p != "D_dt"]
        if has_d_dt:
            keep = max(0, max_params - 1)
            param_names = param_names[:keep] + ["D_dt"]
        else:
            param_names = param_names[:max_params]

    if len(param_names) == 0:
        return

    # Extract relevant columns
    indices = [all_names.index(p) for p in param_names]
    plot_samples = samples[:, indices]

    # Get truth values if provided
    truth_vals = None
    if truths is not None:
        truth_vals = [truths.get(p, None) for p in param_names]

    if _HAS_GETDIST:
        # Use getdist for nicer plots
        mc_samples = MCSamples(samples=plot_samples, names=param_names,
                               settings={"smooth_scale_1D": 0.5, "smooth_scale_2D": 0.5})
        g = gd_plots.get_subplot_plotter(subplot_size=2)
        g.triangle_plot(mc_samples, params=param_names, filled=True,
                        markers=truth_vals if truth_vals else None)
        if save_path:
            g.export(save_path, dpi=dpi)
    elif _HAS_CORNER:
        fig = corner.corner(plot_samples, labels=param_names, truths=truth_vals,
                           show_titles=True, title_kwargs={"fontsize": 10})
        fig.suptitle(title, fontsize=14)
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


__all__ = [
    "nautilus_corner_plot",
    "pso_corner_plot",
    "plot_posterior_ray_tracing",
    "plot_corner_posterior",
]
