"""
ALPACA Plotting and Visualization

All plotting functions for the lens modeling pipeline.
Provides visualization of:
    - PSF comparison (initial vs reconstructed)
    - Herculens model residuals
    - Sampling diagnostics
    - Corner plots
    - Ray tracing scatter plots
    - Marginalized posterior distributions
    - Multi-start optimization history

Author: hkrizic
"""

import os
from typing import Dict, Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PSF Comparison Plot
# =============================================================================

def plot_psf_comparison(
    psf_initial: np.ndarray,
    psf_reconstructed: np.ndarray,
    save_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
):
    """Create PSF comparison plot (initial vs reconstructed vs residual).

    Args:
        psf_initial: Initial PSF array.
        psf_reconstructed: Reconstructed PSF array.
        save_path: Path to save the plot. If None, not saved.
        dpi: Plot resolution.
        show: If True, display the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Initial PSF
    im0 = axes[0].imshow(psf_initial, origin="lower", cmap="viridis")
    axes[0].set_title("Initial PSF")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Reconstructed PSF
    im1 = axes[1].imshow(psf_reconstructed, origin="lower", cmap="viridis")
    axes[1].set_title("Reconstructed PSF")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Residual
    residual = psf_reconstructed - psf_initial
    vmax = np.max(np.abs(residual))
    im2 = axes[2].imshow(residual, origin="lower", cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Residual (Reconstructed - Initial)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved PSF comparison to: {save_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Herculens Residual Plot
# =============================================================================

def plot_herculens_residual(
    prob_model,
    params: Dict,
    save_path: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
):
    """Save Herculens model residual plot.

    Args:
        prob_model: ProbModel instance with lens_image attribute.
        params: Parameter dictionary for the model.
        save_path: Path to save the plot. If None, not saved.
        dpi: Plot resolution.
        show: If True, display the plot.
    """
    from herculens.Inference.legacy import Plotter

    kwargs = prob_model.params2kwargs(params)
    lens_image = prob_model.lens_image

    # Create plotter
    plotter = Plotter(lens_image)

    # Generate model summary
    fig = plotter.model_summary(
        lens_image,
        kwargs,
        show_source=True,
        kwargs_grid_source=dict(pixel_scale_factor=1),
    )

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Herculens residual plot to: {save_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Sampling Diagnostics Plot
# =============================================================================

def plot_diagnostics(
    samples: np.ndarray,
    param_names: List[str],
    sampler: str,
    nuts_results: Optional[Dict] = None,
    nautilus_results: Optional[Dict] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
):
    """Create and save sampling diagnostics plot.

    Args:
        samples: Posterior samples array of shape (n_samples, n_params).
        param_names: List of parameter names.
        sampler: Sampler name ("nuts" or "nautilus").
        nuts_results: NUTS results dict (optional).
        nautilus_results: Nautilus results dict (optional).
        save_path: Path to save the plot. If None, not saved.
        dpi: Plot resolution.
        show: If True, display the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Sample distribution for key parameters
    key_params = ["D_dt", "lens_theta_E", "lens_gamma"]
    for param in key_params:
        if param in param_names:
            idx = param_names.index(param)
            axes[0, 0].hist(samples[:, idx], bins=50, alpha=0.6, label=param)
    axes[0, 0].set_xlabel("Parameter value")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Key parameter distributions")
    axes[0, 0].legend()

    # Plot 2: Trace plot for D_dt (if present)
    if "D_dt" in param_names:
        idx = param_names.index("D_dt")
        axes[0, 1].plot(samples[:, idx], alpha=0.7, lw=0.5)
        axes[0, 1].set_xlabel("Sample")
        axes[0, 1].set_ylabel("D_dt [Mpc]")
        axes[0, 1].set_title("D_dt trace")
    else:
        axes[0, 1].text(0.5, 0.5, "D_dt not found", ha="center", va="center",
                        transform=axes[0, 1].transAxes)

    # Plot 3: Autocorrelation (simplified)
    if "D_dt" in param_names:
        idx = param_names.index("D_dt")
        x = samples[:, idx]
        n = len(x)
        max_lag = min(100, n // 4)
        acf = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        acf = acf[n-1:n+max_lag] / acf[n-1]
        axes[1, 0].plot(acf)
        axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Lag")
        axes[1, 0].set_ylabel("Autocorrelation")
        axes[1, 0].set_title("D_dt autocorrelation")
    else:
        axes[1, 0].text(0.5, 0.5, "D_dt not found", ha="center", va="center",
                        transform=axes[1, 0].transAxes)

    # Plot 4: Summary statistics
    summary_text = f"Sampler: {sampler.upper()}\n"
    summary_text += f"Total samples: {samples.shape[0]}\n"
    summary_text += f"Parameters: {len(param_names)}\n\n"

    if sampler == "nuts" and nuts_results is not None:
        summary_text += f"Chains: {nuts_results['config'].get('num_chains', 'N/A')}\n"
        summary_text += f"Warmup: {nuts_results['config'].get('num_warmup', 'N/A')}\n"
        acc_rate = nuts_results.get('acceptance_rate')
        if acc_rate is not None:
            summary_text += f"Acceptance rate: {acc_rate:.3f}\n"
    elif sampler == "nautilus" and nautilus_results is not None:
        log_ev = nautilus_results.get('log_evidence')
        if log_ev is not None:
            summary_text += f"Log evidence: {log_ev:.2f}\n"
        summary_text += f"Raw samples: {nautilus_results.get('n_raw_samples', 'N/A')}\n"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title("Summary")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved diagnostics plot to: {save_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Corner Plot
# =============================================================================

def plot_corner(
    samples: np.ndarray,
    param_names: List[str],
    save_path: Optional[str] = None,
    dpi: int = 200,
    show: bool = False,
):
    """Create and save corner plot for key parameters using GetDist.

    Args:
        samples: Posterior samples array of shape (n_samples, n_params).
        param_names: List of parameter names.
        save_path: Path to save the plot. If None, not saved.
        dpi: Plot resolution.
        show: If True, display the plot.
    """
    from getdist import plots, MCSamples

    # Select key parameters for corner plot
    key_params = [
        "D_dt", "lens_theta_E", "lens_gamma", "lens_e1", "lens_e2",
        "light_Re_L", "light_n_L", "light_e1_L", "light_e2_L",
    ]

    available = [p for p in key_params if p in param_names]
    if len(available) < 2:
        # Fallback to first few parameters
        available = param_names[:min(8, len(param_names))]

    indices = [param_names.index(p) for p in available]
    samples_subset = [samples[:, i] for i in indices]

    settings = {"smooth_scale_1D": 0.5, "smooth_scale_2D": 0.5}
    mcsamples = MCSamples(samples=samples_subset, names=available, settings=settings)

    g = plots.get_subplot_plotter(subplot_size=2)
    g.settings.legend_fontsize = 14
    g.settings.axes_labelsize = 12

    g.triangle_plot(
        [mcsamples],
        params=available,
        filled=True,
        colors=["tab:blue"],
        contour_colors=["tab:blue"],
        contour_lws=[2],
    )

    if save_path is not None:
        g.export(save_path, dpi=dpi)
        plt.close()
        print(f"Saved corner plot to: {save_path}")
    elif show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# Ray Tracing Scatter Plot
# =============================================================================

def plot_ray_tracing_scatter(
    samples: np.ndarray,
    param_names: List[str],
    mass_model,
    save_path: Optional[str] = None,
    n_samples: int = 100,
    dpi: int = 200,
    random_seed: int = 42,
    show: bool = False,
):
    """Create and save ray tracing scatter plot (without histogram).

    Args:
        samples: Posterior samples array of shape (n_total, n_params).
        param_names: List of parameter names.
        mass_model: Herculens MassModel instance.
        save_path: Path to save the plot. If None, not saved.
        n_samples: Number of samples to use for ray tracing.
        dpi: Plot resolution.
        random_seed: Random seed for sample selection.
        show: If True, display the plot.
    """
    # Find image position parameters
    n_images = sum(1 for p in param_names if p.startswith('x_image_'))
    if n_images == 0:
        print("Warning: No x_image_* parameters found, skipping ray tracing plot")
        return

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

    # Sample subset
    n_total = samples.shape[0]
    n_samples = min(n_samples, n_total)
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(n_total, n_samples, replace=False)

    all_x_src = []
    all_y_src = []

    for idx in sample_indices:
        kwargs_lens = get_kwargs_lens(idx)
        if kwargs_lens is None:
            continue

        x_imgs = np.array([get_param(f'x_image_{i}', idx) for i in range(n_images)])
        y_imgs = np.array([get_param(f'y_image_{i}', idx) for i in range(n_images)])

        x_src, y_src = mass_model.ray_shooting(x_imgs, y_imgs, kwargs_lens)
        all_x_src.append(x_src)
        all_y_src.append(y_src)

    if len(all_x_src) == 0:
        print("Warning: Could not compute ray tracing, skipping plot")
        return

    all_x_src = np.array(all_x_src)
    all_y_src = np.array(all_y_src)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'][:n_images]
    labels = [f'Image {chr(65+i)}' for i in range(n_images)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full view
    ax = axes[0]
    for img_idx in range(n_images):
        x_src_all = all_x_src[:, img_idx]
        y_src_all = all_y_src[:, img_idx]
        ax.scatter(x_src_all, y_src_all, s=5, alpha=0.3, color=colors[img_idx], label=labels[img_idx])

    mean_x = np.mean(all_x_src)
    mean_y = np.mean(all_y_src)
    ax.scatter(mean_x, mean_y, s=200, marker='*', color='red', edgecolor='black',
               zorder=10, label='Mean source')
    ax.set_xlabel('x [arcsec]')
    ax.set_ylabel('y [arcsec]')
    ax.set_title(f'Back-traced source positions\n({n_images} images x {len(all_x_src)} samples)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Zoomed view
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

    spreads = np.sqrt(np.std(all_x_src, axis=1)**2 + np.std(all_y_src, axis=1)**2)
    median_spread_mas = np.median(spreads) * 1000

    ax.set_xlabel('x [arcsec]')
    ax.set_ylabel('y [arcsec]')
    ax.set_title(f'Zoomed view\nSpread: {median_spread_mas:.2f} mas (median)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved ray tracing scatter plot to: {save_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Marginalized Posteriors Plot
# =============================================================================

def plot_marginalized_posteriors(
    samples: np.ndarray,
    param_names: List[str],
    save_path: Optional[str] = None,
    dpi: int = 150,
    show: bool = False,
):
    """Create and save marginalized 1D posteriors for key parameters.

    Args:
        samples: Posterior samples array of shape (n_samples, n_params).
        param_names: List of parameter names.
        save_path: Path to save the plot. If None, not saved.
        dpi: Plot resolution.
        show: If True, display the plot.
    """
    # Key parameters to plot
    key_params = [
        "D_dt", "lens_theta_E", "lens_gamma", "lens_e1", "lens_e2",
        "lens_center_x", "lens_center_y", "lens_gamma1", "lens_gamma2",
        "light_Re_L", "light_n_L", "light_e1_L", "light_e2_L",
    ]

    available = [p for p in key_params if p in param_names]
    if len(available) == 0:
        available = param_names[:min(12, len(param_names))]

    n_params = len(available)
    ncols = 4
    nrows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for i, param in enumerate(available):
        idx = param_names.index(param)
        vals = samples[:, idx]

        ax = axes[i]
        ax.hist(vals, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')

        mean_val = np.mean(vals)
        median_val = np.median(vals)
        std_val = np.std(vals)

        ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.3f}')
        ax.axvline(mean_val, color='orange', linestyle=':', linewidth=2,
                   label=f'Mean: {mean_val:.3f}')

        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.set_title(f'{param}\n(std: {std_val:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved marginalized posteriors to: {save_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Multi-start History Plot
# =============================================================================

def plot_multistart_history(
    best_trace: Sequence[float],
    final_losses: Sequence[float],
    outdir: Optional[str] = None,
    rel_eps: float = 0.01,
    filename: str = "best_loss_vs_run.png",
    chi2_given: bool = False,
    dpi: int = 150,
    show: bool = True,
):
    """Plot optimization convergence across multi-start runs.

    Args:
        best_trace: Best loss value achieved after each run.
        final_losses: Final loss value of each individual run.
        outdir: Directory to save the plot. If None, not saved.
        rel_eps: Relative improvement threshold for marking last significant gain.
        filename: Output filename within outdir.
        chi2_given: If True, label axes as reduced chi^2 instead of loss.
        dpi: Plot resolution.
        show: If True, display the plot.
    """
    best_trace = np.asarray(best_trace, float)
    final_losses = np.asarray(final_losses, float)
    runs = np.arange(len(best_trace))

    fig = plt.figure(figsize=(7, 4.5))
    if chi2_given:
        plt.step(runs, best_trace, where="post", label="Best-so-far (reduced chi2)")
        plt.scatter(runs, final_losses, s=16, alpha=0.6, label="Per-run final reduced chi2")
    else:
        plt.step(runs, best_trace, where="post", label="Best-so-far (safe loss)")
        plt.scatter(runs, final_losses, s=16, alpha=0.6, label="Per-run final loss")

    if rel_eps is not None and len(best_trace) > 1:
        improvements = best_trace[:-1] - best_trace[1:]
        rel_impr = improvements / np.maximum(1e-12, best_trace[:-1])
        sig_idxs = np.where(rel_impr > rel_eps)[0]
        if sig_idxs.size > 0:
            last_sig = int(sig_idxs[-1] + 1)
            plt.axvline(last_sig, linestyle="--", alpha=0.5,
                        label=f"Last >{int(rel_eps*100)}% gain @ run {last_sig}")

    if chi2_given:
        plt.xlabel("Run #")
        plt.ylabel(r"Reduced $\chi^2$")
        plt.title(r"Best-so-far reduced $\chi^2$ vs. multi-start run")
    else:
        plt.xlabel("Run #")
        plt.ylabel("Safe loss")
        plt.title("Best-so-far loss vs. multi-start run")

    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()

    if outdir is not None:
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=dpi)
        print(f"Saved multi-start history plot to: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


__all__ = [
    "plot_psf_comparison",
    "plot_herculens_residual",
    "plot_diagnostics",
    "plot_corner",
    "plot_ray_tracing_scatter",
    "plot_marginalized_posteriors",
    "plot_multistart_history",
]
