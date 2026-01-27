"""
Diagnostic plots for lens modeling optimization and sampling.

Contains functions for multi-start optimization history, PSF comparison,
chain diagnostics, and NUTS sampling diagnostics.

Functions:
    - plot_multistart_history: Track convergence across optimization runs.
    - plot_multistart_summary: Multi-start chi^2 and loss distribution.
    - plot_chain_diagnostics: MCMC chain trace and histogram plots.
    - plot_psf_comparison: PSF initial vs final comparison.
    - plot_nuts_diagnostics: Comprehensive NUTS sampling diagnostics.
"""

import os
from collections.abc import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from numpyro.handlers import seed as numpyro_seed
from numpyro.infer.util import constrain_fn


def plot_multistart_history(
    best_trace: Sequence[float],
    final_losses: Sequence[float],
    outdir: str | None = None,
    rel_eps: float = 0.01,
    filename: str = "best_loss_vs_run.png",
    chi2_given: bool = False,
):
    """Plot optimization convergence across multi-start runs.

    Shows best-so-far loss vs run number with per-run final losses overlaid.
    Marks the last run where significant improvement (> rel_eps) occurred.

    Args:
        best_trace: Best loss value achieved after each run.
        final_losses: Final loss value of each individual run.
        outdir: Directory to save the plot. If None, not saved.
        rel_eps: Relative improvement threshold for marking last significant gain.
        filename: Output filename within outdir.
        chi2_given: If True, label axes as reduced chi^2 instead of loss.
    """
    best_trace = np.asarray(best_trace, float)
    final_losses = np.asarray(final_losses, float)
    runs = np.arange(len(best_trace))

    plt.figure(figsize=(7, 4.5))
    if chi2_given:
        plt.step(runs, best_trace, where="post", label="Best-so-far (reduced chi²)")
        plt.scatter(
            runs, final_losses, s=16, alpha=0.6, label="Per-run final reduced chi²"
        )
    else:
        plt.step(runs, best_trace, where="post", label="Best-so-far (safe loss)")
        plt.scatter(runs, final_losses, s=16, alpha=0.6, label="Per-run final loss")

    if rel_eps is not None and len(best_trace) > 1:
        improvements = best_trace[:-1] - best_trace[1:]
        rel_impr = improvements / np.maximum(1e-12, best_trace[:-1])
        sig_idxs = np.where(rel_impr > rel_eps)[0]
        if sig_idxs.size > 0:
            last_sig = int(sig_idxs[-1] + 1)
            plt.axvline(
                last_sig,
                linestyle="--",
                alpha=0.5,
                label=f"Last >{int(rel_eps*100)}% gain @ run {last_sig}",
            )
    if chi2_given:
        plt.xlabel("Run #")
        plt.ylabel(r"Reduced $\chi^2$")
        plt.title(r"Best-so-far reduced $\chi^2$ vs. multi-start run")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
    else:
        plt.xlabel("Run #")
        plt.ylabel("Safe loss")
        plt.title("Best-so-far loss vs. multi-start run")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
    plt.tight_layout()

    if outdir is not None:
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=150)
        print(f"Saved multi-start history plot to: {path}")

    plt.show()


def plot_multistart_summary(
    summary: dict,
    save_path: str,
    dpi: int = 300,
) -> None:
    """Plot multi-start optimization summary."""
    chi2_reds = np.array(summary.get("chi2_reds", []))
    losses = np.array(summary.get("all_losses", summary.get("final_losses", [])))
    best_run = summary.get("best_run", 0)

    if len(chi2_reds) == 0 and len(losses) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Chi^2 distribution
    if len(chi2_reds) > 0:
        runs = np.arange(len(chi2_reds))
        colors = ["green" if i == best_run else "steelblue" for i in runs]
        axes[0].bar(runs, chi2_reds, color=colors, edgecolor="black", alpha=0.7)
        axes[0].axhline(chi2_reds[best_run], color="red", linestyle="--",
                        label=f"Best: {chi2_reds[best_run]:.3f}")
        axes[0].set_xlabel("Run Index")
        axes[0].set_ylabel("Reduced χ²")
        axes[0].set_title("Multi-start χ² Distribution")
        axes[0].legend()

    # Loss distribution
    if len(losses) > 0:
        axes[1].hist(losses, bins=min(20, len(losses)), color="steelblue",
                     edgecolor="black", alpha=0.7)
        if best_run < len(losses):
            axes[1].axvline(losses[best_run], color="red", linestyle="--",
                            label=f"Best: {losses[best_run]:.3f}")
        axes[1].set_xlabel("Loss")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Loss Distribution")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_chain_diagnostics(
    chain: np.ndarray,
    param_names: list[str],
    save_path: str,
    n_params_to_plot: int = 8,
    dpi: int = 300,
) -> None:
    """Plot chain traces for MCMC diagnostics."""
    if chain.ndim == 3:
        # (n_steps, n_walkers, n_dim) -> flatten walkers
        n_steps, n_walkers, n_dim = chain.shape
        chain_flat = chain.reshape(n_steps, -1, n_dim).mean(axis=1)  # Average over walkers
    else:
        chain_flat = chain
        n_dim = chain.shape[1]

    n_plot = min(n_params_to_plot, n_dim, len(param_names))

    fig, axes = plt.subplots(n_plot, 2, figsize=(14, 2.5 * n_plot))
    if n_plot == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_plot):
        # Trace plot
        axes[i, 0].plot(chain_flat[:, i], alpha=0.7, lw=0.5)
        axes[i, 0].set_ylabel(param_names[i] if i < len(param_names) else f"param_{i}")
        axes[i, 0].set_xlabel("Step")

        # Histogram
        axes[i, 1].hist(chain_flat[:, i], bins=50, density=True, alpha=0.7)
        axes[i, 1].set_xlabel(param_names[i] if i < len(param_names) else f"param_{i}")
        axes[i, 1].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_psf_comparison(
    psf_initial: np.ndarray,
    psf_final: np.ndarray,
    psf_iterations: list[np.ndarray],
    sigma_map: np.ndarray | None,
    save_path: str,
    dpi: int = 300,
) -> None:
    """
    Plot PSF comparison: initial vs final.
    UPDATED: Always plots normalized residuals (Residual / sigma) if sigma_map is available.
    """
    n_iter = len(psf_iterations)

    fig, axes = plt.subplots(2, n_iter + 2, figsize=(4 * (n_iter + 2), 8))

    # --- Row 1: PSF images (Log Scale) ---
    vmin_log = max(1e-6, np.nanpercentile(psf_initial[psf_initial > 0], 1))
    vmax_log = np.nanpercentile(psf_initial, 99)

    # Initial
    im0 = axes[0, 0].imshow(psf_initial, origin="lower", cmap="viridis",
                             norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
    axes[0, 0].set_title("Initial PSF")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Iterations
    for i, psf_iter in enumerate(psf_iterations):
        im = axes[0, i + 1].imshow(psf_iter, origin="lower", cmap="viridis",
                                    norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
        axes[0, i + 1].set_title(f"Iteration {i + 1}")
        plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

    # Final
    im_final = axes[0, -1].imshow(psf_final, origin="lower", cmap="viridis",
                                   norm=LogNorm(vmin=vmin_log, vmax=vmax_log))
    axes[0, -1].set_title("Final PSF")
    plt.colorbar(im_final, ax=axes[0, -1], fraction=0.046, pad=0.04)

    # --- Row 2: Normalized Residuals ---
    axes[1, 0].axis("off")

    # Helper to calculate normalized residual safely
    def get_norm_residual(arr1, arr2, sigma):
        diff = arr1 - arr2
        if sigma is None:
            return diff, False # Fallback to raw subtraction

        eps = 1e-12
        res_norm = np.zeros_like(diff)
        # Avoid division by zero or NaN
        mask = (sigma > eps) & np.isfinite(sigma)
        np.divide(diff, sigma, out=res_norm, where=mask)
        return res_norm, True

    # Plot Iteration Residuals
    for i, psf_iter in enumerate(psf_iterations):
        # Determine what to subtract
        if i == 0:
            prev = psf_initial
            title_base = f"Iter {i+1} - Initial"
        else:
            prev = psf_iterations[i - 1]
            title_base = f"Iter {i+1} - Iter {i}"

        # Calculate Residual
        resid, is_norm = get_norm_residual(psf_iter, prev, sigma_map)

        # Set Title
        title = f"({title_base}) / σ" if is_norm else title_base

        # Determine scale (symmetric)
        vmax_r = np.nanpercentile(np.abs(resid), 99)
        if vmax_r == 0:
            vmax_r = 1e-5  # prevent crash on empty plot

        im_r = axes[1, i + 1].imshow(resid, origin="lower", cmap="seismic",
                                      vmin=-vmax_r, vmax=vmax_r)
        axes[1, i + 1].set_title(title)
        plt.colorbar(im_r, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    # Plot Final - Initial Residual (Always normalized if possible)
    resid_final, is_norm_final = get_norm_residual(psf_final, psf_initial, sigma_map)
    title_final = "(Final - Initial) / σ" if is_norm_final else "Final - Initial"

    vmax_f = np.nanpercentile(np.abs(resid_final), 99)
    if vmax_f == 0:
        vmax_f = 1e-5

    im_fn = axes[1, -1].imshow(resid_final, origin="lower", cmap="seismic",
                                vmin=-vmax_f, vmax=vmax_f)
    axes[1, -1].set_title(title_final)
    plt.colorbar(im_fn, ax=axes[1, -1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_nuts_diagnostics(
    nuts_results: dict,
    prob_model=None,
    param_names: list = None,
    max_params: int = 6,
    rolling_window: int = 50,
    figsize_per_chain: tuple[float, float] = (14, 10),
    outdir: str = None,
    dpi: int = 150,
):
    """Generate comprehensive diagnostic plots for NUTS sampling results.

    Creates one figure per chain showing:
    1. Log-density (negative loss) trace with divergence markers
    2. Parameter evolution with rolling mean and ±1σ uncertainty bands

    NUTS samples are in unconstrained space. If prob_model is provided,
    samples are transformed to constrained (physical) space.

    Args:
        nuts_results: Output from run_nuts_numpyro containing samples_by_chain,
            log_density_by_chain, divergences, and config.
        prob_model: Probabilistic model for unconstrained to constrained transform.
        param_names: Subset of parameter names to plot. If None, uses highest
            variance parameters.
        max_params: Maximum number of parameters to show (default 6).
        rolling_window: Window size for rolling statistics (default 50).
        figsize_per_chain: Figure size (width, height) for each chain's plot.
        outdir: Directory to save figures. If None, figures are displayed.
        dpi: Resolution for saved figures.

    Returns:
        List of figure objects, one per chain.

    Note:
        Uncertainty bands show ±1σ computed over a rolling window. Divergent
        transitions are marked with red vertical lines.
    """
    # Extract data from results
    samples_by_chain = nuts_results.get("samples_by_chain")
    log_density_by_chain = nuts_results.get("log_density_by_chain")
    divergences = nuts_results.get("divergences")
    config = nuts_results.get("config", {})

    if samples_by_chain is None:
        raise ValueError("nuts_results must contain 'samples_by_chain'. "
                         "Ensure run_nuts_numpyro was called with the result stored.")

    # Get dimensions
    first_key = list(samples_by_chain.keys())[0]
    num_chains, num_samples = samples_by_chain[first_key].shape[:2]

    # Transform from unconstrained to constrained (physical) space
    if prob_model is not None:
        constrained_by_chain = {k: [] for k in samples_by_chain.keys()}
        seeded_model = numpyro_seed(prob_model.model, rng_seed=0)

        for chain_idx in range(num_chains):
            chain_constrained = {k: [] for k in samples_by_chain.keys()}
            for sample_idx in range(num_samples):
                # Extract single sample for this chain
                sample_i = {}
                for k in samples_by_chain.keys():
                    arr = samples_by_chain[k]
                    if arr.ndim > 2:
                        sample_i[k] = jnp.array(arr[chain_idx, sample_idx])
                    else:
                        sample_i[k] = jnp.array(arr[chain_idx, sample_idx])

                # Transform to constrained space
                constrained_i = constrain_fn(seeded_model, (), {}, sample_i)
                for k in chain_constrained.keys():
                    chain_constrained[k].append(np.asarray(constrained_i[k]))

            for k in constrained_by_chain.keys():
                constrained_by_chain[k].append(np.array(chain_constrained[k]))

        # Stack into (num_chains, num_samples, ...) arrays
        samples_by_chain = {k: np.array(v) for k, v in constrained_by_chain.items()}

    # Handle divergences - reshape to (num_chains, num_samples) if needed
    if divergences is not None:
        divergences = np.asarray(divergences)
        if divergences.ndim == 1:
            # Reshape flat divergences to per-chain
            divergences = divergences.reshape(num_chains, num_samples)

    # Determine which parameters to plot
    all_param_names = list(samples_by_chain.keys())

    if param_names is None:
        # Select parameters with highest variance (most informative)
        variances = {}
        for pname in all_param_names:
            arr = np.asarray(samples_by_chain[pname])
            # Flatten across chains and samples for variance calculation
            if arr.ndim > 2:
                arr = arr.reshape(num_chains * num_samples, -1)
                variances[pname] = np.mean(np.var(arr, axis=0))
            else:
                variances[pname] = np.var(arr.reshape(-1))

        # Sort by variance and take top max_params
        sorted_params = sorted(variances.keys(), key=lambda k: variances[k], reverse=True)
        param_names = sorted_params[:max_params]
    else:
        # Validate provided names
        param_names = [p for p in param_names if p in all_param_names][:max_params]

    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("No valid parameters to plot.")

    # Helper function for rolling statistics
    def rolling_stats(x, window):
        """Compute rolling mean and std."""
        n = len(x)
        means = np.zeros(n)
        stds = np.zeros(n)

        for i in range(n):
            start = max(0, i - window + 1)
            window_data = x[start:i + 1]
            means[i] = np.mean(window_data)
            stds[i] = np.std(window_data)

        return means, stds

    figures = []

    for chain_idx in range(num_chains):
        # Create figure with subplots
        # Layout: top row for log-density, remaining rows for parameters
        n_rows = 1 + n_params
        fig = plt.figure(figsize=(figsize_per_chain[0], figsize_per_chain[1] * n_rows / 4))
        gs = GridSpec(n_rows, 1, height_ratios=[1.5] + [1] * n_params, hspace=0.3)

        sample_indices = np.arange(num_samples)

        # ============================================================
        # Panel 1: Log-density trace with divergence markers
        # ============================================================
        ax_logp = fig.add_subplot(gs[0])

        if log_density_by_chain is not None:
            logp_chain = np.asarray(log_density_by_chain[chain_idx])
            ax_logp.plot(sample_indices, logp_chain, 'b-', lw=0.5, alpha=0.7, label='Log-density')

            # Add rolling mean
            logp_mean, logp_std = rolling_stats(logp_chain, rolling_window)
            ax_logp.plot(sample_indices, logp_mean, 'b-', lw=1.5, label=f'Rolling mean (w={rolling_window})')

            # Mark divergences
            if divergences is not None:
                div_chain = divergences[chain_idx]
                div_indices = np.where(div_chain)[0]
                if len(div_indices) > 0:
                    # Draw vertical lines at divergence locations
                    for div_idx in div_indices:
                        ax_logp.axvline(div_idx, color='red', alpha=0.5, lw=0.8)
                    # Add a single legend entry for divergences
                    ax_logp.axvline(div_indices[0], color='red', alpha=0.5, lw=0.8,
                                   label=f'Divergences (n={len(div_indices)})')

        ax_logp.set_xlabel('Sample index')
        ax_logp.set_ylabel('Log-density')
        ax_logp.set_title(f'Chain {chain_idx + 1}: Log-density Trace', fontsize=12, fontweight='bold')
        ax_logp.legend(loc='lower right', fontsize=9)
        ax_logp.set_xlim(0, num_samples - 1)
        ax_logp.grid(True, alpha=0.3)

        # ============================================================
        # Panels 2+: Parameter evolution with uncertainty bands
        # ============================================================
        for param_idx, pname in enumerate(param_names):
            ax = fig.add_subplot(gs[1 + param_idx])

            param_data = np.asarray(samples_by_chain[pname])

            # Handle multi-dimensional parameters (take first component)
            if param_data.ndim > 2:
                param_chain = param_data[chain_idx, :, 0]
                display_name = f"{pname}[0]"
            else:
                param_chain = param_data[chain_idx, :]
                display_name = pname

            # Plot raw trace
            ax.plot(sample_indices, param_chain, 'k-', lw=0.3, alpha=0.4)

            # Compute and plot rolling statistics
            param_mean, param_std = rolling_stats(param_chain, rolling_window)

            # Plot mean line
            ax.plot(sample_indices, param_mean, 'b-', lw=1.5, label='Rolling mean')

            # Plot ±1σ bands
            ax.fill_between(sample_indices,
                           param_mean - param_std,
                           param_mean + param_std,
                           alpha=0.3, color='blue', label='±1σ')

            # Mark divergences on parameter trace
            if divergences is not None:
                div_chain = divergences[chain_idx]
                div_indices = np.where(div_chain)[0]
                if len(div_indices) > 0:
                    ax.scatter(div_indices, param_chain[div_indices],
                              c='red', s=15, alpha=0.7, zorder=5, marker='x')

            ax.set_xlabel('Sample index')
            ax.set_ylabel(display_name)
            ax.set_xlim(0, num_samples - 1)
            ax.grid(True, alpha=0.3)

            # Add legend only to first parameter panel
            if param_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

            # Add summary statistics as text
            final_mean = np.mean(param_chain)
            final_std = np.std(param_chain)
            stats_text = f'μ={final_mean:.4g}, σ={final_std:.4g}'
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add overall figure title
        n_div = 0
        if divergences is not None:
            n_div = np.sum(divergences[chain_idx])

        warmup = config.get('num_warmup', 'N/A')
        fig.suptitle(f'NUTS Diagnostics - Chain {chain_idx + 1}/{num_chains}\n'
                    f'Samples: {num_samples}, Warmup: {warmup}, '
                    f'Divergences: {n_div}',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            save_path = os.path.join(outdir, f"nuts_diagnostics_chain{chain_idx + 1:02d}.png")
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"[NUTS Diagnostics] Saved chain {chain_idx + 1} plot to: {save_path}")

        figures.append(fig)

    if outdir is None:
        plt.show()

    return figures


def visualize_initial_guess(init_params, lens_image, prob_model, data, plotter):
    """Create a diagnostic plot comparing initial model guess to data.

    Args:
        init_params: Initial parameter dictionary.
        lens_image: Herculens LensImage object.
        prob_model: Probabilistic model with params2kwargs method.
        data: Observed image array.
        plotter: Herculens Plotter object for color normalization.

    Returns:
        Tuple of (fig, axes) matplotlib objects.
    """
    from herculens.Util import plot_util
    from matplotlib.colors import TwoSlopeNorm

    initial_model = lens_image.model(**prob_model.params2kwargs(init_params))

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    axes[0].set_title("Initial guess model")
    im = axes[0].imshow(initial_model, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(im)

    axes[1].set_title("Data")
    im = axes[1].imshow(data, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(im)

    axes[2].set_title("Difference")
    im = axes[2].imshow(initial_model - data, origin='lower', norm=TwoSlopeNorm(0), cmap=plotter.cmap_res)
    plot_util.nice_colorbar(im)

    fig.tight_layout()
    return fig, axes


__all__ = [
    "plot_multistart_history",
    "plot_multistart_summary",
    "plot_chain_diagnostics",
    "plot_psf_comparison",
    "plot_nuts_diagnostics",
    "visualize_initial_guess",
]
