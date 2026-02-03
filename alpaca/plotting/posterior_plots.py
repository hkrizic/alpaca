"""
Posterior distribution plots for lens modeling.

author: hkrizic

Contains corner plots for Nautilus and PSO posteriors, ray-tracing
convergence analysis from posterior samples, and generic corner plots.

Functions
---------
plot_posterior_ray_tracing
    Source plane convergence analysis from posterior samples.
plot_corner_posterior
    Generic corner plot from posterior samples.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples

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


def plot_posterior_ray_tracing(
    samples: np.ndarray,
    param_names: list[str],
    mass_model,
    n_samples: int | None = 100,
    save_path: str | None = None,
    dpi: int = 300,
    random_seed: int = 42,
) -> dict:
    """
    Ray-trace image positions to the source plane for posterior samples.

    Produces a scatter plot of back-traced source positions and a histogram of
    source-position spread as a convergence quality metric.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples array of shape ``(n_total, n_params)``.
    param_names : list of str
        Parameter names corresponding to columns in ``samples``.
    mass_model : MassModel
        Herculens ``MassModel`` instance (EPL + SHEAR).
    n_samples : int or None, optional
        Number of posterior samples to use. If ``None``, all samples are
        used, by default 100.
    save_path : str or None, optional
        Base path for saving plots (appends ``_scatter.png``). If ``None``,
        no file is written.
    dpi : int, optional
        Resolution for saved figures, by default 300.
    random_seed : int, optional
        Random seed for reproducible sample selection, by default 42.

    Returns
    -------
    dict
        Dictionary containing:

        - ``all_x_src`` : np.ndarray of shape ``(n_samples, n_images)``
          Back-traced x source positions.
        - ``all_y_src`` : np.ndarray of shape ``(n_samples, n_images)``
          Back-traced y source positions.
        - ``spreads`` : np.ndarray
          Spread values for each posterior sample (arcsec).
        - ``mean_spread_mas`` : float
          Mean spread in milliarcseconds.
        - ``median_spread_mas`` : float
          Median spread in milliarcseconds.
        - ``quality`` : str
          Quality assessment string.

    Raises
    ------
    ValueError
        If no ``x_image_*`` parameters are found in ``param_names``.
    KeyError
        If required lens parameters are missing from ``param_names``.
    """
    def get_param(name, sample_idx=None):
        """
        Retrieve a parameter column or single value from the samples array.

        Parameters
        ----------
        name : str
            Parameter name to look up.
        sample_idx : int or None, optional
            If provided, return the value for that sample index. If ``None``,
            return the full column.

        Returns
        -------
        np.ndarray or float
            Parameter values for all samples or a single sample.

        Raises
        ------
        KeyError
            If ``name`` is not found in ``param_names``.
        """
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
        """
        Build Herculens ``kwargs_lens`` for a single posterior sample.

        Parameters
        ----------
        sample_idx : int
            Index into the samples array.

        Returns
        -------
        list of dict
            Two-element list: EPL parameters and SHEAR parameters.
        """
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
        """
        Extract image-plane positions for a single posterior sample.

        Parameters
        ----------
        sample_idx : int
            Index into the samples array.

        Returns
        -------
        x_imgs : np.ndarray
            x-coordinates of the lensed images.
        y_imgs : np.ndarray
            y-coordinates of the lensed images.
        """
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
    """
    Create a corner plot from posterior samples.

    Uses ``getdist`` when available for publication-quality triangle plots,
    falling back to the ``corner`` package. If ``D_dt`` is among the
    parameters it is always placed last. Parameters are automatically
    limited to ``max_params``.

    Parameters
    ----------
    posterior : dict
        Dictionary with keys ``"samples"`` (2-D array of shape
        ``(n_samples, n_params)``) and ``"param_names"`` (list of str).
    param_names : list of str or None, optional
        Subset of parameter names to include. If ``None``, the first
        ``max_params`` parameters are used (with ``D_dt`` forced last
        when present).
    max_params : int, optional
        Maximum number of parameters to display, by default 10.
    save_path : str or None, optional
        File path to save the figure. If ``None``, the plot is not saved.
    title : str, optional
        Figure title (used only with the ``corner`` backend),
        by default ``"Posterior"``.
    dpi : int, optional
        Resolution for the saved figure, by default 300.
    truths : dict of str to float or None, optional
        Known true values to mark on the plot, keyed by parameter name.

    Returns
    -------
    None
        The plot is saved to disk at ``save_path`` when provided.
    """
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
    "plot_posterior_ray_tracing",
    "plot_corner_posterior",
]
