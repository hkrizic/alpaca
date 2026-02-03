"""
Model visualization plots for lens modeling.

author: hkrizic

Contains functions for plotting best-fit models, Nautilus mean models,
PSO best models, and ray-tracing checks.

Functions
---------
plot_model_summary_custom
    Custom data/model/residual summary.
plot_ray_tracing_check
    Ray-trace image positions to source plane.
"""


import matplotlib.pyplot as plt
import numpy as np


def plot_model_summary_custom(
    img: np.ndarray,
    model_img: np.ndarray,
    noise_map: np.ndarray,
    save_path: str,
    title: str = "Model Summary",
    dpi: int = 300,
) -> None:
    """
    Plot model summary showing data, model, raw residuals, and normalized residuals.

    Creates a four-panel figure comparing the observed image with the
    model prediction. The third panel shows the raw difference and the
    fourth shows the noise-normalized residual, clipped to [-5, 5] sigma.

    Parameters
    ----------
    img : np.ndarray
        Observed image data (2-D array).
    model_img : np.ndarray
        Model-predicted image (2-D array, same shape as ``img``).
    noise_map : np.ndarray
        Per-pixel noise standard deviation map (2-D array, same shape as
        ``img``). Used to compute the normalized residual.
    save_path : str
        File path to save the generated figure.
    title : str, optional
        Overall figure title, by default ``"Model Summary"``.
    dpi : int, optional
        Resolution for the saved figure, by default 300.

    Returns
    -------
    None
        The plot is saved to disk at ``save_path``.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Data
    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)
    im0 = axes[0].imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Data")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Model
    im1 = axes[1].imshow(model_img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Model")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Residual
    resid = img - model_img
    vmax_r = np.nanpercentile(np.abs(resid), 99)
    im2 = axes[2].imshow(resid, origin="lower", cmap="seismic", vmin=-vmax_r, vmax=vmax_r)
    axes[2].set_title("Residual (Data - Model)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Normalized residual
    eps = 1e-12
    resid_norm = np.zeros_like(resid)
    np.divide(resid, noise_map, out=resid_norm, where=np.isfinite(noise_map) & (noise_map > eps))
    im3 = axes[3].imshow(resid_norm, origin="lower", cmap="seismic", vmin=-5, vmax=5)
    axes[3].set_title("Normalized Residual")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ray_tracing_check(
    best_params: dict,
    mass_model,
    save_path: str | None = None,
    dpi: int = 300,
    title: str = "Ray Tracing Check",
) -> dict:
    """
    Ray-trace the 4 image positions back to the source plane to verify model quality.

    If the lens model is correct, all 4 rays should converge to the same source position.

    Parameters
    ----------
    best_params : Dict
        Best-fit parameters from optimization. Accepts two formats:
        - Array format: 'x_image', 'y_image' as arrays (multistart-style)
        - Flattened format: 'x_image_0', 'x_image_1', etc. (posterior sample-style)
        Must also contain lens_theta_E, lens_e1, lens_e2, lens_center_x, lens_center_y,
        lens_gamma, lens_gamma1, lens_gamma2, and optionally src_center_x, src_center_y.
    mass_model : MassModel
        Herculens MassModel instance (EPL + SHEAR).
    save_path : str, optional
        Path to save the plot. If None, plot is shown but not saved.
    dpi : int
        Plot resolution.
    title : str
        Plot title.

    Returns
    -------
    Dict
        Dictionary with ray tracing results:
        - x_source: array of 4 back-traced x positions
        - y_source: array of 4 back-traced y positions
        - spread_arcsec: standard deviation of source positions (arcsec)
        - spread_mas: spread in milliarcseconds
    """
    # Extract image positions - handle both array and flattened formats
    if 'x_image' in best_params and not isinstance(best_params['x_image'], (int, float)):
        # Array format (multistart-style): x_image is a list/array
        x_img = np.array(best_params['x_image'])
        y_img = np.array(best_params['y_image'])
    elif 'x_image_0' in best_params:
        # Flattened format (posterior sample-style): x_image_0, x_image_1, etc.
        n_images = sum(1 for k in best_params if k.startswith('x_image_'))
        x_img = np.array([best_params[f'x_image_{i}'] for i in range(n_images)])
        y_img = np.array([best_params[f'y_image_{i}'] for i in range(n_images)])
    else:
        raise KeyError("best_params must contain either 'x_image' array or 'x_image_0', 'x_image_1', etc.")

    # Build kwargs_lens from best_params
    kwargs_lens = [
        dict(
            theta_E=float(best_params["lens_theta_E"]),
            e1=float(best_params["lens_e1"]),
            e2=float(best_params["lens_e2"]),
            center_x=float(best_params["lens_center_x"]),
            center_y=float(best_params["lens_center_y"]),
            gamma=float(best_params["lens_gamma"]),
        ),
        dict(
            gamma1=float(best_params["lens_gamma1"]),
            gamma2=float(best_params["lens_gamma2"]),
            ra_0=0.0,
            dec_0=0.0,
        ),
    ]

    # Ray-trace to source plane
    x_src, y_src = mass_model.ray_shooting(x_img, y_img, kwargs_lens)
    x_src = np.array(x_src)
    y_src = np.array(y_src)

    # Compute spread (should be ~0 for good model)
    spread = np.sqrt(np.std(x_src)**2 + np.std(y_src)**2)
    spread_mas = spread * 1000

    # Get model source center for comparison
    src_center_x = float(best_params.get("src_center_x", np.mean(x_src)))
    src_center_y = float(best_params.get("src_center_y", np.mean(y_src)))

    # Create plot
    n_images = len(x_img)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'][:n_images]
    image_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:n_images]
    labels = [f'Image {letter}' for letter in image_letters]

    # Left: Image plane
    ax = axes[0]
    for i in range(n_images):
        ax.scatter(x_img[i], y_img[i], s=150, color=colors[i], edgecolor='black',
                   marker='o', label=labels[i], zorder=5)
        ax.annotate(image_letters[i], (x_img[i], y_img[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=11, fontweight='bold')

    # Mark lens center
    lens_x = float(best_params["lens_center_x"])
    lens_y = float(best_params["lens_center_y"])
    ax.scatter(lens_x, lens_y, s=200, marker='+', color='black', linewidths=3,
               label='Lens center', zorder=6)

    ax.set_xlabel('x [arcsec]', fontsize=11)
    ax.set_ylabel('y [arcsec]', fontsize=11)
    ax.set_title('Image Plane', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: Source plane
    ax = axes[1]
    for i in range(n_images):
        ax.scatter(x_src[i], y_src[i], s=150, color=colors[i], edgecolor='black',
                   marker='o', label=f'From {labels[i]}', zorder=5)
        ax.annotate(image_letters[i], (x_src[i], y_src[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=11, fontweight='bold')

    # Mark model source center
    ax.scatter(src_center_x, src_center_y, s=200, marker='*', color='red',
               edgecolor='black', label='Source center (model)', zorder=6)

    # Set appropriate zoom
    margin = max(spread * 5, 0.02)
    mean_x, mean_y = np.mean(x_src), np.mean(y_src)
    ax.set_xlim(mean_x - margin, mean_x + margin)
    ax.set_ylim(mean_y - margin, mean_y + margin)

    ax.set_xlabel('x [arcsec]', fontsize=11)
    ax.set_ylabel('y [arcsec]', fontsize=11)
    ax.set_title(f'Source Plane (back-traced)\nSpread = {spread_mas:.2f} mas', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return {
        "x_source": x_src,
        "y_source": y_src,
        "x_image": x_img,
        "y_image": y_img,
        "spread_arcsec": spread,
        "spread_mas": spread_mas,
        "src_center_model": (src_center_x, src_center_y),
    }


__all__ = [
    "plot_model_summary_custom",
    "plot_ray_tracing_check",
]
