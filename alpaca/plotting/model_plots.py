"""
Model visualization plots for lens modeling.

Contains functions for plotting best-fit models, Nautilus mean models,
PSO best models, and ray-tracing checks.

Functions:
    - plot_bestfit_model: Data/model/residual comparison plots.
    - nautilus_mean_model_plot: Nautilus posterior mean model and residuals.
    - pso_best_model_plot: PSO best-fit model using plot_bestfit_model.
    - plot_model_summary_custom: Custom data/model/residual summary.
    - plot_ray_tracing_check: Ray-trace image positions to source plane.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bestfit_model(
    prob_model,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    plotter,
    params: dict,
    outdir: str | None = None,
    tag: str = "bestfit",
    plotherculens: bool = True,
    plot_ownresiduals: bool = False,
    print_chi2: bool = True,
):
    """Plot best-fit model summary with data/model/residuals comparison.

    Args:
        prob_model: ProbModel instance providing params2kwargs method.
        lens_image: Herculens LensImage object.
        img: Observed image array.
        noise_map: Per-pixel noise standard deviation.
        plotter: Herculens Plotter object for styling.
        params: Constrained parameter dict from optimization or sampling.
        outdir: Directory to save plots. If None, not saved.
        tag: Filename prefix for saved plots.
        plotherculens: If True, generate Herculens model summary plot.
        plot_ownresiduals: If True, generate custom data/model/residual plot.
        print_chi2: If True, print reduced chi^2 to console.
    """
    kwargs_best = prob_model.params2kwargs(params)
    model_img = lens_image.model(**kwargs_best)
    resid = (img - model_img) / (noise_map + 1e-12)

    if plotherculens:
        fig = plotter.model_summary(
            lens_image,
            kwargs_best,
            show_source=True,
            kwargs_grid_source=dict(pixel_scale_factor=1),
        )
        if outdir is not None:
            diag_path = os.path.join(outdir, f"{tag}_herculens_diagnostics.png")
            plt.savefig(diag_path, dpi=200, bbox_inches="tight")
            print(f"Saved best-fit herculens-diagnostics to: {diag_path}")
        plt.show()

    if plot_ownresiduals:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        im0 = ax[0].imshow(img, origin="lower", cmap="afmhot")
        ax[0].set_title("Data")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(model_img, origin="lower", cmap="afmhot")
        ax[1].set_title("Model")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(resid, origin="lower", cmap="bwr", vmin=-5, vmax=5)
        ax[2].set_title("Residuals (S/N)")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if outdir is not None:
            diag_path = os.path.join(outdir, f"{tag}_own_diagnostics.png")
            plt.savefig(diag_path, dpi=200, bbox_inches="tight")
            print(f"Saved best-fit own-diagnostics to: {diag_path}")

        plt.show()

    chi2 = float(np.sum(resid**2))
    n_pix = img.size
    n_param = getattr(prob_model, "num_parameters_for_chi2", prob_model.num_parameters)
    chi2_red = chi2 / max(1, (n_pix - n_param))
    if print_chi2:
        print(f"reduced chi^2 = {chi2_red:.3f}")


def nautilus_mean_model_plot(
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    plotter,
    prior,
    loglike,
    paramdict_to_kwargs,
    rung: int,
    code_id: int,
    seed: int,
    number_live: int,
    out_dir: str | None = None,
):
    """Plot Nautilus posterior mean model and residuals.

    Args:
        lens_image: Herculens LensImage object.
        img: Observed image array.
        noise_map: Per-pixel noise standard deviation.
        plotter: Herculens Plotter object.
        prior: Nautilus prior object.
        loglike: Log-likelihood function.
        paramdict_to_kwargs: Function to convert parameter dict to kwargs_lens.
        rung: TDLMC rung number.
        code_id: Code ID within the rung.
        seed: Random seed identifying the system.
        number_live: Number of live points used.
        out_dir: Output directory. If None, uses default path.
    """
    from alpaca.sampler.nautilus import load_posterior_from_checkpoint

    if out_dir is None:
        out_dir = os.path.join(
            "nautilus_output",
            f"rung{rung}",
            f"code{code_id}",
            f"f160w-seed{seed}",
        )
    os.makedirs(out_dir, exist_ok=True)

    ckpt = os.path.join(
        "nautilus_output",
        f"run_checkpoint_rung{rung}_seed{seed}_{number_live}.hdf5",
    )

    sampler, points, log_w, log_l = load_posterior_from_checkpoint(
        prior, loglike, n_live=number_live, filepath=ckpt
    )
    weights = np.exp(log_w)
    cols = sampler.prior.keys
    df = pd.DataFrame(points, columns=cols)

    def wmean(x, w):
        w = np.asarray(w, float)
        w /= (w.sum() + 1e-300)
        return float(np.dot(np.asarray(x, float), w))

    mean_sample = {k: wmean(df[k].values, weights) for k in cols}
    kwargs_mean = paramdict_to_kwargs(mean_sample)

    fig = plotter.model_summary(
        lens_image,
        kwargs_mean,
        show_source=True,
        kwargs_grid_source=dict(pixel_scale_factor=1),
    )
    out_summary = os.path.join(
        out_dir,
        f"nautilus_mean_model_summary_rung{rung}_seed{seed}_nlive{number_live}.png",
    )
    plt.savefig(out_summary, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved mean-model summary to:", out_summary)

    model_img = lens_image.model(**kwargs_mean)
    resid = (img - model_img) / (noise_map + 1e-12)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    im0 = ax[0].imshow(img, origin="lower", cmap="afmhot")
    ax[0].set_title("Data")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(model_img, origin="lower", cmap="afmhot")
    ax[1].set_title("Mean model")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(resid, origin="lower", cmap="bwr", vmin=-5, vmax=5)
    ax[2].set_title("Residuals (S/N)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_diag = os.path.join(
        out_dir,
        f"nautilus_mean_model_diagnostics_rung{rung}_seed{seed}_nlive{number_live}.png",
    )
    plt.savefig(out_diag, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved NAUTILUS diagnostics to:", out_diag)

    chi2 = float(np.sum(resid**2))
    n_pix = img.size
    n_params_eff = len(cols)
    chi2_red = chi2 / max(1, (n_pix - n_params_eff))
    print(f"chi^2 = {chi2:.2f}   |   chi^2_red = {chi2_red:.3f}")


def pso_best_model_plot(
    prob_model,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    plotter,
    best_params_json: dict,
    out_dir: str | None = None,
    tag: str = "pso_best",
    plotherculens: bool = True,
    plot_ownresiduals: bool = False,
    print_chi2: bool = True,
):
    """Plot PSO best-fit model using same machinery as plot_bestfit_model.

    Handles both array-style parameters (x_image, y_image, ps_amp) and
    flattened parameters (x_image_0, x_image_1, ...).

    Args:
        prob_model: ProbModel instance.
        lens_image: Herculens LensImage object.
        img: Observed image array.
        noise_map: Per-pixel noise standard deviation.
        plotter: Herculens Plotter object.
        best_params_json: Best-fit parameter dict from PSO.
        out_dir: Output directory. If None, not saved.
        tag: Filename prefix for saved plots.
        plotherculens: If True, generate Herculens model summary.
        plot_ownresiduals: If True, generate custom residual plot.
        print_chi2: If True, print reduced chi^2.

    Returns:
        Output from plot_bestfit_model.
    """
    params = dict(best_params_json)

    has_flat_ps = any(k.startswith("x_image_") for k in params) and "x_image" not in params

    if has_flat_ps:
        indices = sorted(
            {
                int(k.split("_")[-1])
                for k in params.keys()
                if k.startswith("x_image_")
            }
        )
        if indices:
            x_im = np.array([params[f"x_image_{i}"] for i in indices])
            y_im = np.array([params[f"y_image_{i}"] for i in indices])
            amp_im = np.array([params[f"ps_amp_{i}"] for i in indices])

            params["x_image"] = x_im
            params["y_image"] = y_im
            params["ps_amp"] = amp_im

            for i in indices:
                params.pop(f"x_image_{i}", None)
                params.pop(f"y_image_{i}", None)
                params.pop(f"ps_amp_{i}", None)

    return plot_bestfit_model(
        prob_model=prob_model,
        lens_image=lens_image,
        img=img,
        noise_map=noise_map,
        plotter=plotter,
        params=params,
        outdir=out_dir,
        tag=tag,
        plotherculens=plotherculens,
        plot_ownresiduals=plot_ownresiduals,
        print_chi2=print_chi2,
    )


def plot_model_summary_custom(
    img: np.ndarray,
    model_img: np.ndarray,
    noise_map: np.ndarray,
    save_path: str,
    title: str = "Model Summary",
    dpi: int = 300,
) -> None:
    """Plot model summary: data, model, residuals."""
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
    "plot_bestfit_model",
    "nautilus_mean_model_plot",
    "pso_best_model_plot",
    "plot_model_summary_custom",
    "plot_ray_tracing_check",
]
