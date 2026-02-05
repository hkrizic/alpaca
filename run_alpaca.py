"""
Lens Modeling Pipeline Script (Generic)

Entry-point for running the Alpaca lens modeling pipeline on arbitrary data.
Specify your data paths, point-source image positions, and time delays below,
then run this script.  All other pipeline settings live in ``run_config.py``.

Pipeline phases:
    1. PSF Reconstruction - Iterative reconstruction using STARRED
    2. Gradient Descent Optimization - Two-phase MAP estimation
    3. Posterior Sampling - NUTS (NumPyro) or Nautilus

For TDLMC data, see run_alpaca_tdc.py instead (uses TDC folder structure).

Author: hkrizic
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from alpaca.pipeline import run_pipeline
from alpaca.utils.bic import compute_bic_from_results
from run_config import load_config, load_data, parse_positions_and_delays

# =============================================================================
# 1. DATA PATHS  (edit these to point to your FITS / npy files)
# =============================================================================
IMAGE_PATH = "./data/image.fits"           # 2-D lens image
PSF_PATH = "./data/psf.fits"               # 2-D PSF kernel
NOISE_MAP_PATH = "./data/noise_map.fits"   # 2-D noise map (same shape as image)

# =============================================================================
# 2. APPROXIMATE POINT SOURCE IMAGE POSITIONS  (arcsec, relative to lens center)
#
#    These approximate positions are used to LABEL the auto-detected images
#    (A, B, C, D), so that time delays are assigned to the correct pairs.
#    The pipeline always auto-detects the actual positions from the image.
#    The first entry is the reference image for time delays.
# =============================================================================
IMAGE_POSITIONS = {
    "A": (-0.088, -1.209),
    "B": ( 1.328,  0.057),
    "C": (-1.307,  0.1  ),
    "D": (-0.052,  1.227),
}

# =============================================================================
# 3. TIME DELAYS  (days, relative to the reference image)
#
#    Keys must match the non-reference images in IMAGE_POSITIONS.
#    Values are (delay, 1-sigma error) in days.
#    Set to None to disable time-delay likelihood entirely.
# =============================================================================
TIME_DELAYS = {
    "B": (-11.645, 0.25), # B-A
    "C": (-9.403,  0.25), # C-A
    "D": (-2.139,  0.25), # D-A
}
# TIME_DELAYS = None   # uncomment to run without time delays


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Run the complete ALPACA lens modeling pipeline on generic data.

    Loads data from paths defined at the top of this script, builds
    a ``PipelineConfig`` via ``run_config.load_config``, executes the
    three pipeline phases (PSF reconstruction, gradient descent, posterior
    sampling), and prints a summary of the results including parameter
    estimates, marginalized posterior plots, and BIC.

    Returns
    -------
    None
        All outputs are saved to disk and printed to stdout.
    """
    print("=" * 50)
    print("=" * 50)
    print("ALPACA Lens Modeling Pipeline")
    print("=" * 50)
    print("=" * 50)
    print("\nThis script runs the complete lens modeling pipeline:")
    print("  1. PSF Reconstruction")
    print("  2. Gradient Descent Optimization (MAP estimation)")
    print("  3. Posterior Sampling (NUTS or Nautilus)")
    print("\nEdit run_config.py for pipeline settings.")
    print("\nStarting pipeline...\n")

    # ---- Configuration ----
    use_time_delays = TIME_DELAYS is not None
    config = load_config(use_time_delays=use_time_delays)

    # ---- Load data ----
    print("\nLoading data...")
    print(f"  Image:     {IMAGE_PATH}")
    print(f"  PSF:       {PSF_PATH}")
    print(f"  Noise map: {NOISE_MAP_PATH}")

    img, psf_kernel, noise_map = load_data(IMAGE_PATH, PSF_PATH, NOISE_MAP_PATH)

    print(f"  Image shape: {img.shape}")
    print(f"  PSF shape:   {psf_kernel.shape}")

    # ---- Parse positions and time delays ----
    print("\nPoint source images:")
    image_positions, measured_delays, delay_errors, labels = (
        parse_positions_and_delays(IMAGE_POSITIONS, TIME_DELAYS)
    )

    n_images = len(labels)
    print(f"  {n_images} images: {', '.join(labels)}")
    for label, (x, y) in IMAGE_POSITIONS.items():
        print(f"    {label}: ({x:+.4f}, {y:+.4f}) arcsec")

    if TIME_DELAYS is not None:
        print("\nTime delays:")
    else:
        print("\nNo time delays provided (running without time-delay likelihood).")

    # ---- Run pipeline ----
    results = run_pipeline(
        config=config,
        img=img,
        psf_kernel=psf_kernel,
        noise_map=noise_map,
        image_positions=image_positions,
        measured_delays=measured_delays,
        delay_errors=delay_errors,
        time_delay_labels=labels,
        verbose=True,
    )

    # ---- Explore results ----
    output_dir = results["output_dirs"]["root"]
    print(f"\nResults saved to: {output_dir}")
    print("\nDirectory structure:")
    for name, path in results["output_dirs"].items():
        rel_path = os.path.relpath(path, output_dir)
        print(f"  {name}: {rel_path}")

    # PSF Reconstruction Results
    if "psf_result" in results and results["psf_result"] is not None:
        psf_result = results["psf_result"]
        print("\nPSF Reconstruction:")
        print(f"  Number of iterations: {psf_result['n_iterations']}")
    else:
        print("\nPSF reconstruction was not run.")

    # Multi-start Optimization Results
    if "multistart_summary" in results and results["multistart_summary"] is not None:
        ms_summary = results["multistart_summary"]
        print("\nMulti-start Optimization:")
        print(f"  Best loss: {ms_summary.get('best_loss', 'N/A'):.4f}")
        print(f"  Best chi2_red: {ms_summary.get('best_chi2_red', 'N/A'):.4f}")

    # Posterior Sampling Results
    if "posterior" in results and results["posterior"] is not None:
        posterior = results["posterior"]
        samples = posterior["samples"]
        param_names = posterior.get("param_names", [])

        print("\nPosterior Sampling:")
        print(f"  Engine: {posterior.get('engine', 'N/A')}")
        print(f"  Number of samples: {samples.shape[0]}")

        print("\n  Parameter summaries (mean +/- std):")
        key_params = [
            "lens_theta_E", "lens_gamma", "lens_e1", "lens_e2",
            "light_Re_L", "light_n_L", "D_dt",
        ]
        for param in key_params:
            if param in param_names:
                idx = param_names.index(param)
                vals = samples[:, idx]
                print(f"    {param}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        if "log_sigma_rayshoot_sys" in param_names:
            idx = param_names.index("log_sigma_rayshoot_sys")
            log_vals = samples[:, idx]
            vals = np.exp(log_vals)
            print(f"    sigma_rayshoot_sys: {np.mean(vals)*1000:.4f} +/- {np.std(vals)*1000:.4f} mas")
        elif "sigma_rayshoot_sys" in param_names:
            idx = param_names.index("sigma_rayshoot_sys")
            vals = samples[:, idx]
            print(f"    sigma_rayshoot_sys: {np.mean(vals)*1000:.4f} +/- {np.std(vals)*1000:.4f} mas")

        # Marginalized Posteriors
        key_params_plot = [
            p for p in ["lens_theta_E", "lens_gamma", "light_Re_L", "light_n_L", "D_dt"]
            if p in param_names
        ]

        if key_params_plot:
            fig, axes = plt.subplots(1, len(key_params_plot), figsize=(4*len(key_params_plot), 4))
            if len(key_params_plot) == 1:
                axes = [axes]

            for ax, param in zip(axes, key_params_plot):
                idx = param_names.index(param)
                vals = samples[:, idx]
                ax.hist(vals, bins=50, density=True, alpha=0.7, color="steelblue")
                ax.axvline(np.median(vals), color="red", linestyle="--")
                ax.set_xlabel(param)

            plt.tight_layout()
            plot_save_path = os.path.join(output_dir, "marginalized_posteriors.png")
            plt.savefig(plot_save_path)
            print(f"\nCustom analysis plot saved to: {plot_save_path}")
            plt.show()

        # BIC
        try:
            bic_info = compute_bic_from_results(results)
            print("\n" + "=" * 50)
            print("BIC (Bayesian Information Criterion)")
            print("=" * 50)
            print(f"  BIC = {bic_info['bic']:.2f}")
            print(f"  n_params = {bic_info['n_params']}")
            print(f"  n_pixels = {bic_info['n_pixels']}")
            print(f"  log(L_max) = {bic_info['log_L_max']:.2f}")
            print(f"  shapelets_n_max = {bic_info['shapelets_n_max']}")
            print("\nNote: Lower BIC is better for model comparison.")
        except Exception as e:
            print(f"\nCould not compute BIC: {e}")


if __name__ == "__main__":
    main()
