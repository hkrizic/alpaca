"""
Lens Modeling Pipeline Script (TDLMC)

Entry-point for TDLMC (Time Delay Lens Modeling Challenge) data.
Loads images and time delays from the TDC folder structure and runs
the complete lens modeling pipeline.

Pipeline phases:
    1. PSF Reconstruction - Iterative reconstruction using STARRED
    2. Gradient Descent Optimization - Two-phase MAP estimation
    3. Posterior Sampling - NUTS (NumPyro) or Nautilus

Supports multiple source models: Sersic, Shapelets, or Correlated Fields.
Edit run_config_TDC.py to change settings, then run this script.

For generic (non-TDC) usage, see run_alpaca.py instead.

Author: hkrizic
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from alpaca.pipeline import run_pipeline
from alpaca.sampler.gradient_descent import compute_bic_from_results
from run_config_TDC import load_config, load_tdlmc_data


def main():
    # =============================================================================
    # 1. TITLE AND INTRODUCTION
    # =============================================================================

    print("=" * 50)
    print("=" * 50)
    print("ALPACA Lens Modeling Pipeline")
    print("=" * 50)
    print("=" * 50)
    print("\nThis script runs the complete lens modeling pipeline:")
    print("  1. PSF Reconstruction")
    print("  2. Gradient Descent Optimization (MAP estimation)")
    print("  3. Posterior Sampling (NUTS or Nautilus)")
    print("\nEdit run_config_TDC.py to change configuration settings.")
    print("For generic (non-TDC) usage, see run_alpaca.py instead.")
    print("\nStarting pipeline...\n")

    # =============================================================================
    # 2. CONFIGURATION  (edit run_config.py, then call load_config())
    # =============================================================================
    config = load_config()

    # =============================================================================
    # 3. LOAD DATA
    # =============================================================================
    img, psf_kernel, noise_map = load_tdlmc_data()

    # =============================================================================
    # 4. RUN THE PIPELINE
    # =============================================================================
    results = run_pipeline(
        config=config,
        img=img,
        psf_kernel=psf_kernel,
        noise_map=noise_map,
        verbose=True,
    )

    # =============================================================================
    # 5. EXPLORE RESULTS
    # =============================================================================
    output_dir = results["output_dirs"]["root"]
    print(f"\nResults saved to: {output_dir}")
    print("\nDirectory structure:")
    for name, path in results["output_dirs"].items():
        rel_path = os.path.relpath(path, output_dir)
        print(f"  {name}: {rel_path}")

    # --- 5.1 PSF Reconstruction Results ---
    if "psf_result" in results and results["psf_result"] is not None:
        psf_result = results["psf_result"]
        print("\nPSF Reconstruction:")
        print(f"  Number of iterations: {psf_result['n_iterations']}")
    else:
        print("\nPSF reconstruction was not run.")

    # --- 5.2 Multi-start Optimization Results ---
    if "multistart_summary" in results and results["multistart_summary"] is not None:
        ms_summary = results["multistart_summary"]
        print("\nMulti-start Optimization:")
        print(f"  Number of starts: {ms_summary.get('n_starts', 'N/A')}")
        print(f"  Best run: {ms_summary.get('best_run', 'N/A')}")
        print(f"  Best loss: {ms_summary.get('best_loss', 'N/A'):.4f}")

        chi2_reds = ms_summary.get('chi2_reds', [])
        if len(chi2_reds) > 0:
            best_chi2 = chi2_reds[ms_summary.get('best_run', 0)]
            print(f"  Best chi2_red: {best_chi2:.4f}")

    # --- 5.3 Posterior Sampling Results ---
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

        # Report ray shooting systematic error if sampled
        if "log_sigma_rayshoot_sys" in param_names:
            idx = param_names.index("log_sigma_rayshoot_sys")
            log_vals = samples[:, idx]
            vals = np.exp(log_vals)  # Transform to linear space
            print(f"    sigma_rayshoot_sys: {np.mean(vals)*1000:.4f} +/- {np.std(vals)*1000:.4f} mas")
        elif "sigma_rayshoot_sys" in param_names:
            idx = param_names.index("sigma_rayshoot_sys")
            vals = samples[:, idx]
            print(f"    sigma_rayshoot_sys: {np.mean(vals)*1000:.4f} +/- {np.std(vals)*1000:.4f} mas")

        # --- 5.4 Marginalized Posteriors ---
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

        # --- 5.5 Compute and Report BIC ---
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
