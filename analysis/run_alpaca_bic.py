"""
BIC Scan: Shapelet Order Selection

Runs gradient descent (MAP only) for a single TDC lens across multiple
shapelet orders n_max. Computes BIC for each, then plots BIC vs n_max
to identify the optimal source complexity.

Edit run_config_TDC.py to change configuration settings (N_MULTISTART, etc.).

Usage:
    python analysis/run_alpaca_bic.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is on sys.path when running from analysis/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpaca.pipeline import run_pipeline
from alpaca.utils.bic import compute_bic
from run_config_TDC import load_config, load_tdlmc_data, RUNG, CODE_ID, SEED

# =============================================================================
# USER SETTINGS
# =============================================================================
N_MAX_VALUES = [0, 4, 8, 12, 16, 20, 24, 28, 32]
OUTPUT_ROOT = os.path.join(REPO_ROOT, "results", "bic_scan")


def _save_progress(records):
    """Save summary JSON and plot after each completed n_max run."""
    # JSON
    summary_path = os.path.join(OUTPUT_ROOT, "bic_summary.json")
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)

    # Plot (need at least 1 point)
    nmax = np.array([r["n_max"] for r in records])
    bic_arr = np.array([r["bic"] for r in records])
    chi2r = np.array([r["chi2_red"] for r in records])
    best_idx = int(np.argmin(bic_arr))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel(r"$n_{\max}$ (shapelet order)", fontsize=13)
    ax1.set_ylabel("BIC", fontsize=13, color="tab:blue")
    ax1.plot(nmax, bic_arr, "o-", color="tab:blue", lw=2, ms=7, label="BIC")
    ax1.plot(nmax[best_idx], bic_arr[best_idx], "*", color="red", ms=18,
             zorder=5, label=f"min BIC (n_max={records[best_idx]['n_max']})")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left", fontsize=11)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$\chi^2_{\rm red}$", fontsize=13, color="tab:orange")
    ax2.plot(nmax, chi2r, "s--", color="tab:orange", lw=1.5, ms=6,
             label=r"$\chi^2_{\rm red}$")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.legend(loc="upper right", fontsize=11)

    ax1.set_xticks(N_MAX_VALUES)
    fig.suptitle(f"BIC Scan  (rung {RUNG}, code {CODE_ID}, seed {SEED})  "
                 f"[{len(records)}/{len(N_MAX_VALUES)} done]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    plot_path = os.path.join(OUTPUT_ROOT, "bic_vs_nmax.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("BIC SCAN: Shapelet Order Selection")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Load config and data (exactly like run_alpaca_tdc.py)
    # -------------------------------------------------------------------------
    config = load_config()
    img, psf_kernel, noise_map = load_tdlmc_data()

    # Override: gradient descent only
    config.run_psf_reconstruction = False
    config.run_sampling = False
    # Time delays require measured_delays/delay_errors to be passed to
    # run_pipeline(); disable for the BIC scan since the time-delay
    # likelihood is the same across n_max and doesn't affect the comparison.
    config.gradient_descent_config.use_time_delays = False

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Loop over n_max values
    # -------------------------------------------------------------------------
    records = []

    for n_max in N_MAX_VALUES:
        print(f"\n{'=' * 60}")
        print(f"  n_max = {n_max}")
        print(f"{'=' * 60}")

        config.output_dir = os.path.join(OUTPUT_ROOT, f"nmax_{n_max}")
        config.shapelets_n_max = n_max
        config.use_source_shapelets = (n_max > 0)

        results = run_pipeline(
            config=config,
            img=img,
            psf_kernel=psf_kernel,
            noise_map=noise_map,
            verbose=True,
        )

        # BIC from best-fit image chi-squared (use prob_model's boosted noise)
        best_params = results["best_params"]
        prob_model = results["setup"]["prob_model"]
        model_img = np.asarray(prob_model.model_image_from_params(best_params))
        data = np.asarray(prob_model.data)
        boosted_noise = np.asarray(prob_model.noise_map)
        chi2 = float(np.sum(((data - model_img) / boosted_noise) ** 2))

        n_pix = int(data.size)
        n_params = sum(
            len(v) if isinstance(v, (list, np.ndarray)) else 1
            for v in best_params.values()
        )
        log_L_max = -0.5 * chi2
        bic = float(compute_bic(
            {"log_likelihood": np.array([log_L_max])}, n_pix, n_params
        ))
        chi2_red = chi2 / max(n_pix - n_params, 1)
        best_loss = results.get("multistart_summary", {}).get("best_loss", float("nan"))

        records.append({
            "n_max": n_max,
            "n_params": n_params,
            "chi2": chi2,
            "chi2_red": chi2_red,
            "bic": bic,
            "best_loss": best_loss,
        })
        print(f"  => n_params={n_params}  chi2_red={chi2_red:.4f}  BIC={bic:.1f}")

        # Update JSON + plot after every run so progress is visible
        _save_progress(records)

    # -------------------------------------------------------------------------
    # 3. Final summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BIC SCAN SUMMARY")
    print("=" * 70)
    print(f"{'n_max':>5}  {'n_params':>8}  {'chi2_red':>10}  {'BIC':>14}  {'best_loss':>12}")
    print("-" * 55)
    for r in records:
        print(f"{r['n_max']:5d}  {r['n_params']:8d}  {r['chi2_red']:10.4f}  "
              f"{r['bic']:14.1f}  {r['best_loss']:12.4f}")

    best_idx = int(np.argmin([r["bic"] for r in records]))
    print(f"\nOptimal n_max = {records[best_idx]['n_max']}  "
          f"(BIC = {records[best_idx]['bic']:.1f})")
    print(f"\nResults in: {OUTPUT_ROOT}")
    print(f"  bic_summary.json  (updated after each run)")
    print(f"  bic_vs_nmax.png   (updated after each run)")


if __name__ == "__main__":
    main()
