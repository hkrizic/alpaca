"""
Benchmarking & plotting helpers for lens modeling inference runs.

This module knows how to:

- Read and analyse multi-start optimisation summaries written by
  `run_gradient_descent`.

- Read the NAUTILUS timing JSON logs created by `run_nautilus`
  (stored under `logs/benchmark_timing` next to the checkpoint).

The goal is to extract as much information as possible about:

- Wall-clock times.
- Number of likelihood / log-prob calls.
- Scaling with n_live (NAUTILUS).
- CPU / JAX device environment (if present in the summaries/logs).

Note: EMCEE-related benchmarking functions have been removed.
"""

import glob
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _ensure_ax(ax=None):
    """Return (fig, ax) such that ax is always a valid matplotlib Axes."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _maybe_show_and_save(fig, show: bool = True, savepath: str | None = None):
    """Common logic for show/save."""
    if savepath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------
# MULTI-START BENCHMARKING
# ----------------------------------------------------------------------


def load_multistart_summary(outdir: str, verbose: bool = True) -> dict[str, Any]:
    """
    Load a multi-start summary (and best-fit parameters) from disk.

    Load a multi-start summary and best-fit parameters from disk.

    Parameters
    ----------
    outdir : str
        Directory containing `multi_start_summary.json` and
        `best_fit_params_runXX.json`.

    Returns
    -------
    summary : dict
        Summary with 'best_params_json' filled in if available.
    """
    outdir = os.path.abspath(outdir)
    summary_path = os.path.join(outdir, "multi_start_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No multi-start summary found at {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)

    best_run = summary.get("best_run", 0)
    best_params_path = os.path.join(outdir, f"best_fit_params_run{best_run:02d}.json")
    if os.path.exists(best_params_path):
        with open(best_params_path) as f:
            best_params = json.load(f)
        summary["best_params_json"] = best_params

    if verbose:
        print(f"[multistart] Loaded summary from: {summary_path}")
        if "best_params_json" in summary:
            print(f"[multistart] Loaded best-fit parameters from: {best_params_path}")

    return summary


def summarise_multistart(summary: dict[str, Any]) -> dict[str, Any]:
    """
    Compute derived statistics from a multi-start summary.

    Returns
    -------
    stats : dict with keys like:
        - n_starts
        - n_param
        - best_loss
        - median_final_loss
        - mean_final_loss
        - final_loss_std
        - chi2_red_median / chi2_red_mean / chi2_red_std
        - total_runtime (if present)
        - env (if present)
    """
    results = summary.get("results", [])
    n_starts = summary.get("n_starts", len(results))
    n_param = summary.get("n_param", None)

    if "final_losses" in summary:
        finals = np.asarray(summary["final_losses"], dtype=float)
    else:
        finals = np.asarray([r["final_loss"] for r in results], dtype=float)

    if "chi2_reds" in summary:
        chi2 = np.asarray(summary["chi2_reds"], dtype=float)
    else:
        chi2 = np.asarray(
            [r.get("chi2_red", np.nan) for r in results], dtype=float
        )

    timing = summary.get("timing", {})
    total_runtime = timing.get("total", None)

    stats = dict(
        n_starts=int(n_starts),
        n_param=int(n_param) if n_param is not None else None,
        best_loss=float(summary.get("best_loss", np.min(finals))),
        median_final_loss=float(np.nanmedian(finals)),
        mean_final_loss=float(np.nanmean(finals)),
        final_loss_std=float(np.nanstd(finals)),
        median_chi2_red=float(np.nanmedian(chi2)) if chi2.size else None,
        mean_chi2_red=float(np.nanmean(chi2)) if chi2.size else None,
        chi2_red_std=float(np.nanstd(chi2)) if chi2.size else None,
        total_runtime=float(total_runtime) if total_runtime is not None else None,
        env=summary.get("env", None),
    )
    return stats


def print_multistart_summary(summary: dict[str, Any]) -> None:
    """Pretty-print multi-start benchmark statistics."""
    stats = summarise_multistart(summary)
    print("\n=== Multi-start optimisation summary ===")
    print(f"  #starts           : {stats['n_starts']}")
    if stats["n_param"] is not None:
        print(f"  #parameters       : {stats['n_param']}")
    print(f"  best loss         : {stats['best_loss']:.6g}")
    print(
        f"  final loss (median/mean ± std): "
        f"{stats['median_final_loss']:.6g} / "
        f"{stats['mean_final_loss']:.6g} ± {stats['final_loss_std']:.3g}"
    )
    if stats["median_chi2_red"] is not None:
        print(
            f"  chi2_red (median/mean ± std): "
            f"{stats['median_chi2_red']:.3g} / "
            f"{stats['mean_chi2_red']:.3g} ± {stats['chi2_red_std']:.3g}"
        )
    if stats["total_runtime"] is not None:
        print(f"  total runtime     : {stats['total_runtime']:.3f} s")

    env = stats["env"]
    if env is not None:
        print("\n  Environment:")
        print(f"    python          : {env.get('python_version', 'unknown')}")
        print(f"    platform        : {env.get('platform', 'unknown')}")
        print(f"    cpu_count       : {env.get('cpu_count', 'unknown')}")
        print(f"    jax devices     : {env.get('jax_device_count', 'unknown')} "
              f"({', '.join(env.get('jax_platforms', []) or [])})")
        for k, v in env.items():
            if k.startswith("env_"):
                print(f"    {k} = {v}")
        print("========================================\n")


def plot_multistart_best_trace(
    summary: dict[str, Any],
    ax=None,
    show: bool = True,
    savepath: str | None = None,
):
    """Plot best-so-far loss as a function of run index."""
    best_trace = np.asarray(summary.get("best_trace", []), dtype=float)
    if best_trace.size == 0:
        raise ValueError("No 'best_trace' found in multi-start summary.")

    fig, ax = _ensure_ax(ax)
    ax.plot(np.arange(len(best_trace)), best_trace, marker="o")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Best-so-far loss")
    ax.set_title("Multi-start: best loss after each run")
    ax.grid(True, alpha=0.3)

    _maybe_show_and_save(fig, show=show, savepath=savepath)
    return fig, ax


def plot_multistart_losses_and_chi2(
    summary: dict[str, Any],
    ax=None,
    show: bool = True,
    savepath: str | None = None,
):
    """
    Scatter plot: final loss and reduced chi^2 vs run index.
    """
    results = summary.get("results", [])
    if not results:
        raise ValueError("No 'results' list in multi-start summary.")

    runs = np.asarray([r["run"] for r in results], dtype=int)
    fin = np.asarray([r["final_loss"] for r in results], dtype=float)
    chi2 = np.asarray([r.get("chi2_red", np.nan) for r in results], dtype=float)

    fig, ax = _ensure_ax(ax)
    ax.scatter(runs, fin, label="final loss", marker="o")
    if np.isfinite(chi2).any():
        ax.scatter(runs, chi2, label="reduced chi²", marker="s")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Value")
    ax.set_title("Multi-start: final loss and χ²_red per run")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _maybe_show_and_save(fig, show=show, savepath=savepath)
    return fig, ax


def plot_multistart_timing_breakdown(
    summary: dict[str, Any],
    ax=None,
    show: bool = True,
    savepath: str | None = None,
):
    """
    Stacked bar plot of timing per run (if available):
    sampling/unconstrain, Adam, L-BFGS-B, I/O.
    """
    results = summary.get("results", [])
    if not results:
        raise ValueError("No 'results' list in multi-start summary.")

    runs = np.asarray([r["run"] for r in results], dtype=int)

    t_sample = np.asarray(
        [r.get("t_sample_unconstrain", 0.0) for r in results], dtype=float
    )
    t_adam = np.asarray([r.get("t_adam", 0.0) for r in results], dtype=float)
    t_lbfgs = np.asarray([r.get("t_lbfgs", 0.0) for r in results], dtype=float)
    t_io = np.asarray([r.get("t_io", 0.0) for r in results], dtype=float)

    fig, ax = _ensure_ax(ax)
    width = 0.8
    ax.bar(runs, t_sample, width, label="sample+unconstrain")
    ax.bar(runs, t_adam, width, bottom=t_sample, label="Adam preopt")
    ax.bar(runs, t_lbfgs, width, bottom=t_sample + t_adam, label="L-BFGS-B")
    ax.bar(
        runs,
        t_io,
        width,
        bottom=t_sample + t_adam + t_lbfgs,
        label="I/O",
    )

    ax.set_xlabel("Run index")
    ax.set_ylabel("Time per run [s]")
    ax.set_title("Multi-start timing breakdown per run")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    _maybe_show_and_save(fig, show=show, savepath=savepath)
    return fig, ax


def plot_multistart_all(
    summary: dict[str, Any],
    outdir: str | None = None,
    show: bool = True,
):
    """
    Convenience function: produce three standard plots for a multi-start run.

    - best trace vs run index
    - final loss & chi2 vs run index
    - per-run timing breakdown

    If outdir is given, PNG files are saved there.
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        p1 = os.path.join(outdir, "multistart_best_trace.png")
        p2 = os.path.join(outdir, "multistart_losses_chi2.png")
        p3 = os.path.join(outdir, "multistart_timing.png")
    else:
        p1 = p2 = p3 = None

    plot_multistart_best_trace(summary, show=show, savepath=p1)
    plot_multistart_losses_and_chi2(summary, show=show, savepath=p2)
    plot_multistart_timing_breakdown(summary, show=show, savepath=p3)


# ----------------------------------------------------------------------
# NAUTILUS BENCHMARKING (nested sampling)
# ----------------------------------------------------------------------


def load_nautilus_timing_logs(
    path: str,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Load NAUTILUS timing JSON logs.

    Parameters
    ----------
    path : str
        Either:
          - a NAUTILUS checkpoint file (HDF5), or
          - the 'logs/benchmark_timing' directory itself.

    Returns
    -------
    records : list[dict]
        One dict per timing JSON, with all keys in the "timing" section,
        plus:
            - "checkpoint"
            - "timestamp"
            - "log_path"
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        base_dir = p.parent
        ckpt_name = p.stem
        log_dir = base_dir / "logs" / "benchmark_timing"
        prefix = f"timing_{ckpt_name}_"
    else:
        # assume path is a directory containing timing_*.json files
        log_dir = p
        prefix = "timing_"

    if not log_dir.is_dir():
        raise FileNotFoundError(f"No timing log directory found at {log_dir}")

    pattern = str(log_dir / f"{prefix}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No timing JSON files matching {pattern}")

    records: list[dict[str, Any]] = []
    for fpath in files:
        with open(fpath) as f:
            payload = json.load(f)

        timing = payload.get("timing", payload)
        rec = dict(timing)
        rec["checkpoint"] = payload.get("checkpoint", None)
        rec["timestamp"] = payload.get("timestamp", None)
        rec["log_path"] = os.path.abspath(fpath)
        records.append(rec)

    # Sort by timestamp if available
    def _key(rec):
        ts = rec.get("timestamp")
        return ts or rec["log_path"]

    records.sort(key=_key)

    if verbose:
        print(f"[nautilus] Loaded {len(records)} timing logs from {log_dir}")
    return records


def summarise_nautilus(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary statistics across multiple NAUTILUS runs.

    Returns
    -------
    stats : dict with keys like:
        - n_runs
        - median_total_runtime
        - median_n_loglike_calls
        - median_loglike_time
        - median_overhead_fraction
        - n_live values, if present
    """
    if not records:
        raise ValueError("Empty records list.")

    total = np.asarray(
        [r.get("total_runtime", np.nan) for r in records], dtype=float
    )
    n_calls = np.asarray(
        [r.get("n_loglike_calls", np.nan) for r in records], dtype=float
    )
    log_tot = np.asarray(
        [r.get("loglike_total", np.nan) for r in records], dtype=float
    )
    overhead = np.asarray(
        [r.get("sampler_overhead", np.nan) for r in records], dtype=float
    )
    n_live = np.asarray(
        [r.get("n_live", np.nan) for r in records], dtype=float
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        overhead_frac = overhead / total

    stats = dict(
        n_runs=len(records),
        median_total_runtime=float(np.nanmedian(total)),
        median_n_loglike_calls=float(np.nanmedian(n_calls)),
        median_loglike_time=float(np.nanmedian(log_tot)),
        median_overhead_fraction=float(np.nanmedian(overhead_frac)),
        n_live_values=n_live.tolist(),
    )
    return stats


def print_nautilus_summary(records: list[dict[str, Any]]) -> None:
    """Pretty-print summary of NAUTILUS timing logs."""
    stats = summarise_nautilus(records)
    print("\n=== NAUTILUS timing summary ===")
    print(f"  #runs          : {stats['n_runs']}")
    print(f"  total runtime  : median {stats['median_total_runtime']:.3f} s")
    print(f"  n_loglike_calls: median {stats['median_n_loglike_calls']:.3g}")
    print(f"  loglike time   : median {stats['median_loglike_time']:.3f} s")
    print(
        f"  overhead frac  : median {stats['median_overhead_fraction']:.3f} "
        "(sampler overhead / total runtime)"
    )

    # If environment keys are present (from new run_nautilus), print them for
    # the latest record as a representative environment.
    env_keys = [
        "python_version",
        "platform",
        "cpu_count",
        "jax_device_count",
        "jax_platforms",
    ]
    latest = records[-1]
    if any(k in latest for k in env_keys):
        print("\n  Environment (from latest run):")
        for k in env_keys:
            if k in latest:
                print(f"    {k}: {latest[k]}")
        for k, v in latest.items():
            if str(k).startswith("env_"):
                print(f"    {k} = {v}")
    print("================================\n")


def _extract_array(records: list[dict[str, Any]], key: str) -> np.ndarray:
    """Helper: get float array from a list of dicts."""
    return np.asarray([r.get(key, np.nan) for r in records], dtype=float)


def plot_nautilus_scaling(
    records: list[dict[str, Any]],
    show: bool = True,
    outdir: str | None = None,
):
    """
    Convenience function: produce several scaling plots for NAUTILUS runs:

    - total runtime vs n_live
    - n_loglike_calls vs n_live
    - average loglike time vs n_live
    - overhead fraction vs n_live
    """
    n_live = _extract_array(records, "n_live")
    total = _extract_array(records, "total_runtime")
    n_calls = _extract_array(records, "n_loglike_calls")
    log_tot = _extract_array(records, "loglike_total")
    overhead = _extract_array(records, "sampler_overhead")

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_call = log_tot / n_calls
        overhead_frac = overhead / total

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # 1) runtime vs n_live
    fig1, ax1 = plt.subplots()
    ax1.scatter(n_live, total)
    ax1.set_xlabel("n_live")
    ax1.set_ylabel("Total runtime [s]")
    ax1.set_title("NAUTILUS: total runtime vs n_live")
    ax1.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig1,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_runtime_vs_nlive.png")
                  if outdir is not None else None),
    )

    # 2) n_loglike_calls vs n_live
    fig2, ax2 = plt.subplots()
    ax2.scatter(n_live, n_calls)
    ax2.set_xlabel("n_live")
    ax2.set_ylabel("n_loglike_calls")
    ax2.set_title("NAUTILUS: likelihood calls vs n_live")
    ax2.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig2,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_calls_vs_nlive.png")
                  if outdir is not None else None),
    )

    # 3) avg loglike time vs n_live
    fig3, ax3 = plt.subplots()
    ax3.scatter(n_live, avg_call)
    ax3.set_xlabel("n_live")
    ax3.set_ylabel("\u27E8time per loglike call\u27E9 [s]")
    ax3.set_title("NAUTILUS: mean loglike time vs n_live")
    ax3.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig3,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_avg_loglike_time_vs_nlive.png")
                  if outdir is not None else None),
    )

    # 4) overhead fraction vs n_live
    fig4, ax4 = plt.subplots()
    ax4.scatter(n_live, overhead_frac)
    ax4.set_xlabel("n_live")
    ax4.set_ylabel("Overhead / total runtime")
    ax4.set_title("NAUTILUS: sampler overhead fraction vs n_live")
    ax4.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig4,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_overhead_fraction_vs_nlive.png")
                  if outdir is not None else None),
    )


__all__ = [
    # multistart
    "load_multistart_summary",
    "summarise_multistart",
    "print_multistart_summary",
    "plot_multistart_best_trace",
    "plot_multistart_losses_and_chi2",
    "plot_multistart_timing_breakdown",
    "plot_multistart_all",
    # nautilus
    "load_nautilus_timing_logs",
    "summarise_nautilus",
    "print_nautilus_summary",
    "plot_nautilus_scaling",
]
