"""
alpaca.pipeline

Unified lens modeling pipeline combining:
1. Iterative PSF reconstruction (STARRED)
2. Multi-start optimization (MAP estimation)
3. Posterior sampling (NUTS/NumPyro, Nautilus)

All outputs are organized in structured folders with comprehensive plotting.
"""

from alpaca.pipeline.runner import load_pipeline_results, quick_pipeline, run_pipeline
from alpaca.plotting.diagnostics import (
    plot_chain_diagnostics,
    plot_multistart_summary,
    plot_psf_comparison,
)

# Re-export plotting utilities for convenience
from alpaca.plotting.model_plots import plot_model_summary_custom, plot_ray_tracing_check
from alpaca.plotting.posterior_plots import plot_corner_posterior

__all__ = [
    # Main functions
    "run_pipeline",
    "quick_pipeline",
    "load_pipeline_results",
    # Plotting utilities (re-exported for convenience)
    "plot_psf_comparison",
    "plot_model_summary_custom",
    "plot_multistart_summary",
    "plot_ray_tracing_check",
    "plot_corner_posterior",
    "plot_chain_diagnostics",
]
