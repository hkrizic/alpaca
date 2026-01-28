"""
Plotting module for the Alpaca lens modeling package.

Provides visualization tools organized into four submodules:

- **model_plots**: Best-fit model visualization, data/model/residual comparisons,
  and ray-tracing checks.
- **posterior_plots**: Corner plots from Nautilus and PSO posteriors,
  and posterior ray-tracing convergence analysis.
- **diagnostics**: Multi-start optimization history, PSF comparison,
  chain diagnostics, and NUTS sampling diagnostics.
- **benchmarking**: Benchmarking summaries and scaling plots for
  multi-start optimization and NAUTILUS nested sampling.
"""

from alpaca.plotting.diagnostics import (
    plot_chain_diagnostics,
    plot_multistart_summary,
    plot_nuts_diagnostics,
    plot_psf_comparison,
)
from alpaca.plotting.model_plots import (
    plot_model_summary_custom,
    plot_ray_tracing_check,
)
from alpaca.plotting.posterior_plots import (
    plot_corner_posterior,
    plot_posterior_ray_tracing,
)

__all__ = [
    # model_plots
    "plot_model_summary_custom",
    "plot_ray_tracing_check",
    # posterior_plots
    "plot_posterior_ray_tracing",
    "plot_corner_posterior",
    # diagnostics
    "plot_multistart_summary",
    "plot_chain_diagnostics",
    "plot_psf_comparison",
    "plot_nuts_diagnostics",
]
