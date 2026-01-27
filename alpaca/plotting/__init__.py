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

from alpaca.plotting.benchmarking import (
    load_multistart_summary,
    load_nautilus_timing_logs,
    plot_multistart_all,
    plot_multistart_best_trace,
    plot_multistart_losses_and_chi2,
    plot_multistart_timing_breakdown,
    plot_nautilus_scaling,
    print_multistart_summary,
    print_nautilus_summary,
    summarise_multistart,
    summarise_nautilus,
)
from alpaca.plotting.diagnostics import (
    plot_chain_diagnostics,
    plot_multistart_history,
    plot_multistart_summary,
    plot_nuts_diagnostics,
    plot_psf_comparison,
)
from alpaca.plotting.model_plots import (
    nautilus_mean_model_plot,
    plot_bestfit_model,
    plot_model_summary_custom,
    plot_ray_tracing_check,
    pso_best_model_plot,
)
from alpaca.plotting.posterior_plots import (
    nautilus_corner_plot,
    plot_corner_posterior,
    plot_posterior_ray_tracing,
    pso_corner_plot,
)

__all__ = [
    # model_plots
    "plot_bestfit_model",
    "nautilus_mean_model_plot",
    "pso_best_model_plot",
    "plot_model_summary_custom",
    "plot_ray_tracing_check",
    # posterior_plots
    "nautilus_corner_plot",
    "pso_corner_plot",
    "plot_posterior_ray_tracing",
    "plot_corner_posterior",
    # diagnostics
    "plot_multistart_history",
    "plot_multistart_summary",
    "plot_chain_diagnostics",
    "plot_psf_comparison",
    "plot_nuts_diagnostics",
    # benchmarking
    "load_multistart_summary",
    "summarise_multistart",
    "print_multistart_summary",
    "plot_multistart_best_trace",
    "plot_multistart_losses_and_chi2",
    "plot_multistart_timing_breakdown",
    "plot_multistart_all",
    "load_nautilus_timing_logs",
    "summarise_nautilus",
    "print_nautilus_summary",
    "plot_nautilus_scaling",
]
