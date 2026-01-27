"""Tests for alpaca.plotting module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")


class TestImportsModelPlots:
    """Verify model_plots functions are importable."""

    def test_imports(self):
        from alpaca.plotting import (
            plot_bestfit_model,
            nautilus_mean_model_plot,
            pso_best_model_plot,
            plot_model_summary_custom,
            plot_ray_tracing_check,
        )
        assert callable(plot_bestfit_model)
        assert callable(nautilus_mean_model_plot)
        assert callable(pso_best_model_plot)
        assert callable(plot_model_summary_custom)
        assert callable(plot_ray_tracing_check)


class TestImportsPosteriorPlots:
    """Verify posterior_plots functions are importable."""

    def test_imports(self):
        from alpaca.plotting import (
            nautilus_corner_plot,
            pso_corner_plot,
            plot_posterior_ray_tracing,
            plot_corner_posterior,
        )
        assert callable(nautilus_corner_plot)
        assert callable(pso_corner_plot)
        assert callable(plot_posterior_ray_tracing)
        assert callable(plot_corner_posterior)


class TestImportsDiagnostics:
    """Verify diagnostics functions are importable."""

    def test_imports(self):
        from alpaca.plotting import (
            plot_multistart_history,
            plot_multistart_summary,
            plot_chain_diagnostics,
            plot_psf_comparison,
            plot_nuts_diagnostics,
        )
        assert callable(plot_multistart_history)
        assert callable(plot_multistart_summary)
        assert callable(plot_chain_diagnostics)
        assert callable(plot_psf_comparison)
        assert callable(plot_nuts_diagnostics)


class TestImportsBenchmarking:
    """Verify benchmarking functions are importable."""

    def test_imports(self):
        from alpaca.plotting import (
            load_multistart_summary,
            summarise_multistart,
            print_multistart_summary,
            plot_multistart_best_trace,
            plot_multistart_losses_and_chi2,
            plot_multistart_timing_breakdown,
            plot_multistart_all,
            load_nautilus_timing_logs,
            summarise_nautilus,
            print_nautilus_summary,
            plot_nautilus_scaling,
        )
        assert callable(load_multistart_summary)
        assert callable(summarise_multistart)
        assert callable(print_multistart_summary)
        assert callable(plot_multistart_best_trace)
        assert callable(plot_multistart_losses_and_chi2)
        assert callable(plot_multistart_timing_breakdown)
        assert callable(plot_multistart_all)
        assert callable(load_nautilus_timing_logs)
        assert callable(summarise_nautilus)
        assert callable(print_nautilus_summary)
        assert callable(plot_nautilus_scaling)


class TestNoEmcee:
    """Ensure no emcee-related functions exist."""

    def test_no_emcee_in_all(self):
        from alpaca.plotting import __all__

        for name in __all__:
            assert "emcee" not in name.lower()
