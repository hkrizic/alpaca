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
            plot_model_summary_custom,
            plot_ray_tracing_check,
        )
        assert callable(plot_model_summary_custom)
        assert callable(plot_ray_tracing_check)



class TestImportsPosteriorPlots:
    """Verify posterior_plots functions are importable."""

    def test_imports(self):
        from alpaca.plotting import (
            plot_posterior_ray_tracing,
            plot_corner_posterior,
        )
        assert callable(plot_posterior_ray_tracing)
        assert callable(plot_corner_posterior)



class TestImportsDiagnostics:
    """Verify diagnostics functions are importable."""

    def test_imports(self):
        from alpaca.plotting import (
            plot_multistart_summary,
            plot_chain_diagnostics,
            plot_psf_comparison,
            plot_nuts_diagnostics,
        )
        assert callable(plot_multistart_summary)
        assert callable(plot_chain_diagnostics)
        assert callable(plot_psf_comparison)
        assert callable(plot_nuts_diagnostics)



class TestNoBenchmarking:
    """Ensure benchmarking module has been removed."""

    def test_no_benchmarking_exports(self):
        import alpaca.plotting as plotting_mod
        assert not hasattr(plotting_mod, "load_multistart_summary")
        assert not hasattr(plotting_mod, "summarise_multistart")
        assert not hasattr(plotting_mod, "print_multistart_summary")
        assert not hasattr(plotting_mod, "plot_multistart_best_trace")
        assert not hasattr(plotting_mod, "plot_nautilus_scaling")


class TestNoEmcee:
    """Ensure no emcee-related functions exist."""

    def test_no_emcee_in_all(self):
        from alpaca.plotting import __all__

        for name in __all__:
            assert "emcee" not in name.lower()
