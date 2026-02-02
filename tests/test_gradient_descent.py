"""Tests for alpaca.sampler.gradient_descent module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("jax")


class TestImports:
    """Verify all public functions are importable from the gradient descent module."""

    def test_import_optimizer_functions(self):
        from alpaca.sampler.gradient_descent import (
            make_safe_loss,
            run_gradient_descent,
            load_multistart_summary,
        )
        assert callable(make_safe_loss)
        assert callable(run_gradient_descent)
        assert callable(load_multistart_summary)

    def test_import_bic_functions(self):
        from alpaca.utils.bic import compute_bic, compute_bic_from_results
        assert callable(compute_bic)
        assert callable(compute_bic_from_results)


class TestComputeBic:
    """Test BIC computation with synthetic data."""

    def test_compute_bic_basic(self):
        import numpy as np
        from alpaca.utils.bic import compute_bic

        # compute_bic expects a posterior dict with 'log_likelihood' and 'param_names'
        posterior = {
            "log_likelihood": np.array([-50.0, -48.0, -52.0]),
            "param_names": ["a", "b"],
        }
        bic = compute_bic(posterior, n_pixels=100, n_params=2)
        assert isinstance(bic, float)
        # BIC = k * ln(N) - 2 * max(loglike)
        # = 2 * ln(100) - 2 * (-48) = 2*4.605... + 96 = ~105.21
        expected = 2 * np.log(100) - 2 * (-48.0)
        assert bic == pytest.approx(expected, rel=1e-5)
