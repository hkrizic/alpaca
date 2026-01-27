"""Tests for alpaca.sampler.nuts module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("jax")
pytest.importorskip("numpyro")


class TestImports:
    """Verify all public functions are importable."""

    def test_import_sampler(self):
        from alpaca.sampler.nuts import run_nuts_numpyro, load_nuts_samples
        assert callable(run_nuts_numpyro)
        assert callable(load_nuts_samples)

    def test_import_posterior(self):
        from alpaca.sampler.nuts import get_nuts_posterior
        assert callable(get_nuts_posterior)
