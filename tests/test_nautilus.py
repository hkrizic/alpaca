"""Tests for alpaca.sampler.nautilus module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("jax")


class TestImports:
    """Verify all public functions are importable."""

    def test_import_prior(self):
        from alpaca.sampler.nautilus import (
            build_nautilus_prior,
            build_nautilus_prior_and_loglike,
        )
        assert callable(build_nautilus_prior)
        assert callable(build_nautilus_prior_and_loglike)

    def test_import_sampler(self):
        from alpaca.sampler.nautilus import run_nautilus, load_posterior_from_checkpoint
        assert callable(run_nautilus)
        assert callable(load_posterior_from_checkpoint)

    def test_import_posterior(self):
        from alpaca.sampler.nautilus import get_nautilus_posterior
        assert callable(get_nautilus_posterior)
