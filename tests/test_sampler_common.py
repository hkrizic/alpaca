"""Tests for alpaca.sampler constants, priors, losses, and utils."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# These modules require JAX, scipy, numpy -- skip if unavailable.
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
jax = pytest.importorskip("jax")


class TestEnvironmentInfo:
    """Test get_environment_info utility."""

    def test_returns_dict(self):
        from alpaca.sampler.utils import get_environment_info

        info = get_environment_info()
        assert isinstance(info, dict)
        assert "python_version" in info
        assert "platform" in info
        assert "pid" in info

    def test_has_cpu_count(self):
        from alpaca.sampler.utils import get_environment_info

        info = get_environment_info()
        assert "cpu_count" in info
        assert isinstance(info["cpu_count"], (int, type(None)))


class TestNow:
    """Test now() timer utility."""

    def test_returns_float(self):
        from alpaca.sampler.utils import now

        t = now()
        assert isinstance(t, float)
        assert t > 0


class TestTnorm:
    """Test truncated normal distribution builder."""

    def test_basic_call(self):
        from alpaca.sampler.nautilus.prior_utils import tnorm

        rv = tnorm(mu=0.0, sigma=1.0, lo=-2.0, hi=2.0)
        # Should return a frozen scipy distribution with ppf callable
        assert callable(rv.ppf)
        assert callable(rv.pdf)

    def test_samples_within_bounds(self):
        from alpaca.sampler.nautilus.prior_utils import tnorm

        rv = tnorm(mu=5.0, sigma=1.0, lo=3.0, hi=7.0)
        samples = rv.rvs(size=1000)
        assert np.all(samples >= 3.0)
        assert np.all(samples <= 7.0)


class TestBoundedPrior:
    """Test _bounded_prior utility."""

    def test_basic_call(self):
        from alpaca.sampler.nautilus.prior_utils import _bounded_prior

        rv = _bounded_prior(mu=1.0, sigma=0.5, lo=0.0, hi=2.0)
        assert callable(rv.ppf)


class TestGetRng:
    """Test _get_rng random number generator utility."""

    def test_with_seed(self):
        from alpaca.sampler.utils import _get_rng

        rng = _get_rng(42)
        assert isinstance(rng, np.random.RandomState)

    def test_without_seed(self):
        from alpaca.sampler.utils import _get_rng

        rng = _get_rng(None)
        # When seed is None, returns np.random module itself
        assert rng is np.random
