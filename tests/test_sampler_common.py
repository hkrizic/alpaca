"""Tests for alpaca.sampler constants, priors, losses, and utils."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# These modules require JAX, scipy, numpy -- skip if unavailable.
np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
jax = pytest.importorskip("jax")


class TestConstants:
    """Verify physical constants are exported."""

    def test_constants_exist(self):
        from alpaca.sampler.constants import C_KM_S, D_DT_MIN, D_DT_MAX

        assert C_KM_S == pytest.approx(299792.458)
        assert D_DT_MIN == 500.0
        assert D_DT_MAX == 10000.0


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
        from alpaca.sampler.priors import tnorm

        rv = tnorm(mu=0.0, sigma=1.0, lo=-2.0, hi=2.0)
        # Should return a frozen scipy distribution with ppf callable
        assert callable(rv.ppf)
        assert callable(rv.pdf)

    def test_samples_within_bounds(self):
        from alpaca.sampler.priors import tnorm

        rv = tnorm(mu=5.0, sigma=1.0, lo=3.0, hi=7.0)
        samples = rv.rvs(size=1000)
        assert np.all(samples >= 3.0)
        assert np.all(samples <= 7.0)


class TestBoundedPrior:
    """Test _bounded_prior utility."""

    def test_basic_call(self):
        from alpaca.sampler.priors import _bounded_prior

        rv = _bounded_prior(mu=1.0, sigma=0.5, lo=0.0, hi=2.0)
        assert callable(rv.ppf)


class TestVectorToParamdict:
    """Test _vector_to_paramdict."""

    def test_basic_mapping(self):
        from alpaca.sampler.utils import _vector_to_paramdict

        names = ["a", "b", "c"]
        vec = np.array([1.0, 2.0, 3.0])
        result = _vector_to_paramdict(vec, names)
        assert isinstance(result, dict)
        assert result["a"] == 1.0
        assert result["b"] == 2.0
        assert result["c"] == 3.0


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
