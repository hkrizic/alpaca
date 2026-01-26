"""
Test ALPACA likelihood functions.
"""

import pytest
import numpy as np


class TestImagingLikelihood:
    """Test imaging likelihood functions."""

    def test_imaging_loglike_basic(self, small_image, small_noise_map):
        """Test basic imaging log-likelihood computation."""
        from alpaca.likelihood import imaging_loglike

        # Model that matches data exactly
        model = small_image.copy()
        ll = imaging_loglike(small_image, model, small_noise_map)

        # Perfect match should give high likelihood (chi2 ~ 0)
        assert np.isfinite(ll)

    def test_imaging_loglike_mismatch(self, small_image, small_noise_map):
        """Test log-likelihood decreases with model mismatch."""
        from alpaca.likelihood import imaging_loglike

        model_good = small_image.copy()
        model_bad = small_image + 1.0  # Offset model

        ll_good = imaging_loglike(small_image, model_good, small_noise_map)
        ll_bad = imaging_loglike(small_image, model_bad, small_noise_map)

        # Good model should have higher likelihood
        assert ll_good > ll_bad

    def test_imaging_loglike_noise_scaling(self, small_image):
        """Test log-likelihood scales correctly with noise."""
        from alpaca.likelihood import imaging_loglike

        model = small_image + 0.5  # Some mismatch

        noise_low = np.ones_like(small_image) * 0.1
        noise_high = np.ones_like(small_image) * 1.0

        ll_low_noise = imaging_loglike(small_image, model, noise_low)
        ll_high_noise = imaging_loglike(small_image, model, noise_high)

        # Higher noise = higher tolerance = higher likelihood for same mismatch
        assert ll_high_noise > ll_low_noise

    def test_imaging_loglike_shape_mismatch(self, small_image, small_noise_map):
        """Test error on shape mismatch."""
        from alpaca.likelihood import imaging_loglike

        model_wrong_shape = np.zeros((16, 16))

        with pytest.raises((ValueError, Exception)):
            imaging_loglike(small_image, model_wrong_shape, small_noise_map)


class TestTimeDelayLikelihood:
    """Test time delay likelihood functions."""

    def test_time_delay_loglike_import(self):
        """Test time delay log-likelihood can be imported."""
        from alpaca.likelihood import time_delay_loglike
        assert callable(time_delay_loglike)


class TestRayshootLikelihood:
    """Test ray shooting consistency likelihood."""

    def test_rayshoot_loglike_import(self):
        """Test rayshoot consistency log-likelihood can be imported."""
        from alpaca.likelihood import rayshoot_consistency_loglike
        assert callable(rayshoot_consistency_loglike)


class TestChi2Computation:
    """Test chi-squared computation utilities."""

    def test_reduced_chi2(self, small_image, small_noise_map):
        """Test reduced chi2 computation."""
        from alpaca.likelihood import imaging_loglike

        model = small_image.copy()
        ll = imaging_loglike(small_image, model, small_noise_map)

        # For perfect match, chi2 should be near 0
        # log-likelihood is -0.5 * chi2, so chi2 = -2 * ll
        chi2 = -2.0 * ll
        n_pix = small_image.size
        chi2_red = chi2 / n_pix

        # For perfect match chi2_red should be near 0
        assert chi2_red < 1.0
