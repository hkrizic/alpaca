"""
Test ALPACA PSF reconstruction module.
"""

import pytest
import numpy as np


class TestPSFImports:
    """Test PSF module imports."""

    def test_import_psf_module(self):
        """Test PSF module can be imported."""
        from alpaca import psf
        assert psf is not None

    def test_import_reconstruct_PSF(self):
        """Test reconstruct_PSF can be imported."""
        from alpaca.psf import reconstruct_PSF
        assert callable(reconstruct_PSF)

    def test_import_starred_flag(self):
        """Test STARRED availability flag."""
        from alpaca.psf import _HAS_STARRED
        assert isinstance(_HAS_STARRED, bool)

    def test_import_cutout_functions(self):
        """Test cutout functions can be imported."""
        from alpaca.psf import (
            build_centered_cutouts_from_isolated,
            build_centered_noise_cutouts,
        )
        assert callable(build_centered_cutouts_from_isolated)
        assert callable(build_centered_noise_cutouts)


class TestCutoutFunctions:
    """Test cutout extraction functions."""

    def test_build_centered_cutouts_basic(self):
        """Test basic centered cutout extraction."""
        from alpaca.psf import build_centered_cutouts_from_isolated

        # Create mock isolated images (4 point sources)
        n_ps = 4
        img_size = 64
        isolated_images = [
            np.random.randn(img_size, img_size) for _ in range(n_ps)
        ]

        # Point source positions (y, x) in pixel coordinates
        peaks_px = np.array([
            [20.5, 20.5],
            [20.5, 43.5],
            [43.5, 20.5],
            [43.5, 43.5],
        ])

        cutout_size = 21
        cutouts, masks = build_centered_cutouts_from_isolated(
            isolated_images,
            peaks_px,
            cutout_size,
            mask_other_peaks=True,
            mask_radius=5,
        )

        assert cutouts.shape == (n_ps, cutout_size, cutout_size)
        assert masks is not None
        assert masks.shape == (n_ps, cutout_size, cutout_size)

    def test_build_centered_cutouts_odd_size(self):
        """Test cutout size is forced to be odd."""
        from alpaca.psf import build_centered_cutouts_from_isolated

        isolated_images = [np.random.randn(64, 64)]
        peaks_px = np.array([[32.0, 32.0]])

        # Even cutout size should be made odd
        cutout_size = 20
        cutouts, _ = build_centered_cutouts_from_isolated(
            isolated_images,
            peaks_px,
            cutout_size,
            mask_other_peaks=False,
        )

        # Should be 21 (20 + 1)
        assert cutouts.shape[1] == 21
        assert cutouts.shape[2] == 21

    def test_build_centered_noise_cutouts(self):
        """Test noise cutout extraction."""
        from alpaca.psf import build_centered_noise_cutouts

        noise_map = np.ones((64, 64)) * 0.1
        peaks_px = np.array([
            [20.0, 20.0],
            [40.0, 40.0],
        ])

        cutout_size = 15
        sigma2_cutouts = build_centered_noise_cutouts(
            noise_map,
            peaks_px,
            cutout_size,
            noise_map_is_sigma=True,
        )

        assert sigma2_cutouts.shape == (2, cutout_size, cutout_size)
        # sigma^2 = 0.1^2 = 0.01
        assert np.allclose(sigma2_cutouts, 0.01, rtol=0.1)


class TestIsolatedImages:
    """Test isolated point source image generation."""

    def test_generate_isolated_ps_images_import(self):
        """Test generate_isolated_ps_images can be imported."""
        from alpaca.psf import generate_isolated_ps_images
        assert callable(generate_isolated_ps_images)


class TestReconstructPSF:
    """Test PSF reconstruction (basic tests without STARRED)."""

    def test_reconstruct_PSF_fallback(self, small_psf_kernel):
        """Test reconstruct_PSF returns current PSF when STARRED unavailable."""
        from alpaca.psf import reconstruct_PSF, _HAS_STARRED

        # If STARRED not available, should return current PSF
        if not _HAS_STARRED:
            isolated_images = [np.random.randn(64, 64) for _ in range(4)]
            peaks_px = np.array([[20, 20], [20, 40], [40, 20], [40, 40]])
            noise_map = np.ones((64, 64)) * 0.1

            result = reconstruct_PSF(
                current_psf=small_psf_kernel,
                peaks_px=peaks_px,
                noise_map=noise_map,
                isolated_images=isolated_images,
                verbose=False,
            )

            # Should return current PSF as fallback
            assert result.shape == small_psf_kernel.shape


def _starred_available():
    """Check if STARRED is available."""
    try:
        import starred
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _starred_available(), reason="STARRED not installed")
class TestReconstructPSFWithSTARRED:
    """Test PSF reconstruction with STARRED (only runs if STARRED installed)."""

    def test_reconstruct_psf_starred(self, small_psf_kernel):
        """Test PSF reconstruction with STARRED."""
        from alpaca.psf import reconstruct_PSF

        # Create synthetic isolated images with a known PSF
        n_ps = 4
        img_size = 99

        # Simple Gaussian PSF
        psf_size = small_psf_kernel.shape[0]

        isolated_images = []
        for _ in range(n_ps):
            img = np.zeros((img_size, img_size))
            cy, cx = img_size // 2, img_size // 2
            # Add PSF at center
            half = psf_size // 2
            img[cy-half:cy+half+1, cx-half:cx+half+1] = small_psf_kernel * 1000
            # Add noise
            img += np.random.randn(img_size, img_size) * 0.1
            isolated_images.append(img)

        peaks_px = np.array([
            [img_size // 2, img_size // 2],
        ] * n_ps)

        noise_map = np.ones((img_size, img_size)) * 0.1

        result = reconstruct_PSF(
            current_psf=small_psf_kernel,
            peaks_px=peaks_px,
            noise_map=noise_map,
            isolated_images=isolated_images,
            cutout_size=31,
            supersampling_factor=1,
            verbose=False,
        )

        # Result should be normalized
        assert np.isclose(result.sum(), 1.0, rtol=0.01)
        # Result should have reasonable shape
        assert result.shape == small_psf_kernel.shape
