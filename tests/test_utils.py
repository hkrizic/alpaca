"""
Test ALPACA utility functions.
"""

import pytest
import numpy as np


class TestPixelGrids:
    """Test pixel grid creation utilities."""

    def test_make_pixel_grids_basic(self, small_image):
        """Test basic pixel grid creation."""
        from alpaca.utils import make_pixel_grids

        pix_scl = 0.08
        result = make_pixel_grids(small_image, pix_scl)

        # Should return 5 values: pixel_grid, ps_grid, xgrid, ygrid, pix_scl
        assert len(result) == 5
        pixel_grid, ps_grid, xgrid, ygrid, returned_pix_scl = result

        assert xgrid.shape == small_image.shape
        assert ygrid.shape == small_image.shape
        assert returned_pix_scl == pix_scl

    def test_make_pixel_grids_centered(self, small_image):
        """Test pixel grids are centered at origin."""
        from alpaca.utils import make_pixel_grids

        pix_scl = 0.08
        _, _, xgrid, ygrid, _ = make_pixel_grids(small_image, pix_scl)

        npix = small_image.shape[0]
        # Center should be near (0, 0)
        center_x = xgrid[npix // 2, npix // 2]
        center_y = ygrid[npix // 2, npix // 2]

        assert abs(center_x) < pix_scl
        assert abs(center_y) < pix_scl

    def test_make_pixel_grids_scale(self, small_image):
        """Test pixel grid scale is correct."""
        from alpaca.utils import make_pixel_grids

        pix_scl = 0.1
        _, _, xgrid, _, _ = make_pixel_grids(small_image, pix_scl)

        # Check spacing between pixels
        dx = np.abs(xgrid[0, 1] - xgrid[0, 0])
        assert np.isclose(dx, pix_scl, rtol=1e-5)


class TestPointSourceDetection:
    """Test point source detection utilities."""

    def test_detect_point_sources_import(self):
        """Test point source detection can be imported."""
        from alpaca.utils import detect_point_sources
        assert callable(detect_point_sources)

    def test_detect_point_sources_basic(self, rng):
        """Test basic point source detection."""
        from alpaca.utils import detect_point_sources

        # Create image with bright spots
        img = rng.normal(0, 0.1, (64, 64))
        # Add point sources at known locations (away from center which is masked)
        img[15, 15] = 10.0
        img[15, 48] = 8.0
        img[48, 15] = 6.0
        img[48, 48] = 4.0

        # Create coordinate grids
        pix_scl = 0.08
        npix = 64
        half = (npix - 1) / 2.0
        x1d = (np.arange(npix) - half) * pix_scl
        xgrid, ygrid = np.meshgrid(x1d, x1d)

        result = detect_point_sources(
            img, xgrid, ygrid,
            n_sources=4,
            lens_mask_radius=0.3,
            min_sep=0.1,
        )

        # Should return (peaks, x0s, y0s, peak_vals)
        assert len(result) == 4
        peaks, x0s, y0s, peak_vals = result
        assert len(x0s) == 4
        assert len(y0s) == 4
        assert len(peak_vals) == 4


class TestNoiseHandling:
    """Test noise handling utilities."""

    def test_boost_noise_import(self):
        """Test boost noise function can be imported."""
        from alpaca.utils import boost_noise_around_point_sources
        assert callable(boost_noise_around_point_sources)


class TestMaskUtilities:
    """Test mask creation utilities."""

    def test_make_arc_mask_import(self):
        """Test arc mask function can be imported."""
        from alpaca.utils import make_arc_mask
        assert callable(make_arc_mask)

    def test_make_arc_mask_basic(self):
        """Test basic arc mask creation."""
        from alpaca.utils import make_arc_mask

        npix = 64
        pix_scl = 0.08
        half = (npix - 1) / 2.0
        x1d = (np.arange(npix) - half) * pix_scl
        xgrid, ygrid = np.meshgrid(x1d, x1d)

        mask = make_arc_mask(
            xgrid, ygrid,
            inner_radius=0.3,
            outer_radius=1.5,
        )

        assert mask.shape == (npix, npix)
        # Mask should have some True values (inside annulus)
        assert np.sum(mask) > 0
