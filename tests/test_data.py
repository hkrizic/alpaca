"""Tests for alpaca.data module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("numpy")
pytest.importorskip("astropy")


class TestImports:
    """Verify public API is importable."""

    def test_import_setup(self):
        from alpaca.data import setup_lens, make_pixel_grids
        assert callable(setup_lens)
        assert callable(make_pixel_grids)

    def test_no_tdlmc_functions(self):
        """Verify TDLMC-specific functions have been removed from the package."""
        import alpaca.data as data_mod
        assert not hasattr(data_mod, "tdlmc_paths")
        assert not hasattr(data_mod, "load_tdlmc_image")
        assert not hasattr(data_mod, "setup_tdlmc_lens")

    def test_import_detection(self):
        from alpaca.data import detect_ps_images_centered, make_plotter
        assert callable(detect_ps_images_centered)
        assert callable(make_plotter)

    def test_import_masks(self):
        from alpaca.data import (
            load_custom_arc_mask,
            make_source_arc_mask,
            save_arc_mask_visualization,
        )
        assert callable(load_custom_arc_mask)
        assert callable(make_source_arc_mask)
        assert callable(save_arc_mask_visualization)

    def test_import_noise(self):
        from alpaca.data import (
            auto_noise_boost_radius,
            boost_noise_around_point_sources,
        )
        assert callable(auto_noise_boost_radius)
        assert callable(boost_noise_around_point_sources)
