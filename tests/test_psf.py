"""Tests for alpaca.psf module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

np = pytest.importorskip("numpy")
pytest.importorskip("jax")


class TestImports:
    """Verify public API is importable."""

    def test_import_iterations(self):
        from alpaca.psf import run_psf_reconstruction_iterations
        assert callable(run_psf_reconstruction_iterations)

    def test_import_isolation(self):
        from alpaca.psf import isolate_point_sources, generate_isolated_ps_images
        assert callable(isolate_point_sources)
        assert callable(generate_isolated_ps_images)



class TestPsfUtils:
    """Test simple PSF utility functions."""

    def test_ensure_dir(self, tmp_path):
        from alpaca.psf.utils import _ensure_dir

        new_dir = str(tmp_path / "subdir" / "nested")
        _ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_circular_mask_centered(self):
        from alpaca.psf.utils import _circular_mask_centered

        # Returns float array with 0s inside circle, 1s outside
        mask = _circular_mask_centered(21, cy=10.0, cx=10.0, radius=5)
        assert mask.shape == (21, 21)
        # Center pixel should be zeroed (inside circle)
        assert mask[10, 10] == 0.0
        # Corner pixel should be 1 (outside circle)
        assert mask[0, 0] == 1.0

    def test_center_crop_or_pad(self):
        from alpaca.psf.utils import _center_crop_or_pad

        arr = np.ones((10, 10))
        # Crop to smaller
        cropped = _center_crop_or_pad(arr, (6, 6))
        assert cropped.shape == (6, 6)
        assert np.all(cropped == 1.0)

        # Pad to larger
        padded = _center_crop_or_pad(arr, (14, 14))
        assert padded.shape == (14, 14)
