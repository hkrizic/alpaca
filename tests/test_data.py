"""Tests for alpaca.data module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("numpy")
pytest.importorskip("astropy")


class TestImports:
    """Verify public API is importable."""

    def test_import_loader(self):
        from alpaca.data import tdlmc_paths, load_tdlmc_image
        assert callable(tdlmc_paths)
        assert callable(load_tdlmc_image)

    def test_import_setup(self):
        from alpaca.data import setup_tdlmc_lens, make_pixel_grids
        assert callable(setup_tdlmc_lens)
        assert callable(make_pixel_grids)


class TestTdlmcPaths:
    """Test tdlmc_paths path construction."""

    def test_returns_tuple(self, tmp_path):
        from alpaca.data.loader import tdlmc_paths

        folder, outdir = tdlmc_paths(str(tmp_path), rung=2, code_id=1, seed=120)
        assert isinstance(folder, str)
        assert isinstance(outdir, str)
        assert "rung2" in folder
        assert "code1" in folder
        assert "seed120" in folder

    def test_creates_outdir(self, tmp_path):
        from alpaca.data.loader import tdlmc_paths

        _, outdir = tdlmc_paths(str(tmp_path), rung=0, code_id=3, seed=99)
        assert os.path.isdir(outdir)
