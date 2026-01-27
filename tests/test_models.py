"""Tests for alpaca.models module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("jax")
pytest.importorskip("numpyro")


class TestImports:
    """Verify public API is importable."""

    def test_import_prob_model(self):
        from alpaca.models import ProbModel
        assert ProbModel is not None

    def test_import_prob_model_corr_field(self):
        from alpaca.models import ProbModelCorrField
        assert ProbModelCorrField is not None

    def test_import_helpers(self):
        from alpaca.models.prob_model import make_lens_image, create_corr_field
        assert callable(make_lens_image)
        assert callable(create_corr_field)
