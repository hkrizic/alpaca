"""
Test that all ALPACA modules can be imported correctly.
"""

import pytest


class TestPackageImports:
    """Test basic package imports."""

    def test_import_alpaca(self):
        """Test main package import."""
        import alpaca
        assert hasattr(alpaca, "__version__")
        assert alpaca.__version__ == "1.0.0"

    def test_import_data(self):
        """Test data module import."""
        from alpaca import data
        assert hasattr(data, "tdlmc_paths")
        assert hasattr(data, "load_image")

    def test_import_models(self):
        """Test models module import."""
        from alpaca import models
        assert hasattr(models, "make_lens_image")
        assert hasattr(models, "ProbModel")

    def test_import_likelihood(self):
        """Test likelihood module import."""
        from alpaca import likelihood
        assert hasattr(likelihood, "imaging_loglike")

    def test_import_inference(self):
        """Test inference module import."""
        from alpaca import inference
        assert hasattr(inference, "run_gradient_descent")
        assert hasattr(inference, "run_nuts")
        assert hasattr(inference, "run_nautilus")

    def test_import_utils(self):
        """Test utils module import."""
        from alpaca import utils
        assert hasattr(utils, "make_pixel_grids")

    def test_import_output(self):
        """Test output module import."""
        from alpaca import output
        assert hasattr(output, "print_gradient_descent_results")

    def test_import_plotting(self):
        """Test plotting module import."""
        from alpaca import plotting
        # Check it's importable
        assert plotting is not None

    def test_import_benchmarking(self):
        """Test benchmarking module import."""
        from alpaca import benchmarking
        assert hasattr(benchmarking, "load_multistart_summary")
        assert hasattr(benchmarking, "summarise_nuts")
        assert hasattr(benchmarking, "summarise_nautilus")

    def test_import_psf(self):
        """Test PSF module import."""
        from alpaca import psf
        assert hasattr(psf, "reconstruct_PSF")
        assert hasattr(psf, "_HAS_STARRED")

    def test_import_config(self):
        """Test config import."""
        from alpaca import PipelineConfig
        assert PipelineConfig is not None


class TestSubmoduleImports:
    """Test specific submodule imports."""

    def test_import_prob_model(self):
        """Test ProbModel import."""
        from alpaca.models import ProbModel
        assert ProbModel is not None

    def test_import_prob_model_corrfield(self):
        """Test ProbModelCorrField import."""
        from alpaca.models import ProbModelCorrField
        assert ProbModelCorrField is not None

    def test_import_gradient_descent(self):
        """Test gradient descent import."""
        from alpaca.inference import run_gradient_descent
        assert callable(run_gradient_descent)

    def test_import_nuts(self):
        """Test NUTS import."""
        from alpaca.inference import run_nuts
        assert callable(run_nuts)

    def test_import_nautilus(self):
        """Test Nautilus import."""
        from alpaca.inference import run_nautilus
        assert callable(run_nautilus)

    def test_import_psf_functions(self):
        """Test PSF function imports."""
        from alpaca.psf import (
            reconstruct_PSF,
            reconstruct_psf_from_star_rotations,
            generate_isolated_ps_images,
            build_centered_cutouts_from_isolated,
        )
        assert callable(reconstruct_PSF)
        assert callable(reconstruct_psf_from_star_rotations)
        assert callable(generate_isolated_ps_images)
        assert callable(build_centered_cutouts_from_isolated)
