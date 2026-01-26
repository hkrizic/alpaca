"""
Integration tests for ALPACA pipeline.

These tests run the full pipeline on actual TDC data but with minimal
iterations to verify syntax and integration. Results will be nonsense
but the code paths are validated.

Run with: pytest tests/test_integration.py --run-integration -v
"""

import pytest
import os
import tempfile
import shutil

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# Path to TDC data (adjust if needed)
TDC_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "TDC"
)


def _tdc_data_available():
    """Check if TDC data is available for testing."""
    test_path = os.path.join(TDC_BASE_PATH, "rung2", "code1", "f160w-seed119")
    return os.path.exists(test_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for test results."""
    tmpdir = tempfile.mkdtemp(prefix="alpaca_test_")
    yield tmpdir
    # Cleanup after test
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def minimal_shapelets_config(temp_output_dir):
    """Create a minimal config for quick shapelets pipeline test."""
    from alpaca import PipelineConfig

    return PipelineConfig(
        # Data
        base_path=os.path.dirname(TDC_BASE_PATH),  # Parent of TDC folder
        rung=2,
        code_id=1,
        seed=119,

        # Source model
        use_source_shapelets=True,
        use_corr_fields=False,
        shapelets_n_max=4,  # Minimal for speed

        # Likelihood
        use_time_delays=True,
        use_rayshoot_consistency=False,

        # Minimal gradient descent (just 2 starts, few steps)
        n_starts_initial=2,
        n_top_for_refinement=1,
        n_refinement_perturbations=1,
        adam_steps_initial=10,
        adam_steps_refinement=10,
        lbfgs_maxiter_initial=5,
        lbfgs_maxiter_refinement=5,

        # Minimal NUTS (nonsense results but validates syntax)
        nuts_num_warmup=5,
        nuts_num_samples=10,
        nuts_num_chains=1,

        # Minimal Nautilus
        nautilus_n_live=10,
    )


@pytest.fixture
def minimal_corrfields_config(temp_output_dir):
    """Create a minimal config for quick correlated fields pipeline test."""
    from alpaca import PipelineConfig

    return PipelineConfig(
        # Data
        base_path=os.path.dirname(TDC_BASE_PATH),
        rung=2,
        code_id=1,
        seed=119,

        # Source model
        use_source_shapelets=False,
        use_corr_fields=True,
        corr_field_num_pixels=40,  # Minimal for speed

        # Likelihood
        use_time_delays=True,
        use_rayshoot_consistency=False,

        # Minimal gradient descent
        n_starts_initial=2,
        n_top_for_refinement=1,
        n_refinement_perturbations=1,
        adam_steps_initial=10,
        adam_steps_refinement=10,
        lbfgs_maxiter_initial=5,
        lbfgs_maxiter_refinement=5,

        # Minimal NUTS
        nuts_num_warmup=5,
        nuts_num_samples=10,
        nuts_num_chains=1,

        # Minimal Nautilus
        nautilus_n_live=10,
    )


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_shapelets_gradient_descent_only(self, minimal_shapelets_config):
        """Test shapelets pipeline with gradient descent only (no sampling)."""
        from alpaca.pipeline import run_pipeline

        results = run_pipeline(
            config=minimal_shapelets_config,
            run_sampling=False,
            verbose=True,
        )

        # Basic checks
        assert results is not None
        assert "gd_results" in results
        assert "best_params" in results["gd_results"]
        assert results["nuts_results"] is None
        assert results["nautilus_results"] is None

        # Check output files exist
        run_dir = results["run_dir"]
        assert os.path.exists(run_dir)
        assert os.path.exists(os.path.join(run_dir, "gradient_descent", "best_fit_params.json"))

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_shapelets_with_nuts(self, minimal_shapelets_config):
        """Test shapelets pipeline with NUTS sampling."""
        from alpaca.pipeline import run_pipeline

        results = run_pipeline(
            config=minimal_shapelets_config,
            run_sampling=True,
            sampler="nuts",
            verbose=True,
        )

        # Basic checks
        assert results is not None
        assert "nuts_results" in results
        assert results["nuts_results"] is not None
        assert "samples" in results["nuts_results"]

        # Check output files
        run_dir = results["run_dir"]
        assert os.path.exists(os.path.join(run_dir, "sampling", "nuts_samples.npz"))
        assert os.path.exists(os.path.join(run_dir, "final_outputs", "output.json"))

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_shapelets_with_nautilus(self, minimal_shapelets_config):
        """Test shapelets pipeline with Nautilus sampling."""
        from alpaca.pipeline import run_pipeline

        results = run_pipeline(
            config=minimal_shapelets_config,
            run_sampling=True,
            sampler="nautilus",
            verbose=True,
        )

        # Basic checks
        assert results is not None
        assert "nautilus_results" in results
        assert results["nautilus_results"] is not None

        # Check output files
        run_dir = results["run_dir"]
        assert os.path.exists(os.path.join(run_dir, "sampling", "nautilus_samples.npz"))

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_corrfields_gradient_descent_only(self, minimal_corrfields_config):
        """Test correlated fields pipeline with gradient descent only."""
        from alpaca.pipeline import run_pipeline

        results = run_pipeline(
            config=minimal_corrfields_config,
            run_sampling=False,
            verbose=True,
        )

        # Basic checks
        assert results is not None
        assert "gd_results" in results
        assert "best_params" in results["gd_results"]

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_corrfields_with_nuts(self, minimal_corrfields_config):
        """Test correlated fields pipeline with NUTS sampling."""
        from alpaca.pipeline import run_pipeline

        results = run_pipeline(
            config=minimal_corrfields_config,
            run_sampling=True,
            sampler="nuts",
            verbose=True,
        )

        # Basic checks
        assert results is not None
        assert results["nuts_results"] is not None


class TestSetupIntegration:
    """Integration tests for lens system setup."""

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_setup_lens_system_shapelets(self, minimal_shapelets_config):
        """Test lens system setup with shapelets."""
        from alpaca.models import setup_lens_system

        setup = setup_lens_system(minimal_shapelets_config)

        assert setup is not None
        assert "prob_model" in setup
        assert "img" in setup
        assert "noise_map" in setup
        assert "psf_kernel" in setup
        assert "x0s" in setup
        assert len(setup["x0s"]) > 0  # Should detect point sources

    @pytest.mark.skipif(
        not _tdc_data_available(),
        reason="TDC data not available"
    )
    def test_setup_lens_system_corrfields(self, minimal_corrfields_config):
        """Test lens system setup with correlated fields."""
        from alpaca.models import setup_lens_system

        setup = setup_lens_system(minimal_corrfields_config)

        assert setup is not None
        assert "prob_model" in setup
