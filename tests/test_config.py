"""
Test ALPACA configuration classes.
"""

import pytest
import tempfile
import os


class TestPipelineConfig:
    """Test PipelineConfig class."""

    def test_config_creation_defaults(self):
        """Test basic config creation with defaults."""
        from alpaca import PipelineConfig

        config = PipelineConfig()
        assert config.rung == 2
        assert config.pix_scl == 0.08
        assert config.supersampling_factor == 5

    def test_config_custom_values(self):
        """Test config with custom values."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            rung=3,
            code_id=2,
            seed=123,
            pix_scl=0.05,
        )
        assert config.rung == 3
        assert config.code_id == 2
        assert config.seed == 123
        assert config.pix_scl == 0.05

    def test_config_shapelets_settings(self):
        """Test shapelets configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_source_shapelets=True,
            use_corr_fields=False,
            shapelets_n_max=8,
            shapelets_beta_min=0.01,
            shapelets_beta_max=0.5,
        )
        assert config.use_source_shapelets is True
        assert config.use_corr_fields is False
        assert config.shapelets_n_max == 8

    def test_config_corrfield_settings(self):
        """Test correlated field configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_source_shapelets=False,
            use_corr_fields=True,
            corr_field_num_pixels=100,
        )
        assert config.use_corr_fields is True
        assert config.use_source_shapelets is False
        assert config.corr_field_num_pixels == 100

    def test_config_time_delay_settings(self):
        """Test time delay configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_time_delays=True,
            measured_delays=(10.0, 20.0, 30.0),
            delay_errors=(1.0, 1.5, 2.0),
        )
        assert config.use_time_delays is True
        assert len(config.measured_delays) == 3

    def test_config_gradient_descent_settings(self):
        """Test gradient descent configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            n_starts_initial=100,
            adam_steps_initial=1000,
            adam_lr=1e-3,
            lbfgs_maxiter_initial=500,
        )
        assert config.n_starts_initial == 100
        assert config.adam_steps_initial == 1000
        assert config.adam_lr == 1e-3

    def test_config_nuts_settings(self):
        """Test NUTS configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            nuts_num_warmup=500,
            nuts_num_samples=1000,
            nuts_target_accept=0.85,
            nuts_max_tree_depth=12,
            nuts_chain_method="vectorized",
        )
        assert config.nuts_num_warmup == 500
        assert config.nuts_num_samples == 1000
        assert config.nuts_target_accept == 0.85
        assert config.nuts_max_tree_depth == 12
        assert config.nuts_chain_method == "vectorized"

    def test_config_nautilus_settings(self):
        """Test Nautilus configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            nautilus_n_live=500,
            nautilus_n_batch=32,
            nautilus_pool=4,
            nautilus_n_posterior_samples=1000,
        )
        assert config.nautilus_n_live == 500
        assert config.nautilus_n_batch == 32
        assert config.nautilus_pool == 4
        assert config.nautilus_n_posterior_samples == 1000

    def test_config_validation_both_source_models(self):
        """Test validation fails when both source models enabled."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_source_shapelets=True,
            use_corr_fields=True,
        )
        with pytest.raises(ValueError, match="Cannot use both"):
            config.validate()

    def test_config_validation_no_source_model(self):
        """Test validation fails when no source model enabled."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_source_shapelets=False,
            use_corr_fields=False,
        )
        with pytest.raises(ValueError, match="Must enable"):
            config.validate()

    def test_config_validation_success(self):
        """Test validation succeeds with valid config."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_source_shapelets=True,
            use_corr_fields=False,
        )
        # Should not raise
        config.validate()

    def test_config_to_dict(self):
        """Test config to_dict method."""
        from alpaca import PipelineConfig

        config = PipelineConfig(rung=3, seed=456)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["rung"] == 3
        assert d["seed"] == 456

    def test_config_save_load(self):
        """Test config save and load."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            rung=3,
            seed=789,
            pix_scl=0.1,
            use_source_shapelets=True,
            use_corr_fields=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            config.save(path)

            # Load and verify
            loaded = PipelineConfig.load(path)
            assert loaded.rung == 3
            assert loaded.seed == 789
            assert loaded.pix_scl == 0.1
            assert loaded.use_source_shapelets is True

    def test_config_psf_reconstruction_settings(self):
        """Test PSF reconstruction configuration."""
        from alpaca import PipelineConfig

        config = PipelineConfig(
            use_psf_reconstruction=True,
            psf_reconstruction_iterations=2,
            psf_cutout_size=99,
            psf_supersampling_factor=3,
            psf_mask_other_peaks=True,
            psf_mask_radius=8,
            psf_rotation_mode="180",
        )
        assert config.use_psf_reconstruction is True
        assert config.psf_reconstruction_iterations == 2
        assert config.psf_cutout_size == 99
        assert config.psf_supersampling_factor == 3
        assert config.psf_mask_other_peaks is True
        assert config.psf_mask_radius == 8
        assert config.psf_rotation_mode == "180"
