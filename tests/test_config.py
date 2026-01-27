"""Tests for alpaca.config dataclasses."""

import pytest

from alpaca.config import (
    PSFReconstructionConfig,
    GradientDescentConfig,
    SamplerConfig,
    PlottingConfig,
    CorrFieldConfig,
    PipelineConfig,
)


class TestPSFReconstructionConfig:
    """Tests for PSFReconstructionConfig."""

    def test_defaults(self):
        cfg = PSFReconstructionConfig()
        assert cfg.n_iterations == 3
        assert cfg.starred_cutout_size == 99
        assert cfg.starred_supersampling_factor == 3
        assert isinstance(cfg.run_multistart, bool)

    def test_override(self):
        cfg = PSFReconstructionConfig(n_iterations=5, starred_cutout_size=51)
        assert cfg.n_iterations == 5
        assert cfg.starred_cutout_size == 51


class TestGradientDescentConfig:
    """Tests for GradientDescentConfig."""

    def test_defaults(self):
        cfg = GradientDescentConfig()
        assert cfg.n_starts_initial == 50
        assert cfg.adam_steps_initial == 500
        assert cfg.lbfgs_maxiter_initial == 600
        assert cfg.adam_lr == 5e-3
        assert cfg.random_seed == 73

    def test_phases_exist(self):
        cfg = GradientDescentConfig()
        assert hasattr(cfg, "n_starts_initial")
        assert hasattr(cfg, "n_top_for_refinement")
        assert hasattr(cfg, "adam_steps_refinement")
        assert hasattr(cfg, "lbfgs_maxiter_refinement")


class TestSamplerConfig:
    """Tests for SamplerConfig."""

    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.sampler == "default"
        assert cfg.nuts_num_warmup == 1000
        assert cfg.nautilus_n_live == 1000

    def test_valid_sampler_values(self):
        for val in ("nuts", "nautilus", "default"):
            cfg = SamplerConfig(sampler=val)
            assert cfg.sampler == val

    def test_no_emcee_field(self):
        """Verify emcee fields have been removed."""
        cfg = SamplerConfig()
        assert not hasattr(cfg, "emcee_n_walkers")
        assert not hasattr(cfg, "emcee_n_steps")
        assert not hasattr(cfg, "emcee_burnin_fraction")
        assert not hasattr(cfg, "emcee_thin")


class TestPlottingConfig:
    """Tests for PlottingConfig."""

    def test_defaults(self):
        cfg = PlottingConfig()
        assert cfg.save_plots is True
        assert cfg.dpi == 300
        assert cfg.plot_format == "png"


class TestCorrFieldConfig:
    """Tests for CorrFieldConfig."""

    def test_defaults(self):
        cfg = CorrFieldConfig()
        assert cfg.num_pixels == 80
        assert cfg.exponentiate is True
        assert cfg.interpolation_type == "fast_bilinear"

    def test_tuples(self):
        cfg = CorrFieldConfig()
        assert isinstance(cfg.loglogavgslope, tuple)
        assert len(cfg.loglogavgslope) == 2
        assert isinstance(cfg.fluctuations, tuple)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.rung == 2
        assert cfg.code_id == 1
        assert cfg.seed == 120
        assert cfg.run_sampling is True

    def test_nested_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.psf_config, PSFReconstructionConfig)
        assert isinstance(cfg.gradient_descent_config, GradientDescentConfig)
        assert isinstance(cfg.sampler_config, SamplerConfig)
        assert isinstance(cfg.plotting_config, PlottingConfig)
        assert isinstance(cfg.corr_field_config, CorrFieldConfig)

    def test_override_nested(self):
        cfg = PipelineConfig(
            psf_config=PSFReconstructionConfig(n_iterations=7),
            sampler_config=SamplerConfig(sampler="nuts"),
        )
        assert cfg.psf_config.n_iterations == 7
        assert cfg.sampler_config.sampler == "nuts"

    def test_source_model_fields(self):
        cfg = PipelineConfig()
        assert cfg.use_source_shapelets is True
        assert cfg.use_corr_fields is False
        assert cfg.shapelets_n_max == 6
