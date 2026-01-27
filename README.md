<p align="center">
  <img src="alpaca/alpaca_logo.png" alt="ALPACA Logo" width="100%">
</p>

# ALPACA (Automated Lens-modelling Pipeline for Accelerated TD Cosmography Analysis)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/hkrizic/alpaca/actions/workflows/tests.yml/badge.svg)](https://github.com/hkrizic/alpaca/actions/workflows/tests.yml)

ALPACA is a modular, JAX-accelerated pipeline for gravitational lens modeling and time-delay cosmography. It provides flexible source reconstruction methods and multiple Bayesian inference backends for robust parameter estimation and cosmological constraints.

## Features

- **Inference Methods**
  - Multi-start gradient descent (Adam + L-BFGS-B)
  - NUTS Hamiltonian Monte Carlo via NumPyro
  - Nautilus nested sampling for evidence estimation

- **Time-Delay Cosmography**
  - Joint modeling of imaging and time delays
  - Direct inference of the time-delay distance D_dt
  - Ray-shooting consistency likelihood

- **PSF Reconstruction**
  - STARRED-based PSF reconstruction from lensed point sources
  - Iterative PSF refinement

- **Source Reconstruction**
  - Shapelets basis decomposition
  - Correlated Fields (Gaussian Process) reconstruction with NIFTy-inspired priors

- **Performance**
  - JAX-accelerated likelihood evaluation
  - GPU support for fast inference
  - Automatic differentiation for HMC

## Installation

### From source

```bash
git clone https://github.com/hkrizic/alpaca.git
cd alpaca
pip install -e ".[full]"
```

### Requirements

- Python >= 3.10
- JAX >= 0.4
- NumPyro >= 0.12
- NumPy, SciPy, Matplotlib, Astropy

## Quick Start

### 1. Configure the pipeline

Edit `run_config.py` to set your data paths, source model, sampler, and pipeline phases:

```python
# run_config.py (excerpt)

BASE_DIR = "."
RUNG = 2
CODE_ID = 1
SEED = 120

USE_SHAPELETS = True        # Shapelets source model
SHAPELETS_N_MAX = 6

SAMPLER = "nuts"            # "nuts", "nautilus", or "default"

RUN_PSF_RECONSTRUCTION = False
RUN_MULTISTART = True
RUN_SAMPLING = True
```

The `load_config()` function at the bottom of `run_config.py` builds a `PipelineConfig` from these settings.

### 2. Run the pipeline

```bash
python run_alpaca.py
```

The pipeline executes three phases in sequence:

1. **PSF Reconstruction** -- Iterative PSF estimation using STARRED.
2. **Gradient Descent Optimization** -- Multi-start MAP estimation with Adam + L-BFGS.
3. **Posterior Sampling** -- Bayesian inference via NUTS or Nautilus.

### 3. Use the pipeline programmatically

```python
from alpaca.config import PipelineConfig
from alpaca.pipeline import run_pipeline, quick_pipeline, load_pipeline_results

# Run the full pipeline with a config object
results = run_pipeline(config=config, verbose=True)

# Or load previously saved results
results = load_pipeline_results(output_dir)
```

## Configuration

ALPACA uses a dataclass-based configuration system defined in `alpaca/config.py`. The user-facing settings live in `run_config.py` as plain Python variables:

```python
from run_config import load_config

config = load_config()  # Builds a PipelineConfig from run_config.py settings
```

Or build a `PipelineConfig` directly:

```python
from alpaca.config import (
    PipelineConfig,
    PSFReconstructionConfig,
    GradientDescentConfig,
    SamplerConfig,
    PlottingConfig,
    CorrFieldConfig,
)

config = PipelineConfig(
    base_dir="/path/to/data",
    rung=2,
    code_id=1,
    seed=120,

    # Source model (choose one)
    use_source_shapelets=True,
    shapelets_n_max=6,

    # Likelihood terms
    use_rayshoot_consistency=True,
    rayshoot_consistency_sigma=0.0002,
    use_rayshoot_systematic_error=True,

    # Phase toggles
    run_psf_reconstruction=False,
    run_multistart=True,
    run_sampling=True,

    # Phase configurations
    psf_config=PSFReconstructionConfig(
        n_iterations=4,
        multistart_starts_per_iteration=20,
    ),
    gradient_descent_config=GradientDescentConfig(
        n_starts_initial=50,
        adam_steps_initial=500,
        lbfgs_maxiter_initial=600,
        use_time_delays=True,
    ),
    sampler_config=SamplerConfig(
        sampler="nuts",
        use_time_delays=True,
        nuts_num_warmup=3000,
        nuts_num_samples=5000,
    ),
    plotting_config=PlottingConfig(
        plot_corner=True,
        plot_chains=True,
    ),
)
```

## Inference Methods

### Gradient Descent

Multi-start optimization with Adam pre-conditioning and L-BFGS-B refinement:

```python
from alpaca.sampler.gradient_descent import run_gradient_descent

results = run_gradient_descent(
    prob_model,
    n_starts=50,
    adam_steps=500,
    lbfgs_maxiter=600,
)
best_params = results["best_params"]
```

### NUTS (Hamiltonian Monte Carlo)

Full posterior sampling with the No-U-Turn Sampler:

```python
from alpaca.sampler.nuts import run_nuts_numpyro

results = run_nuts_numpyro(
    prob_model,
    best_params=best_params,  # Initialize near MAP
    num_warmup=3000,
    num_samples=5000,
)
samples = results["samples"]
```

### Nautilus (Nested Sampling)

Evidence estimation and posterior sampling:

```python
from alpaca.sampler.nautilus import run_nautilus

results = run_nautilus(
    prob_model,
    best_params=best_params,
    n_live=1000,
)
log_evidence = results["log_evidence"]
samples = results["samples"]
```

## PSF Reconstruction

Reconstruct the PSF from lensed point source images using STARRED:

```python
from alpaca.psf import reconstruct_PSF, generate_isolated_ps_images

# Generate isolated point source images
isolated_images = generate_isolated_ps_images(
    data_image,
    prob_model,
    best_params,
    n_point_sources=4,
)

# Reconstruct PSF
new_psf = reconstruct_PSF(
    current_psf=psf_kernel,
    peaks_px=point_source_positions,
    noise_map=noise_map,
    isolated_images=isolated_images,
    cutout_size=99,
    supersampling_factor=3,
)
```

## Project Structure

```
.
├── run_alpaca.py              # Entry point: runs the pipeline and explores results
├── run_config.py              # User-editable configuration (edit this, then run)
├── run_alpaca.sh              # SLURM submission script for HPC clusters
├── tests/                     # Test suite (pytest)
│
└── alpaca/
    ├── __init__.py
    ├── config.py              # Configuration dataclasses (PipelineConfig, etc.)
    │
    ├── pipeline/              # Pipeline orchestration
    │   ├── __init__.py        #   re-exports: run_pipeline, quick_pipeline, load_pipeline_results
    │   ├── runner.py          #   main pipeline runner
    │   ├── io.py              #   output directory setup, FITS/JSON serialization
    │   ├── setup.py           #   model building, point-source matching, time-delay loading
    │   └── stages/
    │       ├── sampling.py    #   NUTS and Nautilus sampling stages
    │       └── plotting.py    #   posterior plot generation stage
    │
    ├── data/                  # Data loading and preprocessing
    │   ├── loader.py          #   FITS data loading
    │   ├── setup.py           #   high-level lens setup (setup_tdlmc_lens)
    │   ├── grids.py           #   pixel grid construction
    │   ├── masks.py           #   source arc masks (annular and custom)
    │   ├── detection.py       #   point-source image detection
    │   └── noise.py           #   noise boosting around point sources
    │
    ├── models/                # Probabilistic lens model
    │   └── prob_model.py
    │
    ├── sampler/               # Optimization and sampling
    │   ├── constants.py       #   physical constants (D_dt bounds, c)
    │   ├── priors.py          #   truncated-normal and bounded priors
    │   ├── losses.py          #   time-delay and ray-shooting loss functions
    │   ├── utils.py           #   RNG, environment info, conversions
    │   ├── likelihood.py      #   likelihood construction
    │   ├── gradient_descent/
    │   │   ├── optimizer.py   #   Adam + L-BFGS multi-start optimization
    │   │   └── bic.py         #   Bayesian Information Criterion
    │   ├── nautilus/
    │   │   ├── prior.py       #   Nautilus prior mapping
    │   │   ├── sampler.py     #   Nautilus nested sampler wrapper
    │   │   └── posterior.py   #   posterior extraction from Nautilus
    │   └── nuts/
    │       ├── sampler.py     #   NumPyro NUTS wrapper
    │       └── posterior.py   #   posterior extraction from NUTS
    │
    ├── plotting/              # Visualization
    │   ├── model_plots.py     #   best-fit model and residual maps
    │   ├── posterior_plots.py #   corner plots and posterior analysis
    │   ├── diagnostics.py     #   optimization and chain diagnostics
    │   └── benchmarking.py    #   timing and performance plots
    │
    ├── psf/                   # PSF reconstruction (STARRED)
    │   ├── reconstruction.py  #   main PSF reconstruction driver
    │   ├── iterations.py      #   iterative refinement loop
    │   ├── isolation.py       #   star isolation and cutout extraction
    │   └── utils.py           #   PSF helper functions
    │
    └── utils/                 # Shared utilities
        ├── cosmology.py       #   D_dt, time-delay prediction, H0 conversions
        └── jax_helpers.py     #   JAX pytree utilities
```

## Testing

```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

ALPACA builds upon several excellent packages:

- [Herculens](https://github.com/Herculens/herculens) for differentiable lens modeling
- [JAX](https://github.com/google/jax) for automatic differentiation
- [NumPyro](https://github.com/pyro-ppl/numpyro) for probabilistic programming
- [Nautilus](https://github.com/johannesulf/nautilus) for nested sampling
- [STARRED](https://gitlab.com/cosmograil/starred) for PSF reconstruction
