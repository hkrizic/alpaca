<p align="center">
  <img src="alpaca/alpaca_logo.png" alt="ALPACA Logo" width="100%">
</p>

# ALPACA

**Automated Lens-modelling Pipeline for Accelerated TD Cosmography Analysis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ALPACA is a modular, JAX-accelerated pipeline for gravitational lens modeling and time-delay cosmography. It provides flexible source reconstruction methods and multiple Bayesian inference backends for robust parameter estimation and cosmological constraints.

## Features

- **Source Reconstruction**
  - Shapelets basis decomposition
  - Correlated Fields (Gaussian Process) reconstruction with NIFTy-inspired priors

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

- **Performance**
  - JAX-accelerated likelihood evaluation
  - GPU support for fast inference
  - Automatic differentiation for HMC

## Installation

### From source

```bash
git clone https://github.com/alpaca-team/alpaca.git
cd alpaca
pip install -e .
```

### With optional dependencies

```bash
# For development (includes pytest, ruff)
pip install -e ".[dev]"

# For PSF reconstruction with STARRED
pip install -e ".[starred]"

# For Nautilus nested sampling
pip install -e ".[nautilus]"

# Full installation
pip install -e ".[full]"
```

### Requirements

- Python >= 3.10
- JAX >= 0.4
- NumPyro >= 0.12
- NumPy, SciPy, Matplotlib, Astropy

## Quick Start

```python
from alpaca import PipelineConfig
from alpaca.models import setup_lens_system
from alpaca.inference import run_gradient_descent, run_nuts

# Configure the pipeline
config = PipelineConfig(
    base_path="/path/to/data",
    rung=2,
    code_id=1,
    use_source_shapelets=True,
    shapelets_n_max=6,
    use_time_delays=True,
)
config.validate()

# Set up the lens system
setup = setup_lens_system(config)

# Run gradient descent to find MAP estimate
gd_results = run_gradient_descent(
    setup["prob_model"],
    n_starts=50,
    verbose=True,
)

# Run NUTS sampling for posterior inference
nuts_results = run_nuts(
    setup["prob_model"],
    best_params=gd_results["best_params"],
    num_warmup=1000,
    num_samples=2000,
)
```

## Configuration

ALPACA uses a dataclass-based configuration system:

```python
from alpaca import PipelineConfig

config = PipelineConfig(
    # Data settings
    base_path="/path/to/data",
    rung=2,
    code_id=1,

    # Source model (choose one)
    use_source_shapelets=True,   # Shapelets decomposition
    use_corr_fields=False,       # OR Correlated Fields

    # Shapelets settings
    shapelets_n_max=6,
    shapelets_beta_min=0.02,
    shapelets_beta_max=0.6,

    # Likelihood terms
    use_time_delays=True,
    use_rayshoot_consistency=False,

    # Inference settings
    n_starts_initial=50,
    nuts_num_warmup=1000,
    nuts_num_samples=2000,
)

# Save/load configuration
config.save("config.json")
loaded_config = PipelineConfig.load("config.json")
```

## Inference Methods

### Gradient Descent

Multi-start optimization with Adam pre-conditioning and L-BFGS-B refinement:

```python
from alpaca.inference import run_gradient_descent

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
from alpaca.inference import run_nuts

results = run_nuts(
    prob_model,
    best_params=best_params,  # Initialize near MAP
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
)
samples = results["samples"]
```

### Nautilus (Nested Sampling)

Evidence estimation and posterior sampling:

```python
from alpaca.inference import run_nautilus

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
alpaca/
├── data/           # Data loading utilities
├── models/         # Lens models and probabilistic models
├── likelihood/     # Likelihood functions
├── inference/      # Inference backends (gradient descent, NUTS, Nautilus)
├── psf/            # PSF reconstruction
├── utils/          # Utility functions
├── plotting/       # Visualization tools
├── benchmarking/   # Timing and benchmarking utilities
└── output/         # Output and diagnostics
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=alpaca --cov-report=term-missing
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
