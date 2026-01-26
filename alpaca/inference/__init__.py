"""
Inference methods for lens modeling.

All methods use the unified likelihood defined in the probabilistic model,
ensuring consistency across different inference approaches.

Available samplers:
- NUTS (No-U-Turn Sampler): Hamiltonian Monte Carlo with automatic tuning
- Nautilus: Nested sampling for evidence estimation and posterior inference
"""

from .gradient_descent import run_gradient_descent
from .nuts import run_nuts, load_nuts_samples, get_nuts_posterior
from .nautilus import (
    run_nautilus,
    load_posterior_from_checkpoint,
    build_prior_and_loglike,
    get_nautilus_posterior,
)

__all__ = [
    "run_gradient_descent",
    "run_nuts",
    "load_nuts_samples",
    "get_nuts_posterior",
    "run_nautilus",
    "load_posterior_from_checkpoint",
    "build_prior_and_loglike",
    "get_nautilus_posterior",
]
