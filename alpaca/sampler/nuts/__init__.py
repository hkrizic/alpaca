"""NUTS (No-U-Turn Sampler) module for Alpaca.

Provides Hamiltonian Monte Carlo sampling via NumPyro's NUTS implementation
with multi-device parallelization support.
"""

from alpaca.sampler.nuts.posterior import get_nuts_posterior
from alpaca.sampler.nuts.sampler import load_nuts_samples, run_nuts_numpyro

__all__ = [
    "get_nuts_posterior",
    "load_nuts_samples",
    "run_nuts_numpyro",
]
