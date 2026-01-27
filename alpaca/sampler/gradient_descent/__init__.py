"""Gradient descent optimization module for MAP estimation.

Provides two-phase gradient descent (Adam + L-BFGS) with multi-start
exploration and refinement, plus BIC computation utilities.
"""

from alpaca.sampler.gradient_descent.bic import compute_bic, compute_bic_from_results
from alpaca.sampler.gradient_descent.optimizer import (
    adam_preopt,
    adam_preopt_legacy,
    load_multistart_summary,
    make_safe_loss,
    run_gradient_descent,
)

__all__ = [
    "compute_bic",
    "compute_bic_from_results",
    "adam_preopt",
    "adam_preopt_legacy",
    "load_multistart_summary",
    "make_safe_loss",
    "run_gradient_descent",
]
