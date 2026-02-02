"""Gradient descent optimization module for MAP estimation.

Provides two-phase gradient descent (Adam + L-BFGS) with multi-start
exploration and refinement.

author: hkrizic
"""

from alpaca.sampler.gradient_descent.optimizer import (
    load_multistart_summary,
    make_safe_loss,
    run_gradient_descent,
)

__all__ = [
    "load_multistart_summary",
    "make_safe_loss",
    "run_gradient_descent",
]
