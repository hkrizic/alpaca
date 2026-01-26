"""
Likelihood module for TDLMC pipeline.

This module defines all likelihood terms in ONE place, eliminating duplication.
The same functions are used for gradient descent, NUTS, and Nautilus.
"""

from .imaging import imaging_loglike, imaging_chi2
from .time_delay import time_delay_loglike, fermat_potential
from .rayshoot import rayshoot_consistency_loglike
from .combined import CombinedLikelihood

__all__ = [
    "imaging_loglike",
    "imaging_chi2",
    "time_delay_loglike",
    "fermat_potential",
    "rayshoot_consistency_loglike",
    "CombinedLikelihood",
]
