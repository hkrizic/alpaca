"""
ALPACA: Automated Lens-modelling Pipeline for Accelerated TD Cosmography Analysis

A modular pipeline for gravitational lens modeling with support for:
- Shapelets source reconstruction
- Correlated Fields (Gaussian Process) source reconstruction
- Time delay cosmography
- Multiple inference methods (gradient descent, NUTS, Nautilus)
- PSF reconstruction
"""

from . import data
from . import models
from . import likelihood
from . import inference
from . import utils
from . import output
from . import plotting
from . import benchmarking
from . import psf
from .config import PipelineConfig

__version__ = "1.0.0"
__all__ = [
    "data",
    "models",
    "likelihood",
    "inference",
    "utils",
    "output",
    "plotting",
    "benchmarking",
    "psf",
    "PipelineConfig",
]
