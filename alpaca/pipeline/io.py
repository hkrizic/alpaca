"""
alpaca.pipeline.io

I/O utility functions for the pipeline: directory creation, FITS and JSON saving,
and output directory structure setup. They are used in runner.py to organize
and save the results of each pipeline stage.
"""

from __future__ import annotations

import json
import os

import numpy as np
from astropy.io import fits


def _ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def _save_fits(path: str, data: np.ndarray, overwrite: bool = True) -> None:
    """Save array as FITS file."""
    fits.writeto(path, np.asarray(data, dtype=np.float64), overwrite=overwrite)


def _save_json(path: str, data: dict) -> None:
    """Save dictionary as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def _make_output_structure(base_outdir: str) -> dict[str, str]:
    """Create organized output directory structure."""
    dirs = {
        "root": base_outdir,
        "psf": os.path.join(base_outdir, "psf_reconstruction"),
        "psf_plots": os.path.join(base_outdir, "psf_reconstruction", "plots"),
        "psf_fits": os.path.join(base_outdir, "psf_reconstruction", "fits"),
        "multistart": os.path.join(base_outdir, "multistart"),
        "multistart_plots": os.path.join(base_outdir, "multistart", "plots"),
        "sampling": os.path.join(base_outdir, "sampling"),
        "sampling_plots": os.path.join(base_outdir, "sampling", "plots"),
        "sampling_chains": os.path.join(base_outdir, "sampling", "chains"),
        "posterior": os.path.join(base_outdir, "posterior"),
        "posterior_plots": os.path.join(base_outdir, "posterior", "plots"),
        "posterior_draws": os.path.join(base_outdir, "posterior", "draws"),
    }
    for d in dirs.values():
        _ensure_dir(d)
    return dirs
