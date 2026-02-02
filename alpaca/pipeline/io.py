"""
alpaca.pipeline.io

I/O utility functions for the pipeline: directory creation, FITS and JSON saving,
and output directory structure setup. They are used in runner.py to organize
and save the results of each pipeline stage.

author: hkrizic
"""

from __future__ import annotations

import json
import os

import numpy as np
from astropy.io import fits


def _ensure_dir(path: str) -> str:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str
        Filesystem path to create.

    Returns
    -------
    str
        The same *path* that was passed in, for convenient chaining.
    """
    os.makedirs(path, exist_ok=True)
    return path


def _save_fits(path: str, data: np.ndarray, overwrite: bool = True) -> None:
    """
    Save a NumPy array as a FITS file.

    The array is cast to ``float64`` before writing.

    Parameters
    ----------
    path : str
        Destination file path (including ``.fits`` extension).
    data : np.ndarray
        Array to write.
    overwrite : bool, optional
        If ``True`` (default), overwrite an existing file at *path*.

    Returns
    -------
    None
    """
    fits.writeto(path, np.asarray(data, dtype=np.float64), overwrite=overwrite)


def _save_json(path: str, data: dict) -> None:
    """
    Save a dictionary as a pretty-printed JSON file.

    Scalar NumPy values are converted to ``float``; other non-serializable
    objects are converted to ``str`` via a fallback handler.

    Parameters
    ----------
    path : str
        Destination file path (including ``.json`` extension).
    data : dict
        Dictionary to serialize.

    Returns
    -------
    None
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))


def _make_output_structure(base_outdir: str) -> dict[str, str]:
    """
    Create the organized output directory tree for a pipeline run.

    The following subdirectories are created under *base_outdir*:

    - ``psf_reconstruction/`` (plots, fits)
    - ``multistart/`` (plots)
    - ``sampling/`` (plots, chains)
    - ``posterior/`` (plots, draws)

    Parameters
    ----------
    base_outdir : str
        Root output directory for the pipeline run.

    Returns
    -------
    dict of str
        Mapping from short keys (e.g. ``"psf_plots"``, ``"sampling_chains"``)
        to the corresponding absolute directory paths.
    """
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
