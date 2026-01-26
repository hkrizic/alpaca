"""
Data loading functions for TDLMC challenge.
"""

import os
from typing import Tuple, Dict, Optional
import numpy as np
from astropy.io import fits


def tdlmc_paths(
    base: str,
    rung: int,
    code_id: int,
    seed: int,
) -> Tuple[str, str]:
    """
    Return (data_folder, output_dir) paths for a TDLMC system.

    Parameters
    ----------
    base : str
        Base directory containing TDC data.
    rung : int
        TDLMC rung number (0, 1, 2, or 3).
    code_id : int
        Code identifier.
    seed : int
        Random seed for the system.

    Returns
    -------
    data_folder : str
        Path to drizzled image folder.
    output_dir : str
        Path for output results.
    """
    code = f"code{code_id}"
    data_folder = os.path.join(
        base,
        f"TDC/rung{rung}/{code}/f160w-seed{seed}/drizzled_image",
    )
    output_dir = os.path.join(
        base,
        f"TDC_results/rung{rung}/{code}/f160w-seed{seed}",
    )
    os.makedirs(output_dir, exist_ok=True)
    return data_folder, output_dir


def load_image(folder: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load image, PSF kernel, and noise map from a TDLMC folder.

    Parameters
    ----------
    folder : str
        Path to drizzled_image folder.

    Returns
    -------
    img : np.ndarray
        Observed lens image.
    psf_kernel : np.ndarray
        Point spread function kernel.
    noise_map : np.ndarray
        Per-pixel noise standard deviation.
    """
    img = fits.getdata(
        os.path.join(folder, "lens-image.fits"), header=False
    ).astype(np.float64)

    psf_kernel = fits.getdata(
        os.path.join(folder, "psf.fits"), header=False
    ).astype(np.float64)

    noise_map = fits.getdata(
        os.path.join(folder, "noise_map.fits"), header=False
    ).astype(np.float64)

    return img, psf_kernel, noise_map


def load_truth(
    base: str,
    rung: int,
    code_id: int,
    seed: int,
) -> Optional[Dict]:
    """
    Load truth values for a TDLMC system (if available).

    Parameters
    ----------
    base : str
        Base directory containing TDC data.
    rung : int
        TDLMC rung number.
    code_id : int
        Code identifier.
    seed : int
        Random seed for the system.

    Returns
    -------
    truth : Dict or None
        Dictionary with truth values, or None if not available.
        Keys may include: D_dt, time_delays, image_positions, etc.
    """
    code = f"code{code_id}"
    truth_path = os.path.join(
        base,
        f"TDC/rung{rung}/{code}/f160w-seed{seed}/truth.json",
    )

    if not os.path.exists(truth_path):
        return None

    import json
    with open(truth_path) as f:
        return json.load(f)


def get_cutouts(
    img: np.ndarray,
    positions: list,
    box_size: int = 25,
) -> list:
    """
    Extract cutouts around specified positions.

    Parameters
    ----------
    img : np.ndarray
        Full image array.
    positions : list
        List of (y, x) pixel positions.
    box_size : int
        Size of cutout box (must be odd).

    Returns
    -------
    cutouts : list
        List of cutout arrays.
    """
    half = box_size // 2
    ny, nx = img.shape
    cutouts = []

    for (yc, xc) in positions:
        y0 = max(0, yc - half)
        y1 = min(ny, yc + half + 1)
        x0 = max(0, xc - half)
        x1 = min(nx, xc + half + 1)
        cutouts.append(img[y0:y1, x0:x1].copy())

    return cutouts
