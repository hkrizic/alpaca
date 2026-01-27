"""
Data I/O utilities for TDLMC strong-lensing images.
"""

import os

import numpy as np
from astropy.io import fits


def tdlmc_paths(base: str, rung: int, code_id: int, seed: int) -> tuple[str, str]:
    """Return (folder, outdir) paths for the drizzled TDLMC image and results."""
    code = f"code{code_id}"
    folder = os.path.join(
        base,
        f"TDC/rung{rung}/{code}/f160w-seed{seed}/drizzled_image",
    )
    outdir = os.path.join(
        base,
        f"TDC_results/rung{rung}/{code}/f160w-seed{seed}",
    )
    os.makedirs(outdir, exist_ok=True)
    return folder, outdir


def load_tdlmc_image(folder: str):
    """Load image, PSF kernel and noise map from a TDLMC drizzled_image folder."""
    img = fits.getdata(os.path.join(folder, "lens-image.fits"), header=False).astype(
        np.float64
    )
    psf_kernel = fits.getdata(
        os.path.join(folder, "psf.fits"), header=False
    ).astype(np.float64)
    noise_map = fits.getdata(
        os.path.join(folder, "noise_map.fits"), header=False
    ).astype(np.float64)
    return img, psf_kernel, noise_map
