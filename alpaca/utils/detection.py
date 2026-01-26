"""
Point source detection utilities.
"""

from typing import Tuple, List
import numpy as np
from scipy.ndimage import maximum_filter


def detect_point_sources(
    img: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    n_sources: int = 4,
    lens_mask_radius: float = 0.5,
    local_win: int = 3,
    min_peak_frac: float = 0.15,
    min_sep: float = 0.18,
) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect bright point-source images, assuming lens galaxy is centered.

    Parameters
    ----------
    img : np.ndarray
        Observed image.
    xgrid, ygrid : np.ndarray
        Coordinate grids (arcsec).
    n_sources : int
        Number of point sources to detect.
    lens_mask_radius : float
        Radius to mask lens center (arcsec).
    local_win : int
        Local window size for peak detection.
    min_peak_frac : float
        Minimum peak fraction threshold.
    min_sep : float
        Minimum separation between detected images (arcsec).

    Returns
    -------
    peaks_px : List[Tuple[int, int]]
        Pixel coordinates of detected peaks.
    x_positions : np.ndarray
        RA positions (arcsec).
    y_positions : np.ndarray
        Dec positions (arcsec).
    peak_fluxes : np.ndarray
        Peak flux values.
    """
    ny, nx = img.shape
    dist_ang = np.hypot(xgrid, ygrid)
    mask_lens = dist_ang < lens_mask_radius

    local_max = maximum_filter(img, size=local_win, mode="nearest")
    is_local_max = img == local_max

    peak_max = float(img[~mask_lens].max())
    threshold = min_peak_frac * peak_max

    candidates = np.vstack(
        np.where(is_local_max & (~mask_lens) & (img >= threshold))
    ).T

    if candidates.size == 0:
        return [], np.array([]), np.array([]), np.array([])

    # Sort by brightness
    cand_vals = img[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(cand_vals)[::-1]
    candidates = candidates[order]

    # Select peaks with minimum separation
    picked_px = []
    picked_ra = []
    picked_dec = []
    picked_flux = []

    for (yy, xx) in candidates:
        ra = float(xgrid[yy, xx])
        dec = float(ygrid[yy, xx])
        flux = float(img[yy, xx])

        # Check separation from already picked sources
        ok = True
        for pra, pdec in zip(picked_ra, picked_dec):
            if np.hypot(ra - pra, dec - pdec) < min_sep:
                ok = False
                break
        if not ok:
            continue

        picked_px.append((int(yy), int(xx)))
        picked_ra.append(ra)
        picked_dec.append(dec)
        picked_flux.append(flux)

        if len(picked_px) >= n_sources:
            break

    return (
        picked_px,
        np.array(picked_ra),
        np.array(picked_dec),
        np.array(picked_flux),
    )
