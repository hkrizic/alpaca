"""
Point source detection and plotter creation utilities.

Provides helpers for creating Herculens ``Plotter`` instances and detecting
bright point-source images in a gravitational lens image under the assumption
that the lens galaxy is centered at the origin of the coordinate grid.

author: hkrizic
"""

import numpy as np
from herculens.Analysis.plot import Plotter
from scipy.ndimage import maximum_filter


def make_plotter(img: np.ndarray) -> Plotter:
    """
    Create a Herculens ``Plotter`` configured from image percentiles.

    The colour scale is set so that *vmin* corresponds to the 0.5th
    percentile (floored at 1e-6) and *vmax* to the 99.7th percentile
    of the image pixel values.

    Parameters
    ----------
    img : np.ndarray
        2-D image array used to determine the colour limits.

    Returns
    -------
    Plotter
        A Herculens ``Plotter`` instance with flux limits and residual
        vmax pre-configured, and the data already attached.
    """
    vmin = max(1e-6, float(np.percentile(img, 0.5)))
    vmax = float(np.percentile(img, 99.7))
    plotter = Plotter(flux_vmin=vmin, flux_vmax=vmax, res_vmax=5)
    plotter.set_data(img)
    return plotter


def detect_ps_images_centered(
    img: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    n_wanted: int = 4,
    lens_mask_radius: float = 0.5,
    local_win: int = 3,
    min_peak_frac: float = 0.15,
    min_sep: float = 0.18,
):
    """
    Detect bright point-source images, assuming lens galaxy is centered.

    Candidates are identified as local maxima outside a central mask and
    above a brightness threshold.  They are then greedily selected in
    descending brightness order, enforcing a minimum angular separation.

    Parameters
    ----------
    img : np.ndarray
        2-D array of the image data.
    xgrid : np.ndarray
        2-D array of x (RA) coordinates of the image pixels.
    ygrid : np.ndarray
        2-D array of y (Dec) coordinates of the image pixels.
    n_wanted : int
        Number of point-source images to detect.
    lens_mask_radius : float
        Radius of the central mask region (around the origin) to exclude
        from detection, in the same units as the coordinate grids.
    local_win : int
        Size of the local window for the maximum filter used to find
        local maxima.
    min_peak_frac : float
        Minimum fraction of the brightest peak for a candidate to be
        considered valid.
    min_sep : float
        Minimum angular separation between detected images, in the same
        units as the coordinate grids.

    Returns
    -------
    picked : list of tuple[int, int]
        List of ``(y, x)`` pixel indices of the detected point-source
        images.
    picked_ra : np.ndarray
        1-D array of RA coordinates of the detected point-source images.
    picked_dec : np.ndarray
        1-D array of Dec coordinates of the detected point-source images.
    picked_flux : np.ndarray
        1-D array of flux values of the detected point-source images.
    """
    ny, nx = img.shape
    dist_ang = np.hypot(xgrid, ygrid) # 2D array of distances from center
    mask_lens = dist_ang < lens_mask_radius # mask with 1 in central lens region and 0 elsewhere

    # Find all local maxima, take only those as cand which are outside the central lens region and above threshold min_peak_frac * value of the brightest peak
    local_max = maximum_filter(img, size=local_win, mode="nearest")
    is_local_max = img == local_max # boolean mask of local maxima
    peak_max = float(img[~mask_lens].max())
    thr = min_peak_frac * peak_max
    cand = np.vstack(np.where(is_local_max & (~mask_lens) & (img >= thr))).T
    if cand.size == 0:
        return [], np.array([]), np.array([]), np.array([])

    cand_vals = img[cand[:, 0], cand[:, 1]] # brightness values of the candidates
    order = np.argsort(cand_vals)[::-1] # sort candidates by brightness descending
    cand = cand[order] # sorted candidates

    picked, picked_ra, picked_dec, picked_flux = [], [], [], []
    for (yy, xx) in cand:
        ra = float(xgrid[yy, xx]) # RA coordinate of the candidate (since xgrid is 2D array of RA coordinates of each pixel)
        dec = float(ygrid[yy, xx]) # Dec coordinate of the candidate (since ygrid is 2D array of Dec coordinates of each pixel)
        fl = float(img[yy, xx]) # brightness value of the candidate (flux)
        ok = True
        for pra, pdec in zip(picked_ra, picked_dec):
            if np.hypot(ra - pra, dec - pdec) < min_sep: # too close to an already picked image
                ok = False
                break
        if not ok:
            continue

        picked.append((int(yy), int(xx)))
        picked_ra.append(ra)
        picked_dec.append(dec)
        picked_flux.append(fl)
        if len(picked) >= n_wanted:
            break

    return picked, np.array(picked_ra), np.array(picked_dec), np.array(picked_flux)
