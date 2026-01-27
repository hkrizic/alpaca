"""Point source detection and plotter creation utilities."""

import numpy as np
from herculens.Analysis.plot import Plotter
from scipy.ndimage import maximum_filter


def make_plotter(img: np.ndarray) -> Plotter:
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
    """Detect bright point-source images, assuming lens galaxy is centered."""
    ny, nx = img.shape
    dist_ang = np.hypot(xgrid, ygrid)
    mask_lens = dist_ang < lens_mask_radius

    local_max = maximum_filter(img, size=local_win, mode="nearest")
    is_local_max = img == local_max

    peak_max = float(img[~mask_lens].max())
    thr = min_peak_frac * peak_max

    cand = np.vstack(np.where(is_local_max & (~mask_lens) & (img >= thr))).T
    if cand.size == 0:
        return [], np.array([]), np.array([]), np.array([])

    cand_vals = img[cand[:, 0], cand[:, 1]]
    order = np.argsort(cand_vals)[::-1]
    cand = cand[order]

    picked, picked_ra, picked_dec, picked_flux = [], [], [], []
    for (yy, xx) in cand:
        ra = float(xgrid[yy, xx])
        dec = float(ygrid[yy, xx])
        fl = float(img[yy, xx])
        ok = True
        for pra, pdec in zip(picked_ra, picked_dec):
            if np.hypot(ra - pra, dec - pdec) < min_sep:
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
