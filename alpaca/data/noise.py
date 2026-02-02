"""
Noise boosting utilities around point sources.

Provides functions to automatically compute an appropriate boosting radius
and to multiplicatively increase the noise map near detected point-source
positions so that the sampler does not over-fit the PSF wings.

author: hkrizic
"""

from __future__ import annotations

import numpy as np


def auto_noise_boost_radius(
    xgrid: np.ndarray,
    x_images: np.ndarray,
    y_images: np.ndarray,
    frac_min_sep: float = 0.4,
    min_npix: float = 2.5,
    max_npix: float = 6.0,
) -> float:
    """
    Compute the noise-boost radius from point-source image separations.

    The radius is set to a fraction of the minimum pairwise separation
    between point-source images, clamped to lie within a specified range
    in pixel units and then converted back to angular units.  When fewer
    than two images are supplied, a default of 3 pixels is used.

    Parameters
    ----------
    xgrid : np.ndarray
        2-D array of x (RA) coordinates of the image pixels.  Used to
        infer the pixel scale from the median spacing along the first row.
    x_images : np.ndarray
        1-D array of x (RA) coordinates of detected point-source images.
    y_images : np.ndarray
        1-D array of y (Dec) coordinates of detected point-source images.
    frac_min_sep : float
        Fraction of the minimum pairwise separation to use as the
        boosting radius.
    min_npix : float
        Minimum allowed radius in pixel units.
    max_npix : float
        Maximum allowed radius in pixel units.

    Returns
    -------
    float
        Noise-boost radius in angular (arcsec) units.
    """
    xgrid = np.asarray(xgrid, float)
    x_images = np.atleast_1d(np.asarray(x_images, float))
    y_images = np.atleast_1d(np.asarray(y_images, float))

    if xgrid.ndim != 2 or xgrid.shape[1] < 2:
        raise ValueError("xgrid must be a 2D array with at least 2 pixels in x.")

    pix_scl = float(np.median(np.diff(xgrid[0])))

    n_images = x_images.size
    if n_images < 2:
        return float(3.0 * pix_scl)

    seps = []
    for i in range(n_images):
        for j in range(i + 1, n_images):
            dx = x_images[i] - x_images[j]
            dy = y_images[i] - y_images[j]
            seps.append(np.hypot(dx, dy))
    min_sep = float(np.min(seps))

    radius_ang = frac_min_sep * min_sep # radius in angular units (fraction * minimal separation)
    rad_pix = radius_ang / pix_scl # radius in pixel units
    rad_pix_clamped = float(np.clip(rad_pix, min_npix, max_npix)) # clamp radius in pixel units

    return rad_pix_clamped * pix_scl


def boost_noise_around_point_sources(
    noise_map: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    x_images: np.ndarray,
    y_images: np.ndarray,
    radius: float | None = None,
    f_max: float = 5.0,
    frac_min_sep: float = 0.4,
    min_npix: float = 2.5,
    max_npix: float = 6.0,
) -> np.ndarray:
    """
    Multiply the noise map by a linearly-decaying boost around each point source.

    For every detected point-source image, the noise is scaled by a factor
    that equals *f_max* at the image centre and decreases linearly to 1.0
    at distance *radius*.  Where the boost regions of multiple images
    overlap the maximum factor is kept.

    Parameters
    ----------
    noise_map : np.ndarray
        2-D noise (sigma) map with the same shape as the image.
    xgrid : np.ndarray
        2-D array of x (RA) coordinates of the image pixels.
    ygrid : np.ndarray
        2-D array of y (Dec) coordinates of the image pixels.
    x_images : np.ndarray
        1-D array of x (RA) coordinates of detected point-source images.
    y_images : np.ndarray
        1-D array of y (Dec) coordinates of detected point-source images.
    radius : float or None
        Boosting radius in angular units.  If ``None``, it is computed
        automatically via :func:`auto_noise_boost_radius`.
    f_max : float
        Maximum multiplicative factor applied at the centre of each
        point source (clamped to be at least 1.0).
    frac_min_sep : float
        Passed to :func:`auto_noise_boost_radius` when *radius* is None.
    min_npix : float
        Passed to :func:`auto_noise_boost_radius` when *radius* is None.
    max_npix : float
        Passed to :func:`auto_noise_boost_radius` when *radius* is None.

    Returns
    -------
    np.ndarray
        Boosted noise map with the same shape as the input *noise_map*.
    """
    noise_map = np.asarray(noise_map, float)
    xgrid = np.asarray(xgrid, float)
    ygrid = np.asarray(ygrid, float)
    x_images = np.atleast_1d(np.asarray(x_images, float))
    y_images = np.atleast_1d(np.asarray(y_images, float))

    if x_images.size == 0:
        return noise_map.copy()

    f_max = float(max(f_max, 1.0)) # ensure f_max >= 1

    if radius is None:
        # auto-compute radius based on min separation between point-source images
        radius = auto_noise_boost_radius( 
            xgrid=xgrid,
            x_images=x_images,
            y_images=y_images,
            frac_min_sep=frac_min_sep,
            min_npix=min_npix,
            max_npix=max_npix,
        )

    radius = float(radius)
    if radius <= 0:
        return noise_map.copy()

    factor_map = np.ones_like(noise_map, dtype=float)

    for xi, yi in zip(x_images, y_images):
        r = np.hypot(xgrid - xi, ygrid - yi)
        w = 1.0 - r / radius # linear weight decreasing from 1 at r=0 to 0 at r=radius
        w = np.clip(w, 0.0, 1.0) # everything beyond radius gets weight 0
        local_factor = 1.0 + (f_max - 1.0) * w # f_max at r=0, and 1.0 at r>=radius
        factor_map = np.maximum(factor_map, local_factor) # if overlapping regions, take the maximum factor

    boosted_noise = noise_map * factor_map
    return boosted_noise
