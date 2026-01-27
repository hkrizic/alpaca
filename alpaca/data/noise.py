"""Noise boosting utilities around point sources."""

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

    radius_ang = frac_min_sep * min_sep
    rad_pix = radius_ang / pix_scl
    rad_pix_clamped = float(np.clip(rad_pix, min_npix, max_npix))

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
    noise_map = np.asarray(noise_map, float)
    xgrid = np.asarray(xgrid, float)
    ygrid = np.asarray(ygrid, float)
    x_images = np.atleast_1d(np.asarray(x_images, float))
    y_images = np.atleast_1d(np.asarray(y_images, float))

    if x_images.size == 0:
        return noise_map.copy()

    f_max = float(max(f_max, 1.0))

    if radius is None:
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
        w = 1.0 - r / radius
        w = np.clip(w, 0.0, 1.0)
        local_factor = 1.0 + (f_max - 1.0) * w
        factor_map = np.maximum(factor_map, local_factor)

    boosted_noise = noise_map * factor_map
    return boosted_noise
