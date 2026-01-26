"""
Pixel grid utilities.
"""

from typing import Tuple
import numpy as np
from herculens.Coordinates.pixel_grid import PixelGrid


def make_pixel_grids(
    img: np.ndarray,
    pix_scl: float = 0.08,
    ps_oversample: int = 2,
) -> Tuple[PixelGrid, PixelGrid, np.ndarray, np.ndarray, float]:
    """
    Build the main PixelGrid and a supersampled grid for point sources.

    Parameters
    ----------
    img : np.ndarray
        Observed image (used to determine dimensions).
    pix_scl : float
        Pixel scale in arcsec/pixel.
    ps_oversample : int
        Oversampling factor for point source grid.

    Returns
    -------
    pixel_grid : PixelGrid
        Main pixel grid for imaging.
    ps_grid : PixelGrid
        Supersampled grid for point sources.
    xgrid : np.ndarray
        X coordinate grid (arcsec).
    ygrid : np.ndarray
        Y coordinate grid (arcsec).
    pix_scl : float
        Pixel scale (returned for convenience).
    """
    npix_y, npix_x = img.shape
    assert npix_x == npix_y, "Expect a square image."
    npix = npix_x

    half_size = npix * pix_scl / 2.0
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2.0
    transform_pix2angle = pix_scl * np.eye(2)

    pixel_grid = PixelGrid(
        nx=npix,
        ny=npix,
        ra_at_xy_0=ra_at_xy_0,
        dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=transform_pix2angle,
    )

    # High-res PS grid for subpixel point-source positions
    ps_grid_npix = ps_oversample * npix + 1
    ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix
    ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.0
    ps_grid_ra0 = ps_grid_dec0 = -ps_grid_half_size + ps_grid_pix_scl / 2.0

    ps_grid = PixelGrid(
        nx=ps_grid_npix,
        ny=ps_grid_npix,
        ra_at_xy_0=ps_grid_ra0,
        dec_at_xy_0=ps_grid_dec0,
        transform_pix2angle=ps_grid_pix_scl * np.eye(2),
    )

    xgrid, ygrid = pixel_grid.pixel_coordinates
    return pixel_grid, ps_grid, xgrid, ygrid, pix_scl
