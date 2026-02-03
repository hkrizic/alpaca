"""
Pixel grid construction.

Builds a main ``PixelGrid`` and a supersampled point-source grid for
sub-pixel positioning, following the Herculens conventions.

author: hkrizic
"""

import numpy as np
from herculens.Coordinates.pixel_grid import PixelGrid


def make_pixel_grids(
    img: np.ndarray,
    pix_scl: float = 0.08,
    ps_oversample: int = 2,
):
    """
    Build the main PixelGrid and a supersampled grid for point sources.

    Constructs two Herculens ``PixelGrid`` objects: one at the native
    resolution of the image, and a higher-resolution grid used for
    sub-pixel point-source positioning.  The grids are centered on the
    origin following the Herculens example notebooks.

    Parameters
    ----------
    img : np.ndarray
        2-D square image array whose shape determines the grid size.
    pix_scl : float
        Pixel scale in arcseconds per pixel.
    ps_oversample : int
        Oversampling factor for the point-source grid.  A value of 2
        produces a grid with ``2 * npix + 1`` pixels per side.

    Returns
    -------
    pixel_grid : PixelGrid
        Main pixel grid at the native image resolution.
    ps_grid : PixelGrid
        Supersampled pixel grid for point-source modelling.
    xgrid : np.ndarray
        2-D array of RA coordinates for each pixel in the main grid.
    ygrid : np.ndarray
        2-D array of Dec coordinates for each pixel in the main grid.
    pix_scl : float
        The pixel scale (echoed back for convenience).
    """
    npix_y, npix_x = img.shape
    assert npix_x == npix_y, "Expect a square image."
    npix = npix_x

    half_size = npix * pix_scl / 2.0
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2.0
    transform_pix2angle = pix_scl * np.eye(2) # diagonal matrix with pixel scale

    pixel_grid = PixelGrid(
        nx=npix,
        ny=npix,
        ra_at_xy_0=ra_at_xy_0,
        dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=transform_pix2angle,
    )

    # High-res PS grid for subpixel point-source positions
    ps_grid_npix = ps_oversample * npix + 1 # for ps_oversample = 2: 2n+1 PS grid points for n image grid points
    ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix # adjust pixel scale so that PS grid covers same size as image grid (scale * ratio of npix)
    ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.0
    ps_grid_ra0 = ps_grid_dec0 = -ps_grid_half_size + ps_grid_pix_scl / 2.0
    ps_grid = PixelGrid(
        nx=ps_grid_npix,
        ny=ps_grid_npix,
        ra_at_xy_0=ps_grid_ra0,
        dec_at_xy_0=ps_grid_dec0,
        transform_pix2angle=ps_grid_pix_scl * np.eye(2),
    )

    xgrid, ygrid = pixel_grid.pixel_coordinates # coordinate grid of the image pixels (so that xgrid, ygrid have shape (npix, npix) with RA and Dec coordinate-values of each pixel)
    return pixel_grid, ps_grid, xgrid, ygrid, pix_scl
