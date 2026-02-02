"""
Arc mask creation and visualization for correlated field source models.

Provides functions to create annular masks that isolate the lensed arc
region, load custom masks from file, and save overlay visualizations.

author: hkrizic
"""

import numpy as np


def make_source_arc_mask(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    inner_radius: float = 0.3,
    outer_radius: float = 2.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Create an annular mask for the source region (lensed arc).

    This mask is used with adaptive_grid=True in PixelatedLight to define
    the region where the source model should be evaluated. The mask excludes
    the central lens galaxy region and limits the outer extent.

    Parameters
    ----------
    xgrid : np.ndarray
        X coordinates grid (arcsec).
    ygrid : np.ndarray
        Y coordinates grid (arcsec).
    inner_radius : float
        Inner radius to mask out the lens center (arcsec).
    outer_radius : float
        Outer radius for the source region (arcsec).
    center_x : float
        X coordinate of the mask center (arcsec).
    center_y : float
        Y coordinate of the mask center (arcsec).

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates the arc region.
    """
    dist = np.hypot(xgrid - center_x, ygrid - center_y)
    mask = (dist > inner_radius) & (dist < outer_radius)
    return mask.astype(bool)


def load_custom_arc_mask(mask_path: str, expected_shape: tuple) -> np.ndarray:
    """
    Load a custom arc mask from file.

    Parameters
    ----------
    mask_path : str
        Path to the mask file (FITS or .npy format).
    expected_shape : tuple
        Expected shape of the mask (should match image shape).

    Returns
    -------
    np.ndarray
        Boolean mask array.
    """
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    elif mask_path.endswith('.fits') or mask_path.endswith('.fit'):
        from astropy.io import fits
        mask = fits.getdata(mask_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}. Use .npy or .fits")

    mask = np.asarray(mask, dtype=bool)

    if mask.shape != expected_shape:
        raise ValueError(
            f"Custom mask shape {mask.shape} doesn't match expected shape {expected_shape}"
        )

    return mask


def save_arc_mask_visualization(
    img: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    title: str = "Arc Mask Overlay",
    cmap: str = "gray",
    mask_color: str = "red",
    mask_alpha: float = 0.5,
    img_alpha: float = 0.5,
    vmin_percentile: float = 1,
    vmax_percentile: float = 99,
):
    """
    Save a visualization of the arc mask overlaid on the image.

    Parameters
    ----------
    img : np.ndarray
        The lens image.
    mask : np.ndarray
        Boolean mask (True = arc region).
    save_path : str
        Path to save the PNG file.
    title : str
        Plot title.
    cmap : str
        Colormap for the image.
    mask_color : str
        Color for the mask overlay.
    mask_alpha : float
        Transparency of the mask (0-1).
    img_alpha : float
        Transparency of the image (0-1).
    vmin_percentile : float
        Lower percentile for image scaling.
    vmax_percentile : float
        Upper percentile for image scaling.

    Returns
    -------
    str
        The path where the PNG file was saved (same as *save_path*).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scale image for display
    vmin = np.nanpercentile(img, vmin_percentile)
    vmax = np.nanpercentile(img, vmax_percentile)

    # Show image with alpha
    ax.imshow(
        img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=img_alpha
    )

    # Create mask overlay (transparent where False, colored where True)
    mask_rgba = np.zeros((*mask.shape, 4))
    # Convert mask_color to RGB
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(mask_color)
    mask_rgba[mask, 0] = rgb[0]
    mask_rgba[mask, 1] = rgb[1]
    mask_rgba[mask, 2] = rgb[2]
    mask_rgba[mask, 3] = mask_alpha

    ax.imshow(mask_rgba, origin='lower')

    ax.set_title(title)
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return save_path
