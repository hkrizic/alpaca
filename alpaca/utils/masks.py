"""
Mask utilities for source region definition.
"""

import numpy as np


def make_arc_mask(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    inner_radius: float = 0.3,
    outer_radius: float = 2.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> np.ndarray:
    """
    Create an annular mask for the source region (lensed arc).

    Parameters
    ----------
    xgrid, ygrid : np.ndarray
        Coordinate grids (arcsec).
    inner_radius : float
        Inner radius to mask out lens center (arcsec).
    outer_radius : float
        Outer radius for source region (arcsec).
    center_x, center_y : float
        Mask center coordinates (arcsec).

    Returns
    -------
    mask : np.ndarray
        Boolean mask where True indicates the arc region.
    """
    dist = np.hypot(xgrid - center_x, ygrid - center_y)
    mask = (dist > inner_radius) & (dist < outer_radius)
    return mask.astype(bool)


def load_arc_mask(
    mask_path: str,
    expected_shape: tuple,
) -> np.ndarray:
    """
    Load a custom arc mask from file.

    Parameters
    ----------
    mask_path : str
        Path to mask file (.npy or .fits).
    expected_shape : tuple
        Expected mask shape.

    Returns
    -------
    mask : np.ndarray
        Boolean mask array.
    """
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    elif mask_path.endswith('.fits') or mask_path.endswith('.fit'):
        from astropy.io import fits
        mask = fits.getdata(mask_path)
    else:
        raise ValueError(f"Unsupported mask format: {mask_path}")

    mask = np.asarray(mask, dtype=bool)

    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} doesn't match expected {expected_shape}"
        )

    return mask


def save_mask_visualization(
    img: np.ndarray,
    mask: np.ndarray,
    save_path: str,
    title: str = "Arc Mask Overlay",
    mask_alpha: float = 0.5,
    img_alpha: float = 0.5,
):
    """
    Save visualization of mask overlaid on image.

    Parameters
    ----------
    img : np.ndarray
        Image to overlay mask on.
    mask : np.ndarray
        Boolean mask.
    save_path : str
        Output path for PNG file.
    title : str
        Plot title.
    mask_alpha, img_alpha : float
        Transparency values.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(8, 8))

    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99)

    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, alpha=img_alpha)

    # Create mask overlay
    mask_rgba = np.zeros((*mask.shape, 4))
    rgb = mcolors.to_rgb('red')
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
