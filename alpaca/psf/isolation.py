"""
Point source isolation and cutout extraction for PSF reconstruction.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndimage_shift

from alpaca.psf.utils import _build_model_with_single_ps_zeroed, _circular_mask_centered


def isolate_point_sources(model_with_ps: np.ndarray, model_without_ps: np.ndarray) -> np.ndarray:
    return np.asarray(model_with_ps) - np.asarray(model_without_ps)


def generate_isolated_ps_images(
    data_image: np.ndarray,
    prob_model,
    best_params: dict,
    n_point_sources: int = 4,
) -> list[np.ndarray]:
    """
    Generate isolated point source images:
        isolated_i = data - model(with PS_i amp = 0)

    Each returned image is the FULL frame (e.g. 99x99) and is NOT shifted/recentered.
    """
    data = np.asarray(data_image)
    isolated_images: list[np.ndarray] = []
    for i in range(int(n_point_sources)):
        model_without_ps_i = _build_model_with_single_ps_zeroed(prob_model, best_params, ps_index=i)
        isolated_images.append(data - np.asarray(model_without_ps_i))
    return isolated_images


# ----------------------------------------------------------------------------- #
# Centered cutouts from isolated images (STARRED-compatible)
# ----------------------------------------------------------------------------- #
def build_centered_cutouts_from_isolated(
    isolated_images: list[np.ndarray],
    peaks_px: np.ndarray,
    cutout_size: int,
    *,
    mask_other_peaks: bool = True,
    mask_radius: int = 8,
    global_mask_map: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract CENTERED cutouts from isolated images for STARRED.

    Each cutout is centered on its corresponding PS position, so that
    the PS appears at the center of each stamp. This is required by STARRED's
    internal noise propagation which shifts maps by -x0, -y0.

    Parameters
    ----------
    isolated_images : List[np.ndarray]
        List of isolated images, one per point source.
    peaks_px : np.ndarray
        Array of (y, x) pixel coordinates for each point source.
    cutout_size : int
        Size of square cutouts (must be odd).
    mask_other_peaks : bool
        If True, mask positions of other point sources in each cutout.
    mask_radius : int
        Radius for masking other point sources.
    global_mask_map : np.ndarray | None
        Optional global mask in full-frame coordinates.

    Returns
    -------
    cutouts : (N, cutout_size, cutout_size) float
        Centered cutouts, one per PS.
    masks : (N, cutout_size, cutout_size) float or None
        1=keep, 0=mask. None if no masking requested.
    """
    peaks_px = np.asarray(peaks_px, float)
    N = len(isolated_images)

    if len(peaks_px) != N:
        raise ValueError(f"Number of isolated images ({N}) must match number of peaks ({len(peaks_px)})")

    # Ensure odd cutout size
    if cutout_size % 2 == 0:
        cutout_size += 1

    r = cutout_size // 2

    # Get image shape from first isolated image
    img_shape = np.asarray(isolated_images[0]).shape
    ny, nx = img_shape

    cutouts: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []

    for j, (iso_img, (py, px)) in enumerate(zip(isolated_images, peaks_px)):
        iso_img = np.asarray(iso_img, float)
        py = float(py)
        px = float(px)
        y_int, x_int = int(py), int(px)
        shift_y = -(py - y_int)
        shift_x = -(px - x_int)

        # Extract cutout centered on PS position
        y_min = max(0, y_int - r)
        y_max = min(ny, y_int + r + 1)
        x_min = max(0, x_int - r)
        x_max = min(nx, x_int + r + 1)

        cutout = iso_img[y_min:y_max, x_min:x_max].copy()

        # Pad if near edge to ensure constant size
        if cutout.shape != (cutout_size, cutout_size):
            pad_y = (r - (y_int - y_min), r + 1 - (y_max - y_int))
            pad_x = (r - (x_int - x_min), r + 1 - (x_max - x_int))
            cutout = np.pad(cutout, (pad_y, pad_x), mode="constant", constant_values=0)

        if shift_y != 0.0 or shift_x != 0.0:
            cutout = ndimage_shift(
                cutout,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=0.0,
            )

        cutouts.append(cutout)

        # Build mask for this cutout
        mask = np.ones((cutout_size, cutout_size), dtype=float)

        # Apply global mask if provided (transform to cutout coordinates)
        if global_mask_map is not None:
            gm = np.asarray(global_mask_map)
            gm_cutout = gm[y_min:y_max, x_min:x_max].copy()
            if gm_cutout.shape != (cutout_size, cutout_size):
                gm_cutout = np.pad(gm_cutout, (pad_y, pad_x), mode="constant", constant_values=0)
            if shift_y != 0.0 or shift_x != 0.0:
                gm_cutout = ndimage_shift(
                    gm_cutout,
                    shift=(shift_y, shift_x),
                    order=0,
                    mode="constant",
                    cval=0.0,
                )
            mask *= (gm_cutout > 0.5).astype(float)

        # Mask other PS positions (in cutout coordinates)
        if mask_other_peaks:
            for i, (qy, qx) in enumerate(peaks_px):
                if i == j:
                    continue
                # Position of other PS relative to this cutout's center
                cy_in_cutout = r + (float(qy) - py)
                cx_in_cutout = r + (float(qx) - px)

                # Only mask if within cutout bounds
                if 0 <= cy_in_cutout < cutout_size and 0 <= cx_in_cutout < cutout_size:
                    mask *= _circular_mask_centered(cutout_size, cy_in_cutout, cx_in_cutout, mask_radius)

        masks_list.append(mask)

    cutouts_arr = np.asarray(cutouts, dtype=float)

    need_masks = mask_other_peaks or (global_mask_map is not None)
    if need_masks:
        masks_arr = np.asarray(masks_list, dtype=float)
        return cutouts_arr, masks_arr

    return cutouts_arr, None


def build_centered_noise_cutouts(
    noise_map: np.ndarray,
    peaks_px: np.ndarray,
    cutout_size: int,
    *,
    noise_map_is_sigma: bool = True,
) -> np.ndarray:
    """
    Extract centered noise cutouts (as sigma^2) matching the PS cutouts.

    Parameters
    ----------
    noise_map : np.ndarray
        Full-frame noise map.
    peaks_px : np.ndarray
        Array of (y, x) pixel coordinates for each point source.
    cutout_size : int
        Size of square cutouts (must be odd).
    noise_map_is_sigma : bool
        If True, noise_map contains sigma (std dev). If False, contains variance.

    Returns
    -------
    sigma2_cutouts : (N, cutout_size, cutout_size) float
        Variance (sigma^2) cutouts.
    """
    peaks_px = np.asarray(peaks_px, float)
    noise_map = np.asarray(noise_map, float)
    ny, nx = noise_map.shape

    if cutout_size % 2 == 0:
        cutout_size += 1

    r = cutout_size // 2

    cutouts: list[np.ndarray] = []

    for (py, px) in peaks_px:
        py = float(py)
        px = float(px)
        y_int, x_int = int(py), int(px)
        shift_y = -(py - y_int)
        shift_x = -(px - x_int)

        y_min = max(0, y_int - r)
        y_max = min(ny, y_int + r + 1)
        x_min = max(0, x_int - r)
        x_max = min(nx, x_int + r + 1)

        cutout = noise_map[y_min:y_max, x_min:x_max].copy()
        pad_val = np.median(cutout[cutout > 0]) if np.any(cutout > 0) else 1.0

        if cutout.shape != (cutout_size, cutout_size):
            pad_y = (r - (y_int - y_min), r + 1 - (y_max - y_int))
            pad_x = (r - (x_int - x_min), r + 1 - (x_max - x_int))
            # Pad with median noise value
            cutout = np.pad(cutout, (pad_y, pad_x), mode="constant", constant_values=pad_val)

        if shift_y != 0.0 or shift_x != 0.0:
            cutout = ndimage_shift(
                cutout,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=pad_val,
            )

        cutouts.append(cutout)

    cutouts_arr = np.asarray(cutouts, dtype=float)

    # Convert to variance if input is sigma
    if noise_map_is_sigma:
        cutouts_arr = cutouts_arr ** 2

    return cutouts_arr
