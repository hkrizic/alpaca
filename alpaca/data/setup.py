"""
High-level lens setup: setup_lens orchestrates grid, detection, mask,
noise, and model creation for a single lens system.
"""

import os

import numpy as np

from alpaca.data.detection import detect_ps_images_centered, make_plotter
from alpaca.data.grids import make_pixel_grids
from alpaca.data.masks import (
    load_custom_arc_mask,
    make_source_arc_mask,
    save_arc_mask_visualization,
)
from alpaca.data.noise import boost_noise_around_point_sources


def setup_lens(
    img: np.ndarray,
    psf_kernel: np.ndarray,
    noise_map: np.ndarray,
    *,
    n_ps_detect: int = 4,
    pix_scl: float = 0.08,
    ps_oversample: int = 2,
    lens_mask_radius: float = 0.5,
    local_win: int = 3,
    min_peak_frac: float = 0.15,
    min_sep: float = 0.18,
    supersampling_factor: int = 5,
    convolution_type: str = "jax_scipy_fft",
    use_source_shapelets: bool = False,
    shapelets_n_max: int = 6,
    boost_noise_around_ps: bool = False,
    boost_kwargs: dict | None = None,
    use_rayshoot_systematic_error: bool = False,
    rayshoot_sys_error_min: float = 0.00005,
    rayshoot_sys_error_max: float = 0.005,
    # Correlated Fields parameters
    use_corr_fields: bool = False,
    corr_field_num_pixels: int = 80,
    corr_field_mean_intensity: float | None = None,
    corr_field_offset_std: tuple[float, float] = (0.5, 1e-3),
    corr_field_loglogavgslope: tuple[float, float] = (-6., 0.5),
    corr_field_fluctuations: tuple[float, float] = (1., 0.5),
    corr_field_flexibility: tuple[float, float] | None = None,
    corr_field_asperity: tuple[float, float] | None = None,
    corr_field_cropped_border_size: int = 20,
    corr_field_exponentiate: bool = True,
    corr_field_interpolation: str = 'fast_bilinear',
    arc_mask_inner_radius: float = 0.3,
    arc_mask_outer_radius: float = 2.5,
    custom_arc_mask_path: str | None = None,
    output_dir: str | None = None,
) -> dict:
    """
    High-level convenience wrapper that builds everything needed for inference.

    Parameters
    ----------
    img : np.ndarray
        2-D lens image array.
    psf_kernel : np.ndarray
        2-D PSF kernel array.
    noise_map : np.ndarray
        2-D noise map array (same shape as *img*).
    n_ps_detect : int
        Number of point sources to detect.
    pix_scl : float
        Pixel scale in arcsec.
    ps_oversample : int
        Point source grid oversampling factor.
    lens_mask_radius : float
        Radius to mask lens center for PS detection.
    local_win : int
        Local window for peak detection.
    min_peak_frac : float
        Minimum peak fraction threshold.
    min_sep : float
        Minimum separation between detected images.
    supersampling_factor : int
        Supersampling factor for source model.
    convolution_type : str
        PSF convolution method.
    use_source_shapelets : bool
        Use Shapelets basis for source (mutually exclusive with use_corr_fields).
    shapelets_n_max : int
        Maximum shapelet order.
    boost_noise_around_ps : bool
        Boost noise around detected point sources.
    boost_kwargs : dict, optional
        Additional kwargs for noise boosting.
    use_rayshoot_systematic_error : bool
        Include ray shooting systematic error.
    rayshoot_sys_error_min, rayshoot_sys_error_max : float
        Bounds for systematic error.
    use_corr_fields : bool
        Use Correlated Fields for source (mutually exclusive with use_source_shapelets).
    corr_field_num_pixels : int
        Number of pixels per side for source grid.
    corr_field_mean_intensity : float, optional
        Mean source intensity (estimated from data if None).
    corr_field_offset_std : tuple
        Prior for offset std (mean, std).
    corr_field_loglogavgslope : tuple
        Prior for smoothness (mean, std). More negative = smoother.
    corr_field_fluctuations : tuple
        Prior for fluctuations (mean, std).
    corr_field_flexibility, corr_field_asperity : tuple, optional
        Additional GP priors.
    corr_field_cropped_border_size : int
        Border cropping for FFT.
    corr_field_exponentiate : bool
        Exponentiate field for positivity.
    corr_field_interpolation : str
        Interpolation method for pixelated source.
    arc_mask_inner_radius, arc_mask_outer_radius : float
        Radii for arc mask (arcsec).

    Returns
    -------
    Dict
        Setup dictionary with all components for inference.
    """
    # Local imports to avoid circular dependencies:
    # prob_model imports nothing from data.setup, but data.setup creates
    # ProbModel / ProbModelCorrField instances, so we import here.
    from alpaca.models.prob_model import (
        ProbModel,
        ProbModelCorrField,
        create_corr_field,
        make_lens_image,
    )

    # Validate mutual exclusivity
    if use_source_shapelets and use_corr_fields:
        raise ValueError(
            "Cannot use both Shapelets and Correlated Fields. "
            "Set only one of use_source_shapelets or use_corr_fields to True."
        )

    pixel_grid, ps_grid, xgrid, ygrid, pix_scl = make_pixel_grids(
        img, pix_scl=pix_scl, ps_oversample=ps_oversample
    )

    # Detect point-source images
    peaks_px, x0s, y0s, peak_vals = detect_ps_images_centered(
        img,
        xgrid,
        ygrid,
        n_wanted=n_ps_detect,
        lens_mask_radius=lens_mask_radius,
        local_win=local_win,
        min_peak_frac=min_peak_frac,
        min_sep=min_sep,
    )

    if boost_noise_around_ps:
        _kwargs = dict(
            radius=None,
            f_max=5.0,
            frac_min_sep=0.4,
            min_npix=2.5,
            max_npix=6.0,
        )
        if boost_kwargs is not None:
            _kwargs.update(boost_kwargs)

        noise_map = boost_noise_around_point_sources(
            noise_map=noise_map,
            xgrid=xgrid,
            ygrid=ygrid,
            x_images=x0s,
            y_images=y0s,
            **_kwargs,
        )

    # Create arc mask for correlated fields
    source_arc_mask = None
    if use_corr_fields:
        if custom_arc_mask_path is not None:
            # Load custom mask from file
            source_arc_mask = load_custom_arc_mask(
                mask_path=custom_arc_mask_path,
                expected_shape=img.shape,
            )
            print(f"Loaded custom arc mask from: {custom_arc_mask_path}")
        else:
            # Create annular mask
            source_arc_mask = make_source_arc_mask(
                xgrid=xgrid,
                ygrid=ygrid,
                inner_radius=arc_mask_inner_radius,
                outer_radius=arc_mask_outer_radius,
            )

        # Save mask visualization if output directory provided
        if output_dir is not None:
            mask_vis_path = os.path.join(output_dir, "arc_mask_overlay.png")
            save_arc_mask_visualization(
                img=img,
                mask=source_arc_mask,
                save_path=mask_vis_path,
                title=f"Arc Mask (inner={arc_mask_inner_radius}\", outer={arc_mask_outer_radius}\")"
                if custom_arc_mask_path is None else "Custom Arc Mask",
                mask_alpha=0.5,
                img_alpha=0.5,
            )
            print(f"Saved arc mask visualization to: {mask_vis_path}")

    # Build lens image with appropriate source model
    (
        lens_image,
        noise,
        psf,
        mass_model,
        lens_light_model,
        source_light_model,
        point_source_model,
    ) = make_lens_image(
        img,
        psf_kernel,
        noise_map,
        pixel_grid,
        ps_grid,
        supersampling_factor=supersampling_factor,
        convolution_type=convolution_type,
        use_source_shapelets=use_source_shapelets,
        shapelets_n_max=shapelets_n_max,
        use_corr_fields=use_corr_fields,
        corr_field_num_pixels=corr_field_num_pixels,
        corr_field_interpolation=corr_field_interpolation,
        source_arc_mask=source_arc_mask,
    )

    plotter = make_plotter(img)

    # Create the appropriate probabilistic model
    source_field = None
    if use_corr_fields:
        # Create CorrelatedField and ProbModelCorrField
        source_field = create_corr_field(
            source_model=source_light_model,
            img=img,
            num_pixels=corr_field_num_pixels,
            mean_intensity=corr_field_mean_intensity,
            offset_std=corr_field_offset_std,
            loglogavgslope=corr_field_loglogavgslope,
            fluctuations=corr_field_fluctuations,
            flexibility=corr_field_flexibility,
            asperity=corr_field_asperity,
            cropped_border_size=corr_field_cropped_border_size,
            exponentiate=corr_field_exponentiate,
        )
        prob_model = ProbModelCorrField(
            lens_image=lens_image,
            img=img,
            noise_map=noise_map,
            xgrid=xgrid,
            ygrid=ygrid,
            x0s=x0s,
            y0s=y0s,
            peak_vals=peak_vals,
            source_field=source_field,
            num_pixels=corr_field_num_pixels,
            use_rayshoot_systematic_error=use_rayshoot_systematic_error,
            rayshoot_sys_error_min=rayshoot_sys_error_min,
            rayshoot_sys_error_max=rayshoot_sys_error_max,
        )
    else:
        # Use standard ProbModel (with or without Shapelets)
        prob_model = ProbModel(
            lens_image=lens_image,
            img=img,
            noise_map=noise_map,
            xgrid=xgrid,
            ygrid=ygrid,
            x0s=x0s,
            y0s=y0s,
            peak_vals=peak_vals,
            use_source_shapelets=use_source_shapelets,
            shapelets_n_max=shapelets_n_max,
            use_rayshoot_systematic_error=use_rayshoot_systematic_error,
            rayshoot_sys_error_min=rayshoot_sys_error_min,
            rayshoot_sys_error_max=rayshoot_sys_error_max,
        )

    return dict(
        img=img,
        psf_kernel=psf_kernel,
        noise_map=noise_map,
        pixel_grid=pixel_grid,
        ps_grid=ps_grid,
        xgrid=xgrid,
        ygrid=ygrid,
        peaks_px=peaks_px,
        x0s=x0s,
        y0s=y0s,
        peak_vals=peak_vals,
        lens_image=lens_image,
        plotter=plotter,
        prob_model=prob_model,
        pix_scl=pix_scl,
        # Correlated Fields specific
        source_field=source_field,
        source_arc_mask=source_arc_mask,
        use_corr_fields=use_corr_fields,
    )
