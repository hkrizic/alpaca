"""
Setup utilities for lens modeling.

Provides convenience functions to set up a complete lens modeling system.
"""

from typing import Dict, Optional
import numpy as np

from ..config import PipelineConfig
from ..data import tdlmc_paths, load_image
from ..utils import (
    make_pixel_grids,
    detect_point_sources,
    boost_noise_around_point_sources,
    make_arc_mask,
    load_arc_mask,
    save_mask_visualization,
)
from .lens_image import make_lens_image, create_corr_field
from .prob_model import ProbModel
from .prob_model_corrfield import ProbModelCorrField


def setup_lens_system(config: PipelineConfig) -> Dict:
    """
    Set up a complete lens modeling system from configuration.

    This is the main entry point for creating all the components needed
    for lens modeling.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    Dict
        Dictionary containing:
        - data_folder, output_dir: paths
        - img, psf_kernel, noise_map: image data
        - pixel_grid, ps_grid, xgrid, ygrid: coordinate grids
        - x0s, y0s, peak_vals: detected point sources
        - lens_image: Herculens LensImage object
        - prob_model: ProbModel or ProbModelCorrField
        - source_field: CorrelatedField (if using corr fields)
        - source_arc_mask: arc mask (if using corr fields)
    """
    config.validate()

    # Load data
    data_folder, output_dir = tdlmc_paths(
        config.base_path, config.rung, config.code_id, config.seed
    )
    img, psf_kernel, noise_map = load_image(data_folder)

    # Build pixel grids
    pixel_grid, ps_grid, xgrid, ygrid, pix_scl = make_pixel_grids(
        img, pix_scl=config.pix_scl, ps_oversample=config.ps_oversample
    )

    # Detect point sources
    peaks_px, x0s, y0s, peak_vals = detect_point_sources(
        img, xgrid, ygrid,
        n_sources=config.n_ps_detect,
        lens_mask_radius=config.lens_mask_radius,
        local_win=config.local_win,
        min_peak_frac=config.min_peak_frac,
        min_sep=config.min_sep,
    )

    # Store original noise map BEFORE boosting (needed for PSF reconstruction)
    noise_map_original = np.copy(noise_map)

    # Optionally boost noise around point sources
    if config.boost_noise_around_ps:
        noise_map = boost_noise_around_point_sources(
            noise_map, xgrid, ygrid, x0s, y0s,
            f_max=config.boost_noise_f_max,
            frac_min_sep=config.boost_noise_frac_min_sep,
            min_npix=config.boost_noise_min_npix,
            max_npix=config.boost_noise_max_npix,
        )

    # Create arc mask for correlated fields
    source_arc_mask = None
    if config.use_corr_fields:
        if config.custom_arc_mask_path is not None:
            source_arc_mask = load_arc_mask(
                config.custom_arc_mask_path, img.shape
            )
        else:
            source_arc_mask = make_arc_mask(
                xgrid, ygrid,
                inner_radius=config.arc_mask_inner_radius,
                outer_radius=config.arc_mask_outer_radius,
            )

        # Save mask visualization
        save_mask_visualization(
            img, source_arc_mask,
            f"{output_dir}/arc_mask_overlay.png",
            title=f"Arc Mask (r_in={config.arc_mask_inner_radius}\", r_out={config.arc_mask_outer_radius}\")",
        )

    # Build lens image
    (
        lens_image, noise, psf, mass_model,
        lens_light_model, source_light_model, point_source_model,
    ) = make_lens_image(
        img, psf_kernel, noise_map, pixel_grid, ps_grid,
        supersampling_factor=config.supersampling_factor,
        convolution_type=config.convolution_type,
        use_source_shapelets=config.use_source_shapelets,
        shapelets_n_max=config.shapelets_n_max,
        use_corr_fields=config.use_corr_fields,
        corr_field_num_pixels=config.corr_field_num_pixels,
        corr_field_interpolation=config.corr_field_interpolation,
        source_arc_mask=source_arc_mask,
    )

    # Prepare time delay data
    measured_delays = None
    delay_errors = None
    if config.use_time_delays and config.measured_delays is not None:
        measured_delays = np.array(config.measured_delays)
        delay_errors = np.array(config.delay_errors)

    # Create probabilistic model
    source_field = None
    if config.use_corr_fields:
        source_field = create_corr_field(
            source_light_model, img,
            num_pixels=config.corr_field_num_pixels,
            mean_intensity=config.corr_field_mean_intensity,
            offset_std=config.corr_field_offset_std,
            loglogavgslope=config.corr_field_loglogavgslope,
            fluctuations=config.corr_field_fluctuations,
            flexibility=config.corr_field_flexibility,
            asperity=config.corr_field_asperity,
            cropped_border_size=config.corr_field_cropped_border_size,
            exponentiate=config.corr_field_exponentiate,
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
            num_pixels=config.corr_field_num_pixels,
            measured_delays=measured_delays,
            delay_errors=delay_errors,
            use_rayshoot_consistency=config.use_rayshoot_consistency,
            rayshoot_consistency_sigma=config.rayshoot_consistency_sigma,
            use_rayshoot_systematic_error=config.use_rayshoot_systematic_error,
            rayshoot_sys_error_min=config.rayshoot_sys_error_min,
            rayshoot_sys_error_max=config.rayshoot_sys_error_max,
            D_dt_min=config.D_dt_min,
            D_dt_max=config.D_dt_max,
        )
    else:
        prob_model = ProbModel(
            lens_image=lens_image,
            img=img,
            noise_map=noise_map,
            xgrid=xgrid,
            ygrid=ygrid,
            x0s=x0s,
            y0s=y0s,
            peak_vals=peak_vals,
            use_source_shapelets=config.use_source_shapelets,
            shapelets_n_max=config.shapelets_n_max,
            shapelets_beta_min=config.shapelets_beta_min,
            shapelets_beta_max=config.shapelets_beta_max,
            shapelets_amp_sigma=config.shapelets_amp_sigma,
            measured_delays=measured_delays,
            delay_errors=delay_errors,
            use_rayshoot_consistency=config.use_rayshoot_consistency,
            rayshoot_consistency_sigma=config.rayshoot_consistency_sigma,
            use_source_position_rayshoot=config.use_source_position_rayshoot,
            use_rayshoot_systematic_error=config.use_rayshoot_systematic_error,
            rayshoot_sys_error_min=config.rayshoot_sys_error_min,
            rayshoot_sys_error_max=config.rayshoot_sys_error_max,
            D_dt_min=config.D_dt_min,
            D_dt_max=config.D_dt_max,
        )

    return dict(
        data_folder=data_folder,
        output_dir=output_dir,
        img=img,
        psf_kernel=psf_kernel,
        noise_map=noise_map,
        noise_map_original=noise_map_original,  # Original noise map before boosting (for PSF reconstruction)
        pixel_grid=pixel_grid,
        ps_grid=ps_grid,
        xgrid=xgrid,
        ygrid=ygrid,
        pix_scl=pix_scl,
        peaks_px=peaks_px,
        x0s=x0s,
        y0s=y0s,
        peak_vals=peak_vals,
        lens_image=lens_image,
        prob_model=prob_model,
        source_field=source_field,
        source_arc_mask=source_arc_mask,
    )
