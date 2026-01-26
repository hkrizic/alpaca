"""
Combined likelihood: brings together all likelihood terms.

This class provides a unified interface for computing the total log-likelihood,
which can be used by gradient descent, NUTS, and Nautilus.
"""

from typing import Dict, Optional, Callable
import jax.numpy as jnp
import numpy as np

from .imaging import imaging_loglike
from .time_delay import time_delay_loglike
from .rayshoot import rayshoot_consistency_loglike


class CombinedLikelihood:
    """
    Combined likelihood for lens modeling.

    Computes: L_total = L_imaging + L_time_delay + L_rayshoot

    This class is the SINGLE SOURCE OF TRUTH for likelihood computation,
    used by all inference methods.
    """

    def __init__(
        self,
        lens_image,
        data: np.ndarray,
        noise_map: np.ndarray,
        # Time delay settings
        measured_delays: Optional[np.ndarray] = None,
        delay_errors: Optional[np.ndarray] = None,
        # Rayshoot consistency settings
        use_rayshoot_consistency: bool = False,
        rayshoot_sigma_fixed: float = 0.0002,
        use_source_position_rayshoot: bool = True,
        use_rayshoot_systematic_error: bool = False,
    ):
        """
        Initialize combined likelihood.

        Parameters
        ----------
        lens_image : LensImage
            Herculens forward model.
        data : np.ndarray
            Observed image data.
        noise_map : np.ndarray
            Per-pixel noise standard deviation.
        measured_delays : np.ndarray, optional
            Observed time delays relative to image 0 (length n_images - 1).
        delay_errors : np.ndarray, optional
            1-sigma uncertainties on time delays.
        use_rayshoot_consistency : bool
            Enable ray shooting consistency term.
        rayshoot_sigma_fixed : float
            Fixed astrometric uncertainty for rayshoot term (arcsec).
        use_source_position_rayshoot : bool
            Use sampled source position as reference (True) or mean of
            ray-traced positions (False). For Correlated Fields, this
            should be False since there's no src_center_x/y.
        use_rayshoot_systematic_error : bool
            Include systematic error parameter in rayshoot term.
        """
        self.lens_image = lens_image
        self.data = jnp.asarray(data)
        self.noise_map = jnp.asarray(noise_map)
        self.mass_model = lens_image.MassModel

        # Time delay settings
        self.use_time_delays = (
            measured_delays is not None and delay_errors is not None
        )
        if self.use_time_delays:
            self.measured_delays = jnp.asarray(measured_delays)
            self.delay_errors = jnp.asarray(delay_errors)
        else:
            self.measured_delays = None
            self.delay_errors = None

        # Rayshoot settings
        self.use_rayshoot_consistency = use_rayshoot_consistency
        self.rayshoot_sigma_fixed = rayshoot_sigma_fixed
        self.use_source_position_rayshoot = use_source_position_rayshoot
        self.use_rayshoot_systematic_error = use_rayshoot_systematic_error

    def __call__(
        self,
        params: Dict,
        kwargs_model: Dict,
    ) -> jnp.ndarray:
        """
        Compute total log-likelihood.

        Parameters
        ----------
        params : Dict
            Flat parameter dictionary containing:
            - D_dt: time-delay distance (if using time delays)
            - log_sigma_rayshoot_sys: log systematic error (if enabled)
            - src_center_x, src_center_y: source position (if using sampled reference)
        kwargs_model : Dict
            Model kwargs dict with keys:
            - kwargs_lens: lens model parameters
            - kwargs_source: source model parameters
            - kwargs_lens_light: lens light parameters
            - kwargs_point_source: point source parameters

        Returns
        -------
        jnp.ndarray
            Total log-likelihood value.
        """
        # Generate model image
        model_img = self.lens_image.model(**kwargs_model)

        # Imaging likelihood
        ll = imaging_loglike(self.data, model_img, self.noise_map)

        # Time delay likelihood
        if self.use_time_delays:
            D_dt = params["D_dt"]
            x_image = kwargs_model["kwargs_point_source"][0]["ra"]
            y_image = kwargs_model["kwargs_point_source"][0]["dec"]
            kwargs_lens = kwargs_model["kwargs_lens"]

            ll_td = time_delay_loglike(
                D_dt=D_dt,
                x_image=x_image,
                y_image=y_image,
                kwargs_lens=kwargs_lens,
                mass_model=self.mass_model,
                measured_delays=self.measured_delays,
                delay_errors=self.delay_errors,
            )
            ll = ll + ll_td

        # Rayshoot consistency likelihood
        if self.use_rayshoot_consistency:
            x_image = kwargs_model["kwargs_point_source"][0]["ra"]
            y_image = kwargs_model["kwargs_point_source"][0]["dec"]
            kwargs_lens = kwargs_model["kwargs_lens"]

            # Determine reference position
            if self.use_source_position_rayshoot:
                ref_pos = "sampled"
                src_x = params.get("src_center_x")
                src_y = params.get("src_center_y")
            else:
                ref_pos = "mean"
                src_x = None
                src_y = None

            # Determine systematic error
            if self.use_rayshoot_systematic_error:
                log_sigma_sys = params["log_sigma_rayshoot_sys"]
                sigma_sys = jnp.exp(log_sigma_sys)
            else:
                sigma_sys = None

            ll_rayshoot = rayshoot_consistency_loglike(
                x_image=x_image,
                y_image=y_image,
                kwargs_lens=kwargs_lens,
                mass_model=self.mass_model,
                sigma_fixed=self.rayshoot_sigma_fixed,
                sigma_systematic=sigma_sys,
                reference_position=ref_pos,
                src_center_x=src_x,
                src_center_y=src_y,
            )
            ll = ll + ll_rayshoot

        return ll

    def imaging_only(self, kwargs_model: Dict) -> jnp.ndarray:
        """Compute imaging likelihood only (for quick evaluation)."""
        model_img = self.lens_image.model(**kwargs_model)
        return imaging_loglike(self.data, model_img, self.noise_map)

    def as_loss(self) -> Callable:
        """
        Return negative log-likelihood as a loss function.

        Returns a function that takes (params, kwargs_model) and returns
        the loss (negative log-likelihood) for minimization.
        """
        def loss_fn(params: Dict, kwargs_model: Dict) -> jnp.ndarray:
            return -self.__call__(params, kwargs_model)
        return loss_fn
