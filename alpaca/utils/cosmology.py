"""
Cosmology utility functions.

Utilities for strong lens modeling workflows. Provides cosmological distance
calculations, time delay predictions, and H0 conversions.

Main components:
    - compute_D_dt / predict_time_delay: Cosmological time delay calculations
    - Ddt_2_H0 / Dd_2_H0: Convert distances to H0 constraints

Author: martin-millon, hkrizic
"""

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import scipy
from astropy import constants as cst


def cast_ints_to_floats_in_dict(d):
    """Recursively convert all integers in a nested dict/list to floats.

    Args:
        d: Input value (dict, list, int, float, str, or None).

    Returns:
        Same structure with all int values converted to float.

    Raises:
        TypeError: If an unsupported type is encountered.
    """
    if isinstance(d, dict):
        return {k: cast_ints_to_floats_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [cast_ints_to_floats_in_dict(item) for item in d]
    elif isinstance(d, int):
        return float(d)
    elif isinstance(d, (str, float)):
        return d
    elif d is None:
        return None
    else:
        raise TypeError(f"Unsupported type: {type(d)} with value: {d}")


def compute_D_dt(z_lens, z_source, cosmology):
    """Compute the time-delay distance D_dt.

    Args:
        z_lens: Redshift of the lens galaxy.
        z_source: Redshift of the background source.
        cosmology: Astropy cosmology object (e.g., FlatLambdaCDM).

    Returns:
        Time-delay distance in Mpc.
    """
    D_l = cosmology.angular_diameter_distance(z_lens).to(u.Mpc).value
    D_s = cosmology.angular_diameter_distance(z_source).to(u.Mpc).value
    D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).to(u.Mpc).value
    D_dt = (1 + z_lens) * D_l * D_s / D_ls
    return D_dt


def predict_time_delay(delta_phi, z_lens, z_source, cosmology=None, D_dt=None):
    """Predict time delay from Fermat potential difference.

    Args:
        delta_phi: Fermat potential difference between images (arcsec^2).
        z_lens: Lens redshift.
        z_source: Source redshift.
        cosmology: Astropy cosmology object. Required if D_dt not provided.
        D_dt: Time-delay distance in Mpc. If None, computed from cosmology.

    Returns:
        Predicted time delay in days.

    Raises:
        ValueError: If neither cosmology nor D_dt is provided.
    """
    Mpc_to_m = 3.085677581491367e22
    s_to_days = 86400

    if cosmology is None and D_dt is None:
        raise ValueError("Either cosmology or D_dt must be provided.")
    if D_dt is None:
        D_dt = compute_D_dt(z_lens, z_source, cosmology)

    delta_phi = (delta_phi * u.arcsec**2).to(u.rad**2).value
    delta_t = (D_dt / cst.c.to(u.m / u.s).value * delta_phi) * Mpc_to_m / s_to_days
    return delta_t


def model_Ddt_with_pred_samples(pred_fermat_pot_samples, obs_delays, obs_errors, z_lens, z_source):
    """NumPyro model for D_dt inference marginalizing over Fermat potential samples.

    Uses Monte Carlo integration to marginalize over predicted Fermat potential
    samples when computing the likelihood of observed time delays.

    Args:
        pred_fermat_pot_samples: Array of shape (S, N) with S samples of N delay predictions.
        obs_delays: Observed time delays, shape (N,).
        obs_errors: Observational uncertainties, shape (N,).
        z_lens: Lens redshift.
        z_source: Source redshift.
    """
    Ddt = numpyro.sample("D_dt", dist.Uniform(0.0, 15000.0))
    predicted_delays = predict_time_delay(pred_fermat_pot_samples, z_lens, z_source, D_dt=Ddt)

    def log_prob_single(pred):
        return jnp.sum(dist.Normal(pred, obs_errors).log_prob(obs_delays))

    log_liks = jax.vmap(log_prob_single)(predicted_delays)
    log_marginal = jax.scipy.special.logsumexp(log_liks) - jnp.log(predicted_delays.shape[0])
    numpyro.factor("log_marginal_likelihood", log_marginal)


def integrand(z, omegaM, omegaL):
    """Integrand for comoving distance calculation in FLRW cosmology.

    Args:
        z: Redshift.
        omegaM: Matter density parameter.
        omegaL: Dark energy density parameter.

    Returns:
        1/E(z) where E(z) is the dimensionless Hubble parameter.

    Raises:
        RuntimeError: If the denominator becomes negative (unphysical cosmology).
    """
    omegaM = float(omegaM)
    omegaL = float(omegaL)
    denom = (1.0 - omegaM - omegaL) * (1.0 + z)**2 + omegaM * (1.0 + z)**3 + omegaL
    if denom < 0:
        raise RuntimeError("'denom < 0' in integrand")
    return 1.0 / np.sqrt(denom)


def Ddt_2_H0(Ddt, z_lens, z_source, Omega_M, Omega_L):
    """Convert time-delay distance to H0.

    Args:
        Ddt: Time-delay distance in Mpc.
        z_lens: Lens redshift.
        z_source: Source redshift.
        Omega_M: Matter density parameter.
        Omega_L: Dark energy density parameter.

    Returns:
        Hubble constant H0 in km/s/Mpc.
    """
    c = cst.c
    A = (scipy.integrate.quad(integrand, 0.0, z_source, args=(Omega_M, Omega_L))[0] *
         scipy.integrate.quad(integrand, 0, z_lens, args=(Omega_M, Omega_L))[0]) / \
         scipy.integrate.quad(integrand, z_lens, z_source, args=(Omega_M, Omega_L))[0]
    return (c.to('km/s').value * A) / Ddt


def Dd_2_H0(Dd, z_lens, Omega_M, Omega_L):
    """Convert angular diameter distance to lens to H0.

    Args:
        Dd: Angular diameter distance to lens in Mpc.
        z_lens: Lens redshift.
        Omega_M: Matter density parameter.
        Omega_L: Dark energy density parameter.

    Returns:
        Hubble constant H0 in km/s/Mpc.
    """
    c = cst.c
    A = scipy.integrate.quad(integrand, 0.0, z_lens, args=(Omega_M, Omega_L))[0]
    return (c.to('km/s').value * A) / (Dd * (1 + z_lens))


