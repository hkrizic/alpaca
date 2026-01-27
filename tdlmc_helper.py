"""
TDLMC-specific helpers for the Alpaca pipeline.

This module contains all Time Delay Lens Modeling Challenge (TDLMC) specific
logic that was decoupled from the generic alpaca package. It provides:

- TDC path construction and data loading
- TDC truth file parsing
- Time delay data loading and matching
- Convenience functions to prepare TDC data for the generic pipeline
"""

from __future__ import annotations

import ast
import itertools
import os
import re

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------------
# Path construction and image loading (from alpaca/data/loader.py)
# ---------------------------------------------------------------------------

def tdlmc_paths(base: str, rung: int, code_id: int, seed: int) -> tuple[str, str]:
    """Return (folder, outdir) paths for the drizzled TDLMC image and results."""
    code = f"code{code_id}"
    folder = os.path.join(
        base,
        f"TDC/rung{rung}/{code}/f160w-seed{seed}/drizzled_image",
    )
    outdir = os.path.join(
        base,
        f"TDC_results/rung{rung}/{code}/f160w-seed{seed}",
    )
    os.makedirs(outdir, exist_ok=True)
    return folder, outdir


def load_tdlmc_image(folder: str):
    """Load image, PSF kernel and noise map from a TDLMC drizzled_image folder."""
    img = fits.getdata(os.path.join(folder, "lens-image.fits"), header=False).astype(
        np.float64
    )
    psf_kernel = fits.getdata(
        os.path.join(folder, "psf.fits"), header=False
    ).astype(np.float64)
    noise_map = fits.getdata(
        os.path.join(folder, "noise_map.fits"), header=False
    ).astype(np.float64)
    return img, psf_kernel, noise_map


# ---------------------------------------------------------------------------
# TDC truth/info file parsing (from alpaca/utils/cosmology.py)
# ---------------------------------------------------------------------------

def cast_ints_to_floats_in_dict(d):
    """Recursively convert all integers in a nested dict/list to floats."""
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


def parse_lens_info_file(filepath):
    """Parse a TDC 'evil team' lens info text file."""
    def extract_value(pattern, text, eval_type=float, default=None):
        match = re.search(pattern, text)
        if match:
            try:
                return eval_type(match.group(1))
            except Exception:
                return default
        return default

    with open(filepath) as f:
        content = f.read()

    lens_data = {
        "units": {
            "angle": "arcseconds",
            "phi_G": "radians (from x-axis, anticlockwise)"
        },
        "cosmology": {
            "model": "FlatLambdaCDM",
            "Om": extract_value(r"Om\s*=\s*([0-9.]+)", content),
            "H0": extract_value(r"H0:\s*([0-9.]+)", content)
        },
        "pixel_size": {
            "before_drizzle": extract_value(r"Pixel size is ([0-9.]+)''", content),
            "after_drizzle": extract_value(r"and ([0-9.]+)'' after drizzle", content)
        },
        "time_delay_distance": extract_value(r"Time delay distance:.*?([0-9.]+)Mpc", content),
        "time_delays_BCD_minus_A": ast.literal_eval(extract_value(r"Time delay of BCD - A\s*:\s*array\((.*?)\)", content, str, "[]")),
        "zeropoint_AB": extract_value(r"Zeropoint of filter.*?:\s*([0-9.]+)", content),
        "redshifts": {"lens": None, "source": None},
        "lens_mass_model": {"SPEMD": None, "shear": None},
        "lens_light": None,
        "source_light": {
            "host_name": None,
            "center_pos": None,
            "mag": None,
            "R_eff": None
        },
        "AGN_light": {"source_plane": {}, "image_plane": {}},
        "magnitudes": {},
        "velocity_dispersion": extract_value(r"Measured velocity dispersion.*?:\s*([0-9.]+)", content),
        "kappa_ext": extract_value(r"kappa_ext.*?:\s*([0-9.]+)", content),
        "time_delays_with_kappa_ext": ast.literal_eval(extract_value(r"Time delay with external kappa.*?:\s*array\((.*?)\)", content, str, "[]"))
    }

    redshift_match = re.search(r"Lens/Source redshift:\s*\[([0-9.,\s]+)\]", content)
    if redshift_match:
        z_l, z_s = map(float, redshift_match.group(1).split(","))
        lens_data["redshifts"]["lens"] = z_l
        lens_data["redshifts"]["source"] = z_s

    match = re.search(r"SPEMD:(\{.*?\})", content, re.DOTALL)
    if match:
        lens_data["lens_mass_model"]["SPEMD"] = ast.literal_eval(match.group(1))

    match = re.search(r"Shear:\s*\((\{.*?\}),\s*(\{.*?\})\)", content)
    if match:
        shear = ast.literal_eval(match.group(1))
        lens_data["lens_mass_model"]["shear"] = shear

    match = re.search(r"Lens light:\s*\n\s*(\{.*?\})", content, re.DOTALL)
    if match:
        lens_data["lens_light"] = ast.literal_eval(match.group(1))

    match = re.search(r"Host galaxy name:\s*(\S+)\s*CenterPos:\s*array\((.*?)\)", content)
    if match:
        lens_data["source_light"]["host_name"] = match.group(1)
        lens_data["source_light"]["center_pos"] = ast.literal_eval(match.group(2))

    lens_data["source_light"]["mag"] = extract_value(r"Host mag:\s*([0-9.]+)", content)
    lens_data["source_light"]["R_eff"] = extract_value(r"Host R_eff:\s*([0-9.]+)", content)

    lens_data["AGN_light"]["source_plane"]["position"] = list(map(float, re.findall(r"AGN position in source plane:\s*([0-9.\-]+),\s*([0-9.\-]+)", content)[0]))
    lens_data["AGN_light"]["source_plane"]["amplitude"] = extract_value(r"AGN amplitude in source plane:\s*([0-9.]+)", content)

    match = re.search(r"AGN position in image plane:\s*x:\s*array\((.*?)\)\s*y:\s*array\((.*?)\)", content, re.DOTALL)
    if match:
        lens_data["AGN_light"]["image_plane"]["x"] = ast.literal_eval(match.group(1))
        lens_data["AGN_light"]["image_plane"]["y"] = ast.literal_eval(match.group(2))

    match = re.search(r"AGN amplitude in image plane:\s*array\((.*?)\)", content)
    if match:
        lens_data["AGN_light"]["image_plane"]["amplitude"] = ast.literal_eval(match.group(1))

    lens_data["magnitudes"]["host_galaxy_image_plane"] = extract_value(r"Host galaxy mag in the image plane:\s*([0-9.]+)", content)
    lens_data["magnitudes"]["AGN_total_image_plane"] = extract_value(r"AGN total mag in the image plane:\s*([0-9.]+)", content)

    return cast_ints_to_floats_in_dict(lens_data)


def parse_good_team_lens_info_file(filepath):
    """Parse a TDC 'good team' lens info text file."""
    with open(filepath) as f:
        content = f.read()

    def extract_value(pattern, eval_type=float, default=None):
        match = re.search(pattern, content)
        if match:
            try:
                return eval_type(match.group(1))
            except Exception:
                return default
        return default

    lens_data = {
        "pixel_size": {
            "before_drizzle": extract_value(r"Pixel size is ([0-9.]+)''"),
            "after_drizzle": extract_value(r"and ([0-9.]+)'' after drizzle")
        },
        "zeropoint_AB": extract_value(r"Zeropoint of filter \(AB system\):\s*([0-9.]+)"),
        "redshifts": {"lens": None, "source": None},
        "external_convergence": {"kappa_ext": None, "kappa_ext_error": None},
        "velocity_dispersion": {"value": None, "error": None},
        "time_delays_BCD_minus_A": {"values": None, "errors": None}
    }

    match = re.search(r"Lens/Source redshift:\s*\[([0-9.,\s]+)\]", content)
    if match:
        z_lens, z_source = map(float, match.group(1).split(","))
        lens_data["redshifts"]["lens"] = z_lens
        lens_data["redshifts"]["source"] = z_source

    match = re.search(r"External Convergence: Kext=\s*([-\d.]+)\s*\+/-\s*([0-9.]+)", content)
    if match:
        lens_data["external_convergence"]["kappa_ext"] = float(match.group(1))
        lens_data["external_convergence"]["kappa_ext_error"] = float(match.group(2))

    match = re.search(r"Measured velocity dispersion:\s*([0-9.]+)km/s, error level:\s*([0-9.]+)km/s", content)
    if match:
        lens_data["velocity_dispersion"]["value"] = float(match.group(1))
        lens_data["velocity_dispersion"]["error"] = float(match.group(2))

    match = re.search(r"Time delay of BCD - A\s*:\s*array\((.*?)\)days,\s*error level:\s*array\((.*?)\)days", content)
    if match:
        delays = ast.literal_eval(f"[{match.group(1)}]")
        delay_errors = ast.literal_eval(f"[{match.group(2)}]")
        lens_data["time_delays_BCD_minus_A"]["values"] = delays[0]
        lens_data["time_delays_BCD_minus_A"]["errors"] = delay_errors[0]

    return lens_data


# ---------------------------------------------------------------------------
# Time delay data loading (from alpaca/pipeline/setup.py)
# ---------------------------------------------------------------------------

def load_time_delay_data(base_dir, rung, code_id, seed, x0s, y0s, verbose=False,
                          fallback_to_truth=False):
    """Load measured time delays from TDC files and align to detected image ordering.

    Returns
    -------
    measured_delays : np.ndarray
    delay_errors : np.ndarray
    det_labels : list
    used_fallback : bool
    truth_positions : tuple or None
    """
    from alpaca.pipeline.setup import _match_point_sources

    good_team_path = os.path.join(
        base_dir,
        f"TDC/rung{rung}/code{code_id}/f160w-seed{seed}/lens_info_for_Good_team.txt",
    )
    open_box_path = os.path.join(
        base_dir,
        f"TDC/rung{rung}_open_box/code{code_id}/f160w-seed{seed}/lens_all_info.txt",
    )

    if not os.path.exists(good_team_path):
        raise FileNotFoundError(f"Missing measured delays file: {good_team_path}")
    if not os.path.exists(open_box_path):
        raise FileNotFoundError(
            f"Missing open-box file for image matching: {open_box_path}"
        )

    good_info = parse_good_team_lens_info_file(good_team_path)
    delays = good_info["time_delays_BCD_minus_A"]["values"]
    errors = good_info["time_delays_BCD_minus_A"]["errors"]
    if delays is None or errors is None:
        raise ValueError(f"Time delays not found in {good_team_path}")
    delays = np.asarray(delays, float)
    errors = np.asarray(errors, float)
    if delays.size != errors.size:
        raise ValueError("Time-delay values and errors must have same length.")

    truth_info = parse_lens_info_file(open_box_path)
    x_truth = np.asarray(truth_info["AGN_light"]["image_plane"]["x"], float)
    y_truth = np.asarray(truth_info["AGN_light"]["image_plane"]["y"], float)

    nps_detected = len(np.atleast_1d(x0s))
    nps_truth = x_truth.size
    used_fallback = False
    truth_positions = None

    if nps_detected != nps_truth:
        if fallback_to_truth:
            if verbose:
                print(f"WARNING: Detected {nps_detected} images but truth has {nps_truth}.")
                print("         Using truth positions as fallback.")
            x0s = x_truth
            y0s = y_truth
            used_fallback = True
            truth_positions = (x_truth, y_truth)
        else:
            raise ValueError(
                f"Detected {nps_detected} images but truth has {nps_truth} positions. "
                f"Set fallback_to_truth=True to use truth positions."
            )

    nps = len(np.atleast_1d(x0s))
    if nps > 4:
        raise ValueError("Time-delay mapping supports up to 4 images (A-D).")
    if nps < 2:
        raise ValueError("Need at least two point-source images for time delays.")

    labels = ["A", "B", "C", "D"][:nps]
    perm = _match_point_sources(x0s, y0s, x_truth, y_truth)
    det_labels = [labels[j] for j in perm]

    delay_map = {"A": 0.0}
    err_map = {"A": 0.0}
    for idx, label in enumerate(labels[1:]):
        if idx >= delays.size:
            break
        delay_map[label] = float(delays[idx])
        err_map[label] = float(errors[idx])

    t0 = delay_map.get(det_labels[0])
    s0 = err_map.get(det_labels[0])
    if t0 is None or s0 is None:
        raise ValueError("Could not map reference image to delay data.")

    measured = []
    measured_err = []
    for i in range(1, nps):
        label = det_labels[i]
        if label not in delay_map:
            raise ValueError(f"No time-delay data for image {label}.")
        ti = delay_map[label]
        si = err_map[label]
        measured.append(ti - t0)
        measured_err.append(np.sqrt(si * si + s0 * s0))

    if verbose:
        mapping = ", ".join(f"img{i}={lab}" for i, lab in enumerate(det_labels))
        print(f"Time-delay mapping: {mapping}")

    return (np.asarray(measured, float), np.asarray(measured_err, float),
            det_labels, used_fallback, truth_positions)


# ---------------------------------------------------------------------------
# Truth value extraction for corner plots
# ---------------------------------------------------------------------------

def load_tdlmc_truth_values(base_dir, rung, code_id, seed):
    """Load truth parameter values from TDC open-box file for corner plot markers.

    Returns dict mapping parameter names to truth values, suitable for passing
    as truth_values to corner plot functions.
    """
    code = f"code{code_id}"
    truth_file = os.path.join(
        base_dir, f"TDC/rung{rung}_open_box/{code}/f160w-seed{seed}/lens_all_info.txt"
    )
    lens_info = parse_lens_info_file(truth_file)

    def _phi_to_rad(phi):
        return np.deg2rad(phi) if np.abs(phi) > 2 * np.pi else float(phi)

    def _e1e2_from_q_phi(q, phi):
        e = (1 - q) / (1 + q)
        return e * np.cos(2 * phi), e * np.sin(2 * phi)

    thetaE_true = lens_info["lens_mass_model"]["SPEMD"]["theta_E"]
    gamma_true = lens_info["lens_mass_model"]["SPEMD"]["gamma"]
    q_mass = lens_info["lens_mass_model"]["SPEMD"]["q"]
    phi_mass = _phi_to_rad(lens_info["lens_mass_model"]["SPEMD"].get("phi_G", 0.0))
    e1_mass_true, e2_mass_true = _e1e2_from_q_phi(q_mass, phi_mass)

    q_light = lens_info["lens_light"]["q"]
    phi_light = _phi_to_rad(lens_info["lens_light"]["phi_G"])
    e1_L_true, e2_L_true = _e1e2_from_q_phi(q_light, phi_light)
    R_true = lens_info["lens_light"]["R_sersic"]
    n_true = lens_info["lens_light"]["n_sersic"]

    return {
        "lens_theta_E": thetaE_true,
        "lens_gamma": gamma_true,
        "lens_e1": e1_mass_true,
        "lens_e2": e2_mass_true,
        "light_Re_L": R_true,
        "light_n_L": n_true,
        "light_e1_L": e1_L_true,
        "light_e2_L": e2_L_true,
    }
