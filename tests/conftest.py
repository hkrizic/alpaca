"""
Shared pytest fixtures for ALPACA tests.
"""

import pytest
import numpy as np


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests on TDC data (slow)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires --run-integration to run)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration"):
        # Run all tests including integration
        return

    skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def rng(random_seed):
    """NumPy random generator with fixed seed."""
    return np.random.default_rng(random_seed)


@pytest.fixture
def small_image(rng):
    """Small test image (32x32)."""
    return rng.normal(0, 1, (32, 32)).astype(np.float64)


@pytest.fixture
def small_noise_map():
    """Small noise map (32x32) with constant noise."""
    return np.ones((32, 32), dtype=np.float64) * 0.1


@pytest.fixture
def small_psf_kernel():
    """Small PSF kernel (11x11) - simple Gaussian."""
    size = 11
    center = size // 2
    y, x = np.ogrid[:size, :size]
    sigma = 1.5
    kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float64)


@pytest.fixture
def mock_point_sources():
    """Mock point source positions and amplitudes."""
    return {
        "x_image": np.array([0.5, -0.5, 0.3, -0.3]),
        "y_image": np.array([0.3, -0.3, -0.5, 0.5]),
        "amp": np.array([100.0, 80.0, 60.0, 40.0]),
    }


@pytest.fixture
def mock_lens_params():
    """Mock lens model parameters."""
    return {
        "theta_E": 1.2,
        "e1": 0.05,
        "e2": -0.03,
        "center_x": 0.0,
        "center_y": 0.0,
        "gamma": 2.0,
        "gamma1": 0.02,
        "gamma2": -0.01,
    }


@pytest.fixture
def mock_source_params():
    """Mock source light parameters."""
    return {
        "amp": 50.0,
        "R_sersic": 0.3,
        "n_sersic": 1.0,
        "e1": 0.1,
        "e2": 0.05,
        "center_x": 0.1,
        "center_y": -0.05,
    }
