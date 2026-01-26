"""
Test ALPACA benchmarking utilities.
"""

import pytest
import numpy as np
import json
import tempfile
import os


class TestMultistartBenchmarking:
    """Test multi-start gradient descent benchmarking."""

    @pytest.fixture
    def mock_multistart_summary(self):
        """Create a mock multi-start summary."""
        return {
            "n_starts": 5,
            "n_param": 20,
            "best_run": 2,
            "best_loss": 100.5,
            "final_losses": [150.0, 120.0, 100.5, 130.0, 110.0],
            "chi2_reds": [1.5, 1.2, 1.0, 1.3, 1.1],
            "best_trace": [150.0, 120.0, 100.5, 100.5, 100.5],
            "results": [
                {"run": 0, "final_loss": 150.0, "chi2_red": 1.5},
                {"run": 1, "final_loss": 120.0, "chi2_red": 1.2},
                {"run": 2, "final_loss": 100.5, "chi2_red": 1.0},
                {"run": 3, "final_loss": 130.0, "chi2_red": 1.3},
                {"run": 4, "final_loss": 110.0, "chi2_red": 1.1},
            ],
            "timing": {"total": 120.5},
        }

    def test_summarise_multistart(self, mock_multistart_summary):
        """Test multi-start summary statistics."""
        from alpaca.benchmarking import summarise_multistart

        stats = summarise_multistart(mock_multistart_summary)

        assert stats["n_starts"] == 5
        assert stats["n_param"] == 20
        assert stats["best_loss"] == 100.5
        assert np.isclose(stats["median_final_loss"], 120.0)
        assert stats["total_runtime"] == 120.5

    def test_print_multistart_summary(self, mock_multistart_summary, capsys):
        """Test multi-start summary printing."""
        from alpaca.benchmarking import print_multistart_summary

        print_multistart_summary(mock_multistart_summary)
        captured = capsys.readouterr()

        assert "Multi-start" in captured.out
        assert "5" in captured.out  # n_starts

    def test_load_multistart_summary(self, mock_multistart_summary):
        """Test loading multi-start summary from file."""
        from alpaca.benchmarking import load_multistart_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save summary
            summary_path = os.path.join(tmpdir, "multi_start_summary.json")
            with open(summary_path, "w") as f:
                json.dump(mock_multistart_summary, f)

            # Load and verify
            loaded = load_multistart_summary(tmpdir, verbose=False)
            assert loaded["n_starts"] == 5
            assert loaded["best_loss"] == 100.5


class TestNautilusBenchmarking:
    """Test Nautilus benchmarking utilities."""

    @pytest.fixture
    def mock_nautilus_records(self):
        """Create mock Nautilus timing records."""
        return [
            {"total_runtime": 300.0, "sampling_time": 280.0, "n_live": 500, "n_dims": 20},
            {"total_runtime": 350.0, "sampling_time": 320.0, "n_live": 500, "n_dims": 20},
            {"total_runtime": 600.0, "sampling_time": 550.0, "n_live": 1000, "n_dims": 20},
        ]

    def test_summarise_nautilus(self, mock_nautilus_records):
        """Test Nautilus summary statistics."""
        from alpaca.benchmarking import summarise_nautilus

        stats = summarise_nautilus(mock_nautilus_records)

        assert stats["n_runs"] == 3
        assert np.isclose(stats["median_total_runtime"], 350.0)
        assert np.isclose(stats["median_sampling_time"], 320.0)

    def test_print_nautilus_summary(self, mock_nautilus_records, capsys):
        """Test Nautilus summary printing."""
        from alpaca.benchmarking import print_nautilus_summary

        print_nautilus_summary(mock_nautilus_records)
        captured = capsys.readouterr()

        assert "Nautilus" in captured.out
        assert "3" in captured.out  # n_runs


class TestNUTSBenchmarking:
    """Test NUTS benchmarking utilities."""

    @pytest.fixture
    def mock_nuts_results(self):
        """Create mock NUTS results."""
        return {
            "num_samples": 2000,
            "num_warmup": 1000,
            "time_total": 450.0,
            "time_sampling": 300.0,
            "time_warmup": 150.0,
        }

    def test_summarise_nuts(self, mock_nuts_results):
        """Test NUTS summary statistics."""
        from alpaca.benchmarking import summarise_nuts

        stats = summarise_nuts(mock_nuts_results)

        assert stats["num_samples"] == 2000
        assert stats["num_warmup"] == 1000
        assert stats["time_total"] == 450.0

    def test_print_nuts_summary(self, mock_nuts_results, capsys):
        """Test NUTS summary printing."""
        from alpaca.benchmarking import print_nuts_summary

        print_nuts_summary(mock_nuts_results)
        captured = capsys.readouterr()

        assert "NUTS" in captured.out
        assert "2000" in captured.out  # num_samples


class TestBenchmarkingPlots:
    """Test benchmarking plot functions (without displaying)."""

    @pytest.fixture
    def mock_multistart_summary(self):
        """Create a mock multi-start summary for plotting."""
        return {
            "n_starts": 5,
            "best_trace": [150.0, 120.0, 100.5, 100.5, 100.5],
            "results": [
                {"run": 0, "final_loss": 150.0, "chi2_red": 1.5, "t_adam": 10.0, "t_lbfgs": 5.0},
                {"run": 1, "final_loss": 120.0, "chi2_red": 1.2, "t_adam": 12.0, "t_lbfgs": 6.0},
                {"run": 2, "final_loss": 100.5, "chi2_red": 1.0, "t_adam": 11.0, "t_lbfgs": 5.5},
                {"run": 3, "final_loss": 130.0, "chi2_red": 1.3, "t_adam": 10.5, "t_lbfgs": 5.2},
                {"run": 4, "final_loss": 110.0, "chi2_red": 1.1, "t_adam": 11.5, "t_lbfgs": 5.8},
            ],
        }

    def test_plot_multistart_best_trace(self, mock_multistart_summary):
        """Test best trace plot creation."""
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for CI
        from alpaca.benchmarking import plot_multistart_best_trace

        fig, ax = plot_multistart_best_trace(mock_multistart_summary, show=False)
        assert fig is not None
        assert ax is not None

    def test_plot_multistart_losses_and_chi2(self, mock_multistart_summary):
        """Test losses and chi2 plot creation."""
        import matplotlib
        matplotlib.use("Agg")
        from alpaca.benchmarking import plot_multistart_losses_and_chi2

        fig, ax = plot_multistart_losses_and_chi2(mock_multistart_summary, show=False)
        assert fig is not None
        assert ax is not None
