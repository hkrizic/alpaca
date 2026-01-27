"""Tests for alpaca.utils module."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

np = pytest.importorskip("numpy")
pytest.importorskip("jax")
pytest.importorskip("scipy")


class TestImports:
    """Verify public API is importable."""

    def test_import_cosmology_functions(self):
        from alpaca.utils import (
            compute_D_dt,
            predict_time_delay,
            Ddt_2_H0,
            Dd_2_H0,
            parse_lens_info_file,
            cast_ints_to_floats_in_dict,
        )
        assert callable(compute_D_dt)
        assert callable(predict_time_delay)
        assert callable(Ddt_2_H0)
        assert callable(Dd_2_H0)
        assert callable(parse_lens_info_file)
        assert callable(cast_ints_to_floats_in_dict)


class TestCastIntsToFloats:
    """Test the recursive int-to-float converter."""

    def test_simple_int(self):
        from alpaca.utils.cosmology import cast_ints_to_floats_in_dict

        assert cast_ints_to_floats_in_dict(5) == 5.0
        assert isinstance(cast_ints_to_floats_in_dict(5), float)

    def test_float_passthrough(self):
        from alpaca.utils.cosmology import cast_ints_to_floats_in_dict

        assert cast_ints_to_floats_in_dict(3.14) == 3.14

    def test_string_passthrough(self):
        from alpaca.utils.cosmology import cast_ints_to_floats_in_dict

        assert cast_ints_to_floats_in_dict("hello") == "hello"

    def test_none_passthrough(self):
        from alpaca.utils.cosmology import cast_ints_to_floats_in_dict

        assert cast_ints_to_floats_in_dict(None) is None

    def test_nested_dict(self):
        from alpaca.utils.cosmology import cast_ints_to_floats_in_dict

        data = {"a": 1, "b": {"c": 2, "d": 3.0}, "e": "text"}
        result = cast_ints_to_floats_in_dict(data)
        assert result["a"] == 1.0
        assert isinstance(result["a"], float)
        assert result["b"]["c"] == 2.0
        assert isinstance(result["b"]["c"], float)
        assert result["b"]["d"] == 3.0
        assert result["e"] == "text"

    def test_list(self):
        from alpaca.utils.cosmology import cast_ints_to_floats_in_dict

        result = cast_ints_to_floats_in_dict([1, 2, 3])
        assert result == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in result)


class TestIsBatchedPytree:
    """Test is_batched_pytree utility."""

    def test_unbatched(self):
        from alpaca.utils.jax_helpers import is_batched_pytree

        tree = {"a": np.array(1.0), "b": np.array(2.0)}
        assert is_batched_pytree(tree) is False

    def test_batched(self):
        from alpaca.utils.jax_helpers import is_batched_pytree

        tree = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}
        assert is_batched_pytree(tree) is True
