"""Negative controls for the exact parts of the native acceptance gate."""

import numpy as np
import pytest

from parity_support import assert_result, encode_value


@pytest.mark.parametrize("dtype", [np.float32, np.float64, object])
def test_allowed_floating_roundoff_passes(dtype):
    assert_result(np.array([1.0 + 1e-10], dtype=dtype), encode_value(np.array([1.0], dtype=dtype)))


@pytest.mark.parametrize("actual, expected", [
    (np.array([2**53], object), np.array([2**53 + 1], object)),
    (np.array([False], object), np.array([True], object)),
    (np.array([1.0], object), np.array([1], object)),
    (np.array([1], object), np.array([True], object)),
])
def test_boxed_integral_payloads_require_exact_values_and_kinds(actual, expected):
    with pytest.raises(AssertionError):
        assert_result(actual, encode_value(expected))


@pytest.mark.parametrize("actual, expected", [
    (left, right) for left in (True, 1, 1.0, complex(1., 0.))
    for right in (True, 1, 1.0, complex(1., 0.)) if type(left) is not type(right)
])
def test_boxed_numeric_families_match_in_both_directions(actual, expected):
    # The old codec preserves bool/int/float families. Complex expected values
    # use a live record because the historical JSON exporter cannot encode them.
    record = {"dtype": "object", "shape": [1], "values": [expected]}
    with pytest.raises(AssertionError):
        assert_result(np.array([actual], dtype=object), record)


@pytest.mark.parametrize("actual, expected", [(np.array([1.0]), 1.0), (np.array(1.0), 1.0), (1.0, np.array([1.0]))])
def test_scalar_array_structure_changes_fail(actual, expected):
    with pytest.raises(AssertionError):
        assert_result(actual, encode_value(expected))


@pytest.mark.parametrize("actual, expected", [
    (0.0, -0.0),
    (np.array([0.0]), np.array([-0.0])),
    (np.float32(0.0), np.float32(-0.0)),
    (np.array([0.0], object), np.array([-0.0], object)),
    (np.ma.array([0.0], mask=[True]), np.ma.array([-0.0], mask=[True])),
])
def test_zero_sign_changes_fail(actual, expected):
    with pytest.raises(AssertionError):
        assert_result(actual, encode_value(expected))


@pytest.mark.parametrize("actual, expected", [(complex(0., 1.), complex(-0., 1.)), (complex(1., 0.), complex(1., -0.))])
def test_zero_sign_changes_in_complex_components_fail(actual, expected):
    # The historical JSON codec cannot encode complex payloads. Exercise the
    # comparator with independent live values instead of changing old fixtures.
    with pytest.raises(AssertionError):
        assert_result(np.array([actual]), {"dtype": "complex128", "shape": [1], "values": [expected]})
