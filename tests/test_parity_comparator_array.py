"""Negative controls for the exact parts of the native acceptance gate."""

import numpy as np
import pytest

from parity_support import assert_result, encode_value


@pytest.mark.parametrize("dtype", [np.float32, np.float64, object])
def test_allowed_floating_roundoff_passes(dtype):
    actual = np.array([1.0 + 1e-10], dtype=dtype)
    expected = np.array([1.0], dtype=dtype)
    if dtype is np.float32:
        actual = np.nextafter(expected, np.array([2.], dtype=dtype))
    assert np.all(actual != expected), "The perturbation must survive the input dtype"
    assert_result(actual, encode_value(expected))


def test_batches_numeric_leaves_with_their_original_dtype_and_tolerance(monkeypatch):
    import parity_support

    original = parity_support._assert_floating
    calls = []

    def observe(actual, expected, single=False):
        value = np.asarray(actual)
        calls.append((value.dtype.name, single, value.shape))
        return original(actual, expected, single)

    monkeypatch.setattr(parity_support, '_assert_floating', observe)
    actual = dict(first=1., second=2., third=3., narrow=np.array([4., 5.], np.float32))
    expected = dict(first=1., second=2., third=3.,
                    narrow=dict(dtype='float32', shape=[2], values=[4., 5.]))
    assert_result(actual, expected)
    assert sorted(calls) == [('float32', True, (2,)), ('float64', False, (3,))]


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


@pytest.mark.parametrize('complex_first', [False, True])
@pytest.mark.parametrize('scalar', [False, True])
def test_complex_mismatch_is_rejected_beside_real_leaves(complex_first, scalar):
    expected = 1.+0j if scalar else dict(dtype='complex128', shape=[1], values=[1.+0j])
    actual = dict(real=1., complex=1.+100j if scalar else np.array([1.+100j]))
    reference = dict(real=1., complex=expected)
    if complex_first:
        actual = dict(reversed(list(actual.items())))
        reference = dict(reversed(list(reference.items())))
    with pytest.raises(AssertionError):
        assert_result(actual, reference)


@pytest.mark.parametrize('actual,expected', [
    (np.array([np.nan, 1.]), np.array([1., np.nan])),
    (np.array([np.inf]), np.array([-np.inf])),
    (np.array([1.]), np.array([np.inf])),
    (np.array([1.], np.float32), np.array([1.], np.float64)),
    (np.ones((1, 1)), np.ones(1)),
    (np.ma.array([1., 2.], mask=[False, True]), np.ma.array([1., 2.], mask=[True, False])),
    (np.array([1.01], np.float32), np.array([1.], np.float32)),
    (np.array([1.+1e-5]), np.array([1.])),
])
def test_numeric_batches_preserve_exact_boundaries(actual, expected):
    with pytest.raises(AssertionError):
        assert_result(dict(first=1., value=actual), dict(first=1., value=encode_value(expected)))


@pytest.mark.parametrize('actual,expected', [
    (2., 1.),
    (np.array([2.]), dict(dtype='float64', shape=[1], values=[1.])),
])
def test_numeric_mismatch_reports_the_nested_field_path(actual, expected):
    actual = dict(fuel=dict(depth=actual))
    expected = dict(fuel=dict(depth=expected))
    with pytest.raises(AssertionError, match=r"\$\['fuel'\]\['depth'\]"):
        assert_result(actual, expected)


def test_numeric_profiles_do_not_share_the_looser_float32_tolerance():
    actual = dict(narrow=np.array([1.], np.float32), strict=1.+1e-7)
    expected = dict(narrow=dict(dtype='float32', shape=[1], values=[1.]), strict=1.)
    with pytest.raises(AssertionError):
        assert_result(actual, expected)


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
