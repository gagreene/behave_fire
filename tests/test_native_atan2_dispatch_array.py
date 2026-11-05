"""Exercise atan2 dispatch independently from exp and through unusual operands."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


@pytest.mark.parametrize("input_kind", ["float32", "mixed_integer"])
@pytest.mark.parametrize("orientation", ["RelativeToUpslope", "RelativeToNorth"])
def test_direction_parallel_batches_match_python_and_each_other(
        input_kind,
        orientation
):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native
    from behave_fire.rust._threads import _THREADS

    count = 10_080
    if input_kind == "mixed_integer":
        x = np.resize(np.array([-128, -127, -1, 0, 1, 126, 127], dtype=np.int8), count)
        y = np.resize(np.array([0, 126, 65534, 65535, 1, 17, 127], dtype=np.uint16), count)
    else:
        center = np.float32(np.tan(np.radians(.5)))
        x = np.resize(np.array([-1., -127., 1., 1., -1., -1., 0., -0.,
                                np.inf, -np.inf, np.nan, 1., 1., 1.], dtype=np.float32), count)
        y = np.resize(np.array([65534., 126., 0., -0., 0., -0., 1., -1.,
                                np.inf, np.inf, 1., np.nextafter(center, np.float32(-np.inf)),
                                center, np.nextafter(center, np.float32(np.inf))], dtype=np.float32), count)
    rate = np.ones(count)
    rate[::13] = 0.
    with np.errstate(all="ignore"):
        expected = reference.calculate_direction_of_max_spread(x, y, rate, orientation, 95.)
    previous = None
    for workers in (0, 1, 4):
        token = _THREADS.set(workers)
        try:
            with np.errstate(all="ignore"):
                actual = native.calculate_direction_of_max_spread(x, y, rate, orientation, 95.)
        finally:
            _THREADS.reset(token)
        assert actual.shape == expected.shape == (count,)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        np.testing.assert_array_equal(actual == 0., expected == 0.)
        snapped = 275. if orientation == "RelativeToNorth" else 0.
        np.testing.assert_array_equal(actual == snapped, expected == snapped)
        np.testing.assert_allclose(actual, expected, rtol=1e-6 if expected.dtype == np.float32 else 1e-9,
                                   atol=1e-7 if expected.dtype == np.float32 else 1e-12)
        bits = actual.view(np.uint32 if actual.dtype == np.float32 else np.uint64)
        if previous is not None:
            np.testing.assert_array_equal(bits, previous)
        previous = bits.copy()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("orientation", ["RelativeToUpslope", "RelativeToNorth"])
@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("layout", ["scalar", "strided"])
def test_direction_snap_boundary_matches_python(dtype, orientation, sign, layout):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native

    center = dtype(np.tan(np.radians(sign * .5)))
    boundaries = np.array([np.nextafter(center, dtype(-np.inf)), center,
                           np.nextafter(center, dtype(np.inf))], dtype=dtype)
    storage = np.repeat(boundaries, 2)
    ys = [value for value in boundaries] if layout == "scalar" else [storage[::2]]
    for y in ys:
        x = dtype(1.) if layout == "scalar" else np.ones_like(storage)[::2]
        rate = np.ones(np.shape(y))
        expected = reference.calculate_direction_of_max_spread(x, y, rate, orientation, 95.)
        actual = native.calculate_direction_of_max_spread(x, y, rate, orientation, 95.)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        snapped = 275. if orientation == "RelativeToNorth" else 0.
        np.testing.assert_array_equal(actual == snapped, expected == snapped)
        np.testing.assert_allclose(actual, expected, rtol=1e-6 if expected.dtype == np.float32 else 1e-9,
                                   atol=1e-7 if expected.dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("length", [1, 7, 16, 17, 31])
@pytest.mark.parametrize("layout", ["contiguous", "reverse", "broadcast", "scalar"])
def test_direction_special_values_and_vector_tails_match_python(length, layout):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native

    cases = np.array([0., -0., 1., -1., np.inf, -np.inf, np.nan,
                      np.finfo(np.float32).max, np.finfo(np.float32).tiny,
                      np.nextafter(np.float32(0), np.float32(1)), 65534.], dtype=np.float32)
    x = np.resize(cases, length)
    y = np.resize(cases[::-1], length)
    if layout == "reverse":
        x, y = x[::-1], y[::-1]
    elif layout == "broadcast":
        x, y = np.broadcast_to(x, (2, length)), np.broadcast_to(y, (2, length))
    elif layout == "scalar":
        x, y = x[0], y[0]
    with np.errstate(all="ignore"):
        expected = reference.calculate_direction_of_max_spread(x, y, np.ones(np.shape(x)), "RelativeToNorth", 95.)
        actual = native.calculate_direction_of_max_spread(x, y, np.ones(np.shape(x)), "RelativeToNorth", 95.)
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
