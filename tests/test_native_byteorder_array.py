"""Preserve floating storage precision independently from array byte order."""
import numpy as np
import pytest

pytestmark = pytest.mark.rust


@pytest.mark.parametrize("dtype", ["<f2", ">f2", "<f4", ">f4", "<f8", ">f8"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("operation", ["direction", "flame_length"])
def test_floating_byte_order_preserves_normalized_reference(
        dtype,
        reverse,
        masked,
        operation
):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native

    x = np.array([-1., -127., 1.] if operation == "direction" else [1., 2., 126.], dtype=dtype)
    y = np.array([126., 126., -0.], dtype=dtype)
    if masked:
        x = np.ma.array(x, mask=[False, True, False], fill_value=-999.)
        y = np.ma.array(y, mask=[False, True, False], fill_value=-999.)
    if reverse:
        x, y = x[::-1], y[::-1]
    # The policy oracle widens half precision independently from the bridge.
    expected_x = x.astype(np.float32) if x.dtype.itemsize == 2 else x
    expected_y = y.astype(np.float32) if y.dtype.itemsize == 2 else y
    if operation == "direction":
        expected = reference.calculate_direction_of_max_spread(
            expected_x, expected_y, np.ones(3), "RelativeToNorth", 95.)
        actual = native.calculate_direction_of_max_spread(x, y, np.ones(3), "RelativeToNorth", 95.)
    else:
        expected = reference.calculate_flame_length(expected_x)
        actual = native.calculate_flame_length(x)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape == (3,)
    assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    if np.ma.isMaskedArray(expected):
        assert actual.fill_value == expected.fill_value
    actual_data, expected_data = np.ma.getdata(actual), np.ma.getdata(expected)
    np.testing.assert_array_equal(np.signbit(actual_data), np.signbit(expected_data))
    np.testing.assert_allclose(actual_data, expected_data,
                               rtol=1e-6 if expected.dtype == np.float32 else 1e-9,
                               atol=1e-7 if expected.dtype == np.float32 else 1e-12)
