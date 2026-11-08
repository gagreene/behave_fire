"""Prepared boundary paths retain unit types, masks, layouts, and error behavior."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust

CLASSES = ("Area", "BasalArea", "Density", "FirelineIntensity", "Fraction", "HeatOfCombustion",
           "HeatPerUnitArea", "HeatSink", "HeatSourceAndReactionIntensity", "Length", "Loading",
           "Pressure", "Slope", "Speed", "SurfaceAreaToVolume", "Temperature", "Time")


def _values(layout):
    values = np.array([-0.0, 0.0, 0.125, 1.25, np.nan, np.inf, -np.inf])
    if layout == "single":
        return values.astype(np.float32)
    if layout == "scalar":
        return np.array(-0.0)
    if layout == "empty":
        return np.empty((2, 0))
    if layout == "strided":
        return values[::-1]
    if layout == "endian":
        return values.astype(">f8")
    if layout == "readonly":
        values.flags.writeable = False
    if layout == "masked":
        return np.ma.array(values, mask=[False, True, False, False, True, False, False], fill_value=-17.0)
    return values


@pytest.mark.parametrize("class_name", CLASSES)
@pytest.mark.parametrize("direction", ["to_base", "from_base"])
@pytest.mark.parametrize("layout", ["double", "single", "scalar", "empty", "strided", "endian", "readonly", "masked"])
@pytest.mark.parametrize("code", [0, 1])
def test_prepared_conversions_match_reference_for_all_unit_classes(class_name, direction, layout, code):
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units as native

    method = "toBaseUnits" if direction == "to_base" else "fromBaseUnits"
    expected_function = getattr(getattr(reference, class_name + "Units"), method)
    actual_function = getattr(getattr(native, class_name + "Units"), method)
    values = _values(layout)
    with np.errstate(all="ignore"):
        widened = np.asarray(np.ma.asarray(values, dtype=np.float64).filled(np.nan))
        expected = expected_function(widened, code)
        actual = actual_function(values, code)
    expected, actual = np.asanyarray(expected), np.asanyarray(actual)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    if np.ma.isMaskedArray(expected):
        np.testing.assert_equal(actual.fill_value, expected.fill_value)
    left, right = np.ma.getdata(actual), np.ma.getdata(expected)
    np.testing.assert_allclose(left, right, rtol=1e-6 if right.dtype == np.float32 else 1e-9, atol=1e-7 if right.dtype == np.float32 else 1e-12, equal_nan=True)
    zero = (left == 0) & (right == 0)
    np.testing.assert_array_equal(np.signbit(left[zero]), np.signbit(right[zero]))
@pytest.mark.parametrize("class_name", CLASSES)
@pytest.mark.parametrize("selector", [-1, 999, 1.5])
def test_prepared_converters_follow_reference_selector_conventions(class_name, selector):
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units as native

    expected_function = getattr(reference, class_name + "Units").toBaseUnits
    actual_function = getattr(native, class_name + "Units").toBaseUnits
    values = np.array([1.0, 2.0])
    try:
        expected = expected_function(values, selector)
    except (IndexError, TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            actual_function(values, selector)
    else:
        np.testing.assert_allclose(actual_function(values, selector), expected, rtol=1e-9, atol=1e-12)

def test_prepared_native_unit_loop_preserves_varying_selectors_and_threads():
    from behave_fire import _rust
    from behave_fire.components.behave_units import LengthUnits

    values = np.linspace(-3.0, 13.0, 10000)
    selectors = (np.arange(values.size) % 8).astype(np.float64)
    expected = np.array([LengthUnits.toBaseUnits(value, int(code)).item()
                         for value, code in zip(values, selectors)])
    for threads in (1, 4):
        actual = _rust.batch("units.Length.to_base", {"value": values, "units": selectors}, {},
                             values.shape, False, threads=threads)[""]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
