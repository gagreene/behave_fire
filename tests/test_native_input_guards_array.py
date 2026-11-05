"""Reject opaque scalar arithmetic before entering the native boundary."""

from enum import IntEnum

import numpy as np
import pytest


pytestmark = pytest.mark.rust


class _CustomFloat(float):
    def __add__(self, other):
        return 777.0


class _CustomInt(int):
    def __add__(self, other):
        return 999


class _CustomNumpyFloat(np.float64):
    def __add__(self, other):
        return 777.0


class _CustomNumpyInt(np.int64):
    def __add__(self, other):
        return 999


class _Mode(int):
    def __new__(cls, label):
        value = super().__new__(cls, 1)
        value.label = label
        return value

    def __str__(self):
        return self.label


def _forbid_native(*args, **kwargs):
    raise AssertionError("unsupported arithmetic reached native execution")


@pytest.mark.parametrize("scalar_type", [_CustomFloat, _CustomInt, _CustomNumpyFloat, _CustomNumpyInt])
@pytest.mark.parametrize("container", [lambda value: value, lambda value: [value], lambda value: (value,)], ids=["scalar", "list", "tuple"])
def test_crown_input_coercion_preserves_custom_numeric_conversion(scalar_type, container):
    from behave_fire.components import crown as reference
    from behave_fire.rust import crown

    arguments = dict(fuel_model_grid=10, wind_speed=container(scalar_type(2)), wind_direction=0.0, slope=0.0)
    expected = reference.coerce_crown_fire_inputs(**arguments)
    actual = crown.coerce_crown_fire_inputs(**arguments)
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "S":
            assert actual[key] == expected[key]
        else:
            assert actual[key].dtype == expected[key].dtype
            np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("scalar_type", [_CustomFloat, _CustomInt, _CustomNumpyFloat, _CustomNumpyInt])
@pytest.mark.parametrize("container", [lambda value: value, lambda value: [value], lambda value: (value,)], ids=["scalar", "list", "tuple"])
def test_custom_raw_scalars_are_rejected_before_native_execution(
        monkeypatch,
        scalar_type,
        container
):
    from behave_fire.rust import surface
    from behave_fire.rust._bridge import _native

    monkeypatch.setattr(_native, "batch", _forbid_native)
    with pytest.raises(TypeError, match="opaque.*arithmetic"):
        surface.calculate_flanking_spread_rate(container(scalar_type(2)), 3, 1.0)


@pytest.mark.parametrize("scalar_type", [_CustomFloat, _CustomInt, _CustomNumpyFloat, _CustomNumpyInt])
def test_custom_scalar_in_later_partition_is_rejected_before_first_native_call(
        monkeypatch,
        scalar_type
):
    from behave_fire.rust import crown
    from behave_fire.rust._bridge import _native

    monkeypatch.setattr(_native, "batch", _forbid_native)
    with pytest.raises(TypeError, match="opaque.*arithmetic"):
        crown.calculate_crown_fire_ratios(2.0, 4.0, scalar_type(4), 8.0)


@pytest.mark.parametrize("scalar_type", [_CustomFloat, _CustomInt, _CustomNumpyFloat, _CustomNumpyInt])
def test_explicit_real_coercion_still_accepts_custom_numeric_values(scalar_type):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    value = scalar_type(2)
    expected = reference.calculate_fireline_intensity(value, 3.0)
    actual = surface.calculate_fireline_intensity(value, 3.0)
    np.testing.assert_array_equal(actual, expected)


def test_python_backend_custom_arithmetic_remains_unchanged():
    from behave_fire.components import surface

    np.testing.assert_array_equal(surface.calculate_flanking_spread_rate(_CustomFloat(2), 3, 1.0), 388.5)


@pytest.mark.parametrize("scalar_type", [_CustomFloat, _CustomInt, _CustomNumpyFloat, _CustomNumpyInt])
def test_stateful_custom_scalars_are_rejected_before_native_execution(monkeypatch, scalar_type):
    from behave_fire.rust import FireSize
    from behave_fire.rust._bridge import _native

    model = FireSize()
    monkeypatch.setattr(_native, "state_call", _forbid_native)
    with pytest.raises(TypeError, match="opaque.*arithmetic"):
        model.calculate_fire_basic_dimensions(False, scalar_type(2), 5, 10.0, 0)


@pytest.mark.parametrize("scalar_type", [_CustomFloat, _CustomInt, _CustomNumpyFloat, _CustomNumpyInt])
def test_stateful_explicit_unit_coercions_remain_supported(scalar_type):
    from behave_fire.components.fire_size import FireSize as PythonFireSize
    from behave_fire.rust import FireSize

    reference = PythonFireSize()
    actual = FireSize()
    arguments = (False, scalar_type(2), 0, scalar_type(10), 0)
    reference.calculate_fire_basic_dimensions(*arguments)
    actual.calculate_fire_basic_dimensions(*arguments)
    np.testing.assert_array_equal(actual.get_fire_length_to_width_ratio(), reference.get_fire_length_to_width_ratio())


@pytest.mark.parametrize("name,arguments", [
    ("calculate_direction_of_max_spread", (1.0, 1.0, 1.0, _Mode("RelativeToNorth"), 30.0)),
    ("calculate_twenty_foot_wind_speed", (2.0, 0, _Mode("TenMeter"))),
    ("calculate_surface_wind_adjustment_factor", (0.0, 0.0, 0.0, 1.0, _Mode("UserInput"), 0.3)),
])
def test_string_normalized_mode_controls_remain_supported(name, arguments):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    expected = getattr(reference, name)(*arguments)
    actual = getattr(surface, name)(*arguments)
    np.testing.assert_array_equal(actual, expected)


def test_unit_enum_subclasses_remain_valid_controls():
    from behave_fire.rust import surface

    class CrownFlag(IntEnum):
        YES = 1

    expected = surface.calculate_fire_area(2.0, 1.0, 2.0, 1.0, True)
    actual = surface.calculate_fire_area(2.0, 1.0, 2.0, 1.0, CrownFlag.YES)
    np.testing.assert_array_equal(actual, expected)
