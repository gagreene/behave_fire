"""Supported numeric-object payloads retain boxed types and exact integer values."""

from decimal import Decimal

import numpy as np
import pytest

pytestmark=pytest.mark.rust


class CustomFloat(float):
    def __add__(self, other):
        return 777.


class CustomInt(int):
    def __add__(self, other):
        return 999


class CustomNumpyFloat(np.float64):
    def __add__(self, other):
        return 777.


class CustomNumpyInt(np.int64):
    def __add__(self, other):
        return 999


def test_boxed_integer_addition_is_exact_without_fixed_width_wrapping():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    forward=np.array([2**53+1,2**63-1],dtype=object)
    backing=np.array([2,1],dtype=object)
    expected=reference.calculate_fire_length(forward,backing,1)
    actual=surface.calculate_fire_length(forward,backing,1)
    assert actual.dtype==expected.dtype==object
    assert all(type(item) is int for item in actual)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('scalar_class', [np.longlong, np.ulonglong])
def test_boxed_longlong_arithmetic_preserves_payloads(scalar_class):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    forward = np.array([scalar_class(2**53 + 1)], dtype=object)
    backing = np.array([scalar_class(2)], dtype=object)
    expected = reference.calculate_fire_length(forward, backing, 1)
    actual = surface.calculate_fire_length(forward, backing, 1)
    assert actual.dtype == expected.dtype == object
    assert type(actual[0]) is type(expected[0])
    assert actual[0].dtype == expected[0].dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('scalar_class', [np.longlong, np.ulonglong])
def test_boxed_longlong_subclasses_remain_unsupported(scalar_class):
    from behave_fire.rust import surface

    class CustomLongLong(scalar_class):
        def __add__(self, other):
            return 999

    forward = np.array([CustomLongLong(2)], dtype=object)
    with pytest.raises(TypeError, match='opaque object arithmetic'):
        surface.calculate_fire_length(forward, np.array([3], dtype=object), 1)


def test_mixed_boxed_numeric_payload_types_are_preserved():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    forward=np.array([2**53+1,np.float32(1.25),2.5],dtype=object)
    backing=np.array([2,np.float32(.5),.2],dtype=object)
    expected=reference.calculate_fire_length(forward,backing,1)
    actual=surface.calculate_fire_length(forward,backing,1)
    assert actual.dtype==expected.dtype==object
    assert [type(value) for value in actual]==[type(value) for value in expected]
    np.testing.assert_array_equal(actual,expected)


def test_normalizing_entry_point_accepts_numeric_decimal_coercion():
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units
    values=np.array([Decimal('1.23456789')],dtype=object)
    expected=reference.speed_to_base(values,5)
    actual=behave_units.speed_to_base(values,5)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('scalar_class', [CustomInt, CustomFloat, CustomNumpyInt, CustomNumpyFloat])
def test_normalizing_entry_point_preserves_numeric_subclass_coercion(scalar_class):
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units

    values = np.array([scalar_class(2)], dtype=object)
    expected = reference.speed_to_base(values, 5)
    actual = behave_units.speed_to_base(values, 5)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('scalar_class,reference_result', [(CustomInt, 999), (CustomFloat, 777.), (CustomNumpyInt, 999), (CustomNumpyFloat, 777.)])
def test_raw_object_arithmetic_rejects_numeric_subclasses(scalar_class, reference_result):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    forward = np.array([scalar_class(2)], dtype=object)
    backing = np.array([3], dtype=object)
    np.testing.assert_array_equal(reference.calculate_fire_length(forward, backing, 1), np.array([reference_result], dtype=object))
    with pytest.raises(TypeError, match='opaque object arithmetic'):
        surface.calculate_fire_length(forward, backing, 1)


def test_unrepresentable_boxed_integer_output_requires_python_backend():
    from behave_fire.rust import surface
    with pytest.raises(TypeError):
        surface.calculate_fire_length(np.array([2**64-1],dtype=object),np.array([1],dtype=object),1)
