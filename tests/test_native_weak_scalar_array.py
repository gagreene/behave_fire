"""Known follow-up contracts for eager weak-scalar arithmetic and empty arrays."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


@pytest.mark.parametrize('exponent',[-1,np.array([-1],dtype=np.int64)])
def test_empty_integer_power_does_not_invent_element_errors(exponent):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    arguments=(np.empty(0,dtype=np.int64),np.empty(0,dtype=np.int64),exponent,1.,1.,1.)
    expected=reference.calculate_wind_phi(*arguments)
    actual=surface.calculate_wind_phi(*arguments)
    assert actual.shape==expected.shape==(0,)
    assert actual.dtype==expected.dtype


def test_negative_weak_passive_flame_preserves_generated_complex_dtype():
    from behave_fire.components import crown as reference
    from behave_fire.rust import crown
    arguments=(-1.,1000.,-1.,0.,1.)
    expected=reference.calculate_passive_crown_fire_behavior(*arguments)
    actual=crown.calculate_passive_crown_fire_behavior(*arguments)
    key='passive_crown_fire_flame_length'
    assert actual[key].dtype==expected[key].dtype==np.complex128
    np.testing.assert_array_equal(actual[key],expected[key])


def test_negative_weak_wind_power_preserves_generated_complex_dtype():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    arguments=(1.,-1.,1.5,1.,1.,1.)
    expected=reference.calculate_wind_phi(*arguments)
    actual=surface.calculate_wind_phi(*arguments)
    assert actual.dtype==expected.dtype==np.complex128
    np.testing.assert_array_equal(actual,expected)


def test_weak_backing_scalar_zero_denominator_raises():
    from behave_fire.rust import surface
    with pytest.raises(ZeroDivisionError):
        surface.calculate_backing_spread_rate(1.,-1.)


def test_weak_crown_fraction_scalar_zero_denominator_raises():
    from behave_fire.rust import crown
    with pytest.raises(ZeroDivisionError):
        crown.calculate_crown_fraction_burned(1.,1.,1.)


def test_weak_eccentricity_scalar_power_overflow_raises():
    from behave_fire.rust import surface
    with pytest.raises(OverflowError):
        surface.calculate_eccentricity(1e200)
