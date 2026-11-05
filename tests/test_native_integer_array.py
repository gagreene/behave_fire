"""Fixed-width arithmetic must happen before floating promotion at native boundaries."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def test_arbitrary_precision_raw_integer_requires_python_backend():
    from behave_fire.rust import surface
    with pytest.raises(TypeError,match='Python|python'):
        surface.calculate_fire_length(2**64,0,1)


@pytest.mark.parametrize("dtype", [np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64])
def test_integer_addition_wraps_before_elapsed_time_promotion(dtype):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    forward=np.array([np.iinfo(dtype).max],dtype=dtype)
    backing=np.array([1],dtype=dtype)
    with np.errstate(all='ignore'):
        expected=reference.calculate_fire_length(forward,backing,1.0)
        actual=surface.calculate_fire_length(forward,backing,1.0)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize("dtype", [np.int64,np.uint64])
def test_integer_output_retains_bits_above_float64_precision(dtype):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    forward=np.array([2**53+1],dtype=dtype)
    backing=np.array([2],dtype=dtype)
    elapsed=np.array([3],dtype=dtype)
    expected=reference.calculate_fire_length(forward,backing,elapsed)
    actual=surface.calculate_fire_length(forward,backing,elapsed)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize("dtype", [np.int32,np.int64,np.uint32,np.uint64])
def test_integer_square_wraps_before_eccentricity_promotion(dtype):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    values=np.array([np.iinfo(dtype).max],dtype=dtype)
    with np.errstate(all='ignore'):
        expected=reference.calculate_eccentricity(values)
        actual=surface.calculate_eccentricity(values)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)


def test_large_integer_remains_accepted_by_float64_normalizing_entry_point():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    value=2**64+1
    expected=reference.calculate_fireline_intensity(value,1.)
    actual=surface.calculate_fireline_intensity(value,1.)
    assert actual.dtype==expected.dtype==np.float64
    np.testing.assert_array_equal(actual,expected)


def test_numeric_object_arithmetic_preserves_reference_boxing():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    arguments=(np.array([1.],dtype=object),np.array([.5]))
    expected=reference.calculate_backing_spread_rate(*arguments)
    actual=surface.calculate_backing_spread_rate(*arguments)
    assert actual.dtype==expected.dtype==object
    np.testing.assert_array_equal(actual,expected)


def test_numeric_object_coercion_matches_normalizing_reference():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface
    values=np.array([2**64+1,1.5],dtype=object)
    expected=reference.calculate_fireline_intensity(values,1.)
    actual=surface.calculate_fireline_intensity(values,1.)
    assert actual.dtype==expected.dtype==np.float64
    np.testing.assert_array_equal(actual,expected)


def test_partitioned_raw_object_is_rejected_before_native_execution(monkeypatch):
    from behave_fire.rust import crown
    from behave_fire.rust._bridge import _native

    def forbidden(*args,**kwargs):
        raise AssertionError('native execution began before raw-object validation')

    monkeypatch.setattr(_native,'batch',forbidden)
    values=dict(is_surface_fire=True,is_passive_crown=False,surface_ros=1.,surface_hpua=2.,surface_fli=3.,passive_ros=1.,passive_hpua=2.,passive_fli=3.,passive_fl=4.,crown_ros=1.,crown_hpua=2.,crown_fli=3.,crown_fl=np.array([object()],dtype=object))
    with pytest.raises(TypeError,match='Python'):
        crown.assign_final_crown_fire_behavior(**values)


def test_python_bool_and_numpy_bool_addition_are_distinct():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    for first,second in ((True,True),(np.array([True]),np.array([True]))):
        expected=reference.calculate_flanking_spread_rate(first,second,1.)
        actual=surface.calculate_flanking_spread_rate(first,second,1.)
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize("left,right", [(np.int64,np.uint64),(np.int32,np.uint32),(np.int16,np.uint8)])
def test_signed_unsigned_promotion_matches_numpy(left,right):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    forward=np.array([-1,3],dtype=left)
    backing=np.array([np.iinfo(right).max,2],dtype=right)
    with np.errstate(all='ignore'):
        expected=reference.calculate_fire_length(forward,backing,1)
        actual=surface.calculate_fire_length(forward,backing,1)
    assert actual.dtype==expected.dtype
    np.testing.assert_array_equal(actual,expected)
