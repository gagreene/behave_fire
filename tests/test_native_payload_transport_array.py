"""Object, generated-complex, and empty provenance cross the owned boundary."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def _assert_payload(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_payload(actual[key], expected[key])
        return
    actual, expected = np.asanyarray(actual), np.asanyarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
    left, right = np.ma.getdata(actual), np.ma.getdata(expected)
    if right.dtype == object:
        for got, wanted in zip(left.flat, right.flat):
            assert type(got) is type(wanted)
            if isinstance(wanted, (int, np.integer, bool, np.bool_)):
                assert got == wanted
            else:
                np.testing.assert_allclose(got, wanted, rtol=1e-6 if isinstance(wanted, np.float32) else 1e-9, atol=1e-12)
    else:
        np.testing.assert_allclose(left, right, rtol=1e-6 if right.dtype in (np.float32, np.complex64) else 1e-9, atol=1e-12)


@pytest.mark.parametrize('real,imaginary', [(np.inf, np.inf), (-np.inf, np.nan), (np.nan, -np.inf), (-0.0, 0.0)])
@pytest.mark.parametrize('single', [False, True])
def test_complex_reconstruction_does_not_perform_complex_arithmetic(real, imaginary, single):
    from behave_fire.rust._bridge import _restore_payload

    metadata = {'': {'single': single, 'complex': True, 'imag': np.array([imaginary])}}
    result = _restore_payload(np.array([real]), metadata, '')
    assert result.dtype == (np.complex64 if single else np.complex128)
    np.testing.assert_array_equal(result.real, np.array([real], dtype=result.real.dtype))
    np.testing.assert_array_equal(result.imag, np.array([imaginary], dtype=result.real.dtype))
    np.testing.assert_array_equal(np.signbit(result.real), np.signbit(np.array([real])))


def test_empty_work_preserves_eager_scalar_bounds_errors():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    arguments = (300, np.empty(0, dtype=np.int8), 1.)
    with pytest.raises(OverflowError):
        reference.calculate_flanking_spread_rate(*arguments)
    with pytest.raises(OverflowError):
        surface.calculate_flanking_spread_rate(*arguments)


@pytest.mark.parametrize('packing', [np.float32(1.), np.float64(1.)])
def test_generated_complex_field_retains_result_precision(packing):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    arguments = (1., -1., 1.5, 1., packing, 1.)
    _assert_payload(surface.calculate_wind_phi(*arguments), reference.calculate_wind_phi(*arguments))


@pytest.mark.parametrize('payload', [1.5, np.float32(1.5), 2, np.int8(2), np.uint64(2**63 + 1), True])
@pytest.mark.parametrize('shape', [(), (1,), (0,)])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('operation', ['calculate_flame_length', 'calculate_residence_time', 'calculate_surface_wind_coefficients'])
def test_object_payload_rank_and_dtype(operation, payload, shape, masked):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    value = np.empty(shape, dtype=object)
    value[...] = payload
    if masked:
        value = np.ma.array(value, mask=False)
    with np.errstate(all='ignore'):
        try:
            expected = getattr(reference, operation)(value)
        except Exception as error:
            with pytest.raises(type(error)):
                getattr(surface, operation)(value)
        else:
            actual = getattr(surface, operation)(value)
            _assert_payload(actual, expected)
