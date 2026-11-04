"""Preserve the reference atan2 inner-loop layout before native owned copying."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def _layout(name):
    x = np.array([-1., -127., 1., -1., -0., 0., 2., -1.], dtype=np.float32)
    y = np.array([65534., 126., 1., np.finfo(np.float32).max, 1., 1., 3., -126.], dtype=np.float32)
    if name == 'reverse':
        return x[::-1], y[::-1]
    if name == 'positive-slice':
        return np.repeat(x, 2)[::2], np.repeat(y, 2)[::2]
    if name == 'negative-slice':
        return np.repeat(x, 2)[::-2], np.repeat(y, 2)[::-2]
    if name == 'readonly':
        x.setflags(write=False)
        return x, y
    if name == 'weak-int':
        return x[::-1], 65534
    if name == 'weak-float':
        return x[::-1], 65534.
    if name == 'unaligned-reverse':
        a = np.ndarray(x.shape, dtype=np.float32, buffer=bytearray(x.nbytes + 1), offset=1)
        b = np.ndarray(y.shape, dtype=np.float32, buffer=bytearray(y.nbytes + 1), offset=1)
        a[:], b[:] = x, y
        return a[::-1], b[::-1]
    if name == 'cast-both':
        return np.array([-1, -127, 2, 1], dtype=np.int8)[::-1], np.array([65534, 126, 3, 2], dtype=np.uint16)[::-1]
    if name == 'half-normalized':
        a = np.array([-1, -127, 2, 1], dtype=np.int8)[::-1]
        return a, a
    if name == 'C-outer-reverse':
        return x.reshape(2, 4)[::-1], y.reshape(2, 4)[::-1]
    if name == 'F-outer-reverse':
        return np.asfortranarray(x.reshape(2, 4))[:, ::-1], np.asfortranarray(y.reshape(2, 4))[:, ::-1]
    if name == 'C-inner-reverse':
        return x.reshape(2, 4)[:, ::-1], y.reshape(2, 4)[:, ::-1]
    if name == 'F-inner-reverse':
        return np.asfortranarray(x.reshape(2, 4))[::-1], np.asfortranarray(y.reshape(2, 4))[::-1]
    if name == 'broadcast':
        return np.broadcast_to(x, (2, 8)), np.broadcast_to(y, (2, 8))
    if name == 'singleton-negative':
        return x[:1][::-1], y[:1][::-1]
    if name == 'singleton-mixed-x':
        return x[:1][::-1], np.array([65534], dtype=np.uint16)
    if name == 'singleton-mixed-y':
        return np.array([-1], dtype=np.int8), y[:1][::-1]
    if name == 'singleton-huge-stride':
        return (np.lib.stride_tricks.as_strided(x[:1], shape=(1,), strides=(2**30,)),
                np.lib.stride_tricks.as_strided(y[:1], shape=(1,), strides=(2**30,)))
    raise AssertionError(name)


@pytest.mark.parametrize('name,expected', [
    ('reverse', 0), ('negative-slice', 0), ('weak-int', 0), ('weak-float', 0),
    ('positive-slice', 1), ('readonly', 1), ('unaligned-reverse', 1),
    ('cast-both', 1), ('half-normalized', 1), ('C-outer-reverse', 1),
    ('F-outer-reverse', 1), ('broadcast', 1), ('singleton-negative', 0),
    ('singleton-mixed-x', 0), ('singleton-mixed-y', 0),
    ('singleton-huge-stride', 0),
])
def test_direction_layout_selects_transport_profile_before_owned_copy(name, expected, monkeypatch):
    from behave_fire.rust import _bridge

    class Captured(Exception):
        pass

    captured = []

    def batch(operation, arrays, *args, **kwargs):
        captured.append(np.array(arrays['_svml_atan2'], copy=True))
        raise Captured()

    monkeypatch.setattr(_bridge, '_atan2_math_profile', lambda: 1)
    monkeypatch.setattr(_bridge._native, 'batch', batch)
    x, y = _layout(name)
    with pytest.raises(Captured):
        _bridge._call('surface.calculate_direction_of_max_spread', dict(
            x_component=x, y_component=y, rate_vector=1., wind_orientation_mode='RelativeToNorth', aspect=95.))
    assert len(captured) == 1
    assert np.all(captured[0] == expected)


@pytest.mark.parametrize('name', ['reverse', 'positive-slice', 'negative-slice', 'readonly', 'weak-int',
                                  'weak-float', 'unaligned-reverse', 'cast-both', 'half-normalized',
                                  'C-outer-reverse', 'F-outer-reverse', 'C-inner-reverse', 'F-inner-reverse',
                                  'broadcast', 'singleton-negative', 'singleton-mixed-x',
                                  'singleton-mixed-y', 'singleton-huge-stride'])
def test_direction_layouts_match_actual_python_kernel(name):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface as native

    x, y = _layout(name)
    dtypes = [float if isinstance(value, float) and not isinstance(value, np.generic)
              else int if isinstance(value, int) and not isinstance(value, (bool, np.generic))
              else np.asanyarray(value).dtype for value in (y, x)]
    if np.arctan2.resolve_dtypes((*dtypes, None))[2] == np.dtype('float16'):
        reference_x, reference_y = np.asanyarray(x).astype(np.float32), np.asanyarray(y).astype(np.float32)
    else:
        reference_x, reference_y = x, y
    with np.errstate(all='ignore'):
        expected = reference.calculate_direction_of_max_spread(reference_x, reference_y, 1., 'RelativeToNorth', 95.)
        actual = native.calculate_direction_of_max_spread(x, y, 1., 'RelativeToNorth', 95.)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
